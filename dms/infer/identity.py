"""Driver identity embedding with SFace, and driver-change detection.

Engine contract (MANIFEST role ``driver_identity``)::

    IN   "data" FLOAT [1, 3, 112, 112]     OUT  "fc1" FLOAT [1, 128]

Preprocessing: **RGB** (OpenCV's own reference call is
``blobFromImage(..., swapRB=true)``), **RAW 0-255** - no ``/255``, no mean, no
std.  The crop is a 5-point similarity-transform alignment onto the ArcFace
reference landmarks, done by :func:`dms.infer.align_5pt` on a local
neighbourhood rather than over the whole frame.

Output: ``fc1`` is NOT L2-normalised by the graph (a probe measured a norm of
4.52), so it must be normalised before any comparison.  OpenCV's published
thresholds for this model are ``cosine >= 0.363`` for "same identity", which is
what :data:`DEFAULT_COSINE_THRESHOLD` uses.

What this is for.  Not access control - it is 128 dimensions of "is this the
same person the drowsiness windows were accumulated on".  PERCLOS, the blink
history and the pose EWMA are only meaningful for one driver; stitching two
people's windows together produces a fatigue score that belongs to nobody.  The
tracker's IoU id is not enough on its own, because an id survives a driver swap
at a rest stop if the two people overlap for one frame and it does not survive
an occlusion.  :meth:`IdentityEmbedder.observe` gives the state machine a
``changed`` edge it can hang ``AlertType.DRIVER_CHANGED`` on.

Cost.  19.7 MB of engine and 2.2 ms per call - the largest artifact in the DMS
set, and not a per-frame model.  It runs once per newly-seen track id and then
at most every ``refresh_n`` calls for confirmation.  Load it at startup with the
rest, not lazily: on a board with ~2.5 GB free the deserialisation spike is
better paid at boot than under load.

Privacy.  Embeddings are held in memory only, keyed by an integer slot, and the
gallery is bounded by ``max_identities`` with least-recently-seen eviction.
Nothing here writes an embedding to disk; ``privacy.record_faces`` in the config
governs imagery and this module never touches storage.

Failure behaviour: :meth:`embed` returns None and :meth:`observe` returns an
:class:`IdentityObservation` with ``label=None`` and ``changed=False`` when the
engine is absent or fails.  "Unknown" is never reported as "changed" - an absent
identity model must not manufacture DRIVER_CHANGED edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dms.infer import ModelWrapper, align_5pt, to_blob

#: OpenCV's published SFace decision threshold on cosine similarity.
DEFAULT_COSINE_THRESHOLD = 0.363
#: Minimum fraction of the alignment neighbourhood that must be real image.
MIN_CROP_INSIDE = 0.60
#: How many distinct people to remember.
DEFAULT_MAX_IDENTITIES = 8
#: Re-embed a known track this often (in calls) to catch a swap behind one id.
DEFAULT_REFRESH_N = 150


@dataclass
class IdentityObservation:
    """What the identity model concluded about one face on one frame."""

    label: Optional[str] = None
    similarity: float = 0.0
    changed: bool = False
    #: True when the embedding was actually recomputed on this call.
    refreshed: bool = False


def l2_normalise(v: np.ndarray) -> np.ndarray:
    """Unit-norm a vector. Returns zeros unchanged rather than dividing by zero."""
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return x
    return x / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors that are already unit-norm."""
    return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


class IdentityEmbedder(ModelWrapper):
    """SFace embeddings plus a bounded in-memory gallery."""

    capability = "identity"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        name: str = "identity",
    ) -> None:
        super().__init__(engine=engine, cfg=cfg, name=name)
        self.threshold = float(self.cfg.get("cosine_threshold", DEFAULT_COSINE_THRESHOLD))
        self.max_identities = int(self.cfg.get("max_identities", DEFAULT_MAX_IDENTITIES))
        self.refresh_n = max(1, int(self.cfg.get("refresh_n", DEFAULT_REFRESH_N)))
        self.size = 112
        if engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(engine)
            if len(shape) == 4:
                self.size = int(shape[2])
        #: label -> unit embedding
        self._gallery: Dict[str, np.ndarray] = {}
        #: label -> monotonically increasing last-seen counter, for eviction
        self._last_seen: Dict[str, int] = {}
        #: track id -> label currently assigned to it
        self._by_track: Dict[int, str] = {}
        self._calls: Dict[int, int] = {}
        self._tick = 0
        self._next_label = 1

    def embed(self, bgr: np.ndarray, kps5: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Unit-norm 128-D embedding for one aligned face, or None."""
        if bgr is None or kps5 is None or not self.available:
            return None
        aligned, inside = align_5pt(bgr, kps5, self.size)
        if aligned is None or inside < MIN_CROP_INSIDE:
            return None
        # SFace: RGB, raw 0-255.
        blob = to_blob(aligned, rgb=True, scale=1.0, mean=None, std=None)
        out = self.run_engine({self._in_names[0]: blob})
        if out is None:
            return None
        vec = np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(-1)
        if vec.size < 2 or not np.isfinite(vec).all():
            self.note_failure(ValueError("bad embedding shape %s" % (vec.shape,)))
            return None
        return l2_normalise(vec)

    def match(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        """Best gallery match for an embedding: ``(label, similarity)``."""
        best_label: Optional[str] = None
        best_sim = -1.0
        for label, known in self._gallery.items():
            sim = cosine(emb, known)
            if sim > best_sim:
                best_sim, best_label = sim, label
        if best_label is None or best_sim < self.threshold:
            return None, float(max(0.0, best_sim))
        return best_label, float(best_sim)

    def _enrol(self, emb: np.ndarray) -> str:
        if len(self._gallery) >= self.max_identities:
            oldest = min(self._last_seen, key=lambda k: self._last_seen[k])
            self._gallery.pop(oldest, None)
            self._last_seen.pop(oldest, None)
            for tid, lab in list(self._by_track.items()):
                if lab == oldest:
                    self._by_track.pop(tid, None)
        label = "driver-%d" % self._next_label
        self._next_label += 1
        self._gallery[label] = emb
        self._last_seen[label] = self._tick
        return label

    def observe(
        self,
        bgr: np.ndarray,
        kps5: Optional[np.ndarray],
        track_id: Optional[int],
    ) -> IdentityObservation:
        """Identify the driver on this frame and report whether they changed.

        Runs the engine only on a track id's first sighting and every
        ``refresh_n`` calls thereafter; in between it returns the cached label
        with ``refreshed=False``.
        """
        if track_id is None:
            return IdentityObservation()
        tid = int(track_id)
        self._tick += 1
        n = self._calls.get(tid, 0)
        self._calls[tid] = n + 1
        known = self._by_track.get(tid)
        if known is not None and (n % self.refresh_n) != 0:
            self._last_seen[known] = self._tick
            return IdentityObservation(label=known, similarity=1.0, changed=False)
        if not self.available:
            return IdentityObservation(label=known)

        emb = self.embed(bgr, kps5)
        if emb is None:
            return IdentityObservation(label=known)
        label, sim = self.match(emb)
        if label is None:
            label = self._enrol(emb)
            sim = 1.0
        else:
            # Running mean keeps the template current under changing light
            # without letting one bad frame move it far.
            self._gallery[label] = l2_normalise(0.9 * self._gallery[label] + 0.1 * emb)
        self._last_seen[label] = self._tick
        changed = known is not None and label != known
        self._by_track[tid] = label
        return IdentityObservation(label=label, similarity=sim, changed=changed, refreshed=True)

    def forget(self, track_id: int) -> None:
        """Drop per-track bookkeeping for a track that has gone away."""
        self._by_track.pop(int(track_id), None)
        self._calls.pop(int(track_id), None)

    def known_labels(self) -> List[str]:
        return sorted(self._gallery)
