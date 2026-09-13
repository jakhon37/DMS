"""Which occupant is the driver, and is it still the same person.

Picking the driver by "largest face in the seat ROI" every frame is unstable in
exactly the cabin geometry this system exists for: a driver and a front
passenger of similar apparent size make the argmax oscillate, and every
oscillation stitches one person's eye closures, yawns and head pose onto the
other's alert windows.

:class:`DriverSelector` adds the three things that make the choice stick:

* **a seat-ROI centre bias** - a face near the middle of the driver ROI outranks
  one clinging to its edge at equal size;
* **track persistence** - a track the tracker has matched for many frames
  outranks one minted this frame at the same apparent size;
* **incumbency plus dwell** - the current driver's score is multiplied by
  ``incumbent_bonus``, and a challenger must lead for ``switch_dwell_s`` AND
  ``switch_min_frames`` before the handover happens.  A marginal, momentary lead
  changes nothing;
* **identity, when it is available** - an SFace embedding decides ties, and it
  also re-attaches the driver to a NEW track id after the tracker broke, so an
  occlusion does not read as a driver change.

The selector also owns ``t_last_match_s``: the last instant a *matched*
detection sat inside the seat ROI.  That, not the survival of a coasting track
id, is what ``FACE_LOST`` is defined on.

Units: ROIs are normalised ``[x0, y0, x1, y1]`` fractions of the frame; boxes
are full-frame pixels; ``t_s`` is seconds on the frame clock; embeddings are
any-length float vectors, L2-normalised internally.  Failure behaviour: every
entry point tolerates empty input and returns "no driver" rather than raising.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dms.infer.scrfd import FaceDet
from dms.track.iou_tracker import Track

__all__ = [
    "DriverPick",
    "DriverSelector",
    "SelectorConfig",
    "label_occupants",
    "pick_driver",
]


def _area(f: FaceDet) -> float:
    return max(0.0, float(f.xyxy[2] - f.xyxy[0])) * max(0.0, float(f.xyxy[3] - f.xyxy[1]))


def _centre(f: FaceDet) -> Tuple[float, float]:
    return 0.5 * (float(f.xyxy[0]) + float(f.xyxy[2])), 0.5 * (float(f.xyxy[1]) + float(f.xyxy[3]))


def _roi_px(roi_xyxy_norm: Sequence[float], frame_wh) -> Tuple[float, float, float, float]:
    w, h = float(frame_wh[0]), float(frame_wh[1])
    x1, y1, x2, y2 = [float(v) for v in roi_xyxy_norm]
    return x1 * w, y1 * h, x2 * w, y2 * h


def _in_roi(f: FaceDet, roi_xyxy_norm: Sequence[float], frame_wh) -> bool:
    rx1, ry1, rx2, ry2 = _roi_px(roi_xyxy_norm, frame_wh)
    cx, cy = _centre(f)
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _base_score(f: FaceDet) -> float:
    """Size-dominated score.  The driver is the nearest face to a cabin camera."""
    return _area(f) * (0.3 + float(f.score))


def _centre_weight(f: FaceDet, roi_xyxy_norm: Optional[Sequence[float]], frame_wh) -> float:
    """1.0 at the ROI centre, falling to ~0.67 at its corner.  1.0 with no ROI."""
    if not roi_xyxy_norm or len(roi_xyxy_norm) != 4:
        return 1.0
    rx1, ry1, rx2, ry2 = _roi_px(roi_xyxy_norm, frame_wh)
    half_w = max(1e-6, 0.5 * (rx2 - rx1))
    half_h = max(1e-6, 0.5 * (ry2 - ry1))
    cx, cy = _centre(f)
    dx = (cx - 0.5 * (rx1 + rx2)) / half_w
    dy = (cy - 0.5 * (ry1 + ry2)) / half_h
    dist = min(1.5, math.sqrt(dx * dx + dy * dy))
    return 1.0 / (1.0 + 0.5 * dist)


def pick_driver(
    faces: Sequence[FaceDet],
    roi_xyxy_norm: Optional[List[float]],
    frame_wh,
    fallback: bool = True,
) -> Optional[FaceDet]:
    """Stateless driver pick: largest ``area * score`` face inside the seat ROI.

    Kept as-is for callers that have no tracker.  It has NO temporal stickiness,
    so anything that owns alert state should use :class:`DriverSelector`.

    If no face is in the ROI and ``fallback`` is False, returns ``None``
    (``FACE_LOST``) rather than promoting a rear passenger to driver.
    """
    if not faces:
        return None
    pool = list(faces)
    if roi_xyxy_norm and len(roi_xyxy_norm) == 4:
        in_roi = [f for f in faces if _in_roi(f, roi_xyxy_norm, frame_wh)]
        if in_roi:
            pool = in_roi
        elif not fallback:
            return None
    return max(pool, key=_base_score)


def _l2(vec) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return None
    return arr / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class DriverPick:
    """Result of one selection round."""

    track: Optional[Track] = None
    face: Optional[FaceDet] = None
    track_id: Optional[int] = None
    #: True on the frame the driver becomes a DIFFERENT PERSON.  Re-acquiring
    #: the same person under a new track id (proved by identity) is not a
    #: change, and neither is the first driver of the session.
    changed: bool = False
    #: cosine similarity to the stored driver embedding, when identity ran.
    identity_score: Optional[float] = None
    score: float = 0.0
    #: candidate track ids considered this frame, best first.
    candidates: List[int] = field(default_factory=list)
    reason: str = "none"


@dataclass
class SelectorConfig:
    """Tunables for :class:`DriverSelector`.  Times in seconds."""

    #: multiplier applied to the incumbent's score, so a marginal size
    #: difference cannot flip the selection frame to frame.
    incumbent_bonus: float = 1.3
    #: a challenger must lead for this long AND this many consecutive frames.
    switch_dwell_s: float = 0.6
    switch_min_frames: int = 3
    #: how long the incumbent id stays reserved while it is not matched, so a
    #: brief occlusion does not hand DRV to the passenger.
    reacquire_s: float = 2.0
    #: SFace cosine at or above which two embeddings are the same identity
    #: (OpenCV's published operating point for sface).
    identity_same_thr: float = 0.363
    #: weight of the identity term when scoring a candidate.
    identity_bonus: float = 0.6
    #: EWMA factor folding a fresh embedding into the stored driver template.
    identity_tau: float = 0.2
    #: weight of track persistence: a track that has been matched many times
    #: outranks one the detector minted this frame at the same apparent size.
    persistence_bonus: float = 0.25
    #: hit count at which the persistence bonus saturates.
    persistence_frames: int = 15
    #: fall back to the best face outside the ROI when the ROI is empty.
    fallback: bool = True


class DriverSelector(object):
    """Stateful, hysteretic driver-vs-passenger selection.

    One instance per camera.  Call :meth:`select` once per frame with the live
    tracks; it returns a :class:`DriverPick` and updates
    :attr:`t_last_match_s`.

    ``embeddings`` maps ``track_id -> identity vector`` for whichever tracks the
    identity model ran on this frame.  It is entirely optional: with no
    embeddings the selector degrades to ROI + size + incumbency, which is still
    stable, and it never claims an identity it did not measure.
    """

    def __init__(
        self,
        roi_xyxy_norm: Optional[Sequence[float]] = None,
        cfg: Optional[SelectorConfig] = None,
    ) -> None:
        self.roi = list(roi_xyxy_norm) if roi_xyxy_norm else None
        self.cfg = cfg or SelectorConfig()
        self.driver_track_id: Optional[int] = None
        self.driver_embedding: Optional[np.ndarray] = None
        #: last instant a MATCHED detection sat in the seat ROI; FACE_LOST is
        #: defined on this, independent of any coasting track id.
        self.t_last_match_s: Optional[float] = None
        self._challenger_id: Optional[int] = None
        self._challenger_hold_s = 0.0
        self._challenger_frames = 0
        self._t_prev: Optional[float] = None
        self._t_driver_seen: Optional[float] = None

    # ------------------------------------------------------------- helpers --
    def reset(self) -> None:
        self.driver_track_id = None
        self.driver_embedding = None
        self._challenger_id = None
        self._challenger_hold_s = 0.0
        self._challenger_frames = 0
        self._t_driver_seen = None

    def _candidates(self, tracks: Sequence[Track], frame_wh) -> Tuple[List[Track], str]:
        matched = [t for t in tracks if t.face is not None and t.lost == 0]
        if not matched:
            return [], "no_match"
        if self.roi and len(self.roi) == 4:
            in_roi = [t for t in matched if _in_roi(t.face, self.roi, frame_wh)]
            if in_roi:
                return in_roi, "roi"
            if not self.cfg.fallback:
                return [], "roi_empty"
            return matched, "fallback"
        return matched, "no_roi"

    # -------------------------------------------------------------- select --
    def select(
        self,
        tracks: Sequence[Track],
        frame_wh,
        t_s: Optional[float] = None,
        embeddings: Optional[Dict[int, Any]] = None,
    ) -> DriverPick:
        cfg = self.cfg
        now = None if t_s is None else float(t_s)
        dt = 0.0
        if now is not None and self._t_prev is not None:
            dt = max(0.0, min(1.0, now - self._t_prev))
        if now is not None:
            self._t_prev = now

        pool, pool_reason = self._candidates(tracks, frame_wh)
        reason = pool_reason
        if not pool:
            # Nobody eligible.  Keep the incumbent id reserved for a while so a
            # brief occlusion does not renumber the driver, but report no face:
            # presence is what FACE_LOST is built on and it is False here.
            if (
                now is not None
                and self._t_driver_seen is not None
                and (now - self._t_driver_seen) > cfg.reacquire_s
            ):
                self.driver_track_id = None
            self._challenger_id = None
            self._challenger_hold_s = 0.0
            self._challenger_frames = 0
            return DriverPick(reason=reason)

        emb = {}
        if embeddings:
            for tid, vec in embeddings.items():
                norm = _l2(vec)
                if norm is not None:
                    emb[int(tid)] = norm

        scored: List[Tuple[float, Track, Optional[float]]] = []
        for tr in pool:
            base = _base_score(tr.face) * _centre_weight(tr.face, self.roi, frame_wh)
            hits = max(0, int(getattr(tr, "hits", 1)))
            base *= 1.0 + cfg.persistence_bonus * min(
                1.0, hits / float(max(1, cfg.persistence_frames))
            )
            sim = _cosine(emb.get(tr.track_id), self.driver_embedding)
            if sim is not None:
                # Identity moves the score by at most +/- identity_bonus of its
                # own magnitude - a tiebreak, never an override of geometry.
                base *= 1.0 + cfg.identity_bonus * sim
            if tr.track_id == self.driver_track_id:
                base *= cfg.incumbent_bonus
            scored.append((base, tr, sim))
        scored.sort(key=lambda s: (-s[0], s[1].track_id))
        best_score, best, best_sim = scored[0]
        candidates = [t.track_id for _, t, _ in scored]
        by_id = dict((t.track_id, sc) for sc, t, _ in scored)

        incumbent = None
        for _, tr, sim in scored:
            if tr.track_id == self.driver_track_id:
                incumbent = (tr, sim)
                break

        chosen, chosen_sim = best, best_sim
        if incumbent is not None and best.track_id != incumbent[0].track_id:
            # A challenger leads.  Make it earn the handover.
            if self._challenger_id == best.track_id:
                self._challenger_hold_s += dt
                self._challenger_frames += 1
            else:
                self._challenger_id = best.track_id
                self._challenger_hold_s = dt
                self._challenger_frames = 1
            enough_time = now is None or self._challenger_hold_s >= cfg.switch_dwell_s
            enough_frames = self._challenger_frames >= cfg.switch_min_frames
            if not (enough_time and enough_frames):
                chosen, chosen_sim = incumbent[0], incumbent[1]
                reason = "incumbent_held"
        else:
            self._challenger_id = None
            self._challenger_hold_s = 0.0
            self._challenger_frames = 0

        previous = self.driver_track_id
        changed = False
        if previous is not None and chosen.track_id != previous:
            # Same person under a new id?  Only identity can answer that, and
            # only when it actually measured this track.
            same = chosen_sim is not None and chosen_sim >= cfg.identity_same_thr
            changed = not same
            reason = "reacquired" if same else "handover"
            self._challenger_id = None
            self._challenger_hold_s = 0.0
            self._challenger_frames = 0

        self.driver_track_id = chosen.track_id
        if now is not None:
            self._t_driver_seen = now
            # Only a match INSIDE the seat ROI refreshes t_last_match: a
            # fallback pick is a passenger being shown, not a driver being seen.
            if pool_reason in ("roi", "no_roi"):
                self.t_last_match_s = now

        vec = emb.get(chosen.track_id)
        if vec is not None:
            if self.driver_embedding is None or changed:
                self.driver_embedding = vec
            else:
                alpha = cfg.identity_tau
                blended = _l2((1.0 - alpha) * self.driver_embedding + alpha * vec)
                self.driver_embedding = vec if blended is None else blended

        return DriverPick(
            track=chosen,
            face=chosen.face,
            track_id=chosen.track_id,
            changed=changed,
            identity_score=chosen_sim,
            score=by_id.get(chosen.track_id, best_score),
            candidates=candidates,
            reason=reason,
        )


def label_occupants(
    live: Sequence[Track], driver_face_or_id: Optional[Any]
) -> List[Tuple[Track, str]]:
    """Assign ``DRV`` / ``P1`` / ``P2`` ... for the overlay.  Alerts stay on DRV.

    Accepts either the driver's :class:`FaceDet` (legacy call from ``dms/app.py``)
    or its integer track id, because the selector now returns an id.
    """
    driver_tid: Optional[int] = None
    if isinstance(driver_face_or_id, int):
        driver_tid = driver_face_or_id
    elif driver_face_or_id is not None:
        for t in live:
            if t.face is driver_face_or_id:
                driver_tid = t.track_id
                break
    out: List[Tuple[Track, str]] = []
    p = 1
    for t in sorted(live, key=lambda x: x.track_id):
        if driver_tid is not None and t.track_id == driver_tid:
            out.append((t, "DRV"))
        else:
            out.append((t, "P%d" % p))
            p += 1
    return out
