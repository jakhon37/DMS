"""Model wrappers and the preprocessing primitives they share.

Everything in this module is pure numpy/OpenCV and imports nothing from
:mod:`dms.runtime`, so the whole preprocessing layer is unit-testable off-board
with no TensorRT and no GPU.

PROJECT RULE ENFORCED HERE: no function in this package may run a resize or a
warp over a full >=720p frame in order to produce a small crop.  Every crop is
taken by slicing the sub-rectangle out of the frame first (a numpy view, free),
padding only the part that falls outside the frame, and resizing that.  For a
40x24 eye crop that is roughly four thousand pixels of work instead of nine
hundred thousand.  The one deliberate exception is the face detector's own
letterbox, which genuinely has to see the whole frame.

Units: image arrays are uint8 BGR HxWx3 as OpenCV delivers them; boxes are
float ``xyxy`` in full-frame pixels; blobs are float32 NCHW.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dms.types import ModelHealth

log = logging.getLogger("dms.infer")

#: ImageNet statistics, in 0-1 units, RGB order.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ maths ---
def sigmoid(x):
    """Numerically stable logistic. Accepts scalars and arrays, returns float64."""
    a = np.asarray(x, dtype=np.float64)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(x, axis: int = -1):
    """Softmax along ``axis``, computed after subtracting the max."""
    a = np.asarray(x, dtype=np.float64)
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=axis, keepdims=True)


# ------------------------------------------------------------------ boxes ---
def clip_box(xyxy, frame_wh: Tuple[int, int]) -> np.ndarray:
    """Clamp an xyxy box to the frame. Returns a new float32 array."""
    w, h = int(frame_wh[0]), int(frame_wh[1])
    b = np.asarray(xyxy, dtype=np.float32).reshape(4).copy()
    b[0] = min(max(b[0], 0.0), w - 1.0)
    b[1] = min(max(b[1], 0.0), h - 1.0)
    b[2] = min(max(b[2], 0.0), w - 1.0)
    b[3] = min(max(b[3], 0.0), h - 1.0)
    return b


def expand_box(xyxy, frac: float) -> np.ndarray:
    """Grow an xyxy box by ``frac`` of its own width/height on every side."""
    b = np.asarray(xyxy, dtype=np.float32).reshape(4)
    w = float(b[2] - b[0])
    h = float(b[3] - b[1])
    return np.array(
        [b[0] - frac * w, b[1] - frac * h, b[2] + frac * w, b[3] + frac * h],
        dtype=np.float32,
    )


def square_box(xyxy) -> np.ndarray:
    """Smallest square, centred on the box, that contains it."""
    b = np.asarray(xyxy, dtype=np.float32).reshape(4)
    cx = 0.5 * float(b[0] + b[2])
    cy = 0.5 * float(b[1] + b[3])
    side = 0.5 * max(float(b[2] - b[0]), float(b[3] - b[1]))
    return np.array([cx - side, cy - side, cx + side, cy + side], dtype=np.float32)


def sixdrepnet_expand(xyxy) -> np.ndarray:
    """The head-pose crop expansion used by the 6DRepNet reference code.

    Upstream ``expand_bbox`` really does swap height and width::

        x_min -= 0.2 * height ; y_min -= 0.2 * width
        x_max += 0.2 * height ; y_max += 0.2 * width

    and then clamps the two minima at zero.  The quirk is preserved on purpose:
    the weights were trained on crops produced this way, and "fixing" it shifts
    the operating point of every angle the network predicts.  Clamping at zero is
    left to :func:`crop_patch`, which pads instead of shifting the crop.
    """
    b = np.asarray(xyxy, dtype=np.float32).reshape(4)
    w = float(b[2] - b[0])
    h = float(b[3] - b[1])
    return np.array(
        [b[0] - 0.2 * h, b[1] - 0.2 * w, b[2] + 0.2 * h, b[3] + 0.2 * w],
        dtype=np.float32,
    )


def box_inside_fraction(xyxy, frame_wh: Tuple[int, int]) -> float:
    """Fraction of the box's area that lies inside the frame, in [0, 1]."""
    w, h = int(frame_wh[0]), int(frame_wh[1])
    b = np.asarray(xyxy, dtype=np.float64).reshape(4)
    area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area <= 0.0:
        return 0.0
    ix = max(0.0, min(b[2], w) - max(b[0], 0.0))
    iy = max(0.0, min(b[3], h) - max(b[1], 0.0))
    return float(ix * iy / area)


# ------------------------------------------------------------------ crops ---
def crop_patch(
    bgr: np.ndarray,
    xyxy,
    out_w: int,
    out_h: int,
    *,
    interp: int = cv2.INTER_LINEAR,
    border_value: int = 0,
) -> Tuple[Optional[np.ndarray], float]:
    """Crop ``xyxy`` out of ``bgr`` and resize it to ``out_w`` x ``out_h``.

    Returns ``(patch, inside_fraction)``.  ``patch`` is None only when the box
    does not intersect the frame at all.  Parts of the box outside the frame are
    filled with ``border_value`` rather than being clipped away, so the crop
    keeps the geometry the model was trained on; ``inside_fraction`` tells the
    caller how much of what it is looking at is real image, and every wrapper
    refuses to publish a MODEL-sourced result below its own threshold.

    Only the intersecting sub-rectangle is ever touched, so this is O(crop), not
    O(frame).
    """
    h_img, w_img = bgr.shape[:2]
    b = np.asarray(xyxy, dtype=np.float64).reshape(4)
    x1, y1 = int(np.floor(b[0])), int(np.floor(b[1]))
    x2, y2 = int(np.ceil(b[2])), int(np.ceil(b[3]))
    # A 1x1 source is both information-free and unsupported: OpenCV 4.5.4 cannot
    # build a Mat from a 1x1x3 buffer at all ("_step >= minstep"). Widen instead
    # of letting a degenerate box raise out of a wrapper.
    if x2 - x1 < 2:
        x2 = x1 + 2
    if y2 - y1 < 2:
        y2 = y1 + 2
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(w_img, x2), min(h_img, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0
    sub = bgr[iy1:iy2, ix1:ix2]
    pad_l, pad_t = ix1 - x1, iy1 - y1
    pad_r, pad_b = x2 - ix2, y2 - iy2
    if pad_l or pad_t or pad_r or pad_b:
        sub = cv2.copyMakeBorder(
            sub, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(border_value,) * 3
        )
    inside = float((ix2 - ix1) * (iy2 - iy1)) / float((x2 - x1) * (y2 - y1))
    # `sub` stays a strided view of the frame - cv2 wraps those directly, so no
    # copy of the crop is made and nothing outside it is ever read.
    patch = cv2.resize(sub, (int(out_w), int(out_h)), interpolation=interp)
    return patch, inside


def to_blob(
    patch_bgr: np.ndarray,
    *,
    rgb: bool = True,
    scale: float = 1.0 / 255.0,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """HxWx3 uint8 BGR -> contiguous (1, 3, H, W) float32.

    ``scale`` is applied first, then ``mean`` is subtracted and ``std`` divided,
    both interpreted in the post-scale units and in the channel order that
    ``rgb`` selects.  Pass ``scale=1.0, mean=None, std=None`` for the models that
    want raw 0-255 (YOLOX, SFace, 2d106det).
    """
    img = patch_bgr[..., ::-1] if rgb else patch_bgr
    x = np.asarray(img, dtype=np.float32) * np.float32(scale)
    if mean is not None:
        x = x - np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    if std is not None:
        x = x / np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None, ...])


#: ArcFace 112x112 reference landmarks, in SCRFD keypoint order
#: (image-left eye, image-right eye, nose, image-left mouth, image-right mouth).
ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_5pt(
    bgr: np.ndarray,
    kps5: np.ndarray,
    out_size: int = 112,
    ref: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """5-point similarity-transform alignment, done on a local crop.

    Returns ``(aligned, inside_fraction)``.  Equivalent to InsightFace's
    ``norm_crop`` / OpenCV's ``FaceRecognizerSF.alignCrop``, except that the warp
    runs over a padded neighbourhood of the keypoints instead of the whole frame,
    which is the difference between ~0.2 ms and ~6 ms at 720p.
    """
    if ref is None:
        ref = ARCFACE_REF_112
    k = np.asarray(kps5, dtype=np.float32).reshape(-1, 2)
    if k.shape[0] != 5 or not np.isfinite(k).all():
        return None, 0.0
    ref_scaled = np.asarray(ref, dtype=np.float32) * (float(out_size) / 112.0)

    # Work on a neighbourhood twice the keypoint spread so the warp source is a
    # small array; then shift the transform by the crop origin.
    lo = k.min(axis=0)
    hi = k.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    pad = span
    box = np.array([lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad], dtype=np.float32)
    h_img, w_img = bgr.shape[:2]
    x1, y1 = int(np.floor(box[0])), int(np.floor(box[1]))
    x2, y2 = int(np.ceil(box[2])), int(np.ceil(box[3]))
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(w_img, x2), min(h_img, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0
    sub = bgr[iy1:iy2, ix1:ix2]
    pad_l, pad_t = ix1 - x1, iy1 - y1
    pad_r, pad_b = x2 - ix2, y2 - iy2
    if pad_l or pad_t or pad_r or pad_b:
        sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    local = k - np.array([x1, y1], dtype=np.float32)
    M, _inliers = cv2.estimateAffinePartial2D(
        local.reshape(-1, 1, 2), ref_scaled.reshape(-1, 1, 2), method=cv2.LMEDS
    )
    if M is None:
        return None, 0.0
    aligned = cv2.warpAffine(
        sub, M, (int(out_size), int(out_size)), flags=cv2.INTER_LINEAR, borderValue=0
    )
    inside = _aligned_inside_fraction(M, np.array([x1, y1], dtype=np.float32), out_size, (w_img, h_img))
    return aligned, inside


def _aligned_inside_fraction(M, crop_origin, out_size, frame_wh):
    """Fraction of the ALIGNED output that came from inside the frame.

    Maps the four corners of the ``out_size`` square back into full-frame
    coordinates and intersects that (rotated) quadrilateral with the frame.
    Measuring the output rather than the source neighbourhood is what makes the
    number mean "how much of what the model sees is real image"; the padded
    neighbourhood the warp reads from is deliberately larger than the face and
    routinely hangs off the frame edge without costing the crop anything.
    """
    M3 = np.vstack([np.asarray(M, dtype=np.float64), [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(M3)[:2]
    s = float(out_size)
    corners = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float64)
    hom = np.hstack([corners, np.ones((4, 1))])
    quad = (hom @ inv.T)[:, :2] + np.asarray(crop_origin, dtype=np.float64)
    w, h = float(frame_wh[0]), float(frame_wh[1])
    frame = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    area, _pts = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    total = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if total < 1e-6:
        return 0.0
    return float(min(1.0, max(0.0, area / total)))


# ------------------------------------------------------- engine introspection
def engine_io_names(engine) -> Tuple[List[str], List[str]]:
    """``(input_names, output_names)`` for a TrtEngine-like object.

    Tries the public attributes first (``input_name`` / ``output_names``) and
    falls back to the private binding lists.  Written defensively on purpose:
    the runtime workstream owns :mod:`dms.runtime.trt_engine` and has been
    reshaping its internals, and a wrapper must not break because a private
    attribute was renamed.
    """
    ins: List[str] = []
    outs: List[str] = []
    name = getattr(engine, "input_name", None)
    if isinstance(name, str):
        ins = [name]
    names = getattr(engine, "input_names", None)
    if not ins and names:
        ins = [str(n) for n in names]
    onames = getattr(engine, "output_names", None)
    if onames:
        outs = [str(n) for n in onames]
    if not ins or not outs:
        binds = getattr(engine, "_bindings", None)
        if isinstance(binds, dict):
            if not ins:
                ins = [b.name for b in binds.values() if getattr(b, "is_input", False)]
            if not outs:
                outs = [b.name for b in binds.values() if not getattr(b, "is_input", True)]
    if not ins:
        ins = [b.name for b in getattr(engine, "_inputs", [])]
    if not outs:
        outs = [b.name for b in getattr(engine, "_outputs", [])]
    if not ins or not outs:
        raise RuntimeError("cannot determine IO names for %r" % (engine,))
    return ins, outs


def engine_input_shape(engine) -> Tuple[int, ...]:
    """Static NCHW shape of the engine's first input."""
    shape = getattr(engine, "input_shape", None)
    if shape is not None:
        return tuple(int(s) for s in shape)
    binds = getattr(engine, "_inputs", None)
    if binds:
        return tuple(int(s) for s in binds[0].shape)
    raise RuntimeError("cannot determine input shape for %r" % (engine,))


# ------------------------------------------------------------- wrapper base -
class ModelWrapper(object):
    """Common shape for every capability wrapper.

    A wrapper is constructed with an engine (or ``None``, meaning the engine is
    absent and the capability must degrade), holds a :class:`dms.types.ModelHealth`
    record, and exposes exactly one ``infer(...)`` entry point that returns one of
    the dataclasses in :mod:`dms.types`.

    Contract that the main loop depends on: ``infer`` NEVER raises.  A model that
    throws - a dead engine, a CUDA hang, a shape surprise after a rebuild - is
    recorded as a failure on ``self.health`` and the wrapper falls through to its
    geometric fallback or to ``source=UNAVAILABLE``.  The registry turns a run of
    consecutive failures into a MODEL_DEGRADED signal; nothing turns it into a
    traceback out of the frame loop.
    """

    capability = "unknown"

    #: Consecutive engine failures after which the wrapper stops calling the
    #: engine at all.  Retrying a dead engine every frame costs more than the
    #: signal is worth, and each retry can carry a 500 ms wait() timeout.
    max_consecutive_failures = 15

    def __init__(self, engine=None, cfg=None, name=""):
        self.engine = engine
        self.cfg = dict(cfg or {})
        self.health = ModelHealth(name=name or self.capability, loaded=engine is not None)
        self._disabled = False
        self._runs = 0
        self._sum_ms = 0.0
        if engine is not None:
            ins, outs = engine_io_names(engine)
            self._in_names = ins
            self._out_names = outs
        else:
            self._in_names = []
            self._out_names = []

    @property
    def available(self):
        # type: () -> bool
        """True when the engine is loaded and has not been disabled by failures."""
        return self.engine is not None and not self._disabled

    def run_engine(self, feed):
        """Run the engine, time it, update health. Returns the output dict or None.

        Checks ``self.engine`` explicitly as well as ``available``: a subclass
        that owns more than one engine (the occlusion gate owns two) overrides
        ``available`` to mean "at least one is loaded", and this must not then
        dereference the missing one.
        """
        if self.engine is None or not self.available:
            return None
        t0 = time.monotonic()
        try:
            out = self.engine.infer(feed)
        except Exception as exc:  # noqa: BLE001 - deliberate: never escape to the loop
            self.note_failure(exc)
            return None
        ms = (time.monotonic() - t0) * 1000.0
        self._runs += 1
        self._sum_ms += ms
        self.health.consecutive_failures = 0
        self.health.last_ok_mono_s = time.monotonic()
        self.health.mean_ms = self._sum_ms / self._runs
        return out

    def note_failure(self, exc):
        """Record a failure. Disables the engine once the run is long enough."""
        self.health.consecutive_failures += 1
        if self.health.consecutive_failures >= self.max_consecutive_failures:
            self._disabled = True
            self.health.loaded = False
        log.warning(
            "%s inference failed (%d in a row): %s",
            self.capability,
            self.health.consecutive_failures,
            exc,
        )

    def close(self):
        """Release the engine if it owns one. Safe to call twice."""
        eng = self.engine
        self.engine = None
        self.health.loaded = False
        if eng is not None and hasattr(eng, "close"):
            try:
                eng.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s engine close failed: %s", self.capability, exc)


# Re-exported last so that submodules may do ``from dms.infer import to_blob``
# while this package is still initialising.
from dms.infer.landmarks import Landmark106  # noqa: E402
from dms.infer.scrfd import ScrfdDetector  # noqa: E402

__all__ = [
    "ARCFACE_REF_112",
    "ModelWrapper",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Landmark106",
    "ScrfdDetector",
    "align_5pt",
    "box_inside_fraction",
    "clip_box",
    "crop_patch",
    "engine_input_shape",
    "engine_io_names",
    "expand_box",
    "sigmoid",
    "sixdrepnet_expand",
    "softmax",
    "square_box",
    "to_blob",
]
