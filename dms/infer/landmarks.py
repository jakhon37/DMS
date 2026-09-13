"""InsightFace 2d106det. Crop contract matches Landmark.get() 1.5x loose square.

Engine contract (MANIFEST role ``landmarks_106``)::

    IN   "data" FLOAT [1, 3, 192, 192]      OUT  "fc1" FLOAT [1, 212]

Preprocessing: **RGB**, raw 0-255 (the Sub/Mul that would divide by 255 is baked
into the graph - ``strings 2d106det.onnx`` shows ``_minusscalar0``/``_mulscalar0``
- so the wrapper must NOT scale), NCHW.  Upstream builds this blob with
``cv2.dnn.blobFromImage(..., swapRB=True)`` and swapRB is unconditional.  The
previous revision of this file fed BGR; that does not fail loudly, it just
degrades the regression, and it is the most likely single cause of the
impossible open-eye ratios and the +/-80 deg pitch noise seen on replay.

Output: 106 (x, y) pairs in [-1, 1] on the 192 crop, mapped back to full-frame
pixels through the inverse of the crop affine.

Failure behaviour: :meth:`Landmark106.infer_one` returns None - never a
half-valid point set - when the crop is mostly outside the frame, when the crop
is degenerate, or when the engine call raises.  :meth:`infer_detail` returns the
same information plus the crop coverage so the caller can mark the landmark
signal unreliable instead of discarding the frame.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from dms.infer import box_inside_fraction, to_blob
from dms.runtime.trt_engine import TrtEngine

#: Minimum fraction of the 1.5x crop square that must lie inside the frame.
#: Below this the regressor is looking mostly at zero padding and the points it
#: returns are an extrapolation, not a measurement.
MIN_CROP_INSIDE = 0.55

#: Minimum detector box side, in pixels, worth running the regressor on.  At
#: 720p a 60 px face is about 25 px of inter-ocular distance, which is the point
#: below which the eye lid points stop separating.
MIN_FACE_PX = 48.0


def loose_crop_affine(xyxy: np.ndarray, out_size: int = 192, scale: float = 1.5) -> np.ndarray:
    """2x3 affine: original px -> 192 crop. Square of side scale*max(w,h) at box center."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = scale * max(w, h)
    src = np.array(
        [
            [cx - side * 0.5, cy - side * 0.5],
            [cx + side * 0.5, cy - side * 0.5],
            [cx - side * 0.5, cy + side * 0.5],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0.0, 0.0], [float(out_size), 0.0], [0.0, float(out_size)]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def crop_square_xyxy(xyxy: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """The xyxy of the square that :func:`loose_crop_affine` maps to the crop."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half = 0.5 * scale * max(w, h)
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def invert_affine(M: np.ndarray) -> np.ndarray:
    M3 = np.vstack([M, np.array([0.0, 0.0, 1.0], dtype=np.float64)])
    return np.linalg.inv(M3)[:2]


def apply_affine(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.hstack([pts.astype(np.float32), ones])
    out = hom @ M.T
    return out[:, :2]


class Landmark106:
    input_size = 192

    def __init__(self, engine: TrtEngine, *, min_face_px: float = MIN_FACE_PX) -> None:
        from dms.infer import engine_input_shape, engine_io_names

        self.engine = engine
        ins, outs = engine_io_names(engine)
        self._in = ins[0]
        self._out = outs[0]
        shape = engine_input_shape(engine)
        if len(shape) == 4:
            self.input_size = int(shape[2])
        self.min_face_px = float(min_face_px)

    def infer_detail(
        self, bgr: np.ndarray, xyxy: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """``(landmarks_106x2_in_full_frame_px, crop_inside_fraction)``.

        ``landmarks`` is None when the crop is unusable; the fraction is still
        reported so the caller can tell "face at the frame edge" from "engine
        refused".
        """
        box = np.asarray(xyxy, dtype=np.float32).reshape(4)
        side = max(float(box[2] - box[0]), float(box[3] - box[1]))
        if side < self.min_face_px:
            return None, 1.0
        square = crop_square_xyxy(box, 1.5)
        h_img, w_img = bgr.shape[:2]
        inside = box_inside_fraction(square, (w_img, h_img))
        if inside < MIN_CROP_INSIDE:
            return None, inside

        M = loose_crop_affine(box, self.input_size, 1.5)
        # warpAffine with a 192x192 dsize costs O(dst): it samples 36 864 output
        # pixels out of the frame and never touches the rest, so this is not the
        # full-frame resize the project rule forbids. Keeping the warp (rather
        # than a slice-and-resize) preserves the exact sub-pixel crop geometry
        # 2d106det was trained with, and invert_affine below maps the points back.
        crop = cv2.warpAffine(
            bgr, M, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR, borderValue=0
        )
        # 2d106det: RGB, raw 0-255 (the /255 lives inside the graph).
        blob = to_blob(crop, rgb=True, scale=1.0, mean=None, std=None)
        raw = self.engine.infer({self._in: blob})[self._out]
        pred = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        if pred.shape[0] != 106:
            raise RuntimeError("2d106det expected 106 points, got %s" % (pred.shape,))
        if not np.isfinite(pred).all():
            return None, inside
        # output is [-1, 1] on the 192 crop
        crop_xy = (pred + 1.0) * (self.input_size * 0.5)
        Minv = invert_affine(M)
        return apply_affine(crop_xy, Minv), inside

    def infer_one(self, bgr: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        """Return (106, 2) in full-frame pixels, or None if the crop is unusable."""
        return self.infer_detail(bgr, xyxy)[0]
