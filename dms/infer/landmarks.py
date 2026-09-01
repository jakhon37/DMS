"""InsightFace 2d106det. Crop contract matches Landmark.get() 1.5x loose square."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from dms.runtime.trt_engine import TrtEngine


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

    def __init__(self, engine: TrtEngine) -> None:
        self.engine = engine
        self._in = engine._inputs[0].name
        self._out = engine._outputs[0].name

    def infer_one(self, bgr: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        """Return (106, 2) in full-frame pixels, or None if crop is empty."""
        M = loose_crop_affine(xyxy, self.input_size, 1.5)
        crop = cv2.warpAffine(
            bgr, M, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR, borderValue=0
        )
        if crop.size == 0:
            return None
        # 2d106det ONNX includes Sub/Mul; feed 0-255 BGR NCHW (InsightFace mxnet path).
        blob = np.transpose(crop.astype(np.float32), (2, 0, 1))[None, ...]
        blob = np.ascontiguousarray(blob)
        raw = self.engine.infer({self._in: blob})[self._out]
        pred = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
        if pred.shape[0] != 106:
            raise RuntimeError("2d106det expected 106 points, got %s" % (pred.shape,))
        # output is [-1, 1] on the 192 crop
        crop_xy = (pred + 1.0) * (self.input_size * 0.5)
        Minv = invert_affine(M)
        return apply_affine(crop_xy, Minv)
