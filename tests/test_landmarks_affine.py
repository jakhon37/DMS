from __future__ import annotations

import numpy as np

from dms.geometry.mar import mouth_aspect_ratio
from dms.infer.landmarks import apply_affine, invert_affine, loose_crop_affine


def test_affine_roundtrip():
    xyxy = np.array([100.0, 80.0, 220.0, 220.0], dtype=np.float32)
    M = loose_crop_affine(xyxy, 192, 1.5)
    Minv = invert_affine(M)
    # box center should map near (96, 96)
    cx, cy = 160.0, 150.0
    crop = apply_affine(np.array([[cx, cy]], dtype=np.float32), M)
    assert abs(crop[0, 0] - 96) < 2
    assert abs(crop[0, 1] - 96) < 2
    back = apply_affine(crop, Minv)
    assert np.allclose(back, [[cx, cy]], atol=1e-2)


def test_mar_open_mouth():
    lm = np.zeros((106, 2), dtype=np.float32)
    lm[87] = [0, 0]
    lm[93] = [10, 0]
    lm[89] = [5, -4]
    lm[95] = [5, 4]
    assert mouth_aspect_ratio(lm) > 0.7
