from __future__ import annotations

import numpy as np

from dms.geometry.ear import eye_aspect_ratio, face_ear
from dms.geometry.face106 import (
    LEFT_EYE,
    LEFT_EYE_CORNER,
    MOUTH_LEFT,
    NOSE_TIP,
    PNP_INDICES,
    RIGHT_EYE,
    RIGHT_EYE_CORNER,
)


def test_uniface_groups_count_106():
    covered = set(range(0, 33)) | set(range(33, 51)) | set(range(51, 63))
    covered |= set(LEFT_EYE) | set(RIGHT_EYE) | set(range(87, 106))
    assert NOSE_TIP in covered
    assert LEFT_EYE_CORNER in LEFT_EYE
    assert RIGHT_EYE_CORNER in RIGHT_EYE
    assert MOUTH_LEFT == 87
    assert len(PNP_INDICES) == 6


def test_ear_open_vs_closed():
    open_eye = np.array(
        [[0, 0], [1, -2], [2, -2], [3, 0], [2, 2], [1, 2]], dtype=np.float32
    )
    closed = np.array(
        [[0, 0], [1, -0.1], [2, -0.1], [3, 0], [2, 0.1], [1, 0.1]], dtype=np.float32
    )
    assert eye_aspect_ratio(open_eye) > 0.5
    assert eye_aspect_ratio(closed) < 0.2


def test_face_ear_uses_uniface_indices():
    lm = np.zeros((106, 2), dtype=np.float32)
    # LEFT_EYE 63:72, RIGHT_EYE 76:84 — wide open boxes
    lm[63:72] = [[0, 0], [1, -2], [2, -2], [3, 0], [2, 2], [1, 2], [0.5, 1], [2.5, 1], [1.5, 0]]
    lm[76:84] = [[0, 0], [1, -2], [2, -2], [3, 0], [2, 2], [1, 2], [0.5, 1], [2.5, 1]]
    assert face_ear(lm) > 0.5
