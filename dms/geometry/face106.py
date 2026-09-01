"""UniFace Landmark106 topology for InsightFace 2d106det.

Only this module may define 106-pt indices. EAR/MAR/PnP must import names from here.
Cookbook: https://yakhyo.github.io/uniface/modules/landmarks/
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Groups (inclusive ranges in UniFace docs).
CONTOUR = range(0, 33)
BROW_L = range(33, 42)
BROW_R = range(42, 51)
NOSE = range(51, 63)
LEFT_EYE = range(63, 72)  # 63:72
RIGHT_EYE = range(76, 84)  # 76:84
MOUTH = range(87, 106)

CHIN = 16
NOSE_TIP = 51
LEFT_EYE_CORNER = 63
RIGHT_EYE_CORNER = 76
MOUTH_LEFT = 87
MOUTH_RIGHT = 93
MOUTH_UP = 89
MOUTH_LOW = 95

# Soukupová & Čech p1..p6 on the documented eye slices (starting map).
LEFT_EYE_EAR: Tuple[int, int, int, int, int, int] = (63, 64, 66, 67, 68, 70)
RIGHT_EYE_EAR: Tuple[int, int, int, int, int, int] = (76, 77, 79, 80, 81, 83)

PNP_INDICES: Tuple[int, int, int, int, int, int] = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_CORNER,
    RIGHT_EYE_CORNER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)

# mm, OpenCV camera frame, origin at nose tip.
PNP_OBJECT_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 32.0, -28.0],
        [-34.0, -30.0, -30.0],
        [34.0, -30.0, -30.0],
        [-20.0, 20.0, -28.0],
        [20.0, 20.0, -28.0],
    ],
    dtype=np.float64,
)


def points(landmarks: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
    return landmarks[list(idxs)]
