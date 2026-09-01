from __future__ import annotations

import numpy as np

from dms.geometry.face106 import MOUTH_LEFT, MOUTH_LOW, MOUTH_RIGHT, MOUTH_UP


def mouth_aspect_ratio(landmarks106: np.ndarray) -> float:
    left = landmarks106[MOUTH_LEFT]
    right = landmarks106[MOUTH_RIGHT]
    up = landmarks106[MOUTH_UP]
    low = landmarks106[MOUTH_LOW]
    width = np.linalg.norm(right - left)
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(up - low) / width)
