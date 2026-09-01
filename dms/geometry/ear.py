from __future__ import annotations

import numpy as np

from dms.geometry.face106 import LEFT_EYE_EAR, RIGHT_EYE_EAR


def eye_aspect_ratio(eye6: np.ndarray) -> float:
    """Soukupová & Čech EAR on 6 points (p1..p6)."""
    if eye6.shape[0] < 6:
        raise ValueError("need 6 eye points")
    p1, p2, p3, p4, p5, p6 = eye6[:6]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def face_ear(landmarks106: np.ndarray) -> float:
    left = eye_aspect_ratio(landmarks106[list(LEFT_EYE_EAR)])
    right = eye_aspect_ratio(landmarks106[list(RIGHT_EYE_EAR)])
    return 0.5 * (left + right)
