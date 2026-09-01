from __future__ import annotations

import numpy as np

from dms.geometry.face106 import LEFT_EYE, RIGHT_EYE


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


def contour_ear(eye_pts: np.ndarray) -> float:
    """Height/width of the eye point cloud. Stable when 6-pt order is unverified."""
    w = float(eye_pts[:, 0].max() - eye_pts[:, 0].min())
    h = float(eye_pts[:, 1].max() - eye_pts[:, 1].min())
    if w < 1e-6:
        return 0.0
    return h / w


def face_ear(landmarks106: np.ndarray) -> float:
    # UniFace 9/8-pt eye slices are not Soukupová-ordered; use contour EAR for v1.
    left = contour_ear(landmarks106[list(LEFT_EYE)])
    right = contour_ear(landmarks106[list(RIGHT_EYE)])
    return 0.5 * (left + right)
