from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from dms.geometry.face106 import PNP_INDICES, PNP_OBJECT_POINTS


def camera_matrix(width: int, height: int) -> np.ndarray:
    f = float(width)
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from OpenCV rotation matrix (camera frame)."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    rad2deg = 180.0 / np.pi
    return float(yaw * rad2deg), float(pitch * rad2deg), float(roll * rad2deg)


def solve_head_pose(
    landmarks106: np.ndarray,
    frame_wh: Tuple[int, int],
) -> Optional[Tuple[float, float, float]]:
    image_pts = landmarks106[list(PNP_INDICES)].astype(np.float64)
    if not np.isfinite(image_pts).all():
        return None
    w, h = frame_wh
    cam = camera_matrix(w, h)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        PNP_OBJECT_POINTS, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((R, tvec.reshape(3, 1)))
    _a, _b, _c, _d, _e, _f, euler = cv2.decomposeProjectionMatrix(proj)
    pitch, yaw, roll = [float(v) for v in np.asarray(euler).reshape(-1)[:3]]
    yaw, pitch, roll = _wrap180(yaw), _wrap180(pitch), _wrap180(roll)
    # decomposeProjectionMatrix often returns pitch near ±180 for slight nod.
    if abs(pitch) > 90.0:
        pitch = pitch - (180.0 if pitch > 0 else -180.0)
    return yaw, pitch, roll


def _wrap180(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg
