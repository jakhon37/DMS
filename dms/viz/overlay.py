from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from dms.geometry.face106 import LEFT_EYE, MOUTH, NOSE_TIP, RIGHT_EYE
from dms.infer.scrfd import FaceDet


def draw_faces(
    bgr: np.ndarray,
    faces: List[FaceDet],
    *,
    driver: Optional[FaceDet] = None,
    landmarks106: Optional[np.ndarray] = None,
    ear: Optional[float] = None,
    mar: Optional[float] = None,
    pose: Optional[Tuple[float, float, float]] = None,
    track_id: Optional[int] = None,
) -> np.ndarray:
    vis = bgr.copy()
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.xyxy]
        color = (0, 255, 255) if driver is not None and f is driver else (0, 255, 0)
        thick = 3 if f is driver else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(
            vis,
            "%.2f" % f.score,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        if f.kps is not None:
            for x, y in f.kps:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
    if landmarks106 is not None:
        for i, (x, y) in enumerate(landmarks106):
            if i in LEFT_EYE or i in RIGHT_EYE:
                col = (255, 80, 80)
            elif i in MOUTH:
                col = (80, 80, 255)
            else:
                col = (0, 200, 255)
            cv2.circle(vis, (int(x), int(y)), 1, col, -1)
    y = 24
    cv2.putText(vis, "faces=%d" % len(faces), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    if ear is not None:
        y += 26
        cv2.putText(vis, "EAR=%.3f" % ear, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    if mar is not None:
        y += 26
        cv2.putText(vis, "MAR=%.3f" % mar, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
    if pose is not None:
        yaw, pitch, roll = pose
        y += 26
        cv2.putText(
            vis,
            "yaw=%.0f pitch=%.0f" % (yaw, pitch),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        if landmarks106 is not None:
            tdx, tdy = int(landmarks106[NOSE_TIP][0]), int(landmarks106[NOSE_TIP][1])
            _draw_axis(vis, yaw, pitch, roll, tdx, tdy, size=80)
    if track_id is not None:
        y += 26
        cv2.putText(vis, "id=%d" % track_id, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return vis


def _draw_axis(img, yaw, pitch, roll, tdx, tdy, size=80) -> None:
    """Same convention as models/headpose/drepnet/utils.py draw_axis."""
    pitch = pitch * np.pi / 180.0
    yaw = -(yaw * np.pi / 180.0)
    roll = roll * np.pi / 180.0
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)
