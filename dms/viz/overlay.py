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
    alerts: Optional[List[str]] = None,
    yaw_rel: Optional[float] = None,
    occupant_labels: Optional[List[Tuple[object, str]]] = None,
) -> np.ndarray:
    vis = bgr.copy()
    label_by_face = {}
    if occupant_labels:
        for tr, role in occupant_labels:
            if getattr(tr, "face", None) is not None:
                label_by_face[id(tr.face)] = role
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.xyxy]
        is_drv = driver is not None and f is driver
        color = (0, 255, 255) if is_drv else (0, 255, 0)
        thick = 3 if is_drv else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        role = label_by_face.get(id(f), "DRV" if is_drv else "P")
        cv2.putText(
            vis,
            "%s %.2f" % (role, f.score),
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
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
    n_pax = max(0, len(faces) - (1 if driver is not None else 0))
    cv2.putText(
        vis,
        "occ=%d drv=%d pax=%d" % (len(faces), 1 if driver is not None else 0, n_pax),
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
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
    if yaw_rel is not None:
        y += 26
        cv2.putText(vis, "yaw_rel=%.0f" % yaw_rel, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2, cv2.LINE_AA)
    if alerts:
        y += 28
        txt = " | ".join(alerts)
        cv2.putText(vis, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
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
