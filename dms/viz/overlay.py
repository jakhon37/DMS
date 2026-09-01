from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from dms.geometry.face106 import LEFT_EYE, MOUTH, RIGHT_EYE
from dms.infer.scrfd import FaceDet


def draw_faces(
    bgr: np.ndarray,
    faces: List[FaceDet],
    *,
    driver: Optional[FaceDet] = None,
    landmarks106: Optional[np.ndarray] = None,
    ear: Optional[float] = None,
    mar: Optional[float] = None,
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
    return vis
