from __future__ import annotations

from typing import List

import cv2
import numpy as np

from dms.infer.scrfd import FaceDet


def draw_faces(bgr: np.ndarray, faces: List[FaceDet]) -> np.ndarray:
    vis = bgr.copy()
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            "%.2f" % f.score,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if f.kps is not None:
            for x, y in f.kps:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
    cv2.putText(
        vis,
        "faces=%d" % len(faces),
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return vis
