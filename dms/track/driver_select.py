from __future__ import annotations

from typing import List, Optional, Sequence

from dms.infer.scrfd import FaceDet


def pick_driver(faces: Sequence[FaceDet], roi_xyxy_norm: Optional[List[float]], frame_wh) -> Optional[FaceDet]:
    """Prefer the highest-score face whose center is inside the seat ROI; else max score."""
    if not faces:
        return None
    w, h = frame_wh
    in_roi: List[FaceDet] = []
    if roi_xyxy_norm and len(roi_xyxy_norm) == 4:
        x1, y1, x2, y2 = roi_xyxy_norm
        rx1, ry1, rx2, ry2 = x1 * w, y1 * h, x2 * w, y2 * h
        for f in faces:
            cx = 0.5 * (float(f.xyxy[0]) + float(f.xyxy[2]))
            cy = 0.5 * (float(f.xyxy[1]) + float(f.xyxy[3]))
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                in_roi.append(f)
    pool = in_roi or list(faces)
    return max(pool, key=lambda f: f.score)
