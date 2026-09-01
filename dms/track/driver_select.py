from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from dms.infer.scrfd import FaceDet
from dms.track.iou_tracker import Track


def _area(f: FaceDet) -> float:
    return max(0.0, float(f.xyxy[2] - f.xyxy[0])) * max(0.0, float(f.xyxy[3] - f.xyxy[1]))


def _in_roi(f: FaceDet, roi_xyxy_norm: Sequence[float], frame_wh) -> bool:
    w, h = frame_wh
    x1, y1, x2, y2 = roi_xyxy_norm
    rx1, ry1, rx2, ry2 = x1 * w, y1 * h, x2 * w, y2 * h
    cx = 0.5 * (float(f.xyxy[0]) + float(f.xyxy[2]))
    cy = 0.5 * (float(f.xyxy[1]) + float(f.xyxy[3]))
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def pick_driver(
    faces: Sequence[FaceDet],
    roi_xyxy_norm: Optional[List[float]],
    frame_wh,
    fallback: bool = True,
) -> Optional[FaceDet]:
    """Driver = largest (area*score) face in the seat ROI.

    If the ROI is empty and fallback is False, return None (FACE_LOST) rather
    than labeling a rear passenger as the driver.
    """
    if not faces:
        return None
    pool = list(faces)
    if roi_xyxy_norm and len(roi_xyxy_norm) == 4:
        in_roi = [f for f in faces if _in_roi(f, roi_xyxy_norm, frame_wh)]
        if in_roi:
            pool = in_roi
        elif not fallback:
            return None
    return max(pool, key=lambda f: _area(f) * (0.3 + float(f.score)))


def label_occupants(live: Sequence[Track], driver_face: Optional[FaceDet]) -> List[Tuple[Track, str]]:
    """Assign DRV / P1 / P2 ... for overlay. Alerts stay on DRV only."""
    driver_tid = None
    for t in live:
        if driver_face is not None and t.face is driver_face:
            driver_tid = t.track_id
            break
    out: List[Tuple[Track, str]] = []
    p = 1
    for t in sorted(live, key=lambda x: x.track_id):
        if t.track_id == driver_tid:
            out.append((t, "DRV"))
        else:
            out.append((t, "P%d" % p))
            p += 1
    return out
