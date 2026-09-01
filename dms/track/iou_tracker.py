from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from dms.infer.scrfd import FaceDet


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    score: float
    lost: int = 0
    face: Optional[FaceDet] = None


class IouTracker:
    def __init__(self, iou_min: float = 0.3, max_lost: int = 20) -> None:
        self.iou_min = iou_min
        self.max_lost = max_lost
        self._next = 1
        self.tracks: List[Track] = []

    def update(self, faces: Sequence[FaceDet]) -> List[Track]:
        unused = set(range(len(faces)))
        for tr in self.tracks:
            best_i, best = None, self.iou_min
            for i in unused:
                v = _iou(tr.xyxy, faces[i].xyxy)
                if v > best:
                    best, best_i = v, i
            if best_i is None:
                tr.lost += 1
                tr.face = None
                continue
            f = faces[best_i]
            unused.discard(best_i)
            tr.xyxy = f.xyxy.copy()
            tr.score = f.score
            tr.lost = 0
            tr.face = f
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        for i in unused:
            f = faces[i]
            self.tracks.append(
                Track(track_id=self._next, xyxy=f.xyxy.copy(), score=f.score, lost=0, face=f)
            )
            self._next += 1
        return self.tracks
