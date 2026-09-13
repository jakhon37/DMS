"""BYTE-lite face tracking.

Association is two-stage, the cheap half of ByteTrack:

1. high-confidence detections (``score >= high_score``) are matched to every
   live track, greedily, best IoU first;
2. tracks still unmatched get a second pass against the LOW-confidence
   detections that stage 1 ignored, at a looser IoU gate.  A face going briefly
   dark or half out of frame usually survives as a weak detection, and using it
   for association only - never to spawn a new id - keeps the id alive without
   letting detector noise mint tracks.

Only high-confidence leftovers spawn new tracks.  An unmatched track coasts for
``max_lost`` frames before it is dropped, so a blink or a one-frame miss does
not renumber the driver.

Coasting is for ID CONTINUITY ONLY.  A coasting track has ``face is None`` and
``lost > 0``; it must never count as a present driver, because ``FACE_LOST`` is
defined on the last *matched* detection in the seat ROI, not on the existence of
a track id.  :attr:`Track.t_last_match_s` records that instant when the caller
passes a timestamp.

Units: boxes are full-frame pixels ``[x1, y1, x2, y2]``; ``t_s`` is seconds on
the frame clock.  Failure behaviour: :meth:`IouTracker.update` never raises; a
non-finite or empty box simply fails to match anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dms.infer.scrfd import FaceDet


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two xyxy boxes.  0.0 when either is empty."""
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
    """One tracked face.

    ``lost`` counts consecutive frames with no matched detection; ``0`` means
    matched this frame, which is the only state that counts as present.
    ``hits`` is the total number of matched frames and ``age`` the number of
    frames since the track was created - together they are the persistence a
    driver selector needs to prefer an established track over a fresh one.
    ``t_last_match_s`` is ``None`` until the caller passes a timestamp.
    """

    track_id: int
    xyxy: np.ndarray
    score: float
    lost: int = 0
    face: Optional[FaceDet] = None
    hits: int = 1
    age: int = 1
    t_start_s: Optional[float] = None
    t_last_match_s: Optional[float] = None

    @property
    def matched(self) -> bool:
        """Matched a detection on the most recent update."""
        return self.lost == 0 and self.face is not None


class IouTracker:
    """Greedy IoU tracker with a low-score second association pass.

    ``iou_min``    gate for the high-score pass.
    ``iou_min_low`` gate for the low-score recovery pass; looser because a weak
                   detection is usually a poorly localised one.
    ``high_score`` detections at or above this may spawn a track; below it they
                   are association fodder only.
    ``max_lost``   frames a track may coast before it is dropped (~1.3 s at
                   15 FPS with the default 20).
    """

    def __init__(
        self,
        iou_min: float = 0.3,
        max_lost: int = 20,
        iou_min_low: float = 0.2,
        high_score: float = 0.45,
    ) -> None:
        self.iou_min = float(iou_min)
        self.iou_min_low = float(iou_min_low)
        self.high_score = float(high_score)
        self.max_lost = int(max_lost)
        self._next = 1
        self.tracks: List[Track] = []

    def reset(self) -> None:
        """Drop every track.  Ids are NOT reused - a new person gets a new id."""
        self.tracks = []

    # ------------------------------------------------------------- update --
    def update(self, faces: Sequence[FaceDet], t_s: Optional[float] = None) -> List[Track]:
        """Associate ``faces`` to the live tracks and return every live track.

        The returned list includes coasting tracks (``lost > 0``); filter on
        :attr:`Track.matched` for presence.
        """
        for tr in self.tracks:
            tr.age += 1

        high = [i for i, f in enumerate(faces) if float(f.score) >= self.high_score]
        low = [i for i, f in enumerate(faces) if float(f.score) < self.high_score]

        unmatched_tracks = list(range(len(self.tracks)))
        matched_dets: Dict[int, int] = {}  # det index -> track index

        for det_pool, gate in ((high, self.iou_min), (low, self.iou_min_low)):
            pool = [i for i in det_pool if i not in matched_dets]
            if not pool or not unmatched_tracks:
                continue
            pairs: List[Tuple[float, int, int]] = []
            for ti in unmatched_tracks:
                for di in pool:
                    value = _iou(self.tracks[ti].xyxy, faces[di].xyxy)
                    if value > gate:
                        pairs.append((value, ti, di))
            # Greedy global assignment: the single best pair wins, then the best
            # remaining, and so on.  Cheap, deterministic, and it does not let
            # track order decide who claims a contested detection.
            pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
            taken_t = set()
            taken_d = set()
            for _, ti, di in pairs:
                if ti in taken_t or di in taken_d:
                    continue
                taken_t.add(ti)
                taken_d.add(di)
                matched_dets[di] = ti
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in taken_t]

        for di, ti in matched_dets.items():
            tr = self.tracks[ti]
            face = faces[di]
            tr.xyxy = np.asarray(face.xyxy, dtype=np.float32).copy()
            tr.score = float(face.score)
            tr.lost = 0
            tr.face = face
            tr.hits += 1
            if t_s is not None:
                tr.t_last_match_s = float(t_s)

        for ti in unmatched_tracks:
            tr = self.tracks[ti]
            tr.lost += 1
            tr.face = None

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        for di in high:
            if di in matched_dets:
                continue
            face = faces[di]
            self.tracks.append(
                Track(
                    track_id=self._next,
                    xyxy=np.asarray(face.xyxy, dtype=np.float32).copy(),
                    score=float(face.score),
                    lost=0,
                    face=face,
                    hits=1,
                    age=1,
                    t_start_s=None if t_s is None else float(t_s),
                    t_last_match_s=None if t_s is None else float(t_s),
                )
            )
            self._next += 1
        return self.tracks

    def matched_tracks(self) -> List[Track]:
        """Tracks that matched a detection on the most recent update."""
        return [t for t in self.tracks if t.matched]
