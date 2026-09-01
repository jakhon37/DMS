"""SCRFD-500m face detector. Decode follows UniFace, not the uniface package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from dms.runtime.trt_engine import TrtEngine


@dataclass
class FaceDet:
    xyxy: np.ndarray  # float32 [4] full-frame px
    score: float
    kps: Optional[np.ndarray] = None  # (5, 2) full-frame px


def resize_letterbox(frame_bgr: np.ndarray, size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float]:
    """UniFace resize_image: fit inside size, top-left pad 0 (not YOLO pad-114)."""
    height, width = size[1], size[0]
    im_ratio = float(frame_bgr.shape[0]) / frame_bgr.shape[1]
    model_ratio = height / float(width)
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)
    factor = float(new_height) / frame_bgr.shape[0]
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:new_height, :new_width] = resized
    return canvas, factor


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms_xyxy(dets: np.ndarray, threshold: float) -> List[int]:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= threshold)[0] + 1]
    return keep


def _anchor_centers(height: int, width: int, stride: int, num_anchors: int = 2) -> np.ndarray:
    fm_h = height // stride
    fm_w = width // stride
    y, x = np.mgrid[:fm_h, :fm_w]
    centers = np.stack((x, y), axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    if num_anchors > 1:
        centers = np.tile(centers[:, None, :], (1, num_anchors, 1)).reshape(-1, 2)
    return centers


def _as_2d(arr: np.ndarray, cols: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    a = np.reshape(a, (-1, cols))
    return a


class ScrfdDetector:
    strides = (8, 16, 32)
    num_anchors = 2
    input_size = (640, 640)

    def __init__(self, engine: TrtEngine, conf: float = 0.5, iou: float = 0.4) -> None:
        self.engine = engine
        self.conf = conf
        self.iou = iou
        self._input_name = engine._inputs[0].name
        self._out_names = [b.name for b in engine._outputs]
        self._centers = {s: _anchor_centers(640, 640, s, self.num_anchors) for s in self.strides}

    def preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        canvas, factor = resize_letterbox(bgr, self.input_size)
        blob = canvas.astype(np.float32)
        blob = (blob - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob), factor

    def detect(self, bgr: np.ndarray) -> List[FaceDet]:
        blob, factor = self.preprocess(bgr)
        raw = self.engine.infer({self._input_name: blob})
        return self.postprocess(raw, factor, orig_hw=bgr.shape[:2])

    def postprocess(self, raw: dict, factor: float, orig_hw: Tuple[int, int]) -> List[FaceDet]:
        outs = [raw[n] for n in self._out_names]
        grouped = _group_outputs(outs)
        scores_l: List[np.ndarray] = []
        boxes_l: List[np.ndarray] = []
        kps_l: List[np.ndarray] = []
        for stride, (scores, bbox, kps) in zip(self.strides, grouped):
            scores = _as_2d(scores, 1).reshape(-1)
            bbox = _as_2d(bbox, 4) * stride
            kps = _as_2d(kps, 10) * stride
            centers = self._centers[stride]
            n = min(scores.shape[0], centers.shape[0], bbox.shape[0])
            scores, bbox, kps, centers = scores[:n], bbox[:n], kps[:n], centers[:n]
            pos = np.where(scores >= self.conf)[0]
            if pos.size == 0:
                continue
            boxes_l.append(distance2bbox(centers, bbox)[pos])
            scores_l.append(scores[pos])
            kps_l.append(distance2kps(centers, kps)[pos])
        if not scores_l:
            return []
        scores = np.concatenate(scores_l, axis=0)
        boxes = np.concatenate(boxes_l, axis=0) / factor
        kps = np.concatenate(kps_l, axis=0) / factor
        order = scores.argsort()[::-1]
        boxes, scores, kps = boxes[order], scores[order], kps[order]
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
        keep = nms_xyxy(dets, self.iou)
        h, w = orig_hw
        faces: List[FaceDet] = []
        for i in keep:
            xyxy = np.clip(boxes[i], [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
            pts = kps[i].reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            faces.append(FaceDet(xyxy=xyxy.astype(np.float32), score=float(scores[i]), kps=pts.astype(np.float32)))
        return faces


def _group_outputs(outs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map 9 SCRFD heads. TRT often emits score/bbox/kps interleaved per stride."""
    if len(outs) != 9:
        raise RuntimeError("SCRFD expected 9 outputs, got %d" % len(outs))
    cols = []
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        last = a.shape[-1]
        cols.append(last)
    # interleaved: (1,4,10) x 3
    if cols[:3] == [1, 4, 10]:
        return [(outs[i], outs[i + 1], outs[i + 2]) for i in (0, 3, 6)]
    # UniFace ORT: 3 scores, 3 bbox, 3 kps
    if cols[:3] == [1, 1, 1]:
        return [(outs[i], outs[3 + i], outs[6 + i]) for i in range(3)]
    raise RuntimeError("unrecognized SCRFD output layout: %s" % cols)
