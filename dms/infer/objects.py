"""In-cabin object detection (phone, bottle, cup) with YOLOX-Nano.

Engine contract (MANIFEST role ``cabin_objects``)::

    IN   "images" FLOAT [1, 3, 416, 416]    OUT  "output" FLOAT [1, 3549, 85]

Preprocessing - YOLOX is the odd one out and every part of this matters:

* **BGR**, no colour conversion.  It is the only BGR model in this project.
* **RAW 0-255**.  No ``/255``, no mean, no std.  Dividing by 255 does not error,
  it silences the model: the sibling ADAS workstream measured 10 detections with
  raw input and 0 with ``/255`` on the same frame.
* Letterbox ``r = min(416/h, 416/w)``, resize to ``(int(w*r), int(h*r))``, paste
  TOP-LEFT into a 416x416 canvas prefilled with 114.  YOLOX does not centre the
  pad, so the decode subtracts no offset.

Output decode - the grid decode is NOT baked into the graph, but the sigmoids
ARE.  3549 = 52^2 + 26^2 + 13^2, strides 8/16/32 concatenated in that order.
Each row is ``[dx, dy, dw, dh, obj, cls_0..cls_79]``::

    cx = (dx + grid_x) * stride      w = exp(dw) * stride
    cy = (dy + grid_y) * stride      h = exp(dh) * stride
    score = obj * cls_i

then to xyxy, divide by ``r``, then class-wise NMS.  ``obj`` and ``cls`` are
already sigmoid-activated; applying another sigmoid squashes every score into
[0.5, 0.73] and the detector stops firing.

COCO-80 indices used here: person 0, bottle 39, cup 41, **cell phone 67**.

ROI.  The detector runs on a region of interest, not the whole frame - by
default the lower part of the cabin view where hands and a phone can be.  That
keeps the letterbox source small (a 1280x396 slice rather than a 1280x720 frame)
and it keeps a phone on the passenger's lap out of the driver's alert.

Cadence.  Phones do not appear for one frame.  :meth:`ObjectDetector.detect`
honours ``every_n`` and returns the previous result in between, so the 4.9 ms
engine costs 4.9/N ms per frame.

Licence: YOLOX is Apache-2.0.  The AGPL gate that previously kept this capability
off no longer applies - that gate was about YOLOv8.

Failure behaviour: :meth:`detect` returns an empty list rather than raising, and
:meth:`phone_confidence` returns 0.0.  0.0 means "no evidence", and with no
engine loaded the capability reports ``available == False`` so the registry can
mark it degraded instead of letting /healthz advertise a model that is not
running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dms.infer import ModelWrapper, to_blob
from dms.infer.scrfd import nms_xyxy

#: COCO-80 class indices this project cares about.
COCO_PERSON = 0
COCO_BOTTLE = 39
COCO_CUP = 41
COCO_CELL_PHONE = 67

COCO_NAME_TO_ID: Dict[str, int] = {
    "person": COCO_PERSON,
    "bottle": COCO_BOTTLE,
    "cup": COCO_CUP,
    "cell phone": COCO_CELL_PHONE,
}

YOLOX_STRIDES = (8, 16, 32)
YOLOX_PAD_VALUE = 114
DEFAULT_CONF = 0.30
DEFAULT_NMS_IOU = 0.45


@dataclass
class ObjectDet:
    """One detection, in FULL-FRAME pixels."""

    xyxy: np.ndarray
    score: float
    cls_id: int
    name: str = ""


def yolox_grids(net_h: int, net_w: int, strides: Sequence[int] = YOLOX_STRIDES) -> Tuple[np.ndarray, np.ndarray]:
    """``(grid_xy, stride_per_row)`` for a YOLOX head, concatenated by stride."""
    grids = []
    expanded = []
    for s in strides:
        fh, fw = net_h // s, net_w // s
        yv, xv = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        grid = np.stack((xv, yv), axis=-1).reshape(-1, 2).astype(np.float32)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), float(s), dtype=np.float32))
    return np.concatenate(grids, axis=0), np.concatenate(expanded, axis=0)


def decode_yolox(
    raw: np.ndarray,
    net_h: int,
    net_w: int,
    ratio: float,
    conf_thres: float = DEFAULT_CONF,
    iou_thres: float = DEFAULT_NMS_IOU,
    keep_classes: Optional[Sequence[int]] = None,
) -> List[Tuple[np.ndarray, float, int]]:
    """Decode a (1, N, 85) YOLOX head into ROI-pixel ``(xyxy, score, cls)``.

    ``ratio`` is the letterbox scale that was applied to the ROI; boxes are
    divided by it, with no pad offset because YOLOX pads bottom/right only.
    """
    pred = np.asarray(raw, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2 or pred.shape[1] < 6:
        raise ValueError("YOLOX output must be (N, 5+C), got %s" % (pred.shape,))
    grid, stride = yolox_grids(net_h, net_w)
    if pred.shape[0] != grid.shape[0]:
        raise ValueError(
            "YOLOX emitted %d rows but the %dx%d grid has %d - engine and decode disagree"
            % (pred.shape[0], net_w, net_h, grid.shape[0])
        )
    cxcy = (pred[:, 0:2] + grid) * stride
    wh = np.exp(np.clip(pred[:, 2:4], -20.0, 20.0)) * stride
    obj = pred[:, 4:5]
    cls = pred[:, 5:]
    scores = obj * cls
    if keep_classes is not None:
        mask = np.zeros(cls.shape[1], dtype=bool)
        for c in keep_classes:
            if 0 <= int(c) < mask.shape[0]:
                mask[int(c)] = True
        scores = np.where(mask[None, :], scores, 0.0)
    best_cls = np.argmax(scores, axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_cls]
    keep = np.where(best_score >= conf_thres)[0]
    if keep.size == 0:
        return []
    cxcy, wh = cxcy[keep], wh[keep]
    best_cls, best_score = best_cls[keep], best_score[keep]
    boxes = np.empty((keep.size, 4), dtype=np.float32)
    boxes[:, 0] = cxcy[:, 0] - 0.5 * wh[:, 0]
    boxes[:, 1] = cxcy[:, 1] - 0.5 * wh[:, 1]
    boxes[:, 2] = cxcy[:, 0] + 0.5 * wh[:, 0]
    boxes[:, 3] = cxcy[:, 1] + 0.5 * wh[:, 1]
    boxes /= float(ratio)

    out: List[Tuple[np.ndarray, float, int]] = []
    for c in np.unique(best_cls):
        sel = np.where(best_cls == c)[0]
        dets = np.hstack((boxes[sel], best_score[sel][:, None])).astype(np.float32)
        for i in nms_xyxy(dets, iou_thres):
            out.append((boxes[sel][i], float(best_score[sel][i]), int(c)))
    out.sort(key=lambda t: -t[1])
    return out


def letterbox_topleft(
    bgr: np.ndarray, net_w: int, net_h: int, pad_value: int = YOLOX_PAD_VALUE
) -> Tuple[np.ndarray, float]:
    """YOLOX letterbox: aspect-preserving resize pasted TOP-LEFT on a 114 canvas."""
    h, w = bgr.shape[:2]
    r = min(float(net_h) / float(h), float(net_w) / float(w))
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((net_h, net_w, 3), pad_value, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r


class ObjectDetector(ModelWrapper):
    """YOLOX-Nano over a cabin ROI, throttled to every Nth frame."""

    capability = "objects"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        name: str = "objects",
    ) -> None:
        super().__init__(engine=engine, cfg=cfg, name=name)
        roi = self.cfg.get("roi") or [0.0, 0.45, 1.0, 1.0]
        self.roi = [float(v) for v in roi]
        # An explicitly empty list means "every COCO class"; an absent key means
        # the default of phones only.  `or` would conflate the two.
        names = self.cfg.get("classes")
        if names is None:
            names = ["cell phone"]
        self.classes = [COCO_NAME_TO_ID[n] for n in names if n in COCO_NAME_TO_ID]
        self._id_to_name = {v: k for k, v in COCO_NAME_TO_ID.items()}
        self.conf = float(self.cfg.get("conf", DEFAULT_CONF))
        self.iou = float(self.cfg.get("iou", DEFAULT_NMS_IOU))
        self.every_n = max(1, int(self.cfg.get("every_n", 2)))
        self.net_w, self.net_h = 416, 416
        if engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(engine)
            if len(shape) == 4:
                self.net_h, self.net_w = int(shape[2]), int(shape[3])
        self._n = 0
        self._last: List[ObjectDet] = []

    def _roi_px(self, frame_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = int(frame_wh[0]), int(frame_wh[1])
        x1 = int(max(0.0, min(1.0, self.roi[0])) * w)
        y1 = int(max(0.0, min(1.0, self.roi[1])) * h)
        x2 = int(max(0.0, min(1.0, self.roi[2])) * w)
        y2 = int(max(0.0, min(1.0, self.roi[3])) * h)
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    def detect(self, bgr: np.ndarray, *, force: bool = False) -> List[ObjectDet]:
        """Objects in the cabin ROI, in full-frame pixels.

        Returns the previous result on the frames this call is throttled out of,
        so the caller sees a stable list rather than a flicker.
        """
        if bgr is None or not self.available:
            return []
        n = self._n
        self._n += 1
        if not force and (n % self.every_n) != 0:
            return self._last

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = self._roi_px((w, h))
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        canvas, ratio = letterbox_topleft(roi, self.net_w, self.net_h)
        # YOLOX: BGR, raw 0-255, no mean/std.
        blob = to_blob(canvas, rgb=False, scale=1.0, mean=None, std=None)
        out = self.run_engine({self._in_names[0]: blob})
        if out is None:
            return self._last
        try:
            dets = decode_yolox(
                out[self._out_names[0]],
                self.net_h,
                self.net_w,
                ratio,
                conf_thres=self.conf,
                iou_thres=self.iou,
                keep_classes=self.classes or None,
            )
        except ValueError as exc:
            self.note_failure(exc)
            return self._last
        result: List[ObjectDet] = []
        for box, score, cls_id in dets:
            full = box.copy()
            full[0] += x1
            full[1] += y1
            full[2] += x1
            full[3] += y1
            result.append(
                ObjectDet(
                    xyxy=full.astype(np.float32),
                    score=score,
                    cls_id=cls_id,
                    name=self._id_to_name.get(cls_id, str(cls_id)),
                )
            )
        self._last = result
        return result

    def phone_confidence(self, dets: Sequence[ObjectDet]) -> float:
        """Highest cell-phone score in ``dets``. 0.0 means no evidence."""
        best = 0.0
        for d in dets:
            if d.cls_id == COCO_CELL_PHONE and d.score > best:
                best = d.score
        return float(best)
