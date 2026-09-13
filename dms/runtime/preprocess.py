"""Full-frame preprocessing, done once and allocation-free.

Exactly one full-frame geometric operation is allowed on the hot path: letterbox the
captured 720p frame into the detector input. Everything downstream (landmarks, eye,
mouth, occlusion crops) works on small crops of the same frame.

This module owns that operation. Every destination buffer is preallocated in the
:class:`Letterbox` instance and reused, so a steady-state frame costs zero Python
allocations - the previous per-frame ``astype``/``-``/``/``/``transpose``/
``ascontiguousarray`` chain allocated roughly 25 MB per frame for one 4.9 MB blob.

Where it is used
----------------
:class:`Letterbox` is built by ``DmsApp._setup_perception`` from the SCRFD engine's
own input size and called in the ``letterbox`` budget stage of ``DmsApp._perceive``;
:class:`BgraToBgr` runs one step earlier on every captured frame. Until 2026-09-13
only ``BgraToBgr`` was wired in and this class was reachable only from
``scripts/bench.py`` - the benchmark quoted its numbers while the application still
ran the allocating chain inside ``ScrfdDetector.preprocess``.

MEASURED on the Xavier NX at 1280x720 (200 iterations, p50), ``Letterbox.blob``
against that chain: 4.99 ms vs 6.70 ms into a 640x640 blob and 3.67 ms vs 5.45 ms
into 512x512, with a maximum absolute difference of 1.19e-07 between the two blobs
and an identical ``factor`` at every source size tested. ``ScrfdDetector.preprocess``
and the free function ``dms.infer.scrfd.resize_letterbox`` still exist and are still
what an external caller gets; ``dms/infer/objects.py:letterbox_topleft`` is a third
implementation that has NOT been collapsed onto this one (both files belong to
another workstream - see the handoff).

Units and conventions
---------------------
* Images are HWC uint8 as OpenCV delivers them (BGR unless stated otherwise).
* ``scale``/``mean``/``std`` implement ``out = (px * scale - mean) / std`` on the raw
  0..255 pixel value, per channel, in the *output* colour order.
* ``factor`` is the single uniform resize ratio; ``pad_x``/``pad_y`` are the pixel
  offsets of the pasted image inside the canvas. Map a detection back to source
  coordinates with ``geom.to_source(xy)``.

Threading: a :class:`Letterbox` owns mutable scratch buffers and is **not**
thread-safe. Give each worker thread its own instance.

Failure behaviour: a frame of the wrong dtype or channel count raises ``ValueError``
immediately - there is no silent reinterpretation of the buffer.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# ImageNet statistics on the 0..1 range, as used by the head-pose and gaze engines.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pad value conventions seen on this board.
PAD_YOLO = 114  # YOLOX / YOLOv5 letterbox grey
PAD_SCRFD = 0  # UniFace SCRFD pads with black, top-left only


class Geometry(object):
    """Where the resized image sits inside the canvas, and how to undo it."""

    __slots__ = ("src_h", "src_w", "dst_h", "dst_w", "factor", "pad_x", "pad_y", "new_w", "new_h")

    def __init__(self, src_h, src_w, dst_h, dst_w, factor, pad_x, pad_y, new_w, new_h):
        # type: (int, int, int, int, float, int, int, int, int) -> None
        self.src_h = src_h
        self.src_w = src_w
        self.dst_h = dst_h
        self.dst_w = dst_w
        self.factor = factor
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.new_w = new_w
        self.new_h = new_h

    def to_source(self, xy):
        # type: (np.ndarray) -> np.ndarray
        """Map canvas-space coordinates (...,2) or boxes (...,4) back to source pixels."""
        arr = np.asarray(xy, dtype=np.float32)
        out = arr.copy()
        if arr.shape[-1] % 2 != 0:
            raise ValueError("expected an even trailing dimension, got %s" % (arr.shape,))
        out[..., 0::2] -= self.pad_x
        out[..., 1::2] -= self.pad_y
        out /= self.factor
        return out

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Geometry(src=%dx%d dst=%dx%d factor=%.5f pad=(%d,%d))" % (
            self.src_w,
            self.src_h,
            self.dst_w,
            self.dst_h,
            self.factor,
            self.pad_x,
            self.pad_y,
        )


class Letterbox(object):
    """Aspect-preserving resize into a fixed detector input, reusing its buffers.

    Parameters
    ----------
    dst_hw : (height, width) of the network input, in pixels.
    pad_value : the byte written to the padded region (114 for YOLO, 0 for SCRFD).
    center : True pads on all four sides (YOLOv5/YOLOP), False pads bottom/right only
        (YOLOX, SCRFD/UniFace). Getting this wrong shifts every detection.
    to_rgb : emit the blob in RGB order. Free - the channel planes are copied
        out anyway, so the swap is just a reversed view.
    scale, mean, std : normalisation, applied as ``(px * scale - mean) / std``.
    layout : ``"nchw"`` (default) or ``"nhwc"`` for the returned blob.
    """

    def __init__(
        self,
        dst_hw,  # type: Tuple[int, int]
        pad_value=PAD_YOLO,  # type: int
        center=False,  # type: bool
        to_rgb=False,  # type: bool
        scale=1.0,  # type: float
        mean=0.0,  # type: object
        std=1.0,  # type: object
        layout="nchw",  # type: str
        interpolation=cv2.INTER_LINEAR,  # type: int
    ):
        # type: (...) -> None
        h, w = int(dst_hw[0]), int(dst_hw[1])
        if h <= 0 or w <= 0:
            raise ValueError("dst_hw must be positive, got %s" % (dst_hw,))
        if layout not in ("nchw", "nhwc"):
            raise ValueError("layout must be nchw or nhwc, got %r" % (layout,))
        self.dst_h = h
        self.dst_w = w
        self.pad_value = int(pad_value)
        self.center = bool(center)
        self.to_rgb = bool(to_rgb)
        self.layout = layout
        self.interpolation = interpolation

        mean_a = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_a = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean_a.size not in (1, 3) or std_a.size not in (1, 3):
            raise ValueError("mean/std must be scalar or per-channel (3,)")
        if np.any(std_a == 0):
            raise ValueError("std must be non-zero")
        # Fold (px * scale - mean) / std into a single multiply-add per channel:
        # px * a + b. Applied plane by plane with *scalar* operands - a (3,) operand
        # broadcast over the last axis costs ~6x more (7.0 ms vs 1.2 ms at 640x640 on
        # this board) because numpy cannot use its contiguous inner loop.
        self._a = np.broadcast_to(np.float32(scale) / std_a, (3,)).astype(np.float32)
        self._b = np.broadcast_to(-mean_a / std_a, (3,)).astype(np.float32)

        self._canvas = np.empty((h, w, 3), dtype=np.uint8)
        self._canvas[:] = self.pad_value
        if layout == "nchw":
            self._blob = np.empty((1, 3, h, w), dtype=np.float32)
        else:
            self._blob = np.empty((1, h, w, 3), dtype=np.float32)
        self._geom = None  # type: Optional[Geometry]

    # ------------------------------------------------------------------ geometry

    def geometry(self, src_hw):
        # type: (Tuple[int, int]) -> Geometry
        """Geometry for a source of this size. Cached; recomputed only on a change."""
        sh, sw = int(src_hw[0]), int(src_hw[1])
        g = self._geom
        if g is not None and g.src_h == sh and g.src_w == sw:
            return g
        if sh <= 0 or sw <= 0:
            raise ValueError("source frame must be non-empty, got %sx%s" % (sw, sh))
        factor = min(self.dst_h / float(sh), self.dst_w / float(sw))
        new_w = max(1, int(round(sw * factor)))
        new_h = max(1, int(round(sh * factor)))
        # Clamp: rounding can overshoot by one pixel on odd sizes.
        new_w = min(new_w, self.dst_w)
        new_h = min(new_h, self.dst_h)
        if self.center:
            pad_x = (self.dst_w - new_w) // 2
            pad_y = (self.dst_h - new_h) // 2
        else:
            pad_x = 0
            pad_y = 0
        g = Geometry(sh, sw, self.dst_h, self.dst_w, factor, pad_x, pad_y, new_w, new_h)
        self._geom = g
        # The pad region only changes when the geometry does, so repaint it here.
        self._canvas[:] = self.pad_value
        return g

    # -------------------------------------------------------------------- canvas

    def canvas(self, frame_bgr):
        # type: (np.ndarray) -> Tuple[np.ndarray, Geometry]
        """Resize+paste into the reusable uint8 canvas. Returns (canvas, geometry).

        The canvas is always BGR - ``to_rgb`` is applied in :meth:`blob`, where the
        channel swap is free because the planes are copied out anyway. The canvas is
        owned by this Letterbox and is overwritten by the next call.
        """
        if frame_bgr.dtype != np.uint8:
            raise ValueError("expected uint8 frame, got %s" % frame_bgr.dtype)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("expected HxWx3 BGR frame, got %s" % (frame_bgr.shape,))
        g = self.geometry(frame_bgr.shape[:2])
        roi = self._canvas[g.pad_y : g.pad_y + g.new_h, g.pad_x : g.pad_x + g.new_w]
        if (g.new_h, g.new_w) == (g.src_h, g.src_w):
            np.copyto(roi, frame_bgr)
        else:
            cv2.resize(
                frame_bgr,
                (g.new_w, g.new_h),
                dst=roi,
                interpolation=self.interpolation,
            )
        return self._canvas, g

    # ---------------------------------------------------------------------- blob

    def blob(self, frame_bgr):
        # type: (np.ndarray) -> Tuple[np.ndarray, Geometry]
        """Full letterbox + normalise into the reusable float32 network blob.

        Returns (blob, geometry). The blob is owned by this Letterbox: it is valid
        until the next call, which is exactly the lifetime TrtEngine.submit needs.
        """
        canvas, g = self.canvas(frame_bgr)
        src = canvas[:, :, ::-1] if self.to_rgb else canvas
        blob = self._blob
        if self.layout == "nchw":
            # One cast+transpose pass for all three planes, then scalar ops on each
            # contiguous plane. mean/std are in output-channel order.
            blob[0] = src.transpose(2, 0, 1)
            for c in range(3):
                plane = blob[0, c]
                a = float(self._a[c])
                b = float(self._b[c])
                if a != 1.0:
                    np.multiply(plane, np.float32(a), out=plane)
                if b != 0.0:
                    np.add(plane, np.float32(b), out=plane)
        else:
            blob[0] = src
            for c in range(3):
                plane = blob[0, :, :, c]
                a = float(self._a[c])
                b = float(self._b[c])
                if a != 1.0:
                    np.multiply(plane, np.float32(a), out=plane)
                if b != 0.0:
                    np.add(plane, np.float32(b), out=plane)
        return blob, g

    # ------------------------------------------------------------- constructors

    @classmethod
    def for_scrfd(cls, dst_hw=(640, 640)):
        # type: (Tuple[int, int]) -> Letterbox
        """UniFace SCRFD: RGB-agnostic BGR in, top-left pad 0, (px - 127.5) / 127.5."""
        return cls(dst_hw, pad_value=PAD_SCRFD, center=False, to_rgb=False,
                   scale=1.0, mean=127.5, std=127.5)

    @classmethod
    def for_yolox(cls, dst_hw=(416, 416)):
        # type: (Tuple[int, int]) -> Letterbox
        """YOLOX: BGR, top-left pad 114, raw 0..255 (no /255, no mean/std)."""
        return cls(dst_hw, pad_value=PAD_YOLO, center=False, to_rgb=False,
                   scale=1.0, mean=0.0, std=1.0)

    @classmethod
    def for_imagenet(cls, dst_hw, pad_value=PAD_YOLO, center=True):
        # type: (Tuple[int, int], int, bool) -> Letterbox
        """RGB, /255 then ImageNet mean/std - the head-pose and gaze convention."""
        return cls(dst_hw, pad_value=pad_value, center=center, to_rgb=True,
                   scale=1.0 / 255.0, mean=IMAGENET_MEAN, std=IMAGENET_STD)


class BgraToBgr(object):
    """Drop the alpha plane of the captured BGRx frame once, into a reused buffer.

    ``frame.full_bgra[..., :3]`` is a stride-4 view: every OpenCV call downstream
    (resize, warpAffine, the overlay) materialises its own contiguous copy, so the
    same 2.7 MB de-interleave happens two to four times per frame. Doing it once with
    ``cvtColor`` is a single vectorised pass.

    Not thread-safe; the returned array is valid until the next call.
    """

    __slots__ = ("_out",)

    def __init__(self):
        # type: () -> None
        self._out = None  # type: Optional[np.ndarray]

    def __call__(self, bgra):
        # type: (np.ndarray) -> np.ndarray
        if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("expected HxWx4 uint8 BGRA, got %s %s" % (bgra.shape, bgra.dtype))
        h, w = bgra.shape[:2]
        out = self._out
        if out is None or out.shape[0] != h or out.shape[1] != w:
            out = np.empty((h, w, 3), dtype=np.uint8)
            self._out = out
        cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=out)
        return out


def bgra_to_bgr(bgra, out=None):
    # type: (np.ndarray, Optional[np.ndarray]) -> np.ndarray
    """One-shot BGRA -> contiguous BGR. Pass ``out`` to reuse a buffer."""
    if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("expected HxWx4 uint8 BGRA, got %s %s" % (bgra.shape, bgra.dtype))
    h, w = bgra.shape[:2]
    if out is None:
        out = np.empty((h, w, 3), dtype=np.uint8)
    elif out.shape != (h, w, 3) or out.dtype != np.uint8:
        raise ValueError("out must be %s uint8, got %s %s" % ((h, w, 3), out.shape, out.dtype))
    cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR, dst=out)
    return out


def normalize_chw(canvas_bgr, scale=1.0, mean=0.0, std=1.0, to_rgb=False, out=None):
    # type: (np.ndarray, float, object, object, bool, Optional[np.ndarray]) -> np.ndarray
    """Normalise an already-sized HWC uint8 crop into a (1,3,H,W) float32 blob.

    For the small per-crop models (eye, mouth, occlusion, head pose) whose input is a
    resized crop rather than a letterboxed full frame. Pass ``out`` to reuse a buffer.
    """
    if canvas_bgr.dtype != np.uint8 or canvas_bgr.ndim != 3 or canvas_bgr.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8, got %s %s" % (canvas_bgr.shape, canvas_bgr.dtype))
    h, w = canvas_bgr.shape[:2]
    if out is None:
        out = np.empty((1, 3, h, w), dtype=np.float32)
    elif out.shape != (1, 3, h, w) or out.dtype != np.float32:
        raise ValueError("out must be %s float32" % ((1, 3, h, w),))
    src = canvas_bgr[:, :, ::-1] if to_rgb else canvas_bgr
    mean_a = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_a = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_a.size not in (1, 3) or std_a.size not in (1, 3):
        raise ValueError("mean/std must be scalar or per-channel (3,)")
    if np.any(std_a == 0):
        raise ValueError("std must be non-zero")
    a = np.broadcast_to(np.float32(scale) / std_a, (3,)).astype(np.float32)
    b = np.broadcast_to(-mean_a / std_a, (3,)).astype(np.float32)
    out[0] = src.transpose(2, 0, 1)
    for c in range(3):
        plane = out[0, c]
        if float(a[c]) != 1.0:
            np.multiply(plane, np.float32(a[c]), out=plane)
        if float(b[c]) != 0.0:
            np.add(plane, np.float32(b[c]), out=plane)
    return out


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PAD_YOLO",
    "PAD_SCRFD",
    "Geometry",
    "Letterbox",
    "BgraToBgr",
    "bgra_to_bgr",
    "normalize_chw",
]
