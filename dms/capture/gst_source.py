from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np

from dms.capture.pipelines import build_pipeline
from dms.config.schema import AppConfig
from dms.types import Frame, SourceType

log = logging.getLogger("dms.capture")


class GstSource:
    """One 720p BGRx appsink. Copies packed BGRx before unmap."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._q: Queue = Queue(maxsize=max(1, cfg.capture.queue_depth))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline = None
        self._frame_id = 0
        self.source = SourceType(cfg.source.type)

    def start(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        desc = build_pipeline(self._cfg)
        log.info("gst pipeline: %s", desc)
        pipeline = Gst.parse_launch(desc)
        if pipeline is None:
            raise RuntimeError("failed to parse GStreamer pipeline")
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer failed to PLAYING")
        self._pipeline = pipeline
        self._thread = threading.Thread(target=self._loop, name="gst-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        assert self._pipeline is not None
        sink = self._pipeline.get_by_name("full")
        if sink is None:
            log.error("appsink 'full' missing")
            return
        w, h = self._cfg.source.width, self._cfg.source.height
        while not self._stop.is_set():
            sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            if buf is None:
                continue
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                packed = _copy_bgra(mapinfo, buf, GstVideo, w, h)
            finally:
                buf.unmap(mapinfo)
            self._frame_id += 1
            frame = Frame(
                frame_id=self._frame_id,
                t_mono_ns=time.monotonic_ns(),
                full_bgra=packed,
                ok=True,
                source=self.source,
                width=packed.shape[1],
                height=packed.shape[0],
            )
            self._push(frame)

    def _push(self, frame: Frame) -> None:
        try:
            self._q.put_nowait(frame)
        except Full:
            try:
                self._q.get_nowait()
            except Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except Full:
                pass

    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        try:
            return self._q.get(timeout=timeout_s)
        except Empty:
            return None

    def stop(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None


def _copy_bgra(mapinfo, buf, GstVideo, width: int, height: int) -> np.ndarray:
    """Copy mapped GST memory into owned packed (H, W, 4) uint8. Never queue a view."""
    data = np.frombuffer(mapinfo.data, dtype=np.uint8)
    stride = width * 4
    meta = None
    try:
        meta = GstVideo.buffer_get_video_meta(buf)
    except Exception:
        meta = None
    if meta is not None and getattr(meta, "stride", None):
        try:
            stride = int(meta.stride[0])
        except Exception:
            stride = width * 4
    if stride == width * 4 and data.size >= height * stride:
        return np.array(data[: height * stride].reshape(height, width, 4), copy=True)
    out = np.empty((height, width, 4), dtype=np.uint8)
    src = data
    row_bytes = width * 4
    for y in range(height):
        start = y * stride
        out[y].reshape(-1)[:] = src[start : start + row_bytes]
    return out
