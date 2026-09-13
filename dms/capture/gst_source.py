"""GStreamer capture: one 720p BGRx appsink, hardened for 24/7 operation.

Design points that matter for a soak:

* The capture thread body is wrapped in ``try/except BaseException``. Anything that
  escapes - a malformed buffer, an NVDEC hiccup, a PyGI type error - is logged with a
  traceback, sets :attr:`error`, and pushes a ``Frame(ok=False)`` sentinel so the
  consumer learns about it on the very next ``read`` instead of spinning at 1 Hz
  forever against a dead pipeline.
* The bus is drained on **every** iteration, not only when a sample times out. On a
  healthy 30 fps pipeline ``try-pull-sample`` always returns a sample, so the old
  code never popped the bus and every STATE_CHANGED / STREAM_STATUS / QOS / element
  message from ``nvarguscamerasrc`` and ``nvv4l2decoder`` accumulated in native
  memory for the life of the process.
* :meth:`restart` genuinely returns the pipeline to ``Gst.State.NULL`` and rebuilds
  it, clearing the stop/eos/error flags, so the supervisor can reconnect in place.
* Buffer PTS is captured alongside the arrival time. ``t_mono_ns`` is when this thread
  got round to the buffer (use it for latency accounting only); ``pts_ns`` is the
  media timestamp and is what drowsiness durations must be measured against.

Units: ``t_mono_ns`` and ``pts_ns`` are nanoseconds; timeouts are seconds.

Failure behaviour: nothing here raises out of the capture thread. :meth:`start` raises
``RuntimeError`` if the pipeline cannot be parsed or reach PLAYING - that is a
configuration error the caller must see.
"""
from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np

from dms.capture.pipelines import build_pipeline
from dms.config.schema import AppConfig
from dms.types import Frame, SourceType

log = logging.getLogger("dms.capture")

#: Gst.CLOCK_TIME_NONE - an unset PTS.
CLOCK_TIME_NONE = 0xFFFFFFFFFFFFFFFF

#: try-pull-sample timeout. Also sets how often the bus is drained (10 Hz).
_PULL_TIMEOUT_MS = 100


def frame_pts_ns(frame: Frame) -> int:
    """Media timestamp of a frame in nanoseconds, falling back to arrival time.

    ``pts_ns`` is attached by this module; :class:`dms.types.Frame` does not declare
    it yet (see the handoff note). Always read it through this helper so the fallback
    is consistent.
    """
    pts = getattr(frame, "pts_ns", None)
    if pts is None or pts <= 0:
        return int(frame.t_mono_ns)
    return int(pts)


def frame_bgr(frame: Frame) -> Optional[np.ndarray]:
    """Contiguous BGR view of a frame.

    Returns the single de-interleaved copy made in the capture thread when available,
    otherwise falls back to the stride-4 ``full_bgra[..., :3]`` view (which every
    OpenCV call downstream has to materialise again).
    """
    bgr = getattr(frame, "bgr", None)
    if bgr is not None:
        return bgr
    if frame.full_bgra is None:
        return None
    return frame.full_bgra[..., :3]


class GstSource:
    """One 720p BGRx appsink. Copies packed BGRx before unmap; never queues a view."""

    def __init__(self, cfg: AppConfig, emit_bgr: bool = True) -> None:
        self._cfg = cfg
        self._q: Queue = Queue(maxsize=max(1, cfg.capture.queue_depth))
        self._stop = threading.Event()
        self._eos = threading.Event()
        self._error = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pipeline = None
        self._frame_id = 0
        self._emit_bgr = bool(emit_bgr)
        self._lifecycle = threading.Lock()
        self.source = SourceType(cfg.source.type)
        self.last_error = ""
        self.dropped = 0
        self.pushed = 0
        self.bus_warnings = 0
        self.starts = 0

    # --------------------------------------------------------------- properties

    @property
    def eos(self) -> bool:
        return self._eos.is_set()

    @property
    def error(self) -> bool:
        """True once the capture thread or the pipeline reported a fault."""
        return self._error.is_set()

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and not self._stop.is_set()

    # ------------------------------------------------------------------ control

    def start(self) -> None:
        """Build the pipeline, go to PLAYING and spawn the capture thread.

        Clears the stop/eos/error flags first so an instance can be started again
        after :meth:`stop`.
        """
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("GstSource is already running")
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            self._stop.clear()
            self._eos.clear()
            self._error.clear()
            self.last_error = ""
            # A stopped instance can still hold pre-fault frames; never hand one
            # to the consumer after a reconnect.
            self._drain_queue()
            desc = build_pipeline(self._cfg)
            log.info("gst pipeline: %s", desc)
            try:
                pipeline = Gst.parse_launch(desc)
            except Exception as exc:
                raise RuntimeError("failed to parse GStreamer pipeline: %s" % exc)
            if pipeline is None:
                raise RuntimeError("failed to parse GStreamer pipeline")
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                raise RuntimeError("GStreamer failed to PLAYING")
            self._pipeline = pipeline
            self.starts += 1
            self._thread = threading.Thread(target=self._loop, name="gst-capture", daemon=True)
            self._thread.start()

    def restart(self) -> None:
        """Tear the pipeline down to NULL and bring a fresh one up.

        Drains the frame queue so the consumer cannot be handed a pre-fault frame
        after the reconnect. Raises whatever :meth:`start` raises.
        """
        self.stop()
        self._drain_queue()
        self._frame_id = 0
        self.start()

    def stop(self) -> None:
        """Stop the thread and return the pipeline to NULL. Idempotent."""
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
            self._thread = None
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    log.error("capture thread did not exit within 3 s")
            pipeline = self._pipeline
            self._pipeline = None
            if pipeline is not None:
                try:
                    import gi

                    gi.require_version("Gst", "1.0")
                    from gi.repository import Gst

                    bus = pipeline.get_bus()
                    if bus is not None:
                        # Drop anything still queued so it cannot outlive the pipeline.
                        bus.set_flushing(True)
                    pipeline.set_state(Gst.State.NULL)
                    pipeline.get_state(Gst.SECOND)
                except Exception:
                    log.error("failed to tear pipeline down to NULL", exc_info=True)

    # --------------------------------------------------------------------- read

    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        """Next frame, a ``Frame(ok=False)`` sentinel after a fault, or None on timeout."""
        try:
            return self._q.get(timeout=timeout_s)
        except Empty:
            return None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except Empty:
                return

    # --------------------------------------------------------------- thread body

    def _loop(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 - a dead capture thread must be loud
            self.last_error = "capture thread crashed: %r" % (exc,)
            log.error("capture thread crashed", exc_info=True)
            self._error.set()
            self._eos.set()
        finally:
            self._push_sentinel()

    def _run(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo

        pipeline = self._pipeline
        if pipeline is None:
            self.last_error = "pipeline is not built"
            self._error.set()
            self._eos.set()
            return
        sink = pipeline.get_by_name("full")
        if sink is None:
            self.last_error = "appsink 'full' missing from the pipeline"
            log.error("%s", self.last_error)
            self._error.set()
            self._eos.set()
            return
        w, h = self._cfg.source.width, self._cfg.source.height
        bus = pipeline.get_bus()

        while not self._stop.is_set():
            # Drain the bus every iteration - not only when a sample times out.
            if bus is not None and self._pump_bus(bus, Gst):
                break
            sample = sink.emit("try-pull-sample", _PULL_TIMEOUT_MS * Gst.MSECOND)
            if sample is None:
                # EOS is only final once the appsink has also run dry; breaking on the
                # bus message alone would discard up to max-buffers frames of a clip.
                if self._eos.is_set():
                    break
                continue
            buf = sample.get_buffer()
            if buf is None:
                continue
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.last_error = "gst buffer map failed"
                log.error("%s", self.last_error)
                self._error.set()
                break
            try:
                packed = _copy_bgra(mapinfo, buf, GstVideo, w, h)
            finally:
                buf.unmap(mapinfo)
            pts = int(buf.pts) if buf.pts is not None else CLOCK_TIME_NONE
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
            # Extra channels dms.types.Frame does not declare yet; read them through
            # frame_pts_ns() / frame_bgr() so the fallback stays in one place.
            frame.pts_ns = 0 if pts == CLOCK_TIME_NONE else pts
            if self._emit_bgr:
                frame.bgr = cv2.cvtColor(packed, cv2.COLOR_BGRA2BGR)
            self._push(frame)

    def _pump_bus(self, bus, Gst) -> bool:
        """Drain every pending bus message. Returns True when the loop must stop now.

        An ERROR stops immediately. An EOS only records the flag: the appsink may
        still hold buffered samples, and the pull loop breaks once it runs dry.
        """
        fatal = False
        while True:
            msg = bus.pop()
            if msg is None:
                return fatal
            mtype = msg.type
            if mtype == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                self.last_error = "gst error: %s (%s)" % (err, debug)
                log.error("%s", self.last_error)
                self._error.set()
                self._eos.set()
                fatal = True
            elif mtype == Gst.MessageType.EOS:
                log.info("gst EOS after %d frames", self._frame_id)
                self._eos.set()
            elif mtype == Gst.MessageType.WARNING:
                warn, debug = msg.parse_warning()
                self.bus_warnings += 1
                # nvv4l2decoder is chatty; log the first few then sample.
                if self.bus_warnings <= 10 or self.bus_warnings % 100 == 0:
                    log.warning("gst warning #%d: %s (%s)", self.bus_warnings, warn, debug)
            # Every other message type (STATE_CHANGED, STREAM_STATUS, QOS, TAG,
            # LATENCY, ELEMENT...) is dropped here, which is the point.

    def _push(self, frame: Frame) -> None:
        """File replay blocks so no frame is lost; live sources drop the oldest."""
        if self._cfg.source.type == "file":
            while not self._stop.is_set():
                try:
                    self._q.put(frame, timeout=0.2)
                    self.pushed += 1
                    return
                except Full:
                    continue
            return
        try:
            self._q.put_nowait(frame)
            self.pushed += 1
            return
        except Full:
            pass
        try:
            self._q.get_nowait()
            self.dropped += 1
        except Empty:
            pass
        try:
            self._q.put_nowait(frame)
            self.pushed += 1
        except Full:
            self.dropped += 1

    def _push_sentinel(self) -> None:
        """Wake a blocked reader with Frame(ok=False) when the thread died on a fault.

        A clean stop or a file EOS is not a fault and pushes nothing - those are
        reported through :attr:`eos` and the absence of further frames.
        """
        if not self._error.is_set():
            return
        sentinel = Frame(
            frame_id=self._frame_id,
            t_mono_ns=time.monotonic_ns(),
            full_bgra=None,
            ok=False,
            source=self.source,
            width=self._cfg.source.width,
            height=self._cfg.source.height,
        )
        sentinel.pts_ns = 0
        try:
            self._q.put_nowait(sentinel)
            return
        except Full:
            pass
        try:
            self._q.get_nowait()
        except Empty:
            pass
        try:
            self._q.put_nowait(sentinel)
        except Full:
            log.error("could not enqueue the capture failure sentinel")


def _copy_bgra(mapinfo, buf, GstVideo, width: int, height: int) -> np.ndarray:
    """Copy mapped GST memory into an owned packed (H, W, 4) uint8 array.

    Honours the video meta stride when the sink hands back padded rows. Raises
    ``ValueError`` on a short buffer rather than reading past the mapping.
    """
    data = np.frombuffer(mapinfo.data, dtype=np.uint8)
    row_bytes = width * 4
    stride = row_bytes
    meta = None
    try:
        meta = GstVideo.buffer_get_video_meta(buf)
    except Exception:  # no video meta on this buffer; the default stride applies
        meta = None
    if meta is not None and getattr(meta, "stride", None):
        try:
            stride = int(meta.stride[0])
        except (IndexError, TypeError, ValueError):
            stride = row_bytes
    if stride < row_bytes:
        raise ValueError("gst stride %d is smaller than a %dpx BGRx row" % (stride, width))
    need = height * stride
    if data.size < need:
        raise ValueError(
            "gst buffer is %d bytes, need %d for %dx%d stride=%d"
            % (data.size, need, width, height, stride)
        )
    rows = data[:need].reshape(height, stride)
    if stride == row_bytes:
        return np.array(rows.reshape(height, width, 4), copy=True)
    return np.ascontiguousarray(rows[:, :row_bytes].reshape(height, width, 4))
