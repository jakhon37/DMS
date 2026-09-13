"""Capture supervision: keep frames flowing for 24/7 operation, or say loudly that
they are not.

A :class:`CaptureSupervisor` owns one capture source and turns every way it can die -
an exception in the capture thread, a GStreamer ERROR/EOS, an nvargus-daemon restart,
a USB re-enumeration, or a silent stall - into the same three observable things:

1. a ``Frame(ok=False)`` sentinel returned once from :meth:`read`, so the main loop
   cannot mistake a dead camera for a slow one,
2. a ``camera_fail`` enter/exit edge on :meth:`take_events`, for the caller to write to
   the event JSONL as ``AlertType.CAMERA_FAIL``,
3. a reconnect attempt on the specified exponential backoff
   (0.5 s, 1 s, 2 s, 5 s, 5 s), after which ``camera_status`` becomes ``"failed"``.

Units: every duration is seconds of ``time.monotonic``; ``t_mono_ns`` on a Frame is
nanoseconds. Rates are frames per second.

Testability: the supervisor never imports GStreamer. It drives any object with
``start()``, ``read(timeout_s)``, ``stop()``, optional ``restart()`` and optional
``eos`` / ``error`` / ``last_error`` attributes, so the fault-injection tests use a
plain fake. :meth:`for_config` builds the real GstSource factory lazily.

Failure behaviour: nothing here raises during steady-state operation. A source that
raises from ``read``/``start``/``restart`` is counted as a failure and retried; the
exception text is kept in ``last_error`` and logged with a traceback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from collections import deque

from dms.types import Frame, SourceType

log = logging.getLogger("dms.runtime.supervisor")

#: docs/production-dms-design.md: retry pipeline start with this backoff ladder.
DEFAULT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0, 5.0)


class CaptureState(str, Enum):
    """Lifecycle of the supervised source, as reported on /healthz."""

    IDLE = "idle"  # constructed, start() not called yet
    RUNNING = "running"  # frames are flowing (or the first one is still pending)
    RETRYING = "retrying"  # a failure happened, a reconnect is scheduled
    FAILED = "failed"  # the backoff ladder is exhausted
    FINISHED = "finished"  # a file source reached EOS; not an error
    STOPPED = "stopped"  # stop() was called


#: What ``camera_status`` reports for each state, matching the health contract.
_CAMERA_STATUS = {
    CaptureState.IDLE: "starting",
    CaptureState.RUNNING: "ok",
    CaptureState.RETRYING: "retrying",
    CaptureState.FAILED: "failed",
    CaptureState.FINISHED: "finished",
    CaptureState.STOPPED: "stopped",
}


@dataclass
class CaptureEvent:
    """An enter/exit edge the caller should turn into an AlertEvent."""

    kind: str  # "camera_fail" | "pipeline_starved"
    edge: str  # "enter" | "exit"
    t_mono_s: float
    detail: str = ""


class RollingStats:
    """Fixed-window latency/interval statistics with a bounded, constant footprint.

    Percentiles come from the last ``maxlen`` samples only, so a regression in hour 7
    of a soak is actually visible; the lifetime mean is kept with Welford's method so
    no sample is retained for it. Units are whatever the caller adds (this codebase
    uses milliseconds for stage latency and seconds for frame intervals).
    """

    __slots__ = ("_w", "count", "_mean", "_m2", "_max", "_min")

    def __init__(self, maxlen: int = 300) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._w: Deque[float] = deque(maxlen=maxlen)
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._max = float("-inf")
        self._min = float("inf")

    def add(self, x: float) -> None:
        v = float(x)
        self._w.append(v)
        self.count += 1
        delta = v - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (v - self._mean)
        if v > self._max:
            self._max = v
        if v < self._min:
            self._min = v

    def __len__(self) -> int:
        return len(self._w)

    @property
    def window(self) -> List[float]:
        return list(self._w)

    def percentile(self, q: float) -> float:
        """Nearest-rank percentile over the window. q in [0, 1]. 0.0 when empty."""
        if not self._w:
            return 0.0
        xs = sorted(self._w)
        idx = int(round(q * (len(xs) - 1)))
        return xs[max(0, min(len(xs) - 1, idx))]

    @property
    def p50(self) -> float:
        return self.percentile(0.50)

    @property
    def p95(self) -> float:
        return self.percentile(0.95)

    @property
    def p99(self) -> float:
        return self.percentile(0.99)

    @property
    def window_mean(self) -> float:
        return (sum(self._w) / len(self._w)) if self._w else 0.0

    @property
    def lifetime_mean(self) -> float:
        return self._mean if self.count else 0.0

    @property
    def lifetime_max(self) -> float:
        return self._max if self.count else 0.0

    @property
    def lifetime_min(self) -> float:
        return self._min if self.count else 0.0

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Flat dict for /healthz and the Prometheus text exposition."""
        return {
            prefix + "p50": round(self.p50, 3),
            prefix + "p95": round(self.p95, 3),
            prefix + "p99": round(self.p99, 3),
            prefix + "mean": round(self.window_mean, 3),
            prefix + "lifetime_mean": round(self.lifetime_mean, 3),
            prefix + "lifetime_max": round(self.lifetime_max, 3),
            prefix + "count": float(self.count),
        }


class CaptureSupervisor:
    """Own a capture source; restart it on backoff and report honestly meanwhile.

    Parameters
    ----------
    source_factory : callable returning a fresh, unstarted source object.
    backoff_s : reconnect delays, one per consecutive failure.
    retry_forever : when the ladder is exhausted, keep retrying at ``backoff_s[-1]``
        while still reporting ``camera_status == "failed"``. Default False matches the
        design (five attempts, then failed and no further attempts).
    stall_timeout_s : with no frame and no error for this long, treat the source as
        dead. This is the only thing that catches a capture thread that died silently.
    first_frame_timeout_s : deadline for frame #1 after a (re)start. Defaults to
        ``stall_timeout_s``; a corrupt clip or a missing plugin trips it instead of
        hanging forever.
    eos_is_end : True for file replay, where EOS means "clip finished", not "camera
        died". False for csi/usb/test, where EOS is a fault.
    starved_below_fps : emit a ``pipeline_starved`` edge when the measured rate stays
        below this for ``starved_for_s``. 0 disables the check.
    """

    def __init__(
        self,
        source_factory: Callable[[], Any],
        *,
        backoff_s: Sequence[float] = DEFAULT_BACKOFF_S,
        retry_forever: bool = False,
        stall_timeout_s: float = 5.0,
        first_frame_timeout_s: Optional[float] = None,
        eos_is_end: bool = False,
        starved_below_fps: float = 0.0,
        starved_for_s: float = 10.0,
        source_type: SourceType = SourceType.FILE,
        frame_wh: Sequence[int] = (1280, 720),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not backoff_s:
            raise ValueError("backoff_s must not be empty")
        if stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be > 0")
        self._factory = source_factory
        self._backoff = tuple(float(x) for x in backoff_s)
        self._retry_forever = bool(retry_forever)
        self._stall_timeout_s = float(stall_timeout_s)
        self._first_frame_timeout_s = float(
            stall_timeout_s if first_frame_timeout_s is None else first_frame_timeout_s
        )
        self._eos_is_end = bool(eos_is_end)
        self._starved_below_fps = float(starved_below_fps)
        self._starved_for_s = float(starved_for_s)
        self._source_type = source_type
        self._frame_w, self._frame_h = int(frame_wh[0]), int(frame_wh[1])
        self._clock = clock
        self._sleep = sleep

        self._source: Optional[Any] = None
        self._state = CaptureState.IDLE
        self._events: List[CaptureEvent] = []
        self._pending_sentinel: Optional[Frame] = None

        self._consecutive_failures = 0
        self.failures_total = 0
        self.restarts = 0
        self.frames_total = 0
        self.sentinels_emitted = 0
        self.last_error = ""

        self._next_attempt_s = 0.0
        self._last_frame_s = 0.0
        self._started_s = 0.0
        self._seen_first_frame = False
        self._camera_fail_active = False
        self._starved_active = False
        self._starved_since_s = 0.0
        self._id_offset = 0
        self._last_out_id = 0
        self.intervals = RollingStats(maxlen=300)

    # ---------------------------------------------------------------- factories

    @classmethod
    def for_config(cls, cfg, **kwargs) -> "CaptureSupervisor":
        """Supervise a real :class:`dms.capture.GstSource` built from ``cfg``.

        GStreamer is imported only when the factory is actually called, so importing
        this module stays free on a machine without gi.
        """

        def _make():
            from dms.capture.gst_source import GstSource

            return GstSource(cfg)

        kwargs.setdefault("eos_is_end", cfg.source.type == "file")
        kwargs.setdefault("source_type", SourceType(cfg.source.type))
        kwargs.setdefault("frame_wh", (cfg.source.width, cfg.source.height))
        return cls(_make, **kwargs)

    # --------------------------------------------------------------- properties

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def camera_status(self) -> str:
        """The string /healthz publishes: starting | ok | retrying | failed | ..."""
        return _CAMERA_STATUS[self._state]

    @property
    def healthy(self) -> bool:
        """True only while frames are actually flowing."""
        return self._state == CaptureState.RUNNING and self._seen_first_frame

    @property
    def finished(self) -> bool:
        """A file source reached EOS - a clean end of replay, not a fault."""
        return self._state == CaptureState.FINISHED

    @property
    def failed(self) -> bool:
        return self._state == CaptureState.FAILED

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def camera_fail_active(self) -> bool:
        return self._camera_fail_active

    @property
    def last_frame_mono_s(self) -> float:
        return self._last_frame_s

    @property
    def frames_dropped(self) -> int:
        """Frames the source threw away because the consumer was behind."""
        return int(getattr(self._source, "dropped", 0) or 0)

    @property
    def fps(self) -> float:
        """Rolling frame rate over the interval window. 0.0 before two frames."""
        m = self.intervals.window_mean
        return (1.0 / m) if m > 0 else 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Everything the health endpoint needs, in one consistent read."""
        return {
            "state": self._state.value,
            "camera": self.camera_status,
            "fps": round(self.fps, 2),
            "frames_total": self.frames_total,
            "frames_dropped": self.frames_dropped,
            "capture_restarts": self.restarts,
            "capture_failures": self.failures_total,
            "consecutive_failures": self._consecutive_failures,
            "last_frame_age_s": round(max(0.0, self._clock() - self._last_frame_s), 2)
            if self._last_frame_s
            else None,
            "camera_fail_active": self._camera_fail_active,
            "pipeline_starved": self._starved_active,
            "last_error": self.last_error,
        }

    def take_events(self) -> List[CaptureEvent]:
        """Drain the pending enter/exit edges. Call once per main-loop iteration."""
        evs = self._events
        self._events = []
        return evs

    # ------------------------------------------------------------------ control

    def start(self) -> bool:
        """Start the source. Returns True on success; a failure schedules a retry."""
        if self._state not in (CaptureState.IDLE, CaptureState.STOPPED):
            return self._state == CaptureState.RUNNING
        self._state = CaptureState.IDLE
        return self._open(initial=True)

    def _open(self, initial: bool = False) -> bool:
        now = self._clock()
        try:
            src = self._factory()
            src.start()
        except Exception as exc:
            log.error("capture start failed: %s", exc, exc_info=True)
            self._source = None
            self._fail("start failed: %r" % (exc,), now)
            return False
        self._source = src
        self._state = CaptureState.RUNNING
        self._started_s = now
        self._last_frame_s = now
        self._seen_first_frame = False
        if not initial:
            self.restarts += 1
            log.warning("capture restarted (restart #%d)", self.restarts)
        return True

    def stop(self) -> None:
        """Tear the source down. Idempotent."""
        src = self._source
        self._source = None
        self._state = CaptureState.STOPPED
        if src is not None:
            try:
                src.stop()
            except Exception as exc:
                log.error("capture stop failed: %s", exc, exc_info=True)

    def reset(self) -> None:
        """Clear a FAILED state so the caller can attempt recovery manually."""
        if self._state == CaptureState.FAILED:
            self._consecutive_failures = 0
            self._state = CaptureState.RETRYING
            self._next_attempt_s = self._clock()

    # --------------------------------------------------------------------- read

    def read(self, timeout_s: float = 1.0) -> Optional[Frame]:
        """Return the next frame, a one-shot ``Frame(ok=False)`` sentinel, or None.

        Never blocks longer than ``timeout_s``. A None means "nothing this tick" and
        is not by itself a fault; consult :attr:`state` / :meth:`take_events`.
        """
        sentinel = self._pending_sentinel
        if sentinel is not None:
            self._pending_sentinel = None
            self.sentinels_emitted += 1
            return sentinel

        state = self._state
        if state in (CaptureState.STOPPED, CaptureState.FINISHED):
            return None
        if state == CaptureState.IDLE:
            if not self.start():
                return self._take_sentinel()
            state = self._state
        if state == CaptureState.FAILED:
            if not self._retry_forever:
                self._nap(timeout_s)
                return None
            state = CaptureState.RETRYING
        if state == CaptureState.RETRYING:
            now = self._clock()
            if now < self._next_attempt_s:
                self._nap(min(timeout_s, self._next_attempt_s - now))
                return None
            self._reopen()
            return self._take_sentinel()

        src = self._source
        if src is None:
            self._fail("source vanished", self._clock())
            return self._take_sentinel()
        try:
            frame = src.read(timeout_s=timeout_s)
        except Exception as exc:
            log.error("capture read raised: %s", exc, exc_info=True)
            self._fail("read raised: %r" % (exc,), self._clock())
            return self._take_sentinel()

        now = self._clock()
        if frame is not None and getattr(frame, "ok", True):
            self._on_frame(frame, now)
            return frame
        if frame is not None:
            # The source produced its own failure sentinel.
            self._fail(str(getattr(src, "last_error", "") or "source sentinel"), now)
            return self._take_sentinel()

        if getattr(src, "error", False):
            self._fail(str(getattr(src, "last_error", "") or "capture error"), now)
            return self._take_sentinel()
        if getattr(src, "eos", False):
            if self._eos_is_end:
                log.info("capture reached EOS after %d frames", self.frames_total)
                self._state = CaptureState.FINISHED
                return None
            self._fail("unexpected EOS on a live source", now)
            return self._take_sentinel()

        deadline = self._first_frame_timeout_s if not self._seen_first_frame else self._stall_timeout_s
        if now - self._last_frame_s > deadline:
            self._fail(
                "no frame for %.1f s (%s)"
                % (now - self._last_frame_s, "first frame" if not self._seen_first_frame else "stall"),
                now,
            )
            return self._take_sentinel()
        return None

    # ------------------------------------------------------------------ internals

    def _nap(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(min(seconds, 1.0))

    def _take_sentinel(self) -> Optional[Frame]:
        s = self._pending_sentinel
        if s is None:
            return None
        self._pending_sentinel = None
        self.sentinels_emitted += 1
        return s

    def _on_frame(self, frame: Frame, now: float) -> None:
        if self._last_frame_s and self._seen_first_frame:
            self.intervals.add(max(0.0, now - self._last_frame_s))
        self._last_frame_s = now
        self._seen_first_frame = True
        self.frames_total += 1
        self._state = CaptureState.RUNNING
        # Frame ids must stay monotonic across restarts: the watchdog gate in the app
        # only pings systemd when the id advanced.
        out_id = int(frame.frame_id) + self._id_offset
        if out_id <= self._last_out_id:
            self._id_offset += self._last_out_id - out_id + 1
            out_id = int(frame.frame_id) + self._id_offset
        frame.frame_id = out_id
        self._last_out_id = out_id
        if self._consecutive_failures or self._camera_fail_active:
            self._consecutive_failures = 0
            if self._camera_fail_active:
                self._camera_fail_active = False
                self._events.append(
                    CaptureEvent("camera_fail", "exit", now, "frames resumed after %d restart(s)" % self.restarts)
                )
                log.warning("camera recovered after %d restart(s)", self.restarts)
        self._check_starvation(now)

    def _check_starvation(self, now: float) -> None:
        if self._starved_below_fps <= 0 or len(self.intervals) < 5:
            return
        slow = self.fps < self._starved_below_fps
        if slow:
            if self._starved_since_s == 0.0:
                self._starved_since_s = now
            elif not self._starved_active and now - self._starved_since_s >= self._starved_for_s:
                self._starved_active = True
                self._events.append(
                    CaptureEvent(
                        "pipeline_starved",
                        "enter",
                        now,
                        "%.1f fps < %.1f for %.0f s" % (self.fps, self._starved_below_fps, self._starved_for_s),
                    )
                )
        else:
            self._starved_since_s = 0.0
            if self._starved_active:
                self._starved_active = False
                self._events.append(
                    CaptureEvent("pipeline_starved", "exit", now, "%.1f fps" % self.fps)
                )

    def _fail(self, detail: str, now: float) -> None:
        self._consecutive_failures += 1
        self.failures_total += 1
        self.last_error = detail
        if not self._camera_fail_active:
            self._camera_fail_active = True
            self._events.append(CaptureEvent("camera_fail", "enter", now, detail))
        self._pending_sentinel = self._make_sentinel(now)
        if self._consecutive_failures <= len(self._backoff):
            delay = self._backoff[self._consecutive_failures - 1]
            self._state = CaptureState.RETRYING
            self._next_attempt_s = now + delay
            log.error(
                "CAMERA_FAIL %s - reconnect attempt %d/%d in %.1f s",
                detail,
                self._consecutive_failures,
                len(self._backoff),
                delay,
            )
        else:
            self._state = CaptureState.FAILED
            self._next_attempt_s = now + self._backoff[-1]
            log.error(
                "CAMERA_FAIL %s - %d consecutive failures, camera=failed%s",
                detail,
                self._consecutive_failures,
                " (still retrying)" if self._retry_forever else "",
            )

    def _make_sentinel(self, now: float) -> Frame:
        return Frame(
            frame_id=self._last_out_id,
            t_mono_ns=int(now * 1e9),
            full_bgra=None,
            ok=False,
            source=self._source_type,
            width=self._frame_w,
            height=self._frame_h,
        )

    def _reopen(self) -> None:
        """Tear the source to NULL and bring it back. Schedules the next retry on failure."""
        now = self._clock()
        src = self._source
        self._id_offset = self._last_out_id
        if src is not None and hasattr(src, "restart"):
            try:
                src.restart()
            except Exception as exc:
                log.error("capture restart failed: %s", exc, exc_info=True)
                try:
                    src.stop()
                except Exception:
                    log.error("capture stop after failed restart also failed", exc_info=True)
                self._source = None
                self._fail("restart failed: %r" % (exc,), now)
                return
            self.restarts += 1
            self._state = CaptureState.RUNNING
            self._started_s = now
            self._last_frame_s = now
            self._seen_first_frame = False
            log.warning("capture restarted in place (restart #%d)", self.restarts)
            return
        if src is not None:
            try:
                src.stop()
            except Exception:
                log.error("capture stop before reopen failed", exc_info=True)
            self._source = None
        self._open(initial=False)


__all__ = [
    "DEFAULT_BACKOFF_S",
    "CaptureEvent",
    "CaptureState",
    "CaptureSupervisor",
    "RollingStats",
]
