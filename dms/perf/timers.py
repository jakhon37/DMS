"""Bounded, allocation-free stage timing for the frame loop.

Why this exists
---------------
The main loop used to keep ``infer_ms = []`` for the lifetime of the process,
append one float per frame and ``sorted()`` the whole list twice a second.  At
the target 15 FPS an eight-hour soak reaches ~432,000 elements: the sort alone
costs several hundred milliseconds and lands on one frame in fifteen, which is a
guaranteed p99 violation that only appears in the soak.  Worse, a lifetime
percentile can never show a regression - it is structurally incapable of
alerting, because one bad hour cannot move a number averaged over eight.

:class:`StageTimer` replaces it with a fixed-size ring buffer per stage.  Memory
is constant, the percentile is over a rolling window, and a span costs about
three microseconds on this board (measured; see ``tests/test_perf.py``, which
fails the build if it regresses past 20 us).

Units
-----
Every duration recorded, stored and reported by this module is **milliseconds**
(float).  ``window`` is a count of samples, not seconds: at 15 FPS the default
300 is a 20 s window.  ``capture_lag`` and similar gauges are recorded through
the same machinery with :meth:`StageTimer.record`; they are ages, not durations,
but the statistics are the same.

Usage
-----
::

    timer = StageTimer(window=300)
    with timer.stage("detect"):
        faces = detector.detect(bgr)
    timer.record("capture_lag", (time.monotonic_ns() - frame.t_mono_ns) / 1e6)
    timer.mark_frame()
    stats = timer.snapshot()          # {"detect": StageStats(...), ...}

Failure behaviour
-----------------
* Re-entering the same stage raises :class:`StageReentryError` immediately
  rather than silently recording a nested, wrong duration.  Two *different*
  stages may nest freely.
* An exception inside a ``with`` block still records the elapsed time and then
  propagates - a stage that raised is still a stage that consumed time, and
  hiding it would make a crash loop look free.
* :meth:`snapshot` of a stage with no samples reports ``count=0`` and ``0.0``
  for every percentile.  It never invents a plausible number.

Thread safety
-------------
A :class:`StageTimer` is **not** thread-safe and deliberately takes no lock:
the frame loop is single-threaded and a lock would double the span cost.  Give
each thread its own timer and merge with :meth:`StageTimer.merge_snapshots`.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Default ring size. 300 samples is 20 s at the 15 FPS freeze target.
DEFAULT_WINDOW = 300

#: Percentiles published by :meth:`StageStats.as_dict`, in order.
PERCENTILES = (50, 95, 99)

_perf_counter = time.perf_counter


class StageReentryError(RuntimeError):
    """The same stage was entered twice without leaving it."""


class StageStats(object):
    """Immutable summary of one stage over the current window.

    All millisecond fields describe the *window* (the last ``window`` samples);
    ``count`` and ``total_count`` differ once the ring has wrapped - ``count``
    is what the percentiles were computed from, ``total_count`` is how many
    samples the stage has ever seen.
    """

    __slots__ = ("name", "count", "total_count", "mean_ms", "max_ms", "last_ms", "pct")

    def __init__(self, name, count, total_count, mean_ms, max_ms, last_ms, pct):
        # type: (str, int, int, float, float, float, Dict[int, float]) -> None
        self.name = name
        self.count = int(count)
        self.total_count = int(total_count)
        self.mean_ms = float(mean_ms)
        self.max_ms = float(max_ms)
        self.last_ms = float(last_ms)
        self.pct = dict(pct)

    @property
    def p50(self):
        # type: () -> float
        return self.pct.get(50, 0.0)

    @property
    def p95(self):
        # type: () -> float
        return self.pct.get(95, 0.0)

    @property
    def p99(self):
        # type: () -> float
        return self.pct.get(99, 0.0)

    def as_dict(self):
        # type: () -> Dict[str, object]
        out = {
            "count": self.count,
            "total_count": self.total_count,
            "mean_ms": round(self.mean_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }
        for p in PERCENTILES:
            out["p%d_ms" % p] = round(self.pct.get(p, 0.0), 3)
        return out

    def __repr__(self):  # pragma: no cover - debugging aid
        return "StageStats(%s n=%d p50=%.2f p95=%.2f max=%.2f)" % (
            self.name,
            self.count,
            self.p50,
            self.p95,
            self.max_ms,
        )


class Stage(object):
    """One named ring buffer, and its own context manager.

    The object *is* the context manager, so ``with timer.stage("x"):`` costs one
    dict lookup and two attribute writes - no per-frame object is created.  That
    is the whole reason this class is not a ``contextlib.contextmanager``, which
    allocates a generator and a frame on every entry.
    """

    __slots__ = ("name", "_buf", "_cap", "_n", "_i", "_open_t", "total_count", "_sum", "_scratch")

    def __init__(self, name, window=DEFAULT_WINDOW):
        # type: (str, int) -> None
        cap = int(window)
        if cap < 1:
            raise ValueError("window must be >= 1, got %r" % (window,))
        self.name = str(name)
        self._cap = cap
        # float64 storage: writing a Python float into a numpy array does not
        # create a PyFloat object, so a steady-state record() allocates nothing.
        self._buf = np.zeros(cap, dtype=np.float64)
        self._scratch = np.empty(cap, dtype=np.float64)
        self._n = 0  # samples currently in the ring (<= cap)
        self._i = 0  # next write index
        self._open_t = -1.0  # perf_counter at __enter__, -1 when not open
        self.total_count = 0
        self._sum = 0.0

    # ------------------------------------------------------------- recording
    def record(self, ms):
        # type: (float) -> None
        """Add one millisecond sample, overwriting the oldest when full."""
        i = self._i
        buf = self._buf
        if self._n == self._cap:
            self._sum -= buf[i]
        else:
            self._n += 1
        buf[i] = ms
        self._sum += ms
        self._i = i + 1 if i + 1 < self._cap else 0
        self.total_count += 1

    def __enter__(self):
        # type: () -> Stage
        if self._open_t >= 0.0:
            raise StageReentryError(
                "stage %r is already open; nesting a stage inside itself would "
                "record the inner span twice and the outer span not at all" % self.name
            )
        self._open_t = _perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (object, object, object) -> bool
        t0 = self._open_t
        self._open_t = -1.0
        if t0 >= 0.0:
            self.record((_perf_counter() - t0) * 1000.0)
        return False  # never swallow

    # ------------------------------------------------------------ statistics
    @property
    def count(self):
        # type: () -> int
        """Samples currently in the window."""
        return self._n

    @property
    def open(self):
        # type: () -> bool
        """True between ``__enter__`` and ``__exit__``."""
        return self._open_t >= 0.0

    def samples(self):
        # type: () -> np.ndarray
        """A copy of the window, oldest-first. For tests and offline analysis."""
        if self._n < self._cap:
            return self._buf[: self._n].copy()
        return np.concatenate((self._buf[self._i :], self._buf[: self._i]))

    def stats(self):
        # type: () -> StageStats
        """Window statistics. Allocation-light: percentiles use a reused scratch."""
        n = self._n
        if n == 0:
            return StageStats(self.name, 0, self.total_count, 0.0, 0.0, 0.0, {p: 0.0 for p in PERCENTILES})
        view = self._scratch[:n]
        view[:] = self._buf[:n] if n < self._cap else self._buf
        last_i = self._i - 1 if self._i > 0 else self._cap - 1
        pct = {}  # type: Dict[int, float]
        for p in PERCENTILES:
            pct[p] = float(_percentile_inplace(view, p))
        return StageStats(
            self.name,
            n,
            self.total_count,
            self._sum / n,
            float(view.max()),
            float(self._buf[last_i]),
            pct,
        )

    def reset(self):
        # type: () -> None
        """Drop the window. ``total_count`` is preserved - it is a lifetime counter."""
        self._n = 0
        self._i = 0
        self._sum = 0.0
        self._open_t = -1.0


def _percentile_inplace(view, p):
    # type: (np.ndarray, int) -> float
    """Nearest-rank percentile via partial selection. Reorders ``view``.

    Nearest-rank (not linear interpolation) is deliberate: a latency budget is
    checked against a sample that actually occurred, and with n=300 the
    interpolated value can sit between two frames that both passed or both
    failed.  ``np.partition`` is O(n), so this stays cheap at 2 Hz.
    """
    n = view.shape[0]
    if n == 1:
        return float(view[0])
    k = int(round((p / 100.0) * (n - 1)))
    if k < 0:
        k = 0
    elif k > n - 1:
        k = n - 1
    view.partition(k)
    return float(view[k])


class StageTimer(object):
    """A named collection of :class:`Stage` ring buffers plus a frame counter."""

    __slots__ = ("_stages", "_window", "frames", "_t_first", "_t_last_frame", "_frames_at_mark",
                 "_t_at_mark")

    def __init__(self, window=DEFAULT_WINDOW, stages=()):
        # type: (int, Iterable[str]) -> None
        if int(window) < 1:
            raise ValueError("window must be >= 1, got %r" % (window,))
        self._window = int(window)
        self._stages = {}  # type: Dict[str, Stage]
        for name in stages:
            self.stage(name)
        self.frames = 0
        self._t_first = None  # type: Optional[float]
        self._t_last_frame = None  # type: Optional[float]
        self._frames_at_mark = 0
        self._t_at_mark = None  # type: Optional[float]

    # ---------------------------------------------------------------- access
    @property
    def window(self):
        # type: () -> int
        return self._window

    def stage(self, name):
        # type: (str) -> Stage
        """The ring for ``name``, created on first use."""
        s = self._stages.get(name)
        if s is None:
            s = Stage(name, self._window)
            self._stages[name] = s
        return s

    def __contains__(self, name):
        # type: (object) -> bool
        return name in self._stages

    def names(self):
        # type: () -> List[str]
        """Stage names in creation order (dicts are insertion-ordered on 3.7+)."""
        return list(self._stages)

    def record(self, name, ms):
        # type: (str, float) -> None
        """Record a millisecond sample for ``name`` without timing a block."""
        self.stage(name).record(ms)

    # ----------------------------------------------------------------- frame
    def mark_frame(self):
        # type: () -> None
        """Count one completed frame. Drives :meth:`fps` and :meth:`rolling_fps`."""
        now = _perf_counter()
        if self._t_first is None:
            self._t_first = now
            self._t_at_mark = now
        self._t_last_frame = now
        self.frames += 1

    def fps(self):
        # type: () -> float
        """Lifetime average FPS since the first frame. 0.0 before two frames."""
        if self._t_first is None or self._t_last_frame is None:
            return 0.0
        dt = self._t_last_frame - self._t_first
        if dt <= 0.0:
            return 0.0
        return (self.frames - 1) / dt

    def rolling_fps(self):
        # type: () -> float
        """FPS since the previous call, then re-arm. Use once per report tick.

        A lifetime average cannot fall: after an hour at 15 FPS, ten minutes at
        3 FPS still reports 12.5.  This is the number that actually moves when
        the pipeline degrades.
        """
        now = _perf_counter()
        t0 = self._t_at_mark
        n0 = self._frames_at_mark
        self._t_at_mark = now
        self._frames_at_mark = self.frames
        if t0 is None:
            return 0.0
        dt = now - t0
        dn = self.frames - n0
        if dt <= 0.0 or dn <= 0:
            return 0.0
        return dn / dt

    # ------------------------------------------------------------- reporting
    def snapshot(self):
        # type: () -> Dict[str, StageStats]
        """Window statistics for every stage that has at least one sample."""
        return dict((n, s.stats()) for n, s in self._stages.items() if s.count > 0)

    def as_dict(self):
        # type: () -> Dict[str, Dict[str, object]]
        """JSON-ready ``{stage: {count, mean_ms, p50_ms, p95_ms, p99_ms, max_ms}}``."""
        return dict((n, st.as_dict()) for n, st in self.snapshot().items())

    def p95_map(self):
        # type: () -> Dict[str, float]
        """``{stage: p95_ms}`` - the shape ``health.latency_ms`` wants."""
        return dict((n, round(st.p95, 1)) for n, st in self.snapshot().items())

    def reset(self):
        # type: () -> None
        """Clear every window. Frame counters and lifetime counts are kept."""
        for s in self._stages.values():
            s.reset()

    @staticmethod
    def merge_snapshots(snapshots):
        # type: (Sequence[Dict[str, StageStats]]) -> Dict[str, Dict[str, object]]
        """Combine per-thread snapshots by taking the worst p95/max per stage.

        Percentiles do not add, so this deliberately reports the worst observer
        rather than an average that would understate a stalled thread.
        """
        out = {}  # type: Dict[str, Dict[str, object]]
        for snap in snapshots:
            for name, st in snap.items():
                cur = out.get(name)
                d = st.as_dict()
                if cur is None:
                    out[name] = d
                    continue
                for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"):
                    if d.get(key, 0.0) > cur.get(key, 0.0):
                        cur[key] = d[key]
                cur["count"] = int(cur.get("count", 0)) + int(d.get("count", 0))
                cur["total_count"] = int(cur.get("total_count", 0)) + int(d.get("total_count", 0))
        return out


class NullStage(object):
    """A stage that measures nothing, for the disabled timer."""

    __slots__ = ("name",)

    def __init__(self, name="null"):
        # type: (str) -> None
        self.name = name

    def __enter__(self):
        # type: () -> NullStage
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (object, object, object) -> bool
        return False

    def record(self, ms):
        # type: (float) -> None
        return None

    @property
    def count(self):
        # type: () -> int
        return 0

    def stats(self):
        # type: () -> StageStats
        return StageStats(self.name, 0, 0, 0.0, 0.0, 0.0, {p: 0.0 for p in PERCENTILES})

    def reset(self):
        # type: () -> None
        return None


class NullTimer(StageTimer):
    """Same interface, no measurement. Costs one attribute lookup per span.

    Use when timing itself must not perturb a measurement (inside ``bench.py``
    micro-benchmarks) or when a caller wants the code path without the data.
    """

    __slots__ = ("_null",)

    def __init__(self):
        # type: () -> None
        StageTimer.__init__(self, window=1)
        self._null = NullStage()

    def stage(self, name):
        # type: (str) -> NullStage
        return self._null

    def record(self, name, ms):
        # type: (str, float) -> None
        return None

    def snapshot(self):
        # type: () -> Dict[str, StageStats]
        return {}


__all__ = [
    "DEFAULT_WINDOW",
    "PERCENTILES",
    "NullStage",
    "NullTimer",
    "Stage",
    "StageReentryError",
    "StageStats",
    "StageTimer",
]
