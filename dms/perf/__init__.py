"""Performance instrumentation: bounded stage timers and the latency budget.

Two modules, one job - make the freeze gate measurable:

``dms.perf.timers``
    :class:`~dms.perf.timers.StageTimer`, a fixed-memory ring buffer per stage
    with rolling p50/p95/p99.  Replaces the unbounded ``infer_ms`` list that grew
    to ~432,000 floats over an eight-hour soak and was fully sorted twice a
    second.

``dms.perf.budget``
    :class:`~dms.perf.budget.LatencyBudget`, the per-stage target table and the
    writer for ``latency_budget.json`` - the freeze deliverable that names which
    stage blew its budget.

Importing this package pulls in numpy and nothing else: no OpenCV, no TensorRT,
no CUDA.  It is safe to import in a unit test on a machine with no GPU, which is
the point - the budget rules are testable without the board.

Units: milliseconds for latency, MiB for memory, MHz for clocks.
"""
from __future__ import annotations

from typing import Optional

from dms.perf.budget import (
    E2E,
    GATE_E2E_P95_MS,
    GATE_PEAK_RSS_MB,
    STAGES,
    LatencyBudget,
    StageSpec,
    default_targets,
    stage_names,
)
from dms.perf.timers import (
    DEFAULT_WINDOW,
    NullTimer,
    Stage,
    StageReentryError,
    StageStats,
    StageTimer,
)


def sample_board(budget, throttled=False):
    # type: (LatencyBudget, bool) -> None
    """Fold one board reading (RSS, MemAvailable, GPU clock, GPU temp) into ``budget``.

    Call at the same cadence as the periodic health update - roughly twice a
    second is plenty; every reader is a small sysfs/procfs read.  Every value is
    optional: :mod:`dms.io.thermal` returns ``None`` for an unreadable node and
    this function forwards that ``None`` rather than substituting a plausible
    constant, so a missing sensor shows up as a null in ``latency_budget.json``
    instead of a comforting number.

    Failure behaviour: an ImportError (thermal unavailable off-board) is
    propagated - this is a diagnostic helper, not a hot-path call, and silently
    doing nothing would leave the resource block empty with no explanation.
    """
    from dms.io.thermal import gpu_clock_mhz, mem_available_mb, ram_used_mb, read_temps

    budget.observe_rss(ram_used_mb())
    budget.observe_mem_available(mem_available_mb())
    budget.observe_gpu_clock(gpu_clock_mhz())
    gpu_c, _cpu_c = read_temps()
    budget.observe_temp(gpu_c, throttled=throttled)


def peak_rss_mb():
    # type: () -> Optional[int]
    """Kernel high-water RSS (``VmHWM``) in MiB - accurate without polling.

    ``ram_used_mb()`` samples ``VmRSS`` at the instant it is called and will miss
    a spike between samples.  ``VmHWM`` is the kernel's own maximum since process
    start, so it is the honest number for a soak report.  Returns ``None`` when
    ``/proc/self/status`` is unreadable.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


__all__ = [
    "DEFAULT_WINDOW",
    "E2E",
    "GATE_E2E_P95_MS",
    "GATE_PEAK_RSS_MB",
    "STAGES",
    "LatencyBudget",
    "NullTimer",
    "Stage",
    "StageReentryError",
    "StageSpec",
    "StageStats",
    "StageTimer",
    "default_targets",
    "peak_rss_mb",
    "sample_board",
    "stage_names",
]
