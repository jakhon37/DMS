"""In-process metric registry rendered as Prometheus text (exposition format 0.0.4).

Why this exists
---------------
A fleet cannot be operated on ``fps`` and RSS. The two questions an operator asks
after an incident are *was this unit observing the driver at all* and *which build
was on it*, and neither is answerable from a gauge of frames per second. This module
owns the counters that answer them: alert edges by type, engine errors by engine,
event-write failures, camera restarts, dropped frames, and a ``dms_build_info``
identity series.

Design
------
Deliberately dependency-free (stdlib only) and deliberately tiny. There is no
``prometheus_client`` on the board and the project forbids adding pip packages, so
this is a hand-rolled registry with exactly the three sample types the endpoint
needs: ``counter`` (monotonic, only ever ``inc``), ``gauge`` (set/inc/dec) and the
``info``-style gauge that carries build identity in its labels and the constant
value 1.

Units
-----
Every series name carries its unit as a suffix, as Prometheus requires:
``_seconds``, ``_ms``, ``_mb``, ``_bytes``, ``_total`` for counters. Temperatures are
degrees Celsius, latencies milliseconds, sizes mebibytes (MiB) where the name says
``_mb`` and bytes where it says ``_bytes``.

Concurrency
-----------
Every mutation and the whole of :meth:`Registry.render` take one registry-wide
:class:`threading.Lock`. The HTTP handler thread scrapes while the pipeline thread
counts, so an unlocked read could observe a half-written label map. The lock is held
for microseconds; it has never been on a latency path.

Failure behaviour
-----------------
Nothing in this module raises during steady-state operation. Programming errors
raise at *registration* time (a bad metric name, a duplicate name with a different
type or label set) so they surface on the first import, never mid-flight. Passing an
unknown label to ``inc``/``set`` raises :class:`ValueError` — that is a code bug, not
a runtime condition, and a silently-dropped sample would be worse. Counters refuse a
negative increment for the same reason.
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("dms.metrics")

#: Prometheus metric-name grammar.
_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
#: Prometheus label-name grammar (``__``-prefixed names are reserved).
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_LabelKey = Tuple[str, ...]


def _fmt(value):
    # type: (float) -> str
    """Render one sample value in the exposition format.

    NaN and the infinities have spellings of their own; everything else goes out as
    a plain float so a scraper never has to parse an integer-vs-float distinction.
    """
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return "%d" % int(v)
    return repr(v)


def _escape(value):
    # type: (str) -> str
    """Escape a label *value*: backslash, double quote and newline, in that order."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Metric(object):
    """One named series family. Subclassed by :class:`Counter` and :class:`Gauge`."""

    kind = "untyped"

    def __init__(self, name, documentation, labelnames, registry):
        # type: (str, str, Sequence[str], "Registry") -> None
        if not _NAME_RE.match(name):
            raise ValueError("illegal metric name %r" % (name,))
        for label in labelnames:
            if not _LABEL_RE.match(label) or label.startswith("__"):
                raise ValueError("illegal label name %r on %s" % (label, name))
        if len(set(labelnames)) != len(labelnames):
            raise ValueError("duplicate label name on %s" % name)
        self.name = name
        self.documentation = documentation or name
        self.labelnames = tuple(labelnames)
        self._registry = registry
        self._values = {}  # type: Dict[_LabelKey, float]
        if not self.labelnames:
            # An unlabelled series must exist at zero from the first scrape, so a
            # scraper can tell "nothing has happened yet" from "this unit does not
            # report that at all".
            self._values[()] = 0.0

    # -- internals -------------------------------------------------------------

    def _key(self, labels):
        # type: (Dict[str, Any]) -> _LabelKey
        if set(labels) != set(self.labelnames):
            raise ValueError(
                "%s expects labels %s, got %s"
                % (self.name, list(self.labelnames), sorted(labels))
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def samples(self):
        # type: () -> List[Tuple[_LabelKey, float]]
        """Snapshot of every series in this family. Caller must hold the registry lock."""
        return sorted(self._values.items())

    def value(self, **labels):
        # type: (**Any) -> float
        """Current value of one series (0.0 if it has never been touched)."""
        key = self._key(labels)
        with self._registry.lock:
            return self._values.get(key, 0.0)

    def clear(self):
        # type: () -> None
        """Drop every series in this family. Tests only."""
        with self._registry.lock:
            self._values.clear()
            if not self.labelnames:
                self._values[()] = 0.0


class Counter(_Metric):
    """Monotonically increasing total. Resets only when the process restarts."""

    kind = "counter"

    def inc(self, amount=1.0, **labels):
        # type: (float, **Any) -> None
        """Add *amount* (must be >= 0) to one series."""
        if amount < 0:
            raise ValueError("counter %s cannot decrease" % self.name)
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)


class Gauge(_Metric):
    """A value that goes up and down: a temperature, a queue depth, a boolean 0/1."""

    kind = "gauge"

    def set(self, value, **labels):
        # type: (float, **Any) -> None
        """Replace one series' value."""
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = float(value)

    def set_bool(self, flag, **labels):
        # type: (Any, **Any) -> None
        """Publish a boolean as 1/0, the Prometheus convention for state."""
        self.set(1.0 if flag else 0.0, **labels)

    def inc(self, amount=1.0, **labels):
        # type: (float, **Any) -> None
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount=1.0, **labels):
        # type: (float, **Any) -> None
        self.inc(-amount, **labels)

    def set_to_current_time(self, **labels):
        # type: (**Any) -> None
        self.set(time.time(), **labels)


class Registry(object):
    """A named collection of metric families with one lock and one text rendering."""

    def __init__(self):
        # type: () -> None
        self.lock = threading.RLock()
        self._metrics = {}  # type: Dict[str, _Metric]

    def _make(self, cls, name, documentation, labelnames):
        # type: (type, str, str, Sequence[str]) -> Any
        with self.lock:
            existing = self._metrics.get(name)
            if existing is not None:
                # Re-importing a module must not explode; a genuine collision must.
                if not isinstance(existing, cls) or existing.labelnames != tuple(labelnames):
                    raise ValueError(
                        "metric %s already registered as %s%s"
                        % (name, existing.kind, list(existing.labelnames))
                    )
                return existing
            metric = cls(name, documentation, labelnames, self)
            self._metrics[name] = metric
            return metric

    def counter(self, name, documentation, labelnames=()):
        # type: (str, str, Sequence[str]) -> Counter
        return self._make(Counter, name, documentation, labelnames)

    def gauge(self, name, documentation, labelnames=()):
        # type: (str, str, Sequence[str]) -> Gauge
        return self._make(Gauge, name, documentation, labelnames)

    def get(self, name):
        # type: (str) -> Optional[_Metric]
        with self.lock:
            return self._metrics.get(name)

    def names(self):
        # type: () -> List[str]
        with self.lock:
            return sorted(self._metrics)

    def reset(self):
        # type: () -> None
        """Zero every series without unregistering anything. Tests only."""
        with self.lock:
            for metric in self._metrics.values():
                metric.clear()

    def render(self):
        # type: () -> str
        """The whole registry as Prometheus text, families in stable name order.

        Every registered family contributes its ``# HELP``/``# TYPE`` header even
        when it has no samples yet. Always ends with a newline; an exposition body
        that does not is rejected by some scrapers.
        """
        out = []  # type: List[str]
        with self.lock:
            for name in sorted(self._metrics):
                metric = self._metrics[name]
                samples = metric.samples()
                # HELP/TYPE are emitted even when a labelled family has no samples
                # yet: a scrape must let an operator tell "this unit reports no
                # engine errors" from "this unit does not report engine errors".
                out.append("# HELP %s %s" % (name, metric.documentation.replace("\n", " ")))
                out.append("# TYPE %s %s" % (name, metric.kind))
                for key, value in samples:
                    if key:
                        labels = ",".join(
                            '%s="%s"' % (ln, _escape(lv))
                            for ln, lv in zip(metric.labelnames, key)
                        )
                        out.append("%s{%s} %s" % (name, labels, _fmt(value)))
                    else:
                        out.append("%s %s" % (name, _fmt(value)))
        return "\n".join(out) + "\n"


#: The registry every DMS component writes to. One per process, by design.
REGISTRY = Registry()

# --------------------------------------------------------------------- identity

BUILD_INFO = REGISTRY.gauge(
    "dms_build_info",
    "Build identity of the running unit; value is always 1.",
    ("version", "git_sha", "trt_version", "config"),
)
UP = REGISTRY.gauge("dms_up", "1 while the DMS process is serving health.")
UPTIME_SECONDS = REGISTRY.gauge(
    "dms_uptime_seconds", "Seconds since the process notified READY."
)

# ----------------------------------------------------------------------- frames

FRAMES_TOTAL = REGISTRY.counter(
    "dms_frames_total", "Frames handed to the pipeline by the capture source."
)
FRAMES_PROCESSED_TOTAL = REGISTRY.counter(
    "dms_frames_processed_total", "Frames that completed inference."
)
FRAMES_DROPPED_TOTAL = REGISTRY.counter(
    "dms_frames_dropped_total",
    "Frames discarded before inference (queue full, decode error, bad frame).",
    ("reason",),
)
FRAME_ID = REGISTRY.gauge(
    "dms_frame_id", "Frame id of the most recently processed frame (not a rate)."
)
FPS = REGISTRY.gauge("dms_fps", "Processed frames per second over the last window.")
STAGE_LATENCY_MS = REGISTRY.gauge(
    "dms_stage_latency_ms", "Pipeline stage latency in milliseconds.", ("stage", "quantile")
)

# ----------------------------------------------------------------------- alerts

ALERTS_ENTER_TOTAL = REGISTRY.counter(
    "dms_alerts_enter_total", "Alert rising edges.", ("type", "severity")
)
ALERTS_EXIT_TOTAL = REGISTRY.counter(
    "dms_alerts_exit_total", "Alert falling edges.", ("type", "severity")
)
ALERTS_ACTIVE = REGISTRY.gauge("dms_alerts_active", "Alerts currently latched on.")

# ---------------------------------------------------------------------- engines

MODEL_LOADED = REGISTRY.gauge(
    "dms_model_loaded", "1 when the engine deserialised and is usable.", ("model",)
)
MODEL_FAILURES = REGISTRY.gauge(
    "dms_model_consecutive_failures",
    "Consecutive inference failures for this engine; 0 when healthy.",
    ("model",),
)
MODEL_MEAN_MS = REGISTRY.gauge(
    "dms_model_mean_ms", "Rolling mean inference latency in milliseconds.", ("model",)
)
ENGINE_ERRORS_TOTAL = REGISTRY.counter(
    "dms_engine_errors_total",
    "Engine failures by kind (hang, execution, shape, load).",
    ("engine", "kind"),
)
MANIFEST_OK = REGISTRY.gauge(
    "dms_manifest_ok", "1 when every configured engine matched engines/MANIFEST.json."
)

# ----------------------------------------------------------------------- camera

CAMERA_UP = REGISTRY.gauge("dms_camera_up", "1 while the capture source is delivering frames.")
CAMERA_RESTARTS_TOTAL = REGISTRY.counter(
    "dms_camera_restarts_total", "Capture pipeline reconnect attempts."
)

# ------------------------------------------------------------------- event log

EVENT_WRITES_TOTAL = REGISTRY.counter(
    "dms_event_writes_total", "AlertEvent records appended to the JSONL."
)
EVENT_WRITE_FAILURES_TOTAL = REGISTRY.counter(
    "dms_event_write_failures_total", "Failed JSONL appends, by errno name.", ("errno",)
)
EVENT_DROPPED_TOTAL = REGISTRY.counter(
    "dms_event_dropped_total", "AlertEvents discarded because the log was unwritable.", ("reason",)
)
EVENT_ROTATIONS_TOTAL = REGISTRY.counter(
    "dms_event_rotations_total", "In-process JSONL rotations, by trigger.", ("reason",)
)
EVENT_FSYNCS_TOTAL = REGISTRY.counter(
    "dms_event_fsyncs_total", "fsync() calls on the JSONL (durability barriers)."
)
EVENTS_WRITABLE = REGISTRY.gauge(
    "dms_events_writable", "1 while AlertEvents are reaching durable storage."
)
EVENTS_BYTES = REGISTRY.gauge("dms_events_bytes", "Size of the live JSONL in bytes.")

# ---------------------------------------------------------------------- storage

DISK_FREE_MB = REGISTRY.gauge(
    "dms_disk_free_mb", "Free space on the filesystem holding this path, MiB.", ("path",)
)
DISK_FULL = REGISTRY.gauge(
    "dms_disk_full", "1 while any writer has hit ENOSPC/EDQUOT and stopped."
)

# ------------------------------------------------------------------------ clips

CLIPS_ENABLED = REGISTRY.gauge(
    "dms_clips_enabled", "1 only when clips.enabled AND privacy.record_faces are both true."
)
CLIPS_WRITTEN_TOTAL = REGISTRY.counter(
    "dms_clips_written_total", "Incident clips successfully muxed.", ("severity",)
)
CLIPS_DROPPED_TOTAL = REGISTRY.counter(
    "dms_clips_dropped_total", "Incident clips not written, by reason.", ("reason",)
)
CLIPS_BYTES = REGISTRY.gauge("dms_clips_bytes", "Bytes currently held in the clip directory.")
CLIPS_QUOTA_BYTES = REGISTRY.gauge("dms_clips_quota_bytes", "Configured clip quota in bytes.")
CLIPS_SWEPT_TOTAL = REGISTRY.counter(
    "dms_clips_swept_total", "Clip files deleted by the quota/retention sweeper.", ("reason",)
)

# ---------------------------------------------------------------------- thermal

GPU_TEMP_C = REGISTRY.gauge("dms_gpu_temp_c", "GPU thermal zone, degrees Celsius.")
THERMAL_C = REGISTRY.gauge("dms_thermal_c", "CPU/SoC thermal zone, degrees Celsius.")
GPU_CLOCK_MHZ = REGISTRY.gauge("dms_gpu_clock_mhz", "Current GPU clock, MHz.")
RAM_MB = REGISTRY.gauge("dms_ram_mb", "Resident set size of this process, MiB.")
THERMAL_THROTTLE_SECONDS = REGISTRY.counter(
    "dms_thermal_throttle_seconds_total", "Cumulative seconds spent above a thermal limit."
)
DEGRADED = REGISTRY.gauge("dms_degraded", "1 while the unit is running in a degraded mode.")

# --------------------------------------------------------------------- lifecycle

WATCHDOG_PINGS_TOTAL = REGISTRY.counter(
    "dms_watchdog_pings_total", "WATCHDOG=1 datagrams sent to systemd."
)
HEALTH_STALE = REGISTRY.gauge(
    "dms_health_stale", "1 when the health snapshot has not been refreshed in time."
)
HEALTH_REQUESTS_TOTAL = REGISTRY.counter(
    "dms_health_requests_total", "HTTP requests to the health server.", ("route", "code")
)
PRIVACY_RECORD_FACES = REGISTRY.gauge(
    "dms_privacy_record_faces", "Effective privacy.record_faces (1 = face imagery permitted)."
)
PRIVACY_TELEMETRY = REGISTRY.gauge(
    "dms_privacy_telemetry", "Effective privacy.telemetry (1 = state snapshots permitted)."
)


# ------------------------------------------------------------------- convenience


def record_alert(event):
    # type: (Any) -> None
    """Count one :class:`dms.state.alerts.AlertEvent` edge.

    Called from :meth:`dms.io.events.EventLog.write_alert` so the counters exist
    whenever the JSONL is enabled. When ``events.enabled`` is false the application
    must call this directly, otherwise alert counters stay at zero — see
    ``deploy/README.md``.

    Duck-typed on purpose: anything with ``.type``, ``.severity`` and ``.edge``
    works, so the tests do not need the real enum.
    """
    try:
        etype = getattr(event.type, "value", event.type)
        sev = getattr(event.severity, "value", event.severity)
        edge = str(getattr(event, "edge", "enter"))
    except AttributeError:
        log.warning("record_alert got a non-AlertEvent: %r", type(event))
        return
    counter = ALERTS_EXIT_TOTAL if edge == "exit" else ALERTS_ENTER_TOTAL
    counter.inc(type=str(etype), severity=str(sev))


def _git_sha(root):
    # type: (str) -> str
    """Best-effort source revision. Returns '' rather than guessing.

    Order: ``$DMS_GIT_SHA`` (set by the installer), ``/etc/dms/release`` (written by
    ``deploy/setup_jetson.sh``), then ``git rev-parse`` in a checkout. A production
    unit has no .git, which is exactly why the installer writes the release file.
    """
    env = os.environ.get("DMS_GIT_SHA", "").strip()
    if env:
        return env[:40]
    try:
        with open("/etc/dms/release") as f:
            for line in f:
                if line.startswith("git_sha="):
                    return line.split("=", 1)[1].strip()[:40]
    except OSError:
        pass
    if os.path.isdir(os.path.join(root, ".git")):
        try:
            out = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return out.decode("ascii", "replace").strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return ""


def set_build_info(version="", git_sha=None, trt_version="", config=""):
    # type: (str, Optional[str], str, str) -> Dict[str, str]
    """Publish ``dms_build_info`` and return the labels that were used.

    Every label is a string; unknown values become ``"unknown"`` rather than being
    omitted, because a scrape with a missing label silently changes the series
    identity and breaks ``count by (git_sha)`` across a fleet.
    """
    if not version:
        try:
            import dms

            version = getattr(dms, "__version__", "")
        except ImportError:
            version = ""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    labels = {
        "version": version or "unknown",
        "git_sha": (git_sha if git_sha is not None else _git_sha(root)) or "unknown",
        "trt_version": trt_version or "unknown",
        "config": config or "unknown",
    }
    BUILD_INFO.set(1.0, **labels)
    return labels


def render():
    # type: () -> str
    """The default registry as Prometheus text."""
    return REGISTRY.render()


__all__ = [
    "Counter",
    "Gauge",
    "Registry",
    "REGISTRY",
    "record_alert",
    "render",
    "set_build_info",
]
