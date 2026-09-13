"""Operations surface: durable events, health/metrics HTTP, board sensors, actuators.

Import order matters. :mod:`dms.io.metrics` owns the process-wide registry and every
other module in this package writes to it, so it is imported first; importing a
sibling before it would try to resolve ``dms.io.metrics`` through a half-initialised
package.

Nothing here imports GStreamer, TensorRT or ``Jetson.GPIO`` at module scope — those
are pulled in lazily inside :class:`dms.io.clips.GstSegmentWriter` and
:class:`dms.io.gpio_alert.GpioAlert`, so the whole package imports and unit-tests on
a machine with none of them.
"""
from dms.io import metrics  # noqa: F401  (must be first: everything else uses it)
from dms.io.thermal import (
    disk_free_mb,
    gpu_clock_mhz,
    mem_available_mb,
    ram_used_mb,
    read_temps,
    thermal_throttle,
    zone_report,
)
from dms.io.sd_notify import (
    notify_ready,
    notify_status,
    notify_stopping,
    notify_watchdog,
    sd_notify,
    under_systemd,
    watchdog_ping_interval_s,
)
from dms.io.events import EventLog, default_events_path, read_events
from dms.io.health import HealthServer, HealthState, is_loopback
from dms.io.gpio_alert import GpioAlert
from dms.io.clips import ClipRing

__all__ = [
    "ClipRing",
    "EventLog",
    "GpioAlert",
    "HealthServer",
    "HealthState",
    "default_events_path",
    "disk_free_mb",
    "gpu_clock_mhz",
    "is_loopback",
    "mem_available_mb",
    "metrics",
    "notify_ready",
    "notify_status",
    "notify_stopping",
    "notify_watchdog",
    "ram_used_mb",
    "read_events",
    "read_temps",
    "sd_notify",
    "thermal_throttle",
    "under_systemd",
    "watchdog_ping_interval_s",
    "zone_report",
]
