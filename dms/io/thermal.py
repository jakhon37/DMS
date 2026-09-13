"""Board resource readings from sysfs: temperatures, GPU clock, RSS, disk headroom.

No ``nvpmodel``, no ``jetson_clocks``, no ``jetson-stats`` — the board forbids that
package and this process must never change the power state of a moving vehicle. Every
value here is a read of a kernel file.

Zone selection
--------------
The previous implementation picked the CPU zone with ``"thermal" in type.lower()``
plus a ``name.endswith("0")`` fallback, which on this Xavier NX happens to land on
``thermal_zone0 = CPU-therm`` by luck. It would just as happily have selected
``thermal_fan_est`` — a *fan speed estimate*, not a die temperature — on a board whose
zones are ordered differently, and ``endswith("0")`` also matches ``thermal_zone10``.
Since ``thermal_c`` gates :func:`thermal_throttle`, which disables the overlay and
sets ``health.degraded``, a wrong zone means the thermal guard fires at the wrong
temperature or never fires at all.

Selection is now explicit: an ordered preference list of zone *types* per role,
matched case-insensitively and exactly, then a fallback that skips any zone on
:data:`EXCLUDED_SUBSTRINGS`. Zones are walked in numeric order
(``thermal_zone2`` before ``thermal_zone10``), and the chosen zone for each role is
logged once at startup and exposed by :func:`zone_report` for ``/healthz``.

This board reads: zone0 CPU-therm, 1 GPU-therm, 2 AUX-therm, 3 AO-therm, 4 PMIC-Die,
5 thermal_fan_est, 6 iwlwifi_1.

Units: temperatures degrees Celsius (sysfs gives millidegrees), clocks MHz (Tegra
devfreq ``cur_freq`` gives Hz), memory and disk MiB.

Failure behaviour: every reader returns ``None`` when the node is absent or
unparseable — never a plausible constant. An unreadable thermal zone therefore
disables the thermal guard rather than silently reporting a safe temperature, and
that state is visible because ``gpu_temp_c``/``thermal_c`` go null on ``/healthz``.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("dms.thermal")

#: Root of the kernel thermal-zone tree. Overridable for tests.
THERMAL_ROOT = "/sys/class/thermal"
#: Tegra GPU devfreq root; ``cur_freq`` under it is in Hz.
GPU_DEVFREQ_ROOT = "/sys/devices/gpu.0/devfreq"

#: Zone ``type`` strings that are the GPU die, most-preferred first.
GPU_ZONE_TYPES = ("gpu-therm", "gpu_thermal", "gpu-thermal", "gpu")
#: Zone ``type`` strings that are the CPU/SoC die, most-preferred first.
CPU_ZONE_TYPES = (
    "cpu-therm",
    "cpu-thermal",
    "cpu_thermal",
    "soc0-thermal",
    "soc1-thermal",
    "soc2-thermal",
    "soc-thermal",
)
#: Never accept a zone whose type contains one of these as a die temperature.
#: ``fan_est`` is a fan-speed estimate, ``pmic``/``ao``/``aux`` are off-die regulators
#: and always-on domains, ``iwlwifi``/``bat`` belong to other devices entirely.
EXCLUDED_SUBSTRINGS = (
    "fan_est",
    "fan-est",
    "pmic",
    "iwlwifi",
    "aux",
    "ao-therm",
    "ao_therm",
    "bat",
    "tboard",
    "tdiode",
)

_logged_zones = False


def ram_used_mb():
    # type: () -> int
    """VmRSS of this process in MiB. 0 when /proc is unreadable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def mem_available_mb():
    # type: () -> Optional[int]
    """System-wide MemAvailable in MiB — the number that predicts an OOM kill."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_text(path):
    # type: (str) -> Optional[str]
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _zone_index(name):
    # type: (str) -> int
    """Numeric suffix of ``thermal_zoneN``; a huge number when it does not parse.

    ``sorted()`` on the directory listing is lexicographic, so ``thermal_zone10``
    sorts before ``thermal_zone2``. On a board with ten or more zones that alone
    changes which zone a positional fallback picks.
    """
    tail = name[len("thermal_zone"):]
    try:
        return int(tail)
    except ValueError:
        return 1 << 30


def list_zones(root=None):
    # type: (Optional[str]) -> List[Tuple[str, str, Optional[float]]]
    """Every readable thermal zone as ``(name, type, celsius)``, in numeric order."""
    base = root or THERMAL_ROOT
    out = []  # type: List[Tuple[str, str, Optional[float]]]
    if not os.path.isdir(base):
        return out
    try:
        names = os.listdir(base)
    except OSError:
        return out
    for name in sorted((n for n in names if n.startswith("thermal_zone")), key=_zone_index):
        z = os.path.join(base, name)
        typ = _read_text(os.path.join(z, "type"))
        if typ is None:
            continue
        raw = _read_text(os.path.join(z, "temp"))
        temp = None  # type: Optional[float]
        if raw:
            try:
                # sysfs thermal zones are millidegrees Celsius, unconditionally.
                temp = int(raw) / 1000.0
            except ValueError:
                temp = None
        out.append((name, typ, temp))
    return out


def _pick(zones, preferred):
    # type: (List[Tuple[str, str, Optional[float]]], Tuple[str, ...]) -> Optional[Tuple[str, str, float]]
    """First zone whose lowercased type exactly matches the earliest preference."""
    for want in preferred:
        for name, typ, temp in zones:
            if temp is None:
                continue
            if typ.strip().lower() == want:
                return name, typ, temp
    return None


def _fallback(zones):
    # type: (List[Tuple[str, str, Optional[float]]]) -> Optional[Tuple[str, str, float]]
    """Lowest-numbered zone that is not on the exclusion list."""
    for name, typ, temp in zones:
        if temp is None:
            continue
        low = typ.strip().lower()
        if any(bad in low for bad in EXCLUDED_SUBSTRINGS):
            continue
        return name, typ, temp
    return None


def select_zones(root=None):
    # type: (Optional[str]) -> Dict[str, Optional[Tuple[str, str, float]]]
    """Resolve the ``gpu`` and ``cpu`` roles to concrete zones. No caching.

    Returns ``{"gpu": (name, type, celsius) | None, "cpu": ... }``. The CPU role
    falls back to the lowest-numbered non-excluded zone; the GPU role never falls
    back, because there is no sensible substitute for a GPU die temperature and
    reporting a CPU reading as ``gpu_temp_c`` would move the 80 °C gate.
    """
    zones = list_zones(root)
    gpu = _pick(zones, GPU_ZONE_TYPES)
    cpu = _pick(zones, CPU_ZONE_TYPES)
    if cpu is None:
        cpu = _fallback(zones)
    return {"gpu": gpu, "cpu": cpu}


def read_temps(root=None):
    # type: (Optional[str]) -> Tuple[Optional[float], Optional[float]]
    """Return ``(gpu_c, thermal_c)`` in degrees Celsius, ``None`` where unavailable.

    Logs the chosen zone for each role exactly once per process, so an operator can
    confirm from the journal which sensor the thermal guard is actually watching.
    """
    global _logged_zones
    sel = select_zones(root)
    if not _logged_zones:
        _logged_zones = True
        log.info(
            "thermal zones selected gpu=%s cpu=%s",
            "none" if sel["gpu"] is None else "%s/%s" % (sel["gpu"][0], sel["gpu"][1]),
            "none" if sel["cpu"] is None else "%s/%s" % (sel["cpu"][0], sel["cpu"][1]),
        )
    gpu = None if sel["gpu"] is None else sel["gpu"][2]
    cpu = None if sel["cpu"] is None else sel["cpu"][2]
    return gpu, cpu


def zone_report(root=None):
    # type: (Optional[str]) -> Dict[str, object]
    """Which zone backs each role plus every zone read, for ``/healthz``."""
    sel = select_zones(root)
    out = {}  # type: Dict[str, object]
    for role in ("gpu", "cpu"):
        pick = sel[role]
        out[role] = None if pick is None else {"zone": pick[0], "type": pick[1], "c": round(pick[2], 1)}
    out["all"] = [
        {"zone": n, "type": t, "c": None if c is None else round(c, 1)} for n, t, c in list_zones(root)
    ]
    return out


def gpu_clock_mhz(root=None):
    # type: (Optional[str]) -> Optional[float]
    """Current GPU clock in MHz, or ``None`` when the node is absent or not in Hz.

    Tegra's ``devfreq/*/cur_freq`` is Hz (gv11b idles near 76 800 000 and tops out
    around 1 109 000 000). A value below 1 MHz is a devfreq node in some other unit;
    rather than guess, this returns ``None`` and logs once — a wrong clock reading
    would make the ``CLOCKS_LOW`` warning meaningless in both directions.
    """
    base = root or GPU_DEVFREQ_ROOT
    if not os.path.isdir(base):
        return None
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return None
    for name in names:
        path = os.path.join(base, name, "cur_freq")
        if not os.path.isfile(path):
            continue
        raw = _read_text(path)
        if not raw:
            return None
        try:
            hz = float(raw)
        except ValueError:
            return None
        if hz < 1e6:
            log.debug("devfreq %s reads %s; not Hz, ignoring", path, raw)
            return None
        return hz / 1e6
    return None


def disk_free_mb(path):
    # type: (str) -> Optional[float]
    """Free space on the filesystem holding *path*, MiB. ``None`` if it cannot stat.

    Walks up to the nearest existing ancestor, so this works for a directory the
    caller is about to create.
    """
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent
    try:
        return shutil.disk_usage(p).free / (1024.0 * 1024.0)
    except OSError:
        return None


def disk_report(paths):
    # type: (List[str]) -> Dict[str, Optional[float]]
    """``{path: free MiB}`` for the state locations, for ``/healthz``."""
    return {p: disk_free_mb(p) for p in paths}


def thermal_throttle(gpu_c, thermal_c, gpu_lim=80.0, therm_lim=75.0):
    # type: (Optional[float], Optional[float], float, float) -> bool
    """True when either die is at or above its limit. Unknown temperatures do not throttle."""
    if gpu_c is not None and gpu_c >= gpu_lim:
        return True
    if thermal_c is not None and thermal_c >= therm_lim:
        return True
    return False


__all__ = [
    "disk_free_mb",
    "disk_report",
    "gpu_clock_mhz",
    "list_zones",
    "mem_available_mb",
    "ram_used_mb",
    "read_temps",
    "select_zones",
    "thermal_throttle",
    "zone_report",
]
