"""Board temps / clocks / RSS from sysfs. No nvpmodel, no jetson_clocks."""
from __future__ import annotations

import os
from typing import Optional, Tuple


def ram_used_mb():
    # type: () -> int
    """VmRSS of this process in MiB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _read_milli(path):
    # type: (str) -> Optional[float]
    try:
        with open(path) as f:
            raw = f.read().strip()
        if not raw:
            return None
        val = float(raw)
        if val > 1000:
            return val / 1000.0
        return val
    except (OSError, ValueError):
        return None


def read_temps():
    # type: () -> Tuple[Optional[float], Optional[float]]
    """Return (gpu_c, thermal_c)."""
    gpu = None
    thermal = None
    base = "/sys/class/thermal"
    if not os.path.isdir(base):
        return None, None
    for name in sorted(os.listdir(base)):
        if not name.startswith("thermal_zone"):
            continue
        z = os.path.join(base, name)
        typ = ""
        try:
            with open(os.path.join(z, "type")) as f:
                typ = f.read().strip()
        except OSError:
            continue
        temp = _read_milli(os.path.join(z, "temp"))
        if temp is None:
            continue
        low = typ.lower()
        if gpu is None and ("gpu" in low):
            gpu = temp
        if thermal is None and ("thermal" in low or low in ("CPU-therm", "cpu-thermal", "soc0-thermal")):
            thermal = temp
        if thermal is None and name.endswith("0"):
            thermal = temp
    return gpu, thermal


def gpu_clock_mhz():
    # type: () -> Optional[float]
    root = "/sys/devices/gpu.0/devfreq"
    if not os.path.isdir(root):
        return None
    try:
        for name in os.listdir(root):
            path = os.path.join(root, name, "cur_freq")
            if os.path.isfile(path):
                hz = _read_milli(path)
                if hz is None:
                    return None
                # cur_freq is Hz, not milli
                with open(path) as f:
                    raw = float(f.read().strip())
                if raw > 10000:
                    return raw / 1e6
                return raw
    except (OSError, ValueError):
        return None
    return None


def thermal_throttle(gpu_c, thermal_c, gpu_lim=80.0, therm_lim=75.0):
    # type: (Optional[float], Optional[float], float, float) -> bool
    if gpu_c is not None and gpu_c >= gpu_lim:
        return True
    if thermal_c is not None and thermal_c >= therm_lim:
        return True
    return False
