#!/usr/bin/env python3
"""Run every tests/replay/*.mp4 through dms.app --detect and record what it cost.

Sequential on purpose: the Xavier NX cannot hold several TRT sessions at once,
and two clips running together would measure contention rather than the
pipeline.

  PYTHONPATH=. ./.venv/bin/python scripts/run_all_replay.py
  PYTHONPATH=. ./.venv/bin/python scripts/run_all_replay.py --max-frames 60
  PYTHONPATH=. ./.venv/bin/python scripts/run_all_replay.py --baseline tests/replay/runs/<prev>/summary.json

What this writes
----------------
``<run_dir>/summary.json`` - one machine-readable record per clip, so a
regression between two runs is a diff rather than an archaeology exercise.  Per
clip it carries:

===========================  ==============================================
``seconds`` / ``fps_wall``   wall time and frames/second measured by *this*
                             script, independent of anything the app reports
``frames`` / ``last_id``     from the app's final log line
``infer_p95_ms``             the app's own published latency, whatever it
                             currently means (see the note below)
``peak_rss_mb``              the child's ``VmHWM``, sampled from ``/proc``
``min_mem_available_mb``     system headroom low-water mark during the clip
``gpu_clock_mhz``            min/max devfreq ``cur_freq`` seen during the clip
``loadavg_before/after``     board contention, so a contaminated run is
                             visible instead of silently slow
``alerts``                   counts by alert type from ``events.jsonl``
``latency_budget``           the gate block from ``latency_budget.json`` when
                             the app writes one (see ``DMS_LATENCY_BUDGET_OUT``)
===========================  ==============================================

Why the sampler exists
----------------------
Peak RSS is half of the freeze gate ("peak RAM under 5.0 GB of the measured
6.7 GiB") and after nine full-clip runs there was no memory figure in any
artifact.  ``VmHWM`` is the kernel's own high-water mark, so it is exact and
needs no polling to be accurate - the sampler polls only so that MemAvailable,
the GPU clock and the load average are captured *during* the clip rather than
after it.

``DMS_LATENCY_BUDGET_OUT``
--------------------------
This script exports that variable pointing at ``<clip_dir>/latency_budget.json``
for every child.  ``dms.app`` does not read it yet; when the frame loop is wired
to :class:`dms.perf.LatencyBudget` it should write its report there on shutdown,
and this script will fold the gate verdict into ``summary.json`` automatically.
Until then the field is simply absent - it is never faked.

Units: milliseconds, MiB, MHz, seconds.

Failure behaviour: a clip that exits non-zero is recorded with its return code
and the batch continues; the script exits 1 if any clip failed.  A missing
``events.jsonl`` or unparseable log yields ``null`` fields, never a zero that
would read as "measured and fine".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#: day_driver lives in default.yaml, not configs/day_driver.yaml.
CONFIG_BY_STEM = {
    "day_driver": "configs/default.yaml",
}

#: Sampling period for the child-process resource sampler, seconds.
SAMPLE_S = 0.5

#: Regexes over the app's own log lines. Kept here, in one place, so a change to
#: the log format breaks one function loudly instead of many fields quietly.
RE_PERIODIC = re.compile(
    r"frames=(?P<frames>\d+)\s+fps=(?P<fps>[0-9.]+)\s+infer_p95_ms=(?P<p95>[0-9.]+)"
)
RE_FINAL = re.compile(
    r"stopped frames=(?P<frames>\d+) last_id=(?P<last_id>\d+) "
    r"mean_infer_ms=(?P<mean>[0-9.]+) total_face_hits=(?P<hits>\d+)"
)
RE_WARN = re.compile(r"(THERMAL_THROTTLE|CLOCKS_LOW|PIPELINE_STARVED|ENGINE_FAIL|MODEL_DEGRADED)")


def config_for_clip(mp4_path):
    stem = os.path.splitext(os.path.basename(mp4_path))[0]
    if stem in CONFIG_BY_STEM:
        return CONFIG_BY_STEM[stem]
    named = os.path.join("configs", stem + ".yaml")
    if os.path.isfile(os.path.join(ROOT, named)):
        return named
    return "configs/default.yaml"


def list_clips(clips_dir):
    names = []
    for name in sorted(os.listdir(clips_dir)):
        if name.endswith(".mp4"):
            names.append(os.path.join(clips_dir, name))
    return names


# ------------------------------------------------------------------ sysfs ---
def _proc_field(path, key):
    """First integer on the line starting with ``key`` in ``path``, or None."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _avail_mb():
    kb = _proc_field("/proc/meminfo", "MemAvailable:")
    return None if kb is None else kb // 1024


def _loadavg():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _gpu_clock_mhz():
    base = "/sys/devices/gpu.0/devfreq"
    try:
        for name in sorted(os.listdir(base)):
            with open(os.path.join(base, name, "cur_freq")) as f:
                return int(f.read().strip()) // 1000000
    except (OSError, ValueError):
        return None
    return None


def wait_for_memory(min_avail_mb, poll_s=10):
    """Block until MemAvailable is high enough (other TRT jobs may be building)."""
    if min_avail_mb <= 0:
        return
    while True:
        avail = _avail_mb()
        if avail is None or avail >= min_avail_mb:
            print("mem avail=%s MiB (need %d) - go" % (avail, min_avail_mb))
            return
        print("mem avail=%d MiB < %d; waiting for other processes" % (avail, min_avail_mb))
        time.sleep(poll_s)


class ChildSampler(object):
    """Poll a child's ``/proc`` and the board's while it runs.

    Records the child's ``VmHWM`` (kernel high-water RSS, exact), and the
    minimum system MemAvailable, the GPU clock range and the peak load average
    observed during the clip.  Every field is ``None`` when the node could not
    be read - a run on a board without a Tegra devfreq node reports no clock,
    it does not report zero.
    """

    def __init__(self, pid, period_s=SAMPLE_S):
        # type: (int, float) -> None
        self.pid = int(pid)
        self.period_s = float(period_s)
        self.peak_rss_mb = None
        self.min_avail_mb = None
        self.gpu_min_mhz = None
        self.gpu_max_mhz = None
        self.max_load = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="clip-sampler")
        self._thread.daemon = True

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._sample()  # one final read; VmHWM survives until the process reaps
        return self

    def _sample(self):
        status = "/proc/%d/status" % self.pid
        hwm = _proc_field(status, "VmHWM:")
        if hwm is not None:
            mb = hwm // 1024
            if self.peak_rss_mb is None or mb > self.peak_rss_mb:
                self.peak_rss_mb = mb
        avail = _avail_mb()
        if avail is not None and (self.min_avail_mb is None or avail < self.min_avail_mb):
            self.min_avail_mb = avail
        mhz = _gpu_clock_mhz()
        if mhz is not None:
            if self.gpu_min_mhz is None or mhz < self.gpu_min_mhz:
                self.gpu_min_mhz = mhz
            if self.gpu_max_mhz is None or mhz > self.gpu_max_mhz:
                self.gpu_max_mhz = mhz
        load = _loadavg()
        if load is not None and (self.max_load is None or load > self.max_load):
            self.max_load = load
        self.samples += 1

    def _run(self):
        while not self._stop.wait(self.period_s):
            self._sample()

    def as_dict(self):
        return {
            "peak_rss_mb": self.peak_rss_mb,
            "min_mem_available_mb": self.min_avail_mb,
            "gpu_clock_min_mhz": self.gpu_min_mhz,
            "gpu_clock_max_mhz": self.gpu_max_mhz,
            "max_loadavg": self.max_load,
            "resource_samples": self.samples,
        }


# ------------------------------------------------------------------- child --
def _run_one(cmd, log_path, env):
    """Run one clip, streaming its output to stdout and ``log_path``.

    Returns ``(returncode, sampler)``.
    """
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        sampler = ChildSampler(proc.pid).start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            rc = proc.wait()
        finally:
            sampler.stop()
        return rc, sampler


def parse_log(path):
    """Latency and warning facts from one clip's ``log.txt``.

    ``infer_p95_ms`` is whatever ``dms.app`` published under that name.  Today
    that span covers detect + track + landmarks + geometry only - not capture,
    not the overlay, not the writer - so it is recorded under the app's own
    name and must not be presented as end-to-end.  ``latency_budget.json``,
    when the app writes one, is the end-to-end source of truth.
    """
    out = {
        "frames": None,
        "last_id": None,
        "mean_infer_ms": None,
        "infer_p95_ms": None,
        "total_face_hits": None,
        "alert_lines": 0,
        "warnings": {},
    }
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = RE_PERIODIC.search(line)
                if m:
                    out["infer_p95_ms"] = float(m.group("p95"))
                m = RE_FINAL.search(line)
                if m:
                    out["frames"] = int(m.group("frames"))
                    out["last_id"] = int(m.group("last_id"))
                    out["mean_infer_ms"] = float(m.group("mean"))
                    out["total_face_hits"] = int(m.group("hits"))
                if " ALERT " in line:
                    out["alert_lines"] += 1
                w = RE_WARN.search(line)
                if w:
                    key = w.group(1)
                    out["warnings"][key] = out["warnings"].get(key, 0) + 1
    except OSError:
        return out
    return out


def parse_events(path):
    """``{alert_type: count}`` from an events JSONL, or ``None`` if absent."""
    if not os.path.isfile(path):
        return None
    counts = {}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                key = str(rec.get("type") or rec.get("alert") or "unknown")
                counts[key] = counts.get(key, 0) + 1
    except OSError:
        return None
    return counts


def read_budget(path):
    """The gate + offenders from a ``latency_budget.json``, or ``None``."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            rep = json.load(f)
    except (OSError, ValueError):
        return None
    return {
        "gate": rep.get("gate"),
        "offenders": rep.get("offenders"),
        "accounting": rep.get("accounting"),
    }


# ----------------------------------------------------------------- compare --
#: Fields compared against a baseline, and whether a *higher* value is worse.
COMPARE_FIELDS = (
    ("seconds", True),
    ("fps_wall", False),
    ("mean_infer_ms", True),
    ("infer_p95_ms", True),
    ("peak_rss_mb", True),
)


def compare(results, baseline_path, tol_pct):
    """Per-clip deltas against a previous ``summary.json``.

    ``regressions`` lists any field that moved in the bad direction by more than
    ``tol_pct``.  This is the whole point of writing a machine-readable summary:
    the previous format recorded ``{clip, config, rc, seconds}`` and could not
    show that a change made the pipeline slower.
    """
    try:
        with open(baseline_path) as f:
            base = json.load(f)
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc), "path": baseline_path}
    by_clip = dict((c.get("clip"), c) for c in (base.get("clips") or []))
    deltas = {}
    regressions = []
    for rec in results:
        prev = by_clip.get(rec.get("clip"))
        if prev is None:
            continue
        d = {}
        for field, higher_is_worse in COMPARE_FIELDS:
            a = prev.get(field)
            b = rec.get(field)
            if a is None or b is None or not a:
                continue
            pct = 100.0 * (b - a) / float(a)
            d[field] = {"was": a, "now": b, "delta_pct": round(pct, 1)}
            worse = pct > tol_pct if higher_is_worse else pct < -tol_pct
            if worse:
                regressions.append("%s.%s %s -> %s (%+.1f%%)" % (rec["clip"], field, a, b, pct))
        if d:
            deltas[rec["clip"]] = d
    return {
        "status": "ok",
        "path": baseline_path,
        "tolerance_pct": tol_pct,
        "clips": deltas,
        "regressions": regressions,
    }


# -------------------------------------------------------------------- main --
def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch-run all replay mp4s into a new folder")
    parser.add_argument("--clips-dir", default=os.path.join(ROOT, "tests/replay"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "tests/replay/runs"))
    parser.add_argument("--run-dir", default="", help="reuse this run folder instead of a new timestamp")
    parser.add_argument("--only", nargs="*", default=[], help="only these clip stems")
    parser.add_argument("--skip-complete", action="store_true", help="skip clips that already have preview.jpg")
    parser.add_argument(
        "--min-avail-mb",
        type=int,
        default=1800,
        help="wait until MemAvailable >= this before each clip (0 = never wait)",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 = until file EOS")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-overlay", action="store_true",
                        help="skip --save-video/--save-preview: measures the production path, "
                             "where the overlay is off and costs nothing")
    parser.add_argument("--baseline", default="", help="previous summary.json to diff against")
    parser.add_argument("--tolerance-pct", type=float, default=10.0,
                        help="regression threshold for --baseline, percent")
    parser.add_argument("--max-loadavg", type=float, default=1.5,
                        help="mark the run contaminated above this 1-minute load average")
    parser.add_argument("--strict-load", action="store_true",
                        help="refuse to start a clip when the load average exceeds --max-loadavg")
    args = parser.parse_args(argv)

    clips = list_clips(args.clips_dir)
    if args.only:
        want = set(args.only)
        clips = [p for p in clips if os.path.splitext(os.path.basename(p))[0] in want]
    if not clips:
        sys.stderr.write("no mp4 files in %s\n" % args.clips_dir)
        return 2

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print("run_dir %s clips=%d" % (run_dir, len(clips)))
    results = []
    contaminated = False
    t_all = time.monotonic()
    for mp4 in clips:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        clip_dir = os.path.join(run_dir, stem)
        cfg = config_for_clip(mp4)
        if args.skip_complete and os.path.isfile(os.path.join(clip_dir, "log.txt")):
            print("=== %s skip (log.txt exists) ===" % stem)
            results.append({"clip": stem, "config": cfg, "rc": 0, "seconds": 0, "skipped": True})
            continue
        wait_for_memory(args.min_avail_mb)
        os.makedirs(clip_dir, exist_ok=True)
        load_before = _loadavg()
        if load_before is not None and load_before > args.max_loadavg:
            msg = "loadavg %.2f > %.2f: another job is on this board" % (load_before, args.max_loadavg)
            if args.strict_load:
                sys.stderr.write("refusing to time %s: %s\n" % (stem, msg))
                return 3
            print("WARNING %s: %s - timings for this clip are contaminated" % (stem, msg))
            contaminated = True
        cmd = [
            args.python,
            "-m",
            "dms.app",
            "--config",
            cfg,
            "--source",
            mp4,
            "--detect",
            "--events",
            os.path.join(clip_dir, "events.jsonl"),
            "--log-level",
            "INFO",
        ]
        if not args.no_overlay:
            cmd.extend([
                "--save-preview", os.path.join(clip_dir, "preview.jpg"),
                "--save-video", os.path.join(clip_dir, "overlay.avi"),
            ])
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        clip_env = dict(env)
        # Documented hook: when dms.app is wired to dms.perf.LatencyBudget it
        # should write its report here. Absent today; never faked.
        clip_env["DMS_LATENCY_BUDGET_OUT"] = os.path.join(clip_dir, "latency_budget.json")
        print("=== %s config=%s overlay=%s ===" % (stem, cfg, not args.no_overlay))
        t0 = time.monotonic()
        rc, sampler = _run_one(cmd, os.path.join(clip_dir, "log.txt"), clip_env)
        dt = time.monotonic() - t0
        parsed = parse_log(os.path.join(clip_dir, "log.txt"))
        rec = {
            "clip": stem,
            "config": cfg,
            "rc": rc,
            "seconds": round(dt, 1),
            "overlay": not args.no_overlay,
            "loadavg_before": load_before,
            "loadavg_after": _loadavg(),
        }
        rec.update(parsed)
        rec.update(sampler.as_dict())
        frames = parsed.get("frames")
        rec["fps_wall"] = round(frames / dt, 2) if frames and dt > 0 else None
        rec["ms_per_frame_wall"] = round(1000.0 * dt / frames, 1) if frames else None
        rec["alerts"] = parse_events(os.path.join(clip_dir, "events.jsonl"))
        budget = read_budget(os.path.join(clip_dir, "latency_budget.json"))
        if budget is not None:
            rec["latency_budget"] = budget
        results.append(rec)
        print("=== %s rc=%s in %.1fs frames=%s fps_wall=%s peak_rss=%sMiB gpu=%s-%sMHz ===" % (
            stem, rc, dt, frames, rec["fps_wall"], rec["peak_rss_mb"],
            rec["gpu_clock_min_mhz"], rec["gpu_clock_max_mhz"]))

    timed = [r for r in results if r.get("fps_wall")]
    summary = {
        "schema": "dms.replay_summary/2",
        "run_dir": run_dir,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.monotonic() - t_all, 1),
        "overlay": not args.no_overlay,
        "contaminated": contaminated,
        "failed": [r["clip"] for r in results if r["rc"] != 0],
        "aggregate": {
            "clips_timed": len(timed),
            "worst_ms_per_frame_wall": max([r["ms_per_frame_wall"] for r in timed]) if timed else None,
            "best_ms_per_frame_wall": min([r["ms_per_frame_wall"] for r in timed]) if timed else None,
            "worst_infer_p95_ms": max([r["infer_p95_ms"] for r in timed if r.get("infer_p95_ms")] or [0]) or None,
            "peak_rss_mb": max([r["peak_rss_mb"] for r in results if r.get("peak_rss_mb")] or [0]) or None,
            "min_mem_available_mb": min(
                [r["min_mem_available_mb"] for r in results if r.get("min_mem_available_mb")] or [0]) or None,
        },
        "clips": results,
    }
    if args.baseline:
        summary["comparison"] = compare(results, args.baseline, args.tolerance_pct)
    path = os.path.join(run_dir, "summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    print("wrote %s failed=%s" % (path, summary["failed"]))
    agg = summary["aggregate"]
    print("worst %s ms/frame wall, peak RSS %s MiB, MemAvailable low-water %s MiB" % (
        agg["worst_ms_per_frame_wall"], agg["peak_rss_mb"], agg["min_mem_available_mb"]))
    cmp_ = summary.get("comparison") or {}
    for line in cmp_.get("regressions", []):
        print("REGRESSION %s" % line)
    if summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
