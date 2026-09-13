"""Per-stage latency budget: targets in, actuals in, ``latency_budget.json`` out.

The freeze gate for this project is stated once, in
``docs/production-dms-design.md``: **1280x720 end-to-end p95 <= 66 ms** (>= 15 FPS)
at ``MODE_20W_6CORE`` with **peak RSS < 5.0 GB**.  Until now nothing in the tree
measured either half of it.  ``health.latency_ms["e2e_p95"]`` was the detector +
landmark span only - 55-60% of true end-to-end - so a run could report 67 ms while
actually spending 149 ms per frame, which is exactly what run ``20260909-042432``
did.

This module turns that gate into an artifact.  It owns:

* the canonical stage list (:data:`STAGES`) so every producer spells the stages
  the same way and a budget file from one run is comparable with the next;
* per-stage target milliseconds, defaulted from measurement on this board and
  overridable from config;
* the verdict - which stage blew its budget, by how much, and whether the
  overall gate passed - written to ``latency_budget.json``.

Units
-----
Milliseconds for every latency, MiB for every memory figure, MHz for clocks,
degrees Celsius for temperatures.  ``share_pct`` is a stage's p50 as a
percentage of the e2e p50.

What "blew the budget" means
----------------------------
A stage fails when its **p95** exceeds its target.  p95 rather than mean because
the gate is a p95, and a stage whose mean is fine but whose tail is not is
precisely the failure mode that a mean hides.  A stage with fewer than
:data:`MIN_SAMPLES` samples in the window is reported as ``insufficient_data``
and never as a pass - claiming a stage met a budget on three samples is the kind
of false assurance this file exists to prevent.

Failure behaviour
-----------------
Nothing here raises into the frame loop.  :meth:`LatencyBudget.observe` and
:meth:`LatencyBudget.stage` are the hot-path entry points and do no I/O.
:meth:`LatencyBudget.write` performs the only file write, atomically
(``tmp`` + ``os.replace``), and propagates an ``OSError`` to its caller - a
budget file that silently failed to appear is worse than a crash in a script.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

from dms.perf.timers import DEFAULT_WINDOW, StageStats, StageTimer

#: A stage needs at least this many samples in the window before a pass/fail
#: verdict is meaningful.
MIN_SAMPLES = 20

#: The design gate, restated here as the default so a caller that passes no
#: config still measures against the real contract.
GATE_E2E_P95_MS = 66.0
GATE_PEAK_RSS_MB = 5000.0


class StageSpec(object):
    """One row of the budget table.

    ``target_ms`` is the p95 allowance.  ``optional`` marks a stage that does not
    run every frame (the object net at ``every_n``, identity on a new track id,
    occlusion on a state change); an optional stage that never ran is reported
    ``not_run``, not ``fail``.  ``group`` is used only for the human-readable
    rollup.
    """

    __slots__ = ("name", "target_ms", "optional", "group", "note")

    def __init__(self, name, target_ms, group, optional=False, note=""):
        # type: (str, float, str, bool, str) -> None
        self.name = str(name)
        self.target_ms = float(target_ms)
        self.group = str(group)
        self.optional = bool(optional)
        self.note = str(note)

    def as_dict(self):
        # type: () -> Dict[str, object]
        return {
            "target_ms": self.target_ms,
            "group": self.group,
            "optional": self.optional,
            "note": self.note,
        }


#: Canonical stage order.  Targets are the **measured p95 of the reachable
#: production profile on this board** (see ``docs/performance.md``, bench run
#: ``cadence-01``) rounded up to the next sensible figure - not aspirations: a
#: ``fail`` here must mean "something got worse", never "we wish this were
#: faster".  The nine non-optional targets sum to 40.0 ms, leaving 26 ms of the
#: 66 ms gate for the cadenced heads and scheduling margin.  Do not tune these
#: without re-running ``scripts/bench.py`` and updating ``docs/performance.md``.
STAGES = (
    StageSpec("capture", 2.0, "io", note="GstSource.read() dequeue of an owned BGRA frame"),
    StageSpec("capture_lag", 60.0, "io", note="age of the frame at dequeue, not a duration"),
    StageSpec("bgra", 1.5, "pre", note="one BGRA->contiguous BGR pass over the frame (measured 0.76)"),
    StageSpec("letterbox", 5.0, "pre", note="detector letterbox + normalise into the blob (measured 4.32)"),
    StageSpec("detect_gpu", 12.0, "gpu", note="SCRFD-512 H2D + execute + D2H round trip (measured 9.94)"),
    StageSpec("detect_post", 4.0, "post", note="SCRFD decode over 3 strides + NMS (measured 3.27)"),
    StageSpec("track", 1.0, "post", note="IoU tracker update + driver selection (measured 0.44)"),
    StageSpec("crop", 2.0, "pre", note="all per-head crops sliced out of the frame"),
    StageSpec("landmarks", 11.0, "gpu", note="2d106det warp + engine + inverse affine (measured 9.29)"),
    StageSpec("headpose", 11.0, "gpu", optional=True, note="6DRepNet MobileNetV2 224 (measured 9.43)"),
    StageSpec("eyestate", 1.5, "gpu", optional=True, note="OCEC batch-2 eye engine (measured 0.84)"),
    StageSpec("mouthstate", 1.0, "gpu", optional=True, note="VSDLM mouth engine (measured 0.42)"),
    StageSpec("occlusion", 8.0, "gpu", optional=True, note="SGC + MWC, every 15th frame (measured 7.31)"),
    StageSpec("gaze", 8.0, "gpu", optional=True, note="MobileGaze 448 - licence gated, off in production"),
    StageSpec("objects", 25.0, "gpu", optional=True, note="YOLOX-Nano 416 every 20th frame (measured 22.33)"),
    StageSpec("identity", 15.0, "gpu", optional=True, note="SFace 112, once per new track id (measured 12.79)"),
    StageSpec("fusion", 1.5, "state", note="fusion + DriverMonitor.update + alert edges (measured 0.70)"),
    StageSpec("overlay", 4.0, "viz", optional=True, note="opt-in; 0 when display and save are off"),
    StageSpec("io", 3.0, "io", optional=True, note="video writer, event log, health publish"),
)

STAGE_BY_NAME = dict((s.name, s) for s in STAGES)  # type: Dict[str, StageSpec]

#: Stages excluded from the "sum of parts" rollup because they are gauges rather
#: than work the frame loop performs.
GAUGE_STAGES = frozenset(("capture_lag",))

#: The end-to-end stage name.  Timed by the caller around one whole frame.
E2E = "e2e"


def default_targets():
    # type: () -> Dict[str, float]
    """``{stage: target_ms}`` from :data:`STAGES`, plus the e2e gate."""
    out = dict((s.name, s.target_ms) for s in STAGES)
    out[E2E] = GATE_E2E_P95_MS
    return out


class LatencyBudget(object):
    """Targets + a :class:`~dms.perf.timers.StageTimer` + the verdict writer.

    Construct one per process.  The frame loop times stages through it exactly
    as it would through a bare ``StageTimer``::

        budget = LatencyBudget.from_config(cfg)
        with budget.stage("detect_gpu"):
            ...
        budget.observe_rss(ram_used_mb())
        budget.mark_frame()
        ...
        budget.write(os.path.join(run_dir, "latency_budget.json"))
    """

    def __init__(
        self,
        targets=None,  # type: Optional[Dict[str, float]]
        window=DEFAULT_WINDOW,  # type: int
        gate_e2e_p95_ms=GATE_E2E_P95_MS,  # type: float
        gate_peak_rss_mb=GATE_PEAK_RSS_MB,  # type: float
        timer=None,  # type: Optional[StageTimer]
        label="",  # type: str
    ):
        # type: (...) -> None
        self.targets = default_targets()
        if targets:
            for k, v in targets.items():
                self.targets[str(k)] = float(v)
        self.gate_e2e_p95_ms = float(gate_e2e_p95_ms)
        self.gate_peak_rss_mb = float(gate_peak_rss_mb)
        self.timer = timer if timer is not None else StageTimer(window=window)
        self.label = str(label)
        self.peak_rss_mb = 0
        self.min_avail_mb = None  # type: Optional[int]
        self.gpu_clock_min_mhz = None  # type: Optional[int]
        self.gpu_clock_max_mhz = None  # type: Optional[int]
        self.max_gpu_temp_c = None  # type: Optional[float]
        self.thermal_events = 0
        self.context = {}  # type: Dict[str, object]
        self._t_start = time.time()
        self._mono_start = time.monotonic()

    # -------------------------------------------------------------- factories
    @classmethod
    def from_mapping(cls, mapping, **kw):
        # type: (Optional[dict], object) -> LatencyBudget
        """Build from ``{"window": int, "gate_e2e_p95_ms": float, "targets": {...}}``.

        Unknown stage names in ``targets`` are kept: a workstream may add a stage
        before this table knows about it, and dropping its budget silently would
        be worse than reporting a row with no spec.
        """
        m = dict(mapping or {})
        window = int(m.get("window", DEFAULT_WINDOW))
        gate = float(m.get("gate_e2e_p95_ms", GATE_E2E_P95_MS))
        rss = float(m.get("gate_peak_rss_mb", GATE_PEAK_RSS_MB))
        targets = m.get("targets") or {}
        return cls(targets=targets, window=window, gate_e2e_p95_ms=gate,
                   gate_peak_rss_mb=rss, **kw)  # type: ignore[arg-type]

    @classmethod
    def from_config(cls, cfg, **kw):
        # type: (object, object) -> LatencyBudget
        """Build from an ``AppConfig``-shaped object.

        Reads ``cfg.perf`` when the schema grows one; until then it falls back to
        the built-in table, so this is safe to call today.  A pydantic model, a
        plain object and a dict are all accepted.
        """
        perf = None
        if isinstance(cfg, dict):
            perf = cfg.get("perf")
        else:
            perf = getattr(cfg, "perf", None)
        if perf is not None and not isinstance(perf, dict):
            for meth in ("dict", "model_dump"):
                fn = getattr(perf, meth, None)
                if callable(fn):
                    try:
                        perf = fn()
                        break
                    except TypeError:
                        continue
        if not isinstance(perf, dict):
            perf = None
        return cls.from_mapping(perf, **kw)  # type: ignore[arg-type]

    @classmethod
    def from_file(cls, path, **kw):
        # type: (str, object) -> LatencyBudget
        """Build from a JSON file holding the :meth:`from_mapping` shape."""
        with open(path, "r") as f:
            return cls.from_mapping(json.load(f), **kw)  # type: ignore[arg-type]

    # ------------------------------------------------------------- hot path --
    def stage(self, name):
        # type: (str) -> object
        """Context manager for one stage. Same object every call - no allocation."""
        return self.timer.stage(name)

    def record(self, name, ms):
        # type: (str, float) -> None
        self.timer.record(name, ms)

    def mark_frame(self):
        # type: () -> None
        self.timer.mark_frame()

    def observe_rss(self, mb):
        # type: (Optional[int]) -> None
        """Track peak RSS in MiB. ``None`` (unreadable /proc) is ignored."""
        if mb is None:
            return
        if int(mb) > self.peak_rss_mb:
            self.peak_rss_mb = int(mb)

    def observe_mem_available(self, mb):
        # type: (Optional[int]) -> None
        """Track the *lowest* system MemAvailable seen, in MiB."""
        if mb is None:
            return
        if self.min_avail_mb is None or int(mb) < self.min_avail_mb:
            self.min_avail_mb = int(mb)

    def observe_gpu_clock(self, mhz):
        # type: (Optional[int]) -> None
        """Track the clock range the run actually executed at.

        Without this a latency number is unattributable: this board idles at
        114.75 MHz under ``nvhost_podgov`` and peaks at 1109.25 MHz, and a
        sequential pipeline with a ~10% GPU duty cycle holds the governor near
        the floor.  A 3x clock deficit is worth 10-20 ms/frame.
        """
        if mhz is None:
            return
        m = int(mhz)
        if self.gpu_clock_min_mhz is None or m < self.gpu_clock_min_mhz:
            self.gpu_clock_min_mhz = m
        if self.gpu_clock_max_mhz is None or m > self.gpu_clock_max_mhz:
            self.gpu_clock_max_mhz = m

    def observe_temp(self, gpu_c, throttled=False):
        # type: (Optional[float], bool) -> None
        if gpu_c is not None:
            c = float(gpu_c)
            if self.max_gpu_temp_c is None or c > self.max_gpu_temp_c:
                self.max_gpu_temp_c = c
        if throttled:
            self.thermal_events += 1

    def set_context(self, **kw):
        # type: (object) -> None
        """Attach run metadata (clip, config, resolution, engines) to the report."""
        self.context.update(kw)

    # -------------------------------------------------------------- verdicts
    def rows(self):
        # type: () -> List[Dict[str, object]]
        """One row per observed stage plus every non-optional stage with no data."""
        snap = self.timer.snapshot()
        e2e = snap.get(E2E)
        e2e_p50 = e2e.p50 if e2e is not None else 0.0
        out = []  # type: List[Dict[str, object]]
        seen = set()
        order = [s.name for s in STAGES] + [E2E]
        for name in order + sorted(n for n in snap if n not in order):
            if name in seen:
                continue
            seen.add(name)
            spec = STAGE_BY_NAME.get(name)
            st = snap.get(name)
            target = self.targets.get(name)
            optional = bool(spec.optional) if spec is not None else True
            row = {
                "stage": name,
                "group": spec.group if spec is not None else ("gate" if name == E2E else "other"),
                "target_ms": target,
                "optional": optional,
                "note": spec.note if spec is not None else "",
            }
            if st is None:
                if optional or spec is None:
                    row["status"] = "not_run"
                    out.append(row)
                    continue
                row["status"] = "missing"
                row["detail"] = "no samples for a stage the pipeline is expected to run"
                out.append(row)
                continue
            row.update(st.as_dict())
            if e2e_p50 > 0.0 and name != E2E and name not in GAUGE_STAGES:
                row["share_pct"] = round(100.0 * st.p50 / e2e_p50, 1)
            if st.count < MIN_SAMPLES:
                row["status"] = "insufficient_data"
            elif target is None:
                row["status"] = "unbudgeted"
            elif st.p95 <= target:
                row["status"] = "pass"
            else:
                row["status"] = "fail"
                row["over_ms"] = round(st.p95 - target, 3)
            out.append(row)
        return out

    def offenders(self):
        # type: () -> List[Dict[str, object]]
        """Failing rows, worst overrun first. This is the headline of the report."""
        bad = [r for r in self.rows() if r.get("status") in ("fail", "missing")]
        bad.sort(key=lambda r: float(r.get("over_ms") or 0.0), reverse=True)
        return bad

    def sum_of_parts_ms(self):
        # type: () -> Tuple[float, float]
        """``(amortised per-frame stage cost, e2e mean)`` in ms.

        **Amortised**, not a raw sum of p50s: a head that runs on every tenth
        frame contributes ``mean_ms * count / frames``, one tenth of its cost.
        Adding the raw p50s of cadenced stages over-counts badly - the object net
        at 22 ms every tenth frame would appear as 22 ms on every frame - and
        that arithmetic is exactly how a budget starts lying.

        A large positive gap between the two numbers means real cost is happening
        outside every instrumented stage, which is the failure the old single
        timer had.  A negative gap means a stage is being double counted, usually
        because it is nested inside another timed stage.
        """
        snap = self.timer.snapshot()
        frames = self.timer.frames
        if frames <= 0:
            e2e_st = snap.get(E2E)
            frames = e2e_st.count if e2e_st is not None else 0
        total = 0.0
        for name, st in snap.items():
            if name == E2E or name in GAUGE_STAGES:
                continue
            if frames > 0:
                total += st.mean_ms * st.count / float(frames)
            else:
                total += st.mean_ms
        e2e = snap.get(E2E)
        return total, (e2e.mean_ms if e2e is not None else 0.0)

    def raw_sum_p50_ms(self):
        # type: () -> float
        """Unweighted sum of stage p50s - meaningful only when nothing is cadenced."""
        snap = self.timer.snapshot()
        return sum(st.p50 for name, st in snap.items()
                   if name != E2E and name not in GAUGE_STAGES)

    def gate(self):
        # type: () -> Dict[str, object]
        """The freeze gate verdict: e2e p95 and peak RSS against their limits."""
        snap = self.timer.snapshot()
        e2e = snap.get(E2E)
        e2e_p95 = e2e.p95 if e2e is not None else None
        n = e2e.count if e2e is not None else 0
        latency_ok = e2e_p95 is not None and n >= MIN_SAMPLES and e2e_p95 <= self.gate_e2e_p95_ms
        rss_ok = self.peak_rss_mb > 0 and self.peak_rss_mb <= self.gate_peak_rss_mb
        return {
            "e2e_p95_ms": None if e2e_p95 is None else round(e2e_p95, 2),
            "e2e_p95_limit_ms": self.gate_e2e_p95_ms,
            "e2e_samples": n,
            "latency_pass": bool(latency_ok),
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_limit_mb": self.gate_peak_rss_mb,
            "rss_pass": bool(rss_ok),
            "rss_measured": self.peak_rss_mb > 0,
            "fps_mean": round(self.timer.fps(), 2),
            "pass": bool(latency_ok and rss_ok),
        }

    def report(self, extra=None):
        # type: (Optional[dict]) -> Dict[str, object]
        """The whole ``latency_budget.json`` payload as a dict."""
        parts, e2e_mean = self.sum_of_parts_ms()
        rep = {
            "schema": "dms.latency_budget/1",
            "label": self.label,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.monotonic() - self._mono_start, 2),
            "frames": self.timer.frames,
            "window_samples": self.timer.window,
            "gate": self.gate(),
            "stages": self.rows(),
            "offenders": [r["stage"] for r in self.offenders()],
            "accounting": {
                "sum_stage_amortised_ms": round(parts, 2),
                "e2e_mean_ms": round(e2e_mean, 2),
                "unaccounted_ms": round(e2e_mean - parts, 2) if e2e_mean > 0.0 else None,
                "sum_stage_p50_ms": round(self.raw_sum_p50_ms(), 2),
                "note": "amortised = mean_ms * count / frames per stage, so a cadenced "
                        "head contributes only the fraction of frames it runs on",
            },
            "resources": {
                "peak_rss_mb": self.peak_rss_mb,
                "min_mem_available_mb": self.min_avail_mb,
                "gpu_clock_min_mhz": self.gpu_clock_min_mhz,
                "gpu_clock_max_mhz": self.gpu_clock_max_mhz,
                "max_gpu_temp_c": self.max_gpu_temp_c,
                "thermal_events": self.thermal_events,
            },
            "context": dict(self.context),
        }
        if extra:
            rep.update(extra)
        return rep

    def write(self, path, extra=None):
        # type: (str, Optional[dict]) -> str
        """Write ``latency_budget.json`` atomically. Returns the path written."""
        rep = self.report(extra)
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
        return path

    # --------------------------------------------------------------- display
    def markdown(self):
        # type: () -> str
        """A fixed-width table of the current rows, for logs and docs."""
        rows = self.rows()
        head = "| stage | p50 ms | p95 ms | p99 ms | target | share | status |"
        sep = "|---|---:|---:|---:|---:|---:|---|"
        lines = [head, sep]
        for r in rows:
            if r.get("status") == "not_run":
                continue
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    r["stage"],
                    _fmt(r.get("p50_ms")),
                    _fmt(r.get("p95_ms")),
                    _fmt(r.get("p99_ms")),
                    _fmt(r.get("target_ms")),
                    "" if r.get("share_pct") is None else ("%.1f%%" % r["share_pct"]),
                    r.get("status", ""),
                )
            )
        g = self.gate()
        lines.append("")
        lines.append(
            "gate: e2e p95 %s ms / %.0f ms -> %s ; peak RSS %s MiB / %.0f MiB -> %s"
            % (
                _fmt(g["e2e_p95_ms"]),
                self.gate_e2e_p95_ms,
                "PASS" if g["latency_pass"] else "FAIL",
                g["peak_rss_mb"],
                self.gate_peak_rss_mb,
                "PASS" if g["rss_pass"] else ("NOT MEASURED" if not g["rss_measured"] else "FAIL"),
            )
        )
        return "\n".join(lines)


def _fmt(v):
    # type: (object) -> str
    if v is None:
        return "-"
    try:
        return "%.2f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def stage_names():
    # type: () -> List[str]
    """Canonical stage names in pipeline order, ending with ``e2e``."""
    return [s.name for s in STAGES] + [E2E]


__all__ = [
    "E2E",
    "GATE_E2E_P95_MS",
    "GATE_PEAK_RSS_MB",
    "GAUGE_STAGES",
    "MIN_SAMPLES",
    "STAGES",
    "STAGE_BY_NAME",
    "LatencyBudget",
    "StageSpec",
    "default_targets",
    "stage_names",
]
