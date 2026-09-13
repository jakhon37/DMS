"""Driver Monitoring System - process entry point and frame loop.

This module owns assembly and nothing else.  Every capability lives behind an
interface that degrades on its own; the job here is to wire them together, to
keep the loop tight, and to make sure that whatever fails, the unit reports what
actually happened rather than what was requested.

Layout of one frame
-------------------
1. ``CaptureSupervisor.read`` -> a ``Frame``, a one-shot failure sentinel, or
   ``None``.  The supervisor owns reconnection, backoff and the
   ``camera_fail`` / ``pipeline_starved`` edges.
2. One BGRA -> contiguous BGR conversion, reused by everything downstream.
3. SCRFD (letterbox / engine / decode, timed separately), the IoU tracker and
   the hysteretic ``DriverSelector``.
4. 2d106det landmarks for the driver, then the model heads in dependency order:
   occlusion FIRST (it gates the eye and mouth channels), then eyes, mouth,
   head pose, gaze, objects, identity.
5. A ``DriverSignals`` is assembled and handed to ``DriverMonitor.update``,
   which returns alert EDGES.  Those edges are the safety record: they go to the
   JSONL event log, the Prometheus counters, the GPIO critical line and the clip
   ring trigger.
6. Overlay, video writer, health and metrics - all opt-in, all skipped entirely
   when nothing consumes them.

Clocks
------
The state machine is driven by MEDIA time on a file/test source (the GStreamer
buffer PTS) and by MONOTONIC time on a live camera.  A wall clock on replay runs
at the consumer's speed, so a 0.33 s blink at 8 FPS processing accumulated over
a second of "closed" and every duration threshold fired roughly three times too
eagerly.  When no frame arrives at all the clock is still advanced by the
wall-clock delta, so a frozen camera produces FACE_LOST instead of silence.

Failure behaviour
-----------------
* A bad config never reaches the loop: :func:`dms.config.load_config` raises.
* A capture failure is a supervised state, not an exception.  The unit keeps
  running, reports ``camera=failed``, serves ``/healthz`` 503 and stops petting
  the systemd watchdog, so a permanently dead camera escalates to a restart.
* An engine that fails to load degrades that capability (or, for the face
  detector, disables perception) and raises ENGINE_FAIL; ``require_engines``
  makes it fatal instead.  An engine whose sha256 or size disagrees with
  ``engines/MANIFEST.json`` is a swapped plan and is ALWAYS fatal, before READY,
  whatever ``require_engines`` says - see :func:`engine_swap_detected`.  Every
  engine the profile will load is hashed, not just the two loaded here; the list
  comes from :func:`engines_to_verify`.
* An exception inside the per-frame perception path is counted, published on
  ``/healthz`` and skips the frame.  After
  :data:`MAX_CONSECUTIVE_INFER_FAILURES` in a row the process exits non-zero so
  systemd restarts it rather than running blind.
* Teardown (events, GPIO, clips, engines, capture, health) always runs.

The freeze gate
---------------
``perf.gate_e2e_p95_ms`` / ``perf.gate_peak_rss_mb`` are evaluated on EVERY run,
on every health refresh, by :meth:`DmsApp._evaluate_gate` - not only when
``DMS_LATENCY_BUDGET_OUT`` asks for the JSON artifact, which is how a shipped
profile came to miss its own gate by 6-12 ms while ``/healthz`` reported
``ok: true, degraded: false``.  The verdict reaches an operator three ways: the
``dms_perf_gate_*`` gauges on ``/metrics``, a ``perf gate PASS/FAIL ...`` note on
``/healthz`` (with ``degraded: true`` once a decided miss has persisted for
:data:`GATE_DEGRADE_REFRESHES` refreshes), ``latency_ms.gate_limit_p95``
alongside ``latency_ms.e2e_p95`` in the same ``/healthz`` body so the miss is
machine-readable on the first refresh rather than after the degrade dwell, and
one ``PERF_GATE PASS|FAIL|UNDECIDED`` line at shutdown.  All four are derived
from :func:`_gate_verdict`, so they cannot disagree - they used to: a short run
printed ``PERF_GATE UNDECIDED`` and ``wrote ... gate=FAIL`` about the same
window, because ``gate['pass']`` is False for an unjudgeable window as well as
for a missed one.

MEASURED on this board 2026-09-13, 300 replay frames, n=3 per profile, one
process at a time under the GPU mutex: the gate is MISSED by the shipped
profiles on real clips.  ``default.yaml`` runs e2e p95 median 79.6 ms against
its own 66.0 ms limit on the shipped SCRFD-640 detector, and 78.5 ms on the
SCRFD-512 engine that was briefly shipped to buy the pass.  Run-to-run spread
is 7-16 ms, wider than the margin the gate is judged by, so a single PASS is
not evidence of a pass.  The app does not round that away: it reports the miss
and the operator decides.  See docs/performance.md section 7.

Units: seconds for time, milliseconds for latency, MiB for memory, degrees for
angles.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from dms.capture import frame_bgr, frame_pts_ns, validate_source
from dms.config.schema import load_config
from dms.io import metrics
from dms.io.clips import ClipRing
from dms.io.events import default_events_path, from_config as event_log_from_config
from dms.io.gpio_alert import GpioAlert
from dms.io.health import HealthServer, HealthState
from dms.io.sd_notify import (
    notify_ready,
    notify_status,
    notify_stopping,
    notify_watchdog,
    watchdog_ping_interval_s,
)
from dms.io.thermal import gpu_clock_mhz, ram_used_mb, read_temps, thermal_throttle, zone_report
from dms.jsonlog import LOG_LEVELS, setup_logging
from dms.perf import LatencyBudget, peak_rss_mb, sample_board
from dms.perf.budget import MIN_SAMPLES as GATE_MIN_SAMPLES
from dms.runtime import (
    BgraToBgr,
    CaptureState,
    CaptureSupervisor,
    Letterbox,
    ManifestError,
    RollingStats,
    check_runtime,
)
from dms.state.alerts import AlertEvent, AlertType, Severity
from dms.types import DriverSignals

log = logging.getLogger("dms.app")

# --- freeze-gate metrics.  Registered here rather than in ``dms.io.metrics``
# because the gate is an application-level verdict over the latency budget.
# ``Registry.gauge`` returns the existing family on a re-import, so declaring
# them at module scope is safe.
PERF_GATE_PASS = metrics.REGISTRY.gauge(
    "dms_perf_gate_pass",
    "1 while the freeze gate (e2e p95 and peak RSS) is met over the live window, "
    "0 while it is missed. Absent until the window holds enough samples to judge.",
)
PERF_GATE_E2E_P95_MS = metrics.REGISTRY.gauge(
    "dms_perf_gate_e2e_p95_ms",
    "End-to-end p95 latency the gate is judged on, milliseconds.",
)
PERF_GATE_E2E_P95_LIMIT_MS = metrics.REGISTRY.gauge(
    "dms_perf_gate_e2e_p95_limit_ms", "Configured end-to-end p95 limit, milliseconds."
)
PERF_GATE_PEAK_RSS_MB = metrics.REGISTRY.gauge(
    "dms_perf_gate_peak_rss_mb", "Peak resident set size observed this run, MiB."
)
PERF_GATE_SAMPLES = metrics.REGISTRY.gauge(
    "dms_perf_gate_e2e_samples",
    "e2e samples in the window behind the current gate verdict.",
)

#: Consecutive per-frame inference failures tolerated before the process exits.
#: Skipping a frame is the right answer to a transient CUDA hiccup; skipping
#: every frame forever is a unit that looks alive and monitors nothing.
MAX_CONSECUTIVE_INFER_FAILURES = 30

#: How often the health snapshot, board sensors and metrics are refreshed.
#: A wall-clock gate, not a frame count: at 7 FPS a 15-frame gate lagged
#: ``/healthz`` by over two seconds.
HEALTH_REFRESH_S = 0.5

#: Frames the pipeline must have processed before CLOCKS_LOW is believed.  The
#: Tegra GPU governor idles at 114 MHz and ramps under load, so the clock read
#: on frame 1 says nothing about the clock the run actually gets.
CLOCKS_SETTLE_FRAMES = 60

#: Clock samples (one per health refresh, so 2 Hz) needed before CLOCKS_LOW is
#: believed.  MEASURED under sustained DMS load on this board by sampling
#: ``/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq`` at 50 Hz for 8 s:
#: 306 MHz 61% of the time, 204 MHz 23%, 114.75 MHz 16%.  The pipeline's GPU duty
#: cycle is low enough that a handful of 2 Hz samples lands on an idle gap and
#: reports a peak of 204 MHz - which is how this warning came to fire on 100% of
#: runs.  Twenty samples make a missed 306 MHz peak a 1e-8 event.
CLOCKS_SETTLE_SAMPLES = 20

#: Clamp for a video writer frame rate derived from measurement.
WRITER_FPS_RANGE = (1.0, 60.0)

#: Consecutive health refreshes a decided gate miss must persist before the unit
#: reports itself degraded.  At :data:`HEALTH_REFRESH_S` = 0.5 s this is 10 s -
#: long enough that one slow clip section does not flap the flag, short enough
#: that a unit that is genuinely over budget says so within a breath.
GATE_DEGRADE_REFRESHES = 20

#: How often the operator-facing progress line is logged, in seconds.  The
#: field order of that line and of the final summary is part of the contract
#: with ``scripts/run_all_replay.py``, which parses both.
PROGRESS_LOG_S = 2.0

_STOP = threading.Event()
_RELOAD = threading.Event()


def _handle_stop(signum, frame) -> None:  # pragma: no cover - signal path
    _STOP.set()


def _handle_hup(signum, frame) -> None:  # pragma: no cover - signal path
    _RELOAD.set()


def install_signal_handlers() -> None:
    """SIGINT/SIGTERM stop the loop, SIGHUP reopens the event log.

    Installed before the config is read and before ``check_runtime`` hashes the
    engines, so a Ctrl-C during a five-second startup is honoured.
    """
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_hup)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dms", description="Xavier NX Driver Monitoring System")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--source", default="", help="override source.path and force file replay")
    p.add_argument("--max-frames", type=int, default=0, help="exit after N frames (0 = until EOS)")
    p.add_argument(
        "--detect",
        dest="detect",
        action="store_true",
        default=None,
        help="force the perception stack on (it is on by default)",
    )
    p.add_argument(
        "--no-detect",
        dest="detect",
        action="store_false",
        help="capture only: no engines are loaded and no alerts are produced",
    )
    p.add_argument("--save-preview", default="", help="write the last annotated JPEG")
    p.add_argument("--save-video", default="", help="write an annotated MJPG avi")
    p.add_argument(
        "--calibrate-forward",
        action="store_true",
        help="write median pose/EAR to the profile's calibration file and exit 0",
    )
    p.add_argument("--events", default="", help="override the AlertEvent JSONL path")
    p.add_argument("--log-level", default="INFO", choices=list(LOG_LEVELS))
    return p


# --------------------------------------------------------------------- helpers


#: Stage p95s republished on /healthz alongside the end-to-end figure.  Chosen
#: because they are the ones an operator can act on: the detector, the landmark
#: warp, the head pose and the state machine.
HEALTH_STAGES = ("detect_gpu", "landmarks", "headpose", "fusion")


def _percentile_map(budget: LatencyBudget, infer: RollingStats) -> Dict[str, float]:
    """``{stage_pXX: ms}`` for ``/healthz`` and the Prometheus stage gauges.

    ``e2e_*`` covers one WHOLE frame - capture, inference, overlay and IO - and
    is the number the 66 ms freeze gate is defined on.  ``infer_*`` covers the
    perception span alone.  The previous build published an infer-only number
    under the ``e2e_p95`` label, which made the gate unmeasurable.
    """
    snap = budget.timer.snapshot()
    out = {}  # type: Dict[str, float]
    e2e = snap.get("e2e")
    if e2e is not None:
        out["e2e_p50"] = round(e2e.p50, 1)
        out["e2e_p95"] = round(e2e.p95, 1)
        out["e2e_p99"] = round(e2e.p99, 1)
        # The limit travels with the measurement.  ``/healthz`` has no
        # first-class ``perf_gate`` object (that belongs in dms/io/health.py,
        # which this workstream does not own), so without this a scraper sees
        # ``e2e_p95: 79.6`` and has no way to know it is a miss until the
        # 10 s degrade dwell flips ``degraded``.  Emitted only once there is
        # an e2e sample, so an empty window stays an empty map.
        #
        # Named ``gate_limit_p95`` and not ``e2e_p95_limit`` because
        # ``HealthState.publish_metrics`` splits every key in this map into
        # ``dms_stage_latency_ms{stage=...,quantile=...}``: the latter spells
        # itself as a pipeline STAGE called "e2e_p95_limit", which is not a
        # stage and does not take 66 ms.  This spelling yields
        # ``{stage="gate_limit",quantile="p95"}``, which says what it is.
        out["gate_limit_p95"] = round(float(budget.gate_e2e_p95_ms), 1)
    if len(infer):
        out["infer_p50"] = round(infer.p50, 1)
        out["infer_p95"] = round(infer.p95, 1)
    lag = snap.get("capture_lag")
    if lag is not None:
        out["capture_lag_p95"] = round(lag.p95, 1)
    for name in HEALTH_STAGES:
        st = snap.get(name)
        if st is not None:
            out["%s_p95" % name] = round(st.p95, 1)
    return out


def health_endpoint_label(bind: str, port: Optional[int], server_built: bool) -> str:
    """What the READY line may honestly say about the health endpoint.

    ``port`` is what :meth:`HealthServer.start` returned: ``None`` means the bind
    was refused.  The READY line used to print the configured ``bind:port``
    whichever it was, so a deploy script that greps READY was told health was up
    on 127.0.0.1:8088 while nothing was listening there.
    """
    if not server_built:
        return "off"
    if port is None:
        return "disabled(bind refused)"
    return "%s:%s" % (bind, port)


def manifest_error_text(exc: BaseException) -> str:
    """The operator string for a failed engine-provenance check.

    :class:`dms.runtime.manifest.ManifestError` already begins with its own
    contractual code - ``ENGINE_FAIL`` for a missing, unlisted or hash-mismatched
    file, ``ENGINE_VERSION_MISMATCH`` for a runtime or manifest disagreement.
    Prefixing all of them with ENGINE_VERSION_MISMATCH produced
    ``ENGINE_VERSION_MISMATCH ENGINE_FAIL missing <path>`` and sent operators
    hunting a hash problem that did not exist.  An OSError carries no code, so it
    is labelled ENGINE_FAIL - the check could not be completed.
    """
    msg = str(exc)
    if msg.startswith(("ENGINE_FAIL", "ENGINE_VERSION_MISMATCH")):
        return msg
    return "ENGINE_FAIL %s" % msg


def engine_swap_detected(msg: str) -> bool:
    """True when a manifest failure means the file on disk was SWAPPED.

    ``dms.runtime.manifest`` reports a recorded-hash or recorded-size
    disagreement as ``ENGINE_FAIL sha256 mismatch`` / ``ENGINE_FAIL size
    mismatch``.  That is categorically different from an engine that is absent or
    unlisted: the plan is present and is not the plan the manifest was built
    from.  Measured on this board: one flipped byte in mwc_n_48 reached READY and
    then killed the process with CUDA error 700 mid-frame, so continuing is not
    an option even when ``require_engines`` is false.
    """
    return "sha256 mismatch" in msg or "size mismatch" in msg


def engines_to_verify(cfg) -> List[str]:
    """Every engine file this configuration will actually deserialise.

    Two of them are loaded directly by :meth:`DmsApp._setup_perception` (the
    face detector and the 106-point landmark net); the rest are resolved from
    MANIFEST *roles* by :class:`dms.infer.registry.ModelRegistry`.  Listing only
    the first two - which is what this app did until now - left seven engines,
    among them the head-pose net and both occlusion gates, loading with no hash
    check at all, so a swapped plan reached a safety decision unverified.

    The role selection mirrors ``ModelRegistry._engines_for`` exactly: a
    capability whose backend is not ``model`` loads nothing, a licence-gated
    capability loads nothing while ``research_models_allowed`` is false, and for
    interchangeable roles only the first one present on disk is loaded - so only
    that one is listed.  Paths that do not exist are still returned: a missing
    required engine must be reported by the manifest check, not skipped here.
    """
    from dms.infer.registry import BACKEND_MODEL, CAPABILITIES, ModelRegistry

    out = [cfg.models.face.engine, cfg.models.landmarks.engine]
    try:
        reg = ModelRegistry(
            backends=cfg.models.backend_map(),
            capability_cfg=cfg.models.capability_cfg(),
            research_models_allowed=cfg.models.research_models_allowed,
        )
    except (ValueError, ImportError) as exc:  # bad backend name / no TRT here
        log.warning("cannot enumerate registry engines for verification: %s", exc)
        return out
    for cap, spec in CAPABILITIES.items():
        if reg.backends[cap] != BACKEND_MODEL:
            continue
        if spec.licence_gated and not reg.research_models_allowed:
            continue
        for role in spec.roles:
            path = reg.role_path(role)
            if path is None:
                continue
            out.append(path)
            if spec.alternatives:
                break
    return out


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return float(s[n // 2])
    return 0.5 * (float(s[n // 2 - 1]) + float(s[n // 2]))


class DmsApp(object):
    """Owns every component for one process lifetime.

    Construction is cheap and side-effect free; :meth:`setup` acquires
    resources and :meth:`teardown` releases them.  ``setup`` returns an exit
    code on a fatal problem and ``None`` when the loop may start.
    """

    def __init__(self, cfg, args) -> None:
        self.cfg = cfg
        self.args = args
        self.detect = bool(args.detect) if args.detect is not None else True

        # -- components, all None until setup()
        self.health: Optional[HealthState] = None
        self.hsv: Optional[HealthServer] = None
        self.gpio: Optional[GpioAlert] = None
        self.events = None
        self.clips: Optional[ClipRing] = None
        self.sup: Optional[CaptureSupervisor] = None
        self.registry = None
        self.detector = None
        self.landmarker = None
        self.face_engine = None
        self.lmk_engine = None
        self.tracker = None
        self.selector = None
        self.monitor = None
        self.overlay = None
        self.budget: Optional[LatencyBudget] = None
        self.to_bgr = BgraToBgr()
        #: The one full-frame letterbox on the hot path
        #: (:class:`dms.runtime.preprocess.Letterbox`), built in
        #: :meth:`_setup_perception` once the engine's input size is known.
        self.letterbox: Optional[Letterbox] = None
        self.writer = None
        self.preview = None
        #: Perception-span latency, kept out of the budget so that wrapping the
        #: individual stages does not double-count them in the accounting block.
        self.infer_ms = RollingStats(maxlen=cfg.perf.window)
        #: track ids whose per-track model caches are still live.
        self._live_tracks = set()

        # -- loop state
        self.frames = 0
        self.face_hits = 0
        self.last_frame_id = 0
        self.infer_failures = 0
        self.infer_failures_total = 0
        self.thermal_on = False
        self.clocks_low_logged = False
        #: Highest GPU clock seen so far, MHz.  CLOCKS_LOW is judged on this
        #: peak, and only once the pipeline has settled and enough samples exist:
        #: read once on frame 1 against a 300 MHz threshold, the warning fired on
        #: every single run while the board was in fact reaching 306 MHz.
        self.gpu_clock_max_mhz = 0.0
        #: How many clock samples that peak is drawn from.
        self.gpu_clock_samples = 0
        #: Freeze-gate state: last verdict, consecutive decided misses, and
        #: whether the miss is currently long enough to count as degraded.
        self._gate_last: Dict[str, Any] = {}
        self._gate_fail_streak = 0
        self._gate_degraded = False
        self._health_port: Optional[int] = None
        self.alert_names: List[str] = []
        self.calib_yaw: List[float] = []
        self.calib_pitch: List[float] = []
        self.calib_ear: List[float] = []
        self._t_media_s = 0.0
        self._t_media_mono = 0.0
        self._last_health_s = 0.0
        self._last_progress_s = 0.0
        self._board_sensors_warned = False
        self._last_wd_s = 0.0
        self._last_wd_frame = -1
        self._wd_interval_s = 5.0
        self._events_path = ""
        self._exit_code = 0

    # ------------------------------------------------------------------ setup
    def setup(self) -> Optional[int]:
        cfg, args = self.cfg, self.args

        # --- health first: it is the surface every later failure is reported on.
        self.health = HealthState(stale_after_s=cfg.health.stale_after_s)
        self.health.source = cfg.source.type
        self.health.dev_replay = cfg.is_replay
        self.health.camera = "starting"
        self.health.privacy = {
            "record_faces": cfg.privacy.record_faces,
            "telemetry": cfg.privacy.telemetry,
        }
        self.health.thermal_zones = zone_report()
        self.health.set_build(trt_version=_trt_version(), config=os.path.abspath(args.config))
        if not cfg.forward_zero_calibrated and not _zero_is_set(cfg.forward_zero):
            self.health.note(
                "forward_zero is not calibrated for this installation: GAZE_AWAY and "
                "the gaze half of DISTRACTION are suppressed"
            )

        try:
            self.hsv = HealthServer.from_config(self.health, cfg)
            port = self.hsv.start()
        except RuntimeError as exc:
            log.error("HEALTH_FATAL %s", exc)
            return 1
        if port is None and cfg.health.required:
            log.error("HEALTH_FATAL health.required is set but the listener did not start")
            return 1
        # start() returns None when the bind was refused.  Remember that: the
        # READY line used to advertise the configured endpoint either way, so a
        # deploy script that greps READY was told health was up when nothing was
        # listening on the port.
        self._health_port = port
        if port is None:
            self.health.note(
                "health endpoint is NOT listening: bind %s:%s was refused"
                % (cfg.health.bind, cfg.health.port)
            )

        # --- manifest / engine provenance.  EVERY engine this config will load is
        # hashed before READY - all nine on a full profile, not just the two the
        # app loads by hand - because the hash check is the only thing standing
        # between a swapped .engine and a safety decision.  Measured cost of the
        # full set on this board: see the sha256 timing logged below.
        engines = engines_to_verify(cfg)
        t_hash = time.monotonic()
        try:
            check_runtime(engines, require_hash=cfg.require_engines)
            self.health.manifest = {"ok": True, "error": ""}
            log.info(
                "engine provenance: %d engines verified in %.2fs (require_hash=%s): %s",
                len(engines),
                time.monotonic() - t_hash,
                cfg.require_engines,
                ", ".join(sorted(os.path.basename(p) for p in engines if p)),
            )
        except (ManifestError, OSError) as exc:
            msg = manifest_error_text(exc)
            self.health.manifest = {"ok": False, "error": msg}
            log.error("%s", msg)
            # A recorded hash or size that does not match is a SWAPPED engine, and
            # that is fatal whatever require_engines says: the file on disk is not
            # the file this manifest was built from.  Measured: a single flipped
            # byte in mwc_n_48 got as far as READY and then took the process down
            # with CUDA error 700 mid-frame.  A missing or unlisted engine is a
            # different, softer failure and stays governed by require_engines.
            if engine_swap_detected(msg):
                log.error(
                    "ENGINE_TAMPER refusing to start: an engine on disk differs "
                    "from engines/MANIFEST.json. Rebuild it or restore the plan "
                    "the manifest records."
                )
                return 1
            if cfg.require_engines:
                return 1
            log.warning("continuing with unverified engines (require_engines is false)")

        # --- ops surfaces
        self.gpio = GpioAlert(cfg.alerts.gpio_pin, buzzer=cfg.alerts.buzzer)
        self.gpio.arm_signal_handlers()

        if cfg.events.enabled or args.events:
            self._events_path = args.events or default_events_path(
                cfg.source.dev, cfg.events.path
            )
            try:
                self.events = event_log_from_config(
                    cfg, override_path=args.events, on_disk_full=self._on_disk_full
                )
            except OSError as exc:
                log.error("EVENT_LOG_FATAL %s", exc)
                if cfg.events.strict:
                    return 1
                self.events = None
            if self.events is not None:
                self.health.events = self.events.stats()
        else:
            log.warning("events.enabled is false: no durable alert record will be written")

        self.clips = ClipRing.from_config(cfg, on_disk_full=self._on_disk_full)
        self.clips.start()
        self.health.clips = self.clips.snapshot()

        self.health.disk_paths = sorted(
            set(
                p
                for p in (
                    os.path.dirname(self._events_path) or ".",
                    cfg.clips.dir if cfg.clips.enabled else "",
                )
                if p
            )
        )
        self.health.refresh_disk()

        # --- perf
        self.budget = LatencyBudget.from_config(cfg, label=os.path.basename(args.config))
        self.budget.set_context(
            config=os.path.abspath(args.config),
            source=cfg.source.type,
            clip=cfg.source.path,
            resolution="%dx%d" % (cfg.source.width, cfg.source.height),
            detect=self.detect,
            note=(
                "the per-head crops are cut inside each wrapper, so their cost is "
                "already inside that head's stage and the 'crop' stage has no samples"
            ),
        )

        # --- overlay (returns None before touching a pixel when nothing wants it)
        from dms.viz import OverlayRenderer

        self.overlay = OverlayRenderer.from_flags(
            save_video=bool(args.save_video),
            save_preview=bool(args.save_preview),
            display=cfg.display.enabled,
            scale=cfg.display.scale,
        )

        # --- perception
        if self.detect:
            rc = self._setup_perception()
            if rc is not None:
                return rc
        else:
            log.warning("perception disabled (--no-detect): capture and health only")
            self.health.engines = {"face": "off", "landmarks": "off"}

        # --- capture
        first_ff = cfg.camera.first_frame_timeout_s or cfg.camera.stall_timeout_s
        self.sup = CaptureSupervisor.for_config(
            cfg,
            backoff_s=cfg.camera.backoff_s,
            retry_forever=cfg.camera.retry_forever,
            stall_timeout_s=cfg.camera.stall_timeout_s,
            first_frame_timeout_s=first_ff,
            starved_below_fps=cfg.camera.starved_below_fps,
            starved_for_s=cfg.camera.starved_for_s,
        )
        started = self.sup.start()
        self.health.camera = self.sup.camera_status
        if not started:
            log.error("CAMERA_FAIL capture did not start: %s", self.sup.last_error)
            self._alert(AlertType.CAMERA_FAIL, Severity.CRITICAL, "enter", time.monotonic())
            if cfg.camera.fail_fatal:
                log.error("camera.fail_fatal is set: exiting so the supervisor restarts us")
                return 1
            log.warning(
                "camera.fail_fatal is false: staying up on the retry ladder. The unit "
                "reports camera=%s, /healthz answers 503 and the watchdog is NOT pinged.",
                self.sup.camera_status,
            )

        self._wd_interval_s = watchdog_ping_interval_s(default=5.0)
        notify_ready("dms %s source=%s detect=%s" % (_version(), cfg.source.type, self.detect))
        log.info(
            "READY source=%s detect=%s engines=%s events=%s health=%s watchdog_every=%.1fs",
            cfg.source.type,
            self.detect,
            self.health.engines,
            self._events_path or "off",
            health_endpoint_label(cfg.health.bind, self._health_port, self.hsv is not None),
            self._wd_interval_s,
        )
        return None

    def _setup_perception(self) -> Optional[int]:
        cfg = self.cfg
        from dms.infer.landmarks import Landmark106
        from dms.infer.registry import ModelRegistry
        from dms.infer.scrfd import ScrfdDetector
        from dms.runtime import EngineLoadError, TrtEngine
        from dms.state.driver_state import DriverMonitor
        from dms.track.driver_select import DriverSelector, SelectorConfig
        from dms.track.iou_tracker import IouTracker

        try:
            log.info("loading detector %s", cfg.models.face.engine)
            self.face_engine = TrtEngine(cfg.models.face.engine)
            self.detector = ScrfdDetector(
                self.face_engine, conf=cfg.models.face.conf, iou=cfg.models.face.iou
            )
            # Detector preprocessing runs through the allocation-free
            # dms.runtime.preprocess.Letterbox, sized from the engine so a 512
            # or a 640 plan both work.  MEASURED on this board at 1280x720:
            # blob p50 4.99 ms vs 6.70 ms (640) and 3.67 ms vs 5.45 ms (512) for
            # ScrfdDetector.preprocess's allocating chain, with a max absolute
            # difference of 1.19e-07 between the two blobs.
            net_w, net_h = self.detector.input_size
            self.letterbox = Letterbox.for_scrfd((net_h, net_w))
            if tuple(cfg.capture.letterbox) != (net_w, net_h):
                log.warning(
                    "LETTERBOX_MISMATCH capture.letterbox is %s but %s takes %dx%d; "
                    "the engine wins. Fix the profile so it describes what runs.",
                    tuple(cfg.capture.letterbox),
                    os.path.basename(cfg.models.face.engine),
                    net_w,
                    net_h,
                )
        except (EngineLoadError, OSError, RuntimeError) as exc:
            log.error("ENGINE_FAIL face detector %s: %s", cfg.models.face.engine, exc)
            metrics.ENGINE_ERRORS_TOTAL.inc(engine="face", kind="load")
            self._alert(AlertType.ENGINE_FAIL, Severity.CRITICAL, "enter", time.monotonic())
            self.health.set_model("face", loaded=False, ok=False)
            if cfg.require_engines:
                return 1
            log.warning("perception disabled: the face detector is the root of the pipeline")
            self.detect = False
            self.health.engines = {"face": "missing", "landmarks": "off"}
            self.health.note("face detector failed to load: running capture-only")
            return None
        self.health.set_model("face", loaded=True, ok=True)

        try:
            log.info("loading landmarks %s", cfg.models.landmarks.engine)
            self.lmk_engine = TrtEngine(cfg.models.landmarks.engine)
            self.landmarker = Landmark106(self.lmk_engine)
            self.health.set_model("landmarks", loaded=True, ok=True)
        except (EngineLoadError, OSError, RuntimeError) as exc:
            log.error("ENGINE_FAIL landmarks %s: %s", cfg.models.landmarks.engine, exc)
            metrics.ENGINE_ERRORS_TOTAL.inc(engine="landmarks", kind="load")
            self._alert(AlertType.ENGINE_FAIL, Severity.WARN, "enter", time.monotonic())
            self.health.set_model("landmarks", loaded=False, ok=False)
            if cfg.require_engines:
                return 1
            self.health.note("landmarks engine missing: eye/mouth/pose run on nothing")

        self.registry = ModelRegistry(
            backends=cfg.models.backend_map(),
            capability_cfg=cfg.models.capability_cfg(),
            research_models_allowed=cfg.models.research_models_allowed,
        ).load()

        engines = {
            "face": "gpu" if self.detector is not None else "missing",
            "landmarks": "gpu" if self.landmarker is not None else "missing",
        }
        engines.update(self.registry.describe())
        self.health.engines = engines
        for cap, mh in self.registry.health_map().items():
            self.health.set_model(cap, loaded=mh.loaded, ok=mh.loaded or None)
        for role, err in self.registry.load_errors().items():
            self.health.note("engine role %s failed to load: %s" % (role, err))
            metrics.ENGINE_ERRORS_TOTAL.inc(engine=role, kind="load")
        degraded = self.registry.degraded_capabilities()
        if degraded:
            log.warning("MODEL_DEGRADED capabilities not running a model: %s", degraded)
        log.info("model registry: %s", self.registry.describe())
        log.info("engine files: %s", self.registry.engine_files())

        self.tracker = IouTracker()
        self.selector = DriverSelector(
            cfg.driver_roi, SelectorConfig(fallback=cfg.driver_fallback)
        )
        self.monitor = DriverMonitor(cfg)
        return None

    # ------------------------------------------------------------------- loop
    def run(self) -> int:
        cfg, args = self.cfg, self.args
        sup, budget, health = self.sup, self.budget, self.health
        overlay = self.overlay
        max_frames = int(args.max_frames or 0)

        # Seed the state-machine clock so the ticks before frame #1 advance from
        # the same origin the frames will use.
        self._t_media_mono = time.monotonic()
        self._t_media_s = 0.0 if cfg.is_replay else self._t_media_mono

        while not _STOP.is_set():
            if _RELOAD.is_set():
                _RELOAD.clear()
                self._reopen_logs()

            t_frame = time.monotonic()
            with budget.stage("capture"):
                frame = sup.read(timeout_s=1.0)

            self._drain_capture_events()

            if frame is None:
                if sup.finished:
                    log.info("source finished after %d frames", self.frames)
                    break
                # A None is "nothing this tick".  It is not by itself a fault,
                # but the state machine still has to be advanced or a frozen
                # camera produces silence rather than FACE_LOST.
                self._tick_no_frame(t_frame)
                continue

            if not frame.ok:
                # One-shot failure sentinel from the supervisor.
                self._tick_no_frame(t_frame)
                continue

            bgra = frame.full_bgra
            if bgra is None:
                self._tick_no_frame(t_frame)
                continue

            self.frames += 1
            self.last_frame_id = int(frame.frame_id)
            budget.record("capture_lag", (time.monotonic_ns() - frame.t_mono_ns) / 1e6)

            with budget.stage("bgra"):
                pre = frame_bgr(frame)
                bgr = pre if pre is not None and pre.flags["C_CONTIGUOUS"] else self.to_bgr(bgra)

            t_s = self._frame_clock(frame, t_frame)

            events: List[AlertEvent] = []
            faces: List[Any] = []
            driver_face = None
            landmarks = None
            pose = None
            ear = None
            mar = None
            track_id = None
            if self.detect and self.detector is not None:
                t_infer = time.monotonic()
                try:
                    faces, driver_face, track_id, landmarks, signals = self._perceive(bgr, t_s)
                    self.infer_ms.add((time.monotonic() - t_infer) * 1000.0)
                    self.infer_failures = 0
                except Exception as exc:  # noqa: BLE001 - degrade the frame, not the unit
                    self.infer_failures += 1
                    self.infer_failures_total += 1
                    metrics.ENGINE_ERRORS_TOTAL.inc(engine="pipeline", kind=type(exc).__name__)
                    health.set_model("face", ok=False)
                    log.error(
                        "inference failed on frame %d (%d consecutive): %s",
                        self.frames,
                        self.infer_failures,
                        exc,
                        exc_info=self.infer_failures == 1,
                    )
                    if self.infer_failures >= MAX_CONSECUTIVE_INFER_FAILURES:
                        log.error(
                            "ENGINE_FAIL %d consecutive inference failures; exiting so the "
                            "service manager restarts a clean process",
                            self.infer_failures,
                        )
                        self._alert(AlertType.ENGINE_FAIL, Severity.CRITICAL, "enter", t_s)
                        self._exit_code = 1
                        break
                    self._tick_no_frame(t_frame)
                    continue

                self.face_hits += len(faces)
                pose = signals.pose
                ear = _ear_of(signals)
                mar = signals.mouth.mar if signals.mouth is not None else None
                with budget.stage("fusion"):
                    events = self.monitor.update(
                        t_s,
                        present=driver_face is not None,
                        track_id=track_id,
                        signals=signals,
                        degraded=self.registry.degraded_capabilities(),
                        t_last_match=self.selector.t_last_match_s,
                    )
                self._publish_alerts(events, t_s)
                self._collect_calibration(signals)
            else:
                # Capture-only: nothing to fuse, but the loop still has to be
                # observable, so the frame is counted and health refreshed.
                pass

            vis = None
            if overlay.enabled:
                with budget.stage("overlay"):
                    vis = self._render(
                        bgr, faces, driver_face, landmarks, ear, mar, pose, track_id
                    )
                self.preview = vis

            if self.args.save_video or self.clips.active:
                with budget.stage("io"):
                    self._write_video(vis)
                    if self.clips.active:
                        self.clips.push(bgr, t_s)
                        self.clips.tick(t_s)

            budget.record("e2e", (time.monotonic() - t_frame) * 1000.0)
            budget.mark_frame()
            self._periodic(t_frame, t_s)

            if max_frames and self.frames >= max_frames:
                log.info("reached --max-frames %d", max_frames)
                break

        return self._exit_code

    # ------------------------------------------------------------ perception
    def _perceive(self, bgr, t_s: float):
        """Run every model for one frame and assemble a :class:`DriverSignals`.

        Raises on an engine error; the caller converts that into a skipped frame
        and, past a threshold, a non-zero exit.
        """
        cfg = self.cfg
        budget = self.budget
        reg = self.registry
        det = self.detector
        h, w = bgr.shape[:2]

        with budget.stage("letterbox"):
            if self.letterbox is not None:
                blob, geom = self.letterbox.blob(bgr)
                factor = geom.factor
            else:  # no engine-derived size (a detector injected by a test)
                blob, factor = det.preprocess(bgr)
        with budget.stage("detect_gpu"):
            raw = det.engine.infer({det.engine.input_name: blob})
        with budget.stage("detect_post"):
            faces = det.postprocess(raw, factor, orig_hw=(h, w))

        with budget.stage("track"):
            tracks = self.tracker.update(faces, t_s=t_s)
            live = [t for t in tracks if t.face is not None]
            pick = self.selector.select(tracks, (w, h), t_s=t_s)
        driver_face = pick.face
        track_id = pick.track_id

        signals = DriverSignals(track_id=-1 if track_id is None else int(track_id))
        landmarks = None
        if driver_face is None:
            # No driver in the seat: run nothing per-head.  Every channel stays
            # UNAVAILABLE, which is what makes FACE_LOST and the honesty alerts
            # fire instead of a stale "eyes open".
            return faces, None, None, None, signals

        crop_box = driver_face.box_for_crop()

        # Occlusion first: it decides whether the eye and mouth channels may be
        # believed at all this frame.
        with budget.stage("occlusion"):
            occ = reg.occlusion.infer(bgr, crop_box, track_id=track_id)
        signals.occlusion = occ

        if self.landmarker is not None:
            with budget.stage("landmarks"):
                landmarks = self.landmarker.infer_one(bgr, crop_box)

        if landmarks is not None:
            with budget.stage("eyestate"):
                left, right = reg.eyestate.infer(bgr, landmarks, eyes_reliable=occ.eyes_reliable)
            signals.left_eye, signals.right_eye = left, right
            with budget.stage("mouthstate"):
                signals.mouth = reg.mouthstate.infer(
                    bgr, landmarks, mouth_reliable=occ.mouth_reliable
                )

        with budget.stage("headpose"):
            signals.pose = reg.headpose.infer(
                bgr, crop_box, landmarks, kps5=driver_face.kps, frame_wh=(w, h)
            )

        if reg.state("gaze") == "model":
            with budget.stage("gaze"):
                signals.gaze = reg.gaze.infer(bgr, driver_face.xyxy)

        if reg.state("objects") == "model":
            with budget.stage("objects"):
                dets = reg.objects.detect(bgr)
                signals.phone_conf = reg.objects.phone_confidence(dets)

        if reg.state("identity") == "model":
            with budget.stage("identity"):
                obs = reg.identity.observe(bgr, driver_face.kps, track_id)
            signals.identity = obs.label

        signals.reliable = occ.eyes_reliable and occ.mouth_reliable
        self._forget_dead_tracks(live)
        return faces, driver_face, track_id, landmarks, signals

    def _forget_dead_tracks(self, live) -> None:
        """Drop per-track model caches for ids the tracker has retired.

        Without this the occlusion and identity caches grow for the life of the
        process: one entry per track id ever seen, which on a busy multi-occupant
        clip is unbounded.
        """
        alive = set(int(t.track_id) for t in live)
        for tid in list(self._live_tracks - alive):
            self.registry.forget_track(tid)
            self._live_tracks.discard(tid)
        self._live_tracks |= alive

    # ---------------------------------------------------------------- clocks
    def _frame_clock(self, frame, t_mono: float) -> float:
        """Seconds on the timeline the state machine runs on.

        Media time (buffer PTS) for a file or test source, monotonic time for a
        live camera.  Both are recorded so :meth:`_tick_no_frame` can keep the
        clock moving while no frames arrive.
        """
        if self.cfg.is_replay:
            t_s = frame_pts_ns(frame) / 1e9
        else:
            t_s = frame.t_mono_ns / 1e9
        self._t_media_s = t_s
        self._t_media_mono = t_mono
        return t_s

    def _tick_no_frame(self, t_mono: float) -> None:
        """Advance the state machine over an interval that produced no frame.

        Without this a frozen camera is silence, not FACE_LOST: the monitor is
        only ever advanced by a frame, so no time passes for it and no dwell
        ever completes.  The clock is advanced by real elapsed time even on a
        media-time source, because the driver keeps existing while the pipeline
        does not.
        """
        if self._t_media_mono:
            t_s = self._t_media_s + max(0.0, t_mono - self._t_media_mono)
        else:
            t_s = t_mono
        self._t_media_s = t_s
        self._t_media_mono = t_mono
        # Nothing is advanced before frame #1: until capture has delivered
        # something there is no driver to have lost, and the condition that
        # matters ("the camera never started") is the supervisor's
        # first_frame_timeout -> CAMERA_FAIL, not FACE_LOST.  Ticking here would
        # also make the alert stream depend on how long TensorRT took to warm up.
        if self.monitor is not None and self.frames > 0:
            try:
                events = self.monitor.update(
                    t_s,
                    present=False,
                    track_id=None,
                    degraded=self.registry.degraded_capabilities()
                    if self.registry is not None
                    else None,
                    t_last_match=self.selector.t_last_match_s if self.selector else None,
                )
            except Exception as exc:  # noqa: BLE001 - the state machine must not kill us
                log.error("state machine raised on an empty tick: %s", exc, exc_info=True)
                events = []
            self._publish_alerts(events, t_s)
        self._periodic(t_mono, t_s)

    # ------------------------------------------------------------- reporting
    def _publish_alerts(self, events: List[AlertEvent], t_s: float) -> None:
        for ev in events:
            self._emit(ev)
        if self.monitor is not None:
            self.alert_names = [
                "%s:%s" % (k.value, v.value) for k, v in self.monitor.active.items()
            ]
            self.gpio.set_critical(self.monitor.highest_severity() == Severity.CRITICAL)

    def _alert(self, atype: AlertType, severity: Severity, edge: str, t_s: float, **extra) -> None:
        """Construct and publish an ops-layer alert the state machine cannot see."""
        self._emit(
            AlertEvent(
                t_mono_s=float(t_s),
                type=atype,
                severity=severity,
                edge=edge,
                extra=dict((k, float(v)) for k, v in extra.items()),
            )
        )

    def _emit(self, ev: AlertEvent) -> None:
        log.info(
            "ALERT %s %s %s%s",
            ev.edge,
            ev.type.value,
            ev.severity.value,
            "" if not ev.extra else " %s" % (ev.extra,),
        )
        if self.events is not None:
            # write_alert already increments the Prometheus counters.
            self.events.write_alert(ev)
        else:
            metrics.record_alert(ev)
        if ev.edge in ("enter", "escalate") and ev.severity in (Severity.WARN, Severity.CRITICAL):
            if self.clips is not None and self.clips.active:
                self.clips.trigger(ev.type.value, ev.severity, ev.t_mono_s)

    def _drain_capture_events(self) -> None:
        """Turn the supervisor's edges into AlertEvents. Once per iteration."""
        for cev in self.sup.take_events():
            atype = (
                AlertType.CAMERA_FAIL
                if cev.kind == "camera_fail"
                else AlertType.PIPELINE_STARVED
            )
            severity = Severity.CRITICAL if atype is AlertType.CAMERA_FAIL else Severity.WARN
            log.warning("CAPTURE %s %s %s", cev.kind, cev.edge, cev.detail)
            self._alert(atype, severity, cev.edge, cev.t_mono_s)

    def _on_disk_full(self, ev: AlertEvent) -> None:
        """EventLog / ClipRing hand us a ready-made DISK_FULL edge.

        The event log has already written (or tried to write) the record and
        bumped the counters, so this only logs and degrades: calling
        :meth:`_emit` here would recurse straight back into the failing writer.
        """
        log.error("DISK_FULL %s %s", ev.edge, ev.extra)
        if self.health is not None:
            self.health.degraded = ev.edge == "enter"
            self.health.note("disk full on the event or clip path")

    def _reopen_logs(self) -> None:
        log.info("SIGHUP: reopening the event log")
        if self.events is not None:
            try:
                self.events.reopen()
            except OSError as exc:
                log.error("event log reopen failed: %s", exc)

    # ---------------------------------------------------------------- output
    def _render(self, bgr, faces, driver_face, landmarks, ear, mar, pose, track_id):
        if not self.overlay.enabled:
            return None
        from dms.track.driver_select import label_occupants

        yaw_rel = None
        pose_tuple = None
        if pose is not None and pose.source.value != "unavailable":
            pose_tuple = (pose.yaw, pose.pitch, pose.roll)
            if self.monitor is not None and self.monitor.yaw_ewma is not None:
                from dms.state.fusion import angdiff

                yaw_rel = angdiff(self.monitor.yaw_ewma, self.cfg.forward_zero.yaw)
        live = [t for t in (self.tracker.matched_tracks() if self.tracker else [])]
        return self.overlay.render(
            bgr,
            faces,
            driver=driver_face,
            landmarks106=landmarks,
            ear=ear,
            mar=mar,
            pose=pose_tuple,
            track_id=track_id,
            alerts=self.alert_names,
            yaw_rel=yaw_rel,
            occupant_labels=label_occupants(live, track_id),
        )

    def _writer_fps(self) -> float:
        """Frame rate for the annotated video, in Hz.

        A file source delivers every decoded frame (the appsink does not drop),
        so the clip's own rate is correct and the saved video stays aligned with
        media-time event timestamps.  A live source drops, so the honest rate is
        the measured one; writing at the nominal 30 makes a 8 FPS recording play
        back nearly four times too fast and unalignable with events.jsonl.
        """
        if self.cfg.is_replay:
            return float(self.cfg.source.fps)
        mean = self.sup.intervals.window_mean if self.sup is not None else 0.0
        if mean <= 0.0:
            return float(self.cfg.source.fps)
        lo, hi = WRITER_FPS_RANGE
        return float(min(hi, max(lo, 1.0 / mean)))

    def _write_video(self, vis) -> None:
        if not self.args.save_video or vis is None:
            return
        if self.writer is None:
            if not self.cfg.is_replay and len(self.sup.intervals) < 10:
                return  # wait for a measured rate rather than guessing
            import cv2

            h, w = vis.shape[:2]
            fps = self._writer_fps()
            d = os.path.dirname(self.args.save_video)
            if d:
                os.makedirs(d, exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.args.save_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
            )
            if not self.writer.isOpened():
                log.error("could not open video writer %s", self.args.save_video)
                self.writer = False
            else:
                log.info("writing %s at %.1f fps (%dx%d)", self.args.save_video, fps, w, h)
        if self.writer:
            self.writer.write(vis)

    # -------------------------------------------------------------- periodic
    def _periodic(self, t_mono: float, t_s: float) -> None:
        """Health, board sensors, metrics and the systemd watchdog."""
        if t_mono - self._last_health_s >= HEALTH_REFRESH_S:
            elapsed = t_mono - self._last_health_s if self._last_health_s else 0.0
            self._last_health_s = t_mono
            self._refresh_health(t_s, elapsed)
        if t_mono - self._last_progress_s >= PROGRESS_LOG_S:
            self._last_progress_s = t_mono
            self._log_progress(t_s)
        if t_mono - self._last_wd_s >= self._wd_interval_s:
            self._last_wd_s = t_mono
            self._pet_watchdog()

    def _log_progress(self, t_s: float) -> None:
        """One human-readable progress line.

        ``frames=`` / ``fps=`` / ``infer_p95_ms=`` lead the line because
        ``scripts/run_all_replay.py`` parses them; everything after is free-form.
        """
        snap = self.budget.timer.snapshot()
        e2e = snap.get("e2e")
        log.info(
            "frames=%d fps=%.1f infer_p95_ms=%.1f e2e_p95_ms=%.1f camera=%s t=%.1fs "
            "alerts=%s",
            self.frames,
            self.sup.fps if self.sup is not None else 0.0,
            self.infer_ms.p95,
            e2e.p95 if e2e is not None else 0.0,
            self.sup.camera_status if self.sup is not None else "off",
            t_s,
            ",".join(self.alert_names) or "-",
        )

    def _refresh_health(self, t_s: float, elapsed_s: float = HEALTH_REFRESH_S) -> None:
        cfg, sup, health, budget = self.cfg, self.sup, self.health, self.budget
        gpu_c, th_c = read_temps()
        clock = gpu_clock_mhz()
        hot = thermal_throttle(gpu_c, th_c)
        if hot and not self.thermal_on:
            log.warning("THERMAL_THROTTLE gpu=%s thermal=%s", gpu_c, th_c)
            self.thermal_on = True
            self._alert(AlertType.THERMAL_THROTTLE, Severity.WARN, "enter", t_s)
        elif not hot and self.thermal_on:
            self.thermal_on = False
            self._alert(AlertType.THERMAL_THROTTLE, Severity.WARN, "exit", t_s)
        if self.thermal_on:
            metrics.THERMAL_THROTTLE_SECONDS.inc(max(0.0, elapsed_s))
        self.overlay.thermal_gate(hot)
        if clock is not None:
            self.gpu_clock_max_mhz = max(self.gpu_clock_max_mhz, float(clock))
            self.gpu_clock_samples += 1
        if (
            self.detect
            and not self.clocks_low_logged
            and self.frames >= CLOCKS_SETTLE_FRAMES
            and self.gpu_clock_samples >= CLOCKS_SETTLE_SAMPLES
            and 0.0 < self.gpu_clock_max_mhz < 300
        ):
            log.warning(
                "CLOCKS_LOW gpu_clock_mhz=%.1f (peak of %d samples over %d frames) "
                "- latency numbers taken now are not comparable with a run at full "
                "clocks",
                self.gpu_clock_max_mhz,
                self.gpu_clock_samples,
                self.frames,
            )
            self.clocks_low_logged = True

        budget.observe_rss(peak_rss_mb())
        gate, gate_degraded = self._evaluate_gate()

        snap = sup.snapshot()
        camera = snap["camera"]
        if camera == "ok" and not sup.healthy:
            # The supervisor reports RUNNING as soon as the pipeline is built,
            # which is before frame #1 exists.  Publishing "ok" there would let
            # a source that reconnects but never delivers look healthy between
            # restarts, so it is reported as still starting.
            camera = "starting"
        with health.lock:
            health.camera = camera
            health.fps = snap["fps"]
            health.frame_id = self.last_frame_id
            health.frames_total = sup.frames_total
            health.frames_dropped = snap["frames_dropped"]
            health.camera_restarts = sup.restarts
            health.latency_ms = _percentile_map(budget, self.infer_ms)
            health.ram_mb = ram_used_mb()
            health.gpu_temp_c = gpu_c
            health.thermal_c = th_c
            health.gpu_clock_mhz = clock
            # A sustained gate miss is degraded, not not-ok: the unit is still
            # watching the driver, just slower than its budget, and flipping
            # ``ok`` would make /healthz answer 503 and take a working monitor
            # out of service over a few milliseconds of tail latency.  degraded
            # is the designed signal for "running, off spec" and it reaches an
            # operator through /healthz, dms_degraded and the notes below.
            health.degraded = bool(
                hot or self.infer_failures or snap["camera_fail_active"] or gate_degraded
            )
            # /healthz has no first-class field for the gate verdict, so it is
            # published as an operator note (and as the dms_perf_gate_* gauges on
            # /metrics).  See the handoff: a `perf_gate` object in
            # HealthState.as_dict belongs in dms/io/health.py.
            health.notes = [n for n in health.notes if not n.startswith("perf gate")]
            health.notes.append(_gate_note(gate, gate_degraded))
            health.alerts_active = list(self.alert_names)
            health.ok = bool(sup.healthy) and self.infer_failures == 0
            if self.events is not None:
                health.events = self.events.stats()
            if self.clips is not None:
                health.clips = self.clips.snapshot()
        health.refresh_disk()
        if self.registry is not None:
            for cap, mh in self.registry.health_map().items():
                health.set_model(
                    cap,
                    loaded=mh.loaded,
                    ok=None if mh.consecutive_failures == 0 else False,
                    mean_ms=mh.mean_ms,
                )
        if self.monitor is not None:
            bad = sorted(k for k, v in self.monitor.unreliable.items() if v)
            with health.lock:
                health.notes = [
                    n for n in health.notes if not n.startswith("unreliable:")
                ]
                if bad:
                    health.notes.append("unreliable: " + ",".join(bad))
        health.mark_update()
        try:
            sample_board(budget, throttled=hot)
        except ImportError as exc:  # pragma: no cover - board-only helper
            if not self._board_sensors_warned:
                self._board_sensors_warned = True
                log.warning(
                    "board sensors unavailable (%s): the resources block of "
                    "latency_budget.json will be empty",
                    exc,
                )
        health.publish_metrics()

    def _evaluate_gate(self) -> Tuple[Dict[str, Any], bool]:
        """Judge the freeze gate on the live window and publish the verdict.

        Called from every health refresh, not only when
        ``DMS_LATENCY_BUDGET_OUT`` asks for a JSON artifact.  Until this ran on
        the normal path the shipped profiles missed their own
        ``perf.gate_e2e_p95_ms`` by 6-12 ms and ``/healthz`` still answered
        ``ok: true, degraded: false``.

        Returns the verdict dict and whether the miss has been sustained for
        :data:`GATE_DEGRADE_REFRESHES` refreshes.  A window with fewer than the
        budget's ``MIN_SAMPLES`` e2e samples is *undecided*: it is neither a pass
        nor a fail, and it never sets the streak.
        """
        gate = self.budget.gate()
        self._gate_last = gate
        decided = int(gate.get("e2e_samples") or 0) >= GATE_MIN_SAMPLES
        if decided and not gate["pass"]:
            self._gate_fail_streak += 1
        else:
            self._gate_fail_streak = 0
        degraded = self._gate_fail_streak >= GATE_DEGRADE_REFRESHES

        PERF_GATE_E2E_P95_LIMIT_MS.set(float(gate["e2e_p95_limit_ms"]))
        PERF_GATE_PEAK_RSS_MB.set(float(gate.get("peak_rss_mb") or 0.0))
        PERF_GATE_SAMPLES.set(float(gate.get("e2e_samples") or 0))
        if gate.get("e2e_p95_ms") is not None:
            PERF_GATE_E2E_P95_MS.set(float(gate["e2e_p95_ms"]))
        if decided:
            PERF_GATE_PASS.set(1.0 if gate["pass"] else 0.0)

        if degraded and not self._gate_degraded:
            self._gate_degraded = True
            log.warning(
                "PERF_GATE_FAIL e2e_p95=%s ms (limit %s ms) peak_rss=%s MiB "
                "(limit %s MiB) sustained %.1fs: this unit is over its latency "
                "budget and now reports degraded",
                gate["e2e_p95_ms"],
                gate["e2e_p95_limit_ms"],
                gate["peak_rss_mb"],
                gate["peak_rss_limit_mb"],
                self._gate_fail_streak * HEALTH_REFRESH_S,
            )
        elif self._gate_degraded and not degraded:
            self._gate_degraded = False
            log.info(
                "PERF_GATE_RECOVERED e2e_p95=%s ms is back inside the %s ms limit",
                gate["e2e_p95_ms"],
                gate["e2e_p95_limit_ms"],
            )
        return gate, degraded

    def _pet_watchdog(self) -> None:
        """Ping systemd only while frames are actually arriving.

        The previous build pinged unconditionally from the camera-failure path,
        so a unit with a dead camera reported itself alive for ever.  Silence is
        the escalation path: ``WatchdogSec`` then restarts the unit.
        """
        sup = self.sup
        advancing = self.last_frame_id != self._last_wd_frame
        if sup.state is CaptureState.FAILED or sup.camera_fail_active or not advancing:
            notify_status(
                "camera=%s frames=%d - watchdog ping withheld"
                % (sup.camera_status, self.frames)
            )
            return
        notify_watchdog()
        self._last_wd_frame = self.last_frame_id

    # ------------------------------------------------------------ calibration
    def _collect_calibration(self, signals: DriverSignals) -> None:
        if not self.args.calibrate_forward:
            return
        pose = signals.pose
        if pose is None or pose.source.value == "unavailable" or pose.confidence <= 0.0:
            return
        ear = _ear_of(signals)
        if ear is None or ear <= 0.0:
            return
        self.calib_yaw.append(float(pose.yaw))
        self.calib_pitch.append(float(pose.pitch))
        self.calib_ear.append(float(ear))

    def _write_calibration(self) -> None:
        if not self.args.calibrate_forward:
            return
        if not self.calib_yaw:
            log.error(
                "calibration produced no usable samples: no frame had both a "
                "confident head pose and a valid EAR"
            )
            self._exit_code = self._exit_code or 3
            return
        import yaml

        from dms.config.schema import calibration_path

        out = calibration_path(self.args.config)
        if not out:
            log.error("this profile disables the calibration file (calibration: '')")
            self._exit_code = self._exit_code or 3
            return
        payload = {
            "forward_zero": {
                "yaw": float(_median(self.calib_yaw)),
                "pitch": float(_median(self.calib_pitch)),
                "roll": 0.0,
            },
            "forward_zero_calibrated": True,
            "state": {"ear_open_median": float(_median(self.calib_ear))},
        }
        with open(out, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        log.info(
            "wrote %s from %d samples: forward_zero=%s ear_open_median=%.3f",
            out,
            len(self.calib_yaw),
            payload["forward_zero"],
            payload["state"]["ear_open_median"],
        )

    # ---------------------------------------------------------------- teardown
    def teardown(self) -> None:
        """Release everything, in the order that cannot leave hardware asserted."""
        notify_stopping("shutting down")
        if self.gpio is not None:
            try:
                self.gpio.set_critical(False)
            except Exception:  # noqa: BLE001
                log.error("could not release the GPIO alert line", exc_info=True)
        if self.sup is not None:
            try:
                self.sup.stop()
            except Exception:  # noqa: BLE001
                log.error("capture stop failed", exc_info=True)
        if self.writer:
            try:
                self.writer.release()
            except Exception:  # noqa: BLE001
                log.error("video writer release failed", exc_info=True)
        if self.args.save_preview and self.preview is not None:
            try:
                import cv2

                os.makedirs(os.path.dirname(self.args.save_preview) or ".", exist_ok=True)
                cv2.imwrite(self.args.save_preview, self.preview)
                log.info("wrote preview %s", self.args.save_preview)
            except Exception:  # noqa: BLE001
                log.error("preview write failed", exc_info=True)
        self._write_calibration()
        if self.clips is not None:
            try:
                self.clips.stop()
            except Exception:  # noqa: BLE001
                log.error("clip ring stop failed", exc_info=True)
        if self.events is not None:
            try:
                self.events.close()
            except Exception:  # noqa: BLE001
                log.error("event log close failed", exc_info=True)
        if self.registry is not None:
            try:
                self.registry.close()
            except Exception:  # noqa: BLE001
                log.error("model registry close failed", exc_info=True)
        for engine in (self.lmk_engine, self.face_engine):
            if engine is not None:
                try:
                    engine.close()
                except Exception:  # noqa: BLE001
                    log.error("engine close failed", exc_info=True)
        self._write_budget()
        if self.gpio is not None:
            self.gpio.close()
        if self.hsv is not None:
            self.hsv.stop()

    def _log_gate_verdict(self) -> None:
        """The end-of-run PASS/FAIL line. Always logged, artifact or not.

        The trailing parenthesis names WHICH half of the gate decided the
        verdict and by how much.  ``FAIL`` on its own sent readers to the
        latency numbers when the miss could equally have been memory, and sent
        nobody to the fact that a 300-frame observation on this board has a
        7-16 ms spread - so a PASS reports its margin too, and a margin thinner
        than that spread is not a pass anybody should plan on.
        """
        if self.budget is None:
            return
        self.budget.observe_rss(peak_rss_mb())
        gate = self.budget.gate()
        verdict, why = _gate_verdict(gate)
        level = log.info if verdict == "PASS" else log.warning
        level(
            "PERF_GATE %s e2e_p95=%s ms limit=%s ms peak_rss=%s MiB limit=%s MiB "
            "samples=%s fps=%s config=%s (%s)",
            verdict,
            gate["e2e_p95_ms"],
            gate["e2e_p95_limit_ms"],
            gate["peak_rss_mb"],
            gate["peak_rss_limit_mb"],
            gate["e2e_samples"],
            gate["fps_mean"],
            os.path.basename(self.args.config),
            why,
        )

    def _write_budget(self) -> None:
        out = os.environ.get("DMS_LATENCY_BUDGET_OUT", "")
        if not out or self.budget is None:
            return
        try:
            self.budget.observe_rss(peak_rss_mb())
            path = self.budget.write(out)
            gate = self.budget.gate()
            # Same three-valued verdict as the PERF_GATE line.  This line used
            # to be two-valued and called an UNDECIDED window "FAIL", so the
            # two lines of a short run contradicted each other.
            verdict, why = _gate_verdict(gate)
            log.info(
                "wrote %s gate=%s e2e_p95=%s ms peak_rss=%s MiB (%s)",
                path,
                verdict,
                gate["e2e_p95_ms"],
                gate["peak_rss_mb"],
                why,
            )
        except OSError as exc:
            log.error("could not write the latency budget to %s: %s", out, exc)

    def summarise(self) -> None:
        budget = self.budget
        snap = budget.timer.snapshot() if budget is not None else {}
        self._log_gate_verdict()
        e2e = snap.get("e2e")
        log.info(
            "stopped frames=%s last_id=%s mean_infer_ms=%.1f total_face_hits=%s "
            "e2e_p50_ms=%.1f e2e_p95_ms=%.1f infer_p95_ms=%.1f fps=%.1f infer_failures=%d",
            self.frames,
            self.last_frame_id,
            self.infer_ms.lifetime_mean,
            self.face_hits,
            e2e.p50 if e2e else 0.0,
            e2e.p95 if e2e else 0.0,
            self.infer_ms.p95,
            budget.timer.fps() if budget else 0.0,
            self.infer_failures_total,
        )
        if self.monitor is not None:
            log.info(
                "final active=%s unreliable=%s perclos=%.3f(valid=%s)",
                self.alert_names,
                sorted(k for k, v in self.monitor.unreliable.items() if v),
                self.monitor.perclos,
                self.monitor.perclos_valid,
            )


# --------------------------------------------------------------------- extras


def _gate_verdict(gate: Dict[str, Any]) -> Tuple[str, str]:
    """``(PASS|FAIL|UNDECIDED, why)`` for one :meth:`LatencyBudget.gate` dict.

    The single source of the verdict, so the shutdown ``PERF_GATE`` line, the
    ``latency_budget.json`` line and the ``/healthz`` note cannot disagree.

    Two distinctions this makes that ``gate['pass']`` does not:

    * A window with fewer than :data:`GATE_MIN_SAMPLES` e2e samples is
      UNDECIDED, not FAIL.  ``gate['pass']`` is False for both, which is how a
      20-frame run came to print ``PERF_GATE UNDECIDED`` and, in the next line,
      ``gate=FAIL``.
    * ``why`` names the half that decided it and the signed margin.  On a miss
      that is the difference between "look at latency" and "look at memory"; on
      a pass it is the difference between a real pass and a 0.4 ms margin on a
      board whose run-to-run spread is 7-16 ms.
    """
    if int(gate.get("e2e_samples") or 0) < GATE_MIN_SAMPLES:
        return "UNDECIDED", "only %s of %s e2e samples, not judgeable" % (
            gate.get("e2e_samples"),
            GATE_MIN_SAMPLES,
        )
    p95 = gate.get("e2e_p95_ms")
    limit = gate.get("e2e_p95_limit_ms")
    over_ms = None
    if p95 is not None and limit:
        over_ms = float(p95) - float(limit)
    why = []
    if not gate.get("latency_pass"):
        why.append(
            "e2e p95 OVER by %.2f ms" % over_ms
            if over_ms is not None
            else "e2e p95 unavailable"
        )
    if not gate.get("rss_measured"):
        why.append("peak RSS never measured")
    elif not gate.get("rss_pass"):
        why.append(
            "peak RSS OVER by %.0f MiB"
            % (float(gate["peak_rss_mb"]) - float(gate["peak_rss_limit_mb"]))
        )
    if why:
        return "FAIL", "; ".join(why)
    # abs(), not -over_ms: at exactly the limit -over_ms is -0.0 and formats as
    # "-0.00 ms", which reads like a miss in the one line that must not.
    return "PASS", "e2e p95 inside by %.2f ms" % (
        abs(over_ms) if over_ms is not None else 0.0
    )


def _gate_note(gate: Dict[str, Any], degraded: bool) -> str:
    """One operator-readable line for /healthz notes describing the gate."""
    verdict, why = _gate_verdict(gate)
    if verdict == "UNDECIDED":
        return "perf gate: undecided, %s of %s e2e samples" % (
            gate.get("e2e_samples"),
            GATE_MIN_SAMPLES,
        )
    return "perf gate %s: e2e p95 %s ms vs %s ms limit, peak rss %s vs %s MiB - %s%s" % (
        verdict,
        gate["e2e_p95_ms"],
        gate["e2e_p95_limit_ms"],
        gate["peak_rss_mb"],
        gate["peak_rss_limit_mb"],
        why,
        " (sustained: degraded)" if degraded else "",
    )


def _ear_of(signals: DriverSignals) -> Optional[float]:
    """Mean EAR across whichever eyes reported one, or None."""
    vals = [
        e.ear
        for e in (signals.left_eye, signals.right_eye)
        if e is not None and e.ear > 0.0
    ]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _zero_is_set(zero) -> bool:
    return any(abs(float(getattr(zero, k, 0.0))) > 1e-9 for k in ("yaw", "pitch", "roll"))


def _version() -> str:
    try:
        from dms import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "0"


def _trt_version() -> str:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:  # noqa: BLE001
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    install_signal_handlers()
    _STOP.clear()
    _RELOAD.clear()

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2

    if args.source:
        cfg.source.type = "file"
        cfg.source.path = args.source
        cfg.source.dev = True
        cfg.camera.retry_forever = False
        cfg.camera.fail_fatal = True

    # Imagery egress is refused at startup, not silently ignored later.  These
    # are the three paths that can put a driver's face somewhere it can be seen
    # or stored; the clip ring enforces its own gate and logs the reason, so it
    # only warrants a warning here.
    if (args.save_video or args.save_preview) and not cfg.privacy.record_faces:
        log.error(
            "PRIVACY --save-video/--save-preview write driver imagery and "
            "privacy.record_faces is false in %s",
            args.config,
        )
        return 2
    if cfg.display.enabled and not cfg.privacy.record_faces:
        log.error(
            "PRIVACY display.enabled renders the driver's face and "
            "privacy.record_faces is false in %s",
            args.config,
        )
        return 2
    if cfg.clips.enabled and not cfg.privacy.record_faces:
        log.warning(
            "clips.enabled is true but privacy.record_faces is false: the clip ring "
            "will refuse to record. Set both, or set neither."
        )

    try:
        warnings = validate_source(cfg)
    except ValueError as exc:
        log.error("bad source configuration: %s", exc)
        return 2
    for w in warnings:
        log.warning("source: %s", w)

    if cfg.is_replay and cfg.source.type == "file" and not os.path.isfile(cfg.source.path):
        log.error("replay file missing: %s", cfg.source.path)
        return 2
    if cfg.is_replay and not cfg.source.dev:
        log.warning(
            "SOURCE_FILE source.type=%s but source.dev is false: this profile replays a "
            "file while claiming to be a vehicle unit",
            cfg.source.type,
        )

    app = DmsApp(cfg, args)
    rc = None
    try:
        rc = app.setup()
        if rc is not None:
            return rc
        rc = app.run()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log.info("interrupted")
        rc = 0
    except Exception as exc:  # noqa: BLE001 - report, then tear down cleanly
        log.error("FATAL unhandled exception in the frame loop: %s", exc, exc_info=True)
        rc = 1
    finally:
        try:
            app.summarise()
        except Exception:  # noqa: BLE001
            log.error("summary failed", exc_info=True)
        app.teardown()
    return int(rc or app._exit_code or 0)


if __name__ == "__main__":
    sys.exit(main())
