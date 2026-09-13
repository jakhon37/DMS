"""Runtime tests: capture supervision, rolling metrics, preprocessing, TrtEngine.

No GStreamer and no GPU are needed for anything here except the TrtEngine block,
which is opt-in via ``DMS_TRT_TESTS=1`` because it touches the shared GPU and must be
run under ``flock /tmp/jetson-gpu.lock``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dms.runtime.preprocess import (
    BgraToBgr,
    Letterbox,
    bgra_to_bgr,
    normalize_chw,
)
from dms.runtime.supervisor import (
    DEFAULT_BACKOFF_S,
    CaptureState,
    CaptureSupervisor,
    RollingStats,
)
from dms.types import Frame, SourceType

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------- fakes


class FakeClock:
    """Deterministic monotonic clock; sleep() just advances it."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class ScriptedSource:
    """A capture source whose failure mode is scripted, with no GStreamer at all.

    ``plan`` is a shared dict so the test can heal the source between restarts:
      good_frames : frames emitted before the fault fires
      fault       : "error" | "eos" | "stall" | None
      start_fails : number of start() calls that raise before one succeeds
    """

    def __init__(self, plan: dict, log: list) -> None:
        self.plan = plan
        self.log = log
        self.eos = False
        self.error = False
        self.last_error = ""
        self.dropped = 0
        self.started = False
        self._emitted = 0
        self._id = 0

    def start(self) -> None:
        if self.plan.get("start_fails", 0) > 0:
            self.plan["start_fails"] -= 1
            self.log.append("start-fail")
            raise RuntimeError("scripted start failure")
        self.started = True
        self.eos = False
        self.error = False
        self._emitted = 0
        self.log.append("start")

    def restart(self) -> None:
        self.log.append("restart")
        self.stop()
        self.start()

    def stop(self) -> None:
        self.started = False
        self.log.append("stop")

    def read(self, timeout_s: float = 1.0):
        if not self.started:
            return None
        if self._emitted < self.plan.get("good_frames", 0):
            self._emitted += 1
            self._id += 1
            return Frame(
                frame_id=self._id,
                t_mono_ns=self._id * 33_000_000,
                full_bgra=np.zeros((4, 4, 4), dtype=np.uint8),
                ok=True,
                source=SourceType.TEST,
                width=4,
                height=4,
            )
        fault = self.plan.get("fault")
        if fault == "error":
            self.error = True
            self.last_error = "scripted pipeline error"
        elif fault == "eos":
            self.eos = True
        # "stall" and None just return nothing.
        return None


def make_supervisor(plan, log, **kwargs):
    clock = FakeClock()
    kwargs.setdefault("clock", clock)
    kwargs.setdefault("sleep", clock.sleep)
    kwargs.setdefault("stall_timeout_s", 2.0)
    kwargs.setdefault("source_type", SourceType.TEST)
    sup = CaptureSupervisor(lambda: ScriptedSource(plan, log), **kwargs)
    return sup, clock


# ------------------------------------------------------------------ supervisor


def test_healthy_source_reports_running_and_counts_frames():
    plan = {"good_frames": 3, "fault": None}
    log = []
    sup, clock = make_supervisor(plan, log)
    assert sup.state is CaptureState.IDLE
    frames = [sup.read(timeout_s=0.1) for _ in range(3)]
    assert all(f is not None and f.ok for f in frames)
    assert sup.state is CaptureState.RUNNING
    assert sup.camera_status == "ok"
    assert sup.frames_total == 3
    assert sup.healthy is True
    assert sup.take_events() == []


def test_error_emits_one_sentinel_and_a_camera_fail_enter_edge():
    plan = {"good_frames": 1, "fault": "error"}
    log = []
    sup, clock = make_supervisor(plan, log)
    assert sup.read(timeout_s=0.1).ok is True
    sentinel = sup.read(timeout_s=0.1)
    assert sentinel is not None and sentinel.ok is False
    assert sentinel.full_bgra is None
    assert sup.state is CaptureState.RETRYING
    assert sup.camera_status == "retrying"
    evs = sup.take_events()
    assert [(e.kind, e.edge) for e in evs] == [("camera_fail", "enter")]
    assert "scripted pipeline error" in evs[0].detail
    # The sentinel is one-shot: the next read is a plain None while backing off.
    assert sup.read(timeout_s=0.1) is None
    assert sup.sentinels_emitted == 1


def test_backoff_ladder_is_the_documented_one_and_then_fails():
    plan = {"good_frames": 0, "fault": "error"}
    log = []
    sup, clock = make_supervisor(plan, log, stall_timeout_s=1.0)
    delays = []
    prev_failures = 0
    # Pump until the supervisor gives up.
    for _ in range(400):
        before = clock.t
        sup.read(timeout_s=1.0)
        if sup.consecutive_failures != prev_failures:
            prev_failures = sup.consecutive_failures
            delays.append(round(sup._next_attempt_s - before, 3))
        if sup.state is CaptureState.FAILED:
            break
    assert sup.state is CaptureState.FAILED
    assert sup.camera_status == "failed"
    assert delays[: len(DEFAULT_BACKOFF_S)] == list(DEFAULT_BACKOFF_S)
    assert sup.consecutive_failures == len(DEFAULT_BACKOFF_S) + 1
    # Exactly one enter edge for the whole outage, no spurious exits.
    kinds = [(e.kind, e.edge) for e in sup.take_events()]
    assert kinds == [("camera_fail", "enter")]


def test_recovers_after_restart_and_emits_the_exit_edge():
    plan = {"good_frames": 2, "fault": "error"}
    log = []
    sup, clock = make_supervisor(plan, log)
    assert sup.read(timeout_s=0.1).ok
    assert sup.read(timeout_s=0.1).ok
    assert sup.read(timeout_s=0.1).ok is False  # sentinel
    assert sup.camera_fail_active is True
    # Heal the source, then let the backoff expire.
    plan["fault"] = None
    plan["good_frames"] = 10
    for _ in range(20):
        frame = sup.read(timeout_s=1.0)
        if frame is not None and frame.ok:
            break
    assert sup.state is CaptureState.RUNNING
    assert sup.restarts == 1
    assert "restart" in log
    assert sup.camera_fail_active is False
    edges = [(e.kind, e.edge) for e in sup.take_events()]
    assert ("camera_fail", "enter") in edges
    assert ("camera_fail", "exit") in edges


def test_frame_ids_stay_monotonic_across_a_restart():
    plan = {"good_frames": 3, "fault": "error"}
    log = []
    sup, clock = make_supervisor(plan, log)
    ids = []
    for _ in range(3):
        ids.append(sup.read(timeout_s=0.1).frame_id)
    sup.read(timeout_s=0.1)  # sentinel
    plan["fault"] = None
    plan["good_frames"] = 5
    for _ in range(20):
        f = sup.read(timeout_s=1.0)
        if f is not None and f.ok:
            ids.append(f.frame_id)
        if len(ids) >= 6:
            break
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert ids[3] > ids[2], "the restarted source restarts at frame 1; ids must not repeat"


def test_stall_with_no_error_still_trips_the_watchdog():
    plan = {"good_frames": 1, "fault": "stall"}
    log = []
    sup, clock = make_supervisor(plan, log, stall_timeout_s=2.0)
    assert sup.read(timeout_s=0.1).ok
    # No error, no EOS - just silence. Advance past the stall timeout.
    for _ in range(5):
        clock.sleep(1.0)
        frame = sup.read(timeout_s=0.1)
        if frame is not None:
            break
    assert frame is not None and frame.ok is False
    assert sup.state is CaptureState.RETRYING
    assert "no frame for" in sup.last_error


def test_first_frame_deadline_catches_a_source_that_never_delivers():
    plan = {"good_frames": 0, "fault": None}
    log = []
    sup, clock = make_supervisor(plan, log, stall_timeout_s=30.0, first_frame_timeout_s=2.0)
    sup.start()
    frame = None
    for _ in range(5):
        clock.sleep(1.0)
        frame = sup.read(timeout_s=0.1)
        if frame is not None:
            break
    assert frame is not None and frame.ok is False
    assert "first frame" in sup.last_error


def test_eos_is_a_clean_end_for_a_file_source():
    plan = {"good_frames": 2, "fault": "eos"}
    log = []
    sup, clock = make_supervisor(plan, log, eos_is_end=True)
    assert sup.read(timeout_s=0.1).ok
    assert sup.read(timeout_s=0.1).ok
    assert sup.read(timeout_s=0.1) is None
    assert sup.finished is True
    assert sup.state is CaptureState.FINISHED
    assert sup.take_events() == []


def test_eos_on_a_live_source_is_a_fault():
    plan = {"good_frames": 1, "fault": "eos"}
    log = []
    sup, clock = make_supervisor(plan, log, eos_is_end=False)
    assert sup.read(timeout_s=0.1).ok
    sentinel = sup.read(timeout_s=0.1)
    assert sentinel is not None and sentinel.ok is False
    assert "EOS" in sup.last_error


def test_start_failure_is_retried_not_raised():
    plan = {"good_frames": 5, "fault": None, "start_fails": 2}
    log = []
    sup, clock = make_supervisor(plan, log)
    assert sup.start() is False
    assert sup.state is CaptureState.RETRYING
    frame = None
    for _ in range(30):
        frame = sup.read(timeout_s=1.0)
        if frame is not None and frame.ok:
            break
    assert frame is not None and frame.ok
    assert log.count("start-fail") == 2
    assert sup.failures_total == 2


def test_retry_forever_keeps_trying_while_reporting_failed():
    plan = {"good_frames": 0, "fault": "error"}
    log = []
    sup, clock = make_supervisor(plan, log, retry_forever=True, stall_timeout_s=1.0)
    for _ in range(300):
        sup.read(timeout_s=1.0)
        if sup.state is CaptureState.FAILED and sup.restarts >= 7:
            break
    assert sup.state is CaptureState.FAILED
    assert sup.camera_status == "failed"
    assert sup.restarts >= 7, "retry_forever must keep attempting past the ladder"


def test_snapshot_has_the_health_counters():
    plan = {"good_frames": 2, "fault": None}
    log = []
    sup, clock = make_supervisor(plan, log)
    sup.read(timeout_s=0.1)
    snap = sup.snapshot()
    for key in (
        "state",
        "camera",
        "fps",
        "frames_total",
        "frames_dropped",
        "capture_restarts",
        "capture_failures",
        "consecutive_failures",
        "camera_fail_active",
        "pipeline_starved",
    ):
        assert key in snap
    assert snap["camera"] == "ok"


def test_pipeline_starved_edges():
    plan = {"good_frames": 1000, "fault": None}
    log = []
    sup, clock = make_supervisor(plan, log, starved_below_fps=5.0, starved_for_s=1.0)
    # 2 fps: each read advances the clock by 0.5 s.
    for _ in range(20):
        clock.sleep(0.5)
        sup.read(timeout_s=0.1)
    kinds = [(e.kind, e.edge) for e in sup.take_events()]
    assert ("pipeline_starved", "enter") in kinds
    assert sup.snapshot()["pipeline_starved"] is True


def test_stop_is_idempotent_and_terminal():
    plan = {"good_frames": 5, "fault": None}
    log = []
    sup, clock = make_supervisor(plan, log)
    sup.read(timeout_s=0.1)
    sup.stop()
    sup.stop()
    assert sup.state is CaptureState.STOPPED
    assert sup.read(timeout_s=0.1) is None
    assert log.count("stop") == 1


# ----------------------------------------------------------------- RollingStats


def test_rolling_stats_window_is_bounded_and_percentiles_track_recent_samples():
    rs = RollingStats(maxlen=100)
    for i in range(10_000):
        rs.add(float(i))
    assert len(rs) == 100
    assert rs.count == 10_000
    # The window only holds the last 100 samples, so p50 is near 9950, not 5000.
    assert 9900 <= rs.p50 <= 9999
    assert rs.p95 >= rs.p50
    assert rs.p99 >= rs.p95
    assert abs(rs.lifetime_mean - 4999.5) < 1e-6
    assert rs.lifetime_max == 9999.0
    stats = rs.stats(prefix="infer_ms_")
    assert stats["infer_ms_count"] == 10_000.0


def test_rolling_stats_empty_is_zero_not_an_error():
    rs = RollingStats(maxlen=8)
    assert rs.p50 == 0.0 and rs.p95 == 0.0 and rs.window_mean == 0.0
    with pytest.raises(ValueError):
        RollingStats(maxlen=0)


# ------------------------------------------------------------------ preprocess


def test_letterbox_720p_to_640_is_topleft_padded_and_reuses_its_buffers():
    lb = Letterbox((640, 640), pad_value=114, center=False)
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    canvas1, g = lb.canvas(frame)
    assert g.factor == pytest.approx(0.5)
    assert (g.new_w, g.new_h) == (640, 360)
    assert (g.pad_x, g.pad_y) == (0, 0)
    assert canvas1[0, 0].tolist() == [200, 200, 200]
    assert canvas1[400, 0].tolist() == [114, 114, 114]
    canvas2, _ = lb.canvas(frame)
    assert canvas2 is canvas1, "the canvas must be reused, not reallocated"
    blob1, _ = lb.blob(frame)
    blob2, _ = lb.blob(frame)
    assert blob1 is blob2
    assert blob1.shape == (1, 3, 640, 640)
    assert blob1.dtype == np.float32


def test_letterbox_centered_padding_and_back_mapping():
    lb = Letterbox((640, 640), pad_value=114, center=True)
    frame = np.full((720, 1280, 3), 10, dtype=np.uint8)
    canvas, g = lb.canvas(frame)
    assert g.pad_y == 140 and g.pad_x == 0
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[320, 320].tolist() == [10, 10, 10]
    # A box at the very corner of the pasted region maps back to the frame corner.
    box = np.array([g.pad_x, g.pad_y, g.pad_x + g.new_w, g.pad_y + g.new_h], dtype=np.float32)
    back = g.to_source(box)
    assert back[0] == pytest.approx(0.0)
    assert back[1] == pytest.approx(0.0)
    assert back[2] == pytest.approx(1280.0, abs=1.0)
    assert back[3] == pytest.approx(720.0, abs=1.0)


def test_letterbox_scrfd_normalisation_matches_the_reference_formula():
    lb = Letterbox.for_scrfd((640, 640))
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    blob, g = lb.blob(frame)
    # (255 - 127.5) / 127.5 == 1.0 inside the image, (0 - 127.5)/127.5 == -1.0 in the pad.
    assert blob[0, 0, 0, 0] == pytest.approx(1.0)
    assert blob[0, 0, 500, 0] == pytest.approx(-1.0)
    assert g.factor == pytest.approx(0.5)


def test_letterbox_imagenet_rgb_swaps_channels():
    lb = Letterbox.for_imagenet((224, 224), center=False)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[..., 2] = 255  # pure red in BGR
    blob, _ = lb.blob(frame)
    # After BGR->RGB the red must land in channel 0.
    r = blob[0, 0, 10, 10]
    b = blob[0, 2, 10, 10]
    assert r > b
    assert r == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-4)


def test_letterbox_geometry_recomputes_when_the_source_size_changes():
    lb = Letterbox((320, 320), pad_value=114)
    a = np.full((480, 640, 3), 50, dtype=np.uint8)
    b = np.full((640, 480, 3), 50, dtype=np.uint8)
    _, ga = lb.canvas(a)
    canvas, gb = lb.canvas(b)
    assert (ga.new_w, ga.new_h) == (320, 240)
    assert (gb.new_w, gb.new_h) == (240, 320)
    # The pad region must be repainted, not left over from the previous geometry.
    assert canvas[10, 300].tolist() == [114, 114, 114]


def test_letterbox_rejects_bad_input():
    lb = Letterbox((64, 64))
    with pytest.raises(ValueError):
        lb.canvas(np.zeros((8, 8, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        lb.canvas(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        Letterbox((0, 64))
    with pytest.raises(ValueError):
        Letterbox((64, 64), std=0.0)


def test_bgra_to_bgr_is_contiguous_and_matches_the_slice():
    rng = np.random.RandomState(7)
    bgra = rng.randint(0, 255, (32, 48, 4)).astype(np.uint8)
    bgr = bgra_to_bgr(bgra)
    assert bgr.flags["C_CONTIGUOUS"]
    assert np.array_equal(bgr, bgra[..., :3])
    conv = BgraToBgr()
    first = conv(bgra)
    second = conv(bgra)
    assert first is second, "BgraToBgr must reuse its destination buffer"
    with pytest.raises(ValueError):
        bgra_to_bgr(bgra[..., :3])


def test_normalize_chw_matches_the_manual_formula():
    crop = np.full((24, 40, 3), 128, dtype=np.uint8)
    out = normalize_chw(crop, scale=1.0 / 255.0)
    assert out.shape == (1, 3, 24, 40)
    assert out[0, 0, 0, 0] == pytest.approx(128 / 255.0)
    buf = np.empty((1, 3, 24, 40), dtype=np.float32)
    again = normalize_chw(crop, scale=1.0 / 255.0, out=buf)
    assert again is buf


# ------------------------------------------------------------------ TrtEngine
#
# Opt-in: these load a real engine and therefore need the shared GPU.
#   flock /tmp/jetson-gpu.lock -c "DMS_TRT_TESTS=1 python -m pytest tests/test_supervisor.py -q"

_TRT_ENGINE = os.path.join(ROOT, "engines", "ocec_s_24x40_fp16_gpu.engine")

trt_only = pytest.mark.skipif(
    os.environ.get("DMS_TRT_TESTS") != "1" or not os.path.isfile(_TRT_ENGINE),
    reason="set DMS_TRT_TESTS=1 and run under flock /tmp/jetson-gpu.lock",
)


@trt_only
def test_trt_engine_lifecycle_is_clean():
    pytest.importorskip("tensorrt")
    from dms.runtime.trt_engine import EngineBusyError, TrtEngine, TrtExecutionError

    with TrtEngine(_TRT_ENGINE) as eng:
        assert eng.input_shape == (1, 3, 24, 40)
        assert eng.output_names
        blob = np.zeros(eng.input_shape, dtype=np.float32)
        out = eng.infer({eng.input_name: blob})
        assert set(out) == set(eng.output_names)
        # copy=True must hand back an array that survives the next inference.
        first = out[eng.output_names[0]].copy()
        eng.infer({eng.input_name: blob})
        assert np.array_equal(first, out[eng.output_names[0]])
        # One item in flight at a time.
        item = eng.submit({eng.input_name: blob})
        with pytest.raises(EngineBusyError):
            eng.submit({eng.input_name: blob})
        eng.wait(item)
        # Zero-copy path returns the pinned buffer itself.
        z1 = eng.infer({eng.input_name: blob}, copy=False)
        z2 = eng.infer({eng.input_name: blob}, copy=False)
        assert z1[eng.output_names[0]] is z2[eng.output_names[0]]
    assert eng.closed
    eng.close()  # idempotent
    with pytest.raises(TrtExecutionError):
        eng.infer({"x": np.zeros((1, 3, 24, 40), dtype=np.float32)})


@trt_only
def test_trt_engine_rejects_a_wrong_shape_and_a_missing_file():
    pytest.importorskip("tensorrt")
    from dms.runtime.trt_engine import EngineLoadError, TrtEngine

    with pytest.raises(EngineLoadError):
        TrtEngine(os.path.join(ROOT, "engines", "definitely-not-here.engine"))
    with TrtEngine(_TRT_ENGINE) as eng:
        with pytest.raises(ValueError):
            eng.infer({eng.input_name: np.zeros((1, 3, 24, 41), dtype=np.float32)})
        # The failed submit must not leave the engine stuck busy.
        assert eng.infer({eng.input_name: np.zeros(eng.input_shape, dtype=np.float32)})
