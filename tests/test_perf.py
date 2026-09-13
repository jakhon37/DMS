"""Tests for the performance instrumentation, the budget and the overlay.

Nothing here needs a GPU, TensorRT or an engine: these are the guards that can
run anywhere, and they cover the three defects that cost the most and were
invisible to the previous suite -

* the timer that grew without bound and was fully sorted twice a second,
* the budget that was never computed at all, and
* the overlay that ran on every frame in the headless production configuration.

The suite deliberately asserts on *behaviour that costs money*: that the ring
buffer stops growing, that a disabled overlay touches no pixels, and that a span
costs microseconds rather than milliseconds.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dms.perf import LatencyBudget, NullTimer, StageTimer, peak_rss_mb  # noqa: E402
from dms.perf.budget import (  # noqa: E402
    E2E,
    STAGE_BY_NAME,
    GATE_E2E_P95_MS,
    MIN_SAMPLES,
    STAGES,
    default_targets,
    stage_names,
)
from dms.perf.timers import Stage, StageReentryError  # noqa: E402


def _load_script(name):
    """Import a file from ``scripts/`` that is not on the import path."""
    path = os.path.join(ROOT, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("dms_script_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================ dms.perf.timers


class TestStageRing:
    def test_window_is_bounded(self):
        """The whole point: memory does not grow with uptime."""
        s = Stage("x", window=50)
        for i in range(100000):
            s.record(float(i))
        assert s.count == 50
        assert s.total_count == 100000
        assert s._buf.nbytes == 50 * 8

    def test_window_holds_the_most_recent_samples(self):
        s = Stage("x", window=4)
        for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            s.record(v)
        assert sorted(s.samples().tolist()) == [3.0, 4.0, 5.0, 6.0]
        assert s.samples().tolist() == [3.0, 4.0, 5.0, 6.0]  # oldest first

    def test_percentiles_are_nearest_rank(self):
        s = Stage("x", window=100)
        for i in range(1, 101):
            s.record(float(i))
        st = s.stats()
        assert st.count == 100
        assert st.p50 == pytest.approx(50.0, abs=1.0)
        assert st.p95 == pytest.approx(95.0, abs=1.0)
        assert st.p99 == pytest.approx(99.0, abs=1.0)
        assert st.max_ms == 100.0
        assert st.mean_ms == pytest.approx(50.5)

    def test_percentile_is_windowed_not_lifetime(self):
        """A lifetime percentile cannot show a regression; a windowed one must."""
        s = Stage("x", window=100)
        for _ in range(10000):
            s.record(10.0)
        assert s.stats().p95 == pytest.approx(10.0)
        for _ in range(100):
            s.record(90.0)
        assert s.stats().p95 == pytest.approx(90.0)

    def test_empty_stage_reports_zero_not_a_guess(self):
        st = Stage("x").stats()
        assert st.count == 0
        assert st.p50 == st.p95 == st.p99 == st.max_ms == 0.0

    def test_single_sample(self):
        s = Stage("x")
        s.record(7.5)
        st = s.stats()
        assert st.p50 == st.p95 == st.p99 == 7.5

    def test_last_ms_tracks_the_newest_sample(self):
        s = Stage("x", window=3)
        for v in (1.0, 2.0, 3.0, 4.0):
            s.record(v)
        assert s.stats().last_ms == 4.0

    def test_reset_keeps_lifetime_count(self):
        s = Stage("x", window=8)
        for i in range(20):
            s.record(float(i))
        s.reset()
        assert s.count == 0
        assert s.total_count == 20
        assert s.stats().p95 == 0.0

    def test_rejects_a_zero_window(self):
        with pytest.raises(ValueError):
            Stage("x", window=0)


class TestStageContextManager:
    def test_records_elapsed_time(self):
        s = Stage("x")
        with s:
            time.sleep(0.01)
        assert s.count == 1
        assert 5.0 < s.stats().last_ms < 200.0

    def test_records_even_when_the_body_raises(self):
        """A stage that raised is still a stage that consumed time."""
        s = Stage("x")
        with pytest.raises(RuntimeError):
            with s:
                raise RuntimeError("boom")
        assert s.count == 1

    def test_reentry_is_refused(self):
        s = Stage("x")
        with s:
            with pytest.raises(StageReentryError):
                with s:
                    pass

    def test_open_flag(self):
        s = Stage("x")
        assert not s.open
        with s:
            assert s.open
        assert not s.open


class TestStageTimer:
    def test_stage_object_is_reused(self):
        """No per-frame allocation: the same CM object every call."""
        t = StageTimer()
        assert t.stage("a") is t.stage("a")

    def test_snapshot_skips_empty_stages(self):
        t = StageTimer()
        t.stage("never_used")
        t.record("used", 1.0)
        assert list(t.snapshot()) == ["used"]

    def test_p95_map_shape(self):
        t = StageTimer()
        for _ in range(30):
            t.record("detect", 5.0)
        assert t.p95_map() == {"detect": 5.0}

    def test_rolling_fps_falls_when_throughput_falls(self):
        t = StageTimer()
        t.mark_frame()
        t.rolling_fps()  # arm
        for _ in range(10):
            t.mark_frame()
        fast = t.rolling_fps()
        time.sleep(0.05)
        t.mark_frame()
        slow = t.rolling_fps()
        assert fast > slow

    def test_as_dict_is_json_serialisable(self):
        t = StageTimer()
        for i in range(5):
            t.record("a", float(i))
        json.dumps(t.as_dict())

    def test_merge_takes_the_worst_observer(self):
        a, b = StageTimer(), StageTimer()
        for _ in range(30):
            a.record("x", 1.0)
            b.record("x", 9.0)
        merged = StageTimer.merge_snapshots([a.snapshot(), b.snapshot()])
        assert merged["x"]["p95_ms"] == 9.0
        assert merged["x"]["count"] == 60

    def test_null_timer_is_a_no_op(self):
        t = NullTimer()
        with t.stage("anything"):
            pass
        t.record("anything", 5.0)
        assert t.snapshot() == {}


class TestTimerOverhead:
    def test_span_costs_microseconds(self):
        """A timer that costs milliseconds cannot measure a 66 ms budget.

        Measured ~3 us on the Xavier NX; the 60 us ceiling is loose enough not to
        flake on a loaded board and tight enough to catch a real regression such
        as reintroducing an unbounded list or a lock.
        """
        t = StageTimer()
        n = 20000
        t0 = time.perf_counter()
        for _ in range(n):
            with t.stage("s"):
                pass
        per_us = (time.perf_counter() - t0) / n * 1e6
        assert per_us < 60.0, "span cost %.1f us" % per_us

    def test_snapshot_cost_does_not_grow_with_uptime(self):
        """The defect this class replaces: sorting a lifetime list every 15 frames."""
        t = StageTimer(window=300)
        for _ in range(300):
            t.record("s", 1.0)
        t0 = time.perf_counter()
        t.snapshot()
        short = time.perf_counter() - t0
        for _ in range(200000):
            t.record("s", 1.0)
        t0 = time.perf_counter()
        t.snapshot()
        long = time.perf_counter() - t0
        assert long < short * 8 + 0.005


# ============================================================ dms.perf.budget


class TestBudgetTable:
    def test_stage_names_are_unique_and_ordered(self):
        names = [s.name for s in STAGES]
        assert len(names) == len(set(names))
        assert stage_names()[-1] == E2E

    def test_mandatory_targets_fit_inside_the_gate(self):
        """The non-optional stage targets must not already exceed the gate."""
        total = sum(s.target_ms for s in STAGES if not s.optional and s.name != "capture_lag")
        assert total <= GATE_E2E_P95_MS, "budget table sums to %.1f ms" % total

    def test_every_stage_has_a_note(self):
        for s in STAGES:
            assert s.note, "%s has no note saying what it covers" % s.name

    def test_default_targets_include_the_gate(self):
        assert default_targets()[E2E] == GATE_E2E_P95_MS


class TestBudgetVerdicts:
    def _fill(self, b, stage, ms, n=MIN_SAMPLES + 5):
        for _ in range(n):
            b.record(stage, ms)

    def test_stage_over_target_is_a_fail_named_in_offenders(self):
        b = LatencyBudget()
        self._fill(b, "detect_gpu", 30.0)
        self._fill(b, "letterbox", 1.0)
        names = [r["stage"] for r in b.offenders()]
        assert "detect_gpu" in names
        assert "letterbox" not in names

    def test_offenders_are_ranked_worst_first(self):
        b = LatencyBudget()
        self._fill(b, "letterbox", 9.0)      # 1 ms over
        self._fill(b, "detect_gpu", 26.0)    # 20 ms over
        assert [r["stage"] for r in b.offenders()][0] == "detect_gpu"

    def test_few_samples_never_reads_as_a_pass(self):
        b = LatencyBudget()
        self._fill(b, "detect_gpu", 0.1, n=3)
        row = [r for r in b.rows() if r["stage"] == "detect_gpu"][0]
        assert row["status"] == "insufficient_data"

    def test_optional_stage_that_never_ran_is_not_a_failure(self):
        b = LatencyBudget()
        self._fill(b, E2E, 10.0)
        row = [r for r in b.rows() if r["stage"] == "objects"][0]
        assert row["status"] == "not_run"

    def test_mandatory_stage_that_never_ran_is_missing(self):
        b = LatencyBudget()
        self._fill(b, E2E, 10.0)
        row = [r for r in b.rows() if r["stage"] == "letterbox"][0]
        assert row["status"] == "missing"

    def test_unknown_stage_is_reported_not_dropped(self):
        b = LatencyBudget()
        self._fill(b, "brand_new_head", 3.0)
        row = [r for r in b.rows() if r["stage"] == "brand_new_head"][0]
        assert row["status"] == "unbudgeted"

    def test_share_pct_is_relative_to_e2e(self):
        b = LatencyBudget()
        self._fill(b, E2E, 40.0)
        self._fill(b, "detect_gpu", 10.0)
        row = [r for r in b.rows() if r["stage"] == "detect_gpu"][0]
        assert row["share_pct"] == pytest.approx(25.0, abs=0.5)

    def test_gauge_stage_has_no_share(self):
        b = LatencyBudget()
        self._fill(b, E2E, 40.0)
        self._fill(b, "capture_lag", 120.0)
        row = [r for r in b.rows() if r["stage"] == "capture_lag"][0]
        assert "share_pct" not in row

    def test_unaccounted_time_is_reported(self):
        b = LatencyBudget()
        for _ in range(MIN_SAMPLES + 5):
            b.record(E2E, 50.0)
            b.record("detect_gpu", 5.0)
            b.mark_frame()
        parts, e2e = b.sum_of_parts_ms()
        assert e2e == pytest.approx(50.0)
        assert parts == pytest.approx(5.0)
        assert b.report()["accounting"]["unaccounted_ms"] == pytest.approx(45.0)

    def test_cadenced_stage_is_amortised_not_counted_every_frame(self):
        """A 20 ms head on every tenth frame costs 2 ms per frame, not 20."""
        b = LatencyBudget()
        for i in range(100):
            b.record(E2E, 30.0)
            if i % 10 == 0:
                b.record("objects", 20.0)
            b.mark_frame()
        parts, e2e = b.sum_of_parts_ms()
        assert parts == pytest.approx(2.0, abs=0.05)
        assert b.raw_sum_p50_ms() == pytest.approx(20.0)
        assert e2e == pytest.approx(30.0)


class TestBudgetGate:
    def test_gate_passes_only_when_both_halves_pass(self):
        b = LatencyBudget()
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 40.0)
        b.observe_rss(1200)
        assert b.gate()["pass"] is True

    def test_latency_over_the_limit_fails(self):
        b = LatencyBudget()
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 90.0)
        b.observe_rss(1200)
        g = b.gate()
        assert g["latency_pass"] is False and g["pass"] is False

    def test_unmeasured_rss_is_not_a_pass(self):
        """Half the contract had no evidence for nine runs. Silence is not a pass."""
        b = LatencyBudget()
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 10.0)
        g = b.gate()
        assert g["rss_measured"] is False
        assert g["rss_pass"] is False
        assert g["pass"] is False

    def test_too_few_frames_is_not_a_pass(self):
        b = LatencyBudget()
        b.record(E2E, 1.0)
        b.observe_rss(100)
        assert b.gate()["latency_pass"] is False

    def test_rss_over_the_limit_fails(self):
        b = LatencyBudget(gate_peak_rss_mb=1000.0)
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 10.0)
        b.observe_rss(2000)
        assert b.gate()["rss_pass"] is False


class TestBudgetResources:
    def test_peak_rss_is_a_maximum_and_ignores_none(self):
        b = LatencyBudget()
        b.observe_rss(100)
        b.observe_rss(None)
        b.observe_rss(50)
        b.observe_rss(300)
        assert b.peak_rss_mb == 300

    def test_mem_available_tracks_the_minimum(self):
        b = LatencyBudget()
        for v in (2000, 900, 1500, None):
            b.observe_mem_available(v)
        assert b.min_avail_mb == 900

    def test_gpu_clock_range(self):
        b = LatencyBudget()
        for v in (114, 921, None, 306):
            b.observe_gpu_clock(v)
        r = b.report()["resources"]
        assert r["gpu_clock_min_mhz"] == 114 and r["gpu_clock_max_mhz"] == 921

    def test_thermal_events_counted(self):
        b = LatencyBudget()
        b.observe_temp(70.0, throttled=False)
        b.observe_temp(85.0, throttled=True)
        r = b.report()["resources"]
        assert r["max_gpu_temp_c"] == 85.0 and r["thermal_events"] == 1


class TestBudgetIo:
    def test_write_produces_a_valid_report(self, tmp_path):
        b = LatencyBudget(label="unit")
        b.set_context(clip="day_driver", profile="unit")
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 55.0)
            b.record("detect_gpu", 20.0)
            b.mark_frame()
        b.observe_rss(1500)
        path = str(tmp_path / "latency_budget.json")
        b.write(path)
        with open(path) as f:
            rep = json.load(f)
        assert rep["schema"] == "dms.latency_budget/1"
        assert rep["gate"]["e2e_p95_ms"] == pytest.approx(55.0)
        assert "detect_gpu" in rep["offenders"]
        assert rep["context"]["clip"] == "day_driver"
        assert rep["frames"] == MIN_SAMPLES + 1
        assert not os.path.exists(path + ".tmp")

    def test_write_creates_missing_directories(self, tmp_path):
        b = LatencyBudget()
        b.record(E2E, 1.0)
        path = str(tmp_path / "a" / "b" / "latency_budget.json")
        b.write(path)
        assert os.path.isfile(path)

    def test_markdown_names_the_gate_verdict(self):
        b = LatencyBudget()
        for _ in range(MIN_SAMPLES + 1):
            b.record(E2E, 99.0)
        md = b.markdown()
        assert "FAIL" in md and "e2e" in md

    def test_from_mapping_overrides_targets_and_gate(self):
        b = LatencyBudget.from_mapping(
            {"window": 32, "gate_e2e_p95_ms": 40.0, "targets": {"detect_gpu": 1.0}}
        )
        assert b.timer.window == 32
        assert b.gate_e2e_p95_ms == 40.0
        assert b.targets["detect_gpu"] == 1.0
        assert b.targets["landmarks"] == STAGE_BY_NAME["landmarks"].target_ms

    def test_from_config_tolerates_a_config_with_no_perf_section(self):
        class Cfg(object):
            pass

        b = LatencyBudget.from_config(Cfg())
        assert b.gate_e2e_p95_ms == GATE_E2E_P95_MS

    def test_from_config_reads_a_dict_config(self):
        b = LatencyBudget.from_config({"perf": {"gate_e2e_p95_ms": 33.0}})
        assert b.gate_e2e_p95_ms == 33.0

    def test_from_file(self, tmp_path):
        p = tmp_path / "budget.json"
        p.write_text(json.dumps({"gate_peak_rss_mb": 123.0}))
        assert LatencyBudget.from_file(str(p)).gate_peak_rss_mb == 123.0


def test_peak_rss_reads_vmhwm():
    v = peak_rss_mb()
    assert v is None or v > 0


# ==================================================================== dms.viz


@pytest.fixture
def frame():
    rng = np.random.RandomState(7)
    return rng.randint(0, 255, (240, 320, 3), dtype=np.uint8)


@pytest.fixture
def face():
    from dms.infer.scrfd import FaceDet

    return FaceDet(
        xyxy=np.array([40.0, 30.0, 140.0, 160.0], dtype=np.float32),
        score=0.87,
        kps=np.array([[60, 70], [110, 70], [85, 100], [65, 130], [105, 130]], dtype=np.float32),
        xyxy_raw=np.array([40.0, 30.0, 140.0, 160.0], dtype=np.float32),
    )


@pytest.fixture
def lmk():
    return np.stack([np.linspace(45, 135, 106), np.linspace(35, 155, 106)], axis=1).astype(np.float32)


class TestOverlayRenderer:
    def test_disabled_renderer_returns_none_and_draws_nothing(self, frame, face, lmk):
        """PERF-02: the production configuration must not render a pixel."""
        from dms.viz import OverlayRenderer

        r = OverlayRenderer.disabled()
        assert r.enabled is False
        assert r.render(frame, [face], driver=face, landmarks106=lmk) is None
        assert r.frames_drawn == 0
        assert r._buf is None

    def test_from_flags_is_off_when_nothing_consumes_the_pixels(self):
        from dms.viz import OverlayRenderer

        assert OverlayRenderer.from_flags().enabled is False
        assert OverlayRenderer.from_flags(display=True).enabled is True
        assert OverlayRenderer.from_flags(save_video=True).enabled is True
        assert OverlayRenderer.from_flags(save_preview=True).enabled is True

    def test_file_output_renders_full_resolution(self):
        from dms.viz import OverlayRenderer

        assert OverlayRenderer.from_flags(save_video=True).scale == 1.0

    def test_live_display_renders_downscaled(self):
        from dms.viz import OverlayRenderer, PREVIEW_SCALE

        assert OverlayRenderer.from_flags(display=True).scale == PREVIEW_SCALE

    def test_buffer_is_reused_between_frames(self, frame, face):
        from dms.viz import OverlayRenderer

        r = OverlayRenderer(enabled=True, scale=1.0)
        a = r.render(frame, [face])
        b = r.render(frame, [face])
        assert a is b
        assert r.frames_drawn == 2

    def test_output_is_contiguous(self, frame, face):
        from dms.viz import OverlayRenderer

        out = OverlayRenderer(enabled=True, scale=0.5).render(frame, [face])
        assert out.flags["C_CONTIGUOUS"]

    def test_scale_sets_the_output_size(self, frame, face, lmk):
        from dms.viz import OverlayRenderer

        out = OverlayRenderer(enabled=True, scale=0.5).render(frame, [face], landmarks106=lmk)
        assert out.shape == (120, 160, 3)

    def test_max_width_caps_the_canvas(self, frame, face):
        from dms.viz import OverlayRenderer

        r = OverlayRenderer(enabled=True, scale=1.0, max_width=160)
        assert r.render(frame, [face]).shape[1] == 160

    def test_thermal_gate_sheds_and_restores(self, frame, face):
        from dms.viz import OverlayRenderer

        r = OverlayRenderer(enabled=True, scale=1.0)
        r.thermal_gate(True)
        assert r.enabled is False and r.configured is True
        assert r.render(frame, [face]) is None
        r.thermal_gate(False)
        assert r.render(frame, [face]) is not None

    def test_rejects_an_impossible_scale(self):
        from dms.viz import OverlayRenderer

        with pytest.raises(ValueError):
            OverlayRenderer(enabled=True, scale=0.0)
        with pytest.raises(ValueError):
            OverlayRenderer(enabled=True, scale=1.5)

    def test_shared_null_renderer_is_off(self):
        from dms.viz import NULL_RENDERER

        assert NULL_RENDERER.enabled is False


class TestDrawFaces:
    def test_source_frame_is_never_modified(self, frame, face, lmk):
        from dms.viz import draw_faces

        before = frame.copy()
        draw_faces(frame, [face], driver=face, landmarks106=lmk, ear=0.3, mar=0.2,
                   pose=(1.0, 2.0, 3.0), track_id=1, alerts=["a:b"], yaw_rel=4.0)
        assert np.array_equal(frame, before)

    def test_something_is_actually_drawn(self, frame, face, lmk):
        from dms.viz import draw_faces

        out = draw_faces(frame, [face], driver=face, landmarks106=lmk)
        assert not np.array_equal(out, frame)

    def test_legacy_keyword_signature_still_works(self, frame, face, lmk):
        """dms/app.py calls this exact shape; changing it would break the app."""
        from dms.viz.overlay import draw_faces

        out = draw_faces(
            frame, [face], driver=face, landmarks106=lmk, ear=0.31, mar=0.22,
            pose=(1.0, 2.0, 3.0), track_id=5, alerts=["x:y"], yaw_rel=-3.0,
            occupant_labels=None,
        )
        assert out.shape == frame.shape

    def test_out_buffer_is_used_in_place(self, frame, face):
        from dms.viz.overlay import draw_faces

        buf = np.zeros_like(frame)
        out = draw_faces(frame, [face], out=buf)
        assert out is buf

    def test_out_buffer_of_the_wrong_shape_is_refused(self, frame, face):
        from dms.viz.overlay import draw_faces

        with pytest.raises(ValueError):
            draw_faces(frame, [face], out=np.zeros((10, 10, 3), np.uint8))

    def test_wrong_dtype_is_refused(self, face):
        from dms.viz.overlay import draw_faces

        with pytest.raises(ValueError):
            draw_faces(np.zeros((10, 10, 3), np.float32), [face])

    def test_landmarks_outside_the_frame_are_dropped_not_clamped(self, frame):
        from dms.viz.overlay import draw_faces

        far = np.full((106, 2), 10000.0, dtype=np.float32)
        out = draw_faces(frame, [], landmarks106=far)
        # Only the HUD text at the top-left may differ; no border artefacts.
        assert np.array_equal(out[100:, :], frame[100:, :])

    def test_occupant_labels_are_applied(self, frame, face):
        from dms.viz.overlay import draw_faces

        class Track(object):
            pass

        tr = Track()
        tr.face = face
        out = draw_faces(frame, [face], driver=face, occupant_labels=[(tr, "PAX")])
        assert out.shape == frame.shape

    def test_non_106_point_sets_do_not_crash(self, frame):
        from dms.viz.overlay import draw_faces

        pts = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        draw_faces(frame, [], landmarks106=pts)


class TestFrameContiguity:
    """PERF-03 guard: the invariant that made three hidden 2.7 MB copies a frame."""

    def test_the_bgra_alpha_slice_is_not_contiguous(self):
        bgra = np.zeros((72, 128, 4), dtype=np.uint8)
        assert not bgra[..., :3].flags["C_CONTIGUOUS"]

    def test_bgra_to_bgr_produces_a_contiguous_frame(self):
        from dms.runtime.preprocess import bgra_to_bgr

        bgra = np.zeros((72, 128, 4), dtype=np.uint8)
        out = bgra_to_bgr(bgra)
        assert out.flags["C_CONTIGUOUS"] and out.shape == (72, 128, 3)

    def test_overlay_accepts_a_strided_view_but_the_output_is_contiguous(self):
        from dms.viz.overlay import draw_faces

        bgra = np.zeros((72, 128, 4), dtype=np.uint8)
        out = draw_faces(bgra[..., :3], [])
        assert out.flags["C_CONTIGUOUS"]


# ================================================================== scripts


class TestBenchScript:
    def test_stats_of_an_empty_sample_set(self):
        bench = _load_script("bench")
        assert bench._stats([]) == {"n": 0}

    def test_stats_percentiles(self):
        bench = _load_script("bench")
        st = bench._stats([float(i) for i in range(1, 101)])
        assert st["n"] == 100
        assert st["p50_ms"] == pytest.approx(50.0, abs=1.0)
        assert st["p95_ms"] == pytest.approx(95.0, abs=1.0)
        assert st["min_ms"] == 1.0 and st["max_ms"] == 100.0

    def test_gpu_lock_detects_an_inherited_lock(self, monkeypatch):
        bench = _load_script("bench")
        monkeypatch.setenv("DMS_GPU_LOCK_HELD", "1")
        lock = bench.GpuLock()
        assert lock.inherited is True and lock.enabled is False
        with lock:
            pass  # must not block

    def test_markdown_renders_without_optional_sections(self):
        bench = _load_script("bench")
        md = bench.to_markdown({"generated_utc": "now"})
        assert "DMS bench" in md

    def test_cadence_gate(self):
        bench = _load_script("bench")
        cad = {"objects": (10, 7), "identity": (0, 0)}
        assert bench._due(cad, "objects", 7) is True
        assert bench._due(cad, "objects", 8) is False
        assert bench._due(cad, "identity", 0) is True
        assert bench._due(cad, "identity", 1) is False
        assert bench._due(cad, "headpose", 3) is True  # not listed = every frame
        assert bench._due(None, "objects", 3) is True

    def test_parse_cadence(self):
        bench = _load_script("bench")
        cad = bench.parse_cadence("objects=20:7,occlusion=15:3")
        assert cad["objects"] == (20, 7)
        assert cad["occlusion"] == (15, 3)
        assert cad["headpose"] == bench.PRODUCTION_CADENCE["headpose"]  # untouched

    def test_parse_cadence_rejects_a_malformed_spec(self):
        bench = _load_script("bench")
        with pytest.raises(ValueError):
            bench.parse_cadence("objects=every-other-frame")

    def test_overridden_cadence_still_staggers(self):
        bench = _load_script("bench")
        cad = bench.parse_cadence("objects=20:7,occlusion=15:3")
        both = [i for i in range(600)
                if bench._due(cad, "objects", i) and bench._due(cad, "occlusion", i)]
        assert both == []

    def test_production_cadence_staggers_the_expensive_heads(self):
        """Coincident spikes are what turn a good p50 into a bad p95."""
        bench = _load_script("bench")
        cad = bench.PRODUCTION_CADENCE
        both = [i for i in range(300)
                if bench._due(cad, "objects", i) and bench._due(cad, "occlusion", i)]
        assert both == []


class TestRunAllReplay:
    def test_parse_log_extracts_latency_and_frames(self, tmp_path):
        mod = _load_script("run_all_replay")
        p = tmp_path / "log.txt"
        p.write_text(
            "INFO frames=15 fps=6.7 infer_p95_ms=79.0 faces_last=1 ear=0.3 mar=0.2 pose=None\n"
            "INFO frames=30 fps=6.6 infer_p95_ms=67.0 faces_last=1 ear=0.3 mar=0.2 pose=None\n"
            "INFO ALERT rise microsleep critical\n"
            "WARNING CLOCKS_LOW gpu_clock_mhz=114\n"
            "INFO stopped frames=30 last_id=30 mean_infer_ms=55.6 total_face_hits=29\n"
        )
        got = mod.parse_log(str(p))
        assert got["frames"] == 30
        assert got["infer_p95_ms"] == 67.0  # the last periodic line, not the first
        assert got["mean_infer_ms"] == 55.6
        assert got["total_face_hits"] == 29
        assert got["alert_lines"] == 1
        assert got["warnings"]["CLOCKS_LOW"] == 1

    def test_parse_log_of_a_missing_file_returns_nulls_not_zeros(self):
        mod = _load_script("run_all_replay")
        got = mod.parse_log("/nonexistent/log.txt")
        assert got["frames"] is None and got["mean_infer_ms"] is None

    def test_parse_events_counts_by_type(self, tmp_path):
        mod = _load_script("run_all_replay")
        p = tmp_path / "events.jsonl"
        p.write_text(
            json.dumps({"type": "microsleep"}) + "\n"
            + json.dumps({"type": "microsleep"}) + "\n"
            + "not json\n"
            + json.dumps({"type": "distraction"}) + "\n"
        )
        assert mod.parse_events(str(p)) == {"microsleep": 2, "distraction": 1}

    def test_parse_events_absent_is_none(self):
        mod = _load_script("run_all_replay")
        assert mod.parse_events("/nonexistent/events.jsonl") is None

    def test_read_budget_absent_is_none(self):
        mod = _load_script("run_all_replay")
        assert mod.read_budget("/nonexistent/latency_budget.json") is None

    def test_read_budget_extracts_the_gate(self, tmp_path):
        mod = _load_script("run_all_replay")
        p = tmp_path / "latency_budget.json"
        p.write_text(json.dumps({"gate": {"pass": False}, "offenders": ["detect_gpu"]}))
        got = mod.read_budget(str(p))
        assert got["gate"]["pass"] is False and got["offenders"] == ["detect_gpu"]

    def test_compare_flags_a_slowdown(self, tmp_path):
        mod = _load_script("run_all_replay")
        base = tmp_path / "summary.json"
        base.write_text(json.dumps({"clips": [
            {"clip": "day_driver", "seconds": 100.0, "fps_wall": 15.0,
             "mean_infer_ms": 40.0, "infer_p95_ms": 50.0, "peak_rss_mb": 1000},
        ]}))
        now = [{"clip": "day_driver", "seconds": 150.0, "fps_wall": 10.0,
                "mean_infer_ms": 60.0, "infer_p95_ms": 75.0, "peak_rss_mb": 1010}]
        cmp_ = mod.compare(now, str(base), 10.0)
        assert cmp_["status"] == "ok"
        joined = " ".join(cmp_["regressions"])
        assert "seconds" in joined and "fps_wall" in joined
        assert "peak_rss_mb" not in joined  # 1% is inside tolerance

    def test_compare_reports_an_improvement_without_flagging_it(self, tmp_path):
        mod = _load_script("run_all_replay")
        base = tmp_path / "summary.json"
        base.write_text(json.dumps({"clips": [
            {"clip": "c", "seconds": 100.0, "fps_wall": 10.0, "mean_infer_ms": 40.0,
             "infer_p95_ms": 50.0, "peak_rss_mb": 1000},
        ]}))
        now = [{"clip": "c", "seconds": 50.0, "fps_wall": 20.0, "mean_infer_ms": 20.0,
                "infer_p95_ms": 25.0, "peak_rss_mb": 900}]
        cmp_ = mod.compare(now, str(base), 10.0)
        assert cmp_["regressions"] == []
        assert cmp_["clips"]["c"]["fps_wall"]["delta_pct"] == 100.0

    def test_compare_with_a_missing_baseline_is_reported_not_raised(self):
        mod = _load_script("run_all_replay")
        assert mod.compare([], "/nonexistent/summary.json", 10.0)["status"] == "unavailable"

    def test_sampler_reads_this_process(self):
        mod = _load_script("run_all_replay")
        s = mod.ChildSampler(os.getpid(), period_s=0.01).start()
        time.sleep(0.05)
        s.stop()
        d = s.as_dict()
        assert d["peak_rss_mb"] is None or d["peak_rss_mb"] > 0
        assert d["resource_samples"] >= 1

    def test_config_for_clip_maps_day_driver_to_default(self):
        mod = _load_script("run_all_replay")
        assert mod.config_for_clip("/x/day_driver.mp4") == "configs/default.yaml"


class TestCalibrateForward:
    def test_a_clean_calibration_passes(self):
        mod = _load_script("calibrate_forward")
        veh = {"forward_zero": {"yaw": 2.0, "pitch": -3.0, "roll": 0.0},
               "state": {"ear_open_median": 0.72}}
        assert mod.check_result(veh, samples=100) == []

    def test_a_wild_yaw_is_rejected(self):
        mod = _load_script("calibrate_forward")
        veh = {"forward_zero": {"yaw": 62.0, "pitch": 0.0}, "state": {"ear_open_median": 0.7}}
        problems = mod.check_result(veh)
        assert any("yaw" in p for p in problems)

    def test_a_closed_eye_calibration_is_rejected(self):
        mod = _load_script("calibrate_forward")
        veh = {"forward_zero": {"yaw": 0.0, "pitch": 0.0}, "state": {"ear_open_median": 0.02}}
        assert any("ear_open_median" in p for p in mod.check_result(veh))

    def test_too_few_samples_is_rejected(self):
        mod = _load_script("calibrate_forward")
        veh = {"forward_zero": {"yaw": 0.0, "pitch": 0.0}, "state": {"ear_open_median": 0.7}}
        assert any("frames contributed" in p for p in mod.check_result(veh, samples=3))

    def test_missing_fields_are_rejected(self):
        mod = _load_script("calibrate_forward")
        assert mod.check_result({}) != []

    def test_vehicle_yaml_sits_beside_the_config(self):
        mod = _load_script("calibrate_forward")
        assert mod.vehicle_yaml_path("/a/b/configs/production.yaml") == "/a/b/configs/vehicle.yaml"
