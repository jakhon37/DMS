"""Unit spec for dms/state/fusion.py.

These tests pin the primitives the alert machine is built on: the time-constant
filters, the Schmitt band, and - the important one - that PERCLOS is a TIME
integral with an explicit validity test, not a frame count.
"""
from __future__ import annotations

import math
import os

import pytest

from dms.config.schema import load_config
from dms.state.fusion import (
    ABOVE,
    BELOW,
    CircularEwma,
    FusionConfig,
    PerclosWindow,
    SchmittLatch,
    SignalFuser,
    angdiff,
    ewma_alpha,
    finite,
    wrap180,
)
from dms.types import (
    DriverSignals,
    EyeState,
    GazeVector,
    HeadPose,
    MouthState,
    Occlusion,
    SignalSource,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cfg():
    return load_config(os.path.join(ROOT, "configs/default.yaml"))


# ------------------------------------------------------------- primitives ---
def test_finite_rejects_nan_none_and_strings():
    assert finite(1.5) == 1.5
    assert finite(None) is None
    assert finite(float("nan")) is None
    assert finite(float("inf")) is None
    assert finite("x", 3.0) == 3.0


def test_ewma_alpha_is_time_based_not_frame_based():
    # One time constant of elapsed time always mixes in the same 63%, whatever
    # the frame rate.  That is the whole point.
    assert ewma_alpha(0.25, 0.25) == pytest.approx(1.0 - math.exp(-1.0))
    slow = ewma_alpha(1.0 / 6.0, 0.25)
    fast = ewma_alpha(1.0 / 15.0, 0.25)
    assert slow > fast > 0.0
    # two fast steps approximate one slow step of the same total duration
    assert (1 - (1 - fast) ** 2.5) == pytest.approx(slow, abs=0.02)
    assert ewma_alpha(0.1, 0.0) == 1.0
    assert ewma_alpha(0.0, 0.25) == 0.0


def test_wrap180_and_angdiff():
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(180.0) == pytest.approx(180.0)
    assert wrap180(float("nan")) == 0.0
    assert angdiff(-63.0, -63.0) == pytest.approx(0.0)
    assert angdiff(170.0, -170.0) == pytest.approx(-20.0)
    assert angdiff(20.0, -63.0) == pytest.approx(83.0)


def test_circular_ewma_does_not_walk_across_the_wrap():
    """A linear EWMA of +179 and -179 gives 0; the circular one gives 180."""
    f = CircularEwma(tau_s=0.25)
    assert f.value is None
    f.update(179.0, 0.0)
    f.update(-179.0, 10.0)  # 40 time constants: fully converged on the new one
    assert abs(abs(f.value) - 179.0) < 1.0
    g = CircularEwma(tau_s=0.25)
    g.update(179.0, 0.0)
    g.update(-179.0, 0.25)
    assert abs(g.value) > 170.0, "a linear average would have landed near 0"


def test_circular_ewma_ignores_non_finite_samples():
    f = CircularEwma(tau_s=0.25)
    f.update(10.0, 0.0)
    f.update(float("nan"), 0.1)
    assert f.value == pytest.approx(10.0)


def test_schmitt_band_has_hysteresis():
    s = SchmittLatch(enter_thr=0.40, exit_thr=0.55, direction=BELOW)
    assert s.band(0.30) is True
    s.update(0.30, 0.1)
    assert s.state is True
    # inside the band the latch holds its previous state
    assert s.band(0.50) is True
    assert s.band(0.60) is False


def test_schmitt_dwell_leaks_instead_of_resetting():
    """One contrary frame must cost one frame of credit, not the whole run."""
    dt = 0.1
    s = SchmittLatch(0.40, 0.55, BELOW, enter_dwell_s=1.0, exit_dwell_s=0.1)
    for _ in range(9):
        s.update(0.10, dt)
    assert s.state is False
    s.update(0.90, dt)  # a single glitch
    assert s.state is False
    s.update(0.10, dt)
    assert s.state is False, "leak should have cost exactly one frame"
    s.update(0.10, dt)
    assert s.state is True


def test_schmitt_above_direction():
    s = SchmittLatch(enter_thr=0.65, exit_thr=0.50, direction=ABOVE, enter_dwell_s=0.2)
    s.update(0.70, 0.1)
    assert s.state is False
    s.update(0.70, 0.1)
    assert s.state is True
    s.update(0.60, 0.1)  # inside the band, holds
    assert s.state is True
    s.update(0.40, 0.1)
    assert s.state is False


def test_schmitt_none_value_changes_nothing():
    s = SchmittLatch(0.40, 0.55, BELOW)
    s.update(0.1, 0.1)
    assert s.update(None, 0.1) is True
    assert s.band(None) is True


def test_schmitt_rejects_a_bad_direction():
    with pytest.raises(ValueError):
        SchmittLatch(0.4, 0.5, "sideways")


# ----------------------------------------------------------------- perclos ---
def test_perclos_is_time_weighted_not_frame_counted():
    """Closed sampled fast, open sampled slow: a frame count would read 0.71."""
    w = PerclosWindow(window_s=120.0, min_span_s=0.0, min_rate_hz=0.0, min_coverage=0.0)
    t = 0.0
    n_closed = n_open = 0
    for _ in range(10):
        for _ in range(45):  # 3 s of closure at 15 Hz
            w.add(t, 1 / 15.0, True)
            t += 1 / 15.0
            n_closed += 1
        for _ in range(18):  # 3 s of open at 6 Hz
            w.add(t, 1 / 6.0, False)
            t += 1 / 6.0
            n_open += 1
    count_weighted = n_closed / float(n_closed + n_open)
    assert count_weighted == pytest.approx(0.714, abs=0.01)
    assert w.value == pytest.approx(0.5, abs=0.02)


def test_perclos_needs_span_rate_and_coverage():
    w = PerclosWindow(window_s=60.0, min_span_s=30.0, min_rate_hz=8.0, min_coverage=0.6)
    assert not w.valid and not w.starved  # empty: warming up, not broken

    # 6 FPS for 40 s: long enough, but below the 8 Hz the design requires.
    t = 0.0
    for _ in range(int(40 * 6)):
        w.add(t, 1 / 6.0, False)
        t += 1 / 6.0
    assert w.span_s > 30.0
    assert w.rate_hz == pytest.approx(6.0, abs=0.1)
    assert not w.valid, "6 FPS must not produce a PERCLOS number"
    assert w.starved, "long enough and still invalid is a real degradation"


def test_perclos_is_valid_at_ten_hz():
    w = PerclosWindow(window_s=60.0, min_span_s=30.0, min_rate_hz=8.0, min_coverage=0.6)
    t = 0.0
    for i in range(400):  # 40 s at 10 Hz, half closed
        w.add(t, 0.1, (i % 10) < 5)
        t += 0.1
    assert w.valid
    assert not w.starved
    assert w.value == pytest.approx(0.5, abs=0.02)
    assert w.coverage == pytest.approx(1.0, abs=0.02)


def test_perclos_coverage_falls_when_samples_stop():
    """A driver absent for half the window leaves coverage around 0.5."""
    w = PerclosWindow(window_s=60.0, min_span_s=10.0, min_rate_hz=0.0, min_coverage=0.6)
    t = 0.0
    for _ in range(200):  # 20 s at 10 Hz
        w.add(t, 0.1, False)
        t += 0.1
    assert w.valid
    t += 20.0  # 20 s with no face: nothing is added
    w.add(t, 0.1, False)
    assert w.coverage < 0.6
    assert not w.valid


def test_perclos_span_is_measured_to_now_not_to_the_newest_sample():
    """A window nobody feeds any more must stop being valid, not go stale."""
    w = PerclosWindow(window_s=60.0, min_span_s=30.0, min_rate_hz=8.0, min_coverage=0.6)
    t = 0.0
    for _ in range(400):  # 40 s at 10 Hz
        w.add(t, 0.1, False)
        t += 0.1
    assert w.valid
    t += 20.0  # 20 s with no usable eye measurement at all
    w.advance(t)
    assert w.span_s == pytest.approx(59.9, abs=0.2)
    assert w.rate_hz < 8.0
    assert not w.valid and w.starved


def test_perclos_clamps_a_long_gap():
    """A 5 s gap must not be credited to whichever state came after it."""
    w = PerclosWindow(min_span_s=0.0, min_rate_hz=0.0, min_coverage=0.0, max_gap_s=0.5)
    w.add(0.0, 0.0, False)
    w.add(1.0, 0.1, False)
    w.add(6.0, 5.0, True)
    assert w.value == pytest.approx(0.5 / 0.6, abs=1e-6)


def test_perclos_window_slides_and_resets():
    w = PerclosWindow(window_s=10.0, min_span_s=0.0, min_rate_hz=0.0, min_coverage=0.0)
    t = 0.0
    for _ in range(100):
        w.add(t, 0.1, True)
        t += 0.1
    assert w.value == pytest.approx(1.0)
    for _ in range(100):
        w.add(t, 0.1, False)
        t += 0.1
    assert w.value == pytest.approx(0.0, abs=1e-6)
    w.reset()
    assert w.value == 0.0 and not w.valid


# ------------------------------------------------------------------ fuser ---
def _eye(prob=None, ear=None, source=SignalSource.MODEL):
    return EyeState(
        prob_open=0.0 if prob is None else prob,
        ear=0.0 if ear is None else ear,
        source=source,
    )


def _signals(**kw):
    return DriverSignals(**kw)


def _fuser(**overrides):
    cfg = FusionConfig()
    cfg.calibrated = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SignalFuser(cfg)


def test_model_wins_over_geometry_when_both_are_present():
    f = _fuser()
    s = _signals(
        left_eye=EyeState(prob_open=0.05, ear=0.90, source=SignalSource.MODEL),
        right_eye=EyeState(prob_open=0.05, ear=0.90, source=SignalSource.MODEL),
    )
    v = f.update(0.0, True, s)
    assert v.eyes.source is SignalSource.MODEL
    assert v.eye_closed_instant is True, "the model says shut; the EAR is ignored"
    assert v.eye_open_score == pytest.approx(0.05)


def test_geometry_is_used_when_the_model_is_unavailable():
    f = _fuser(ear_closed=0.20, ear_open_median=0.0)
    s = _signals(
        left_eye=_eye(ear=0.10, source=SignalSource.GEOMETRY),
        right_eye=_eye(ear=0.10, source=SignalSource.GEOMETRY),
    )
    v = f.update(0.0, True, s)
    assert v.eyes.source is SignalSource.GEOMETRY
    assert v.eyes.reliable is True
    assert v.eye_closed_instant is True
    assert v.eyes.confidence <= 0.6, "a contour EAR may never claim model-grade confidence"


def test_no_eye_signal_at_all_is_unreliable_not_zero():
    f = _fuser()
    v = f.update(0.0, True, _signals())
    assert v.eyes.source is SignalSource.UNAVAILABLE
    assert v.eyes.reliable is False
    assert v.eyes.reason == "no_signal"
    assert v.reliable is False


def test_absent_driver_is_never_reliable():
    f = _fuser()
    s = _signals(left_eye=_eye(prob=0.9), right_eye=_eye(prob=0.9))
    v = f.update(0.0, False, s)
    assert v.eyes.reliable is False and v.eyes.reason == "absent"


def test_sunglasses_make_the_eye_channel_unreliable_without_faking_a_value():
    f = _fuser()
    s = _signals(
        left_eye=_eye(prob=0.05),
        right_eye=_eye(prob=0.05),
        occlusion=Occlusion(sunglasses=0.9, source=SignalSource.MODEL),
    )
    v = f.update(0.0, True, s)
    assert v.eyes_occluded is True
    assert v.eyes.reliable is False and v.eyes.reason == "occluded"
    assert v.eyes.source is SignalSource.MODEL, "provenance is still reported"


def test_mask_makes_the_mouth_channel_unreliable():
    f = _fuser()
    s = _signals(
        mouth=MouthState(prob_open=0.9, source=SignalSource.MODEL),
        occlusion=Occlusion(mask=0.8, source=SignalSource.MODEL),
    )
    v = f.update(0.0, True, s)
    assert v.mouth_occluded is True and v.mouth.reliable is False


def test_an_absent_occlusion_model_does_not_invent_an_occlusion():
    f = _fuser()
    s = _signals(
        left_eye=_eye(prob=0.05),
        right_eye=_eye(prob=0.05),
        occlusion=Occlusion(sunglasses=0.99, source=SignalSource.UNAVAILABLE),
    )
    v = f.update(0.0, True, s)
    assert v.eyes_occluded is False and v.eyes.reliable is True


def test_pose_below_its_confidence_gate_is_unusable():
    f = _fuser(pose_min_conf=0.35)
    s = _signals(pose=HeadPose(yaw=10.0, confidence=0.10, source=SignalSource.MODEL))
    v = f.update(0.0, True, s)
    assert v.pose.reliable is False and v.pose.reason == "low_confidence"


def test_pose_with_a_high_reprojection_residual_is_unusable():
    f = _fuser(pose_max_reproj_px=6.0)
    s = _signals(
        pose=HeadPose(yaw=10.0, confidence=0.9, reprojection_error=40.0, source=SignalSource.GEOMETRY)
    )
    v = f.update(0.0, True, s)
    assert v.pose.reliable is False and v.pose.reason == "high_reprojection"


def test_uncalibrated_pose_is_reported_but_never_reliable():
    f = _fuser(calibrated=False)
    s = _signals(pose=HeadPose(yaw=40.0, confidence=0.9, source=SignalSource.MODEL))
    v = f.update(0.0, True, s)
    assert v.pose.reason == "uncalibrated" and v.pose.reliable is False
    assert v.gaze.reliable is False
    assert v.gaze_off_road is False, "no forward zero means off-road is undefined"


def test_gaze_needs_a_usable_pose_however_good_the_eye_model_is():
    f = _fuser()
    s = _signals(
        pose=HeadPose(confidence=0.0, source=SignalSource.UNAVAILABLE),
        gaze=GazeVector(yaw=45.0, pitch=0.0, confidence=0.99, source=SignalSource.MODEL),
    )
    v = f.update(0.0, True, s)
    assert v.gaze.source is SignalSource.UNAVAILABLE
    assert v.gaze_off_road is False


def test_gaze_adds_eye_direction_to_head_pose():
    f = _fuser(gaze_yaw_deg=25.0, pose_tau_s=0.0, gaze_tau_s=0.0)
    s = _signals(
        pose=HeadPose(yaw=15.0, confidence=0.9, source=SignalSource.MODEL),
        gaze=GazeVector(yaw=20.0, pitch=0.0, confidence=0.9, source=SignalSource.MODEL),
    )
    v = f.update(0.0, True, s)
    assert v.yaw_rel == pytest.approx(15.0, abs=0.5)
    assert v.gaze_yaw == pytest.approx(35.0, abs=0.5)
    assert v.gaze_off_road is True
    assert v.gaze.reason == "eye+pose"


def test_gaze_falls_back_to_pose_only_when_the_eye_model_is_quiet():
    f = _fuser(pose_tau_s=0.0)
    s = _signals(pose=HeadPose(yaw=40.0, confidence=0.9, source=SignalSource.MODEL))
    v = f.update(0.0, True, s)
    assert v.gaze.reason == "pose_only"
    assert v.gaze_off_road is True


def test_forward_zero_is_subtracted():
    f = _fuser(forward_yaw=-63.0, forward_pitch=-6.0, pose_tau_s=0.0, calibrated=True)
    s = _signals(pose=HeadPose(yaw=-62.0, pitch=-7.0, confidence=0.9, source=SignalSource.MODEL))
    v = f.update(0.0, True, s)
    assert v.yaw_rel == pytest.approx(1.0, abs=0.2)
    assert v.pitch_rel == pytest.approx(-1.0, abs=0.2)
    assert v.gaze_off_road is False


def test_perclos_only_integrates_present_measured_frames():
    f = _fuser(perclos_min_span_s=0.0, perclos_min_rate_hz=0.0, perclos_min_coverage=0.0)
    closed = _signals(left_eye=_eye(prob=0.05), right_eye=_eye(prob=0.05))
    t = 0.0
    for _ in range(20):
        f.update(t, True, closed)
        t += 0.1
    assert f.view.perclos == pytest.approx(1.0)
    # 20 absent frames must not extend the closed integral
    before = f.perclos._sum_dt
    for _ in range(20):
        f.update(t, False, closed)
        t += 0.1
    assert f.perclos._sum_dt == pytest.approx(before)


def test_a_gap_longer_than_max_gap_integrates_nothing():
    f = _fuser(max_gap_s=0.5, perclos_min_span_s=0.0, perclos_min_rate_hz=0.0, perclos_min_coverage=0.0)
    closed = _signals(left_eye=_eye(prob=0.05), right_eye=_eye(prob=0.05))
    f.update(0.0, True, closed)
    v = f.update(30.0, True, closed)
    assert v.dt_s == 0.0


def test_reset_wipes_every_window_but_can_keep_the_clock():
    f = _fuser(perclos_min_span_s=0.0, perclos_min_rate_hz=0.0, perclos_min_coverage=0.0)
    closed = _signals(left_eye=_eye(prob=0.05), right_eye=_eye(prob=0.05))
    t = 0.0
    for _ in range(20):
        f.update(t, True, closed)
        t += 0.1
    assert f.view.perclos > 0.9
    f.reset(keep_clock=True)
    assert f.perclos.value == 0.0
    assert f._t_prev == pytest.approx(t - 0.1)
    f.reset()
    assert f._t_prev is None


def test_update_never_raises_on_a_frame_with_nothing_in_it():
    f = _fuser()
    v = f.update(float("nan"), False, None)
    assert v.present is False and v.reliable is False


def test_fusion_config_reads_the_app_config_and_infers_calibration():
    cfg = _cfg()
    cfg.forward_zero.yaw = 0.0
    cfg.forward_zero.pitch = 0.0
    cfg.forward_zero.roll = 0.0
    fc = FusionConfig.from_app_config(cfg)
    assert fc.calibrated is False
    assert fc.gaze_yaw_deg == pytest.approx(cfg.state.gaze_yaw_deg)
    assert fc.mar_yawn == pytest.approx(cfg.state.mar_yawn)
    cfg.forward_zero.yaw = -61.9
    assert FusionConfig.from_app_config(cfg).calibrated is True


def test_effective_ear_threshold_matches_the_documented_formula():
    fc = FusionConfig(ear_closed=0.50, ear_open_median=0.27)
    assert fc.ear_closed_effective == pytest.approx(0.70 * 0.27)
    fc2 = FusionConfig(ear_closed=0.18, ear_open_median=0.27)
    assert fc2.ear_closed_effective == pytest.approx(0.18)


# ------------------------------------------- one generation of thresholds ---
# Two defects lived here.  (1) The FusionConfig dataclass defaults were the OLD
# broken-landmark EAR/MAR scale (0.50 / 0.75 / 0.65 / 0.50) that this project's
# own config schema rejects by name, so any FusionConfig() built without an
# AppConfig read every open eye as closed.  (2) The four Schmitt latches copied
# their thresholds at __init__ while _fuse_eyes re-read cfg live every frame, so
# one frame could be judged against two threshold generations and a retune
# silently stopped moving the decision boundary.

def test_fusion_config_defaults_are_the_corrected_scale_and_the_schema_agrees():
    """The dataclass defaults and dms/config/schema.py cannot drift apart.

    Both must equal the geometry modules that own these numbers, and the
    dataclass defaults must survive the schema's own validators - the ones that
    reject the old scale by name.
    """
    from dms.config.schema import StateConfig
    from dms.geometry.ear import EAR_CLOSED_RECOMMENDED, EAR_OPEN_MEDIAN_RECOMMENDED
    from dms.geometry.mar import MAR_YAWN_RECOMMENDED

    fc = FusionConfig()
    schema_default = StateConfig()
    assert fc.ear_closed == pytest.approx(EAR_CLOSED_RECOMMENDED)
    assert fc.ear_open_median == pytest.approx(EAR_OPEN_MEDIAN_RECOMMENDED)
    assert fc.mar_yawn == pytest.approx(MAR_YAWN_RECOMMENDED)
    for name in ("ear_closed", "ear_open_median", "mar_yawn", "mar_close"):
        assert getattr(fc, name) == pytest.approx(getattr(schema_default, name)), (
            "FusionConfig.%s (%r) has drifted from StateConfig.%s (%r)"
            % (name, getattr(fc, name), name, getattr(schema_default, name))
        )
    # The validators that raise on the old scale must accept these values.
    StateConfig(
        ear_closed=fc.ear_closed,
        ear_open_median=fc.ear_open_median,
        mar_yawn=fc.mar_yawn,
        mar_close=fc.mar_close,
    )


def test_a_default_fusion_config_calls_a_real_open_eye_open():
    """Measured EAR on this board's own replay clips is 0.34-0.35.

    Under the old 0.50 default every one of those frames read CLOSED, i.e.
    MICROSLEEP on every frame.  Pin the opposite.
    """
    fc = FusionConfig()
    for measured in (0.3514, 0.3444, 0.3533):
        assert measured > fc.ear_closed_effective


def _geom_eye_signals(ear):
    return _signals(
        left_eye=_eye(ear=ear, source=SignalSource.GEOMETRY),
        right_eye=_eye(ear=ear, source=SignalSource.GEOMETRY),
    )


def _run(f, sig, n, t0=0.0, dt=0.1):
    v = None
    for i in range(n):
        v = f.update(t0 + i * dt, True, sig)
    return v


def test_retuning_ear_closed_moves_the_latch_not_just_the_confidence():
    """A post-construction retune must change the DECISION, not only the number
    printed next to it.  Before the fix the latch stayed on the boot-time
    threshold while the confidence was computed from the new one."""
    f = _fuser(ear_closed=0.18, ear_open_median=0.27)
    v = _run(f, _geom_eye_signals(0.20), 10, 0.0)
    assert v.eye_closed is False
    f.cfg.ear_closed = 0.50
    f.cfg.ear_open_median = 0.75
    v = _run(f, _geom_eye_signals(0.20), 10, 1.0)
    assert f._eye_geom.enter_thr == pytest.approx(f.cfg.ear_closed_effective)
    assert v.eye_closed is True
    assert v.eye_closed_instant is True


def test_retuning_mar_yawn_moves_the_mouth_latch_too():
    f = _fuser(mar_yawn=0.45, mar_close=0.35)
    sig = _signals(mouth=MouthState(mar=0.40, source=SignalSource.GEOMETRY))
    v = _run(f, sig, 10, 0.0)
    assert v.mouth_open is False
    f.cfg.mar_yawn = 0.30
    f.cfg.mar_close = 0.20
    v = _run(f, sig, 10, 1.0)
    assert f._mouth_geom.enter_thr == pytest.approx(0.30)
    assert v.mouth_open is True


def test_every_threshold_the_fuser_holds_is_re_read_every_frame():
    """No latch, window or filter may cache a tunable across a frame boundary."""
    f = _fuser()
    cfg = f.cfg
    cfg.eye_prob_closed = 0.11
    cfg.eye_prob_open = 0.12
    cfg.ear_closed = 0.13
    cfg.ear_open_median = 0.90        # effective = min(0.13, 0.63) = 0.13
    cfg.ear_hysteresis = 0.07
    cfg.eye_close_dwell_s = 0.31
    cfg.eye_open_dwell_s = 0.32
    cfg.mouth_prob_open = 0.61
    cfg.mouth_prob_closed = 0.62
    cfg.mar_yawn = 0.63
    cfg.mar_close = 0.64
    cfg.mouth_open_dwell_s = 0.33
    cfg.mouth_close_dwell_s = 0.34
    cfg.perclos_window_s = 41.0
    cfg.perclos_min_span_s = 21.0
    cfg.perclos_min_rate_hz = 3.0
    cfg.perclos_min_coverage = 0.42
    cfg.max_gap_s = 0.9
    cfg.pose_tau_s = 0.77
    cfg.gaze_tau_s = 0.88
    f.update(0.0, True, _geom_eye_signals(0.30))

    assert (f._eye_model.enter_thr, f._eye_model.exit_thr) == (0.11, 0.12)
    assert f._eye_geom.enter_thr == pytest.approx(0.13)
    assert f._eye_geom.exit_thr == pytest.approx(0.20)
    assert (f._eye_geom.enter_dwell_s, f._eye_geom.exit_dwell_s) == (0.31, 0.32)
    assert (f._eye_model.enter_dwell_s, f._eye_model.exit_dwell_s) == (0.31, 0.32)
    assert (f._mouth_model.enter_thr, f._mouth_model.exit_thr) == (0.61, 0.62)
    assert (f._mouth_geom.enter_thr, f._mouth_geom.exit_thr) == (0.63, 0.64)
    assert (f._mouth_geom.enter_dwell_s, f._mouth_geom.exit_dwell_s) == (0.33, 0.34)
    assert f.perclos.window_s == 41.0
    assert f.perclos.min_span_s == 21.0
    assert f.perclos.min_rate_hz == 3.0
    assert f.perclos.min_coverage == 0.42
    assert f.perclos.max_gap_s == 0.9
    assert f._yaw.tau_s == f._pitch.tau_s == f._roll.tau_s == 0.77
    assert f._gaze_yaw.tau_s == f._gaze_pitch.tau_s == 0.88


def test_a_retune_does_not_drop_a_live_eye_closure():
    """Re-applying thresholds must not reset latch state or the dwell clock.

    A calibration landing in the middle of a real microsleep must not restart
    the 1.5 s clock - that would be a safety regression hiding inside a fix.
    """
    f = _fuser(ear_closed=0.18, ear_open_median=0.27, eye_close_dwell_s=0.15)
    shut = _geom_eye_signals(0.05)
    v = _run(f, shut, 10, 0.0)
    assert v.eye_closed is True
    held_on = f._eye_geom._hold_on
    assert held_on > 0.15
    f.cfg.ear_closed = 0.17           # a small recalibration, 0.05 still shut
    v = f.update(1.0, True, shut)
    assert v.eye_closed is True, "the closure survived the retune"
    assert f._eye_geom._hold_on > held_on, "the dwell clock kept running"


def test_the_confidence_and_the_decision_use_the_same_threshold():
    """The reported eye confidence is |ear - thr| / thr.  It and the latch must
    be computed from one number, so a frame can never report high confidence in
    a decision the latch made against a different threshold."""
    f = _fuser(ear_closed=0.18, ear_open_median=0.27)
    f.cfg.ear_closed = 0.40
    f.cfg.ear_open_median = 0.60      # effective = min(0.40, 0.42) = 0.40
    v = _run(f, _geom_eye_signals(0.20), 5, 0.0)
    thr = f.cfg.ear_closed_effective
    assert f._eye_geom.enter_thr == pytest.approx(thr)
    assert v.eyes.confidence == pytest.approx(min(0.6, abs(0.20 - thr) / thr))
    assert v.eye_closed is True       # 0.20 < 0.40, and the latch agrees
