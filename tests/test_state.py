"""Executable specification of DMS alert behaviour.

Every test here drives :class:`dms.state.driver_state.DriverMonitor` with a
synthetic :class:`dms.types.DriverSignals` time series at a REAL frame rate and
asserts exact edge counts and edge timings.  The pipeline runs at a variable
6-15 FPS, so the same scenario is replayed at 6, 10 and 15 FPS wherever the
answer must not depend on the rate.

Read this file as the contract:

* a one-frame glitch never toggles an alert, at any rate;
* a duration threshold means the same number of SECONDS at every rate;
* PERCLOS is a time integral and is refused, not guessed, below 8 Hz;
* an occluded or missing signal produces DROWSINESS_UNRELIABLE, never a
  confident drowsiness or gaze alert;
* alert state never crosses a driver change;
* the edge stream is always ``enter (escalate)* exit`` per type.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import pytest

from dms.config.schema import load_config
from dms.state.alerts import EDGE_ENTER, EDGE_ESCALATE, EDGE_EXIT, AlertType, Severity
from dms.state.driver_state import AlertTuning, DriverMonitor, angdiff
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
FPS_CASES = (6.0, 10.0, 15.0)
#: float slop when comparing an accumulated dwell against its threshold
SLOP = 1e-6
MODEL = SignalSource.MODEL
UNAVAIL = SignalSource.UNAVAILABLE

OPEN = 0.95
SHUT = 0.03


# ------------------------------------------------------------------ setup ---
def _cfg(yaw=0.0, pitch=0.0, roll=0.0):
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    cfg.forward_zero.yaw = float(yaw)
    cfg.forward_zero.pitch = float(pitch)
    cfg.forward_zero.roll = float(roll)
    return cfg


def _mon(cfg=None, calibrated=True, **tuning):
    mon = DriverMonitor(cfg if cfg is not None else _cfg())
    mon.fusion_cfg.calibrated = bool(calibrated)
    for key, value in tuning.items():
        assert hasattr(mon.tuning, key), "no such tuning knob: %s" % key
        setattr(mon.tuning, key, value)
    return mon


def sig(
    prob_open=OPEN,
    mouth_prob=0.02,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    pose_conf=0.9,
    sunglasses=0.0,
    mask=0.0,
    phone=0.0,
    track_id=1,
    identity=None,
    gaze_yaw=None,
    gaze_pitch=None,
    gaze_conf=0.9,
    eyes=True,
    pose=True,
    occlusion=True,
):
    """One frame of model-sourced signals.  ``eyes``/``pose`` False = no model."""
    eye_src = MODEL if eyes else UNAVAIL
    gaze = GazeVector()
    if gaze_yaw is not None or gaze_pitch is not None:
        gaze = GazeVector(
            yaw=float(gaze_yaw or 0.0),
            pitch=float(gaze_pitch or 0.0),
            confidence=float(gaze_conf),
            source=MODEL,
        )
    return DriverSignals(
        track_id=int(track_id),
        left_eye=EyeState(prob_open=float(prob_open), source=eye_src),
        right_eye=EyeState(prob_open=float(prob_open), source=eye_src),
        mouth=MouthState(prob_open=float(mouth_prob), source=MODEL),
        pose=HeadPose(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            confidence=float(pose_conf),
            source=MODEL if pose else UNAVAIL,
        ),
        gaze=gaze,
        occlusion=Occlusion(
            sunglasses=float(sunglasses),
            mask=float(mask),
            source=MODEL if occlusion else UNAVAIL,
        ),
        phone_conf=float(phone),
        identity=identity,
    )


def drive(mon, fps, seconds, fn=None, t0=0.0, degraded=None, **kw):
    """Replay ``seconds`` of frames at ``fps``; return (events, last_t).

    ``fn(t, i)`` returns ``(present, DriverSignals)``.  With no ``fn`` every
    frame is present and built from ``kw`` via :func:`sig`.
    """
    dt = 1.0 / float(fps)
    n = int(round(seconds * fps))
    out = []
    last = t0
    for i in range(n):
        t = t0 + i * dt
        if fn is None:
            present, signals = True, sig(**kw)
        else:
            present, signals = fn(t, i)
        out += mon.update(t, present=present, signals=signals, degraded=degraded)
        last = t
    return out, last + dt


def ev(events, typ, edge=None):
    return [e for e in events if e.type is typ and (edge is None or e.edge == edge)]


def assert_edge_grammar(events, mon):
    """Every type's edge stream must read ``enter (escalate)* exit`` forever."""
    assert [e.t_mono_s for e in events] == sorted(e.t_mono_s for e in events)
    by_type = defaultdict(list)
    for e in events:
        by_type[e.type].append(e)
    for typ, items in by_type.items():
        active = False
        for e in items:
            if e.edge == EDGE_ENTER:
                assert not active, "%s entered twice without an exit" % typ
                active = True
            elif e.edge == EDGE_ESCALATE:
                assert active, "%s escalated while inactive" % typ
            elif e.edge == EDGE_EXIT:
                assert active, "%s exited while inactive" % typ
                active = False
            else:
                raise AssertionError("unknown edge %r" % e.edge)
        if typ is AlertType.DRIVER_CHANGED:
            assert not active, "DRIVER_CHANGED is momentary and must self-close"
            continue
        assert active == (typ in mon.active), "%s latch disagrees with the log" % typ


# ------------------------------------------------------------------ basics ---
def test_angdiff_wrap():
    assert angdiff(-63.0, -63.0) == pytest.approx(0.0)
    assert angdiff(170.0, -170.0) == pytest.approx(-20.0)


def test_a_fresh_monitor_raises_nothing_and_knows_it_is_blind():
    mon = _mon()
    events = mon.update(0.0, present=False)
    assert events == []
    assert mon.highest_severity() is None
    assert all(mon.unreliable.values())


def test_alert_tuning_reads_the_yaml_and_keeps_unknown_defaults():
    cfg = _cfg()
    tun = AlertTuning.from_app_config(cfg)
    assert tun.microsleep_s == pytest.approx(cfg.state.microsleep_s)
    assert tun.gaze_warn_s == pytest.approx(cfg.state.gaze_warn_s)
    assert tun.perclos_crit == pytest.approx(cfg.state.perclos_crit)
    assert tun.assume_moving is cfg.state.assume_moving
    # not present in StateConfig today, so the module default must survive
    assert tun.no_driver_grace_s == 10.0
    assert tun.distraction_gaze_s == 6.0


# -------------------------------------------------------------- microsleep ---
@pytest.mark.parametrize("fps", FPS_CASES)
def test_no_alert_ever_fires_on_a_single_bad_frame(fps):
    mon = _mon()

    def fn(t, i):
        glitch = i == int(2.0 * fps)
        return True, sig(
            prob_open=SHUT if glitch else OPEN,
            mouth_prob=0.99 if glitch else 0.02,
            yaw=80.0 if glitch else 0.0,
            phone=0.99 if glitch else 0.0,
        )

    events, _ = drive(mon, fps, 5.0, fn)
    assert ev(events, AlertType.MICROSLEEP) == []
    assert ev(events, AlertType.YAWN) == []
    assert ev(events, AlertType.GAZE_AWAY) == []
    assert ev(events, AlertType.PHONE) == []
    assert mon.active == {}
    assert_edge_grammar(events, mon)


@pytest.mark.parametrize("fps", FPS_CASES)
def test_microsleep_enters_after_the_configured_seconds_at_any_rate(fps):
    mon = _mon()
    t_close = 2.0

    def fn(t, i):
        return True, sig(prob_open=SHUT if t >= t_close else OPEN)

    events, _ = drive(mon, fps, 6.0, fn)
    enters = ev(events, AlertType.MICROSLEEP, EDGE_ENTER)
    assert len(enters) == 1
    assert enters[0].severity is Severity.CRITICAL
    delay = enters[0].t_mono_s - t_close
    lo = mon.tuning.microsleep_s - 1.0 / fps - 1e-6
    hi = mon.tuning.microsleep_s + mon.fusion_cfg.eye_close_dwell_s + 1.0 / fps
    assert lo <= delay <= hi, "%.3f s at %s FPS is outside [%.3f, %.3f]" % (delay, fps, lo, hi)
    assert ev(events, AlertType.MICROSLEEP, EDGE_EXIT) == []
    assert mon.active[AlertType.MICROSLEEP] is Severity.CRITICAL


@pytest.mark.parametrize("fps", FPS_CASES)
def test_microsleep_exits_when_the_eyes_reopen(fps):
    mon = _mon()

    def fn(t, i):
        return True, sig(prob_open=SHUT if 2.0 <= t < 6.0 else OPEN)

    events, _ = drive(mon, fps, 8.0, fn)
    assert len(ev(events, AlertType.MICROSLEEP, EDGE_ENTER)) == 1
    exits = ev(events, AlertType.MICROSLEEP, EDGE_EXIT)
    assert len(exits) == 1
    assert exits[0].severity is Severity.CRITICAL
    assert 6.0 <= exits[0].t_mono_s <= 6.0 + 0.10 + mon.tuning.microsleep_exit_s + 2.0 / fps
    assert AlertType.MICROSLEEP not in mon.active
    assert_edge_grammar(events, mon)


@pytest.mark.parametrize("fps", FPS_CASES)
def test_a_single_open_frame_does_not_restart_the_microsleep_clock(fps):
    """The dominant miss mechanism in a real cabin: one frame of EAR jitter."""
    clean = _mon()

    def closed_only(t, i):
        return True, sig(prob_open=SHUT if t >= 2.0 else OPEN)

    ok, _ = drive(clean, 10.0, 6.0, closed_only)
    t_clean = ev(ok, AlertType.MICROSLEEP, EDGE_ENTER)[0].t_mono_s

    noisy = _mon()
    glitch_i = int(3.0 * 10.0)

    def with_glitch(t, i):
        if i == glitch_i:
            return True, sig(prob_open=OPEN)
        return True, sig(prob_open=SHUT if t >= 2.0 else OPEN)

    events, _ = drive(noisy, 10.0, 6.0, with_glitch)
    enters = ev(events, AlertType.MICROSLEEP, EDGE_ENTER)
    assert len(enters) == 1, "one noisy frame must not cancel the alert"
    assert enters[0].t_mono_s - t_clean <= 0.45, "the glitch should cost a few frames, not restart"


def test_microsleep_cannot_stay_latched_once_the_face_is_gone():
    """A critical must never be asserted about a driver declared missing."""
    mon = _mon()

    def fn(t, i):
        if t < 4.0:
            return True, sig(prob_open=SHUT)
        return False, None

    events, _ = drive(mon, 10.0, 10.0, fn)
    assert len(ev(events, AlertType.MICROSLEEP, EDGE_ENTER)) == 1
    exits = ev(events, AlertType.MICROSLEEP, EDGE_EXIT)
    assert len(exits) == 1 and exits[0].t_mono_s <= 4.6
    assert AlertType.MICROSLEEP not in mon.active
    assert AlertType.FACE_LOST in mon.active
    assert_edge_grammar(events, mon)


def test_cooldown_is_measured_from_the_previous_enter_not_the_exit():
    mon = _mon()

    def fn(t, i):
        block = t % 13.0
        return True, sig(prob_open=SHUT if block < 2.5 else OPEN)

    events, _ = drive(mon, 10.0, 40.0, fn)
    enters = ev(events, AlertType.MICROSLEEP, EDGE_ENTER)
    assert len(enters) == 3, "three closures 13 s apart must all alert"
    lat = mon.latches[AlertType.MICROSLEEP]
    assert lat.last_enter_s == pytest.approx(enters[-1].t_mono_s)
    last_exit = ev(events, AlertType.MICROSLEEP, EDGE_EXIT)[-1].t_mono_s
    assert lat.last_enter_s < last_exit, "an exit must not reset the cooldown window"


# ------------------------------------------------------------------ perclos ---
def test_perclos_is_time_weighted_not_frame_weighted():
    """Closure sampled fast and openness sampled slow: 50% of TIME, 71% of FRAMES."""
    mon = _mon(perclos_enter_s=1e9)  # keep the alert out of the way
    t = 0.0
    n_closed = n_open = 0
    for _ in range(12):
        for _ in range(45):  # 3 s closed at 15 FPS
            mon.update(t, present=True, signals=sig(prob_open=SHUT))
            t += 1 / 15.0
            n_closed += 1
        for _ in range(18):  # 3 s open at 6 FPS
            mon.update(t, present=True, signals=sig(prob_open=OPEN))
            t += 1 / 6.0
            n_open += 1
    assert n_closed / float(n_closed + n_open) == pytest.approx(0.714, abs=0.01)
    assert mon.view.perclos_valid is True
    assert mon.perclos == pytest.approx(0.5, abs=0.03)


def test_perclos_is_refused_below_eight_hz():
    mon = _mon()

    def fn(t, i):
        return True, sig(prob_open=SHUT if (i % 6) < 3 else OPEN)  # 50% of the time

    events, _ = drive(mon, 6.0, 60.0, fn)
    assert mon.view.perclos_rate_hz == pytest.approx(6.0, abs=0.2)
    assert mon.view.perclos_valid is False
    assert mon.view.perclos_starved is True
    assert ev(events, AlertType.PERCLOS_HIGH) == [], "6 FPS may not produce a PERCLOS alert"
    unreliable = ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)
    assert len(unreliable) == 1, "the operator must be told PERCLOS is unmeasurable"
    assert unreliable[0].extra["perclos"] == 1.0
    assert unreliable[0].extra["eyes"] == 0.0
    assert mon.unreliable["perclos"] is True
    assert_edge_grammar(events, mon)


@pytest.mark.parametrize("fps", (10.0, 15.0))
def test_perclos_high_fires_once_at_a_sufficient_rate(fps):
    half = int(fps * 0.5)

    def fn(t, i):
        return True, sig(prob_open=SHUT if (i % (2 * half)) < half else OPEN)

    mon = _mon()
    events, _ = drive(mon, fps, 60.0, fn)
    assert mon.view.perclos_valid is True
    assert mon.perclos == pytest.approx(0.5, abs=0.05)
    enters = ev(events, AlertType.PERCLOS_HIGH, EDGE_ENTER)
    assert len(enters) == 1
    assert enters[0].severity is Severity.CRITICAL
    assert enters[0].t_mono_s >= mon.fusion_cfg.perclos_min_span_s
    assert ev(events, AlertType.PERCLOS_HIGH, EDGE_EXIT) == []
    assert ev(events, AlertType.MICROSLEEP) == [], "0.5 s closures are not microsleeps"
    assert ev(events, AlertType.DROWSINESS_UNRELIABLE) == []
    assert_edge_grammar(events, mon)


def test_perclos_warn_escalates_to_critical_on_its_own_edge():
    """P11: a WARN that becomes a CRITICAL must be visible in the log."""
    mon = _mon()

    def fn(t, i):
        closed = (i % 4) == 0 if t < 45.0 else (i % 5) >= 2  # 25% then 60%
        return True, sig(prob_open=SHUT if closed else OPEN)

    events, _ = drive(mon, 10.0, 115.0, fn)
    enters = ev(events, AlertType.PERCLOS_HIGH, EDGE_ENTER)
    escalations = ev(events, AlertType.PERCLOS_HIGH, EDGE_ESCALATE)
    assert len(enters) == 1 and enters[0].severity is Severity.WARN
    assert len(escalations) == 1 and escalations[0].severity is Severity.CRITICAL
    assert escalations[0].t_mono_s > enters[0].t_mono_s
    assert mon.active[AlertType.PERCLOS_HIGH] is Severity.CRITICAL
    assert_edge_grammar(events, mon)


def test_perclos_ignores_frames_where_no_face_was_present():
    mon = _mon()
    drive(mon, 10.0, 20.0, lambda t, i: (True, sig(prob_open=OPEN)))
    before = mon.view.perclos_coverage
    drive(mon, 10.0, 20.0, lambda t, i: (False, None), t0=20.0)
    assert mon.view.perclos_coverage < before
    assert mon.view.perclos_valid is False, "half the window has no observation"


# ---------------------------------------------------------------- occlusion ---
def test_sunglasses_turn_eye_alerts_into_drowsiness_unreliable():
    mon = _mon()
    events, _ = drive(mon, 10.0, 8.0, lambda t, i: (True, sig(prob_open=SHUT, sunglasses=0.92)))
    assert ev(events, AlertType.MICROSLEEP) == []
    assert ev(events, AlertType.PERCLOS_HIGH) == []
    enters = ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)
    assert len(enters) == 1
    assert enters[0].extra["eyes"] == 1.0
    assert enters[0].extra["sunglasses"] == pytest.approx(0.92, abs=1e-3)
    assert mon.unreliable["eyes"] is True
    assert_edge_grammar(events, mon)


def test_removing_the_sunglasses_restores_the_eye_channel():
    mon = _mon()

    def fn(t, i):
        return True, sig(prob_open=OPEN, sunglasses=0.92 if t < 6.0 else 0.02)

    events, _ = drive(mon, 10.0, 14.0, fn)
    assert len(ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)) == 1
    exits = ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_EXIT)
    assert len(exits) == 1 and 6.0 <= exits[0].t_mono_s <= 8.5
    assert mon.unreliable["eyes"] is False


def test_a_mask_suppresses_yawn_without_faking_a_closed_mouth():
    mon = _mon()
    events, _ = drive(mon, 10.0, 8.0, lambda t, i: (True, sig(mouth_prob=0.97, mask=0.88)))
    assert ev(events, AlertType.YAWN) == []
    assert mon.unreliable["mouth"] is True
    assert mon.view.mouth_occluded is True


def test_yawn_fires_when_the_mouth_is_visible():
    mon = _mon()

    def fn(t, i):
        return True, sig(mouth_prob=0.97 if 2.0 <= t < 3.5 else 0.02)

    events, _ = drive(mon, 10.0, 6.0, fn)
    enters = ev(events, AlertType.YAWN, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].severity is Severity.INFO
    assert 2.3 <= enters[0].t_mono_s <= 2.9
    assert len(ev(events, AlertType.YAWN, EDGE_EXIT)) == 1


def test_three_yawns_in_the_window_raise_a_fatigue_cluster():
    mon = _mon()

    def fn(t, i):
        yawning = (t % 20.0) < 1.5
        return True, sig(mouth_prob=0.97 if yawning else 0.02)

    events, _ = drive(mon, 10.0, 50.0, fn)
    assert len(ev(events, AlertType.YAWN, EDGE_ENTER)) == 3
    clusters = ev(events, AlertType.FATIGUE_CLUSTER, EDGE_ENTER)
    assert len(clusters) == 1 and clusters[0].severity is Severity.WARN
    assert clusters[0].extra["yawns"] == 3.0
    assert_edge_grammar(events, mon)


# --------------------------------------------------------------------- gaze ---
def test_an_uncalibrated_rig_can_never_raise_a_confident_gaze_alert():
    mon = _mon(cfg=_cfg(0.0, 0.0, 0.0), calibrated=False)
    events, _ = drive(mon, 10.0, 10.0, lambda t, i: (True, sig(yaw=70.0)))
    assert ev(events, AlertType.GAZE_AWAY) == []
    assert ev(events, AlertType.DISTRACTION) == []
    enters = ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].extra["pose"] == 1.0
    assert mon.view.pose.reason == "uncalibrated"


def test_a_low_confidence_pose_can_never_raise_a_gaze_alert():
    mon = _mon()
    events, _ = drive(mon, 10.0, 10.0, lambda t, i: (True, sig(yaw=70.0, pose_conf=0.05)))
    assert ev(events, AlertType.GAZE_AWAY) == []
    enters = ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].extra["pose"] == 1.0
    assert mon.view.pose.reason == "low_confidence"


def test_a_missing_pose_engine_can_never_raise_a_gaze_alert():
    mon = _mon()
    events, _ = drive(mon, 10.0, 10.0, lambda t, i: (True, sig(pose=False)))
    assert ev(events, AlertType.GAZE_AWAY) == []
    assert len(ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)) == 1


@pytest.mark.parametrize("fps", FPS_CASES)
def test_gaze_away_enters_at_warn_and_escalates_to_critical(fps):
    mon = _mon()
    t_away = 2.0

    def fn(t, i):
        return True, sig(yaw=70.0 if t >= t_away else 0.0)

    events, _ = drive(mon, fps, 10.0, fn)
    enters = ev(events, AlertType.GAZE_AWAY, EDGE_ENTER)
    escalations = ev(events, AlertType.GAZE_AWAY, EDGE_ESCALATE)
    assert len(enters) == 1 and enters[0].severity is Severity.WARN
    assert len(escalations) == 1 and escalations[0].severity is Severity.CRITICAL
    d_enter = enters[0].t_mono_s - t_away
    d_crit = escalations[0].t_mono_s - t_away
    # a threshold may be reached up to one frame interval early: the interval
    # that straddles the moment the driver looked away is credited in full.
    assert (
        mon.tuning.gaze_warn_s - 1.0 / fps - SLOP
        <= d_enter
        <= mon.tuning.gaze_warn_s + 0.5 + 1.0 / fps
    )
    assert (
        mon.tuning.gaze_crit_s - 1.0 / fps - SLOP
        <= d_crit
        <= mon.tuning.gaze_crit_s + 0.5 + 1.0 / fps
    )
    assert mon.active[AlertType.GAZE_AWAY] is Severity.CRITICAL
    assert_edge_grammar(events, mon)


def test_gaze_away_exits_and_reports_the_severity_it_held():
    mon = _mon()

    def fn(t, i):
        return True, sig(yaw=70.0 if 2.0 <= t < 9.0 else 0.0)

    events, _ = drive(mon, 10.0, 13.0, fn)
    exits = ev(events, AlertType.GAZE_AWAY, EDGE_EXIT)
    assert len(exits) == 1 and exits[0].severity is Severity.CRITICAL
    assert AlertType.GAZE_AWAY not in mon.active
    assert_edge_grammar(events, mon)


def test_gaze_away_is_measured_against_the_forward_zero():
    mon = _mon(cfg=_cfg(yaw=-63.0, pitch=-6.0))
    events, _ = drive(mon, 10.0, 6.0, lambda t, i: (True, sig(yaw=-62.0, pitch=-7.0)))
    assert ev(events, AlertType.GAZE_AWAY) == [], "the driver is looking straight ahead"
    assert AlertType.GAZE_AWAY not in mon.active
    events, _ = drive(mon, 10.0, 6.0, lambda t, i: (True, sig(yaw=20.0, pitch=-7.0)), t0=6.0)
    assert len(ev(events, AlertType.GAZE_AWAY, EDGE_ENTER)) == 1


def test_gaze_away_cannot_latch_through_a_lost_face():
    """P06: the recorded failure was one enter and zero exits over 347 s."""
    mon = _mon()

    def fn(t, i):
        if t < 5.0:
            return True, sig(yaw=70.0)
        return False, None

    events, _ = drive(mon, 10.0, 12.0, fn)
    assert len(ev(events, AlertType.GAZE_AWAY, EDGE_ENTER)) == 1
    assert len(ev(events, AlertType.GAZE_AWAY, EDGE_EXIT)) == 1
    assert AlertType.GAZE_AWAY not in mon.active
    assert_edge_grammar(events, mon)


def test_eye_gaze_is_added_to_head_pose():
    """Head 15 deg plus eyes 20 deg is off-road even though neither alone is."""
    mon = _mon()
    events, _ = drive(
        mon, 10.0, 6.0, lambda t, i: (True, sig(yaw=15.0, gaze_yaw=20.0, gaze_conf=0.9))
    )
    assert len(ev(events, AlertType.GAZE_AWAY, EDGE_ENTER)) == 1
    assert mon.view.gaze.reason == "eye+pose"
    head_only = _mon()
    events, _ = drive(head_only, 10.0, 6.0, lambda t, i: (True, sig(yaw=15.0)))
    assert ev(events, AlertType.GAZE_AWAY) == []


# -------------------------------------------------------------- face lost ---
@pytest.mark.parametrize("fps", FPS_CASES)
def test_face_lost_is_measured_from_the_last_match(fps):
    mon = _mon()
    t_gone = 3.0

    def fn(t, i):
        return (True, sig()) if t < t_gone else (False, None)

    events, _ = drive(mon, fps, 12.0, fn)
    enters = ev(events, AlertType.FACE_LOST, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].severity is Severity.WARN
    last_match = t_gone - 1.0 / fps
    assert (
        mon.tuning.face_lost_s - SLOP
        <= enters[0].t_mono_s - last_match
        <= mon.tuning.face_lost_s + 1.0 / fps
    )
    escalations = ev(events, AlertType.FACE_LOST, EDGE_ESCALATE)
    assert len(escalations) == 1 and escalations[0].severity is Severity.CRITICAL
    assert (
        mon.tuning.face_lost_crit_s - 1.0 / fps - SLOP
        <= escalations[0].t_mono_s - enters[0].t_mono_s
        <= mon.tuning.face_lost_crit_s + 1.0 / fps
    )
    assert_edge_grammar(events, mon)


def test_face_lost_exits_only_after_a_sustained_reacquisition():
    mon = _mon()

    def fn(t, i):
        if t < 2.0:
            return True, sig()
        if t < 6.0:
            return False, None
        return True, sig()

    events, _ = drive(mon, 10.0, 10.0, fn)
    exits = ev(events, AlertType.FACE_LOST, EDGE_EXIT)
    assert len(exits) == 1
    assert (
        mon.tuning.face_lost_exit_s - 0.1
        <= exits[0].t_mono_s - 6.0
        <= mon.tuning.face_lost_exit_s + 0.2
    )
    assert AlertType.FACE_LOST not in mon.active


def test_face_lost_fires_when_a_face_was_never_seen():
    """P13: a lens blocked at boot must not look identical to a healthy system."""
    mon = _mon()
    events, _ = drive(mon, 10.0, 20.0, lambda t, i: (False, None))
    enters = ev(events, AlertType.FACE_LOST, EDGE_ENTER)
    assert len(enters) == 1
    assert enters[0].t_mono_s == pytest.approx(mon.tuning.no_driver_grace_s, abs=0.15)


def test_face_lost_does_not_fire_during_the_first_frames():
    mon = _mon()
    events = mon.update(0.0, present=False)
    assert events == []
    events, _ = drive(mon, 10.0, 0.5, lambda t, i: (False, None), t0=0.1)
    assert ev(events, AlertType.FACE_LOST) == []


def test_face_lost_follows_t_last_match_even_when_the_caller_says_present():
    """A coasting track id must not be able to hide a lost face (design 2.2/A6)."""
    mon = _mon()
    events = []
    for i in range(80):
        t = i * 0.1
        # the caller keeps a coasting id alive and reports present, but the
        # selector saw no matched detection in the seat ROI after t = 2.0
        events += mon.update(
            t, present=True, track_id=1, signals=sig(), t_last_match=min(t, 2.0)
        )
    enters = ev(events, AlertType.FACE_LOST, EDGE_ENTER)
    assert len(enters) == 1
    assert enters[0].t_mono_s == pytest.approx(3.0, abs=0.11)


def test_an_explicit_t_last_match_keeps_face_lost_disarmed():
    mon = _mon()
    events = []
    for i in range(60):
        t = i * 0.1
        events += mon.update(t, present=False, track_id=1, t_last_match=t)
    assert ev(events, AlertType.FACE_LOST) == []


def test_face_lost_severity_follows_assume_moving():
    parked = _mon(assume_moving=False)
    events, _ = drive(parked, 10.0, 20.0, lambda t, i: (False, None))
    enters = ev(events, AlertType.FACE_LOST, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].severity is Severity.WARN
    assert ev(events, AlertType.FACE_LOST, EDGE_ESCALATE) == []
    assert parked.highest_severity() is Severity.WARN

    immediate = _mon(face_lost_crit_s=0.0)
    events, _ = drive(immediate, 10.0, 20.0, lambda t, i: (False, None))
    enters = ev(events, AlertType.FACE_LOST, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].severity is Severity.CRITICAL


# ---------------------------------------------------------- driver change ---
def test_a_driver_change_resets_every_window_and_emits_one_event():
    mon = _mon()

    def fn(t, i):
        if t < 6.0:
            return True, sig(prob_open=SHUT, track_id=1)
        return True, sig(prob_open=OPEN, track_id=2)

    events, _ = drive(mon, 10.0, 12.0, fn)
    changed = ev(events, AlertType.DRIVER_CHANGED)
    assert [e.edge for e in changed] == [EDGE_ENTER, EDGE_EXIT]
    assert changed[0].t_mono_s == changed[1].t_mono_s
    assert changed[0].extra == {"from_track": 1.0, "to_track": 2.0}

    at_change = [e for e in events if e.t_mono_s == pytest.approx(changed[0].t_mono_s)]
    kinds = [(e.type, e.edge) for e in at_change]
    assert (AlertType.MICROSLEEP, EDGE_EXIT) in kinds, "the old driver's alert must resolve"
    assert kinds.index((AlertType.MICROSLEEP, EDGE_EXIT)) < kinds.index(
        (AlertType.DRIVER_CHANGED, EDGE_ENTER)
    )
    assert AlertType.MICROSLEEP not in mon.active
    assert mon.track_id == 2
    assert_edge_grammar(events, mon)


def test_a_driver_change_discards_the_perclos_window():
    mon = _mon()
    drive(mon, 10.0, 40.0, lambda t, i: (True, sig(prob_open=SHUT, track_id=1)))
    assert mon.view.perclos_valid is True and mon.perclos > 0.9
    mon.update(40.0, present=True, signals=sig(prob_open=OPEN, track_id=2))
    assert mon.view.perclos_valid is False
    assert mon.perclos == 0.0, "one person's PERCLOS must never be charged to another"


def test_identity_change_under_a_stable_track_id_is_still_a_driver_change():
    mon = _mon()

    def fn(t, i):
        who = "alice" if t < 4.0 else "bob"
        return True, sig(track_id=7, identity=who)

    events, _ = drive(mon, 10.0, 8.0, fn)
    changed = ev(events, AlertType.DRIVER_CHANGED, EDGE_ENTER)
    assert len(changed) == 1
    assert changed[0].t_mono_s == pytest.approx(4.0, abs=0.15)
    assert mon.identity == "bob"


def test_the_first_driver_of_a_session_is_not_a_change():
    mon = _mon()
    events, _ = drive(mon, 10.0, 3.0, lambda t, i: (True, sig(track_id=4, identity="alice")))
    assert ev(events, AlertType.DRIVER_CHANGED) == []
    assert mon.track_id == 4 and mon.identity == "alice"


# --------------------------------------------------------- phone / distract --
def test_phone_needs_a_sustained_detection():
    mon = _mon()

    def brief(t, i):
        return True, sig(phone=0.9 if 2.0 <= t < 2.4 else 0.0)

    events, _ = drive(mon, 10.0, 6.0, brief)
    assert ev(events, AlertType.PHONE) == []
    assert ev(events, AlertType.DISTRACTION) == []

    mon = _mon()

    def held(t, i):
        return True, sig(phone=0.9 if t >= 2.0 else 0.0)

    events, _ = drive(mon, 10.0, 8.0, held)
    enters = ev(events, AlertType.PHONE, EDGE_ENTER)
    assert len(enters) == 1
    delay = enters[0].t_mono_s - 2.0
    assert mon.tuning.phone_enter_s - 0.1 <= delay <= mon.tuning.phone_enter_s + 0.2


def test_distraction_is_raised_by_the_phone():
    mon = _mon()
    events, _ = drive(mon, 10.0, 8.0, lambda t, i: (True, sig(phone=0.9 if t >= 2.0 else 0.0)))
    enters = ev(events, AlertType.DISTRACTION, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].severity is Severity.WARN
    delay = enters[0].t_mono_s - 2.0
    assert mon.tuning.distraction_phone_s - 0.1 <= delay <= mon.tuning.distraction_phone_s + 0.2
    assert enters[0].extra["phone_s"] >= mon.tuning.distraction_phone_s - 0.11


def test_distraction_is_raised_by_sustained_off_road_gaze():
    mon = _mon()
    events, _ = drive(mon, 10.0, 14.0, lambda t, i: (True, sig(yaw=70.0 if t >= 2.0 else 0.0)))
    enters = ev(events, AlertType.DISTRACTION, EDGE_ENTER)
    assert len(enters) == 1
    delay = enters[0].t_mono_s - 2.0
    assert mon.tuning.distraction_gaze_s - 0.1 <= delay <= mon.tuning.distraction_gaze_s + 0.6
    assert enters[0].extra["gaze_s"] >= mon.tuning.distraction_gaze_s - 0.11


def test_distraction_exits_once_the_driver_re_engages():
    mon = _mon()

    def fn(t, i):
        return True, sig(phone=0.9 if 2.0 <= t < 8.0 else 0.0)

    events, _ = drive(mon, 10.0, 14.0, fn)
    exits = ev(events, AlertType.DISTRACTION, EDGE_EXIT)
    assert len(exits) == 1
    assert 8.0 <= exits[0].t_mono_s <= 8.0 + mon.tuning.distraction_exit_s + 0.2
    assert AlertType.DISTRACTION not in mon.active
    assert_edge_grammar(events, mon)


# --------------------------------------------------------- model degraded ---
def test_model_degraded_tracks_the_registry():
    mon = _mon()
    events, _ = drive(mon, 10.0, 6.0, lambda t, i: (True, sig()), degraded=["headpose", "gaze"])
    enters = ev(events, AlertType.MODEL_DEGRADED, EDGE_ENTER)
    assert len(enters) == 1 and enters[0].extra["capabilities"] == 2.0
    assert (
        mon.tuning.degraded_enter_s - 0.1
        <= enters[0].t_mono_s
        <= mon.tuning.degraded_enter_s + 0.2
    )
    events, _ = drive(mon, 10.0, 10.0, lambda t, i: (True, sig()), t0=6.0, degraded=[])
    assert len(ev(events, AlertType.MODEL_DEGRADED, EDGE_EXIT)) == 1
    assert AlertType.MODEL_DEGRADED not in mon.active


def test_model_degraded_is_silent_when_the_caller_says_nothing():
    mon = _mon()
    events, _ = drive(mon, 10.0, 10.0, lambda t, i: (True, sig()))
    assert ev(events, AlertType.MODEL_DEGRADED) == []


# ------------------------------------------------------------ integration ---
def test_a_long_mixed_drive_keeps_the_edge_grammar_and_severity_order():
    """One 150 s drive exercising every alert; the log must stay well-formed."""
    mon = _mon()

    def fn(t, i):
        phase = t % 50.0
        present = not (30.0 <= phase < 34.0)
        if not present:
            return False, None
        closed = 10.0 <= phase < 13.0 or (i % 7) == 0
        return True, sig(
            prob_open=SHUT if closed else OPEN,
            mouth_prob=0.97 if 20.0 <= phase < 21.5 else 0.02,
            yaw=70.0 if 40.0 <= phase < 46.0 else 0.0,
            sunglasses=0.9 if 24.0 <= phase < 27.0 else 0.0,
            phone=0.9 if 15.0 <= phase < 18.0 else 0.0,
            track_id=1 if t < 100.0 else 2,
        )

    events, _ = drive(mon, 12.0, 150.0, fn)
    assert_edge_grammar(events, mon)
    seen = set(e.type for e in events)
    for expected in (
        AlertType.MICROSLEEP,
        AlertType.YAWN,
        AlertType.GAZE_AWAY,
        AlertType.PHONE,
        AlertType.DISTRACTION,
        AlertType.FACE_LOST,
        AlertType.DROWSINESS_UNRELIABLE,
        AlertType.DRIVER_CHANGED,
    ):
        assert expected in seen, "%s never fired in the mixed drive" % expected
    if mon.active:
        order = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}
        assert mon.highest_severity() == max(mon.active.values(), key=lambda s: order[s])


def test_every_event_carries_the_track_id_of_the_driver_it_describes():
    mon = _mon()
    events, _ = drive(mon, 10.0, 6.0, lambda t, i: (True, sig(prob_open=SHUT, track_id=9)))
    assert events
    assert all(e.track_id == 9 for e in events)


# ---------------------------------------------------------- geometry bridge -
# The four-float ``ear=/mar=/yaw=/pitch=`` entry point into DriverMonitor.
#
# Who uses it: a profile configured with no model engines, and these tests.
# Who does NOT: ``dms/app.py``.  Both of its ``monitor.update`` call sites pass
# ``signals=``.  An earlier version of this file claimed the opposite in its
# docstring and then asserted against the OLD broken-landmark EAR scale
# (ear_closed 0.50 / ear_open_median 0.75) by mutating ``fusion_cfg`` after
# construction.  Both halves were wrong, which is why the tests below read the
# threshold off the monitor instead of assuming one.

#: The corrected roll-invariant EAR scale these fixtures are written against -
#: ``dms.geometry.ear.EAR_CLOSED_RECOMMENDED``, and what ``configs/default.yaml``
#: carries.  A recalibration that moves it must fail HERE, loudly, rather than
#: quietly making the closed-eye fixture value read as an open eye.
EAR_CLOSED_EFFECTIVE = 0.18
#: Comfortably below the threshold: a genuinely closed eye.
EAR_SHUT = 0.10
#: Comfortably above threshold + ear_hysteresis (0.03): a genuinely open eye.
EAR_WIDE = 0.30


def test_the_geometry_fixtures_are_written_against_the_shipped_ear_scale():
    """Guard the two constants above against the next recalibration."""
    mon = _mon()
    thr = mon.fusion_cfg.ear_closed_effective
    assert thr == pytest.approx(EAR_CLOSED_EFFECTIVE), (
        "configs/default.yaml now yields an effective closed-EAR threshold of "
        "%r, not %r. The EAR scale moved: retune EAR_SHUT / EAR_WIDE in this "
        "file to sit either side of it." % (thr, EAR_CLOSED_EFFECTIVE)
    )
    assert EAR_SHUT < thr, "EAR_SHUT must read as a CLOSED eye"
    assert EAR_WIDE > thr + mon.fusion_cfg.ear_hysteresis, "EAR_WIDE must read as OPEN"


def test_the_legacy_geometry_call_path_still_drives_the_machine():
    """The geometry float path drives MICROSLEEP enter AND exit end to end.

    Six seconds of a closed eye, then four of an open one, through
    ``ear=``/``mar=``/``yaw=``/``pitch=`` only - no DriverSignals anywhere.
    """
    mon = _mon(cfg=_cfg(yaw=-63.0, pitch=-6.0))
    events = []
    for i in range(60):
        events += mon.update(
            i * 0.1, present=True, track_id=1, ear=EAR_SHUT, mar=0.30,
            yaw=-63.0, pitch=-6.0,
        )
    assert len(ev(events, AlertType.MICROSLEEP, EDGE_ENTER)) == 1
    assert mon.view.eyes.source is SignalSource.GEOMETRY
    assert mon.view.eye_closed is True
    assert mon.ear_ewma == pytest.approx(EAR_SHUT)
    assert mon.yaw_ewma == pytest.approx(-63.0, abs=0.5)
    for i in range(60, 100):
        events += mon.update(
            i * 0.1, present=True, track_id=1, ear=EAR_WIDE, mar=0.30,
            yaw=-63.0, pitch=-6.0,
        )
    assert mon.view.eye_closed is False
    assert len(ev(events, AlertType.MICROSLEEP, EDGE_EXIT)) == 1
    assert len(ev(events, AlertType.MICROSLEEP, EDGE_ENTER)) == 1


def test_an_open_eye_on_the_corrected_scale_raises_no_microsleep():
    """0.20 is an OPEN eye now.  Under the old 0.50 scale it was a closed one.

    This is the exact value the pre-fix version of this file drove for six
    seconds while asserting a MICROSLEEP; pinning the opposite result stops the
    old scale from creeping back in.
    """
    mon = _mon(cfg=_cfg(yaw=-63.0, pitch=-6.0))
    events = []
    for i in range(60):
        events += mon.update(
            i * 0.1, present=True, track_id=1, ear=0.20, mar=0.30,
            yaw=-63.0, pitch=-6.0,
        )
    assert 0.20 > mon.fusion_cfg.ear_closed_effective
    assert mon.view.eye_closed is False
    assert ev(events, AlertType.MICROSLEEP) == []


def test_retuning_the_ear_threshold_mid_stream_moves_the_decision():
    """A calibration written onto ``fusion_cfg`` must take effect immediately.

    ``DriverMonitor.recalibrate`` and ``_sync_calibration`` edit ``fusion_cfg``
    in place, and ``--calibrate-forward`` measures a new
    ``state.ear_open_median`` at runtime before writing it out, so the fusion
    layer must not cache thresholds.  It
    used to: the Schmitt latches copied ``ear_closed_effective`` at
    construction while ``_fuse_eyes`` re-read it live, so one frame was judged
    against two threshold generations and this assertion failed.
    """
    mon = _mon(cfg=_cfg(yaw=-63.0, pitch=-6.0))
    for i in range(20):
        mon.update(i * 0.1, present=True, track_id=1, ear=0.20, mar=0.30,
                   yaw=-63.0, pitch=-6.0)
    assert mon.view.eye_closed is False       # 0.20 > 0.18
    # Raise the closed threshold above 0.20; the same measurement is now closed.
    mon.fusion_cfg.ear_closed = 0.25
    mon.fusion_cfg.ear_open_median = 0.40     # effective = min(0.25, 0.28) = 0.25
    assert mon.fusion_cfg.ear_closed_effective == pytest.approx(0.25)
    for i in range(20, 40):
        mon.update(i * 0.1, present=True, track_id=1, ear=0.20, mar=0.30,
                   yaw=-63.0, pitch=-6.0)
    assert mon.view.eye_closed is True
    # ...and back.  The EXIT threshold is enter + ear_hysteresis (0.03), so to
    # re-open on the same 0.20 measurement the new effective threshold has to
    # be below 0.17; 0.10 is.  (Restoring 0.18 would leave 0.20 inside the
    # 0.18-0.21 band, where a Schmitt latch correctly holds its state - that is
    # hysteresis working, not a cached threshold.)
    mon.fusion_cfg.ear_closed = 0.10
    mon.fusion_cfg.ear_open_median = 0.20     # effective = min(0.10, 0.14) = 0.10
    assert mon.fusion_cfg.ear_closed_effective == pytest.approx(0.10)
    for i in range(40, 60):
        mon.update(i * 0.1, present=True, track_id=1, ear=0.20, mar=0.30,
                   yaw=-63.0, pitch=-6.0)
    assert mon.view.eye_closed is False


def test_the_legacy_path_reports_unavailable_when_geometry_produced_nothing():
    mon = _mon()
    events = []
    for i in range(40):
        events += mon.update(i * 0.1, present=True, track_id=1, ear=None, mar=None, yaw=None, pitch=None)
    assert mon.view.eyes.source is SignalSource.UNAVAILABLE
    assert ev(events, AlertType.MICROSLEEP) == []
    assert len(ev(events, AlertType.DROWSINESS_UNRELIABLE, EDGE_ENTER)) == 1


# ======================================================================= #
#  Synthetic golden: end-to-end coverage of every alert the state machine  #
#  can raise.                                                              #
# ======================================================================= #
#
# WHY THIS EXISTS
# ---------------
# ``tests/replay/golden/*.json`` is the project's end-to-end behavioural guard
# and its Phase 1 exit criterion, but the nine clips behind it are DASHCAM
# montages, not cabin footage (``tests/replay/CLIPS.md``).  Nobody in them ever
# closes their eyes at the camera, so those goldens pin only four alert types -
# driver_changed, drowsiness_unreliable, face_lost and yawn - and MICROSLEEP and
# PERCLOS_HIGH, the two alerts this product exists to raise, had ZERO end-to-end
# coverage.  A change that made MICROSLEEP permanently unreachable passed all
# nine clips green; that is exactly what the wrong EAR scale did.
#
# This golden closes that gap without pretending to have footage we do not have.
# It replays a scripted 200 s DriverSignals stream - the perception layer's
# OUTPUT, not its input - through the real DriverMonitor and pins the exact
# per-type edge counts and first-enter timings.  It is not ground truth either:
# it pins the behaviour of the state machine against the behaviour it had when
# it was recorded.  What it does prove is that all eleven alert types the
# monitor can raise are still REACHABLE, in the right order, at the right time.
#
# TOLERANCES
# ----------
# Exact counts, and +/- 2 frames on the timings.  Unlike the clip goldens there
# is nothing to be loose about: no decode, no GPU, no wall clock - just float
# arithmetic over a fixed sequence, so the stream is reproducible bit for bit.
# ``test_the_synthetic_golden_drive_is_deterministic`` proves that rather than
# asserting it.
#
# RE-RECORDING (read the diff, do not just accept it):
#     DMS_RECORD_SYNTHETIC_GOLDEN=1 PYTHONPATH=. ./.venv/bin/python -m pytest -q \
#         tests/test_state.py -k synthetic_golden

SYNTHETIC_DIR = os.path.join(ROOT, "tests", "replay", "golden", "synthetic")
SYNTHETIC_GOLDEN = os.path.join(SYNTHETIC_DIR, "scripted_drive.json")
ALERT_COVERAGE = os.path.join(SYNTHETIC_DIR, "alert_coverage.json")
CLIP_GOLDEN_GLOB = os.path.join(ROOT, "tests", "replay", "golden", "*.json")

SYNTHETIC_FPS = 12.0
SYNTHETIC_DURATION_S = 200.0
#: Timing slack, in frames.  A dwell that completes between two ticks lands on
#: whichever tick the arithmetic puts it on.
SYNTHETIC_SLACK_S = 2.0 / SYNTHETIC_FPS

#: The eleven alert types :class:`DriverMonitor` can raise.  The other five
#: (camera_fail, engine_fail, pipeline_starved, thermal_throttle, disk_full) are
#: raised by dms/app.py, dms/io/events.py and dms/io/clips.py from pipeline and
#: disk health, never by the state machine, and cannot appear in any replay of a
#: run that succeeded.
MONITOR_ALERT_TYPES = (
    AlertType.FACE_LOST,
    AlertType.MICROSLEEP,
    AlertType.PERCLOS_HIGH,
    AlertType.YAWN,
    AlertType.FATIGUE_CLUSTER,
    AlertType.GAZE_AWAY,
    AlertType.PHONE,
    AlertType.DISTRACTION,
    AlertType.DROWSINESS_UNRELIABLE,
    AlertType.DRIVER_CHANGED,
    AlertType.MODEL_DEGRADED,
)


def synthetic_script(t, i):
    """``(present, DriverSignals | None, degraded)`` for media time ``t``.

    One pass, no modulo phases, so every window below can be read against
    ``configs/default.yaml`` directly.  Each segment is sized from the dwell it
    has to complete plus the cooldown of the alert before it:

    ``0-40``     eyes shut on one frame in four.  25% closed TIME clears
                 ``perclos_warn`` 0.20; PERCLOS needs a 30 s span at >= 8 Hz
                 before it is even valid, so this is the shortest honest way in.
                 A single 1/12 s frame is under ``eye_close_dwell_s`` 0.15, so
                 this is a blink pattern and must NOT raise MICROSLEEP.
    ``40-44``    eyes held shut -> MICROSLEEP (``microsleep_s`` 1.5).
    ``44-60``    recovery; PERCLOS drains, ``microsleep_cooldown_s`` 10 elapses.
    ``60/78/96`` three 1 s yawns, >``yawn_cooldown_s`` 15 apart -> YAWN x3 and
                 then FATIGUE_CLUSTER (``fatigue_yawns`` 3 in 180 s).
    ``105-112``  head yawed 70 deg -> GAZE_AWAY at 2 s, escalating at 4 s, and
                 DISTRACTION at ``distraction_gaze_s`` 6.
    ``125-128``  phone confidence 0.9 -> PHONE at ``phone_enter_s`` 0.8.
    ``140-144``  sunglasses 0.9 -> the eye channel is unmeasurable ->
                 DROWSINESS_UNRELIABLE, never a confident drowsiness alert.
    ``155-160``  no face -> FACE_LOST at 1 s, escalating at
                 ``face_lost_crit_s`` 3.
    ``175-185``  two capabilities degraded -> MODEL_DEGRADED at 2 s.
    ``185-200``  a new track id -> DRIVER_CHANGED, and every window is wiped.
                 Last, because that reset would otherwise hide what follows it.
    """
    degraded = ["headpose", "gaze"] if 175.0 <= t < 185.0 else []
    if 155.0 <= t < 160.0:
        return False, None, degraded
    blink = t < 40.0 and (i % 4) == 0
    shut = 40.0 <= t < 44.0
    yawning = (60.0 <= t < 61.0) or (78.0 <= t < 79.0) or (96.0 <= t < 97.0)
    return (
        True,
        sig(
            prob_open=SHUT if (blink or shut) else OPEN,
            mouth_prob=0.97 if yawning else 0.02,
            yaw=70.0 if 105.0 <= t < 112.0 else 0.0,
            sunglasses=0.9 if 140.0 <= t < 144.0 else 0.0,
            phone=0.9 if 125.0 <= t < 128.0 else 0.0,
            track_id=2 if t >= 185.0 else 1,
        ),
        degraded,
    )


def run_synthetic_drive():
    """Replay :func:`synthetic_script` through a real monitor. -> (mon, events)."""
    mon = _mon()
    events = []
    dt = 1.0 / SYNTHETIC_FPS
    for i in range(int(round(SYNTHETIC_DURATION_S * SYNTHETIC_FPS))):
        t = i * dt
        present, signals, degraded = synthetic_script(t, i)
        events += mon.update(
            t, present=present, track_id=None, signals=signals, degraded=degraded
        )
    return mon, events


def observe_synthetic(events):
    """Edge counts and first-enter times per alert type, as a golden records them."""
    obs = {}
    for e in events:
        row = obs.setdefault(e.type.value, {"edges": {}, "first_enter_s": None})
        row["edges"][e.edge] = row["edges"].get(e.edge, 0) + 1
        if e.edge == EDGE_ENTER and row["first_enter_s"] is None:
            row["first_enter_s"] = round(e.t_mono_s, 4)
    return obs


def test_the_synthetic_golden_drive_is_deterministic():
    """Two runs of the same script must produce the identical edge stream.

    This is what licenses the exact counts in the golden below.  If it ever
    fails, loosen the golden - do not paper over a non-deterministic state
    machine with a band.
    """
    _, a = run_synthetic_drive()
    _, b = run_synthetic_drive()
    key = lambda e: (round(e.t_mono_s, 9), e.type.value, e.edge, e.severity.value)
    assert [key(e) for e in a] == [key(e) for e in b]


def test_the_synthetic_golden_pins_every_alert_the_monitor_can_raise():
    """The regression gate the nine dashcam clips cannot be."""
    mon, events = run_synthetic_drive()
    assert_edge_grammar(events, mon)
    obs = observe_synthetic(events)

    if os.environ.get("DMS_RECORD_SYNTHETIC_GOLDEN") == "1":
        _record_synthetic_golden(obs)

    with open(SYNTHETIC_GOLDEN) as f:
        golden = json.load(f)
    assert golden["schema"] == "dms.synthetic_golden/1"
    want = golden["alerts"]

    assert sorted(want) == sorted(obs), (
        "alert TYPES changed: golden has %s, this run produced %s"
        % (sorted(want), sorted(obs))
    )
    for atype in sorted(want):
        assert obs[atype]["edges"] == want[atype]["edges"], (
            "%s edge counts changed: golden %s, observed %s"
            % (atype, want[atype]["edges"], obs[atype]["edges"])
        )
        assert obs[atype]["first_enter_s"] == pytest.approx(
            want[atype]["first_enter_s"], abs=SYNTHETIC_SLACK_S
        ), "%s first enter moved" % atype

    # The point of the whole fixture: none of these may quietly become
    # unreachable.  Named explicitly so the failure message says which.
    for atype in MONITOR_ALERT_TYPES:
        assert atype.value in obs, (
            "%s is no longer reachable end to end. Every alert the state machine "
            "can raise must appear in this drive." % atype.value
        )


def _record_synthetic_golden(obs):
    if not os.path.isdir(SYNTHETIC_DIR):
        os.makedirs(SYNTHETIC_DIR)
    doc = {
        "schema": "dms.synthetic_golden/1",
        "source": "tests/test_state.py::synthetic_script",
        "config": "configs/default.yaml",
        "fps": SYNTHETIC_FPS,
        "duration_s": SYNTHETIC_DURATION_S,
        "frames": int(round(SYNTHETIC_DURATION_S * SYNTHETIC_FPS)),
        "timing_slack_s": round(SYNTHETIC_SLACK_S, 6),
        "note": (
            "Scripted DriverSignals replayed through the real DriverMonitor. No "
            "video, no models, no GPU: this pins the STATE MACHINE end to end, "
            "and exists because the nine dashcam replay clips cannot reach "
            "microsleep or perclos_high. Counts are exact because the drive is "
            "pure arithmetic over a fixed sequence; timings carry +/- 2 frames. "
            "Not ground truth - it pins behaviour, not accuracy."
        ),
        "alerts": obs,
    }
    with open(SYNTHETIC_GOLDEN, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


# ----------------------------------------------------- the coverage ledger ---
def test_the_alert_coverage_ledger_tells_the_truth():
    """``alert_coverage.json`` must match what the goldens actually pin.

    Every one of the 16 AlertType members is listed there with where it is
    guarded and, when it is guarded nowhere, why.  This test recomputes all of
    that from the fixtures, so the ledger cannot quietly go stale: if a clip
    starts producing microsleep, or the synthetic drive stops producing it, this
    fails and somebody has to write down what changed.
    """
    with open(ALERT_COVERAGE) as f:
        ledger = json.load(f)
    entries = ledger["types"]
    assert sorted(entries) == sorted(t.value for t in AlertType), (
        "the ledger and AlertType disagree: %s"
        % sorted(set(entries) ^ set(t.value for t in AlertType))
    )

    clip_pinned = set()
    clip_files = sorted(glob.glob(CLIP_GOLDEN_GLOB))
    assert clip_files, "no clip goldens found at %s" % CLIP_GOLDEN_GLOB
    for path in clip_files:
        with open(path) as f:
            clip_pinned |= set(json.load(f).get("alerts", {}))

    with open(SYNTHETIC_GOLDEN) as f:
        synth_pinned = set(json.load(f)["alerts"])

    for name, row in sorted(entries.items()):
        assert row["clip_goldens"] == (name in clip_pinned), (
            "%s: ledger says clip_goldens=%s, the nine clip goldens say %s"
            % (name, row["clip_goldens"], name in clip_pinned)
        )
        assert row["synthetic_golden"] == (name in synth_pinned), (
            "%s: ledger says synthetic_golden=%s, the synthetic golden says %s"
            % (name, row["synthetic_golden"], name in synth_pinned)
        )
        if not (row["clip_goldens"] or row["synthetic_golden"]):
            assert row["why_unpinned"].strip(), (
                "%s is pinned by no golden and the ledger gives no reason" % name
            )

    # The honest statement of the gap, enforced: every alert the STATE MACHINE
    # can raise is pinned end to end; the five raised from pipeline/disk health
    # are the only unpinned ones, and each carries a reason.
    unpinned = set(
        n for n, r in entries.items() if not (r["clip_goldens"] or r["synthetic_golden"])
    )
    assert unpinned == set(ledger["not_reachable_in_any_replay"]), (
        "unpinned types %s do not match the ledger's declared gap %s"
        % (sorted(unpinned), sorted(ledger["not_reachable_in_any_replay"]))
    )
    assert not (unpinned & set(t.value for t in MONITOR_ALERT_TYPES)), (
        "a state-machine alert has lost its end-to-end guard: %s" % sorted(unpinned)
    )


def test_the_clip_goldens_alone_do_not_cover_the_drowsiness_path():
    """Pin the GAP itself, so nobody reads nine green clips as full coverage.

    If a future cabin clip does reach microsleep, this fails - and the right
    response is to celebrate, update the ledger and delete this test.
    """
    clip_pinned = set()
    for path in sorted(glob.glob(CLIP_GOLDEN_GLOB)):
        with open(path) as f:
            clip_pinned |= set(json.load(f).get("alerts", {}))
    assert AlertType.MICROSLEEP.value not in clip_pinned
    assert AlertType.PERCLOS_HIGH.value not in clip_pinned
