"""Spec for dms/track: BYTE-lite association and stable driver selection.

The property under test throughout is STABILITY.  An id that churns, or a DRV
label that flips between two similar faces, silently stitches one person's
drowsiness windows onto another - which is the failure the alert machine cannot
detect from the inside.
"""
from __future__ import annotations

import numpy as np
import pytest

from dms.infer.scrfd import FaceDet
from dms.track.driver_select import (
    DriverSelector,
    SelectorConfig,
    label_occupants,
    pick_driver,
)
from dms.track.iou_tracker import IouTracker, Track, _iou

FRAME_WH = (1280, 720)


def _face(x1, y1, x2, y2, score=0.9):
    return FaceDet(xyxy=np.array([x1, y1, x2, y2], np.float32), score=float(score))


def _box(cx, cy, size, score=0.9):
    h = size / 2.0
    return _face(cx - h, cy - h, cx + h, cy + h, score)


# ---------------------------------------------------------------- tracker ---
def test_iou_basics():
    a = np.array([0, 0, 10, 10], np.float32)
    assert _iou(a, a) == pytest.approx(1.0)
    assert _iou(a, np.array([20, 20, 30, 30], np.float32)) == 0.0
    assert _iou(a, np.array([0, 0, 0, 0], np.float32)) == 0.0


def test_tracker_keeps_an_id_through_small_motion():
    tr = IouTracker()
    t0 = tr.update([_face(10, 10, 50, 50, 0.9)], t_s=0.0)
    assert t0[0].track_id == 1 and t0[0].hits == 1 and t0[0].age == 1
    t1 = tr.update([_face(12, 11, 52, 49, 0.8)], t_s=0.1)
    assert t1[0].track_id == 1
    assert t1[0].lost == 0 and t1[0].hits == 2 and t1[0].age == 2
    assert t1[0].t_last_match_s == pytest.approx(0.1)
    assert t1[0].matched is True


def test_tracker_coasts_then_drops():
    tr = IouTracker(max_lost=3)
    tr.update([_face(10, 10, 50, 50)], t_s=0.0)
    for i in range(3):
        live = tr.update([], t_s=0.1 * (i + 1))
        assert len(live) == 1
        assert live[0].matched is False, "a coasting track is NOT present"
        assert live[0].face is None
        assert live[0].t_last_match_s == pytest.approx(0.0)
    assert tr.update([], t_s=1.0) == []


def test_a_low_score_detection_keeps_an_id_alive_but_never_spawns_one():
    tr = IouTracker(high_score=0.45)
    tr.update([_face(10, 10, 50, 50, 0.9)], t_s=0.0)
    live = tr.update([_face(11, 11, 51, 51, 0.20)], t_s=0.1)
    assert len(live) == 1 and live[0].track_id == 1 and live[0].matched
    # the same weak detection somewhere new must not become a track
    live = tr.update([_face(600, 400, 640, 440, 0.20)], t_s=0.2)
    assert [t.track_id for t in live] == [1]
    assert live[0].matched is False


def test_greedy_assignment_is_global_not_first_come():
    """Two tracks, two detections: each must take its own best partner."""
    tr = IouTracker()
    tr.update([_box(200, 200, 100), _box(210, 205, 96)], t_s=0.0)
    ids = sorted(t.track_id for t in tr.tracks)
    assert ids == [1, 2]
    live = tr.update([_box(212, 206, 96), _box(202, 201, 100)], t_s=0.1)
    assert sorted(t.track_id for t in live) == [1, 2]
    assert all(t.matched for t in live)


def test_reset_does_not_reuse_ids():
    tr = IouTracker()
    tr.update([_face(10, 10, 50, 50)])
    tr.reset()
    live = tr.update([_face(10, 10, 50, 50)])
    assert live[0].track_id == 2


def test_tracker_tolerates_a_frame_with_no_timestamp():
    tr = IouTracker()
    live = tr.update([_face(10, 10, 50, 50)])
    assert live[0].t_last_match_s is None


# -------------------------------------------------------- legacy pick API ---
def test_pick_driver_no_fallback_returns_none():
    pax = _face(10, 10, 80, 80)
    assert pick_driver([pax], [0.5, 0.0, 1.0, 1.0], (640, 480), fallback=False) is None


def test_pick_driver_largest_in_roi_not_highest_score():
    small_hi = _face(10, 10, 40, 50, 0.99)
    big_lo = _face(200, 80, 400, 360, 0.70)
    assert pick_driver([small_hi, big_lo], [0.4, 0.0, 1.0, 1.0], (640, 480)) is big_lo


def test_pick_driver_on_an_empty_list():
    assert pick_driver([], [0.0, 0.0, 1.0, 1.0], (640, 480)) is None


def test_label_occupants_accepts_a_face_or_a_track_id():
    tr = IouTracker()
    live = tr.update([_box(300, 300, 200), _box(900, 300, 180)])
    by_face = dict((lbl, t.track_id) for t, lbl in label_occupants(live, live[1].face))
    assert by_face["DRV"] == live[1].track_id
    by_id = dict((lbl, t.track_id) for t, lbl in label_occupants(live, live[0].track_id))
    assert by_id["DRV"] == live[0].track_id
    assert set(by_id) == {"DRV", "P1"}


# --------------------------------------------------------------- selector ---
ROI = [0.0, 0.0, 0.65, 1.0]


def _drive(sel, tracker, frames, roi_wh=FRAME_WH, fps=10.0, embeddings=None):
    """Feed a list of per-frame detection lists; return the picks."""
    picks = []
    for i, faces in enumerate(frames):
        live = tracker.update(faces, t_s=i / fps)
        emb = None
        if embeddings is not None:
            emb = embeddings(i, live)
        picks.append(sel.select(live, roi_wh, t_s=i / fps, embeddings=emb))
    return picks


def test_selector_picks_the_driver_seat_face_and_records_t_last_match():
    sel = DriverSelector(ROI)
    tracker = IouTracker()
    driver = _box(300, 360, 200)
    passenger = _box(1000, 360, 260)  # bigger, but outside the LHD seat ROI
    picks = _drive(sel, tracker, [[driver, passenger]] * 5)
    assert picks[-1].track_id is not None
    assert picks[-1].face is driver
    assert picks[-1].reason in ("roi", "incumbent_held")
    assert sel.t_last_match_s == pytest.approx(0.4)


def test_selector_does_not_flip_on_a_marginal_size_lead():
    """The classic failure: a leaning passenger becomes momentarily larger."""
    sel = DriverSelector(ROI, SelectorConfig(switch_dwell_s=0.6, switch_min_frames=3))
    tracker = IouTracker()
    frames = []
    for i in range(20):
        # both inside the ROI; the second face is 8% larger on odd frames
        a = _box(300, 360, 200)
        b = _box(600, 360, 216 if i % 2 else 190)
        frames.append([a, b])
    picks = _drive(sel, tracker, frames)
    ids = [p.track_id for p in picks]
    assert len(set(ids)) == 1, "driver id must not oscillate: %r" % (ids,)
    assert not any(p.changed for p in picks)


def test_selector_hands_over_after_a_sustained_lead():
    sel = DriverSelector(ROI, SelectorConfig(switch_dwell_s=0.6, switch_min_frames=3))
    tracker = IouTracker()
    small = _box(300, 360, 120)
    big = _box(600, 360, 320)
    frames = [[small]] * 5 + [[small, big]] * 15
    picks = _drive(sel, tracker, frames)
    first_id = picks[0].track_id
    assert picks[5].track_id == first_id, "no handover on the challenger's first frame"
    assert picks[6].track_id == first_id
    assert picks[-1].track_id != first_id
    changes = [i for i, p in enumerate(picks) if p.changed]
    assert len(changes) == 1, "exactly one handover edge"
    assert picks[changes[0]].reason == "handover"
    # the dwell is 0.6 s = 6 frames at 10 FPS, plus the frame the challenger
    # appeared on, so the switch lands on frame 5 + 1 + 6
    assert 10 <= changes[0] <= 13


def test_an_established_track_outranks_a_bigger_newcomer():
    """Track persistence, isolated: incumbency and dwell are switched off."""
    sel = DriverSelector(
        ROI,
        SelectorConfig(
            incumbent_bonus=1.0, switch_dwell_s=0.0, switch_min_frames=1, persistence_bonus=0.25
        ),
    )
    tracker = IouTracker()
    established = _box(300, 360, 200)
    newcomer = _box(600, 360, 216)  # 8% larger, minted this frame
    picks = _drive(sel, tracker, [[established]] * 20 + [[established, newcomer]] * 2)
    assert picks[-1].track_id == picks[0].track_id
    assert picks[-1].face is established

    # with persistence off, the same geometry hands over immediately
    sel2 = DriverSelector(
        ROI,
        SelectorConfig(
            incumbent_bonus=1.0, switch_dwell_s=0.0, switch_min_frames=1, persistence_bonus=0.0
        ),
    )
    picks2 = _drive(sel2, IouTracker(), [[established]] * 20 + [[established, newcomer]] * 2)
    assert picks2[-1].face is newcomer


def test_selector_reports_no_driver_when_the_roi_is_empty_and_fallback_is_off():
    sel = DriverSelector(ROI, SelectorConfig(fallback=False))
    tracker = IouTracker()
    picks = _drive(sel, tracker, [[_box(1100, 360, 200)]] * 4)
    assert all(p.track_id is None for p in picks)
    assert picks[-1].reason == "roi_empty"
    assert sel.t_last_match_s is None, "a passenger must not refresh t_last_match"


def test_selector_fallback_shows_a_passenger_but_does_not_refresh_t_last_match():
    sel = DriverSelector(ROI, SelectorConfig(fallback=True))
    tracker = IouTracker()
    picks = _drive(sel, tracker, [[_box(1100, 360, 200)]] * 4)
    assert picks[-1].track_id is not None
    assert picks[-1].reason in ("fallback", "incumbent_held")
    assert sel.t_last_match_s is None


def test_selector_reports_no_driver_when_nothing_is_detected():
    sel = DriverSelector(ROI)
    tracker = IouTracker()
    picks = _drive(sel, tracker, [[_box(300, 360, 200)]] * 3 + [[]] * 3)
    assert picks[2].track_id is not None
    assert picks[-1].track_id is None and picks[-1].reason == "no_match"


def test_identity_reattaches_the_same_person_to_a_new_track_id():
    """A break in tracking is not a driver change if the face is the same."""
    sel = DriverSelector(ROI)
    tracker = IouTracker(max_lost=0)
    vec = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    frames = [[_box(300, 360, 200)]] * 4 + [[]] * 2 + [[_box(500, 200, 200)]] * 4

    def emb(i, live):
        return dict((t.track_id, vec) for t in live if t.matched)

    picks = _drive(sel, tracker, frames, embeddings=emb)
    assert picks[3].track_id == 1
    assert picks[-1].track_id == 2, "the tracker minted a new id"
    changed = [p for p in picks if p.changed]
    assert changed == [], "identity proves it is the same driver"
    reacquire = picks[6]
    assert reacquire.track_id == 2 and reacquire.reason == "reacquired"
    assert reacquire.identity_score == pytest.approx(1.0, abs=1e-5)


def test_identity_mismatch_is_reported_as_a_driver_change():
    sel = DriverSelector(ROI)
    tracker = IouTracker(max_lost=0)
    a = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    frames = [[_box(300, 360, 200)]] * 4 + [[]] * 2 + [[_box(500, 200, 200)]] * 4

    def emb(i, live):
        vec = a if i < 4 else b
        return dict((t.track_id, vec) for t in live if t.matched)

    picks = _drive(sel, tracker, frames, embeddings=emb)
    changed = [p for p in picks if p.changed]
    assert len(changed) == 1 and changed[0].reason == "handover"
    assert changed[0].identity_score == pytest.approx(0.0, abs=1e-5)


def test_selector_ignores_a_degenerate_embedding():
    sel = DriverSelector(ROI)
    tracker = IouTracker()

    def emb(i, live):
        return dict((t.track_id, np.zeros(4, np.float32)) for t in live)

    picks = _drive(sel, tracker, [[_box(300, 360, 200)]] * 3, embeddings=emb)
    assert picks[-1].identity_score is None
    assert sel.driver_embedding is None


def test_selector_reset_forgets_the_driver():
    sel = DriverSelector(ROI)
    tracker = IouTracker()
    _drive(sel, tracker, [[_box(300, 360, 200)]] * 3)
    assert sel.driver_track_id is not None
    sel.reset()
    assert sel.driver_track_id is None and sel.driver_embedding is None


def test_selector_without_a_roi_uses_every_matched_track():
    sel = DriverSelector(None)
    tracker = IouTracker()
    picks = _drive(sel, tracker, [[_box(1100, 360, 300)]] * 3)
    assert picks[-1].track_id is not None
    assert sel.t_last_match_s == pytest.approx(0.2)
