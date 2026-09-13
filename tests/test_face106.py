"""Tests for the 106-point topology and the geometry built on it.

The previous version of this file was tautological: it asserted
``LEFT_EYE_CORNER in LEFT_EYE`` where ``LEFT_EYE_CORNER == LEFT_EYE.start``, so
every slice could have been off by ten and the suite would still have been
green.  These tests check the table against a REAL landmark set captured from
the shipping engine on a replay frame (``tests/fixtures/landmarks106.npz``),
which is the only thing that can catch a wrong index.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dms.geometry.ear import contour_ear, eye_aspect_ratio, eye_ratio, face_ear, face_ear_detail
from dms.geometry.face106 import (
    BROW_L,
    BROW_R,
    CHIN,
    CONTOUR,
    EYE_LID_PAIR_OFFSETS,
    INNER_LIP,
    LEFT_EYE,
    LEFT_EYE_CENTRE,
    LEFT_EYE_OUTER,
    MOUTH,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE,
    NOSE_TIP,
    OUTER_LIP,
    PNP_INDICES,
    PNP_OBJECT_POINTS,
    RIGHT_EYE,
    RIGHT_EYE_CENTRE,
    RIGHT_EYE_OUTER,
    eye_box,
    eye_contour,
    eye_corners,
    inter_ocular,
    landmarks_agree_with_kps,
    mouth_box,
    roll_from_eyes,
    roll_from_kps,
)
from dms.geometry.mar import mouth_aspect_ratio, mouth_aspect_ratio_detail

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "landmarks106.npz")


def _fixture():
    if not os.path.isfile(FIXTURE):
        pytest.skip("landmark fixture not captured on this machine: %s" % FIXTURE)
    return np.load(FIXTURE)


# ------------------------------------------------------------- table shape --
def test_groups_partition_all_106_points_exactly_once():
    covered = []
    for grp in (CONTOUR, LEFT_EYE, BROW_L, MOUTH, NOSE, RIGHT_EYE, BROW_R):
        covered.extend(list(grp))
    assert sorted(covered) == list(range(106)), "groups must tile 0..105 with no gap or overlap"


def test_group_sizes_match_the_verified_topology():
    assert len(CONTOUR) == 33
    assert len(LEFT_EYE) == len(RIGHT_EYE) == 10
    assert len(BROW_L) == len(BROW_R) == 9
    assert len(MOUTH) == 20
    assert len(NOSE) == 15


def test_eyes_use_equal_point_counts():
    """The old table averaged a 9-point eye with an 8-point eye."""
    lm = np.zeros((106, 2), dtype=np.float32)
    assert eye_contour(lm, True).shape == eye_contour(lm, False).shape == (8, 2)


def test_lip_rings_partition_the_mouth_block():
    assert sorted(list(OUTER_LIP) + list(INNER_LIP)) == list(MOUTH)
    assert len(OUTER_LIP) == 12
    assert len(INNER_LIP) == 8


def test_pnp_indices_are_in_their_declared_groups():
    assert NOSE_TIP in NOSE
    assert CHIN in CONTOUR
    assert LEFT_EYE_OUTER in LEFT_EYE
    assert RIGHT_EYE_OUTER in RIGHT_EYE
    assert MOUTH_LEFT in OUTER_LIP
    assert MOUTH_RIGHT in OUTER_LIP
    assert len(PNP_INDICES) == len(set(PNP_INDICES)) == 6


def test_pnp_object_points_have_anatomical_depth_ordering():
    """Nose tip nearest the camera, chin markedly shallower than the eye corners.

    +Z is away from the camera, so the nose tip - the most protruding part of a
    face - must be the minimum and everything else strictly positive.  The
    previous table had chin z = -28 against eye corners at z = -30: the chin sat
    as deep in the head as the eye sockets, and the whole depth axis pointed the
    wrong way.  Both errors bias solvePnP's rotation and are the documented
    source of the +/-80 deg pitch noise.
    """
    nose, chin, eye_l, eye_r, mouth_l, mouth_r = PNP_OBJECT_POINTS
    assert nose[2] == 0.0
    assert min(chin[2], eye_l[2], eye_r[2], mouth_l[2], mouth_r[2]) > 0.0
    assert chin[2] < eye_l[2] and chin[2] < eye_r[2]
    assert chin[2] < 0.6 * eye_l[2]
    # left/right symmetry
    assert np.allclose(eye_l[0], -eye_r[0])
    assert np.allclose(mouth_l[0], -mouth_r[0])
    # y grows downward: eyes above the nose tip, chin and mouth below it
    assert eye_l[1] < 0.0 and eye_r[1] < 0.0
    assert chin[1] > 0.0 and mouth_l[1] > 0.0
    # adult scale in millimetres
    assert 70.0 < (eye_r[0] - eye_l[0]) < 110.0
    assert 50.0 < chin[1] < 90.0


def test_recommended_thresholds_are_on_the_new_metric_scale():
    """Guard the replacement thresholds the config workstream must adopt.

    The shipped configs/*.yaml still carry ``ear_closed: 0.50`` and
    ``mar_yawn: 0.65``, which were calibrated against the OLD bounding-box EAR
    (0.67-0.88 on open eyes) and the OLD "MAR" that was actually measuring the
    image-right eye (0.52 on a closed mouth).  Against the metrics in this
    module those values mean "every open eye is closed" and "a yawn is
    impossible".  These constants are the authoritative replacements; this test
    pins them to the band the fixture and the replay measurements support so a
    later edit cannot quietly drift them back.
    """
    from dms.geometry.ear import (
        EAR_CLOSED_RECOMMENDED,
        EAR_MAX_PLAUSIBLE,
        EAR_OPEN_MEDIAN_RECOMMENDED,
    )
    from dms.geometry.mar import MAR_YAWN_RECOMMENDED

    assert 0.12 < EAR_CLOSED_RECOMMENDED < 0.22
    assert 0.22 < EAR_OPEN_MEDIAN_RECOMMENDED < 0.40
    assert EAR_CLOSED_RECOMMENDED < EAR_OPEN_MEDIAN_RECOMMENDED < EAR_MAX_PLAUSIBLE
    # driver_state uses min(ear_closed, 0.70 * ear_open_median) as the effective
    # threshold; that must stay inside the closed band, not above the open one.
    effective = min(EAR_CLOSED_RECOMMENDED, 0.70 * EAR_OPEN_MEDIAN_RECOMMENDED)
    assert 0.12 < effective < 0.22
    assert 0.30 < MAR_YAWN_RECOMMENDED < 0.60


def test_the_fixture_open_eye_would_not_trip_the_recommended_closed_threshold():
    from dms.geometry.ear import EAR_CLOSED_RECOMMENDED

    d = _fixture()
    mean, left, right, valid = face_ear_detail(d["landmarks"])
    assert valid
    assert min(left, right) > EAR_CLOSED_RECOMMENDED, (
        "an open eye at %.3f/%.3f must sit above the closed threshold %.3f"
        % (left, right, EAR_CLOSED_RECOMMENDED)
    )


# -------------------------------------------------- fixture-backed checks ---
def test_fixture_landmarks_agree_with_detector_keypoints():
    """The five SCRFD keypoints must land on the indices the table claims."""
    d = _fixture()
    lm, kps = d["landmarks"], d["kps"]
    iod = inter_ocular(lm)
    assert iod > 20.0
    pairs = {
        "left eye": (LEFT_EYE_CENTRE, 0),
        "right eye": (RIGHT_EYE_CENTRE, 1),
        "nose tip": (NOSE_TIP, 2),
        "left mouth": (MOUTH_LEFT, 3),
        "right mouth": (MOUTH_RIGHT, 4),
    }
    for name, (idx, k) in pairs.items():
        err = float(np.linalg.norm(lm[idx] - kps[k]) / iod)
        assert err < 0.25, "%s: index %d is %.2f IOD from the detector keypoint" % (name, idx, err)


def test_fixture_eye_groups_sit_above_the_brow_groups_they_pair_with():
    d = _fixture()
    lm = d["landmarks"]
    # brows are above (smaller y) their eyes on a roughly upright face
    assert lm[list(BROW_L)][:, 1].mean() < lm[list(LEFT_EYE)][:, 1].mean()
    assert lm[list(BROW_R)][:, 1].mean() < lm[list(RIGHT_EYE)][:, 1].mean()
    # image-left eye really is to the left of the image-right eye
    assert lm[list(LEFT_EYE)][:, 0].mean() < lm[list(RIGHT_EYE)][:, 0].mean()


def test_fixture_eye_corner_offsets_are_the_extremes_in_x():
    d = _fixture()
    lm = d["landmarks"]
    for left in (True, False):
        pts = eye_contour(lm, left)
        c_lo, c_hi = eye_corners(lm, left)
        assert c_lo[0] <= pts[:, 0].min() + 1e-3
        assert c_hi[0] >= pts[:, 0].max() - 1e-3


def test_fixture_lid_pairs_have_upper_above_lower():
    d = _fixture()
    lm = d["landmarks"]
    from dms.geometry.face106 import eye_base

    for left in (True, False):
        base = eye_base(left)
        for up, lo in EYE_LID_PAIR_OFFSETS:
            assert lm[base + up][1] < lm[base + lo][1], "offset %d must be above %d" % (up, lo)


def test_fixture_open_eye_ear_is_anatomically_plausible():
    """An open eye reads 0.20-0.45. The old bounding-box metric read 0.67-0.88."""
    d = _fixture()
    lm = d["landmarks"]
    mean, left, right, valid = face_ear_detail(lm)
    assert valid
    for v in (left, right):
        assert 0.15 < v < 0.50, "eye ratio %.3f is outside the anatomical range" % v
    assert abs(left - right) < 0.12


def test_fixture_closed_mouth_mar_is_near_zero():
    d = _fixture()
    lm = d["landmarks"]
    mar, valid = mouth_aspect_ratio_detail(lm)
    assert valid
    assert mar < 0.15, "a closed mouth must not read %.3f" % mar


def test_fixture_eye_and_mouth_boxes_are_inside_the_face_box():
    d = _fixture()
    lm, box = d["landmarks"], d["box"]
    pad = 0.25 * max(box[2] - box[0], box[3] - box[1])
    for b in (eye_box(lm, True), eye_box(lm, False), mouth_box(lm)):
        assert b[0] > box[0] - pad and b[1] > box[1] - pad
        assert b[2] < box[2] + pad and b[3] < box[3] + pad
        assert b[2] > b[0] and b[3] > b[1]


def test_fixture_roll_estimates_agree():
    d = _fixture()
    lm, kps = d["landmarks"], d["kps"]
    assert abs(roll_from_eyes(lm) - roll_from_kps(kps)) < 12.0


# ----------------------------------------------------- synthetic behaviour --
def _synthetic_eye(base: int, cx: float, cy: float, half_w: float, half_h: float):
    """Build one eye group in the verified offset order."""
    pts = np.zeros((10, 2), dtype=np.float32)
    pts[0] = [cx, cy + half_h]  # lower mid
    pts[1] = [cx, cy]  # centre
    pts[5] = [cx, cy]  # centre duplicate
    pts[2] = [cx - half_w, cy]  # corner, small x
    pts[6] = [cx + half_w, cy]  # corner, large x
    pts[3] = [cx - 0.5 * half_w, cy + 0.8 * half_h]
    pts[4] = [cx + 0.5 * half_w, cy + 0.8 * half_h]
    pts[7] = [cx, cy - half_h]
    pts[8] = [cx - 0.5 * half_w, cy - 0.8 * half_h]
    pts[9] = [cx + 0.5 * half_w, cy - 0.8 * half_h]
    return base, pts


def _synthetic_face(eye_half_h: float = 6.0, roll_deg: float = 0.0) -> np.ndarray:
    lm = np.zeros((106, 2), dtype=np.float32)
    for base, pts in (
        _synthetic_eye(33, 70.0, 100.0, 18.0, eye_half_h),
        _synthetic_eye(87, 130.0, 100.0, 18.0, eye_half_h),
    ):
        lm[base : base + 10] = pts
    lm[NOSE_TIP] = [100.0, 130.0]
    lm[CHIN] = [100.0, 190.0]
    for i, idx in enumerate(OUTER_LIP):
        ang = 2.0 * np.pi * i / len(OUTER_LIP)
        lm[idx] = [100.0 + 30.0 * np.cos(ang), 160.0 + 10.0 * np.sin(ang)]
    lm[MOUTH_LEFT] = [70.0, 160.0]
    lm[MOUTH_RIGHT] = [130.0, 160.0]
    for i, idx in enumerate(INNER_LIP):
        ang = 2.0 * np.pi * i / len(INNER_LIP)
        lm[idx] = [100.0 + 22.0 * np.cos(ang), 160.0 + 1.0 * np.sin(ang)]
    if roll_deg:
        th = np.radians(roll_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        c = np.array([100.0, 130.0], dtype=np.float32)
        lm = (lm - c) @ R.T + c
    return lm.astype(np.float32)


def test_ear_is_invariant_to_head_roll():
    """The whole point of the rewrite: 30 deg of roll must not open a shut eye."""
    shut = _synthetic_face(eye_half_h=0.3)
    flat = face_ear(shut)
    for roll in (-30.0, -15.0, 15.0, 30.0, 45.0):
        rolled = face_ear(_synthetic_face(eye_half_h=0.3, roll_deg=roll))
        assert abs(rolled - flat) < 0.02, "roll %.0f changed EAR %.3f -> %.3f" % (roll, flat, rolled)
    assert flat < 0.06


def test_ear_separates_open_from_closed():
    assert face_ear(_synthetic_face(eye_half_h=6.0)) > 0.25
    assert face_ear(_synthetic_face(eye_half_h=0.3)) < 0.06


def test_mar_is_invariant_to_head_roll():
    flat = mouth_aspect_ratio(_synthetic_face())
    for roll in (-30.0, 30.0):
        assert abs(mouth_aspect_ratio(_synthetic_face(roll_deg=roll)) - flat) < 0.02


def test_contour_ear_is_also_roll_invariant():
    """The index-free fallback must not read roll as opening either."""
    shut = _synthetic_face(eye_half_h=0.3)
    flat = contour_ear(eye_contour(shut, True))
    rolled = contour_ear(eye_contour(_synthetic_face(eye_half_h=0.3, roll_deg=30.0), True))
    assert abs(rolled - flat) < 0.02


def test_soukupova_six_point_form_still_works():
    open_eye = np.array([[0, 0], [1, -2], [2, -2], [3, 0], [2, 2], [1, 2]], dtype=np.float32)
    closed = np.array([[0, 0], [1, -0.1], [2, -0.1], [3, 0], [2, 0.1], [1, 0.1]], dtype=np.float32)
    assert eye_aspect_ratio(open_eye) > 0.5
    assert eye_aspect_ratio(closed) < 0.2


def test_degenerate_landmarks_report_invalid_not_zero_ear():
    lm = np.zeros((106, 2), dtype=np.float32)
    mean, left, right, valid = face_ear_detail(lm)
    assert not valid
    assert mean == 0.0
    mar, mar_valid = mouth_aspect_ratio_detail(lm)
    assert not mar_valid


def test_non_finite_landmarks_do_not_raise():
    lm = _synthetic_face()
    lm[10] = [np.nan, np.inf]
    assert face_ear(lm) == 0.0
    assert mouth_aspect_ratio(lm) == 0.0
    assert eye_ratio(lm, True) > 0.0  # the eye block itself is still finite


def test_kps_cross_check_flags_a_displaced_landmark_set():
    lm = _synthetic_face()
    kps = np.stack(
        [lm[LEFT_EYE_CENTRE], lm[RIGHT_EYE_CENTRE], lm[NOSE_TIP], lm[MOUTH_LEFT], lm[MOUTH_RIGHT]]
    )
    agree, err = landmarks_agree_with_kps(lm, kps)
    assert agree and err < 1e-6
    shifted = lm + np.array([40.0, 0.0], dtype=np.float32)
    agree, err = landmarks_agree_with_kps(shifted, kps)
    assert not agree and err > 0.25


def test_kps_cross_check_passes_when_no_keypoints_available():
    agree, err = landmarks_agree_with_kps(_synthetic_face(), None)
    assert agree and err == 0.0


def test_roll_sign_convention():
    """Positive roll = the image-right eye sits lower."""
    lm = _synthetic_face()
    lm[RIGHT_EYE_CENTRE] = lm[RIGHT_EYE_CENTRE] + np.array([0.0, 10.0], dtype=np.float32)
    assert roll_from_eyes(lm) > 0.0
    kps = np.array([[0.0, 0.0], [10.0, 5.0], [5.0, 8.0], [2.0, 12.0], [8.0, 12.0]], dtype=np.float32)
    assert roll_from_kps(kps) > 0.0
