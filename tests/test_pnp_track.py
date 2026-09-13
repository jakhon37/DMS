"""Head-pose solve, its rejection gates, and the tracker/driver-selection glue."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dms.geometry.face106 import (
    LEFT_EYE_CENTRE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE_TIP,
    PNP_INDICES,
    PNP_OBJECT_POINTS,
    RIGHT_EYE_CENTRE,
)
from dms.geometry.pnp import (
    YAW_SIGN,
    _wrap180,
    camera_matrix,
    camera_matrix_from_fov,
    rotation_to_ypr,
    solve_head_pose,
    solve_head_pose_full,
)
from dms.infer.scrfd import FaceDet
from dms.track.driver_select import pick_driver
from dms.track.iou_tracker import IouTracker, _iou
from dms.types import SignalSource

FRAME_WH = (1280, 720)


def _project(yaw_deg: float, pitch_deg: float, roll_deg: float, distance_mm: float = 700.0):
    """Render the six PnP model points for a known head orientation.

    Angles are in the PROJECT convention, so this function applies YAW_SIGN in
    reverse to build the raw rotation - which makes it an independent check that
    the solver undoes exactly what the renderer did.
    """
    K = camera_matrix(*FRAME_WH)
    rx = np.radians(pitch_deg)
    ry = np.radians(YAW_SIGN * yaw_deg)
    rz = np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[0.0], [0.0], [distance_mm]], dtype=np.float64)
    pts, _ = cv2.projectPoints(PNP_OBJECT_POINTS, rvec, tvec, K, np.zeros((4, 1)))
    return pts.reshape(-1, 2)


def _landmarks_from(image_pts: np.ndarray) -> np.ndarray:
    """Scatter six projected points into a full 106-point array."""
    lm = np.zeros((106, 2), dtype=np.float64)
    for idx, pt in zip(PNP_INDICES, image_pts):
        lm[idx] = pt
    # the eye centres are only used for the inter-ocular scale; put them a
    # little inboard of the outer corners
    lm[LEFT_EYE_CENTRE] = image_pts[2] + (image_pts[3] - image_pts[2]) * 0.2
    lm[RIGHT_EYE_CENTRE] = image_pts[3] + (image_pts[2] - image_pts[3]) * 0.2
    return lm


# ------------------------------------------------------------- rotation ----
def test_rotation_identity_near_zero():
    yaw, pitch, roll = rotation_to_ypr(np.eye(3))
    assert abs(yaw) < 1e-3 and abs(pitch) < 1e-3 and abs(roll) < 1e-3


def test_rotation_to_ypr_is_single_branch_over_the_useful_range():
    """No mirrored solution, so no +/-180 cliff to patch up afterwards."""
    for yaw in range(-60, 61, 15):
        for pitch in range(-30, 31, 15):
            ry, rx = np.radians(yaw), np.radians(pitch)
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            got_yaw, got_pitch, got_roll = rotation_to_ypr(Ry @ Rx)
            assert abs(got_yaw - yaw) < 1e-6
            assert abs(got_pitch - pitch) < 1e-6
            assert abs(got_roll) < 1e-6


def test_wrap180():
    assert _wrap180(190.0) == pytest.approx(-170.0)
    assert _wrap180(-190.0) == pytest.approx(170.0)
    assert _wrap180(180.0) == pytest.approx(180.0)
    assert _wrap180(float("nan")) == 0.0


def test_camera_matrix_from_fov():
    K = camera_matrix_from_fov(1280, 720, 90.0)
    assert K[0, 0] == pytest.approx(640.0)
    assert K[0, 2] == pytest.approx(640.0) and K[1, 2] == pytest.approx(360.0)
    wide = camera_matrix_from_fov(1280, 720, 120.0)
    assert wide[0, 0] < K[0, 0]
    with pytest.raises(ValueError):
        camera_matrix_from_fov(1280, 720, 0.0)


# ------------------------------------------------------------ solve gate ----
def test_pnp_recovers_a_synthetic_sweep_without_sign_flips():
    """The old decomposeProjectionMatrix path spanned 169 deg on a stationary head."""
    prev_yaw = None
    for yaw in range(-45, 46, 5):
        lm = _landmarks_from(_project(yaw, 0.0, 0.0))
        pose = solve_head_pose_full(lm, FRAME_WH)
        assert pose.source is SignalSource.GEOMETRY, "yaw %d was rejected" % yaw
        assert abs(pose.yaw - yaw) < 3.0, "yaw %d recovered as %.1f" % (yaw, pose.yaw)
        assert abs(pose.pitch) < 3.0 and abs(pose.roll) < 3.0
        if prev_yaw is not None:
            assert pose.yaw > prev_yaw, "yaw is not monotonic at %d" % yaw
        prev_yaw = pose.yaw


def test_pnp_recovers_pitch_and_roll():
    for pitch in (-25, -10, 0, 10, 25):
        pose = solve_head_pose_full(_landmarks_from(_project(0.0, pitch, 0.0)), FRAME_WH)
        assert abs(pose.pitch - pitch) < 3.0, "pitch %d -> %.1f" % (pitch, pose.pitch)
    for roll in (-25, 0, 25):
        pose = solve_head_pose_full(_landmarks_from(_project(0.0, 0.0, roll)), FRAME_WH)
        assert abs(pose.roll - roll) < 3.0, "roll %d -> %.1f" % (roll, pose.roll)


def test_pnp_never_reports_the_180_degree_branch():
    for yaw in range(-60, 61, 10):
        for pitch in (-20, 0, 20):
            pose = solve_head_pose_full(_landmarks_from(_project(yaw, pitch, 0.0)), FRAME_WH)
            assert abs(pose.pitch) < 90.0
            assert abs(pose.yaw) < 90.0


def test_pnp_reprojection_gate_rejects_scrambled_landmarks():
    """A pose with a high residual must return confidence 0, not a wrong angle."""
    pts = _project(0.0, 0.0, 0.0)
    lm = _landmarks_from(pts)
    good = solve_head_pose_full(lm, FRAME_WH)
    assert good.confidence > 0.5
    assert good.reprojection_error < 1.0
    # move the chin 60 px sideways: geometrically impossible for a rigid head
    lm[PNP_INDICES[1]] = lm[PNP_INDICES[1]] + np.array([60.0, 0.0])
    bad = solve_head_pose_full(lm, FRAME_WH)
    assert bad.source is SignalSource.UNAVAILABLE
    assert bad.confidence == 0.0
    assert solve_head_pose(lm, FRAME_WH) is None


def test_pnp_rejects_a_face_that_is_too_small():
    lm = _landmarks_from(_project(0.0, 0.0, 0.0, distance_mm=12000.0))
    pose = solve_head_pose_full(lm, FRAME_WH)
    assert pose.source is SignalSource.UNAVAILABLE


def test_pnp_rejects_non_finite_landmarks():
    lm = _landmarks_from(_project(0.0, 0.0, 0.0))
    lm[NOSE_TIP] = [np.nan, 0.0]
    assert solve_head_pose_full(lm, FRAME_WH).source is SignalSource.UNAVAILABLE
    assert solve_head_pose(lm, FRAME_WH) is None


def test_pnp_rejects_landmarks_that_disagree_with_the_detector_keypoints():
    lm = _landmarks_from(_project(0.0, 0.0, 0.0))
    kps = np.stack(
        [lm[LEFT_EYE_CENTRE], lm[RIGHT_EYE_CENTRE], lm[NOSE_TIP], lm[MOUTH_LEFT], lm[MOUTH_RIGHT]]
    )
    assert solve_head_pose_full(lm, FRAME_WH, kps5=kps).source is SignalSource.GEOMETRY
    displaced = kps + np.array([80.0, 0.0])
    assert solve_head_pose_full(lm, FRAME_WH, kps5=displaced).source is SignalSource.UNAVAILABLE


def test_pnp_yaw_sign_matches_the_documented_convention():
    """yaw > 0 must mean the driver turned toward their own LEFT.

    Independent of the solver: when the driver turns to their own left, their
    nose swings toward image-right relative to the eye midpoint.
    """
    pts = _project(30.0, 0.0, 0.0)
    lm = _landmarks_from(pts)
    pose = solve_head_pose_full(lm, FRAME_WH)
    assert pose.yaw > 15.0
    eye_mid_x = 0.5 * (lm[LEFT_EYE_CENTRE][0] + lm[RIGHT_EYE_CENTRE][0])
    assert lm[NOSE_TIP][0] > eye_mid_x, "positive yaw must move the nose to image-right"

    neg = solve_head_pose_full(_landmarks_from(_project(-30.0, 0.0, 0.0)), FRAME_WH)
    assert neg.yaw < -15.0


def test_pnp_confidence_falls_as_the_residual_grows():
    lm = _landmarks_from(_project(0.0, 0.0, 0.0))
    clean = solve_head_pose_full(lm, FRAME_WH)
    noisy = lm.copy()
    noisy[PNP_INDICES[4]] = noisy[PNP_INDICES[4]] + np.array([4.0, 0.0])
    dirty = solve_head_pose_full(noisy, FRAME_WH)
    assert dirty.source is SignalSource.GEOMETRY
    assert 0.0 < dirty.confidence < clean.confidence


def test_solve_head_pose_returns_a_plain_tuple_for_legacy_callers():
    lm = _landmarks_from(_project(10.0, 5.0, 0.0))
    out = solve_head_pose(lm, FRAME_WH)
    assert out is not None and len(out) == 3
    assert all(np.isfinite(out))


# ---------------------------------------------------------------- tracker ---
def test_iou_overlap():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([0, 0, 10, 10], dtype=np.float32)
    assert _iou(a, b) > 0.99
    c = np.array([20, 20, 30, 30], dtype=np.float32)
    assert _iou(a, c) == 0.0


def test_tracker_keeps_id():
    tr = IouTracker()
    f1 = FaceDet(xyxy=np.array([10, 10, 50, 50], np.float32), score=0.9)
    t0 = tr.update([f1])
    assert t0[0].track_id == 1
    f2 = FaceDet(xyxy=np.array([12, 11, 52, 49], np.float32), score=0.8)
    t1 = tr.update([f2])
    assert t1[0].track_id == 1
    assert t1[0].lost == 0


def test_pick_driver_no_fallback_returns_none():
    pax = FaceDet(xyxy=np.array([10, 10, 80, 80], np.float32), score=0.9)
    assert pick_driver([pax], [0.5, 0.0, 1.0, 1.0], (640, 480), fallback=False) is None


def test_pick_driver_largest_in_roi_not_highest_score():
    small_hi = FaceDet(xyxy=np.array([10, 10, 40, 50], np.float32), score=0.99)
    big_lo = FaceDet(xyxy=np.array([200, 80, 400, 360], np.float32), score=0.70)
    d = pick_driver([small_hi, big_lo], [0.4, 0.0, 1.0, 1.0], (640, 480))
    assert d is big_lo
