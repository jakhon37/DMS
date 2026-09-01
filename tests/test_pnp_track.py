from __future__ import annotations

import numpy as np

from dms.geometry.pnp import rotation_to_ypr, solve_head_pose
from dms.infer.scrfd import FaceDet
from dms.track.driver_select import pick_driver
from dms.track.iou_tracker import IouTracker, _iou


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


def test_pnp_returns_finite_on_frontalish():
    lm = np.zeros((106, 2), dtype=np.float32)
    # UniFace PnP indices: 51 nose, 16 chin, 63 L eye, 76 R eye, 87 L mouth, 93 R
    lm[51] = [320, 200]
    lm[16] = [320, 280]
    lm[63] = [280, 180]
    lm[76] = [360, 180]
    lm[87] = [300, 240]
    lm[93] = [340, 240]
    pose = solve_head_pose(lm, (640, 480))
    assert pose is not None
    yaw, pitch, roll = pose
    assert np.isfinite([yaw, pitch, roll]).all()


def test_pick_driver_no_fallback_returns_none():
    pax = FaceDet(xyxy=np.array([10, 10, 80, 80], np.float32), score=0.9)
    assert pick_driver([pax], [0.5, 0.0, 1.0, 1.0], (640, 480), fallback=False) is None


def test_pick_driver_largest_in_roi_not_highest_score():
    small_hi = FaceDet(xyxy=np.array([10, 10, 40, 50], np.float32), score=0.99)
    big_lo = FaceDet(xyxy=np.array([200, 80, 400, 360], np.float32), score=0.70)
    # ROI is the right half (minivan-style)
    d = pick_driver([small_hi, big_lo], [0.4, 0.0, 1.0, 1.0], (640, 480))
    assert d is big_lo


def test_rotation_identity_near_zero():
    yaw, pitch, roll = rotation_to_ypr(np.eye(3))
    assert abs(yaw) < 1e-3
    assert abs(pitch) < 1e-3
    assert abs(roll) < 1e-3
