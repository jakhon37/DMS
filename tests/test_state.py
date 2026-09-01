from __future__ import annotations

from dms.config.schema import load_config
from dms.state.alerts import AlertType, Severity
from dms.state.driver_state import DriverMonitor, angdiff


def _mon(cfg=None):
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = cfg or load_config(os.path.join(root, "configs/default.yaml"))
    return DriverMonitor(cfg)


def test_angdiff_wrap():
    assert abs(angdiff(-63.0, -63.0)) < 1e-6
    assert abs(angdiff(170, -170) - (-20)) < 1e-6 or abs(angdiff(170, -170) - 20) < 25


def test_microsleep_needs_duration():
    m = _mon()
    ev = []
    # present, eyes open
    for i in range(5):
        ev += m.update(i * 0.1, present=True, track_id=1, ear=0.80, mar=0.4, yaw=0.0, pitch=0.0)
    assert AlertType.MICROSLEEP not in m.active
    # closed but only 0.5s
    t0 = 1.0
    for i in range(6):
        ev += m.update(t0 + i * 0.1, present=True, track_id=1, ear=0.20, mar=0.4, yaw=0.0, pitch=0.0)
    assert AlertType.MICROSLEEP not in m.active
    # continue closed to 1.6s
    for i in range(12):
        ev += m.update(1.6 + i * 0.1, present=True, track_id=1, ear=0.20, mar=0.4, yaw=0.0, pitch=0.0)
    types = [e.type for e in ev if e.edge == "enter"]
    assert AlertType.MICROSLEEP in types
    assert m.active[AlertType.MICROSLEEP] == Severity.CRITICAL


def test_gaze_uses_forward_zero():
    m = _mon()
    m.cfg.forward_zero.yaw = -63.0
    m.cfg.forward_zero.pitch = -6.0
    ev = []
    for i in range(30):
        ev += m.update(
            i * 0.1,
            present=True,
            track_id=1,
            ear=0.8,
            mar=0.4,
            yaw=-62.0,
            pitch=-7.0,
        )
    assert AlertType.GAZE_AWAY not in m.active
    for i in range(30):
        ev += m.update(
            3.0 + i * 0.1,
            present=True,
            track_id=1,
            ear=0.8,
            mar=0.4,
            yaw=20.0,
            pitch=-7.0,
        )
    assert AlertType.GAZE_AWAY in m.active


def test_face_lost_not_on_first_frames():
    m = _mon()
    ev = m.update(0.0, present=False, track_id=None, ear=None, mar=None, yaw=None, pitch=None)
    assert not any(e.type == AlertType.FACE_LOST for e in ev)
    m.update(0.1, present=True, track_id=1, ear=0.8, mar=0.4, yaw=0.0, pitch=0.0)
    ev = []
    for i in range(15):
        ev += m.update(1.0 + i * 0.1, present=False, track_id=None, ear=None, mar=None, yaw=None, pitch=None)
    assert any(e.type == AlertType.FACE_LOST and e.edge == "enter" for e in ev)
