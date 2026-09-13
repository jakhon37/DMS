"""Integration tests for the assembled application.

Three layers, so that most of the value is available with no GPU:

1. Pure-function and argument-handling tests for ``dms.app`` - these run
   anywhere and cover the refusals (privacy gate, bad config, missing clip).
2. Structural tests over the golden replay fixtures - that every clip in
   ``tests/replay`` has a golden, that the goldens are well formed, and that
   they are internally consistent.
3. ``@pytest.mark.board`` tests that actually execute the pipeline, including
   the golden replay comparison itself - one test per clip, which is the gate
   that fails on a detection-quality regression.  They are skipped unless
   ``DMS_REPLAY_TESTS=1`` is set, because they need the GPU and the whole GPU is
   shared with other work on this board.  Run them as::

       DMS_REPLAY_TESTS=1 flock /tmp/jetson-gpu.lock -c \\
           "PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/test_app_integration.py"

   Layer 2 is deliberately not just structural any more.  A green suite used to
   be compatible with a 15.8% drop in driver-face recall, because the only thing
   that measured recall was ``tests/replay/golden/run_golden.py``, which pytest
   never invoked: the SCRFD-640 -> 512 swap put three of the nine goldens into
   regression and the suite stayed green.  ``test_golden_detector_matches_the_
   profile_today`` closes that with no GPU and no clip, by comparing the
   detector each golden was RECORDED against with the detector its profile
   selects now; a swap is then a red suite everywhere, not only on the board.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import pytest

from dms import app as dms_app
from dms.config.schema import load_config
from dms.types import DriverSignals, EyeState

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN_DIR = os.path.join(ROOT, "tests", "replay", "golden")
CLIPS = sorted(glob.glob(os.path.join(ROOT, "tests", "replay", "*.mp4")))
BOARD = pytest.mark.board
NEEDS_BOARD = pytest.mark.skipif(
    os.environ.get("DMS_REPLAY_TESTS") != "1",
    reason="set DMS_REPLAY_TESTS=1 to run the GPU replay tests (hold the GPU mutex)",
)


# ------------------------------------------------------------------- helpers


def test_ear_of_averages_the_eyes_that_reported():
    from dms.types import SignalSource

    sig = DriverSignals()
    assert dms_app._ear_of(sig) is None
    sig.left_eye = EyeState(ear=0.20, source=SignalSource.GEOMETRY)
    assert dms_app._ear_of(sig) == pytest.approx(0.20)
    sig.right_eye = EyeState(ear=0.30, source=SignalSource.GEOMETRY)
    assert dms_app._ear_of(sig) == pytest.approx(0.25)


def test_median():
    assert dms_app._median([]) == 0.0
    assert dms_app._median([3.0]) == 3.0
    assert dms_app._median([1.0, 3.0]) == pytest.approx(2.0)
    assert dms_app._median([5.0, 1.0, 3.0]) == 3.0


def test_zero_is_set():
    from dms.config.schema import ForwardZero

    assert dms_app._zero_is_set(ForwardZero()) is False
    assert dms_app._zero_is_set(ForwardZero(yaw=-3.0)) is True
    assert dms_app._zero_is_set(ForwardZero(roll=0.5)) is True


def test_percentile_map_reports_e2e_not_infer_under_the_e2e_label():
    """The old build published an infer-only number as ``e2e_p95``.

    That made the 66 ms freeze gate unmeasurable from ``/healthz``.
    """
    from dms.perf import LatencyBudget
    from dms.runtime import RollingStats

    budget = LatencyBudget()
    infer = RollingStats(maxlen=32)
    for i in range(40):
        budget.record("e2e", 50.0 + i)
        budget.record("detect_gpu", 10.0)
        infer.add(20.0)
    out = dms_app._percentile_map(budget, infer)
    assert out["e2e_p95"] > out["infer_p95"]
    assert out["detect_gpu_p95"] == pytest.approx(10.0)
    assert "capture_lag_p95" not in out  # nothing recorded it


def test_percentile_map_is_empty_before_any_sample():
    from dms.perf import LatencyBudget
    from dms.runtime import RollingStats

    assert dms_app._percentile_map(LatencyBudget(), RollingStats()) == {}


# --------------------------------------------------------------- CLI refusals


def test_bad_config_path_returns_2(tmp_path):
    assert dms_app.main(["--config", str(tmp_path / "nope.yaml")]) == 2


def test_invalid_config_returns_2(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("models:\n  face: {engine: e}\n  landmarks: {engine: e}\nboom: 1\n")
    assert dms_app.main(["--config", str(p)]) == 2


def test_missing_replay_clip_returns_2(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "source: {type: file, path: /nonexistent/clip.mp4}\n"
        "models:\n  face: {engine: e}\n  landmarks: {engine: e}\n"
        "calibration: ''\n"
    )
    assert dms_app.main(["--config", str(p)]) == 2


def test_save_video_refused_without_the_privacy_switch(tmp_path):
    """Imagery egress is refused at argument-validation time, not ignored later."""
    cfg = os.path.join(ROOT, "configs", "testsrc.yaml")
    assert load_config(cfg).privacy.record_faces is False
    rc = dms_app.main(
        ["--config", cfg, "--max-frames", "1", "--save-video", str(tmp_path / "o.avi")]
    )
    assert rc == 2
    assert not (tmp_path / "o.avi").exists()


def test_save_preview_refused_without_the_privacy_switch(tmp_path):
    cfg = os.path.join(ROOT, "configs", "testsrc.yaml")
    rc = dms_app.main(
        ["--config", cfg, "--max-frames", "1", "--save-preview", str(tmp_path / "o.jpg")]
    )
    assert rc == 2


def test_display_refused_without_the_privacy_switch(tmp_path):
    """The overlay renders the driver's face onto a screen; that is egress too."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "source: {type: test}\n"
        "models:\n  face: {engine: e}\n  landmarks: {engine: e}\n"
        "display: {enabled: true}\n"
        "calibration: ''\n"
    )
    assert dms_app.main(["--config", str(p), "--max-frames", "1"]) == 2


def test_bad_log_level_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        dms_app.build_parser().parse_args(["--log-level", "BOGUS"])


def test_detect_defaults_to_on_and_can_be_turned_off():
    p = dms_app.build_parser()
    assert p.parse_args([]).detect is None  # None means "the profile decides" -> on
    assert p.parse_args(["--detect"]).detect is True
    assert p.parse_args(["--no-detect"]).detect is False


def test_source_override_forces_a_fatal_file_replay(tmp_path, monkeypatch):
    """``--source`` must not inherit a vehicle's forgiving camera policy."""
    cfg = load_config(os.path.join(ROOT, "configs", "production.yaml"))
    assert cfg.camera.retry_forever is True
    # main() applies the override before any resource is acquired; replicate it.
    cfg.source.type = "file"
    cfg.source.path = CLIPS[0] if CLIPS else "x.mp4"
    cfg.camera.retry_forever = False
    cfg.camera.fail_fatal = True
    assert cfg.is_replay is True


# ------------------------------------------------------------ watchdog policy


class _FakeSupervisor(object):
    """Just enough of CaptureSupervisor for the watchdog policy test."""

    def __init__(self, state, camera_fail_active=False, camera_status="ok"):
        self.state = state
        self.camera_fail_active = camera_fail_active
        self.camera_status = camera_status


def _watchdog_probe(state, camera_fail_active, frame_id, last_wd_frame, monkeypatch):
    from dms.runtime import CaptureState

    pinged = []
    monkeypatch.setattr(dms_app, "notify_watchdog", lambda: pinged.append(1))
    monkeypatch.setattr(dms_app, "notify_status", lambda text: None)
    app = dms_app.DmsApp.__new__(dms_app.DmsApp)
    app.sup = _FakeSupervisor(state, camera_fail_active)
    app.frames = 10
    app.last_frame_id = frame_id
    app._last_wd_frame = last_wd_frame
    dms_app.DmsApp._pet_watchdog(app)
    return bool(pinged), app._last_wd_frame


def test_watchdog_is_pinged_only_while_frames_advance(monkeypatch):
    """OPS-07: the old build pinged for ever from the camera-failure loop.

    A unit whose camera is dead must go SILENT so ``WatchdogSec`` restarts it.
    """
    from dms.runtime import CaptureState

    ok, last = _watchdog_probe(CaptureState.RUNNING, False, 42, 41, monkeypatch)
    assert ok and last == 42

    # same frame id twice: the pipeline is stalled, do not claim liveness
    stalled, _ = _watchdog_probe(CaptureState.RUNNING, False, 42, 42, monkeypatch)
    assert not stalled

    # camera failing, even with a fresh frame id
    failing, _ = _watchdog_probe(CaptureState.RUNNING, True, 43, 42, monkeypatch)
    assert not failing

    # the ladder is exhausted
    dead, _ = _watchdog_probe(CaptureState.FAILED, False, 44, 43, monkeypatch)
    assert not dead


# ------------------------------------------------------------ golden fixtures


def test_golden_directory_exists():
    assert os.path.isdir(GOLDEN_DIR), "tests/replay/golden/ is the Phase 1 exit criterion"
    assert os.path.isfile(os.path.join(GOLDEN_DIR, "run_golden.py"))


@pytest.mark.skipif(not CLIPS, reason="no replay clips checked out")
def test_every_clip_has_a_golden():
    missing = []
    for clip in CLIPS:
        stem = os.path.splitext(os.path.basename(clip))[0]
        if not os.path.isfile(os.path.join(GOLDEN_DIR, "%s.json" % stem)):
            missing.append(stem)
    assert not missing, "clips with no golden: %s (run run_golden.py --record)" % missing


GOLDENS = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.json")))


def _run_golden_module():
    """Import ``tests/replay/golden/run_golden.py`` as a module."""
    sys.path.insert(0, GOLDEN_DIR)
    try:
        import run_golden

        return run_golden
    finally:
        sys.path.pop(0)


#: ``[(stem, config, max_frames), ...]`` - the clip set the harness checks.
GOLDEN_CLIPS = _run_golden_module().CLIPS


@pytest.mark.skipif(not GOLDENS, reason="no goldens recorded")
@pytest.mark.parametrize("path", GOLDENS, ids=[os.path.basename(p) for p in GOLDENS])
def test_golden_is_well_formed(path):
    with open(path) as f:
        g = json.load(f)
    assert g["schema"] == "dms.replay_golden/3"
    assert os.path.isfile(os.path.join(ROOT, g["clip"])), g["clip"]
    assert os.path.isfile(os.path.join(ROOT, g["config"])), g["config"]
    # Schema 3: a golden that does not say which detector produced it cannot be
    # checked against the profile, which is the whole point of the field.
    engine = g.get("detector_engine")
    assert engine, "golden has no detector_engine (re-record it)"
    assert os.path.isfile(os.path.join(ROOT, engine)), engine
    assert g["frames"] > 0
    assert g["frames"] <= g["max_frames"]
    lo, hi = g["face_hits"]
    assert 0 <= lo <= hi
    for atype, edges in g["alerts"].items():
        for edge, bounds in edges.items():
            assert edge in ("enter", "escalate", "exit"), (atype, edge)
            assert len(bounds) == 2 and 0 <= bounds[0] <= bounds[1], (atype, edge, bounds)


@pytest.mark.skipif(not GOLDENS, reason="no goldens recorded")
@pytest.mark.parametrize("path", GOLDENS, ids=[os.path.basename(p) for p in GOLDENS])
def test_golden_alert_types_are_real(path):
    from dms.state.alerts import AlertType

    known = set(t.value for t in AlertType)
    with open(path) as f:
        g = json.load(f)
    for atype in g["alerts"]:
        assert atype in known, atype


@pytest.mark.skipif(not GOLDENS, reason="no goldens recorded")
def test_goldens_record_the_uncalibrated_posture():
    """No profile is calibrated, so no gaze-derived alert may appear in a golden.

    If one does, either a calibration leaked into the repo or the fusion gate
    stopped working - both are regressions worth failing on.
    """
    offenders = []
    for path in GOLDENS:
        with open(path) as f:
            g = json.load(f)
        for atype in g["alerts"]:
            if atype in ("gaze_away", "distraction"):
                offenders.append((os.path.basename(path), atype))
    assert not offenders, (
        "gaze-derived alerts in a golden recorded from an uncalibrated profile: %s" % offenders
    )


@pytest.mark.skipif(not GOLDENS, reason="no goldens recorded")
@pytest.mark.parametrize("path", GOLDENS, ids=[os.path.basename(p) for p in GOLDENS])
def test_golden_detector_matches_the_profile_today(path):
    """A golden is a baseline for ONE detector. Swapping it invalidates the band.

    This is the test the SCRFD-512 regression walked past.  On 2026-09-13 all
    twelve profiles were repointed from ``scrfd_500m_640_fp16_gpu.engine`` to
    ``scrfd_500m_512_fp16_gpu.engine`` to buy latency.  Measured consequence on
    ``day_driver``: driver-ROI face-hit frames 1105 -> 930 (-15.8%) and
    ``driver_changed`` enter/exit 2 -> 10 against a golden band of [1, 3].
    Three of the nine goldens went red - and the pytest suite stayed green,
    because nothing in it ran the comparison.

    Needs no GPU, no clip and no engine file loaded: it reads the golden and the
    profile.  So the swap is red on a laptop, in CI, and on the board, within a
    second of being made.  If the new detector is genuinely wanted, re-record
    the goldens deliberately (``run_golden.py --record``) and read the diff.
    """
    with open(path) as f:
        g = json.load(f)
    recorded = g.get("detector_engine")
    if not recorded:
        pytest.fail("golden %s has no detector_engine" % os.path.basename(path))
    cfg = load_config(os.path.join(ROOT, g["config"]))
    assert cfg.models.face.engine == recorded, (
        "%s was recorded against %s but %s now selects %s. Either put the "
        "detector back or re-record the golden on purpose - the recorded alert "
        "bands and face-hit band do not carry across a detector swap."
        % (os.path.basename(path), recorded, g["config"], cfg.models.face.engine)
    )


def test_golden_compare_rejects_a_detector_swap():
    """The harness's own guard, checked without running anything."""
    run_golden = _run_golden_module()
    bad = run_golden.compare(
        "x",
        {
            "frames": 10,
            "face_hits": [0, 5],
            "alerts": {},
            "detector_engine": "engines/scrfd_500m_640_fp16_gpu.engine",
        },
        {
            "frames": 10,
            "face_hits": 3,
            "alerts": {},
            "detector_engine": "engines/scrfd_500m_512_fp16_gpu.engine",
        },
    )
    assert any("now selects" in b for b in bad), bad


def test_golden_band_helper():
    sys.path.insert(0, GOLDEN_DIR)
    try:
        import run_golden
    finally:
        sys.path.pop(0)
    assert run_golden.band(0) == [0, 1]
    assert run_golden.band(4) == [3, 5]
    assert run_golden.band(100) == [75, 125]
    assert run_golden.in_band(3, [3, 5])
    assert not run_golden.in_band(6, [3, 5])
    # A new alert type that is not in the golden is always a regression.
    bad = run_golden.compare(
        "x",
        {"frames": 10, "face_hits": [0, 5], "alerts": {}},
        {"frames": 10, "face_hits": 3, "alerts": {"microsleep": {"enter": 1}}},
    )
    assert any("NEW alert" in b for b in bad)


# ------------------------------------------------------------- board / replay


@BOARD
@NEEDS_BOARD
def test_capture_only_smoke():
    """The process starts, serves health and stops cleanly with no engines."""
    rc = _run_app(["--config", "configs/testsrc.yaml", "--no-detect", "--max-frames", "20"])
    assert rc == 0


@BOARD
@NEEDS_BOARD
def test_full_pipeline_smoke(tmp_path):
    events = str(tmp_path / "events.jsonl")
    rc = _run_app(
        [
            "--config",
            "configs/default.yaml",
            "--max-frames",
            "60",
            "--events",
            events,
        ]
    )
    assert rc == 0
    assert os.path.isfile(events)


@BOARD
@NEEDS_BOARD
@pytest.mark.parametrize("stem", [c[0] for c in GOLDEN_CLIPS])
def test_golden_replay(stem):
    """Replay one clip and compare every count against its recorded band.

    THE detection-quality gate.  Before this existed only ``day_driver`` was
    checked from pytest, and only if somebody thought to set the env var; the
    other eight clips were checked by hand or not at all, which is how three
    goldens sat red through a fix round.  One test per clip so a failure names
    the clip, and so a missing clip skips instead of failing the set.

    Each clip is a ~30 s process.  Hold the GPU mutex around the whole run:

        flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. \\
            ./.venv/bin/python -m pytest -q tests/test_app_integration.py"
    """
    golden = os.path.join(GOLDEN_DIR, "%s.json" % stem)
    clip = os.path.join(ROOT, "tests", "replay", "%s.mp4" % stem)
    if not os.path.isfile(clip):
        pytest.skip("clip %s not checked out" % os.path.basename(clip))
    if not os.path.isfile(golden):
        pytest.skip("no golden for %s (run run_golden.py --record)" % stem)
    runner = os.path.join(GOLDEN_DIR, "run_golden.py")
    proc = subprocess.run(
        [sys.executable, runner, "--only", stem],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, out


def _run_app(argv):
    proc = subprocess.run(
        [sys.executable, "-m", "dms.app"] + argv,
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout.decode("utf-8", "replace"))
    return proc.returncode
