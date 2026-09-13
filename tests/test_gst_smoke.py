"""GStreamer smoke tests.

The live-pipeline tests below touch nvvidconv (hardware) and must therefore be run
under the shared board mutex:

    flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_gst_smoke.py -q"

They skip cleanly when gi is unavailable or when DMS_SKIP_GST=1.
"""
from __future__ import annotations

import os
import time

import pytest

from dms.capture.pipelines import build_pipeline
from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

gst_enabled = pytest.mark.skipif(
    os.environ.get("DMS_SKIP_GST") == "1", reason="explicit skip"
)


def _test_cfg():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    cfg.source.type = "test"
    cfg.source.width = 640
    cfg.source.height = 360
    cfg.source.fps = 30
    cfg.capture.queue_depth = 2
    return cfg


@gst_enabled
def test_parse_videotestsrc_pipeline():
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    cfg.source.type = "test"
    desc = build_pipeline(cfg)
    pipeline = Gst.parse_launch(desc)
    assert pipeline is not None
    pipeline.set_state(Gst.State.NULL)


def _read_n(src, n, budget_s=20.0):
    frames = []
    deadline = time.monotonic() + budget_s
    while len(frames) < n and time.monotonic() < deadline:
        f = src.read(timeout_s=1.0)
        if f is not None and f.ok:
            frames.append(f)
    return frames


@gst_enabled
def test_videotestsrc_delivers_frames_with_pts_and_contiguous_bgr():
    pytest.importorskip("gi")
    from dms.capture.gst_source import GstSource, frame_bgr, frame_pts_ns

    cfg = _test_cfg()
    src = GstSource(cfg)
    try:
        src.start()
    except Exception as exc:  # no nvvidconv / no display stack in this environment
        pytest.skip("videotestsrc pipeline would not start: %s" % exc)
    try:
        frames = _read_n(src, 5)
        assert len(frames) == 5, "only got %d frames" % len(frames)
        f = frames[-1]
        assert f.full_bgra.shape == (cfg.source.height, cfg.source.width, 4)
        assert f.full_bgra.dtype.name == "uint8"
        bgr = frame_bgr(f)
        assert bgr.shape == (cfg.source.height, cfg.source.width, 3)
        assert bgr.flags["C_CONTIGUOUS"]
        # PTS must advance across frames and must not be the arrival time.
        pts = [frame_pts_ns(x) for x in frames]
        assert pts == sorted(pts)
        assert pts[-1] > pts[0]
        assert [x.frame_id for x in frames] == sorted(x.frame_id for x in frames)
        assert src.error is False
        assert src.eos is False
    finally:
        src.stop()


@gst_enabled
def test_restart_returns_the_pipeline_to_null_and_rebuilds():
    pytest.importorskip("gi")
    from dms.capture.gst_source import GstSource

    cfg = _test_cfg()
    src = GstSource(cfg)
    try:
        src.start()
    except Exception as exc:
        pytest.skip("videotestsrc pipeline would not start: %s" % exc)
    try:
        assert len(_read_n(src, 2)) == 2
        src.restart()
        assert src.starts == 2
        assert src.error is False
        assert src.eos is False
        frames = _read_n(src, 3)
        assert len(frames) == 3, "capture did not resume after restart"
        # The rebuilt pipeline numbers from 1 again; the supervisor is what makes
        # the ids the application sees monotonic.
        assert frames[0].frame_id >= 1
    finally:
        src.stop()


@gst_enabled
def test_stop_then_start_works_on_the_same_instance():
    pytest.importorskip("gi")
    from dms.capture.gst_source import GstSource

    cfg = _test_cfg()
    src = GstSource(cfg)
    try:
        src.start()
    except Exception as exc:
        pytest.skip("videotestsrc pipeline would not start: %s" % exc)
    assert len(_read_n(src, 1)) == 1
    src.stop()
    # The old code left _stop set, so the second start() spawned a thread that
    # exited immediately and no frame ever arrived again.
    src.start()
    try:
        assert len(_read_n(src, 2)) == 2
    finally:
        src.stop()


@gst_enabled
def test_supervisor_drives_a_real_source_end_to_end():
    pytest.importorskip("gi")
    from dms.runtime.supervisor import CaptureState, CaptureSupervisor

    cfg = _test_cfg()
    sup = CaptureSupervisor.for_config(cfg, stall_timeout_s=10.0)
    try:
        if not sup.start():
            pytest.skip("videotestsrc pipeline would not start")
        got = 0
        deadline = time.monotonic() + 20.0
        while got < 5 and time.monotonic() < deadline:
            f = sup.read(timeout_s=1.0)
            if f is not None and f.ok:
                got += 1
        assert got == 5
        assert sup.state is CaptureState.RUNNING
        assert sup.snapshot()["camera"] == "ok"
        assert sup.take_events() == []
    finally:
        sup.stop()


@gst_enabled
def test_supervisor_recovers_when_the_capture_thread_is_killed():
    """Fault injection: tear the live pipeline out from under the supervisor."""
    pytest.importorskip("gi")
    from dms.runtime.supervisor import CaptureState, CaptureSupervisor

    cfg = _test_cfg()
    sup = CaptureSupervisor.for_config(cfg, stall_timeout_s=3.0)
    try:
        if not sup.start():
            pytest.skip("videotestsrc pipeline would not start")
        assert sup.read(timeout_s=2.0) is not None
        # Simulate the DMSRT-01 failure: the capture thread dies without notice.
        src = sup._source
        src._stop.set()
        src._thread.join(timeout=5.0)
        src.last_error = "injected capture thread death"
        src._error.set()

        recovered = False
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            f = sup.read(timeout_s=1.0)
            if f is not None and f.ok and sup.restarts >= 1:
                recovered = True
                break
        assert recovered, "supervisor never recovered (state=%s err=%s)" % (
            sup.state,
            sup.last_error,
        )
        assert sup.state is CaptureState.RUNNING
        assert sup.restarts >= 1
        edges = [(e.kind, e.edge) for e in sup.take_events()]
        assert ("camera_fail", "enter") in edges
        assert ("camera_fail", "exit") in edges
    finally:
        sup.stop()
