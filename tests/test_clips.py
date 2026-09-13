"""Clip ring: the privacy gate, the quota, the trigger window and ENOSPC survival.

Everything here runs without a GPU and without GStreamer: the encoder and the muxer
are injected, which is exactly why they are injectable.
"""
from __future__ import annotations

import errno
import os
import time

import pytest

from dms.io import metrics
from dms.io.clips import CLIP_EXT, RING_SUBDIR, ClipRing, GstSegmentWriter, dir_size_bytes
from dms.state.alerts import AlertType


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


class FakeWriter(object):
    """Stands in for nvv4l2h264enc + splitmuxsink."""

    def __init__(self, directory, width, height, fps, segment_s, max_files, bitrate):
        self.directory = directory
        self.width = width
        self.height = height
        self.fps = fps
        self.segment_s = segment_s
        self.max_files = max_files
        self.bitrate = bitrate
        self.started = False
        self.stopped = False
        self.pushed = 0
        self.raise_on_push = None

    def start(self):
        self.started = True

    def push(self, frame, pts_ns):
        if self.raise_on_push is not None:
            raise self.raise_on_push
        self.pushed += 1

    def stop(self):
        self.stopped = True


def _ring(tmp_path, **kwargs):
    """A ring with both switches on and a fake encoder/muxer, unless overridden."""
    made = {}

    def factory(**kw):
        made["writer"] = FakeWriter(**kw)
        return made["writer"]

    concat_calls = []

    def concat(ffmpeg, segments, out):
        concat_calls.append((ffmpeg, list(segments), out))
        with open(out, "wb") as f:
            f.write(b"\0" * 1024)

    params = dict(
        enabled=True,
        record_faces=True,
        directory=str(tmp_path / "clips"),
        pre_s=10.0,
        post_s=5.0,
        segment_s=2.0,
        quota_mb=2048,
        retain_h=24,
        writer_factory=factory,
        concat=concat,
    )
    params.update(kwargs)
    ring = ClipRing(**params)
    return ring, made, concat_calls


def _segment(ring, name, age_s=0.0, size=512):
    """Drop a finalised ring segment on disk with a chosen age."""
    os.makedirs(ring.ring_dir, exist_ok=True)
    path = os.path.join(ring.ring_dir, name)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    when = time.time() - age_s
    os.utime(path, (when, when))
    return path


# --------------------------------------------------------------- privacy gate


def test_a_default_install_records_nothing_and_creates_no_directory(tmp_path):
    ring = ClipRing(directory=str(tmp_path / "clips"))
    assert ring.start() is False
    assert ring.active is False
    assert "clips.enabled=false" in ring.reason
    assert "privacy.record_faces=false" in ring.reason
    assert not os.path.exists(str(tmp_path / "clips"))
    assert metrics.CLIPS_ENABLED.value() == 0.0


def test_clips_enabled_alone_is_refused(tmp_path):
    """The design's Phase 2 acceptance: record_faces=false refuses even with clips on."""
    ring, _made, _c = _ring(tmp_path, record_faces=False)
    assert ring.start() is False
    assert ring.reason == "privacy.record_faces=false (refusing to encode in-cabin video)"
    assert not os.path.exists(ring.directory)
    assert ring.push(object()) is False
    assert ring.trigger("microsleep", "critical") is False


def test_record_faces_alone_is_refused(tmp_path):
    ring, _made, _c = _ring(tmp_path, enabled=False)
    assert ring.start() is False
    assert ring.reason == "clips.enabled=false"
    assert not os.path.exists(ring.directory)


def test_both_switches_on_starts_the_ring(tmp_path):
    ring, made, _c = _ring(tmp_path)
    try:
        assert ring.start() is True
        assert ring.reason == ""
        assert made["writer"].started is True
        assert os.path.isdir(ring.ring_dir)
        assert metrics.CLIPS_ENABLED.value() == 1.0
        assert ring.push(b"frame", 1.0) is True
        assert made["writer"].pushed == 1
    finally:
        ring.stop()
    assert made["writer"].stopped is True
    assert metrics.CLIPS_ENABLED.value() == 0.0


def test_closing_the_privacy_gate_at_runtime_stops_the_ring(tmp_path):
    ring, made, _c = _ring(tmp_path)
    try:
        assert ring.start() is True
        ring.record_faces = False
        assert ring.push(b"frame", 1.0) is False
        assert ring.active is False
        assert made["writer"].stopped is True
    finally:
        ring.stop()


def test_snapshot_reports_the_posture_for_healthz(tmp_path):
    ring = ClipRing(directory=str(tmp_path / "clips"))
    ring.start()
    snap = ring.snapshot()
    assert snap["enabled"] is False
    assert snap["configured"] is False
    assert snap["record_faces"] is False
    assert "clips.enabled=false" in snap["reason"]
    assert snap["quota_mb"] == 2048


# -------------------------------------------------------------------- trigger


def test_trigger_writes_a_clip_after_the_post_roll(tmp_path):
    clock = [100.0]
    ring, _made, calls = _ring(tmp_path, clock=lambda: clock[0])
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=4.0)
        _segment(ring, "seg_00001.mp4", age_s=2.0)
        assert ring.trigger("microsleep", "critical") is True
        assert ring.tick() == []  # post-roll has not elapsed
        clock[0] = 106.0
        written = ring.tick()
        assert len(written) == 1
        assert written[0].endswith(CLIP_EXT)
        assert "critical_microsleep" in os.path.basename(written[0])
        assert os.path.isfile(written[0])
        assert len(calls[0][1]) == 2  # both segments were concatenated
        assert metrics.CLIPS_WRITTEN_TOTAL.value(severity="critical") == 1.0
    finally:
        ring.stop()


def test_triggers_inside_one_window_are_coalesced(tmp_path):
    clock = [100.0]
    ring, _made, calls = _ring(tmp_path, clock=lambda: clock[0])
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=1.0)
        ring.trigger("yawn", "warn")
        clock[0] = 102.0
        ring.trigger("microsleep", "critical")
        clock[0] = 110.0
        written = ring.tick()
        assert len(written) == 1
        # the worst severity in the window wins
        assert "critical_yawn" in os.path.basename(written[0])
        assert len(calls) == 1
    finally:
        ring.stop()


def test_info_severity_never_triggers_a_clip(tmp_path):
    ring, _made, _c = _ring(tmp_path)
    try:
        ring.start()
        assert ring.trigger("blink", "info") is False
    finally:
        ring.stop()


def test_trigger_on_an_inactive_ring_is_counted_not_crashed(tmp_path):
    ring = ClipRing(directory=str(tmp_path / "clips"))
    ring.start()
    assert ring.trigger("microsleep", "critical") is False
    assert metrics.CLIPS_DROPPED_TOTAL.value(reason="ring_inactive") == 1.0


def test_no_finalised_segments_drops_the_clip_without_raising(tmp_path):
    clock = [100.0]
    ring, _made, calls = _ring(tmp_path, clock=lambda: clock[0])
    try:
        ring.start()
        ring.trigger("microsleep", "critical")
        clock[0] = 106.0
        assert ring.tick() == []
        assert calls == []
        assert metrics.CLIPS_DROPPED_TOTAL.value(reason="no_segments") == 1.0
        assert ring.active is True  # still recording; only this clip was lost
    finally:
        ring.stop()


def test_reason_is_sanitised_into_the_filename(tmp_path):
    clock = [100.0]
    ring, _made, _c = _ring(tmp_path, clock=lambda: clock[0])
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=1.0)
        ring.trigger("../../etc/passwd", "warn")
        clock[0] = 110.0
        written = ring.tick()
        assert len(written) == 1
        base = os.path.basename(written[0])
        assert os.path.dirname(os.path.abspath(written[0])) == os.path.abspath(ring.directory)
        assert ".." not in base
        assert base.endswith(CLIP_EXT)
    finally:
        ring.stop()


# ---------------------------------------------------------------------- quota


def test_quota_sweep_deletes_oldest_first(tmp_path):
    """Fault injection: a 1 MiB quota in the test config."""
    ring, _made, _c = _ring(tmp_path, quota_mb=1)
    try:
        ring.start()
        for i in range(6):
            path = os.path.join(ring.directory, "clip_%d%s" % (i, CLIP_EXT))
            with open(path, "wb") as f:
                f.write(b"\0" * 300 * 1024)  # 300 KiB each -> 1.8 MiB total
            when = time.time() - (100 - i * 10)
            os.utime(path, (when, when))
        deleted, freed = ring.sweep()
        assert deleted >= 2
        assert freed > 0
        assert dir_size_bytes(ring.directory) <= 1024 * 1024
        # the newest survive, the oldest are gone
        assert os.path.isfile(os.path.join(ring.directory, "clip_5" + CLIP_EXT))
        assert not os.path.isfile(os.path.join(ring.directory, "clip_0" + CLIP_EXT))
        assert metrics.CLIPS_SWEPT_TOTAL.value(reason="quota") >= 2
    finally:
        ring.stop()


def test_retention_sweep_deletes_by_age_even_when_under_quota(tmp_path):
    ring, _made, _c = _ring(tmp_path, retain_h=1, quota_mb=2048)
    try:
        ring.start()
        old = os.path.join(ring.directory, "old" + CLIP_EXT)
        new = os.path.join(ring.directory, "new" + CLIP_EXT)
        for path, age in ((old, 7200), (new, 60)):
            with open(path, "wb") as f:
                f.write(b"\0" * 1024)
            os.utime(path, (time.time() - age, time.time() - age))
        deleted, _freed = ring.sweep()
        assert deleted == 1
        assert not os.path.isfile(old)
        assert os.path.isfile(new)
        assert metrics.CLIPS_SWEPT_TOTAL.value(reason="retention") == 1.0
    finally:
        ring.stop()


def test_the_sweeper_never_deletes_live_ring_segments(tmp_path):
    ring, _made, _c = _ring(tmp_path, quota_mb=1, retain_h=1)
    try:
        ring.start()
        seg = _segment(ring, "seg_00000.mp4", age_s=7200, size=900 * 1024)
        ring.sweep()
        assert os.path.isfile(seg), "splitmuxsink owns the ring; deleting it corrupts a clip"
    finally:
        ring.stop()


def test_quota_bytes_are_published(tmp_path):
    ring, _made, _c = _ring(tmp_path, quota_mb=64)
    assert metrics.CLIPS_QUOTA_BYTES.value() == 64 * 1024 * 1024


# ------------------------------------------------------------------ disk full


def test_enospc_in_the_muxer_emits_disk_full_and_keeps_the_process_alive(tmp_path):
    seen = []
    clock = [100.0]

    def boom(_ffmpeg, _segments, _out):
        raise OSError(errno.ENOSPC, "No space left on device")

    ring, made, _c = _ring(tmp_path, clock=lambda: clock[0], concat=boom, on_disk_full=seen.append)
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=1.0)
        ring.trigger("microsleep", "critical")
        clock[0] = 106.0
        assert ring.tick() == []  # no exception escapes
        assert ring.disk_full is True
        assert ring.active is False
        assert made["writer"].stopped is True
        assert [e.type for e in seen] == [AlertType.DISK_FULL]
        assert metrics.DISK_FULL.value() == 1.0
    finally:
        ring.stop()


def test_ffmpeg_reporting_no_space_on_stderr_is_also_disk_full(tmp_path):
    seen = []
    clock = [100.0]

    def boom(_ffmpeg, _segments, _out):
        raise RuntimeError("ffmpeg concat failed (rc=1): av_interleaved_write_frame(): "
                           "No space left on device")

    ring, _made, _c = _ring(tmp_path, clock=lambda: clock[0], concat=boom, on_disk_full=seen.append)
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=1.0)
        ring.trigger("microsleep", "critical")
        clock[0] = 106.0
        ring.tick()
        assert ring.disk_full is True
        assert seen and seen[0].type is AlertType.DISK_FULL
    finally:
        ring.stop()


def test_a_muxer_error_that_is_not_enospc_keeps_recording(tmp_path):
    clock = [100.0]

    def boom(_ffmpeg, _segments, _out):
        raise RuntimeError("ffmpeg concat failed (rc=1): invalid data")

    ring, _made, _c = _ring(tmp_path, clock=lambda: clock[0], concat=boom)
    try:
        ring.start()
        _segment(ring, "seg_00000.mp4", age_s=1.0)
        ring.trigger("microsleep", "critical")
        clock[0] = 106.0
        ring.tick()
        assert ring.disk_full is False
        assert ring.active is True
        assert metrics.CLIPS_DROPPED_TOTAL.value(reason="mux_error") == 1.0
    finally:
        ring.stop()


def test_low_free_space_refuses_to_start(tmp_path, monkeypatch):
    import shutil as shutil_mod

    class _Usage(object):
        total = 1
        used = 1
        free = 10 * 1024 * 1024  # 10 MiB

    monkeypatch.setattr(shutil_mod, "disk_usage", lambda _p: _Usage())
    ring, _made, _c = _ring(tmp_path, min_free_mb=256.0)
    assert ring.start() is False
    assert "free" in ring.reason
    assert metrics.CLIPS_DROPPED_TOTAL.value(reason="low_disk") == 1.0


def test_enospc_while_pushing_frames_stops_the_ring(tmp_path):
    seen = []
    ring, made, _c = _ring(tmp_path, on_disk_full=seen.append)
    try:
        ring.start()
        made["writer"].raise_on_push = OSError(errno.ENOSPC, "No space left on device")
        assert ring.push(b"frame", 1.0) is False
        assert ring.disk_full is True
        assert seen and seen[0].type is AlertType.DISK_FULL
    finally:
        ring.stop()


def test_an_encoder_that_refuses_to_start_disables_clips_without_raising(tmp_path):
    def factory(**_kw):
        raise RuntimeError("nvv4l2h264enc not available")

    ring = ClipRing(
        enabled=True, record_faces=True, directory=str(tmp_path / "clips"),
        writer_factory=factory, concat=lambda *a: None,
    )
    assert ring.start() is False
    assert "encoder failed" in ring.reason
    assert ring.push(b"x") is False


def test_missing_ffmpeg_disables_clips(tmp_path):
    ring = ClipRing(
        enabled=True, record_faces=True, directory=str(tmp_path / "clips"),
        ffmpeg=str(tmp_path / "no-such-ffmpeg"), writer_factory=FakeWriter,
    )
    assert ring.start() is False
    assert "ffmpeg not found" in ring.reason


# ------------------------------------------------------------------- pipeline


def test_gst_pipeline_description_matches_the_design():
    writer = GstSegmentWriter(
        directory="/var/lib/dms/clips/ring", width=1280, height=720, fps=30,
        segment_s=10.0, max_files=5, bitrate=4000000,
    )
    desc = writer.description()
    assert "appsrc" in desc
    assert "nvvidconv" in desc
    assert "nvv4l2h264enc" in desc
    assert "splitmuxsink" in desc
    assert "max-size-time=10000000000" in desc
    assert "max-files=5" in desc
    assert "iframeinterval=300" in desc
    assert "width=1280,height=720,framerate=30/1" in desc
    assert "/var/lib/dms/clips/ring/seg_%05d.mp4" in desc


def test_ring_holds_enough_segments_for_the_pre_roll():
    ring = ClipRing(pre_s=10.0, post_s=5.0, segment_s=2.0)
    assert ring.max_ring_files >= int((10.0 + 5.0) / 2.0)


def test_from_config_reads_both_switches():
    from dms.config.schema import AppConfig

    cfg = AppConfig.parse_obj(
        {
            "models": {
                "face": {"engine": "engines/x.engine"},
                "landmarks": {"engine": "engines/y.engine"},
            },
            "clips": {"enabled": True, "dir": "/var/lib/dms/clips", "quota_mb": 512},
            "privacy": {"record_faces": False, "telemetry": False},
        }
    )
    ring = ClipRing.from_config(cfg, writer_factory=FakeWriter)
    assert ring.enabled is True
    assert ring.record_faces is False
    assert ring.quota_bytes == 512 * 1024 * 1024
    assert ring.gate_reason().startswith("privacy.record_faces=false")
    assert ring.ring_dir.endswith(os.path.join("clips", RING_SUBDIR))
