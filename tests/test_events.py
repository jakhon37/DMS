"""Durability, rotation, ENOSPC survival and privacy gating for dms/io/events.py.

Includes the fault injection the readiness plan asks for: a full disk, a corrupt
events file, and a missing state directory.
"""
from __future__ import annotations

import errno
import json
import os

import pytest

from dms.io import metrics
from dms.io.events import (
    SCHEMA_VERSION,
    EventLog,
    default_events_path,
    default_fallback_path,
    read_events,
)
from dms.state.alerts import AlertEvent, AlertType, Severity


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


def _ev(t=1.0, type_=AlertType.FACE_LOST, sev=Severity.CRITICAL, edge="enter", **extra):
    return AlertEvent(t_mono_s=t, type=type_, severity=sev, edge=edge, extra=extra)


def _lines(path):
    with open(path) as f:
        return [json.loads(l) for l in f.read().strip().splitlines() if l.strip()]


class _FullFile(object):
    """A file object whose write always fails with a chosen errno."""

    def __init__(self, code=errno.ENOSPC):
        self.code = code
        self.closed = False

    def write(self, _data):
        raise OSError(self.code, os.strerror(self.code))

    def flush(self):
        pass

    def tell(self):
        return 0

    def fileno(self):
        return 0

    def close(self):
        self.closed = True


# --------------------------------------------------------------------- paths


def test_default_events_path_lab_vs_prod():
    assert default_events_path(True, "") == os.path.join("data", "events.jsonl")
    assert default_events_path(False, "") == "/var/lib/dms/events.jsonl"
    assert default_events_path(True, "/tmp/x.jsonl") == "/tmp/x.jsonl"


def test_production_fallback_is_run_not_the_application_directory():
    """OPS-05: falling back into the app dir puts alerts where logrotate never looks."""
    assert default_fallback_path(False) == "/run/dms/events.jsonl"
    assert default_fallback_path(True) == os.path.join("data", "events.jsonl")


def test_event_log_falls_back_when_unwritable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = EventLog("/proc/dms-nope/events.jsonl", dev=True)
    try:
        assert log.path.endswith(os.path.join("data", "events.jsonl"))
        assert log.fallback is True
        assert log.stats()["fallback"] is True
    finally:
        log.close()
    assert os.path.isfile(os.path.join(str(tmp_path), "data", "events.jsonl"))


def test_strict_mode_fails_closed_instead_of_relocating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError):
        EventLog("/proc/dms-nope/events.jsonl", strict=True)


def test_missing_state_directory_is_created_with_a_tight_mode(tmp_path):
    target = tmp_path / "deep" / "nested" / "events.jsonl"
    log = EventLog(str(target))
    try:
        assert log.write_alert(_ev()) is True
    finally:
        log.close()
    assert target.is_file()
    assert oct(os.stat(str(target)).st_mode & 0o777) == oct(0o640)
    assert oct(os.stat(str(target.parent)).st_mode & 0o777) == oct(0o750)


# --------------------------------------------------------------------- records


def test_event_log_appends_across_reopen(tmp_path):
    p = str(tmp_path / "events.jsonl")
    with EventLog(p) as log:
        log.write_alert(_ev(1.0, edge="enter"))
    with EventLog(p) as log2:
        log2.write_alert(_ev(2.0, edge="exit"))
    recs = _lines(p)
    assert len(recs) == 2
    assert recs[0]["edge"] == "enter" and recs[1]["edge"] == "exit"
    assert recs[0]["type"] == "face_lost"


def test_record_carries_schema_version_both_clocks_boot_id_and_sequence(tmp_path):
    p = str(tmp_path / "events.jsonl")
    with EventLog(p) as log:
        log.write_alert(_ev(t=123.5, dur_s=1.8))
        log.write_alert(_ev(t=124.0))
    a, b = _lines(p)
    assert a["v"] == SCHEMA_VERSION
    assert a["kind"] == "alert"
    assert a["t"] == 123.5  # monotonic, as the state machine timed it
    assert a["ts"] > 1_500_000_000  # wall clock, for CAN/journald correlation
    assert a["boot"] and isinstance(a["boot"], str)
    assert a["seq"] == 1 and b["seq"] == 2
    assert a["extra"] == {"dur_s": 1.8}
    assert a["track_id"] is None


def test_alert_writes_are_counted_in_metrics(tmp_path):
    p = str(tmp_path / "events.jsonl")
    with EventLog(p) as log:
        log.write_alert(_ev(sev=Severity.WARN, type_=AlertType.YAWN, edge="enter"))
    assert metrics.ALERTS_ENTER_TOTAL.value(type="yawn", severity="warn") == 1.0
    assert metrics.EVENT_WRITES_TOTAL.value() == 1.0


# ------------------------------------------------------------------ durability


def test_critical_events_are_fsynced_immediately(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, fsync="critical", fsync_interval_s=3600.0)
    try:
        log.write_alert(_ev(sev=Severity.WARN))
        warn_syncs = len(calls)
        log.write_alert(_ev(sev=Severity.CRITICAL))
        assert len(calls) == warn_syncs + 1
    finally:
        monkeypatch.undo()
        log.close()


def test_fsync_never_policy_still_flushes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, fsync="never")
    try:
        log.write_alert(_ev(sev=Severity.CRITICAL))
        assert calls == []
    finally:
        monkeypatch.undo()
        log.close()
    assert len(_lines(p)) == 1


def test_unknown_fsync_policy_is_a_configuration_error(tmp_path):
    with pytest.raises(ValueError):
        EventLog(str(tmp_path / "e.jsonl"), fsync="sometimes")


# -------------------------------------------------------------------- rotation


def test_in_process_size_rotation(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, max_bytes=400, backups=2, fsync="never")
    try:
        for i in range(20):
            log.write_alert(_ev(t=float(i)))
    finally:
        log.close()
    assert os.path.isfile(p)
    assert os.path.isfile(p + ".1")
    assert os.path.getsize(p) <= 400 + 300
    assert not os.path.isfile(p + ".3")  # backups=2
    assert metrics.EVENT_ROTATIONS_TOTAL.value(reason="size") > 0


def test_external_rotation_is_detected_and_the_file_reopened(tmp_path):
    """logrotate `create` renames the file away; we must not keep writing to the old inode."""
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, stat_interval_s=0.0, fsync="never")
    try:
        log.write_alert(_ev(t=1.0))
        os.rename(p, p + ".1")
        log.write_alert(_ev(t=2.0))
    finally:
        log.close()
    assert len(_lines(p)) == 1, "the second record must land in the new file"
    assert len(_lines(p + ".1")) == 1
    assert metrics.EVENT_ROTATIONS_TOTAL.value(reason="external") >= 1


def test_reopen_is_the_sighup_entry_point(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, stat_interval_s=3600.0, fsync="never")
    try:
        log.write_alert(_ev(t=1.0))
        os.rename(p, p + ".1")
        assert log.reopen() is True
        log.write_alert(_ev(t=2.0))
    finally:
        log.close()
    assert len(_lines(p)) == 1


# ------------------------------------------------------------------- disk full


def test_enospc_does_not_raise_emits_disk_full_and_keeps_the_process_alive(tmp_path):
    seen = []
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p, on_disk_full=seen.append, retry_s=3600.0)
    try:
        log._fh = _FullFile(errno.ENOSPC)
        assert log.write_alert(_ev()) is False  # no exception
        assert log.disk_full is True
        assert log.stats()["writable"] is False
        assert [e.type for e in seen] == [AlertType.DISK_FULL]
        assert seen[0].edge == "enter"
        # subsequent alerts are dropped, still without raising
        assert log.write_alert(_ev(t=2.0)) is False
        assert metrics.DISK_FULL.value() == 1.0
        assert metrics.EVENT_WRITE_FAILURES_TOTAL.value(errno="ENOSPC") == 1.0
        assert metrics.EVENT_DROPPED_TOTAL.value(reason="disk_full") >= 1.0
    finally:
        log._fh = None
        log.close()


def test_edquot_is_treated_as_a_full_disk(tmp_path):
    seen = []
    log = EventLog(str(tmp_path / "e.jsonl"), on_disk_full=seen.append, retry_s=3600.0)
    try:
        log._fh = _FullFile(errno.EDQUOT)
        assert log.write_alert(_ev()) is False
        assert log.disk_full is True
        assert seen and seen[0].type is AlertType.DISK_FULL
    finally:
        log._fh = None
        log.close()


def test_disk_full_recovers_and_records_the_gap(tmp_path):
    """When space returns, the JSONL must say why it has a hole in it."""
    seen = []
    p = str(tmp_path / "events.jsonl")
    ticks = [0.0]
    log = EventLog(p, on_disk_full=seen.append, retry_s=5.0, clock=lambda: ticks[0])
    try:
        good = log._fh
        log._fh = _FullFile()
        assert log.write_alert(_ev()) is False
        assert log.write_alert(_ev()) is False  # dropped, inside the retry window
        # disk gets space back, and the retry deadline passes
        log._fh = good
        ticks[0] = 10.0
        assert log.write_alert(_ev(t=99.0, type_=AlertType.YAWN, sev=Severity.WARN)) is True
    finally:
        log.close()
    recs = _lines(p)
    gap = [r for r in recs if r["type"] == "disk_full"]
    assert len(gap) == 1
    assert gap[0]["edge"] == "exit"
    assert gap[0]["extra"]["dropped"] == 2.0
    assert gap[0]["extra"]["seconds"] == 10.0
    assert [e.edge for e in seen] == ["enter", "exit"]
    assert metrics.DISK_FULL.value() == 0.0


def test_a_one_megabyte_quota_config_stops_writing_instead_of_filling_the_disk(tmp_path, monkeypatch):
    """Fault injection: min_free_mb is the guard that keeps headroom for the system."""
    import shutil as shutil_mod

    class _Usage(object):
        total = 1024 * 1024
        used = 1024 * 1024
        free = 1 * 1024 * 1024  # 1 MiB free

    monkeypatch.setattr(shutil_mod, "disk_usage", lambda _p: _Usage())
    seen = []
    log = EventLog(str(tmp_path / "e.jsonl"), min_free_mb=64.0, on_disk_full=seen.append,
                   retry_s=3600.0)
    try:
        assert log.write_alert(_ev()) is False
        assert log.disk_full is True
        assert seen and seen[0].type is AlertType.DISK_FULL
    finally:
        log.close()


def test_disk_full_callback_that_raises_cannot_break_the_writer(tmp_path):
    def boom(_ev):
        raise RuntimeError("health callback exploded")

    log = EventLog(str(tmp_path / "e.jsonl"), on_disk_full=boom, retry_s=3600.0)
    try:
        log._fh = _FullFile()
        assert log.write_alert(_ev()) is False  # must not propagate
    finally:
        log._fh = None
        log.close()


def test_non_space_io_errors_are_reported_but_not_disk_full(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    try:
        log._fh = _FullFile(errno.EIO)
        assert log.write_alert(_ev()) is False
        assert log.disk_full is False
        assert log.writable is False
        assert metrics.EVENT_WRITE_FAILURES_TOTAL.value(errno="EIO") == 1.0
    finally:
        log._fh = None
        log.close()


# ------------------------------------------------------------------- lifecycle


def test_close_is_idempotent_and_write_after_close_does_not_raise(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p)
    log.write_alert(_ev())
    log.close()
    log.close()
    assert log.write_alert(_ev(t=2.0)) is False  # was AttributeError before
    assert log.dropped == 1
    assert metrics.EVENT_DROPPED_TOTAL.value(reason="closed") == 1.0


def test_context_manager_closes_on_exception(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p)
    with pytest.raises(ValueError):
        with log:
            log.write_alert(_ev())
            raise ValueError("boom")
    assert log._closed is True
    assert len(_lines(p)) == 1


# --------------------------------------------------------------- corrupt input


def test_a_torn_final_line_is_terminated_not_glued_to_the_next_record(tmp_path):
    """Fault injection: a power cut mid-write leaves a partial line."""
    p = str(tmp_path / "events.jsonl")
    with open(p, "w") as f:
        f.write('{"v":1,"type":"micro')  # no newline: torn by a power cut
    with EventLog(p) as log:
        assert log.write_alert(_ev()) is True
    raw = open(p).read().splitlines()
    assert len(raw) == 2
    assert raw[0] == '{"v":1,"type":"micro'
    assert json.loads(raw[1])["type"] == "face_lost"


def test_read_events_skips_corrupt_lines_by_default(tmp_path):
    p = str(tmp_path / "events.jsonl")
    with open(p, "w") as f:
        f.write('{"v":1,"seq":1}\n')
        f.write("not json at all\n")
        f.write("\n")
        f.write('{"v":1,"seq":2}\n')
    assert [r["seq"] for r in read_events(p)] == [1, 2]
    with pytest.raises(ValueError):
        list(read_events(p, skip_bad=False))


# ---------------------------------------------------------------- privacy gate


def test_snapshots_are_refused_unless_privacy_telemetry_is_true(tmp_path):
    """OPS-10: privacy.telemetry was dead config; this is its enforcement point."""
    p = str(tmp_path / "events.jsonl")
    with EventLog(p, telemetry=False) as log:
        assert log.write_snapshot({"perclos": 0.1}) is False
    assert not os.path.getsize(p)
    assert metrics.EVENT_DROPPED_TOTAL.value(reason="privacy_telemetry") == 1.0

    with EventLog(p, telemetry=True) as log:
        assert log.write_snapshot({"perclos": 0.1}) is True
    rec = _lines(p)[0]
    assert rec["kind"] == "snapshot"
    assert rec["payload"] == {"perclos": 0.1}


def test_from_config_reads_privacy_and_event_keys():
    from dms.config.schema import AppConfig
    from dms.io.events import from_config

    cfg = AppConfig.parse_obj(
        {
            "models": {
                "face": {"engine": "engines/x.engine"},
                "landmarks": {"engine": "engines/y.engine"},
            },
            "events": {"enabled": True, "path": ""},
            "privacy": {"record_faces": False, "telemetry": True},
            "source": {"type": "file", "dev": True},
        }
    )
    log = from_config(cfg)
    try:
        assert log.telemetry is True
        assert log.path.endswith(os.path.join("data", "events.jsonl"))
        assert log.fsync_policy == "critical"
        assert log.max_bytes == 64 * 1024 * 1024
    finally:
        log.close()
