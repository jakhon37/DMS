from __future__ import annotations

import json
import os

from dms.io.events import EventLog, default_events_path
from dms.state.alerts import AlertEvent, AlertType, Severity


def test_default_events_path_lab_vs_prod():
    assert default_events_path(True, "") == os.path.join("data", "events.jsonl")
    assert default_events_path(False, "") == "/var/lib/dms/events.jsonl"
    assert default_events_path(True, "/tmp/x.jsonl") == "/tmp/x.jsonl"


def test_event_log_falls_back_when_unwritable(tmp_path, monkeypatch):
    import dms.io.events as evmod

    monkeypatch.chdir(tmp_path)
    log = EventLog("/proc/dms-nope/events.jsonl")
    assert log.path.endswith("data/events.jsonl")
    log.close()
    assert os.path.isfile(os.path.join(str(tmp_path), "data", "events.jsonl"))


def test_event_log_appends(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p)
    log.write_alert(
        AlertEvent(t_mono_s=1.0, type=AlertType.FACE_LOST, severity=Severity.CRITICAL, edge="enter")
    )
    log.close()
    log2 = EventLog(p)
    log2.write_alert(
        AlertEvent(t_mono_s=2.0, type=AlertType.FACE_LOST, severity=Severity.CRITICAL, edge="exit")
    )
    log2.close()
    lines = open(p).read().strip().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["edge"] == "enter" and b["edge"] == "exit"
    assert a["type"] == "face_lost"
