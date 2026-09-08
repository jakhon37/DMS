from __future__ import annotations

import json
import urllib.error
import urllib.request

from dms.io.health import HealthServer, HealthState
from dms.io.thermal import ram_used_mb, thermal_throttle


def test_healthz_and_metrics():
    state = HealthState()
    state.source = "file"
    state.dev_replay = True
    state.camera = "ok"
    state.fps = 12.5
    state.frame_id = 42
    state.latency_ms = {"e2e_p95": 80.0}
    srv = HealthServer(state, bind="127.0.0.1", port=0)
    port = srv.start()
    assert port is not None
    try:
        with urllib.request.urlopen("http://127.0.0.1:%s/healthz" % port, timeout=2) as r:
            body = json.loads(r.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["source"] == "file"
        assert body["dev_replay"] is True
        assert body["camera"] == "ok"
        assert body["frame_id"] == 42
        with urllib.request.urlopen("http://127.0.0.1:%s/metrics" % port, timeout=2) as r:
            text = r.read().decode("utf-8")
        assert "dms_fps" in text
        assert "dms_stage_latency_ms" in text
        try:
            urllib.request.urlopen("http://127.0.0.1:%s/nope" % port, timeout=2)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        srv.stop()


def test_thermal_throttle_thresholds():
    assert thermal_throttle(80.0, 50.0) is True
    assert thermal_throttle(50.0, 75.0) is True
    assert thermal_throttle(50.0, 50.0) is False
    assert thermal_throttle(None, None) is False


def test_ram_used_mb_positive():
    assert ram_used_mb() > 0
