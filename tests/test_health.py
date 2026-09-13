"""Health endpoint status codes, exposure controls and board sensor selection."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from dms.io import metrics
from dms.io.health import HealthServer, HealthState, _BoundedServer, is_loopback
from dms.io.thermal import (
    disk_free_mb,
    gpu_clock_mhz,
    list_zones,
    ram_used_mb,
    read_temps,
    select_zones,
    thermal_throttle,
    zone_report,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


def _get(port, path, method="GET", headers=None):
    """Return (status, body). An HTTPError is a status too, not an exception here."""
    req = urllib.request.Request("http://127.0.0.1:%s%s" % (port, path), method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.getcode(), r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


@pytest.fixture
def running():
    """A started HealthServer on an ephemeral port, with its state."""
    servers = []

    def _start(state=None, **kwargs):
        st = state if state is not None else HealthState()
        srv = HealthServer(st, bind="127.0.0.1", port=0, **kwargs)
        port = srv.start()
        assert port is not None
        servers.append(srv)
        return st, port

    yield _start
    for srv in servers:
        srv.stop()


def _healthy_state():
    state = HealthState()
    state.source = "file"
    state.dev_replay = True
    state.camera = "ok"
    state.fps = 12.5
    state.frame_id = 42
    state.engines = {"face": "gpu", "landmarks": "gpu"}
    state.latency_ms = {"e2e_p95": 80.0}
    state.mark_update()
    return state


# ------------------------------------------------------------------ happy path


def test_healthz_and_metrics(running):
    state, port = running(_healthy_state())
    code, body = _get(port, "/healthz")
    assert code == 200
    doc = json.loads(body)
    assert doc["ok"] is True
    assert doc["source"] == "file"
    assert doc["dev_replay"] is True
    assert doc["camera"] == "ok"
    assert doc["frame_id"] == 42
    assert doc["stale"] is False
    assert doc["uptime_s"] >= 0.0

    code, text = _get(port, "/metrics")
    assert code == 200
    assert "dms_fps 12.5" in text
    assert 'dms_stage_latency_ms{stage="e2e",quantile="p95"} 80' in text
    assert "dms_up 1" in text

    assert _get(port, "/nope")[0] == 404
    assert _get(port, "/health")[0] == 200
    assert _get(port, "/")[0] == 200


def test_head_is_allowed_and_returns_no_body(running):
    _state, port = running(_healthy_state())
    code, body = _get(port, "/healthz", method="HEAD")
    assert code == 200
    assert body == ""


# ------------------------------------------------------- status code semantics


def test_healthz_returns_503_when_not_ok(running):
    """OPS-06: a DMS with a dead camera used to report HTTP 200 to every prober."""
    state, port = running(_healthy_state())
    assert _get(port, "/healthz")[0] == 200
    state.ok = False
    state.camera = "failed"
    code, body = _get(port, "/healthz")
    assert code == 503
    assert json.loads(body)["camera"] == "failed"


def test_degraded_is_still_200_because_the_unit_still_works(running):
    state, port = running(_healthy_state())
    state.degraded = True
    code, body = _get(port, "/healthz")
    assert code == 200
    assert json.loads(body)["degraded"] is True


def test_livez_stays_200_even_when_blind(running):
    state, port = running(_healthy_state())
    state.ok = False
    state.camera = "failed"
    code, body = _get(port, "/livez")
    assert code == 200
    assert json.loads(body)["live"] is True


def test_readyz_distinguishes_booting_blind_and_ready(running):
    state = HealthState()
    state.engines = {"face": "gpu", "landmarks": "missing"}
    _s, port = running(state)

    # still booting: no frames yet
    assert _get(port, "/readyz")[0] == 503

    state.camera = "ok"
    state.frame_id = 10
    state.mark_update()
    # an engine is missing: not ready
    assert _get(port, "/readyz")[0] == 503

    state.engines["landmarks"] = "gpu"
    state.mark_update()
    assert _get(port, "/readyz")[0] == 200

    # camera dies mid-run
    state.camera = "failed"
    state.mark_update()
    assert _get(port, "/readyz")[0] == 503


def test_a_stale_snapshot_is_reported_and_answered_503(running):
    """A hung pipeline used to keep serving a frozen snapshot with a 200."""
    ticks = [1000.0]
    state = HealthState(stale_after_s=5.0, startup_grace_s=0.0, clock=lambda: ticks[0])
    state.camera = "ok"
    state.frame_id = 7
    state.mark_update()
    _s, port = running(state)
    assert _get(port, "/healthz")[0] == 200

    ticks[0] = 1010.0  # pipeline stopped refreshing 10 s ago
    code, body = _get(port, "/healthz")
    doc = json.loads(body)
    assert code == 503
    assert doc["stale"] is True
    assert doc["ok"] is False
    assert doc["snapshot_age_s"] == pytest.approx(10.0, abs=0.5)
    assert _get(port, "/readyz")[0] == 503
    assert _get(port, "/livez")[0] == 200
    assert _get(port, "/metrics")[0] == 200  # a broken unit must stay scrapable


def test_startup_grace_prevents_a_false_stale_before_the_first_frame():
    state = HealthState(stale_after_s=1.0, startup_grace_s=60.0, clock=lambda: 0.0)
    assert state.is_stale() is False
    assert state.is_ready() is False


# ---------------------------------------------------------------- http surface


def test_write_methods_are_rejected(running):
    _s, port = running(_healthy_state())
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        code, _body = _get(port, "/healthz", method=method)
        assert code == 405, method


def test_handler_sets_a_socket_timeout_and_http_1_0():
    from dms.io.health import _handler

    handler = _handler(HealthState(), "")
    assert handler.timeout == 5.0
    assert handler.protocol_version == "HTTP/1.0"


def test_connection_cap_rejects_and_closes():
    """One half-open connection must not be able to pin a thread for ever."""

    class _FakeSock(object):
        def __init__(self):
            self.sent = b""
            self.closed = False

        def sendall(self, data):
            self.sent += data

        def close(self):
            self.closed = True

    srv = _BoundedServer.__new__(_BoundedServer)
    srv._max_connections = 1
    srv._active = 1
    import threading

    srv._count_lock = threading.Lock()
    sock = _FakeSock()
    srv.process_request(sock, ("127.0.0.1", 1234))
    assert b"503" in sock.sent
    assert sock.closed is True
    assert srv._active == 1  # the rejected request never took a slot


# ------------------------------------------------------------------- exposure


def test_is_loopback():
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.1.1")
    assert is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("")
    assert not is_loopback("192.168.1.10")
    assert not is_loopback("::")


def test_non_loopback_bind_is_refused_without_an_explicit_opt_in():
    """/healthz publishes live driver alert state; it must not reach the vehicle bus."""
    srv = HealthServer(HealthState(), bind="0.0.0.0", port=0)
    assert srv.start() is None
    assert "allow_remote" in srv.error
    srv.stop()


def test_required_makes_a_refused_bind_fatal():
    srv = HealthServer(HealthState(), bind="0.0.0.0", port=0, required=True)
    with pytest.raises(RuntimeError):
        srv.start()


def test_bearer_token_is_enforced_on_everything_but_livez(running):
    state = _healthy_state()
    _s, port = running(state, token="s3cret")
    assert _get(port, "/healthz")[0] == 401
    assert _get(port, "/metrics")[0] == 401
    assert _get(port, "/livez")[0] == 200
    assert _get(port, "/healthz", headers={"Authorization": "Bearer s3cret"})[0] == 200
    assert _get(port, "/healthz", headers={"Authorization": "Bearer wrong!"})[0] == 401


def test_a_taken_port_is_reported_not_swallowed(running):
    _s, port = running(_healthy_state())
    clash = HealthServer(HealthState(), bind="127.0.0.1", port=port)
    assert clash.start() is None
    assert "bind" in clash.error
    clash.stop()


# ------------------------------------------------------------- state contents


def test_model_health_and_manifest_reach_the_body_and_metrics(running):
    state = _healthy_state()
    state.set_model("scrfd", loaded=True, ok=True, mean_ms=3.7)
    state.set_model("gaze", loaded=False)
    state.set_model("gaze", ok=False)
    state.manifest = {"ok": True, "error": ""}
    state.privacy = {"record_faces": False, "telemetry": False}
    state.clips = {"enabled": False, "reason": "clips.enabled=false"}
    state.events = {"writable": True, "path": "/var/lib/dms/events.jsonl"}
    state.note("gaze disabled: Gaze360 weights are research-use only")
    state.mark_update()
    _s, port = running(state)

    doc = json.loads(_get(port, "/healthz")[1])
    scrfd = doc["models"]["scrfd"]
    assert scrfd["loaded"] is True
    assert scrfd["consecutive_failures"] == 0
    assert scrfd["mean_ms"] == 3.7
    assert 0.0 <= scrfd["last_ok_age_s"] < 5.0
    assert doc["models"]["gaze"]["loaded"] is False
    assert doc["models"]["gaze"]["consecutive_failures"] == 1
    assert doc["manifest"]["ok"] is True
    assert doc["privacy"] == {"record_faces": False, "telemetry": False}
    assert doc["events"]["writable"] is True
    assert doc["notes"]

    text = _get(port, "/metrics")[1]
    assert 'dms_model_loaded{model="scrfd"} 1' in text
    assert 'dms_model_consecutive_failures{model="gaze"} 1' in text
    assert "dms_manifest_ok 1" in text
    assert "dms_privacy_record_faces 0" in text


def test_disk_headroom_is_reported(running, tmp_path):
    state = _healthy_state()
    state.disk_paths = [str(tmp_path)]
    state.refresh_disk()
    _s, port = running(state)
    doc = json.loads(_get(port, "/healthz")[1])
    assert doc["disk_free_mb"][str(tmp_path)] > 0
    assert 'dms_disk_free_mb{path="%s"}' % tmp_path in _get(port, "/metrics")[1]


def test_build_info_is_served_and_exported(running):
    state = _healthy_state()
    state.set_build(version="1.2.3", git_sha="abc123", trt_version="8.5.2.2", config="prod")
    _s, port = running(state)
    doc = json.loads(_get(port, "/healthz")[1])
    assert doc["build"]["git_sha"] == "abc123"
    assert 'dms_build_info{version="1.2.3",git_sha="abc123"' in _get(port, "/metrics")[1]


def test_health_requests_are_counted(running):
    _s, port = running(_healthy_state())
    _get(port, "/healthz")
    _get(port, "/nope")
    text = _get(port, "/metrics")[1]
    assert 'dms_health_requests_total{route="/healthz",code="200"} 1' in text
    assert 'dms_health_requests_total{route="unknown",code="404"} 1' in text


# ------------------------------------------------------------------- thermal


def _fake_zones(tmp_path, zones):
    root = tmp_path / "thermal"
    root.mkdir()
    for idx, (typ, milli) in enumerate(zones):
        z = root / ("thermal_zone%d" % idx)
        z.mkdir()
        (z / "type").write_text(typ + "\n")
        if milli is not None:
            (z / "temp").write_text("%d\n" % milli)
    return str(root)


def test_zone_selection_on_this_board(tmp_path):
    root = _fake_zones(
        tmp_path,
        [
            ("CPU-therm", 37500),
            ("GPU-therm", 36000),
            ("AUX-therm", 36500),
            ("AO-therm", 36500),
            ("PMIC-Die", 100000),
            ("thermal_fan_est", 0),
            ("iwlwifi_1", 40000),
        ],
    )
    sel = select_zones(root)
    assert sel["gpu"][1] == "GPU-therm"
    assert sel["cpu"][1] == "CPU-therm"
    assert read_temps(root) == (36.0, 37.5)


def test_a_fan_speed_estimate_is_never_mistaken_for_a_die_temperature(tmp_path):
    """OPS-19: `'thermal' in type.lower()` matched thermal_fan_est."""
    root = _fake_zones(
        tmp_path,
        [
            ("thermal_fan_est", 0),
            ("PMIC-Die", 100000),
            ("AO-therm", 40000),
            ("GPU-therm", 55000),
            ("CPU-therm", 60000),
        ],
    )
    gpu, cpu = read_temps(root)
    assert gpu == 55.0
    assert cpu == 60.0


def test_zone_ordering_is_numeric_not_lexicographic(tmp_path):
    """thermal_zone10 must not win over thermal_zone2 in the fallback."""
    zones = [("unknown_zone_%d" % i, 30000 + i * 1000) for i in range(12)]
    zones[0] = ("thermal_fan_est", 0)
    root = _fake_zones(tmp_path, zones)
    names = [n for n, _t, _c in list_zones(root)]
    assert names[:3] == ["thermal_zone0", "thermal_zone1", "thermal_zone2"]
    assert names[10] == "thermal_zone10"
    sel = select_zones(root)
    # zone0 is excluded (fan estimate), so the fallback is zone1, not zone10.
    assert sel["cpu"][0] == "thermal_zone1"


def test_no_zones_means_no_reading_not_a_safe_looking_constant(tmp_path):
    assert read_temps(str(tmp_path / "absent")) == (None, None)
    assert select_zones(str(tmp_path / "absent")) == {"gpu": None, "cpu": None}


def test_unreadable_temp_node_is_skipped(tmp_path):
    root = _fake_zones(tmp_path, [("GPU-therm", None), ("CPU-therm", 42000)])
    gpu, cpu = read_temps(root)
    assert gpu is None
    assert cpu == 42.0


def test_zone_report_names_the_zone_backing_each_role(tmp_path):
    root = _fake_zones(tmp_path, [("CPU-therm", 37000), ("GPU-therm", 36000)])
    rep = zone_report(root)
    assert rep["gpu"]["zone"] == "thermal_zone1"
    assert rep["cpu"]["type"] == "CPU-therm"
    assert len(rep["all"]) == 2


def test_gpu_clock_is_hz_to_mhz_with_no_magic_threshold(tmp_path):
    root = tmp_path / "devfreq" / "17000000.gv11b"
    root.mkdir(parents=True)
    (root / "cur_freq").write_text("114750000\n")
    assert gpu_clock_mhz(str(tmp_path / "devfreq")) == pytest.approx(114.75)
    (root / "cur_freq").write_text("5000\n")  # not Hz: refuse rather than mislead
    assert gpu_clock_mhz(str(tmp_path / "devfreq")) is None
    assert gpu_clock_mhz(str(tmp_path / "absent")) is None


def test_thermal_throttle_thresholds():
    assert thermal_throttle(80.0, 50.0) is True
    assert thermal_throttle(50.0, 75.0) is True
    assert thermal_throttle(50.0, 50.0) is False
    assert thermal_throttle(None, None) is False


def test_ram_used_mb_positive():
    assert ram_used_mb() > 0


def test_disk_free_mb_walks_up_to_an_existing_ancestor(tmp_path):
    assert disk_free_mb(str(tmp_path)) > 0
    assert disk_free_mb(str(tmp_path / "does" / "not" / "exist")) > 0
    assert disk_free_mb(os.sep) > 0


# --------------------------------------------- staleness without an explicit stamp


def test_an_advancing_frame_id_counts_as_a_refresh(running):
    """Works against a pipeline that only assigns frame_id and never calls mark_update."""
    ticks = [1000.0]
    state = HealthState(stale_after_s=5.0, startup_grace_s=1.0, clock=lambda: ticks[0])
    state.camera = "ok"
    state.engines = {"face": "gpu"}
    _s, port = running(state)

    state.frame_id = 1  # first frame, no mark_update() anywhere
    assert _get(port, "/healthz")[0] == 200
    assert _get(port, "/readyz")[0] == 200

    ticks[0] = 1003.0
    state.frame_id = 40  # still advancing
    assert _get(port, "/healthz")[0] == 200

    ticks[0] = 1020.0  # frame id frozen for 17 s: the pipeline hung
    code, body = _get(port, "/healthz")
    assert code == 503
    assert json.loads(body)["stale"] is True


def test_frame_counters_are_mirrored_into_prometheus_totals(running):
    state = _healthy_state()
    state.frames_total = 300
    state.frames_dropped = 12
    state.camera_restarts = 2
    _s, port = running(state)
    text = _get(port, "/metrics")[1]
    assert "dms_frames_total 300" in text
    assert 'dms_frames_dropped_total{reason="capture"} 12' in text
    assert "dms_camera_restarts_total 2" in text
    # a counter never goes backwards, even if the process resets its own int
    state.frames_total = 5
    text = _get(port, "/metrics")[1]
    assert "dms_frames_total 300" in text
