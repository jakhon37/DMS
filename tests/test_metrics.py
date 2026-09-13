"""Registry semantics and Prometheus exposition format for dms/io/metrics.py."""
from __future__ import annotations

import math

import pytest

from dms.io import metrics
from dms.io.metrics import Counter, Gauge, Registry


@pytest.fixture(autouse=True)
def _clean_default_registry():
    """The default registry is process-global; keep tests from leaking into each other."""
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


def test_counter_starts_at_zero_and_increments():
    reg = Registry()
    c = reg.counter("t_total", "help")
    assert c.value() == 0.0
    c.inc()
    c.inc(2.5)
    assert c.value() == 3.5


def test_counter_refuses_to_decrease():
    reg = Registry()
    c = reg.counter("t_total", "help")
    with pytest.raises(ValueError):
        c.inc(-1)


def test_gauge_set_inc_dec_and_bool():
    reg = Registry()
    g = reg.gauge("g", "help")
    g.set(4)
    g.inc()
    g.dec(2)
    assert g.value() == 3.0
    g.set_bool(True)
    assert g.value() == 1.0
    g.set_bool(0)
    assert g.value() == 0.0


def test_labels_must_match_exactly():
    reg = Registry()
    c = reg.counter("t_total", "help", ("a", "b"))
    c.inc(a="1", b="2")
    with pytest.raises(ValueError):
        c.inc(a="1")
    with pytest.raises(ValueError):
        c.inc(a="1", b="2", c="3")


def test_illegal_names_are_rejected_at_registration():
    reg = Registry()
    with pytest.raises(ValueError):
        reg.counter("has-a-dash", "help")
    with pytest.raises(ValueError):
        reg.gauge("ok_name", "help", ("bad-label",))
    with pytest.raises(ValueError):
        reg.gauge("ok_name2", "help", ("__reserved",))
    with pytest.raises(ValueError):
        reg.gauge("ok_name3", "help", ("dup", "dup"))


def test_reregistering_the_same_metric_is_idempotent_but_a_collision_raises():
    reg = Registry()
    a = reg.counter("t_total", "help", ("x",))
    b = reg.counter("t_total", "help", ("x",))
    assert a is b
    with pytest.raises(ValueError):
        reg.gauge("t_total", "help", ("x",))
    with pytest.raises(ValueError):
        reg.counter("t_total", "help", ("y",))


def test_render_shape_and_ordering():
    reg = Registry()
    reg.counter("z_total", "zed").inc()
    g = reg.gauge("a_gauge", "ay", ("k",))
    g.set(2, k="one")
    text = reg.render()
    assert text.endswith("\n")
    lines = text.strip().splitlines()
    # families sorted by name: a_gauge before z_total
    assert lines[0] == "# HELP a_gauge ay"
    assert lines[1] == "# TYPE a_gauge gauge"
    assert lines[2] == 'a_gauge{k="one"} 2'
    assert "# TYPE z_total counter" in lines
    assert "z_total 1" in lines


def test_label_values_are_escaped():
    reg = Registry()
    g = reg.gauge("g", "help", ("msg",))
    g.set(1, msg='a"b\\c\nd')
    line = [l for l in reg.render().splitlines() if l.startswith("g{")][0]
    assert line == 'g{msg="a\\"b\\\\c\\nd"} 1'


def test_special_float_values():
    reg = Registry()
    g = reg.gauge("g", "help")
    g.set(float("inf"))
    assert "g +Inf" in reg.render()
    g.set(float("-inf"))
    assert "g -Inf" in reg.render()
    g.set(float("nan"))
    assert "g NaN" in reg.render()
    g.set(0.5)
    assert "g 0.5" in reg.render()


def test_help_text_never_breaks_the_line_format():
    reg = Registry()
    reg.gauge("g", "line one\nline two").set(1)
    text = reg.render()
    assert "# HELP g line one line two" in text
    assert len([l for l in text.splitlines() if l.startswith("# HELP")]) == 1


class _FakeEvent(object):
    def __init__(self, type_, severity, edge):
        self.type = type_
        self.severity = severity
        self.edge = edge


def test_record_alert_counts_enter_and_exit_separately():
    metrics.record_alert(_FakeEvent("microsleep", "critical", "enter"))
    metrics.record_alert(_FakeEvent("microsleep", "critical", "enter"))
    metrics.record_alert(_FakeEvent("microsleep", "critical", "exit"))
    assert metrics.ALERTS_ENTER_TOTAL.value(type="microsleep", severity="critical") == 2.0
    assert metrics.ALERTS_EXIT_TOTAL.value(type="microsleep", severity="critical") == 1.0


def test_record_alert_accepts_real_alert_enums():
    from dms.state.alerts import AlertEvent, AlertType, Severity

    metrics.record_alert(
        AlertEvent(t_mono_s=1.0, type=AlertType.PERCLOS_HIGH, severity=Severity.WARN, edge="enter")
    )
    assert metrics.ALERTS_ENTER_TOTAL.value(type="perclos_high", severity="warn") == 1.0


def test_build_info_always_carries_every_label():
    labels = metrics.set_build_info(version="", git_sha="", trt_version="", config="")
    assert set(labels) == {"version", "git_sha", "trt_version", "config"}
    assert labels["git_sha"] == "unknown"
    # version falls back to dms.__version__ rather than 'unknown'
    assert labels["version"] not in ("", None)
    assert metrics.BUILD_INFO.value(**labels) == 1.0


def test_build_info_reads_the_release_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DMS_GIT_SHA", "deadbeefcafe")
    labels = metrics.set_build_info(version="9.9.9", trt_version="8.5.2.2", config="prod")
    assert labels["git_sha"] == "deadbeefcafe"
    assert 'git_sha="deadbeefcafe"' in metrics.render()


def test_default_registry_exposes_the_series_a_fleet_alerts_on():
    text = metrics.render()
    for name in (
        "dms_up",
        "dms_uptime_seconds",
        "dms_frames_total",
        "dms_frames_dropped_total",
        "dms_alerts_enter_total",
        "dms_engine_errors_total",
        "dms_event_write_failures_total",
        "dms_events_writable",
        "dms_camera_restarts_total",
        "dms_disk_full",
        "dms_clips_enabled",
        "dms_manifest_ok",
        "dms_thermal_throttle_seconds_total",
        "dms_watchdog_pings_total",
        "dms_privacy_record_faces",
    ):
        assert "# TYPE %s " % name in text, name


def test_every_series_has_a_help_line():
    text = metrics.render()
    types = {l.split()[2] for l in text.splitlines() if l.startswith("# TYPE ")}
    helps = {l.split()[2] for l in text.splitlines() if l.startswith("# HELP ")}
    assert types == helps


def test_counters_use_the_total_suffix():
    for name in metrics.REGISTRY.names():
        metric = metrics.REGISTRY.get(name)
        if isinstance(metric, Counter):
            assert name.endswith("_total"), name
        else:
            assert isinstance(metric, Gauge)


def test_reset_zeroes_without_unregistering():
    metrics.EVENT_WRITES_TOTAL.inc(5)
    assert metrics.EVENT_WRITES_TOTAL.value() == 5.0
    metrics.REGISTRY.reset()
    assert metrics.EVENT_WRITES_TOTAL.value() == 0.0
    assert "dms_event_writes_total" in metrics.REGISTRY.names()


def test_render_is_thread_safe_under_concurrent_writers():
    import threading

    reg = Registry()
    c = reg.counter("t_total", "help", ("i",))
    errors = []

    def worker(idx):
        try:
            for _ in range(200):
                c.inc(i=str(idx))
                reg.render()
        except Exception as exc:  # pragma: no cover - only on a locking bug
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for i in range(4):
        assert c.value(i=str(i)) == 200.0


def test_fmt_keeps_integers_integral():
    assert metrics._fmt(3.0) == "3"
    assert metrics._fmt(3.25) == "3.25"
    assert not math.isnan(float(metrics._fmt(1)))
