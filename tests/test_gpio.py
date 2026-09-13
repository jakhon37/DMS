"""GPIO critical latch and systemd notification.

The property that matters most here is fail-safe release: the buzzer must never be
left asserted with the DMS process dead.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

from dms.io import metrics
from dms.io.gpio_alert import GpioAlert
from dms.io.sd_notify import (
    notify_ready,
    notify_status,
    notify_stopping,
    notify_watchdog,
    sd_notify,
    under_systemd,
    watchdog_interval_s,
    watchdog_ping_interval_s,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


class FakeGpio(object):
    """Enough of Jetson.GPIO to drive one output pin."""

    BOARD = "BOARD"
    OUT = "OUT"
    HIGH = 1
    LOW = 0

    def __init__(self, fail_on_output=False):
        self.mode = None
        self.setup_calls = []
        self.writes = []
        self.cleaned = []
        self.warnings = None
        self.fail_on_output = fail_on_output

    def setwarnings(self, flag):
        self.warnings = flag

    def setmode(self, mode):
        self.mode = mode

    def setup(self, pin, direction, initial=None):
        self.setup_calls.append((pin, direction, initial))

    def output(self, pin, level):
        if self.fail_on_output:
            raise RuntimeError("gpio bus error")
        self.writes.append((pin, level))

    def cleanup(self, pin):
        self.cleaned.append(pin)


# ------------------------------------------------------------------ disabled


def test_gpio_noop_without_pin():
    g = GpioAlert(None, buzzer=False)
    g.set_critical(True)
    g.set_critical(False)
    g.close()
    assert g.snapshot()["enabled"] is False
    assert g.reason == "no pin"


def test_gpio_noop_buzzer_off():
    g = GpioAlert(7, buzzer=False)
    assert g._gpio is None
    g.set_critical(True)
    g.close()
    assert g.reason == "alerts.buzzer=false"


def test_a_missing_library_disables_the_output_rather_than_failing_to_start():
    class _Boom(object):
        def setwarnings(self, _f):
            raise ImportError("no Jetson.GPIO here")

    g = GpioAlert(7, buzzer=True, gpio=_Boom())
    assert g._gpio is None
    assert g.healthy is False
    assert "init failed" in g.reason
    g.set_critical(True)  # must not raise
    g.close()


# --------------------------------------------------------------------- drive


def test_edges_reach_the_pin_exactly_once():
    fake = FakeGpio()
    g = GpioAlert(7, buzzer=True, gpio=fake)
    try:
        assert fake.mode == "BOARD"
        assert fake.setup_calls == [(7, "OUT", 0)]
        g.set_critical(True)
        g.set_critical(True)  # idempotent
        g.set_critical(False)
        assert fake.writes == [(7, 1), (7, 0)]
        assert g.assert_count == 1
        assert g.active is False
    finally:
        g.close()


def test_active_low_wiring_is_inverted():
    fake = FakeGpio()
    g = GpioAlert(7, buzzer=True, active_high=False, gpio=fake)
    try:
        assert fake.setup_calls == [(7, "OUT", 1)]  # released == HIGH
        g.set_critical(True)
        assert fake.writes == [(7, 0)]
    finally:
        g.close()


def test_a_write_failure_is_reported_not_raised():
    fake = FakeGpio(fail_on_output=True)
    g = GpioAlert(7, buzzer=True, gpio=fake)
    try:
        g.set_critical(True)
        assert g.write_failures == 1
        assert g.healthy is False
        assert "write failed" in g.reason
        assert g.active is False
    finally:
        g.close()


# ----------------------------------------------------------------- fail safe


def test_close_releases_the_pin_and_is_idempotent():
    fake = FakeGpio()
    g = GpioAlert(7, buzzer=True, gpio=fake)
    g.set_critical(True)
    g.close()
    g.close()
    assert fake.writes[-1] == (7, 0), "the latch must be released"
    assert fake.cleaned == [7]


def test_context_manager_releases_the_latch_on_an_exception():
    fake = FakeGpio()
    with pytest.raises(ValueError):
        with GpioAlert(7, buzzer=True, gpio=fake) as g:
            g.set_critical(True)
            raise ValueError("main loop exploded")
    assert fake.writes[-1] == (7, 0)
    assert fake.cleaned == [7]


def test_atexit_drops_the_latch_even_when_the_process_crashes(tmp_path):
    """An unhandled exception in the main loop must not leave the buzzer screaming."""
    script = tmp_path / "crash.py"
    script.write_text(
        "import sys, json\n"
        "sys.path.insert(0, %r)\n" % ROOT
        + "from dms.io.gpio_alert import GpioAlert\n"
        "log = []\n"
        "class Fake(object):\n"
        "    BOARD='BOARD'; OUT='OUT'; HIGH=1; LOW=0\n"
        "    def setwarnings(self, f): pass\n"
        "    def setmode(self, m): pass\n"
        "    def setup(self, p, d, initial=None): pass\n"
        "    def output(self, p, v): log.append(v); print('OUT %d' % v, flush=True)\n"
        "    def cleanup(self, p): print('CLEANUP', flush=True)\n"
        "g = GpioAlert(7, buzzer=True, gpio=Fake())\n"
        "g.set_critical(True)\n"
        "raise RuntimeError('unhandled')\n"
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
    )
    out = proc.stdout.decode()
    assert proc.returncode != 0  # the crash still propagates
    assert "OUT 1" in out
    assert out.rstrip().endswith("CLEANUP")
    assert "OUT 0" in out.split("OUT 1", 1)[1], "the pin must be driven low on exit"


def test_snapshot_reports_hold_time():
    ticks = [0.0]
    fake = FakeGpio()
    g = GpioAlert(7, buzzer=True, gpio=fake, clock=lambda: ticks[0])
    try:
        g.set_critical(True)
        ticks[0] = 4.0
        snap = g.snapshot()
        assert snap["asserted"] is True
        assert snap["asserted_seconds"] == 4.0
        g.set_critical(False)
        assert g.snapshot()["asserted_seconds"] == 4.0
        assert g.snapshot()["assert_count"] == 1
    finally:
        g.close()


# ------------------------------------------------------------------ sd_notify


def test_sd_notify_noop_without_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert under_systemd() is False
    assert sd_notify("READY=1") is False
    assert notify_ready() is False
    assert notify_watchdog() is False
    assert notify_status("x") is False
    assert notify_stopping() is False


def test_sd_notify_delivers_a_datagram(tmp_path, monkeypatch):
    path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(5)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", path)
        assert under_systemd() is True
        assert notify_ready("engines loaded") is True
        assert srv.recv(4096) == b"READY=1\nSTATUS=engines loaded"
        assert notify_watchdog() is True
        assert srv.recv(4096) == b"WATCHDOG=1"
        assert metrics.WATCHDOG_PINGS_TOTAL.value() == 1.0
        assert notify_stopping() is True
        assert srv.recv(4096) == b"STOPPING=1"
    finally:
        srv.close()


def test_a_dead_notify_socket_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "not-a-socket"))
    assert sd_notify("READY=1") is False


def test_abstract_socket_paths_are_translated(monkeypatch):
    from dms.io.sd_notify import notify_socket_path

    monkeypatch.setenv("NOTIFY_SOCKET", "@/org/freedesktop/systemd1/notify")
    assert notify_socket_path() == "\0/org/freedesktop/systemd1/notify"


def test_watchdog_interval_from_the_environment(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval_s() is None
    assert watchdog_ping_interval_s(default=5.0) == 5.0

    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    assert watchdog_interval_s() == 30.0
    # ping at half the deadline: one missed ping of margin before systemd kills us
    assert watchdog_ping_interval_s() == 15.0

    monkeypatch.setenv("WATCHDOG_USEC", "not-a-number")
    assert watchdog_interval_s() is None


def test_watchdog_pid_belongs_to_the_process_that_was_started(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))
    assert watchdog_interval_s() == 30.0
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() + 1))
    assert watchdog_interval_s() is None
