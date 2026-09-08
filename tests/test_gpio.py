from __future__ import annotations

from dms.io.gpio_alert import GpioAlert
from dms.io.sd_notify import sd_notify


def test_gpio_noop_without_pin():
    g = GpioAlert(None, buzzer=False)
    g.set_critical(True)
    g.set_critical(False)
    g.close()


def test_gpio_noop_buzzer_off():
    g = GpioAlert(7, buzzer=False)
    assert g._gpio is None
    g.set_critical(True)
    g.close()


def test_sd_notify_noop_without_socket():
    sd_notify("READY=1")
    sd_notify("WATCHDOG=1")
