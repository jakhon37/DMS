"""Jetson.GPIO critical latch. No-op if pin is unset or the library is missing."""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("dms.gpio")


class GpioAlert(object):
    def __init__(self, pin, buzzer=False):
        self._pin = pin
        self._gpio = None
        self._asserted = False
        if pin is None or not buzzer:
            return
        try:
            import Jetson.GPIO as GPIO  # type: ignore

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(int(pin), GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            log.info("gpio alert pin=%s", pin)
        except Exception as exc:
            log.warning("gpio init failed pin=%s: %s", pin, exc)
            self._gpio = None

    def set_critical(self, on):
        # type: (bool) -> None
        if self._gpio is None or self._pin is None:
            return
        if bool(on) == self._asserted:
            return
        try:
            self._gpio.output(int(self._pin), self._gpio.HIGH if on else self._gpio.LOW)
            self._asserted = bool(on)
        except Exception as exc:
            log.warning("gpio write failed: %s", exc)

    def close(self):
        if self._gpio is None or self._pin is None:
            return
        try:
            self._gpio.output(int(self._pin), self._gpio.LOW)
            self._gpio.cleanup(int(self._pin))
        except Exception:
            pass
        self._gpio = None
