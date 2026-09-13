"""Physical critical-alert output: one GPIO pin driving a buzzer or lamp.

This is the only physical actuator the DMS owns, so its failure mode matters more
than its feature set. Two rules:

1. **Fail safe on exit.** The latch is released (pin LOW) by :meth:`close`, which is
   registered with :mod:`atexit` and also runs from ``__exit__``. Previously
   ``close()`` was reachable only on the clean return path of ``main()``, so an
   unhandled exception, an OOM kill or the watchdog's SIGKILL left the buzzer
   asserted HIGH with the DMS process dead — a screaming cabin and no monitoring.
   :meth:`arm_signal_handlers` additionally installs SIGTERM/SIGINT handlers (chaining
   whatever was there before), which covers ``systemctl stop`` and Ctrl-C. Nothing in
   software can cover SIGKILL — the watchdog's kill, or an OOM kill — so the pin must
   also be wired fail-safe in hardware: the buzzer/lamp de-energised when the GPIO is
   released to its input default. ``deploy/README.md`` § "Physical alert output"
   records that requirement.
2. **Fail quiet on init.** A missing ``Jetson.GPIO``, a pin already exported by
   another process, or no permission on ``/dev/gpiochip*`` disables the output and
   logs once. A DMS with no buzzer still monitors the driver; a DMS that refuses to
   start because a lamp is unplugged does not.

Configuration: ``alerts.gpio_pin`` (BOARD numbering) and ``alerts.buzzer``. Both must
be set — a pin number with ``buzzer: false`` is an explicit "wired but muted" state,
not an oversight, and is honoured as a no-op.

Units: none — the pin is a boolean. :attr:`asserted_seconds` is seconds.

Failure behaviour: no method raises. A write failure is logged, counted, and leaves
:attr:`healthy` false so ``/healthz`` can show that the physical alert is not working.
"""
from __future__ import annotations

import atexit
import logging
import signal
import time
from typing import Any, Dict, Optional

log = logging.getLogger("dms.gpio")


class GpioAlert(object):
    """Drive one output pin HIGH while a CRITICAL alert is latched.

    Args:
        pin: BOARD-numbered header pin, or ``None`` to disable.
        buzzer: master enable. ``False`` disables the output even with a pin set.
        active_high: set ``False`` for a relay board that sinks current.
        gpio: inject a fake GPIO module in tests; anything with ``setmode``,
            ``setup``, ``output``, ``cleanup``, ``BOARD``, ``OUT``, ``HIGH``, ``LOW``.
        clock: injectable monotonic clock.
    """

    def __init__(self, pin, buzzer=False, active_high=True, gpio=None, clock=time.monotonic):
        # type: (Optional[int], bool, bool, Any, Any) -> None
        self._pin = None if pin is None else int(pin)
        self._buzzer = bool(buzzer)
        self._active_high = bool(active_high)
        self._clock = clock
        self._gpio = None  # type: Any
        self._asserted = False
        # None, not 0.0: with an injected clock t=0.0 is a real timestamp and a
        # falsy sentinel would silently discard the first hold period.
        self._since = None  # type: Optional[float]
        self.asserted_seconds = 0.0
        self.assert_count = 0
        self.write_failures = 0
        self.healthy = False
        self.reason = ""
        self._closed = False
        self._prev_handlers = {}  # type: Dict[int, Any]

        if self._pin is None or not self._buzzer:
            self.reason = "no pin" if self._pin is None else "alerts.buzzer=false"
            log.info("gpio alert inactive: %s", self.reason)
            return
        try:
            if gpio is None:
                import Jetson.GPIO as gpio  # type: ignore
            gpio.setwarnings(False)
            gpio.setmode(gpio.BOARD)
            gpio.setup(self._pin, gpio.OUT, initial=self._off_level(gpio))
            self._gpio = gpio
            self.healthy = True
            log.info("gpio alert pin=%s active_%s", self._pin, "high" if self._active_high else "low")
        except Exception as exc:
            self.reason = "init failed: %s" % exc
            log.warning("gpio init failed pin=%s: %s", self._pin, exc)
            self._gpio = None
        atexit.register(self.close)

    # ------------------------------------------------------------------ levels

    def _on_level(self, gpio):
        # type: (Any) -> Any
        return gpio.HIGH if self._active_high else gpio.LOW

    def _off_level(self, gpio):
        # type: (Any) -> Any
        return gpio.LOW if self._active_high else gpio.HIGH

    # ------------------------------------------------------------------- drive

    @property
    def active(self):
        # type: () -> bool
        """True when the pin is currently asserted."""
        return self._asserted

    def set_critical(self, on):
        # type: (bool) -> None
        """Assert or release the latch. Idempotent; only edges reach the hardware."""
        want = bool(on)
        if self._gpio is None or self._pin is None:
            return
        if want == self._asserted:
            return
        try:
            self._gpio.output(self._pin, self._on_level(self._gpio) if want else self._off_level(self._gpio))
        except Exception as exc:
            self.write_failures += 1
            self.healthy = False
            self.reason = "write failed: %s" % exc
            log.warning("gpio write failed pin=%s: %s", self._pin, exc)
            return
        now = self._clock()
        if want:
            self._since = now
            self.assert_count += 1
        elif self._since is not None:
            self.asserted_seconds += max(0.0, now - self._since)
            self._since = None
        self._asserted = want
        self.healthy = True

    def arm_signal_handlers(self, signals=(signal.SIGTERM, signal.SIGINT)):
        # type: (Any) -> None
        """Release the latch on SIGTERM/SIGINT, then chain the previous handler.

        Only call this from the main thread of a process that owns its signal
        disposition. The previous handler is preserved and invoked, so an application
        that already installs a stop handler keeps working.
        """
        for sig in signals:
            try:
                previous = signal.getsignal(sig)
                self._prev_handlers[sig] = previous
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError) as exc:  # not the main thread / bad signal
                log.debug("could not arm gpio handler for %s: %s", sig, exc)

    def _on_signal(self, signum, frame):
        # type: (int, Any) -> None
        self.close()
        previous = self._prev_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            import os

            os.kill(os.getpid(), signum)

    # --------------------------------------------------------------- lifecycle

    def close(self):
        # type: () -> None
        """Drive the pin to its inactive level and release it. Idempotent.

        Registered with :mod:`atexit`, so a crash out of the main loop still drops
        the buzzer. Failures here are swallowed by design: this runs during teardown,
        often while the interpreter is finalising.
        """
        if self._closed:
            return
        self._closed = True
        gpio, self._gpio = self._gpio, None
        if gpio is None or self._pin is None:
            return
        try:
            gpio.output(self._pin, self._off_level(gpio))
        except Exception as exc:
            log.debug("gpio release write failed: %s", exc)
        try:
            gpio.cleanup(self._pin)
        except Exception as exc:
            log.debug("gpio cleanup failed: %s", exc)
        if self._asserted and self._since is not None:
            self.asserted_seconds += max(0.0, self._clock() - self._since)
            self._since = None
        self._asserted = False
        log.info("gpio alert released pin=%s", self._pin)

    def __enter__(self):
        # type: () -> "GpioAlert"
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> bool
        self.close()
        return False

    def snapshot(self):
        # type: () -> Dict[str, Any]
        """State for ``/healthz`` under the ``gpio`` key."""
        held = self.asserted_seconds
        if self._asserted and self._since is not None:
            held += max(0.0, self._clock() - self._since)
        return {
            "pin": self._pin,
            "enabled": self._gpio is not None,
            "asserted": self._asserted,
            "healthy": self.healthy,
            "reason": self.reason,
            "assert_count": self.assert_count,
            "asserted_seconds": round(held, 2),
            "write_failures": self.write_failures,
        }


__all__ = ["GpioAlert"]
