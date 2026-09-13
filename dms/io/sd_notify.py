"""systemd readiness/watchdog notification over ``$NOTIFY_SOCKET``.

Written by hand against the sd_notify protocol rather than linking ``libsystemd`` or
installing ``python-systemd``: the board's dependency policy forbids adding packages,
and the protocol is four lines of ``AF_UNIX``/``SOCK_DGRAM``.

The socket path may be a filesystem path or, when it starts with ``@``, a Linux
abstract socket (``\\0``-prefixed). Both are handled.

Units: :func:`watchdog_interval_s` returns seconds; ``$WATCHDOG_USEC`` is microseconds.

Failure behaviour: every function is a no-op outside systemd (no ``$NOTIFY_SOCKET``)
and logs — never raises — when the datagram cannot be delivered. A failed
notification must not take down a safety service; systemd's own watchdog will notice
the silence, which is the correct escalation path.

Watchdog contract
-----------------
``WatchdogSec=`` in the unit means *systemd kills the service if it does not hear
``WATCHDOG=1`` within that interval*. The rule the application must follow is that
the ping proves **frames are still flowing**, not merely that the process is
scheduled: :func:`watchdog_interval_s` gives the deadline, and the caller is expected
to ping at roughly half of it and to stop pinging when the pipeline stops advancing.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import Optional

from dms.io import metrics

log = logging.getLogger("dms.sd_notify")

_warned = False


def notify_socket_path():
    # type: () -> Optional[str]
    """The socket address from the environment, translated for abstract sockets.

    Returns ``None`` outside systemd. Only the process systemd started (or its
    children, if the unit sets ``NotifyAccess=all``) may notify.
    """
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return None
    if path.startswith("@"):
        return "\0" + path[1:]
    return path


def under_systemd():
    # type: () -> bool
    """True when a notification would actually go somewhere."""
    return notify_socket_path() is not None


def sd_notify(message):
    # type: (str) -> bool
    """Send one sd_notify datagram. Returns True when it was delivered.

    A fresh socket per call: notifications happen at most a few times a second and a
    long-lived fd would have to be re-created after a systemd restart anyway.
    """
    global _warned
    addr = notify_socket_path()
    if not addr:
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
        sock.settimeout(1.0)
        sock.connect(addr)
        sock.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        if not _warned:
            _warned = True
            log.warning("NOTIFY_SOCKET send failed (%s); further failures at debug", exc)
        else:
            log.debug("NOTIFY_SOCKET send failed: %s", exc)
        return False
    finally:
        if sock is not None:
            sock.close()


def notify_ready(status=""):
    # type: (str) -> bool
    """``READY=1`` — the unit is up. Sent once, after engines load and capture starts."""
    msg = "READY=1"
    if status:
        msg += "\nSTATUS=" + status.replace("\n", " ")
    return sd_notify(msg)


def notify_status(status):
    # type: (str) -> bool
    """``STATUS=`` — the one line ``systemctl status`` shows. Cheap; call freely."""
    return sd_notify("STATUS=" + status.replace("\n", " "))


def notify_watchdog():
    # type: () -> bool
    """``WATCHDOG=1`` — pet the watchdog. Counted in ``dms_watchdog_pings_total``."""
    ok = sd_notify("WATCHDOG=1")
    if ok:
        metrics.WATCHDOG_PINGS_TOTAL.inc()
    return ok


def notify_stopping(status=""):
    # type: (str) -> bool
    """``STOPPING=1`` — shutting down deliberately, so systemd does not call it a crash."""
    msg = "STOPPING=1"
    if status:
        msg += "\nSTATUS=" + status.replace("\n", " ")
    return sd_notify(msg)


def notify_reloading():
    # type: () -> bool
    """``RELOADING=1`` — used by the SIGHUP handler around a config/log reopen."""
    return sd_notify("RELOADING=1")


def watchdog_interval_s():
    # type: () -> Optional[float]
    """``WatchdogSec`` as seconds, or ``None`` when the watchdog is not armed.

    Honours ``$WATCHDOG_PID``: systemd sets it so that a forked child does not
    mistakenly believe the watchdog is its responsibility.
    """
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw:
        return None
    pid = os.environ.get("WATCHDOG_PID")
    if pid and pid.strip().isdigit() and int(pid) != os.getpid():
        return None
    try:
        usec = int(raw)
    except ValueError:
        log.warning("WATCHDOG_USEC=%r is not an integer; watchdog disabled", raw)
        return None
    if usec <= 0:
        return None
    return usec / 1e6


def watchdog_ping_interval_s(default=5.0):
    # type: (float) -> float
    """How often to ping: half the systemd deadline, or *default* outside systemd.

    Half is the interval systemd's own documentation recommends; it leaves one whole
    missed ping of margin before the unit is killed.
    """
    deadline = watchdog_interval_s()
    if deadline is None:
        return float(default)
    return max(0.5, deadline / 2.0)


__all__ = [
    "notify_ready",
    "notify_reloading",
    "notify_status",
    "notify_stopping",
    "notify_socket_path",
    "notify_watchdog",
    "sd_notify",
    "under_systemd",
    "watchdog_interval_s",
    "watchdog_ping_interval_s",
]
