from __future__ import annotations

import logging
import os
import socket

log = logging.getLogger("dms.sd_notify")


def sd_notify(message: str) -> None:
    """Write a datagram to $NOTIFY_SOCKET. No libc sd_notify, no libsystemd."""
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return
    if path.startswith("@"):
        sock_path = "\0" + path[1:]
    else:
        sock_path = path
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(sock_path)
        sock.sendall(message.encode("utf-8"))
        sock.close()
    except OSError as exc:
        log.warning("NOTIFY_SOCKET send failed: %s", exc)
