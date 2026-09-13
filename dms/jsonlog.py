"""Structured stdout logging.

Every record carries both clocks. ``ts`` is wall time (what journald shows) and
``ts_mono`` is ``time.monotonic`` seconds - the same clock ``events.jsonl`` timestamps
alerts with. On a vehicle with no RTC the wall clock jumps at the first NTP/GNSS sync,
so log lines can go backwards while event timestamps do not; carrying both is what
makes post-incident correlation possible at all.

Units: ``ts`` is Unix epoch seconds (float), ``ts_mono`` is monotonic seconds since an
arbitrary origin (float).

Failure behaviour: :func:`setup_logging` raises ``ValueError`` on an unknown level
name rather than letting ``Logger.setLevel`` fail later with a bare traceback; use
:data:`LOG_LEVELS` for an argparse ``choices=``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

#: Accepted --log-level values, in increasing severity. Use as argparse choices.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_SERVICE = "dms"
_PID = os.getpid()

#: LogRecord attributes that are not user payload.
_RESERVED = frozenset(
    (
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName",
    )
)


def resolve_level(level: Any) -> int:
    """Map a level name or number to a logging level int. Raises ValueError."""
    if isinstance(level, int):
        return level
    name = str(level).upper().strip()
    value = logging.getLevelName(name)
    if not isinstance(value, int):
        raise ValueError(
            "unknown log level %r; expected one of %s" % (level, ", ".join(LOG_LEVELS))
        )
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with both clocks and process identity."""

    def __init__(self, service: str = _SERVICE) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "ts_mono": round(time.monotonic(), 6),
            "service": self.service,
            "pid": _PID,
            "thread": record.threadName or threading.current_thread().name,
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in payload:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", service: str = _SERVICE, stream: Optional[Any] = None) -> int:
    """Install the JSON formatter on the root logger. Returns the resolved level."""
    resolved = resolve_level(level)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JsonFormatter(service=service))
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(resolved)
    return resolved
