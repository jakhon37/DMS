"""Append-only AlertEvent JSONL. Lab: data/events.jsonl. Prod: /var/lib/dms/events.jsonl."""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, TextIO

from dms.state.alerts import AlertEvent

log = logging.getLogger("dms.events")


def default_events_path(dev, configured=""):
    # type: (bool, str) -> str
    if configured:
        return configured
    if dev:
        return os.path.join("data", "events.jsonl")
    return "/var/lib/dms/events.jsonl"


class EventLog(object):
    def __init__(self, path):
        # type: (str) -> None
        self.path = path
        self._fh = self._open(path)
        log.info("events jsonl %s", self.path)

    def _open(self, path):
        # type: (str) -> TextIO
        d = os.path.dirname(path)
        try:
            if d:
                os.makedirs(d, exist_ok=True)
            return open(path, "a")
        except OSError as exc:
            fallback = os.path.join("data", "events.jsonl")
            if os.path.abspath(path) == os.path.abspath(fallback):
                raise
            log.warning("events path %s not writable (%s); using %s", path, exc, fallback)
            self.path = fallback
            fd = os.path.dirname(fallback)
            if fd:
                os.makedirs(fd, exist_ok=True)
            return open(fallback, "a")

    def write_alert(self, ev):
        # type: (AlertEvent) -> None
        self._fh.write(
            json.dumps(
                {
                    "t": ev.t_mono_s,
                    "type": ev.type.value,
                    "severity": ev.severity.value,
                    "edge": ev.edge,
                    "track_id": ev.track_id,
                    "extra": ev.extra,
                }
            )
            + "\n"
        )
        self._fh.flush()

    def close(self):
        # type: () -> None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
