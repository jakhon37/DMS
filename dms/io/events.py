"""Durable append-only AlertEvent JSONL — the unit's only permanent safety record.

Lab default ``data/events.jsonl``; production ``/var/lib/dms/events.jsonl``.

Why this file is written the way it is
--------------------------------------
This is a vehicle. Ignition cut, a 12 V brown-out and an OOM kill all drop whatever
is still in the page cache, and the records that get dropped are the last ones before
the incident — exactly the ones an investigator wants. ``flush()`` alone only moves
bytes from Python's buffer into the kernel, so this module adds an explicit
:func:`os.fsync` barrier whose cost is bounded by policy (see *Durability* below).

The second failure this module exists to survive is a full disk. Before, an ENOSPC
propagated out of the main loop, the process exited non-zero, systemd restarted it,
it filled the disk again and after five restarts the DMS unit sat in ``failed`` —
driver monitoring silently gone because a log file grew. Here, ENOSPC is a state:
writing stops, ``DISK_FULL`` is raised to the caller, the health endpoint reports
``events_writable: false``, and the writer retries on a backoff. Nothing raises.

Record schema (``v`` = 1)
-------------------------
=============  =========================================================
``v``          schema version integer; bump on any incompatible change
``ts``         wall clock, Unix epoch seconds (float). May jump at NTP sync
``t``          ``time.monotonic()`` seconds — the clock alerts are timed on
``boot``       boot id (``/proc/sys/kernel/random/boot_id``) or a per-process
               uuid4 off-board; makes ``t`` orderable across restarts
``seq``        per-process record counter from 1
``kind``       ``"alert"`` for AlertEvents, ``"snapshot"`` for telemetry
``type``       AlertType value, e.g. ``"microsleep"``
``severity``   ``info`` | ``warn`` | ``critical``
``edge``       ``enter`` | ``exit``
``track_id``   driver track id or null
``extra``      free-form float map from the state machine
=============  =========================================================

Units: ``ts``/``t`` seconds, ``max_bytes`` bytes, ``min_free_mb`` MiB.

Durability policy (``fsync`` argument)
--------------------------------------
``"critical"`` (default) fsyncs every CRITICAL record immediately and everything else
at most once per ``fsync_interval_s``; ``"always"`` fsyncs every record; ``"interval"``
only rate-limits; ``"never"`` flushes to the kernel and stops there (lab use). On this
Xavier NX an fsync of one short line to the NVMe measures well under a millisecond and
alert edges are rare, so ``"critical"`` costs nothing measurable and bounds worst-case
loss to one second of non-critical edges.

Rotation
--------
Two independent mechanisms, because logrotate only runs from a daily cron and a burst
of alerts can outrun it:

* *in-process* — at ``max_bytes`` the live file is renamed to ``.1`` (shifting older
  generations up to ``backups``) and reopened. This is the real bound.
* *external* — logrotate ``create`` renames the file out from under the open fd. The
  writer stats the path every ``stat_interval_s`` and reopens when the inode changes,
  so no ``copytruncate`` (and no lost lines) is needed. ``reopen()`` is also the
  SIGHUP entry point for a ``postrotate`` trigger.

Failure behaviour
-----------------
No method here raises during operation. ``write_alert`` returns ``True`` on a durable
append and ``False`` on any refusal (closed, disk full, unwritable), counting the
reason in :mod:`dms.io.metrics`. Construction degrades to a fallback path and finally
to a disabled writer unless ``strict=True``, which makes an unwritable primary path
fatal at startup — before READY, which is where a production unit should fail.
"""
from __future__ import annotations

import atexit
import errno
import json
import logging
import os
import shutil
import time
import uuid
from typing import Any, Callable, Dict, Iterator, Optional, TextIO, Tuple

from dms.io import metrics
from dms.state.alerts import AlertEvent, AlertType, Severity

log = logging.getLogger("dms.events")

#: Schema version written as ``v`` on every record.
SCHEMA_VERSION = 1

#: Errnos that mean "the filesystem is out of room", as opposed to a broken path.
_FULL_ERRNOS = frozenset((errno.ENOSPC, errno.EDQUOT))

#: Accepted values for the ``fsync`` policy argument.
FSYNC_POLICIES = ("never", "interval", "critical", "always")

_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_MIN_FREE_MB = 64.0
_FILE_MODE = 0o640
_DIR_MODE = 0o750


def default_events_path(dev, configured=""):
    # type: (bool, str) -> str
    """Configured path wins; otherwise lab writes under ``data/``, production under /var/lib."""
    if configured:
        return configured
    if dev:
        return os.path.join("data", "events.jsonl")
    return "/var/lib/dms/events.jsonl"


def default_fallback_path(dev):
    # type: (bool) -> str
    """Where to go when the configured path cannot be opened.

    Lab falls back to ``data/events.jsonl`` (relative, next to the checkout).
    Production falls back to ``/run/dms/events.jsonl``: ``RuntimeDirectory=dms``
    already exists, it is tmpfs so it can never fill the root filesystem, and
    ``deploy/dms.logrotate`` covers it. It is *not* durable across a reboot, which is
    why the fallback is reported on ``/healthz`` as ``events_fallback: true`` rather
    than being silent.
    """
    if dev:
        return os.path.join("data", "events.jsonl")
    return "/run/dms/events.jsonl"


def _boot_id():
    # type: () -> str
    """Kernel boot id, or a per-process uuid4 when /proc is unavailable (off-board)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return str(uuid.uuid4())


def read_events(path, skip_bad=True):
    # type: (str, bool) -> Iterator[Dict[str, Any]]
    """Iterate records from a JSONL, tolerating a torn final line.

    A power cut can leave a partial line behind. With ``skip_bad`` (the default) such
    a line is logged at WARNING and skipped so one truncated record cannot make the
    whole incident history unreadable; with ``skip_bad=False`` the ``ValueError``
    propagates for a tool that wants to know.
    """
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                if not skip_bad:
                    raise
                log.warning("events %s:%d is not valid JSON; skipped", path, lineno)
                continue
            if isinstance(obj, dict):
                yield obj
            elif not skip_bad:
                raise ValueError("%s:%d is not a JSON object" % (path, lineno))


class EventLog(object):
    """Append-only JSONL writer with fsync, rotation, quota and ENOSPC survival.

    Args:
        path: configured destination, e.g. ``/var/lib/dms/events.jsonl``.
        fsync: one of :data:`FSYNC_POLICIES`.
        fsync_interval_s: upper bound between fsync barriers for non-critical records.
        max_bytes: in-process rotation threshold. 0 disables in-process rotation.
        backups: how many ``.1 .. .N`` generations to keep. Must match
            ``deploy/dms.logrotate`` so the two do not fight.
        min_free_mb: refuse to write below this much free space, leaving headroom for
            the rest of the system rather than being the process that fills the disk.
        strict: raise instead of falling back / degrading when the path is unusable.
        allow_fallback: try :func:`default_fallback_path` when the primary fails.
        dev: lab flag, only used to choose the fallback path.
        telemetry: privacy switch for :meth:`write_snapshot`; alerts are unaffected.
        on_disk_full: called with a synthetic ``AlertEvent(DISK_FULL)`` on each edge.
        retry_s: how long to stay in the disk-full state before retrying a write.
        stat_interval_s: how often to re-stat the path to notice an external
            rotation. 0 checks on every write; a negative value disables the check.
        clock: injectable ``time.monotonic`` for tests.
    """

    def __init__(
        self,
        path,
        fsync="critical",
        fsync_interval_s=1.0,
        max_bytes=_DEFAULT_MAX_BYTES,
        backups=2,
        min_free_mb=_DEFAULT_MIN_FREE_MB,
        strict=False,
        allow_fallback=True,
        dev=True,
        telemetry=False,
        on_disk_full=None,
        retry_s=5.0,
        stat_interval_s=2.0,
        clock=time.monotonic,
    ):
        # type: (str, str, float, int, int, float, bool, bool, bool, bool, Optional[Callable[[AlertEvent], None]], float, float, Callable[[], float]) -> None
        if fsync not in FSYNC_POLICIES:
            raise ValueError("fsync must be one of %s, got %r" % (list(FSYNC_POLICIES), fsync))
        self.configured_path = path
        self.path = path
        self.fsync_policy = fsync
        self.fsync_interval_s = float(fsync_interval_s)
        self.max_bytes = int(max_bytes)
        self.backups = max(0, int(backups))
        self.min_free_mb = float(min_free_mb)
        self.strict = bool(strict)
        self.telemetry = bool(telemetry)
        self.retry_s = float(retry_s)
        self.stat_interval_s = float(stat_interval_s)
        self._clock = clock
        self._on_disk_full = on_disk_full

        self.boot_id = _boot_id()
        self.seq = 0
        self.fallback = False
        self.writable = False
        self.disk_full = False
        self.open_error = ""  # type: str
        self.dropped = 0
        self.rotations = 0
        self.write_failures = 0

        self._fh = None  # type: Optional[TextIO]
        self._ino = None  # type: Optional[Tuple[int, int]]
        self._closed = False
        self._last_fsync = clock()
        self._last_stat = 0.0
        self._retry_at = 0.0
        self._full_since = 0.0
        self._full_dropped = 0
        self._warned_closed = False
        self._atexit_registered = False

        self._open_initial(path, allow_fallback=allow_fallback, dev=dev)
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True
        metrics.EVENTS_WRITABLE.set_bool(self.writable)

    # ----------------------------------------------------------------- opening

    def _open_initial(self, path, allow_fallback, dev):
        # type: (str, bool, bool) -> None
        try:
            self._open(path)
            return
        except OSError as exc:
            first = exc
        fallback = default_fallback_path(dev)
        same = os.path.abspath(fallback) == os.path.abspath(path)
        if self.strict or not allow_fallback or same:
            self.open_error = "%s: %s" % (path, first)
            if self.strict:
                # Fail closed, before READY: a production unit that cannot record
                # alerts must not pretend to be monitoring.
                raise first
            log.error("EVENTS_UNWRITABLE %s (%s); alerts will not be recorded", path, first)
            return
        log.warning(
            "EVENTS_FALLBACK configured=%s unwritable (%s); using %s — logrotate covers "
            "both paths but the fallback is not the durable location",
            path,
            first,
            fallback,
        )
        try:
            self._open(fallback)
            self.fallback = True
        except OSError as exc:
            self.open_error = "%s: %s (fallback %s: %s)" % (path, first, fallback, exc)
            log.error("EVENTS_UNWRITABLE %s", self.open_error)

    def _open(self, path):
        # type: (str) -> None
        """Open one path for append, 0640, creating its directory 0750."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, mode=_DIR_MODE, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
        try:
            # os.open's mode is masked by umask (022 under systemd unless UMask= is
            # set), so a 0640 request can land as 0644 and publish the driver's
            # drowsiness history to every local account. Fix it explicitly.
            try:
                os.fchmod(fd, _FILE_MODE)
            except OSError as exc:  # a foreign owner or a filesystem without modes
                log.debug("could not chmod %s to 0%o: %s", path, _FILE_MODE, exc)
            fh = os.fdopen(fd, "a")
        except Exception:
            os.close(fd)
            raise
        self._fh = fh
        self.path = path
        self.writable = True
        self.open_error = ""
        st = os.fstat(fh.fileno())
        self._ino = (st.st_dev, st.st_ino)
        self._last_stat = self._clock()
        self._ensure_newline(st.st_size)
        log.info("events jsonl %s (fsync=%s max_bytes=%s)", self.path, self.fsync_policy, self.max_bytes)

    def _ensure_newline(self, size):
        # type: (int) -> None
        """Terminate a torn final line so the next record cannot be glued onto it."""
        if size <= 0 or self._fh is None:
            return
        try:
            with open(self.path, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                last = probe.read(1)
        except OSError as exc:
            log.debug("could not probe tail of %s: %s", self.path, exc)
            return
        if last != b"\n":
            log.warning("events %s ends mid-record; terminating the torn line", self.path)
            try:
                self._fh.write("\n")
                self._fh.flush()
            except OSError as exc:
                log.warning("could not terminate torn line in %s: %s", self.path, exc)

    def reopen(self):
        # type: () -> bool
        """Close and reopen the current path. SIGHUP / logrotate ``postrotate`` entry point."""
        if self._closed:
            return False
        self._close_fh()
        try:
            self._open(self.path)
        except OSError as exc:
            self.writable = False
            self.open_error = "%s: %s" % (self.path, exc)
            log.error("EVENTS_REOPEN_FAILED %s", self.open_error)
            metrics.EVENTS_WRITABLE.set_bool(False)
            return False
        metrics.EVENTS_WRITABLE.set_bool(True)
        return True

    # ---------------------------------------------------------------- rotation

    def _check_external_rotation(self, now):
        # type: (float) -> None
        """Reopen if something renamed or replaced the file under our fd."""
        if self.stat_interval_s < 0:
            return
        if self.stat_interval_s > 0 and now - self._last_stat < self.stat_interval_s:
            return
        self._last_stat = now
        try:
            st = os.stat(self.path)
            ident = (st.st_dev, st.st_ino)
        except OSError:
            ident = None
        if ident != self._ino:
            log.info("events %s was rotated externally; reopening", self.path)
            metrics.EVENT_ROTATIONS_TOTAL.inc(reason="external")
            self.rotations += 1
            self.reopen()

    def _rotate(self):
        # type: () -> None
        """Shift ``.N-1 -> .N`` and the live file to ``.1``, then reopen. Never raises."""
        self._close_fh()
        try:
            for i in range(self.backups, 0, -1):
                src = self.path if i == 1 else "%s.%d" % (self.path, i - 1)
                dst = "%s.%d" % (self.path, i)
                if os.path.exists(src):
                    os.replace(src, dst)
            if self.backups == 0 and os.path.exists(self.path):
                os.unlink(self.path)
        except OSError as exc:
            log.error("events rotation failed for %s: %s", self.path, exc)
        metrics.EVENT_ROTATIONS_TOTAL.inc(reason="size")
        self.rotations += 1
        try:
            self._open(self.path)
        except OSError as exc:
            self.writable = False
            self.open_error = "%s: %s" % (self.path, exc)
            log.error("EVENTS_UNWRITABLE after rotation: %s", self.open_error)
            metrics.EVENTS_WRITABLE.set_bool(False)

    def _maybe_rotate(self, extra_bytes):
        # type: (int) -> None
        if self.max_bytes <= 0 or self._fh is None:
            return
        try:
            size = self._fh.tell()
        except (OSError, ValueError):
            return
        if size + extra_bytes > self.max_bytes:
            self._rotate()

    # --------------------------------------------------------------- disk full

    def _free_mb(self):
        # type: () -> Optional[float]
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        try:
            usage = shutil.disk_usage(d)
        except OSError:
            return None
        return usage.free / (1024.0 * 1024.0)

    def _enter_disk_full(self, detail, now):
        # type: (str, float) -> None
        if self.disk_full:
            return
        self.disk_full = True
        self.writable = False
        self._full_since = now
        self._full_dropped = 0
        self._retry_at = now + self.retry_s
        metrics.DISK_FULL.set(1.0)
        metrics.EVENTS_WRITABLE.set_bool(False)
        log.error("DISK_FULL events %s: %s — alert recording suspended", self.path, detail)
        self._notify_disk_full("enter", detail, now)

    def _exit_disk_full(self, now):
        # type: (float) -> None
        if not self.disk_full:
            return
        seconds = max(0.0, now - self._full_since)
        dropped = self._full_dropped
        self.disk_full = False
        self.writable = self._fh is not None
        metrics.DISK_FULL.set(0.0)
        metrics.EVENTS_WRITABLE.set_bool(self.writable)
        log.warning(
            "DISK_FULL cleared for %s after %.1f s; %d alert records were lost",
            self.path,
            seconds,
            dropped,
        )
        self._notify_disk_full("exit", "recovered", now, seconds=seconds, dropped=dropped)
        # Record the gap itself, so the JSONL says why it has a hole in it.
        self._emit(
            {
                "kind": "alert",
                "type": AlertType.DISK_FULL.value,
                "severity": Severity.WARN.value,
                "edge": "exit",
                "track_id": None,
                "extra": {"seconds": round(seconds, 3), "dropped": float(dropped)},
            },
            critical=True,
            now=now,
        )

    def _notify_disk_full(self, edge, detail, now, seconds=0.0, dropped=0):
        # type: (str, str, float, float, int) -> None
        if self._on_disk_full is None:
            return
        extra = {"seconds": round(seconds, 3), "dropped": float(dropped)}
        event = AlertEvent(
            t_mono_s=now,
            type=AlertType.DISK_FULL,
            severity=Severity.WARN,
            edge=edge,
            track_id=None,
            extra=extra,
        )
        try:
            self._on_disk_full(event)
        except Exception:  # a health/alert callback must never break the writer
            log.exception("on_disk_full callback raised (edge=%s, %s)", edge, detail)

    # ------------------------------------------------------------------- write

    def write_alert(self, ev):
        # type: (AlertEvent) -> bool
        """Append one AlertEvent. Returns True when the bytes reached the kernel.

        Never raises: a closed log, a full disk or an I/O error is counted, logged
        (throttled) and reported as ``False``.
        """
        metrics.record_alert(ev)
        record = {
            "kind": "alert",
            "type": getattr(ev.type, "value", ev.type),
            "severity": getattr(ev.severity, "value", ev.severity),
            "edge": ev.edge,
            "track_id": ev.track_id,
            "extra": dict(ev.extra or {}),
        }
        critical = record["severity"] == Severity.CRITICAL.value
        return self._emit(record, critical=critical, now=self._clock(), t_mono=ev.t_mono_s)

    def write_snapshot(self, payload):
        # type: (Dict[str, Any]) -> bool
        """Append a periodic driver-state snapshot, if ``privacy.telemetry`` allows it.

        Refuses with ``False`` and no disk write when ``telemetry`` is false — this is
        the enforcement point for ``privacy.telemetry``, which was previously a config
        key nothing read. Snapshots are never fsynced individually; they ride the
        interval barrier.
        """
        if not self.telemetry:
            metrics.EVENT_DROPPED_TOTAL.inc(reason="privacy_telemetry")
            return False
        record = {"kind": "snapshot", "payload": dict(payload)}
        return self._emit(record, critical=False, now=self._clock())

    def _emit(self, record, critical, now, t_mono=None):
        # type: (Dict[str, Any], bool, float, Optional[float]) -> bool
        if self._closed or self._fh is None:
            if not self._warned_closed:
                log.error(
                    "events write after close/open-failure (%s); alerts are being dropped",
                    self.open_error or self.path,
                )
                self._warned_closed = True
            self.dropped += 1
            metrics.EVENT_DROPPED_TOTAL.inc(reason="closed")
            return False
        if self.disk_full:
            if now < self._retry_at:
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False
            self._retry_at = now + self.retry_s
            free = self._free_mb()
            if free is not None and free < self.min_free_mb:
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False
            self._exit_disk_full(now)

        self._check_external_rotation(now)

        full = {"v": SCHEMA_VERSION, "ts": time.time(), "t": now if t_mono is None else t_mono,
                "boot": self.boot_id, "seq": self.seq + 1}
        full.update(record)
        line = json.dumps(full, sort_keys=False, default=str) + "\n"
        self._maybe_rotate(len(line))
        if self._fh is None:
            self.dropped += 1
            metrics.EVENT_DROPPED_TOTAL.inc(reason="closed")
            return False

        free = self._free_mb()
        if free is not None:
            metrics.DISK_FREE_MB.set(free, path=os.path.dirname(os.path.abspath(self.path)) or "/")
            if free < self.min_free_mb:
                self._enter_disk_full("free=%.1f MiB < min_free_mb=%.1f" % (free, self.min_free_mb), now)
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False

        try:
            self._fh.write(line)
            self._fh.flush()
        except OSError as exc:
            return self._on_write_error(exc, now)
        self.seq += 1
        metrics.EVENT_WRITES_TOTAL.inc()
        try:
            metrics.EVENTS_BYTES.set(self._fh.tell())
        except (OSError, ValueError):
            pass
        if self._should_fsync(critical, now):
            try:
                os.fsync(self._fh.fileno())
                self._last_fsync = now
                metrics.EVENT_FSYNCS_TOTAL.inc()
            except OSError as exc:
                return self._on_write_error(exc, now)
        return True

    def _should_fsync(self, critical, now):
        # type: (bool, float) -> bool
        if self.fsync_policy == "never":
            return False
        if self.fsync_policy == "always":
            return True
        if self.fsync_policy == "critical" and critical:
            return True
        return (now - self._last_fsync) >= self.fsync_interval_s

    def _on_write_error(self, exc, now):
        # type: (OSError, float) -> bool
        name = errno.errorcode.get(exc.errno or 0, str(exc.errno))
        self.write_failures += 1
        metrics.EVENT_WRITE_FAILURES_TOTAL.inc(errno=name)
        if exc.errno in _FULL_ERRNOS:
            self._enter_disk_full("%s writing %s" % (name, self.path), now)
        else:
            log.error("events write failed on %s: %s", self.path, exc)
            self.writable = False
            metrics.EVENTS_WRITABLE.set_bool(False)
        self.dropped += 1
        if self.disk_full:
            # The record that hit ENOSPC is lost too; count it in the gap summary.
            self._full_dropped += 1
        return False

    # ------------------------------------------------------------------- state

    def stats(self):
        # type: () -> Dict[str, Any]
        """Snapshot for ``/healthz``. Cheap; safe to call on every health refresh."""
        size = None
        try:
            if self._fh is not None:
                size = self._fh.tell()
        except (OSError, ValueError):
            size = None
        return {
            "path": self.path,
            "configured_path": self.configured_path,
            "writable": bool(self.writable and not self.disk_full),
            "fallback": self.fallback,
            "disk_full": self.disk_full,
            "bytes": size,
            "max_bytes": self.max_bytes,
            "written": self.seq,
            "dropped": self.dropped,
            "rotations": self.rotations,
            "write_failures": self.write_failures,
            "fsync": self.fsync_policy,
            "schema_version": SCHEMA_VERSION,
            "error": self.open_error,
        }

    # --------------------------------------------------------------- lifecycle

    def _close_fh(self):
        # type: () -> None
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except (OSError, ValueError) as exc:
            log.debug("fsync during close failed: %s", exc)
        try:
            fh.close()
        except OSError as exc:
            log.debug("close failed: %s", exc)

    def close(self):
        # type: () -> None
        """Flush, fsync and close. Idempotent, and registered with :mod:`atexit`."""
        if self._closed:
            return
        self._closed = True
        self._close_fh()
        self.writable = False
        metrics.EVENTS_WRITABLE.set_bool(False)

    def __enter__(self):
        # type: () -> "EventLog"
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> bool
        self.close()
        return False


def from_config(cfg, override_path="", on_disk_full=None):
    # type: (Any, str, Optional[Callable[[AlertEvent], None]]) -> EventLog
    """Build an :class:`EventLog` from an :class:`dms.config.schema.AppConfig`.

    Reads the optional keys ``events.fsync``, ``events.fsync_interval_s``,
    ``events.max_mb``, ``events.backups``, ``events.min_free_mb`` and
    ``events.strict`` with :func:`getattr` defaults, so this works against the
    current schema and against one that has grown those keys. ``privacy.telemetry``
    is read here and enforced in :meth:`EventLog.write_snapshot`.
    """
    ev = getattr(cfg, "events", None)
    privacy = getattr(cfg, "privacy", None)
    dev = bool(getattr(getattr(cfg, "source", None), "dev", True))
    path = override_path or default_events_path(dev, getattr(ev, "path", "") or "")
    max_mb = float(getattr(ev, "max_mb", 64.0) or 0.0)
    return EventLog(
        path,
        fsync=str(getattr(ev, "fsync", "critical")),
        fsync_interval_s=float(getattr(ev, "fsync_interval_s", 1.0)),
        max_bytes=int(max_mb * 1024 * 1024),
        backups=int(getattr(ev, "backups", 2)),
        min_free_mb=float(getattr(ev, "min_free_mb", _DEFAULT_MIN_FREE_MB)),
        strict=bool(getattr(ev, "strict", False)),
        dev=dev,
        telemetry=bool(getattr(privacy, "telemetry", False)),
        on_disk_full=on_disk_full,
    )


__all__ = [
    "EventLog",
    "SCHEMA_VERSION",
    "default_events_path",
    "default_fallback_path",
    "from_config",
    "read_events",
]
