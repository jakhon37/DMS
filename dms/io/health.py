"""Health and metrics HTTP endpoint (stdlib only — no Flask, no FastAPI).

Routes
------
=============  ======  ====================================================
``/livez``     200     the process is running and the HTTP thread responds.
                       Never 503 — a liveness probe that fails on a dead
                       camera makes systemd kill a unit that is correctly
                       reporting a dead camera.
``/readyz``    200/503 200 only when every configured engine is loaded, the
                       camera is delivering, the frame id is advancing and
                       the snapshot is fresh. This is the probe a fleet
                       gates on.
``/healthz``   200/503 503 when ``ok`` is false or the snapshot is stale;
``/health``            200 with ``"degraded": true`` for a thermal or
``/``                  model degradation, which is a working unit.
``/metrics``   200     Prometheus text, always 200 — a scraper must be able
                       to scrape a broken unit, that is the point.
=============  ======  ====================================================

Any other path is 404; any method other than GET/HEAD is 405 with ``Allow``.

Why the status codes matter
---------------------------
Every standard prober — ``curl -f``, a systemd ``ExecStartPre`` gate, an uptime
monitor, a fleet agent — keys off the HTTP status and never parses the body. The
previous implementation returned 200 unconditionally, so a unit with a dead camera
and ``ok=false`` reported "healthy" to all of them. Worse, ``health`` fields were
refreshed only every fifteenth frame, so a hung pipeline froze fps, temperatures and
alert state at their last values and kept serving them as current. Both are fixed
here: :meth:`HealthState.mark_update` stamps a monotonic timestamp on every refresh
and a snapshot older than ``stale_after_s`` is reported as ``stale`` and answered 503.

Binding and exposure
--------------------
The default bind is ``127.0.0.1`` and a non-loopback bind is **refused** unless
``allow_remote=True`` is passed explicitly. ``/healthz`` publishes ``alerts_active``
— the driver's live drowsiness and distraction state, which is personal data — so a
config that quietly set ``bind: 0.0.0.0`` would publish it unauthenticated to the
whole vehicle network. When remote access is genuinely wanted, ``token`` enables
``Authorization: Bearer`` checking on everything except ``/livez``.

Resource limits: a 5 s socket timeout (a half-open connection previously pinned a
thread forever) and a hard cap on concurrent connections, over which the server
answers 503 and closes.

Units: ``fps`` frames per second, ``latency_ms`` milliseconds, ``ram_mb`` MiB,
temperatures degrees Celsius, ``uptime_s``/``stale_after_s`` seconds.

Failure behaviour: :meth:`HealthServer.start` returns ``None`` and logs an error when
the port is taken or the bind is refused; pass ``required=True`` to make that raise
instead, which is what a production profile should do rather than run blind.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from dms.io import metrics
from dms.io.thermal import disk_free_mb
from dms.types import ModelHealth

log = logging.getLogger("dms.health")

#: Addresses that keep the endpoint on the box.
LOOPBACK_BINDS = frozenset(("127.0.0.1", "::1", "localhost", "127.0.1.1"))

#: Snapshot older than this (seconds) means the pipeline is not refreshing health.
DEFAULT_STALE_AFTER_S = 5.0
#: No staleness or readiness verdict during startup; matches TimeoutStartSec=90.
DEFAULT_STARTUP_GRACE_S = 60.0


def is_loopback(bind):
    # type: (str) -> bool
    """True when *bind* keeps the listener off the network.

    An empty bind or ``0.0.0.0``/``::`` means every interface and is not loopback.
    """
    b = (bind or "").strip()
    if not b:
        return False
    if b in LOOPBACK_BINDS:
        return True
    try:
        packed = socket.inet_pton(socket.AF_INET, b)
        return packed[0] == 127
    except (OSError, ValueError):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, b)
        return b in ("::1",)
    except (OSError, ValueError):
        return False


class HealthState(object):
    """Mutable snapshot served at ``/healthz``. Thread-safe via :attr:`lock`.

    The pipeline thread mutates the plain attributes under ``with state.lock:`` and
    calls :meth:`mark_update`; the HTTP threads only ever read through
    :meth:`as_dict`, which takes the same lock.
    """

    def __init__(self, stale_after_s=DEFAULT_STALE_AFTER_S, startup_grace_s=DEFAULT_STARTUP_GRACE_S,
                 clock=time.monotonic):
        # type: (float, float, Any) -> None
        self.lock = threading.RLock()
        self._clock = clock
        self.started_mono = clock()
        self.last_update_mono = 0.0
        self._last_frame_seen = 0
        self.stale_after_s = float(stale_after_s)
        self.startup_grace_s = float(startup_grace_s)

        # -- liveness / readiness
        self.ok = True
        self.degraded = False
        self.camera = "starting"
        self.source = "file"
        self.dev_replay = True
        self.engines = {}  # type: Dict[str, str]
        self.models = {}  # type: Dict[str, ModelHealth]
        self.manifest = {"ok": None, "error": ""}  # type: Dict[str, Any]

        # -- throughput
        self.fps = 0.0
        self.latency_ms = {"e2e_p95": 0.0}  # type: Dict[str, float]
        self.frame_id = 0
        self.frames_total = 0
        self.frames_dropped = 0
        self.camera_restarts = 0

        # -- board
        self.ram_mb = 0
        self.mem_available_mb = None  # type: Optional[int]
        self.gpu_temp_c = None  # type: Optional[float]
        self.thermal_c = None  # type: Optional[float]
        self.gpu_clock_mhz = None  # type: Optional[float]
        self.thermal_zones = {}  # type: Dict[str, Any]

        # -- driver state and posture
        self.alerts_active = []  # type: List[str]
        self.privacy = {"record_faces": False, "telemetry": False}  # type: Dict[str, bool]
        self.clips = {"enabled": False, "reason": "not configured"}  # type: Dict[str, Any]
        self.events = {"writable": None, "path": ""}  # type: Dict[str, Any]
        self.disk = {}  # type: Dict[str, Optional[float]]
        self.disk_paths = []  # type: List[str]
        self.build = {}  # type: Dict[str, str]
        self.notes = []  # type: List[str]

    # ------------------------------------------------------------------ update

    def mark_update(self):
        # type: () -> None
        """Stamp the snapshot as fresh. Call at the end of every health refresh."""
        with self.lock:
            self.last_update_mono = self._clock()

    def _observe(self, now):
        # type: (float) -> None
        """Treat an advancing frame id as an implicit refresh.

        The pipeline is expected to call :meth:`mark_update`, but a caller that only
        assigns ``frame_id`` still gets working staleness detection: an id that has
        not moved between two observations more than ``stale_after_s`` apart is a
        stalled pipeline whatever else the snapshot says. Frame id 0 does not count —
        before the first frame there is nothing to be stale about, which is what
        ``startup_grace_s`` covers.
        """
        if self.frame_id > 0 and self.frame_id != self._last_frame_seen:
            self._last_frame_seen = self.frame_id
            self.last_update_mono = now

    def set_build(self, **labels):
        # type: (**str) -> None
        """Record and publish build identity (version, git_sha, trt_version, config)."""
        info = metrics.set_build_info(**labels)
        with self.lock:
            self.build = info

    def set_model(self, name, loaded=None, ok=None, mean_ms=None):
        # type: (str, Optional[bool], Optional[bool], Optional[float]) -> ModelHealth
        """Create or update one :class:`dms.types.ModelHealth` and mirror it to metrics.

        ``ok=True`` clears the failure streak and stamps ``last_ok_mono_s``;
        ``ok=False`` increments it. A model with a non-zero streak makes the unit
        ``degraded`` but not un-ready — a missing mouth classifier should not take a
        working drowsiness pipeline offline.
        """
        with self.lock:
            mh = self.models.get(name)
            if mh is None:
                mh = ModelHealth(name=name)
                self.models[name] = mh
            if loaded is not None:
                mh.loaded = bool(loaded)
            if ok is True:
                mh.consecutive_failures = 0
                mh.last_ok_mono_s = self._clock()
            elif ok is False:
                mh.consecutive_failures += 1
            if mean_ms is not None:
                mh.mean_ms = float(mean_ms)
            metrics.MODEL_LOADED.set_bool(mh.loaded, model=name)
            metrics.MODEL_FAILURES.set(mh.consecutive_failures, model=name)
            metrics.MODEL_MEAN_MS.set(mh.mean_ms, model=name)
            return mh

    def note(self, text):
        # type: (str) -> None
        """Attach a one-line operator-visible note (e.g. 'gaze disabled: licence')."""
        with self.lock:
            if text not in self.notes:
                self.notes.append(text)

    def refresh_disk(self):
        # type: () -> Dict[str, Optional[float]]
        """Re-stat every path in :attr:`disk_paths` and publish free MiB."""
        out = {}  # type: Dict[str, Optional[float]]
        for path in list(self.disk_paths):
            free = disk_free_mb(path)
            out[path] = None if free is None else round(free, 1)
            if free is not None:
                metrics.DISK_FREE_MB.set(free, path=path)
        with self.lock:
            self.disk = out
        return out

    # ------------------------------------------------------------------ verdict

    def _age_s(self, now):
        # type: (float) -> float
        base = self.last_update_mono or self.started_mono
        return max(0.0, now - base)

    def _in_grace(self, now):
        # type: (float) -> bool
        return self.last_update_mono == 0.0 and (now - self.started_mono) < self.startup_grace_s

    def is_stale(self):
        # type: () -> bool
        """True when the pipeline has stopped refreshing the snapshot."""
        with self.lock:
            now = self._clock()
            self._observe(now)
            if self._in_grace(now):
                return False
            return self._age_s(now) > self.stale_after_s

    def is_ok(self):
        # type: () -> bool
        """Liveness verdict behind ``/healthz``: the operator flag AND freshness."""
        with self.lock:
            return bool(self.ok) and not self.is_stale()

    def is_ready(self):
        # type: () -> bool
        """Readiness verdict behind ``/readyz``.

        Requires: ``ok``, a camera reporting ``ok``, every engine in
        :attr:`engines` marked loaded (a value of ``missing`` or ``failed`` is not),
        at least one processed frame, and a fresh snapshot.
        """
        with self.lock:
            if not self.ok or self.is_stale():
                return False
            if self.camera != "ok":
                return False
            if self.frame_id <= 0:
                return False
            for name, status in self.engines.items():
                if str(status) in ("missing", "failed", "error"):
                    return False
            return True

    # ----------------------------------------------------------------- rendering

    def as_dict(self):
        # type: () -> Dict[str, Any]
        """Full JSON body. Top-level keys are stable; new detail goes in sub-objects."""
        with self.lock:
            now = self._clock()
            stale = self.is_stale()
            return {
                "ok": bool(self.ok) and not stale,
                "live": True,
                "ready": self.is_ready(),
                "degraded": bool(self.degraded),
                "stale": stale,
                "camera": self.camera,
                "source": self.source,
                "dev_replay": self.dev_replay,
                "engines": dict(self.engines),
                "models": {
                    k: {
                        "loaded": v.loaded,
                        "consecutive_failures": v.consecutive_failures,
                        "mean_ms": round(v.mean_ms, 2),
                        "last_ok_age_s": (
                            None if not v.last_ok_mono_s else round(now - v.last_ok_mono_s, 1)
                        ),
                    }
                    for k, v in self.models.items()
                },
                "manifest": dict(self.manifest),
                "fps": round(self.fps, 2),
                "latency_ms": dict(self.latency_ms),
                "frame_id": int(self.frame_id),
                "frames_total": int(self.frames_total),
                "frames_dropped": int(self.frames_dropped),
                "camera_restarts": int(self.camera_restarts),
                "ram_mb": int(self.ram_mb),
                "mem_available_mb": self.mem_available_mb,
                "gpu_temp_c": self.gpu_temp_c,
                "thermal_c": self.thermal_c,
                "gpu_clock_mhz": self.gpu_clock_mhz,
                "thermal_zones": dict(self.thermal_zones),
                "alerts_active": list(self.alerts_active),
                "privacy": dict(self.privacy),
                "clips": dict(self.clips),
                "events": dict(self.events),
                "disk_free_mb": dict(self.disk),
                "build": dict(self.build),
                "notes": list(self.notes),
                "uptime_s": round(now - self.started_mono, 1),
                "snapshot_age_s": round(self._age_s(now), 2),
                "stale_after_s": self.stale_after_s,
            }

    def publish_metrics(self):
        # type: () -> None
        """Copy the snapshot into the metric registry. Called before every scrape."""
        d = self.as_dict()
        metrics.UP.set(1.0)
        metrics.UPTIME_SECONDS.set(d["uptime_s"])
        metrics.FPS.set(d["fps"])
        metrics.FRAME_ID.set(d["frame_id"])
        metrics.RAM_MB.set(d["ram_mb"])
        metrics.DEGRADED.set_bool(d["degraded"])
        metrics.HEALTH_STALE.set_bool(d["stale"])
        metrics.CAMERA_UP.set_bool(d["camera"] == "ok")
        metrics.ALERTS_ACTIVE.set(len(d["alerts_active"]))
        metrics.PRIVACY_RECORD_FACES.set_bool(d["privacy"].get("record_faces"))
        metrics.PRIVACY_TELEMETRY.set_bool(d["privacy"].get("telemetry"))
        # The pipeline owns monotonic totals as plain ints; mirror the increments into
        # the counters so a fleet gets rate() without the application having to know
        # the metric registry exists.
        _advance(metrics.FRAMES_TOTAL, d["frames_total"])
        _advance(metrics.FRAMES_PROCESSED_TOTAL, d["frame_id"])
        _advance(metrics.FRAMES_DROPPED_TOTAL, d["frames_dropped"], reason="capture")
        _advance(metrics.CAMERA_RESTARTS_TOTAL, d["camera_restarts"])
        if d["manifest"].get("ok") is not None:
            metrics.MANIFEST_OK.set_bool(d["manifest"]["ok"])
        if d["gpu_temp_c"] is not None:
            metrics.GPU_TEMP_C.set(d["gpu_temp_c"])
        if d["thermal_c"] is not None:
            metrics.THERMAL_C.set(d["thermal_c"])
        if d["gpu_clock_mhz"] is not None:
            metrics.GPU_CLOCK_MHZ.set(d["gpu_clock_mhz"])
        if d["events"].get("writable") is not None:
            metrics.EVENTS_WRITABLE.set_bool(d["events"]["writable"])
        for path, free in d["disk_free_mb"].items():
            if free is not None:
                metrics.DISK_FREE_MB.set(free, path=path)
        for key, value in d["latency_ms"].items():
            stage, quantile = _split_latency_key(key)
            try:
                metrics.STAGE_LATENCY_MS.set(float(value), stage=stage, quantile=quantile)
            except (TypeError, ValueError):
                continue

    def metrics_text(self):
        # type: () -> str
        """Prometheus exposition body for ``/metrics``."""
        self.publish_metrics()
        return metrics.render()


def _advance(counter, total, **labels):
    # type: (Any, float, **Any) -> None
    """Raise *counter* to *total*, ignoring a total that went backwards (a restart)."""
    have = counter.value(**labels)
    if total > have:
        counter.inc(total - have, **labels)


def _split_latency_key(key):
    # type: (str) -> Any
    """``'e2e_p95' -> ('e2e', 'p95')``; anything else keeps the key as the stage."""
    head, sep, tail = str(key).rpartition("_")
    if sep and tail.startswith("p") and tail[1:].isdigit():
        return head or "e2e", tail
    return str(key), "value"


# ------------------------------------------------------------------------ server


def _handler(state, token):
    # type: (HealthState, str) -> Any
    class Handler(BaseHTTPRequestHandler):
        # Explicit, not inherited: HTTP/1.0 closes every connection, which is what we
        # want from a probe endpoint, and the 5 s timeout stops one half-open socket
        # from holding a worker thread for the life of the process.
        protocol_version = "HTTP/1.0"
        timeout = 5.0
        server_version = "dms-health"
        sys_version = ""

        def log_message(self, fmt, *args):
            # Requests are counted in dms_health_requests_total; per-request stdout
            # lines would flood journald at fleet scrape rates.
            return

        def _send(self, code, body, content_type, extra_headers=()):
            data = body.encode("utf-8") if isinstance(body, str) else body
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for key, value in extra_headers:
                    self.send_header(key, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
            except (OSError, ValueError) as exc:  # client hung up mid-response
                log.debug("health response aborted: %s", exc)

        def _json(self, code, obj, route):
            metrics.HEALTH_REQUESTS_TOTAL.inc(route=route, code=str(code))
            self._send(code, json.dumps(obj), "application/json")

        def _authorised(self, route):
            if not token or route == "/livez":
                return True
            header = self.headers.get("Authorization", "")
            expected = "Bearer " + token
            # Constant-time-ish: lengths differ only when the token is wrong anyway.
            if len(header) == len(expected) and _consteq(header, expected):
                return True
            metrics.HEALTH_REQUESTS_TOTAL.inc(route=route, code="401")
            self._send(
                401,
                json.dumps({"error": "unauthorised"}),
                "application/json",
                (("WWW-Authenticate", 'Bearer realm="dms"'),),
            )
            return False

        def _route(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/health", "/"):
                path = "/healthz"
            if not self._authorised(path):
                return
            if path == "/livez":
                self._json(200, {"live": True, "pid": os.getpid()}, path)
                return
            if path == "/readyz":
                ready = state.is_ready()
                self._json(200 if ready else 503, state.as_dict(), path)
                return
            if path == "/healthz":
                body = state.as_dict()
                # 503 when not ok or stale. A degraded-but-working unit stays 200:
                # thermal throttling is a mode, not an outage.
                code = 200 if body["ok"] else 503
                self._json(code, body, path)
                return
            if path == "/metrics":
                text = state.metrics_text()
                metrics.HEALTH_REQUESTS_TOTAL.inc(route=path, code="200")
                self._send(200, text, "text/plain; version=0.0.4; charset=utf-8")
                return
            self._json(404, {"error": "not found", "routes": sorted(ROUTES)}, "unknown")

        def do_GET(self):
            self._route()

        def do_HEAD(self):
            self._route()

        def _reject_method(self):
            metrics.HEALTH_REQUESTS_TOTAL.inc(route="_", code="405")
            self._send(
                405,
                json.dumps({"error": "method not allowed"}),
                "application/json",
                (("Allow", "GET, HEAD"),),
            )

        do_POST = _reject_method
        do_PUT = _reject_method
        do_DELETE = _reject_method
        do_PATCH = _reject_method
        do_OPTIONS = _reject_method

    return Handler


ROUTES = ("/healthz", "/livez", "/readyz", "/metrics")


def _consteq(a, b):
    # type: (str, str) -> bool
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0


class _BoundedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a hard cap on simultaneous connections."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, max_connections=16):
        # type: (Any, Any, int) -> None
        self._max_connections = int(max_connections)
        self._active = 0
        self._count_lock = threading.Lock()
        ThreadingHTTPServer.__init__(self, address, handler)

    def process_request(self, request, client_address):
        with self._count_lock:
            over = self._active >= self._max_connections
            if not over:
                self._active += 1
        if over:
            log.warning("health connection refused: %d already active", self._max_connections)
            metrics.HEALTH_REQUESTS_TOTAL.inc(route="_", code="503")
            try:
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            except OSError:
                pass
            self.close_request(request)
            return
        ThreadingHTTPServer.process_request(self, request, client_address)

    def shutdown_request(self, request):
        try:
            ThreadingHTTPServer.shutdown_request(self, request)
        finally:
            with self._count_lock:
                self._active = max(0, self._active - 1)


class HealthServer(object):
    """Owns the listening socket and the serving thread.

    Args:
        state: the :class:`HealthState` to serve.
        bind: interface. Anything other than loopback needs ``allow_remote=True``.
        port: TCP port; 0 asks the kernel for a free one (tests).
        allow_remote: explicit opt-in to a non-loopback bind. Logged at WARNING every
            time it is used, because it publishes driver state to the network.
        token: when set, ``Authorization: Bearer <token>`` is required on every route
            except ``/livez``. Read it from a root-owned file, never from the config.
        max_connections: concurrent connection cap.
        required: raise instead of returning ``None`` when the listener cannot start.
    """

    def __init__(self, state, bind="127.0.0.1", port=8088, allow_remote=False, token="",
                 max_connections=16, required=False):
        # type: (HealthState, str, int, bool, str, int, bool) -> None
        self.state = state
        self.bind = bind
        self.port = int(port)
        self.allow_remote = bool(allow_remote)
        self.token = token or ""
        self.max_connections = int(max_connections)
        self.required = bool(required)
        self.error = ""
        self._httpd = None  # type: Optional[_BoundedServer]
        self._thread = None  # type: Optional[threading.Thread]

    @classmethod
    def from_config(cls, state, cfg, **kwargs):
        # type: (HealthState, Any, **Any) -> "HealthServer"
        """Build from ``cfg.health``, reading optional keys with getattr defaults."""
        hc = getattr(cfg, "health", None)
        token = str(getattr(hc, "token_file", "") or "")
        secret = ""
        if token:
            try:
                with open(token) as f:
                    secret = f.read().strip()
            except OSError as exc:
                log.error("health token_file %s unreadable: %s", token, exc)
        return cls(
            state,
            bind=str(getattr(hc, "bind", "127.0.0.1")),
            port=int(getattr(hc, "port", 8088)),
            allow_remote=bool(getattr(hc, "allow_remote", False)),
            token=secret,
            max_connections=int(getattr(hc, "max_connections", 16)),
            required=bool(getattr(hc, "required", False)),
            **kwargs
        )

    def _refuse(self, reason):
        # type: (str) -> Optional[int]
        self.error = reason
        log.error("HEALTH_BIND_REFUSED %s", reason)
        if self.required:
            raise RuntimeError("health endpoint refused: %s" % reason)
        return None

    def start(self):
        # type: () -> Optional[int]
        """Bind and serve. Returns the bound port, or ``None`` when it refused."""
        if not is_loopback(self.bind) and not self.allow_remote:
            return self._refuse(
                "bind=%s is not loopback and health.allow_remote is false; /healthz "
                "publishes live driver alert state" % self.bind
            )
        if not is_loopback(self.bind):
            if self.token:
                log.warning(
                    "HEALTH_REMOTE_BIND bind=%s port=%s — bearer token required",
                    self.bind, self.port,
                )
            else:
                log.warning(
                    "HEALTH_REMOTE_BIND bind=%s port=%s with NO token: live driver "
                    "alert state is readable by anything on this network",
                    self.bind, self.port,
                )
        try:
            self._httpd = _BoundedServer(
                (self.bind, self.port), _handler(self.state, self.token), self.max_connections
            )
        except OSError as exc:
            self._httpd = None
            return self._refuse("bind %s:%s failed: %s" % (self.bind, self.port, exc))
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="dms-health", daemon=True
        )
        self._thread.start()
        actual = self._httpd.server_address[1]
        self.port = actual
        self.error = ""
        log.info("health listening http://%s:%s/healthz (routes: %s)", self.bind, actual,
                 " ".join(ROUTES))
        return actual

    def stop(self):
        # type: () -> None
        """Stop serving and close the socket. Idempotent."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        metrics.UP.set(0.0)

    def __enter__(self):
        # type: () -> "HealthServer"
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # type: (Any, Any, Any) -> bool
        self.stop()
        return False


__all__ = ["HealthServer", "HealthState", "ROUTES", "is_loopback"]
