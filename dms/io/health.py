"""Localhost health HTTP (stdlib). No Flask/FastAPI."""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

log = logging.getLogger("dms.health")


class HealthState(object):
    """Mutable snapshot served at /healthz. Thread-safe via a lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ok = True
        self.camera = "starting"
        self.source = "file"
        self.dev_replay = True
        self.engines = {}  # type: Dict[str, str]
        self.fps = 0.0
        self.latency_ms = {"e2e_p95": 0.0}  # type: Dict[str, float]
        self.ram_mb = 0
        self.gpu_temp_c = None  # type: Optional[float]
        self.thermal_c = None  # type: Optional[float]
        self.gpu_clock_mhz = None  # type: Optional[float]
        self.alerts_active = []  # type: list
        self.frame_id = 0
        self.degraded = False

    def as_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "ok": self.ok,
                "camera": self.camera,
                "source": self.source,
                "dev_replay": self.dev_replay,
                "engines": dict(self.engines),
                "fps": round(self.fps, 2),
                "latency_ms": dict(self.latency_ms),
                "ram_mb": int(self.ram_mb),
                "gpu_temp_c": self.gpu_temp_c,
                "thermal_c": self.thermal_c,
                "gpu_clock_mhz": self.gpu_clock_mhz,
                "alerts_active": list(self.alerts_active),
                "frame_id": int(self.frame_id),
                "degraded": self.degraded,
            }

    def metrics_text(self) -> str:
        d = self.as_dict()
        lines = [
            "# HELP dms_fps Processed frames per second",
            "# TYPE dms_fps gauge",
            "dms_fps %s" % d["fps"],
            "# HELP dms_ram_mb Process / system available snapshot",
            "# TYPE dms_ram_mb gauge",
            "dms_ram_mb %s" % d["ram_mb"],
            "# HELP dms_frame_id Last processed frame id",
            "# TYPE dms_frame_id counter",
            "dms_frame_id %s" % d["frame_id"],
            "# HELP dms_stage_latency_ms Stage latency p95",
            "# TYPE dms_stage_latency_ms gauge",
        ]
        for stage, val in d["latency_ms"].items():
            lines.append('dms_stage_latency_ms{stage="%s"} %s' % (stage, val))
        if d["gpu_temp_c"] is not None:
            lines.append("# TYPE dms_gpu_temp_c gauge")
            lines.append("dms_gpu_temp_c %s" % d["gpu_temp_c"])
        if d["thermal_c"] is not None:
            lines.append("# TYPE dms_thermal_c gauge")
            lines.append("dms_thermal_c %s" % d["thermal_c"])
        return "\n".join(lines) + "\n"


def _handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code, body, content_type):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/healthz", "/health", "/"):
                self._send(200, json.dumps(state.as_dict()), "application/json")
                return
            if path == "/metrics":
                self._send(200, state.metrics_text(), "text/plain; version=0.0.4")
                return
            self._send(404, json.dumps({"error": "not found"}), "application/json")

    return Handler


class HealthServer(object):
    def __init__(self, state, bind="127.0.0.1", port=8088):
        self.state = state
        self.bind = bind
        self.port = int(port)
        self._httpd = None  # type: Optional[ThreadingHTTPServer]
        self._thread = None  # type: Optional[threading.Thread]

    def start(self) -> Optional[int]:
        try:
            self._httpd = ThreadingHTTPServer((self.bind, self.port), _handler(self.state))
            self._httpd.daemon_threads = True
        except OSError as exc:
            log.warning("health bind %s:%s failed: %s", self.bind, self.port, exc)
            self._httpd = None
            return None
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="dms-health", daemon=True)
        self._thread.start()
        actual = self._httpd.server_address[1]
        log.info("health listening http://%s:%s/healthz", self.bind, actual)
        return actual

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
