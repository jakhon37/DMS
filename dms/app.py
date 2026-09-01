from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from dms.capture.gst_source import GstSource
from dms.config.schema import load_config
from dms.io.sd_notify import sd_notify
from dms.jsonlog import setup_logging

log = logging.getLogger("dms.app")
_STOP = False


def _handle_stop(signum, frame) -> None:
    global _STOP
    _STOP = True


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dms", description="Xavier NX Driver Monitoring System")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-frames", type=int, default=0, help="exit after N frames (0 = run)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error("bad config: %s", exc)
        return 2

    if cfg.source.type in ("file", "test") and not cfg.source.dev:
        log.warning("SOURCE_FILE source.type=%s path=%s (set source.dev true for lab)", cfg.source.type, cfg.source.path)

    if cfg.source.type == "file":
        import os

        if not os.path.isfile(cfg.source.path):
            log.error("replay file missing: %s", cfg.source.path)
            return 2

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    src = GstSource(cfg)
    try:
        src.start()
    except Exception as exc:
        log.error("capture start failed: %s", exc)
        if cfg.camera.fail_fatal or cfg.source.type == "file":
            return 1
        log.warning("camera.fail_fatal=false; notifying READY anyway")
        sd_notify("READY=1")
        while not _STOP:
            time.sleep(1.0)
            sd_notify("WATCHDOG=1")
        return 0

    sd_notify("READY=1")
    t0 = time.monotonic()
    n = 0
    last_id = 0
    while not _STOP:
        frame = src.read(timeout_s=1.0)
        if frame is None:
            continue
        if not frame.ok or frame.full_bgra is None:
            continue
        n += 1
        last_id = frame.frame_id
        if n % 30 == 0:
            dt = time.monotonic() - t0
            fps = n / dt if dt > 0 else 0.0
            log.info("capture ok frames=%s fps=%.1f shape=%s", n, fps, frame.full_bgra.shape)
        sd_notify("WATCHDOG=1")
        if args.max_frames and n >= args.max_frames:
            break
    src.stop()
    log.info("stopped frames=%s last_id=%s", n, last_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
