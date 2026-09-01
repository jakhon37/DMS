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
    parser.add_argument("--detect", action="store_true", help="run SCRFD face detector")
    parser.add_argument("--save-preview", default="", help="write last annotated JPEG")
    parser.add_argument("--save-video", default="", help="write annotated MJPG avi")
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

    detector = None
    landmarker = None
    writer = None
    preview = None
    if args.detect:
        import os

        from dms.geometry.ear import face_ear
        from dms.geometry.mar import mouth_aspect_ratio
        from dms.infer.landmarks import Landmark106
        from dms.infer.scrfd import ScrfdDetector
        from dms.runtime.trt_engine import TrtEngine
        from dms.track.driver_select import pick_driver
        from dms.viz.overlay import draw_faces

        log.info("loading detector %s", cfg.models.face.engine)
        face_engine = TrtEngine(cfg.models.face.engine)
        detector = ScrfdDetector(face_engine, conf=cfg.models.face.conf, iou=cfg.models.face.iou)
        if os.path.isfile(cfg.models.landmarks.engine):
            log.info("loading landmarks %s", cfg.models.landmarks.engine)
            lmk_engine = TrtEngine(cfg.models.landmarks.engine)
            landmarker = Landmark106(lmk_engine)
        else:
            log.warning("landmarks engine missing: %s", cfg.models.landmarks.engine)

    sd_notify("READY=1")
    t0 = time.monotonic()
    n = 0
    last_id = 0
    n_faces = 0
    infer_ms = []
    while not _STOP:
        frame = src.read(timeout_s=1.0)
        if frame is None:
            continue
        if not frame.ok or frame.full_bgra is None:
            continue
        n += 1
        last_id = frame.frame_id
        bgr = frame.full_bgra[..., :3]
        vis = bgr
        faces = []
        ear = mar = None
        if detector is not None:
            t1 = time.monotonic()
            faces = detector.detect(bgr)
            h, w = bgr.shape[:2]
            driver = pick_driver(faces, cfg.driver_roi, (w, h))
            lmk = None
            if landmarker is not None and driver is not None:
                lmk = landmarker.infer_one(bgr, driver.xyxy)
                if lmk is not None:
                    ear = face_ear(lmk)
                    mar = mouth_aspect_ratio(lmk)
            infer_ms.append((time.monotonic() - t1) * 1000.0)
            n_faces += len(faces)
            vis = draw_faces(bgr, faces, driver=driver, landmarks106=lmk, ear=ear, mar=mar)
            preview = vis
        if args.save_video:
            if writer is None:
                import cv2

                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    args.save_video, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (w, h)
                )
                if not writer.isOpened():
                    log.error("could not open video writer %s", args.save_video)
                    writer = None
            if writer is not None:
                writer.write(vis)
        if n % 15 == 0:
            dt = time.monotonic() - t0
            fps = n / dt if dt > 0 else 0.0
            p95 = 0.0
            if infer_ms:
                xs = sorted(infer_ms)
                p95 = xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1)))]
            log.info(
                "frames=%s fps=%.1f infer_p95_ms=%.1f faces_last=%s ear=%s mar=%s",
                n,
                fps,
                p95,
                len(faces),
                None if ear is None else round(ear, 3),
                None if mar is None else round(mar, 3),
            )
        sd_notify("WATCHDOG=1")
        if args.max_frames and n >= args.max_frames:
            break
    src.stop()
    if writer is not None:
        writer.release()
    if args.save_preview and preview is not None:
        import cv2
        import os

        os.makedirs(os.path.dirname(args.save_preview) or ".", exist_ok=True)
        cv2.imwrite(args.save_preview, preview)
        log.info("wrote preview %s", args.save_preview)
    mean_ms = (sum(infer_ms) / len(infer_ms)) if infer_ms else 0.0
    log.info(
        "stopped frames=%s last_id=%s mean_infer_ms=%.1f total_face_hits=%s",
        n,
        last_id,
        mean_ms,
        n_faces,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
