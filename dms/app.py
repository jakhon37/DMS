from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from dms.capture.gst_source import GstSource
from dms.config.schema import load_config
from dms.io.gpio_alert import GpioAlert
from dms.io.health import HealthServer, HealthState
from dms.io.sd_notify import sd_notify
from dms.io.thermal import gpu_clock_mhz, ram_used_mb, read_temps, thermal_throttle
from dms.jsonlog import setup_logging

log = logging.getLogger("dms.app")
_STOP = False


def _handle_stop(signum, frame) -> None:
    global _STOP
    _STOP = True


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dms", description="Xavier NX Driver Monitoring System")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="", help="override source.path for file replay")
    parser.add_argument("--max-frames", type=int, default=0, help="exit after N frames (0 = until EOS on file)")
    parser.add_argument("--detect", action="store_true", help="run SCRFD face detector")
    parser.add_argument("--save-preview", default="", help="write last annotated JPEG")
    parser.add_argument("--save-video", default="", help="write annotated MJPG avi")
    parser.add_argument("--calibrate-forward", action="store_true", help="median pose/EAR -> configs/vehicle.yaml")
    parser.add_argument("--events", default="", help="append AlertEvent JSONL")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        log.error("bad config: %s", exc)
        return 2

    if args.source:
        cfg.source.type = "file"
        cfg.source.path = args.source
        cfg.source.dev = True

    if cfg.source.type in ("file", "test") and not cfg.source.dev:
        log.warning("SOURCE_FILE source.type=%s path=%s (set source.dev true for lab)", cfg.source.type, cfg.source.path)

    if cfg.source.type == "file":
        if not os.path.isfile(cfg.source.path):
            log.error("replay file missing: %s", cfg.source.path)
            return 2

    if cfg.require_engines:
        missing = [p for p in (cfg.models.face.engine, cfg.models.landmarks.engine) if p and not os.path.isfile(p)]
        if missing:
            log.error("ENGINE_FAIL missing %s", missing)
            return 1

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    health = HealthState()
    health.source = cfg.source.type
    health.dev_replay = cfg.source.type in ("file", "test")
    health.camera = "starting"
    hsv = HealthServer(health, bind=cfg.health.bind, port=cfg.health.port)
    hsv.start()
    gpio = GpioAlert(cfg.alerts.gpio_pin, buzzer=cfg.alerts.buzzer)

    detector = None
    landmarker = None
    writer = None
    preview = None
    if args.detect:
        from dms.geometry.ear import face_ear
        from dms.geometry.mar import mouth_aspect_ratio
        from dms.geometry.pnp import solve_head_pose
        from dms.infer.landmarks import Landmark106
        from dms.infer.scrfd import ScrfdDetector
        from dms.runtime.trt_engine import TrtEngine
        from dms.state.alerts import Severity
        from dms.state.driver_state import DriverMonitor, angdiff
        from dms.track.driver_select import label_occupants, pick_driver
        from dms.track.iou_tracker import IouTracker
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

    health.engines = {
        "face": cfg.models.face.device if detector is not None else "missing",
        "landmarks": "gpu" if landmarker is not None else "missing",
        "pose": cfg.models.pose.mode,
        "objects": "off" if not cfg.models.objects.enabled else cfg.models.objects.engine,
    }

    src = GstSource(cfg)
    try:
        src.start()
        health.camera = "ok"
    except Exception as exc:
        log.error("capture start failed: %s", exc)
        health.camera = "failed"
        if cfg.source.type not in ("file", "test"):
            health.ok = False
        if cfg.camera.fail_fatal or cfg.source.type == "file":
            hsv.stop()
            gpio.close()
            return 1
        log.warning("camera.fail_fatal=false; notifying READY anyway")
        sd_notify("READY=1")
        while not _STOP:
            time.sleep(1.0)
            sd_notify("WATCHDOG=1")
            gpu_c, th_c = read_temps()
            with health.lock:
                health.ram_mb = ram_used_mb()
                health.gpu_temp_c = gpu_c
                health.thermal_c = th_c
                health.gpu_clock_mhz = gpu_clock_mhz()
        hsv.stop()
        gpio.close()
        return 0

    if args.save_video:
        d = os.path.dirname(args.save_video)
        if d:
            os.makedirs(d, exist_ok=True)
    if args.events:
        d = os.path.dirname(args.events)
        if d:
            os.makedirs(d, exist_ok=True)

    sd_notify("READY=1")
    t0 = time.monotonic()
    n = 0
    last_id = 0
    n_faces = 0
    misses = 0
    infer_ms = []
    last_wd = 0.0
    last_wd_id = -1
    thermal_on = False
    clocks_low_logged = False
    tracker = IouTracker() if args.detect else None
    monitor = DriverMonitor(cfg) if args.detect else None
    calib_yaw, calib_pitch, calib_ear = [], [], []
    event_fh = open(args.events, "a") if args.events else None
    while not _STOP:
        frame = src.read(timeout_s=1.0)
        if frame is None:
            if cfg.source.type == "file" and src.eos:
                break
            if cfg.source.type == "file" and n > 0 and misses >= 5:
                break
            misses += 1
            continue
        misses = 0
        if not frame.ok or frame.full_bgra is None:
            continue
        n += 1
        last_id = frame.frame_id
        bgr = frame.full_bgra[..., :3]
        vis = bgr
        faces = []
        alert_names = []
        ear = mar = None
        pose = None
        if detector is not None:
            t1 = time.monotonic()
            faces = detector.detect(bgr)
            h, w = bgr.shape[:2]
            tracks = tracker.update(faces) if tracker is not None else []
            live = [t for t in tracks if t.face is not None]
            driver_face = pick_driver(
                [t.face for t in live], cfg.driver_roi, (w, h), fallback=cfg.driver_fallback
            )
            driver_track = None
            for t in live:
                if t.face is driver_face:
                    driver_track = t
                    break
            lmk = None
            pose = None
            if landmarker is not None and driver_face is not None:
                lmk = landmarker.infer_one(bgr, driver_face.xyxy)
                if lmk is not None:
                    ear = face_ear(lmk)
                    mar = mouth_aspect_ratio(lmk)
                    pose = solve_head_pose(lmk, (w, h))
            infer_ms.append((time.monotonic() - t1) * 1000.0)
            n_faces += len(faces)
            t_s = frame.t_mono_ns / 1e9
            present = driver_face is not None
            yaw = pitch = None
            if pose is not None:
                yaw, pitch = pose[0], pose[1]
            alert_names = []
            yaw_rel = None
            if monitor is not None:
                evs = monitor.update(
                    t_s,
                    present=present,
                    track_id=None if driver_track is None else driver_track.track_id,
                    ear=ear,
                    mar=mar,
                    yaw=yaw,
                    pitch=pitch,
                )
                for ev in evs:
                    log.info("ALERT %s %s %s", ev.edge, ev.type.value, ev.severity.value)
                    if event_fh is not None:
                        import json

                        event_fh.write(
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
                alert_names = ["%s:%s" % (k.value, v.value) for k, v in monitor.active.items()]
                if monitor.yaw_ewma is not None:
                    yaw_rel = angdiff(monitor.yaw_ewma, cfg.forward_zero.yaw)
                gpio.set_critical(monitor.highest_severity() == Severity.CRITICAL)
            if args.calibrate_forward and present and yaw is not None and ear is not None:
                calib_yaw.append(yaw)
                calib_pitch.append(pitch if pitch is not None else 0.0)
                calib_ear.append(ear)
            want_overlay = bool(args.save_video or args.save_preview or cfg.display.enabled)
            if want_overlay or not health.degraded:
                vis = draw_faces(
                    bgr,
                    faces,
                    driver=driver_face,
                    landmarks106=lmk,
                    ear=ear,
                    mar=mar,
                    pose=pose,
                    track_id=None if driver_track is None else driver_track.track_id,
                    alerts=alert_names,
                    yaw_rel=yaw_rel,
                    occupant_labels=label_occupants(live, driver_face),
                )
            preview = vis
        if args.save_video:
            if writer is None:
                import cv2

                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    args.save_video, cv2.VideoWriter_fourcc(*"MJPG"), float(cfg.source.fps), (w, h)
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
                "frames=%s fps=%.1f infer_p95_ms=%.1f faces_last=%s ear=%s mar=%s pose=%s",
                n,
                fps,
                p95,
                len(faces),
                None if ear is None else round(ear, 3),
                None if mar is None else round(mar, 3),
                None if pose is None else (round(pose[0], 1), round(pose[1], 1)),
            )
            gpu_c, th_c = read_temps()
            clock = gpu_clock_mhz()
            hot = thermal_throttle(gpu_c, th_c)
            if hot and not thermal_on:
                log.warning("THERMAL_THROTTLE gpu=%s thermal=%s", gpu_c, th_c)
                thermal_on = True
            if not hot:
                thermal_on = False
            if clock is not None and clock < 300 and not clocks_low_logged:
                log.warning("CLOCKS_LOW gpu_clock_mhz=%s", clock)
                clocks_low_logged = True
            with health.lock:
                health.fps = fps
                health.latency_ms = {"e2e_p95": round(p95, 1)}
                health.frame_id = last_id
                health.ram_mb = ram_used_mb()
                health.gpu_temp_c = gpu_c
                health.thermal_c = th_c
                health.gpu_clock_mhz = clock
                health.degraded = hot
                health.alerts_active = list(alert_names) if detector is not None else []
        now = time.monotonic()
        if now - last_wd >= 5.0 and last_id != last_wd_id:
            sd_notify("WATCHDOG=1")
            last_wd = now
            last_wd_id = last_id
        if args.max_frames and n >= args.max_frames:
            break
    src.stop()
    if writer is not None:
        writer.release()
    if args.save_preview and preview is not None:
        import cv2

        os.makedirs(os.path.dirname(args.save_preview) or ".", exist_ok=True)
        cv2.imwrite(args.save_preview, preview)
        log.info("wrote preview %s", args.save_preview)
    if event_fh is not None:
        event_fh.close()
    if args.calibrate_forward and calib_yaw:
        import yaml

        def _med(xs):
            s = sorted(xs)
            return s[len(s) // 2]

        veh = {
            "forward_zero": {
                "yaw": float(_med(calib_yaw)),
                "pitch": float(_med(calib_pitch)),
                "roll": 0.0,
            },
            "state": {"ear_open_median": float(_med(calib_ear))},
        }
        outp = os.path.join(os.path.dirname(os.path.abspath(args.config)), "vehicle.yaml")
        with open(outp, "w") as f:
            yaml.safe_dump(veh, f, sort_keys=False)
        log.info("wrote %s n=%s zero=%s ear_open=%s", outp, len(calib_yaw), veh["forward_zero"], veh["state"])
    mean_ms = (sum(infer_ms) / len(infer_ms)) if infer_ms else 0.0
    log.info(
        "stopped frames=%s last_id=%s mean_infer_ms=%.1f total_face_hits=%s",
        n,
        last_id,
        mean_ms,
        n_faces,
    )
    hsv.stop()
    gpio.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
