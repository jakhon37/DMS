#!/usr/bin/env python3
"""Look-forward calibration: measure this vehicle's zero and check it is sane.

The state machine's distraction thresholds are angles *relative to a forward
zero* (``forward_zero.yaw/pitch`` in ``configs/vehicle.yaml``).  That zero is a
property of the camera mounting, not of the driver, so it must be measured on
the installed hardware once and then left alone.  A wrong zero silently biases
every gaze alert in one direction: a 15 deg mounting error against a 25 deg
threshold means the driver trips the warning 10 deg early on one side and never
on the other.

This wrapper drives ``dms.app --detect --calibrate-forward`` for a fixed
duration and then **validates and reports** what was written, which the app
itself does not do.  It exists so a bad calibration fails here, loudly, instead
of quietly shifting the alert thresholds of a shipped vehicle.

Procedure
---------
1. Park the vehicle. Sit in the driver's seat in your normal driving posture.
2. Look straight down the road, eyes open, for the whole capture window.
3. Run::

       PYTHONPATH=. ./.venv/bin/python scripts/calibrate_forward.py \
           --config configs/production.yaml --duration-s 5

4. Read the verdict. On ``OK`` the new ``configs/vehicle.yaml`` is in place and
   the previous one is saved next to it with a timestamp.

Checks applied to the result
----------------------------
==========================  ==================================================
samples                     at least ``--min-samples`` frames contributed a
                            pose and an EAR; too few means the face was not
                            being detected and the median is noise
``|yaw|``, ``|pitch|``      within :data:`MAX_ABS_YAW_DEG` /
                            :data:`MAX_ABS_PITCH_DEG` of zero - a larger value
                            is a mounting problem or a driver who was not
                            looking forward, not a calibration
``ear_open_median``         within :data:`EAR_OPEN_RANGE`; outside it the eyes
                            were closed, or the landmark model is misbehaving
                            and the drowsiness threshold derived from it would
                            be wrong
==========================  ==================================================

Units: angles in degrees, EAR dimensionless, duration in seconds.

Failure behaviour: exits 0 only when the file was written *and* every check
passed.  Exit 1 = the app failed; exit 2 = bad arguments or no config; exit 3 =
calibration produced no file; exit 4 = the file was written but a check failed,
and the previous ``vehicle.yaml`` has been restored so the vehicle keeps its
last known-good zero.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: A forward zero further from straight ahead than this is a mounting fault.
MAX_ABS_YAW_DEG = 35.0
MAX_ABS_PITCH_DEG = 30.0

#: Plausible range for the open-eye contour EAR on the 106-point model.
EAR_OPEN_RANGE = (0.15, 1.30)

#: Below this many contributing frames the medians are not a measurement.
DEFAULT_MIN_SAMPLES = 45


def vehicle_yaml_path(config_path):
    # type: (str) -> str
    """``vehicle.yaml`` sits beside the config; ``dms.app`` writes it there."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), "vehicle.yaml")


def check_result(veh, min_samples=DEFAULT_MIN_SAMPLES, samples=None):
    # type: (dict, int, object) -> list
    """Return a list of human-readable problems. Empty list means the zero is usable."""
    problems = []
    fz = veh.get("forward_zero") or {}
    st = veh.get("state") or {}
    yaw = fz.get("yaw")
    pitch = fz.get("pitch")
    ear = st.get("ear_open_median")
    if yaw is None or pitch is None:
        problems.append("forward_zero is missing yaw or pitch")
    else:
        if abs(float(yaw)) > MAX_ABS_YAW_DEG:
            problems.append(
                "forward_zero.yaw = %.1f deg, more than %.0f deg off centre - the camera "
                "mounting or the driver's head position is wrong, not the calibration"
                % (float(yaw), MAX_ABS_YAW_DEG)
            )
        if abs(float(pitch)) > MAX_ABS_PITCH_DEG:
            problems.append(
                "forward_zero.pitch = %.1f deg, more than %.0f deg off centre"
                % (float(pitch), MAX_ABS_PITCH_DEG)
            )
    if ear is None:
        problems.append("state.ear_open_median is missing")
    else:
        lo, hi = EAR_OPEN_RANGE
        if not (lo <= float(ear) <= hi):
            problems.append(
                "state.ear_open_median = %.3f is outside the plausible open-eye range "
                "%.2f-%.2f; the driver's eyes were probably closed during the capture"
                % (float(ear), lo, hi)
            )
    if samples is not None and int(samples) < int(min_samples):
        problems.append(
            "only %d frames contributed a pose and an EAR (need %d) - the face was not "
            "being detected reliably" % (int(samples), int(min_samples))
        )
    return problems


def main(argv=None):
    # type: (object) -> int
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--duration-s", type=float, default=5.0,
                    help="capture window; converted to --max-frames using source.fps")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="explicit frame count; overrides --duration-s")
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument("--source", default="", help="override source.path (lab replay only)")
    ap.add_argument("--force", action="store_true",
                    help="keep the new vehicle.yaml even if a check fails")
    ap.add_argument("--log-level", default="INFO")
    args, passthrough = ap.parse_known_args(argv)

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    if not os.path.isfile(cfg_path):
        sys.stderr.write("no such config: %s\n" % cfg_path)
        return 2

    from dms.config.schema import load_config

    try:
        cfg = load_config(cfg_path)
    except Exception as exc:  # noqa: BLE001 - a bad config is a user error, report it
        sys.stderr.write("bad config %s: %s\n" % (cfg_path, exc))
        return 2

    if cfg.source.type in ("file", "test") and not args.source:
        sys.stderr.write(
            "WARNING: source.type=%s. A forward zero measured from a replay clip describes "
            "that clip's camera, not this vehicle's. Calibrate from the installed camera.\n"
            % cfg.source.type
        )

    frames = args.max_frames or max(1, int(round(args.duration_s * max(1, cfg.source.fps))))

    out_path = vehicle_yaml_path(cfg_path)
    backup = ""
    previous = None
    if os.path.isfile(out_path):
        import yaml

        with open(out_path) as f:
            previous = yaml.safe_load(f) or {}
        backup = "%s.%s.bak" % (out_path, time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(out_path, backup)
        print("previous calibration saved to %s" % backup)
    before_mtime = os.path.getmtime(out_path) if os.path.isfile(out_path) else -1.0

    argv_app = ["--detect", "--calibrate-forward", "--config", cfg_path,
                "--max-frames", str(frames), "--log-level", args.log_level]
    if args.source:
        argv_app.extend(["--source", args.source])
    argv_app.extend(passthrough)

    print("capturing %d frames (~%.1f s at %d fps) - look straight ahead"
          % (frames, frames / float(max(1, cfg.source.fps)), cfg.source.fps))

    from dms.app import main as app_main

    rc = app_main(argv_app)
    if rc != 0:
        sys.stderr.write("dms.app exited %d; calibration not written\n" % rc)
        return 1

    if not os.path.isfile(out_path) or os.path.getmtime(out_path) <= before_mtime:
        sys.stderr.write(
            "no calibration was written to %s. The app only writes one when at least one "
            "frame produced a pose and an EAR - the face was never detected.\n" % out_path
        )
        return 3

    import yaml

    with open(out_path) as f:
        veh = yaml.safe_load(f) or {}
    problems = check_result(veh, min_samples=args.min_samples)

    fz = veh.get("forward_zero") or {}
    st = veh.get("state") or {}
    print("")
    print("forward_zero.yaw    = %s deg" % fz.get("yaw"))
    print("forward_zero.pitch  = %s deg" % fz.get("pitch"))
    print("state.ear_open_median = %s" % st.get("ear_open_median"))
    if previous:
        pfz = previous.get("forward_zero") or {}
        if pfz.get("yaw") is not None and fz.get("yaw") is not None:
            print("delta from previous: yaw %+.1f deg, pitch %+.1f deg"
                  % (float(fz["yaw"]) - float(pfz["yaw"]),
                     float(fz.get("pitch", 0.0)) - float(pfz.get("pitch", 0.0))))
    print("")

    if not problems:
        print("OK  %s" % out_path)
        return 0

    for p in problems:
        sys.stderr.write("CHECK FAILED: %s\n" % p)
    if args.force:
        sys.stderr.write("--force given: keeping the new calibration anyway\n")
        return 0
    if backup:
        shutil.copy2(backup, out_path)
        sys.stderr.write("restored the previous calibration from %s\n" % backup)
    else:
        os.remove(out_path)
        sys.stderr.write("removed %s; this vehicle has no calibration\n" % out_path)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
