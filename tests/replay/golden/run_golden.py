#!/usr/bin/env python3
"""Golden replay regression runner.

Phase 1 of the project plan calls for the local clips to be replayed on every
change with EXPECTED alert counts, so that a perception or state-machine change
that silently stops producing MICROSLEEP, or starts producing a hundred
GAZE_AWAYs, fails loudly instead of being noticed months later.

What a golden records
---------------------
Per clip, from a fixed number of frames of a fixed profile:

``frames``           frames the app processed (exact; several clips are shorter
                     than ``max_frames`` and stop at EOS, so the OBSERVED count
                     is what is recorded)
``face_hits``        total detections summed over the run (a band)
``alerts``           per ``AlertType``, the number of ``enter`` / ``escalate`` /
                     ``exit`` edges in ``events.jsonl`` (a band each)
``detector_engine``  (schema 3) the face engine the profile selected when the
                     golden was recorded.  A golden is only a baseline for the
                     detector it was taken with, and swapping the detector under
                     a recorded golden is exactly how a 15.8% recall regression
                     shipped: ``tests/test_app_integration.py`` compares this
                     field against what the profile selects TODAY, with no GPU
                     and no clip, so that swap now fails the ordinary pytest run.
``gate``             (reference only) the ``PERF_GATE`` verdict of the run, so a
                     batch replay shows whether each clip met its own latency
                     gate.  Never asserted - see the latency note below.

Every count is a BAND, ``[min, max]``, not a number.  The pipeline is
deterministic in its arithmetic - the state machine runs on media time taken
from the GStreamer buffer PTS, so the same clip produces the same intervals -
but the loop is not perfectly deterministic at its edges: a read timeout at the
end of a clip can add one extra empty tick, and the tick that straddles the
instant a dwell completes can land either side of the threshold.  A band of
+/- ``TOLERANCE`` (with a floor of ``FLOOR`` events) absorbs that without
absorbing a real regression, which is a change of KIND (an alert type that
appears or disappears) or of ORDER OF MAGNITUDE.

What a golden is NOT
--------------------
It is not ground truth.  The nine replay clips are dashcam montages, not cabin
footage (see ``tests/replay/CLIPS.md``); nobody has labelled a single real
microsleep in them.  These numbers pin the BEHAVIOUR OF THE SOFTWARE against
the behaviour it had when they were recorded.  Read a change here as "something
moved, come and look", never as "the system got less accurate".

Latency is recorded for reference and never asserted: this board's GPU governor
has been observed to move an identical inference by 5x, so a latency assertion
would fail on a busy board and pass on an idle one.  Measured 2026-09-13, n=3
per profile: the run-to-run e2e p95 spread is 7-16 ms on the nine clips, against
a 66 ms gate - so the recorded ``gate`` verdict is one observation, not a
property.  Read it, do not gate on it.

Usage
-----
    # normally you do not run this by hand: pytest runs it, one test per clip,
    # behind DMS_REPLAY_TESTS=1 (see tests/test_app_integration.py).  It is
    # gated because it needs the GPU and the whole GPU is shared:
    flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. \\
        ./.venv/bin/python -m pytest -q tests/test_app_integration.py"

    # check every clip that has a golden (this is the regression gate)
    flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python \\
        tests/replay/golden/run_golden.py"

    # re-record after a deliberate behaviour change, and READ THE DIFF
    flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python \\
        tests/replay/golden/run_golden.py --record"

Exit codes: 0 all clips within band, 1 at least one regression, 2 a run failed
to execute (missing clip, missing engine, non-zero exit from ``dms.app``).
"""
from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPLAY = os.path.join(ROOT, "tests", "replay")

SCHEMA = "dms.replay_golden/3"

#: Fractional half-width of a recorded band.
TOLERANCE = 0.25
#: Absolute half-width floor, in events.  A type that fired 3 times may fire
#: 2 or 4 without that being news.
FLOOR = 1

#: Clips in the golden set, with the profile each is replayed under and how many
#: frames to process.  300 frames is ten seconds of media at 30 fps - long
#: enough for the 1.5 s microsleep and 2 s gaze dwells to complete, short enough
#: that the whole set runs in a few minutes on a contended board.
CLIPS = [
    ("day_driver", "configs/default.yaml", 300),
    ("minivan", "configs/minivan.yaml", 300),
    ("red_car_crash", "configs/red_car_crash.yaml", 300),
    ("family_vacation_crash", "configs/family_vacation_crash.yaml", 300),
    ("rear_end_accident", "configs/rear_end_accident.yaml", 300),
    ("guy_rear_ended", "configs/guy_rear_ended.yaml", 300),
    ("bmw_rear_ended", "configs/bmw_rear_ended.yaml", 300),
    ("getting_rear_ended", "configs/getting_rear_ended.yaml", 300),
    ("rear_end_whiplash", "configs/rear_end_whiplash.yaml", 300),
]

RE_GATE = re.compile(
    r"PERF_GATE (?P<verdict>PASS|FAIL|UNDECIDED) e2e_p95=(?P<p95>[0-9.]+) ms "
    r"limit=(?P<limit>[0-9.]+) ms"
)

RE_FINAL = re.compile(
    r"stopped frames=(?P<frames>\d+) last_id=(?P<last_id>\d+) "
    r"mean_infer_ms=(?P<mean>[0-9.]+) total_face_hits=(?P<hits>\d+) "
    r"e2e_p50_ms=(?P<p50>[0-9.]+) e2e_p95_ms=(?P<p95>[0-9.]+)"
)


def golden_path(stem):
    return os.path.join(HERE, "%s.json" % stem)


def profile_detector_engine(config):
    """The face engine ``config`` selects right now, or ``None`` if unreadable.

    Kept tolerant on purpose: this module is imported by pytest off-board, where
    ``dms.config`` and its dependencies may not be importable.  ``None`` means
    "could not tell", and every caller treats that as "do not judge".
    """
    try:
        from dms.config.schema import load_config
    except Exception:  # pragma: no cover - depends on the environment
        return None
    try:
        return load_config(os.path.join(ROOT, config)).models.face.engine
    except Exception:  # pragma: no cover - a broken profile fails elsewhere
        return None


def band(value):
    """``[min, max]`` around an observed count."""
    half = max(FLOOR, int(round(abs(value) * TOLERANCE)))
    return [max(0, int(value) - half), int(value) + half]


def in_band(value, bounds):
    return bounds[0] <= value <= bounds[1]


def run_clip(stem, config, frames, python, timeout_s):
    """Run one clip and return ``(observation, log_text)``.

    Raises ``RuntimeError`` when the child could not produce a usable run.
    """
    clip = os.path.join("tests", "replay", "%s.mp4" % stem)
    if not os.path.isfile(os.path.join(ROOT, clip)):
        raise RuntimeError("clip missing: %s" % clip)
    events = os.path.join("/tmp", "golden_%s_%d.jsonl" % (stem, os.getpid()))
    if os.path.exists(events):
        os.unlink(events)
    cmd = [
        python,
        "-m",
        "dms.app",
        "--config",
        config,
        "--source",
        clip,
        "--max-frames",
        str(frames),
        "--events",
        events,
        "--log-level",
        "INFO",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        out = proc.communicate(timeout=timeout_s)[0].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("%s: timed out after %.0f s" % (stem, timeout_s))
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise RuntimeError("%s: dms.app exited %d\n%s" % (stem, proc.returncode, tail))

    m = RE_FINAL.search(out)
    if m is None:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise RuntimeError("%s: no summary line in the log\n%s" % (stem, tail))

    alerts = {}
    if os.path.isfile(events):
        for line in open(events):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") != "alert":
                continue
            key = str(rec.get("type"))
            edge = str(rec.get("edge"))
            alerts.setdefault(key, {}).setdefault(edge, 0)
            alerts[key][edge] += 1
        os.unlink(events)

    g = RE_GATE.search(out)
    obs = {
        "frames": int(m.group("frames")),
        "face_hits": int(m.group("hits")),
        "alerts": alerts,
        "detector_engine": profile_detector_engine(config),
        "gate": None if g is None else {
            "verdict": g.group("verdict"),
            "e2e_p95_ms": float(g.group("p95")),
            "limit_ms": float(g.group("limit")),
        },
        "latency_ms": {
            "mean_infer": float(m.group("mean")),
            "e2e_p50": float(m.group("p50")),
            "e2e_p95": float(m.group("p95")),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    return obs, out


def to_golden(stem, config, frames, obs):
    alerts = {}
    for atype, edges in sorted(obs["alerts"].items()):
        alerts[atype] = dict((edge, band(n)) for edge, n in sorted(edges.items()))
    return {
        "schema": SCHEMA,
        "clip": "tests/replay/%s.mp4" % stem,
        "config": config,
        "max_frames": frames,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The OBSERVED count, not the requested one: several clips are shorter
        # than max_frames and stop at EOS.
        "frames": int(obs["frames"]),
        "face_hits": band(obs["face_hits"]),
        "alerts": alerts,
        # Schema 3.  A golden is a baseline for ONE detector; see the header.
        "detector_engine": obs.get("detector_engine"),
        "gate": obs.get("gate"),
        "reference_latency_ms": obs["latency_ms"],
        "note": (
            "Bands pin software behaviour, not ground truth: these clips are dashcam "
            "montages with no labelled drowsiness events. Latency is reference only - "
            "this board's GPU governor moves an identical inference by up to 5x."
        ),
    }


def compare(stem, golden, obs):
    """Return a list of human-readable regression strings (empty = pass)."""
    bad = []
    want_engine = golden.get("detector_engine")
    got_engine = obs.get("detector_engine")
    if want_engine and got_engine and want_engine != got_engine:
        bad.append(
            "%s: golden was recorded against %s, the profile now selects %s - "
            "re-record deliberately or put the detector back"
            % (stem, os.path.basename(want_engine), os.path.basename(got_engine))
        )
    if obs["frames"] != golden["frames"]:
        bad.append(
            "%s: processed %d frames, golden says %d"
            % (stem, obs["frames"], golden["frames"])
        )
    if not in_band(obs["face_hits"], golden["face_hits"]):
        bad.append(
            "%s: face_hits %d outside %s"
            % (stem, obs["face_hits"], golden["face_hits"])
        )
    want = golden.get("alerts", {})
    got = obs.get("alerts", {})
    for atype in sorted(set(want) | set(got)):
        w = want.get(atype, {})
        g = got.get(atype, {})
        for edge in sorted(set(w) | set(g)):
            n = g.get(edge, 0)
            bounds = w.get(edge)
            if bounds is None:
                bad.append(
                    "%s: NEW alert %s/%s fired %d times (not in the golden)"
                    % (stem, atype, edge, n)
                )
            elif not in_band(n, bounds):
                bad.append(
                    "%s: %s/%s fired %d times, golden band %s"
                    % (stem, atype, edge, n, bounds)
                )
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="write new goldens")
    ap.add_argument("--only", action="append", default=[], help="clip stem; repeatable")
    ap.add_argument(
        "--python",
        default=os.path.join(ROOT, ".venv", "bin", "python"),
        help="interpreter used to run dms.app",
    )
    ap.add_argument("--timeout-s", type=float, default=420.0)
    ap.add_argument("--keep-going", action="store_true", help="do not stop on a failed run")
    args = ap.parse_args(argv)

    python = args.python if os.path.isfile(args.python) else sys.executable
    clips = [c for c in CLIPS if not args.only or c[0] in args.only]
    if not clips:
        print("no clips selected", file=sys.stderr)
        return 2

    failures = []
    errors = []
    rows = []
    for stem, config, frames in clips:
        try:
            obs, _log = run_clip(stem, config, frames, python, args.timeout_s)
        except RuntimeError as exc:
            errors.append(str(exc))
            print("ERROR %s" % exc, file=sys.stderr)
            if args.keep_going:
                continue
            return 2

        alert_summary = ",".join(
            "%s:%s" % (a, sum(e.values())) for a, e in sorted(obs["alerts"].items())
        )
        gate = obs.get("gate") or {}
        rows.append(
            "%-24s frames=%-4d faces=%-5d p50=%-6.1f p95=%-6.1f gate=%-9s %s"
            % (
                stem,
                obs["frames"],
                obs["face_hits"],
                obs["latency_ms"]["e2e_p50"],
                obs["latency_ms"]["e2e_p95"],
                gate.get("verdict", "?"),
                alert_summary or "-",
            )
        )
        print(rows[-1])

        path = golden_path(stem)
        if args.record:
            with open(path, "w") as f:
                json.dump(to_golden(stem, config, frames, obs), f, indent=2, sort_keys=True)
                f.write("\n")
            print("  recorded %s" % os.path.relpath(path, ROOT))
            continue
        if not os.path.isfile(path):
            errors.append("%s: no golden at %s (run with --record)" % (stem, path))
            print("  MISSING GOLDEN", file=sys.stderr)
            continue
        with open(path) as f:
            golden = json.load(f)
        bad = compare(stem, golden, obs)
        for line in bad:
            print("  REGRESSION %s" % line, file=sys.stderr)
        failures.extend(bad)
        if not bad:
            print("  ok")

    print("")
    if args.record:
        print("recorded %d golden(s)" % len(rows))
        return 0 if not errors else 2
    if errors:
        print("%d run error(s)" % len(errors), file=sys.stderr)
        return 2
    if failures:
        print("%d regression(s)" % len(failures), file=sys.stderr)
        return 1
    print("all %d clip(s) within their golden bands" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
