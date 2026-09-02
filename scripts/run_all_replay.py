#!/usr/bin/env python3
"""Run every tests/replay/*.mp4 through dms.app --detect into one run folder.

Sequential on purpose: Xavier NX cannot hold several SCRFD+106 TRT sessions.

  PYTHONPATH=. python3.8 scripts/run_all_replay.py
  PYTHONPATH=. python3.8 scripts/run_all_replay.py --max-frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# day_driver lives in default.yaml, not configs/day_driver.yaml.
CONFIG_BY_STEM = {
    "day_driver": "configs/default.yaml",
}


def config_for_clip(mp4_path):
    stem = os.path.splitext(os.path.basename(mp4_path))[0]
    if stem in CONFIG_BY_STEM:
        return CONFIG_BY_STEM[stem]
    named = os.path.join("configs", stem + ".yaml")
    if os.path.isfile(os.path.join(ROOT, named)):
        return named
    return "configs/default.yaml"


def list_clips(clips_dir):
    names = []
    for name in sorted(os.listdir(clips_dir)):
        if name.endswith(".mp4"):
            names.append(os.path.join(clips_dir, name))
    return names


def _run_one(cmd, log_path, env):
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        return proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch-run all replay mp4s into a new folder")
    parser.add_argument("--clips-dir", default=os.path.join(ROOT, "tests/replay"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "tests/replay/runs"))
    parser.add_argument("--max-frames", type=int, default=0, help="0 = until file EOS")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    clips = list_clips(args.clips_dir)
    if not clips:
        sys.stderr.write("no mp4 files in %s\n" % args.clips_dir)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    print("run_dir %s clips=%d" % (run_dir, len(clips)))
    results = []
    t_all = time.monotonic()
    for mp4 in clips:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        clip_dir = os.path.join(run_dir, stem)
        os.makedirs(clip_dir, exist_ok=True)
        cfg = config_for_clip(mp4)
        cmd = [
            args.python,
            "-m",
            "dms.app",
            "--config",
            cfg,
            "--source",
            mp4,
            "--detect",
            "--save-preview",
            os.path.join(clip_dir, "preview.jpg"),
            "--save-video",
            os.path.join(clip_dir, "overlay.avi"),
            "--events",
            os.path.join(clip_dir, "events.jsonl"),
            "--log-level",
            "INFO",
        ]
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        print("=== %s config=%s ===" % (stem, cfg))
        t0 = time.monotonic()
        rc = _run_one(cmd, os.path.join(clip_dir, "log.txt"), env)
        dt = time.monotonic() - t0
        rec = {"clip": stem, "config": cfg, "rc": rc, "seconds": round(dt, 1)}
        results.append(rec)
        print("=== %s rc=%s in %.1fs ===" % (stem, rc, dt))

    summary = {
        "run_dir": run_dir,
        "seconds": round(time.monotonic() - t_all, 1),
        "failed": [r["clip"] for r in results if r["rc"] != 0],
        "clips": results,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("wrote %s failed=%s" % (os.path.join(run_dir, "summary.json"), summary["failed"]))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
