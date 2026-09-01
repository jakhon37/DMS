#!/usr/bin/env python3
"""ONNX -> TensorRT FP16 via trtexec. Parses [DLA]/[GPU] in verbose logs."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")


def parse_dla_fraction(log_text: str) -> float:
    dla = len(re.findall(r"\[DLA\]", log_text))
    gpu = len(re.findall(r"\[GPU\]", log_text))
    total = dla + gpu
    if total == 0:
        return 0.0
    return dla / float(total)


def build(onnx: str, engine: str, *, dla: int | None, workspace: str) -> None:
    cmd = [
        TRTEXEC,
        "--onnx=" + onnx,
        "--saveEngine=" + engine,
        "--fp16",
        "--memPoolSize=workspace:" + workspace,
        "--verbose",
    ]
    if dla is not None:
        cmd.extend(["--useDLACore=" + str(dla), "--allowGPUFallback"])
    print(" ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path = engine + ".build.log"
    with open(log_path, "w") as f:
        f.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        raise SystemExit("trtexec failed for %s" % onnx)
    frac = parse_dla_fraction(proc.stdout)
    print("dla_layer_fraction=%.3f log=%s" % (frac, log_path))
    if dla is not None and frac < 0.80:
        print("WARN: DLA fallback >20%% GPU; discard DLA engine per design R2")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", default=os.path.join(ROOT, "engines"))
    args = p.parse_args()
    face = os.path.join(args.onnx_dir, "scrfd_500m_kps.onnx")
    lmk = os.path.join(args.onnx_dir, "2d106det.onnx")
    if not os.path.isfile(face) or not os.path.isfile(lmk):
        print("missing ONNX; run scripts/fetch_onnx.sh", file=sys.stderr)
        return 2
    os.makedirs(args.onnx_dir, exist_ok=True)
    build(face, os.path.join(args.onnx_dir, "scrfd_500m_640_fp16_gpu.engine"), dla=None, workspace="512M")
    build(lmk, os.path.join(args.onnx_dir, "2d106det_192_fp16_gpu.engine"), dla=None, workspace="64M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
