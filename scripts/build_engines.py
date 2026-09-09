#!/usr/bin/env python3
"""ONNX -> TensorRT FP16 via trtexec. Live progress; parses [DLA]/[GPU] in logs.

Run with:  python3 scripts/build_engines.py
Not:       bash scripts/build_engines.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from typing import List, Optional, TextIO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")

# trtexec is chatty; these lines are the useful milestones.
_PHASE = re.compile(
    r"(Parsing ONNX|Building engine|Engine built|Starting engine|"
    r"Timing|Throughput|Latency|&&&& PASSED|&&&& FAILED|"
    r"\[E\]|error|Error|Tactic:|Selected tactics|"
    r"FP16|Int8|DLA|finish emitted)",
    re.IGNORECASE,
)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_dla_fraction(log_text: str) -> float:
    dla = len(re.findall(r"\[DLA\]", log_text))
    gpu = len(re.findall(r"\[GPU\]", log_text))
    total = dla + gpu
    if total == 0:
        return 0.0
    return dla / float(total)


def _pump(
    proc: subprocess.Popen,
    log_fh: TextIO,
    *,
    raw: bool,
    label: str,
    stop_hb: threading.Event,
    last_line: List[float],
) -> str:
    chunks: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        last_line[0] = time.monotonic()
        log_fh.write(line)
        log_fh.flush()
        chunks.append(line)
        shown = raw or bool(_PHASE.search(line))
        if shown:
            sys.stdout.write("[%s] %s | %s" % (_ts(), label, line))
            sys.stdout.flush()
    return "".join(chunks)


def _heartbeat(label: str, t0: float, last_line: List[float], stop: threading.Event) -> None:
    while not stop.wait(15.0):
        elapsed = int(time.monotonic() - t0)
        silent = int(time.monotonic() - last_line[0])
        sys.stdout.write(
            "[%s] %s | still running  elapsed=%ds  last_log=%ds ago "
            "(tactic search can be quiet for a few minutes)\n"
            % (_ts(), label, elapsed, silent)
        )
        sys.stdout.flush()


def build(
    onnx: str,
    engine: str,
    *,
    dla: Optional[int],
    workspace: str,
    index: int,
    total: int,
    raw: bool,
) -> None:
    label = "%d/%d %s" % (index, total, os.path.basename(engine))
    cmd = [
        TRTEXEC,
        "--onnx=" + onnx,
        "--saveEngine=" + engine,
        "--fp16",
        "--memPoolSize=workspace:" + workspace,
        "--verbose",
    ]
    # UniFace SCRFD ONNX is dynamic HxW; without profiles TRT baked 1x1 (unusable).
    if os.path.basename(onnx).startswith("scrfd"):
        cmd.extend(
            [
                "--minShapes=input.1:1x3x640x640",
                "--optShapes=input.1:1x3x640x640",
                "--maxShapes=input.1:1x3x640x640",
            ]
        )
    if dla is not None:
        cmd.extend(["--useDLACore=" + str(dla), "--allowGPUFallback"])

    print("[%s] === %s ===" % (_ts(), label))
    print("[%s] onnx      %s (%.1f MB)" % (_ts(), onnx, os.path.getsize(onnx) / 1e6))
    print("[%s] engine    %s" % (_ts(), engine))
    print("[%s] workspace %s  fp16  dla=%s" % (_ts(), workspace, dla))
    print("[%s] cmd       %s" % (_ts(), " ".join(cmd)))
    sys.stdout.flush()

    log_path = engine + ".build.log"
    t0 = time.monotonic()
    last_line = [t0]
    stop_hb = threading.Event()
    hb = threading.Thread(
        target=_heartbeat, args=(label, t0, last_line, stop_hb), daemon=True
    )
    hb.start()

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            text = _pump(proc, log_fh, raw=raw, label=label, stop_hb=stop_hb, last_line=last_line)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        rc = proc.wait()

    stop_hb.set()
    elapsed = time.monotonic() - t0
    size = os.path.getsize(engine) if os.path.isfile(engine) else 0
    print("[%s] %s | finished rc=%s  %.1fs  engine=%.1f MB  log=%s" % (
        _ts(), label, rc, elapsed, size / 1e6, log_path
    ))
    if rc != 0:
        print("[%s] last 30 log lines:" % _ts(), file=sys.stderr)
        sys.stderr.write("".join(text.splitlines(True)[-30:]))
        raise SystemExit("trtexec failed for %s" % onnx)
    try:
        from dms.runtime.manifest import record_engine_sha256

        record_engine_sha256(engine)
    except Exception as exc:
        print("[%s] WARN: MANIFEST update failed: %s" % (_ts(), exc))
    frac = parse_dla_fraction(text)
    print("[%s] %s | dla_layer_fraction=%.3f" % (_ts(), label, frac))
    if dla is not None and frac < 0.80:
        print("[%s] WARN: DLA fallback >20%% GPU; discard DLA engine per design R2" % _ts())
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(description="Build TensorRT FP16 engines (live log).")
    p.add_argument("--onnx-dir", default=os.path.join(ROOT, "engines"))
    p.add_argument("--raw", action="store_true", help="print every trtexec line (very noisy)")
    args = p.parse_args()
    if not os.path.isfile(TRTEXEC):
        print("trtexec not found: %s" % TRTEXEC, file=sys.stderr)
        return 2
    face = os.path.join(args.onnx_dir, "scrfd_500m_kps.onnx")
    lmk = os.path.join(args.onnx_dir, "2d106det.onnx")
    if not os.path.isfile(face) or not os.path.isfile(lmk):
        print("missing ONNX; run: bash scripts/fetch_onnx.sh", file=sys.stderr)
        return 2
    os.makedirs(args.onnx_dir, exist_ok=True)
    jobs = [
        (face, os.path.join(args.onnx_dir, "scrfd_500m_640_fp16_gpu.engine"), None, "512M"),
        (lmk, os.path.join(args.onnx_dir, "2d106det_192_fp16_gpu.engine"), None, "64M"),
    ]
    print("[%s] building %d engines with %s" % (_ts(), len(jobs), TRTEXEC))
    for i, (onnx, engine, dla, ws) in enumerate(jobs, 1):
        build(onnx, engine, dla=dla, workspace=ws, index=i, total=len(jobs), raw=args.raw)
    print("[%s] all engines ok" % _ts())
    return 0


if __name__ == "__main__":
    sys.exit(main())
