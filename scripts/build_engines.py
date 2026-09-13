#!/usr/bin/env python3
"""ONNX -> TensorRT FP16 engine builder, driven by engines/MANIFEST.json.

Run with:  PYTHONPATH=. python3 scripts/build_engines.py
Not:       bash scripts/build_engines.py

Every engine this project loads must be **fully static**: dms/runtime/trt_engine.py
refuses any binding whose shape contains a -1 ("dynamic shape on %s - v1 requires
static engines"). Several upstream ONNX files ship a dynamic batch axis (the PINTO
classifiers) or dynamic H/W (SCRFD), so each build spec in MANIFEST["builds"] names
the exact shape to freeze, and after every build the engine is re-opened and checked:
a surviving -1 fails the build rather than shipping something the runtime cannot load.

Pipeline per spec
    0. parse the ONNX with tensorrt.OnnxParser and reconcile the declared input shapes
       with the shape the spec asks for. Only genuinely dynamic inputs get
       --min/opt/maxShapes: trtexec rejects explicit shapes on a fully static graph
       ("Static model does not take explicit shapes"). A static input whose declared
       shape disagrees with the spec is a configuration error and blocks the build.
    1. trtexec --onnx ... --fp16 [--min/opt/maxShapes=<frozen shape>]   (live log)
    2. re-open the engine with the TensorRT runtime, read the real binding contract
    3. if any binding is still dynamic, rebuild through the TensorRT Python API with
       the network input shapes overwritten to the frozen shape
    4. trtexec --loadEngine round trip: proves the artifact deserialises and gives the
       measured latency (ms) recorded in the manifest
    5. write file/sha256/bytes/bindings/latency back into MANIFEST["engines"]

A spec that cannot be built is not dropped silently: the trtexec error tail is written
to MANIFEST["blocked"][<id>] so the wrapper workstream ships a stub for that capability.

Units: every latency in this file and in the manifest is milliseconds, measured by
trtexec on an otherwise idle GPU at whatever clocks nvpmodel currently sets - it is a
build-time sanity number, not the in-pipeline budget. device_memory_bytes is the
TensorRT execution-context scratch requirement in bytes, excluding the weights.

Failure behaviour: any build failure raises SystemExit(1) after recording the blocked
entry, unless --keep-going is given, in which case remaining specs still build and the
exit code is 1 at the end. The manifest is written back after every spec, so an
interrupted run leaves a consistent file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")
ENGINE_DIR = os.path.join(ROOT, "engines")
MANIFEST_PATH = os.path.join(ENGINE_DIR, "MANIFEST.json")

# trtexec is chatty; these lines are the useful milestones.
_PHASE = re.compile(
    r"(Parsing ONNX|Building engine|Engine built|Starting engine|"
    r"Timing|Throughput|Latency|&&&& PASSED|&&&& FAILED|"
    r"\[E\]|error|Error|Tactic:|Selected tactics|"
    r"FP16|Int8|DLA|finish emitted)",
    re.IGNORECASE,
)
_LAT_MEDIAN = re.compile(r"Latency:.*median = ([0-9.]+) ms", re.IGNORECASE)
_LAT_P99 = re.compile(r"Latency:.*percentile\(99(?:\.00)?%\) = ([0-9.]+) ms", re.IGNORECASE)
_THROUGHPUT = re.compile(r"Throughput: ([0-9.]+) qps", re.IGNORECASE)


def _ts():
    # type: () -> str
    return time.strftime("%H:%M:%S")


def parse_dla_fraction(log_text):
    # type: (str) -> float
    dla = len(re.findall(r"\[DLA\]", log_text))
    gpu = len(re.findall(r"\[GPU\]", log_text))
    total = dla + gpu
    if total == 0:
        return 0.0
    return dla / float(total)


def avail_mb():
    # type: () -> int
    """MemAvailable in MiB, or -1 when /proc/meminfo is unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (IOError, OSError, ValueError):
        pass
    return -1


def _pump(proc, log_fh, raw, label, last_line):
    # type: (subprocess.Popen, TextIO, bool, str, List[float]) -> str
    chunks = []  # type: List[str]
    assert proc.stdout is not None
    for line in proc.stdout:
        last_line[0] = time.monotonic()
        log_fh.write(line)
        log_fh.flush()
        chunks.append(line)
        if raw or bool(_PHASE.search(line)):
            sys.stdout.write("[%s] %s | %s" % (_ts(), label, line))
            sys.stdout.flush()
    return "".join(chunks)


def _heartbeat(label, t0, last_line, stop):
    # type: (str, float, List[float], threading.Event) -> None
    while not stop.wait(15.0):
        elapsed = int(time.monotonic() - t0)
        silent = int(time.monotonic() - last_line[0])
        sys.stdout.write(
            "[%s] %s | still running  elapsed=%ds  last_log=%ds ago "
            "(tactic search can be quiet for a few minutes)\n"
            % (_ts(), label, elapsed, silent)
        )
        sys.stdout.flush()


def _run_logged(cmd, log_path, label, raw, append=False):
    # type: (Sequence[str], str, str, bool, bool) -> Tuple[int, str, float]
    """Run a trtexec command, streaming milestones. Returns (rc, full_log, seconds)."""
    t0 = time.monotonic()
    last_line = [t0]
    stop_hb = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(label, t0, last_line, stop_hb), daemon=True)
    hb.start()
    try:
        with open(log_path, "a" if append else "w") as log_fh:
            log_fh.write("\n$ %s\n" % " ".join(cmd))
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            try:
                text = _pump(proc, log_fh, raw, label, last_line)
            except KeyboardInterrupt:
                proc.terminate()
                raise
            rc = proc.wait()
    finally:
        stop_hb.set()
    return rc, text, time.monotonic() - t0


# --------------------------------------------------------------------------------------
# engine introspection
# --------------------------------------------------------------------------------------


def onnx_input_shapes(onnx_path):
    # type: (str) -> Dict[str, List[int]]
    """Declared input shapes of an ONNX graph, read with TensorRT's own parser.

    The `onnx` python package is forbidden on this board, so the ONNX is parsed into a
    throwaway TensorRT network purely to enumerate its inputs. A -1 in the returned shape
    is a dynamic axis. Raises RuntimeError if the graph does not parse.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("OnnxParser failed on %s: %s" % (onnx_path, errs))
    shapes = {}  # type: Dict[str, List[int]]
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shapes[t.name] = [int(d) for d in t.shape]
    del parser, network, builder
    return shapes


def reconcile_shapes(spec_shapes, onnx_shapes, onnx_name):
    # type: (Dict[str, List[int]], Dict[str, List[int]], str) -> Tuple[Dict[str, List[int]], List[str]]
    """Split the requested shapes into (shapes trtexec must be told, problems).

    trtexec rejects --minShapes on a graph with no dynamic axis at all ("Static model does
    not take explicit shapes"), so a fully static ONNX must be built with no shape args.
    A static ONNX whose declared shape disagrees with the build spec is a configuration
    error, not something to silently build at the wrong size.
    """
    profile = {}  # type: Dict[str, List[int]]
    problems = []  # type: List[str]
    for name, want in sorted(spec_shapes.items()):
        have = onnx_shapes.get(name)
        if have is None:
            problems.append(
                "%s: build spec names input %r but the graph inputs are %s"
                % (onnx_name, name, sorted(onnx_shapes))
            )
            continue
        if len(have) != len(want):
            problems.append(
                "%s: input %s has rank %d in the graph, spec asks for rank %d"
                % (onnx_name, name, len(have), len(want))
            )
            continue
        if any(d < 0 for d in have):
            profile[name] = list(want)
            continue
        if list(have) != list(want):
            problems.append(
                "%s: input %s is statically %s in the graph but the spec asks for %s"
                % (onnx_name, name, have, list(want))
            )
    for name, have in sorted(onnx_shapes.items()):
        if any(d < 0 for d in have) and name not in profile:
            problems.append(
                "%s: input %s is dynamic (%s) and the build spec names no shape for it"
                % (onnx_name, name, have)
            )
    return profile, problems


def engine_bindings(engine_path):
    # type: (str) -> Dict[str, Any]
    """Deserialise an engine and return its real binding contract.

    Returns {"inputs": [{name, dtype, shape}], "outputs": [...],
             "device_memory_bytes": int, "dynamic": bool}.
    Raises RuntimeError if the blob does not deserialise.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("failed to deserialize %s" % engine_path)
    inputs = []  # type: List[Dict[str, Any]]
    outputs = []  # type: List[Dict[str, Any]]
    dynamic = False
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [int(x) for x in engine.get_tensor_shape(name)]
        if any(s < 0 for s in shape):
            dynamic = True
        rec = {
            "name": name,
            "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
            "shape": shape,
        }
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(rec)
        else:
            outputs.append(rec)
    info = {
        "inputs": inputs,
        "outputs": outputs,
        "device_memory_bytes": int(engine.device_memory_size),
        "dynamic": dynamic,
    }
    del engine, runtime
    return info


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------


def _shape_args(shapes):
    # type: (Dict[str, Sequence[int]]) -> List[str]
    """min == opt == max for every named input: freezes the profile to one shape."""
    args = []  # type: List[str]
    for flag in ("--minShapes=", "--optShapes=", "--maxShapes="):
        spec = ",".join(
            "%s:%s" % (name, "x".join(str(int(d)) for d in dims))
            for name, dims in sorted(shapes.items())
        )
        args.append(flag + spec)
    return args


def build_with_trtexec(onnx, engine, shapes, workspace, dla, label, raw):
    # type: (str, str, Dict[str, Sequence[int]], str, Optional[int], str, bool) -> Tuple[int, str, float]
    cmd = [
        TRTEXEC,
        "--onnx=" + onnx,
        "--saveEngine=" + engine,
        "--fp16",
        "--memPoolSize=workspace:" + workspace,
        "--verbose",
    ]
    if shapes:
        cmd.extend(_shape_args(shapes))
    if dla is not None:
        cmd.extend(["--useDLACore=" + str(dla), "--allowGPUFallback"])
    return _run_logged(cmd, engine + ".build.log", label, raw)


def _workspace_bytes(workspace):
    # type: (str) -> int
    m = re.match(r"^([0-9]+)([KMG]?)$", str(workspace).strip().upper())
    if not m:
        raise ValueError("bad workspace spec %r (expected e.g. 512M)" % workspace)
    scale = {"": 1 << 20, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]
    return int(m.group(1)) * scale


def build_static_with_python_api(onnx, engine, shapes, workspace, label):
    # type: (str, str, Dict[str, Sequence[int]], str, str) -> None
    """Fallback: overwrite the network input shapes, then build with no profile.

    trtexec always attaches an optimization profile when --minShapes is given, and on
    TensorRT 8.5 a profiled engine can keep reporting -1 for the profiled axis even when
    min == opt == max. Rewriting the parsed network input shape to a constant removes the
    dynamic axis from the graph, so the serialized engine reports a concrete shape and
    dms/runtime/trt_engine.py accepts it.

    Raises RuntimeError when the ONNX does not parse, when a dynamic input is not named
    in the build spec, or when the builder returns no plan.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("OnnxParser failed on %s: %s" % (onnx, errs))
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        want = shapes.get(tensor.name)
        if want is None:
            if any(int(d) < 0 for d in tensor.shape):
                raise RuntimeError(
                    "input %s of %s is dynamic but the build spec names no shape"
                    % (tensor.name, os.path.basename(onnx))
                )
            continue
        tensor.shape = tuple(int(d) for d in want)
        print("[%s] %s | froze input %s -> %s" % (_ts(), label, tensor.name, tuple(want)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _workspace_bytes(workspace))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT builder returned no engine for %s" % onnx)
    with open(engine, "wb") as f:
        f.write(bytes(plan))
    del plan, config, parser, network, builder


def measure_engine(engine, label, raw, iterations=100):
    # type: (str, str, bool, int) -> Dict[str, Any]
    """trtexec --loadEngine round trip. Proves the artifact loads; returns latency ms."""
    cmd = [
        TRTEXEC,
        "--loadEngine=" + engine,
        "--iterations=%d" % iterations,
        "--warmUp=200",
        "--duration=0",
        "--avgRuns=10",
    ]
    rc, text, _ = _run_logged(cmd, engine + ".build.log", label + " loadEngine", raw, append=True)
    out = {"loadEngine_rc": rc}  # type: Dict[str, Any]
    med = _LAT_MEDIAN.search(text)
    p99 = _LAT_P99.search(text)
    thr = _THROUGHPUT.search(text)
    if med:
        out["latency_median_ms"] = float(med.group(1))
    if p99:
        out["latency_p99_ms"] = float(p99.group(1))
    if thr:
        out["throughput_qps"] = float(thr.group(1))
    return out


# --------------------------------------------------------------------------------------
# manifest plumbing
# --------------------------------------------------------------------------------------


def load_manifest(path=MANIFEST_PATH):
    # type: (str) -> Dict[str, Any]
    with open(path) as f:
        return json.load(f)


def save_manifest(man, path=MANIFEST_PATH):
    # type: (Dict[str, Any], str) -> None
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# Same hash the runtime checks at load time - import it rather than reimplement it, so the
# builder can never disagree with dms/runtime/manifest.py about what an engine hashes to.
from dms.runtime.manifest import sha256_file  # noqa: E402  (needs ROOT on sys.path first)


def trt_version():
    # type: () -> str
    import tensorrt as trt

    return str(trt.__version__)


def _onnx_record(man, onnx_name):
    # type: (Dict[str, Any], str) -> Dict[str, Any]
    for rec in (man.get("onnx") or {}).values():
        if isinstance(rec, dict) and rec.get("file") == onnx_name:
            return rec
    return {}


def record_success(man, spec, engine_path, bindings, perf, build_seconds):
    # type: (Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Dict[str, Any], float) -> None
    engines = man.setdefault("engines", {})
    onnx_rec = _onnx_record(man, spec["onnx"])
    entry = engines.get(spec["id"])
    if not isinstance(entry, dict):
        entry = {}
    entry.update(
        {
            "file": os.path.basename(engine_path),
            "device": "dla%d" % spec["dla"] if spec.get("dla") is not None else "gpu",
            "role": spec.get("role"),
            "precision": "fp16",
            "sha256": sha256_file(engine_path),
            "bytes": os.path.getsize(engine_path),
            "onnx": spec["onnx"],
            "onnx_sha256": onnx_rec.get("sha256"),
            "source_url": onnx_rec.get("url"),
            "license": onnx_rec.get("license"),
            "trt_version": trt_version(),
            "inputs": bindings["inputs"],
            "outputs": bindings["outputs"],
            "device_memory_bytes": bindings["device_memory_bytes"],
            "build_seconds": round(build_seconds, 1),
        }
    )
    entry.update(perf)
    engines[spec["id"]] = entry
    blocked = man.get("blocked") or {}
    blocked.pop(spec["id"], None)
    man["blocked"] = blocked


def record_blocked(man, spec, reason, log_tail):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> None
    blocked = man.setdefault("blocked", {})
    blocked[spec["id"]] = {
        "onnx": spec["onnx"],
        "engine": spec["engine"],
        "role": spec.get("role"),
        "reason": reason,
        "trt_version": trt_version(),
        "trtexec_error": log_tail.strip()[-4000:],
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    man.get("engines", {}).pop(spec["id"], None)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def _error_tail(text, lines=40):
    # type: (str, int) -> str
    errs = [ln for ln in text.splitlines() if "[E]" in ln or "Error" in ln]
    if errs:
        return "\n".join(errs[-lines:])
    return "\n".join(text.splitlines()[-lines:])


def measure_spec(spec, man, index, total, raw):
    # type: (Dict[str, Any], Dict[str, Any], int, int, bool) -> bool
    """Record the contract of an engine that is already on disk, without rebuilding it.

    Used for engines whose sha256 is already pinned and depended on elsewhere: rebuilding
    would change the hash for no benefit, but the binding contract and the measured
    latency still belong in the manifest.
    """
    engine = os.path.join(ENGINE_DIR, spec["engine"])
    label = "%d/%d %s" % (index, total, spec["id"])
    if not os.path.isfile(engine):
        print("[%s] %s | not on disk, nothing to measure" % (_ts(), label), file=sys.stderr)
        return False
    try:
        bindings = engine_bindings(engine)
    except Exception as exc:
        record_blocked(man, spec, "engine does not deserialize: %s" % exc, "")
        print("[%s] %s | DESERIALIZE FAILED %s" % (_ts(), label, exc), file=sys.stderr)
        return False
    if bindings["dynamic"]:
        still = [b for b in bindings["inputs"] + bindings["outputs"]
                 if any(s < 0 for s in b["shape"])]
        record_blocked(man, spec, "engine on disk has dynamic bindings: %s" % still, "")
        print("[%s] %s | DYNAMIC ON DISK" % (_ts(), label), file=sys.stderr)
        return False
    perf = measure_engine(engine, label, raw)
    if perf.get("loadEngine_rc") != 0:
        record_blocked(man, spec, "trtexec --loadEngine round trip failed", "")
        print("[%s] %s | LOADENGINE FAILED" % (_ts(), label), file=sys.stderr)
        return False
    prior = (man.get("engines") or {}).get(spec["id"]) or {}
    record_success(man, spec, engine, bindings, perf, float(prior.get("build_seconds") or 0.0))
    print("[%s] %s | measured  %.2f MB  devmem=%.2f MB  median=%s ms"
          % (_ts(), label, os.path.getsize(engine) / 1e6,
             bindings["device_memory_bytes"] / 1e6, perf.get("latency_median_ms")))
    for b in bindings["inputs"]:
        print("[%s]     IN  %-22s %-8s %s" % (_ts(), b["name"], b["dtype"], b["shape"]))
    for b in bindings["outputs"]:
        print("[%s]     OUT %-22s %-8s %s" % (_ts(), b["name"], b["dtype"], b["shape"]))
    return True


def verify_spec(spec, man, index, total):
    # type: (Dict[str, Any], Dict[str, Any], int, int) -> bool
    """Load the engine through the runtime's own TrtEngine and run one inference.

    This is stricter than the trtexec --loadEngine round trip the build already does:
    TrtEngine refuses a dynamic binding, allocates host/device buffers from the binding
    shapes and executes with execute_async_v3, so it exercises exactly the path the app
    takes. The input is synthetic (uniform noise in the model's own normalised range), so
    the values mean nothing - what is checked is that the engine runs and that the output
    range still matches the semantics recorded in engines[].output_semantics_probe
    (sigmoid heads inside [0,1], gaze heads NOT softmaxed, SFace NOT L2-normalised, YOLOX
    xy still raw grid offsets). A drifted claim here means the wrapper contract is wrong.
    """
    import numpy as np

    from dms.runtime.trt_engine import TrtEngine

    engine_path = os.path.join(ENGINE_DIR, spec["engine"])
    label = "%d/%d %s" % (index, total, spec["id"])
    if not os.path.isfile(engine_path):
        print("[%s] %s | not on disk" % (_ts(), label), file=sys.stderr)
        return False
    try:
        eng = TrtEngine(engine_path)
    except Exception as exc:
        print("[%s] %s | TrtEngine REFUSED IT: %s" % (_ts(), label, exc), file=sys.stderr)
        return False
    try:
        rng = np.random.RandomState(0)
        lo, hi = _probe_range(spec["id"])
        feeds = {b.name: rng.uniform(lo, hi, size=b.shape).astype(b.dtype) for b in eng._inputs}
        try:
            out = eng.infer(feeds)
        except Exception as exc:
            print("[%s] %s | INFER FAILED: %s" % (_ts(), label, exc), file=sys.stderr)
            return False
        summary = " | ".join(
            "%s%s [%.4g, %.4g]" % (n, tuple(a.shape), float(a.min()), float(a.max()))
            for n, a in sorted(out.items())
        )
        print("[%s] %s | ok  %s" % (_ts(), label, summary))
        probe = ((man.get("engines") or {}).get(spec["id"]) or {}).get("output_semantics_probe")
        problems = _check_semantics(spec["id"], out, probe)
        for p in problems:
            print("[%s] %s | SEMANTICS DRIFT %s" % (_ts(), label, p), file=sys.stderr)
        return not problems
    finally:
        eng.close()


def _probe_range(engine_id):
    # type: (str) -> Tuple[float, float]
    """Plausible input range for a synthetic probe, per the model's own normalisation."""
    if engine_id.startswith("scrfd"):
        return -1.0, 1.0  # (px - 127.5) / 128
    if engine_id.startswith("hpe") or engine_id.startswith("mobilenetv2_gaze"):
        return -2.2, 2.7  # ImageNet-normalised
    if engine_id.startswith("yolox") or engine_id.startswith("sface") or engine_id.startswith("2d106"):
        return 0.0, 255.0  # raw pixels
    return 0.0, 1.0  # px / 255


def _check_semantics(engine_id, out, probe):
    # type: (str, Dict[str, Any], Optional[Dict[str, Any]]) -> List[str]
    """Re-assert the recorded output semantics. Returns a list of drifted claims."""
    import numpy as np

    problems = []  # type: List[str]
    if not probe:
        return problems
    if probe.get("in_unit_interval"):
        arr = np.asarray(list(out.values())[0], dtype=np.float64)
        if arr.min() < 0.0 or arr.max() > 1.0:
            problems.append("output left [0,1]: [%.4g, %.4g]" % (arr.min(), arr.max()))
    if "det" in probe:
        R = np.asarray(out["rotation_matrix"], dtype=np.float64)[0]
        err = float(np.abs(R @ R.T - np.eye(3)).max())
        det = float(np.linalg.det(R))
        if err > 1e-2 or abs(det - 1.0) > 1e-2:
            problems.append("not a rotation matrix: |RRt-I|max=%.4g det=%.6f" % (err, det))
    if probe.get("objcls_sigmoid_baked_in") is not None:
        o = np.asarray(out["output"], dtype=np.float64)[0]
        if not (o[:, 4:].min() >= 0.0 and o[:, 4:].max() <= 1.0):
            problems.append("YOLOX obj/cls left [0,1] - sigmoid no longer baked in")
        if (o[:, :2].max() > 100.0) != bool(probe.get("grid_decode_baked_in")):
            problems.append("YOLOX grid-decode-baked-in flag flipped")
    if probe.get("l2_normalized_by_graph") is False:
        n = float(np.linalg.norm(np.asarray(out["fc1"], dtype=np.float64)[0]))
        if abs(n - 1.0) < 1e-3:
            problems.append("SFace output is now L2-normalised by the graph (norm=%.4f)" % n)
    if probe.get("logits_not_probabilities"):
        s = float(np.asarray(out["yaw"], dtype=np.float64).sum())
        if abs(s - 1.0) < 0.05:
            problems.append("gaze yaw now sums to 1 - the graph gained a softmax")
    return problems


def build_spec(spec, man, index, total, raw, min_avail_mb):
    # type: (Dict[str, Any], Dict[str, Any], int, int, bool, int) -> bool
    onnx = os.path.join(ENGINE_DIR, spec["onnx"])
    engine = os.path.join(ENGINE_DIR, spec["engine"])
    label = "%d/%d %s" % (index, total, spec["id"])
    shapes = {k: list(v) for k, v in (spec.get("shapes") or {}).items()}
    workspace = spec.get("workspace") or "256M"
    dla = spec.get("dla")

    if not os.path.isfile(onnx):
        record_blocked(man, spec, "missing ONNX; run bash scripts/fetch_onnx.sh", onnx)
        print("[%s] %s | MISSING ONNX %s" % (_ts(), label, onnx), file=sys.stderr)
        return False

    have = avail_mb()
    if min_avail_mb > 0 and 0 <= have < min_avail_mb:
        record_blocked(man, spec, "only %d MiB available, need %d" % (have, min_avail_mb), "")
        print("[%s] %s | LOW RAM %d MiB" % (_ts(), label, have), file=sys.stderr)
        return False

    print("[%s] === %s ===" % (_ts(), label))
    print("[%s] onnx      %s (%.1f MB)" % (_ts(), spec["onnx"], os.path.getsize(onnx) / 1e6))
    print("[%s] engine    %s" % (_ts(), spec["engine"]))

    try:
        declared = onnx_input_shapes(onnx)
    except Exception as exc:
        record_blocked(man, spec, "ONNX does not parse: %s" % exc, "")
        print("[%s] %s | ONNX PARSE FAILED %s" % (_ts(), label, exc), file=sys.stderr)
        return False
    profile, problems = reconcile_shapes(shapes, declared, spec["onnx"])
    if problems:
        record_blocked(man, spec, "build spec disagrees with the ONNX graph", "\n".join(problems))
        for p in problems:
            print("[%s] %s | SHAPE MISMATCH %s" % (_ts(), label, p), file=sys.stderr)
        return False

    print("[%s] declared  %s" % (_ts(), declared))
    print("[%s] profile   %s  workspace=%s  dla=%s  avail=%d MiB"
          % (_ts(), profile or "(graph is fully static - no --*Shapes)", workspace, dla, have))
    sys.stdout.flush()

    rc, text, seconds = build_with_trtexec(onnx, engine, profile, workspace, dla, label, raw)
    if rc != 0 or not os.path.isfile(engine):
        record_blocked(man, spec, "trtexec build failed rc=%s" % rc, _error_tail(text))
        print("[%s] %s | BUILD FAILED rc=%s" % (_ts(), label, rc), file=sys.stderr)
        sys.stderr.write(_error_tail(text) + "\n")
        return False

    try:
        bindings = engine_bindings(engine)
    except Exception as exc:  # deserialisation is the contract; failure blocks the spec
        record_blocked(man, spec, "engine does not deserialize: %s" % exc, _error_tail(text))
        print("[%s] %s | DESERIALIZE FAILED %s" % (_ts(), label, exc), file=sys.stderr)
        return False

    if bindings["dynamic"]:
        print("[%s] %s | trtexec engine still dynamic; rebuilding through the "
              "TensorRT Python API with frozen input shapes" % (_ts(), label))
        try:
            build_static_with_python_api(onnx, engine, shapes, workspace, label)
            bindings = engine_bindings(engine)
        except Exception as exc:
            record_blocked(man, spec, "static rebuild failed: %s" % exc, _error_tail(text))
            print("[%s] %s | STATIC REBUILD FAILED %s" % (_ts(), label, exc), file=sys.stderr)
            return False
        if bindings["dynamic"]:
            still = [b for b in bindings["inputs"] + bindings["outputs"]
                     if any(s < 0 for s in b["shape"])]
            record_blocked(
                man, spec,
                "engine still reports a dynamic binding after freezing: %s" % still,
                _error_tail(text),
            )
            print("[%s] %s | STILL DYNAMIC - trt_engine.py would refuse it" % (_ts(), label),
                  file=sys.stderr)
            return False

    perf = measure_engine(engine, label, raw)
    if perf.get("loadEngine_rc") != 0:
        record_blocked(man, spec, "trtexec --loadEngine round trip failed", _error_tail(text))
        print("[%s] %s | LOADENGINE FAILED" % (_ts(), label), file=sys.stderr)
        return False

    record_success(man, spec, engine, bindings, perf, seconds)
    frac = parse_dla_fraction(text)
    if dla is not None and frac < 0.80:
        print("[%s] WARN: DLA fallback >20%% GPU (frac=%.3f); discard per design R2"
              % (_ts(), frac))
    print("[%s] %s | ok  %.1fs build  %.2f MB  devmem=%.2f MB  median=%s ms"
          % (_ts(), label, seconds, os.path.getsize(engine) / 1e6,
             bindings["device_memory_bytes"] / 1e6, perf.get("latency_median_ms")))
    for b in bindings["inputs"]:
        print("[%s]     IN  %-22s %-8s %s" % (_ts(), b["name"], b["dtype"], b["shape"]))
    for b in bindings["outputs"]:
        print("[%s]     OUT %-22s %-8s %s" % (_ts(), b["name"], b["dtype"], b["shape"]))
    sys.stdout.flush()
    return True


def main():
    # type: () -> int
    p = argparse.ArgumentParser(description="Build TensorRT FP16 engines from MANIFEST builds.")
    p.add_argument("--manifest", default=MANIFEST_PATH)
    p.add_argument("--only", action="append", default=[],
                   help="build only these MANIFEST build ids (repeatable, substring match)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a spec whose engine is already on disk with the manifest sha256")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a failed spec; still exits 1")
    p.add_argument("--measure-only", action="store_true",
                   help="do not rebuild: re-read the binding contract of the engine already "
                        "on disk and record its trtexec --loadEngine latency")
    p.add_argument("--verify", action="store_true",
                   help="do not rebuild: load each engine through dms/runtime/trt_engine.py, "
                        "run one synthetic inference and re-assert the recorded output "
                        "semantics. Run this under the GPU mutex.")
    p.add_argument("--min-avail-mb", type=int, default=1200,
                   help="refuse to start a build below this MemAvailable (MiB); 0 disables")
    p.add_argument("--raw", action="store_true", help="print every trtexec line (very noisy)")
    args = p.parse_args()

    if not os.path.isfile(TRTEXEC):
        print("trtexec not found: %s" % TRTEXEC, file=sys.stderr)
        return 2
    man = load_manifest(args.manifest)
    specs = man.get("builds") or []
    if not specs:
        print('MANIFEST has no "builds" list; nothing to do', file=sys.stderr)
        return 2
    if args.only:
        specs = [s for s in specs if any(o in s["id"] for o in args.only)]
        if not specs:
            print("no build id matched %s" % args.only, file=sys.stderr)
            return 2

    todo = []  # type: List[Dict[str, Any]]
    for s in specs:
        engine = os.path.join(ENGINE_DIR, s["engine"])
        rec = (man.get("engines") or {}).get(s["id"]) or {}
        if (args.skip_existing and not args.measure_only and not args.verify
                and os.path.isfile(engine) and rec.get("sha256")):
            if sha256_file(engine) == rec["sha256"]:
                print("[%s] skip %s (engine present, sha matches manifest)" % (_ts(), s["id"]))
                continue
        todo.append(s)
    if not todo:
        print("[%s] nothing to build" % _ts())
        return 0

    mode = "verifying" if args.verify else ("measuring" if args.measure_only else "building")
    print("[%s] %s %d engine(s) with %s (TRT %s)"
          % (_ts(), mode, len(todo), TRTEXEC, trt_version()))
    failed = []  # type: List[str]
    for i, spec in enumerate(todo, 1):
        ok = False
        try:
            if args.verify:
                ok = verify_spec(spec, man, i, len(todo))
            elif args.measure_only:
                ok = measure_spec(spec, man, i, len(todo), args.raw)
            else:
                ok = build_spec(spec, man, i, len(todo), args.raw, args.min_avail_mb)
        finally:
            if not args.verify:  # verification never rewrites the pinned records
                save_manifest(man, args.manifest)
        if not ok:
            failed.append(spec["id"])
            if not args.keep_going:
                print("[%s] aborting after %s (use --keep-going to continue)"
                      % (_ts(), spec["id"]), file=sys.stderr)
                return 1
    if failed:
        if args.verify:
            print("[%s] verification failed: %s" % (_ts(), ", ".join(failed)), file=sys.stderr)
        else:
            print('[%s] blocked: %s (recorded in MANIFEST["blocked"])'
                  % (_ts(), ", ".join(failed)), file=sys.stderr)
        return 1
    print("[%s] all engines %s" % (_ts(), "verified" if args.verify else "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
