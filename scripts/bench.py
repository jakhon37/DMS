#!/usr/bin/env python3
"""Measure this board: every engine in isolation, every CPU stage, then the pipeline.

The freeze gate is ``1280x720 end-to-end p95 <= 66 ms`` with ``peak RSS < 5.0 GB``
at ``MODE_20W_6CORE``.  Nothing in the tree measured it, so nothing could argue
about it.  This script produces the evidence, as JSON (machine-comparable between
runs) and Markdown (pasteable into ``docs/performance.md``).

Sections, selectable with ``--sections``
----------------------------------------
``env``
    What the board was doing while we measured: nvpmodel profile, GPU devfreq
    governor and the clock range actually observed, temperatures, MemAvailable,
    load average, OpenCV thread count, TensorRT version.  A latency number
    without this block is not reproducible - this GPU idles at 114.75 MHz under
    ``nvhost_podgov`` and peaks at 1109.25 MHz.
``engines``
    Each engine loaded alone and executed ``--iters`` times.  Reports the **host
    round trip** (what the pipeline actually pays: H2D + execute + D2H + the
    completion poll) and, with ``--trtexec``, the **GPU compute** median from
    ``trtexec --loadEngine``, which excludes transfers.  The gap between the two
    is the transfer and synchronisation overhead.
``cpu``
    The CPU stages, on a real frame from a real clip: the implicit copy behind
    ``frame.full_bgra[..., :3]``, the detector preprocess as currently written
    versus the preallocated path, the landmark warp from a strided view versus a
    contiguous one, and the overlay in each of its modes.
``dvfs``
    The same engine at four different CPU gaps between submits, with the devfreq
    clock sampled alongside.  This is the section that explains why an engine
    that measures 3.4 ms in the ``engines`` section costs 12 ms inside the
    pipeline: ``nvhost_podgov`` scales on utilisation and a sequential pipeline
    presents a low duty cycle.
``pipeline``
    The assembled loop over ``--frames`` frames of ``--clip``, timed per stage
    with :class:`dms.perf.LatencyBudget`, in up to three profiles: ``current``
    (what ``dms/app.py`` does today), ``optimised`` (contiguous frame plus the
    preallocated letterbox) and ``full`` (optimised plus every per-frame head).

GPU mutex
---------
Other agents share this board.  Every GPU section is executed while holding
``/tmp/jetson-gpu.lock``.  If the caller already holds it (``flock ... -c
"bench.py"``), set ``DMS_GPU_LOCK_HELD=1`` so this process does not deadlock
waiting for a lock its own parent owns - ``flock`` locks the open file
description, not the process tree.

Units: milliseconds for every latency, MiB for memory, MHz for clocks, degrees
Celsius for temperatures.

Failure behaviour: a section that cannot run (no engines, no clip, no TensorRT)
records ``{"status": "skipped", "reason": ...}`` in the report and the script
continues; it never fabricates a timing.  A non-zero exit status means the script
itself failed, not that the gate failed - the gate verdict is in the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GPU_LOCK = "/tmp/jetson-gpu.lock"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
DEFAULT_CLIP = os.path.join(ROOT, "tests", "replay", "day_driver.mp4")
# Under tests/replay/runs/, which .gitignore already excludes - bench artifacts are
# measurements of one board at one moment, not source.
DEFAULT_OUT = os.path.join(ROOT, "tests", "replay", "runs", "bench")

SECTIONS = ("env", "engines", "cpu", "dvfs", "pipeline")


# --------------------------------------------------------------------- lock --
class GpuLock(object):
    """Exclusive ``flock`` on the shared board lock, or a no-op when inherited.

    ``DMS_GPU_LOCK_HELD=1`` means an ancestor already holds it; re-locking the
    same file from a child would block forever because ``flock`` locks the open
    file description, so in that case this is a no-op that says so.
    """

    def __init__(self, path=GPU_LOCK, enabled=True):
        # type: (str, bool) -> None
        self.path = path
        self.enabled = bool(enabled) and os.environ.get("DMS_GPU_LOCK_HELD", "") not in ("1", "true", "yes")
        self.inherited = os.environ.get("DMS_GPU_LOCK_HELD", "") in ("1", "true", "yes")
        self._fd = None

    def __enter__(self):
        if not self.enabled:
            return self
        import fcntl

        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl

        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
        return False


# ---------------------------------------------------------------- utilities --
def _read(path):
    # type: (str) -> str
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _stats(samples_ms):
    # type: (list) -> dict
    """min / p50 / p95 / p99 / max / mean over a list of millisecond samples."""
    if not samples_ms:
        return {"n": 0}
    xs = sorted(samples_ms)
    n = len(xs)

    def q(p):
        k = int(round((p / 100.0) * (n - 1)))
        return round(xs[max(0, min(n - 1, k))], 3)

    return {
        "n": n,
        "min_ms": round(xs[0], 3),
        "p50_ms": q(50),
        "p95_ms": q(95),
        "p99_ms": q(99),
        "max_ms": round(xs[-1], 3),
        "mean_ms": round(sum(xs) / n, 3),
    }


def timeit(fn, iters, warmup=10):
    # type: (object, int, int) -> dict
    """Call ``fn`` ``iters`` times after ``warmup``, timing each call separately.

    Per-call timing (not a total divided by n) is deliberate: the tail is the
    thing the 66 ms gate cares about, and an average hides it.
    """
    perf = time.perf_counter
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = perf()
        fn()
        out.append((perf() - t0) * 1000.0)
    return _stats(out)


# --------------------------------------------------------------------- env --
def section_env():
    # type: () -> dict
    """Board state at measurement time. Everything here is a kernel file read."""
    from dms.io.thermal import gpu_clock_mhz, mem_available_mb, read_temps

    devfreq = ""
    base = "/sys/devices/gpu.0/devfreq"
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            devfreq = os.path.join(base, name)
            break
    gpu_c, cpu_c = read_temps()
    trt_version = None
    try:
        import tensorrt as trt

        trt_version = trt.__version__
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        trt_version = "unavailable: %s" % (exc,)
    cv_threads = None
    cv_version = None
    try:
        import cv2

        cv_threads = cv2.getNumThreads()
        cv_version = cv2.__version__
    except Exception as exc:  # noqa: BLE001
        cv_version = "unavailable: %s" % (exc,)
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "nvpmodel": _read("/var/lib/nvpmodel/status"),
        "gpu_devfreq": devfreq,
        "gpu_governor": _read(os.path.join(devfreq, "governor")) if devfreq else "",
        "gpu_cur_mhz": gpu_clock_mhz(),
        "gpu_min_mhz": _mhz(_read(os.path.join(devfreq, "min_freq"))) if devfreq else None,
        "gpu_max_mhz": _mhz(_read(os.path.join(devfreq, "max_freq"))) if devfreq else None,
        "gpu_temp_c": gpu_c,
        "cpu_temp_c": cpu_c,
        "mem_available_mb": mem_available_mb(),
        "loadavg": _read("/proc/loadavg"),
        "cpu_count": os.cpu_count(),
        "opencv": cv_version,
        "opencv_threads": cv_threads,
        "tensorrt": trt_version,
    }


def _mhz(raw):
    # type: (str) -> object
    try:
        return int(raw) // 1000000
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- engines --
def _manifest_engines(engine_dir):
    # type: (str) -> list
    """``[(role, filename, path)]`` for every engine record that is on disk."""
    path = os.path.join(engine_dir, "MANIFEST.json")
    try:
        with open(path) as f:
            man = json.load(f)
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read %s: %s" % (path, exc))
    out = []
    for key, rec in sorted((man.get("engines") or {}).items()):
        if not isinstance(rec, dict):
            continue
        fname = rec.get("file") or (key + ".engine")
        p = os.path.join(engine_dir, str(fname))
        if os.path.isfile(p):
            out.append((str(rec.get("role") or key), str(fname), p))
    return out


def trtexec_gpu_ms(engine_path, iters=100):
    # type: (str, int) -> dict
    """GPU compute median/p99 from ``trtexec --loadEngine``, transfers excluded.

    Returns ``{"status": "skipped"|"ok"|"failed", ...}``. The caller must already
    hold the GPU lock.
    """
    if not os.path.isfile(TRTEXEC):
        return {"status": "skipped", "reason": "trtexec not found at %s" % TRTEXEC}
    cmd = [
        TRTEXEC,
        "--loadEngine=%s" % engine_path,
        "--iterations=%d" % iters,
        "--warmUp=200",
        "--avgRuns=%d" % max(10, iters // 2),
        "--noDataTransfers",
        "--useSpinWait",
        "--separateProfileRun",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "reason": str(exc)}
    txt = p.stdout or ""
    if p.returncode != 0:
        return {"status": "failed", "reason": "trtexec rc=%d" % p.returncode,
                "tail": txt.strip().splitlines()[-5:]}
    out = {"status": "ok"}
    m = re.search(r"GPU Compute Time:.*?median = ([0-9.]+) ms", txt)
    if m:
        out["gpu_median_ms"] = float(m.group(1))
    m = re.search(r"GPU Compute Time:.*?percentile\(9[09](?:\.00)?%\) = ([0-9.]+) ms", txt)
    if m:
        out["gpu_p99_ms"] = float(m.group(1))
    m = re.search(r"GPU Compute Time:.*?mean = ([0-9.]+) ms", txt)
    if m:
        out["gpu_mean_ms"] = float(m.group(1))
    m = re.search(r"Throughput: ([0-9.]+) qps", txt)
    if m:
        out["qps"] = float(m.group(1))
    return out


def section_engines(engine_dir, iters, only, use_trtexec):
    # type: (str, int, list, bool) -> dict
    """Load each engine alone; time the host round trip the pipeline actually pays."""
    import numpy as np

    from dms.runtime.trt_engine import TrtEngine

    try:
        entries = _manifest_engines(engine_dir)
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc)}
    if only:
        entries = [e for e in entries if any(k in e[1] or k in e[0] for k in only)]
    if not entries:
        return {"status": "skipped", "reason": "no engines matched"}

    rng = np.random.RandomState(1234)
    results = {}
    for role, fname, path in entries:
        rec = {"role": role, "file": fname, "file_mb": round(os.path.getsize(path) / 1048576.0, 2)}
        eng = None
        try:
            eng = TrtEngine(path)
            feed = {}
            for name in eng.input_names:
                shape = eng.shape_of(name)
                feed[name] = rng.rand(*shape).astype(np.float32)
            rec["inputs"] = dict((n, list(eng.shape_of(n))) for n in eng.input_names)
            rec["outputs"] = dict((n, list(eng.shape_of(n))) for n in eng.output_names)
            rec["context_mb"] = round(eng.device_memory_bytes / 1048576.0, 2)
            rec["io_mb"] = round(eng.io_bytes / 1048576.0, 2)

            def run(_e=eng, _f=feed):
                _e.infer(_f, copy=False)

            rec["host_roundtrip"] = timeit(run, iters, warmup=max(10, iters // 10))
            if use_trtexec:
                rec["trtexec"] = trtexec_gpu_ms(path, iters=iters)
                gpu = rec["trtexec"].get("gpu_median_ms")
                if gpu is not None:
                    rec["overhead_ms"] = round(rec["host_roundtrip"]["p50_ms"] - gpu, 3)
            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - one bad engine must not lose the run
            rec["status"] = "failed"
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            if eng is not None:
                try:
                    eng.close()
                except Exception:  # noqa: BLE001
                    pass
        results[fname] = rec
        print("  engine %-40s %s" % (fname, rec.get("host_roundtrip", rec.get("error"))))
    return {"status": "ok", "iters": iters, "engines": results}


# --------------------------------------------------------------------- cpu --
def _first_frame(clip, index=0):
    # type: (str, int) -> object
    import cv2

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % clip)
    frame = None
    try:
        for _ in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("clip %s ended before frame %d" % (clip, index))
    finally:
        cap.release()
    return frame


def section_cpu(clip, iters):
    # type: (str, int) -> dict
    """CPU-only costs, on a real 720p frame. No GPU, no lock needed."""
    import cv2
    import numpy as np

    from dms.infer.landmarks import loose_crop_affine
    from dms.runtime.preprocess import Letterbox, bgra_to_bgr
    from dms.viz.overlay import draw_faces
    from dms.viz.renderer import OverlayRenderer

    try:
        bgr_src = _first_frame(clip, 150)
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc)}
    h, w = bgr_src.shape[:2]
    # The capture path delivers BGRx; reproduce that layout exactly, because the
    # stride-4 slice is the thing being measured.
    bgra = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2BGRA)
    strided = bgra[..., :3]
    contig = np.ascontiguousarray(strided)
    bgr_buf = np.empty((h, w, 3), dtype=np.uint8)

    out = {"status": "ok", "iters": iters, "frame": [h, w], "stages": {}}
    st = out["stages"]

    st["slice_gather_ascontiguous"] = dict(
        timeit(lambda: np.ascontiguousarray(strided), iters),
        note="the hidden cost of frame.full_bgra[..., :3]: every OpenCV call on that "
             "stride-4 view materialises this copy internally",
    )
    st["bgra_to_bgr_cvtColor"] = dict(
        timeit(lambda: bgra_to_bgr(bgra, out=bgr_buf), iters),
        note="doing it once, explicitly, into a reused buffer",
    )

    # Detector preprocess: current code path vs the preallocated one.
    lb640 = Letterbox.for_scrfd((640, 640))
    lb512 = Letterbox.for_scrfd((512, 512))

    def scrfd_current(src):
        from dms.infer.scrfd import resize_letterbox

        canvas, factor = resize_letterbox(src, (640, 640))
        blob = canvas.astype(np.float32)
        blob = (blob - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob), factor

    st["scrfd_pre_current_contiguous"] = dict(
        timeit(lambda: scrfd_current(contig), iters),
        note="dms/infer/scrfd.py preprocess() as written, fed a contiguous frame",
    )
    st["scrfd_pre_current_strided"] = dict(
        timeit(lambda: scrfd_current(strided), iters),
        note="the same, fed the stride-4 view the app actually passes",
    )
    st["scrfd_pre_letterbox_640"] = dict(
        timeit(lambda: lb640.blob(contig), iters),
        note="dms/runtime/preprocess.Letterbox.for_scrfd((640,640)), preallocated",
    )
    st["scrfd_pre_letterbox_512"] = dict(
        timeit(lambda: lb512.blob(contig), iters),
        note="the same at the 512 detector input",
    )
    st["scrfd_pre_blobFromImage_640"] = dict(
        timeit(
            lambda: cv2.dnn.blobFromImage(
                lb640.canvas(contig)[0], scalefactor=1.0 / 127.5, size=(640, 640),
                mean=(127.5, 127.5, 127.5), swapRB=False
            ),
            iters,
        ),
        note="cv2.dnn.blobFromImage on the same canvas, for comparison",
    )

    # Landmark crop: warping the full frame vs warping only the crop rectangle.
    box = np.array([w * 0.35, h * 0.25, w * 0.55, h * 0.62], dtype=np.float32)
    M = loose_crop_affine(box, 192, 1.5)
    st["landmark_warp_full_strided"] = dict(
        timeit(lambda: cv2.warpAffine(strided, M, (192, 192)), iters),
        note="warpAffine over the whole 720p stride-4 view (what the old code did)",
    )
    st["landmark_warp_full_contiguous"] = dict(
        timeit(lambda: cv2.warpAffine(contig, M, (192, 192)), iters),
        note="the same warp over a contiguous frame",
    )

    # Overlay, in each mode.
    from dms.infer.scrfd import FaceDet

    face = FaceDet(
        xyxy=box.copy(), score=0.93,
        kps=np.array([[w * .41, h * .36], [w * .49, h * .36], [w * .45, h * .44],
                      [w * .42, h * .52], [w * .48, h * .52]], dtype=np.float32),
        xyxy_raw=box.copy(),
    )
    lmk = np.stack([
        np.linspace(w * 0.36, w * 0.54, 106),
        np.linspace(h * 0.27, h * 0.60, 106),
    ], axis=1).astype(np.float32)
    kw = dict(driver=face, landmarks106=lmk, ear=0.31, mar=0.22, pose=(4.0, -7.0, 1.5),
              track_id=3, alerts=["distraction:warn"], yaw_rel=-6.0)

    st["overlay_full_alloc"] = dict(
        timeit(lambda: draw_faces(contig, [face], **kw), iters),
        note="scale=1.0, allocating a fresh canvas each frame (the old behaviour)",
    )
    r_full = OverlayRenderer(enabled=True, scale=1.0)
    st["overlay_full_reused"] = dict(
        timeit(lambda: r_full.render(contig, [face], **kw), iters),
        note="scale=1.0 into a reused buffer",
    )
    r_half = OverlayRenderer(enabled=True, scale=0.5)
    st["overlay_half_reused"] = dict(
        timeit(lambda: r_half.render(contig, [face], **kw), iters),
        note="scale=0.5 (640x360) into a reused buffer",
    )
    r_off = OverlayRenderer.disabled()
    st["overlay_disabled"] = dict(
        timeit(lambda: r_off.render(contig, [face], **kw), iters),
        note="the production configuration: one attribute test, no pixels touched",
    )
    st["overlay_strided_alloc"] = dict(
        timeit(lambda: draw_faces(strided, [face], **kw), iters),
        note="the old path in full: overlay on the stride-4 view",
    )

    # The output stage. --save-video is on for every clip in run_all_replay, and an
    # MJPG encode of a 720p frame on this CPU is the single most expensive thing the
    # loop does - more than the detector.
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="dms-bench-")
    for tag, img in (("720p", contig), ("360p", cv2.resize(contig, (640, 360)))):
        vw = cv2.VideoWriter(os.path.join(tmpdir, "w_%s.avi" % tag),
                             cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                             (img.shape[1], img.shape[0]))
        if vw.isOpened():
            st["videowriter_mjpg_%s" % tag] = dict(
                timeit(lambda _v=vw, _i=img: _v.write(_i), max(20, iters // 2)),
                note="cv2.VideoWriter MJPG write of one %s frame" % tag,
            )
            vw.release()
        else:
            st["videowriter_mjpg_%s" % tag] = {"n": 0, "note": "writer would not open"}
    try:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
    except OSError:
        pass

    # Timer overhead, so nobody has to wonder whether the instrument moved the needle.
    from dms.perf.timers import StageTimer

    t = StageTimer(window=300)

    def span():
        with t.stage("x"):
            pass

    st["stage_timer_span"] = dict(timeit(span, max(iters, 2000)),
                                  note="cost of one instrumented span (dms.perf.timers)")
    return out


# -------------------------------------------------------------------- dvfs --
#: Engines probed by the ``dvfs`` section, and the CPU gaps (ms) inserted between
#: submits.  50 ms is roughly the CPU time a full frame spends outside the GPU in
#: the ``production`` profile, so that column is the duty cycle the real pipeline
#: presents to the governor.
DVFS_ENGINES = (
    "scrfd_500m_512_fp16_gpu.engine",
    "2d106det_192_fp16_gpu.engine",
    "hpe_mobilenetv2_224_fp16_gpu.engine",
)
DVFS_GAPS_MS = (0.0, 10.0, 25.0, 50.0)


def _gpu_cur_mhz():
    # type: () -> object
    base = "/sys/devices/gpu.0/devfreq"
    try:
        for name in sorted(os.listdir(base)):
            with open(os.path.join(base, name, "cur_freq")) as f:
                return int(f.read().strip()) // 1000000
    except (OSError, ValueError):
        return None
    return None


def section_dvfs(engine_dir, iters=80, settle=40):
    # type: (str, int, int) -> dict
    """Is the pipeline's engine latency a governor artefact? Measure at four duty cycles.

    For each engine and each CPU gap: run ``settle`` inferences so the governor
    reaches its steady state for that duty cycle, then time ``iters`` more while
    sampling ``cur_freq``.  The caller must hold the GPU lock.

    The "CPU gap" is busy numpy work, not a sleep: a sleeping process still
    yields the core, whereas the real pipeline is computing, which is what the
    governor and the 20 W budget actually see.
    """
    import numpy as np

    from dms.runtime.trt_engine import TrtEngine

    def burn(ms):
        t_end = time.perf_counter() + ms / 1000.0
        x = np.zeros((256, 256), dtype=np.float32)
        while time.perf_counter() < t_end:
            x += 1.0

    out = {"status": "ok", "iters": iters, "gaps_ms": list(DVFS_GAPS_MS), "engines": {}}
    for name in DVFS_ENGINES:
        path = os.path.join(engine_dir, name)
        if not os.path.isfile(path):
            out["engines"][name] = {"status": "skipped", "reason": "not on disk"}
            continue
        eng = None
        rec = {"status": "ok", "gaps": {}}
        try:
            eng = TrtEngine(path)
            rng = np.random.RandomState(0)
            feed = {n: rng.rand(*eng.shape_of(n)).astype(np.float32) for n in eng.input_names}
            for gap in DVFS_GAPS_MS:
                for _ in range(settle):
                    eng.infer(feed, copy=False)
                    if gap:
                        burn(gap)
                ts, clocks = [], []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    eng.infer(feed, copy=False)
                    ts.append((time.perf_counter() - t0) * 1000.0)
                    clocks.append(_gpu_cur_mhz())
                cl = sorted(c for c in clocks if c)
                st = _stats(ts)
                st["clock_min_mhz"] = cl[0] if cl else None
                st["clock_median_mhz"] = cl[len(cl) // 2] if cl else None
                st["clock_max_mhz"] = cl[-1] if cl else None
                rec["gaps"]["%g" % gap] = st
                print("  dvfs %-38s gap=%4.0f ms  p50=%6.2f ms  clock %s MHz"
                      % (name, gap, st["p50_ms"], st["clock_median_mhz"]))
        except Exception as exc:  # noqa: BLE001
            rec = {"status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)}
        finally:
            if eng is not None:
                try:
                    eng.close()
                except Exception:  # noqa: BLE001
                    pass
        out["engines"][name] = rec
    return out


# ---------------------------------------------------------------- pipeline --
class _ClipReader(object):
    """Frames from an mp4 as (BGRA, BGR-contiguous), loops at EOF.

    The real capture path hands the loop a C-contiguous ``(H, W, 4)`` BGRx
    buffer; ``cv2.VideoCapture`` gives BGR.  The conversion is done outside the
    timed region so the benchmark measures the pipeline, not the decoder.
    """

    def __init__(self, path, count):
        # type: (str, int) -> None
        import cv2

        self.cv2 = cv2
        self.path = path
        self.count = int(count)
        self.frames = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("cannot open %s" % path)
        try:
            while len(self.frames) < self.count:
                ok, f = cap.read()
                if not ok:
                    break
                self.frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2BGRA))
        finally:
            cap.release()
        if not self.frames:
            raise RuntimeError("no frames decoded from %s" % path)

    def __len__(self):
        return len(self.frames)


def _profile_current(reader, detector, landmarker, budget, tracker, frames):
    """Exactly what dms/app.py does today: strided view, detect, landmarks."""
    import numpy as np

    from dms.geometry.ear import face_ear
    from dms.geometry.mar import mouth_aspect_ratio
    from dms.geometry.pnp import solve_head_pose
    from dms.track.driver_select import pick_driver

    for i in range(frames):
        bgra = reader.frames[i % len(reader.frames)]
        with budget.stage("e2e"):
            with budget.stage("bgra"):
                bgr = bgra[..., :3]
            with budget.stage("detect"):
                faces = detector.detect(bgr)
            h, w = bgr.shape[:2]
            with budget.stage("track"):
                tracks = tracker.update(faces)
                live = [t for t in tracks if t.face is not None]
                driver = pick_driver([t.face for t in live], [0.0, 0.0, 0.65, 1.0], (w, h), fallback=True)
            if landmarker is not None and driver is not None:
                with budget.stage("landmarks"):
                    lmk = landmarker.infer_one(bgr, driver.xyxy)
                if lmk is not None:
                    with budget.stage("fusion"):
                        face_ear(lmk)
                        mouth_aspect_ratio(lmk)
                        solve_head_pose(lmk, (w, h))
        budget.mark_frame()


def _profile_optimised(reader, detector, landmarker, budget, tracker, frames, letterbox,
                       heads=None, cadence=None):
    """Contiguous frame + preallocated letterbox + split detect, optional heads."""
    import numpy as np

    from dms.runtime.preprocess import BgraToBgr
    from dms.track.driver_select import pick_driver

    to_bgr = BgraToBgr()
    in_name = detector.engine.input_names[0]
    for i in range(frames):
        bgra = reader.frames[i % len(reader.frames)]
        with budget.stage("e2e"):
            with budget.stage("bgra"):
                bgr = to_bgr(bgra)
            h, w = bgr.shape[:2]
            with budget.stage("letterbox"):
                blob, geom = letterbox.blob(bgr)
            with budget.stage("detect_gpu"):
                raw = detector.engine.infer({in_name: blob}, copy=False)
            with budget.stage("detect_post"):
                faces = detector.postprocess(raw, geom.factor, orig_hw=(h, w))
            with budget.stage("track"):
                tracks = tracker.update(faces)
                live = [t for t in tracks if t.face is not None]
                driver = pick_driver([t.face for t in live], [0.0, 0.0, 0.65, 1.0], (w, h), fallback=True)
            lmk = None
            if landmarker is not None and driver is not None:
                with budget.stage("landmarks"):
                    lmk = landmarker.infer_one(bgr, driver.box_for_crop())
            if heads and driver is not None:
                _run_heads(heads, budget, bgr, driver, lmk, i, cadence)
            if lmk is not None:
                with budget.stage("fusion"):
                    from dms.geometry.ear import face_ear
                    from dms.geometry.mar import mouth_aspect_ratio

                    face_ear(lmk)
                    mouth_aspect_ratio(lmk)
        budget.mark_frame()


#: Cadence for the ``production`` profile: ``head -> (every_n, phase)``.
#:
#: The phases are offset so the two expensive intermittent heads (the object net
#: and the occlusion gate) never land on the same frame - coincident spikes are
#: what turns a good p50 into a bad p95.
#:
#: The object period is 20, not 10, and the reason is a percentile rather than a
#: hunch: at every tenth frame the object net runs on 10% of frames, so the p95
#: lands *on* an object-net frame and inherits its full 23-29 ms.  Measured back
#: to back on this board, only that period changed: every 10th = 44.2 ms p50 /
#: 77.6 ms p95 (fails the 66 ms gate), every 20th = 37.3 / 62.2 (passes).
PRODUCTION_CADENCE = {
    "headpose": (1, 0),
    "eyestate": (1, 0),
    "mouthstate": (1, 0),
    "occlusion": (15, 3),
    "objects": (20, 7),
    "identity": (0, 0),  # every_n = 0 means "first frame only" (new track id)
}


def _due(cadence, name, frame_i):
    """True when ``name`` runs on this frame under ``cadence`` (None = every frame)."""
    if cadence is None:
        return True
    spec = cadence.get(name)
    if spec is None:
        return True
    every, phase = spec
    if every <= 0:
        return frame_i == phase
    return (frame_i % every) == (phase % every)


def _run_heads(reg, budget, bgr, driver, lmk, frame_i, cadence=None):
    """Per-frame heads, each timed under its own stage, honouring ``cadence``.

    A head that is not due this frame is not timed either: a stage whose samples
    are mostly zero would report a meaningless p50.  The cost of a cadenced head
    is therefore read from its own p50/p95 divided by its period, which is what
    ``docs/performance.md`` does.
    """
    occ = None
    if reg.occlusion is not None and _due(cadence, "occlusion", frame_i):
        with budget.stage("occlusion"):
            occ = reg.occlusion.infer(bgr, driver.box_for_crop(), track_id=0)
    if reg.headpose is not None and _due(cadence, "headpose", frame_i):
        with budget.stage("headpose"):
            reg.headpose.infer(bgr, driver.box_for_crop(), lmk, kps5=driver.kps)
    if reg.eyestate is not None and lmk is not None and _due(cadence, "eyestate", frame_i):
        with budget.stage("eyestate"):
            reg.eyestate.infer(bgr, lmk, eyes_reliable=True if occ is None else occ.eyes_reliable)
    if reg.mouthstate is not None and lmk is not None and _due(cadence, "mouthstate", frame_i):
        with budget.stage("mouthstate"):
            reg.mouthstate.infer(bgr, lmk, mouth_reliable=True if occ is None else occ.mouth_reliable)
    if reg.objects is not None and _due(cadence, "objects", frame_i):
        with budget.stage("objects"):
            reg.objects.detect(bgr, force=True)
    if reg.identity is not None and _due(cadence, "identity", frame_i):
        with budget.stage("identity"):
            reg.identity.embed(bgr, driver.kps)


def parse_cadence(spec):
    # type: (str) -> dict
    """``"objects=20:7,occlusion=15:3"`` -> ``{"objects": (20, 7), ...}``.

    ``every=0`` means "first frame only" (a once-per-track head such as identity).
    Raises ``ValueError`` on a malformed spec rather than silently running a head
    on every frame, which would quietly invalidate the measurement.
    """
    out = dict(PRODUCTION_CADENCE)
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, rest = part.partition("=")
        every, _, phase = rest.partition(":")
        try:
            out[name.strip()] = (int(every), int(phase or 0))
        except ValueError:
            raise ValueError("bad cadence %r; expected name=every[:phase]" % (part,))
    return out


def section_pipeline(clip, frames, profiles, face_engine, lmk_engine, engine_dir,
                     cadence_override=None):
    # type: (str, int, list, str, str, str, dict) -> dict
    """Run the assembled loop and produce a LatencyBudget per profile."""
    from dms.infer.landmarks import Landmark106
    from dms.infer.scrfd import ScrfdDetector
    from dms.perf import LatencyBudget, peak_rss_mb, sample_board
    from dms.runtime.preprocess import Letterbox
    from dms.runtime.trt_engine import TrtEngine
    from dms.track.iou_tracker import IouTracker

    if not os.path.isfile(face_engine):
        return {"status": "skipped", "reason": "no face engine at %s" % face_engine}
    try:
        reader = _ClipReader(clip, min(frames, 240))
    except RuntimeError as exc:
        return {"status": "skipped", "reason": str(exc)}

    out = {"status": "ok", "clip": os.path.basename(clip), "frames": frames,
           "decoded_frames": len(reader), "profiles": {}}
    for prof in profiles:
        det_engine = None
        lm_engine = None
        reg = None
        try:
            det_path = face_engine
            if prof in ("optimised512", "production", "full512"):
                cand = os.path.join(engine_dir, "scrfd_500m_512_fp16_gpu.engine")
                if not os.path.isfile(cand):
                    out["profiles"][prof] = {"status": "skipped", "reason": "no 512 engine"}
                    continue
                det_path = cand
            det_engine = TrtEngine(det_path)
            detector = ScrfdDetector(det_engine, conf=0.5, iou=0.4)
            landmarker = None
            if os.path.isfile(lmk_engine):
                lm_engine = TrtEngine(lmk_engine)
                landmarker = Landmark106(lm_engine)
            budget = LatencyBudget(label=prof)
            budget.set_context(profile=prof, clip=os.path.basename(clip),
                               detector=os.path.basename(det_path),
                               resolution="%dx%d" % (reader.frames[0].shape[1], reader.frames[0].shape[0]))
            tracker = IouTracker()
            heads = None
            cadence = None
            if prof in ("full", "full512", "production"):
                from dms.infer.registry import ModelRegistry

                reg = ModelRegistry(engine_dir=engine_dir, backends={
                    "headpose": "model", "eyestate": "model", "mouthstate": "model",
                    "occlusion": "model", "objects": "model", "identity": "model",
                    "gaze": "off",
                }).load()
                heads = reg
                if prof == "production":
                    cadence = cadence_override or PRODUCTION_CADENCE
                budget.set_context(heads=reg.describe(), cadence=cadence)
            sample_board(budget)
            t0 = time.monotonic()
            if prof == "current":
                _profile_current(reader, detector, landmarker, budget, tracker, frames)
            else:
                net = det_engine.input_shape
                lb = Letterbox.for_scrfd((int(net[2]), int(net[3])))
                _profile_optimised(reader, detector, landmarker, budget, tracker, frames, lb,
                                   heads, cadence)
            wall = time.monotonic() - t0
            sample_board(budget)
            budget.observe_rss(peak_rss_mb())
            rep = budget.report()
            rep["wall_seconds"] = round(wall, 2)
            rep["measured_fps"] = round(frames / wall, 2) if wall > 0 else None
            out["profiles"][prof] = rep
            print("  profile %-12s e2e p50=%.1f p95=%.1f ms  fps=%.1f" % (
                prof, budget.timer.stage("e2e").stats().p50,
                budget.timer.stage("e2e").stats().p95, frames / wall if wall else 0.0))
        except Exception as exc:  # noqa: BLE001 - report and continue to the next profile
            out["profiles"][prof] = {"status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)}
            print("  profile %-12s FAILED %s: %s" % (prof, type(exc).__name__, exc))
        finally:
            if reg is not None:
                try:
                    reg.close()
                except Exception:  # noqa: BLE001
                    pass
            for e in (lm_engine, det_engine):
                if e is not None:
                    try:
                        e.close()
                    except Exception:  # noqa: BLE001
                        pass
    return out


# ------------------------------------------------------------------ output --
def to_markdown(report):
    # type: (dict) -> str
    lines = ["# DMS bench - %s" % report.get("generated_utc", ""), ""]
    env = report.get("env") or {}
    if env:
        lines += [
            "Board: %s, kernel %s, %s, nvpmodel `%s`, %s cores."
            % (env.get("hostname"), env.get("kernel"), env.get("python"),
               env.get("nvpmodel"), env.get("cpu_count")),
            "",
            "GPU devfreq governor `%s`, %s-%s MHz, at %s MHz when sampled. "
            "OpenCV %s (%s threads), TensorRT %s."
            % (env.get("gpu_governor"), env.get("gpu_min_mhz"), env.get("gpu_max_mhz"),
               env.get("gpu_cur_mhz"), env.get("opencv"), env.get("opencv_threads"),
               env.get("tensorrt")),
            "",
            "MemAvailable %s MiB, loadavg `%s`, GPU %s C / CPU %s C."
            % (env.get("mem_available_mb"), env.get("loadavg"),
               env.get("gpu_temp_c"), env.get("cpu_temp_c")),
            "",
        ]
    eng = report.get("engines") or {}
    if eng.get("status") == "ok":
        lines += ["## Engines in isolation", "",
                  "| engine | role | file MB | ctx MB | host p50 | host p95 | GPU p50 | overhead |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        rows = sorted(eng["engines"].items(),
                      key=lambda kv: (kv[1].get("host_roundtrip") or {}).get("p50_ms", 1e9))
        for fname, rec in rows:
            hr = rec.get("host_roundtrip") or {}
            tx = rec.get("trtexec") or {}
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                fname.replace(".engine", ""), rec.get("role"), rec.get("file_mb"),
                rec.get("context_mb"), _f(hr.get("p50_ms")), _f(hr.get("p95_ms")),
                _f(tx.get("gpu_median_ms")), _f(rec.get("overhead_ms"))))
        lines.append("")
    cpu = report.get("cpu") or {}
    if cpu.get("status") == "ok":
        lines += ["## CPU stages (1280x720, single frame, %d iterations)" % cpu.get("iters", 0), "",
                  "| stage | p50 ms | p95 ms | note |", "|---|---:|---:|---|"]
        for name, rec in cpu["stages"].items():
            lines.append("| %s | %s | %s | %s |" % (name, _f(rec.get("p50_ms")),
                                                    _f(rec.get("p95_ms")), rec.get("note", "")))
        lines.append("")
    dv = report.get("dvfs") or {}
    if dv.get("status") == "ok":
        lines += ["## GPU clock versus duty cycle", "",
                  "Same engine, same input; only the CPU gap between submits changes.", "",
                  "| engine | " + " | ".join("gap %g ms" % g for g in dv["gaps_ms"]) + " |",
                  "|---" + "|---:" * len(dv["gaps_ms"]) + "|"]
        for name, rec in dv["engines"].items():
            if rec.get("status") != "ok":
                continue
            cells = []
            for g in dv["gaps_ms"]:
                st = rec["gaps"].get("%g" % g, {})
                cells.append("%s ms @ %s MHz" % (_f(st.get("p50_ms")), st.get("clock_median_mhz")))
            lines.append("| %s | %s |" % (name.replace(".engine", ""), " | ".join(cells)))
        lines.append("")
    pipe = report.get("pipeline") or {}
    if pipe.get("status") == "ok":
        lines += ["## Assembled pipeline (%s, %d frames)" % (pipe.get("clip"), pipe.get("frames", 0)), ""]
        for prof, rep in pipe.get("profiles", {}).items():
            if rep.get("status") in ("skipped", "failed"):
                lines += ["### %s - %s (%s)" % (prof, rep.get("status"),
                                                rep.get("reason") or rep.get("error")), ""]
                continue
            g = rep.get("gate", {})
            lines += ["### %s" % prof, "",
                      "e2e p50 %s ms, p95 **%s ms** (limit %s), measured %s FPS, peak RSS %s MiB."
                      % (_f(_stage_of(rep, "e2e", "p50_ms")), _f(g.get("e2e_p95_ms")),
                         g.get("e2e_p95_limit_ms"), rep.get("measured_fps"), g.get("peak_rss_mb")),
                      "",
                      "| stage | p50 ms | p95 ms | share | target | status |",
                      "|---|---:|---:|---:|---:|---|"]
            for row in rep.get("stages", []):
                if row.get("status") in ("not_run", "missing"):
                    continue
                lines.append("| %s | %s | %s | %s | %s | %s |" % (
                    row["stage"], _f(row.get("p50_ms")), _f(row.get("p95_ms")),
                    "" if row.get("share_pct") is None else "%.1f%%" % row["share_pct"],
                    _f(row.get("target_ms")), row.get("status")))
            acc = rep.get("accounting", {})
            lines += ["", "amortised sum of stages = %s ms vs e2e mean %s ms "
                      "(unaccounted %s ms; raw p50 sum %s ms)"
                      % (acc.get("sum_stage_amortised_ms"), acc.get("e2e_mean_ms"),
                         acc.get("unaccounted_ms"), acc.get("sum_stage_p50_ms")), ""]
    return "\n".join(lines) + "\n"


def _stage_of(rep, stage, key):
    for row in rep.get("stages", []):
        if row.get("stage") == stage:
            return row.get(key)
    return None


def _f(v):
    if v is None:
        return "-"
    try:
        return "%.2f" % float(v)
    except (TypeError, ValueError):
        return str(v)


# -------------------------------------------------------------------- main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="", help="name this run instead of using a timestamp")
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--engine-dir", default=os.path.join(ROOT, "engines"))
    ap.add_argument("--face-engine", default="")
    ap.add_argument("--landmark-engine", default="")
    ap.add_argument("--iters", type=int, default=100, help="iterations per engine / CPU stage")
    ap.add_argument("--frames", type=int, default=120, help="frames per pipeline profile")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument(
        "--profiles",
        default="current,optimised,optimised512,full",
        help="current = dms/app.py as written; optimised = contiguous frame + preallocated "
             "letterbox; optimised512 = the same on the 512 detector; full = optimised512 "
             "plus every head on every frame; full512 = alias of full on the 512 detector; "
             "production = full512 with the cadences in PRODUCTION_CADENCE",
    )
    ap.add_argument("--only", action="append", default=[], help="engine name substring filter")
    ap.add_argument("--cadence", default="",
                    help="override the production profile cadence, e.g. "
                         "'objects=20:7,occlusion=15:3' (every[:phase]; every=0 means once)")
    ap.add_argument("--trtexec", dest="trtexec", action="store_true", default=True)
    ap.add_argument("--no-trtexec", dest="trtexec", action="store_false")
    ap.add_argument("--no-lock", dest="lock", action="store_false", default=True)
    args = ap.parse_args(argv)

    want = [s.strip() for s in args.sections.split(",") if s.strip()]
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        sys.stderr.write("unknown section(s): %s (known: %s)\n" % (bad, list(SECTIONS)))
        return 2
    face_engine = args.face_engine or os.path.join(args.engine_dir, "scrfd_500m_640_fp16_gpu.engine")
    lmk_engine = args.landmark_engine or os.path.join(args.engine_dir, "2d106det_192_fp16_gpu.engine")

    stamp = args.tag or time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "schema": "dms.bench/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv[1:],
        "out_dir": out_dir,
    }
    if "env" in want:
        print("[env]")
        report["env"] = section_env()
    if "cpu" in want:
        print("[cpu] %d iterations" % args.iters)
        report["cpu"] = section_cpu(args.clip, args.iters)
    lock = GpuLock(enabled=args.lock)
    if lock.inherited:
        print("GPU lock inherited from parent (DMS_GPU_LOCK_HELD=1)")
    if "engines" in want or "pipeline" in want or "dvfs" in want:
        with lock:
            if "engines" in want:
                print("[engines] %d iterations each" % args.iters)
                report["engines"] = section_engines(args.engine_dir, args.iters, args.only, args.trtexec)
            if "dvfs" in want:
                print("[dvfs] duty-cycle sweep")
                report["dvfs"] = section_dvfs(args.engine_dir)
            if "pipeline" in want:
                profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
                print("[pipeline] %d frames, profiles=%s" % (args.frames, profiles))
                cadence_override = parse_cadence(args.cadence) if args.cadence else None
                report["pipeline"] = section_pipeline(
                    args.clip, args.frames, profiles, face_engine, lmk_engine,
                    args.engine_dir, cadence_override
                )
    if "env" in want:
        report["env_after"] = section_env()

    jpath = os.path.join(out_dir, "bench.json")
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    os.replace(tmp, jpath)
    mpath = os.path.join(out_dir, "bench.md")
    with open(mpath, "w") as f:
        f.write(to_markdown(report))
    print("wrote %s and %s" % (jpath, mpath))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
