# DMS — Driver Monitoring System for Jetson Xavier NX

A single-process, single-threaded-inference driver monitor. Native TensorRT FP16,
GStreamer NVMM capture, no DeepStream, no PyTorch on the board. Everything runs on
one Xavier NX at 20 W / 6 cores.

**Status: integrated and running end to end, not vehicle-validated.** It has never
seen a camera or a real cabin. Read [`docs/STATUS.md`](docs/STATUS.md) before
believing any number here, and [`docs/production-readiness.md`](docs/production-readiness.md)
for the gap list.

---

## 1. Quick start

```bash
cd /home/nvidia/myspace/DMS

# unit tests (CPU only, no GPU needed)
PYTHONPATH=. ./.venv/bin/python -m pytest -q

# smoke: synthetic source, 30 frames
PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/testsrc.yaml --max-frames 30

# full pipeline on a replay clip
PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300

# golden replay regression over all nine clips
PYTHONPATH=. ./.venv/bin/python tests/replay/golden/run_golden.py
```

Anything that touches the GPU on a shared board must hold the mutex:

```bash
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
```

Perception is **on by default**; `--no-detect` runs capture and health only.

---

## 2. What one frame does

```
CaptureSupervisor.read()            supervised GStreamer appsink, reconnect ladder
  -> BGRA -> contiguous BGR         one copy, reused by everything
  -> SCRFD-500m                     letterbox / engine / decode, timed separately
  -> IoU tracker + DriverSelector   hysteretic driver-vs-passenger pick
  -> 2d106det                       106 landmarks for the driver only
  -> OcclusionDetector              FIRST: sunglasses + mask gate the next two
  -> EyeState / MouthState          OCEC + VSDLM, EAR / MAR fallback
  -> HeadPose                       6DRepNet MobileNetV2, solvePnP fallback
  -> ObjectDetector / Identity      YOLOX-Nano phone (every 20th), SFace (per track)
  -> DriverSignals -> DriverMonitor fusion, Schmitt hysteresis, alert EDGES
  -> events.jsonl / GPIO / clips / /healthz / /metrics
```

The state machine runs on **media time** (GStreamer buffer PTS) for a file or test
source and on **monotonic** time for a live camera. When no frame arrives the clock
is still advanced by real elapsed time, so a frozen camera produces `FACE_LOST`
rather than silence.

---

## 3. Models

All engines are FP16, fully static, built on-board by `scripts/build_engines.py`
from hash-pinned ONNX. `engines/MANIFEST.json` is the source of truth; the app
verifies every engine's sha256 against it at startup and refuses to start when
`require_engines: true` and the hash does not match.

| Capability | Engine | Licence | State |
| --- | --- | --- | --- |
| Face detection | `scrfd_500m_640_fp16_gpu` (and `_512_`) | InsightFace research terms | live |
| Landmarks (106) | `2d106det_192_fp16_gpu` | InsightFace research terms | live |
| Head pose | `hpe_mobilenetv2_224_fp16_gpu` | MIT (6DRepNet-style) | live, solvePnP fallback |
| Head pose (fallback) | `hpe_mobilenetv3_small_224_fp16_gpu` | MIT | loaded only if the primary is absent |
| Eye open/closed | `ocec_s_24x40_b2_fp16_gpu` (batch 2) | Apache-2.0 | live, EAR fallback under 20 px |
| Mouth open/closed | `vsdlm_s_30x48_fp16_gpu` | Apache-2.0 | live, MAR fallback under 30 px |
| Sunglasses gate | `sgc_is_s_48_fp16_gpu` | Apache-2.0 | live, every 15th frame per track |
| Mask gate | `mwc_n_48_fp16_gpu` | Apache-2.0 | live, every 15th frame per track |
| Phone / cabin objects | `yolox_nano_416_fp16_gpu` | **Apache-2.0** | live, every 20th frame |
| Driver identity | `sface_112_fp16_gpu` | Apache-2.0 | live, per new track id |
| Eye gaze | `mobilenetv2_gaze_448_fp16_gpu` | **Gaze360: research / non-commercial only** | **OFF and not even deserialised** |

Two licence notes that matter:

* **The phone detector is no longer AGPL.** The old readme said "Phone YOLO is off
  (AGPL)". That is obsolete: YOLOX-Nano is Apache-2.0 and the capability is on by
  default.
* **Gaze is licence-gated in code, not in prose.** `models.backends.gaze` defaults
  to `off`, and setting it to `model` without `models.research_models_allowed: true`
  is a config error. Even when requested, the registry refuses to deserialise the
  engine unless the research flag is set. No shipped profile sets it.

Every capability resolves to `model`, `geometry` or `off` and reports **what
actually happened**, never what was requested. A capability that asked for a model
and got the geometric fallback appears in `degraded_capabilities()`, raises
`MODEL_DEGRADED`, and shows in `/healthz`.

---

## 4. Alerts

`FACE_LOST`, `MICROSLEEP`, `PERCLOS_HIGH`, `YAWN`, `FATIGUE_CLUSTER`, `GAZE_AWAY`,
`PHONE`, `DISTRACTION`, plus the honesty alerts `DROWSINESS_UNRELIABLE`,
`DRIVER_CHANGED`, `MODEL_DEGRADED`, and the ops alerts `CAMERA_FAIL`,
`PIPELINE_STARVED`, `THERMAL_THROTTLE`, `ENGINE_FAIL`, `DISK_FULL`. Every one of
them now has a producer.

Each alert is a leaky-integrator latch with separate enter/exit dwell and a
cooldown measured from the enter edge. The edge grammar per type is
`enter (escalate)* exit`. Edges are written to `events.jsonl` with a schema
version, a wall clock, a monotonic clock, the kernel boot id and a sequence
number, so a post-incident reconstruction survives a reboot and an NTP jump.

Gating that is enforced, not documented:

* sunglasses detected → the eye channel is `UNAVAILABLE`; `MICROSLEEP` and
  `PERCLOS_HIGH` cannot fire and `DROWSINESS_UNRELIABLE` is raised instead;
* mask detected → `YAWN` suppressed;
* pose confidence below the gate, or **no calibration**, → `GAZE_AWAY` impossible.

Because no unit has been calibrated, `GAZE_AWAY` and the gaze half of
`DISTRACTION` are currently unreachable on every shipped profile. That is
deliberate and visible: `DROWSINESS_UNRELIABLE` fires instead and `/healthz`
carries the note.

---

## 5. Configuration

`dms/config/schema.py` is strict: **every section forbids unknown keys**, values
are range-checked, and cross-section contradictions are refused. A typo in a
drowsiness threshold is an error at startup, not a silently-ignored line.

| Profile | Purpose |
| --- | --- |
| `configs/default.yaml` | Lab replay of `day_driver.mp4`; the golden fixtures are recorded against it |
| `configs/production.yaml` | Vehicle: CSI, hash-verified engines, fsynced events, mandatory health, capture retried forever |
| `configs/testsrc.yaml` | `videotestsrc` smoke; nothing written to disk |
| `configs/systemd-lab.yaml` | Headless lab unit until a camera exists |
| `configs/<clip>.yaml` | One per replay clip: seat ROI and detector confidence for that framing |
| `configs/vehicle.yaml` | Per-unit calibration, merged over **every** profile. Ships empty. |

**The EAR and MAR scale changed.** The 106-point landmark table this project
computed every drowsiness signal from was wrong: "EAR" was measured over mouth
points and "MAR" over an eye. The corrected roll-invariant EAR reads 0.21–0.40 on
open eyes, so the old `ear_closed: 0.50 / ear_open_median: 0.75` would latch
`MICROSLEEP` CRITICAL on every frame. The schema now **rejects** a config on the
old scale by name.

Privacy defaults are off and enforced at startup: `--save-video`, `--save-preview`
and `display.enabled` are all refused unless `privacy.record_faces` is true, and
the clip ring refuses to encode without it.

---

## 6. Operations

| Route | Codes | Meaning |
| --- | --- | --- |
| `/healthz` | 200 / 503 | 503 when not ok or the snapshot is stale |
| `/livez` | 200 | the process is alive; never fails |
| `/readyz` | 200 / 503 | engines loaded, camera ok, a frame has arrived, fresh |
| `/metrics` | 200 | Prometheus text, ~40 series, stdlib only |

The endpoint binds loopback. A non-loopback bind is refused unless
`health.allow_remote` is set, because `/healthz` publishes live driver alert state.

`systemd`: `Type=notify`, `WATCHDOG=1` is sent **only while frames are actually
arriving**. A dead camera therefore goes silent and `WatchdogSec` restarts the
unit — the previous build pinged for ever from its camera-failure loop. See
[`deploy/README.md`](deploy/README.md).

---

## 7. Measured performance (2026-09-13, this board)

300 frames of `day_driver.mp4`, all nine heads live, overlay off, GPU governor at
its 114 MHz idle floor, board shared with other work:

| | value | gate |
| --- | --- | --- |
| e2e p50 | 58.6 ms | — |
| **e2e p95** | **87.2 ms** | ≤ 66 ms → **FAIL** |
| processed FPS | 16.2 | ≥ 15 → pass |
| **peak RSS** | **1376 MiB** | < 5000 MiB → **pass** |
| unaccounted time | 1.1 ms of 60.9 ms | instrumentation covers the loop |

Biggest stages (p50): `detect_gpu` 17.0, `headpose` 13.4, `landmarks` 9.6,
`letterbox` 9.3, `detect_post` 3.6. The pipeline is dominated by the Python data
path and TensorRT host overhead, not by the networks — `detect_gpu` is 17 ms of
wall time around 3.7 ms of measured GPU compute.

The three fixes that close the gap are all in files this workstream does not own
and are listed in the handoff: `Letterbox` in `dms/infer/scrfd.py` (~5 ms),
per-engine CUDA event reuse in `dms/runtime/trt_engine.py` (~6 ms), and the
SCRFD-512 engine (~10 ms).

**SCRFD 512 vs 640, measured over 200 frames per clip:**

| clip | 640 detections | 512 detections | 640 p50 | 512 p50 |
| --- | --- | --- | --- | --- |
| `day_driver` (small faces) | 483 | **352** | 58.3 ms | 48.4 ms |
| `minivan` (in-cabin) | 800 | 800 | 56.5 ms | 42.7 ms |
| `red_car_crash` (in-cabin) | 200 | 200 | 55.0 ms | 42.6 ms |

512 is free on in-cabin framing and costs 27% of detections on small dashcam
faces. The profiles ship 640; the swap is a one-line config change with the
numbers written next to it.

Reproduce: `DMS_LATENCY_BUDGET_OUT=/tmp/lb.json ... -m dms.app --config configs/default.yaml --max-frames 300`.

---

## 8. What this is not

* **Not vehicle-validated.** No camera has ever been attached. `/dev/video*` does
  not exist on this board; everything is replay.
* **Not calibrated.** No `forward_zero` exists for any installation, so every
  absolute head angle is a relative signal and gaze alerts are suppressed.
* **Not accuracy-validated.** The nine replay clips are dashcam montages, not
  cabin footage. No microsleep, yawn or over-shoulder look has ever been labelled.
  The eye classifier was *measured* to invert below 20 px of eye width on this
  footage; the floor that protects against it is derived from the same
  measurement it protects against.
* **Not soaked.** The longest run is a few minutes. No 8 h result exists.
* **Not certified.** UNECE R171 / Euro NCAP are explicit non-goals.
* **Not installed.** `deploy/dms.service` has never been loaded: `sudo` on this
  board needs a password.

---

## 9. Layout

```
dms/app.py            assembly and the frame loop
dms/config/           strict schema + loader
dms/capture/          GStreamer pipelines and the hardened appsink source
dms/runtime/          TensorRT lifetime, manifest enforcement, capture supervisor
dms/infer/            SCRFD, 2d106det, and the seven model wrappers + registry
dms/geometry/         106-point topology, EAR, MAR, solvePnP
dms/state/            fusion, hysteresis, the alert state machine
dms/track/            IoU tracker, hysteretic driver selection
dms/io/               events, health, metrics, clips, thermal, GPIO, sd_notify
dms/perf/             stage timers and the latency budget
dms/viz/              overlay renderer (allocation-free, opt-in)
engines/              *.engine + MANIFEST.json (artifacts gitignored)
tests/replay/golden/  golden alert-count fixtures and their runner
deploy/               systemd unit, logrotate, tmpfiles, installer
```

Constraints that bind: Python 3.8, TensorRT 8.5.2.2, CUDA 11.4, system OpenCV
4.5.4 (GStreamer, no CUDA). Never `pip install` torch, tensorflow, onnx,
onnxruntime, pycuda, opencv-python, ultralytics or mediapipe on this board.
