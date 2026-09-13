# DMS production readiness — assessment and plan

| Field | Value |
| --- | --- |
| **Date** | 2026-09-13 (revised; first issued 2026-09-09) |
| **Board** | Jetson Xavier NX Dev Kit, JetPack 5.1.6, TensorRT 8.5.2.2, 6.7 GiB RAM, `MODE_20W_6CORE` |
| **Architecture** | [`production-dms-design.md`](production-dms-design.md) (rev 5) is the v1 contract; [`architecture.md`](architecture.md) describes what was actually built |
| **This document** | Gap analysis of the *running tree* against that contract |
| **Verdict** | **Not production-ready, but no longer a lab toy.** Phases 0, 1 and 2 are complete in code; Phase 3 (camera) is untouched because no camera exists, and Phase 4 (speed and soak) fails its latency gate by 32%. |

**Production-level** here means the design v1 freeze, **not** UNECE R171 / Euro
NCAP certification (explicit non-goal). Freeze gates from the design:

| Gate | Target | Measured 2026-09-13 | |
| --- | --- | --- | --- |
| e2e p95 at 1280×720, 20 W 6-core | ≤ 66 ms | **87.2 ms** | ✗ |
| Processed frame rate | ≥ 15 FPS | 16.2 FPS | ✓ |
| Peak RSS | < 5.0 GB | 1.34 GB | ✓ |
| Free RAM | ≥ 1.5 GB | 3.05 GB min available | ✓ |
| systemd notify + watchdog, health on localhost | present | present, watchdog gated on frames | ✓ |
| Alerts with hysteresis, local JSONL, optional clip quota | present | present | ✓ |
| Replay works with no camera | required | the only mode that has ever run | ✓ |
| 8 h soak without watchdog fire | required | **never run** | ✗ |

---

## 1. What is actually in the tree

### Works on this NX (measured, this revision)

| Piece | Evidence |
| --- | --- |
| Supervised GStreamer capture with a reconnect ladder | `dms/runtime/supervisor.py`; demonstrated against an absent CSI camera |
| Eleven FP16 TensorRT engines, hash-pinned, all in the frame loop | `engines/MANIFEST.json`; `dms.app` loads and runs every one |
| Media-time state machine (buffer PTS on replay) | `dms/app.py`; golden replay is reproducible run to run |
| Fusion with Schmitt hysteresis, leaky dwell, time-weighted PERCLOS | `dms/state/fusion.py`, `dms/state/driver_state.py` |
| Occlusion gating of the eye and mouth channels | sunglasses → `DROWSINESS_UNRELIABLE`, not "eyes open" |
| Durable `events.jsonl` with boot id, seq, fsync policy, ENOSPC handling | `dms/io/events.py` |
| `/healthz` `/livez` `/readyz` `/metrics` with real status codes | 503 verified during a camera failure |
| Golden replay over nine clips | `tests/replay/golden/`; two consecutive runs identical |
| Strict config schema, ~120 declared keys, cross-section refusals | `dms/config/schema.py`; a typo is now an error |
| Latency budget with an accounting check | 1.1 ms unaccounted of a 60.9 ms mean frame |

### What the reference replay shows

`configs/default.yaml`, 300 frames of `day_driver.mp4`, all nine heads, overlay
off, GPU governor at its 114 MHz idle floor:

```
e2e p50 58.6 ms  p95 87.2 ms  16.2 FPS  peak RSS 1376 MiB
detect_gpu 17.0  headpose 13.4  landmarks 9.6  letterbox 9.3  detect_post 3.6  (p50 ms)
```

The networks are not the cost. `detect_gpu` is 17 ms of wall time around 3.7 ms
of measured GPU compute; the rest is TensorRT host overhead and numpy.

---

## 2. Gap list

Severity: **Blocker** = cannot ship a vehicle image; **High** = false/missed
alerts or ops failure; **Med** = design debt; **Low** = hygiene.
Status: **CLOSED**, **OPEN**, or **PARTIAL**.

### 2.1 Runtime and packaging

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| G1 | systemd not installed on this board | Blocker | **OPEN** — `sudo` needs a password here. The unit, logrotate, tmpfiles and a staged installer exist and are unit-tested; none has ever been loaded. |
| G2 | Production JSONL not on by default | Blocker | **CLOSED** — `events.enabled: true` in every profile; `dms/io/events.py` writes schema-v1 records with fsync policy, rotation and ENOSPC handling. |
| G3 | Clip ring missing | High | **PARTIAL** — implemented, quota/retention/privacy/ENOSPC unit-tested against an injected writer. The `nvv4l2h264enc` + `splitmuxsink` path has never been executed; first enablement must be a supervised bench test. |
| G4 | MANIFEST not enforced at load | High | **CLOSED** — `check_runtime` runs unconditionally, verifies TRT version and every engine sha256, and sets `health.manifest` / `dms_manifest_ok`. `require_engines: true` makes a mismatch fatal. |
| G5 | Leftover `models/` prototype tree | Med | **CLOSED** (Phase 0). |
| G6 | `production.yaml` EAR threshold wrong | High | **CLOSED, and the reverse defect fixed too** — the landmark topology was wrong, so both the old 0.21 *and* the 0.50/0.75 "fix" were meaningless. Every profile is now on the corrected scale and the schema rejects the old one by name. |
| G7 | venv layout for the service | Med | **CLOSED** — the installer stages a release with its own venv and an ABI check. Unrun. |
| G8 | Health unix socket | Low | **OPEN**, deliberately. TCP loopback with an optional bearer token is enough for v1. |
| G9 | 1 Hz tegrastats thread | Low | **OPEN**, deliberately. Sysfs zones + VmRSS + VmHWM, sampled with the health refresh. |
| G10 | *(new)* `camera.fail_fatal: false` meant "report READY for ever with no camera and no retry" | **Blocker** | **CLOSED** — see §3. |

### 2.2 Perception and alerts

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| A1 | No camera ever run | Blocker | **OPEN**. Argus, exposure lock and a physical unplug are all untested. The *software* failure path is now demonstrated (§3). |
| A2 | PnP unstable on profile / sunglasses / hat | High | **CLOSED** — a real head-pose network is the primary path; PnP is the fallback and is now gated on a reprojection residual and a keypoint cross-check, returning confidence 0 instead of a confident wrong angle. |
| A3 | Gaze uses absolute PnP against one `vehicle.yaml` | High | **PARTIAL** — the merge now applies to every profile, and an uncalibrated zero suppresses gaze alerts entirely. No calibration exists, so gaze is unreachable today. |
| A4 | Driver ROI is per-clip YAML | High | **OPEN** — nine YouTube framings are still nine YouTube framings. A vehicle needs seat + camera extrinsics. |
| A5 | EAR is contour, not 6-point | Med | **CLOSED** — roll-invariant inner-lid EAR on the verified topology; open eyes read 0.21–0.40, closed below 0.15. |
| A6 | FACE_LOST uses last *present* sample | Med | **CLOSED** — defined on `DriverSelector.t_last_match_s`, the last matched detection inside the seat ROI. |
| A7 | Greedy IoU tracker | Low | **CLOSED** — BYTE-lite with global assignment and a low-score second pass. |
| A8 | Phone net not built; AGPL | High | **CLOSED** — YOLOX-Nano, **Apache-2.0**, on by default at every 20th frame. The AGPL blocker is gone. |
| A9 | Night / IR illuminator | High once a camera exists | **OPEN** — contract only. |
| A10 | `assume_moving: true` for ever | Med | **OPEN**, documented as fail-safe. `alerts.mute_when_parked` is now *refused* by the schema rather than silently dead. |
| A11 | *(new)* The 106-point landmark table was wrong | **Blocker** | **CLOSED** — re-derived on-board; EAR was being computed over mouth points. |
| A12 | *(new)* The eye classifier inverts below ~20 px of eye width | High | **PARTIAL** — gated by `models.eyestate.min_eye_box_px`, a floor derived from the same footage it protects against. Needs cabin data. |

### 2.3 Performance and reliability

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| P1 | e2e not budgeted; p95 misses the gate | Blocker | **PARTIAL** — now measured properly (87.2 ms p95, 1.1 ms unaccounted) and still fails. Three quantified fixes remain: `Letterbox` in `dms/infer/scrfd.py` (~5 ms), CUDA event reuse in `dms/runtime/trt_engine.py` (~6 ms), SCRFD-512 (~10 ms). |
| P2 | No 8 h soak | Blocker | **OPEN**. Longest run is minutes. |
| P3 | File vs live drop policy | Med | **CLOSED** as designed; live rate still unmeasured. |
| P4 | Idle GPU clock logs CLOCKS_LOW | Low | **CLOSED** — logged once, only when perception is loaded, and the log says the number is not comparable with a full-clock run. |
| P5 | numpy 1.23 vs distro OpenCV | Med | **OPEN** — the installer has the ABI check; the lab venv is still mixed. |
| P6 | NVDEC backpressure warnings | Low | **OPEN**, harmless. |
| P7 | *(new)* The state machine ran on wall time on replay | **Blocker** | **CLOSED** — media time from buffer PTS. Every archived `events.jsonl` from before this change is timing-distorted and should be discarded. |
| P8 | *(new)* Unbounded `infer_ms` list, sorted twice a second | Med | **CLOSED** — fixed-memory ring buffers. |

### 2.4 Tests and ops

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| T1 | No golden alert counts | High | **CLOSED** — `tests/replay/golden/`, nine clips, banded counts, a `--record` mode and a regression runner. |
| T2 | Replay mp4s gitignored | Med | **PARTIAL** — the goldens and the `.npz` perception fixtures are committed, so a regression is catchable without the video; reproducing a *recording* still needs the clips. |
| T3 | No `latency_budget.json` gate | High | **CLOSED** — written at shutdown via `DMS_LATENCY_BUDGET_OUT`, with a four-state verdict and an explicit refusal to pass on unmeasured RSS. |
| T4 | readme stale | Low | **CLOSED**. |
| T5 | Top-level `main.py` + `config/` template | Med | **CLOSED** (Phase 0). |
| T6 | *(new)* No CI | Med | **OPEN** — `pytest.ini` now declares `pythonpath`, `--strict-markers` and a `board` marker so a CI job can run `pytest -m "not board"`. Nothing runs it. |

---

## 3. The camera-failure defect, and what replaced it

**The defect.** `configs/production.yaml` carried `camera.fail_fatal: false`, and
the old `dms/app.py` treated that as: log a warning, `sd_notify("READY=1")`, then
loop for ever sending `WATCHDOG=1` once a second with no camera, no retry, and
`/healthz` still 200. A vehicle whose camera died at boot would have reported
itself healthy indefinitely.

**The replacement.** `camera.fail_fatal` now decides only what a *startup* failure
means. Everything else is explicit policy:

```yaml
camera:
  fail_fatal: false        # a camera absent at boot is not fatal
  retry_forever: true      # a brownout should self-heal
  backoff_s: [0.5, 1.0, 2.0, 5.0, 5.0]
  stall_timeout_s: 5.0
  first_frame_timeout_s: 10.0
  starved_below_fps: 5.0
```

Demonstrated on this board with `source.type: csi` and no camera:

* the ladder runs 0.5 / 1 / 2 / 5 / 5 s and then reports `camera=failed (still retrying)`;
* exactly one `camera_fail` CRITICAL **edge** reaches `events.jsonl`;
* `GET /healthz` → **503**, `GET /readyz` → 503, `GET /livez` → 200;
* `dms_camera_up 0`, `dms_alerts_enter_total{type="camera_fail"} 1`;
* `WATCHDOG=1` is **withheld** while no frames arrive, so `WatchdogSec=30`
  escalates a permanently dead camera to a unit restart;
* the process stays alive and recovers on its own if the camera returns.

---

## 4. Remaining plan

### Phase 3 — Camera (blocked on hardware)

| Work | Acceptance |
| --- | --- |
| Attach a CSI or USB camera; lock `source.type` | 720p BGRx, `health.camera=ok`, READY < 90 s |
| Physical unplug | `CAMERA_FAIL`, ladder, `/healthz` 503, watchdog silence, recovery on replug |
| Operator calibration in the real seat | `configs/vehicle.yaml` with a measured `forward_zero` and `forward_zero_calibrated: true`; gaze alerts become reachable |
| Seat ROI from `seat: lhd/rhd` and camera extrinsics | One `production.yaml` + one `vehicle.yaml`, not nine framings |
| Night / IR | Y-mean gate; no PHONE from RGB below the light threshold |

**Exit:** 10 min live run; FACE_LOST only when the driver leaves frame.

### Phase 4 — Speed and soak

| Work | Expected | Owner file |
| --- | --- | --- |
| `Letterbox` in `ScrfdDetector.preprocess` | −5 ms p50 | `dms/infer/scrfd.py` |
| One CUDA event per engine instead of create/destroy per inference | −6 ms/frame | `dms/runtime/trt_engine.py` |
| SCRFD-512 **after** a cabin-recall check | −10 ms p50 | `configs/*.yaml` |
| 8 h looped replay | RSS stable, no watchdog fire, no FD leak | — |
| Thermal: drop the object net as well as the overlay above 80 °C | rehearse with tegrastats | `dms/app.py` |

**Exit:** `latency_budget.json` from this NX with `gate.pass = true`, plus a soak log.

### Phase 5 — Cabin data (was "phone"; phone is done)

The phone detector shipped. The remaining product risk is that **nothing in the
alert layer has been validated on driver-facing footage**. Two clips with labelled
events (3 × eyes closed ≥ 2 s, 3 × yawns, 3 × over-shoulder looks) would let the
eye and mouth operating points, the head-pose pitch scale and the golden bands be
set from measurement instead of judgement. Until then every threshold is a
defensible starting point and the golden fixtures pin software behaviour only.

### Phase 6 — Freeze

Tag `v1.0-nx` with: engines + MANIFEST hashes, `production.yaml`, the soak log,
the golden fixtures, a passing `latency_budget.json`, and the known-limitations
list from [`STATUS.md`](STATUS.md).

---

## 5. What not to do

- Do not `pip install` uniface, opencv-python, TensorFlow, Torch, ONNX, ORT,
  pycuda, ultralytics or mediapipe on the NX.
- Do not install DeepStream to "make it production".
- Do not treat the YouTube ROI YAMLs as vehicle calibration.
- Do not enable `models.backends.gaze` — the weights are Gaze360, research use only.
- Do not enable clips or the display without `privacy.record_faces` (both are refused).
- Do not call `nvpmodel` / `jetson_clocks` from the app.
- Do not claim UNECE or Euro NCAP.
- Do not compare two latency numbers taken in different `flock` sessions: this
  board's GPU governor has been observed to move an identical inference by 5×.
- Do not re-record the golden fixtures to make a regression go away. Read the diff.
