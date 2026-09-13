# DMS — Driver Monitoring System for Jetson Xavier NX

A single-process, single-threaded-inference driver monitor. Native TensorRT FP16,
GStreamer NVMM capture, no DeepStream, no PyTorch on the board. Everything runs on
one Xavier NX at 20 W / 6 cores.

**Status: integrated and running end to end, not vehicle-validated.** It has never
seen a camera or a real cabin. Read [`docs/STATUS.md`](docs/STATUS.md) before
believing any number here, and [`docs/production-readiness.md`](docs/production-readiness.md)
for the gap list.

Everything measured in this file was re-taken on this board with the GPU mutex held
and with other agents' work running alongside (loadavg 2.4–4.5). The latency and
detector figures were re-measured on **2026-09-13 evening KST**, repeated (n ≥ 3 per
profile, n = 9 on `default`) because the run-to-run spread is several times the gate's
margin; the rest was taken between 16:45 and 17:10 KST the same day. Where a claim
could not be reproduced it has been corrected or deleted.

**Three things a reader should know before the rest:**

1. **The p95 latency gate is missed, on real footage, on both detectors.** Ten of the
   twelve runnable profiles miss `perf.gate_e2e_p95_ms: 66.0` at their measured median
   (§7). That is a recorded, accepted degrade: the unit reports the miss rather than
   being configured around it.
2. **The shipped detector is SCRFD-640 again.** It was swapped to SCRFD-512 earlier the
   same day to buy the gate. Measured: the swap bought 5.4 ms of p50, not the ≈ 13 ms
   the gate needs, and cost 15.8% of the driver-ROI face hits on `day_driver`. It was
   reverted. The 512 point is kept as `configs/lowlatency.yaml` (§7).
3. **The golden replay comparison now runs inside pytest**, one test per clip, behind
   `DMS_REPLAY_TESTS=1`. Until today it ran only when somebody typed it, which is how
   three goldens sat red for a whole fix round while `pytest -q` stayed green.

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

# golden replay regression over all nine clips - normally you do not run this by
# hand, pytest runs it (one test per clip) when DMS_REPLAY_TESTS=1 is set:
flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. \\
    ./.venv/bin/python -m pytest -q tests/test_app_integration.py"

# ... or directly, which is what that test shells out to
PYTHONPATH=. ./.venv/bin/python tests/replay/golden/run_golden.py
```

### The environment gates

Four groups of tests are skipped unless you ask for them, because they need
hardware that a laptop does not have. Nothing else in the suite is gated.

| variable | unlocks | cost |
| --- | --- | --- |
| `DMS_REPLAY_TESTS=1` | the golden replay comparison (9 clips, one test each) and two pipeline smoke tests, in `tests/test_app_integration.py` | GPU, ~6 min, hold the mutex |
| `DMS_TRT_TESTS=1` | TensorRT supervisor tests in `tests/test_supervisor.py` | GPU |

`pytest -q` with neither set is the number quoted below, and it is **not** a
detection-quality result: it does not replay a single frame of video.

Anything that touches the GPU on a shared board must hold the mutex:

```bash
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
```

Perception is **on by default**; `--no-detect` runs capture and health only.

Every one of those five commands was executed as written on 2026-09-13 while this
file was revised. What they print today:

| command | result |
| --- | --- |
| `pytest -q` (no gates set) | **821 passed, 15 skipped in 87.79 s**, 0 failures |
| `pytest -q` with `DMS_REPLAY_TESTS=1`, GPU mutex held | **58 passed in 276.20 s**, 0 failures — the nine golden clips are nine of those tests |
| `--config configs/testsrc.yaml --max-frames 30` | `PERF_GATE PASS e2e_p95=36.59 ms … fps=30.25` |
| `--config configs/default.yaml --max-frames 300` | `PERF_GATE FAIL e2e_p95=… ms limit=66.0 ms … (e2e p95 OVER by … ms)` — nine runs today, median 77.79 ms, **0 passed**; see §7 |
| `run_golden.py` | `all 9 clip(s) within their golden bands` |

The suite's one long-standing failure
(`test_the_legacy_geometry_call_path_still_drives_the_machine`, which fed `ear=0.20`
and expected `MICROSLEEP` on the corrected scale where 0.20 is an *open* eye) is
fixed: the fixture now drives a genuinely closed eye and reads the effective
threshold off the monitor instead of assuming one.

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
from hash-pinned ONNX. `engines/MANIFEST.json` is the source of truth. **Every engine
the configuration will load is sha256-verified at startup** — not two of them, which
is what an earlier build actually did while its own comment claimed the check was
unconditional. A sha256 or size mismatch is now fatal before READY regardless of
`require_engines`, because a swapped plan is not missing metadata: measured, it
reached READY and then killed the process with CUDA error 700.

Verified in a real run today (`configs/default.yaml`):

```
engine provenance: 9 engines verified in 0.10s (require_hash=False):
  2d106det_192, hpe_mobilenetv2_224, mwc_n_48, ocec_s_24x40_b2, scrfd_500m_640,
  sface_112, sgc_is_s_48, vsdlm_s_30x48, yolox_nano_416
```

**Thirteen `.engine` files are on disk. Nine deserialise in a run. Four never do.**

`state` below is deliberately not the word "live", which reads as "working". It is
one of **real** (produces useful output, with evidence), **weak** (loads, runs, and
its output is measurably poor or unvalidated), **fallback** or **disabled**.

| Capability | Engine | Licence | State | Evidence |
| --- | --- | --- | --- | --- |
| Face detection | `scrfd_500m_640_fp16_gpu` | InsightFace research terms | **real** | 583 face hits over 300 frames of `day_driver`, 1201 over `minivan`, measured today on nine consecutive runs with identical counts |
| Face detection (alt) | `scrfd_500m_512_fp16_gpu` | InsightFace research terms | **not loaded by any shipped profile** | selected only by `configs/lowlatency.yaml`. Costs 15.8% of the driver-ROI face hits on `day_driver` (1105 → 930 frames) for 5.4 ms of p50; see §7 |
| Landmarks (106) | `2d106det_192_fp16_gpu` | InsightFace research terms | **real** | drives EAR/MAR on the re-derived topology; open eyes read 0.21–0.40, closed below 0.15 |
| Head pose | `hpe_mobilenetv2_224_fp16_gpu` | MIT (6DRepNet-style) | **real, uncalibrated** | loads and runs every frame; solvePnP fallback is gated on a reprojection residual. No `forward_zero` exists, so every absolute angle is a relative signal |
| Head pose (fallback) | `hpe_mobilenetv3_small_224_fp16_gpu` | MIT | **not loaded** | fallback only; the primary is present |
| Eye open/closed | `ocec_s_24x40_b2_fp16_gpu` (batch 2) | Apache-2.0 | **weak** | measured through this wrapper over 6350 eyes on the replay corpus: only **16% reach the model path** (84% are below the 20 px floor and use the EAR ratio); on the model path corr(prob_open, eye ratio) **+0.29** while OCEC calls 3× as many eyes closed as the geometry does. Below the floor the correlation is **−0.00** over all 6350 eyes. `MIN_EYE_BOX_PX` exists to refuse that |
| Eye open/closed (alt) | `ocec_s_24x40_fp16_gpu` (batch 1) | Apache-2.0 | **not loaded** | the batch-2 engine is selected |
| Mouth open/closed | `vsdlm_s_30x48_fp16_gpu` | Apache-2.0 | **weak, miscalibrated** | measured over 3175 mouths: on the model path VSDLM calls **52.4%** of mouths open where the inner-lip ratio calls **18.3%**, corr **+0.14**. Corroborated by my own golden run today, which raised YAWN on `red_car_crash` (×2), `guy_rear_ended` (×2) and `rear_end_whiplash` (×4) — dashcam clips in which nobody yawns |
| Sunglasses gate | `sgc_is_s_48_fp16_gpu` | Apache-2.0 | **real on 2 of 4 positives** | re-measured by me today at the **shipped** threshold 0.50, every 15th frame, largest face, each clip's own config: `bmw_rear_ended` 1.00, `family_vacation_crash` 0.75, `getting_rear_ended` 0.28, `minivan` **0.14** of refreshes declare sunglasses; **all five labelled negatives 0.00** |
| Mask gate | `mwc_n_48_fp16_gpu` | Apache-2.0 | **weak, positive class never exercised** | no clip in this repository contains a masked face, so `prob_masked` has never seen a true positive. It is not clean on negatives either: my run today fires on **11%** of `bmw_rear_ended` refreshes (a bearded driver) and 0.00 everywhere else |
| Phone / cabin objects | `yolox_nano_416_fp16_gpu` | **Apache-2.0** | **loaded, never exercised** | runs every 20th frame; **no clip in the corpus contains a phone**, so no `PHONE` alert has ever been raised from real footage. The class-to-alert path is unit-tested only |
| Driver identity | `sface_112_fp16_gpu` | Apache-2.0 | **real, and detector-sensitive** | it does re-identify: **4** `driver_changed` edges (2 enter, 2 exit) over 300 frames of `day_driver` on the shipped SCRFD-640, inside the golden band of [1, 3] each. On the SCRFD-512 detector the same clip produced **20** (10 enter, 10 exit) — the identity layer is the most detector-sensitive thing in the pipeline and it is what put 3 of the 9 goldens into regression during the 512 experiment (§7) |
| Eye gaze | `mobilenetv2_gaze_448_fp16_gpu` | **Gaze360: research / non-commercial only** | **disabled** | licence-gated off in every profile and **not deserialised** — confirmed absent from the 9-engine provenance line above |

**On the two "weak" eye/mouth rows and the sunglasses row.** The 6350-eye and
3175-mouth figures were measured by the model workstream on 2026-09-13 and are
recorded in the `dms/infer/eyestate.py` and `dms/infer/mouthstate.py` docstrings;
I did not re-run them. The sunglasses and mask figures **are** my own re-measurement,
run today through the shipped `OcclusionDetector` at each clip's own config, and they
disagree with the numbers in `dms/infer/occlusion.py` on the two hard positives — that
docstring records `minivan` 0.29 and `getting_rear_ended` 0.41 at threshold 0.50 where
I measure 0.14 and 0.28. Two plausible causes: `dms/infer/occlusion.py` was still
being edited while I measured (mtime 16:41, my run 17:05), and my probe drives one
persistent track id whereas the app's tracker restarts ids across a montage cut, which
resets the 5-refresh vote window. **Treat both sets as order-of-magnitude.** The
direction is not in doubt: the gate finds two of the four sunglasses drivers reliably,
one marginally, and one poorly, and it does not false-positive.

One more configuration note before the licences: every profile pins
`occlusion.sunglasses_threshold: 0.50`, which overrides the module's own measured
default of 0.40. At 0.40 the hard positives improve (`minivan` 0.14 → 0.20,
`getting_rear_ended` 0.28 → 0.32, `family_vacation_crash` 0.75 → 0.88, measured today)
and the negatives stay at 0.00. Missing sunglasses is the more dangerous error, so
0.40 is the recommended value; the configs have not been changed because they are not
this document's to change.

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
them has a producer in code — which is a different and much weaker claim than "every
one of them has been observed to fire correctly". The per-type ledger of what is
actually pinned by a test, and what is not pinned by anything, is
`tests/replay/golden/synthetic/alert_coverage.json`; `ENGINE_FAIL` has no
behavioural test at all. See §8.

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

## 7. Measured performance and the golden replay (2026-09-13, this board)

All runs below: 300 frames per run, GPU mutex held, one `dms.app` at a time, overlay
off, GPU governor left alone, `nvpmodel` 8, board shared with other agents. The gate
is `perf.gate_e2e_p95_ms: 66.0` and `perf.gate_peak_rss_mb: 5000.0`; the app evaluates
it itself at every health refresh and prints one `PERF_GATE PASS|FAIL|UNDECIDED` line
per run, now with the half that decided the verdict and the signed margin.

### Everything here is a median of several runs, and that is not pedantry

`configs/default.yaml`, the same 300 frames of `day_driver.mp4`, **nine runs** today:

```
66.50  68.75  72.21  73.22  77.79  80.89  81.30  86.93  99.45   (e2e p95, ms)
                            ^^^^^ median                        gate = 66.0
```

**Spread 32.95 ms against a gate margin measured in single milliseconds, and 0 of the
9 runs passed.** Same binary, same clip, same config, inside one hour. Every number
below is therefore a median with its range, and any single `PERF_GATE PASS` you
happen to see on this board is a sample from that distribution, not a property of the
software. `CLOCKS_LOW` fires during sweeps at a peak GPU clock of 204 MHz against a
1109 MHz ceiling, which is the mechanism.

### Every shipped profile, n = 3 runs each (n = 9 for `default` and `lowlatency`)

| profile | frames | e2e p95 median | range | FPS median | peak RSS | face hits | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `testsrc` | 300 | **35.12** | 34.78–35.75 | 29.5 | 1341 MiB | 0 | **PASS** 3/3 |
| `systemd-lab` | 300 | **36.50** | 35.42–39.83 | 28.6 | 1331 MiB | 0 | **PASS** 3/3 |
| `rear_end_whiplash` | 300 | 69.60 | 69.36–77.05 | 20.4 | 1385 MiB | 552 | **FAIL** 0/3 |
| `bmw_rear_ended` | 300 | 77.12 | 66.17–79.03 | 18.9 | 1383 MiB | 300 | **FAIL** 0/3 |
| `lowlatency` (SCRFD-512) | 300 | 77.15 | 65.26–82.51 | 19.6 | 1374 MiB | 452 | **FAIL** 1/9 |
| `default` (`day_driver`) | 300 | 77.79 | 66.50–99.45 | 18.8 | 1386 MiB | 583 | **FAIL** 0/9 |
| `minivan` | 300 | 80.38 | 65.70–98.32 | 17.3 | 1380 MiB | 1201 | **FAIL** 1/3 |
| `getting_rear_ended` | 300 | 82.16 | 76.36–82.87 | 17.1 | 1382 MiB | 589 | **FAIL** 0/3 |
| `family_vacation_crash` | 300 | 82.57 | 75.92–89.82 | 17.1 | 1372 MiB | 643 | **FAIL** 0/3 |
| `red_car_crash` | 300 | 84.21 | 78.79–91.23 | 16.3 | 1390 MiB | 300 | **FAIL** 0/3 |
| `guy_rear_ended` | 300 | 85.75 | 75.99–93.71 | 16.8 | 1384 MiB | 300 | **FAIL** 0/3 |
| `rear_end_accident` | 186† | 85.79 | 75.06–87.38 | 17.4 | 1389 MiB | 305 | **FAIL** 0/3 |
| `production` | — | — | — | — | — | — | **not runnable here**‡ |

† the clip is shorter than 300 frames. ‡ `source.type: csi` and there is no camera —
but it does not get that far: it loads and hash-verifies all nine engines and then
exits on `EVENT_LOG_FATAL [Errno 13] Permission denied: '/var/lib/dms'`, which needs
root to fix. Both reasons are real; the event log is the one you hit first.

**Verdict: 2 PASS, 10 FAIL, 1 unrunnable.** The two that pass render a synthetic
pattern and find zero faces — they exercise capture, the engines and the loop, not the
per-face cascade. **No profile that does real work met the gate at its median.**

Memory and throughput are not the problem and never have been: **peak RSS
1331–1390 MiB against 5000 MiB** (27% of budget) and **16.3–29.5 FPS against ≥ 15**.

`tests/test_config.py` now pins all three of those facts: the gate value itself
(`PRODUCT_GATE_E2E_P95_MS`, so it cannot be raised to manufacture a pass), a repeated
per-profile measurement for every shipped profile, and `KNOWN_GATE_MISSES`, which must
agree with the measurements in both directions. A profile may miss its gate. It may
not miss it silently.

### The SCRFD 640 → 512 swap, and why it was reverted the same day

Earlier on 2026-09-13 all twelve profiles were repointed from
`engines/scrfd_500m_640_fp16_gpu.engine` to `scrfd_500m_512_fp16_gpu.engine` to get
inside the 66 ms gate. It has been reverted. Both halves of the trade were
re-measured today.

**What it bought — paired A/B, six pairs, alternating which engine ran first, 300
frames each, so thermal drift and whatever else the board was doing cannot favour one
side:**

| | SCRFD-640 | SCRFD-512 | paired delta |
| --- | ---: | ---: | ---: |
| e2e p95 median | 79.55 ms | 74.39 ms | **+7.44 ms** (640 slower in 5 of 6 pairs) |
| e2e p50 median | 52.25 ms | 47.40 ms | **+5.40 ms** |
| runs passing the 66 ms gate | **0 of 6** | **0 of 6** | — |

The 5.4 ms of p50 is real and reproduces the figure `docs/performance.md` §7
predicted. It is also not enough: the median run needs to come down by ~13 ms to meet
the gate, and **neither engine passed a single one of those twelve runs**. The trade
was made to buy a pass and it does not buy one. (Two sequential single-engine sweeps
earlier in the evening disagreed about the sign of the difference — 512 sweep median
78.51 ms, 640 sweep median 72.21 ms — which is what forced the paired design. Never
compare two sweeps on this board.)

**What it cost — both engines through the shipped `ScrfdDetector` over all nine replay
clips (16,451 frames, conf 0.5), counting frames whose face centre lands in
`driver_roi` [0, 0, 0.65, 1.0]:**

| clip | frames | 640 | 512 | delta |
| --- | ---: | ---: | ---: | ---: |
| `day_driver` | 6218 | 1105 | 930 | **−15.8%** |
| `rear_end_accident` | 186 | 160 | 153 | −4.4% |
| `rear_end_whiplash` | 528 | 263 | 257 | −2.3% |
| `family_vacation_crash` | 1873 | 1609 | 1585 | −1.5% |
| `minivan` | 1440 | 1431 | 1416 | −1.0% |
| `red_car_crash` | 2808 | 2767 | 2743 | −0.9% |
| `bmw_rear_ended` | 398 | 297 | 297 | 0.0% |
| `guy_rear_ended` | 1830 | 0 | 0 | 0.0% (no face found either way) |
| `getting_rear_ended` | 1170 | 1091 | 1105 | +1.3% |
| **total** | **16451** | **8723** | **8486** | **−2.7%** |

(The corpus figures in this table are the reviewer's re-measurement over the full
16,451-frame corpus, not mine; my own runs reproduce the per-clip `total_face_hits`
that follow from them exactly — see the golden section below.)

98% of the 176 `day_driver` frames the 512 loses are **≥ 60 px wide, median 75 px** —
the driver at medium range, not background faces. The downstream consequence is not a
statistic: over the same 300 frames, `driver_changed` enter/exit goes **2 → 10** against
a golden band of [1, 3]. The identity layer loses and re-acquires the driver five
times more often. The one mitigating measurement: the longest run of consecutive
frames on which 512 loses the driver is 11 (≈ 0.37 s at 30 fps), short of the 1.0 s
`face_lost_s` dwell, so it does not raise `FACE_LOST` on its own.

The 512 operating point is kept, documented, as **`configs/lowlatency.yaml`** —
identical to `default.yaml` except for the detector and its letterbox, with the whole
table above restated in its own header. It is not the default and no golden is
recorded against it.

### Golden replay: 9 of 9 green, and it is now a pytest failure when it is not

```
flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. \
    ./.venv/bin/python -m pytest -q tests/test_app_integration.py"
-> 58 passed in 276.20s (0:04:36)
```

All nine fixtures are back inside their recorded bands with **no band re-recorded**.
The revert restored them exactly: measured `total_face_hits` on the 640 detector is
583 / 1201 / 300 / 643 / 305 / 300 / 300 / 589 / 552, and the midpoint of each
recorded `face_hits` band is 583 / 1201 / 300 / 643 / 305 / 300 / 300 / 589 / 552.
Nine of nine, exact. (That is also the evidence used to stamp the pre-existing
fixtures with `detector_engine: scrfd_500m_640` — schema 3 — without re-recording
them: under the 512 detector five of those nine numbers differ.)

**The hole this closes.** Until today the comparison above was not part of `pytest`.
The suite contained three structural tests over the fixtures (schema shape, files
exist, alert names are real enum members) and one board-gated test that ran a single
clip. So `pytest -q` could be, and was, green while three fixtures were red and
driver-face recall on the flagship clip had dropped 15.8%. There are now two layers:

* **`test_golden_detector_matches_the_profile_today`** — no GPU, no clip, no engine
  loaded. It compares the detector each fixture was recorded against with the detector
  its profile selects now. Reintroducing the 512 default makes `pytest -q` red in
  **15 seconds on any machine**, naming the fixture and both engines.
* **`test_golden_replay[<clip>]`** — one test per clip behind `DMS_REPLAY_TESTS=1`,
  which runs the real comparison and fails on any band violation.

Demonstrated, not asserted. With the 512 default put back in `configs/default.yaml`:

```
FAILED tests/test_app_integration.py::test_golden_detector_matches_the_profile_today[day_driver.json]
FAILED tests/test_config.py::test_only_the_low_latency_profile_ships_a_different_detector[default.yaml]
FAILED tests/test_config.py::test_the_low_latency_profile_exists_and_is_only_a_detector_swap
3 failed, 215 passed, 11 skipped in 15.19s
```

and on the board, the harness itself:

```
REGRESSION day_driver: golden was recorded against scrfd_500m_640_fp16_gpu.engine,
                       the profile now selects scrfd_500m_512_fp16_gpu.engine
REGRESSION day_driver: driver_changed/enter fired 10 times, golden band [1, 3]
REGRESSION day_driver: driver_changed/exit  fired 10 times, golden band [1, 3]
day_driver  frames=300 faces=452 p50=44.3 p95=69.7 gate=FAIL
```

`configs/default.yaml` was then restored and verified byte-identical by `md5sum -c`.

### The gate miss is reported, not hidden

One run of `configs/default.yaml`, 400 frames, `curl` against `/healthz` at t = 25 s:

```json
"degraded": true,
"latency_ms": { "e2e_p50": 54.7, "e2e_p95": 84.7, "gate_limit_p95": 66.0 },
"notes": ["perf gate FAIL: e2e p95 84.74 ms vs 66.0 ms limit, peak rss 1384 vs
           5000.0 MiB - e2e p95 OVER by 18.74 ms (sustained: degraded)"]
```

```
dms_perf_gate_pass 0          dms_perf_gate_e2e_p95_ms 84.74
dms_perf_gate_e2e_p95_limit_ms 66      dms_perf_gate_e2e_samples 244
```

and at shutdown:

```
PERF_GATE FAIL e2e_p95=70.69 ms limit=66.0 ms peak_rss=1384 MiB limit=5000.0 MiB
          samples=300 fps=17.68 config=default.yaml (e2e p95 OVER by 4.69 ms)
```

`gate_limit_p95` in `latency_ms` is new: `/healthz` has no first-class `perf_gate`
object (that belongs in `dms/io/health.py`), so without the limit travelling next to
the measurement a scraper saw `e2e_p95: 84.7` and could not tell it was a miss until
the 10 s degrade dwell flipped `degraded`.

Also fixed: a run too short to judge used to print `PERF_GATE UNDECIDED` and, two
lines later, `wrote /tmp/lb.json gate=FAIL` — because `gate['pass']` is False both for
an unjudgeable window and for a missed one. Both lines and the `/healthz` note now
come from one `_gate_verdict()`:

```
PERF_GATE UNDECIDED e2e_p95=34.86 ms limit=66.0 ms ... samples=15 (only 15 of 20 e2e samples, not judgeable)
wrote /tmp/ws_lb.json gate=UNDECIDED e2e_p95=34.86 ms peak_rss=1331 MiB (only 15 of 20 e2e samples, not judgeable)
```

### Where the time goes

Reproduce with:

```bash
DMS_LATENCY_BUDGET_OUT=/tmp/lb.json flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
```

The per-stage breakdown in [`docs/performance.md`](docs/performance.md) predates the
`Letterbox` rewiring and was taken on the 512 detector, so it is 5.4 ms of p50 optimistic
against what ships today; the stage totals in
`latency_budget.json` from a run you make yourself are the current source of truth.
What has not changed is the shape of the problem: the networks are not the cost. The
whole model zoo is under 25 ms of pure GPU compute, and the pipeline spends its budget
on the Python data path, TensorRT host round-trip overhead, and a GPU governor that
was observed at 204 MHz against an 1109 MHz ceiling during these runs.

## 8. What this is not

Read this section as the counterweight to every number above.

* **Not vehicle-validated.** No camera has ever been attached. `/dev/video*` does
  not exist on this board; everything is replay. `configs/production.yaml` — the
  vehicle profile — cannot be run here at all. Three documents used to give the reason
  as "`csi`, no camera"; that is true but it is not the error you get. The profile
  loads and hash-verifies all nine engines and then exits on
  `EVENT_LOG_FATAL [Errno 13] Permission denied: /var/lib/dms`, before it reaches
  the camera. Both are real blockers and the event-log one needs root.
* **Not calibrated.** No `forward_zero` exists for any installation, so every
  absolute head angle is a relative signal and `GAZE_AWAY` and the gaze half of
  `DISTRACTION` are unreachable on every shipped profile. That is deliberate and
  visible, not an omission.
* **Not accuracy-validated, and two of the drowsiness models are measurably
  poor.** The nine replay clips are dashcam montages, not cabin footage. No
  microsleep, yawn or over-shoulder look has ever been labelled anywhere in this
  repository. Specifically: OCEC reaches its model path on only **16%** of eyes here
  and calls 3× as many closed as the geometry does; VSDLM calls **52%** of mouths
  open against the geometry's **18%**, and the shipped pipeline consequently raises
  YAWN on clips in which nobody yawns. `MIN_EYE_BOX_PX` and `MIN_MOUTH_BOX_PX`
  protect against the worst of it, and both floors were derived from the same
  footage they protect against.
* **Not latency-passing.** 10 of the 12 runnable profiles miss the 66 ms p95 gate at
  their measured median, and the two that pass find no faces. `default` was measured
  nine times today: median 77.79 ms, range 66.50–99.45, **0 passes**. Swapping to the
  smaller detector was tried and reverted — it is worth 5.4 ms of p50 and the gap is
  ~13 ms. See §7. This is a declared degrade: `KNOWN_GATE_MISSES` in
  `tests/test_config.py` fails the suite if a profile misses the gate without being
  listed, or is listed without missing it.
* **Regression-clean today, and now actually checked.** 9 of the 9 golden fixtures are
  inside their bands on the shipped detector, with no band re-recorded, and the
  comparison runs as nine pytest cases behind `DMS_REPLAY_TESTS=1` (§7). It was not
  part of `pytest` at all until today.
* **Not soaked.** The longest run in this repository's history is 300 frames —
  about 20 seconds. There is no 8 h result, no long-run RSS curve, no
  file-descriptor audit, and no evidence at all about what happens after an hour.
* **Not exercised on its own alert surface.** `PHONE` has never fired from real
  footage because no clip contains a phone; the mask gate has never seen a masked
  face; `ENGINE_FAIL` has no behavioural test anywhere in `tests/`. The golden
  fixtures pin 4 of the 16 `AlertType` members, and neither `MICROSLEEP` nor
  `PERCLOS_HIGH` is among them — those two are pinned instead by a synthetic
  scripted-signal golden (`tests/replay/golden/synthetic/scripted_drive.json`),
  which proves the state machine turns a closed-eye *signal* into a `MICROSLEEP` at
  the right instant and proves nothing about whether perception produces that signal
  from a real closed eye. The full ledger is
  `tests/replay/golden/synthetic/alert_coverage.json`.
* **Not certified.** UNECE R171 / Euro NCAP are explicit non-goals. No hazard
  analysis, no ISO 26262 work, no redundancy analysis.
* **Not installed.** `deploy/dms.service` has never been loaded: `sudo` on this
  board needs a password. The unit, logrotate policy, tmpfiles config and staged
  installer exist and are unit-tested; none has ever been executed by systemd.
* **Not measured on a quiet board.** Every latency number here was taken while
  other agents were working on the same six cores. `nxserver.bin` and `polkitd`
  were each observed spinning on a core for long stretches. Neither can be stopped
  without root.

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
