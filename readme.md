# DMS — Driver Monitoring System for Jetson Xavier NX

A single-process, single-threaded-inference driver monitor. Native TensorRT FP16,
GStreamer NVMM capture, no DeepStream, no PyTorch on the board. Everything runs on
one Xavier NX at 20 W / 6 cores.

**Status: integrated and running end to end, not vehicle-validated.** It has never
seen a camera or a real cabin. Read [`docs/STATUS.md`](docs/STATUS.md) before
believing any number here, and [`docs/production-readiness.md`](docs/production-readiness.md)
for the gap list.

Everything measured in this file was re-taken on **2026-09-13 between 16:45 and 17:10
KST**, on this board, with the GPU mutex held and with other agents' work running
alongside (loadavg 2.6–4.0). Where a claim could not be reproduced it has been
corrected or deleted. **Two things a reader should know before the rest:** the
p95 latency gate is failed by most shipped profiles today and the run-to-run spread on
one profile exceeds its own margin (§7), and 3 of the 9 golden replay fixtures are
currently red because the shipped detector changed after they were recorded (§7).

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

Every one of those five commands was executed as written on 2026-09-13 while this
file was revised. What they print today:

| command | result |
| --- | --- |
| `pytest -q` | **763 passed, 7 skipped in 84.04 s**, 0 failures |
| `--config configs/testsrc.yaml --max-frames 30` | `PERF_GATE PASS e2e_p95=36.59 ms … fps=30.25` |
| `--config configs/default.yaml --max-frames 300` | `PERF_GATE PASS e2e_p95=65.28 ms … fps=21.34` on one run and `FAIL e2e_p95=72.69 ms` on the next — see §7 |
| `run_golden.py` | 6 clips `ok`, **3 clips with 7 regressions** — see §7 |

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
  2d106det_192, hpe_mobilenetv2_224, mwc_n_48, ocec_s_24x40_b2, scrfd_500m_512,
  sface_112, sgc_is_s_48, vsdlm_s_30x48, yolox_nano_416
```

**Thirteen `.engine` files are on disk. Nine deserialise in a run. Four never do.**

`state` below is deliberately not the word "live", which reads as "working". It is
one of **real** (produces useful output, with evidence), **weak** (loads, runs, and
its output is measurably poor or unvalidated), **fallback** or **disabled**.

| Capability | Engine | Licence | State | Evidence |
| --- | --- | --- | --- | --- |
| Face detection | `scrfd_500m_512_fp16_gpu` | InsightFace research terms | **real** | 452 driver-face hits over 300 frames of `day_driver`, 1207 over `minivan`, today |
| Face detection (alt) | `scrfd_500m_640_fp16_gpu` | InsightFace research terms | **not loaded** | every profile now selects the 512 engine; 640 is the alternate |
| Landmarks (106) | `2d106det_192_fp16_gpu` | InsightFace research terms | **real** | drives EAR/MAR on the re-derived topology; open eyes read 0.21–0.40, closed below 0.15 |
| Head pose | `hpe_mobilenetv2_224_fp16_gpu` | MIT (6DRepNet-style) | **real, uncalibrated** | loads and runs every frame; solvePnP fallback is gated on a reprojection residual. No `forward_zero` exists, so every absolute angle is a relative signal |
| Head pose (fallback) | `hpe_mobilenetv3_small_224_fp16_gpu` | MIT | **not loaded** | fallback only; the primary is present |
| Eye open/closed | `ocec_s_24x40_b2_fp16_gpu` (batch 2) | Apache-2.0 | **weak** | measured through this wrapper over 6350 eyes on the replay corpus: only **16% reach the model path** (84% are below the 20 px floor and use the EAR ratio); on the model path corr(prob_open, eye ratio) **+0.29** while OCEC calls 3× as many eyes closed as the geometry does. Below the floor the correlation is **−0.00** over all 6350 eyes. `MIN_EYE_BOX_PX` exists to refuse that |
| Eye open/closed (alt) | `ocec_s_24x40_fp16_gpu` (batch 1) | Apache-2.0 | **not loaded** | the batch-2 engine is selected |
| Mouth open/closed | `vsdlm_s_30x48_fp16_gpu` | Apache-2.0 | **weak, miscalibrated** | measured over 3175 mouths: on the model path VSDLM calls **52.4%** of mouths open where the inner-lip ratio calls **18.3%**, corr **+0.14**. Corroborated by my own golden run today, which raised YAWN on `red_car_crash` (×2), `guy_rear_ended` (×2) and `rear_end_whiplash` (×4) — dashcam clips in which nobody yawns |
| Sunglasses gate | `sgc_is_s_48_fp16_gpu` | Apache-2.0 | **real on 2 of 4 positives** | re-measured by me today at the **shipped** threshold 0.50, every 15th frame, largest face, each clip's own config: `bmw_rear_ended` 1.00, `family_vacation_crash` 0.75, `getting_rear_ended` 0.28, `minivan` **0.14** of refreshes declare sunglasses; **all five labelled negatives 0.00** |
| Mask gate | `mwc_n_48_fp16_gpu` | Apache-2.0 | **weak, positive class never exercised** | no clip in this repository contains a masked face, so `prob_masked` has never seen a true positive. It is not clean on negatives either: my run today fires on **11%** of `bmw_rear_ended` refreshes (a bearded driver) and 0.00 everywhere else |
| Phone / cabin objects | `yolox_nano_416_fp16_gpu` | **Apache-2.0** | **loaded, never exercised** | runs every 20th frame; **no clip in the corpus contains a phone**, so no `PHONE` alert has ever been raised from real footage. The class-to-alert path is unit-tested only |
| Driver identity | `sface_112_fp16_gpu` | Apache-2.0 | **real, and noisy** | it does re-identify: 20 `driver_changed` edges over 300 frames of `day_driver` today. On a multi-face dashcam montage that is arguably too many — it is what put 3 of the 9 golden clips into regression (§7) |
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

All runs below: 300 frames, GPU mutex held, overlay off, GPU governor left alone,
board shared with other agents (`/proc/loadavg` 2.6–4.0 throughout — recorded, because
it matters). The gate is `perf.gate_e2e_p95_ms: 66.0` and
`perf.gate_peak_rss_mb: 5000.0`, and the app now evaluates it itself at every health
refresh and prints one `PERF_GATE PASS|FAIL|UNDECIDED` line per run.

### The most important measurement is the spread, not the number

`configs/default.yaml`, the same 300 frames of `day_driver.mp4`, three times in the
same hour:

| run | e2e p50 | e2e p95 | FPS | verdict |
| --- | ---: | ---: | ---: | --- |
| sweep, first of eleven | — | 79.96 (rolling window; set `health.degraded`) | — | **FAIL** |
| standalone | 42.0 ms | **65.3 ms** | 21.3 | **PASS**, by 0.7 ms |
| standalone, immediately after, same session | 45.4 ms | **72.7 ms** | 20.3 | **FAIL** |

Same binary, same clip, same config, minutes apart. **The run-to-run spread on this
board is larger than the margin the gate is being judged by.** Any single number
quoted for this pipeline — including every number below — is a sample from that
distribution, and no PASS taken on a shared board should be read as a property of the
software. `CLOCKS_LOW` also fired during the sweep at a peak GPU clock of 204 MHz
against a 1109 MHz ceiling.

### Every shipped profile, one sweep

| profile | frames | e2e p50 | e2e p95 | FPS | peak RSS | driver-face hits | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `testsrc` | 300 | 33.1 | **34.8** | 30.0 | 1324 MiB | 0 | **PASS** |
| `systemd-lab` | 300 | 33.2 | **35.1** | 29.6 | 1339 MiB | 0 | **PASS** |
| `rear_end_accident` | 186† | 42.0 | **65.2** | 23.1 | 1368 MiB | 292 | **PASS** |
| `default` (`day_driver`) | 300 | 42.0 | 65.3 | 21.3 | 1366 MiB | 452 | PASS / FAIL — see above |
| `bmw_rear_ended` | 300 | 43.7 | **68.1** | 19.8 | 1385 MiB | 300 | **FAIL** |
| `rear_end_whiplash` | 300 | 42.2 | **68.7** | 21.8 | 1387 MiB | 540 | **FAIL** |
| `red_car_crash` | 300 | 48.6 | **71.3** | 18.9 | 1363 MiB | 300 | **FAIL** |
| `guy_rear_ended` | 300 | 46.3 | **76.3** | 18.9 | 1372 MiB | 300 | **FAIL** |
| `family_vacation_crash` | 300 | 50.8 | **78.5** | 18.1 | 1375 MiB | 565 | **FAIL** |
| `minivan` | 300 | 49.5 | **78.6** | 17.8 | 1370 MiB | 1207 | **FAIL** |
| `getting_rear_ended` | 300 | 46.3 | **79.7** | 18.5 | 1370 MiB | 589 | **FAIL** |
| `production` | — | — | — | — | — | — | **not runnable**: `source.type: csi` and there is no camera |

† the clip is shorter than 300 frames.

**Verdict: 3 PASS, 8 FAIL, 1 unrunnable, on a shared board.** The two profiles that
pass comfortably (`testsrc`, `systemd-lab`) render a synthetic pattern and find zero
faces, so they exercise capture, the engines and the loop but not the per-face
cascade. Of the profiles that do real work, only the shortest clip passed, and it
passed by 0.8 ms.

The FPS half of the contract is met everywhere: **17.8–30.0 FPS against a ≥ 15 FPS
target**. So is memory: **peak RSS 1324–1387 MiB against a 5000 MiB limit**, 27% of
budget, with no profile approaching it.

### What changed since the last revision of this file, and what it bought

Two of the three fixes this document used to list as unapplied handoffs **are now in
the shipped path**, and this section had not been updated for them:

* `dms.runtime.preprocess.Letterbox` is wired into `DmsApp._setup_perception` /
  `_perceive`, built from the SCRFD engine's own input shape. `dms/app.py:876` calls
  `self.letterbox.blob(bgr)`; `ScrfdDetector.preprocess` is now only a fallback.
* Every profile ships `engines/scrfd_500m_512_fp16_gpu.engine`, not the 640.

CUDA-event reuse in `dms/runtime/trt_engine.py` is still **not** applied.

The earlier revision of this section reported 58.6 ms p50 / 87.2 ms p95 / 16.2 FPS on
`configs/default.yaml` with the 640 detector and no `Letterbox`. Today the same
profile measures 42.0–45.4 ms p50 and 65.3–72.7 ms p95 at 20.3–21.3 FPS. That is a
real improvement of roughly **14 ms of p50** — but it is measured across two different
board conditions, so treat it as "the two changes are worth something in the
neighbourhood of 10–15 ms" rather than as a delta.

### Golden replay: 6 of 9 clips green, 3 red, and the cause is known

```
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python tests/replay/golden/run_golden.py"
```

Result today: **7 regressions across 3 clips**; `minivan`, `red_car_crash`,
`family_vacation_crash`, `guy_rear_ended`, `bmw_rear_ended` and `getting_rear_ended`
are `ok`.

```
REGRESSION day_driver:        driver_changed/enter fired 10 times, golden band [1, 3]
REGRESSION day_driver:        driver_changed/exit  fired 10 times, golden band [1, 3]
REGRESSION rear_end_accident: driver_changed/enter fired  3 times, golden band [4, 6]
REGRESSION rear_end_accident: driver_changed/exit  fired  3 times, golden band [4, 6]
REGRESSION rear_end_whiplash: NEW alert driver_changed/enter        (not in the golden)
REGRESSION rear_end_whiplash: NEW alert driver_changed/exit         (not in the golden)
REGRESSION rear_end_whiplash: NEW alert drowsiness_unreliable/exit  (not in the golden)
```

**This is not a mystery and it should not be re-recorded away.** The fixtures were
recorded at `2026-09-13T01:36–01:40Z` (= 10:36–10:40 KST). `dms/runtime/preprocess.py`
was modified at 14:07, `dms/app.py` at 14:30 and `configs/*.yaml` at 14:32 — that is
the `Letterbox` rewiring and the SCRFD 640 → 512 swap, both of which land *after* the
baseline was taken. A different detector at a different input size finds different
faces, which moves track ids, which is exactly what `driver_changed` counts. The
golden is doing its job: it caught an unreviewed perception change.

Re-recording the fixtures would launder that change into the baseline. Whoever owns
the 512 decision should read the diff and re-record deliberately, with a note.

### Where the time goes

Reproduce with:

```bash
DMS_LATENCY_BUDGET_OUT=/tmp/lb.json flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
```

The per-stage breakdown in [`docs/performance.md`](docs/performance.md) predates the
`Letterbox` and 512 changes and is being read with that caveat; the stage totals in
`latency_budget.json` from a run you make yourself are the current source of truth.
What has not changed is the shape of the problem: the networks are not the cost. The
whole model zoo is under 25 ms of pure GPU compute, and the pipeline spends its budget
on the Python data path, TensorRT host round-trip overhead, and a GPU governor that
was observed at 204 MHz against an 1109 MHz ceiling during these runs.

## 8. What this is not

Read this section as the counterweight to every number above.

* **Not vehicle-validated.** No camera has ever been attached. `/dev/video*` does
  not exist on this board; everything is replay. `configs/production.yaml` — the
  vehicle profile — cannot be run here at all, because its source is `csi`.
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
* **Not latency-passing.** 8 of the 11 runnable profiles fail the 66 ms p95 gate
  today, and the one profile measured three times moved from 65.3 ms (PASS) to
  79.96 ms (FAIL) within the hour. See §7.
* **Not regression-clean.** 3 of the 9 golden fixtures are red, for a known and
  documented reason (§7). They have deliberately not been re-recorded.
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
