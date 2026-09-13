# DMS production readiness — assessment and plan

| Field | Value |
| --- | --- |
| **Date** | 2026-09-13, revised 17:15 KST after an adversarial review and fix pass (first issued 2026-09-09) |
| **Board** | Jetson Xavier NX Dev Kit, JetPack 5.1.6, TensorRT 8.5.2.2, 6.7 GiB RAM, `MODE_20W_6CORE` |
| **Architecture** | [`production-dms-design.md`](production-dms-design.md) (rev 5) is the v1 contract; [`architecture.md`](architecture.md) describes what was actually built |
| **This document** | Gap analysis of the *running tree* against that contract |
| **Verdict** | **Not production-ready, but no longer a lab toy.** Phases 0, 1 and 2 are complete in code; Phase 3 (camera) is untouched because no camera exists; Phase 4 fails its latency gate on 8 of the 11 runnable profiles and has never been soaked for longer than 300 frames. |
| **How to read the numbers** | Every measurement in this revision was re-taken on this board between 16:45 and 17:10 KST on 2026-09-13, at `/proc/loadavg` 2.6-4.0, by running the commands in [`STATUS.md`](STATUS.md#reproduce-in-ten-minutes). **The run-to-run spread on the latency gate exceeds its margin** — `configs/default.yaml` measured 65.3 ms p95 (PASS) and 72.7 ms p95 (FAIL) on two consecutive runs in the same session. Treat every single figure as n=1. |

**Production-level** here means the design v1 freeze, **not** UNECE R171 / Euro
NCAP certification (explicit non-goal). Freeze gates from the design:

| Gate | Target | Measured 2026-09-13 17:00 | |
| --- | --- | --- | --- |
| e2e p95 at 1280×720, 20 W 6-core | ≤ 66 ms | **65.2-79.7 ms across 11 profiles; 3 PASS, 8 FAIL** | ✗ |
| — and repeatably | (implied) | **no**: one profile spanned 65.3 → 72.7 ms on consecutive runs | ✗ |
| Processed frame rate | ≥ 15 FPS | 17.8-30.0 FPS, every profile | ✓ |
| Peak RSS | < 5.0 GB | 1.32-1.39 GB, every profile | ✓ |
| Free RAM | ≥ 1.5 GB | 3.38 GB available at the start of the sweep | ✓ |
| systemd notify + watchdog, health on localhost | present | present, watchdog gated on frames | ✓ |
| The freeze gate is *evaluated by the running unit* | required | present — every health refresh, 5 `dms_perf_gate_*` gauges, one `PERF_GATE` line per run. It previously was not evaluated at all outside `DMS_LATENCY_BUDGET_OUT` | ✓ |
| Every engine the run loads is sha256-verified | required | present — 9 of 9, fatal before READY. It was 2 of 9 | ✓ |
| Alerts with hysteresis, local JSONL, optional clip quota | present | present | ✓ |
| Replay works with no camera | required | the only mode that has ever run | ✓ |
| Golden replay green | required | **6 of 9 clips; 3 red** on a known cause (§2.4 T1) | ✗ |
| 8 h soak without watchdog fire | required | **never run** — longest run in the project's history is 300 frames | ✗ |

---

## 1. What is actually in the tree

### Works on this NX (measured, this revision)

| Piece | Evidence |
| --- | --- |
| Supervised GStreamer capture with a reconnect ladder | `dms/runtime/supervisor.py`; demonstrated against an absent CSI camera |
| **Nine** FP16 TensorRT engines live in the frame loop, every one sha256-verified before READY | the startup line `engine provenance: 9 engines verified in 0.10s`, and the nine `loaded .../engines/...` lines, from a real run today. Thirteen `.engine` files exist on disk; four are never deserialised — the Gaze360 engine (licence-gated off), `hpe_mobilenetv3_small` (fallback), `ocec_s_24x40` batch-1 and `scrfd_500m_640` (both superseded alternates). A previous revision of this row said "eleven ... all in the frame loop"; that was wrong |
| Media-time state machine (buffer PTS on replay) | `dms/app.py`; golden replay is reproducible run to run |
| Fusion with Schmitt hysteresis, leaky dwell, time-weighted PERCLOS | `dms/state/fusion.py`, `dms/state/driver_state.py` |
| Occlusion gating of the eye and mouth channels | sunglasses → `DROWSINESS_UNRELIABLE`, not "eyes open" |
| Durable `events.jsonl` with boot id, seq, fsync policy, ENOSPC handling | `dms/io/events.py` |
| `/healthz` `/livez` `/readyz` `/metrics` with real status codes | 503 verified during a camera failure |
| Golden replay over nine clips, plus a synthetic scripted-signal golden for the drowsiness path | `tests/replay/golden/` and `tests/replay/golden/synthetic/`. The runner is deterministic; **the result today is 6 clips `ok` and 3 red**, on a known cause — see §2.4 T1. The mechanism works; the baseline is stale against the shipped detector |
| Strict config schema, ~120 declared keys, cross-section refusals | `dms/config/schema.py`; a typo is now an error |
| Latency budget with an accounting check | 1.1 ms unaccounted of a 60.9 ms mean frame |

### What the reference replay shows

`configs/default.yaml`, 300 frames of `day_driver.mp4`, all nine engines, overlay
off, board at loadavg 2.6-4.0 with a GPU clock observed peaking at 204 MHz:

```
run 1 (in an 11-profile sweep):  e2e p95 79.96 ms  ->  PERF_GATE FAIL, health.degraded set
run 2 (standalone):              e2e p50 42.0 ms  p95 65.3 ms  21.3 FPS  peak RSS 1366 MiB  -> PASS
run 3 (standalone, immediately): e2e p50 45.4 ms  p95 72.7 ms  20.3 FPS  peak RSS 1366 MiB  -> FAIL
```

Same binary, same clip, same config, three verdicts. **The measurement is not
repeatable to within the gate's margin, and that is the finding.** Two of the three
optimisations this document's Phase 4 asked for are now merged (`Letterbox` wired into
`dms/app.py`, SCRFD-512 in every profile), which moved p50 from 58.6 ms to 42-45 ms;
the third (per-engine CUDA event reuse) is not.

The networks are still not the cost. The whole model zoo is under 25 ms of pure GPU
compute; the budget goes on the Python data path, TensorRT host round-trip overhead,
and a governor that never approaches its 1109 MHz ceiling.

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
| G4 | MANIFEST not enforced at load | High | **CLOSED, and the reverse defect fixed too** — `check_runtime` ran unconditionally but verified only **2 of the 9** engines the process loads, directly under a comment claiming the check was "the only thing standing between a swapped `.engine` and a safety decision"; the other seven got `os.path.isfile` and nothing else. `engines_to_verify(cfg)` now mirrors the registry's own resolution and hashes every engine the run will deserialise (measured: 9 engines in 0.10 s), and a sha256 or size mismatch is fatal **before READY regardless of `require_engines`** — measured, a swapped plan reached READY and then killed the process with CUDA error 700. |
| G5 | Leftover `models/` prototype tree | Med | **CLOSED** (Phase 0). |
| G6 | `production.yaml` EAR threshold wrong | High | **CLOSED, and the reverse defect fixed too** — the landmark topology was wrong, so both the old 0.21 *and* the 0.50/0.75 "fix" were meaningless. Every profile is now on the corrected scale and the schema rejects the old one by name. |
| G7 | venv layout for the service | Med | **CLOSED** — the installer stages a release with its own venv and an ABI check. Unrun. |
| G8 | Health unix socket | Low | **OPEN**, deliberately. TCP loopback with an optional bearer token is enough for v1. |
| G9 | 1 Hz tegrastats thread | Low | **OPEN**, deliberately. Sysfs zones + VmRSS + VmHWM, sampled with the health refresh. |
| G10 | *(new)* `camera.fail_fatal: false` meant "report READY for ever with no camera and no retry" | **Blocker** | **CLOSED** — see §3. |
| G11 | *(new)* **The freeze gate was never evaluated.** `perf.gate_e2e_p95_ms` was computed only inside `_write_budget()`, which returns immediately unless `DMS_LATENCY_BUDGET_OUT` is set. In normal operation nothing checked it and an over-budget unit reported itself healthy. | **Blocker** | **CLOSED** — `DmsApp._evaluate_gate()` runs at every health refresh (2 Hz), publishes `dms_perf_gate_{pass,e2e_p95_ms,e2e_p95_limit_ms,peak_rss_mb,e2e_samples}`, writes a note into `/healthz`, and sets `health.degraded` after 20 consecutive *decided* misses. Every run ends with one `PERF_GATE PASS\|FAIL\|UNDECIDED` line. A window below `MIN_SAMPLES` is `undecided` and never counts as a pass. **The gate now fails honestly rather than not being asked.** |
| G12 | *(new)* `capture.letterbox` / `capture.letterbox_pad_value` were declared in the schema and set in all 12 profiles, and read by no code at all — they said `[640, 640]` whatever engine shipped | Low | **CLOSED** — the app logs `LETTERBOX_MISMATCH` when the declaration disagrees with the engine's input shape (the engine wins), and all 12 profiles now declare `[512, 512]`. |

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
| A12 | *(new)* The eye classifier inverts below ~20 px of eye width | High | **PARTIAL, and re-measured worse than "inverts"** — over 6350 eyes on this corpus the correlation below the floor is **−0.00**, i.e. uninformative rather than inverted, while 38% are called closed. Only **16%** of eyes on this footage reach the model path at all. Gated by `models.eyestate.min_eye_box_px`, a floor derived from the same footage it protects against. Needs cabin data. |
| A13 | *(new)* **VSDLM calls the majority of mouths open on this corpus, so YAWN and FATIGUE are not trustworthy.** Measured through the shipped wrapper over 3175 mouths: 52.4% called open against the inner-lip ratio's 18.3%, corr +0.14. | High | **OPEN** — the measurement is recorded in `dms/infer/mouthstate.py` and corroborated by the golden run, which raises YAWN on three dashcam clips in which nobody yawns. Not retuned: there is no labelled yawn anywhere in this repository, so any new threshold would be invented rather than measured. Options are (a) require MAR agreement before `MouthState.open`, (b) raise `models.mouthstate.prob_open_threshold`, (c) drive YAWN from the geometry until a labelled clip exists. |
| A14 | *(new)* **The mask gate has never been exercised against a positive, and false-positives on negatives.** | Med | **OPEN** — no clip in `tests/replay` contains a masked face. On negatives the 5-refresh vote declares "masked" on 11% of `bmw_rear_ended` refreshes (a bearded driver) and 0.00 elsewhere, measured today. It must not be described as validated. One labelled still or short clip with a masked driver would close the positive half. |
| A15 | *(new)* **The sunglasses gate finds two of four positives reliably, one marginally, one poorly.** | Med | **PARTIAL** — measured today at the shipped 0.50 threshold: `bmw_rear_ended` 1.00, `family_vacation_crash` 0.75, `getting_rear_ended` 0.28, `minivan` 0.14 of refreshes; **all five labelled negatives 0.00**. The crop convention was the root defect (the box was being squared and padded rather than clipped) and fixing it roughly tripled the response on the hard positive. Every profile pins `sunglasses_threshold: 0.50`, overriding the module's measured default of 0.40; at 0.40 the positives improve and the negatives stay at 0.00, so 0.40 is the recommended value. |
| A16 | *(new)* **`PHONE` has never fired from real footage.** | Med | **OPEN** — the YOLOX-Nano object head loads and runs every 20th frame, but no clip in the corpus contains a phone, so the detector-to-alert path is exercised by unit tests only. |

### 2.3 Performance and reliability

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| P1 | e2e not budgeted; p95 misses the gate | Blocker | **PARTIAL, and two of the three fixes are now merged.** `Letterbox` is wired into `dms/app.py` (built from the engine's own input shape; `dms/infer/scrfd.py` was not touched and keeps its allocating `preprocess` as a fallback) and SCRFD-512 ships in all 12 profiles. That moved `configs/default.yaml` from 58.6 ms p50 to 42-45 ms. It still fails: **3 PASS / 8 FAIL across the 11 runnable profiles**, and the gate is not repeatable to within its own margin. CUDA event reuse in `dms/runtime/trt_engine.py` (~6 ms) is **not** applied. The honest next step is not another micro-optimisation but locking the GPU clock and re-measuring on an idle board — which needs root. |
| P2 | No 8 h soak | Blocker | **OPEN, and untouched.** The longest run in this project's history is 300 frames — about 20 seconds. There is no long-run RSS curve, no FD audit, and no evidence of any kind about behaviour after the first minute. This is the largest single hole in the readiness argument and no amount of further code changes it. |
| P9 | *(new)* **Latency measurement is not repeatable on this board.** `configs/default.yaml` measured 65.3 ms p95 (PASS) and 72.7 ms p95 (FAIL) on consecutive runs in one session, and 79.96 ms in a third. `CLOCKS_LOW` fired at a peak GPU clock of 204 MHz against an 1109 MHz ceiling. | High | **OPEN.** Until the clocks can be locked (root) and the board can be quiesced (other agents share it), no gate verdict taken here is a property of the software. Every figure in this document is n=1. |
| P3 | File vs live drop policy | Med | **CLOSED** as designed; live rate still unmeasured. |
| P4 | Idle GPU clock logs CLOCKS_LOW | Low | **CLOSED** — logged once, only when perception is loaded, and the log says the number is not comparable with a full-clock run. |
| P5 | numpy 1.23 vs distro OpenCV | Med | **OPEN** — the installer has the ABI check; the lab venv is still mixed. |
| P6 | NVDEC backpressure warnings | Low | **OPEN**, harmless. |
| P7 | *(new)* The state machine ran on wall time on replay | **Blocker** | **CLOSED** — media time from buffer PTS. Every archived `events.jsonl` from before this change is timing-distorted and should be discarded. |
| P8 | *(new)* Unbounded `infer_ms` list, sorted twice a second | Med | **CLOSED** — fixed-memory ring buffers. |

### 2.4 Tests and ops

| ID | Gap | Sev | Status |
| --- | --- | --- | --- |
| T1 | No golden alert counts | High | **CLOSED as a mechanism, RED as a result.** `tests/replay/golden/` has nine clips, banded counts, a `--record` mode and a regression runner, plus a new synthetic drowsiness golden (below). Today **6 of 9 clips are `ok` and 3 report 7 regressions** — `day_driver`, `rear_end_accident` and `rear_end_whiplash`, all on `driver_changed` edge counts. The cause is known: the fixtures were recorded at 10:36-10:40 KST and `dms/runtime/preprocess.py` (14:07), `dms/app.py` (14:30) and `configs/*.yaml` (14:32) all changed afterwards — the `Letterbox` rewiring and the SCRFD 640 → 512 swap. A different detector finds different faces, which moves track ids. **The fixtures have deliberately not been re-recorded**: doing so would launder an unreviewed perception change into the baseline, which is the exact failure mode the golden exists to prevent. Whoever owns the 512 decision should read the diff and re-record with a note. |
| T7 | *(new)* **The nine clip goldens pin only 4 of 16 `AlertType` members**, and neither `MICROSLEEP` nor `PERCLOS_HIGH` — the two alerts the product exists to raise. | High | **PARTIAL** — `tests/replay/golden/synthetic/scripted_drive.json` replays a scripted 200 s / 2400-frame `DriverSignals` stream through the real `DriverMonitor` and pins exact per-type edge counts and first-enter timings for eleven types including `microsleep` and `perclos_high`. That proves the **state machine**, not perception: nothing in this repository proves the perception layer produces a closed-eye signal from a real closed eye. `tests/replay/golden/synthetic/alert_coverage.json` is a per-type ledger recording exactly what is pinned where, and for the five types pinned nowhere (`camera_fail`, `disk_full`, `engine_fail`, `pipeline_starved`, `thermal_throttle`) exactly what does and does not guard them — including the honest admission that **`engine_fail` has no behavioural test at all**. |
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

| Work | Expected | Owner file | Status 2026-09-13 |
| --- | --- | --- | --- |
| `Letterbox` on the hot path | −5 ms p50 | `dms/app.py` (not `scrfd.py` as planned) | **DONE** |
| SCRFD-512 **after** a cabin-recall check | −10 ms p50 | `configs/*.yaml` | **DONE — but the recall check was skipped.** It shipped on latency evidence alone and three golden fixtures went red |
| One CUDA event per engine instead of create/destroy per inference | −6 ms/frame | `dms/runtime/trt_engine.py` | **OPEN** |
| **Lock the GPU clock and re-measure on a quiesced board** | removes the dominant source of variance | operator / systemd, needs root | **OPEN, and now the top item.** The gate's verdict currently depends on which minute you run it |
| 8 h looped replay | RSS stable, no watchdog fire, no FD leak | — | **OPEN, never attempted** |
| Thermal: drop the object net as well as the overlay above 80 °C | rehearse with tegrastats | `dms/app.py` | **OPEN** |
| Re-record the golden fixtures against SCRFD-512, deliberately, with a reviewed diff | restores the regression signal | `tests/replay/golden/*.json` | **OPEN** |

**Exit:** `latency_budget.json` from this NX with `gate.pass = true` **on every
runnable profile, reproduced twice**, plus a soak log, plus a green golden run. The
"reproduced twice" clause is new and is there because a single PASS has already been
observed to be followed immediately by a FAIL.

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
  Three of them are red right now, for a reason that is written down in §2.4 T1. That
  is the mechanism working.
- Do not quote a single latency run as the gate result. Run it twice. If the two
  verdicts disagree, that is the result.
- Do not describe a model as "live". Say **real**, **weak**, **fallback** or
  **disabled**, and put the measurement next to it. Two of the drowsiness models on
  this board load and run and produce output that is measurably poor; "live" hides
  exactly that.
- Do not claim an engine count from `ls engines/`. Thirteen files exist and nine
  deserialise. Parse the `loaded .../engines/...` lines of a real run.
- Do not treat the synthetic drowsiness golden as evidence that microsleep detection
  works. It pins the state machine's response to a scripted *signal*. Nothing in this
  repository connects a real closed eye to that signal.
