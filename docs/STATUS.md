# Status — 2026-09-13 (revised, evening KST)

Six workstreams (engines, models, runtime, ops, alerts, perf) landed, an integration
pass wired them into one running system, four adversarial reviewers then re-verified
the result on this board and proved a list of defects, and a fix pass closed most of
them. This document is the honest state of the system after that, on this Xavier NX,
today.

**Everything numeric below was measured on this board by running the commands written
at the bottom of this file.** Nothing is carried over from an earlier revision, and
nothing is taken from a workstream's self-report. The board was shared with other
agents throughout (`/proc/loadavg` 2.4–4.5); that is recorded because it changes the
answer.

Two rounds of measurement are quoted. The perception, model-status and alert figures
were taken between 16:45 and 17:10 KST. **The latency figures and the detector choice
were re-measured in the evening, repeated (n ≥ 3 per profile, n = 9 on `default`,
plus a paired A/B for the detector) rather than once**, after the SCRFD-512 detector
adopted that afternoon was reverted to SCRFD-640. Where an evening measurement
replaced an afternoon one, the old number is shown and marked, because the difference
between "measured once" and "measured nine times" is the whole point of this
revision.

## Verdict

**A complete, integrated driver-monitoring pipeline that runs end to end on replay
with nine TensorRT FP16 engines live in the frame loop — and no vehicle evidence of
any kind.** It is a credible v1 codebase. It is not a validated product, it does not
currently meet its own latency gate on any profile that does real work, and two of
its three drowsiness signals are measurably weak on the only footage available.

A previous revision of this document claimed "eleven real neural models and no
stubs". That was wrong twice over. **Nine engines deserialise in a run**; thirteen
`.engine` files exist on disk. The Gaze360 engine is licence-gated off and is never
deserialised, `hpe_mobilenetv3_small` is a fallback that loads only if the primary is
absent, and `ocec_s_24x40` (batch 1) and `scrfd_500m_512` are alternates that no
shipped profile selects (`scrfd_500m_512` is selected only by `configs/lowlatency.yaml`,
which is a documented low-latency variant, not a default). "No stubs" was also wrong: the eye and mouth channels carry
documented EAR/MAR geometry fallbacks below their pixel floors, which is a good
design and is not the absence of a fallback.

| Working, measured | Missing or unproven |
| --- | --- |
| **Nine** TensorRT FP16 engines, every one sha256-verified at startup, all live in the frame loop | No camera has ever been attached (`/dev/video*` absent); `configs/production.yaml` cannot run here |
| Media-time state machine, hysteresis, occlusion gating, honesty alerts | No calibration → every gaze alert is suppressed by design |
| Supervised capture: reconnect ladder, `CAMERA_FAIL`, `/healthz` 503, watchdog withheld | Never exercised against a real camera unplug |
| Durable `events.jsonl` (boot id, seq, fsync policy, ENOSPC handling) | Clip ring's GStreamer encode path still never executed |
| `/healthz` `/livez` `/readyz` `/metrics`, ~40 Prometheus series + 5 new `dms_perf_gate_*` gauges | Health endpoint never scraped by anything but curl |
| Golden replay runs over all nine clips, **is now nine pytest cases** behind `DMS_REPLAY_TESTS=1`, and 9 of 9 are green on the shipped detector with no band re-recorded | Until today the comparison was not in `pytest` at all — a green suite was compatible with a 15.8% recall regression; the nine clip goldens pin 4 of 16 alert types and neither `MICROSLEEP` nor `PERCLOS_HIGH`, which a synthetic scripted-signal golden pins instead |
| Peak RSS 1331–1390 MiB against a 5000 MiB gate | **e2e p95 misses the 66 ms gate at the median on 10 of 12 runnable profiles — on both detectors.** Declared, not hidden: `KNOWN_GATE_MISSES` in `tests/test_config.py` |
| Strict config schema; the old broken EAR scale is rejected by name | Thresholds are starting points, swept on no cabin footage |
| `deploy/` unit, logrotate, tmpfiles, staged installer | Never installed: `sudo` needs a password on this board |
| The freeze gate is now *evaluated* every health refresh and published | Nothing has ever run for longer than 300 frames — no soak, ever |
| Face detection, landmarks, head pose and identity all produce real output | The eye and mouth classifiers are measurably weak (§"Model status") |

## Model status — what is real, what is weak, what is off

Classification, with the evidence for each. `state` is deliberately not the word
"live", which reads as "working".

| Capability | Engine | State | Evidence |
| --- | --- | --- | --- |
| Face detection | `scrfd_500m_640` | **real** | 583 face hits over 300 frames of `day_driver`, 1201 over `minivan`, identical on nine consecutive runs. The `scrfd_500m_512` alternate scores 452 and 1207 on the same clips and costs 15.8% of the driver-ROI hits on `day_driver`; it was shipped for part of today and reverted |
| Landmarks | `2d106det_192` | **real** | drives EAR/MAR on the re-derived 106-point topology; open eyes read 0.21–0.40, closed below 0.15 |
| Head pose | `hpe_mobilenetv2_224` | **real, uncalibrated** | runs every frame; no `forward_zero` exists, so every absolute angle is relative |
| Eye open/closed | `ocec_s_24x40_b2` | **weak** | over 6350 eyes on this corpus only 16% reach the model path; on it, corr with the eye ratio is +0.29 while OCEC calls 3× as many eyes closed. Across all 6350 the correlation is −0.00 |
| Mouth open/closed | `vsdlm_s_30x48` | **weak, miscalibrated** | 52.4% of mouths called open vs the inner-lip ratio's 18.3%, corr +0.14. Corroborated by today's golden run raising YAWN ×2 on `red_car_crash`, ×2 on `guy_rear_ended`, ×4 on `rear_end_whiplash` |
| Sunglasses gate | `sgc_is_s_48` | **real on 2 of 4 positives** | re-measured today at the shipped 0.50 threshold: `bmw` 1.00, `family_vacation` 0.75, `getting_rear_ended` 0.28, `minivan` 0.14; all five negatives 0.00 |
| Mask gate | `mwc_n_48` | **weak, positive class never exercised** | no clip contains a masked face; fires on 11% of `bmw_rear_ended` refreshes (bearded driver), 0.00 elsewhere |
| Objects / phone | `yolox_nano_416` | **loaded, never exercised** | no clip contains a phone; no `PHONE` alert has ever fired from real footage |
| Driver identity | `sface_112` | **real, and noisy** | 20 `driver_changed` edges over 300 frames of `day_driver`; it is what put 3 golden clips into regression |
| Eye gaze | `mobilenetv2_gaze_448` | **disabled** | licence-gated (Gaze360, non-commercial); absent from the 9-engine provenance line |

The eye and mouth figures were measured by the model workstream on 2026-09-13 and are
recorded in the `dms/infer/eyestate.py` and `dms/infer/mouthstate.py` docstrings; they
were not re-run for this document. The sunglasses and mask figures **were** re-run for
this document, and they disagree with `dms/infer/occlusion.py` on the two hard
positives (that docstring records `minivan` 0.29 and `getting_rear_ended` 0.41 at
threshold 0.50 where this measurement gives 0.14 and 0.28). Two plausible causes:
`occlusion.py` was still being edited while the measurement ran, and the probe drives
one persistent track id whereas the app's tracker restarts ids across a montage cut,
which resets the 5-refresh vote window. Treat both sets as order-of-magnitude.

## Changelog — what changed in this redevelopment

*(This repository has no `CHANGELOG.md`. This section is the changelog entry for the
redevelopment, and it is the place to add the next one.)*

Beyond the integration work, this pass was driven by four adversarial reviews that
re-verified every claim on this board and proved a list of defects with numerical
repro scripts. In rough order of consequence:

1. **The freeze gate is now actually evaluated.** It was computed only inside
   `_write_budget()`, which returns immediately unless `DMS_LATENCY_BUDGET_OUT` is
   set — so in normal operation nothing checked `perf.gate_e2e_p95_ms` and an
   over-budget unit reported itself healthy. `DmsApp._evaluate_gate()` now runs at
   every health refresh (2 Hz), publishes five `dms_perf_gate_*` gauges, writes a
   note into `/healthz`, and sets `health.degraded` once a *decided* miss has
   persisted 20 refreshes. Every run ends with one `PERF_GATE PASS|FAIL|UNDECIDED`
   line. A window with too few samples is `undecided` and never counts as a pass.
2. **Every engine is verified, not two of them.** `dms/app.py` carried a comment
   saying the sha256 check was unconditional and "the only thing standing between a
   swapped `.engine` and a safety decision", directly above a line that verified 2 of
   the 9 engines the process loads. `engines_to_verify(cfg)` now mirrors the
   registry's own resolution and hashes everything the run will deserialise; a
   mismatch is fatal *before* READY regardless of `require_engines`, because a swapped
   plan measurably reached READY and then killed the process with CUDA error 700.
3. **`Letterbox` is wired into the frame loop; the SCRFD-512 swap was reverted.**
   `dms.runtime.preprocess.Letterbox` was a 349-line exported module no production
   path used — only `scripts/bench.py` imported it, so the benchmark measured a
   preprocessing stage the application did not have. It is now built from the SCRFD
   engine's own input shape in `_setup_perception` and called from `_perceive`.
   All 12 profiles were also repointed at `scrfd_500m_512_fp16_gpu.engine` to buy the
   66 ms gate; that half has been **reverted**. Re-measured in the evening as a paired
   A/B: the 512 engine is worth 5.4 ms of p50 and 7.4 ms of p95, the gap to the gate is
   ~13 ms, **0 of 6 runs passed on either engine**, and the 512 costs 15.8% of the
   driver-ROI face hits on `day_driver` and put 3 goldens into regression. Every
   shipped profile is back on `scrfd_500m_640_fp16_gpu.engine`; the 512 point survives
   as `configs/lowlatency.yaml`. See `docs/performance.md` §7.
4. **`FusionConfig`'s dataclass defaults were still on the old broken EAR scale**
   (0.50 / 0.75), which `dms/config/schema.py` rejects by name — so any
   `SignalFuser()` built without an `AppConfig` read every open eye as closed. It now
   imports the geometry constants directly. The fusion layer also read half its
   thresholds live and froze the other half at construction, so one frame could be
   judged against two threshold generations; there is now one mechanism
   (`_apply_cfg`, measured at 11.8 µs/call against 133 µs for the whole of `update()`).
5. **The golden replay grew a synthetic drowsiness golden.** The nine clip fixtures
   pin only 4 of 16 `AlertType` members, and `MICROSLEEP` and `PERCLOS_HIGH` — the two
   alerts the product exists to raise — had zero end-to-end coverage.
   `tests/replay/golden/synthetic/scripted_drive.json` replays a scripted 200 s
   `DriverSignals` stream through the real `DriverMonitor` and pins exact per-type edge
   counts. It proves the state machine, not perception, and
   `tests/replay/golden/synthetic/alert_coverage.json` says so per type, including the
   five types nothing pins at all.
6. **Operator ergonomics.** `CLOCKS_LOW` no longer fires on the first frame of every
   run (it now needs 60 frames and 20 clock samples). A missing engine file is logged
   as `ENGINE_FAIL`, not `ENGINE_VERSION_MISMATCH ENGINE_FAIL`. A refused health bind
   is reported as `health=disabled(bind refused)` in the READY line instead of
   advertising an endpoint that is not there. `capture.letterbox` was declared in the
   schema, set in all 12 profiles and read by no code at all; it is now a checked
   declaration that logs `LETTERBOX_MISMATCH` when it disagrees with the engine.
7. **Documentation claims that could not be reproduced were deleted or corrected.**
   The specific ones, recorded so they are not reintroduced: "eleven real neural
   models and no stubs" and "11 TensorRT FP16 engines … all live in the frame loop"
   (nine deserialise); `docs/performance.md`'s Verdict row "`dms.app` as shipped
   today, 2 engines, 8.6 FPS, FAIL by 1.8x" (a two-engine app that no longer existed,
   pessimistic by ~2x, and contradicting `readme.md` on the same day); the same
   table's "optimised512 … 28.4 FPS … pass, 24 ms margin" row (a `scripts/bench.py`
   *partial-pipeline* microbenchmark — detector plus landmarks only — presented beside
   a whole-system gate verdict with no label); the model table's "live" against the
   sunglasses and mask gates (one finds two of four positives well, the other has
   never seen a positive); `dms/app.py`'s comment that the hash check was "the only
   thing standing between a swapped `.engine` and a safety decision" above a line
   verifying 2 of 9; and `dms/state/driver_state.py`'s claim that `dms/app.py` still
   passes raw floats (both of its call sites pass `signals=`).

8. **The earlier integration fixes stand**: media-time state machine on buffer PTS,
   a capture stall producing `FACE_LOST` rather than silence, `configs/production.yaml`
   as a real vehicle config, a strict schema, and the calibration merge applying to
   every profile.

## Measured, this board, 2026-09-13 (latency re-measured in the evening)

### The distribution matters more than any single number

`configs/default.yaml`, the same 300 frames of `day_driver.mp4`, **nine runs** this
evening, one process at a time under the GPU mutex:

```
66.50  68.75  72.21  73.22  77.79  80.89  81.30  86.93  99.45   (e2e p95, ms)
                            ^^^^^ median 77.79                  gate 66.00
```

**Range 32.95 ms, 0 of 9 inside the gate.** The revision of this section written at
17:10 quoted three runs (79.96 / 65.3 / 72.7) and reported the 65.3 as a PASS "by
0.7 ms". Nine runs later that PASS has not reproduced once, and it never was one: a
0.7 ms margin inside a 33 ms range is a coin toss, not a result. `CLOCKS_LOW` fires
during sweeps at a peak GPU clock of **204 MHz** against an 1109 MHz ceiling, which is
the mechanism. Every latency figure below is a median of n runs with its range, and
the same numbers are pinned in `tests/test_config.py::MEASURED_E2E_P95_MS`.

### Every profile, n = 3 runs each (n = 9 for `default` and `lowlatency`)

300 frames per run (186 for `rear_end_accident`), `nvpmodel` 8, one `dms.app` at a
time under the GPU mutex. Medians, with the observed range, because the run-to-run
spread is 6.5–33 ms and the gate margin is measured in single milliseconds.

| profile | n | e2e p95 median | range | FPS median | peak RSS | face hits | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `testsrc` | 3 | **35.12** | 34.78–35.75 | 29.5 | 1341 MiB | 0 | **PASS 3/3** |
| `systemd-lab` | 3 | **36.50** | 35.42–39.83 | 28.6 | 1331 MiB | 0 | **PASS 3/3** |
| `rear_end_whiplash` | 3 | 69.60 | 69.36–77.05 | 20.4 | 1385 MiB | 552 | FAIL 0/3 |
| `bmw_rear_ended` | 3 | 77.12 | 66.17–79.03 | 18.9 | 1383 MiB | 300 | FAIL 0/3 |
| `lowlatency` (SCRFD-512) | 9 | 77.15 | 65.26–82.51 | 19.6 | 1374 MiB | 452 | FAIL 1/9 |
| `default` | 9 | 77.79 | 66.50–99.45 | 18.8 | 1386 MiB | 583 | FAIL 0/9 |
| `minivan` | 3 | 80.38 | 65.70–98.32 | 17.3 | 1380 MiB | 1201 | FAIL 1/3 |
| `getting_rear_ended` | 3 | 82.16 | 76.36–82.87 | 17.1 | 1382 MiB | 589 | FAIL 0/3 |
| `family_vacation_crash` | 3 | 82.57 | 75.92–89.82 | 17.1 | 1372 MiB | 643 | FAIL 0/3 |
| `red_car_crash` | 3 | 84.21 | 78.79–91.23 | 16.3 | 1390 MiB | 300 | FAIL 0/3 |
| `guy_rear_ended` | 3 | 85.75 | 75.99–93.71 | 16.8 | 1384 MiB | 300 | FAIL 0/3 |
| `rear_end_accident` | 3 | 85.79 | 75.06–87.38 | 17.4 | 1389 MiB | 305 | FAIL 0/3 |
| `production` | — | — | — | — | — | — | **not runnable here**† |

† Three documents used to give the reason as "`csi`, no camera". True, but not the
error you get: the profile loads and hash-verifies all nine engines and then exits on
`EVENT_LOG_FATAL [Errno 13] Permission denied: '/var/lib/dms'` before it ever reaches
the camera. Both need fixing; the event-log path needs root and you hit it first.

**2 PASS, 10 FAIL, 1 unrunnable.** The two passes render a synthetic pattern and find
zero faces, so they never exercise the per-face cascade. **No profile that does real
work met the gate at its median** — on either detector. Of the 33 individual
real-clip runs on the shipped SCRFD-640 detector, **exactly one passed**: `minivan` at
65.70 ms, 0.3 ms inside the limit and at the bottom of its own 32.6 ms range.

The previous revision of this table reported one run per profile and called
`rear_end_accident` and `default` passes at 65.2 and 65.3 ms. Neither has reproduced:
`rear_end_accident` measures a median of 85.79 ms over three runs and `default` 77.79
over nine, with no pass in nine. A single observation of this pipeline carries almost
no information at this precision.

The other two halves of the contract are met everywhere: **16.3–29.5 FPS** against a
≥ 15 FPS target, and **peak RSS 1331–1390 MiB** against a 5000 MiB limit — 28% of
budget.

### Engines actually deserialised

Parsed from the `loaded …/engines/…` log lines of a real run, not from the manifest:

```
engine provenance: 9 engines verified in 0.10s (require_hash=False):
  2d106det_192, hpe_mobilenetv2_224, mwc_n_48, ocec_s_24x40_b2, scrfd_500m_640,
  sface_112, sgc_is_s_48, vsdlm_s_30x48, yolox_nano_416

model registry: {'headpose': 'model', 'eyestate': 'model', 'mouthstate': 'model',
                 'occlusion': 'model', 'gaze': 'off', 'objects': 'model',
                 'identity': 'model'}
```

Thirteen `.engine` files exist. Nine run. The four that do not:
`mobilenetv2_gaze_448` (licence-gated off), `hpe_mobilenetv3_small_224` (fallback),
`ocec_s_24x40` (batch-1 alternate), `scrfd_500m_512` (the low-latency alternate; only
`configs/lowlatency.yaml` selects it, and no shipped profile does).

### Test suite

```
PYTHONPATH=. ./.venv/bin/python -m pytest -q
-> 821 passed, 15 skipped in 87.79s

flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. \
    ./.venv/bin/python -m pytest -q tests/test_app_integration.py"
-> 58 passed in 276.20s        # the nine golden clips are nine of these
```

**Read the two lines together.** The first needs no GPU and replays no video: it is a
code-and-config result, not a detection-quality result. The second is the detection
gate, and until today it did not exist — the golden comparison was a script somebody
had to remember to run, which is how three fixtures stayed red through a whole fix
round while `pytest -q` reported zero failures. The board tests are skipped unless
`DMS_REPLAY_TESTS=1` (replay/golden) or `DMS_TRT_TESTS=1` (TensorRT supervisor) is
set. The 15 skips are exactly: 11 behind `DMS_REPLAY_TESTS` (the nine golden clips
and two pipeline smoke tests), 2 behind `DMS_TRT_TESTS`, and 2 for tools this board
does not have (`shellcheck`, `logrotate`). Nothing is skipped to hide a failure, and
both variables are now documented in `readme.md` §1 — no document mentioned them
before today.

Zero failures. The long-standing
`test_the_legacy_geometry_call_path_still_drives_the_machine` failure named in the
previous revision of this document is fixed: the fixture fed `ear=0.20` and expected
`MICROSLEEP`, but on the corrected EAR scale 0.20 is an **open** eye, so the machine
was right and the fixture was stale. It now drives a genuinely closed eye and reads
the effective threshold off the monitor rather than assuming one, with a companion
test that fails loudly on the next recalibration.

### Golden replay — 9 of 9 green, and now enforced by pytest

```
day_driver               frames=300  faces=583   p50=46.5  p95=75.8  gate=FAIL  driver_changed:4,drowsiness_unreliable:5,face_lost:3
  ok
all 9 clip(s) within their golden bands
```

**All nine fixtures are green on the shipped detector, and not one band was
re-recorded.** They were red earlier today because the detector under them changed:
the fixtures were recorded at `2026-09-13T01:36–01:40Z` on SCRFD-640, and the
afternoon's SCRFD-512 swap moved which faces are found, which moves track ids, which
is what `driver_changed` counts. `day_driver` went from 4 `driver_changed` edges to
20 against bands of [1, 3] each. Reverting the detector restored the fixtures exactly:
measured `total_face_hits` on 640 is 583 / 1201 / 300 / 643 / 305 / 300 / 300 / 589 /
552 and the midpoint of each recorded `face_hits` band is the same nine numbers.

**What changed structurally, and it is the more important half.** The comparison above
was not part of `pytest`. The suite had three structural tests over the fixtures
(schema shape, referenced files exist, alert names are real enum members) and one
board-gated test covering a single clip; the real harness was a script somebody had to
remember to run. That is exactly how a 15.8% drop in driver-face recall shipped and
survived a fix round with `pytest -q` reporting zero failures. Now:

* **`test_golden_detector_matches_the_profile_today`** — no GPU, no clip, no engine.
  Each fixture records the detector it was taken with (schema 3, `detector_engine`),
  and this compares it with what the profile selects today. Reintroducing the 512
  default turns `pytest -q` red in 15 seconds on any machine.
* **`test_golden_replay[<clip>]`** — nine tests, one per clip, behind
  `DMS_REPLAY_TESTS=1`, running the real comparison and failing on any band violation.

Both were demonstrated by putting the 512 default back: `3 failed, 215 passed` off
board in 15 s, and on board `REGRESSION day_driver: golden was recorded against
scrfd_500m_640_fp16_gpu.engine, the profile now selects scrfd_500m_512_fp16_gpu.engine`
followed by the two `driver_changed` band violations. `configs/default.yaml` was then
restored and checked byte-identical with `md5sum -c`.

## The four biggest open items

1. **Latency: the gate is missed, and the measurement is barely repeatable.** 10 of
   12 runnable profiles miss the 66 ms p95 gate at their median and the two that pass
   find no faces; the profile measured nine times spanned 66.50 → 99.45 ms and passed
   none. The cheap fix was tried today and reverted: SCRFD-512 is worth 5.4 ms of p50
   against a ~13 ms gap and costs 15.8% of the driver-ROI face hits on `day_driver`.
   One of the three known fixes is applied (`Letterbox`); the second (per-engine CUDA
   event reuse in
   `dms/runtime/trt_engine.py`, ~6 ms) is not. Beyond that the honest next step is
   not another micro-optimisation, it is **locking the GPU clock and re-measuring on
   an idle board** — the governor was observed at 204 MHz here against an 1109 MHz
   ceiling, and locking it needs root, which this environment does not have.
2. **No cabin footage, and two weak drowsiness models.** Every threshold in the
   fusion layer and every band in the golden fixtures is a starting point. The eye
   classifier reaches its model path on 16% of eyes here; the mouth classifier calls
   half of all mouths open. Two driver-facing clips with labelled events (3 × eyes
   closed ≥ 2 s, 3 × yawns, 3 × over-shoulder looks) would turn the whole alert layer
   from plausible into measured. This is the single highest-value missing artefact in
   the project and it cannot be produced by writing code.
3. **No camera, no calibration, no soak, no systemd.** Until a `forward_zero` is
   measured in a real seat, gaze alerts are unreachable and every absolute head angle
   is relative. Nothing has ever run longer than 300 frames. The unit has never been
   loaded.
4. **The golden baseline is a software baseline, not ground truth.** All nine clips
   are green against the shipped detector and the comparison now runs in `pytest`, so
   it is a clean signal again — but the bands pin the behaviour the software had when
   they were recorded, on dashcam montages with no labelled drowsiness event anywhere
   in the repository. They will catch a change; they cannot tell you the system got
   better. That still needs cabin footage (item 2).

## Reproduce in ten minutes

```bash
cd /home/nvidia/myspace/DMS
PYTHONPATH=. ./.venv/bin/python -m pytest -q
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/testsrc.yaml --max-frames 30"
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
flock /tmp/jetson-gpu.lock -c "DMS_REPLAY_TESTS=1 PYTHONPATH=. ./.venv/bin/python -m pytest -q tests/test_app_integration.py"
```

Run the third one **at least three times** and take the median. Two runs of this
pipeline routinely differ by 15 ms and one pair differed by 33 ms; a single
`PERF_GATE` line is a sample, not a verdict, and no document in this repository is
allowed to quote one as if it were. The fourth command is the detection-quality gate
and takes about 4½ minutes.

Check the board's load first (`cat /proc/loadavg`) and record it beside any number you
keep. Every figure in this document was taken at loadavg 2.4–4.5.

## Local artifacts (not in git)

- `engines/*.engine`, `engines/*.onnx` — `bash scripts/fetch_onnx.sh && python3 scripts/build_engines.py`
- `tests/replay/*.mp4` — catalogue in [`../tests/replay/CLIPS.md`](../tests/replay/CLIPS.md)
- `tests/replay/golden/*.json` — **committed**; the regression baseline
- `tests/replay/golden/synthetic/*.json` — **committed**; the scripted drowsiness golden and the per-alert coverage ledger
- `data/events.jsonl` — lab alert log
