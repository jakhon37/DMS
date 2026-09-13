# DMS performance on the Xavier NX

Every number in this document was measured on the target board, by
`scripts/bench.py` or `scripts/run_all_replay.py`, on 2026-09-13. Nothing here
is estimated, scaled from another device, or copied from a datasheet. Where a
number is uncertain the uncertainty is stated; where something has not been
measured it says so.

**Board.** Jetson Xavier NX, L4T R35.6.5 / JetPack 5.1.6, Ubuntu 20.04, 6 Carmel
cores, `nvpmodel` profile `pmode:0008` (MODE_20W_6CORE), CUDA 11.4,
TensorRT 8.5.2.2, OpenCV 4.5.4 (GStreamer yes, CUDA no, TBB, 6 threads),
Python 3.8.10. GPU devfreq governor `nvhost_podgov`, 114.75-1109.25 MHz.
6.7 GiB RAM, no camera - everything is replay of local 1280x720 mp4 clips.

**The gate.** `docs/production-dms-design.md`: 1280x720 end-to-end **p95 <= 66 ms**
(>= 15 FPS) with **peak RSS < 5.0 GB**.

---

## 1. Verdict

| configuration | e2e p50 | e2e p95 | FPS | gate |
|---|---:|---:|---:|---|
| `dms.app` as shipped today, headless flags, 2 engines | - | - | **8.6** | **FAIL by 1.8x** |
| `dms.app` as shipped today, `--save-video --save-preview` | - | - | **5.2** | **FAIL by 2.9x** |
| core pipeline (SCRFD-640 + 2d106det), CPU fixes applied | 39.2 ms | 60.5 ms | 23.3 | pass |
| core pipeline (SCRFD-512 + 2d106det), CPU fixes applied | **33.6 ms** | **41.7 ms** | **28.4** | **pass, 24 ms margin** |
| all nine engines (eight capabilities), every head on every frame | 72.5 ms | 94.7 ms | 13.2 | **FAIL** |
| nine engines, cadenced, object net every 10th frame | 44.2 ms | 77.6 ms | 20.3 | **FAIL** |
| nine engines, cadenced, object net every 20th frame (section 8) | **37.3 ms** | **62.2 ms** | **24.6** | **pass** |

Peak RSS never exceeded **2.0 GiB** in any configuration measured, against a
5.0 GB limit. **The memory half of the gate passes with 60% headroom. The
latency half does not pass today and is not a matter of tuning: the shipped code
spends more time on avoidable CPU copies than on all inference combined.**

Three statements, each backed by a measurement below:

1. **15 FPS with the core pipeline is comfortably reachable** - 28.4 FPS
   measured with the 512 detector - but only after the CPU work described in
   section 5, none of which is in files this workstream owns.
2. **15 FPS with all nine engines running on every frame is not reachable.**
   72.5 ms p50 / 94.7 ms p95 measured. It is not close, and no amount of tuning
   inside the wrappers closes a 29 ms gap.
3. **15 FPS with all nine engines at sensible cadences is reachable**, and the
   single decision that makes it so is the object net's period. Measured back to
   back in one session, on the same clip, with only that period changed:
   every 10th frame gives 44.2 ms p50 / **77.6 ms p95** (fail); every 20th frame
   gives 37.3 ms p50 / **62.2 ms p95** (pass), 24.6 FPS. The margin is 4 ms and
   this board is shared, so section 8 also lists what buys more of it.

---

## 2. Reproducing this

```bash
# CPU-only sections; no GPU, no lock needed
PYTHONPATH=. ./.venv/bin/python scripts/bench.py --sections env,cpu --iters 100

# GPU sections. bench.py takes /tmp/jetson-gpu.lock itself; if a parent already
# holds it, tell the child so it does not deadlock on its own parent's lock.
DMS_GPU_LOCK_HELD=1 flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python scripts/bench.py --sections engines,pipeline \
     --iters 100 --frames 200 --profiles current,optimised,optimised512,full512,production"

# whole-clip end-to-end through dms.app, with per-clip latency, RSS and a diff
flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python scripts/run_all_replay.py --max-frames 200 --no-overlay \
     --baseline tests/replay/runs/<previous>/summary.json"
```

Output lands in `tests/replay/runs/bench/<tag>/{bench.json,bench.md}` and
`tests/replay/runs/<stamp>/summary.json`, both gitignored.

The bench profiles are:

| profile | what it runs |
|---|---|
| `current` | exactly what `dms/app.py` does today: the stride-4 BGRA slice, `ScrfdDetector.detect`, `Landmark106.infer_one`, EAR/MAR/solvePnP |
| `optimised` | contiguous BGR frame, `dms.runtime.preprocess.Letterbox`, split detect (engine call and decode timed separately), SCRFD-640 |
| `optimised512` | the same on the 512 detector engine |
| `full512` | `optimised512` plus head pose, eye state, mouth state, occlusion, objects and identity, every head on every frame |
| `production` | `full512` with the cadences in `PRODUCTION_CADENCE` (`--cadence` overrides them) |

---

## 3. The environment caveat that dominates everything

**The GPU never reached its 1109 MHz ceiling in any measurement**, and where it
settles depends on recent history rather than on the current workload alone.
`nvhost_podgov` scales on utilisation, and a sequential pipeline that spends
25-50 ms of every frame on the CPU presents a low duty cycle.

`scripts/bench.py --sections dvfs` runs the same engine on the same input at four
different CPU gaps between submits, sampling `cur_freq` on every iteration. The
gap is busy numpy work, not a sleep, because a sleeping process yields the core
and the real pipeline does not. Two sweeps, minutes apart, GPU mutex held both
times:

Sweep A (board recently idle):

| engine | gap 0 ms | gap 10 ms | gap 25 ms | gap 50 ms |
|---|---:|---:|---:|---:|
| `scrfd_500m_512` | 9.36 @ 306 MHz | **5.92 @ 510 MHz** | 6.01 @ 510 MHz | 9.29 @ 306 MHz |
| `2d106det_192` | 5.12 @ 306 MHz | **3.40 @ 408 MHz** | 3.96 @ 408 MHz | **12.19 @ 114 MHz** |
| `hpe_mobilenetv2_224` | 4.68 @ 510 MHz | 3.52 @ 599 MHz | **3.42 @ 599 MHz** | **11.74 @ 114 MHz** |

Sweep B (immediately after a long GPU-busy period):

| engine | gap 0 ms | gap 10 ms | gap 25 ms | gap 50 ms |
|---|---:|---:|---:|---:|
| `scrfd_500m_512` | 9.64 @ 306 MHz | 6.22 @ 408 MHz | 6.34 @ 408 MHz | 6.43 @ 408 MHz |
| `2d106det_192` | 3.32 @ 408 MHz | 2.35 @ 510 MHz | 3.28 @ 510 MHz | 3.29 @ 510 MHz |
| `hpe_mobilenetv2_224` | 3.09 @ 599 MHz | 3.02 @ 701 MHz | 3.06 @ 701 MHz | 3.03 @ 701 MHz |

What this establishes, and what it does not:

* **It does establish that the clock, not the model, sets the latency.** The
  identical `2d106det` inference measured **2.35 ms at 510 MHz and 12.19 ms at
  114 MHz** - 5.2x - within the same hour, on the same board, with the same
  input. Every engine figure in this document is a clock-dependent figure, not a
  property of the engine.
* **It does establish that the ceiling is never reached.** The highest clock
  observed in any sweep was 701 MHz, 63% of the 1109 MHz maximum. Nothing this
  pipeline does makes `podgov` go to the top.
* **It does not establish a clean monotonic relationship between the frame gap
  and the clock.** Sweep A shows the 50 ms gap collapsing to 114 MHz; sweep B,
  starting from a hot governor, holds 408-701 MHz at the same gap. The
  governor's state carries across runs. That is a warning about *measurement*,
  not a result: a single timing run on this board cannot be trusted to ±30%.

Consequences for how this project measures anything:

* **The board is shared with other agents.** Repeating the identical
  `production` profile at four different times gave e2e p95 of 58.5, 62.2, 66.3
  and 77.6 ms. `scripts/run_all_replay.py` now records `loadavg_before` and
  `loadavg_after` per clip and sets `contaminated: true` in `summary.json` above
  `--max-loadavg` (default 1.5), so a contaminated timing run is identifiable
  instead of merely puzzling; `--strict-load` refuses to start one.
* **Compare within a session, never across sessions.** The cleanest numbers in
  this document (`cad-a`, `cad-b`, `cad-core` in section 6) are three profiles
  run back to back inside one `flock`, which is the only way to hold the
  governor and the neighbours roughly constant.
* **Lock the clocks before spending effort on INT8 or DLA.** The application must
  never call `nvpmodel` or `jetson_clocks` (project rule), so this is an
  operator or systemd step - `jetson_clocks`, or raising
  `/sys/devices/gpu.0/devfreq/17000000.gv11b/min_freq`. Both need root, which
  this workstream does not have. The gate should therefore be reported at
  **both** DVFS-default and clocks-locked, and only the DVFS-default number is a
  commitment to a customer.

---

## 4. Per-engine cost

`--sections engines`, 100 iterations each after warm-up, engines loaded one at a
time, GPU mutex held. "host round trip" is what the pipeline pays: pinned H2D +
`execute_async_v3` + D2H + the completion-event poll. "GPU p50" is
`trtexec --loadEngine --noDataTransfers`, transfers excluded.

| engine | role | file MB | ctx MB | host p50 | host p95 | GPU p50 | overhead |
|---|---|---:|---:|---:|---:|---:|---:|
| `ocec_s_24x40` | eye_state | 1.18 | 0.25 | 0.86 | 1.01 | 0.58 | 0.28 |
| `ocec_s_24x40_b2` | eye_state_pair | 1.74 | 0.50 | 0.92 | 1.11 | 0.65 | 0.27 |
| `vsdlm_s_30x48` | mouth_state | 1.21 | 0.37 | 0.92 | 1.19 | 0.63 | 0.30 |
| `mwc_n_48` | mouth_occlusion | 1.40 | 0.60 | 1.09 | 1.19 | 0.47 | 0.62 |
| `sgc_is_s_48` | eye_occlusion | 1.73 | 0.60 | 1.25 | 1.49 | 0.66 | 0.59 |
| `hpe_mobilenetv2_224` | head_pose | 4.69 | 3.35 | 2.16 | 2.32 | 1.38 | 0.78 |
| `scrfd_500m_512` | face_detector_fast | 1.65 | 6.45 | **3.15** | 3.41 | 2.06 | 1.08 |
| `sface_112` | driver_identity | 18.79 | 3.06 | 3.15 | 3.27 | 2.14 | 1.01 |
| `hpe_mobilenetv3_small_224` | head_pose_fallback | 4.66 | 1.60 | 3.81 | 4.00 | 2.33 | 1.48 |
| `2d106det_192` | landmarks_106 | 3.61 | 1.13 | 4.10 | 8.64 | 1.08 | 3.02 |
| `scrfd_500m_640` | face_detector | 1.69 | 6.96 | **4.50** | 4.65 | 2.65 | 1.85 |
| `mobilenetv2_gaze_448` | gaze | 5.58 | 16.08 | 5.59 | 5.91 | 3.80 | 1.79 |
| `yolox_nano_416` | cabin_objects | 3.00 | 5.03 | 5.71 | 5.94 | 4.44 | 1.27 |

Reading it:

* **Total pure GPU compute for the whole model zoo is under 25 ms.** Inference
  is not the problem.
* **Host overhead is 0.3-3.0 ms per engine** on top of GPU compute, and it does
  not scale with model size - `2d106det` pays 3.0 ms of overhead on 1.1 ms of
  compute. That is the fixed cost of a submit/poll round trip (`TrtEngine` polls
  the completion event at 0.5 ms granularity, so a sub-millisecond engine
  frequently waits a full extra poll interval). Six small heads therefore cost
  ~6 ms of pure overhead per frame, which is as much as the models themselves.
  `_poll_event` already backs off adaptively from 0.2 ms to 2 ms, so the
  remaining identifiable fixed cost is the `cudaEventCreate`/`cudaEventDestroy`
  pair issued on every single inference.
* Loading everything at once costs ~53 MB of weights plus ~48 MB of context
  scratch, which is negligible against a 5.0 GB budget. **Memory is not a reason
  to drop a model on this board; latency is.**

---

## 5. Where the non-GPU time goes

`--sections cpu`, 100 iterations, one real 1280x720 frame from `day_driver.mp4`.

| stage | p50 ms | p95 ms | what it is |
|---|---:|---:|---|
| `slice_gather_ascontiguous` | **7.19** | 14.34 | the hidden cost of `frame.full_bgra[..., :3]`: a stride-4 view that every OpenCV call silently materialises |
| `bgra_to_bgr_cvtColor` | **0.33** | 0.56 | doing that conversion once, explicitly, into a reused buffer |
| `scrfd_pre_current_contiguous` | 39.47 | 42.39 | `dms/infer/scrfd.py::preprocess` as written, on a contiguous frame |
| `scrfd_pre_current_strided` | **86.26** | 87.77 | the same, on the stride-4 view the app actually passes it |
| `scrfd_pre_letterbox_640` | **4.06** | 5.13 | `dms.runtime.preprocess.Letterbox.for_scrfd((640,640))` |
| `scrfd_pre_letterbox_512` | 3.30 | 4.45 | the same at 512 |
| `scrfd_pre_blobFromImage_640` | 27.16 | 28.13 | `cv2.dnn.blobFromImage` on the same canvas |
| `landmark_warp_full_strided` | **26.77** | 27.95 | `warpAffine` over the whole 720p stride-4 view |
| `landmark_warp_full_contiguous` | **1.87** | 1.93 | the same warp over a contiguous frame |
| `overlay_full_alloc` | 6.22 | 6.76 | overlay at full resolution, fresh canvas per frame |
| `overlay_full_reused` | 6.09 | 6.46 | overlay at full resolution into a reused buffer |
| `overlay_half_reused` | **3.63** | 3.84 | overlay at 640x360 into a reused buffer |
| `overlay_strided_alloc` | **30.28** | 31.85 | the old path in full: overlay on the stride-4 view |
| `overlay_disabled` | **0.00** | 0.01 | the production configuration |
| `videowriter_mjpg_720p` | **76.84** | 79.00 | one `cv2.VideoWriter` MJPG frame at 720p |
| `videowriter_mjpg_360p` | 20.77 | 22.06 | the same at 360p |
| `stage_timer_span` | 0.01 | 0.01 | one instrumented span (`dms.perf.timers`) |

Four findings, in order of size:

1. **`cv2.VideoWriter` MJPG at 720p costs 76.8 ms per frame** - more than the
   entire rest of the pipeline. `scripts/run_all_replay.py` passed
   `--save-video` for every clip, so every archived timing run in this repo was
   measured with a 77 ms/frame encoder in the loop. End-to-end confirmation:
   `day_driver` 191.6 ms/frame with the overlay flags versus 116.8 ms/frame
   without, a 74.8 ms delta against the 76.8 ms micro-benchmark.
2. **The stride-4 slice roughly doubles or triples every OpenCV call on it.**
   The detector preprocess goes 39.5 -> 86.3 ms and the landmark warp
   1.9 -> 26.8 ms purely from being handed a non-contiguous view. Converting
   once with `cvtColor` costs 0.33 ms.
3. **The detector preprocess is allocator-bound, not arithmetic-bound.** The
   current five-pass float path allocates ~20 MB of temporaries per frame;
   glibc returns blocks that large to the kernel, so every frame re-faults them.
   `Letterbox`, which preallocates every destination, is **10x faster on the
   same arithmetic** (39.5 -> 4.1 ms). `cv2.dnn.blobFromImage` only reaches
   27.2 ms, because it still allocates its output - preallocation, not the C++
   inner loop, is where the win is.
4. **The overlay ran on every frame in the headless configuration** because
   `dms/app.py:243` reads `if want_overlay or not health.degraded:` where `and`
   was meant. That is 30.3 ms per frame of drawing that is immediately
   discarded, and it also inverts the intent: the overlay was drawn *only* when
   the board was **not** thermally degraded.

**Accounting for the 116.8 ms/frame that `dms.app` actually costs today**
(day_driver, 200 frames, no overlay flags, measured wall clock):

| item | ms/frame | how it was measured |
|---|---:|---|
| capture (GStreamer decode + BGRA copy + queue) | 6.6 | marginal cost of 800 frames vs 200 frames with `--detect` off (7.730 s vs 3.770 s) |
| detect (preprocess + engine + decode) | 36.6 | bench `current` profile, in-pipeline p50 |
| landmarks (warp + engine + inverse affine) | 19.2 | bench `current` profile |
| overlay drawn and discarded | 30.3 | CPU section, `overlay_strided_alloc` |
| EAR + MAR + solvePnP | 2.4 | bench `current` profile, `fusion` |
| tracker + driver selection | 0.4 | bench `current` profile |
| **accounted** | **95.5** | |
| residual: app bookkeeping, health/thermal reads, event log, Python loop | ~21 | by difference |

The residual is the one number here obtained by subtraction rather than
measurement. Closing it needs `dms.perf.StageTimer` wired into `dms/app.py`,
which is the handoff in section 11 - the tooling exists and the frame loop does
not use it yet.

---

## 6. Assembled pipeline

`--sections pipeline --frames 150`, `day_driver.mp4`, all five profiles in one
process so they are directly comparable (bench tag `final-01`; the board carried
loadavg 1.9 during this run, so treat the absolute values as a slow case and the
*differences* as the result).

| profile | e2e p50 | e2e p95 | FPS | peak RSS |
|---|---:|---:|---:|---:|
| `current` | 58.7 | 86.9 | 15.9 | 1193 MiB |
| `optimised` (640) | 39.2 | 60.5 | 23.3 | 1193 MiB |
| `optimised512` | **33.6** | **41.7** | **28.4** | 1193 MiB |
| `full512` (every head, every frame) | 72.5 | 94.7 | 13.2 | 1830 MiB |
| `production` (cadenced) | 39.3 | 66.3 | 22.6 | 1849 MiB |

Per-stage, `optimised512` versus `current`:

| stage | current p50 | optimised512 p50 | delta |
|---|---:|---:|---:|
| bgra | 0.01 (a view, cost deferred) | 0.77 (explicit `cvtColor`) | +0.76 |
| detect (pre + gpu + post) | 36.58 | 3.72 + 13.19 + 2.98 = 19.89 | **-16.7** |
| landmarks | 19.23 | 9.67 | **-9.6** |
| fusion | 2.44 | 0.58 | -1.9 |
| **e2e** | **58.69** | **33.62** | **-25.1** |

`fusion` falls because `current` runs `solve_head_pose` (an OpenCV `solvePnP`
per frame) and `optimised512` does not - in the real pipeline that work moves to
the head-pose engine, priced separately at 9.2 ms.

The `production` profile at 150-200 frames, repeated on the same board at
different contention levels and with two object-net periods:

| run | objects period | e2e p50 | e2e p95 | FPS |
|---|---|---:|---:|---:|
| `cadence-01` (quiet board) | every 10th | 36.4 | **58.5** | 25.7 |
| `final-01` (loadavg 1.9) | every 10th | 39.3 | **66.3** | 22.6 |
| `cad-a` | every 10th | 44.2 | **77.6** | 20.3 |
| `cad-b` (immediately after `cad-a`) | **every 20th** | **37.3** | **62.2** | **24.6** |
| `cad-core` (no heads at all, same session) | - | 32.7 | 43.3 | 29.9 |

`cad-a`, `cad-b` and `cad-core` ran back to back inside one `flock` session, so
they are the cleanest comparison in this document: **-15.4 ms of p95 for one
cadence change**, and the nine-engine cadenced pipeline costs 18.9 ms of p95
over the bare core pipeline.

---

## 7. SCRFD 512 versus 640 - recommendation

| | 640 | 512 | delta |
|---|---:|---:|---:|
| engine alone, host round trip p50 | 4.50 ms | 3.15 ms | -1.35 ms |
| engine alone, GPU compute p50 (`trtexec`) | 2.65 ms | 2.06 ms | -0.59 ms |
| preprocess (`Letterbox`) p50 | 4.06 ms | 3.30 ms | -0.76 ms |
| in-pipeline `detect_gpu` p50 | 16.90 ms | 13.19 ms | **-3.71 ms** |
| in-pipeline `detect_post` p50 | 4.35 ms | 2.98 ms | **-1.37 ms** |
| context scratch | 6.96 MB | 6.45 MB | -0.51 MB |
| **whole pipeline e2e p50** | **39.24 ms** | **33.62 ms** | **-5.62 ms** |
| **whole pipeline e2e p95** | **60.45 ms** | **41.71 ms** | **-18.74 ms** |

Both columns come from the same process (`final-01`), so they are comparable
with each other; the p95 row in particular reflects one run on a board carrying
loadavg 1.9, and the **p50 delta of -5.6 ms is the figure to plan with** - a
quieter session (`cad-core`) put the 512 core pipeline at 32.7 ms p50 / 43.3 ms
p95, consistent with the p50 but not with an 18.7 ms p95 saving.

**Recommendation: ship the 512 engine.** It costs 5.6 ms of p50 for the whole
pipeline - far more than the 1.35 ms the
engine-alone comparison suggests, because the decode also shrinks (8192/2048/512
anchor rows instead of 12800/3200/800) and the smaller preprocess and smaller
D2H buffers compound. On a 66 ms budget that is 8.5% of the whole allowance, and
it is the cheapest item on the ranked list that nobody has to write code for.

**The caveat this workstream cannot settle.** 512 vs 640 is an accuracy/latency
trade and only the latency half has been measured. A driver's face at 720p is
typically 150-250 px wide, well above SCRFD-500m's smallest reliable scale at
either input size, so the loss should be small - but "should be" is not a
measurement. Before the switch is made permanent, someone must run both engines
over the nine replay clips and compare detection recall and box IoU on the
driver track. `scripts/bench.py --profiles optimised,optimised512` gives the
latency half of that comparison; the accuracy half belongs to the perception
workstream.

---

## 8. The degrade profile that is actually reachable

Nine engines (eight capabilities: detector, landmarks, head pose, eye state,
mouth state, occlusion via two heads, objects, identity) on every frame is
72.5 ms p50 / 94.7 ms p95 - a hard fail. The
following cadences were measured (`--cadence` in `scripts/bench.py`) and are what
should ship:

| capability | cadence | cost when it runs | amortised | why |
|---|---|---:|---:|---|
| face detector (SCRFD-512) | every frame | 13.2 ms | 13.2 ms | everything downstream depends on it |
| landmarks (2d106det) | every frame | 9.2 ms | 9.2 ms | EAR/MAR geometry and every crop box |
| head pose (MobileNetV2) | every frame | 9.2 ms | 9.2 ms | distraction timing needs per-frame yaw |
| eye state (OCEC batch-2) | every frame | 0.76 ms | 0.76 ms | a blink is 100-300 ms; sub-sampling loses microsleeps |
| mouth state (VSDLM) | every frame | 0.33 ms | 0.33 ms | free |
| occlusion (SGC + MWC) | every 15th, phase 3 | 7.7 ms | 0.51 ms | sunglasses and masks do not appear mid-blink |
| objects (YOLOX-Nano) | **every 20th, phase 7** | 23-29 ms | 1.2-1.5 ms | see below |
| identity (SFace) | once per new track id | 12.4 ms | ~0 | a driver change is a rare event |

Two rules that matter as much as the periods:

* **Stagger the phases.** The two expensive intermittent heads must never land
  on the same frame. `PRODUCTION_CADENCE` uses `objects=(20,7)` and
  `occlusion=(15,3)`, whose residues never coincide (`tests/test_perf.py`
  asserts this). Coincident spikes are what turns a good p50 into a bad p95.
* **Choose the object-net period from the percentile, not from intuition.**
  At every 10th frame the object net runs on 10% of frames, so the p95 lands
  *on* an object-net frame and inherits its full 23-29 ms. At every 20th it runs
  on 5% and the p95 sits just below it. Measured, that single change is worth
  **15.4 ms of p95** for no change in phone-detection latency that matters (a
  phone in the hand persists for seconds, not frames).

Measured with these cadences (`--cadence objects=20:7,occlusion=15:3`):
**e2e p50 37.3 ms, p95 62.2 ms, 24.6 FPS**, peak RSS 1.99 GiB. The same
configuration with the object net at every 10th frame instead measured 44.2 /
77.6 / 20.3 in the immediately preceding run - the period is worth 15.4 ms of
p95 on its own.

**Honest conclusion.** With the CPU fixes and these cadences the pipeline passes
the 66 ms p95 gate with **3.8 ms of margin** and passes the 15 FPS gate with 64%
of margin, at 40% of the RSS limit. Four milliseconds is not a comfortable
latency margin on a board whose GPU clock has been observed to move an identical
inference by 5.2x, so at least one of the following should be done before the
gate is called met, in order of preference:

1. **Move the object net off the frame thread.** At 23-29 ms it is 60% of the
   p95 spike and it is the only head whose result is not needed within one
   frame. Running it on a second thread against its own `TrtEngine` and
   publishing the last result would remove the spike entirely; the GPU has the
   capacity (its own compute is 4.4 ms).
2. **Lock the GPU clock** (section 3). The duty-cycle sweeps show up to 5.2x on
   a single engine between 114 and 510 MHz, and the ceiling of 1109 MHz is never
   reached at all. Holding 510 MHz instead of the 306-408 MHz the pipeline
   actually sees is worth roughly 5-8 ms per frame across the detector and the
   landmark head; the exact figure cannot be measured without root.
3. **Drop head pose to the geometric fallback** when the budget is tight.
   `solve_head_pose` costs 2.4 ms against the engine's 9.2 ms; the accuracy
   cost is real and belongs to the perception workstream to judge, but the
   mechanism exists (`ModelRegistry(backends={"headpose": "geometry"})`).

---

## 9. Ranked optimisation list

Savings are measured, per frame, at 1280x720. "Owner" is the workstream that
owns the file - none of the top four are in `dms/perf`, `dms/viz` or `scripts/`.

| # | change | measured saving | effort | owner |
|---:|---|---:|---|---|
| 1 | Stop writing MJPG video during timing runs; when a video is genuinely wanted, write it at 360p | **-76.8 ms** (720p) / -56 ms (360p instead of 720p) | S | done - `run_all_replay.py --no-overlay`; the app's `--save-video` remains a debug flag |
| 2 | Fix `if want_overlay or not health.degraded:` to `and`, and take the overlay through `OverlayRenderer` | **-30.3 ms** | S | `dms/app.py` (integrator) |
| 3 | Replace `ScrfdDetector.preprocess` with `dms.runtime.preprocess.Letterbox` | -35 ms standalone, **-16.7 ms in-pipeline** (with #4) | M | `dms/infer/scrfd.py` (perception) |
| 4 | Convert BGRA->BGR once per frame with `cvtColor` into a reused buffer; never pass the stride-4 view downstream | **-9.6 ms** on landmarks, and it is what makes #3 achievable | M | `dms/app.py` + `dms/infer/*` |
| 5 | Ship the 512 detector engine instead of the 640 | **-5.6 ms p50** (p95 saving seen once at -18.7 ms; do not plan on it) | S | `configs/*.yaml` |
| 6 | Run the object net every 20th frame with a staggered phase (or move it to a second thread) | **-15.4 ms p95**, measured back to back | S | `configs/*.yaml` `objects.every_n` |
| 7 | Lock the GPU clock for timing runs and report the gate at both clocks | up to 5.2x on one engine between 114 and 510 MHz; est. 5-8 ms/frame, not directly measurable without root | S | operator/systemd, not the app |
| 8 | Create each engine's completion event once with `cudaEventDisableTiming` instead of create+destroy per inference | not isolated; the total host-over-GPU overhead it sits inside is 0.27-3.02 ms per engine, ~6 ms/frame across six small heads | S | `dms/runtime/trt_engine.py` (runtime) |
| 9 | Pin `cv2.setNumThreads` | not measured - see section 12 | S | `dms/app.py` |
| 10 | Async submit/wait overlap of the detector with the previous frame's post-processing | not measured; upper bound is the 13.2 ms `detect_gpu` stage | L | `dms/app.py` + `dms/runtime` |

**Rejected, with reasons:**

* **INT8.** The whole model zoo is 25 ms of GPU compute against a frame that
  currently loses 100+ ms to CPU copies and DVFS. INT8 would halve part of the
  25 ms at the cost of a calibration set, a per-model accuracy re-validation and
  a second set of engines to hash-pin and maintain. It is the right optimisation
  for a pipeline that is GPU-bound; this one is not, and will not be until items
  1-7 are done. Revisit only after re-measuring with the clocks locked.
* **DLA.** `2d106det` is MobileNet-0.5 with PReLU, which TRT 8.5's DLA does not
  support, so the layer audit will reject it; and it is 1.08 ms of GPU compute.
  The `dla_core` parameter exists in `TrtEngine` and is now honoured, but no DLA
  engine has ever been built and the payoff is a rounding error. The
  `models.landmarks.dla_engine` key in every config file points at a file that
  does not exist and should be removed.
* **Batching the small heads.** Tempting - six heads pay ~6 ms of fixed submit
  overhead - but they have different input shapes and different crops, so a
  single batched engine would have to be built as one graph. Not worth it before
  items 1-7.

---

## 10. The instrumentation this workstream added

### `dms.perf.timers`

`StageTimer` keeps a fixed-size ring buffer per stage (default 300 samples =
20 s at 15 FPS) in a preallocated float64 array. Recording a sample allocates
nothing; a span costs **~10 us measured** including the measurement harness
around it, which is 0.02% of a 66 ms frame even with twenty stages.

It replaces `infer_ms = []`, which grew one Python float per frame forever and
was fully sorted every fifteen frames. At 15 FPS an eight-hour soak reaches
~432,000 elements; `tests/test_perf.py::test_snapshot_cost_does_not_grow_with_uptime`
asserts the new snapshot cost is flat after 200,000 samples.

The old percentile was also a *lifetime* percentile - structurally incapable of
showing a regression, because one bad minute cannot move a number averaged over
an hour. Both the archived run and a fresh one show it decaying monotonically
while throughput barely moves:

| | frames 15 | 45 | 135 | 195 |
|---|---:|---:|---:|---:|
| `perf-nooverlay/day_driver`, `infer_p95_ms` | 115.4 | 84.2 | 79.0 | **74.8** |
| the same run, `fps` | 8.3 | 10.4 | 11.4 | 11.9 |

and over a whole clip in the archived `20260909-042432` run: 141.9 -> 79.0 ->
74.9 -> **67.0** across frames 15 to 6000, while fps went 4.0 -> 6.7. A metric
that only ever falls cannot raise an alarm.

### `dms.perf.budget`

`LatencyBudget` holds the canonical stage list, the per-stage target
milliseconds and the writer for **`latency_budget.json`** - the freeze
deliverable that names which stage blew its budget. Targets default to the
measured p95 of the reachable production profile rounded up, not to aspirations,
so a `fail` means "something got worse". The nine mandatory targets sum to
40.0 ms, leaving 26 ms of the 66 ms gate for the cadenced heads and margin.

The report distinguishes four states that a single boolean would have conflated:
`pass`, `fail`, `insufficient_data` (fewer than 20 samples - never reported as a
pass) and `not_run`/`missing`. The gate block reports latency and RSS
separately, and **an unmeasured peak RSS is not a pass** - that is the specific
failure mode that let nine full-clip runs record zero memory evidence against a
gate that is half about memory.

Accounting is amortised: a head that runs on every tenth frame contributes
`mean_ms * count / frames`, one tenth of its cost. Summing raw p50s across
cadenced stages over-counts by 2x and is how a budget starts lying; both numbers
are reported so the discrepancy is visible.

### `dms.viz`

`OverlayRenderer` decides once whether anything will consume the pixels and
returns `None` before touching a frame when nothing will (`overlay_disabled`:
0.00 ms). When it is on it owns the destination buffer, so a frame costs one
`cv2.resize`/`np.copyto` and no allocation, and it can render at a `scale` -
0.5 costs 3.63 ms against 6.09 ms at full resolution. Landmarks are painted by
three vectorised colour-grouped pixel assignments instead of 106 `cv2.circle`
calls. `thermal_gate(True)` sheds the overlay under thermal pressure and
restores the operator's setting when the board cools.

`draw_faces` keeps its original signature, so `dms/app.py` continues to work
unchanged while the integrator moves it onto the renderer.

### `scripts/run_all_replay.py`

Now samples the child's `/proc/<pid>/status` `VmHWM` (the kernel's exact
high-water RSS), the system `MemAvailable` low-water mark, the GPU devfreq clock
range and the load average while each clip runs, parses the app's own log for
frames and latency, counts alerts by type from `events.jsonl`, and writes it all
to `summary.json` (`schema: dms.replay_summary/2`). `--baseline <previous
summary.json>` diffs the two and prints a `REGRESSION` line per field that moved
in the bad direction by more than `--tolerance-pct`.

It also exports `DMS_LATENCY_BUDGET_OUT=<clip_dir>/latency_budget.json` to every
child. `dms.app` does not read it yet; when the frame loop is wired to
`LatencyBudget` it should write its report there, and the batch summary will pick
up the gate verdict automatically. Until then the field is simply absent.

---

## 11. What the frame loop still has to do

`dms/app.py` is not owned by this workstream. The tooling is built and tested;
the loop does not call it. In priority order:

1. `bgr = frame.full_bgra[..., :3]` -> `bgr = self._to_bgr(frame.full_bgra)`
   using `dms.runtime.preprocess.BgraToBgr`. Everything else depends on this.
2. `if want_overlay or not health.degraded:` -> an `OverlayRenderer` built from
   the three flags; guard `preview = vis` and `writer.write(vis)` on
   `overlay.enabled`; call `overlay.thermal_gate(hot)` where `health.degraded`
   is set.
3. `infer_ms = []` -> `budget = LatencyBudget.from_config(cfg)`, wrap the stages
   named in `dms.perf.budget.STAGES`, publish `budget.timer.p95_map()` into
   `health.latency_ms`, rename the existing key to `infer_p95` and add a real
   `e2e_p95`, and write `budget.write(os.environ["DMS_LATENCY_BUDGET_OUT"])` on
   shutdown when that variable is set.
4. Record `capture_lag_ms = (time.monotonic_ns() - frame.t_mono_ns) / 1e6` as a
   stage; it is the cheapest possible signal that the consumer has fallen behind
   the producer.
5. `budget.observe_rss(ram_used_mb())` in the periodic block and
   `budget.observe_rss(dms.perf.peak_rss_mb())` at shutdown.
6. Add a `perf:` section to `dms/config/schema.py` (`window`,
   `gate_e2e_p95_ms`, `gate_peak_rss_mb`, `targets`) - `LatencyBudget.from_config`
   already reads it and falls back to the built-in table when it is absent.

---

## 12. Not measured

Stated so nobody mistakes silence for a result.

* **An eight-hour soak.** Every figure here is 150-200 frames. The ring buffer
  and the reused overlay buffer are bounded by construction and tested as such,
  but no long-run RSS curve exists. Peak RSS at 200 frames is 0.74-2.01 GiB
  against a 5.0 GB limit.
* **The clocks-locked gate.** Needs root. See section 3.
* **`cv2.setNumThreads`.** OpenCV runs 6 TBB threads by default and the capture
  thread, the TensorRT host code and the letterbox all compete for the same six
  cores inside a 20 W envelope. Nothing in `dms/` pins the count and the cost of
  the contention has not been measured. It is item 9 on the ranked list rather
  than higher because it is a guess until someone sweeps it.
* **Accuracy of anything.** This is a latency and memory document. The 512 vs
  640 recommendation, the head-pose-versus-geometry fallback and every cadence
  in section 8 all have an accuracy dimension that belongs to the perception
  workstream.
* **A live camera.** There is no `/dev/video*` on this board. The 6.6 ms capture
  figure is GStreamer mp4 decode, which a CSI sensor path will not resemble.
