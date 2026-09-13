# Architecture of the assembled system

How the pieces actually fit together, as built. The v1 contract is
[`production-dms-design.md`](production-dms-design.md); this document describes
the running tree and the places where it deviates.

---

## 1. Process shape

One process. One inference thread (the main loop). One capture thread inside
`GstSource`. One daemon thread serving health. Nothing else — `multiprocessing`
is banned repo-wide and `tests/test_no_mp.py` enforces it.

```
                     ┌──────────────────────────┐
  GStreamer  ───────►│ GstSource (capture thrd) │
  NVMM appsink       │  BGRx copy before unmap  │
                     └────────────┬─────────────┘
                                  │ Queue(depth=1)
                     ┌────────────▼─────────────┐
                     │   CaptureSupervisor      │  reconnect ladder, stall
                     │   read() -> Frame | None │  detection, camera_fail edges
                     └────────────┬─────────────┘
                                  │
  ┌───────────────────────────────▼──────────────────────────────────┐
  │                        dms/app.py frame loop                     │
  │  BGR ─► SCRFD ─► tracker ─► DriverSelector ─► 2d106det           │
  │            └─► ModelRegistry: occlusion, eyes, mouth, pose,      │
  │                gaze, objects, identity                           │
  │        ─► DriverSignals ─► DriverMonitor ─► AlertEvent edges     │
  └───┬──────────┬───────────┬──────────────┬───────────┬────────────┘
      │          │           │              │           │
   EventLog   GpioAlert   ClipRing    HealthState    LatencyBudget
  events.jsonl  critical   pre/post     /healthz      stage timers
                 line       clips       /metrics    latency_budget.json
```

## 2. Ownership and lifetime

`DmsApp` owns everything. `setup()` acquires, `run()` iterates, `teardown()`
releases — in a `try/finally` so a crash in the loop still releases the GPIO
line, closes the event log, stops the clip ring and closes every TensorRT engine.

`ModelRegistry` owns the `TrtEngine` objects for the seven capability wrappers;
`DmsApp` owns the two for SCRFD and 2d106det. Wrappers borrow engines and never
close them.

Teardown order is deliberate: **GPIO first** (a physical alert line must not stay
asserted while the rest of the shutdown runs), then capture, then the encoders,
then the logs, then the engines, then health.

## 3. The three honesty rules

These are what separate this from a demo.

**A capability reports what happened, not what was asked.** `ModelRegistry`
resolves each capability to `model` / `geometry` / `off` after trying to load. A
missing engine degrades; it never raises into the frame loop. Anything that asked
for `model` and did not get it appears in `degraded_capabilities()`, drives
`MODEL_DEGRADED`, and is visible in `/healthz.engines`.

**A signal that cannot be trusted is `UNAVAILABLE`, never a plausible number.**
Sunglasses make the eye channel unavailable rather than "open". An eye box under
20 source pixels falls through to geometry because the classifier was *measured*
to invert below that. An uncalibrated pose zero makes gaze unavailable. In every
case `DROWSINESS_UNRELIABLE` fires instead of a confident wrong answer.

**A failure is a state, not an exception.** A dead camera is `camera=failed` plus
`CAMERA_FAIL` plus `/healthz` 503 plus a withheld watchdog ping. A full disk is
`DISK_FULL` plus a counted, explained hole in the event log. An engine error is a
skipped frame and a counter, until 30 consecutive failures make it a non-zero
exit so the service manager restarts a clean process.

## 4. Clocks

Three, deliberately distinct:

| Clock | Source | Used for |
| --- | --- | --- |
| **media** | GStreamer buffer PTS | the alert state machine on file/test sources |
| **monotonic** | `time.monotonic()` | the state machine on a live camera; all latency |
| **wall** | `time.time()` | the `ts` field of every log and event record |

Every event record carries wall, monotonic **and** the kernel boot id, so a
post-incident reconstruction survives a reboot and an NTP or GNSS step.

Using the consumer's wall clock on replay was a real defect: at ~8 FPS processing
of 30 FPS media, a 10-frame blink accumulated 1.17 s of "closed" against a 1.5 s
microsleep threshold. Media time removes that entirely and is what makes the
golden fixtures reproducible.

When no frame arrives, the loop still advances the machine by real elapsed time
(`_tick_no_frame`). Without it a frozen camera is silence. The tick is suppressed
before frame #1: until capture has delivered something there is no driver to have
lost, and the condition that matters — "the camera never started" — is the
supervisor's first-frame timeout.

## 5. Alert plumbing

Alerts have exactly three producers, and every declared `AlertType` has one:

| Producer | Alerts |
| --- | --- |
| `DriverMonitor.update()` | FACE_LOST, MICROSLEEP, PERCLOS_HIGH, YAWN, FATIGUE_CLUSTER, GAZE_AWAY, PHONE, DISTRACTION, DROWSINESS_UNRELIABLE, DRIVER_CHANGED, MODEL_DEGRADED |
| `CaptureSupervisor.take_events()` | CAMERA_FAIL, PIPELINE_STARVED |
| `dms/app.py` directly | THERMAL_THROTTLE (hot/cool edges), ENGINE_FAIL (load and persistent-failure), DISK_FULL (via the `EventLog`/`ClipRing` callback) |

`DmsApp._emit` is the single funnel: it logs, writes the JSONL record (which also
increments the Prometheus counters — calling `record_alert` again would double
count), and triggers the clip ring on warn/critical rising edges.

## 6. Instrumentation

`LatencyBudget` wraps a fixed-memory `StageTimer`: one ring buffer per stage, no
allocation per span, nearest-rank percentiles into a reused scratch array. The
frame loop times 16 stages plus `e2e`.

The report includes an **accounting** block: the amortised sum of stage means
against the e2e mean. On the reference run that is 59.8 ms against 60.9 ms — 1.1 ms
unaccounted, which is the evidence that the stage table actually covers the loop
rather than measuring a convenient subset.

Two known gaps in the table, both stated in the report's `context`:

* `crop` has no samples. The per-head crops are cut *inside* each wrapper, so
  their cost is already inside that head's stage; a separate `crop` row would
  double count. It shows as `missing`, which is honest, rather than as a zero.
* `letterbox` / `detect_gpu` / `detect_post` are split by calling
  `ScrfdDetector.preprocess`, `engine.infer` and `postprocess` from the app rather
  than `detect()`. That is public API, but it means the app knows the detector has
  three phases.

`DMS_LATENCY_BUDGET_OUT=<path>` makes the process write `latency_budget.json` at
shutdown, including the pass/fail gate and the peak RSS. An **unmeasured** peak
RSS is explicitly not a pass.

## 7. Deviations from the design document

| Deviation | Why |
| --- | --- |
| FACE_LOST enters WARN and escalates to CRITICAL after 3 s | An instant critical off a 1.0 s detector gap is the textbook alarm-fatigue generator. `state.face_lost_crit_s: 0.0` restores the literal design. |
| PERCLOS_HIGH has a 2.0 s enter dwell | With none, the first valid window can fire a critical off one sample. |
| PERCLOS is time-weighted and validity-gated | A count-weighted window at 6 FPS reports a number it has no right to. It now reports "invalid" instead. |
| Perception is on by default; `--detect` is legacy | A vehicle profile invoked without a flag must not silently run capture-only. |
| The calibration file merges into every profile | The old default.yaml-only rule made `production.yaml` uncalibratable, which made gaze alerts permanently dead on the one profile that needs them. |
| `models.pose.mode`, `models.objects.enabled/engine`, `models.landmarks.dla_engine`, `health.watchdog_s` deleted | Dead keys. Backends are selected by `models.backends`; the DLA engine named by the old key never existed. |

## 8. Where the bodies are buried

* The **106-point landmark topology** was wrong until this cycle: EAR was computed
  over mouth points. Every threshold that predates the fix is meaningless, which
  is why `configs/vehicle.yaml` was emptied and why the schema rejects the old
  scale by name.
* The **eye classifier inverts on small faces** (42% closed where geometry says
  4%, below ~20 px of eye width). `models.eyestate.min_eye_box_px` is a floor
  derived from the same footage it protects against, not a validated threshold.
* **Head-pose pitch sign** is the weakest of the three axes (correlation −0.58
  against a noisy geometric reference). Yaw is solid (−0.96). No nod-off threshold
  should be tuned on pitch before a real cabin clip exists.
* **Camera intrinsics are fabricated.** `camera_matrix()` assumes fx = fy = frame
  width, about 53° horizontal FOV; a cabin DMS lens is 100–140°. The model path
  does not depend on intrinsics, which is one more reason to prefer it over PnP.
* **`Frame.pts_ns` and `Frame.bgr` are attached attributes**, not declared
  dataclass fields, because `dms/types.py` belongs to the orchestrator. Always
  read them through `dms.capture.frame_pts_ns` / `frame_bgr`; a
  `dataclasses.replace()` anywhere in the pipeline would silently drop both.
