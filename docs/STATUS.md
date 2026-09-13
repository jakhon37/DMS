# Status — 2026-09-13

Six workstreams (engines, models, runtime, ops, alerts, perf) landed, and this
integration pass wired them into one running system. This document is the honest
state of that system on this Xavier NX today.

## Verdict

**A complete, integrated driver-monitoring pipeline that runs end to end on
replay, with eleven real neural models and no stubs — and no vehicle evidence of
any kind.** It is a credible v1 codebase. It is not a validated product.

| Working, measured | Missing or unproven |
| --- | --- |
| 11 TensorRT FP16 engines, hash-pinned, all live in the frame loop | No camera has ever been attached (`/dev/video*` absent) |
| Media-time state machine, hysteresis, occlusion gating, honesty alerts | No calibration → every gaze alert is suppressed by design |
| Supervised capture: reconnect ladder, `CAMERA_FAIL`, `/healthz` 503, watchdog withheld | Never exercised against a real camera unplug |
| Durable `events.jsonl` (boot id, seq, fsync policy, ENOSPC handling) | Clip ring's GStreamer encode path still never executed |
| `/healthz` `/livez` `/readyz` `/metrics`, ~40 Prometheus series | Health endpoint never scraped by anything but curl |
| Golden replay over all nine clips, reproducible run to run | Goldens pin software behaviour, **not** ground truth |
| Peak RSS 1376 MiB against a 5000 MiB gate | e2e p95 87.2 ms against a 66 ms gate — **fails** |
| Strict config schema; the old broken EAR scale is now rejected by name | Thresholds are starting points, swept on no cabin footage |
| `deploy/` unit, logrotate, tmpfiles, staged installer | Never installed: `sudo` needs a password on this board |

## What changed in this pass

1. **`dms/app.py` rewritten** around the pieces the other workstreams built:
   `CaptureSupervisor`, `ModelRegistry`, a per-frame `DriverSignals`, the new
   `DriverMonitor`, `LatencyBudget` stage timers, `EventLog`, `ClipRing`,
   `HealthState`/`HealthServer`, `OverlayRenderer`. The unbounded `infer_ms`
   list and its twice-a-second full sort are gone.
2. **The time base is fixed.** The state machine runs on GStreamer buffer PTS for
   file/test sources. Previously it ran on the consumer's wall clock, so every
   duration threshold fired roughly three times too eagerly on replay and every
   archived `events.jsonl` was timing-distorted.
3. **A capture stall now produces alerts.** The loop advances the state machine on
   frames that never arrive, so a frozen camera yields `FACE_LOST` instead of
   silence.
4. **`configs/production.yaml` is a vehicle config.** The verified defect —
   `camera.fail_fatal: false` meaning "notify READY, ping the watchdog for ever,
   never retry" — is fixed. See below.
5. **The config schema is strict.** Every section forbids unknown keys; ~120 keys
   the other workstreams read by name are now declared, range-checked and
   documented. A config on the old EAR scale is refused with the reason.
6. **Golden replay exists** (`tests/replay/golden/`), the Phase 1 exit criterion.
7. **The calibration merge applies to every profile**, not only `default.yaml`
   (DMS-P16), and the stale `configs/vehicle.yaml` — computed from the *wrong*
   landmark topology — was emptied rather than carried forward.

## The camera-failure fix, demonstrated

Run with `source.type: csi` on a board with no camera, `retry_forever: true`:

```
CAMERA_FAIL unexpected EOS on a live source - reconnect attempt 1/5 in 0.5 s
...                                         - reconnect attempt 5/5 in 5.0 s
CAMERA_FAIL ... - 7 consecutive failures, camera=failed (still retrying)
```

```
GET /healthz -> 503   {"ok": false, "camera": "retrying", "degraded": true, ...}
GET /livez   -> 200
GET /readyz  -> 503
dms_camera_up 0
dms_alerts_enter_total{type="camera_fail",severity="critical"} 1
```

One `camera_fail` CRITICAL edge in `events.jsonl` (an edge, not a flood), the
process stays alive and self-heals, and `WATCHDOG=1` is withheld while no frames
arrive, so `WatchdogSec` escalates a permanently dead camera to a unit restart.

## Measured, this board, 2026-09-13

`configs/default.yaml`, 300 frames, all nine heads, overlay off, GPU governor at
its 114 MHz idle floor, board shared with other work:

```
e2e p50 58.6 ms   e2e p95 87.2 ms   16.2 FPS   peak RSS 1376 MiB
unaccounted 1.1 ms of 60.9 ms mean  (the stage table covers the loop)
stages p50: detect_gpu 17.0  headpose 13.4  landmarks 9.6  letterbox 9.3
            detect_post 3.6  fusion 0.6  track 0.4  capture 0.15  bgra 0.014
```

`capture_lag` p50 is 118 ms: on a file source the decoder runs ahead of the
consumer and frames sit in the appsink queue. That is expected on replay and is
**not** a live-capture latency figure.

Golden replay, nine clips, two consecutive runs: identical detection counts and
identical alert counts, all within band.

## The three biggest open items

1. **Latency.** 87 ms p95 against a 66 ms gate. The pipeline is dominated by the
   Python data path and TensorRT host overhead, not by the networks: `detect_gpu`
   is 17 ms of wall time around 3.7 ms of GPU compute. The fixes are known,
   quantified, and in files this workstream does not own.
2. **No cabin footage.** Every threshold in the fusion layer and every band in the
   golden fixtures is a starting point. Two driver-facing clips with labelled
   events (3 × eyes closed ≥ 2 s, 3 × yawns, 3 × over-shoulder looks) would turn
   the whole alert layer from plausible into measured.
3. **No calibration and no camera.** Until a `forward_zero` is measured in a real
   seat, gaze alerts are unreachable and every absolute head angle is relative.

## Reproduce in five minutes

```bash
cd /home/nvidia/myspace/DMS
PYTHONPATH=. ./.venv/bin/python -m pytest -q
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/testsrc.yaml --max-frames 30"
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python tests/replay/golden/run_golden.py"
```

## Known test failure

`tests/test_state.py::test_the_legacy_geometry_call_path_still_drives_the_machine`
fails. It feeds `ear=0.20` and expects `MICROSLEEP`; on the corrected EAR scale
0.20 is an **open** eye (open reads 0.21–0.40, closed below 0.15), so the machine
is right and the fixture is stale. The test's `mon.fusion_cfg.ear_closed = 0.50`
lines are a no-op — `SignalFuser` snapshots its thresholds into `SchmittLatch`
objects at construction — so the test only ever passed because the *config
default* used to be the broken-scale 0.50/0.75.

Verified fix (all 61 tests in that file pass with it), in a file this workstream
does not own:

```
tests/test_state.py:869  ear=0.20  ->  ear=0.10
tests/test_state.py:873  approx(0.20)  ->  approx(0.10)
```

## Local artifacts (not in git)

- `engines/*.engine`, `engines/*.onnx` — `bash scripts/fetch_onnx.sh && python3 scripts/build_engines.py`
- `tests/replay/*.mp4` — catalogue in [`../tests/replay/CLIPS.md`](../tests/replay/CLIPS.md)
- `tests/replay/golden/*.json` — **committed**; the regression baseline
- `data/events.jsonl` — lab alert log
