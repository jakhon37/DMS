# DMS production readiness — assessment and plan

| Field | Value |
| --- | --- |
| **Date** | 2026-09-09 |
| **Board** | Jetson Xavier NX Dev Kit, JetPack 5.1.6, TensorRT 8.5.2.2, 6.7 GiB RAM, `MODE_20W_6CORE` |
| **Architecture** | [`docs/production-dms-design.md`](production-dms-design.md) (rev 5) — still the v1 contract |
| **This document** | Gap analysis of the *running tree* vs that contract, and the work to reach a vehicle-engineering freeze |
| **Verdict** | **Not production-ready.** It is a working **replay lab** with a systemd *unit file*. Camera absence is one gap among several that would fail a vehicle image. |

**Production-level** here means the design v1 freeze, **not** UNECE R171 / Euro NCAP certification (explicit non-goal). Freeze gates from the design:

- 1280×720, e2e p95 **≤ 66 ms** (≥15 FPS) at 20 W 6-core
- Peak RSS **< 5.0 GB**, ≥1.5 GB free
- systemd notify + watchdog, health on localhost
- Alerts with hysteresis, local JSONL, optional clip quota
- Replay works with **no camera**; CSI is the vehicle source
- 8 h soak without watchdog fire

---

## 1. What is actually in the tree

### Works on this NX (measured)

| Piece | Evidence |
| --- | --- |
| GStreamer file/test capture, BGRx copy-before-unmap | `dms/capture/`; `nvv4l2decoder` + `nvvidconv` |
| SCRFD-500m FP16 GPU + 2d106det FP16 GPU | `engines/*.engine`; UniFace cookbook only (K17) |
| Driver pick in seat ROI, IoU tracks, occupant labels | `dms/track/`; crash clips need **right-side** ROI |
| Contour EAR / MAR / PnP yaw-pitch | `dms/geometry/`; open-eye EAR ≈ 0.75 on day_driver |
| Alert machines | FACE_LOST, MICROSLEEP, PERCLOS, YAWN, FATIGUE_CLUSTER, GAZE_AWAY |
| `--calibrate-forward` → `configs/vehicle.yaml` | Merged **only** for `default.yaml` |
| Batch replay | `scripts/run_all_replay.py`; run `tests/replay/runs/20260909-042432/` all 9 clips `rc=0` |
| Health HTTP | `GET 127.0.0.1:8088/healthz` and `/metrics` (stdlib) |
| `sd_notify` READY / WATCHDOG | `deploy/dms.service` (`Type=notify`, `TimeoutStartSec=90`) |
| GPIO latch | no-op unless `alerts.gpio_pin` + `buzzer` |

Python 3.8, TensorRT 8.5, no DeepFace / ORT / pip OpenCV — those constraints are held.

### What the last full replay showed

Processed **every frame** (file `drop=false`). Wall times: `day_driver` ~10 min for 3:27 of video; `red_car_crash` ~9 min for 1:52. Infer p95 typically **65–100 ms** (~6–10 FPS processed). That **misses** the 66 ms e2e / 15 FPS gate.

Uncalibrated 3/4 and profile faces fire `gaze_away` immediately (yaw tens of degrees vs 25° deadband; PnP pitch often folded from ±180). Driver ROI is **per YouTube framing**, not a vehicle seat.

---

## 2. Gap list (what to improve)

Severity: **Blocker** = cannot ship a vehicle image; **High** = false/missed alerts or ops failure; **Med** = design debt; **Low** = hygiene.

### 2.1 Runtime and packaging

| ID | Gap | Sev | Notes |
| --- | --- | --- | --- |
| G1 | systemd **not installed** on this board | Blocker | No `dms` user, no `/opt/dms`, unit not enabled. Files only. |
| G2 | Production JSONL is not on by default | Blocker | `--events` is a CLI flag. No `dms/io/events.py`. Nothing writes `/var/lib/dms/events.jsonl` unless asked. |
| G3 | Clip ring missing (PR-10) | High | `clips.enabled` is YAML-only. No `nvv4l2h264enc` ring, concat, quota, ENOSPC → `DISK_FULL`. |
| G4 | `MANIFEST.json` not enforced at load | High | `engines: {}`; no TRT-version refuse (`ENGINE_VERSION_MISMATCH`). |
| G5 | Leftover `models/` prototype tree | Med | TF/Haar/ORT/pycuda still in git (PR-11.5). Confuses operators and CI. |
| G6 | `production.yaml` EAR closed **0.21** | High | v1 EAR is **contour** (~0.50 closed / 0.75 open). Soukupová 0.21 would never trip microsleep — or would, wrongly, if mixed. |
| G7 | No `.venv` in the documented `--system-site-packages` sense for the service | Med | Lab used user-site + `python3.8`. Service expects `/opt/dms/.venv`. |
| G8 | Health unix socket `/run/dms/health.sock` not implemented | Low | TCP localhost is enough for v1. |
| G9 | `jtop` / 1 Hz tegrastats thread not used | Low | Sysfs temps + VmRSS only. |

### 2.2 Perception and alerts (quality)

| ID | Gap | Sev | Notes |
| --- | --- | --- | --- |
| A1 | **No camera** (CSI/USB/NIR) ever run | Blocker | Pipelines exist as strings. Argus, exposure lock, unplug/`CAMERA_FAIL` retry untested. |
| A2 | PnP unstable on profile / sunglasses / hat | High | Pitch wrap is a hack; crash clips show pitch 70–80° and constant `gaze_away`. |
| A3 | Gaze uses absolute PnP vs a single `vehicle.yaml` | High | Zeros from `day_driver` must not apply to minivan/crash (merge is default.yaml-only — good — but crash configs have **zero** forward_zero). |
| A4 | Driver ROI is per-clip YAML | High | Vehicle needs seat+camera extrinsics (`seat: lhd/rhd`), not nine YouTube framings. |
| A5 | EAR is contour, not 6-pt Soukupová | Med | Correct given UniFace slice order; 6-pt order still unverified against InsightFace diagram. |
| A6 | FACE_LOST uses last *present* sample, not `t_last_match` | Med | Design: unmatched coasting tracks must not suppress FACE_LOST. |
| A7 | Tracker is greedy IoU, not BYTE-lite (high then low score) | Low | Fine for one driver; IDs jump if detector flickers. |
| A8 | Phone net not built | High for “phone” goal | `models.objects.enabled: false`. COCO YOLOv8n is **AGPL** (design OQ-10). Do not enable silently. |
| A9 | Night / Y_mean / IR illuminator | High once camera exists | Contract only. Sunglasses + RGB will false MICROSLEEP. |
| A10 | `assume_moving: true` forever | Med | No CAN/GPIO speed. Mute-when-parked is dead. Acceptable if documented as fail-safe. |

### 2.3 Performance and reliability

| ID | Gap | Sev | Notes |
| --- | --- | --- | --- |
| P1 | Infer p95 **~80 ms** typical; e2e not even budgeted as a metric | Blocker | 15 FPS gate fails on current sequential SCRFD+106 every frame. |
| P2 | No 8 h soak | Blocker | Watchdog, RSS, NVDEC `OutputBufferUnavailable`, FD leaks unknown. |
| P3 | File replay vs live drop policy | Med | Live `drop=true`; file `drop=false`. Correct, but live 15 FPS still unmeasured. |
| P4 | GPU idle clock 115 MHz logs `CLOCKS_LOW` | Low | Real under load when TRT runs; do not treat idle as throttle. |
| P5 | numpy 1.23 vs distro OpenCV built against 1.17 | Med | R7 ABI. venv + ABI test exists in setup script; lab is mixed. |
| P6 | NVDEC backpressure warnings on file replay | Low | `max-buffers=4`; did not fail clips. |

### 2.4 Tests and ops

| ID | Gap | Sev | Notes |
| --- | --- | --- | --- |
| T1 | No golden alert counts on named mp4s | High | 37 CPU tests; no “this clip must not gaze_away for 60 s of forward driving”. |
| T2 | Replay mp4s gitignored | Med | CI cannot reproduce without a release tarball. |
| T3 | No latency_budget.json gate | High | Design PR-12. |
| T4 | readme still says “not a full drowsiness pipeline” | Low | Stale. |
| T5 | Top-level `main.py` + `config/` still the old template | Med | Conflicts with `python -m dms.app`. |

---

## 3. Target architecture (unchanged)

Do not redesign. Keep K1–K17:

- Python 3.8 + native TensorRT FP16 + GStreamer NVMM
- SCRFD-500m + 2d106det + PnP (no B1g2, no UniFace pip)
- Single 720p BGRx appsink
- systemd notify, localhost health, local-only privacy
- Phone net only after an explicit AGPL accept

Camera product recommendation remains NIR CSI + 850 nm IR; software already has `source.type: csi|usb|file|test`.

---

## 4. Implementation plan

Work is ordered so each slice is reviewable and leaves the board bootable. Do **not** start phone YOLO or soak until the previous phase’s acceptance box is ticked.

### Phase 0 — Honest vehicle image (packaging)

**Goal:** A systemd unit that starts on this NX with **file** source, writes alerts to disk, refuses bad engines, and does not contain dead TF trees.

| Work | Acceptance |
| --- | --- |
| PR-11.5 delete `models/face_*`, `models/headpose`, stub `main.py` / `config/` | `pytest` still green; `python -m dms.app --help` is the only entry |
| `dms/io/events.py`: always append JSONL (`/var/lib/dms/events.jsonl` in prod, `data/events.jsonl` in lab) | Restart preserves file; one line per alert edge |
| Load-time MANIFEST: TRT version + engine sha256 | Wrong TRT → exit 1 `ENGINE_VERSION_MISMATCH` before READY |
| Fix `production.yaml` `ear_closed: 0.50`, `ear_open_median: 0.75` | Matches contour EAR |
| `setup_jetson.sh` on this board with **file** `source.dev: true` until CSI exists | `systemctl start dms`; `curl 127.0.0.1:8088/healthz` → `ok`, `dev_replay: true` |
| Document CLOCKS_LOW only when TRT is loaded and clock < 300 | No spam on videotestsrc |

**Exit:** `systemctl status dms` active; JSONL growing on `testsrc` or looped mp4; journald JSON logs.

### Phase 1 — Alert quality on replay (trust)

**Goal:** Alerts mean something on the nine local clips, not just “boxes exist”.

| Work | Acceptance |
| --- | --- |
| Per-vehicle `forward_zero` file (not only `default.yaml` merge) | `configs/vehicles/<id>.yaml` or `vehicle.yaml` next to the active config |
| PnP: drop pose if `solvePnP` reprojection error high or landmarks off-face | Sunglasses/profile → `drowsiness_unreliable` / no GAZE_AWAY, not yaw=50° |
| FACE_LOST = `now - t_last_match` in ROI (design) | Occlusion 1.0 s enters; coasting id does not hide it |
| Golden tests: `tests/replay/golden/*.json` expected enter counts ±tolerance | `day_driver` forward look: **no** GAZE_AWAY; minivan: DRV is wheel-side |
| Optional: verify UniFace 6-pt eye order; if valid, Soukupová EAR + recalibrate | Document which EAR is live in YAML comments |

**Exit:** Golden suite passes on NX; overlay on `day_driver` / `minivan` / `red_car_crash` reviewed once.

### Phase 2 — Persistence (PR-10)

**Goal:** Disk is the recorder; privacy defaults hold.

| Work | Acceptance |
| --- | --- |
| JSONL rotation via existing `deploy/dms.logrotate` installed | size 50 MB |
| Clip ring **only if** `clips.enabled and privacy.record_faces` | Default both false → zero mp4s |
| `nvv4l2h264enc` + `splitmuxsink` 10 s ring; ffmpeg `-c copy` on warn/critical | Quota 2 GB / 24 h; ENOSPC → `DISK_FULL`, process stays up |
| `privacy.record_faces: false` refuses encode even if clips.enabled | Unit test |

**Exit:** Forced `DISK_FULL` (quota 1 MB in a test config) does not crash dms.

### Phase 3 — Camera

**Goal:** Live source, not YouTube.

| Work | Acceptance |
| --- | --- |
| Plug USB or CSI; lock `source.type` | 720p BGRx, `health.camera=ok`, READY < 90 s |
| Unplug: `CAMERA_FAIL`, retry, READY stays if `fail_fatal: false` | Watchdog does not kill the unit |
| Night: Y_mean on unpadded letterbox; conf 0.30 if Y<30; no PHONE from RGB | Fixture or dark-room clip |
| Operator `calibrate_forward` in the real seat | `vehicle.yaml` yaw/pitch/EAR from 5 s look-ahead |
| Seat ROI from `seat: lhd/rhd`, not playlist index | One production.yaml + vehicle.yaml |

**Exit:** 10 min live run, FACE_LOST only when driver leaves frame.

### Phase 4 — Speed and soak (PR-12 minus leftover-delete)

**Goal:** Hit the numeric freeze gates.

| Work | Acceptance |
| --- | --- |
| Instrument e2e (dequeue → alert) p50/p95; write `latency_budget.json` | Nightly on NX |
| If p95 > 66 ms: (1) skip overlay in prod, (2) landmarks only on DRV (already), (3) try DLA 2d106det if `[DLA]` audit passes, (4) process every 2nd frame **only** as a documented degrade, not a silent cheat | Gate is 15 FPS **effective samples** for PERCLOS (≥8 Hz required) |
| 8 h looped file or videotestsrc | RSS < 5.0 GB; no watchdog; no FD leak |
| Thermal: GPU ≥80 °C drops overlay/object net | Rehearse with `tegrastats` |

**Exit:** Checked-in `latency_budget.json` from this NX; soak log attached to the freeze commit.

### Phase 5 — Phone (optional, license-gated)

**Do not start** until someone accepts **AGPL-3.0** for Ultralytics YOLOv8n, or a MIT/Apache phone detector is pinned.

| Work | Acceptance |
| --- | --- |
| `dms/infer/objects.py`, 320², every N=2, cabin ROI | No overlap with face engine |
| Wire into existing PHONE latch | 0.8 s enter / 0.4 s exit |
| AGPL notice in MANIFEST + readme | Engine not fetched by default |

If AGPL is rejected: leave `objects.enabled: false` and drop “phone” from v1 goals.

### Phase 6 — Freeze

Tag `v1.0-nx`. Include: engines + MANIFEST hashes, production.yaml, soak numbers, golden replay, known limitations (no cert, PnP not a gaze net, RGB night weak).

---

## 5. Suggested sequence (calendar)

Assuming one engineer on this NX, camera hardware in week 3:

| Week | Phase | Outcome |
| --- | --- | --- |
| 1 | 0 | Headless `dms.service` on file source, JSONL, dead code gone |
| 1–2 | 1 | Golden alerts; PnP/gaze not lying on `day_driver` |
| 2 | 2 | Clip ring behind privacy flags |
| 3 | 3 | First live camera + seat calibration |
| 3–4 | 4 | 15 FPS or documented degrade; 8 h soak |
| later | 5 | Phone only with license decision |
| 4 | 6 | Freeze tag |

Parallelism that is safe: Phase 0 leftover-delete ∥ JSONL; Phase 2 ∥ Phase 1 tests; Phase 5 never blocks freeze if phone is dropped from v1.

---

## 6. What not to do

- Do not `pip install uniface`, `opencv-python`, TensorFlow, Torch, ORT on the NX.
- Do not install DeepStream to “make it production”.
- Do not treat YouTube crash ROI YAMLs as vehicle calibration.
- Do not enable clips with `privacy.record_faces: false`.
- Do not call `nvpmodel` / `jetson_clocks` from the app (setup script only, `--apply-power`).
- Do not claim UNECE / Euro NCAP.
- Do not start 8 h soak while `gaze_away` fires on every 3/4 face — it will only record junk.

---

## 7. Immediate next slice

If the next command is “go”, start **Phase 0** in this order:

1. Delete leftover `models/` + stub `main.py` / `config/`
2. `dms/io/events.py` + production JSONL path
3. MANIFEST check + `production.yaml` EAR thresholds
4. `sudo bash deploy/setup_jetson.sh` with file-source `/etc/dms/default.yaml` (no `--apply-power` unless you want 20 W lock)

Camera work waits on hardware. Alert-quality (Phase 1) can start on the nine local mp4s the same week as Phase 0.
