# DMS: Driver Monitoring on Jetson Xavier NX

In-cabin drowsiness and distraction pipeline on a Jetson Xavier NX (JetPack 5.1.6, TensorRT 8.5, Python 3.8). It runs as a replay lab on recorded clips, because no camera is attached to the board.

**Not production-ready. Development paused on 2026-09-09.** Snapshot: [`docs/STATUS.md`](docs/STATUS.md). Architecture: [`docs/production-dms-design.md`](docs/production-dms-design.md). Gap plan: [`docs/production-readiness.md`](docs/production-readiness.md).

## Status

| Working in the replay lab | Not done yet |
| --- | --- |
| File and test-source GStreamer input at 720p | CSI or USB camera |
| SCRFD-500m detector and 2d106det landmarks in TensorRT FP16 | 15 FPS / 66 ms end-to-end target (inference is about 80 ms) |
| EAR, MAR and PnP head pose, IoU tracks, driver vs passenger | Trusted gaze (PnP is noisy on uncalibrated clips) |
| Alert state machines and JSONL events | Clip ring, phone detection, 8-hour soak |
| systemd unit file and `/healthz` | Service enabled on the board |
| Nine local clips with per-clip ROI config | Clips in CI |

## Layout

```
dms/                  runtime (python -m dms.app)
configs/              default, production, systemd-lab, per-clip replay
deploy/               dms.service, setup_jetson.sh, logrotate
engines/              ONNX + TRT (gitignored except MANIFEST.json)
scripts/              fetch_onnx, build_engines, run_all_replay, calibrate_forward
tests/                pytest (CPU) + tests/replay/ (mp4 gitignored)
docs/                 design, readiness, status
tools/export/         off-box ONNX notes
```

Entry point is **`python3.8 -m dms.app`**. The old `main.py` / `models/` / `config/` stubs were deleted in Phase 0.

## Setup

```bash
python3.8 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
# optional: pip install 'cuda-python>=11.4,<12' || true
```

System site-packages must provide `cv2` (GStreamer build), `tensorrt`, `gi`.

```bash
bash scripts/fetch_onnx.sh
PYTHONPATH=. python3.8 scripts/build_engines.py
PYTHONPATH=. python3.8 -m pytest -q
PYTHONPATH=. python3.8 -m dms.app --config configs/testsrc.yaml --max-frames 30
PYTHONPATH=. python3.8 -m dms.app --config configs/default.yaml --detect --max-frames 60
```

Health while running: `http://127.0.0.1:8088/healthz`. Lab alerts append to `data/events.jsonl`.

## Constraints on this board

- Do not `pip install uniface`, `opencv-python`, TensorFlow, Torch, or ONNX Runtime.
- Do not call `nvpmodel` / `jetson_clocks` from the app.
- Do not treat YouTube ROI yaml as vehicle calibration.

## Replay

Clips and ROI notes: [`tests/replay/CLIPS.md`](tests/replay/CLIPS.md). mp4 is gitignored.

```bash
PYTHONPATH=. python3.8 scripts/run_all_replay.py              # until EOS
PYTHONPATH=. python3.8 scripts/run_all_replay.py --max-frames 60
# resume a folder, skip finished, wait if RAM is tight:
PYTHONPATH=. python3.8 scripts/run_all_replay.py \
  --run-dir tests/replay/runs/<stamp> --skip-complete --min-avail-mb 1800
```

Forward-look calibration (writes gitignored `configs/vehicle.yaml`, merged only into `default.yaml`):

```bash
PYTHONPATH=. python3.8 scripts/calibrate_forward.py --config configs/default.yaml
```

## systemd (not enabled on this board)

```bash
sudo bash deploy/setup_jetson.sh --lab       # videotestsrc until CSI exists
# sudo bash deploy/setup_jetson.sh --vehicle # configs/production.yaml (csi)
sudo systemctl enable --now dms.service
curl -s http://127.0.0.1:8088/healthz
```

`--apply-power` is the only way the script runs `nvpmodel -m 8`. `jetson_clocks` is never automatic.

## Config map

| File | Use |
| --- | --- |
| `configs/default.yaml` | Lab file replay (`day_driver.mp4`), `source.dev: true` |
| `configs/testsrc.yaml` | `videotestsrc`, no TRT unless `--detect` |
| `configs/systemd-lab.yaml` | Headless unit until CSI; `require_engines: true` |
| `configs/production.yaml` | Vehicle CSI; contour EAR 0.50 / 0.75 |
| `configs/<clip>.yaml` | Per-video driver ROI (see CLIPS.md) |

## Models

UniFace is the **ONNX + 106-pt cookbook**, never a runtime. Detector: **SCRFD-500m** (MIT). Landmarks: **2d106det**. Hashes in `engines/MANIFEST.json` are checked at load (wrong TRT version or engine sha256 → exit 1). Phone YOLO is off (AGPL). Details: [`tools/export/README.md`](tools/export/README.md).
