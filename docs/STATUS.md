# Pause snapshot — 2026-09-09

Development is paused after **Phase 0**. Resume from [`production-readiness.md`](production-readiness.md) **Phase 1** (alert quality / golden replay). Do not start phone YOLO or soak yet.

## Verdict

Working **replay lab** on this Xavier NX. **Not** a vehicle image.

| In | Out |
| --- | --- |
| File/test GStreamer 720p BGRx | CSI/USB never attached |
| SCRFD-500m + 2d106det TensorRT FP16 | 15 FPS / 66 ms e2e gate (infer ~80 ms) |
| EAR/MAR/PnP, IoU tracks, DRV vs Pax | Trusted gaze (PnP noisy; uncalibrated clips) |
| Alert machines + JSONL | Clip ring, phone net, 8 h soak |
| systemd **unit file** + `/healthz` | `systemctl enable` (needs `sudo` password on this board) |
| Nine local mp4s + per-clip ROI yaml | mp4s gitignored; not in CI |

## Last useful commits

| Hash | What |
| --- | --- |
| `c7715f4` | Phase 0: delete prototypes, JSONL, MANIFEST, systemd-lab |
| `bd418ac` | Production-readiness assessment |
| `95f9482` | Health HTTP + systemd unit + GPIO stub |
| `e496c0e` | Batch replay runner |

## Local artifacts (not in git)

- `engines/*.engine` and `engines/*.onnx` — rebuild with `bash scripts/fetch_onnx.sh && python3.8 scripts/build_engines.py`
- `tests/replay/*.mp4` — catalog in [`tests/replay/CLIPS.md`](../tests/replay/CLIPS.md)
- `tests/replay/runs/` — last full batch `20260909-042432/`
- `configs/vehicle.yaml` — day_driver forward zeros, gitignored
- `data/events.jsonl` — lab JSONL

## Resume in five minutes

```bash
cd /home/nvidia/myspace/DMS
PYTHONPATH=. python3.8 -m pytest -q
PYTHONPATH=. python3.8 -m dms.app --config configs/testsrc.yaml --max-frames 30
# engines still on disk:
PYTHONPATH=. python3.8 -m dms.app --config configs/default.yaml --detect --max-frames 30
```

Next product slice: **Phase 1** in the readiness doc (golden alerts, PnP reject, per-vehicle zeros). Optional ops: `sudo bash deploy/setup_jetson.sh --lab`.

## Constraints that still bind

Python 3.8 + TensorRT 8.5 + GStreamer OpenCV. No `pip install uniface`, `opencv-python`, TensorFlow, Torch, ORT. UniFace is cookbook only. No DeepStream.
