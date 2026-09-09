# DMS — Driver Monitoring System (Jetson Xavier NX)

Production-oriented DMS for **Jetson Xavier NX / JetPack 5.1.6 / TensorRT 8.5**.

- Architecture: [`docs/production-dms-design.md`](docs/production-dms-design.md)
- Status vs a vehicle freeze: [`docs/production-readiness.md`](docs/production-readiness.md) — **not production-ready yet**. Phase 0: packaging, JSONL, MANIFEST, leftover delete.

## This board

- Python 3.8.10, CUDA 11.4, TensorRT 8.5.2.2, OpenCV 4.5.4 **with GStreamer**, no DeepStream, no camera.
- Do **not** `pip install opencv-python`, `tensorflow`, `onnxruntime`, or `uniface`.

## Setup

```bash
python3.8 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
# optional: pip install 'cuda-python>=11.4,<12' || true   # ctypes cudart is the fallback
```

```bash
python -m dms.app --help
python -m dms.app --config configs/default.yaml --max-frames 30
# if no mp4 yet, use the test source:
#   edit source.type: test   or
python -c "from dms.config.schema import load_config; c=load_config('configs/default.yaml'); c.source.type='test'"
```

Fetch models (network):

```bash
bash scripts/fetch_onnx.sh
python scripts/build_engines.py
```

Tests (CPU, no TRT required):

```bash
PYTHONPATH=. pytest -q
```

## systemd (headless)

`READY=1` is sent after engines load even if CSI is missing (`camera.fail_fatal: false`). Health: `http://127.0.0.1:8088/healthz`.

```bash
sudo bash deploy/setup_jetson.sh          # does not change nvpmodel unless --apply-power
# sudo bash deploy/setup_jetson.sh --apply-power
sudo systemctl enable --now dms.service
curl -s http://127.0.0.1:8088/healthz
```

Unit: `deploy/dms.service` (`Type=notify`, `TimeoutStartSec=90`, `WatchdogSec=30`, `User=dms`). Do not call `nvpmodel` / `jetson_clocks` from the app.

## UniFace

[yakhyo/uniface](https://github.com/yakhyo/uniface) is the **ONNX + 106-pt index cookbook**. Runtime is TensorRT, not UniFace/ORT. Face detector is **SCRFD-500m** (MIT), not YOLOv8-face (GPL).
