# DMS — Driver Monitoring System (Jetson Xavier NX)

Production-oriented DMS for **Jetson Xavier NX / JetPack 5.1.6 / TensorRT 8.5**.
Design: [`docs/production-dms-design.md`](docs/production-dms-design.md).

Phase 1 (this tree): package, YAML config, GStreamer **file/test** capture, TensorRT wrapper, UniFace-pinned ONNX hashes. Not a full drowsiness pipeline yet.

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

## UniFace

[yakhyo/uniface](https://github.com/yakhyo/uniface) is the **ONNX + 106-pt index cookbook**. Runtime is TensorRT, not UniFace/ORT. Face detector is **SCRFD-500m** (MIT), not YOLOv8-face (GPL).
