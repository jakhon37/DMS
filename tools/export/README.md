# Off-box ONNX (do not run on the Xavier NX)

UniFace is a **cookbook**: pin URLs and SHA-256 from
https://github.com/yakhyo/uniface (MIT library; model licenses vary).

Do **not** `pip install uniface` on JetPack 5.1.6 (Python 3.8, no ORT, no pip OpenCV).

## v1 face — SCRFD-500m (MIT)

- URL: https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx
- SHA256: `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`
- Decode: InsightFace 3-stride (8/16/32), 2 anchors, `(x-127.5)/127.5`

## v1 landmarks — 2d106det

- URL: https://github.com/yakhyo/uniface/releases/download/weights/2d106det.onnx
- SHA256: `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf`
- Topology: UniFace Landmark106 (see `dms/geometry/face106.py`)
- Crop: InsightFace 1.5× loose square affine to 192

## Phone (optional, AGPL)

Ultralytics YOLOv8n COCO 320. Only if product accepts AGPL (open question 10).

## On-device

```bash
bash scripts/fetch_onnx.sh
python3 scripts/build_engines.py   # trtexec FP16
```
