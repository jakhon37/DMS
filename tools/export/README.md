# ONNX model cards — DMS engine supply chain

Everything the DMS pipeline runs arrives here as a **prebuilt `.onnx`** and is turned into a
TensorRT FP16 `.engine` **on the board**. The Xavier NX image forbids `torch`, `onnx`,
`onnxruntime`, `pycuda`, `opencv-python` and `ultralytics`, so nothing in this directory is a
runtime dependency — it is the pinned recipe.

```bash
bash scripts/fetch_onnx.sh                                   # download + verify pinned sha256
flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python scripts/build_engines.py --skip-existing --keep-going"
flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python scripts/build_engines.py --verify"   # ~11 s, all 13
```

Other agents share this GPU: **every** trtexec or TensorRT command must be wrapped in
`flock /tmp/jetson-gpu.lock -c "…"`. Prefer one flock per engine (`--only <id>`) over one
flock around the whole run, so a two-hour build sweep does not lock everyone else out.

`scripts/build_engines.py` is driven by `engines/MANIFEST.json` → `builds[]`. Every engine is
FP16 and **fully static**: `dms/runtime/trt_engine.py` raises on any binding with a `-1`
dimension, so each build spec names the exact shape to freeze and the builder re-opens the
engine afterwards to prove no dynamic axis survived. If trtexec leaves a profiled (still
dynamic) engine, the builder rebuilds it through the TensorRT Python API with the parsed
network input shape overwritten to a constant.

A model that will not build is **never dropped silently** — the trtexec error tail lands in
`MANIFEST["blocked"][<id>]` and the wrapper workstream ships an honest stub
(`source=UNAVAILABLE` / `is_mock=True`) for that capability instead.

The authoritative, machine-readable copy of everything below — pinned hashes, binding names,
shapes, dtypes, colour order, normalisation, output semantics, measured latency — is
`engines/MANIFEST.json`. This file is the human-readable companion.

## Licence summary

| Model | Licence | Ship in a commercial vehicle image? |
| --- | --- | --- |
| SCRFD-500m-kps | MIT | yes, with attribution |
| 2d106det | InsightFace model terms | **re-read before shipping** — non-commercial leaning |
| head pose (mobilenetv2 / mobilenetv3_small) | MIT | yes |
| OCEC / VSDLM / SGC / MWC | MIT | yes |
| gaze (mobilenetv2_gaze) | MIT code, **Gaze360 weights** | **no** — Gaze360 is research-use only |
| YOLOX-Nano | Apache-2.0 | yes — this is why the AGPL Ultralytics plan was dropped |
| SFace | Apache-2.0 | yes |

## Preprocessing at a glance

Colour order and normalisation are per-model and are **not** interchangeable. `cv2` decodes
BGR; every RGB model below needs an explicit `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`.

| Model | Input | Colour | Formula |
| --- | --- | --- | --- |
| SCRFD-500m | `1x3x640x640` / `1x3x512x512` | RGB | `(px - 127.5) / 128.0` |
| 2d106det | `1x3x192x192` | RGB | raw 0–255 |
| head pose | `1x3x224x224` | RGB | `(px/255 - imagenet_mean) / imagenet_std` |
| gaze | `1x3x448x448` | RGB | `(px/255 - imagenet_mean) / imagenet_std` |
| OCEC / VSDLM / SGC / MWC | see below | RGB | `px / 255` |
| YOLOX-Nano | `1x3x416x416` | **BGR** | raw 0–255, letterbox pad **114** |
| SFace | `1x3x112x112` | RGB | raw 0–255 |

`imagenet_mean = [0.485, 0.456, 0.406]`, `imagenet_std = [0.229, 0.224, 0.225]`.

## Measured on this board (TRT 8.5.2.2, FP16, GPU)

`trtexec --loadEngine`, 100 iterations after a 200 ms warm-up, on an otherwise idle GPU with
the GPU mutex held. These are **build-time sanity numbers**, not the in-pipeline budget —
nothing here accounts for preprocessing, H2D/D2H, or contention with the other engines.
`ctx MB` is the TensorRT execution-context scratch (`device_memory_size`), which is *on top of*
the deserialised weights.

| engine | role | file MB | ctx MB | median ms | p99 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `sface_112_fp16_gpu` | driver identity | 19.70 | 3.21 | 2.22 | 2.36 |
| `mobilenetv2_gaze_448_fp16_gpu` | gaze | 5.85 | 16.86 | 4.11 | 4.22 |
| `hpe_mobilenetv2_224_fp16_gpu` | head pose | 4.92 | 3.51 | 1.49 | 1.53 |
| `hpe_mobilenetv3_small_224_fp16_gpu` | head pose (fallback) | 4.89 | 1.68 | 2.45 | 9.39 |
| `2d106det_192_fp16_gpu` | landmarks | 3.78 | 1.18 | 1.13 | 1.54 |
| `yolox_nano_416_fp16_gpu` | cabin objects | 3.15 | 5.28 | 4.90 | 4.97 |
| `ocec_s_24x40_b2_fp16_gpu` | both eyes, batch 2 | 1.82 | 0.52 | 0.76 | 1.05 |
| `sgc_is_s_48_fp16_gpu` | sunglasses | 1.81 | 0.63 | 0.70 | 0.93 |
| `scrfd_500m_640_fp16_gpu` | face @640 | 1.77 | 7.30 | 3.67 | 3.77 |
| `scrfd_500m_512_fp16_gpu` | face @512 | 1.73 | 6.77 | 2.48 | 2.56 |
| `mwc_n_48_fp16_gpu` | face mask | 1.47 | 0.63 | 0.54 | 0.66 |
| `vsdlm_s_30x48_fp16_gpu` | mouth state | 1.27 | 0.39 | 0.84 | 0.97 |
| `ocec_s_24x40_fp16_gpu` | one eye | 1.24 | 0.26 | 0.76 | 0.90 |
| **all 13** | | **53.40** | **48.22** | | |

Three things worth reading off that table:

- **SCRFD 512 buys 1.19 ms** over 640 (2.48 vs 3.67 median) for a 21 % smaller input. That is
  the accuracy/latency lever the perf workstream asked for.
- **The batch-2 eye engine is free.** Two eyes in one launch costs 0.76 ms — the same as one
  eye — so always use `ocec_s_24x40_b2` when both eyes are visible.
- **`hpe_mobilenetv3_small` is not the cheaper option.** It is *slower* than mobilenetv2
  (2.45 vs 1.49 ms median, and a 9.39 ms p99 tail) because depthwise+SE blocks map poorly onto
  Xavier's tensor cores. Ship `hpe_mobilenetv2`; the small variant is kept only as a
  memory-pressure fallback (1.68 MB context vs 3.51 MB).

A realistic **per-frame** DMS set — SCRFD 512 + 2d106det + head pose mnv2 + eye pair + mouth —
is `2.48 + 1.13 + 1.49 + 0.76 + 0.84 = 6.70 ms` of pure GPU compute, `13.35 MB` of file-backed
weights and `13.04 MB` of context scratch. The occlusion gates (SGC/MWC), YOLOX and SFace do
not run every frame: SGC/MWC on a change of state, YOLOX at `objects.every_n`, SFace once per
new track id. Loading every engine at once is 53.40 MB of weights + 48.22 MB of scratch, but
the pipeline never needs that — the two head-pose variants and the two SCRFD sizes are
alternatives, not a set.

---

## Face detection — SCRFD-500m-kps (MIT)

- URL: <https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx>
- SHA256: `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`
- ONNX input is dynamic H×W (`1x3x-1x-1`), so the size is a **build-time** decision. Two
  engines are shipped: `scrfd_500m_640_fp16_gpu.engine` (accuracy) and
  `scrfd_500m_512_fp16_gpu.engine` (latency).
- Input `input.1`, RGB NCHW, `(px - 127.5) / 128.0`, UniFace letterbox (fit inside, **top-left
  pad 0** — not YOLO's centred pad-114).
- 9 outputs, 3 strides (8/16/32) × {score `(N,1)`, bbox `(N,4)`, kps `(N,10)`}, 2 anchors per
  cell. bbox/kps are distances in stride units. Decode lives in `dms/infer/scrfd.py`.
- `N` scales with the build size, so the engine tells you which grid it wants:
  640 → `12800 / 3200 / 800`, 512 → `8192 / 2048 / 512` (i.e. `2·(size/stride)²`).
  Output tensor names are the raw ONNX node ids: scores `443 / 468 / 493`,
  boxes `446 / 471 / 496`, kps `449 / 474 / 499` — the same names at both sizes.

> **Using the 512 engine needs a one-line wrapper change.** `ScrfdDetector` hardcodes
> `input_size = (640, 640)` and builds its anchor grids with `_anchor_centers(640, 640, …)`.
> Both must come from `engine.input_shape[2:]`, or the 512 engine decodes against the wrong
> grid and silently mislocates every face.

## Landmarks — 2d106det (InsightFace model terms)

- URL: <https://github.com/yakhyo/uniface/releases/download/weights/2d106det.onnx>
- SHA256: `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf`
- Input `data`, RGB NCHW `1x3x192x192`, raw 0–255, InsightFace 1.5× loose square affine crop.
- Output `fc1` `(1, 212)` = 106 `(x, y)` pairs in `[-1, 1]` crop space → `pt = (v + 1) * 96`.
- Topology: `dms/geometry/face106.py`.

## Head pose — yakhyo/head-pose-estimation (MIT, 6DRepNet-derived)

Replaces `solvePnP`, which readiness A2 names as the top false-alert source (the ad-hoc
`PNP_OBJECT_POINTS` head model biases pitch, and the ±180° pitch unwrap is a hack).

- mobilenetv2: <https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv2.onnx>
  — SHA256 `1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9`
- mobilenetv3_small (cheaper fallback):
  <https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx>
  — SHA256 `e8ae4d932b3d13221638fc72e171603e020c6da28b770753f76146867f40e190`
- Input `input`, RGB NCHW `1x3x224x224`, ImageNet normalisation. Crop = face bbox expanded by
  `expand_bbox(factor=0.2)`: `x ± 0.2*height`, `y ± 0.2*width` (upstream really does swap the
  two — keep the quirk, the weights were trained with it), clamped at 0, then resized to 224.
- Output `rotation_matrix` `(1, 3, 3)`. The 6D→R orthonormalisation is **already inside the
  graph**; the output is a true rotation matrix. Verified on-board: mobilenetv2
  `|RRᵀ − I|max = 5.2e-8`, `det = 1.0000000`; mobilenetv3_small `|RRᵀ − I|max = 8.5e-4`,
  `det = 0.99907`. No re-orthonormalisation is needed in the wrapper.
- Euler in **radians**, upstream `compute_euler_angles_from_rotation_matrices`:

  ```
  sy = sqrt(R[0,0]**2 + R[1,0]**2)
  if sy >= 1e-6:  pitch = atan2(R[2,1], R[2,2]); yaw = atan2(-R[2,0], sy); roll = atan2(R[1,0], R[0,0])
  else:           pitch = atan2(-R[1,2], R[1,1]); yaw = atan2(-R[2,0], sy); roll = 0.0
  ```

  Upstream reports `euler[:,0]` as **pitch**, `euler[:,1]` as **yaw**, `euler[:,2]` as **roll**.
  There is no wrap-around branch to write: a rotation matrix cannot fold at ±180°.

## Eye open/closed — PINTO0309/OCEC (MIT)

- URL: <https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx>
- SHA256: `9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64`
- Input `images`, RGB NCHW, **24 high × 40 wide** (`cv2.resize(crop, (40, 24), INTER_LINEAR)`,
  aspect intentionally not preserved), `px / 255`, no mean/std. Tight eye box, **no margin**.
- Output `prob_open` — already a sigmoid probability that the eye is **open**. Upstream
  decision threshold `0.50`.
- Two engines: batch-1 and a **batch-2** variant (`ocec_s_24x40_b2_fp16_gpu.engine`) so both
  eyes go in one launch — index 0 left eye, index 1 right eye.

## Mouth open/closed — PINTO0309/VSDLM (MIT)

- URL: <https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_s.onnx>
- SHA256: `47e89d9e919ad83def655d7caddfb54e0b9b6031b91528173a5e0be363484eeb`
- Input `images`, RGB NCHW, **30 high × 48 wide**, `px / 255`. Mouth box with pixel margins
  top 2, bottom 6, left 2, right 2.
- Output `prob_open` — sigmoid probability the mouth is open. Threshold `0.50`.

## Eye occlusion (sunglasses) — PINTO0309/SGC (MIT)

- URL: <https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_is_s_48x48.onnx>
- SHA256: `005daa03aaa0f7120d94300e1115633cd886f7846e4321af11854bfd880873da`
- Input `images`, RGB NCHW `48x48`, `px / 255`. Crop is the **head** box, not the eye box.
- Output `prob_sunglasses` — sigmoid. Upstream `SUNGLASSES_SCORE_THRESHOLD = 0.50`.
- This is the gate that must set `Occlusion.eyes` and force
  `AlertType.DROWSINESS_UNRELIABLE` instead of letting EAR/PERCLOS free-run on dark lenses.

## Mouth occlusion (face mask) — PINTO0309/MWC (MIT)

- URL: <https://github.com/PINTO0309/MWC/releases/download/onnx/mwc_n_48x48.onnx>
- SHA256: `c2e83bc6fa0fdade463d9e8c78a9ccfa74a678f86631b4c8ee1fe2663d7f291d`
- Input `images`, RGB NCHW `48x48`, `px / 255`. Crop is the **head** box.
- Output `prob_masked` — sigmoid. Threshold `0.50`.

## Gaze — yakhyo/gaze-estimation (MIT code, **Gaze360 weights: research use only**)

- URL: <https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx>
- SHA256: `b81312df85c7ac1c1b5f78c573620d22c2719cb839650e15f12dc7eecb7744a4`
- Input `input`, RGB NCHW `1x3x448x448`, ImageNet normalisation. Crop = the **raw** detector
  face bbox, no expansion.
- Outputs `yaw` and `pitch`, both `(1, 90)` **logits** — there is no softmax in the graph
  (verified on-board: a probe row summed to −160.4, not 1.0).

  ```
  deg = (softmax(logits) * arange(90)).sum() * 4 - 180      # bins=90, binwidth=4, offset=180
  rad = radians(deg)
  ```

  `yaw` is horizontal, `pitch` vertical, both relative to the face crop.
- **Licence gate.** The weights are trained on Gaze360, whose dataset licence is
  non-commercial research use only. The engine is built and pinned so the pipeline can be
  developed and evaluated today, but a production build must report `source=UNAVAILABLE`
  until the model is retrained on a commercially licensed corpus. Recorded as
  `onnx.gaze.license_caveat` in `MANIFEST.json`.

## In-cabin objects / phone — YOLOX-Nano (Apache-2.0)

This is the Apache-2.0 replacement for the AGPL-3.0 Ultralytics YOLOv8n plan. No licence
acceptance is required, so the phone-distraction path is unblocked.

- URL: <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx>
- SHA256: `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d`
- Input `images`, **BGR** NCHW `1x3x416x416`, **raw 0–255** (no `/255`, no mean/std).
  Letterbox: `r = min(416/h, 416/w)`, resize into a canvas prefilled with **114**, anchored
  **top-left** (YOLOX does not centre the pad).
- Output `output` `(1, 3549, 85)`. `3549 = 52² + 26² + 13²`, strides 8/16/32 concatenated in
  that order. Per row `[dx, dy, dw, dh, obj, cls_0..cls_79]`.
  - `obj` and `cls` are **already sigmoid-activated in the graph** — verified on-board, the
    probe saw `[1.2e-7, 0.604]`. Do **not** sigmoid them again.
  - The grid decode is **not** baked in — verified on-board, raw `xy` spanned `[-1.64, 2.51]`
    rather than 0–416:
    `cx = (dx + grid_x) * stride`, `cy = (dy + grid_y) * stride`,
    `w = exp(dw) * stride`, `h = exp(dh) * stride`, in 416-space.
  - `score = obj * cls_i`; convert to xyxy, divide by `r`, class-wise NMS
    (upstream `nms_thr=0.45`, `score_thr=0.1`).
- COCO-80 class order. `person` 0, `bottle` 39, `cup` 41, **`cell phone` 67**.

## Driver identity — SFace (Apache-2.0)

- URL: <https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx>
- SHA256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- Input `data`, RGB NCHW `1x3x112x112`, **raw 0–255**. The crop is a 5-point similarity
  transform (`alignCrop`) onto the ArcFace reference landmarks, in SCRFD's own kps order —
  left eye, right eye, nose, left mouth corner, right mouth corner:

  ```
  [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
   [41.5493, 92.3655], [70.7299, 92.2041]]
  ```

  Reference: `blobFromImage(aligned, 1, Size(112,112), Scalar(0,0,0), swapRB=true, crop=false)`
  in OpenCV `modules/objdetect/src/face_recognize.cpp`.
- Output `fc1` `(1, 128)` — **not** L2-normalised by the graph (verified on-board: a probe
  embedding had `‖v‖₂ = 4.52`); normalise before comparing. OpenCV thresholds: cosine
  `>= 0.363` or L2 on normalised vectors `<= 1.128` → same identity.
- Run once per new track id, not per frame; it feeds `AlertType.DRIVER_CHANGED` and makes
  `dms/track/driver_select.py` more than pure geometry.

---

## Blocked models

None. `MANIFEST["blocked"]` is empty — all 13 engines build, deserialise, and load through
`dms/runtime/trt_engine.py`. If a future model lands there, the wrapper for that capability
must ship a stub that reports `source=UNAVAILABLE` / `is_mock=True`, never a plausible constant.

## How the contracts above were verified

Claims about output semantics are cheap to get wrong and expensive to debug, so each one was
checked by running the built engine on this board through the project's own loader:

| claim | check | result |
| --- | --- | --- |
| head pose emits a real rotation matrix | `\|RRᵀ − I\|max`, `det R` | `5.2e-8`, `1.0000000` (mnv2) |
| gaze outputs are logits, not probabilities | row sum of `yaw` | `−160.4` → logits |
| the four PINTO heads emit probabilities | output range | all inside `[0, 1]` |
| YOLOX grid decode is not baked in | range of `output[:, :2]` | `[-1.64, 2.51]` → raw offsets |
| YOLOX obj/cls are sigmoid-activated | range of `output[:, 4:]` | `[1.2e-7, 0.604]` → activated |
| SFace embeddings are not normalised | `‖fc1‖₂` | `4.52` → caller must normalise |
| every engine is static | load through `TrtEngine` | all 13 accepted |

## Two TensorRT 8.5 gotchas this repo has already hit

1. **`trtexec` refuses `--minShapes` on a fully static ONNX** — "Static model does not take
   explicit shapes since the shape of inference tensors will be determined by the model
   itself". So the builder parses the graph first and only emits shape flags for inputs that
   actually carry a `-1`. A static input whose declared shape disagrees with the build spec is
   reported as a configuration error instead of being built at the wrong size.
2. **A dynamic ONNX built with no shape flags silently freezes every dynamic dim to 1.** That
   is how the SCRFD engine once ended up baked at `1x3x1x1`. Always name the shape in
   `builds[]`, and never trust an engine you have not re-opened and inspected.

## Adding a model

1. Add `name|url|sha256` to `MODELS` in `scripts/fetch_onnx.sh` — **download first, hash the
   real bytes, then pin**. Never write a hash you did not compute.
2. Add an `onnx.<role>` record to `engines/MANIFEST.json` with the licence, the binding
   contract and the preprocessing/postprocessing contract. Read the upstream demo code for
   the normalisation; do not guess it.
3. Add a `builds[]` entry naming the frozen input shape.
4. `flock /tmp/jetson-gpu.lock -c "PYTHONPATH=. ./.venv/bin/python scripts/build_engines.py --only <id>"`.
5. `… scripts/build_engines.py --verify` — this loads the engine through
   `dms/runtime/trt_engine.py`, not just trtexec, and re-asserts the recorded output
   semantics. A trtexec engine can still carry a dynamic axis that the runtime refuses, and a
   re-exported upstream ONNX can quietly gain or lose a softmax.

`--measure-only` re-reads the binding contract and latency of an engine already on disk
without rebuilding it — use it for engines whose sha256 is already pinned and depended on.

To inspect an ONNX without the `onnx` package (which is forbidden on this board), parse it
with `tensorrt.OnnxParser` and walk `network.get_input(i)` / `network.get_output(i)`, or run
`/usr/src/tensorrt/bin/trtexec --onnx=… --verbose`.
