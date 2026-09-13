#!/usr/bin/env bash
# Download every ONNX the DMS pipeline builds engines from, verifying a pinned SHA-256.
#
# Nothing here installs a Python package: the board forbids torch/onnx/onnxruntime, so the
# models arrive as prebuilt .onnx and are turned into TensorRT engines on-board by
# scripts/build_engines.py.
#
# Every hash below was produced by downloading the asset on this Xavier NX and running
# sha256sum. A mismatch aborts the whole script with exit 1 - a changed release asset is a
# supply-chain event, not something to work around.
#
# Usage:
#   bash scripts/fetch_onnx.sh            # fetch everything missing / re-verify what is there
#   bash scripts/fetch_onnx.sh ocec_s     # fetch only entries whose filename matches a pattern
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/engines"
MANIFEST="${DEST}/MANIFEST.json"
mkdir -p "${DEST}"

# name|url|sha256  (name is the file written under engines/)
MODELS=(
  # --- v1 face detection + 106-pt landmarks (UniFace-pinned) -------------------------------
  "scrfd_500m_kps.onnx|https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx|5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a"
  "2d106det.onnx|https://github.com/yakhyo/uniface/releases/download/weights/2d106det.onnx|f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"
  # --- head pose (6DRepNet-style rotation-matrix regressor, replaces solvePnP) --------------
  "hpe_mobilenetv2.onnx|https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv2.onnx|1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9"
  "hpe_mobilenetv3_small.onnx|https://github.com/yakhyo/head-pose-estimation/releases/download/weights/mobilenetv3_small.onnx|e8ae4d932b3d13221638fc72e171603e020c6da28b770753f76146867f40e190"
  # --- per-eye / per-mouth open-closed classifiers and occlusion gates ----------------------
  "ocec_s.onnx|https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx|9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64"
  "vsdlm_s.onnx|https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_s.onnx|47e89d9e919ad83def655d7caddfb54e0b9b6031b91528173a5e0be363484eeb"
  "sgc_is_s_48x48.onnx|https://github.com/PINTO0309/SGC/releases/download/onnx/sgc_is_s_48x48.onnx|005daa03aaa0f7120d94300e1115633cd886f7846e4321af11854bfd880873da"
  "mwc_n_48x48.onnx|https://github.com/PINTO0309/MWC/releases/download/onnx/mwc_n_48x48.onnx|c2e83bc6fa0fdade463d9e8c78a9ccfa74a678f86631b4c8ee1fe2663d7f291d"
  # --- eye gaze (L2CS-style binned yaw/pitch; Gaze360 weights, research-use - see MANIFEST) -
  "mobilenetv2_gaze.onnx|https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx|b81312df85c7ac1c1b5f78c573620d22c2719cb839650e15f12dc7eecb7744a4"
  # --- in-cabin object / phone detection (Apache-2.0; replaces the AGPL Ultralytics plan) ---
  "yolox_nano.onnx|https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx|c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
  # --- driver identity embedding (Apache-2.0) ----------------------------------------------
  "face_recognition_sface_2021dec.onnx|https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)

sha_of() { sha256sum "$1" | awk "{print \$1}"; }

fetch_one() {
  local out="$1" url="$2" sha="$3"
  if [[ -f "${out}" ]]; then
    local have
    have="$(sha_of "${out}")"
    if [[ "${have}" == "${sha}" ]]; then
      echo "ok   $(basename "${out}") (cached, sha verified)"
      return 0
    fi
    echo "stale $(basename "${out}") sha=${have} != pinned ${sha}; re-downloading" >&2
    rm -f "${out}"
  fi
  echo "fetch ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "${out}.tmp" "${url}"
  else
    wget -O "${out}.tmp" "${url}"
  fi
  local got
  got="$(sha_of "${out}.tmp")"
  if [[ "${got}" != "${sha}" ]]; then
    echo "HASH MISMATCH $(basename "${out}"): expected ${sha} got ${got}" >&2
    rm -f "${out}.tmp"
    exit 1
  fi
  mv "${out}.tmp" "${out}"
  echo "ok   $(basename "${out}") ${got}"
}

FILTER="${1:-}"
COUNT=0
for entry in "${MODELS[@]}"; do
  IFS="|" read -r name url sha <<<"${entry}"
  if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
    continue
  fi
  fetch_one "${DEST}/${name}" "${url}" "${sha}"
  COUNT=$((COUNT + 1))
done

if [[ "${COUNT}" -eq 0 ]]; then
  echo "no model matched filter ${FILTER}" >&2
  exit 1
fi

echo "${COUNT} ONNX verified under ${DEST} (contract: ${MANIFEST})"
echo "next: PYTHONPATH=. python3 scripts/build_engines.py"
