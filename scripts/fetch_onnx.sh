#!/usr/bin/env bash
# Download UniFace-pinned ONNX (cookbook only — do not pip install uniface).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/engines"
MANIFEST="${DEST}/MANIFEST.json"
mkdir -p "${DEST}"

fetch_one() {
  local url="$1" sha="$2" out="$3"
  echo "fetch ${url} -> ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${out}.tmp" "${url}"
  else
    wget -O "${out}.tmp" "${url}"
  fi
  local got
  got="$(sha256sum "${out}.tmp" | awk '{print $1}')"
  if [[ "${got}" != "${sha}" ]]; then
    echo "HASH MISMATCH ${out}: expected ${sha} got ${got}" >&2
    rm -f "${out}.tmp"
    exit 1
  fi
  mv "${out}.tmp" "${out}"
  echo "ok ${out} ${got}"
}

fetch_one \
  "https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx" \
  "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a" \
  "${DEST}/scrfd_500m_kps.onnx"

fetch_one \
  "https://github.com/yakhyo/uniface/releases/download/weights/2d106det.onnx" \
  "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf" \
  "${DEST}/2d106det.onnx"

echo "wrote ONNX under ${DEST} (see ${MANIFEST})"
echo "next: python3 scripts/build_engines.py"
