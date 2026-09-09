#!/usr/bin/env bash
# One-shot NX install. Run as root from the repo root.
# Does NOT call nvpmodel/jetson_clocks unless --apply-power is passed.
# Default --lab installs videotestsrc config (no camera). --vehicle installs CSI production.yaml.
set -euo pipefail

APPLY_POWER=0
MODE=lab
PREFIX=/opt/dms
REPO="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<EOF
Usage: sudo $0 [--lab|--vehicle] [--apply-power] [--prefix DIR]

  --lab           (default) /etc/dms/default.yaml from configs/systemd-lab.yaml (test source)
  --vehicle       /etc/dms/default.yaml from configs/production.yaml (csi)
  --apply-power   run: nvpmodel -m 8  (MODE_20W_6CORE). jetson_clocks is NEVER auto.
  --prefix DIR    install prefix (default /opt/dms)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --apply-power) APPLY_POWER=1 ;;
    --lab) MODE=lab ;;
    --vehicle) MODE=vehicle ;;
    --prefix) PREFIX="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if [ "$(id -u)" -ne 0 ]; then
  echo "setup_jetson.sh must run as root" >&2
  exit 1
fi

GROUPS=video
if getent group gpio >/dev/null 2>&1; then
  GROUPS=video,gpio
fi
id dms >/dev/null 2>&1 || useradd --system --home "$PREFIX" --groups "$GROUPS" --shell /usr/sbin/nologin dms
install -d -o dms -g dms "$PREFIX" /var/lib/dms /etc/dms

if [ ! -x "$PREFIX/.venv/bin/python" ]; then
  python3.8 -m venv --system-site-packages "$PREFIX/.venv"
  "$PREFIX/.venv/bin/pip" install -U pip wheel
  "$PREFIX/.venv/bin/pip" install -r "$REPO/requirements.txt"
  "$PREFIX/.venv/bin/pip" install 'cuda-python>=11.4,<12' || echo "cuda-python missing; ctypes libcudart fallback"
fi

# ABI go/no-go (must not throw)
"$PREFIX/.venv/bin/python" -c "import numpy, cv2, tensorrt; a=numpy.zeros((8,8,3), numpy.uint8); cv2.cvtColor(a, cv2.COLOR_BGR2RGB); print('numpy', numpy.__version__, 'cv2', cv2.__version__, 'trt', tensorrt.__version__)"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '.git' --exclude 'tests/replay/*.mp4' --exclude 'tests/replay/runs' "$REPO"/ "$PREFIX"/
else
  tar -C "$REPO" --exclude '.git' --exclude 'tests/replay/runs' -cf - . | tar -C "$PREFIX" -xf -
fi
chown -R dms:dms "$PREFIX"

if [ "$MODE" = "vehicle" ]; then
  install -m 0644 "$REPO/configs/production.yaml" /etc/dms/default.yaml
  echo "installed CSI production.yaml — confirm sensor before enable"
else
  install -m 0644 "$REPO/configs/systemd-lab.yaml" /etc/dms/default.yaml
  echo "installed systemd-lab.yaml (test source, source.dev true) until CSI exists"
fi
chown dms:dms /etc/dms/default.yaml

install -m 0644 "$REPO/deploy/dms.service" /etc/systemd/system/dms.service
if [ -d /etc/logrotate.d ]; then
  install -m 0644 "$REPO/deploy/dms.logrotate" /etc/logrotate.d/dms
fi
systemctl daemon-reload

echo "sudo nvpmodel -m 8   # MODE_20W_6CORE"
echo "sudo jetson_clocks   # OPTIONAL, thermal risk — never auto"

if [ "$APPLY_POWER" -eq 1 ]; then
  nvpmodel -m 8
  echo "applied nvpmodel -m 8"
else
  echo "power unchanged (pass --apply-power to run nvpmodel -m 8)"
fi

echo "enable with: systemctl enable --now dms.service"
echo "health: curl -s http://127.0.0.1:8088/healthz"
