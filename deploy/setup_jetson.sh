#!/usr/bin/env bash
#
# DMS installer for a Jetson Xavier NX (L4T R35.6.5 / JetPack 5.1.6).
#
# Design rules, in order of importance:
#
#   1. Never half-install. Everything is staged into
#      $PREFIX/releases/<release-id> and only becomes live when the
#      $PREFIX/current symlink is swapped, which is the last step. A failure
#      anywhere before that leaves the running install untouched, and `set -euo
#      pipefail` plus an ERR trap make sure a failure is a failure.
#   2. Rollback is one symlink. `ln -sfn releases/<old> current && systemctl restart
#      dms` puts the previous release back, venv included.
#   3. Code and config are root-owned and read-only to the service. The dms account
#      owns only /var/lib/dms. A process that can rewrite engines/MANIFEST.json can
#      defeat its own integrity check, so it must not be able to.
#   4. Ship only what runs. No .venv from the dev tree, no tests/, no docs/, no
#      data/events.jsonl from the lab, no configs/vehicle.yaml (this board's
#      calibration), and above all none of tests/replay/*.jpg|*.avi — ~11 MB of
#      rendered overlays of real drivers' faces that the previous rsync copied onto
#      every unit.
#   5. Never touch the power state of a vehicle unless asked: nvpmodel runs only
#      under --apply-power, and jetson_clocks is never run at all.
#
# This script cannot be executed by CI (it needs root). It is validated by
# `bash -n`, by `shellcheck`, and by `--dry-run`, which performs every read-only
# precondition check for real and prints every mutating command without running it.
# tests/test_deploy.py runs all three.
#
# Usage:
#   sudo bash deploy/setup_jetson.sh [--lab|--vehicle] [options]
#
set -euo pipefail

PROG="$(basename "$0")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

MODE=lab
PREFIX=/opt/dms
APPLY_POWER=0
DRY_RUN=0
FORCE_CONFIG=0
NO_RESTART=0
RELEASE_ID=""
STATE_DIR=/var/lib/dms
LOG_DIR=/var/log/dms
CONF_DIR=/etc/dms
SERVICE=dms.service
SERVICE_USER=dms

usage() {
  cat <<EOF
Usage: sudo $PROG [--lab|--vehicle] [options]

  --lab            (default) /etc/dms/default.yaml from configs/systemd-lab.yaml
                   (videotestsrc; no camera needed)
  --vehicle        /etc/dms/default.yaml from configs/production.yaml (CSI)
  --apply-power    run 'nvpmodel -m 8' (MODE_20W_6CORE). jetson_clocks is NEVER run.
  --prefix DIR     install prefix (default $PREFIX). The unit file is rewritten to match.
  --release ID     name the release directory (default: git short sha, else a UTC stamp)
  --force-config   overwrite an existing /etc/dms/default.yaml (it is preserved by default)
  --no-restart     install everything but leave the running service alone
  --dry-run        run every check, print every change, modify nothing. No root needed.
  -h, --help       this text

Rollback:
  ls $PREFIX/releases
  ln -sfn $PREFIX/releases/<old> $PREFIX/current && systemctl restart $SERVICE
EOF
}

# --------------------------------------------------------------------- logging

info() { printf '[ ok ] %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
die() {
  printf '[FAIL] %s\n' "$*" >&2
  exit 1
}

on_err() {
  local rc=$?
  local line=${1:-?}
  printf '\n[FAIL] %s aborted at line %s (exit %s).\n' "$PROG" "$line" "$rc" >&2
  if [ -n "${RELEASE_DIR:-}" ] && [ -d "${RELEASE_DIR:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
    printf '[FAIL] The staged release %s was NOT activated; the previous install is untouched.\n' \
      "$RELEASE_DIR" >&2
    printf '[FAIL] Remove it with: rm -rf %s\n' "$RELEASE_DIR" >&2
  fi
  exit "$rc"
}
trap 'on_err $LINENO' ERR

# Run a mutating command, honouring --dry-run.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry ] %s\n' "$*"
    return 0
  fi
  "$@"
}

# Run a mutating shell snippet (redirections, globs), honouring --dry-run.
run_sh() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry ] sh -c %s\n' "$1"
    return 0
  fi
  sh -c "$1"
}

# ------------------------------------------------------------------ arguments

while [ $# -gt 0 ]; do
  case "$1" in
    --lab) MODE=lab ;;
    --vehicle) MODE=vehicle ;;
    --apply-power) APPLY_POWER=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --force-config) FORCE_CONFIG=1 ;;
    --no-restart) NO_RESTART=1 ;;
    --prefix)
      [ $# -ge 2 ] || die "--prefix needs a directory"
      PREFIX="$2"
      shift
      ;;
    --release)
      [ $# -ge 2 ] || die "--release needs an id"
      RELEASE_ID="$2"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf '[FAIL] unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$PREFIX" in
  /*) ;;
  *) die "--prefix must be an absolute path, got '$PREFIX'" ;;
esac
# run_sh single-quotes paths into `sh -c`; keep the character set boring so it cannot
# be broken out of, and so the systemd unit rewrite below stays a plain sed.
case "$PREFIX" in
  *[!A-Za-z0-9._/-]*) die "--prefix may only contain [A-Za-z0-9._/-], got '$PREFIX'" ;;
esac
PREFIX="${PREFIX%/}"
[ -n "$PREFIX" ] || die "--prefix must not be /"

# ---------------------------------------------------------------- preconditions

step "Preconditions"

if [ "$DRY_RUN" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
  die "$PROG must run as root (or use --dry-run)"
fi

command -v systemctl >/dev/null 2>&1 || die "systemctl not found; this is not a systemd host"
command -v install >/dev/null 2>&1 || die "coreutils 'install' not found"
command -v useradd >/dev/null 2>&1 || die "'useradd' not found"

PYTHON_BIN=""
for candidate in python3.8 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)'; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  fi
done
[ -n "$PYTHON_BIN" ] || die "python3.8 is required (the board ships 3.8.10 and the venv must match)"
info "python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

"$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1 ||
  die "python3.8-venv is missing: apt-get install python3.8-venv"

# The venv is --system-site-packages, so cv2/tensorrt/gi must already be importable
# from the system interpreter. Checking here means a missing JetPack component fails
# before anything is written, not after.
"$PYTHON_BIN" - <<'PY' || die "system site-packages are incomplete (see the error above)"
import sys
missing = []
for mod in ("numpy", "cv2", "tensorrt", "gi"):
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001 - we want the reason printed
        missing.append("%s (%s)" % (mod, exc))
if missing:
    sys.stderr.write("missing system modules: %s\n" % ", ".join(missing))
    sys.exit(1)
PY
info "system modules present: numpy, cv2, tensorrt, gi"

for required in \
  "$REPO/dms/app.py" \
  "$REPO/requirements.txt" \
  "$REPO/deploy/$SERVICE" \
  "$REPO/deploy/dms.logrotate" \
  "$REPO/deploy/dms-tmpfiles.conf" \
  "$REPO/configs/production.yaml" \
  "$REPO/configs/systemd-lab.yaml"; do
  [ -f "$required" ] || die "repository is incomplete: $required is missing"
done
info "repository layout looks complete: $REPO"

if [ ! -f "$REPO/engines/MANIFEST.json" ]; then
  die "engines/MANIFEST.json is missing; build engines before installing (scripts/build_engines.py)"
fi

ENGINE_COUNT="$(find "$REPO/engines" -maxdepth 1 -name '*.engine' | wc -l | tr -d ' ')"
if [ "$ENGINE_COUNT" -eq 0 ]; then
  warn "no .engine files in $REPO/engines — the service will fail its manifest check at start"
else
  info "engines to deploy: $ENGINE_COUNT"
fi

# 1 GiB is comfortably more than a release (code + a fresh venv + the engine set).
DF_TARGET="$PREFIX"
while [ ! -d "$DF_TARGET" ] && [ "$DF_TARGET" != "/" ]; do
  DF_TARGET="$(dirname "$DF_TARGET")"
done
FREE_KB="$(df -Pk "$DF_TARGET" | awk 'NR==2 {print $4}')"
if [ "${FREE_KB:-0}" -lt 1048576 ]; then
  die "less than 1 GiB free on $DF_TARGET (${FREE_KB} KiB); refusing to install"
fi
info "free space on $DF_TARGET: $((FREE_KB / 1024)) MiB"

if [ -z "$RELEASE_ID" ]; then
  if command -v git >/dev/null 2>&1 && git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    RELEASE_ID="$(git -C "$REPO" rev-parse --short=12 HEAD)"
    if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
      RELEASE_ID="${RELEASE_ID}-dirty-$(date -u +%Y%m%dT%H%M%SZ)"
      warn "working tree is dirty; release id is $RELEASE_ID"
    fi
  else
    RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  fi
fi
case "$RELEASE_ID" in
  *[!A-Za-z0-9._-]*) die "release id '$RELEASE_ID' contains characters outside [A-Za-z0-9._-]" ;;
esac

RELEASES_DIR="$PREFIX/releases"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
CURRENT_LINK="$PREFIX/current"
info "release id: $RELEASE_ID"
info "release dir: $RELEASE_DIR"
if [ "$DRY_RUN" -eq 1 ]; then
  warn "DRY RUN — nothing below is executed"
fi

# ------------------------------------------------------------- service account

step "Service account"

# NOTE: the variable below must NOT be called GROUPS. GROUPS is a bash special
# array holding the caller's own group ids; assignments to it are silently ignored,
# so the previous `GROUPS=video; useradd --groups "$GROUPS"` created the dms account
# in group 0 (root) and nobody noticed, because SupplementaryGroups= in the unit
# supplied the real device groups at runtime.
EXTRA_GROUPS=video
if getent group gpio >/dev/null 2>&1; then
  EXTRA_GROUPS=video,gpio
else
  warn "no 'gpio' group on this board; the alert pin will not be usable by $SERVICE_USER"
fi

if id "$SERVICE_USER" >/dev/null 2>&1; then
  info "user $SERVICE_USER already exists"
  run usermod --append --groups "$EXTRA_GROUPS" "$SERVICE_USER"
else
  run useradd --system --home-dir "$STATE_DIR" --no-create-home \
    --groups "$EXTRA_GROUPS" --shell /usr/sbin/nologin "$SERVICE_USER"
  if [ "$DRY_RUN" -eq 0 ]; then
    info "created system user $SERVICE_USER"
  fi
fi

if [ "$DRY_RUN" -eq 0 ]; then
  ACTUAL_GROUPS="$(id -nG "$SERVICE_USER")"
  case " $ACTUAL_GROUPS " in
    *" video "*) : ;;
    *) die "post-condition failed: $SERVICE_USER is not in group 'video' (groups: $ACTUAL_GROUPS)" ;;
  esac
  case " $ACTUAL_GROUPS " in
    *" root "*) die "post-condition failed: $SERVICE_USER is in group 'root' — refusing to continue" ;;
    *) : ;;
  esac
  info "groups for $SERVICE_USER: $ACTUAL_GROUPS"
fi

# ---------------------------------------------------------------- directories

step "Directories"

# Code and config: root-owned, world-readable, not writable by the service.
run install -d -o root -g root -m 0755 "$PREFIX" "$RELEASES_DIR" "$CONF_DIR"
# State: the only thing the service may write. 0750 keeps the driver's alert history
# off every other local account on the head unit.
run install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$STATE_DIR" "$LOG_DIR"
info "prefix=$PREFIX state=$STATE_DIR logs=$LOG_DIR config=$CONF_DIR"

# ------------------------------------------------------------ stage the release

step "Stage release"

run install -d -o root -g root -m 0755 "$RELEASE_DIR"

# Explicit allow-list. Anything not named here does not reach a vehicle.
copy_dir() {
  local src="$1" dst="$2"
  shift 2
  local extra=()
  if [ $# -gt 0 ]; then
    extra=("$@")
  fi
  if [ ! -d "$src" ]; then
    warn "skipping missing directory $src"
    return 0
  fi
  if command -v rsync >/dev/null 2>&1; then
    run rsync -a --delete \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.pyo' \
      ${extra[@]+"${extra[@]}"} "$src/" "$dst/"
  else
    run install -d -m 0755 "$dst"
    run_sh "tar -C '$src' --exclude='__pycache__' --exclude='*.pyc' -cf - . | tar -C '$dst' -xf -"
  fi
}

copy_dir "$REPO/dms" "$RELEASE_DIR/dms"
copy_dir "$REPO/scripts" "$RELEASE_DIR/scripts"
copy_dir "$REPO/deploy" "$RELEASE_DIR/deploy"
# configs/vehicle.yaml is this board's measured forward-zero and EAR baseline. It is
# machine-specific calibration and must never be copied onto another unit.
copy_dir "$REPO/configs" "$RELEASE_DIR/configs" --exclude 'vehicle.yaml'

run install -d -o root -g root -m 0755 "$RELEASE_DIR/engines"
run install -m 0644 "$REPO/engines/MANIFEST.json" "$RELEASE_DIR/engines/MANIFEST.json"
if [ "$ENGINE_COUNT" -gt 0 ]; then
  run_sh "install -m 0644 '$REPO'/engines/*.engine '$RELEASE_DIR/engines/'"
fi
run install -m 0644 "$REPO/requirements.txt" "$RELEASE_DIR/requirements.txt"
if [ -f "$REPO/readme.md" ]; then
  run install -m 0644 "$REPO/readme.md" "$RELEASE_DIR/readme.md"
fi

# NOT deployed, on purpose: .venv (rebuilt below), data/ (lab event records),
# tests/ (including ~11 MB of rendered face imagery from the replay clips),
# docs/, tools/, .git/, configs/vehicle.yaml.

run chown -R root:root "$RELEASE_DIR"
run chmod -R go-w "$RELEASE_DIR"
info "staged $(du -sh "$RELEASE_DIR" 2>/dev/null | cut -f1 || echo '?') into $RELEASE_DIR"

# ------------------------------------------------------------------- virtualenv

step "Virtualenv"

VENV="$RELEASE_DIR/.venv"
run "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
# Dependencies are installed on EVERY release, never skipped because a venv already
# exists: the previous script guarded the whole pip block on the venv's absence, so a
# changed requirements.txt was silently never applied on an upgrade.
run "$VENV/bin/pip" install --disable-pip-version-check --no-input -q -U pip wheel
if [ -f "$REPO/requirements.lock" ]; then
  # A hash-pinned lock file, when the project has one, is installed with
  # --require-hashes so a compromised or substituted wheel fails the install.
  run "$VENV/bin/pip" install --disable-pip-version-check --no-input -q \
    --require-hashes -r "$REPO/requirements.lock"
  info "installed hash-pinned requirements.lock"
else
  run "$VENV/bin/pip" install --disable-pip-version-check --no-input -q \
    -r "$RELEASE_DIR/requirements.txt"
  warn "requirements.txt carries versions but no --hash= lines; the venv is not"
  warn "reproducible. Generate requirements.lock with pip-compile --generate-hashes."
fi

# cuda-python is optional: dms/runtime/cudart.py falls back to ctypes+libcudart. The
# previous script swallowed the failure with an echo, so two units could silently end
# up on different code paths. Record which path this unit took.
# Override CUDA_PYTHON_SPEC with an exact pin (cuda-python==11.8.3) for a
# reproducible fleet build; the resolved version is recorded either way.
CUDA_PYTHON_SPEC="${CUDA_PYTHON_SPEC:-cuda-python>=11.4,<12}"
CUDA_PYTHON_STATE=absent
if run "$VENV/bin/pip" install --disable-pip-version-check --no-input -q "$CUDA_PYTHON_SPEC"; then
  if [ "$DRY_RUN" -eq 0 ]; then
    CUDA_PYTHON_STATE="$("$VENV/bin/pip" show cuda-python 2>/dev/null |
      awk '/^Version:/ {print "installed-" $2}')"
    [ -n "$CUDA_PYTHON_STATE" ] || CUDA_PYTHON_STATE=absent
  else
    CUDA_PYTHON_STATE="installed-dry-run"
  fi
else
  warn "cuda-python ($CUDA_PYTHON_SPEC) could not be installed;"
  warn "dms/runtime/cudart.py will use its ctypes libcudart fallback on this unit."
fi
info "cuda-python: $CUDA_PYTHON_STATE (recorded in $CONF_DIR/release)"

step "ABI go/no-go"
if [ "$DRY_RUN" -eq 0 ]; then
  "$VENV/bin/python" - <<'PY' || die "ABI check failed inside the new venv"
import numpy, cv2, tensorrt
a = numpy.zeros((8, 8, 3), numpy.uint8)
cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
print("    numpy %s  cv2 %s  tensorrt %s" % (numpy.__version__, cv2.__version__, tensorrt.__version__))
PY
  ( cd "$RELEASE_DIR" && PYTHONPATH=. "$VENV/bin/python" -c 'import dms.app' ) ||
    die "the staged tree does not import (dms.app)"
  info "ABI and import check passed"
else
  printf '[dry ] %s -c "import numpy, cv2, tensorrt"\n' "$VENV/bin/python"
fi

# --------------------------------------------------------------------- config

step "Configuration"

if [ "$MODE" = "vehicle" ]; then
  SRC_CONFIG="$REPO/configs/production.yaml"
else
  SRC_CONFIG="$REPO/configs/systemd-lab.yaml"
fi

if [ -f "$CONF_DIR/default.yaml" ] && [ "$FORCE_CONFIG" -eq 0 ]; then
  info "keeping the existing $CONF_DIR/default.yaml (pass --force-config to replace it)"
else
  run install -o root -g root -m 0644 "$SRC_CONFIG" "$CONF_DIR/default.yaml"
  info "installed $(basename "$SRC_CONFIG") as $CONF_DIR/default.yaml"
fi
# root-owned: the service must not be able to relax its own privacy settings,
# driver_roi or health.bind and have the change survive a reboot.
run chown root:root "$CONF_DIR/default.yaml"
run chmod 0644 "$CONF_DIR/default.yaml"

if [ "$MODE" = "vehicle" ]; then
  warn "vehicle profile installed: confirm the CSI sensor and driver_roi before enabling"
fi

# ---------------------------------------------------------------- system units

step "systemd, tmpfiles and logrotate"

UNIT_TMP="$(mktemp)"
trap 'rm -f "$UNIT_TMP"' EXIT
if [ "$PREFIX" = "/opt/dms" ]; then
  cp "$REPO/deploy/$SERVICE" "$UNIT_TMP"
else
  # The unit hardcodes /opt/dms in WorkingDirectory, ExecStart, Documentation and
  # ReadWritePaths. --prefix used to be accepted and then ignored here, producing a
  # unit that pointed at a directory the installer had not created.
  sed "s|/opt/dms|$PREFIX|g" "$REPO/deploy/$SERVICE" >"$UNIT_TMP"
  info "rewrote the unit for prefix $PREFIX"
fi
run install -o root -g root -m 0644 "$UNIT_TMP" "/etc/systemd/system/$SERVICE"

run install -o root -g root -m 0644 "$REPO/deploy/dms-tmpfiles.conf" /etc/tmpfiles.d/dms.conf
if command -v systemd-tmpfiles >/dev/null 2>&1; then
  run systemd-tmpfiles --create /etc/tmpfiles.d/dms.conf
fi

if [ -d /etc/logrotate.d ]; then
  run install -o root -g root -m 0644 "$REPO/deploy/dms.logrotate" /etc/logrotate.d/dms
  if command -v logrotate >/dev/null 2>&1 && [ "$DRY_RUN" -eq 0 ]; then
    logrotate -d /etc/logrotate.d/dms >/dev/null 2>&1 ||
      warn "logrotate rejected /etc/logrotate.d/dms; check it with: logrotate -d /etc/logrotate.d/dms"
  fi
else
  warn "/etc/logrotate.d does not exist; the event JSONL will only be bounded in-process"
fi

run systemctl daemon-reload
if [ "$DRY_RUN" -eq 0 ] && command -v systemd-analyze >/dev/null 2>&1; then
  systemd-analyze verify "/etc/systemd/system/$SERVICE" ||
    warn "systemd-analyze reported problems with $SERVICE (see above)"
fi

# ------------------------------------------------------------------- activate

step "Activate"

WAS_ACTIVE=0
if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
  WAS_ACTIVE=1
fi

if [ "$WAS_ACTIVE" -eq 1 ] && [ "$NO_RESTART" -eq 0 ]; then
  info "stopping $SERVICE before the symlink swap"
  run systemctl stop "$SERVICE"
fi

# Atomic: ln -sfn onto a temp name then mv, so `current` is never briefly missing.
run_sh "ln -sfn '$RELEASE_DIR' '$PREFIX/.current.new' && mv -Tf '$PREFIX/.current.new' '$CURRENT_LINK'"
info "$CURRENT_LINK -> $RELEASE_DIR"

RELEASE_FILE="$CONF_DIR/release"
if [ "$DRY_RUN" -eq 0 ]; then
  {
    echo "release=$RELEASE_ID"
    echo "git_sha=${RELEASE_ID%%-dirty*}"
    echo "installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode=$MODE"
    echo "prefix=$PREFIX"
    echo "cuda_python=$CUDA_PYTHON_STATE"
    echo "engines=$ENGINE_COUNT"
  } >"$RELEASE_FILE"
  chown root:root "$RELEASE_FILE"
  chmod 0644 "$RELEASE_FILE"
  info "wrote $RELEASE_FILE (dms/io/metrics.py publishes git_sha as dms_build_info)"
else
  printf '[dry ] write %s\n' "$RELEASE_FILE"
fi

# Keep the two most recent previous releases for rollback; delete older ones.
prune_old_releases() {
  local keep=3 n=0 old
  [ -d "$RELEASES_DIR" ] || return 0
  while IFS= read -r old; do
    [ -n "$old" ] || continue
    old="${old%/}"
    [ "$old" = "$RELEASE_DIR" ] && continue
    n=$((n + 1))
    if [ "$n" -ge "$keep" ]; then
      printf '       pruning old release %s\n' "$old"
      rm -rf -- "$old"
    fi
    # shellcheck disable=SC2012  # release ids are restricted to [A-Za-z0-9._-]
  done <<EOF
$(ls -1dt "$RELEASES_DIR"/*/ 2>/dev/null || true)
EOF
}
if [ "$DRY_RUN" -eq 0 ]; then
  prune_old_releases
fi

if [ "$NO_RESTART" -eq 1 ]; then
  warn "--no-restart: $SERVICE was not started. Activate with: systemctl restart $SERVICE"
elif [ "$WAS_ACTIVE" -eq 1 ]; then
  run systemctl start "$SERVICE"
  info "restarted $SERVICE"
else
  info "$SERVICE was not running; enable it with: systemctl enable --now $SERVICE"
fi

# ----------------------------------------------------------------------- power

step "Power"
if [ "$APPLY_POWER" -eq 1 ]; then
  command -v nvpmodel >/dev/null 2>&1 || die "--apply-power given but nvpmodel is not installed"
  run nvpmodel -m 8
  info "applied nvpmodel -m 8 (MODE_20W_6CORE)"
else
  info "power mode unchanged. To apply: sudo nvpmodel -m 8   # MODE_20W_6CORE"
fi
info "jetson_clocks is never run by this script (thermal risk on a sealed enclosure)"

# ------------------------------------------------------------------- summary

step "Done"
cat <<EOF
  release     $RELEASE_ID
  current     $CURRENT_LINK -> $RELEASE_DIR
  config      $CONF_DIR/default.yaml ($MODE)
  state       $STATE_DIR  (events.jsonl, 0640 dms:dms)
  unit        /etc/systemd/system/$SERVICE

  systemctl enable --now $SERVICE
  systemctl status $SERVICE
  journalctl -u $SERVICE -o cat -f | jq .
  curl -fsS http://127.0.0.1:8088/healthz | jq .
  curl -fsS http://127.0.0.1:8088/readyz  >/dev/null && echo ready
  curl -fsS http://127.0.0.1:8088/metrics | head

  rollback:   ln -sfn $RELEASES_DIR/<old> $CURRENT_LINK && systemctl restart $SERVICE
EOF
