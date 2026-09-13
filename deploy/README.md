# DMS deployment and operations

Everything in this directory is what turns the repository into a service on a Jetson
Xavier NX. Read this before running `setup_jetson.sh` on a vehicle.

| File | Installed as | Purpose |
| --- | --- | --- |
| `setup_jetson.sh` | — | staged, atomic installer with rollback |
| `dms.service` | `/etc/systemd/system/dms.service` | the unit, sandboxed |
| `dms-tmpfiles.conf` | `/etc/tmpfiles.d/dms.conf` | state directory ownership and modes |
| `dms.logrotate` | `/etc/logrotate.d/dms` | AlertEvent JSONL rotation |

---

## 1. Install

```bash
# lab board, no camera: videotestsrc profile
sudo bash deploy/setup_jetson.sh --lab

# vehicle: CSI profile. Confirm the sensor and driver_roi first.
sudo bash deploy/setup_jetson.sh --vehicle

# see exactly what it would do, as any user, changing nothing
bash deploy/setup_jetson.sh --dry-run --vehicle
```

The installer never modifies a running install in place. It stages everything into
`/opt/dms/releases/<git-sha>`, builds a fresh virtualenv there, runs an ABI and import
check against it, and only then swaps the `/opt/dms/current` symlink. If any step
fails the previous release is still live and the script says so.

**Rollback is one symlink:**

```bash
ls -1t /opt/dms/releases
sudo ln -sfn /opt/dms/releases/<previous> /opt/dms/current
sudo systemctl restart dms
```

The three most recent releases are kept; older ones are pruned on install.

### What is and is not deployed

Deployed: `dms/`, `configs/` (minus `vehicle.yaml`), `deploy/`, `scripts/`,
`engines/MANIFEST.json`, `engines/*.engine`, `requirements.txt`, `readme.md`.

Deliberately **not** deployed:

* `.venv/` from the development tree — the release builds its own.
* `tests/` — including `tests/replay/*.jpg` and `*.avi`, roughly 11 MB of rendered
  overlays of real drivers' faces. Shipping those to every vehicle is the exact
  opposite of the data-minimisation posture the design claims (K16).
* `data/` — lab `events.jsonl` with real alert records.
* `configs/vehicle.yaml` — this board's measured forward-zero and EAR baseline.
  Machine-specific calibration; installing it on another unit mis-zeroes gaze.
* `docs/`, `tools/`, `.git/`.

### Ownership model

| Path | Owner | Mode | Writable by `dms`? |
| --- | --- | --- | --- |
| `/opt/dms/**` (code, venv, engines, MANIFEST) | `root:root` | `0755` / `go-w` | **no** |
| `/etc/dms/default.yaml` | `root:root` | `0644` | **no** |
| `/etc/dms/release` | `root:root` | `0644` | **no** |
| `/var/lib/dms` | `dms:dms` | `0750` | yes |
| `/var/lib/dms/events.jsonl` | `dms:dms` | `0640` | yes |
| `/run/dms` | `dms:dms` | `0750` | yes |
| `/var/log/dms` | `dms:dms` | `0750` | yes |

A service that can rewrite `engines/MANIFEST.json` can rewrite the hashes that
validate its own engines, so the provenance check would be circular. It cannot.
`ProtectSystem=strict` plus `ReadWritePaths=/var/lib/dms /run/dms /var/log/dms`
enforces the same thing a second time at the kernel level.

`0750`/`0640` on the state directory is not cosmetic: `events.jsonl` is the driver's
complete drowsiness, microsleep, yawn and distraction history. At the systemd default
of `0755`/`0644` every local account on the head unit can read it.

---

## 2. Health and metrics

The endpoint binds `127.0.0.1:8088` by default and **refuses** to bind anywhere else
unless `health.allow_remote` is explicitly true, because `/healthz` publishes
`alerts_active` — live driver state.

```bash
curl -fsS http://127.0.0.1:8088/healthz | jq .     # 200 healthy, 503 not
curl -fsS http://127.0.0.1:8088/livez              # 200 while the process runs
curl -fsS http://127.0.0.1:8088/readyz             # 200 only when actually monitoring
curl -fsS http://127.0.0.1:8088/metrics            # Prometheus text
```

| Route | 200 means | 503 means |
| --- | --- | --- |
| `/livez` | the process is scheduled and the HTTP thread answers | never |
| `/readyz` | engines loaded, camera `ok`, frame id advancing, snapshot fresh | still booting, blind, or stalled |
| `/healthz` | `ok` and the snapshot is fresh | `ok=false` or the pipeline stopped refreshing |
| `/metrics` | always — a scraper must be able to scrape a broken unit | never |

A **degraded** unit (thermal throttling, one model failing) answers `/healthz` with
200 and `"degraded": true`. It is working, just not at full capability; taking it out
of service would be worse than running it.

`"stale": true` means the pipeline stopped refreshing the snapshot for longer than
`stale_after_s` (5 s). That is how a hung pipeline is distinguished from a slow one —
previously a hang froze fps, temperatures and alert state at their last values and
kept serving them with a 200.

### Metrics worth alerting on

| Series | Alert when |
| --- | --- |
| `dms_up` | absent for 2 min — the unit is gone |
| `dms_health_stale` | `1` for 30 s — pipeline hung |
| `dms_camera_up` | `0` for 60 s — blind unit |
| `dms_events_writable` | `0` — alerts are not being recorded |
| `dms_disk_full` | `1` — ENOSPC somewhere |
| `dms_disk_free_mb` | `< 200` |
| `rate(dms_alerts_enter_total[1h])` | `== 0` across a whole shift — a silently broken detector looks exactly like a healthy one otherwise |
| `dms_model_consecutive_failures` | `> 5` |
| `increase(dms_engine_errors_total[10m])` | `> 0` |
| `dms_manifest_ok` | `0` — an engine does not match its recorded hash |
| `count by (git_sha) (dms_build_info)` | more than one sha in the fleet after a rollout |

`dms_build_info{version,git_sha,trt_version,config}` comes from `/etc/dms/release`,
which the installer writes. Without it you cannot tell which units took the last
deploy.

### Remote scraping (opt-in, discouraged)

```yaml
health:
  bind: 0.0.0.0
  allow_remote: true
  token_file: /etc/dms/health.token   # root:root 0600
```

With `token_file` set, every route except `/livez` requires
`Authorization: Bearer <token>`. Without it the server still starts but logs
`HEALTH_REMOTE_BIND ... with NO token` at WARNING on every boot. Prefer an SSH tunnel
or a node exporter reading loopback.

---

## 3. The event log

`/var/lib/dms/events.jsonl`, one JSON object per line, schema version 1:

```json
{"v":1,"ts":1757731200.41,"t":91234.55,"boot":"5d0…","seq":42,"kind":"alert",
 "type":"microsleep","severity":"critical","edge":"enter","track_id":3,"extra":{"dur_s":1.8}}
```

* `t` is `CLOCK_MONOTONIC` — the clock alerts are timed on. It restarts near zero at
  every boot, which is why `boot` (the kernel boot id) is on every record: sort by
  `(boot, t)`, correlate with a CAN log or journald using `ts`.
* CRITICAL records are `fsync`ed immediately; everything else at most one second
  later. Power loss can therefore cost at most one second of non-critical edges.
* Rotation is bounded **in-process** at `events.max_mb` (64 MB) because logrotate
  runs from a daily timer and cannot bound an alert burst.
* On ENOSPC the writer stops, emits `DISK_FULL`, sets `events_writable: false` on
  `/healthz`, and retries every 5 s. **The process stays up.** When space returns it
  writes a `disk_full`/`exit` record naming how many records were lost and for how
  long, so the hole in the file is explained inside the file.

Reading it back safely (tolerates a torn final line after a power cut):

```python
from dms.io.events import read_events
for rec in read_events("/var/lib/dms/events.jsonl"):
    ...
```

Rotation check:

```bash
sudo logrotate -d /etc/logrotate.d/dms     # dry run, prints decisions
sudo logrotate -f /etc/logrotate.d/dms     # force one now
```

`create` (not `copytruncate`) is used, so no line is lost in the rotation window;
`EventLog` notices the inode change within 2 s and the `postrotate` `SIGHUP` makes it
immediate.

---

## 4. Clips

Off by default and gated on **two** switches:

```yaml
clips:
  enabled: false
privacy:
  record_faces: false
```

`dms/io/clips.py` refuses to encode a single frame unless both `clips.enabled` and
`privacy.record_faces` are true, and does not even create `/var/lib/dms/clips`. When both are true the unit logs
`CLIP_RING_ACTIVE …` at WARNING on every start — in-cabin video is personal data and
its recording should be loud.

When on: a 10 s-segment H.264 ring (`nvv4l2h264enc` → `splitmuxsink`), concatenated
with `ffmpeg -c copy` on a warn/critical edge into
`/var/lib/dms/clips/<UTC>_<severity>_<reason>.mp4`. A `quota_mb` (2048) and `retain_h`
(24) sweep runs oldest-first after every write. ENOSPC stops the ring, raises
`DISK_FULL` and leaves the process running.

---

## 5. systemd

### Restart and watchdog

`Restart=always` with `RestartSec=5`, `StartLimitIntervalSec=300`,
`StartLimitBurst=5`. A clean `exit(0)` restarts too: for a safety function, an
unexpected clean exit must not silently end monitoring for the rest of the drive.

Consequence to know about: the **lab** profile uses `videotestsrc`, which never ends.
If you point a unit at a finite mp4 it will exit cleanly at EOS, restart five times
and land in `failed`. That is intended — use `--source` by hand for replay, not the
unit.

`WatchdogSec=30`. The deadline only starts after `READY=1`, so it is independent of
`TimeoutStartSec=90`. The application pings at `WatchdogSec/2`
(`dms.io.sd_notify.watchdog_ping_interval_s`) and only while the frame id is
advancing, so a hung pipeline is killed and restarted rather than petted for ever.

There is no `OnFailure=` unit here because none exists yet. When the fleet gains a
notifier, add `OnFailure=dms-failure-notify.service` to `[Unit]`; referencing a unit
that does not exist would make the failure handling itself fail.

### Sandbox

Enabled and safe on this board: `NoNewPrivileges`, `ProtectSystem=strict` +
`ReadWritePaths`, `ProtectHome`, `PrivateTmp`, `ProtectKernel{Tunables,Modules,Logs}`,
`ProtectControlGroups`, `ProtectClock`, `ProtectHostname`, `RestrictSUIDSGID`,
`RestrictNamespaces`, `RestrictRealtime`, `LockPersonality`,
`RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK`, empty
`CapabilityBoundingSet`, `MemoryMax=5G`, `OOMPolicy=stop`, `TasksMax=256`,
`LimitCORE=0`, `UMask=0027`.

Deliberately **not** set:

* `MemoryDenyWriteExecute` — the CUDA/TensorRT runtime JITs and needs W+X.
* `PrivateDevices` — it hides `/dev/nvhost-*`, `/dev/nvmap` and `/dev/video*`.

### Validating the tight sandbox

`DevicePolicy=closed` + an explicit `DeviceAllow=` list and `SystemCallFilter=` are
present in `dms.service` **commented out**. They are the right hardening but a wrong
device or a syscall the CUDA runtime needs turns into a start failure, and this work
had no root on the board to boot-test them. Validate them like this, on a bench unit:

```bash
sudo sed -i 's/^#\(DevicePolicy\|DeviceAllow\|SystemCallFilter\|SystemCallErrorNumber\|SystemCallArchitectures\)/\1/' \
  /etc/systemd/system/dms.service
sudo systemctl daemon-reload && sudo systemctl restart dms
sleep 30
curl -fsS http://127.0.0.1:8088/readyz && echo "sandbox OK"
sudo journalctl -u dms -b --no-pager | grep -iE 'operation not permitted|EPERM|nvmap|nvhost'
```

If it comes up ready with no EPERM in the journal, commit the same change to
`deploy/dms.service`. If it does not, the journal names the missing device node — add
it to the list rather than abandoning the whole directive.

### Useful commands

```bash
systemctl status dms
journalctl -u dms -o cat -f | jq .            # jsonlog writes one JSON object per line
systemd-analyze verify /etc/systemd/system/dms.service
systemd-analyze security dms.service          # exposure score
systemctl show dms -p MemoryMax -p WatchdogUSec -p Restart
sudo systemctl kill -s HUP dms.service        # reopen the event log
```

### Physical alert output

`alerts.gpio_pin` + `alerts.buzzer` drive one pin HIGH while a CRITICAL alert is
latched. `dms/io/gpio_alert.py` releases it from `close()`, from `__exit__`, from an
`atexit` hook and — when the application calls `arm_signal_handlers()` — from
SIGTERM/SIGINT. That covers every shutdown the process can observe.

It cannot cover SIGKILL: the watchdog kill and the OOM killer give the process no
chance to run anything. **The output must therefore be wired fail-safe**: the buzzer
or lamp de-energised when the pin is released to its power-on input state. If the
harness is active-low, set `active_high: false` so "released" is the correct
electrical level. Verify on the bench with `systemctl kill -s KILL dms` and confirm
the cabin goes quiet.

---

## 6. Supply chain

* `scripts/fetch_onnx.sh` verifies a pinned sha256 for every ONNX, deletes and
  re-downloads a stale file, and **aborts with exit 1** if freshly downloaded bytes
  do not match the pin. That is fail-closed and was audited as part of this work.
* `engines/MANIFEST.json` pins each engine's sha256 and the required TensorRT
  version; `dms/runtime/manifest.py` enforces both before READY.
* `requirements.txt` pins versions but carries no `--hash=` lines. The installer
  therefore prints a warning on every run and prefers `requirements.lock` if the
  repository grows one: generate it with
  `pip-compile --generate-hashes -o requirements.lock requirements.txt` (off-board)
  and the installer will use `pip install --require-hashes`.
* `cuda-python` is optional. The resolved version — or `absent`, meaning
  `dms/runtime/cudart.py` is on its ctypes `libcudart` fallback — is recorded in
  `/etc/dms/release`, so two units can no longer differ silently. Pin it exactly for
  a fleet build with `CUDA_PYTHON_SPEC='cuda-python==11.8.3' sudo -E bash
  deploy/setup_jetson.sh …`.

---

## 7. Incident checklist

1. `curl -s localhost:8088/healthz | jq '{ok,ready,degraded,stale,camera,events,disk_free_mb}'`
2. `journalctl -u dms -b -o cat | jq -c 'select(.level!="INFO")' | tail -50`
3. `sudo tail -5 /var/lib/dms/events.jsonl | jq .` — is `seq` advancing, is there a
   `disk_full` record explaining a gap?
4. `curl -s localhost:8088/metrics | grep -E 'dms_(up|camera_up|health_stale|disk_full|alerts_enter_total|engine_errors_total)'`
5. `cat /etc/dms/release` — which build is on this unit?
6. `systemctl show dms -p NRestarts -p ExecMainStartTimestamp`
