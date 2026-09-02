# Replay clips (local only — `*.mp4` is gitignored)

Download with the aarch64 yt-dlp binary, 720p H.264 (`-f 136`). `family_vacation_crash.mp4` is 360p (`-f 134`; no 720p on that listing).

| File | Config | Source | Duration | Notes |
| --- | --- | --- | --- | --- |
| `day_driver.mp4` | `configs/default.yaml` | [JOSHSCP30fs](https://www.youtube.com/watch?v=JOSHSCP30fs) | ~3:27 | Single driver, 3/4 view. Calibrate `vehicle.yaml`. |
| `minivan.mp4` | `configs/minivan.yaml` | [Z0cqqe7aU70](https://www.youtube.com/watch?v=Z0cqqe7aU70) | 1:00 | Multi-occupant; driver on **right**; `driver_fallback: false`. |
| `red_car_crash.mp4` | `configs/red_car_crash.yaml` | [0RQewd3JYwU](https://www.youtube.com/watch?v=0RQewd3JYwU) | 1:52 | [Red Car Crash](https://www.youtube.com/playlist?list=PLRx21j1TOaNbMaPnPvHCF2p23taexwnR6) inside view. Driver on **left**; `driver_fallback: false`. |
| `family_vacation_crash.mp4` | `configs/family_vacation_crash.yaml` | [S6eIh11oHwQ](https://www.youtube.com/watch?v=S6eIh11oHwQ) | 1:02 | Same playlist, 360p H.264. Driver on **right** + passenger; `driver_fallback: false`. |

All local `*.mp4` files (with overlay + last-frame JPEG + events) into a new folder:

```bash
PYTHONPATH=. python3.8 scripts/run_all_replay.py
# or a short pass
PYTHONPATH=. python3.8 scripts/run_all_replay.py --max-frames 60
```

Writes `tests/replay/runs/<timestamp>/<clip>/{preview.jpg,overlay.avi,events.jsonl,log.txt}`.

```bash
# single driver
PYTHONPATH=. python3 -m dms.app --config configs/default.yaml --detect --max-frames 60

# minivan occupants
PYTHONPATH=. python3 -m dms.app --config configs/minivan.yaml --detect --max-frames 60

# inside crash
PYTHONPATH=. python3 -m dms.app --config configs/red_car_crash.yaml --detect --max-frames 60 \
  --save-preview tests/replay/preview_crash.jpg

# family vacation crash (driver on the right)
PYTHONPATH=. python3 -m dms.app --config configs/family_vacation_crash.yaml --detect --max-frames 60 \
  --save-preview tests/replay/preview_family.jpg
```
