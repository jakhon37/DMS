# Replay clips (local only — `*.mp4` is gitignored)

Download with the aarch64 yt-dlp binary, 720p H.264 (`-f 136`).

| File | Config | Source | Duration | Notes |
| --- | --- | --- | --- | --- |
| `day_driver.mp4` | `configs/default.yaml` | [JOSHSCP30fs](https://www.youtube.com/watch?v=JOSHSCP30fs) | ~3:27 | Single driver, 3/4 view. Calibrate `vehicle.yaml`. |
| `minivan.mp4` | `configs/minivan.yaml` | [Z0cqqe7aU70](https://www.youtube.com/watch?v=Z0cqqe7aU70) | 1:00 | Multi-occupant; driver on **right**; `driver_fallback: false`. |
| `red_car_crash.mp4` | `configs/red_car_crash.yaml` | [0RQewd3JYwU](https://www.youtube.com/watch?v=0RQewd3JYwU) | 1:52 | [Red Car Crash](https://www.youtube.com/playlist?list=PLRx21j1TOaNbMaPnPvHCF2p23taexwnR6) inside view. Driver on **left**; `driver_fallback: false`. |

```bash
# single driver
PYTHONPATH=. python3 -m dms.app --config configs/default.yaml --detect --max-frames 60

# minivan occupants
PYTHONPATH=. python3 -m dms.app --config configs/minivan.yaml --detect --max-frames 60

# inside crash
PYTHONPATH=. python3 -m dms.app --config configs/red_car_crash.yaml --detect --max-frames 60 \
  --save-preview tests/replay/preview_crash.jpg
```
