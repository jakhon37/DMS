# Replay clips and the golden regression set

`tests/replay/*.mp4` is **gitignored** — the clips are local only. What *is*
committed is `tests/replay/golden/*.json`: the expected alert counts for each
clip, which is the regression baseline for the whole pipeline.

Download with the aarch64 `yt-dlp` binary, 720p H.264 (`-f 136`).
`family_vacation_crash.mp4` is 360p (`-f 134`; no 720p on that listing).

| File | Config | Source | Duration | Notes |
| --- | --- | --- | --- | --- |
| `day_driver.mp4` | `configs/default.yaml` | [JOSHSCP30fs](https://www.youtube.com/watch?v=JOSHSCP30fs) | ~3:27 | Single driver, 3/4 view. The reference profile. |
| `minivan.mp4` | `configs/minivan.yaml` | [Z0cqqe7aU70](https://www.youtube.com/watch?v=Z0cqqe7aU70) | 1:00 | Multi-occupant; driver on **right**; `driver_fallback: false`. |
| `red_car_crash.mp4` | `configs/red_car_crash.yaml` | [0RQewd3JYwU](https://www.youtube.com/watch?v=0RQewd3JYwU) | 1:52 | Inside view, single driver on **left**; `driver_fallback: false`. |
| `family_vacation_crash.mp4` | `configs/family_vacation_crash.yaml` | [S6eIh11oHwQ](https://www.youtube.com/watch?v=S6eIh11oHwQ) | 1:02 | 360p. Driver on **right** + passenger; `driver_fallback: false`. |
| `rear_end_accident.mp4` | `configs/rear_end_accident.yaml` | [0rR33uj1NUA](https://www.youtube.com/watch?v=0rR33uj1NUA) | 0:06 | Driver on **right** (wheel); hoodie is a passenger. Shorter than 300 frames. |
| `guy_rear_ended.mp4` | `configs/guy_rear_ended.yaml` | [35wu7Bx88_M](https://www.youtube.com/watch?v=35wu7Bx88_M) | 1:01 | Single driver on **right**. |
| `bmw_rear_ended.mp4` | `configs/bmw_rear_ended.yaml` | [JKLoycTetrE](https://www.youtube.com/watch?v=JKLoycTetrE) | 0:13 | Single driver on **right**. |
| `getting_rear_ended.mp4` | `configs/getting_rear_ended.yaml` | [uW9EuuQJDuM](https://www.youtube.com/watch?v=uW9EuuQJDuM) | 0:39 | Driver on **right**; the woman on the left is a passenger. |
| `rear_end_whiplash.mp4` | `configs/rear_end_whiplash.yaml` | [R43Q9dHhNF8](https://www.youtube.com/watch?v=R43Q9dHhNF8) | 0:18 | Driver on **right** (wheel); yellow shirt is a passenger. |

## THESE ARE NOT CABIN FOOTAGE

Every clip is a dashcam montage or a crash compilation. Faces are 70–220 px, so
the derived eye boxes are a median of **19 source pixels** wide — which is below
the resolution at which the eye classifier was measured to work, and is why the
model falls through to the geometric EAR on most of this footage.

Nothing in these clips is labelled. There is not one annotated microsleep, yawn or
over-shoulder look in the repository. Consequently:

* no threshold in `dms/state/fusion.py` has been validated against ground truth;
* the golden fixtures pin **software behaviour**, not accuracy;
* `MICROSLEEP` and `PERCLOS_HIGH` do not fire on any clip, which is expected and
  proves nothing either way.

**The single highest-value test asset this project could acquire** is two
driver-facing clips of 2–3 minutes with labelled events: 3 × eyes closed ≥ 2 s,
3 × yawns, 3 × over-shoulder looks, ideally one with sunglasses. That would turn
the eye and mouth operating points, the head-pose pitch scale and the golden
bands from judgement into measurement.

---

## Golden replay

```bash
# check every clip against its golden — this is the regression gate
flock /tmp/jetson-gpu.lock -c \
  "cd ~/myspace/DMS && PYTHONPATH=. ./.venv/bin/python tests/replay/golden/run_golden.py"

# one clip
... tests/replay/golden/run_golden.py --only day_driver

# re-record after a DELIBERATE behaviour change, then read the diff
... tests/replay/golden/run_golden.py --record
```

Exit 0 = every clip within band, 1 = a regression, 2 = a run failed to execute.

Each golden records, over a fixed frame count of a fixed profile: the frames
processed (exact), the total detections (a ±25% band, floor ±1) and the count of
every alert enter/escalate/exit edge (same band). Latency is recorded for
reference and **never asserted** — this board's GPU governor moves an identical
inference by up to 5×.

A regression is a change of *kind* (an alert type that appears or disappears) or
of *order of magnitude*. An alert type not present in the golden is always a
failure, even a plausible one.

### Recorded baseline (2026-09-13, SCRFD-640, all nine heads)

| clip | frames | detections | alert edges |
| --- | --- | --- | --- |
| `day_driver` | 300 | 583 | driver_changed 4, drowsiness_unreliable 5, face_lost 3 |
| `minivan` | 300 | 1201 | drowsiness_unreliable 1 |
| `red_car_crash` | 300 | 300 | drowsiness_unreliable 1, yawn 2 |
| `family_vacation_crash` | 300 | 643 | drowsiness_unreliable 1 |
| `rear_end_accident` | 186 (EOS) | 305 | driver_changed 10, drowsiness_unreliable 2, yawn 2 |
| `guy_rear_ended` | 300 | 300 | drowsiness_unreliable 1, yawn 2 |
| `bmw_rear_ended` | 300 | 300 | drowsiness_unreliable 1 |
| `getting_rear_ended` | 300 | 589 | drowsiness_unreliable 1 |
| `rear_end_whiplash` | 300 | 552 | drowsiness_unreliable 1, yawn 2 |

Two consecutive runs produced identical detection and alert counts, which is what
media-time (buffer PTS) scheduling buys: the arithmetic no longer depends on how
fast the board happened to be.

`drowsiness_unreliable` on every clip is correct and expected: no profile is
calibrated, so the pose channel cannot be trusted and the machine says so instead
of publishing a gaze alert. `driver_changed` on the montages is also real — those
clips cut between different vehicles.

---

## Ad-hoc replay

```bash
# single driver, 300 frames
flock /tmp/jetson-gpu.lock -c \
  "PYTHONPATH=. ./.venv/bin/python -m dms.app --config configs/default.yaml --max-frames 300"

# minivan occupants
... -m dms.app --config configs/minivan.yaml --max-frames 300

# with a latency budget written at shutdown
DMS_LATENCY_BUDGET_OUT=/tmp/lb.json ... -m dms.app --config configs/default.yaml --max-frames 300
```

`--save-preview` and `--save-video` write driver imagery and are **refused**
unless `privacy.record_faces: true` is set in the profile. That is deliberate:
the default posture is that no frame ever leaves the process.

Batch across every clip, with per-run RSS and load sampling:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_all_replay.py --max-frames 300
```

writes `tests/replay/runs/<timestamp>/<clip>/{events.jsonl,log.txt,...}`.
