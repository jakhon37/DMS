"""The driver alert state machine.

One :class:`DriverMonitor` watches one driver.  It consumes a
:class:`dms.types.DriverSignals` per frame - model output where a model ran,
classical geometry where it did not - hands it to :class:`dms.state.fusion.SignalFuser`
for provenance-tagged fusion, and turns the fused view into edge-triggered
:class:`dms.state.alerts.AlertEvent` records.

What this module guarantees
---------------------------
* **No alert on a single frame.**  Every alert is a leaky-integrator latch with
  separate enter and exit dwell times, so one bad frame costs one frame of
  credit instead of toggling an output or resetting an accumulation.
* **Exactly one event per edge**, in the order ``enter (escalate*) exit``.  An
  alert whose severity rises emits an ``escalate`` edge instead of silently
  mutating the record, so ``events.jsonl`` can be audited by pairing edges.
* **Cooldown is measured from the previous ENTER**, not from the resolution, so
  a driver having repeated microsleeps five seconds apart is alerted on every
  one of them.
* **Nothing latches through a blind spell.**  Losing the face makes the exit
  condition of every face-derived alert true, so a critical cannot be asserted
  about a driver the system has declared missing.
* **Nothing is stitched across two people.**  A change of track id or of
  identity emits exits for everything active, emits ``DRIVER_CHANGED``, and
  resets every window and filter.
* **Unmeasurable is not the same as fine.**  Sunglasses, a lost eye channel, an
  uncalibrated or low-confidence head pose, or a frame rate too low for PERCLOS
  raise ``DROWSINESS_UNRELIABLE`` rather than letting an eye- or pose-derived
  alert fire on a signal nobody can vouch for.

Units and time base
-------------------
``t_s`` is seconds on whatever clock the capture layer stamps frames with; on
file replay that must be media time, not wall time, or every duration threshold
is scaled by the replay speed.  Angles are degrees, PERCLOS a fraction.

Failure behaviour
-----------------
:meth:`DriverMonitor.update` never raises.  A frame with no signals at all is a
valid input: it is treated as "nothing was measured", which arms ``FACE_LOST``
and ``DROWSINESS_UNRELIABLE`` and disarms everything else.  The state machine is
driven entirely by the timestamps it is given, so a stalled pipeline that stops
calling ``update`` freezes the alert clock - the caller must keep calling it with
``present=False`` on a capture timeout.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

from dms.state.alerts import (
    EDGE_ENTER,
    EDGE_ESCALATE,
    EDGE_EXIT,
    AlertEvent,
    AlertType,
    Severity,
    max_severity,
    severity_rank,
)
from dms.state.fusion import (
    DWELL_EPS,
    FusedView,
    FusionConfig,
    SignalFuser,
    angdiff,
    finite,
    wrap180,
)
from dms.types import (
    DriverSignals,
    EyeState,
    GazeVector,
    HeadPose,
    MouthState,
    Occlusion,
    SignalSource,
)

__all__ = ["AlertTuning", "DriverMonitor", "angdiff", "wrap180"]


@dataclass
class AlertTuning:
    """Enter/exit dwell times, cooldowns and thresholds, all seconds unless noted.

    Defaults are the table in ``docs/production-dms-design.md``.  Two deliberate
    deviations, both documented at the field:

    * ``face_lost_crit_s`` - the design makes FACE_LOST critical immediately when
      the vehicle is assumed moving.  A 1.0 s detector gap is common enough that
      an instant critical is the classic alarm-fatigue generator, so FACE_LOST
      now enters at WARN and *escalates* to CRITICAL after this long.  Set it to
      0.0 for the literal design behaviour.
    * ``perclos_enter_s`` - the design has no enter dwell.  With none, the first
      valid PERCLOS window can fire a critical on a single sample.

    :meth:`from_app_config` overlays anything ``cfg.state`` actually defines, by
    name, so a YAML profile can retune without a schema change and an old YAML
    keeps working.
    """

    # presence
    face_lost_s: float = 1.0
    face_lost_exit_s: float = 0.40
    face_lost_cooldown_s: float = 5.0
    face_lost_crit_s: float = 3.0
    #: how long after startup a never-seen driver counts as lost.  Without this
    #: a camera aimed at the headrest, a blocked lens or a dead detector is
    #: indistinguishable from "not started yet" and produces silence forever.
    no_driver_grace_s: float = 10.0

    # eyes
    microsleep_s: float = 1.5
    microsleep_exit_s: float = 0.20
    microsleep_cooldown_s: float = 10.0

    # perclos
    perclos_warn: float = 0.20
    perclos_crit: float = 0.40
    perclos_exit_margin: float = 0.05
    perclos_enter_s: float = 2.0
    perclos_exit_s: float = 10.0
    perclos_cooldown_s: float = 30.0

    # mouth
    yawn_enter_s: float = 0.40
    yawn_exit_s: float = 0.0
    yawn_cooldown_s: float = 15.0
    fatigue_yawns: int = 3
    fatigue_window_s: float = 180.0
    fatigue_exit_s: float = 0.0
    fatigue_cooldown_s: float = 30.0

    # gaze
    gaze_warn_s: float = 2.0
    gaze_crit_s: float = 4.0
    gaze_exit_s: float = 0.50
    gaze_cooldown_s: float = 5.0

    # objects
    phone_conf: float = 0.50
    phone_enter_s: float = 0.80
    phone_exit_s: float = 0.40
    phone_cooldown_s: float = 10.0

    # distraction roll-up
    distraction_gaze_s: float = 6.0
    distraction_phone_s: float = 1.5
    distraction_exit_s: float = 1.5
    distraction_cooldown_s: float = 20.0

    # honesty alerts
    unreliable_enter_s: float = 1.0
    unreliable_exit_s: float = 2.0
    unreliable_cooldown_s: float = 20.0
    degraded_enter_s: float = 2.0
    degraded_exit_s: float = 5.0
    degraded_cooldown_s: float = 60.0

    # global
    assume_moving: bool = True
    #: intervals longer than this are a gap; no dwell is integrated across them.
    max_gap_s: float = 0.5
    #: how fast a dwell accumulator bleeds off on a contrary sample, as a
    #: multiple of the interval.  1.0 = symmetric, <1.0 = more forgiving of
    #: noise.  Never 0.0: an accumulator that only rises can never be wrong.
    leak: float = 1.0

    _FLOAT_FIELDS = (
        "face_lost_s",
        "face_lost_exit_s",
        "face_lost_cooldown_s",
        "face_lost_crit_s",
        "no_driver_grace_s",
        "microsleep_s",
        "microsleep_exit_s",
        "microsleep_cooldown_s",
        "perclos_warn",
        "perclos_crit",
        "perclos_exit_margin",
        "perclos_enter_s",
        "perclos_exit_s",
        "perclos_cooldown_s",
        "yawn_enter_s",
        "yawn_exit_s",
        "yawn_cooldown_s",
        "fatigue_window_s",
        "fatigue_exit_s",
        "fatigue_cooldown_s",
        "gaze_warn_s",
        "gaze_crit_s",
        "gaze_exit_s",
        "gaze_cooldown_s",
        "phone_conf",
        "phone_enter_s",
        "phone_exit_s",
        "phone_cooldown_s",
        "distraction_gaze_s",
        "distraction_phone_s",
        "distraction_exit_s",
        "distraction_cooldown_s",
        "unreliable_enter_s",
        "unreliable_exit_s",
        "unreliable_cooldown_s",
        "degraded_enter_s",
        "degraded_exit_s",
        "degraded_cooldown_s",
        "max_gap_s",
        "leak",
    )

    @classmethod
    def from_app_config(cls, cfg: Any) -> "AlertTuning":
        out = cls()
        state = getattr(cfg, "state", None)
        if state is None:
            return out
        for name in cls._FLOAT_FIELDS:
            value = finite(getattr(state, name, None))
            if value is not None:
                setattr(out, name, value)
        n_yawns = getattr(state, "fatigue_yawns", None)
        if isinstance(n_yawns, int) and n_yawns > 0:
            out.fatigue_yawns = n_yawns
        moving = getattr(state, "assume_moving", None)
        if isinstance(moving, bool):
            out.assume_moving = moving
        if out.leak <= 0.0:
            out.leak = 1.0
        return out


@dataclass
class _Latch:
    """One alert's hysteresis state."""

    active: bool = False
    hold_true: float = 0.0
    hold_false: float = 0.0
    last_enter_s: float = -1e9
    severity: Severity = Severity.WARN


_LATCHED_TYPES = (
    AlertType.FACE_LOST,
    AlertType.MICROSLEEP,
    AlertType.PERCLOS_HIGH,
    AlertType.YAWN,
    AlertType.FATIGUE_CLUSTER,
    AlertType.GAZE_AWAY,
    AlertType.PHONE,
    AlertType.DISTRACTION,
    AlertType.DROWSINESS_UNRELIABLE,
    AlertType.MODEL_DEGRADED,
)


class DriverMonitor(object):
    """Fusion + hysteresis for one driver.

    Construct one per driver.  ``dms/app.py`` keeps a single instance and relies
    on :meth:`update` detecting a change of ``track_id`` or ``identity`` and
    resetting itself; either arrangement is safe.
    """

    def __init__(
        self,
        cfg: Any,
        tuning: Optional[AlertTuning] = None,
        fusion_cfg: Optional[FusionConfig] = None,
    ) -> None:
        self.cfg = cfg
        self.tuning = tuning or AlertTuning.from_app_config(cfg)
        self.fusion_cfg = fusion_cfg or FusionConfig.from_app_config(cfg)
        self.fuser = SignalFuser(self.fusion_cfg)
        self.latches: Dict[AlertType, _Latch] = {t: _Latch() for t in _LATCHED_TYPES}
        self.active: Dict[AlertType, Severity] = {}
        #: Per-capability honesty flags, mirrored onto /healthz.  Written every
        #: frame; ``True`` means "this signal must not be used for an alert".
        self.unreliable: Dict[str, bool] = {
            "eyes": True,
            "mouth": True,
            "pose": True,
            "gaze": True,
            "perclos": True,
        }
        self.track_id: Optional[int] = None
        self.identity: Optional[str] = None
        self.view: FusedView = FusedView()
        self._t_prev: Optional[float] = None
        self._t_first: Optional[float] = None
        self._t_last_match: Optional[float] = None
        self._present_hold = 0.0
        self._yawn_times: Deque[float] = deque()
        self._distract_gaze_s = 0.0
        self._distract_phone_s = 0.0

    # ------------------------------------------------------------ helpers --
    def highest_severity(self) -> Optional[Severity]:
        """Highest severity currently active, or ``None``.  Drives GPIO."""
        return max_severity(self.active.values())

    @property
    def perclos(self) -> float:
        """Time-weighted PERCLOS.  Only meaningful when :attr:`perclos_valid`."""
        return self.view.perclos

    @property
    def perclos_valid(self) -> bool:
        return self.view.perclos_valid

    # Legacy read-only views kept for ``dms/app.py`` and the overlay.  They are
    # smoothed ABSOLUTE angles (the caller still subtracts the forward zero).
    @property
    def yaw_ewma(self) -> Optional[float]:
        return self.fuser._yaw.value

    @property
    def pitch_ewma(self) -> Optional[float]:
        return self.fuser._pitch.value

    @property
    def ear_ewma(self) -> Optional[float]:
        if self.view.eyes.source is SignalSource.GEOMETRY:
            return self.view.eye_open_score
        return None

    @property
    def mar_ewma(self) -> Optional[float]:
        if self.view.mouth.source is SignalSource.GEOMETRY:
            return self.view.mouth_score
        return None

    def recalibrate(
        self, yaw: float, pitch: float, roll: float = 0.0, calibrated: bool = True
    ) -> None:
        """Install a new forward zero without rebuilding the monitor."""
        self.fusion_cfg.forward_yaw = finite(yaw, 0.0) or 0.0
        self.fusion_cfg.forward_pitch = finite(pitch, 0.0) or 0.0
        self.fusion_cfg.forward_roll = finite(roll, 0.0) or 0.0
        self.fusion_cfg.calibrated = bool(calibrated)

    def _sync_calibration(self) -> None:
        """Pick up a forward zero edited on ``cfg`` after construction.

        ``--calibrate-forward`` and the tests both mutate ``cfg.forward_zero``
        in place; three float comparisons per frame is cheaper than making the
        caller remember to rebuild the monitor.

        Only the forward zero is copied here.  The EAR/MAR thresholds do not
        need a sync step: :meth:`dms.state.fusion.SignalFuser._apply_cfg`
        re-reads the whole of ``fusion_cfg`` every frame, so editing
        ``fusion_cfg.ear_closed`` mid-stream moves the decision boundary on the
        very next frame.
        """
        zero = getattr(self.cfg, "forward_zero", None)
        if zero is None:
            return
        yaw = finite(getattr(zero, "yaw", 0.0), 0.0) or 0.0
        pitch = finite(getattr(zero, "pitch", 0.0), 0.0) or 0.0
        roll = finite(getattr(zero, "roll", 0.0), 0.0) or 0.0
        fc = self.fusion_cfg
        if yaw == fc.forward_yaw and pitch == fc.forward_pitch and roll == fc.forward_roll:
            return
        fc.forward_yaw, fc.forward_pitch, fc.forward_roll = yaw, pitch, roll
        fc.calibrated = not (abs(yaw) < 1e-9 and abs(pitch) < 1e-9 and abs(roll) < 1e-9)

    # -------------------------------------------------------------- update --
    def update(
        self,
        t_s: float,
        *,
        present: bool,
        track_id: Optional[int] = None,
        signals: Optional[DriverSignals] = None,
        degraded: Optional[Sequence[str]] = None,
        t_last_match: Optional[float] = None,
        ear: Optional[float] = None,
        mar: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> List[AlertEvent]:
        """Advance the machine one frame and return the edges it produced.

        ``signals`` is the production path and the only one ``dms/app.py``
        uses: both of its call sites pass ``signals=``, never raw floats.

        ``ear``/``mar``/``yaw``/``pitch``/``roll`` are the geometry-only path,
        kept for a profile configured with no model engines and for tests that
        want to drive the machine from four numbers.  They are wrapped into a
        :class:`DriverSignals` with ``source=GEOMETRY`` and a fixed nominal pose
        confidence (``FusionConfig.legacy_pose_conf``), because raw floats carry
        no quality metric of their own - which is why it is the weaker path, not
        a deprecated one.  ``ear`` and ``mar`` must be on the CORRECTED
        roll-invariant scales (``dms.geometry.ear`` / ``dms.geometry.mar``): the
        same thresholds apply as on the ``signals`` path.  Passing both uses
        ``signals`` and ignores the floats.

        ``degraded`` is the list of capabilities that asked for a model and did
        not get one (``ModelRegistry.degraded_capabilities()``); it drives
        ``MODEL_DEGRADED`` and nothing else.

        ``t_last_match`` is the last instant a MATCHED detection sat in the seat
        ROI, as reported by :attr:`dms.track.driver_select.DriverSelector.t_last_match_s`.
        FACE_LOST is defined on it, so passing it makes the alert independent of
        whatever the caller decided ``present`` means.  Omitted, it is derived
        from ``present``, which is equivalent as long as ``present`` is itself
        "matched in the ROI this frame" and not "a track id still exists".
        """
        self._sync_calibration()
        tun = self.tuning
        now = finite(t_s, 0.0) or 0.0
        if self._t_first is None:
            self._t_first = now
        if self._t_prev is None:
            dt = 0.0
        else:
            dt = now - self._t_prev
            if dt < 0.0 or dt > tun.max_gap_s:
                dt = 0.0
        self._t_prev = now

        if signals is None:
            signals = self._signals_from_geometry(track_id, ear, mar, yaw, pitch, roll)
        present = bool(present)

        events: List[AlertEvent] = []
        events += self._check_driver_change(now, present, track_id, signals)

        tid = self._effective_track_id(track_id, signals)
        if present and tid is not None:
            self.track_id = tid
        if present and signals.identity:
            self.identity = signals.identity

        view = self.fuser.update(now, present, signals)
        self.view = view
        self.unreliable = {
            "eyes": not view.eyes.usable,
            "mouth": not view.mouth.usable,
            "pose": not view.pose.usable,
            "gaze": not view.gaze.usable,
            "perclos": not view.perclos_valid,
        }

        given = finite(t_last_match)
        if given is None:
            matched = present
            if present:
                self._t_last_match = now
        else:
            # An explicit t_last_match is AUTHORITATIVE.  A caller that keeps a
            # coasting track alive and reports present=True must not be able to
            # hide a lost face - that is the whole point of the design defining
            # FACE_LOST on the last matched detection in the seat ROI.
            self._t_last_match = given
            matched = (now - given) <= max(dt, 1e-6) + 1e-9
        if matched:
            self._present_hold += dt
        else:
            self._present_hold = 0.0

        events += self._face_lost(now, dt, view)
        events += self._microsleep(now, dt, view)
        events += self._perclos_high(now, dt, view)
        events += self._yawn(now, dt, view)
        events += self._fatigue_cluster(now, dt, view, events)
        events += self._gaze_away(now, dt, view)
        events += self._phone(now, dt, view)
        events += self._distraction(now, dt, view)
        events += self._drowsiness_unreliable(now, dt, view)
        events += self._model_degraded(now, dt, degraded)
        return events

    # ------------------------------------------------------- legacy bridge --
    def _signals_from_geometry(
        self,
        track_id: Optional[int],
        ear: Optional[float],
        mar: Optional[float],
        yaw: Optional[float],
        pitch: Optional[float],
        roll: Optional[float],
    ) -> DriverSignals:
        """Wrap raw geometry floats as an honest GEOMETRY-sourced DriverSignals."""
        ear_v = finite(ear)
        mar_v = finite(mar)
        yaw_v = finite(yaw)
        pitch_v = finite(pitch)
        roll_v = finite(roll, 0.0) or 0.0
        eye_src = SignalSource.GEOMETRY if ear_v is not None else SignalSource.UNAVAILABLE
        mouth_src = SignalSource.GEOMETRY if mar_v is not None else SignalSource.UNAVAILABLE
        pose_src = (
            SignalSource.GEOMETRY
            if (yaw_v is not None and pitch_v is not None)
            else SignalSource.UNAVAILABLE
        )
        return DriverSignals(
            track_id=-1 if track_id is None else int(track_id),
            left_eye=EyeState(ear=ear_v or 0.0, source=eye_src),
            right_eye=EyeState(ear=ear_v or 0.0, source=eye_src),
            mouth=MouthState(mar=mar_v or 0.0, source=mouth_src),
            pose=HeadPose(
                yaw=yaw_v or 0.0,
                pitch=pitch_v or 0.0,
                roll=roll_v,
                confidence=self.fusion_cfg.legacy_pose_conf if pose_src is SignalSource.GEOMETRY else 0.0,
                reprojection_error=0.0,
                source=pose_src,
            ),
            gaze=GazeVector(),
            occlusion=Occlusion(),
        )

    @staticmethod
    def _effective_track_id(
        track_id: Optional[int], signals: DriverSignals
    ) -> Optional[int]:
        if track_id is not None:
            return int(track_id)
        sig_id = getattr(signals, "track_id", -1)
        if isinstance(sig_id, int) and sig_id >= 0:
            return sig_id
        return None

    # ------------------------------------------------------- driver change --
    def _check_driver_change(
        self,
        now: float,
        present: bool,
        track_id: Optional[int],
        signals: DriverSignals,
    ) -> List[AlertEvent]:
        """Detect a driver swap and wipe every window before it can be stitched.

        Triggered by a change of track id or, when the identity embedder is
        running, by a change of identity even under a stable track id (a swap
        during a stop the tracker coasted through).  A first assignment is not a
        change.
        """
        if not present:
            return []
        tid = self._effective_track_id(track_id, signals)
        identity = signals.identity
        changed = False
        if tid is not None and self.track_id is not None and tid != self.track_id:
            changed = True
        if identity and self.identity and identity != self.identity:
            changed = True
        if not changed:
            return []

        events: List[AlertEvent] = []
        for typ in _LATCHED_TYPES:
            lat = self.latches[typ]
            if lat.active:
                events.append(
                    AlertEvent(now, typ, lat.severity, EDGE_EXIT, self.track_id, {"driver_changed": 1.0})
                )
            self.latches[typ] = _Latch()
        self.active.clear()
        self.fuser.reset(keep_clock=True)
        self._yawn_times.clear()
        self._distract_gaze_s = 0.0
        self._distract_phone_s = 0.0
        self._present_hold = 0.0
        extra = {
            "from_track": float(self.track_id if self.track_id is not None else -1),
            "to_track": float(tid if tid is not None else -1),
        }
        # DRIVER_CHANGED is momentary: it describes an instant, not a condition,
        # so both edges carry the same timestamp and the log still pairs.
        events.append(AlertEvent(now, AlertType.DRIVER_CHANGED, Severity.INFO, EDGE_ENTER, tid, dict(extra)))
        events.append(AlertEvent(now, AlertType.DRIVER_CHANGED, Severity.INFO, EDGE_EXIT, tid, dict(extra)))
        self.track_id = tid
        self.identity = identity
        return events

    # ------------------------------------------------------------- alerts ---
    def _face_lost(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        if self._t_last_match is None:
            # Never saw a face.  Silence here would make a lens blocked at boot
            # look identical to a healthy system, so arm on a longer grace.
            grace = max(tun.face_lost_s, tun.no_driver_grace_s)
            since = now - (self._t_first if self._t_first is not None else now)
            lost = since >= grace
        else:
            since = now - self._t_last_match
            lost = since >= tun.face_lost_s
        lat = self.latches[AlertType.FACE_LOST]
        held = lat.hold_true + (dt if lost else 0.0)
        sev = Severity.WARN
        if tun.assume_moving and (
            held >= tun.face_lost_crit_s - DWELL_EPS or tun.face_lost_crit_s <= 0.0
        ):
            sev = Severity.CRITICAL
        return self._step(
            AlertType.FACE_LOST,
            now,
            dt,
            enter_cond=lost,
            exit_cond=self._present_hold >= tun.face_lost_exit_s,
            enter_s=0.0,
            exit_s=0.0,
            cooldown_s=tun.face_lost_cooldown_s,
            sev=sev,
            extra={"since_match_s": round(since, 3)},
        )

    def _microsleep(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        usable = view.present and view.eyes.usable
        closed = usable and view.eye_closed
        return self._step(
            AlertType.MICROSLEEP,
            now,
            dt,
            enter_cond=closed,
            exit_cond=not closed,
            enter_s=tun.microsleep_s,
            exit_s=tun.microsleep_exit_s,
            cooldown_s=tun.microsleep_cooldown_s,
            sev=Severity.CRITICAL,
            extra={"eye_score": round(view.eye_open_score, 4)},
        )

    def _perclos_high(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        level: Optional[Severity] = None
        if view.perclos_valid:
            if view.perclos >= tun.perclos_crit:
                level = Severity.CRITICAL
            elif view.perclos >= tun.perclos_warn:
                level = Severity.WARN
        lat = self.latches[AlertType.PERCLOS_HIGH]
        below = view.perclos < (tun.perclos_warn - tun.perclos_exit_margin)
        return self._step(
            AlertType.PERCLOS_HIGH,
            now,
            dt,
            enter_cond=level is not None,
            exit_cond=(not view.perclos_valid) or below,
            enter_s=tun.perclos_enter_s,
            exit_s=tun.perclos_exit_s,
            cooldown_s=tun.perclos_cooldown_s,
            sev=level or lat.severity,
            extra={
                "perclos": round(view.perclos, 4),
                "rate_hz": round(view.perclos_rate_hz, 2),
                "valid": 1.0 if view.perclos_valid else 0.0,
            },
        )

    def _yawn(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        # A mask makes the mouth unmeasurable; suppressing YAWN is the honest
        # answer, and DROWSINESS_UNRELIABLE carries the fact that we are blind.
        usable = view.present and view.mouth.usable
        yawning = usable and view.mouth_open
        return self._step(
            AlertType.YAWN,
            now,
            dt,
            enter_cond=yawning,
            exit_cond=not yawning,
            enter_s=tun.yawn_enter_s,
            exit_s=tun.yawn_exit_s,
            cooldown_s=tun.yawn_cooldown_s,
            sev=Severity.INFO,
            extra={"mouth_score": round(view.mouth_score, 4)},
        )

    def _fatigue_cluster(
        self, now: float, dt: float, view: FusedView, so_far: Sequence[AlertEvent]
    ) -> List[AlertEvent]:
        tun = self.tuning
        for ev in so_far:
            if ev.type is AlertType.YAWN and ev.edge == EDGE_ENTER:
                self._yawn_times.append(ev.t_mono_s)
        while self._yawn_times and self._yawn_times[0] < now - tun.fatigue_window_s:
            self._yawn_times.popleft()
        n = len(self._yawn_times)
        return self._step(
            AlertType.FATIGUE_CLUSTER,
            now,
            dt,
            enter_cond=n >= tun.fatigue_yawns,
            exit_cond=n < tun.fatigue_yawns,
            enter_s=0.0,
            exit_s=tun.fatigue_exit_s,
            cooldown_s=tun.fatigue_cooldown_s,
            sev=Severity.WARN,
            extra={"yawns": float(n)},
        )

    def _gaze_away(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        # `view.gaze.usable` is False whenever the pose is missing, below its
        # confidence gate, or the vehicle has no forward zero - so an
        # uncalibrated rig can never raise a confident gaze alert.
        away = view.present and view.gaze.usable and view.gaze_off_road
        lat = self.latches[AlertType.GAZE_AWAY]
        held = lat.hold_true + (dt if away else 0.0)
        sev = Severity.CRITICAL if held >= tun.gaze_crit_s - DWELL_EPS else Severity.WARN
        return self._step(
            AlertType.GAZE_AWAY,
            now,
            dt,
            enter_cond=away,
            exit_cond=not away,
            enter_s=tun.gaze_warn_s,
            exit_s=tun.gaze_exit_s,
            cooldown_s=tun.gaze_cooldown_s,
            sev=sev,
            extra={"gaze_yaw": round(view.gaze_yaw, 2), "gaze_pitch": round(view.gaze_pitch, 2)},
        )

    def _phone(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        tun = self.tuning
        seen = view.present and view.phone_conf >= tun.phone_conf
        return self._step(
            AlertType.PHONE,
            now,
            dt,
            enter_cond=seen,
            exit_cond=not seen,
            enter_s=tun.phone_enter_s,
            exit_s=tun.phone_exit_s,
            cooldown_s=tun.phone_cooldown_s,
            sev=Severity.WARN,
            extra={"phone_conf": round(view.phone_conf, 4)},
        )

    def _distraction(self, now: float, dt: float, view: FusedView) -> List[AlertEvent]:
        """Behavioural roll-up: sustained off-road gaze OR a phone in the cabin.

        GAZE_AWAY answers "is the driver looking away right now"; DISTRACTION
        answers "has the driver been disengaged long enough to matter", which is
        a longer dwell and a different operator response.  The two accumulators
        are kept separately so a driver who alternates between the phone and the
        road does not reset both.
        """
        tun = self.tuning
        away = view.present and view.gaze.usable and view.gaze_off_road
        phone = view.present and view.phone_conf >= tun.phone_conf
        leak = tun.leak * dt
        self._distract_gaze_s = (
            self._distract_gaze_s + dt if away else max(0.0, self._distract_gaze_s - leak)
        )
        self._distract_phone_s = (
            self._distract_phone_s + dt if phone else max(0.0, self._distract_phone_s - leak)
        )
        distracted = (
            self._distract_gaze_s >= tun.distraction_gaze_s - DWELL_EPS
            or self._distract_phone_s >= tun.distraction_phone_s - DWELL_EPS
        )
        return self._step(
            AlertType.DISTRACTION,
            now,
            dt,
            enter_cond=distracted,
            exit_cond=not (away or phone),
            enter_s=0.0,
            exit_s=tun.distraction_exit_s,
            cooldown_s=tun.distraction_cooldown_s,
            sev=Severity.WARN,
            extra={
                "gaze_s": round(self._distract_gaze_s, 2),
                "phone_s": round(self._distract_phone_s, 2),
            },
        )

    def _drowsiness_unreliable(
        self, now: float, dt: float, view: FusedView
    ) -> List[AlertEvent]:
        """Say so when the drowsiness function cannot be performed.

        Raised while a driver IS present but one of the signals the drowsiness
        alerts are built on cannot be trusted: the eye channel is occluded or
        missing, the head pose is missing / below its confidence gate /
        uncalibrated, or the pipeline is too slow for a valid PERCLOS window.
        Warm-up (the first ``perclos_min_span_s``) is not a fault and is
        excluded - only a window that is long enough and still invalid counts.
        """
        eyes_bad = view.present and not view.eyes.usable
        pose_bad = view.present and not view.pose.usable
        perclos_bad = view.present and view.perclos_starved
        bad = eyes_bad or pose_bad or perclos_bad
        tun = self.tuning
        return self._step(
            AlertType.DROWSINESS_UNRELIABLE,
            now,
            dt,
            enter_cond=bad,
            exit_cond=not bad,
            enter_s=tun.unreliable_enter_s,
            exit_s=tun.unreliable_exit_s,
            cooldown_s=tun.unreliable_cooldown_s,
            sev=Severity.WARN,
            extra={
                "eyes": 1.0 if eyes_bad else 0.0,
                "pose": 1.0 if pose_bad else 0.0,
                "perclos": 1.0 if perclos_bad else 0.0,
                "sunglasses": round(view.sunglasses, 3),
                "mask": round(view.mask, 3),
            },
        )

    def _model_degraded(
        self, now: float, dt: float, degraded: Optional[Sequence[str]]
    ) -> List[AlertEvent]:
        if degraded is None:
            return []
        n = len(degraded)
        tun = self.tuning
        return self._step(
            AlertType.MODEL_DEGRADED,
            now,
            dt,
            enter_cond=n > 0,
            exit_cond=n == 0,
            enter_s=tun.degraded_enter_s,
            exit_s=tun.degraded_exit_s,
            cooldown_s=tun.degraded_cooldown_s,
            sev=Severity.WARN,
            extra={"capabilities": float(n)},
        )

    # --------------------------------------------------------------- latch --
    def _step(
        self,
        typ: AlertType,
        t_s: float,
        dt: float,
        *,
        enter_cond: bool,
        exit_cond: bool,
        enter_s: float,
        exit_s: float,
        cooldown_s: float,
        sev: Severity,
        extra: Optional[Dict[str, float]] = None,
    ) -> List[AlertEvent]:
        """Advance one alert latch and emit at most one edge.

        The accumulators leak by ``tuning.leak * dt`` on a contrary sample
        instead of resetting, which is what lets a 1.5 s eye closure survive one
        noisy frame at 8 FPS.  ``cooldown_s`` is measured from the previous
        ENTER, never from the previous exit.

        Known and deliberate bias: the interval that STRADDLES the moment a
        condition became true is credited to the condition in full, so an alert
        fires up to one frame interval early (167 ms at 6 FPS on a 1.5 s
        threshold).  Erring early is the safe direction for a warning, and the
        alternative - crediting the straddling interval to the previous state -
        is exactly as wrong in the unsafe direction.
        """
        lat = self.latches[typ]
        leak = self.tuning.leak * dt
        if enter_cond:
            lat.hold_true += dt
        else:
            lat.hold_true = max(0.0, lat.hold_true - leak)
        if exit_cond:
            lat.hold_false += dt
        else:
            lat.hold_false = max(0.0, lat.hold_false - leak)

        out: List[AlertEvent] = []
        payload = dict(extra or {})
        tid = self.track_id

        if not lat.active:
            ready = lat.hold_true >= enter_s - DWELL_EPS
            cooled = (t_s - lat.last_enter_s) >= cooldown_s - DWELL_EPS
            if enter_cond and ready and cooled:
                lat.active = True
                lat.severity = sev
                lat.last_enter_s = t_s
                lat.hold_false = 0.0
                self.active[typ] = sev
                out.append(AlertEvent(t_s, typ, sev, EDGE_ENTER, tid, payload))
            return out

        if severity_rank(sev) > severity_rank(lat.severity):
            lat.severity = sev
            self.active[typ] = sev
            out.append(AlertEvent(t_s, typ, sev, EDGE_ESCALATE, tid, dict(payload)))
        if exit_cond and lat.hold_false >= exit_s - DWELL_EPS:
            lat.active = False
            lat.hold_true = 0.0
            self.active.pop(typ, None)
            out.append(AlertEvent(t_s, typ, lat.severity, EDGE_EXIT, tid, dict(payload)))
        return out
