from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from dms.config.schema import AppConfig
from dms.state.alerts import AlertEvent, AlertType, Severity


def angdiff(a: float, b: float) -> float:
    d = a - b
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


@dataclass
class _Latch:
    active: bool = False
    hold_true: float = 0.0
    hold_false: float = 0.0
    last_edge_s: float = -1e9
    severity: Severity = Severity.WARN


class DriverMonitor:
    """Per-driver EWMA + hysteresis. dt is capture-time between processed frames."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        st = cfg.state
        self.alpha_ear = 0.30
        self.alpha_pose = 0.20
        self.ear_ewma: Optional[float] = None
        self.mar_ewma: Optional[float] = None
        self.yaw_ewma: Optional[float] = None
        self.pitch_ewma: Optional[float] = None
        self.perclos = 0.0
        self._closed_win: Deque[Tuple[float, bool]] = deque()
        self._t_prev: Optional[float] = None
        self._t_last_present: Optional[float] = None
        self._t_present_hold = 0.0
        self._yawn_times: Deque[float] = deque()
        self.latches: Dict[AlertType, _Latch] = {
            AlertType.FACE_LOST: _Latch(),
            AlertType.MICROSLEEP: _Latch(),
            AlertType.PERCLOS_HIGH: _Latch(),
            AlertType.YAWN: _Latch(),
            AlertType.FATIGUE_CLUSTER: _Latch(),
            AlertType.GAZE_AWAY: _Latch(),
        }
        self.active: Dict[AlertType, Severity] = {}
        self.unreliable: Dict[str, bool] = {}

    def highest_severity(self) -> Optional[Severity]:
        order = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}
        if not self.active:
            return None
        return max(self.active.values(), key=lambda s: order[s])

    def update(
        self,
        t_s: float,
        *,
        present: bool,
        track_id: Optional[int],
        ear: Optional[float],
        mar: Optional[float],
        yaw: Optional[float],
        pitch: Optional[float],
    ) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        dt = 0.0 if self._t_prev is None else max(0.0, min(0.5, t_s - self._t_prev))
        self._t_prev = t_s
        st = self.cfg.state
        z = self.cfg.forward_zero

        if ear is not None:
            self.ear_ewma = ear if self.ear_ewma is None else self.alpha_ear * ear + (1 - self.alpha_ear) * self.ear_ewma
        if mar is not None:
            self.mar_ewma = mar if self.mar_ewma is None else self.alpha_ear * mar + (1 - self.alpha_ear) * self.mar_ewma
        if yaw is not None:
            self.yaw_ewma = yaw if self.yaw_ewma is None else self.alpha_pose * yaw + (1 - self.alpha_pose) * self.yaw_ewma
        if pitch is not None:
            self.pitch_ewma = (
                pitch if self.pitch_ewma is None else self.alpha_pose * pitch + (1 - self.alpha_pose) * self.pitch_ewma
            )

        if present:
            self._t_last_present = t_s
            self._t_present_hold += dt
        else:
            self._t_present_hold = 0.0

        closed = False
        if self.ear_ewma is not None:
            thr = st.ear_closed
            if st.ear_open_median > 0:
                thr = min(thr, 0.70 * st.ear_open_median)
            closed = self.ear_ewma < thr
        self._closed_win.append((t_s, closed))
        cutoff = t_s - 60.0
        while self._closed_win and self._closed_win[0][0] < cutoff:
            self._closed_win.popleft()
        if self._closed_win:
            n_c = sum(1 for _, c in self._closed_win if c)
            self.perclos = n_c / float(len(self._closed_win))
        else:
            self.perclos = 0.0

        yaw_rel = pitch_rel = None
        if self.yaw_ewma is not None:
            yaw_rel = angdiff(self.yaw_ewma, z.yaw)
        if self.pitch_ewma is not None:
            pitch_rel = angdiff(self.pitch_ewma, z.pitch)
        gaze_away = False
        if yaw_rel is not None and pitch_rel is not None:
            gaze_away = abs(yaw_rel) >= st.gaze_yaw_deg or abs(pitch_rel) >= st.gaze_pitch_deg

        lost = (
            (not present)
            and self._t_last_present is not None
            and (t_s - self._t_last_present) >= st.face_lost_s
        )
        lost_exit = present and self._t_present_hold >= 0.4

        events += self._step(
            AlertType.FACE_LOST,
            t_s,
            dt,
            enter_cond=lost,
            exit_cond=lost_exit,
            enter_s=0.0,
            exit_s=0.0,
            cooldown_s=5.0,
            sev=Severity.CRITICAL if st.assume_moving else Severity.WARN,
            track_id=track_id,
        )
        events += self._step(
            AlertType.MICROSLEEP,
            t_s,
            dt,
            enter_cond=closed and present,
            exit_cond=(not closed) and present,
            enter_s=st.microsleep_s,
            exit_s=0.20,
            cooldown_s=10.0,
            sev=Severity.CRITICAL,
            track_id=track_id,
            extra={"ear": self.ear_ewma or 0.0},
        )
        perclos_level = None
        if self.perclos >= st.perclos_crit:
            perclos_level = Severity.CRITICAL
        elif self.perclos >= st.perclos_warn:
            perclos_level = Severity.WARN
        events += self._step(
            AlertType.PERCLOS_HIGH,
            t_s,
            dt,
            enter_cond=perclos_level is not None,
            exit_cond=self.perclos < (st.perclos_warn - 0.05),
            enter_s=0.0,
            exit_s=10.0,
            cooldown_s=30.0,
            sev=perclos_level or Severity.WARN,
            track_id=track_id,
            extra={"perclos": self.perclos},
        )
        yawn_now = present and self.mar_ewma is not None and self.mar_ewma >= st.mar_yawn
        events += self._step(
            AlertType.YAWN,
            t_s,
            dt,
            enter_cond=yawn_now,
            exit_cond=self.mar_ewma is not None and self.mar_ewma < 0.50,
            enter_s=0.40,
            exit_s=0.0,
            cooldown_s=15.0,
            sev=Severity.INFO,
            track_id=track_id,
            extra={"mar": self.mar_ewma or 0.0},
        )
        for ev in events:
            if ev.type == AlertType.YAWN and ev.edge == "enter":
                self._yawn_times.append(t_s)
        while self._yawn_times and self._yawn_times[0] < t_s - 180.0:
            self._yawn_times.popleft()
        cluster = len(self._yawn_times) >= 3
        events += self._step(
            AlertType.FATIGUE_CLUSTER,
            t_s,
            dt,
            enter_cond=cluster,
            exit_cond=len(self._yawn_times) < 3,
            enter_s=0.0,
            exit_s=0.0,
            cooldown_s=30.0,
            sev=Severity.WARN,
            track_id=track_id,
        )
        gaze_enter_s = st.gaze_warn_s
        gaze_sev = Severity.WARN
        if self.latches[AlertType.GAZE_AWAY].hold_true >= st.gaze_crit_s or (
            self.latches[AlertType.GAZE_AWAY].active
            and self.latches[AlertType.GAZE_AWAY].hold_true >= st.gaze_crit_s
        ):
            gaze_sev = Severity.CRITICAL
        events += self._step(
            AlertType.GAZE_AWAY,
            t_s,
            dt,
            enter_cond=gaze_away and present,
            exit_cond=(not gaze_away) and present,
            enter_s=gaze_enter_s,
            exit_s=0.50,
            cooldown_s=5.0,
            sev=gaze_sev,
            track_id=track_id,
            extra={"yaw_rel": yaw_rel or 0.0, "pitch_rel": pitch_rel or 0.0},
        )
        # escalate gaze to critical if already active past crit duration
        if AlertType.GAZE_AWAY in self.active and self.latches[AlertType.GAZE_AWAY].hold_true >= st.gaze_crit_s:
            self.active[AlertType.GAZE_AWAY] = Severity.CRITICAL
            self.latches[AlertType.GAZE_AWAY].severity = Severity.CRITICAL
        return events

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
        track_id: Optional[int],
        extra: Optional[Dict[str, float]] = None,
    ) -> List[AlertEvent]:
        lat = self.latches[typ]
        if enter_cond:
            lat.hold_true += dt
            lat.hold_false = 0.0
        else:
            lat.hold_true = 0.0 if not lat.active else lat.hold_true
            if exit_cond or not enter_cond:
                lat.hold_false += dt
                if not lat.active:
                    lat.hold_true = 0.0
        out: List[AlertEvent] = []
        extra = extra or {}
        if (not lat.active) and enter_cond and lat.hold_true >= enter_s:
            if (t_s - lat.last_edge_s) >= cooldown_s:
                lat.active = True
                lat.severity = sev
                lat.last_edge_s = t_s
                self.active[typ] = sev
                out.append(AlertEvent(t_s, typ, sev, "enter", track_id, extra))
        if lat.active and exit_cond and lat.hold_false >= exit_s:
            lat.active = False
            lat.hold_true = 0.0
            lat.last_edge_s = t_s
            self.active.pop(typ, None)
            out.append(AlertEvent(t_s, typ, lat.severity, "exit", track_id, extra))
        return out
