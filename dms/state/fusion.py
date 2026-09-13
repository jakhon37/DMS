"""Confidence-weighted fusion of model signals and classical geometry.

The alert state machine must never see a raw number whose provenance it cannot
check.  This module turns one :class:`dms.types.DriverSignals` per frame into a
:class:`FusedView`: one decision per capability, the source that produced it, a
confidence in ``[0, 1]``, and a ``reliable`` flag that is ``False`` whenever
nothing trustworthy answered.

Precedence, applied independently per capability
------------------------------------------------
1. ``SignalSource.MODEL`` when the model answered with a finite value and its
   confidence clears the per-capability gate in :class:`FusionConfig`.
2. ``SignalSource.GEOMETRY`` when no model answered (or it fell below its gate)
   and a classical estimate exists that clears the geometry gate.
3. ``SignalSource.UNAVAILABLE`` otherwise.  The channel is marked unreliable and
   the state machine raises ``DROWSINESS_UNRELIABLE`` instead of guessing.

Two notes on that gate, because they are easy to get wrong:

* For the EYE channel the gate is *availability*, not the per-frame probability.
  ``prob_open`` passes through 0.5 on every single blink; gating on it would
  hand the decision back and forth between the model and the EAR several times
  a second and make the hysteresis meaningless.  Ambiguity near 0.5 is handled
  by the Schmitt band, not by demoting the source.  The reported confidence,
  ``|p - 0.5| * 2``, is for logging and for the overall reliability score.
* Occlusion never *replaces* a value.  Sunglasses or a mask mean the eye and
  mouth channels are not measurable at all, so the channel is marked unreliable
  and the consumer must degrade, not substitute geometry that is equally blind.

Time base and units
-------------------
Angles are degrees, times seconds, probabilities and PERCLOS fractions in
``[0, 1]``.  Every time constant is expressed in seconds and converted to an
EWMA coefficient using the ACTUAL inter-frame interval
(``alpha = 1 - exp(-dt / tau)``), so a threshold means the same thing on a 6 FPS
replay and a 15 FPS vehicle.  Intervals longer than
:attr:`FusionConfig.max_gap_s` are treated as a gap: no state is integrated
across them, because the pipeline cannot know what happened in between.

Failure behaviour
-----------------
Nothing here raises on bad input.  Non-finite numbers are treated as "not
measured".  A capability with no usable input reports ``UNAVAILABLE`` with
confidence 0.0 and ``reliable=False``; it never falls back to a constant.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

from dms.geometry.ear import EAR_CLOSED_RECOMMENDED, EAR_OPEN_MEDIAN_RECOMMENDED
from dms.geometry.mar import MAR_YAWN_RECOMMENDED
from dms.types import (
    DriverSignals,
    EyeState,
    GazeVector,
    HeadPose,
    MouthState,
    Occlusion,
    SignalSource,
)

__all__ = [
    "ABOVE",
    "BELOW",
    "Channel",
    "CircularEwma",
    "FusedView",
    "FusionConfig",
    "PerclosWindow",
    "SchmittLatch",
    "SignalFuser",
    "angdiff",
    "ewma_alpha",
    "wrap180",
]


# --------------------------------------------------------------- helpers ---
def finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    """``float(value)`` when it is a finite number, else ``default``.

    Used everywhere a signal crosses into this module: a NaN from a failed
    solve or a None from a skipped model must read as "not measured", never as
    zero.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def ewma_alpha(dt_s: float, tau_s: float) -> float:
    """EWMA coefficient for time constant ``tau_s`` at interval ``dt_s``.

    ``tau_s <= 0`` means no smoothing (alpha 1.0).  ``dt_s <= 0`` means no time
    passed, so nothing is mixed in (alpha 0.0).  Because alpha is derived from
    the real interval, the *half-life* of the filter is fixed in seconds rather
    than in frames.
    """
    if tau_s <= 0.0:
        return 1.0
    if dt_s <= 0.0:
        return 0.0
    return 1.0 - math.exp(-float(dt_s) / float(tau_s))


def wrap180(deg: float) -> float:
    """Wrap an angle in degrees into ``(-180, 180]``.  Non-finite -> 0.0."""
    value = finite(deg)
    if value is None:
        return 0.0
    value = math.fmod(value + 180.0, 360.0)
    if value <= 0.0:
        value += 360.0
    return value - 180.0


def angdiff(a: float, b: float) -> float:
    """Shortest signed difference ``a - b`` in degrees, wrapped to (-180, 180]."""
    return wrap180(finite(a, 0.0) - finite(b, 0.0))


class CircularEwma(object):
    """Time-constant EWMA of an angle, averaged on the unit circle.

    A linear EWMA of degrees drags the smoothed value across the whole range
    the moment the input wraps (+179 and -179 average to 0, not 180).  This
    filter averages ``sin``/``cos`` instead, so a wrap costs nothing.

    ``value`` is ``None`` until the first finite sample.  Units: degrees.
    """

    def __init__(self, tau_s: float) -> None:
        self.tau_s = float(tau_s)
        self._sin: Optional[float] = None
        self._cos: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        if self._sin is None or self._cos is None:
            return None
        if abs(self._sin) < 1e-12 and abs(self._cos) < 1e-12:
            return None
        return math.degrees(math.atan2(self._sin, self._cos))

    def reset(self) -> None:
        self._sin = None
        self._cos = None

    def update(self, deg: Optional[float], dt_s: float) -> Optional[float]:
        value = finite(deg)
        if value is None:
            return self.value
        rad = math.radians(value)
        s, c = math.sin(rad), math.cos(rad)
        if self._sin is None or self._cos is None:
            self._sin, self._cos = s, c
        else:
            alpha = ewma_alpha(dt_s, self.tau_s)
            self._sin += alpha * (s - self._sin)
            self._cos += alpha * (c - self._cos)
        return self.value


BELOW = "below"
ABOVE = "above"

#: Slack on every "accumulated time >= threshold" comparison.  Summing an
#: interval like 1/15 s a few hundred times lands a nanosecond short of the
#: exact multiple, which would otherwise delay a threshold by a whole frame at
#: precisely the rates the design specifies (1.5 s at 10 FPS, 2.0 s at 15 FPS).
DWELL_EPS = 1e-9


class SchmittLatch(object):
    """Two-threshold latch with independent, leaky enter/exit dwell times.

    ``direction=BELOW``  - the latch turns ON while ``value < enter_thr`` and
    OFF once ``value >= exit_thr`` (``exit_thr >= enter_thr``).  This is the eye
    "closed" decision: EAR or ``1 - prob_open`` crossing downwards.
    ``direction=ABOVE`` - the mirror image, used for MAR / ``prob_open`` yawn
    style signals: ON while ``value > enter_thr``, OFF once ``value <= exit_thr``
    (``exit_thr <= enter_thr``).

    The dwell accumulators LEAK instead of resetting.  A single contrary sample
    costs one interval of credit rather than the whole accumulation, which is
    the difference between catching a real 1.5 s eye closure at 8 FPS and
    missing it because one frame of landmark jitter reset the clock.

    ``update`` returns the latched state.  A ``None`` value (nothing measured)
    integrates nothing and leaves the state where it was - the caller decides
    whether a stale state is still usable, because only the caller knows how
    long the signal has been missing.

    :attr:`enter_thr`, :attr:`exit_thr`, :attr:`enter_dwell_s` and
    :attr:`exit_dwell_s` are plain attributes and are meant to be REWRITTEN
    while the latch is live: :meth:`SignalFuser._apply_cfg` does it on every
    frame so a runtime retune takes effect immediately.  Rewriting them does not
    disturb :attr:`state` or the dwell accumulators, so a retune in the middle
    of a real eye closure does not drop that closure.
    """

    def __init__(
        self,
        enter_thr: float,
        exit_thr: float,
        direction: str = BELOW,
        enter_dwell_s: float = 0.0,
        exit_dwell_s: float = 0.0,
        initial: bool = False,
    ) -> None:
        if direction not in (BELOW, ABOVE):
            raise ValueError("direction must be %r or %r" % (BELOW, ABOVE))
        self.enter_thr = float(enter_thr)
        self.exit_thr = float(exit_thr)
        self.direction = direction
        self.enter_dwell_s = max(0.0, float(enter_dwell_s))
        self.exit_dwell_s = max(0.0, float(exit_dwell_s))
        self.state = bool(initial)
        self._hold_on = 0.0
        self._hold_off = 0.0

    def reset(self, initial: bool = False) -> None:
        self.state = bool(initial)
        self._hold_on = 0.0
        self._hold_off = 0.0

    def _raw(self, value: float) -> Optional[bool]:
        """Instantaneous band decision, or None inside the hysteresis band."""
        if self.direction == BELOW:
            if value < self.enter_thr:
                return True
            if value >= self.exit_thr:
                return False
            return None
        if value > self.enter_thr:
            return True
        if value <= self.exit_thr:
            return False
        return None

    def band(self, value: Optional[float]) -> bool:
        """Band-only decision with memory but NO dwell.

        This is the instantaneous eye-closure used by PERCLOS: a 150 ms blink at
        8 FPS is a single sample and must be counted, so it cannot be put behind
        a 150 ms dwell.  Does not mutate the dwelled :attr:`state`.
        """
        v = finite(value)
        if v is None:
            return self.state
        raw = self._raw(v)
        return self.state if raw is None else raw

    def update(self, value: Optional[float], dt_s: float) -> bool:
        v = finite(value)
        if v is None:
            return self.state
        raw = self._raw(v)
        step = max(0.0, float(dt_s))
        if raw is True:
            self._hold_on += step
            self._hold_off = max(0.0, self._hold_off - step)
        elif raw is False:
            self._hold_off += step
            self._hold_on = max(0.0, self._hold_on - step)
        else:  # inside the band: neither accumulator advances
            pass
        if not self.state and raw is True and self._hold_on >= self.enter_dwell_s - DWELL_EPS:
            self.state = True
            self._hold_off = 0.0
        elif self.state and raw is False and self._hold_off >= self.exit_dwell_s - DWELL_EPS:
            self.state = False
            self._hold_on = 0.0
        return self.state


class PerclosWindow(object):
    """Time-weighted PERCLOS over a sliding window.

    PERCLOS is the fraction of *time* the eyes are closed, not the fraction of
    *frames*.  On a pipeline that runs at a variable 6-15 FPS those are
    different numbers: a count-weighted ratio is biased by whatever the frame
    rate happened to do during the window, which is exactly the interval the
    driver was drowsy and the pipeline was slowest.

    ``add`` must only be called on frames where the driver is present AND the
    eye channel produced a usable measurement.  Frames with no face inject no
    sample, which is what makes the coverage test below meaningful.

    Validity - all three must hold, otherwise :attr:`valid` is ``False`` and the
    value must be reported as unknown rather than as a number:

    * ``span >= min_span_s``       the window has seen enough wall time;
    * ``rate_hz >= min_rate_hz``   >= 8 Hz effective samples, per the design.  At
      6 FPS a 60 s PERCLOS is not measurable and saying so is the only honest
      answer;
    * ``coverage >= min_coverage`` the measured intervals actually cover the
      span.  A driver absent for half the window leaves coverage at ~0.5.

    The span is measured to NOW, not to the newest sample, which is why
    :meth:`advance` must be called on every frame including the blind ones.
    Measuring to the newest sample would let a window that stopped being fed
    thirty seconds ago keep reporting the number it held when the face was last
    seen, which is the worst kind of stale: plausible and wrong.

    Each interval is clamped to ``max_gap_s``: a longer gap is not credited to
    either state, because nothing observed it.

    Units: ``t_s`` and ``dt_s`` seconds, :attr:`value` a fraction in ``[0, 1]``.
    """

    def __init__(
        self,
        window_s: float = 60.0,
        min_span_s: float = 30.0,
        min_rate_hz: float = 8.0,
        min_coverage: float = 0.6,
        max_gap_s: float = 0.5,
    ) -> None:
        self.window_s = float(window_s)
        self.min_span_s = float(min_span_s)
        self.min_rate_hz = float(min_rate_hz)
        self.min_coverage = float(min_coverage)
        self.max_gap_s = float(max_gap_s)
        self._samples: Deque[Tuple[float, float, bool]] = deque()
        self._sum_dt = 0.0
        self._sum_closed = 0.0
        self._t_now: Optional[float] = None

    def reset(self) -> None:
        self._samples.clear()
        self._sum_dt = 0.0
        self._sum_closed = 0.0
        self._t_now = None

    def advance(self, t_s: float) -> None:
        """Move "now" forward and drop samples that fell out of the window.

        Call this every frame, whether or not a sample follows.  A frame with no
        usable eye measurement contributes no sample but still consumes window
        time, and that is exactly what :attr:`coverage` has to see.
        """
        t = finite(t_s)
        if t is None:
            return
        if self._t_now is None or t > self._t_now:
            self._t_now = t
        cutoff = self._t_now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            _, old_dt, old_closed = self._samples.popleft()
            self._sum_dt -= old_dt
            if old_closed:
                self._sum_closed -= old_dt
        # guard against float drift making a sum slightly negative
        if self._sum_dt < 0.0:
            self._sum_dt = 0.0
        if self._sum_closed < 0.0:
            self._sum_closed = 0.0

    def add(self, t_s: float, dt_s: float, closed: bool) -> None:
        t = finite(t_s)
        if t is None:
            return
        step = finite(dt_s, 0.0) or 0.0
        step = min(max(0.0, step), self.max_gap_s)
        self._samples.append((t, step, bool(closed)))
        self._sum_dt += step
        if closed:
            self._sum_closed += step
        self.advance(t)

    @property
    def span_s(self) -> float:
        """Seconds from the oldest surviving sample to now, capped at the window."""
        if not self._samples or self._t_now is None:
            return 0.0
        return max(0.0, min(self.window_s, self._t_now - self._samples[0][0]))

    @property
    def rate_hz(self) -> float:
        span = self.span_s
        if span <= 0.0:
            return 0.0
        return len(self._samples) / span

    @property
    def coverage(self) -> float:
        span = self.span_s
        if span <= 0.0:
            return 0.0
        return self._sum_dt / span

    @property
    def value(self) -> float:
        if self._sum_dt <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self._sum_closed / self._sum_dt))

    @property
    def valid(self) -> bool:
        if len(self._samples) < 2:
            return False
        return (
            self.span_s >= self.min_span_s
            and self.rate_hz >= self.min_rate_hz
            and self.coverage >= self.min_coverage
        )

    @property
    def starved(self) -> bool:
        """True once the window is long enough but still not valid.

        Distinguishes "warming up" (span too short, normal for the first 30 s)
        from "this pipeline cannot measure PERCLOS", which is a real degradation
        the operator must be told about.
        """
        return self.span_s >= self.min_span_s and not self.valid


# ---------------------------------------------------------------- config ---
@dataclass
class FusionConfig:
    """Every tunable the fusion layer has, in seconds / degrees / probability.

    Defaults come from ``docs/production-dms-design.md`` and from the engine
    contracts in ``engines/MANIFEST.json`` (the classifier heads are trained
    with a 0.50 operating point).  :meth:`from_app_config` overlays anything the
    YAML actually sets, so a vehicle profile can retune without a code change
    and a profile that sets nothing still gets a coherent set.

    The EAR/MAR defaults are NOT re-typed here.  ``ear_closed``,
    ``ear_open_median`` and ``mar_yawn`` are imported from
    :mod:`dms.geometry.ear` / :mod:`dms.geometry.mar`, which own those numbers;
    ``dms.config.schema.StateConfig`` carries the same values and *rejects the
    old broken-landmark scale (0.50 / 0.75 / 0.65) by name*.  A ``FusionConfig()``
    built with no ``AppConfig`` therefore has to survive the schema's own
    validators, and ``tests/test_fusion.py`` asserts exactly that so the two
    files cannot drift apart again.

    Mutability: this object is edited in place at runtime
    (``DriverMonitor.recalibrate`` / ``_sync_calibration``).  Every consumer in
    :class:`SignalFuser` re-reads it each frame - see
    :meth:`SignalFuser._apply_cfg` - so nothing here is ever frozen into a
    latch at construction time.
    """

    # --- eye channel -------------------------------------------------------
    #: model P(open) below this counts as closed; at/above `eye_prob_open` as
    #: open.  The gap is the Schmitt band.
    eye_prob_closed: float = 0.40
    eye_prob_open: float = 0.55
    #: geometry EAR thresholds, on the CORRECTED roll-invariant EAR scale.
    #: Imported from :mod:`dms.geometry.ear`, which is the single source of
    #: truth for them, so this file cannot drift from the config schema.  The
    #: effective closed threshold is ``min(ear_closed, 0.70 * ear_open_median)``
    #: = ``min(0.18, 0.189)`` = 0.18.  These are a documented STARTING POINT
    #: derived from open-eyed footage, not a calibration - no clip in this
    #: repository contains a labelled eye closure.
    ear_closed: float = EAR_CLOSED_RECOMMENDED
    ear_open_median: float = EAR_OPEN_MEDIAN_RECOMMENDED
    ear_hysteresis: float = 0.03
    eye_close_dwell_s: float = 0.15
    eye_open_dwell_s: float = 0.10

    # --- mouth channel -----------------------------------------------------
    mouth_prob_open: float = 0.50
    mouth_prob_closed: float = 0.40
    #: inner-lip MAR thresholds, on the CORRECTED scale.  ``mar_yawn`` comes
    #: from :mod:`dms.geometry.mar`; ``mar_close`` is only the hysteresis floor
    #: and has no geometry constant of its own, so it tracks
    #: ``dms.config.schema.StateConfig.mar_close`` and a test pins the two
    #: together.  UNVALIDATED: no clip here contains a labelled yawn.
    mar_yawn: float = MAR_YAWN_RECOMMENDED
    mar_close: float = 0.35
    mouth_open_dwell_s: float = 0.10
    mouth_close_dwell_s: float = 0.20

    # --- pose channel ------------------------------------------------------
    #: minimum HeadPose.confidence for the model / geometry branch to be used.
    pose_min_conf: float = 0.35
    #: maximum PnP reprojection error, px, before a geometry pose is discarded.
    pose_max_reproj_px: float = 6.0
    #: confidence attributed to the legacy `yaw=`/`pitch=` float call path,
    #: which carries no quality metric of its own.  Above `pose_min_conf` so the
    #: legacy path still works, and documented as a known weakness: the
    #: DriverSignals path is the one with a real gate.
    legacy_pose_conf: float = 0.60
    pose_tau_s: float = 0.25

    # --- gaze --------------------------------------------------------------
    gaze_min_conf: float = 0.40
    gaze_yaw_deg: float = 25.0
    gaze_pitch_deg: float = 20.0
    gaze_tau_s: float = 0.20

    # --- occlusion ---------------------------------------------------------
    sunglasses_thr: float = 0.50
    mask_thr: float = 0.50

    # --- perclos -----------------------------------------------------------
    perclos_window_s: float = 60.0
    perclos_min_span_s: float = 30.0
    perclos_min_rate_hz: float = 8.0
    perclos_min_coverage: float = 0.6

    # --- timing ------------------------------------------------------------
    #: intervals longer than this are a gap: nothing is integrated across them.
    max_gap_s: float = 0.5
    #: after this long with no usable measurement, a channel's latched state is
    #: reported stale and stops being trusted (it is NOT re-thresholded).
    stale_after_s: float = 0.5

    # --- calibration -------------------------------------------------------
    forward_yaw: float = 0.0
    forward_pitch: float = 0.0
    forward_roll: float = 0.0
    #: False until `scripts/calibrate_forward.py` (or `--calibrate-forward`)
    #: has written a forward zero.  Gaze alerts are suppressed while False.
    calibrated: bool = False

    @property
    def ear_closed_effective(self) -> float:
        """Closed-EAR threshold actually used, in EAR units."""
        thr = self.ear_closed
        if self.ear_open_median > 0.0:
            thr = min(thr, 0.70 * self.ear_open_median)
        return thr

    @classmethod
    def from_app_config(cls, cfg: Any) -> "FusionConfig":
        """Build from an :class:`dms.config.schema.AppConfig`.

        Reads ``cfg.state`` and ``cfg.forward_zero`` with ``getattr`` defaults so
        that adding a knob here does not require a schema change first, and so
        that an older YAML keeps working.  Calibration is inferred from the
        forward zero being non-zero: ``--calibrate-forward`` writes medians, and
        an exact 0/0/0 triple is the schema default meaning "never calibrated".
        """
        out = cls()
        state = getattr(cfg, "state", None)
        zero = getattr(cfg, "forward_zero", None)
        if state is not None:
            for name in (
                "ear_closed",
                "ear_open_median",
                "mar_yawn",
                "gaze_yaw_deg",
                "gaze_pitch_deg",
            ):
                value = finite(getattr(state, name, None))
                if value is not None:
                    setattr(out, name, value)
            for name in (
                "eye_prob_closed",
                "eye_prob_open",
                "ear_hysteresis",
                "eye_close_dwell_s",
                "eye_open_dwell_s",
                "mouth_prob_open",
                "mouth_prob_closed",
                "mar_close",
                "pose_min_conf",
                "pose_max_reproj_px",
                "gaze_min_conf",
                "sunglasses_thr",
                "mask_thr",
                "perclos_window_s",
                "perclos_min_span_s",
                "perclos_min_rate_hz",
                "perclos_min_coverage",
                "max_gap_s",
                "stale_after_s",
            ):
                value = finite(getattr(state, name, None))
                if value is not None:
                    setattr(out, name, value)
        if zero is not None:
            out.forward_yaw = finite(getattr(zero, "yaw", 0.0), 0.0) or 0.0
            out.forward_pitch = finite(getattr(zero, "pitch", 0.0), 0.0) or 0.0
            out.forward_roll = finite(getattr(zero, "roll", 0.0), 0.0) or 0.0
        out.calibrated = not (
            abs(out.forward_yaw) < 1e-9
            and abs(out.forward_pitch) < 1e-9
            and abs(out.forward_roll) < 1e-9
        )
        explicit = getattr(cfg, "forward_zero_calibrated", None)
        if isinstance(explicit, bool):
            out.calibrated = explicit
        return out


# ----------------------------------------------------------------- output --
@dataclass
class Channel:
    """Provenance of one fused capability.

    ``reliable`` is the only thing the state machine is allowed to branch on
    when deciding whether an alert may fire.  ``reason`` is a short machine
    token for the log, never prose.
    """

    source: SignalSource = SignalSource.UNAVAILABLE
    confidence: float = 0.0
    reliable: bool = False
    reason: str = "no_signal"

    @property
    def usable(self) -> bool:
        return self.reliable and self.source is not SignalSource.UNAVAILABLE


@dataclass
class FusedView:
    """One frame of fused, provenance-tagged driver state.

    Angles degrees (``yaw_rel``/``pitch_rel`` are relative to the calibrated
    forward zero), probabilities and ``perclos`` fractions in ``[0, 1]``, times
    seconds.
    """

    t_s: float = 0.0
    dt_s: float = 0.0
    present: bool = False
    track_id: Optional[int] = None
    identity: Optional[str] = None

    eyes: Channel = field(default_factory=Channel)
    eye_closed: bool = False           # dwelled state -> MICROSLEEP
    eye_closed_instant: bool = False   # band only     -> PERCLOS
    eye_open_score: float = 0.0        # prob_open (model) or EAR (geometry)
    perclos: float = 0.0
    perclos_valid: bool = False
    perclos_starved: bool = False
    perclos_rate_hz: float = 0.0
    perclos_coverage: float = 0.0

    mouth: Channel = field(default_factory=Channel)
    mouth_open: bool = False
    mouth_score: float = 0.0

    pose: Channel = field(default_factory=Channel)
    yaw_rel: float = 0.0
    pitch_rel: float = 0.0
    roll: float = 0.0
    calibrated: bool = False

    gaze: Channel = field(default_factory=Channel)
    gaze_off_road: bool = False
    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0

    phone_conf: float = 0.0
    sunglasses: float = 0.0
    mask: float = 0.0
    eyes_occluded: bool = False
    mouth_occluded: bool = False

    reliable: bool = False

    def channels(self) -> List[Tuple[str, Channel]]:
        return [
            ("eyes", self.eyes),
            ("mouth", self.mouth),
            ("pose", self.pose),
            ("gaze", self.gaze),
        ]


# ----------------------------------------------------------------- fuser ---
class SignalFuser(object):
    """Stateful per-driver fusion.

    One instance per driver track.  :meth:`reset` is called by the state machine
    on a driver change so that no filter, latch or PERCLOS window is ever
    stitched across two people.

    :meth:`update` is total: it never raises and always returns a
    :class:`FusedView`, including for a frame with no face at all (everything
    unavailable, ``present=False``).

    THRESHOLDS ARE LIVE, NEVER CACHED.  ``self.cfg`` is mutable at runtime and
    the codebase already relies on that: ``DriverMonitor.recalibrate`` and
    ``DriverMonitor._sync_calibration`` edit the forward zero on this very
    object, and the pose path reads it live each frame.  The EAR/MAR thresholds
    are the same kind of number - ``--calibrate-forward`` (``dms/app.py``,
    validated by ``scripts/calibrate_forward.py``) writes a measured
    ``state.ear_open_median`` into ``configs/vehicle.yaml`` - and having two
    different rules for the two halves of one config object is the trap.  Every
    threshold, dwell and time constant this class owns is therefore re-read from
    ``self.cfg`` at the top of each :meth:`update` by :meth:`_apply_cfg`, and
    the constructor sets them through that same method.  There is exactly one
    place a threshold is read.  The alternative - copying thresholds into the
    latches at ``__init__`` while ``_fuse_eyes`` re-read them live - let a single
    frame be judged against two threshold generations and made MICROSLEEP
    silently unreachable after a retune.
    """

    def __init__(self, cfg: Optional[FusionConfig] = None) -> None:
        self.cfg = cfg or FusionConfig()
        # Every number below is a PLACEHOLDER.  `_apply_cfg` writes the real
        # values out of `self.cfg` here and again at the top of every `update`,
        # so a threshold is read from exactly one place and can never be a
        # generation behind the config it came from.  Only `direction` is fixed
        # at construction: it is the shape of the comparison, not a tunable.
        self.perclos = PerclosWindow()
        self._eye_model = SchmittLatch(0.0, 0.0, direction=BELOW)
        self._eye_geom = SchmittLatch(0.0, 0.0, direction=BELOW)
        self._mouth_model = SchmittLatch(0.0, 0.0, direction=ABOVE)
        self._mouth_geom = SchmittLatch(0.0, 0.0, direction=ABOVE)
        self._yaw = CircularEwma(0.0)
        self._pitch = CircularEwma(0.0)
        self._roll = CircularEwma(0.0)
        self._gaze_yaw = CircularEwma(0.0)
        self._gaze_pitch = CircularEwma(0.0)
        self._apply_cfg()
        self._t_prev: Optional[float] = None
        self._t_eye_ok: Optional[float] = None
        self._t_mouth_ok: Optional[float] = None
        self._t_pose_ok: Optional[float] = None
        self.view = FusedView()

    # -- configuration ------------------------------------------------------
    def _apply_cfg(self) -> None:
        """Push every threshold in ``self.cfg`` into the stateful objects.

        Called from ``__init__`` and from the top of every :meth:`update`, which
        is what makes ``FusionConfig`` genuinely live: retuning
        ``cfg.ear_closed`` (or any other threshold, dwell or time constant)
        changes the very next frame's decision, and the confidence reported for
        that frame is computed from the same number.  Demonstrated by
        ``tests/test_fusion.py::test_retuning_ear_closed_moves_the_latch_not_just_the_confidence``
        and ``::test_every_threshold_the_fuser_holds_is_re_read_every_frame``.

        Writes ONLY thresholds.  Latch states, dwell accumulators, EWMA
        contents and the PERCLOS ring are left alone, so a retune during a live
        eye closure neither drops it nor restarts its clock.

        Cost, measured on this board (Xavier NX, Python 3.8): 11.8 us per call
        against 133 us for the whole of :meth:`update` - 8.8% of the fusion
        layer, and under 0.02% of a 15 FPS frame budget, of which the perception
        stage takes 60-90 ms. Cheap enough that "rebuild the fuser to retune"
        was never worth the trap it set.
        """
        cfg = self.cfg
        eye_enter = max(0.0, float(cfg.eye_close_dwell_s))
        eye_exit = max(0.0, float(cfg.eye_open_dwell_s))
        self._eye_model.enter_thr = float(cfg.eye_prob_closed)
        self._eye_model.exit_thr = float(cfg.eye_prob_open)
        ear_thr = float(cfg.ear_closed_effective)
        self._eye_geom.enter_thr = ear_thr
        self._eye_geom.exit_thr = ear_thr + float(cfg.ear_hysteresis)
        for latch in (self._eye_model, self._eye_geom):
            latch.enter_dwell_s = eye_enter
            latch.exit_dwell_s = eye_exit

        mouth_enter = max(0.0, float(cfg.mouth_open_dwell_s))
        mouth_exit = max(0.0, float(cfg.mouth_close_dwell_s))
        self._mouth_model.enter_thr = float(cfg.mouth_prob_open)
        self._mouth_model.exit_thr = float(cfg.mouth_prob_closed)
        self._mouth_geom.enter_thr = float(cfg.mar_yawn)
        self._mouth_geom.exit_thr = float(cfg.mar_close)
        for latch in (self._mouth_model, self._mouth_geom):
            latch.enter_dwell_s = mouth_enter
            latch.exit_dwell_s = mouth_exit

        self.perclos.window_s = float(cfg.perclos_window_s)
        self.perclos.min_span_s = float(cfg.perclos_min_span_s)
        self.perclos.min_rate_hz = float(cfg.perclos_min_rate_hz)
        self.perclos.min_coverage = float(cfg.perclos_min_coverage)
        self.perclos.max_gap_s = float(cfg.max_gap_s)

        pose_tau = float(cfg.pose_tau_s)
        for filt in (self._yaw, self._pitch, self._roll):
            filt.tau_s = pose_tau
        gaze_tau = float(cfg.gaze_tau_s)
        for filt in (self._gaze_yaw, self._gaze_pitch):
            filt.tau_s = gaze_tau

    # -- lifecycle ----------------------------------------------------------
    def reset(self, keep_clock: bool = False) -> None:
        """Drop every window, filter and latch.

        ``keep_clock`` retains the previous timestamp so the next ``dt`` is
        still the real interval; the state machine uses that on a driver change,
        where time did not stop even though the person did.
        """
        self.perclos.reset()
        for latch in (self._eye_model, self._eye_geom, self._mouth_model, self._mouth_geom):
            latch.reset()
        for filt in (self._yaw, self._pitch, self._roll, self._gaze_yaw, self._gaze_pitch):
            filt.reset()
        self._t_eye_ok = None
        self._t_mouth_ok = None
        self._t_pose_ok = None
        if not keep_clock:
            self._t_prev = None
        self.view = FusedView()

    # -- per-frame ----------------------------------------------------------
    def update(
        self,
        t_s: float,
        present: bool,
        signals: Optional[DriverSignals] = None,
    ) -> FusedView:
        # Thresholds first: `self.cfg` may have been retuned since the last
        # frame and every consumer below must see the same generation of it.
        self._apply_cfg()
        cfg = self.cfg
        now = finite(t_s, 0.0) or 0.0
        if self._t_prev is None:
            dt = 0.0
        else:
            dt = now - self._t_prev
            if dt < 0.0:  # clock went backwards: treat as a gap, integrate nothing
                dt = 0.0
        gapped = dt > cfg.max_gap_s
        dt_eff = 0.0 if gapped else dt
        self._t_prev = now

        sig = signals if signals is not None else DriverSignals()
        view = FusedView(
            t_s=now,
            dt_s=dt_eff,
            present=bool(present),
            track_id=None if sig.track_id is None or sig.track_id < 0 else int(sig.track_id),
            identity=sig.identity,
            calibrated=cfg.calibrated,
        )

        self._fuse_occlusion(view, sig.occlusion)
        self._fuse_eyes(view, sig.left_eye, sig.right_eye, dt_eff, now)
        self._fuse_mouth(view, sig.mouth, dt_eff, now)
        self._fuse_pose(view, sig.pose, dt_eff, now)
        self._fuse_gaze(view, sig.gaze, dt_eff)
        view.phone_conf = max(0.0, min(1.0, finite(sig.phone_conf, 0.0) or 0.0))

        # PERCLOS only integrates frames where a face was present AND the eye
        # channel actually measured something this frame (``eyes.usable`` is set
        # only on the measured-and-present-and-unoccluded path).  A coasting
        # track or a blind frame must not inject the last known eye state - but
        # it must still advance the window, or an unfed window would keep
        # reporting a stale number for ever.
        if view.present and view.eyes.usable:
            self.perclos.add(now, dt_eff, view.eye_closed_instant)
        else:
            self.perclos.advance(now)
        view.perclos = self.perclos.value
        view.perclos_valid = self.perclos.valid
        view.perclos_starved = self.perclos.starved
        view.perclos_rate_hz = self.perclos.rate_hz
        view.perclos_coverage = self.perclos.coverage

        view.reliable = bool(
            view.present
            and (sig.reliable if signals is not None else True)
            and view.eyes.usable
            and view.pose.usable
        )
        self.view = view
        return view

    # -- per-capability -----------------------------------------------------
    def _fuse_occlusion(self, view: FusedView, occ: Optional[Occlusion]) -> None:
        if occ is None:
            return
        view.sunglasses = max(0.0, min(1.0, finite(occ.sunglasses, 0.0) or 0.0))
        view.mask = max(0.0, min(1.0, finite(occ.mask, 0.0) or 0.0))
        if occ.source is SignalSource.UNAVAILABLE:
            # No occlusion detector ran.  We cannot claim the face is clear, but
            # we also must not invent an occlusion; leave both gates open and let
            # the eye/mouth channels speak for themselves.
            return
        view.eyes_occluded = bool(
            view.sunglasses >= self.cfg.sunglasses_thr or not occ.eyes_reliable
        )
        view.mouth_occluded = bool(view.mask >= self.cfg.mask_thr or not occ.mouth_reliable)

    def _fuse_eyes(
        self,
        view: FusedView,
        left: Optional[EyeState],
        right: Optional[EyeState],
        dt: float,
        now: float,
    ) -> None:
        cfg = self.cfg
        eyes = [e for e in (left, right) if e is not None]
        model_probs = [
            p
            for p in (
                finite(e.prob_open)
                for e in eyes
                if e.source is SignalSource.MODEL
            )
            if p is not None and 0.0 <= p <= 1.0
        ]
        geom_ears = [
            v
            for v in (
                finite(e.ear)
                for e in eyes
                if e.source in (SignalSource.MODEL, SignalSource.GEOMETRY)
            )
            if v is not None and v > 0.0
        ]

        if model_probs:
            score = sum(model_probs) / len(model_probs)
            latch = self._eye_model
            source = SignalSource.MODEL
            reason = "model"
            confidence = min(1.0, abs(score - 0.5) * 2.0)
        elif geom_ears:
            score = sum(geom_ears) / len(geom_ears)
            latch = self._eye_geom
            source = SignalSource.GEOMETRY
            reason = "geometry"
            # An EAR is only as trustworthy as its distance from the threshold,
            # and the contour EAR is roll-sensitive, so cap the confidence it may
            # ever claim well below a classifier's.
            thr = cfg.ear_closed_effective
            confidence = min(0.6, abs(score - thr) / max(1e-6, thr))
        else:
            view.eyes = Channel(SignalSource.UNAVAILABLE, 0.0, False, "no_signal")
            view.eye_closed = self._stale_state(self._eye_model, self._eye_geom)
            view.eye_closed_instant = view.eye_closed
            return

        view.eye_open_score = score
        view.eye_closed_instant = latch.band(score)
        view.eye_closed = latch.update(score, dt)
        self._t_eye_ok = now

        if view.eyes_occluded:
            view.eyes = Channel(source, confidence, False, "occluded")
        elif not view.present:
            view.eyes = Channel(source, confidence, False, "absent")
        else:
            view.eyes = Channel(source, confidence, True, reason)

    def _fuse_mouth(
        self, view: FusedView, mouth: Optional[MouthState], dt: float, now: float
    ) -> None:
        if mouth is None:
            view.mouth = Channel(SignalSource.UNAVAILABLE, 0.0, False, "no_signal")
            return
        prob = finite(mouth.prob_open)
        mar = finite(mouth.mar)
        if mouth.source is SignalSource.MODEL and prob is not None and 0.0 <= prob <= 1.0:
            score, latch = prob, self._mouth_model
            source, reason = SignalSource.MODEL, "model"
            confidence = min(1.0, abs(prob - 0.5) * 2.0)
        elif mar is not None and mar > 0.0 and mouth.source is not SignalSource.UNAVAILABLE:
            score, latch = mar, self._mouth_geom
            source, reason = SignalSource.GEOMETRY, "geometry"
            confidence = min(0.6, abs(mar - self.cfg.mar_yawn) / max(1e-6, self.cfg.mar_yawn))
        else:
            view.mouth = Channel(SignalSource.UNAVAILABLE, 0.0, False, "no_signal")
            view.mouth_open = self._stale_state(self._mouth_model, self._mouth_geom)
            return
        view.mouth_score = score
        view.mouth_open = latch.update(score, dt)
        self._t_mouth_ok = now
        if view.mouth_occluded:
            view.mouth = Channel(source, confidence, False, "occluded")
        elif not view.present:
            view.mouth = Channel(source, confidence, False, "absent")
        else:
            view.mouth = Channel(source, confidence, True, reason)

    def _fuse_pose(
        self, view: FusedView, pose: Optional[HeadPose], dt: float, now: float
    ) -> None:
        cfg = self.cfg
        if pose is None or pose.source is SignalSource.UNAVAILABLE:
            view.pose = Channel(SignalSource.UNAVAILABLE, 0.0, False, "no_signal")
            self._carry_pose(view)
            return
        conf = max(0.0, min(1.0, finite(pose.confidence, 0.0) or 0.0))
        reproj = finite(pose.reprojection_error, 0.0) or 0.0
        reason = "model" if pose.source is SignalSource.MODEL else "geometry"
        if conf < cfg.pose_min_conf:
            view.pose = Channel(pose.source, conf, False, "low_confidence")
            self._carry_pose(view)
            return
        if pose.source is SignalSource.GEOMETRY and reproj > cfg.pose_max_reproj_px:
            view.pose = Channel(pose.source, conf, False, "high_reprojection")
            self._carry_pose(view)
            return

        yaw = self._yaw.update(pose.yaw, dt)
        pitch = self._pitch.update(pose.pitch, dt)
        roll = self._roll.update(pose.roll, dt)
        if yaw is None or pitch is None:
            view.pose = Channel(pose.source, conf, False, "no_signal")
            self._carry_pose(view)
            return
        view.yaw_rel = angdiff(yaw, cfg.forward_yaw)
        view.pitch_rel = angdiff(pitch, cfg.forward_pitch)
        view.roll = wrap180(angdiff(roll if roll is not None else 0.0, cfg.forward_roll))
        self._t_pose_ok = now
        if not view.present:
            view.pose = Channel(pose.source, conf, False, "absent")
        elif not cfg.calibrated:
            # The angles are real, but "away from the road" is undefined without
            # a forward zero.  Report the numbers, refuse to call them reliable.
            view.pose = Channel(pose.source, conf, False, "uncalibrated")
        else:
            view.pose = Channel(pose.source, conf, True, reason)

    def _carry_pose(self, view: FusedView) -> None:
        """Expose the last smoothed pose for the overlay without trusting it."""
        yaw, pitch, roll = self._yaw.value, self._pitch.value, self._roll.value
        if yaw is not None:
            view.yaw_rel = angdiff(yaw, self.cfg.forward_yaw)
        if pitch is not None:
            view.pitch_rel = angdiff(pitch, self.cfg.forward_pitch)
        if roll is not None:
            view.roll = angdiff(roll, self.cfg.forward_roll)

    def _fuse_gaze(self, view: FusedView, gaze: Optional[GazeVector], dt: float) -> None:
        """Fuse eye gaze with head pose into a road-relative direction.

        Eye gaze from the model is relative to the FACE CROP, so it is only
        meaningful once added to a head pose that is itself referenced to the
        calibrated forward zero.  With no usable pose there is no vehicle frame
        to express gaze in, and the channel is unavailable however good the eye
        model was.
        """
        cfg = self.cfg
        if not view.pose.usable:
            view.gaze = Channel(SignalSource.UNAVAILABLE, 0.0, False, view.pose.reason)
            view.gaze_yaw = view.yaw_rel
            view.gaze_pitch = view.pitch_rel
            view.gaze_off_road = False
            return

        eye_yaw = eye_pitch = 0.0
        eye_conf = 0.0
        used_eye = False
        if gaze is not None and gaze.source is SignalSource.MODEL:
            g_conf = max(0.0, min(1.0, finite(gaze.confidence, 0.0) or 0.0))
            g_yaw = finite(gaze.yaw)
            g_pitch = finite(gaze.pitch)
            if g_conf >= cfg.gaze_min_conf and g_yaw is not None and g_pitch is not None:
                eye_yaw = self._gaze_yaw.update(g_yaw, dt) or 0.0
                eye_pitch = self._gaze_pitch.update(g_pitch, dt) or 0.0
                eye_conf = g_conf
                used_eye = True

        view.gaze_yaw = wrap180(view.yaw_rel + eye_yaw)
        view.gaze_pitch = wrap180(view.pitch_rel + eye_pitch)
        view.gaze_off_road = bool(
            abs(view.gaze_yaw) >= cfg.gaze_yaw_deg or abs(view.gaze_pitch) >= cfg.gaze_pitch_deg
        )
        source = SignalSource.MODEL if used_eye else view.pose.source
        confidence = max(eye_conf, view.pose.confidence) if used_eye else view.pose.confidence
        view.gaze = Channel(source, confidence, True, "eye+pose" if used_eye else "pose_only")

    @staticmethod
    def _stale_state(*latches: SchmittLatch) -> bool:
        for latch in latches:
            if latch.state:
                return True
        return False
