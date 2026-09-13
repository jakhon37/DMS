"""Typed, strictly-validated configuration for the DMS.

Every nested section forbids unknown keys (``extra = "forbid"``).  This is a
safety property, not tidiness: before this, a typo in ``ear_closed`` was
silently ignored and the unit ran on the default drowsiness threshold while the
operator believed it was running on theirs.  A bad config must fail inside
:func:`load_config` with a message naming the key, never reach the frame loop.

Units
-----
Times are SECONDS unless the name ends in ``_ms`` / ``_ns``.  Angles are
DEGREES.  Sizes are PIXELS.  Probabilities and ratios are dimensionless in
[0, 1].  Memory is MiB.  Rates are Hz.

Calibration merge
-----------------
``AppConfig.calibration`` names a per-unit override file (default
``vehicle.yaml``) resolved relative to the directory of the config being
loaded.  It is merged into EVERY profile, not only ``default.yaml`` as before -
the old rule made ``configs/production.yaml`` impossible to calibrate, which in
turn left every gaze-derived alert permanently suppressed on the one profile
that needs them.  Set ``calibration: ""`` to disable the merge.

Failure behaviour
-----------------
:func:`load_config` raises ``ValueError`` listing every offending key path for
an unknown key, an out-of-range value, or an inconsistent combination (for
example a non-loopback ``health.bind`` without ``health.allow_remote``).  It
never returns a partially-valid config.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

#: Backends a capability may be resolved to.  Mirrors
#: ``dms.infer.registry.VALID_BACKENDS``, duplicated here so that importing the
#: config schema does not import TensorRT.
VALID_BACKENDS = ("model", "geometry", "off")

#: Capabilities the model registry knows about.
CAPABILITIES = (
    "headpose",
    "eyestate",
    "mouthstate",
    "occlusion",
    "gaze",
    "objects",
    "identity",
)


class Strict(BaseModel):
    """Base for every section: unknown keys are an error, values are validated."""

    class Config:
        extra = "forbid"
        allow_mutation = True


def _positive(name):
    def _check(cls, v):
        if v is None:
            return v
        if float(v) <= 0.0:
            raise ValueError("%s must be > 0, got %r" % (name, v))
        return v

    return _check


def _unit_interval(name):
    def _check(cls, v):
        if v is None:
            return v
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("%s must be in [0, 1], got %r" % (name, v))
        return v

    return _check


# --------------------------------------------------------------------- source


class SourceConfig(Strict):
    """Where frames come from.  ``dev`` marks a lab/replay source explicitly."""

    type: str = "file"
    path: str = "tests/replay/day_driver.mp4"
    dev: bool = True
    device: str = "/dev/video0"
    sensor_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    @validator("type")
    def _type(cls, v):
        allowed = {"csi", "usb", "file", "test"}
        if v not in allowed:
            raise ValueError("source.type must be one of %s" % sorted(allowed))
        return v

    @validator("width", "height", "fps")
    def _geometry(cls, v):
        if v <= 0:
            raise ValueError("source geometry/fps must be > 0, got %r" % v)
        return v

    @validator("sensor_id")
    def _sensor(cls, v):
        if v < 0:
            raise ValueError("source.sensor_id must be >= 0")
        return v


class CaptureConfig(Strict):
    """Frame plumbing.  ``queue_depth`` 1 means newest-frame-wins.

    ``letterbox`` is the (width, height) of the detector input this profile
    EXPECTS.  It is declarative: the authority is the SCRFD engine, whose input
    shape ``DmsApp._setup_perception`` reads to build the one full-frame
    :class:`dms.runtime.preprocess.Letterbox`.  The app logs LETTERBOX_MISMATCH
    when the two disagree, which is the only reason to keep the key - it used to
    be read by nothing at all and silently said 640 whatever the engine was.
    ``letterbox_pad_value`` records the SCRFD convention (0, top-left); the
    letterbox for the face detector is constructed from that convention directly.
    """

    full_size: Tuple[int, int] = (1280, 720)
    letterbox: Tuple[int, int] = (640, 640)
    letterbox_pad_value: int = 0
    queue_depth: int = 1

    @validator("queue_depth")
    def _depth(cls, v):
        if not (1 <= v <= 16):
            raise ValueError("capture.queue_depth must be 1..16")
        return v

    @validator("letterbox_pad_value")
    def _pad(cls, v):
        if not (0 <= v <= 255):
            raise ValueError("capture.letterbox_pad_value must be 0..255")
        return v


class CameraConfig(Strict):
    """Capture-failure policy, consumed by :class:`dms.runtime.CaptureSupervisor`.

    ``fail_fatal`` decides only what a source that will not start at STARTUP
    means: true exits non-zero so systemd restarts the unit, false enters the
    supervised retry ladder.  It never means "pretend to be healthy" - a unit
    with a dead camera reports ``camera=failed``, serves /healthz 503, emits
    CAMERA_FAIL and stops petting the systemd watchdog once the ladder is
    exhausted, whichever way this flag is set.  That distinction is the whole
    point: the previous build treated ``fail_fatal: false`` as licence to
    notify READY and ping the watchdog forever with no camera and no retry.
    """

    fail_fatal: bool = False
    #: keep retrying at the last backoff step forever instead of giving up.
    #: True is correct for a vehicle: a camera brownout should self-heal.
    retry_forever: bool = True
    #: reconnect delays in seconds, one per consecutive failure.
    backoff_s: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 5.0])
    #: with no frame and no error for this long, the source is declared dead.
    stall_timeout_s: float = 5.0
    #: deadline for frame #1 after a (re)start; 0 means "same as stall_timeout_s".
    first_frame_timeout_s: float = 0.0
    #: emit PIPELINE_STARVED below this measured rate; 0 disables the check.
    starved_below_fps: float = 0.0
    starved_for_s: float = 10.0

    @validator("backoff_s")
    def _backoff(cls, v):
        if not v:
            raise ValueError("camera.backoff_s must not be empty")
        if any(float(x) < 0.0 for x in v):
            raise ValueError("camera.backoff_s entries must be >= 0")
        return [float(x) for x in v]

    _v_stall = validator("stall_timeout_s", allow_reuse=True)(_positive("camera.stall_timeout_s"))
    _v_starved = validator("starved_for_s", allow_reuse=True)(_positive("camera.starved_for_s"))

    @validator("first_frame_timeout_s", "starved_below_fps")
    def _non_negative(cls, v, field):
        if float(v) < 0.0:
            raise ValueError("camera.%s must be >= 0" % field.name)
        return float(v)


# --------------------------------------------------------------------- models


class FaceModelConfig(Strict):
    engine: str
    conf: float = 0.5
    iou: float = 0.4
    device: str = "gpu"

    _v_conf = validator("conf", allow_reuse=True)(_unit_interval("models.face.conf"))
    _v_iou = validator("iou", allow_reuse=True)(_unit_interval("models.face.iou"))

    @validator("device")
    def _device(cls, v):
        if v not in ("gpu", "dla0", "dla1"):
            raise ValueError("models.face.device must be gpu, dla0 or dla1")
        return v


class LandmarksModelConfig(Strict):
    engine: str
    device: str = "gpu"

    @validator("device")
    def _device(cls, v):
        if v not in ("gpu", "dla0", "dla1"):
            raise ValueError("models.landmarks.device must be gpu, dla0 or dla1")
        return v


class HeadPoseCapConfig(Strict):
    """No tunables today; the sign convention is pinned in code by measurement."""


class EyeStateCapConfig(Strict):
    """OCEC operating point plus the resolution floor below which it is wrong.

    ``min_eye_box_px`` is not a performance knob.  The classifier was measured
    on this board to invert below roughly 20 source pixels of eye width - it
    calls 42% of eyes closed where the geometry calls 4% - so under the floor
    the wrapper falls through to the geometric EAR instead.
    """

    prob_open_threshold: float = 0.50
    ear_closed: float = 0.18
    crop_margin: float = 0.0
    min_eye_box_px: float = 20.0

    _v_p = validator("prob_open_threshold", allow_reuse=True)(
        _unit_interval("models.eyestate.prob_open_threshold")
    )
    _v_e = validator("ear_closed", allow_reuse=True)(_unit_interval("models.eyestate.ear_closed"))

    @validator("crop_margin", "min_eye_box_px")
    def _non_negative(cls, v, field):
        if float(v) < 0.0:
            raise ValueError("models.eyestate.%s must be >= 0" % field.name)
        return float(v)


class MouthStateCapConfig(Strict):
    prob_open_threshold: float = 0.50
    #: geometric inner-lip ratio above which the mouth counts as open.  This is
    #: the fallback channel threshold, NOT the yawn threshold (``state.mar_yawn``).
    mar_open: float = 0.25
    min_mouth_box_px: float = 30.0

    _v_p = validator("prob_open_threshold", allow_reuse=True)(
        _unit_interval("models.mouthstate.prob_open_threshold")
    )

    @validator("mar_open", "min_mouth_box_px")
    def _non_negative(cls, v, field):
        if float(v) < 0.0:
            raise ValueError("models.mouthstate.%s must be >= 0" % field.name)
        return float(v)


class OcclusionCapConfig(Strict):
    sunglasses_threshold: float = 0.50
    mask_threshold: float = 0.50
    head_box_expand: float = 0.25
    #: run the two 48x48 heads on every Nth call per track and reuse the cached
    #: answer in between.  Sunglasses do not appear mid-blink.
    every_n: int = 15

    _v_s = validator("sunglasses_threshold", allow_reuse=True)(
        _unit_interval("models.occlusion.sunglasses_threshold")
    )
    _v_m = validator("mask_threshold", allow_reuse=True)(
        _unit_interval("models.occlusion.mask_threshold")
    )

    @validator("head_box_expand")
    def _expand(cls, v):
        if not (0.0 <= float(v) <= 2.0):
            raise ValueError("models.occlusion.head_box_expand must be 0..2")
        return float(v)

    @validator("every_n")
    def _n(cls, v):
        if v < 1:
            raise ValueError("models.occlusion.every_n must be >= 1")
        return v


class GazeCapConfig(Strict):
    """No tunables; the capability is gated by ``models.research_models_allowed``."""


class ObjectsCapConfig(Strict):
    """YOLOX-Nano over a cabin ROI.

    ``every_n`` 20 is a measured choice, not a guess: at every 10th frame the
    whole-pipeline p95 lands on an object-net frame and inherits its full
    ~23 ms, taking p95 from 62.2 ms (inside the 66 ms gate) to 77.6 ms (outside).
    """

    roi: List[float] = Field(default_factory=lambda: [0.0, 0.45, 1.0, 1.0])
    classes: List[str] = Field(default_factory=lambda: ["cell phone"])
    conf: float = 0.30
    iou: float = 0.45
    every_n: int = 20

    @validator("roi")
    def _roi(cls, v):
        if len(v) != 4:
            raise ValueError("models.objects.roi must be [x1, y1, x2, y2] normalised")
        if any(not (0.0 <= float(x) <= 1.0) for x in v):
            raise ValueError("models.objects.roi entries must be in [0, 1]")
        if not (v[0] < v[2] and v[1] < v[3]):
            raise ValueError("models.objects.roi must have x1 < x2 and y1 < y2")
        return [float(x) for x in v]

    _v_conf = validator("conf", allow_reuse=True)(_unit_interval("models.objects.conf"))
    _v_iou = validator("iou", allow_reuse=True)(_unit_interval("models.objects.iou"))

    @validator("every_n")
    def _n(cls, v):
        if v < 1:
            raise ValueError("models.objects.every_n must be >= 1")
        return v


class IdentityCapConfig(Strict):
    #: OpenCV published operating point for SFace.  Raising it makes the system
    #: more willing to call the same person a new driver, which is the safe
    #: direction: a false "same person" silently stitches two people's windows.
    cosine_threshold: float = 0.363
    max_identities: int = 8
    refresh_n: int = 150

    _v_c = validator("cosine_threshold", allow_reuse=True)(
        _unit_interval("models.identity.cosine_threshold")
    )

    @validator("max_identities", "refresh_n")
    def _n(cls, v, field):
        if v < 1:
            raise ValueError("models.identity.%s must be >= 1" % field.name)
        return v


class BackendsConfig(Strict):
    """Per-capability backend REQUEST: ``model`` / ``geometry`` / ``off``.

    What the unit actually achieved is reported by ``ModelRegistry.describe()``
    and on /healthz; a capability that asked for ``model`` and got ``geometry``
    appears in ``degraded_capabilities()`` and raises MODEL_DEGRADED.
    """

    headpose: str = "model"
    eyestate: str = "model"
    mouthstate: str = "model"
    occlusion: str = "model"
    #: gaze weights are Gaze360-trained: research / non-commercial only.  Off by
    #: default, and the engine is not even deserialised unless
    #: ``models.research_models_allowed`` is true.
    gaze: str = "off"
    objects: str = "model"
    identity: str = "model"

    @validator("*", pre=True)
    def _backend(cls, v, field):
        if v is False or v is True:
            # YAML 1.1 turns a bare `off` into the boolean False (and `on` into
            # True).  Say so, because the fix is one pair of quotes and the
            # error is otherwise baffling.
            raise ValueError(
                "models.backends.%s got the YAML boolean %r - write it quoted, "
                "e.g. %s: 'off'" % (field.name, v, field.name)
            )
        if v not in VALID_BACKENDS:
            raise ValueError(
                "models.backends.%s must be one of %s, got %r"
                % (field.name, list(VALID_BACKENDS), v)
            )
        return v


class ModelsConfig(Strict):
    face: FaceModelConfig
    landmarks: LandmarksModelConfig
    backends: BackendsConfig = Field(default_factory=BackendsConfig)
    #: Master switch for research-licence weights (Gaze360).  MUST stay false on
    #: any profile that is not a research bench.
    research_models_allowed: bool = False
    headpose: HeadPoseCapConfig = Field(default_factory=HeadPoseCapConfig)
    eyestate: EyeStateCapConfig = Field(default_factory=EyeStateCapConfig)
    mouthstate: MouthStateCapConfig = Field(default_factory=MouthStateCapConfig)
    occlusion: OcclusionCapConfig = Field(default_factory=OcclusionCapConfig)
    gaze: GazeCapConfig = Field(default_factory=GazeCapConfig)
    objects: ObjectsCapConfig = Field(default_factory=ObjectsCapConfig)
    identity: IdentityCapConfig = Field(default_factory=IdentityCapConfig)

    def capability_cfg(self) -> Dict[str, dict]:
        """``{capability: dict}`` in the shape ``ModelRegistry`` expects."""
        return dict((name, getattr(self, name).dict()) for name in CAPABILITIES)

    def backend_map(self) -> Dict[str, str]:
        """``{capability: requested backend}``."""
        return self.backends.dict()


# ---------------------------------------------------------------------- state


class StateConfig(Strict):
    """Fusion thresholds and alert dwell / cooldown times.

    Every key here is read BY NAME by
    :meth:`dms.state.fusion.FusionConfig.from_app_config` and
    :meth:`dms.state.driver_state.AlertTuning.from_app_config`; declaring them
    makes them validated and documented rather than merely tolerated.

    THE EAR AND MAR SCALE CHANGED.  Before the 106-point landmark topology was
    corrected, ``face_ear`` was computed over MOUTH points and ``mar`` over the
    image-right EYE.  The old defaults (``ear_closed: 0.50``,
    ``ear_open_median: 0.75``, ``mar_yawn: 0.65``) belong to that broken metric.
    The values below are the measured replacements exported as
    ``dms.geometry.ear.EAR_CLOSED_RECOMMENDED`` /
    ``EAR_OPEN_MEDIAN_RECOMMENDED`` and ``dms.geometry.mar.MAR_YAWN_RECOMMENDED``.
    Leaving the old numbers in place latches MICROSLEEP CRITICAL on every frame
    and drives the GPIO line permanently, so a config on the old scale is
    rejected outright by the validator below.
    """

    # -- eye channel
    ear_closed: float = 0.18
    ear_open_median: float = 0.27
    eye_prob_closed: float = 0.40
    eye_prob_open: float = 0.55
    ear_hysteresis: float = 0.03
    eye_close_dwell_s: float = 0.15
    eye_open_dwell_s: float = 0.10

    # -- mouth channel
    mar_yawn: float = 0.45
    mar_close: float = 0.35
    mouth_prob_open: float = 0.50
    mouth_prob_closed: float = 0.40

    # -- pose / gaze channel
    pose_min_conf: float = 0.35
    pose_max_reproj_px: float = 6.0
    gaze_min_conf: float = 0.40
    gaze_yaw_deg: float = 25.0
    gaze_pitch_deg: float = 20.0

    # -- occlusion gates
    sunglasses_thr: float = 0.50
    mask_thr: float = 0.50

    # -- PERCLOS window and validity
    perclos_warn: float = 0.20
    perclos_crit: float = 0.40
    perclos_window_s: float = 60.0
    perclos_min_span_s: float = 30.0
    perclos_min_rate_hz: float = 8.0
    perclos_min_coverage: float = 0.6
    perclos_exit_margin: float = 0.05
    perclos_enter_s: float = 2.0
    perclos_exit_s: float = 10.0
    perclos_cooldown_s: float = 30.0

    # -- timing
    max_gap_s: float = 0.5
    stale_after_s: float = 0.5

    # -- presence
    face_lost_s: float = 1.0
    face_lost_exit_s: float = 0.40
    face_lost_cooldown_s: float = 5.0
    #: FACE_LOST enters WARN and escalates to CRITICAL after this long.  0.0
    #: restores the design-document behaviour (instant CRITICAL when moving).
    face_lost_crit_s: float = 3.0
    no_driver_grace_s: float = 10.0

    # -- microsleep
    microsleep_s: float = 1.5
    microsleep_exit_s: float = 0.20
    microsleep_cooldown_s: float = 10.0

    # -- yawn / fatigue
    yawn_enter_s: float = 0.40
    yawn_exit_s: float = 0.0
    yawn_cooldown_s: float = 15.0
    fatigue_yawns: int = 3
    fatigue_window_s: float = 180.0
    fatigue_exit_s: float = 0.0
    fatigue_cooldown_s: float = 30.0

    # -- gaze away
    gaze_warn_s: float = 2.0
    gaze_crit_s: float = 4.0
    gaze_exit_s: float = 0.50
    gaze_cooldown_s: float = 5.0

    # -- phone / distraction
    phone_conf: float = 0.50
    phone_enter_s: float = 0.80
    phone_exit_s: float = 0.40
    phone_cooldown_s: float = 10.0
    distraction_gaze_s: float = 6.0
    distraction_phone_s: float = 1.5
    distraction_exit_s: float = 1.5
    distraction_cooldown_s: float = 20.0

    # -- honesty alerts
    unreliable_enter_s: float = 1.0
    unreliable_exit_s: float = 2.0
    unreliable_cooldown_s: float = 20.0
    degraded_enter_s: float = 2.0
    degraded_exit_s: float = 5.0
    degraded_cooldown_s: float = 60.0

    # -- global
    assume_moving: bool = True
    #: how fast a dwell accumulator bleeds off on a contrary sample, as a
    #: multiple of the frame interval.  Never 0: an accumulator that only rises
    #: can never be wrong.
    leak: float = 1.0

    @validator(
        "ear_closed",
        "ear_open_median",
        "eye_prob_closed",
        "eye_prob_open",
        "ear_hysteresis",
        "mar_yawn",
        "mar_close",
        "mouth_prob_open",
        "mouth_prob_closed",
        "pose_min_conf",
        "gaze_min_conf",
        "sunglasses_thr",
        "mask_thr",
        "perclos_warn",
        "perclos_crit",
        "perclos_min_coverage",
        "perclos_exit_margin",
        "phone_conf",
    )
    def _prob(cls, v, field):
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("state.%s must be in [0, 1], got %r" % (field.name, v))
        return float(v)

    @validator(
        "eye_close_dwell_s",
        "eye_open_dwell_s",
        "perclos_window_s",
        "perclos_min_span_s",
        "perclos_min_rate_hz",
        "max_gap_s",
        "stale_after_s",
        "face_lost_s",
        "microsleep_s",
        "fatigue_window_s",
        "gaze_warn_s",
        "gaze_crit_s",
        "distraction_gaze_s",
        "distraction_phone_s",
        "pose_max_reproj_px",
        "gaze_yaw_deg",
        "gaze_pitch_deg",
    )
    def _strictly_positive(cls, v, field):
        if float(v) <= 0.0:
            raise ValueError("state.%s must be > 0, got %r" % (field.name, v))
        return float(v)

    @validator("leak")
    def _leak(cls, v):
        if not (0.0 < float(v) <= 4.0):
            raise ValueError("state.leak must be in (0, 4]")
        return float(v)

    @validator("fatigue_yawns")
    def _yawns(cls, v):
        if v < 1:
            raise ValueError("state.fatigue_yawns must be >= 1")
        return v

    @root_validator
    def _consistent(cls, values):
        if values.get("perclos_crit", 0.0) < values.get("perclos_warn", 0.0):
            raise ValueError("state.perclos_crit must be >= state.perclos_warn")
        if values.get("gaze_crit_s", 0.0) < values.get("gaze_warn_s", 0.0):
            raise ValueError("state.gaze_crit_s must be >= state.gaze_warn_s")
        if values.get("eye_prob_open", 1.0) < values.get("eye_prob_closed", 0.0):
            raise ValueError("state.eye_prob_open must be >= state.eye_prob_closed")
        if values.get("mouth_prob_open", 1.0) < values.get("mouth_prob_closed", 0.0):
            raise ValueError("state.mouth_prob_open must be >= state.mouth_prob_closed")
        ear_closed = values.get("ear_closed")
        ear_open = values.get("ear_open_median")
        if ear_closed is not None and ear_open is not None and ear_closed >= ear_open:
            raise ValueError(
                "state.ear_closed (%r) must be below state.ear_open_median (%r): a "
                "closed eye reads a SMALLER ratio than an open one" % (ear_closed, ear_open)
            )
        if ear_open is not None and float(ear_open) > 0.55:
            raise ValueError(
                "state.ear_open_median %r is on the OLD broken-landmark EAR scale. The "
                "corrected roll-invariant EAR reads 0.21-0.40 on open eyes; use "
                "dms.geometry.ear.EAR_OPEN_MEDIAN_RECOMMENDED (0.27)." % ear_open
            )
        return values


# ------------------------------------------------------------------ IO / ops


class AlertsConfig(Strict):
    gpio_pin: Optional[int] = None
    buzzer: bool = False
    #: reserved: no vehicle-speed input exists on this unit, so enabling it
    #: would silence real alerts unconditionally.  The validator refuses true.
    mute_when_parked: bool = False

    @validator("gpio_pin")
    def _pin(cls, v):
        if v is not None and not (1 <= v <= 40):
            raise ValueError("alerts.gpio_pin must be a BOARD pin 1..40")
        return v

    @validator("mute_when_parked")
    def _mute(cls, v):
        if v:
            raise ValueError(
                "alerts.mute_when_parked is not implemented: there is no vehicle speed "
                "input on this unit, so enabling it would silence alerts unconditionally"
            )
        return v


class ClipsConfig(Strict):
    enabled: bool = False
    dir: str = "/var/lib/dms/clips"
    pre_s: float = 10.0
    post_s: float = 5.0
    segment_s: float = 10.0
    quota_mb: int = 2048
    retain_h: int = 24
    ffmpeg: str = "/usr/bin/ffmpeg"
    bitrate: int = 4000000
    min_free_mb: float = 256.0

    _v_pre = validator("pre_s", allow_reuse=True)(_positive("clips.pre_s"))
    _v_post = validator("post_s", allow_reuse=True)(_positive("clips.post_s"))
    _v_seg = validator("segment_s", allow_reuse=True)(_positive("clips.segment_s"))

    @validator("quota_mb", "retain_h", "bitrate")
    def _pos_int(cls, v, field):
        if v <= 0:
            raise ValueError("clips.%s must be > 0" % field.name)
        return v

    @validator("min_free_mb")
    def _free(cls, v):
        if float(v) < 0.0:
            raise ValueError("clips.min_free_mb must be >= 0")
        return float(v)


class EventsConfig(Strict):
    enabled: bool = True
    #: empty -> data/events.jsonl when source.dev, else /var/lib/dms/events.jsonl
    path: str = ""
    #: never | interval | critical | always.  "critical" fsyncs every CRITICAL
    #: record immediately and everything else at most every fsync_interval_s,
    #: which bounds power-loss loss to one second of non-critical edges.
    fsync: str = "critical"
    fsync_interval_s: float = 1.0
    max_mb: float = 64.0
    backups: int = 2
    min_free_mb: float = 64.0
    #: fail startup rather than silently falling back to the tmpfs path.
    strict: bool = False

    @validator("fsync")
    def _fsync(cls, v):
        allowed = {"never", "interval", "critical", "always"}
        if v not in allowed:
            raise ValueError("events.fsync must be one of %s" % sorted(allowed))
        return v

    @validator("backups")
    def _backups(cls, v):
        if v < 0:
            raise ValueError("events.backups must be >= 0")
        return v

    @validator("fsync_interval_s", "max_mb", "min_free_mb")
    def _non_negative(cls, v, field):
        if float(v) < 0.0:
            raise ValueError("events.%s must be >= 0" % field.name)
        return float(v)


class DisplayConfig(Strict):
    enabled: bool = False
    #: overlay render scale; 0.5 renders a 640x360 preview of a 720p frame at
    #: roughly a third of the cost.
    scale: float = 1.0

    @validator("scale")
    def _scale(cls, v):
        if not (0.0 < float(v) <= 1.0):
            raise ValueError("display.scale must be in (0, 1]")
        return float(v)


class PrivacyConfig(Strict):
    """Imagery and telemetry egress switches.  Both default OFF and are enforced.

    ``record_faces`` gates the clip ring, ``--save-video``, ``--save-preview``
    and ``display.enabled``.  ``telemetry`` gates periodic driver-state
    snapshots in the event log.  Alert EDGES are always written: they are the
    safety record and they carry no imagery.
    """

    record_faces: bool = False
    telemetry: bool = False


class HealthConfig(Strict):
    bind: str = "127.0.0.1"
    port: int = 8088
    #: explicit opt-in to a non-loopback bind; /healthz publishes live alert state.
    allow_remote: bool = False
    #: path to a root-owned file holding a bearer token; empty disables auth.
    token_file: str = ""
    max_connections: int = 16
    #: make a failed bind fatal at startup instead of running blind.
    required: bool = False
    #: /healthz answers 503 when the snapshot is older than this.
    stale_after_s: float = 5.0

    @validator("port")
    def _port(cls, v):
        if not (0 <= v <= 65535):
            raise ValueError("health.port must be 0..65535")
        return v

    @validator("max_connections")
    def _conns(cls, v):
        if not (1 <= v <= 256):
            raise ValueError("health.max_connections must be 1..256")
        return v

    _v_stale = validator("stale_after_s", allow_reuse=True)(_positive("health.stale_after_s"))

    @root_validator
    def _remote(cls, values):
        bind = str(values.get("bind", ""))
        if not values.get("allow_remote") and not is_loopback(bind):
            raise ValueError(
                "health.bind=%s is not loopback and health.allow_remote is false; "
                "/healthz publishes live driver alert state" % bind
            )
        return values


def is_loopback(bind: str) -> bool:
    """True when ``bind`` cannot receive traffic from another host.

    Mirrors :func:`dms.io.health.is_loopback` without importing it, so the
    schema stays independent of the ops package.
    """
    b = (bind or "").strip()
    if b in ("", "127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"):
        return True
    return b.startswith("127.")


class PowerConfig(Strict):
    """Recorded for the deploy scripts.  The application never calls nvpmodel."""

    nvpmodel: int = 8
    jetson_clocks: bool = False


class PerfConfig(Strict):
    """Latency budget.  Consumed by :meth:`dms.perf.LatencyBudget.from_config`."""

    #: samples retained per stage (300 = 20 s at 15 FPS).
    window: int = 300
    gate_e2e_p95_ms: float = 66.0
    gate_peak_rss_mb: float = 5000.0
    #: per-stage target overrides in ms; unknown stage names are kept and
    #: reported rather than dropped.
    targets: Dict[str, float] = Field(default_factory=dict)

    @validator("window")
    def _window(cls, v):
        if not (20 <= v <= 100000):
            raise ValueError("perf.window must be 20..100000")
        return v

    _v_gate = validator("gate_e2e_p95_ms", allow_reuse=True)(_positive("perf.gate_e2e_p95_ms"))
    _v_rss = validator("gate_peak_rss_mb", allow_reuse=True)(_positive("perf.gate_peak_rss_mb"))

    @validator("targets")
    def _targets(cls, v):
        for name, value in v.items():
            if float(value) <= 0.0:
                raise ValueError("perf.targets.%s must be > 0" % name)
        return dict((str(k), float(x)) for k, x in v.items())


class ForwardZero(Strict):
    """Head-pose zero for the installed camera, in degrees.  See ``calibration``."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


# ------------------------------------------------------------------ top level


class AppConfig(Strict):
    source: SourceConfig = Field(default_factory=SourceConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    models: ModelsConfig
    state: StateConfig = Field(default_factory=StateConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    clips: ClipsConfig = Field(default_factory=ClipsConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)
    driver_roi: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.65, 1.0])
    #: if False, an empty seat ROI means "no driver" rather than promoting a
    #: passenger.  False is correct for a multi-occupant clip.
    driver_fallback: bool = True
    forward_zero: ForwardZero = Field(default_factory=ForwardZero)
    #: Tri-state.  ``None`` (the default, and what every shipped profile leaves
    #: it as) means "infer it from forward_zero being non-zero", which is what
    #: :meth:`dms.state.fusion.FusionConfig.from_app_config` does.  ``true`` or
    #: ``false`` is an explicit human statement that overrides the inference,
    #: and is what ``scripts/calibrate_forward.py`` writes into the calibration
    #: file after a real calibration.  Gaze-derived alerts are suppressed
    #: whenever this resolves false: an uncalibrated pose zero cannot produce a
    #: trustworthy "looking away" decision.
    forward_zero_calibrated: Optional[bool] = None
    seat: str = "lhd"
    #: per-unit override file merged over this profile, resolved next to it.
    #: Empty disables the merge.
    calibration: str = "vehicle.yaml"
    #: verify every engine sha256 against engines/MANIFEST.json and refuse to
    #: start on a mismatch or an unlisted engine.
    require_engines: bool = False

    @validator("driver_roi")
    def _roi(cls, v):
        if len(v) != 4:
            raise ValueError("driver_roi must be [x1, y1, x2, y2] normalised")
        if any(not (0.0 <= float(x) <= 1.0) for x in v):
            raise ValueError("driver_roi entries must be in [0, 1]")
        if not (v[0] < v[2] and v[1] < v[3]):
            raise ValueError("driver_roi must have x1 < x2 and y1 < y2")
        return [float(x) for x in v]

    @validator("seat")
    def _seat(cls, v):
        if v not in ("lhd", "rhd"):
            raise ValueError("seat must be lhd or rhd")
        return v

    @root_validator
    def _cross_section(cls, values):
        models = values.get("models")
        privacy = values.get("privacy")
        clips = values.get("clips")
        display = values.get("display")
        if (
            models is not None
            and models.backends.gaze == "model"
            and not models.research_models_allowed
        ):
            raise ValueError(
                "models.backends.gaze='model' requires models.research_models_allowed=true. "
                "The gaze weights are Gaze360-trained (research / non-commercial only) and "
                "must never be enabled on a production profile."
            )
        # The imagery switches are NOT cross-checked here.  Each has a runtime
        # enforcement point that refuses and says why - the clip ring's own
        # privacy gate, and the startup check in dms/app.py for the display and
        # the --save-* flags - and duplicating the rule in the schema would
        # forbid the very combination those defensive layers exist to handle.
        del privacy, clips, display
        return values

    # -------------------------------------------------------------- helpers --
    @property
    def is_replay(self) -> bool:
        """True for a lab/file/test source, where EOS means "clip finished"."""
        return self.source.type in ("file", "test")

    def imagery_allowed(self) -> bool:
        """Whether any path may render or persist driver imagery."""
        return bool(self.privacy.record_faces)


# --------------------------------------------------------------------- loader


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _format_errors(path: str, exc: ValidationError) -> str:
    lines = ["invalid config %s:" % path]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        lines.append("  %s: %s" % (loc or "<root>", err.get("msg", "invalid")))
    return "\n".join(lines)


def calibration_path(path: str, raw: Optional[dict] = None) -> str:
    """Absolute path of the per-unit calibration file for ``path``, or ``""``.

    Reads ``calibration:`` out of the raw YAML rather than the parsed model,
    because the merge has to happen before validation.
    """
    name = "vehicle.yaml"
    if raw is not None and "calibration" in raw:
        name = str(raw.get("calibration") or "")
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    return os.path.join(os.path.dirname(os.path.abspath(path)), name)


def load_config(path: str) -> AppConfig:
    """Parse ``path``, merge its calibration file, and validate strictly.

    Raises ``ValueError`` naming every offending key.  Never returns a config
    the frame loop cannot trust.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("config %s must be a YAML mapping" % path)

    veh = calibration_path(path, raw)
    if veh and os.path.isfile(veh):
        with open(veh, "r") as f:
            extra = yaml.safe_load(f) or {}
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError("calibration file %s must be a YAML mapping" % veh)
        raw = _merge(raw, extra)

    try:
        return AppConfig.parse_obj(raw)
    except ValidationError as exc:
        raise ValueError(_format_errors(path, exc))


__all__ = [
    "CAPABILITIES",
    "VALID_BACKENDS",
    "AlertsConfig",
    "AppConfig",
    "BackendsConfig",
    "CameraConfig",
    "CaptureConfig",
    "ClipsConfig",
    "DisplayConfig",
    "EventsConfig",
    "ForwardZero",
    "HealthConfig",
    "ModelsConfig",
    "PerfConfig",
    "PowerConfig",
    "PrivacyConfig",
    "SourceConfig",
    "StateConfig",
    "Strict",
    "calibration_path",
    "is_loopback",
    "load_config",
]
