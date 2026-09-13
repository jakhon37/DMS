from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    CSI = "csi"
    USB = "usb"
    FILE = "file"
    TEST = "test"


@dataclass
class Frame:
    frame_id: int
    t_mono_ns: int
    full_bgra: Optional[np.ndarray]
    ok: bool = True
    source: SourceType = SourceType.FILE
    width: int = 1280
    height: int = 720


class SignalSource(str, Enum):
    """Which producer computed a signal. Lets alerts weight model output over geometry."""

    GEOMETRY = "geometry"
    MODEL = "model"
    UNAVAILABLE = "unavailable"


@dataclass
class EyeState:
    """Per-eye openness. `prob_open` is a model probability; `ear` is the geometric fallback."""

    prob_open: float = 0.0
    ear: float = 0.0
    closed: bool = False
    source: SignalSource = SignalSource.UNAVAILABLE


@dataclass
class MouthState:
    prob_open: float = 0.0
    mar: float = 0.0
    open: bool = False
    source: SignalSource = SignalSource.UNAVAILABLE


@dataclass
class HeadPose:
    """Degrees, camera frame. yaw>0 = driver looks to their left."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    confidence: float = 0.0
    reprojection_error: float = 0.0
    source: SignalSource = SignalSource.UNAVAILABLE


@dataclass
class GazeVector:
    """Eye gaze in degrees relative to the head, plus the fused absolute direction."""

    yaw: float = 0.0
    pitch: float = 0.0
    confidence: float = 0.0
    source: SignalSource = SignalSource.UNAVAILABLE


@dataclass
class Occlusion:
    """Face-covering detections that gate which drowsiness signals may be trusted."""

    sunglasses: float = 0.0
    mask: float = 0.0
    eyes_reliable: bool = True
    mouth_reliable: bool = True
    source: SignalSource = SignalSource.UNAVAILABLE


@dataclass
class DriverSignals:
    """Everything the state machine consumes for one driver on one frame."""

    track_id: int = -1
    left_eye: EyeState = None
    right_eye: EyeState = None
    mouth: MouthState = None
    pose: HeadPose = None
    gaze: GazeVector = None
    occlusion: Occlusion = None
    phone_conf: float = 0.0
    identity: Optional[str] = None
    reliable: bool = True

    def __post_init__(self) -> None:
        if self.left_eye is None:
            self.left_eye = EyeState()
        if self.right_eye is None:
            self.right_eye = EyeState()
        if self.mouth is None:
            self.mouth = MouthState()
        if self.pose is None:
            self.pose = HeadPose()
        if self.gaze is None:
            self.gaze = GazeVector()
        if self.occlusion is None:
            self.occlusion = Occlusion()


@dataclass
class ModelHealth:
    """Per-engine liveness, surfaced on /healthz and used to degrade rather than crash."""

    name: str
    loaded: bool = False
    last_ok_mono_s: float = 0.0
    consecutive_failures: int = 0
    mean_ms: float = 0.0
