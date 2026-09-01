from __future__ import annotations

from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, validator


class SourceConfig(BaseModel):
    type: str = "file"
    path: str = "tests/replay/day_driver.mp4"
    dev: bool = True
    device: str = "/dev/video0"
    sensor_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    @validator("type")
    def _type(cls, v: str) -> str:
        allowed = {"csi", "usb", "file", "test"}
        if v not in allowed:
            raise ValueError("source.type must be one of %s" % sorted(allowed))
        return v


class CaptureConfig(BaseModel):
    full_size: Tuple[int, int] = (1280, 720)
    letterbox: Tuple[int, int] = (640, 640)
    letterbox_pad_value: int = 0
    queue_depth: int = 1


class CameraConfig(BaseModel):
    fail_fatal: bool = False


class FaceModelConfig(BaseModel):
    engine: str
    conf: float = 0.5
    iou: float = 0.4
    device: str = "gpu"


class LandmarksModelConfig(BaseModel):
    engine: str
    device: str = "gpu"
    dla_engine: Optional[str] = None


class PoseModelConfig(BaseModel):
    mode: str = "pnp"
    engine: Optional[str] = None
    pnp_fallback: bool = True


class ObjectsModelConfig(BaseModel):
    enabled: bool = False
    engine: str = ""
    every_n: int = 2
    roi: List[float] = Field(default_factory=lambda: [0.0, 0.45, 1.0, 1.0])
    classes: List[str] = Field(default_factory=lambda: ["cell phone"])
    conf: float = 0.50


class ModelsConfig(BaseModel):
    face: FaceModelConfig
    landmarks: LandmarksModelConfig
    pose: PoseModelConfig = PoseModelConfig()
    objects: ObjectsModelConfig = ObjectsModelConfig()


class StateConfig(BaseModel):
    ear_closed: float = 0.21
    mar_yawn: float = 0.65
    perclos_warn: float = 0.20
    perclos_crit: float = 0.40
    microsleep_s: float = 1.5
    gaze_yaw_deg: float = 25.0
    gaze_pitch_deg: float = 20.0
    gaze_warn_s: float = 2.0
    gaze_crit_s: float = 4.0
    face_lost_s: float = 1.0
    assume_moving: bool = True


class AlertsConfig(BaseModel):
    gpio_pin: Optional[int] = None
    buzzer: bool = False
    mute_when_parked: bool = False


class ClipsConfig(BaseModel):
    enabled: bool = False
    dir: str = "/var/lib/dms/clips"
    pre_s: float = 10.0
    post_s: float = 5.0
    quota_mb: int = 2048
    retain_h: int = 24
    ffmpeg: str = "/usr/bin/ffmpeg"


class DisplayConfig(BaseModel):
    enabled: bool = False


class PrivacyConfig(BaseModel):
    record_faces: bool = False
    telemetry: bool = False


class HealthConfig(BaseModel):
    bind: str = "127.0.0.1"
    port: int = 8088
    watchdog_s: int = 30


class PowerConfig(BaseModel):
    nvpmodel: int = 8
    jetson_clocks: bool = False


class ForwardZero(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class AppConfig(BaseModel):
    class Config:
        extra = "forbid"

    source: SourceConfig = SourceConfig()
    capture: CaptureConfig = CaptureConfig()
    camera: CameraConfig = CameraConfig()
    models: ModelsConfig
    state: StateConfig = StateConfig()
    alerts: AlertsConfig = AlertsConfig()
    clips: ClipsConfig = ClipsConfig()
    display: DisplayConfig = DisplayConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    health: HealthConfig = HealthConfig()
    power: PowerConfig = PowerConfig()
    driver_roi: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.65, 1.0])
    forward_zero: ForwardZero = ForwardZero()
    seat: str = "lhd"
    require_engines: bool = False


def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.parse_obj(raw)
