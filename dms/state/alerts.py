from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class AlertType(str, Enum):
    FACE_LOST = "face_lost"
    MICROSLEEP = "microsleep"
    PERCLOS_HIGH = "perclos_high"
    YAWN = "yawn"
    FATIGUE_CLUSTER = "fatigue_cluster"
    GAZE_AWAY = "gaze_away"
    PHONE = "phone"
    CAMERA_FAIL = "camera_fail"
    ENGINE_FAIL = "engine_fail"


@dataclass
class AlertEvent:
    t_mono_s: float
    type: AlertType
    severity: Severity
    edge: str  # enter | exit
    track_id: Optional[int] = None
    extra: Dict[str, float] = field(default_factory=dict)
