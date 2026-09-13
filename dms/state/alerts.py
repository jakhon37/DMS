"""Alert vocabulary: types, severities, edges and the one event record.

Every alert in the system is an edge-triggered :class:`AlertEvent`.  Nothing
downstream (``dms/io/events.py``, ``dms/io/clips.py``, ``dms/io/metrics.py``,
the overlay) sees per-frame flags - it sees edges, so the JSONL can be replayed
and audited by pairing them.

Edge vocabulary
---------------
``enter``     the alert became active at ``t_mono_s``.
``escalate``  the alert stayed active but its severity ROSE.  Emitted so that a
              WARN that becomes CRITICAL is visible in the log instead of being
              silently mutated; ``dms/io/metrics.py`` counts it with the rising
              edges, which is what an operator means by "a critical was raised".
``exit``      the alert became inactive.  ``severity`` carries the severity the
              alert held at the moment it resolved, so enters/escalates/exits
              balance per type.

Invariant the state machine guarantees: for one alert type the edge sequence is
always ``enter (escalate*) exit``, never two enters without an exit between.

Times are seconds on the same monotonic (or media, on replay) clock the frame
timestamps use.  ``extra`` is a flat ``str -> float`` map so it survives JSONL
and Prometheus labels without nesting; booleans are written as ``0.0``/``1.0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional


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
    THERMAL_THROTTLE = "thermal_throttle"
    ENGINE_FAIL = "engine_fail"
    PIPELINE_STARVED = "pipeline_starved"
    DISK_FULL = "disk_full"
    DROWSINESS_UNRELIABLE = "drowsiness_unreliable"
    DRIVER_CHANGED = "driver_changed"
    MODEL_DEGRADED = "model_degraded"
    DISTRACTION = "distraction"


#: Edge names.  Kept as plain strings because they cross a JSONL boundary.
EDGE_ENTER = "enter"
EDGE_ESCALATE = "escalate"
EDGE_EXIT = "exit"
EDGES = (EDGE_ENTER, EDGE_ESCALATE, EDGE_EXIT)

#: Total order on severity.  ``critical > warn > info``.
SEVERITY_ORDER: Dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARN: 1,
    Severity.CRITICAL: 2,
}


def severity_rank(sev: Severity) -> int:
    """Rank of ``sev`` in :data:`SEVERITY_ORDER`; unknown values rank lowest."""
    return SEVERITY_ORDER.get(sev, 0)


def max_severity(severities: Iterable[Severity]) -> Optional[Severity]:
    """Highest severity in ``severities``, or ``None`` when it is empty."""
    best: Optional[Severity] = None
    for s in severities:
        if best is None or severity_rank(s) > severity_rank(best):
            best = s
    return best


@dataclass
class AlertEvent:
    t_mono_s: float
    type: AlertType
    severity: Severity
    edge: str  # enter | escalate | exit
    track_id: Optional[int] = None
    extra: Dict[str, float] = field(default_factory=dict)
