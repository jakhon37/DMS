"""Runtime services: TensorRT execution, manifest enforcement, capture supervision.

Import order matters here: :mod:`dms.runtime.trt_engine` pulls :mod:`dms.runtime.cudart`
back out of this package while it is still initialising, so the engine module is
imported first and the rest follow.
"""
from dms.runtime.trt_engine import (
    Binding,
    DlaHangError,
    EngineBusyError,
    EngineLoadError,
    TrtEngine,
    TrtExecutionError,
    WorkItem,
)
from dms.runtime.manifest import ManifestError, check_runtime, load_manifest
from dms.runtime.preprocess import BgraToBgr, Geometry, Letterbox, bgra_to_bgr
from dms.runtime.supervisor import (
    CaptureEvent,
    CaptureState,
    CaptureSupervisor,
    RollingStats,
)

__all__ = [
    "Binding",
    "BgraToBgr",
    "CaptureEvent",
    "CaptureState",
    "CaptureSupervisor",
    "DlaHangError",
    "EngineBusyError",
    "EngineLoadError",
    "Geometry",
    "Letterbox",
    "ManifestError",
    "RollingStats",
    "TrtEngine",
    "TrtExecutionError",
    "WorkItem",
    "bgra_to_bgr",
    "check_runtime",
    "load_manifest",
]
