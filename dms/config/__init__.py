"""Configuration: the strict schema and its loader.

``load_config`` is the only supported entry point.  It merges the per-unit
calibration file over the profile and then validates strictly, raising
``ValueError`` naming every offending key rather than returning a config the
frame loop cannot trust.
"""
from dms.config.schema import (
    CAPABILITIES,
    VALID_BACKENDS,
    AppConfig,
    calibration_path,
    is_loopback,
    load_config,
)

__all__ = [
    "CAPABILITIES",
    "VALID_BACKENDS",
    "AppConfig",
    "calibration_path",
    "is_loopback",
    "load_config",
]
