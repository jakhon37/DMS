"""Driver Monitoring System - Jetson Xavier NX runtime.

One process: supervised GStreamer capture, eleven FP16 TensorRT engines, a
hysteretic alert state machine, and a local ops surface (JSONL events, health,
Prometheus metrics, GPIO, an optional clip ring).

Entry point: ``python -m dms.app --config configs/<profile>.yaml``.
See ``readme.md`` for what is and is not validated.
"""

__version__ = "1.0.0-rc1"
