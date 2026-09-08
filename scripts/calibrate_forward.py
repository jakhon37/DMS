#!/usr/bin/env python3
"""5 s look-forward calibration. Writes configs/vehicle.yaml via dms.app."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dms.app import main

if __name__ == "__main__":
    raise SystemExit(main(["--detect", "--calibrate-forward"] + sys.argv[1:]))
