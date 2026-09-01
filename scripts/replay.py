#!/usr/bin/env python3
"""Phase-1 replay: file/test source, no detectors."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dms.app import main

if __name__ == "__main__":
    raise SystemExit(main())
