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
