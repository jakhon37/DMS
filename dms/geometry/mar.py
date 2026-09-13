"""Mouth aspect ratio.

Units: landmark pixels in, dimensionless ratio out (inner-lip separation divided
by inner-lip width).  A closed mouth measures ~0.01-0.05, speech peaks around
0.2-0.3, a yawn exceeds 0.5.

Failure behaviour: returns 0.0 when the mouth width degenerates.  0.0 means "no
measurement", never "mouth wide open"; :func:`mouth_aspect_ratio_detail` carries
the explicit validity flag.

The previous implementation used indices 87/93/89/95 as the mouth corners and
upper/lower lip.  On the engine this project ships those indices are the
*image-right eye* (87-96), so the "MAR" being logged was an eye statistic - which
is why a closed mouth rested at 0.52.  The correct mouth block is 52-71; see
:mod:`dms.geometry.face106` for how that was established.

The ratio is measured on the INNER lip ring.  The outer ring moves when the lips
purse without the jaw opening, so an outer-lip ratio reads speech and smiling as
partial yawns; the inner ring is close to zero unless the mouth is genuinely
open.  Like the EAR, the separation is projected onto the axis perpendicular to
the corner-to-corner line, so the value does not change with head roll.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from dms.geometry.face106 import (
    INNER_LIP_PAIRS,
    MOUTH_INNER_LEFT,
    MOUTH_INNER_RIGHT,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    inter_ocular,
)

MAR_CLOSED_TYPICAL = 0.02
MAR_YAWN_TYPICAL = 0.60

# ---------------------------------------------------------------------------
# THRESHOLD SCALE CHANGE - as with the EAR, the shipped configs are calibrated
# against the OLD metric.  ``mar_yawn: 0.65`` was set against a statistic that
# rested at 0.52 on a CLOSED mouth (it was measuring the image-right eye).  The
# inner-lip ratio here rests at 0.01-0.02 closed and measured a median of 0.115
# with a p90 of 0.271 over 66 replay faces, most of whom were talking.  A
# threshold of 0.65 against this metric means YAWN can essentially never fire.
# ---------------------------------------------------------------------------

#: Replacement for ``state.mar_yawn``.  Set well above the measured talking p90
#: of 0.271 and well below a real yawn, which opens the inner lip to more than
#: half the mouth width.  UNVALIDATED: no clip in this repository contains a
#: labelled yawn, so this is a defensible starting point, not a calibration.
MAR_YAWN_RECOMMENDED = 0.45


def mouth_aspect_ratio(landmarks106: np.ndarray) -> float:
    """Inner-lip opening over inner-lip width. 0.0 means "no measurement"."""
    return mouth_aspect_ratio_detail(landmarks106)[0]


def mouth_aspect_ratio_detail(
    landmarks106: np.ndarray,
    min_inter_ocular_px: float = 12.0,
) -> Tuple[float, bool]:
    """``(mar, valid)``.

    ``valid`` is False when the face is too small for the 20 mouth points to
    separate (inter-ocular below ``min_inter_ocular_px``) or when the mouth
    corners coincide.
    """
    lm = np.asarray(landmarks106, dtype=np.float64)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return 0.0, False
    axis = lm[MOUTH_INNER_RIGHT] - lm[MOUTH_INNER_LEFT]
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return 0.0, False
    u = axis / width
    normal = np.array([-u[1], u[0]], dtype=np.float64)
    total = 0.0
    for up, lo in INNER_LIP_PAIRS:
        total += abs(float(np.dot(lm[up] - lm[lo], normal)))
    mar = total / (len(INNER_LIP_PAIRS) * width)
    valid = inter_ocular(lm) >= min_inter_ocular_px
    return mar, valid


def mouth_width_over_inter_ocular(landmarks106: np.ndarray) -> float:
    """Outer mouth width normalised by inter-ocular distance.

    A stable per-person constant near 0.9-1.1 for a frontal face; it collapses
    as the head turns, so it is a cheap independent profile detector that does
    not need a pose solve.  Returns 0.0 when unmeasurable.
    """
    lm = np.asarray(landmarks106, dtype=np.float64)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return 0.0
    iod = inter_ocular(lm)
    if iod < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[MOUTH_RIGHT] - lm[MOUTH_LEFT]) / iod)
