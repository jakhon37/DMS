"""Eye aspect ratio.

Units: all inputs are landmark pixels in the full frame; all outputs are
dimensionless ratios (lid separation divided by eye width).

Failure behaviour: every function returns 0.0 - never raises and never returns
NaN - when the eye width degenerates to zero, which is what happens when the
landmark regressor collapses on a crop that missed the face.  Callers must treat
0.0 as "no measurement", not as "eye fully shut"; :func:`face_ear_detail`
reports that distinction explicitly through its ``valid`` flag.

Why this is not a bounding-box ratio any more.  The previous implementation took
``(max_y - min_y) / (max_x - min_x)`` over the eye point cloud in frame
coordinates.  That statistic is a function of head roll: a shut eye is a thin
segment of length L, and rotating it by theta gives an axis-aligned box of
w = L*cos(theta), h = L*sin(theta), so the ratio is tan(theta) - 0.27 at 15 deg
of roll and 0.58 at 30 deg, both above the 0.50 "closed" threshold that was
configured.  Measured on the replay clips the raw value never fell below 0.67 in
410 samples and MICROSLEEP never fired once.  The ratio below is computed in the
eye's own frame (lid separation measured perpendicular to the corner-to-corner
axis) and is therefore invariant to roll by construction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from dms.geometry.face106 import (
    EYE_LID_PAIR_OFFSETS,
    eye_base,
    eye_box,
    eye_corners,
    inter_ocular,
)

#: Soukupova & Cech report ~0.30 for an open eye and <0.15 for a closed one on
#: 6-point dlib landmarks.  The 2d106det lid points sit slightly inside the lid
#: margin, so measured open values on this project's clips run 0.25-0.33.
EAR_OPEN_TYPICAL = 0.29
EAR_CLOSED_TYPICAL = 0.10
# ---------------------------------------------------------------------------
# THRESHOLD SCALE CHANGE - the shipped configs are calibrated against the OLD
# metric and are WRONG for this one.
#
# The previous axis-aligned bounding-box ratio measured 0.67-0.88 on open eyes,
# and every configs/*.yaml carries ``ear_closed: 0.50`` / ``ear_open_median:
# 0.75`` to match.  The roll-invariant ratio here measures 0.21-0.40 on the same
# footage (380 eyes: p10 0.210, median 0.272, p90 0.404), so a threshold of 0.50
# now reads EVERY OPEN EYE AS CLOSED and would latch MICROSLEEP permanently.
#
# :data:`EAR_CLOSED_RECOMMENDED` and :data:`EAR_OPEN_MEDIAN_RECOMMENDED` are the
# replacement values, and they are the single authoritative source for them -
# the config workstream must copy these, not invent new ones.  They are a
# STARTING POINT derived from open-eyed footage plus the published closed-eye
# figure, not a calibration: nothing in this repository contains a labelled eye
# closure, so the final numbers still have to come from a cabin clip.
# ---------------------------------------------------------------------------

#: Replacement for ``state.ear_closed``.  Soukupova & Cech put a closed eye
#: below 0.15 and an open one near 0.30; 0.18 sits just above the closed band
#: and below the measured p10 of 0.210 for open eyes.
EAR_CLOSED_RECOMMENDED = 0.18
#: Replacement for ``state.ear_open_median`` - the measured open-eye median.
#: Note that driver_state derives an effective threshold of
#: ``min(ear_closed, 0.70 * ear_open_median)``, which with these two values is
#: ``min(0.18, 0.19) = 0.18``; with the old pair it was 0.50.
EAR_OPEN_MEDIAN_RECOMMENDED = 0.27

#: A human palpebral fissure is roughly three times wider than it is tall even
#: at full surprise, so a ratio above this is not an eye - it is a landmark fit
#: that has collapsed or scattered.  Measured on the replay clips, 380 eyes ran
#: 0.13-0.40 apart from a handful of outliers up to 1.27, all of which were bad
#: fits.  Values above the cap are reported as invalid, not clamped.
EAR_MAX_PLAUSIBLE = 0.60


def eye_aspect_ratio(eye6: np.ndarray) -> float:
    """Classic Soukupova & Cech EAR on 6 ordered points p1..p6.

    p1/p4 are the corners, p2/p3 the upper lid (p1 -> p4 order) and p6/p5 the
    lower lid.  Kept for callers that already hold a 6-point eye; the 106-point
    path uses :func:`eye_ratio` instead because it has three lid pairs, not two.
    """
    pts = np.asarray(eye6, dtype=np.float64)
    if pts.shape[0] < 6:
        raise ValueError("need 6 eye points, got %d" % pts.shape[0])
    p1, p2, p3, p4, p5, p6 = pts[:6]
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def eye_ratio(landmarks106: np.ndarray, left: bool) -> float:
    """Roll-invariant aspect ratio of one eye from the 106-point set.

    Builds the eye-local frame from the two corner landmarks, measures each of
    the three lid pairs along the axis perpendicular to it, and divides the mean
    separation by the corner-to-corner distance.  Returns 0.0 when the eye is
    degenerate (both corners at the same pixel).
    """
    lm = np.asarray(landmarks106, dtype=np.float64)
    c_lo, c_hi = eye_corners(lm, left)
    axis = c_hi - c_lo
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return 0.0
    u = axis / width
    normal = np.array([-u[1], u[0]], dtype=np.float64)
    base = eye_base(left)
    total = 0.0
    for up_off, lo_off in EYE_LID_PAIR_OFFSETS:
        total += abs(float(np.dot(lm[base + up_off] - lm[base + lo_off], normal)))
    return total / (len(EYE_LID_PAIR_OFFSETS) * width)


def contour_ear(eye_pts: np.ndarray) -> float:
    """Roll-invariant height/width of an unordered eye point cloud.

    Kept because it needs no index ordering at all, so it is the honest fallback
    if the topology in :mod:`dms.geometry.face106` is ever found to be wrong for
    a re-exported engine.  Unlike the old version it uses the point cloud's own
    principal axis rather than the image axes, so it does not read head roll as
    eye opening.
    """
    pts = np.asarray(eye_pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    centred = pts - pts.mean(axis=0, keepdims=True)
    # Principal axis of the cloud = the long (corner-to-corner) direction.
    cov = centred.T @ centred
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    normal = np.array([-u[1], u[0]], dtype=np.float64)
    proj_u = centred @ u
    proj_n = centred @ normal
    width = float(proj_u.max() - proj_u.min())
    height = float(proj_n.max() - proj_n.min())
    if width < 1e-6:
        return 0.0
    return height / width


def face_ear(landmarks106: np.ndarray) -> float:
    """Mean of the two eye ratios. 0.0 means "no usable measurement"."""
    return face_ear_detail(landmarks106)[0]


def face_ear_detail(
    landmarks106: np.ndarray,
    min_inter_ocular_px: float = 12.0,
) -> Tuple[float, float, float, bool]:
    """``(mean, left, right, valid)``.

    ``valid`` is False when the two eyes are closer together than
    ``min_inter_ocular_px`` (the face is too small or too far in profile for the
    lid points to carry information) or when either eye is degenerate.  At
    720p and the intended cabin geometry an inter-ocular distance below about
    12 px corresponds to a face under ~35 px wide, well below the 60 px the
    landmark regressor needs.
    """
    lm = np.asarray(landmarks106, dtype=np.float64)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return 0.0, 0.0, 0.0, False
    iod = inter_ocular(lm)
    left = eye_ratio(lm, True)
    right = eye_ratio(lm, False)
    valid = (
        iod >= min_inter_ocular_px
        and 0.0 < left <= EAR_MAX_PLAUSIBLE
        and 0.0 < right <= EAR_MAX_PLAUSIBLE
    )
    return 0.5 * (left + right), left, right, valid


def eye_openness_asymmetry(landmarks106: np.ndarray) -> float:
    """|left - right| eye ratio.

    Large values mean one eye is occluded or the landmarks are on a profile
    face; the drowsiness path should treat that as unreliable rather than
    averaging a good eye with a garbage one.
    """
    _, left, right, valid = face_ear_detail(landmarks106)
    if not valid:
        return 0.0
    return abs(left - right)


def eye_crop_boxes(
    landmarks106: np.ndarray,
    margin: float = 0.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(left_box, right_box) xyxy in full-frame pixels for the eye-state model.

    Returns None if the landmarks are unusable.  OCEC was trained on tight
    zero-margin eye crops, hence the default.
    """
    lm = np.asarray(landmarks106, dtype=np.float32)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return None
    lb = eye_box(lm, True, margin=margin)
    rb = eye_box(lm, False, margin=margin)
    if lb[2] - lb[0] < 2.0 or rb[2] - rb[0] < 2.0:
        return None
    return lb, rb
