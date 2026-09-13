"""Geometric head pose from six 106-point landmarks via solvePnP.

This is the FALLBACK path.  When ``engines/hpe_mobilenetv2_224_fp16_gpu.engine``
is present, :mod:`dms.infer.headpose` supersedes it and returns
``source=SignalSource.MODEL``; everything here reports
``source=SignalSource.GEOMETRY``.

Units and conventions
---------------------
* Object points: millimetres, OpenCV camera axes (+X image right, +Y image down,
  +Z away from the camera), origin at the nose tip.  See
  :data:`dms.geometry.face106.PNP_OBJECT_POINTS`.
* Returned angles: degrees.
    ``yaw``   > 0  the driver has turned their head toward their own LEFT.
    ``pitch`` > 0  the driver has nodded DOWN (chin toward chest).
    ``roll``  > 0  the driver's head is tilted so the image-right eye sits lower.
  The yaw sign flip relative to the raw Tait-Bryan extraction is explicit in
  :data:`YAW_SIGN` and documented at its definition; the convention matches the
  docstring of :class:`dms.types.HeadPose`.
* ``reprojection_error`` is the RMS residual in pixels, and ``confidence`` is
  that residual normalised against the inter-ocular distance and mapped to
  [0, 1].

Failure behaviour
-----------------
:func:`solve_head_pose_full` NEVER returns a confident wrong angle.  It returns a
``HeadPose`` with ``confidence == 0.0`` and ``source == UNAVAILABLE`` when

* the landmarks are not finite, or the face is smaller than ``min_iod_px``,
* the 106 points disagree with the detector's own 5 keypoints by more than
  ``kps_tol_frac`` of the inter-ocular distance (the crop went to the wrong
  place),
* solvePnP fails, or
* the reprojection RMS exceeds ``max_reproj_frac`` of the inter-ocular distance.

The state machine must suppress GAZE_AWAY when confidence is 0; that alone
removes most of the spurious GAZE_AWAY seen on profile faces.

Camera intrinsics
-----------------
There is no calibration on this vehicle yet.  :func:`camera_matrix` keeps the
historical fx = fy = frame width (about 53 deg horizontal field of view) so that
existing behaviour is reproducible, but a cabin DMS lens is 100-140 deg and
strongly barrel distorted; at 120 deg the correct fx is 0.29 * width, a factor of
3.5 away.  Until real intrinsics exist every absolute angle here is a relative
signal, not a measurement.  :func:`camera_matrix_from_fov` and the ``cam`` /
``dist`` arguments exist so a calibration can be dropped in with no call-site
change, and :data:`DEFAULT_HFOV_DEG` can be set once at startup.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from dms.geometry.face106 import (
    PNP_INDICES,
    PNP_OBJECT_POINTS,
    inter_ocular,
    landmarks_agree_with_kps,
)
from dms.types import HeadPose, SignalSource

#: Set this once at startup when the lens is known.  None keeps fx = fy = width.
DEFAULT_HFOV_DEG = None  # type: Optional[float]

#: Sign that converts the raw rotation about the camera's +Y (down) axis into the
#: project convention "yaw > 0 = driver looks to their own left".
#:
#: Derivation: the model's +X is the driver's LEFT cheek (index 2 of
#: PNP_OBJECT_POINTS is the *image*-left eye corner at x = -45, and for a driver
#: facing the camera image-left is the driver's right).  A rotation Ry(b) sends a
#: point at x = +45 to z' = -45*sin(b), so b > 0 brings the driver's left cheek
#: TOWARD the camera (z decreasing) - which happens when the driver turns their
#: head to their own RIGHT.  Hence the flip.
#:
#: The result is cross-checked without solvePnP in
#: ``test_pnp_yaw_sign_matches_the_documented_convention``: a driver turning to
#: their own left swings the nose tip toward image-right relative to the eye
#: midpoint, and the test asserts the solver and that cue agree.
YAW_SIGN = -1.0


def camera_matrix(width: int, height: int) -> np.ndarray:
    """3x3 pinhole intrinsics. Uses :data:`DEFAULT_HFOV_DEG` when it is set."""
    if DEFAULT_HFOV_DEG:
        return camera_matrix_from_fov(width, height, float(DEFAULT_HFOV_DEG))
    f = float(width)
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def camera_matrix_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """3x3 pinhole intrinsics from a horizontal field of view in degrees."""
    if not (1.0 < hfov_deg < 179.0):
        raise ValueError("hfov_deg must be in (1, 179), got %r" % (hfov_deg,))
    f = (0.5 * float(width)) / math.tan(math.radians(0.5 * hfov_deg))
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from a rotation matrix, single branch.

    Decomposes R = Rz(roll) @ Ry(yaw) @ Rx(pitch).  Unlike
    ``cv2.decomposeProjectionMatrix`` this has exactly one solution, so there is
    no mirrored branch to detect and no +/-180 cliff to patch up.  The gimbal
    case (|yaw| -> 90 deg) is handled by pinning roll to 0 and folding the
    remaining freedom into pitch, which is the standard convention.

    Returns the RAW rotation angles; :func:`solve_head_pose_full` applies
    :data:`YAW_SIGN` on top.
    """
    R = np.asarray(R, dtype=np.float64)
    sy = float(math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    r2d = 180.0 / math.pi
    return yaw * r2d, pitch * r2d, roll * r2d


def _wrap180(deg: float) -> float:
    """Fold an angle in degrees into (-180, 180]. Non-finite input folds to 0.0."""
    d = float(deg)
    if not math.isfinite(d):
        return 0.0
    d = math.fmod(d, 360.0)
    if d <= -180.0:
        d += 360.0
    elif d > 180.0:
        d -= 360.0
    return d


def solve_head_pose_full(
    landmarks106: np.ndarray,
    frame_wh: Tuple[int, int],
    *,
    kps5: Optional[np.ndarray] = None,
    cam: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    min_iod_px: float = 12.0,
    max_reproj_frac: float = 0.12,
    kps_tol_frac: float = 0.25,
) -> HeadPose:
    """Solve head pose and gate it. Always returns a HeadPose, never raises.

    ``max_reproj_frac`` is the RMS reprojection residual allowed, as a fraction
    of the inter-ocular distance.  0.12 was chosen so that a residual of about
    an eighth of the eye separation - roughly one eye width of total
    disagreement across six points - is the point at which the solve is declared
    untrustworthy.  Re-tune it against a labelled cabin clip; it is the single
    knob that decides how often the pose path stays silent.
    """
    lm = np.asarray(landmarks106, dtype=np.float64)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return HeadPose(source=SignalSource.UNAVAILABLE)
    iod = inter_ocular(lm)
    if iod < min_iod_px:
        return HeadPose(source=SignalSource.UNAVAILABLE)
    agree, _err = landmarks_agree_with_kps(lm, kps5, tol_frac=kps_tol_frac)
    if not agree:
        return HeadPose(source=SignalSource.UNAVAILABLE)

    image_pts = np.ascontiguousarray(lm[list(PNP_INDICES)])
    w, h = int(frame_wh[0]), int(frame_wh[1])
    K = camera_matrix(w, h) if cam is None else np.asarray(cam, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64) if dist is None else np.asarray(dist, dtype=np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            PNP_OBJECT_POINTS, image_pts, K, D, flags=cv2.SOLVEPNP_EPNP
        )
        if ok:
            # EPNP is robust but coarse; one iterative refinement from it is both
            # cheap and free of the "converged to the mirror pose" failure that
            # SOLVEPNP_ITERATIVE alone shows on near-planar six-point sets.
            ok, rvec, tvec = cv2.solvePnP(
                PNP_OBJECT_POINTS,
                image_pts,
                K,
                D,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return HeadPose(source=SignalSource.UNAVAILABLE)
    if not ok or rvec is None or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return HeadPose(source=SignalSource.UNAVAILABLE)

    proj, _ = cv2.projectPoints(PNP_OBJECT_POINTS, rvec, tvec, K, D)
    resid = proj.reshape(-1, 2) - image_pts
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    if not math.isfinite(rms):
        return HeadPose(source=SignalSource.UNAVAILABLE)
    tol = max_reproj_frac * iod
    if rms > tol:
        return HeadPose(reprojection_error=rms, source=SignalSource.UNAVAILABLE)

    R, _ = cv2.Rodrigues(rvec)
    yaw_raw, pitch_raw, roll_raw = rotation_to_ypr(R)
    conf = float(max(0.0, min(1.0, 1.0 - rms / tol))) if tol > 0 else 0.0
    return HeadPose(
        yaw=_wrap180(YAW_SIGN * yaw_raw),
        pitch=_wrap180(pitch_raw),
        roll=_wrap180(roll_raw),
        confidence=conf,
        reprojection_error=rms,
        source=SignalSource.GEOMETRY,
    )


def solve_head_pose(
    landmarks106: np.ndarray,
    frame_wh: Tuple[int, int],
    **kwargs,
) -> Optional[Tuple[float, float, float]]:
    """Back-compatible wrapper: ``(yaw, pitch, roll)`` in degrees, or None.

    None now also covers "solved but rejected by the reprojection gate", which
    is the behaviour the drowsiness path wants: no angle at all beats a
    confident wrong one.  Callers that need the confidence and the residual
    should use :func:`solve_head_pose_full`.
    """
    pose = solve_head_pose_full(landmarks106, frame_wh, **kwargs)
    if pose.source is SignalSource.UNAVAILABLE or pose.confidence <= 0.0:
        return None
    return pose.yaw, pose.pitch, pose.roll
