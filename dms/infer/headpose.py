"""Head pose from a 6DRepNet-style rotation-matrix regressor.

Engine contract (MANIFEST role ``head_pose``)::

    IN   "input"           FLOAT [1, 3, 224, 224]
    OUT  "rotation_matrix" FLOAT [1, 3, 3]

Preprocessing: RGB; crop = the detector box put through
:func:`dms.infer.sixdrepnet_expand` (which reproduces upstream's height/width
swap on purpose); ``cv2.resize`` to 224x224; ``(px/255 - imagenet_mean) /
imagenet_std``.

Output: a true rotation matrix - the 6D -> R orthonormalisation is inside the
graph and was measured on-board at ``|RR^T - I|max = 5.2e-8``, ``det = 1.0``.
Do not re-orthonormalise.  Euler extraction is the single-branch Tait-Bryan
decomposition in :func:`dms.geometry.pnp.rotation_to_ypr`, so unlike the PnP
path there is no +/-180 fold to undo.

Units: degrees.  Sign convention as documented on :class:`dms.types.HeadPose`
(yaw > 0 = driver turned toward their own left, pitch > 0 = nodded down, roll >
0 = image-right eye lower).  :data:`MODEL_YAW_SIGN` / :data:`MODEL_PITCH_SIGN` /
:data:`MODEL_ROLL_SIGN` map the network's native convention onto that.  They
were MEASURED against two references that do not use the network - a pure
landmark yaw cue and the landmark eye-axis roll - over 312 faces from the replay
clips; the numbers are at their definition.  They must be re-checked once a real
cabin camera exists, because a mirrored or rotated mounting changes them.

Failure behaviour: :meth:`HeadPoseEstimator.infer` always returns a
:class:`dms.types.HeadPose`.  When the engine is absent or fails it falls back
to :func:`dms.geometry.pnp.solve_head_pose_full` (``source=GEOMETRY``); when
that is also unusable it returns ``source=UNAVAILABLE`` with
``confidence=0.0``.  It never raises and never returns a plausible constant.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from dms.geometry.pnp import _wrap180, rotation_to_ypr, solve_head_pose_full
from dms.infer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelWrapper,
    crop_patch,
    sixdrepnet_expand,
    to_blob,
)
from dms.types import HeadPose, SignalSource

#: Native-convention -> project-convention sign map for the 6DRepNet head.
#:
#: MEASURED, NOT ASSUMED.  312 faces were sampled across the nine replay clips
#: and the network's raw Euler angles were regressed against two references that
#: do not use the network at all:
#:
#:   * a pure-landmark yaw cue - the signed offset of the nose tip from the eye
#:     centre midpoint, normalised by inter-ocular distance and measured along
#:     the eye axis so head roll cannot leak into it.  A driver turning toward
#:     their own left swings the nose toward image-right, so a positive cue is a
#:     positive project-convention yaw by definition
#:     (:func:`yaw_cue_from_landmarks`);
#:   * the landmark eye-axis roll from
#:     :func:`dms.geometry.face106.roll_from_eyes`;
#:   * the geometric solve in :mod:`dms.geometry.pnp`, whose own conventions are
#:     independently pinned by ``tests/test_pnp_track.py``.
#:
#: Measured on 287 samples after dropping 25 near-gimbal outliers:
#:
#:     corr(raw yaw,   nose cue)   = -0.960     raw yaw   = -0.805 * pnp_yaw
#:     corr(raw roll,  eye-axis)   = +0.746     raw roll  = +0.864 * lmk_roll
#:     corr(raw pitch, pnp pitch)  = -0.578     raw pitch = -0.941 * pnp_pitch
#:
#: Hence yaw and pitch are negated and roll is not.  The three constants are
#: pinned by ``tests/test_infer_models.py::test_headpose_signs_match_the_
#: captured_calibration``, which replays the captured sample set.
#:
#: HONESTY NOTE ON PITCH.  The pitch correlation is only -0.58, against a
#: geometric solve that is itself the weakest of the three angles, on dashcam
#: montage footage rather than cabin footage.  The SIGN is safe - a +0.58 would
#: have been needed to argue the other way - but the SCALE is not established,
#: and no nod-off threshold should be tuned against this angle until it has been
#: checked on a real driver-facing clip.
MODEL_YAW_SIGN = -1.0
MODEL_PITCH_SIGN = -1.0
MODEL_ROLL_SIGN = 1.0

#: Below this fraction of the expanded crop being real image, a MODEL result is
#: refused: the network would be reading zero padding as cheek.
MIN_CROP_INSIDE = 0.60

#: Roll disagreement, in degrees, between the network and the landmark eye axis
#: at which confidence has fallen to zero.  The eye axis is a direct measurement
#: of in-plane rotation, so a network that disagrees with it by more than this
#: is looking at something else.
ROLL_DISAGREE_DEG = 35.0


def yaw_cue_from_landmarks(landmarks106: np.ndarray) -> Optional[float]:
    """Sign-only yaw cue from the 106 points, needing no pose solve.

    Returns the nose tip's horizontal offset from the eye-centre midpoint,
    divided by the inter-ocular distance, rotated into the eye-axis frame so
    that head roll does not leak into it.  Positive = the nose has moved toward
    image-right = the driver has turned toward their own left.  Returns None
    when the landmarks are unusable.

    This is a *cue*, not a measurement: it saturates past about 40 degrees and
    it is person-dependent at the few-percent level.  Its job is to fix a sign,
    not to produce an angle.
    """
    from dms.geometry.face106 import (
        LEFT_EYE_CENTRE,
        NOSE_TIP,
        RIGHT_EYE_CENTRE,
        inter_ocular,
    )

    lm = np.asarray(landmarks106, dtype=np.float64)
    if lm.shape[0] != 106 or not np.isfinite(lm).all():
        return None
    iod = inter_ocular(lm)
    if iod < 1e-3:
        return None
    mid = 0.5 * (lm[LEFT_EYE_CENTRE] + lm[RIGHT_EYE_CENTRE])
    axis = lm[RIGHT_EYE_CENTRE] - lm[LEFT_EYE_CENTRE]
    u = axis / iod
    return float(np.dot(lm[NOSE_TIP] - mid, u) / iod)


class HeadPoseEstimator(ModelWrapper):
    """Head pose, model first and solvePnP second."""

    capability = "headpose"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        geometry_fallback: bool = True,
        name: str = "headpose",
    ) -> None:
        super().__init__(engine=engine, cfg=cfg, name=name)
        self.geometry_fallback = bool(geometry_fallback)
        self.input_size = 224
        if engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(engine)
            if len(shape) == 4:
                self.input_size = int(shape[2])

    # -------------------------------------------------------------- model --
    def _infer_model(self, bgr: np.ndarray, bbox: np.ndarray) -> Optional[Tuple[HeadPose, float]]:
        crop_box = sixdrepnet_expand(bbox)
        patch, inside = crop_patch(bgr, crop_box, self.input_size, self.input_size)
        if patch is None or inside < MIN_CROP_INSIDE:
            return None
        blob = to_blob(patch, rgb=True, scale=1.0 / 255.0, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        out = self.run_engine({self._in_names[0]: blob})
        if out is None:
            return None
        R = np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(3, 3)
        if not np.isfinite(R).all():
            self.note_failure(ValueError("non-finite rotation matrix"))
            return None
        yaw, pitch, roll = rotation_to_ypr(R)
        pose = HeadPose(
            yaw=_wrap180(MODEL_YAW_SIGN * yaw),
            pitch=_wrap180(MODEL_PITCH_SIGN * pitch),
            roll=_wrap180(MODEL_ROLL_SIGN * roll),
            confidence=1.0,
            reprojection_error=0.0,
            source=SignalSource.MODEL,
        )
        return pose, inside

    # --------------------------------------------------------------- entry --
    def infer(
        self,
        bgr: np.ndarray,
        bbox: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        *,
        kps5: Optional[np.ndarray] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> HeadPose:
        """Estimate head pose for one face.

        ``bbox`` must be the UNCLIPPED detector box (``FaceDet.box_for_crop()``);
        clipping first and expanding second walks the crop off any face at the
        frame edge.  ``landmarks`` and ``kps5`` are optional: they are used to
        cross-check roll and to drive the geometric fallback.
        """
        if bgr is None or bbox is None:
            return HeadPose(source=SignalSource.UNAVAILABLE)
        if frame_wh is None:
            frame_wh = (int(bgr.shape[1]), int(bgr.shape[0]))

        got = self._infer_model(bgr, bbox) if self.available else None
        if got is not None:
            pose, inside = got
            pose.confidence = self._confidence(pose, landmarks, inside)
            if pose.confidence > 0.0:
                return pose
            # A model result the roll cross-check rejects is worse than no
            # result: fall through rather than publish it.

        if self.geometry_fallback and landmarks is not None:
            return solve_head_pose_full(landmarks, frame_wh, kps5=kps5)
        return HeadPose(source=SignalSource.UNAVAILABLE)

    def _confidence(
        self, pose: HeadPose, landmarks: Optional[np.ndarray], inside: float
    ) -> float:
        """Confidence in [0, 1] from crop coverage and the roll cross-check."""
        conf = float(min(1.0, max(0.0, (inside - MIN_CROP_INSIDE) / (1.0 - MIN_CROP_INSIDE))))
        conf = 0.25 + 0.75 * conf  # a fully-inside crop scores 1.0, a marginal one 0.25
        if landmarks is not None:
            from dms.geometry.face106 import inter_ocular, roll_from_eyes

            lm = np.asarray(landmarks, dtype=np.float64)
            if lm.shape[0] == 106 and np.isfinite(lm).all() and inter_ocular(lm) >= 12.0:
                delta = abs(_wrap180(pose.roll - roll_from_eyes(lm)))
                conf *= max(0.0, 1.0 - delta / ROLL_DISAGREE_DEG)
        return float(min(1.0, max(0.0, conf)))
