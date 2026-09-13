"""Eye gaze from MobileGaze, fused with head pose into an absolute direction.

Engine contract (MANIFEST role ``gaze``)::

    IN   "input" FLOAT [1, 3, 448, 448]
    OUT  "yaw"   FLOAT [1, 90]      "pitch" FLOAT [1, 90]

Preprocessing: RGB; the crop is the RAW detector face box with NO expansion;
``cv2.resize`` to 448x448; ``(px/255 - imagenet_mean) / imagenet_std``.

Output: BOTH HEADS ARE LOGITS.  There is no softmax in the graph - a probe row
summed to -160.4, not 1.0.  Decode with the Gaze360 binning (90 bins, 4 degrees
per bin, 180 degree offset)::

    deg = (softmax(logits) * arange(90)).sum() * 4 - 180

Units: degrees, relative to the FACE CROP, i.e. relative to the head.  ``yaw`` is
horizontal and ``pitch`` vertical.

*** LICENCE GATE - READ BEFORE ENABLING ***
The engine is built from Gaze360-trained weights.  The Gaze360 dataset licence is
non-commercial research use only.  The engine existing on disk is not permission
to run it in a product.  This wrapper therefore refuses to run unless it is
constructed with ``allow_research_models=True``, and reports
``source=UNAVAILABLE`` otherwise, with a one-time warning naming the licence.
The registry passes that flag through from ``research_models_allowed``, which a
production profile must leave False.  This is the enforcement the engine-build
report asked for; documentation alone was the gap.

Fusion.  :func:`fuse_absolute_gaze` adds the head pose to the eye gaze to give
the direction the driver is actually looking, which is the quantity a
distraction alert needs: a driver whose head is turned 30 degrees away but whose
eyes are turned 30 degrees back IS looking at the road, and a head-pose-only
GAZE_AWAY fires on them every time.  The sum is a small-angle approximation -
strictly the two rotations compose rather than add - and it is accurate to
better than 2 degrees while both angles stay under about 30 degrees.  Past that
the function reduces confidence rather than pretending.

Failure behaviour: always returns a :class:`dms.types.GazeVector`; never raises.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from dms.geometry.pnp import _wrap180
from dms.infer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelWrapper,
    crop_patch,
    log,
    softmax,
    to_blob,
)
from dms.types import GazeVector, HeadPose, SignalSource

#: Gaze360 head configuration baked into the training of these weights.
GAZE_BINS = 90
GAZE_BIN_WIDTH_DEG = 4.0
GAZE_ANGLE_OFFSET_DEG = 180.0

MIN_CROP_INSIDE = 0.60

#: Beyond this combined |angle| the additive fusion is no longer a small-angle
#: approximation and confidence is faded out linearly to zero at twice this.
FUSION_LINEAR_LIMIT_DEG = 30.0

LICENCE_NOTE = (
    "mobilenetv2_gaze_448 is trained on Gaze360, whose dataset licence is "
    "non-commercial research use only. Set research_models_allowed=true only in "
    "a research profile."
)


def decode_gaze_bins(logits: np.ndarray) -> float:
    """Gaze360 expectation decode: 90 logits -> one angle in degrees."""
    v = np.asarray(logits, dtype=np.float64).reshape(-1)
    n = v.shape[0]
    p = softmax(v, axis=0)
    idx = np.arange(n, dtype=np.float64)
    return float((p * idx).sum() * GAZE_BIN_WIDTH_DEG - GAZE_ANGLE_OFFSET_DEG)


def fuse_absolute_gaze(pose: Optional[HeadPose], gaze: GazeVector) -> GazeVector:
    """Head pose + head-relative eye gaze -> absolute gaze direction.

    Returns a new :class:`dms.types.GazeVector` whose angles are in the same
    camera-referenced convention as :class:`dms.types.HeadPose`.  When either
    input is unavailable the other is passed through with its own confidence, so
    a system with head pose but no gaze model still produces the head direction
    - clearly labelled as such through ``source``.
    """
    have_pose = pose is not None and pose.source is not SignalSource.UNAVAILABLE and pose.confidence > 0.0
    have_gaze = gaze.source is not SignalSource.UNAVAILABLE and gaze.confidence > 0.0
    if not have_pose and not have_gaze:
        return GazeVector(source=SignalSource.UNAVAILABLE)
    if not have_gaze:
        return GazeVector(
            yaw=pose.yaw,
            pitch=pose.pitch,
            confidence=pose.confidence,
            source=pose.source,
        )
    if not have_pose:
        return GazeVector(
            yaw=gaze.yaw,
            pitch=gaze.pitch,
            confidence=gaze.confidence * 0.5,  # head-relative only: half a measurement
            source=gaze.source,
        )
    yaw = _wrap180(pose.yaw + gaze.yaw)
    pitch = _wrap180(pose.pitch + gaze.pitch)
    mag = max(abs(yaw), abs(pitch))
    fade = 1.0
    if mag > FUSION_LINEAR_LIMIT_DEG:
        fade = max(0.0, 1.0 - (mag - FUSION_LINEAR_LIMIT_DEG) / FUSION_LINEAR_LIMIT_DEG)
    return GazeVector(
        yaw=yaw,
        pitch=pitch,
        confidence=float(pose.confidence * gaze.confidence * fade),
        source=SignalSource.MODEL,
    )


class GazeEstimator(ModelWrapper):
    """Head-relative eye gaze. Licence-gated; degrades to UNAVAILABLE."""

    capability = "gaze"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        allow_research_models: bool = False,
        name: str = "gaze",
    ) -> None:
        super().__init__(engine=engine, cfg=cfg, name=name)
        self.allow_research_models = bool(allow_research_models)
        self.input_size = 448
        self._licence_warned = False
        if engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(engine)
            if len(shape) == 4:
                self.input_size = int(shape[2])
        if engine is not None and not self.allow_research_models:
            self._warn_licence()

    def _warn_licence(self) -> None:
        if not self._licence_warned:
            log.warning("GAZE DISABLED BY LICENCE GATE: %s", LICENCE_NOTE)
            self._licence_warned = True

    @property
    def available(self) -> bool:
        return (
            self.engine is not None and not self._disabled and self.allow_research_models
        )

    def infer(
        self,
        bgr: np.ndarray,
        face_xyxy: Optional[np.ndarray],
    ) -> GazeVector:
        """Head-relative gaze for one face, in degrees."""
        if bgr is None or face_xyxy is None:
            return GazeVector(source=SignalSource.UNAVAILABLE)
        if self.engine is not None and not self.allow_research_models:
            self._warn_licence()
            return GazeVector(source=SignalSource.UNAVAILABLE)
        if not self.available:
            return GazeVector(source=SignalSource.UNAVAILABLE)

        patch, inside = crop_patch(bgr, face_xyxy, self.input_size, self.input_size)
        if patch is None or inside < MIN_CROP_INSIDE:
            return GazeVector(source=SignalSource.UNAVAILABLE)
        blob = to_blob(patch, rgb=True, scale=1.0 / 255.0, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        out = self.run_engine({self._in_names[0]: blob})
        if out is None:
            return GazeVector(source=SignalSource.UNAVAILABLE)
        try:
            yaw_logits = out["yaw"]
            pitch_logits = out["pitch"]
        except KeyError:
            # Binding names differ from the manifest: fall back to declared order.
            if len(self._out_names) < 2:
                self.note_failure(ValueError("gaze engine has %d outputs" % len(self._out_names)))
                return GazeVector(source=SignalSource.UNAVAILABLE)
            yaw_logits = out[self._out_names[0]]
            pitch_logits = out[self._out_names[1]]
        yaw = decode_gaze_bins(yaw_logits)
        pitch = decode_gaze_bins(pitch_logits)
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            self.note_failure(ValueError("non-finite gaze decode"))
            return GazeVector(source=SignalSource.UNAVAILABLE)
        conf = float(min(1.0, max(0.0, (inside - MIN_CROP_INSIDE) / (1.0 - MIN_CROP_INSIDE))))
        return GazeVector(
            yaw=_wrap180(yaw),
            pitch=_wrap180(pitch),
            confidence=0.25 + 0.75 * conf,
            source=SignalSource.MODEL,
        )

    def infer_absolute(
        self,
        bgr: np.ndarray,
        face_xyxy: Optional[np.ndarray],
        pose: Optional[HeadPose],
    ) -> Tuple[GazeVector, GazeVector]:
        """``(head_relative, absolute)`` in one call."""
        rel = self.infer(bgr, face_xyxy)
        return rel, fuse_absolute_gaze(pose, rel)
