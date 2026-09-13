"""Eye open/closed from the OCEC classifier, with EAR as the fallback.

Engine contract (MANIFEST roles ``eye_state_pair`` and ``eye_state``)::

    IN   "images"    FLOAT [2, 3, 24, 40]   OUT  "prob_open" FLOAT [2]   (pair)
    IN   "images"    FLOAT [1, 3, 24, 40]   OUT  "prob_open" FLOAT [1]   (single)

Preprocessing: RGB; ``cv2.resize(eye_crop, (40, 24))`` - width 40, HEIGHT 24, the
aspect ratio is deliberately NOT preserved; tight eye box with zero margin;
``px / 255`` only, no mean and no std.

Output: ``prob_open`` is already a sigmoid probability that the eye is OPEN.  Do
not apply another sigmoid.  Upstream's decision threshold is 0.50.

Batching: the batch-2 engine costs the same 0.76 ms as the batch-1 engine, so
both eyes always go in one launch when it is available.  Index 0 is the
image-LEFT eye and index 1 is the image-RIGHT eye, matching
:mod:`dms.geometry.face106`.  ``TrtEngine.submit`` requires an exact shape
match, so the batch-2 engine is always fed a full (2, 3, 24, 40) array.  If only
the batch-1 engine built, the wrapper runs it twice instead - the shape is read
from the engine's own binding, so neither case needs a configuration flag.

Units: ``prob_open`` is a probability in [0, 1]; ``ear`` is the dimensionless
roll-invariant ratio from :mod:`dms.geometry.ear`.

Failure behaviour: :meth:`EyeStateEstimator.infer` always returns two
:class:`dms.types.EyeState` values.  With no engine, or on an engine failure, it
falls back to the EAR threshold and reports ``source=GEOMETRY``; when the
landmarks themselves are unusable it reports ``source=UNAVAILABLE`` with
``closed=False``, which the state machine must read as "no evidence", never as
"eyes open".

MEASURED CALIBRATION STATUS - READ BEFORE TUNING MICROSLEEP
-----------------------------------------------------------
OCEC was trained on eye crops produced by a whole-body detector, not on boxes
derived from 2d106det, so the operating point had to be measured rather than
assumed.  Re-measured 2026-09-13 through THIS wrapper (not a bespoke script),
every 5th frame of all nine replay clips, 6350 eyes::

    eyes reaching the MODEL path      1012  (16%)
    eyes below MIN_EYE_BOX_PX         5338  (84%, geometric ratio instead)

    on the MODEL path:
      median prob_open                     0.971
      fraction OCEC calls closed           0.241
      fraction the eye ratio calls closed  0.074
      corr(prob_open, eye ratio)          +0.29

So on faces big enough to pass the floor OCEC agrees in direction (+0.29) but
still calls three times as many eyes closed as the geometry does.  That is a
false-MICROSLEEP source, and it is why MICROSLEEP must not be driven by
``prob_open`` alone until a labelled cabin clip exists.  Below the floor it is
worse than noise - across all 6350 eyes the correlation is -0.00 and 38% are
called closed - which is what :data:`MIN_EYE_BOX_PX` exists to refuse.  The eye
boxes on this footage are a median of 19 source pixels wide because the replay
clips are dashcam montages; a real cabin camera lands above the floor.

CROP GEOMETRY: ROLL NORMALISATION WAS TRIED AND MEASURABLY DID NOT HELP
----------------------------------------------------------------------
The crops are axis-aligned boxes in image space, so a rolled head rotates the
eye out of upright - visibly so: on ``guy_rear_ended`` (roll +20 to +34 deg) the
crop is a diagonal slash of brow, spectacle frame and lid.  The obvious fix is to
rotate the crop upright with ``-roll_from_eyes(landmarks)`` before the resize.
It was implemented and measured against the roll-invariant eye ratio, over 1652
eyes at or above the 20 px floor, with the crop RECTANGLE HELD THE SAME SIZE so
that rotation is the only variable (a naive upright crop is also tighter, which
would confound the two)::

    subset                    axis-aligned            upright, same size
                              p(closed) FP    corr    p(closed) FP    corr
    all (n=1652)              0.241     0.208 +0.27   0.228     0.196 +0.24
    |roll| <10 deg (n=1420)   0.242     0.209 +0.28   0.226     0.194 +0.25
    |roll| >=15 deg (n=134)   0.194     0.168 +0.27   0.209     0.178 +0.25
    |roll| >=25 deg (n=36)    0.250     0.257 -0.05   0.306     0.314 +0.18

    FP = fraction called closed among eyes the geometric ratio calls OPEN

Rotation moves nothing outside the noise, and in the high-roll subset - exactly
the regime it was supposed to rescue - it is slightly WORSE on the false-closed
rate.  So it is NOT shipped.  This is a negative result on a proxy reference
(the geometric ratio, not labels) with only 134 eyes above 15 degrees of roll, so
it is not proof that rotation cannot help; it is proof that it does not help
enough to be visible on this corpus, and shipping a crop change that no
measurement supports would be worse than leaving the simple one in place.

A labelled cabin clip with real eye closures is still required before
``prob_open_threshold`` or ``ear_closed`` can be called calibrated; nothing in
this repository contains one.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from dms.geometry.ear import eye_crop_boxes, eye_ratio, face_ear_detail
from dms.infer import ModelWrapper, crop_patch, to_blob
from dms.types import EyeState, SignalSource

#: Model decision threshold on P(open).
DEFAULT_PROB_OPEN_THRESHOLD = 0.50
#: Geometric decision threshold on the roll-invariant eye ratio.  Soukupova &
#: Cech's 0.21 for dlib 6-point EAR; the 2d106det lid points sit slightly inside
#: the lid margin so the open-eye value runs a little lower here.
DEFAULT_EAR_CLOSED = 0.18
#: Minimum fraction of the eye crop that must be inside the frame.
MIN_CROP_INSIDE = 0.70
#: Minimum source width of an eye box, in pixels, for OCEC to be trusted.  Below
#: this the 40x24 input is an upscale of a handful of pixels and the classifier
#: was MEASURED to carry no information at all - correlation -0.00 with the
#: geometric ratio over 6350 eyes, while calling 38% of them closed (see the
#: module docstring).  An earlier revision described that as "inverting"; the
#: larger sample says uninformative, not inverted, which is a different and
#: slightly less alarming failure.  20 px of eye corresponds to roughly 55 px of
#: inter-ocular distance.
MIN_EYE_BOX_PX = 20.0


class EyeStateEstimator(ModelWrapper):
    """Per-eye openness, OCEC first and EAR second."""

    capability = "eyestate"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        single_engine=None,
        geometry_fallback: bool = True,
        name: str = "eyestate",
    ) -> None:
        super().__init__(engine=engine or single_engine, cfg=cfg, name=name)
        # The batch-2 engine costs the same as the batch-1 engine, so it is
        # always preferred; the single engine is what the wrapper falls back to
        # when only that one built, and it then runs twice per frame.
        self.single_engine = single_engine if engine is not None else None
        self.geometry_fallback = bool(geometry_fallback)
        self.prob_open_threshold = float(
            self.cfg.get("prob_open_threshold", DEFAULT_PROB_OPEN_THRESHOLD)
        )
        self.ear_closed = float(self.cfg.get("ear_closed", DEFAULT_EAR_CLOSED))
        self.crop_margin = float(self.cfg.get("crop_margin", 0.0))
        self.min_eye_box_px = float(self.cfg.get("min_eye_box_px", MIN_EYE_BOX_PX))
        self.batch = 2
        self.in_w, self.in_h = 40, 24
        if self.engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(self.engine)
            if len(shape) == 4:
                self.batch = int(shape[0])
                self.in_h, self.in_w = int(shape[2]), int(shape[3])

    # -------------------------------------------------------------- model --
    def _probs(
        self, bgr: np.ndarray, boxes: Tuple[np.ndarray, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
        patches = []
        insides = []
        for box in boxes:
            patch, inside = crop_patch(bgr, box, self.in_w, self.in_h)
            if patch is None:
                return None
            patches.append(patch)
            insides.append(inside)
        blobs = [to_blob(p, rgb=True, scale=1.0 / 255.0) for p in patches]
        if self.batch >= 2:
            feed = np.ascontiguousarray(np.concatenate(blobs[: self.batch], axis=0))
            out = self.run_engine({self._in_names[0]: feed})
            if out is None:
                return None
            probs = np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(-1)
        else:
            # Batch-1 engine: two launches, one per eye.
            values = []
            for blob in blobs:
                out = self.run_engine({self._in_names[0]: blob})
                if out is None:
                    return None
                values.append(
                    float(np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(-1)[0])
                )
            probs = np.array(values, dtype=np.float64)
        if probs.shape[0] < 2 or not np.isfinite(probs).all():
            self.note_failure(ValueError("bad prob_open shape %s" % (probs.shape,)))
            return None
        return probs, (insides[0], insides[1])

    # --------------------------------------------------------------- entry --
    def infer(
        self,
        bgr: np.ndarray,
        landmarks106: Optional[np.ndarray],
        *,
        eyes_reliable: bool = True,
    ) -> Tuple[EyeState, EyeState]:
        """``(left_eye, right_eye)`` for the image-left and image-right eye.

        ``eyes_reliable=False`` (sunglasses detected upstream) forces
        ``source=UNAVAILABLE``: neither OCEC nor the EAR can see through a lens,
        and reporting "eyes open" behind sunglasses is exactly the silent failure
        the DROWSINESS_UNRELIABLE alert exists to prevent.
        """
        if landmarks106 is None or bgr is None:
            return EyeState(), EyeState()
        if not eyes_reliable:
            return EyeState(), EyeState()

        mean_ear, ear_l, ear_r, ear_valid = face_ear_detail(landmarks106)
        boxes = eye_crop_boxes(landmarks106, margin=self.crop_margin)

        big_enough = boxes is not None and min(
            float(boxes[0][2] - boxes[0][0]), float(boxes[1][2] - boxes[1][0])
        ) >= self.min_eye_box_px
        if self.available and boxes is not None and big_enough:
            got = self._probs(bgr, boxes)
            if got is not None:
                probs, insides = got
                if min(insides) >= MIN_CROP_INSIDE:
                    return (
                        self._state_from_prob(float(probs[0]), ear_l),
                        self._state_from_prob(float(probs[1]), ear_r),
                    )

        if self.geometry_fallback and ear_valid:
            return (
                self._state_from_ear(ear_l),
                self._state_from_ear(ear_r),
            )
        return EyeState(), EyeState()

    def _state_from_prob(self, prob_open: float, ear: float) -> EyeState:
        p = float(min(1.0, max(0.0, prob_open)))
        return EyeState(
            prob_open=p,
            ear=float(ear),
            closed=p < self.prob_open_threshold,
            source=SignalSource.MODEL,
        )

    def _state_from_ear(self, ear: float) -> EyeState:
        e = float(ear)
        if e <= 0.0:
            return EyeState()
        return EyeState(
            prob_open=0.0,
            ear=e,
            closed=e < self.ear_closed,
            source=SignalSource.GEOMETRY,
        )

    def ear_only(self, landmarks106: np.ndarray, left: bool) -> float:
        """Direct access to the geometric ratio, for calibration tooling."""
        return eye_ratio(landmarks106, left)
