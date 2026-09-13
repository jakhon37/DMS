"""Mouth open/closed from the VSDLM classifier, with MAR as the fallback.

Engine contract (MANIFEST role ``mouth_state``)::

    IN   "images"    FLOAT [1, 3, 30, 48]   OUT  "prob_open" FLOAT [1]

Preprocessing: RGB; ``cv2.resize(mouth_crop, (48, 30))`` - width 48, HEIGHT 30;
``px / 255`` only.  The crop is the mouth box grown by ASYMMETRIC margins
(top 2, bottom 6, left 2, right 2) taken from upstream's ``demo_vsdlm.py``.
Those margins are FRAME pixels, applied to the box before any resize, which is
what :func:`vsdlm_crop_box` now does; see its docstring for the correction and
its measured effect.

Output: ``prob_open`` is already a sigmoid probability that the mouth is OPEN.
Threshold 0.50.

Units: probability in [0, 1]; ``mar`` is the dimensionless inner-lip ratio from
:mod:`dms.geometry.mar`.

MEASURED CALIBRATION STATUS - VSDLM CALLS HALF OF ALL MOUTHS OPEN HERE
----------------------------------------------------------------------
Re-measured 2026-09-13 through THIS wrapper, every 5th frame of all nine replay
clips, 3175 mouths (1667 of them, 53%, above :data:`MIN_MOUTH_BOX_PX` and so on
the MODEL path)::

    on the MODEL path (n=1667):
      median prob_open                  0.562
      fraction VSDLM calls OPEN         0.524
      fraction the inner-lip ratio      0.183      (median MAR 0.128)
        calls open (MAR >= 0.25)
      corr(prob_open, MAR)             +0.14

That is the headline and it is not good: on this footage VSDLM calls the
majority of mouths open while the geometry calls fewer than one in five open,
and the two agree only weakly.  A previous revision of this docstring reported
"median 0.111, 38% open, corr +0.21" from a 66-face sample; that sample was too
small to be representative and the number was wrong - the correction is the
whole point of this block.  At the shipped 0.50 threshold ``MouthState.open`` is
true about half the time, which is why ``dms.app`` raises two YAWN alerts in 30 s
of ``minivan`` footage in which nobody yawns (the 15 s YAWN cooldown is the only
thing keeping that from being continuous).  YAWN and FATIGUE must not be trusted
on this signal until it is calibrated against a clip containing real yawns; the
inner-lip ratio is the better-behaved of the two channels today.

Crop geometry that was tried and rejected.  Roll normalisation - rotating the
crop upright by ``-roll_from_eyes(landmarks)`` before the resize - was
implemented and measured, with the crop rectangle held the same size so rotation
is the only variable.  Against MAR, over 2778 mouths at least 30 px wide,
reporting the fraction called open among mouths MAR calls CLOSED::

    subset                  scaled px   frame px    upright     upright
                            (old)       (upstream)  tight box   same-size box
    all        (n=2778)     0.508       0.523       0.516       0.534
    |roll|<10  (n=2071)     0.505       0.509       0.495       0.511
    |roll|>=15 (n= 250)     0.466       0.520       0.602       0.624

Rotation makes the false-open rate WORSE in exactly the high-roll regime it was
meant to rescue (0.466 -> 0.624), so it is not shipped.  The rotation is
geometrically correct - the rendered crops go from a diagonal slash to a level
lip line - the classifier simply does not reward it here.  The change from the
scaled margins to upstream's frame-pixel margins costs 0.015 on the same
statistic and is kept because the previous code claimed upstream fidelity it did
not have; see :func:`vsdlm_crop_box`.

:data:`MIN_MOUTH_BOX_PX` refuses the model below the resolution it was measured
at.  The 0.50 threshold is upstream's, not a calibrated one.

Failure behaviour: always returns a :class:`dms.types.MouthState`.  Model first,
MAR second, ``UNAVAILABLE`` third.  A masked mouth (``mouth_reliable=False``)
returns ``UNAVAILABLE`` rather than a guess - a mask flattens the lip contour and
both the classifier and the MAR read it as a permanently closed mouth, which
would silently disable YAWN detection.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from dms.geometry.face106 import mouth_box
from dms.geometry.mar import mouth_aspect_ratio_detail
from dms.infer import ModelWrapper, crop_patch, to_blob
from dms.types import MouthState, SignalSource

DEFAULT_PROB_OPEN_THRESHOLD = 0.50
#: Inner-lip ratio at which the mouth counts as open for the geometric path.
#: A closed mouth measures ~0.02 and a yawn >0.5, so 0.25 sits in the gap.
DEFAULT_MAR_OPEN = 0.25
MIN_CROP_INSIDE = 0.70
#: Minimum source width of the mouth box, in pixels, for VSDLM to be trusted.
#: A mouth is about half the inter-ocular distance wide, so 30 px of mouth is
#: roughly 60 px of IOD - the same resolution floor the eye classifier needs.
MIN_MOUTH_BOX_PX = 30.0

#: Upstream demo_vsdlm.py margins, in crop pixels: (top, bottom, left, right).
VSDLM_MARGIN_PX = (2, 6, 2, 2)


def vsdlm_crop_box(
    mouth_xyxy: np.ndarray,
    out_w: int = 48,
    out_h: int = 30,
    margin_px=VSDLM_MARGIN_PX,
) -> np.ndarray:
    """Grow a mouth box by the upstream margins, IN FRAME PIXELS.

    Upstream ``demo_vsdlm.py`` applies them to the detector's mouth box in the
    frame, before any resize::

        mouth_crop = crop_image_with_margin(image, box, margin_top=2,
                                            margin_bottom=6, margin_left=2,
                                            margin_right=2)
        # x1 = max(int(box.x1) - margin_left, 0) ; y1 = max(int(box.y1) - margin_top, 0)

    ``out_w`` / ``out_h`` are therefore NOT used and are kept only so existing
    call sites do not have to change.  A previous revision multiplied each
    margin by the resize scale and its docstring claimed that was what upstream
    did; it was not.  Measured effect of the correction over 2778 mouths at least
    30 px wide across the nine replay clips: median ``prob_open`` 0.550 -> 0.579,
    fraction called open 0.525 -> 0.538, correlation with the inner-lip ratio
    +0.14 -> +0.13.  Small, as expected - but the attribution to upstream is now
    true instead of false.
    """
    b = np.asarray(mouth_xyxy, dtype=np.float32).reshape(4)
    top, bottom, left, right = margin_px
    return np.array(
        [b[0] - left, b[1] - top, b[2] + right, b[3] + bottom],
        dtype=np.float32,
    )


class MouthStateEstimator(ModelWrapper):
    """Mouth openness, VSDLM first and MAR second."""

    capability = "mouthstate"

    def __init__(
        self,
        engine=None,
        cfg=None,
        *,
        geometry_fallback: bool = True,
        name: str = "mouthstate",
    ) -> None:
        super().__init__(engine=engine, cfg=cfg, name=name)
        self.geometry_fallback = bool(geometry_fallback)
        self.prob_open_threshold = float(
            self.cfg.get("prob_open_threshold", DEFAULT_PROB_OPEN_THRESHOLD)
        )
        self.mar_open = float(self.cfg.get("mar_open", DEFAULT_MAR_OPEN))
        self.min_mouth_box_px = float(self.cfg.get("min_mouth_box_px", MIN_MOUTH_BOX_PX))
        self.in_w, self.in_h = 48, 30
        if engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(engine)
            if len(shape) == 4:
                self.in_h, self.in_w = int(shape[2]), int(shape[3])

    def infer(
        self,
        bgr: np.ndarray,
        landmarks106: Optional[np.ndarray],
        *,
        mouth_reliable: bool = True,
    ) -> MouthState:
        if landmarks106 is None or bgr is None or not mouth_reliable:
            return MouthState()
        mar, mar_valid = mouth_aspect_ratio_detail(landmarks106)

        raw_box = mouth_box(landmarks106)
        if self.available and float(raw_box[2] - raw_box[0]) >= self.min_mouth_box_px:
            box = vsdlm_crop_box(raw_box, self.in_w, self.in_h)
            patch, inside = crop_patch(bgr, box, self.in_w, self.in_h)
            if patch is not None and inside >= MIN_CROP_INSIDE:
                blob = to_blob(patch, rgb=True, scale=1.0 / 255.0)
                out = self.run_engine({self._in_names[0]: blob})
                if out is not None:
                    probs = np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(-1)
                    if probs.size >= 1 and np.isfinite(probs[0]):
                        p = float(min(1.0, max(0.0, probs[0])))
                        return MouthState(
                            prob_open=p,
                            mar=float(mar),
                            open=p >= self.prob_open_threshold,
                            source=SignalSource.MODEL,
                        )
                    self.note_failure(ValueError("bad prob_open %s" % (probs.shape,)))

        if self.geometry_fallback and mar_valid:
            return MouthState(
                prob_open=0.0,
                mar=float(mar),
                open=mar >= self.mar_open,
                source=SignalSource.GEOMETRY,
            )
        return MouthState()
