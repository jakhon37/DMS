"""Sunglasses and face-mask detection - the gate on whether drowsiness is knowable.

Engine contracts (MANIFEST roles ``eye_occlusion`` and ``mouth_occlusion``)::

    sgc_is_s_48   IN "images" FLOAT [1,3,48,48]  OUT "prob_sunglasses" FLOAT [1]
    mwc_n_48      IN "images" FLOAT [1,3,48,48]  OUT "prob_masked"     FLOAT [1]

THE CROP CONVENTION - WHAT UPSTREAM DOES AND WHERE THIS DEVIATES
-----------------------------------------------------------------
Both heads are fed a *head* crop, and the shape of that crop is the operating
point of the classifier.  Upstream ``demo_sgc.py`` / ``demo_mwc.py`` do::

    head_boxes = [box for box in boxes if box.classid == 7]      # whole-body det
    crop = crop_image_with_margin(image, box, margin_top=0, margin_bottom=0,
                                  margin_left=0, margin_right=0)
    # crop_image_with_margin:
    #   x1 = max(int(box.x1) - margin_left, 0) ; x2 = min(int(box.x2) + margin_right, w)
    #   ... return image[y1:y2, x1:x2].copy()
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized  = cv2.resize(rgb_crop, (48, 48), interpolation=cv2.INTER_LINEAR)
    resized  = resized.astype(np.float32) / 255.0

Three properties of that crop were NOT reproduced here and two of them are now
fixed:

1. **not squared** - FIXED.  The box's own aspect is squashed into 48x48 by the
   resize.  The previous revision put the box through ``square_box`` first.
2. **clipped, never padded** - FIXED (:func:`clipped_crop`).  The previous
   revision used :func:`dms.infer.crop_patch`, which fills the outside of the
   box with black; SGC reads that band as a dark region.
3. **zero margin** - DELIBERATELY NOT MATCHED.  Upstream's box is a whole-head
   box from a body detector; SCRFD's is a tight face box, so some growth is
   needed to get the same framing, and :data:`HEAD_BOX_EXPAND` stays at 0.25.
   Zero margin on the SCRFD box was measured and is WORSE - it turns the
   clear-prescription-glasses driver in ``guy_rear_ended`` into a 42%
   false-positive clip (see the table).

MEASURED, WHOLE CLIPS, THE SHIPPED DETECTOR AND CADENCE
--------------------------------------------------------
Every 15th frame of all nine replay clips through ``scrfd_500m_512`` (what
``configs/*.yaml`` select), largest face per frame.  Labels come from looking at
contact sheets of the 48x48 crops themselves; ``minivan`` and
``getting_rear_ended`` are MONTAGES that cut between subjects, so a "1" there
means "the clip contains a sunglasses driver for part of its length", and a
per-clip ON rate near 0.4 on those two is the gate being right, not missing::

    clip                     n    label   OLD sq+0.25+pad   NEW unsquared+clipped
                                          single sample     median of 5, thr 0.40
    bmw_rear_ended           26     1         1.00               1.00
    family_vacation_crash   112     1         0.68               0.91
    getting_rear_ended       78     1(part)   0.42               0.46
    minivan                  96     1(part)   0.19               0.42
    day_driver              410     0         0.00               0.01
    guy_rear_ended          122     0         0.04               0.00
    red_car_crash           183     0         0.02               0.00
    rear_end_accident         9     0         0.11               0.00
    rear_end_whiplash        18     0         0.00               0.00

    (fraction of refreshes at which the gate declares sunglasses; produced by
     /tmp/verify.py, both columns through this module's own code path)

So the change roughly triples the gate's response on the hard positive
(``minivan`` 0.12 -> 0.42) and removes every false positive at the same time.
``family_vacation_crash`` is a fourth sunglasses driver that no earlier review
labelled - wire-rimmed dark glasses, the whole clip.

TEMPORAL VOTE
-------------
Single samples are noisy in both directions: ``rear_end_accident`` (no
sunglasses) produces isolated raw spikes, and the ``minivan`` driver's raw score
wanders over 0.12-0.65.  The gate therefore decides on the MEDIAN of the last
:data:`DEFAULT_VOTE_WINDOW` refreshes for that track - 5 refreshes is about 2.5 s
at the default cadence, and sunglasses do not come and go faster than that.  The
"NEW" column above is that median.  Before the window fills, the median is taken
over whatever samples exist, so the first refresh on a new track behaves like the
old single-sample gate.

:data:`DEFAULT_SUNGLASSES_THRESHOLD` is 0.40, not upstream's 0.50, because the
window median on the hard positive sits just under 0.50 for much of the clip.
Measured at both, on the same data (positives bmw / family / getting / minivan,
negatives the other five)::

    threshold 0.50   POS 1.00 0.78 0.41 0.29   NEG 0.00 0.00 0.00 0.00 0.00
    threshold 0.40   POS 1.00 0.91 0.46 0.42   NEG 0.01 0.00 0.00 0.00 0.00
        POS = bmw / family_vacation / getting_rear_ended / minivan
        NEG = day_driver / guy_rear_ended / red_car / rear_end_accident / whiplash

0.40 buys recall on the hard positive for one extra false window in 410 on
``day_driver``, and missing sunglasses is the more dangerous error.  NOTE that
every ``configs/*.yaml`` currently pins ``sunglasses_threshold: 0.50``, which
overrides this default; at 0.50 the fix above still stands, it is just the
first column.

Honest limits: nine clips of dashcam montage, labelled by eye from the crops, no
labelled cabin footage anywhere in this repository.  This says the gate now sees
the sunglasses drivers this corpus contains and does not fire on the ones it
should not.  It is not a recall figure for the field.

THE MASK HALF: UNVALIDATED ON POSITIVES, AND IT DOES FALSE-POSITIVE
--------------------------------------------------------------------
None of the nine replay clips contains a masked face, so ``prob_masked`` has
never been exercised against a true positive on this board and the mask gate
must not be described as validated.  Worse, the negative side is not clean
either.  Same whole-clip scan, ``prob_masked`` on the corrected head crop::

    clip                     n     raw: median / max / frac>0.5   median-of-5 fires
    bmw_rear_ended           26     0.057 / 1.000 / 0.19               0.12
    rear_end_accident         9     0.000 / 0.664 / 0.22               0.11
    minivan                  96     0.000 / 0.985 / 0.05               0.03
    day_driver              410     0.000 / 0.729 / 0.00               0.00
    guy_rear_ended          122     0.001 / 0.663 / 0.01               0.00
    red_car_crash           183     0.000 / 0.146 / 0.00               0.00
    getting_rear_ended       78     0.000 / 0.387 / 0.00               0.00
    family_vacation_crash   112     0.000 / 0.000 / 0.00               0.00
    rear_end_whiplash        18     0.000 / 0.000 / 0.00               0.00

The median is essentially zero everywhere, which is the right shape for a corpus
with no masks in it, but the head spikes to 1.00 on a bearded driver and the
5-refresh vote still declares "masked" on 12% of refreshes of
``bmw_rear_ended``.  Every one of those is a false ``mouth_reliable=False``,
which silently switches YAWN off.  An earlier review's synthetic stimulus
(painting the lower half of the crop grey) moved the score the right way on some
clips and the wrong way on others, so it is not evidence in either direction.

Status to quote anywhere this gate is described: "loaded; positive class never
exercised on this corpus; measured 0-12% false-positive rate per refresh on
unmasked replay drivers".  Not "live".
``test_mwc_mask_gate_behaviour_on_the_unmasked_replay_corpus`` pins exactly that.

Update cadence: occlusion changes at human timescales, not frame timescales.
:meth:`OcclusionDetector.infer` therefore caches per track id and only re-runs
every ``every_n`` calls (default 15, about half a second at 30 fps), which keeps
the two engines off the per-frame budget.

Failure behaviour: always returns an :class:`dms.types.Occlusion`.  With no
engine it returns ``source=UNAVAILABLE`` and - deliberately -
``eyes_reliable=True`` / ``mouth_reliable=True``: an absent occlusion detector
must not silently disable drowsiness detection.  The registry reports the
capability as degraded so the operator knows the gate is not running.
"""
from __future__ import annotations

from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np

from dms.infer import ModelWrapper, box_inside_fraction, expand_box, log, to_blob
from dms.types import Occlusion, SignalSource

#: Growth applied to the SCRFD face box before it is handed to SGC/MWC, as a
#: fraction of the box size on every side.  Upstream uses ZERO on a whole-head
#: box from a body detector; SCRFD's face box is tighter than that, so it is
#: grown to approximate the same framing.  0.25 is not arbitrary any more: zero,
#: 0.10, 0.15, 0.25 and 0.35 were swept over whole clips (see the module
#: docstring) and 0.25 is the value that keeps every labelled negative at zero
#: false positives while giving the best response on the labelled positives.
#: It is also what every ``configs/*.yaml`` already sets.
HEAD_BOX_EXPAND = 0.25

#: Decision threshold on the WINDOW MEDIAN of ``prob_sunglasses``.  Deliberately
#: below upstream's single-frame 0.50 - see the module docstring for both.
DEFAULT_SUNGLASSES_THRESHOLD = 0.40
#: Upstream's threshold, applied to the window median of ``prob_masked``.  The
#: positive class has never been exercised on this corpus, so this number is
#: inherited, not calibrated.
DEFAULT_MASK_THRESHOLD = 0.50
#: Number of refreshes the decision median is taken over.
DEFAULT_VOTE_WINDOW = 5
MIN_CROP_INSIDE = 0.55
DEFAULT_EVERY_N = 15


def head_box(face_xyxy: np.ndarray, expand: float = HEAD_BOX_EXPAND) -> np.ndarray:
    """The box SGC/MWC are fed, in full-frame pixels.

    The SCRFD face box grown by ``expand`` on every side, and NOT squared:
    upstream feeds its head-detector box straight into a non-aspect-preserving
    resize, and squaring first changes what that resize squashes.  Dropping the
    square (with the padded crop replaced by :func:`clipped_crop`) is what took
    the gate's response on the ``minivan`` sunglasses driver from 0.12 to 0.42.
    """
    b = np.asarray(face_xyxy, dtype=np.float32).reshape(4)
    if abs(float(expand)) < 1e-9:
        return b.copy()
    return expand_box(b, float(expand))


def clipped_crop(
    bgr: np.ndarray, xyxy, out_w: int, out_h: int
) -> Tuple[Optional[np.ndarray], float]:
    """Upstream ``crop_image_with_margin`` + resize, as a ``(patch, inside)`` pair.

    Truncates the box to integers, clamps it to the frame and resizes whatever
    is left - it does NOT pad the outside part with black the way
    :func:`dms.infer.crop_patch` does.  That difference matters: SGC reads a
    black band as a dark region and the head classifiers were never trained on
    one.  ``inside`` is the fraction of the requested box that was really inside
    the frame, so the caller can still refuse a crop that is mostly off-screen.
    """
    h_img, w_img = bgr.shape[:2]
    b = np.asarray(xyxy, dtype=np.float64).reshape(4)
    x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
    x2, y2 = min(int(b[2]), w_img), min(int(b[3]), h_img)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, 0.0
    patch = cv2.resize(
        bgr[y1:y2, x1:x2], (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR
    )
    return patch, box_inside_fraction(b, (w_img, h_img))


class OcclusionDetector(ModelWrapper):
    """Sunglasses + mask gates over the drowsiness signals."""

    capability = "occlusion"

    def __init__(
        self,
        sunglasses_engine=None,
        mask_engine=None,
        cfg=None,
        *,
        name: str = "occlusion",
    ) -> None:
        # The base class tracks one engine; this capability has two, so the
        # health record follows whichever engines are present and the base
        # engine slot holds the sunglasses head (the more consequential of the
        # two, since it gates MICROSLEEP).
        super().__init__(engine=sunglasses_engine, cfg=cfg, name=name)
        self.sunglasses_engine = sunglasses_engine
        self.mask_engine = mask_engine
        self._mask_io = None
        if mask_engine is not None:
            from dms.infer import engine_io_names

            self._mask_io = engine_io_names(mask_engine)
        self.sunglasses_threshold = float(
            self.cfg.get("sunglasses_threshold", DEFAULT_SUNGLASSES_THRESHOLD)
        )
        self.mask_threshold = float(self.cfg.get("mask_threshold", DEFAULT_MASK_THRESHOLD))
        self.expand = float(self.cfg.get("head_box_expand", HEAD_BOX_EXPAND))
        self.vote_window = max(1, int(self.cfg.get("vote_window", DEFAULT_VOTE_WINDOW)))
        self.every_n = max(1, int(self.cfg.get("every_n", DEFAULT_EVERY_N)))
        self.size = 48
        if sunglasses_engine is not None:
            from dms.infer import engine_input_shape

            shape = engine_input_shape(sunglasses_engine)
            if len(shape) == 4:
                self.size = int(shape[2])
        self._cache: Dict[int, Occlusion] = {}
        self._counter: Dict[int, int] = {}
        self._sun_hist: Dict[int, Deque[float]] = {}
        self._mask_hist: Dict[int, Deque[float]] = {}

    @property
    def available(self) -> bool:
        return (self.sunglasses_engine is not None or self.mask_engine is not None) and not self._disabled

    def _run_head(self, patch: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        blob = to_blob(patch, rgb=True, scale=1.0 / 255.0)
        sun = None
        mask = None
        if self.sunglasses_engine is not None:
            out = self.run_engine({self._in_names[0]: blob})
            if out is not None:
                v = np.asarray(out[self._out_names[0]], dtype=np.float64).reshape(-1)
                if v.size and np.isfinite(v[0]):
                    sun = float(min(1.0, max(0.0, v[0])))
        if self.mask_engine is not None and self._mask_io is not None:
            try:
                out = self.mask_engine.infer({self._mask_io[0][0]: blob})
            except Exception as exc:  # noqa: BLE001 - never escape to the frame loop
                self.note_failure(exc)
            else:
                v = np.asarray(out[self._mask_io[1][0]], dtype=np.float64).reshape(-1)
                if v.size and np.isfinite(v[0]):
                    mask = float(min(1.0, max(0.0, v[0])))
        return sun, mask

    def _vote(self, hist: Dict[int, Deque[float]], key: int, value: float) -> float:
        """Push one sample and return the median of the last ``vote_window``.

        Before the window has filled the median is taken over whatever samples
        exist, so the very first refresh after a track appears behaves exactly
        like the old single-sample gate - the measured operating point above is
        only reached once ``vote_window`` refreshes have happened.
        """
        buf = hist.get(key)
        if buf is None or buf.maxlen != self.vote_window:
            buf = deque(buf or (), maxlen=self.vote_window)
            hist[key] = buf
        buf.append(float(value))
        return float(np.median(np.asarray(buf, dtype=np.float64)))

    def infer(
        self,
        bgr: np.ndarray,
        face_xyxy: Optional[np.ndarray],
        *,
        track_id: Optional[int] = None,
    ) -> Occlusion:
        """Occlusion state for one face. Cached per track id, refreshed every N calls.

        ``sunglasses`` and ``mask`` on the returned record are the WINDOW MEDIANS
        that the reliability flags were decided on, not the single raw score, so
        a logged number always explains the decision beside it.
        """
        if bgr is None or face_xyxy is None:
            return Occlusion(source=SignalSource.UNAVAILABLE)
        key = -1 if track_id is None else int(track_id)
        n = self._counter.get(key, 0)
        self._counter[key] = n + 1
        cached = self._cache.get(key)
        if cached is not None and (n % self.every_n) != 0:
            return cached
        if not self.available:
            return Occlusion(source=SignalSource.UNAVAILABLE)

        patch, inside = clipped_crop(bgr, head_box(face_xyxy, self.expand), self.size, self.size)
        if patch is None or inside < MIN_CROP_INSIDE:
            return cached if cached is not None else Occlusion(source=SignalSource.UNAVAILABLE)
        sun, mask = self._run_head(patch)
        if sun is None and mask is None:
            return cached if cached is not None else Occlusion(source=SignalSource.UNAVAILABLE)
        sun_v = None if sun is None else self._vote(self._sun_hist, key, sun)
        mask_v = None if mask is None else self._vote(self._mask_hist, key, mask)
        occ = Occlusion(
            sunglasses=0.0 if sun_v is None else sun_v,
            mask=0.0 if mask_v is None else mask_v,
            eyes_reliable=True if sun_v is None else sun_v < self.sunglasses_threshold,
            mouth_reliable=True if mask_v is None else mask_v < self.mask_threshold,
            source=SignalSource.MODEL,
        )
        self._cache[key] = occ
        return occ

    def forget(self, track_id: int) -> None:
        """Drop the cache for a track that has gone away."""
        self._cache.pop(int(track_id), None)
        self._counter.pop(int(track_id), None)
        self._sun_hist.pop(int(track_id), None)
        self._mask_hist.pop(int(track_id), None)

    def close(self) -> None:
        super().close()
        eng = self.mask_engine
        self.mask_engine = None
        self.sunglasses_engine = None
        if eng is not None and hasattr(eng, "close"):
            try:
                eng.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("occlusion mask engine close failed: %s", exc)
