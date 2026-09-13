"""Model-layer wrappers: shapes, ranges, degradation, and the registry.

Everything here runs WITHOUT a GPU.  Engines are replaced by fakes that assert
the exact blob shape and dtype the real engine binds and return synthetic
outputs with the documented semantics.  The two on-board tests at the end load
the real engines and are skipped when the engine files are absent, so this file
is CI-safe off the Jetson.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dms.geometry.face106 import (
    CHIN,
    INNER_LIP,
    LEFT_EYE_CENTRE,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE_TIP,
    OUTER_LIP,
    RIGHT_EYE_CENTRE,
)
from dms.infer.eyestate import EyeStateEstimator
from dms.infer.gaze import GazeEstimator, decode_gaze_bins, fuse_absolute_gaze
from dms.infer.headpose import HeadPoseEstimator, yaw_cue_from_landmarks
from dms.infer.identity import IdentityEmbedder, cosine, l2_normalise
from dms.infer.mouthstate import MouthStateEstimator, vsdlm_crop_box
from dms.infer.objects import (
    COCO_CELL_PHONE,
    ObjectDetector,
    decode_yolox,
    letterbox_topleft,
    yolox_grids,
)
from dms.infer.occlusion import OcclusionDetector, head_box
from dms.infer.registry import CAPABILITIES, ModelRegistry
from dms.types import GazeVector, HeadPose, SignalSource

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE_DIR = os.path.join(ROOT, "engines")


# ------------------------------------------------------------------ fakes ---
class FakeEngine:
    """Asserts the bound shape/dtype, then returns canned outputs."""

    def __init__(self, in_name, in_shape, outputs, *, fail=False):
        self.input_name = in_name
        self.input_shape = tuple(in_shape)
        self._outputs = dict(outputs)
        self.output_names = list(outputs)
        self.fail = fail
        self.calls = 0
        self.last_blob = None

    def infer(self, feed):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated engine failure")
        assert list(feed) == [self.input_name], "wrapper fed %s" % list(feed)
        blob = feed[self.input_name]
        assert blob.dtype == np.float32, blob.dtype
        assert blob.shape == self.input_shape, "%s vs %s" % (blob.shape, self.input_shape)
        assert blob.flags["C_CONTIGUOUS"]
        self.last_blob = blob
        return {k: np.array(v, copy=True) for k, v in self._outputs.items()}

    def close(self):
        self._outputs = {}


def _frame(w=1280, h=720, value=120):
    rng = np.random.RandomState(7)
    img = rng.randint(value - 20, value + 20, size=(h, w, 3)).astype(np.uint8)
    return img


def _face_landmarks(cx=640.0, cy=360.0, iod=60.0, eye_half_h=6.0):
    """A plausible 106-point set at the verified indices."""
    lm = np.zeros((106, 2), dtype=np.float32)
    for base, ex in ((33, cx - iod * 0.5), (87, cx + iod * 0.5)):
        half_w = iod * 0.3
        lm[base + 0] = [ex, cy + eye_half_h]
        lm[base + 1] = [ex, cy]
        lm[base + 5] = [ex, cy]
        lm[base + 2] = [ex - half_w, cy]
        lm[base + 6] = [ex + half_w, cy]
        lm[base + 3] = [ex - 0.5 * half_w, cy + 0.8 * eye_half_h]
        lm[base + 4] = [ex + 0.5 * half_w, cy + 0.8 * eye_half_h]
        lm[base + 7] = [ex, cy - eye_half_h]
        lm[base + 8] = [ex - 0.5 * half_w, cy - 0.8 * eye_half_h]
        lm[base + 9] = [ex + 0.5 * half_w, cy - 0.8 * eye_half_h]
    lm[NOSE_TIP] = [cx, cy + iod * 0.5]
    lm[CHIN] = [cx, cy + iod * 1.6]
    for i, idx in enumerate(OUTER_LIP):
        a = 2.0 * np.pi * i / len(OUTER_LIP)
        lm[idx] = [cx + iod * 0.5 * np.cos(a), cy + iod * 1.0 + iod * 0.16 * np.sin(a)]
    for i, idx in enumerate(INNER_LIP):
        a = 2.0 * np.pi * i / len(INNER_LIP)
        lm[idx] = [cx + iod * 0.36 * np.cos(a), cy + iod * 1.0 + iod * 0.02 * np.sin(a)]
    lm[MOUTH_LEFT] = [cx - iod * 0.5, cy + iod * 1.0]
    lm[MOUTH_RIGHT] = [cx + iod * 0.5, cy + iod * 1.0]
    # the brow / contour / nose blocks are unused by these tests but must be
    # finite, so park them on the face centre rather than at the origin
    for i in range(106):
        if not lm[i].any():
            lm[i] = [cx, cy]
    return lm


def _face_box(cx=640.0, cy=360.0, iod=60.0):
    return np.array([cx - iod, cy - iod, cx + iod, cy + iod * 1.9], dtype=np.float32)


# --------------------------------------------------------------- headpose ---
def _rot(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    rx, ry, rz = np.radians([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32).reshape(1, 3, 3)


def test_headpose_feeds_the_documented_blob_and_decodes_the_matrix():
    """Decode check: raw Euler out of R, then the measured sign map applied."""
    from dms.infer.headpose import MODEL_PITCH_SIGN, MODEL_ROLL_SIGN, MODEL_YAW_SIGN

    eng = FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot(20.0, -10.0, 5.0)})
    hp = HeadPoseEstimator(engine=eng)
    pose = hp.infer(_frame(), _face_box(), _face_landmarks())
    assert eng.calls == 1
    assert pose.source is SignalSource.MODEL
    assert pose.yaw == pytest.approx(MODEL_YAW_SIGN * 20.0, abs=0.5)
    assert pose.pitch == pytest.approx(MODEL_PITCH_SIGN * -10.0, abs=0.5)
    assert pose.roll == pytest.approx(MODEL_ROLL_SIGN * 5.0, abs=0.5)
    assert 0.0 < pose.confidence <= 1.0
    # imagenet-normalised RGB stays in a sane range
    assert -3.0 < float(eng.last_blob.min()) and float(eng.last_blob.max()) < 3.0


def test_headpose_falls_back_to_pnp_when_the_engine_is_absent():
    hp = HeadPoseEstimator(engine=None)
    assert not hp.available
    pose = hp.infer(_frame(), _face_box(), _face_landmarks())
    assert pose.source in (SignalSource.GEOMETRY, SignalSource.UNAVAILABLE)
    assert pose.source is not SignalSource.MODEL


def test_headpose_falls_back_when_the_engine_raises():
    eng = FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot()}, fail=True)
    hp = HeadPoseEstimator(engine=eng)
    pose = hp.infer(_frame(), _face_box(), _face_landmarks())
    assert pose.source is not SignalSource.MODEL
    assert hp.health.consecutive_failures == 1


def test_headpose_disables_a_persistently_failing_engine():
    eng = FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot()}, fail=True)
    hp = HeadPoseEstimator(engine=eng)
    for _ in range(hp.max_consecutive_failures + 3):
        hp.infer(_frame(), _face_box(), _face_landmarks())
    assert not hp.available
    assert not hp.health.loaded
    assert eng.calls == hp.max_consecutive_failures


def test_headpose_returns_unavailable_when_nothing_can_answer():
    hp = HeadPoseEstimator(engine=None, geometry_fallback=False)
    pose = hp.infer(_frame(), _face_box(), _face_landmarks())
    assert pose.source is SignalSource.UNAVAILABLE
    assert pose.confidence == 0.0


def test_headpose_never_raises_on_junk_input():
    hp = HeadPoseEstimator(engine=None)
    assert hp.infer(None, None).source is SignalSource.UNAVAILABLE
    assert hp.infer(_frame(), None).source is SignalSource.UNAVAILABLE


SIGN_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "headpose_signs.npz")


def test_headpose_signs_match_the_captured_calibration():
    """Pin MODEL_YAW/PITCH/ROLL_SIGN against references that never see the net.

    The fixture holds the network's RAW Euler output for 312 faces sampled
    across the replay clips, alongside three independent references: the
    pure-landmark nose cue, the landmark eye-axis roll and the geometric solve.
    Applying the sign constants must turn every one of those into a POSITIVE
    correlation.  Flipping any constant flips a correlation and fails here,
    which is the only cheap guard against a silently mirrored head-pose signal.
    """
    from dms.infer.headpose import MODEL_PITCH_SIGN, MODEL_ROLL_SIGN, MODEL_YAW_SIGN

    if not os.path.isfile(SIGN_FIXTURE):
        pytest.skip("head-pose sign fixture not captured on this machine")
    d = np.load(SIGN_FIXTURE)
    assert d["raw_yaw"].size >= 100

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        assert m.sum() >= 50
        return float(np.corrcoef(x[m], y[m])[0, 1])

    yaw = MODEL_YAW_SIGN * d["raw_yaw"]
    pitch = MODEL_PITCH_SIGN * d["raw_pitch"]
    roll = MODEL_ROLL_SIGN * d["raw_roll"]
    # drop the near-gimbal outliers the runtime roll cross-check also rejects
    ok = (np.abs(roll) < 60.0) & (np.abs(pitch) < 60.0) & (np.abs(yaw) < 70.0)
    assert ok.mean() > 0.85, "too many outliers: %.2f" % ok.mean()

    assert corr(yaw[ok], d["yaw_cue"][ok]) > 0.80, "model yaw is mirrored w.r.t. the nose cue"
    assert corr(roll[ok], d["lmk_roll"][ok]) > 0.60, "model roll is mirrored w.r.t. the eye axis"
    assert corr(yaw[ok], d["pnp_yaw"][ok]) > 0.70, "model and geometric yaw disagree in sign"
    assert corr(pitch[ok], d["pnp_pitch"][ok]) > 0.35, "model and geometric pitch disagree in sign"


def test_yaw_cue_sign():
    """Positive cue = nose toward image-right = driver turned to their own left."""
    lm = _face_landmarks()
    assert yaw_cue_from_landmarks(lm) == pytest.approx(0.0, abs=1e-6)
    lm[NOSE_TIP] = lm[NOSE_TIP] + np.array([15.0, 0.0], dtype=np.float32)
    assert yaw_cue_from_landmarks(lm) > 0.0
    lm[NOSE_TIP] = lm[NOSE_TIP] - np.array([40.0, 0.0], dtype=np.float32)
    assert yaw_cue_from_landmarks(lm) < 0.0
    assert yaw_cue_from_landmarks(np.zeros((106, 2), dtype=np.float32)) is None


# --------------------------------------------------------------- eyestate ---
def test_eyestate_batches_both_eyes_into_one_launch():
    eng = FakeEngine("images", (2, 3, 24, 40), {"prob_open": np.array([0.92, 0.04], np.float32)})
    es = EyeStateEstimator(engine=eng)
    left, right = es.infer(_frame(), _face_landmarks())
    assert eng.calls == 1, "both eyes must go in a single inference"
    assert left.source is SignalSource.MODEL and right.source is SignalSource.MODEL
    assert left.prob_open == pytest.approx(0.92) and not left.closed
    assert right.prob_open == pytest.approx(0.04) and right.closed
    assert 0.0 <= float(eng.last_blob.min()) and float(eng.last_blob.max()) <= 1.0


def test_eyestate_uses_the_batch1_engine_twice_when_only_it_exists():
    eng = FakeEngine("images", (1, 3, 24, 40), {"prob_open": np.array([0.9], np.float32)})
    es = EyeStateEstimator(engine=None, single_engine=eng)
    assert es.available and es.batch == 1
    left, right = es.infer(_frame(), _face_landmarks())
    assert eng.calls == 2, "one launch per eye"
    assert left.source is SignalSource.MODEL and right.source is SignalSource.MODEL
    assert left.prob_open == pytest.approx(0.9)


def test_eyestate_falls_back_to_ear():
    es = EyeStateEstimator(engine=None)
    open_l, open_r = es.infer(_frame(), _face_landmarks(eye_half_h=8.0))
    assert open_l.source is SignalSource.GEOMETRY and not open_l.closed
    shut_l, shut_r = es.infer(_frame(), _face_landmarks(eye_half_h=0.3))
    assert shut_l.source is SignalSource.GEOMETRY and shut_l.closed
    assert shut_r.closed


def test_eyestate_is_unavailable_behind_sunglasses():
    eng = FakeEngine("images", (2, 3, 24, 40), {"prob_open": np.array([0.9, 0.9], np.float32)})
    es = EyeStateEstimator(engine=eng)
    left, right = es.infer(_frame(), _face_landmarks(), eyes_reliable=False)
    assert left.source is SignalSource.UNAVAILABLE and right.source is SignalSource.UNAVAILABLE
    assert not left.closed and not right.closed  # no evidence, not "eyes open"
    assert eng.calls == 0


def test_eyestate_refuses_the_model_on_eyes_too_small_to_resolve():
    """Measured: below ~20 px of eye the classifier inverts. Fall back, do not guess."""
    eng = FakeEngine("images", (2, 3, 24, 40), {"prob_open": np.array([0.01, 0.01], np.float32)})
    es = EyeStateEstimator(engine=eng)
    tiny = _face_landmarks(iod=18.0, eye_half_h=2.0)  # eye box ~11 px wide
    left, right = es.infer(_frame(), tiny)
    assert eng.calls == 0, "the model must not run below the resolution floor"
    assert left.source is SignalSource.GEOMETRY
    assert not left.closed, "a small open eye must not be reported closed"


def test_mouthstate_refuses_the_model_on_a_mouth_too_small_to_resolve():
    eng = FakeEngine("images", (1, 3, 30, 48), {"prob_open": np.array([0.99], np.float32)})
    ms = MouthStateEstimator(engine=eng)
    out = ms.infer(_frame(), _face_landmarks(iod=25.0))  # mouth box ~25 px wide
    assert eng.calls == 0
    assert out.source is SignalSource.GEOMETRY and not out.open


def test_ear_rejects_a_physiologically_impossible_ratio():
    from dms.geometry.ear import EAR_MAX_PLAUSIBLE, face_ear_detail

    lm = _face_landmarks(eye_half_h=40.0)  # taller than it is wide: a bad fit
    mean, left, right, valid = face_ear_detail(lm)
    assert left > EAR_MAX_PLAUSIBLE
    assert not valid, "a collapsed landmark fit must be invalid, not a big EAR"


def test_eyestate_without_landmarks_is_unavailable():
    es = EyeStateEstimator(engine=None)
    left, right = es.infer(_frame(), None)
    assert left.source is SignalSource.UNAVAILABLE and right.source is SignalSource.UNAVAILABLE


def test_eyestate_falls_back_when_the_engine_raises():
    eng = FakeEngine("images", (2, 3, 24, 40), {"prob_open": np.zeros(2, np.float32)}, fail=True)
    es = EyeStateEstimator(engine=eng)
    left, _ = es.infer(_frame(), _face_landmarks())
    assert left.source is SignalSource.GEOMETRY


# -------------------------------------------------------------- mouthstate --
def test_mouthstate_model_path():
    eng = FakeEngine("images", (1, 3, 30, 48), {"prob_open": np.array([0.81], np.float32)})
    ms = MouthStateEstimator(engine=eng)
    out = ms.infer(_frame(), _face_landmarks())
    assert eng.calls == 1
    assert out.source is SignalSource.MODEL
    assert out.prob_open == pytest.approx(0.81) and out.open


def test_mouthstate_falls_back_to_mar():
    ms = MouthStateEstimator(engine=None)
    out = ms.infer(_frame(), _face_landmarks())
    assert out.source is SignalSource.GEOMETRY
    assert not out.open  # the synthetic mouth is closed


def test_mouthstate_is_unavailable_behind_a_mask():
    ms = MouthStateEstimator(engine=None)
    out = ms.infer(_frame(), _face_landmarks(), mouth_reliable=False)
    assert out.source is SignalSource.UNAVAILABLE


def test_vsdlm_margins_are_applied_in_frame_pixels_like_upstream():
    """Upstream demo_vsdlm.py subtracts the margins from the FRAME box.

        x1 = max(int(box.x1) - margin_left, 0)
        y1 = max(int(box.y1) - margin_top,  0)
        x2 = min(int(box.x2) + margin_right,  w)
        y2 = min(int(box.y2) + margin_bottom, h)

    No resize scale is involved, so the margin must not depend on the box size.
    The previous revision multiplied by ``w / out_w``, which on this 96 x 60 box
    turned the 6 px bottom margin into 12 px, and claimed upstream fidelity for
    it.
    """
    box = np.array([100.0, 200.0, 196.0, 260.0], dtype=np.float32)  # 96 x 60
    grown = vsdlm_crop_box(box, 48, 30)
    assert grown[0] == pytest.approx(100.0 - 2.0)
    assert grown[1] == pytest.approx(200.0 - 2.0)
    assert grown[2] == pytest.approx(196.0 + 2.0)
    assert grown[3] == pytest.approx(260.0 + 6.0)
    assert grown[3] - box[3] > box[1] - grown[1], "bottom margin must exceed the top margin"
    # ... and it stays 6 px however big the mouth is.
    big = np.array([0.0, 0.0, 400.0, 250.0], dtype=np.float32)
    assert vsdlm_crop_box(big, 48, 30)[3] == pytest.approx(250.0 + 6.0)


# --------------------------------------------------------------- occlusion --
def test_head_box_is_grown_but_never_squared():
    """Squaring is the half of the old crop that cost the gate its recall.

    Upstream resizes its head box straight to 48x48 without preserving aspect,
    so squaring first changes what the resize squashes.  Measured over whole
    replay clips at the shipped detector and cadence, dropping the square (and
    the black padding, see :func:`clipped_crop`) took the gate's response on the
    ``minivan`` sunglasses driver from 0.12 to 0.42 of refreshes while removing
    every false positive on the five labelled negative clips.  The 0.25 growth
    is kept: SCRFD's face box is tighter than a head box, and zero growth makes
    ``guy_rear_ended`` (clear prescription glasses) a 42% false-positive clip.
    """
    face = np.array([100.0, 100.0, 200.0, 230.0], dtype=np.float32)
    hb = head_box(face)
    assert hb[0] < face[0] and hb[2] > face[2], "the face box must be grown"
    assert (hb[2] - hb[0]) != pytest.approx(hb[3] - hb[1]), "must NOT be squared"
    # the aspect of the source box must survive the growth
    assert (hb[2] - hb[0]) / (hb[3] - hb[1]) == pytest.approx(
        (face[2] - face[0]) / (face[3] - face[1])
    )
    assert list(head_box(face, 0.0)) == pytest.approx(list(face))


def test_clipped_crop_drops_the_outside_instead_of_padding_it_black():
    """SGC reads a black pad band as a dark region; upstream never pads.

    ``crop_patch`` fills the outside with zeros, which is right for the eye and
    head-pose crops that need the geometry preserved and wrong for a classifier
    trained on ``image[y1:y2, x1:x2]``.
    """
    from dms.infer import crop_patch
    from dms.infer.occlusion import clipped_crop

    frame = np.full((240, 320, 3), 200, np.uint8)
    box = np.array([-40.0, -40.0, 60.0, 60.0], dtype=np.float32)  # 40% off-frame
    padded, _ = crop_patch(frame, box, 48, 48)
    clipped, inside = clipped_crop(frame, box, 48, 48)
    assert padded.min() == 0, "crop_patch is expected to pad with black"
    assert clipped.min() == 200, "clipped_crop must contain frame pixels only"
    assert clipped.shape == (48, 48, 3)
    assert inside == pytest.approx(0.36, abs=0.02)
    # a box wholly outside the frame has nothing to resize
    assert clipped_crop(frame, np.array([-90.0, -90.0, -10.0, -10.0]), 48, 48)[0] is None


def test_occlusion_decides_on_the_window_median_not_a_single_sample():
    """One spike must not flip the gate; a sustained majority must.

    ``rear_end_accident`` (no sunglasses) produces two consecutive raw samples of
    0.77 and 0.73 with the upstream crop, and ``minivan`` (sunglasses) only
    scores above 0.50 on 7 of 12 samples, so the decision statistic is the median
    of the last ``vote_window`` refreshes rather than the latest one.
    """
    class Seq:
        input_name = "images"
        input_shape = (1, 3, 48, 48)
        output_names = ["prob_sunglasses"]

        def __init__(self, values):
            self.values = list(values)
            self.i = 0

        def infer(self, feed):
            v = self.values[min(self.i, len(self.values) - 1)]
            self.i += 1
            return {"prob_sunglasses": np.array([v], np.float32)}

    frame, face = _frame(), _face_box()
    # two-sample spike inside an otherwise clean track
    eng = Seq([0.02, 0.03, 0.77, 0.73, 0.03, 0.02])
    od = OcclusionDetector(sunglasses_engine=eng, cfg={"every_n": 1, "vote_window": 5})
    flags = [od.infer(frame, face, track_id=1).eyes_reliable for _ in range(6)]
    assert all(flags), "a two-sample spike must not disable the eye channel: %s" % flags

    # a real, sustained positive at the level minivan actually reaches
    eng2 = Seq([0.49, 0.61, 0.16, 0.42, 0.72, 0.52])
    od2 = OcclusionDetector(sunglasses_engine=eng2, cfg={"every_n": 1, "vote_window": 5})
    flags2 = [od2.infer(frame, face, track_id=1).eyes_reliable for _ in range(6)]
    assert not flags2[-1], "a sustained positive must disable the eye channel"
    assert od2.infer(frame, face, track_id=1).sunglasses >= 0.45

    # the reported number is the statistic the decision was made on
    eng3 = Seq([0.9, 0.1, 0.1])
    od3 = OcclusionDetector(sunglasses_engine=eng3, cfg={"every_n": 1, "vote_window": 5})
    for _ in range(3):
        occ = od3.infer(frame, face, track_id=7)
    assert occ.sunglasses == pytest.approx(0.1), "median of (0.9, 0.1, 0.1)"
    assert occ.eyes_reliable


def test_occlusion_model_path_and_gating():
    sun = FakeEngine("images", (1, 3, 48, 48), {"prob_sunglasses": np.array([0.91], np.float32)})
    msk = FakeEngine("images", (1, 3, 48, 48), {"prob_masked": np.array([0.02], np.float32)})
    od = OcclusionDetector(sunglasses_engine=sun, mask_engine=msk)
    occ = od.infer(_frame(), _face_box(), track_id=1)
    assert occ.source is SignalSource.MODEL
    assert occ.sunglasses == pytest.approx(0.91) and not occ.eyes_reliable
    assert occ.mask == pytest.approx(0.02) and occ.mouth_reliable


def test_occlusion_caches_between_refreshes():
    sun = FakeEngine("images", (1, 3, 48, 48), {"prob_sunglasses": np.array([0.9], np.float32)})
    od = OcclusionDetector(sunglasses_engine=sun, cfg={"every_n": 5})
    for _ in range(5):
        od.infer(_frame(), _face_box(), track_id=3)
    assert sun.calls == 1
    od.infer(_frame(), _face_box(), track_id=3)
    assert sun.calls == 2
    od.forget(3)
    od.infer(_frame(), _face_box(), track_id=3)
    assert sun.calls == 3


def test_occlusion_without_engines_does_not_disable_drowsiness():
    od = OcclusionDetector()
    occ = od.infer(_frame(), _face_box(), track_id=1)
    assert occ.source is SignalSource.UNAVAILABLE
    assert occ.eyes_reliable and occ.mouth_reliable


# -------------------------------------------------------------------- gaze --
def test_gaze_is_refused_without_the_research_licence_flag():
    logits = np.zeros((1, 90), np.float32)
    logits[0, 45] = 20.0
    eng = FakeEngine("input", (1, 3, 448, 448), {"yaw": logits, "pitch": logits})
    gz = GazeEstimator(engine=eng, allow_research_models=False)
    assert not gz.available
    out = gz.infer(_frame(), _face_box())
    assert out.source is SignalSource.UNAVAILABLE
    assert eng.calls == 0, "a licence-gated engine must never be executed"


def test_gaze_runs_when_research_models_are_allowed():
    yaw_logits = np.full((1, 90), -20.0, np.float32)
    yaw_logits[0, 50] = 20.0
    pitch_logits = np.full((1, 90), -20.0, np.float32)
    pitch_logits[0, 45] = 20.0
    eng = FakeEngine("input", (1, 3, 448, 448), {"yaw": yaw_logits, "pitch": pitch_logits})
    gz = GazeEstimator(engine=eng, allow_research_models=True)
    out = gz.infer(_frame(), _face_box())
    assert out.source is SignalSource.MODEL
    assert out.yaw == pytest.approx(50 * 4 - 180, abs=1.0)
    assert out.pitch == pytest.approx(45 * 4 - 180, abs=1.0)


def test_decode_gaze_bins_is_an_expectation_over_a_softmax():
    logits = np.zeros(90, np.float32)
    logits[45] = 50.0
    assert decode_gaze_bins(logits) == pytest.approx(45 * 4 - 180, abs=0.1)
    # a flat distribution decodes to the centre of the range
    assert decode_gaze_bins(np.zeros(90, np.float32)) == pytest.approx(44.5 * 4 - 180, abs=0.1)
    # never NaN, even on a huge-magnitude row
    assert np.isfinite(decode_gaze_bins(np.full(90, 1e4, np.float32)))


def test_fuse_absolute_gaze_adds_head_and_eye():
    pose = HeadPose(yaw=20.0, pitch=-5.0, confidence=0.9, source=SignalSource.MODEL)
    eye = GazeVector(yaw=-18.0, pitch=2.0, confidence=0.8, source=SignalSource.MODEL)
    fused = fuse_absolute_gaze(pose, eye)
    assert fused.yaw == pytest.approx(2.0)
    assert fused.pitch == pytest.approx(-3.0)
    assert fused.source is SignalSource.MODEL
    assert fused.confidence == pytest.approx(0.72)


def test_fuse_passes_through_head_pose_when_gaze_is_absent():
    pose = HeadPose(yaw=20.0, pitch=-5.0, confidence=0.9, source=SignalSource.GEOMETRY)
    fused = fuse_absolute_gaze(pose, GazeVector())
    assert fused.yaw == pytest.approx(20.0)
    assert fused.source is SignalSource.GEOMETRY


def test_fuse_reports_unavailable_when_neither_side_has_anything():
    assert fuse_absolute_gaze(None, GazeVector()).source is SignalSource.UNAVAILABLE
    assert fuse_absolute_gaze(HeadPose(), GazeVector()).source is SignalSource.UNAVAILABLE


def test_fuse_fades_confidence_outside_the_small_angle_regime():
    pose = HeadPose(yaw=45.0, confidence=1.0, source=SignalSource.MODEL)
    eye = GazeVector(yaw=40.0, confidence=1.0, source=SignalSource.MODEL)
    assert fuse_absolute_gaze(pose, eye).confidence < 0.3


# ----------------------------------------------------------------- objects --
def test_yolox_grid_row_count():
    grid, stride = yolox_grids(416, 416)
    assert grid.shape[0] == 52 * 52 + 26 * 26 + 13 * 13 == 3549
    assert stride[0] == 8.0 and stride[-1] == 32.0


def test_letterbox_pads_bottom_right_only():
    img = np.full((360, 640, 3), 200, np.uint8)
    canvas, r = letterbox_topleft(img, 416, 416)
    assert canvas.shape == (416, 416, 3)
    assert r == pytest.approx(416.0 / 640.0)
    assert canvas[0, 0, 0] == 200  # top-left is image
    assert canvas[-1, -1, 0] == 114  # bottom-right is pad


def _yolox_head_with_one_phone(net=416, stride_cell=(20, 20)):
    """Build a (1, 3549, 85) head with a single confident cell-phone anchor."""
    raw = np.zeros((1, 3549, 85), np.float32)
    gx, gy = stride_cell
    idx = gy * 52 + gx  # first (stride 8) block
    raw[0, idx, 0:2] = 0.5
    raw[0, idx, 2:4] = np.log(np.array([4.0, 5.0], np.float32))  # w=32, h=40 at stride 8
    raw[0, idx, 4] = 0.9
    raw[0, idx, 5 + COCO_CELL_PHONE] = 0.8
    return raw


def test_decode_yolox_places_the_box_where_the_grid_says():
    raw = _yolox_head_with_one_phone()
    dets = decode_yolox(raw, 416, 416, ratio=1.0, conf_thres=0.3)
    assert len(dets) == 1
    box, score, cls_id = dets[0]
    assert cls_id == COCO_CELL_PHONE
    assert score == pytest.approx(0.72, abs=1e-3)
    cx = (0.5 + 20) * 8
    cy = (0.5 + 20) * 8
    assert box[0] == pytest.approx(cx - 16.0, abs=0.5)
    assert box[1] == pytest.approx(cy - 20.0, abs=0.5)
    assert box[2] == pytest.approx(cx + 16.0, abs=0.5)


def test_decode_yolox_honours_the_letterbox_ratio():
    raw = _yolox_head_with_one_phone()
    a = decode_yolox(raw, 416, 416, ratio=1.0, conf_thres=0.3)[0][0]
    b = decode_yolox(raw, 416, 416, ratio=0.5, conf_thres=0.3)[0][0]
    assert np.allclose(b, a * 2.0, atol=0.5)


def test_decode_yolox_class_filter():
    raw = _yolox_head_with_one_phone()
    assert decode_yolox(raw, 416, 416, 1.0, keep_classes=[COCO_CELL_PHONE])
    assert decode_yolox(raw, 416, 416, 1.0, keep_classes=[0]) == []


def test_decode_yolox_rejects_a_row_count_mismatch():
    with pytest.raises(ValueError):
        decode_yolox(np.zeros((1, 100, 85), np.float32), 416, 416, 1.0)


def test_object_detector_maps_roi_boxes_back_to_the_frame():
    raw = _yolox_head_with_one_phone()
    eng = FakeEngine("images", (1, 3, 416, 416), {"output": raw})
    od = ObjectDetector(engine=eng, cfg={"roi": [0.0, 0.5, 1.0, 1.0], "every_n": 1, "conf": 0.3})
    dets = od.detect(_frame())
    assert len(dets) == 1
    assert dets[0].name == "cell phone"
    assert dets[0].xyxy[1] >= 360.0, "the ROI y-offset must be added back"
    assert od.phone_confidence(dets) == pytest.approx(0.72, abs=1e-3)
    # YOLOX takes raw 0-255 BGR: the blob must not have been divided by 255
    assert float(eng.last_blob.max()) > 1.5


def test_object_detector_throttles():
    raw = _yolox_head_with_one_phone()
    eng = FakeEngine("images", (1, 3, 416, 416), {"output": raw})
    od = ObjectDetector(engine=eng, cfg={"every_n": 4, "conf": 0.3})
    for _ in range(8):
        od.detect(_frame())
    assert eng.calls == 2


def test_object_detector_without_an_engine_is_silent():
    od = ObjectDetector(engine=None)
    assert od.detect(_frame()) == []
    assert od.phone_confidence([]) == 0.0
    assert not od.available


# ---------------------------------------------------------------- identity --
def test_l2_normalise_and_cosine():
    v = np.array([3.0, 4.0])
    assert np.linalg.norm(l2_normalise(v)) == pytest.approx(1.0)
    assert np.allclose(l2_normalise(np.zeros(4)), np.zeros(4))
    a = l2_normalise(np.array([1.0, 0.0]))
    assert cosine(a, a) == pytest.approx(1.0)


def _kps(cx=640.0, cy=360.0, s=1.0):
    from dms.infer import ARCFACE_REF_112

    return (ARCFACE_REF_112 - 56.0) * s + np.array([cx, cy], np.float32)


def test_identity_enrols_then_recognises():
    vec = np.arange(128, dtype=np.float32).reshape(1, 128)
    eng = FakeEngine("data", (1, 3, 112, 112), {"fc1": vec})
    ide = IdentityEmbedder(engine=eng, cfg={"refresh_n": 1})
    first = ide.observe(_frame(), _kps(), track_id=1)
    assert first.label == "driver-1" and first.refreshed and not first.changed
    second = ide.observe(_frame(), _kps(), track_id=1)
    assert second.label == "driver-1" and not second.changed
    assert ide.known_labels() == ["driver-1"]


def test_identity_reports_a_change_when_the_embedding_flips():
    a = np.zeros((1, 128), np.float32)
    a[0, 0] = 1.0
    eng = FakeEngine("data", (1, 3, 112, 112), {"fc1": a})
    ide = IdentityEmbedder(engine=eng, cfg={"refresh_n": 1})
    ide.observe(_frame(), _kps(), track_id=1)
    b = np.zeros((1, 128), np.float32)
    b[0, 64] = 1.0
    eng._outputs["fc1"] = b
    obs = ide.observe(_frame(), _kps(), track_id=1)
    assert obs.changed and obs.label == "driver-2"


def test_identity_without_an_engine_never_claims_a_change():
    ide = IdentityEmbedder(engine=None)
    obs = ide.observe(_frame(), _kps(), track_id=1)
    assert obs.label is None and not obs.changed
    assert ide.embed(_frame(), _kps()) is None


def test_identity_gallery_is_bounded():
    eng = FakeEngine("data", (1, 3, 112, 112), {"fc1": np.zeros((1, 128), np.float32)})
    ide = IdentityEmbedder(engine=eng, cfg={"refresh_n": 1, "max_identities": 2})
    for i in range(5):
        v = np.zeros((1, 128), np.float32)
        v[0, i * 20] = 1.0
        eng._outputs["fc1"] = v
        ide.observe(_frame(), _kps(), track_id=i)
    assert len(ide.known_labels()) <= 2


# ---------------------------------------------------------------- registry --
def test_gaze_defaults_to_off_and_its_engine_is_never_deserialised(tmp_path):
    """Licence enforcement, not licence documentation."""
    man = {"engines": {"g": {"file": "gaze.engine", "role": "gaze"}}}
    (tmp_path / "gaze.engine").write_bytes(b"x")
    loaded = []

    def factory(path):
        loaded.append(os.path.basename(path))
        return FakeEngine("input", (1, 3, 448, 448), {"yaw": np.zeros((1, 90), np.float32),
                                                      "pitch": np.zeros((1, 90), np.float32)})

    # default: not even requested
    reg = ModelRegistry(manifest=man, engine_dir=str(tmp_path), engine_factory=factory).load()
    assert reg.state("gaze") == "off"
    assert loaded == []
    assert "gaze" not in reg.degraded_capabilities()

    # explicitly requested but the licence flag is off: still never loaded
    reg = ModelRegistry(
        manifest=man,
        engine_dir=str(tmp_path),
        backends={"gaze": "model"},
        engine_factory=factory,
    ).load()
    assert loaded == []
    assert reg.state("gaze") == "off"
    assert "gaze" in reg.degraded_capabilities()

    # research profile: now it loads
    reg = ModelRegistry(
        manifest=man,
        engine_dir=str(tmp_path),
        backends={"gaze": "model"},
        research_models_allowed=True,
        engine_factory=factory,
    ).load()
    assert loaded == ["gaze.engine"]
    assert reg.state("gaze") == "model"


def test_registry_with_an_empty_manifest_degrades_everything():
    reg = ModelRegistry(
        manifest={}, engine_dir="/nonexistent", backends={c: "model" for c in CAPABILITIES}
    ).load()
    for cap, spec in CAPABILITIES.items():
        assert reg.state(cap) == ("geometry" if spec.has_geometry else "off"), cap
    assert set(reg.degraded_capabilities()) == set(CAPABILITIES)
    assert reg.headpose is not None and reg.eyestate is not None
    # and the wrappers still answer with well-formed dataclasses
    pose = reg.headpose.infer(_frame(), _face_box(), _face_landmarks())
    assert pose.source is not SignalSource.MODEL
    assert reg.objects.detect(_frame()) == []


def test_registry_does_not_load_an_alternative_it_will_not_use(tmp_path):
    """Loading the head-pose fallback alongside the primary wastes 6.6 MB."""
    loaded = []

    class _Eng(FakeEngine):
        pass

    man = {
        "engines": {
            "a": {"file": "primary.engine", "role": "head_pose"},
            "b": {"file": "fallback.engine", "role": "head_pose_fallback"},
        }
    }
    for f in ("primary.engine", "fallback.engine"):
        (tmp_path / f).write_bytes(b"x")

    def factory(path):
        loaded.append(os.path.basename(path))
        return _Eng("input", (1, 3, 224, 224), {"rotation_matrix": _rot()})

    reg = ModelRegistry(manifest=man, engine_dir=str(tmp_path), engine_factory=factory).load()
    assert loaded == ["primary.engine"], loaded
    assert reg.state("headpose") == "model"
    # but the occlusion gate needs BOTH of its roles, so it must load both
    assert not CAPABILITIES["occlusion"].alternatives
    assert CAPABILITIES["headpose"].alternatives


def test_registry_falls_through_to_the_alternative_when_the_primary_is_absent(tmp_path):
    man = {"engines": {"b": {"file": "fallback.engine", "role": "head_pose_fallback"}}}
    (tmp_path / "fallback.engine").write_bytes(b"x")
    reg = ModelRegistry(
        manifest=man,
        engine_dir=str(tmp_path),
        engine_factory=lambda p: FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot()}),
    ).load()
    assert reg.state("headpose") == "model"
    assert reg.engine_files() == {"head_pose_fallback": "fallback.engine"}


def test_registry_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        ModelRegistry(manifest={}, backends={"headpose": "magic"})


def test_registry_geometry_backend_never_loads_an_engine(tmp_path):
    made = []

    def factory(path):
        made.append(path)
        raise AssertionError("must not be called")

    man = {"engines": {"hp": {"file": "x.engine", "role": "head_pose"}}}
    p = tmp_path / "x.engine"
    p.write_bytes(b"not-a-real-engine")
    reg = ModelRegistry(
        manifest=man,
        engine_dir=str(tmp_path),
        backends={c: "geometry" for c in CAPABILITIES},
        engine_factory=factory,
    ).load()
    assert made == []
    assert reg.state("headpose") == "geometry"
    assert reg.degraded_capabilities() == []


def test_registry_reports_a_failed_engine_load_without_raising(tmp_path):
    man = {"engines": {"hp": {"file": "x.engine", "role": "head_pose"}}}
    (tmp_path / "x.engine").write_bytes(b"garbage")

    def factory(path):
        raise RuntimeError("failed to deserialize")

    reg = ModelRegistry(manifest=man, engine_dir=str(tmp_path), engine_factory=factory).load()
    assert reg.state("headpose") == "geometry"
    assert "headpose" in reg.degraded_capabilities()
    assert "head_pose" in reg.load_errors()


def test_registry_loads_and_wires_fakes(tmp_path):
    specs = {
        "head_pose": ("hpe.engine", FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot()})),
        "eye_state_pair": ("ocec2.engine", FakeEngine("images", (2, 3, 24, 40), {"prob_open": np.array([0.9, 0.9], np.float32)})),
        "mouth_state": ("vsdlm.engine", FakeEngine("images", (1, 3, 30, 48), {"prob_open": np.array([0.1], np.float32)})),
        "eye_occlusion": ("sgc.engine", FakeEngine("images", (1, 3, 48, 48), {"prob_sunglasses": np.array([0.1], np.float32)})),
        "mouth_occlusion": ("mwc.engine", FakeEngine("images", (1, 3, 48, 48), {"prob_masked": np.array([0.1], np.float32)})),
        "cabin_objects": ("yolox.engine", FakeEngine("images", (1, 3, 416, 416), {"output": np.zeros((1, 3549, 85), np.float32)})),
        "driver_identity": ("sface.engine", FakeEngine("data", (1, 3, 112, 112), {"fc1": np.ones((1, 128), np.float32)})),
    }
    man = {"engines": {}}
    for role, (fname, _eng) in specs.items():
        (tmp_path / fname).write_bytes(b"x")
        man["engines"][role] = {"file": fname, "role": role}
    by_file = {fname: eng for _r, (fname, eng) in specs.items()}

    reg = ModelRegistry(
        manifest=man,
        engine_dir=str(tmp_path),
        engine_factory=lambda p: by_file[os.path.basename(p)],
    ).load()
    for cap in ("headpose", "eyestate", "mouthstate", "occlusion", "objects", "identity"):
        assert reg.state(cap) == "model", cap
    assert reg.state("gaze") == "off"
    assert reg.degraded_capabilities() == []
    assert set(reg.health_map()) == set(CAPABILITIES)

    frame, lm, box = _frame(), _face_landmarks(), _face_box()
    assert reg.headpose.infer(frame, box, lm).source is SignalSource.MODEL
    assert reg.eyestate.infer(frame, lm)[0].source is SignalSource.MODEL
    assert reg.mouthstate.infer(frame, lm).source is SignalSource.MODEL
    assert reg.occlusion.infer(frame, box, track_id=1).source is SignalSource.MODEL
    assert reg.identity.observe(frame, _kps(), track_id=1).label is not None
    assert reg.headpose.health.mean_ms >= 0.0
    reg.forget_track(1)
    reg.close()


def test_registry_health_map_tracks_failures(tmp_path):
    eng = FakeEngine("input", (1, 3, 224, 224), {"rotation_matrix": _rot()}, fail=True)
    (tmp_path / "hpe.engine").write_bytes(b"x")
    man = {"engines": {"hp": {"file": "hpe.engine", "role": "head_pose"}}}
    reg = ModelRegistry(
        manifest=man, engine_dir=str(tmp_path), engine_factory=lambda p: eng
    ).load()
    assert reg.state("headpose") == "model"  # it loaded; it just fails at runtime
    reg.headpose.infer(_frame(), _face_box(), _face_landmarks())
    assert reg.health_map()["headpose"].consecutive_failures == 1


# ------------------------------------------------------ on-board GPU tests --
# These skip themselves when the engine files or TensorRT are absent, so the
# file stays runnable in CI off the Jetson.
def _engine_path(name):
    return os.path.join(ENGINE_DIR, name)


def _skip_without(name):
    p = _engine_path(name)
    if not os.path.isfile(p):
        pytest.skip("engine not on this machine: %s" % name)
    pytest.importorskip("tensorrt")
    return p


def test_real_registry_loads_the_engines_that_are_present():
    pytest.importorskip("tensorrt")
    if not os.path.isdir(ENGINE_DIR):
        pytest.skip("no engines directory")
    reg = ModelRegistry(backends={"gaze": "off"})
    if not reg.manifest().get("engines"):
        pytest.skip("no MANIFEST engines")
    reg.load()
    try:
        states = reg.describe()
        assert set(states) == set(CAPABILITIES)
        for cap, state in states.items():
            assert state in ("model", "geometry", "off"), (cap, state)
        for cap, health in reg.health_map().items():
            assert health.consecutive_failures == 0, cap
    finally:
        reg.close()


#: Sunglasses labels for the replay clips, assigned by looking at contact sheets
#: of the 48x48 head crops the gate is actually fed
#: (``/tmp/contact_<clip>.png``).  1 = a driver in dark sunglasses appears.
#: ``minivan`` and ``getting_rear_ended`` are MONTAGES that cut between
#: subjects - only part of each shows the sunglasses wearer - so they are
#: labelled 1 but only asserted weakly.  ``guy_rear_ended`` wears CLEAR
#: prescription glasses and is a negative on purpose: it is the clip a too-tight
#: crop turns into a false positive.  ``family_vacation_crash`` (wire-rimmed
#: dark glasses throughout) was unlabelled by every earlier review.
SUNGLASSES_CLIPS = {
    # clip: (label, "strong" | "partial")
    "bmw_rear_ended": (1, "strong"),
    "family_vacation_crash": (1, "strong"),
    "minivan": (1, "partial"),
    "day_driver": (0, "strong"),
    "guy_rear_ended": (0, "strong"),
    "red_car_crash": (0, "strong"),
}
REPLAY_DIR = os.path.join(ROOT, "tests", "replay")
#: Replay sampling that matches what ``configs/*.yaml`` actually run: the 512
#: detector and one occlusion refresh every 15th frame.
REPLAY_STRIDE = 15
REPLAY_REFRESHES = 24

#: SCRFD is the slow part of these on-board tests, so the sampled faces are
#: detected once and shared between them.
_REPLAY_FACES = {}


def _sample_faces(clip, n_wanted=REPLAY_REFRESHES, stride=REPLAY_STRIDE):
    """Up to ``n_wanted`` (frame, face_xyxy) pairs from a replay clip, cached.

    No minimum face size: the point is to reproduce what the pipeline sees, and
    the pipeline has no face-size gate in front of the occlusion models.
    """
    import cv2

    from dms.infer.scrfd import ScrfdDetector
    from dms.runtime.trt_engine import TrtEngine

    key = (clip, n_wanted, stride)
    if key in _REPLAY_FACES:
        return _REPLAY_FACES[key]
    engine = TrtEngine(_engine_path("scrfd_500m_512_fp16_gpu.engine"))
    try:
        det = ScrfdDetector(engine)
        cap = cv2.VideoCapture(os.path.join(REPLAY_DIR, clip + ".mp4"))
        out = []
        n = 0
        while len(out) < n_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % stride:
                continue
            faces = det.detect(frame)
            if not faces:
                continue
            f = max(faces, key=lambda d: d.xyxy[2] - d.xyxy[0])
            out.append((frame, np.asarray(f.xyxy, np.float32)))
        cap.release()
    finally:
        engine.close()
    _REPLAY_FACES[key] = out
    return out


def _gate_verdicts(engine_name, engine_kw):
    """(clip -> (n, fired)) for one occlusion engine over the labelled clips."""
    from dms.runtime.trt_engine import TrtEngine

    path = _skip_without(engine_name)
    _skip_without("scrfd_500m_512_fp16_gpu.engine")
    if not os.path.isdir(REPLAY_DIR):
        pytest.skip("no replay clips on this machine")
    verdicts = {}
    for clip in SUNGLASSES_CLIPS:
        if not os.path.isfile(os.path.join(REPLAY_DIR, clip + ".mp4")):
            continue
        samples = _sample_faces(clip)
        if len(samples) < 12:
            continue
        eng = TrtEngine(path)
        try:
            det = OcclusionDetector(cfg={"every_n": 1}, **{engine_kw: eng})
            flags = []
            for frame, box in samples:
                occ = det.infer(frame, box, track_id=1)
                if occ.source is not SignalSource.MODEL:
                    continue
                flags.append(occ)
        finally:
            eng.close()
        verdicts[clip] = flags
    return verdicts


def test_sgc_gate_catches_the_labelled_sunglasses_drivers_in_the_replay_set():
    """The regression the shipped crop convention failed.

    Measured over the first 24 refreshes of each clip, at the shipped detector
    and cadence, old crop (``square_box(expand_box(face, 0.25))`` through the
    zero-padding ``crop_patch``) versus new (unsquared, clipped, 5-refresh
    median)::

        clip                    label   OLD ON   NEW ON
        bmw_rear_ended            1      24/24    24/24
        family_vacation_crash     1      18/24    24/24
        minivan                   1       0/24     3/24
        day_driver                0       0/24     0/24
        guy_rear_ended            0       0/24     0/24
        red_car_crash             0       0/24     0/24

    ``minivan`` is the proven defect: a driver in dark sunglasses whom the old
    crop never once flagged, so MICROSLEEP and PERCLOS ran on eyes the system
    could not see.  Over the WHOLE clip the rate goes 0.19 -> 0.42; the 3/24
    asserted here is the prefix a fast test can afford.
    """
    verdicts = _gate_verdicts("sgc_is_s_48_fp16_gpu.engine", "sunglasses_engine")
    assert len(verdicts) >= 4, "not enough replay clips: %s" % list(verdicts)
    summary = {c: (sum(1 for o in f if not o.eyes_reliable), len(f)) for c, f in verdicts.items()}
    for clip, flags in verdicts.items():
        label, strength = SUNGLASSES_CLIPS[clip]
        fired, total = summary[clip]
        if label == 1 and strength == "strong":
            assert fired >= total - 1, "gate missed %s: %d/%d (%s)" % (clip, fired, total, summary)
        elif label == 1:
            assert fired >= 1, (
                "gate never fired on the %s sunglasses driver: %d/%d (%s)"
                % (clip, fired, total, summary)
            )
        else:
            assert fired == 0, "false positive on %s: %d/%d (%s)" % (clip, fired, total, summary)


def test_mwc_mask_gate_behaviour_on_the_unmasked_replay_corpus():
    """The mask gate is UNVALIDATED, and it is not even clean on negatives.

    No replay clip contains a masked face, so ``prob_masked``'s positive class
    has never been exercised on this board.  On the negative side the median is
    ~0.00 on every clip, but the head spikes to 1.00 on a bearded driver and the
    5-refresh vote still declares "masked" on 3 of the first 24 refreshes of
    ``bmw_rear_ended`` - each one a silent YAWN shutdown.

    This test pins both halves of that: the gate must stay quiet in the median
    (a preprocessing regression that made MWC fire constantly would break it),
    and the known false-positive rate must not grow.  It deliberately does NOT
    assert zero, because zero is not what was measured, and it cannot assert
    anything about true masks until a masked face exists in ``tests/replay``.
    """
    verdicts = _gate_verdicts("mwc_n_48_fp16_gpu.engine", "mask_engine")
    assert len(verdicts) >= 4, "not enough replay clips: %s" % list(verdicts)
    summary = {}
    for clip, flags in verdicts.items():
        fired = sum(1 for o in flags if not o.mouth_reliable)
        med = float(np.median([o.mask for o in flags]))
        summary[clip] = (fired, len(flags), round(med, 3))
        assert med < 0.50, "mask gate median fired on unmasked %s: %s" % (clip, summary)
        assert fired <= max(1, len(flags) // 4), (
            "mask false-positive rate grew on %s: %s" % (clip, summary)
        )


def test_real_eye_engine_returns_probabilities():
    from dms.runtime.trt_engine import TrtEngine

    path = _skip_without("ocec_s_24x40_b2_fp16_gpu.engine")
    eng = TrtEngine(path)
    try:
        es = EyeStateEstimator(engine=eng)
        left, right = es.infer(_frame(), _face_landmarks())
        for st in (left, right):
            assert st.source is SignalSource.MODEL
            assert 0.0 <= st.prob_open <= 1.0
    finally:
        eng.close()
