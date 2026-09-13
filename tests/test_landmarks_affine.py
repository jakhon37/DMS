"""Landmark crop affine, crop-coverage gating, and the shared crop primitives."""
from __future__ import annotations

import numpy as np
import pytest

from dms.infer import (
    ARCFACE_REF_112,
    align_5pt,
    box_inside_fraction,
    clip_box,
    crop_patch,
    expand_box,
    sigmoid,
    sixdrepnet_expand,
    softmax,
    square_box,
    to_blob,
)
from dms.infer.landmarks import (
    MIN_CROP_INSIDE,
    apply_affine,
    crop_square_xyxy,
    invert_affine,
    loose_crop_affine,
)


def test_affine_roundtrip():
    xyxy = np.array([100.0, 80.0, 220.0, 220.0], dtype=np.float32)
    M = loose_crop_affine(xyxy, 192, 1.5)
    Minv = invert_affine(M)
    cx, cy = 160.0, 150.0
    crop = apply_affine(np.array([[cx, cy]], dtype=np.float32), M)
    assert abs(crop[0, 0] - 96) < 2
    assert abs(crop[0, 1] - 96) < 2
    back = apply_affine(crop, Minv)
    assert np.allclose(back, [[cx, cy]], atol=1e-2)


def test_affine_roundtrip_over_a_grid_of_boxes():
    for box in (
        [0.0, 0.0, 40.0, 40.0],
        [600.0, 300.0, 900.0, 700.0],
        [-30.0, -20.0, 90.0, 140.0],  # box hanging off the top-left corner
        [1200.0, 640.0, 1400.0, 820.0],  # off the bottom-right
    ):
        xyxy = np.array(box, dtype=np.float32)
        M = loose_crop_affine(xyxy, 192, 1.5)
        Minv = invert_affine(M)
        pts = np.random.RandomState(0).uniform(-50, 250, size=(20, 2)).astype(np.float32)
        back = apply_affine(apply_affine(pts, M), Minv)
        assert np.allclose(back, pts, atol=1e-3)


def test_crop_square_matches_the_affine_source_rect():
    """crop_square_xyxy must describe exactly what loose_crop_affine samples."""
    xyxy = np.array([100.0, 80.0, 220.0, 220.0], dtype=np.float32)
    M = loose_crop_affine(xyxy, 192, 1.5)
    square = crop_square_xyxy(xyxy, 1.5)
    corners = np.array(
        [[square[0], square[1]], [square[2], square[1]], [square[0], square[3]]],
        dtype=np.float32,
    )
    mapped = apply_affine(corners, M)
    assert np.allclose(mapped, [[0, 0], [192, 0], [0, 192]], atol=1e-3)


def test_box_inside_fraction():
    assert box_inside_fraction([10, 10, 20, 20], (100, 100)) == pytest.approx(1.0)
    assert box_inside_fraction([-10, 0, 10, 10], (100, 100)) == pytest.approx(0.5)
    assert box_inside_fraction([-50, -50, -10, -10], (100, 100)) == pytest.approx(0.0)
    assert box_inside_fraction([0, 0, 0, 0], (100, 100)) == pytest.approx(0.0)


def test_min_crop_inside_is_a_real_gate():
    """The old `if crop.size == 0` guard was unreachable; this one is not."""
    assert 0.0 < MIN_CROP_INSIDE < 1.0
    # a face box whose 1.5x square is mostly off-frame must fall below the gate
    box = np.array([-40.0, 10.0, 40.0, 110.0], dtype=np.float32)
    square = crop_square_xyxy(box, 1.5)
    assert box_inside_fraction(square, (1280, 720)) < MIN_CROP_INSIDE


# ------------------------------------------------------------ crop_patch ----
def test_crop_patch_shape_and_content():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:40, 50:90] = 200
    patch, inside = crop_patch(img, [50, 20, 90, 40], 40, 20)
    assert patch.shape == (20, 40, 3)
    assert inside == pytest.approx(1.0)
    assert int(patch.mean()) == 200


def test_crop_patch_pads_instead_of_shifting():
    img = np.full((100, 200, 3), 90, dtype=np.uint8)
    patch, inside = crop_patch(img, [-20, -20, 20, 20], 40, 40, border_value=7)
    assert patch.shape == (40, 40, 3)
    assert inside == pytest.approx(0.25, abs=0.02)
    assert patch[0, 0, 0] == 7  # padded region
    assert patch[-1, -1, 0] == 90  # real image


def test_crop_patch_returns_none_when_fully_outside():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    patch, inside = crop_patch(img, [500, 500, 540, 540], 10, 10)
    assert patch is None
    assert inside == 0.0


def test_crop_patch_handles_a_degenerate_box():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    patch, _ = crop_patch(img, [10.0, 10.0, 10.0, 10.0], 8, 8)
    assert patch is not None and patch.shape == (8, 8, 3)


def test_crop_patch_reads_only_the_requested_rectangle():
    """Regression guard for the project rule: crops are O(crop), not O(frame).

    Everything outside the requested rectangle of a 720p frame is filled with a
    marker value; if any of it reached the output the assertion below fails,
    which is what a full-frame resize path would do.
    """
    big = np.full((720, 1280, 3), 255, dtype=np.uint8)
    big[300:324, 600:640] = 33
    patch, inside = crop_patch(big, [600, 300, 640, 324], 40, 24)
    assert patch.shape == (24, 40, 3)
    assert inside == pytest.approx(1.0)
    assert patch.min() == 33 and patch.max() == 33


# ----------------------------------------------------------------- boxes ----
def test_expand_square_and_clip():
    b = np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32)
    e = expand_box(b, 0.25)
    assert e[0] == pytest.approx(75.0) and e[3] == pytest.approx(350.0)
    s = square_box(b)
    assert (s[2] - s[0]) == pytest.approx(s[3] - s[1])
    assert (s[2] - s[0]) == pytest.approx(200.0)
    c = clip_box([-5, -5, 5000, 5000], (640, 480))
    assert list(c) == [0.0, 0.0, 639.0, 479.0]


def test_sixdrepnet_expand_keeps_the_upstream_axis_swap():
    """x grows by 0.2*height and y by 0.2*width - the quirk is deliberate."""
    b = np.array([0.0, 0.0, 100.0, 200.0], dtype=np.float32)  # w=100, h=200
    e = sixdrepnet_expand(b)
    assert e[0] == pytest.approx(-40.0)  # 0.2 * height
    assert e[1] == pytest.approx(-20.0)  # 0.2 * width
    assert e[2] == pytest.approx(140.0)
    assert e[3] == pytest.approx(220.0)


# ------------------------------------------------------------------ blob ----
def test_to_blob_layout_and_channel_order():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 1] = 20  # G
    img[..., 2] = 30  # R
    blob = to_blob(img, rgb=True, scale=1.0)
    assert blob.shape == (1, 3, 4, 6)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    assert blob[0, 0].mean() == pytest.approx(30.0)  # first plane is R
    assert blob[0, 2].mean() == pytest.approx(10.0)  # last plane is B
    bgr_blob = to_blob(img, rgb=False, scale=1.0)
    assert bgr_blob[0, 0].mean() == pytest.approx(10.0)


def test_to_blob_normalisation():
    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    blob = to_blob(img, rgb=True, scale=1.0 / 255.0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert np.allclose(blob, 1.0)


def test_align_5pt_puts_the_keypoints_on_the_arcface_reference():
    rng = np.random.RandomState(1)
    img = rng.randint(0, 255, size=(400, 600, 3)).astype(np.uint8)
    # a similarity transform of the reference: scale 2, rotate 10 deg, translate
    th = np.radians(10.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    kps = (ARCFACE_REF_112 @ R.T) * 2.0 + np.array([200.0, 120.0], dtype=np.float32)
    aligned, inside = align_5pt(img, kps, 112)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)
    assert inside == pytest.approx(1.0)


def test_align_5pt_rejects_bad_keypoints():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert align_5pt(img, np.zeros((3, 2), dtype=np.float32))[0] is None
    bad = np.full((5, 2), np.nan, dtype=np.float32)
    assert align_5pt(img, bad)[0] is None


# ----------------------------------------------------------------- maths ----
def test_sigmoid_is_stable_at_the_extremes():
    v = sigmoid(np.array([-800.0, 0.0, 800.0]))
    assert np.isfinite(v).all()
    assert v[0] == pytest.approx(0.0)
    assert v[1] == pytest.approx(0.5)
    assert v[2] == pytest.approx(1.0)


def test_softmax_sums_to_one():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)
    assert p[1] > p[0] > p[2]
