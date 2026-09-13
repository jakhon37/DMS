"""SCRFD decode: letterbox, NMS, head grouping, grid sizing, and a real fixture.

The numerical regression test loads ``tests/fixtures/scrfd_decode.npz``, which
holds the nine raw head tensors captured from the shipping engine on one replay
frame together with the boxes the decode produced at the time.  It runs the pure
decode - no GPU, no TensorRT - and asserts the boxes still match.  That is the
only thing that catches a silent decode regression.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from dms.infer.scrfd import (
    FaceDet,
    ScrfdDetector,
    _anchor_centers,
    _group_outputs,
    distance2bbox,
    nms_xyxy,
    resize_letterbox,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "scrfd_decode.npz")


class _FakeEngine:
    """A TrtEngine stand-in that replays captured tensors. No GPU needed."""

    def __init__(self, in_name, in_shape, out_names, outputs=None):
        self.input_name = in_name
        self.input_shape = tuple(in_shape)
        self.output_names = list(out_names)
        self._outputs = outputs or {}
        self.calls = 0

    def infer(self, feed):
        self.calls += 1
        return dict(self._outputs)


def test_resize_720p_to_640():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, factor = resize_letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert abs(factor - (360 / 720)) < 1e-6


def test_resize_720p_to_512():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, factor = resize_letterbox(img, (512, 512))
    assert canvas.shape == (512, 512, 3)
    assert abs(factor - (288 / 720)) < 1e-6


def test_nms_keeps_higher_score():
    dets = np.array(
        [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.5], [50, 50, 80, 80, 0.8]], dtype=np.float32
    )
    keep = nms_xyxy(dets, 0.4)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_nms_on_empty_input():
    assert nms_xyxy(np.zeros((0, 5), dtype=np.float32), 0.4) == []


def test_group_interleaved_layout():
    outs = [
        np.zeros((12800, 1), np.float32),
        np.zeros((12800, 4), np.float32),
        np.zeros((12800, 10), np.float32),
        np.zeros((3200, 1), np.float32),
        np.zeros((3200, 4), np.float32),
        np.zeros((3200, 10), np.float32),
        np.zeros((800, 1), np.float32),
        np.zeros((800, 4), np.float32),
        np.zeros((800, 10), np.float32),
    ]
    g = _group_outputs(outs)
    assert len(g) == 3
    assert g[0][0].shape[0] == 12800
    assert g[1][1].shape[-1] == 4
    assert g[2][2].shape[-1] == 10


def test_group_rejects_an_unknown_layout():
    with pytest.raises(RuntimeError):
        _group_outputs([np.zeros((10, 7), np.float32)] * 9)


def test_anchor_counts_match_the_documented_row_counts():
    for size, expect in ((640, (12800, 3200, 800)), (512, (8192, 2048, 512))):
        got = tuple(_anchor_centers(size, size, s, 2).shape[0] for s in (8, 16, 32))
        assert got == expect, "%d: %s" % (size, got)


def test_distance2bbox():
    pts = np.array([[10.0, 20.0]], dtype=np.float32)
    dist = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    assert np.allclose(distance2bbox(pts, dist), [[9.0, 18.0, 13.0, 24.0]])


# --------------------------------------------------- engine-driven sizing ---
def _out_names():
    return [
        "443", "446", "449",
        "468", "471", "474",
        "493", "496", "499",
    ]


def _zero_heads(rows):
    outs = {}
    names = _out_names()
    for i, n in enumerate(rows):
        outs[names[3 * i + 0]] = np.zeros((n, 1), np.float32)
        outs[names[3 * i + 1]] = np.zeros((n, 4), np.float32)
        outs[names[3 * i + 2]] = np.zeros((n, 10), np.float32)
    return outs


def test_detector_reads_its_input_size_from_the_engine():
    """The 512 engine must not be decoded against a 640 grid."""
    for size, rows in ((640, (12800, 3200, 800)), (512, (8192, 2048, 512))):
        eng = _FakeEngine("input.1", (1, 3, size, size), _out_names(), _zero_heads(rows))
        det = ScrfdDetector(eng, conf=0.5)
        assert det.input_size == (size, size)
        for stride, n in zip((8, 16, 32), rows):
            assert det._centers[stride].shape[0] == n


def test_detector_raises_when_grid_and_engine_disagree():
    """The old `min(len(scores), len(centers))` clamp hid this and mislocated faces."""
    eng = _FakeEngine("input.1", (1, 3, 640, 640), _out_names(), _zero_heads((8192, 2048, 512)))
    det = ScrfdDetector(eng, conf=0.5)
    with pytest.raises(RuntimeError) as ei:
        det.postprocess(eng.infer({}), 0.5, (720, 1280))
    assert "mislocated" in str(ei.value)


def test_detect_on_an_all_zero_head_returns_nothing():
    eng = _FakeEngine("input.1", (1, 3, 640, 640), _out_names(), _zero_heads((12800, 3200, 800)))
    det = ScrfdDetector(eng, conf=0.5)
    assert det.detect(np.zeros((720, 1280, 3), np.uint8)) == []
    assert eng.calls == 1


def test_facedet_keeps_the_unclipped_box_for_crops():
    clipped = np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32)
    raw = np.array([-30.0, -20.0, 50.0, 50.0], dtype=np.float32)
    f = FaceDet(xyxy=clipped, score=0.9, xyxy_raw=raw)
    assert np.allclose(f.box_for_crop(), raw)
    # and degrades to the clipped box when no raw box was recorded
    assert np.allclose(FaceDet(xyxy=clipped, score=0.9).box_for_crop(), clipped)


# ------------------------------------------------------ numerical fixture ---
def test_decode_matches_the_captured_fixture():
    if not os.path.isfile(FIXTURE):
        pytest.skip("SCRFD fixture not captured on this machine: %s" % FIXTURE)
    d = np.load(FIXTURE)
    names = [str(n) for n in d["out_names"]]
    outs = {n: d["out_%d" % i] for i, n in enumerate(names)}
    eng = _FakeEngine(
        str(d["in_name"]), tuple(int(v) for v in d["in_shape"]), names, outs
    )
    det = ScrfdDetector(eng, conf=float(d["conf"]), iou=float(d["iou"]))
    faces = det.postprocess(
        outs, float(d["factor"]), (int(d["orig_h"]), int(d["orig_w"]))
    )
    expect_boxes = d["boxes"]
    expect_scores = d["scores"]
    expect_kps = d["kps"]
    assert len(faces) == expect_boxes.shape[0], "detection count changed"
    got_boxes = np.stack([f.xyxy for f in faces]) if faces else np.zeros((0, 4), np.float32)
    got_scores = np.array([f.score for f in faces], dtype=np.float32)
    got_kps = np.stack([f.kps for f in faces]) if faces else np.zeros((0, 5, 2), np.float32)
    assert np.allclose(got_boxes, expect_boxes, atol=0.05)
    assert np.allclose(got_scores, expect_scores, atol=1e-4)
    assert np.allclose(got_kps, expect_kps, atol=0.05)
