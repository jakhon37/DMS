from __future__ import annotations

import numpy as np

from dms.infer.scrfd import _group_outputs, nms_xyxy, resize_letterbox


def test_resize_720p_to_640():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, factor = resize_letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert abs(factor - (360 / 720)) < 1e-6  # width-limited: 640/1280=0.5, height 360


def test_nms_keeps_higher_score():
    dets = np.array(
        [
            [0, 0, 10, 10, 0.9],
            [1, 1, 11, 11, 0.5],
            [50, 50, 80, 80, 0.8],
        ],
        dtype=np.float32,
    )
    keep = nms_xyxy(dets, 0.4)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


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
