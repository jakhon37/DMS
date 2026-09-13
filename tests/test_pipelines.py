from __future__ import annotations

import os

import pytest

from dms.capture.pipelines import build_pipeline, gst_quote, validate_source
from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def cfg_default():
    return load_config(os.path.join(ROOT, "configs/default.yaml"))


def test_file_pipeline_is_bgrx_not_bgr():
    cfg = cfg_default()
    desc = build_pipeline(cfg)
    assert "format=BGRx" in desc
    assert "format=BGR !" not in desc
    assert "appsink name=full" in desc
    assert "filesrc" in desc
    assert "drop=false" in desc


def test_test_pipeline():
    cfg = cfg_default()
    cfg.source.type = "test"
    desc = build_pipeline(cfg)
    assert "videotestsrc" in desc
    assert "format=BGRx" in desc
    assert "drop=true" in desc


def test_usb_and_csi_pipelines():
    cfg = cfg_default()
    cfg.source.type = "usb"
    desc = build_pipeline(cfg)
    assert 'v4l2src device="/dev/video0"' in desc
    assert "nvv4l2decoder mjpeg=1" in desc
    cfg.source.type = "csi"
    desc = build_pipeline(cfg)
    assert "nvarguscamerasrc sensor-id=0" in desc
    assert "memory:NVMM" in desc


def test_unknown_source_type_raises():
    cfg = cfg_default()
    cfg.source.type = "carrier-pigeon"
    with pytest.raises(ValueError):
        build_pipeline(cfg)


# ------------------------------------------------------- parse_launch injection


def test_paths_with_spaces_are_quoted_not_broken():
    cfg = cfg_default()
    cfg.source.path = "/data/clips/day driver 01.mp4"
    desc = build_pipeline(cfg)
    assert 'location="/data/clips/day driver 01.mp4"' in desc


def test_a_path_that_would_rewire_the_pipeline_is_contained():
    cfg = cfg_default()
    cfg.source.path = "/tmp/x ! fakesink name=full ! filesrc location=/etc/shadow"
    desc = build_pipeline(cfg)
    # The whole hostile value must sit inside one quoted property value, so the
    # pipeline still has exactly one appsink and one filesrc element.
    first = desc.index('"')
    last = desc.index('"', first + 1)
    quoted = desc[first + 1 : last]
    assert quoted == cfg.source.path
    outside = desc[:first] + desc[last + 1 :]
    assert outside.count("appsink name=full") == 1
    assert outside.count("filesrc") == 1
    assert "fakesink" not in outside


@pytest.mark.parametrize("bad", ['/tmp/a"b.mp4', "/tmp/a\\b.mp4", "/tmp/a\nb.mp4", ""])
def test_unrepresentable_paths_are_rejected(bad):
    cfg = cfg_default()
    cfg.source.path = bad
    with pytest.raises(ValueError):
        build_pipeline(cfg)


def test_gst_quote_directly():
    assert gst_quote("/dev/video0", "source.device") == '"/dev/video0"'
    with pytest.raises(ValueError):
        gst_quote("", "source.path")
    with pytest.raises(ValueError):
        gst_quote('a"b', "source.path")


def test_zero_geometry_is_rejected():
    cfg = cfg_default()
    cfg.source.width = 0
    with pytest.raises(ValueError):
        build_pipeline(cfg)
    cfg = cfg_default()
    cfg.source.fps = 0
    with pytest.raises(ValueError):
        build_pipeline(cfg)


# ---------------------------------------------------------------- validation


def test_validate_source_rejects_a_missing_clip(tmp_path):
    cfg = cfg_default()
    cfg.source.type = "file"
    cfg.source.path = str(tmp_path / "not-here.mp4")
    with pytest.raises(ValueError) as ei:
        validate_source(cfg)
    assert "not a file" in str(ei.value)


def test_validate_source_accepts_an_existing_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    cfg = cfg_default()
    cfg.source.type = "file"
    cfg.source.path = str(clip)
    assert validate_source(cfg) == []


def test_validate_source_checks_the_device_node_shape():
    cfg = cfg_default()
    cfg.source.type = "usb"
    cfg.source.device = "/dev/ttyUSB0"
    with pytest.raises(ValueError) as ei:
        validate_source(cfg)
    assert "/dev/videoN" in str(ei.value)


def test_validate_source_warns_but_does_not_fail_on_an_absent_camera():
    cfg = cfg_default()
    cfg.source.type = "usb"
    cfg.source.device = "/dev/video31"
    warnings = validate_source(cfg)
    assert any("does not exist" in w for w in warnings)


def test_validate_source_test_pattern_needs_nothing():
    cfg = cfg_default()
    cfg.source.type = "test"
    assert validate_source(cfg) == []
