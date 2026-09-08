from __future__ import annotations

import os
import sys

import pytest

from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_replay_batch_config_for_clip():
    sys_path_scripts = os.path.join(ROOT, "scripts")
    if sys_path_scripts not in sys.path:
        sys.path.insert(0, sys_path_scripts)
    import run_all_replay as batch

    assert batch.config_for_clip("tests/replay/day_driver.mp4") == "configs/default.yaml"
    assert batch.config_for_clip("tests/replay/minivan.mp4") == "configs/minivan.yaml"
    assert batch.config_for_clip("tests/replay/red_car_crash.mp4") == "configs/red_car_crash.yaml"
    assert batch.config_for_clip("tests/replay/family_vacation_crash.mp4") == "configs/family_vacation_crash.yaml"
    assert batch.config_for_clip("tests/replay/rear_end_accident.mp4") == "configs/rear_end_accident.yaml"
    assert batch.config_for_clip("tests/replay/guy_rear_ended.mp4") == "configs/guy_rear_ended.yaml"
    assert batch.config_for_clip("tests/replay/bmw_rear_ended.mp4") == "configs/bmw_rear_ended.yaml"
    assert batch.config_for_clip("tests/replay/getting_rear_ended.mp4") == "configs/getting_rear_ended.yaml"
    assert batch.config_for_clip("tests/replay/rear_end_whiplash.mp4") == "configs/rear_end_whiplash.yaml"
    assert batch.config_for_clip("tests/replay/unknown_clip.mp4") == "configs/default.yaml"


def test_default_yaml_loads():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    assert cfg.source.type == "file"
    assert cfg.source.dev is True
    assert cfg.source.width == 1280
    assert cfg.require_engines is False


def test_production_yaml_loads():
    cfg = load_config(os.path.join(ROOT, "configs/production.yaml"))
    assert cfg.source.type == "csi"
    assert cfg.source.dev is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("boom: 1\nsource: {type: file, path: x}\nmodels:\n  face: {engine: e}\n  landmarks: {engine: e}\n")
    with pytest.raises(Exception):
        load_config(str(p))


def test_bad_source_type(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("source: {type: webcam}\nmodels:\n  face: {engine: e}\n  landmarks: {engine: e}\n")
    with pytest.raises(Exception):
        load_config(str(p))
