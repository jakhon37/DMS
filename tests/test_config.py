from __future__ import annotations

import os

import pytest

from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
