from __future__ import annotations

import os

from dms.capture.pipelines import build_pipeline
from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_file_pipeline_is_bgrx_not_bgr():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    desc = build_pipeline(cfg)
    assert "format=BGRx" in desc
    assert "format=BGR !" not in desc
    assert "appsink name=full" in desc
    assert "filesrc" in desc
    assert "drop=false" in desc


def test_test_pipeline():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    cfg.source.type = "test"
    desc = build_pipeline(cfg)
    assert "videotestsrc" in desc
    assert "format=BGRx" in desc
    assert "drop=true" in desc
