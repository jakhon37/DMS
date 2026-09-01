from __future__ import annotations

import os

import pytest

from dms.capture.pipelines import build_pipeline
from dms.config.schema import load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.skipif(os.environ.get("DMS_SKIP_GST") == "1", reason="explicit skip")
def test_parse_videotestsrc_pipeline():
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    cfg.source.type = "test"
    desc = build_pipeline(cfg)
    pipeline = Gst.parse_launch(desc)
    assert pipeline is not None
    pipeline.set_state(Gst.State.NULL)
