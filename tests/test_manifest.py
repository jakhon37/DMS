from __future__ import annotations

import json
import os

import pytest

from dms.runtime.manifest import (
    ManifestError,
    check_engine_file,
    check_trt_version,
    load_manifest,
    sha256_file,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_manifest_loads_and_lists_engines():
    man = load_manifest()
    assert man["trt_version_required"] == "8.5.2.2"
    assert "scrfd_500m_640_fp16_gpu" in man["engines"]
    assert man["engines"]["2d106det_192_fp16_gpu"]["sha256"]


def test_sha256_mismatch_raises(tmp_path):
    man = {
        "engines": {
            "x": {"file": "bogus.engine", "sha256": "00" * 32},
        }
    }
    p = tmp_path / "bogus.engine"
    p.write_bytes(b"not-an-engine")
    with pytest.raises(ManifestError) as ei:
        check_engine_file(man, str(p), require_hash=True)
    assert "sha256 mismatch" in str(ei.value)


def test_trt_version_mismatch():
    pytest.importorskip("tensorrt")
    with pytest.raises(ManifestError) as ei:
        check_trt_version({"trt_version_required": "9.9.9.9"})
    assert "ENGINE_VERSION_MISMATCH" in str(ei.value)


def test_live_engines_match_manifest_if_present():
    man = load_manifest()
    for rec in man["engines"].values():
        path = os.path.join(ROOT, "engines", rec["file"])
        if not os.path.isfile(path):
            pytest.skip("engine not on disk")
        assert sha256_file(path) == rec["sha256"]
