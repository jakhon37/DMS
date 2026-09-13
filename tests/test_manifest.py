from __future__ import annotations

import json
import os

import pytest

from dms.runtime.manifest import (
    ManifestError,
    atomic_write_json,
    check_binding_contract,
    check_engine_file,
    check_environment,
    check_runtime,
    check_trt_version,
    cuda_version_str,
    l4t_release,
    load_manifest,
    record_engine_sha256,
    runtime_env,
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


# ------------------------------------------------------------------ invariants


def test_every_engine_record_declares_a_static_binding_contract():
    """The whole pipeline depends on this: real bindings, no dynamic dimension."""
    man = load_manifest()
    for key, rec in man["engines"].items():
        assert rec.get("inputs"), "%s has no recorded inputs" % key
        assert rec.get("outputs"), "%s has no recorded outputs" % key
        for side in ("inputs", "outputs"):
            for ent in rec[side]:
                assert ent.get("name"), "%s %s entry has no name" % (key, side)
                shape = ent.get("shape")
                assert shape, "%s %s %s has no shape" % (key, side, ent.get("name"))
                assert all(int(x) > 0 for x in shape), "%s %s %s is dynamic: %s" % (
                    key,
                    side,
                    ent.get("name"),
                    shape,
                )


def test_manifest_blocked_section_is_empty_or_explained():
    man = load_manifest()
    blocked = man.get("blocked") or {}
    for name, reason in blocked.items():
        assert reason, "blocked entry %s must carry a reason" % name


def test_dynamic_shape_in_a_record_is_rejected():
    rec = {"file": "x.engine", "inputs": [{"name": "in", "shape": [1, 3, -1, -1]}]}
    with pytest.raises(ManifestError) as ei:
        check_binding_contract(rec, "x.engine")
    assert "dynamic" in str(ei.value)


# ------------------------------------------------------------------- matching


def _write(tmp_path, name, data=b"engine-bytes"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_exact_path_beats_basename(tmp_path):
    good = _write(tmp_path, "a/model.engine", b"good")
    other = _write(tmp_path, "b/model.engine", b"other")
    man = {
        "engines": {
            "a": {"path": str(good), "sha256": sha256_file(str(good))},
            "b": {"path": str(other), "sha256": sha256_file(str(other))},
        }
    }
    check_engine_file(man, str(good), require_hash=True)
    check_engine_file(man, str(other), require_hash=True)


def test_ambiguous_basename_raises_instead_of_guessing(tmp_path):
    target = _write(tmp_path, "c/model.engine", b"target")
    man = {
        "engines": {
            "a": {"path": "engines/x/model.engine", "sha256": "00" * 32},
            "b": {"path": "engines/y/model.engine", "sha256": "11" * 32},
        }
    }
    with pytest.raises(ManifestError) as ei:
        check_engine_file(man, str(target), require_hash=True)
    assert "ambiguous" in str(ei.value)


def test_size_mismatch_is_caught_before_hashing(tmp_path):
    p = _write(tmp_path, "m.engine", b"1234567890")
    man = {"engines": {"m": {"file": "m.engine", "bytes": 999, "sha256": "00" * 32}}}
    with pytest.raises(ManifestError) as ei:
        check_engine_file(man, str(p), require_hash=True)
    assert "size mismatch" in str(ei.value)


def test_unlisted_engine_is_fatal_only_when_hashes_are_required(tmp_path):
    p = _write(tmp_path, "unknown.engine")
    man = {"engines": {}}
    check_engine_file(man, str(p), require_hash=False)  # warns
    with pytest.raises(ManifestError) as ei:
        check_engine_file(man, str(p), require_hash=True)
    assert "not listed" in str(ei.value)


def test_missing_engine_file_is_always_fatal(tmp_path):
    with pytest.raises(ManifestError) as ei:
        check_engine_file({"engines": {}}, str(tmp_path / "nope.engine"))
    assert "ENGINE_FAIL missing" in str(ei.value)


# ----------------------------------------------------------------- load / write


def test_corrupt_manifest_is_a_manifest_error_not_a_traceback(tmp_path):
    p = tmp_path / "MANIFEST.json"
    p.write_text('{"engines": {')
    with pytest.raises(ManifestError) as ei:
        load_manifest(str(p))
    assert "ENGINE_VERSION_MISMATCH" in str(ei.value)


def test_missing_manifest_is_a_manifest_error(tmp_path):
    with pytest.raises(ManifestError) as ei:
        load_manifest(str(tmp_path / "absent.json"))
    assert "MANIFEST missing" in str(ei.value)


def test_atomic_write_leaves_no_temp_file_and_round_trips(tmp_path):
    p = tmp_path / "MANIFEST.json"
    atomic_write_json(str(p), {"a": 1, "engines": {}})
    assert json.loads(p.read_text())["a"] == 1
    atomic_write_json(str(p), {"a": 2, "engines": {}})
    assert json.loads(p.read_text())["a"] == 2
    leftovers = [n for n in os.listdir(str(tmp_path)) if n.endswith(".tmp")]
    assert leftovers == []


def test_record_engine_sha256_preserves_the_rich_record(tmp_path):
    eng = _write(tmp_path, "r.engine", b"payload")
    p = tmp_path / "MANIFEST.json"
    atomic_write_json(
        str(p),
        {
            "trt_version_required": "8.5.2.2",
            "engines": {
                "r": {
                    "file": "r.engine",
                    "license": "MIT",
                    "inputs": [{"name": "in", "shape": [1, 3, 8, 8]}],
                    "sha256": "00" * 32,
                    "bytes": 1,
                }
            },
        },
    )
    record_engine_sha256(str(eng), str(p))
    rec = json.loads(p.read_text())["engines"]["r"]
    assert rec["sha256"] == sha256_file(str(eng))
    assert rec["bytes"] == len(b"payload")
    assert rec["license"] == "MIT"
    assert rec["inputs"][0]["name"] == "in"


# ------------------------------------------------------------------- runtime


def test_runtime_env_reports_what_it_can():
    env = runtime_env()
    assert set(env) == {"trt_version", "cuda_version", "cuda_version_encoded", "l4t_release"}


def test_cuda_version_str_encoding():
    assert cuda_version_str(11040) == "11.4"
    assert cuda_version_str(None) is None


def test_l4t_release_parsing(tmp_path):
    p = tmp_path / "nv_tegra_release"
    p.write_text("# R35 (release), REVISION: 6.5, GCID: 1234, BOARD: t186ref\n")
    assert l4t_release(str(p)) == "35.6.5"
    assert l4t_release(str(tmp_path / "absent")) is None


def test_environment_pins_are_enforced_when_declared():
    pytest.importorskip("tensorrt")
    with pytest.raises(ManifestError) as ei:
        check_environment({"cuda_version_required": "99.9"})
    assert "cuda" in str(ei.value)
    with pytest.raises(ManifestError) as ei:
        check_environment({"l4t_release_required": "99.9.9"})
    assert "l4t" in str(ei.value)
    # Undeclared pins are skipped, not invented.
    check_environment({})


def test_check_runtime_skips_blank_paths_and_deduplicates():
    pytest.importorskip("tensorrt")
    face = os.path.join(ROOT, "engines", "scrfd_500m_640_fp16_gpu.engine")
    if not os.path.isfile(face):
        pytest.skip("engine not on disk")
    man = check_runtime([face, "", None, face], require_hash=True)
    assert man["trt_version_required"] == "8.5.2.2"
