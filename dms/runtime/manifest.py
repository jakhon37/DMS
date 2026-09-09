"""Load-time MANIFEST checks. Wrong TRT or engine hash → fail before READY."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("dms.manifest")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFEST = os.path.join(ROOT, "engines", "MANIFEST.json")


class ManifestError(RuntimeError):
    pass


def sha256_file(path):
    # type: (str) -> str
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path=None):
    # type: (Optional[str]) -> Dict[str, Any]
    p = path or DEFAULT_MANIFEST
    if not os.path.isfile(p):
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST missing: %s" % p)
    with open(p) as f:
        return json.load(f)


def running_trt_version():
    # type: () -> str
    import tensorrt as trt

    return str(trt.__version__)


def check_trt_version(manifest):
    # type: (Dict[str, Any]) -> None
    need = str(manifest.get("trt_version_required") or "")
    have = running_trt_version()
    if not need:
        return
    if have != need and not have.startswith(need):
        raise ManifestError("ENGINE_VERSION_MISMATCH need=%s have=%s" % (need, have))


def _engine_record(manifest, engine_path):
    # type: (Dict[str, Any], str) -> Optional[Dict[str, Any]]
    base = os.path.basename(engine_path)
    engines = manifest.get("engines") or {}
    for rec in engines.values():
        if isinstance(rec, dict) and rec.get("file") == base:
            return rec
    return None


def check_engine_file(manifest, engine_path, require_hash=False):
    # type: (Dict[str, Any], str, bool) -> None
    if not os.path.isfile(engine_path):
        raise ManifestError("ENGINE_FAIL missing %s" % engine_path)
    rec = _engine_record(manifest, engine_path)
    if rec is None:
        if require_hash:
            raise ManifestError("ENGINE_FAIL %s not listed in MANIFEST" % os.path.basename(engine_path))
        log.warning("MANIFEST has no sha256 for %s", os.path.basename(engine_path))
        return
    expect = rec.get("sha256")
    if not expect:
        if require_hash:
            raise ManifestError("ENGINE_FAIL no sha256 for %s" % os.path.basename(engine_path))
        return
    got = sha256_file(engine_path)
    if got.lower() != str(expect).lower():
        raise ManifestError(
            "ENGINE_FAIL sha256 mismatch %s expect=%s got=%s"
            % (os.path.basename(engine_path), expect, got)
        )


def check_runtime(engine_paths, manifest_path=None, require_hash=False):
    # type: (list, Optional[str], bool) -> Dict[str, Any]
    man = load_manifest(manifest_path)
    check_trt_version(man)
    for p in engine_paths:
        if p:
            check_engine_file(man, p, require_hash=require_hash)
    return man


def record_engine_sha256(engine_path, manifest_path=None):
    # type: (str, Optional[str]) -> None
    """Update MANIFEST engines[].sha256 after a successful trtexec build."""
    p = manifest_path or DEFAULT_MANIFEST
    man = load_manifest(p)
    engines = man.setdefault("engines", {})
    base = os.path.basename(engine_path)
    digest = sha256_file(engine_path)
    key = os.path.splitext(base)[0]
    rec = engines.get(key) if isinstance(engines.get(key), dict) else None
    if rec is None:
        for k, v in list(engines.items()):
            if isinstance(v, dict) and v.get("file") == base:
                rec = v
                key = k
                break
    if rec is None:
        rec = {"file": base, "device": "gpu"}
        engines[key] = rec
    rec["file"] = base
    rec["sha256"] = digest
    rec["bytes"] = os.path.getsize(engine_path)
    with open(p, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    log.info("MANIFEST updated %s sha256=%s", base, digest)
