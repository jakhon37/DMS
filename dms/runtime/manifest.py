"""Load-time MANIFEST enforcement: wrong TRT, wrong CUDA or a changed engine byte
must fail before the process notifies READY.

The MANIFEST at ``engines/MANIFEST.json`` is the trust anchor for every ``.engine``
plan on the board. This module answers one question - *is the runtime I am about to
start the runtime these plans were built for* - and raises :class:`ManifestError` with
one of two contractual prefixes when it is not:

``ENGINE_VERSION_MISMATCH``
    the manifest is missing/unreadable, or the running TensorRT / CUDA / L4T release
    differs from the one recorded at build time. A TensorRT plan is only valid for the
    exact TRT build plus a compatible cuDNN/driver, so a byte-identical engine file can
    still be unusable after a JetPack patch update - the sha256 check alone cannot see
    that, which is why the environment is pinned too.
``ENGINE_FAIL``
    a required engine is missing, unlisted, or its sha256 does not match.

Units: sizes are bytes, versions are the vendor strings (``8.5.2.2``) or the CUDA
integer encoding 1000*major + 10*minor (11040 == CUDA 11.4).

Failure behaviour: nothing here degrades silently. Optional checks (a manifest that
does not record ``cuda_version_required``) are skipped with a debug log; every check
that *can* run and fails raises.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("dms.manifest")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFEST = os.path.join(ROOT, "engines", "MANIFEST.json")
ENGINE_DIR = os.path.join(ROOT, "engines")

NV_TEGRA_RELEASE = "/etc/nv_tegra_release"


class ManifestError(RuntimeError):
    """The engine set on disk does not match this manifest or this runtime."""


def sha256_file(path):
    # type: (str) -> str
    """Streaming sha256 of a file, hex lowercase. Reads in 1 MiB chunks."""
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
    """Read and parse the manifest. Any read/parse failure is ENGINE_VERSION_MISMATCH."""
    p = path or DEFAULT_MANIFEST
    if not os.path.isfile(p):
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST missing: %s" % p)
    try:
        with open(p) as f:
            man = json.load(f)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError on py3.8
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST unreadable %s: %s" % (p, exc))
    except OSError as exc:
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST unreadable %s: %s" % (p, exc))
    if not isinstance(man, dict):
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST is not an object: %s" % p)
    return man


def atomic_write_json(path, obj):
    # type: (str, Dict[str, Any]) -> None
    """Write JSON so a power cut can never leave a truncated trust anchor behind."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(d):
        os.makedirs(d)
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    dirfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


# --------------------------------------------------------------------- runtime


def running_trt_version():
    # type: () -> str
    """TensorRT python version string, e.g. 8.5.2.2. Raises ManifestError if absent."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ManifestError("ENGINE_VERSION_MISMATCH tensorrt is not importable: %s" % exc)
    return str(trt.__version__)


def running_cuda_version():
    # type: () -> Optional[int]
    """CUDA runtime version as 1000*major + 10*minor, or None if libcudart is absent."""
    try:
        from dms.runtime import cudart

        return cudart.runtime_version()
    except Exception as exc:  # missing libcudart, no device, ctypes failure
        log.debug("cuda runtime version unavailable: %s", exc)
        return None


def cuda_version_str(encoded):
    # type: (Optional[int]) -> Optional[str]
    """11040 -> '11.4'. None passes through."""
    if not encoded:
        return None
    return "%d.%d" % (int(encoded) // 1000, (int(encoded) % 1000) // 10)


def l4t_release(path=NV_TEGRA_RELEASE):
    # type: (str) -> Optional[str]
    """Parse /etc/nv_tegra_release into 'R.revision' form, e.g. '35.6.5'. None off-board."""
    try:
        with open(path) as f:
            head = f.readline()
    except OSError:
        return None
    m = re.search(r"R(\d+)\s*\(release\)\s*,\s*REVISION:\s*([0-9.]+)", head)
    if not m:
        return None
    return "%s.%s" % (m.group(1), m.group(2).rstrip("."))


def runtime_env():
    # type: () -> Dict[str, Any]
    """Everything we can pin an engine plan against. Values may be None when unknown."""
    cuda = running_cuda_version()
    try:
        trt_v = running_trt_version()
    except ManifestError:
        trt_v = None
    return {
        "trt_version": trt_v,
        "cuda_version": cuda_version_str(cuda),
        "cuda_version_encoded": cuda,
        "l4t_release": l4t_release(),
    }


def _version_ok(have, need):
    # type: (Optional[str], str) -> bool
    """A recorded pin matches when it is the whole version or a dotted prefix of it."""
    if have is None:
        return False
    return have == need or have.startswith(need.rstrip(".") + ".")


def check_trt_version(manifest):
    # type: (Dict[str, Any]) -> None
    """Refuse to run plans built for a different TensorRT."""
    need = str(manifest.get("trt_version_required") or "")
    if not need:
        return
    have = running_trt_version()
    if have != need and not have.startswith(need):
        raise ManifestError("ENGINE_VERSION_MISMATCH need=%s have=%s" % (need, have))


def check_environment(manifest):
    # type: (Dict[str, Any]) -> None
    """Check every environment pin the manifest declares. Undeclared pins are skipped.

    Recognised keys: ``trt_version_required``, ``cuda_version_required`` (``'11.4'``),
    ``l4t_release_required`` (``'35.6.5'``).
    """
    check_trt_version(manifest)
    need_cuda = manifest.get("cuda_version_required")
    if need_cuda:
        have = cuda_version_str(running_cuda_version())
        if have is None:
            raise ManifestError(
                "ENGINE_VERSION_MISMATCH cuda_version_required=%s but libcudart is unusable" % need_cuda
            )
        if not _version_ok(have, str(need_cuda)):
            raise ManifestError(
                "ENGINE_VERSION_MISMATCH cuda need=%s have=%s" % (need_cuda, have)
            )
    else:
        log.debug("MANIFEST does not pin cuda_version_required")
    need_l4t = manifest.get("l4t_release_required")
    if need_l4t:
        have_l4t = l4t_release()
        if have_l4t is None:
            raise ManifestError(
                "ENGINE_VERSION_MISMATCH l4t_release_required=%s but %s is unreadable"
                % (need_l4t, NV_TEGRA_RELEASE)
            )
        if not _version_ok(have_l4t, str(need_l4t)):
            raise ManifestError(
                "ENGINE_VERSION_MISMATCH l4t need=%s have=%s" % (need_l4t, have_l4t)
            )
    else:
        log.debug("MANIFEST does not pin l4t_release_required")


# --------------------------------------------------------------------- records


def _record_paths(rec):
    # type: (Dict[str, Any]) -> Tuple[str, str]
    """(declared path as written, absolute path it resolves to)."""
    declared = str(rec.get("path") or rec.get("file") or "")
    if not declared:
        return "", ""
    if os.path.isabs(declared):
        return declared, os.path.abspath(declared)
    if os.sep in declared or "/" in declared:
        return declared, os.path.abspath(os.path.join(ROOT, declared))
    # A bare filename means engines/<file>.
    return declared, os.path.abspath(os.path.join(ENGINE_DIR, declared))


def _engine_record(manifest, engine_path):
    # type: (Dict[str, Any], str) -> Optional[Dict[str, Any]]
    """Resolve one engine path to its manifest record.

    Exact absolute-path matches win. A basename-only match is accepted when it is
    unique; two records sharing a basename raise rather than guessing.
    """
    engines = manifest.get("engines") or {}
    if not isinstance(engines, dict):
        raise ManifestError("ENGINE_VERSION_MISMATCH MANIFEST engines is not an object")
    want_abs = os.path.abspath(engine_path)
    want_base = os.path.basename(engine_path)
    exact = []  # type: List[Tuple[str, Dict[str, Any]]]
    by_base = []  # type: List[Tuple[str, Dict[str, Any]]]
    for key, rec in engines.items():
        if not isinstance(rec, dict):
            continue
        declared, rec_abs = _record_paths(rec)
        if not declared:
            continue
        if rec_abs == want_abs:
            exact.append((key, rec))
        elif os.path.basename(declared) == want_base:
            by_base.append((key, rec))
    if len(exact) > 1:
        raise ManifestError(
            "ENGINE_VERSION_MISMATCH %s matches %d MANIFEST records (%s)"
            % (engine_path, len(exact), ", ".join(k for k, _ in exact))
        )
    if exact:
        return exact[0][1]
    if len(by_base) > 1:
        raise ManifestError(
            "ENGINE_VERSION_MISMATCH basename %s is ambiguous across MANIFEST records (%s)"
            % (want_base, ", ".join(k for k, _ in by_base))
        )
    if by_base:
        return by_base[0][1]
    return None


def check_binding_contract(rec, engine_path):
    # type: (Dict[str, Any], str) -> None
    """If the record carries a binding contract, refuse -1 (dynamic) dimensions.

    The whole pipeline assumes static engines; a dynamic binding recorded in the
    manifest means the build silently froze or left a dimension open.
    """
    for side in ("inputs", "outputs"):
        entries = rec.get(side)
        if not entries:
            continue
        if isinstance(entries, dict):
            entries = list(entries.values())
        for ent in entries:
            shape = ent.get("shape") if isinstance(ent, dict) else None
            if not shape:
                continue
            if any(int(x) < 0 for x in shape):
                raise ManifestError(
                    "ENGINE_FAIL %s records a dynamic %s shape %s"
                    % (os.path.basename(engine_path), side[:-1], list(shape))
                )


def check_engine_file(manifest, engine_path, require_hash=False):
    # type: (Dict[str, Any], str, bool) -> None
    """Verify one engine exists and matches its manifest record.

    require_hash=True (production, ``require_engines: true``) makes an unlisted engine
    or a missing sha256 fatal. With require_hash=False those are warnings, but a
    recorded hash that does not match is always fatal.
    """
    if not os.path.isfile(engine_path):
        raise ManifestError("ENGINE_FAIL missing %s" % engine_path)
    rec = _engine_record(manifest, engine_path)
    if rec is None:
        if require_hash:
            raise ManifestError(
                "ENGINE_FAIL %s not listed in MANIFEST" % os.path.basename(engine_path)
            )
        log.warning("MANIFEST has no sha256 for %s", os.path.basename(engine_path))
        return
    check_binding_contract(rec, engine_path)
    expect_bytes = rec.get("bytes")
    if expect_bytes:
        actual = os.path.getsize(engine_path)
        if int(actual) != int(expect_bytes):
            raise ManifestError(
                "ENGINE_FAIL size mismatch %s expect=%s got=%s"
                % (os.path.basename(engine_path), expect_bytes, actual)
            )
    expect = rec.get("sha256")
    if not expect:
        if require_hash:
            raise ManifestError(
                "ENGINE_FAIL no sha256 for %s" % os.path.basename(engine_path)
            )
        log.warning("MANIFEST record for %s carries no sha256", os.path.basename(engine_path))
        return
    got = sha256_file(engine_path)
    if got.lower() != str(expect).lower():
        raise ManifestError(
            "ENGINE_FAIL sha256 mismatch %s expect=%s got=%s"
            % (os.path.basename(engine_path), expect, got)
        )


def check_runtime(engine_paths, manifest_path=None, require_hash=False, check_env=True):
    # type: (list, Optional[str], bool, bool) -> Dict[str, Any]
    """Full pre-READY gate: manifest parses, runtime matches, every engine verifies.

    ``engine_paths`` may contain None/'' entries (an unconfigured optional model);
    those are skipped. Returns the parsed manifest so the caller can log versions.
    """
    man = load_manifest(manifest_path)
    if check_env:
        check_environment(man)
    else:
        check_trt_version(man)
    seen = set()
    for p in engine_paths:
        if not p:
            continue
        key = os.path.abspath(p)
        if key in seen:
            continue
        seen.add(key)
        check_engine_file(man, p, require_hash=require_hash)
    return man


def record_engine_sha256(engine_path, manifest_path=None):
    # type: (str, Optional[str]) -> None
    """Update MANIFEST engines[].sha256/bytes after a successful build, atomically.

    Only sha256 and bytes are touched; every other field the builder wrote (bindings,
    licence, latency) is preserved.
    """
    p = manifest_path or DEFAULT_MANIFEST
    man = load_manifest(p)
    engines = man.setdefault("engines", {})
    base = os.path.basename(engine_path)
    digest = sha256_file(engine_path)
    key = os.path.splitext(base)[0]
    rec = engines.get(key) if isinstance(engines.get(key), dict) else None
    if rec is None:
        for k, v in list(engines.items()):
            if isinstance(v, dict) and os.path.basename(str(v.get("path") or v.get("file") or "")) == base:
                rec = v
                key = k
                break
    if rec is None:
        rec = {"file": base, "device": "gpu"}
        engines[key] = rec
    rec["file"] = base
    rec["sha256"] = digest
    rec["bytes"] = os.path.getsize(engine_path)
    atomic_write_json(p, man)
    log.info("MANIFEST updated %s sha256=%s", base, digest)
