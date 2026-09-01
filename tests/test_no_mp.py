from __future__ import annotations

import ast
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DMS = os.path.join(ROOT, "dms")
BANNED = {"multiprocessing", "multiprocess", "torch.multiprocessing"}
BANNED_NAMES = {"ProcessPoolExecutor"}
SUBPROCESS_OK = {"io"}


def _py_files(root: str):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_no_multiprocessing_under_dms():
    offenders = []
    for path in _py_files(DMS):
        rel = os.path.relpath(path, DMS)
        src = open(path, "r").read()
        tree = ast.parse(src, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BANNED or alias.name in BANNED:
                        offenders.append("%s: import %s" % (rel, alias.name))
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if node.module in BANNED or top in BANNED:
                    offenders.append("%s: from %s" % (rel, node.module))
                if node.module == "concurrent.futures":
                    for alias in node.names:
                        if alias.name in BANNED_NAMES:
                            offenders.append("%s: ProcessPoolExecutor" % rel)
            if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                pkg = rel.split(os.sep)[0]
                if pkg not in SUBPROCESS_OK:
                    offenders.append("%s: subprocess outside dms/io" % rel)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess":
                        pkg = rel.split(os.sep)[0]
                        if pkg not in SUBPROCESS_OK:
                            offenders.append("%s: subprocess outside dms/io" % rel)
    assert not offenders, "R15: " + "; ".join(offenders)


def test_requirements_allowlist():
    path = os.path.join(ROOT, "requirements.txt")
    lines = []
    for line in open(path):
        s = line.split("#", 1)[0].strip().lower()
        if s:
            lines.append(s)
    text = "\n".join(lines)
    for banned in ("opencv-python", "tensorflow", "torch", "onnxruntime", "uniface", "pycuda"):
        assert banned not in text, "R12 banned dep %s in requirements.txt" % banned
