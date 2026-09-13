"""Behavioural checks on the shipped deployment artefacts.

These used to be substring checks for `Type=notify` and `useradd`, which is why the
`GROUPS=video` bug (a silent no-op that put the service account in group root) sat in
a green suite. The assertions here are about properties an operator depends on:
the sandbox is on, the installer cannot half-install, and logrotate does what its
comment claims.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIT = os.path.join(ROOT, "deploy/dms.service")
SETUP = os.path.join(ROOT, "deploy/setup_jetson.sh")
LOGROTATE = os.path.join(ROOT, "deploy/dms.logrotate")
TMPFILES = os.path.join(ROOT, "deploy/dms-tmpfiles.conf")
DEPLOY_README = os.path.join(ROOT, "deploy/README.md")


def _text(path):
    with open(path) as f:
        return f.read()


def _directives(text):
    """{key: [values]} for every uncommented `Key=Value` line in a unit file."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, _sep, value = line.partition("=")
        out.setdefault(key.strip(), []).append(value.strip())
    return out


# ------------------------------------------------------------------- the unit


def test_dms_service_core_contract():
    text = _text(UNIT)
    d = _directives(text)
    assert d["Type"] == ["notify"]
    assert d["User"] == ["dms"]
    assert d["Group"] == ["dms"]
    assert d["SupplementaryGroups"] == ["video gpio"]
    assert d["StateDirectory"] == ["dms"]
    assert d["TimeoutStartSec"] == ["90"]
    assert d["WatchdogSec"] == ["30"]
    assert "--detect" in text
    assert "/etc/dms/default.yaml" in text
    # the unit must never change the board's power state
    assert "nvpmodel" not in text
    assert "jetson_clocks" not in text


def test_watchdog_is_shorter_than_the_start_timeout_and_the_stop_timeout_is_bounded():
    d = _directives(_text(UNIT))
    watchdog = int(d["WatchdogSec"][0])
    start = int(d["TimeoutStartSec"][0])
    stop = int(d["TimeoutStopSec"][0])
    # WatchdogSec only starts counting after READY=1, so it is independent of
    # TimeoutStartSec — but a watchdog longer than the start timeout would mean a
    # unit that hangs after READY is tolerated longer than one that never starts.
    assert watchdog < start
    assert 0 < stop <= 30
    assert int(d["RestartSec"][0]) >= 1


def test_dms_service_is_sandboxed():
    """OPS-02: the unit had no hardening at all."""
    d = _directives(_text(UNIT))
    for key, expected in (
        ("NoNewPrivileges", "true"),
        ("ProtectSystem", "strict"),
        ("ProtectHome", "true"),
        ("PrivateTmp", "true"),
        ("ProtectKernelTunables", "true"),
        ("ProtectKernelModules", "true"),
        ("ProtectControlGroups", "true"),
        ("RestrictSUIDSGID", "true"),
        ("RestrictNamespaces", "true"),
        ("RestrictRealtime", "true"),
        ("LockPersonality", "true"),
    ):
        assert d.get(key) == [expected], key
    assert d["CapabilityBoundingSet"] == [""]
    assert "AF_UNIX" in d["RestrictAddressFamilies"][0]


def test_writable_paths_are_only_the_state_locations():
    """OPS-03: the service must not be able to rewrite its own code or MANIFEST."""
    d = _directives(_text(UNIT))
    rw = set(d["ReadWritePaths"][0].split())
    assert rw == {"/var/lib/dms", "/run/dms", "/var/log/dms"}
    assert "/opt/dms" not in rw
    assert "/etc/dms" not in rw


def test_memory_is_capped_so_the_board_is_not_oom_killed():
    d = _directives(_text(UNIT))
    assert d["MemoryMax"] == ["5G"]
    assert d["OOMPolicy"] == ["stop"]
    assert d["TasksMax"] == ["256"]
    # A core dump of this process contains frames of the driver's face.
    assert d["LimitCORE"] == ["0"]


def test_state_directory_mode_keeps_the_alert_history_private():
    d = _directives(_text(UNIT))
    assert d["StateDirectoryMode"] == ["0750"]
    assert d["RuntimeDirectoryMode"] == ["0750"]
    assert d["UMask"] == ["0027"]


def test_restart_policy_escalates_a_crash_loop():
    text = _text(UNIT)
    d = _directives(text)
    # `on-failure` let a clean exit(0) end driver monitoring silently.
    assert d["Restart"] == ["always"]
    assert d["StartLimitBurst"] == ["5"]
    assert d["StartLimitIntervalSec"] == ["300"]
    # StartLimit* belong in [Unit] on systemd >= 229; in [Service] they are ignored.
    unit_section = text.split("[Service]")[0]
    assert "StartLimitBurst" in unit_section


def test_directives_that_would_break_cuda_are_not_enabled():
    d = _directives(_text(UNIT))
    # TensorRT/CUDA JIT needs W+X; PrivateDevices hides /dev/nvhost-* and /dev/nvmap.
    assert "MemoryDenyWriteExecute" not in d
    assert "PrivateDevices" not in d
    # The untested device/syscall allowlists must stay commented out until a board
    # boot-test validates them (deploy/README.md documents the procedure).
    assert "DevicePolicy" not in d
    assert "SystemCallFilter" not in d
    text = _text(UNIT)
    assert "#DevicePolicy=closed" in text
    assert "#SystemCallFilter=@system-service" in text
    assert "Validating the tight sandbox" in _text(DEPLOY_README)


def test_unit_points_at_the_release_symlink_so_rollback_is_possible():
    text = _text(UNIT)
    assert "WorkingDirectory=/opt/dms/current" in text
    assert "ExecStart=/opt/dms/current/.venv/bin/python" in text


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze absent")
def test_systemd_analyze_accepts_the_unit():
    proc = subprocess.run(
        ["systemd-analyze", "verify", UNIT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    out = proc.stdout.decode("utf-8", "replace")
    # A missing ExecStart binary is expected off a deployed unit; a syntax error or an
    # unknown directive is not.
    for bad in ("Unknown lvalue", "Unknown section", "Failed to parse", "Invalid"):
        assert bad not in out, out


# --------------------------------------------------------------- the installer


def test_setup_jetson_script_syntax():
    subprocess.check_call(["bash", "-n", SETUP])
    text = _text(SETUP)
    assert "--apply-power" in text
    assert "useradd" in text
    assert "configs/production.yaml" in text
    assert "configs/systemd-lab.yaml" in text
    assert "--lab" in text
    assert "/etc/dms/default.yaml" in text
    assert "jetson_clocks is never run" in text.lower() or "never auto" in text.lower()


def test_setup_never_assigns_to_the_bash_special_variable_GROUPS():
    """OPS-01: `GROUPS=video` is silently ignored, so useradd got --groups 0 (root)."""
    text = _text(SETUP)
    assert not re.search(r"(?m)^\s*GROUPS=", text)
    assert not re.search(r"(?m)^\s*(export\s+)?GROUPS=", text)
    assert "EXTRA_GROUPS=video" in text
    assert '--groups "$EXTRA_GROUPS"' in text
    # and it verifies the result rather than assuming it
    assert "id -nG" in text


def test_bash_really_ignores_an_assignment_to_GROUPS():
    """The premise of the test above, asserted rather than assumed."""
    out = subprocess.check_output(
        ["bash", "-c", 'GROUPS=video; printf "%s" "$GROUPS"'],
        stderr=subprocess.STDOUT,
    ).decode()
    assert out != "video"


def test_setup_refuses_to_deploy_private_data_and_the_dev_venv():
    """OPS-12: the old rsync copied .venv, lab events and ~11 MB of driver faces."""
    text = _text(SETUP)
    assert "vehicle.yaml" in text  # excluded from configs/
    for token in ("copy_dir \"$REPO/dms\"", "copy_dir \"$REPO/configs\""):
        assert token in text
    # an allow-list, not a deny-list: tests/, data/, docs/, tools/ are never named
    # as things to copy
    assert 'copy_dir "$REPO/tests"' not in text
    assert 'copy_dir "$REPO/data"' not in text
    assert 'copy_dir "$REPO/docs"' not in text
    assert "--delete" in text


def test_setup_stages_a_release_and_swaps_a_symlink():
    text = _text(SETUP)
    assert "releases" in text
    assert "current" in text
    assert "ln -sfn" in text
    assert "systemctl stop" in text
    assert "rollback" in text.lower()


def test_setup_makes_code_and_config_root_owned():
    """OPS-03: chown -R dms:dms /opt/dms let the process rewrite its own MANIFEST."""
    text = _text(SETUP)
    assert "chown -R root:root" in text
    assert 'chown root:root "$CONF_DIR/default.yaml"' in text
    assert "chown -R dms:dms" not in text
    assert 'install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$STATE_DIR"' in text


def test_setup_always_installs_dependencies():
    """A guard on the venv's existence meant an upgrade never applied requirements."""
    text = _text(SETUP)
    assert 'if [ ! -x "$PREFIX/.venv/bin/python" ]' not in text
    assert "requirements.txt" in text
    assert "--require-hashes" in text  # used when requirements.lock exists


def test_setup_records_the_release_for_healthz():
    text = _text(SETUP)
    assert "git_sha=" in text
    assert "$CONF_DIR/release" in text or "RELEASE_FILE" in text


def test_setup_substitutes_the_prefix_into_the_unit():
    """--prefix used to be accepted and then ignored, producing an unstartable unit."""
    text = _text(SETUP)
    assert 's|/opt/dms|$PREFIX|g' in text


def test_setup_dry_run_changes_nothing_and_exits_zero(tmp_path):
    proc = subprocess.run(
        ["bash", SETUP, "--dry-run", "--vehicle", "--prefix", str(tmp_path / "opt-dms")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, out
    assert "DRY RUN" in out
    assert "[dry ]" in out
    assert not os.path.exists(str(tmp_path / "opt-dms"))
    # every mutating step was printed, not run
    assert "useradd" in out or "usermod" in out
    assert "systemctl daemon-reload" in out


def test_setup_rejects_an_unknown_argument():
    proc = subprocess.run(
        ["bash", SETUP, "--dry-run", "--not-a-flag"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    assert proc.returncode == 2


def test_setup_rejects_a_relative_prefix():
    proc = subprocess.run(
        ["bash", SETUP, "--dry-run", "--prefix", "relative/dir"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    assert proc.returncode != 0
    assert b"absolute" in proc.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_is_clean():
    proc = subprocess.run(
        ["shellcheck", "--severity=warning", SETUP],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------- logrotate


def test_logrotate_jsonl():
    text = _text(LOGROTATE)
    assert "events.jsonl" in text
    assert "50M" in text


def test_logrotate_uses_maxsize_not_bare_size():
    """OPS-13: bare `size` disables `daily`; `rotate 14` then meant 14x50 MB, no age bound."""
    text = _text(LOGROTATE)
    assert re.search(r"(?m)^\s*maxsize\s+50M\s*$", text)
    assert not re.search(r"(?m)^\s*size\s+\d+", text)
    assert re.search(r"(?m)^\s*daily\s*$", text)
    assert re.search(r"(?m)^\s*maxage\s+\d+", text)


def test_logrotate_does_not_lose_lines_and_owns_its_files():
    text = _text(LOGROTATE)
    assert re.search(r"(?m)^\s*create\s+0640\s+dms\s+dms\s*$", text)
    assert re.search(r"(?m)^\s*su\s+dms\s+dms\s*$", text)
    # copytruncate loses every line written between the copy and the truncate
    assert not re.search(r"(?m)^\s*copytruncate\s*$", text)
    assert "SIGHUP" in text


def test_logrotate_covers_the_fallback_path_too():
    """OPS-05: the app can fall back to /run/dms/events.jsonl."""
    text = _text(LOGROTATE)
    assert "/var/lib/dms/events.jsonl" in text
    assert "/run/dms/events.jsonl" in text


@pytest.mark.skipif(shutil.which("logrotate") is None, reason="logrotate not installed")
def test_logrotate_config_parses():
    proc = subprocess.run(
        ["logrotate", "-d", LOGROTATE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert "error:" not in out.lower(), out


# ----------------------------------------------------------------- tmpfiles


def test_tmpfiles_locks_down_the_state_directory():
    text = _text(TMPFILES)
    assert re.search(r"(?m)^d\s+/var/lib/dms\s+0750\s+dms\s+dms", text)
    assert re.search(r"(?m)^d\s+/var/log/dms\s+0750\s+dms\s+dms", text)
    assert re.search(r"(?m)^d\s+/run/dms\s+0750\s+dms\s+dms", text)


def test_tmpfiles_does_not_pre_create_the_clip_directory():
    """A default install must not even hint that in-cabin video is recorded."""
    text = _text(TMPFILES)
    assert not re.search(r"(?m)^d\s+/var/lib/dms/clips", text)


# ------------------------------------------------------------------- runbook


def test_deploy_readme_documents_the_operational_contract():
    text = _text(DEPLOY_README)
    for token in (
        "/livez",
        "/readyz",
        "/healthz",
        "/metrics",
        "rollback",
        "dms_build_info",
        "privacy.record_faces",
        "requirements.lock",
    ):
        assert token in text, token


# ------------------------------------------- runtime contracts an operator greps
#
# These are deployment properties even though they live in dms/app.py: they are
# what a deploy script, a journald alarm and a Prometheus scrape actually read.
# Each one is a regression guard for a defect that shipped.


def _default_cfg():
    from dms.config.schema import load_config

    return load_config(os.path.join(ROOT, "configs", "default.yaml"))


def test_engine_provenance_covers_every_engine_the_app_loads():
    """All nine engines are hashed before READY, not just the two loaded by hand.

    The shipped app verified ``[face, landmarks]`` under a comment claiming the
    check was unconditional, while ModelRegistry loaded seven more - head pose,
    both eye/mouth classifiers, both occlusion gates, YOLOX and SFace - with no
    hash check at all.
    """
    from dms.app import engines_to_verify
    from dms.infer.registry import BACKEND_MODEL, CAPABILITIES, ModelRegistry

    cfg = _default_cfg()
    verified = set(os.path.basename(p) for p in engines_to_verify(cfg))
    assert os.path.basename(cfg.models.face.engine) in verified
    assert os.path.basename(cfg.models.landmarks.engine) in verified

    reg = ModelRegistry(
        backends=cfg.models.backend_map(),
        capability_cfg=cfg.models.capability_cfg(),
        research_models_allowed=cfg.models.research_models_allowed,
    )
    expected = set()
    for cap, spec in CAPABILITIES.items():
        if reg.backends[cap] != BACKEND_MODEL:
            continue
        if spec.licence_gated and not reg.research_models_allowed:
            continue
        for role in spec.roles:
            path = reg.role_path(role)
            if path is not None:
                expected.add(os.path.basename(path))
                if spec.alternatives:
                    break
    missing = expected - verified
    assert not missing, "engines loaded but never hash-verified: %s" % sorted(missing)
    assert len(verified) >= 9, verified


def test_licence_gated_engine_is_not_pulled_into_the_verification_list():
    """gaze is off by default: verifying it would demand a plan we must not load."""
    from dms.app import engines_to_verify

    cfg = _default_cfg()
    names = [os.path.basename(p) for p in engines_to_verify(cfg)]
    assert not any("gaze" in n for n in names), names


def test_ready_line_never_advertises_a_health_endpoint_that_did_not_bind():
    """A refused bind must read as disabled, not as a live endpoint.

    ``HealthServer.start()`` returns None when the port is taken; the READY line
    printed the configured bind:port regardless, so a deploy script probing the
    advertised endpoint got a connection refused it could not explain.
    """
    from dms.app import health_endpoint_label

    assert health_endpoint_label("127.0.0.1", 8088, True) == "127.0.0.1:8088"
    assert health_endpoint_label("127.0.0.1", None, True) == "disabled(bind refused)"
    assert health_endpoint_label("127.0.0.1", None, False) == "off"


def test_a_missing_engine_is_not_reported_as_a_version_mismatch():
    """Each manifest failure keeps its own code; nothing is double-prefixed."""
    from dms.app import manifest_error_text
    from dms.runtime.manifest import ManifestError

    missing = manifest_error_text(ManifestError("ENGINE_FAIL missing engines/x.engine"))
    assert missing == "ENGINE_FAIL missing engines/x.engine"
    assert "ENGINE_VERSION_MISMATCH" not in missing

    mismatch = manifest_error_text(ManifestError("ENGINE_VERSION_MISMATCH need=8.5 have=8.6"))
    assert mismatch == "ENGINE_VERSION_MISMATCH need=8.5 have=8.6"
    assert mismatch.count("ENGINE_VERSION_MISMATCH") == 1

    # An OSError carries no code of its own, so one is supplied - once.
    oserr = manifest_error_text(OSError("permission denied"))
    assert oserr == "ENGINE_FAIL permission denied"


def test_a_swapped_engine_is_fatal_even_when_require_engines_is_false():
    """A hash/size mismatch is a swapped plan, not missing metadata.

    Proven on the board: one flipped byte in mwc_n_48_fp16_gpu.engine reached
    READY under the shipped profile (require_engines: false) and then took the
    process down with CUDA error 700 and SIGABRT. Absent or unlisted engines are
    a softer failure and stay governed by require_engines.
    """
    from dms.app import engine_swap_detected

    assert engine_swap_detected("ENGINE_FAIL sha256 mismatch mwc_n_48.engine expect=a got=b")
    assert engine_swap_detected("ENGINE_FAIL size mismatch mwc_n_48.engine expect=1 got=2")
    assert not engine_swap_detected("ENGINE_FAIL missing engines/x.engine")
    assert not engine_swap_detected("ENGINE_FAIL x.engine not listed in MANIFEST")
    assert not engine_swap_detected("ENGINE_VERSION_MISMATCH need=8.5 have=8.6")


def _gate_app(e2e_ms, limit_ms=66.0):
    """A DmsApp with nothing but a LatencyBudget holding synthetic e2e samples."""
    from dms import app as dms_app
    from dms.perf.budget import MIN_SAMPLES, LatencyBudget

    app = dms_app.DmsApp.__new__(dms_app.DmsApp)
    app.budget = LatencyBudget(gate_e2e_p95_ms=limit_ms, label="test")
    for _ in range(MIN_SAMPLES + 5):
        app.budget.record("e2e", e2e_ms)
    app.budget.observe_rss(1000)
    app._gate_last = {}
    app._gate_fail_streak = 0
    app._gate_degraded = False
    return app


def test_the_latency_gate_is_evaluated_without_the_budget_env_var():
    """The gate is computed on the normal path, and published to /metrics.

    It used to be computed only inside ``_write_budget()``, which returns
    immediately unless ``DMS_LATENCY_BUDGET_OUT`` is set - so a unit running
    12 ms over its own limit reported ``ok: true, degraded: false`` and no
    metric carried a verdict.
    """
    from dms import app as dms_app

    assert "DMS_LATENCY_BUDGET_OUT" not in os.environ
    dms_app.PERF_GATE_PASS.clear()
    app = _gate_app(80.0)
    gate, degraded = app._evaluate_gate()

    assert gate["e2e_p95_ms"] == pytest.approx(80.0, abs=0.5)
    assert gate["pass"] is False
    assert not degraded, "one failing refresh must not flap the degraded flag"
    assert dict(dms_app.PERF_GATE_PASS.samples())[()] == 0.0
    assert dict(dms_app.PERF_GATE_E2E_P95_MS.samples())[()] == pytest.approx(80.0, abs=0.5)
    assert dict(dms_app.PERF_GATE_E2E_P95_LIMIT_MS.samples())[()] == 66.0


def test_a_sustained_gate_miss_makes_the_unit_degraded_and_a_pass_does_not():
    from dms import app as dms_app

    app = _gate_app(80.0)
    verdicts = [app._evaluate_gate()[1] for _ in range(dms_app.GATE_DEGRADE_REFRESHES)]
    assert verdicts[-1] is True, "a sustained miss must set degraded"
    assert verdicts[0] is False

    ok = _gate_app(40.0)
    assert not any(ok._evaluate_gate()[1] for _ in range(dms_app.GATE_DEGRADE_REFRESHES + 5))
    assert ok._gate_last["pass"] is True
    assert dict(dms_app.PERF_GATE_PASS.samples())[()] == 1.0


def test_an_undecided_window_is_neither_a_pass_nor_a_fail():
    """Fewer samples than the budget's minimum is not evidence of anything."""
    from dms import app as dms_app
    from dms.perf.budget import LatencyBudget

    app = dms_app.DmsApp.__new__(dms_app.DmsApp)
    app.budget = LatencyBudget(gate_e2e_p95_ms=66.0)
    app.budget.record("e2e", 200.0)
    app.budget.observe_rss(1000)
    app._gate_last, app._gate_fail_streak, app._gate_degraded = {}, 0, False
    gate, degraded = app._evaluate_gate()
    assert degraded is False
    assert app._gate_fail_streak == 0
    assert "undecided" in dms_app._gate_note(gate, degraded)


def test_the_gate_verdict_is_logged_at_shutdown(caplog):
    """PERF_GATE PASS/FAIL is on every run's log, artifact or not."""
    import logging

    class _Args(object):
        config = "configs/default.yaml"

    app = _gate_app(80.0)
    app.args = _Args()
    with caplog.at_level(logging.INFO, logger="dms.app"):
        app._log_gate_verdict()
    lines = [r.getMessage() for r in caplog.records if "PERF_GATE" in r.getMessage()]
    assert len(lines) == 1, caplog.text
    assert lines[0].startswith("PERF_GATE FAIL")
    assert "limit=66.0 ms" in lines[0]

    ok = _gate_app(40.0)
    ok.args = _Args()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="dms.app"):
        ok._log_gate_verdict()
    assert "PERF_GATE PASS" in caplog.text


def test_the_shipped_letterbox_is_the_one_in_the_frame_loop():
    """dms.runtime.preprocess.Letterbox is what the detector stage actually runs.

    It was a 349-line exported module that only ``scripts/bench.py`` imported,
    while the application ran the allocating chain in
    ``ScrfdDetector.preprocess`` - so the benchmark measured a preprocessing
    stage the product did not have.  Both halves are asserted here: the wiring,
    and that the two implementations agree numerically.
    """
    import inspect

    import numpy as np

    from dms.app import DmsApp
    from dms.infer.scrfd import resize_letterbox
    from dms.runtime.preprocess import Letterbox

    source = inspect.getsource(DmsApp._perceive)
    assert "self.letterbox.blob(bgr)" in source, source[:400]
    assert 'budget.stage("letterbox")' in source

    frame = np.random.default_rng(7).integers(0, 255, (720, 1280, 3)).astype(np.uint8)
    for size in ((640, 640), (512, 512)):
        canvas, factor = resize_letterbox(frame, size)
        shipped = np.ascontiguousarray(
            np.transpose((canvas.astype(np.float32) - 127.5) / 127.5, (2, 0, 1))[None, ...]
        )
        blob, geom = Letterbox.for_scrfd((size[1], size[0])).blob(frame)
        assert blob.shape == shipped.shape
        assert geom.factor == factor
        assert float(np.abs(blob - shipped).max()) < 1e-5


def test_clocks_low_is_not_judged_on_the_first_frame():
    """The Tegra governor idles at 114 MHz; CLOCKS_LOW fired on 100% of runs."""
    import inspect

    import dms.app as dms_app

    assert dms_app.CLOCKS_SETTLE_FRAMES >= 30
    assert dms_app.CLOCKS_SETTLE_SAMPLES >= 20
    source = inspect.getsource(dms_app.DmsApp._refresh_health)
    assert "self.frames >= CLOCKS_SETTLE_FRAMES" in source
    assert "self.gpu_clock_samples >= CLOCKS_SETTLE_SAMPLES" in source
    assert "self.gpu_clock_max_mhz" in source
