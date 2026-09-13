"""Configuration contract tests.

The schema is a safety device: it is the only thing that stops a typo in a
drowsiness threshold from being silently ignored, a research-licensed model from
being enabled in a vehicle, or a health endpoint from publishing live driver
state to a network.  These tests assert those refusals, not just that the files
parse.
"""
from __future__ import annotations

import glob
import os
import sys

import pytest

from dms.config.schema import (
    CAPABILITIES,
    VALID_BACKENDS,
    AppConfig,
    calibration_path,
    is_loopback,
    load_config,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT, "configs")

#: Every profile except the per-unit calibration overlay, which is a fragment.
PROFILES = sorted(
    p
    for p in glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
    if os.path.basename(p) != "vehicle.yaml"
)

MINIMAL = (
    "source: {type: file, path: %s}\n"
    "models:\n"
    "  face: {engine: e}\n"
    "  landmarks: {engine: e}\n"
    "calibration: ''\n"
) % os.path.join(ROOT, "tests", "replay", "day_driver.mp4")


def write(tmp_path, body, name="c.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


# ------------------------------------------------------------------ inventory


def test_profiles_exist():
    assert len(PROFILES) >= 10, "expected the replay profiles plus default/production"
    names = set(os.path.basename(p) for p in PROFILES)
    assert {"default.yaml", "production.yaml", "testsrc.yaml", "systemd-lab.yaml"} <= names


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_every_profile_loads(path):
    cfg = load_config(path)
    assert cfg.models.face.engine
    assert cfg.models.landmarks.engine
    assert set(cfg.models.backend_map()) == set(CAPABILITIES)
    assert set(cfg.models.capability_cfg()) == set(CAPABILITIES)


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_every_profile_is_on_the_corrected_ear_scale(path):
    """The old thresholds latched MICROSLEEP CRITICAL on every frame.

    They were computed against a landmark table that put "EAR" on the mouth.
    A profile carrying them is a config that drives the GPIO line permanently.
    """
    from dms.geometry.ear import EAR_CLOSED_RECOMMENDED, EAR_OPEN_MEDIAN_RECOMMENDED
    from dms.geometry.mar import MAR_YAWN_RECOMMENDED

    cfg = load_config(path)
    assert cfg.state.ear_closed == pytest.approx(EAR_CLOSED_RECOMMENDED)
    assert cfg.state.ear_open_median == pytest.approx(EAR_OPEN_MEDIAN_RECOMMENDED)
    assert cfg.state.mar_yawn == pytest.approx(MAR_YAWN_RECOMMENDED)


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_no_profile_enables_research_weights(path):
    """Gaze360 weights are research / non-commercial only."""
    cfg = load_config(path)
    assert cfg.models.research_models_allowed is False
    assert cfg.models.backends.gaze == "off"


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_no_profile_declares_itself_calibrated(path):
    """No shipped profile may claim a calibration that was never performed."""
    cfg = load_config(path)
    assert not cfg.forward_zero_calibrated, path
    assert (cfg.forward_zero.yaw, cfg.forward_zero.pitch, cfg.forward_zero.roll) == (
        0.0,
        0.0,
        0.0,
    ), path


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_no_profile_leaks_imagery_by_default(path):
    cfg = load_config(path)
    assert cfg.privacy.record_faces is False
    assert cfg.privacy.telemetry is False
    assert cfg.clips.enabled is False
    assert cfg.display.enabled is False
    assert is_loopback(cfg.health.bind)


# ------------------------------------------------------------------- profiles


def test_default_yaml_is_the_replay_profile():
    cfg = load_config(os.path.join(CONFIG_DIR, "default.yaml"))
    assert cfg.source.type == "file"
    assert cfg.source.dev is True
    assert cfg.source.width == 1280
    assert cfg.require_engines is False
    assert cfg.is_replay is True


def test_production_yaml_is_a_real_vehicle_config():
    """The verified defect this replaces: production carried the LAB camera policy.

    ``camera.fail_fatal: false`` used to mean "notify READY, ping the watchdog
    for ever, never retry".  It now means "retry on the backoff ladder while
    reporting failure", which is only correct because the retry policy is
    explicit and the watchdog is gated on frames arriving.
    """
    cfg = load_config(os.path.join(CONFIG_DIR, "production.yaml"))
    assert cfg.source.type == "csi"
    assert cfg.source.dev is False
    assert cfg.is_replay is False
    # engine provenance and durability
    assert cfg.require_engines is True
    assert cfg.events.enabled is True
    assert cfg.events.path == "/var/lib/dms/events.jsonl"
    assert cfg.events.strict is True
    assert cfg.events.fsync == "critical"
    # the camera policy
    assert cfg.camera.fail_fatal is False
    assert cfg.camera.retry_forever is True
    assert cfg.camera.backoff_s == [0.5, 1.0, 2.0, 5.0, 5.0]
    assert cfg.camera.starved_below_fps > 0.0, "production must detect a starved pipeline"
    # the health endpoint is not optional on a vehicle
    assert cfg.health.required is True
    assert is_loopback(cfg.health.bind)
    # honest calibration state: unset means "infer", and 0/0/0 infers false
    assert not cfg.forward_zero_calibrated
    assert (cfg.forward_zero.yaw, cfg.forward_zero.pitch, cfg.forward_zero.roll) == (
        0.0,
        0.0,
        0.0,
    )


#: The detector every shipped profile must select.  Pinned, not inferred: on
#: 2026-09-13 all twelve profiles were repointed at the 512 engine to buy
#: latency, it did not reliably buy the pass, and it cost 15.8% of the
#: driver-ROI face hits on the flagship clip.  See docs/performance.md s.7.
SHIPPED_FACE_ENGINE = "engines/scrfd_500m_640_fp16_gpu.engine"

#: The one profile that is allowed to differ, and what it is allowed to be.
LOW_LATENCY_PROFILE = "lowlatency.yaml"
LOW_LATENCY_FACE_ENGINE = "engines/scrfd_500m_512_fp16_gpu.engine"


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_only_the_low_latency_profile_ships_a_different_detector(path):
    """One documented exception, and it must be the documented one.

    Without this, "make the gate pass" is one line of YAML in twelve files away
    at any time, and the golden bands - which are recorded per detector - stop
    meaning anything.  If a new detector is genuinely wanted, change this
    constant deliberately and re-record the goldens in the same change.
    """
    cfg = load_config(path)
    name = os.path.basename(path)
    if name == LOW_LATENCY_PROFILE:
        assert cfg.models.face.engine == LOW_LATENCY_FACE_ENGINE, name
    else:
        assert cfg.models.face.engine == SHIPPED_FACE_ENGINE, (
            "%s selects %s; only %s may differ from %s"
            % (name, cfg.models.face.engine, LOW_LATENCY_PROFILE, SHIPPED_FACE_ENGINE)
        )


def test_the_low_latency_profile_exists_and_is_only_a_detector_swap():
    """``lowlatency.yaml`` must stay a one-knob variant of ``default.yaml``.

    It is kept so the 512 operating point remains available and measurable
    after the revert.  If it drifts into a second set of thresholds, comparing
    the two stops measuring the detector and starts measuring the drift.
    """
    lo = os.path.join(CONFIG_DIR, LOW_LATENCY_PROFILE)
    assert os.path.isfile(lo), "the documented low-latency profile is missing"
    a = load_config(os.path.join(CONFIG_DIR, "default.yaml")).dict()
    b = load_config(lo).dict()
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert differing == ["capture", "models"], differing
    assert a["capture"]["letterbox"] == (640, 640)
    assert b["capture"]["letterbox"] == (512, 512)
    a["capture"]["letterbox"] = b["capture"]["letterbox"] = None
    a["models"]["face"]["engine"] = b["models"]["face"]["engine"] = None
    assert a["capture"] == b["capture"]
    assert a["models"] == b["models"]


def test_lab_profiles_use_a_synthetic_source():
    for name in ("testsrc.yaml", "systemd-lab.yaml"):
        cfg = load_config(os.path.join(CONFIG_DIR, name))
        assert cfg.source.type == "test", name


def test_multi_occupant_profiles_do_not_promote_a_passenger():
    for name in ("minivan.yaml", "family_vacation_crash.yaml", "red_car_crash.yaml"):
        cfg = load_config(os.path.join(CONFIG_DIR, name))
        assert cfg.driver_fallback is False, name


# ------------------------------------------------------------------ refusals


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, "boom: 1\n" + MINIMAL))
    assert "boom" in str(exc.value)


def test_unknown_nested_key_rejected(tmp_path):
    """A typo inside a section used to be silently ignored."""
    body = MINIMAL + "state:\n  ear_clsoed: 0.2\n"
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "ear_clsoed" in str(exc.value)


def test_old_ear_scale_rejected(tmp_path):
    body = MINIMAL + "state:\n  ear_closed: 0.50\n  ear_open_median: 0.75\n"
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "OLD broken-landmark EAR scale" in str(exc.value)


def test_ear_closed_above_open_rejected(tmp_path):
    body = MINIMAL + "state:\n  ear_closed: 0.30\n  ear_open_median: 0.27\n"
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "ear_closed" in str(exc.value)


def test_bad_source_type(tmp_path):
    body = "source: {type: webcam}\nmodels:\n  face: {engine: e}\n  landmarks: {engine: e}\n"
    with pytest.raises(ValueError):
        load_config(write(tmp_path, body))


def test_gaze_model_requires_the_research_flag(tmp_path):
    body = MINIMAL + "models:\n  backends:\n    gaze: model\n"
    # the nested `models` key would replace the mapping above, so build it whole
    body = (
        "source: {type: test}\n"
        "models:\n"
        "  face: {engine: e}\n"
        "  landmarks: {engine: e}\n"
        "  backends: {gaze: model}\n"
        "calibration: ''\n"
    )
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "research_models_allowed" in str(exc.value)


def test_yaml_off_boolean_gets_a_useful_message(tmp_path):
    """YAML 1.1 turns a bare ``off`` into False; the error must say so."""
    body = (
        "source: {type: test}\n"
        "models:\n"
        "  face: {engine: e}\n"
        "  landmarks: {engine: e}\n"
        "  backends: {objects: off}\n"
        "calibration: ''\n"
    )
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "quoted" in str(exc.value)


def test_unknown_backend_rejected(tmp_path):
    body = (
        "source: {type: test}\n"
        "models:\n"
        "  face: {engine: e}\n"
        "  landmarks: {engine: e}\n"
        "  backends: {headpose: sometimes}\n"
        "calibration: ''\n"
    )
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "sometimes" in str(exc.value)
    for good in VALID_BACKENDS:
        assert good in str(exc.value)


def test_imagery_switches_are_enforced_at_runtime_not_in_the_schema(tmp_path):
    """The schema validates values; the app and the clip ring enforce policy.

    ``clips.enabled`` without ``privacy.record_faces`` must PARSE, because the
    clip ring is the enforcement point and refusing here would make its own
    defensive gate untestable.  ``display.enabled`` without it is refused by
    ``dms.app.main`` at startup instead - see test_app_integration.py.
    """
    cfg = load_config(write(tmp_path, MINIMAL + "clips:\n  enabled: true\n"))
    assert cfg.clips.enabled is True
    assert cfg.privacy.record_faces is False
    cfg2 = load_config(write(tmp_path, MINIMAL + "display:\n  enabled: true\n", "d.yaml"))
    assert cfg2.display.enabled is True


def test_remote_health_bind_requires_opt_in(tmp_path):
    body = MINIMAL + "health:\n  bind: 0.0.0.0\n"
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "allow_remote" in str(exc.value)
    ok = load_config(write(tmp_path, MINIMAL + "health:\n  bind: 0.0.0.0\n  allow_remote: true\n"))
    assert ok.health.bind == "0.0.0.0"


def test_mute_when_parked_is_refused(tmp_path):
    body = MINIMAL + "alerts:\n  mute_when_parked: true\n"
    with pytest.raises(ValueError) as exc:
        load_config(write(tmp_path, body))
    assert "vehicle speed" in str(exc.value)


def test_driver_roi_must_be_a_box(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, MINIMAL + "driver_roi: [0.5, 0.0, 0.2, 1.0]\n"))
    with pytest.raises(ValueError):
        load_config(write(tmp_path, MINIMAL + "driver_roi: [0.0, 0.0, 1.0]\n"))


def test_perclos_crit_must_not_be_below_warn(tmp_path):
    body = MINIMAL + "state:\n  perclos_warn: 0.4\n  perclos_crit: 0.2\n"
    with pytest.raises(ValueError):
        load_config(write(tmp_path, body))


def test_empty_backoff_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, MINIMAL + "camera:\n  backoff_s: []\n"))


# --------------------------------------------------------------- calibration


def test_calibration_merges_into_every_profile(tmp_path):
    """DMS-P16: the merge used to apply to ``default.yaml`` only.

    That made ``production.yaml`` impossible to calibrate, which in turn made
    every gaze-derived alert permanently suppressed on the vehicle profile.
    """
    (tmp_path / "vehicle.yaml").write_text(
        "forward_zero: {yaw: -3.5, pitch: 2.0}\nforward_zero_calibrated: true\n"
    )
    for name in ("default.yaml", "production.yaml", "anything.yaml"):
        path = write(tmp_path, "source: {type: test}\nmodels:\n"
                               "  face: {engine: e}\n  landmarks: {engine: e}\n", name)
        cfg = load_config(path)
        assert cfg.forward_zero.yaw == -3.5, name
        assert cfg.forward_zero_calibrated is True, name


def test_calibration_can_be_disabled(tmp_path):
    (tmp_path / "vehicle.yaml").write_text("forward_zero: {yaw: -3.5}\n")
    path = write(tmp_path, MINIMAL)  # MINIMAL sets calibration: ''
    cfg = load_config(path)
    assert cfg.forward_zero.yaw == 0.0
    assert calibration_path(path, {"calibration": ""}) == ""


def test_local_vehicle_yaml_carries_no_stale_calibration():
    """The old file held a forward zero computed from the WRONG landmarks.

    It also carried ``state.ear_open_median: 0.754``, which the merge would have
    pushed into every profile and so re-broken the corrected EAR thresholds.
    ``configs/vehicle.yaml`` is gitignored (it is per-unit), so this only checks
    the working copy on whatever machine runs the suite - which is exactly where
    a stale file would do the damage.
    """
    import yaml

    path = os.path.join(CONFIG_DIR, "vehicle.yaml")
    if not os.path.isfile(path):
        pytest.skip("no local calibration file on this unit")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    stale = (data.get("state") or {}).get("ear_open_median")
    assert stale is None or stale <= 0.55, (
        "configs/vehicle.yaml carries ear_open_median=%r, which is on the OLD "
        "broken-landmark EAR scale and would be merged into every profile" % stale
    )
    if data.get("forward_zero"):
        assert data.get("forward_zero_calibrated") is True, (
            "a calibration file that sets forward_zero must also state "
            "forward_zero_calibrated: true so the claim is explicit"
        )


def test_calibration_file_is_used_by_the_default_profile():
    assert calibration_path(os.path.join(CONFIG_DIR, "default.yaml")).endswith(
        os.path.join("configs", "vehicle.yaml")
    )


# ------------------------------------------------------------------- helpers


def test_is_loopback():
    for good in ("127.0.0.1", "localhost", "::1", "127.0.0.53", ""):
        assert is_loopback(good), good
    for bad in ("0.0.0.0", "192.168.1.10", "::"):
        assert not is_loopback(bad), bad


def test_capability_cfg_shape():
    cfg = load_config(os.path.join(CONFIG_DIR, "default.yaml"))
    caps = cfg.models.capability_cfg()
    assert caps["eyestate"]["min_eye_box_px"] == 20.0
    assert caps["objects"]["every_n"] == 20
    assert caps["identity"]["cosine_threshold"] == pytest.approx(0.363)
    assert caps["occlusion"]["every_n"] >= 1
    assert isinstance(AppConfig.parse_obj, object)


def test_replay_batch_config_for_clip():
    sys_path_scripts = os.path.join(ROOT, "scripts")
    if sys_path_scripts not in sys.path:
        sys.path.insert(0, sys_path_scripts)
    import run_all_replay as batch

    assert batch.config_for_clip("tests/replay/day_driver.mp4") == "configs/default.yaml"
    assert batch.config_for_clip("tests/replay/minivan.mp4") == "configs/minivan.yaml"
    assert batch.config_for_clip("tests/replay/red_car_crash.mp4") == "configs/red_car_crash.yaml"
    assert (
        batch.config_for_clip("tests/replay/family_vacation_crash.mp4")
        == "configs/family_vacation_crash.yaml"
    )
    assert (
        batch.config_for_clip("tests/replay/rear_end_accident.mp4")
        == "configs/rear_end_accident.yaml"
    )
    assert batch.config_for_clip("tests/replay/guy_rear_ended.mp4") == "configs/guy_rear_ended.yaml"
    assert batch.config_for_clip("tests/replay/bmw_rear_ended.mp4") == "configs/bmw_rear_ended.yaml"
    assert (
        batch.config_for_clip("tests/replay/getting_rear_ended.mp4")
        == "configs/getting_rear_ended.yaml"
    )
    assert (
        batch.config_for_clip("tests/replay/rear_end_whiplash.mp4")
        == "configs/rear_end_whiplash.yaml"
    )
    assert batch.config_for_clip("tests/replay/unknown_clip.mp4") == "configs/default.yaml"


# --------------------------------------------------- the gate a profile ships

#: The product latency gate, from docs/production-dms-design.md: 1280x720 end to
#: end, p95 <= 66 ms, peak RSS < 5.0 GB.  PINNED here so that "make the gate
#: pass" cannot be spelled "raise the gate".  Changing these two numbers is a
#: product decision and has to be made in this file, deliberately, with the
#: design document updated in the same change.
PRODUCT_GATE_E2E_P95_MS = 66.0
PRODUCT_GATE_PEAK_RSS_MB = 5000.0

#: End-to-end p95 MEASURED per profile on this Xavier NX (nvpmodel 8
#: MODE_20W_6CORE, 1280x720, 300 frames per run - 186 for rear_end_accident,
#: which is shorter - one dms.app process at a time under flock
#: /tmp/jetson-gpu.lock, 2026-09-13 evening KST), as
#: ``(runs, median_ms, min_ms, max_ms)``.
#:
#: REPEATED, not spot-checked, because the run-to-run spread on this board is
#: 6.5-33 ms and is therefore several times the margin the gate is judged by.  A
#: single observation of this pipeline is a sample from that distribution and
#: means almost nothing; the median of several is the number to reason with, and
#: the min/max are quoted so nobody mistakes the median for a guarantee.
#: ``default`` and ``lowlatency`` carry n=9 because they are the pair the
#: detector decision turns on (3 sweep runs each plus 6 paired A/B runs).
MEASURED_E2E_P95_MS = {
    "default.yaml": (9, 77.79, 66.50, 99.45),
    "lowlatency.yaml": (9, 77.15, 65.26, 82.51),
    "minivan.yaml": (3, 80.38, 65.70, 98.32),
    "red_car_crash.yaml": (3, 84.21, 78.79, 91.23),
    "family_vacation_crash.yaml": (3, 82.57, 75.92, 89.82),
    "rear_end_accident.yaml": (3, 85.79, 75.06, 87.38),
    "guy_rear_ended.yaml": (3, 85.75, 75.99, 93.71),
    "bmw_rear_ended.yaml": (3, 77.12, 66.17, 79.03),
    "getting_rear_ended.yaml": (3, 82.16, 76.36, 82.87),
    "rear_end_whiplash.yaml": (3, 69.60, 69.36, 77.05),
    "testsrc.yaml": (3, 35.12, 34.78, 35.75),
    "systemd-lab.yaml": (3, 36.50, 35.42, 39.83),
}

#: Profiles that cannot be executed on this board at all, and why.  An entry
#: here is a promise that the profile is unmeasurable, not that it is fast.
UNRUNNABLE_PROFILES = {
    "production.yaml": (
        "source type csi and there is no camera on this board; it also writes "
        "to /var/lib/dms, which is not writable without root, so it exits on "
        "the event log before it reaches the camera"
    ),
}

#: Profiles MEASURED to miss the product latency gate, with the median that
#: missed it.  This list is the honest record of a known degrade: the shipped
#: build does not meet its latency gate on real footage, on either detector, and
#: says so in the PERF_GATE line, in the /healthz note and here.  Ten of the thirteen
#: shipped profiles are on it - ten of the twelve that run on this board, the other
#: two being the synthetic sources that find no faces.  Shortening this list requires
#: a measurement, not an edit.
KNOWN_GATE_MISSES = {
    "default.yaml",
    "lowlatency.yaml",
    "minivan.yaml",
    "red_car_crash.yaml",
    "family_vacation_crash.yaml",
    "rear_end_accident.yaml",
    "guy_rear_ended.yaml",
    "bmw_rear_ended.yaml",
    "getting_rear_ended.yaml",
    "rear_end_whiplash.yaml",
}


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_every_profile_declares_the_product_latency_gate(path):
    """No profile may quietly relax the gate it is judged by.

    The predecessor of this test asserted the opposite direction - that the
    declared gate was at least as large as the detector's measured latency -
    which a profile could satisfy by declaring a bigger gate.  A gate a profile
    chooses for itself is not a gate.
    """
    cfg = load_config(path)
    assert cfg.perf.gate_e2e_p95_ms == pytest.approx(PRODUCT_GATE_E2E_P95_MS), (
        "%s declares gate_e2e_p95_ms=%.1f; the product gate is %.1f ms and "
        "moving it is a design decision, not a profile tweak."
        % (os.path.basename(path), cfg.perf.gate_e2e_p95_ms, PRODUCT_GATE_E2E_P95_MS)
    )
    assert cfg.perf.gate_peak_rss_mb == pytest.approx(PRODUCT_GATE_PEAK_RSS_MB), (
        os.path.basename(path),
        cfg.perf.gate_peak_rss_mb,
    )


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_every_profile_has_a_repeated_latency_measurement_on_record(path):
    """A shipped profile must have been measured, more than once, recently.

    "Measured" used to mean one number per DETECTOR, which is how a profile
    could inherit a 63.1 ms figure taken on a different clip in a different
    board condition and be considered characterised.  Per profile, n >= 3.
    """
    name = os.path.basename(path)
    if name in UNRUNNABLE_PROFILES:
        assert name not in MEASURED_E2E_P95_MS, (
            "%s is listed as unrunnable but also carries a measurement" % name
        )
        return
    row = MEASURED_E2E_P95_MS.get(name)
    assert row is not None, (
        "%s has no entry in MEASURED_E2E_P95_MS. Measure it on the board "
        "(at least three runs of 300 frames under flock /tmp/jetson-gpu.lock) "
        "and record (runs, median, min, max) before shipping it." % name
    )
    runs, med, lo, hi = row
    assert runs >= 3, "%s: %d run(s) is a spot check, not a measurement" % (name, runs)
    assert lo <= med <= hi, (name, row)


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_a_profile_that_misses_the_gate_is_declared_as_a_miss(path):
    """The honesty test. Measured reality and declared reality must agree.

    Both directions are failures:

    * a profile whose measured median exceeds the gate and is NOT in
      ``KNOWN_GATE_MISSES`` is an undeclared degrade - which is what the repo
      shipped before the SCRFD-512 swap, and what the swap was made to hide;
    * a profile in ``KNOWN_GATE_MISSES`` whose measured median is now inside the
      gate is stale pessimism, and stale honesty rots into noise that readers
      learn to skip.

    Note what this test does NOT do: it does not fail a profile for missing the
    gate.  The build misses the gate on nine of twelve profiles and that is the
    recorded, accepted state - see docs/performance.md section 1. It fails a
    profile for missing the gate SILENTLY.
    """
    name = os.path.basename(path)
    if name in UNRUNNABLE_PROFILES:
        assert name not in KNOWN_GATE_MISSES, name
        return
    runs, med, lo, hi = MEASURED_E2E_P95_MS[name]
    cfg = load_config(path)
    misses = med > cfg.perf.gate_e2e_p95_ms
    if misses:
        assert name in KNOWN_GATE_MISSES, (
            "%s measures a median e2e p95 of %.2f ms against its own %.1f ms "
            "gate and is not declared in KNOWN_GATE_MISSES. Declare it, and say "
            "so in docs/performance.md - do not swap a model to make the number "
            "look better." % (name, med, cfg.perf.gate_e2e_p95_ms)
        )
    else:
        assert name not in KNOWN_GATE_MISSES, (
            "%s is declared a gate miss but measures a median of %.2f ms "
            "against a %.1f ms gate. Re-measure and update both."
            % (name, med, cfg.perf.gate_e2e_p95_ms)
        )


def test_the_recorded_spread_is_wider_than_the_gate_margin():
    """Guards the reason every other number in this section is a median.

    If this ever stops being true the board has changed and the whole
    measurement protocol (n >= 3, medians, min/max) can be simplified. Until
    then, a single PASS is not evidence of a pass, and no document in this repo
    may quote one as if it were.
    """
    worst = 0.0
    for name, (runs, med, lo, hi) in MEASURED_E2E_P95_MS.items():
        if name in ("testsrc.yaml", "systemd-lab.yaml"):
            continue  # synthetic source, no faces: not the working pipeline
        worst = max(worst, hi - lo)
    assert worst > 10.0, (
        "recorded run-to-run spread has collapsed to %.1f ms; re-check whether "
        "the median-of-n protocol is still needed" % worst
    )


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_shipped_face_and_landmark_engines_are_on_disk_and_in_the_manifest(path):
    """An engine a profile names must be hashable at startup, not just present."""
    import json

    manifest = json.load(open(os.path.join(ROOT, "engines", "MANIFEST.json")))
    listed = set()
    for rec in (manifest.get("engines") or {}).values():
        if isinstance(rec, dict):
            name = rec.get("file") or rec.get("path")
            if name:
                listed.add(os.path.basename(str(name)))
    cfg = load_config(path)
    for engine in (cfg.models.face.engine, cfg.models.landmarks.engine):
        base = os.path.basename(engine)
        assert os.path.isfile(os.path.join(ROOT, engine)), "%s: %s missing" % (path, engine)
        assert base in listed, "%s: %s is not listed in engines/MANIFEST.json" % (path, base)


@pytest.mark.parametrize("path", PROFILES, ids=[os.path.basename(p) for p in PROFILES])
def test_capture_letterbox_describes_the_detector_the_profile_ships(path):
    """capture.letterbox must match the face engine's real input shape.

    The key was read by nothing and said ``[640, 640]`` in every profile; the app
    now builds its one full-frame letterbox from the engine and logs
    LETTERBOX_MISMATCH when the profile disagrees, so the profile has to be true.
    """
    import json

    manifest = json.load(open(os.path.join(ROOT, "engines", "MANIFEST.json")))
    cfg = load_config(path)
    base = os.path.basename(cfg.models.face.engine)
    shape = None
    for rec in (manifest.get("engines") or {}).values():
        if not isinstance(rec, dict):
            continue
        if os.path.basename(str(rec.get("file") or rec.get("path") or "")) != base:
            continue
        inputs = rec.get("inputs") or []
        if inputs and inputs[0].get("shape"):
            shape = inputs[0]["shape"]
    assert shape is not None, "MANIFEST records no input shape for %s" % base
    net_h, net_w = int(shape[2]), int(shape[3])
    assert tuple(cfg.capture.letterbox) == (net_w, net_h), (
        "%s declares capture.letterbox=%s but %s takes %dx%d"
        % (os.path.basename(path), tuple(cfg.capture.letterbox), base, net_w, net_h)
    )
