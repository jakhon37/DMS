from __future__ import annotations

import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_dms_service_unit():
    text = open(os.path.join(ROOT, "deploy/dms.service")).read()
    assert "Type=notify" in text
    assert "TimeoutStartSec=90" in text
    assert "WatchdogSec=30" in text
    assert "SupplementaryGroups=video gpio" in text
    assert "StateDirectory=dms" in text
    assert "User=dms" in text
    assert "--detect" in text
    assert "/etc/dms/default.yaml" in text
    assert "nvpmodel" not in text
    assert "jetson_clocks" not in text


def test_setup_jetson_script_syntax():
    script = os.path.join(ROOT, "deploy/setup_jetson.sh")
    subprocess.check_call(["bash", "-n", script])
    text = open(script).read()
    assert "--apply-power" in text
    assert "useradd" in text
    assert "configs/production.yaml" in text
    assert "configs/systemd-lab.yaml" in text
    assert "--lab" in text
    assert "/etc/dms/default.yaml" in text
    assert "jetson_clocks is NEVER auto" in text or "never auto" in text.lower()


def test_logrotate_jsonl():
    text = open(os.path.join(ROOT, "deploy/dms.logrotate")).read()
    assert "events.jsonl" in text
    assert "50M" in text
