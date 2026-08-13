import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reverse_tunnel_notty_session_cannot_pin_paid_pod() -> None:
    script = (REPO_ROOT / "scripts" / "pod_idle_stop.sh").read_text()

    assert "sshd[^ ]*:.*@pts/[0-9]+" in script
    assert "pgrep -f 'sshd[^ ]*:.*@'" not in script
    assert "user@notty" in script


def test_remote_job_busy_marker_requires_exact_linux_process_identity() -> None:
    script = (REPO_ROOT / "scripts" / "pod_idle_stop.sh").read_text()

    assert 'path = "/root/palaestra_remote_job_busy.json"' in script
    assert 'stat.S_IMODE(before.st_mode) == 0o600' in script
    assert 'before.st_uid == 0' in script
    assert 'set(marker) != {"format", "pid", "pgid", "start_ticks"}' in script
    assert 'raise ValueError("duplicate JSON key")' in script
    assert 'fields[0] != "Z"' in script
    assert 'int(fields[2]) == pid' in script
    assert 'int(fields[3]) == pid' in script
    assert 'int(fields[19]) == marker["start_ticks"]' in script
    assert 'BUSY_REASON="exact remote-job wrapper identity' in script


def test_shared_volume_paths_are_namespaced_and_overrides_are_normalized() -> None:
    script = (REPO_ROOT / "scripts" / "pod_idle_stop.sh").read_text()

    assert 'POD_LOG_NAMESPACE="/$RUNPOD_POD_ID"' in script
    assert 'POD_IDLE_LOG:-/workspace/logs${POD_LOG_NAMESPACE}/pod_idle_stop.log' in script
    assert 'POD_IDLE_EVAC_ROOT:-/workspace/logs/evacuation${POD_LOG_NAMESPACE}' in script
    assert 'validate_safe_absolute_path "$LOG" POD_IDLE_LOG' in script
    assert 'validate_safe_absolute_path "$EVAC_ROOT" POD_IDLE_EVAC_ROOT' in script
    assert 'EVAC_DIR="$EVAC_ROOT/' in script


@pytest.mark.parametrize("unsafe", ["relative", "/", "/tmp//x", "/tmp/../x", "/tmp/./x"])
def test_unsafe_durable_path_override_fails_before_watchdog_probe(unsafe: str) -> None:
    environment = dict(os.environ)
    environment["POD_IDLE_LOG"] = unsafe
    environment["POD_IDLE_EVAC_ROOT"] = "/tmp/palaestra-watchdog-test"
    result = subprocess.run(
        ["/bin/bash", str(REPO_ROOT / "scripts" / "pod_idle_stop.sh"), "--once"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "must be" in result.stderr
