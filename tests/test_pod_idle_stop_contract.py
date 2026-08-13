from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reverse_tunnel_notty_session_cannot_pin_paid_pod() -> None:
    script = (REPO_ROOT / "scripts" / "pod_idle_stop.sh").read_text()

    assert "sshd[^ ]*:.*@pts/[0-9]+" in script
    assert "pgrep -f 'sshd[^ ]*:.*@'" not in script
    assert "user@notty" in script
