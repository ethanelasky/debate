from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validated_b200_template_is_in_paid_launch_allowlist() -> None:
    launcher = (REPO_ROOT / "scripts" / "pod_create.sh").read_text()
    topologies = (REPO_ROOT / "configs" / "topologies.yaml").read_text()

    assert '"NVIDIA B200|a9dk3g7cny"' in launcher
    assert "2xB200:" in topologies
