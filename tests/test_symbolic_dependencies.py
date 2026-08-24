from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = (
    "math-verify[antlr4_13_2]==0.9.0",
    "latex2sympy2-extended==1.11.0",
    "sympy==1.14.0",
    "antlr4-python3-runtime==4.13.2",
)
PROVISION_SCRIPTS = (
    ROOT / "scripts" / "provision_sm90.sh",
    ROOT / "scripts" / "provision_blackwell.sh",
)


def test_project_declares_exact_symbolic_dependency_stack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    for pin in PINS:
        assert pin in dependencies


def _symbolic_dependency_block(script: Path) -> str:
    text = script.read_text()
    match = re.search(r"SYMBOLIC_DEPS=\(\n(?P<body>.*?)\n\)", text, re.DOTALL)
    assert match is not None, f"missing SYMBOLIC_DEPS block in {script}"
    return match.group("body")


def test_pod_provisions_use_the_same_exact_symbolic_pins() -> None:
    blocks = [_symbolic_dependency_block(script) for script in PROVISION_SCRIPTS]
    assert blocks[0] == blocks[1]

    for pin in PINS:
        assert f'"{pin}"' in blocks[0]
    for script in PROVISION_SCRIPTS:
        assert '"${SYMBOLIC_DEPS[@]}"' in script.read_text()


def test_pod_run_has_bounded_exact_version_preflight_before_judge_start() -> None:
    text = (ROOT / "scripts" / "pod_run.sh").read_text()
    preflight = text.index('timeout -k 10s 60 "$PY"')
    judge_start = text.index('echo "== starting judge vLLM server =="')

    assert preflight < judge_start
    assert "importlib.import_module" in text
    assert "from importlib import metadata" in text
    assert "exit 4" in text[preflight:judge_start]
    assert "--no-deps" in text[:preflight]
    for distribution, version in (
        ("math-verify", "0.9.0"),
        ("latex2sympy2-extended", "1.11.0"),
        ("sympy", "1.14.0"),
        ("antlr4-python3-runtime", "4.13.2"),
    ):
        assert f'"{distribution}": "{version}"' in text[preflight:judge_start]


def test_changed_shell_scripts_parse() -> None:
    scripts = (*PROVISION_SCRIPTS, ROOT / "scripts" / "pod_run.sh")
    for script in scripts:
        subprocess.run(["bash", "-n", script], check=True)
