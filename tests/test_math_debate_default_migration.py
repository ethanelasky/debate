"""The math debate prompt and useful descendants have one active identity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from infra.config import (
    load_config_with_includes,
    load_experiment,
    resolve_all_experiments,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "math_pc_debate.yaml"
PROMPT = ROOT / "infra" / "prompts" / "debate" / "hendrycks_math.yaml"
PROMPT_CONFIG = {
    "file_path": "infra/prompts/debate/hendrycks_math.yaml",
    "entry": "math_proposer_critic",
}
DESCENDANTS = (
    "mathl5_qwen35_pc_debate_cispo_verl_sub8",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_warm",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_rlvrs20",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget",
    "math_pc_no_rebuttals",
)


def test_useful_descendants_resolve_through_the_default_prompt_only() -> None:
    entries = resolve_all_experiments(load_config_with_includes(CONFIG))
    prompt_entries = resolve_all_experiments(load_config_with_includes(PROMPT))
    legacy_marker = "box" + "req"
    legacy_base = "mathl5_qwen35_pc_debate_cispo_verl_" + legacy_marker

    for name in DESCENDANTS:
        assert load_experiment(CONFIG, name)["prompt_config"] == PROMPT_CONFIG
    assert legacy_base not in entries
    assert not any(name.startswith(legacy_base + "_") for name in entries)
    assert "math_proposer_critic_" + legacy_marker not in prompt_entries


def test_retired_prompt_variants_are_absent() -> None:
    marker = "box" + "req"
    second_marker = "first" + "speech"
    prompt_dir = PROMPT.parent
    assert not (prompt_dir / f"hendrycks_math_{marker}.yaml").exists()
    assert not (prompt_dir / f"hendrycks_math_{marker}_{second_marker}.yaml").exists()


def test_tracked_active_sources_contain_no_retired_variant_names() -> None:
    markers = (("box" + "req"), ("first" + "speech"))
    tracked = subprocess.run(
        ["git", "ls-files", "configs", "infra", "scripts", "tests"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    active_suffixes = {".json", ".md", ".py", ".yaml", ".yml"}

    offenders: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.suffix not in active_suffixes:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        if any(marker in lowered for marker in markers):
            offenders.append(relative)
    assert offenders == []


def test_renamed_job_specs_have_coherent_commands_and_output_roots() -> None:
    cases = {
        "debate_warm.json": (
            "mathl5_qwen35_pc_debate_cispo_verl_sub8_warm",
            "warm01",
            "debate-sub8-warm",
            "debate_sub8_warm",
        ),
        "debate_kenton.json": (
            "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton",
            "kenton01",
            "debate-sub8-kenton",
            "debate_sub8_kenton",
        ),
        "debate_kenton_rlvrs20.json": (
            "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_rlvrs20",
            "rlvrs20",
            "debate-sub8-kenton-rlvrs20",
            "debate_sub8_kenton_rlvrs20",
        ),
        "debate_kenton_propbudget.json": (
            "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget",
            "propbudget",
            "debate-kenton-propbudget",
            "debate_kenton_propbudget",
        ),
    }

    for filename, (experiment, namespace, artifact_slug, output_slug) in cases.items():
        spec = json.loads((ROOT / "infra" / "jobd" / filename).read_text())
        job = spec["jobs"][0]
        outputs = job["outputs"]
        assert f"bash scripts/pod_run.sh debate {experiment}" in job["command"]
        assert outputs[0]["remote"] == f"/workspace/checkpoints/{experiment}/{namespace}/"
        assert outputs[0]["local"] == f"~/code/debate/artifacts/{artifact_slug}/checkpoints"
        assert outputs[1] == {
            "remote": "/workspace/debate/transcripts",
            "local": f"~/code/debate/outputs/{output_slug}/transcripts",
            "every_s": 1800.0,
        }
        assert outputs[2] == {
            "remote": "/workspace/debate/docent",
            "local": f"~/code/debate/outputs/{output_slug}/docent",
            "every_s": 1800.0,
        }
