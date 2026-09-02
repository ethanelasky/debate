"""The math debate prompt and useful descendants have one active identity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from infra.config import (
    load_config_with_includes,
    load_experiment,
    resolve_all_experiments,
)
from infra.envs.debate.protocol import Protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "math_pc_debate.yaml"
PROMPT = ROOT / "infra" / "prompts" / "debate" / "hendrycks_math.yaml"
PROMPT_CONFIG = {
    "file_path": "infra/prompts/debate/hendrycks_math.yaml",
    "entry": "math_proposer_critic",
}
ALICE_ONLY = "mathl5_qwen35_pc_debate_norebut_propbudget_rlvrs20_alice"
ALICE_ONLY_SMOKE = ALICE_ONLY + "_smoke"
DESCENDANTS = (
    "mathl5_qwen35_pc_debate_cispo_verl_sub8",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_warm",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_rlvrs20",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget",
    "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget_rlvrs20",
    "math_pc_no_rebuttals",
    ALICE_ONLY,
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


def test_default_pc_debate_descendants_use_task_only_openings() -> None:
    entries = resolve_all_experiments(load_config_with_includes(CONFIG))

    for name in ("math_pc_debate", *DESCENDANTS):
        assert load_experiment(CONFIG, name)["first_speech_non_debate_aware"] is True

    # The former solo alias is now the default PC-debate behavior, so keeping
    # another experiment name would falsely imply two scientific arms.
    assert "math_pc_solo" not in entries


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
            "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget_rlvrs20",
            "ab-cleanctx-propbudget-c002-01",
            "debate-ab-cleanctx-propbudget-c002-rlvrs20",
            "debate_ab_cleanctx_propbudget_c002_rlvrs20",
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

    ab_rerun = json.loads(
        (ROOT / "infra" / "jobd" / "debate_kenton_rlvrs20.json").read_text()
    )["jobs"][0]
    assert ab_rerun["gpu_type"] == "NVIDIA B200"
    assert ab_rerun["max_attempts"] == 1


def test_warm_start_ab_arm_prices_alice_proposal_excess_proportionally() -> None:
    experiment = load_experiment(
        CONFIG,
        "mathl5_qwen35_pc_debate_cispo_verl_sub8_kenton_propbudget_rlvrs20",
    )
    terms = {term["name"]: term for term in experiment["scoring"]["shaping"]}
    assert terms["proposal_token_budget"] == {
        "name": "proposal_token_budget",
        "kind": "budget_penalty",
        "coeff": 0.002,
        "limit": 4000,
        "mode": "proportional",
        "counts": "total",
        "slots": ["proposal"],
    }
    assert experiment["agents"]["alice"]["model_settings"]["model_file_path"] == (
        "/workspace/models/qwen35-rlvr-sub8-s20"
    )


def test_step20_no_rebuttal_arm_trains_only_alice_against_a_frozen_bob() -> None:
    experiment = load_experiment(
        CONFIG,
        ALICE_ONLY,
    )
    compiled = Protocol.parse(experiment["protocol"]).compile()

    assert [slot.slot.name for slot in compiled] == [
        "proposal",
        "critique",
        "deliberation",
        "verdict",
    ]
    assert experiment["first_speech_non_debate_aware"] is True
    assert [
        speaker
        for speaker, agent in experiment["agents"].items()
        if agent.get("trained")
    ] == ["alice"]

    alice = experiment["agents"]["alice"]
    bob = experiment["agents"]["bob"]
    assert alice["model_settings"]["model_file_path"] == (
        "/workspace/models/qwen35-rlvr-sub8-s20"
    )
    assert bob["trained"] is False
    assert bob["frozen_policy"] is True
    assert bob["model_settings"]["model_file_path"] == (
        "/workspace/models/qwen35-rlvr-sub8-s20"
    )
    assert bob["model_settings"]["base_url"] == "http://127.0.0.1:8789/v1"

    slots = {slot.slot.name: slot.slot for slot in compiled}
    assert slots["critique"].max_think_tokens == 4000
    assert slots["critique"].max_visible_tokens == 320
    terms = {term["name"]: term for term in experiment["scoring"]["shaping"]}
    assert terms["proposal_token_budget"] == {
        "name": "proposal_token_budget",
        "kind": "budget_penalty",
        "coeff": 0.002,
        "limit": 4000,
        "mode": "proportional",
        "counts": "total",
        "slots": ["proposal"],
    }


def test_alice_only_live_smoke_changes_run_size_not_scientific_protocol() -> None:
    full = load_experiment(CONFIG, ALICE_ONLY)
    smoke = load_experiment(
        ROOT / "infra" / "jobd" / "debate_s20_aliceonly_smoke.yaml",
        ALICE_ONLY_SMOKE,
    )

    for key in (
        "protocol",
        "prompt_config",
        "agents",
        "judge_config",
        "scoring",
        "dataset",
        "fresh_positions",
        "flip",
        "first_speech_non_debate_aware",
    ):
        assert smoke.get(key) == full.get(key)
    assert smoke["training"] == {
        **full["training"],
        "steps": 1,
        "batch_size": 2,
        "group_size": 2,
        "eval_every": 0,
        "final_test_eval": False,
        "save_every": 1,
    }


def test_alice_only_full_job_is_gated_on_the_live_routing_smoke() -> None:
    spec = yaml.safe_load(
        (ROOT / "infra" / "jobd" / "debate_s20_aliceonly.yaml").read_text()
    )
    smoke, full = spec["jobs"]

    assert smoke["name"] == "debate-s20-aliceonly-smoke"
    assert smoke["command"] == "bash scripts/run_s20_aliceonly_job.sh smoke"
    assert full["name"] == "debate-s20-aliceonly"
    assert full["depends_on"] == [smoke["name"]]
    assert full["command"] == "bash scripts/run_s20_aliceonly_job.sh full"
    assert all(job["max_attempts"] == 1 for job in (smoke, full))
    assert all(job["gpu_type"] == "NVIDIA B200" for job in (smoke, full))
    assert all(
        job["inputs"][0]["local"] == "~/code/debate-aliceonly-source"
        for job in (smoke, full)
    )
    assert smoke["inputs"][1] == full["inputs"][1] == {
        "local": "~/code/debate/artifacts/rlvr-cispo-sub8-kenton/checkpoints/step-00020",
        "remote": "/workspace/rlvr-s20-adapter",
    }
