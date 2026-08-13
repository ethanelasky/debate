from infra.config import load_experiment
from infra.envs.debate.protocol import Protocol
from infra.run_debate import split_agents, validate_experiment, validate_trained_seats


def test_olmo32think_debate_splices_latest_debate_and_live_grpo_recipe():
    debate = load_experiment("configs/math_pc_olmo.yaml", "math_pc_olmo_l5_g8_nocot")
    arm = load_experiment("configs/math_pc_olmo.yaml", "math_pc_olmo32think_cispo_os3")
    live = load_experiment("configs/math_rlvr_olmo.yaml", "aime_rlvr_olmo32think_cispo_os3")

    validate_experiment(arm)
    trained, frozen = split_agents(arm)
    validate_trained_seats(trained, arm["training"])

    # Debate semantics inherit from the latest completed debate — EXCEPT the
    # protocol, which is think-native for this arm (2026-08-12): native
    # <think> replaces scratchpad/plan turns, and the solo-opening flag
    # requires solution-first. The slot asserts below pin its exact shape.
    for key in ("prompt_config", "fresh_positions", "plan_tokens", "scoring"):
        assert arm[key] == debate[key]
    assert arm["dataset"] == debate["dataset"]
    slots = [(s.speaker, s.slot.name, s.slot.max_total_tokens) for s in Protocol.parse(arm["protocol"]).compile()]
    # Think-native protocol (2026-08-12): native <think> replaces every
    # scratchpad/plan turn; solution-first is REQUIRED by
    # first_speech_non_debate_aware.
    assert slots == [
        ("alice", "proposal", 1500),
        ("bob", "critique", 1500),
        ("alice", "defense", 1500),
        ("bob", "rebuttal", 1500),
        ("judge", "deliberation", 1000),
        ("judge", "verdict", 256),
    ]
    think_caps = [
        (s.speaker, s.slot.max_think_tokens)
        for s in Protocol.parse(arm["protocol"]).compile()
        if s.speaker != "judge"
    ]
    assert think_caps == [("alice", 1000), ("bob", 1000), ("alice", 1000), ("bob", 1000)]

    assert set(trained) == {"alice", "bob"}
    assert {s.model_file_path for s in trained.values()} == {"allenai/Olmo-3-32B-Think-DPO"}
    assert all(s.enable_thinking is True for s in trained.values())
    assert frozen["judge"].model_file_path == "Qwen/Qwen3.5-4B"

    # Top-level optimization knobs come from the currently running 32B GRPO
    # arm. The context/topology block is checked separately because debate
    # retains its protocol-derived lengths and judge-safe rollout allocation.
    same_training = {
        "lora_rank",
        "loss",
        "ppo_epochs",
        "adv_length_norm",
        # "steps" deliberately NOT pinned: horizon is a per-arm choice, not
        # part of the optimizer recipe — the RLVR arms went to 500
        # (2026-08-12) while a debate step costs ~10x an RLVR step.
        "batch_size",
        "group_size",
        "lr",
        "kl_coef",
        "kl_discount_factor",
        "eval_every",
        "eval_n",
        "save_every",
        "dynamic_sampling_retries",
        "oversample_factor",
        "kl_mechanism",
        "drop_zero_advantage",
        "eval_split",
        "final_test_eval",
        "warmup_steps",
        "lr_schedule",
        "min_lr_ratio",
    }
    assert {k: arm["training"][k] for k in same_training} == {
        k: live["training"][k] for k in same_training
    }
    verl = arm["training"]["verl"]
    assert verl["n_gpus"] == 2
    assert verl["rollout_tp"] == 2
    assert verl["gpu_memory_utilization"] == 0.33
    assert verl["prompt_length"] == 8192
    assert verl["response_length"] == 2048
    assert verl["max_token_len_per_gpu"] == 16384
    assert verl["extra_overrides"] == live["training"]["verl"]["extra_overrides"]


def test_olmo32_instruct_aime_debate_twin():
    """The AIME instruct debate arm is the debate twin of the live instruct
    RLVR arm: same optimizer recipe, same dataset/eval shape, think budgets
    gone (no think channel on the instruct model). Throughput deviations
    (Ethan, 2026-08-13) are pinned explicitly below."""
    arm = load_experiment("configs/math_pc_olmo.yaml", "aime_pc_olmo32_cispo_os2")
    think = load_experiment("configs/math_pc_olmo.yaml", "math_pc_olmo32think_cispo_os3")
    live = load_experiment("configs/math_rlvr_olmo.yaml", "aime_rlvr_olmo32_plan1k_cispo_os3")

    validate_experiment(arm)
    trained, frozen = split_agents(arm)
    validate_trained_seats(trained, arm["training"])

    assert arm["dataset"] == live["dataset"]
    assert arm["plan_tokens"] == live["plan_tokens"]
    assert arm["first_speech_non_debate_aware"] is True

    compiled = Protocol.parse(arm["protocol"]).compile()
    slots = [(s.speaker, s.slot.name, s.slot.max_total_tokens) for s in compiled]
    # Think-arm totals except the 300-token rebuttal (throughput: the closing
    # speech responds to the defense, it does not re-derive).
    assert slots == [
        ("alice", "proposal", 1500),
        ("bob", "critique", 1500),
        ("alice", "defense", 1500),
        ("bob", "rebuttal", 300),
        ("judge", "deliberation", 1000),
        ("judge", "verdict", 256),
    ]
    assert all(s.slot.max_think_tokens is None for s in compiled)

    assert set(trained) == {"alice", "bob"}
    assert {s.model_file_path for s in trained.values()} == {"/workspace/models/olmo32-bf16"}
    assert all(not s.enable_thinking for s in trained.values())
    assert frozen["judge"].model_file_path == "Qwen/Qwen3.5-4B"

    # oversample_factor deliberately NOT in this set: the debate arm runs 2
    # against the twin's 3 (first steps showed 0-6 degenerates per 192
    # debates — the third draw was waste).
    same_training = {
        "lora_rank", "loss", "ppo_epochs", "adv_length_norm", "batch_size",
        "group_size", "lr", "kl_coef", "kl_discount_factor", "eval_every",
        "eval_n", "save_every", "dynamic_sampling_retries",
        "kl_mechanism", "drop_zero_advantage", "eval_split", "final_test_eval",
        "warmup_steps", "lr_schedule", "min_lr_ratio",
    }
    assert {k: arm["training"][k] for k in same_training} == {
        k: live["training"][k] for k in same_training
    }
    assert arm["training"]["oversample_factor"] == 2

    # Engine shapes identical to the think debate arm's, including the
    # judge-safe 0.33 rollout allocation (0.45 died on a vLLM scheduler bug).
    assert arm["training"]["verl"] == think["training"]["verl"]
