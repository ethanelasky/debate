from infra.config import load_experiment
from infra.envs.debate.protocol import Protocol
from infra.run_debate import split_agents, validate_experiment, validate_trained_seats


def test_olmo32think_debate_splices_latest_debate_and_live_grpo_recipe():
    debate = load_experiment("configs/math_pc_olmo.yaml", "math_pc_olmo_l5_g8_nocot")
    arm = load_experiment("configs/math_pc_olmo.yaml", "math_pc_olmo32think_grpo_os3")
    live = load_experiment("configs/math_rlvr_olmo.yaml", "aime_rlvr_olmo32think_grpo_os3")

    validate_experiment(arm)
    trained, frozen = split_agents(arm)
    validate_trained_seats(trained, arm["training"])

    # Debate semantics are inherited byte-for-byte from the latest completed
    # debate. Protocol.compile() is ordered, so this also locks serial stages.
    for key in ("protocol", "prompt_config", "fresh_positions", "plan_tokens", "scoring"):
        assert arm[key] == debate[key]
    assert arm["dataset"] == debate["dataset"]
    slots = [(s.speaker, s.slot.name, s.slot.max_total_tokens) for s in Protocol.parse(arm["protocol"]).compile()]
    assert slots == [
        ("alice", "plan", 1000),
        ("alice", "proposal", 2000),
        ("bob", "scratchpad", 500),
        ("bob", "critique", 500),
        ("alice", "scratchpad", 500),
        ("alice", "defense", 500),
        ("bob", "scratchpad", 500),
        ("bob", "rebuttal", 500),
        ("judge", "deliberation", 1000),
        ("judge", "verdict", 256),
    ]

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
        "steps",
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
