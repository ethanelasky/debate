"""Debate RL runner: experiment YAML -> DebateEnv + backend -> train loop.

Config shape (ONE agents block; the trained seat is marked inline — its
model_settings drive the training backend AND its sampling; `training:` holds
optimization/loop knobs only):

    math_pc:
      topology: {turns: [...]}
      prompt_config: {file_path: ..., entry: ...}
      agents:
        alice:
          trained: true
          model_settings: {model_type: tinker, model_file_path: Qwen/Qwen3.5-4B,
                           lora_rank: 32, enable_thinking: true,
                           max_new_tokens: 3072,
                           sampling: {train: {temperature: 1.0, top_p: 1.0}}}
        bob:   {model_settings: {...}}       # frozen, built by the factory
        judge: {model_settings: {...}}
      judge_config: {schema_name: competitive, retries: 4}
      scoring: {scoring: continuous, confidence_source: json, shaping: [...]}
      dataset: {levels: [1, 2], relaxed_extraction: true}
      training: {loss: {kind: ppo, clip_low: 0.8, clip_high: 1.2}, ppo_epochs: 1,
                 steps: 100, batch_size: 8, group_size: 4, lr: 1e-5,
                 kl_coef: 0.02, wandb_project: ..., eval_every: 20, ...}
"""

from __future__ import annotations

import argparse

from infra.backend.base import LossSpec, SamplingParams
from infra.config import load_experiment, parse_model_settings
from infra.envs.answer_parsing import extract_last_number, extract_number_from_boxed_answer
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.topology import Topology
from infra.envs.math_env import MathEnv
from infra.models.base import ModelSettings, resolved_sampling_profile
from infra.models.factory import instantiate_model
from infra.train import Config, train


def strict_extract(text):
    return extract_number_from_boxed_answer(text)


def relaxed_extract(text):
    v = extract_number_from_boxed_answer(text)
    return v if v is not None else extract_last_number(text)


def split_agents(exp: dict) -> tuple[dict[str, ModelSettings], dict[str, ModelSettings]]:
    """agents block -> (trained, frozen) settings by speaker."""
    trained: dict[str, ModelSettings] = {}
    frozen: dict[str, ModelSettings] = {}
    for speaker, agent in (exp.get("agents") or {}).items():
        settings = parse_model_settings(agent["model_settings"])
        (trained if agent.get("trained") else frozen)[speaker] = settings
    return trained, frozen


def build_env(exp: dict, trained: dict[str, ModelSettings], frozen: dict[str, ModelSettings]) -> DebateEnv:
    topo_spec = exp["topology"]
    if isinstance(topo_spec, str):
        topo_spec = exp["_topologies"][topo_spec]
    topology = Topology.parse(topo_spec)

    frozen_models = {
        speaker: instantiate_model(settings, is_debater=speaker != "judge", binding="train")
        for speaker, settings in frozen.items()
    }

    # Per-trained-seat sampling + template kwargs; trained seats must sample
    # UNBIASED (temp/top_p 1.0) — anything else corrupts the ratio anchor
    # (the old repo's §6 gate, kept hard).
    # Token budgets are the TOPOLOGY's job (per-slot caps); the seat's base
    # max_tokens is just a ceiling derived from its largest slot cap.
    caps_by_speaker: dict[str, list] = {}
    for cs in topology.compile():
        caps_by_speaker.setdefault(cs.speaker, []).append(cs.slot.max_total_tokens)
    trained_sampling: dict[str, SamplingParams] = {}
    trained_chat_kwargs: dict[str, dict] = {}
    for speaker, settings in trained.items():
        profile = resolved_sampling_profile(settings, "train")
        temp = profile.temperature if profile.temperature is not None else 1.0
        top_p = profile.top_p if profile.top_p is not None else 1.0
        if temp != 1.0 or top_p != 1.0:
            raise ValueError(
                f"trained seat {speaker!r} samples at temperature={temp}, top_p={top_p}; "
                "trained seats must use 1.0/1.0 (sampler logprobs anchor the PPO ratio — "
                "biased sampling corrupts the anchor)"
            )
        caps = caps_by_speaker.get(speaker, [])
        if any(c is None for c in caps):
            raise ValueError(
                f"trained seat {speaker!r} has topology slot(s) without max_total_tokens; "
                "set per-slot budgets in the topology (speech budgets are format-specific)"
            )
        trained_sampling[speaker] = SamplingParams(max_tokens=max(caps), temperature=temp, top_p=top_p)
        if settings.enable_thinking is not None:
            trained_chat_kwargs[speaker] = {"enable_thinking": bool(settings.enable_thinking)}

    ds = exp.get("dataset") or {}
    config = DebateEnvConfig(
        topology=topology,
        prompt_file=exp["prompt_config"]["file_path"],
        prompt_entry=exp["prompt_config"]["entry"],
        trained_speakers=list(trained),
        frozen_models=frozen_models,
        trained_sampling=trained_sampling,
        trained_chat_kwargs=trained_chat_kwargs,
        judge=JudgeConfig(**(exp.get("judge_config") or {})),
        scoring=ScoringConfig(**(exp.get("scoring") or {})),
        fresh_positions=exp.get("fresh_positions", True),
        flip=exp.get("flip", False),
    )
    task_source = MathEnv(
        seed=int(ds.get("seed", 0)),
        levels=tuple(ds.get("levels", (5,))),
        eval_subset_size=int(ds.get("eval_subset_size", 512)),
    )
    extractor = relaxed_extract if ds.get("relaxed_extraction", True) else strict_extract
    return DebateEnv(config, task_source, extractor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-file", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", default=None, help="override training.wandb_project")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    parser.add_argument("--load", default=None)
    args = parser.parse_args()

    exp = load_experiment(args.experiment_file, args.experiment)
    trained, frozen = split_agents(exp)
    if not trained:
        raise ValueError("no agent has trained: true")
    trained_settings = list(trained.values())
    base_models = {s.model_file_path for s in trained_settings}
    if len(base_models) != 1:
        raise ValueError(f"trained seats must share one base model, got {base_models}")

    env = build_env(exp, trained, frozen)
    tr = exp.get("training") or {}

    from infra.backend.tinker import TinkerBackend

    lead = trained_settings[0]
    backend = TinkerBackend(lead.model_file_path, lora_rank=int(lead.lora_rank or 32))
    if args.load:
        backend.load(args.load)

    profile = resolved_sampling_profile(lead, "train")
    cfg = Config(
        base_model=lead.model_file_path,
        steps=args.steps if args.steps is not None else int(tr.get("steps", 100)),
        batch_size=int(tr.get("batch_size", 8)),
        group_size=int(tr.get("group_size", 4)),
        micro_batch=int(tr.get("micro_batch", 64)),
        lr=float(tr.get("lr", 1e-5)),
        loss=LossSpec(**(tr.get("loss") or {})),
        ppo_epochs=int(tr.get("ppo_epochs", 1)),
        adv_length_norm=str(tr.get("adv_length_norm", "none")),
        kl_coef=float(tr.get("kl_coef", 0.0)),
        kl_discount_factor=float(tr.get("kl_discount_factor", 0.0)),
        sampling=SamplingParams(
            # no ceiling here: budgets are the topology's per-slot caps, and
            # Policy hard-errors on any generation left unbounded
            max_tokens=None,
            temperature=profile.temperature if profile.temperature is not None else 1.0,
            top_p=profile.top_p if profile.top_p is not None else 1.0,
        ),
        eval_every=int(tr.get("eval_every", 20)),
        eval_n=int(tr.get("eval_n", 64)),
        save_every=int(tr.get("save_every", 50)),
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=args.experiment,
        chat_template_kwargs=(
            {"enable_thinking": bool(lead.enable_thinking)} if lead.enable_thinking is not None else None
        ),
    )
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
