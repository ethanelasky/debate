"""Debate RL runner: experiment YAML -> DebateEnv + backend -> train loop.

Experiment entries use the same _includes/_extends resolver as everything else:

    math_pc:
      topology: {turns: [...]}                  # inline, or a _topologies ref
      prompt_config: {file_path: ..., entry: math_proposer_critic}
      trained: [alice]
      agents:                                   # frozen seats, built by the factory
        bob:   {model_settings: {...}}
        judge: {model_settings: {...}}
      judge_config: {schema_name: competitive, retries: 4}
      scoring: {scoring: continuous, confidence_source: json, shaping: []}
      dataset: {levels: [1, 2], relaxed_extraction: true}
      training: {base_model: Qwen/Qwen3.5-4B, lora_rank: 32, steps: 100,
                 batch_size: 8, group_size: 4, lr: 1e-5, max_tokens: 3072,
                 eval_every: 20, eval_n: 64, save_every: 50}
"""

from __future__ import annotations

import argparse

from infra.backend.base import SamplingParams
from infra.config import load_experiment, parse_model_settings
from infra.envs.answer_parsing import extract_last_number, extract_number_from_boxed_answer
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.topology import Topology
from infra.envs.math_env import MathEnv
from infra.models.factory import instantiate_model
from infra.train import Config, train


def strict_extract(text):
    return extract_number_from_boxed_answer(text)


def relaxed_extract(text):
    v = extract_number_from_boxed_answer(text)
    return v if v is not None else extract_last_number(text)


def build_env(exp: dict) -> DebateEnv:
    topo_spec = exp["topology"]
    if isinstance(topo_spec, str):
        topo_spec = exp["_topologies"][topo_spec]
    topology = Topology.parse(topo_spec)

    frozen = {}
    for speaker, agent in (exp.get("agents") or {}).items():
        settings = parse_model_settings(agent["model_settings"])
        frozen[speaker] = instantiate_model(settings, is_debater=speaker != "judge", binding="train")

    ds = exp.get("dataset") or {}
    extractor = relaxed_extract if ds.get("relaxed_extraction", True) else strict_extract

    config = DebateEnvConfig(
        topology=topology,
        prompt_file=exp["prompt_config"]["file_path"],
        prompt_entry=exp["prompt_config"]["entry"],
        trained_speakers=list(exp.get("trained") or []),
        frozen_models=frozen,
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
    env = build_env(exp)
    tr = exp.get("training") or {}

    from infra.backend.tinker import TinkerBackend

    backend = TinkerBackend(tr.get("base_model", "Qwen/Qwen3.5-4B"), lora_rank=int(tr.get("lora_rank", 32)))
    if args.load:
        backend.load(args.load)

    cfg = Config(
        base_model=tr.get("base_model", "Qwen/Qwen3.5-4B"),
        steps=args.steps if args.steps is not None else int(tr.get("steps", 100)),
        batch_size=int(tr.get("batch_size", 8)),
        group_size=int(tr.get("group_size", 4)),
        micro_batch=int(tr.get("micro_batch", 64)),
        lr=float(tr.get("lr", 1e-5)),
        sampling=SamplingParams(
            max_tokens=int(tr.get("max_tokens", 3072)),
            temperature=float(tr.get("temperature", 1.0)),
            top_p=float(tr.get("top_p", 1.0)),
        ),
        kl_coef=float(tr.get("kl_coef", 0.0)),
        kl_discount_factor=float(tr.get("kl_discount_factor", 0.0)),
        eval_every=int(tr.get("eval_every", 20)),
        eval_n=int(tr.get("eval_n", 64)),
        save_every=int(tr.get("save_every", 50)),
        # opt-OUT: yaml's project (or a default) unless --no-wandb
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=args.experiment,
        chat_template_kwargs=(
            {"enable_thinking": bool(tr["enable_thinking"])} if "enable_thinking" in tr else None
        ),
    )
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
