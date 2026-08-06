"""RLVR runner: experiment YAML -> task-source env (direct verifiable reward)
-> train loop. The judge-free baseline for the debate experiments: same model,
same backend, same GRPO machinery — reward comes from answer verification
instead of a judge, so it isolates optimization from reward quality.

Config shape:

    math_rlvr_olmo_l5:
      model: allenai/Olmo-3-7B-Instruct-DPO
      max_completion_tokens: 2000
      dataset: {type: math, levels: 5, seed: 0}
      training: {backend: verl, verl: {...}, lora_rank: 32, loss: {...},
                 steps: 100, batch_size: 8, group_size: 8, lr: 1e-4, ...}

`enable_thinking` (optional) is the debate runner's per-seat knob at RLVR's one
policy: it becomes Config.chat_template_kwargs, so it reaches training AND eval
rollouts through the same Policy. Hybrid-thinking models (Qwen3.x) render an
empty <think></think> block at False and open <think> at True; the toggle is
verified against the tokenizer at launch (see _check_thinking_toggle) rather
than trusted, because a template that ignores the kwarg drops it in silence —
the same latent failure the OpenRouter seats hit (MB_DEBATE_PLAN 2026-08-02).
Omit the key for models without a thinking mode.

MB blind choice (dataset.type: monitoringbench) needs nothing extra: the task
source renders its RLVR prompt from the family's answer-generation config
(infra/prompts/tasks/monitoringbench.yaml, dataset.prompt_file to override) —
the same file the eval arm's debate packs splice their blind view from, so the
RLVR prompt is byte-identical to the eval arm's blind view by construction.
"""

from __future__ import annotations

import os

from infra.backend.base import SamplingParams
from infra.config import load_experiment, reject_unknown_keys
from infra.envs.tasks import get_family
from infra.run_common import (
    TRAINING_KEYS,
    VERL_KEYS,
    build_backend,
    run_identity_suffix,
    runner_parser,
    training_config_kwargs,
)
from infra.train import Config, train

EXPERIMENT_KEYS = {
    "model",
    "enable_thinking",
    "max_completion_tokens",
    "plan_tokens",
    "dataset",
    "training",
}


def _check_thinking_toggle(tokenizer, enable_thinking: bool) -> None:
    """Fail unless the policy's chat template actually reads enable_thinking.

    apply_chat_template passes unknown kwargs into the Jinja globals, where a
    template that never references them ignores them without a word — so a
    config declaring a no-think arm would train a thinking one and nothing in
    the logs would say so. Rendering the same probe both ways separates the two
    cases: a template that honors the knob produces different tokens.
    """
    probe = [{"role": "user", "content": "probe"}]

    def render(value: bool):
        out = tokenizer.apply_chat_template(
            probe, add_generation_prompt=True, tokenize=True, enable_thinking=value
        )
        if not isinstance(out, list):
            out = out["input_ids"]
        return out[0] if out and isinstance(out[0], list) else out

    if render(True) == render(False):
        raise ValueError(
            f"enable_thinking: {str(enable_thinking).lower()} was set, but this model's chat "
            "template renders identically with enable_thinking True and False — the setting "
            "would be silently dropped. Remove the key (models without a thinking mode need "
            "no toggle) or point the arm at a hybrid-thinking model."
        )


def validate_experiment(exp: dict) -> None:
    """Reject config keys this runner never reads, before anything is built.

    Same contract as run_debate.validate_experiment: a typo must fail at
    launch instead of silently falling back to a default. Adding a new
    `exp.get(...)`/`tr.get(...)` read means adding the key here (or to
    run_common.TRAINING_KEYS / VERL_KEYS, which both runners share).
    """
    reject_unknown_keys(exp, EXPERIMENT_KEYS, "rlvr experiment")
    tr = exp.get("training") or {}
    reject_unknown_keys(tr, TRAINING_KEYS, "training")
    reject_unknown_keys(tr.get("verl") or {}, VERL_KEYS, "training.verl")


def main() -> None:
    args = runner_parser(__doc__).parse_args()

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels

    ds = dict(exp.get("dataset") or {})
    family = get_family(ds.pop("type", None))
    ds.pop("relaxed_extraction", None)  # debate-only knob; source envs score both
    env = family.source(ds)
    # plan_tokens: two-turn plan-then-answer rollouts (train AND eval — one
    # env, one rollout path), matching the debate arm's pre-solution scratchpad
    # slot. See infra/envs/planned.py.
    plan_tokens = exp.get("plan_tokens")
    if plan_tokens is not None:
        from infra.envs.planned import PlannedEnv

        env = PlannedEnv(env, int(plan_tokens))

    tr = exp.get("training") or {}
    model_path = str(exp["model"])
    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    backend = build_backend(
        tr, model_path, run_name, lr_override=args.lr, load_given=bool(args.load)
    )
    if args.load:
        backend.load(args.load)

    max_tokens = exp.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError("set max_completion_tokens (per-generation budget) in the experiment")

    enable_thinking = exp.get("enable_thinking")
    if enable_thinking is not None:
        _check_thinking_toggle(backend.tokenizer, bool(enable_thinking))

    cfg = Config(
        base_model=model_path,
        sampling=SamplingParams(max_tokens=int(max_tokens), temperature=1.0, top_p=1.0),
        chat_template_kwargs=(
            None if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
        ),
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=run_name,
        **training_config_kwargs(tr, args),
    )

    # Comprehensive docent capture, single-turn twin of run_debate's: every
    # training rollout's kept samples -> docent/step-NNNNN.jsonl.
    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import export_jsonl
        from infra.envs.singleturn_docent import agent_runs

        records = getattr(env_, "last_rollout_records", None)
        if not records:
            return
        os.makedirs("docent", exist_ok=True)
        export_jsonl(agent_runs(records), f"docent/step-{step:05d}.jsonl")

    cfg.on_rollout = _export_docent
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
