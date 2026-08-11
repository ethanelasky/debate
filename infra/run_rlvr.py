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

`think_tokens` (optional; requires enable_thinking: true, mutually exclusive
with plan_tokens) is the thinking-model twin of plan_tokens: instead of a
scratchpad TURN, the model's native private <think> phase is the sequential
compute. Generation stays single-turn but runs through budget-forced sampling
(infra/envs/base.py): the think phase is HARD-CAPPED at think_tokens —
</think> force-injected at the cap, masked from training — and the visible
answer continues under max_completion_tokens. One SlotLimits on the source env
covers train AND eval rollouts. dataset.think_overshoot_penalty (math/aime)
prices a force-closed think phase in the reward.

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
    resolve_topology,
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
    "think_tokens",
    "dataset",
    "training",
    # Rollout sampling profile (CS285-hw4 validation arms sample at 0.8/0.95
    # like the reference implementation). Default 1.0/1.0 — the debate arms'
    # unbiased-ratio anchor — when omitted. The backend recomputes logprobs at
    # the sampling temperature (VerlBackend._last_temperature), so a tempered
    # profile is ratio-safe.
    "temperature",
    "top_p",
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
        # An identical render has two meanings. An ALWAYS-thinking model
        # (Olmo Think line: the template opens <think> unconditionally, no
        # toggle exists) is fine when the config asked for thinking — the
        # behavior matches intent even though the knob is inert. A template
        # with no think mode at all is the silent-drop hazard this guard
        # exists for. The rendered generation prompt separates them.
        rendered_text = tokenizer.apply_chat_template(
            probe, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        if enable_thinking and "<think>" in rendered_text:
            return
        raise ValueError(
            f"enable_thinking: {str(enable_thinking).lower()} was set, but this model's chat "
            "template renders identically with enable_thinking True and False — the setting "
            "would be silently dropped. Remove the key (models without a thinking mode need "
            "no toggle) or point the arm at a hybrid-thinking model. (Always-thinking "
            "templates whose generation prompt opens <think> pass this check when "
            "enable_thinking is true.)"
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

    think_tokens = exp.get("think_tokens")
    if think_tokens is not None:
        if exp.get("plan_tokens") is not None:
            raise ValueError(
                "think_tokens and plan_tokens are mutually exclusive: the native "
                "<think> phase IS the private scratchpad, so a plan turn on top "
                "would double the sequential compute the arms exist to compare. "
                "Drop one."
            )
        if int(think_tokens) <= 0:
            raise ValueError(f"think_tokens must be a positive token budget, got {think_tokens!r}")
        if exp.get("enable_thinking") is not True:
            raise ValueError(
                "think_tokens hard-caps the model's native think phase, which the "
                "chat template only opens with enable_thinking: true — set the key "
                "explicitly, or the cap would police a channel that never opens."
            )
        emt = tr.get("eval_max_tokens")
        mct = exp.get("max_completion_tokens")
        if emt is not None and mct is not None and int(emt) < int(think_tokens) + int(mct):
            raise ValueError(
                f"training.eval_max_tokens={emt} is below think_tokens + "
                f"max_completion_tokens = {int(think_tokens) + int(mct)}: eval "
                "rollouts run under the same think/total SlotLimits as training, "
                "and a smaller eval ceiling silently SHRINKS the think cap "
                "(budget-forced sampling takes the min). Raise it, or drop the "
                "key so eval samples under the training budget."
            )


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
    source_env = env  # the task-source env keeps ownership of reward_sample
    # plan_tokens: two-turn plan-then-answer rollouts (train AND eval — one
    # env, one rollout path), matching the debate arm's pre-solution scratchpad
    # slot. See infra/envs/planned.py.
    plan_tokens = exp.get("plan_tokens")

    tr = exp.get("training") or {}
    model_path = str(exp["model"])
    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )

    max_tokens = exp.get("max_completion_tokens")
    if max_tokens is None:
        raise ValueError("set max_completion_tokens (per-generation budget) in the experiment")

    # think_tokens: native-<think> single-turn rollouts (validated against
    # plan_tokens/enable_thinking in validate_experiment). One SlotLimits on
    # the SOURCE env covers train AND eval — both run through
    # SingleTurnEnv.rollout — and routes generation through
    # budget_forced_sample. The per-request total is think + answer: that sum
    # is what the sampler ceiling must allow (Policy.predict clamps to
    # min(params.max_tokens, max_total_tokens) — a bare max_completion_tokens
    # ceiling would silently shrink the think cap) and what the rollout
    # engine must fit in one request (gen_budgets vs verl.response_length).
    think_tokens = exp.get("think_tokens")
    total_budget = int(max_tokens)
    # plan_tokens: the PLAN turn is the largest single generation, so the
    # sampler ceiling must be raised to it — Policy.predict min()-clamps slot
    # limits against params.max_tokens, which is exactly how the 8k plan
    # silently shrank to max_completion_tokens (1000) in every pre-fix plan
    # arm (2026-08-08). The answer turn keeps its own explicit cap instead.
    if plan_tokens is not None:
        from infra.envs.planned import PlannedEnv

        env = PlannedEnv(env, int(plan_tokens), answer_max_tokens=int(max_tokens))
        total_budget = max(int(max_tokens), int(plan_tokens))
    gen_budgets = {
        "max_completion_tokens": max_tokens,
        "training.eval_max_tokens": tr.get("eval_max_tokens"),
    }
    if plan_tokens is not None:
        gen_budgets["plan_tokens"] = int(plan_tokens)
    if think_tokens is not None:
        from infra.envs.base import SlotLimits

        total_budget = int(think_tokens) + int(max_tokens)
        source_env.slot_limits = SlotLimits(
            max_think_tokens=int(think_tokens), max_total_tokens=total_budget
        )
        gen_budgets["think_tokens + max_completion_tokens"] = total_budget

    backend = build_backend(
        tr,
        model_path,
        run_name,
        lr_override=args.lr,
        load_given=bool(args.load),
        gen_budgets=gen_budgets,
        topology=resolve_topology(),
    )
    if args.load:
        backend.load(args.load)

    # Tokenizer injection, attribute convention: envs have no tokenizer of
    # their own; a logprob-reading reward (MB's dataset.answer_conf_coeff)
    # needs one to map completion tokens to char offsets. Set on the SOURCE
    # env — the object whose reward_sample reads it — not the PlannedEnv
    # wrapper, so no attribute forwarding is involved (least-magic option).
    # Envs that never read `tokenizer` simply carry an unused attribute.
    source_env.tokenizer = backend.tokenizer

    enable_thinking = exp.get("enable_thinking")
    if enable_thinking is not None:
        _check_thinking_toggle(backend.tokenizer, bool(enable_thinking))

    cfg = Config(
        checkpoint_dir=getattr(getattr(backend, "config", None), "checkpoint_dir", None),
        base_model=model_path,
        # total_budget, not max_completion_tokens: with think_tokens set the
        # one generation carries think AND answer, and Policy.predict treats
        # this as the ceiling over the SlotLimits total.
        sampling=SamplingParams(
            max_tokens=total_budget,
            temperature=float(exp.get("temperature", 1.0)),
            top_p=float(exp.get("top_p", 1.0)),
        ),
        chat_template_kwargs=(
            None if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
        ),
        # The TEAM entity, not the API key's personal default namespace —
        # without it, runs land where the rest of the team cannot see them.
        wandb_entity=(None if args.no_wandb else args.wandb_entity or tr.get("wandb_entity")),
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=run_name,
        **training_config_kwargs(tr, args),
    )

    # Comprehensive docent capture, single-turn twin of run_debate's: every
    # training rollout's kept samples -> docent/<run>/step-NNNNN.jsonl. The run
    # name carries the sweep suffix, for the same reason checkpoints do: arms of
    # one sweep run concurrently from the same working directory, and a shared
    # docent/step-NNNNN.jsonl would have them overwrite each other's rollouts
    # step for step, leaving one file per step drawn from whichever arm wrote last.
    docent_dir = os.path.join("docent", run_name)

    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import export_jsonl
        from infra.envs.singleturn_docent import agent_runs

        records = getattr(env_, "last_rollout_records", None)
        if not records:
            return
        os.makedirs(docent_dir, exist_ok=True)
        export_jsonl(agent_runs(records), os.path.join(docent_dir, f"step-{step:05d}.jsonl"))

    cfg.on_rollout = _export_docent
    train(env, backend, cfg)


if __name__ == "__main__":
    main()
