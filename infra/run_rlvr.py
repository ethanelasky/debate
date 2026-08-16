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

import hashlib
import json
import os
from urllib.parse import urlsplit

from infra.backend.base import LossSpec, OptimParams, SamplingParams
from infra.config import load_experiment, reject_unknown_keys
from infra.envs.tasks import get_family
from infra.launch_namespace import (
    claim_directory,
    resolve_launch_namespace,
    safe_path_component,
)
from infra.run_common import (
    TRAINING_KEYS,
    VERL_KEYS,
    acquire_wandb_resume_cli_lease,
    apply_wandb_resume_cli_authorization,
    apply_topology,
    build_backend,
    resolve_topology,
    release_wandb_resume_cli_lease,
    run_identity_suffix,
    runner_parser,
    training_config_kwargs,
)
from infra.train import Config, resolve_protocol_identity, train, validate_resume_args

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
    # unbiased-ratio anchor — when omitted. Tempered profiles are re-anchored
    # in the train loop (backend.forward overwrite; see infra/train.py), since
    # vLLM's sampler logprobs come from raw logits and would sit off the
    # training engine's tempered recompute.
    "temperature",
    "top_p",
    # Floor on generated tokens (vLLM min_tokens; hw4 parity uses 8).
    "min_completion_tokens",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_model_locator(value: object) -> str | None:
    """Return a model identifier only when it cannot carry URL credentials.

    Model IDs and local checkpoint paths are scientific inputs.  URL userinfo,
    queries, and fragments are a different matter: they commonly contain API
    keys.  Refuse those forms instead of making a secret-derived hash part of
    the immutable W&B identity.
    """
    if value is None:
        return None
    locator = str(value)
    if "://" in locator:
        parsed = urlsplit(locator)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "model identifiers in protocol identity may not contain URL "
                "credentials, query parameters, or fragments"
            )
    return locator


def _effective_protocol_training(tr: dict, args=None) -> dict:
    """Training-loop fields that alter what is rolled out or evaluated.

    Continuation and operational plumbing (steps, lr, checkpoints, logging,
    save/eval cadence) is intentionally absent.  In particular, extending a
    run or changing its continuation LR must not manufacture a new protocol.
    """
    def configured(key: str, default):
        value = tr.get(key)
        return default if value is None else value

    batch_size = (
        getattr(args, "batch_size", None)
        if args is not None and getattr(args, "batch_size", None) is not None
        else configured("batch_size", Config.batch_size)
    )
    group_size = (
        getattr(args, "group_size", None)
        if args is not None and getattr(args, "group_size", None) is not None
        else configured("group_size", Config.group_size)
    )
    return {
        "batch_size": int(batch_size),
        "group_size": int(group_size),
        "dynamic_sampling_retries": int(
            configured("dynamic_sampling_retries", Config.dynamic_sampling_retries)
        ),
        "oversample_factor": float(configured("oversample_factor", Config.oversample_factor)),
        "rl_seed": int(configured("rl_seed", Config.seed)),
        "eval_n": int(configured("eval_n", Config.eval_n)),
        "eval_split": str(configured("eval_split", Config.eval_split)),
        "final_test_eval": bool(configured("final_test_eval", Config.final_test_eval)),
        "eval_max_tokens": (
            None
            if tr.get("eval_max_tokens", Config.eval_max_tokens) is None
            else int(tr["eval_max_tokens"])
        ),
    }


def _redacted_verl_overrides(values: object) -> list[str]:
    """Keep arbitrary algorithm overrides without retaining credentials/paths.

    ``extra_overrides`` is an escape hatch and can change the optimizer or
    loss, so dropping it would make the identity incomplete.  It can also be
    (mis)used for secrets or machine-local paths.  Preserve order because
    Hydra's last duplicate wins, but replace sensitive values before hashing.
    """
    out: list[str] = []
    secret_markers = (
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "bearer_token",
        "secret",
        "password",
        "credential",
    )
    operational_path_markers = (
        "checkpoint", "output_dir", "save_dir", "log_dir", "logging_dir", "wandb_dir"
    )
    for raw in values or ():
        item = str(raw)
        if "=" not in item:
            out.append(item)
            continue
        key, value = item.split("=", 1)
        normalized = key.lstrip("+").lower()
        if any(marker in normalized for marker in secret_markers) or normalized.endswith(
            (".token", "_token", "-token")
        ):
            value = "<redacted>"
        elif os.path.isabs(value) and any(
            marker in normalized for marker in operational_path_markers
        ):
            value = "<operational-path>"
        elif "://" in value:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                value = "<redacted-url>"
        out.append(f"{key}={value}")
    return out


def _effective_learning_protocol(tr: dict, topology: dict | None = None) -> dict:
    """Resolve the immutable optimization contract from runner configuration.

    Only explicitly operational/continuation knobs are excluded: run length,
    peak learning rate, resume/checkpoint/output locations, W&B/logging, and
    save/eval cadence.  When a field's scientific status is debatable, it is
    retained: false-positive resume refusal is safer than silently continuing
    under a different learning algorithm.
    """
    def configured(key: str, default):
        value = tr.get(key)
        return default if value is None else value

    loss = LossSpec(**(tr.get("loss") or {}))
    optim = OptimParams(lr=Config.lr)
    backend = str(tr.get("backend", "tinker"))
    payload = {
        "backend": backend,
        "lora_rank": int(configured("lora_rank", Config.lora_rank)),
        "loss": {
            "kind": loss.kind,
            "clip_low": float(loss.clip_low),
            "clip_high": float(loss.clip_high),
            # Neither supported backend adds an entropy loss today. Record the
            # effective value so adding one cannot happen outside this schema.
            "entropy_coefficient": 0.0,
        },
        "updates": {
            "micro_batch": int(configured("micro_batch", Config.micro_batch)),
            "ppo_epochs": int(configured("ppo_epochs", Config.ppo_epochs)),
        },
        "advantages": {
            "norm_by_std": bool(configured("norm_adv_by_std", Config.norm_adv_by_std)),
            "population_std": bool(
                configured("adv_population_std", Config.adv_population_std)
            ),
            "length_norm": str(configured("adv_length_norm", Config.adv_length_norm)),
            "drop_zero": bool(
                configured("drop_zero_advantage", Config.drop_zero_advantage)
            ),
        },
        "kl": {
            "coefficient": float(configured("kl_coef", Config.kl_coef)),
            "mechanism": str(configured("kl_mechanism", Config.kl_mechanism)),
            "discount_factor": float(
                configured("kl_discount_factor", Config.kl_discount_factor)
            ),
        },
        "optimizer": {
            # Peak lr is deliberately mutable for continuations/sweeps. These
            # remaining AdamW/clip settings are fixed runner defaults.
            "betas": list(optim.betas),
            "eps": float(optim.eps),
            "weight_decay": float(optim.weight_decay),
            "grad_clip": float(optim.grad_clip),
            "warmup_steps": int(configured("warmup_steps", Config.warmup_steps)),
            "lr_schedule": str(configured("lr_schedule", Config.lr_schedule)),
            "min_lr_ratio": float(configured("min_lr_ratio", Config.min_lr_ratio)),
        },
    }
    if backend == "verl":
        # Importing this module is dependency-light; verl/ray imports remain
        # local to backend construction. These are the resolved algorithm and
        # numerical-execution settings, not the checkpoint or GPU-memory knobs.
        from infra.backend.verl import VerlBackendConfig

        # Identity must describe the same effective settings build_backend
        # receives: topology supplies defaults, while the arm wins. Capacity
        # and output-only keys remain excluded below.
        v = apply_topology(dict(tr.get("verl") or {}), topology or {})
        payload["verl"] = {
            "strategy": str(v.get("strategy", VerlBackendConfig.strategy)),
            "n_gpus": int(v.get("n_gpus", VerlBackendConfig.n_gpus)),
            "prompt_length": int(v.get("prompt_length", VerlBackendConfig.prompt_length)),
            "response_length": int(v.get("response_length", VerlBackendConfig.response_length)),
            "max_token_len_per_gpu": int(
                v.get("max_token_len_per_gpu", VerlBackendConfig.max_token_len_per_gpu)
            ),
            "rollout_tp": int(v.get("rollout_tp", VerlBackendConfig.rollout_tp)),
            "use_remove_padding": bool(
                v.get("use_remove_padding", VerlBackendConfig.use_remove_padding)
            ),
            "extra_overrides": _redacted_verl_overrides(v.get("extra_overrides")),
        }
    return payload


def rlvr_protocol_identity(
    exp: dict, dataset_type: str, family, *, args=None, topology: dict | None = None
) -> dict[str, str]:
    """Family identity plus the exact RLVR rollout/evaluation contract."""
    base = resolve_protocol_identity(dataset_type, family)
    tr = exp.get("training") or {}
    if exp.get("max_completion_tokens") is None:
        raise ValueError("set max_completion_tokens (per-generation budget) in the experiment")
    answer_tokens = int(exp["max_completion_tokens"])
    plan_tokens = exp.get("plan_tokens")
    think_tokens = exp.get("think_tokens")
    if plan_tokens is not None:
        structure = "planned"
        sampler_cap = max(answer_tokens, int(plan_tokens))
    elif think_tokens is not None:
        structure = "native_think"
        sampler_cap = answer_tokens + int(think_tokens)
    else:
        structure = "direct"
        sampler_cap = answer_tokens

    training_protocol = _effective_protocol_training(tr, args)
    eval_cap = training_protocol["eval_max_tokens"]
    payload = {
        "schema": "rlvr-runner-v2",
        "model": _safe_model_locator(exp.get("model")),
        "learning": _effective_learning_protocol(tr, topology),
        "rollout": {
            "structure": structure,
            "answer_max_tokens": answer_tokens,
            "plan_tokens": None if plan_tokens is None else int(plan_tokens),
            "think_tokens": None if think_tokens is None else int(think_tokens),
            "sampler": {
                "max_tokens": sampler_cap,
                "min_tokens": (
                    None
                    if exp.get("min_completion_tokens") is None
                    else int(exp["min_completion_tokens"])
                ),
                "temperature": float(exp.get("temperature", 1.0)),
                "top_p": float(exp.get("top_p", 1.0)),
            },
            "enable_thinking": (
                None
                if exp.get("enable_thinking") is None
                else bool(exp["enable_thinking"])
            ),
        },
        "evaluation": {
            "environment": structure,
            "sampling": {
                "max_tokens": sampler_cap if eval_cap is None else eval_cap,
                "min_tokens": None,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            **training_protocol,
        },
    }
    runner = {
        "runner_protocol": "rlvr-runner-v2",
        "runner_rollout_structure": structure,
        "runner_protocol_sha256": _canonical_sha256(payload),
    }
    collision = set(base) & set(runner)
    if collision:
        raise ValueError(f"family protocol identity collides with runner keys: {sorted(collision)}")
    return base | runner


def _sampling_params(exp: dict, total_budget: int) -> SamplingParams:
    """The run's rollout sampling profile — extracted so tests can pin the
    experiment-key -> SamplingParams hop through the same function main()
    uses (a hand-built mirror in the test pinned nothing; round-4 audit)."""
    return SamplingParams(
        max_tokens=total_budget,
        min_tokens=(
            int(exp["min_completion_tokens"])
            if exp.get("min_completion_tokens") is not None
            else None
        ),
        temperature=float(exp.get("temperature", 1.0)),
        top_p=float(exp.get("top_p", 1.0)),
    )


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


def _main(cleanups: list) -> None:
    args = runner_parser(__doc__).parse_args()
    validate_resume_args(args)
    launch_namespace = resolve_launch_namespace()
    resume_lease_handoff = acquire_wandb_resume_cli_lease(args)
    try:
        _main_after_resume_frontier(
            cleanups, args, launch_namespace, resume_lease_handoff
        )
    finally:
        release_wandb_resume_cli_lease(resume_lease_handoff)


def _main_after_resume_frontier(
    cleanups: list,
    args,
    launch_namespace: str,
    resume_lease_handoff,
) -> None:

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels

    tr = exp.get("training") or {}
    model_path = str(exp["model"])
    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    config_kwargs = training_config_kwargs(tr, args)
    docent_dir = str(
        claim_directory(
            os.path.join(
                "docent",
                safe_path_component(run_name, fallback="run"),
                launch_namespace,
            )
        )
    )
    transcript_dir = None
    if config_kwargs.get("log_transcripts", Config.log_transcripts):
        transcript_dir = str(
            claim_directory(
                os.path.join(
                    "transcripts",
                    safe_path_component(run_name, fallback="run"),
                    launch_namespace,
                )
            )
        )

    ds = dict(exp.get("dataset") or {})
    dataset_type = ds.pop("type", None)
    family = get_family(dataset_type)
    # Register immediately: source() may start family-owned workers and may
    # itself fail partway through construction.
    cleanups.append(family.close)
    ds.pop("relaxed_extraction", None)  # debate-only knob; source envs score both
    env = family.source(ds)
    topology = resolve_topology()
    protocol_identity = rlvr_protocol_identity(
        exp, dataset_type, family, args=args, topology=topology
    )
    source_env = env  # the task-source env keeps ownership of reward_sample
    # plan_tokens: two-turn plan-then-answer rollouts (train AND eval — one
    # env, one rollout path), matching the debate arm's pre-solution scratchpad
    # slot. See infra/envs/planned.py.
    plan_tokens = exp.get("plan_tokens")

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
        topology=topology,
        launch_namespace=launch_namespace,
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
        sampling=_sampling_params(exp, total_budget),
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
        launch_namespace=launch_namespace,
        transcript_dir=transcript_dir,
        protocol_identity=protocol_identity,
        **config_kwargs,
    )
    apply_wandb_resume_cli_authorization(
        cfg, args, resume_lease_handoff=resume_lease_handoff
    )

    # Comprehensive docent capture, single-turn twin of run_debate's: every
    # training rollout's kept samples lands under the same immutable launch
    # namespace used by checkpoints, transcripts and W&B provenance.
    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import export_jsonl_claimed
        from infra.envs.singleturn_docent import agent_runs

        records = getattr(env_, "last_rollout_records", None)
        if not records:
            raise RuntimeError(
                "requested RLVR rollout retained no records (including an "
                "all-fidelity-dropped rollout); refusing missing local Docent evidence"
            )
        export_jsonl_claimed(
            agent_runs(records), docent_dir, f"step-{step:05d}.jsonl"
        )

    cfg.on_rollout = _export_docent
    train(env, backend, cfg)


def main() -> None:
    cleanups: list = []
    try:
        _main(cleanups)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


if __name__ == "__main__":
    main()
