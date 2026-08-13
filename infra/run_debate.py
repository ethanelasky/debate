"""Debate RL runner: experiment YAML -> DebateEnv + backend -> train loop.

Config shape (ONE agents block; the trained seat is marked inline — its
model_settings drive the training backend AND its sampling; `training:` holds
optimization/loop knobs only):

    math_pc:
      protocol: {turns: [...]}                # per-slot token budgets live HERE
      prompt_config: {file_path: ..., entry: ...}
      agents:
        alice:
          trained: true
          model_settings: {model_type: tinker, model_file_path: Qwen/Qwen3.5-4B,
                           enable_thinking: true,
                           sampling: {train: {temperature: 1.0, top_p: 1.0}}}
        bob:   {model_settings: {...}}       # frozen, built by the factory
        judge: {model_settings: {...}}
      judge_config: {schema_name: competitive, retries: 4}
      scoring: {scoring: continuous, confidence_source: json, shaping: [...]}
      dataset: {type: math, levels: [3, 4], relaxed_extraction: true}
      training: {lora_rank: 32,               # the SHARED adapter's rank
                 loss: {kind: ppo, clip_low: 0.8, clip_high: 1.2}, ppo_epochs: 1,
                 adv_length_norm: trajectory, steps: 100, batch_size: 8,
                 group_size: 4, lr: 1e-5, kl_coef: 0.02,
                 wandb_project: ..., eval_every: 5, ...}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from infra.backend.base import LossSpec, OptimParams, SamplingParams
from infra.config import load_experiment, parse_model_settings, reject_unknown_keys
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.protocol import Protocol
from infra.envs.tasks import get_family
from infra.models.base import ModelSettings, ModelType, resolved_sampling_profile
from infra.models.factory import instantiate_model
from infra.run_common import (
    TRAINING_KEYS,
    VERL_KEYS,
    apply_topology,
    build_backend,
    resolve_topology,
    run_identity_suffix,
    runner_parser,
    training_config_kwargs,
)
from infra.train import Config, resolve_protocol_identity, train, validate_resume_args

EXPERIMENT_KEYS = {
    "protocol",
    "prompt_config",
    "agents",
    "judge_config",
    "scoring",
    "dataset",
    "training",
    "fresh_positions",
    "flip",
    "first_speech_non_debate_aware",
    "plan_tokens",
}

AGENT_KEYS = {"trained", "model_settings"}


def _safe_docent_component(value: str, *, fallback: str) -> str:
    """Return a readable, traversal-safe component without slug collisions."""
    if (
        len(value.encode("utf-8")) <= 128
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    ):
        return value
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")[:80]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{slug or fallback}-{digest}"


def _docent_launch_id(pid: int | None = None) -> str:
    """Return an auditable identifier unique to a simultaneous process launch."""
    return f"pid-{os.getpid() if pid is None else pid}"


def _docent_run_dir(run_name: str, *, launch_id: str | None = None) -> str:
    """Return a stable Docent directory for one process launch of one run.

    Sweep arms share a working directory, so the run name must namespace their
    rollout files.  Unsafe or overlong names get a readable slug plus a digest:
    the digest prevents distinct names that sanitize alike from colliding.  A
    launch component prevents simultaneous commands with the same run name
    from overwriting identically numbered steps.
    """
    launch_id = _docent_launch_id() if launch_id is None else launch_id
    return os.path.join(
        "docent",
        _safe_docent_component(run_name, fallback="run"),
        _safe_docent_component(launch_id, fallback="launch"),
    )


def validate_experiment(exp: dict) -> None:
    """Reject config keys no runner reads, before anything is built.

    Without this a typo reads a default in silence — a misspelled
    `first_speech_non_debate_aware` trains the standard arm for 100 steps and
    reports it as the solo-context one. Nested blocks with typed constructors
    (judge_config, scoring, model_settings, dataset) already reject their own
    unknown keys; this covers the dict-shaped remainder.

    Error direction: adding a new `exp.get("new_key")` read to this file means
    adding "new_key" here too, or every run dies at launch with a one-line fix.
    """
    reject_unknown_keys(exp, EXPERIMENT_KEYS, "experiment")
    for speaker, agent in (exp.get("agents") or {}).items():
        if isinstance(agent, dict):
            reject_unknown_keys(agent, AGENT_KEYS, f"agents.{speaker}")
    tr = exp.get("training") or {}
    reject_unknown_keys(tr, TRAINING_KEYS, "training")
    reject_unknown_keys(tr.get("verl") or {}, VERL_KEYS, "training.verl")


def split_agents(exp: dict) -> tuple[dict[str, ModelSettings], dict[str, ModelSettings]]:
    """agents block -> (trained, frozen) settings by speaker."""
    trained: dict[str, ModelSettings] = {}
    frozen: dict[str, ModelSettings] = {}
    for speaker, agent in (exp.get("agents") or {}).items():
        settings = parse_model_settings(agent["model_settings"])
        (trained if agent.get("trained") else frozen)[speaker] = settings
    return trained, frozen


# training.backend -> the model_type trained seats must declare: verl serves
# rollouts through the local OpenAI shim, tinker samples through tinker.
BACKEND_MODEL_TYPES = {"tinker": ModelType.TINKER, "verl": ModelType.LOCAL}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_locator(value: object, *, field: str) -> str | None:
    """Reject URL forms that could smuggle credentials into identity hashes."""
    if value is None:
        return None
    locator = str(value)
    if "://" in locator:
        parsed = urlsplit(locator)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                f"{field} may not contain URL credentials, query parameters, or "
                "fragments when constructing protocol identity"
            )
    return locator


def _resolved_protocol(exp: dict) -> Protocol:
    proto_spec = exp["protocol"]
    if isinstance(proto_spec, str):
        proto_spec = exp["_protocols"][proto_spec]
    return Protocol.parse(proto_spec)


def _protocol_payload(protocol: Protocol) -> list[dict]:
    return [
        {
            "turn": cs.turn,
            "speaker": cs.speaker,
            "sequence": cs.seq,
            "name": cs.slot.name,
            "kind": cs.slot.kind.value,
            "visibility": cs.slot.visibility.value,
            "max_think_tokens": cs.slot.max_think_tokens,
            "max_visible_tokens": cs.slot.max_visible_tokens,
            "max_total_tokens": cs.slot.max_total_tokens,
        }
        for cs in protocol.compile()
    ]


def _model_payload(settings: ModelSettings) -> dict:
    train_sampling = resolved_sampling_profile(settings, "train")
    return {
        "model_type": settings.model_type.name.lower(),
        "model": _safe_locator(settings.model_file_path, field="model_file_path"),
        "alias": settings.alias,
        "resolved_max_new_tokens": settings.resolved_max_new_tokens,
        "resolved_reasoning_effort": settings.resolved_reasoning_effort,
        "resolved_thinking_budget_tokens": settings.resolved_thinking_budget_tokens,
        "enable_thinking": settings.enable_thinking,
        "capture_token_logprobs": settings.capture_token_logprobs,
        "require_token_logprobs": settings.require_token_logprobs,
        "provider_order": settings.provider_order,
        "quantizations": settings.quantizations,
        "allow_fallbacks": settings.allow_fallbacks,
        "data_collection": settings.data_collection,
        "base_url": _safe_locator(settings.base_url, field="base_url"),
        "train_sampling": train_sampling.model_dump(mode="json"),
    }


def _effective_protocol_training(tr: dict, args=None) -> dict:
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
    """Retain algorithm overrides while stripping credentials/output paths."""
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
    """Resolve the immutable learning algorithm, excluding operations only.

    Run length, peak learning rate, resume/checkpoint/output locations,
    W&B/logging, and save/eval cadence remain mutable. Ambiguous fields stay in
    the identity so a resume cannot silently change the optimization method.
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
        from infra.backend.verl import VerlBackendConfig

        # Match build_backend's exact topology-default/arm-override merge.
        # Only the already-classified immutable subset is selected below;
        # memory capacity and output locations remain operational.
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


def debate_protocol_identity(
    exp: dict,
    dataset_type: str,
    family,
    trained: dict[str, ModelSettings],
    frozen: dict[str, ModelSettings],
    *,
    args=None,
    topology: dict | None = None,
) -> dict[str, str]:
    """Family identity plus debate topology, prompts, models, and scoring."""
    base = resolve_protocol_identity(dataset_type, family)
    prompt = exp["prompt_config"]
    prompt_bytes = Path(prompt["file_path"]).read_bytes()
    protocol = _resolved_protocol(exp)
    tr = exp.get("training") or {}
    training_protocol = _effective_protocol_training(tr, args)
    plan_tokens = exp.get("plan_tokens")
    agents = {
        speaker: {"trained": speaker in trained, **_model_payload(settings)}
        for speaker, settings in sorted((trained | frozen).items())
    }
    payload = {
        "schema": "debate-runner-v2",
        "protocol": _protocol_payload(protocol),
        "prompt": {
            "entry": str(prompt["entry"]),
            "file_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        },
        "agents": agents,
        "learning": _effective_learning_protocol(tr, topology),
        "judge": JudgeConfig(**(exp.get("judge_config") or {})).model_dump(mode="json"),
        "scoring": ScoringConfig(**(exp.get("scoring") or {})).model_dump(mode="json"),
        "positions": {
            "fresh_positions": bool(exp.get("fresh_positions", True)),
            "flip": bool(exp.get("flip", False)),
            "first_speech_non_debate_aware": bool(
                exp.get("first_speech_non_debate_aware", False)
            ),
        },
        "solution_extraction": (
            "relaxed"
            if bool((exp.get("dataset") or {}).get("relaxed_extraction", True))
            else "strict"
        ),
        "direct_eval": {
            "wrapper": "planned" if plan_tokens is not None else "direct",
            "plan_tokens": None if plan_tokens is None else int(plan_tokens),
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": training_protocol["eval_max_tokens"],
            },
            **training_protocol,
        },
    }
    runner = {
        "runner_protocol": "debate-runner-v2",
        "runner_protocol_sha256": _canonical_sha256(payload),
        "runner_prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "runner_prompt_entry": str(prompt["entry"]),
    }
    collision = set(base) & set(runner)
    if collision:
        raise ValueError(f"family protocol identity collides with runner keys: {sorted(collision)}")
    return base | runner


def validate_trained_seats(trained: dict[str, ModelSettings], tr: dict) -> None:
    """Trained seats share ONE adapter and only contribute model_file_path to
    the backend, so a divergent path or a model_type that contradicts
    training.backend would otherwise pass in silence."""
    settings = list(trained.values())
    if not settings:
        return
    lead_path = settings[0].model_file_path
    paths = {s.model_file_path for s in settings}
    if len(paths) != 1:
        raise ValueError(
            f"trained seats must share one base model (one shared adapter): "
            f"expected every model_file_path to equal {lead_path!r}, got {sorted(map(str, paths))}"
        )
    backend_kind = str(tr.get("backend", "tinker"))
    expected = BACKEND_MODEL_TYPES.get(backend_kind)
    if expected is None:
        return  # build_backend rejects unknown kinds with its own message
    for speaker, s in trained.items():
        if s.model_type is not expected:
            got = s.model_type.name.lower() if isinstance(s.model_type, ModelType) else s.model_type
            raise ValueError(
                f"trained seat {speaker!r} declares model_type {got!r}, but "
                f"training.backend {backend_kind!r} requires {expected.name.lower()!r} "
                "for trained seats (the backend, not the model wrapper, serves them)"
            )


def debate_gen_budgets(protocol: Protocol, trained: dict[str, ModelSettings], tr: dict) -> dict:
    """Named generation budgets for build_backend's response_length
    cross-check. Only rollout-engine generations belong here: TRAINED seats'
    protocol slot caps plus the RLVR eval budget. Judge-seat slots (and any
    frozen debater's) are excluded — frozen seats generate on their own model
    server, never through the verl rollout engine, so response_length cannot
    truncate them and a large judge deliberation cap must not fail the build."""
    budgets: dict = {"training.eval_max_tokens": (tr or {}).get("eval_max_tokens")}
    for cs in protocol.compile():
        if cs.speaker not in trained:
            continue
        cap = cs.slot.max_total_tokens
        if cap is None:
            continue
        key = f"slot:{cs.slot.name}"
        # same-named slots across trained speakers: the check compares maxima,
        # so keeping the largest cap loses nothing
        budgets[key] = max(cap, budgets.get(key) or 0)
    return budgets


def build_env(exp: dict, trained: dict[str, ModelSettings], frozen: dict[str, ModelSettings]) -> DebateEnv:
    protocol = _resolved_protocol(exp)

    frozen_models = {
        speaker: instantiate_model(settings, is_debater=speaker != "judge", binding="train")
        for speaker, settings in frozen.items()
    }
    # FrozenSeat forwards these per predict call — the factory bakes sampling
    # into tinker seats only, so without this the local/API seats sample at
    # server defaults instead of the YAML profile.
    frozen_sampling = {
        speaker: resolved_sampling_profile(settings, "train") for speaker, settings in frozen.items()
    }
    decision = protocol.decision_slot
    judge_settings = frozen.get(decision.speaker) if decision is not None else None

    # Per-trained-seat sampling + template kwargs; trained seats must sample
    # UNBIASED (temp/top_p 1.0) — anything else corrupts the ratio anchor
    # (the old repo's §6 gate, kept hard).
    # Token budgets are the PROTOCOL's job (per-slot caps); the seat's base
    # max_tokens is just a ceiling derived from its largest slot cap.
    caps_by_speaker: dict[str, list] = {}
    for cs in protocol.compile():
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
                f"trained seat {speaker!r} has protocol slot(s) without max_total_tokens; "
                "set per-slot budgets in the protocol (speech budgets are format-specific)"
            )
        trained_sampling[speaker] = SamplingParams(max_tokens=max(caps), temperature=temp, top_p=top_p)
        if settings.enable_thinking is not None:
            trained_chat_kwargs[speaker] = {"enable_thinking": bool(settings.enable_thinking)}

    ds = dict(exp.get("dataset") or {})
    family = get_family(ds.pop("type", None))
    try:
        config = DebateEnvConfig(
            protocol=protocol,
            prompt_file=exp["prompt_config"]["file_path"],
            prompt_entry=exp["prompt_config"]["entry"],
            trained_speakers=list(trained),
            frozen_models=frozen_models,
            trained_sampling=trained_sampling,
            trained_chat_kwargs=trained_chat_kwargs,
            frozen_sampling=frozen_sampling,
            judge_model_settings=judge_settings,
            judge=JudgeConfig(**(exp.get("judge_config") or {})),
            scoring=ScoringConfig(**(exp.get("scoring") or {})),
            fresh_positions=exp.get("fresh_positions", True),
            flip=exp.get("flip", False),
            first_speech_non_debate_aware=bool(
                exp.get("first_speech_non_debate_aware", False)
            ),
        )
        relaxed = bool(ds.pop("relaxed_extraction", True))
        task_source = family.source(ds)
        return DebateEnv(config, task_source, family, relaxed_extraction=relaxed)
    except BaseException:
        family.close()
        raise


def _main(cleanups: list) -> None:
    args = runner_parser(__doc__).parse_args()
    validate_resume_args(args)

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels
    trained, frozen = split_agents(exp)
    if not trained:
        raise ValueError("no agent has trained: true")
    trained_settings = list(trained.values())
    tr = exp.get("training") or {}
    validate_trained_seats(trained, tr)

    dataset_type = (exp.get("dataset") or {}).get("type")
    env = build_env(exp, trained, frozen)
    cleanups.append(env.family.close)
    topology = resolve_topology()
    protocol_identity = debate_protocol_identity(
        exp, dataset_type, env.family, trained, frozen, args=args, topology=topology
    )
    lead = trained_settings[0]

    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    backend = build_backend(
        tr,
        lead.model_file_path,
        run_name,
        lr_override=args.lr,
        load_given=bool(args.load),
        gen_budgets=debate_gen_budgets(env.protocol, trained, tr),
        topology=topology,
    )
    if args.load:
        backend.load(args.load)

    profile = resolved_sampling_profile(lead, "train")
    cfg = Config(
        checkpoint_dir=getattr(getattr(backend, "config", None), "checkpoint_dir", None),
        base_model=lead.model_file_path,
        sampling=SamplingParams(
            # no ceiling here: budgets are the protocol's per-slot caps, and
            # Policy hard-errors on any generation left unbounded
            max_tokens=None,
            temperature=profile.temperature if profile.temperature is not None else 1.0,
            top_p=profile.top_p if profile.top_p is not None else 1.0,
        ),
        # The TEAM entity, not the API key's personal default namespace —
        # without it, runs land where the rest of the team cannot see them.
        wandb_entity=(None if args.no_wandb else args.wandb_entity or tr.get("wandb_entity")),
        wandb_project=(
            None if args.no_wandb else args.wandb_project or tr.get("wandb_project") or "debate"
        ),
        run_name=run_name,
        protocol_identity=protocol_identity,
        chat_template_kwargs=(
            {"enable_thinking": bool(lead.enable_thinking)} if lead.enable_thinking is not None else None
        ),
        **training_config_kwargs(tr, args),
    )
    # Comprehensive docent capture: every training rollout's debates -> JSONL.
    # The run-specific directory prevents concurrent sweep arms from overwriting
    # one another's identically numbered steps.
    launch_id = _docent_launch_id()
    docent_dir = _docent_run_dir(run_name, launch_id=launch_id)

    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import agent_runs, export_jsonl

        os.makedirs(docent_dir, exist_ok=True)
        path = os.path.join(docent_dir, f"step-{step:05d}.jsonl")
        export_jsonl(agent_runs(env_), path)

    cfg.on_rollout = _export_docent

    # RLVR evals: measure proposal accuracy on the held-out split via the
    # task source itself (MathEnv), not a debate. plan_tokens wraps the eval in
    # the same plan-then-answer shape as the protocol's pre-proposal plan slot,
    # so the policy is measured in the mode it is trained in (set it to that
    # slot's max_total_tokens).
    eval_env = env.task_source
    plan_tokens = exp.get("plan_tokens")
    if plan_tokens is not None:
        from infra.envs.planned import PlannedEnv

        eval_env = PlannedEnv(eval_env, int(plan_tokens))
    train(env, backend, cfg, eval_env=eval_env)


def main() -> None:
    cleanups: list = []
    try:
        _main(cleanups)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


if __name__ == "__main__":
    main()
