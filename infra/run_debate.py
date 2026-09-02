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
import os
from pathlib import Path

from infra.backend.base import SamplingParams
from infra.backend.vllm_readonly import VLLMCompletionsBackend
from infra.config import load_experiment, parse_model_settings, reject_unknown_keys
from infra.envs.base import Policy
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.protocol import Protocol
from infra.envs.tasks import get_family
from infra.models.base import ModelSettings, ModelType, resolved_sampling_profile
from infra.models.factory import instantiate_model
from infra.launch_namespace import (
    claim_directory,
    resolve_launch_namespace,
    safe_path_component,
    validate_launch_namespace,
)
from infra.local_metrics_evidence import scheduler_artifact_attempt_root
from infra.run_common import (
    EVAL_SLOT_LIMIT_KEYS,
    TRAINING_KEYS,
    VERL_KEYS,
    acquire_wandb_resume_cli_lease,
    apply_wandb_resume_cli_authorization,
    build_backend,
    build_eval_source,
    canonical_sha256,
    effective_learning_protocol,
    effective_protocol_training,
    eval_slot_limits,
    prepare_eval_env,
    release_wandb_resume_cli_lease,
    resolve_topology,
    run_identity_suffix,
    runner_parser,
    safe_locator,
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
    "eval_dataset",
    "eval_slot_limits",
    "training",
    "fresh_positions",
    "flip",
    "first_speech_non_debate_aware",
    "plan_tokens",
}

AGENT_KEYS = {"trained", "frozen_policy", "model_settings"}


def _docent_run_dir(run_name: str, *, launch_namespace: str) -> str:
    """Return the Docent directory for one immutable launch of one run.

    Sweep arms share a working directory, so the run name must namespace their
    rollout files.  Unsafe or overlong names get a readable slug plus a digest:
    the digest prevents distinct names that sanitize alike from colliding. The
    already-validated launch namespace is preserved byte-for-byte. Scheduler
    runs put the namespace above ``docent/`` in their retained attempt root;
    manual runs retain the historical ``docent/<run>/<namespace>`` layout.
    """
    namespace = validate_launch_namespace(launch_namespace)
    attempt_root = scheduler_artifact_attempt_root(namespace)
    if attempt_root is not None:
        return os.path.join(
            attempt_root,
            "docent",
            safe_path_component(run_name, fallback="run"),
        )
    return os.path.join(
        "docent", safe_path_component(run_name, fallback="run"), namespace
    )


def validate_experiment(exp: dict) -> None:
    """Reject config keys no runner reads, before anything is built.

    Without this a typo reads a default in silence — a misspelled
    `first_speech_non_debate_aware` can train a debate-aware opening for 100
    steps while reporting it as the task-only one. Nested blocks with typed
    constructors (judge_config, scoring, model_settings, dataset) already
    reject their own unknown keys; this covers the dict-shaped remainder.

    Error direction: adding a new `exp.get("new_key")` read to this file means
    adding "new_key" here too, or every run dies at launch with a one-line fix.
    """
    reject_unknown_keys(exp, EXPERIMENT_KEYS, "experiment")
    reject_unknown_keys(
        exp.get("eval_slot_limits") or {}, EVAL_SLOT_LIMIT_KEYS, "eval_slot_limits"
    )
    for speaker, agent in (exp.get("agents") or {}).items():
        if isinstance(agent, dict):
            reject_unknown_keys(agent, AGENT_KEYS, f"agents.{speaker}")
            if agent.get("trained") and agent.get("frozen_policy"):
                raise ValueError(
                    f"agents.{speaker} cannot be both trained and frozen_policy"
                )
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
            # Present only when the slot overrides its seat: an arm that never
            # sets it keeps the identity sha it already published.
            **(
                {"enable_thinking": bool(cs.slot.enable_thinking)}
                if cs.slot.enable_thinking is not None
                else {}
            ),
        }
        for cs in protocol.compile()
    ]


def _model_payload(settings: ModelSettings) -> dict:
    train_sampling = resolved_sampling_profile(settings, "train")
    return {
        "model_type": settings.model_type.name.lower(),
        "model": safe_locator(settings.model_file_path, field="model_file_path"),
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
        "base_url": safe_locator(settings.base_url, field="base_url"),
        "train_sampling": train_sampling.model_dump(mode="json"),
    }


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
    training_protocol = effective_protocol_training(tr, args)
    plan_tokens = exp.get("plan_tokens")
    eval_pool = dict(exp.get("eval_dataset") or {})
    eval_limits = dict(exp.get("eval_slot_limits") or {})
    agents = {
        speaker: {
            "trained": speaker in trained,
            **(
                {"frozen_policy": True}
                if ((exp.get("agents") or {}).get(speaker) or {}).get(
                    "frozen_policy"
                )
                else {}
            ),
            **_model_payload(settings),
        }
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
        "learning": effective_learning_protocol(tr, topology),
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
            # Pool and generation caps of the dev/test instrument, present only
            # when the arm declares them: an arm that reads its task source at
            # the plain eval_max_tokens cap keeps the identity it published.
            **(
                {"dataset": {k: str(v) for k, v in sorted(eval_pool.items())}}
                if eval_pool
                else {}
            ),
            **(
                {"slot_limits": {k: int(v) for k, v in sorted(eval_limits.items())}}
                if eval_limits
                else {}
            ),
        },
    }
    runner = {
        "runner_protocol": "debate-runner-v2",
        "runner_protocol_sha256": canonical_sha256(payload),
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

    frozen_policy_speakers = {
        speaker
        for speaker, agent in (exp.get("agents") or {}).items()
        if isinstance(agent, dict) and agent.get("frozen_policy")
    }
    unknown_frozen_policies = frozen_policy_speakers - set(frozen)
    if unknown_frozen_policies:
        raise ValueError(
            "frozen_policy speakers must be untrained agents: "
            f"{sorted(unknown_frozen_policies)}"
        )

    frozen_models = {
        speaker: instantiate_model(settings, is_debater=speaker != "judge", binding="train")
        for speaker, settings in frozen.items()
        if speaker not in frozen_policy_speakers
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

    frozen_policies: dict[str, Policy] = {}
    for speaker in sorted(frozen_policy_speakers):
        settings = frozen[speaker]
        if settings.model_type is not ModelType.LOCAL:
            raise ValueError(
                f"frozen_policy seat {speaker!r} requires model_type 'local', "
                f"got {settings.model_type.name.lower()!r}"
            )
        if not settings.model_file_path or not settings.base_url:
            raise ValueError(
                f"frozen_policy seat {speaker!r} requires model_file_path and base_url"
            )
        caps = caps_by_speaker.get(speaker, [])
        if not caps or any(cap is None for cap in caps):
            raise ValueError(
                f"frozen_policy seat {speaker!r} requires max_total_tokens on every slot"
            )
        profile = resolved_sampling_profile(settings, "train")
        backend = VLLMCompletionsBackend(
            base_url=settings.base_url,
            model=settings.model_file_path,
            tokenizer_path=settings.model_file_path,
        )
        frozen_policies[speaker] = Policy(
            backend,
            SamplingParams(
                max_tokens=max(caps),
                temperature=(
                    profile.temperature if profile.temperature is not None else 1.0
                ),
                top_p=profile.top_p if profile.top_p is not None else 1.0,
            ),
            (
                {"enable_thinking": bool(settings.enable_thinking)}
                if settings.enable_thinking is not None
                else None
            ),
        )

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
            frozen_policies=frozen_policies,
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
    trained, frozen = split_agents(exp)
    if not trained:
        raise ValueError("no agent has trained: true")
    trained_settings = list(trained.values())
    tr = exp.get("training") or {}
    validate_trained_seats(trained, tr)
    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    config_kwargs = training_config_kwargs(tr, args)
    docent_dir = str(
        claim_directory(
            _docent_run_dir(run_name, launch_namespace=launch_namespace)
        )
    )
    transcript_dir = None
    if config_kwargs.get("log_transcripts", Config.log_transcripts):
        artifact_attempt_root = scheduler_artifact_attempt_root(launch_namespace)
        transcript_path = (
            os.path.join(
                artifact_attempt_root,
                "transcripts",
                safe_path_component(run_name, fallback="run"),
            )
            if artifact_attempt_root is not None
            else os.path.join(
                "transcripts",
                safe_path_component(run_name, fallback="run"),
                launch_namespace,
            )
        )
        transcript_dir = str(
            claim_directory(transcript_path)
        )

    dataset_type = (exp.get("dataset") or {}).get("type")
    env = build_env(exp, trained, frozen)
    cleanups.append(env.family.close)
    # Built before the backend so a bad eval pool fails in seconds rather than
    # after the rollout engine is up.
    eval_source, eval_source_close = build_eval_source(exp)
    if eval_source_close is not None:
        cleanups.append(eval_source_close)
    topology = resolve_topology()
    protocol_identity = debate_protocol_identity(
        exp, dataset_type, env.family, trained, frozen, args=args, topology=topology
    )
    lead = trained_settings[0]

    backend = build_backend(
        tr,
        lead.model_file_path,
        run_name,
        lr_override=args.lr,
        load_given=bool(args.load),
        gen_budgets=debate_gen_budgets(env.protocol, trained, tr),
        topology=topology,
        launch_namespace=launch_namespace,
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
        launch_namespace=launch_namespace,
        transcript_dir=transcript_dir,
        protocol_identity=protocol_identity,
        chat_template_kwargs=(
            {"enable_thinking": bool(lead.enable_thinking)} if lead.enable_thinking is not None else None
        ),
        **config_kwargs,
    )
    apply_wandb_resume_cli_authorization(
        cfg, args, resume_lease_handoff=resume_lease_handoff
    )
    # Comprehensive docent capture: every training rollout's debates -> JSONL.
    # The run-specific directory prevents concurrent sweep arms from overwriting
    # one another's identically numbered steps.
    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import agent_runs, export_jsonl_claimed

        states = getattr(env_, "last_states", None)
        if not states:
            raise RuntimeError(
                "requested debate rollout retained no states; refusing missing "
                "local Docent evidence"
            )
        export_jsonl_claimed(
            agent_runs(env_, states=states),
            docent_dir,
            f"step-{step:05d}.jsonl",
        )

    cfg.on_rollout = _export_docent

    # RLVR evals: proposal accuracy on a held-out split through a task source,
    # not through a debate. `eval_dataset:` supplies its own pool; without one
    # the training pool's dev/test carve is read. DebateEnv never routes a
    # rollout through its task source, so eval caps set here reach the eval
    # only.
    eval_env = prepare_eval_env(
        env.task_source if eval_source is None else eval_source,
        limits=eval_slot_limits(exp),
        plan_tokens=exp.get("plan_tokens"),
    )
    train(env, backend, cfg, eval_env=eval_env)


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


def main() -> None:
    cleanups: list = []
    try:
        _main(cleanups)
    finally:
        for cleanup in reversed(cleanups):
            cleanup()


if __name__ == "__main__":
    main()
