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

import os

from infra.backend.base import SamplingParams
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
    build_backend,
    run_identity_suffix,
    runner_parser,
    training_config_kwargs,
)
from infra.train import Config, train

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
    proto_spec = exp["protocol"]
    if isinstance(proto_spec, str):
        proto_spec = exp["_protocols"][proto_spec]
    protocol = Protocol.parse(proto_spec)

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
        first_speech_non_debate_aware=bool(exp.get("first_speech_non_debate_aware", False)),
    )
    relaxed = bool(ds.pop("relaxed_extraction", True))
    task_source = family.source(ds)
    return DebateEnv(config, task_source, family, relaxed_extraction=relaxed)


def main() -> None:
    args = runner_parser(__doc__).parse_args()

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

    env = build_env(exp, trained, frozen)
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
        chat_template_kwargs=(
            {"enable_thinking": bool(lead.enable_thinking)} if lead.enable_thinking is not None else None
        ),
        **training_config_kwargs(tr, args),
    )
    # Comprehensive docent capture: every training rollout's debates -> JSONL
    def _export_docent(step: int, env_) -> None:
        from infra.envs.debate.docent_export import agent_runs, export_jsonl

        os.makedirs("docent", exist_ok=True)
        export_jsonl(agent_runs(env_), f"docent/step-{step:05d}.jsonl")

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


if __name__ == "__main__":
    main()
