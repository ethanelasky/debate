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

import argparse
import os

from infra.backend.base import LossSpec, SamplingParams
from infra.config import load_experiment, parse_model_settings, reject_unknown_keys
from infra.envs.debate.env import DebateEnv, DebateEnvConfig
from infra.envs.debate.judge import JudgeConfig
from infra.envs.debate.rewards import ScoringConfig
from infra.envs.debate.protocol import Protocol
from infra.envs.tasks import get_family
from infra.models.base import ModelSettings, resolved_sampling_profile
from infra.models.factory import instantiate_model
from infra.train import Config, train


# Every key the runners actually read out of `training:` — the union of
# run_debate.main, run_rlvr.main, and build_backend. run_rlvr imports these.
TRAINING_KEYS = {
    "backend",
    "verl",
    "lora_rank",
    "loss",
    "lr",
    "steps",
    "batch_size",
    "group_size",
    "micro_batch",
    "ppo_epochs",
    "adv_length_norm",
    "kl_coef",
    "kl_discount_factor",
    "eval_every",
    "eval_max_tokens",
    "eval_n",
    "save_every",
    "wandb_project",
    "wandb_entity",
}

# training-block keys that map 1:1 onto Config fields, with their YAML casts.
# Values pass through ONLY when the YAML sets them; everything absent falls to
# Config's dataclass defaults — the single source of truth. No fallback
# literals at the call sites: a default lives in exactly one place.
CONFIG_CASTS: tuple[tuple[str, type], ...] = (
    ("steps", int),
    ("batch_size", int),
    ("group_size", int),
    ("micro_batch", int),
    ("lr", float),
    ("ppo_epochs", int),
    ("adv_length_norm", str),
    ("kl_coef", float),
    ("kl_discount_factor", float),
    ("eval_every", int),
    ("eval_max_tokens", int),
    ("eval_n", int),
    ("save_every", int),
)


def training_config_kwargs(tr: dict, args: argparse.Namespace) -> dict:
    """training block + CLI sweep overrides -> Config kwargs, present keys only.
    Shared by the debate and RLVR runners."""
    kw: dict = {k: cast(tr[k]) for k, cast in CONFIG_CASTS if tr.get(k) is not None}
    if tr.get("loss"):
        kw["loss"] = LossSpec(**tr["loss"])
    for k in ("steps", "batch_size", "group_size", "lr"):
        if getattr(args, k) is not None:
            kw[k] = getattr(args, k)
    return kw


# training.verl — read only in build_backend.
VERL_KEYS = {
    "n_gpus",
    "strategy",
    "gpu_memory_utilization",
    "prompt_length",
    "response_length",
    "max_token_len_per_gpu",
    "rollout_tp",
    "use_remove_padding",
    "checkpoint_dir",
    "extra_overrides",
}

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


def build_env(exp: dict, trained: dict[str, ModelSettings], frozen: dict[str, ModelSettings]) -> DebateEnv:
    proto_spec = exp["protocol"]
    if isinstance(proto_spec, str):
        proto_spec = exp["_protocols"][proto_spec]
    protocol = Protocol.parse(proto_spec)

    frozen_models = {
        speaker: instantiate_model(settings, is_debater=speaker != "judge", binding="train")
        for speaker, settings in frozen.items()
    }

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
        judge=JudgeConfig(**(exp.get("judge_config") or {})),
        scoring=ScoringConfig(**(exp.get("scoring") or {})),
        fresh_positions=exp.get("fresh_positions", True),
        flip=exp.get("flip", False),
        first_speech_non_debate_aware=bool(exp.get("first_speech_non_debate_aware", False)),
    )
    relaxed = bool(ds.pop("relaxed_extraction", True))
    task_source = family.source(ds)
    return DebateEnv(config, task_source, family, relaxed_extraction=relaxed)


def run_identity_suffix(
    lr: float | None,
    levels: str | None,
    group_size: int | None,
    batch_size: int | None,
) -> str:
    """The `-lr…-L…-g…-b…` tail that distinguishes sweep arms of one experiment.

    ONE definition, used for both the wandb run name and the checkpoint
    namespace. Split them and two arms of an lr sweep get distinct wandb runs
    but the same checkpoint directory, which is exactly the clobbering the
    namespacing exists to prevent. Changing the format here changes existing
    wandb run names too — leave it alone unless you mean to break that history.
    """
    return (
        (f"-lr{lr:g}" if lr is not None else "")
        + (f"-L{levels}" if levels is not None else "")
        + (f"-g{group_size}" if group_size is not None else "")
        + (f"-b{batch_size}" if batch_size is not None else "")
    )


def check_legacy_checkpoint_layout(root: str, namespaced: str) -> None:
    """Refuse to run when `root` holds pre-namespacing checkpoints.

    Checkpoints used to be written straight into training.verl.checkpoint_dir
    with no experiment in the name, so two arms sharing a volume overwrote each
    other's `final`. Now every run writes under <root>/<run name>. Leftovers
    from the old layout are ambiguous — we cannot tell which arm produced them —
    and leaving them where nothing will ever read them again both strands that
    work and leaves the next arm free to overwrite it. Fail instead.
    """
    if not os.path.isdir(root):
        return
    stale = sorted(
        e for e in os.listdir(root) if e == "final" or e.startswith("step-")
    )
    if not stale:
        return
    raise RuntimeError(
        f"{root} contains checkpoints from the pre-namespacing layout: "
        f"{', '.join(stale[:5])}{' ...' if len(stale) > 5 else ''}. "
        "Checkpoints are now written per run, so these would be orphaned where "
        "nothing reads them and a later arm could overwrite them. Move them "
        f"into the run subdirectory they belong to (e.g. {namespaced}) or point "
        "training.verl.checkpoint_dir somewhere else, then rerun."
    )


def check_fresh_run_over_existing_checkpoints(namespaced: str, load_given: bool) -> None:
    """Refuse to start from scratch on top of a previous attempt's checkpoints.

    There is no auto-resume: resume happens only when the operator passes
    --load <path>. So the natural crash-recovery instinct — rerun the same
    command — would train from step 0 and overwrite step-00025, step-00050,
    final one at a time, leaving a directory whose entries come from two
    different lineages and which no later --load can disambiguate. With --load
    the operator has named a checkpoint deliberately, so overwrites are a
    choice and this guard stands down.
    """
    if load_given or not os.path.isdir(namespaced):
        return
    existing = sorted(
        e for e in os.listdir(namespaced) if e == "final" or e.startswith("step-")
    )
    if not existing:
        return
    raise RuntimeError(
        f"{namespaced} already holds checkpoints from an earlier attempt: "
        f"{', '.join(existing[:5])}{' ...' if len(existing) > 5 else ''}. "
        "Starting fresh here would overwrite them one step at a time and mix "
        "two lineages in one directory. Either pass --load <path> to continue "
        "deliberately from an explicit checkpoint, or move/delete that "
        "directory first, then rerun."
    )


def build_backend(
    tr: dict,
    model_path: str,
    run_name: str,
    lr_override: float | None = None,
    load_given: bool = False,
):
    """training block -> Backend. Shared by the debate and RLVR runners.

    `run_name` (experiment + sweep suffix) namespaces the checkpoint directory:
    a shared network volume would otherwise have a 2-step smoke run's `final`
    clobber a 100-step run's, and two arms of one sweep clobber each other's.
    `load_given` says whether the operator passed --load, which is the only
    form of resume there is; see check_fresh_run_over_existing_checkpoints.

    Same only-if-present rule as training_config_kwargs: knobs the YAML omits
    fall to the backend config's own dataclass defaults.
    """
    backend_kind = str(tr.get("backend", "tinker"))
    if backend_kind == "tinker":
        from infra.backend.tinker import TinkerBackend

        # Tinker checkpoints live service-side under the run, not in a local
        # directory, so there is nothing here to namespace.
        kw = {"lora_rank": int(tr["lora_rank"])} if tr.get("lora_rank") is not None else {}
        return TinkerBackend(model_path, **kw)
    if backend_kind == "verl":
        from infra.backend.verl import VerlBackend, VerlBackendConfig

        v = dict(tr.get("verl") or {})
        ckpt_root = str(v.get("checkpoint_dir", VerlBackendConfig.checkpoint_dir))
        ckpt_dir = os.path.join(ckpt_root, run_name)
        check_legacy_checkpoint_layout(ckpt_root, ckpt_dir)
        check_fresh_run_over_existing_checkpoints(ckpt_dir, load_given)
        verl_casts: tuple[tuple[str, type], ...] = (
            ("n_gpus", int),
            ("strategy", str),
            ("gpu_memory_utilization", float),
            ("prompt_length", int),
            ("response_length", int),
            ("max_token_len_per_gpu", int),
            ("rollout_tp", int),
            ("use_remove_padding", bool),
        )
        vkw: dict = {k: cast(v[k]) for k, cast in verl_casts if v.get(k) is not None}
        if tr.get("lora_rank") is not None:
            vkw["lora_rank"] = int(tr["lora_rank"])
        if lr_override is not None:
            vkw["lr"] = float(lr_override)
        elif tr.get("lr") is not None:
            vkw["lr"] = float(tr["lr"])
        if tr.get("loss"):
            vkw["loss"] = LossSpec(**tr["loss"])
        if v.get("extra_overrides"):
            vkw["extra_overrides"] = tuple(v["extra_overrides"])
        return VerlBackend(
            VerlBackendConfig(model_path=model_path, checkpoint_dir=ckpt_dir, **vkw)
        )
    raise ValueError(f"training.backend must be tinker|verl, got {backend_kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-file", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", default=None, help="override training.wandb_project")
    parser.add_argument("--wandb-entity", default=None, help="override training.wandb_entity")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    parser.add_argument("--lr", type=float, default=None, help="override training.lr (sweeps)")
    parser.add_argument("--levels", default=None, help="override dataset.levels (e.g. 4 or 3-4)")
    parser.add_argument("--group-size", type=int, default=None, help="override training.group_size (sweeps)")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size (sweeps)")
    parser.add_argument("--load", default=None)
    args = parser.parse_args()

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    if args.levels is not None:
        exp.setdefault("dataset", {})["levels"] = args.levels
    trained, frozen = split_agents(exp)
    if not trained:
        raise ValueError("no agent has trained: true")
    trained_settings = list(trained.values())
    base_models = {s.model_file_path for s in trained_settings}
    if len(base_models) != 1:
        raise ValueError(f"trained seats must share one base model, got {base_models}")

    env = build_env(exp, trained, frozen)
    tr = exp.get("training") or {}
    lead = trained_settings[0]

    run_name = args.experiment + run_identity_suffix(
        args.lr, args.levels, args.group_size, args.batch_size
    )
    backend = build_backend(
        tr, lead.model_file_path, run_name, lr_override=args.lr, load_given=bool(args.load)
    )
    if args.load:
        backend.load(args.load)

    profile = resolved_sampling_profile(lead, "train")
    cfg = Config(
        base_model=lead.model_file_path,
        sampling=SamplingParams(
            # no ceiling here: budgets are the protocol's per-slot caps, and
            # Policy hard-errors on any generation left unbounded
            max_tokens=None,
            temperature=profile.temperature if profile.temperature is not None else 1.0,
            top_p=profile.top_p if profile.top_p is not None else 1.0,
        ),
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
    # task source itself (MathEnv), not a debate
    train(env, backend, cfg, eval_env=env.task_source)


if __name__ == "__main__":
    main()
