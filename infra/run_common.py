"""Plumbing shared by the train runners (run_debate, run_rlvr): the
training-block key contract, backend construction with checkpoint
namespacing, the sweep-suffix run identity, and the common CLI.

This module exists so the dependency points the right way: run_rlvr is the
judge-free control arm and must not import from run_debate (the treatment)
to get machinery both arms share. Nothing here knows about debates, judges,
or task families.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re

from infra.backend.base import LossSpec


# Every key the runners actually read out of `training:` — the union of
# run_debate.main, run_rlvr.main, and build_backend.
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
    "log_transcripts",
    "norm_adv_by_std",
    "dynamic_sampling_retries",
    "oversample_factor",
    "warmup_steps",
    "lr_schedule",
    "min_lr_ratio",
    "adv_population_std",
    "drop_zero_advantage",
    "kl_mechanism",
    "rl_seed",
}

def _strict_bool(value) -> bool:
    """YAML bool cast: bool("false") is True, so a quoted string would set the
    opposite of what the config spells. Only real bools (and 0/1 ints) pass."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f"expected a YAML bool (true/false, unquoted) or 0/1, got {value!r} — "
        "a quoted string like \"false\" is truthy and would flip the knob"
    )


# training-block keys that map 1:1 onto Config fields, with their YAML casts.
# Values pass through ONLY when the YAML sets them; everything absent falls to
# Config's dataclass defaults — the single source of truth. No fallback
# literals at the call sites: a default lives in exactly one place.
CONFIG_CASTS: tuple[tuple[str, object], ...] = (
    ("steps", int),
    ("batch_size", int),
    ("group_size", int),
    ("lora_rank", int),
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
    ("log_transcripts", _strict_bool),
    ("norm_adv_by_std", _strict_bool),
    # DAPO dynamic sampling: retry rounds for re-rolling zero-variance groups
    # on fresh tasks (see Config.dynamic_sampling_retries). 0/absent = off.
    ("dynamic_sampling_retries", int),
    # Upfront variant: one oversized draw, keep first batch_size healthy
    # groups (see Config.oversample_factor). 1.0/absent = off.
    ("oversample_factor", float),
    # Linear lr warmup steps (see Config.warmup_steps). 0/absent = off.
    ("warmup_steps", int),
    # Post-warmup shape: constant (default) or cosine to lr*min_lr_ratio.
    ("lr_schedule", str),
    ("min_lr_ratio", float),
    # hw4/DeepSeekMath parity knobs (see Config for semantics).
    ("adv_population_std", _strict_bool),
    ("drop_zero_advantage", _strict_bool),
    ("kl_mechanism", str),
    ("rl_seed", int),
)

# YAML key -> Config field where the names differ. The YAML key is rl_seed so
# the training seed cannot be misread as dataset.seed; the Config field named
# `seed` predates the distinction.
CONFIG_FIELD_NAMES = {"rl_seed": "seed"}


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


TOPOLOGY_FILE = "configs/topologies.yaml"


def _short_gpu_name(raw: str) -> str:
    """'NVIDIA H100 80GB HBM3' -> 'H100'; 'NVIDIA B200' -> 'B200'."""
    words = [w for w in raw.split() if w.upper() not in ("NVIDIA", "GEFORCE", "TESLA")]
    return words[0].upper() if words else "UNKNOWN"


def detect_topology_key() -> str | None:
    """'<count>x<model>' from nvidia-smi, or None on a GPU-less machine."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if out.returncode != 0 or not names:
        return None
    return f"{len(names)}x{_short_gpu_name(names[0])}"


def resolve_topology(path: str = TOPOLOGY_FILE) -> dict:
    """Hardware plumbing defaults for the machine we are on, auto-detected.

    No CLI override on purpose: the yaml file is the single control surface.
    Detected GPUs whose key is missing from the file are a HARD error: a
    topology nobody smoke-tested must not launch on defaults tuned for
    different silicon (the H200 util-0.50 VRAM boot failure). GPU-less
    machines (local tests) resolve to {} — arms that carry a full verl block
    keep working unchanged.
    """
    key = detect_topology_key()
    if key is None:
        return {}
    import yaml

    with open(path) as fh:
        table = yaml.safe_load(fh) or {}
    if key not in table:
        raise RuntimeError(
            f"topology {key!r} is not in {path} (known: {sorted(table)}). Add an "
            "entry for this hardware and validate it with one smoke run before "
            "the first paid launch."
        )
    entry = dict(table[key] or {})
    unknown = set(entry) - VERL_KEYS
    if unknown:
        raise RuntimeError(
            f"topology {key!r} in {path} carries keys outside training.verl's "
            f"contract: {sorted(unknown)} (allowed: {sorted(VERL_KEYS)})"
        )
    os.environ["DEBATE_TOPOLOGY"] = key  # provenance: read by train._env_identity
    return entry


def apply_topology(verl_cfg: dict, topology: dict) -> dict:
    """Topology supplies defaults; the arm's own keys win. extra_overrides is
    the one list-valued key, and 'arm wins' would silently DROP the topology's
    engine flags (e.g. disable_custom_all_reduce) the moment an arm adds one
    of its own — so that key concatenates (topology first, deduped) instead."""
    merged = dict(topology) | {k: v for k, v in verl_cfg.items() if k != "extra_overrides"}
    extras = list(topology.get("extra_overrides") or [])
    extras += [e for e in (verl_cfg.get("extra_overrides") or []) if e not in extras]
    if extras:
        merged["extra_overrides"] = extras
    return merged


def runner_parser(description: str | None) -> argparse.ArgumentParser:
    """The CLI both train runners share: experiment selection plus sweep
    overrides. Runner-specific flags are added by the caller."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-file", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", default=None, help="override training.wandb_project")
    parser.add_argument("--wandb-entity", default=None, help="override training.wandb_entity")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--steps", type=int, default=None, help="override training.steps")
    parser.add_argument("--lr", type=float, default=None, help="override training.lr (sweeps)")
    parser.add_argument("--levels", default=None, help="override dataset.levels (e.g. 4 or 3-4)")
    parser.add_argument(
        "--group-size", type=int, default=None, help="override training.group_size (sweeps)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override training.batch_size (sweeps)"
    )
    parser.add_argument("--load", default=None)
    parser.add_argument(
        "--wandb-resume",
        default=None,
        metavar="RUN_ID",
        help="append to this existing wandb run (resume='must') instead of "
        "starting a new one — for continuations via --load",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="first step index of a continuation; the loop runs [start-step, steps) "
        "so saves/evals keep the original lineage's numbering. Default: inferred "
        "from a --load path ending in step-NNNNN, else 0",
    )
    return parser


def resolved_start_step(args: argparse.Namespace) -> int:
    """--start-step, else the step number in a `--load .../step-NNNNN` path."""
    if getattr(args, "start_step", None) is not None:
        return args.start_step
    if getattr(args, "load", None):
        m = re.search(r"step-(\d+)/?$", args.load)
        if m:
            return int(m.group(1))
    return 0


def training_config_kwargs(tr: dict, args: argparse.Namespace) -> dict:
    """training block + CLI sweep overrides -> Config kwargs, present keys only."""
    kw: dict = {
        CONFIG_FIELD_NAMES.get(k, k): cast(tr[k])
        for k, cast in CONFIG_CASTS
        if tr.get(k) is not None
    }
    if tr.get("loss"):
        kw["loss"] = LossSpec(**tr["loss"])
    for k in ("steps", "batch_size", "group_size", "lr"):
        if getattr(args, k) is not None:
            kw[k] = getattr(args, k)
    start = resolved_start_step(args)
    if start:
        kw["start_step"] = start
    if getattr(args, "wandb_resume", None):
        kw["wandb_run_id"] = args.wandb_resume
    return kw


def run_id() -> str:
    """A per-LAUNCH id, so two runs can never share a checkpoint directory.

    $VOL is shared across the team, and run_name alone is not unique: two people
    launching the same experiment get the same path. The existing guard
    (check_fresh_run_over_existing_checkpoints) refuses to start fresh over
    someone else's checkpoints, but it is check-then-act -- two launches inside
    the same minute both see an empty directory and both proceed, and concurrent
    writers to one .pt file produce a TORN file rather than a lost one: it looks
    valid and fails at --load, possibly days later.

    UTC to the second, so it sorts chronologically and is legible in a listing.
    Deliberately carries no owner or hostname: the id answers "which launch",
    and identity belongs in the run's own metadata, not in a path.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    gen_budgets: dict | None = None,
    topology: dict | None = None,
):
    """training block -> Backend. Shared by the debate and RLVR runners.

    `run_name` (experiment + sweep suffix) namespaces the checkpoint directory:
    a shared network volume would otherwise have a 2-step smoke run's `final`
    clobber a 100-step run's, and two arms of one sweep clobber each other's.
    `load_given` says whether the operator passed --load, which is the only
    form of resume there is; see check_fresh_run_over_existing_checkpoints.

    `gen_budgets` names each caller-side generation budget ({label: max
    tokens or None}) so the verl branch can cross-check them against
    response_length: the rollout engine truncates any longer request at
    response_length (stop_reason="length") with no other trace. run_rlvr
    passes max_completion_tokens and training.eval_max_tokens; run_debate
    passes its trained-seat slot caps plus eval_max_tokens (judge slots are
    served elsewhere and excluded — see debate_gen_budgets).

    Same only-if-present rule as training_config_kwargs: knobs the YAML omits
    fall to the backend config's own dataclass defaults.
    """
    backend_kind = str(tr.get("backend", "tinker"))
    if str(tr.get("kl_mechanism", "advantage")) == "loss" and backend_kind != "verl":
        # Tinker HAS ref_logprobs, so the train loop's capability check would
        # pass, stamp the datums, skip the advantage penalty — and tinker's
        # forward_backward would ignore the field: kl_coef silently becomes a
        # no-op (round-3 audit). Only verl consumes ref_log_prob in-loss.
        raise RuntimeError(
            f"training.kl_mechanism 'loss' requires backend 'verl' (got "
            f"{backend_kind!r}); on other backends the stamped ref_logprobs "
            "are ignored and the KL vanishes without a trace."
        )
    if "verl" in tr and backend_kind != "verl":
        # A verl block under any other backend is dead config: every knob in
        # it (response_length, n_gpus, checkpoint_dir, ...) would be ignored
        # without a word while the run trains on the other backend's defaults.
        raise RuntimeError(
            f"training.verl is set but training.backend resolved to {backend_kind!r}; "
            "the entire verl block would be silently ignored. Set training.backend: "
            "verl or remove the block."
        )
    if backend_kind == "tinker":
        from infra.backend.tinker import TinkerBackend

        # Tinker checkpoints live service-side under the run, not in a local
        # directory, so there is nothing here to namespace.
        kw = {"lora_rank": int(tr["lora_rank"])} if tr.get("lora_rank") is not None else {}
        return TinkerBackend(model_path, **kw)
    if backend_kind == "verl":
        from infra.backend.verl import VerlBackend, VerlBackendConfig

        v = apply_topology(dict(tr.get("verl") or {}), topology or {})
        ckpt_root = str(v.get("checkpoint_dir", VerlBackendConfig.checkpoint_dir))
        # run_id makes the directory unique per LAUNCH, so a shared volume
        # cannot serve two runs the same path. The wandb run keeps the bare
        # run_name -- wandb ids are already unique, so only the filesystem needs
        # this -- and the resolved path is logged into the run's config so the
        # mapping from run name back to checkpoints stays discoverable.
        ckpt_dir = os.path.join(ckpt_root, f"{run_name}-{run_id()}")
        check_legacy_checkpoint_layout(ckpt_root, ckpt_dir)
        check_fresh_run_over_existing_checkpoints(ckpt_dir, load_given)
        verl_casts: tuple[tuple[str, object], ...] = (
            ("n_gpus", int),
            ("strategy", str),
            ("gpu_memory_utilization", float),
            ("prompt_length", int),
            ("response_length", int),
            ("max_token_len_per_gpu", int),
            ("rollout_tp", int),
            ("use_remove_padding", _strict_bool),
        )
        vkw: dict = {k: cast(v[k]) for k, cast in verl_casts if v.get(k) is not None}
        response_length = int(vkw.get("response_length", VerlBackendConfig.response_length))
        budgets = {k: int(b) for k, b in (gen_budgets or {}).items() if b is not None}
        if any(b > response_length for b in budgets.values()):
            named = ", ".join(f"{k}={b}" for k, b in budgets.items())
            raise RuntimeError(
                f"generation budget exceeds training.verl.response_length="
                f"{response_length} ({named}); the rollout engine truncates such "
                "requests at response_length with stop_reason='length' and no other "
                "trace. Raise response_length or lower the budget."
            )
        if tr.get("lora_rank") is not None:
            vkw["lora_rank"] = int(tr["lora_rank"])
        if str(tr.get("kl_mechanism", "advantage")) == "loss":
            # In-loss KL: the coefficient rides training.kl_coef; the backend
            # needs it at construction (use_kl_loss is a worker-config knob).
            if tr.get("kl_coef") is None or float(tr["kl_coef"]) <= 0:
                raise RuntimeError(
                    "training.kl_mechanism 'loss' needs training.kl_coef > 0 "
                    "— the coefficient feeds verl's kl_loss_coef."
                )
            vkw["kl_loss_coef"] = float(tr["kl_coef"])
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
