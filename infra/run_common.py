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
import os
import re

from infra.backend.base import LossSpec


WANDB_ENTITY = "palaestra-research"
WANDB_PROJECT = "debate-rebuild"


def wandb_config_kwargs(
    *,
    domain: str,
    run_type: str,
    experiment: str,
    model_path: str,
    enabled: bool = True,
    configured_project: str | None = None,
) -> dict:
    """Build the canonical W&B metadata passed to :class:`infra.train.Config`.

    Grouping is deliberately independent of ``run_name``: sweep suffixes keep
    identifying individual runs and checkpoint directories, while every arm
    of an experiment remains together under ``domain/run_type/experiment``.
    """
    components = {
        "domain": domain,
        "run_type": run_type,
        "experiment": experiment,
    }
    for label, component in components.items():
        if not component or not component.strip():
            raise ValueError(f"W&B group component {label} must not be empty")
        if "/" in component:
            raise ValueError(f"W&B group component {label} must not contain '/': {component!r}")
    if enabled and configured_project not in (None, WANDB_PROJECT):
        raise ValueError(
            f"W&B project must be {WANDB_PROJECT!r}, got {configured_project!r}"
        )

    scope = "smoke" if "smoke" in experiment.lower() else "full"
    return {
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT if enabled else None,
        "wandb_group": f"{domain}/{run_type}/{experiment}",
        "wandb_job_type": "train",
        "wandb_tags": [
            f"domain:{domain}",
            f"run_type:{run_type}",
            f"experiment:{experiment}",
            f"model:{model_path}",
            f"scope:{scope}",
        ],
    }


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
    "log_transcripts",
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
    ("log_transcripts", bool),
)


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


def runner_parser(description: str | None) -> argparse.ArgumentParser:
    """The CLI both train runners share: experiment selection plus sweep
    overrides. Runner-specific flags are added by the caller."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-file", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--wandb-project",
        choices=[WANDB_PROJECT],
        default=None,
        help=f"W&B project (organization policy requires {WANDB_PROJECT})",
    )
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
    kw: dict = {k: cast(tr[k]) for k, cast in CONFIG_CASTS if tr.get(k) is not None}
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
