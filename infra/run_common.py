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
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    "eval_split",
    "final_test_eval",
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
    # 3-way split: eval_split "dev" logs dev/*; final_test_eval runs the one
    # test/ pass after the last step (see Config).
    ("eval_split", str),
    ("final_test_eval", _strict_bool),
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


CANONICAL_OLMO32_MODEL = "/workspace/models/olmo32-bf16"
CANONICAL_OLMO32_REPO = "allenai/Olmo-3.1-32B-Instruct-DPO"
CANONICAL_OLMO32_REVISION = "fc84a4f699916fcf585aa54371f47897fa934d5c"
CANONICAL_BF16_CONVERTER = "hf-safetensors-floating-to-bfloat16-v1"
CANONICAL_BF16_MARKER = ".bf16-conversion.json"


def _artifact_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing, not a regular file, or is a symlink: {path}")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return value


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_child(root: Path, name: object, label: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} contains an invalid empty filename")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or pure.as_posix() != name
        or "\\" in name
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} contains unsafe filename {name!r}")
    return root.joinpath(*pure.parts)


def validate_local_policy_artifact(
    model: str,
    *,
    canonical_olmo32_path: str = CANONICAL_OLMO32_MODEL,
    canonical_size_range: tuple[int, int] = (55_000_000_000, 75_000_000_000),
) -> str:
    """Fail closed on incomplete local model artifacts before backend startup.

    Generic local checkpoints retain the historical config + indexed/unindexed
    weight check.  The canonical OLMo-32B path is stronger: it is produced by
    ``convert_hf_safetensors_bf16.py``, so its final marker is the commit point.
    Requiring and checking that marker prevents a hard-killed conversion from
    looking launchable merely because all weight shards and config exist.
    """

    if not os.path.isabs(model):
        return model
    root = Path(os.path.abspath(model))
    canonical_root = Path(os.path.abspath(canonical_olmo32_path))
    root_resolved = root.resolve(strict=False)
    canonical_resolved = canonical_root.resolve(strict=False)
    targets_canonical = root == canonical_root or root_resolved == canonical_resolved
    if targets_canonical and (
        root != canonical_root
        or root_resolved != canonical_resolved
        or root_resolved != root
        or canonical_resolved != canonical_root
    ):
        raise ValueError(
            f"canonical OLMo artifact {model!r} must use the exact non-symlink path "
            f"{canonical_olmo32_path!r}; aliases and symlink components are refused"
        )
    config_path = root / "config.json"
    if not root.is_dir() or not config_path.is_file():
        raise ValueError(
            f"local policy artifact {model!r} is missing its directory or config.json; "
            "attach the volume containing the pinned model rather than falling back to HF"
        )
    model_config = _artifact_json(config_path, "local policy config")

    if not targets_canonical:
        indices = [
            path
            for path in (
                root / "model.safetensors.index.json",
                root / "pytorch_model.bin.index.json",
            )
            if path.is_file()
        ]
        if indices:
            referenced: set[str] = set()
            for index_path in indices:
                body = _artifact_json(index_path, "local policy index")
                weight_map = body.get("weight_map")
                if isinstance(weight_map, dict):
                    referenced.update(
                        name for name in weight_map.values() if isinstance(name, str)
                    )
            missing = sorted(
                name
                for name in referenced
                if not (root / name).is_file() or (root / name).stat().st_size == 0
            )
            if not referenced or missing:
                detail = ", ".join(missing[:5]) if missing else "index has no weight_map entries"
                raise ValueError(f"local policy artifact {model!r} is incomplete: {detail}")
        else:
            weights = [*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")]
            if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
                raise ValueError(f"local policy artifact {model!r} has no non-empty weight files")
        return model

    if root.is_symlink():
        raise ValueError(f"canonical OLMo artifact {model!r} may not be a symlink")
    index_path = root / "model.safetensors.index.json"
    marker_path = root / CANONICAL_BF16_MARKER
    index = _artifact_json(index_path, "canonical OLMo safetensors index")
    marker = _artifact_json(marker_path, "canonical OLMo conversion marker")

    architecture = model_config.get("architectures") or []
    if "Olmo3ForCausalLM" not in architecture:
        raise ValueError(
            f"canonical OLMo artifact {model!r} has unexpected architectures={architecture!r}"
        )
    if model_config.get("torch_dtype") != "bfloat16" or (
        "dtype" in model_config and model_config.get("dtype") != "bfloat16"
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} does not declare bfloat16")

    source = marker.get("source")
    if (
        marker.get("complete") is not True
        or marker.get("schema_version") != 1
        or marker.get("converter") != CANONICAL_BF16_CONVERTER
        or not isinstance(source, dict)
        or source.get("repo") != CANONICAL_OLMO32_REPO
        or source.get("revision") != CANONICAL_OLMO32_REVISION
    ):
        raise ValueError(
            f"canonical OLMo artifact {model!r} has no complete conversion marker "
            "for the pinned converter/repository/revision"
        )
    source_index_sha256 = source.get("index_sha256")
    output_index_sha256 = marker.get("output_index_sha256")
    if (
        not isinstance(source_index_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_index_sha256) is None
        or not isinstance(output_index_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", output_index_sha256) is None
        or output_index_sha256 != _artifact_sha256(index_path)
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} index integrity is invalid")

    support_names = marker.get("support_files")
    source_support_hashes = source.get("support_sha256")
    output_support_hashes = marker.get("output_support_sha256")
    if (
        not isinstance(support_names, list)
        or not isinstance(source_support_hashes, dict)
        or not isinstance(output_support_hashes, dict)
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} has invalid support-file provenance")
    if (
        set(support_names) != set(source_support_hashes)
        or set(support_names) != set(output_support_hashes)
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} support-file marker is inconsistent")
    for name in support_names:
        path = _artifact_child(root, name, "canonical support marker")
        source_hash = source_support_hashes.get(name)
        output_hash = output_support_hashes.get(name)
        if (
            not path.is_file()
            or path.is_symlink()
            or not isinstance(source_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
            or not isinstance(output_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", output_hash) is None
            or _artifact_sha256(path) != output_hash
        ):
            raise ValueError(f"canonical OLMo support file failed integrity check: {name!r}")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"canonical OLMo index {index_path} has no non-empty weight_map")
    referenced = {_artifact_child(root, name, "canonical weight index") for name in weight_map.values()}
    referenced_names = {path.relative_to(root).as_posix() for path in referenced}
    marker_weight_names = marker.get("weight_shards")
    marker_source_identities = marker.get("weight_shard_identity")
    marker_records = marker.get("shards")
    if (
        not isinstance(marker_weight_names, list)
        or set(marker_weight_names) != referenced_names
        or not isinstance(marker_source_identities, dict)
        or set(marker_source_identities) != referenced_names
        or not isinstance(marker_records, dict)
        or set(marker_records) != referenced_names
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} marker/index shard sets differ")

    weight_file_bytes = 0
    tensor_bytes = 0
    for path in sorted(referenced):
        name = path.relative_to(root).as_posix()
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"canonical OLMo artifact {model!r} has missing/unsafe shard {name!r}")
        record = marker_records[name]
        source_identity = marker_source_identities[name]
        if not isinstance(record, dict) or not isinstance(source_identity, dict):
            raise ValueError(f"canonical OLMo artifact {model!r} has invalid record for {name!r}")
        file_bytes = record.get("file_bytes")
        shard_tensor_bytes = record.get("tensor_bytes")
        output_sha256 = record.get("sha256")
        source_size = source_identity.get("size")
        source_sha256 = source_identity.get("sha256")
        if (
            not isinstance(file_bytes, int)
            or isinstance(file_bytes, bool)
            or file_bytes != path.stat().st_size
            or not isinstance(shard_tensor_bytes, int)
            or isinstance(shard_tensor_bytes, bool)
            or shard_tensor_bytes <= 0
            or not isinstance(output_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", output_sha256) is None
            or _artifact_sha256(path) != output_sha256
            or not isinstance(source_size, int)
            or isinstance(source_size, bool)
            or source_size <= 0
            or not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        ):
            raise ValueError(f"canonical OLMo shard failed integrity check: {name!r}")
        weight_file_bytes += file_bytes
        tensor_bytes += shard_tensor_bytes

    minimum, maximum = canonical_size_range
    if minimum < 0 or maximum < minimum or not minimum <= weight_file_bytes <= maximum:
        raise ValueError(
            f"canonical OLMo artifact {model!r} has {weight_file_bytes} weight bytes; "
            f"expected the bf16 conversion ({minimum}-{maximum} byte guard)"
        )
    index_metadata = index.get("metadata")
    if (
        marker.get("weight_file_bytes") != weight_file_bytes
        or marker.get("tensor_bytes") != tensor_bytes
        or not isinstance(index_metadata, dict)
        or index_metadata.get("total_size") != tensor_bytes
    ):
        raise ValueError(f"canonical OLMo artifact {model!r} aggregate sizes are inconsistent")
    return model


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
    # pod_run.sh validates early for operator feedback, but the invariant also
    # belongs here so direct module/console-script launches cannot bypass it.
    validate_local_policy_artifact(model_path)
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
