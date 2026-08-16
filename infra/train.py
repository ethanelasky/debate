"""The training loop. This is the whole thing."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import math
import os
import pwd
import re
import stat
import sys
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from infra.backend.base import Backend, LossSpec, OptimParams, SamplingParams
from infra.envs.base import Env, Policy, Trajectory
from infra.launch_namespace import (
    claim_directory,
    require_claimed_directory,
    resolve_launch_namespace,
    safe_path_component,
    validate_launch_namespace,
)
from infra.rl.datums import grpo_pack


_WANDB_RESUME_CAPABILITY_ISSUER = object()
_WANDB_RESUME_HANDOFF_ISSUER = object()
_WANDB_RESUME_SDK_VERSION = "0.28.1"


class _WandbResumeRunningCapability:
    """Opaque runner-issued policy receipt for one exact W&B run ID.

    This prevents ordinary YAML, environment, or direct ``Config`` plumbing
    from enabling the human escape hatch. It is intentionally not described
    as cryptographic authentication: Python code in this process can import
    private names and subvert policy if it deliberately chooses to do so.
    """

    __slots__ = ("run_id", "_issuer")

    def __init__(self, run_id: str, issuer: object):
        if issuer is not _WANDB_RESUME_CAPABILITY_ISSUER:
            raise TypeError("W&B resume capability may only be runner-issued")
        self.run_id = run_id
        self._issuer = issuer


def _issue_wandb_resume_running_capability(
    run_id: str,
) -> _WandbResumeRunningCapability:
    if not isinstance(run_id, str):
        raise ValueError("cannot issue W&B resume capability for a non-string run ID")
    return _WandbResumeRunningCapability(
        run_id, _WANDB_RESUME_CAPABILITY_ISSUER
    )


@dataclass
class Config:
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    steps: int = 100
    batch_size: int = 8        # tasks per step
    group_size: int = 8        # samples per task (GRPO group)
    micro_batch: int = 64      # datums per forward_backward call
    lr: float = 1e-5
    loss: LossSpec = field(default_factory=LossSpec)
    ppo_epochs: int = 1  # passes over each rollout batch (one optim_step per epoch)
    optim: OptimParams | None = None
    sampling: SamplingParams = field(default_factory=lambda: SamplingParams(max_tokens=512))
    chat_template_kwargs: dict | None = None
    norm_adv_by_std: bool = True
    # Population (/n) vs sample (/(n-1)) std in group normalization. Sample is
    # verl parity (the campaign default); population matches hw4/DeepSeekMath.
    adv_population_std: bool = False
    adv_length_norm: str = "none"   # none|datum|trajectory|count — see grpo_pack
    # Keep zero-advantage datums in the batch (False) instead of dropping them
    # (True, default). Under seq-mean loss aggregation the denominator is the
    # ROW count, so dropping inflates the effective lr by batch/kept — hw4
    # keeps all rows.
    drop_zero_advantage: bool = True
    # DAPO-style dynamic sampling. With group-normalized advantages, a group
    # whose trajectories all earn the same reward has zero variance and is
    # dropped whole by grpo_pack (pack/n_datums_dropped_zero_advantage) — at
    # high-agreement stages most of the batch is wasted compute. When > 0,
    # degenerate groups are re-rolled on FRESH tasks for up to this many retry
    # rounds so the batch carries full gradient signal; the cap keeps a
    # saturated pool from looping forever. 0 = off, loop identical to before.
    dynamic_sampling_retries: int = 0
    # DAPO dynamic sampling, upfront variant: draw batch_size * factor groups
    # in the ONE initial rollout and keep the first batch_size non-degenerate
    # ones (draw order). Trades tokens for wall-clock: no serial retry rounds,
    # so a step pays one generation latency instead of 1 + retries. Mutually
    # exclusive with dynamic_sampling_retries (both police the same waste).
    # 1.0 = off, loop identical to before.
    oversample_factor: float = 1.0
    kl_coef: float = 0.0            # 0 = no reference-KL penalty
    # Where kl_coef acts. "loss" (default): verl's native differentiable in-loss
    # KL (k3/low_var_kl vs the frozen ref, recomputed each minibatch) —
    # hw4/DeepSeekMath semantics; the loop stamps datum.ref_logprobs once per
    # step and the backend forwards them. Requires a backend exposing
    # ref_logprobs (verl + LoRA). "advantage": centered k1 penalty added to
    # advantages once per step from the behavior logprobs — reshapes credit
    # within the batch, zero net pull toward ref (the pre-2026-08-11 mechanism;
    # the only option on backends without ref_logprobs, e.g. tinker).
    # Inert either way at kl_coef 0.
    kl_mechanism: str = "loss"
    # Linear lr warmup over the first N steps (0 = off, constant lr). Applied
    # in the loop by rescaling OptimParams.lr per step: verl's own scheduler
    # never advances on our tinker-worker path (optimizer_step only), and the
    # backend rewrites lr into the param groups every step anyway — so the
    # loop is the one place a schedule can actually take effect.
    warmup_steps: int = 0
    # After warmup: "constant" holds peak lr; "cosine" decays it to
    # lr * min_lr_ratio by the final step. A constant full-size step forever
    # is how the instruct flagship orbited its peak instead of settling
    # (Ethan, 2026-08-11: "we should have lr decay").
    lr_schedule: str = "constant"
    min_lr_ratio: float = 0.1
    kl_discount_factor: float = 0.0  # >0 smears future KL onto earlier tokens
    eval_every: int = 20
    eval_n: int = 128
    eval_max_tokens: int | None = None  # eval env generations need an explicit budget
    # Which split the periodic eval reads, and its metric prefix: "test" ->
    # eval/ (historical behavior, default); "dev" -> dev/ (the 3-way-split
    # protocol, Ethan 2026-08-11: decisions read dev, test stays unread).
    eval_split: str = "test"
    # Run ONE test-split eval after the final step, logged under test/ — the
    # single read of the held-out set under the 3-way protocol.
    final_test_eval: bool = False
    save_every: int = 50
    # Optional external provenance. Fresh init/log/finish are best-effort;
    # resume identity validation, init, and namespace-history append gate a
    # continuation because otherwise its lineage cannot be verified.
    wandb_project: str | None = None  # None = no W&B
    # wandb entity (team). None falls back to the API key's DEFAULT entity,
    # which is a personal namespace — runs then land somewhere the rest of the
    # team cannot see. Set it explicitly in the experiment config.
    wandb_entity: str | None = None
    log_transcripts: bool = True  # required local rollout evidence; W&B is optional
    transcript_every: int = 10  # best-effort W&B upload cadence (local is every step)
    # Where this run's checkpoints actually landed. Recorded because the
    # directory carries the launch namespace that the W&B display name does
    # not, so the mapping from a run back to checkpoints stays explicit.
    checkpoint_dir: str | None = None
    run_name: str | None = None
    # Operational identity of this process launch. It is provenance, not part
    # of protocol_identity or the W&B display name.
    launch_namespace: str | None = None
    # Claimed once before training, then reused by every transcript step.
    transcript_dir: str | None = None
    # Continuation plumbing: start_step makes the loop run [start_step, steps)
    # so step indices, save names, and eval cadence continue the original
    # lineage; wandb_run_id appends to that existing wandb run (resume="must")
    # instead of starting a new one at x=0.
    start_step: int = 0
    wandb_run_id: str | None = None
    # Private operational receipt issued after the shared runner validates an
    # explicit CLI escape hatch. init=False keeps it outside public Config
    # construction (including YAML-derived kwargs). Accepted uses are recorded
    # by launch namespace in ``wandb_resume_running_overrides``.
    _wandb_resume_running_capability: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Runner-only ownership transfer for an already-held same-host resume
    # lease. Direct train callers leave this absent and acquire in _make_logger.
    _wandb_resume_lease_handoff: object | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Immutable scientific identity of the task population and grading
    # protocol. Runners resolve this only after TaskFamily.source() has loaded
    # its cohort; resumed W&B runs must match it exactly before mutable config
    # is updated.
    protocol_identity: dict[str, str] = field(default_factory=dict)
    # Runner-owned local Docent persistence after each train rollout. Failures
    # propagate because missing analysis evidence invalidates the workload.
    on_rollout: object | None = None
    seed: int = 0


def _aggregate(trajs: list[Trajectory], prefix: str) -> dict[str, float]:
    if not trajs:
        return {f"{prefix}/n": 0.0}
    mean_reward = sum(t.reward for t in trajs) / len(trajs)
    out: dict[str, float] = {
        f"{prefix}/reward_mean": mean_reward,
        f"{prefix}/reward_std": (
            sum((t.reward - mean_reward) ** 2 for t in trajs) / len(trajs)
        )
        ** 0.5,
        f"{prefix}/n": float(len(trajs)),
    }
    keys = {k for t in trajs for k, v in t.info.items() if isinstance(v, (int, float))}
    for k in sorted(keys):
        vals = [t.info[k] for t in trajs if isinstance(t.info.get(k), (int, float))]
        out[f"{prefix}/{k}"] = sum(vals) / len(vals)
    # Per-seat breakdown: in zero-sum self-play the cross-seat means are
    # identities (reward 0, confidence 0.5) — the signal lives per seat.
    seats = {t.info.get("seat") for t in trajs if isinstance(t.info.get("seat"), str)}
    for seat in sorted(s for s in seats if s):
        seat_trajs = [t for t in trajs if t.info.get("seat") == seat]
        out[f"{prefix}/{seat}/reward_mean"] = sum(t.reward for t in seat_trajs) / len(seat_trajs)
        for k in ("judge_conf_json", "judge_conf_logit", "solution_correct", "won"):
            vals = [t.info[k] for t in seat_trajs if isinstance(t.info.get(k), (int, float))]
            if vals:
                out[f"{prefix}/{seat}/{k}"] = sum(vals) / len(vals)
    return out


def _rollout_info_metrics(env: Env, prefix: str) -> dict[str, float]:
    """Flatten env.last_rollout_info (debate env's drop counters) if present."""
    info = getattr(env, "last_rollout_info", None)
    if not info:
        return {}
    out: dict[str, float] = {}
    for k, v in info.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}/{k}"] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    out[f"{prefix}/{k}/{kk}"] = float(vv)
    return out


class _RolloutInfoAccumulator:
    """Sum rollout census counts and recompute explicitly declared ratios.

    Environments may expose ``rollout_rate_specs`` as
    ``rate_name -> (numerator_name, denominator_name)``. This prevents a
    dynamic-resampling retry from adding already-derived rates (which can
    exceed one and weights unequal rollout sizes incorrectly). Any published
    ``*_rate`` must declare its components rather than receiving an implicit
    and potentially dishonest aggregation rule.
    """

    def __init__(self, env: Env, prefix: str):
        self.env = env
        self.prefix = prefix
        self.totals: dict[str, float] = {}
        self.rates_seen: set[str] = set()

    def capture(self) -> None:
        specs = getattr(self.env, "rollout_rate_specs", {})
        for key, value in _rollout_info_metrics(self.env, self.prefix).items():
            name = key.removeprefix(f"{self.prefix}/")
            if name.endswith("_rate"):
                if name not in specs:
                    raise RuntimeError(
                        f"{type(self.env).__name__} published rollout rate {name!r} "
                        "without declaring numerator/denominator in rollout_rate_specs"
                    )
                self.rates_seen.add(name)
                continue
            self.totals[key] = self.totals.get(key, 0.0) + value

    def metrics(self) -> dict[str, float]:
        out = dict(self.totals)
        specs = getattr(self.env, "rollout_rate_specs", {})
        for rate in self.rates_seen:
            numerator, denominator = specs[rate]
            numerator_value = out.get(f"{self.prefix}/{numerator}", 0.0)
            denominator_value = out.get(f"{self.prefix}/{denominator}", 0.0)
            out[f"{self.prefix}/{rate}"] = (
                numerator_value / denominator_value if denominator_value else 0.0
            )
        return out


def _log_transcripts(cfg: "Config", step: int, env: Env, split: str) -> None:
    """Persist required local evidence; external upload stays best-effort."""
    if not cfg.log_transcripts:
        return
    transcript_run_name = getattr(cfg, "_transcript_run_name", None)
    # Local JSONL is written EVERY round; only the wandb upload obeys the
    # transcript_every cadence. Lost rollout data is unrecoverable.
    upload = bool(cfg.wandb_project) and not (
        split == "train" and cfg.transcript_every > 1 and step % cfg.transcript_every != 0
    )
    if transcript_run_name is None or cfg.transcript_dir is None:
        raise RuntimeError("transcript sink was not claimed before training")
    from infra.transcript_log import log_rollout_transcripts

    log_rollout_transcripts(
        step,
        env,
        split,
        run_name=transcript_run_name,
        launch_namespace=cfg.launch_namespace,
        output_dir=cfg.transcript_dir,
        upload=upload,
    )


def _eval_policy(policy: Policy, cfg: "Config") -> Policy:
    """The greedy-eval policy, with eval_max_tokens applied when set."""
    if cfg.eval_max_tokens is None:
        return policy
    return Policy(
        policy.backend,
        replace(policy.params, max_tokens=cfg.eval_max_tokens),
        policy.chat_template_kwargs,
    )


def evaluate(env: Env, policy: Policy, n: int, split: str = "test", prefix: str = "eval") -> dict[str, float]:
    groups = env.rollout(env.tasks(n, split=split), policy.greedy(), group_size=1)
    return _aggregate([t for g in groups for t in g], prefix) | _rollout_info_metrics(env, prefix)


def _is_degenerate(group: list[Trajectory]) -> bool:
    """No gradient signal under group normalization: fewer than 2 trajectories,
    or every trajectory carrying exactly the same reward — zero std, which is
    the condition under which grpo_pack drops the whole group. Judged on the
    shared outcome reward only; per-slot datum_rewards variance does not rescue
    a group here."""
    return len(group) < 2 or all(t.reward == group[0].reward for t in group)


def _resample_degenerate(
    env: Env,
    policy: Policy,
    group_size: int,
    groups: list[list[Trajectory]],
    retries: int,
    capture_rollout_info: Callable[[], None] | None = None,
) -> dict[str, float]:
    """DAPO dynamic sampling: replace degenerate groups in-place with rollouts
    on fresh tasks, up to `retries` rounds. A replacement that is itself
    degenerate stays in the pool for the next round; whatever is still
    degenerate at the cap stays in the batch, where grpo_pack drops it exactly
    as it would without the toggle — no behavior cliff."""
    degenerate = [i for i, g in enumerate(groups) if _is_degenerate(g)]
    n_resampled = 0
    for _ in range(retries):
        if not degenerate:
            break
        fresh = env.rollout(env.tasks(len(degenerate), "train"), policy, group_size)
        if capture_rollout_info is not None:
            capture_rollout_info()
        # env.tasks may return fewer than asked (exhausted pool); unpaired
        # slots stay degenerate rather than dropping out of the accounting.
        still = list(degenerate[len(fresh):])
        for slot, group in zip(degenerate, fresh):
            if _is_degenerate(group):
                still.append(slot)
            else:
                groups[slot] = group
                n_resampled += 1
        degenerate = still
    return {
        "train/resampled_groups": float(n_resampled),
        "train/degenerate_after_resample": float(len(degenerate)),
    }


def _select_oversampled(
    drawn: list[list[Trajectory]], batch_size: int
) -> tuple[list[list[Trajectory]], dict[str, float]]:
    """DAPO dynamic sampling, upfront variant (Config.oversample_factor):
    keep the FIRST batch_size non-degenerate groups in draw order. When the
    draw has too few, degenerate groups fill the open slots, where grpo_pack
    drops them exactly as it would without the toggle — no behavior cliff.
    Draw order (not reward or variance) decides survival, so the kept batch
    has the same per-problem distribution a serial retry chain converges to."""
    keep = [g for g in drawn if not _is_degenerate(g)][:batch_size]
    if len(keep) < batch_size:
        keep.extend(g for g in drawn if _is_degenerate(g))
        keep = keep[:batch_size]
    return keep, {
        "train/oversample_drawn": float(len(drawn)),
        "train/oversample_degenerate": float(sum(1 for g in drawn if _is_degenerate(g))),
        "train/degenerate_kept": float(sum(1 for g in keep if _is_degenerate(g))),
    }


class _Phases:
    """Wall-clock per phase of a step.

    Everything here is synchronous from Python's side — verl's calls block until
    the GPU work they wrapped is done — so plain monotonic deltas are honest.
    That is NOT true of raw torch ops, which queue asynchronously and would need
    a cuda synchronize before timing meant anything.
    """

    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.monotonic() - t0

    def metrics(self, total: float) -> dict[str, float]:
        out = {f"phase/{k}_s": v for k, v in self.t.items()}
        out["phase/unattributed_s"] = max(0.0, total - sum(self.t.values()))
        return out


def _validate_train_config(backend: Backend, cfg: Config) -> None:
    """Validate before opening external logging state."""
    if cfg.start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {cfg.start_step}")
    if cfg.start_step > cfg.steps:
        raise ValueError(
            f"start_step must be <= steps, got start_step={cfg.start_step}, "
            f"steps={cfg.steps}"
        )
    if cfg.oversample_factor < 1.0:
        raise ValueError(
            f"oversample_factor must be >= 1.0 (1.0 = off), got {cfg.oversample_factor}"
        )
    if cfg.oversample_factor > 1.0 and cfg.dynamic_sampling_retries > 0:
        raise ValueError(
            "oversample_factor and dynamic_sampling_retries are mutually "
            "exclusive: both replace degenerate groups, one upfront and one "
            "serially — set exactly one, or the step would pay for both."
        )
    if cfg.lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {cfg.lr_schedule!r}")
    if not (0.0 <= cfg.min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {cfg.min_lr_ratio}")
    if cfg.kl_mechanism not in ("advantage", "loss"):
        raise ValueError(
            f"kl_mechanism must be 'advantage' or 'loss', got {cfg.kl_mechanism!r}"
        )
    if cfg.kl_coef > 0 and cfg.kl_mechanism == "loss" and not hasattr(backend, "ref_logprobs"):
        raise ValueError(
            "kl_mechanism 'loss' needs a backend exposing ref_logprobs "
            "(verl + LoRA); this backend does not."
        )
    _validate_wandb_resume_override(cfg)


def train(env: Env, backend: Backend, cfg: Config, eval_env: Env | None = None) -> None:
    """Train while owning exactly one logger/W&B run for the full call.

    ``eval_env`` evaluates on a different env than training (e.g. debate
    training with plain-RLVR MathEnv evals). None evaluates on ``env``.
    """
    _validate_train_config(backend, cfg)
    if getattr(cfg, "_transcript_sink_used", False):
        raise FileExistsError(
            "refusing to reuse a Config whose transcript launch destination "
            "was already consumed"
        )
    backend_namespace = getattr(
        getattr(backend, "config", None), "launch_namespace", None
    )
    if backend_namespace is not None:
        backend_namespace = validate_launch_namespace(backend_namespace)
    if cfg.launch_namespace is None:
        cfg.launch_namespace = (
            backend_namespace
            if backend_namespace is not None
            else resolve_launch_namespace()
        )
    else:
        cfg.launch_namespace = validate_launch_namespace(cfg.launch_namespace)
        if (
            backend_namespace is not None
            and cfg.launch_namespace != backend_namespace
        ):
            raise ValueError(
                "backend and Config carry different launch namespaces: "
                f"backend={backend_namespace!r}, config={cfg.launch_namespace!r}"
            )
    # Production runners reserve this before constructing their backend.
    # Compatibility callers use the deterministic "run" component when they
    # have no display run name; the immutable namespace remains the provenance
    # discriminator. In every case reservation precedes W&B initialization.
    if not cfg.log_transcripts and cfg.transcript_dir is not None:
        raise ValueError("transcript_dir cannot be set when log_transcripts is false")
    transcript_run_name: str | None = None
    if cfg.log_transcripts:
        transcript_run_name = str(cfg.run_name or "run")
        expected_transcript_dir = os.path.join(
            "transcripts",
            safe_path_component(transcript_run_name, fallback="run"),
            cfg.launch_namespace,
        )
        if cfg.transcript_dir is None:
            cfg.transcript_dir = str(claim_directory(expected_transcript_dir))
        else:
            if os.path.normpath(cfg.transcript_dir) != os.path.normpath(
                expected_transcript_dir
            ):
                raise ValueError(
                    "transcript_dir must be the preclaimed directory derived from "
                    "the operational run component and launch namespace: "
                    f"expected {expected_transcript_dir}, got {cfg.transcript_dir}"
                )
            require_claimed_directory(cfg.transcript_dir)
    logger: _RunLogger | None = None
    try:
        logger = _make_logger(cfg)
        if transcript_run_name is not None:
            cfg._transcript_run_name = transcript_run_name
            cfg._transcript_sink_used = True
        _train_with_logger(env, backend, cfg, eval_env, logger)
    finally:
        close = getattr(logger, "close", None) if logger is not None else None
        if close is not None:
            close()


def _train_with_logger(
    env: Env,
    backend: Backend,
    cfg: Config,
    eval_env: Env | None,
    logger: Callable[[int, dict[str, Any]], None],
) -> None:
    policy = Policy(backend, cfg.sampling, cfg.chat_template_kwargs)
    base_optim = cfg.optim or OptimParams(lr=cfg.lr)

    for step in range(cfg.start_step, cfg.steps):
        # Schedule, indexed by the ABSOLUTE step so continuations keep the
        # lineage's position: linear warmup (0 -> lr over warmup_steps), then
        # constant or cosine decay to lr * min_lr_ratio at the final step.
        scale = 1.0
        if cfg.warmup_steps > 0:
            scale = min(1.0, (step + 1) / cfg.warmup_steps)
        if cfg.lr_schedule == "cosine" and step >= cfg.warmup_steps:
            span = max(1, cfg.steps - cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / span
            floor = cfg.min_lr_ratio
            scale = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        optim = base_optim if scale == 1.0 else replace(base_optim, lr=base_optim.lr * scale)
        t0 = time.monotonic()
        ph = _Phases()
        with ph("sync_sampler"):
            backend.sync_sampler()
        # Eval and save at the TOP of the step (Ethan, 2026-08-12): label N
        # means the policy after EXACTLY N updates — step 0 is a true
        # pre-training baseline, checkpoints carry the same meaning as their
        # matching eval points, and N/K intervals yield N/K + 1 evals with
        # the last one at x = steps (after the loop). The engine was just
        # synced, so the eval serves the current weights.
        eval_metrics: dict[str, float] = {}
        if cfg.eval_every and step % cfg.eval_every == 0:
            eval_policy = _eval_policy(policy, cfg)
            with ph("evaluate"):
                prefix = "dev" if cfg.eval_split == "dev" else "eval"
                eval_metrics.update(
                    evaluate(eval_env or env, eval_policy, cfg.eval_n, cfg.eval_split, prefix)
                )
                _log_transcripts(cfg, step, eval_env or env, prefix)
        # `step > cfg.start_step`, not `> 0`: a continuation's first step index
        # can hit the save cadence (start 25, save_every 25) and would re-save
        # step-00025 — overwriting the very checkpoint it just loaded with a
        # one-step-newer lineage under the same name.
        if cfg.save_every and step > cfg.start_step and step % cfg.save_every == 0:
            eval_metrics["checkpoint_saved"] = 1.0
            with ph("save"):
                backend.save(f"step-{step:05d}")
        with ph("rollout"):
            ds_metrics: dict[str, float] = {}
            rollout_info = _RolloutInfoAccumulator(env, "train")

            def capture_rollout_info() -> None:
                rollout_info.capture()

            if cfg.oversample_factor > 1.0:
                # One oversized generation round instead of serial retries.
                # env.tasks may return fewer than asked (exhausted pool);
                # selection keeps whatever arrived. last_rollout_records /
                # transcripts then cover ALL drawn groups, kept or not —
                # same single-final-rollout stance as the retry path below.
                n_draw = int(round(cfg.batch_size * cfg.oversample_factor))
                drawn = env.rollout(env.tasks(n_draw, "train"), policy, cfg.group_size)
                capture_rollout_info()
                groups, ds_metrics = _select_oversampled(drawn, cfg.batch_size)
            else:
                groups = env.rollout(env.tasks(cfg.batch_size, "train"), policy, cfg.group_size)
                capture_rollout_info()
                if cfg.dynamic_sampling_retries > 0:
                    ds_metrics = _resample_degenerate(
                        env,
                        policy,
                        cfg.group_size,
                        groups,
                        cfg.dynamic_sampling_retries,
                        capture_rollout_info,
                    )
        # on_rollout/_log_transcripts fire exactly once, after the final
        # resample round: env's last-rollout state (last_rollout_records /
        # last_states / last_rollout_info) reflects only the FINAL rollout, so
        # with dynamic sampling transcript capture sees just that round's
        # records. Acceptable; merging capture states across rounds is out of
        # scope.
        if cfg.on_rollout is not None:
            # Runner callbacks persist the second required local analysis view
            # (Docent JSONL). A missing file invalidates the workload, so local
            # write failures propagate instead of becoming a successful run.
            cfg.on_rollout(step, env)
        metrics = _aggregate([t for g in groups for t in g], "train")
        metrics.update(rollout_info.metrics())
        metrics.update(ds_metrics)  # keys only when the toggle is on; 0.0 otherwise
        # The lr actually applied this step — the only visible trace of the
        # warmup schedule (verl's loss/optim metrics don't carry it).
        metrics["train/lr"] = optim.lr
        _log_transcripts(cfg, step, env, "train")

        with ph("pack"):
            datums, pack_stats = grpo_pack(
                groups,
                norm_by_std=cfg.norm_adv_by_std,
                population_std=cfg.adv_population_std,
                drop_zero_advantage=cfg.drop_zero_advantage,
                length_normalize=cfg.adv_length_norm,
                shuffle_seed=cfg.seed + step,
            )
        metrics.update(pack_stats)

        if datums:
            from infra.rl.kl import apply_kl_penalty, entropy_proxy, ref_kl_metrics

            if cfg.sampling.temperature != 1.0 and hasattr(backend, "forward"):
                # Tempered sampling breaks the ratio anchor: vLLM's returned
                # logprobs come from RAW logits while the training engine
                # recomputes at logits/T, so epoch-1 ratios sit off 1.0 and
                # clipping fires on exploratory tokens before any update
                # (diff-sample audit, 2026-08-11: ~3.7% outside the clip at
                # T=0.8, step 0). Re-anchor on the training engine's own
                # forward — same pass the loss uses — restoring ratio == 1 in
                # epoch 1. One extra forward per step, only when tempered.
                with ph("anchor_logprobs"):
                    for datum, lp in zip(datums, backend.forward(datums)):
                        datum.sampler_logprobs = lp

            metrics["train/policy_token_entropy"] = entropy_proxy(datums)
            if cfg.kl_coef > 0 and cfg.kl_mechanism == "loss":
                # In-loss KL: stamp the frozen-ref logprobs once per step; the
                # backend forwards them and verl's kl_loss term (k3, per
                # minibatch vs the CURRENT policy) does the differentiable
                # pull. hw4/DeepSeekMath semantics.
                with ph("kl_ref_logprobs"):
                    for datum, lp in zip(datums, backend.ref_logprobs(datums)):
                        datum.ref_logprobs = lp
                metrics.update(ref_kl_metrics(datums))
            elif cfg.kl_coef > 0:
                # ref_logprobs is a FULL forward pass over every datum on the
                # frozen base — the cost of kl_coef, and worth seeing separately.
                with ph("kl_ref_logprobs"):
                    metrics.update(
                        apply_kl_penalty(backend, datums, cfg.kl_coef, cfg.kl_discount_factor)
                    )
            import random as _random

            for epoch in range(max(1, cfg.ppo_epochs)):
                # later epochs go off-policy; the ratio in ppo/importance losses
                # corrects against the stored sampler logprobs
                _random.Random(cfg.seed * 1000 + step * 10 + epoch).shuffle(datums)
                with ph("forward_backward"):
                    for i in range(0, len(datums), cfg.micro_batch):
                        backend.forward_backward(datums[i : i + cfg.micro_batch], cfg.loss)
                with ph("optim_step"):
                    metrics.update(backend.optim_step(optim))

        metrics.update(eval_metrics)  # top-of-step eval/save, labeled this step
        metrics["step_seconds"] = time.monotonic() - t0
        # SingleTurnEnv splits its rollout into generate/reward; fold that in so
        # "rollout" is broken down rather than opaque.
        for k, v in (getattr(env, "last_phase_seconds", None) or {}).items():
            metrics[f"phase/rollout_{k}_s"] = float(v)
        metrics.update(ph.metrics(metrics["step_seconds"]))
        logger(step, metrics)

    if cfg.eval_every or cfg.final_test_eval:
        backend.sync_sampler()
        final_policy = _eval_policy(policy, cfg)
    if cfg.eval_every:
        # Fencepost (Ethan, 2026-08-12): in-loop evals run at the TOP of each
        # step (label N = after exactly N updates), so the final policy needs
        # its own point — N/K intervals means N/K + 1 evals, the +1 at
        # x = steps, paired with the `final` checkpoint below.
        prefix = "dev" if cfg.eval_split == "dev" else "eval"
        final_env = eval_env or env
        metrics = evaluate(
            final_env, final_policy, cfg.eval_n, cfg.eval_split, prefix
        )
        _log_transcripts(cfg, cfg.steps, final_env, prefix)
        logger(cfg.steps, metrics)
    if cfg.final_test_eval:
        # The 3-way protocol's single read of the held-out test split.
        final_env = eval_env or env
        metrics = evaluate(final_env, final_policy, cfg.eval_n, "test", "test")
        _log_transcripts(cfg, cfg.steps, final_env, "test")
        logger(cfg.steps, metrics)

    backend.save("final")


def _env_identity() -> dict[str, str]:
    """Execution-environment provenance for the wandb config (Ethan,
    2026-08-10): which image/env produced a run must be readable off the run
    itself once pods are DC-independent and envs ship as tarballs. Every field
    degrades to 'unknown' rather than failing — provenance must never block
    training."""
    ident: dict[str, str] = {}
    for key, var in (
        ("env/pod_id", "RUNPOD_POD_ID"),
        ("env/image", "RUNPOD_IMAGE_NAME"),
        ("env/topology", "DEBATE_TOPOLOGY"),  # set by run_common.resolve_topology
    ):
        ident[key] = os.environ.get(var, "unknown")
    for key, mod in (("env/torch", "torch"), ("env/vllm", "vllm")):
        try:
            ident[key] = __import__(mod).__version__
        except Exception:
            ident[key] = "unknown"
    ident["env/python_prefix"] = sys.prefix  # which venv (verl-b200 vs verl-sm90)
    ver_file = os.path.join(sys.prefix, "ENV_VERSION")
    if os.path.exists(ver_file):  # stamped into portable env tarballs
        with open(ver_file) as fh:
            ident["env/tarball_version"] = fh.read().strip()
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        )
        ident["env/git_commit"] = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        ident["env/git_dirty"] = "yes" if dirty.stdout.strip() else "no"
    except Exception:
        ident["env/git_commit"] = "unknown"
    return ident


def _save_dirty_patch(run) -> None:
    """Monkey-patched runs carry their own patch (Ethan, 2026-08-10: research
    needs low-friction uncommitted launches; provenance comes from RECORDING
    the delta, not forbidding it). Saved as a file on the wandb run. Never
    blocks training."""
    try:
        import subprocess

        diff = subprocess.run(
            ["git", "diff", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout
        if diff.strip():
            path = os.path.join(run.dir, "uncommitted.patch")
            with open(path, "w") as fh:
                fh.write(diff)
            run.save(path, policy="now")
    except Exception as e:
        print(f"[wandb] dirty-patch capture skipped: {type(e).__name__}: {e}")


_ACTIVE_WANDB_RESUME_LOCKS: weakref.WeakSet["_WandbResumeLock"] = weakref.WeakSet()


def _close_wandb_resume_locks_after_fork() -> None:
    """A child must not prolong a resume lease acquired by its parent.

    ``flock`` follows the inherited open-file description across ``fork``.
    Merely marking the descriptor close-on-exec therefore leaves a fork-only
    child capable of keeping the lock alive after the trainer exits. Close all
    inherited lease descriptors in the child before it can run Python code.
    """
    for lease in tuple(_ACTIVE_WANDB_RESUME_LOCKS):
        lease._close_after_fork()
    _ACTIVE_WANDB_RESUME_LOCKS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_wandb_resume_locks_after_fork)


class _WandbResumeLock:
    """One retained, non-blocking local lease for an exact W&B run ID."""

    def __init__(self, fd: int, path: str):
        self._fd = fd
        self.path = path
        self._released = False
        _ACTIVE_WANDB_RESUME_LOCKS.add(self)

    def __del__(self) -> None:
        # CPython may discard a just-returned temporary before the caller can
        # store it when control flow arrives at that boundary. Keep the kernel
        # descriptor owned by the object itself as a last-resort cleanup.
        try:
            self.release()
        except BaseException:
            pass

    def _close_after_fork(self) -> None:
        if self._released:
            return
        self._released = True
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        _ACTIVE_WANDB_RESUME_LOCKS.discard(self)
        fd, self._fd = self._fd, -1
        if fd >= 0:
            os.close(fd)

    @property
    def held(self) -> bool:
        return not self._released and self._fd >= 0


def _wandb_resume_lock_root() -> str:
    """Persistent per-account state, independent of caller-controlled HOME."""
    return os.path.join(
        pwd.getpwuid(os.geteuid()).pw_dir,
        ".local",
        "state",
        "debate",
        "wandb-resume-locks",
    )


def _open_wandb_resume_lock_root(state_root: str | None = None) -> tuple[int, str]:
    root = os.path.abspath(state_root or _wandb_resume_lock_root())
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    created = False
    try:
        os.mkdir(root, 0o700)
        created = True
    except FileExistsError:
        pass
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise RuntimeError(
            f"unsafe W&B resume lock directory {root!r}: {exc.strerror or exc}"
        ) from exc
    try:
        if created:
            os.fchmod(root_fd, 0o700)
        info = os.fstat(root_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"W&B resume lock root is not a directory: {root}")
        if info.st_uid != os.geteuid():
            raise RuntimeError(
                f"W&B resume lock root is not owned by uid {os.geteuid()}: {root}"
            )
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise RuntimeError(
                f"W&B resume lock root must have mode 0700: {root}"
            )
        return root_fd, root
    except BaseException:
        os.close(root_fd)
        raise


def _acquire_wandb_resume_lock(
    run_id: str, *, state_root: str | None = None
) -> _WandbResumeLock:
    """Acquire a fail-fast process lease without trusting the run ID as a path.

    This protects ordinary concurrent launches by the same account. It is not
    claimed as a security boundary against another malicious process running
    as that same uid, which can manipulate the account's own state directory.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("W&B resume run ID must be a nonempty string")
    basename = hashlib.sha256(run_id.encode("utf-8")).hexdigest() + ".lock"
    root_fd, root = _open_wandb_resume_lock_root(state_root)
    fd = -1
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK
    flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        try:
            fd = os.open(
                basename,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
            created = True
        except FileExistsError:
            fd = os.open(basename, flags, dir_fd=root_fd)
        if created:
            os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        path = os.path.join(root, basename)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"W&B resume lock is not a regular file: {path}")
        if info.st_uid != os.geteuid():
            raise RuntimeError(
                f"W&B resume lock is not owned by uid {os.geteuid()}: {path}"
            )
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError(f"W&B resume lock must have mode 0600: {path}")
        if info.st_nlink != 1:
            raise RuntimeError(
                f"W&B resume lock must have exactly one hard link: {path}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another local process is already resuming W&B run {run_id!r}"
            ) from exc
        lease: _WandbResumeLock | None = None
        try:
            lease = _WandbResumeLock(fd, path)
            fd = -1
            return lease
        except BaseException:
            # Until RETURN_VALUE completes there is no caller that can own
            # the lease. In particular, an asynchronous control-flow
            # exception can land after ``fd`` moved into the object but before
            # the return. Close that object here; otherwise the outer finally
            # sees fd == -1 and cannot recover the kernel lock.
            if lease is not None:
                fd = -1
                try:
                    lease.release()
                except BaseException:
                    pass
            raise
    except OSError as exc:
        raise RuntimeError(
            f"unsafe W&B resume lock for run {run_id!r}: {exc.strerror or exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(root_fd)


class _WandbResumeLeaseHandoff:
    """One-shot transfer of an early runner lease into one exact Config."""

    __slots__ = (
        "run_id",
        "_lease",
        "_config_ref",
        "_consumed",
        "_issuer",
    )

    def __init__(
        self,
        run_id: str,
        lease: _WandbResumeLock,
        issuer: object,
    ):
        if issuer is not _WANDB_RESUME_HANDOFF_ISSUER:
            raise TypeError("W&B resume lease handoff may only be runner-issued")
        self.run_id = run_id
        self._lease: _WandbResumeLock | None = lease
        self._config_ref: weakref.ReferenceType[Config] | None = None
        self._consumed = False
        self._issuer = issuer

    def __del__(self) -> None:
        # See _WandbResumeLock.__del__: a temporary handoff can be lost after
        # its factory returns but before the caller stores the result.
        try:
            self.release()
        except BaseException:
            pass

    def bind(self, cfg: Config, raw_run_id: Any) -> None:
        if self._issuer is not _WANDB_RESUME_HANDOFF_ISSUER:
            raise ValueError("invalid W&B resume lease handoff issuer")
        if self._consumed or self._lease is None or not self._lease.held:
            raise ValueError("W&B resume lease handoff is no longer live")
        if self._config_ref is not None:
            raise ValueError("W&B resume lease handoff was already bound")
        if cfg.wandb_run_id != raw_run_id:
            raise ValueError(
                "runner Config W&B run ID does not match the validated CLI run ID"
            )
        if cfg._wandb_resume_lease_handoff is not None:
            raise ValueError("runner Config already carries a W&B resume lease")
        cfg.wandb_run_id = self.run_id
        self._config_ref = weakref.ref(cfg)
        cfg._wandb_resume_lease_handoff = self

    def take(self, cfg: Config, run_id: str) -> _WandbResumeLock:
        if (
            self._issuer is not _WANDB_RESUME_HANDOFF_ISSUER
            or self._consumed
            or self._lease is None
            or not self._lease.held
            or self.run_id != run_id
            or self._config_ref is None
            or self._config_ref() is not cfg
            or cfg._wandb_resume_lease_handoff is not self
        ):
            raise ValueError(
                "invalid, stale, reused, or mismatched W&B resume lease handoff"
            )
        lease = self._lease
        try:
            self._lease = None
            self._consumed = True
            cfg._wandb_resume_lease_handoff = None
            return lease
        except BaseException:
            # The caller does not own the lease until the return completes.
            # Abort the one-shot transfer and close it if control flow lands
            # after it left ``self`` but before it reached the caller.
            self._lease = None
            self._consumed = True
            try:
                if cfg._wandb_resume_lease_handoff is self:
                    cfg._wandb_resume_lease_handoff = None
            except BaseException:
                pass
            try:
                lease.release()
            except BaseException:
                pass
            raise

    def release(self) -> None:
        if self._lease is not None:
            self._lease.release()
            self._lease = None
        cfg = self._config_ref() if self._config_ref is not None else None
        if cfg is not None and cfg._wandb_resume_lease_handoff is self:
            cfg._wandb_resume_lease_handoff = None


def _acquire_wandb_resume_lease_handoff(
    raw_run_id: Any,
) -> _WandbResumeLeaseHandoff:
    """Validate and lock at the runner frontier without constructing an API."""
    import wandb

    run_id = _validate_wandb_resume_sdk_contract(wandb, raw_run_id)
    lease = _acquire_wandb_resume_lock(run_id)
    handoff: _WandbResumeLeaseHandoff | None = None
    try:
        handoff = _WandbResumeLeaseHandoff(
            run_id, lease, _WANDB_RESUME_HANDOFF_ISSUER
        )
        return handoff
    except BaseException:
        # Until the return completes, this factory still owns the lock. The
        # handoff finalizer covers loss of the temporary after the return but
        # before the CPython caller stores it.
        try:
            (handoff or lease).release()
        except BaseException:
            pass
        raise


def _validate_wandb_resume_override(cfg: Config) -> bool:
    # Refuse the retired public bit even when attached dynamically after
    # dataclass construction. A plain bool/object is not a runner receipt.
    if "wandb_resume_running_override" in vars(cfg):
        raise ValueError(
            "wandb_resume_running_override is not a public Config field; use "
            "the explicit shared-runner CLI flag"
        )
    if cfg.wandb_run_id is not None and not cfg.wandb_project:
        raise ValueError(
            "wandb_run_id requires W&B logging to be enabled with a project"
        )
    capability = getattr(cfg, "_wandb_resume_running_capability", None)
    if capability is None:
        return False
    if (
        type(capability) is not _WandbResumeRunningCapability
        or capability._issuer is not _WANDB_RESUME_CAPABILITY_ISSUER
        or capability.run_id != cfg.wandb_run_id
    ):
        raise ValueError(
            "invalid or mismatched W&B running-resume capability; use the "
            "explicit shared-runner CLI flag for this exact run ID"
        )
    if not cfg.wandb_project:
        raise ValueError(
            "W&B running-resume capability requires logging to be enabled"
        )
    return True


def _validate_wandb_resume_sdk_contract(wandb_module: Any, run_id: Any) -> str:
    """Validate version and raw run ID with the pinned SDK, without I/O.

    ``wandb.Settings(run_id=...)`` is the 0.28.1 validation authority used by
    ``wandb.init`` itself. Keeping that object as the oracle avoids maintaining
    a subtly different local character/type/whitespace contract.
    """
    module_version = getattr(wandb_module, "__version__", None)
    try:
        distribution_version = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "W&B resume requires installed wandb=="
            f"{_WANDB_RESUME_SDK_VERSION}; distribution metadata is missing"
        ) from exc
    if (
        module_version != _WANDB_RESUME_SDK_VERSION
        or distribution_version != _WANDB_RESUME_SDK_VERSION
    ):
        raise RuntimeError(
            "W&B resume requires exact SDK version "
            f"{_WANDB_RESUME_SDK_VERSION}; loaded={module_version!r}, "
            f"installed={distribution_version!r}"
        )
    try:
        validated = wandb_module.Settings(run_id=run_id).run_id
    except Exception as exc:
        raise ValueError(
            "invalid W&B resume run ID under wandb "
            f"{_WANDB_RESUME_SDK_VERSION}: {exc}"
        ) from exc
    if not isinstance(validated, str):
        raise RuntimeError(
            "pinned W&B SDK returned a non-string run ID after validation"
        )
    return validated


def _wandb_config_payload(cfg: Config) -> dict[str, Any]:
    """Config fields safe for ordinary W&B publication.

    The private runner capability is never published. Accepted uses have their
    own append-only namespace history instead. The retired public key is also
    stripped defensively, though validation refuses a Config carrying it.
    """
    payload = vars(cfg).copy()
    payload.pop("_wandb_resume_running_capability", None)
    payload.pop("_wandb_resume_lease_handoff", None)
    payload.pop("wandb_resume_running_override", None)
    return payload


def _validated_namespace_history(stored: dict[str, Any], key: str) -> list[str]:
    history = stored.get(key, [])
    try:
        if not isinstance(history, list):
            raise ValueError
        validated = [validate_launch_namespace(item) for item in history]
        if len(set(validated)) != len(validated):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"refusing to resume W&B run with malformed {key} history: {history!r}"
        ) from None
    return validated


def _require_ordered_namespace_subset(
    subset: list[str], history: list[str], *, subset_name: str
) -> None:
    """Require every audit entry to occur in launch history in the same order."""
    history_index = 0
    for namespace in subset:
        while (
            history_index < len(history)
            and history[history_index] != namespace
        ):
            history_index += 1
        if history_index == len(history):
            raise ValueError(
                f"refusing to resume W&B run because {subset_name} is not an "
                "order-consistent subset of launch_namespaces"
            )
        history_index += 1


class _RunLogger:
    """Best-effort W&B telemetry after any gating resume handshake.

    Fresh-run initialization, metric logging, and finish are external
    provenance only: ordinary network/client failures are reported without
    invalidating locally evidenced training. Resume identity verification,
    resume initialization, and namespace-history append happen before this
    logger is returned and remain fail-closed in ``_make_logger``.
    """

    def __init__(self, run=None, resume_lock: _WandbResumeLock | None = None):
        self.run = run
        self._resume_lock = resume_lock
        self._closed = False

    def __call__(self, step: int, metrics: dict[str, Any]) -> None:
        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as exc:
                print(
                    f"[wandb] metric log failed: {type(exc).__name__}; "
                    "local training continues",
                    file=sys.stderr,
                )
        shown = {k: round(v, 4) for k, v in sorted(metrics.items()) if isinstance(v, float)}
        print(f"[step {step}] {shown}")

    def close(self) -> None:
        """Finish the owned run at most once; no-W&B loggers are a no-op."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.run is not None:
                try:
                    self.run.finish()
                except Exception as exc:
                    print(
                        f"[wandb] finish failed: {type(exc).__name__}; "
                        "local training remains authoritative",
                        file=sys.stderr,
                    )
        finally:
            if self._resume_lock is not None:
                self._resume_lock.release()
                self._resume_lock = None


def _make_logger(cfg: Config) -> _RunLogger:
    """Open W&B with a fail-closed resume handshake and best-effort fresh run."""
    running_override = _validate_wandb_resume_override(cfg)
    cfg.launch_namespace = (
        resolve_launch_namespace()
        if cfg.launch_namespace is None
        else validate_launch_namespace(cfg.launch_namespace)
    )
    logger = _RunLogger()
    if cfg.wandb_project:
        if cfg.wandb_run_id is not None:
            # The client itself is part of the explicit resume handshake; if
            # it cannot be imported, the continuation cannot be verified.
            import wandb

            run_id = _validate_wandb_resume_sdk_contract(
                wandb, cfg.wandb_run_id
            )
            # Preserve the pinned SDK's own accepted normalization (for
            # example, its bytes-to-string coercion) in subsequent provenance
            # config as well as the lock and API path.
            cfg.wandb_run_id = run_id
            # The local lease fires immediately for the common duplicate-launch
            # case and stays live through run.finish(). It precedes every W&B
            # API operation, including the single public-run fetch below.
            handoff = getattr(cfg, "_wandb_resume_lease_handoff", None)
            if handoff is None:
                # Direct train callers have no runner frontier; preserve their
                # behavior by acquiring at the earliest point available here.
                resume_lock = _acquire_wandb_resume_lock(run_id)
            elif type(handoff) is _WandbResumeLeaseHandoff:
                resume_lock = handoff.take(cfg, run_id)
            else:
                raise ValueError("invalid W&B resume lease handoff on Config")
            run = None
            try:
                # resume="must": appending to the named run is the entire
                # point; a silent fresh run would re-split the x-axis. The
                # public object supplies BOTH state and config so the remote
                # guard adds no second API fetch.
                api = wandb.Api()
                run_path = (
                    f"{cfg.wandb_entity}/{cfg.wandb_project}/{run_id}"
                    if cfg.wandb_entity
                    else f"{cfg.wandb_project}/{run_id}"
                )
                public_run = api.run(run_path)
                try:
                    public_state = public_run.state
                except Exception as exc:
                    raise RuntimeError(
                        "refusing to resume W&B run because its remote state "
                        f"could not be read: {type(exc).__name__}: {exc}"
                    ) from exc
                terminal_states = {"finished", "crashed", "failed", "killed"}
                if not isinstance(public_state, str) or public_state not in (
                    terminal_states | {"running"}
                ):
                    raise RuntimeError(
                        "refusing to resume W&B run with unknown or missing "
                        f"remote state: {public_state!r}"
                    )
                if public_state == "running":
                    if not running_override:
                        raise RuntimeError(
                            "refusing to resume W&B run while its remote state "
                            "is 'running'; after confirming the old process is "
                            "dead, retry with --wandb-resume-running-override"
                        )
                    override_used = True
                else:
                    if running_override:
                        raise ValueError(
                            "--wandb-resume-running-override is unnecessary: "
                            f"W&B run state is already terminal ({public_state!r})"
                        )
                    override_used = False

                stored_config = dict(public_run.config)
                stored_identity = stored_config.get("protocol_identity")
                current_identity = dict(cfg.protocol_identity)
                if stored_identity != current_identity:
                    raise ValueError(
                        "refusing to resume W&B run with a different or missing "
                        f"protocol_identity: stored={stored_identity!r}, "
                        f"current={current_identity!r}"
                    )
                stored_namespaces = _validated_namespace_history(
                    stored_config, "launch_namespaces"
                )
                override_namespaces = _validated_namespace_history(
                    stored_config, "wandb_resume_running_overrides"
                )
                _require_ordered_namespace_subset(
                    override_namespaces,
                    stored_namespaces,
                    subset_name="wandb_resume_running_overrides",
                )
                if cfg.launch_namespace in stored_namespaces:
                    raise ValueError(
                        "refusing to reuse launch namespace already recorded by "
                        f"this W&B run: {cfg.launch_namespace}"
                    )
                launch_namespaces = [*stored_namespaces, cfg.launch_namespace]
                if override_used:
                    if cfg.launch_namespace in override_namespaces:
                        raise ValueError(
                            "refusing to duplicate a W&B running-state override "
                            f"record for launch namespace {cfg.launch_namespace}"
                        )
                    override_namespaces = [
                        *override_namespaces,
                        cfg.launch_namespace,
                    ]
                    print(
                        "[wandb] explicit running-state resume override accepted "
                        f"for run {run_id!r}, launch namespace "
                        f"{cfg.launch_namespace!r}",
                        file=sys.stderr,
                    )
                run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    id=run_id,
                    resume="must",
                    # An inherited WANDB_MODE=offline/disabled would otherwise
                    # make resume semantics inapplicable. Explicit call
                    # settings take precedence in the pinned SDK.
                    mode="online",
                )
                try:
                    run_finish = run.finish
                    run_config_update = run.config.update
                except Exception as exc:
                    raise RuntimeError(
                        "W&B resume initialization returned an invalid run "
                        f"object: {type(exc).__name__}: {exc}"
                    ) from exc
                if not callable(run_finish) or not callable(run_config_update):
                    raise RuntimeError(
                        "W&B resume initialization returned an invalid run "
                        "object without callable finish/config.update"
                    )
                logger = _RunLogger(run, resume_lock=resume_lock)
                resume_lock = None
                # Continuations may intentionally change operational knobs such
                # as lr or steps. Update those only after identity equality,
                # and omit identity so its immutable value is not rewritten.
                mutable_config = _wandb_config_payload(cfg)
                mutable_config.pop("protocol_identity")
                run.config.update(
                    mutable_config
                    | _env_identity()
                    | {
                        "launch_namespaces": launch_namespaces,
                        "wandb_resume_running_overrides": override_namespaces,
                    },
                    allow_val_change=True,
                )
            except BaseException:
                if run is not None and logger.run is run:
                    try:
                        logger.close()
                    except BaseException:
                        pass
                else:
                    # wandb.init may have created a remote running attempt
                    # before an interrupt prevented _RunLogger from owning it.
                    # Finish that independently tracked object before dropping
                    # the local lease so it does not remain remotely running
                    # without its namespace-history update.
                    if run is not None:
                        try:
                            finish = getattr(run, "finish", None)
                            if callable(finish):
                                finish()
                        except BaseException:
                            pass
                    if resume_lock is not None:
                        try:
                            resume_lock.release()
                        except BaseException:
                            pass
                raise
        else:
            try:
                import wandb
            except Exception as exc:
                print(
                    f"[wandb] fresh client unavailable: {type(exc).__name__}; "
                    "continuing with local evidence only",
                    file=sys.stderr,
                )
                return logger
            try:
                run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,   # None -> wandb's default entity
                    name=cfg.run_name,
                    config=_wandb_config_payload(cfg)
                    | _env_identity()
                    | {"launch_namespaces": [cfg.launch_namespace]},
                )
            except Exception as exc:
                print(
                    f"[wandb] fresh init failed: {type(exc).__name__}; "
                    "continuing with local evidence only",
                    file=sys.stderr,
                )
                return logger
            logger = _RunLogger(run)
        try:
            if run is not None and dict(run.config).get("env/git_dirty") == "yes":
                _save_dirty_patch(run)
        except BaseException as exc:
            if isinstance(exc, Exception):
                print(
                    f"[wandb] post-init provenance check failed: "
                    f"{type(exc).__name__}; local training continues",
                    file=sys.stderr,
                )
            else:
                # The resume lease has already moved into ``logger``, but
                # train() has not received it yet and therefore cannot close
                # it in its own finally block. Release here while preserving
                # the original control-flow exception even if finish fails.
                try:
                    logger.close()
                except BaseException:
                    pass
                raise
    return logger


def resolve_protocol_identity(dataset_type: str, family: Any) -> dict[str, str]:
    """Combine the registry key with a family's resolved protocol metadata.

    ``dataset_type`` is runner-owned because it selects the registry entry;
    allowing a family to restate it would create two competing authorities.
    Call this only after ``family.source()`` has resolved its cohort.
    """
    if not isinstance(dataset_type, str) or not dataset_type.strip():
        raise ValueError(
            f"dataset_type must be a nonempty string, got {dataset_type!r}"
        )
    family_identity = family.protocol_identity()
    if not isinstance(family_identity, dict):
        raise ValueError(
            "TaskFamily.protocol_identity() must return dict[str, str], got "
            f"{type(family_identity).__name__}"
        )
    if "dataset_type" in family_identity:
        raise ValueError(
            "TaskFamily.protocol_identity() must not include reserved key "
            "'dataset_type'; the runner-owned registry key is authoritative"
        )
    invalid = [
        (key, value)
        for key, value in family_identity.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid:
        raise ValueError(
            "TaskFamily.protocol_identity() must return dict[str, str]; "
            f"invalid entries: {invalid!r}"
        )
    return {"dataset_type": dataset_type} | family_identity


def validate_resume_args(args: Any) -> None:
    """Reject continuation flags that cannot identify a valid lineage."""
    start_step = getattr(args, "start_step", None)
    if start_step is not None and start_step < 0:
        raise ValueError(f"--start-step must be >= 0, got {start_step}")

    load = getattr(args, "load", None)
    checkpoint_step = None
    if load:
        basename = str(load).rstrip("/").rsplit("/", 1)[-1]
        match = re.fullmatch(r"step-(\d+)", basename)
        if match is not None:
            checkpoint_step = int(match.group(1))
    if (
        start_step is not None
        and checkpoint_step is not None
        and start_step != checkpoint_step
    ):
        raise ValueError(
            f"--start-step {start_step} does not match checkpoint basename "
            f"step-{checkpoint_step:05d}"
        )

    wandb_resume = getattr(args, "wandb_resume", None)
    running_override = getattr(args, "wandb_resume_running_override", False)
    if running_override and wandb_resume is None:
        raise ValueError(
            "--wandb-resume-running-override requires --wandb-resume RUN_ID"
        )
    if wandb_resume is None:
        return
    if getattr(args, "no_wandb", False):
        raise ValueError("--wandb-resume cannot be combined with --no-wandb")
    if not load:
        raise ValueError("--wandb-resume requires --load <checkpoint>")
    # A step-N checkpoint is saved before update N and therefore carries its
    # absolute continuation step in the name. ``final`` and service-owned
    # opaque checkpoint URIs carry optimizer state but no portable step
    # metadata. Guessing zero would restart the W&B x-axis, LR schedule, and
    # step-seeded shuffles, so those require the operator to state the step.
    if start_step is None and checkpoint_step is None:
        raise ValueError(
            "--wandb-resume with a final or opaque checkpoint requires an "
            "explicit --start-step; only load paths ending in step-N encode "
            "their absolute continuation step"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["math"], default="math")
    parser.add_argument("--base-model", default=Config.base_model)
    parser.add_argument("--lora-rank", type=int, default=Config.lora_rank)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--group-size", type=int, default=Config.group_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--loss", choices=["ppo", "importance_sampling"], default="ppo")
    parser.add_argument("--eval-every", type=int, default=Config.eval_every)
    parser.add_argument("--eval-n", type=int, default=Config.eval_n)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--load", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    cfg = Config(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        lr=args.lr,
        loss=LossSpec(kind=args.loss),
        sampling=SamplingParams(max_tokens=args.max_tokens),
        eval_every=args.eval_every,
        eval_n=args.eval_n,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    from infra.backend.tinker import TinkerBackend
    from infra.envs.tasks import get_family

    backend = TinkerBackend(cfg.base_model, lora_rank=cfg.lora_rank)
    if args.load:
        backend.load(args.load)
    family = get_family(args.env)
    try:
        env = family.source({"seed": cfg.seed})
        cfg.protocol_identity = resolve_protocol_identity(args.env, family)
        train(env, backend, cfg)
    finally:
        family.close()


if __name__ == "__main__":
    main()
