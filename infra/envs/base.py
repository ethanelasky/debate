"""Env layer: one rollout path for training AND evaluation.

Training calls rollout(train tasks, temp>0, group_size=G); evaluation is
rollout(test tasks, temp 0, group_size=1) plus aggregation. There is no other
eval machinery.
"""

from __future__ import annotations

import concurrent.futures
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from infra.backend.base import Backend, Datum, Region, Sample, SamplingParams

Message = dict[str, str]

THINK_CLOSE = "</think>"
FORCED_CLOSE_TEXT = "</think>\n\n"


@dataclass(frozen=True)
class SlotLimits:
    """The three hard per-slot token caps (DESIGN-debate-env.md §3)."""

    max_think_tokens: Optional[int] = None
    max_visible_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None

    @property
    def two_phase(self) -> bool:
        return self.max_think_tokens is not None or self.max_visible_tokens is not None


@dataclass
class Task:
    messages: list[Message]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    datums: list[Datum]  # one per policy turn; advantages filled by grpo_pack
    reward: float
    info: dict[str, Any] = field(default_factory=dict)
    # Optional per-datum rewards (slot-specific bonuses on top of the shared
    # outcome). When set, grpo_pack normalizes each datum position against the
    # same position across its group, giving every slot its own baseline.
    # None = every datum shares `reward` (identical to the uniform case).
    datum_rewards: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if self.datum_rewards is not None and len(self.datum_rewards) != len(self.datums):
            raise ValueError(
                f"datum_rewards has {len(self.datum_rewards)} entries for {len(self.datums)} datums"
            )


class Policy:
    """Chat-template rendering over backend.sample().

    Presents a batch predict() so multi-turn env code (the salvaged debate
    round loop) can treat it like the old repo's Model interface. Samples come
    back with prompt_tokens attached so envs can build Datums directly.
    """

    def __init__(self, backend: Backend, params: SamplingParams, chat_template_kwargs: dict | None = None):
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.params = params
        self.chat_template_kwargs = chat_template_kwargs or {}

    def render(self, messages: list[Message]) -> list[int]:
        out = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self.chat_template_kwargs
        )
        if not isinstance(out, list):  # BatchEncoding from some tokenizer wrappers
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def predict(
        self, convos: list[list[Message]], n: int = 1, limits: Optional[SlotLimits] = None
    ) -> list[list[Sample]]:
        """result[i] holds the n samples for convos[i]. With two-phase limits
        set, runs budget-forced sampling (think capped at max_think_tokens with
        a forced </think> injection; visible capped separately)."""
        prompts = [self.render(c) for c in convos]
        params = self.params
        if limits is not None and limits.max_total_tokens is not None:
            ceiling = params.max_tokens
            params = replace(
                params,
                max_tokens=(
                    limits.max_total_tokens if ceiling is None else min(ceiling, limits.max_total_tokens)
                ),
            )
        if params.max_tokens is None and not (limits is not None and limits.two_phase and limits.max_visible_tokens):
            raise ValueError(
                "no token budget for this generation: set the slot's max_total_tokens "
                "(or SamplingParams.max_tokens)"
            )
        if limits is not None and limits.two_phase:
            results = budget_forced_sample(
                self.backend.sample, self.tokenizer, prompts, params, limits, n=n
            )
        else:
            results = self.backend.sample(prompts, params, n=n)
        for prompt, samples in zip(prompts, results):
            for s in samples:
                s.prompt_tokens = prompt
        return results

    def greedy(self) -> "Policy":
        # Full reset of the sampling profile, not just temperature: the eval
        # contract is "greedy, full distribution, no length floor". top_p is
        # a no-op at temp 0 today but say so explicitly; min_tokens is a
        # TRAINING-sampler knob (exploration floor) and must not leak into
        # eval decoding (hw4's eval has no min_new_tokens).
        return Policy(
            self.backend,
            replace(self.params, temperature=0.0, top_p=1.0, min_tokens=None),
            self.chat_template_kwargs,
        )


class Env(ABC):
    @abstractmethod
    def tasks(self, n: int, split: str = "train") -> list[Task]: ...

    @abstractmethod
    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        """One group of rewarded trajectories per task. Samples failing
        fidelity checks are dropped (and counted in last_rollout_info), not
        trained on."""


def _validate_predict_results(
    results: list[list[Sample]], *, requests: int, samples_per_request: int, stage: str
) -> None:
    """Enforce Policy.predict's rectangular result contract at env boundaries."""
    if len(results) != requests:
        raise RuntimeError(
            f"{stage} generation returned {len(results)} result groups for "
            f"{requests} requests"
        )
    for i, samples in enumerate(results):
        if len(samples) != samples_per_request:
            raise RuntimeError(
                f"{stage} generation request {i} returned {len(samples)} samples; "
                f"expected exactly {samples_per_request}"
            )


class SingleTurnEnv(Env):
    """One generation per task, scored by reward(): the RLVR rollout shape,
    shared by every task-source env (math, codecontests, ...).

    Subclasses implement tasks() and reward(). The only axis they differ on is
    `grade_workers`: > 1 runs reward() in a thread pool, for verifiers that
    shell out to a subprocess (code execution). Pure-python graders leave it at
    1 and score inline rather than pay for a pool they cannot use.

    Each rollout() overwrites `last_rollout_records` — one dict per KEPT
    sample (prompt messages, completion, reward, info) — retained for
    transcript export (wandb; the single-turn twin of DebateEnv.last_states).
    ``export_meta()`` is the one boundary where task-private grader state is
    removed.  Consumers must only ever see the already-redacted records.
    """

    grade_workers: int = 1
    last_rollout_records: list[dict[str, Any]] = []
    # Batch-level coverage belongs here rather than on an arbitrary
    # trajectory: it must remain observable when a group (or the whole batch)
    # has no fidelity-kept samples.
    last_rollout_info: dict[str, int | float] = {}

    #: Flat overshoot penalty. A sample longer than `soft_token_budget` loses
    #: `overshoot_penalty` from its reward — a constant, NOT scaled by how far
    #: over it went. The old repo ramped the penalty between a soft budget and
    #: a hard limit (coef 0.10 at 4096/8192 cost a half-way overshoot 0.05);
    #: flat is the deliberate change, so the gradient says "stay under the
    #: budget" rather than "be marginally shorter".
    #:
    #: This does NOT change the sampler cap: generation still stops at the
    #: backend's response_length. It only prices length in the reward, so a
    #: budget above that cap can never fire. Off by default.
    soft_token_budget: Optional[int] = None
    overshoot_penalty: float = 0.0

    #: Hard per-generation token caps for EVERY rollout this env runs — train
    #: and eval alike, since both go through this one rollout(). Injected by
    #: run_rlvr when the experiment sets think_tokens (native-<think> arms):
    #: predict() then runs budget_forced_sample, which force-injects </think>
    #: at the think cap and marks the injection as a "forced_close" region on
    #: the Sample (masked from training; reward_sample overrides may price
    #: it). None = the plain single-phase path.
    slot_limits: Optional[SlotLimits] = None

    @abstractmethod
    def reward(self, task: Task, text: str) -> tuple[float, dict[str, Any]]:
        """Completion text -> (reward, metric info), one call per kept sample.
        Every info key should be present on EVERY branch, so eval-time averages
        are means over all samples rather than over whichever branch set it."""

    def reward_sample(self, task: Task, sample: Sample) -> tuple[float, dict[str, Any]]:
        """Sample-level scoring seam: rollout() routes every kept sample
        through here — the inline AND the grade_workers pool branch — so a
        grader that needs more than the text (e.g. MB's confidence tiebreaker,
        which reads the sampler's per-token logprobs) can override this
        without forking the rollout shape. The base implementation is EXACTLY
        `self.reward(task, sample.text)`: subclasses that only implement
        reward() see no behavior change."""
        return self.reward(task, sample.text)

    def export_meta(self, task: Task) -> dict[str, Any]:
        """Return the task metadata safe to retain in transcript records.

        Most single-turn tasks only use ``bindings`` as a large prompt-render
        payload, so the historical default copies everything except that key.
        Environments whose metadata also contains private verifier material
        must override this method with an explicit allowlist.  Keeping the
        policy here makes ``last_rollout_records`` safe before any Docent or
        W&B exporter receives it.
        """
        return {k: v for k, v in task.meta.items() if k != "bindings"}

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        # Clear externally observed state before rendering or generation. If
        # either fails, callers must not mistake the previous rollout's
        # records or coverage for this one.
        self.last_rollout_records = []
        self.last_rollout_info = {
            "tasks_requested": len(tasks),
            "samples_attempted": 0,
            "samples_kept": 0,
            "samples_dropped_fidelity": 0,
        }
        # Split generation from scoring: they are the two halves of a rollout and
        # have completely different cost drivers (GPU token throughput vs CPU
        # subprocess execution), so a combined number hides which one to attack.
        _t0 = time.monotonic()
        # slot_limits None must leave the predict CALL byte-identical, not
        # just equivalent: existing policy stubs (tests) accept no `limits`
        # kwarg, and the default path must never depend on them doing so.
        if self.slot_limits is None:
            results = policy.predict([t.messages for t in tasks], n=group_size)
        else:
            results = policy.predict(
                [t.messages for t in tasks], n=group_size, limits=self.slot_limits
            )
        _t_generate = time.monotonic() - _t0
        _validate_predict_results(
            results,
            requests=len(tasks),
            samples_per_request=group_size,
            stage="single-turn",
        )
        groups: list[list[Trajectory]] = [[] for _ in tasks]
        kept: list[tuple[int, Sample]] = []
        n_dropped = 0
        n_attempted = sum(len(samples) for samples in results)
        for gi, samples in enumerate(results):
            for s in samples:
                if not s.fidelity_ok():
                    n_dropped += 1
                    continue
                kept.append((gi, s))

        # Overwritten on every rollout, before scoring, so fidelity coverage
        # remains auditable even when no Trajectory can be constructed.
        self.last_rollout_info = {
            "tasks_requested": len(tasks),
            "samples_attempted": n_attempted,
            "samples_kept": len(kept),
            "samples_dropped_fidelity": n_dropped,
        }

        _t1 = time.monotonic()
        workers = max(1, self.grade_workers)
        if workers > 1 and kept:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self.reward_sample, tasks[gi], s) for gi, s in kept]
                future_indices = {future: index for index, future in enumerate(futures)}
                ordered_scores: list[tuple[float, dict[str, Any]] | None] = [
                    None
                ] * len(futures)
                try:
                    # Observe whichever verifier finishes first rather than
                    # blocking on submission order.  If one result proves the
                    # grading boundary is broken, cancel every request that has
                    # not started yet; executing the rest of a now-invalid eval
                    # only delays trainer teardown and burns paid compute.
                    for future in concurrent.futures.as_completed(futures):
                        ordered_scores[future_indices[future]] = future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
                if any(score is None for score in ordered_scores):
                    raise RuntimeError("parallel reward scoring completed incompletely")
                scored = [score for score in ordered_scores if score is not None]
        else:
            scored = [self.reward_sample(tasks[gi], s) for gi, s in kept]

        self.last_phase_seconds = {"generate": _t_generate, "reward": time.monotonic() - _t1}

        records = []
        for (gi, s), (reward, info) in zip(kept, scored):
            # Length is priced here rather than in reward(), which only sees
            # text; token counts live on the Sample. Applied BEFORE the
            # Trajectory and the record are built so both carry the same
            # penalized reward the optimizer sees.
            over = bool(self.soft_token_budget and len(s.tokens) > self.soft_token_budget)
            info = {
                **info,
                "tokens": float(len(s.tokens)),
                "truncated": float(s.stop_reason == "length"),
                "over_budget": float(over),
            }
            if over:
                reward -= self.overshoot_penalty
            groups[gi].append(Trajectory(datums=[datum_from_sample(s)], reward=reward, info=info))
            records.append(
                {
                    "task_index": gi,
                    "meta": self.export_meta(tasks[gi]),
                    "messages": tasks[gi].messages,
                    "completion": s.text,
                    "stop_reason": s.stop_reason,
                    "reward": reward,
                    "info": info,
                }
            )
        self.last_rollout_records = records
        return groups


def datum_from_sample(s: Sample) -> Datum:
    assert s.prompt_tokens is not None, "sample must come from Policy.predict"
    mask = None
    if s.regions is not None and any(r.kind == "forced_close" for r in s.regions):
        mask = [1.0] * len(s.tokens)
        for r in s.regions:
            if r.kind == "forced_close":
                for i in range(r.start, r.end):
                    mask[i] = 0.0  # injected, unsampled: never trained on
    return Datum(
        tokens=s.prompt_tokens + s.tokens,
        prompt_len=len(s.prompt_tokens),
        sampler_logprobs=s.logprobs,
        advantages=[0.0] * len(s.tokens),
        mask=mask,
    )


# ------------------------------------------------ budget-forced sampling


SampleFn = Callable[..., list[list[Sample]]]  # Backend.sample signature


def _close_ids(tokenizer) -> list[int]:
    ids = tokenizer.encode(FORCED_CLOSE_TEXT, add_special_tokens=False)
    if tokenizer.decode(ids) != FORCED_CLOSE_TEXT:
        raise ValueError("forced-close text does not round-trip through this tokenizer")
    return list(ids)


def budget_forced_sample(
    sample_fn: SampleFn,
    tokenizer,
    prompts: list[list[int]],
    params: SamplingParams,
    limits: SlotLimits,
    n: int = 1,
) -> list[list[Sample]]:
    """Two-phase sampling enforcing hard think/visible caps.

    Phase 1 samples with stop </think> capped at the think budget; a cap hit
    gets </think> force-injected (logprob 0.0, datum mask 0.0 via regions).
    Phase 2 continues from the extended prefix under the visible/total caps.
    A model that never opens <think> (and wasn't given one by the template)
    is passed through as all-visible — think caps only bind on thinking — and
    if phase 1's cap cut it off, generation continues under the remaining
    visible/total budget (no injection).
    """
    close = _close_ids(tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    inf = 10**9
    total = min(limits.max_total_tokens or inf, params.max_tokens or inf)
    if total >= inf and (limits.max_think_tokens is None or limits.max_visible_tokens is None):
        raise ValueError(
            "unbounded two-phase generation: set max_total_tokens, or both "
            "max_think_tokens and max_visible_tokens"
        )
    p1_cap = min(limits.max_think_tokens or inf, max(1, total - len(close) - 1))

    phase1 = sample_fn(prompts, replace(params, max_tokens=p1_cap, stop=[THINK_CLOSE]), n=n)
    _validate_predict_results(
        phase1,
        requests=len(prompts),
        samples_per_request=n,
        stage="two-phase phase 1",
    )

    # Flatten, decide per sample, bucket phase-2 continuations by their cap.
    flat: list[dict] = []
    for pi, samples in enumerate(phase1):
        for s in samples:
            prompt_tail = tokenizer.decode(prompts[pi][-16:])
            decoded = s.text or tokenizer.decode(s.tokens)
            in_think = "<think>" in decoded or ("<think>" in prompt_tail and THINK_CLOSE not in prompt_tail)
            entry: dict = {"pi": pi, "p1": s, "close": [], "p2": None, "forced": False}
            if not in_think:
                # never thought: everything visible, single-phase semantics
                entry["all_visible"] = True
            elif decoded.rstrip().endswith(THINK_CLOSE):
                pass  # natural close (sampled, real logprobs); phase 2 continues
            elif s.tokens and eos_id is not None and s.tokens[-1] == eos_id:
                entry["died_in_think"] = True
            else:
                entry["close"] = close
                entry["forced"] = True
            flat.append(entry)

    buckets: dict[int, list[int]] = {}
    for j, e in enumerate(flat):
        if e.get("died_in_think"):
            continue
        if e.get("all_visible"):
            # The think cap must not become a total cap: a never-thinking
            # sample cut at the phase-1 cap continues under the remaining
            # visible/total budget (its close is empty, so the shared bucket
            # pass extends the raw prefix with no injection). Every phase-1
            # token here is visible, so it counts against max_visible_tokens.
            cont_cap = min(limits.max_visible_tokens or inf, total) - len(e["p1"].tokens)
            if e["p1"].stop_reason != "length" or cont_cap <= 0:
                continue
            e["p2_cap"] = cont_cap
            buckets.setdefault(cont_cap, []).append(j)
            continue
        p2_cap = min(limits.max_visible_tokens or inf, total - len(e["p1"].tokens) - len(e["close"]))
        if p2_cap <= 0:
            e["exhausted"] = True
            continue
        e["p2_cap"] = p2_cap
        buckets.setdefault(p2_cap, []).append(j)

    for p2_cap, idxs in buckets.items():
        prefixes = [prompts[flat[j]["pi"]] + flat[j]["p1"].tokens + flat[j]["close"] for j in idxs]
        outs = sample_fn(prefixes, replace(params, max_tokens=p2_cap), n=1)
        _validate_predict_results(
            outs,
            requests=len(prefixes),
            samples_per_request=1,
            stage="two-phase phase 2",
        )
        for j, out in zip(idxs, outs):
            flat[j]["p2"] = out[0]

    results: list[list[Sample]] = [[] for _ in prompts]
    for e in flat:
        p1: Sample = e["p1"]
        if e.get("all_visible"):
            cont: Optional[Sample] = e["p2"]
            tokens = p1.tokens + (cont.tokens if cont else [])
            results[e["pi"]].append(
                Sample(
                    tokens=tokens,
                    logprobs=p1.logprobs + (cont.logprobs if cont else []),
                    text=tokenizer.decode(tokens),
                    stop_reason=cont.stop_reason if cont is not None else p1.stop_reason,
                    regions=(Region("visible", 0, len(tokens)),),
                )
            )
            continue
        n1 = len(p1.tokens)
        if e.get("died_in_think"):
            results[e["pi"]].append(
                Sample(
                    tokens=p1.tokens,
                    logprobs=p1.logprobs,
                    text=tokenizer.decode(p1.tokens),
                    stop_reason="stop",
                    regions=(Region("think", 0, n1),),
                )
            )
            continue
        close_toks: list[int] = e["close"]
        p2: Optional[Sample] = e["p2"]
        tokens = p1.tokens + close_toks + (p2.tokens if p2 else [])
        logprobs = p1.logprobs + [0.0] * len(close_toks) + (p2.logprobs if p2 else [])
        regions = [Region("think", 0, n1)]
        if close_toks:
            regions.append(Region("forced_close", n1, n1 + len(close_toks)))
        if p2 and p2.tokens:
            start = n1 + len(close_toks)
            regions.append(Region("visible", start, start + len(p2.tokens)))
        stop_reason = p2.stop_reason if p2 is not None else "length"  # exhausted: no room for phase 2
        results[e["pi"]].append(
            Sample(
                tokens=tokens,
                logprobs=logprobs,
                text=tokenizer.decode(tokens),
                stop_reason=stop_reason,
                regions=tuple(regions),
            )
        )
    return results
