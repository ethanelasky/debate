"""Plan-then-answer rollouts: a two-turn wrapper over a SingleTurnEnv.

The debate protocols give the proposer a private scratchpad turn before its
solution slot (the CoT surrogate when thinking is off). Without this wrapper
the RLVR arm answers cold in one turn, which confounds the debate-vs-RLVR
comparison twice over: the debate seat gets extra sequential compute, and the
in-training RLVR eval measures the debate-trained policy in a mode it was
never trained in. PlannedEnv is the matching RLVR shape:

    turn 1  [system?] + PLAN_USER cue        -> private plan (capped)
    turn 2  ... + assistant(plan) + the task's own eliciting message(s)
                                             -> answer, scored by inner.reward

The plan cue is the task family's `plan` template (task_prompts.py), the same
bytes the debate pack splices as its pre-solution `plan` slot cue — so the
plan turn and the answer turn each render byte-identically across the two
arms (modulo the system card, the one deliberate difference).

Both turns' datums train under the shared trajectory reward (Trajectory is
explicitly one-datum-per-policy-turn); the plan gets no reward of its own.
"""

from __future__ import annotations

from typing import Any

from infra.envs.base import (
    Env,
    Policy,
    SingleTurnEnv,
    SlotLimits,
    Task,
    Trajectory,
    datum_from_sample,
)
from infra.envs.task_prompts import _PLACEHOLDER, PLAN_TEMPLATE_NAME, PROBLEM_PLACEHOLDER


class PlannedEnv(Env):
    """Wraps a task-source env whose GenerationPrompts supply a `plan` cue.

    The inner env keeps ownership of tasks() and reward(); this class owns
    only the two-turn rollout shape. Works for any family whose Task.meta
    carries the problem text as meta["question"] and whose plan template
    binds it via <PROBLEM> (math, codecontests)."""

    def __init__(self, inner: SingleTurnEnv, plan_max_tokens: int):
        prompts = getattr(inner, "prompts", None)
        named = prompts.supplied_templates() if prompts is not None else {}
        if PLAN_TEMPLATE_NAME not in named:
            raise ValueError(
                f"plan-then-answer rollouts need a `plan` template, but "
                f"{type(inner).__name__}'s prompt config supplies "
                f"{sorted(named) or 'none'} — add a `plan` key to the task's "
                "answer-generation config (infra/prompts/tasks/<family>.yaml)"
            )
        if plan_max_tokens <= 0:
            raise ValueError(f"plan_max_tokens must be positive, got {plan_max_tokens}")
        self.inner = inner
        self.plan_template = named[PLAN_TEMPLATE_NAME]
        self.plan_max_tokens = int(plan_max_tokens)
        self.last_rollout_records: list[dict[str, Any]] = []

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        return self.inner.tasks(n, split)

    def _plan_cue(self, task: Task) -> str:
        question = task.meta.get("question")
        if question is None:
            raise ValueError(
                f"plan-then-answer rollout: task from {type(self.inner).__name__} has no "
                'meta["question"] to bind the plan cue\'s <PROBLEM> placeholder'
            )
        # Same strictness as GenerationPrompts.render: the only placeholder a
        # plan cue may carry is the row-content one, single-pass substituted.
        extra = sorted(set(_PLACEHOLDER.findall(self.plan_template)) - {"PROBLEM"})
        if extra:
            raise ValueError(f"unbound placeholder(s) {extra} in plan template")
        return self.plan_template.replace(PROBLEM_PLACEHOLDER, str(question)).strip()

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        # Turn 1: the task's leading system message (if any) + the plan cue.
        plan_convos: list[list[dict[str, str]]] = []
        for t in tasks:
            head = [dict(m) for m in t.messages[:1] if m["role"] == "system"]
            plan_convos.append(head + [{"role": "user", "content": self._plan_cue(t)}])
        plan_results = policy.predict(
            plan_convos, n=group_size, limits=SlotLimits(max_total_tokens=self.plan_max_tokens)
        )

        # Turn 2: each surviving plan continues into the task's own messages
        # (everything after the leading system message — the eliciting cue,
        # rendered by the inner env exactly as its single-turn arm would).
        pending: list[tuple[int, Any, list[dict[str, str]]]] = []
        n_dropped = 0
        for ti, samples in enumerate(plan_results):
            body = [dict(m) for m in tasks[ti].messages if m["role"] != "system"]
            for s in samples:
                if not s.fidelity_ok():
                    n_dropped += 1
                    continue
                convo = plan_convos[ti] + [{"role": "assistant", "content": s.text.strip()}] + body
                pending.append((ti, s, convo))
        answer_results = (
            policy.predict([c for _, _, c in pending], n=1) if pending else []
        )

        groups: list[list[Trajectory]] = [[] for _ in tasks]
        records: list[dict[str, Any]] = []
        for (ti, plan_s, convo), answers in zip(pending, answer_results):
            a = answers[0]
            if not a.fidelity_ok():
                n_dropped += 1
                continue
            reward, info = self.inner.reward(tasks[ti], a.text)
            groups[ti].append(
                Trajectory(
                    datums=[datum_from_sample(plan_s), datum_from_sample(a)],
                    reward=reward,
                    info=info,
                )
            )
            records.append(
                {
                    "task_index": ti,
                    "meta": {k: v for k, v in tasks[ti].meta.items() if k != "bindings"},
                    "messages": convo,
                    "completion": a.text,
                    "stop_reason": a.stop_reason,
                    "reward": reward,
                    "info": info,
                }
            )
        self.last_rollout_records = records
        # Same deliberate contract violation as SingleTurnEnv.rollout: a
        # batch-level counter stamped on one sample.
        if n_dropped and groups and groups[0]:
            groups[0][0].info["samples_dropped_fidelity"] = float(n_dropped)
        return groups
