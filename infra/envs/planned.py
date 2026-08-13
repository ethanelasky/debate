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
    _validate_predict_results,
)
from infra.envs.task_prompts import (
    _PLACEHOLDER,
    ANSWER_TEMPLATE_NAME,
    PLAN_TEMPLATE_NAME,
    PROBLEM_PLACEHOLDER,
)


class PlannedEnv(Env):
    """Wraps a task-source env whose GenerationPrompts supply a `plan` cue.

    The inner env keeps ownership of tasks() and reward(); this class owns
    only the two-turn rollout shape. Works for any family whose Task.meta
    carries the problem text as meta["question"] and whose plan template
    binds it via <PROBLEM> (math, codecontests)."""

    def __init__(self, inner: SingleTurnEnv, plan_max_tokens: int, answer_max_tokens: int | None = None):
        prompts = getattr(inner, "prompts", None)
        named = prompts.supplied_templates() if prompts is not None else {}
        # The turn-1/turn-2 split is positional (everything but the last
        # message is context; the last one elicits the answer), so the last
        # message must BE the eliciting one. True for every family today;
        # checked because a family that appended a message after its cue would
        # otherwise silently plan against a truncated context.
        if prompts is not None and ANSWER_TEMPLATE_NAME in named:
            if named[ANSWER_TEMPLATE_NAME] != prompts.messages[-1]["content"]:
                raise ValueError(
                    f"plan-then-answer rollouts split context from cue positionally, but "
                    f"{type(inner).__name__}'s {ANSWER_TEMPLATE_NAME} message is not the last "
                    "of its solo context — reorder the family's answer-generation config"
                )
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
        # The answer turn's own cap. Without it the answer inherits the
        # sampler ceiling, which run_rlvr must raise to fit the PLAN turn —
        # min()-clamping is how the plan silently shrank to 1000 before.
        self.answer_max_tokens = int(answer_max_tokens) if answer_max_tokens is not None else None
        self.last_rollout_records: list[dict[str, Any]] = []
        self.last_rollout_info: dict[str, int | float] = {}

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        return self.inner.tasks(n, split)

    def _plan_cue(self, task: Task) -> str:
        question = task.meta.get("question")
        if question is None:
            raise ValueError(
                f"plan-then-answer rollout: task from {type(self.inner).__name__} has no "
                'meta["question"] to bind the plan cue\'s <PROBLEM> placeholder'
            )
        if PROBLEM_PLACEHOLDER in self.plan_template and not str(question).strip():
            # Families whose row content is a context message of its own (MB)
            # leave meta["question"] empty; a cue that still asks for <PROBLEM>
            # would render an empty problem block and the plan turn would be
            # about nothing — the exact failure 2e6ba0a fixed for math.
            raise ValueError(
                f"plan cue carries {PROBLEM_PLACEHOLDER} but "
                f'{type(self.inner).__name__} tasks have an empty meta["question"] — '
                "drop the placeholder (the row content reaches the plan turn as a "
                "context message) or give the family a question to bind"
            )
        # Same strictness as GenerationPrompts.render: the only placeholder a
        # plan cue may carry is the row-content one, single-pass substituted.
        extra = sorted(set(_PLACEHOLDER.findall(self.plan_template)) - {"PROBLEM"})
        if extra:
            raise ValueError(f"unbound placeholder(s) {extra} in plan template")
        return self.plan_template.replace(PROBLEM_PLACEHOLDER, str(question)).strip()

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        # Clear externally observed state before rendering or generation. The
        # counters are then advanced only after each stage satisfies the
        # Policy.predict result-shape contract.
        self.last_rollout_records = []
        self.last_rollout_info = {
            "tasks_requested": len(tasks),
            "plan_samples_attempted": 0,
            "plan_samples_kept": 0,
            "plan_samples_dropped_fidelity": 0,
            "answer_requests": 0,
            "answer_samples_attempted": 0,
            "answer_samples_kept": 0,
            "answer_samples_dropped_fidelity": 0,
            "samples_dropped_fidelity": 0,
        }
        # Turn 1: the task's CONTEXT (everything before the eliciting message)
        # + the plan cue. For math/codecontests that context is just the system
        # card and the problem rides in the cue's <PROBLEM>; for MB it is the
        # card plus the trajectory message, which is how the row content
        # reaches the plan turn without being rendered twice — a second copy of
        # a 50k-token trajectory would not fit the context at all.
        plan_convos: list[list[dict[str, str]]] = []
        for t in tasks:
            head = [dict(m) for m in t.messages[:-1]]
            plan_convos.append(head + [{"role": "user", "content": self._plan_cue(t)}])
        plan_results = policy.predict(
            plan_convos, n=group_size, limits=SlotLimits(max_total_tokens=self.plan_max_tokens)
        )
        _validate_predict_results(
            plan_results,
            requests=len(tasks),
            samples_per_request=group_size,
            stage="plan",
        )

        # Turn 2: each surviving plan continues into the eliciting message,
        # rendered by the inner env exactly as its single-turn arm would. The
        # context is already in the conversation from turn 1, so only the cue
        # is appended.
        pending: list[tuple[int, Any, list[dict[str, str]]]] = []
        plan_samples_attempted = sum(len(samples) for samples in plan_results)
        plan_samples_dropped = 0
        for ti, samples in enumerate(plan_results):
            body = [dict(tasks[ti].messages[-1])]
            for s in samples:
                if not s.fidelity_ok():
                    plan_samples_dropped += 1
                    continue
                convo = plan_convos[ti] + [{"role": "assistant", "content": s.text.strip()}] + body
                pending.append((ti, s, convo))
        self.last_rollout_info.update(
            {
                "plan_samples_attempted": plan_samples_attempted,
                "plan_samples_kept": len(pending),
                "plan_samples_dropped_fidelity": plan_samples_dropped,
                "answer_requests": len(pending),
                "samples_dropped_fidelity": plan_samples_dropped,
            }
        )
        answer_limits = (
            SlotLimits(max_total_tokens=self.answer_max_tokens)
            if self.answer_max_tokens is not None
            else None
        )
        answer_results = (
            policy.predict([c for _, _, c in pending], n=1, limits=answer_limits) if pending else []
        )
        _validate_predict_results(
            answer_results,
            requests=len(pending),
            samples_per_request=1,
            stage="answer",
        )
        answer_samples_attempted = sum(len(samples) for samples in answer_results)
        aligned_answers = []
        for pending_item, samples in zip(pending, answer_results):
            answer = samples[0]
            aligned_answers.append((*pending_item, answer, answer.fidelity_ok()))
        answer_samples_kept = sum(int(fidelity_ok) for *_, fidelity_ok in aligned_answers)
        answer_samples_dropped = answer_samples_attempted - answer_samples_kept
        self.last_rollout_info.update(
            {
                "answer_samples_attempted": answer_samples_attempted,
                "answer_samples_kept": answer_samples_kept,
                "answer_samples_dropped_fidelity": answer_samples_dropped,
                "samples_dropped_fidelity": plan_samples_dropped + answer_samples_dropped,
            }
        )

        groups: list[list[Trajectory]] = [[] for _ in tasks]
        records: list[dict[str, Any]] = []
        for ti, plan_s, convo, a, fidelity_ok in aligned_answers:
            if not fidelity_ok:
                continue
            # Scored via the inner env's SAMPLE-level seam with the ANSWER
            # sample, so a logprob-reading grader (MB's confidence tiebreaker)
            # sees the same Sample here as in the single-turn arm. The plan
            # turn stays unrewarded — only `a` is ever scored.
            reward, info = self.inner.reward_sample(tasks[ti], a)
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
                    # PlannedEnv is only a rollout-shape wrapper. Metadata
                    # disclosure remains owned by the source environment so
                    # verifier-private fields are redacted before any
                    # transcript, Docent, or W&B consumer can observe them.
                    "meta": self.inner.export_meta(tasks[ti]),
                    "messages": convo,
                    "completion": a.text,
                    "stop_reason": a.stop_reason,
                    "reward": reward,
                    "info": info,
                }
            )
        self.last_rollout_records = records
        return groups
