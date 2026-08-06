"""DebateEnv: protocol-driven debate rollouts as an Env.

One rollout path (train + eval). Reward is judge-only (DESIGN-debate-env.md
§5); gt correctness from solution slots lands in info as metrics.

Trajectory granularity: one Trajectory per (debate, trained seat); datums are
that seat's slot datums in flat order. Rewards are per-datum
(Trajectory.datum_rewards): each datum carries the shared outcome plus
seat-level shaping deltas, and slot-targeted deltas land only on their own
slot's datum; Trajectory.reward keeps the datum mean for logging.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from infra.envs.base import Env, Policy, Task, Trajectory
from infra.envs.debate.judge import JudgeConfig, Verdict, parse_verdict, verdict_from_slot
from infra.envs.debate.prompts import (
    PLACEHOLDER,
    PromptLibrary,
    RenderedPrompts,
    load_prompt_library,
    validate_prompts,
)
from infra.envs.debate.rewards import (
    RoundTokenReport,
    ScoringConfig,
    SlotTokenCounts,
    build_shaping,
    score,
    validate_scoring,
)
from infra.envs.debate.round import (
    DebateRound,
    DebateState,
    FrozenSeat,
    PolicySeat,
    SeatRunner,
    SlotRecord,
)
from infra.envs.debate.protocol import Kind, Protocol, Visibility
from infra.envs.task_prompts import PROBLEM_PLACEHOLDER, TASK_SUPPLIED_TEMPLATES
from infra.models.base import Model, ModelSettings

DISPLAY_NAMES = ("Debater_A", "Debater_B")

# The five binding keys _build_state owns; task-supplied bindings may not
# collide with them.
CORE_BINDING_KEYS = frozenset({"NAME", "TOPIC", "POSITION", "OPPONENT_NAME", "OPPONENT_POSITION"})
_BINDING_KEY = re.compile(r"[A-Z_][A-Z0-9_]*")


def task_bindings(meta: dict[str, Any]) -> dict[str, str]:
    """Per-task extra bindings (task.meta['bindings']), validated eagerly:
    keys are uppercase placeholder names, values are str, and none collide
    with the core keys. Every violation is listed in one error. Only an
    absent key (or explicit None) means "no bindings" — any other
    non-mapping value, falsy included ([], "", 0), is malformed metadata."""
    extra = meta.get("bindings")
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise ValueError(f"task meta 'bindings' must be a dict, got {type(extra).__name__}")
    errors: list[str] = []
    for k, v in extra.items():
        if not isinstance(k, str) or _BINDING_KEY.fullmatch(k) is None:
            errors.append(f"binding key {k!r} must match [A-Z_][A-Z0-9_]*")
        elif k in CORE_BINDING_KEYS:
            errors.append(f"binding key {k!r} collides with a core binding key")
        if not isinstance(v, str):
            errors.append(f"binding {k!r} value must be str, got {type(v).__name__}")
    if errors:
        raise ValueError("invalid task bindings:\n  " + "\n  ".join(errors))
    return dict(extra)


def _splice(template: str, subs: dict[str, str]) -> str:
    """Replace <NAME> with task-supplied template text. Unlike render(), the
    result is still a template: what gets spliced in carries its own
    placeholders, to be bound in the normal render pass."""
    for name, text in subs.items():
        template = template.replace(f"<{name}>", text)
    return template


@dataclass
class DebateEnvConfig:
    protocol: Protocol
    prompt_file: str
    prompt_entry: str
    trained_speakers: list[str]                 # subset of debater speakers; [] = pure eval
    frozen_models: dict[str, Model]             # speaker -> model (must cover judge + untrained debaters)
    # Per-trained-seat overrides applied on top of the rollout policy (one
    # shared adapter, per-seat sampling params / chat-template kwargs, e.g.
    # enable_thinking differing between proposer and critic).
    trained_sampling: dict[str, Any] = field(default_factory=dict)      # speaker -> SamplingParams
    trained_chat_kwargs: dict[str, dict] = field(default_factory=dict)  # speaker -> template kwargs
    # Per-frozen-seat resolved sampling profiles (speaker -> SamplingProfile,
    # train binding), forwarded per predict call by FrozenSeat; an absent
    # speaker keeps the wrapper/server defaults.
    frozen_sampling: dict[str, Any] = field(default_factory=dict)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    # The judge seat's ModelSettings, when the builder has them: lets the env
    # check the scoring/judge sampling contract (validate_scoring) at
    # construction. None (configs assembled without settings) skips the check.
    judge_model_settings: Optional[ModelSettings] = None
    fresh_positions: bool = True
    flip: bool = False                          # assigned mode only: mirrored second arm
    # Opening solution slot generated under the task source's own messages
    # (byte-identical to the RLVR arm) instead of the debater system card.
    first_speech_non_debate_aware: bool = False
    # Verdict retry count; None defers to judge.retries. Both spellings exist
    # in the wild (runners set judge_config.retries, some tests set this
    # field), so an explicit value here wins and None keeps judge.retries
    # authoritative.
    verdict_retries: Optional[int] = None
    # Post-hoc TRANSCRIPT-VISIBLE cap (cl100k tokens) on debater speech slots,
    # applied after generation, before the text enters any context. Judge
    # slots and decision/solution slots are never truncated. None = off.
    speech_token_limit: Optional[int] = None
    # CHOICE mode (round.PositionBinder): positions start UNBOUND (like fresh
    # mode) and the binder maps a solution slot's extraction onto every seat's
    # POSITION/OPPONENT_POSITION mid-round (e.g. MB blind side choice). When
    # set, gold/distractor are ignored at state build and the default
    # fresh-positions binding is replaced.
    position_binder: Optional[Any] = None
    # Feedback retries for unparseable solution slots (mirrors verdict
    # retries); active only when a feedback builder (failed_text, attempt) ->
    # str is supplied — its wording must match the solution prompt's format.
    solution_retries: int = 0
    solution_retry_feedback: Optional[Any] = None


class DebateEnv(Env):
    """task_source supplies problems: any object with tasks(n, split) whose
    Task.meta carries {"question"} plus whatever the family's grade() reads
    (any TaskFamily.source() qualifies). `family` (a TaskFamily, duck-typed
    to keep this module import-light) owns everything task-specific:
    extractor() binds solution slots to positions, grade_batch() scores them
    (owning the execution shape — pool, batched call, whatever the verifier
    wants), format_flags() feeds shaping terms."""

    def __init__(self, config: DebateEnvConfig, task_source, family, relaxed_extraction: bool = True):
        self.config = config
        self.task_source = task_source
        self.family = family
        self.solution_extractor = family.extractor(relaxed_extraction)

        self.protocol = config.protocol
        speakers = self.protocol.speakers
        self.judge_speaker = self._find_judge_speaker()
        self.debaters = [s for s in speakers if s != self.judge_speaker]
        if len(self.debaters) > 2:
            raise ValueError("at most two debater speakers supported")
        unknown = set(config.trained_speakers) - set(self.debaters)
        if unknown:
            raise ValueError(f"trained_speakers not in protocol: {sorted(unknown)}")
        missing = set(speakers) - set(config.trained_speakers) - set(config.frozen_models)
        if missing:
            raise ValueError(f"speakers without models: {sorted(missing)}")
        # The logit confidence channel is only P(sampled continuation) under an
        # untouched judge distribution — checked here, not mid-run, whenever
        # the builder supplied the judge's settings. json-source scoring
        # passes through unconditionally (validate_scoring returns early).
        if config.judge_model_settings is not None:
            validate_scoring(config.scoring, config.judge_model_settings)
        if config.flip and (config.fresh_positions or config.position_binder is not None):
            raise ValueError("flip requires assigned positions (fresh/choice modes have no fixed sides)")
        if config.first_speech_non_debate_aware:
            # choice mode (position_binder) also qualifies: the side is the
            # speaker's own blind choice, so nothing assigned is debate-aware.
            if not (config.fresh_positions or config.position_binder is not None):
                raise ValueError(
                    "first_speech_non_debate_aware requires fresh_positions or a "
                    "position_binder (choice mode): an assigned position to defend is "
                    "inherently debate-aware"
                )
            first = self.protocol.compile()[0]
            if first.slot.kind != Kind.SOLUTION or first.speaker == self.judge_speaker:
                raise ValueError(
                    "first_speech_non_debate_aware requires the first protocol slot to be a "
                    f"debater solution slot (got {first.speaker}/{first.slot.name} "
                    f"kind={first.slot.kind.value})"
                )
            if first.slot.visibility != Visibility.PUBLIC:
                raise ValueError(
                    "first_speech_non_debate_aware requires the opening solution slot to be "
                    f"public (got {first.slot.visibility.value}): the solo speech must enter "
                    "the transcript as a public record (DESIGN-pc-format.md row 10)"
                )

        if not config.fresh_positions and config.position_binder is None:
            # Probe the contract up front: assigned mode reads meta["gold"] /
            # meta["distractor"] per debate, and a missing key would otherwise
            # die as a bare KeyError mid-rollout after backends are up.
            # Choice mode (position_binder) binds sides from the blind choice
            # instead, so it is exempt.
            probe = task_source.tasks(1)
            if probe and not {"gold", "distractor"} <= set(probe[0].meta):
                raise ValueError(
                    "assigned-position debate (fresh_positions=False) requires "
                    'Task.meta["gold"] and Task.meta["distractor"], but task source '
                    f"{type(task_source).__name__} does not provide them"
                )

        # Grammar forcing (decision_json_schema) is only consumed by LocalModel
        # judges; API/tinker judges silently drop json_schema, so verdict
        # quality then rests entirely on parse retries.
        if config.judge.schema_name == "competitive":
            judge_model = config.frozen_models.get(self.judge_speaker)
            if judge_model is not None:
                try:
                    from infra.models.local_model import LocalModel
                except Exception:  # noqa: BLE001 - optional dep; treat as non-local
                    LocalModel = None
                if LocalModel is None or not isinstance(judge_model, LocalModel):
                    warnings.warn(
                        f"judge model {type(judge_model).__name__} does not consume "
                        "json_schema: grammar-forced verdicts are unavailable for this "
                        "judge type and verdict quality depends on judge.retries",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        self.lib: PromptLibrary = load_prompt_library(
            config.prompt_file, config.prompt_entry, self.protocol
        )
        self._inject_task_prompt_templates()
        self.prompts = RenderedPrompts(self.lib)
        # Choice mode (position_binder) defers position binding like fresh
        # mode but binds BOTH deferred names for every speaker from the single
        # solution slot; validate_prompts checks the matching bindability rule.
        validate_prompts(
            self.lib,
            self.protocol,
            fresh_positions=config.fresh_positions,
            choice_positions=config.position_binder is not None,
        )
        self.shaping = build_shaping(config.scoring.shaping)
        # build_shaping accepts any slots/flag; a typo would silently zero the
        # term's bonus, so the names are checked against the protocol and the
        # family here.
        slot_names = {cs.slot.name for cs in self.protocol.compile()}
        flag_names = set(family.format_flags(""))
        for term in self.shaping:
            term_slots = getattr(term, "slots", None)
            if term_slots is not None:
                unknown_slots = sorted(set(term_slots) - slot_names)
                if unknown_slots:
                    raise ValueError(
                        f"shaping term {type(term).__name__} targets unknown slot(s) "
                        f"{unknown_slots}; protocol slots: {sorted(slot_names)}"
                    )
            term_flag = getattr(term, "flag", None)
            if term_flag is not None and term_flag not in flag_names:
                raise ValueError(
                    f"shaping term {type(term).__name__} gates on unknown flag {term_flag!r}; "
                    f"{type(family).__name__} format flags: {sorted(flag_names) or '(none)'}"
                )
        self.display: dict[str, str] = {s: DISPLAY_NAMES[i] for i, s in enumerate(self.debaters)}

    def _inject_task_prompt_templates(self) -> None:
        """Splice the task's answer-generation messages into any template that
        asks for one by its splice name (<ANSWER_GEN_USER>: math/CC proposal
        slot, MB blind_assessment cue; <TRAJECTORY_USER>: MB shared pre_debate
        trajectory message), so the debate stage IS the RLVR message rather
        than a re-typed copy that can drift from it.

        This is a template-level splice, not a var substitution: the spliced
        text still carries placeholders (the task config's <PROBLEM> is rebound
        to this layer's <TOPIC>), and render() makes a single pass that never
        re-scans substituted text, so the wording has to be part of the
        template before rendering starts.

        Both shipped packs splice: one answer prompt per family, rendered
        identically by the RLVR arm and by the debate proposal slot. Math kept
        a separate debate wording until that asymmetry was deliberately
        dropped."""
        supplied = getattr(self.task_source, "prompts", None)
        available = supplied.supplied_templates() if supplied is not None else {}
        available = {
            k: v.replace(PROBLEM_PLACEHOLDER, "<TOPIC>") for k, v in available.items()
        }

        requested = {
            m.group(1)
            for tmpl in self._all_templates()
            for m in PLACEHOLDER.finditer(tmpl)
        } & set(TASK_SUPPLIED_TEMPLATES)
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(
                f"prompt entry {self.config.prompt_entry!r} references {missing}, which the task "
                f"source must supply, but {type(self.task_source).__name__} supplied "
                f"{sorted(available) or 'nothing'}. These come from the task family's "
                "answer-generation config (infra/prompts/tasks/<family>.yaml)."
            )
        if not requested:
            return

        subs = {k: available[k] for k in requested}
        self.lib.system = {s: _splice(t, subs) for s, t in self.lib.system.items()}
        self.lib.preamble = {
            s: [_splice(t, subs) for t in msgs] for s, msgs in self.lib.preamble.items()
        }
        self.lib.shared_pre_debate = _splice(self.lib.shared_pre_debate, subs)
        self.lib.slots = {name: _splice(t, subs) for name, t in self.lib.slots.items()}

    def _all_templates(self):
        for tmpl in self.lib.system.values():
            yield tmpl
        for msgs in self.lib.preamble.values():
            yield from msgs
        yield from self.lib.slots.values()

    def _find_judge_speaker(self) -> str:
        d = self.protocol.decision_slot
        if d is None:
            raise ValueError("debate protocol needs a decision slot (judge-only rewards)")
        return d.speaker

    # ------------------------------------------------------------------ env

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        return self.task_source.tasks(n, split)

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        from dataclasses import replace as _replace

        cfg = self.config
        seats: dict[str, SeatRunner] = {}
        for speaker in self.protocol.speakers:
            if speaker in cfg.trained_speakers:
                seat_policy = policy
                if speaker in cfg.trained_sampling or speaker in cfg.trained_chat_kwargs:
                    # same backend/adapter. The seat override owns the token
                    # ceiling and its train-time temperature/top_p; the
                    # INCOMING policy owns the sampling MODE — greedy eval
                    # (temperature 0) keeps its mode on override seats.
                    override = cfg.trained_sampling.get(speaker)
                    params = policy.params
                    if override is not None:
                        updates: dict[str, Any] = {"max_tokens": override.max_tokens}
                        if params.temperature != 0.0:
                            updates["temperature"] = override.temperature
                            updates["top_p"] = override.top_p
                        params = _replace(params, **updates)
                    seat_policy = Policy(
                        policy.backend,
                        params,
                        cfg.trained_chat_kwargs.get(speaker, policy.chat_template_kwargs),
                    )
                seats[speaker] = PolicySeat(seat_policy)
            else:
                seats[speaker] = FrozenSeat(
                    cfg.frozen_models[speaker], sampling=cfg.frozen_sampling.get(speaker)
                )

        arms = [False, True] if cfg.flip else [False]
        # base group id = (task index, arm): flip arms are separate GRPO groups.
        # Final groups additionally split BY TRAINED SEAT (below): proposer and
        # critic rewards are anti-correlated, so normalizing them together
        # would manufacture advantage out of the seat identity.
        states: list[DebateState] = []
        state_group: list[int] = []
        group_count = 0
        for ti, task in enumerate(tasks):
            for arm in arms:
                gid = group_count
                group_count += 1
                for _ in range(group_size):
                    states.append(self._build_state(task, flipped=arm))
                    state_group.append(gid)

        # Grammar-force the verdict on servers that support it (vLLM): a
        # free-text judge mid-thought at the deliberation cap keeps reasoning
        # into the verdict slot and never emits JSON (smoke: 15/16 debates
        # lost to verdict_unparseable). Confidence bounds stay parser-side.
        decision_json_schema = None
        if cfg.judge.schema_name == "competitive":
            decision_json_schema = {
                "type": "object",
                "properties": {
                    "winner": {"type": "string", "enum": [*self.display.values(), "Tie"]},
                    "confidence": {"type": "number"},
                },
                "required": ["winner", "confidence"],
                "additionalProperties": False,
            }
        round_ = DebateRound(
            self.protocol,
            seats,
            self.prompts,
            verdict_parser=lambda text: parse_verdict(
                text, cfg.judge.schema_name, list(self.display.values())
            ),
            verdict_retries=(
                cfg.verdict_retries if cfg.verdict_retries is not None else cfg.judge.retries
            ),
            judge_schema=cfg.judge.schema_name,
            solution_extractor=self.solution_extractor,
            fresh_positions=cfg.fresh_positions,
            decision_json_schema=decision_json_schema,
            speech_token_limit=cfg.speech_token_limit,
            position_binder=cfg.position_binder,
            solution_retries=cfg.solution_retries,
            solution_retry_feedback=cfg.solution_retry_feedback,
        )
        round_.run(states)

        # Grade all solutions up front, deduped and in parallel: codecontests
        # grading is a subprocess with up to a 90s timeout per call, so serial
        # per-state grading of a 32-debate batch can burn tens of minutes for
        # a metrics-only scalar. _trajectories only looks the results up.
        grades, grade_errors = self._grade_solutions(states)
        self._attach_labels(states, grades)

        # one GRPO group per (task, arm, trained seat)
        by_seat_group: dict[tuple[int, str], list[Trajectory]] = {}
        fail_reasons: dict[str, int] = {}
        n_failed = 0
        n_unscoreable = 0
        for st, gid in zip(states, state_group):
            if st.failed is not None:
                n_failed += 1
                fail_reasons[st.failed.split(":")[0]] = fail_reasons.get(st.failed.split(":")[0], 0) + 1
                continue
            trajs = self._trajectories(st, grades)
            if trajs is None:
                n_unscoreable += 1
                continue
            for traj in trajs:
                by_seat_group.setdefault((gid, traj.info.get("seat", "")), []).append(traj)
        groups: list[list[Trajectory]] = [g for _, g in sorted(by_seat_group.items())] or [[]]
        # Always recorded, even when everything dropped (else failure is mute).
        self.last_rollout_info = {
            "debates": len(states),
            "debates_failed": n_failed,
            "debates_unscoreable": n_unscoreable,
            "fail_reasons": fail_reasons,
            "grade_errors": grade_errors,
        }
        self.last_states = states  # retained for transcript export (docent)
        for g in groups:
            if g:
                g[0].info["debates_failed"] = float(n_failed)
                g[0].info["debates_unscoreable"] = float(n_unscoreable)
                break
        return groups

    # ------------------------------------------------------------- internals

    def _build_state(self, task: Task, flipped: bool) -> DebateState:
        cfg = self.config
        question = task.meta.get("question")
        if question is None:
            # An explicit "" is legal (MB binds its content via task bindings
            # and never renders <TOPIC>); an ABSENT key is a broken contract.
            raise ValueError(
                f"task source {type(self.task_source).__name__} produced a task "
                'without meta["question"]; the TaskFamily contract '
                "(infra/envs/tasks/base.py) requires every Task.meta to carry "
                '{"question": str} — it is what DebateEnv binds as the debate TOPIC'
            )
        extra = task_bindings(task.meta)
        bindings: dict[str, dict[str, str]] = {}
        a, b = (self.debaters + [None, None])[:2]
        names = {s: self.display[s] for s in self.debaters}
        positions: dict[str, str] = {}
        # Choice mode: positions stay unbound at build (the binder fills them
        # once the solution slot's extraction lands), same as fresh mode.
        if not cfg.fresh_positions and cfg.position_binder is None:
            gold, distractor = str(task.meta["gold"]), str(task.meta["distractor"])
            first, second = (distractor, gold) if flipped else (gold, distractor)
            positions = {a: first} if b is None else {a: first, b: second}

        for s in self.debaters:
            other = b if s == a else a
            bindings[s] = {
                "NAME": names[s],
                "TOPIC": question,
                "POSITION": positions.get(s, ""),
                "OPPONENT_NAME": names.get(other, "") if other else "",
                "OPPONENT_POSITION": positions.get(other, "") if other else "",
                **extra,
            }
        bindings[self.judge_speaker] = {
            "NAME": names.get(a, ""),
            "OPPONENT_NAME": names.get(b, "") if b else "",
            "TOPIC": question,
            "POSITION": positions.get(a, ""),
            "OPPONENT_POSITION": positions.get(b, "") if b else "",
            **extra,
        }
        state = DebateState(bindings=bindings)
        if cfg.first_speech_non_debate_aware:
            # Fail loud on malformed shapes rather than let render_context emit
            # a malformed context or silently fall back to the library cue in
            # the proposer's later history (row 10). Contract shape: optional
            # leading system, then user/assistant messages STARTING and ENDING
            # on user, with no consecutive assistant messages. Consecutive
            # USER messages are legal by design: a byte-stable task message
            # (e.g. the MB trajectory) may precede the eliciting cue as its
            # own message so message-boundary prompt caches can reuse it.
            roles = [m.get("role") for m in task.messages]
            body = roles[1:] if roles[:1] == ["system"] else roles
            if (
                not body
                or body[0] != "user"
                or body[-1] != "user"
                or any(r not in ("user", "assistant") for r in body)
                or any(r1 == r2 == "assistant" for r1, r2 in zip(body, body[1:]))
            ):
                raise ValueError(
                    "first_speech_non_debate_aware: task messages must be an optional "
                    "system message then user/assistant messages starting and ending "
                    "on a user message, with no consecutive assistant messages "
                    f"(got roles {roles})"
                )
            state.first_slot_messages = list(task.messages)
        # ground truth about which arm produced this debate, carried into
        # transcripts/exports (a misspelled experiment key silently reads as
        # False — the metadata makes that visible after the fact)
        state.meta.update(
            {
                "task": task.meta,
                "flipped": flipped,
                "solo_first_speech": cfg.first_speech_non_debate_aware,
            }
        )
        return state

    def _verdict(self, st: DebateState) -> Optional[Verdict]:
        rec = next(
            (r for r in reversed(st.records) if r.slot.slot.kind == Kind.DECISION), None
        )
        if rec is None:
            return None
        decode_fn = None
        seat = self.config.frozen_models.get(self.judge_speaker)
        # Attribute access never raises: the base Model.decode_tokens raises
        # NotImplementedError only when CALLED. Detect support by override so
        # a tokenizer-less seat yields NO_PIECES, not MISALIGNED.
        if seat is not None and type(seat).decode_tokens is not Model.decode_tokens:
            decode_fn = seat.decode_tokens
        return verdict_from_slot(
            rec.text,
            rec.response if rec.response is not None else rec.sample,
            decode_fn,
            self.config.judge,
            list(self.display.values()),
        )

    def _token_report(self, st: DebateState) -> RoundTokenReport:
        counts: dict[tuple[str, str], SlotTokenCounts] = {}
        for r in st.records:
            if r.sample is None:
                continue
            think = visible = 0
            if r.sample.regions:
                for reg in r.sample.regions:
                    if reg.kind == "think":
                        think += reg.end - reg.start
                    elif reg.kind == "visible":
                        visible += reg.end - reg.start
            else:
                visible = len(r.sample.tokens)
            flags: dict[str, float] = {}
            if r.slot.slot.kind == Kind.SOLUTION:
                flags.update(self.family.format_flags(r.text))
                flags["extracted"] = float(r.extracted is not None)
            counts[(r.slot.speaker, r.slot.slot.name)] = SlotTokenCounts(
                think=think,
                visible=visible,
                total=len(r.sample.tokens),
                cap_total=r.slot.slot.max_total_tokens,
                flags=flags,
            )
        return RoundTokenReport(counts=counts)

    def _grade_key(self, st: DebateState, solution: Any) -> tuple[Any, str]:
        """Dedup key for grading: same task + same solution text = one grade.
        The task component prefers a stable name/question; id() of the shared
        meta dict is the last resort (states in a group share the Task.meta)."""
        task_meta = st.meta.get("task") or {}
        tkey = task_meta.get("name") or task_meta.get("question") or id(task_meta)
        return (tkey, str(solution))

    def _grade_solutions(
        self, states: list[DebateState]
    ) -> tuple[dict[tuple[Any, str], Optional[bool]], int]:
        """Every (task, solution) pair _trajectories will need, deduped, then
        graded through family.grade_batch — the family owns the execution
        shape (thread pool for subprocess verifiers, one batched call for a
        learned one) and the never-raises contract; this side owns the dedup."""
        pending: dict[tuple[Any, str], tuple[dict[str, Any], Any]] = {}
        for st in states:
            if st.failed is not None:
                continue
            solutions = {
                r.slot.speaker: r.extracted
                for r in st.records
                if r.slot.slot.kind == Kind.SOLUTION
            }
            for speaker in self.config.trained_speakers:
                solution = solutions.get(speaker)
                if solution is None:
                    continue
                key = self._grade_key(st, solution)
                if key not in pending:
                    pending[key] = (st.meta.get("task") or {}, solution)
        if not pending:
            return {}, 0
        keys = list(pending)
        results = self.family.grade_batch([pending[k] for k in keys])
        return dict(zip(keys, results)), int(getattr(self.family, "last_grade_errors", 0))

    def _attach_labels(
        self, states: list[DebateState], grades: dict[tuple[Any, str], Optional[bool]]
    ) -> None:
        """Park each solution slot's ground-truth label on the debate it came
        from, so transcript exports carry it.

        Judge accuracy is derived from transcripts (winner x side x label),
        never from aggregated metrics (AGENTS.md), and the label cannot be
        recovered downstream: math's `gt` survives the export by being a
        scalar, but codecontests' test cases are deliberately withheld from
        every export, so re-grading offline would mean rejoining to the source
        dataset and re-running the same 90s-timeout verifier that already ran
        here.

        st.meta["grades"][speaker] semantics, distinct on purpose:
          absent -> never graded (no solution slot, or an untrained seat:
                    _grade_solutions only grades trained_speakers)
          None   -> graded and ungradeable (no ground truth, or grade raised)
          bool   -> the label
        """
        for st in states:
            if st.failed is not None:
                continue
            labels: dict[str, Optional[bool]] = {}
            for r in st.records:
                if r.slot.slot.kind != Kind.SOLUTION or r.extracted is None:
                    continue
                key = self._grade_key(st, r.extracted)
                if key in grades:
                    labels[r.slot.speaker] = grades[key]
            st.meta["grades"] = labels

    def _attach_labels(
        self, states: list[DebateState], grades: dict[tuple[Any, str], Optional[bool]]
    ) -> None:
        """Park each solution slot's ground-truth label on the debate it came
        from, so transcript exports carry it.

        Judge accuracy is derived from transcripts (winner x side x label),
        never from aggregated metrics (AGENTS.md), and the label cannot be
        recovered downstream: math's `gt` survives the export by being a
        scalar, but codecontests' test cases are deliberately withheld from
        every export, so re-grading offline would mean rejoining to the source
        dataset and re-running the same 90s-timeout verifier that already ran
        here.

        st.meta["grades"][speaker] semantics, distinct on purpose:
          absent -> never graded (no solution slot, or an untrained seat:
                    _grade_solutions only grades trained_speakers)
          None   -> graded and ungradeable (no ground truth, or grade raised)
          bool   -> the label
        """
        for st in states:
            if st.failed is not None:
                continue
            labels: dict[str, Optional[bool]] = {}
            for r in st.records:
                if r.slot.slot.kind != Kind.SOLUTION or r.extracted is None:
                    continue
                key = self._grade_key(st, r.extracted)
                if key in grades:
                    labels[r.slot.speaker] = grades[key]
            st.meta["grades"] = labels

    def _trajectories(
        self, st: DebateState, grades: dict[tuple[Any, str], Optional[bool]]
    ) -> Optional[list[Trajectory]]:
        cfg = self.config
        verdict = self._verdict(st)
        if verdict is None:
            return None
        mode = cfg.judge.schema_name
        seat_rewards = score(verdict, mode, cfg.scoring)
        if any(not sr.scoreable for sr in seat_rewards.values()):
            return None

        report = self._token_report(st)
        # shaping terms key on report's speaker names; give them a speaker-keyed
        # view of the seat rewards (score() keys by display name)
        scored_by_speaker = {
            sp: seat_rewards[self.display[sp]] for sp in self.debaters if self.display[sp] in seat_rewards
        }
        deltas = [term.apply(scored_by_speaker, report) for term in self.shaping]

        solutions = {
            r.slot.speaker: r.extracted
            for r in st.records
            if r.slot.slot.kind == Kind.SOLUTION
        }

        trajs = []
        for speaker in cfg.trained_speakers:
            display = self.display[speaker]
            base = seat_rewards[display].value if display in seat_rewards else 0.0
            seat_delta = sum(d.per_seat.get(speaker, 0.0) for d in deltas)
            slot_records = [r for r in st.records if r.slot.speaker == speaker and r.datum is not None]
            datums = [r.datum for r in slot_records]
            if not datums:
                continue
            # Per-datum rewards: shared outcome + seat-level deltas on every
            # datum, slot-targeted deltas only on their own slot's datum.
            datum_rewards = [
                base
                + seat_delta
                + sum(d.per_slot.get((speaker, r.slot.slot.name), 0.0) for d in deltas)
                for r in slot_records
            ]
            shaped = sum(datum_rewards) / len(datum_rewards)
            info: dict[str, Any] = {
                "seat": speaker,
                "reward_base": float(base),
                "reward_shaped": float(shaped),
                "cell": seat_rewards[display].cell if display in seat_rewards else "none",
                "source": seat_rewards[display].source if display in seat_rewards else "none",
                "verdict_ok": float(verdict.ok),
                "flipped": float(st.meta.get("flipped", False)),
            }
            for src in ("json", "logit"):
                conf = verdict.confidence.get(display)
                if conf is not None:
                    v = getattr(conf, src)
                    if v is not None:
                        info[f"judge_conf_{src}"] = float(v)
            if solutions.get(speaker) is not None:
                correct = grades.get(self._grade_key(st, solutions[speaker]))
                if correct is not None:
                    info["solution_correct"] = float(correct)
            trajs.append(
                Trajectory(datums=datums, reward=shaped, info=info, datum_rewards=datum_rewards)
            )
        if not trajs and not cfg.trained_speakers:
            # pure-eval mode: emit a datum-less trajectory carrying metrics
            display = self.display[self.debaters[0]]
            base = seat_rewards.get(display)
            trajs.append(
                Trajectory(
                    datums=[],
                    reward=base.value if base else 0.0,
                    info={"transcript": st.transcript(), "verdict_ok": float(verdict.ok)},
                )
            )
        return trajs
