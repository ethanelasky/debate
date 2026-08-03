"""DebateEnv: topology-driven debate rollouts as an Env.

One rollout path (train + eval). Reward is judge-only (DESIGN-debate-env.md
§5); gt correctness from solution slots lands in info as metrics.

Trajectory granularity: one Trajectory per (debate, trained seat); datums are
that seat's slot datums in flat order; reward = seat ladder/score value plus
shaping deltas. NOTE: per_slot shaping deltas are folded into the trajectory
reward sum for now — grpo_pack computes one advantage per trajectory, so
slot-targeted penalties are totalized rather than per-datum. Upgrade path:
per-datum advantage offsets in grpo_pack.
"""

from __future__ import annotations

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
from infra.envs.debate.topology import Kind, Topology, Visibility
from infra.envs.task_prompts import PROBLEM_PLACEHOLDER, TASK_SUPPLIED_TEMPLATES
from infra.models.base import Model

DISPLAY_NAMES = ("Debater_A", "Debater_B")


def _splice(template: str, subs: dict[str, str]) -> str:
    """Replace <NAME> with task-supplied template text. Unlike render(), the
    result is still a template: what gets spliced in carries its own
    placeholders, to be bound in the normal render pass."""
    for name, text in subs.items():
        template = template.replace(f"<{name}>", text)
    return template


@dataclass
class DebateEnvConfig:
    topology: Topology
    prompt_file: str
    prompt_entry: str
    trained_speakers: list[str]                 # subset of debater speakers; [] = pure eval
    frozen_models: dict[str, Model]             # speaker -> model (must cover judge + untrained debaters)
    # Per-trained-seat overrides applied on top of the rollout policy (one
    # shared adapter, per-seat sampling params / chat-template kwargs, e.g.
    # enable_thinking differing between proposer and critic).
    trained_sampling: dict[str, Any] = field(default_factory=dict)      # speaker -> SamplingParams
    trained_chat_kwargs: dict[str, dict] = field(default_factory=dict)  # speaker -> template kwargs
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    fresh_positions: bool = True
    flip: bool = False                          # assigned mode only: mirrored second arm
    # Opening solution slot generated under the task source's own messages
    # (byte-identical to the RLVR arm) instead of the debater system card.
    first_speech_non_debate_aware: bool = False
    verdict_retries: int = 4


class DebateEnv(Env):
    """task_source supplies problems: any object with tasks(n, split) whose
    Task.meta carries {"question"} plus whatever the family's grade() reads
    (any TaskFamily.source() qualifies). `family` (a TaskFamily, duck-typed
    to keep this module import-light) owns everything task-specific:
    extractor() binds solution slots to positions, grade() scores them,
    format_flags() feeds shaping terms."""

    def __init__(self, config: DebateEnvConfig, task_source, family, relaxed_extraction: bool = True):
        self.config = config
        self.task_source = task_source
        self.family = family
        self.solution_extractor = family.extractor(relaxed_extraction)

        self.topology = config.topology
        speakers = self.topology.speakers
        self.judge_speaker = self._find_judge_speaker()
        self.debaters = [s for s in speakers if s != self.judge_speaker]
        if len(self.debaters) > 2:
            raise ValueError("at most two debater speakers supported")
        unknown = set(config.trained_speakers) - set(self.debaters)
        if unknown:
            raise ValueError(f"trained_speakers not in topology: {sorted(unknown)}")
        missing = set(speakers) - set(config.trained_speakers) - set(config.frozen_models)
        if missing:
            raise ValueError(f"speakers without models: {sorted(missing)}")
        if config.flip and config.fresh_positions:
            raise ValueError("flip requires assigned positions (fresh mode has no sides)")
        if config.first_speech_non_debate_aware:
            if not config.fresh_positions:
                raise ValueError(
                    "first_speech_non_debate_aware requires fresh_positions: an assigned "
                    "position to defend is inherently debate-aware"
                )
            first = self.topology.compile()[0]
            if first.slot.kind != Kind.SOLUTION or first.speaker == self.judge_speaker:
                raise ValueError(
                    "first_speech_non_debate_aware requires the first topology slot to be a "
                    f"debater solution slot (got {first.speaker}/{first.slot.name} "
                    f"kind={first.slot.kind.value})"
                )
            if first.slot.visibility != Visibility.PUBLIC:
                raise ValueError(
                    "first_speech_non_debate_aware requires the opening solution slot to be "
                    f"public (got {first.slot.visibility.value}): the solo speech must enter "
                    "the transcript as a public record (DESIGN-pc-format.md row 10)"
                )

        self.lib: PromptLibrary = load_prompt_library(config.prompt_file, config.prompt_entry)
        self._inject_task_prompt_templates()
        self.prompts = RenderedPrompts(self.lib)
        validate_prompts(self.lib, self.topology, fresh_positions=config.fresh_positions)
        self.shaping = build_shaping(config.scoring.shaping)
        self.display: dict[str, str] = {s: DISPLAY_NAMES[i] for i, s in enumerate(self.debaters)}

    def _inject_task_prompt_templates(self) -> None:
        """Splice the task's answer-generation prompt into any slot that asks
        for it by <ANSWER_GEN_USER>, so the debate proposal IS the RLVR user
        message rather than a re-typed copy that can drift from it.

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
                "answer-generation config (infra/envs/tasks/prompt_configs/<family>.yaml)."
            )
        if not requested:
            return

        subs = {k: available[k] for k in requested}
        self.lib.system = {s: _splice(t, subs) for s, t in self.lib.system.items()}
        self.lib.slots = {
            name: (
                {sp: _splice(t, subs) for sp, t in entry.items()}
                if isinstance(entry, dict)
                else _splice(entry, subs)
            )
            for name, entry in self.lib.slots.items()
        }

    def _all_templates(self):
        for tmpl in self.lib.system.values():
            yield tmpl
        for entry in self.lib.slots.values():
            yield from (entry.values() if isinstance(entry, dict) else [entry])

    def _find_judge_speaker(self) -> str:
        d = self.topology.decision_slot
        if d is None:
            raise ValueError("debate topology needs a decision slot (judge-only rewards)")
        return d.speaker

    # ------------------------------------------------------------------ env

    def tasks(self, n: int, split: str = "train") -> list[Task]:
        return self.task_source.tasks(n, split)

    def rollout(self, tasks: list[Task], policy: Policy, group_size: int) -> list[list[Trajectory]]:
        from dataclasses import replace as _replace

        cfg = self.config
        seats: dict[str, SeatRunner] = {}
        for speaker in self.topology.speakers:
            if speaker in cfg.trained_speakers:
                seat_policy = policy
                if speaker in cfg.trained_sampling or speaker in cfg.trained_chat_kwargs:
                    # same backend/adapter. The seat override owns the token
                    # ceiling; the INCOMING policy owns the sampling mode
                    # (train temp vs greedy eval) — composing them keeps
                    # policy.greedy() greedy on override seats.
                    override = cfg.trained_sampling.get(speaker)
                    params = (
                        _replace(policy.params, max_tokens=override.max_tokens)
                        if override is not None
                        else policy.params
                    )
                    seat_policy = Policy(
                        policy.backend,
                        params,
                        cfg.trained_chat_kwargs.get(speaker, policy.chat_template_kwargs),
                    )
                seats[speaker] = PolicySeat(seat_policy)
            else:
                seats[speaker] = FrozenSeat(cfg.frozen_models[speaker])

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
            self.topology,
            seats,
            self.prompts,
            verdict_parser=lambda text: parse_verdict(
                text, cfg.judge.schema_name, list(self.display.values())
            ),
            verdict_retries=cfg.judge.retries,
            solution_extractor=self.solution_extractor,
            fresh_positions=cfg.fresh_positions,
            decision_json_schema=decision_json_schema,
        )
        round_.run(states)

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
            trajs = self._trajectories(st)
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
        question = task.meta.get("question") or (task.messages[-1]["content"] if task.messages else "")
        bindings: dict[str, dict[str, str]] = {}
        a, b = (self.debaters + [None, None])[:2]
        names = {s: self.display[s] for s in self.debaters}
        positions: dict[str, str] = {}
        if not cfg.fresh_positions:
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
            }
        bindings[self.judge_speaker] = {
            "NAME": names.get(a, ""),
            "OPPONENT_NAME": names.get(b, "") if b else "",
            "TOPIC": question,
            "POSITION": positions.get(a, ""),
            "OPPONENT_POSITION": positions.get(b, "") if b else "",
        }
        state = DebateState(bindings=bindings)
        if cfg.first_speech_non_debate_aware:
            # Fail loud on malformed shapes rather than let render_context emit
            # a non-alternating context or silently fall back to the library
            # cue in the proposer's later history (row 10). Contract shape:
            # optional leading system, then user/assistant alternation ending
            # on the user message that serves as the solo cue.
            roles = [m.get("role") for m in task.messages]
            body = roles[1:] if roles[:1] == ["system"] else roles
            if not body or body[-1] != "user" or any(
                r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(body)
            ):
                raise ValueError(
                    "first_speech_non_debate_aware: task messages must be an optional "
                    "system message then user/assistant alternation ending on the user "
                    f"message that serves as the solo cue (got roles {roles})"
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
        if seat is not None:
            try:
                decode_fn = seat.decode_tokens
            except NotImplementedError:
                decode_fn = None
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

    def _trajectories(self, st: DebateState) -> Optional[list[Trajectory]]:
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
                correct = self.family.grade(st.meta.get("task", {}), solutions[speaker])
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
