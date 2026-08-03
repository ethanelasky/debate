"""Debate turn loop: context assembly (rules 2-3), per-speaker rendering
(rule 4), batched generation over a group of simultaneous debates.

Design (see DESIGN-debate-env.md + the round plan):
- ONE immutable slot-record store per debate; every reader's message list is a
  pure function of it. No per-reader transcript copies.
- Rendering orders records by REVEAL time, not flat index: others' turn-t
  public slots enter a reader's view at the t->t+1 boundary — after the
  reader's own turn-t assistant messages — so each speaker's rendered context
  is a strict prefix of its later ones (modulo ephemeral aging out, which is
  the semantics).
- Alternation: every slot has an instruction cue rendered as user content
  immediately before its assistant generation; others' public speeches buffer
  into the same pending user message. By construction: one system message,
  strict user/assistant alternation, always ending on user.
- The judge is not a special case: its stream is the attributed public
  transcript; only its prompt templates differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from infra.backend.base import Datum, Sample, SamplingParams
from infra.envs.base import Policy, SlotLimits, datum_from_sample
from infra.envs.debate.topology import CompiledSlot, Kind, Topology, Visibility
from infra.models.base import Model, ModelInput, ModelResponse

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


def slot_limits(cs: CompiledSlot) -> SlotLimits:
    return SlotLimits(cs.slot.max_think_tokens, cs.slot.max_visible_tokens, cs.slot.max_total_tokens)


class PromptLibrary(Protocol):
    """Per-slot templates; the only place speaker identity/instructions live."""

    def system(self, speaker: str, bindings: dict[str, str]) -> str: ...

    def instruction(self, slot_name: str, speaker: str, bindings: dict[str, str]) -> str: ...

    def attributed(self, author_name: str, slot_name: str, text: str) -> str: ...


@dataclass
class SlotRecord:
    slot: CompiledSlot
    text: str                       # post-think visible text (think stripped at record time)
    thinking: Optional[str] = None  # stored; NEVER rendered into any context
    extracted: Any = None           # solution: parsed answer; decision: verdict dict
    sample: Optional[Sample] = None      # trained seats
    datum: Optional[Datum] = None        # trained seats (advantages zeroed; mask from budget forcing)
    response: Optional[ModelResponse] = None  # frozen seats
    retries: int = 0


@dataclass
class DebateState:
    bindings: dict[str, dict[str, str]]  # speaker -> {NAME, TOPIC, POSITION, OPPONENT_POSITION, ...}
    records: list[SlotRecord] = field(default_factory=list)
    failed: Optional[str] = None         # fail reason; None = live
    meta: dict[str, Any] = field(default_factory=dict)
    # first_speech_non_debate_aware: the task source's own messages, used
    # verbatim as the context of the first compiled slot (no debate framing) so
    # that speech shares the RLVR arm's prompt distribution exactly.
    first_slot_messages: Optional[list[Message]] = None

    def transcript(self) -> list[dict[str, Any]]:
        return [
            {
                "speaker": r.slot.speaker,
                "turn": r.slot.turn,
                "name": r.slot.slot.name,
                "kind": r.slot.slot.kind.value,
                "visibility": r.slot.slot.visibility.value,
                "text": r.text,
                "thinking": r.thinking,
                "extracted": r.extracted,
                "retries": r.retries,
            }
            for r in self.records
        ]


# ------------------------------------------------------------- seat runners


@dataclass
class GenRequest:
    messages: list[Message]
    limits: SlotLimits
    # Decision slots only: a JSON Schema the server should constrain decoding
    # to (vLLM response_format). Free-text judges ramble past their token cap
    # instead of emitting the verdict object; grammar-forcing is the only cap
    # that binds.
    json_schema: Optional[dict] = None


@dataclass
class SlotResult:
    text: str
    thinking: Optional[str] = None
    sample: Optional[Sample] = None
    datum: Optional[Datum] = None
    mask: Optional[list[float]] = None
    response: Optional[ModelResponse] = None
    failed: bool = False
    fail_reason: Optional[str] = None
    retries: int = 0


class SeatRunner(ABC):
    trained: bool = False

    @abstractmethod
    def generate(self, requests: list[GenRequest]) -> list[SlotResult]: ...


def _split_think(raw: str) -> tuple[Optional[str], str]:
    import re

    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        return m.group(1).strip(), raw[m.end() :].strip()
    return None, raw.strip()


class PolicySeat(SeatRunner):
    """Trained seat over envs.base.Policy. Builds the Datum at generation time
    so misalignment fails immediately. Slot limits run through Policy's
    budget-forced sampling; injected </think> tokens arrive masked."""

    trained = True

    def __init__(self, policy: Policy):
        self.policy = policy

    def generate(self, requests: list[GenRequest]) -> list[SlotResult]:
        # One batched predict per distinct limit set (usually one).
        results: list[Optional[SlotResult]] = [None] * len(requests)
        by_limits: dict[SlotLimits, list[int]] = {}
        for i, r in enumerate(requests):
            by_limits.setdefault(r.limits, []).append(i)

        for limits, idxs in by_limits.items():
            convos = [requests[i].messages for i in idxs]
            outs = self.policy.predict(convos, n=1, limits=limits)
            for i, samples in zip(idxs, outs):
                s = samples[0]
                if not s.fidelity_ok():
                    results[i] = SlotResult(text="", failed=True, fail_reason="fidelity")
                    continue
                if s.regions is not None:
                    # exact split from regions — no regex
                    think_spans = [r for r in s.regions if r.kind == "think"]
                    visible_spans = [r for r in s.regions if r.kind == "visible"]
                    tok = self.policy.tokenizer
                    thinking = (
                        tok.decode([t for r in think_spans for t in s.tokens[r.start : r.end]]).strip()
                        or None
                    )
                    if thinking is not None:
                        thinking = thinking.removeprefix("<think>").strip()
                    text = tok.decode(
                        [t for r in visible_spans for t in s.tokens[r.start : r.end]]
                    ).strip()
                else:
                    thinking, text = _split_think(s.text)
                results[i] = SlotResult(text=text, thinking=thinking, sample=s, datum=datum_from_sample(s))
        return [r if r is not None else SlotResult(text="", failed=True, fail_reason="missing") for r in results]


class FrozenSeat(SeatRunner):
    """Frozen seat over a debate.models Model. max_think/max_visible are
    unenforceable on API seats — warn once; honor max_total only."""

    trained = False

    def __init__(self, model: Model, default_max_tokens: int = 1024):
        self.model = model
        self.default_max_tokens = default_max_tokens
        self._warned_limits = False

    def generate(self, requests: list[GenRequest]) -> list[SlotResult]:
        import warnings

        if not self._warned_limits:
            if any(r.limits.max_think_tokens or r.limits.max_visible_tokens for r in requests):
                warnings.warn(f"seat {self.model.alias}: think/visible caps unenforceable on frozen API seats")
                self._warned_limits = True

        results: list[Optional[SlotResult]] = [None] * len(requests)
        by_max: dict[int, list[int]] = {}
        for i, r in enumerate(requests):
            by_max.setdefault(r.limits.max_total_tokens or self.default_max_tokens, []).append(i)

        for max_tokens, idxs in by_max.items():
            inputs = [
                [ModelInput(role=m["role"], content=m["content"]) for m in requests[i].messages] for i in idxs
            ]
            # requests come from one topology step, so json_schema is uniform
            schemas = {id(requests[i].json_schema) for i in idxs}
            if len(schemas) > 1:
                raise ValueError(f"{self.model.alias}: mixed json_schema within one generate batch")
            extra = {}
            if requests[idxs[0]].json_schema is not None:
                extra["json_schema"] = requests[idxs[0]].json_schema
            responses = self.model.predict(inputs, max_new_tokens=max_tokens, num_return_sequences=1, **extra)
            if len(responses) != len(idxs):
                raise RuntimeError(
                    f"{self.model.alias}: predict returned {len(responses)} for {len(idxs)} inputs"
                )
            for i, resp in zip(idxs, responses):
                if resp.failed:
                    results[i] = SlotResult(text="", failed=True, fail_reason="model_failed", response=resp)
                    continue
                thinking = resp.thinking
                text = resp.speech
                if thinking is None:
                    thinking, text = _split_think(resp.speech)
                results[i] = SlotResult(text=text, thinking=thinking, response=resp)
        return [r if r is not None else SlotResult(text="", failed=True, fail_reason="missing") for r in results]


# ------------------------------------------------- context assembly + render


def visible_records(state: DebateState, current: CompiledSlot) -> list[SlotRecord]:
    """Rules 2-3, ordered by reveal time so contexts are prefix-stable."""
    S, t = current.speaker, current.turn
    out: list[SlotRecord] = []
    for rec in state.records:  # all flat_index < current by construction
        if rec.slot.speaker == S:
            if rec.slot.slot.visibility == Visibility.EPHEMERAL and rec.slot.turn != t:
                continue
            out.append(rec)
        elif rec.slot.slot.visibility == Visibility.PUBLIC and rec.slot.turn < t:
            out.append(rec)

    def reveal_key(rec: SlotRecord):
        own = rec.slot.speaker == S
        return (rec.slot.turn if own else rec.slot.turn + 1, 1 if own else 0, rec.slot.index)

    out.sort(key=reveal_key)
    return out


def _solo_cue(messages: list[Message]) -> Optional[str]:
    """The solo (non-debate) user turn that actually elicited the first speech."""
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return None


def render_context(
    state: DebateState, current: CompiledSlot, prompts: PromptLibrary
) -> list[Message]:
    """Rule 4. Invariants: one system message first; strict user/assistant
    alternation; ends on a user message (the current slot's instruction cue).

    Exception: with first_slot_messages set, the FIRST compiled slot is
    generated under the task source's own messages verbatim — no debate system
    card, no instruction cue — and the author's own history later renders that
    solo user turn in place of the slot's cue, so what the author sees as the
    thing it answered is what it actually answered."""
    S = current.speaker
    solo = state.first_slot_messages
    if solo is not None and current.index == 0:
        return [dict(m) for m in solo]
    solo_cue = _solo_cue(solo) if solo is not None else None

    msgs: list[Message] = [{"role": "system", "content": prompts.system(S, state.bindings[S])}]
    pending: list[str] = []
    for rec in visible_records(state, current):
        if rec.slot.speaker == S:
            if solo_cue is not None and rec.slot.index == 0:
                pending.append(solo_cue)
            else:
                pending.append(prompts.instruction(rec.slot.slot.name, S, state.bindings[S]))
            msgs.append({"role": "user", "content": "\n\n".join(pending)})
            pending = []
            msgs.append({"role": "assistant", "content": rec.text})
        else:
            author_name = state.bindings[rec.slot.speaker]["NAME"]
            pending.append(prompts.attributed(author_name, rec.slot.slot.name, rec.text))
    pending.append(prompts.instruction(current.slot.name, S, state.bindings[S]))
    msgs.append({"role": "user", "content": "\n\n".join(pending)})
    return msgs


# ---------------------------------------------------------------- turn loop


VerdictParser = Callable[[str], Optional[dict]]


class DebateRound:
    def __init__(
        self,
        topology: Topology,
        seats: dict[str, SeatRunner],
        prompts: PromptLibrary,
        *,
        verdict_parser: Optional[VerdictParser] = None,
        verdict_retries: int = 4,
        solution_extractor: Optional[Callable[[str], Any]] = None,
        fresh_positions: bool = False,
        decision_json_schema: Optional[dict] = None,
    ):
        missing = set(topology.speakers) - set(seats)
        if missing:
            raise ValueError(f"topology speakers without seats: {sorted(missing)}")
        self.topology = topology
        self.slots = topology.compile()
        self.seats = seats
        self.prompts = prompts
        self.verdict_parser = verdict_parser
        self.verdict_retries = verdict_retries
        self.decision_json_schema = decision_json_schema
        self.solution_extractor = solution_extractor
        self.fresh_positions = fresh_positions

    def run(self, states: list[DebateState]) -> list[DebateState]:
        for step in self.slots:
            live = [i for i, st in enumerate(states) if st.failed is None]
            if not live:
                break
            runner = self.seats[step.speaker]
            schema = self.decision_json_schema if step.slot.kind == Kind.DECISION else None
            requests = [
                GenRequest(render_context(states[i], step, self.prompts), slot_limits(step), json_schema=schema)
                for i in live
            ]
            results = runner.generate(requests)
            if len(results) != len(requests):
                raise RuntimeError(f"seat {step.speaker}: {len(results)} results for {len(requests)} requests")
            if step.slot.kind == Kind.DECISION and self.verdict_parser is not None:
                results = self._retry_unparseable(step, states, live, results, runner)
            for i, res in zip(live, results):
                self._ingest(states[i], step, res)
        return states

    def _retry_unparseable(
        self,
        step: CompiledSlot,
        states: list[DebateState],
        live: list[int],
        results: list[SlotResult],
        runner: SeatRunner,
    ) -> list[SlotResult]:
        retries = [0] * len(results)
        for attempt in range(self.verdict_retries):
            bad = [
                j
                for j, res in enumerate(results)
                if not res.failed and self.verdict_parser(res.text) is None
            ]
            if not bad:
                break
            requests = [
                GenRequest(
                    render_context(states[live[j]], step, self.prompts),
                    slot_limits(step),
                    json_schema=self.decision_json_schema,
                )
                for j in bad
            ]
            fresh = runner.generate(requests)
            for j, res in zip(bad, fresh):
                retries[j] += 1
                res.retries = retries[j]
                results[j] = res
        return results

    def _ingest(self, state: DebateState, step: CompiledSlot, res: SlotResult) -> None:
        if res.failed:
            state.failed = f"{step.speaker}/{step.slot.name}: {res.fail_reason}"
            return
        record = SlotRecord(
            slot=step,
            text=res.text,
            thinking=res.thinking,
            sample=res.sample,
            datum=res.datum,
            response=res.response,
            retries=res.retries,
        )
        if not res.text.strip():
            state.meta.setdefault("empty_slots", []).append(f"{step.speaker}/{step.slot.name}@{step.turn}")

        if step.slot.kind == Kind.SOLUTION and self.solution_extractor is not None:
            record.extracted = self.solution_extractor(res.text)
            if self.fresh_positions:
                if record.extracted is None:
                    state.failed = f"{step.speaker}/{step.slot.name}: unparseable solution"
                    return
                v = record.extracted
                position = f"{v:g}" if isinstance(v, float) else str(v)
                state.bindings[step.speaker]["POSITION"] = position
                for other in state.bindings:
                    if other != step.speaker:
                        state.bindings[other]["OPPONENT_POSITION"] = position

        if step.slot.kind == Kind.DECISION and self.verdict_parser is not None:
            record.extracted = self.verdict_parser(res.text)
            if record.extracted is None:
                state.failed = "verdict_unparseable"
                return

        state.records.append(record)
