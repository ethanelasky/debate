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
from typing import Any, Callable, Optional, Protocol as TypingProtocol

from infra.backend.base import Datum, Sample, SamplingParams
from infra.envs.base import Policy, SlotLimits, _validate_predict_results, datum_from_sample
from infra.envs.debate.protocol import CompiledSlot, Kind, Protocol, Visibility
from infra.models.base import Model, ModelInput, ModelResponse, SamplingProfile, SpeechStructure

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


def slot_limits(cs: CompiledSlot) -> SlotLimits:
    return SlotLimits(cs.slot.max_think_tokens, cs.slot.max_visible_tokens, cs.slot.max_total_tokens)


class PromptLibrary(TypingProtocol):
    """Per-slot templates; the only place speaker identity/instructions live."""

    def system(self, speaker: str, bindings: dict[str, str]) -> str: ...

    def instruction(self, slot_name: str, speaker: str, bindings: dict[str, str]) -> str: ...

    def preamble_messages(self, speaker: str, bindings: dict[str, str]) -> list[str]: ...

    def attributed(
        self,
        author_name: str,
        slot_name: str,
        text: str,
        *,
        reader: Optional[str] = None,
        author: Optional[str] = None,
        reader_bindings: Optional[dict[str, str]] = None,
    ) -> str: ...


@dataclass
class SlotRecord:
    slot: CompiledSlot
    text: str                       # post-think visible text (think stripped at record time)
    thinking: Optional[str] = None  # stored; NEVER rendered into any context
    extracted: Any = None           # solution: parsed answer; decision: verdict dict
    # Solution slots only. DebateEnv's answer extractor computes this from the
    # same family parse that produced ``extracted`` so shaping, census, and
    # transcript metadata cannot drift by reparsing independently.
    answer_format_valid: Optional[bool] = None
    sample: Optional[Sample] = None      # trained seats
    datum: Optional[Datum] = None        # trained seats (advantages zeroed; mask from budget forcing)
    response: Optional[ModelResponse] = None  # frozen seats
    retries: int = 0
    truncated: bool = False              # speech_token_limit fired on this slot's visible text


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
                "truncated": r.truncated,
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
    # Decision slots generate under SpeechStructure.DECISION so wrappers that
    # gate logprob capture on the structure (LocalModel) return the token
    # channel the judge-logit scan reads.
    decision: bool = False


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
    # Pre-opened template (Olmo-Think line): the chat template emits <think>
    # in the GENERATION PROMPT, so the completion carries only the close —
    # the pair regex can never match and the private reasoning would enter
    # the public transcript verbatim (measured live: 27% of the 32B think
    # debate arm's transcript views leaked ~1.4k tokens each, 2026-08-12).
    j = raw.find("</think>")
    if j >= 0:
        return raw[:j].strip(), raw[j + len("</think>") :].strip()
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
            _validate_predict_results(
                outs,
                requests=len(convos),
                samples_per_request=1,
                stage="trained seat",
            )
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

    def __init__(
        self,
        model: Model,
        default_max_tokens: int = 1024,
        sampling: Optional[SamplingProfile] = None,
    ):
        self.model = model
        self.default_max_tokens = default_max_tokens
        # The seat's resolved sampling profile (env config plumbs the YAML's
        # train profile here); without it the wrapper/server defaults apply.
        self.sampling = sampling
        self._warned_limits = False

    def generate(self, requests: list[GenRequest]) -> list[SlotResult]:
        import warnings

        if not self._warned_limits:
            if any(r.limits.max_think_tokens or r.limits.max_visible_tokens for r in requests):
                warnings.warn(f"seat {self.model.alias}: think/visible caps unenforceable on frozen API seats")
                self._warned_limits = True

        # Profile fields ride as predict kwargs; wrappers place the ones they
        # support and ignore the rest.
        sampling_kwargs: dict[str, Any] = {}
        if self.sampling is not None:
            for name in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                value = getattr(self.sampling, name)
                if value is not None:
                    sampling_kwargs[name] = value

        results: list[Optional[SlotResult]] = [None] * len(requests)
        by_max: dict[int, list[int]] = {}
        for i, r in enumerate(requests):
            by_max.setdefault(r.limits.max_total_tokens or self.default_max_tokens, []).append(i)

        for max_tokens, idxs in by_max.items():
            inputs = [
                [ModelInput(role=m["role"], content=m["content"]) for m in requests[i].messages] for i in idxs
            ]
            # requests come from one protocol step, so json_schema/decision are uniform
            schemas = {id(requests[i].json_schema) for i in idxs}
            if len(schemas) > 1:
                raise ValueError(f"{self.model.alias}: mixed json_schema within one generate batch")
            if len({requests[i].decision for i in idxs}) > 1:
                raise ValueError(f"{self.model.alias}: mixed decision kinds within one generate batch")
            extra = dict(sampling_kwargs)
            if requests[idxs[0]].json_schema is not None:
                extra["json_schema"] = requests[idxs[0]].json_schema
            if requests[idxs[0]].decision:
                extra["speech_structure"] = SpeechStructure.DECISION
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


# -------------------------------------------------- post-hoc speech truncation


_SPEECH_ENCODER = None


def _speech_encoder():
    """cl100k_base, lazily — only speech_token_limit users pay the tiktoken
    import (matches the old repo's lazy-encoder pattern)."""
    global _SPEECH_ENCODER
    if _SPEECH_ENCODER is None:
        import tiktoken

        _SPEECH_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _SPEECH_ENCODER


def truncate_speech_to_token_limit(speech: str, token_limit: int) -> tuple[str, bool]:
    """Post-hoc TRANSCRIPT-VISIBLE truncation, ported from the old repo's
    utils/string_utils.truncate_speech_to_token_limit (working tree): count
    cl100k_base tokens, cut at token_limit, decode the kept prefix. Text up to
    a trailing '</think>' is preserved uncounted (the old Qwen handling —
    applied unconditionally here since model aliases are not plumbed in; in
    this repo thinking is stripped before recording, so the branch is normally
    inert). Returns (text, fired)."""
    if not speech or token_limit <= 0:
        return speech, False
    prefix, countable = "", speech
    if "</think>" in speech:
        end = speech.rfind("</think>") + len("</think>")
        prefix, countable = speech[:end], speech[end:]
    enc = _speech_encoder()
    tokens = enc.encode(countable)
    if len(tokens) <= token_limit:
        return speech, False
    return prefix + enc.decode(tokens[:token_limit]), True


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


def render_context(
    state: DebateState, current: CompiledSlot, prompts: PromptLibrary
) -> list[Message]:
    """Rule 4. Invariants: one system message first; ends on a user message
    (the current slot's instruction cue); transcript-derived content strictly
    alternates user/assistant. Configured preamble messages render as SEPARATE
    user messages right after the system message, before any transcript
    content — consecutive user messages at the context head are deliberate
    (message-boundary prompt caches can then reuse the byte-stable shared
    message across seats; frozen API seats accept repeated roles).

    Exception: with first_slot_messages set, the FIRST compiled slot is
    GENERATED under the task source's own messages verbatim — no debate
    system card, no preamble, no cue. The author's LATER contexts render
    normally: debate framing (system + preamble messages), then slot 0 like
    any other own slot — its cue as user content, its answer as the first
    assistant turn. The answer is thereby presented as if it had been
    produced under the debate framing (deliberate, 2026-08-04); packs keep
    that honest by making the slot-0 cue byte-identical to the solo
    eliciting message (both MB's blind instructions and the math packs'
    <ANSWER_GEN_USER> are single-sourced, so this holds by construction).
    The generation view is the ONE deliberate exception; if a second context
    exception is ever needed, convert to per-slot context sources instead of
    adding another branch here."""
    S = current.speaker
    solo = state.first_slot_messages
    if solo is not None and current.index == 0:
        return [dict(m) for m in solo]
    visible = visible_records(state, current)

    msgs: list[Message] = [{"role": "system", "content": prompts.system(S, state.bindings[S])}]
    for pm in prompts.preamble_messages(S, state.bindings[S]):
        msgs.append({"role": "user", "content": pm})
    pending: list[str] = []
    for rec in visible:
        if rec.slot.speaker == S:
            pending.append(prompts.instruction(rec.slot.slot.name, S, state.bindings[S]))
            msgs.append({"role": "user", "content": "\n\n".join(pending)})
            pending = []
            msgs.append({"role": "assistant", "content": rec.text})
        else:
            author = rec.slot.speaker
            pending.append(
                prompts.attributed(
                    state.bindings[author]["NAME"],
                    rec.slot.slot.name,
                    rec.text,
                    reader=S,
                    author=author,
                    reader_bindings=state.bindings[S],
                )
            )
    pending.append(prompts.instruction(current.slot.name, S, state.bindings[S]))
    msgs.append({"role": "user", "content": "\n\n".join(pending)})
    return msgs


# ------------------------------------------------------- verdict retry feedback


def create_retry_feedback(
    failed_text: str, retry_attempt: int, schema: str, seat_names: list[str]
) -> str:
    """Error-feedback user message for an unparseable-verdict retry, ported
    from the old repo's Judge._create_retry_prompt (ai_debate/debate/judge.py,
    working tree). The scaffolding (PARSING ERROR header, previous-response
    snippet capped at 500 chars, format-block framing, closing "Ensure" list)
    and the collaborative branch keep the old wording verbatim. The
    competitive example lines substitute THIS round's seat display names — the
    old winner-shape constants were byte-copies of the old math arm's decision
    instruction ('"Debater A" ... the Proposer's boxed answer'), whose seat
    naming and win semantics do not exist here; the confidence / Tie /
    no-other-keys bullets and the closing sentence are kept verbatim."""
    error_msg = (
        f"PARSING ERROR: Your previous verdict could not be parsed. "
        f"This is retry attempt {retry_attempt}.\n\n"
    )
    if failed_text:
        snippet = failed_text[:500]
        if len(failed_text) > 500:
            snippet += "..."
        error_msg += f"Your previous response:\n{snippet}\n\n"
    error_msg += "Please provide your verdict in the following JSON format:\n```json\n{\n"

    if schema == "competitive":
        names = " | ".join(f'"{n}"' for n in seat_names)
        error_msg += f'  "winner": {names} | "Tie",\n'
        error_msg += '  "confidence": 0.50-1.00\n'
        error_msg += "}\n```\n\n"
        error_msg += "Ensure:\n"
        error_msg += (
            f'- "winner" is exactly one of {names}, written exactly as shown\n'
        )
        error_msg += "- Confidence must be between 0.50 and 1.00 inclusive.\n"
        error_msg += (
            '- "Tie" is reserved for a debate that is UNRESOLVABLE for a structural '
            "reason — for example, the question is cut off or malformed. An objection "
            "you could not fully confirm or refute is NOT a tie: choose the more "
            "likely winner and express the uncertainty in your confidence. If you do "
            'declare "Tie", set confidence to 0.50.\n'
        )
        error_msg += "- Do not add any other keys to the JSON object.\n"
        error_msg += "- The JSON is valid and complete\n"
        error_msg += (
            "\nThe JSON object must be the very last thing you output, with nothing "
            "after it.\n"
        )
        return error_msg

    for i, alias in enumerate(seat_names):
        error_msg += f'  "{alias}": {{\n'
        error_msg += '    "verdict": "correct" or "incorrect",\n'
        error_msg += '    "confidence": <number between 0 and 1>\n'
        error_msg += "  }"
        if i < len(seat_names) - 1:
            error_msg += ","
        error_msg += "\n"

    error_msg += "}\n```\n\n"
    error_msg += "Ensure:\n"
    error_msg += "- The verdict field contains exactly 'correct' or 'incorrect' (lowercase)\n"
    error_msg += "- The confidence field is a number between 0.0 and 1.0\n"
    error_msg += "- The JSON is valid and complete\n"

    return error_msg


# ---------------------------------------------------------------- turn loop


VerdictParser = Callable[[str], Optional[dict]]
# A solution answer plus the generic strict-format result from one family
# parse. Keeping this tuple internal lets DebateRound stay task-domain blind.
AnswerExtractor = Callable[[str], tuple[Any, bool]]

# Custom mid-round position binding: called with (state, chooser_speaker,
# extracted) when a solution slot's extraction lands; replaces the default
# fresh-positions binding (which only sets the chooser's own POSITION and
# others' OPPONENT_POSITION — a binder can set every seat's, e.g. MB's
# chosen-side stance mapping for chooser, opponent, and judge).
PositionBinder = Callable[["DebateState", str, Any], None]


class DebateRound:
    def __init__(
        self,
        protocol: Protocol,
        seats: dict[str, SeatRunner],
        prompts: PromptLibrary,
        *,
        verdict_parser: Optional[VerdictParser] = None,
        verdict_retries: int = 4,
        judge_schema: str = "competitive",
        answer_extractor: Optional[AnswerExtractor] = None,
        fresh_positions: bool = False,
        decision_json_schema: Optional[dict] = None,
        speech_token_limit: Optional[int] = None,
        position_binder: Optional[PositionBinder] = None,
        solution_retries: int = 0,
        solution_retry_feedback: Optional[Callable[[str, int], str]] = None,
    ):
        missing = set(protocol.speakers) - set(seats)
        if missing:
            raise ValueError(f"protocol speakers without seats: {sorted(missing)}")
        self.protocol = protocol
        self.slots = protocol.compile()
        self.seats = seats
        self.prompts = prompts
        self.verdict_parser = verdict_parser
        self.verdict_retries = verdict_retries
        self.decision_json_schema = decision_json_schema
        self.judge_schema = judge_schema  # shapes the retry feedback's example
        self.answer_extractor = answer_extractor
        self.fresh_positions = fresh_positions
        self.position_binder = position_binder
        # Solution-slot retry mirrors the verdict retry, but only when a
        # feedback builder is supplied (its wording is format-specific and
        # must match what the solution prompt asked for).
        self.solution_retries = solution_retries
        self.solution_retry_feedback = solution_retry_feedback
        # Truncation targets SPEECH slots of non-judge speakers only; the
        # judge is the decision slot's speaker (its deliberation/verdict text
        # is never cut).
        self.speech_token_limit = speech_token_limit
        decision = protocol.decision_slot
        self._judge_speaker = decision.speaker if decision is not None else None

    def run(self, states: list[DebateState]) -> list[DebateState]:
        # first_slot_messages replaces slot 0's context wholesale; its author
        # only replays that exchange later when slot 0 is PUBLIC (the replay
        # keys off visibility-filtered records), so a non-public slot 0 would
        # silently drop the solo turn from every context.
        if (
            self.slots
            and self.slots[0].slot.visibility != Visibility.PUBLIC
            and any(st.first_slot_messages is not None for st in states)
        ):
            raise ValueError(
                f"first_speech_non_debate_aware requires a PUBLIC first slot, got "
                f"{self.slots[0].speaker}/{self.slots[0].slot.name} "
                f"({self.slots[0].slot.visibility.value})"
            )
        for step in self.slots:
            live = [i for i, st in enumerate(states) if st.failed is None]
            if not live:
                break
            runner = self.seats[step.speaker]
            decision = step.slot.kind == Kind.DECISION
            schema = self.decision_json_schema if decision else None
            requests = [
                GenRequest(
                    render_context(states[i], step, self.prompts),
                    slot_limits(step),
                    json_schema=schema,
                    decision=decision,
                )
                for i in live
            ]
            results = runner.generate(requests)
            if len(results) != len(requests):
                raise RuntimeError(f"seat {step.speaker}: {len(results)} results for {len(requests)} requests")
            if step.slot.kind == Kind.DECISION and self.verdict_parser is not None:
                results = self._retry_unparseable(step, states, live, results, runner)
            if (
                step.slot.kind == Kind.SOLUTION
                and self.answer_extractor is not None
                and self.solution_retry_feedback is not None
            ):
                results = self._retry_with_feedback(
                    step,
                    states,
                    live,
                    results,
                    runner,
                    parser=lambda text: self.answer_extractor(text)[0],
                    feedback=lambda st, text, attempt: self.solution_retry_feedback(text, attempt),
                    max_retries=self.solution_retries,
                )
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
        """Verdict feedback retry, ported from the old repo's
        Judge.render_decision. Counting/exhaustion semantics are unchanged."""

        def feedback(st: DebateState, failed_text: str, attempt: int) -> str:
            bindings = st.bindings[step.speaker]
            seat_names = [
                n for n in (bindings.get("NAME", ""), bindings.get("OPPONENT_NAME", "")) if n
            ]
            return create_retry_feedback(failed_text, attempt, self.judge_schema, seat_names)

        return self._retry_with_feedback(
            step,
            states,
            live,
            results,
            runner,
            parser=self.verdict_parser,
            feedback=feedback,
            max_retries=self.verdict_retries,
            json_schema=self.decision_json_schema,
        )

    def _retry_with_feedback(
        self,
        step: CompiledSlot,
        states: list[DebateState],
        live: list[int],
        results: list[SlotResult],
        runner: SeatRunner,
        *,
        parser: Callable[[str], Any],
        feedback: Callable[[DebateState, str, int], str],
        max_retries: int,
        json_schema: Optional[dict] = None,
    ) -> list[SlotResult]:
        """Shared feedback-retry loop (old repo's Judge.render_decision shape):
        each retry re-renders the base context and appends the latest FAILED
        attempt as an assistant turn plus an error-feedback user message, so
        the seat sees what it wrote and why it was rejected. Attempts do not
        accumulate: retry N carries only the attempt-N-1 failure."""
        retries = [0] * len(results)
        for attempt in range(max_retries):
            bad = [
                j
                for j, res in enumerate(results)
                if not res.failed and parser(res.text) is None
            ]
            if not bad:
                break
            requests = []
            for j in bad:
                st = states[live[j]]
                messages = render_context(st, step, self.prompts) + [
                    {"role": "assistant", "content": results[j].text},
                    {"role": "user", "content": feedback(st, results[j].text, retries[j] + 1)},
                ]
                requests.append(
                    GenRequest(
                        messages,
                        slot_limits(step),
                        json_schema=json_schema,
                        decision=step.slot.kind == Kind.DECISION,
                    )
                )
            fresh = runner.generate(requests)
            if len(fresh) != len(requests):
                raise RuntimeError(
                    f"seat {step.speaker} retry generation returned {len(fresh)} results "
                    f"for {len(requests)} requests"
                )
            for j, res in zip(bad, fresh):
                retries[j] += 1
                res.retries = retries[j]
                results[j] = res
        return results

    def _ingest(self, state: DebateState, step: CompiledSlot, res: SlotResult) -> None:
        if res.failed:
            state.failed = f"{step.speaker}/{step.slot.name}: {res.fail_reason}"
            return
        text, truncated = res.text, False
        if (
            self.speech_token_limit is not None
            and step.slot.kind == Kind.SPEECH
            and step.speaker != self._judge_speaker
        ):
            text, truncated = truncate_speech_to_token_limit(text, self.speech_token_limit)
        record = SlotRecord(
            slot=step,
            text=text,
            thinking=res.thinking,
            sample=res.sample,
            datum=res.datum,
            response=res.response,
            retries=res.retries,
            truncated=truncated,
        )
        if not res.text.strip():
            state.meta.setdefault("empty_slots", []).append(f"{step.speaker}/{step.slot.name}@{step.turn}")

        if step.slot.kind == Kind.SOLUTION and self.answer_extractor is not None:
            record.extracted, record.answer_format_valid = self.answer_extractor(res.text)
            binds_position = self.position_binder is not None or self.fresh_positions
            if record.extracted is None and binds_position:
                # Record the failed attempt (extracted None) before failing the
                # round: its retry count is otherwise unobservable downstream.
                state.records.append(record)
                state.failed = f"{step.speaker}/{step.slot.name}: unparseable solution"
                return
            if self.position_binder is not None:
                self.position_binder(state, step.speaker, record.extracted)
            elif self.fresh_positions:
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
