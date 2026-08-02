"""Verdict parsing + judge-logit confidence.

Two confidence channels, never merged:
- json: the confidence the judge wrote in its own verdict object.
- logit: P(the judge continues with the winner it named), read off the sampled
  token logprobs at the DECISION POINT — the first character where the emitted
  value diverges from its nearest competitor in the candidate set. Structural,
  so it needs no marker dialect: the schema and the seat names determine it.

Caveat on the logit channel: p is P(continue with v | prefix). Probability mass
can leak to strings outside the candidate set, so p_A + p_B <= 1 rather than
== 1.

Generation and retries belong to round.py; this module only parses and scores
an already-generated verdict slot.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel

Schema = Literal["competitive", "collaborative"]
DecodeFn = Callable[[list[int]], str]

CORRECT_WORDS = frozenset({"correct", "right", "yes"})
INCORRECT_WORDS = frozenset({"incorrect", "wrong", "no"})
TIE_WORDS = frozenset({"tie", "draw"})
NEITHER_WORDS = frozenset({"neither", "none"})
VERDICT_WORDS = CORRECT_WORDS | INCORRECT_WORDS | TIE_WORDS

# Byte-level-BPE surface markers from OpenAI-compatible logprobs payloads.
_BPE_SURFACE = str.maketrans({"Ġ": " ", "Ċ": "\n"})

_SEP = re.compile(r"[\s_\-]+")


def _norm(s: str) -> str:
    """Collapsing normalization, for NAME EQUALITY only (changes length)."""
    return _SEP.sub(" ", str(s).strip().lower())


def _norm_flat(s: str) -> str:
    """Length-preserving normalization, for CHAR-OFFSET arithmetic: separators
    map to a single space each so 'Debater_A'/'Debater B' compare position by
    position."""
    return "".join(" " if c in "_-" or c.isspace() else c for c in s.lower())


def match_seat(value: str, seat_names: list[str]) -> Optional[str]:
    """Resolve an emitted name to a seat.

    Exact match after separator normalization, else a unique MULTI-TOKEN suffix
    of exactly one seat (so a seat 'run3_Debater_A' accepts 'Debater_A').
    Substring matching is refused in BOTH directions: a seat name occurring
    inside the value would make "not Debater_A" resolve to Debater_A, and a
    bare single token like "B" carries too little to distinguish seats whose
    names we do not control.
    """
    v = _norm(value)
    if not v:
        return None
    for name in seat_names:
        if _norm(name) == v:
            return name
    if " " not in v:
        return None
    hits = [n for n in seat_names if _norm(n).endswith(" " + v)]
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------------- schemas


class SeatVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIE = "tie"
    UNKNOWN = "unknown"


class LogitStatus(Enum):
    SCORED = "scored"
    NO_PIECES = "no_pieces"
    MISALIGNED = "misaligned"
    TRUNCATED = "truncated"
    VALUE_NOT_FOUND = "value_not_found"
    NO_DECISION_TOKEN = "no_decision_token"
    TIE_WORD = "tie_word"
    BAD_LOGPROB = "bad_logprob"


#: Pieces were available and the scan still lost the signal — worth alarming on.
#: NO_PIECES/TRUNCATED/TIE_WORD are absence by design.
DEGRADED_LOGIT_STATUSES = frozenset(
    {LogitStatus.MISALIGNED, LogitStatus.VALUE_NOT_FOUND, LogitStatus.NO_DECISION_TOKEN, LogitStatus.BAD_LOGPROB}
)


@dataclass(frozen=True)
class Confidence:
    json: Optional[float]
    logit: Optional[float]
    json_provenance: Literal["elicited", "absent", "tie"]
    logit_status: LogitStatus


@dataclass
class Verdict:
    schema: Schema
    seats: dict[str, SeatVerdict]
    confidence: dict[str, Confidence]
    winner: Optional[str]
    ok: bool
    raw: str
    attempts: int = 1
    attempt_raws: list[str] = field(default_factory=list)
    logit_pieces_source: Optional[Literal["texts", "ids"]] = None


class JudgeConfig(BaseModel):
    # 'schema' shadows a BaseModel attribute.
    schema_name: Schema = "competitive"
    retries: int = 4


# ------------------------------------------------------------------- parsing


def _clamp_conf(c: Any) -> Optional[float]:
    if c is None:
        return None
    try:
        v = float(c)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if not 0.5 <= v <= 1.0:
        warnings.warn(f"judge confidence {v} outside [0.5, 1.0]; clamping")
        v = min(max(v, 0.5), 1.0)
    return v


def _validate(obj: Any, schema: Schema, seat_names: list[str]) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if schema == "competitive":
        w = obj.get("winner")
        if not isinstance(w, str) or not w.strip():
            return None
        n = _norm(w)
        if n not in TIE_WORDS and n not in NEITHER_WORDS and match_seat(w, seat_names) is None:
            return None
        return obj
    resolved: dict[str, str] = {}
    for key, val in obj.items():
        seat = match_seat(key, seat_names)
        if seat is None:
            continue  # unexpected key; warned once in _interpret, on the accepted parse
        if not isinstance(val, dict):
            return None
        v = val.get("verdict")
        if not isinstance(v, str) or _norm(v) not in VERDICT_WORDS:
            return None
        resolved[seat] = key
    if set(resolved) != set(seat_names):
        return None
    return obj


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_verdict(text: str, schema: Schema, seat_names: list[str]) -> Optional[dict]:
    """Fenced ```json blocks first (last block first), else every '{' in the
    text is tried as an object start and the LAST one that validates wins — a
    judge that restates the schema example before its real verdict must not be
    read off the example."""
    if not text:
        return None
    for m in reversed(list(_FENCE.finditer(text))):
        try:
            obj = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        got = _validate(obj, schema, seat_names)
        if got is not None:
            return got

    decoder = json.JSONDecoder()
    found: list[dict] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except ValueError:
            continue
        got = _validate(obj, schema, seat_names)
        if got is not None:
            found.append(got)
    return found[-1] if found else None


def make_verdict_parser(cfg: JudgeConfig, seat_names: list[str]) -> Callable[[str], Optional[dict]]:
    """The thin callable round.py drives its retry loop with."""
    return lambda text: parse_verdict(text, cfg.schema_name, seat_names)


# -------------------------------------------------------------- interpretation


@dataclass(frozen=True)
class _Interpretation:
    seats: dict[str, SeatVerdict]
    winner: Optional[str]
    json_conf: dict[str, Optional[float]]
    tie: bool


def _interpret(parsed: dict, schema: Schema, seat_names: list[str]) -> _Interpretation:
    if schema == "competitive":
        w = str(parsed.get("winner", ""))
        c = _clamp_conf(parsed.get("confidence"))
        n = _norm(w)
        if n in TIE_WORDS:
            return _Interpretation({s: SeatVerdict.TIE for s in seat_names}, None, {s: None for s in seat_names}, True)
        if n in NEITHER_WORDS:
            # Double loss, NOT a tie: both seats ruled incorrect.
            return _Interpretation(
                {s: SeatVerdict.INCORRECT for s in seat_names}, None, {s: None for s in seat_names}, False
            )
        winner = match_seat(w, seat_names)
        seats = {s: (SeatVerdict.CORRECT if s == winner else SeatVerdict.INCORRECT) for s in seat_names}
        conf = {s: (c if s == winner else (None if c is None else 1.0 - c)) for s in seat_names}
        return _Interpretation(seats, winner, conf, False)

    seats: dict[str, SeatVerdict] = {}
    conf: dict[str, Optional[float]] = {}
    for key, val in parsed.items():
        seat = match_seat(key, seat_names)
        if seat is None or not isinstance(val, dict):
            warnings.warn(f"judge verdict has unexpected key {key!r}; ignoring")
            continue
        word = _norm(str(val.get("verdict", "")))
        if word in CORRECT_WORDS:
            seats[seat] = SeatVerdict.CORRECT
        elif word in INCORRECT_WORDS:
            seats[seat] = SeatVerdict.INCORRECT
        elif word in TIE_WORDS:
            seats[seat] = SeatVerdict.TIE
        else:
            seats[seat] = SeatVerdict.UNKNOWN
        conf[seat] = _clamp_conf(val.get("confidence"))
    for s in seat_names:
        seats.setdefault(s, SeatVerdict.UNKNOWN)
        conf.setdefault(s, None)
    seats = {s: seats[s] for s in seat_names}
    if any(v == SeatVerdict.TIE for v in seats.values()):
        # A tie is a property of the round, not of one seat.
        seats = {s: SeatVerdict.TIE for s in seat_names}
        conf = {s: None for s in seat_names}
        return _Interpretation(seats, None, conf, True)
    winners = [s for s, v in seats.items() if v == SeatVerdict.CORRECT]
    winner = winners[0] if len(winners) == 1 and len(seat_names) > 1 else None
    return _Interpretation(seats, winner, {s: conf[s] for s in seat_names}, False)


# --------------------------------------------------------------- logit scan


def _pieces_and_logprobs(response: Any, decode_fn: Optional[DecodeFn]):
    """-> (pieces, logprobs, source) or (None, None, status)."""
    texts = getattr(response, "response_token_texts", None)
    logprobs = list(getattr(response, "response_logprobs", None) or getattr(response, "logprobs", None) or [])
    if texts:
        if not logprobs or len(texts) != len(logprobs) or any(not isinstance(t, str) for t in texts):
            return None, None, LogitStatus.MISALIGNED
        return [t.translate(_BPE_SURFACE) for t in texts], logprobs, "texts"

    tokens = list(getattr(response, "response_tokens", None) or getattr(response, "tokens", None) or [])
    if decode_fn is None or not tokens or not logprobs:
        return None, None, LogitStatus.NO_PIECES
    if len(tokens) != len(logprobs):
        return None, None, LogitStatus.MISALIGNED
    try:
        pieces = [decode_fn([t]) for t in tokens]
    except Exception:
        return None, None, LogitStatus.MISALIGNED
    if any(not isinstance(p, str) for p in pieces):
        return None, None, LogitStatus.MISALIGNED
    return pieces, logprobs, "ids"


def _flex(name: str) -> str:
    return r"[\s_\-]+".join(re.escape(p) for p in _SEP.split(name.strip()) if p)


def _value_end(s: str, start: int) -> int:
    i = start
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == '"':
            return i
        i += 1
    return -1


def decision_point(value: str, value_start: int, candidates: list[str]) -> int:
    """First char at which `value` diverges from its NEAREST competitor."""
    v = _norm_flat(value)
    best = 0
    for c in candidates:
        cn = _norm_flat(c)
        if cn == v:
            continue
        k = 0
        while k < len(v) and k < len(cn) and v[k] == cn[k]:
            k += 1
        best = max(best, k)
    return value_start + best


def decision_index(pieces: list[str], d: int) -> Optional[int]:
    """First token whose cumulative decoded prefix STRICTLY crosses d.

    The strict '>' is load-bearing: Qwen3-8B tokenizes 'Debater A' as
    ['De','b','ater',' A'], so a decision point landing exactly on the
    'ater'/' A' boundary must select ' A'; '>=' would select 'ater' and score
    the wrong token.
    """
    cum = 0
    for i, piece in enumerate(pieces):
        cum += len(piece)
        if cum > d:
            return i
    return None


def _score_span(
    pieces: list[str], logprobs: list, d: int, value_end: int
) -> tuple[Optional[float], LogitStatus]:
    start_idx = decision_index(pieces, d)
    if start_idx is None:
        return None, LogitStatus.NO_DECISION_TOKEN
    total = 0.0
    start = sum(len(p) for p in pieces[:start_idx])
    for i in range(start_idx, len(pieces)):
        if start >= value_end:
            break
        start += len(pieces[i])
        try:
            total += float(logprobs[i])
        except (TypeError, ValueError):
            return None, LogitStatus.BAD_LOGPROB
    try:
        p = math.exp(total)
    except OverflowError:
        return None, LogitStatus.BAD_LOGPROB
    if not math.isfinite(p):
        return None, LogitStatus.BAD_LOGPROB
    return min(max(p, 0.0), 1.0), LogitStatus.SCORED


def _scan(
    response: Any,
    decode_fn: Optional[DecodeFn],
    schema: Schema,
    seat_names: list[str],
    parsed: Optional[dict],
) -> tuple[dict[str, tuple[Optional[float], LogitStatus]], Optional[str]]:
    def all_seats(status: LogitStatus):
        return {s: (None, status) for s in seat_names}

    if response is None or parsed is None:
        return all_seats(LogitStatus.NO_PIECES), None

    pieces, logprobs, source = _pieces_and_logprobs(response, decode_fn)
    if pieces is None:
        return all_seats(source), None
    # Truncation gate on BOTH channels: a value that never finished is not a
    # decision, and a schema-shaped fragment inside reasoning must not score.
    if getattr(response, "stop_reason", None) != "stop":
        return all_seats(LogitStatus.TRUNCATED), source

    S = "".join(pieces)
    reparsed = parse_verdict(S, schema, seat_names)
    if reparsed is None or _interpret(reparsed, schema, seat_names) != _interpret(parsed, schema, seat_names):
        # The joined pieces must be the same text the accepted parse came from.
        return all_seats(LogitStatus.MISALIGNED), source

    if schema == "competitive":
        matches = list(re.finditer(r'"winner"\s*:\s*"', S))
        if not matches:
            return all_seats(LogitStatus.VALUE_NOT_FOUND), source
        vs = matches[-1].end()
        ve = _value_end(S, vs)
        if ve < 0:
            return all_seats(LogitStatus.VALUE_NOT_FOUND), source
        v = S[vs:ve]
        n = _norm(v)
        if n in TIE_WORDS or n in NEITHER_WORDS:
            # No named winner: nothing directional to score.
            return all_seats(LogitStatus.TIE_WORD), source
        winner = match_seat(v, seat_names)
        if winner is None:
            return all_seats(LogitStatus.VALUE_NOT_FOUND), source
        candidates = list(seat_names) + sorted(TIE_WORDS | NEITHER_WORDS)
        p, status = _score_span(pieces, logprobs, decision_point(v, vs, candidates), ve)
        if status != LogitStatus.SCORED:
            return all_seats(status), source
        return {s: ((p, status) if s == winner else (1.0 - p, status)) for s in seat_names}, source

    out: dict[str, tuple[Optional[float], LogitStatus]] = {}
    candidates = sorted(VERDICT_WORDS)
    for seat in seat_names:
        pat = re.compile(r'"' + _flex(seat) + r'"\s*:\s*\{[^{}]*?"verdict"\s*:\s*"')
        matches = list(pat.finditer(S))
        if not matches:
            out[seat] = (None, LogitStatus.VALUE_NOT_FOUND)
            continue
        vs = matches[-1].end()
        ve = _value_end(S, vs)
        if ve < 0:
            out[seat] = (None, LogitStatus.VALUE_NOT_FOUND)
            continue
        v = S[vs:ve]
        word = _norm(v)
        if word in TIE_WORDS:
            out[seat] = (None, LogitStatus.TIE_WORD)
            continue
        if word not in CORRECT_WORDS and word not in INCORRECT_WORDS:
            out[seat] = (None, LogitStatus.VALUE_NOT_FOUND)
            continue
        p, status = _score_span(pieces, logprobs, decision_point(v, vs, candidates), ve)
        if status != LogitStatus.SCORED:
            out[seat] = (None, status)
        else:
            out[seat] = (p if word in CORRECT_WORDS else 1.0 - p, status)
    return out, source


def scan_decision_logit(
    response: Any,
    decode_fn: Optional[DecodeFn],
    schema: Schema,
    seat_names: list[str],
    parsed: Optional[dict],
) -> dict[str, tuple[Optional[float], LogitStatus]]:
    """P(seat correct) per seat from the sampled decision tokens, with the
    reason when there is none. `response` is a ModelResponse (frozen seat) or a
    backend Sample (trained seat)."""
    return _scan(response, decode_fn, schema, seat_names, parsed)[0]


# ------------------------------------------------------------------- entry


def verdict_from_slot(
    text: str,
    response: Any,
    decode_fn: Optional[DecodeFn],
    cfg: JudgeConfig,
    seat_names: list[str],
    *,
    attempts: int = 1,
    attempt_raws: Optional[list[str]] = None,
) -> Verdict:
    """Build the full Verdict from one already-generated judge slot: parse the
    accepted attempt, then scan ITS logprobs. Both confidence channels come
    from the same attempt."""
    schema = cfg.schema_name
    parsed = parse_verdict(text, schema, seat_names)
    if parsed is None:
        return Verdict(
            schema=schema,
            seats={s: SeatVerdict.UNKNOWN for s in seat_names},
            confidence={s: Confidence(None, None, "absent", LogitStatus.NO_PIECES) for s in seat_names},
            winner=None,
            ok=False,
            raw=text,
            attempts=attempts,
            attempt_raws=list(attempt_raws or []),
        )

    interp = _interpret(parsed, schema, seat_names)
    logits, source = _scan(response, decode_fn, schema, seat_names, parsed)
    confidence = {}
    for s in seat_names:
        jc = interp.json_conf.get(s)
        provenance = "tie" if interp.tie else ("elicited" if jc is not None else "absent")
        lp, status = logits.get(s, (None, LogitStatus.NO_PIECES))
        confidence[s] = Confidence(jc, lp, provenance, status)
    return Verdict(
        schema=schema,
        seats=dict(interp.seats),
        confidence=confidence,
        winner=interp.winner,
        ok=True,
        raw=text,
        attempts=attempts,
        attempt_raws=list(attempt_raws or []),
        logit_pieces_source=source,
    )
