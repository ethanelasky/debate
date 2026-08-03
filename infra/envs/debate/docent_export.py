"""Export debate rollouts to Docent (transluce) for transcript analysis.

Comprehensive by design: one AgentRun per debate carrying
  - an "omniscient" transcript: every slot in flat order, including private /
    ephemeral slots and thinking (as reasoning content) — the analyst's view;
  - one transcript per speaker: the EXACT rendered context of that speaker's
    final generation (regenerated via render_context, which is a pure function
    of the record store) plus its final response — "what the model saw";
  - metadata: task, verdict (winner, per-seat rulings, BOTH confidence
    channels + logit status), rewards, failure state, bindings.

Usage:
    runs = agent_runs(env)                      # after env.rollout(...)
    export_jsonl(runs, "debates.jsonl")         # local file
    upload(runs, collection_name="math-pc-rl")  # needs DOCENT_API_KEY
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ContentReasoning,
    ContentText,
    SystemMessage,
    UserMessage,
)

from infra.envs.debate.round import DebateState, render_context
from infra.envs.debate.topology import Visibility


def _assistant(text: str, thinking: Optional[str] = None, **meta) -> AssistantMessage:
    content = []
    if thinking:
        content.append(ContentReasoning(reasoning=thinking))
    content.append(ContentText(text=text))
    return AssistantMessage(content=content, metadata=meta or {})


def _omniscient_transcript(state: DebateState) -> Transcript:
    messages: list[Any] = [
        SystemMessage(
            content="Omniscient debate view: every slot in generation order, "
            "including private/ephemeral slots and reasoning no other "
            "participant saw."
        )
    ]
    for rec in state.records:
        cs = rec.slot
        messages.append(
            _assistant(
                rec.text,
                thinking=rec.thinking,
                speaker=cs.speaker,
                display_name=state.bindings.get(cs.speaker, {}).get("NAME", cs.speaker),
                turn=cs.turn,
                slot=cs.slot.name,
                kind=cs.slot.kind.value,
                visibility=cs.slot.visibility.value,
                extracted=None if rec.extracted is None else str(rec.extracted),
                retries=rec.retries,
            )
        )
    return Transcript(name="omniscient", messages=messages)


def _speaker_view(state: DebateState, speaker: str, env) -> Optional[Transcript]:
    last = None
    for rec in state.records:
        if rec.slot.speaker == speaker:
            last = rec
    if last is None:
        return None
    rendered = render_context(state_before(state, last), last.slot, env.prompts)
    messages: list[Any] = []
    for m in rendered:
        role = m["role"]
        if role == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif role == "user":
            messages.append(UserMessage(content=m["content"]))
        else:
            messages.append(_assistant(m["content"]))
    messages.append(_assistant(last.text, thinking=last.thinking, slot=last.slot.slot.name))
    return Transcript(name=f"view:{speaker}", messages=messages)


def state_before(state: DebateState, rec) -> DebateState:
    """A shallow view of the state as it was before `rec` was generated."""
    clone = DebateState(bindings=state.bindings, meta=state.meta)
    clone.records = [r for r in state.records if r.slot.index < rec.slot.index]
    # without this, exported proposer views render the debate card + library
    # cue for a first speech actually generated solo
    clone.first_slot_messages = state.first_slot_messages
    return clone


def _verdict_metadata(env, state: DebateState) -> dict[str, Any]:
    v = env._verdict(state)
    if v is None:
        return {"verdict": None}
    return {
        "verdict": {
            "ok": v.ok,
            "winner": v.winner,
            "seats": {k: s.value for k, s in v.seats.items()},
            "confidence": {
                k: {"json": c.json, "logit": c.logit, "logit_status": c.logit_status.name}
                for k, c in v.confidence.items()
            },
            "attempts": v.attempts,
        }
    }


def agent_runs(env, states: Optional[list[DebateState]] = None) -> list[AgentRun]:
    """Build one AgentRun per debate from the env's last rollout (or given
    states). Includes failed debates — those are often the interesting ones."""
    states = states if states is not None else getattr(env, "last_states", [])
    runs = []
    for i, st in enumerate(states):
        transcripts = [_omniscient_transcript(st)]
        for speaker in env.topology.speakers:
            view = _speaker_view(st, speaker, env)
            if view is not None:
                transcripts.append(view)
        meta: dict[str, Any] = {
            "debate_index": i,
            "failed": st.failed,
            "flipped": bool(st.meta.get("flipped", False)),
            "empty_slots": st.meta.get("empty_slots", []),
            "task": {k: v for k, v in st.meta.get("task", {}).items() if isinstance(v, (str, int, float, bool))},
            "bindings": {s: b.get("NAME") for s, b in st.bindings.items()},
        }
        if st.failed is None:
            meta.update(_verdict_metadata(env, st))
        runs.append(
            AgentRun(
                name=f"debate-{i}",
                description=str(st.meta.get("task", {}).get("question", ""))[:300],
                transcripts=transcripts,
                metadata=meta,
            )
        )
    return runs


def export_jsonl(runs: list[AgentRun], path: str) -> str:
    with open(path, "w") as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")
    return path


def upload(runs: list[AgentRun], collection_name: str, collection_id: Optional[str] = None) -> str:
    """Push runs to a Docent collection. Requires DOCENT_API_KEY."""
    from docent import Docent

    client = Docent(api_key=os.environ["DOCENT_API_KEY"])
    if collection_id is None:
        collection_id = client.create_collection(name=collection_name)
    client.add_agent_runs(collection_id, runs)
    return collection_id
