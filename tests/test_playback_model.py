"""Playback model + judge-ablation cache round trip.

The judge-prompt ablation is only valid if every rung judges byte-identical
debates. That rests on one invariant, tested here end to end offline:

    the context a debater seat is asked with during a REPLAY run is the same
    context it was asked with during the RECORDING run

so the cache — keyed by sha256 of the rendered context — hits every time. The
recorder has to reconstruct each slot's context from the finished state, which
means truncating the state's records back to what existed when that slot ran;
skipping that truncation is the obvious way to get this wrong, so it is tested
directly.

Also covers the deliberate absence of a fallback: an unrecorded context must
raise, never silently substitute a speech.

Everything here is scripted and synthetic — never the real MB data files.
"""

from __future__ import annotations

import json

import pytest

from infra.envs.debate.round import render_context
from infra.models.base import ModelInput
from infra.models.playback_model import PlaybackCacheMiss, PlaybackModel, context_key
from test_choice_mode import choice_config, prompt_file  # noqa: F401  (fixture)
from infra.envs.base import Task
from infra.envs.debate.env import DebateEnv
from infra.envs.tasks.monitoringbench import (
    MB_POSITION_ATTACK,
    MB_POSITION_HONEST,
    MonitoringBenchFamily,
)
from test_debate_env import ScriptedModel


class MBTaskSource:
    """Like test_choice_mode's, but each task carries DISTINCT trajectory text.

    That difference is the realistic case and the one worth testing: with
    identical trajectories every row's pre-choice context is identical, so the
    cache holds one key for many rows and the collision guard fires instead of
    the round trip being exercised."""

    def __init__(self, n=1):
        self.n = n

    def tasks(self, n, split="test"):
        out = []
        for i in range(min(n, self.n)):
            background = f"SYNTHETIC-TRAJECTORY-{i} do-not-print-me"
            out.append(
                Task(
                    messages=[
                        {"role": "user", "content": f"TRAJECTORY MESSAGE: {background}"},
                        {"role": "user", "content": "BLIND INSTRUCTIONS. End with the answer tag."},
                    ],
                    meta={
                        "task_id": f"row_{i}",
                        "label": "attack" if i % 2 == 0 else "honest",
                        "question": "",
                        "gold": MB_POSITION_ATTACK,
                        "distractor": MB_POSITION_HONEST,
                        "bindings": {"PROBLEM": background},
                    },
                )
            )
        return out

# ScriptedModel pops ONE entry per request, and a protocol step batches every
# state, so a seat's script is (entries for slot 1) * n_tasks + (slot 2) * n ...
def _scripts(n):
    return {
        "alice": [f"Alice blind {i}. <answer>attack</answer>" for i in range(n)]
        + [f"Alice rebuttal {i}." for i in range(n)],
        "bob": [f"Bob opening {i}." for i in range(n)] + [f"Bob rebuttal {i}." for i in range(n)],
        "judge": ['{"winner": "Debater_A", "confidence": 0.8}'] * n,
    }


def _record_cache(env, states):
    """The recorder from scripts/mb_judge_ablation.py, inlined so the test
    exercises the invariant rather than the script's argv handling."""
    out = []
    for st in states:
        if st.failed is not None:
            continue
        full = st.records
        try:
            for i, rec in enumerate(full):
                if rec.slot.speaker == env.judge_speaker:
                    continue
                st.records = full[:i]
                out.append(
                    {
                        "key": context_key(render_context(st, rec.slot, env.prompts)),
                        "speech": rec.text,
                        "thinking": rec.thinking,
                        "slot": f"{rec.slot.speaker}/{rec.slot.slot.name}@{rec.slot.turn}",
                        "task_id": (st.meta.get("task") or {}).get("task_id"),
                    }
                )
        finally:
            st.records = full
    return out


def _env(prompt_file, n_tasks=2):  # noqa: F811
    s = _scripts(n_tasks)
    return DebateEnv(
        choice_config(
            prompt_file,
            ScriptedModel("alice", s["alice"]),
            ScriptedModel("bob", s["bob"]),
            ScriptedModel("judge", s["judge"]),
        ),
        MBTaskSource(n_tasks),
        MonitoringBenchFamily(),
    )


def test_replay_reproduces_the_recorded_debate(prompt_file, tmp_path):  # noqa: F811
    """Record, then replay with playback debaters: every lookup hits, and the
    replayed transcript is identical to the recorded one."""
    rec_env = _env(prompt_file)
    tasks = rec_env.task_source.tasks(2)
    rec_env.rollout(tasks, policy=None, group_size=1)
    entries = _record_cache(rec_env, rec_env.last_states)

    # 2 debaters x 2 slots x 2 tasks; judge slots are never cached.
    assert len(entries) == 8
    assert all(e["slot"].split("/")[0] in ("alice", "bob") for e in entries)
    assert len({e["key"] for e in entries}) == 8  # no context collisions

    cache = tmp_path / "cache.jsonl"
    cache.write_text("".join(json.dumps(e) + "\n" for e in entries))

    play_env = _env(prompt_file)
    for speaker in play_env.debaters:
        play_env.config.frozen_models[speaker] = PlaybackModel(
            alias=f"playback-{speaker}", cache_path=cache
        )
    play_env.rollout(tasks, policy=None, group_size=1)

    for before, after in zip(rec_env.last_states, play_env.last_states):
        assert after.failed is None
        assert [(r.slot.speaker, r.slot.slot.name, r.text) for r in after.records] == [
            (r.slot.speaker, r.slot.slot.name, r.text) for r in before.records
        ]
        # The blind choice drove position binding in both runs — if playback
        # were feeding the wrong speech to the wrong slot, this would drift.
        assert after.meta["chosen_side"] == before.meta["chosen_side"] == "attack"
    assert sum(m.hits for m in play_env.config.frozen_models.values() if isinstance(m, PlaybackModel)) == 8


def test_recording_must_truncate_state_records(prompt_file, tmp_path):  # noqa: F811
    """Rendering a slot's context against the FINISHED state yields a key no
    replay will ever ask for: render_context pulls in every record of the
    current speaker regardless of turn, so the seat would be shown its own
    later speeches. This pins the failure the truncation prevents."""
    env = _env(prompt_file, n_tasks=1)
    tasks = env.task_source.tasks(1)
    env.rollout(tasks, policy=None, group_size=1)
    st = env.last_states[0]

    # alice's turn-1 slot: the last record is her own later speech.
    idx, rec = next(
        (i, r)
        for i, r in enumerate(st.records)
        if r.slot.speaker == "alice" and r.slot.turn == 1
    )
    untruncated = context_key(render_context(st, rec.slot, env.prompts))
    saved = st.records
    st.records = saved[:idx]
    truncated = context_key(render_context(st, rec.slot, env.prompts))
    st.records = saved
    assert untruncated != truncated

    correct = {e["key"] for e in _record_cache(env, env.last_states)}
    assert truncated in correct and untruncated not in correct


def test_cache_miss_raises_rather_than_substituting(prompt_file, tmp_path):  # noqa: F811
    """A miss means a rung perturbed a debater's context. That must abort the
    run — a nearest-match or call-order fallback would turn a broken pairing
    invariant into a silently wrong experiment."""
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"key": context_key([{"role": "user", "content": "x"}]), "speech": "s"}) + "\n")
    model = PlaybackModel(alias="playback", cache_path=cache)

    assert model.predict([[ModelInput(role="user", content="x")]])[0].speech == "s"
    with pytest.raises(PlaybackCacheMiss, match="no recorded speech"):
        model.predict([[ModelInput(role="user", content="a different context")]])

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        PlaybackModel(alias="playback", cache_path=empty)


def test_conflicting_speeches_for_one_context_are_refused(tmp_path):
    """Two rows with byte-identical trajectories render identical pre-choice
    contexts. Keeping the last write would replay one row's speech for both,
    mis-attributing it invisibly — so loading must fail instead."""
    key = context_key([{"role": "user", "content": "same context"}])
    path = tmp_path / "conflict.jsonl"
    path.write_text(
        json.dumps({"key": key, "speech": "row 0 speech", "slot": "alice/blind@0"})
        + "\n"
        + json.dumps({"key": key, "speech": "row 1 speech", "slot": "alice/blind@0"})
        + "\n"
    )
    with pytest.raises(ValueError, match="conflicting speeches"):
        PlaybackModel(alias="playback", cache_path=path)

    # Same context, same speech is harmless — dedup, not a conflict.
    ok = tmp_path / "dup.jsonl"
    ok.write_text((json.dumps({"key": key, "speech": "identical", "slot": "alice/blind@0"}) + "\n") * 2)
    assert len(PlaybackModel(alias="playback", cache_path=ok).entries) == 1


def test_context_key_matches_across_message_and_modelinput_forms(tmp_path):
    """The recorder hashes rendered dicts, the replay hashes ModelInputs. If
    those two disagreed, every lookup would miss."""
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert context_key(msgs) == context_key(
        [ModelInput(role=m["role"], content=m["content"]) for m in msgs]
    )
    # Role and content are separated, so content cannot impersonate a role.
    assert context_key([{"role": "system", "content": "a"}]) != context_key(
        [{"role": "system", "content": "", "x": ""}, {"role": "a", "content": ""}]
    )
