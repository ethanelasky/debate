"""Playback model: replays recorded speeches instead of calling an API.

WHAT IT IS FOR. Ablating the JUDGE prompt is a paired experiment: the judge
prompt has no effect on any debater's context, so every rung of the ladder
should be judging BYTE-IDENTICAL transcripts. Regenerating the speeches per
rung would both pay for them N times and let debater sampling noise sit on top
of every rung-to-rung difference — at n~120 that noise is comparable to the
effects being measured.

So the debate is generated once with real models, every debater slot's
(context -> speech) pair is recorded, and each rung re-runs with the debater
seats bound to this model and only the judge live.

KEYED BY THE RENDERED CONTEXT, not by task id or call order. The key is
sha256 over the exact message list the seat is asked with, which makes the
cache self-validating: if a rung's prompt changes a debater's context by even
one byte, the lookup MISSES and the run fails loudly instead of quietly
comparing rungs whose debaters saw different things. Call-order keying would
have hidden exactly that. A miss is therefore a real finding, not a cache
problem to paper over.

Recording lives in scripts/mb_judge_ablation.py — the cache is derived from a
completed rollout by re-rendering each slot's context against the state
truncated to the records that existed when that slot ran.

    {"key": "<sha256>", "speech": "...", "thinking": "...", "slot": "alice/pre_speech@1", "task_id": "..."}

`slot` and `task_id` are diagnostics for miss reporting only — never keys.

SAFETY: cache files hold speech text about red-team trajectories. They are
run artifacts, not something to print; this module never logs speech content,
and miss reports name slots and hashes only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from infra.models.base import (
    DEFAULT_MAX_NEW_TOKENS,
    Model,
    ModelInput,
    ModelResponse,
    SpeechStructure,
)


def context_key(messages: list[dict[str, str]] | list[ModelInput]) -> str:
    """sha256 over (role, content) pairs. Accepts rendered message dicts or
    ModelInputs so the recorder and the replay hash the same way."""
    parts = []
    for m in messages:
        if isinstance(m, ModelInput):
            role, content = m.role, m.content
        else:
            role, content = m["role"], m["content"]
        role = getattr(role, "api_name", role)
        parts.append(f"{role}\x00{content}")
    return hashlib.sha256("\x01".join(parts).encode("utf-8")).hexdigest()


class PlaybackCacheMiss(KeyError):
    """A context that was never recorded. Means the seat's view diverged from
    the recording run — the one thing a judge-prompt ablation must not do."""


class PlaybackModel(Model):
    """Replays recorded speeches by rendered-context hash.

    Deliberately has no fallback: no nearest-match, no call-order guess, no
    live-model escape hatch. Any of those would convert a broken invariant
    into a silently wrong experiment."""

    def __init__(
        self,
        alias: str,
        cache_path: str | Path,
        is_debater: bool = True,
        **kwargs,
    ):
        super().__init__(alias=alias, is_debater=is_debater)
        self.cache_path = Path(cache_path)
        self.entries: dict[str, dict] = {}
        conflicts: list[str] = []
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                prior = self.entries.get(rec["key"])
                # Two slots sharing a context is fine only if they produced the
                # same speech. If they didn't, the later record would silently
                # overwrite the earlier one and BOTH rows would replay one
                # row's speech — a mis-attribution invisible in the results.
                # (Reachable when two rows carry byte-identical trajectories:
                # their pre-choice contexts are then identical too.)
                if prior is not None and prior["speech"] != rec["speech"]:
                    conflicts.append(f"{rec['key'][:16]}... {prior.get('slot')} vs {rec.get('slot')}")
                self.entries[rec["key"]] = rec
        if conflicts:
            raise ValueError(
                f"playback cache {self.cache_path}: {len(conflicts)} context(s) recorded "
                f"with conflicting speeches — replay would mis-attribute them. "
                f"Usually means two rows have byte-identical trajectories. "
                f"First: {conflicts[0]}"
            )
        if not self.entries:
            raise ValueError(f"playback cache {self.cache_path} is empty")
        self.hits = 0

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_return_sequences: int = 1,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        **kwargs,
    ) -> list[ModelResponse]:
        out: list[ModelResponse] = []
        for conversation in inputs:
            key = context_key(conversation)
            rec = self.entries.get(key)
            if rec is None:
                raise PlaybackCacheMiss(
                    f"{self.alias}: no recorded speech for this context "
                    f"(sha256 {key[:16]}..., {len(conversation)} messages, "
                    f"{sum(len(m.content) for m in conversation)} chars). The seat's "
                    f"view differs from the recording run — for a judge-prompt "
                    f"ablation that means a rung changed a DEBATER's context, which "
                    f"invalidates the comparison. Cache: {self.cache_path} "
                    f"({len(self.entries)} entries)."
                )
            self.hits += 1
            for _ in range(num_return_sequences):
                out.append(
                    ModelResponse(
                        speech=rec["speech"],
                        thinking=rec.get("thinking"),
                        prompt="\n".join(m.content for m in conversation),
                    )
                )
        return out

    def copy(
        self, alias: Optional[str] = None, is_debater: Optional[bool] = None, **kwargs
    ) -> "PlaybackModel":
        return PlaybackModel(
            alias=alias or self.alias,
            cache_path=self.cache_path,
            is_debater=self.is_debater if is_debater is None else is_debater,
        )
