"""Single-turn (RLVR) rollouts -> Docent AgentRuns.

The single-turn twin of infra/envs/debate/docent_export.py: one AgentRun per
kept sample from ``SingleTurnEnv.last_rollout_records`` — the exact prompt
messages, the completion, and grading metadata. Reuses the debate module's
``export_jsonl``/``upload`` for output, so RLVR and debate transcripts land in
docent the same way.

Usage (mirrors the debate export):
    runs = agent_runs(env.last_rollout_records)
    export_jsonl(runs, "docent/step-00042.jsonl")
"""

from __future__ import annotations

from typing import Any

from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import AssistantMessage, SystemMessage, UserMessage

_ROLE = {"system": SystemMessage, "user": UserMessage, "assistant": AssistantMessage}


def _msg(m: dict[str, str]):
    return _ROLE[m["role"]](content=m["content"])


def agent_runs(records: list[dict[str, Any]]) -> list[AgentRun]:
    runs = []
    for rec in records:
        info = rec.get("info") or {}
        meta = rec.get("meta") or {}
        correct = info.get("correct_relaxed", info.get("correct"))
        status = (
            "correct" if correct == 1.0 else "incorrect" if correct == 0.0 else "ungraded"
        )
        messages = [_msg(m) for m in rec["messages"]]
        messages.append(AssistantMessage(content=rec.get("completion") or ""))
        runs.append(
            AgentRun(
                name=f"task-{rec.get('task_index')} [{meta.get('split', '?')}] -> {status}",
                transcripts=[Transcript(messages=messages)],
                metadata={
                    "task": meta,
                    "reward": rec.get("reward"),
                    "info": info,
                    "stop_reason": rec.get("stop_reason"),
                    "task_index": rec.get("task_index"),
                },
            )
        )
    return runs
