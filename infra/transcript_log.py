"""Rollout transcripts -> wandb: a forever-collection plus per-step samples.

Whenever a wandb run is active (Config.log_transcripts, default true), every
training rollout and every eval pass is captured (Ethan, 2026-08-05):

  - FULL transcripts land in wandb Artifacts under two STABLE collection
    names — ``train-transcripts`` and ``eval-transcripts`` (type
    ``transcripts``). Stable names are the point: every run of the project
    appends versions to the same two collections, so the archive accumulates
    across runs indefinitely. Files inside are namespaced
    ``{run_name}/{split}-step-NNNNN.jsonl``; artifact metadata carries run,
    step, and row count. The jsonl is also left on local disk under
    ``transcripts/`` (best-effort scratch; wandb is the durable copy — pod
    container disks are wiped on stop).

  - N samples per log point go to a ``wandb.Table`` under
    ``{split}/transcript_samples`` for eyeballing in the UI. Sample text is
    truncated to keep the table light; the artifact jsonl never is.

Shapes: DebateEnv retains ``last_states`` (full slot records) — logged in
omniscient order, thinking included; the artifact jsonl for debates is the
docent export (one comprehensive AgentRun per debate). Single-turn envs
retain ``last_rollout_records`` — prompt messages + completion per kept
sample.

MB note: prompts embed the (HF-released) red-team trajectories, so transcript
artifacts belong in a PRIVATE wandb project. The no-trajectory-on-stdout rule
is untouched — nothing here prints content.
"""

from __future__ import annotations

import json
import os
from typing import Any

SAMPLE_ROWS = 4
SAMPLE_CHARS = 25_000  # per-cell truncation for the sample table only


def _truncate(text: str) -> str:
    if len(text) <= SAMPLE_CHARS:
        return text
    return text[:SAMPLE_CHARS] + f"\n... [truncated {len(text) - SAMPLE_CHARS} chars; full text in the transcripts artifact]"


def flatten_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def flatten_debate_state(state) -> str:
    """Omniscient slot order, thinking included — the analyst's view."""
    parts = []
    for rec in state.records:
        cs = rec.slot
        head = f"[{cs.speaker}/{cs.slot.name} turn={cs.turn} vis={cs.slot.visibility.value}]"
        body = f"<think>\n{rec.thinking}\n</think>\n{rec.text}" if rec.thinking else rec.text
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def singleturn_sample_rows(records: list[dict], step: int) -> list[list[Any]]:
    rows = []
    for r in records[:SAMPLE_ROWS]:
        meta = r.get("meta", {})
        task = str(meta.get("task_id") or meta.get("name") or meta.get("question") or r.get("task_index"))
        # Prompt and completion truncate SEPARATELY: an MB prompt (60k tokens
        # of trajectory) must not swallow the budget of the completion, which
        # is the part worth eyeballing in the table.
        transcript = (
            _truncate(flatten_messages(r["messages"]))
            + "\n\n[assistant]\n"
            + _truncate(r["completion"] or "")
        )
        rows.append([step, task[:200], float(r["reward"]), json.dumps(r["info"]), transcript])
    return rows


def debate_sample_rows(states: list, step: int) -> list[list[Any]]:
    rows = []
    for st in states[:SAMPLE_ROWS]:
        meta = getattr(st, "meta", {}) or {}
        task = str(meta.get("task_id") or meta.get("question") or "")
        reward = float(meta.get("reward", 0.0)) if isinstance(meta.get("reward"), (int, float)) else 0.0
        rows.append([step, task[:200], reward, json.dumps({k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}), _truncate(flatten_debate_state(st))])
    return rows


def _write_singleturn_jsonl(records: list[dict], path: str) -> int:
    """Docent-format AgentRuns, matching the debate path: RLVR transcripts
    belong in docent too (Ethan, 2026-08-06), and one artifact format means
    one ingest path. Falls back to raw records only if the docent build fails
    — a lost analysis view is acceptable, a lost archive is not."""
    from infra.envs.debate.docent_export import export_jsonl
    from infra.envs.singleturn_docent import agent_runs

    try:
        export_jsonl(agent_runs(records), path)
        return len(records)
    except Exception as e:
        print(f"[transcripts] single-turn docent build failed ({type(e).__name__}: {e}); writing raw records")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        return len(records)


def log_rollout_transcripts(step: int, env, split: str, run_name: str | None = None, upload: bool = True) -> None:
    """Persist env's most recent rollout: local JSONL ALWAYS (insufficient
    logs are unacceptable — Ethan, 2026-08-14), wandb artifact + table only
    when `upload` and a run is active. Never raises past itself into the
    train loop — the caller wraps, but keep failures here cheap anyway.
    """
    import wandb

    if run_name is None:
        if wandb.run is None:
            return
        run_name = wandb.run.name or wandb.run.id
    states = getattr(env, "last_states", None)
    records = getattr(env, "last_rollout_records", None)

    out_dir = os.path.join("transcripts", run_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}-step-{step:05d}.jsonl")

    if states:
        # Full fidelity via the docent export (omniscient + per-speaker views).
        from infra.envs.debate.docent_export import agent_runs, export_jsonl

        export_jsonl(agent_runs(env), path)
        n_rows = len(states)
        rows = debate_sample_rows(states, step)
    elif records:
        n_rows = _write_singleturn_jsonl(records, path)
        rows = singleturn_sample_rows(records, step)
    else:
        return

    if not (upload and wandb.run is not None):
        return

    artifact = wandb.Artifact(
        f"{split}-transcripts",
        type="transcripts",
        metadata={"run": run_name, "step": step, "rows": n_rows, "kind": "debate" if states else "single_turn"},
    )
    artifact.add_file(path, name=f"{run_name}/{os.path.basename(path)}")
    wandb.run.log_artifact(artifact)

    table = wandb.Table(columns=["step", "task", "reward", "info", "transcript"], data=rows)
    wandb.run.log({f"{split}/transcript_samples": table}, step=step)
