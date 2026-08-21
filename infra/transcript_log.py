"""Rollout transcripts -> required local JSONL plus best-effort W&B copies.

With ``Config.log_transcripts`` (default true), every training rollout and
every eval pass is captured (Ethan, 2026-08-05):

  - FULL transcripts are success-gating local evidence under the manual path
    ``transcripts/{safe_run_name}/{launch_namespace}/`` or, when the scheduler
    supplies its retained root, under
    ``{artifact_root}/{launch_namespace}/transcripts/{safe_run_name}/``. When W&B is active,
    best-effort copies land in Artifacts under a STABLE collection name per
    split — ``{split}-transcripts`` (type ``transcripts``), including the
    train/dev/test/eval split names used by callers. Stable names are the
    point: every run of the project appends versions to the corresponding
    split collection, so the archive accumulates across runs indefinitely.
    Files inside are namespaced
    ``{safe_run_name}/{launch_namespace}/{split}-step-NNNNN.jsonl``; artifact
    metadata carries the scalar launch namespace, run, step, and row count.
    The same namespaced jsonl is also left on local disk under ``transcripts/``.

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

Filesystem boundary: local creation uses a retained directory descriptor and
never follows a replacement pathname. W&B's public Artifact API accepts only a
pathname, however, so configured transcript roots must remain owner-trusted and
stable through ``artifact.add_file``. The immediately preceding identity check
detects an observed replacement; it is not authority against a hostile rename
after that check.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from infra.launch_namespace import (
    claim_directory,
    open_claimed_text_file,
    require_claimed_directory,
    resolve_launch_namespace,
    safe_path_component,
    validate_launch_namespace,
)
from infra.local_metrics_evidence import scheduler_artifact_attempt_root

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


def _write_jsonl_lines(output_dir: str, filename: str, lines: list[str]) -> None:
    with open_claimed_text_file(output_dir, filename) as f:
        for line in lines:
            f.write(line + "\n")


def _write_singleturn_jsonl(
    records: list[dict], output_dir: str, filename: str
) -> int:
    """Docent-format AgentRuns, matching the debate path: RLVR transcripts
    belong in docent too (Ethan, 2026-08-06), and one artifact format means
    one ingest path. Falls back to raw records only if the docent build fails
    — a lost analysis view is acceptable, a lost archive is not."""
    from infra.envs.singleturn_docent import agent_runs

    try:
        lines = [run.model_dump_json() for run in agent_runs(records)]
    except Exception as e:
        print(f"[transcripts] single-turn docent build failed ({type(e).__name__}: {e}); writing raw records")
        lines = [json.dumps(record, default=str) for record in records]
    _write_jsonl_lines(output_dir, filename, lines)
    return len(records)


def log_rollout_transcripts(
    step: int,
    env,
    split: str,
    run_name: str | None = None,
    launch_namespace: str | None = None,
    output_dir: str | None = None,
    upload: bool = True,
) -> None:
    """Persist env's most recent rollout to its required local JSONL.

    Local serialization, reservation and write failures propagate so an
    analyzed workload cannot report success without transcript evidence.
    W&B artifact/table work is a separate best-effort phase: it is attempted
    only after the local file and directory authority are verified.
    """
    wandb_module = None
    if run_name is None:
        # Low-level compatibility path only. Production callers always pass
        # the pre-resolved operational run component explicitly.
        import wandb as wandb_module

        if wandb_module.run is None:
            return
        run_name = wandb_module.run.name or wandb_module.run.id
    namespace = (
        resolve_launch_namespace()
        if launch_namespace is None
        else validate_launch_namespace(launch_namespace)
    )
    states = getattr(env, "last_states", None)
    records = getattr(env, "last_rollout_records", None)
    if not states and not records:
        raise RuntimeError(
            "requested rollout retained no transcript records; refusing an "
            "analyzed workload with missing local evidence"
        )

    safe_run_name = safe_path_component(run_name, fallback="run")
    artifact_attempt_root = scheduler_artifact_attempt_root(namespace)
    expected_dir = (
        os.path.join(artifact_attempt_root, "transcripts", safe_run_name)
        if artifact_attempt_root is not None
        else os.path.join("transcripts", safe_run_name, namespace)
    )
    if output_dir is None:
        out_dir = str(claim_directory(expected_dir))
    else:
        if os.path.normpath(output_dir) != os.path.normpath(expected_dir):
            raise ValueError(
                "transcript output_dir does not match run_name and "
                f"launch_namespace: expected {expected_dir}, got {output_dir}"
            )
        out_dir = output_dir
    filename = f"{split}-step-{step:05d}.jsonl"
    path = os.path.join(out_dir, filename)

    if states:
        # Full fidelity via the docent export (omniscient + per-speaker views).
        from infra.envs.debate.docent_export import agent_runs

        lines = [run.model_dump_json() for run in agent_runs(env)]
        _write_jsonl_lines(out_dir, filename, lines)
        n_rows = len(states)
        rows = debate_sample_rows(states, step)
    elif records:
        n_rows = _write_singleturn_jsonl(records, out_dir, filename)
        rows = singleturn_sample_rows(records, step)
    # W&B still consumes a pathname. Refuse an already-observed replacement
    # before handing it off; local creation itself used retained dirfd authority.
    # The configured artifact root's trusted-stability contract extends through
    # add_file because the W&B API has no descriptor-based ingestion boundary.
    require_claimed_directory(out_dir)
    if not upload:
        return
    try:
        if wandb_module is None:
            import wandb as wandb_module
        if wandb_module.run is None:
            return
        artifact = wandb_module.Artifact(
            f"{split}-transcripts",
            type="transcripts",
            metadata={
                "run": run_name,
                "launch_namespace": namespace,
                "step": step,
                "rows": n_rows,
                "kind": "debate" if states else "single_turn",
            },
        )
        artifact.add_file(
            path,
            name=f"{safe_run_name}/{namespace}/{os.path.basename(path)}",
        )
        wandb_module.run.log_artifact(artifact)

        table = wandb_module.Table(
            columns=["step", "task", "reward", "info", "transcript"],
            data=rows,
        )
        wandb_module.run.log({f"{split}/transcript_samples": table}, step=step)
    except Exception as exc:
        print(
            f"[transcripts] W&B upload failed: {type(exc).__name__}; "
            "local JSONL remains authoritative",
            file=sys.stderr,
        )
