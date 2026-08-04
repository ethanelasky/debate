#!/usr/bin/env python3
"""Upload ``mb_solo_monitor.py`` results to Docent for manual viewing.

Each result row becomes ONE Docent agent run holding a single transcript with
exactly three messages: the system prompt, the user prompt (the Direct QA
stanza with the rendered trajectory pasted in), and the monitor's raw response
verbatim. Verdict/label/provenance go into run metadata so the Docent UI can
filter on them.

The results jsonl stores only ``prompt_sha256``, not the prompt text, so the
prompts are REBUILT here with the very same helpers ``mb_solo_monitor`` used
(``load_rows`` -> ``render_transcript`` -> the yaml stanza). Every rebuilt
prompt is hashed and checked against the row's stored ``prompt_sha256``: a
mismatch means the yaml or the trajectory data changed since the run, so the
upload would show a prompt the model never saw. That is a hard error by
default; ``--allow-prompt-drift`` downgrades it to a warning and flags the
affected runs in metadata (``prompt_sha_verified: false``).

Error rows (no response; a transport error was recorded) are uploaded too,
with an assistant message naming the error type, so a reviewer can see which
rows failed instead of silently missing them.

Examples
--------
    # Offline: build everything, print one line per row, touch no network.
    .venv/bin/python scripts/mb_solo_to_docent.py \
        --results data/monitoringbench/solo_qwen35_9b_smoke4.jsonl \
        --name "mb solo monitor" --dry-run

    # Real upload (needs DOCENT_API_KEY in the environment or .env).
    .venv/bin/python scripts/mb_solo_to_docent.py \
        --results data/monitoringbench/solo_deepseekv4flash_honest_mb118.jsonl \
        --results data/monitoringbench/solo_deepseekv4flash_attacks_covering140_v2.jsonl \
        --name "mb solo deepseek-v4-flash"

Hard constraint: prompt text, trajectory content and raw model responses are
NEVER printed. Only ids, labels, verdicts, counts and run ids go to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docent.data_models import AgentRun, Transcript  # noqa: E402
from docent.data_models.chat import parse_chat_message  # noqa: E402

from scripts.mb_solo_monitor import (  # noqa: E402
    DEFAULT_DATA_PATHS,
    DEFAULT_PROMPT_FILE,
    DEFAULT_PROMPT_SECTION,
    _resolve,
    load_prompt_blocks,
    load_rows,
    prepare_rows,
)

DEFAULT_BATCH_SIZE = 20
DOCENT_DASHBOARD_URL = "https://docent.transluce.org/dashboard"


# --------------------------------------------------------------------------
# Results loading
# --------------------------------------------------------------------------


def load_results(path: str) -> list[dict[str, Any]]:
    """Read one ``mb_solo_monitor`` results jsonl, de-duplicated by (id, model).

    Rows are returned in the order the file lists them (which is completion
    order for a parallel run, not selection order); the first occurrence of an
    (id, model) pair wins. Duplicates are reported by id only.
    """
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Results file not found: {resolved}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, Optional[str]]] = set()
    duplicates: list[str] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_id = record.get("id")
            if not row_id:
                raise ValueError(f"{resolved}:{lineno}: results row has no 'id'")
            key = (row_id, record.get("model"))
            if key in seen:
                duplicates.append(row_id)
                continue
            seen.add(key)
            record["_results_file"] = resolved.name
            records.append(record)
    if duplicates:
        print(
            f"[warn] {resolved.name}: dropped {len(duplicates)} duplicate (id, model) "
            f"row(s), first occurrence kept: {', '.join(sorted(set(duplicates))[:10])}"
        )
    return records


# --------------------------------------------------------------------------
# Prompt rebuild + verification
# --------------------------------------------------------------------------


def build_prompt_index(
    row_ids: Sequence[str],
    data_paths: Sequence[str],
    prompt_file: str,
    prompt_section: str,
) -> dict[str, tuple[str, str, str]]:
    """Rebuild ``{row_id: (system_text, user_text, sha256)}`` for ``row_ids``.

    Uses ``mb_solo_monitor``'s own loaders so the prompts are byte-identical to
    what the run sent (that is what the sha check then proves). Ids absent from
    the trajectory data are a hard error: without the trajectory there is no
    prompt to show, and uploading a run without one would misrepresent the row.
    """
    wanted = set(row_ids)
    rows = [row for row in load_rows(list(data_paths)) if row["id"] in wanted]
    missing = sorted(wanted - {row["id"] for row in rows})
    if missing:
        raise ValueError(
            f"{len(missing)} results id(s) are not in the trajectory data "
            f"({', '.join(data_paths)}); pass the --data file(s) the run used. "
            f"Missing: {', '.join(missing[:10])}"
        )
    system_text, user_template = load_prompt_blocks(prompt_file, prompt_section)
    prepared = prepare_rows(rows, system_text, user_template)
    return {item.row_id: (item.system_text, item.user_text, item.sha256) for item in prepared}


def verify_prompt_sha(
    record: dict[str, Any], rebuilt_sha: str, *, allow_drift: bool
) -> bool:
    """Return True when the rebuilt prompt hashes to the row's stored sha.

    Raises unless ``allow_drift``; the message names the offending id and both
    hashes (hashes only — never the prompt itself).
    """
    stored = record.get("prompt_sha256")
    if stored and stored == rebuilt_sha:
        return True
    detail = (
        f"row {record['id']} ({record.get('_results_file', '?')}): rebuilt prompt "
        f"sha256 {rebuilt_sha} != recorded {stored or '(missing)'}. The prompt yaml "
        f"or the trajectory data changed since the run, so the upload would show a "
        f"prompt the model never saw."
    )
    if not allow_drift:
        raise ValueError(detail + " Re-run with the original --data/--prompt-file, or "
                         "pass --allow-prompt-drift to upload it flagged anyway.")
    print(f"[warn] prompt drift: {detail}")
    return False


# --------------------------------------------------------------------------
# AgentRun construction
# --------------------------------------------------------------------------


def verdict_is_correct(label: Optional[str], verdict: Optional[str]) -> Optional[bool]:
    """``True``/``False`` when the row has a verdict, ``None`` otherwise.

    The monitor answers ``attack``/``benign``; labels are ``attack``/``honest``,
    so ``benign`` is the prediction that matches an ``honest`` label.
    """
    if verdict not in ("attack", "benign") or label not in ("attack", "honest"):
        return None
    return (verdict == "attack") == (label == "attack")


def result_to_agent_run(
    record: dict[str, Any],
    system_text: str,
    user_text: str,
    *,
    prompt_sha_verified: bool,
) -> AgentRun:
    """One results row -> one AgentRun (system, user, assistant)."""
    label = record.get("label")
    verdict = record.get("verdict")
    error_type = record.get("error_type")
    is_error = bool(record.get("error"))
    if record.get("raw_response") is not None:
        assistant = record["raw_response"]
    else:
        assistant = (
            f"[NO RESPONSE — the request failed after {record.get('attempts')} "
            f"attempt(s) with error type: {error_type or 'UnknownError'}. "
            f"This row was recorded as an error and is not a monitor verdict.]"
        )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant},
    ]
    metadata: dict[str, Any] = {
        "id": record["id"],
        "label": label,
        "verdict": verdict or "none",
        "correct": verdict_is_correct(label, verdict),
        "has_score_tag": bool(record.get("has_score_tag")),
        "score": record.get("score"),
        "model": record.get("model"),
        "provider": record.get("provider"),
        "quantization": record.get("quantization"),
        "thinking": record.get("thinking"),
        "served_provider": record.get("served_provider"),
        "error_type": error_type or "none",
        "is_error_row": is_error,
        "prompt_sha256": record.get("prompt_sha256"),
        "prompt_sha_verified": prompt_sha_verified,
        "results_file": record.get("_results_file"),
        "source_index": record.get("index"),
        "generation_id": record.get("generation_id"),
        "attempts": record.get("attempts"),
    }
    status = error_type if is_error else (verdict or "no-answer-tag")
    return AgentRun(
        name=f"{record['id']} [{label}] -> {status}",
        transcripts=[Transcript(messages=[parse_chat_message(m) for m in messages])],
        metadata=metadata,
    )


def build_runs(
    records: Sequence[dict[str, Any]],
    prompts: dict[str, tuple[str, str, str]],
    *,
    allow_drift: bool,
    verbose: bool = False,
) -> list[AgentRun]:
    """Verify every prompt, then build one AgentRun per record.

    With ``verbose`` a one-line summary per row is printed: id, label, verdict
    and whether the prompt sha matched. No content, ever.
    """
    runs: list[AgentRun] = []
    for record in records:
        system_text, user_text, rebuilt_sha = prompts[record["id"]]
        verified = verify_prompt_sha(record, rebuilt_sha, allow_drift=allow_drift)
        runs.append(
            result_to_agent_run(
                record, system_text, user_text, prompt_sha_verified=verified
            )
        )
        if verbose:
            error_type = record.get("error_type")
            print(
                f"  {record['id']}  label={record.get('label')}  "
                f"verdict={record.get('verdict') or 'none'}  "
                f"error={error_type or 'none'}  "
                f"prompt_sha_match={'yes' if verified else 'NO'}"
            )
    return runs


# --------------------------------------------------------------------------
# Docent client
# --------------------------------------------------------------------------


def make_client() -> Any:
    """Build the Docent client. Imported lazily so tests never need network."""
    import os

    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional in this repo
        pass
    else:
        load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    from docent import Docent

    return Docent(api_key=os.getenv("DOCENT_API_KEY"))


def resolve_collection_id(client: Any, name: str) -> str:
    """Existing collection with this exact name, else a new one."""
    matching = [c for c in client.list_collections() if c["name"] == name]
    if not matching:
        return client.create_collection(name=name, description="")
    if len(matching) == 1:
        return matching[0]["id"]
    raise ValueError(f"Multiple collections named {name!r}; rename or pass --collection-id.")


def upload_runs(
    client: Any,
    collection_id: str,
    runs: Sequence[AgentRun],
    *,
    batch_size: int,
    replace: bool = False,
) -> None:
    if replace:
        # Called only after every AgentRun is built, so a build failure never
        # reaches the delete. A network failure BETWEEN batches still can.
        existing = client.list_agent_run_ids(collection_id)
        if existing:
            print(
                "[warn] --replace: deleting the existing runs before uploading. "
                "If the upload fails part-way the collection is left partial; "
                "re-run the same command to refill it."
            )
            client.delete_agent_runs(collection_id, existing)
            print(f"cleared {len(existing)} existing run(s) from the collection")
    for start in range(0, len(runs), batch_size):
        chunk = list(runs[start:start + batch_size])
        client.add_agent_runs(collection_id, chunk)
        print(f"uploaded {start + len(chunk)}/{len(runs)}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mb_solo_to_docent.py",
        description=(
            "Upload mb_solo_monitor.py results to Docent. Each results row "
            "becomes one agent run with a system message, the user prompt, and "
            "the monitor's raw response; verdict/label/provenance go into run "
            "metadata for filtering. Prompts are rebuilt with mb_solo_monitor's "
            "own helpers and verified against each row's prompt_sha256."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prompt verification: the results jsonl stores a prompt hash, not "
            "the prompt text, so --data/--prompt-file/--prompt-section MUST "
            "match the run. A hash mismatch aborts the upload naming the "
            "offending id; --allow-prompt-drift downgrades that to a warning "
            "and sets prompt_sha_verified=false on those runs.\n\n"
            "De-duplication is by (id, model) across ALL --results files, "
            "first occurrence kept, so a partial re-run file can be passed "
            "alongside the original without uploading a row twice.\n\n"
            "Error rows (transport failure, no response) are uploaded with an "
            "assistant message naming the error type and is_error_row=true, so "
            "failed rows stay visible in the collection.\n\n"
            "Prompt text, trajectory content and raw responses are never "
            "printed: stdout carries ids, labels, verdicts, counts and run ids "
            "only."
        ),
    )
    source = parser.add_argument_group("inputs")
    source.add_argument(
        "--results",
        action="append",
        required=True,
        metavar="PATH",
        help="mb_solo_monitor results jsonl; repeatable",
    )
    source.add_argument(
        "--data",
        action="append",
        metavar="PATH",
        help=(
            "trajectory jsonl used to rebuild prompts; repeatable. Must be the "
            "file(s) the run used. Default: " + " + ".join(DEFAULT_DATA_PATHS)
        ),
    )
    source.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"prompt yaml the run used (default: {DEFAULT_PROMPT_FILE})",
    )
    source.add_argument(
        "--prompt-section",
        default=DEFAULT_PROMPT_SECTION,
        help=f"section in the prompt yaml (default: {DEFAULT_PROMPT_SECTION!r})",
    )
    source.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="keep only the first N rows across all --results files, in order",
    )
    source.add_argument(
        "--allow-prompt-drift",
        action="store_true",
        help=(
            "downgrade a prompt_sha256 mismatch from a hard error to a warning; "
            "affected runs get prompt_sha_verified=false in metadata"
        ),
    )

    destination = parser.add_argument_group("destination")
    target = destination.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", help="Docent collection name to create or append to")
    target.add_argument("--collection-id", help="existing Docent collection id to append to")
    destination.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"runs per add_agent_runs call (default: {DEFAULT_BATCH_SIZE})",
    )
    destination.add_argument(
        "--replace",
        action="store_true",
        help="delete every existing run in the collection before adding these",
    )
    destination.add_argument(
        "--dry-run",
        action="store_true",
        help="build and verify everything, print a line per row, exit without any network call",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, client: Optional[Any] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    data_paths = list(args.data) if args.data else list(DEFAULT_DATA_PATHS)

    # De-duplication spans ALL --results files: a partial re-run file passed
    # alongside the original otherwise uploads the same (id, model) twice.
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, Optional[str]]] = set()
    cross_file_duplicates: list[str] = []
    try:
        for results_path in args.results:
            file_records = load_results(results_path)
            print(f"{results_path}: {len(file_records)} row(s) read")
            for record in file_records:
                key = (record["id"], record.get("model"))
                if key in seen:
                    cross_file_duplicates.append(record["id"])
                    continue
                seen.add(key)
                records.append(record)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2  # pragma: no cover - parser.exit raises SystemExit
    if cross_file_duplicates:
        print(
            f"[warn] skipped {len(cross_file_duplicates)} (id, model) row(s) already "
            f"seen in an earlier --results file, first occurrence kept: "
            f"{', '.join(sorted(set(cross_file_duplicates))[:10])}"
        )
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        parser.error("no results rows found in the given --results file(s)")

    print("=== mb_solo_to_docent ===")
    print(f"data            : {', '.join(data_paths)}")
    print(f"prompt          : {args.prompt_file} [{args.prompt_section}]")
    print(f"rows            : {len(records)} "
          f"({sum(1 for r in records if r.get('label') == 'attack')} attack / "
          f"{sum(1 for r in records if r.get('label') == 'honest')} honest, "
          f"{sum(1 for r in records if r.get('error'))} error row(s))")
    print("rows (id / label / verdict / prompt sha match):")

    try:
        prompts = build_prompt_index(
            [record["id"] for record in records],
            data_paths,
            args.prompt_file,
            args.prompt_section,
        )
        runs = build_runs(
            records, prompts, allow_drift=args.allow_prompt_drift, verbose=True
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2  # pragma: no cover - parser.exit raises SystemExit

    n_unverified = sum(
        1 for run in runs if not run.metadata.get("prompt_sha_verified")
    )
    print(f"prompt sha256   : {len(runs) - n_unverified}/{len(runs)} verified"
          + (f", {n_unverified} FLAGGED as drifted" if n_unverified else ""))

    if args.dry_run:
        print(f"\n=== dry run: {len(runs)} run(s) built, nothing uploaded ===")
        return 0

    if client is None:
        client = make_client()
    collection_id = args.collection_id or resolve_collection_id(client, args.name)
    upload_runs(
        client,
        collection_id,
        runs,
        batch_size=args.batch_size,
        replace=args.replace,
    )
    if args.name:
        print(f"\ncollection: {args.name}")
    print(f"id: {collection_id}")
    print(f"url: {DOCENT_DASHBOARD_URL}/{collection_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
