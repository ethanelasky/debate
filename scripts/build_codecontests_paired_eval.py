#!/usr/bin/env python3
"""Build the self-contained paired CodeContests held-out evaluation artifact.

The runtime must not join test suites from two files: a missing or stale
sidecar can silently change which problems are evaluated.  This builder joins
the deterministic <=10-case DeepMind reward suite already stored in
``test.jsonl`` with the deterministic <=10-case CodeContests-O extraction,
then writes one row per problem carrying both named suites.

Usage:
    python scripts/build_codecontests_paired_eval.py \
        --gdm-test data/codecontests/test.jsonl \
        --cco-eval data/codecontests/cco_eval.jsonl \
        --out data/codecontests/paired_test.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_codecontests_rlvr import (
    HF_REPO as GDM_REPO,
    HF_REVISION as GDM_REVISION,
    MAX_RLVR_TESTS,
    MAX_RLVR_TOTAL_BYTES,
    MAX_SINGLE_TEST_BYTES,
)
from scripts.fetch_cco_eval import (
    HF_REPO as CCO_REPO,
    HF_REVISION as CCO_REVISION,
    MAX_CCO_TESTS,
    MAX_SINGLE_TEST_BYTES as MAX_CCO_SINGLE_TEST_BYTES,
    MAX_TOTAL_BYTES as MAX_CCO_TOTAL_BYTES,
    SEED as CCO_SEED,
)

STABLE_META_FIELDS = (
    "name",
    "problem",
    "cf_contest_id",
    "cf_index",
    "cf_rating",
    "difficulty",
    "source",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_paired_rows(
    gdm_rows: list[dict[str, Any]], cco_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join by stripped problem name, preserving the GDM held-out order."""
    cco_by_name: dict[str, dict[str, Any]] = {}
    for row in cco_rows:
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("CCO row has no stable problem name")
        if name in cco_by_name:
            raise ValueError(f"duplicate CCO problem name: {name!r}")
        ci, co = row.get("cco_inputs") or [], row.get("cco_outputs") or []
        if not ci or len(ci) != len(co):
            raise ValueError(f"malformed CCO suite for {name!r}")
        cco_by_name[name] = {"cco_inputs": list(ci), "cco_outputs": list(co)}

    seen_gdm: set[str] = set()
    paired: list[dict[str, Any]] = []
    for row in gdm_rows:
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("GDM row has no stable problem name")
        if name in seen_gdm:
            raise ValueError(f"duplicate GDM problem name: {name!r}")
        seen_gdm.add(name)
        if name not in cco_by_name:
            continue
        problem = str(row.get("problem", "")).strip()
        rating = row.get("cf_rating")
        if not problem:
            raise ValueError(f"GDM row has no problem statement: {name!r}")
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise ValueError(f"GDM row has no numeric cf_rating: {name!r}")
        gi, go = row.get("rlvr_inputs") or [], row.get("rlvr_outputs") or []
        if not gi or len(gi) != len(go):
            raise ValueError(f"malformed GDM suite for {name!r}")
        out = {field: row.get(field) for field in STABLE_META_FIELDS}
        out["name"] = name
        out.update(
            {
                "gdm_inputs": list(gi),
                "gdm_outputs": list(go),
                **cco_by_name[name],
            }
        )
        paired.append(out)
    return paired


def _slice_count(rows: list[dict[str, Any]], lo: int, hi: int) -> int:
    return sum(
        isinstance(row.get("cf_rating"), (int, float))
        and not isinstance(row.get("cf_rating"), bool)
        and lo <= row["cf_rating"] <= hi
        for row in rows
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gdm-test", default="data/codecontests/test.jsonl")
    ap.add_argument("--cco-eval", default="data/codecontests/cco_eval.jsonl")
    ap.add_argument("--out", default="data/codecontests/paired_test.jsonl")
    args = ap.parse_args()

    gdm_path = Path(args.gdm_test)
    cco_path = Path(args.cco_eval)
    out_path = Path(args.out)
    gdm_rows = _read_jsonl(gdm_path)
    cco_rows = _read_jsonl(cco_path)
    rows = build_paired_rows(gdm_rows, cco_rows)
    if not rows:
        raise SystemExit("FATAL: GDM/CCO intersection is empty")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ratings = Counter(str(row.get("cf_rating")) for row in rows)
    manifest = {
        "schema": "codecontests_paired_eval/v1",
        "join": "stripped name; GDM held-out order; intersection only",
        "sources": {
            "gdm": {
                "repo": GDM_REPO,
                "revision": GDM_REVISION,
                "artifact": str(gdm_path),
                "sha256": _sha256(gdm_path),
                "fields": ["rlvr_inputs", "rlvr_outputs"],
            },
            "cco": {
                "repo": CCO_REPO,
                "revision": CCO_REVISION,
                "artifact": str(cco_path),
                "sha256": _sha256(cco_path),
                "fields": ["cco_inputs", "cco_outputs"],
            },
        },
        "suites": {
            "gdm": {
                "max_tests": MAX_RLVR_TESTS,
                "max_single_test_bytes": MAX_SINGLE_TEST_BYTES,
                "max_total_bytes": MAX_RLVR_TOTAL_BYTES,
                "selection": "seeded rlvr sample from public_tests+private_tests",
            },
            "cco": {
                "max_tests": MAX_CCO_TESTS,
                "max_single_test_bytes": MAX_CCO_SINGLE_TEST_BYTES,
                "max_total_bytes": MAX_CCO_TOTAL_BYTES,
                "seed": CCO_SEED,
                "selection": "seeded sample from CodeContests-O corner_cases",
            },
        },
        "counts": {
            "gdm_heldout_problems": len(gdm_rows),
            "cco_staging_problems": len(cco_rows),
            "paired_problems": len(rows),
            "gdm_without_cco": len(gdm_rows) - len(rows),
            "gdm_cases": sum(len(row["gdm_inputs"]) for row in rows),
            "cco_cases": sum(len(row["cco_inputs"]) for row in rows),
            "cf_rating_800_1000": _slice_count(rows, 800, 1000),
            "cf_rating_800_1100": _slice_count(rows, 800, 1100),
            "by_cf_rating": dict(sorted(ratings.items(), key=lambda item: int(item[0]))),
        },
        "output": {
            "file": out_path.name,
            "rows": len(rows),
            "size_bytes": out_path.stat().st_size,
            "sha256": _sha256(out_path),
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
