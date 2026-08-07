#!/usr/bin/env python3
"""Build the CodeContests RLVR dataset from DeepMind's CodeContests.

Pulls ONLY the projected columns from `deepmind/code_contests` at a pinned
revision — the 95% of that dataset which is `solutions` / `incorrect_solutions`
is never transferred. Emits one JSONL per split plus a manifest.

Two independent suites per problem, kept separate on purpose:

  rlvr_tests   a seeded random sample of <=MAX_RLVR_TESTS drawn from
               public_tests + private_tests. This is the RLVR arm's REWARD
               signal. It is deliberately small and therefore weak: the median
               problem has only ~4 public+private cases, and 40% have no
               private tests at all.

  truth_tests  everything from public+private that the sample did NOT take.
               Disjoint from rlvr_tests by construction, so proposer accuracy
               is never measured on a case the RLVR arm optimized against.
               OFTEN EMPTY, and that is fine: the median problem has only ~4
               public+private cases, so a <=10 sample usually consumes them
               all. Requiring a non-empty remainder dropped 30% of problems --
               the typical ones -- which is why it is not required.

The debate arm uses NEITHER at training time (its reward is judge-only).

The HELD-OUT split additionally carries two eval suites, which exist only
there and are never touched during training:

  cc_eval      every public+private case for that problem, not the <=10
               sample. On a held-out problem there is no reward suite to stay
               disjoint from, so nothing has to be held back. This is the
               in-distribution eval: the same KIND of test the RLVR arm
               trains on, on problems it never saw.

  cco_eval     CodeContests-O corner cases (~34 per problem, inputs at the
               problem's real constraint limits rather than DeepMind's small
               mutations). Much stronger, and it covers 460 of the 501
               held-out problems. Built by scripts/fetch_cco_eval.py, which
               reads only the columns it needs so the 324 GB of CCO never
               moves. Both suites are graded in-run, giving two overlayable
               accuracy curves -- see docs/codecontests-dataset-provenance.md.

Usage:
    python scripts/build_codecontests_rlvr.py --out data/codecontests
    python scripts/build_codecontests_rlvr.py --out /tmp/cc --limit-shards 2   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.envs.tasks.codecontests import _MULTI_ANSWER_PHRASES  # verbatim from the old loader

# ---------------------------------------------------------------- provenance
HF_REPO = "deepmind/code_contests"
HF_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"

# Columns we read. Everything else — solutions, incorrect_solutions,
# generated_tests — is never fetched.
PROJECTED = [
    "name", "description", "public_tests", "private_tests",
    "cf_contest_id", "cf_index", "cf_rating", "difficulty",
    "source", "time_limit", "memory_limit_bytes", "input_file", "output_file",
]

# ------------------------------------------------------------------- limits
MAX_RLVR_TESTS = 10          # cases in the RLVR reward suite
MAX_SINGLE_TEST_BYTES = 500_000    # per case, either side (old repo's value)
MAX_RLVR_TOTAL_BYTES = 2_000_000   # whole reward suite: 10x500KB through a
                                   # subprocess on every rollout is not viable
MIN_TOTAL_TESTS = 1          # >=1 case for the reward suite; a held-out
                             # remainder is a bonus, not a requirement


def _pairs(field) -> list[tuple[str, str]]:
    """A CodeContests test field -> [(input, output)]."""
    return [(i, o) for i, o in zip(field["input"], field["output"]) if i is not None and o is not None]


def _oversized(pairs) -> bool:
    return any(max(len(i), len(o)) > MAX_SINGLE_TEST_BYTES for i, o in pairs)


def _is_multi_answer(description: str) -> bool:
    d = (description or "").lower()
    return any(p in d for p in _MULTI_ANSWER_PHRASES)


def build_row(row, rng: random.Random) -> tuple[dict | None, str | None]:
    """-> (row, None) if admitted, else (None, rejection_reason)."""
    name = (row.get("name") or "").strip()
    desc = (row.get("description") or "").strip()
    if not name or not desc:
        return None, "invalid_task_fields"
    # stdin/stdout only: file-based I/O problems can't be piped
    if (row.get("input_file") or "") or (row.get("output_file") or ""):
        return None, "not_stdin_stdout"
    if _is_multi_answer(desc):
        return None, "multi_answer"

    pool = _pairs(row["public_tests"]) + _pairs(row["private_tests"])
    if len(pool) < MIN_TOTAL_TESTS:
        return None, "too_few_tests"
    if _oversized(pool):
        return None, "single_test_too_large"

    # Seeded draw for the reward suite; remainder is held-out ground truth.
    order = list(range(len(pool)))
    rng.shuffle(order)
    rlvr_idx, total = [], 0
    for j in order:
        if len(rlvr_idx) >= MAX_RLVR_TESTS:
            break
        i, o = pool[j]
        if total + len(i) + len(o) > MAX_RLVR_TOTAL_BYTES:
            continue
        rlvr_idx.append(j)
        total += len(i) + len(o)
    if not rlvr_idx:
        return None, "rlvr_suite_empty"
    truth_idx = [j for j in range(len(pool)) if j not in set(rlvr_idx)]

    return {
        "name": name,
        "problem": desc,
        "rlvr_inputs": [pool[j][0] for j in rlvr_idx],
        "rlvr_outputs": [pool[j][1] for j in rlvr_idx],
        "truth_inputs": [pool[j][0] for j in truth_idx],
        "truth_outputs": [pool[j][1] for j in truth_idx],
        "cf_contest_id": row.get("cf_contest_id"),
        "cf_index": row.get("cf_index"),
        "cf_rating": row.get("cf_rating"),
        "difficulty": row.get("difficulty"),
        "source": row.get("source"),
        "n_public": len(_pairs(row["public_tests"])),
        "n_private": len(_pairs(row["private_tests"])),
    }, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=500, help="problems held out as the test split")
    ap.add_argument("--limit-shards", type=int, default=0, help="0 = all (smoke: use 2)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    shards = sorted(
        s.rfilename for s in HfApi().dataset_info(HF_REPO, revision=HF_REVISION).siblings
        if s.rfilename.endswith(".parquet")
    )
    if args.limit_shards:
        shards = [s for s in shards if "/train-" in s][: args.limit_shards]
    print(f"{len(shards)} shards from {HF_REPO}@{HF_REVISION[:12]}", flush=True)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    rejects: Counter = Counter()
    t0 = time.time()
    for n, shard in enumerate(shards, 1):
        with fs.open(f"datasets/{HF_REPO}@{HF_REVISION}/{shard}", "rb") as fh:
            table = pq.ParquetFile(fh).read(columns=PROJECTED)
        for rec in table.to_pylist():
            row, reason = build_row(rec, rng)
            if row is None:
                rejects[reason] += 1
            else:
                rows.append(row)
        print(f"  [{n}/{len(shards)}] {shard.split('/')[-1][:34]:34s} "
              f"kept={len(rows)} dropped={sum(rejects.values())} ({time.time()-t0:.0f}s)", flush=True)

    # Hold out whole CONTESTS, so near-duplicate problems from one contest
    # cannot straddle the split.
    by_contest: dict = {}
    for r in rows:
        by_contest.setdefault(r["cf_contest_id"], []).append(r)
    contests = sorted(by_contest, key=lambda c: (c is None, c))
    random.Random(args.seed + 1).shuffle(contests)
    test, test_contests = [], set()
    for c in contests:
        if len(test) >= args.holdout:
            break
        test.extend(by_contest[c])
        test_contests.add(c)
    train = [r for r in rows if r["cf_contest_id"] not in test_contests]

    written = {}
    for split, data in (("train", train), ("test", test)):
        p = out / f"{split}.jsonl"
        with open(p, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        written[split] = {"file": p.name, "rows": len(data), "sha256": h,
                          "size_bytes": p.stat().st_size}
        print(f"wrote {p}  {len(data)} rows  sha256={h[:16]}…", flush=True)

    manifest = {
        "source": {"repo": HF_REPO, "revision": HF_REVISION, "projected_columns": PROJECTED,
                   "shards_read": len(shards)},
        "suites": {
            "rlvr_tests": "seeded sample of public_tests+private_tests; the RLVR arm's reward",
            "truth_tests": "the complement of that sample; ground truth for BOTH arms",
        },
        "limits": {"max_rlvr_tests": MAX_RLVR_TESTS,
                   "max_single_test_bytes": MAX_SINGLE_TEST_BYTES,
                   "max_rlvr_total_bytes": MAX_RLVR_TOTAL_BYTES,
                   "min_total_tests": MIN_TOTAL_TESTS},
        "seed": args.seed,
        "split": {"policy": "whole contests held out (cf_contest_id)",
                  "holdout_target": args.holdout,
                  "test_contests": len(test_contests)},
        "counts": {"admitted": len(rows), "rejected": sum(rejects.values()),
                   "rejections": dict(rejects),
                   "with_nonempty_truth_suite": sum(1 for r in rows if r["truth_inputs"]),
                   "rlvr_suite_sizes": dict(sorted(Counter(len(r["rlvr_inputs"]) for r in rows).items())),
                   "truth_suite_sizes": dict(sorted(Counter(len(r["truth_inputs"]) for r in rows).items()))},
        "outputs": written,
        "filters": {"multi_answer_phrases": list(_MULTI_ANSWER_PHRASES),
                    "note": "phrase list is verbatim from the old repo's loader"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nadmitted {len(rows)}  rejected {sum(rejects.values())}  {dict(rejects)}")
    print(f"train {len(train)}  test {len(test)} ({len(test_contests)} contests)")
    print(f"manifest -> {out/'manifest.json'}")


if __name__ == "__main__":
    main()
