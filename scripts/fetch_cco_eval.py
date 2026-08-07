#!/usr/bin/env python3
"""Extract CodeContests-O corner cases for the held-out split.

CCO is 324 GB across 1386 parquet shards, and we need ~460 problems out of the
11,407 it contains. Downloading it is not on the table, so this reads parquet
COLUMNARLY over HTTP: pass 1 pulls only `name` from every shard to learn where
each problem lives, pass 2 pulls only `corner_cases` from the shards that
actually hold one of ours. `description`, `generator`, `checker` and `results`
-- which are the bulk of those 324 GB -- are never transferred.

CCO ships testlib `checker` programs for problems admitting multiple valid
outputs. We do not run them and do not need to: the build already drops
multi-answer problems (_MULTI_ANSWER_PHRASES), so every problem that survives
into our split is exact-comparison safe.

Names carry trailing whitespace in CCO ('584_B. Kolya and Tanya ') but not in
DeepMind's release, so the join is on the stripped name.

Usage:
    python scripts/fetch_cco_eval.py --test data/codecontests/test.jsonl \
        --out data/codecontests/cco_eval.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

HF_REPO = "caijanfeng/CodeContests-O"
N_SHARDS = 1386
# Mirrors the build's per-case ceiling: one case above this cannot be piped
# through a subprocess in reasonable time, and CCO reaches 690 KB on problems
# where scale is the point.
MAX_SINGLE_TEST_BYTES = 500_000
# Per-problem ceiling. CCO averages ~1.3 MB/problem; this bounds the tail so a
# single pathological problem cannot dominate an eval's wall-clock.
MAX_TOTAL_BYTES = 4_000_000


def _shard(i: int) -> str:
    return f"datasets/{HF_REPO}/data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def _read(i: int, columns: list[str], retries: int = 3):
    import fsspec
    import pyarrow.parquet as pq

    for attempt in range(retries):
        try:
            with fsspec.filesystem("hf").open(_shard(i), "rb") as f:
                return pq.ParquetFile(f).read(columns=columns)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def build_index(workers: int) -> dict[str, int]:
    """stripped problem name -> shard index. Reads only the `name` column."""
    index: dict[str, int] = {}
    failures: list[int] = []

    def one(i: int):
        try:
            return i, _read(i, ["name"]).column("name").to_pylist()
        except Exception as e:
            return i, e

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for done, (i, names) in enumerate(ex.map(one, range(N_SHARDS)), 1):
            if isinstance(names, Exception):
                failures.append(i)
            else:
                for n in names:
                    index[n.strip()] = i
            if done % 200 == 0:
                print(f"  index {done}/{N_SHARDS} shards, {len(index)} problems", flush=True)
    if failures:
        # Loud: a silently short index looks identical to genuine absence from
        # CCO, and would understate coverage rather than fail.
        raise SystemExit(f"FATAL: {len(failures)} shards unreadable: {failures[:10]}")
    return index


def cases_from(raw: list) -> tuple[list[str], list[str], str | None]:
    """CCO corner_cases -> (inputs, outputs), or a reason for skipping."""
    inputs, outputs, total = [], [], 0
    for c in raw:
        try:
            i, o = c["input"]["stdin"], c["output"]["stdout"]
        except (KeyError, TypeError):
            return [], [], "malformed_case"
        if not isinstance(i, str) or not isinstance(o, str):
            return [], [], "non_string_case"
        if len(i) > MAX_SINGLE_TEST_BYTES or len(o) > MAX_SINGLE_TEST_BYTES:
            continue  # drop the oversized case, keep the problem
        total += len(i) + len(o)
        if total > MAX_TOTAL_BYTES:
            break
        inputs.append(i)
        outputs.append(o)
    if not inputs:
        return [], [], "no_usable_cases"
    return inputs, outputs, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default="data/codecontests/test.jsonl")
    ap.add_argument("--out", default="data/codecontests/cco_eval.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--index-cache", default=None, help="reuse a prior name index")
    args = ap.parse_args()

    wanted = [json.loads(l)["name"] for l in open(args.test)]
    print(f"held-out problems: {len(wanted)}")

    if args.index_cache and Path(args.index_cache).exists():
        index = json.load(open(args.index_cache))
        print(f"index loaded from cache: {len(index)} CCO problems")
    else:
        print(f"pass 1: indexing {N_SHARDS} shards by name (name column only)")
        index = build_index(args.workers)
        if args.index_cache:
            json.dump(index, open(args.index_cache, "w"))
        print(f"  indexed {len(index)} CCO problems")

    hits = {n: index[n.strip()] for n in wanted if n.strip() in index}
    missing = [n for n in wanted if n.strip() not in index]
    by_shard: dict[int, list[str]] = {}
    for n, s in hits.items():
        by_shard.setdefault(s, []).append(n)
    print(f"matched {len(hits)}/{len(wanted)} ({100*len(hits)/len(wanted):.1f}%) "
          f"across {len(by_shard)} shards; pass 2 pulls corner_cases from those")

    rows, skipped = [], {}

    def fetch(shard: int):
        try:
            t = _read(shard, ["name", "corner_cases"])
        except Exception as e:
            return shard, e
        want = {n.strip() for n in by_shard[shard]}
        out = []
        names = t.column("name").to_pylist()
        for j, nm in enumerate(names):
            if nm.strip() in want:
                out.append((nm.strip(), t.column("corner_cases")[j].as_py()))
        return shard, out

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for done, (shard, got) in enumerate(ex.map(fetch, sorted(by_shard)), 1):
            if isinstance(got, Exception):
                raise SystemExit(f"FATAL: shard {shard} failed in pass 2: {got}")
            for name, raw in got:
                inputs, outputs, why = cases_from(raw or [])
                if why:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                rows.append({"name": name, "cco_inputs": inputs, "cco_outputs": outputs})
            if done % 50 == 0:
                print(f"  fetch {done}/{len(by_shard)} shards, {len(rows)} problems", flush=True)

    # Sorted so the artifact is byte-stable across runs and its sha256 means
    # something; thread completion order is not deterministic.
    rows.sort(key=lambda r: r["name"])
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_cases = sum(len(r["cco_inputs"]) for r in rows)
    sha = hashlib.sha256(outp.read_bytes()).hexdigest()
    manifest = {
        "source": HF_REPO,
        "n_shards": N_SHARDS,
        "held_out_problems": len(wanted),
        "matched": len(hits),
        "written": len(rows),
        "missing_from_cco": len(missing),
        "skipped": skipped,
        "total_cases": n_cases,
        "mean_cases_per_problem": round(n_cases / len(rows), 1) if rows else 0,
        "bytes": outp.stat().st_size,
        "sha256": sha,
        "max_single_test_bytes": MAX_SINGLE_TEST_BYTES,
        "max_total_bytes": MAX_TOTAL_BYTES,
    }
    json.dump(manifest, open(outp.with_suffix(".manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
