#!/usr/bin/env python3
"""Extract CodeContests-O corner cases for the held-out split.

CCO is 324 GB across 1386 parquet shards, and we need ~460 problems out of the
11,407 it contains. Downloading it is not on the table, so this reads parquet
COLUMNARLY over HTTP: pass 1 pulls only `name` from every shard to learn where
each problem lives, pass 2 pulls only `corner_cases` from the shards that
actually hold one of ours. `description`, `generator`, `checker` and `results`
-- which are the bulk of those 324 GB -- are never transferred.

CCO ships testlib `checker` programs for problems admitting multiple valid
outputs. We do not run them. Instead, any problem where CCO and DeepMind give
DIFFERENT expected output for the same input is dropped: two independent
regenerations disagreeing is direct evidence that the problem is multi-answer
or that one release's data is bad, and either way exact comparison would
mis-grade it. That catches cases the build's phrase filter cannot -- 1089_F
"Fractions" never says "print any", its multiplicity is implicit in the task --
but it only fires where the two releases happen to share an input, so it is a
lower bound, not a guarantee that survivors are clean.

Cases are capped exactly as the reward suite is (<=10, 500 KB each, 2 MB total),
so a CCO eval costs the same order as a DeepMind one and neither curve is
inflated by simply running more code.

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
import random
import re
import sys
import time
from pathlib import Path

HF_REPO = "caijanfeng/CodeContests-O"
N_SHARDS = 1386
# Same three caps the reward suite uses (build_codecontests_rlvr.py:78-80), so
# a CCO eval costs the same order as a DeepMind one and the two curves are not
# confounded by one suite simply running more code than the other.
MAX_CCO_TESTS = 10
MAX_SINGLE_TEST_BYTES = 500_000
MAX_TOTAL_BYTES = 2_000_000
# Sampling seed. Per-problem and keyed on the name, so the chosen cases do not
# depend on shard iteration order (which is thread-nondeterministic).
SEED = 0


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


def parse_cases(raw: list) -> tuple[list[tuple[str, str]], str | None]:
    """CCO corner_cases -> [(input, output)], or a reason for skipping.

    Oversized cases are dropped individually rather than dropping the problem:
    CCO reaches 690 KB on problems where scale is the whole point, and the rest
    of that problem's cases are still perfectly good.
    """
    pairs = []
    for c in raw or []:
        try:
            i, o = c["input"]["stdin"], c["output"]["stdout"]
        except (KeyError, TypeError):
            return [], "malformed_case"
        if not isinstance(i, str) or not isinstance(o, str):
            return [], "non_string_case"
        if len(i) > MAX_SINGLE_TEST_BYTES or len(o) > MAX_SINGLE_TEST_BYTES:
            continue
        pairs.append((i, o))
    if not pairs:
        return [], "no_usable_cases"
    return pairs, None


def sample_cases(pairs: list[tuple[str, str]], name: str) -> tuple[list[str], list[str]]:
    """Seeded <=MAX_CCO_TESTS sample under the total-bytes budget."""
    order = list(range(len(pairs)))
    random.Random(f"{SEED}:{name}").shuffle(order)
    inputs, outputs, total = [], [], 0
    for j in order:
        if len(inputs) >= MAX_CCO_TESTS:
            break
        i, o = pairs[j]
        if total + len(i) + len(o) > MAX_TOTAL_BYTES:
            continue
        total += len(i) + len(o)
        inputs.append(i)
        outputs.append(o)
    return inputs, outputs


def normalize_output(s: str) -> str:
    """Byte-for-byte the runner's comparator (codecontests.py:420).

    Duplicated rather than imported because it lives inside the runner SCRIPT
    TEMPLATE, not at module scope. If that one changes, change this one.
    """
    out = []
    for line in s.split("\n"):
        line = line.rstrip()

        def nf(m):
            try:
                return f"{float(m.group(0)):g}"
            except (ValueError, OverflowError):
                return m.group(0)

        out.append(re.sub(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", nf, line))
    while out and not out[-1]:
        out.pop()
    return "\n".join(out).lower()


def disagrees(pairs: list[tuple[str, str]], row: dict) -> tuple[str, str] | None:
    """Does CCO contradict DeepMind on an input they BOTH carry?

    Two independent regenerations producing different expected output for the
    same input means one of three things, and all three make the problem
    unusable for exact-comparison grading: the problem admits multiple valid
    answers (which _MULTI_ANSWER_PHRASES cannot catch when the multiplicity is
    implicit in the task rather than stated -- 1089_F "Fractions" is exactly
    that), or one release's data is corrupt (we found DeepMind entries that are
    whitespace where a value belongs), or the checker is non-trivial.

    This only fires on problems where the two releases happen to share an
    input, so it is a lower bound on how many bad problems exist -- not a
    guarantee that the survivors are clean.
    """
    gdm = list(zip(row.get("rlvr_inputs") or [], row.get("rlvr_outputs") or []))
    gdm += list(zip(row.get("truth_inputs") or [], row.get("truth_outputs") or []))
    by_input = {normalize_output(i): o for i, o in pairs}
    for i, o in gdm:
        k = normalize_output(i)
        if k in by_input and normalize_output(by_input[k]) != normalize_output(o):
            return (normalize_output(o)[:60], normalize_output(by_input[k])[:60])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default="data/codecontests/test.jsonl")
    ap.add_argument("--out", default="data/codecontests/cco_eval.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--index-cache", default=None, help="reuse a prior name index")
    args = ap.parse_args()

    test_rows = [json.loads(l) for l in open(args.test)]
    by_name = {r["name"]: r for r in test_rows}
    wanted = [r["name"] for r in test_rows]
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

    rows, skipped, conflicts = [], {}, []

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
                pairs, why = parse_cases(raw)
                if why:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                # Cross-check on the FULL case list before sampling: a <=10
                # sample might not include the contradicting input, and a
                # problem that is unusable is unusable either way.
                clash = disagrees(pairs, by_name[name])
                if clash:
                    conflicts.append({"name": name, "deepmind": clash[0], "cco": clash[1]})
                    continue
                inputs, outputs = sample_cases(pairs, name)
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
        "dropped_disagreement": len(conflicts),
        "disagreement_examples": conflicts[:10],
        "total_cases": n_cases,
        "mean_cases_per_problem": round(n_cases / len(rows), 1) if rows else 0,
        "bytes": outp.stat().st_size,
        "sha256": sha,
        "max_cco_tests": MAX_CCO_TESTS,
        "max_single_test_bytes": MAX_SINGLE_TEST_BYTES,
        "max_total_bytes": MAX_TOTAL_BYTES,
        "seed": SEED,
    }
    json.dump(manifest, open(outp.with_suffix(".manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
