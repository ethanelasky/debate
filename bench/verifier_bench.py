#!/usr/bin/env python3
"""Benchmark harness for the CodeContests verifier.

Fixture: bench/verifier_cases.jsonl — 632 REAL completions from the
Olmo-3.1-32B run, each with the exact rlvr suite it was graded against.

A candidate implementation is only acceptable if it returns the SAME
pass/fail verdict as the baseline on every case. Speed with different
verdicts is not a win, it is a different experiment.

Usage:
    python bench/verifier_bench.py --baseline            # record the reference
    python bench/verifier_bench.py --candidate mod:fn    # score a candidate

`mod:fn` must expose fn(code, inputs, outputs, timeout) -> dict with at least
{"passed": bool}, matching run_stdin_tests' contract.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "bench" / "verifier_cases.jsonl"
BASELINE = ROOT / "bench" / "baseline_verdicts.json"


def load_cases(limit: int | None = None) -> list[dict]:
    cases = [json.loads(l) for l in open(FIXTURE)]
    return cases[:limit] if limit else cases


def resolve(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def run(fn, cases: list[dict], workers: int, timeout: int) -> tuple[dict, float, list[float]]:
    """-> ({name#i: passed}, wallclock, per-case seconds)."""
    times: list[float] = [0.0] * len(cases)

    def one(idx_case):
        i, c = idx_case
        t = time.perf_counter()
        try:
            res = fn(c["code"], c["inputs"], c["outputs"], timeout=timeout)
            passed = bool(res.get("passed"))
        except Exception as e:  # a crashing verifier is a failed candidate
            passed = f"ERROR:{type(e).__name__}"
        times[i] = time.perf_counter() - t
        return f"{c['name']}#{i}", passed

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = dict(pool.map(one, enumerate(cases)))
    return verdicts, time.perf_counter() - t0, times


def report(label: str, wall: float, times: list[float]) -> None:
    s = sorted(times)
    print(f"  {label:22s} wallclock {wall:7.1f}s   sum {sum(times):7.1f}s   "
          f"median {statistics.median(s):.3f}s   p99 {s[int(0.99 * (len(s) - 1))]:6.2f}s   max {max(s):6.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--candidate", help="module:function")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_cases(args.limit)
    print(f"{len(cases)} cases, workers={args.workers}, timeout={args.timeout}s")

    if args.baseline:
        from infra.envs.tasks.codecontests import run_stdin_tests

        v, wall, times = run(run_stdin_tests, cases, args.workers, args.timeout)
        BASELINE.write_text(json.dumps({"verdicts": v, "wallclock": wall,
                                        "workers": args.workers, "timeout": args.timeout}))
        report("baseline", wall, times)
        print(f"  passed: {sum(1 for x in v.values() if x is True)}/{len(v)}")
        print(f"  wrote {BASELINE}")
        return

    if not args.candidate:
        ap.error("need --baseline or --candidate")
    if not BASELINE.exists():
        ap.error("run --baseline first")

    base = json.loads(BASELINE.read_text())
    fn = resolve(args.candidate)
    v, wall, times = run(fn, cases, args.workers, args.timeout)

    report("baseline (recorded)", base["wallclock"], [0.0])
    report("candidate", wall, times)

    keys = set(base["verdicts"]) & set(v)
    disagree = [k for k in keys if base["verdicts"][k] != v[k]]
    errors = [k for k in keys if isinstance(v[k], str)]
    print(f"\n  verdict agreement : {len(keys) - len(disagree)}/{len(keys)} "
          f"({100 * (len(keys) - len(disagree)) / max(1, len(keys)):.2f}%)")
    print(f"  candidate errors  : {len(errors)}")
    if disagree:
        print(f"  DISAGREEMENTS ({len(disagree)}), first 10:")
        for k in disagree[:10]:
            print(f"     {k}: baseline={base['verdicts'][k]} candidate={v[k]}")
    speedup = base["wallclock"] / wall if wall else float("inf")
    verdict = "ACCEPTABLE" if not disagree and not errors else "REJECTED (verdicts changed)"
    print(f"\n  speedup {speedup:.2f}x   ->  {verdict}")


if __name__ == "__main__":
    main()
