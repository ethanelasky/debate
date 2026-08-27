#!/usr/bin/env python3
"""passk_capped_eval sweep -> a train_filter_file carve.

The sweep reports per-row solve counts by INDEX into the split it walked; a
filter file is a list of problem_key hashes. This maps one to the other by
rebuilding the same env from the same dataset args, so the carve is a pure
function of (sweep output, dataset args) and can be rebuilt for any model.

    python scripts/passk_capped_eval.py --base-url http://127.0.0.1:8790/v1 \
        --model Qwen/Qwen3.5-9B --family math --split train --full-pool \
        --dataset-args '{"levels": 5, "seed": 0}' \
        --think-cap 4000 --visible-cap 1024 --n 8 --out sweep9b.json
    python scripts/carve_from_passk.py --passk sweep9b.json --family math \
        --dataset-args '{"levels": 5, "seed": 0}' --keep-below 8 \
        --out configs/filters/l5_sub8_qwen35_9b.json

The sweep MUST have run with --full-pool: without it the split is sampled
rather than walked, and an index no longer names a row.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--passk", required=True, help="passk_capped_eval --out json")
    p.add_argument("--family", required=True)
    p.add_argument("--dataset-args", default="{}")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--keep-below",
        type=int,
        default=8,
        help="keep rows solved FEWER than this many times (8 = drop always-solved)",
    )
    p.add_argument(
        "--keep-above",
        type=int,
        default=-1,
        help="keep rows solved MORE than this many times (0 = drop never-solved)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from infra.envs.tasks import get_family
    from infra.envs.tasks.math import problem_key

    summary = json.load(open(args.passk))
    if summary.get("split") != args.split:
        raise SystemExit(
            f"sweep walked split {summary.get('split')!r}, not {args.split!r}"
        )
    family = get_family(args.family)
    env = family.source(json.loads(args.dataset_args))
    pool = getattr(env, f"{args.split}_rows")

    rows = summary["rows"]
    if len(rows) != len(pool):
        raise SystemExit(
            f"sweep covered {len(rows)} rows but the split rebuilt to {len(pool)}; "
            "the dataset args here must match the sweep's, and the sweep must "
            "have run with --full-pool"
        )
    seen = {r["idx"] for r in rows}
    if seen != set(range(len(pool))):
        raise SystemExit("sweep rows are not a complete 0..n-1 index range")

    histogram = Counter(r["correct"] for r in rows)
    keep = [
        problem_key(pool[r["idx"]]["problem"])
        for r in sorted(rows, key=lambda r: r["idx"])
        if args.keep_above < r["correct"] < args.keep_below
    ]
    if len(set(keep)) != len(keep):
        raise SystemExit("duplicate problem_key in the carve: the pool has repeats")
    if not keep:
        raise SystemExit("carve is empty")

    with open(args.out, "w") as f:
        json.dump(sorted(keep), f, indent=1)

    n = len(rows)
    k = summary.get("k")
    print(f"solve histogram over {n} rows (k={k}):")
    for correct in sorted(histogram):
        share = histogram[correct] / n
        print(f"  {correct}/{k}: {histogram[correct]:>5d}  ({share:6.1%})")
    print(f"carve: {len(keep)} of {n} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
