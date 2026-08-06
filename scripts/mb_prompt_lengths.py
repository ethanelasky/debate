"""Prompt-token-length census for the MB RLVR context, under any tokenizer.

The RLVR/blind-choice prompt is the family's rendered answer-generation context
([system: monitor card][user: trajectory][user: blind instructions]) — the same
messages the eval arm's blind view splices in. Its token length is
tokenizer-specific, so a model swap (OLMo-3 -> Qwen3.6) invalidates the
`training.verl.prompt_length` / tinker `max_context_length` budget that was
sized against the old one. This reports the distribution so the budget can be
set from data rather than guessed.

    python scripts/mb_prompt_lengths.py \
        --model Qwen/Qwen3.6-35B-A3B --no-think --budget 65536 --completion 4096

Splits default to the canonical task-disjoint files; --files/--split-file
override for other pools. Prints counts and quantiles ONLY — never row content
(data/monitoringbench/*.jsonl carry red-team trajectories).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_FILES = [
    "data/monitoringbench/honest_mb550_norm.jsonl",
    "data/monitoringbench/attacks_mb550_norm.jsonl",
]
DEFAULT_SPLITS = {
    "train": "data/monitoringbench/splits/train_ids.txt",
    "dev": "data/monitoringbench/splits/dev_ids.txt",
    "eval": "data/monitoringbench/splits/eval_ids.txt",
}


def quantile(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id whose tokenizer + chat template to use")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument(
        "--split-file",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="repeatable; default = the canonical train/dev/eval id files",
    )
    ap.add_argument("--no-think", action="store_true", help="apply_chat_template(enable_thinking=False)")
    ap.add_argument("--budget", type=int, default=None, help="total context ceiling to check against")
    ap.add_argument("--completion", type=int, default=0, help="tokens reserved for the completion")
    args = ap.parse_args()

    splits = DEFAULT_SPLITS
    if args.split_file:
        splits = dict(s.split("=", 1) for s in args.split_file)

    from transformers import AutoTokenizer

    from infra.envs.tasks.monitoringbench import MonitoringBenchTaskSource

    tok = AutoTokenizer.from_pretrained(args.model)
    chat_kwargs = {"enable_thinking": False} if args.no_think else {}

    # Whole pool, then group by the split id files ourselves: the source only
    # knows the train/test pair, and this census wants every split (incl. eval).
    src = MonitoringBenchTaskSource(files=args.files)
    by_id = {r.id: r for r in src.rows}

    print(f"model={args.model} no_think={args.no_think}")
    all_lens: list[int] = []
    for name, path in splits.items():
        ids = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
        rows = [by_id[i] for i in ids if i in by_id]
        lens = []
        for row in rows:
            msgs = src._task(row).messages
            ids = tok.apply_chat_template(
                [
                    m if isinstance(m, dict) else {"role": m.role, "content": m.content}
                    for m in msgs
                ],
                add_generation_prompt=True,
                tokenize=True,
                **chat_kwargs,
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            lens.append(len(ids))
        all_lens += lens
        print(
            f"{name:<6} n={len(lens):<4} min={min(lens):>7} p50={quantile(lens, .5):>7} "
            f"p95={quantile(lens, .95):>7} p99={quantile(lens, .99):>7} max={max(lens):>7}"
        )

    if args.budget:
        room = args.budget - args.completion
        over = [n for n in all_lens if n > room]
        print(
            f"budget={args.budget} completion={args.completion} -> prompt room={room}: "
            f"{len(over)}/{len(all_lens)} rows over ({100 * len(over) / len(all_lens):.1f}%)"
        )


if __name__ == "__main__":
    main()
