"""pass@1 / pass@k eval of a served model on MathArena HMMT.

    python scripts/hmmt_pass_eval.py --base-url http://127.0.0.1:8790/v1 \
        --model Qwen/Qwen3.5-4B [--dataset MathArena/hmmt_feb_2025] [--n 8] \
        [--temperature 1.0] [--top-p 1.0] [--max-tokens 20000] [--out results.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import urllib.request


def ask(base_url: str, model: str, problem: str, args) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{problem}\n\nPut your final answer within \\boxed{{}}.",
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.load(resp)
    usage = out.get("usage") or {}
    return out["choices"][0]["message"].get("content") or "", usage.get("completion_tokens")


def last_boxed(text: str) -> str | None:
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    i = starts[-1]
    depth = 1
    j = i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i : j - 1] if depth == 0 else None


def grade(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    from math_verify import parse, verify

    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return pred.strip() == gold.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="MathArena/hmmt_feb_2025")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from datasets import load_dataset

    if "hendrycks" in args.dataset:
        import random

        raw = load_dataset(args.dataset, split="test")
        pool = [r for r in raw if r.get("level") == "Level 5"]
        rows_src = random.Random(0).sample(pool, min(args.limit, len(pool)))
        ds = [
            {
                "problem_idx": str(i),
                "problem": r["problem"],
                "answer": last_boxed(r["solution"]) or "",
                "problem_type": r.get("type", ""),
            }
            for i, r in enumerate(rows_src)
        ]
        ds = [r for r in ds if r["answer"]]
    else:
        ds = list(load_dataset(args.dataset, split="train"))
    jobs = [(i, row["problem"], str(row["answer"])) for i, row in enumerate(ds) for _ in range(args.n)]

    completions: dict[int, list[bool]] = {i: [] for i, _, _ in jobs}
    # Every sample lands on disk with its full text and token count as soon as
    # it is graded: lost rollout data is unrecoverable.
    samples_path = (args.out or "eval") + ".samples.jsonl"
    samples_f = open(samples_path, "w")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(ask, args.base_url, args.model, prob, args): (i, gold)
            for i, prob, gold in jobs
        }
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i, gold = futs[fut]
            try:
                text, n_tokens = fut.result()
                ok = grade(last_boxed(text), gold)
            except Exception as e:
                print(f"[gen-error] problem {i}: {type(e).__name__}: {e}")
                text, n_tokens, ok = None, None, False
            samples_f.write(json.dumps({"problem": i, "gold": gold, "correct": ok,
                                        "completion_tokens": n_tokens, "text": text}) + "\n")
            samples_f.flush()
            completions[i].append(ok)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} samples done")
    samples_f.close()

    rows = []
    for i, row in enumerate(ds):
        wins = completions[i]
        rows.append(
            {
                "idx": str(row.get("problem_idx", i)),
                "type": str(row.get("problem_type", "")),
                "correct": sum(wins),
                "n": len(wins),
            }
        )
    n_total = sum(r["n"] for r in rows)
    n_correct = sum(r["correct"] for r in rows)
    pass1 = n_correct / n_total if n_total else 0.0
    passk = sum(1 for r in rows if r["correct"] > 0) / len(rows) if rows else 0.0
    # results hit disk before any printing: a report bug must not lose the data
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"pass1": pass1, "passk": passk, "k": args.n, "rows": rows}, f, indent=2)
    for r in rows:
        print(f"  problem {r['idx']:>3}  {r['correct']}/{r['n']}  {r['type']}")
    print(f"pass@1 = {pass1:.4f}  ({n_correct}/{n_total} samples)")
    print(f"pass@{args.n} = {passk:.4f}  ({sum(1 for r in rows if r['correct'] > 0)}/{len(rows)} problems)")


if __name__ == "__main__":
    main()
