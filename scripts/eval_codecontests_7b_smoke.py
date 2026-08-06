#!/usr/bin/env python3
"""Inference-only OLMo 7B smoke over easy CodeContests rating bands.

Uses the repository's exact CodeContests prompt composition, code extractor,
and verifier.  It performs no optimizer step and writes every completion plus
a machine-readable aggregate summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from infra.envs.tasks.codecontests import CodeContestsEnv


MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
REVISION = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
BANDS = ((800, 1000), (1100, 1200), (1300, 1400))


def _band_name(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _band_rows(rows: list[dict[str, Any]], lo: int, hi: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if isinstance(row.get("cf_rating"), int) and lo <= row["cf_rating"] <= hi
    ]


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def _grade(
    env: CodeContestsEnv,
    task: Any,
    output: Any,
    *,
    kind: str,
    band: str,
    row: dict[str, Any],
    task_index: int,
) -> dict[str, Any]:
    reward, info = env.reward(task, output.text)
    return {
        "kind": kind,
        "band": band,
        "task_index": task_index,
        "sample_index": int(output.index),
        "name": row["name"],
        "cf_rating": row["cf_rating"],
        "difficulty": row.get("difficulty"),
        "reward": reward,
        "correct": bool(info["correct"]),
        "has_code": bool(info["has_code"]),
        "cpp_code": bool(info["cpp_code"]),
        "exec_timeout": bool(info["exec_timeout"]),
        "exec_error": bool(info["exec_error"]),
        "tests_passed_frac": info["tests_passed_frac"],
        "finish_reason": output.finish_reason,
        "token_count": len(output.token_ids),
        "text": output.text,
    }


def _summary(records: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"inventory": inventory, "greedy": {}, "sampled_g8": {}}

    greedy = [r for r in records if r["kind"] == "greedy_test"]
    for band in [_band_name(*x) for x in BANDS]:
        rows = [r for r in greedy if r["band"] == band]
        correct = sum(r["correct"] for r in rows)
        result["greedy"][band] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "wilson_95": _wilson(correct, len(rows)),
            "has_code_rate": sum(r["has_code"] for r in rows) / len(rows) if rows else None,
            "timeout_rate": sum(r["exec_timeout"] for r in rows) / len(rows) if rows else None,
            "exec_error_rate": sum(r["exec_error"] for r in rows) / len(rows) if rows else None,
            "length_stop_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows)
            if rows
            else None,
            "mean_tokens": sum(r["token_count"] for r in rows) / len(rows) if rows else None,
        }

    for cap in (1000, 1200, 1400):
        rows = [r for r in greedy if r["cf_rating"] <= cap]
        correct = sum(r["correct"] for r in rows)
        result["greedy"][f"cumulative_le_{cap}"] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "wilson_95": _wilson(correct, len(rows)),
        }

    sampled = [r for r in records if r["kind"] == "sampled_train_g8"]
    for band in [_band_name(*x) for x in BANDS]:
        rows = [r for r in sampled if r["band"] == band]
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["task_index"]].append(row)
        correct_by_group = [sum(r["correct"] for r in group) for group in groups.values()]
        n_groups = len(correct_by_group)
        mixed = sum(0 < value < 8 for value in correct_by_group)
        any_pass = sum(value > 0 for value in correct_by_group)
        all_pass = sum(value == 8 for value in correct_by_group)
        correct = sum(correct_by_group)
        result["sampled_g8"][band] = {
            "tasks": n_groups,
            "rollouts": len(rows),
            "correct": correct,
            "sample_accuracy": correct / len(rows) if rows else None,
            "wilson_95": _wilson(correct, len(rows)),
            "mixed_groups": mixed,
            "mixed_group_rate": mixed / n_groups if n_groups else None,
            "pass_at_8_tasks": any_pass,
            "pass_at_8_rate": any_pass / n_groups if n_groups else None,
            "all_fail_groups": sum(value == 0 for value in correct_by_group),
            "all_pass_groups": all_pass,
            "successes_per_group": dict(
                sorted((str(k), v) for k, v in __import__("collections").Counter(correct_by_group).items())
            ),
            "has_code_rate": sum(r["has_code"] for r in rows) / len(rows) if rows else None,
            "timeout_rate": sum(r["exec_timeout"] for r in rows) / len(rows) if rows else None,
            "exec_error_rate": sum(r["exec_error"] for r in rows) / len(rows) if rows else None,
            "length_stop_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows)
            if rows
            else None,
            "mean_tokens": sum(r["token_count"] for r in rows) / len(rows) if rows else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/codecontests"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--sampled-tasks-per-band", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=104729)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = CodeContestsEnv(
        path=str(args.data_dir / "train.jsonl"),
        test_path=str(args.data_dir / "test.jsonl"),
        eval_subset_size=100_000,
        timeout_seconds=10,
    )

    inventory: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    for band_index, (lo, hi) in enumerate(BANDS):
        band = _band_name(lo, hi)
        train_rows = _band_rows(env.train_rows, lo, hi)
        test_rows = _band_rows(env.test_rows, lo, hi)
        sample_n = min(args.sampled_tasks_per_band, len(train_rows))
        sampled_rows = random.Random(args.seed + band_index).sample(train_rows, sample_n)
        inventory[band] = {
            "train_available": len(train_rows),
            "test_available": len(test_rows),
            "train_sampled_g8": sample_n,
        }
        for kind, rows, n in (
            ("greedy_test", test_rows, 1),
            ("sampled_train_g8", sampled_rows, 8),
        ):
            for task_index, row in enumerate(rows):
                task = env._task(row, "test" if kind == "greedy_test" else "train")
                requests.append(
                    {
                        "kind": kind,
                        "band": band,
                        "row": row,
                        "task": task,
                        "task_index": task_index,
                        "n": n,
                    }
                )

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="bfloat16",
        max_model_len=6144,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        max_num_seqs=128,
    )
    tokenizer = llm.get_tokenizer()

    records: list[dict[str, Any]] = []
    started = time.time()
    for kind, n, temperature in (("greedy_test", 1, 0.0), ("sampled_train_g8", 8, 1.0)):
        selected = [request for request in requests if request["kind"] == kind]
        prompts = [
            tokenizer.apply_chat_template(
                request["task"].messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for request in selected
        ]
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed + (0 if kind == "greedy_test" else 1),
        )
        outputs = llm.generate(prompts, params, use_tqdm=True)
        jobs: list[tuple[Any, Any, dict[str, Any]]] = []
        for request, generated in zip(selected, outputs):
            for output in generated.outputs:
                jobs.append((request["task"], output, request))
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(
                    _grade,
                    env,
                    task,
                    output,
                    kind=kind,
                    band=request["band"],
                    row=request["row"],
                    task_index=request["task_index"],
                )
                for task, output, request in jobs
            ]
            records.extend(future.result() for future in futures)

    records.sort(key=lambda r: (r["kind"], r["band"], r["task_index"], r["sample_index"]))
    records_path = args.out_dir / "records.jsonl"
    with records_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summary(records, inventory)
    summary["run"] = {
        "model": args.model,
        "revision": args.revision,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sampled_tasks_per_band": args.sampled_tasks_per_band,
        "elapsed_seconds": time.time() - started,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
