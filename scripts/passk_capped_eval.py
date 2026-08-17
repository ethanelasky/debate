"""pass@1 / pass@k eval of a served model under a task family's own dev
protocol: the env renders the messages and grades with env.reward, so
correct_strict here is the same quantity as the training run's dev metric.

The chat template is applied LOCALLY (tokenizer.apply_chat_template with
add_generation_prompt) and generation runs against /v1/completions on raw
text, sidestepping server-side template re-rendering of assembled turns.
Budget-forced think sampling is reproduced in two phases, the same shape as
infra.envs.base.budget_forced_sample: phase 1 caps the think stream (stop
["</think>"]); the close is re-injected as "</think>\n\n" whether sampled or
forced; phase 2 continues from prompt + think + close under the visible cap.
--think-cap 0 disables forcing (single request at --max-tokens).

    python scripts/passk_capped_eval.py --base-url http://127.0.0.1:8790/v1 \
        --model Qwen/Qwen3.5-4B --family amc \
        --dataset-args '{"prompt_file": "infra/prompts/tasks/math_nocot.yaml"}' \
        --think-cap 4000 --visible-cap 1024 --n 8 --out results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.request

FORCED_CLOSE_TEXT = "</think>\n\n"


def _post(base_url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def sample_once(base_url: str, model: str, prompt: str, args) -> dict:
    base = {"model": model, "temperature": args.temperature, "top_p": args.top_p}
    if not args.think_cap:
        out = _post(base_url, {**base, "prompt": prompt, "max_tokens": args.max_tokens})
        choice = out["choices"][0]
        return {
            "completion": choice.get("text") or "",
            "think_forced": False,
            "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
        }
    p1 = _post(
        base_url,
        {**base, "prompt": prompt, "max_tokens": args.think_cap, "stop": ["</think>"]},
    )
    c1 = p1["choices"][0]
    think = c1.get("text") or ""
    forced = c1.get("finish_reason") == "length"
    partial = think + FORCED_CLOSE_TEXT
    p2 = _post(
        base_url,
        {**base, "prompt": prompt + partial, "max_tokens": args.visible_cap},
    )
    visible = p2["choices"][0].get("text") or ""
    n1 = (p1.get("usage") or {}).get("completion_tokens") or 0
    n2 = (p2.get("usage") or {}).get("completion_tokens") or 0
    return {
        "completion": partial + visible,
        "think_forced": forced,
        "completion_tokens": n1 + n2,
        "think_tokens": n1,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--dataset-args", default="{}")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--think-cap", type=int, default=0)
    p.add_argument("--visible-cap", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from transformers import AutoTokenizer

    from infra.envs.tasks import get_family

    family = get_family(args.family)
    env = family.source(json.loads(args.dataset_args))
    pool = getattr(env, f"{args.split}_rows")
    tasks = env.tasks(len(pool), split=args.split)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            list(t.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for t in tasks
    ]

    jobs = [(i, t, prompts[i]) for i, t in enumerate(tasks) for _ in range(args.n)]
    results: dict[int, list[bool]] = {i: [] for i, _, _ in jobs}
    forced_ct = 0
    samples_path = (args.out or "eval") + ".samples.jsonl"
    samples_f = open(samples_path, "w")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(sample_once, args.base_url, args.model, pr, args): (i, t)
            for i, t, pr in jobs
        }
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i, t = futs[fut]
            try:
                s = fut.result()
                _, info = env.reward(t, s["completion"])
                ok = bool(info.get("correct_strict"))
            except Exception as e:
                print(f"[gen-error] problem {i}: {type(e).__name__}: {e}", flush=True)
                s, ok = {"completion": None, "think_forced": None}, False
            forced_ct += 1 if s.get("think_forced") else 0
            samples_f.write(
                json.dumps(
                    {
                        "problem": i,
                        "gt": t.meta.get("gt"),
                        "correct": ok,
                        "think_forced": s.get("think_forced"),
                        "completion_tokens": s.get("completion_tokens"),
                        "think_tokens": s.get("think_tokens"),
                        "text": s.get("completion"),
                    }
                )
                + "\n"
            )
            samples_f.flush()
            results[i].append(ok)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} samples done", flush=True)
    samples_f.close()

    rows = [
        {"idx": i, "correct": sum(w), "n": len(w)} for i, w in sorted(results.items())
    ]
    n_total = sum(r["n"] for r in rows)
    n_correct = sum(r["correct"] for r in rows)
    pass1 = n_correct / n_total if n_total else 0.0
    passk = sum(1 for r in rows if r["correct"] > 0) / len(rows) if rows else 0.0
    forced_rate = forced_ct / n_total if n_total else 0.0
    summary = {
        "pass1": pass1,
        "passk": passk,
        "k": args.n,
        "n_problems": len(rows),
        "think_forced_rate": forced_rate,
        "think_cap": args.think_cap,
        "visible_cap": args.visible_cap,
        "family": args.family,
        "split": args.split,
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"pass@1 = {pass1:.4f}  ({n_correct}/{n_total} samples)")
    print(f"pass@{args.n} = {passk:.4f}  ({sum(1 for r in rows if r['correct'] > 0)}/{len(rows)} problems)")
    print(f"think_forced_rate = {forced_rate:.4f}")


if __name__ == "__main__":
    main()
