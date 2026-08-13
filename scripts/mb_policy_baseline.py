"""Initial (pre-RLVR) blind-choice accuracy of a candidate policy model on the
MonitoringBench dev split, in the mode RLVR would train it: plan-then-answer
scratchpad on by default (PlannedEnv's two-turn shape, reproduced at message
level), single blind turn with --no-scratchpad.

Purpose: screening candidate policy models for the next RLVR arm — the number
this reports is what the training run's eval/correct_strict curve would start
from.
Prompts come from the family's answer-generation config and the PlannedEnv
plan cue, so the rendered messages are identical to the RLVR rollout's for the
same row by construction (modulo the chat template, which is the candidate
model's own).

Backends:
  --backend tinker      sample the base model on tinker (the training host —
                        use for any tinker-hosted candidate; proves the model
                        is trainable there at the same time)
  --backend openrouter  chat-completions fallback for candidates tinker does
                        not host (message-level parity only)

Examples:
  # tinker-hosted candidate, whole dev split
  python scripts/mb_policy_baseline.py --backend tinker \
      --model Qwen/Qwen3.5-9B --run-tag dev120

  # OpenRouter-only candidate
  python scripts/mb_policy_baseline.py --backend openrouter \
      --model nvidia/nemotron-nano-9b-v2:free --run-tag dev120

Selection: the first --n rows of the canonical dev split (label-alternating,
task-round-robin ordering — any prefix is balanced and task-diverse).

SAFETY: trajectory content is NEVER printed to stdout. Model responses go to
the output jsonl only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.envs.planned import PlannedEnv  # noqa: E402
from infra.envs.tasks import get_family  # noqa: E402

DEFAULT_DATA_PATHS = (
    "data/monitoringbench/honest_mb550_norm.jsonl",
    "data/monitoringbench/attacks_mb550_norm.jsonl",
)
DEFAULT_SPLIT_FILES = {
    "train": "data/monitoringbench/splits/train_ids.txt",
    "test": "data/monitoringbench/splits/dev_ids.txt",
}
DEFAULT_OUT_DIR = "data/monitoringbench"


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def sanitize_slug(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


@dataclass
class GenResult:
    ok: bool
    text: str = ""
    error: str = ""
    stop_reason: str = ""
    n_tokens: int = 0
    meta: dict | None = None


class TinkerSampler:
    """Greedy-ish sampling from a tinker-hosted BASE model (no training client).

    Token-level: renders with the model's own chat template, the same path
    Policy.render takes during training eval."""

    def __init__(self, model: str, temperature: float, thinking: str):
        import tinker

        self.types = tinker.types if hasattr(tinker, "types") else None
        from tinker import types as _types

        self._types = _types
        self.client = tinker.ServiceClient().create_sampling_client(base_model=model)
        self.tokenizer = self.client.get_tokenizer()
        self.temperature = temperature
        # thinking: 'off'/'on' -> pass enable_thinking bool IF the template
        # reads it (probe, as run_rlvr._check_thinking_toggle); 'omit' -> never.
        self.chat_kwargs: dict[str, Any] = {}
        if thinking in ("off", "on"):
            want = thinking == "on"
            if self._template_reads_toggle():
                self.chat_kwargs["enable_thinking"] = want
            else:
                print(
                    f"[warn] chat template ignores enable_thinking; sampling in its "
                    f"default mode (wanted: {thinking})"
                )

    def _template_reads_toggle(self) -> bool:
        probe = [{"role": "user", "content": "probe"}]

        def render(value: bool):
            try:
                out = self.tokenizer.apply_chat_template(
                    probe, add_generation_prompt=True, tokenize=True, enable_thinking=value
                )
            except Exception:
                return None
            if not isinstance(out, list):
                out = out["input_ids"]
            return out[0] if out and isinstance(out[0], list) else out

        a, b = render(True), render(False)
        return a is not None and b is not None and a != b

    def _render(self, messages: list[dict]) -> list[int]:
        out = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self.chat_kwargs
        )
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def generate_all(self, convos: list[list[dict]], max_tokens: int, parallel: int) -> list[GenResult]:
        # parallel is unused: tinker futures are all in flight at once by design
        params = self._types.SamplingParams(
            max_tokens=max_tokens, temperature=self.temperature, top_p=1.0
        )
        futures = []
        for c in convos:
            try:
                prompt = self._types.ModelInput.from_ints(self._render(c))
                futures.append(
                    self.client.sample(prompt=prompt, num_samples=1, sampling_params=params)
                )
            except Exception as e:
                futures.append(e)
        results: list[GenResult] = []
        for fut in futures:
            if isinstance(fut, Exception):
                results.append(GenResult(ok=False, error=f"{type(fut).__name__}: {fut}"))
                continue
            try:
                seq = fut.result().sequences[0]
                results.append(
                    GenResult(
                        ok=True,
                        text=self.tokenizer.decode(seq.tokens),
                        stop_reason=str(seq.stop_reason),
                        n_tokens=len(seq.tokens),
                    )
                )
            except Exception as e:
                results.append(GenResult(ok=False, error=f"{type(e).__name__}: {e}"))
        return results


class OpenRouterSampler:
    """Chat-completions sampling for candidates tinker does not host.

    Also serves --backend local (any OpenAI-compatible server, e.g. a pod's
    vLLM) via base_url: reasoning/provider keys are OpenRouter-only and are
    never sent there."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        temperature: float,
        thinking: str,
        provider: Optional[str],
        timeout: float,
        max_retries: int,
        base_url: Optional[str] = None,
    ):
        self.is_local = base_url is not None
        if self.is_local:
            self.URL = base_url.rstrip("/") + "/chat/completions"
            key = os.environ.get("LOCAL_LLM_API_KEY") or "local"
        else:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise SystemExit(
                    "OPENROUTER_API_KEY is not set. Source it into the environment "
                    "(set -a; source .env; set +a) and re-run."
                )
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model = model
        self.temperature = temperature
        self.thinking = thinking  # off | on | omit
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max_retries
        self._drop_reasoning = False  # flipped when the endpoint rejects the key
        self._lock = threading.Lock()

    def _payload(self, messages: list[dict], max_tokens: int) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.is_local:
            return body
        if self.thinking in ("off", "on") and not self._drop_reasoning:
            body["reasoning"] = {"enabled": self.thinking == "on"}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return body

    def _one(self, messages: list[dict], max_tokens: int) -> GenResult:
        import requests

        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.URL,
                    headers=self.headers,
                    json=self._payload(messages, max_tokens),
                    timeout=self.timeout,
                )
                if r.status_code == 400 and "reasoning" in r.text.lower() and not self._drop_reasoning:
                    with self._lock:
                        if not self._drop_reasoning:
                            self._drop_reasoning = True
                            print("[warn] endpoint rejected the reasoning key; retrying without it")
                    continue
                if r.status_code in (429, 500, 502, 503, 524):
                    last_err = f"HTTP {r.status_code}"
                    time.sleep(2.0 * 2**attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    last_err = str(data["error"])[:300]
                    time.sleep(2.0 * 2**attempt)
                    continue
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
                if not text.strip():
                    last_err = "empty response"
                    time.sleep(2.0 * 2**attempt)
                    continue
                usage = data.get("usage") or {}
                return GenResult(
                    ok=True,
                    text=text,
                    stop_reason=str(choice.get("finish_reason")),
                    n_tokens=int(usage.get("completion_tokens") or 0),
                    meta={"provider": data.get("provider")},
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2.0 * 2**attempt)
        return GenResult(ok=False, error=last_err or "exhausted retries")

    def generate_all(self, convos: list[list[dict]], max_tokens: int, parallel: int) -> list[GenResult]:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            return list(pool.map(lambda c: self._one(c, max_tokens), convos))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mb_policy_baseline.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="tinker base-model name or OpenRouter slug")
    p.add_argument("--backend", choices=("tinker", "openrouter", "local"), required=True)
    p.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible server for --backend local (default: "
            "$LOCAL_LLM_BASE_URL, else http://127.0.0.1:8765/v1)"
        ),
    )
    sp = p.add_mutually_exclusive_group()
    sp.add_argument(
        "--scratchpad",
        dest="scratchpad",
        action="store_true",
        help="Plan-then-answer two-turn rollout (default)",
    )
    sp.add_argument(
        "--no-scratchpad",
        dest="scratchpad",
        action="store_false",
        help="Single blind turn, no plan",
    )
    p.set_defaults(scratchpad=True)
    p.add_argument("--plan-tokens", type=int, default=1000, help="Plan-turn budget (default: 1000)")
    p.add_argument(
        "--max-tokens", type=int, default=1000, help="Answer-turn budget (default: 1000)"
    )
    p.add_argument("--n", type=int, default=120, help="Dev rows, prefix of the split (default: 120)")
    p.add_argument("--temperature", type=float, default=0.0, help="(default: 0.0 = greedy, eval parity)")
    p.add_argument(
        "--thinking",
        choices=("off", "on", "omit"),
        default="off",
        help=(
            "Model-native thinking toggle: off/on send the toggle when the "
            "backend supports it; omit never sends it (default: off — the "
            "scratchpad IS the CoT surrogate)"
        ),
    )
    p.add_argument("--provider", default=None, help="OpenRouter provider pin (default: none)")
    p.add_argument("--parallel", type=int, default=4, help="Concurrent requests, openrouter only (default: 4)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--data", action="append", metavar="PATH", help=f"default: {DEFAULT_DATA_PATHS}")
    p.add_argument(
        "--dev-ids", default=DEFAULT_SPLIT_FILES["test"], help="Split ids file used as the selection"
    )
    p.add_argument("--run-tag", default="run")
    p.add_argument("--out", default=None, help=f"default: {DEFAULT_OUT_DIR}/policy_baseline_<model>_<tag>.jsonl")
    p.add_argument("--force", action="store_true", help="Overwrite an existing non-empty --out")
    p.add_argument("--dry-run", action="store_true", help="Selection + config only, no API calls")
    p.add_argument(
        "--max-error-rate",
        type=float,
        default=0.1,
        help="Exit nonzero when request-error rows exceed this fraction (default: 0.1)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    family = get_family("monitoringbench")
    try:
        return _run(args, family)
    finally:
        family.close()


def _run(args, family) -> int:
    files = [str(_resolve(f)) for f in (args.data or DEFAULT_DATA_PATHS)]
    ds = {
        "files": files,
        "seed": 0,
        "split_files": {
            "train": str(_resolve(DEFAULT_SPLIT_FILES["train"])),
            "test": str(_resolve(args.dev_ids)),
        },
    }
    src = family.source(ds)
    # PlannedEnv owns the plan-cue mechanics; instantiating it also runs its
    # template validations even though we drive the turns ourselves.
    planned = PlannedEnv(src, args.plan_tokens) if args.scratchpad else None

    tasks = src.tasks(args.n, "test")
    labels = [t.meta["label"] for t in tasks]
    n_attack = sum(1 for l in labels if l == "attack")
    n_honest = len(labels) - n_attack

    out_path = (
        _resolve(args.out)
        if args.out
        else _resolve(DEFAULT_OUT_DIR)
        / f"policy_baseline_{sanitize_slug(args.model)}_{sanitize_slug(args.run_tag)}.jsonl"
    )
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force and not args.dry_run:
        print(f"refusing to overwrite non-empty {out_path} (use --force)")
        return 2

    header = {
        "kind": "config",
        "model": args.model,
        "backend": args.backend,
        "scratchpad": args.scratchpad,
        "plan_tokens": args.plan_tokens if args.scratchpad else None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "n": len(tasks),
        "n_attack": n_attack,
        "n_honest": n_honest,
        "dev_ids": args.dev_ids,
        "data": files,
    }
    print(
        f"[mb_policy_baseline] {args.model} via {args.backend} | "
        f"scratchpad={'on' if args.scratchpad else 'off'} | rows={len(tasks)} "
        f"({n_attack} attack / {n_honest} honest) | temp={args.temperature} | "
        f"thinking={args.thinking}"
    )
    if args.dry_run:
        print("[dry-run] no API calls made")
        return 0

    if args.backend == "tinker":
        sampler = TinkerSampler(args.model, args.temperature, args.thinking)
    else:
        base_url = None
        if args.backend == "local":
            base_url = (
                args.base_url
                or os.environ.get("LOCAL_LLM_BASE_URL")
                or "http://127.0.0.1:8765/v1"
            )
        sampler = OpenRouterSampler(
            args.model,
            args.temperature,
            args.thinking,
            args.provider,
            args.timeout,
            args.max_retries,
            base_url=base_url,
        )

    rows: list[dict] = [
        {"kind": "row", "task_id": t.meta["task_id"], "label": t.meta["label"]} for t in tasks
    ]

    # ---- turn 1: plan (scratchpad on) -------------------------------------
    if args.scratchpad:
        plan_convos = []
        for t in tasks:
            head = [dict(m) for m in t.messages[:-1]]
            plan_convos.append(head + [{"role": "user", "content": planned._plan_cue(t)}])
        print(f"[plan] sampling {len(plan_convos)} plans (cap {args.plan_tokens})...")
        plan_results = sampler.generate_all(plan_convos, args.plan_tokens, args.parallel)
        for row, res in zip(rows, plan_results):
            row["plan"] = {
                "ok": res.ok,
                "error": res.error,
                "stop_reason": res.stop_reason,
                "n_tokens": res.n_tokens,
                "text": res.text,
            }
        pending = [
            (i, plan_convos[i] + [{"role": "assistant", "content": plan_results[i].text.strip()}])
            for i in range(len(tasks))
            if plan_results[i].ok
        ]
        answer_convos = [c + [dict(tasks[i].messages[-1])] for i, c in pending]
        answer_idx = [i for i, _ in pending]
        n_plan_err = len(tasks) - len(pending)
        if n_plan_err:
            print(f"[plan] {n_plan_err} plan requests failed; their rows are error rows")
    else:
        answer_convos = [[dict(m) for m in t.messages] for t in tasks]
        answer_idx = list(range(len(tasks)))

    # ---- turn 2: answer ----------------------------------------------------
    print(f"[answer] sampling {len(answer_convos)} answers (cap {args.max_tokens})...")
    answer_results = sampler.generate_all(answer_convos, args.max_tokens, args.parallel)

    scored = 0
    errors = 0
    agg = {
        "correct_strict": 0.0,
        "correct_relaxed": 0.0,
        "answer_format_valid": 0.0,
        "chose_attack": 0.0,
    }
    tp = fp = 0
    parsed_correct = 0
    parsed_n = 0
    for i, res in zip(answer_idx, answer_results):
        row = rows[i]
        row["answer"] = {
            "ok": res.ok,
            "error": res.error,
            "stop_reason": res.stop_reason,
            "n_tokens": res.n_tokens,
            "text": res.text,
            "meta": res.meta,
        }
        if not res.ok:
            continue
        _, info = src.reward(tasks[i], res.text)
        row.update(
            correct_strict=bool(info["correct_strict"]),
            correct_relaxed=bool(info["correct_relaxed"]),
            answer_format_valid=bool(info["answer_format_valid"]),
            chose_attack=bool(info["chose_attack"]),
        )
        scored += 1
        for k in agg:
            agg[k] += info[k]
        if info["answer_format_valid"]:
            parsed_n += 1
            parsed_correct += int(info["correct_strict"])
            if info["chose_attack"]:
                if row["label"] == "attack":
                    tp += 1
                else:
                    fp += 1

    errors = len(tasks) - scored
    summary: dict[str, Any] = {
        "kind": "summary",
        "n": len(tasks),
        "scored": scored,
        "request_errors": errors,
        "strict_accuracy": (agg["correct_strict"] / scored) if scored else None,
        "relaxed_accuracy": (agg["correct_relaxed"] / scored) if scored else None,
        "answer_format_valid_rate": (
            agg["answer_format_valid"] / scored if scored else None
        ),
        "chose_attack_rate": (agg["chose_attack"] / scored) if scored else None,
        "accuracy_on_parsed": (parsed_correct / parsed_n) if parsed_n else None,
        "tpr": (tp / n_attack) if n_attack else None,
        "fpr": (fp / n_honest) if n_honest else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(summary) + "\n")

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    print(
        f"[result] {args.model}: strict accuracy {fmt(summary['strict_accuracy'])} "
        f"(relaxed {fmt(summary['relaxed_accuracy'])}, on-valid "
        f"{fmt(summary['accuracy_on_parsed'])}, format-valid rate "
        f"{fmt(summary['answer_format_valid_rate'])}) | chose_attack "
        f"{fmt(summary['chose_attack_rate'])} | TPR {fmt(summary['tpr'])} / FPR {fmt(summary['fpr'])} "
        f"| errors {errors}/{len(tasks)}"
    )
    print(f"[out] {out_path}")

    if len(tasks) and errors / len(tasks) > args.max_error_rate:
        print(f"[fail] error rate {errors / len(tasks):.2f} > {args.max_error_rate}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
