"""Worst-case memory probe for MB RLVR on verl — run BEFORE the first smoke.

MB transcripts run to ~58k prompt tokens vs math's ~1k, and a smoke's random
rows can all be short: passing it proves nothing about the row that OOMs at
step 37. This probe deliberately selects the LONGEST rows in the pool and
walks them through both memory regimes of a real step:

  1. AUDIT (CPU-only): render every pool row through the real blind-choice
     templates and the model's chat template; report the prompt-token
     distribution and hard-fail listing any row over training.verl
     prompt_length (ids and token counts only — never contents).
  2. GEN (--gpu): vLLM rollout on the top-K longest prompts at the config's
     response budget. vLLM self-throttles concurrency against its KV pool, so
     this mostly proves the engine accepts max-length sequences at all.
  3. TRAIN (--gpu): ONE forward_backward + optim_step on K datums padded to
     the full prompt_length + response_length — the packing-unit worst case
     (max_token_len_per_gpu must fit one such datum). This is the phase math
     never exercised past 16384 tokens; it is the real OOM risk.

Peak GPU memory is sampled via nvidia-smi in a background thread (robust to
verl worker processes), reported per phase.

Usage (pod):   python scripts/mb_verl_memprobe.py --gpu
Usage (local): python scripts/mb_verl_memprobe.py          # audit only

HARD SAFETY RULE (inherited from the task module): the data files contain
red-team attack trajectories. Nothing here may print, log, or embed
trajectory or prompt CONTENT — rows are identified by id and token count.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GpuPeak:
    """Background nvidia-smi sampler; .peak_mib is the max seen since reset."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return max((int(x) for x in out.stdout.split()), default=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_mib = max(self.peak_mib, self._sample())
            except Exception:
                pass  # transient nvidia-smi failure: keep sampling
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-file", default="configs/mb_rlvr.yaml")
    parser.add_argument("--experiment", default="mb_rlvr_olmo")
    parser.add_argument("--top-k", type=int, default=4, help="longest rows to probe on GPU")
    parser.add_argument("--gpu", action="store_true", help="run the gen + train phases (pod)")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=None,
        help="override training.verl.gpu_memory_utilization for this probe run",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from infra.config import load_experiment
    from infra.envs.tasks import get_family
    from infra.run_rlvr import validate_experiment

    exp = load_experiment(args.experiment_file, args.experiment)
    validate_experiment(exp)
    v = exp["training"]["verl"]
    if args.gpu_memory_utilization is not None:
        v["gpu_memory_utilization"] = args.gpu_memory_utilization
        print(f"OVERRIDE gpu_memory_utilization={args.gpu_memory_utilization}")
    prompt_cap, resp_cap = int(v["prompt_length"]), int(v["response_length"])

    ds = dict(exp["dataset"])
    ds.pop("type")
    family = get_family("monitoringbench")
    try:
        _run_probe(args, exp, prompt_cap, resp_cap, ds, family, AutoTokenizer)
    finally:
        family.close()


def _run_probe(args, exp, prompt_cap, resp_cap, ds, family, tokenizer_cls) -> None:
    env = family.source(ds)

    # ---- phase 1: audit — every pool row, real templates, real chat template
    tokenizer = tokenizer_cls.from_pretrained(str(exp["model"]))

    def render_len(messages) -> int:
        # Policy.render's unwrap: some tokenizer wrappers return a
        # BatchEncoding (len() = its KEY COUNT, silently ~2) or a nested list.
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return len(out)

    all_tasks = [env._task(r) for r in env.rows]
    lengths = sorted((render_len(t.messages), t.meta["task_id"]) for t in all_tasks)
    n = len(lengths)
    print(f"pool: {n} rows | prompt tokens min/median/max: "
          f"{lengths[0][0]}/{lengths[n // 2][0]}/{lengths[-1][0]} | cap {prompt_cap}")
    over = [(tid, ln) for ln, tid in lengths if ln > prompt_cap]
    if over:
        for tid, ln in over:
            print(f"  OVER CAP: {tid} = {ln} tokens", file=sys.stderr)
        sys.exit(f"{len(over)} row(s) exceed prompt_length={prompt_cap} — raise the cap "
                 "or drop the rows; do not launch training")
    if lengths[-1][0] < 4096:
        sys.exit(
            f"suspiciously short: longest rendered prompt is {lengths[-1][0]} tokens, but MB "
            "transcripts run to tens of thousands — the render path is broken; fix before trusting "
            "this audit"
        )
    print(f"audit OK: longest row {lengths[-1][1]} = {lengths[-1][0]} tokens fits the cap")

    if not args.gpu:
        print("(audit-only; rerun with --gpu on the pod for the gen + train phases)")
        return

    # ---- GPU phases: real backend from the real training block
    from infra.backend.base import Datum, LossSpec, OptimParams, SamplingParams
    from infra.envs.base import Policy
    from infra.run_debate import build_backend

    top = [tid for _, tid in lengths[-args.top_k:]]
    by_id = {t.meta["task_id"]: t for t in all_tasks}
    tasks = [by_id[tid] for tid in top]
    print(f"probing top-{len(tasks)} longest rows: {top}")

    backend = build_backend(exp["training"], str(exp["model"]), run_name="memprobe")
    backend.sync_sampler()
    policy = Policy(backend, SamplingParams(max_tokens=resp_cap, temperature=1.0, top_p=1.0))

    with GpuPeak() as peak:
        results = policy.predict([t.messages for t in tasks], n=1)
    gen_lens = [len(s[0].tokens) for s in results]
    print(f"gen OK: completion tokens {gen_lens} | peak GPU {peak.peak_mib} MiB")

    # Pad every completion to the full response budget: the packing-unit worst
    # case (prompt_cap + resp_cap tokens through one forward_backward). Token
    # VALUES don't affect memory; logprob 0 makes the ppo ratio 1.
    pad_id = tokenizer.eos_token_id
    datums = []
    for (samples, task) in zip(results, tasks):
        s = samples[0]
        completion = (s.tokens + [pad_id] * resp_cap)[:resp_cap]
        datums.append(Datum(
            tokens=s.prompt_tokens + completion,
            prompt_len=len(s.prompt_tokens),
            sampler_logprobs=[0.0] * len(completion),
            advantages=[1.0] * len(completion),
        ))
    loss = LossSpec(**(exp["training"].get("loss") or {}))
    with GpuPeak() as peak:
        backend.forward_backward(datums, loss)
        backend.optim_step(OptimParams(lr=float(exp["training"]["lr"])))
    lens = sorted(len(d.tokens) for d in datums)
    print(f"train OK: {len(datums)} datums, {lens[0]}-{lens[-1]} tokens | "
          f"peak GPU {peak.peak_mib} MiB")

    # The transition the first smoke died on (2026-08-05): waking the slept
    # vLLM AFTER a train step re-pins its ~32GB against whatever the training
    # allocator still holds. A probe that stops at optim_step certifies
    # nothing about a real step boundary — this phase is the point.
    with GpuPeak() as peak:
        backend.sync_sampler()
        wake_out = policy.predict([tasks[0].messages], n=1)
    print(f"wake+resample OK: {len(wake_out[0][0].tokens)} completion tokens | "
          f"peak GPU {peak.peak_mib} MiB")
    print("memprobe PASSED — smoke is safe to launch at these lengths")


if __name__ == "__main__":
    main()
