"""Per-token sampler-vs-trainer logprob diagnostic for the verl backend.

Smoke showed kl/policy_vs_ref_k1 ~ 1.5 at step 0 where LoRA-at-init implies
policy == ref (tinker shows ~3e-4). That k1 is mean(sampler_lp - fsdp_lp), so
one of the two paths is wrong and the PPO ratio anchor is broken. This replays
the exact code paths on one prompt and prints the per-token pattern:
uniform offset -> different weights/processing; spiky boundary -> alignment;
shift-by-one match -> extraction offset.

Run on the pod:
  cd /root/debate && PYTHONPATH=. HF_HOME=/workspace/hf \
    /workspace/envs/verl-main/bin/python scripts/diag_logprob.py
"""

import statistics

from infra.backend.base import Datum, LossSpec, SamplingParams
from infra.backend.verl import VerlBackend, VerlBackendConfig

MODEL = "allenai/Olmo-3-7B-Instruct-DPO"

backend = VerlBackend(
    VerlBackendConfig(
        model_path=MODEL,
        n_gpus=1,
        strategy="fsdp2",
        gpu_memory_utilization=0.33,
        prompt_length=8192,
        response_length=1024,
        max_token_len_per_gpu=16384,
        rollout_tp=1,
        lora_rank=32,
        lr=1e-5,
        loss=LossSpec(kind="ppo", clip_low=0.8, clip_high=1.2),
    )
)
tok = backend.tokenizer
# Long context on purpose: OLMo-3 has sliding_window=4096 on 3/4 layers. The
# short-prompt run agreed (mean diff 0.04); if the FSDP rmpad path ignores the
# window while vLLM honors it, divergence should appear only past ~4k tokens.
filler = (
    "Lemma %d: For any positive integer n, the sum of the first n odd numbers "
    "equals n squared, which one can verify by induction on n with base case 1. "
)
long_context = "".join(filler % i for i in range(220))
msgs = [
    {
        "role": "user",
        "content": long_context
        + "\n\nNow: prove that the sum of the first n odd numbers is n^2. Walk through it carefully.",
    }
]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
if hasattr(prompt, "input_ids") or isinstance(prompt, dict):
    prompt = prompt["input_ids"]
prompt = list(prompt)

samples = backend.sample([prompt], SamplingParams(max_tokens=300, temperature=1.0, top_p=1.0), n=1)
s = samples[0][0]
n = len(s.tokens)
print(f"stop={s.stop_reason} n_tokens={n} n_logprobs={len(s.logprobs)}")
assert len(s.logprobs) == n, "sampler logprobs misaligned with tokens"

datum = Datum(
    tokens=prompt + list(s.tokens),
    prompt_len=len(prompt),
    sampler_logprobs=list(s.logprobs),
    advantages=[0.0] * n,
    mask=[1.0] * n,
)
fwd = backend.forward([datum])[0]
ref = backend.ref_logprobs([datum])[0]
assert len(fwd) == n, f"fwd len {len(fwd)} != {n}"

diffs = [sl - f for sl, f in zip(s.logprobs, fwd)]
print(f"mean sampler lp: {statistics.mean(s.logprobs):.4f}")
print(f"mean fsdp fwd lp: {statistics.mean(fwd):.4f}")
print(f"mean diff (sampler-fwd): {statistics.mean(diffs):.4f}   <- ~k1 at step 0")
print(f"mean |diff|: {statistics.mean([abs(d) for d in diffs]):.4f}")
print(f"max |diff|: {max(abs(d) for d in diffs):.4f} at t={max(range(n), key=lambda t: abs(diffs[t]))}")
print("first 10 sampler:", [round(x, 3) for x in s.logprobs[:10]])
print("first 10 fwd:    ", [round(x, 3) for x in fwd[:10]])
print("first 10 diff:   ", [round(x, 3) for x in diffs[:10]])
print("last 10 diff:    ", [round(x, 3) for x in diffs[-10:]])
sh_p1 = statistics.mean([abs(a - b) for a, b in zip(s.logprobs[1:], fwd[:-1])])
sh_m1 = statistics.mean([abs(a - b) for a, b in zip(s.logprobs[:-1], fwd[1:])])
print(f"mean|diff| if fwd shifted +1: {sh_p1:.4f}  -1: {sh_m1:.4f}")
print(f"fwd-vs-ref mean diff (should be ~0, adapter zero): {statistics.mean([f - r for f, r in zip(fwd, ref)]):.6f}")

import json

with open("/root/diag_tokens.json", "w") as f:
    json.dump(
        {"prompt": prompt, "tokens": list(s.tokens), "sampler_lps": list(s.logprobs), "fwd_lps": fwd},
        f,
    )
print("dumped /root/diag_tokens.json")
