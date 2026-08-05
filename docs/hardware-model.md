# A mental model for GPU cost: training and serving

**Status: working notes, incomplete.** This is not a finished reference and was
not meant to be written in one pass — it is the place to accumulate worked
arithmetic and hard-won corrections as they come up, so they do not have to be
rediscovered. Add to it when a number surprises you; delete from it when
something here turns out to be wrong.

General, not specific to one experiment. Numbers are worked for a 32B bf16 model
on H200 (143 GB, ~4.8 TB/s HBM) because that is what we run, but the *shapes*
are what matter.

Known gaps: no MoE, no quantisation, no multi-node (all TP here is intra-node),
no attention-variant detail (MQA/GQA/sliding window change the KV numbers), no
measured MFU.

---

## 1. What actually occupies a GPU

Four things, and they scale differently:

| what | size | scales with |
|---|---|---|
| **weights** | `params x bytes` | model only |
| **gradients** | `params x bytes` | model only (training) |
| **optimizer state** | Adam: `params x 8B` (two fp32 moments) | model only (training) |
| **activations** | `batch x seq x hidden x layers x bytes` | **batch and sequence** |
| **KV cache** (serving) | `2 x layers x kv_heads x head_dim x seq x concurrency x bytes` | **context and concurrency** |

For a 32B:

```
weights bf16                       59.6 GB
gradients bf16                     59.6 GB
Adam m+v fp32 (full finetune)     238.4 GB
                                  --------
full finetune, at rest            357.6 GB   -> 2.5 H200s before any activations

LoRA r32 adapter                    0.2 GB
LoRA total (frozen base)           59.6 GB   -> 7.5 GB/card at TP=8
```

**This is the single most useful ratio to hold in your head: full finetuning a
model costs ~6x its weights; LoRA costs ~1x.** Optimizer state, not weights, is
what puts full finetuning out of reach. Everything else about our setup follows
from choosing LoRA.

The first three are *fixed* once you pick a model. Activations and KV cache are
the ones you control at runtime, and they are where OOMs actually come from.

---

## 2. Loading: disk to CPU to GPU

Three hops, each with a different bandwidth:

```
disk/network  ->  CPU RAM  ->  GPU HBM
   ~1-10 GB/s     ~25-50 GB/s (PCIe4/5 x16)   ~4.8 TB/s (on-card)
```

Loading a 120 GB checkpoint is dominated by the **first** hop. Once resident,
weights never move again — which is why model load is a one-off minutes-long
cost and then irrelevant.

The rule that follows: **avoid host<->device traffic in the steady state.**
Anything crossing PCIe per step (moving logits to CPU, syncing optimizer state)
is ~100x slower than staying on-card. Our per-step CPU work is the verifier,
which is CPU-only by nature — but it *blocks* the step, so GPUs idle through it.

---

## 3. Parallelism: four schemes, distinguished by what they communicate

| scheme | splits | communicates | when |
|---|---|---|---|
| **Data (DP)** | the batch | gradients, once per step | model fits on one card |
| **Tensor (TP)** | each weight matrix | activations, **every layer** | model does not fit |
| **Pipeline (PP)** | layers into stages | activations between stages | very deep models |
| **FSDP / ZeRO** | params+grads+optimizer across DP ranks | all-gather params **per layer** | trade bandwidth for memory |

The useful distinction is **communication frequency**:

- DP talks **once per step** -> tolerates slow interconnect, scales across nodes.
- TP talks **many times per token** -> needs NVLink; do not span nodes.
- FSDP talks **per layer** -> between the two; it is DP that gave up memory
  redundancy to fit bigger models.

TP is what lets a 60 GB model live on cards that each hold 143 GB *with room for
KV cache*: at TP=8 each card holds 7.5 GB of weights. Without it, one card must
hold all 60 GB and there is nothing left to cache with.

**The cost of TP is that every card is busy on every token.** You cannot overlap
two rollouts on a TP=8 group; the whole group is one logical engine.

---

## 4. Serving: prefill vs decode, and why batching is everything

Generation has two phases with opposite bottlenecks:

**Prefill** (process the prompt): all tokens at once, big matmuls, **compute-bound**.
Scales with `prompt_tokens`.

**Decode** (emit one token): reads *every weight* to produce *one token* per
sequence. **Memory-bandwidth-bound.**

```
one decode step reads 60 GB of weights / 4.8 TB/s = 13.3 ms
  batch   1 -> 13.33 ms per sequence
  batch   8 ->  1.67 ms per sequence
  batch  64 ->  0.21 ms per sequence
```

The weight read is paid **once per step regardless of batch size**. That is the
whole reason batching exists: at batch 1 you use ~1/64th of the hardware. It is
also why *latency* and *throughput* pull in opposite directions — a single
user's token latency is ~13 ms no matter what, and batching improves tokens/sec
without improving time-to-first-token for anyone.

**KV cache is what limits how much you can batch:**

```
per token: 2(K,V) x 64 layers x 8 kv-heads x 128 dim x 2B = 256 KB
  ctx 12288 x   8 concurrent =  24 GB
  ctx 12288 x  64 concurrent = 192 GB
  ctx 12288 x 256 concurrent = 768 GB
```

Context length and concurrency multiply. Doubling max context halves how many
sequences fit. This is the actual meaning of `gpu_memory_utilization`: it is the
fraction of the card handed to the serving engine for **weights + KV cache**,
and if weights eat all of it there is no cache and the engine refuses to start.

---

## 5. The training step, end to end

```
  sync weights   trainer -> inference engine (only the trainable delta; for
                 LoRA that is ~0.2 GB, not 60 GB)
  generate       prefill + decode, bandwidth-bound, TP across the group
  score          reward. CPU here. GPUs IDLE for its whole duration.
  pack           form micro-batches by TOKEN COUNT, not sequence count
  forward        activations grow with batch x seq
  backward       gradients, ~2x forward cost
  reduce         all-reduce (DP) or all-gather+reduce-scatter (FSDP)
  optim step     apply to the trainable params only
```

Two things to internalise:

**Micro-batching is by tokens, not sequences.** A token budget per GPU bounds
peak activation memory. Raising the logical batch size makes *more micro-batches*
at the same peak memory — it costs wall-clock, not OOM. That is why batch size
is a throughput dial and sequence length is a memory dial.

**Sequence length is quadratic in attention and linear in everything else.**
Doubling context more than doubles activation cost. When a step OOMs, the
sequence length is the first thing to look at, not the batch size.

---

## 6. Coordination: what the CPU is actually doing

The CPU never does math. It:

- **launches kernels** — GPU work is queued asynchronously; the CPU can run ahead
- **runs collectives' control plane** — the actual all-reduce is GPU-to-GPU
- **holds the dataloader and any non-GPU scoring**

Two consequences worth remembering:

**GPU calls are async, so timing them naively lies.** `t=time(); gpu_op(); time()-t`
measures the *launch*, not the work, unless you synchronise.

**Collectives are barriers.** All ranks must arrive. One slow rank stalls all of
them, so a straggler costs `n_gpus x delay`, not `delay`.

---

## 7. Where limits come from, and how to read them

A limit is only meaningful in the currency it bounds. The one we got wrong:

`RLIMIT_AS` bounds **virtual address space**, not resident memory. Our verifier
sets it to 4 GB per subprocess, and concurrency was capped at 8 on the reasoning
that `8 x 4 GB = 32 GB`. But measured RSS is **~21 MB** per process — three
orders of magnitude less. The cap was computed in virtual bytes and spent as if
they were physical ones, which cost ~3x throughput for no safety gain. 4 GB is
fine *as a backstop* (it stops a runaway `[0]*10**9`); using it to derive a
concurrency budget was the error.

Related traps:

- `free` / `nproc` report the **host**; `/sys/fs/cgroup/memory.max` and `cpu.max`
  report **your container**. A 1-GPU pod advertised 2 TB / 208 cores while its
  cgroup allowed 234 GB / 22 CPUs.
- `RLIMIT_AS` cannot be raised on macOS at all — a dev-box "limit" that silently
  does nothing.
- When raising a concurrency knob does nothing, look for a **second** limit
  underneath. Ours had a semaphore below the thread pool; three measurements
  said "concurrency does not help" when concurrency had never changed.

---

## Quick reference

| question | first thing to compute |
|---|---|
| will this model fit for training? | `params x 12B` (full) or `params x 2B` (LoRA) |
| will it fit for serving? | `params x 2B + KV(ctx, concurrency)` |
| why is generation slow? | `weights_bytes / HBM_bandwidth` = floor per decode step |
| why did batch size OOM? | it probably did not — check sequence length |
| why doesn't more parallelism help? | find the *binding* limit; there is usually a second one |
| is this CPU cost free? | no, if it blocks a step: idle GPUs are the real price |
