# Blackwell (1xB200, sm100) bring-up smoke — ~1 hour of paid pod time

Validates `scripts/provision_blackwell.sh` and the training stack on sm100
before any real run is scheduled there. Every step has a pass condition and a
time box; if a step blows its box, stop the pod and bring the logs home — the
hour is for evidence, not for debugging on the meter.

Prereqs (local): `~/.runpod/config.toml` with a valid apikey; the repo clean
enough to rsync; the Blackwell research dossier resolved against the
`# BLACKWELL-VERIFY:` tags in `provision_blackwell.sh` (the flash-attn pin and
the 10.0-vs-10.0a question change what step 1 does).

Cost expectation: B200 secure-cloud has listed around $6/GPU-hr (check the
console at rent time — treat anything over ~$8 as a mispick). Budget: 1xB200
smoke ≈ $6-8; the 2x extension ≈ $12-16/hr for the ~20 min it needs. Volume
storage bills separately and survives the pod.

## Pre-flight (dossier 2026-08-08)

- Host driver must be >= 580 (`nvidia-smi | head -3`) — CUDA 13 floor; a lower
  driver fails at first kernel, not at install.
- The venv is `/workspace/envs/verl-b200` (NOT verl-main — that one carries an
  sm90 flash-attn). Launches: `PY=/workspace/envs/verl-b200/bin/python bash
  scripts/pod_run.sh ...`.
- flash-attn is skipped by default; expected attention backends in vllm logs:
  FLASHINFER (or the bundled FA path) on sm100. Grep the serve log for
  `Using .* attention backend`.
- A 1x smoke CANNOT validate TP2/NVLink; the TP2 arm needs an actual 2xB200
  allocation (NV18 in `nvidia-smi topo -m`).

## 0. Rent + restore (10 min box)

```bash
# local — B200 instead of the H200 default; cu1300 image is REQUIRED
# (provision fatals without a CUDA 13.x nvcc)
GPU_TYPE="NVIDIA B200" bash scripts/pod_create.sh b200-smoke
```

- Volume caveat: the existing volume (verl env, `/workspace/hf`,
  `/workspace/models/olmo32-bf16`) is pinned to its datacenter. If no B200
  capacity exists there, the pod comes up volume-less: `/workspace` is then
  container disk, everything downloads cold, and step 3 needs `HF_HOME`
  pointed somewhere with ~10GB free. Note which case you are in — it changes
  the pass/fail reading of every timing below.
- After EVERY pod start (stop wipes `/root`, including runpodctl auth):
  ```bash
  # on the pod
  runpodctl config --apiKey <key>
  export RUNPOD_POD_ID=<id>
  ```
  Without both, `pod_run.sh` refuses to launch (exit 6, watchdog auth guard).
  Do NOT export the pod-scoped `RUNPOD_API_KEY` from `/proc/1/environ` — it
  overrides `config.toml` and breaks `runpodctl get pod`, tripping the same
  guard.
- Sync the repo: `bash scripts/pod_sync.sh <ip> <port>` (direct SSH ip:port,
  not the ssh.runpod.io proxy — the proxy swallows non-interactive commands).

## 1. Provision (25 min box)

```bash
# on the pod — FLASH_ATTN_SKIP=1 on the FIRST pass keeps this inside the box;
# a flash-attn source attempt can add up to 2h and is a separate experiment
VOL=/workspace FLASH_ATTN_SKIP=1 bash /root/debate/scripts/provision_blackwell.sh
```

Pass: `DONE (flash-attn: skip)` and, above it, the sanity block printing
`capability (10, 0)`, torch's arch list, and `bf16 matmul on NVIDIA B200 ok`.
Record the `flash-attn PATH:` line and the arch list verbatim — both go in the
dossier. If the dossier pinned a Blackwell flash-attn, drop `FLASH_ATTN_SKIP`
and pass its pin via `FLASH_ATTN_PIN=` (or `FLASH_ATTN_WHEEL=` for a wheel
built on a previous sm100 pod — never an sm90 one).

## 2. (a) torch CUDA sanity (2 min box)

Provision already ran this; re-run standalone only if you need it isolated:

```bash
/workspace/envs/verl-main/bin/python - <<'PY'
import torch
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
a = torch.randn(1024, 1024, device="cuda")
b = a.to(torch.bfloat16)
print("fp32", (a @ a).sum().item(), "| bf16", (b @ b).float().sum().item())
PY
```

Pass: capability `(10, 0)`, both matmuls print finite numbers.
Fail signature: `no kernel image is available for execution on the device` —
the torch wheel lacks sm_100; the pins need the dossier, nothing downstream is
worth running.

## 3. (b) vllm serve + 1-token completion (15 min box)

Same launch shape as `pod_run.sh`'s judge server, same real-completion probe
(a dead EngineCore keeps answering `/v1/models`; only a generate proves life):

```bash
export HF_HOME=/workspace/hf   # volume-less pod: use /root/hf and expect a cold ~8GB pull
/workspace/envs/verl-main/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-4B --port 8100 \
  --gpu-memory-utilization 0.18 --max-model-len 16384 \
  --max-num-seqs 32 --enable-prefix-caching > /root/vllm_smoke.log 2>&1 &

# poll until it answers (≤900s warm, longer only on a cold pull)
curl -fsS -m 30 http://127.0.0.1:8100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.5-4B", "prompt": "1+1=", "max_tokens": 1}'
```

Pass: a JSON body whose `choices[0].text` exists (an error body also contains
the string "choices" — read the JSON, don't grep it).

Then record the attention backend vllm chose — the single most important
datum this smoke produces on the skip path:

```bash
grep -iE "attention backend|flashinfer|flash[_-]?attn|triton" /root/vllm_smoke.log
```

Kill the server (whole process group — a parent-only kill leaves EngineCore
children holding their GPU allocation): `pkill -9 -f vllm.entrypoints`.

## 4. (c) vllm serve TP2 — only if 2 GPUs (10 min box)

Same command plus `--tensor-parallel-size 2` (fresh port, fresh log). Pass:
same probe. This is the arm that exercises NCCL on sm100; a hang at
"Waiting for NCCL" beyond ~5 min is the finding — capture the log and move on.

## 5. (d) repo smoke arm (20 min box)

```bash
cd /root/debate && /workspace/envs/verl-main/bin/python -m pip install -q -e . --no-deps
bash scripts/pod_run.sh rlvr math_rlvr_olmo_smoke     # 1 GPU, configs/math_rlvr_olmo.yaml
```

- `pod_run.sh` exports `HF_HOME=/workspace/hf` itself; the OLMo-3-7B pull is
  ~15GB if the volume cache is absent.
- Prereq recap: step 0's `runpodctl config` + `RUNPOD_POD_ID`, or this exits 6.
- Pass: 2 steps complete, `kl k1 ≈ 0 at step 0` (the engine-consistency
  canary — ~1.5 there means the transformers/rope patches did not land), no
  zero-datum steps. On the H100 this smoke's steps ran ~125s at batch 2 ×
  group 4; note the B200 number for the dossier.
- 2xB200 extension: `bash scripts/pod_run.sh rlvr aime_rlvr_olmo32_smoke`
  (2 GPUs, TP2 rollout, `model: /workspace/models/olmo32-bf16`). Requires the
  volume: the model MUST be the pre-converted bf16 copy — see failure
  signatures. On a volume-less pod skip this arm; there is no time in the box
  to re-download and re-convert 121GB.
- Optional if minutes remain: `python scripts/mb_verl_memprobe.py` (worst-case
  gen/train/wake cycle; the wake phase is what catches step-boundary OOMs).
  On 180GB the H100-tuned fractions (0.45 rollout / 0.18 judge) leave ~2.2x
  the byte headroom, so a pass is expected — a FAIL is a real finding.

## 6. (e) Teardown (5 min box)

```bash
# on the pod: flush anything worth keeping to /workspace or rsync home FIRST —
# stop wipes the container disk (/root: repo, logs, runpodctl auth)
runpodctl stop pod $RUNPOD_POD_ID
```

Then in the console: verify the pod shows stopped (an accepted stop can lag
minutes), and **terminate** it if there is no follow-up session planned — a
stopped pod still bills its container-disk reservation. Sanity-check the
bill: 1x smoke ≈ 1 GPU-hr ≈ $6-8; anything larger means the pod outlived the
session.

## Known failure signatures (from the H200/H100 bring-up — expect the same shapes)

| Signature | Meaning | Action |
|---|---|---|
| `RuntimeError: cancelled` from rollout | Worker silently OOM-killed — vLLM 0.24 has no init timeout, a dead worker is the ONLY way to this error. Seen twice loading the fp32 121GB OLMo-32B under TP2. | `dmesg | grep -i oom`; use `/workspace/models/olmo32-bf16` (61GB, boots clean) — never the HF fp32 shards |
| `no kernel image is available for execution on the device` | A wheel (torch/vllm/flash-attn) carries no sm_100 code | Identify which via the failing import/launch; that pin goes back to the dossier |
| OOM in `create_and_map` at wake-after-train | Trainer allocator cache squatting on the memory vLLM re-pins. Fixed by the `sync_sampler` flush (commit bb83923) + `expandable_segments`; should not reproduce with 180GB headroom | If it does anyway, that IS the finding — run `mb_verl_memprobe.py` and capture |
| `CheckpointError` (saved bf16, recomputed fp32) | `modeling_olmo3.py` rope-dtype patch missing | Re-run provision (patches are idempotent) |
| undefined-symbol on `import vllm` | torch moved after vllm was installed (ABI tie) | `uv pip install --reinstall vllm==0.24.*`; never up/downgrade torch in place |
| `kl k1 ~1.5` at step 0 | transformers <5.13 YaRN regression, or the rope shims absent | Check `transformers.__version__` ≥5.13, re-run provision |
| server answers `/v1/models` but every completion fails | EngineCore dead behind a live api_server | Real-completion probe (step 3) is the only valid health check |
| checkpoint fills the disk (~55GB per save) | Full-state save instead of lora-only | `save_lora_only` is set by `infra/backend/verl.py` when `lora_rank>0`; a full save means lora_rank=0 got configured |

## Bring home for the dossier

The `flash-attn PATH:` line; torch arch list + capability; the vllm attention
backend line from step 3; TP2 pass/fail + NCCL notes; smoke step time vs the
H100's ~125s; any signature from the table above with its log.
