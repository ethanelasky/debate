#!/usr/bin/env bash
# Provision a self-contained verl-main env for the debate VerlBackend.
# Modeled on the prior repo's run-verified provision.sh: the pod image only
# supplies the NVIDIA driver + nvcc; python/torch/vllm live in a uv-managed
# env (persistent iff $VOL is a network volume; otherwise pod-lifetime).
#
# Usage (on the pod):  VOL=/workspace bash provision_pod.sh
#
# Hard-won pins (2026-07-31, first bring-up on 2x RTX 4090 / 62GB RAM):
#  * vllm==0.24.* — the tuple verl main's CI actually tests (image vllm024.*).
#    An unpinned resolve pulls the newest vllm and with it a torch so new that
#    flash-attn has NO prebuilt wheel (wheel matrix stops at torch 2.10).
#  * flash-attn builds from source against torch 2.11+cu130. Single-arch
#    (TORCH_CUDA_ARCH_LIST from nvidia-smi) + MAX_JOBS=2: cutlass units spike
#    >15GB RSS each — MAX_JOBS=4+ gets OOM-Killed on 62GB, nproc is fatal.
#  * Never downgrade torch after vllm is installed: vllm's compiled extension
#    is ABI-tied to the exact torch it resolved with (undefined-symbol import
#    errors otherwise). Repair = `uv pip install --reinstall vllm==<pin>`.
set -euo pipefail

VERL_PIN="${VERL_PIN:-e9618406de5bad40041d7612554e465ec2003ec1}"
# 0.24 + the olmo2.py shim below. An in-place 0.26 upgrade was attempted
# (native per-layer rope support, same torch pin) but died in flash-attn
# 2.8.3's cute/cutlass interface ('cutlass.cute.core' has no 'ThrMma') after
# two other dep skews; retry 0.26 only as a FRESH provision with a jointly
# resolved env, never in-place.
VLLM_PIN="${VLLM_PIN:-vllm==0.24.*}"
MAX_JOBS="${MAX_JOBS:-2}"
VOL="${VOL:-/workspace}"
ENV_DIR="$VOL/envs/verl-main"
export HF_HOME="${HF_HOME:-$VOL/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VOL/uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VOL/uv/python}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

log() { echo -e "\n=== [provision] $* ==="; }

log "GPU / driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "FATAL: no nvidia-smi"; exit 1; }

# Single-arch flash-attn build: ~5x less compile work and memory than the
# default multi-arch fan-out.
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

# Toolkit preference (2026-08-02, learned overnight): a REAL system CUDA
# toolkit whose major matches torch's (cu130 -> 13.x) is the only reliable
# build env — use an image like runpod/pytorch:*-cu1300-*. The pip-shipped
# nvidia/cu13 fragments version-skew against each other (nvcc vs cccl vs
# ptxas) and are a LAST resort only.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
if ! nvcc --version 2>/dev/null | grep -q "release 13"; then
  VENV_CU="$ENV_DIR/lib/python3.12/site-packages/nvidia/cu13"
  if [ -x "$VENV_CU/bin/nvcc" ]; then
    log "WARNING: system nvcc missing/not 13.x; falling back to pip toolchain fragments (skew-prone)"
    export CUDA_HOME="$VENV_CU"; export PATH="$CUDA_HOME/bin:$PATH"
  else
    echo "FATAL: no CUDA 13.x nvcc (use a runpod/pytorch:*-cu1300-* image)"; exit 1
  fi
fi

command -v uv >/dev/null 2>&1 || {
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

[ -x "$ENV_DIR/bin/python" ] || { log "creating venv (python 3.12)"; uv venv "$ENV_DIR" --python 3.12 --seed; }
P="$ENV_DIR/bin/python"

log "vllm ($VLLM_PIN) + verl @ ${VERL_PIN:0:7} (joint resolve)"
uv pip install -p "$P" "$VLLM_PIN" \
  "verl @ git+https://github.com/volcengine/verl@${VERL_PIN}" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "datasets>=3.1" "wandb>=0.18" jinja2 codetiming \
  backoff "docent-python>=0.1.74" "pydantic>=2.0" "openai>=1.0" pyyaml \
  "transformers>=5.13,<6"
# transformers >=5.13 is a HARD floor: 5.10-5.12 misapply OLMo-3's YaRN factor
# to sliding-attention layers (HF regression #39847, fixed in #46911), which
# skews trainer logprobs ~1 nat/token vs vLLM past 4096 ctx and silently
# breaks the PPO ratio anchor (shows up as kl k1 ~1.5 at step 0).

# vllm 0.24 reads the pre-5.13 flat rope_parameters schema for olmo2/3 and
# KeyErrors on the new per-layer-type dict; shim it to accept both (same
# semantics: yarn on full-attention layers, default rope on sliding).
log "patching vllm olmo2.py for transformers>=5.13 rope schema"
"$P" - <<'PYEOF'
import vllm.model_executor.models.olmo2 as m
path = m.__file__
src = open(path).read()
old = '''        if sliding_window is None:
            rope_parameters = self.config.rope_parameters
        else:
            rope_theta = self.config.rope_parameters["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}'''
new = '''        _rp = self.config.rope_parameters
        if "rope_theta" not in _rp and ("full_attention" in _rp or "sliding_attention" in _rp):
            rope_parameters = _rp["full_attention" if sliding_window is None else "sliding_attention"]
        elif sliding_window is None:
            rope_parameters = _rp
        else:
            rope_theta = _rp["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}'''
if old in src:
    open(path, "w").write(src.replace(old, new))
    print("patched", path)
elif "_rp" in src:
    print("already patched")
else:
    raise SystemExit(f"FATAL: rope block not found in {path} — vllm version changed, re-check")
PYEOF

# transformers >=5.13 returns olmo3 rope cos/sin in fp32; FSDP2 casts layer
# inputs to bf16 on the hooked forward but NOT on activation-checkpoint
# recompute (pytorch #159359), so training dies with CheckpointError (saved
# bf16, recomputed fp32). Cast the bounded outputs to x.dtype at return —
# computation stays fp32, same numerics the FSDP2 cast produced pre-patch.
log "patching transformers modeling_olmo3.py rope output dtype"
"$P" - <<'PYEOF'
import transformers.models.olmo3.modeling_olmo3 as m
path = m.__file__
src = open(path).read()
old = '''            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos, sin'''
new = '''            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)'''
if old in src:
    open(path, "w").write(src.replace(old, new))
    print("patched", path)
elif "return cos.to(x.dtype), sin.to(x.dtype)" in src:
    print("already patched")
else:
    raise SystemExit(f"FATAL: rope return not found in {path} — transformers changed, re-check")
PYEOF

log "flash-attn (source build; wheel cached in \$UV_CACHE_DIR for later pods; nvcc: $(nvcc --version | grep -o 'release [0-9.]*')"
uv pip install -p "$P" ninja packaging
# flash-attn's setup ignores TORCH_CUDA_ARCH_LIST; it wants FLASH_ATTN_CUDA_ARCHS
# ("80;90" style, no dots) — without it the build fans out to every arch incl.
# sm_100/sm_120 and dies.
FLASH_ATTN_CUDA_ARCHS="$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.' | tr ';' ';')" \
MAX_JOBS="$MAX_JOBS" uv pip install -p "$P" --no-build-isolation "flash-attn==2.8.3.post1"

log "exporting built wheel to $VOL/wheels"
mkdir -p "$VOL/wheels"
find "${UV_CACHE_DIR:-/root/.cache/uv}" /root/.cache/uv -name "flash_attn*.whl" -newer "$ENV_DIR/bin/python" -exec cp -v {} "$VOL/wheels/" \; 2>/dev/null || true

log "sanity"
"$P" - <<'PY'
import torch, vllm, verl, flash_attn, ray
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
print("torch", torch.__version__, "| vllm", vllm.__version__, "| verl", verl.__version__,
      "| flash_attn", flash_attn.__version__, "| ray", ray.__version__)
print("TinkerActorRolloutRefWorker: ok")
PY
log "DONE — activate with: source $ENV_DIR/bin/activate"
