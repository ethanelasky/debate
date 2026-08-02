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

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version >/dev/null 2>&1 || { echo "FATAL: nvcc missing (need a devel image)"; exit 1; }

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
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "datasets>=3.1" "wandb>=0.18" jinja2 codetiming

log "flash-attn (source build; wheel cached in \$UV_CACHE_DIR for later pods)"
uv pip install -p "$P" ninja packaging
MAX_JOBS="$MAX_JOBS" uv pip install -p "$P" --no-build-isolation "flash-attn==2.8.3.post1"

log "sanity"
"$P" - <<'PY'
import torch, vllm, verl, flash_attn, ray
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
print("torch", torch.__version__, "| vllm", vllm.__version__, "| verl", verl.__version__,
      "| flash_attn", flash_attn.__version__, "| ray", ray.__version__)
print("TinkerActorRolloutRefWorker: ok")
PY
log "DONE — activate with: source $ENV_DIR/bin/activate"
