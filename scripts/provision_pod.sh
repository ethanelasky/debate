#!/usr/bin/env bash
# Provision a self-contained verl-sm90 env for the debate VerlBackend.
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
SYMBOLIC_DEPS=(
  "math-verify[antlr4_13_2]==0.9.0"
  "latex2sympy2-extended==1.11.0"
  "sympy==1.14.0"
  "antlr4-python3-runtime==4.13.2"
  # antlr 4.13.2 excludes every stable hydra (they pin antlr 4.9.*); without
  # these dev pins the resolver backtracks to hydra 0.11/omegaconf 1.4, which
  # verl cannot import (no omegaconf.MISSING).
  "omegaconf==2.4.0.dev3"
  "hydra-core==1.4.0.dev1"
  # GDN models (Qwen3.5) die at EngineCore init on unpinned cutlass; the trio
  # must move together — unpinned libs-cu13 pulls 4.7.0 against dsl 4.5.2.
  "nvidia-cutlass-dsl==4.5.2"
  "nvidia-cutlass-dsl-libs-base==4.5.2"
  "nvidia-cutlass-dsl-libs-cu13==4.5.2"
)
VOL="${VOL:-/workspace}"
ENV_DIR="$VOL/envs/verl-sm90"
export HF_HOME="${HF_HOME:-$VOL/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VOL/uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VOL/uv/python}"
# Every long step below is bounded by timeout(1) — an unbounded stall is the
# exact failure this script must not have. Checked before the first $VOL touch
# because that mkdir is itself an NFS operation that can hang.
command -v timeout >/dev/null 2>&1 || {
  echo "FATAL: no timeout(1) (coreutils) on this image; provisioning not started" >&2; exit 1; }

# $VOL is a network volume on most pods: mkdir hangs if it is unreachable.
timeout -k 30s 120 mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" || {
  echo "FATAL: mkdir under \$VOL=$VOL failed or hung >120s (network volume unreachable?)" >&2; exit 1; }

log() { echo -e "\n=== [provision] $* ==="; }

# ---------------------------------------------------------------- shared-volume
# $VOL is shared: other people's pods run out of this same env while this script
# is capable of rewriting it. `uv pip install` into a live env swaps packages
# under a running trainer -- best case an ImportError mid-run, worst case a
# partially-swapped package that imports and misbehaves (flash_attn_2_cuda
# against a different torch is exactly that shape). So the env is WRITE-ONCE.
STAMP="$ENV_DIR/.provisioned"
LOCK="$VOL/locks/provision.lock"
LOCK_TTL_SECONDS="${LOCK_TTL_SECONDS:-7200}"

# The test is the INTERPRETER, not the stamp. An env built before this stamp
# existed has none and is still a live dependency -- the shared team volume is
# exactly that, with four jobs importing out of it right now. Keying the guard
# on the stamp alone would rebuild it and swap packages under all four, which is
# the precise failure this guard exists to prevent. Stamp absence means
# UNVERSIONED, never UNBUILT.
if [ -x "$ENV_DIR/bin/python" ] && [ -z "${FORCE_PROVISION:-}" ]; then
  log "env already present at $ENV_DIR — nothing to do"
  [ -f "$STAMP" ] && cat "$STAMP" || echo "(no stamp: predates stamping; provenance unknown)"
  echo "Set FORCE_PROVISION=1 to rebuild, but ONLY when no other pod is running"
  echo "out of $ENV_DIR: rebuilding mutates an env other runs are executing."
  exit 0
fi

# mkdir, not a lock FILE: on NFS-backed storage mkdir is atomic where
# open(O_EXCL) is not reliably so. The lease has a TTL because a pod that dies
# mid-provision would otherwise wedge the volume for everyone, forever.
timeout -k 10s 60 mkdir -p "$VOL/locks" || {
  echo "FATAL: cannot create $VOL/locks" >&2; exit 1; }
if ! mkdir "$LOCK" 2>/dev/null; then
  HELD_AT=$(cat "$LOCK/acquired_at" 2>/dev/null || echo 0)
  AGE=$(( $(date +%s) - HELD_AT ))
  if [ "$HELD_AT" != "0" ] && [ "$AGE" -lt "$LOCK_TTL_SECONDS" ]; then
    echo "FATAL: another pod is provisioning (lock held ${AGE}s by $(cat "$LOCK/owner" 2>/dev/null))." >&2
    echo "  Wait for it, or if that pod is gone, remove $LOCK by hand." >&2
    exit 1
  fi
  echo "WARNING: breaking stale provision lock (age ${AGE}s > TTL ${LOCK_TTL_SECONDS}s)" >&2
fi
date +%s > "$LOCK/acquired_at"
echo "${RUNPOD_POD_ID:-unknown-pod} pid $$" > "$LOCK/owner"
# Released on ANY exit: a lock that outlives its holder is the failure this
# whole mechanism is supposed to prevent.
trap 'rm -rf "$LOCK"' EXIT

# Free space is shared fate: when the volume fills, every concurrent run fails
# on its next write, not just whoever tipped it over. Checkpoints are ~3 GB a
# save, so a few parallel runs move this fast.
FREE_PCT=$(df -P "$VOL" 2>/dev/null | awk 'NR==2 {gsub(/%/,"",$5); print 100-$5}')
if [ -n "$FREE_PCT" ] && [ "$FREE_PCT" -lt 20 ]; then
  echo "WARNING: only ${FREE_PCT}% free on $VOL — grow the volume before running." >&2
  echo "         A full volume fails EVERY concurrent run, not just this one." >&2
fi

log "GPU / driver"
# nvidia-smi blocks forever on a wedged driver / ECC-recovering GPU.
timeout -k 30s 60 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "FATAL: no nvidia-smi (or it hung >60s: driver wedged, recreate the pod)" >&2; exit 1; }

# Single-arch flash-attn build: ~5x less compile work and memory than the
# default multi-arch fan-out.
ARCH="$(timeout -k 30s 60 nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" || {
  echo "FATAL: could not read compute_cap from nvidia-smi (hung >60s or driver wedged)" >&2; exit 1; }
[ -n "$ARCH" ] || { echo "FATAL: empty compute_cap from nvidia-smi; cannot pick a flash-attn arch" >&2; exit 1; }
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

# Toolkit preference (2026-08-02, learned overnight): a REAL system CUDA
# toolkit whose major matches torch's (cu130 -> 13.x) is the only reliable
# build env — use an image like runpod/pytorch:*-cu1300-*. The pip-shipped
# nvidia/cu13 fragments version-skew against each other (nvcc vs cccl vs
# ptxas) and are a LAST resort only.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# "release 13" also matches a future "release 130"; anchor on the dot. nvcc can
# hang on a half-mounted toolkit dir, hence the bound.
if ! timeout -k 5s 30 nvcc --version 2>/dev/null | grep -q "release 13\."; then
  VENV_CU="$ENV_DIR/lib/python3.12/site-packages/nvidia/cu13"
  if [ -x "$VENV_CU/bin/nvcc" ]; then
    log "WARNING: system nvcc missing/not 13.x; falling back to pip toolchain fragments (skew-prone)"
    export CUDA_HOME="$VENV_CU"; export PATH="$CUDA_HOME/bin:$PATH"
  else
    echo "FATAL: no CUDA 13.x nvcc (use a runpod/pytorch:*-cu1300-* image)"; exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  # The installer script itself unpacks and writes to ~/.local; bound the sh
  # side too, not just the download.
  if ! curl -LsSf -m 120 https://astral.sh/uv/install.sh | timeout -k 30s 300 sh; then
    echo "FATAL: uv install failed or exceeded its deadline (astral.sh unreachable, or the installer stalled writing to \$HOME)" >&2; exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || {
    echo "FATAL: uv still not on PATH after install; expected \$HOME/.local/bin/uv" >&2; exit 1; }
fi

# The interpreter download can stall on a bad mirror; $VOL may also be NFS.
if [ ! -x "$ENV_DIR/bin/python" ]; then
  log "creating venv (python 3.12)"
  timeout -k 30s 900 uv venv "$ENV_DIR" --python 3.12 --seed || {
    echo "FATAL: uv venv at $ENV_DIR failed or exceeded 900s (python 3.12 download stalled, or \$VOL not writable)" >&2; exit 1; }
fi
P="$ENV_DIR/bin/python"

# verl caps transformers at <5.11 (setup.py + requirements.txt, still true at
# main ddfbf4e). Our floor is >=5.13 and non-negotiable — see the note below.
# The two are unsatisfiable together, so the resolve HAS to be told which one
# wins; without this the install dies with "your requirements are
# unsatisfiable" and nothing provisions.
#
# Overriding is safe as far as verl's own reason goes: its bound exists for a
# sink-less-model bug fixed in transformers 5.6.1 (huggingface/transformers
# #45588), and `<5.11` is an untested-ceiling, not a known break. It is still
# an override of an upstream pin — if verl starts using a transformers API
# that moved after 5.11, this is the first place to look.
OVERRIDE="$VOL/verl-transformers-override.txt"
printf 'transformers==5.14.1\n' > "$OVERRIDE"

log "vllm ($VLLM_PIN) + verl @ ${VERL_PIN:0:7} (joint resolve, transformers override)"
timeout -k 30s 3600 uv pip install -p "$P" --override "$OVERRIDE" "$VLLM_PIN" \
  "verl @ git+https://github.com/volcengine/verl@${VERL_PIN}" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "datasets>=3.1" "wandb==0.28.1" jinja2 codetiming \
  backoff "docent-python==0.1.77" "pydantic>=2.0" "openai>=1.0" pyyaml \
  "${SYMBOLIC_DEPS[@]}" \
  "transformers==5.14.1" || {
  echo "FATAL: joint vllm+verl resolve failed or exceeded 3600s (git clone of verl hanging, or NFS-stalled \$UV_CACHE_DIR=$UV_CACHE_DIR)" >&2; exit 1; }
# transformers >=5.13 is a HARD floor: 5.10-5.12 misapply OLMo-3's YaRN factor
# to sliding-attention layers (HF regression #39847, fixed in #46911), which
# skews trainer logprobs ~1 nat/token vs vLLM past 4096 ctx and silently
# breaks the PPO ratio anchor (shows up as kl k1 ~1.5 at step 0).

# vllm 0.24 reads the pre-5.13 flat rope_parameters schema for olmo2/3 and
# KeyErrors on the new per-layer-type dict; shim it to accept both (same
# semantics: yarn on full-attention layers, default rope on sliding).
log "patching vllm olmo2.py for transformers>=5.13 rope schema"
timeout -k 30s 600 "$P" - <<'PYEOF' || { echo "FATAL: vllm olmo2.py patch failed or exceeded 600s (importing vllm can stall on a first-touch NFS site-packages)" >&2; exit 1; }
import vllm.model_executor.models.olmo2 as m
path = m.__file__
src = open(path).read()
old = '''        if sliding_window is None:
            rope_parameters = self.config.rope_parameters
        else:
            rope_theta = self.config.rope_parameters["rope_theta"]
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}'''
new = '''        # debate-rope-shim
        _rp = self.config.rope_parameters
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
elif "debate-rope-shim" in src:
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
timeout -k 30s 600 "$P" - <<'PYEOF' || { echo "FATAL: transformers modeling_olmo3.py patch failed or exceeded 600s" >&2; exit 1; }
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

# peft disable_adapter() (the ref-logprob pass) temporarily flips adapter
# requires_grad off. If that pass runs before any training forward, FSDP2
# lazy-init caches "no trainable params" per group and later reduce-scatters
# bf16 grads into fp32 params — fatal under torch>=2.11 grad_dtype
# enforcement (diagnosed 2026-08-03; debate runs dodged it only because their
# first pass was a training forward). Force lazy-init right after the wrap,
# while adapters are trainable.
log "patching verl transformer_impl.py fsdp2 lazy-init"
timeout -k 30s 600 "$P" - <<'PYEOF' || { echo "FATAL: verl transformer_impl.py lazy-init patch failed or exceeded 600s" >&2; exit 1; }
import verl.workers.engine.fsdp.transformer_impl as m
path = m.__file__.replace(".pyc", ".py")
SENTINEL = "# debate-fsdp2-lazyinit"
ANCHOR = "            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)"
PATCH = f"""{ANCHOR}
            {SENTINEL}: peft disable_adapter() (ref-logprob pass) flips adapter
            # requires_grad off; if that pass runs first, FSDP2 lazy-init caches
            # "no trainable params" per group and later reduces bf16 grads into
            # fp32 params (fatal under torch>=2.11 grad_dtype enforcement).
            # Initialize dtype attrs NOW, while adapters are trainable.
            module._get_fsdp_state()._lazy_init()"""
src = open(path).read()
if SENTINEL in src:
    print("already patched")
elif ANCHOR in src:
    open(path, "w").write(src.replace(ANCHOR, PATCH, 1))
    print("patched", path)
else:
    raise SystemExit(f"FATAL: fsdp2 load anchor not found in {path} — verl changed, re-check")
PYEOF

# Captured in its own statement: inside $(...) a hung/failed nvcc is invisible
# because `log` itself succeeds.
NVCC_VER="$(timeout -k 5s 30 nvcc --version | grep -o 'release [0-9.]*')" || {
  echo "FATAL: nvcc --version failed or hung >30s just before the flash-attn build (CUDA_HOME=$CUDA_HOME)" >&2; exit 1; }
# A prebuilt wheel skips a ~45min source build entirely. There is no published
# wheel for torch 2.11+cu130 (the matrix stops at 2.10), so the only source of
# one is a previous pod: after a successful provision, copy
#   $UV_CACHE_DIR/sdists-v9/pypi/flash-attn/*/*/flash_attn-*.whl
# off the pod and pass it back here on the next one:
#   FLASH_ATTN_WHEEL=/root/flash_attn-...whl bash scripts/provision_pod.sh
# The wheel is tied to the exact python/torch/CUDA of the pod that built it, so
# a mismatched one fails loudly at import rather than silently misbehaving —
# which is why this verifies the import before continuing.
if [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
  [ -f "$FLASH_ATTN_WHEEL" ] || { echo "FATAL: FLASH_ATTN_WHEEL=$FLASH_ATTN_WHEEL not found" >&2; exit 1; }
  log "flash-attn from prebuilt wheel ($FLASH_ATTN_WHEEL) — skipping the source build"
  timeout -k 30s 300 uv pip install -p "$P" "$FLASH_ATTN_WHEEL" || {
    echo "FATAL: installing $FLASH_ATTN_WHEEL failed" >&2; exit 1; }
  # torch FIRST: flash_attn_2_cuda links against libc10.so, which only lands on
  # the loader path once torch is imported. Testing the extension alone reports
  # a bogus "wheel built for a different torch".
  timeout -k 10s 180 "$P" -c "import torch, flash_attn, flash_attn_2_cuda" || {
    echo "FATAL: $FLASH_ATTN_WHEEL installed but its CUDA extension will not import — wheel was built for a different python/torch/CUDA. Unset FLASH_ATTN_WHEEL to build from source." >&2; exit 1; }
  log "flash-attn wheel verified"
else

log "flash-attn (source build; wheel cached in \$UV_CACHE_DIR for later pods; nvcc: $NVCC_VER"
timeout -k 30s 600 uv pip install -p "$P" ninja packaging || {
  echo "FATAL: installing ninja/packaging failed or exceeded 600s; without ninja the flash-attn build serializes and will not finish" >&2; exit 1; }
# flash-attn's setup ignores TORCH_CUDA_ARCH_LIST; it wants FLASH_ATTN_CUDA_ARCHS
# ("80;90" style, no dots) — without it the build fans out to every arch incl.
# sm_100/sm_120 and dies.
FLASH_ATTN_CUDA_ARCHS="$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.' | tr ';' ';')" \
MAX_JOBS="$MAX_JOBS" timeout -k 30s 7200 uv pip install -p "$P" --no-build-isolation "flash-attn==2.8.3.post1" || {
  echo "FATAL: flash-attn source build failed or exceeded 7200s with MAX_JOBS=$MAX_JOBS / archs $TORCH_CUDA_ARCH_LIST." >&2
  echo "  A build that merely OOM-Killed exits fast; a 2h wall means nvcc is thrashing (arch fan-out) or a cutlass unit is stuck — check dmesg and free -g." >&2
  exit 1; }
fi

log "exporting built wheel to $VOL/wheels"
# Non-fatal (the env is already built) but NOT silent: an empty wheel cache
# makes the next pod repeat the 2h build, so a total failure has to be visible.
timeout -k 30s 60 mkdir -p "$VOL/wheels" || {
  echo "WARNING: mkdir $VOL/wheels failed or hung >60s; flash-attn wheel NOT exported — the next pod will rebuild it" >&2; }
# One of the two cache paths is normally absent; find's rc is therefore not a
# usable signal — the ls below is what decides whether the export worked.
timeout -k 30s 300 find "${UV_CACHE_DIR:-/root/.cache/uv}" /root/.cache/uv -name "flash_attn*.whl" -newer "$ENV_DIR/bin/python" -exec cp -v {} "$VOL/wheels/" \; 2>/dev/null || true
ls "$VOL/wheels"/flash_attn*.whl >/dev/null 2>&1 || {
  echo "WARNING: no flash_attn*.whl in $VOL/wheels after export; the next pod repeats the source build" >&2; }

log "sanity"
timeout -k 30s 900 "$P" - <<'PY' || { echo "FATAL: sanity import failed or exceeded 900s — a hang here is usually torch/vllm probing a wedged GPU, not a slow import" >&2; exit 1; }
import torch, vllm, verl, flash_attn, ray
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
print("torch", torch.__version__, "| vllm", vllm.__version__, "| verl", verl.__version__,
      "| flash_attn", flash_attn.__version__, "| ray", ray.__version__)
print("TinkerActorRolloutRefWorker: ok")
PY
# Written ONLY after the sanity import passes, so the stamp means "this env
# was proven to work", not merely "the script reached the end". A half-built env
# with a stamp would be worse than none: every later pod would skip provisioning
# and fail at import instead.
cat > "$STAMP" <<STAMPEOF
provisioned_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
by_pod=${RUNPOD_POD_ID:-unknown}
vllm_pin=$VLLM_PIN
verl_pin=$VERL_PIN
flash_attn=${FLASH_ATTN_WHEEL:-built-from-source}
STAMPEOF

log "DONE — activate with: source $ENV_DIR/bin/activate"
