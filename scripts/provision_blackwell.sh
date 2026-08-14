#!/usr/bin/env bash
# Provision the provision_pod.sh env on an sm100 (Blackwell / B200) machine.
# SAME venv contents and layout ($VOL/envs/verl-sm90) so pod_run.sh, the
# configs, and infra/backend/verl.py need no changes. Diverges from
# provision_pod.sh ONLY where an sm90-era choice stops holding on sm100:
#  * flash-attn: the sm90 script's wheel/source pair assumes FA2's Hopper
#    kernel matrix. Here the source build is ATTEMPTED (arch 100, bounded) and
#    ANY failure — including a build that succeeds but will not import — is a
#    loud SKIP, never a fatal: vllm selects its own attention backend at serve
#    time (FLASHINFER/TRITON when flash-attn is absent), and nothing in
#    infra/backend/verl.py names one, so the skip is contained to this script.
#  * sanity runs a real bf16 CUDA matmul, not just imports: an sm90-only
#    extension imports cleanly and dies at first kernel launch ("no kernel
#    image is available"), which on a paid B200 must surface HERE, not twenty
#    minutes into a smoke run.
# Open sm100 questions are tagged '# BLACKWELL-VERIFY:' — resolve each against
# the Blackwell research dossier before first paid use.
#
# Usage (on the pod):  VOL=/workspace bash provision_blackwell.sh
#   FLASH_ATTN_SKIP=1    skip the flash-attn attempt outright (fastest bring-up)
#   FLASH_ATTN_WHEEL=…   wheel from a PREVIOUS sm100 pod (never an sm90 one)
set -euo pipefail

VERL_PIN="${VERL_PIN:-e9618406de5bad40041d7612554e465ec2003ec1}"
# Same pin as provision_pod.sh: the verl-side patches below anchor on this
# commit's source, and the two provisions must stay byte-comparable.
# BLACKWELL-VERIFY: vllm 0.24 cu130 wheels must carry sm_100 SASS (or PTX that
# JITs to it) in their compiled ops; the dossier should confirm from the wheel
# metadata / release notes. The sanity matmul below tests torch, not vllm —
# vllm's kernels are first exercised by blackwell_smoke.md step (b).
VLLM_PIN="${VLLM_PIN:-vllm==0.24.*}"
# 2 was forced by a 62GB-RAM host (cutlass units spike >15GB RSS each). B200
# hosts ship several times that; raise via MAX_JOBS only after `free -g`
# confirms ~16GB per job, and only for the flash-attn attempt.
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
# verl-b200, NOT verl-sm90: the volume is shared with sm90 pods and verl-sm90
# holds an sm90-built flash-attn that backend autodetection could pick up.
# pod_run.sh finds this env via PY=/workspace/envs/verl-b200/bin/python.
ENV_DIR="$VOL/envs/${VENV_NAME:-verl-b200}"
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

log() { echo -e "\n=== [provision-blackwell] $* ==="; }

log "GPU / driver"
# nvidia-smi blocks forever on a wedged driver / ECC-recovering GPU.
timeout -k 30s 60 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "FATAL: no nvidia-smi (or it hung >60s: driver wedged, recreate the pod)" >&2; exit 1; }

ARCH="$(timeout -k 30s 60 nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" || {
  echo "FATAL: could not read compute_cap from nvidia-smi (hung >60s or driver wedged)" >&2; exit 1; }
[ -n "$ARCH" ] || { echo "FATAL: empty compute_cap from nvidia-smi; cannot pick an arch" >&2; exit 1; }
# This script encodes sm100 decisions (flash-attn skip logic, the kernel-launch
# sanity). On anything else those decisions are wrong in both directions:
# provision_pod.sh is the script for sm90-era cards.
case "$ARCH" in
  10.*) : ;;
  *) [ "${ALLOW_NON_SM100:-0}" = 1 ] || {
       echo "FATAL: compute_cap=$ARCH is not sm100; use scripts/provision_pod.sh (or ALLOW_NON_SM100=1 to override)" >&2; exit 1; } ;;
esac
# B200 reports compute_cap 10.0, and nvidia-smi emits it bare — so the derived
# list is "10.0" (verified against the csv,noheader format; no unit suffix to
# mangle). Single-arch keeps the flash-attn attempt ~5x cheaper, as on sm90.
# BLACKWELL-VERIFY: whether torch extensions built here need "10.0a" (the
# arch-conditional feature set cutlass kernels target on Blackwell) instead of
# plain "10.0". If the dossier says yes, export TORCH_CUDA_ARCH_LIST=10.0a at
# invocation — the FLASH_ATTN_CUDA_ARCHS derivation below strips the suffix.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$ARCH}"
log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS"

# Toolkit preference identical to provision_pod.sh: a REAL system CUDA toolkit
# whose major matches torch's (cu130 -> 13.x) is the only reliable build env.
# CUDA 13.x targets sm_100 natively; the sm100-specific question is only
# whether the PIP fragments' nvcc does too, and that is probed at the
# flash-attn attempt (the one consumer of nvcc here).
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
else
  # A network volume moved over from an sm90 pod carries a flash-attn compiled
  # for sm_90 only: it imports, then dies at first launch. Not fatal — the
  # sanity matmul and the conditional flash-attn verify below are the deciders
  # — but the operator must know WHY they fail if they do.
  log "WARNING: reusing existing venv at $ENV_DIR — if it was provisioned on sm90, its source-built extensions may lack sm100 kernels; the sanity step decides"
fi
P="$ENV_DIR/bin/python"

# verl caps transformers at <5.11; our floor is >=5.13 and non-negotiable (see
# the note after the install). Unsatisfiable together, so the resolve HAS to be
# told which one wins. Rationale for why the override is safe lives in
# provision_pod.sh; nothing about it is arch-dependent.
OVERRIDE="$VOL/verl-transformers-override.txt"
printf 'transformers>=5.13,<6\n' > "$OVERRIDE"

# uv HARDLINKS installed files into the shared $UV_CACHE_DIR by default, and
# the in-place source patches below write THROUGH those links: a patched
# olmo2.py lands in /workspace/uv and surfaces pre-patched — with whatever
# comment wording THIS revision wrote — in every env that later installs the
# same wheel, verl-sm90 included. --link-mode=copy on every install in this
# script keeps this env's patches out of the cache; the semantic
# already-patched checks below tolerate files that leaked from envs
# provisioned before this guard existed.
log "vllm ($VLLM_PIN) + verl @ ${VERL_PIN:0:7} (joint resolve, transformers override)"
timeout -k 30s 3600 uv pip install -p "$P" --link-mode=copy --override "$OVERRIDE" "$VLLM_PIN" \
  "verl @ git+https://github.com/volcengine/verl@${VERL_PIN}" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "datasets>=3.1" "wandb>=0.18" jinja2 codetiming \
  backoff "docent-python>=0.1.74" "pydantic>=2.0" "openai>=1.0" pyyaml ninja \
  "${SYMBOLIC_DEPS[@]}" \
  "transformers>=5.13,<6" || {
  echo "FATAL: joint vllm+verl resolve failed or exceeded 3600s (git clone of verl hanging, or NFS-stalled \$UV_CACHE_DIR=$UV_CACHE_DIR)" >&2; exit 1; }
# transformers >=5.13 is a HARD floor: 5.10-5.12 misapply OLMo-3's YaRN factor
# to sliding-attention layers (HF regression #39847, fixed in #46911), which
# skews trainer logprobs ~1 nat/token vs vLLM past 4096 ctx and silently
# breaks the PPO ratio anchor (shows up as kl k1 ~1.5 at step 0).
# ninja is load-bearing on the default flash-attn-skip path, not just for
# source builds: vllm's FLASHINFER backend JIT-compiles kernels at first serve
# and shells out to ninja(1); without it EngineCore dies while /v1/models
# keeps answering — a zombie only a real completion catches.

# The three patches below are pure-python and arch-independent; kept verbatim
# from provision_pod.sh so the two provisions stay byte-comparable.

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
elif "debate-rope-shim" in src or '_rp["full_attention" if sliding_window is None else "sliding_attention"]' in src:
    # Semantic check, not sentinel-only: uv hardlinks site-packages into the
    # shared cache, so this file can arrive ALREADY patched by another env's
    # provision whose revision wrote different comment wording. Only the
    # patched code pattern is stable across revisions.
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
# enforcement. Force lazy-init right after the wrap, while adapters are
# trainable.
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

# verl's attention_utils dispatches to flash_attn.bert_padding on the CUDA
# path with no fallback, and infra/backend/verl.py's _pack pipeline reaches it
# on every batch — use_remove_padding=False does NOT route around it. With
# flash-attn skipped (the sm100 default) the first training batch dies on that
# import. Fall back to transformers' pure-torch equivalents plus einops.
# Harmless on sm90: the except arm fires only when flash_attn will not import.
log "patching verl attention_utils.py flash_attn fallback"
timeout -k 30s 600 "$P" - <<'PYEOF' || { echo "FATAL: verl attention_utils.py shim failed or exceeded 600s" >&2; exit 1; }
import importlib
import verl.utils.attention_utils as m
path = m.__file__.replace(".pyc", ".py")
SENTINEL = "debate-b200-shim"
ANCHOR = "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
PATCH = """        # debate-b200-shim: no flash-attn on sm100; transformers >=5.13 ships
        # pure-torch equivalents of three of these symbols, einops the fourth.
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            from einops import rearrange
            from transformers.modeling_flash_attention_utils import (
                _index_first_axis as index_first_axis,
                _pad_input as pad_input,
                _unpad_input as unpad_input,
            )"""
src = open(path).read()
if SENTINEL in src:
    print("already patched")
elif ANCHOR in src:
    open(path, "w").write(src.replace(ANCHOR, PATCH, 1))
    print("patched", path)
else:
    raise SystemExit(f"FATAL: flash_attn import anchor not found in {path} — verl changed, re-check")
# Resolve all four symbols through the (possibly fallback) path NOW: a missing
# private transformers helper must fail here, not at the first training batch.
importlib.reload(m)
fns = m._get_attention_functions()
assert all(callable(f) for f in fns), fns
print("attention_utils dispatch ok:", sorted({getattr(f, "__module__", "?") for f in fns}))
PYEOF

# ---------------------------------------------------------------- flash-attn
# The sm100 divergence. Three exits, and the one taken is announced in a form
# grep can find later ("flash-attn PATH:"):
#   wheel   — operator-supplied wheel from a previous SM100 pod; verified or fatal
#   source  — pinned build attempted with archs=100; any failure downgrades to skip
#   skip    — vllm serves with its own fallback (FLASHINFER/TRITON), trainer
#             side falls back too (see BLACKWELL-VERIFY below)
# BLACKWELL-VERIFY: whether flash-attn 2.8.3.post1 can target sm100 AT ALL
# (FLASH_ATTN_CUDA_ARCHS=100 accepted, kernels correct — FA2's Blackwell path,
# if any, may reuse pre-Hopper kernel shapes: correct but slow). If the dossier
# names a different pinnable version or a published sm100 wheel for
# torch 2.11+cu130, set FLASH_ATTN_PIN / FLASH_ATTN_WHEEL accordingly.
# BLACKWELL-VERIFY: what verl's FSDP trainer path does with flash-attn ABSENT —
# transformers must fall back to sdpa/eager rather than error at model load.
# If it errors, the skip path is serve-only and flash-attn becomes a hard
# requirement here; the smoke arm (blackwell_smoke.md step d) is the test.
# BLACKWELL-VERIFY: whether the vllm 0.24 wheel needs a separate
# flashinfer-python install for its FLASHINFER backend or falls back to TRITON
# without it. Do NOT freelance `uv pip install flashinfer-python` — its
# dependency tree can move torch, and vllm's compiled extension is ABI-tied to
# the exact torch it resolved with. If the dossier wants flashinfer, it must
# name a pin installable with --no-deps against this torch.
FLASH_ATTN_PIN="${FLASH_ATTN_PIN:-flash-attn==2.8.3.post1}"
FA_STATE=skip

# Captured in its own statement: inside $(...) a hung/failed nvcc is invisible
# because `log` itself succeeds.
NVCC_VER="$(timeout -k 5s 30 nvcc --version | grep -o 'release [0-9.]*')" || {
  echo "FATAL: nvcc --version failed or hung >30s (CUDA_HOME=$CUDA_HOME)" >&2; exit 1; }

# Default SKIP (dossier 2026-08-08): no official flash-attn wheel exists for
# torch2.11+cu130+cp312 on sm100, and vllm 0.24 AUTO-selects FlashInfer on
# sm100 with a bundled FA path — external flash-attn is optional there, and
# the FSDP training side runs BF16+SDPA. Set FLASH_ATTN_SKIP=0 to attempt.
if [ "${FLASH_ATTN_SKIP:-1}" = 1 ]; then
  log "flash-attn PATH: skip (FLASH_ATTN_SKIP=1) — vllm will use its FLASHINFER/TRITON fallback"
elif [ -n "${FLASH_ATTN_WHEEL:-}" ]; then
  [ -f "$FLASH_ATTN_WHEEL" ] || { echo "FATAL: FLASH_ATTN_WHEEL=$FLASH_ATTN_WHEEL not found" >&2; exit 1; }
  log "flash-attn PATH: wheel ($FLASH_ATTN_WHEEL) — skipping the source build"
  timeout -k 30s 300 uv pip install -p "$P" --link-mode=copy "$FLASH_ATTN_WHEEL" || {
    echo "FATAL: installing $FLASH_ATTN_WHEEL failed" >&2; exit 1; }
  # torch FIRST: flash_attn_2_cuda links against libc10.so, which only lands on
  # the loader path once torch is imported. Testing the extension alone reports
  # a bogus "wheel built for a different torch". An sm90-built wheel passes
  # this import test and fails only at kernel launch — the sanity matmul below
  # does not exercise it, so wheel provenance is on the operator.
  timeout -k 10s 180 "$P" -c "import torch, flash_attn, flash_attn_2_cuda" || {
    echo "FATAL: $FLASH_ATTN_WHEEL installed but its CUDA extension will not import — wheel was built for a different python/torch/CUDA. Unset FLASH_ATTN_WHEEL to attempt the source build." >&2; exit 1; }
  FA_STATE=wheel
  log "flash-attn wheel verified"
else
  # Cheap disqualifier before a potentially multi-hour build: a toolkit that
  # cannot emit sm_100 (stale pip fragments) makes the attempt pointless.
  if ! timeout -k 5s 30 nvcc --list-gpu-arch 2>/dev/null | grep -q "compute_100"; then
    log "flash-attn PATH: skip — nvcc at $CUDA_HOME cannot target compute_100; vllm will use its FLASHINFER/TRITON fallback"
  else
    log "flash-attn PATH: source attempt ($FLASH_ATTN_PIN, archs from $TORCH_CUDA_ARCH_LIST; nvcc: $NVCC_VER)"
    # Fatal, unlike the build below: a dead installer THIS late means the env
    # or network broke mid-provision, and a skip would mask that.
    timeout -k 30s 600 uv pip install -p "$P" --link-mode=copy ninja packaging || {
      echo "FATAL: installing ninja/packaging failed or exceeded 600s; without ninja the flash-attn build serializes and will not finish" >&2; exit 1; }
    # flash-attn's setup ignores TORCH_CUDA_ARCH_LIST; it wants
    # FLASH_ATTN_CUDA_ARCHS ("80;90" style, no dots) — "10.0" -> "100". The sed
    # strips a feature suffix ("10.0a" -> "100"): the suffix belongs to the
    # torch/cutlass arch string, not to this env var's format.
    FA_ARCHS="$(echo "$TORCH_CUDA_ARCH_LIST" | tr -d '.' | sed 's/[a-z]*$//')"
    if FLASH_ATTN_CUDA_ARCHS="$FA_ARCHS" MAX_JOBS="$MAX_JOBS" \
        timeout -k 30s 7200 uv pip install -p "$P" --link-mode=copy --no-build-isolation "$FLASH_ATTN_PIN"; then
      # A build that links is not a build that runs; and a flash_attn that
      # imports but is broken flips transformers/vllm auto-detection toward a
      # backend that dies at runtime — so on verify failure it is REMOVED, not
      # left installed.
      if timeout -k 10s 180 "$P" -c "import torch, flash_attn, flash_attn_2_cuda"; then
        FA_STATE=source
        log "flash-attn PATH: source build succeeded and imports"
      else
        log "flash-attn PATH: skip — build succeeded but the extension will not import on sm100; uninstalling so autodetection cannot pick it"
        timeout -k 30s 300 uv pip uninstall -p "$P" flash-attn || {
          echo "FATAL: could not uninstall the broken flash-attn; a half-working extension left on the path WILL be auto-selected and die at runtime" >&2; exit 1; }
      fi
    else
      log "flash-attn PATH: skip — source build failed or exceeded 7200s (MAX_JOBS=$MAX_JOBS, archs $FA_ARCHS); vllm will use its FLASHINFER/TRITON fallback"
    fi
  fi
fi

if [ "$FA_STATE" = source ]; then
  log "exporting built wheel to $VOL/wheels"
  # Non-fatal (the env is already built) but NOT silent: an empty wheel cache
  # makes the next sm100 pod repeat the build. Wheels here are sm100-only;
  # never feed one to provision_pod.sh on an sm90 pod.
  timeout -k 30s 60 mkdir -p "$VOL/wheels" || {
    echo "WARNING: mkdir $VOL/wheels failed or hung >60s; flash-attn wheel NOT exported — the next pod will rebuild it" >&2; }
  # One of the two cache paths is normally absent; find's rc is therefore not a
  # usable signal — the ls below is what decides whether the export worked.
  timeout -k 30s 300 find "${UV_CACHE_DIR:-/root/.cache/uv}" /root/.cache/uv -name "flash_attn*.whl" -newer "$ENV_DIR/bin/python" -exec cp -v {} "$VOL/wheels/" \; 2>/dev/null || true
  ls "$VOL/wheels"/flash_attn*.whl >/dev/null 2>&1 || {
    echo "WARNING: no flash_attn*.whl in $VOL/wheels after export; the next pod repeats the source build" >&2; }
fi

log "sanity (imports + sm100 kernel launch; flash-attn state: $FA_STATE)"
# Unlike provision_pod.sh this launches a real bf16 matmul: on a new arch,
# import success proves packaging, not that any wheel carries sm_100 code.
FA_STATE="$FA_STATE" timeout -k 30s 900 "$P" - <<'PY' || { echo "FATAL: sanity failed — either an import broke, or no wheel carries sm_100 kernels ('no kernel image is available' above means the torch/vllm pins need the Blackwell dossier), or the GPU is wedged" >&2; exit 1; }
import os
import torch, vllm, verl, ray
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

fa_state = os.environ["FA_STATE"]
fa_ver = "SKIPPED"
if fa_state in ("wheel", "source"):
    import flash_attn, flash_attn_2_cuda  # noqa: F401
    fa_ver = flash_attn.__version__

cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print("capability", cap, "| torch arch list", arch_list)
if not any("100" in a for a in arch_list):
    # PTX from an older arch can still JIT forward, so this alone is a warning;
    # the matmul below is the hard test.
    print("WARNING: no sm_100/compute_100 entry in torch's arch list — running on JIT'd PTX at best")

a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
s = (a @ a).float().sum().item()
assert s == s, "bf16 matmul returned NaN"
print("bf16 matmul on", torch.cuda.get_device_name(0), "ok")

print("torch", torch.__version__, "| vllm", vllm.__version__, "| verl", verl.__version__,
      "| flash_attn", fa_ver, "| ray", ray.__version__)
print("TinkerActorRolloutRefWorker: ok")
PY
log "DONE (flash-attn: $FA_STATE) — activate with: source $ENV_DIR/bin/activate"
