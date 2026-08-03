#!/usr/bin/env bash
# All-on-pod training launcher for both run modes:
#   bash scripts/pod_run.sh debate <experiment> [runner args...]  # starts judge vLLM
#   bash scripts/pod_run.sh rlvr   <experiment> [runner args...]  # no judge
# Prereqs: provision_pod.sh has built $VOL/envs/verl-main; repo synced to /root/debate.
set -euo pipefail

MODE="${1:?usage: pod_run.sh <debate|rlvr> <experiment> [args...]}"
EXP="${2:?usage: pod_run.sh <debate|rlvr> <experiment> [args...]}"
PY="${PY:-/workspace/envs/verl-main/bin/python}"
# venv bin (ninja for JIT kernels) + cuda toolkit on PATH for all children
export PATH="$(dirname "$PY"):/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
cd /root/debate

case "$MODE" in
  debate) RUNNER=infra.run_debate; CONFIG=configs/math_pc_olmo.yaml ;;
  rlvr)   RUNNER=infra.run_rlvr;   CONFIG=configs/math_rlvr_olmo.yaml ;;
  *) echo "unknown mode $MODE (debate|rlvr)"; exit 2 ;;
esac

if [ "$MODE" = debate ]; then
  # Health = a REAL 1-token completion, not /v1/models: an api_server whose
  # EngineCore was killed keeps answering metadata while every generate fails
  # (burned 8 zero-datum steps on 2026-08-03 — all debates died at the judge).
  judge_ok() {
    curl -s -m 30 http://127.0.0.1:8788/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3.5-4B", "prompt": "1+1=", "max_tokens": 1}' \
      | grep -q '"choices"'
  }
  if ! judge_ok; then
    pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
    sleep 3
    echo "== starting judge vLLM server =="
    nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3.5-4B --port 8788 \
      --gpu-memory-utilization 0.18 --max-model-len 16384 \
      --max-num-seqs 32 \
      > /root/judge_server.log 2>&1 &
    until judge_ok; do sleep 5; done
    echo "== judge server up =="
  fi
fi

"$PY" -m pip install -q -e . --no-deps
exec "$PY" -m "$RUNNER" --experiment-file "$CONFIG" --experiment "$EXP" "${@:3}"
