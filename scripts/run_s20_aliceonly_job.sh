#!/usr/bin/env bash
# Reconstruct the pinned RLVR step-20 base, then run either the live smoke or
# the full proposer-only no-rebuttal experiment. Intended for jobd pods only.
set -euo pipefail

MODE="${1:?usage: run_s20_aliceonly_job.sh smoke|full}"
case "$MODE" in
  smoke)
    CONFIG=infra/jobd/debate_s20_aliceonly_smoke.yaml
    EXP=mathl5_qwen35_pc_debate_norebut_propbudget_rlvrs20_alice_smoke
    NAMESPACE=s20-aliceonly-smoke-01
    ;;
  full)
    CONFIG=configs/math_pc_debate.yaml
    EXP=mathl5_qwen35_pc_debate_norebut_propbudget_rlvrs20_alice
    NAMESPACE=s20-aliceonly-01
    ;;
  *)
    echo "FATAL: mode must be smoke or full, got $MODE" >&2
    exit 2
    ;;
esac

export POD_IDLE_STOP=0
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HOME=/workspace/hf
export CONFIG
export DEBATE_LAUNCH_NAMESPACE="$NAMESPACE"

# The profile may rebound from B200 to H200; use the environment restored by
# that profile instead of assuming one architecture-specific name.
PY=""
for candidate in verl-b200 verl-sm90; do
  if [ -x "/workspace/envs/$candidate/bin/python" ]; then
    PY="/workspace/envs/$candidate/bin/python"
    break
  fi
done
test -n "$PY" || {
  echo "FATAL: no verl environment under /workspace/envs" >&2
  ls -la /workspace/envs >&2
  exit 1
}
export PY
echo "== using $PY =="
nvidia-smi --query-gpu=name --format=csv,noheader

# Pin all three files before reconstructing the merged step-20 model. The
# checkpoint is FSDP2-sharded across two ranks; merging rank zero alone would
# silently fold in half of every LoRA matrix.
ADAPTER=/workspace/rlvr-s20-adapter
MERGED=/workspace/models/qwen35-rlvr-sub8-s20
SENTINEL=/workspace/models/.qwen35-rlvr-sub8-s20.ok
printf '%s  %s\n' \
  eccebdc69c251f8a1e9e7b14b07f858a5bb50cb06021876e275bfda49b7fa23e "$ADAPTER/model_world_size_2_rank_0.pt" \
  229e3b2d95d69e34006f13e8010563b499a86d2792adc7b3a5b3837b4aa9210e "$ADAPTER/model_world_size_2_rank_1.pt" \
  349f62eff696826bc2adec7661a8559b44883e6d137cff5a31cd64adf6a9e8ab "$ADAPTER/lora_train_meta.json" \
  | sha256sum -c -

if [ -f "$SENTINEL" ]; then
  echo "== merged base already present at $MERGED =="
else
  # $MERGED is a derived cache with the three verified inputs above as its
  # source. A no-sentinel directory is an interrupted reconstruction, never a
  # run artifact; clear only that explicit path before rebuilding it.
  rm -rf "$MERGED"
  mkdir -p /workspace/models
  BASE_SNAP="$(find "$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots" \
    -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
  if [ -z "$BASE_SNAP" ] || [ ! -f "$BASE_SNAP/config.json" ]; then
    echo "== base snapshot not cached; downloading Qwen/Qwen3.5-4B =="
    hf download Qwen/Qwen3.5-4B
    BASE_SNAP="$(find "$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots" \
      -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
  fi
  test -n "$BASE_SNAP" || {
    echo "FATAL: no Qwen3.5-4B snapshot under $HF_HOME" >&2
    exit 1
  }
  test -f "$BASE_SNAP/config.json"
  echo "== merging $ADAPTER into $BASE_SNAP =="
  "$PY" scripts/merge_lora_ckpt.py \
    --ckpt "$ADAPTER" --base "$BASE_SNAP" --out "$MERGED" \
    2>&1 | tee /tmp/merge-s20-aliceonly.log
  grep -q 'applied 346 deltas; unmatched: 0' /tmp/merge-s20-aliceonly.log || {
    echo "FATAL: unexpected merge result" >&2
    exit 1
  }
  touch "$SENTINEL"
fi
test -f "$MERGED/config.json"

bash scripts/pod_run.sh debate "$EXP"

if [ "$MODE" = smoke ]; then
  TRANSCRIPT="transcripts/$EXP/$NAMESPACE/train-step-00000.jsonl"
  test -s "$TRANSCRIPT" || {
    echo "FATAL: smoke produced no training transcript at $TRANSCRIPT" >&2
    exit 1
  }
  "$PY" - "$TRANSCRIPT" <<'PYEOF'
import json
import sys

path = sys.argv[1]
expected = [
    ("alice", "proposal"),
    ("bob", "critique"),
    ("judge", "deliberation"),
    ("judge", "verdict"),
]
complete = 0
rows = 0
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        rows += 1
        item = json.loads(line)
        omniscient = [
            view for view in item.get("transcripts", [])
            if view.get("name") == "omniscient"
        ]
        if len(omniscient) != 1:
            raise SystemExit(f"debate {rows}: expected one omniscient view")
        observed = []
        for message in omniscient[0].get("messages", []):
            if message.get("role") != "assistant":
                continue
            meta = message.get("metadata") or {}
            observed.append((meta.get("speaker"), meta.get("slot")))
        if any(slot in {"alice_rebuttal", "bob_rebuttal"} for _, slot in observed):
            raise SystemExit(f"debate {rows}: rebuttal leaked into no-rebuttal smoke")
        if set((item.get("metadata") or {}).get("grades", {})) - {"alice"}:
            raise SystemExit(f"debate {rows}: a frozen seat was graded as trained")
        if observed == expected and not (item.get("metadata") or {}).get("failed"):
            complete += 1
if rows == 0 or complete == 0:
    raise SystemExit(
        f"smoke routing gate failed: rows={rows}, complete four-slot debates={complete}"
    )
print(f"SMOKE_ROUTING_VERIFIED rows={rows} complete={complete} slots={expected}")
PYEOF
fi
