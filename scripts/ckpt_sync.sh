#!/usr/bin/env bash
# Continuous checkpoint off-pod sync (Ethan, 2026-08-12: "checkpoints have to
# be synced back to the persistent volume before shutdown"). Host deaths give
# no warning, so this syncs CONTINUOUSLY after each save rather than hooking
# shutdown: every completed step-*/final dir under CKPT_ROOT uploads once to
# a private HF repo (the durable store reachable from every DC — the volume's
# S3 endpoint needs console-generated credentials we don't have yet; swap
# _upload() for an S3 sync when they exist).
#
# Usage:  CKPT_ROOT=/workspace/checkpoints RUN_NAME=<arm+suffix> \
#           setsid nohup bash scripts/ckpt_sync.sh > /root/ckpt_sync.log 2>&1 &
# No-ops (exits 0) when CKPT_ROOT lives on the network volume — those
# checkpoints are already durable.
set -u
CKPT_ROOT="${CKPT_ROOT:-/workspace/checkpoints}"
RUN_NAME="${RUN_NAME:?set RUN_NAME (the run's checkpoint namespace)}"
PYBIN="${PYBIN:-/workspace/envs/verl-b200/bin/python}"
QUIESCENT_SECS=90   # a dir this old is a finished write, not a mid-save
INTERVAL=120

if df -P "$CKPT_ROOT" 2>/dev/null | grep -q runpodfs; then
  echo "ckpt_sync: $CKPT_ROOT is volume-backed; nothing to do"
  exit 0
fi

STATE=/root/.ckpt_synced
touch "$STATE"

while true; do
  now=$(date +%s)
  for dir in "$CKPT_ROOT"/"$RUN_NAME"*/step-* "$CKPT_ROOT"/"$RUN_NAME"*/final; do
    [ -d "$dir" ] || continue
    grep -qxF "$dir" "$STATE" && continue
    mtime=$(stat -c %Y "$dir")
    [ $(( now - mtime )) -lt $QUIESCENT_SECS ] && continue
    echo "[$(date -u +%H:%M:%S)] uploading $dir"
    if HF_DIR="$dir" HF_RUN="$RUN_NAME" "$PYBIN" - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=open("/root/.cache/huggingface/token").read().strip())
repo = f"ethanelasky/ckpt-{os.environ['HF_RUN']}"[:96]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
d = os.environ["HF_DIR"]
api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                  path_in_repo=os.path.basename(d))
print("ok")
PYEOF
    then
      echo "$dir" >> "$STATE"
    else
      echo "[$(date -u +%H:%M:%S)] upload FAILED for $dir; will retry"
    fi
  done
  sleep $INTERVAL
done
