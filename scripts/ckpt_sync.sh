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
    # S3-to-volume when credentials exist (the durable store Ethan named);
    # HF private repo otherwise.
    if [ -f /root/.runpod/s3.env ]; then set -a; . /root/.runpod/s3.env; set +a; fi
    upload_ok=0
    SYNC_DIR="$dir" SYNC_RUN="$RUN_NAME" "$PYBIN" - <<'PYEOF' && upload_ok=1
import os, pathlib
d = pathlib.Path(os.environ["SYNC_DIR"]); run = os.environ["SYNC_RUN"]
if os.environ.get("AWS_ACCESS_KEY_ID"):
    import boto3
    s3 = boto3.client("s3", region_name="us-ca-2",
                      endpoint_url="https://s3api-us-ca-2.runpod.io")
    for f in d.rglob("*"):
        if f.is_file():
            key = f"checkpoints/{run}/{d.name}/{f.relative_to(d)}"
            s3.upload_file(str(f), "guppgnq1g0", key)
else:
    from huggingface_hub import HfApi
    api = HfApi(token=open("/root/.cache/huggingface/token").read().strip())
    repo = f"ethanelasky/ckpt-{run}"[:96]
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(d), repo_id=repo, repo_type="model",
                      path_in_repo=d.name)
print("ok")
PYEOF
    if [ "$upload_ok" = 1 ]; then
      echo "$dir" >> "$STATE"
    else
      echo "[$(date -u +%H:%M:%S)] upload FAILED for $dir; will retry"
    fi
  done
  sleep $INTERVAL
done
