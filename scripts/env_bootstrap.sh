#!/usr/bin/env bash
# Restore a pre-built training env on a FRESH pod in ANY datacenter — the
# DC-independence path (Ethan, 2026-08-10). The tarball on HF holds
# /workspace/envs/$VENV_NAME plus /workspace/uv/python with /-relative paths,
# so extraction recreates the exact layout the matching provision script
# builds, whether /workspace is a network volume or plain container disk.
#
# Usage (on the pod):   HF_TOKEN=... bash scripts/env_bootstrap.sh
#
#   ENV_REPO=...        HF dataset repo holding the tarball
#   TARBALL=...         tarball filename inside that repo
#   VENV_NAME=...       env directory under /workspace/envs
#   ENV_LOCAL_ROOT=...  where the bytes land (default /opt/envroot, container
#                       disk); /workspace/envs/$VENV_NAME becomes a symlink
#   ENV_LOCAL_MIN_GB=.. free GB required there, else extract onto /workspace
#                       as before (default 40)
#
# A pod stop wipes container disk but keeps the volume, so the symlink is left
# dangling: -e follows it, reads false, and the env restores again.
#
# Defaults restore the Blackwell (sm100 / B200) env, which is what every
# existing caller expects. The sm90 (H100/H200) pair is:
#
#   ENV_REPO=ethanelasky/verl-sm90-env \
#   TARBALL=verl-sm90-portable.tar.zst \
#   VENV_NAME=verl-sm90 bash scripts/env_bootstrap.sh
#
# The three move together — a tarball built by provision_blackwell.sh contains
# verl-b200 and one built by provision_sm90.sh contains verl-sm90, so pointing
# ENV_REPO at one while leaving VENV_NAME at the other yields an env that
# extracts fine and then fails its import check.
#
# After it prints BOOTSTRAP_OK: PY=/workspace/envs/$VENV_NAME/bin/python as on
# volume-backed pods; pod_run.sh needs no changes. Models/checkpoints are NOT
# in the tarball — think-track models pull from HF, instruct-track bf16 needs
# the volume or a separate copy.
set -euo pipefail

ENV_REPO="${ENV_REPO:-ethanelasky/verl-b200-env}"
TARBALL="${TARBALL:-verl-b200-portable.tar.zst}"
VENV_NAME="${VENV_NAME:-verl-b200}"
ENV_DIR="/workspace/envs/$VENV_NAME"

if [ -e "$ENV_DIR/bin/python" ]; then
  echo "env already present at $ENV_DIR — nothing to do"
  exit 0
fi

command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; }

# hf CLI if present, else curl, else a stdlib-python streaming download (the
# base image has python3 but not necessarily huggingface_hub or curl).
#
# EVERY path must be able to give up. An unbounded read on a dropped
# connection does not fail, it hangs: a live restore stalled at 2.47GB and
# held a 2xB200 for 55 minutes until the job timeout, having written nothing
# for 54 of them. A stall now ends the transfer, and the transfer resumes from
# what is already on disk rather than starting the multi-GB download again.
cd /root
STAGE_START=$SECONDS
URL="https://huggingface.co/datasets/$ENV_REPO/resolve/main/$TARBALL"
if command -v hf >/dev/null 2>&1; then
  # Xet off. With it on the download holds live TLS sockets while every worker
  # thread sits in futex_wait -- twice on one H200, 2026-08-26, frozen at 2.0GB
  # of 3.64GB. Xet is the default in huggingface_hub >=1.x, so this has to be
  # turned OFF explicitly rather than merely not turned on.
  #
  # No retry loop here on purpose: a stalled restore is jobd's to notice
  # (stall_after_s) and jobd's to retry (max_attempts). A timeout+retry wrapper
  # in this script duplicated both, and GNU timeout setpgid()s itself into a new
  # process group, which put the download out from under jobd's cancel.
  export HF_HUB_DISABLE_XET=1
  # Deprecated in hub 1.x and ignored; unset so it stops printing a warning
  # that reads like a cause when somebody is debugging a slow restore.
  unset HF_HUB_ENABLE_HF_TRANSFER
  hf download "$ENV_REPO" "$TARBALL" --repo-type dataset --local-dir /root
elif command -v curl >/dev/null 2>&1; then
  # --speed-limit/--speed-time is the stall detector: under 100KB/s averaged
  # over 60s is a dead socket, not a slow one. --retry then reconnects and
  # --continue-at resumes at the byte already written.
  curl -fL --retry 8 --retry-delay 5 --retry-all-errors --continue-at - \
    --connect-timeout 30 --speed-limit 102400 --speed-time 60 \
    -H "Authorization: Bearer ${HF_TOKEN:?HF_TOKEN required to restore the env}" \
    -o "/root/$TARBALL" "$URL"
else
  URL="$URL" TARBALL="$TARBALL" python3 - <<'PYEOF'
import os, pathlib, urllib.request

url, tarball = os.environ["URL"], os.environ["TARBALL"]
tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
path = pathlib.Path("/root") / tarball
for attempt in range(8):
    have = path.stat().st_size if path.exists() else 0
    headers = {"Authorization": f"Bearer {tok}"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        # timeout applies to each socket read, so a dead connection raises
        # instead of blocking forever.
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=120
        ) as r:
            if have and r.status != 206:  # server ignored Range: start over
                have, mode = 0, "wb"
            else:
                mode = "ab" if have else "wb"
            total = have + int(r.headers.get("Content-Length") or 0)
            with open(path, mode) as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
        if total and path.stat().st_size >= total:
            break
    except Exception as exc:  # noqa: BLE001 — any transport failure retries
        print(f"download attempt {attempt + 1} failed at {have} bytes: {exc}", flush=True)
else:
    raise SystemExit("env tarball download did not complete after 8 attempts")
PYEOF
fi

echo "download: $((SECONDS - STAGE_START))s"
STAGE_START=$SECONDS

# Extract onto container disk, not onto /workspace.
#
# /workspace is a RunPod network volume: MooseFS over FUSE. Measured on a live
# B200, creating a file there costs 6.85ms against 0.084ms on the container
# overlay, and sequential write runs 455MB/s against 3.3GB/s. The b200 env is
# 19GB across 93,342 files, so extracting in place spends ~11 minutes in
# metadata round trips where local disk takes ~14 seconds.
#
# The venv bakes absolute /workspace/envs/<name> paths into its shebangs and
# pyvenv.cfg, so that path has to keep resolving -- hence a symlink rather
# than a different prefix.
LOCAL_ROOT="${ENV_LOCAL_ROOT:-/opt/envroot}"
NEED_GB="${ENV_LOCAL_MIN_GB:-40}"
mkdir -p "$LOCAL_ROOT"
AVAIL_GB=$(df -BG --output=avail "$LOCAL_ROOT" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -ge "$NEED_GB" ]; then
  echo "extracting to $LOCAL_ROOT (${AVAIL_GB}G free), linked into /workspace"
  tar -I zstd -xf "/root/$TARBALL" -C "$LOCAL_ROOT"
  # One link per leaf the tarball actually placed -- envs/<name>, uv/python --
  # rather than one link over /workspace/envs, so restoring a second env onto
  # this pod later does not have to share the link.
  find "$LOCAL_ROOT/workspace" -mindepth 2 -maxdepth 2 | while IFS= read -r SRC; do
    DST="/workspace/${SRC#"$LOCAL_ROOT"/workspace/}"
    mkdir -p "$(dirname "$DST")"
    # An earlier restore that died mid-extract leaves a real directory here,
    # and ln would then put the link INSIDE it and hide the whole env.
    rm -rf "$DST"
    ln -sfn "$SRC" "$DST"
  done
else
  # A slow restore beats a run that dies on ENOSPC an hour in.
  echo "only ${AVAIL_GB}G free on $LOCAL_ROOT; extracting onto /workspace" >&2
  rmdir "$LOCAL_ROOT" 2>/dev/null || true
  tar -I zstd -xf "/root/$TARBALL" -C /
fi
echo "extract: $((SECONDS - STAGE_START))s"
rm -f "/root/$TARBALL"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "FATAL: $TARBALL did not contain $ENV_DIR — ENV_REPO/TARBALL/VENV_NAME disagree" >&2
  exit 1
fi

# The venv must import its heavy deps through the restored interpreter path.
"$ENV_DIR/bin/python" - <<'PYEOF'
import torch, vllm, verl  # noqa: F401
print("imports ok:", torch.__version__, vllm.__version__)
PYEOF
echo BOOTSTRAP_OK
