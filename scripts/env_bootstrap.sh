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
URL="https://huggingface.co/datasets/$ENV_REPO/resolve/main/$TARBALL"
if command -v hf >/dev/null 2>&1; then
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

tar -I zstd -xf "/root/$TARBALL" -C /
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
