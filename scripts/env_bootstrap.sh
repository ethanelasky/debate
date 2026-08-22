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
# verl-b200 and one built by provision_pod.sh contains verl-sm90, so pointing
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

# hf CLI if present, else a stdlib-python streaming download (base image has
# python3 but not necessarily huggingface_hub).
cd /root
if command -v hf >/dev/null 2>&1; then
  hf download "$ENV_REPO" "$TARBALL" --repo-type dataset --local-dir /root
else
  ENV_REPO="$ENV_REPO" TARBALL="$TARBALL" python3 - <<'PYEOF'
import os, urllib.request
repo = os.environ["ENV_REPO"]
tarball = os.environ["TARBALL"]
tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
url = f"https://huggingface.co/datasets/{repo}/resolve/main/{tarball}"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
with urllib.request.urlopen(req) as r, open(f"/root/{tarball}", "wb") as f:
    while chunk := r.read(1 << 22):
        f.write(chunk)
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
