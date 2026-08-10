#!/usr/bin/env bash
# Restore the pre-built Blackwell training env on a FRESH pod in ANY
# datacenter — the DC-independence path (Ethan, 2026-08-10). The tarball on HF
# holds /workspace/envs/verl-b200 plus /workspace/uv/python with /-relative
# paths, so extraction recreates the exact layout provision_blackwell.sh
# builds, whether /workspace is a network volume or plain container disk.
#
# Usage (on the pod):   HF_TOKEN=... bash scripts/env_bootstrap.sh
#   ENV_REPO=...        override the HF dataset repo (default below)
#
# After it prints BOOTSTRAP_OK: PY=/workspace/envs/verl-b200/bin/python as on
# volume-backed pods; pod_run.sh needs no changes. Models/checkpoints are NOT
# in the tarball — think-track models pull from HF, instruct-track bf16 needs
# the volume or a separate copy.
set -euo pipefail

ENV_REPO="${ENV_REPO:-ethanelasky/verl-b200-env}"
TARBALL="verl-b200-portable.tar.zst"

if [ -e /workspace/envs/verl-b200/bin/python ]; then
  echo "env already present at /workspace/envs/verl-b200 — nothing to do"
  exit 0
fi

command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; }

# hf CLI if present, else a stdlib-python streaming download (base image has
# python3 but not necessarily huggingface_hub).
cd /root
if command -v hf >/dev/null 2>&1; then
  hf download "$ENV_REPO" "$TARBALL" --repo-type dataset --local-dir /root
else
  python3 - <<'PYEOF'
import os, urllib.request
repo = os.environ.get("ENV_REPO", "ethanelasky/verl-b200-env")
tok = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
url = f"https://huggingface.co/datasets/{repo}/resolve/main/verl-b200-portable.tar.zst"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
with urllib.request.urlopen(req) as r, open("/root/verl-b200-portable.tar.zst", "wb") as f:
    while chunk := r.read(1 << 22):
        f.write(chunk)
PYEOF
fi

tar -I zstd -xf "/root/$TARBALL" -C /
rm -f "/root/$TARBALL"

# The venv must import its heavy deps through the restored interpreter path.
/workspace/envs/verl-b200/bin/python - <<'PYEOF'
import torch, vllm, verl  # noqa: F401
print("imports ok:", torch.__version__, vllm.__version__)
PYEOF
echo BOOTSTRAP_OK
