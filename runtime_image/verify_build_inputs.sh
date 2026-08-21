#!/usr/bin/env bash
# Provider-free D043 build-input frontier. This deliberately performs no
# dependency resolution, install, image build, network access, or provider call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
LOCK="$SCRIPT_DIR/dependency.lock"
SPEC="$SCRIPT_DIR/build-spec.json"

if [ ! -f "$LOCK" ] || [ ! -f "$SPEC" ]; then
  echo "D043 REFUSED: runtime_image/dependency.lock and build-spec.json are not yet available" >&2
  echo "D043 REFUSED: do not substitute pyproject.toml, uv.lock, a pip freeze, or the mutable /workspace environment" >&2
  exit 2
fi

exec /usr/bin/python3 -I -S "$SCRIPT_DIR/generate_runtime_metadata.py" \
  --dependency-lock "$LOCK" \
  --build-spec "$SPEC" \
  --check-inputs-only
