#!/usr/bin/env bash
# Provider-free D043 build-input frontier. This deliberately performs no
# dependency resolution, install, image build, network access, or provider call.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
LOCK="$SCRIPT_DIR/dependency.lock"
SPEC="$SCRIPT_DIR/build-spec.json"
SEED="${DEBATE_RUNTIME_BINARY_SEED:-}"
RAW_SEED="${DEBATE_RUNTIME_UNCOMPRESSED_SEED:-}"

if [ ! -f "$LOCK" ] || [ ! -f "$SPEC" ] || { [ -z "$SEED" ] && [ -z "$RAW_SEED" ]; } || { [ -n "$SEED" ] && [ -n "$RAW_SEED" ]; }; then
  echo "D043 REFUSED: the tracked lock/spec and exactly one content-addressed seed are required" >&2
  echo "D043 REFUSED: set DEBATE_RUNTIME_BINARY_SEED or DEBATE_RUNTIME_UNCOMPRESSED_SEED" >&2
  echo "D043 REFUSED: the seed must be the exact pinned HF revision/LFS object; no fetch or resolution is performed" >&2
  exit 2
fi

if [ -n "$SEED" ]; then
  exec /usr/bin/python3 -I -S "$SCRIPT_DIR/generate_runtime_metadata.py" \
    --dependency-lock "$LOCK" \
    --build-spec "$SPEC" \
    --binary-seed "$SEED" \
    --check-inputs-only
fi

exec /usr/bin/python3 -I -S "$SCRIPT_DIR/generate_runtime_metadata.py" \
  --dependency-lock "$LOCK" \
  --build-spec "$SPEC" \
  --uncompressed-seed "$RAW_SEED" \
  --check-inputs-only
