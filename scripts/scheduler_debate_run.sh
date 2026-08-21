#!/usr/bin/env bash
# Debate-owned half of a scheduler RunPod attempt.
#
# The scheduler/remote supervisor owns CREATE, the compute deadline, the outer
# terminal record, collection, and STOP. This wrapper owns one exact scientific
# command and the retained worker-side evidence needed to collect it. It accepts
# no arguments, so neither a job document nor ambient shell state can append a
# resume or warm-start flag.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly PATH

if [ "$#" -ne 0 ]; then
  echo "scheduler_debate_run: no arguments are accepted" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

case "$REPO_ROOT/" in
  /workspace/*) ;;
  *)
    echo "scheduler_debate_run: frozen staged repository must be below /workspace" >&2
    echo "scheduler_debate_run: got $REPO_ROOT" >&2
    exit 2
    ;;
esac

: "${DEBATE_LAUNCH_NAMESPACE:?scheduler must set DEBATE_LAUNCH_NAMESPACE}"
: "${DEBATE_ARTIFACT_ROOT:?scheduler must set DEBATE_ARTIFACT_ROOT}"
: "${DEBATE_CHECKPOINT_DESTINATION_FILE:?scheduler must set DEBATE_CHECKPOINT_DESTINATION_FILE}"
: "${DEBATE_ATTEMPT_IDENTITY_SHA256:?scheduler must set DEBATE_ATTEMPT_IDENTITY_SHA256}"
: "${DEBATE_SNAPSHOT_SHA256:?scheduler must set DEBATE_SNAPSHOT_SHA256}"
: "${DEBATE_DEADLINE_EPOCH:?scheduler must set DEBATE_DEADLINE_EPOCH}"
: "${RUNPOD_POD_ID:?provider must set RUNPOD_POD_ID}"

if [ "$EUID" != 10001 ]; then
  echo "scheduler_debate_run: dedicated uid 10001 is required" >&2
  exit 2
fi

# Freeze every operational override here. In particular, ambient CONFIG,
# checkpoint JSON, idle-watchdog, resume, or runner knobs cannot change the
# command whose handshake is recorded. Provider-template credential variables
# remain in the environment; neither this wrapper nor its evidence serializes
# their values.
unset CKPT_DESTINATION_JSON CKPT_DIR CKPT_SYNC_ONCE CKPT_SYNC_STATE \
  CKPT_SYNC_PID_FILE CKPT_SYNC_LOCK_FILE CONFIG POD_IDLE_STOP \
  DEBATE_SCHEDULER_MODE POD_RUN_STATE_DIR PY PYBIN
export CONFIG=configs/math_pc_debate.yaml
export POD_IDLE_STOP=0
export DEBATE_SCHEDULER_MODE=1
export PY=/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12
export PYBIN=/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12
export PYTHONDONTWRITEBYTECODE=1

SUPERVISOR_PY=/usr/bin/python3
readonly SUPERVISOR_PY
if [ ! -x "$SUPERVISOR_PY" ]; then
  echo "scheduler_debate_run: trusted system Python is unavailable" >&2
  exit 2
fi
exec "$SUPERVISOR_PY" -I -S \
  "$SCRIPT_DIR/scheduler_debate_supervisor.py" --repo-root "$REPO_ROOT"
