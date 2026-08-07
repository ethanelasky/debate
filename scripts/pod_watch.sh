#!/usr/bin/env bash
# Watch a training run and report only what it can actually establish.
#
#   bash scripts/pod_watch.sh <ip> <port> <logfile> <expected_steps>
#
# Written because ad-hoc watchers got this wrong twice, in both directions:
#
#   One declared cap1024 DEAD sixty seconds after launch. The trainer was mid
#   CUDA-graph capture -- pod_run.sh had not yet exec'd it, so no process
#   matched. The watcher then started the NEXT arm, which pod_run.sh's trainer
#   guard correctly refused; that guard is the only reason the real run survived.
#
#   Another reported ALL ARMS DONE for runs that had died in the first minute,
#   because "process gone" and "finished" look the same to pgrep.
#
# Hence two rules, both load-bearing:
#   1. A run is not dead until GRACE has elapsed AND two consecutive polls miss
#      it. Model load plus graph capture runs ~8 min before the first step line.
#   2. Absence is never success. Completion requires the step COUNT to have
#      reached the target; anything less is reported as a death, with the log.
#
# Exit codes:
#   0  completed: reached expected_steps
#   1  died: process gone before reaching expected_steps
#   2  bad arguments
#   3  still running when MAX_POLLS ran out (no verdict -- not a failure)
set -euo pipefail

IP="${1:?usage: pod_watch.sh <ip> <port> <logfile> <expected_steps>}"
PORT="${2:?usage: pod_watch.sh <ip> <port> <logfile> <expected_steps>}"
LOG="${3:?usage: pod_watch.sh <ip> <port> <logfile> <expected_steps>}"
WANT="${4:?usage: pod_watch.sh <ip> <port> <logfile> <expected_steps>}"

GRACE_POLLS="${GRACE_POLLS:-12}"   # polls before a miss can mean anything (~12min)
MAX_POLLS="${MAX_POLLS:-360}"
INTERVAL="${INTERVAL:-60}"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $PORT root@$IP"

dead=0
for i in $(seq 1 "$MAX_POLLS"); do
  # `[i]nfra` so the pattern does not match this very command line -- a pkill
  # written without that trick once matched its own ssh invocation and killed
  # the shell before the kill landed, silently producing no output.
  N=$($SSH "grep -c '^\[step' $LOG 2>/dev/null" 2>/dev/null | tr -d '\r ' || echo "")
  A=$($SSH "ps -eo args | grep -c '[i]nfra.run_rlvr'" 2>/dev/null | tr -d '\r ' || echo "")
  echo "[${i}m] steps=${N:-?}/$WANT alive=${A:-?}"

  if [ "${A:-1}" = "0" ]; then
    dead=$((dead + 1))
    if [ "$dead" -ge 2 ] && [ "$i" -gt "$GRACE_POLLS" ]; then
      if [ "${N:-0}" -ge "$WANT" ]; then
        echo "COMPLETED ($N steps)"
        exit 0
      fi
      echo "DIED after ${N:-0}/$WANT steps. Tail:" >&2
      $SSH "tail -30 $LOG" >&2 || true
      exit 1
    fi
  else
    dead=0
  fi
  sleep "$INTERVAL"
done

echo "NO VERDICT: still running after $MAX_POLLS polls -- watch expired, the run did not" >&2
exit 3
