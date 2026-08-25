#!/usr/bin/env bash
# Emit one line per thing an agent would act on for a running jobd job.
#
# Written for Monitor: stdout is the event stream, one notification per line.
# The design constraint is volume. A naive filter that echoes every [step N]
# produces a notification every few minutes -- several hundred over a real run
# -- and the harness throttles a monitor that noisy, so the watch dies exactly
# when you stopped paying attention to it. So progress is reported by EXCEPTION:
# silence means stepping normally, and you hear about it when it stops.
#
#   failures        -> immediately, deduplicated
#   no progress     -> once, after STALL polls without the step number moving
#   resumed         -> once, if it starts moving again
#   heartbeat       -> every BEAT polls, so silence is never ambiguous
#
# jobd logs is a SNAPSHOT, not a stream (for a live job it ssh's to the pod and
# tails once), so this polls. That is one bounded ssh per POLL seconds.
#
# Usage:  scripts/watch_run.sh <job-id-or-name>
#         POLL=60 STALL=15 BEAT=30 scripts/watch_run.sh my-job
set -uo pipefail

JOB="${1:?usage: watch_run.sh <job-id-or-name>}"
POLL="${POLL:-60}"      # seconds between polls
STALL="${STALL:-15}"    # polls without progress before calling it stalled
BEAT="${BEAT:-30}"      # polls between "still alive" heartbeats

# The repo's own failure convention: pod_run.sh, env_bootstrap.sh and the
# provision scripts announce every fatal as "FATAL: ...", deliberately so it is
# greppable. The rest are runtime deaths that would not go through that path.
FAIL='FATAL|Traceback|CUDA out of memory|no kernel image is available|OutOfMemoryError|Killed|Refusing to start'

# jobd may be a console script or only importable; accept either.
jobd_cmd() {
  if command -v jobd >/dev/null 2>&1; then jobd "$@"
  else python3 -m jobd "$@"; fi
}

seen=""; last=""; quiet=0; ticks=0
while true; do
  out="$(jobd_cmd logs "$JOB" --tail 400 2>/dev/null || true)"

  # Failures: report each distinct one once. Sorting is safe here because these
  # are alerts, not a reconstruction of the log.
  hits="$(printf '%s\n' "$out" | grep -E "$FAIL" || true)"
  if [ -n "$hits" ]; then
    comm -13 <(printf '%s\n' "$seen" | sort -u) <(printf '%s\n' "$hits" | sort -u) | cut -c1-200
    seen="$hits"
  fi

  step="$(printf '%s\n' "$out" | grep -oE '\[step [0-9]+\]' | tail -1)"
  if [ -n "$step" ] && [ "$step" != "$last" ]; then
    [ "$quiet" -ge "$STALL" ] && echo "$JOB: progress resumed at $step"
    last="$step"; quiet=0
  else
    quiet=$((quiet + 1))
    [ "$quiet" -eq "$STALL" ] && \
      echo "$JOB: STALLED - no progress past ${last:-<no step yet>} in $((STALL * POLL / 60))m"
  fi

  ticks=$((ticks + 1))
  [ $((ticks % BEAT)) -eq 0 ] && echo "$JOB: alive, ${last:-<pre-training>}"
  sleep "$POLL"
done
