#!/usr/bin/env bash
# Pod-side idle watchdog: reaps the leftover judge vLLM and then STOPS the pod,
# so an H100 does not bill overnight after the last eval finished. Normally
# launched detached by pod_run.sh (setsid nohup ... > /root/pod_idle_stop.out),
# but safe to start by hand:
#   IDLE_MINUTES=45 setsid nohup bash scripts/pod_idle_stop.sh > /root/pod_idle_stop.out 2>&1 &
#   bash scripts/pod_idle_stop.sh --once     # print the busy/idle verdict and exit; kills nothing
# STOP, never remove: stopping ends GPU billing, but RunPod WIPES the container
# disk — /root, including the synced repo at /root/debate — on stop. Only the
# /workspace network volume survives. Anything under /root that is worth keeping
# must be evacuated to /workspace BEFORE the stop (see evacuate() below).
# Destroying a pod is a human decision and this script never makes it.
set -euo pipefail

# Terminal WRITES count as human activity, and two different things produce
# them. (1) An ATTACHED tmux client: its status line redraws every ~15s and
# refreshes the pty's mtime, so the pod stays busy for as long as the session
# has a client. (2) Any PRINTING PROGRAM left running in a pane — `watch`,
# `top`, `tail -f`, a progress spinner — which keeps writing to the pane's pty
# slave whether or not a client is attached; detaching does NOT quiet it. So
# close web-terminal tabs AND stop long-lived monitors when walking away, or the
# pod cannot auto-stop. A detached session whose panes are all sitting at an
# idle shell prompt does not pin it.
# Also: a monitor whose own command line embeds one of the guarded trainer
# strings (e.g. `watch pgrep -f 'python -m infra.run_rlvr'`) is itself matched by
# busy pattern (a) and reads as a live run. Operator hint: write such monitors
# with patterns that cannot match themselves literally, e.g.
# `watch pgrep -f 'python -m infra.run_[r]lvr'`.
IDLE_MINUTES="${IDLE_MINUTES:-30}"    # consecutive idle minutes required to tear down
GRACE_MINUTES="${GRACE_MINUTES:-10}"  # never tear down this soon after start (slow trainer boot, pip installs)
ONCE=0
if [ "${1:-}" = "--once" ]; then ONCE=1; fi

# Bounding comes FIRST, before anything else in this script touches /workspace:
# the very first writes are the log-path setup below, and an unbounded mkdir or
# touch against a sick NFS volume wedges the watchdog in uninterruptible sleep
# at startup. pod_run.sh's `kill -0` then sees a live pid, reports the run
# protected, and the pod bills all night behind a watchdog that never armed.
# Every external command the watchdog runs must be bounded for the same reason:
# a hung probe on a sick pod wedges the watchdog precisely when it is the only
# thing standing between a dead run and an overnight H100 bill.
HAVE_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then HAVE_TIMEOUT=1; fi
run_bounded() {
  local secs="$1"; shift
  if [ "$HAVE_TIMEOUT" -eq 1 ]; then
    timeout -k 15 "$secs" "$@"
  else
    "$@"
  fi
}

# /workspace is the network volume and SURVIVES a pod stop; /root does not. The
# whole point of the log is to be readable after the pod is off, when the only
# evidence left of why an idle pod died is this file. Both steps are bounded:
# a timeout is treated exactly like a permission failure and falls back to the
# container disk, so a stalled volume costs us the durable log, never the
# watchdog itself.
LOG=/workspace/logs/pod_idle_stop.log
LOG_FALLBACK=0
LOG_WARN=""
if ! run_bounded 15 mkdir -p /workspace/logs 2>/dev/null \
   || ! run_bounded 15 touch "$LOG" 2>/dev/null; then
  LOG=/root/pod_idle_stop.log
  LOG_FALLBACK=1
  LOG_WARN="WARNING: /workspace not writable or not responding within 15s; logging to $LOG, which is WIPED when the pod stops"
fi

# Logging must never kill or wedge the watchdog. /workspace is NFS: during a
# volume outage or a maintenance window an append can FAIL — fatal under set -e
# — or HANG forever, in both cases exactly when the watchdog matters most; that
# is the "bills the pod all night" failure this script exists to prevent. So:
# bound every write, swallow every error. stderr is silenced BEFORE the append
# so a failing redirect stays quiet too.
log() {
  local line
  line="$(date -u +%FT%TZ) $*"
  if [ "$HAVE_TIMEOUT" -eq 1 ]; then
    timeout -k 5 10 sh -c 'printf "%s\n" "$1" 2>/dev/null >> "$2"' _ "$line" "$LOG" 2>/dev/null || true
  else
    printf '%s\n' "$line" 2>/dev/null >> "$LOG" || true
  fi
}
if [ -n "$LOG_WARN" ]; then log "$LOG_WARN"; echo "$LOG_WARN" >&2; fi

# --- startup checks: fail loudly, the launcher's .out file is the surfacing point
fatal() { log "FATAL: $*"; echo "FATAL: $*" >&2; exit 1; }
if [ "$ONCE" -eq 0 ]; then
  # A non-numeric threshold NEVER tears down: `[ 30m -ge ... ]` exits rc 2, which
  # the enclosing `if` reads as false on every single tick, so the pod bills
  # forever behind a watchdog that looks healthy. Reject it here, at launch,
  # while a human is still reading the launcher's output — same philosophy as
  # the auth probe below.
  case "${IDLE_MINUTES}" in
    ''|*[!0-9]*) fatal "IDLE_MINUTES='${IDLE_MINUTES}' is not an integer number of minutes (digits only, e.g. IDLE_MINUTES=45); a non-numeric threshold silently never fires and the pod would bill forever" ;;
  esac
  case "${GRACE_MINUTES}" in
    ''|*[!0-9]*) fatal "GRACE_MINUTES='${GRACE_MINUTES}' is not an integer number of minutes (digits only, e.g. GRACE_MINUTES=10); a non-numeric grace window silently never expires and the pod would bill forever" ;;
  esac

  [ -n "${RUNPOD_POD_ID:-}" ] || fatal "RUNPOD_POD_ID is unset; without a pod id this watchdog cannot stop anything — the pod will bill until stopped by hand"
  command -v runpodctl >/dev/null 2>&1 || fatal "runpodctl not on PATH; the watchdog could reap the judge but never stop the pod, so it refuses to run at all"

  # Two watchdogs would race on the teardown and double-stop, so the newer one
  # stands down — but ZERO watchdogs is the expensive outcome, so the OLDEST
  # instance always survives: we exit only for an instance CLEARLY older than
  # us. If ages cannot be read we keep running; two watchdogs are survivable,
  # none is not.
  # "Clearly older" means 2+ whole seconds, and our own age is re-read for
  # every comparison rather than sampled once before the loop. `etimes` is
  # integer seconds and our age advances between reads, so a single early
  # sample makes us look younger than we are; two watchdogs started within the
  # same second can then each read the other as strictly older and BOTH exit,
  # leaving the pod unwatched. Inside a ±1s band the ages are a tie and the pid
  # comparison alone decides, which no rounding can read asymmetrically: the
  # lower pid survives on both sides.
  # The pattern requires the `bash <path>/pod_idle_stop.sh` command form so an
  # editor, pager or grep with this filename on its command line is not mistaken
  # for a running watchdog. Interpreter flags between `bash` and the script path
  # are tolerated — a watchdog launched for debugging as
  # `bash -x scripts/pod_idle_stop.sh` is a real watchdog, and a pattern that
  # missed it would let a second instance start alongside it and race on the
  # teardown. pod_run.sh's sibling pattern (live_watchdog_pids) matches the same
  # forms; the two must agree on what a running watchdog looks like.
  # Elapsed seconds for a pid, or empty when unreadable. `etimes` is a procps
  # keyword: a BSD ps (darwin) answers by printing its whole keyword list, so
  # anything that is not purely digits is discarded rather than compared.
  etimes_of() {
    local v
    v="$(ps -o etimes= -p "$1" 2>/dev/null | head -n 1 | tr -d '[:space:]')" || v=""
    case "${v:-x}" in *[!0-9]*) v="" ;; esac
    printf '%s' "$v"
  }
  for p in $(pgrep -f 'bash( -[^ ]+)* [^ ]*pod_idle_stop\.sh' 2>/dev/null || true); do
    [ "$p" = "$$" ] && continue
    [ "$p" = "$PPID" ] && continue
    # A --once verdict run exits in seconds and never stops anything.
    OTHER_ARGS="$(ps -o args= -p "$p" 2>/dev/null || true)"
    case "$OTHER_ARGS" in *--once*) continue ;; esac
    SELF_ETIMES="$(etimes_of "$$")"
    OTHER_ETIMES="$(etimes_of "$p")"
    if [ -z "$SELF_ETIMES" ] || [ -z "$OTHER_ETIMES" ]; then
      log "another pod_idle_stop.sh is running (pid $p) but process ages are unreadable; staying up rather than risk leaving no watchdog"
      continue
    fi
    if [ "$OTHER_ETIMES" -ge $(( SELF_ETIMES + 2 )) ] \
       || { [ "$OTHER_ETIMES" -le $(( SELF_ETIMES + 1 )) ] \
            && [ "$OTHER_ETIMES" -ge $(( SELF_ETIMES - 1 )) ] \
            && [ "$p" -lt "$$" ]; }; then
      log "another pod_idle_stop.sh has seniority (pid $p, ${OTHER_ETIMES}s vs our ${SELF_ETIMES}s — older, or same age with a lower pid); exiting"
      exit 0
    fi
  done

  # The first real API call would otherwise happen 30+ unattended idle minutes
  # from now: an unauthorized or misconfigured runpodctl must fail HERE, while a
  # human is still watching the launcher's output, not at teardown time.
  if ! run_bounded 60 runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
    fatal "'runpodctl get pod $RUNPOD_POD_ID' failed or timed out at startup; the watchdog would not be able to stop the pod either — check RUNPOD_API_KEY in the environment and 'runpodctl config' / auth, and that the pod id is correct"
  fi

  # ARMED MARKER — the launcher's proof that protection actually exists. Every
  # startup check above has now passed, so from here the watchdog will run. A
  # launcher cannot infer that from timing: `runpodctl get pod` may hang to its
  # 60s bound and FATAL long after a `kill -0` liveness check said "alive", and
  # the multi-hour run would then proceed unprotected. pod_run.sh waits for this
  # file instead of guessing. Nothing ever deletes it: a pod stop wipes /root,
  # and a stale marker left by a dead watchdog is handled on the consumer side
  # by checking the recorded pid is still a live watchdog.
  # Bounded like every other write, but /root is container-local disk and cannot
  # hang the way the NFS volume can; a failure here is logged and tolerated
  # rather than fatal, since a working watchdog with no marker is still worth
  # far more than no watchdog.
  ARMED_MARKER=/root/pod_idle_stop.armed
  if run_bounded 15 sh -c 'printf "%s\n" "$1" > "$2"' _ "$$" "$ARMED_MARKER" 2>/dev/null; then
    log "armed (pid $$ recorded in $ARMED_MARKER)"
  else
    log "WARNING: armed, but could not write the marker $ARMED_MARKER; a launcher waiting on it will time out even though this watchdog is running"
  fi
fi

NVIDIA_SMI_WARNED=0
CGROUP_CPU_WARNED=0
CPU_PREV_USEC=""   # container CPU usage at the previous poll, microseconds
CPU_PREV_AT=""     # $SECONDS at that poll, so the rate uses the ACTUAL interval

# Which cgroup files to read, resolved once at startup — the membership never
# changes for a running process.
#
# The ROOT files (/sys/fs/cgroup/cpu.stat, .../cpuacct/cpuacct.usage) are only
# OUR container's accounting when the container has a PRIVATE cgroup namespace.
# Under cgroupns=host — the older Docker default, and entirely plausible on a
# RunPod host — the root files aggregate the WHOLE HOST: every tenant's CPU. A
# signal fed by that would sit above any threshold forever and never let the
# watchdog stop the pod, which is the same failure that disqualified
# /proc/loadavg. So ask /proc/self/cgroup where WE actually live and read the
# per-container file; the root files stay as the fallback, which is correct
# under a private namespace (the path is `/` and it is the same file) and is
# no worse than before when /proc/self/cgroup is missing or unparseable.
CGROUP_V2_STAT=""
CGROUP_V1_USAGE=""
if [ -r /proc/self/cgroup ]; then
  # v2: a single `0::<path>` line.
  CG_PATH="$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup 2>/dev/null || true)"
  case "$CG_PATH" in
    /*)
      if [ "$CG_PATH" = / ]; then CG_PATH=""; fi   # root: no cosmetic '//' in paths and logs
      if [ -r "/sys/fs/cgroup${CG_PATH}/cpu.stat" ]; then
        CGROUP_V2_STAT="/sys/fs/cgroup${CG_PATH}/cpu.stat"
      fi
      ;;
  esac
  # v1: the line whose controller list contains cpuacct, e.g. `4:cpu,cpuacct:<path>`.
  CG_PATH="$(awk -F: '$2 ~ /(^|,)cpuacct(,|$)/ { print $3; exit }' /proc/self/cgroup 2>/dev/null || true)"
  case "$CG_PATH" in
    /*)
      if [ "$CG_PATH" = / ]; then CG_PATH=""; fi
      for c in "/sys/fs/cgroup/cpu,cpuacct${CG_PATH}/cpuacct.usage" \
               "/sys/fs/cgroup/cpuacct${CG_PATH}/cpuacct.usage"; do
        if [ -r "$c" ]; then CGROUP_V1_USAGE="$c"; break; fi
      done
      ;;
  esac
  unset CG_PATH
fi
[ -n "$CGROUP_V2_STAT" ]  || CGROUP_V2_STAT=/sys/fs/cgroup/cpu.stat
[ -n "$CGROUP_V1_USAGE" ] || CGROUP_V1_USAGE=/sys/fs/cgroup/cpuacct/cpuacct.usage

# Total CPU time this CONTAINER has burned, in microseconds, or non-zero when
# no cgroup accounting is readable. cgroup v2 first (the common case on modern
# hosts), then the v1 nanosecond counter.
cgroup_cpu_usec() {
  local v=""
  if [ -r "$CGROUP_V2_STAT" ]; then
    v="$(awk '$1 == "usage_usec" { print $2; exit }' "$CGROUP_V2_STAT" 2>/dev/null || true)"
    case "${v:-x}" in *[!0-9]*) v="" ;; esac
    if [ -n "$v" ]; then printf '%s' "$v"; return 0; fi
  fi
  if [ -r "$CGROUP_V1_USAGE" ]; then
    v="$(head -n 1 "$CGROUP_V1_USAGE" 2>/dev/null | tr -d '[:space:]')" || v=""
    case "${v:-x}" in *[!0-9]*) v="" ;; esac
    if [ -n "$v" ]; then printf '%s' "$(( v / 1000 ))"; return 0; fi   # ns -> usec
  fi
  return 1
}

# Sets BUSY_REASON and returns 0 when the pod is doing real work.
# NOTE: vLLM processes are deliberately NOT busy signals. An idle serving judge
# sits at ~0% util while pinning ~14GB forever — it is exactly what this script
# exists to reap, so counting it as work would make the watchdog never fire.
is_busy() {
  BUSY_REASON=""
  local pids

  # (a) trainer / eval entrypoints in every form they are actually launched:
  #     the module form (`python -m infra.run_debate`), the DIRECT FILE PATH
  #     form (`$PY infra/run_debate.py`), which matches no module pattern and
  #     used to slip the guard entirely, the launchers, and the console scripts
  #     (pyproject maps debate-rl to infra.run_debate:main and debate-train to
  #     infra.train:main). No pid is excluded here: pod_run.sh execs the
  #     trainer, so the launcher's pid IS the live trainer's pid and excluding it
  #     made the watchdog blind to the run it babysits. The watchdog's own
  #     cmdline (`bash .../pod_idle_stop.sh`) matches none of these patterns.
  #     Each pattern demands a COMMAND form, exactly as pod_run.sh's own trainer
  #     guard does, so an editor, a pager, a grep or a wrapper shell that merely
  #     mentions one of these names is not read as a live run. A false positive
  #     on the busy side is NOT harmless here: it pins an H100 to billing
  #     forever, which is the failure this whole script exists to prevent (a
  #     stray zsh carrying "pod_run.sh" in its command string did exactly that).
  #     The module alternative tolerates interpreter flags between the
  #     interpreter and `-m` — `python -u -m infra.run_debate`,
  #     `python3 -X faulthandler -m infra.run_rlvr` — in the same
  #     flag-anchored form pod_run.sh's guard uses, so the two agree on what a
  #     live trainer looks like. They diverged: the guard accepted those forms
  #     while this pattern did not, so a `-u`-launched trainer wedged on a dead
  #     judge read as idle here and could be stopped mid-run. The two launcher
  #     patterns tolerate bash interpreter flags for the same reason, in the same
  #     form the watchdog-seniority pattern above uses: `bash -x
  #     scripts/pod_run.sh` is a real run, and a pattern that missed it would let
  #     the watchdog stop a pod mid-provision.
  for pat in 'python[^ ]*( -[XW] [^ ]+| -[^ ]+)* -m infra\.(run_debate|run_rlvr|train)' \
             'python[^ ]* [^ ]*infra/(run_debate|run_rlvr|train)\.py' \
             'bin/debate-rl' \
             'bin/debate-train' \
             'bash( -[^ ]+)* [^ ]*pod_run\.sh' \
             'bash( -[^ ]+)* [^ ]*provision_pod\.sh'; do
    pids="$(pgrep -f "$pat" 2>/dev/null || true)"
    for p in $pids; do
      BUSY_REASON="process matching '$pat' (pid $p)"
      return 0
    done
  done

  # (b) a human is on the pod over ssh. Two spellings of the per-session
  # process exist: `sshd: user@pts/N` up to OpenSSH 9.7, and `sshd-session:
  # user@pts/N` from 9.8 on, where the session was split out of sshd. Matching
  # only the first spelling makes a live operator invisible on any modern
  # image, so the pattern accepts both.
  pids="$(pgrep -f 'sshd[^ ]*:.*@' 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    BUSY_REASON="active ssh session (pid $(echo "$pids" | tr '\n' ' '))"
    return 0
  fi

  # (c) GPU actually computing. A missing, broken or HUNG nvidia-smi reads as 0%
  # rather than crashing or wedging the loop — a watchdog that dies or blocks on
  # a probe error is a watchdog that bills the pod all night.
  local util=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    util="$(run_bounded 20 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  fi
  if [ -z "$util" ] && [ "$NVIDIA_SMI_WARNED" -eq 0 ]; then
    log "WARNING: nvidia-smi missing, hanging past its 20s bound, or returning nothing; GPU utilization treated as 0% (idle) from here on"
    NVIDIA_SMI_WARNED=1
  fi
  local i=0
  for u in $util; do
    case "$u" in
      ''|*[!0-9]*) i=$(( i + 1 )); continue ;;   # '[N/A]' on a wedged/MIG device
    esac
    if [ "$u" -gt 5 ]; then
      BUSY_REASON="gpu$i utilization ${u}%"
      return 0
    fi
    i=$(( i + 1 ))
  done

  # (d) a human typing somewhere that is not ssh: the RunPod web terminal and
  # Jupyter terminals are ptys, not sshd children. Recent WRITE activity on a
  # pty — not merely its existence — is the signal, so an active tmux counts
  # while a tmux left open and forgotten a week ago cannot pin the pod forever.
  # /dev/pts is Linux-only (darwin has /dev/ttys*, which is not equivalent);
  # any find error reads as not-busy.
  #
  # KNOWN LIMITATION, accepted deliberately: two distinct things keep a pty's
  # mtime fresh forever, and detaching only cures the first.
  #   1. An ATTACHED tmux client. The status line redraws every status-interval
  #      (~15s), which refreshes the pty's mtime, so a web-terminal browser TAB
  #      LEFT OPEN on an attached session holds this signal busy indefinitely.
  #   2. Any PRINTING PROGRAM left running in a pane — `watch`, `top`,
  #      `tail -f`, a progress bar. It writes to the pane's pty slave on its own
  #      schedule regardless of whether a client is attached, so a DETACHED
  #      session running one pins the pod just as hard as an attached one. It is
  #      only a detached session sitting at idle shell prompts that goes quiet.
  # Close web-terminal tabs and stop long-lived monitors when walking away.
  if [ -d /dev/pts ]; then
    local ptys=""
    ptys="$(find /dev/pts -type c -newermt '-120 seconds' 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${ptys// /}" ]; then
      BUSY_REASON="terminal activity in the last 120s on ${ptys% }"
      return 0
    fi
  fi

  # (e) CPU-side work with no process pattern and no GPU: a pod-side evacuation
  # (rsync/tar/huggingface-cli upload), a pip build, a Jupyter kernel crunching.
  # Killing the pod mid-upload is how artifacts get lost.
  #
  # Measured as OUR CONTAINER's cgroup CPU delta across the tick, NOT the load
  # average: /proc/loadavg is not namespaced, so a pod reads the whole HOST's
  # load. On a shared multi-tenant host that sits above 1.0 essentially always,
  # which would pin this signal to BUSY forever and stop the watchdog from ever
  # stopping the pod — the exact failure this script exists to prevent, arrived
  # at from the opposite direction.
  #
  # Busy = more than 30% of one core over the interval actually elapsed. The
  # first poll has no baseline and so never fires; neither does --once, which
  # only ever polls once (fine: --once reports the instantaneous signals).
  # Signals (a)-(d) return before reaching here, so a baseline can be older
  # than one tick — the arithmetic uses the real elapsed time, which turns that
  # into an average over a longer window rather than a wrong rate.
  local now_usec=""
  if now_usec="$(cgroup_cpu_usec)"; then
    if [ -n "$CPU_PREV_USEC" ] && [ -n "$CPU_PREV_AT" ]; then
      local elapsed=$(( SECONDS - CPU_PREV_AT ))
      if [ "$elapsed" -gt 0 ]; then
        local delta=$(( now_usec - CPU_PREV_USEC ))
        # 30% of one core for $elapsed seconds, in microseconds.
        if [ "$delta" -gt $(( elapsed * 300000 )) ]; then
          CPU_PREV_USEC="$now_usec"
          CPU_PREV_AT=$SECONDS
          BUSY_REASON="container cpu $(( delta / (elapsed * 10000) ))% of one core over ${elapsed}s"
          return 0
        fi
      fi
    fi
    CPU_PREV_USEC="$now_usec"
    CPU_PREV_AT=$SECONDS
  elif [ "$CGROUP_CPU_WARNED" -eq 0 ]; then
    log "WARNING: no readable cgroup CPU accounting ($CGROUP_V2_STAT or $CGROUP_V1_USAGE); container CPU is not a busy signal from here on"
    CGROUP_CPU_WARNED=1
  fi

  return 1
}

if [ "$ONCE" -eq 1 ]; then
  if is_busy; then
    echo "BUSY: $BUSY_REASON"
  else
    echo "IDLE"
  fi
  exit 0
fi

kill_judge() {
  # Parent-only kills leave VLLM::EngineCore children holding their GPU
  # allocation (see pod_run.sh); kill every layer by name.
  pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -9 -f 'EngineCore_DP' 2>/dev/null || true
}

# Copies everything worth keeping off the container disk, which the stop WIPES,
# onto the /workspace network volume, which survives.
#
# Failure policy: best-effort for logs, the uncommitted diff, the status listing
# and the untracked-file archive — each is attempted and logged on its own, and
# a failure of any one of them never blocks the rest. They are nice
# to have and none of them is worth a billed night. The per-step training
# transcripts under /root/debate/docent are NOT: they are unreproducible output
# of GPU-hours already paid for. If they exist and cannot be copied, this
# returns 1 and the caller ABORTS the teardown and keeps the pod up. One more
# billed cycle is cheap next to losing the run's transcripts, and a human can
# intervene — the log says exactly what happened and why the pod is still up.
evacuate() {
  # A FRESH directory per teardown. A fixed destination merge-overwrites: docent
  # step files restart at step-00000.jsonl on every run, so a second evacuation
  # would silently overwrite the first run's transcripts file-for-file while the
  # log still reported success. Timestamped, nothing can clobber anything.
  EVAC_DIR="/workspace/logs/evacuation/$(date -u +%Y%m%dT%H%M%SZ)"
  log "evacuating /root artifacts to $EVAC_DIR (the stop wipes the container disk)"
  if ! run_bounded 15 mkdir -p "$EVAC_DIR" 2>/dev/null; then
    log "WARNING: could not create $EVAC_DIR within 15s (is /workspace mounted, writable and responding?); evacuation will fail"
  fi

  local f
  for f in /root/judge_server.log /root/judge_server.log.prev /root/pod_idle_stop.out; do
    [ -f "$f" ] || continue
    if run_bounded 120 cp -f "$f" "$EVAC_DIR/" 2>/dev/null; then
      log "evacuated $f"
    else
      log "WARNING: could not evacuate $f (best-effort; continuing)"
    fi
  done

  # Operator-run trainers and evals commonly land their output straight in /root
  # (`nohup ... > /root/rlvr_smoke.log`), which the stop wipes like everything
  # else there. Glob-copied rather than named: the filenames are whatever the
  # operator chose. Our own log is skipped when it is HERE, i.e. the /workspace
  # fallback is active — copying a file that is still being appended to would
  # capture a torn snapshot, and the real fix in that case is the volume.
  local g
  # `*.out.prev` is globbed explicitly: pod_run.sh rotates the previous
  # watchdog's log to /root/pod_idle_stop.out.prev and keeps it precisely as
  # evidence of how that watchdog died, and neither *.log.prev nor *.out matches
  # it.
  for g in /root/*.log /root/*.log.prev /root/*.out /root/*.out.prev; do
    [ -f "$g" ] || continue                       # unmatched glob stays literal
    if [ "$LOG_FALLBACK" -eq 1 ] && [ "$g" = "$LOG" ]; then continue; fi
    case "$g" in /root/judge_server.log|/root/judge_server.log.prev|/root/pod_idle_stop.out) continue ;; esac
    if run_bounded 120 cp -f "$g" "$EVAC_DIR/" 2>/dev/null; then
      log "evacuated $g"
    else
      log "WARNING: could not evacuate $g (best-effort; continuing)"
    fi
  done

  if [ -d /root/debate/.git ]; then
    # `diff HEAD`, not plain `diff`: the latter shows unstaged changes only, so
    # anything the operator had `git add`ed was reported evacuated and lost.
    if run_bounded 120 sh -c 'git -C /root/debate diff HEAD > "$1"' _ "$EVAC_DIR/uncommitted.diff" 2>/dev/null; then
      log "evacuated uncommitted working-tree diff (staged + unstaged) to $EVAC_DIR/uncommitted.diff"
    else
      log "WARNING: could not capture the uncommitted diff of /root/debate (best-effort; continuing)"
    fi
    # The diff cannot represent untracked files at all, and the status listing
    # is what tells a human afterwards which files the diff and tarball SHOULD
    # contain — without it a partial capture is indistinguishable from a clean
    # tree.
    if run_bounded 120 sh -c 'git -C /root/debate status --porcelain > "$1"' _ "$EVAC_DIR/git-status.txt" 2>/dev/null; then
      log "evacuated working-tree status to $EVAC_DIR/git-status.txt"
    else
      log "WARNING: could not capture 'git status --porcelain' of /root/debate (best-effort; continuing)"
    fi
    # Untracked files appear in neither the diff nor the status listing's
    # contents, so they need an archive of their own. `docent` is excluded: it
    # is untracked but evacuated verbatim below, and tarring it here would
    # double both the time and the space. The list is materialised first so
    # that an empty tree logs nothing rather than reporting an empty tarball
    # as a successful evacuation, and so a failure of the listing itself is
    # not masked by tar happily writing an empty archive.
    # The list goes through a NUL-delimited temp file, never a variable: command
    # substitution drops NUL bytes, which would silently defeat -z on any path
    # containing whitespace. rc 3 means "nothing untracked to archive".
    local urc=0
    run_bounded 300 sh -c '
      cd /root/debate || exit 1
      t="$(mktemp)" || exit 1
      git ls-files --others --exclude-standard -z -- . ":(exclude)docent" > "$t" || { rm -f "$t"; exit 1; }
      if [ ! -s "$t" ]; then rm -f "$t"; exit 3; fi
      tar -czf "$1" -C /root/debate --null -T "$t" --no-recursion; rc=$?
      rm -f "$t"; exit "$rc"
    ' _ "$EVAC_DIR/untracked.tar.gz" 2>/dev/null || urc=$?
    if [ "$urc" -eq 0 ]; then
      log "evacuated untracked (non-ignored, excluding docent) working-tree files to $EVAC_DIR/untracked.tar.gz"
    elif [ "$urc" -ne 3 ]; then
      log "WARNING: could not archive the untracked files of /root/debate (rc $urc; best-effort; continuing)"
    fi
  fi

  if [ -d /root/debate/docent ] && [ -n "$(ls -A /root/debate/docent 2>/dev/null || true)" ]; then
    # The one artifact class worth waiting half an hour for: a whole run's
    # per-step transcripts, unreproducible, and a large directory landing on a
    # slow NFS volume. The 120s bound every other copy uses is a DETERMINISTIC
    # failure for this one — it would abort the teardown forever and leave a
    # partial copy behind in each abandoned EVAC_DIR. So: a 1800s bound, the
    # size logged first so the log explains the long pause, and the copy staged
    # at docent.partial and renamed to docent only on success, so an abandoned
    # partial stays identifiable and can never be mistaken for a complete
    # evacuation. Abort-on-failure policy is unchanged.
    DOCENT_SIZE="$(run_bounded 30 du -sh /root/debate/docent 2>/dev/null | awk '{print $1}' || true)"
    log "evacuating /root/debate/docent (per-step training transcripts, ${DOCENT_SIZE:-size unknown}); allowing up to 30 minutes"
    if run_bounded 1800 cp -a /root/debate/docent "$EVAC_DIR/docent.partial" 2>/dev/null \
       && run_bounded 60 mv "$EVAC_DIR/docent.partial" "$EVAC_DIR/docent" 2>/dev/null; then
      log "evacuated /root/debate/docent (per-step training transcripts)"
    else
      log "ERROR: FAILED to evacuate /root/debate/docent — the per-step training transcripts would be DESTROYED by the pod stop"
      # Drop this attempt's own partial copy. It is OUR incomplete artifact and
      # its source is intact at /root/debate/docent, so nothing is lost; leaving
      # one behind per retry cycle would slowly fill the very volume whose
      # sickness caused the failure. Nothing else in $EVAC_DIR is touched.
      if run_bounded 300 rm -rf "$EVAC_DIR/docent.partial" 2>/dev/null; then
        log "removed the incomplete $EVAC_DIR/docent.partial (source intact at /root/debate/docent)"
      else
        log "WARNING: could not remove the incomplete $EVAC_DIR/docent.partial; it is not a valid evacuation and will consume /workspace space until removed by hand"
      fi
      return 1
    fi
  fi
  return 0
}

# STOPPING MARKER — "a teardown is committed for this pod". Written the moment
# the watchdog commits (after the post-kill last look says idle, before the
# evacuation), removed on every path that abandons the teardown and returns to
# idle watch. It is NOT removed on the committed path: if the stop lands, the
# wipe of /root removes it for us. So a marker whose recorded pid is still a
# live watchdog means "this pod is being stopped, do not start work" —
# pod_run.sh refuses to launch past it.
# The pid is what makes it interpretable after a crash: a marker left by a dead
# watchdog is stale, exactly as with the armed marker.
# /root is container-local disk and cannot hang the way the NFS volume can, but
# both calls are bounded anyway and neither can trip set -e: a marker we failed
# to write only costs us the advisory, never the teardown.
STOPPING_MARKER=/root/pod_idle_stop.stopping
# 1 only when the marker on disk is known to record an accepted stop, so the
# abort messages can state that as a fact instead of asserting it blindly: the
# append can fail, and a message claiming a record that does not exist sends an
# operator looking for evidence that was never written.
ACCEPT_RECORDED=0
mark_stopping() {
  # Line 1 is always OUR pid, so it names the watchdog that owns the committed
  # teardown — which is what clear_stopping's ownership check reads.
  #
  # But it is NOT a blind truncation. A recommitted teardown INHERITS
  # responsibility for any still-unlanded stop an earlier cycle got accepted: an
  # `accepted` or `accepted-orphan` line already on disk records a stop the API
  # took and nothing pod-side can recall, and it must survive the recommit. So
  # when the existing marker carries either word we write `<our pid>\naccepted`
  # rather than the pid alone. Erasing it instead would leave a later
  # pre-acceptance abort — one that finds no accepted line and so falls through
  # to `rm -f` — deleting the only pod-side evidence of a stop still in flight.
  # With the line carried forward that same abort rewrites the marker to
  # `accepted-orphan`, which is the truth.
  # Fail direction if the bounded read itself fails (marker exists, head fails):
  # carry stays 0 and we truncate to the pid alone, destroying an accepted line
  # we could not see. Tolerated because it is the pre-change behavior and a root
  # read of a small file on /root's container-local disk essentially cannot fail
  # — the alternative (assume carry on read failure) would stamp `accepted` onto
  # markers that never had one, warning operators off launches forever.
  local carry=0
  if [ -e "$STOPPING_MARKER" ] \
     && run_bounded 15 sh -c 'head -n 20 "$1" 2>/dev/null | grep -qxE "accepted|accepted-orphan"' _ "$STOPPING_MARKER" 2>/dev/null; then
    carry=1
  fi
  if [ "$carry" -eq 1 ]; then
    if run_bounded 15 sh -c 'printf "%s\naccepted\n" "$1" > "$2"' _ "$$" "$STOPPING_MARKER" 2>/dev/null; then
      ACCEPT_RECORDED=1
      log "teardown committed (pid $$ recorded in $STOPPING_MARKER), CARRYING FORWARD an accepted stop from an earlier cycle: that stop is unrecallable and may still land, so the 'accepted' line stays on the marker under the new owner's pid"
    else
      ACCEPT_RECORDED=0
      log "WARNING: could not rewrite the stopping marker $STOPPING_MARKER while carrying forward an earlier cycle's accepted stop; the teardown proceeds, but the marker may no longer record that a stop is pending — verify at https://www.runpod.io/console/pods before starting any work"
    fi
    return 0
  fi
  ACCEPT_RECORDED=0
  if run_bounded 15 sh -c 'printf "%s\n" "$1" > "$2"' _ "$$" "$STOPPING_MARKER" 2>/dev/null; then
    log "teardown committed (pid $$ recorded in $STOPPING_MARKER); pod_run.sh will refuse to start work while this marker names a live watchdog"
  else
    log "WARNING: could not write the stopping marker $STOPPING_MARKER; the teardown proceeds, but a run started from now on will not be warned off"
  fi
}
# Appends the `accepted` line the moment the API accepts a stop (line 2 of a
# fresh marker; still an append, and so possibly a later line, on a marker that
# already carries one from an earlier cycle). If the watchdog dies
# in the post-accept sleep, the marker's pid is dead, but a stop the API accepted
# is still unrecallable and pending — this line is the durable pod-side record of
# that, and it is what lets a launcher tell "stale marker from a crashed
# watchdog" apart from "a stop is already on its way to this pod".
mark_accepted() {
  if run_bounded 15 sh -c 'printf "%s\n" accepted >> "$1"' _ "$STOPPING_MARKER" 2>/dev/null; then
    ACCEPT_RECORDED=1
    log "recorded the accepted stop in $STOPPING_MARKER (appended the line 'accepted'); the stop is unrecallable from here even if this watchdog dies"
  else
    ACCEPT_RECORDED=0
    log "WARNING: could not append the accepted line to $STOPPING_MARKER; a launch after this watchdog dies will under-warn (a stop is pending that nothing pod-side can recall)"
  fi
}
# Removes the marker ONLY when its first line is our own pid. Two watchdogs in
# interleaved teardowns could otherwise clear each other's committed marker while
# a stop is in flight, which is precisely the window the marker exists to cover.
#
# ONE EXCEPTION, and it is the important one: a marker carrying an `accepted`
# line records a stop the API already took, which NOTHING pod-side can recall.
# Every abort path calls this function, so a busy-abort in a cycle AFTER one that
# got a stop accepted would otherwise delete the only durable pod-side evidence
# that a stop is still pending — and the delay between acceptance and the halt is
# minutes to arbitrary. Such a marker is therefore not removed but REWRITTEN to
# the single line `accepted-orphan`: the owning pid is gone (this watchdog is
# back in idle watch and will not finish that teardown), yet the stop it accepted
# is still out there. pod_run.sh reads that first line as a hard refusal and
# tells the operator to verify on the console before launching anything.
# If a later cycle commits a fresh teardown, mark_stopping takes over line 1 with
# the new owner's pid but KEEPS the accepted line, because the earlier stop is
# still out there alongside the new commitment; see mark_stopping.
clear_stopping() {
  [ -e "$STOPPING_MARKER" ] || return 0
  local owner=""
  owner="$(run_bounded 15 head -n 1 "$STOPPING_MARKER" 2>/dev/null | tr -d '[:space:]' || true)"
  # `accepted` and `accepted-orphan` are OURS TO HANDLE, not a foreign owner: they
  # name a stop rather than a watchdog, so declining them left a pid-less marker
  # that no abort path could ever touch while a live watchdog kept re-refusing —
  # making pod_run.sh's "wait for the watchdog to abort" advice permanently false.
  # Same allowance the direct-rewrite abort path already spells out below.
  if [ -n "$owner" ] && [ "$owner" != "$$" ] && [ "$owner" != "accepted" ] && [ "$owner" != "accepted-orphan" ]; then
    log "NOT removing $STOPPING_MARKER: it names owner '$owner', not us ($$) — another watchdog's teardown may still be committed, and clearing it could green-light a launch into a stop that is already in flight"
    return 0
  fi
  # Already the exact end state the accepted-carrying branch below produces, so
  # leave the file alone rather than rewriting identical content over it.
  if [ "$owner" = "accepted-orphan" ]; then
    log "LEAVING $STOPPING_MARKER in place: it already records an orphaned accepted stop (single line 'accepted-orphan') for pod ${RUNPOD_POD_ID}. That stop CANNOT BE RECALLED; pod_run.sh will demand console verification before any launch. Verify at https://www.runpod.io/console/pods that the pod is NOT stopping, then remove $STOPPING_MARKER by hand."
    return 0
  fi
  # Bounded and line-capped: the marker is a two-line advisory, never a file
  # worth streaming, and a read that hangs must not wedge the watchdog.
  if run_bounded 15 sh -c 'head -n 20 "$1" 2>/dev/null | grep -qx accepted' _ "$STOPPING_MARKER" 2>/dev/null; then
    if run_bounded 15 sh -c 'printf "%s\n" accepted-orphan > "$1"' _ "$STOPPING_MARKER" 2>/dev/null; then
      log "ORPHANED ACCEPTED STOP: a stop for pod ${RUNPOD_POD_ID} was ACCEPTED by the API in an earlier teardown cycle and never landed, and it CANNOT BE RECALLED. This watchdog has abandoned that teardown and is back in idle watch, so $STOPPING_MARKER now records the orphaned accepted stop (single line 'accepted-orphan') instead of being removed. pod_run.sh will demand console verification before any launch. Verify at https://www.runpod.io/console/pods that the pod is NOT stopping, then remove $STOPPING_MARKER by hand."
    else
      log "WARNING: could not rewrite $STOPPING_MARKER to 'accepted-orphan'; a stop accepted in an earlier cycle is still pending and unrecallable, and the marker now under-describes it — verify at https://www.runpod.io/console/pods before starting any work"
    fi
    return 0
  fi
  if ! run_bounded 15 rm -f "$STOPPING_MARKER" 2>/dev/null; then
    log "WARNING: could not remove the stopping marker $STOPPING_MARKER; this watchdog is back in idle watch, but pod_run.sh will REFUSE every launch while the marker names our live pid — remove $STOPPING_MARKER by hand"
  fi
}

STARTED=$SECONDS
IDLE_TICKS=0
WAS_BUSY=-1        # -1 = unknown, so the first tick always logs the initial state
TICKS=0
log "watchdog started (pid $$, pod ${RUNPOD_POD_ID}): teardown after ${IDLE_MINUTES} idle minutes, grace ${GRACE_MINUTES} minutes"

while :; do
  # Sleep first so IDLE_TICKS counts minutes actually OBSERVED idle. Evaluating
  # at t=0 would let a 30-minute threshold fire after 29 minutes of watching.
  sleep 60

  if is_busy; then
    IDLE_TICKS=0
    if [ "$WAS_BUSY" -ne 1 ]; then
      log "busy: $BUSY_REASON"
      WAS_BUSY=1
    fi
  else
    IDLE_TICKS=$(( IDLE_TICKS + 1 ))
    if [ "$WAS_BUSY" -ne 0 ]; then
      log "idle: no trainer/eval process, no ssh session, gpu util <=5%, no terminal activity in 120s, container cpu <=30% of one core"
      WAS_BUSY=0
    fi
  fi

  TICKS=$(( TICKS + 1 ))
  if [ $(( TICKS % 10 )) -eq 0 ]; then
    log "heartbeat: idle for ${IDLE_TICKS}/${IDLE_MINUTES} minutes ($(( (SECONDS - STARTED) / 60 ))m since start)"
  fi

  # The grace window covers a watchdog that starts before the work does: a pod
  # booting into a cold pip install and an HF weight pull can look idle by every
  # probe here while a run is minutes from starting.
  if [ "$IDLE_TICKS" -ge "$IDLE_MINUTES" ] \
     && [ $(( SECONDS - STARTED )) -ge $(( GRACE_MINUTES * 60 )) ]; then
    log "TEARDOWN: idle for ${IDLE_TICKS} consecutive minutes (no infra./pod_run.sh/provision_pod.sh/debate-rl process, no ssh session, gpu util <=5%, no terminal activity in 120s, container cpu <=30% of one core)"
    log "killing judge vLLM / EngineCore processes"
    kill_judge
    sleep 5

    # Last look before an irreversible-ish action: an operator who sshed in or
    # kicked off a run during the kill window must win this race, not lose a pod
    # out from under them.
    if is_busy; then
      log "ABORTING teardown: became busy during the kill window ($BUSY_REASON); pod stays up, resuming idle watch"
      IDLE_TICKS=0
      WAS_BUSY=1
      continue
    fi

    # COMMITTED from here: the pod is idle and the judge is dead. Announce it on
    # disk before the evacuation, not after, because the evacuation itself can
    # take half an hour — a window in which an operator could otherwise ssh in
    # and start a multi-hour run that the stop would then halt mid-step.
    mark_stopping

    if ! evacuate; then
      clear_stopping
      log "ABORTING teardown: keeping the pod UP AND BILLING to preserve the training transcripts under /root/debate/docent, which a stop would wipe. Copy them off by hand, then stop the pod (or delete/empty the directory to let the watchdog proceed). Resuming idle watch."
      IDLE_TICKS=0
      WAS_BUSY=-1
      continue
    fi

    # From here the watchdog never gives up ON AN IDLE POD. It leaves this loop
    # only two ways: the pod actually halts, which kills this process with the
    # container, or the pod BECOMES BUSY again, in which case the teardown is
    # abandoned and the watchdog re-enters idle watch. Both ways a cycle can
    # fail to halt the pod — a stop accepted but not acted on, and three
    # attempts the API never accepted at all (a transient outage at teardown
    # time is exactly as plausible as either) — end in the same place: an
    # unstopped, billing H100 with no one watching it. Exiting on the second
    # would surrender the protection this script exists to provide, so a failed
    # cycle shouts and retries forever instead.
    # But retrying forever without ever looking again is its own hazard: an API
    # outage can hold this loop for hours while the pod stays up and healthy, an
    # operator sshes in and starts a multi-hour run, and the eventually-landing
    # stop halts it mid-step and wipes transcripts written after the one and
    # only evacuation. So busy is re-checked at the top of every cycle and again
    # after each post-accept wait: protecting a live run outranks finishing a
    # stop.
    # IRREDUCIBLE RESIDUAL: a stop the API already ACCEPTED before a re-check can
    # still land seconds after we abandon the cycle — nothing pod-side can recall
    # it. The stopping marker plus pod_run.sh's refusal to launch past it narrow
    # that window to the seconds between our re-check and the halt; they cannot
    # close it.
    STOP_CYCLE=0
    ABORTED_BUSY=0
    while :; do
      STOP_CYCLE=$(( STOP_CYCLE + 1 ))
      if is_busy; then
        log "ABORTING teardown mid-stop-cycle: became busy ($BUSY_REASON) before this cycle issued any stop. No stop was accepted in THIS cycle — but an EARLIER cycle may have had one accepted, and such a stop cannot be recalled; if so, clear_stopping preserves it as an 'accepted-orphan' marker rather than deleting it, and pod_run.sh's stopping-marker check will catch it. Anyone starting work should let pod_run.sh run that check."
        clear_stopping
        ABORTED_BUSY=1
        break
      fi
      ACCEPTED=0
      LADDER_BUSY=0
      for attempt in 1 2 3; do
        RC=0
        # Bounded: a wedged API would otherwise freeze the watchdog inside this
        # command substitution forever — no retry, no FATAL, pod bills all night.
        OUT="$(run_bounded 120 runpodctl stop pod "$RUNPOD_POD_ID" 2>&1)" || RC=$?
        if [ -n "$OUT" ]; then log "runpodctl: $(printf '%s' "$OUT" | tr '\n' ' ')"; fi
        if [ "$RC" -eq 0 ]; then
          log "pod ${RUNPOD_POD_ID} stop ACCEPTED (cycle $STOP_CYCLE, attempt $attempt); GPU billing ends when it halts. The pod was stopped, not removed: the /workspace network volume survives, but the container disk (/root, including /root/debate) is WIPED — see $EVAC_DIR."
          ACCEPTED=1
          mark_accepted
          break
        fi
        if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
          log "WARNING: 'runpodctl stop pod ${RUNPOD_POD_ID}' TIMED OUT after 120s (rc $RC, cycle $STOP_CYCLE, attempt $attempt/3); retrying in 30s"
        else
          log "WARNING: 'runpodctl stop pod ${RUNPOD_POD_ID}' failed (rc $RC, cycle $STOP_CYCLE, attempt $attempt/3); retrying in 30s"
        fi
        # The ladder can issue stops up to ~62s after the cycle-top is_busy said
        # idle, so re-check before each retry: an operator who sshes in or starts
        # a run inside the ladder wins the race exactly as they would at the top
        # of the cycle. Same abort path — nothing has been accepted in THIS
        # cycle on this branch. That is not the same as "no stop in flight":
        # from the second cycle onward an earlier cycle may have had one
        # accepted, and clear_stopping preserves that as an 'accepted-orphan'
        # marker rather than erasing it.
        if [ "$attempt" -lt 3 ]; then
          sleep 30
          if is_busy; then
            log "ABORTING teardown mid-stop-ladder: became busy ($BUSY_REASON) before attempt $(( attempt + 1 ))/3. No stop has been accepted in THIS cycle — but an EARLIER cycle may have had one accepted, which cannot be recalled; clear_stopping preserves that case as an 'accepted-orphan' marker, so let pod_run.sh's stopping-marker check run before starting work."
            clear_stopping
            ABORTED_BUSY=1
            LADDER_BUSY=1
            break
          fi
        fi
      done
      if [ "$LADDER_BUSY" -eq 1 ]; then break; fi
      if [ "$ACCEPTED" -ne 1 ]; then
        log "URGENT: all 3 attempts to stop pod ${RUNPOD_POD_ID} failed in cycle $STOP_CYCLE — the judge is dead but the pod is STILL RUNNING AND BILLING. Stop it by hand at https://www.runpod.io/console/pods. This watchdog is NOT giving up: retrying the whole cycle in 5 minutes, and every 5 minutes after that, until the pod halts."
        echo "URGENT: could not stop pod ${RUNPOD_POD_ID}; it is STILL RUNNING AND BILLING — stop it by hand at https://www.runpod.io/console/pods (watchdog retrying every 5 minutes)" >&2
        sleep 300
        continue
      fi
      # The stop is asynchronous. If it lands, the container dies and takes this
      # process with it; still being here after 5 minutes means it did not land,
      # so re-issue rather than exit and leave the pod unprotected.
      log "waiting for the pod to halt"
      sleep 300
      if is_busy; then
        # Two different truths, and saying the wrong one is how an operator ends
        # up trusting a record that was never written. clear_stopping's
        # accepted-orphan rewrite only triggers on an `accepted` line it can
        # READ, so when the append failed that path would instead delete the
        # marker: rewrite it here directly, and name the console as the only
        # source of truth if even that fails.
        if [ "$ACCEPT_RECORDED" -eq 1 ]; then
          log "ABORTING teardown mid-stop-cycle: became busy ($BUSY_REASON) while waiting for an ACCEPTED stop to land. That stop CANNOT BE RECALLED and may still halt this pod at any moment; it is recorded in $STOPPING_MARKER, which clear_stopping rewrites to 'accepted-orphan' rather than deleting, so pod_run.sh's stopping-marker check will refuse launches until a human verifies at https://www.runpod.io/console/pods."
          clear_stopping
        else
          log "ABORTING teardown mid-stop-cycle: became busy ($BUSY_REASON) while waiting for an ACCEPTED stop to land. That stop CANNOT BE RECALLED and may still halt this pod at any moment. It could NOT be recorded in $STOPPING_MARKER earlier, so https://www.runpod.io/console/pods is the ONLY source of truth about it; writing the orphan record now as a best effort."
          ORPHAN_OWNER=""
          if [ -e "$STOPPING_MARKER" ]; then
            ORPHAN_OWNER="$(run_bounded 15 head -n 1 "$STOPPING_MARKER" 2>/dev/null | tr -d '[:space:]' || true)"
          fi
          if [ -n "$ORPHAN_OWNER" ] && [ "$ORPHAN_OWNER" != "$$" ] && [ "$ORPHAN_OWNER" != "accepted" ] && [ "$ORPHAN_OWNER" != "accepted-orphan" ]; then
            log "NOT rewriting $STOPPING_MARKER: it names owner '$ORPHAN_OWNER', not us ($$) — another watchdog's teardown may still be committed. An accepted stop of OURS is nevertheless pending and unrecorded; verify at https://www.runpod.io/console/pods before starting any work"
          elif run_bounded 15 sh -c 'printf "%s\n" accepted-orphan > "$1"' _ "$STOPPING_MARKER" 2>/dev/null; then
            log "ORPHANED ACCEPTED STOP: wrote 'accepted-orphan' to $STOPPING_MARKER directly, so pod_run.sh will demand console verification before any launch. Verify at https://www.runpod.io/console/pods that the pod is NOT stopping, then remove $STOPPING_MARKER by hand."
          else
            log "WARNING: could not write 'accepted-orphan' to $STOPPING_MARKER either; there is NO pod-side record of the pending accepted stop for pod ${RUNPOD_POD_ID} — https://www.runpod.io/console/pods is the only source of truth, and a launch from here will not be warned off"
          fi
        fi
        ABORTED_BUSY=1
        break
      fi
      log "WARNING: pod ${RUNPOD_POD_ID} still alive ${STOP_CYCLE}x5 minutes after an accepted stop; re-issuing the stop (verify at https://www.runpod.io/console/pods)"
    done
    if [ "$ABORTED_BUSY" -eq 1 ]; then
      IDLE_TICKS=0
      WAS_BUSY=1
      continue
    fi
  fi
done
