#!/usr/bin/env bash
# All-on-pod training launcher for both run modes:
#   bash scripts/pod_run.sh debate <experiment> [runner args...]  # starts judge vLLM
#   bash scripts/pod_run.sh rlvr   <experiment> [runner args...]  # no judge
# Prereqs: provision_pod.sh has built $VOL/envs/verl-main; repo synced to /root/debate.
#
# Exit codes (a SEPARATE namespace from pod_up.sh's — same numbers, different
# meanings; read them against this script only):
#   1  a required argument is missing — bash's own exit status for the
#      ${var:?message} expansions below, not a code this script chooses
#   2  unknown mode, unresolvable judge config, or legacy checkpoint layout
#   3  judge vLLM failed to start, serve, or free its port
#   4  editable pip install failed or timed out
#   5  a trainer is already running, another pod_run.sh holds the launch lock,
#      or the pod is mid-teardown (a watchdog has committed to stopping it)
#   6  the idle watchdog could not be started (pod would bill unprotected)
# Outside this table: the final `exec` of the trainer can fail with the shell's
# own 126/127 (not executable / not found). The EXIT trap is already dropped by
# then, so in debate mode the judge server is left RUNNING — the idle watchdog
# is what reaps it.
set -euo pipefail

MODE="${1:?usage: pod_run.sh <debate|rlvr> <experiment> [args...]}"
EXP="${2:?usage: pod_run.sh <debate|rlvr> <experiment> [args...]}"

# Launch-phase mutual exclusion. The trainer guard below cannot separate two
# pod_run.sh invocations racing here: neither has exec'd its trainer yet, so
# both pass the guard, both sweep vLLM (the second sweep SIGKILLs the first's
# booting judge), and both launch watchdogs that then defer to each other and
# exit. The lock is released just before the exec, where the trainer guard
# takes over as the protection for the RUNNING phase.
exec 9>/root/pod_run.lock
flock -n 9 || {
  echo "FATAL: another pod_run.sh is mid-launch (lock held); wait for it to exec its trainer" >&2
  exit 5
}

# Pids of LIVE watchdogs, one per line, empty when there are none. A `--once`
# instance is excluded: it prints a busy/idle verdict, exits within seconds and
# protects nothing, so counting it would let a run launch unwatched with no
# warning at all. pod_idle_stop.sh's own twin-dedup excludes `--once` for the
# same reason; the two scripts must agree on what a watchdog is.
# The `bash <path>` form keeps a tmux'd editor or an in-flight grep of the
# watchdog's own filename from reading as a running watchdog. Interpreter FLAGS
# between `bash` and the path are tolerated (`bash -x .../pod_idle_stop.sh`, a
# routine way to debug the watchdog): without that, a flag-launched watchdog
# reads as "no watchdog at all" and this launch both starts a twin and misreads
# every marker it wrote as belonging to a dead process.
# Defined up here because the teardown check below is the first thing that needs
# it; the watchdog launch and the armed wait use it later.
live_watchdog_pids() {
  local p args
  for p in $(pgrep -f 'bash( -[^ ]+)* [^ ]*pod_idle_stop\.sh' 2>/dev/null || true); do
    args="$(ps -o args= -p "$p" 2>/dev/null || true)"
    case "$args" in *--once*) continue ;; esac
    printf '%s\n' "$p"
  done
}

# Teardown interlock. pod_idle_stop.sh writes its pid to
# /root/pod_idle_stop.stopping the moment it COMMITS to stopping the pod, and
# removes it if it abandons that teardown; once the API has ACCEPTED a stop it
# appends a second line, the word `accepted`, to the same file. Marker present +
# that pid still a live watchdog means a `runpodctl stop` may already be
# accepted and in flight: the API can be slow or retrying, so the pod looks
# perfectly healthy while it is minutes from halting. Without this check the
# launch just sees a live watchdog, reads it as protection, and execs the
# trainer — which then dies mid-step when the stop lands. Refuse instead; the
# operator can wait the watchdog out — it re-checks busy at the top of each stop
# cycle and after each post-accept wait, and those re-checks are separated by the
# 300s post-accept wait (longer when an attempt times out), so an active SSH
# session usually aborts the teardown within ~5 minutes; the marker is written
# BEFORE the artifact evacuation and no busy check runs during it, so an
# evacuation already in flight (up to ~30 min for a large docent directory)
# delays the abort that long.
# A marker whose pid is NOT a live watchdog is never silently removed: the
# watchdog may have died (or been killed) AFTER the API accepted its stop, in
# which case the stop is still in flight and this file is the only pod-side
# record of it. Removing it would unmask exactly the launch this interlock
# exists to refuse, so we refuse and hand the decision to the operator.
#
# Every marker state this block distinguishes, and why each is fatal:
#   live pid              — a teardown is committed and its watchdog is alive;
#                           it may still abort, so wait it out.
#   live pid + accepted   — same, except a stop is already in flight; killing
#                           the watchdog cannot recall it.
#   dead pid              — watchdog died before any stop was accepted (most
#                           likely); operator verifies the console and clears.
#   dead pid + accepted   — watchdog died AFTER a stop was accepted; the stop is
#                           still in flight and this file is its only record.
#   accepted-orphan       — the watchdog aborted a teardown it had already got
#                           accepted: no pid to wait on, the stop is unrecallable
#                           and may still land. Written by pod_idle_stop.sh's
#                           busy-abort, which rewrites (never deletes) an
#                           accepted-carrying marker.
#   unparseable/empty     — the writer cannot be identified. With a live watchdog
#                           this is a marker mid-write (mark_stopping truncates
#                           before it writes), so treat it as the live case and
#                           wait; with none, as the dead case.
#   bare `accepted` alone — mark_stopping's write failed and mark_accepted's
#                           append created the file, so the first line is the
#                           state word and no pid was ever recorded: accepted,
#                           writer unknown.
# No state auto-removes the marker; every one exits 5.
if [ -f /root/pod_idle_stop.stopping ]; then
  STOPPING_PID=""
  STOPPING_STATE=""
  {
    IFS= read -r STOPPING_PID || true
    IFS= read -r STOPPING_STATE || true
  } < /root/pod_idle_stop.stopping 2>/dev/null || true
  STOPPING_PID="$(printf '%s' "${STOPPING_PID:-}" | tr -d '[:space:]')"
  STOPPING_STATE="$(printf '%s' "${STOPPING_STATE:-}" | tr -d '[:space:]')"
  # Checked BEFORE the pid parse: the first line is a STATE WORD here, not a pid,
  # and parsing it as one would blank it and route this to the unparseable case.
  if [ "$STOPPING_PID" = accepted-orphan ]; then
    echo "FATAL: /root/pod_idle_stop.stopping says 'accepted-orphan': a previously accepted 'runpodctl stop' never landed and cannot be recalled; it may still land at any moment. Verify at https://www.runpod.io/console/pods that the pod is not stopping, then remove /root/pod_idle_stop.stopping yourself and re-run." >&2
    exit 5
  fi
  # A first line of literally `accepted` is the state word, not a pid: only
  # mark_accepted's append wrote to this file, so the stop IS accepted and the
  # writer's pid was never recorded. Parsing it as a pid would blank it and drop
  # the accepted fact, downgrading the diagnosis to "most likely no stop landed".
  STOPPING_ACCEPTED=0
  if [ "$STOPPING_PID" = accepted ]; then
    STOPPING_ACCEPTED=1
    STOPPING_PID=""
  elif [ "$STOPPING_STATE" = accepted ]; then
    STOPPING_ACCEPTED=1
  fi
  case "${STOPPING_PID:-x}" in *[!0-9]*) STOPPING_PID="" ;; esac
  if [ -n "$STOPPING_PID" ] && live_watchdog_pids | grep -qx "$STOPPING_PID"; then
    echo "FATAL: the idle watchdog (pid $STOPPING_PID) has committed to stopping this pod (/root/pod_idle_stop.stopping); an accepted 'runpodctl stop' may land at any moment and would halt the container mid-training-step. Wait for the watchdog to abort the teardown — an active SSH session (this one counts) makes it abort usually within ~5 minutes, but up to ~30 min if an artifact evacuation is already mid-flight, since no busy check runs during one. Killing the watchdog instead is NOT a shortcut: if its stop was already accepted, the kill leaves that stop in flight and unrecallable, so if you must kill it, first verify at https://www.runpod.io/console/pods that the pod is not stopping — the dead-marker check below will require exactly that verification before it lets any launch through." >&2
    exit 5
  fi
  # Marker present, writer unidentifiable, but a watchdog IS alive. Treat this as
  # the live case, not the dead one: mark_stopping truncates the marker before it
  # writes the pid, so a live watchdog that is mid-write (or whose write failed)
  # leaves exactly this file behind while proceeding to stop the pod. Routing it
  # to the dead branch would print instructions that greenlight a launch minutes
  # before that watchdog issues its first stop.
  if [ -z "$STOPPING_PID" ] && [ -n "$(live_watchdog_pids)" ]; then
    if [ "$STOPPING_ACCEPTED" = 1 ]; then
      echo "FATAL: /root/pod_idle_stop.stopping records a committed teardown whose writer cannot be identified (no pid in the marker), the marker says the stop was ACCEPTED by the API, and a live idle watchdog is running (pids: $(live_watchdog_pids | tr '\n' ' ')) — a stop may land at any moment and would halt the container mid-training-step. Wait for the watchdog to abort the teardown (an active SSH session, this one included, usually aborts it within ~5 minutes, up to ~30 min during an artifact evacuation). Do NOT kill the watchdog blindly: an accepted stop survives the kill and cannot be recalled. Do NOT remove the marker to get past this check." >&2
    else
      echo "FATAL: /root/pod_idle_stop.stopping records a committed teardown whose writer cannot be identified (no pid in the marker), and a live idle watchdog is running (pids: $(live_watchdog_pids | tr '\n' ' ')) — most likely that watchdog is mid-write and about to stop this pod, which would halt the container mid-training-step. Wait for it to abort the teardown (an active SSH session, this one included, usually aborts it within ~5 minutes, up to ~30 min during an artifact evacuation). Do NOT kill the watchdog blindly: if its stop was already accepted, the kill leaves that stop in flight and unrecallable. Do NOT remove the marker to get past this check." >&2
    fi
    exit 5
  fi
  if [ "$STOPPING_ACCEPTED" = 1 ]; then
    echo "FATAL: /root/pod_idle_stop.stopping records a committed teardown by pid '${STOPPING_PID:-unparseable}', which is no longer a live watchdog, and the marker says the stop was ACCEPTED by the API — it may land at any moment and would halt the container mid-training-step. Verify at https://www.runpod.io/console/pods that the pod is not stopping, then remove /root/pod_idle_stop.stopping yourself and re-run." >&2
  else
    echo "FATAL: /root/pod_idle_stop.stopping records a committed teardown by pid '${STOPPING_PID:-unparseable}', which is no longer a live watchdog, and the marker does not say 'accepted' — so the watchdog most likely died before any stop was accepted. Verify at https://www.runpod.io/console/pods that the pod is not stopping anyway, then remove /root/pod_idle_stop.stopping yourself and re-run." >&2
  fi
  exit 5
fi

PY="${PY:-/workspace/envs/verl-main/bin/python}"
# Resolved BEFORE the cd: an invocation by relative path would otherwise leave
# the sibling-script lookup below pointing at the wrong directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Toolkit preference mirrors provision_pod.sh, which means the SAME test it
# runs: a system nvcc reporting `release 13.` (anchored on the dot so a future
# "release 130" does not match), else the venv's pip-shipped nvidia/cu13
# fragments — which is what a fallback-provisioned pod actually BUILT against.
# Testing mere directory existence is not equivalent: an image shipping CUDA
# 12.x at /usr/local/cuda passes that test while provision fell back to cu13,
# leaving runtime-JIT'd kernels compiling against a toolkit the build never saw.
# If neither test holds we leave the default in place and let the first CUDA
# consumer fail loudly rather than silently pick a toolkit nothing was built for.
CUDA_HOME=/usr/local/cuda
# Globbed rather than hardcoding python3.12: a venv rebuilt on a newer
# interpreter would otherwise leave this path pointing at a directory that no
# longer exists, silently disabling the cu13 fallback with no diagnostic. An
# unmatched glob stays literal, fails the -d test, and leaves VENV_CU empty.
VENV_CU=""
for _cu in "$(dirname "$PY")"/../lib/python3.*/site-packages/nvidia/cu13; do
  if [ -d "$_cu" ]; then VENV_CU="$_cu"; break; fi
done
unset _cu
# nvcc can hang on a half-mounted toolkit dir, hence the bound (as in provision).
if ! timeout -k 5s 30 "$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep -q "release 13\."; then
  if [ -n "$VENV_CU" ] && [ -x "$VENV_CU/bin/nvcc" ]; then
    CUDA_HOME="$VENV_CU"
  fi
fi
export CUDA_HOME
# venv bin (ninja for JIT kernels) + cuda toolkit on PATH for all children
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
# provision_pod.sh caches weights under $VOL/hf; without the same export at run
# time every pull lands on the container disk and is re-downloaded after each
# pod stop/start (RunPod wipes /root, only /workspace survives).
export HF_HOME="${HF_HOME:-/workspace/hf}"
cd /root/debate

# Everything below assumes an otherwise-idle pod: the stale-vLLM sweep kills by
# pattern, and a trainer's COLOCATED verl rollout is itself a VLLM::EngineCore
# process. Re-running this script while a trainer is mid-flight would SIGKILL
# that trainer's rollout engine out from under it.
# The pattern requires the `python … -m <module>` form so that an editor, a
# tail, or an operator's own `grep infra.run_debate` cannot masquerade as a
# live trainer and abort the launch. It also matches the CONSOLE-SCRIPT form:
# pyproject ships `debate-rl = infra.run_debate:main` and
# `debate-train = infra.train:main`, and configs/math_pc_olmo.yaml documents
# `debate-rl` as the run command, so a trainer started that way has no
# `-m infra.` on its cmdline — it would slip past this guard while the watchdog's
# own busy patterns match it, and the unconditional sweep below would then
# SIGKILL the live run's engines.
# The DIRECT FILE PATH form (`$PY infra/run_debate.py`) matches neither of those
# and slipped through as well, so it is listed too. It is the one alternative
# with no command-form anchor — `vim infra/run_debate.py` reads as a live
# trainer — which costs a spurious abort the operator can see and explain,
# versus a SIGKILL of a real run's engines. The abort is the cheaper mistake.
# The module alternative tolerates INTERPRETER FLAGS between the interpreter and
# `-m`: `python -u -m infra.run_debate` (nohup + -u is a common operator habit)
# and `python3 -X faulthandler -m infra.run_rlvr` matched none of the three
# alternatives, so such a run read as "no trainer" and the sweep below SIGKILLed
# its own live engines. Flags taking a SEPARATE-WORD value are spelled out per
# flag (`-X`/`-W`) rather than allowing any bare word between flags: a blanket
# "anything may follow a flag" would also match `grep -r -m infra.run_debate`,
# putting an operator's own grep back in the position of aborting the launch.
TRAINER_PIDS="$(pgrep -f 'python[^ ]*( -[XW] [^ ]+| -[^ ]+)* -m infra\.(run_debate|run_rlvr)|infra/run_(debate|rlvr)\.py|bin/debate-(rl|train)' || true)"
if [ -n "$TRAINER_PIDS" ]; then
  echo "FATAL: a trainer is already running on this pod (pids: $(echo "$TRAINER_PIDS" | tr '\n' ' ')); refusing to kill vLLM processes or start another run — stop it first" >&2
  exit 5
fi

# CONFIG is overridable so a mode is not welded to one task family — the
# defaults are the MATH arms, but CodeContests (and anything added later) lives
# in its own file:
#   CONFIG=configs/codecontests_rlvr_olmo.yaml bash scripts/pod_run.sh rlvr <exp>
# The experiment name is still validated against whichever file is selected, so
# a mismatched pair fails at load with the available names listed.
case "$MODE" in
  debate) RUNNER=infra.run_debate; CONFIG="${CONFIG:-configs/math_pc_olmo.yaml}" ;;
  rlvr)   RUNNER=infra.run_rlvr;   CONFIG="${CONFIG:-configs/math_rlvr_olmo.yaml}" ;;
  *) echo "unknown mode $MODE (debate|rlvr)" >&2; exit 2 ;;
esac
[ -f "$CONFIG" ] || { echo "FATAL: config $CONFIG not found (cwd $(pwd))" >&2; exit 2; }

# Idle watchdog: once the pod has been idle for IDLE_MINUTES (default 30) — no
# trainer/eval processes, no active SSH, GPU idle — it kills leftover vLLM
# servers and STOPS the pod. An H100 left running after a finished run is the
# most expensive failure mode here. Set POD_IDLE_STOP=0 to keep the pod up.

# Asserts that SOME watchdog is running, or aborts the launch. Which process it
# is does not matter: setsid can fork (making our $! short-lived), and a
# watchdog that finds an older twin defers to it and exits 0 — both are fine,
# an unwatched pod is not. $1 says when we noticed, since the two call sites
# catch different deaths.
watchdog_alive_or_fatal() {
  local survivor
  survivor="$(live_watchdog_pids)"
  if [ -n "$survivor" ]; then
    echo "== idle watchdog running (pid $(echo "$survivor" | tr '\n' ' ')) =="
    return 0
  fi
  echo "FATAL: no idle watchdog is running ($1); last 20 lines of /root/pod_idle_stop.out:" >&2
  tail -20 /root/pod_idle_stop.out >&2 || true
  echo "Refusing to start an unprotected multi-hour run — an H100 left billing after it finishes is the most expensive failure here. Fix the watchdog (usually RUNPOD_POD_ID unset or runpodctl missing), or set POD_IDLE_STOP=0 to opt out deliberately." >&2
  exit 6
}

WATCHDOG_LAUNCHED=0
# Latched, never re-derived: records that a watchdog was ALIVE here, which is
# what makes the armed wait below this run's responsibility. See the comment
# on that gate for why the fact — not the current liveness — is the condition.
WATCHDOG_PRESENT=0
if [ "${POD_IDLE_STOP:-1}" != 0 ]; then
  if [ -n "$(live_watchdog_pids)" ]; then
    WATCHDOG_PRESENT=1
  else
    # This file is what watchdog_alive_or_fatal tails as death evidence; a
    # relaunch that truncated it would destroy the previous watchdog's own
    # fatal message — the same reason the judge log is rotated below.
    if [ -f /root/pod_idle_stop.out ]; then
      mv -f /root/pod_idle_stop.out /root/pod_idle_stop.out.prev
    fi
    # 9>&- so the watchdog does not inherit the launch lock: flock lives on the
    # OPEN FILE DESCRIPTION, so a detached child holding a copy of fd 9 keeps the
    # lock held long after we close ours — every later pod_run.sh would then exit 5
    # with the false diagnosis "another pod_run.sh is mid-launch", for the
    # watchdog's whole lifetime.
    setsid nohup bash "$SCRIPT_DIR/pod_idle_stop.sh" > /root/pod_idle_stop.out 2>&1 9>&- &
    WATCHDOG_PID=$!
    WATCHDOG_LAUNCHED=1
    # Fire-and-forget would hide the watchdog's own startup fatals (no
    # RUNPOD_POD_ID, no runpodctl) and leave a multi-hour run billing unprotected
    # — the exact failure the watchdog exists to prevent. So: verify it lives.
    sleep 3
    if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
      watchdog_alive_or_fatal "it exited within 3 seconds of launch"
    fi
  fi
fi

# Kills the judge's whole process group. A parent-only kill leaves the
# VLLM::EngineCore children alive holding their GPU allocation, so the next
# start dies in CUDA OOM against a server nobody can see (seen in production).
# No-op unless we started a judge ourselves.
kill_judge() {
  # `|| true` on both kills: this function is the EXIT trap, and a kill of an
  # already-dead process is the LAST command of its && list. Under errexit that
  # failure would abort the trap before `return 0`, and the trap's own non-zero
  # status then REWRITES the script's exit code — a documented `exit 3` would
  # surface to the operator as rc 1.
  [ -n "${JUDGE_PGID:-}" ] && kill -9 -"$JUDGE_PGID" 2>/dev/null || true
  [ -n "${JUDGE_PID:-}" ] && kill -9 "$JUDGE_PID" 2>/dev/null || true
  return 0
}

# The editable install has stalled indefinitely against an NFS-backed $VOL;
# a bounded failure is recoverable, a silent hang bills the pod. Runs BEFORE
# the judge: an install failure then aborts with nothing to clean up.
if ! timeout -k 30s 600 "$PY" -m pip install -q -e . --no-deps; then
  echo "FATAL: editable install of /root/debate failed or exceeded 600s (NFS stall writing to site-packages?), or $PY / timeout is missing (rc 127)" >&2
  exit 4
fi

# Stale vLLM sweep, BOTH modes. Previously this ran only in debate mode and
# only when a health probe failed, which left two holes: an rlvr run inherited
# a leftover judge (0.45 rollout + 0.18 judge + FSDP init peak = CUDA OOM with
# nothing pointing at the judge), and a HEALTHY judge masked orphaned trainer
# EngineCore processes still holding ~26GB. The trainer guard above is what
# makes an unconditional sweep safe.
# `ray::` catches the FSDP/rollout actors orphaned when a driver is SIGKILLed
# (the OOM killer); they hold GPU memory that the vLLM patterns cannot see, so
# a fast re-run would CUDA-OOM against invisible allocations. Safe for the same
# reason as the rest of the sweep: the trainer guard proved no live driver.
SWEEP_PATTERNS=('vllm.entrypoints.openai.api_server' 'VLLM::EngineCore' 'EngineCore_DP' 'ray::')
# Say what is about to die, so the operator learns what was left over and why
# rather than watching processes vanish silently. Quiet on a clean pod.
SWEEP_VICTIMS="$(for pat in "${SWEEP_PATTERNS[@]}"; do pgrep -af "$pat" 2>/dev/null || true; done | sort -u)"
if [ -n "$SWEEP_VICTIMS" ]; then
  echo "== sweep will kill: =="
  echo "$SWEEP_VICTIMS"
fi
for pat in "${SWEEP_PATTERNS[@]}"; do
  pkill -9 -f "$pat" 2>/dev/null || true
done
sleep 3

if [ "$MODE" = debate ]; then
  # Derive the judge's model and port FROM THE EXPERIMENT, never a second copy
  # here: a config-side judge change against a hardcoded launch means the
  # trainer 404s on every judge call, every datum drops, and the run burns
  # H100-hours producing no training signal. Loaded with the repo's own
  # resolver (inheritance/presets applied) from /root/debate, as the runner does.
  JUDGE_CFG="$("$PY" - "$CONFIG" "$EXP" <<'PYEOF'
import os
import sys
from urllib.parse import urlparse

from infra.config import load_experiment
from infra.run_debate import check_legacy_checkpoint_layout

exp = load_experiment(sys.argv[1], sys.argv[2])
ms = ((exp.get("agents") or {}).get("judge") or {}).get("model_settings") or {}
url, model = ms.get("base_url"), ms.get("model_file_path")
if not url or not model:
    raise SystemExit(
        f"experiment {sys.argv[2]!r} in {sys.argv[1]} has no judge "
        "model_settings.base_url + model_file_path; debate mode cannot serve it"
    )
port = urlparse(url).port
if not port:
    raise SystemExit(f"judge base_url {url!r} has no explicit port; cannot serve/probe it")

# Same pre-namespacing checkpoint guard the runner applies when it builds the
# backend, hoisted in front of the judge boot: the abort is decidable in
# seconds, and paying ~15 minutes of judge startup first to learn it is waste.
# Only the legacy (root-level) half belongs here — the namespaced-directory
# checks depend on runner CLI flags this script does not parse, so the runner
# keeps owning those.
tr = exp.get("training") or {}
# .get(key, default), matching the build in infra/run_debate.py. A falsy "or"
# would silently rewrite an explicitly empty checkpoint_dir into the default and
# guard a different path than the runner does.
ckpt_root = str((tr.get("verl") or {}).get("checkpoint_dir", "checkpoints/verl"))
check_legacy_checkpoint_layout(ckpt_root, os.path.join(ckpt_root, sys.argv[2]))

print(model)
print(port)
PYEOF
)" || {
    echo "FATAL: config or checkpoint-layout error for $CONFIG:$EXP (see error above)" >&2
    exit 2
  }
  { IFS= read -r JUDGE_MODEL; IFS= read -r JUDGE_PORT; } <<< "$JUDGE_CFG"
  echo "== judge from config: $JUDGE_MODEL on port $JUDGE_PORT =="

  # Health = a REAL 1-token completion, not /v1/models: an api_server whose
  # EngineCore was killed keeps answering metadata while every generate fails
  # (burned 8 zero-datum steps on 2026-08-03 — all debates died at the judge).
  # An error body and an empty choices[] both contain the string "choices", so
  # the payload is parsed and choices[0].text required rather than grepped.
  # Startup probe only: a pre-existing judge is never adopted (see sweep above).
  judge_ok() {
    curl -fsS -m 30 "http://127.0.0.1:$JUDGE_PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\": \"$JUDGE_MODEL\", \"prompt\": \"1+1=\", \"max_tokens\": 1}" \
      | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
c = d.get('choices') or []
sys.exit(0 if c and isinstance(c[0], dict) and c[0].get('text') is not None else 1)"
  }
  # Binds 0.0.0.0 like the server does: a 127.0.0.1-only probe passes while
  # something else holds the wildcard bind the server needs.
  port_free() {
    python3 -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', $JUDGE_PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()"
  }
  # A still-bound port is an unreaped survivor of the sweep: starting anyway
  # gives a server that dies on bind while judge_ok passes against the zombie.
  for _ in 1 2 3 4 5; do port_free && break; sleep 2; done
  if ! port_free; then
    echo "FATAL: port $JUDGE_PORT still bound after killing stale judge processes; find the holder (ps aux | grep -i vllm) and kill it, then re-run" >&2
    exit 3
  fi
  echo "== starting judge vLLM server =="
  # The usual reason we are here is "the judge died mid-run"; truncating its log
  # would destroy the only evidence of why. Keep the previous incarnation.
  if [ -f /root/judge_server.log ]; then
    mv -f /root/judge_server.log /root/judge_server.log.prev
  fi
  # gpu-memory-utilization is a server-launch parameter, not a config field:
  # 0.18 leaves room for the trainer's rollout engine + FSDP init peak on 80GB.
  # Own process group so cleanup can reap EngineCore children as a unit.
  setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port "$JUDGE_PORT" \
    --gpu-memory-utilization 0.18 --max-model-len 16384 \
    --max-num-seqs 32 \
    > /root/judge_server.log 2>&1 9>&- &   # 9>&-: do not inherit the launch lock
  JUDGE_PID=$!
  JUDGE_PGID="$(ps -o pgid= -p "$JUDGE_PID" 2>/dev/null | tr -d ' ' || true)"
  # Never group-kill our own group: if setsid did not detach, fall back to
  # the single pid rather than SIGKILLing this script and the trainer.
  if [ -z "$JUDGE_PGID" ] || [ "$JUDGE_PGID" = "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
    JUDGE_PGID=""
  fi
  trap kill_judge EXIT
  # Bounded, liveness-checked wait. A missing/gated weight pull or a CUDA-OOM
  # start kills the server outright, and an unbounded `until judge_ok` then
  # bills the pod indefinitely against a process that no longer exists.
  # 900s covers a cold HF pull of the 4B on a slow volume.
  JUDGE_DEADLINE=$(( SECONDS + 900 ))
  until judge_ok; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
      echo "FATAL: judge vLLM (pid $JUDGE_PID) exited before serving; last 40 lines of /root/judge_server.log:" >&2
      tail -40 /root/judge_server.log >&2 || true
      kill_judge
      exit 3
    fi
    if [ "$SECONDS" -ge "$JUDGE_DEADLINE" ]; then
      echo "FATAL: judge vLLM alive but not answering /v1/completions on 127.0.0.1:$JUDGE_PORT after 900s; check /root/judge_server.log (last 40 lines below) and nvidia-smi for a wedged engine" >&2
      tail -40 /root/judge_server.log >&2 || true
      kill_judge
      exit 3
    fi
    sleep 5
  done
  echo "== judge server up =="
fi

# Wait for the watchdog to say it is ARMED whenever idle-stop is in force. The
# +3s check above catches instant deaths (RUNPOD_POD_ID unset) but nothing
# slower: the watchdog's auth probe is bounded at 60s and can FATAL a full
# minute after launch. Timing-based second looks do not close that — on a re-run
# the pip install finishes in seconds, so a watchdog that dies at ~+60s is still
# alive at every earlier checkpoint and a multi-hour run proceeds unprotected.
# So wait for an AFFIRMATIVE signal instead: pod_idle_stop.sh writes its pid to
# /root/pod_idle_stop.armed only after all of its startup checks (env, dedupe,
# auth probe) have passed. This sits immediately before the exec so that in
# debate mode the ~minutes of judge boot have already overlapped the wait.
#
# The marker cannot be stale in any way that matters:
#   - a marker from a previous boot cannot exist: RunPod WIPES /root on stop;
#   - a marker left by a watchdog that has since died fails the live-pid test;
#   - a watchdog that was ALREADY running when we started (WATCHDOG_PRESENT=1)
#     has an armed marker that persists for the pod's uptime, so if it armed
#     long ago this wait is satisfied on the first iteration.
# That last case is why the wait is NOT restricted to watchdogs this run
# launched: a pre-existing watchdog can still be inside its ~90s startup (a
# manual launch, or one orphaned by a pod_run.sh killed mid-launch). It is live,
# so the launch block above skips starting our own, and if it then fatals at the
# auth probe the multi-hour run proceeds with no watchdog and no warning.
#
# The gate reads the LATCHED observation (WATCHDOG_PRESENT), deliberately NOT
# re-evaluating liveness here. Everything between that observation and this line
# — the pip install, the sweep, and in debate mode a judge boot — can span ~25
# minutes, and a pre-existing watchdog can die anywhere in it. Re-testing
# liveness would make that death SKIP the whole block, exec'ing a multi-hour run
# with no watchdog and no warning: verbatim the failure this block exists to
# close. Observed-alive-then-dead is exactly the case that must fail loudly, and
# the machinery inside handles it — the armed marker's pid fails the live test,
# live_watchdog_pids comes back empty, and watchdog_alive_or_fatal exits 6.
if [ "${POD_IDLE_STOP:-1}" != 0 ] \
  && { [ "$WATCHDOG_LAUNCHED" = 1 ] || [ "$WATCHDOG_PRESENT" = 1 ]; }; then
  ARMED_DEADLINE=$(( SECONDS + 90 ))
  while :; do
    ARMED_PID=""
    if [ -f /root/pod_idle_stop.armed ]; then
      ARMED_PID="$(tr -d '[:space:]' < /root/pod_idle_stop.armed 2>/dev/null || true)"
      case "${ARMED_PID:-x}" in *[!0-9]*) ARMED_PID="" ;; esac
    fi
    if [ -n "$ARMED_PID" ] && live_watchdog_pids | grep -qx "$ARMED_PID"; then
      echo "== idle watchdog armed (pid $ARMED_PID) =="
      break
    fi
    # No live watchdog AT ALL means it died during startup; there is nothing
    # left to wait for, so fail now instead of burning the remaining deadline.
    # A surviving instance that is not the one we launched (setsid forked, or a
    # twin took over) is accepted by watchdog_alive_or_fatal, which returns.
    if [ -z "$(live_watchdog_pids)" ]; then
      watchdog_alive_or_fatal "it died during startup, after surviving the first 3 seconds"
      break
    fi
    if [ "$SECONDS" -ge "$ARMED_DEADLINE" ]; then
      watchdog_alive_or_fatal "it never wrote /root/pod_idle_stop.armed within 90s of launch, so it never finished its startup checks"
      break
    fi
    sleep 1
  done
fi

# The trainer inherits the judge; exec replaces this shell, so drop the trap
# first — a fired EXIT here would kill the server the debate run depends on.
trap - EXIT
# Release the launch lock: an exec'd trainer would otherwise inherit fd 9 and
# hold it for the whole run, and the running phase is the trainer guard's job.
# From here a second pod_run.sh is almost always stopped by that guard, with a
# message that names the live trainer rather than a lock. Not quite always: a
# racing pod_run.sh can take the freed lock in the microsecond window between
# this close and the exec below, when neither guard sees a trainer. Practically
# unhittable, and it is the same both-passed-the-guard outcome the lock exists
# to make rare rather than impossible.
exec 9>&-
exec "$PY" -m "$RUNNER" --experiment-file "$CONFIG" --experiment "$EXP" "${@:3}"
