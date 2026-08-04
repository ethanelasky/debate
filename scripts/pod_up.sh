#!/usr/bin/env bash
# Local-side pod resume that refuses to start a pod whose machine is inside a
# maintenance window, polling until the window ends. RunPod only reports these
# windows on the authenticated GraphQL API (myself.pods.machine.maintenance*);
# the public status page / Better Stack feeds omit them entirely, and pods
# started into one bill while their network volume is unreachable.
# Only gates RESUME of an existing pod: a fresh create can't see the target
# machine's window beforehand (fields hang off pods you already own).
#   bash scripts/pod_up.sh [pod-id]
#
# Exit codes (automation reads these; 8 is the only one that means "money"):
#   0  pod start accepted
#   2  ~/.runpod/config.toml missing, or no apikey in it
#   4  still gated after 24h / 200 maintenance polls (pod NOT started)
#   5  pod id absent from this account, or 5 unusable queries in a row
#   6  refused: a maintenance window opens too soon (pod NOT started)
#   7  start failed, or start wedged and the pod is confirmed not running
#   8  start wedged and the pod IS or MAY BE running and BILLING — check now
set -euo pipefail

POD="${1:-1pyjcotweljvff}"   # olmo-debate-us: H100 80GB, US-CA-2, repo on container disk
# Under set -e a missing config file aborts on sed's own error, which buries the
# actionable diagnostic; check the file first.
CFG="$HOME/.runpod/config.toml"
[ -f "$CFG" ] || { echo "no $CFG — run 'runpodctl config --apiKey <key>' first" >&2; exit 2; }
KEY=$(sed -n "s/^apikey *= *'\(.*\)'/\1/p" "$CFG")
[ -n "$KEY" ] || { echo "no apikey in $CFG" >&2; exit 2; }

# Bearer header, never ?api_key=: a query-string secret leaks into argv, shell
# history and any proxy/CDN access log the request passes through.
API=https://api.runpod.io/graphql

# Wall-clock ceiling + iteration cap. Announced windows run hours, not days, so
# a poll still going after 24h (or after 200 rounds) means the reported end time
# is stuck or being pushed out — that is a different machine's problem, and
# silently waiting on it strands whatever automation is blocked on this start.
# No timeout(1) on Darwin, hence deadline arithmetic rather than a wrapper.
DEADLINE=$(( $(date +%s) + 86400 ))
MAX_POLLS=200
POLLS=0
FAILS=0
while :; do
  POLLS=$(( POLLS + 1 ))
  if [ "$POLLS" -gt "$MAX_POLLS" ]; then
    echo "FATAL: pod $POD still gated after $MAX_POLLS maintenance polls; the window end is not advancing — inspect the pod's machine at https://www.runpod.io/console/pods" >&2
    exit 4
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "FATAL: pod $POD still inside a maintenance window after 24h of polling; move the work to another pod/machine" >&2
    exit 4
  fi
  RESP=$(curl -s -m 20 "$API" \
    -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { pods { id name machine { maintenanceStart maintenanceEnd maintenanceNote } } } }"}') || RESP=""
  # prints seconds until the window ends if now is inside it, else 0;
  # exits 3 if a window opens too soon to be worth starting into,
  # exits 4 if the account demonstrably has no such pod
  RC=0
  REMAIN=$(POD="$POD" FORCE_START="${FORCE_START:-}" python3 -c "
import json, os, sys
from datetime import datetime, timezone
pods = json.loads(sys.argv[1])['data']['myself']['pods']
pod = next((p for p in pods if p['id'] == os.environ['POD']), None)
if pod is None:
    print('pod ' + os.environ['POD'] + ' not found in this account', file=sys.stderr)
    sys.exit(4)
m = pod['machine'] or {}
start, end = m.get('maintenanceStart'), m.get('maintenanceEnd')
note = m.get('maintenanceNote')
if not start:
    print(0); sys.exit()
parse = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
now = datetime.now(timezone.utc)
force = os.environ.get('FORCE_START', '').strip().lower() in ('1', 'true', 'yes')
# A window with a start but no end is announced-or-running with its end not yet
# published; the old 'needs both fields' test read that as no window at all and
# started straight into it. An unknown end is treated as an active window.
if not end:
    if parse(start) <= now:
        print(600, flush=True)
        print(f'maintenance since {start} with NO published end: {note}', file=sys.stderr)
        sys.exit()
    window = f'{start}, end not yet published'
    rerun = 'Re-run once the window has an end time published'
elif parse(start) <= now < parse(end):
    print(int((parse(end) - now).total_seconds()), flush=True)
    print(f'maintenance until {end}: {note}', file=sys.stderr)
    sys.exit()
else:
    window = f'{start} - {end}'
    rerun = f'Re-run after {end}'
print(0)
if now < parse(start):
    lead = int((parse(start) - now).total_seconds())
    if lead < 1800 and not force:
        print(f'FATAL: maintenance window opens in {lead}s ({window}); a pod '
              f'started now loses its network volume minutes into a billed run. '
              f'{rerun}, or FORCE_START=1 to override.', file=sys.stderr)
        sys.exit(3)
    print(f'WARNING: upcoming window {window}; starting anyway', file=sys.stderr)
" "$RESP") || RC=$?
  # Refusing to start into an imminent window is a decision, not a query
  # failure: exit rather than burn retries against it.
  if [ "$RC" -eq 3 ]; then
    echo "pod $POD NOT started" >&2
    exit 6
  fi
  # A parsed response that simply lacks the pod is a settled answer too —
  # retrying it four more times only delays the same verdict.
  if [ "$RC" -eq 4 ]; then
    echo "FATAL: pod $POD is not in this account; pod NOT started" >&2
    exit 5
  fi
  [ "$RC" -eq 0 ] || REMAIN=""
  # Empty/garbage means the curl timed out or the API returned an error
  # envelope — neither worth waiting out forever, but a single blip shouldn't
  # abort an otherwise-fine wait.
  case "$REMAIN" in
    ''|*[!0-9]*)
      FAILS=$(( FAILS + 1 ))
      if [ "$FAILS" -ge 5 ]; then
        echo "FATAL: 5 consecutive RunPod maintenance queries for pod $POD returned no usable answer (API down, apikey rejected, or the API answered in an unexpected shape — see the python errors above); pod NOT started" >&2
        exit 5
      fi
      echo "maintenance query failed (attempt $FAILS/5); retrying in 30s" >&2
      sleep 30
      continue ;;
  esac
  FAILS=0
  [ "$REMAIN" -eq 0 ] && break
  SLEEP=$(( REMAIN < 540 ? REMAIN + 60 : 600 ))
  echo "in maintenance window, ${REMAIN}s left; re-checking in ${SLEEP}s"
  sleep "$SLEEP"
done

# runpodctl blocks indefinitely against a stuck RunPod API. No timeout(1) on
# Darwin, so the bound comes from python; stdio is inherited, keeping the
# printed pod info byte-identical on success.
pod_cmd() {
  SECS="$1"; shift
  python3 -c "
import subprocess, sys
try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
except FileNotFoundError:
    sys.exit(127)
" "$SECS" "$@"
}

# Echoes the pod's desiredStatus; rc 1 if the API answer is unusable.
pod_status() {
  S=$(curl -s -m 20 "$API" \
    -H "Authorization: Bearer $KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { pods { id name desiredStatus } } }"}') || S=""
  POD="$POD" python3 -c "
import json, os, sys
try:
    pods = json.loads(sys.argv[1])['data']['myself']['pods']
    print(next(x for x in pods if x['id'] == os.environ['POD'])['desiredStatus'])
except Exception:
    sys.exit(1)
" "$S"
}

RC=0
pod_cmd 120 runpodctl start pod "$POD" || RC=$?
if [ "$RC" -ne 0 ]; then
  if [ "$RC" -eq 124 ]; then
    # The expensive ambiguity: RunPod may have accepted the start before the
    # call wedged, in which case the pod is billing with nobody watching. A
    # single status read is stale within seconds — the queued start can land
    # after it — so poll across ~40s before claiming nothing is billing.
    echo "FATAL: 'runpodctl start pod $POD' did not return within 120s; the start may already have been ACCEPTED" >&2
    ANSWERED=0
    N=1
    while [ "$N" -le 3 ]; do
      if [ "$N" -gt 1 ]; then sleep 20; fi
      SRC=0
      DESIRED=$(pod_status) || SRC=$?
      if [ "$SRC" -ne 0 ]; then
        echo "  poll $N/3: status query failed" >&2
      else
        ANSWERED=$(( ANSWERED + 1 ))
        echo "  poll $N/3: desiredStatus=$DESIRED" >&2
        if [ "$DESIRED" = "RUNNING" ]; then
          echo "  the pod IS BILLING; stop it explicitly if this start was not wanted" >&2
          exit 8
        fi
      fi
      N=$(( N + 1 ))
    done
    if [ "$ANSWERED" -eq 0 ]; then
      echo "  status queries ALSO failed — assume the pod may be RUNNING and BILLING; check https://www.runpod.io/console/pods NOW" >&2
      exit 8
    fi
    echo "  never observed RUNNING ($ANSWERED of 3 status polls answered, over ~40s), so nothing appears to be billing — check https://www.runpod.io/console/pods again in a minute anyway" >&2
    exit 7
  fi
  echo "FATAL: 'runpodctl start pod $POD' failed (rc $RC)" >&2
  exit 7
fi
# A clean rc from 'start' only means the request was accepted. One bounded read
# so a start that was accepted and then dropped doesn't read as a success.
SRC=0
DESIRED=$(pod_status) || SRC=$?
if [ "$SRC" -ne 0 ]; then
  echo "WARNING: could not read pod $POD status after start; check https://www.runpod.io/console/pods" >&2
elif [ "$DESIRED" != "RUNNING" ]; then
  echo "WARNING: start accepted but pod $POD desiredStatus=$DESIRED, not RUNNING; check https://www.runpod.io/console/pods" >&2
fi
echo "note: no idle watchdog is armed until pod_run.sh runs on the pod — a pod left idle without it bills until stopped by hand." >&2
# Informational only: the pod is already started, so a slow read is not worth
# failing the run over.
pod_cmd 60 runpodctl get pod "$POD" -a || {
  echo "WARNING: 'runpodctl get pod $POD -a' did not return within 60s; pod $POD WAS started — check https://www.runpod.io/console/pods" >&2; }
