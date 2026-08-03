#!/usr/bin/env bash
# Local-side pod resume that refuses to start a pod whose machine is inside a
# maintenance window, polling until the window ends. RunPod only reports these
# windows on the authenticated GraphQL API (myself.pods.machine.maintenance*);
# the public status page / Better Stack feeds omit them entirely, and pods
# started into one bill while their network volume is unreachable.
# Only gates RESUME of an existing pod: a fresh create can't see the target
# machine's window beforehand (fields hang off pods you already own).
#   bash scripts/pod_up.sh [pod-id]
set -euo pipefail

POD="${1:-1pyjcotweljvff}"   # olmo-debate-us: H100 80GB, US-CA-2, repo on container disk
KEY=$(sed -n "s/^apikey *= *'\(.*\)'/\1/p" "$HOME/.runpod/config.toml")
[ -n "$KEY" ] || { echo "no apikey in ~/.runpod/config.toml"; exit 2; }

while :; do
  RESP=$(curl -s -m 20 "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { pods { id name machine { maintenanceStart maintenanceEnd maintenanceNote } } } }"}')
  # prints seconds until the window ends if now is inside it, else 0
  REMAIN=$(POD="$POD" python3 -c "
import json, os, sys
from datetime import datetime, timezone
pods = json.loads(sys.argv[1])['data']['myself']['pods']
pod = next((p for p in pods if p['id'] == os.environ['POD']), None)
if pod is None:
    sys.exit(f'pod {os.environ[\"POD\"]} not found in this account')
m = pod['machine'] or {}
start, end = m.get('maintenanceStart'), m.get('maintenanceEnd')
if not (start and end):
    print(0); sys.exit()
parse = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
now = datetime.now(timezone.utc)
if parse(start) <= now < parse(end):
    print(int((parse(end) - now).total_seconds()), flush=True)
    print(f'maintenance until {end}: {m.get(\"maintenanceNote\")}', file=sys.stderr)
else:
    print(0)
    if now < parse(start):
        print(f'WARNING: upcoming window {start} - {end}; starting anyway', file=sys.stderr)
" "$RESP")
  [ "$REMAIN" -eq 0 ] && break
  SLEEP=$(( REMAIN < 540 ? REMAIN + 60 : 600 ))
  echo "in maintenance window, ${REMAIN}s left; re-checking in ${SLEEP}s"
  sleep "$SLEEP"
done

runpodctl start pod "$POD"
runpodctl get pod "$POD" -a
