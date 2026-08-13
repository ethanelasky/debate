#!/usr/bin/env bash
# Canonical fresh-Pod launcher for this repository.
#
# This script never creates or deletes a Pod directly. Paid lifecycle mutations
# go through runpod-safe, which records ownership and installs a provider-side
# termination deadline. A launch is successful only after an SSH command runs
# on the Pod; status strings and pod-list runtime fields are not readiness.
#
# Usage:
#   bash scripts/pod_create.sh [name]
#   GPU_COUNT=4 TTL_MINUTES=720 bash scripts/pod_create.sh cc-capsweep
#   DRY_RUN=1 bash scripts/pod_create.sh inspect-only
#
# Important overrides:
#   GPU_TYPE        default: NVIDIA H200
#   GPU_COUNT       default: 1
#   DC_ID           exact datacenter; otherwise the best reported candidate
#   VOLUME_ID       optional network volume (forces that volume's datacenter)
#   TTL_MINUTES     default: 720; provider-enforced crash backstop
#   READY_DEADLINE  default: 600 seconds
#   ALLOW_UNTESTED  set to 1 for a GPU/template pair not proven below
set -euo pipefail

NAME="${1:-cc-$(date +%m%d-%H%M)}"
SAFE=/opt/homebrew/bin/runpod-safe
GPU_TYPE="${GPU_TYPE:-NVIDIA H200}"
GPU_COUNT="${GPU_COUNT:-1}"
TEMPLATE_ID="${TEMPLATE_ID:-a9dk3g7cny}"
IMAGE="${IMAGE:-runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster}"
DISK_GB="${DISK_GB:-200}"
TTL_MINUTES="${TTL_MINUTES:-720}"
READY_DEADLINE="${READY_DEADLINE:-600}"
VOLUME_ID="${VOLUME_ID:-}"
VOLUME_MOUNT="${VOLUME_MOUNT:-/workspace}"
PORTS="${PORTS:-22/tcp,22/udp,8888/http,8889/http}"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"

[ -x "$SAFE" ] || { echo "FATAL: $SAFE is missing or not executable" >&2; exit 2; }
[ -s "$KEY" ] || { echo "FATAL: SSH key $KEY is missing; run 'runpodctl doctor'" >&2; exit 2; }
case "$GPU_COUNT:$DISK_GB:$TTL_MINUTES:$READY_DEADLINE" in
  *[!0-9:]*|*::*|:*|*:) echo "FATAL: count, disk, TTL, and readiness values must be positive integers" >&2; exit 2 ;;
esac
[ "$GPU_COUNT" -gt 0 ] && [ "$DISK_GB" -gt 0 ] && [ "$TTL_MINUTES" -gt 0 ] && [ "$READY_DEADLINE" -gt 0 ] || {
  echo "FATAL: count, disk, TTL, and readiness values must be positive" >&2; exit 2; }

# Verify the pinned public template before spending. The create call below uses
# the template alone: passing both --template-id and --image makes some
# runpodctl versions silently take the custom-image path and discard the
# template, which is the failure this launcher exists to prevent.
TEMPLATE_JSON=$(runpodctl template get "$TEMPLATE_ID" -o json) || {
  echo "FATAL: cannot resolve template $TEMPLATE_ID" >&2; exit 2; }
printf '%s' "$TEMPLATE_JSON" | python3 -c '
import json, sys
template, image = json.load(sys.stdin), sys.argv[1]
actual = template.get("imageName")
if actual != image:
    print(
        f"template image mismatch: {actual!r} != {image!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
required = {"22/tcp"}
if not required.issubset(set(template.get("ports") or [])):
    print("template does not expose TCP SSH", file=sys.stderr)
    raise SystemExit(1)
' "$IMAGE" || { echo "FATAL: template contract failed" >&2; exit 2; }

# Keep the production default narrow. New pairs are cheap to test once, but a
# typo should not allocate an arbitrary expensive GPU fleet.
case "$GPU_TYPE|$TEMPLATE_ID" in
  "NVIDIA H200|a9dk3g7cny") ;;
  # Validated by the 32B AIME campaign and the CodeContests TP2 smokes; see
  # configs/topologies.yaml. The template contract above independently pins
  # the image and SSH exposure before this allowlist is consulted.
  "NVIDIA B200|a9dk3g7cny") ;;
  *)
    [ "${ALLOW_UNTESTED:-0}" = 1 ] || {
      echo "FATAL: unproven GPU/template pair: $GPU_TYPE / $TEMPLATE_ID" >&2
      echo "Set ALLOW_UNTESTED=1 for one bounded smoke test, then add the proven pair." >&2
      exit 2
    }
    ;;
esac

# A network volume cannot move between datacenters. Resolve that constraint
# before looking at stock so we never create an unusable Pod elsewhere.
if [ -n "$VOLUME_ID" ]; then
  VOLUME_DC=$(runpodctl network-volume list -o json | python3 -c '
import json, sys
wanted = sys.argv[1]
matches = [v for v in json.load(sys.stdin) if v.get("id") == wanted]
if len(matches) != 1 or not matches[0].get("dataCenterId"):
    raise SystemExit(1)
print(matches[0]["dataCenterId"])
' "$VOLUME_ID") || {
    echo "FATAL: network volume $VOLUME_ID was not found uniquely" >&2; exit 3; }
  if [ -n "${DC_ID:-}" ] && [ "$DC_ID" != "$VOLUME_DC" ]; then
    echo "FATAL: volume $VOLUME_ID is in $VOLUME_DC, not requested DC $DC_ID" >&2
    exit 3
  fi
  DC_ID="$VOLUME_DC"
  echo "volume $VOLUME_ID pins launch to $DC_ID"
fi

# Stock is advisory, but an empty stock marker is a reliable reason not to
# create. Rank High/Medium/Low and choose exactly one datacenter. We do not use
# repeated create calls as a capacity probe because each accepted call bills.
export GPU_TYPE
export DC_ID="${DC_ID:-}"
CANDIDATES=$(runpodctl datacenter list -o json | python3 -c '
import json, os, sys
gpu = os.environ["GPU_TYPE"]
wanted_dc = os.environ.get("DC_ID", "")
rank = {"High": 0, "Medium": 1, "Low": 2}
rows = []
for dc in json.load(sys.stdin):
    if wanted_dc and dc.get("id") != wanted_dc:
        continue
    for entry in dc.get("gpuAvailability") or []:
        if entry.get("gpuId") == gpu and entry.get("stockStatus") in rank:
            rows.append((rank[entry["stockStatus"]], dc["id"], entry["stockStatus"]))
for _, dc, stock in sorted(rows):
    print(f"{dc}:{stock}")
')
[ -n "$CANDIDATES" ] || {
  echo "FATAL: no reported stock for $GPU_TYPE${DC_ID:+ in $DC_ID}; nothing created" >&2
  exit 3
}
CHOSEN="${CANDIDATES%%$'\n'*}"
DC_ID="${CHOSEN%%:*}"
STOCK="${CHOSEN##*:}"
echo "selected $DC_ID ($STOCK stock) for $GPU_COUNT x $GPU_TYPE"
if [ "$GPU_COUNT" -gt 1 ]; then
  echo "NOTE: stock is per GPU type and does not guarantee $GPU_COUNT colocated GPUs." >&2
fi

VOL_ARGS=()
if [ -n "$VOLUME_ID" ]; then
  VOL_ARGS=(--network-volume-id "$VOLUME_ID" --volume-mount-path "$VOLUME_MOUNT")
fi
CREATE_ARGS=(
  --ttl-minutes "$TTL_MINUTES"
  --name "$NAME"
  --template-id "$TEMPLATE_ID"
  --compute-type GPU
  --gpu-id "$GPU_TYPE"
  --gpu-count "$GPU_COUNT"
  --cloud-type SECURE
  --data-center-ids "$DC_ID"
  --container-disk-in-gb "$DISK_GB"
  --ports "$PORTS"
)

# Capability preflight is free and catches an old runpod-safe installation
# before it can write a pending transaction or contact the create endpoint.
"$SAFE" create --dry-run "${CREATE_ARGS[@]}" ${VOL_ARGS[@]+"${VOL_ARGS[@]}"} >/dev/null || {
  echo "FATAL: runpod-safe rejected the template-aware launch contract" >&2
  exit 4
}
if [ "${DRY_RUN:-0}" = 1 ]; then
  "$SAFE" create --dry-run "${CREATE_ARGS[@]}" ${VOL_ARGS[@]+"${VOL_ARGS[@]}"}
  exit 0
fi

echo "creating through runpod-safe; server TTL is ${TTL_MINUTES} minutes"
CREATE_JSON=$("$SAFE" create "${CREATE_ARGS[@]}" ${VOL_ARGS[@]+"${VOL_ARGS[@]}"}) || {
  echo "FATAL: create did not produce owned state; inspect 'runpod-safe audit' and recover by exact name $NAME" >&2
  "$SAFE" audit >&2 || true
  exit 4
}
POD=$(printf '%s' "$CREATE_JSON" | python3 -c '
import json, re, sys
payload = json.load(sys.stdin)
found = set()
def walk(value):
    if isinstance(value, dict):
        for key in ("id", "podId", "pod_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9]+", candidate):
                found.add(candidate)
        for key in ("pod", "data", "result"):
            if key in value:
                walk(value[key])
    elif isinstance(value, list):
        for item in value:
            walk(item)
walk(payload)
if len(found) != 1:
    raise SystemExit(1)
print(found.pop())
') || {
  echo "FATAL: could not extract the exact owned Pod ID from create response" >&2
  "$SAFE" audit >&2 || true
  exit 4
}
echo "owned Pod ID: $POD"

READY=0
cleanup_failed_launch() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [ "$READY" -ne 1 ]; then
    echo "launch failed before SSH readiness; deleting owned Pod $POD" >&2
    "$SAFE" delete "$POD" >&2 || echo "URGENT: owned Pod deletion failed; it may still be billing" >&2
    "$SAFE" audit >&2 || true
  fi
  exit "$rc"
}
trap cleanup_failed_launch EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Support both runpodctl ssh-info JSON shapes seen in released CLIs. Never eval
# the provider-rendered ssh command; extract only the host and numeric port.
endpoint_from_info() {
  python3 -c '
import json, shlex, sys
try:
    data = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
ip, port = data.get("ip"), data.get("port")
if not ip or not port:
    nested = data.get("ssh") if isinstance(data.get("ssh"), dict) else {}
    ip, port = nested.get("ip"), nested.get("port")
if not ip or not port:
    command = data.get("sshCommand") or data.get("ssh_command")
    if isinstance(command, str):
        words = shlex.split(command)
        hosts = [w.split("@", 1)[-1] for w in words if "@" in w and not w.startswith("-")]
        ports = [words[i + 1] for i, w in enumerate(words[:-1]) if w == "-p"]
        if len(hosts) == 1 and len(ports) == 1:
            ip, port = hosts[0], ports[0]
if not isinstance(ip, str) or not str(port).isdigit():
    raise SystemExit(1)
print(ip, int(port))
'
}

DEADLINE=$(( $(date +%s) + READY_DEADLINE ))
LAST_INFO=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  LAST_INFO=$(runpodctl ssh info "$POD" 2>/dev/null) || LAST_INFO=""
  ENDPOINT=$(printf '%s' "$LAST_INFO" | endpoint_from_info 2>/dev/null) || ENDPOINT=""
  if [ -n "$ENDPOINT" ]; then
    IP="${ENDPOINT%% *}"
    PORT="${ENDPOINT##* }"
    if ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o ConnectTimeout=10 -o BatchMode=yes "root@$IP" -p "$PORT" true 2>/dev/null; then
      READY=1
      break
    fi
  fi
  printf '.'
  sleep 15
done
echo
[ "$READY" -eq 1 ] || {
  echo "FATAL: Pod $POD did not execute SSH within ${READY_DEADLINE}s" >&2
  echo "last ssh info: ${LAST_INFO:-<empty>}" >&2
  exit 5
}

trap - EXIT INT TERM HUP
EXPIRES=$(python3 -c '
import json, pathlib, sys
path = pathlib.Path.home() / ".local/state/runpod-safety/active" / (sys.argv[1] + ".json")
print(json.loads(path.read_text())["expires_at"])
' "$POD" 2>/dev/null || printf 'unknown')
COST=$(runpodctl pod get "$POD" -o json 2>/dev/null | python3 -c '
import json, sys
try:
    value = json.load(sys.stdin).get("costPerHr")
    print("unknown" if value is None else value)
except Exception:
    print("unknown")
' || printf 'unknown')

echo "READY"
echo "  pod_id=$POD"
echo "  hourly_cost=$COST"
echo "  server_expiry=$EXPIRES"
echo "  ssh=root@$IP port=$PORT"
echo "  sync: bash scripts/pod_sync.sh $IP $PORT --with-data"
