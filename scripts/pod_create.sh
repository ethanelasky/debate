#!/usr/bin/env bash
# Create a pod and prove it is USABLE, or delete it and exit nonzero.
#
#   bash scripts/pod_create.sh [name]
#   GPU_TYPE='NVIDIA H200' GPU_COUNT=4 bash scripts/pod_create.sh cc-capsweep
#
# The companion to pod_up.sh, which only ever RESUMES an existing pod. Creation
# has had no script until now, which is why it has been ad-hoc `runpodctl`
# invocations and why the 2026-08-07 failures were possible at all.
#
# Two things this exists to prevent, both of which cost real money:
#
#   1. `desiredStatus: RUNNING` is what we ASKED for, not what happened, and
#      billing follows the ask. Two H100 pods sat at RUNNING for 13 and 22
#      minutes with runtime:null and no container, billing $13.16/h the whole
#      time. The only honest readiness test is a command that executes ON the
#      pod, so that is what READY_DEADLINE gates on -- and the pod is DELETED
#      when the deadline passes rather than left to bill indefinitely.
#
#   2. RunPod places the pod for you unless told otherwise, and a region with no
#      stock returns "no longer any instances available" even when the GPU is
#      plentiful elsewhere. H200 was available in seven datacenters at the exact
#      moment four create attempts were failing. So: probe first, then create
#      into a NAMED datacenter.
#
# Exit codes (a separate namespace from pod_up.sh's and pod_run.sh's):
#   0  pod created and a remote command succeeded on it
#   2  runpodctl missing, or no API key configured
#   3  no datacenter has stock for the requested GPU (nothing created)
#   4  create call itself was refused (nothing created)
#   5  created but never became usable before READY_DEADLINE -- POD DELETED
#   6  refused: (gpuType, image) is not a known-good pair and FORCE_IMAGE unset
set -euo pipefail

NAME="${1:-cc-$(date +%m%d-%H%M)}"
GPU_TYPE="${GPU_TYPE:-NVIDIA H200}"
GPU_COUNT="${GPU_COUNT:-4}"
# Pinned because the flash-attn wheel in $VOL/wheels is ABI-tied to this exact
# python/torch/CUDA. Changing the image means rebuilding it from source (~45min).
IMAGE="${IMAGE:-runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster}"
DISK_GB="${DISK_GB:-200}"
COST_CEILING="${COST_CEILING:-25.0}"
READY_DEADLINE="${READY_DEADLINE:-420}"   # seconds; ~6min, vs the 22 we burned
VOLUME_ID="${VOLUME_ID-s3hz0k8153}"       # `-` not `:-`: VOLUME_ID= (empty) must mean
                                          # "no volume", not "use the default". CANNOT be added later.

# Known-good (gpu, image) pairs. The -cluster image runs on H200 and never
# started a container on H100 80GB HBM3 -- same arch family, same CUDA, still
# dead. Until a pair is confirmed it is a guess, and a guess costs a bring-up.
# Add a row only after a pod on that pair has actually run a step.
is_known_good() {
  case "$1|$2" in
    "NVIDIA H200|runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster") return 0 ;;
    *) return 1 ;;
  esac
}

command -v runpodctl >/dev/null || { echo "FATAL: runpodctl not on PATH" >&2; exit 2; }
[ -s "$HOME/.runpod/config.toml" ] || { echo "FATAL: ~/.runpod/config.toml missing (runpodctl doctor)" >&2; exit 2; }

if ! is_known_good "$GPU_TYPE" "$IMAGE"; then
  if [ -z "${FORCE_IMAGE:-}" ]; then
    echo "REFUSED: ($GPU_TYPE, $IMAGE) is not a known-good pair." >&2
    echo "  The -cluster image billed for 35min across two H100 pods without ever" >&2
    echo "  starting a container. Set FORCE_IMAGE=1 to try anyway -- and if it" >&2
    echo "  works, add the row to is_known_good() so the next person is not guessing." >&2
    exit 6
  fi
  echo "WARNING: ($GPU_TYPE, $IMAGE) unproven; FORCE_IMAGE set, continuing" >&2
fi

# ---------------------------------------------------------------- 1. probe
# Read-only. `create` is NOT a capacity query -- using it as one created two
# unwanted pods on 2026-08-07.
# A network volume is DATACENTER-LOCAL: a pod can only mount one in its own DC.
# So when a volume is requested its DC WINS over the probe -- there is no point
# finding better stock somewhere the volume cannot follow. That pinning is the
# real cost of using a volume, and it is worth knowing before stock runs out.
PINNED_DC=""
if [ -n "$VOLUME_ID" ]; then
  PINNED_DC=$(runpodctl network-volume list -o json 2>/dev/null | python3 -c "
import sys, json
vid = sys.argv[1]
for v in json.load(sys.stdin):
    if v.get('id') == vid:
        print(v.get('dataCenterId') or '')
" "$VOLUME_ID") || PINNED_DC=""
  [ -n "$PINNED_DC" ] || { echo "FATAL: volume $VOLUME_ID not found on this account" >&2; exit 3; }
  echo "volume $VOLUME_ID pins us to $PINNED_DC"
fi

echo "probing datacenters for $GPU_COUNT x $GPU_TYPE ..."
# ALL candidates, best stock first -- not just the best one. A datacenter can
# report stock and still refuse the create: "Low" is a per-GPU signal and says
# nothing about whether GPU_COUNT of them are free TOGETHER on one host. AP-JP-1
# reported Low for H200 and refused a 4-GPU create in the same breath. Trying a
# single DC and giving up turns a transient shortage into a hard failure.
DCS=$(runpodctl datacenter list -o json 2>/dev/null | python3 -c "
import sys, json
want, pinned = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '')
rank = {'High': 0, 'Medium': 1, 'Low': 2}
best = []
for dc in json.load(sys.stdin):
    if pinned and dc['id'] != pinned:
        continue
    for g in dc.get('gpuAvailability') or []:
        if g.get('gpuId') == want and g.get('stockStatus'):
            best.append((rank.get(g['stockStatus'], 9), dc['id'], g['stockStatus']))
best.sort()
for _, dc_id, stock in best:
    print(f'{dc_id}:{stock}')
" "$GPU_TYPE" "$PINNED_DC") || DCS=""

echo "  candidates: $(echo "$DCS" | tr '\n' ' ')"

# ---------------------------------------------------------------- 2. create
# macOS ships bash 3.2, where expanding an EMPTY array under `set -u` is an
# "unbound variable" error rather than nothing. The ${arr[@]+"${arr[@]}"} guard
# is what makes the no-volume path work at all.
VOL_ARG=(); [ -n "$VOLUME_ID" ] && VOL_ARG=(--networkVolumeId "$VOLUME_ID")
[ -z "$VOLUME_ID" ] && echo "NOTE: no VOLUME_ID -- \$VOL will be container disk, so provisioning
      (~80min: resolve, flash-attn, 121GB of weights) repeats on every pod and
      a stop wipes it. A volume cannot be attached after creation." >&2
[ -n "$VOLUME_ID" ] && echo "mounting volume $VOLUME_ID at /workspace"

POD=""; DC_ID=""
for cand in $DCS; do
  try_dc="${cand%%:*}"
  echo "  trying $try_dc (${cand##*:}) ..."
  OUT=$(runpodctl create pod --name "$NAME" --gpuType "$GPU_TYPE" --gpuCount "$GPU_COUNT" \
    --imageName "$IMAGE" --containerDiskSize "$DISK_GB" --dataCenterId "$try_dc" \
    --secureCloud --startSSH --ports '22/tcp' --cost "$COST_CEILING" ${VOL_ARG[@]+"${VOL_ARG[@]}"} 2>&1) || OUT="$OUT"
  POD=$(printf '%s' "$OUT" | sed -n 's/.*pod "\([a-z0-9]*\)" created.*/\1/p')
  if [ -n "$POD" ]; then DC_ID="$try_dc"; break; fi
  echo "    refused: $(printf '%s' "$OUT" | grep -i error | head -1)"
done
[ -n "$POD" ] || { echo "FATAL: every candidate datacenter refused $GPU_COUNT x $GPU_TYPE. Nothing created." >&2; exit 4; }
echo "created $POD in $DC_ID -- billing has STARTED"

# ---------------------------------------------------------------- 3. readiness
# The pod is not usable until a command RUNS on it. Anything short of that --
# desiredStatus, runtime being non-null, a port appearing -- has been observed
# to be true while the box was unreachable.
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
DEADLINE=$(( $(date +%s) + READY_DEADLINE ))
READY=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  EP=$(runpodctl pod list -o json 2>/dev/null | python3 -c "
import sys, json
pod = sys.argv[1]
for p in json.load(sys.stdin):
    if p.get('id') == pod:
        for x in ((p.get('runtime') or {}).get('ports') or []):
            if x.get('privatePort') == 22 and x.get('ip'):
                print(f\"{x['ip']} {x['publicPort']}\")
" "$POD" 2>/dev/null) || EP=""
  if [ -n "$EP" ]; then
    IP="${EP%% *}"; PORT="${EP##* }"
    if ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o ConnectTimeout=10 -o BatchMode=yes "root@$IP" -p "$PORT" true 2>/dev/null; then
      READY="$IP $PORT"; break
    fi
  fi
  printf '.'; sleep 15
done
echo

if [ -z "$READY" ]; then
  echo "FATAL: $POD never became usable within ${READY_DEADLINE}s. DELETING it." >&2
  # Deleting is the whole point. A pod stuck before container start bills at the
  # full rate and will do so forever; leaving it "in case it recovers" is how 22
  # minutes of nothing got paid for.
  runpodctl remove pod "$POD" >&2 || echo "WARNING: delete failed -- CHECK THE CONSOLE, IT IS BILLING" >&2
  echo "  Most likely ($GPU_TYPE, $IMAGE) does not start a container here." >&2
  echo "  Diagnose next time with the SSH PROXY, which answers even with no port:" >&2
  echo "    ssh -tt -i $KEY <podid>-<hash>@ssh.runpod.io true" >&2
  echo "  'container not found' there means the image never started; a timeout means no route." >&2
  exit 5
fi

echo "READY  $POD"
echo "  ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@${READY%% *} -p ${READY##* }"
echo "  next: provision with FLASH_ATTN_WHEEL=<wheel> to skip the ~45min source build"
