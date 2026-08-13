#!/usr/bin/env bash
# Reproducibly deploy the authenticated executor into one dedicated Lima VM.
set -euo pipefail

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
readonly DEFAULT_VM="codecontests-judge-v2"
readonly DEFAULT_HOST_PORT="18081"
readonly DEFAULT_ARTIFACT_ROOT="${REPO_ROOT}/../ai-debate-run-artifacts/codecontests-judge-v2"

usage() {
  cat >&2 <<EOF
usage: $0 --run-id ID [options]

options:
  --vm NAME                 Lima instance (default: ${DEFAULT_VM})
  --host-port PORT          loopback-forwarded host port (default: ${DEFAULT_HOST_PORT})
  --artifact-root DIR       existing host artifact root
  --verifier FILE           verifier whose exact digest is frozen
EOF
  exit 2
}

run_id=""
vm_name=$DEFAULT_VM
host_port=$DEFAULT_HOST_PORT
artifact_root=$DEFAULT_ARTIFACT_ROOT
verifier_path="${REPO_ROOT}/infra/envs/tasks/codecontests.py"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ "$#" -ge 2 ]] || usage
      run_id=$2
      shift 2
      ;;
    --vm)
      [[ "$#" -ge 2 ]] || usage
      vm_name=$2
      shift 2
      ;;
    --host-port)
      [[ "$#" -ge 2 ]] || usage
      host_port=$2
      shift 2
      ;;
    --artifact-root)
      [[ "$#" -ge 2 ]] || usage
      artifact_root=$2
      shift 2
      ;;
    --verifier)
      [[ "$#" -ge 2 ]] || usage
      verifier_path=$2
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
  echo "--run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}" >&2
  exit 1
}
[[ "$vm_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
  echo "unsafe Lima instance name" >&2
  exit 1
}
[[ "$host_port" =~ ^[0-9]+$ && "$host_port" -ge 1024 && "$host_port" -le 65535 ]] || {
  echo "host port must be an integer in [1024, 65535]" >&2
  exit 1
}
for command_name in limactl openssl python3 shasum tar; do
  command -v "$command_name" >/dev/null || {
    echo "missing required command: $command_name" >&2
    exit 1
  }
done
[[ -d "$artifact_root" && ! -L "$artifact_root" ]] || {
  echo "artifact root must be an existing non-symlink directory" >&2
  exit 1
}
artifact_root=$(cd -- "$artifact_root" && pwd -P)
[[ -f "$verifier_path" && ! -L "$verifier_path" ]] || {
  echo "verifier must be a non-symlink regular file" >&2
  exit 1
}
verifier_path=$(cd -- "$(dirname -- "$verifier_path")" && printf '%s/%s\n' "$PWD" "$(basename -- "$verifier_path")")

readonly DEPLOYMENT_NONCE=$(openssl rand -hex 16)
readonly BUNDLE_DIR="${artifact_root}/${run_id}"
readonly SERVICE_NAME="codecontests-executor-${run_id}.service"
readonly PROBE_SERVICE_NAME="codecontests-executor-hostile-probe-${run_id}.service"
readonly HOST_URL="http://127.0.0.1:${host_port}"
host_stage=""
guest_stage=""
bundle_created=0
guest_install_attempted=0
completed=0

cleanup_host_stage() {
  if [[ -n "$host_stage" && "$host_stage" == /tmp/codecontests-executor-host.* && -d "$host_stage" ]]; then
    chmod -R u+w "$host_stage" 2>/dev/null || true
    rm -rf -- "$host_stage"
  fi
}

cleanup_guest_stage() {
  if [[ -n "$guest_stage" && "$guest_stage" == /tmp/codecontests-executor-deploy.* ]]; then
    limactl shell "$vm_name" -- /usr/bin/bash -s -- "$guest_stage" <<'EOF' >/dev/null 2>&1 || true
set -euo pipefail
target=$1
[[ "$target" == /tmp/codecontests-executor-deploy.* && -d "$target" && ! -L "$target" ]]
chmod -R u+w "$target" 2>/dev/null || true
rm -rf -- "$target"
EOF
  fi
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ "$completed" -ne 1 && "$guest_install_attempted" -eq 1 && -n "$guest_stage" ]]; then
    if [[ -d "$BUNDLE_DIR" && ! -L "$BUNDLE_DIR" ]]; then
      limactl shell "$vm_name" -- sudo /usr/bin/journalctl \
        --unit "$SERVICE_NAME" --no-pager --output=short-iso \
        >"$BUNDLE_DIR/guest-failure-journal.txt" 2>&1 || true
      limactl shell "$vm_name" -- sudo /usr/bin/journalctl \
        --unit "$PROBE_SERVICE_NAME" --no-pager --output=short-iso \
        >"$BUNDLE_DIR/guest-probe-failure-journal.txt" 2>&1 || true
      limactl shell "$vm_name" -- /usr/bin/bash -s -- "$run_id" "$SERVICE_NAME" <<'EOF' \
        >"$BUNDLE_DIR/guest-failure-state.txt" 2>&1 || true
set -euo pipefail
run_id=$1
service_name=$2
sudo systemctl status --no-pager "$service_name" || true
sudo findmnt -n -M /var/lib/codecontests-executor/rootfs || true
sudo ss -ltnH 'sport = :8080' || true
manifest="/var/lib/codecontests-executor/rootfs-v2.${run_id}.manifest.json"
if sudo test -f "$manifest"; then
  sudo sha256sum "$manifest"
fi
EOF
      chmod 0600 \
        "$BUNDLE_DIR/guest-failure-journal.txt" \
        "$BUNDLE_DIR/guest-probe-failure-journal.txt" \
        "$BUNDLE_DIR/guest-failure-state.txt"
    fi
    if ! limactl shell "$vm_name" -- sudo /usr/bin/bash \
      "$guest_stage/scripts/deploy_codecontests_executor_lima_guest.sh" \
      rollback \
      --run-id "$run_id" \
      --deployment-nonce "$DEPLOYMENT_NONCE"; then
      echo "warning: exact owned-deployment rollback did not complete" >&2
    fi
  fi
  cleanup_guest_stage
  cleanup_host_stage
  if [[ "$completed" -ne 1 && "$bundle_created" -eq 1 && -d "$BUNDLE_DIR" ]]; then
    local failed_bundle="${BUNDLE_DIR}.failed"
    if [[ ! -e "$failed_bundle" && ! -L "$failed_bundle" ]]; then
      mv -- "$BUNDLE_DIR" "$failed_bundle"
      echo "incomplete client bundle preserved at $failed_bundle" >&2
    else
      echo "incomplete client bundle preserved at $BUNDLE_DIR" >&2
    fi
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

[[ ! -e "$BUNDLE_DIR" && ! -L "$BUNDLE_DIR" ]] || {
  echo "refusing to overwrite client bundle: $BUNDLE_DIR" >&2
  exit 1
}
umask 077
mkdir "$BUNDLE_DIR"
chmod 0700 "$BUNDLE_DIR"
bundle_created=1
host_stage=$(mktemp -d "/tmp/codecontests-executor-host.${run_id}.XXXXXX")

readonly -a SOURCE_INPUTS=(
  "codecontests_executor/__init__.py"
  "codecontests_executor/cgroup_gate.py"
  "codecontests_executor/client.py"
  "codecontests_executor/protocol.py"
  "codecontests_executor/sandbox_launcher.py"
  "codecontests_executor/service.py"
  "codecontests_executor/supervisor.py"
  "scripts/build_codecontests_rootfs_artifact.sh"
  "scripts/build_codecontests_rootfs_manifest.py"
  "scripts/measure_codecontests_client_provenance.py"
  "scripts/measure_codecontests_executor_bundle.py"
  "scripts/probe_codecontests_executor_live.py"
  "scripts/capture_codecontests_executor_identity.py"
  "scripts/deploy_codecontests_executor_lima.sh"
  "scripts/deploy_codecontests_executor_lima_guest.sh"
  "deploy/codecontests-executor/verify_staged_executor.py"
  "deploy/codecontests-executor/codecontests-executor.lima.service.in"
  "deploy/codecontests-executor/codecontests-executor-hostile-probe.lima.service.in"
  "deploy/codecontests-executor/codecontests-rootfs.lima.mount.in"
)

snapshot_sources() {
  local output=$1
  local relative
  : >"$output"
  for relative in "${SOURCE_INPUTS[@]}"; do
    [[ -f "${REPO_ROOT}/${relative}" && ! -L "${REPO_ROOT}/${relative}" ]] || {
      echo "unsafe/missing deployment source: $relative" >&2
      return 1
    }
    printf '%s  %s\n' "$(shasum -a 256 "${REPO_ROOT}/${relative}" | awk '{print $1}')" "$relative" >>"$output"
  done
  printf '%s  %s\n' "$(shasum -a 256 "$verifier_path" | awk '{print $1}')" "verifier:${verifier_path}" >>"$output"
}

snapshot_sources "$BUNDLE_DIR/source-inputs.sha256"
openssl rand -hex 32 >"$BUNDLE_DIR/bearer"
openssl rand -hex 32 >"$BUNDLE_DIR/hmac-key"
chmod 0600 "$BUNDLE_DIR/bearer" "$BUNDLE_DIR/hmac-key"
python3 -B "$REPO_ROOT/scripts/measure_codecontests_client_provenance.py" \
  --client "$REPO_ROOT/codecontests_executor/client.py" \
  --protocol "$REPO_ROOT/codecontests_executor/protocol.py" \
  --verifier "$verifier_path" \
  >"$BUNDLE_DIR/client-provenance.json"
chmod 0600 "$BUNDLE_DIR/client-provenance.json" "$BUNDLE_DIR/source-inputs.sha256"

mkdir -p \
  "$host_stage/codecontests_executor" \
  "$host_stage/scripts" \
  "$host_stage/deploy/codecontests-executor"
for source_name in __init__.py cgroup_gate.py client.py protocol.py sandbox_launcher.py service.py supervisor.py; do
  cp -p "$REPO_ROOT/codecontests_executor/$source_name" "$host_stage/codecontests_executor/$source_name"
done
for source_name in build_codecontests_rootfs_artifact.sh build_codecontests_rootfs_manifest.py measure_codecontests_executor_bundle.py probe_codecontests_executor_live.py deploy_codecontests_executor_lima_guest.sh; do
  cp -p "$REPO_ROOT/scripts/$source_name" "$host_stage/scripts/$source_name"
done
for source_name in verify_staged_executor.py codecontests-executor.lima.service.in codecontests-executor-hostile-probe.lima.service.in codecontests-rootfs.lima.mount.in; do
  cp -p "$REPO_ROOT/deploy/codecontests-executor/$source_name" "$host_stage/deploy/codecontests-executor/$source_name"
done
cp -p "$BUNDLE_DIR/bearer" "$BUNDLE_DIR/hmac-key" "$BUNDLE_DIR/client-provenance.json" "$host_stage/"

bash -n "$REPO_ROOT/scripts/deploy_codecontests_executor_lima.sh"
bash -n "$REPO_ROOT/scripts/deploy_codecontests_executor_lima_guest.sh"
CAPTURE_SCRIPT="$REPO_ROOT/scripts/capture_codecontests_executor_identity.py" \
python3 -B - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["CAPTURE_SCRIPT"])
compile(path.read_bytes(), str(path), "exec")
PY
limactl shell "$vm_name" -- /usr/bin/true
guest_stage=$(
  limactl shell "$vm_name" -- /usr/bin/bash -c \
    'umask 077; mktemp -d /tmp/codecontests-executor-deploy.XXXXXXXX'
)
[[ "$guest_stage" == /tmp/codecontests-executor-deploy.* ]] || {
  echo "Lima returned an unsafe guest staging path" >&2
  exit 1
}
tar -C "$host_stage" -cf - . | \
  limactl shell "$vm_name" -- /usr/bin/tar -C "$guest_stage" -xf -

guest_install_attempted=1
limactl shell "$vm_name" -- sudo /usr/bin/bash \
  "$guest_stage/scripts/deploy_codecontests_executor_lima_guest.sh" \
  install \
  --run-id "$run_id" \
  --deployment-nonce "$DEPLOYMENT_NONCE" \
  --stage-dir "$guest_stage"

limactl shell "$vm_name" -- sudo /usr/bin/bash \
  "$guest_stage/scripts/deploy_codecontests_executor_lima_guest.sh" \
  hostile-probe \
  --run-id "$run_id" \
  --deployment-nonce "$DEPLOYMENT_NONCE" \
  --stage-dir "$guest_stage"
limactl shell "$vm_name" -- sudo /usr/bin/tar \
  -C "/var/lib/codecontests-executor/deployments/${run_id}/hostile-probe" \
  -cf - hostile-probe.json | tar -C "$BUNDLE_DIR" -xf -
chmod 0600 "$BUNDLE_DIR/hostile-probe.json"

python3 -B "$REPO_ROOT/scripts/capture_codecontests_executor_identity.py" \
  --url "$HOST_URL" \
  --bearer-file "$BUNDLE_DIR/bearer" \
  --hmac-key-file "$BUNDLE_DIR/hmac-key" \
  --identity-output "$BUNDLE_DIR/identity.json" \
  --envelope-output "$BUNDLE_DIR/identity.envelope.json" \
  >"$BUNDLE_DIR/identity-capture-summary.json"
chmod 0600 \
  "$BUNDLE_DIR/identity.json" \
  "$BUNDLE_DIR/identity.envelope.json" \
  "$BUNDLE_DIR/identity-capture-summary.json"

tar -C "$BUNDLE_DIR" -cf - identity.json identity.envelope.json | \
  limactl shell "$vm_name" -- /usr/bin/tar -C "$guest_stage" -xf -
limactl shell "$vm_name" -- sudo /usr/bin/bash \
  "$guest_stage/scripts/deploy_codecontests_executor_lima_guest.sh" \
  freeze-identity \
  --run-id "$run_id" \
  --deployment-nonce "$DEPLOYMENT_NONCE" \
  --stage-dir "$guest_stage"

snapshot_sources "$host_stage/source-inputs.after.sha256"
cmp -s "$BUNDLE_DIR/source-inputs.sha256" "$host_stage/source-inputs.after.sha256" || {
  echo "deployment inputs changed while identity was being frozen" >&2
  exit 1
}

CC_DEPLOY_BUNDLE="$BUNDLE_DIR" CC_DEPLOY_RUN_ID="$run_id" \
CC_DEPLOY_VM="$vm_name" CC_DEPLOY_SERVICE="$SERVICE_NAME" \
CC_DEPLOY_HOST_URL="$HOST_URL" \
python3 -B - <<'PY'
import json
import os
from pathlib import Path

bundle = Path(os.environ["CC_DEPLOY_BUNDLE"])
identity = json.loads((bundle / "identity.json").read_text(encoding="utf-8"))
deployment = {
    "format": "palaestra.codecontests.lima-client-bundle.v1",
    "run_id": os.environ["CC_DEPLOY_RUN_ID"],
    "vm": os.environ["CC_DEPLOY_VM"],
    "service": os.environ["CC_DEPLOY_SERVICE"],
    "host_loopback_url": os.environ["CC_DEPLOY_HOST_URL"],
    "pod_reverse_tunnel_url": os.environ["CC_DEPLOY_HOST_URL"],
    "service_id": identity["service_id"],
    "protocol_version": identity["protocol_version"],
    "implementation_version": identity["implementation_version"],
    "server_bundle_sha256": identity["server_bundle_sha256"],
    "rootfs_sha256": identity["runtime"]["rootfs_sha256"],
    "rootfs_manifest_file_sha256": identity["runtime"]["rootfs_manifest_file_sha256"],
    "hostile_probe": "hostile-probe.json",
    "client_files": {
        "bearer": "bearer",
        "hmac_key": "hmac-key",
        "identity": "identity.json",
        "client_provenance": "client-provenance.json",
    },
}
encoded = json.dumps(deployment, sort_keys=True, separators=(",", ":")).encode() + b"\n"
path = bundle / "deployment.json"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(descriptor, encoded)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

CC_CHECKSUM_BUNDLE="$BUNDLE_DIR" python3 -B - <<'PY'
import hashlib
import os
from pathlib import Path

bundle = Path(os.environ["CC_CHECKSUM_BUNDLE"])
records = []
for path in sorted(bundle.iterdir(), key=lambda value: value.name):
    if path.name in {"SHA256SUMS", "READY"}:
        continue
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe client-bundle entry: {path.name}")
    records.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
checksum_path = bundle / "SHA256SUMS"
descriptor = os.open(checksum_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(descriptor, "".join(records).encode("ascii"))
    os.fsync(descriptor)
finally:
    os.close(descriptor)
ready_path = bundle / "READY"
descriptor = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(descriptor, b"identity-fetched-twice-and-hmac-verified\n")
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
chmod 0600 "$BUNDLE_DIR"/*

completed=1
echo "executor_lima_deploy_ok run_id=${run_id} vm=${vm_name} service=${SERVICE_NAME} url=${HOST_URL} client_bundle=${BUNDLE_DIR}"
