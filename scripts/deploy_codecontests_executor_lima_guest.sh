#!/usr/bin/env bash
# Root-only guest half of the reproducible Lima executor deployment.
set -euo pipefail

readonly ROOTFS_DIR="/var/lib/codecontests-executor/rootfs"
readonly ROOTFS_ARTIFACT="/var/lib/codecontests-executor/rootfs-v2.tar"
readonly REQUEST_ROOT="/var/lib/codecontests-executor/requests"
readonly OPT_PARENT="/opt/codecontests-executor/deployments"
readonly ETC_PARENT="/etc/codecontests-executor/deployments"
readonly STATE_PARENT="/var/lib/codecontests-executor/deployments"
readonly MOUNT_UNIT_NAME='var-lib-codecontests\x2dexecutor-rootfs.mount'
readonly MOUNT_UNIT_PATH="/etc/systemd/system/${MOUNT_UNIT_NAME}"

usage() {
  cat >&2 <<'EOF'
usage:
  deploy_codecontests_executor_lima_guest.sh install --run-id ID --deployment-nonce HEX --stage-dir DIR
  deploy_codecontests_executor_lima_guest.sh hostile-probe --run-id ID --deployment-nonce HEX --stage-dir DIR
  deploy_codecontests_executor_lima_guest.sh freeze-identity --run-id ID --deployment-nonce HEX --stage-dir DIR
  deploy_codecontests_executor_lima_guest.sh rollback --run-id ID --deployment-nonce HEX
EOF
  exit 2
}

[[ "${EUID}" -eq 0 ]] || {
  echo "guest deployment helper must run as root" >&2
  exit 1
}
[[ "$#" -ge 1 ]] || usage
action=$1
shift
run_id=""
deployment_nonce=""
stage_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run-id)
      [[ "$#" -ge 2 ]] || usage
      run_id=$2
      shift 2
      ;;
    --deployment-nonce)
      [[ "$#" -ge 2 ]] || usage
      deployment_nonce=$2
      shift 2
      ;;
    --stage-dir)
      [[ "$#" -ge 2 ]] || usage
      stage_dir=$2
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
  echo "invalid run ID" >&2
  exit 1
}
[[ "$deployment_nonce" =~ ^[0-9a-f]{32}$ ]] || {
  echo "invalid deployment nonce" >&2
  exit 1
}

readonly RUN_ROOT="${OPT_PARENT}/${run_id}"
readonly CONFIG_ROOT="${ETC_PARENT}/${run_id}"
readonly STATE_ROOT="${STATE_PARENT}/${run_id}"
readonly SERVICE_NAME="codecontests-executor-${run_id}.service"
readonly SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
readonly PROBE_SERVICE_NAME="codecontests-executor-hostile-probe-${run_id}.service"
readonly PROBE_SERVICE_PATH="/etc/systemd/system/${PROBE_SERVICE_NAME}"
readonly SYSCTL_PATH="/etc/sysctl.d/90-codecontests-executor-${run_id}.conf"
readonly ROOTFS_MANIFEST="/var/lib/codecontests-executor/rootfs-v2.${run_id}.manifest.json"
readonly PROBE_OUTPUT_DIR="${STATE_ROOT}/hostile-probe"
readonly PROBE_OUTPUT="${PROBE_OUTPUT_DIR}/hostile-probe.json"
readonly HOST_PATH_PROBE="${STATE_ROOT}/host-path-probe"

require_regular() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "unsafe or missing staged file: $path" >&2
    exit 1
  }
}

marker_matches() {
  local marker=$1
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  [[ "$(<"$marker")" == "${run_id}:${deployment_nonce}" ]]
}

unit_marker_matches() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || return 1
  grep -Fqx "# deployment-run-id=${run_id}" "$path" &&
    grep -Fqx "# deployment-nonce=${deployment_nonce}" "$path"
}

render_template() {
  local source=$1
  local output=$2
  require_regular "$source"
  sed \
    -e "s|@RUN_ID@|${run_id}|g" \
    -e "s|@DEPLOY_NONCE@|${deployment_nonce}|g" \
    -e "s|@RUN_ROOT@|${RUN_ROOT}|g" \
    -e "s|@CONFIG_ROOT@|${CONFIG_ROOT}|g" \
    -e "s|@ROOTFS_MANIFEST_PATH@|${ROOTFS_MANIFEST}|g" \
    -e "s|@SERVICE_NAME@|${SERVICE_NAME}|g" \
    -e "s|@PROBE_SERVICE_NAME@|${PROBE_SERVICE_NAME}|g" \
    -e "s|@PROBE_OUTPUT_DIR@|${PROBE_OUTPUT_DIR}|g" \
    -e "s|@PROBE_OUTPUT@|${PROBE_OUTPUT}|g" \
    -e "s|@HOST_PATH_PROBE@|${HOST_PATH_PROBE}|g" \
    "$source" >"$output"
  if grep -Eq '@[A-Z0-9_]+@' "$output"; then
    echo "unexpanded deployment template placeholder" >&2
    exit 1
  fi
}

wait_for_service() {
  local ready=0 attempt restarts listener
  for attempt in $(seq 1 600); do
    restarts=$(systemctl show "$SERVICE_NAME" -p NRestarts --value)
    [[ "$restarts" == "0" ]] || {
      echo "executor service restarted during live attestation" >&2
      systemctl status --no-pager "$SERVICE_NAME" >&2 || true
      return 1
    }
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
      echo "executor service stopped during live attestation" >&2
      systemctl status --no-pager "$SERVICE_NAME" >&2 || true
      return 1
    fi
    listener=$(ss -ltnH 'sport = :8080' | awk '{print $4}')
    if [[ "$listener" == "127.0.0.1:8080" ]]; then
      ready=1
      break
    fi
    if (( attempt % 15 == 0 )); then
      echo "executor_guest_waiting_for_live_attestation seconds=${attempt}"
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || {
    echo "executor listener did not become ready within 600 seconds" >&2
    systemctl status --no-pager "$SERVICE_NAME" >&2 || true
    return 1
  }
  [[ "$(ss -ltnH 'sport = :8080' | awk '{print $4}')" == "127.0.0.1:8080" ]] || {
    echo "executor listener is not exactly guest loopback:8080" >&2
    return 1
  }
}

validate_stage() {
  [[ "$stage_dir" = /tmp/codecontests-executor-deploy.* ]] || {
    echo "stage directory must be a narrow /tmp deployment path" >&2
    exit 1
  }
  [[ -d "$stage_dir" && ! -L "$stage_dir" ]] || {
    echo "stage directory is missing or unsafe" >&2
    exit 1
  }
  [[ "$(realpath -e -- "$stage_dir")" == "$stage_dir" ]] || {
    echo "stage directory must be canonical" >&2
    exit 1
  }
}

install_deployment() {
  validate_stage
  local package_stage="${stage_dir}/codecontests_executor"
  local scripts_stage="${stage_dir}/scripts"
  local deploy_stage="${stage_dir}/deploy/codecontests-executor"
  local expected_package_entries
  expected_package_entries=$'__init__.py\ncgroup_gate.py\nclient.py\nprotocol.py\nsandbox_launcher.py\nservice.py\nsupervisor.py'
  [[ -d "$package_stage" && ! -L "$package_stage" ]] || {
    echo "staged executor package is missing" >&2
    exit 1
  }
  [[ "$(find "$package_stage" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" == "$expected_package_entries" ]] || {
    echo "staged executor package entry set is not exact" >&2
    exit 1
  }
  local staged_file source_name
  for source_name in __init__.py cgroup_gate.py client.py protocol.py sandbox_launcher.py service.py supervisor.py; do
    require_regular "$package_stage/$source_name"
  done
  local required_stage_files=(
    "${scripts_stage}/build_codecontests_rootfs_artifact.sh"
    "${scripts_stage}/build_codecontests_rootfs_manifest.py"
    "${scripts_stage}/measure_codecontests_executor_bundle.py"
    "${deploy_stage}/verify_staged_executor.py"
    "${deploy_stage}/codecontests-executor.lima.service.in"
    "${deploy_stage}/codecontests-executor-hostile-probe.lima.service.in"
    "${deploy_stage}/codecontests-rootfs.lima.mount.in"
    "${scripts_stage}/probe_codecontests_executor_live.py"
    "${stage_dir}/client-provenance.json"
    "${stage_dir}/bearer"
    "${stage_dir}/hmac-key"
  )
  for staged_file in "${required_stage_files[@]}"; do
    require_regular "$staged_file"
  done

  for staged_file in \
    "$RUN_ROOT" "$CONFIG_ROOT" "$STATE_ROOT" "$SERVICE_PATH" \
    "$PROBE_SERVICE_PATH" "$SYSCTL_PATH" "$ROOTFS_MANIFEST" \
    "$MOUNT_UNIT_PATH"; do
    [[ ! -e "$staged_file" && ! -L "$staged_file" ]] || {
      echo "refusing to overwrite existing deployment path: $staged_file" >&2
      exit 1
    }
  done
  [[ -d "$ROOTFS_DIR" && ! -L "$ROOTFS_DIR" ]] || {
    echo "canonical rootfs is missing or unsafe" >&2
    exit 1
  }
  [[ -x /opt/gvisor/20260721.0/runsc ]] || {
    echo "pinned runsc is unavailable" >&2
    exit 1
  }
  [[ -f /opt/gvisor/20260721.0/gvisor.tar.bz2 ]] || {
    echo "pinned gVisor archive is unavailable" >&2
    exit 1
  }
  [[ "$(nproc)" == "16" ]] || {
    echo "executor VM must expose exactly 16 vCPUs" >&2
    exit 1
  }
  local memory_bytes
  memory_bytes=$(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE)))
  [[ "$memory_bytes" -ge 30000000000 ]] || {
    echo "executor VM exposes less than 30 GB RAM" >&2
    exit 1
  }
  [[ "$(findmnt -n -o FSTYPE /sys/fs/cgroup)" == "cgroup2" ]] || {
    echo "executor VM does not use unified cgroup v2" >&2
    exit 1
  }
  [[ -z "$(ss -ltnH 'sport = :8080')" ]] || {
    echo "guest loopback executor port 8080 is already in use" >&2
    exit 1
  }
  [[ "$(< /proc/sys/kernel/unprivileged_userns_clone)" == "1" ]] || {
    echo "unprivileged user namespaces are disabled" >&2
    exit 1
  }
  [[ "$(wc -l </proc/swaps)" == "1" ]] || {
    echo "executor VM swap must be disabled" >&2
    exit 1
  }

  install -d -m 0755 -o root -g root "$OPT_PARENT"
  install -d -m 0700 -o root -g root "$ETC_PARENT" "$STATE_PARENT"
  install -d -m 0755 -o root -g root \
    "$RUN_ROOT" "$RUN_ROOT/codecontests_executor" "$RUN_ROOT/scripts" \
    "$RUN_ROOT/deploy" "$RUN_ROOT/deploy/codecontests-executor"
  install -d -m 0700 -o root -g root \
    "$CONFIG_ROOT" "$STATE_ROOT" "$PROBE_OUTPUT_DIR"
  printf '%s\n' "${run_id}:${deployment_nonce}" >"$RUN_ROOT/.deployment-owner"
  printf '%s\n' "${run_id}:${deployment_nonce}" >"$CONFIG_ROOT/.deployment-owner"
  printf '%s\n' "${run_id}:${deployment_nonce}" >"$STATE_ROOT/.deployment-owner"
  chmod 0600 "$RUN_ROOT/.deployment-owner" "$CONFIG_ROOT/.deployment-owner" "$STATE_ROOT/.deployment-owner"

  install -m 0444 -o root -g root "$package_stage"/*.py "$RUN_ROOT/codecontests_executor/"
  install -m 0444 -o root -g root \
    "$scripts_stage/build_codecontests_rootfs_artifact.sh" \
    "$scripts_stage/build_codecontests_rootfs_manifest.py" \
    "$scripts_stage/measure_codecontests_executor_bundle.py" \
    "$scripts_stage/probe_codecontests_executor_live.py" \
    "$RUN_ROOT/scripts/"
  install -m 0444 -o root -g root \
    "$deploy_stage/verify_staged_executor.py" \
    "$RUN_ROOT/deploy/codecontests-executor/"
  install -m 0600 -o root -g root "$stage_dir/bearer" "$CONFIG_ROOT/bearer"
  install -m 0600 -o root -g root "$stage_dir/hmac-key" "$CONFIG_ROOT/hmac-key"
  install -m 0600 -o root -g root \
    "$stage_dir/client-provenance.json" "$CONFIG_ROOT/client-provenance.json"
  [[ "$(wc -c <"$CONFIG_ROOT/bearer")" -ge 32 ]] || {
    echo "bearer token is too short" >&2
    exit 1
  }
  [[ "$(wc -c <"$CONFIG_ROOT/hmac-key")" -ge 32 ]] || {
    echo "HMAC key is too short" >&2
    exit 1
  }

  local expected_rootfs_sha
  expected_rootfs_sha=$(
    cd "$RUN_ROOT"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -B -c \
      'from codecontests_executor.protocol import ROOTFS_SHA256; print(ROOTFS_SHA256)'
  )
  [[ "$expected_rootfs_sha" =~ ^[0-9a-f]{64}$ && ! "$expected_rootfs_sha" =~ ^0+$ ]] || {
    echo "production rootfs hash is not pinned" >&2
    exit 1
  }
  if [[ -e "$ROOTFS_ARTIFACT" || -L "$ROOTFS_ARTIFACT" ]]; then
    [[ -f "$ROOTFS_ARTIFACT" && ! -L "$ROOTFS_ARTIFACT" ]] || {
      echo "existing rootfs-v2 artifact is unsafe" >&2
      exit 1
    }
    [[ "$(sha256sum "$ROOTFS_ARTIFACT" | awk '{print $1}')" == "$expected_rootfs_sha" ]] || {
      echo "existing rootfs-v2 artifact does not match protocol pin" >&2
      exit 1
    }
  else
    /usr/bin/bash "$RUN_ROOT/scripts/build_codecontests_rootfs_artifact.sh" \
      "$ROOTFS_DIR" "$ROOTFS_ARTIFACT"
  fi
  chown root:root "$ROOTFS_ARTIFACT"
  chmod 0600 "$ROOTFS_ARTIFACT"

  local manifest_result manifest_file_sha
  manifest_result=$(
    cd "$RUN_ROOT"
    /usr/bin/python3.12 -B scripts/build_codecontests_rootfs_manifest.py \
      --rootfs "$ROOTFS_DIR" \
      --rootfs-artifact "$ROOTFS_ARTIFACT" \
      --output "$ROOTFS_MANIFEST"
  )
  manifest_file_sha=$(sed -n 's/.*file_sha256=\([0-9a-f]\{64\}\).*/\1/p' <<<"$manifest_result")
  [[ "$manifest_file_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "could not parse rootfs manifest measurement" >&2
    exit 1
  }
  chmod 0600 "$ROOTFS_MANIFEST"

  local bundle_sha
  bundle_sha=$(
    cd "$RUN_ROOT"
    /usr/bin/python3.12 -I -B scripts/measure_codecontests_executor_bundle.py \
      --package-dir "$RUN_ROOT/codecontests_executor" \
      --output "$CONFIG_ROOT/server-bundle.inventory.json"
  )
  [[ "$bundle_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid measured server bundle digest" >&2
    exit 1
  }
  chmod 0600 "$CONFIG_ROOT/server-bundle.inventory.json"
  {
    printf 'CODECONTESTS_EXECUTOR_SERVICE_ID=codecontests-judge-v2:%s\n' "$run_id"
    printf 'CODECONTESTS_EXECUTOR_ROOTFS_MANIFEST_SHA256=%s\n' "$manifest_file_sha"
    printf 'CODECONTESTS_EXECUTOR_SERVER_BUNDLE_SHA256=%s\n' "$bundle_sha"
    printf 'CODECONTESTS_EXECUTOR_CLIENT_PROVENANCE_FILE=%s/client-provenance.json\n' "$CONFIG_ROOT"
  } >"$CONFIG_ROOT/environment"
  chmod 0600 "$CONFIG_ROOT/environment"
  printf 'codecontests-host-path-probe:%s:%s\n' \
    "$run_id" "$deployment_nonce" >"$HOST_PATH_PROBE"
  chmod 0400 "$HOST_PATH_PROBE"

  local old_apparmor
  old_apparmor=$(< /proc/sys/kernel/apparmor_restrict_unprivileged_userns)
  [[ "$old_apparmor" == "0" || "$old_apparmor" == "1" ]] || {
    echo "unexpected AppArmor userns sysctl value" >&2
    exit 1
  }
  {
    printf '# deployment-run-id=%s\n' "$run_id"
    printf '# deployment-nonce=%s\n' "$deployment_nonce"
    printf 'kernel.apparmor_restrict_unprivileged_userns = 0\n'
  } >"$SYSCTL_PATH"
  chown root:root "$SYSCTL_PATH"
  chmod 0644 "$SYSCTL_PATH"
  printf '%s\n' "$old_apparmor" >"$STATE_ROOT/apparmor-before"
  chmod 0600 "$STATE_ROOT/apparmor-before"
  /usr/sbin/sysctl --load "$SYSCTL_PATH" >/dev/null
  [[ "$(< /proc/sys/kernel/apparmor_restrict_unprivileged_userns)" == "0" ]] || {
    echo "failed to persist AppArmor userns policy" >&2
    exit 1
  }

  local rendered_mount rendered_service rendered_probe_service
  rendered_mount=$(mktemp "$STATE_ROOT/.mount-unit.XXXXXX")
  rendered_service=$(mktemp "$STATE_ROOT/.service-unit.XXXXXX")
  rendered_probe_service=$(mktemp "$STATE_ROOT/.probe-unit.XXXXXX")
  render_template "$deploy_stage/codecontests-rootfs.lima.mount.in" "$rendered_mount"
  render_template "$deploy_stage/codecontests-executor.lima.service.in" "$rendered_service"
  render_template \
    "$deploy_stage/codecontests-executor-hostile-probe.lima.service.in" \
    "$rendered_probe_service"
  install -m 0644 -o root -g root "$rendered_mount" "$MOUNT_UNIT_PATH"
  install -m 0644 -o root -g root "$rendered_service" "$SERVICE_PATH"
  install -m 0644 -o root -g root \
    "$rendered_probe_service" "$PROBE_SERVICE_PATH"
  rm -f -- "$rendered_mount" "$rendered_service" "$rendered_probe_service"

  find "$RUN_ROOT" -depth -type d -exec chmod 0555 {} +
  find "$RUN_ROOT" -type f -exec chmod a-w {} +
  install -d -m 0700 -o root -g root "$REQUEST_ROOT"
  systemctl daemon-reload
  systemctl enable --now "$MOUNT_UNIT_NAME" >/dev/null
  local mount_options
  mount_options=$(findmnt -n -M "$ROOTFS_DIR" -o OPTIONS)
  [[ ",${mount_options}," == *,ro,* ]] || {
    echo "rootfs dedicated bind mount is not read-only" >&2
    exit 1
  }
  systemctl enable --now "$SERVICE_NAME" >/dev/null
  wait_for_service

  {
    printf 'run_id=%s\n' "$run_id"
    printf 'service_name=%s\n' "$SERVICE_NAME"
    printf 'server_bundle_sha256=%s\n' "$bundle_sha"
    printf 'rootfs_artifact_sha256=%s\n' "$expected_rootfs_sha"
    printf 'rootfs_manifest_file_sha256=%s\n' "$manifest_file_sha"
  } >"$STATE_ROOT/deployment-state"
  chmod 0600 "$STATE_ROOT/deployment-state"
  echo "executor_guest_deploy_ok run_id=${run_id} service=${SERVICE_NAME} bundle_sha256=${bundle_sha}"
}

run_hostile_probe() {
  validate_stage
  marker_matches "$CONFIG_ROOT/.deployment-owner" || {
    echo "deployment ownership marker mismatch" >&2
    exit 1
  }
  unit_marker_matches "$PROBE_SERVICE_PATH" || {
    echo "hostile-probe unit ownership marker mismatch" >&2
    exit 1
  }
  [[ ! -e "$PROBE_OUTPUT" && ! -L "$PROBE_OUTPUT" ]] || {
    echo "hostile-probe artifact already exists" >&2
    exit 1
  }
  systemctl stop "$SERVICE_NAME"
  systemctl is-active --quiet "$SERVICE_NAME" && {
    echo "executor service did not stop before hostile probe" >&2
    exit 1
  }
  [[ -z "$(ss -ltnH 'sport = :8080')" ]] || {
    echo "executor listener remains live before hostile probe" >&2
    exit 1
  }
  [[ -z "$(find "$REQUEST_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "request state remains before hostile probe" >&2
    exit 1
  }
  systemctl reset-failed "$PROBE_SERVICE_NAME" >/dev/null 2>&1 || true
  systemctl start "$PROBE_SERVICE_NAME"
  [[ "$(systemctl show "$PROBE_SERVICE_NAME" -p Result --value)" == "success" ]] || {
    echo "hostile-probe unit did not report success" >&2
    exit 1
  }
  [[ -f "$PROBE_OUTPUT" && ! -L "$PROBE_OUTPUT" ]] || {
    echo "hostile-probe artifact was not published" >&2
    exit 1
  }
  chown root:root "$PROBE_OUTPUT"
  chmod 0600 "$PROBE_OUTPUT"
  local probe_summary
  probe_summary=$(/usr/bin/python3.12 -I -B - "$PROBE_OUTPUT" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
encoded = path.read_bytes()
value = json.loads(encoded)
if value.get("format") != "palaestra.codecontests.executor-live-hostile.v4":
    raise SystemExit("hostile-probe artifact format mismatch")
if value.get("rootfs_manifest_digest_before") != value.get("rootfs_manifest_digest_after"):
    raise SystemExit("hostile-probe rootfs measurement drifted")
if value.get("server_bundle_sha256_before") != value.get("server_bundle_sha256_after"):
    raise SystemExit("hostile-probe server bundle drifted")
if value.get("teardown") != {"gvisor_pids": [], "request_cgroup_children": [], "request_entries": []}:
    raise SystemExit("hostile-probe teardown is not empty")
print(
    "sha256=" + hashlib.sha256(encoded).hexdigest()
    + " cases=" + str(len(value.get("cases", {})))
    + " concurrency=" + str(len(value.get("concurrency_4", {})))
)
PY
  )
  systemctl start "$SERVICE_NAME"
  wait_for_service
  echo "executor_hostile_probe_ok run_id=${run_id} ${probe_summary}"
}

freeze_identity() {
  validate_stage
  marker_matches "$CONFIG_ROOT/.deployment-owner" || {
    echo "deployment ownership marker mismatch" >&2
    exit 1
  }
  systemctl is-active --quiet "$SERVICE_NAME"
  require_regular "$stage_dir/identity.json"
  require_regular "$stage_dir/identity.envelope.json"
  [[ ! -e "$CONFIG_ROOT/frozen-identity.json" && ! -L "$CONFIG_ROOT/frozen-identity.json" ]] || {
    echo "frozen identity already exists" >&2
    exit 1
  }
  [[ ! -e "$CONFIG_ROOT/frozen-identity.envelope.json" && ! -L "$CONFIG_ROOT/frozen-identity.envelope.json" ]] || {
    echo "frozen identity envelope already exists" >&2
    exit 1
  }
  (
    cd "$RUN_ROOT"
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.12 -B - \
      "$stage_dir/identity.json" \
      "$stage_dir/identity.envelope.json" \
      "$CONFIG_ROOT/hmac-key" \
      "codecontests-judge-v2:${run_id}" <<'PY'
import pathlib
import sys
from codecontests_executor.protocol import canonical_json, strict_json_loads, verify_envelope

identity_path, envelope_path, key_path = map(pathlib.Path, sys.argv[1:4])
expected_service_id = sys.argv[4]
identity = strict_json_loads(identity_path.read_bytes())
envelope = strict_json_loads(envelope_path.read_bytes())
key = key_path.read_bytes().rstrip(b"\r\n")
verified = verify_envelope(envelope, key, expected_kind="identity")
if verified != identity or canonical_json(identity) + b"\n" != identity_path.read_bytes():
    raise SystemExit("frozen identity payload/envelope mismatch")
if identity.get("service_id") != expected_service_id:
    raise SystemExit("frozen identity service ID mismatch")
PY
  )
  install -m 0600 -o root -g root \
    "$stage_dir/identity.json" "$CONFIG_ROOT/frozen-identity.json"
  install -m 0600 -o root -g root \
    "$stage_dir/identity.envelope.json" "$CONFIG_ROOT/frozen-identity.envelope.json"
  echo "executor_identity_frozen run_id=${run_id}"
}

rollback_deployment() {
  local ownership_seen=0
  if marker_matches "$CONFIG_ROOT/.deployment-owner"; then
    ownership_seen=1
  elif marker_matches "$RUN_ROOT/.deployment-owner"; then
    ownership_seen=1
  elif marker_matches "$STATE_ROOT/.deployment-owner"; then
    ownership_seen=1
  elif unit_marker_matches "$SERVICE_PATH"; then
    ownership_seen=1
  elif unit_marker_matches "$MOUNT_UNIT_PATH"; then
    ownership_seen=1
  fi
  [[ "$ownership_seen" -eq 1 ]] || {
    echo "refusing rollback without exact deployment ownership marker" >&2
    exit 1
  }

  if unit_marker_matches "$SERVICE_PATH"; then
    systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 || true
    systemctl stop "$SERVICE_NAME" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$SERVICE_NAME"; then
      echo "owned executor service did not stop; preserving its unit" >&2
      exit 1
    fi
    rm -f -- "$SERVICE_PATH"
  fi
  if unit_marker_matches "$PROBE_SERVICE_PATH"; then
    systemctl stop "$PROBE_SERVICE_NAME" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "$PROBE_SERVICE_NAME"; then
      echo "owned hostile-probe service did not stop; preserving its unit" >&2
      exit 1
    fi
    rm -f -- "$PROBE_SERVICE_PATH"
  fi
  if unit_marker_matches "$MOUNT_UNIT_PATH"; then
    systemctl disable "$MOUNT_UNIT_NAME" >/dev/null 2>&1 || true
    systemctl stop "$MOUNT_UNIT_NAME" >/dev/null 2>&1 || true
    if findmnt -n -M "$ROOTFS_DIR" >/dev/null 2>&1; then
      echo "owned rootfs bind mount did not stop; preserving its unit" >&2
      exit 1
    fi
    rm -f -- "$MOUNT_UNIT_PATH"
  fi
  if [[ -f "$SYSCTL_PATH" ]] && \
    grep -Fqx "# deployment-run-id=${run_id}" "$SYSCTL_PATH" && \
    grep -Fqx "# deployment-nonce=${deployment_nonce}" "$SYSCTL_PATH"; then
    if marker_matches "$STATE_ROOT/.deployment-owner" && [[ -f "$STATE_ROOT/apparmor-before" ]]; then
      local previous
      previous=$(<"$STATE_ROOT/apparmor-before")
      if [[ "$previous" == "0" || "$previous" == "1" ]]; then
        /usr/sbin/sysctl -q -w "kernel.apparmor_restrict_unprivileged_userns=${previous}" || true
      fi
    fi
    rm -f -- "$SYSCTL_PATH"
  fi
  if marker_matches "$CONFIG_ROOT/.deployment-owner"; then
    rm -rf -- "$CONFIG_ROOT"
  fi
  if marker_matches "$RUN_ROOT/.deployment-owner"; then
    chmod -R u+w "$RUN_ROOT"
    rm -rf -- "$RUN_ROOT"
  fi
  if marker_matches "$STATE_ROOT/.deployment-owner"; then
    rm -rf -- "$STATE_ROOT"
  fi
  systemctl daemon-reload
  echo "executor_guest_rollback_ok run_id=${run_id} preserved_rootfs_artifacts=1"
}

case "$action" in
  install) install_deployment ;;
  hostile-probe) run_hostile_probe ;;
  freeze-identity) freeze_identity ;;
  rollback) rollback_deployment ;;
  *) usage ;;
esac
