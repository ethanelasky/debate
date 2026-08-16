#!/usr/bin/env bash
# Continuous checkpoint off-pod sync (Ethan, 2026-08-12: "checkpoints have to
# be synced back to the persistent volume before shutdown"). Host deaths give
# no warning, so this syncs CONTINUOUSLY after each save rather than hooking
# shutdown: every completed step-*/final dir under one exact launch directory
# is either already inside its configured durable local directory or uploads
# once to an explicitly configured S3-compatible bucket.
#
# Usage:  CKPT_DIR=/workspace/checkpoints/<run>/<launch-namespace> \
#         RUN_NAME=<arm+suffix> \
#         DEBATE_LAUNCH_NAMESPACE=<exact-launch-namespace> \
#         CKPT_DESTINATION_JSON='<destination JSON>' \
#           setsid nohup bash scripts/ckpt_sync.sh > /root/ckpt_sync.log 2>&1 &
#
# CKPT_DESTINATION_JSON is submission configuration, never inferred from
# ambient credentials. It accepts exactly one of:
#   {"kind":"local","directory":"/absolute/canonical/durable/root"}
#   {"kind":"bucket","endpoint":"https://...","region":"...",
#    "bucket":"...","prefix":"configured/key/prefix"}
# Credentials are intentionally absent from this JSON. For a bucket they stay
# in the ambient provider environment (credential names may also be parsed,
# never sourced, from S3_ENV_FILE).
#
# CKPT_DIR is deliberately exact. This script never searches RUN_NAME prefixes,
# so a concurrent or retried launch cannot select a sibling attempt. It no-ops
# when CKPT_DIR is within the explicitly configured durable local directory.
# Legacy remote paths remain untouched: new uploads are namespaced.
set -u
export LC_ALL=C
readonly LC_ALL

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the exact checkpoint directory for this launch}"
RUN_NAME="${RUN_NAME:?set RUN_NAME (the scientific run name)}"
DEBATE_LAUNCH_NAMESPACE="${DEBATE_LAUNCH_NAMESPACE:?set DEBATE_LAUNCH_NAMESPACE}"
CKPT_DESTINATION_JSON="${CKPT_DESTINATION_JSON:?set CKPT_DESTINATION_JSON from run submission config}"
PYBIN="${PYBIN:-/workspace/envs/verl-b200/bin/python}"
QUIESCENT_SECS="${QUIESCENT_SECS:-90}" # a dir this old is a finished write, not a mid-save
INTERVAL="${INTERVAL:-120}"
S3_ENV_FILE="${S3_ENV_FILE:-/root/.runpod/s3.env}"

if [ "${#DEBATE_LAUNCH_NAMESPACE}" -lt 1 ] \
  || [ "${#DEBATE_LAUNCH_NAMESPACE}" -gt 128 ] \
  || [[ ! "$DEBATE_LAUNCH_NAMESPACE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "ckpt_sync: DEBATE_LAUNCH_NAMESPACE must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}" >&2
  exit 2
fi
case "$QUIESCENT_SECS" in ''|*[!0-9]*) echo "ckpt_sync: QUIESCENT_SECS must be a nonnegative integer" >&2; exit 2 ;; esac
case "$INTERVAL" in ''|*[!0-9]*) echo "ckpt_sync: INTERVAL must be a nonnegative integer" >&2; exit 2 ;; esac
[ -d "$CKPT_DIR" ] || { echo "ckpt_sync: exact checkpoint directory does not exist: $CKPT_DIR" >&2; exit 2; }
[ ! -L "$CKPT_DIR" ] || { echo "ckpt_sync: exact checkpoint directory must not be a symlink: $CKPT_DIR" >&2; exit 2; }

CKPT_DIR_REAL="$(cd "$CKPT_DIR" && pwd -P)" || {
  echo "ckpt_sync: cannot resolve exact checkpoint directory: $CKPT_DIR" >&2
  exit 2
}
if [ "$CKPT_DIR" != "$CKPT_DIR_REAL" ]; then
  echo "ckpt_sync: CKPT_DIR must be an absolute canonical non-symlink path: $CKPT_DIR_REAL" >&2
  exit 2
fi

# Parse the submission-carried destination before touching credentials or a
# provider. Duplicate and unknown JSON keys are rejected: accepting either
# would make the effective destination depend on parser quirks or misspelling.
DESTINATION_FIELDS="$(CKPT_DESTINATION_JSON="$CKPT_DESTINATION_JSON" "$PYBIN" - <<'PYEOF'
import json
import ipaddress
import os
import pathlib
import re
import sys
from urllib.parse import urlsplit


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def plain_string(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


try:
    destination = json.loads(
        os.environ["CKPT_DESTINATION_JSON"], object_pairs_hook=strict_object
    )
    if not isinstance(destination, dict):
        raise ValueError("top level must be an object")
    kind = destination.get("kind")
    if kind == "local":
        if set(destination) != {"kind", "directory"}:
            raise ValueError("local destination has missing or unknown keys")
        directory = plain_string(destination["directory"], "directory")
        path = pathlib.Path(directory)
        if not path.is_absolute():
            raise ValueError("directory must be absolute")
        if path.is_symlink() or not path.is_dir():
            raise ValueError("directory must be an existing non-symlink directory")
        canonical = str(path.resolve(strict=True))
        if directory != canonical:
            raise ValueError("directory must be canonical")
        fields = [kind, canonical, "-", "-", "-", "-"]
    elif kind == "bucket":
        required = {"kind", "endpoint", "region", "bucket", "prefix"}
        if set(destination) != required:
            raise ValueError("bucket destination has missing or unknown keys")
        endpoint = plain_string(destination["endpoint"], "endpoint")
        parsed = urlsplit(endpoint)
        try:
            endpoint_port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint has an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path != ""
            or "\\" in endpoint
            or endpoint != endpoint.strip()
        ):
            raise ValueError("endpoint must be an exact canonical HTTPS origin")
        endpoint_host = parsed.hostname
        if "%" in endpoint_host:
            raise ValueError("endpoint host must not use escapes or IPv6 zones")
        try:
            address = ipaddress.ip_address(endpoint_host)
        except ValueError:
            try:
                endpoint_host.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("endpoint DNS host must be ASCII") from exc
            if (
                endpoint_host != endpoint_host.lower()
                or endpoint_host.endswith(".")
                or len(endpoint_host) > 253
                or all(char in "0123456789." for char in endpoint_host)
            ):
                raise ValueError("endpoint DNS host is not canonical")
            labels = endpoint_host.split(".")
            if any(
                re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                is None
                for label in labels
            ):
                raise ValueError("endpoint DNS host has an invalid label")
            rendered_host = endpoint_host
        else:
            canonical_address = str(address)
            if endpoint_host != canonical_address:
                raise ValueError("endpoint IP address is not canonical")
            rendered_host = (
                f"[{canonical_address}]"
                if address.version == 6
                else canonical_address
            )
        canonical_endpoint = f"https://{rendered_host}"
        if endpoint_port is not None:
            if endpoint_port == 443:
                raise ValueError("endpoint must omit the default HTTPS port")
            canonical_endpoint += f":{endpoint_port}"
        if endpoint != canonical_endpoint:
            raise ValueError("endpoint must be an exact canonical HTTPS origin")
        region = plain_string(destination["region"], "region")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", region) is None:
            raise ValueError("region has an invalid format")
        bucket = plain_string(destination["bucket"], "bucket")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", bucket) is None:
            raise ValueError("bucket has an invalid format")
        prefix = plain_string(destination["prefix"], "prefix")
        if prefix.startswith("/") or prefix.endswith("/") or "\\" in prefix:
            raise ValueError("prefix must be a canonical relative key prefix")
        if any(part in {"", ".", ".."} for part in prefix.split("/")):
            raise ValueError("prefix contains an unsafe path component")
        fields = [kind, "-", endpoint, region, bucket, prefix]
    else:
        raise ValueError("kind must be local or bucket")
except (json.JSONDecodeError, ValueError, OSError) as exc:
    print(f"ckpt_sync: invalid CKPT_DESTINATION_JSON: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("\n".join(fields))
PYEOF
)" || exit $?
DESTINATION_KIND="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '1p')"
DESTINATION_LOCAL_DIRECTORY="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '2p')"
DESTINATION_ENDPOINT="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '3p')"
DESTINATION_REGION="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '4p')"
DESTINATION_BUCKET="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '5p')"
DESTINATION_PREFIX="$(printf '%s\n' "$DESTINATION_FIELDS" | sed -n '6p')"
unset DESTINATION_FIELDS
readonly DESTINATION_KIND DESTINATION_LOCAL_DIRECTORY DESTINATION_ENDPOINT \
  DESTINATION_REGION DESTINATION_BUCKET DESTINATION_PREFIX
export DESTINATION_KIND DESTINATION_ENDPOINT DESTINATION_REGION \
  DESTINATION_BUCKET DESTINATION_PREFIX

EXPECTED_RUN_COMPONENT="$(SYNC_RUN="$RUN_NAME" "$PYBIN" - <<'PYEOF'
import hashlib
import os
import re

value = os.environ["SYNC_RUN"]
if (
    len(value.encode("utf-8")) <= 128
    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is not None
):
    print(value)
else:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")[:80]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    print(f"{slug or 'run'}-{digest}")
PYEOF
)" || {
  echo "ckpt_sync: cannot render RUN_NAME as a safe checkpoint path component" >&2
  exit 2
}
if [ "$(basename "$CKPT_DIR_REAL")" != "$DEBATE_LAUNCH_NAMESPACE" ] \
  || [ "$(basename "$(dirname "$CKPT_DIR_REAL")")" != "$EXPECTED_RUN_COMPONENT" ]; then
  echo "ckpt_sync: CKPT_DIR must be exactly <checkpoint-root>/$EXPECTED_RUN_COMPONENT/$DEBATE_LAUNCH_NAMESPACE" >&2
  exit 2
fi

if [ "$DESTINATION_KIND" = local ]; then
  if [ "$DESTINATION_LOCAL_DIRECTORY" = / ]; then
    LOCAL_CONTAINS_CHECKPOINT=1
  else
    case "$CKPT_DIR_REAL/" in
      "$DESTINATION_LOCAL_DIRECTORY/"*) LOCAL_CONTAINS_CHECKPOINT=1 ;;
      *) LOCAL_CONTAINS_CHECKPOINT=0 ;;
    esac
  fi
  case "$LOCAL_CONTAINS_CHECKPOINT" in
    1)
      LOCAL_DESTINATION_NOOP=1
      ;;
    0)
      echo "ckpt_sync: CKPT_DIR is outside configured local destination" >&2
      exit 2
      ;;
  esac
else
  LOCAL_DESTINATION_NOOP=0
fi

# Namespace local coordination too. Distinct attempts must not overwrite each
# other's pid/state, while duplicate synchronizers for the same immutable
# attempt must not race through the remote no-overwrite boundary.
STATE="${CKPT_SYNC_STATE:-/root/.ckpt_synced.${DEBATE_LAUNCH_NAMESPACE}}"
PID_FILE="${CKPT_SYNC_PID_FILE:-/root/ckpt_sync.${DEBATE_LAUNCH_NAMESPACE}.pid}"
LOCK_FILE="${CKPT_SYNC_LOCK_FILE:-/root/ckpt_sync.${DEBATE_LAUNCH_NAMESPACE}.lock}"
SYNC_ONCE="${CKPT_SYNC_ONCE:-0}"

# A credential file is parsed later for an allowlist of credential names only;
# it is never sourced. Freeze every validated identity, destination,
# executable, and daemon control too, so no later step can redirect work or
# alter which launch is being synchronized.
readonly CKPT_DIR CKPT_DIR_REAL RUN_NAME DEBATE_LAUNCH_NAMESPACE \
  CKPT_DESTINATION_JSON PYBIN QUIESCENT_SECS INTERVAL S3_ENV_FILE \
  EXPECTED_RUN_COMPONENT STATE PID_FILE LOCK_FILE SYNC_ONCE \
  LOCAL_DESTINATION_NOOP

# One canonical digest names the immutable launch attempt and its exact
# checkpoint destination. Every coordination role is bound to this digest so
# an unrelated owner-owned file cannot be adopted merely because its mode is
# private.
ATTEMPT_IDENTITY_DIGEST="$(STATE_SAFE_RUN="$EXPECTED_RUN_COMPONENT" \
  STATE_NAMESPACE="$DEBATE_LAUNCH_NAMESPACE" \
  STATE_CKPT_DIR="$CKPT_DIR_REAL" \
  STATE_DEST_KIND="$DESTINATION_KIND" \
  STATE_DEST_LOCAL="$DESTINATION_LOCAL_DIRECTORY" \
  STATE_DEST_ENDPOINT="$DESTINATION_ENDPOINT" \
  STATE_DEST_REGION="$DESTINATION_REGION" \
  STATE_DEST_BUCKET="$DESTINATION_BUCKET" \
  STATE_DEST_PREFIX="$DESTINATION_PREFIX" "$PYBIN" - <<'PYEOF'
import hashlib
import json
import os

kind = os.environ["STATE_DEST_KIND"]
destination = (
    {"kind": "local", "directory": os.environ["STATE_DEST_LOCAL"]}
    if kind == "local"
    else {
        "kind": "bucket",
        "endpoint": os.environ["STATE_DEST_ENDPOINT"],
        "region": os.environ["STATE_DEST_REGION"],
        "bucket": os.environ["STATE_DEST_BUCKET"],
        "prefix": os.environ["STATE_DEST_PREFIX"],
    }
)
identity = {
    "safe_run": os.environ["STATE_SAFE_RUN"],
    "namespace": os.environ["STATE_NAMESPACE"],
    "checkpoint_dir": os.environ["STATE_CKPT_DIR"],
    "destination": destination,
}
canonical = json.dumps(
    identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PYEOF
)" || {
  echo "ckpt_sync: cannot construct canonical attempt identity" >&2
  exit 2
}
readonly ATTEMPT_IDENTITY_DIGEST

# Validate all coordination paths as one read-only transaction before opening
# or mutating any of their leaves. Their stable canonical parents are pinned
# for the secure opens below.
COORD_PARENT_FIELDS="$(COORD_STATE="$STATE" COORD_PID="$PID_FILE" \
  COORD_LOCK="$LOCK_FILE" COORD_CKPT_DIR="$CKPT_DIR_REAL" \
  COORD_IDENTITY="$ATTEMPT_IDENTITY_DIGEST" "$PYBIN" - <<'PYEOF'
import os
import pathlib
import re
import stat

ckpt = pathlib.Path(os.environ["COORD_CKPT_DIR"])
paths = {
    "state": pathlib.Path(os.environ["COORD_STATE"]),
    "pid": pathlib.Path(os.environ["COORD_PID"]),
    "lock": pathlib.Path(os.environ["COORD_LOCK"]),
}
if len({str(path) for path in paths.values()}) != len(paths):
    raise RuntimeError("coordination paths must be pairwise distinct")
parent_stats = {}
identity = os.environ["COORD_IDENTITY"]
state_record = re.compile(
    rb"complete-v1\t(?:final|step-[A-Za-z0-9._-]+)\t[0-9a-f]{64}\n"
)


def read_exact_role(path, role, expected_stat):
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot securely open existing {role} leaf") from exc
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (
            expected_stat.st_dev,
            expected_stat.st_ino,
        ):
            raise RuntimeError(f"{role} leaf changed while validating")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
    finally:
        os.close(fd)
    if role == "lock":
        valid = content == f"lock-v1\t{identity}\n".encode("ascii")
    elif role == "pid":
        valid = re.fullmatch(
            rb"pid-v1\t" + identity.encode("ascii") + rb"\t[1-9][0-9]*\n",
            content,
        ) is not None
    else:
        lines = content.splitlines(keepends=True)
        header = f"identity-v1\t{identity}\n".encode("ascii")
        valid = (
            bool(lines)
            and lines[0] == header
            and all(state_record.fullmatch(line) is not None for line in lines[1:])
            and len(set(lines[1:])) == len(lines[1:])
        )
    if not valid:
        raise RuntimeError(f"existing {role} leaf has the wrong role or identity")


for role, path in paths.items():
    raw = str(path)
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise RuntimeError(f"{role} path contains control characters")
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RuntimeError(f"{role} path must be an absolute canonical leaf")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or str(parent.resolve()) != str(parent):
        raise RuntimeError(f"{role} parent must be a stable canonical directory")
    parent_stat = os.lstat(parent)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise RuntimeError(f"{role} parent must be owned by euid with mode 0700")
    parent_stats[role] = parent_stat
    if os.path.commonpath((str(ckpt), raw)) == str(ckpt):
        raise RuntimeError(f"{role} path must be outside CKPT_DIR")
    try:
        leaf = os.lstat(path)
    except FileNotFoundError:
        continue
    if (
        not stat.S_ISREG(leaf.st_mode)
        or leaf.st_nlink != 1
        or leaf.st_uid != os.geteuid()
        or stat.S_IMODE(leaf.st_mode) != 0o600
    ):
        raise RuntimeError(
            f"existing {role} leaf must be regular, non-symlink, single-link mode 0600"
        )
    if str(path.resolve()) != raw:
        raise RuntimeError(f"existing {role} leaf must be canonical")
    read_exact_role(path, role, leaf)
print(
    "\t".join(
        str(value)
        for role in ("state", "pid", "lock")
        for value in (parent_stats[role].st_dev, parent_stats[role].st_ino)
    )
)
PYEOF
)" || {
  echo "ckpt_sync: unsafe coordination path configuration" >&2
  exit 2
}
IFS=$'\t' read -r STATE_PARENT_DEVICE STATE_PARENT_INODE \
  PID_PARENT_DEVICE PID_PARENT_INODE LOCK_PARENT_DEVICE LOCK_PARENT_INODE \
  <<< "$COORD_PARENT_FIELDS"
unset COORD_PARENT_FIELDS
readonly STATE_PARENT_DEVICE STATE_PARENT_INODE PID_PARENT_DEVICE \
  PID_PARENT_INODE LOCK_PARENT_DEVICE LOCK_PARENT_INODE

# A configured local destination certifies the existing launch tree as already
# durable. Inspect the entire tree without following links before creating any
# coordination leaf, so unsafe content cannot receive a successful no-op.
if [ "$LOCAL_DESTINATION_NOOP" = 1 ]; then
  LOCAL_CKPT_DIR="$CKPT_DIR_REAL" "$PYBIN" - <<'PYEOF' || {
import os
import pathlib
import stat

root = pathlib.Path(os.environ["LOCAL_CKPT_DIR"])
for current, dirnames, filenames in os.walk(root, followlinks=False):
    current_path = pathlib.Path(current)
    current_stat = os.lstat(current_path)
    if not stat.S_ISDIR(current_stat.st_mode):
        raise RuntimeError(f"local checkpoint entry is not a directory: {current_path}")
    for name in dirnames:
        entry = current_path / name
        entry_stat = os.lstat(entry)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise RuntimeError(f"refusing symlink in local checkpoint tree: {entry}")
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise RuntimeError(f"refusing non-directory in local checkpoint tree: {entry}")
    for name in filenames:
        entry = current_path / name
        entry_stat = os.lstat(entry)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise RuntimeError(f"refusing symlink in local checkpoint tree: {entry}")
        if not stat.S_ISREG(entry_stat.st_mode):
            raise RuntimeError(f"refusing non-regular local checkpoint entry: {entry}")
        if entry_stat.st_nlink != 1:
            raise RuntimeError(f"refusing hard-linked local checkpoint entry: {entry}")
PYEOF
    echo "ckpt_sync: unsafe existing local checkpoint tree" >&2
    exit 2
  }
fi

# Securely initialize the lock without truncation, pin it, then retain and
# verify the shell descriptor before flock. Never redirect with `>` here.
LOCK_IDENTITY_FIELDS="$(COORD_PATH="$LOCK_FILE" \
  EXPECTED_PARENT_DEVICE="$LOCK_PARENT_DEVICE" \
  EXPECTED_PARENT_INODE="$LOCK_PARENT_INODE" \
  EXPECTED_ROLE_IDENTITY="$ATTEMPT_IDENTITY_DIGEST" "$PYBIN" - <<'PYEOF'
import fcntl
import os
import pathlib
import secrets
import stat

path = pathlib.Path(os.environ["COORD_PATH"])
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
parent_flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_DIRECTORY"):
    parent_flags |= os.O_DIRECTORY
parent_fd = os.open(path.parent, parent_flags)
parent = os.fstat(parent_fd)
if (
    (parent.st_dev, parent.st_ino) != expected_parent
    or parent.st_uid != os.geteuid()
    or stat.S_IMODE(parent.st_mode) != 0o700
):
    raise RuntimeError("lock parent changed")
fcntl.flock(parent_fd, fcntl.LOCK_EX)
payload = f"lock-v1\t{os.environ['EXPECTED_ROLE_IDENTITY']}\n".encode("ascii")
base_flags = os.O_RDWR | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    base_flags |= os.O_NOFOLLOW
created = False
temporary = None
try:
    fd = os.open(path, base_flags)
except FileNotFoundError:
    temporary = path.with_name(
        f".{path.name}.lock-init-v1-{os.getpid()}-{secrets.token_hex(8)}"
    )
    fd = os.open(temporary, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
    created = True


def write_all(fd, data):
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise RuntimeError("short lock write")
        offset += written


try:
    opened = os.fstat(fd)
    named = os.lstat(temporary if created else path)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        or (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino) != expected_parent
    ):
        raise RuntimeError("lock leaf changed or is unsafe")
    if created:
        write_all(fd, payload)
        os.fsync(fd)
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("lock canonical path appeared during initialization")
    os.lseek(fd, 0, os.SEEK_SET)
    observed = b""
    while True:
        chunk = os.read(fd, 1024)
        if not chunk:
            break
        observed += chunk
    if observed != payload:
        raise RuntimeError("lock role or identity mismatch")
    if created:
        os.rename(temporary, path)
        os.fsync(parent_fd)
        published = os.lstat(path)
        if (published.st_dev, published.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("published lock inode mismatch")
    print(f"{opened.st_dev}\t{opened.st_ino}")
finally:
    os.close(fd)
    os.close(parent_fd)
PYEOF
)" || { echo "ckpt_sync: cannot securely initialize lock" >&2; exit 2; }
IFS=$'\t' read -r LOCK_DEVICE LOCK_INODE <<< "$LOCK_IDENTITY_FIELDS"
unset LOCK_IDENTITY_FIELDS
readonly LOCK_DEVICE LOCK_INODE
exec 9>>"$LOCK_FILE" || { echo "ckpt_sync: cannot retain lock file" >&2; exit 2; }
if ! COORD_PATH="$LOCK_FILE" COORD_FD=9 EXPECTED_DEVICE="$LOCK_DEVICE" \
  EXPECTED_INODE="$LOCK_INODE" EXPECTED_PARENT_DEVICE="$LOCK_PARENT_DEVICE" \
  EXPECTED_PARENT_INODE="$LOCK_PARENT_INODE" \
  EXPECTED_ROLE_IDENTITY="$ATTEMPT_IDENTITY_DIGEST" "$PYBIN" - <<'PYEOF'
import os
import pathlib
import stat

path = pathlib.Path(os.environ["COORD_PATH"])
opened = os.fstat(int(os.environ["COORD_FD"]))
named = os.lstat(path)
expected = (int(os.environ["EXPECTED_DEVICE"]), int(os.environ["EXPECTED_INODE"]))
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
parent = os.lstat(path.parent)
if (
    not stat.S_ISREG(opened.st_mode)
    or opened.st_nlink != 1
    or opened.st_uid != os.geteuid()
    or stat.S_IMODE(opened.st_mode) != 0o600
    or (opened.st_dev, opened.st_ino) != expected
    or (named.st_dev, named.st_ino) != expected
    or (parent.st_dev, parent.st_ino) != expected_parent
    or parent.st_uid != os.geteuid()
    or stat.S_IMODE(parent.st_mode) != 0o700
):
    raise RuntimeError("retained lock descriptor is unsafe")
flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
check_fd = os.open(path, flags)
try:
    checked = os.fstat(check_fd)
    if (checked.st_dev, checked.st_ino) != expected:
        raise RuntimeError("lock changed while checking role")
    content = b""
    while True:
        chunk = os.read(check_fd, 1024)
        if not chunk:
            break
        content += chunk
finally:
    os.close(check_fd)
expected_content = f"lock-v1\t{os.environ['EXPECTED_ROLE_IDENTITY']}\n".encode("ascii")
if content != expected_content:
    raise RuntimeError("retained lock role or identity mismatch")
PYEOF
then
  echo "ckpt_sync: cannot validate retained lock descriptor" >&2
  exit 2
fi
flock -n 9 || {
  echo "ckpt_sync: another synchronizer already owns namespace $DEBATE_LAUNCH_NAMESPACE" >&2
  exit 5
}

# Bind this state file to the full canonical attempt+destination identity
# before any local no-op or provider mutation. Old headerless state is not
# adopted or rewritten. The device/inode returned here is pinned for every
# later completion check/append, so replacing the pathname cannot redirect it.
STATE_IDENTITY_FIELDS="$(STATE_PATH="$STATE" \
  EXPECTED_PARENT_DEVICE="$STATE_PARENT_DEVICE" \
  EXPECTED_PARENT_INODE="$STATE_PARENT_INODE" \
  EXPECTED_ROLE_IDENTITY="$ATTEMPT_IDENTITY_DIGEST" "$PYBIN" - <<'PYEOF'
import fcntl
import os
import pathlib
import re
import secrets
import stat

path = pathlib.Path(os.environ["STATE_PATH"])
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
parent_flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_DIRECTORY"):
    parent_flags |= os.O_DIRECTORY
parent_fd = os.open(path.parent, parent_flags)
parent_before = os.fstat(parent_fd)
if (
    (parent_before.st_dev, parent_before.st_ino) != expected_parent
    or parent_before.st_uid != os.geteuid()
    or stat.S_IMODE(parent_before.st_mode) != 0o700
):
    raise RuntimeError("state parent changed")
fcntl.flock(parent_fd, fcntl.LOCK_EX)
identity_digest = os.environ["EXPECTED_ROLE_IDENTITY"]
header = f"identity-v1\t{identity_digest}\n".encode("ascii")
base_flags = os.O_RDWR | os.O_APPEND | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    base_flags |= os.O_NOFOLLOW
created = False
temporary = None
try:
    fd = os.open(path, base_flags)
except FileNotFoundError:
    temporary = path.with_name(
        f".{path.name}.state-init-v1-{os.getpid()}-{secrets.token_hex(8)}"
    )
    fd = os.open(temporary, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
    created = True


def write_all(fd, data):
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise RuntimeError("short state write")
        offset += written


try:
    opened = os.fstat(fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
    ):
        raise RuntimeError("state must be a private regular file with one link")
    named = os.lstat(temporary if created else path)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError("state pathname changed while opening")
    parent_after = os.fstat(parent_fd)
    if (
        (parent_after.st_dev, parent_after.st_ino) != expected_parent
        or parent_after.st_uid != os.geteuid()
        or stat.S_IMODE(parent_after.st_mode) != 0o700
    ):
        raise RuntimeError("state parent changed while opening")
    if created:
        write_all(fd, header)
        os.fsync(fd)
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("state canonical path appeared during initialization")
    os.lseek(fd, 0, os.SEEK_SET)
    content = b""
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        content += chunk
    lines = content.splitlines(keepends=True)
    if not lines or lines[0] != header:
        raise RuntimeError("state identity header is missing or mismatched")
    completion_pattern = re.compile(
        rb"complete-v1\t(?:final|step-[A-Za-z0-9._-]+)\t[0-9a-f]{64}\n"
    )
    if any(completion_pattern.fullmatch(line) is None for line in lines[1:]):
        raise RuntimeError("state contains a malformed completion record")
    if len(set(lines[1:])) != len(lines[1:]):
        raise RuntimeError("state contains duplicate completion records")
    if created:
        os.rename(temporary, path)
        os.fsync(parent_fd)
        published = os.lstat(path)
        if (published.st_dev, published.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("published state inode mismatch")
    print(f"{identity_digest}\t{opened.st_dev}\t{opened.st_ino}")
finally:
    os.close(fd)
    os.close(parent_fd)
PYEOF
)" || {
  echo "ckpt_sync: cannot initialize or validate checkpoint state identity" >&2
  exit 2
}
IFS=$'\t' read -r STATE_IDENTITY_DIGEST STATE_DEVICE STATE_INODE \
  <<< "$STATE_IDENTITY_FIELDS"
unset STATE_IDENTITY_FIELDS
readonly STATE_IDENTITY_DIGEST STATE_DEVICE STATE_INODE

# Retain one validated read/write descriptor across every provider call. Bash
# cannot request O_NOFOLLOW itself, so the secure initializer above pins the
# inode first; this second open is accepted only when its fstat and the current
# pathname still name that exact inode. No provider/local success occurs before
# this check, and later state reads/appends reuse this descriptor rather than
# reopening the pathname.
exec 8<>"$STATE" || {
  echo "ckpt_sync: cannot retain checkpoint state descriptor" >&2
  exit 2
}
STATE_RETAINED_FD=8
if ! EXPECTED_STATE_DEVICE="$STATE_DEVICE" EXPECTED_STATE_INODE="$STATE_INODE" \
  EXPECTED_PARENT_DEVICE="$STATE_PARENT_DEVICE" \
  EXPECTED_PARENT_INODE="$STATE_PARENT_INODE" \
  STATE_PATH="$STATE" STATE_FD="$STATE_RETAINED_FD" "$PYBIN" - <<'PYEOF'
import os
import pathlib
import stat

fd = int(os.environ["STATE_FD"])
opened = os.fstat(fd)
named = os.lstat(pathlib.Path(os.environ["STATE_PATH"]))
parent = os.lstat(pathlib.Path(os.environ["STATE_PATH"]).parent)
expected = (
    int(os.environ["EXPECTED_STATE_DEVICE"]),
    int(os.environ["EXPECTED_STATE_INODE"]),
)
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
if (
    not stat.S_ISREG(opened.st_mode)
    or opened.st_nlink != 1
    or opened.st_uid != os.geteuid()
    or stat.S_IMODE(opened.st_mode) != 0o600
    or (opened.st_dev, opened.st_ino) != expected
    or (named.st_dev, named.st_ino) != expected
    or (parent.st_dev, parent.st_ino) != expected_parent
    or parent.st_uid != os.geteuid()
    or stat.S_IMODE(parent.st_mode) != 0o700
):
    raise RuntimeError("retained state descriptor does not match initialized inode")
PYEOF
then
  echo "ckpt_sync: cannot validate retained checkpoint state descriptor" >&2
  exit 2
fi
readonly STATE_RETAINED_FD

if ! COORD_PATH="$PID_FILE" COORD_PID="$$" \
  EXPECTED_PARENT_DEVICE="$PID_PARENT_DEVICE" \
  EXPECTED_PARENT_INODE="$PID_PARENT_INODE" \
  EXPECTED_ROLE_IDENTITY="$ATTEMPT_IDENTITY_DIGEST" "$PYBIN" - <<'PYEOF'
import fcntl
import os
import pathlib
import re
import secrets
import stat

path = pathlib.Path(os.environ["COORD_PATH"])
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
parent_flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_DIRECTORY"):
    parent_flags |= os.O_DIRECTORY
parent_fd = os.open(path.parent, parent_flags)
parent_before = os.fstat(parent_fd)
if (
    (parent_before.st_dev, parent_before.st_ino) != expected_parent
    or parent_before.st_uid != os.geteuid()
    or stat.S_IMODE(parent_before.st_mode) != 0o700
):
    raise RuntimeError("pid parent changed")
fcntl.flock(parent_fd, fcntl.LOCK_EX)
payload = (
    f"pid-v1\t{os.environ['EXPECTED_ROLE_IDENTITY']}\t{os.environ['COORD_PID']}\n"
).encode("ascii")
base_flags = os.O_RDWR | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    base_flags |= os.O_NOFOLLOW


def read_all(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def write_all(fd, data):
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise RuntimeError("short pid write")
        offset += written


existing_fd = None
existing_identity = None
try:
    existing_fd = os.open(path, base_flags)
except FileNotFoundError:
    pass
if existing_fd is not None:
    opened = os.fstat(existing_fd)
    named = os.lstat(path)
    parent_now = os.fstat(parent_fd)
    existing_identity = (opened.st_dev, opened.st_ino)
    old_content = read_all(existing_fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or existing_identity != (named.st_dev, named.st_ino)
        or (parent_now.st_dev, parent_now.st_ino) != expected_parent
        or parent_now.st_uid != os.geteuid()
        or stat.S_IMODE(parent_now.st_mode) != 0o700
        or re.fullmatch(
            rb"pid-v1\t"
            + os.environ["EXPECTED_ROLE_IDENTITY"].encode("ascii")
            + rb"\t[1-9][0-9]*\n",
            old_content,
        )
        is None
    ):
        raise RuntimeError("pid leaf changed, has wrong role, or is unsafe")

temporary = path.with_name(
    f".{path.name}.pid-init-v1-{os.getpid()}-{secrets.token_hex(8)}"
)
fd = os.open(temporary, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
try:
    opened = os.fstat(fd)
    named = os.lstat(temporary)
    parent_now = os.fstat(parent_fd)
    opened_identity = (opened.st_dev, opened.st_ino)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened_identity != (named.st_dev, named.st_ino)
        or (parent_now.st_dev, parent_now.st_ino) != expected_parent
        or parent_now.st_uid != os.geteuid()
        or stat.S_IMODE(parent_now.st_mode) != 0o700
    ):
        raise RuntimeError("pid leaf changed or is unsafe")
    write_all(fd, payload)
    os.fsync(fd)
    if read_all(fd) != payload:
        raise RuntimeError("pid reread verification failed")
    parent_before_publish = os.fstat(parent_fd)
    if (
        (parent_before_publish.st_dev, parent_before_publish.st_ino)
        != expected_parent
        or parent_before_publish.st_uid != os.geteuid()
        or stat.S_IMODE(parent_before_publish.st_mode) != 0o700
    ):
        raise RuntimeError("pid parent changed before publish")
    if existing_fd is None:
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("pid canonical path appeared during initialization")
    else:
        current_opened = os.fstat(existing_fd)
        current_named = os.lstat(path)
        if (
            (current_opened.st_dev, current_opened.st_ino) != existing_identity
            or (current_named.st_dev, current_named.st_ino) != existing_identity
            or read_all(existing_fd) != old_content
        ):
            raise RuntimeError("pid canonical role changed before publish")
    os.rename(temporary, path)
    os.fsync(parent_fd)
    published = os.lstat(path)
    if (published.st_dev, published.st_ino) != opened_identity:
        raise RuntimeError("published pid inode mismatch")
finally:
    os.close(fd)
    if existing_fd is not None:
        os.close(existing_fd)
    os.close(parent_fd)
PYEOF
then
  echo "ckpt_sync: cannot securely write pid file $PID_FILE" >&2
  exit 2
fi

if [ "$LOCAL_DESTINATION_NOOP" = 1 ]; then
  echo "ckpt_sync: $CKPT_DIR is inside configured durable local destination; nothing to do"
  exit 0
fi

echo "ckpt_sync: started pid $$ run=$RUN_NAME namespace=$DEBATE_LAUNCH_NAMESPACE dir=$CKPT_DIR"

while true; do
  now=$(date +%s)
  scan_failed=0
  # The only glob is below the caller-supplied exact launch directory. In
  # particular, there is no RUN_NAME* search that can cross launch namespaces.
  shopt -s nullglob
  checkpoint_entries=("$CKPT_DIR"/step-*)
  shopt -u nullglob
  if [ -e "$CKPT_DIR/final" ] || [ -L "$CKPT_DIR/final" ]; then
    checkpoint_entries+=("$CKPT_DIR/final")
  fi
  for dir in "${checkpoint_entries[@]}"; do
    if [ -L "$dir" ]; then
      echo "ckpt_sync: refusing symlinked checkpoint entry outside the exact launch boundary: $dir" >&2
      scan_failed=1
      continue
    fi
    if [ ! -d "$dir" ]; then
      echo "ckpt_sync: refusing non-directory checkpoint entry: $dir" >&2
      scan_failed=1
      continue
    fi
    STATE_RECORD="$(STATE_STEP="$(basename "$dir")" STATE_DIR="$dir" "$PYBIN" - <<'PYEOF'
import hashlib
import os
import re

step = os.environ["STATE_STEP"]
if re.fullmatch(r"(?:final|step-[A-Za-z0-9._-]+)", step) is None:
    raise SystemExit("invalid checkpoint step directory name")
path_digest = hashlib.sha256(os.environ["STATE_DIR"].encode("utf-8")).hexdigest()
print(f"complete-v1\t{step}\t{path_digest}")
PYEOF
)" || { echo "ckpt_sync: cannot construct completion record for $dir" >&2; exit 2; }
    if STATE_PATH="$STATE" STATE_FD="$STATE_RETAINED_FD" \
      EXPECTED_STATE_IDENTITY="$STATE_IDENTITY_DIGEST" \
      EXPECTED_STATE_DEVICE="$STATE_DEVICE" EXPECTED_STATE_INODE="$STATE_INODE" \
      EXPECTED_PARENT_DEVICE="$STATE_PARENT_DEVICE" \
      EXPECTED_PARENT_INODE="$STATE_PARENT_INODE" \
      STATE_RECORD="$STATE_RECORD" "$PYBIN" - <<'PYEOF'
import os
import pathlib
import re
import stat
import sys
import traceback


def validation_failure(exc_type, exc, tb):
    """Reserve exit 1 solely for a valid state with an absent record."""
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()
    os._exit(2)


sys.excepthook = validation_failure

path = pathlib.Path(os.environ["STATE_PATH"])
fd = int(os.environ["STATE_FD"])
opened = os.fstat(fd)
named = os.lstat(path)
parent = os.lstat(path.parent)
expected_inode = (
    int(os.environ["EXPECTED_STATE_DEVICE"]),
    int(os.environ["EXPECTED_STATE_INODE"]),
)
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
if (
    not stat.S_ISREG(opened.st_mode)
    or opened.st_nlink != 1
    or opened.st_uid != os.geteuid()
    or stat.S_IMODE(opened.st_mode) != 0o600
    or (opened.st_dev, opened.st_ino) != expected_inode
    or (named.st_dev, named.st_ino) != expected_inode
    or (parent.st_dev, parent.st_ino) != expected_parent
    or parent.st_uid != os.geteuid()
    or stat.S_IMODE(parent.st_mode) != 0o700
):
    raise RuntimeError("state inode changed")
os.lseek(fd, 0, os.SEEK_SET)
content = b""
while True:
    chunk = os.read(fd, 1024 * 1024)
    if not chunk:
        break
    content += chunk
lines = content.splitlines(keepends=True)
header = f"identity-v1\t{os.environ['EXPECTED_STATE_IDENTITY']}\n".encode("ascii")
pattern = re.compile(rb"complete-v1\t(?:final|step-[A-Za-z0-9._-]+)\t[0-9a-f]{64}\n")
if (
    not lines
    or lines[0] != header
    or any(pattern.fullmatch(line) is None for line in lines[1:])
    or len(set(lines[1:])) != len(lines[1:])
):
    raise RuntimeError("state contents changed or are malformed")
expected = (os.environ["STATE_RECORD"] + "\n").encode("ascii")
raise SystemExit(0 if expected in lines[1:] else 1)
PYEOF
    then
      continue
    else
      STATE_CHECK_STATUS=$?
      if [ "$STATE_CHECK_STATUS" != 1 ]; then
        echo "ckpt_sync: cannot validate completion state for $dir" >&2
        exit 2
      fi
    fi
    mtime=$(stat -c %Y "$dir") || {
      echo "ckpt_sync: cannot read checkpoint directory mtime: $dir" >&2
      scan_failed=1
      continue
    }
    [ $(( now - mtime )) -lt "$QUIESCENT_SECS" ] && continue
    echo "[$(date -u +%H:%M:%S)] uploading $dir"
    # Destination selection came from the submitted JSON above. Credentials
    # remain ambient and cannot switch this branch.
    upload_ok=0
    SYNC_DIR="$dir" SYNC_RUN_COMPONENT="$EXPECTED_RUN_COMPONENT" \
      SYNC_NAMESPACE="$DEBATE_LAUNCH_NAMESPACE" \
      SYNC_S3_ENV_FILE="$S3_ENV_FILE" "$PYBIN" - <<'PYEOF' && upload_ok=1
import hashlib
import json
import os
import pathlib
import secrets
import shlex
import stat

d = pathlib.Path(os.environ["SYNC_DIR"])
run_component = os.environ["SYNC_RUN_COMPONENT"]
namespace = os.environ["SYNC_NAMESPACE"]


def load_credential_file(path: pathlib.Path) -> None:
    """Load credential values only; never execute or import control variables."""
    allowed = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    }
    try:
        named_before = os.lstat(path)
    except FileNotFoundError:
        return
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            "S3 credential file must be euid-owned regular single-link mode 0600"
        ) from exc
    try:
        opened = os.fstat(fd)
        named_after = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino)
            != (named_before.st_dev, named_before.st_ino)
            or (opened.st_dev, opened.st_ino)
            != (named_after.st_dev, named_after.st_ino)
        ):
            raise RuntimeError(
                "S3 credential file must be euid-owned regular single-link mode 0600"
            )
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        text = b"".join(chunks).decode("utf-8", errors="strict")
    finally:
        os.close(fd)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            # Non-credential shell/control content is deliberately ignored,
            # never evaluated.
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens = list(lexer)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid quoting in S3 credential file at line {line_number}"
            ) from exc
        if len(tokens) != 1 or not tokens[0]:
            raise RuntimeError(
                f"invalid credential value in S3 credential file at line {line_number}"
            )
        os.environ[key] = tokens[0]


load_credential_file(pathlib.Path(os.environ["SYNC_S3_ENV_FILE"]))


def validated_regular_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Walk without following links; refuse anything but private regular files."""
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"refusing non-directory checkpoint root: {root}")
    files = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        current_stat = os.lstat(current_path)
        if not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError(f"refusing non-directory checkpoint entry: {current_path}")
        # os.walk identifies symlinked directories before yielding. Inspect
        # every name with lstat and fail before it can descend or an uploader
        # can follow the target.
        for name in dirnames:
            entry = current_path / name
            entry_stat = os.lstat(entry)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise RuntimeError(f"refusing symlink in checkpoint tree: {entry}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise RuntimeError(f"refusing non-directory checkpoint entry: {entry}")
        for name in filenames:
            entry = current_path / name
            entry_stat = os.lstat(entry)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise RuntimeError(f"refusing symlink in checkpoint tree: {entry}")
            if not stat.S_ISREG(entry_stat.st_mode):
                raise RuntimeError(f"refusing non-regular checkpoint entry: {entry}")
            if entry_stat.st_nlink != 1:
                raise RuntimeError(
                    f"refusing hard-linked checkpoint entry (nlink={entry_stat.st_nlink}): {entry}"
                )
            files.append(entry)
    return sorted(files)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# This preflight precedes either provider client. The separately pending source
# snapshot decision still governs mutations that race after this lstat walk.
files = validated_regular_files(d)
if not files:
    raise RuntimeError(f"refusing empty checkpoint directory: {d}")
reserved_relative = pathlib.PurePosixPath(".ckpt-sync-reservation-v1.json")
if any(f.relative_to(d).as_posix() == str(reserved_relative) for f in files):
    raise RuntimeError(
        "refusing checkpoint file with reserved relative path: "
        ".ckpt-sync-reservation-v1.json"
    )

if os.environ["DESTINATION_KIND"] == "bucket":
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, ParamValidationError

    s3 = boto3.client(
        "s3",
        region_name=os.environ["DESTINATION_REGION"],
        endpoint_url=os.environ["DESTINATION_ENDPOINT"],
        config=Config(retries={"total_max_attempts": 1, "mode": "standard"}),
    )

    def disable_s3_region_redirect_retries(client) -> None:
        """Remove only botocore's extra S3 region-redirect retry hook."""
        import botocore.utils

        redirector_types = tuple(
            redirector_type
            for name in ("S3RegionRedirector", "S3RegionRedirectorv2")
            if (redirector_type := getattr(botocore.utils, name, None)) is not None
        )
        if not redirector_types:
            raise RuntimeError(
                "botocore dependency shape has no known S3 region redirector"
            )
        try:
            emitter = client.meta.events._emitter
            handlers = list(emitter._handlers.prefix_search("needs-retry.s3"))
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "botocore dependency shape cannot enumerate S3 retry handlers"
            ) from exc

        def is_redirect_handler(handler) -> bool:
            owner = getattr(handler, "__self__", None)
            function = getattr(handler, "__func__", None)
            return (
                isinstance(owner, redirector_types)
                and getattr(function, "__name__", None) == "redirect_from_error"
                and owner.__class__.__module__ == "botocore.utils"
            )

        matches = [handler for handler in handlers if is_redirect_handler(handler)]
        if len(matches) != 1:
            raise RuntimeError(
                "botocore dependency shape must expose exactly one S3 region "
                f"redirect retry handler; observed {len(matches)}"
            )
        client.meta.events.unregister("needs-retry.s3", handler=matches[0])
        remaining = [
            handler
            for handler in emitter._handlers.prefix_search("needs-retry.s3")
            if is_redirect_handler(handler)
        ]
        if remaining:
            raise RuntimeError(
                "botocore refused to unregister its S3 region redirect retry handler"
            )

    disable_s3_region_redirect_retries(s3)

    def call_s3_once(operation_name: str, method, **kwargs):
        """Call one S3 operation with an absolute one-transport-send budget.

        The client is deliberately single-threaded. The temporary handler is
        operation-specific and removed before another operation may begin.
        """
        event_name = f"before-send.s3.{operation_name}"
        sends = 0

        def transport_budget_guard(request, **event_kwargs):
            del request, event_kwargs
            nonlocal sends
            sends += 1
            if sends > 1:
                raise RuntimeError(
                    f"refusing second transport attempt for S3 {operation_name}"
                )

        try:
            s3.meta.events.register_first(event_name, transport_budget_guard)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"cannot install transport budget for S3 {operation_name}"
            ) from exc
        try:
            return method(**kwargs)
        finally:
            cleanup_error = None
            try:
                s3.meta.events.unregister(
                    event_name, handler=transport_budget_guard
                )
            except (AttributeError, TypeError, ValueError) as exc:
                cleanup_error = exc
            try:
                remaining = list(
                    s3.meta.events._emitter._handlers.prefix_search(event_name)
                )
            except (AttributeError, TypeError) as exc:
                cleanup_error = cleanup_error or exc
                remaining = [transport_budget_guard]
            if any(handler is transport_budget_guard for handler in remaining):
                cleanup_error = cleanup_error or RuntimeError(
                    "temporary handler remains registered"
                )
            if cleanup_error is not None:
                raise RuntimeError(
                    f"cannot remove transport budget for S3 {operation_name}"
                ) from cleanup_error

    bucket = os.environ["DESTINATION_BUCKET"]

    def list_prefix_keys(
        prefix: str, *, expected_key_count: int, refuse_any: bool = False
    ) -> list[str]:
        keys = []
        continuation = None
        seen_tokens = set()
        seen_keys = set()
        while True:
            request = {"Bucket": bucket, "Prefix": prefix}
            if continuation is not None:
                request["ContinuationToken"] = continuation
            response = call_s3_once(
                "ListObjectsV2", s3.list_objects_v2, **request
            )
            page_keys = []
            for item in response.get("Contents", []):
                key = item.get("Key")
                if not isinstance(key, str) or not key.startswith(prefix):
                    raise RuntimeError(
                        f"unsafe S3 prefix-list response for {prefix!r}: {key!r}"
                    )
                if refuse_any:
                    raise RuntimeError(
                        f"refusing occupied S3 step prefix: {prefix}; observed {key}"
                    )
                if key in seen_keys:
                    raise RuntimeError(
                        f"unsafe nonprogressing S3 prefix-list page for {prefix!r}"
                    )
                seen_keys.add(key)
                page_keys.append(key)
            keys.extend(page_keys)
            if len(keys) > expected_key_count:
                raise RuntimeError(
                    f"refusing raced or incomplete S3 step prefix: {prefix}; "
                    "unsafe overfull S3 prefix-list response; "
                    f"expected at most {expected_key_count} key(s)"
                )
            if not response.get("IsTruncated", False):
                return keys
            next_continuation = response.get("NextContinuationToken")
            if (
                not isinstance(next_continuation, str)
                or not next_continuation
                or next_continuation == continuation
                or next_continuation in seen_tokens
                or not page_keys
            ):
                raise RuntimeError(
                    f"unsafe nonprogressing truncated S3 prefix-list response: {prefix}"
                )
            seen_tokens.add(next_continuation)
            continuation = next_continuation

    def destination_exists(key: str) -> bool:
        try:
            call_s3_once("HeadObject", s3.head_object, Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise
        return True

    def verify_remote_bytes(key: str, expected_size: int, expected_sha256: str) -> None:
        response = call_s3_once(
            "GetObject", s3.get_object, Bucket=bucket, Key=key
        )
        remote_size = response.get("ContentLength")
        metadata = response.get("Metadata") or {}
        metadata_sha256 = metadata.get("sha256")
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise RuntimeError(f"unverifiable S3 response body for {key}")
        digest = hashlib.sha256()
        observed_size = 0
        try:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                observed_size += len(chunk)
                digest.update(chunk)
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()
        observed_sha256 = digest.hexdigest()
        if (
            remote_size != expected_size
            or observed_size != expected_size
            or metadata_sha256 != expected_sha256
            or observed_sha256 != expected_sha256
        ):
            raise RuntimeError(
                f"refusing unverified S3 object after PUT: {key}; "
                "remote size/hash does not match the checkpoint manifest"
            )

    step_prefix = (
        f"{os.environ['DESTINATION_PREFIX']}/{run_component}/{namespace}/{d.name}/"
    )
    existing_keys = list_prefix_keys(
        step_prefix, expected_key_count=0, refuse_any=True
    )
    if existing_keys:
        raise RuntimeError(
            f"refusing occupied S3 step prefix: {step_prefix} "
            f"({len(existing_keys)} object(s))"
        )

    file_records = []
    for f in files:
        relative = f.relative_to(d)
        key = f"{step_prefix}{relative}"
        size = f.stat().st_size
        # LoRA adapters are the only synced artifacts: Verl receives
        # actor_rollout_ref.actor.checkpoint.save_lora_only=True whenever
        # lora_rank > 0 (infra/backend/verl.py), and adapters are ~100-200MB.
        # A file over the 5 GiB conditional single-PUT boundary violates that
        # invariant. Refuse it; multipart is intentionally not implemented.
        if size > 5 * 1024**3:
            raise RuntimeError(
                f"refusing unsafe S3 upload for {key}: {size} bytes exceeds "
                "the conditional single-PUT limit; no no-overwrite multipart "
                "boundary is implemented"
            )
        local_hash = sha256_file(f)
        file_records.append(
            {
                "path": relative.as_posix(),
                "key": key,
                "size": size,
                "sha256": local_hash,
                "source": f,
            }
        )

    manifest_document = [
        {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
        for item in file_records
    ]
    manifest_bytes = json.dumps(
        manifest_document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    claim_document = {
        "schema_version": 1,
        "run": run_component,
        "namespace": namespace,
        "step": d.name,
        "claim_nonce": secrets.token_hex(16),
        "checkpoint_manifest_sha256": manifest_sha256,
    }
    claim_core = json.dumps(
        claim_document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    claim_document["claim_digest_sha256"] = hashlib.sha256(claim_core).hexdigest()
    claim_bytes = (
        json.dumps(claim_document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    reservation_key = f"{step_prefix}.ckpt-sync-reservation-v1.json"
    reservation_sha256 = hashlib.sha256(claim_bytes).hexdigest()

    # This is the atomic step claim. Two writers can both observe an empty
    # prefix, but only one conditional create can install the reservation.
    # There is intentionally no adoption/resume path: on any later failure the
    # marker and partials remain immutable evidence and this script exits.
    try:
        call_s3_once(
            "PutObject",
            s3.put_object,
            Bucket=bucket,
            Key=reservation_key,
            Body=claim_bytes,
            ContentLength=len(claim_bytes),
            IfNoneMatch="*",
            Metadata={
                "sha256": reservation_sha256,
                "claim-digest-sha256": claim_document["claim_digest_sha256"],
            },
        )
    except ParamValidationError as exc:
        raise RuntimeError(
            "the installed S3 client cannot express If-None-Match; "
            "refusing a reservation that could overwrite"
        ) from exc
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
            raise
        raise RuntimeError(
            f"refusing concurrently reserved S3 step prefix: {step_prefix}"
        ) from exc
    verify_remote_bytes(
        reservation_key, len(claim_bytes), reservation_sha256
    )

    expected_keys = [reservation_key]
    for item in file_records:
        f = item["source"]
        key = item["key"]
        expected_keys.append(key)
        if destination_exists(key):
            raise RuntimeError(f"refusing occupied S3 destination: {key}")
        try:
            with f.open("rb") as source:
                call_s3_once(
                    "PutObject",
                    s3.put_object,
                    Bucket=bucket,
                    Key=key,
                    Body=source,
                    ContentLength=item["size"],
                    IfNoneMatch="*",
                    Metadata={
                        "sha256": item["sha256"],
                        "claim-digest-sha256": claim_document["claim_digest_sha256"],
                    },
                )
        except ParamValidationError as exc:
            raise RuntimeError(
                "the installed S3 client cannot express If-None-Match; "
                "refusing an upload that could overwrite"
            ) from exc
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
                raise
            raise RuntimeError(
                f"refusing concurrently occupied S3 destination: {key}"
            ) from exc
        verify_remote_bytes(key, item["size"], item["sha256"])
    final_keys = list_prefix_keys(
        step_prefix, expected_key_count=len(expected_keys)
    )
    if set(final_keys) != set(expected_keys) or len(final_keys) != len(expected_keys):
        raise RuntimeError(
            f"refusing raced or incomplete S3 step prefix: {step_prefix}; "
            f"expected {len(expected_keys)} object(s), observed {len(final_keys)}"
        )
print("ok")
PYEOF
    if [ "$upload_ok" = 1 ]; then
      if ! STATE_PATH="$STATE" STATE_FD="$STATE_RETAINED_FD" \
        STATE_RECORD="$STATE_RECORD" \
        EXPECTED_STATE_IDENTITY="$STATE_IDENTITY_DIGEST" \
        EXPECTED_STATE_DEVICE="$STATE_DEVICE" EXPECTED_STATE_INODE="$STATE_INODE" \
        EXPECTED_PARENT_DEVICE="$STATE_PARENT_DEVICE" \
        EXPECTED_PARENT_INODE="$STATE_PARENT_INODE" \
        "$PYBIN" - <<'PYEOF'
import os
import pathlib
import re
import stat

path = pathlib.Path(os.environ["STATE_PATH"])
fd = int(os.environ["STATE_FD"])
opened = os.fstat(fd)
named = os.lstat(path)
parent = os.lstat(path.parent)
expected_inode = (
    int(os.environ["EXPECTED_STATE_DEVICE"]),
    int(os.environ["EXPECTED_STATE_INODE"]),
)
expected_parent = (
    int(os.environ["EXPECTED_PARENT_DEVICE"]),
    int(os.environ["EXPECTED_PARENT_INODE"]),
)
if (
    not stat.S_ISREG(opened.st_mode)
    or opened.st_nlink != 1
    or opened.st_uid != os.geteuid()
    or stat.S_IMODE(opened.st_mode) != 0o600
    or (opened.st_dev, opened.st_ino) != expected_inode
    or (named.st_dev, named.st_ino) != expected_inode
    or (parent.st_dev, parent.st_ino) != expected_parent
    or parent.st_uid != os.geteuid()
    or stat.S_IMODE(parent.st_mode) != 0o700
):
    raise RuntimeError("state inode changed")
os.lseek(fd, 0, os.SEEK_SET)
content = b""
while True:
    chunk = os.read(fd, 1024 * 1024)
    if not chunk:
        break
    content += chunk
lines = content.splitlines(keepends=True)
header = f"identity-v1\t{os.environ['EXPECTED_STATE_IDENTITY']}\n".encode("ascii")
pattern = re.compile(rb"complete-v1\t(?:final|step-[A-Za-z0-9._-]+)\t[0-9a-f]{64}\n")
if (
    not lines
    or lines[0] != header
    or any(pattern.fullmatch(line) is None for line in lines[1:])
    or len(set(lines[1:])) != len(lines[1:])
):
    raise RuntimeError("state contents changed or are malformed")
record = (os.environ["STATE_RECORD"] + "\n").encode("ascii")
if record in lines[1:]:
    raise RuntimeError("completion record unexpectedly already exists")
os.lseek(fd, 0, os.SEEK_END)
offset = 0
while offset < len(record):
    written = os.write(fd, record[offset:])
    if written <= 0:
        raise RuntimeError("short completion-state write")
    offset += written
os.fsync(fd)
os.lseek(fd, 0, os.SEEK_SET)
verified = b""
while True:
    chunk = os.read(fd, 1024 * 1024)
    if not chunk:
        break
    verified += chunk
if verified != content + record:
    raise RuntimeError("completion-state reread verification failed")
PYEOF
      then
        echo "[$(date -u +%H:%M:%S)] checkpoint uploaded and verified but durable state append FAILED for $dir; refusing retry/adoption" >&2
        exit 1
      fi
    else
      echo "[$(date -u +%H:%M:%S)] bucket upload FAILED for $dir; one-shot reservation semantics forbid retry/adoption — leaving any marker/partials as evidence and exiting" >&2
      exit 1
    fi
  done
  if [ "$SYNC_ONCE" = 1 ]; then
    [ "$scan_failed" = 0 ] && exit 0
    exit 1
  fi
  sleep "$INTERVAL"
done
