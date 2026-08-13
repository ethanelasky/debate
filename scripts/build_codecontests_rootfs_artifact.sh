#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_SHA256="83e694da5d1e0b94700da2a195d760527ce609ea631f7302ec930666bae136d0"
readonly EXPECTED_RUNTIME="3.12.3|sympy=1.14.0|mpmath=1.3.0"

usage() {
  echo "usage: $0 [--candidate] [--force] ROOTFS OUTPUT_TAR" >&2
  exit 2
}

force=0
candidate=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --candidate) candidate=1; shift ;;
    --force) force=1; shift ;;
    --*) usage ;;
    *) break ;;
  esac
done
[[ "$#" -eq 2 ]] || usage
rootfs=$1
output=$2

[[ "${EUID}" -eq 0 ]] || {
  echo "rootfs artifact construction must run as root" >&2
  exit 1
}
[[ -d "$rootfs" && ! -L "$rootfs" ]] || {
  echo "rootfs must be a non-symlink directory" >&2
  exit 1
}
[[ "$rootfs" = /* && "$(realpath -e -- "$rootfs")" == "$rootfs" ]] || {
  echo "rootfs must be an absolute canonical directory" >&2
  exit 1
}
[[ "$output" = /* ]] || {
  echo "output must be an absolute path" >&2
  exit 1
}
case "$output" in
  "$rootfs"|"$rootfs"/*)
    echo "output must live outside the rootfs" >&2
    exit 1
    ;;
esac
if [[ "$candidate" -eq 1 && "$(basename -- "$output")" != *.candidate.tar ]]; then
  echo "candidate output must end in .candidate.tar" >&2
  exit 1
fi
if [[ -e "$output" || -L "$output" ]]; then
  [[ "$force" -eq 1 ]] || {
    echo "output exists; pass --force to replace it" >&2
    exit 1
  }
fi

root_owner=$(stat -c '%u:%g' -- "$rootfs")
[[ "$root_owner" == "0:0" ]] || {
  echo "rootfs must be owned by root:root" >&2
  exit 1
}
nonroot_entry=$(find "$rootfs" -xdev \( ! -user root -o ! -group root \) -print -quit)
[[ -z "$nonroot_entry" ]] || {
  echo "rootfs contains a non-root-owned entry: $nonroot_entry" >&2
  exit 1
}
privileged_entry=$(find "$rootfs" -xdev -perm /6000 -print -quit)
[[ -z "$privileged_entry" ]] || {
  echo "rootfs contains a setuid/setgid entry: $privileged_entry" >&2
  exit 1
}

# Execute only the already-staged, credential-free rootfs interpreter.  -I/-B
# proves the packages are on its isolated system path without creating pyc.
runtime=$(
  /usr/sbin/chroot "$rootfs" /usr/bin/python3 -I -B -c \
    'import platform,mpmath,sympy; print(f"{platform.python_version()}|sympy={sympy.__version__}|mpmath={mpmath.__version__}")'
)
[[ "$runtime" == "$EXPECTED_RUNTIME" ]] || {
  echo "rootfs runtime mismatch: $runtime" >&2
  exit 1
}

output_parent=$(dirname -- "$output")
output_name=$(basename -- "$output")
install -d -m 0700 -o root -g root "$output_parent"
temporary=$(mktemp "$output_parent/.${output_name}.XXXXXX.tmp")
cleanup() {
  rm -f -- "$temporary"
}
trap cleanup EXIT

tar \
  --sort=name \
  --mtime=@0 \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --format=posix \
  --pax-option=delete=atime,delete=ctime \
  --atime-preserve=system \
  -cf "$temporary" \
  -C "$rootfs" \
  .
actual_sha256=$(sha256sum "$temporary" | awk '{print $1}')
if [[ "$candidate" -eq 0 && "$EXPECTED_SHA256" =~ ^0+$ ]]; then
  echo "production rootfs pin is unset; build and review --candidate first" >&2
  exit 1
fi
[[ "$candidate" -eq 1 || "$actual_sha256" == "$EXPECTED_SHA256" ]] || {
  echo "normalized rootfs artifact digest mismatch: $actual_sha256" >&2
  exit 1
}
chmod 0600 "$temporary"
sync -f "$temporary"
if [[ "$force" -eq 1 ]]; then
  mv -f -- "$temporary" "$output"
else
  ln -- "$temporary" "$output"
  rm -f -- "$temporary"
fi
sync -f "$output_parent"
trap - EXIT
if [[ "$candidate" -eq 1 ]]; then
  echo "candidate_rootfs_artifact=$output sha256=$actual_sha256 review_and_pin_required=1"
else
  echo "rootfs_artifact=$output sha256=$actual_sha256"
fi
