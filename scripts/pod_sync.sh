#!/usr/bin/env bash
# Push the repo and the built datasets to a pod, and PROVE both arrived.
#
#   bash scripts/pod_sync.sh <ip> <port> [--with-data]
#
# Exists because transfer tools lie in two different ways we have been bitten by:
#
#   scp reported success and moved nothing -- it failed host-key verification
#   while the surrounding command's exit status stayed 0, so a config file that
#   never landed looked identical to one that did. Caught only because the
#   sha256 was compared afterwards.
#
#   A wheel arrived at 57,446,400 of 62,109,228 bytes inside a command that hit
#   its timeout. Truncated, not absent -- so it installed, and failed later at
#   import with an error that pointed at the wrong thing entirely.
#
# So: rsync (which uses ssh with our flags, unlike scp here), then sha256 both
# ends and diff them. Exit status of the copy is NOT evidence.
#
# Exit codes:
#   2  bad arguments, or a source file is missing locally
#   3  repo sync mismatched after transfer
#   4  data sync mismatched after transfer
set -euo pipefail

IP="${1:?usage: pod_sync.sh <ip> <port> [--with-data]}"
PORT="${2:?usage: pod_sync.sh <ip> <port> [--with-data]}"
WITH_DATA="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $PORT"
RSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT"
REMOTE="root@$IP"

# Validate the complete reproducibility bundle BEFORE contacting the pod. The
# trainer reads train.jsonl + paired_test.jsonl, while the source/eval files
# and all three manifests are what prove how those runtime artifacts were
# derived. Syncing only the JSONLs makes a run executable but not auditable.
REQUIRED_CODECONTESTS_DATA=(
  train.jsonl
  test.jsonl
  manifest.json
  cco_eval.jsonl
  cco_eval.manifest.json
  paired_test.jsonl
  paired_test.manifest.json
)
if [ "$WITH_DATA" = "--with-data" ]; then
  for f in "${REQUIRED_CODECONTESTS_DATA[@]}"; do
    [ -f "$REPO_ROOT/data/codecontests/$f" ] || {
      echo "FATAL: data/codecontests/$f missing locally" >&2
      exit 2
    }
  done
  if ! python3 - "$REPO_ROOT/data/codecontests" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])


def load(name):
    path = root / name
    try:
        body = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"FATAL: cannot parse data/codecontests/{name}: {exc}") from exc
    if not isinstance(body, dict):
        raise SystemExit(f"FATAL: data/codecontests/{name} is not a JSON object")
    return body


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name, metadata, size_key):
    path = root / name
    declared_name = metadata.get("file")
    if declared_name is not None and declared_name != name:
        raise SystemExit(
            f"FATAL: data/codecontests/{name} manifest names {declared_name!r}"
        )
    try:
        expected_size = int(metadata[size_key])
        expected_sha = str(metadata["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"FATAL: data/codecontests/{name} manifest lacks valid {size_key}/sha256"
        ) from exc
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"FATAL: data/codecontests/{name} size mismatch: "
            f"manifest={expected_size}, local={actual_size}"
        )
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise SystemExit(
            f"FATAL: data/codecontests/{name} sha256 mismatch: "
            f"manifest={expected_sha}, local={actual_sha}"
        )
    return actual_sha


base = load("manifest.json")
cco = load("cco_eval.manifest.json")
paired = load("paired_test.manifest.json")
try:
    outputs = base["outputs"]
    paired_output = paired["output"]
except (KeyError, TypeError) as exc:
    raise SystemExit("FATAL: CodeContests manifests lack outputs/output metadata") from exc

train_sha = verify("train.jsonl", outputs["train"], "size_bytes")
test_sha = verify("test.jsonl", outputs["test"], "size_bytes")
cco_sha = verify("cco_eval.jsonl", cco, "bytes")
paired_sha = verify("paired_test.jsonl", paired_output, "size_bytes")

# The paired artifact records the exact two staging inputs it joined. Checking
# those links catches a self-consistent but mismatched set of independently
# valid files/manifests.
try:
    paired_gdm_sha = str(paired["sources"]["gdm"]["sha256"])
    paired_cco_sha = str(paired["sources"]["cco"]["sha256"])
except (KeyError, TypeError) as exc:
    raise SystemExit("FATAL: paired_test manifest lacks source sha256 links") from exc
if paired_gdm_sha != test_sha or paired_cco_sha != cco_sha:
    raise SystemExit(
        "FATAL: paired_test manifest source hashes do not match "
        "test.jsonl + cco_eval.jsonl"
    )

print(
    "== local dataset manifests verified: "
    f"train={train_sha[:12]} test={test_sha[:12]} "
    f"cco={cco_sha[:12]} paired={paired_sha[:12]} =="
)
PYEOF
  then
    exit 2
  fi
fi

# GNU sha256sum (pod) and BSD shasum (mac) differ in path prefixes and default
# sort order, so an aggregate hash of the two disagrees even when every file
# matches -- that exact false alarm cost a round of panic. Compare per file.
#
# Exclusions live in `find`, not in a grep afterwards. Two attempts at grepping
# failed for two different reasons -- an unparenthesised `-name a -o -name b
# -exec` group binds -exec only to the last branch (empty list on macOS), and
# then `^\.venv/` anchored against "<hash>  <path>" matches the HASH column, not
# the path. Both "failed" a sync that had actually worked. -not -path cannot
# make either mistake.
PRUNE='-not -path "./.git/*" -not -path "./.venv/*" -not -path "./data/*" -not -path "./.claude/*" -not -path "*/__pycache__/*"'
CODE='\( -name "*.py" -o -name "*.yaml" -o -name "*.sh" \)'
remote_sums() { $SSH "$REMOTE" "cd $1 && find . -type f $PRUNE $CODE -exec sha256sum {} \; | sed 's#  \./#  #' | LC_ALL=C sort -k2"; }
local_sums()  { (cd "$1" && eval "find . -type f $PRUNE $CODE -exec shasum -a 256 {} \;" | sed 's#  \./#  #' | LC_ALL=C sort -k2); }

echo "== repo -> $REMOTE:/root/debate"
rsync -a --delete -e "$RSH" \
  --exclude '.git' --exclude '.claude' --exclude 'data' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'outputs' --exclude 'docent' --exclude 'rollouts' --exclude 'wandb' \
  "$REPO_ROOT/" "$REMOTE:/root/debate/"

echo "== verifying repo"
diff <(local_sums "$REPO_ROOT") <(remote_sums /root/debate) \
  > /tmp/pod_sync_repo.diff 2>&1 && echo "   repo OK" || {
    echo "FATAL: repo mismatch after sync -- first lines:" >&2; head -20 /tmp/pod_sync_repo.diff >&2; exit 3; }

if [ "$WITH_DATA" = "--with-data" ]; then
  echo "== datasets -> $REMOTE:/root/debate/data/codecontests"
  $SSH "$REMOTE" "mkdir -p /root/debate/data/codecontests"
  # macOS ships openrsync/rsync 2.6.9, which has --progress but not the newer
  # aggregate-info progress flag. Keep this operator path portable: verification
  # below is the correctness boundary, not the progress display mode.
  rsync -a --progress -e "$RSH" \
    "$REPO_ROOT/data/codecontests/" "$REMOTE:/root/debate/data/codecontests/"

  echo "== verifying datasets"
  diff <(cd "$REPO_ROOT/data/codecontests" && find . -type f -exec shasum -a 256 {} \; | sed 's#  \./#  #' | LC_ALL=C sort -k2) \
       <($SSH "$REMOTE" "cd /root/debate/data/codecontests && find . -type f -exec sha256sum {} \; | sed 's#  \./#  #' | LC_ALL=C sort -k2") \
    > /tmp/pod_sync_data.diff 2>&1 && echo "   datasets OK" || {
      echo "FATAL: dataset mismatch after sync:" >&2; head -20 /tmp/pod_sync_data.diff >&2; exit 4; }
fi

echo "SYNC VERIFIED"
