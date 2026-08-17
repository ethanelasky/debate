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
  for f in \
    train.jsonl test.jsonl cco_eval.jsonl paired_test.jsonl \
    manifest.json cco_eval.manifest.json paired_test.manifest.json; do
    [ -f "$REPO_ROOT/data/codecontests/$f" ] || { echo "FATAL: data/codecontests/$f missing locally" >&2; exit 2; }
  done
  echo "== datasets -> $REMOTE:/root/debate/data/codecontests"
  $SSH "$REMOTE" "mkdir -p /root/debate/data/codecontests"
  # macOS ships openrsync with the 2.6.9-compatible interface, which does not
  # implement GNU rsync's --info=progress2. Plain --progress is supported by
  # both and affects display only; the checksum diff below remains the gate.
  rsync -a --progress -e "$RSH" \
    "$REPO_ROOT/data/codecontests/" "$REMOTE:/root/debate/data/codecontests/"

  echo "== verifying datasets"
  diff <(cd "$REPO_ROOT/data/codecontests" && find . -type f -exec shasum -a 256 {} \; | sed 's#  \./#  #' | LC_ALL=C sort -k2) \
       <($SSH "$REMOTE" "cd /root/debate/data/codecontests && find . -type f -exec sha256sum {} \; | sed 's#  \./#  #' | LC_ALL=C sort -k2") \
    > /tmp/pod_sync_data.diff 2>&1 && echo "   datasets OK" || {
      echo "FATAL: dataset mismatch after sync:" >&2; head -20 /tmp/pod_sync_data.diff >&2; exit 4; }
fi

echo "SYNC VERIFIED"
