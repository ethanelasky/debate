# RunPod job lifecycle

Use `scripts/runpod_job_supervisor.py` as the foreground owner of every paid
training run. It accepts only an exact `runpod-safe`-owned Pod ID and performs:

1. ownership audit;
2. bounded trainer command;
3. tunnel-supervisor shutdown;
4. bounded evacuation and checksum verification (plus a bounded emergency
   evacuation attempt if either fails);
5. `/opt/homebrew/bin/runpod-safe delete <exact-pod-id>` with bounded retries;
6. final ownership audit and an independent, bounded endpoint-reachability
   probe.

All command values are JSON argv arrays, not shell strings. Every executable
must be an absolute path. The command records a mode-0600 JSONL event log and
exports the exact ID as `RUNPOD_POD_ID` to each work command. A reachability
probe exits zero only when the supposedly deleted target is still reachable;
that contradiction makes the lifecycle fail loudly even if the provider audit
reports it absent. A probe timeout or launch failure is also inconclusive and
fails loudly; only a completed nonzero probe plus an audit that omits the exact
ID proves absence. The supervisor never falls back to untracked deletion.

Preflight parses the owned allocation's server `expires_at` and `expired`
fields and refuses to start training unless the remaining provider-enforced
lease covers the full trainer timeout, the full cleanup deadline, and a
30-second host/provider clock-skew margin. Missing, timezone-naive, malformed,
expired, or undersized deadlines are launch blockers.

The cleanup deadline reserves the worst-case configured time for every delete
attempt, retry delay, final audit, reachability probe, and local process-group
termination before allocating any time to evacuation. This keeps a hung copy or
checksum command from consuming the deletion window.

For a launchd-managed verifier tunnel, the transport-cleanup argv is:

```text
["/usr/bin/python3","/absolute/repo/scripts/install_codecontests_tunnel_supervisor.py","--remove","--mode","launchd","--label","<exact-run-label>","--config","<absolute-run-config>","--ready-timeout-seconds","30"]
```

The evacuation command must copy material logs, transcripts, adapters, and
checkpoints off the Pod. The verification command must independently validate
the copied manifest/checksums. The emergency command should attempt a smaller
last-chance capture of logs, return code, and run metadata. The reachability
probe should be a non-interactive, strict-host-key SSH command (for example,
`ssh ... true`); its ordinary connection failure is the expected post-delete
result.

Lifecycle-specific exit codes are `70` for failed ownership/preflight, `71` for
unverified artifacts after deletion, `72` when deletion is not independently
proven, and `73` when transport cleanup fails. Otherwise the supervisor returns
the trainer status (including `124` for a trainer timeout). Any nonzero status
requires operator review of the JSONL event log.

Set the trainer timeout and the server-side Pod TTL from the run estimate plus
setup, evacuation, and debugging margin. Keep the cleanup deadline shorter than
that remaining margin. The host supervisor must itself be launched from an
independent terminal, `launchd`, or another durable local process owner; if the
host loses power or the supervisor process is killed, only RunPod's
server-enforced TTL remains guaranteed. The installed five-minute
`runpod-safe reap` LaunchAgent is additional deadline defense, not immediate
trainer-exit cleanup.

## Pod-side backup

Do not set `POD_IDLE_STOP=0`. Export the exact `RUNPOD_POD_ID` and let
`scripts/pod_run.sh` arm `scripts/pod_idle_stop.sh` before training. The prior
cap-1024 and cap-2048 launchers explicitly set `POD_IDLE_STOP=0`, bypassing the
repository's guard and leaving cleanup entirely to ad-hoc host monitoring.

The idle watchdog must ignore non-interactive SSH transport sessions (the
reverse tunnel is `ssh -N -T` and appears server-side as `@notty`) while still
treating interactive `@pts/N` sessions as operator activity. Otherwise a
healthy permanent reverse tunnel makes the Pod look busy forever and defeats
idle cleanup. This distinction is an integration requirement for the next
launch; test the pod-side `--once` verdict with the tunnel up before starting
paid training.

## Offline tests

```sh
.venv/bin/python -m pytest -q tests/test_runpod_job_supervisor.py
```

The tests use fake commands and a fake safe wrapper. They create, stop, and
delete no real Pod.
