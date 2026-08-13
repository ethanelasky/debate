# RunPod job lifecycle

Use `scripts/runpod_job_supervisor.py` as the foreground owner of every paid
training run. It accepts only an exact `runpod-safe`-owned Pod ID and performs:

1. ownership audit;
2. bounded trainer command;
3. explicit Pod-side trainer cancellation and proof that its exact private
   process group has stopped;
4. tunnel-supervisor shutdown;
5. bounded evacuation and checksum verification (plus a bounded emergency
   evacuation attempt if either fails);
6. `/opt/homebrew/bin/runpod-safe delete <exact-pod-id>` with bounded retries;
7. final ownership audit and an independent, bounded endpoint-reachability
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

The cleanup deadline reserves the worst-case configured time for trainer
cancellation, every delete attempt, retry delay, final audit, reachability
probe, and local process-group termination before allocating any time to
evacuation. This keeps a hung copy or checksum command from consuming the
deletion window. If exact remote cancellation cannot be proven, evacuation is
still attempted and the Pod is still deleted, but the artifacts are reported
uncertain because they may have changed during the copy.

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

## Reconnecting remote trainer owner

Do not use a raw SSH command as the lifecycle supervisor's trainer command. If
the host network drops, SSH can return 255 while the Pod-side trainer remains
healthy; interpreting that as trainer exit would evacuate and delete a live
run. Use `scripts/runpod_remote_job.py` instead.

Stage one foreground launch script on the Pod, compute its SHA-256 locally,
and create a mode-0600 config like this (the state directory's parent must
already exist on the Pod):

```json
{
  "format": "palaestra.runpod-remote-job.v1",
  "job_id": "cc-cap1024-RUN_ID",
  "ssh_argv": [
    "/usr/bin/ssh", "-T",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=/ABSOLUTE/RUN/known_hosts",
    "-o", "ConnectTimeout=8",
    "-o", "ConnectionAttempts=1",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "RequestTTY=no",
    "-i", "/ABSOLUTE/RUN/trainer_ed25519",
    "-p", "RUNPOD_SSH_PORT",
    "root@RUNPOD_HOST"
  ],
  "remote_state_dir": "/workspace/run-control/cc-cap1024-RUN_ID",
  "remote_launch_script": "/root/run-control/cc-cap1024-RUN_ID/launch.sh",
  "launch_script_sha256": "LOWERCASE_SHA256",
  "poll_interval_seconds": 10,
  "outage_timeout_seconds": 600,
  "ssh_command_timeout_seconds": 30,
  "initialization_timeout_seconds": 30,
  "cancel_timeout_seconds": 10
}
```

The normal lifecycle trainer argv is:

```text
["/usr/bin/python3","/absolute/repo/scripts/runpod_remote_job.py","--config","/absolute/run-control/remote-job.json","--mode","start-and-monitor"]
```

The lifecycle trainer-cancel argv uses the same config and `--mode cancel`.
Configure the supervisor's `--trainer-cancel-timeout-seconds` longer than the
remote config's `cancel_timeout_seconds` plus its bounded SSH-command timeout.

The remote state directory is the atomic single-start marker. An uncertain
start response is retried safely, but an existing marker is never launched a
second time. The detached wrapper records its exact PID/start time, streams the
launcher's output to `launch.log`, and atomically writes `rc`. Polling verifies
the staged launch-script digest every time. A continuous SSH outage is tolerated
up to `outage_timeout_seconds`; a dead wrapper without `rc` fails loudly. The
supervisor then invokes exact cancellation, which inventories the recorded
private session and uses Linux pidfds to stop surviving Bash/trainer descendants
without signalling a reused numeric PID. Cancellation remains available if the
staged script path was replaced or removed after launch.

`--mode start-only` is available for a separately supervised setup phase, and
`--mode monitor` never starts missing work. The latter can be used as the
lifecycle trainer command only when the exact atomic state was already created
and the server TTL still covers the lifecycle supervisor's full trainer and
cleanup budgets. The launch script itself must remain foreground until training
and its final artifact sync are complete.

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
.venv/bin/python -m pytest -q \
  tests/test_runpod_job_supervisor.py tests/test_runpod_remote_job.py
```

The tests use fake commands and a fake safe wrapper. They create, stop, and
delete no real Pod.
