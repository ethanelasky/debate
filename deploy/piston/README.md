# Self-hosted Piston for CodeContests

This runs Piston on a dedicated **amd64 Linux** judge VM. The API binds only
to the VM's loopback interface; untrusted programs have networking disabled.
Piston owns process isolation and execution, while the trainer keeps expected
outputs and decides the verdict.

This deployment defines executor protocol `codecontests-piston-v1`, which is
recorded in each experiment's protocol identity. Version 1 means the exact
base-image digest and Python archive checksum below, byte-exact stdin, a 3 MiB
request ceiling, 1 MiB output/file ceilings, 90-second wall/CPU ceilings,
4 GiB per-run memory, 64 processes/files, four server jobs, and candidate
networking disabled. Any reward-affecting change to that list requires a new
protocol ID in `infra/envs/tasks/piston.py`; otherwise resume safeguards cannot
distinguish the two grading protocols.

The base image is the official `ghcr.io/engineer-man/piston` amd64 image pinned
to immutable manifest digest
`sha256:2f66b7456189c4d713aa986d98eccd0b6ee16d26c7ec5f21b30e942756fd127a`.
That digest's config identifies `linux/amd64` and was created 2025-02-08. The
derived image makes three narrow, fail-closed upstream edits: JSON request bodies
may be up to 3 MiB, enough for CodeContests' roughly 500 KiB cases, and stdin
is passed byte-for-byte instead of silently gaining a trailing newline or
being truncated when Node still has buffered bytes. The build deliberately
fails if any reviewed source shape changes.

## Bring-up

The host needs Docker Engine, Docker Compose v2, unified cgroup v2 (with no
cgroup v1 mounts), `curl`, `jq`, `sha256sum`, and GNU `timeout`. Then run:

```bash
cd deploy/piston
./bootstrap.sh
```

Bootstrap builds and starts the API, installs exactly Python 3.12.0 into
`data/piston/packages`, and requires both a valid `GET /` response and an exact
Python 3.12.0 entry from `GET /api/v2/runtimes`. It independently verifies the
persisted runtime archive against SHA-256
`abc40b3231fc7e713799da2cd79844545c72b3904a4d2ffcc28c4d133ed21d0b`.
Finally it executes real isolated Python processes with 600 KiB of unterminated
stdin: one must read the exact length and final byte, while one exits cleanly
without reading. The API must not restart between probes. Broken
isolate/cgroups, a stock newline-mutating Piston, buffered-input truncation,
and child-pipe EPIPE crashes therefore fail bootstrap rather than appearing
ready.
The runtime directory persists across container replacement. Do not delete it
when updating the image. On a catchable bootstrap failure, the script attempts
to stop the service and checks whether it is still running; it prints an
explicit operator-cleanup error if stopping fails. No shell trap can handle
host power loss or `SIGKILL`, and stopping Docker does not stop VM billing.
The image build/start command itself has a 20-minute timeout; the runtime
package download separately has a 20-minute HTTP deadline. Failure cleanup
bounds Docker stop at 30 seconds and its postcondition check at 10 seconds, so
a wedged daemon yields an explicit operator error instead of hanging the shell.
Any paid judge VM therefore also needs a provider-side termination deadline
and explicit teardown. For RunPod, create it only with
`/opt/homebrew/bin/runpod-safe create` and complete the required artifact,
delete, and audit workflow.

The server-side ceilings are 90 seconds wall and CPU time and 4 GiB memory per
run. Real artifact outputs top out just below 500 KiB, so output and
candidate-created files are capped at 1 MiB; the aggregate isolate-box
filesystem is a disposable 512 MiB tmpfs. The service admits four concurrent
jobs, with 64 processes and 64 open files per job. `/tmp` is also an executable
tmpfs because Piston's isolate sandboxes execute there.

Capacity must cover the limits, not average observed usage: four jobs at 4 GiB
plus the 512 MiB isolate tmpfs, API/container overhead, and the host OS require
a roughly 20 GiB-or-larger production judge. An 8 GiB VM is suitable only for
controlled smoke inputs; it is not safe for adversarial four-way saturation.

Set `MAX_CONCURRENT_PISTON_VERIFIERS=4` on a single trainer (the code default).
If multiple trainers share this service, their configured concurrency totals
must stay at or below four. Piston's own excess-job queue is not a safe
backpressure mechanism: disconnected clients do not cancel queued jobs.

Check readiness later with:

```bash
cd deploy/piston
./bootstrap.sh health
docker compose ps
```

## Trainer-side production preflight

The bootstrap probe checks Piston directly. Before creating any paid GPU pod,
also exercise Piston through the exact CodeContests supervisor and adapter that
training uses. Keep an SSH loopback forward open, then run from the repository
root:

```bash
MAX_CONCURRENT_PISTON_VERIFIERS=4 \
  .venv/bin/python scripts/smoke_piston_verifier.py \
  --url http://127.0.0.1:2000 --runtime 3.12.0
```

Do not create paid GPU compute unless this prints `PISTON PREFLIGHT PASSED`.
The preflight checks byte-exact empty, unterminated, LF, and CRLF stdin; exit
codes 0, 1, and 120; timeout and output-limit classification; 600 KiB requests
both fully consumed and ignored by an early-exiting child; and an
effective-width group of timeout leaders followed by 28 successful jobs. It
rejects an effective concurrency outside one through the
protocol's four-job service capacity, and saturates exactly the configured
number of trainer-side slots. This prevents one preflight from accidentally
exercising Piston's hidden request queue. Combined capacity across separate
trainers remains an operator responsibility.

After a GPU pod exists, establish the same SSH loopback forward inside that
pod and run the same command there **before** model loading or trainer backend
bring-up. Use the concurrency assigned to that trainer. For two simultaneous
trainers sharing this four-job service, preflight and launch each with:

```bash
MAX_CONCURRENT_PISTON_VERIFIERS=2 \
  /workspace/envs/verl-sm90/bin/python scripts/smoke_piston_verifier.py \
  --url http://127.0.0.1:2000 --runtime 3.12.0
```

A pass from the operator machine does not prove the pod's own tunnel, routing,
environment, or checked-out verifier code works. The combined concurrency of
all trainers and preflights sharing one service must never exceed four.

## Production transport

Do not publish port 2000 or put this API directly on the Internet: its package
management routes are administrative. From the trusted trainer host, use a
direct SSH local forward to the judge VM:

```bash
ssh -NT -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:2000:127.0.0.1:2000 judge@JUDGE_HOST
```

Configure the trainer's Piston URL as `http://127.0.0.1:2000`. Keep the SSH
account and tunnel available only to the operator/trainer; the tunnel carries
the whole loopback API, including administrative routes.

Select it per CodeContests dataset (local execution remains the default):

```yaml
dataset:
  verifier: piston
  piston_url: http://127.0.0.1:2000
  piston_python_version: 3.12.0
```

Use the same Piston service and these same settings for every arm in a token-cap
comparison. Changing verifier backends between the 1024 and 2048 arms changes
the reward protocol as well as the generation cap.

To stop execution without deleting the installed runtime:

```bash
cd deploy/piston
docker compose stop
```

Before changing the base digest, independently resolve the official manifest
for `linux/amd64`, inspect its config architecture, and confirm the fail-closed
body-parser patch still applies. Re-run bootstrap and the production verifier
smoke after every digest or runtime change.
