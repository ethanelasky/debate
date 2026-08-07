# RunPod launch contract

Use `scripts/pod_create.sh` for every fresh Pod created for this repository.
Do not use bare `runpodctl create`, the RunPod SDK, or the older ad-hoc launch
commands.

The canonical launcher provides these guarantees:

- creation and deletion go through `/opt/homebrew/bin/runpod-safe`;
- every Pod has a provider-side termination deadline;
- the exact Pod ID is recorded in an ownership receipt;
- the pinned public template is resolved and its image is verified before the
  paid create call;
- template and custom-image launch modes cannot be combined;
- a Pod is ready only after an SSH command actually executes;
- a Pod that misses the readiness deadline is deleted automatically; and
- capacity is probed read-only before exactly one datacenter is selected.

## Commands

Inspect the exact provider request without creating anything:

```bash
DRY_RUN=1 TTL_MINUTES=30 bash scripts/pod_create.sh my-run
```

Create the default one-H200 Pod:

```bash
TTL_MINUTES=720 bash scripts/pod_create.sh my-run
```

Request a multi-GPU Pod explicitly:

```bash
GPU_COUNT=4 TTL_MINUTES=720 bash scripts/pod_create.sh my-run
```

Pin a datacenter with `DC_ID`. Attach a network volume with `VOLUME_ID`; the
launcher then requires the Pod to use that volume's datacenter and refuses the
create if the selected GPU has no reported stock there. Network volumes cannot
move between datacenters.

On success the launcher prints the Pod ID, hourly price, server expiry, SSH
endpoint, and the matching `pod_sync.sh` command. When the work and artifact
checks are complete, delete only the owned ID and audit the account:

```bash
/opt/homebrew/bin/runpod-safe delete POD_ID
/opt/homebrew/bin/runpod-safe audit
```

## Local safety-wrapper verification

The installed `runpod-safe` must support `--template-id` and must reject a
request containing both `--template-id` and `--image`. Its offline lifecycle
suite creates no paid resources:

```bash
/opt/homebrew/bin/python3 \
  ~/.local/share/runpod-safety/test_runpod_safe.py -q
```

Do not use `runtime.ports` or `desiredStatus=RUNNING` as readiness. The launcher
uses `runpodctl ssh info` to discover an endpoint and then runs `ssh ... true`;
only the successful remote command makes the launch ready.
