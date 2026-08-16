# `runpod-safe` replacement seam specification

**Status: DRAFT — NOT APPROVED, NOT IMPLEMENTED.**

This is the standalone proposal required by the amended RunPod launch-seam and
crash-backstop decisions in `docs/job_scheduler.md:647-654`. It does not
authorize implementation, installation, paid creation, resume, termination,
destruction, or auto-destroy. No RunPod `CREATE` path is authorized today.

This document specifies one narrow provider-boundary executable. The scheduler
remains the only reconciler and the only SQLite writer. Approval of the
scheduler plan does not imply approval of this replacement.

Labels used below:

- **CONFIRMED FACT**: directly observed in the repository, installed binary,
  retained project notes, or a cited primary RunPod source.
- **PROPOSED CONTRACT**: a new interface or invariant requiring approval.
- **OPEN DECISION**: a choice Ethan must explicitly make; none is selected here.

## 1. Provenance and current blockers

### 1.1 What survives

**CONFIRMED FACT.** The expected executable and state trees are absent from
this Mac: `/opt/homebrew/bin/runpod-safe`,
`~/.local/share/runpod-safety/`, and `~/.local/state/runpod-safety/` do not
exist. No accessible source package or install recipe has been found in the
local repositories and locations searched so far. This is not proof that no
copy exists: the MBA, Frank's machine, and unmounted/offline backups remain
unchecked.

**CONFIRMED FACT.** The retained vault notes preserve behavioral provenance,
not source. They describe a schema-v2 wrapper with a 720-minute ceiling, a
state lock spanning create, accepted-result persistence before ownership
publication, exact allocation comparison, owner-private no-follow receipts,
durable deletion tombstones, and strict audit categories. They also retain:

- source SHA-256
  `00fc3e1fa6f1551cac3552091c019b000a0f4f6bcb1b568858f32037b42bffe0`;
- test SHA-256
  `7843dd6a05e2362b1f513e3aefe384a419a7aeed054a2910445c7ff36fa39eb6`;
- installed-authority digest
  `6b863825a08c12e51a3a55bbc46bcefc938120232496a9b58cdc5a55bf1038fa`.

Source: the vault's
`2026-07-24-codecontests-rlvr-32b-canary-launch.md:1304-1346`.
Those digests are useful for authenticating a recovered copy; they are not
enough to recreate its contents.

**CONFIRMED FACT.** The repository's current canonical launcher hard-fails
when the old executable is absent (`scripts/pod_create.sh:25,38`), calls its
lost `create --ttl-minutes` interface (`scripts/pod_create.sh:137-166`), reads
a mutable private state path for expiry (`scripts/pod_create.sh:262-266`), and
automatically deletes a newly created Pod when readiness fails
(`scripts/pod_create.sh:194-205`). The launch documentation also directs the
operator to `delete` (`docs/runpod_launch.md:44-50`). Automatic deletion is not
authorized under the current destructive-action gate.

### 1.2 Current RunPod surfaces

All facts in this subsection were checked on 2026-08-14.

**CONFIRMED FACT.** Installed `/opt/homebrew/bin/runpodctl` reports
`2.3.0-be4ced4`; its binary SHA-256 is
`9f5ba34052c63c73a000bef173c61a1f29328bf4ff5930087755da5473b240ce`.
Its Go build metadata binds official source commit
[`be4ced4cdeef`](https://github.com/runpod/runpodctl/commit/be4ced4cdeef56bc3d782f813c5c90291b7c896f)
but also reports `vcs.modified=true`, so the installed binary's bytes—not the
source revision alone—must be pinned.

**CONFIRMED FACT.** This installed CLI exposes GPU-create flags
`--stop-after` and `--terminate-after`. Its exact source forwards the supplied
strings as GraphQL input fields `stopAfter` and `terminateAfter` without local
date validation. The official GraphQL schema types both fields as `DateTime`.
The CLI help/source examples use absolute RFC3339 timestamps. The official CLI
documentation describes `--stop-after` as stopping the Pod and
`--terminate-after` as permanently deleting it. Sources:

- [installed-version create source](https://github.com/runpod/runpodctl/blob/be4ced4cdeef56bc3d782f813c5c90291b7c896f/cmd/pod/create.go);
- [installed-version GraphQL input](https://github.com/runpod/runpodctl/blob/be4ced4cdeef56bc3d782f813c5c90291b7c896f/internal/api/graphql.go);
- [official GraphQL schema](https://github.com/runpod/pulumi-runpod/blob/d6ab724d3113c2961e20c6179b7ad0d2a8c29768/provider/pkg/runpod/schema.graphql#L299-L315);
- [official CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod#--stop-after).

The proposed seam therefore accepts canonical absolute UTC RFC3339 values
only. It does not accept the documentation's alternative duration wording and
does not pass unvalidated free-form values.

**CONFIRMED FACT.** In the inspected installed CLI source, deadline fields are
sent only by the GPU GraphQL create route. Its CPU REST create route drops
them. The replacement must refuse CPU creates. This negative claim is scoped
to the installed/current surfaces inspected; it is not a claim that RunPod
can never add a deadline-capable REST or CPU route.

**CONFIRMED FACT.** The inspected create response, Pod GET/list response,
`pod start`, and `pod update` surfaces do not expose deadline readback or a
way to attach/refresh a deadline on start. Therefore:

- flag acceptance is not per-Pod deadline proof;
- resume cannot presently install a fresh deadline;
- claims about deadline survival across stop/start require independent live
  certification, not inference.

These negative claims are limited to the exact CLI source/help and official
API schemas inspected.

**CONFIRMED FACT.** Official lifecycle documentation currently says
termination permanently deletes data not stored on a network volume and says
network-volume `/workspace` data is preserved whether the Pod is stopped or
terminated, implying that stop is supported. Prior and cached official text
has said that network-volume Pods cannot be stopped. The governing scheduler
plan therefore requires an exact-profile read-only capability probe plus an
integration smoke (`docs/job_scheduler.md:363-372`). This proposal preserves
that gate and does not promote either page version into a permanent capability
assumption. Source:
[Manage Pods](https://docs.runpod.io/pods/manage-pods).

## 2. Boundary and ownership

### 2.1 Process boundary

**PROPOSED CONTRACT.** `runpod-safe` v3 is a versioned executable owned by the
approved independent `ethanelasky/job-scheduler` repository. The scheduler
invokes it as a subprocess. There is no Python import, submodule, daemon,
background timer, provider-selection policy, queue, or second database inside
this seam.

```text
scheduler SQLite intent
        |
        | stable request JSON + exact digest
        v
versioned runpod-safe executable
        |
        | pinned runpodctl bytes, fixed official HTTPS origin
        v
RunPod

runpod-safe append-only journal = provider-boundary evidence
scheduler SQLite               = sole reconciliation authority
```

The executable performs at most one provider mutation for one durable mutation
operation. After a paid attempt reaches STOP or another terminal observation,
`observe-cost` may repeat bounded provider **reads** and append local billing
evidence; it never mutates a provider resource. `audit` is stricter still: it
performs reads and changes neither the journal nor SQLite.

The process surface is:

```text
runpod-safe create       --request <request.json>
runpod-safe resume       --request <request.json>
runpod-safe stop         --request <request.json>
runpod-safe terminate    --request <request.json>   # disabled until §6 approval
runpod-safe observe-cost --operation-id <UUID> --request <request.json> # reads + evidence append
runpod-safe audit        --strict --json            # reads only; no local append
runpod-safe ownership-receipt --operation-id <UUID> # immutable local authority read
runpod-safe cost-receipt      --operation-id <UUID> # immutable local evidence read
runpod-safe view              --operation-id <UUID> # derived, digested, non-authoritative
```

#### 2.1.1 One kernel lock for the journal boundary

**PROPOSED CONTRACT.** Every command that can append the journal—including
`create`, `resume`, `stop`, `terminate`, ambiguity reconciliation, and
`observe-cost`—acquires one owner-private kernel `flock(LOCK_EX)` on
`~/.local/state/runpod-safe-v3/journal.lock` **before** reading the state root,
journal head, stable-ID index, or evidence authority. It holds that same lock
continuously through request/evidence validation, `intent` fsync,
`call_started`/`read_started` fsync, the single provider mutation or bounded
read, outcome and evidence fsync, directory fsync, head/index publication, and
immutable receipt projection. It never drops/reacquires the lock around the
provider call.

The state directory and lock file must be owned by the effective user, mode
`0700` and `0600` respectively, regular, opened no-follow, and rejected for
symlinks, hard links (`st_nlink != 1`), owner mismatch, replacement, or a state
root reached through a symlink. The implementation verifies the opened file
identity before and after locking. A process crash releases the kernel lock,
but its last fsynced journal stage remains authoritative. The next identical
invocation locks first and replays/reconciles that durable stage; a different
digest or conflicting stable identity refuses while holding the lock.

`observe-cost` is a journal appender and therefore uses the same exclusive
lock even though its provider action is read-only. `audit` never appends; it
holds `LOCK_SH` for its entire stable local snapshot and provider comparison,
so no exclusive appender can race the chain it reports. Local receipt/view
reads likewise take `LOCK_SH`. There is no unlocked head lookup or second lock
domain.

### 2.2 Stable caller identity and replay

**PROPOSED CONTRACT.** The caller—not the wrapper—creates and durably stores:

- `client_intent_id`: the scheduler transition identity;
- `operation_id`: the single provider-call identity for that transition;
- the canonical request bytes and their SHA-256.

The scheduler fsyncs these in SQLite before invocation. A manual
`pod_create.sh` caller must first fsync the same request under an owner-private
launcher state directory. The wrapper never generates a replacement identity
on retry.

On invocation:

1. A new `(client_intent_id, operation_id, request_sha256)` appends and fsyncs
   `intent`.
2. For a mutation, the wrapper appends and fsyncs `call_started` immediately
   before the one provider call.
3. It appends and fsyncs exactly one known mutation outcome when available.
4. Repeating the identical tuple replays the durable outcome or performs
   read-only ambiguity reconciliation; it never calls the mutation again.
5. Reusing either stable ID with different canonical bytes/digest is a hard
   conflict. A new operation ID cannot bypass an unresolved prior intent for
   the same transition or target.

A crash after `call_started` but before the network call is conservatively
ambiguous. This may block harmlessly; it cannot duplicate a paid mutation.

`observe-cost` uses the same stable tuple but appends `read_started` and
`billing_observed` events. Identical invocations may repeat provider reads only
until the request's bounded finalization time. They never repeat compute or a
provider mutation.

### 2.3 What an ownership receipt grants

**PROPOSED CONTRACT.** A create ownership receipt exists only when the
provider returned an exact Pod ID and a fresh normalized read verifies the
complete allocation against the requested full spec, or when a lost create
acknowledgement later resolves to one exact unique nonce/full-spec delta.

Names alone grant no authority. Account inventory grants observation only.
An ownership receipt permits later requests to be evaluated for resume or
safe stop. It does **not** grant terminate, delete, network-volume mutation,
or auto-destroy authority.

Foreign/manual Pods are read-only unless the scheduler's separately approved
manual-enrollment contract permits safe stop. This seam has no foreign-Pod
adoption or `--allow-untracked` path.

## 3. Exact request and evidence schemas

All objects are UTF-8 JSON. Duplicate and unknown keys are rejected. Canonical
bytes follow RFC 8785. Digests use `sha256:` plus 64 lowercase hex characters.
Timestamps are UTC RFC3339 ending in `Z`. Provider identities are qualified as
`runpod:pod:<id>` and `runpod:network-volume:<id>`.

### 3.1 Operation request: `runpod-safety.request/v1`

Every listed field is required; an inapplicable value is `null`.

```json
{
  "schema": "runpod-safety.request/v1",
  "client_intent_id": "UUID",
  "operation_id": "UUID",
  "request_sha256": "sha256:<canonical object excluding this field>",
  "operation": "create|resume|stop|terminate|observe-cost",
  "requested_at": "RFC3339Z",
  "caller": {
    "kind": "scheduler|debate-launcher|break-glass",
    "protocol_version": "string",
    "job_id": "string or null",
    "attempt_id": "string or null"
  },
  "provider": {
    "name": "runpod",
    "api_family": "graphql-v1|rest-v1",
    "account_subject_sha256": "sha256:<provider account identity>",
    "credential_source": "env:RUNPOD_API_KEY|runpodctl-config"
  },
  "authority": {
    "kind": "standing-paid-nondestructive|owned-receipt|manual-stop-enrollment|destructive-authorization|billing-observation",
    "reference_sha256": "sha256:<independent authority object>"
  },
  "target": {
    "worker_id": "runpod:pod:<id>",
    "create_nonce": "64 lowercase hex",
    "ownership_receipt_sha256": "sha256:<receipt>"
  },
  "deadline": {
    "action": "stop|terminate|null",
    "at": "RFC3339Z or null",
    "ttl_seconds": "positive integer or null",
    "required_runtime_seconds": "positive integer or null",
    "evacuation_margin_seconds": "nonnegative integer or null",
    "observation_margin_seconds": "positive integer or null",
    "margin_policy_sha256": "sha256:<capability/profile margin policy> or null",
    "capability_certificate_sha256": "sha256:<certificate> or null"
  },
  "observations": {
    "observed_at": "RFC3339Z",
    "valid_until": "RFC3339Z",
    "pod_inventory_sha256": "sha256:<normalized complete inventory>",
    "volume_inventory_sha256": "sha256:<normalized complete inventory>",
    "price_storage_evidence_sha256": "sha256:<normalized retained evidence> or null",
    "estimated_bounded_total_native": "decimal string or null",
    "estimated_bounded_total_usd": "decimal string or null",
    "native_currency": "USD|CREDIT|ISO-4217 code|null",
    "plan_excluded_uncapped_charges": ["stable charge category"],
    "gpu_count": "positive integer or null",
    "evacuation_receipt_sha256": "sha256:<evidence> or null",
    "storage_disposition_sha256": "sha256:<policy> or null"
  },
  "create_spec": {
    "create_nonce": "64 lowercase hex",
    "name": "string containing the nonce",
    "template_id": "string or null",
    "resolved_template_sha256": "sha256:<normalized template> or null",
    "image": "string or null",
    "compute_type": "GPU",
    "gpu_type_id": "string",
    "gpu_count": 1,
    "cloud_type": "SECURE|COMMUNITY",
    "data_center_id": "string",
    "container_disk_gb": "positive integer",
    "pod_volume_gb": "nonnegative integer",
    "volume_mount_path": "absolute normalized path",
    "network_volume": {
      "id": "runpod:network-volume:<id>",
      "data_center_id": "string",
      "size_gb": "positive integer",
      "configuration_sha256": "sha256:<normalized configuration>"
    },
    "ports": ["sorted unique port/protocol"],
    "environment_names": [],
    "docker_args_sha256": "sha256:<bytes> or null"
  },
  "cost_observation": {
    "subject_operation_id": "UUID",
    "worker_id": "runpod:pod:<id>",
    "ownership_receipt_sha256": "sha256:<immutable ownership receipt>",
    "terminal_event_sha256": "sha256:<confirmed stop or terminal evidence>",
    "billed_window_start": "RFC3339Z",
    "billed_window_end": "RFC3339Z",
    "bounded_estimate_native_total": "decimal string",
    "bounded_estimate_usd_total": "decimal string",
    "native_currency": "USD|CREDIT|ISO-4217 code",
    "finalize_by": "RFC3339Z"
  },
  "destructive_authorization_sha256": "sha256:<one-shot artifact> or null"
}
```

Operation constraints:

- `create`: `target` is null; `create_spec`, fresh inventories, offer, price,
  storage observations, and a capability certificate are required. Exactly
  one of `template_id` and `image` is non-null. CPU is refused. Both deadline
  fields/actions cannot be requested. The wrapper computes
  `effective_ttl_seconds = deadline.at - requested_at` in exact seconds. The
  difference must be a positive integer no greater than 1,800; if the request
  retains `deadline.ttl_seconds`, it must equal that derived value exactly.
  Before journaling it also performs checked integer addition and requires
  `effective_ttl_seconds >= required_runtime_seconds +
  evacuation_margin_seconds + observation_margin_seconds`. In v1,
  `required_runtime_seconds` explicitly includes readiness/bring-up time; there
  is no separate hidden readiness margin. Missing/nonintegral/negative inputs
  or arithmetic overflow refuse. The capability certificate and exact profile
  pin `margin_policy_sha256`, the definition of each term, and permitted
  minima; the caller cannot shrink a pinned margin.
  Under the standing paid ceiling the request has one GPU and the wrapper's
  recomputed bounded USD total must be strictly below `$5`.
- `resume`: exact target and ownership receipt are required. Today this
  operation is refused under §5.3.
- `stop`: exact owned target, or a separately approved manual-stop enrollment,
  plus storage-disposition and evacuation evidence are required.
- `terminate`: schema support does not make it executable. It requires the
  independent one-shot artifact in §6; no such path is enabled today.
- `observe-cost`: exact target, subject operation, estimate, billed window,
  and a confirmed STOP/terminal evidence event are required; `deadline` and
  `create_spec` are null. Repeated invocations may issue provider billing reads
  only until `finalize_by` and append §3.5 evidence. They never mutate the Pod
  or rerun compute. The CLI `--operation-id` must exactly equal the request's
  stable `operation_id`; a mismatch refuses before any provider read.
- `environment_names` is empty in v1. The seam is not a job-secret transport.

Clock validation separately permits only a configured, certificate-bound
maximum observation skew when comparing `requested_at` with the wrapper's
clock. Skew never changes either timestamp, the derived TTL, the deadline, or
the 1,800-second ceiling; it can only reject a request whose freshness cannot
be established.

For a nondestructive request, `request_sha256` is the digest of the canonical
object with only `request_sha256` omitted. For a destructive request, the
request must have `authority.kind == "destructive-authorization"` and
`authority.reference_sha256 == destructive_authorization_sha256 ==` the
completed artifact hash. That hash fills both reference fields before the full
request digest is computed. Section 6.1 defines the separate three-field core
projection that avoids the authority-reference cycle. Nondestructive request
digest/core rules are unchanged. All equality and digest checks occur before
any journal write or provider access.

#### 3.1.1 Immutable inventory evidence

The two inventory digests in `observations` are valid only when their exact
canonical bytes already exist as content-addressed immutable evidence:

```text
~/.local/state/runpod-safe-v3/evidence/sha256/<64-hex>.json
```

The file digest must equal its path and request reference. A digest without the
retained bytes is invalid and blocks the operation. The wrapper opens each blob
as an owner-only regular no-follow, link-count-one file and validates it before
appending `intent`.

Each pre- or post-call inventory uses this schema:

```json
{
  "schema": "runpod-safety.inventory-evidence/v1",
  "kind": "pod_inventory|volume_inventory",
  "provider": "runpod",
  "account_subject_sha256": "sha256:<authenticated account identity>",
  "snapshot": {
    "started_at": "RFC3339Z",
    "completed_at": "RFC3339Z",
    "normalization_version": "runpod-inventory/v1",
    "pagination": {
      "page_count": 1,
      "complete": true,
      "termination_reason": "provider_end_of_collection"
    }
  },
  "pods": [
    {
      "id": "runpod:pod:<id>",
      "name": "string",
      "created_at": "RFC3339Z or null",
      "desired_status": "string",
      "compute_type": "GPU|CPU|unknown",
      "gpu_type_id": "string or null",
      "gpu_count": "nonnegative integer or null",
      "cloud_type": "SECURE|COMMUNITY|unknown",
      "data_center_id": "string or null",
      "machine_id": "string or null",
      "template_id": "string or null",
      "image": "string or null",
      "container_disk_gb": "nonnegative integer or null",
      "pod_volume_gb": "nonnegative integer or null",
      "volume_mount_path": "string or null",
      "network_volume_id": "runpod:network-volume:<id> or null",
      "ports": ["sorted unique port/protocol"],
      "cost_per_hour": "decimal string or null"
    }
  ],
  "volumes": [
    {
      "id": "runpod:network-volume:<id>",
      "name": "string or null",
      "data_center_id": "string",
      "size_gb": "nonnegative integer",
      "storage_type": "string or null",
      "configuration_sha256": "sha256:<complete allowlisted configuration>"
    }
  ],
  "evidence_sha256": "sha256:<canonical object excluding this field>"
}
```

For `kind == pod_inventory`, `pods` contains every returned Pod and `volumes`
is empty; for `kind == volume_inventory`, the reverse holds. Items are sorted
by qualified ID. Every provider field used by nonce/full-spec matching,
ownership, cost/storage admission, or volume non-targeting must appear
explicitly; a missing required field makes that comparison `unknown`, not a
match. Provider environment values and other secret-bearing raw fields are
discarded before canonicalization and never retained.

The pre-call `intent` event references both evidence digests. Every later
reconciliation/terminal event references the exact post-call inventory blobs.
The v1 retention contract never prunes a request-referenced evidence blob or
its journal events. Comparing retained before/after canonical bytes—not a
freshly reconstructed digest—makes the exact provider delta reproducible after
a crash or process restart. This normalized complete inventory object is the
sole retained pagination/completeness evidence: `page_count`, `complete`, and
`termination_reason` describe collection, while no unretained page digests are
accepted as evidence.

#### 3.1.2 Retained price and storage evidence

`price_storage_evidence_sha256` references another immutable blob under the
same evidence root. Its allowlisted normalized schema is:

```json
{
  "schema": "runpod-safety.price-storage-evidence/v1",
  "provider": "runpod",
  "account_subject_sha256": "sha256:<authenticated account identity>",
  "observed_at": "RFC3339Z",
  "valid_until": "RFC3339Z",
  "offer_identity_sha256": "sha256:<exact normalized offer>",
  "profile_sha256": "sha256:<exact create/storage profile>",
  "native_currency": "USD|CREDIT|ISO-4217 code",
  "native_currency_quantum": "positive decimal string",
  "usd_currency_quantum": "positive decimal string",
  "usd_conversion": {
    "source_observation_sha256": "sha256:<allowlisted conversion source> or null",
    "observed_at": "RFC3339Z or null",
    "valid_until": "RFC3339Z or null",
    "upper_bound_usd_per_native_unit": "positive decimal string or null",
    "method": "identity-usd|time-bounded-conservative-quote"
  },
  "gpu_count": 1,
  "running_compute_per_gpu_hour": "nonnegative decimal string",
  "running_storage": [
    {
      "kind": "container_disk|pod_volume|network_volume|other",
      "size_gb": "nonnegative integer",
      "price_per_gb_hour": "nonnegative decimal string"
    }
  ],
  "stopped_storage": [
    {
      "kind": "container_disk|pod_volume|network_volume|other",
      "size_gb": "nonnegative integer",
      "price_per_gb_hour": "nonnegative decimal string"
    }
  ],
  "stopped_retention_horizon_seconds": "nonnegative integer",
  "controllable_other_charge_caps": [
    {
      "category": "stable category",
      "maximum_charge": "nonnegative decimal string"
    }
  ],
  "plan_excluded_uncapped_provider_charges": ["bandwidth or another stable category"],
  "computed_bounded_total_native": "nonnegative decimal string",
  "computed_bounded_total_usd": "nonnegative decimal string",
  "evidence_sha256": "sha256:<canonical object excluding this field>"
}
```

Every provider-controlled charge needed for admission must have a fresh rate,
quantity, or explicit cap. For `stopAfter`, the stopped-retention horizon is a
finite approved duration beginning at the deadline and all stopped storage is
priced through that horizon. For `terminateAfter`, the stopped-retention
horizon is exactly zero after termination. An unknown rate/quantity, an
unbounded controllable charge, a missing currency quantum, or an indefinite
stopped horizon blocks create.

Using arbitrary-precision decimal values, the wrapper calculates:

```text
running_seconds = effective_ttl_seconds
compute = running_compute_per_gpu_hour * gpu_count * running_seconds / 3600
running_storage = sum(size_gb * price_per_gb_hour * running_seconds / 3600)
stopped_storage = sum(size_gb * price_per_gb_hour * stopped_retention_horizon_seconds / 3600)
bounded_total_native = compute + running_storage + stopped_storage + sum(other caps)
bounded_total_usd = bounded_total_native * upper_bound_usd_per_native_unit
```

Each native component is rounded upward, never down, to
`native_currency_quantum` before it is summed. If the native currency is USD,
the conversion method is `identity-usd` and the multiplier is exactly `1`. If
it is CREDIT or any non-USD currency, the retained conversion source must be
unambiguous, independently allowlisted, observed no later than admission,
valid through admission, and supply a conservative upper bound in USD per
native unit. The converted total is rounded upward to `usd_currency_quantum`.
Absent, stale, ambiguous, or non-conservative conversion evidence blocks
create; credits or another currency are never compared directly with dollars.

The wrapper recomputes and records both native and USD bounded totals from
retained evidence and the derived TTL, requires exact matches with both caller
fields and both evidence fields, and enforces that the recomputed **USD** value
is strictly below `$5`. The separately displayed
`plan_excluded_uncapped_provider_charges` (for example bandwidth where the
provider offers no admission-time bound) are not included in the bounded
total, so the document and CLI must call it a **bounded estimate**, never an
all-in estimate; their categories must match the retained evidence exactly.

### 3.2 Journal event: `runpod-safety.event/v1`

```json
{
  "schema": "runpod-safety.event/v1",
  "sequence": 1,
  "event_id": "UUID",
  "recorded_at": "RFC3339Z",
  "previous_event_sha256": "sha256:<prior event> or null",
  "event_sha256": "sha256:<canonical event excluding this field>",
  "client_intent_id": "UUID",
  "operation_id": "UUID",
  "request_sha256": "sha256:<request>",
  "operation": "create|resume|stop|terminate|observe-cost",
  "stage": "intent|call_started|read_started|accepted|rejected|ambiguous|accepted_reconciled|confirmed_stopped|confirmed_absent|billing_observed|unsafe",
  "provider_verb": "podFindAndDeployOnDemand|start|stop|delete|billing_read|null",
  "normalized_request_sha256": "sha256:<allowlisted request> or null",
  "normalized_response_sha256": "sha256:<allowlisted response> or null",
  "evidence_sha256": ["sha256:<immutable evidence blob>"],
  "worker_id": "runpod:pod:<id> or null",
  "normalized_allocation_sha256": "sha256:<allocation> or null",
  "deadline": {
    "action": "stop|terminate|null",
    "at": "RFC3339Z or null",
    "provider_field": "stopAfter|terminateAfter|null",
    "capability_certificate_sha256": "sha256:<certificate> or null"
  },
  "storage": {
    "container_disk_disposable": "boolean or null",
    "pod_volume_disposable": "boolean or null",
    "independent_volume_id": "runpod:network-volume:<id> or null",
    "independent_volume_before_sha256": "sha256:<config> or null",
    "independent_volume_after_sha256": "sha256:<config> or null"
  },
  "tool_identity_sha256": "sha256:<install manifest>",
  "credential_identity": {
    "source": "env:RUNPOD_API_KEY|runpodctl-config",
    "account_subject_sha256": "sha256:<provider account identity>"
  },
  "outcome": "pending|accepted|rejected|ambiguous|confirmed_stopped|confirmed_absent|billing_final|billing_partial|billing_unavailable|billing_ambiguous|unsafe",
  "findings": ["stable machine-readable codes"]
}
```

Only allowlisted normalized provider fields are retained: qualified resource
IDs, lifecycle state, allocation fields, observed prices, timestamps, and
HTTP/GraphQL status classes. Raw response bodies, raw error strings, provider
environment values, authorization headers, and secret-derived hashes are
never persisted.

An `intent` for create must reference the retained pre-call Pod and volume
inventory blobs from §3.1.1 and the retained price/storage blob from §3.1.2.
Reconciliation and terminal events reference their retained post-call blobs;
`billing_observed` references the billing blob in §3.5. An event referring to a
missing or digest-mismatched evidence blob is malformed and cannot grant
authority.

The event chain establishes order and detects accidental alteration within an
anchored prefix. A hash chain alone does not prove that its tail was not
truncated. The scheduler stores each returned event/head digest in SQLite, and
manual callers store it with their durable request; those external references
anchor the observed head. Without an external head, audit must describe tail
completeness as unproven.

### 3.3 Immutable ownership receipt: `runpod-safety.ownership-receipt/v1`

An accepted create produces this immutable authority object. It never changes
when billing evidence arrives:

```json
{
  "schema": "runpod-safety.ownership-receipt/v1",
  "operation_id": "UUID",
  "request_sha256": "sha256:<request>",
  "outcome_event_sha256": "sha256:<journal event>",
  "journal_head_sha256": "sha256:<head at issuance>",
  "provider": "runpod",
  "worker_id": "runpod:pod:<id>",
  "create_nonce": "64 lowercase hex",
  "allocation_sha256": "sha256:<normalized actual allocation>",
  "allocation_matches_spec": true,
  "deadline": {
    "action": "stop|terminate",
    "at": "RFC3339Z",
    "capability_certificate_sha256": "sha256:<certificate>",
    "provider_accepted_event_sha256": "sha256:<deadline-bearing create acceptance>"
  },
  "bounded_cost_estimate": {
    "native_total": "decimal string",
    "native_currency": "USD|CREDIT|ISO-4217 code",
    "usd_total": "decimal string",
    "price_storage_evidence_sha256": "sha256:<retained evidence>",
    "plan_excluded_uncapped_charges": ["stable charge category"]
  },
  "findings": ["stable codes"],
  "ownership_receipt_sha256": "sha256:<canonical object excluding this field>"
}
```

The schema has no nullable or conditional authority state: it is issued only
after the exact worker ID/nonce, actual allocation digest, full-spec match,
deadline acceptance/certificate, and bounded estimate are durably known.
Rejected, ambiguous, or allocation-mismatch creates produce only journal
event/outcome evidence and never an ownership receipt or authority. Receipt
existence permits only consideration of audit, resume, and stop under their
separate gates; it never grants terminate, delete, destroy, or network-volume
action. A derived `active/` view may exist for diagnostics but is never
authority and may be deleted/rebuilt only under a separately approved
maintenance procedure.

Every later resume, stop, termination proposal, or cost record binds exactly
`ownership_receipt_sha256`; no combined or mutable receipt is authority.

Each billing observation may produce a separate immutable cost receipt:

```json
{
  "schema": "runpod-safety.cost-receipt/v1",
  "subject_operation_id": "UUID",
  "worker_id": "runpod:pod:<id>",
  "ownership_receipt_sha256": "sha256:<immutable ownership receipt>",
  "billing_observation_sha256": "sha256:<immutable billing evidence>",
  "bounded_estimate_native_total": "decimal string",
  "bounded_estimate_usd_total": "decimal string",
  "native_currency": "USD|CREDIT|ISO-4217 code",
  "actual_total": "decimal string or null",
  "actual_minus_native_estimate": "decimal string or null",
  "actual_currency": "USD|CREDIT|ISO-4217 code",
  "completeness": "final|partial|unavailable|ambiguous",
  "cost_receipt_sha256": "sha256:<canonical object excluding this field>"
}
```

`actual_total` and `actual_minus_native_estimate` are non-null only for exact
final billing evidence in the declared native currency. Partial, unavailable,
or ambiguous receipts remain durable
but cannot pass a paid smoke. A CLI may combine ownership and cost records into
a content-digested view, but that view is derived, carries an explicit
`authority_granted: []`, and is never accepted by a later operation.

### 3.4 Deadline capability certificate

```json
{
  "schema": "runpod-safety.deadline-capability/v1",
  "provider": "runpod",
  "profile_policy_sha256": "sha256:<exact profile>",
  "margin_policy_sha256": "sha256:<term definitions and minimum margins>",
  "tool_identity_sha256": "sha256:<install manifest>",
  "api_schema_sha256": "sha256:<inspected GraphQL schema>",
  "provider_field": "stopAfter|terminateAfter",
  "input_format": "absolute-rfc3339-utc",
  "proof_operation_id": "UUID",
  "accepted_create_event_sha256": "sha256:<event>",
  "automatic_action_observed_at": "RFC3339Z",
  "automatic_action_evidence_sha256": "sha256:<independent observations>",
  "stopped_storage_bound_sha256": "sha256:<bound> or null",
  "independent_volume_before_sha256": "sha256:<config> or null",
  "independent_volume_after_sha256": "sha256:<config> or null",
  "valid_until": "RFC3339Z"
}
```

The certificate is exact to provider field, wrapper/driver bytes, API schema,
profile/storage policy, and proof. Source inspection and an accepted mutation
alone cannot mint it; the provider must independently be observed performing
the scheduled action.

### 3.5 Billing observation: `runpod-safety.billing-observation/v1`

After STOP or another terminal observation, the scheduler submits one stable
`observe-cost` request. Each provider read produces a content-addressed
evidence blob and a local `billing_observed` journal event:

```json
{
  "schema": "runpod-safety.billing-observation/v1",
  "provider": "runpod",
  "account_subject_sha256": "sha256:<authenticated account identity>",
  "subject_operation_id": "UUID",
  "worker_id": "runpod:pod:<id>",
  "ownership_receipt_sha256": "sha256:<immutable ownership receipt>",
  "terminal_event_sha256": "sha256:<confirmed STOP or terminal evidence>",
  "provider_billing_record_ids": ["allowlisted provider record ID"],
  "attribution_source": "documented provider record join or independently tested attribution method",
  "observed_at": "RFC3339Z",
  "finalized_at": "RFC3339Z or null",
  "billed_window": {
    "start": "RFC3339Z",
    "end": "RFC3339Z"
  },
  "charges": {
    "compute": "decimal string or null",
    "storage": "decimal string or null",
    "other": "decimal string or null",
    "total": "decimal string or null",
    "currency": "USD|CREDIT|ISO-4217 code"
  },
  "bounded_estimate_native_total": "decimal string",
  "bounded_estimate_usd_total": "decimal string",
  "actual_minus_native_estimate": "decimal string or null",
  "completeness": "final|partial|unavailable|ambiguous",
  "findings": ["stable machine-readable codes"],
  "evidence_sha256": "sha256:<canonical object excluding this field>"
}
```

The arithmetic is exact decimal arithmetic in the stated native currency;
`actual_minus_native_estimate == charges.total -
bounded_estimate_native_total` when completeness is `final`. The immutable
record preserves both the admission-time native and USD bounded estimates; it
does not retroactively reconvert them. A final record needs either exact provider billing record IDs joined
to this Pod and billed window, or an independently specified and real-path
tested attribution source. This spec does not claim that RunPod currently
offers exact per-operation attribution. If exact attribution or provider
finalization cannot be proven, completeness is `partial`, `unavailable`, or
`ambiguous` as applicable.

Repeated identical `observe-cost` invocations perform provider reads only,
append each distinct observation, and stop when `final` is recorded or
`finalize_by` is reached. At the bound, the last non-final condition is durably
recorded and the operation closes nonpassing. The scheduler stores every
observation/event digest in SQLite and the attempt artifacts. No cost outcome
reruns compute, resumes a Pod, or repeats another mutation.

### 3.6 Audit output: `runpod-safety.audit/v1`

`runpod-safe audit --strict --json` performs reads only and emits:

```json
{
  "schema": "runpod-safety.audit/v1",
  "generated_at": "RFC3339Z",
  "complete": true,
  "journal_head_sha256": "sha256:<head>",
  "externally_anchored_head_sha256": "sha256:<head> or null",
  "tool_identity_sha256": "sha256:<install manifest>",
  "provider_inventory_sha256": "sha256:<normalized complete pods>",
  "provider_volume_inventory_sha256": "sha256:<normalized complete volumes>",
  "resources": [
    {
      "worker_id": "runpod:pod:<id>",
      "classification": "active_owned|pending_create|pending_mutation|archived_owned_absent|archived_owned_live|foreign|ownership_mismatch|malformed",
      "lifecycle": "RUNNING|EXITED|TERMINATED|UNKNOWN",
      "ownership_receipt_sha256": "sha256:<immutable ownership receipt> or null",
      "spec_match": "true|false|unknown",
      "deadline_status": "proven_live|expired|fired|unproven|not_applicable",
      "independent_volume_id": "runpod:network-volume:<id> or null",
      "allowed_actions": ["audit"],
      "findings": ["stable codes"]
    }
  ],
  "findings": ["stable global codes"]
}
```

Exit codes:

- `0`: provider observation complete and no unsafe finding;
- `10`: observation complete with ambiguity, foreign-live, mismatch, or other
  unsafe finding;
- `11`: provider observation unavailable or incomplete;
- `12`: malformed journal, unanchored/truncated uncertainty, broken chain, or
  tool-identity failure;
- `13`: chosen credential absent.

Audit does not append journal events; adopt a Pod; repair, void, resolve, or
quarantine an intent; stop, start, terminate, or delete anything; or edit
SQLite. Its classifications are observations, not state transitions.

## 4. Mutation and crash rules

### 4.1 Lost acknowledgement

**PROPOSED CONTRACT.** A lost `CREATE` acknowledgement remains ambiguous
indefinitely unless a replay of the identical stable request finds exactly one
new provider object that matches both the cryptographic nonce and full
allocation spec, or a true provider contract supplies authoritative rejection
or request absence for that operation. A name match is insufficient. One, two,
or any number of ordinary list omissions do not prove absence.

On the exact unique delta, the same create invocation may append
`accepted_reconciled` without issuing another mutation. Otherwise it returns
the durable ambiguous outcome event. There is no general `recover`, `reap`, `adopt`,
or audit-resolution verb in v1. The scheduler remains responsible for
reconciliation and keeps the intent/resource against capacity caps.

### 4.2 Crash matrix

| Durable point | Provider may have acted? | Recovery |
|---|---:|---|
| Before `intent` | No | No operation exists. |
| `intent` fsynced; no `call_started` | No | Identical invocation may append `call_started` and call once. |
| `call_started` fsynced; no outcome | Yes or no | Ambiguous; never call again. Identical create may perform only §4.1 reads. |
| Provider rejection normalized and fsynced | No accepted mutation | Replay the rejection. A later retry needs a new scheduler transition and new stable IDs. |
| Accepted response parsed; crash before outcome fsync | Yes | Ambiguous unless exact read-only evidence satisfies §4.1 or the target-operation confirmation rule. |
| Accepted outcome fsynced; ownership receipt/result delivery lost | Yes | Replay the same immutable ownership receipt or result; no provider call. |
| Stop acknowledgement lost | Maybe | Observe exact target. Confirm only under a documented/certified authoritative state contract; otherwise ambiguous. |
| Terminate acknowledgement lost | Maybe | Exact GET plus complete inventory are required, but ambiguity remains if the provider contract does not make them authoritative. Never repeat automatically. |
| Provider hard deadline acts during unknown work | Yes | Scheduler records the observed stop/absence and permanently fails the attempt without retry, retaining missing-evidence facts. |
| Journal chain/tool identity fails | Unknown | Refuse every mutation; audit only. |

An acknowledged terminate is not complete merely because the CLI exited zero.
Confirmation requires the exact target GET and complete Pod inventory under an
approved authority contract, plus an unchanged independent-volume inventory.

## 5. Deadline-to-verb mapping

### 5.1 No current create route

**CONFIRMED BLOCKER.** No paid RunPod create is authorized today. The old seam
is missing; no v3 capability certificate exists; the current hard-skipped
integration test is not a service-path proof; and the meaning of the user's
required provider-side “termination deadline” has not been selected between
the following materially different actions.

### 5.2 Exact mapping under decision

| Exact proven profile | Possible create field | Gate |
|---|---|---|
| Stop-capable; every stop-erased local layer satisfies the approved emergency-loss semantics | `stopAfter=<absolute UTC>` | Only if Ethan explicitly accepts automatic **stop** as satisfying the required provider-side termination deadline, an exact capability certificate exists, and finite stopped-storage billing is independently bounded. |
| Stop unsupported, including any exact network-volume profile for which the required probe proves that result | `terminateAfter=<absolute UTC>` | Future destructive Pod termination. Blocked unless Ethan grants a narrow destructive-deadline carveout bound to the create nonce/full request, profile, storage disposition, deadline, and independent-volume identity. |
| Capability unknown, deadline proof absent/stale, or stopped storage unbounded | None | Refuse create. |

There is no fallback between fields and no deadline-less fallback. Both fields
in one request are refused. `terminateAfter` is not rendered safe merely by an
attached independent volume: it still irreversibly deletes the Pod and its
Pod-local layers.

Under the standing paid ceiling, any future proof or smoke also requires:

- one GPU maximum;
- provider-side TTL no more than 30 minutes;
- fresh running-price and storage-size/rate observations;
- wrapper-recomputed bounded USD total strictly below `$5` (with native and
  USD estimates both retained);
- actual cost recorded against that bounded estimate;
- no mutation targeting an independent volume, history, snapshot, or SQLite.

The bounded estimate uses the exact §3.1.2 formula: running compute and running
storage from create to the derived deadline, finite stopped storage through the
explicit retention horizon for `stopAfter` (zero after termination for
`terminateAfter`), and every capped controllable other charge. Uncapped
plan-excluded provider charges are displayed separately and prevent this from
being described as an all-in estimate.

### 5.3 Resume/start

**PROPOSED CURRENT POLICY: REFUSE.** The installed start/update surfaces cannot
attach or read back a new deadline. Resume therefore stays disabled until one
of these is separately approved and certified:

1. a provider surface refreshes and reads back a deadline; or
2. exact-profile live evidence proves the original create-time deadline
   survives stop/start, and enough time remains for runtime, evacuation, and
   action-observation margins.

A Pod stopped by its own deadline is never assumed reusable. Resume never
falls back to terminate-and-recreate, and standing nondestructive authority
does not authorize that sequence.

### 5.4 Bootstrap proof is not circular

A capability certificate requires an actual provider-scheduled action. The
standing authorization cannot bootstrap it, because that authorization itself
requires a proven deadline before create.

Therefore an exact one-off bootstrap request must receive separate approval
before any proof run:

- a `stopAfter` bootstrap explicitly authorizes one otherwise-blocked paid
  create and the automatic stop observation, within the standing numeric
  ceiling;
- a `terminateAfter` bootstrap additionally authorizes that exact future
  destructive termination and its exact Pod-local loss boundary.

Passing source/loopback tests or observing GraphQL acceptance cannot substitute
for this live proof. A failed/ambiguous bootstrap mints no certificate and
does not authorize a retry.

## 6. Terminate/delete authorization

### 6.1 Independent one-shot artifact

**PROPOSED CONTRACT.** Terminate authority is a one-shot artifact issued
independently of the scheduler request. A caller cannot grant itself authority
by inserting fields in request JSON. The artifact schema is:

```json
{
  "schema": "runpod-safety.destructive-authorization/v1",
  "authorization_id": "UUID",
  "issued_at": "RFC3339Z",
  "expires_at": "RFC3339Z",
  "authorized_action": "terminate|terminate_after",
  "worker_id": "runpod:pod:<id> or null",
  "create_nonce": "64 lowercase hex",
  "authorized_request_core_sha256": "sha256:<canonical request core>",
  "ownership_receipt_sha256": "sha256:<receipt> or null",
  "provider_profile": "exact provider-qualified profile name",
  "provider_profile_sha256": "sha256:<exact profile>",
  "deadline_at": "RFC3339Z or null",
  "authorized_effective_ttl_seconds": "positive integer or null",
  "price_and_storage_ceiling": {
    "gpu_count_max": 1,
    "ttl_seconds_max": 1800,
    "estimated_bounded_total_usd_strictly_less_than": "5.00",
    "running_price_per_hour_max": "decimal string",
    "storage_gb_max": "nonnegative integer",
    "stopped_storage_price_per_hour_max": "decimal string",
    "stopped_retention_horizon_seconds_max": "nonnegative integer",
    "admission_currency": "USD"
  },
  "evacuation_receipt_sha256": "sha256:<evidence> or null",
  "independent_volume_id": "runpod:network-volume:<id> or null",
  "independent_volume_configuration_sha256": "sha256:<config> or null",
  "authorizer_record": "immutable reference to Ethan's exact approval",
  "issuer_identity": "pinned independent approval authority",
  "signature_algorithm": "approved algorithm identifier",
  "signing_key_id": "pinned issuer key identifier",
  "signature": "signature over the defined unsigned projection",
  "artifact_sha256": "sha256:<completed artifact excluding this field>"
}
```

The issuer/trust-anchor mechanism is intentionally not selected here. Until a
separate destructive-action decision approves that mechanism and exact scope,
the installed executable has no working manual terminate/delete path and
rejects `terminateAfter` creates. No `delete` alias is proposed.

`authorized_request_core_sha256` breaks the otherwise circular dependency
between request and artifact. Construction order is exact:

1. Require `authority.kind == "destructive-authorization"`, then compute the
   authorized request core from the canonical mutation request with
   `request_sha256`, top-level `destructive_authorization_sha256`, and nested
   `authority.reference_sha256` omitted. This three-field projection exists
   only for destructive requests; nondestructive rules do not change.
2. Populate the artifact, including that core digest, exact `deadline_at`, the
   independently derived `authorized_effective_ttl_seconds`, and ceilings.
3. Sign the canonical artifact with both `signature` and `artifact_sha256`
   omitted, using the declared `signature_algorithm` and `signing_key_id`.
4. Compute `artifact_sha256` over the completed canonical artifact, including
   `signature`, with only `artifact_sha256` omitted.
5. Fill both `authority.reference_sha256` and
   `destructive_authorization_sha256` with that same completed artifact hash,
   require them to be byte-for-byte equal, then compute `request_sha256` over
   the full canonical request with only `request_sha256` omitted.

The artifact independently enumerates the exact action, target Pod or create
nonce, profile, deadline, derived TTL, price/storage ceilings, evacuation
evidence, and independent-volume identity.

Before journaling or provider access, the validator reverses and verifies that
construction:

1. requires destructive authority kind and exact equality among the completed
   artifact's `artifact_sha256`, `authority.reference_sha256`, and top-level
   `destructive_authorization_sha256`, then recomputes the completed artifact
   hash with only `artifact_sha256` omitted;
2. verifies the declared algorithm/key against the trust anchor and verifies
   `signature` over the canonical artifact with both signature/hash fields
   omitted, plus one-shot freshness;
3. recomputes the authorized request core by omitting exactly
   `request_sha256`, top-level `destructive_authorization_sha256`, and nested
   `authority.reference_sha256`, then compares it with
   `authorized_request_core_sha256`;
4. derives effective TTL from request timestamps; requires the artifact's
   `deadline_at` and `authorized_effective_ttl_seconds` to match exactly and
   the derived value to be no greater than `ttl_seconds_max` and 1,800;
5. compares every other enumerated artifact field with the request and
   retained evidence; and
6. recomputes the full request digest after the completed artifact digest is
   embedded.

Any mismatch refuses locally. The artifact cannot authorize a changed target,
nonce, deadline, profile, ceiling, volume, or action.

### 6.2 Structural volume protection

The executable has no network-volume create, update, detach, resize, or delete
verb. A future authorized Pod termination additionally requires:

1. exact wrapper-created ownership or create-nonce binding;
2. no unresolved mutation for that target;
3. lifetime-disposable erased Pod-local layers;
4. verified evacuation unless the exact destructive deadline approval invokes
   the already approved emergency-loss semantics;
5. fresh exact Pod and complete Pod/volume inventories;
6. exact independent-volume identity/configuration before the action;
7. exact GET plus complete inventory confirmation under an approved
   authoritative-absence contract; and
8. unchanged independent-volume identity/configuration afterward.

Lost acknowledgement stays ambiguous when the provider's observation contract
does not prove final absence. The wrapper never repeats termination and never
touches the independent volume.

There is no `on_unready` terminate grant. Readiness failure cannot smuggle
destructive authority through create.

## 7. Audit and scheduler relationship

**PROPOSED CONTRACT.** SQLite owns desired state, retry decisions, caps,
worker assignment, lifecycle policy, and reconciliation. The wrapper journal
records provider-boundary evidence only. The scheduler stores stable operation
IDs, exact request digests, returned receipt/event hashes, and its own
interpretation. Neither system edits the other's history.

Audit reads the journal, the externally anchored head supplied by SQLite or a
manual caller, exact provider inventory, and exact volume inventory. It
reports disagreement. It never fixes disagreement. There is no second
reconciler hidden in `audit --strict`.

`observe-cost` is deliberately separate: it repeats bounded provider billing
reads and appends local evidence events, but performs no provider mutation or
SQLite reconciliation. The scheduler decides when to invoke it and stores its
evidence references and pass/fail interpretation.

The provider hard deadline is the only component that acts while the Mac and
scheduler are off. It is not a wrapper timer.

## 8. Security, credentials, and installation

### 8.1 Credential-neutral execution

**PROPOSED CONTRACT.** Item 18 remains open. Whichever source Ethan selects,
the wrapper extracts only the one approved API-key field:

- `env:RUNPOD_API_KEY`: accept that exact variable only; or
- `runpodctl-config`: open the fixed owner-private TOML file and parse only its
  `apikey` scalar.

It never sources an `.env` file or loads a config wholesale. It then launches
the pinned driver with a scrubbed environment, isolated temporary `HOME`, fixed
minimal `PATH`/locale/TMPDIR, and only `RUNPOD_API_KEY`. Jobs receive the
scheduler's independent cleared environment and never inherit this child
environment.

The production executable:

- pins official `https://api.runpod.io/graphql` and the exact official REST
  origins it uses;
- rejects endpoint overrides, proxy/config URL conflicts, and
  `RUNPOD_GRAPHQL_URL`/`RUNPOD_API_URL`-style substitutions;
- passes no credential in argv or query strings;
- retains only the allowlisted/redacted fields in §3.2;
- never persists raw provider bodies, environment values, headers, config
  contents, errors, or secret/key hashes; and
- records only the selected source name and a hash of the authenticated
  provider account identity.

### 8.2 Source and install identity

**PROPOSED CONTRACT.** Source, schemas, tests, and packaging live in the
approved independent job-scheduler repository. Distribution remains a
versioned executable/process boundary. A user-level versioned install is
proposed:

```text
~/.local/lib/boring-job-scheduler/<release>/bin/runpod-safe
~/.local/bin/runpod-safe -> the approved release
~/.local/state/runpod-safe-v3/events/<sequence>-<event-id>.json
~/.local/state/runpod-safe-v3/evidence/sha256/<digest>.json
~/.local/state/runpod-safe-v3/views/active/...
```

No root-owned one-shot installer or `/opt/homebrew` write is implied. If a
different source/install home is desired, that is an open seam decision.

The immutable install manifest binds:

- scheduler release, commit, protocol, wrapper, and schema digests;
- interpreter realpath/version/digest;
- exact `runpodctl` realpath/version/source revision/binary digest;
- expected official origins; and
- state-root identity.

Mutation refuses on any mismatch. State directories are `0700`; event/request
files are owner-only regular files, opened no-follow, link-count one, written
atomically, and file/directory fsynced. Production has no driver-path, origin,
credential-source, deadline-ceiling, or test-transport override.

Every journal event hash-chains the prior event and explicitly references all
request, inventory, billing, allocation, and terminal evidence blobs needed to
reproduce its conclusion. Referenced v1 events and evidence are retained
indefinitely; no automatic pruning exists. `views/active` is a rebuildable
non-authoritative index. An event or request whose referenced canonical bytes
are absent or digest-mismatched cannot be used for ownership, deadline,
destructive, delta, or cost authority.

## 9. Independent test and proof oracle

### 9.1 Provider-free gates

- Exact JSON Schema, canonicalization, duplicate/unknown-key, stable-ID, and
  same-ID/different-digest rejection.
- Destructive authorization construction in the required order: request-core
  digest omitting both top-level digest fields and nested authority reference;
  signature projection omitting signature/artifact hash; completed artifact
  hash omitting only its hash; equal insertion into both request authority
  references; then full request digest. Mutation refuses wrong authority kind,
  unequal references, projection/order errors, wrong algorithm or key ID,
  core/full-digest cycles, field mismatch, artifact substitution, or changed
  action/target/nonce/profile/deadline/ceiling.
- CPU, missing deadline, both deadlines, template-plus-image, malformed/past
  timestamp, fractional/nonpositive derived TTL, caller-TTL mismatch, clock
  skew outside its separate bound, checked-margin sum overflow, TTL shorter
  than runtime (including readiness) plus pinned evacuation/observation
  margins, stale observation, multi-GPU, over-30-minute derived TTL, and
  recomputed bounded USD total at-or-above-`$5` refusal. No skew allowance may
  lengthen the TTL ceiling.
- Fresh price/storage/resource observation expiry; content-addressed retention
  of exact complete pre/post Pod and volume inventories; account, pagination,
  timing, field-completeness, missing-blob, digest-mismatch, and deterministic
  before/after delta cases.
- Recompute bounded cost from retained rates, quantities, the derived TTL,
  finite stopped horizon, controllable caps, and upward currency-quantum
  rounding. Refuse caller-total mismatch, unknown inputs, controllable
  unbounded charges, indefinite stop storage, or mislabeled/excluded charge
  categories; require terminate-after stopped horizon zero. Test native USD
  identity and time-bounded conservative CREDIT/non-USD conversion, retaining
  both totals; absent/stale/ambiguous conversion refuses and native units are
  never compared directly with `$5`.
- Actual pinned `runpodctl` executed against loopback GraphQL/REST instruments
  outside the production wrapper, proving exact GPU serialization and the
  inspected CPU omission. Production-origin overrides remain rejected.
- Crash injection at every §4.2 boundary for every verb; stable replay never
  repeats a mutation.
- Multiprocess lock tests prove every appender serializes from pre-head lookup
  through receipt publication, crash releases the kernel lock without erasing
  its durable stage, identical contenders replay, conflicting contenders
  refuse, audit observes one shared-lock snapshot, and unsafe lock path/owner/
  link replacement refuses.
- Lost-create exact unique nonce/full-spec delta acceptance; name-only,
  multiple match, changed spec, and any number of list-omission cases remain
  ambiguous.
- Journal corruption/order checks and explicit proof that unanchored hash
  chains cannot claim tail completeness.
- Symlink/hardlink/path replacement, concurrent process, wrong account,
  malformed/redaction, and credential-leak probes.
- Terminate and `terminateAfter` refusal without an independently verifiable
  one-shot artifact; no `on_unready` or delete alias.
- `observe-cost` issues reads only; repeated stable requests append partial,
  unavailable, ambiguous, then final evidence without a mutation. Ownership
  receipt bytes/hash never change; each cost receipt binds that ownership
  receipt and one billing observation. Only a final exact-attribution cost
  receipt contains actual cost; combined views grant no authority. Bounded
  non-final completion makes the paid smoke fail and never reruns compute.
- Exact ownership-receipt issuance only for non-null worker/nonce/allocation,
  `allocation_matches_spec == true`, proven deadline/certificate, and both
  native/USD estimates; rejected, ambiguous, and mismatch outcomes never mint
  a receipt. Later operations accept only its immutable hash.
- Real CLI/socket/service/SQLite/wrapper/artifact composition, not only a fake
  adapter authored beside the code.

### 9.2 Read-only real-path gates

- Through the actual scheduler service path, inspect installed driver identity,
  exact Pod inventory, exact volume inventory, profile capability evidence,
  price/storage observations, and audit output.
- Verify `audit` causes no journal, SQLite, or provider mutation.
- Independently verify the provider's billing finalization and exact
  Pod/operation attribution surface before any `final` cost evidence is
  accepted. If no such surface is proven, exercise honest
  partial/unavailable/ambiguous outcomes instead.
- Keep the present paid smoke unconditionally skipped throughout these gates.

### 9.3 Paid capability proof

Only after the exact bootstrap authorization in §5.4:

- traverse CLI → socket → service → SQLite → wrapper → provider;
- use one GPU, provider TTL derived at no more than 30 minutes, and a
  wrapper-recomputed bounded USD estimate strictly below `$5`, with its native
  quote and conversion evidence retained;
- verify exact create request/nonce/full allocation and SSH-command readiness;
- independently observe the provider-scheduled action;
- after STOP/terminal observation, run bounded `observe-cost` reads and record
  the final actual cost, estimate, and delta using independently proven exact
  attribution;
- compare exact pre/post Pod and volume inventories; and
- mint a certificate only for the exact proven driver/wrapper/API/profile
  tuple.

A `terminateAfter` proof is itself an exact destructive test and stays
separately gated. A failed or ambiguous proof leaves the create route closed.
Partial, unavailable, or ambiguous billing evidence also makes the paid proof
nonpassing; it never triggers another create or reruns compute.

## 10. `pod_create.sh` and `pod_up.sh` migration

Only after this seam is approved, implemented, and certified:

1. Keep `pod_create.sh` hard-failing until the pinned v3 executable and
   capability certificate validate.
2. Build and fsync a canonical v1 request with stable caller-generated IDs;
   invoke `runpod-safe create --request <path>`.
3. Preserve template/image exclusivity, exact datacenter/volume identity,
   stock/price/storage preflight, and SSH-command readiness.
4. Remove the readiness-failure automatic delete trap. Preserve readiness
   evidence and quarantine/operator-block. A safe explicit `STOP` is optional
   only when the exact profile is stop-capable and its storage/disposability
   gates pass; otherwise the approved provider deadline is the sole backstop.
5. Read expiry, Pod ID, bounded cost estimate, and authority only from
   `runpod-safe ownership-receipt --operation-id <stable-id>`, never from a
   combined or mutable `active/` view. This immutable receipt contains no
   actual cost and every later operation binds its exact hash.
6. After STOP/terminal observation, submit the stable `observe-cost` request,
   store every billing evidence/event digest in SQLite and artifacts, and
   read an immutable `cost-receipt` bound to the ownership receipt and billing
   evidence. Only a final exact-attribution cost receipt contains actual cost;
   a bounded non-final result fails the paid smoke without rerunning work.
7. Update `docs/runpod_launch.md` to remove the old delete instructions and
   describe the approved verb/proof only after approval.
8. Migrate `pod_up.sh` from bare `runpodctl start` to the selected resume
   policy. Its read-only maintenance preflight may remain, but no start occurs
   while resume is blocked.

There is no automatic terminate-and-recreate path under standing authority.

## 11. Decisions, separated by authority domain

### 11.1 Namespace contracts

Launch-namespace contracts are outside this document. Their approval or
rejection neither approves nor changes this RunPod seam.

### 11.2 Credential choice

Choose exactly one item-18 source; this proposal does not choose:

- add `RUNPOD_API_KEY` to `~/code/.env`, from which the service parses only
  that key; or
- have the adapter/wrapper parse only `apikey` from
  `~/.runpod/config.toml`.

In both cases the provider subprocess receives the key only in its scrubbed
service-side environment, while the remote job receives the plan's unrelated
cleared environment.

### 11.3 Replacement-seam decisions

These are the minimum design decisions needed before implementation:

1. **Deadline action:** accept certified `stopAfter` as satisfying the hard
   provider-side termination-deadline condition, with a real finite
   stopped-storage bound; grant a narrow exact `terminateAfter` destructive
   carveout; or approve neither. No option is selected.
2. **Proof bootstrap:** choose whether and how to authorize the exact one-off
   otherwise-blocked proof create; `terminateAfter` proof additionally needs
   exact destructive authorization. Source/loopback evidence alone is not a
   live capability certificate.
3. **Resume:** keep resume blocked (proposed current policy), or allow only
   exact-profile certified reuse of an original deadline with all margins.
   Terminate-and-recreate is not an option under standing authority.
4. **GraphQL dependency:** accept or reject binding this safety property to
   the inspected GraphQL create field while no inspected REST/start/update
   surface provides equivalent install/readback.
5. **Journal generation:** start a fresh incompatible v3 journal with no v2
   import, or separately specify a verified v2 recovery/import protocol.
   Digests alone cannot reconstruct v2 records.
6. **Source/install home:** accept the approved independent job-scheduler
   repository plus user-level versioned install above, or specify another
   non-root versioned process-boundary home.

Provenance search should still check the MBA, Frank's machine, and unmounted
backups before treating replacement as unavoidable. A recovered source whose
digests match §1.1 returns for independent inspection; it is not automatically
installed or approved.

Until the applicable credential and replacement-seam decisions are explicit,
all RunPod create/resume implementation and paid execution remain blocked.
