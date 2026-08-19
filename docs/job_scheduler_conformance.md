# Job scheduler conformance and migration

## Authority

This document contains proof obligations, acceptance observations, and sequencing only. It does not define scheduler
behavior. Each row points to the controlling clause in the [operative contract](job_scheduler.md). If wording here
appears to conflict with that clause, the operative clause controls. Authority and superseded history are recorded in
the [decision ledger](job_scheduler_decisions.md).

A test authored beside the implementation is not sufficient on its own. Proof must exercise the real writer, parser,
process boundary, SDK, provider response, or filesystem behavior named below. Provider-destructive tests remain
disabled unless the separate authority in [AUTH-001](job_scheduler.md#auth-001) is granted.

## Clause-to-proof matrix

| Clause | Independent proof | Acceptance observation |
|---|---|---|
| [SCHED-001](job_scheduler.md#sched-001) | Start the packaged launchd service and drive real `submit`, `status`, `logs --follow`, and `artifacts` CLI/socket/SQLite paths from a fresh process. Exercise each read view as an agent and attempt agent lifecycle mutation. Attempt a second service writer and cross-UID socket access; independently inspect artifact-root ownership/mode. Exercise all four worked workflows, exact-profile routing with compatible running, stopped, and unavailable inventories, omission of `max_attempts`, three command failures, and reconciliation transition boundaries. | Operators control while agent mutations, unauthorized peers, and a second writer refuse. The root is owner-only; all workflows, exact routing, FIFO/cardinalities, default three command executions, and one bounded transition behave as specified. |
| [STATE-001](job_scheduler.md#state-001) | Crash-inject before send, after send/before acknowledgement, and after acknowledgement for `CREATE`, `RESUME`, `ATTEMPT START`, `STOP`, and `DESTROY`. Restart from real SQLite each time. Use generic transport errors, confirmed rejection, inventories with same-name resources, nonce/spec mismatches, delayed visibility, and authoritative absence. | Intent and one-call recovery remain durable. Ambiguity quarantines without retry/replacement; only exact nonce/spec can claim a lost create. Attempt accounting and the no-terminal-record outcome match the clause. |
| [EXEC-001](job_scheduler.md#exec-001) | Stage real symlinks, escapes, undeclared files, exact argv/cwd, inherited secrets, and provider-injected named credentials. Issue one attempt, then independently change snapshot bytes, argv, cwd, and cleared environment one at a time and try duplicate issuance of the unchanged identity. Submit paid jobs with missing, zero, negative, nonfinite, and valid positive-finite `max_runtime`. Launch a child that forks descendants, expire it, and restart the service during timeout handling. Inspect the complete remote process tree and captured environment. | Invalid inputs/runtimes and changed or duplicate identities refuse; staged bytes verify. Only the declared environment reaches the job, and expiry removes all descendants before collection/reuse, including after restart. |
| [EVID-001](job_scheduler.md#evid-001) | Use real result, checkpoint-manifest, transcript/Docent, stdout/stderr, terminal, and partial-output writers. Corrupt, omit, and alter files before collection; restart during hashing, collection, and verification; independently hash collected logs, partials, and outputs; inspect attempted GC targets. | Required evidence gates success; all obtainable evidence or absence is durable and independently immutable. Restart never reruns compute, analyzed work needs local transcripts, and GC preserves checkpoint/nondisposable data. |
| [NS-001](job_scheduler.md#ns-001) | Before concurrent onboarding, inspect the checkpoint launch-ID path in `infra/run_common.py`, the divergent Docent paths in `infra/run_debate.py` and `infra/run_rlvr.py`, and the real `run_eval`, local transcript, artifact, checkpoint, Docent, W&B, declaration, and checkpoint-sync sinks. Then launch concurrently through real RLVR, debate, eval, transcript/Docent, W&B, declaration, and checkpoint paths. Include invalid namespaces, a forced reservation conflict, manual UUID generation, and legacy directories. Assert `run_identity_suffix` bytes before and after. | Onboarding waits for safe sinks. One unchanged namespace reaches them; conflicts refuse and eval reservation is exact/atomic. Legacy and scientific/protocol/W&B identities remain unchanged. |
| [CKPT-001](job_scheduler.md#ckpt-001) | Invoke the real checkpoint synchronizer with absent/malformed destination data and ambient AWS credentials. Exercise an exact durable local target and an S3-compatible test bucket with conditional requests, conflicts, corrupt GETs, listing extras/misses, and a file over 5 GiB. Restart after each sync failure and inspect durable evidence. Scan imports and network calls for Hugging Face. | Destination never follows credentials; exact local durability is a no-op. Bucket namespace/reservation, GET/hash, and final listing verify. Failure evidence persists without retry/adoption; oversize and all HF paths refuse. |
| [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001) | Review code and tests for an encoded choice about writer-ready/live-source stabilization, synchronizer coordination/PID/lock publication, or ambiguous partial-prefix continuation. | Implementation remains blocked at this boundary; tests expose the gate and do not silently select a recommendation. |
| [WAND-001](job_scheduler.md#wand-001) | Run against pinned real `wandb==0.28.1`; reject a mismatched runtime before API use and validate the run ID before filesystem/API use. Use two same-UID processes for `flock`, real public Run objects for every state, an offline ambient setting, override attempts through CLI/YAML/env/Config, and interruption at acquisition, state/config access, init, and ownership transfer. Use a two-host barrier so both hosts observe a terminal state before init. | The same-host loser makes no API call and lock lifetime is exact. Only allowed terminal states resume online; only audited run-bound CLI override bypasses `running`. Namespace/audit history is atomic, and the two-host residual race stays visible. |
| [DOC-001](job_scheduler.md#doc-001) | Spawn/exec the production JSONL worker against a loopback server with installed `docent-python==0.1.77`. Before transport creation, drift version, reviewed serializer source hashes, and the 100 MiB threshold. With representative one-batch records, compare canonical non-ASCII JSONL through reconstruction and the uncompressed pinned-serializer payload while varying Unicode escaping/gzip metadata; exercise auth/create/batch/status, status progression, canceled and every malformed census class, redirects/non-2xx, and long-name compact IPC. Use three records whose pinned serialization is each over 50 MiB to prove immediate missing/duplicate `job_id` refusal across multiple genuine batches. Inspect process group, fixed FDs, actual `PIPE_BUF`, stdio, allowlisted environment, parent signal/thread/Torch/tqdm state, and orphan alarm; probe the approved same-process/escaped-descendant limits. Exercise a TERM-ignoring known descendant through the five-minute TERM/250ms-drain/KILL/reap path, plus malformed/order/size frames, last-ID/null windows, interrupts, broken stderr, and local success. Separately, posix-spawn a child running the production-owned explicit transport and poller—not the JSONL worker or upload orchestration—against loopback; prove exact `(10, 120)` on every HTTP phase, `trust_env=false`, every Requests retry component zero, and 101 pending IDs sliced `[100, 1]`. Pair that harness with a real production-transport request under seeded dead ambient proxies to prove proxy bypass. | Drift refuses before transport or mutation. The production JSONL worker proves scientific/serializer-boundary bytes, namespace-named one-collection-per-attempt behavior, representative one/multiple-batch acknowledgements and census, one-send redirect/non-2xx refusal, compact IPC, honest receipts/control flow, no adoption, and authoritative local success. Isolation, parent-state, orphan-alarm, cleanup, and known-descendant observations match the clause without claiming socket cutoff or arbitrary-descendant census. The separate spawned adapter/poller harness proves the exact socket tuple and 100-ID transport slice without pretending to manufacture 101 genuine batches, which would exceed 5 GiB at the reviewed batching boundary; the ambient-proxy probe proves the same owned transport ignores environment routing and has zero retries. Representative proofs do not relax the operative clause. |
| [WORK-001](job_scheduler.md#work-001) | Inventory provider resources without enrollment; enroll manual workers with each attestation missing in turn; leave an old watchdog alive and race its accepted stop with handoff. Test a foreign worker and a different effective user. | Only fully attested exclusive workers become controlled, after durable watchdog handoff; observation, manual/foreign status, and enrollment never grant auto-destroy. |
| [LIFE-001](job_scheduler.md#life-001) | Drive running-to-idle-to-stopped retention with paid, ephemeral, and explicit-free profiles. Inventory every storage layer and inject missing evacuation/hash evidence, queued compatible work, unknown attempts, lost acknowledgements, missing scheduler ownership, stale inventory, an unfenced assignment, a never-ready worker with/without a started attempt, and independently managed volumes/snapshots. Compare independent before/after worker and volume inventories, including identity, existence, attachment, size, and configuration. | Clocks/reuse match the clause; finite retention alone never deletes. Every ownership, fresh-inventory, evidence, ambiguity, disposability, fence, absence, and never-ready gate is observed. Ambiguity quarantines; manual/foreign and protected storage/state remain stable; expected content changes have path-scoped proof. |
| [COST-001](job_scheduler.md#cost-001) | Change price, disk, resources, and capacity between inventory, admission, create/resume, and hot reuse. Return nonfinite, missing, ambiguous, and contradictory provider data. Independently inspect the actual allocation after state changes. | Every paid transition re-admits and verifies actual allocation; all owned/unresolved capacity counts. Unknowns block, and estimates expose uncapped charges without claiming all-in cost. |
| [CRED-001](job_scheduler.md#cred-001) | Start from a cleared inherited environment and a temporary RunPod config containing `apikey`, unrelated keys, duplicate/invalid fields, and hostile shell text. Trace file reads, process environments, argv, SQLite, artifacts, and logs without printing secret values; verify file modes by metadata only. Confirm the scheduler never sources `/Users/ethanelasky/code/debate/.env`, while recording that `scripts/mb_solo_to_docent.py` and `scripts/mb_judge_ablation.py` load that file wholesale. | Only `apikey` is parsed, never sourced/copied; isolated home/endpoints apply. Values never reach job/durable surfaces, both files stay `0600`, and Debate's `.env` is not Option A. |
| [AUTH-001](job_scheduler.md#auth-001) | Static-scan all command and test entry points, then exercise policy gates with destructive environment variables enabled. Build disabled `RD`/`VD` smokes and prove no provider-destructive call is reachable; separately prove `RDP`/`VDP` cannot cross `RDA`/`VDA`. For authorized nondestructive paid smokes, compute provider-deadline minus create/resume timestamp and record GPU count, estimate, actual cost, and targeted storage identity. After each exact destructive authorization is later obtained, run the provider's destructive-GC smoke through the real service path, independently audit protected identities and evidence, and retain immutable safe-destroy signoff. | Disabled smokes grant no execution; destructive calls require exact authorization. Nondestructive runs meet every provider/ceiling/deadline/storage gate without local substitutes. The later GC arc needs both separately authorized smokes/signoffs, neither of which enables auto-destroy. |
| [RUNPOD-001](job_scheduler.md#runpod-001) | Keep the legacy paid smoke inert, then verify upstream driver provenance/digest and the installed v3 path. Verify that the six answers are reconciled, every remaining non-ratified choice has explicit approval, and the resulting spec has independent review before adapter/lifecycle work. Independently inspect the exact GraphQL request, create-relative deadline arithmetic, authoritative provider state, and a fresh empty v3 journal with no v2 import or historical-digest adoption. The sole bootstrap, if its gate is satisfied, traverses only the reconciled/installed minimal safety seam and records estimate versus actual; it never uses ad hoc GraphQL or direct `runpodctl`. After certification, the rewritten nondestructive smoke must separately traverse CLI/socket/service/SQLite/adapter/transfer/wrapper/artifact root. Inventory container, Pod-volume, and network-volume identity before and after without destructive calls. | Creates cannot bypass the installed seam. Only the bounded bootstrap precedes certification; later operations stay nondestructive. Provider-observed create-relative `stopAfter`, fresh journal, disk/residual cost, blocked verbs/profiles, and no-duplicate ambiguity handling match the clause. |
| [VAST-001](job_scheduler.md#vast-001) | Exercise read-only discovery and static/dynamic gates with the `VAST_API_KEY` name present but its value redacted. Attempt every paid and destructive entry point without an independently proven bounded provider mechanism. If that mechanism and second approval later exist, the paid smoke must traverse CLI/socket/service/SQLite/adapter/transfer/wrapper/artifact root and independently inventory worker and volume state. | Key-name presence lifts nothing. Discovery works; all paid/destructive paths block before proof and approval, local watchdogs do not qualify, and unknown facts refuse. Later paid use is only the approved nondestructive path under AUTH-001. |
| [BREAK-001](job_scheduler.md#break-001) | Pause during local transactions and provider calls; crash before and after the durable fence-state record; restart and contend service/recovery tools on the same lock; inject an accepted-but-unacknowledged mutation; keep SQLite/artifacts readable. | Ordered pause and durable fence precede release; restart restores them and one owner recovers. Ambiguity/read views survive, without claiming provider rollback. |
| [REPO-001](job_scheduler.md#repo-001) | Build/install the pinned distribution in a clean environment and invoke Debate through the published CLI/protocol only. Inspect repository contents for all three documents, permissions, release pinning, collaboration settings, first pushed test state, and operator-runbook coverage of machine-off behavior, backstop outcomes, watchdog handoff, and accepted irreversible actions. | There is no import/submodule coupling. All three docs, pins, private/unlicensed state, collaborators, green-first-push, and complete runbook are independently observed. |
| [OPEN-001](job_scheduler.md#open-001) | Scan configuration, CLI help, tests, adapters, and service transitions for every open or blocked path. | Open recommendations and destructive/provider gates remain visibly disabled and cannot be enabled by ambient credentials, test flags, or inferred capability. |

## Mandatory sequencing

Steps 1–5 are a strict prerequisite chain. After step 5, the dependency graph—not numeric order—controls the parallel
workstreams. Every dependency is reviewed against the operative clause, not merely against locally authored tests.

1. Commit or stash the pre-existing dirty `infra/run_common.py` and `infra/train.py` work without mixing it into this
   project.
2. Keep the plan and decisions on an immutable commit for review.
3. Hard-disable the legacy paid RunPod smoke so `RUNPOD_PAID_INTEGRATION=1` cannot create anything.
4. Land the namespace fix as its own commit before scheduler work. Preserve `run_identity_suffix`; record that old
   `docent/<run>/pid-N/` directories and old `checkpoints/{run}/{step}/` keys are orphaned but untouched, and that retired
   Hugging Face repositories remain untouched and unread. Audit the checkpoint launch ID in `infra/run_common.py`, the
   divergent Docent behavior in `infra/run_debate.py` and `infra/run_rlvr.py`, and every eval/local-transcript/artifact/
   checkpoint/Docent/W&B/declaration/checkpoint-sync sink. Do not concurrently onboard an affected workload until all of
   its sinks are safe.
5. Rewrite or delete weak existing tests, then obtain fresh review of the revised test contract. Do not publish the
   independent repository before this gate is green.
6. Implement the local core/service and wrapper/transfer tracks in parallel, then compose them through the real local
   path and restart/fence smoke. Read-only provider discovery and credential/cost policy work may also proceed without
   crossing provider mutation gates.
7. In parallel after step 5, reconcile the six ratified RunPod answers into the v3 seam, obtain approval for every
   remaining non-ratified choice, independently review the resulting spec, and replace/pin the driver. Initial
   stop-capable non-network-volume adapter/lifecycle implementation also depends on the real local composition from
   step 6. The sole bounded bootstrap is unavailable until every prerequisite in
   [RUNPOD-001](job_scheduler.md#runpod-001) holds. Resume and all terminate/destroy paths stay closed.
8. Prove and separately approve Vast's independent bounded backstop before implementing or running its paid service
   path. Read-only Vast discovery may precede that proof.
9. Implement generic local GC gates. Destructive RunPod and Vast smoke implementations remain disabled until Ethan
   authorizes each exact execution. After those authorizations, both destructive smokes and immutable safe-destroy
   signoffs remain required for the later GC migration arc; passing/signing off either still does not authorize
   auto-destroy.
10. Run provider shadow jobs with auto-destroy off, clear concurrent namespace use, and obtain immutable whole-system
    signoff before migrating a provider/profile family. After destructive signoff, return to Ethan for a separate
    provider-specific auto-destroy decision.

At every phase/milestone, run a smoke and use fresh correctness/spec-fidelity, safety/scientific-semantics, test-oracle,
and intent reviewers. After the final milestone, use a fresh whole-system wave against an immutable operative-contract
commit, again including intent rather than only literal wording. Confirmed findings are fixed and the affected wave
repeats until clean.

## Migration dependency graph

```mermaid
flowchart TD
    D[Commit or stash tracked changes in infra run_common and train]
    P[Commit immutable contract]
    H[Hard skip paid RunPod smoke unconditionally]
    N[Standalone namespace fix sink audit tests and commit]
    R[Revise all remaining scheduler tests]
    C[Local core store and service]
    T[Wrapper snapshot and transfer]
    L[Local real path composition]
    V0[Vast read only discovery probe]
    VB[Prove and separately approve Vast backstop]
    RB[Reconcile approve and review RunPod v3 spec]
    RC[Apply approved RunPod credential isolation]
    RI[Implement gated RunPod stop capable smoke]
    RP[Standing authorized RunPod stop capable smoke passes]
    RNB[Refuse RunPod network volume profiles initially]
    G[Generic local GC safety]
    VI[Implement Vast paid service path smoke]
    VP[Standing authorized Vast smoke passes within ceiling]
    RD[Implement gated RunPod destructive GC smoke]
    RDA[Explicit Ethan authorization for exact RunPod destructive smoke]
    RDP[RunPod destroy smoke passes]
    VD[Implement gated Vast destructive GC smoke]
    VDA[Explicit Ethan authorization for exact Vast destructive smoke]
    VDP[Vast destroy smoke passes]
    SHR[RunPod shadow with auto destroy off]
    SHV[Vast shadow with auto destroy off]
    W[Concurrent affected workloads only after sink audit]
    SR[Immutable RunPod whole system signoff]
    DMR[RunPod default migration]
    SV[Immutable Vast whole system signoff]
    DMV[Vast default migration]
    SDR[Immutable RunPod safe destroy signoff]
    EAR[Explicit Ethan authorization for RunPod auto destroy]
    AD[Enable RunPod auto destroy]
    SDV[Immutable Vast safe destroy signoff]
    EAV[Explicit Ethan authorization for Vast auto destroy]
    AV[Enable Vast auto destroy]

    D --> P
    P --> H
    H --> N
    N --> R
    R --> C
    R --> T
    C --> L
    T --> L
    R --> V0
    V0 --> VB
    R --> RB
    R --> RC
    L --> RI
    RB --> RI
    RC --> RI
    RI --> RP
    R --> RNB
    L --> G
    L --> VI
    VB --> VI
    VI --> VP
    G --> RD
    RP --> RD
    RD --> RDA
    RDA --> RDP
    G --> VD
    VP --> VD
    VB --> VD
    VD --> VDA
    VDA --> VDP
    RP --> SHR
    G --> SHR
    VP --> SHV
    G --> SHV
    N --> W
    SHR --> SR
    W --> SR
    SR --> DMR
    SHV --> SV
    W --> SV
    SV --> DMV
    RDP --> SDR
    SHR --> SDR
    W --> SDR
    SDR --> EAR
    EAR --> AD
    VDP --> SDV
    SHV --> SDV
    W --> SDV
    SDV --> EAV
    EAV --> AV
```

## Sequencing and migration acceptance

- Each milestone leaves a durable record of the immutable contract commit, implementation commit, exact test command,
  independent observation, reviewer identities/lenses, and unresolved findings.
- The namespace commit is independently reversible at the code level but does not migrate old sink paths; its orphan
  consequences are recorded before scheduler commits begin.
- Packaging and repository transfer are verified from a clean clone. No history-rewrite ceremony is introduced, no
  Debate artifact is deleted, and the scheduler repository is not published with a deliberately red first commit.
- Paid RunPod proof records the create timestamp, create-carried provider deadline and its at-most-30-minute arithmetic,
  pre-run estimate, actual GPU count, storage allocation, and actual cost. Paid Vast and destructive execution nodes
  remain disabled before their gates regardless of credential presence.
- Every provider smoke uses the real service path. Direct adapter, CLI, fake, or synthetic binding tests can diagnose a
  seam but cannot alone satisfy acceptance.
- V0 is not complete until both the paid RunPod and paid Vast nondestructive service-path smokes succeed after their
  respective gates. A currently blocked gate leaves completion pending; it does not waive that provider from scope.
- The later destructive-GC arc is not complete until separately authorized `RDP` and `VDP` executions pass through the
  real service path and each receives immutable safe-destroy signoff. Those future requirements do not grant current
  execution authority, and signoff does not grant auto-destroy authority.
- Final acceptance requires all non-open matrix rows to pass, all open rows to remain closed, all required Mermaid
  before/after diagrams to agree with the implementation, and the whole-system reviewer wave to be clean.
