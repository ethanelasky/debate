# Job scheduler decision and provenance ledger

## Reading this ledger

This is the chronological authority, provenance, and supersession record for the
[operative contract](job_scheduler.md). It is not a second behavior specification: current behavior is stated only in
the operative clause linked from each entry. Historical values are retained here solely to explain how the current
contract arose. A recommendation or open item is not authority.

The immutable consolidation source is `8b95cd7:docs/job_scheduler.md`. Source spans below use its one-based line
numbers. Decisions made after that commit are explicitly labeled `post-8b95cd7`. Proof and migration obligations live
in [job_scheduler_conformance.md](job_scheduler_conformance.md).

## Authority tiers

- **Ethan direct:** Ethan supplied the decision himself. The checkpoint-destination change was direct and unprompted.
- **Ethan ratified:** Claude recommended a decision and Ethan reviewed and ratified it.
- **Codex-authored, Ethan-approved:** Codex proposed the original namespace contracts 1–4 and Ethan approved them.
- **Claude delegated:** Claude decided only the specifically delegated W&B or Docent substance. For W&B, Ethan has not
  reviewed Claude's remaining detailed mechanism; that portion is reversible on Ethan's review without the usual
  re-approval ceremony. No broader delegated authority exists.
- **Implementation constraint:** a fact imposed by a pinned dependency or platform, not approved design substance.

## Chronological operative decisions

### 2026-08-14 — base approval and amendments

| ID | Provenance | Decision and authority boundary | Current location | Immutable source |
|---|---|---|---|---|
| D-001 | Ethan approved checklist items except recorded amendments/open gates | V0 is the single local service/product boundary; items 1–8, item 12's read-only-discovery/independent-backstop-proof gate, items 13–16, and items 19–21 are approved only as amended by later rows. | [SCHED-001](job_scheduler.md#sched-001), [STATE-001](job_scheduler.md#state-001), [EXEC-001](job_scheduler.md#exec-001), [EVID-001](job_scheduler.md#evid-001), [WORK-001](job_scheduler.md#work-001), [LIFE-001](job_scheduler.md#life-001), [COST-001](job_scheduler.md#cost-001), [VAST-001](job_scheduler.md#vast-001), [BREAK-001](job_scheduler.md#break-001) | `L57-L169`, `L880-L922` |
| D-002 | Codex-authored, Ethan-approved | The original namespace contracts preserve `run_identity_suffix` exactly and record that old Docent pid directories become unreachable but are not deleted. The later direct requirement to land this work first and standalone is recorded separately in D-015. | [NS-001](job_scheduler.md#ns-001) | `L313-L340`, `L971-L1003` |
| D-003 | Codex-authored, Ethan-approved | One immutable launch namespace crosses all sinks; manual launches use one UUID; every sink refuses a conflict. Contract 4 preserves W&B display/scientific identity. The Phase-0 sink audit includes the known `infra/run_common.py`, `infra/run_debate.py`, and `infra/run_rlvr.py` hazards and blocks concurrent affected-workload onboarding until safe. | [NS-001](job_scheduler.md#ns-001) | `L313-L340`, `L723-L730`, `L971-L1003` |
| D-004 | Claude recommended; Ethan reviewed and ratified | Namespace contract 3 inserts the namespace, records that old bucket `checkpoints/{run}/{step}/` keys become unreachable but are not deleted, and initially kept the Hugging Face repo name while splitting out its hash rename. The contract-4 guard is practically vacuous for fresh manual UUIDs and is retained for future scheduler attempts. The later direct checkpoint decision in D-005 strikes the Hugging Face half and its rename question, but not the checkpoint-orphan record or contract-4 note. | [NS-001](job_scheduler.md#ns-001), [CKPT-001](job_scheduler.md#ckpt-001) | `L971-L1003` |
| D-005 | **Ethan direct, unprompted** | Retire Hugging Face as a checkpoint destination. Freeze an explicit `local` or S3-compatible `bucket` destination in run submission; never infer it from ambient credentials. Only LoRA adapters are synced, single conditional PUT is sufficient, and files over 5 GiB refuse. Existing Hugging Face repositories remain untouched and unread. | [CKPT-001](job_scheduler.md#ckpt-001), [NS-001](job_scheduler.md#ns-001) | `L342-L414`, `L880-L889`, `L971-L1003` |
| D-006 | No decision; the immutable record does not attribute an actor to the pending recommendations | The writer-ready/live-source boundary, synchronizer coordination/PID/lock publication, and ambiguous partial-prefix continuation remain recommendations, not decisions. | [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001), [OPEN-001](job_scheduler.md#open-001) | `L342-L414`, `L923-L950` |
| D-007 | Ethan direct amendment | Local transcript/Docent output is success-gating for analyzed workloads. External Docent/W&B uploads stay best-effort; ambiguous mutations are evidence and are not automatically repeated. | [EVID-001](job_scheduler.md#evid-001) | `L416-L438`, `L880-L922` |
| D-008 | Ethan open/rejection | Item 17 is not approved. Disabled destructive-smoke implementations may be built, but eligibility text, a passing execution, or immutable signoff grants neither destruction nor auto-destroy. Exact destructive execution and later provider-specific enablement each require separate approval. | [AUTH-001](job_scheduler.md#auth-001), [OPEN-001](job_scheduler.md#open-001) | `L117-L122`, `L452-L478`, `L890-L905` |
| D-009 | Claude recommended; Ethan reviewed and ratified | Item 18 uses Option B: parse only `apikey` from `~/.runpod/config.toml`, isolated from jobs and cleared environments. Option A meant `/Users/ethanelasky/code/.env`, never `/Users/ethanelasky/code/debate/.env`; `scripts/mb_solo_to_docent.py` and `scripts/mb_judge_ablation.py` load the latter wholesale into `os.environ`. Both credential files were changed from `0644` to `0600`. At the decision date, `/Users/ethanelasky/code/.env` contained only an assignment named `VAST_API_KEY`; no value is recorded here, and its presence lifted no Vast blocker. | [CRED-001](job_scheduler.md#cred-001), [VAST-001](job_scheduler.md#vast-001) | `L123-L126`, `L904-L911`, `L1168-L1206` |
| D-010 | Ethan direct amendment after fact check | At the decision date, `/opt/homebrew/bin/runpod-safe`, `~/.local/share/runpod-safety/`, and `~/.local/state/runpod-safety/` were absent. The guarantee list in `docs/runpod_launch.md` was insufficient to reconstruct a destroy-authorizing wrapper. The broader `docs/runpod_safe_replacement_spec.md` draft was not approved. | [RUNPOD-001](job_scheduler.md#runpod-001), [OPEN-001](job_scheduler.md#open-001) | `L89-L92`, `L890-L950` |
| D-011 | Claude recommended six answers; Ethan reviewed and ratified each | Initial RunPod accepts live-certified `stopAfter` only; permits exactly one create-relative, at-most-30-minute bootstrap; blocks resume; replaces and digest-pins the unverified driver before binding GraphQL behavior; starts a fresh v3 journal; and places source/install/state in the independent repository/path. Evidence at the decision date: the vendor-tap `runpodctl` was 2.3.0 and reported `vcs.modified=true`; inspected upstream v2.9.0 start/update still lacked deadline flags. No other draft clause is approved. | [RUNPOD-001](job_scheduler.md#runpod-001) | `L923-L950` |
| D-012 | Claude recommended in replacement-seam answer 1; Ethan reviewed and ratified | At the decision date, official RunPod pages disagreed about network-volume stop support: [storage types](https://docs.runpod.io/pods/storage/types) and [Pod lifecycle](https://docs.runpod.io/pods/manage-pods). Initial RunPod therefore refuses network-volume profiles and has no `terminateAfter` or terminate/recreate fallback; enabling such profiles needs a later explicit decision. | [RUNPOD-001](job_scheduler.md#runpod-001) | `L480-L494`, `L923-L950`, `L952-L1003` |
| D-012A | Ethan approved product-boundary behavior | One nonterminal job per worker still permits multiple workers to run in parallel. | [SCHED-001](job_scheduler.md#sched-001) | `L139-L169`, `L952-L970` |
| D-012B | Decision-time provider evidence, not mutation authority | The initial Vast policy premise came from Vast's [storage documentation](https://docs.vast.ai/guides/instances/storage/types), [`stop instance`](https://docs.vast.ai/cli/reference/stop-instance), and [`destroy instance`](https://docs.vast.ai/cli/reference/destroy-instance): stop preserved instance data while storage could bill, destroy removed instance-container storage, and separately managed volumes could survive. Fresh facts still gate every mutation. | [VAST-001](job_scheduler.md#vast-001) | `L487-L494` |
| D-013 | Ethan approved recommended outcome | If separately authorized provider action ends ambiguous unknown compute without a terminal record, fail permanently, record missing evidence, and never retry. This grants no provider verb. | [STATE-001](job_scheduler.md#state-001), [AUTH-001](job_scheduler.md#auth-001) | `L268-L290`, `L998-L1001` |
| D-014 | Ethan approved | The scheduler moves to private `ethanelasky/job-scheduler`, communicates with Debate by process/CLI, and is copied without history rewriting. Ownership, names, initial no-license status, green-first-push order, and Frank/Can invitations are approved. | [REPO-001](job_scheduler.md#repo-001) | `L952-L970` |
| D-015 | Ethan direct prerequisite order | Isolate the dirty tree, commit an immutable plan, hard-skip the legacy paid RunPod smoke, land namespace alone, then revise remaining scheduler tests/work. | [Mandatory sequencing](job_scheduler_conformance.md#mandatory-sequencing) | `L1004-L1010` |

### 2026-08-14 — separately delegated namespace-sink decisions

| ID | Provenance | Decision and authority boundary | Current location | Immutable source |
|---|---|---|---|---|
| D-016 | **Split:** Ethan directly rejected waiting for a scheduler lease; Claude chose the remaining mechanism under narrowly delegated W&B authority | Resumes are rejected immediately through a same-host flock followed by one remote-state read, with the explicit stale-heartbeat override and the documented millisecond cross-host residual race. Ethan has not reviewed Claude's remaining detailed substance; that portion is reversible on Ethan review without the usual re-approval ceremony. | [WAND-001](job_scheduler.md#wand-001) | `L1012-L1079` |
| D-017 | Claude delegated originally; Ethan subsequently reviewed the worked example and approved the described after behavior | Each launch uses a namespace-named external Docent collection while the scientific `AgentRun` payload is unchanged. Ethan's later review removes the earlier “substance not reviewed/reversible” status for this Docent behavior. | [DOC-001](job_scheduler.md#doc-001) | `L1081-L1166` |
| D-018 | Split: the 10/120/initial-15-minute timeout contract was recommended, then Ethan approved it; Ethan directly amended only the total from 15 to five minutes | Docent calls use 10-second connect/120-second read timeouts and a five-minute total, with the future scheduler clamp and separate five-minute evidence/shutdown reserve. Timeout/skip receipts, local-success, no-retry/no-adoption, and possible permanent slow-upload ambiguity are accepted. Ethan expected typical uploads likely not to be multi-GB while explicitly leaving open that they may be; this is rationale, not a size/duration promise. | [DOC-001](job_scheduler.md#doc-001) | `L1081-L1166` |
| D-019 | Implementation constraint, not Ethan-approved timeout substance | The pinned Docent 0.1.77 API limits status polling slices to 100 IDs while the exact-census rule remains. | [DOC-001](job_scheduler.md#doc-001) | `L1081-L1166` |

### 2026-08-14 — paid-integration correction

| ID | Provenance | Decision and authority boundary | Current location | Immutable source |
|---|---|---|---|---|
| D-020 | User correction recorded in the immutable plan; that text does not further attribute its actor | Standing paid integration-testing authorization covers nondestructive create/resume/run/stop/collect only within one GPU, create-carried TTL at most 30 minutes after `CREATE` or a fresh deadline on eligible resume/hot reuse, estimate under $5, actual-cost recording, provider-specific gates, and the absolute protected-storage exclusion. It does not make a provider-blocked verb executable. | [AUTH-001](job_scheduler.md#auth-001) | `L1168-L1206` |
| D-021 | Same recorded correction; no additional actor attribution inferred | The legacy RunPod paid smoke remains hard-skipped until rewritten for price/storage/deadline and actual service-path coverage. Only the ratified bootstrap can precede certification. Paid Vast remains blocked until its independent bounded mechanism is proven and separately approved despite `/Users/ethanelasky/code/.env` containing only an assignment named `VAST_API_KEY` at the decision date; no value is recorded here. | [RUNPOD-001](job_scheduler.md#runpod-001), [VAST-001](job_scheduler.md#vast-001) | `L1168-L1206` |

### 2026-08-16 — Docent hard-deadline and post-snapshot documentation decisions

| ID | Provenance | Decision and authority boundary | Current location | Source |
|---|---|---|---|---|
| D-022 | Codex recommended; Ethan reviewed and approved | Retain the alarm-based Docent hard deadline only on the main Python thread, with no other live Python thread, no active `ITIMER_REAL`, and `SIGALRM` neither blocked nor pending. Otherwise skip external upload with a sanitized receipt and keep local evidence authoritative. The conversation sequence corrects the immutable source's colloquial “directly approved” label: Codex recommended the mechanism and Ethan approved it. | [DOC-001](job_scheduler.md#doc-001) | `L1081-L1166` plus the governing conversation |
| D-023 | Codex recommended; Ethan reviewed and approved | Consolidate the plan into one concise operative contract, a non-normative conformance matrix, and this provenance/supersession ledger, without changing semantics or authority. Stable clause IDs and complete source mapping are required. The same recommendation-then-approval provenance rule applies; approval did not originate the proposal. | This three-document set | `post-8b95cd7`, 2026-08-16 governing conversation |
| D-024 | Governing repository policy, not a consolidation-derived decision | Current `AGENTS.md` makes transcripts load-bearing for scientific conclusions and treats a stopped Pod's volume as deep backup rather than the working copy. Those policies are carried into [EVID-001](job_scheduler.md#evid-001) and [LIFE-001](job_scheduler.md#life-001); this ledger does not attribute them to the 2026-08-14 scheduler approval. | [EVID-001](job_scheduler.md#evid-001), [LIFE-001](job_scheduler.md#life-001) | current `AGENTS.md` |

## Historical and superseded statements — non-operative

Every row in this section is historical only. It must not be used to authorize or implement behavior.

| Historical statement | Status and superseding decision |
|---|---|
| All paid provider execution is unauthorized or requires per-run approval. | Superseded by D-020's bounded standing nondestructive authorization. Provider-specific blockers still control. |
| The legacy paid RunPod smoke becomes live when `RUNPOD_PAID_INTEGRATION=1`. | Superseded by D-015/D-021: the flag must remain inert until the revised service-path gate passes. |
| `runpod-safe` is installed and can simply be extended. | Premise disproved; D-010/D-011 govern the absent seam and minimal v3 reconciliation. |
| Rebuild a safety wrapper from the guarantees in `docs/runpod_launch.md`. | Rejected as a bundled second project. Only D-011's six answers are ratified; the broader draft remains unapproved. |
| RunPod `terminateAfter`, a terminate/recreate fallback, resume, or initial network-volume profiles are available. | Rejected or blocked by D-011/D-012. `stopAfter` on stop-capable non-network-volume profiles is the only initial route. |
| Provider credentials come only from `/Users/ethanelasky/code/.env`. | Superseded by D-009 Option B for RunPod. Option A never referred to Debate's `.env`. |
| Checkpoint destination is inferred from `AWS_ACCESS_KEY_ID`, with Hugging Face as fallback. | Superseded by Ethan-direct D-005. Destination is explicit; Hugging Face is retired. |
| Rename the Hugging Face repository with namespace/hash or decide where to place a trailing hash under right truncation. | Moot and struck by D-005. Existing repositories are left untouched and unread. |
| Multipart support is needed to preserve existing full-state checkpoints. | Rejected by D-005's verified LoRA-only production boundary; files over 5 GiB refuse. |
| Concurrent W&B resumes wait on a future scheduler lease; until then the rule is convention only. | Superseded by Ethan's immediate-rejection choice in D-016. The direct guard works now but retains an honest millisecond cross-host race. |
| The W&B guard is an absolute distributed lock. | Never approved. D-016 records no W&B CAS, distributed writers, and the residual race. |
| Docent's delegated decision remained unreviewed by Ethan and freely reversible. | Superseded for Docent by Ethan's later review in D-017. The narrower unreviewed/reversible status remains only for Claude's detailed W&B mechanism. |
| Docent has no separate timeout contract. | Superseded by D-018. |
| Docent's total budget is 15 minutes. | Superseded by Ethan's direct five-minute amendment in D-018. |
| Typical Docent uploads are guaranteed not to be multi-GB or to finish within five minutes. | Never approved. Only the pinned SDK's roughly 100 MiB pre-gzip batching threshold is factual context; Ethan explicitly left size uncertain. |
| The 100-ID polling slice was part of Ethan's timeout approval. | Corrected by D-019: it is a pinned-SDK implementation constraint. |
| A finite stopped-retention value authorizes destroy or auto-destroy. | Never approved; D-008 and [AUTH-001](job_scheduler.md#auth-001) keep both separately gated. |

## Exact open, rejected, and gated items

| Gate | Status | What would change it | Current location |
|---|---|---|---|
| Checkpoint writer-ready/live-source stabilization | **OPEN recommendation** | Ethan's explicit design decision after plain-language tradeoffs | [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001) |
| Checkpoint synchronizer process/coordination and PID/lock publication | **OPEN recommendation** | Ethan's explicit design decision | [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001) |
| Ambiguous partial-prefix continuation | **OPEN recommendation** | Ethan's explicit design decision | [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001) |
| Item 17 auto-destroy | **OPEN; no authority** | Separate exact destructive smoke approval, passing immutable proof, then a later provider-specific enablement approval | [AUTH-001](job_scheduler.md#auth-001) |
| Destructive smokes | Disabled `RD`/`VD` implementation allowed; `RDP`/`VDP` execution **GATED and unauthorized** | `RDA`/`VDA`: Ethan's exact target-specific approval with every listed audit/ceiling | [AUTH-001](job_scheduler.md#auth-001) |
| Broader RunPod v3 draft | **REJECTED as bundled / unreconciled** | Reconcile to D-011's six answers, present remaining design decisions, obtain approval, and independently review | [RUNPOD-001](job_scheduler.md#runpod-001) |
| RunPod resume | **BLOCKED** | A separately approved way to install a fresh provider deadline on resume | [RUNPOD-001](job_scheduler.md#runpod-001) |
| RunPod network-volume, terminate, and destroy paths | **REFUSED/GATED** | New design and explicit authority; standing paid permission is insufficient | [RUNPOD-001](job_scheduler.md#runpod-001) |
| RunPod paid create before certification | **BLOCKED except one exact bootstrap** | Reconciled/installed/pinned seam and live `stopAfter` certificate | [RUNPOD-001](job_scheduler.md#runpod-001) |
| Paid Vast | **TECHNICALLY BLOCKED** | Independently prove a provider-side bounded mechanism and obtain Ethan's separate approval | [VAST-001](job_scheduler.md#vast-001) |
| Unknown provider/storage capability | **FAIL CLOSED** | Authoritative evidence resolving the fact, followed by any still-required approval | [COST-001](job_scheduler.md#cost-001), [OPEN-001](job_scheduler.md#open-001) |

## Semantic mapping from immutable source commit `8b95cd7`

This table covers the complete 1–1206 line source. “Ledger” identifies where authority/history moved; “conformance”
identifies proof, not behavior.

| Old heading and line span | Operative location | Ledger / conformance location |
|---|---|---|
| Title and Status, `L1-L9` | [Status and authority](job_scheduler.md#status-and-authority) | [Reading this ledger](#reading-this-ledger) |
| One-screen topology and ownership, `L10-L24` | [One-screen proposal](job_scheduler.md#one-screen-proposal), [SCHED-001](job_scheduler.md#sched-001), [REPO-001](job_scheduler.md#repo-001) | Matrix rows SCHED-001/REPO-001 |
| One-screen job shape, `L25-L44` | [SCHED-001](job_scheduler.md#sched-001), [EXEC-001](job_scheduler.md#exec-001), [EVID-001](job_scheduler.md#evid-001), [CKPT-001](job_scheduler.md#ckpt-001), [COST-001](job_scheduler.md#cost-001) | Matrix rows SCHED-001/EXEC-001/EVID-001/CKPT-001/COST-001 |
| One-screen CLI example, `L45-L53` | SCHED-001 [CLI contract](job_scheduler.md#cli-contract-and-worked-workflows), [EVID-001](job_scheduler.md#evid-001) | Matrix rows SCHED-001/EVID-001 |
| One-screen boundary summary, `L54-L56` | [SCHED-001](job_scheduler.md#sched-001), [EXEC-001](job_scheduler.md#exec-001) | Matrix rows SCHED-001/EXEC-001 |
| Approval checklist, `L57-L138` | [Approval checklist index](job_scheduler.md#approval-checklist-index) and linked clauses | D-001, D-008 through D-013; all matrix rows |
| Approved product boundary, `L139-L170` | [SCHED-001](job_scheduler.md#sched-001), [STATE-001](job_scheduler.md#state-001), [EXEC-001](job_scheduler.md#exec-001), [EVID-001](job_scheduler.md#evid-001), [LIFE-001](job_scheduler.md#life-001), [AUTH-001](job_scheduler.md#auth-001), [RUNPOD-001](job_scheduler.md#runpod-001), [VAST-001](job_scheduler.md#vast-001), [REPO-001](job_scheduler.md#repo-001) | D-001, D-008, D-012A, D-014; corresponding matrix rows |
| Why this helps—and where shell still wins, `L171-L178` | [SCHED-001](job_scheduler.md#sched-001) | Matrix row SCHED-001 |
| Before and after / Control plane, `L179-L200` | SCHED-001 [control-plane diagrams](job_scheduler.md#control-plane-before-and-after) | Matrix row SCHED-001 |
| Before and after / Execution and transfer, `L201-L220` | EXEC-001 [execution diagrams](job_scheduler.md#execution-and-transfer-before-and-after) | Matrix rows EXEC-001/EVID-001 |
| Before and after / Worker lifecycle, `L221-L250` | LIFE-001 [lifecycle diagrams](job_scheduler.md#worker-lifecycle-before-and-after), [WORK-001](job_scheduler.md#work-001), [AUTH-001](job_scheduler.md#auth-001) | Matrix rows WORK-001/LIFE-001/AUTH-001 |
| Internal responsibility list, `L251-L262` | [SCHED-001](job_scheduler.md#sched-001), [STATE-001](job_scheduler.md#state-001), [EXEC-001](job_scheduler.md#exec-001), [EVID-001](job_scheduler.md#evid-001), [COST-001](job_scheduler.md#cost-001) | Matrix rows SCHED-001/STATE-001/EXEC-001/EVID-001/COST-001 |
| Internal runtime/auth boundary, `L263-L267` | [SCHED-001](job_scheduler.md#sched-001), [EXEC-001](job_scheduler.md#exec-001), [CRED-001](job_scheduler.md#cred-001), [AUTH-001](job_scheduler.md#auth-001) | D-009/D-020; matrix rows SCHED-001/EXEC-001/CRED-001/AUTH-001 |
| State and reconciliation, `L268-L291` | [STATE-001](job_scheduler.md#state-001), [COST-001](job_scheduler.md#cost-001), [AUTH-001](job_scheduler.md#auth-001) | D-013; matrix rows STATE-001/COST-001/AUTH-001 |
| Execution and baseline environment, `L292-L306` | [EXEC-001](job_scheduler.md#exec-001) | Matrix row EXEC-001 |
| Secret injection and service credentials, `L307-L312` | [EXEC-001](job_scheduler.md#exec-001), [CRED-001](job_scheduler.md#cred-001) | D-009; matrix rows EXEC-001/CRED-001 |
| Namespace and sink contracts, `L313-L341` | [NS-001](job_scheduler.md#ns-001), [CKPT-001](job_scheduler.md#ckpt-001), [WAND-001](job_scheduler.md#wand-001), [DOC-001](job_scheduler.md#doc-001) | D-002 through D-005, D-016/D-017; matrix rows NS-001/CKPT-001/WAND-001/DOC-001 |
| Checkpoint destination diagrams and open gates, `L342-L409` | [CKPT-001](job_scheduler.md#ckpt-001), [CKPT-OPEN-001](job_scheduler.md#ckpt-open-001) and checkpoint diagrams | D-004 through D-006; matrix rows CKPT-001/CKPT-OPEN-001 |
| Namespace/checkpoint orphan consequences, `L410-L415` | [NS-001](job_scheduler.md#ns-001), [CKPT-001](job_scheduler.md#ckpt-001) | D-002/D-004/D-005; matrix rows NS-001/CKPT-001 |
| Evidence and disposal remainder, `L416-L439` | [EVID-001](job_scheduler.md#evid-001), [LIFE-001](job_scheduler.md#life-001), [CKPT-001](job_scheduler.md#ckpt-001), [NS-001](job_scheduler.md#ns-001), [WAND-001](job_scheduler.md#wand-001), [DOC-001](job_scheduler.md#doc-001) | D-007/D-008/D-016 through D-019; corresponding matrix rows |
| Enrollment, handoff, and provider authority, `L440-L451` | [WORK-001](job_scheduler.md#work-001), [AUTH-001](job_scheduler.md#auth-001) | D-001/D-008; matrix rows WORK-001/AUTH-001 |
| Idle stop, retention, and GC, `L452-L479` | [LIFE-001](job_scheduler.md#life-001), [AUTH-001](job_scheduler.md#auth-001), [STATE-001](job_scheduler.md#state-001), [EVID-001](job_scheduler.md#evid-001) | D-008/D-013; matrix rows LIFE-001/AUTH-001/STATE-001/EVID-001 |
| Provider/storage facts and network-volume decision, `L480-L495` | [RUNPOD-001](job_scheduler.md#runpod-001), [VAST-001](job_scheduler.md#vast-001) | D-010 through D-012B; matrix rows RUNPOD-001/VAST-001 |
| Cost and capacity policy, `L496-L512` | [COST-001](job_scheduler.md#cost-001) | D-001/D-020; matrix row COST-001 |
| Legacy RunPod smoke gate, `L513-L518` | [RUNPOD-001](job_scheduler.md#runpod-001), [AUTH-001](job_scheduler.md#auth-001), [COST-001](job_scheduler.md#cost-001) | D-011/D-020/D-021; matrix rows RUNPOD-001/AUTH-001/COST-001 |
| Worked workflow / generic paid reuse, `L519-L527` | SCHED-001 [worked workflows](job_scheduler.md#cli-contract-and-worked-workflows), [COST-001](job_scheduler.md#cost-001), [LIFE-001](job_scheduler.md#life-001), [AUTH-001](job_scheduler.md#auth-001) | Matrix rows SCHED-001/COST-001/LIFE-001/AUTH-001 |
| Worked workflow / initial RunPod refusal, `L528-L530` | [RUNPOD-001](job_scheduler.md#runpod-001) | D-011/D-012; matrix row RUNPOD-001 |
| Worked workflow / failure versus ambiguity, `L531-L537` | [STATE-001](job_scheduler.md#state-001), [AUTH-001](job_scheduler.md#auth-001), [EVID-001](job_scheduler.md#evid-001) | D-013; matrix rows STATE-001/AUTH-001/EVID-001 |
| Worked workflows / manual worker, `L538-L542` | [WORK-001](job_scheduler.md#work-001) | Matrix row WORK-001 |
| Worked workflow / mixed-provider routing, `L543-L546` | SCHED-001 [worked workflows](job_scheduler.md#cli-contract-and-worked-workflows), [COST-001](job_scheduler.md#cost-001) | Matrix rows SCHED-001/COST-001 |
| Worked workflow / Vast completion scope, `L547-L549` | [VAST-001](job_scheduler.md#vast-001), [AUTH-001](job_scheduler.md#auth-001), [SCHED-001](job_scheduler.md#sched-001) | D-001/D-020/D-021; matrix rows VAST-001/AUTH-001/SCHED-001 |
| Break-glass and rollback, `L550-L570` | [BREAK-001](job_scheduler.md#break-001), [AUTH-001](job_scheduler.md#auth-001), [STATE-001](job_scheduler.md#state-001), [EVID-001](job_scheduler.md#evid-001) | Matrix rows BREAK-001/AUTH-001/STATE-001/EVID-001 |
| Test revision gate before implementation, `L571-L665` | All operative clauses | [Clause-to-proof matrix](job_scheduler_conformance.md#clause-to-proof-matrix) and [Mandatory sequencing](job_scheduler_conformance.md#mandatory-sequencing) |
| Migration dependency graph, `L666-L770` | [OPEN-001](job_scheduler.md#open-001) and linked gates | [Migration dependency graph](job_scheduler_conformance.md#migration-dependency-graph), [Sequencing and migration acceptance](job_scheduler_conformance.md#sequencing-and-migration-acceptance) |
| Acceptance criteria, `L771-L877` | All operative clauses | [Clause-to-proof matrix](job_scheduler_conformance.md#clause-to-proof-matrix) |
| Approval record / 2026-08-14 amendments, `L878-L922` | Linked current clauses | D-001 through D-010 |
| Ratified RunPod replacement-seam answers, `L923-L951` | [RUNPOD-001](job_scheduler.md#runpod-001) | D-011 and exact-gates table |
| Associated decisions and mandatory prerequisites, `L952-L1011` | [REPO-001](job_scheduler.md#repo-001), [NS-001](job_scheduler.md#ns-001), [CKPT-001](job_scheduler.md#ckpt-001), [STATE-001](job_scheduler.md#state-001) | D-002 through D-006, D-012/D-012A through D-015; mandatory sequence |
| Separately recorded namespace decisions / W&B, `L1012-L1080` | [WAND-001](job_scheduler.md#wand-001) | D-016; matrix row WAND-001 |
| Separately recorded namespace decisions / external Docent, `L1081-L1167` | [DOC-001](job_scheduler.md#doc-001) | D-017 through D-019 and D-022; matrix row DOC-001 |
| Paid-integration authorization correction, `L1168-L1206` | [AUTH-001](job_scheduler.md#auth-001), [RUNPOD-001](job_scheduler.md#runpod-001), [VAST-001](job_scheduler.md#vast-001) | D-020/D-021; matrix rows AUTH-001/RUNPOD-001/VAST-001 |

No source span is intentionally unmapped. D-023 is post-snapshot and D-024 comes from governing `AGENTS.md`; neither has
an `8b95cd7` line span.
