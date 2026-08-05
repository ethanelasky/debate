# Agent notes

## Watching long-running pod jobs

Prefer a `/loop` check-in (ScheduleWakeup re-firing a status-check prompt) over
Monitor/watcher tasks for tracking training runs, smokes, and diagnostics on
pods. Monitors here have a bad track record: filters miss the actual failure
signature, they time out silently and the expiry gets mistaken for "still
running", and stale monitors from superseded attempts fire confusing events.
A loop iteration that ssh's in, reads the tail of the log, and decides what to
do next is more robust than pattern-matching log lines in advance.

If a monitor is ever genuinely needed (sub-minute reaction to a known exact
string), keep exactly one alive: stop the old one before arming a replacement,
and treat its timeout as a failure of the watch, not of the job.

## Ground rules (ported from ai-debate's AGENTS.md where they transfer)

- Never delete outputs or artifacts without approval — including "temp" ones;
  caches with verified sources are the exception, state it when you clear one.
- Never author or reword prompts/eval definitions yourself; pull them verbatim
  (prompt_config headers mark provenance) or bring options for approval.
- Substantial multi-file implementation is delegated to implementer subagents
  with tightly scoped prompts — exact files, cited decisions, focused tests,
  no commits; the orchestrating session scopes work, adjudicates findings, and
  commits. Small single-file fixes may be done directly. (Softened from
  ai-debate's absolute orchestrator-never-implements rule.)
- A passing test oracle is NOT conformance evidence when the oracle's own
  strength is in question (it was authored alongside the code it gates, or
  drives fakes/synthetic bindings): probe the behavior against the spec with
  independent instruments — real writers, real call paths.
- Delete ephemeral debugging/probe tests once they have served their purpose,
  and periodically review test directories for obsolete, redundant, or bloated coverage.
- Reviewer/verifier subagents verify against the authoritative spec/plan text
  itself (exact file + section, or the governing text pasted verbatim), never
  the orchestrator's paraphrase — paraphrases silently drop or invent clauses.
- Multi-step or high-stakes work gets a smoke test plus a wave of FRESH
  parallel reviewer subagents with distinct lenses (correctness/spec-fidelity,
  scientific semantics, tests/regressions) after each step; fix confirmed
  findings and re-review until the wave is clean. After the LAST step, run
  whole-system waves against the plan document itself — per-step waves miss
  clauses that fall between steps — including one reviewer who reviews INTENT
  rather than letter, from an immutable commit.
- Plan multi-step efforts as a whole arc up front: fan out read-only scoping
  subagents to surface each step's code surface (file:line), dependencies,
  gotchas, and user decisions before implementing.
- Take initiative: maintain a dependency graph of the work (logical deps AND
  file overlap, since parallel implementers must edit disjoint files),
  re-derive the blocked/startable frontier at project start and after every
  milestone, and fan out the maximum set of genuinely independent tasks rather
  than defaulting to a sequential chain.
- Plans that rearchitect a subsystem include a validated before/after Mermaid
  diagram per rearchitected subsystem (conservative constructs; one high-level
  diagram is not enough).
- Judge accuracy is derived from transcripts (winner × side × label), never
  from aggregated result files. Debate is not guaranteed to be zero-sum,
  although competitive debate is guaranteed to be zero-sum.

## Decisionmaking

**System design choices are never finalized without the user understanding
them first.** Before a new abstraction, seam, or object contract is landed —
or an existing one is reshaped — explain it to the user in plain terms (what
the pieces are, what each owns, what crosses each boundary, what it makes
easy vs. hard) and get explicit confirmation. Explanation is part of the
work, not a follow-up. A design the user cannot hold in their head cannot be
verified by them, and every later review, debugging session, and experimental
result then has to be taken on faith — verification cost and complexity blow
up exactly where correctness matters most. When in doubt about whether
something counts as system design, assume it does and check. Prefer
explaining in the user's terms (a walkthrough, a diagram, a worked example of
one object's life) over restating the code. Implementation details inside an
already-understood and approved design continue autonomously.

Reviewers and subagents surface evidence, risks, alternatives, and proposed
decisions; their recommendations are not accepted decisions. Label findings as
confirmed bugs, implementation constraints, optional simplifications, or user
decisions. Do not silently promote a recommendation into the plan or reverse
an approved direction. Changes to architecture, scientific semantics, scope,
or evaluation protocol require the tradeoffs to be presented and Ethan's
decision recorded before implementation. Continue autonomously on reversible
implementation details that stay within an approved direction.

## Pods

Before stopping a pod: evacuate artifacts (adapters, merged checkpoints, logs,
eval JSONLs) with checksum verification; a stopped pod's volume is the deep
backup, not the working copy. Watch for orphaned `VLLM::EngineCore` processes
— `pkill -f vllm` does not match them. Trust only env vars the code actually
reads (grep for them); a misspelled knob fails silently.

## Hardware cost

Any change to model size, batch, sequence length, parallelism or concurrency:
do the arithmetic before proposing it, and say which quantity binds. Memory is
`weights + gradients + optimizer + activations` for training, `weights + KV
cache` for serving; the first three scale with the model, the last two with
batch and sequence. Wall-clock is bounded below by the slowest single item in
any fan-out, and GPU-idle time counts even when the work is on CPU.

Measure the distribution, not the mean, and confirm which limit actually
binds — when raising a knob changes nothing, there is usually a second limit
underneath. Read limits in their own currency (address space is not resident
memory; `free`/`nproc` are the host, not the cgroup).

Working notes and worked numbers: `docs/hardware-model.md`.

## Evaluation protocol

Token budgets derive from the training config (here: the protocol's per-slot
caps), never hardcoded. Eval at temperature 0 on a fixed held-out slice.
Zero transport errors — retry, then invalidate the run; never score a
transport error as wrong. Compare runs with paired per-problem tests
(McNemar), not raw accuracy deltas. Equal-step is not equal-token or
equal-dollar — debate arms train several times more tokens per step, so
report matched curves before claiming a win.

## API keys

Keys live in `.env` (never committed). Never print key values; enumerate
names only when verifying. If a needed key is absent, say so and ask — do not
guess alternate names.

<!-- Deliberately NOT ported from ai-debate's AGENTS.md (specific to that
repo): Obsidian vault protocol, worktree-only rule + main-checkout hook,
RunPod/verl AI_DEBATE_* runbook, serve_and_eval.sh, run.csv schema notes,
docs/agents_reference.md pointers. -->
