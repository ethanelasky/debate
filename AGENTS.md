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

### jobd emits its own outcome lines (since jobd 7323d71)

The objection above is really about *guessing*: a filter written in advance
against a training script's output misses the failure nobody predicted. For the
narrow question "did the job end, and how", that guess is no longer needed —
jobd logs every terminal state itself, in a fixed format it has a test for, to
`~/.local/state/jobd/daemon.log`:

```
job job_abc123 (lr3e-3) FAILED after attempt 2: exited 7
job job_abc123 (lr3e-3) SUCCEEDED on attempt 1
job job_x (smoke) CANCELLED after attempt 1: cancelled by user
```

Successes are logged deliberately, not just failures: a watcher that only ever
sees failures cannot tell "nothing failed" from "the watch is broken".

So the "known exact string" exception applies here — but **only to whether the
job ended**. It says nothing about whether the run is *healthy*. Loss
diverging, a hung trainer, a wandb run that quietly stopped stepping: all
invisible to this, and a `/loop` that ssh's in and reads the log tail is still
the right tool for those. Use this to know a run died; use a loop to know a run
is going well.

To watch it, tail the log — do not poll the database:

```js
Monitor({
  description: "jobd job outcomes",
  persistent: true,
  command: `tail -n 0 -F "$HOME/.local/state/jobd/daemon.log" \
    | grep -E --line-buffered "SUCCEEDED|FAILED|CANCELLED|TIMEOUT|crash-looping|Traceback|ERROR"`,
})
```

Each flag earns its place: `-n 0` so existing history is not replayed as new
events, `-F` so it survives log rotation, `--line-buffered` or matches sit
unseen in grep's buffer. `persistent: true` matters most — without it the watch
expires on a timeout, and per the warning above that expiry reads as "still
running".

`crash-looping` catches jobd itself dying repeatedly under launchd, where the
process is always present and the log always says "jobd up", so nothing else
distinguishes it from a healthy daemon.

The one-live-monitor rule above still stands: `TaskStop` the old one before
arming a replacement.

**All of this dies with the session.** Close the terminal and jobd keeps
writing the line with nobody reading it. For a run that dies at 3am to reach a
human, something outside the session has to consume that line. Nothing does
yet — that is unbuilt, not merely unconfigured.

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

## Job scheduler (jobd)

New long-running work goes through **jobd** (`~/code/jobd`, state in
`~/.local/state/jobd`) rather than the manual `pod_create.sh` + rsync + idle-stop
dance. It queues runs, holds immutable state, enforces the deadline, retries
what fails, moves inputs and outputs, and retires idle pods. The manual path in
`## Pods` below still describes what happens *on* a pod and remains accurate;
jobd is what gets you there and what owns the lifecycle.

**Always start with `jobd info`** — one JSON document listing profiles, loaded
secrets, the live fleet, the spec shape, and how to watch a run. It exists so
you never rediscover RunPod GPU-type strings, images, disk sizing or ports.

A **profile** is a named environment: GPU type/count, image, container and
volume sizing, ports, credentials to inject, workdir, and the `setup` shell that
rebuilds the training env on a fresh pod. `debate-b200x2` is this repo's:
2x B200, the cluster pytorch image, `HF_TOKEN`/`WANDB_API_KEY`/`DOCENT_API_KEY`
injected by name, repo synced to `/workspace/debate`, and `setup` running
`scripts/env_bootstrap.sh` then verifying `/workspace/envs/verl-b200/bin/python`.

```bash
jobd profiles ls
jobd submit --profile debate-b200x2 -n mathl5-debate-cispo -- \
  bash scripts/pod_run.sh debate mathl5_qwen35_pc_debate_cispo_verl
```

Declare file movement as `inputs`/`outputs`; do not rsync by hand. The scheduler
does it as part of the attempt, so it is reproducible and repeats on a retry.
Default excludes drop `.git`/`outputs`/`wandb`/`.venv` — the difference between
a 154MB sync and a 26GB one.

Dependency graphs are the main reason it exists: `depends_on` in a spec file,
and a job runs only once every dependency has SUCCEEDED. A permanently failed
job marks everything downstream `blocked` instead of leaving it queued.

Semantics worth knowing:

- A **timeout is not retried** (it blew a bound it asked for); a crash or lost
  pod is, with backoff.
- A **failed `setup` fails the attempt**, so a broken env is never mistaken for
  a broken experiment.
- **Missing declared outputs turn success into failure.**
- The daemon holds no process handle — jobs run detached and write their own
  exit code, so restarting the daemon does not disturb a run.
- `POD_IDLE_STOP=0` under jobd: the scheduler owns the deadline, not
  `pod_run.sh`'s watchdog. Do **not** set `DEBATE_SCHEDULER_MODE=1` unless the
  pod actually runs the immutable `/opt/job-scheduler/debate-runtime` image;
  that flag forces a Python interpreter that does not exist on the standard
  cluster image.

Watching — prefer these to Monitor tasks (see the section above) and to ad-hoc
SSH loops, since they read committed scheduler state rather than guessing from
log lines:

```bash
jobd status <job>                  # full attempt history
jobd logs <job> --tail 50
jobd --json events --since <seq>   # append-only; poll from your last seq
```

Spend guard is two numbers: `max_pods` and each job's `timeout_s`. Note the
RunPod account may carry **other people's pods** — `jobd --json pods ls` shows
only jobd's, so check account-level spend before concluding the fleet is idle.

Full spec: `~/code/jobd/README.md`. Global notes: `~/AGENTS.md`.

## Pods

Before stopping a pod: evacuate artifacts (adapters, merged checkpoints, logs,
eval JSONLs) with checksum verification; a stopped pod's volume is the deep
backup, not the working copy. Watch for orphaned `VLLM::EngineCore` processes
— `pkill -f vllm` does not match them. Trust only env vars the code actually
reads (grep for them); a misspelled knob fails silently.

Cold bring-up costs ~80 min, nearly all of it cacheable: the vllm+verl resolve
(~25 min), the flash-attn source build (~45 min), and the weights (~10 min) all
land under `$VOL` and survive on a network volume. Without one, every pod pays
it again. Pass `FLASH_ATTN_WHEEL=` a wheel from a previous pod to skip the build
— there is no published wheel for torch 2.11+cu130, so a previous pod is the
only source, and the wheel is ABI-tied to that exact python/torch/CUDA.

**A network volume does not need a pod to read or write it.** RunPod serves them
over an S3-compatible API at `https://s3api-<datacenter-lowercased>.runpod.io`
(e.g. `s3api-us-ca-2.runpod.io`), with the volume id as the bucket and S3
credentials generated in the console — separate from the regular API key. So use
an S3 client from anywhere: no pod, no GPU, no hourly rate. Renting a GPU to run
`rsync` is paying a GPU rate to move bytes.

This is also how a volume gets seeded or migrated. Volumes are datacenter-local
and not every datacenter offers them, so a GPU available only elsewhere forces a
new region; COPY the built environment across rather than rebuilding it.
Rebuilding costs the resolve, the flash-attn build and a re-download of every
weight, at GPU prices, to re-derive an environment already known to work.

A trainer is not dead because its process is absent: model load plus CUDA graph
capture runs ~8 min before `pod_run.sh` execs it. Require a written step line or
a repeated observation before concluding anything about a run's state.

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
