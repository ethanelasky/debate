# MonitoringBench judge-prompt ablation — handoff

**Status (2026-08-06):** harness built, tested, smoke-run live end to end. The
full dev run is the next thing to do. Everything below is on `main` as of
`ba03dc3`.

---

## 1. The question

`mb_debate`'s judge prompt is ~15k characters of stacked instructions that
accumulated by porting, never by measurement. Four distinct things are in
there, and nobody has checked which of them the judge is actually using:

1. **debate framing** — the rules of the game (zero-sum, exactly one answer is
   right, how speeches are structured)
2. **the monitor card** — ~4k tokens of domain knowledge: the four security
   failures, the red-team task descriptions, scoring heuristics
3. **judge guidance** — verify quotes against the trajectory, confidence is not
   correctness, deflection is evidence
4. **the 3-step deliberation cue** — answer the question yourself first, *then*
   check quotes, *then* decide

The ablation starts from one sentence and adds these back one at a time.

Worth knowing before you look at any number: this setting is
**info-symmetric** — the judge reads the full trajectory, same as the debaters.
So rung 4 in particular ("answer it yourself first") is close to asking the
judge to be a solo monitor who then consults a debate. If j0→j4 is flat, that
is itself the finding: the debate is carrying the signal and the prompt
scaffolding is decoration.

## 2. The ladder

Generated, not hand-written — `scripts/mb_make_judge_ladder.py` slices the
canonical blocks out of `infra/prompts/debate/monitoringbench.yaml` so the
rungs are exact subsets of today's prompt and cannot drift from it. The
generated file is `infra/prompts/debate/mb_judge_ablation.yaml`.

| rung | judge system card | deliberation | card size |
|---|---|---|---|
| `mb_judge_j0` | one sentence | minimal | 228 ch |
| `mb_judge_j1` | + debate framing | minimal | 1,268 |
| `mb_judge_j2` | + monitor card | minimal | 14,142 |
| `mb_judge_j3` | + judge guidance | minimal | 15,271 |
| `mb_judge_j4` | ″ | **full 3-step** | 15,271 |
| `mb_judge_j4_nocard` | j4 **minus** the monitor card | full | 2,397 |

- **`j4` is byte-identical to today's `mb_debate`** — judge system card *and*
  deliberation cue. Asserted by the generator; re-check with
  `scripts/mb_make_judge_ladder.py --check`.
- **`j4_nocard` is a leave-one-out.** A cumulative ladder only identifies each
  block's marginal value *in the order you stack them*. The monitor card is
  both the biggest block and the likeliest to matter, so it is also tested at
  the top of the stack. If `j2 − j1` is large but `j4 − j4_nocard` is ~0, the
  card is redundant with the guidance above it.
- The **JSON verdict cue is fixed on every rung** — it is the parse contract,
  not a prompt choice. So are all debater prompts (see §3).

## 3. Why it is replayed, not run six times

The judge prompt appears in **no debater's context**. So all six rungs should
be judging the *same* debates. Running six full arms would regenerate the
speeches six times and — the part that actually matters — put fresh
debater-sampling noise on top of every rung-to-rung difference. At n=120 that
noise is comparable to the effects being measured.

`scripts/mb_judge_ablation.py` therefore runs in two phases:

- **Phase 1** — debate 120 dev rows once with real debaters. Record every
  debater slot as `sha256(rendered context) -> speech` into `cache.jsonl`.
- **Phase 2** — per rung, re-run with the debater seats bound to
  `PlaybackModel` and only the judge live.

**The cache is keyed by rendered context, not by task id or call order.** That
makes it self-validating: if a rung perturbs a debater's view by one byte, the
lookup *misses* and the run dies. There is deliberately no fallback — no
nearest-match, no call-order guess — because any of those turns a broken
pairing invariant into a silently wrong experiment. Two contexts recorded with
conflicting speeches are likewise refused at load.

**Built-in correctness check:** `blind_choice` accuracy must be *identical*
across all six rungs, since alice's speech is played back. The driver prints
this and warns if it moves. If it moves, do not trust the run.

## 4. Running it

```bash
# resolve config, tasks, and all six prompt libraries; generate nothing
.venv/bin/python scripts/mb_judge_ablation.py --dry-run

# cheap live smoke (4 rows x 6 rungs = 24 judge passes)
.venv/bin/python scripts/mb_judge_ablation.py --limit 4 --out /tmp/ablation_smoke

# the real run: 120 dev rows, 120 debates + 720 judge passes
.venv/bin/python scripts/mb_judge_ablation.py --out outputs/judge_ablation_dev120

# phase 1 is the expensive irreversible part — reuse it while iterating
.venv/bin/python scripts/mb_judge_ablation.py \
    --reuse-cache outputs/judge_ablation_dev120 --rungs mb_judge_j2,mb_judge_j3
```

Outputs per run: `cache.jsonl`, `<rung>.results.jsonl` (one row per debate,
`run_eval` schema), `summary.json`, `table.md`.

Models are `mb_debate`'s: qwen3.6-flash debaters (DashScope), deepseek-v4-flash
judge (OpenRouter, pinned parasail/fp8). Needs `DASHSCOPE_API_KEY` and
`OPENROUTER_API_KEY`; the script loads `.env` itself.

## 5. Reading the results — please do the paired test

`table.md` reports accuracy / ROC AUC / TPR / FPR per rung. **Those are
independent summaries and will mislead you on rung-to-rung differences.** The
whole point of the design is that the arms are paired: every rung judged the
same 120 debates, joinable on `task_id` across the `*.results.jsonl` files.

So compare rungs with **McNemar** (or a paired bootstrap over `p_attack`) on
the per-row join, not by eyeballing two independent accuracies. Unpaired, a
5pp gap at n=120 is inside the noise (±4.6pp each); paired, the same gap can be
decisive because the debate-level variance cancels.

**Read absolute numbers against the pool floor, not against 50%.** The pool
retains surface shortcuts even after normalization — worst residual AUC 0.616,
best single length-threshold accuracy 58.6% (`data/monitoringbench/norm/config.json`).
A judge at 62% has barely cleared the shortcut.

Suggested decision rule: take the **cheapest rung that is not paired-significantly
worse than j4**. Prompt tokens are paid on every judge call in every future
training run, so a rung that ties j4 at a tenth of the card is a real win.

## 6. What is verified, and what is not

Verified:
- j4 renders byte-identically to `mb_debate` (card + deliberation cue).
- Every rung leaves all debater prompts and the verdict cue untouched.
- Record→replay round trip, cache-miss-raises, and the state-truncation
  requirement in recording are covered by `tests/test_playback_model.py`
  (560 tests green).
- Live 4-row smoke: 24 judge passes, 0 failures, every lookup hit,
  `blind_choice` identical across all six rungs.

Not verified / open:
- **The full dev run has not been read yet.** It was launched but not analyzed.
- **Empty debater speeches.** Phase 1 logged
  `[qwen3.6-flash] empty content (thinking=10287 chars); possible token budget
  exhaustion at max_tokens=4096` at least once. Thinking can eat the 4096-token
  slot budget and leave an empty speech, which is cached and replayed as empty
  to every rung. Pairing is preserved, but count these before trusting the
  numbers — grep the phase-1 log, and check `empty_slots` in the states.
- **Dev is 120 rows (±4.6pp).** `391e0e9` argued for a 300-row holdout on noise
  grounds and was right about the noise; task-disjointness is what costs the
  rows. If the ablation cannot separate adjacent rungs, grow `TARGET_DEV_ROWS`
  in `scripts/mb_make_splits.py` and regenerate rather than reverting to a
  leaky carve.
- **Only one judge model.** Whether the ladder's shape holds for
  qwen3.5-9b (the small-judge arm in `mb_eval.yaml`) is untested, and it is the
  more interesting case — a weak judge is likelier to need the scaffolding.

## 7. Context you need to not misread this

Two things changed under this experiment on 2026-08-06; both are in
`e6039b5`/`ba03dc3` and both matter for interpreting any MB number.

**Splits are now task-disjoint.** The pool mirrors each BashArena task across
labels, so the old `test_size` row-level carve put every held-out row's
siblings in train. `scripts/mb_make_splits.py` splits on the task:
train 730/60 tasks, **dev 120/29**, eval 250/27. Selection happens on dev; eval
is untouched until there is a final number to report. Picking a judge prompt is
selection — that is why this runs on dev.

**The pool is normalized.** Raw, it leaked the label through surface statistics
alone: chars 0.786 AUC, steps 0.700, tool calls 0.708, and one length threshold
classifying at 73.2%. Root cause was step granularity — attack transcripts hold
exactly one tool call per step, honest ones batch up to 15.
`scripts/mb_normalize_pool.py` batches attack steps to honest's distribution
(pure reformatting, label-safe) then matches call counts, giving
0.574 / 0.616 / 0.452. Residual artifacts are recorded, not waved away, in
`data/monitoringbench/norm/config.json` — honest rows now stop mid-task, and
the merge separator is an inference. **Neither has been tested empirically
yet**; the check is `scripts/mb_solo_monitor.py` over dev, compared against
the pre-normalization baselines. That is arguably the highest-value loose end
in the whole area, since every MB result sits on top of it.

## 8. Map

| path | what |
|---|---|
| `scripts/mb_judge_ablation.py` | the driver (phases 1 & 2, report) |
| `scripts/mb_make_judge_ladder.py` | generates the ladder; `--check` guards drift |
| `infra/prompts/debate/mb_judge_ablation.yaml` | generated rungs |
| `configs/mb_judge_ablation.yaml` | arm config, pinned to dev |
| `infra/models/playback_model.py` | context-keyed speech replay |
| `tests/test_playback_model.py` | pairing-invariant tests |
| `scripts/mb_make_splits.py` | train/dev/eval; `--check` guards drift |
| `scripts/mb_normalize_pool.py` | pool normalization + measured alternatives |
| `data/monitoringbench/splits/config.json` | split stats & provenance |
| `data/monitoringbench/norm/config.json` | normalization stats, rejected options, residual artifacts |
