# Proposer-critic debate format (based on the old repo's run arm)

Source: the old `Math Proposer-Critic Debate Prompt` + its sequential variant
(`prompts/configs/hendrycks_math.yaml`), `turn_plan.py`'s PC branch, and the
`MATH_Debate_Logit_qwen35_9b` config (num_speeches: 2, fresh answer,
judge-logit reward). Strike/edit; implementation follows the surviving rows.

## What the old format actually was (recovered facts)

1. **Fully sequential, every round** — PC ignored `sequential_turns`; the
   proposer opens each round and the critic ALWAYS speaks second seeing the
   proposer's speech, including the final round. **The critic gets the last
   word.** (Deliberate: no blind final round in PC.)
2. **Only the proposer has proposal slots** — two planned (draft P0 + graded
   P1), but the run configs made one live call: the graded fresh proposal.
3. **Judge decided in two steps**: a free-reasoning turn (`post_round_judge`)
   then the strict verdict emission (`post_round_judge_without_reasoning`).
4. **Two verdict dialects existed**: per-participant JSON + `FINAL:` line
   (default arm) and single-winner JSON (sequential variant, logit-scanned on
   the winner letter).
5. **Positions were prompt-injected** (`<proposer-solution><OPPONENT_POSITION>`)
   because the old view machinery didn't carry the proposal in the critic's
   transcript. Sequential-variant grading rule: a proposal with no valid
   \boxed answer = critic wins.

## Proposed topology

```yaml
_topologies:
  pc_2round:
    turns:
      # graded artifact; PUBLIC, so the critic/judge see it in-transcript
      # (replaces the old prompt-injection of the proposal text)
      - alice: [{name: proposal, kind: solution,
                 max_think_tokens: 2048, max_total_tokens: 6144}]
      - bob:   [{name: critique,  max_total_tokens: 1024}]   # sees proposal
      - alice: [{name: defense,   max_total_tokens: 1024}]   # sees critique
      - bob:   [{name: rebuttal,  max_total_tokens: 1024}]   # critic's last word
      - judge: [{name: deliberation, visibility: private, max_total_tokens: 768},
                {name: verdict, kind: decision, max_total_tokens: 256}]
```

Round count scales by `_extends`: `pc_3round` appends a defense/rebuttal pair;
`pc_1round` ends after `critique` (then judge). Opt-ins expressible per
experiment without code: per-seat pre-speech scratchpads
(`{name: scratchpad, visibility: private}` before any speech), judge
questions (a judge turn mid-debate with a `questions` slot).

## Decisions (defaults chosen; strike to change)

| # | decision | old behavior | proposed |
|---|---|---|---|
| 1 | proposal doubles as the opening | separate pre-round proposal + R1 "present and defend" opening speech | **merged**: the public proposal IS the opening (one call fewer; the old opening restated the proposal because their views machinery hid it — ours doesn't) |
| 2 | last word | critic speaks last every round | **kept** — topology ends defense → rebuttal |
| 3 | unparseable proposal (no \boxed) | sequential-variant rule: critic auto-wins | **drop the debate + count it** (current env behavior), with `format_reward` shaping carrying the format pressure. Auto-win is trainable but pollutes the judge-reward channel with a non-judge outcome; revisit if format failures dominate drops |
| 4 | verdict schema | per-participant JSON + FINAL line (default arm); winner-JSON (sequential arm) | **competitive winner-JSON** (`{"winner", "confidence"}`) — matches the newer sequential arm, and our structural decision-token scan subsumes their hand-tuned marker/tokenization comments |
| 5 | judge two-step | reasoning turn then verdict turn | **kept** as `deliberation` (private) + `verdict` slots |
| 6 | position binding | full proposal text injected via `<POSITION>`/`<OPPONENT_POSITION>` everywhere | transcript carries the proposal naturally; `<POSITION>` binds the **extracted answer** only (used in critique/verdict cues for emphasis) |
| 7 | judge blindness | judge never saw debater system cards (sequential variant split them into per-debater system blocks) | free under rule 4 — each speaker has its own system prompt by construction |
| 8 | reward | judge-logit continuous (`MATH_Debate_Logit` arm) | `scoring: continuous`, source configurable `json`/`logit`; both logged regardless |
| 9 | think budgets | soft/absent (first_round_think_budget machinery) | hard per-slot caps (topology above); tune per model |
| 10 | proposal generation context | proposal generated under the debater system card (debate-aware: the proposer knows a forced critic is coming before writing a token) | `first_speech_non_debate_aware: true` (experiment config, default false): the opening solution slot's context is the task source's own messages (`Task.messages`) — byte-identical to the RLVR arm — with no debate framing. The speech still enters the transcript as a public record and its datum trains under judge reward in that solo context, so the debate and RLVR arms share one proposal distribution and differ only in reward channel. Requires fresh positions (an assigned position to defend is inherently debate-aware), a PUBLIC opening solution slot as the topology's FIRST slot, and a task source emitting ≥1 user message. Deliberately incompatible with pre-proposal scratchpads: a debate-framed scratchpad before a non-debate-aware proposal would contaminate the isolation AND be absent from the solo context that follows it. The proposer's later turns keep the debate system card; their own history renders the actual solo user message, not the library's proposal cue |

## Prompt layout (Ethan, 2026-08-03)

Prompt packs use the OLD repo's stage vocabulary: `overall_system`,
`debater_system`, `debater_system_{proposer,critic}`, `judge_system`,
`pre_debate`, `pre_debate_judge`, the grading-details stages, and the
`pre_*`/`post_round_*` cue stages, with a per-pack `slot_stages` map from
topology slot names onto cues (topologies stay declarative). Stages compose
per the old repo's real assembly (verified against its speech_format.py +
rendering goldens): one system message per seat (overall + debater + role
card, or overall + judge), preamble user content for the critic
(`pre_debate` + grading details) and judge (`pre_debate_judge` + grading
details). Two deliberate departures: the PROPOSER gets no preamble — its
opening cue is `<ANSWER_GEN_USER>`, byte-identical to the RLVR arm (the old
repo's `pre_debate` was dead code and its debaters never saw the problem;
we don't reproduce that hole) — and there are no past-tense own-speech
labels. Structural guard: the renderer has no proposer preamble stage at
all, so a pack cannot break the RLVR byte-identity by convention drift.

## Prompts (port the run-verified wording)

Upgrade `math_proposer_critic` in `infra/envs/debate/prompt_configs/hendrycks_math.yaml`
using block lists, carrying over the old arm's battle-tested blocks:

- **system.alice / system.bob**: the old `<critic-role>` FORCED-to-disagree card
  (verbatim: always argue INCORRECT, argue a specific different answer, attack
  load-bearing steps with concrete checks, never concede) and `<proposer-role>`
  card, plus the shared `<answer-vs-reasoning>` block ("a minor slip that does
  not change the answer is not grounds") and the `<grading-rule>` block (only
  the final \boxed counts).
- **system.judge**: the old `<critic-bias>` ("the Critic is FORCED to disagree,
  so their objection alone is NOT evidence"), `<standard-of-proof>` (concrete
  demonstrated error that CHANGES the boxed answer; vague doubt insufficient),
  and `<verify>` (recompute at least one load-bearing step yourself) blocks.
- **slots**: `proposal` (solve + present for a non-expert judge + boxed),
  `critique` (old `pre_opening_speech_critic` wording: attack a specific
  load-bearing step with a CONCRETE check; do NOT solve from scratch),
  `defense` (rebut by re-deriving disputed steps; concede only demonstrated
  error), `rebuttal` (old `pre_speech_critic`: address the latest defense,
  attack ANOTHER load-bearing step, never concede), `deliberation` (the old
  `post_round_judge` task block: verify alleged errors yourself before
  accepting), `verdict` (competitive JSON emission, tie = genuine
  unresolvable draw).

## Config shape (the runnable arm)

`configs/math_pc_debate.yaml` gains the topology above; training block
unchanged (Qwen3.5-4B smoke arm; Qwen3.5-9B real arm mirroring the old
`MATH_Debate_Logit_qwen35_9b`: alice trained, frozen Qwen3-8B-class critic +
judge, judge temp/top_p 1.0, `confidence_source: logit` for the logit arm and
`json` for the elicited arm — the two arms the old experiment compared, now a
one-line diff).
