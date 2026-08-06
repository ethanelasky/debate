# Prompt organization

Two directories, one rule: **`tasks/` owns how a solo model answers the task;
`debate/` owns everything debate-specific.** Wording that both arms render
lives in `tasks/` exactly once, and the debate packs splice it in by
reference — never by restating it.

## `tasks/<family>.yaml` — the solo answer-generation context

One file per task family (`math`, `math_nocot`, `codecontests`,
`monitoringbench`), one uniform schema:

```yaml
format_notes: |-        # optional; substituted wherever <FORMAT_NOTES> appears
  ...
messages:               # the solo context, in order
  - role: system
    content: |-
      ...
  - role: user
    name: ANSWER_GEN_USER   # optional: expose this message for pack splicing
    content: |-
      ... <PROBLEM> ...
```

These messages are the **RLVR prompt verbatim** — the family's task source
(`infra/envs/tasks/<family>.py`) renders them into every `Task.messages`, and
`run_rlvr` trains on them directly. Select an alternate file per experiment
with `dataset: {prompt_file: infra/prompts/tasks/<variant>.yaml}` (how
`math_nocot` is wired).

The row's content binds through a placeholder the family owns:

- `<PROBLEM>` — "the question": short text, also carried in
  `meta["question"]`, logged in result rows, always shown to the judge
  (math, codecontests: the problem statement).
- `<BACKGROUND_TEXT>` — "the document": long text bound per-seat via
  `meta["bindings"]`, deliberately kept OUT of `meta["question"]` and logs
  (monitoringbench: the trajectory; MB's safety rule forbids trajectory
  content in any output). A document-shaped family puts the document in a
  message of its own so the debate layer can reuse it as each seat's shared
  preamble.

These are distinct concepts, not synonyms — a QuALITY-style family has both
(story = background, question = topic), so they are never merged.

MB's "answer generation" is Alice's blind side choice: shown the trajectory
alone — no debate framing, no assigned side — she commits to attack vs benign
with an `<answer>` tag. That solo context is `tasks/monitoringbench.yaml`.

## `debate/<family>.yaml` — the debate packs

Role system cards, preambles, per-turn cues, attribution templates — the
schema is documented in `infra/envs/debate/prompts.py`. Anything the solo
context already says is spliced, not restated.

## The splice (single-sourcing across arms)

A `messages:` item with a `name:` (registered in
`infra/envs/task_prompts.py::TASK_SUPPLIED_TEMPLATES`) becomes available to
debate packs as a template placeholder. `DebateEnv` substitutes it at
construction, before rendering, and **fails eagerly** if a pack references a
name its task source does not supply. Current names:

- `<ANSWER_GEN_USER>` — the answer-eliciting user message. Math/CC packs make
  it the `proposal` slot cue; MB choice packs make it the `blind_assessment`
  cue. Spliced content may carry `<PROBLEM>`, which is rebound to the debate
  layer's `<TOPIC>`.
- `<TRAJECTORY_USER>` — the document message. MB packs make it the shared
  `pre_debate` preamble every seat opens with.

This is what guarantees, **by construction**, that for the same row:

1. the RLVR prompt,
2. the choice arms' blind first turn (`first_speech_non_debate_aware` renders
   `Task.messages` verbatim),
3. the cue that replays before the chooser's answer in her later debate
   views, and
4. every seat's trajectory preamble (MB)

are byte-identical. Editing wording in `tasks/` changes every arm together —
which is the point. Tests pin the invariants
(`tests/test_tasks.py::test_math_debate_proposal_slot_equals_rlvr_user_message`,
`tests/test_codecontests.py::test_debate_proposal_slot_equals_rlvr_user_message`,
`tests/test_choice_mode.py::test_rlvr_task_source_prompt_matches_eval_arm`).

## Adding a family

1. Write `tasks/<family>.yaml` (messages list; name the messages packs will
   splice).
2. The family source loads it via `load_generation_prompts(...,
   require_placeholder=...)`, exposes it as `self.prompts`, and builds
   `Task.messages` with `prompts.render({...})`.
3. Debate packs reference the named messages instead of restating them.
