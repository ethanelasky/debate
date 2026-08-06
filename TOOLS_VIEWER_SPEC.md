# Config Viewer port spec (prompts + experiment configs)

Date: 2026-08-03. Orchestrator-authored contract for porting two pages of the old
transcript viewer into `~/debate`. Implementers and reviewers work from THIS file
plus the verbatim source files listed below. Nothing else from the transcript
viewer is in scope — no transcripts, datasets, review queue, chat, evaluation,
Docent, or dataset-management surfaces.

## Motivation (user, 2026-08-03)

Experiments often diverge in only one or two fields; when a config value lives in
two places it is easy to edit one and forget the other, silently un-controlling
an experiment and wasting a paid run. A tabular side-by-side view of configs —
like the prompts editor's comparison table — catches this. The user asked to port
(1) the prompts viewer/editor and (2) the experiment config viewer/editor, and
confirmed **editing stays enabled** (save-back with backups).

## Verbatim sources (copied to scratchpad ref dir)

`/private/tmp/claude-503/-Users-cankucukkurt/b10da993-81ec-476d-b7bb-d6f0ebdfb006/scratchpad/viewer_port_ref/`

| File | Provenance |
|---|---|
| `prompts_routes.py` | old worktree `ai_debate/tools/transcript_viewer/routes/prompts.py` (current) |
| `prompts.html` | old worktree `…/transcript_viewer/templates/prompts.html` (current, 2137 lines) |
| `prompts-config.js` | old worktree `…/static/js/prompts-config.js` (current) |
| `experiment_config.py` | **recovered from git** `54c1562c7^` (deleted by "drop … experiment-runner UI") |
| `experiment_editor.html` | **recovered from git** `54c1562c7^` (1309 lines) |
| `base_current.html`, `base_at_deletion.html`, `main.css`, `template_utils.py` | chrome/reference |

The old repo (read-only) is at `~/ai-debate-monitoring-bench.worktrees/mb-debate-integration`
for any further reference (`git show 54c1562c7^:<path>` recovers deleted files).

## Target layout (all NEW files; `infra/` is READ-ONLY — import from it, never patch it)

```
tools/viewer/
  __init__.py
  __main__.py            # python -m tools.viewer [--port 8080] [--no-browser]
  server.py              # FastAPI app; mounts static; includes the two routers
  routes_prompts.py      # /prompts page + GET/POST /api/prompts
  routes_experiments.py  # /experiments page + file-list/GET/POST endpoints
  templates/  base.html, prompts.html, experiments.html
  static/     css/viewer.css, js/… (whatever the port needs)
tests/test_viewer.py     # FastAPI TestClient tests, synthetic fixtures ONLY
```

`pyproject.toml`: add optional extra `viewer = ["fastapi>=0.110", "uvicorn>=0.29"]`
(jinja2 + pyyaml already in core deps). Nothing added to core dependencies.

## Data sources in the NEW repo

- **Prompts page** targets `infra/envs/debate/prompt_configs/*.yaml`.
  Entry schema per name: `{vars, system, slots, user_preamble, attribution}`
  (+ `_extends`). See `infra/envs/debate/prompts.py::load_prompt_library` — it
  resolves via `resolve_all_experiments(load_config_with_includes(path))`.
- **Experiments page** targets `configs/*.yaml`. Resolution via
  `infra.config.load_config_with_includes` + `resolve_all_experiments`
  (handles `_extends`, `_includes`, `_models`/`_preset`; lists merge BY INDEX).
  Implementer must READ `infra/config.py` and reuse its public functions;
  determine whether preset resolution happens inside `resolve_all_experiments`
  or needs an extra call, and show BOTH raw and resolved faithfully.
- Multi-file support both pages: file picker (experiments) / per-name source-file
  map (prompts, as in old `_build_source_file_map`). Skip `*.backup.yaml`.

## REVISION (user decision, 2026-08-03): rewrite frontend, don't port it

The frontend is a FRESH REWRITE, not a port: do not carry over the old
templates' markup, CSS, or JS. The reference files (`prompts.html`,
`experiment_editor.html`, `prompts-config.js`, `main.css`, `base_*.html`) are
INTERACTION-DESIGN REFERENCE ONLY — consult them for behavior semantics (what
the diff mode does, how inherited values read, the save flow), never copy code
from them. Design goals for the new UI: clean, modern, minimal; system font
stack + CSS variables; vanilla JS, no build step, no external CDNs or web fonts
(must work fully offline); readable on a laptop screen with many columns
(sticky first column / horizontal scroll). Every FUNCTIONAL requirement below
still applies verbatim — the list of behaviors is the contract; only their
implementation and visual design are new.

## Functional port requirements

1. **Prompts page** — adapt the old comparison table to the NEW schema: columns =
   prompt-library entries (e.g. `mb_debate`, `mb_double_consultancy`, …); rows =
   components flattened from `system.<speaker>`, `slots.<slot>.<speaker>`,
   `vars.<name>`, `user_preamble.<speaker>`, `attribution.<reader>.<author>`.
   Keep: entry/component selection with localStorage persistence, inherited-value
   visual distinction (raw vs resolved tells you what's an override), the DIFF
   MODE when exactly two entries are selected (LCS-based, expand/collapse — port
   from prompts.html), unsaved-changes indicator, save. The old "Reusable
   Prompts" section and the "New Prompt" modal's old-schema modes may be dropped
   or adapted minimally — old-schema-only machinery must not survive as dead code.
2. **Experiments page** — port `experiment_editor.html` + `experiment_config.py`:
   experiments as columns, flattened (nested) fields as rows, inherited styling
   from raw-vs-resolved, `_models` presets visible, file picker. This page's whole
   point is spotting single-field divergence between near-identical experiments —
   preserve that reading experience; cosmetic simplification is fine.
3. **Editing** (user decision: keep): POST writes back to the SOURCE file with a
   `.backup.yaml` sibling written first, LiteralDumper (multiline → `|` blocks),
   `sort_keys=False`. Save requirements:
   - Never write when nothing changed.
   - Confirmation dialog before save stating: "Saving rewrites the whole file:
     YAML comments are dropped and formatting is normalized." (True: the pipeline
     is `yaml.safe_load` → `yaml.dump`.) This matters because
     `prompt_configs/monitoringbench.yaml` carries byte-verbatim-ported prompt
     text — content survives, formatting/comments do not.
   - Prompts page save path must preserve the multi-file source mapping (write
     each entry back to its own file, as old `_save_to_source_files` did).
4. **Chrome**: minimal `base.html` (title + nav with exactly two links:
   Prompts, Experiments). Port only the CSS actually used.
5. **Server**: FastAPI + uvicorn, default port 8080, `--no-browser` to skip
   auto-open. No other routes.

## Hard constraints (safety + repo rules)

- The tool reads ONLY yaml config files under `configs/` and
  `infra/envs/debate/prompt_configs/`. It must have NO code path that opens
  anything under `data/` (MonitoringBench attack trajectories live there —
  standing rule: never read/print their contents).
- Tests: synthetic tmp-dir YAML fixtures only; monkeypatch/parameterize the
  config dirs; never load the real MB yaml in tests (it's fine for the running
  tool, not for test output). Tests must pass from a fresh clone (no gitignored
  files required).
- Do not author or reword any prompt content anywhere in this work.
- NO git commits (standing rule until user approves).
- `infra/` untouched. If a helper is missing, implement it inside `tools/viewer/`.

## FIX ROUND 1 (adjudicated 2026-08-03 from the 3-lens review wave)

All items below are REQUIRED unless marked defer/document. Findings dedup'd
across the three reviews; lens evidence lives in the review transcripts.

R1. **`_includes` containment (hard-constraint fix, both pages).** Before calling
    `load_config_with_includes`, pre-read the file's `_includes` list, resolve
    each path (relative to the file; absolute as-is), and reject the request
    with a 400 naming the offending include unless the resolved path's parent
    IS the configured dir (after `resolve()` on both). This closes the
    demonstrated read+HTTP-leak of arbitrary files (incl. data/). Implement in
    tools/viewer only — infra stays untouched.
R2. **Symlink containment.** `yamlio.list_yaml_files` must skip symlinks;
    `resolve_yaml_file` must require `path.resolve().parent ==
    directory.resolve()` (blocks read AND write through symlinks).
R3. **Atomic writes.** Dump to `<path>.tmp` then `os.replace`; a mid-dump
    failure must leave the original bytes intact (reviewer demonstrated a 0-byte
    truncation). All file I/O gets explicit `encoding="utf-8"`.
R4. **Typed-scalar preservation (dates etc.).** JSON transport retypes YAML
    dates/timestamps and non-string keys; the no-op guard then misfires and a
    save silently retypes them. Fix with a recursive retype-preserving merge on
    save (both pages): for each leaf where the incoming JSON value equals the
    JSON-serialization of the existing typed value, keep the existing typed
    value; the no-op guard compares after this merge. Add a targeted test with
    a bare date: untouched GET→POST must be a refused no-op, and an edit
    elsewhere must not retype the date.
R5. **Mixed-type rows (experiments table).** When selected experiments disagree
    on shape at a path (dict/list vs scalar), the scalar side must still get a
    row (render at the parent path) instead of disappearing; editing that cell
    must not throw. This is core to the tool's divergence-spotting purpose.
R6. **Split `_models` preset editing.** Gate editability PER PRESET on
    `presetName in raw._models`; included-only presets render read-only with
    the existing note; do not drop included presets from the `_preset` dropdown
    source. First-keystroke TypeError must be gone.
R7. **Stale "inherited" after clearing.** Clearing a leaf with no parent value
    must remove the resolved leaf too (no stale value styled as inherited).
R8. **Entry-key validation (prompts).** Server-side, flag entry keys outside
    infra's accepted set (`_ENTRY_KEYS` in infra/envs/debate/prompts.py —
    import it; fall back to a mirrored constant with a comment if import is
    unreasonable) and return per-entry validity in the payload; UI shows an
    error badge on invalid entries. The viewer must not present an entry as
    healthy that `load_prompt_library` would refuse.
R9. **Staleness guard on save.** GET returns a per-file mtime token; POST
    requires it and 409s on mismatch ("file changed on disk — reload"). Both
    pages (prompts: per touched file). Closes the silent lost-update between
    two tabs / an external editor.
R10. **Per-file error isolation (prompts page).** A malformed/unresolvable file
    must not 500 the whole page: return per-file errors alongside healthy
    files' data; UI lists broken files. Experiments GET of a malformed file
    returns a clean 4xx/JSON error, not a traceback 500. Non-dict top-level
    YAML gets a sensible message.
R11. **Behavioral guard tests** replacing the near-vacuous greps: (a) fixture
    config with `_includes: ['../outside/x.yaml']` → 400 on both pages;
    (b) symlinked yaml in the dir is neither listed nor readable/writable;
    (c) atomic-write test (patched dump failure leaves original intact);
    (d) date no-op test per R4; (e) stale-mtime 409 test per R9. Keep the greps
    as a cheap extra layer if desired.
R12. Small confirmed UX defects: diff mode requires exactly two SELECTED
    entries (no fallback-to-all trigger); fully-stale localStorage selection
    auto-selects the first entry instead of rendering empty; per-line char-diff
    capped by line length (avoid the ~400MB LCS allocation on long single
    lines); `parseValue` respects the reference type (no bool coercion into
    string-typed fields); POST-side duplicate-entry detection mirrors the GET
    409 instead of silent first-wins; deleting an include-provided experiment
    surfaces "defined in an include — cannot delete here" instead of a silent
    no-op.

Deferred / documented, NOT in this round: in-session recompute of children's
resolved views after editing a parent (would duplicate infra resolution
client-side; save+reload is the refresh path — add a one-line hint in the UI
when an edited entry is extended by others); main-file-overriding-an-include
layout still 409s (legal for infra; no current config uses `_includes`);
single rolling backup slot (spec-compliant; atomic writes reduce the risk it
covered); `tools` as a generic top-level package name (repo-owner call);
`tiktoken` in core deps belongs to the MB round work, not the viewer.

## Acceptance

- `pip install -e ".[viewer]"` then `python -m tools.viewer` serves both pages;
  both real config families render (manual check by orchestrator).
- Raw-vs-resolved/inheritance display verified against a hand-built `_extends`
  chain in tests; save round-trip test proves content equality (`safe_load(old)
  == safe_load(new)` modulo the edit) + backup file created.
- Full repo `pytest` still green (2 known pre-existing torch failures in
  test_kl.py excepted).
