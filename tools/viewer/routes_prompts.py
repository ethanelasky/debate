"""/prompts page + GET/POST /api/prompts.

Data model (new-repo prompt schema, see infra/envs/debate/prompts.py): every
*.yaml under the prompts dir is an independent prompt-library file. Entries
resolve PER FILE via resolve_all_experiments(load_config_with_includes(path)),
exactly the pipeline load_prompt_library uses, so what the page shows as
"resolved" is what the scaffold would load. `raw` is each file's own content
(plain yaml.safe_load) — that is what edits round-trip back to disk.

Save semantics (multi-file, ported from the old _save_to_source_files):
- each entry is written back to the file it came from; new entries go to an
  explicitly chosen file (source_file_overrides) or their _extends parent's;
- a `.backup.yaml` sibling is written before any file is rewritten;
- files whose merged content equals what is on disk are NOT touched;
- entries absent from the payload are KEPT (upsert-only: this UI has no
  delete, so absence can only be a client bug — never silent data loss);
- the client sends the GET's per-file mtime tokens back; a token mismatch on
  any file the save would touch is a 409 (R9);
- the per-file merged dict passes through merge_preserving_types before the
  no-op comparison, so JSON transport never retypes YAML dates/keys (R4).

GET isolation (R10): a malformed/unresolvable file is reported per-file in
`errors` instead of failing the whole page. `_includes` containment (R1) is
NOT isolated — an escaping include is a hard 400. Entry keys outside infra's
accepted set are reported per-entry in `invalid` (R8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request

from infra.config import load_config_with_includes, resolve_all_experiments
from infra.envs.debate.prompts import _DIRECT_ENTRY_KEYS, _STAGE_ENTRY_KEYS
from tools.viewer.yamlio import (
    check_includes_containment,
    check_includes_list,
    dump_yaml,
    list_yaml_files,
    make_backup,
    merge_preserving_types,
    mtime_token,
    typed_equal,
)

router = APIRouter(tags=["prompts"])


def _entry_key_problems(resolved: dict[str, Any]) -> dict[str, list[str]]:
    """R8: per-entry keys that load_prompt_library would refuse. Stage-schema
    entries (`slot_stages` present) may additionally name arbitrary cue stages
    referenced by slot_stages, so only keys that are neither known stage keys
    nor named cue stages count as bad there."""
    invalid: dict[str, list[str]] = {}
    for name, cfg in resolved.items():
        if not isinstance(cfg, dict):
            invalid[name] = ["<entry is not a mapping>"]
            continue
        if "slot_stages" in cfg:
            cues = set((cfg.get("slot_stages") or {}).values()) if isinstance(
                cfg.get("slot_stages"), dict
            ) else set()
            bad = sorted(k for k in cfg if k not in _STAGE_ENTRY_KEYS and k not in cues)
        else:
            bad = sorted(k for k in cfg if k not in _DIRECT_ENTRY_KEYS)
        if bad:
            invalid[name] = bad
    return invalid


def _load_prompts_state(prompts_dir: Path) -> dict[str, Any]:
    """Merge every prompt file into one namespace.

    Returns {raw, resolved, source_files, files, mtimes, errors, invalid}:
    raw/resolved keyed by entry name, source_files mapping entry -> filename,
    files the pickable list, mtimes the R9 staleness tokens, errors per-file
    load failures (R10), invalid per-entry key problems (R8). Duplicate entry
    names across files are a 409 — the save path could only keep one copy, so
    we refuse to present a mergeable view.
    """
    raw: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    source_files: dict[str, str] = {}
    errors: dict[str, str] = {}
    mtimes: dict[str, str] = {}
    files = list_yaml_files(prompts_dir)
    for path in files:
        check_includes_containment(path, prompts_dir)  # R1: hard 400, not isolated
        mtimes[path.name] = mtime_token(path)
        try:
            with open(path, encoding="utf-8") as f:
                file_raw = yaml.safe_load(f) or {}
            if not isinstance(file_raw, dict):
                raise ValueError(
                    "top-level YAML must be a mapping of prompt entries, "
                    f"got {type(file_raw).__name__}"
                )
            file_resolved = resolve_all_experiments(load_config_with_includes(path))
        except HTTPException:
            raise
        except Exception as e:  # malformed yaml, unknown _extends parent, ...
            errors[path.name] = str(e)
            continue
        for key, value in file_raw.items():
            if key == "_includes":
                continue
            if key in source_files:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"prompt entry {key!r} is defined in both "
                        f"{source_files[key]} and {path.name}; rename one"
                    ),
                )
            source_files[key] = path.name
            raw[key] = value
        # a file's resolution may also cover entries pulled in via _includes;
        # identical values, so last-wins merging is safe.
        resolved.update(file_resolved)
    return {
        "raw": raw,
        "resolved": resolved,
        "source_files": source_files,
        "files": [p.name for p in files],
        "mtimes": mtimes,
        "errors": errors,
        "invalid": _entry_key_problems(resolved),
    }


@router.get("/prompts")
async def prompts_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "prompts.html", {"active": "prompts"}
    )


@router.get("/api/prompts")
async def api_get_prompts(request: Request):
    return _load_prompts_state(request.app.state.prompts_dir)


@router.post("/api/prompts")
async def api_save_prompts(request: Request, data: dict[str, Any]):
    prompts_dir: Path = request.app.state.prompts_dir
    save_data = data.get("raw")
    if not isinstance(save_data, dict):
        raise HTTPException(status_code=400, detail="payload must be {'raw': {...}}")
    overrides = data.get("source_file_overrides") or {}
    client_mtimes = data.get("mtimes") or {}

    files = list_yaml_files(prompts_dir)
    filenames = {p.name for p in files}
    source_map: dict[str, Path] = {}
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                file_raw = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            continue  # unparsable file: none of its entries are addressable
        if not isinstance(file_raw, dict):
            continue
        for key in file_raw:
            if key == "_includes":
                continue
            if key in source_map and source_map[key] != path:
                # R12: mirror the GET-side duplicate 409 instead of silently
                # writing into whichever file sorts first.
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"prompt entry {key!r} is defined in both "
                        f"{source_map[key].name} and {path.name}; rename one"
                    ),
                )
            source_map[key] = path

    # Group entries by target file: override > current source > parent's file.
    per_file: dict[Path, dict[str, Any]] = {}
    for entry, entry_data in save_data.items():
        if entry == "_includes":
            continue
        if entry in overrides:
            if overrides[entry] not in filenames:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown target file {overrides[entry]!r} for entry {entry!r}",
                )
            target = prompts_dir / overrides[entry]
        elif entry in source_map:
            target = source_map[entry]
        elif isinstance(entry_data, dict) and entry_data.get("_extends") in source_map:
            target = source_map[entry_data["_extends"]]
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"no target file for new entry {entry!r}; "
                    "pass source_file_overrides"
                ),
            )
        per_file.setdefault(target, {})[entry] = entry_data

    # Plan writes first (merge + no-op skip), then validate staleness for the
    # files that would actually be touched (R9), then write.
    to_write: list[tuple[Path, dict[str, Any]]] = []
    for path, entries in per_file.items():
        try:
            with open(path, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400,
                detail=f"cannot save into {path.name}: file on disk is not valid YAML ({e})",
            )
        if not isinstance(existing, dict):
            raise HTTPException(
                status_code=400,
                detail=f"cannot save into {path.name}: top-level YAML is not a mapping",
            )
        merged: dict[str, Any] = {}
        if "_includes" in existing:  # never edited by the UI; preserve as-is
            merged["_includes"] = existing["_includes"]
        for key in existing:
            if key == "_includes":
                continue
            merged[key] = entries.get(key, existing[key])
        for key, value in entries.items():
            if key not in merged:
                merged[key] = value
        merged = merge_preserving_types(merged, existing)  # R4
        # defense-in-depth: never write a file whose _includes would escape
        check_includes_list(merged.get("_includes"), path.parent, prompts_dir, path.name)
        # typed_equal, not ==: Python's True == 1 would swallow an intentional
        # bool<->int retype edit as "no changes".
        if typed_equal(merged, existing):
            continue  # no-op for this file: no backup, no write
        to_write.append((path, merged))

    for path, _ in to_write:
        token = client_mtimes.get(path.name)
        if not token:
            raise HTTPException(
                status_code=400,
                detail=f"missing mtime token for {path.name} — reload the page",
            )
        if str(token) != mtime_token(path):
            raise HTTPException(
                status_code=409,
                detail=f"{path.name} changed on disk — reload before saving",
            )

    backups: list[str] = []
    written: list[str] = []
    new_mtimes: dict[str, str] = {}
    for path, merged in to_write:
        backups.append(str(make_backup(path)))
        dump_yaml(merged, path)
        written.append(path.name)
        new_mtimes[path.name] = mtime_token(path)

    if not written:
        return {"status": "no_changes", "message": "No changes to save — files left untouched."}
    return {
        "status": "success",
        "message": f"Saved {len(written)} file(s): {', '.join(written)}",
        "written": written,
        "backups": backups,
        "mtimes": new_mtimes,
    }
