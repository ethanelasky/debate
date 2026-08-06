"""/experiments page + file-list/GET/POST endpoints for configs/*.yaml.

Raw vs resolved: `raw` is the file's own yaml.safe_load (the save/round-trip
basis and what marks a field as an explicit override); `resolved` is
resolve_all_experiments(load_config_with_includes(path)) — _includes merged,
_extends chains applied, and _models/_preset expanded (preset expansion
happens INSIDE resolve_all_experiments, after inheritance). `models` comes
from the post-includes dict so presets defined in an included file still show.

Save contract: POST body is {"raw": <whole file>, "mtime": <token from GET>}.
The mtime token 409s if the file changed on disk (R9); the payload is merged
through merge_preserving_types so JSON transport never retypes untouched YAML
dates/keys (R4); the no-op guard compares AFTER that merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request

from infra.config import load_config_with_includes, resolve_all_experiments
from tools.viewer.yamlio import (
    check_includes_containment,
    check_includes_list,
    dump_yaml,
    list_yaml_files,
    make_backup,
    merge_preserving_types,
    mtime_token,
    resolve_yaml_file,
    typed_equal,
)

router = APIRouter(tags=["experiments"])


def _read_yaml_mapping(path: Path) -> dict:
    """safe_load with clean 400s: malformed YAML and non-mapping top levels
    must never surface as traceback 500s (R10)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"{path.name} is not valid YAML: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{path.name}: top-level YAML must be a mapping of experiments, "
                f"got {type(data).__name__}"
            ),
        )
    return data


@router.get("/experiments")
async def experiments_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "experiments.html", {"active": "experiments"}
    )


@router.get("/api/experiments/files")
async def api_experiment_files(request: Request):
    files = [p.name for p in list_yaml_files(request.app.state.configs_dir)]
    return {"files": files, "default": files[0] if files else None}


@router.get("/api/experiments/file/{filename}")
async def api_get_experiment_config(request: Request, filename: str):
    configs_dir: Path = request.app.state.configs_dir
    path = resolve_yaml_file(configs_dir, filename)
    check_includes_containment(path, configs_dir)  # R1: 400 before any include load
    token = mtime_token(path)  # stat BEFORE read: stale token beats silent overwrite
    raw = _read_yaml_mapping(path)
    try:
        loaded = load_config_with_includes(path)
        resolved = resolve_all_experiments(loaded)
    except Exception as e:  # unknown _extends parent, include collision, ...
        raise HTTPException(status_code=400, detail=f"error resolving {filename}: {e}")
    return {
        "raw": raw,
        "resolved": resolved,
        "models": loaded.get("_models", {}),
        "mtime": token,
    }


@router.post("/api/experiments/file/{filename}")
async def api_save_experiment_config(request: Request, filename: str, data: dict[str, Any]):
    path: Path = resolve_yaml_file(request.app.state.configs_dir, filename)
    incoming = data.get("raw")
    if not isinstance(incoming, dict):
        raise HTTPException(
            status_code=400, detail="payload must be {'raw': {...}, 'mtime': <token>}"
        )
    token = data.get("mtime")
    if not token:
        raise HTTPException(status_code=400, detail="missing mtime token — reload the file")
    if str(token) != mtime_token(path):
        raise HTTPException(
            status_code=409, detail=f"{filename} changed on disk — reload before saving"
        )
    existing = _read_yaml_mapping(path)
    merged = merge_preserving_types(incoming, existing)  # R4
    # defense-in-depth: a save must not INTRODUCE an escaping _includes either
    check_includes_list(merged.get("_includes"), path.parent, request.app.state.configs_dir, filename)
    # typed_equal, not ==: Python's True == 1 would swallow an intentional
    # bool<->int retype edit as "no changes".
    if typed_equal(merged, existing):
        return {"status": "no_changes", "message": "No changes to save — file left untouched."}
    backup = make_backup(path)
    dump_yaml(merged, path)
    return {
        "status": "success",
        "message": f"Saved to {filename}",
        "backup": str(backup),
        "mtime": mtime_token(path),
    }
