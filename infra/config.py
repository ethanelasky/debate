"""Experiment YAML loading: _includes / _extends / _preset resolution.

Salvaged from ~/ai-debate experiments/config_utils.py + utils/dict_utils.py,
near-verbatim — the YAML mechanics are the contract (existing experiment
files must load unchanged). The old repo's 900-line ExperimentConfig schema is
deliberately NOT ported; experiments resolve to plain dicts, and the typed
surface is limited to what the scaffold consumes (model_settings blocks).

YAML shape:
    _includes: [base_config.yaml]        # multi-file composition
    _models: {preset_name: {...}}        # model presets
    _some_template:                      # names starting with _ = templates
      agents:
        debaters:
        - model_settings: {_preset: name, ...overrides}
    my_experiment:
      _extends: _some_template           # chained inheritance
      ...child overrides (dicts merge recursively, lists merge BY INDEX)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

from infra.models.base import ModelSettings


def deep_merge(base: dict, override: dict) -> dict:
    """Override wins; nested dicts merge recursively; lists merge BY INDEX
    (a child can override one field of debaters[0] without redeclaring it)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            merged = copy.deepcopy(result[key])
            for i, item in enumerate(value):
                if i < len(merged):
                    if isinstance(merged[i], dict) and isinstance(item, dict):
                        merged[i] = deep_merge(merged[i], item)
                    else:
                        merged[i] = copy.deepcopy(item)
                else:
                    merged.append(copy.deepcopy(item))
            result[key] = merged
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_with_includes(config_path: str | Path, _loaded: set[str] | None = None) -> dict:
    """Load a YAML file, recursively merging its _includes (later includes
    override earlier; the main file overrides all includes). Two SIBLING
    includes defining the same top-level key is an error — that silent
    deep-merge is how experiments get accidentally redefined."""
    config_path = Path(config_path)
    _loaded = _loaded if _loaded is not None else set()
    abs_path = str(config_path.resolve())
    if abs_path in _loaded:
        return {}
    _loaded.add(abs_path)

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    includes = data.pop("_includes", [])
    if not isinstance(includes, list):
        includes = [includes]

    merged: dict = {}
    key_source: dict[str, str] = {}
    for include_path in includes:
        if not Path(include_path).is_absolute():
            include_path = config_path.parent / include_path
        included = load_config_with_includes(include_path, _loaded)
        collisions = sorted(set(key_source) & set(included))
        if collisions:
            raise ValueError(
                f"Duplicate key(s) across sibling includes of {config_path}: "
                f"{', '.join(collisions[:8])} — first from {key_source[collisions[0]]!r}, "
                f"also from {str(include_path)!r}. Rename one."
            )
        for key in included:
            key_source[key] = str(include_path)
        merged = deep_merge(merged, included)

    return deep_merge(merged, data)


def _resolve_model_presets(config: Any, models: dict) -> Any:
    if not isinstance(config, dict):
        return config
    result = {}
    for key, value in config.items():
        if key == "model_settings" and isinstance(value, dict) and "_preset" in value:
            preset = models.get(value["_preset"], {})
            explicit = {k: v for k, v in value.items() if k != "_preset"}
            result[key] = deep_merge(preset, explicit)
        elif isinstance(value, dict):
            result[key] = _resolve_model_presets(value, models)
        elif isinstance(value, list):
            result[key] = [
                _resolve_model_presets(item, models) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


def _resolve_inheritance(name: str, data: dict, cache: dict) -> dict:
    if name in cache:
        return cache[name]
    config = data.get(name)
    if not isinstance(config, dict):
        return config
    if "_extends" in config:
        parent = config["_extends"]
        if parent not in data:
            raise ValueError(f"{name!r} extends unknown parent {parent!r}")
        parent_resolved = _resolve_inheritance(parent, data, cache)
        resolved = deep_merge(parent_resolved, {k: v for k, v in config.items() if k != "_extends"})
    else:
        resolved = copy.deepcopy(config)
    cache[name] = resolved
    return resolved


def resolve_all_experiments(data: dict) -> dict:
    """Apply _extends chains and expand _preset references for every
    experiment in a loaded config dict. Presets expand AFTER inheritance so a
    child's _preset choice overrides its parent's."""
    if not data:
        return {}
    models = data.get("_models", {})
    inheritance_cache: dict[str, dict] = {}
    result = {}
    for name, config in data.items():
        if name == "_models":
            continue
        if not isinstance(config, dict):
            result[name] = config
            continue
        resolved = _resolve_inheritance(name, data, inheritance_cache)
        result[name] = _resolve_model_presets(resolved, models)
    return result


def resolve_experiments_from_file(config_path: str | Path) -> dict:
    return resolve_all_experiments(load_config_with_includes(config_path))


def runnable_experiments(data: dict) -> list[str]:
    """Names not starting with '_' (templates aren't directly runnable)."""
    return [n for n in data if not n.startswith("_") and isinstance(data.get(n), dict)]


def load_experiment(config_path: str | Path, name: str) -> dict:
    """The one-call path: file + experiment name -> fully resolved dict."""
    experiments = resolve_experiments_from_file(config_path)
    if name not in experiments:
        available = ", ".join(runnable_experiments(experiments)) or "<none>"
        raise KeyError(f"experiment {name!r} not in {config_path} (available: {available})")
    return experiments[name]


# --------------------------------------------------------- typed accessors


def parse_model_settings(block: dict) -> ModelSettings:
    return ModelSettings(**block)


def agent_model_settings(experiment: dict) -> dict[str, ModelSettings]:
    """Extract model settings from an experiment's agents section.

    Returns {"debater_0": ..., "debater_1": ..., "judge": ...} for whichever
    slots the experiment defines.
    """
    agents = experiment.get("agents", {}) or {}
    out: dict[str, ModelSettings] = {}
    for i, debater in enumerate(agents.get("debaters", []) or []):
        if isinstance(debater, dict) and "model_settings" in debater:
            out[f"debater_{i}"] = parse_model_settings(debater["model_settings"])
    judge = agents.get("judge")
    if isinstance(judge, dict) and "model_settings" in judge:
        out["judge"] = parse_model_settings(judge["model_settings"])
    return out


def build_agent_models(experiment: dict, binding: str = "eval") -> dict[str, Optional[object]]:
    """Instantiate a Model for each agent slot (None for slots whose type
    isn't directly instantiable, e.g. offline/human)."""
    from infra.models.factory import instantiate_model

    return {
        slot: instantiate_model(settings, is_debater=not slot.startswith("judge"), binding=binding)
        for slot, settings in agent_model_settings(experiment).items()
    }
