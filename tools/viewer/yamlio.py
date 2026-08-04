"""Shared YAML I/O for the viewer: literal-block dumping, backups, file listing,
containment checks, typed-scalar-preserving merge, mtime tokens.

The save pipeline is yaml.safe_load -> yaml.dump: CONTENT survives a round-trip
exactly, YAML comments and hand formatting do not. The UI warns before every
save; the routes refuse to write when nothing changed.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

_YAML_SUFFIXES = (".yaml", ".yml")


def _str_representer(dumper: yaml.Dumper, value: str) -> yaml.Node:
    """Multiline strings dump as | literal blocks; everything else default."""
    if "\n" in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


class LiteralDumper(yaml.SafeDumper):
    """SafeDumper that renders multiline strings as literal blocks."""


LiteralDumper.add_representer(str, _str_representer)


def dump_yaml(data: dict, path: Path) -> None:
    """Atomic write (R3): dump to `<name>.tmp` in the same directory, then
    os.replace over the target. A mid-dump failure leaves the original bytes
    intact and removes the temp file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                Dumper=LiteralDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=1000,
            )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_backup(path: Path) -> Path:
    """Write a `<stem>.backup<suffix>` sibling with the file's current bytes."""
    backup = path.parent / f"{path.stem}.backup{path.suffix}"
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def list_yaml_files(directory: Path) -> list[Path]:
    """Sorted *.yaml / *.yml files in `directory`, skipping *.backup.* files
    and symlinks (R2: no reads or writes through links)."""
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.is_symlink()
        and p.suffix in _YAML_SUFFIXES
        and ".backup" not in p.name
    )


def resolve_yaml_file(directory: Path, filename: str) -> Path:
    """Validate a client-supplied filename to an EXISTING yaml file directly
    inside `directory` (basename only: no traversal, no backups, no symlinks)."""
    if (
        filename != Path(filename).name
        or ".backup" in filename
        or Path(filename).suffix not in _YAML_SUFFIXES
    ):
        raise HTTPException(status_code=400, detail=f"invalid config filename: {filename!r}")
    path = directory / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    # R2: block reads AND writes through symlinks / anything resolving outside
    # the configured directory.
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise HTTPException(
            status_code=400,
            detail=f"{filename} resolves outside the config directory (symlink?)",
        )
    return path


def check_includes_list(includes: Any, base_dir: Path, directory: Path, label: str) -> None:
    """R1 for an in-memory `_includes` value: 400 unless every entry resolves
    to a file directly inside `directory` (relative entries resolve against
    `base_dir`). Used both when reading files and, defense-in-depth, on the
    payload a save is about to write."""
    if includes is None:
        return
    if not isinstance(includes, list):
        includes = [includes]
    directory = directory.resolve()
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = base_dir / include_path
        if include_path.resolve().parent != directory:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"include {include!r} in {label} resolves outside "
                    "the configured directory — refusing"
                ),
            )


def check_includes_containment(path: Path, directory: Path) -> None:
    """R1: reject (400) any `_includes` entry that does not resolve to a file
    directly inside `directory`. Walks includes-of-includes too — every file
    it recurses into has itself passed the containment check first. Files it
    cannot parse are skipped here (the loading path reports those errors)."""
    directory = directory.resolve()
    seen: set[Path] = set()
    stack: list[Path] = [path]
    while stack:
        current = stack.pop()
        current_resolved = current.resolve()
        if current_resolved in seen:
            continue
        seen.add(current_resolved)
        try:
            with open(current, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            continue  # unreadable/unparsable: nothing was included from it
        if not isinstance(data, dict):
            continue
        includes = data.get("_includes", [])
        if not isinstance(includes, list):
            includes = [includes]
        check_includes_list(includes, current.parent, directory, current.name)
        for include in includes:
            include_path = Path(str(include))
            if not include_path.is_absolute():
                include_path = current.parent / include_path
            stack.append(include_path)


def mtime_token(path: Path) -> str:
    """R9 staleness token: the file's st_mtime_ns as a string."""
    return str(path.stat().st_mtime_ns)


# ---------------------------------------------------------------------------
# R4: typed-scalar preservation across the JSON transport.
#
# "JSON-serialization" here means exactly what the GET response contained:
# dates/datetimes/times become their isoformat() strings, non-string dict keys
# become the string json.dumps would emit for them (1 -> "1", True -> "true",
# None -> "null"), and str/int/float/bool/None pass through unchanged.


def _json_image(value: Any) -> Any:
    """The JSON-transported form of a typed Python value."""
    if isinstance(value, dict):
        return {_json_key(k): _json_image(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_image(v) for v in value]
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)  # conservative fallback (yaml binary etc.)


def _json_key(key: Any) -> str:
    """The string a JSON response uses for a (possibly non-string) dict key."""
    image = _json_image(key)
    if isinstance(image, str):
        return image
    if image is True:
        return "true"
    if image is False:
        return "false"
    if image is None:
        return "null"
    return json.dumps(image)  # int/float: matches json.dumps key coercion


def merge_preserving_types(incoming: Any, existing: Any) -> Any:
    """Recursive retype-preserving merge (R4).

    Returns `incoming`, except that wherever a piece of `incoming` equals the
    JSON-serialization of the corresponding piece of `existing`, the existing
    TYPED value is kept (bare dates, datetimes, non-string keys). The no-op
    guard must compare `existing` against this merge's result, not against the
    raw JSON payload — otherwise an untouched GET->POST silently retypes."""
    if isinstance(incoming, dict) and isinstance(existing, dict):
        existing_by_json_key: dict[str, Any] = {}
        for k in existing:
            existing_by_json_key.setdefault(_json_key(k), k)
        out: dict[Any, Any] = {}
        for in_key, in_value in incoming.items():
            if isinstance(in_key, str) and in_key in existing_by_json_key:
                typed_key = existing_by_json_key[in_key]
                out[typed_key] = merge_preserving_types(in_value, existing[typed_key])
            else:
                out[in_key] = in_value
        return out
    if isinstance(incoming, list) and isinstance(existing, list):
        return [
            merge_preserving_types(v, existing[i]) if i < len(existing) else v
            for i, v in enumerate(incoming)
        ]
    image = _json_image(existing)
    # Python's True == 1: require bool-ness to match so an intentional
    # 1 -> true (or true -> 1) edit is not swallowed as "unchanged".
    if isinstance(incoming, bool) == isinstance(image, bool) and incoming == image:
        return existing
    return incoming


def typed_equal(a: Any, b: Any) -> bool:
    """Deep equality that never conflates what Python `==` conflates.

    The no-op save guard must use this instead of `==`: `{"flag": True} ==
    {"flag": 1}` is True in Python, so a plain compare swallows an intentional
    bool<->int/float retype edit as "no changes". Rules: bool is distinguished
    from int/float BEFORE numeric equality (int vs float stays equal — JSON
    cannot distinguish them anyway); dicts/lists recurse (dict keys type-checked
    too, since True hash-collides with 1); typed scalars like date vs datetime
    compare by exact type + value."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) != len(b):
            return False
        if not all(k in b and typed_equal(v, b[k]) for k, v in a.items()):
            return False
        # `True in {1: ...}` is True (hash collision): key types must match too.
        return all(any(typed_equal(ka, kb) for kb in b) for ka in a)
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, list) or isinstance(b, list):
        return False
    if isinstance(a, (datetime.date, datetime.time)) or isinstance(b, (datetime.date, datetime.time)):
        return type(a) is type(b) and a == b
    return a == b
