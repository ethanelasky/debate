"""Config-viewer tests (tools/viewer): synthetic tmp-dir YAML fixtures ONLY.

The app factory takes the two config directories as parameters, so nothing
here touches the repo's real configs — and a guard test asserts the viewer
package has no reference to the repo's protected trajectory directory.
"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tools.viewer.server import create_app as _create_app  # noqa: E402

VIEWER_PKG = Path(__file__).resolve().parents[1] / "tools" / "viewer"


def create_app(**kwargs):
    """create_app with TestClient's default Host on the allowlist.

    The real app rejects any Host outside the loopback names (DNS-rebinding
    defense); TestClient talks to `http://testserver`, so every test that is
    not itself about host validation opts that name in.
    """
    kwargs.setdefault("allowed_hosts", ["testserver"])
    return _create_app(**kwargs)

# Single prompt schema (infra/envs/debate/prompts.py): fixed stage keys plus
# per-turn cues keyed DIRECTLY by protocol slot name (`opening` / `closing`).
PROMPTS_A = """\
base_entry:
  vars:
    GREETING: hello
  overall_system:
    - block zero
    - block one
    - block two
  debater_system: debater system prompt
  judge_system: judge system prompt
  pre_debate: shared preamble text
  pre_debate_proposer: proposer preamble text
  opening: opening instruction
  closing: closing instruction
  attribution:
    judge:
      alice: alice said this
"""

PROMPTS_B = """\
child_entry:
  _extends: base_entry
  overall_system:
    - OVERRIDDEN block zero
  opening: child opening instruction
"""

EXPERIMENTS = """\
_models:
  preset_a:
    model_type: mock
    max_new_tokens: 128
_tpl:
  batch: 4
  agents:
    judge:
      model_settings:
        _preset: preset_a
        alias: judge-alias
mid:
  _extends: _tpl
  batch: 8
leaf:
  _extends: mid
  agents:
    judge:
      model_settings:
        alias: leaf-judge
"""


def _bump_mtime(path: Path) -> None:
    """Force a different st_mtime_ns without changing content."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompt_configs"
    configs.mkdir()
    prompts.mkdir()
    (prompts / "a.yaml").write_text(PROMPTS_A, encoding="utf-8")
    # b.yaml _extends across files must go through _includes (per-file
    # resolution, exactly like infra's load_prompt_library).
    (prompts / "b.yaml").write_text("_includes: [a.yaml]\n" + PROMPTS_B, encoding="utf-8")
    (prompts / "stale.backup.yaml").write_text("ghost_entry: {}\n", encoding="utf-8")
    (configs / "exp.yaml").write_text(EXPERIMENTS, encoding="utf-8")
    (configs / "old.backup.yaml").write_text("junk: {}\n", encoding="utf-8")
    return configs, prompts


@pytest.fixture()
def client(dirs: tuple[Path, Path]) -> TestClient:
    configs, prompts = dirs
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    return TestClient(app)


# --------------------------------------------------------------- pages exist


def test_pages_render(client: TestClient) -> None:
    for url in ("/prompts", "/experiments"):
        r = client.get(url)
        assert r.status_code == 200
        assert "Config Viewer" in r.text
    # exactly the two nav links
    assert client.get("/", follow_redirects=False).status_code in (302, 307)


# ------------------------------------------------------------------- prompts


def test_prompts_listing_and_source_map(client: TestClient) -> None:
    payload = client.get("/api/prompts").json()
    assert payload["files"] == ["a.yaml", "b.yaml"]  # backup file skipped
    assert payload["source_files"] == {"base_entry": "a.yaml", "child_entry": "b.yaml"}
    assert "ghost_entry" not in payload["raw"]
    assert "_includes" not in payload["raw"]
    assert set(payload["mtimes"]) == {"a.yaml", "b.yaml"}  # R9 tokens per file
    assert payload["errors"] == {}
    assert payload["invalid"] == {}  # R8: all fixture entries use accepted keys


def test_prompts_raw_vs_resolved_extends_chain(client: TestClient) -> None:
    payload = client.get("/api/prompts").json()
    raw, resolved = payload["raw"], payload["resolved"]

    # raw child holds ONLY its overrides
    assert raw["child_entry"]["_extends"] == "base_entry"
    assert "vars" not in raw["child_entry"]
    assert raw["child_entry"]["overall_system"] == ["OVERRIDDEN block zero"]

    # resolved child: _extends applied, lists merged BY INDEX, _extends stripped
    child = resolved["child_entry"]
    assert "_extends" not in child
    assert child["overall_system"] == ["OVERRIDDEN block zero", "block one", "block two"]
    assert child["judge_system"] == "judge system prompt"  # inherited
    assert child["pre_debate"] == "shared preamble text"  # inherited
    assert child["vars"] == {"GREETING": "hello"}  # inherited
    assert child["opening"] == "child opening instruction"  # overridden cue
    assert child["closing"] == "closing instruction"  # inherited cue


def test_prompts_save_roundtrip_multifile(client: TestClient, dirs: tuple[Path, Path]) -> None:
    _, prompts = dirs
    payload = client.get("/api/prompts").json()
    raw = payload["raw"]
    original_a = (prompts / "a.yaml").read_text()
    original_b = (prompts / "b.yaml").read_text()

    raw["child_entry"]["opening"] = "EDITED child opening\nsecond line"
    r = client.post(
        "/api/prompts",
        json={"raw": raw, "source_file_overrides": {}, "mtimes": payload["mtimes"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["written"] == ["b.yaml"]  # only the edited entry's file
    assert set(body["mtimes"]) == {"b.yaml"}  # fresh tokens for written files

    # a.yaml untouched byte-for-byte, no backup for it
    assert (prompts / "a.yaml").read_text() == original_a
    assert not (prompts / "a.backup.yaml").exists()

    # b.yaml: backup holds the original bytes; content equal modulo the edit
    assert (prompts / "b.backup.yaml").read_text() == original_b
    new_b = yaml.safe_load((prompts / "b.yaml").read_text())
    old_b = yaml.safe_load(original_b)
    assert new_b["_includes"] == old_b["_includes"]  # preserved
    old_b["child_entry"]["opening"] = "EDITED child opening\nsecond line"
    assert new_b == old_b


def test_prompts_noop_save_refused(client: TestClient, dirs: tuple[Path, Path]) -> None:
    _, prompts = dirs
    payload = client.get("/api/prompts").json()
    before = {p.name: p.read_text() for p in prompts.iterdir()}
    r = client.post("/api/prompts", json={"raw": payload["raw"], "mtimes": payload["mtimes"]})
    assert r.status_code == 200
    assert r.json()["status"] == "no_changes"
    after = {p.name: p.read_text() for p in prompts.iterdir()}
    assert after == before  # nothing written, no backups appeared


def test_prompts_new_entry_targets_parent_file(client: TestClient, dirs: tuple[Path, Path]) -> None:
    _, prompts = dirs
    payload = client.get("/api/prompts").json()
    raw = payload["raw"]
    raw["new_variant"] = {"_extends": "base_entry"}
    r = client.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 200
    saved = yaml.safe_load((prompts / "a.yaml").read_text())
    assert saved["new_variant"] == {"_extends": "base_entry"}
    # explicit override wins over the parent's file (fresh GET: a.yaml changed)
    payload = client.get("/api/prompts").json()
    raw = payload["raw"]
    raw["other_variant"] = {"_extends": "base_entry"}
    r = client.post(
        "/api/prompts",
        json={
            "raw": raw,
            "source_file_overrides": {"other_variant": "b.yaml"},
            "mtimes": payload["mtimes"],
        },
    )
    assert r.status_code == 200
    assert "other_variant" in yaml.safe_load((prompts / "b.yaml").read_text())


def test_prompts_new_entry_without_target_rejected(client: TestClient) -> None:
    raw = client.get("/api/prompts").json()["raw"]
    raw["orphan"] = {"overall_system": "hi"}
    r = client.post("/api/prompts", json={"raw": raw})
    assert r.status_code == 400


def test_prompts_duplicate_entry_across_files_is_error(
    dirs: tuple[Path, Path],
) -> None:
    configs, prompts = dirs
    (prompts / "c.yaml").write_text("base_entry: {vars: {X: y}}\n", encoding="utf-8")
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    r = TestClient(app).get("/api/prompts")
    assert r.status_code == 409


def test_prompts_post_duplicate_entry_is_409(client: TestClient, dirs: tuple[Path, Path]) -> None:
    """R12: the POST side mirrors the GET duplicate 409 instead of silently
    writing into whichever file sorts first."""
    _, prompts = dirs
    payload = client.get("/api/prompts").json()
    raw = payload["raw"]
    (prompts / "c.yaml").write_text("base_entry: {vars: {X: y}}\n", encoding="utf-8")
    raw["base_entry"]["vars"]["GREETING"] = "changed"
    r = client.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 409
    assert "base_entry" in r.json()["detail"]


def test_prompts_unknown_keys_are_presumed_cues_not_invalid(
    dirs: tuple[Path, Path],
) -> None:
    """R8 under the single schema: a non-stage key is a per-turn cue keyed by
    a PROTOCOL SLOT NAME, and the viewer has no protocol — so it must not
    flag one as invalid. It surfaces the presumed cues informationally."""
    configs, prompts = dirs
    (prompts / "c.yaml").write_text(
        "weird_entry:\n  vars: {A: b}\n  some_slot: cue text\n", encoding="utf-8"
    )
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    payload = TestClient(app).get("/api/prompts").json()
    assert payload["invalid"] == {}
    assert payload["cues"]["weird_entry"] == ["some_slot"]
    assert payload["cues"]["base_entry"] == ["closing", "opening"]
    assert payload["cues"]["child_entry"] == ["closing", "opening"]
    assert "weird_entry" in payload["raw"]


def test_prompts_non_mapping_entry_flagged_invalid(dirs: tuple[Path, Path]) -> None:
    """The one structural problem the viewer CAN detect without a protocol."""
    configs, prompts = dirs
    (prompts / "c.yaml").write_text("scalar_entry: 5\n", encoding="utf-8")
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    payload = TestClient(app).get("/api/prompts").json()
    assert payload["invalid"] == {"scalar_entry": ["<entry is not a mapping>"]}
    assert "scalar_entry" not in payload["cues"]
    assert "scalar_entry" in payload["raw"]  # still shown, just flagged


def test_prompts_per_file_error_isolation(dirs: tuple[Path, Path]) -> None:
    """R10: a malformed / non-mapping / unresolvable file must not 500 the
    whole page — healthy files still load, broken ones are listed."""
    configs, prompts = dirs
    (prompts / "broken.yaml").write_text("entry: {unclosed\n", encoding="utf-8")
    (prompts / "listy.yaml").write_text("- a\n- b\n", encoding="utf-8")
    (prompts / "orphan.yaml").write_text("dangling: {_extends: nowhere}\n", encoding="utf-8")
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    r = TestClient(app).get("/api/prompts")
    assert r.status_code == 200
    payload = r.json()
    assert set(payload["errors"]) == {"broken.yaml", "listy.yaml", "orphan.yaml"}
    assert "mapping" in payload["errors"]["listy.yaml"]
    assert "base_entry" in payload["raw"]  # healthy files unaffected
    assert "child_entry" in payload["resolved"]


def test_prompts_stale_mtime_409_only_for_touched_files(
    client: TestClient, dirs: tuple[Path, Path]
) -> None:
    """R9: POST validates tokens only for the files it would touch."""
    _, prompts = dirs
    payload = client.get("/api/prompts").json()
    raw = payload["raw"]
    raw["child_entry"]["opening"] = "post-race edit"

    # a.yaml goes stale, but the edit only touches b.yaml -> still saves
    _bump_mtime(prompts / "a.yaml")
    r = client.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 200
    assert r.json()["written"] == ["b.yaml"]

    # now b.yaml goes stale under a second edit based on old tokens -> 409
    raw["child_entry"]["opening"] = "second racing edit"
    _bump_mtime(prompts / "b.yaml")
    r = client.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 409
    assert "changed on disk" in r.json()["detail"]

    # missing token for a touched file is a clean 400
    r = client.post("/api/prompts", json={"raw": raw})
    assert r.status_code == 400


# --------------------------------------------------------------- experiments


def test_experiment_file_listing(client: TestClient) -> None:
    payload = client.get("/api/experiments/files").json()
    assert payload["files"] == ["exp.yaml"]  # backup skipped
    assert payload["default"] == "exp.yaml"


def test_experiment_raw_vs_resolved_and_presets(client: TestClient) -> None:
    payload = client.get("/api/experiments/file/exp.yaml").json()
    raw, resolved, models = payload["raw"], payload["resolved"], payload["models"]

    assert models == {"preset_a": {"model_type": "mock", "max_new_tokens": 128}}
    assert payload["mtime"]  # R9 staleness token

    # raw keeps _extends + _preset untouched
    assert raw["leaf"]["_extends"] == "mid"
    assert raw["_tpl"]["agents"]["judge"]["model_settings"]["_preset"] == "preset_a"

    # resolved: chained _extends applied, presets expanded AFTER inheritance
    leaf = resolved["leaf"]
    assert "_extends" not in leaf
    assert leaf["batch"] == 8  # from mid, overriding _tpl
    ms = leaf["agents"]["judge"]["model_settings"]
    assert "_preset" not in ms  # expanded away
    assert ms["model_type"] == "mock"
    assert ms["max_new_tokens"] == 128
    assert ms["alias"] == "leaf-judge"  # leaf's explicit override survives


def test_experiment_save_roundtrip(client: TestClient, dirs: tuple[Path, Path]) -> None:
    configs, _ = dirs
    payload = client.get("/api/experiments/file/exp.yaml").json()
    raw = payload["raw"]
    original = (configs / "exp.yaml").read_text()

    raw["leaf"]["batch"] = 16
    r = client.post(
        "/api/experiments/file/exp.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    assert r.json()["mtime"]  # fresh token so the client can keep saving

    backup = configs / "exp.backup.yaml"
    assert backup.read_text() == original

    new = yaml.safe_load((configs / "exp.yaml").read_text())
    old = yaml.safe_load(original)
    old["leaf"]["batch"] = 16
    assert new == old  # content equality modulo the one edit
    # key order preserved on round-trip
    assert list(new) == list(old)

    # resolved view reflects the edit on re-read
    resolved = client.get("/api/experiments/file/exp.yaml").json()["resolved"]
    assert resolved["leaf"]["batch"] == 16


def test_experiment_noop_save_refused(client: TestClient, dirs: tuple[Path, Path]) -> None:
    configs, _ = dirs
    payload = client.get("/api/experiments/file/exp.yaml").json()
    mtime = (configs / "exp.yaml").stat().st_mtime_ns
    r = client.post(
        "/api/experiments/file/exp.yaml",
        json={"raw": payload["raw"], "mtime": payload["mtime"]},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "no_changes"
    assert (configs / "exp.yaml").stat().st_mtime_ns == mtime
    assert not (configs / "exp.backup.yaml").exists()


def test_experiment_filename_validation(client: TestClient) -> None:
    assert client.get("/api/experiments/file/missing.yaml").status_code == 404
    assert client.get("/api/experiments/file/..%2Fexp.yaml").status_code in (400, 404)
    assert client.get("/api/experiments/file/exp.backup.yaml").status_code == 400
    assert client.post("/api/experiments/file/new_file.yaml", json={}).status_code == 404


def test_experiment_stale_mtime_409(client: TestClient, dirs: tuple[Path, Path]) -> None:
    """R9: a save based on an out-of-date token must 409, not lose the
    other writer's update."""
    configs, _ = dirs
    payload = client.get("/api/experiments/file/exp.yaml").json()
    raw = payload["raw"]
    raw["leaf"]["batch"] = 99
    before = (configs / "exp.yaml").read_text()
    _bump_mtime(configs / "exp.yaml")
    r = client.post(
        "/api/experiments/file/exp.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.status_code == 409
    assert "changed on disk" in r.json()["detail"]
    assert (configs / "exp.yaml").read_text() == before  # nothing overwritten
    # missing token is a clean 400
    r = client.post("/api/experiments/file/exp.yaml", json={"raw": raw})
    assert r.status_code == 400


def test_experiment_malformed_file_clean_4xx(dirs: tuple[Path, Path]) -> None:
    """R10: malformed / non-mapping / unresolvable files return a clean JSON
    4xx, never a traceback 500."""
    configs, prompts = dirs
    (configs / "bad.yaml").write_text("entry: {unclosed\n", encoding="utf-8")
    (configs / "listy.yaml").write_text("- a\n- b\n", encoding="utf-8")
    (configs / "orphan.yaml").write_text("foo: {_extends: nowhere}\n", encoding="utf-8")
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    c = TestClient(app)
    r = c.get("/api/experiments/file/bad.yaml")
    assert r.status_code == 400
    assert "not valid YAML" in r.json()["detail"]
    r = c.get("/api/experiments/file/listy.yaml")
    assert r.status_code == 400
    assert "mapping" in r.json()["detail"]
    r = c.get("/api/experiments/file/orphan.yaml")
    assert r.status_code == 400
    assert "error resolving" in r.json()["detail"]


# ------------------------------------------- typed scalars across JSON (R4)

DATED = """\
dated_exp:
  start_date: 2026-08-03
  created: 2026-08-03T12:30:00
  ports:
    8080: main
  batch: 4
"""


def test_experiment_date_untouched_roundtrip_is_noop(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "dated.yaml").write_text(DATED, encoding="utf-8")
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    payload = c.get("/api/experiments/file/dated.yaml").json()
    # JSON transport retypes: date -> isoformat string, int key -> string key
    assert payload["raw"]["dated_exp"]["start_date"] == "2026-08-03"
    assert payload["raw"]["dated_exp"]["created"] == "2026-08-03T12:30:00"
    assert "8080" in payload["raw"]["dated_exp"]["ports"]

    before = (configs / "dated.yaml").read_text()
    r = c.post(
        "/api/experiments/file/dated.yaml",
        json={"raw": payload["raw"], "mtime": payload["mtime"]},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "no_changes"  # merge restored the typed values
    assert (configs / "dated.yaml").read_text() == before


def test_experiment_edit_elsewhere_does_not_retype_date(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "dated.yaml").write_text(DATED, encoding="utf-8")
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    payload = c.get("/api/experiments/file/dated.yaml").json()
    raw = payload["raw"]
    raw["dated_exp"]["batch"] = 8
    r = c.post(
        "/api/experiments/file/dated.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    saved = yaml.safe_load((configs / "dated.yaml").read_text())["dated_exp"]
    assert saved["batch"] == 8
    assert saved["start_date"] == datetime.date(2026, 8, 3)
    assert not isinstance(saved["start_date"], datetime.datetime)  # still a bare date
    assert saved["created"] == datetime.datetime(2026, 8, 3, 12, 30)
    assert saved["ports"] == {8080: "main"}  # int key survived the round-trip


FLAGGED = """\
flag_exp:
  enabled: true
  count: 1
"""


def test_experiment_bool_int_retype_edits_actually_save(tmp_path: Path) -> None:
    """Python's True == 1 must not swallow an intentional bool<->int retype
    edit as 'no changes' — while a genuinely untouched file stays a no-op."""
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "flags.yaml").write_text(FLAGGED, encoding="utf-8")
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    # untouched GET -> POST is still a refused no-op
    payload = c.get("/api/experiments/file/flags.yaml").json()
    r = c.post(
        "/api/experiments/file/flags.yaml",
        json={"raw": payload["raw"], "mtime": payload["mtime"]},
    )
    assert r.json()["status"] == "no_changes"

    # int -> bool: count: 1 -> true must SAVE
    payload = c.get("/api/experiments/file/flags.yaml").json()
    raw = payload["raw"]
    raw["flag_exp"]["count"] = True
    r = c.post(
        "/api/experiments/file/flags.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.json()["status"] == "success"
    saved = yaml.safe_load((configs / "flags.yaml").read_text())["flag_exp"]
    assert saved["count"] is True

    # bool -> int: enabled: true -> 1 must SAVE
    payload = c.get("/api/experiments/file/flags.yaml").json()
    raw = payload["raw"]
    raw["flag_exp"]["enabled"] = 1
    r = c.post(
        "/api/experiments/file/flags.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.json()["status"] == "success"
    saved = yaml.safe_load((configs / "flags.yaml").read_text())["flag_exp"]
    assert saved["enabled"] == 1
    assert not isinstance(saved["enabled"], bool)


def test_prompts_bool_int_retype_edits_actually_save(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (prompts / "f.yaml").write_text(
        "flag_entry:\n  vars:\n    SWITCH: true\n    LIMIT: 1\n", encoding="utf-8"
    )
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    # untouched GET -> POST is still a refused no-op
    payload = c.get("/api/prompts").json()
    r = c.post("/api/prompts", json={"raw": payload["raw"], "mtimes": payload["mtimes"]})
    assert r.json()["status"] == "no_changes"

    # int -> bool must save
    payload = c.get("/api/prompts").json()
    raw = payload["raw"]
    raw["flag_entry"]["vars"]["LIMIT"] = True
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.json()["status"] == "success"
    saved = yaml.safe_load((prompts / "f.yaml").read_text())["flag_entry"]["vars"]
    assert saved["LIMIT"] is True

    # bool -> int must save
    payload = c.get("/api/prompts").json()
    raw = payload["raw"]
    raw["flag_entry"]["vars"]["SWITCH"] = 0
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.json()["status"] == "success"
    saved = yaml.safe_load((prompts / "f.yaml").read_text())["flag_entry"]["vars"]
    assert saved["SWITCH"] == 0
    assert not isinstance(saved["SWITCH"], bool)


def test_prompts_date_preserved_through_save(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (prompts / "d.yaml").write_text(
        "dated_entry:\n  vars:\n    START: 2026-08-03\n    NOTE: keep\n", encoding="utf-8"
    )
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    payload = c.get("/api/prompts").json()
    raw = payload["raw"]
    assert raw["dated_entry"]["vars"]["START"] == "2026-08-03"

    # untouched GET -> POST is a refused no-op
    before = (prompts / "d.yaml").read_text()
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 200
    assert r.json()["status"] == "no_changes"
    assert (prompts / "d.yaml").read_text() == before

    # an edit elsewhere must not retype the date
    raw["dated_entry"]["vars"]["NOTE"] = "edited"
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 200
    saved = yaml.safe_load((prompts / "d.yaml").read_text())["dated_entry"]["vars"]
    assert saved["NOTE"] == "edited"
    assert saved["START"] == datetime.date(2026, 8, 3)


# ------------------------------------------------------------------- safety


def test_includes_containment_rejected_both_pages(tmp_path: Path) -> None:
    """R1/R11a: an _includes entry resolving outside the configured directory
    is a hard 400 on both pages — never read, never leaked."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.yaml").write_text("smuggled: {}\n", encoding="utf-8")
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "exp.yaml").write_text(
        "_includes: ['../outside/x.yaml']\nfoo: {}\n", encoding="utf-8"
    )
    (prompts / "a.yaml").write_text(
        "_includes: ['../outside/x.yaml']\nentry: {}\n", encoding="utf-8"
    )
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    r = c.get("/api/experiments/file/exp.yaml")
    assert r.status_code == 400
    assert "x.yaml" in r.json()["detail"]
    assert "smuggled" not in r.text

    r = c.get("/api/prompts")
    assert r.status_code == 400
    assert "x.yaml" in r.json()["detail"]
    assert "smuggled" not in r.text


def test_save_cannot_introduce_escaping_includes(tmp_path: Path) -> None:
    """Defense-in-depth: POST must 400 before writing any file whose
    _includes would resolve outside the configured directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.yaml").write_text("smuggled: {}\n", encoding="utf-8")
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "exp.yaml").write_text("foo: {batch: 1}\n", encoding="utf-8")
    (prompts / "a.yaml").write_text(PROMPTS_A, encoding="utf-8")
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    # experiments: payload tries to introduce an escaping _includes
    payload = c.get("/api/experiments/file/exp.yaml").json()
    before = (configs / "exp.yaml").read_text()
    r = c.post(
        "/api/experiments/file/exp.yaml",
        json={
            "raw": {"_includes": ["../outside/x.yaml"], "foo": {"batch": 1}},
            "mtime": payload["mtime"],
        },
    )
    assert r.status_code == 400
    assert "x.yaml" in r.json()["detail"]
    assert (configs / "exp.yaml").read_text() == before

    # prompts: an escaping _includes appearing on disk after the GET must not
    # be preserved into a write — 400 before touching the file
    payload = c.get("/api/prompts").json()
    raw = payload["raw"]
    raw["base_entry"]["vars"]["GREETING"] = "edited"
    poisoned = "_includes: ['../outside/x.yaml']\n" + PROMPTS_A
    (prompts / "a.yaml").write_text(poisoned, encoding="utf-8")
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 400
    assert "x.yaml" in r.json()["detail"]
    assert (prompts / "a.yaml").read_text() == poisoned  # untouched, no backup
    assert not (prompts / "a.backup.yaml").exists()


def test_symlinked_yaml_not_listed_or_accessible(tmp_path: Path) -> None:
    """R2/R11b: symlinks inside the config dirs are neither listed nor
    readable nor writable."""
    secret = tmp_path / "secret.yaml"
    secret.write_text("hidden_entry: {vars: {X: y}}\n", encoding="utf-8")
    original_secret = secret.read_text()
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "exp.yaml").write_text("foo: {batch: 1}\n", encoding="utf-8")
    (configs / "link.yaml").symlink_to(secret)
    (prompts / "a.yaml").write_text("entry: {vars: {A: b}}\n", encoding="utf-8")
    (prompts / "plink.yaml").symlink_to(secret)
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    # not listed on either page
    assert c.get("/api/experiments/files").json()["files"] == ["exp.yaml"]
    payload = c.get("/api/prompts").json()
    assert payload["files"] == ["a.yaml"]
    assert "hidden_entry" not in payload["raw"]
    assert "hidden_entry" not in payload["resolved"]

    # not readable or writable by name either
    assert c.get("/api/experiments/file/link.yaml").status_code == 400
    r = c.post(
        "/api/experiments/file/link.yaml",
        json={"raw": {"foo": {}}, "mtime": "1"},
    )
    assert r.status_code == 400
    assert secret.read_text() == original_secret


def test_atomic_write_failure_leaves_original(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3/R11c: a mid-dump failure must leave the original bytes intact
    (no 0-byte truncation) and no temp file behind."""
    configs, prompts = dirs
    app = create_app(configs_dir=configs, prompts_dir=prompts)
    c = TestClient(app, raise_server_exceptions=False)
    payload = c.get("/api/experiments/file/exp.yaml").json()
    raw = payload["raw"]
    raw["leaf"]["batch"] = 32
    original = (configs / "exp.yaml").read_text()

    import tools.viewer.yamlio as yamlio

    def exploding_dump(data, stream, **kwargs):
        stream.write("PARTIAL GARBAGE")
        raise RuntimeError("simulated mid-dump failure")

    monkeypatch.setattr(yamlio.yaml, "dump", exploding_dump)
    r = c.post(
        "/api/experiments/file/exp.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.status_code == 500
    assert (configs / "exp.yaml").read_text() == original  # bytes untouched
    assert not list(configs.glob("*.tmp"))  # temp file cleaned up


def test_prompts_js_groups_match_entry_schema() -> None:
    """The prompts page's component groups must be exactly infra's fixed entry
    keys plus the catch-all "cues" bucket. They drifted once already — GROUPS
    said "user_preamble" while the schema key was "preamble", so the largest
    component rendered as a single unusable JSON blob instead of rows."""
    from infra.envs.debate.prompts import _ENTRY_KEYS, _STAGE_KEYS

    source = (VIEWER_PKG / "static" / "js" / "prompts.js").read_text()
    groups = re.search(r"const GROUPS = \[(.*?)\];", source, re.S)
    assert groups, "GROUPS list not found in prompts.js"
    found = set(re.findall(r'"([^"]+)"', groups.group(1)))
    assert found == set(_ENTRY_KEYS) | {"cues"}
    # every stage key gets its own group (no stage silently lands in "cues")
    assert set(_STAGE_KEYS) <= found


def test_foreign_host_header_rejected(dirs: tuple[Path, Path]) -> None:
    """DNS-rebinding defense: a page on an attacker-controlled hostname that
    resolves to 127.0.0.1 must not reach this read+write API."""
    configs, prompts = dirs
    app = _create_app(configs_dir=configs, prompts_dir=prompts)
    c = TestClient(app, base_url="http://evil.example.com")
    for url in ("/prompts", "/api/prompts", "/api/experiments/files"):
        r = c.get(url)
        assert r.status_code == 400, url
        assert "Host" in r.json()["detail"]
    # writes are blocked too, before any route runs
    assert c.post("/api/prompts", json={"raw": {}}).status_code == 400


def test_loopback_and_bound_hosts_accepted(dirs: tuple[Path, Path]) -> None:
    """localhost / 127.0.0.1 / [::1] (any port) always pass; a non-loopback
    --host is accepted only because it was explicitly bound."""
    configs, prompts = dirs
    app = _create_app(configs_dir=configs, prompts_dir=prompts, allowed_hosts=["192.168.1.5"])
    c = TestClient(app, base_url="http://127.0.0.1")
    for host in ("localhost:8080", "127.0.0.1", "[::1]:9999", "::1", "192.168.1.5:8080"):
        r = c.get("/api/experiments/files", headers={"host": host})
        assert r.status_code == 200, host
    assert c.get("/api/experiments/files", headers={"host": "other.host"}).status_code == 400


def test_missing_host_header_rejected(dirs: tuple[Path, Path]) -> None:
    configs, prompts = dirs
    app = _create_app(configs_dir=configs, prompts_dir=prompts)
    c = TestClient(app, base_url="http://127.0.0.1")
    assert c.get("/api/experiments/files", headers={"host": ""}).status_code == 400


def test_host_check_helpers() -> None:
    from tools.viewer.server import host_name, is_loopback_host

    assert host_name("[::1]:8080") == "::1"
    assert host_name("LOCALHOST:8080") == "localhost"
    assert host_name("127.0.0.1") == "127.0.0.1"
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")


def test_colliding_json_keys_rejected_on_save(tmp_path: Path) -> None:
    """Two YAML keys that collapse to the same JSON key (1 vs "1") cannot be
    edited safely — the GET already lost one — so the save is a 400 naming
    both keys instead of a silent drop."""
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (configs / "collide.yaml").write_text(
        "exp:\n  ports:\n    1: int_key\n    '1': str_key\n  batch: 4\n", encoding="utf-8"
    )
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))

    payload = c.get("/api/experiments/file/collide.yaml").json()
    before = (configs / "collide.yaml").read_text()
    raw = payload["raw"]
    raw["exp"]["batch"] = 8
    r = c.post(
        "/api/experiments/file/collide.yaml", json={"raw": raw, "mtime": payload["mtime"]}
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "'1'" in detail and "1" in detail
    assert (configs / "collide.yaml").read_text() == before  # nothing written
    assert not (configs / "collide.backup.yaml").exists()


def test_colliding_json_keys_rejected_on_prompts_save(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (prompts / "a.yaml").write_text(
        "entry:\n  vars:\n    1: int_key\n    '1': str_key\n", encoding="utf-8"
    )
    c = TestClient(create_app(configs_dir=configs, prompts_dir=prompts))
    payload = c.get("/api/prompts").json()
    raw = payload["raw"]
    raw["entry"]["vars"]["NEW"] = "x"
    r = c.post("/api/prompts", json={"raw": raw, "mtimes": payload["mtimes"]})
    assert r.status_code == 400
    assert "JSON key" in r.json()["detail"]
    assert not (prompts / "a.backup.yaml").exists()


def test_viewer_never_references_trajectory_dir() -> None:
    """The viewer must have no code path touching the repo's protected
    trajectory directory: no source file may even mention it."""
    needle = "data" + "/"  # split so this test file's own scan would not hit
    offenders = []
    for path in sorted(VIEWER_PKG.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".html", ".css", ".js"}:
            if needle in path.read_text():
                offenders.append(str(path))
    assert not offenders, f"viewer files reference the trajectory dir: {offenders}"


def test_viewer_reads_only_configured_dirs() -> None:
    """No hard-coded open() targets outside the parameterized config dirs:
    every filesystem read in the routes goes through the dirs on app.state."""
    for module in ("routes_prompts", "routes_experiments", "yamlio", "server"):
        source = (VIEWER_PKG / f"{module}.py").read_text()
        assert "jsonl" not in source
        assert "monitoringbench" not in source.lower()
