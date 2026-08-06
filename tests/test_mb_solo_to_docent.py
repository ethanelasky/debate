"""Tests for scripts/mb_solo_to_docent.py (no network).

Every Docent interaction goes through a fake client injected into ``main`` (or
a monkeypatched ``make_client``); nothing here builds a real Docent client, and
one test asserts that a sentinel planted in the prompt/trajectory/response
never reaches stdout. All rows are SYNTHETIC — the real
``data/monitoringbench/*.jsonl`` files are never read.

``scripts.mb_solo_monitor`` imports ``infra.envs.tasks.monitoringbench``
(implementer B's module). These tests must not require that module: if it is
absent, a contract-faithful shim (MBRow / render_transcript / load_rows, per
MB_MIGRATION_SPEC.md "Module contract") is installed into ``sys.modules``
before the scripts are imported. When the real module exists, it is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_mb_env_module() -> None:
    """Use implementer B's ``infra.envs.tasks.monitoringbench`` if present; shim it
    from the spec's module contract otherwise (synthetic tests only)."""
    try:
        import infra.envs.tasks.monitoringbench  # noqa: F401

        return
    except ImportError:
        pass

    import types
    from dataclasses import dataclass

    mod = types.ModuleType("infra.envs.tasks.monitoringbench")

    @dataclass
    class MBRow:
        id: str
        label: str
        steps: list
        sample_uuid: Optional[str] = None
        orig_max_suspicion_score: Optional[float] = None
        red_team_objective_accomplished: Optional[bool] = None

    def render_transcript(steps):
        # Verbatim semantics from the old loader (prompt_sha256 continuity).
        blocks = []
        for i, step in enumerate(steps, start=1):
            action = step.get("action") or ""
            responses = step.get("responses") or ""
            blocks.append(f"## Step {i}\n### Action\n{action}\n### Output\n{responses}")
        return "\n\n".join(blocks)

    def load_rows(paths):
        rows = []
        n_dropped = 0
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    rid = record.get("id")
                    label = record.get("label")
                    steps = record.get("steps")
                    if not (rid and label in ("attack", "honest") and steps):
                        n_dropped += 1
                        print(
                            f"monitoringbench(shim): dropped row "
                            f"{Path(path).name}:{lineno} (id={rid!r})",
                            file=sys.stderr,
                        )
                        continue
                    rows.append(
                        MBRow(
                            id=str(rid),
                            label=label,
                            steps=steps,
                            sample_uuid=record.get("sample_uuid"),
                            orig_max_suspicion_score=record.get("orig_max_suspicion_score"),
                            red_team_objective_accomplished=record.get(
                                "red_team_objective_accomplished"
                            ),
                        )
                    )
        if n_dropped:
            print(f"monitoringbench(shim): dropped {n_dropped} row(s)", file=sys.stderr)
        return rows

    mod.MBRow = MBRow
    mod.render_transcript = render_transcript
    mod.load_rows = load_rows
    sys.modules["infra.envs.tasks.monitoringbench"] = mod


_ensure_mb_env_module()

from scripts import mb_solo_monitor as solo  # noqa: E402
from scripts import mb_solo_to_docent as up  # noqa: E402

SENTINEL = "ZZTOPSECRETTRAJECTORYZZ"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _traj_row(row_id: str, label: str, marker: str = "") -> dict[str, Any]:
    return {
        "id": row_id,
        "label": label,
        "steps": [{"action": f"act-{row_id}{marker}", "responses": f"out-{row_id}{marker}"}],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def _write_prompt_yaml(path: Path, *, marker: str = "") -> str:
    section = {
        "overall_system": {"role": "system", "content": [f"SYS{marker}"]},
        "answer_generation_user": {
            "role": "user",
            "content": [f"USER{marker} <BACKGROUND_TEXT> END"],
        },
    }
    # JSON is valid YAML, so this is a legitimate prompt yaml for the loader.
    path.write_text(json.dumps({"Sec": section}), encoding="utf-8")
    return str(path)


def _sha_by_id(data_path: str, prompt_path: str) -> dict[str, str]:
    """The prompt hashes mb_solo_monitor would record for this data+yaml."""
    rows = solo.load_rows([data_path])
    system_text, user_template = solo.load_prompt_blocks(prompt_path, "Sec")
    return {item.row_id: item.sha256 for item in solo.prepare_rows(rows, system_text, user_template)}


def _result_row(
    row_id: str,
    label: str,
    *,
    index: int = 0,
    sha: str = "sha-placeholder",
    verdict: Optional[str] = "attack",
    raw_response: Optional[str] = "reasoning <answer>attack</answer>",
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    has_score_tag: bool = False,
    score: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "id": row_id,
        "label": label,
        "verdict": verdict,
        "has_score_tag": has_score_tag,
        "score": score,
        "raw_response": raw_response,
        "served_provider": "Parasail",
        "generation_id": "gen-abc",
        "usage": {"total_tokens": 12},
        "error": error,
        "error_type": error_type,
        "attempts": 5 if error else 1,
        "prompt_sha256": sha,
        "model": "qwen/qwen3.5-9b",
        "provider": "parasail",
        "quantization": "bf16",
        "thinking": False,
        "provider_block": {"order": ["parasail"], "allow_fallbacks": False},
    }


class FakeDocent:
    """Records every call; never touches the network."""

    def __init__(self, collections: Optional[list[dict[str, str]]] = None) -> None:
        self.collections = list(collections or [])
        self.created: list[str] = []
        self.batches: list[tuple[str, list[Any]]] = []
        self.deleted: list[tuple[str, list[str]]] = []
        self.existing_run_ids: list[str] = []

    def list_collections(self) -> list[dict[str, str]]:
        return self.collections

    def create_collection(self, name: str, description: str = "") -> str:
        collection_id = f"col-{len(self.created)}"
        self.created.append(name)
        self.collections.append({"name": name, "id": collection_id})
        return collection_id

    def list_agent_run_ids(self, collection_id: str) -> list[str]:
        return list(self.existing_run_ids)

    def delete_agent_runs(self, collection_id: str, run_ids: list[str]) -> None:
        self.deleted.append((collection_id, list(run_ids)))

    def add_agent_runs(self, collection_id: str, runs: list[Any]) -> None:
        self.batches.append((collection_id, list(runs)))

    @property
    def uploaded(self) -> list[Any]:
        return [run for _, batch in self.batches for run in batch]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    """A self-contained data file + prompt yaml + matching results file."""
    data_path = _write_jsonl(
        tmp_path / "data.jsonl",
        [_traj_row("attack_1_0", "attack"), _traj_row("honest_2_0", "honest")],
    )
    prompt_path = _write_prompt_yaml(tmp_path / "prompts.yaml")
    shas = _sha_by_id(data_path, prompt_path)
    results_path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _result_row("attack_1_0", "attack", index=0, sha=shas["attack_1_0"]),
            _result_row(
                "honest_2_0",
                "honest",
                index=1,
                sha=shas["honest_2_0"],
                verdict=None,
                raw_response=None,
                error="RateLimitError: 429 too many requests",
                error_type="RateLimitError",
            ),
        ],
    )
    return {
        "data": data_path,
        "prompt": prompt_path,
        "results": results_path,
        "shas": shas,
        "tmp": tmp_path,
    }


def _argv(env: dict[str, Any], *extra: str) -> list[str]:
    return [
        "--results", env["results"],
        "--data", env["data"],
        "--prompt-file", env["prompt"],
        "--prompt-section", "Sec",
        *extra,
    ]


# --------------------------------------------------------------------------
# AgentRun construction
# --------------------------------------------------------------------------


def test_agent_run_has_system_user_assistant_in_order():
    record = _result_row("attack_1_0", "attack", sha="deadbeef")
    record["_results_file"] = "results.jsonl"
    run = up.result_to_agent_run(record, "SYSTEM TEXT", "USER TEXT", prompt_sha_verified=True)

    assert len(run.transcripts) == 1
    messages = run.transcripts[0].messages
    assert [m.role for m in messages] == ["system", "user", "assistant"]
    assert messages[0].content == "SYSTEM TEXT"
    assert messages[1].content == "USER TEXT"
    # raw_response goes in verbatim, untouched.
    assert messages[2].content == record["raw_response"]


def test_agent_run_metadata_carries_filterable_fields():
    record = _result_row(
        "attack_1_0", "attack", index=3, sha="deadbeef", has_score_tag=True, score=7
    )
    record["_results_file"] = "results.jsonl"
    run = up.result_to_agent_run(record, "S", "U", prompt_sha_verified=True)

    metadata = run.metadata
    assert metadata["id"] == "attack_1_0"
    assert metadata["label"] == "attack"
    assert metadata["verdict"] == "attack"
    assert metadata["correct"] is True
    assert metadata["has_score_tag"] is True
    assert metadata["score"] == 7
    assert metadata["model"] == "qwen/qwen3.5-9b"
    assert metadata["provider"] == "parasail"
    assert metadata["quantization"] == "bf16"
    assert metadata["thinking"] is False
    assert metadata["served_provider"] == "Parasail"
    assert metadata["error_type"] == "none"
    assert metadata["is_error_row"] is False
    assert metadata["prompt_sha256"] == "deadbeef"
    assert metadata["prompt_sha_verified"] is True
    assert metadata["results_file"] == "results.jsonl"
    assert metadata["source_index"] == 3


@pytest.mark.parametrize(
    "label,verdict,expected",
    [
        ("attack", "attack", True),
        ("attack", "benign", False),
        ("honest", "benign", True),
        ("honest", "attack", False),
        ("honest", None, None),
        ("attack", "nonsense", None),
    ],
)
def test_correct_maps_benign_to_honest(label, verdict, expected):
    assert up.verdict_is_correct(label, verdict) is expected


def test_error_row_still_becomes_a_run_naming_the_error_type():
    record = _result_row(
        "honest_2_0",
        "honest",
        sha="deadbeef",
        verdict=None,
        raw_response=None,
        error="RateLimitError: 429",
        error_type="RateLimitError",
    )
    record["_results_file"] = "results.jsonl"
    run = up.result_to_agent_run(record, "S", "U", prompt_sha_verified=True)

    messages = run.transcripts[0].messages
    assert [m.role for m in messages] == ["system", "user", "assistant"]
    assert "RateLimitError" in messages[2].content
    assert "NO RESPONSE" in messages[2].content
    assert run.metadata["is_error_row"] is True
    assert run.metadata["error_type"] == "RateLimitError"
    assert run.metadata["verdict"] == "none"
    assert run.metadata["correct"] is None


# --------------------------------------------------------------------------
# prompt rebuild + sha verification
# --------------------------------------------------------------------------


def test_rebuilt_prompt_matches_recorded_sha(env):
    prompts = up.build_prompt_index(
        ["attack_1_0", "honest_2_0"], [env["data"]], env["prompt"], "Sec"
    )
    for row_id, sha in env["shas"].items():
        assert prompts[row_id][2] == sha


def test_verify_prompt_sha_passes_when_matched():
    record = _result_row("attack_1_0", "attack", sha="abc123")
    assert up.verify_prompt_sha(record, "abc123", allow_drift=False) is True


def test_verify_prompt_sha_fails_loudly_with_the_offending_id():
    record = _result_row("attack_1_0", "attack", sha="abc123")
    with pytest.raises(ValueError) as excinfo:
        up.verify_prompt_sha(record, "different", allow_drift=False)
    message = str(excinfo.value)
    assert "attack_1_0" in message
    assert "abc123" in message and "different" in message


def test_missing_sha_is_treated_as_drift():
    record = _result_row("attack_1_0", "attack")
    record.pop("prompt_sha256")
    with pytest.raises(ValueError):
        up.verify_prompt_sha(record, "abc123", allow_drift=False)


def test_allow_prompt_drift_downgrades_to_warning(capsys):
    record = _result_row("attack_1_0", "attack", sha="abc123")
    assert up.verify_prompt_sha(record, "different", allow_drift=True) is False
    assert "attack_1_0" in capsys.readouterr().out


def test_main_aborts_on_prompt_drift(env, capsys):
    # Re-word the prompt yaml AFTER the run: every sha now mismatches.
    _write_prompt_yaml(Path(env["prompt"]), marker="-CHANGED")
    client = FakeDocent()
    with pytest.raises(SystemExit) as excinfo:
        up.main(_argv(env, "--name", "c"), client=client)
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "attack_1_0" in captured.err
    assert client.batches == []


def test_main_with_allow_prompt_drift_uploads_flagged_runs(env, capsys):
    _write_prompt_yaml(Path(env["prompt"]), marker="-CHANGED")
    client = FakeDocent()
    assert up.main(_argv(env, "--name", "c", "--allow-prompt-drift"), client=client) == 0
    assert [run.metadata["prompt_sha_verified"] for run in client.uploaded] == [False, False]
    assert "FLAGGED" in capsys.readouterr().out


def test_main_marks_all_rows_verified_when_nothing_changed(env):
    client = FakeDocent()
    assert up.main(_argv(env, "--name", "c"), client=client) == 0
    assert all(run.metadata["prompt_sha_verified"] for run in client.uploaded)


def test_results_id_absent_from_data_is_a_hard_error(env):
    data_path = _write_jsonl(env["tmp"] / "data2.jsonl", [_traj_row("attack_1_0", "attack")])
    client = FakeDocent()
    with pytest.raises(SystemExit) as excinfo:
        up.main(
            [
                "--results", env["results"],
                "--data", data_path,
                "--prompt-file", env["prompt"],
                "--prompt-section", "Sec",
                "--name", "c",
            ],
            client=client,
        )
    assert excinfo.value.code == 2
    assert client.batches == []


# --------------------------------------------------------------------------
# CLI behaviour
# --------------------------------------------------------------------------


def test_dry_run_touches_no_client(env, monkeypatch, capsys):
    def _boom() -> Any:  # pragma: no cover - must never run
        raise AssertionError("--dry-run must not construct a Docent client")

    monkeypatch.setattr(up, "make_client", _boom)
    assert up.main(_argv(env, "--name", "c", "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "prompt_sha_match=yes" in out


def test_dry_run_prints_one_summary_line_per_row(env, capsys):
    up.main(_argv(env, "--name", "c", "--dry-run"))
    out = capsys.readouterr().out
    assert "attack_1_0  label=attack  verdict=attack" in out
    assert "honest_2_0  label=honest  verdict=none  error=RateLimitError" in out


def test_name_creates_collection_when_absent(env, capsys):
    client = FakeDocent()
    assert up.main(_argv(env, "--name", "brand new"), client=client) == 0
    assert client.created == ["brand new"]
    assert [collection_id for collection_id, _ in client.batches] == ["col-0"]
    assert "col-0" in capsys.readouterr().out


def test_name_reuses_existing_collection(env):
    client = FakeDocent([{"name": "existing", "id": "col-existing"}])
    up.main(_argv(env, "--name", "existing"), client=client)
    assert client.created == []
    assert [collection_id for collection_id, _ in client.batches] == ["col-existing"]


def test_duplicate_collection_names_are_rejected():
    client = FakeDocent([{"name": "dup", "id": "a"}, {"name": "dup", "id": "b"}])
    with pytest.raises(ValueError):
        up.resolve_collection_id(client, "dup")


def test_collection_id_skips_name_lookup(env):
    client = FakeDocent()
    up.main(_argv(env, "--collection-id", "col-given"), client=client)
    assert client.created == []
    assert [collection_id for collection_id, _ in client.batches] == ["col-given"]


def test_name_and_collection_id_are_mutually_exclusive(env):
    with pytest.raises(SystemExit):
        up.main(_argv(env, "--name", "c", "--collection-id", "x"))


def test_one_destination_is_required(env):
    with pytest.raises(SystemExit):
        up.main(_argv(env))


def test_batch_size_splits_the_upload(env):
    client = FakeDocent()
    up.main(_argv(env, "--name", "c", "--batch-size", "1"), client=client)
    assert [len(batch) for _, batch in client.batches] == [1, 1]


def test_limit_truncates_in_file_order(env):
    client = FakeDocent()
    up.main(_argv(env, "--name", "c", "--limit", "1"), client=client)
    assert [run.metadata["id"] for run in client.uploaded] == ["attack_1_0"]


def test_replace_clears_existing_runs_first(env):
    client = FakeDocent()
    client.existing_run_ids = ["old-1", "old-2"]
    up.main(_argv(env, "--name", "c", "--replace"), client=client)
    assert client.deleted and client.deleted[0][1] == ["old-1", "old-2"]


def test_duplicate_result_rows_are_deduped_by_id_and_model(env, capsys):
    rows = [json.loads(line) for line in Path(env["results"]).read_text().splitlines()]
    _write_jsonl(Path(env["results"]), rows + [rows[0]])
    client = FakeDocent()
    up.main(_argv(env, "--name", "c"), client=client)
    assert [run.metadata["id"] for run in client.uploaded] == ["attack_1_0", "honest_2_0"]
    assert "duplicate" in capsys.readouterr().out


def test_duplicate_rows_across_results_files_are_deduped(env, capsys):
    """A partial re-run file passed alongside the original must not upload the
    same (id, model) twice; the first occurrence wins."""
    rows = [json.loads(line) for line in Path(env["results"]).read_text().splitlines()]
    rerun_path = _write_jsonl(env["tmp"] / "rerun.jsonl", [rows[1], rows[0]])
    client = FakeDocent()
    up.main(
        [
            "--results", env["results"],
            "--results", rerun_path,
            "--data", env["data"],
            "--prompt-file", env["prompt"],
            "--prompt-section", "Sec",
            "--name", "c",
        ],
        client=client,
    )
    assert [run.metadata["id"] for run in client.uploaded] == ["attack_1_0", "honest_2_0"]
    out = capsys.readouterr().out
    assert "skipped 2 (id, model) row(s) already seen" in out
    # The kept copies are the ones from the first file.
    assert all(run.metadata["results_file"] == "results.jsonl" for run in client.uploaded)


def test_replace_warns_that_a_failed_upload_leaves_a_partial_collection(env, capsys):
    client = FakeDocent()
    client.existing_run_ids = ["old-1"]
    up.main(_argv(env, "--name", "c", "--replace"), client=client)
    assert "left partial" in capsys.readouterr().out


def test_replace_deletes_nothing_when_a_run_cannot_be_built(env):
    """--replace must not gut the collection when the build step fails."""
    _write_prompt_yaml(Path(env["prompt"]), marker="-CHANGED")  # every sha mismatches
    client = FakeDocent()
    client.existing_run_ids = ["old-1"]
    with pytest.raises(SystemExit):
        up.main(_argv(env, "--name", "c", "--replace"), client=client)
    assert client.deleted == []


def test_missing_results_file_exits_2_without_a_traceback(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        up.main(
            [
                "--results", str(env["tmp"] / "nope.jsonl"),
                "--data", env["data"],
                "--prompt-file", env["prompt"],
                "--prompt-section", "Sec",
                "--name", "c",
            ]
        )
    assert excinfo.value.code == 2
    assert "Results file not found" in capsys.readouterr().err


def test_missing_prompt_file_exits_2_without_a_traceback(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        up.main(
            [
                "--results", env["results"],
                "--data", env["data"],
                "--prompt-file", str(env["tmp"] / "nope.yaml"),
                "--prompt-section", "Sec",
                "--name", "c",
            ]
        )
    assert excinfo.value.code == 2
    assert "Prompt file not found" in capsys.readouterr().err


def test_multiple_results_files_are_concatenated(env):
    rows = [json.loads(line) for line in Path(env["results"]).read_text().splitlines()]
    second = rows[0].copy()
    second["model"] = "other/model"
    second_path = _write_jsonl(env["tmp"] / "results2.jsonl", [second])
    client = FakeDocent()
    up.main(
        [
            "--results", env["results"],
            "--results", second_path,
            "--data", env["data"],
            "--prompt-file", env["prompt"],
            "--prompt-section", "Sec",
            "--name", "c",
        ],
        client=client,
    )
    assert [run.metadata["model"] for run in client.uploaded] == [
        "qwen/qwen3.5-9b", "qwen/qwen3.5-9b", "other/model"
    ]


# --------------------------------------------------------------------------
# the hard constraint: no content on stdout
# --------------------------------------------------------------------------


def test_never_prints_prompt_trajectory_or_response(tmp_path: Path, capsys):
    """A sentinel planted in the yaml, the trajectory and the response must not
    reach stdout/stderr on the full (upload) path."""
    data_path = _write_jsonl(
        tmp_path / "data.jsonl", [_traj_row("attack_9_0", "attack", marker=SENTINEL)]
    )
    prompt_path = _write_prompt_yaml(tmp_path / "prompts.yaml", marker=SENTINEL)
    shas = _sha_by_id(data_path, prompt_path)
    results_path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _result_row(
                "attack_9_0",
                "attack",
                sha=shas["attack_9_0"],
                raw_response=f"the monitor said {SENTINEL} <answer>attack</answer>",
            )
        ],
    )
    client = FakeDocent()
    argv = [
        "--results", results_path,
        "--data", data_path,
        "--prompt-file", prompt_path,
        "--prompt-section", "Sec",
        "--name", "c",
    ]
    assert up.main(argv, client=client) == 0
    # It reached Docent...
    run = client.uploaded[0]
    assert SENTINEL in run.transcripts[0].messages[1].content
    assert SENTINEL in run.transcripts[0].messages[2].content
    # ...but never the terminal.
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out
    assert SENTINEL not in captured.err


def test_dry_run_never_prints_content(tmp_path: Path, capsys):
    data_path = _write_jsonl(
        tmp_path / "data.jsonl", [_traj_row("attack_9_0", "attack", marker=SENTINEL)]
    )
    prompt_path = _write_prompt_yaml(tmp_path / "prompts.yaml", marker=SENTINEL)
    shas = _sha_by_id(data_path, prompt_path)
    results_path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _result_row(
                "attack_9_0",
                "attack",
                sha=shas["attack_9_0"],
                raw_response=f"{SENTINEL} <answer>attack</answer>",
            )
        ],
    )
    up.main(
        [
            "--results", results_path,
            "--data", data_path,
            "--prompt-file", prompt_path,
            "--prompt-section", "Sec",
            "--name", "c",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out
    assert SENTINEL not in captured.err


def test_cli_help_documents_prompt_verification(capsys):
    with pytest.raises(SystemExit):
        up.main(["--help"])
    out = capsys.readouterr().out
    assert "--allow-prompt-drift" in out
    assert "prompt_sha256" in out
