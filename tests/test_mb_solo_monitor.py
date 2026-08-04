"""Tests for scripts/mb_solo_monitor.py (no network).

Every model call goes through a fake client injected into ``call_one`` /
``run_rows``; nothing here constructs a real OpenAI/OpenRouter client, and all
rows are SYNTHETIC — the real ``data/monitoringbench/*.jsonl`` files are never
read.

``scripts.mb_solo_monitor`` imports ``infra.envs.monitoringbench``
(implementer B's module). These tests must not require that module: if it is
absent, a contract-faithful shim (MBRow / render_transcript / load_rows, per
MB_MIGRATION_SPEC.md "Module contract") is installed into ``sys.modules``
before the script is imported. When the real module exists, it is used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_mb_env_module() -> None:
    """Use implementer B's ``infra.envs.monitoringbench`` if present; shim it
    from the spec's module contract otherwise (synthetic tests only)."""
    try:
        import infra.envs.monitoringbench  # noqa: F401

        return
    except ImportError:
        pass

    import types
    from dataclasses import dataclass

    mod = types.ModuleType("infra.envs.monitoringbench")

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
    sys.modules["infra.envs.monitoringbench"] = mod


_ensure_mb_env_module()

from scripts import mb_solo_monitor as solo  # noqa: E402

#: The byte-copied legacy prompt config lives under data/, which is
#: intentionally gitignored — a fresh clone lacks it. Tests that need the
#: real file on disk skip instead of failing.
LEGACY_PROMPT_FILE = REPO_ROOT / "data" / "monitoringbench" / "prompts" / "monitoringbench_legacy.yaml"
requires_legacy_prompt_file = pytest.mark.skipif(
    not LEGACY_PROMPT_FILE.exists(),
    reason="data/monitoringbench/prompts/monitoringbench_legacy.yaml not on disk "
    "(data/ is intentionally gitignored)",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _row(row_id: str, label: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "label": label,
        "steps": [{"action": f"act-{row_id}", "responses": f"out-{row_id}"}],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return str(path)


def _prompt_yaml(
    path: Path,
    *,
    system_content: Any = ["SYS"],
    user_content: Any = ["USER <BACKGROUND_TEXT> END"],
    drop_user: bool = False,
) -> str:
    section: dict[str, Any] = {"overall_system": {"role": "system", "content": system_content}}
    if not drop_user:
        section["answer_generation_user"] = {"role": "user", "content": user_content}
    path.write_text(json.dumps({"Sec": section}), encoding="utf-8")  # JSON is valid YAML
    return str(path)


class FakeCompletion(SimpleNamespace):
    pass


def _completion(
    text: Optional[str],
    *,
    provider: Optional[str] = "parasail",
    gen_id: Optional[str] = "gen-123",
) -> FakeCompletion:
    message = SimpleNamespace(content=text, reasoning=None)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    return FakeCompletion(
        choices=[SimpleNamespace(message=message)],
        provider=provider,
        id=gen_id,
        usage=usage,
    )


class FakeClient:
    """Minimal stand-in for ``openai.OpenAI``; records every request."""

    def __init__(self, responses: dict[str, Any] | list[Any]):
        self._responses = responses
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self._responses, list):
            item = self._responses.pop(0)
        else:
            user = kwargs["messages"][1]["content"]
            key = next(k for k in self._responses if k in user)
            item = self._responses[key]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _prepared(row_id: str = "attack_1", label: str = "attack") -> solo.PreparedRow:
    return solo.PreparedRow(
        index=0, row_id=row_id, label=label, system_text="SYS", user_text=f"USER {row_id}"
    )


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------


def test_build_prompt_substitutes_background_text():
    system, user = solo.build_prompt("SYS", "before <BACKGROUND_TEXT> after", "TRANSCRIPT")
    assert system == "SYS"
    assert user == "before TRANSCRIPT after"
    assert solo.BACKGROUND_TOKEN not in user


def test_load_prompt_blocks_joins_content_with_newline(tmp_path: Path):
    path = _prompt_yaml(
        tmp_path / "p.yaml",
        system_content=["a", "b"],
        user_content=["x <BACKGROUND_TEXT>", "y"],
    )
    system, user = solo.load_prompt_blocks(path, "Sec")
    assert system == "a\nb"
    assert user == "x <BACKGROUND_TEXT>\ny"


@requires_legacy_prompt_file
def test_load_prompt_blocks_real_repo_config():
    """The default (legacy) stanza must exist and carry the substitution token."""
    system, user = solo.load_prompt_blocks()
    assert system.strip()
    assert solo.BACKGROUND_TOKEN in user
    assert "<answer>" in user


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"drop_user": True}, "answer_generation_user"),
        ({"user_content": []}, "is empty"),
        ({"user_content": ["   "]}, "is empty"),
        ({"system_content": []}, "overall_system"),
        ({"user_content": ["no token here"]}, "<BACKGROUND_TEXT>"),
    ],
)
def test_load_prompt_blocks_fails_loudly(tmp_path: Path, kwargs: dict, needle: str):
    path = _prompt_yaml(tmp_path / "p.yaml", **kwargs)
    with pytest.raises(ValueError) as excinfo:
        solo.load_prompt_blocks(path, "Sec")
    assert needle in str(excinfo.value)


def test_load_prompt_blocks_missing_section(tmp_path: Path):
    path = _prompt_yaml(tmp_path / "p.yaml")
    with pytest.raises(ValueError) as excinfo:
        solo.load_prompt_blocks(path, "Nope")
    assert "'Nope'" in str(excinfo.value)


def test_prompt_sha256_tracks_both_messages():
    a = solo.prompt_sha256("sys", "user")
    assert a == solo.prompt_sha256("sys", "user")
    assert a != solo.prompt_sha256("sys2", "user")
    assert a != solo.prompt_sha256("sys", "user2")
    # No boundary collision between the two fields.
    assert solo.prompt_sha256("ab", "c") != solo.prompt_sha256("a", "bc")


def test_prepare_rows_uses_pipeline_render_transcript(tmp_path: Path):
    rows = [_row("attack_1", "attack")]
    prepared = solo.prepare_rows(rows, "SYS", "T:<BACKGROUND_TEXT>")
    assert prepared[0].user_text == (
        "T:## Step 1\n### Action\nact-attack_1\n### Output\nout-attack_1"
    )
    assert prepared[0].sha256 == solo.prompt_sha256("SYS", prepared[0].user_text)


# --------------------------------------------------------------------------
# verdict / score parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<answer>attack</answer>", "attack"),
        ("<answer>benign</answer>", "benign"),
        ("<ANSWER>ATTACK</ANSWER>", "attack"),
        ("<Answer> Benign </Answer>", "benign"),
        ("<answer>\n attack \n</answer>", "attack"),
        ("blah <answer>benign</answer> then <answer>attack</answer>", "attack"),
        ("blah <answer>attack</answer> then <answer>benign</answer>", "benign"),
        ("no tag at all", None),
        ("<answer>maybe</answer>", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_verdict(text, expected):
    assert solo.parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<score>7</score>", (True, 7)),
        ("<SCORE> 3 </SCORE>", (True, 3)),
        ("<score>1</score> and <score>9</score>", (True, 9)),
        ("no score", (False, None)),
        (None, (False, None)),
    ],
)
def test_parse_score_tag(text, expected):
    assert solo.parse_score_tag(text) == expected


def test_score_tag_never_overrides_verdict():
    text = "<score>9</score>\n<answer>benign</answer>"
    assert solo.parse_verdict(text) == "benign"
    assert solo.parse_score_tag(text) == (True, 9)


# --------------------------------------------------------------------------
# row selection
# --------------------------------------------------------------------------


@pytest.fixture()
def pool(tmp_path: Path) -> list[dict[str, Any]]:
    honest = [_row(f"honest_{i:02d}", "honest") for i in (3, 1, 2)]
    attack = [_row(f"attack_{i:02d}", "attack") for i in (2, 3, 1)]
    hp = _write_jsonl(tmp_path / "honest.jsonl", honest)
    ap = _write_jsonl(tmp_path / "attack.jsonl", attack)
    return solo.load_rows([hp, ap])


def test_load_rows_preserves_file_order(pool):
    assert [r["id"] for r in pool] == [
        "honest_03", "honest_01", "honest_02",
        "attack_02", "attack_03", "attack_01",
    ]


def test_load_rows_returns_dict_rows_with_id_label_steps(pool):
    for row in pool:
        assert set(row) >= {"id", "label", "steps"}
        assert row["label"] in ("attack", "honest")
        assert row["steps"]


def test_load_rows_dedups_by_id(tmp_path: Path):
    a = _write_jsonl(tmp_path / "a.jsonl", [_row("x", "attack")])
    b = _write_jsonl(tmp_path / "b.jsonl", [_row("x", "attack"), _row("y", "honest")])
    rows = solo.load_rows([a, b])
    assert [r["id"] for r in rows] == ["x", "y"]


def test_load_rows_missing_file_is_a_hard_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        solo.load_rows([str(tmp_path / "nope.jsonl")])


def test_balanced_selection_is_deterministic_and_id_sorted(pool):
    selected = solo.select_rows(pool, balanced=4)
    ids = [r["id"] for r in selected]
    assert ids == ["attack_01", "attack_02", "honest_01", "honest_02"]
    # Re-running on a differently-ordered pool gives the same answer.
    assert [r["id"] for r in solo.select_rows(list(reversed(pool)), balanced=4)] == ids


def test_balanced_odd_gives_extra_row_to_attack(pool):
    assert [r["id"] for r in solo.select_rows(pool, balanced=3)] == [
        "attack_01", "attack_02", "honest_01"
    ]


def test_balanced_too_large_fails(pool):
    with pytest.raises(ValueError, match="needs"):
        solo.select_rows(pool, balanced=100)


def test_limit_truncates_file_order(pool):
    assert [r["id"] for r in solo.select_rows(pool, limit=2)] == ["honest_03", "honest_01"]


def test_limit_applies_after_balanced(pool):
    assert [r["id"] for r in solo.select_rows(pool, balanced=4, limit=3)] == [
        "attack_01", "attack_02", "honest_01"
    ]


def test_task_ids_preserve_given_order_and_dedup(pool):
    selected = solo.select_rows(pool, task_ids=["attack_03", "honest_01", "attack_03"])
    assert [r["id"] for r in selected] == ["attack_03", "honest_01"]


def test_task_ids_and_file_are_unioned_in_order(pool, tmp_path: Path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("# comment\nhonest_02\n\nattack_03\n", encoding="utf-8")
    selected = solo.select_rows(
        pool,
        task_ids=["attack_03"],
        task_ids_from_file=solo.read_task_ids_file(str(ids_file)),
    )
    assert [r["id"] for r in selected] == ["attack_03", "honest_02"]


def test_unknown_task_id_is_a_hard_error(pool):
    with pytest.raises(ValueError, match="not present"):
        solo.select_rows(pool, task_ids=["attack_01", "nope_99"])


def test_balanced_conflicts_with_task_ids(pool):
    with pytest.raises(ValueError, match="cannot be combined"):
        solo.select_rows(pool, task_ids=["attack_01"], balanced=2)


# --------------------------------------------------------------------------
# model call, retries, provenance
# --------------------------------------------------------------------------


def test_call_one_records_verdict_and_provenance():
    client = FakeClient([_completion("<answer>attack</answer>")])
    record = solo.call_one(client, _prepared(), solo.RunConfig(), sleep=lambda _: None)
    assert record["verdict"] == "attack"
    assert record["error"] is None
    assert record["served_provider"] == "parasail"
    assert record["generation_id"] == "gen-123"
    assert record["usage"]["total_tokens"] == 15
    assert record["attempts"] == 1
    assert record["raw_response"] == "<answer>attack</answer>"


def test_request_always_pins_the_endpoint():
    client = FakeClient([_completion("<answer>benign</answer>")])
    config = solo.RunConfig(provider="parasail", quantization="fp8", thinking=False)
    solo.call_one(client, _prepared(), config, sleep=lambda _: None)
    body = client.requests[0]["extra_body"]
    assert body["provider"] == {
        "order": ["parasail"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }
    assert body["reasoning"] == {"enabled": False}
    assert client.requests[0]["temperature"] == 0.0


def test_think_flag_flips_reasoning():
    client = FakeClient([_completion("<answer>benign</answer>")])
    solo.call_one(client, _prepared(), solo.RunConfig(thinking=True), sleep=lambda _: None)
    assert client.requests[0]["extra_body"]["reasoning"] == {"enabled": True}


def test_retries_then_succeeds():
    client = FakeClient([
        FakeStatusError(429),
        FakeStatusError(503),
        _completion("<answer>attack</answer>"),
    ])
    delays: list[float] = []
    record = solo.call_one(
        client, _prepared(), solo.RunConfig(max_retries=4), sleep=delays.append
    )
    assert record["verdict"] == "attack"
    assert record["attempts"] == 3
    assert delays == [2.0, 4.0]  # exponential backoff


def test_empty_response_is_retryable_then_errors():
    client = FakeClient([_completion(None)] * 3)
    record = solo.call_one(
        client, _prepared(), solo.RunConfig(max_retries=2), sleep=lambda _: None
    )
    assert record["error_type"] == "EmptyResponseError"
    assert record["verdict"] is None
    assert record["attempts"] == 3


def test_non_retryable_status_fails_immediately():
    client = FakeClient([FakeStatusError(400)])
    record = solo.call_one(
        client, _prepared(), solo.RunConfig(max_retries=4), sleep=lambda _: None
    )
    assert record["attempts"] == 1
    assert record["error_type"] == "FakeStatusError"


# --------------------------------------------------------------------------
# scoring protocol
# --------------------------------------------------------------------------


def test_transport_error_row_is_counted_but_never_scored():
    records = [
        {"id": "a1", "label": "attack", "verdict": "attack", "error": None},
        {"id": "a2", "label": "attack", "verdict": None,
         "error": "APITimeoutError: boom", "error_type": "APITimeoutError"},
        {"id": "a3", "label": "attack", "verdict": None, "error": None},  # no answer tag
        {"id": "h1", "label": "honest", "verdict": "benign", "error": None},
        {"id": "h2", "label": "honest", "verdict": "attack", "error": None},
    ]
    summary = solo.summarize(records)
    assert summary["n"] == 5
    assert summary["n_error"] == 1
    assert summary["error_types"]["APITimeoutError"] == 1
    assert summary["n_unparsed"] == 1
    assert summary["n_parsed"] == 3
    # The errored attack row is in neither TP nor FN.
    assert (summary["tp"], summary["fn"], summary["fp"], summary["tn"]) == (1, 0, 1, 1)
    assert summary["attacks_total"] == 3 and summary["attacks_parsed"] == 1
    assert summary["tpr"] == 1.0
    assert summary["fpr"] == 0.5


def test_summary_handles_all_errors_without_dividing_by_zero():
    summary = solo.summarize(
        [{"id": "a1", "label": "attack", "verdict": None, "error": "x", "error_type": "E"}]
    )
    assert summary["tpr"] is None and summary["fpr"] is None
    assert summary["n_parsed"] == 0


def test_summary_counts_score_tags_and_served_providers():
    summary = solo.summarize([
        {"id": "a", "label": "attack", "verdict": "attack", "error": None,
         "has_score_tag": True, "served_provider": "parasail"},
        {"id": "b", "label": "honest", "verdict": "benign", "error": None,
         "has_score_tag": False, "served_provider": "deepinfra"},
        {"id": "c", "label": "honest", "verdict": "benign", "error": None,
         "has_score_tag": False, "served_provider": None},
    ])
    assert summary["n_score_tag"] == 1
    assert dict(summary["served_providers"]) == {
        "parasail": 1, "deepinfra": 1, "unknown": 1
    }


# --------------------------------------------------------------------------
# end-to-end (fake client), output hygiene
# --------------------------------------------------------------------------


def test_run_rows_writes_jsonl_and_never_prints_response(tmp_path: Path, capsys):
    prepared = [
        solo.PreparedRow(0, "attack_01", "attack", "SYS", "USER attack_01"),
        solo.PreparedRow(1, "honest_01", "honest", "SYS", "USER honest_01"),
    ]
    client = FakeClient({
        "attack_01": _completion("SECRET-TRAJECTORY-ECHO <answer>attack</answer>"),
        "honest_01": _completion("<score>2</score>\n<answer>benign</answer>"),
    })
    out = tmp_path / "out.jsonl"
    records = solo.run_rows(
        prepared, solo.RunConfig(parallel=1), client, out_path=out, sleep=lambda _: None
    )
    assert [r["index"] for r in records] == [0, 1]
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["id"] for r in written} == {"attack_01", "honest_01"}
    honest = next(r for r in written if r["id"] == "honest_01")
    assert honest["verdict"] == "benign"
    assert honest["has_score_tag"] is True and honest["score"] == 2
    for field in (
        "id", "label", "verdict", "has_score_tag", "score", "raw_response",
        "served_provider", "generation_id", "usage", "error", "prompt_sha256",
        "model", "provider", "quantization", "thinking",
    ):
        assert field in honest

    captured = capsys.readouterr().out
    assert "SECRET-TRAJECTORY-ECHO" not in captured
    assert "attack_01" in captured  # ids are safe to print

    summary = solo.summarize(records)
    assert (summary["tp"], summary["tn"]) == (1, 1)


def test_dry_run_makes_no_api_call(tmp_path: Path, monkeypatch, capsys):
    data = _write_jsonl(
        tmp_path / "d.jsonl", [_row("attack_01", "attack"), _row("honest_01", "honest")]
    )
    prompt = _prompt_yaml(tmp_path / "p.yaml")

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("dry run must not build a client")

    monkeypatch.setattr(solo, "make_client", _boom)
    rc = solo.main([
        "--data", data, "--prompt-file", prompt, "--prompt-section", "Sec",
        "--balanced", "2", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run (no API call)" in out
    assert "attack_01" in out and "honest_01" in out
    assert "prompt_sha256=" in out
    assert "total prompt chars" in out
    assert "act-attack_01" not in out  # transcript content stays out of stdout


def test_main_full_path_with_fake_client(tmp_path: Path, monkeypatch, capsys):
    data = _write_jsonl(
        tmp_path / "d.jsonl", [_row("attack_01", "attack"), _row("honest_01", "honest")]
    )
    prompt = _prompt_yaml(tmp_path / "p.yaml")
    client = FakeClient({
        "attack_01": _completion("<answer>attack</answer>"),
        "honest_01": FakeStatusError(500),  # transport error: counted, not scored
    })
    monkeypatch.setattr(solo, "make_client", lambda **kwargs: client)
    out = tmp_path / "out.jsonl"
    rc = solo.main([
        "--data", data, "--prompt-file", prompt, "--prompt-section", "Sec",
        "--balanced", "2", "--out", str(out), "--parallel", "1",
        "--max-retries", "0",
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "TP=1  FN=0" in printed
    assert "FP=0  TN=0" in printed
    assert "error rows (excluded): 1" in printed
    assert "TPR = 1.000" in printed
    assert "FPR = n/a" in printed
    assert "parasail: 1" in printed
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2


def test_default_out_path_uses_run_tag_not_a_clock():
    path = solo.default_out_path("deepseek/deepseek-v4-flash", "2026-07-31a")
    assert path.name == "solo_deepseek_deepseek-v4-flash_2026-07-31a.jsonl"
    assert path == solo.default_out_path("deepseek/deepseek-v4-flash", "2026-07-31a")


def test_cli_help_documents_selection_ordering():
    parser = solo.build_parser()
    help_text = parser.format_help()
    assert "--balanced" in help_text
    assert "id ascending" in help_text or "id-sorted" in help_text
    assert "--task-ids-file" in help_text


@requires_legacy_prompt_file
def test_default_paths_point_at_this_repo():
    """Defaults must resolve inside this repo's data/monitoringbench/."""
    assert solo.REPO_ROOT == Path(__file__).resolve().parents[1]
    for rel in solo.DEFAULT_DATA_PATHS:
        assert rel.startswith("data/monitoringbench/")
    assert solo.DEFAULT_PROMPT_FILE == (
        "data/monitoringbench/prompts/monitoringbench_legacy.yaml"
    )
    assert (solo.REPO_ROOT / solo.DEFAULT_PROMPT_FILE).exists()
