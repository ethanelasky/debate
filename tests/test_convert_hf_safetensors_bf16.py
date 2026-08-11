"""Offline tests for the shard-streaming BF16 model converter."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import scripts.convert_hf_safetensors_bf16 as converter


REPO = "fixture/tiny-model"
REVISION = "a" * 40


def _build_source(root: Path) -> dict[str, set[str]]:
    root.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": {
            "embed.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "position_ids": torch.tensor([0, 1, 2], dtype=torch.int64),
        },
        "model-00002-of-00002.safetensors": {
            "head.weight": torch.linspace(0, 1, 10, dtype=torch.float64).reshape(2, 5),
            "enabled": torch.tensor([True, False], dtype=torch.bool),
        },
    }
    weight_map: dict[str, str] = {}
    for number, (name, tensors) in enumerate(shards.items(), 1):
        save_file(tensors, root / name, metadata={"source-note": f"shard-{number}"})
        weight_map.update({tensor_name: name for tensor_name in tensors})

    source_total = sum(
        tensor.numel() * tensor.element_size()
        for tensors in shards.values()
        for tensor in tensors.values()
    )
    (root / converter.INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": source_total, "format": "pt"},
                "weight_map": weight_map,
            }
        )
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyForCausalLM"],
                "torch_dtype": "float32",
                "dtype": "float32",
            }
        )
    )
    (root / "generation_config.json").write_text('{"do_sample": false}\n')
    (root / "tokenizer_config.json").write_text('{"model_max_length": 1024}\n')
    (root / "tokenizer.json").write_text('{"version": "1.0"}\n')
    return {name: set(tensors) for name, tensors in shards.items()}


def _convert(tmp_path: Path, **kwargs) -> converter.ConversionResult:
    source = tmp_path / "source"
    if not source.exists():
        _build_source(source)
    progress = kwargs.pop("progress", lambda _message: None)
    return converter.convert_repository(
        repo=REPO,
        revision=REVISION,
        source_dir=source,
        output_dir=tmp_path / "output",
        staging_dir=tmp_path / "staging",
        progress=progress,
        **kwargs,
    )


def _rewrite_valid_shard(path: Path, *, metadata_extra: dict[str, str] | None = None) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
        metadata = dict(handle.metadata() or {})
    for name, tensor in tensors.items():
        if tensor.is_floating_point():
            tensors[name] = tensor + 1
            break
    metadata.update(metadata_extra or {})
    replacement = path.with_suffix(".replacement")
    save_file(tensors, replacement, metadata=metadata)
    replacement.replace(path)


def test_offline_conversion_streams_shards_and_builds_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    keys_by_shard = _build_source(tmp_path / "source")

    # Local source mode must not instantiate the network-backed source at all.
    monkeypatch.setattr(
        converter,
        "HubModelSource",
        lambda *_args, **_kwargs: pytest.fail("offline conversion attempted network setup"),
    )
    result = _convert(tmp_path)

    output = tmp_path / "output"
    staging_files = {
        path.relative_to(tmp_path / "staging").as_posix()
        for path in (tmp_path / "staging").rglob("*")
        if path.is_file()
    }
    assert staging_files == {converter.STAGING_MARKER_NAME}
    assert result.shards == 2
    assert result.weight_file_bytes == sum(
        (output / name).stat().st_size for name in keys_by_shard
    )

    for number, (name, expected_keys) in enumerate(keys_by_shard.items(), 1):
        with safe_open(output / name, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == expected_keys
            assert handle.get_tensor(
                "embed.weight" if number == 1 else "head.weight"
            ).dtype == torch.bfloat16
            nonfloating = "position_ids" if number == 1 else "enabled"
            expected_dtype = torch.int64 if number == 1 else torch.bool
            assert handle.get_tensor(nonfloating).dtype == expected_dtype
            metadata = handle.metadata()
            assert metadata["source-note"] == f"shard-{number}"
            assert metadata[f"{converter.PROVENANCE_PREFIX}repo"] == REPO
            assert metadata[f"{converter.PROVENANCE_PREFIX}revision"] == REVISION

    config = json.loads((output / "config.json").read_text())
    assert config["torch_dtype"] == "bfloat16"
    assert config["dtype"] == "bfloat16"
    assert (output / "tokenizer.json").read_bytes() == (
        tmp_path / "source" / "tokenizer.json"
    ).read_bytes()
    assert (output / "tokenizer_config.json").read_bytes() == (
        tmp_path / "source" / "tokenizer_config.json"
    ).read_bytes()

    index = json.loads((output / converter.INDEX_NAME).read_text())
    expected_tensor_bytes = 12 * 2 + 3 * 8 + 10 * 2 + 2
    assert index["metadata"] == {"format": "pt", "total_size": expected_tensor_bytes}
    assert result.tensor_bytes == expected_tensor_bytes
    marker = json.loads((output / converter.OUTPUT_MARKER_NAME).read_text())
    assert marker["complete"] is True
    assert set(marker["shards"]) == set(keys_by_shard)
    assert marker["output_index_sha256"] == hashlib.sha256(
        (output / converter.INDEX_NAME).read_bytes()
    ).hexdigest()
    assert marker["output_support_sha256"] == {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in marker["support_files"]
    }


def test_resume_validates_and_skips_complete_output_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _convert(tmp_path)
    output_shards = sorted((tmp_path / "output").glob("model-*.safetensors"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in output_shards}

    def refuse_stage(*_args, **_kwargs):
        pytest.fail("a complete output shard was downloaded or copied again")

    monkeypatch.setattr(converter, "_stage_source_shard", refuse_stage)
    second = _convert(tmp_path)

    assert second == first
    assert {path.name: path.stat().st_mtime_ns for path in output_shards} == mtimes


def test_hard_kill_partials_are_cleaned_only_in_marker_owned_directories(tmp_path: Path):
    _convert(tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"

    # Simulate a kill during conversion: the source shard was fully staged,
    # the atomic BF16 output is incomplete, and another download has a legacy
    # no-PID partial left by the first converter version.
    (output / first).unlink()
    marker_path = output / converter.OUTPUT_MARKER_NAME
    marker = json.loads(marker_path.read_text())
    del marker["shards"][first]
    marker["complete"] = False
    marker_path.write_text(json.dumps(marker))
    shutil.copyfile(source / first, staging / first)
    (output / f".{first}.424242.partial").write_bytes(b"partial converted data")
    (output / ".config.json.424242.partial").write_bytes(b"partial config")
    (staging / f".{second}.download.partial").write_bytes(b"partial download")
    (staging / f".{converter.STAGING_MARKER_NAME}.424242.partial").write_bytes(
        b"partial marker"
    )

    messages: list[str] = []
    _convert(tmp_path, progress=messages.append)

    assert any("cleared 4 owned partial" in message for message in messages)
    assert (output / first).is_file()
    assert not (staging / first).exists()
    assert not list(output.rglob("*.partial"))
    assert not list(staging.rglob("*.partial"))


def test_owned_directory_still_refuses_partial_lookalikes(tmp_path: Path):
    _convert(tmp_path)
    lookalike = tmp_path / "output" / ".model-00001-of-00002.safetensors.mine.partial"
    lookalike.write_bytes(b"not converter-owned")

    with pytest.raises(converter.ConversionError, match="unexpected files"):
        _convert(tmp_path)
    assert lookalike.read_bytes() == b"not converter-owned"


@pytest.mark.parametrize(
    ("directory", "marker_name"),
    [
        ("output", converter.OUTPUT_MARKER_NAME),
        ("staging", converter.STAGING_MARKER_NAME),
    ],
)
def test_first_marker_sigkill_partial_is_recovered(
    tmp_path: Path,
    directory: str,
    marker_name: str,
):
    _build_source(tmp_path / "source")
    root = tmp_path / directory
    root.mkdir()
    # PID reuse must not block recovery: the stable output-adjacent lock, not
    # process-table state, establishes that no conforming writer is live.
    partial = root / f".{marker_name}.{os.getpid()}.partial"
    partial.write_bytes(b'{"schema_version":')

    _convert(tmp_path)

    assert not partial.exists()
    assert (root / marker_name).is_file()


def test_first_marker_partial_with_unowned_data_still_fails_closed(tmp_path: Path):
    _build_source(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    partial = output / f".{converter.OUTPUT_MARKER_NAME}.99999999.partial"
    partial.write_bytes(b'{"schema_version":')
    unrelated = output / "unrelated.txt"
    unrelated.write_text("preserve me")

    with pytest.raises(converter.ConversionError, match="non-empty unowned directory"):
        _convert(tmp_path)
    assert partial.is_file()
    assert unrelated.read_text() == "preserve me"


def test_recorded_output_integrity_cannot_be_rebased(tmp_path: Path):
    _convert(tmp_path)
    shard = tmp_path / "output" / "model-00001-of-00002.safetensors"
    _rewrite_valid_shard(shard, metadata_extra={"post-conversion-change": "yes"})

    with pytest.raises(converter.ConversionError, match="recorded output integrity mismatch"):
        _convert(tmp_path)


def test_unrecorded_self_provenanced_output_is_reconverted_not_blessed(tmp_path: Path):
    _convert(tmp_path)
    output = tmp_path / "output"
    shard_name = "model-00001-of-00002.safetensors"
    shard = output / shard_name
    original_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    marker_path = output / converter.OUTPUT_MARKER_NAME
    marker = json.loads(marker_path.read_text())
    del marker["shards"][shard_name]
    marker["complete"] = False
    marker_path.write_text(json.dumps(marker))

    # Keep every converter provenance key while changing valid BF16 tensor
    # bytes, reproducing the kill-window injection that structural validation
    # alone previously accepted.
    _rewrite_valid_shard(shard, metadata_extra={"injected": "yes"})
    assert hashlib.sha256(shard.read_bytes()).hexdigest() != original_sha256
    messages: list[str] = []

    _convert(tmp_path, progress=messages.append)

    assert any("discarded uncommitted output shard" in message for message in messages)
    with safe_open(shard, framework="pt", device="cpu") as converted:
        with safe_open(
            tmp_path / "source" / shard_name,
            framework="pt",
            device="cpu",
        ) as source:
            assert torch.equal(
                converted.get_tensor("embed.weight"),
                source.get_tensor("embed.weight").to(torch.bfloat16),
            )
        assert "injected" not in (converted.metadata() or {})
    final_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    final_marker = json.loads(marker_path.read_text())
    assert final_marker["shards"][shard_name]["sha256"] == final_sha256


def test_local_source_shard_identity_prevents_mixed_source_resume(tmp_path: Path):
    _convert(tmp_path)
    source_shard = tmp_path / "source" / "model-00001-of-00002.safetensors"
    _rewrite_valid_shard(source_shard)

    with pytest.raises(converter.ConversionError, match="conversion identity mismatch"):
        _convert(tmp_path)


def test_fresh_stage_is_checked_against_plan_to_close_local_source_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    _build_source(source)
    original_stage = converter.LocalModelSource.stage_file
    changed = False

    def mutate_then_stage(self, name: str, destination: Path):
        nonlocal changed
        if not changed:
            changed = True
            _rewrite_valid_shard(self._path(name))
        return original_stage(self, name, destination)

    monkeypatch.setattr(converter.LocalModelSource, "stage_file", mutate_then_stage)
    with pytest.raises(converter.ConversionError, match="immutable conversion plan"):
        _convert(tmp_path)
    assert changed is True
    assert not list((tmp_path / "output").glob("model-*.safetensors"))


def test_output_lock_blocks_same_output_with_a_different_staging_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    _build_source(source)
    output = tmp_path / "output"
    stage_one = tmp_path / "stage-one"
    stage_two = tmp_path / "stage-two"

    with converter._exclusive_conversion_locks(output, stage_one):
        monkeypatch.setattr(
            converter,
            "LocalModelSource",
            lambda *_args: pytest.fail("source discovery happened before output locking"),
        )
        with pytest.raises(converter.ConversionError, match="another converter invocation"):
            converter.convert_repository(
                repo=REPO,
                revision=REVISION,
                source_dir=source,
                output_dir=output,
                staging_dir=stage_two,
                progress=lambda _message: None,
            )
    assert not output.exists()
    assert not (stage_two / converter.STAGING_MARKER_NAME).exists()


def test_resume_refuses_corrupt_or_wrong_dtype_output(tmp_path: Path):
    _convert(tmp_path)
    shard = tmp_path / "output" / "model-00001-of-00002.safetensors"
    original = shard.read_bytes()
    shard.write_bytes(original[:-1])

    with pytest.raises(converter.ConversionError, match="invalid safetensors shard"):
        _convert(tmp_path)


@pytest.mark.parametrize("which", ["output", "staging"])
def test_refuses_nonempty_unowned_directories(tmp_path: Path, which: str):
    _build_source(tmp_path / "source")
    mixed = tmp_path / which
    mixed.mkdir()
    (mixed / "unrelated.txt").write_text("do not overwrite")

    with pytest.raises(converter.ConversionError, match="non-empty unowned directory"):
        _convert(tmp_path)
    assert (mixed / "unrelated.txt").read_text() == "do not overwrite"


def test_refuses_identity_mismatch_on_resume(tmp_path: Path):
    _convert(tmp_path)

    with pytest.raises(converter.ConversionError, match="identity mismatch"):
        converter.convert_repository(
            repo=REPO,
            revision="b" * 40,
            source_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
            staging_dir=tmp_path / "staging",
            progress=lambda _message: None,
        )


def test_optional_weight_size_guard_is_decimal_gb_and_fail_closed(tmp_path: Path):
    with pytest.raises(converter.ConversionError, match="expected 55000000000..75000000000"):
        _convert(
            tmp_path,
            expected_size_bytes=(55_000_000_000, 75_000_000_000),
        )

    # The complete shards remain resumable after a guard failure; changing only
    # the launch-time expectation does not rewrite them.
    result = _convert(tmp_path, expected_size_bytes=(1, 1_000_000))
    assert 1 <= result.weight_file_bytes <= 1_000_000


def test_requires_pinned_commit_and_has_no_token_cli_argument():
    with pytest.raises(converter.ConversionError, match="40-character"):
        converter.convert_repository(
            repo=REPO,
            revision="main",
            source_dir=Path("unused"),
            output_dir=Path("unused-output"),
            staging_dir=Path("unused-staging"),
            progress=lambda _message: None,
        )

    option_strings = {
        option
        for action in converter.build_parser()._actions
        for option in action.option_strings
    }
    assert "--token" not in option_strings
    assert "--hf-token" not in option_strings


def test_hub_source_reads_lfs_object_metadata_and_follows_download_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import huggingface_hub

    payload = b"a remote shard payload"
    lfs = SimpleNamespace(
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        pointer_size=128,
    )
    info = SimpleNamespace(
        sha=REVISION,
        siblings=[SimpleNamespace(rfilename="weights.safetensors", size=None, lfs=lfs)],
    )
    api_calls = []
    api_constructors = []

    class FakeApi:
        def __init__(self, *, endpoint, token):
            api_constructors.append({"endpoint": endpoint, "token": token})

        def model_info(self, **kwargs):
            api_calls.append(kwargs)
            return info

    class RedirectedResponse:
        def __init__(self):
            self._data = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size: int = -1) -> bytes:
            return self._data.read(size)

        def geturl(self) -> str:
            return "https://cdn-lfs.example/redirected-object"

    requests = []

    class FakeOpener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return RedirectedResponse()

    monkeypatch.setenv("HF_TOKEN", "unit-test-secret")
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_url",
        lambda repo, name, revision, repo_type, endpoint: (
            f"{endpoint}/{repo}/resolve/{revision}/{name}"
        ),
    )
    monkeypatch.setattr(
        converter.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    source = converter.HubModelSource(REPO, REVISION)
    descriptor = source.files["weights.safetensors"]
    assert descriptor.size == len(payload)
    assert descriptor.sha256 == hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "staged.safetensors"
    source.stage_file("weights.safetensors", destination)

    assert destination.read_bytes() == payload
    assert api_constructors == [
        {"endpoint": converter.HF_CANONICAL_ENDPOINT, "token": False}
    ]
    assert api_calls == [
        {
            "repo_id": REPO,
            "revision": REVISION,
            "token": "unit-test-secret",
            "files_metadata": True,
        }
    ]
    request, timeout = requests[0]
    assert request.full_url == (
        f"{converter.HF_CANONICAL_ENDPOINT}/{REPO}/resolve/{REVISION}/weights.safetensors"
    )
    assert request.get_header("Authorization") == "Bearer unit-test-secret"
    assert timeout == 120


def test_hostile_hf_endpoint_is_rejected_before_api_or_auth_setup(
    monkeypatch: pytest.MonkeyPatch,
):
    import huggingface_hub

    calls: list[str] = []

    class TrapApi:
        def __init__(self, **_kwargs):
            calls.append("api")

    def trap_opener(*_handlers):
        calls.append("opener")
        raise AssertionError("network opener must not be built")

    monkeypatch.setenv("HF_ENDPOINT", "https://attacker.invalid")
    monkeypatch.setenv("HF_TOKEN", "must-not-leave-process")
    monkeypatch.setattr(huggingface_hub, "HfApi", TrapApi)
    monkeypatch.setattr(converter.urllib.request, "build_opener", trap_opener)

    with pytest.raises(converter.ConversionError, match="non-canonical HF_ENDPOINT"):
        converter.HubModelSource(REPO, REVISION)
    assert calls == []


def test_hub_redirect_handler_strips_token_only_when_origin_changes():
    handler = converter._SameOriginAuthRedirectHandler()
    request = converter.urllib.request.Request(
        "https://huggingface.co/fixture/tiny-model/resolve/" + REVISION + "/weights",
        headers={"Authorization": "Bearer unit-test-secret"},
    )

    same_origin = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://huggingface.co/same-origin-target",
    )
    cross_origin = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://cdn-lfs.example/signed-target",
    )

    assert same_origin.get_header("Authorization") == "Bearer unit-test-secret"
    assert cross_origin.get_header("Authorization") is None
