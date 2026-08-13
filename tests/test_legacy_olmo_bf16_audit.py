from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

import infra.legacy_olmo_bf16 as legacy
from infra.run_common import validate_local_policy_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(
    path: Path, *, dtype: str = "BF16", payload: bytes = b"12345678"
) -> None:
    header = {
        "model.weight": {
            "dtype": dtype,
            "shape": [2, 2],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _tiny_source(root: Path, *, dtype: str = "BF16") -> tuple[dict, dict]:
    root.mkdir()
    upstream_config = {
        "architectures": ["Olmo3ForCausalLM"],
        "dtype": "float32",
        "hidden_size": 2,
    }
    legacy_config = {**upstream_config, "torch_dtype": "bfloat16"}
    (root / "config.json").write_text(json.dumps(legacy_config))
    (root / "tokenizer.json").write_text('{"version":"1.0"}\n')
    shard = root / "model-00001-of-00001.safetensors"
    _write_shard(shard, dtype=dtype)
    index = {
        "metadata": {"total_size": 8},
        "weight_map": {"model.weight": shard.name},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    reference = {
        "schema_version": 1,
        "reference_type": "pinned-huggingface-safetensors-layout",
        "repo": "allenai/test-olmo",
        "revision": "a" * 40,
        "index_sha256": "0" * 64,
        "source_total_parameters": 4,
        "source_tensor_bytes": 16,
        "expected_bf16_tensor_bytes": 8,
        "expected_weight_file_bytes": shard.stat().st_size,
        "output_config_sha256": "0" * 64,
        "output_index_sha256": "0" * 64,
        "legacy_source_config_fields": {
            "dtype": "float32",
            "torch_dtype": "bfloat16",
        },
        "support_sha256": {
            name: _sha256(root / "tokenizer.json")
            for name in legacy.LEGACY_REQUIRED_SUPPORT_FILES
        },
        "shards": {
            shard.name: {
                "source_file_bytes": 16,
                "source_sha256": "0" * 64,
                "source_header_sha256": "0" * 64,
                "source_header_bytes": 1,
                "tensor_count": 1,
                "source_tensor_bytes": 16,
                "bf16_tensor_bytes": 8,
                "tensors": {"model.weight": {"shape": [2, 2], "data_offsets": [0, 8]}},
            }
        },
    }
    normalized = {**legacy_config, "dtype": "bfloat16", "torch_dtype": "bfloat16"}
    normalized_bytes = (
        json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    ).encode()
    reference["output_config_sha256"] = hashlib.sha256(normalized_bytes).hexdigest()
    return reference, upstream_config


def _install_tiny_reference(
    monkeypatch: pytest.MonkeyPatch,
    reference: dict,
    upstream_config: dict,
    source: Path,
) -> None:
    reference_sha = legacy._canonical_json_sha256(reference)
    config_sha = legacy._canonical_json_sha256(upstream_config)
    monkeypatch.setattr(legacy, "load_reference", lambda: reference)
    monkeypatch.setattr(legacy, "REFERENCE_JSON_SHA256", reference_sha)
    monkeypatch.setattr(legacy, "LEGACY_CONFIG_SEMANTIC_SHA256", config_sha)
    monkeypatch.setattr(
        legacy, "LEGACY_CONFIG_FILE_SHA256", _sha256(source / "config.json")
    )
    monkeypatch.setattr(legacy, "LEGACY_SOURCE_PATH", str(source))
    monkeypatch.setattr(legacy, "LEGACY_REQUIRED_SUPPORT_FILES", ("tokenizer.json",))


def test_audit_hardlinks_without_modifying_source_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source)
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    before = {path.name: _sha256(path) for path in source.iterdir()}
    output = tmp_path / "audited"

    manifest = legacy.audit_legacy_artifact(
        source, output, link_mode="hardlink", progress=lambda _message: None
    )

    assert {path.name: _sha256(path) for path in source.iterdir()} == before
    assert legacy.AUDIT_MARKER_NAME not in before
    output_config = json.loads((output / "config.json").read_text())
    assert output_config["dtype"] == output_config["torch_dtype"] == "bfloat16"
    assert _sha256(output / "config.json") == reference["output_config_sha256"]
    assert (source / "config.json").stat().st_ino != (
        output / "config.json"
    ).stat().st_ino
    assert (source / "model-00001-of-00001.safetensors").stat().st_ino == (
        output / "model-00001-of-00001.safetensors"
    ).stat().st_ino
    assert manifest["conversion_provenance"].startswith("unknown;")
    assert "converter" not in manifest
    assert manifest["tensor_count"] == 1
    assert legacy.validate_audited_artifact(output) == manifest
    assert validate_local_policy_artifact(
        str(output),
        audited_legacy_olmo32_path=str(output),
    ) == str(output)


def test_validation_rejects_data_change_even_when_header_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source)
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    output = tmp_path / "audited"
    legacy.audit_legacy_artifact(
        source, output, link_mode="copy", progress=lambda _message: None
    )
    shard = output / "model-00001-of-00001.safetensors"
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(legacy.LegacyArtifactError, match="bytes/record differ"):
        legacy.validate_audited_artifact(output)


def test_validation_rejects_runtime_config_changed_back_to_float32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source)
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    output = tmp_path / "audited"
    legacy.audit_legacy_artifact(
        source, output, link_mode="copy", progress=lambda _message: None
    )
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["dtype"] = "float32"
    config_path.write_text(json.dumps(config))

    with pytest.raises(legacy.LegacyArtifactError, match="does not select OLMo3 BF16"):
        legacy.validate_audited_artifact(output)


def test_audit_rejects_non_bf16_header_without_committing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source, dtype="F16")
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    output = tmp_path / "audited"

    with pytest.raises(legacy.LegacyArtifactError, match="dtype/shape/offset mismatch"):
        legacy.audit_legacy_artifact(
            source, output, link_mode="hardlink", progress=lambda _message: None
        )
    assert not output.exists()
    staging = output.with_name(f".{output.name}.audit-{os.getpid()}.partial")
    assert not staging.exists()


@pytest.mark.parametrize("field", ["materialization", "audited_at", "tensor_count"])
def test_validation_rejects_tampered_marker_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source)
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    output = tmp_path / "audited"
    legacy.audit_legacy_artifact(
        source, output, link_mode="copy", progress=lambda _message: None
    )
    marker = output / legacy.AUDIT_MARKER_NAME
    manifest = json.loads(marker.read_text())
    manifest[field] = {
        "materialization": ["reflink"],
        "audited_at": "yesterday",
        "tensor_count": 9,
    }[field]
    marker.write_text(json.dumps(manifest))

    with pytest.raises(legacy.LegacyArtifactError):
        legacy.validate_audited_artifact(output)


def test_audit_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "legacy"
    reference, upstream_config = _tiny_source(source)
    _install_tiny_reference(monkeypatch, reference, upstream_config, source)
    output = tmp_path / "audited"
    output.mkdir()

    with pytest.raises(legacy.LegacyArtifactError, match="already exists"):
        legacy.audit_legacy_artifact(source, output, progress=lambda _message: None)
