from __future__ import annotations

import csv
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "runtime_image" / "generate_runtime_metadata.py"
INPUT_CHECK = ROOT / "runtime_image" / "verify_build_inputs.sh"


def _module():
    spec = importlib.util.spec_from_file_location("runtime_metadata", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _install_distribution(
    site_packages: Path,
    *,
    name: str,
    version: str,
    module: str,
    direct_url: bool = False,
) -> None:
    package = site_packages / module
    package.mkdir()
    (package / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="ascii"
    )
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="ascii",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
        "Tag: py3-none-any\n",
        encoding="ascii",
    )
    if direct_url:
        (dist_info / "direct_url.json").write_bytes(
            _canonical(
                {
                    "archive_info": {"hash": "sha256=artifact"},
                    "url": "file:///build/wheelhouse/torch.whl",
                }
            )
        )
    record_paths = [
        f"{module}/__init__.py",
        f"{dist_info.name}/METADATA",
        f"{dist_info.name}/WHEEL",
        f"{dist_info.name}/RECORD",
    ]
    if direct_url:
        record_paths.append(f"{dist_info.name}/direct_url.json")
    with (dist_info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for relative in record_paths:
            writer.writerow((relative, "", ""))


def _write_wheel(path: Path, *, name: str, version: str, module: str) -> None:
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    members = {
        f"{module}/__init__.py": f'__version__ = "{version}"\n'.encode("ascii"),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ).encode("ascii"),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }
    record_path = f"{dist_info}/RECORD"
    rows = []
    for member, payload in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        rows.append((member, "sha256=" + digest.decode("ascii"), str(len(payload))))
    rows.append((record_path, "", ""))
    record = "".join(
        ",".join(row) + "\r\n" for row in rows
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as wheel:
        for member, payload in members.items():
            wheel.writestr(member, payload)
        wheel.writestr(record_path, record)


def _fixture(tmp_path: Path):
    module = _module()
    staging = tmp_path / "stage"
    runtime_root = staging / module.RUNTIME_ROOT.removeprefix("/")
    python = staging / module.RUNTIME_PYTHON.removeprefix("/")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixed-python-binary")
    python.chmod(0o555)
    site_packages = (
        runtime_root / "python/lib/python3.12/site-packages"
    )
    site_packages.mkdir(parents=True)
    _install_distribution(
        site_packages,
        name="demo-runtime",
        version="1.2.3",
        module="demo_runtime",
    )
    _install_distribution(
        site_packages,
        name="torch",
        version="2.11.0+cu130",
        module="torch",
        direct_url=True,
    )

    inputs = tmp_path / "inputs"
    artifacts = inputs / "artifacts"
    artifacts.mkdir(parents=True)
    artifact_records = []
    for name, version in (
        ("demo-runtime", "1.2.3"),
        ("torch", "2.11.0+cu130"),
    ):
        artifact = artifacts / f"{name}-{version}.whl"
        _write_wheel(
            artifact,
            name=name,
            version=version,
            module=name.replace("-", "_"),
        )
        artifact_records.append(
            {
                "artifact": {
                    "path": f"artifacts/{artifact.name}",
                    "sha256": _sha256(artifact.read_bytes()),
                },
                "name": name,
                "version": version,
            }
        )
    lock = {
        "distributions": artifact_records,
        "python_version": "3.12.11",
        "runtime_root": module.RUNTIME_ROOT,
        "schema": module.LOCK_SCHEMA,
    }
    lock_path = inputs / "dependency.lock"
    lock_path.write_bytes(_canonical(lock))
    spec = {
        "cuda": {
            "compiled_extensions": [
                {"module": "flash_attn_2_cuda"},
                {"module": "vllm._C"},
            ],
            "compute_capability": [9, 0],
            "device_count": 1,
            "torch_cuda_version": "13.0",
            "torch_distribution": "torch",
            "torch_version": "2.11.0+cu130",
        },
        "python": {"path": module.RUNTIME_PYTHON, "version": "3.12.11"},
        "required_imports": [
            {
                "distribution": "demo-runtime",
                "module": "demo_runtime",
                "version": "1.2.3",
            },
            {"distribution": "torch", "module": "torch", "version": "2.11.0+cu130"},
        ],
        "runtime_root": module.RUNTIME_ROOT,
        "schema": module.SPEC_SCHEMA,
        "site_packages_path": (
            module.RUNTIME_ROOT + "/python/lib/python3.12/site-packages"
        ),
    }
    spec_path = inputs / "build-spec.json"
    spec_path.write_bytes(_canonical(spec))
    return module, staging, runtime_root, lock_path, spec_path


def test_generator_emits_scheduler_manifest_inventory_and_executable_probes(
    tmp_path: Path,
) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)

    manifest = module.generate(staging, lock_path, spec_path)

    names = {
        "dependency.lock",
        "installed-distributions.json",
        "required-import-probe.json",
        "cuda-compatibility-probe.json",
        "runtime-manifest.json",
    }
    assert names <= {path.name for path in runtime_root.iterdir()}
    for name in names:
        payload = (runtime_root / name).read_bytes()
        assert not payload.endswith(b"\n")
        if name.endswith(".json") or name.endswith(".lock"):
            assert _canonical(json.loads(payload)) == payload
        assert os.stat(runtime_root / name).st_mode & 0o777 == 0o444

    unsigned = dict(manifest)
    claimed = unsigned.pop("runtime_manifest_sha256")
    assert claimed == _sha256(_canonical(unsigned))
    assert manifest["python"] == {
        "path": module.RUNTIME_PYTHON,
        "realpath": module.RUNTIME_PYTHON,
        "sha256": _sha256(b"fixed-python-binary"),
        "version": "3.12.11",
    }

    inventory = json.loads((runtime_root / "installed-distributions.json").read_bytes())
    assert [item["name"] for item in inventory["distributions"]] == [
        "demo-runtime",
        "torch",
    ]
    torch = inventory["distributions"][1]
    assert torch["direct_origin"]["path"].endswith("direct_url.json")
    assert torch["direct_origin"]["sha256"].startswith("sha256:")
    assert json.loads(torch["direct_origin"]["canonical_json"])["url"].startswith(
        "file:///"
    )

    import_probe = json.loads((runtime_root / "required-import-probe.json").read_bytes())
    cuda_probe = json.loads((runtime_root / "cuda-compatibility-probe.json").read_bytes())
    for probe, source in (
        (import_probe, module.FIXED_IMPORT_SOURCE),
        (cuda_probe, module.FIXED_CUDA_SOURCE),
    ):
        assert set(probe) == {"schema", "argv", "expected_result"}
        assert probe["argv"][:4] == [module.RUNTIME_PYTHON, "-I", "-c", source]
        assert len(probe["argv"]) == 5
        assert probe["argv"][4].encode("ascii") == _canonical(
            probe["expected_result"]
        )
    assert cuda_probe["expected_result"]["gpu"] == {
        "available": True,
        "compute_capabilities": [[9, 0]],
        "count": 1,
    }


def test_generator_refuses_unhashed_or_incomplete_lock(tmp_path: Path) -> None:
    module, _, _, lock_path, spec_path = _fixture(tmp_path)
    lock = json.loads(lock_path.read_bytes())
    lock["distributions"][0]["artifact"]["sha256"] = "sha256:" + "0" * 64
    lock_path.write_bytes(_canonical(lock))

    with pytest.raises(module.Refusal, match="artifact digest differs"):
        module.validate_lock(lock_path)

    module, staging, _, lock_path, spec_path = _fixture(tmp_path / "second")
    site_packages = (
        staging
        / module.RUNTIME_ROOT.removeprefix("/")
        / "python/lib/python3.12/site-packages"
    )
    _install_distribution(
        site_packages,
        name="ambient-extra",
        version="9.9.9",
        module="ambient_extra",
    )
    _, locked, python_version = module.validate_lock(lock_path)
    spec = module.validate_spec(spec_path, locked, python_version)
    with pytest.raises(module.Refusal, match="absent from dependency lock"):
        module.inventory(staging, spec, locked)


def test_generator_refuses_noncanonical_inputs_and_output_collisions(
    tmp_path: Path,
) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)
    lock = json.loads(lock_path.read_bytes())
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="ascii")
    with pytest.raises(module.Refusal, match="not canonical"):
        module.validate_lock(lock_path)

    lock_path.write_bytes(_canonical(lock))
    (runtime_root / "dependency.lock").write_bytes(b"preexisting")
    with pytest.raises(module.Refusal, match="already exists"):
        module.generate(staging, lock_path, spec_path)


def test_lock_rejects_nonwheel_and_inventory_rejects_post_install_patch(
    tmp_path: Path,
) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)
    lock = json.loads(lock_path.read_bytes())
    artifact_relative = lock["distributions"][0]["artifact"]["path"]
    artifact = lock_path.parent / artifact_relative
    artifact.write_bytes(b"not a wheel")
    lock["distributions"][0]["artifact"]["sha256"] = _sha256(artifact.read_bytes())
    lock_path.write_bytes(_canonical(lock))
    with pytest.raises(module.Refusal, match="valid wheel ZIP"):
        module.validate_lock(lock_path)

    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path / "patched")
    (runtime_root / "python/lib/python3.12/site-packages/demo_runtime/__init__.py").write_bytes(
        b"# untracked post-install patch\n"
    )
    _, locked, python_version = module.validate_lock(lock_path)
    spec = module.validate_spec(spec_path, locked, python_version)
    with pytest.raises(module.Refusal, match="differ from locked wheel source"):
        module.inventory(staging, spec, locked)


def test_inventory_refuses_pep660_editable_origin(tmp_path: Path) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)
    direct_url = next(runtime_root.glob("**/torch-*.dist-info/direct_url.json"))
    direct_url.write_bytes(
        _canonical({"dir_info": {"editable": True}, "url": "file:///workspace/src"})
    )
    _, locked, python_version = module.validate_lock(lock_path)
    spec = module.validate_spec(spec_path, locked, python_version)

    with pytest.raises(module.Refusal, match="editable direct_url.json"):
        module.inventory(staging, spec, locked)


def test_inventory_binds_raw_and_canonical_direct_origin(tmp_path: Path) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)
    direct_url = next(runtime_root.glob("**/torch-*.dist-info/direct_url.json"))
    document = json.loads(direct_url.read_bytes())
    noncanonical = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    direct_url.write_bytes(noncanonical)
    _, locked, python_version = module.validate_lock(lock_path)
    spec = module.validate_spec(spec_path, locked, python_version)

    installed = module.inventory(staging, spec, locked)

    origin = installed["distributions"][1]["direct_origin"]
    assert origin["sha256"] == _sha256(noncanonical)
    assert origin["canonical_json"].encode("ascii") == _canonical(document)


@pytest.mark.parametrize("unsafe_kind", ["unclaimed", "nested-egg", "symlink"])
def test_inventory_refuses_unclaimed_or_unsafe_site_packages_entries(
    tmp_path: Path, unsafe_kind: str
) -> None:
    module, staging, runtime_root, lock_path, spec_path = _fixture(tmp_path)
    site_packages = runtime_root / "python/lib/python3.12/site-packages"
    if unsafe_kind == "unclaimed":
        (site_packages / "unclaimed.py").write_bytes(b"not in any RECORD")
    elif unsafe_kind == "nested-egg":
        nested = site_packages / "package" / "legacy.egg-info"
        nested.mkdir(parents=True)
        (nested / "PKG-INFO").write_bytes(b"legacy")
    else:
        (site_packages / "redirected-package").symlink_to(site_packages / "torch")
    _, locked, python_version = module.validate_lock(lock_path)
    spec = module.validate_spec(spec_path, locked, python_version)

    with pytest.raises(module.Refusal, match="unsafe|not complete"):
        module.inventory(staging, spec, locked)


def test_checked_in_build_frontier_fails_closed_without_fake_lock() -> None:
    assert not (ROOT / "runtime_image" / "dependency.lock").exists()
    assert not (ROOT / "runtime_image" / "build-spec.json").exists()

    result = subprocess.run(
        ["bash", str(INPUT_CHECK)], text=True, capture_output=True, cwd=ROOT
    )

    assert result.returncode == 2
    assert "D043 REFUSED" in result.stderr
    assert "uv.lock" in result.stderr


def test_fixed_probe_sources_match_scheduler_contract_digests() -> None:
    module = _module()
    assert hashlib.sha256(module.FIXED_IMPORT_SOURCE.encode("ascii")).hexdigest() == (
        "4cff35d9b6d9d7ce0d9877f8a48c910950e86110d1b6981252cf84a75c0e0c9c"
    )
    assert hashlib.sha256(module.FIXED_CUDA_SOURCE.encode("ascii")).hexdigest() == (
        "a294898149dba0c6b60486f169e985150e8c152a1ce11c917cf1f576b7cb0ebc"
    )
