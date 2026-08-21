from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "runtime_image" / "generate_runtime_metadata.py"
PREPARER = ROOT / "runtime_image" / "prepare_binary_seed.py"
SEED_DOCKERFILE = ROOT / "runtime_image" / "Dockerfile.binary-seed"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


PYVENV = (
    b"home = /usr/local/bin\n"
    b"implementation = CPython\n"
    b"uv = 0.12.0\n"
    b"version_info = 3.12.3\n"
    b"include-system-site-packages = false\n"
    b"seed = true\n"
)
TARGET_PYVENV = (
    "home = /usr/bin\n"
    "implementation = CPython\n"
    "uv = 0.12.0\n"
    "version_info = 3.12.3\n"
    "include-system-site-packages = false\n"
    "seed = true\n"
)


PTH = b"import _virtualenv\n"


def _tar_bytes(*, unsafe: str | None = None) -> bytes:
    stream = io.BytesIO()
    prefix = "workspace/envs/verl-b200"
    with tarfile.open(fileobj=stream, mode="w") as archive:
        def directory(relative: str) -> None:
            item = tarfile.TarInfo(f"{prefix}/{relative}".rstrip("/"))
            item.type = tarfile.DIRTYPE
            item.mode = 0o755
            archive.addfile(item)

        def regular(relative: str, payload: bytes, mode: int = 0o644) -> None:
            item = tarfile.TarInfo(f"{prefix}/{relative}")
            item.size = len(payload)
            item.mode = mode
            archive.addfile(item, io.BytesIO(payload))

        directory("")
        for path in (
            "bin",
            "remove",
            "lib",
            "lib/python3.12",
            "lib/python3.12/site-packages",
            "lib/python3.12/site-packages/__pycache__",
            "lib/python3.12/site-packages/demo-1.0.dist-info",
            "lib/python3.12/site-packages/torch-2.11.0.dist-info",
        ):
            directory(path)
        link = tarfile.TarInfo(f"{prefix}/bin/python3.12")
        link.type = tarfile.SYMTYPE
        link.linkname = "escape" if unsafe == "symlink" else "python"
        archive.addfile(link)
        regular("remove/editable-marker", b"removed")
        regular("pyvenv.cfg", PYVENV)
        shebang = b"#!/workspace/envs/verl-b200/bin/python\n"
        if unsafe == "rewrite":
            shebang = b"#!/workspace/envs/verl-b200/bin/python3\n"
        regular("bin/tool", shebang + b"print('ok')\n", 0o755)
        regular(
            "bin/activate",
            b'VIRTUAL_ENV="/workspace/envs/verl-b200"\n',
        )
        regular("lib/python3.12/site-packages/demo.py", b"VALUE = 1\n")
        regular("lib/python3.12/site-packages/torch.py", b"__version__ = '2.11.0+cu130'\n")
        regular("lib/python3.12/site-packages/allowed.pth", PTH)
        regular(
            "lib/python3.12/site-packages/__pycache__/demo.cpython-312.pyc",
            b"fake bytecode /workspace/envs/verl-b200",
        )
        regular(
            "lib/python3.12/site-packages/demo-1.0.dist-info/METADATA",
            b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
        )
        regular(
            "lib/python3.12/site-packages/demo-1.0.dist-info/RECORD",
            b"demo.py,,\r\nallowed.pth,,\r\n__pycache__/demo.cpython-312.pyc,,\r\ndemo-1.0.dist-info/METADATA,,\r\ndemo-1.0.dist-info/RECORD,,\r\n",
        )
        regular(
            "lib/python3.12/site-packages/torch-2.11.0.dist-info/METADATA",
            b"Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0\n",
        )
        regular(
            "lib/python3.12/site-packages/torch-2.11.0.dist-info/RECORD",
            b"torch.py,,\r\ntorch-2.11.0.dist-info/METADATA,,\r\ntorch-2.11.0.dist-info/RECORD,,\r\n",
        )
        if unsafe == "editable":
            regular(
                "lib/python3.12/site-packages/__editable__.rogue.pth",
                b"import __editable___rogue\n",
            )
        if unsafe == "local-origin":
            regular(
                "lib/python3.12/site-packages/rogue-1.dist-info/direct_url.json",
                b'{"url":"file:///tmp/rogue"}',
            )
        if unsafe == "pth":
            regular(
                "lib/python3.12/site-packages/rogue.pth",
                b'import sys; sys.path.insert(0,"/workspace")\n',
            )
        if unsafe == "workspace":
            regular(
                "lib/python3.12/site-packages/ambient.txt",
                b"/workspace/mutable-input\n",
            )
        if unsafe == "hardlink":
            link = tarfile.TarInfo(f"{prefix}/bin/hard")
            link.type = tarfile.LNKTYPE
            link.linkname = "python"
            archive.addfile(link)
        if unsafe == "traversal":
            regular("../../escape", b"escape")
        ignored = tarfile.TarInfo("workspace/uv/python")
        ignored.type = tarfile.DIRTYPE
        ignored.mode = 0o755
        archive.addfile(ignored)
        if unsafe == "outside":
            outside = tarfile.TarInfo("workspace/unclaimed")
            outside.type = tarfile.DIRTYPE
            outside.mode = 0o755
            archive.addfile(outside)
    return stream.getvalue()


def _lock_document(seed_bytes: bytes, base_bytes: bytes) -> dict[str, object]:
    compressed = b"synthetic compressed seed identity"
    return {
        "base_image": {
            "python": {"path": "/usr/bin/python3.12", "sha256": _sha(base_bytes)},
            "reference": "example.invalid/runtime@sha256:" + "1" * 64,
        },
        "distributions": [
            {"name": "demo", "version": "1.0"},
            {"name": "torch", "version": "2.11.0"},
        ],
        "provenance_tier": "content-addressed-binary-seed",
        "python_version": "3.12.3",
        "runtime_root": "/opt/job-scheduler/debate-runtime/v1",
        "schema": "runpod-safety.debate-runtime-binary-seed-lock/v2",
        "seed": {
            "format": "tar+zstd",
            "path": "seed.tar.zst",
            "prefix": "workspace/envs/verl-b200",
            "repository": "owner/runtime",
            "revision": "2" * 40,
            "sha256": _sha(compressed),
            "size": len(compressed),
            "uncompressed_path": "seed.tar",
            "uncompressed_sha256": _sha(seed_bytes),
            "uncompressed_size": len(seed_bytes),
        },
        "transformation": {
            "allowed_pth": [
                {
                    "path": "lib/python3.12/site-packages/allowed.pth",
                    "sha256": _sha(PTH),
                }
            ],
            "allowed_workspace_files": [],
            "ignored_archive_members": ["workspace/uv/python"],
            "remove_bytecode": True,
            "removed_links": [{"path": "bin/python3.12", "target": "python"}],
            "removed_paths": ["remove/editable-marker"],
            "rewrite_prefix_paths": ["bin/activate"],
            "source_prefix": "/workspace/envs/verl-b200",
            "source_pyvenv_sha256": _sha(PYVENV),
            "source_shebang": "#!/workspace/envs/verl-b200/bin/python\n",
            "target_pyvenv": TARGET_PYVENV,
            "target_prefix": "/opt/job-scheduler/debate-runtime/v1/python",
            "target_shebang": (
                "#!/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12\n"
            ),
        },
    }


def _patch_fixture_constants(monkeypatch: pytest.MonkeyPatch, module, document) -> None:
    seed = document["seed"]
    base = document["base_image"]
    monkeypatch.setattr(module.contract, "B200_BASE_IMAGE_REFERENCE", base["reference"])
    monkeypatch.setattr(module.contract, "B200_BASE_PYTHON_PATH", base["python"]["path"])
    monkeypatch.setattr(module.contract, "B200_BASE_PYTHON_SHA256", base["python"]["sha256"])
    monkeypatch.setattr(module.contract, "B200_SEED_REPOSITORY", seed["repository"])
    monkeypatch.setattr(module.contract, "B200_SEED_REVISION", seed["revision"])
    monkeypatch.setattr(module.contract, "B200_SEED_PATH", seed["path"])
    monkeypatch.setattr(module.contract, "B200_SEED_SIZE", seed["size"])
    monkeypatch.setattr(module.contract, "B200_SEED_SHA256", seed["sha256"])
    monkeypatch.setattr(
        module.contract, "B200_SEED_UNCOMPRESSED_PATH", seed["uncompressed_path"]
    )
    monkeypatch.setattr(
        module.contract, "B200_SEED_UNCOMPRESSED_SIZE", seed["uncompressed_size"]
    )
    monkeypatch.setattr(
        module.contract,
        "B200_SEED_UNCOMPRESSED_SHA256",
        seed["uncompressed_sha256"],
    )
    monkeypatch.setattr(module.contract, "B200_SEED_PREFIX", seed["prefix"])
    monkeypatch.setattr(module.contract, "B200_ALLOWED_UNCLAIMED_SITE_FILES", ())


def _write_receipt(generator, runtime: Path, document, files, directories) -> bytes:
    lock_payload = _canonical(document)
    receipt = {
        "base_python_sha256": document["base_image"]["python"]["sha256"],
        "dependency_lock_sha256": _sha(lock_payload),
        "directories": directories,
        "files": files,
        "schema": generator.SEED_TRANSFORMATION_SCHEMA,
        "seed_sha256": document["seed"]["sha256"],
        "uncompressed_seed_sha256": document["seed"]["uncompressed_sha256"],
    }
    receipt_path = runtime / generator.SEED_TRANSFORMATION_NAME
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o444)
    os.utime(receipt_path, ns=(0, 0))
    return lock_payload


def _build_spec(generator) -> dict[str, object]:
    return {
        "cuda": {
            "compiled_extensions": [{"module": "demo"}],
            "compute_capability": [10, 0],
            "device_count": 2,
            "device_name": "NVIDIA B200",
            "minimum_host_driver_version": 580,
            "nvlink_link_label": "NV18",
            "nvlink_pairs": [[0, 1]],
            "torch_cuda_version": "13.0",
            "torch_distribution": "torch",
            "torch_distribution_version": "2.11.0",
            "torch_version": "2.11.0+cu130",
        },
        "python": {
            "path": generator.RUNTIME_PYTHON,
            "version": "3.12.3",
        },
        "required_imports": [
            {"distribution": "demo", "module": "demo", "version": "1.0"}
        ],
        "runtime_root": generator.RUNTIME_ROOT,
        "schema": generator.BINARY_SEED_SPEC_SCHEMA,
        "site_packages_path": (
            generator.RUNTIME_ROOT + "/python/lib/python3.12/site-packages"
        ),
    }


@pytest.mark.parametrize("field", ["revision", "path", "sha256"])
def test_binary_seed_lock_refuses_revision_path_or_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    module = _load(GENERATOR, "seed_lock_generator")
    seed_bytes = _tar_bytes()
    base_bytes = b"base-python"
    document = _lock_document(seed_bytes, base_bytes)
    _patch_fixture_constants(monkeypatch, module, document)
    archive = tmp_path / "seed.tar.zst"
    archive.write_bytes(b"synthetic compressed seed identity")
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    module.validate_lock(lock, seed_path=archive)

    changed = json.loads(_canonical(document))
    if field == "revision":
        changed["seed"][field] = "3" * 40
    elif field == "path":
        changed["seed"][field] = "other.tar"
    else:
        changed["seed"][field] = "sha256:" + "3" * 64
    lock.write_bytes(_canonical(changed))
    with pytest.raises(module.Refusal, match="source identity|path differs|bytes differ"):
        module.validate_lock(lock, seed_path=archive)


@pytest.mark.parametrize(
    ("unsafe", "message"),
    [
        ("traversal", "path|prefix"),
        ("outside", "escapes the fixed prefix"),
        ("symlink", "unsafe link"),
        ("editable", "editable hook|PTH"),
        ("local-origin", "local direct origin"),
        ("pth", "PTH"),
        ("workspace", "unreviewed /workspace"),
        ("hardlink", "hard link"),
        ("rewrite", "rewrite is not deterministic"),
    ],
)
def test_binary_seed_transform_refuses_unsafe_members(
    tmp_path: Path, unsafe: str, message: str
) -> None:
    preparer = _load(PREPARER, "unsafe_seed_preparer")
    seed_bytes = _tar_bytes(unsafe=unsafe)
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())

    with base.open("rb") as base_source, pytest.raises(preparer.Refusal, match=message):
        preparer.extract_seed_stream(
            io.BytesIO(seed_bytes),
            destination=tmp_path / "runtime",
            lock_document=document,
            base_python=base_source,
        )


def test_binary_seed_receipt_rejects_extra_final_file(tmp_path: Path) -> None:
    generator = _load(GENERATOR, "receipt_generator")
    preparer = _load(PREPARER, "receipt_preparer")
    seed_bytes = _tar_bytes()
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    staging = tmp_path / "staging"
    staging.mkdir()
    runtime = staging / generator.RUNTIME_ROOT.removeprefix("/")
    runtime.parent.mkdir(parents=True)
    with base.open("rb") as base_source:
        records, directories = preparer.extract_seed_stream(
            io.BytesIO(seed_bytes),
            destination=runtime,
            lock_document=document,
            base_python=base_source,
        )
    lock_payload = _canonical(document)
    receipt = {
        "base_python_sha256": document["base_image"]["python"]["sha256"],
        "dependency_lock_sha256": _sha(lock_payload),
        "directories": directories,
        "files": records,
        "schema": generator.SEED_TRANSFORMATION_SCHEMA,
        "seed_sha256": document["seed"]["sha256"],
        "uncompressed_seed_sha256": document["seed"]["uncompressed_sha256"],
    }
    receipt_path = runtime / generator.SEED_TRANSFORMATION_NAME
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o444)
    os.utime(receipt_path, ns=(0, 0))
    generator.validate_seed_transformation(
        staging, lock_payload=lock_payload, lock_document=document
    )

    extra = runtime / "unexpected"
    extra.write_bytes(b"ambient")
    extra.chmod(0o444)
    with pytest.raises(generator.Refusal, match="unsafe file|differs from its transformation receipt"):
        generator.validate_seed_transformation(
            staging, lock_payload=lock_payload, lock_document=document
        )


def test_decoder_free_raw_tar_preparation_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _load(PREPARER, "raw_tar_preparer")
    seed_bytes = _tar_bytes()
    archive = tmp_path / "seed.tar"
    archive.write_bytes(seed_bytes)
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    document["base_image"]["python"]["path"] = str(base)
    _patch_fixture_constants(monkeypatch, preparer.metadata, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    staging = tmp_path / "staging"
    staging.mkdir()

    receipt = preparer.prepare(
        lock_path=lock,
        archive_path=archive,
        base_python=base,
        staging_root=staging,
        archive_format="tar",
    )

    runtime = staging / preparer.RUNTIME_ROOT.removeprefix("/")
    assert receipt == runtime / "seed-transformation.json"
    assert (runtime / "python/bin/python3.12").read_bytes() == b"base-python"
    assert (runtime / "python/bin/tool").read_bytes().startswith(
        b"#!/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12\n"
    )
    assert (runtime / "python/pyvenv.cfg").read_text("ascii") == TARGET_PYVENV


@pytest.mark.parametrize("raw", [False, True])
def test_binary_seed_input_symlink_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bool
) -> None:
    generator = _load(GENERATOR, f"redirected_seed_{raw}")
    seed_bytes = _tar_bytes()
    document = _lock_document(seed_bytes, b"base-python")
    _patch_fixture_constants(monkeypatch, generator, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    if raw:
        target = tmp_path / "raw-target"
        target.write_bytes(seed_bytes)
        selected = tmp_path / "seed.tar"
        selected.symlink_to(target)
        kwargs = {"uncompressed_seed_path": selected}
    else:
        target = tmp_path / "compressed-target"
        target.write_bytes(b"synthetic compressed seed identity")
        selected = tmp_path / "seed.tar.zst"
        selected.symlink_to(target)
        kwargs = {"seed_path": selected}
    with pytest.raises(generator.Refusal, match="redirected|unsafe"):
        generator.validate_lock(lock, **kwargs)


def test_binary_seed_raw_bytes_are_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load(GENERATOR, "raw_seed_rehash")
    seed_bytes = _tar_bytes()
    document = _lock_document(seed_bytes, b"base-python")
    _patch_fixture_constants(monkeypatch, generator, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    mutated = bytearray(seed_bytes)
    mutated[len(mutated) // 2] ^= 1
    archive = tmp_path / "seed.tar"
    archive.write_bytes(mutated)
    with pytest.raises(generator.Refusal, match="bytes differ"):
        generator.validate_lock(lock, uncompressed_seed_path=archive)


def test_binary_seed_staging_ancestor_symlink_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _load(PREPARER, "ancestor_seed_preparer")
    seed_bytes = _tar_bytes()
    archive = tmp_path / "seed.tar"
    archive.write_bytes(seed_bytes)
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    document["base_image"]["python"]["path"] = str(base)
    _patch_fixture_constants(monkeypatch, preparer.metadata, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    staging = tmp_path / "staging"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging / "opt").symlink_to(outside, target_is_directory=True)

    with pytest.raises(preparer.Refusal, match="ancestor is unsafe"):
        preparer.prepare(
            lock_path=lock,
            archive_path=archive,
            base_python=base,
            staging_root=staging,
            archive_format="tar",
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("drift", ["mode", "mtime", "directory"])
def test_binary_seed_receipt_refuses_tree_metadata_drift(
    tmp_path: Path, drift: str
) -> None:
    generator = _load(GENERATOR, f"tree_drift_{drift}")
    preparer = _load(PREPARER, f"tree_drift_preparer_{drift}")
    seed_bytes = _tar_bytes()
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    staging = tmp_path / "staging"
    staging.mkdir()
    runtime = staging / generator.RUNTIME_ROOT.removeprefix("/")
    runtime.parent.mkdir(parents=True)
    with base.open("rb") as base_source:
        files, directories = preparer.extract_seed_stream(
            io.BytesIO(seed_bytes),
            destination=runtime,
            lock_document=document,
            base_python=base_source,
        )
    lock_payload = _write_receipt(generator, runtime, document, files, directories)
    target = runtime / "python/lib/python3.12/site-packages/demo.py"
    if drift == "mode":
        target.chmod(0o555)
    elif drift == "mtime":
        os.utime(target, ns=(1, 1))
    else:
        injected = runtime / "python/empty"
        injected.parent.chmod(0o755)
        injected.mkdir(mode=0o777)
        injected.parent.chmod(0o555)
        os.utime(injected, ns=(0, 0))
    with pytest.raises(generator.Refusal, match="unsafe|differs"):
        generator.validate_seed_transformation(
            staging, lock_payload=lock_payload, lock_document=document
        )


def test_binary_seed_generate_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _load(PREPARER, "end_to_end_preparer")
    generator = _load(GENERATOR, "end_to_end_generator")
    seed_bytes = _tar_bytes()
    archive = tmp_path / "seed.tar"
    archive.write_bytes(seed_bytes)
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    document["base_image"]["python"]["path"] = str(base)
    _patch_fixture_constants(monkeypatch, preparer.metadata, document)
    _patch_fixture_constants(monkeypatch, generator, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    spec = tmp_path / "build-spec.json"
    spec.write_bytes(_canonical(_build_spec(generator)))
    staging = tmp_path / "staging"
    staging.mkdir()

    receipt_path = preparer.prepare(
        lock_path=lock,
        archive_path=archive,
        base_python=base,
        staging_root=staging,
        archive_format="tar",
    )
    receipt = json.loads(receipt_path.read_bytes())
    manifest = generator.generate(
        staging,
        lock,
        spec,
        uncompressed_seed_path=archive,
    )
    runtime = staging / generator.RUNTIME_ROOT.removeprefix("/")
    inventory = json.loads((runtime / "installed-distributions.json").read_bytes())
    assert manifest["schema"] == generator.BINARY_SEED_MANIFEST_SCHEMA
    assert manifest["seed_transformation_sha256"] == _sha(receipt_path.read_bytes())
    assert inventory["schema"] == generator.BINARY_SEED_INVENTORY_SCHEMA
    assert inventory["runtime_files"] == receipt["files"]
    assert inventory["runtime_directories"] == receipt["directories"]
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o555
    assert runtime.stat().st_mtime_ns == 0
    assert not list(runtime.rglob("*.pyc"))
    assert not list(runtime.rglob("__pycache__"))


def test_binary_seed_transformation_is_repeatable(tmp_path: Path) -> None:
    preparer = _load(PREPARER, "repeatable_seed_preparer")
    seed_bytes = _tar_bytes()
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    results = []
    for name in ("first", "second"):
        destination = tmp_path / name
        with base.open("rb") as base_source:
            files, directories = preparer.extract_seed_stream(
                io.BytesIO(seed_bytes),
                destination=destination,
                lock_document=document,
                base_python=base_source,
            )
        results.append((files, directories))
    assert results[0] == results[1]
    assert all(item["mtime_ns"] == 0 for item in results[0][0])
    assert all(item["mtime_ns"] == 0 for item in results[0][1])


def test_binary_seed_generator_rechecks_final_base_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparer = _load(PREPARER, "python_boundary_preparer")
    generator = _load(GENERATOR, "python_boundary_generator")
    seed_bytes = _tar_bytes()
    archive = tmp_path / "seed.tar"
    archive.write_bytes(seed_bytes)
    base = tmp_path / "base-python"
    base.write_bytes(b"base-python")
    document = _lock_document(seed_bytes, base.read_bytes())
    document["base_image"]["python"]["path"] = str(base)
    _patch_fixture_constants(monkeypatch, preparer.metadata, document)
    _patch_fixture_constants(monkeypatch, generator, document)
    lock = tmp_path / "dependency.lock"
    lock.write_bytes(_canonical(document))
    spec = tmp_path / "build-spec.json"
    spec.write_bytes(_canonical(_build_spec(generator)))
    staging = tmp_path / "staging"
    staging.mkdir()
    receipt_path = preparer.prepare(
        lock_path=lock,
        archive_path=archive,
        base_python=base,
        staging_root=staging,
        archive_format="tar",
    )
    runtime = staging / generator.RUNTIME_ROOT.removeprefix("/")
    python = runtime / "python/bin/python3.12"
    python.chmod(0o755)
    python.write_bytes(b"different-python")
    python.chmod(0o555)
    os.utime(python, ns=(0, 0))
    receipt = json.loads(receipt_path.read_bytes())
    for record in receipt["files"]:
        if record["path"] == "python/bin/python3.12":
            record["sha256"] = _sha(b"different-python")
            record["size"] = len(b"different-python")
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o444)
    os.utime(receipt_path, ns=(0, 0))

    with pytest.raises(generator.Refusal, match="pinned base image"):
        generator.generate(
            staging,
            lock,
            spec,
            uncompressed_seed_path=archive,
        )


def test_checked_binary_seed_contract_facts_are_exact() -> None:
    module = _load(GENERATOR, "checked_seed_generator")
    lock = json.loads((ROOT / "runtime_image" / "dependency.lock").read_bytes())
    assert lock["base_image"] == {
        "python": {
            "path": module.contract.B200_BASE_PYTHON_PATH,
            "sha256": module.contract.B200_BASE_PYTHON_SHA256,
        },
        "reference": module.contract.B200_BASE_IMAGE_REFERENCE,
    }
    assert lock["seed"] == {
        "format": "tar+zstd",
        "path": module.contract.B200_SEED_PATH,
        "prefix": module.contract.B200_SEED_PREFIX,
        "repository": module.contract.B200_SEED_REPOSITORY,
        "revision": module.contract.B200_SEED_REVISION,
        "sha256": module.contract.B200_SEED_SHA256,
        "size": module.contract.B200_SEED_SIZE,
        "uncompressed_path": module.contract.B200_SEED_UNCOMPRESSED_PATH,
        "uncompressed_sha256": module.contract.B200_SEED_UNCOMPRESSED_SHA256,
        "uncompressed_size": module.contract.B200_SEED_UNCOMPRESSED_SIZE,
    }
    assert len(lock["distributions"]) == 279
    assert lock["transformation"]["ignored_archive_members"] == [
        "workspace/uv/python"
    ]
    assert len(lock["transformation"]["allowed_workspace_files"]) == 25
    assert all(
        item["path"].startswith("python/")
        for item in lock["transformation"]["allowed_workspace_files"]
    )
    versions = {item["name"]: item["version"] for item in lock["distributions"]}
    assert "debate" not in versions
    assert {
        name: versions[name]
        for name in (
            "nvidia-cutlass-dsl",
            "nvidia-nccl-cu13",
            "ray",
            "torch",
            "transformers",
            "verl",
            "vllm",
        )
    } == {
        "nvidia-cutlass-dsl": "4.5.2",
        "nvidia-nccl-cu13": "2.28.9",
        "ray": "2.56.1",
        "torch": "2.11.0",
        "transformers": "5.14.1",
        "verl": "0.9.0.dev0",
        "vllm": "0.24.0",
    }
    spec = json.loads((ROOT / "runtime_image" / "build-spec.json").read_bytes())
    assert spec["cuda"]["torch_distribution_version"] == "2.11.0"
    assert spec["cuda"]["torch_version"] == "2.11.0+cu130"
    assert spec["cuda"]["compiled_extensions"] == [{"module": "vllm._C"}]


def test_binary_seed_dockerfile_has_non_overridable_offline_base() -> None:
    source = SEED_DOCKERFILE.read_text("ascii")
    reference = (
        "ghcr.io/ethanelasky/verl-megatron-ssh@sha256:"
        "162f12d0ba6fbceaf56294d783551b2e83aefcc6e76b9138737acd358ee0baa8"
    )
    assert "ARG " not in source
    assert "FROM $" not in source
    assert source.startswith(
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e\n"
    )
    assert source.count("FROM --platform=linux/amd64 " + reference) == 2
    assert source.count("RUN --network=none") == 1
