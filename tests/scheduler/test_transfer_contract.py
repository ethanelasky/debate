"""Executable contract for immutable scheduler inputs and outputs.

One transfer vocabulary covers both directions, so a later reconciliation pass
can re-run collection for an already-finished attempt without re-running it:

``freeze_inputs(snapshot_dir, *, source_root, inputs)``
    Freeze the current source tree and explicitly named regular-file inputs.

``stage_inputs(snapshot, destination, *, copy_operation=None)``
    Copy a frozen snapshot to an attempt and verify destination checksums.  The
    result exposes ``source_root`` and the named ``inputs`` paths.

``collect_attempt(collection_root, *, job_id, attempt_id, command_exit_code,
outputs, logs, copy_operation=None)``
    Retrieve and verify one attempt's declared outputs and stdout/stderr logs.
    The result exposes ``outputs``, ``logs``, ``job_id``, ``attempt_id``, and
    ``complete``.  ``command_exit_code`` alone decides how strict the collection
    is: on exit 0 every declared output plus both logs must arrive and verify;
    on a nonzero exit both logs are still mandatory and declared outputs become
    optional -- but a declared output that does exist is an artifact like any
    other, collected and verified exactly as on exit 0 and exposed in
    ``outputs``.

``copy_operation`` is the single injection seam these tests use -- it accepts
``(source: Path, destination: Path)`` and returns a process-style integer
return code.  Transport, archive layout, and manifest serialization stay
unprescribed here.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import shutil
from pathlib import Path

import pytest


CHECKOUT_BYTES = {
    "train.py": b"tracked-at-submit\n",
    "dirty.py": b"dirty-working-tree-bytes\n",
    "untracked.txt": b"untracked-at-submit\n",
    "configs/run.yaml": b"steps: 12\n",
}
INPUT_BYTES = {"dataset": b'{"id": 1}\n', "override": b"batch: 4\n"}
CREDENTIALS = (
    ".env",
    ".git/config",
    ".venv/lib/token",
    "venv/lib/token",
    ".ssh/id_rsa",
    ".aws/credentials",
)
UNDECLARED = b"undeclared-must-not-be-copied\n"
UNSAFE_PATH = "credential|secret|symlink|regular|escape|unsafe"


@pytest.fixture
def transfer():
    """Import in a fixture so this red contract still collects cleanly."""
    try:
        return importlib.import_module("tools.scheduler.transfer")
    except ModuleNotFoundError as exc:
        if exc.name not in {"tools.scheduler", "tools.scheduler.transfer"}:
            raise
        pytest.fail(
            "tools.scheduler.transfer is not implemented; this is the red "
            "scheduler transfer contract",
            pytrace=False,
        )


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_bytes(path: Path, expected: bytes) -> None:
    """Independent oracle: never use scheduler checksum code in assertions."""
    assert not path.is_symlink()
    assert path.read_bytes() == expected
    assert _sha256(path) == hashlib.sha256(expected).hexdigest()


def _checkout(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A working checkout plus the out-of-tree files one job declares."""
    repo = tmp_path / "repo"
    for relative, payload in CHECKOUT_BYTES.items():
        _write(repo / relative, payload)
    # Checkout metadata, local environments, and credentials must not cross the
    # boundary.  Other checkout content is intentionally frozen as-is.
    for relative in CREDENTIALS:
        _write(repo / relative, b"secret\n")

    dataset = _write(tmp_path / "dataset" / "rows.jsonl", INPUT_BYTES["dataset"])
    _write(tmp_path / "dataset" / ".env", b"DATA_TOKEN=secret\n")
    override = _write(tmp_path / "override.yaml", INPUT_BYTES["override"])
    _write(tmp_path / "undeclared.txt", UNDECLARED)
    return repo, {"dataset": dataset, "override": override}


def _tamper(snapshot_dir: Path, marker: bytes) -> None:
    """Damage frozen bytes without assuming any snapshot layout."""
    frozen = sorted(
        path
        for path in snapshot_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert frozen, "freeze_inputs must persist the bytes it froze"
    for path in frozen:
        payload = path.read_bytes()
        if marker in payload:
            path.write_bytes(payload.replace(marker, b"x" * len(marker)))
            return
    frozen[0].write_bytes(b"tampered")


def _corrupting_copy(source: Path, destination: Path) -> int:
    """Copy successfully according to return code, then damage destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        victim = next(path for path in destination.rglob("*") if path.is_file())
    else:
        shutil.copy2(source, destination)
        victim = destination

    payload = victim.read_bytes()
    victim.write_bytes(payload[: max(0, len(payload) // 2)])
    return 0


def _attempt_files(
    root: Path,
    *,
    output: bytes = b"model-v1",
    stdout: bytes = b"training\n",
    stderr: bytes = b"",
) -> dict[str, dict[str, Path]]:
    return {
        "outputs": {"model": _write(root / "remote" / "model.bin", output)},
        "logs": {
            "stdout": _write(root / "remote" / "stdout.log", stdout),
            "stderr": _write(root / "remote" / "stderr.log", stderr),
        },
    }


def _expect_error(error_type, action, *, context, pattern=None):
    """Run ``action`` and assert it raises ``error_type``, tagging failures
    with ``context`` so a loop-body assertion points at the offending case."""
    try:
        action()
    except error_type as exc:
        if pattern is not None:
            assert re.search(pattern, str(exc)), f"{context}: unexpected message {exc!r}"
    else:
        suffix = f" matching {pattern!r}" if pattern else ""
        pytest.fail(f"{context}: expected {error_type.__name__}{suffix}, none raised")


def _collect(
    transfer,
    collection_root: Path,
    files: dict[str, dict[str, Path]],
    *,
    exit_code: int = 0,
    job_id: str = "train-a",
    attempt_id: str = "attempt-001",
    copy_operation=None,
):
    """One re-runnable collection call: no consumption, no compute rerun."""
    return transfer.collect_attempt(
        collection_root,
        job_id=job_id,
        attempt_id=attempt_id,
        command_exit_code=exit_code,
        outputs=dict(files["outputs"]),
        logs=dict(files["logs"]),
        copy_operation=copy_operation,
    )


def test_snapshot_freezes_the_checkout_and_declared_inputs_only(transfer, tmp_path):
    repo, inputs = _checkout(tmp_path)

    snapshot = transfer.freeze_inputs(
        tmp_path / "snapshots" / "job-1", source_root=repo, inputs=inputs
    )
    staged = transfer.stage_inputs(snapshot, tmp_path / "stage")

    for relative, expected in CHECKOUT_BYTES.items():
        _assert_bytes(staged.source_root / relative, expected)
    assert set(staged.inputs) == set(INPUT_BYTES)
    for name, expected in INPUT_BYTES.items():
        _assert_bytes(staged.inputs[name], expected)

    for relative in CREDENTIALS:
        assert not (staged.source_root / relative).exists()
    staged_files = [path for path in (tmp_path / "stage").rglob("*") if path.is_file()]
    assert staged_files
    for path in staged_files:
        payload = path.read_bytes()
        assert UNDECLARED not in payload
        assert b"DATA_TOKEN" not in payload


def test_every_retry_stages_the_same_frozen_bytes(transfer, tmp_path):
    repo, inputs = _checkout(tmp_path)
    snapshot = transfer.freeze_inputs(
        tmp_path / "snapshot", source_root=repo, inputs=inputs
    )
    first = transfer.stage_inputs(snapshot, tmp_path / "stage-attempt-1")

    # Snapshot identity is byte identity, not a live reference to input paths.
    _write(repo / "train.py", b"changed-after-submit\n")
    _write(repo / "later-untracked.txt", b"created too late\n")
    shutil.rmtree(tmp_path / "dataset")
    inputs["override"].unlink()
    second = transfer.stage_inputs(snapshot, tmp_path / "stage-attempt-2")

    for staged in (first, second):
        for relative, expected in CHECKOUT_BYTES.items():
            _assert_bytes(staged.source_root / relative, expected)
        for name, expected in INPUT_BYTES.items():
            _assert_bytes(staged.inputs[name], expected)
        assert not (staged.source_root / "later-untracked.txt").exists()


def test_staging_verifies_bytes_rather_than_copy_return_codes(transfer, tmp_path):
    for damage in ("tampered-snapshot", "lying-copy-operation"):
        case_root = tmp_path / damage
        repo, inputs = _checkout(case_root)
        snapshot_dir = case_root / "snapshot"
        snapshot = transfer.freeze_inputs(
            snapshot_dir, source_root=repo, inputs=inputs
        )

        copy_operation = None
        if damage == "tampered-snapshot":
            _tamper(snapshot_dir, CHECKOUT_BYTES["train.py"])
        else:
            copy_operation = _corrupting_copy

        _expect_error(
            transfer.TransferError,
            lambda snapshot=snapshot, copy_operation=copy_operation: transfer.stage_inputs(
                snapshot, case_root / "stage", copy_operation=copy_operation
            ),
            context=damage,
            pattern="checksum|integrity|tamper",
        )


def test_exit_zero_needs_every_declared_output_and_both_logs_verified(
    transfer, tmp_path
):
    for damage in (None, "missing-output", "missing-stderr", "lying-copy-operation"):
        case_root = tmp_path / (damage or "clean")
        files = _attempt_files(case_root)
        if damage == "missing-output":
            files["outputs"]["model"].unlink()
        elif damage == "missing-stderr":
            files["logs"]["stderr"].unlink()

        if damage is not None:
            _expect_error(
                transfer.TransferError,
                lambda files=files, damage=damage: _collect(
                    transfer,
                    case_root / "collections",
                    files,
                    copy_operation=(
                        _corrupting_copy if damage == "lying-copy-operation" else None
                    ),
                ),
                context=damage,
            )
            continue

        collected = _collect(transfer, case_root / "collections", files)
        assert collected.complete is True, "clean"
        assert (collected.job_id, collected.attempt_id) == ("train-a", "attempt-001")
        _assert_bytes(collected.outputs["model"], b"model-v1")
        _assert_bytes(collected.logs["stdout"], b"training\n")
        _assert_bytes(collected.logs["stderr"], b"")


def test_nonzero_exit_verifies_both_logs_and_any_output_that_exists(transfer, tmp_path):
    # A failed command still owes both logs; its declared output is optional.
    # Whatever output bytes do exist are artifacts like any other: collected,
    # verified, and frozen under the attempt identity before a retry runs.
    for output_written in (True, False):
        case_root = tmp_path / f"output-written-{output_written}"
        files = _attempt_files(
            case_root,
            output=b"partial-checkpoint",
            stdout=b"started then failed\n",
            stderr=b"out of memory\n",
        )
        if not output_written:
            files["outputs"]["model"].unlink()

        collected = _collect(transfer, case_root / "collections", files, exit_code=23)

        assert collected.complete is True, output_written
        _assert_bytes(collected.logs["stdout"], b"started then failed\n")
        _assert_bytes(collected.logs["stderr"], b"out of memory\n")
        if not output_written:
            assert dict(collected.outputs) == {}, output_written
            continue

        _assert_bytes(collected.outputs["model"], b"partial-checkpoint")
        repeated = _collect(transfer, case_root / "collections", files, exit_code=23)
        assert repeated.outputs["model"] == collected.outputs["model"]
        _assert_bytes(repeated.outputs["model"], b"partial-checkpoint")
        # Reusing the failed attempt's identity for different bytes fails closed.
        _write(files["outputs"]["model"], b"replacement-bytes")
        with pytest.raises(transfer.TransferError):
            _collect(transfer, case_root / "collections", files, exit_code=23)
        _assert_bytes(collected.outputs["model"], b"partial-checkpoint")

    # Damage fails closed on a failed attempt too.  Empty logs survive the lying
    # copy byte-identical, so only the output arrives corrupted there.
    for damage in ("missing-stdout", "missing-stderr", "corrupted-output"):
        case_root = tmp_path / damage
        files = _attempt_files(
            case_root, output=b"partial-checkpoint", stdout=b"", stderr=b""
        )
        if damage.startswith("missing-"):
            files["logs"][damage.removeprefix("missing-")].unlink()

        _expect_error(
            transfer.TransferError,
            lambda files=files, damage=damage, case_root=case_root: _collect(
                transfer,
                case_root / "collections",
                files,
                exit_code=23,
                copy_operation=(
                    _corrupting_copy if damage == "corrupted-output" else None
                ),
            ),
            context=damage,
        )


def test_collection_may_retry_until_verified_then_becomes_immutable(
    transfer, tmp_path
):
    collection_root = tmp_path / "collections"
    files = _attempt_files(
        tmp_path,
        output=b"model-arrived-late",
        stdout=b"unique-stdout\n",
        stderr=b"unique-stderr\n",
    )
    first_copy: dict[str, object] = {}

    def copy_one_then_fail(source: Path, destination: Path) -> int:
        if first_copy:
            return 23
        assert source.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        first_copy["payload"] = source.read_bytes()
        return 0

    with pytest.raises(transfer.TransferError):
        _collect(
            transfer, collection_root, files, copy_operation=copy_one_then_fail
        )

    # A later transfer failure must not roll back an artifact which already
    # copied and verified inside this attempt's immutable namespace.
    preserved = [
        path
        for path in collection_root.rglob("*")
        if path.is_file() and path.read_bytes() == first_copy["payload"]
    ]
    assert len(preserved) == 1
    preserved_path = preserved[0]
    stat = preserved_path.stat()
    preserved_identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def resume_without_overwrite(source: Path, destination: Path) -> int:
        assert source.read_bytes() != first_copy["payload"], (
            "retry tried to copy an already-verified artifact again"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return 0

    collected = _collect(
        transfer,
        collection_root,
        files,
        copy_operation=resume_without_overwrite,
    )
    _assert_bytes(collected.outputs["model"], b"model-arrived-late")
    _assert_bytes(preserved_path, first_copy["payload"])
    stat = preserved_path.stat()
    assert (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == (
        preserved_identity
    )

    repeated = _collect(transfer, collection_root, files)
    assert repeated.outputs["model"] == collected.outputs["model"]
    _assert_bytes(repeated.outputs["model"], b"model-arrived-late")

    sibling = _collect(
        transfer,
        collection_root,
        _attempt_files(tmp_path / "sibling", output=b"attempt-two"),
        attempt_id="attempt-002",
    )
    assert sibling.outputs["model"] != collected.outputs["model"]
    _assert_bytes(sibling.outputs["model"], b"attempt-two")

    # Reusing a verified identity for different bytes must fail closed.
    _write(files["outputs"]["model"], b"replacement-bytes")
    with pytest.raises(transfer.TransferError):
        _collect(transfer, collection_root, files)
    _assert_bytes(collected.outputs["model"], b"model-arrived-late")


def test_untrusted_paths_never_cross_the_transfer_boundary(transfer, tmp_path):
    for unsafe_kind in (
        "explicit-credential",
        "input-symlink",
        "source-root-symlink",
        "output-symlink",
        "stdout-symlink",
        "stderr-symlink",
        "collection-root-symlink",
    ):
        case_root = tmp_path / unsafe_kind
        outside = _write(case_root / "outside" / "secret.txt", b"must not be copied\n")

        if unsafe_kind.startswith(("explicit", "input", "source-root")):
            repo = case_root / "repo"
            _write(repo / "train.py", CHECKOUT_BYTES["train.py"])
            inputs: dict[str, Path] = {}
            if unsafe_kind == "explicit-credential":
                inputs["credentials"] = _write(case_root / ".env", b"TOKEN=secret\n")
            elif unsafe_kind == "input-symlink":
                linked_input = case_root / "linked-input"
                linked_input.symlink_to(outside)
                inputs["dataset"] = linked_input
            else:
                (repo / "linked-outside").symlink_to(
                    outside.parent, target_is_directory=True
                )

            _expect_error(
                transfer.TransferError,
                lambda repo=repo, inputs=inputs, case_root=case_root: transfer.freeze_inputs(
                    case_root / "snapshot", source_root=repo, inputs=inputs
                ),
                context=unsafe_kind,
                pattern=UNSAFE_PATH,
            )
            continue

        files = _attempt_files(case_root)
        collection_root = case_root / "collections"
        redirected = None
        if unsafe_kind == "collection-root-symlink":
            redirected = case_root / "redirected-collection"
            redirected.mkdir()
            collection_root.symlink_to(redirected, target_is_directory=True)
        else:
            stream = unsafe_kind.removesuffix("-symlink")
            source = (
                files["outputs"]["model"]
                if stream == "output"
                else files["logs"][stream]
            )
            source.unlink()
            source.symlink_to(outside)

        _expect_error(
            transfer.TransferError,
            lambda collection_root=collection_root, files=files, unsafe_kind=unsafe_kind: _collect(
                transfer, collection_root, files, attempt_id=f"attempt-{unsafe_kind}"
            ),
            context=unsafe_kind,
            pattern=UNSAFE_PATH,
        )
        if redirected is not None:
            assert list(redirected.iterdir()) == [], unsafe_kind
