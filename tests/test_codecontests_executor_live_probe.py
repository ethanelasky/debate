"""Validation tests for the retained hostile executor probe."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import probe_codecontests_executor_live as probe


def _mock_root_owned_fstat(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fstat = os.fstat

    def root_owned_fstat(descriptor: int) -> os.stat_result:
        values = list(real_fstat(descriptor))
        values[4] = 0
        values[5] = 0
        return os.stat_result(values)

    monkeypatch.setattr(probe.os, "fstat", root_owned_fstat)


def test_host_path_probe_binds_a_narrow_root_owned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "host-only-marker"
    marker.write_bytes(b"host-only\n")
    marker.chmod(0o444)
    _mock_root_owned_fstat(monkeypatch)

    identity = probe._measure_host_path_probe(marker)
    with marker.open("rb") as handle:
        metadata = probe.os.fstat(handle.fileno())

    assert identity == {
        "path": str(marker),
        "size": len(b"host-only\n"),
        "mode": 0o444,
        "uid": 0,
        "gid": 0,
        "device": marker.stat().st_dev,
        "inode": marker.stat().st_ino,
        "nlink": metadata.st_nlink,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": hashlib.sha256(b"host-only\n").hexdigest(),
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("relative/marker", "absolute"),
        ("/", "broad"),
        ("/definitely/absent/palaestra-host-marker", "does not exist"),
    ],
)
def test_host_path_probe_rejects_unsafe_or_missing_paths(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        probe._measure_host_path_probe(value)


def test_host_path_probe_rejects_noncanonical_and_symlink_paths(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker"
    marker.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(marker)

    with pytest.raises(ValueError, match="canonical"):
        probe._measure_host_path_probe(f"{tmp_path}/./marker")
    with pytest.raises(ValueError, match="symlink"):
        probe._measure_host_path_probe(link)


def test_host_path_probe_rejects_unsafe_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker"
    marker.write_bytes(b"x")
    marker.chmod(0o666)
    _mock_root_owned_fstat(monkeypatch)

    with pytest.raises(ValueError, match="group/world writable"):
        probe._measure_host_path_probe(marker)


def test_host_path_probe_rejects_nonroot_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "marker"
    marker.write_bytes(b"x")
    real_fstat = os.fstat

    def nonroot_fstat(descriptor: int) -> os.stat_result:
        values = list(real_fstat(descriptor))
        values[4] = 501
        values[5] = 20
        return os.stat_result(values)

    monkeypatch.setattr(probe.os, "fstat", nonroot_fstat)
    with pytest.raises(ValueError, match="root:root"):
        probe._measure_host_path_probe(marker)


def test_host_path_probe_must_be_absent_from_candidate_rootfs(
    tmp_path: Path,
) -> None:
    identity = {"path": "/opt/palaestra-probe/host-only-marker"}
    measured = probe._attest_host_path_absent_from_rootfs(identity, tmp_path)
    assert measured["candidate_rootfs_path_absent"] is True

    candidate_path = tmp_path / "opt/palaestra-probe/host-only-marker"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_bytes(b"collision")
    with pytest.raises(RuntimeError, match="candidate rootfs"):
        probe._attest_host_path_absent_from_rootfs(identity, tmp_path)


def test_host_path_candidate_source_has_no_personal_machine_literal() -> None:
    selected = "/opt/palaestra-probe/host-only-marker"
    source = probe._credential_host_path_isolation_source(selected)

    compile(source, "<host-path-probe>", "exec")
    assert selected in source
    assert "/Users/" not in source
    assert ".guest" not in source


def test_namespace_filesystem_negative_probe_is_valid_candidate_source() -> None:
    compile(
        probe.NAMESPACE_FILESYSTEM_ESCAPE_DENIAL,
        "<namespace-filesystem-negative-probe>",
        "exec",
    )
    assert "clone_newuser" in probe.NAMESPACE_FILESYSTEM_ESCAPE_DENIAL
    assert '"mount"' in probe.NAMESPACE_FILESYSTEM_ESCAPE_DENIAL
    assert '"unshare"' in probe.NAMESPACE_FILESYSTEM_ESCAPE_DENIAL


@pytest.mark.parametrize(
    "source",
    [
        probe.CANDIDATE_CREDENTIAL_CAPABILITY_STATUS,
        probe.FORCED_CLONE_OUTPUT_CROSSING,
        *probe.SIGKILL_ORIGIN_PROBES.values(),
    ],
)
def test_credential_and_teardown_probes_are_valid_candidate_sources(
    source: str,
) -> None:
    compile(source, "<credential-teardown-probe>", "exec")


def test_every_unexplained_sigkill_origin_is_an_ambiguous_unknown() -> None:
    observed_sources: list[bytes] = []

    class Supervisor:
        def execute(self, request_payload: dict[str, Any]) -> dict[str, Any]:
            observed_sources.append(
                base64.b64decode(
                    request_payload["task"]["code_b64"],
                    validate=True,
                )
            )
            return {
                "outcome": "unknown",
                "category": "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
                "resource_event": None,
                "returncode": -probe.signal.SIGKILL,
                "stdout_b64": "",
                "stderr_b64": "",
            }

    matrix = probe.Matrix(Supervisor())
    probe._run_sigkill_origin_probes(matrix)

    assert observed_sources == [
        source.encode() for source in probe.SIGKILL_ORIGIN_PROBES.values()
    ]
    assert list(matrix.cases) == list(probe.SIGKILL_ORIGIN_PROBES)
    assert {
        (case["outcome"], case["category"], case["returncode"])
        for case in matrix.cases.values()
    } == {
        (
            "unknown",
            "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE",
            -probe.signal.SIGKILL,
        )
    }


def test_forced_interleaving_launchers_compile_and_bind_exact_hooks() -> None:
    production = (
        probe.REPOSITORY_ROOT / "codecontests_executor" / "sandbox_launcher.py"
    ).read_bytes()
    production_digest = hashlib.sha256(production).hexdigest()
    instrumented_digests = set()

    for name in (
        "exit_stop_cont_esrch",
        "clone_output_crossing",
    ):
        instrumented = probe._instrument_launcher_source(production, name)
        compile(
            instrumented,
            f"<forced-launcher-{name}>",
            "exec",
        )
        assert name.encode("ascii") in instrumented
        assert hashlib.sha256(instrumented).hexdigest() != production_digest
        instrumented_digests.add(hashlib.sha256(instrumented).hexdigest())

    assert len(instrumented_digests) == 2


@pytest.mark.parametrize(
    "source",
    [
        probe.FORK_DENIAL,
        probe.RAW_PROCESS_CLONE_DENIAL,
        probe.CLONE3_PROBE,
        probe.THREAD_LEGAL,
        probe.THREAD_LIMIT,
        probe.THREAD_LIMIT_UNCAUGHT,
    ],
)
def test_real_creation_syscall_probes_are_valid_candidate_sources(
    source: str,
) -> None:
    compile(source, "<creation-syscall-probe>", "exec")


def test_missing_cap_kill_regression_removes_only_cap_kill_and_restores_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = probe.supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES
    observed: dict[str, object] = {}

    class Supervisor:
        def execute(self, request_payload: dict[str, Any]) -> dict[str, Any]:
            observed["capabilities"] = (
                probe.supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES
            )
            task = request_payload["task"]
            observed["code"] = base64.b64decode(
                task["code_b64"],
                validate=True,
            )
            return {
                "outcome": "unknown",
                "category": "LAUNCH_ATTESTATION_MISSING",
                "returncode": 125,
                "stdout_b64": "",
                "stderr_b64": "",
            }

    summary = probe._run_missing_cap_kill_regression(Supervisor())

    assert "CAP_KILL" not in observed["capabilities"]  # type: ignore[operator]
    assert observed["code"] == b"import os; os.write(1,b'x'*2097152)"
    assert probe.supervisor_module.TRUSTED_MONITOR_BOOTSTRAP_CAPABILITIES == original
    assert summary["category"] == "LAUNCH_ATTESTATION_MISSING"


def test_output_target_requires_a_safe_root_owned_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "executor-result.json"
    _mock_root_owned_fstat(monkeypatch)

    measured, parent_identity = probe._measure_output_target(output)

    assert measured == output
    assert parent_identity == {
        "device": tmp_path.stat().st_dev,
        "inode": tmp_path.stat().st_ino,
        "mode": tmp_path.stat().st_mode & 0o7777,
        "uid": 0,
        "gid": 0,
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("relative/result.json", "absolute"),
        ("/tmp/result.json", "broad"),
        ("/definitely/absent/parent/result.json", "does not exist"),
        ("/opt/palaestra-probe/result.txt", "suffix"),
    ],
)
def test_output_target_rejects_unsafe_paths(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        probe._measure_output_target(value)


def test_output_target_rejects_existing_leaf_and_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_root_owned_fstat(monkeypatch)
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="already exists"):
        probe._measure_output_target(existing)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        probe._measure_output_target(linked_parent / "result.json")


def test_output_target_rejects_writable_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o777)
    _mock_root_owned_fstat(monkeypatch)
    with pytest.raises(ValueError, match="group/world writable"):
        probe._measure_output_target(tmp_path / "result.json")


def test_output_publication_is_mode_600_and_never_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_root_owned_fstat(monkeypatch)
    output, parent_identity = probe._measure_output_target(tmp_path / "result.json")
    probe._publish_output(output, parent_identity, b'{"ok":true}\n')
    assert output.read_bytes() == b'{"ok":true}\n'
    assert output.stat().st_mode & 0o777 == 0o600

    raced, raced_parent_identity = probe._measure_output_target(tmp_path / "raced.json")
    raced.write_bytes(b"attacker-selected")
    with pytest.raises(FileExistsError):
        probe._publish_output(
            raced,
            raced_parent_identity,
            b"must-not-replace",
        )
    assert raced.read_bytes() == b"attacker-selected"
    assert not list(tmp_path.glob(".raced.json.*.tmp"))


def test_host_probe_retains_and_revalidates_every_descriptor_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "host-only-marker"
    marker.write_bytes(b"host-only\n")
    marker.chmod(0o444)
    _mock_root_owned_fstat(monkeypatch)

    _identity, chain = probe._retain_host_path_probe(marker)
    try:
        assert len(chain.descriptors) == len(marker.parts)
        chain.revalidate()
        displaced = tmp_path / "displaced"
        marker.rename(displaced)
        marker.write_bytes(b"replacement")
        marker.chmod(0o444)
        with pytest.raises(RuntimeError, match="path binding"):
            chain.revalidate()
    finally:
        chain.close()


def test_terminal_host_probe_remeasurement_rejects_same_size_content_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "host-only-marker"
    marker.write_bytes(b"host-only\n")
    marker.chmod(0o444)
    _mock_root_owned_fstat(monkeypatch)
    expected, marker_chain = probe._retain_host_path_probe(marker)
    output, parent_identity, output_chain = probe._retain_output_target(
        tmp_path / "terminal.json"
    )

    def mutate_and_validate() -> None:
        marker.chmod(0o644)
        marker.write_bytes(b"HOST-ONLY\n")
        marker.chmod(0o444)
        probe._assert_retained_host_path_probe(expected, marker_chain)

    try:
        with pytest.raises(RuntimeError, match="host-path probe changed"):
            probe._publish_output(
                output,
                parent_identity,
                b'{"must":"be-removed"}\n',
                descriptor_chain=output_chain,
                post_publish_validator=mutate_and_validate,
            )
        assert not output.exists()
        assert not list(tmp_path.glob(".terminal.json.*.tmp"))
    finally:
        marker_chain.close()
        output_chain.close()


def test_output_publication_uses_retained_parent_chain_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_root_owned_fstat(monkeypatch)
    output, parent_identity, chain = probe._retain_output_target(
        tmp_path / "retained.json"
    )
    try:
        assert len(chain.descriptors) == len(output.parent.parts)
        tmp_path.chmod(0o500)
        with pytest.raises(RuntimeError, match="descriptor"):
            probe._publish_output(
                output,
                parent_identity,
                b'{"must":"fail"}\n',
                descriptor_chain=chain,
            )
        assert not output.exists()
    finally:
        chain.close()


def test_host_path_probe_cli_argument_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.sys,
        "argv",
        ["probe_codecontests_executor_live.py", "--output", "/tmp/result"],
    )
    with pytest.raises(SystemExit) as raised:
        probe.main()
    assert raised.value.code == 2


def test_host_path_probe_cli_rejects_a_broad_path_before_host_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.sys,
        "argv",
        [
            "probe_codecontests_executor_live.py",
            "--output",
            "/tmp/result",
            "--host-path-probe",
            "/",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        probe.main()
    assert raised.value.code == 2
