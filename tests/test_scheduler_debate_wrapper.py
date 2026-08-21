from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL_WRAPPER = ROOT / "scripts" / "scheduler_debate_run.sh"
SUPERVISOR = ROOT / "scripts" / "scheduler_debate_supervisor.py"
POD_RUN = ROOT / "scripts" / "pod_run.sh"
RUNTIME_PYTHON = Path(
    "/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12"
)


def _supervisor_module():
    spec = importlib.util.spec_from_file_location("scheduler_debate_supervisor", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scheduler_wrapper_has_one_fresh_base_launch_and_no_passthrough() -> None:
    shell = SHELL_WRAPPER.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    module = _supervisor_module()

    assert module.LAUNCH_ARGV == (
        "bash",
        "scripts/pod_run.sh",
        "debate",
        "mathl5_qwen35_pc_debate_verl",
    )
    assert 'if [ "$#" -ne 0 ]' in shell
    for forbidden in ("--load", "--start-step", "--wandb-resume"):
        assert forbidden not in shell
        assert forbidden not in supervisor
    assert "POD_IDLE_STOP=0" in shell
    assert "DEBATE_SCHEDULER_MODE=1" in shell
    assert "DEBATE_CHECKPOINT_DESTINATION_FILE" in shell
    assert "DEBATE_WORKLOAD_OUTPUT_ROOT" not in shell
    assert "DEBATE_ATTEMPT_ID:?" not in shell
    assert "DEBATE_CGROUP_PARENT" not in shell
    assert "DEBATE_CGROUP_PARENT" not in supervisor
    assert 'CONTAINMENT_PROOF_FD = 9' in supervisor
    assert 'CHECKPOINT_DESTINATION_FD = 10' in supervisor
    assert 'CHECKPOINT_DESTINATION_PATH = "/proc/self/fd/10"' in supervisor
    assert 'CONTAINMENT_SCHEMA = "runpod-remote.containment-proof/v1"' in supervisor
    assert '"cgroup.procs").write_text' not in supervisor
    assert '"cgroup.kill").write_text' not in supervisor
    assert "SUPERVISOR_PY=/usr/bin/python3" in shell
    assert 'exec "$SUPERVISOR_PY" -I -S' in shell
    assert f"export PY={RUNTIME_PYTHON}" in shell
    assert f"export PYBIN={RUNTIME_PYTHON}" in shell
    assert module.RUNTIME_PYTHON == RUNTIME_PYTHON
    assert "/workspace/envs/" not in shell
    assert "/workspace/envs/" not in supervisor
    assert '"S3_ENV_FILE": "/proc/self/debate-scheduler-no-credential-file"' in supervisor
    for independent_control in ("runpodctl", "pod_idle_stop.sh", "ssh "):
        assert independent_control not in shell
        assert independent_control not in supervisor


def _containment_document() -> dict[str, object]:
    return {
        "schema": "runpod-remote.containment-proof/v1",
        "protocol_version": 1,
        "worker_ref": "pod-123",
        "attempt_id": "attempt-123",
        "attempt_identity_sha256": "a" * 64,
        "snapshot_sha256": "b" * 64,
        "namespace": "launch-123",
        "deadline_epoch": 1_800_000_000,
        "wrapper_release": "remote-wrapper-v1",
        "wrapper_sha256": "c" * 64,
        "workload_uid": 10001,
        "workload_gid": 10001,
        "cgroup_relative_path": "/job-scheduler/attempt-123",
        "artifact_root": "/workspace/artifacts",
        "attempt_root": "/workspace/artifacts/launch-123",
        "evidence_root": "/workspace/job-scheduler/attempt-123",
        "workload_output_root": (
            "/workspace/artifacts/launch-123/scheduler-output"
        ),
        "checkpoint_working_root": (
            "/workspace/checkpoints/mathl5_qwen35_pc_debate_verl/launch-123"
        ),
    }


def test_containment_proof_parser_accepts_only_installed_wrapper_contract() -> None:
    module = _supervisor_module()
    document = _containment_document()

    proof = module._parse_containment_document(document)

    assert proof.protocol_version == 1
    assert proof.attempt_id == "attempt-123"
    assert proof.namespace == "launch-123"
    assert proof.workload_uid == proof.workload_gid == 10001
    assert proof.workload_output_root.endswith("/scheduler-output")

    with_unknown = dict(document, extra="not-installed-wrapper-contract")
    with pytest.raises(module.Refusal, match="unknown or missing"):
        module._parse_containment_document(with_unknown)

    wrong_identity = dict(document, workload_uid=0)
    with pytest.raises(module.Refusal, match="dedicated uid/gid 10001"):
        module._parse_containment_document(wrong_identity)


def test_containment_proof_requires_canonical_json_and_fixed_sealed_memfd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _supervisor_module()
    document = _containment_document()
    canonical = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    reads = iter((canonical, b""))

    class FakeStat:
        st_mode = 0o100400
        st_uid = 0
        st_gid = 0
        st_nlink = 0

    required_seals = 0
    for name in ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE"):
        monkeypatch.setattr(module.fcntl, name, 1 << required_seals, raising=False)
        required_seals += 1
    all_seals = sum(1 << index for index in range(required_seals))
    monkeypatch.setattr(module.fcntl, "F_GET_SEALS", 1034, raising=False)
    monkeypatch.setattr(module.os, "fstat", lambda fd: FakeStat())
    monkeypatch.setattr(module.os, "lseek", lambda *args: 0)
    monkeypatch.setattr(module.os, "read", lambda *args: next(reads))
    monkeypatch.setattr(module.os, "close", lambda fd: None)
    monkeypatch.setattr(module.fcntl, "fcntl", lambda *args: all_seals)

    assert module._read_containment_proof().attempt_id == "attempt-123"

    noncanonical = json.dumps(document).encode("ascii")
    reads = iter((noncanonical, b""))
    with pytest.raises(module.Refusal, match="canonical ASCII"):
        module._read_containment_proof()


def test_load_settings_matches_prefixed_scheduler_digests_to_raw_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _supervisor_module()
    namespace = "launch-123"
    artifact_root = tmp_path / "artifacts"
    attempt_root = artifact_root / namespace
    output_root = attempt_root / "scheduler-output"
    output_root.mkdir(parents=True, mode=0o700)
    artifact_root.chmod(0o700)
    attempt_root.chmod(0o700)
    output_root.chmod(0o700)
    cgroup = tmp_path / "proved-cgroup"
    cgroup.mkdir()

    proof_document = _containment_document()
    proof_document.update(
        {
            "artifact_root": str(artifact_root),
            "attempt_root": str(attempt_root),
            "workload_output_root": str(output_root),
        }
    )
    proof = module._parse_containment_document(proof_document)
    monkeypatch.setattr(module, "_read_containment_proof", lambda: proof)
    monkeypatch.setattr(module, "_validate_runtime_python", lambda: None)
    monkeypatch.setattr(
        module, "_validate_unprivileged_containment", lambda contained: cgroup
    )
    monkeypatch.setattr(module, "_below_workspace", lambda path, field: None)
    monkeypatch.setattr(
        module,
        "_read_destination",
        lambda path: (
            b'{"directory":"/workspace/checkpoints","kind":"local"}\n',
            {"kind": "local", "directory": "/workspace/checkpoints"},
        ),
    )
    canonical_existing = module._canonical_existing_directory

    def canonical_for_test(raw: str, *, field: str) -> Path:
        if field == "checkpoint working parent":
            return artifact_root
        return canonical_existing(raw, field=field)

    monkeypatch.setattr(module, "_canonical_existing_directory", canonical_for_test)
    original_lexists = module.os.path.lexists
    monkeypatch.setattr(
        module.os.path,
        "lexists",
        lambda path: False
        if str(path).startswith("/workspace/checkpoints/")
        else original_lexists(path),
    )
    environ = {
        "DEBATE_LAUNCH_NAMESPACE": namespace,
        "DEBATE_ARTIFACT_ROOT": str(artifact_root),
        "DEBATE_CHECKPOINT_DESTINATION_FILE": "/proc/self/fd/10",
        "DEBATE_ATTEMPT_IDENTITY_SHA256": "sha256:" + "a" * 64,
        "DEBATE_SNAPSHOT_SHA256": "sha256:" + "b" * 64,
        "DEBATE_DEADLINE_EPOCH": "1800000000",
        "RUNPOD_POD_ID": "pod-123",
    }

    settings = module.load_settings(str(ROOT), environ)

    assert settings.attempt_identity == "a" * 64
    assert settings.snapshot_sha256 == "b" * 64
    assert settings.containment.attempt_identity_sha256 == "a" * 64
    assert settings.containment.snapshot_sha256 == "b" * 64

    raw_environment = dict(environ)
    raw_environment["DEBATE_ATTEMPT_IDENTITY_SHA256"] = "a" * 64
    with pytest.raises(module.Refusal, match="canonical sha256"):
        module.load_settings(str(ROOT), raw_environment)


def test_scheduler_runtime_python_is_fixed_and_refuses_mutable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _supervisor_module()
    runtime_python = tmp_path / "python3.12"
    runtime_python.write_bytes(b"runtime-python")
    runtime_python.chmod(0o555)
    monkeypatch.setattr(module, "RUNTIME_PYTHON", runtime_python)
    real_lstat = module.os.lstat

    def root_owned_lstat(path: object):
        info = real_lstat(path)
        if Path(path) != runtime_python:
            return info
        values = list(info)
        values[4] = 0
        values[5] = 0
        return os.stat_result(values)

    monkeypatch.setattr(module.os, "lstat", root_owned_lstat)
    module._validate_runtime_python()

    runtime_python.chmod(0o755)
    with pytest.raises(module.Refusal, match="root-owned mode-0555"):
        module._validate_runtime_python()


def test_scheduler_pod_run_forbids_runtime_override_and_skips_editable_install(
    tmp_path: Path,
) -> None:
    text = POD_RUN.read_text(encoding="utf-8")
    start = text.index("SCHEDULER_RUNTIME_PYTHON=")
    end = text.index("# Toolkit preference", start)
    probe = tmp_path / "runtime-probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        "DEBATE_SCHEDULER_MODE=1\n"
        "PY=/workspace/envs/ambient/bin/python\n"
        + text[start:end],
        encoding="utf-8",
    )

    result = subprocess.run(["bash", str(probe)], text=True, capture_output=True)

    assert result.returncode == 4
    assert "forbids a Python override" in result.stderr
    install = '"$PY" -m pip install -q -e . --no-deps'
    install_position = text.index(install)
    scheduler_skip = text.rfind(
        'if [ "${DEBATE_SCHEDULER_MODE:-0}" != 1 ]; then',
        0,
        install_position,
    )
    assert scheduler_skip != -1
    assert text.index("\nfi", install_position) > install_position


def test_scheduler_mode_does_not_read_repo_dotenv(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    scripts = staged / "scripts"
    scripts.mkdir(parents=True)
    copied = scripts / "pod_run.sh"
    copied.write_text(POD_RUN.read_text(encoding="utf-8"), encoding="utf-8")
    (staged / ".env").write_text("SCHEDULER_DOTENV_POISON=loaded\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()

    text = copied.read_text(encoding="utf-8")
    prefix = text[: text.index('MODE="${1:?usage: pod_run.sh')]
    probe = tmp_path / "probe.sh"
    probe.write_text(
        prefix + 'printf "%s\\n" "${SCHEDULER_DOTENV_POISON:-not-loaded}"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("SCHEDULER_DOTENV_POISON", None)
    env.update(
        {
            "DEBATE_SCHEDULER_MODE": "1",
            "POD_IDLE_STOP": "0",
            "POD_RUN_STATE_DIR": str(state.resolve()),
        }
    )
    result = subprocess.run(
        ["bash", str(probe)], env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "not-loaded"


def test_evidence_manifest_hashes_regular_files_and_refuses_links(tmp_path: Path) -> None:
    module = _supervisor_module()
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"beta")

    records = module._manifest_tree(root)
    assert [item["path"] for item in records] == ["a", "nested/b"]
    assert records[0]["sha256"] == __import__("hashlib").sha256(b"alpha").hexdigest()

    (root / "unsafe").symlink_to(root / "a")
    with pytest.raises(module.Refusal, match="linked evidence"):
        module._manifest_tree(root)


def test_log_sanitizer_removes_terminal_codes_and_common_secret_shapes() -> None:
    module = _supervisor_module()
    cleaned = module._sanitize(
        b"\x1b[31mred\x1b[0m api_key=abcdef Authorization: Bearer token123\r\n"
        b"https://name:password@example.invalid/path\x00\n"
    )

    assert b"\x1b" not in cleaned and b"\x00" not in cleaned
    assert b"abcdef" not in cleaned and b"token123" not in cleaned
    assert b"name:password" not in cleaned
    assert cleaned.count(b"[REDACTED]") == 3


def test_log_pump_redacts_secret_shapes_split_across_read_chunks(
    tmp_path: Path,
) -> None:
    module = _supervisor_module()
    payload = b"x" * (64 * 1024 - 4) + b"api_key=abcdef\n"
    destination = tmp_path / "stdout"
    inherited = tmp_path / "inherited"
    inherited_fd = os.open(inherited, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        module._pump(io.BytesIO(payload), destination, inherited_fd)
    finally:
        os.close(inherited_fd)

    for path in (destination, inherited):
        content = path.read_bytes()
        assert b"abcdef" not in content
        assert b"api_key=[REDACTED]" in content


def test_inner_cgroup_census_excludes_only_its_own_supervisor_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _supervisor_module()
    cgroup = tmp_path / "attempt-cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").write_text("77\n88\n", encoding="ascii")
    monkeypatch.setattr(module.os, "getpid", lambda: 77)

    view = module.UpstreamCgroup(cgroup)

    assert view.pids() == (77, 88)
    assert view.workload_pids() == (88,)
    assert view.populated() is True


def test_checkpoint_sync_environment_uses_sealed_destination_without_lifecycle_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _supervisor_module()
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-cross")
    settings = SimpleNamespace(
        output_root=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        namespace="launch-123",
        destination_bytes=b'{"directory":"/workspace/checkpoints","kind":"local"}\n',
    )
    settings.output_root.mkdir()

    environ = module._sync_environment(settings, once=True)

    assert "RUNPOD_API_KEY" not in environ
    assert environ["DEBATE_CHECKPOINT_DESTINATION_FILE"] == "/proc/self/fd/10"
    assert environ["CKPT_DESTINATION_JSON"] == settings.destination_bytes.decode(
        "ascii"
    )
    assert environ["CKPT_SYNC_ONCE"] == "1"


def test_emergency_workload_terminal_is_no_replace_and_omits_error_message(
    tmp_path: Path,
) -> None:
    module = _supervisor_module()
    settings = SimpleNamespace(output_root=tmp_path)

    module._publish_emergency_terminal(
        settings, RuntimeError("api_key=must-not-be-serialized")
    )
    terminal = tmp_path / "workload-terminal.json"
    first = terminal.read_bytes()
    document = json.loads(first)

    assert document["succeeded"] is False
    assert document["error_type"] == "RuntimeError"
    assert b"must-not-be-serialized" not in first

    module._publish_emergency_terminal(settings, ValueError("replacement"))
    assert terminal.read_bytes() == first


def test_checkpoint_destination_file_is_private_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _supervisor_module()
    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    payload = (
        json.dumps(
            {"kind": "local", "directory": str(destination_dir)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    reads = iter((payload, b""))

    class FakeStat:
        st_mode = 0o100400
        st_uid = 0
        st_gid = 0
        st_nlink = 0
        st_dev = 1
        st_ino = 2
        st_size = len(payload)

    seal_values = (1, 2, 4, 8)
    for name, value in zip(
        ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE"),
        seal_values,
    ):
        monkeypatch.setattr(module.fcntl, name, value, raising=False)
    monkeypatch.setattr(module.fcntl, "F_GET_SEALS", 1034, raising=False)
    monkeypatch.setattr(module.fcntl, "fcntl", lambda *args: sum(seal_values))
    monkeypatch.setattr(module.os, "fstat", lambda fd: FakeStat())
    monkeypatch.setattr(module.os, "lseek", lambda *args: 0)
    monkeypatch.setattr(module.os, "read", lambda *args: next(reads))
    monkeypatch.setattr(module, "_below_workspace", lambda path, field: None)

    actual_payload, document = module._read_destination("/proc/self/fd/10")
    assert actual_payload == payload
    assert document == {"kind": "local", "directory": str(destination_dir)}

    reads = iter((b'{"kind":"local","kind":"bucket"}\n', b""))
    with pytest.raises(module.Refusal, match="duplicate JSON key"):
        module._read_destination("/proc/self/fd/10")

    with pytest.raises(module.Refusal, match="exact sealed FD 10"):
        module._read_destination("/workspace/destination.json")


def test_scheduler_shell_scripts_parse() -> None:
    for script in (POD_RUN, SHELL_WRAPPER):
        subprocess.run(["bash", "-n", script], check=True)
    compile(SUPERVISOR.read_text(encoding="utf-8"), str(SUPERVISOR), "exec")
