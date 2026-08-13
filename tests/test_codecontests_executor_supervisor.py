"""Host-supervisor probes using real pipes/process groups (without claiming gVisor)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

import pytest

from codecontests_executor.protocol import (
    STDOUT_CAP_BYTES,
    make_execute_request,
)
from codecontests_executor.supervisor import (
    SandboxExecutorConfig,
    SandboxSupervisor,
    SupervisorConfigurationError,
    _ProcessIdentity,
    _RequestCgroup,
)

RAW_LIMITS = {
    "time_limit": {"seconds": 1, "nanos": 0},
    "memory_limit_bytes": 4 * 1024**3,
}

_PROBE_PROGRAM = r"""
import base64
import json
import os
import struct
import time
mode = os.environ["PROBE_MODE"]
nonce = os.environ["PALAESTRA_EXECUTOR_LAUNCH_NONCE"]
header = os.read(0, 8)
remaining = struct.unpack("!Q", header)[0]
while remaining:
    remaining -= len(os.read(0, remaining))
os.write(2, f"PALAESTRA_EXECUTOR_READY:{nonce}:monitor_uid=0:monitor_gid=0\n".encode())
assert os.read(0, 1) == b"G"
candidate_returncode = 0
stdout_truncated = False
process_limit_hit = False
process_limit_syscall = None
tracer_killed_main = False
if mode == "large_ok":
    os.write(1, b"x" * 8192)
elif mode == "output_limit":
    # The trusted in-guest relay saturates at exactly the semantic cap and
    # records that boundary before the host sees any cap+1 byte.
    os.write(1, b"x" * (2 * 1024 * 1024))
    candidate_returncode = -9
    stdout_truncated = True
    tracer_killed_main = True
elif mode == "exit125":
    candidate_returncode = 125
elif mode == "self_sigkill":
    candidate_returncode = -9
elif mode == "tracer_sigkill":
    candidate_returncode = -9
    stdout_truncated = True
    tracer_killed_main = True
elif mode == "caught_process_limit":
    os.write(1, b"handled\n")
    process_limit_hit = True
    process_limit_syscall = 56
elif mode == "uncaught_process_limit":
    candidate_returncode = 1
    process_limit_hit = True
    process_limit_syscall = 56
elif mode == "monitor_sigkill":
    os.kill(os.getpid(), 9)
elif mode == "wall":
    time.sleep(1)
status = {
    "version": 1,
    "candidate_ready_attested": True,
    "returncode": candidate_returncode,
    "cpu_usage_us": 0,
    "cpu_limit_us": 2000000,
    "cpu_limit_hit": False,
    "process_peak": 2 if process_limit_hit else 1,
    "process_limit": 2,
    "process_rlimit_nproc": 1,
    "process_limit_hit": process_limit_hit,
    "process_limit_syscall": process_limit_syscall,
    "tracer_killed_main": tracer_killed_main,
    "stdout_truncated": stdout_truncated,
    "stderr_truncated": False,
    "file_space_limit_hit": False,
    "file_space_limit_source": None,
    "file_size_limit_bytes": 2 * 1024 * 1024,
    "writable_limit_bytes": 0,
    "file_limit_signal": None,
    "file_limit_errno": None,
    "file_size_observed_bytes": 0,
    "writable_available_bytes": 0,
}
if mode == "monitor_failure":
    status = {
        "version": 1,
        "candidate_ready_attested": False,
        "error": "RuntimeError",
        "site": "trace_measure",
        "source_line": 3021,
    }
encoded = base64.b64encode(json.dumps(
    status, sort_keys=True, separators=(",", ":")
).encode()).decode()
os.write(2, f"PALAESTRA_EXECUTOR_STATUS:{nonce}:{encoded}\n".encode())
"""

_PRE_READY_EXIT_PROGRAM = r"""
import os
os.read(0, 8)
raise SystemExit(17)
"""


class ProbeSupervisor(SandboxSupervisor):
    def __init__(self, config: SandboxExecutorConfig, mode: str):
        super().__init__(config)
        self.mode = mode
        self.request_mode: int | None = None

    def _build_command(  # type: ignore[no-untyped-def]
        self, state_dir, limits, **_kwargs
    ):
        del limits
        self.request_mode = stat.S_IMODE(os.stat(Path(state_dir).parent).st_mode)
        return [sys.executable, "-c", _PROBE_PROGRAM]

    def _sanitized_env(self, nonce: str) -> dict[str, str]:  # type: ignore[override]
        env = super()._sanitized_env(nonce)
        env["PROBE_MODE"] = self.mode
        return env


class PreReadyExitSupervisor(ProbeSupervisor):
    def _build_command(  # type: ignore[no-untyped-def]
        self, state_dir, limits, **_kwargs
    ):
        del state_dir, limits
        return [sys.executable, "-c", _PRE_READY_EXIT_PROGRAM]


def _request():
    return make_execute_request(
        code="print('candidate')",
        stdin="stdin",
        raw_limits=RAW_LIMITS,
        identity_digest_value="a" * 64,
        ttl_ns=10_000_000_000,
    )


def _probe(tmp_path: Path, mode: str) -> ProbeSupervisor:
    return ProbeSupervisor(
        SandboxExecutorConfig(
            request_root=str(tmp_path),
            drop_runsc_to_nobody=False,
            enforce_request_cgroup=False,
        ),
        mode,
    )


def test_correct_output_larger_than_4k_is_not_truncated(tmp_path):
    supervisor = _probe(tmp_path, "large_ok")
    result = supervisor.execute(_request())
    assert result["outcome"] == "executed"
    assert result["stdout_bytes"] == 8192
    assert result["stdout_truncated"] is False
    assert supervisor.request_mode == 0o700
    assert list(tmp_path.iterdir()) == []


def test_stdout_at_exact_2mib_is_candidate_output_limit_and_fast_teardown(
    tmp_path,
):
    supervisor = _probe(tmp_path, "output_limit")
    started = time.monotonic()
    result = supervisor.execute(_request())
    elapsed = time.monotonic() - started
    assert result["outcome"] == "candidate_failure"
    assert result["category"] == "OUTPUT_LIMIT"
    assert result["stdout_bytes"] == STDOUT_CAP_BYTES
    assert result["stdout_truncated"] is True
    assert elapsed < 5
    assert list(tmp_path.iterdir()) == []


def test_candidate_exit125_after_ready_marker_is_runtime_failure(tmp_path):
    supervisor = _probe(tmp_path, "exit125")
    result = supervisor.execute(_request())
    assert result["outcome"] == "candidate_failure"
    assert result["category"] == "RUNTIME_ERROR"
    assert result["returncode"] == 125


def test_attested_candidate_sigkill_without_tracer_evidence_remains_unknown(
    tmp_path,
):
    supervisor = _probe(tmp_path, "self_sigkill")
    result = supervisor.execute(_request())
    assert result["outcome"] == "unknown"
    assert result["category"] == "AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE"
    assert result["returncode"] == -signal.SIGKILL


def test_tracer_owned_candidate_sigkill_retains_its_output_limit_verdict(tmp_path):
    supervisor = _probe(tmp_path, "tracer_sigkill")
    result = supervisor.execute(_request())
    assert result["outcome"] == "candidate_failure"
    assert result["category"] == "OUTPUT_LIMIT"
    assert result["returncode"] == -signal.SIGKILL


def test_caught_second_thread_denial_can_still_execute_successfully(tmp_path):
    supervisor = _probe(tmp_path, "caught_process_limit")
    result = supervisor.execute(_request())
    assert result["outcome"] == "executed"
    assert base64.b64decode(result["stdout_b64"]) == b"handled\n"
    assert result["guest_process_limit_syscall"] == 56
    assert result["resource_event"] == "GUEST_PROCESS_LIMIT"
    assert result["resource_evidence_source"] == (
        "guest_monitor_ptrace_thread_eagain"
    )


def test_uncaught_second_thread_denial_is_process_limit_failure(tmp_path):
    supervisor = _probe(tmp_path, "uncaught_process_limit")
    result = supervisor.execute(_request())
    assert result["outcome"] == "candidate_failure"
    assert result["category"] == "PROCESS_LIMIT"
    assert result["returncode"] == 1


def test_sigkilled_monitor_without_terminal_attestation_remains_unknown(tmp_path):
    supervisor = _probe(tmp_path, "monitor_sigkill")
    result = supervisor.execute(_request())
    assert result["outcome"] == "unknown"
    assert result["category"] == "LAUNCH_ATTESTATION_MISSING"
    assert result["returncode"] == -signal.SIGKILL


def test_false_terminal_monitor_status_remains_launch_unknown(tmp_path):
    supervisor = _probe(tmp_path, "monitor_failure")

    result = supervisor.execute(_request())

    assert result["outcome"] == "unknown"
    assert result["category"] == "LAUNCH_ATTESTATION_MISSING"
    stderr = base64.b64decode(result["stderr_b64"])
    encoded_status = stderr.rsplit(b":", 1)[-1].strip()
    status = json.loads(base64.b64decode(encoded_status))
    assert status == {
        "version": 1,
        "candidate_ready_attested": False,
        "error": "RuntimeError",
        "site": "trace_measure",
        "source_line": 3021,
    }


def test_pre_ready_controller_exit_does_not_mask_launch_failure_as_drain(
    tmp_path: Path,
) -> None:
    supervisor = PreReadyExitSupervisor(
        SandboxExecutorConfig(
            request_root=str(tmp_path),
            drop_runsc_to_nobody=False,
            enforce_request_cgroup=False,
        ),
        "unused",
    )
    started = time.monotonic()

    result = supervisor.execute(_request())

    assert time.monotonic() - started < 2
    assert result["outcome"] == "unknown"
    assert result["category"] == "LAUNCH_ATTESTATION_MISSING"
    assert result["returncode"] == 17
    assert list(tmp_path.iterdir()) == []


def test_host_ready_boundary_wall_deadline_is_candidate_wall_limit(tmp_path):
    supervisor = _probe(tmp_path, "wall")
    request = make_execute_request(
        code="pass",
        stdin="",
        raw_limits={
            "time_limit": {"seconds": 0, "nanos": 10_000_000},
            "memory_limit_bytes": 4 * 1024**3,
        },
        identity_digest_value="a" * 64,
        ttl_ns=10_000_000_000,
    )
    result = supervisor.execute(request)
    assert result["outcome"] == "candidate_failure"
    assert result["category"] == "WALL_LIMIT"
    assert result["returncode"] is not None


def test_production_command_has_no_candidate_mount_or_shell_interpolation():
    supervisor = SandboxSupervisor()
    limits = _request()["task"]["limits"]
    command = supervisor._build_command("/private/request/runsc-state", limits)
    assert command[0] == "/opt/gvisor/20260721.0/runsc"
    assert "--rootless=true" not in command
    assert "--network=none" in command
    assert "run" in command
    assert any(argument.startswith("--bundle=") for argument in command)
    assert not any(argument.startswith("--volume=") for argument in command)
    assert "print('candidate')" not in command
    assert "stdin" not in command
    config = supervisor._oci_config(
        limits=limits,
        nonce="a" * 64,
        cgroups_path="/test/candidate",
    )
    assert "-c" in config["process"]["args"]
    assert str(limits["effective"]["address_space_bytes"]) in config["process"]["args"]
    assert str(limits["effective"]["cpu_seconds"]) in config["process"]["args"]
    assert config["root"]["readonly"] is True
    assert config["process"]["user"] == {
        "uid": 0,
        "gid": 0,
        "additionalGids": [],
    }
    assert config["process"]["capabilities"] == {
        "bounding": [
            "CAP_KILL",
            "CAP_SETGID",
            "CAP_SETPCAP",
            "CAP_SETUID",
            "CAP_SYS_PTRACE",
        ],
        "effective": [
            "CAP_KILL",
            "CAP_SETGID",
            "CAP_SETPCAP",
            "CAP_SETUID",
            "CAP_SYS_PTRACE",
        ],
        "inheritable": [],
        "permitted": [
            "CAP_KILL",
            "CAP_SETGID",
            "CAP_SETPCAP",
            "CAP_SETUID",
            "CAP_SYS_PTRACE",
        ],
        "ambient": [],
    }
    assert config["process"]["noNewPrivileges"] is True


def test_sanitized_runsc_environment_contains_no_host_secrets(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "never-forward")
    monkeypatch.setenv("CODECONTESTS_EXECUTOR_HMAC_KEY", "never-forward")
    env = SandboxSupervisor._sanitized_env("a" * 64)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "CODECONTESTS_EXECUTOR_HMAC_KEY" not in env
    assert env["PALAESTRA_EXECUTOR_LAUNCH_NONCE"] == "a" * 64


def test_runsc_log_requires_real_eof_and_counts_newlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runsc.log"
    record = b'{"level":"info","msg":"ok"}\n'
    monkeypatch.setattr(
        "codecontests_executor.supervisor._RUNSC_LOG_MAX_BYTES",
        len(record),
    )
    path.write_bytes(record)
    assert SandboxSupervisor._runsc_log_is_clean(str(path))

    path.write_bytes(record[:-1])
    assert not SandboxSupervisor._runsc_log_is_clean(str(path))

    # The byte cap includes the framing newline; this is one byte too large.
    path.write_bytes(record + b"\n")
    assert not SandboxSupervisor._runsc_log_is_clean(str(path))


def test_runsc_log_rejects_malformed_and_tail_warning_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runsc.log"
    path.write_bytes(b'{"level":"info"}\n{"level":')
    assert not SandboxSupervisor._runsc_log_is_clean(str(path))

    path.write_bytes(
        b'{"level":"info","msg":"prefix"}\n{"level":"warning","msg":"fatal tail"}\n'
    )
    assert not SandboxSupervisor._runsc_log_is_clean(str(path))


def test_full_rootfs_tree_measurement_detects_content_and_topology_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rootfs"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"first")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        before = SandboxSupervisor._measure_rootfs_tree(root_fd)
        payload.write_bytes(b"other")
        after_content = SandboxSupervisor._measure_rootfs_tree(root_fd)
        assert after_content != before

        payload.write_bytes(b"first")
        extra = root / "extra"
        extra.mkdir()
        after_topology = SandboxSupervisor._measure_rootfs_tree(root_fd)
        assert after_topology != before
    finally:
        os.close(root_fd)


def test_owned_descendant_inventory_binds_pid_starttime_and_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {
        10: _ProcessIdentity(10, 1, 10, 10, 100),
        11: _ProcessIdentity(11, 10, 10, 10, 101),
        12: _ProcessIdentity(12, 11, 12, 10, 102),
        99: _ProcessIdentity(99, 1, 99, 99, 999),
    }
    monkeypatch.setattr(
        SandboxSupervisor,
        "_snapshot_process_identities",
        classmethod(lambda cls: dict(identities)),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_container_process_ids",
        classmethod(lambda cls, measured, container_id: {11}),
    )
    measured = SandboxSupervisor._stable_owned_inventory(
        proc_pid=10,
        container_id="cc-test",
    )
    assert measured == {pid: identities[pid] for pid in (10, 11, 12)}


def test_process_identity_accepts_fresh_boot_starttime_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = [b"S", b"0", b"1", b"1", *([b"0"] * 15), b"0"]
    raw = b"1 (systemd) " + b" ".join(fields) + b"\n"
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.BytesIO(raw))

    identity = _ProcessIdentity(
        pid=1,
        parent_pid=0,
        process_group=1,
        session=1,
        starttime_ticks=0,
    )
    assert SandboxSupervisor._read_process_identity(1) == identity
    assert SandboxSupervisor._merge_owned_inventories(
        {identity.pid: identity},
        {identity.pid: identity},
    ) == {identity.pid: identity}
    with pytest.raises(
        SupervisorConfigurationError,
        match="PID/starttime identity was reused",
    ):
        SandboxSupervisor._merge_owned_inventories(
            {identity.pid: identity},
            {
                identity.pid: _ProcessIdentity(
                    pid=1,
                    parent_pid=0,
                    process_group=1,
                    session=1,
                    starttime_ticks=1,
                )
            },
        )


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux procfs")
def test_process_identity_snapshot_accepts_real_proc_pid_one() -> None:
    identity = SandboxSupervisor._read_process_identity(1)

    assert identity.pid == 1
    assert identity.starttime_ticks >= 0
    assert SandboxSupervisor._snapshot_process_identities()[1] == identity


def test_owned_inventory_rejects_pid_starttime_reuse_between_frozen_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        (
            {
                10: _ProcessIdentity(10, 1, 10, 10, 100),
                11: _ProcessIdentity(11, 10, 10, 10, 101),
            },
            {
                10: _ProcessIdentity(10, 1, 10, 10, 100),
                11: _ProcessIdentity(11, 10, 10, 10, 202),
            },
        )
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_snapshot_process_identities",
        classmethod(lambda cls: next(snapshots)),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_container_process_ids",
        classmethod(lambda cls, measured, container_id: {11}),
    )
    with pytest.raises(SupervisorConfigurationError, match="inventory changed"):
        SandboxSupervisor._stable_owned_inventory(
            proc_pid=10,
            container_id="cc-test",
        )


def test_owned_inventory_rejects_same_token_with_ancestry_session_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = {10: _ProcessIdentity(10, 1, 10, 10, 100)}
    drifted = {10: _ProcessIdentity(10, 2, 20, 30, 100)}
    monkeypatch.setattr(
        SandboxSupervisor,
        "_snapshot_process_identities",
        classmethod(lambda cls: dict(drifted)),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_container_process_ids",
        classmethod(lambda cls, measured, container_id: set()),
    )

    with pytest.raises(
        SupervisorConfigurationError,
        match="ancestry/session identity drifted",
    ):
        SandboxSupervisor._stable_owned_inventory(
            proc_pid=10,
            container_id="cc-test",
            initial_inventory=retained,
            root_required=False,
        )


def test_owned_inventory_union_retains_exited_process_tokens() -> None:
    gate = _ProcessIdentity(10, 1, 10, 10, 100)
    helper = _ProcessIdentity(11, 10, 10, 10, 101)

    merged = SandboxSupervisor._merge_owned_inventories(
        {gate.pid: gate},
        {helper.pid: helper},
    )

    assert merged == {gate.pid: gate, helper.pid: helper}
    assert SandboxSupervisor._merge_owned_inventories(merged, {}) == merged


def test_owned_inventory_union_rejects_reused_pid() -> None:
    retained = _ProcessIdentity(10, 1, 10, 10, 100)
    reused = _ProcessIdentity(10, 1, 10, 10, 200)

    with pytest.raises(
        SupervisorConfigurationError,
        match="PID/starttime identity was reused",
    ):
        SandboxSupervisor._merge_owned_inventories(
            {retained.pid: retained},
            {reused.pid: reused},
        )


def test_blocked_gate_identity_requires_two_matching_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = iter(
        (
            _ProcessIdentity(10, 1, 10, 10, 100),
            _ProcessIdentity(10, 1, 10, 10, 200),
        )
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_read_process_identity",
        staticmethod(lambda pid: next(reads)),
    )

    with pytest.raises(
        SupervisorConfigurationError,
        match="blocked-gate attestation",
    ):
        SandboxSupervisor._stable_process_identity(10)


def test_unrelated_large_cmdline_is_streamed_without_attestation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _ProcessIdentity(99, 1, 99, 99, 999)
    payload = b"x" * (2 * 1024 * 1024)
    monkeypatch.setattr(
        "builtins.open",
        lambda path, mode: io.BytesIO(payload),
    )

    assert (
        SandboxSupervisor._container_process_ids(
            {identity.pid: identity},
            "cc-target",
        )
        == set()
    )


def test_container_marker_can_span_streaming_cmdline_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _ProcessIdentity(99, 1, 99, 99, 999)
    marker = b"cc-target"
    payload = b"x" * (64 * 1024 - 3) + marker + b"\0tail"
    monkeypatch.setattr(
        "builtins.open",
        lambda path, mode: io.BytesIO(payload),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_read_process_identity",
        staticmethod(lambda pid: identity),
    )

    assert SandboxSupervisor._container_process_ids(
        {identity.pid: identity},
        marker.decode("ascii"),
    ) == {identity.pid}


def test_cleanup_recursively_kills_without_thawing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = tmp_path / "outer"
    controller = outer / "controller"
    candidate = outer / "candidate"
    controller.mkdir(parents=True)
    candidate.mkdir()
    request_cgroup = _RequestCgroup(
        outer=str(outer),
        controller=str(controller),
        candidate=str(candidate),
        oci_path="requests/test/candidate",
        memory_events_before={},
        pids_events_before={},
        outer_controllers=frozenset({"cpu", "memory", "pids"}),
    )
    calls: list[tuple[str, str, str | bool]] = []
    monkeypatch.setattr(
        SandboxSupervisor,
        "_write_cgroup_control",
        staticmethod(lambda path, value: calls.append(("write", path, value))),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_set_cgroup_frozen",
        classmethod(
            lambda cls, path, *, frozen: calls.append(("freeze", path, frozen))
        ),
    )
    monkeypatch.setattr(
        SandboxSupervisor,
        "_keyed_cgroup_values",
        classmethod(lambda cls, path: {"populated": 0}),
    )
    supervisor = object.__new__(SandboxSupervisor)

    supervisor._cleanup_request_cgroup(request_cgroup)

    assert calls == [("write", str(outer / "cgroup.kill"), "1")]
    assert not outer.exists()


def test_attested_terminal_boundary_freezes_and_attests_before_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = tmp_path / "outer"
    controller = outer / "controller"
    candidate = outer / "candidate"
    controller.mkdir(parents=True)
    candidate.mkdir()
    request_cgroup = _RequestCgroup(
        outer=str(outer),
        controller=str(controller),
        candidate=str(candidate),
        oci_path="requests/test/candidate",
        memory_events_before={},
        pids_events_before={},
        outer_controllers=frozenset({"cpu", "memory", "pids"}),
    )
    identity = _ProcessIdentity(10, 1, 10, 10, 100)
    events: list[tuple[object, ...]] = []

    class FakeProcess:
        pid = 10

        @staticmethod
        def poll() -> int:
            return -signal.SIGKILL

    supervisor = object.__new__(SandboxSupervisor)
    supervisor.config = SandboxExecutorConfig(termination_grace_seconds=0.01)
    monkeypatch.setattr(
        supervisor,
        "_set_cgroup_frozen",
        lambda path, *, frozen: events.append(("freeze", path, frozen)),
    )

    def attest(**kwargs):  # type: ignore[no-untyped-def]
        events.append(("attest", kwargs["proc_pid"], kwargs["container_id"]))
        return {identity.pid: identity}

    monkeypatch.setattr(supervisor, "_attest_later_runtime_cgroup", attest)
    monkeypatch.setattr(
        supervisor,
        "_write_cgroup_control",
        lambda path, value: events.append(("write", path, value)),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_process_group",
        lambda _proc: pytest.fail("attested runtime was terminated before freeze"),
    )
    monkeypatch.setattr(
        supervisor,
        "_kill_process_group",
        lambda _proc, sig: events.append(("signal", sig)),
    )

    retained = supervisor._finish_runtime_boundary(
        proc=FakeProcess(),  # type: ignore[arg-type]
        request_cgroup=request_cgroup,
        cgroup_attested=True,
        container_id="cc-test",
        owned_inventory={identity.pid: identity},
    )

    assert retained == {identity.pid: identity}
    assert events == [
        ("freeze", str(outer), True),
        ("attest", 10, "cc-test"),
        ("write", str(outer / "cgroup.kill"), "1"),
        ("signal", signal.SIGKILL),
    ]


def test_unattested_terminal_boundary_preserves_process_group_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    request_cgroup = _RequestCgroup(
        outer="/test/outer",
        controller="/test/outer/controller",
        candidate="/test/outer/candidate",
        oci_path="requests/test/candidate",
        memory_events_before={},
        pids_events_before={},
        outer_controllers=frozenset({"cpu", "memory", "pids"}),
    )

    class FakeProcess:
        pid = 10

        @staticmethod
        def poll() -> int:
            return -signal.SIGTERM

    supervisor = object.__new__(SandboxSupervisor)
    supervisor.config = SandboxExecutorConfig(termination_grace_seconds=0.01)
    monkeypatch.setattr(
        supervisor,
        "_terminate_process_group",
        lambda _proc: events.append(("terminate", signal.SIGTERM)),
    )
    monkeypatch.setattr(
        supervisor,
        "_set_cgroup_frozen",
        lambda *_args, **_kwargs: pytest.fail("unattested runtime was frozen"),
    )
    monkeypatch.setattr(
        supervisor,
        "_kill_process_group",
        lambda _proc, sig: events.append(("signal", sig)),
    )

    retained = supervisor._finish_runtime_boundary(
        proc=FakeProcess(),  # type: ignore[arg-type]
        request_cgroup=request_cgroup,
        cgroup_attested=False,
        container_id="cc-test",
        owned_inventory={},
    )

    assert retained == {}
    assert events == [
        ("terminate", signal.SIGTERM),
        ("signal", signal.SIGKILL),
    ]


def test_trusted_guest_cpu_status_must_cross_the_exact_budget():
    effective = _request()["task"]["limits"]["effective"]
    marker = b"ready\n"
    status_marker = b"status:"
    status = {
        "version": 1,
        "candidate_ready_attested": True,
        "returncode": -9,
        "cpu_usage_us": 2_000_000,
        "cpu_limit_us": 2_000_000,
        "cpu_limit_hit": True,
        "process_peak": 1,
        "process_limit": 2,
        "process_rlimit_nproc": 1,
        "process_limit_hit": False,
        "process_limit_syscall": None,
        "tracer_killed_main": True,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "file_space_limit_hit": False,
        "file_space_limit_source": None,
        "file_size_limit_bytes": 2 * 1024 * 1024,
        "writable_limit_bytes": 0,
        "file_limit_signal": None,
        "file_limit_errno": None,
        "file_size_observed_bytes": 0,
        "writable_available_bytes": 0,
    }

    def framed(cpu_usage_us: int) -> bytes:
        measured = {**status, "cpu_usage_us": cpu_usage_us}
        encoded = base64.b64encode(
            json.dumps(
                measured,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        return marker + status_marker + encoded + b"\n"

    for cpu_usage_us in (2_000_000, 4_000_000):
        _stderr, measured = SandboxSupervisor._trusted_status(
            framed(cpu_usage_us),
            marker=marker,
            status_marker=status_marker,
            effective=effective,
        )
        assert measured["cpu_usage_us"] == cpu_usage_us

    for cpu_usage_us in (1_999_999, 4_000_001):
        with pytest.raises(
            SupervisorConfigurationError,
            match="limit evidence",
        ):
            SandboxSupervisor._trusted_status(
                framed(cpu_usage_us),
                marker=marker,
                status_marker=status_marker,
                effective=effective,
            )


def test_release_archive_and_extracted_runsc_have_independent_pins(
    tmp_path, monkeypatch
):
    archive = tmp_path / "gvisor.tar.bz2"
    binary = tmp_path / "runsc"
    rootfs = tmp_path / "rootfs"
    archive.write_bytes(b"pinned release archive")
    binary.write_bytes(b"independently pinned extracted executable")
    rootfs.mkdir()
    archive.chmod(0o444)
    binary.chmod(0o755)

    real_fstat = os.fstat

    def root_owned_fstat(fd):
        values = list(real_fstat(fd))
        values[4] = 0
        values[5] = 0
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", root_owned_fstat)
    archive_digest = hashlib.sha512(archive.read_bytes()).hexdigest()
    binary_digest = hashlib.sha512(binary.read_bytes()).hexdigest()

    def make_supervisor():
        supervisor = SandboxSupervisor(
            SandboxExecutorConfig(
                runsc_path=str(binary),
                runsc_release_archive_path=str(archive),
                rootfs_path=str(rootfs),
                enforce_request_cgroup=False,
            )
        )
        monkeypatch.setattr(
            supervisor, "_measure_rootfs_mount", lambda: ("test-readonly",)
        )
        monkeypatch.setattr(supervisor, "_attest_runsc_version", lambda _fd: None)
        return supervisor

    with pytest.raises(SupervisorConfigurationError, match="release archive SHA-512"):
        make_supervisor().freeze_runtime(
            expected_runsc_sha512=binary_digest,
            expected_release_archive_sha512="0" * 128,
            expected_runsc_size_bytes=len(binary.read_bytes()),
            expected_release_archive_size_bytes=len(archive.read_bytes()),
        )
    with pytest.raises(SupervisorConfigurationError, match="runsc SHA-512"):
        make_supervisor().freeze_runtime(
            expected_runsc_sha512="0" * 128,
            expected_release_archive_sha512=archive_digest,
            expected_runsc_size_bytes=len(binary.read_bytes()),
            expected_release_archive_size_bytes=len(archive.read_bytes()),
        )
    supervisor = make_supervisor()
    supervisor.freeze_runtime(
        expected_runsc_sha512=binary_digest,
        expected_release_archive_sha512=archive_digest,
        expected_runsc_size_bytes=len(binary.read_bytes()),
        expected_release_archive_size_bytes=len(archive.read_bytes()),
    )
    assert supervisor.frozen_runsc_path.startswith("/proc/self/fd/")
