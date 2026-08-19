"""Provider-free process, FD, protocol, and deadline oracles for Docent."""

import gzip
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import pytest

import infra.envs.debate.docent_export as docent_export
from infra.envs.debate.docent_export import (
    DocentUploadControlFlow,
    DocentUploadFailure,
    DocentUploadResult,
    export_jsonl,
    upload,
)
from infra.envs.singleturn_docent import agent_runs
from test_singleturn_docent import RECORDS


HELPER_SOURCE = r'''
import json, os, signal, socket, sys, time
W = 199
name = os.environ["DEBATE_DOCENT_COLLECTION_NAME"]
namespace = os.environ["DEBATE_DOCENT_LAUNCH_NAMESPACE"]
behavior = __BEHAVIOR__
pid_file = __PID_FILE__
socket_address = __SOCKET_ADDRESS__
cid = "helper-collection"
def raw(data):
    os.write(W, data)
def event(value):
    raw(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
def confirmed():
    event({"event":"collection","id":cid})
def success():
    event({"event":"terminal","ok":True})
if len(sys.argv) != 1 or any("TOP-SECRET" in arg for arg in sys.argv):
    os._exit(91)
if behavior == "environment":
    forbidden = {"HOME", "DOCENT_PROFILE", "DOCENT_COLLECTION_ID", "HTTP_PROXY",
                 "HTTPS_PROXY", "ALL_PROXY", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                 "SSL_CERT_FILE", "SSLKEYLOGFILE", "PYTHONPATH", "LD_PRELOAD",
                 "DYLD_INSERT_LIBRARIES", "DOCENT_TEST_SECRET", "LANG", "LC_ALL"}
    if forbidden.intersection(os.environ): os._exit(93)
    confirmed(); success(); os._exit(0)
if behavior == "fd_census":
    if os.getpid() != os.getpgrp(): os._exit(95)
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        os._exit(96)
    extras = []
    for fd in range(3, 512):
        if fd in {198, 199}: continue
        try: os.fstat(fd)
        except OSError: continue
        extras.append(fd)
    if extras: os._exit(94)
    confirmed(); success(); os._exit(0)
if behavior == "success":
    confirmed(); success(); os._exit(0)
if behavior == "success_fork":
    confirmed(); success()
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True: time.sleep(1)
    with open(pid_file, "w") as f: f.write(str(child))
    os._exit(0)
if behavior == "nonzero_success":
    confirmed(); success(); os._exit(7)
if behavior == "duplicate_keys":
    raw(b'{"event":"terminal","event":"terminal"}\n'); os._exit(2)
if behavior == "oversize":
    raw(b"x" * 5000 + b"\n"); time.sleep(10)
if behavior == "duplicate_collection":
    confirmed(); confirmed(); time.sleep(10)
if behavior == "continuous_writer":
    while True: raw(b"x" * 4096)
if behavior == "term_socket":
    confirmed()
    sock = socket.create_connection(socket_address, timeout=1.0)
    def on_term(_signum, _frame):
        sock.sendall(("TERM " + repr(time.monotonic())).encode())
    signal.signal(signal.SIGTERM, on_term)
    while True: time.sleep(1)
if behavior == "crash_pre":
    os._exit(3)
if behavior == "crash_post":
    confirmed(); os._exit(3)
if behavior in {"hang", "id_hang", "fork_hang", "interrupt"}:
    if behavior in {"id_hang", "fork_hang"}:
        confirmed()
    if behavior == "fork_hang":
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True: time.sleep(1)
        with open(pid_file, "w") as f:
            f.write(str(child))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if behavior == "interrupt":
        os.kill(os.getppid(), signal.SIGINT)
    while True: time.sleep(1)
os._exit(92)
'''


TRANSPORT_ORACLE_SOURCE = r'''
import json, os, sys, time
sys.path.insert(0, __PROJECT_ROOT__)
from requests.adapters import HTTPAdapter
from infra.envs.debate.docent_export import (
    _DocentHTTPTransport, _DocentTimeBudget, _wait_for_docent_jobs,
)
W = 199
def event(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if len(payload) > os.fpathconf(W, "PC_PIPE_BUF"): os._exit(90)
    os.write(W, payload)
deadline = float(os.environ["DEBATE_DOCENT_DEADLINE_MONOTONIC"])
transport = _DocentHTTPTransport(
    os.environ["DOCENT_API_KEY"],
    _DocentTimeBudget(deadline=deadline, clock=time.monotonic, sleeper=time.sleep),
)
if transport.session.trust_env is not False: os._exit(91)
for adapter in transport.session.adapters.values():
    retries = adapter.max_retries
    if any(getattr(retries, field) != 0 for field in
           ("total", "connect", "read", "redirect", "status", "other")):
        os._exit(92)
class RecordingAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        request.headers["X-Test-Timeout"] = json.dumps(list(kwargs["timeout"]))
        return super().send(request, **kwargs)
for prefix in ("http://", "https://"):
    previous = transport.session.adapters[prefix]
    transport.session.mount(prefix, RecordingAdapter(max_retries=previous.max_retries))
try:
    transport.authenticate()
    collection_id = transport.create_collection(
        os.environ["DEBATE_DOCENT_COLLECTION_NAME"]
    )
    event({"event": "collection", "id": collection_id})
    first = transport.enqueue_batch(collection_id, b'{"agent_runs":[]}')
    job_ids = [first] + [f"oracle-job-{index}" for index in range(100)]
    _wait_for_docent_jobs(transport, collection_id, job_ids)
    event({"event": "terminal", "ok": True})
except BaseException as exc:
    name = type(exc).__name__
    event({"event": "terminal", "ok": False,
           "error": name if name.isidentifier() and len(name) <= 128 else "UnknownError"})
finally:
    transport.close()
'''


def _helper(
    tmp_path: Path,
    behavior: str = "success",
    pid_file: Path | None = None,
    socket_address: tuple[str, int] | None = None,
) -> Path:
    path = tmp_path / f"docent-helper-{behavior}.py"
    source = HELPER_SOURCE.replace("__BEHAVIOR__", repr(behavior)).replace(
        "__PID_FILE__", repr(os.fspath(pid_file)) if pid_file is not None else "None"
    ).replace("__SOCKET_ADDRESS__", repr(socket_address))
    path.write_text(source)
    return path.resolve()


def _transport_oracle_helper(tmp_path: Path) -> Path:
    path = tmp_path / "docent-transport-oracle.py"
    project_root = os.fspath(Path(__file__).resolve().parents[1])
    path.write_text(
        TRANSPORT_ORACLE_SOURCE.replace("__PROJECT_ROOT__", repr(project_root))
    )
    return path.resolve()


@pytest.fixture(autouse=True)
def _safe_docent_environment(monkeypatch):
    monkeypatch.setenv("DOCENT_API_KEY", "test-only-key")
    for name in ("DOCENT_API_URL", "DOCENT_FRONTEND_URL", "DOCENT_DOMAIN"):
        monkeypatch.delenv(name, raising=False)


def _canonical_path(tmp_path: Path) -> tuple[Path, list]:
    runs = agent_runs(RECORDS)
    path = tmp_path / "docent.jsonl"
    export_jsonl(runs, str(path))
    return path, runs


def _invoke(path: Path, *, helper: Path, seconds: float = 1.0):
    fd = os.open(path, os.O_RDONLY)
    try:
        return upload(
            fd,
            "math-pc-rl",
            "run-A",
            _deadline_seconds=seconds,
            _worker_path=helper,
        )
    finally:
        os.close(fd)


def test_parent_uses_posix_spawn_fixed_fds_and_no_sensitive_argv(
    tmp_path, monkeypatch
):
    path, _runs = _canonical_path(tmp_path)
    monkeypatch.setenv("DOCENT_TEST_SECRET", "TOP-SECRET-PAYLOAD")
    result = _invoke(path, helper=_helper(tmp_path, "success"))
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="helper-collection",
    )
    source = inspect.getsource(docent_export._spawn_docent_upload_child)
    assert "os.posix_spawn(" in source
    assert "Popen" not in source and "fork" not in source
    assert "_DOCENT_JSONL_FD" in source and "_DOCENT_RECEIPT_FD" in source


def test_child_environment_is_allowlisted_and_excludes_parent_secrets(
    tmp_path, monkeypatch
):
    path, _runs = _canonical_path(tmp_path)
    monkeypatch.setenv("HOME", "/TOP-SECRET-HOME")
    monkeypatch.setenv("HTTP_PROXY", "http://TOP-SECRET-PROXY")
    monkeypatch.setenv("SSLKEYLOGFILE", "/TOP-SECRET-KEYLOG")
    monkeypatch.setenv("DOCENT_PROFILE", "TOP-SECRET-PROFILE")
    monkeypatch.setenv("DOCENT_TEST_SECRET", "TOP-SECRET-PAYLOAD")
    result = _invoke(path, helper=_helper(tmp_path, "environment"))
    assert isinstance(result, DocentUploadResult)
    assert "TOP-SECRET" not in repr(result)


def test_spawn_closes_unrelated_explicitly_inheritable_fd(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    sentinel = os.open(os.devnull, os.O_RDONLY)
    os.set_inheritable(sentinel, True)
    try:
        result = _invoke(path, helper=_helper(tmp_path, "fd_census"))
    finally:
        os.close(sentinel)
    assert isinstance(result, DocentUploadResult)


def test_spawned_child_identity_and_parent_alarm_thread_state_are_isolated(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    helper = _helper(tmp_path, "fd_census")
    probe = tmp_path / "parent-state-probe.py"
    probe.write_text(
        """
import os, signal, sys, threading, time
from pathlib import Path
from infra.envs.debate.docent_export import DocentUploadResult, upload
import torch
from tqdm import tqdm

jsonl, worker = sys.argv[1:]
alarm = signal.SIGALRM
old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {alarm})
old_handler = signal.getsignal(alarm)
stop = threading.Event()
thread = threading.Thread(target=stop.wait)
torch_module = sys.modules["torch"]
previous_monitor_interval = tqdm.monitor_interval
tqdm.monitor_interval = 37
def custom_handler(_signum, _frame):
    raise AssertionError("blocked parent SIGALRM was delivered")
try:
    signal.signal(alarm, custom_handler)
    signal.setitimer(signal.ITIMER_REAL, 0.01, 0.20)
    deadline = time.monotonic() + 1.0
    while alarm not in signal.sigpending():
        if time.monotonic() >= deadline: raise AssertionError("alarm not pending")
        time.sleep(0.002)
    thread.start()
    before_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    before_pending = signal.sigpending()
    before_interval = signal.getitimer(signal.ITIMER_REAL)[1]
    fd = os.open(jsonl, os.O_RDONLY)
    try:
        result = upload(fd, "math-pc-rl", "run-A", _deadline_seconds=1.0,
                        _worker_path=Path(worker))
    finally:
        os.close(fd)
    assert isinstance(result, DocentUploadResult)
    assert thread.is_alive()
    assert sys.modules["torch"] is torch_module
    assert tqdm.monitor_interval == 37
    assert signal.getsignal(alarm) is custom_handler
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == before_mask
    assert signal.sigpending() == before_pending
    after = signal.getitimer(signal.ITIMER_REAL)
    assert before_interval == after[1] == 0.20 and after[0] > 0
    print("OK")
finally:
    signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
    while alarm in signal.sigpending(): signal.sigwait({alarm})
    signal.signal(alarm, old_handler)
    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    stop.set()
    tqdm.monitor_interval = previous_monitor_interval
    if thread.ident is not None: thread.join(timeout=1)
"""
    )
    environment = os.environ.copy()
    environment["DOCENT_API_KEY"] = "test-only-key"
    completed = subprocess.run(
        [sys.executable, os.fspath(probe), os.fspath(path), os.fspath(helper)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert completed.stdout.strip() == "OK"


def test_parent_runtime_signal_thread_torch_and_tqdm_state_is_unchanged(tmp_path):
    from tqdm import tqdm

    path, _runs = _canonical_path(tmp_path)
    before_handler = signal.getsignal(signal.SIGALRM)
    before_timer = signal.getitimer(signal.ITIMER_REAL)
    before_threads = tuple(thread.ident for thread in threading.enumerate())
    before_monitor = tqdm.monitor_interval
    torch_before = "torch" in __import__("sys").modules
    result = _invoke(path, helper=_helper(tmp_path, "success"))
    assert isinstance(result, DocentUploadResult)
    assert signal.getsignal(signal.SIGALRM) is before_handler
    assert signal.getitimer(signal.ITIMER_REAL) == before_timer
    assert tuple(thread.ident for thread in threading.enumerate()) == before_threads
    assert tqdm.monitor_interval == before_monitor
    assert ("torch" in __import__("sys").modules) == torch_before


def _assert_process_gone(pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail("Docent descendant survived process-group sweep")


def test_nominal_success_sweeps_descendant_before_reaping_leader(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    pid_file = tmp_path / "success-descendant.pid"
    result = _invoke(
        path,
        helper=_helper(tmp_path, "success_fork", pid_file),
        seconds=1.0,
    )
    assert isinstance(result, DocentUploadResult)
    _assert_process_gone(int(pid_file.read_text()))


@pytest.mark.parametrize(
    "behavior",
    ["duplicate_keys", "oversize", "duplicate_collection"],
)
def test_malformed_oversize_and_duplicate_receipts_fail_closed(
    tmp_path, monkeypatch, behavior
):
    path, _runs = _canonical_path(tmp_path)
    result = _invoke(path, helper=_helper(tmp_path, behavior), seconds=1.0)
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadProtocolError"


def _receipt_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_no_id_ambiguous_terminal_is_the_only_valid_terminal_without_identity():
    state = docent_export._DocentReceiptState(
        "math-pc-rl--launch-run-A", "run-A"
    )
    docent_export._accept_docent_receipt_frame(
        state,
        _receipt_bytes(
            {
                "event": "terminal",
                "ok": False,
                "error": "RuntimeError",
            }
        ),
    )
    assert state.protocol_error is None
    assert state.terminal == DocentUploadFailure(
        collection_name=state.collection_name,
        launch_namespace=state.launch_namespace,
        collection_id=None,
        error_type="RuntimeError",
    )


@pytest.mark.parametrize(
    "frames",
    [
        # Confirmed success before the required collection identity.
        [{"event": "terminal", "ok": True}],
        # An ID arriving after any terminal frame is out of order.
        [
            {"event": "terminal", "ok": False, "error": "RuntimeError"},
            {"event": "collection", "id": "id-1"},
        ],
        [{"event": "unknown"}],
        [
            {"event": "terminal", "ok": False, "error": "not an identifier"}
        ],
        [
            {"event": "terminal", "ok": False, "error": "RuntimeError", "extra": True}
        ],
        [
            {"event": "collection", "id": "id-1"},
            {"event": "terminal", "ok": True, "id": "id-2"},
        ],
    ],
)
def test_strict_receipt_order_schema_and_identity_matrix(frames):
    state = docent_export._DocentReceiptState(
        "math-pc-rl--launch-run-A", "run-A"
    )
    for frame in frames:
        docent_export._accept_docent_receipt_frame(state, _receipt_bytes(frame))
    assert state.protocol_error == "DocentUploadProtocolError"


def test_partial_receipt_at_eof_is_protocol_failure():
    state = docent_export._DocentReceiptState(
        "math-pc-rl--launch-run-A", "run-A"
    )
    docent_export._consume_docent_receipt_bytes(
        state, bytearray(), b'{"event":"terminal"', eof=True
    )
    assert state.protocol_error == "DocentUploadProtocolError"


def test_success_frame_with_nonzero_exit_is_not_confirmed(tmp_path, monkeypatch):
    path, _runs = _canonical_path(tmp_path)
    result = _invoke(path, helper=_helper(tmp_path, "nonzero_success"))
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="helper-collection",
        error_type="DocentUploadChildExitError",
    )


@pytest.mark.parametrize(
    ("behavior", "expected_id"),
    [("crash_pre", None), ("crash_post", "helper-collection")],
)
def test_child_crash_preserves_only_last_received_collection_id(
    tmp_path, monkeypatch, behavior, expected_id
):
    path, _runs = _canonical_path(tmp_path)
    result = _invoke(path, helper=_helper(tmp_path, behavior))
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id == expected_id
    assert result.error_type == "DocentUploadChildExitError"


def test_deadline_kills_and_reaps_entire_fresh_process_group(tmp_path, monkeypatch):
    path, _runs = _canonical_path(tmp_path)
    pid_file = tmp_path / "descendant.pid"
    result = _invoke(
        path, helper=_helper(tmp_path, "fork_hang", pid_file), seconds=0.4
    )
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadDeadlineError"
    assert result.collection_id == "helper-collection"
    _assert_process_gone(int(pid_file.read_text()))


def test_cleanup_grace_is_reserved_inside_total_deadline(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    started = time.monotonic()
    result = _invoke(path, helper=_helper(tmp_path, "hang"), seconds=0.5)
    elapsed = time.monotonic() - started
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadDeadlineError"
    assert elapsed < 0.56


def test_term_drain_window_can_still_carry_loopback_activity_before_kill(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    received = {}

    def receive_term():
        connection, _address = listener.accept()
        with connection:
            received["payload"] = connection.recv(256)
            received["at"] = time.monotonic()

    receiver = threading.Thread(target=receive_term)
    receiver.start()
    started = time.monotonic()
    result = _invoke(
        path,
        helper=_helper(
            tmp_path,
            "term_socket",
            socket_address=listener.getsockname(),
        ),
        seconds=0.5,
    )
    finished = time.monotonic()
    receiver.join(timeout=2.0)
    listener.close()
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadDeadlineError"
    assert result.collection_id == "helper-collection"
    assert received["payload"].startswith(b"TERM ")
    child_timestamp = float(received["payload"].split(maxsplit=1)[1])
    assert started + 0.15 <= child_timestamp <= received["at"] <= finished
    assert finished - received["at"] <= 0.30
    assert finished - started < 0.56


def test_continuous_receipt_writer_cannot_monopolize_parent(tmp_path):
    path, _runs = _canonical_path(tmp_path)
    started = time.monotonic()
    result = _invoke(
        path, helper=_helper(tmp_path, "continuous_writer"), seconds=0.5
    )
    elapsed = time.monotonic() - started
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadProtocolError"
    assert elapsed < 0.65


def test_keyboard_interrupt_cleans_group_and_returns_sanitized_carrier(
    tmp_path, monkeypatch
):
    path, _runs = _canonical_path(tmp_path)
    result = _invoke(path, helper=_helper(tmp_path, "interrupt"), seconds=2.0)
    assert isinstance(result, DocentUploadControlFlow)
    assert result.kind == "KeyboardInterrupt"
    assert result.failure.collection_id is None
    assert "TOP-SECRET" not in repr(result)


@pytest.mark.parametrize(
    ("exit_value", "expected_code"),
    [(None, 1), ("TOP-SECRET-SYSTEM-EXIT", 1), (7, 7)],
)
def test_system_exit_during_wait_cleans_child_and_preserves_control(
    tmp_path, monkeypatch, exit_value, expected_code
):
    path, _runs = _canonical_path(tmp_path)
    original_select = docent_export.select.select
    fired = False

    def interrupt_once(*args, **kwargs):
        nonlocal fired
        if not fired:
            fired = True
            raise SystemExit(exit_value)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(docent_export.select, "select", interrupt_once)
    result = _invoke(path, helper=_helper(tmp_path, "hang"), seconds=2.0)
    assert result == DocentUploadControlFlow(
        failure=DocentUploadFailure(
            collection_name="math-pc-rl--launch-run-A",
            launch_namespace="run-A",
            collection_id=None,
            error_type="SystemExit",
        ),
        kind="SystemExit",
        exit_code=expected_code,
    )
    assert fired


@pytest.mark.parametrize(
    "content",
    [b"\n", b"{}", b"not-json\n", b" {\"name\":\"x\"}\n"],
)
def test_real_worker_rejects_noncanonical_jsonl_before_docent_construction(
    tmp_path, content
):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(content)
    fd = os.open(path, os.O_RDONLY)
    try:
        result = upload(fd, "math-pc-rl", "run-A", _deadline_seconds=1.0)
    finally:
        os.close(fd)
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id is None
    assert result.error_type in {"ValueError", "ValidationError"}


class _DocentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = []
    batch_body = None
    batch_status = 202
    batch_delay = 0.0
    fail_path = None
    fail_status = 500
    collection_id = "loopback-collection"
    batch_started = None
    batch_trickle_interval = 0.0
    raw_batch_bodies = []
    decoded_batch_bodies = []
    create_body = None
    batch_job_ids = None
    status_responses = None
    status_requests = []
    observed_timeouts = []
    capture_batch_bodies = True
    batch_count = 0

    def log_message(self, _format, *_args):
        pass

    def _reply(self, body, status=200):
        payload = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except BrokenPipeError:
            pass

    def do_GET(self):
        type(self).observed_timeouts.append(self.headers.get("X-Test-Timeout"))
        type(self).calls.append(("GET", self.path))
        if self.path == type(self).fail_path:
            self._reply({"error": "redacted"}, status=type(self).fail_status)
            return
        self._reply({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).observed_timeouts.append(self.headers.get("X-Test-Timeout"))
        type(self).calls.append(("POST", self.path))
        if self.path == type(self).fail_path:
            self._reply({"error": "redacted"}, status=type(self).fail_status)
            return
        if self.path.endswith("/create"):
            type(self).create_body = json.loads(body)
            self._reply({"collection_id": type(self).collection_id})
        elif self.path.endswith("/agent_runs"):
            index = type(self).batch_count
            type(self).batch_count += 1
            if type(self).capture_batch_bodies:
                type(self).raw_batch_bodies.append(body)
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                type(self).batch_body = body
                type(self).decoded_batch_bodies.append(body)
            if type(self).batch_started is not None:
                type(self).batch_started.set()
            if type(self).batch_trickle_interval:
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "1000000")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(type(self).batch_trickle_interval)
                except OSError:
                    pass
                return
            if type(self).batch_delay:
                time.sleep(type(self).batch_delay)
            job_ids = type(self).batch_job_ids
            job_id = "loopback-job" if job_ids is None else job_ids[index]
            self._reply({"job_id": job_id}, status=type(self).batch_status)
        elif self.path.endswith("/agent_runs/jobs/batch_status"):
            request = json.loads(body)
            type(self).status_requests.append(request["job_ids"])
            responses = type(self).status_responses
            if responses is None:
                rows = [
                    {"job_id": job_id, "status": "completed"}
                    for job_id in request["job_ids"]
                ]
            else:
                rows = responses[len(type(self).status_requests) - 1]
            self._reply({"jobs": rows})
        else:
            self._reply({}, status=404)


@pytest.fixture
def docent_loopback(monkeypatch):
    _DocentHandler.calls = []
    _DocentHandler.batch_body = None
    _DocentHandler.batch_status = 202
    _DocentHandler.batch_delay = 0.0
    _DocentHandler.batch_trickle_interval = 0.0
    _DocentHandler.batch_started = None
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    _DocentHandler.create_body = None
    _DocentHandler.batch_job_ids = None
    _DocentHandler.status_responses = None
    _DocentHandler.status_requests = []
    _DocentHandler.observed_timeouts = []
    _DocentHandler.capture_batch_bodies = True
    _DocentHandler.batch_count = 0
    _DocentHandler.raw_batch_bodies = []
    _DocentHandler.decoded_batch_bodies = []
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("DOCENT_API_URL", endpoint)
    monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_spawned_worker_requires_immediate_job_id_before_polling(
    tmp_path, docent_loopback
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.batch_job_ids = [None]
    result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=2.0)
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
        error_type="DocentIngestionAcknowledgementError",
    )
    assert _DocentHandler.status_requests == []


def test_spawned_worker_rejects_duplicate_batch_id_before_later_send(
    tmp_path, docent_loopback
):
    runs = agent_runs(RECORDS)
    # The pinned serializer escapes this character to six ASCII bytes. Each
    # run is therefore >50 MiB but <100 MiB at the reviewed batching boundary:
    # three runs deterministically make three batches without a giant JSONL.
    padding = "é" * 9_000_000
    for run in runs:
        run.metadata["batch_oracle_padding"] = padding
    path = tmp_path / "docent.jsonl"
    export_jsonl(runs, str(path))
    del padding, runs

    _DocentHandler.capture_batch_bodies = False
    _DocentHandler.batch_job_ids = ["duplicate-job", "duplicate-job", "later-job"]
    result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=30.0)
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
        error_type="DocentIngestionAcknowledgementError",
    )
    assert _DocentHandler.batch_count == 2
    assert _DocentHandler.status_requests == []


def test_spawned_worker_observes_pending_running_then_exact_completed_census(
    tmp_path, docent_loopback
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.status_responses = [
        [{"job_id": "loopback-job", "status": "pending"}],
        [{"job_id": "loopback-job", "status": "running"}],
        [{"job_id": "loopback-job", "status": "completed"}],
    ]
    result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=4.0)
    assert isinstance(result, DocentUploadResult)
    assert _DocentHandler.status_requests == [["loopback-job"]] * 3


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {"job_id": "loopback-job", "status": "completed"},
            {"job_id": "extra", "status": "completed"},
        ],
        [
            {"job_id": "loopback-job", "status": "completed"},
            {"job_id": "loopback-job", "status": "completed"},
        ],
        [{"job_id": "loopback-job", "status": "failed"}],
        [{"job_id": "loopback-job", "status": "canceled"}],
        [{"job_id": "loopback-job", "status": "mystery"}],
        [{"job_id": None, "status": "completed"}],
    ],
)
def test_spawned_worker_refuses_bad_real_status_census(
    tmp_path, docent_loopback, rows
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.status_responses = [rows]
    result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=2.0)
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
        error_type="DocentIngestionStatusError",
    )


def test_spawned_explicit_transport_harness_observes_timeouts_and_101_id_slices(
    tmp_path, docent_loopback
):
    # This intentionally exercises the owned production adapter/poller in a
    # real spawned child, not the JSONL worker: producing 101 genuine batches
    # through the pinned >50 MiB batching boundary would require >5 GiB.
    path, _runs = _canonical_path(tmp_path)
    result = _invoke(path, helper=_transport_oracle_helper(tmp_path), seconds=3.0)
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
    )
    assert _DocentHandler.observed_timeouts == ["[10.0, 120.0]"] * 5
    assert [len(chunk) for chunk in _DocentHandler.status_requests] == [100, 1]
    requested = [
        item for chunk in _DocentHandler.status_requests for item in chunk
    ]
    assert requested == ["loopback-job"] + [
        f"oracle-job-{index}" for index in range(100)
    ]


def test_near_deadline_request_still_uses_exact_reviewed_socket_tuple():
    budget = docent_export._DocentTimeBudget(
        deadline=10.0,
        clock=lambda: 9.999999,
        sleeper=lambda _seconds: None,
    )
    assert budget.request_timeout() == (10.0, 120.0)


def test_owned_transport_dynamically_blocks_ambient_proxy_and_has_zero_retries(
    docent_loopback, monkeypatch
):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    budget = docent_export._DocentTimeBudget(
        deadline=time.monotonic() + 10.0,
        clock=time.monotonic,
        sleeper=lambda _seconds: None,
    )
    transport = docent_export._DocentHTTPTransport("explicit-key", budget)
    try:
        assert transport.session.trust_env is False
        for adapter in transport.session.adapters.values():
            retries = adapter.max_retries
            assert {
                field: getattr(retries, field)
                for field in (
                    "total",
                    "connect",
                    "read",
                    "redirect",
                    "status",
                    "other",
                )
            } == {
                "total": 0,
                "connect": 0,
                "read": 0,
                "redirect": 0,
                "status": 0,
                "other": 0,
            }
        # This real loopback request would attempt the seeded dead proxy if
        # Requests were allowed to merge ambient environment configuration.
        transport.authenticate()
    finally:
        transport.close()


def test_real_worker_installed_sdk_receives_exact_scientific_json_bytes_once(
    tmp_path, monkeypatch
):
    records = deepcopy(RECORDS)
    records[0]["completion"] = "café λ remains scientific"
    runs = agent_runs(records)
    path = tmp_path / "docent.jsonl"
    export_jsonl(runs, str(path))
    local_bytes = path.read_bytes()
    _DocentHandler.calls = []
    _DocentHandler.batch_body = None
    _DocentHandler.batch_status = 202
    _DocentHandler.batch_delay = 0.0
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_KEY", "test-only-loopback-key")
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        fd = os.open(path, os.O_RDONLY)
        try:
            result = upload(fd, "math-pc-rl", "run-A", _deadline_seconds=3.0)
        finally:
            os.close(fd)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result == DocentUploadResult(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
    )
    assert _DocentHandler.calls == [
        ("GET", "/rest/api-keys/test"),
        ("POST", "/rest/create"),
        ("POST", "/rest/loopback-collection/agent_runs"),
        ("POST", "/rest/loopback-collection/agent_runs/jobs/batch_status"),
    ]
    from docent.sdk._collections import (
        MAX_AGENT_RUN_PAYLOAD_BYTES,
        yield_agent_run_batches_by_size,
    )

    expected_batches = list(
        yield_agent_run_batches_by_size(runs, MAX_AGENT_RUN_PAYLOAD_BYTES)
    )
    assert len(expected_batches) == 1
    assert _DocentHandler.batch_body == expected_batches[0][1]
    assert path.read_bytes() == local_bytes == b"".join(
        run.model_dump_json().encode("utf-8") + b"\n" for run in runs
    )
    assert "café λ".encode("utf-8") in local_bytes
    assert "café λ".encode("utf-8") not in _DocentHandler.batch_body


def test_long_valid_collection_name_survives_compact_atomic_receipt(
    tmp_path, monkeypatch
):
    path, _runs = _canonical_path(tmp_path)
    receipt_pipe_bufs = []
    original_pipe = os.pipe

    def observed_pipe():
        read_fd, write_fd = original_pipe()
        receipt_pipe_bufs.append(os.fpathconf(write_fd, "PC_PIPE_BUF"))
        return read_fd, write_fd

    monkeypatch.setattr(docent_export.os, "pipe", observed_pipe)
    base_name = "experiment-" + "x" * 12_000
    _DocentHandler.calls = []
    _DocentHandler.create_body = None
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    _DocentHandler.batch_job_ids = None
    _DocentHandler.status_responses = None
    _DocentHandler.status_requests = []
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        fd = os.open(path, os.O_RDONLY)
        try:
            result = upload(fd, base_name, "run-A", _deadline_seconds=3.0)
        finally:
            os.close(fd)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    expected_name = f"{base_name}--launch-run-A"
    assert result == DocentUploadResult(
        collection_name=expected_name,
        launch_namespace="run-A",
        collection_id="loopback-collection",
    )
    assert _DocentHandler.create_body["name"] == expected_name
    assert receipt_pipe_bufs
    assert len(expected_name.encode("utf-8")) > receipt_pipe_bufs[0]


def test_two_gzip_envelopes_vary_while_unicode_sdk_payload_bytes_stay_exact(
    tmp_path, monkeypatch
):
    records = deepcopy(RECORDS)
    records[0]["completion"] = "café λ remains scientific"
    runs = agent_runs(records)
    path = tmp_path / "docent.jsonl"
    export_jsonl(runs, str(path))
    local_bytes = path.read_bytes()
    from docent.sdk._collections import (
        MAX_AGENT_RUN_PAYLOAD_BYTES,
        yield_agent_run_batches_by_size,
    )

    expected = list(
        yield_agent_run_batches_by_size(runs, MAX_AGENT_RUN_PAYLOAD_BYTES)
    )[0][1]
    _DocentHandler.calls = []
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    _DocentHandler.batch_delay = 0.0
    _DocentHandler.batch_trickle_interval = 0.0
    _DocentHandler.raw_batch_bodies = []
    _DocentHandler.decoded_batch_bodies = []
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        first = _invoke(
            path, helper=docent_export._DOCENT_WORKER_PATH, seconds=3.0
        )
        time.sleep(1.1)
        second = _invoke(
            path, helper=docent_export._DOCENT_WORKER_PATH, seconds=3.0
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    assert isinstance(first, DocentUploadResult)
    assert isinstance(second, DocentUploadResult)
    assert len(_DocentHandler.raw_batch_bodies) == 2
    raw_first, raw_second = _DocentHandler.raw_batch_bodies
    assert raw_first.startswith(b"\x1f\x8b") and raw_second.startswith(b"\x1f\x8b")
    assert raw_first != raw_second
    assert raw_first[4:8] != raw_second[4:8]  # permitted gzip mtime header
    assert _DocentHandler.decoded_batch_bodies == [expected, expected]
    assert path.read_bytes() == local_bytes


@pytest.mark.parametrize(
    ("batch_status", "batch_delay", "seconds", "expected_error"),
    [
        (500, 0.0, 2.0, "DocentMutationHTTPStatusError"),
        (202, 2.0, 0.8, "DocentUploadDeadlineError"),
    ],
)
def test_real_worker_batch_failure_or_hang_is_one_send_and_retains_id(
    tmp_path,
    monkeypatch,
    batch_status,
    batch_delay,
    seconds,
    expected_error,
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.calls = []
    _DocentHandler.batch_body = None
    _DocentHandler.batch_status = batch_status
    _DocentHandler.batch_delay = batch_delay
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_KEY", "test-only-loopback-key")
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        fd = os.open(path, os.O_RDONLY)
        try:
            result = upload(
                fd, "math-pc-rl", "run-A", _deadline_seconds=seconds
            )
        finally:
            os.close(fd)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id="loopback-collection",
        error_type=expected_error,
    )
    assert [path for method, path in _DocentHandler.calls if method == "POST"].count(
        "/rest/loopback-collection/agent_runs"
    ) == 1


def test_child_absolute_alarm_ends_orphaned_hanging_upload(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.calls = []
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = "loopback-collection"
    _DocentHandler.batch_delay = 0.0
    _DocentHandler.batch_trickle_interval = 0.05
    _DocentHandler.batch_started = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DocentHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    supervisor_script = tmp_path / "orphan-supervisor.py"
    supervisor_script.write_text(
        """
import os, sys
from infra.envs.debate.docent_export import upload
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
    upload(fd, "math-pc-rl", "run-A", _deadline_seconds=1.5)
finally:
    os.close(fd)
"""
    )
    endpoint = f"http://127.0.0.1:{server.server_port}"
    environment = os.environ.copy()
    environment.update(
        {
            "DOCENT_API_KEY": "test-only-loopback-key",
            "DOCENT_API_URL": endpoint,
            "DOCENT_FRONTEND_URL": endpoint,
        }
    )
    supervisor = subprocess.Popen(
        [sys.executable, os.fspath(supervisor_script), os.fspath(path)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        assert _DocentHandler.batch_started.wait(timeout=2.0)
        children = psutil.Process(supervisor.pid).children(recursive=False)
        assert len(children) == 1
        child_pid = children[0].pid
        assert os.getpgid(child_pid) == child_pid
        orphaned_at = time.monotonic()
        os.kill(supervisor.pid, signal.SIGKILL)
        supervisor.wait(timeout=2.0)
        time.sleep(0.25)
        assert psutil.pid_exists(child_pid)
        absence_deadline = time.monotonic() + 2.5
        while psutil.pid_exists(child_pid) and time.monotonic() < absence_deadline:
            time.sleep(0.02)
        assert not psutil.pid_exists(child_pid)
        assert time.monotonic() - orphaned_at < 2.0
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=2.0)
        if child_pid is not None and psutil.pid_exists(child_pid):
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        _DocentHandler.batch_delay = 0.0
        _DocentHandler.batch_trickle_interval = 0.0
        _DocentHandler.batch_started = None


@pytest.mark.parametrize("status", [300, 302, 500])
@pytest.mark.parametrize(
    ("phase_path", "expected_id", "expected_error"),
    [
        ("/rest/api-keys/test", None, "DocentMutationHTTPStatusError"),
        ("/rest/create", None, "DocentMutationHTTPStatusError"),
        (
            "/rest/loopback-collection/agent_runs",
            "loopback-collection",
            "DocentMutationHTTPStatusError",
        ),
        (
            "/rest/loopback-collection/agent_runs/jobs/batch_status",
            "loopback-collection",
            "DocentIngestionStatusError",
        ),
    ],
)
def test_real_worker_refuses_every_redirect_or_non2xx_after_one_send(
    tmp_path, monkeypatch, status, phase_path, expected_id, expected_error
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.calls = []
    _DocentHandler.batch_body = None
    _DocentHandler.batch_status = 202
    _DocentHandler.batch_delay = 0.0
    _DocentHandler.fail_path = phase_path
    _DocentHandler.fail_status = status
    _DocentHandler.collection_id = "loopback-collection"
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        _DocentHandler.fail_path = None
    if status in {301, 302, 303, 307, 308} and not phase_path.endswith(
        "/jobs/batch_status"
    ):
        expected_error = "DocentMutationRedirectError"
    assert result == DocentUploadFailure(
        collection_name="math-pc-rl--launch-run-A",
        launch_namespace="run-A",
        collection_id=expected_id,
        error_type=expected_error,
    )
    assert [path for _method, path in _DocentHandler.calls].count(phase_path) == 1


@pytest.mark.parametrize("collection_id", [None, "", "../escape", "two words"])
def test_real_worker_rejects_invalid_collection_id_before_any_batch(
    tmp_path, monkeypatch, collection_id
):
    path, _runs = _canonical_path(tmp_path)
    _DocentHandler.calls = []
    _DocentHandler.fail_path = None
    _DocentHandler.collection_id = collection_id
    server = HTTPServer(("127.0.0.1", 0), _DocentHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DOCENT_API_URL", endpoint)
        monkeypatch.setenv("DOCENT_FRONTEND_URL", endpoint)
        result = _invoke(path, helper=docent_export._DOCENT_WORKER_PATH, seconds=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        _DocentHandler.collection_id = "loopback-collection"
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id is None
    assert result.error_type == "DocentCollectionAcknowledgementError"
    assert not any(call_path.endswith("/agent_runs") for _, call_path in _DocentHandler.calls)


def test_permission_error_during_group_kill_is_not_treated_as_absence(monkeypatch):
    monkeypatch.setattr(
        docent_export.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError()),
    )
    with pytest.raises(docent_export.DocentUploadCleanupError):
        docent_export._kill_docent_process_group(12345, signal.SIGKILL)


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="Darwin EPERM rule")
def test_darwin_zombie_only_group_eperm_still_reaps_known_leader(monkeypatch):
    executable = "/usr/bin/true"
    pid = os.posix_spawn(executable, [executable], {}, setpgroup=0)
    deadline = time.monotonic() + 2.0
    while not docent_export._child_exited_without_reaping(pid):
        if time.monotonic() >= deadline:
            pytest.fail("test child did not exit")
        time.sleep(0.005)
    monkeypatch.setattr(
        docent_export.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError()),
    )
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    try:
        status = docent_export._sweep_and_reap_docent_group(
            pid,
            read_fd,
            docent_export._DocentReceiptState("collection", "run-A"),
            bytearray(),
        )
    finally:
        os.close(read_fd)
    assert os.waitstatus_to_exitcode(status) == 0
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_lost_wait_ownership_cannot_accept_terminal_success(tmp_path, monkeypatch):
    path, _runs = _canonical_path(tmp_path)
    original_sweep = docent_export._sweep_and_reap_docent_group

    def reap_then_report_lost(*args, **kwargs):
        original_sweep(*args, **kwargs)
        raise docent_export.DocentUploadChildOwnershipError("competing reaper")

    monkeypatch.setattr(
        docent_export, "_sweep_and_reap_docent_group", reap_then_report_lost
    )
    result = _invoke(path, helper=_helper(tmp_path, "success"))
    assert isinstance(result, DocentUploadFailure)
    assert result.error_type == "DocentUploadChildOwnershipError"


def test_sdk_version_drift_refuses_before_transport_or_mutation(monkeypatch):
    constructions = []

    class ForbiddenTransport:
        def __init__(self, *_args, **_kwargs):
            constructions.append(1)

    monkeypatch.setattr(docent_export, "_DocentHTTPTransport", ForbiddenTransport)
    monkeypatch.setattr(docent_export, "version", lambda _name: "9.9.9")
    result = docent_export._upload_in_child_process(
        agent_runs(RECORDS[:1]),
        "math-pc-rl--launch-run-A",
        "run-A",
        _budget=docent_export._DocentTimeBudget(
            deadline=time.monotonic() + 10,
            clock=time.monotonic,
            sleeper=lambda _seconds: None,
        ),
        on_collection_confirmed=lambda _value: pytest.fail(
            "drift must not mutate"
        ),
    )
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id is None
    assert constructions == []


def test_sdk_serializer_drift_refuses_before_transport_or_mutation(monkeypatch):
    from docent.sdk import _client_util

    constructions = []

    class ForbiddenTransport:
        def __init__(self, *_args, **_kwargs):
            constructions.append(1)

    monkeypatch.setattr(docent_export, "_DocentHTTPTransport", ForbiddenTransport)
    monkeypatch.setattr(
        _client_util,
        "serialize_agent_run",
        lambda _run: b"unreviewed",
    )
    result = docent_export._upload_in_child_process(
        agent_runs(RECORDS[:1]),
        "math-pc-rl--launch-run-A",
        "run-A",
        _budget=docent_export._DocentTimeBudget(
            deadline=time.monotonic() + 10,
            clock=time.monotonic,
            sleeper=lambda _seconds: None,
        ),
        on_collection_confirmed=lambda _value: pytest.fail(
            "drift must not mutate"
        ),
    )
    assert isinstance(result, DocentUploadFailure)
    assert result.collection_id is None
    assert result.error_type == "RuntimeError"
    assert constructions == []


def test_public_deadline_default_is_fixed_five_minutes():
    parameter = inspect.signature(upload).parameters["_deadline_seconds"]
    assert parameter.default == 300.0
