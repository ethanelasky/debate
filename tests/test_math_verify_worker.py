from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infra.envs.tasks.base import GraderInfrastructureError
from infra.envs.tasks.math_verify_worker import (
    CANONICALIZATION_PROTOCOL,
    EXACT_CANONICALIZATIONS,
    MATH_VERIFY_WORKER_PROTOCOL,
    MathVerifyDataError,
    MathVerifyWorker,
    _EXPECTED_VERSIONS,
    _MAX_EXPRESSION_BYTES,
    _PROTOCOL_VERSION,
    _latex_envelope,
    canonicalize_math_expression,
)


def _fake_send(conn, message: dict) -> None:
    conn.send_bytes(
        json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    )


def _fake_hello(conn, *, versions: dict[str, str] | None = None) -> None:
    _fake_send(
        conn,
        {
            "protocol": _PROTOCOL_VERSION,
            "kind": "hello",
            "pid": os.getpid(),
            "versions": _EXPECTED_VERSIONS if versions is None else versions,
        },
    )


def _fake_result(request: dict):
    operation = request["op"]
    if operation == "is_parseable":
        return request["candidate"] != "bad"
    if operation == "grade":
        if request["candidate"] == "bad":
            return None
        return request["gold"] == request["candidate"]
    if operation == "grade_many":
        return [
            None if candidate == "bad" else gold == candidate
            for gold, candidate in request["items"]
        ]
    if operation == "shutdown":
        return True
    raise AssertionError(operation)


def _fake_worker(conn, mode: str = "echo", launches=None) -> None:
    if launches is not None:
        with launches.get_lock():
            launches.value += 1
            launch_number = launches.value
    else:
        launch_number = 1
    if mode == "bad_handshake":
        _fake_hello(conn, versions={**_EXPECTED_VERSIONS, "sympy": "0"})
        time.sleep(1)
        return
    if mode == "startup_error":
        _fake_send(
            conn,
            {
                "protocol": _PROTOCOL_VERSION,
                "kind": "startup_error",
                "error_type": "MissingVerifier",
            },
        )
        return
    _fake_hello(conn)
    while True:
        if mode == "never_read":
            time.sleep(10)
            continue
        try:
            request = json.loads(conn.recv_bytes().decode())
        except (EOFError, OSError):
            return
        operation = request["op"]
        if mode == "crash_always" or (mode == "crash_once" and launch_number == 1):
            os._exit(17)
        if mode == "sleep":
            time.sleep(0.3)
        if mode == "stale_id":
            request_id = request["id"] + 1
        else:
            request_id = request["id"]
        if mode == "malformed":
            conn.send_bytes(b"not-json")
            continue
        if mode == "gold_error":
            _fake_send(
                conn,
                {
                    "protocol": _PROTOCOL_VERSION,
                    "kind": "response",
                    "id": request_id,
                    "status": "gold_error",
                },
            )
            continue
        result = "invalid" if mode == "invalid_result" else _fake_result(request)
        _fake_send(
            conn,
            {
                "protocol": _PROTOCOL_VERSION,
                "kind": "response",
                "id": request_id,
                "status": "ok",
                "result": result,
            },
        )
        if operation == "shutdown":
            return


def _fork_use_worker(worker: MathVerifyWorker, queue) -> None:
    try:
        result = worker.is_parseable("child")
        queue.put((result, worker._process.pid))
    finally:
        worker.close()


def _fork_use_worker_while_parent_lock_held(worker: MathVerifyWorker, queue) -> None:
    try:
        result = worker.is_parseable("child-after-locked-fork")
        queue.put((result, worker._process.pid))
    finally:
        worker.close()


def _worker(mode: str = "echo", *, launches=None, timeout: float = 1.0):
    args = (mode,) if launches is None else (mode, launches)
    return MathVerifyWorker(
        _target=_fake_worker,
        _target_args=args,
        _parent_timeout=timeout,
        _startup_timeout=2.0,
        _shutdown_timeout=0.1,
    )


def test_fake_worker_public_api_and_lazy_start():
    worker = _worker()
    assert worker._process is None
    assert worker.is_parseable("x") is True
    assert worker.is_parseable("bad") is False
    assert worker.grade("x", "x") is True
    assert worker.grade("x", "y") is False
    assert worker.grade("x", "bad") is None
    assert worker.grade_many([("x", "x"), ("x", "y"), ("x", "bad")]) == [
        True,
        False,
        None,
    ]
    worker.close()


def test_empty_batch_does_not_start_worker():
    worker = _worker()
    assert worker.grade_many([]) == []
    assert worker._process is None
    worker.close()


def test_worker_restarts_on_first_crash_with_a_fresh_process():
    ctx = multiprocessing.get_context("spawn")
    launches = ctx.Value("i", 0)
    worker = _worker("crash_once", launches=launches)
    assert worker.grade("x", "x") is True
    assert launches.value == 2
    worker.close()


@pytest.mark.parametrize(
    "mode, message",
    [
        ("crash_always", "failed twice"),
        ("sleep", "failed twice"),
        ("stale_id", "failed twice"),
        ("malformed", "failed twice"),
        ("invalid_result", "failed twice"),
        ("startup_error", "startup failed"),
        ("bad_handshake", "failed twice"),
    ],
)
def test_repeated_transport_or_protocol_failure_is_fatal(mode, message):
    ctx = multiprocessing.get_context("spawn")
    launches = ctx.Value("i", 0)
    worker = _worker(mode, launches=launches, timeout=0.05 if mode == "sleep" else 0.3)
    with pytest.raises(GraderInfrastructureError, match=message):
        worker.is_parseable("x")
    assert launches.value == 2
    worker.close()


def test_pipe_write_itself_is_covered_by_parent_deadline():
    ctx = multiprocessing.get_context("spawn")
    launches = ctx.Value("i", 0)
    worker = _worker("never_read", launches=launches, timeout=0.05)
    # Far larger than a typical pipe buffer, but below the public input bound.
    with pytest.raises(GraderInfrastructureError, match="send timed out"):
        worker.is_parseable("x" * 60_000)
    assert launches.value == 2
    worker.close()


def test_gold_protocol_error_is_fatal_without_transport_retry():
    ctx = multiprocessing.get_context("spawn")
    launches = ctx.Value("i", 0)
    worker = _worker("gold_error", launches=launches)
    with pytest.raises(MathVerifyDataError, match="benchmark gold"):
        worker.grade("bad-gold", "x")
    assert launches.value == 1
    worker.close()


def test_expression_inputs_are_individually_bounded_before_spawn():
    worker = _worker()
    too_large = "x" * (_MAX_EXPRESSION_BYTES + 1)
    with pytest.raises(ValueError, match="65536-byte"):
        worker.is_parseable(too_large)
    with pytest.raises(ValueError, match=r"items\[0\].gold"):
        worker.grade_many([(too_large, "x")])
    with pytest.raises(TypeError, match="candidate must be a string"):
        worker.grade("x", None)  # type: ignore[arg-type]
    assert worker._process is None


def test_calls_from_many_threads_remain_aligned_and_serialized():
    worker = _worker()
    pairs = [(str(index), str(index if index % 2 == 0 else -index)) for index in range(30)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda pair: worker.grade(*pair), pairs))
    assert results == [index % 2 == 0 for index in range(30)]
    worker.close()


def test_close_is_idempotent_and_reaps_the_process():
    worker = _worker()
    assert worker.is_parseable("x")
    process = worker._process
    assert process.is_alive()
    worker.close()
    process.join(timeout=1)
    assert not process.is_alive()
    assert worker._process is None
    assert worker._conn is None
    worker.close()


def test_pickle_drops_live_process_and_pipe_handles():
    worker = _worker()
    assert worker.is_parseable("parent")
    parent_process = worker._process
    restored = pickle.loads(pickle.dumps(worker))
    assert restored._process is None
    assert restored._conn is None
    assert restored.is_parseable("restored")
    assert restored._process.pid != parent_process.pid
    restored.close()
    worker.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_pid_change_drops_inherited_handles_without_killing_parent_worker():
    worker = _worker()
    assert worker.is_parseable("parent")
    parent_worker = worker._process

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_fork_use_worker, args=(worker, queue))
    child.start()
    child.join(timeout=5)
    assert child.exitcode == 0
    result, child_worker_pid = queue.get(timeout=1)
    assert result is True
    assert child_worker_pid != parent_worker.pid
    assert parent_worker.is_alive()
    assert worker.is_parseable("still-parent")
    worker.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_pid_change_replaces_a_lock_held_by_a_vanished_parent_thread():
    worker = _worker()
    assert worker.is_parseable("parent")
    parent_worker = worker._process
    entered = threading.Event()
    release = threading.Event()

    def hold_lock():
        with worker._lock:
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert entered.wait(timeout=1)
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(
        target=_fork_use_worker_while_parent_lock_held,
        args=(worker, queue),
    )
    child.start()
    child.join(timeout=5)
    release.set()
    holder.join(timeout=1)
    assert child.exitcode == 0
    result, child_worker_pid = queue.get(timeout=1)
    assert result is True
    assert child_worker_pid != parent_worker.pid
    assert parent_worker.is_alive()
    worker.close()


def test_source_does_not_enable_free_form_expression_parsing():
    source = (
        Path(__file__).parents[1] / "infra/envs/tasks/math_verify_worker.py"
    ).read_text()
    banned = ("Expr" + "ExtractionConfig", "parse" + "_expr")
    assert all(name not in source for name in banned)
    assert "LatexExtractionConfig" in source
    assert 'fallback_mode="no_fallback"' in source
    assert "malformed_operators=True" in source


def test_public_protocol_constants_are_immutable_and_versioned():
    assert MATH_VERIFY_WORKER_PROTOCOL == "math-verify-worker-v2"
    assert CANONICALIZATION_PROTOCOL == "math-symbolic-exact-canonicalization-v1"
    assert EXACT_CANONICALIZATIONS == (
        (r"\approx 8.24 \text{ mph}", "8.24"),
        ("a + b + c", "a+b+c"),
        (r"\sin^2 t", r"\sin^2{t}"),
    )


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        (r"\approx 8.24 \text{ mph}", "8.24"),
        ("a + b + c", "a+b+c"),
        (r"\sin^2 t", r"\sin^2{t}"),
    ],
)
def test_exact_corpus_canonicalizations_only(raw, canonical):
    assert canonicalize_math_expression(raw) == canonical
    assert canonicalize_math_expression(f" \n{raw}\t ") == canonical


@pytest.mark.parametrize(
    "near_match",
    [
        r"\approx 8.25 \text{ mph}",
        r"x + a + b + c",
        r"\sin^2 x",
    ],
)
def test_canonicalizations_do_not_generalize_to_near_matches(near_match):
    assert canonicalize_math_expression(near_match) == near_match


def test_malformed_expression_cannot_escape_the_synthetic_envelope():
    required_modules = ("math_verify", "latex2sympy2_extended", "antlr4")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip("real Math-Verify stack unavailable: " + ", ".join(missing))
    worker = MathVerifyWorker()
    try:
        assert worker.is_parseable(r"2}\boxed{2") is False
        assert worker.grade("2", r"2}\boxed{2") is None
        for injected in (r"2$ blah $3", r"2$$3", r"2$\boxed{3}$"):
            assert worker.is_parseable(injected) is False
            assert worker.grade("3", injected) is None
        # An escaped dollar is LaTeX content, not an envelope delimiter.  Even
        # preceding backslash parity remains a delimiter; odd parity remains
        # escaped.  A later unescaped dollar still poisons the whole candidate.
        assert worker.is_parseable(r"\$15.48") is True
        assert worker.grade("15.48", r"\$15.48") is True
        with pytest.raises(ValueError, match="math delimiter"):
            _latex_envelope("\\\\$15.48")
        assert _latex_envelope("\\\\\\$15.48") == "$\\\\\\$15.48$"
        # Passing the guard does not imply the resulting LaTeX is meaningful.
        assert worker.is_parseable("\\\\\\$15.48") is False
        assert worker.is_parseable(r"\$15.48$ blah $15.48") is False
        assert worker.grade("15.48", r"\$15.48$ blah $15.48") is None
    finally:
        worker.close()


def test_real_pinned_math_verify_stack_if_installed():
    required_modules = ("math_verify", "latex2sympy2_extended", "antlr4")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip("real Math-Verify stack unavailable: " + ", ".join(missing))
    assert {
        distribution: importlib.metadata.version(distribution)
        for distribution in _EXPECTED_VERSIONS
    } == _EXPECTED_VERSIONS

    worker = MathVerifyWorker()
    try:
        assert worker.is_parseable(r"\frac{1}{2}") is True
        assert worker.grade(r"\frac{1}{2}", r"\frac{2}{4}") is True
        assert worker.grade(r"\frac{1}{2}", r"\frac{3}{4}") is False
        for shorthand, expanded in [
            (r"-\frac12", r"-\frac{1}{2}"),
            (r"\frac{29}4", r"\frac{29}{4}"),
            (r"\tfrac34", r"\frac{3}{4}"),
            (r"\sqrt 2+\sqrt 5", r"\sqrt{2}+\sqrt{5}"),
            (r"4+2\sqrt3", r"4+2\sqrt{3}"),
        ]:
            assert worker.is_parseable(shorthand) is True
            assert worker.grade(expanded, shorthand) is True
            assert worker.grade(shorthand, expanded) is True

        for raw, canonical in [
            (r"\approx 8.24 \text{ mph}", "8.24"),
            ("a + b + c", "a+b+c"),
            (r"\sin^2 t", r"\sin^2{t}"),
        ]:
            assert worker.grade(raw, canonical) is True
            assert worker.grade(canonical, raw) is True

        assert worker.grade(r"\approx 8.24 \text{ mph}", "8.25") is False
        assert worker.grade("a + b + c", "a+b+d") is False
        assert worker.grade(r"\sin^2 t", r"\sin^2{x}") is False
        assert worker.grade_many(
            [("2", "1+1"), ("2", "3"), ("2", r"\frac{")]
        ) == [True, False, None]
        with pytest.raises(MathVerifyDataError):
            worker.grade(r"\frac{", "1")
    finally:
        worker.close()
