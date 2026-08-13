"""Process-isolated adapter for the pinned symbolic-math verifier.

Only extracted LaTeX expressions cross this boundary.  In particular, the
worker never receives model prose and SymPy objects never leave the child
process.  A worker is started lazily because most task families do not need
symbolic grading and because task sources must be safe to construct before a
runner forks or spawns its own processes.
"""

from __future__ import annotations

import importlib.metadata
import json
import multiprocessing
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Optional

from infra.envs.tasks.base import GraderInfrastructureError


_PROTOCOL_VERSION = 2
MATH_VERIFY_WORKER_PROTOCOL = "math-verify-worker-v2"
CANONICALIZATION_PROTOCOL = "math-symbolic-exact-canonicalization-v1"
_MAX_EXPRESSION_BYTES = 64 * 1024
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_GOLD_CACHE_SIZE = 256
_MAX_BATCH_ITEMS = 1
_OPERATION_TIMEOUT_SECONDS = 5
_PARENT_TIMEOUT_SECONDS = 20.0
_SHUTDOWN_TIMEOUT_SECONDS = 1.0

_EXPECTED_VERSIONS = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "sympy": "1.14.0",
    "antlr4-python3-runtime": "4.13.2",
}

EXACT_CANONICALIZATIONS: tuple[tuple[str, str], ...] = (
    (r"\approx 8.24 \text{ mph}", "8.24"),
    ("a + b + c", "a+b+c"),
    (r"\sin^2 t", r"\sin^2{t}"),
)
_EXACT_CANONICALIZATION_MAP = dict(EXACT_CANONICALIZATIONS)


class MathVerifyDataError(GraderInfrastructureError):
    """The benchmark gold answer violates the symbolic grading protocol."""


class _RetryableInfrastructureError(Exception):
    """An unhealthy worker/request which must be retried on a fresh process."""


def canonicalize_math_expression(expression: str) -> str:
    """Apply the protocol's three exact corpus canonicalizations.

    Surrounding whitespace is insignificant.  No substring or pattern rewrite
    is performed: near-matches remain untouched and are left to Math-Verify.
    This function is applied inside the child to gold and candidate expressions
    alike.
    """
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    stripped = expression.strip()
    return _EXACT_CANONICALIZATION_MAP.get(stripped, stripped)


def _json_bytes(message: dict[str, Any]) -> bytes:
    try:
        payload = json.dumps(
            message,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _RetryableInfrastructureError("message is not valid JSON") from exc
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Math-Verify request exceeds {_MAX_MESSAGE_BYTES} encoded bytes"
        )
    return payload


def _decode_message(payload: bytes) -> dict[str, Any]:
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise _RetryableInfrastructureError("worker message exceeds size limit")
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _RetryableInfrastructureError("worker sent malformed JSON") from exc
    if not isinstance(message, dict):
        raise _RetryableInfrastructureError("worker message is not an object")
    return message


def _send_message(conn: Any, message: dict[str, Any]) -> None:
    conn.send_bytes(_json_bytes(message))


def _recv_message(conn: Any) -> dict[str, Any]:
    try:
        payload = conn.recv_bytes(_MAX_MESSAGE_BYTES + 1)
    except (EOFError, OSError) as exc:
        raise _RetryableInfrastructureError("worker pipe closed") from exc
    return _decode_message(payload)


def _connection_send_bytes(conn: Any, payload: bytes) -> None:
    """Parent send seam used by deterministic timeout/race tests."""
    conn.send_bytes(payload)


def _expression_bytes(value: str, *, field: str) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    size = len(value.encode("utf-8"))
    if size > _MAX_EXPRESSION_BYTES:
        raise ValueError(
            f"{field} exceeds the {_MAX_EXPRESSION_BYTES}-byte Math-Verify limit"
        )
    return size


def _latex_envelope(expression: str) -> str:
    """Give LatexExtractionConfig one explicit, synthetic extraction target."""
    # A plain math delimiter is the envelope.  Adding a synthetic ``\boxed``
    # would let malformed content close that box and inject a second, valid box.
    # Unescaped dollar signs are delimiters to Math-Verify, not mathematical
    # content for this extractor-owned expression protocol; reject them rather
    # than letting an expression close the envelope and inject a later target.
    # LaTeX's escaped ``\$`` is content (notably the pinned corpus's currency
    # answers), so an odd run of preceding backslashes is allowed.
    for index, char in enumerate(expression):
        if char != "$":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and expression[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            raise ValueError("extracted expression contains a math delimiter")
    return f"${expression}$"


def _load_math_verify() -> tuple[Callable[..., Any], Callable[..., Any], Any, type[BaseException]]:
    """Import and configure the exact protocol implementation in the child."""
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException

    normalization = NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=True,
        nits=False,
        boxed="all",
        equations=False,
    )
    extraction = LatexExtractionConfig(normalization_config=normalization)
    return parse, verify, extraction, TimeoutException


def _parse_one(
    expression: str,
    *,
    parse: Callable[..., Any],
    extraction: Any,
) -> Any:
    parsed = parse(
        _latex_envelope(canonicalize_math_expression(expression)),
        extraction_config=[extraction],
        fallback_mode="no_fallback",
        raise_on_error=True,
        parsing_timeout=_OPERATION_TIMEOUT_SECONDS,
    )
    if not isinstance(parsed, list) or len(parsed) != 1 or parsed[0] is None:
        raise ValueError("Math-Verify did not produce exactly one parsed scalar")
    return parsed[0]


def _worker_response(request_id: int, status: str, result: Any = None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "protocol": _PROTOCOL_VERSION,
        "kind": "response",
        "id": request_id,
        "status": status,
    }
    if status == "ok":
        response["result"] = result
    return response


def _math_verify_worker_main(conn: Any) -> None:
    """Spawn target.  Keep imports, parsed values, and caches child-local."""
    try:
        versions = {
            distribution: importlib.metadata.version(distribution)
            for distribution in _EXPECTED_VERSIONS
        }
        parse, verify, extraction, timeout_exception = _load_math_verify()
        _send_message(
            conn,
            {
                "protocol": _PROTOCOL_VERSION,
                "kind": "hello",
                "pid": os.getpid(),
                "versions": versions,
            },
        )
    except BaseException as exc:  # startup must fail closed, including import aborts
        try:
            _send_message(
                conn,
                {
                    "protocol": _PROTOCOL_VERSION,
                    "kind": "startup_error",
                    "error_type": type(exc).__name__,
                },
            )
        except BaseException:
            pass
        conn.close()
        return

    gold_cache: OrderedDict[str, Any] = OrderedDict()

    def parse_gold(gold: str) -> Any:
        if gold in gold_cache:
            value = gold_cache.pop(gold)
            gold_cache[gold] = value
            return value
        value = _parse_one(gold, parse=parse, extraction=extraction)
        gold_cache[gold] = value
        if len(gold_cache) > _GOLD_CACHE_SIZE:
            gold_cache.popitem(last=False)
        return value

    def parse_candidate(candidate: str) -> tuple[bool, Any]:
        try:
            return True, _parse_one(candidate, parse=parse, extraction=extraction)
        except timeout_exception:
            raise
        except Exception:  # malformed candidate is data, not worker failure
            return False, None

    def grade_one(gold: str, candidate: str) -> Optional[bool]:
        # Gold is parsed first: a malformed benchmark must never be disguised as
        # a malformed model answer.
        try:
            parsed_gold = parse_gold(gold)
        except timeout_exception:
            raise
        except Exception as exc:
            raise MathVerifyDataError("gold expression is not parseable") from exc

        candidate_ok, parsed_candidate = parse_candidate(candidate)
        if not candidate_ok:
            return None

        def verify_values(target: Any) -> bool:
            return bool(
                verify(
                    parsed_gold,
                    target,
                    strict=True,
                    allow_set_relation_comp=False,
                    float_rounding=6,
                    numeric_precision=15,
                    timeout_seconds=_OPERATION_TIMEOUT_SECONDS,
                    raise_on_error=True,
                )
            )

        try:
            return verify_values(parsed_candidate)
        except timeout_exception:
            raise
        except Exception as candidate_error:
            # Only downgrade a verifier error when the exact same verifier can
            # still validate the gold against itself.  Otherwise the benchmark
            # or verifier protocol is invalid, rather than this candidate being
            # merely ungradeable.
            try:
                gold_is_valid = verify_values(parsed_gold)
            except timeout_exception:
                raise
            except Exception as gold_error:
                raise MathVerifyDataError(
                    "Math-Verify cannot validate benchmark gold"
                ) from gold_error
            if not gold_is_valid:
                raise MathVerifyDataError(
                    "Math-Verify rejects benchmark gold against itself"
                ) from candidate_error
            return None

    try:
        while True:
            try:
                request = _recv_message(conn)
            except _RetryableInfrastructureError:
                return
            if (
                request.get("protocol") != _PROTOCOL_VERSION
                or request.get("kind") != "request"
                or not isinstance(request.get("id"), int)
                or not isinstance(request.get("op"), str)
            ):
                return
            request_id = request["id"]
            operation = request["op"]
            if operation == "shutdown":
                _send_message(conn, _worker_response(request_id, "ok", True))
                return
            try:
                if operation == "is_parseable":
                    ok, _ = parse_candidate(request["candidate"])
                    result: Any = ok
                elif operation == "grade":
                    result = grade_one(request["gold"], request["candidate"])
                elif operation == "grade_many":
                    items = request["items"]
                    if not isinstance(items, list):
                        return
                    result = [grade_one(item[0], item[1]) for item in items]
                else:
                    return
                response = _worker_response(request_id, "ok", result)
            except timeout_exception:
                response = _worker_response(request_id, "operation_timeout")
            except MathVerifyDataError:
                response = _worker_response(request_id, "gold_error")
            except (KeyError, IndexError, TypeError, ValueError):
                # Parent validation means this is a malformed wire request.
                return
            _send_message(conn, response)
    finally:
        conn.close()


class MathVerifyWorker:
    """Lazy, restartable, thread-safe owner of one Math-Verify subprocess."""

    def __init__(
        self,
        *,
        _target: Callable[..., None] = _math_verify_worker_main,
        _target_args: Sequence[Any] = (),
        _parent_timeout: float = _PARENT_TIMEOUT_SECONDS,
        _startup_timeout: Optional[float] = None,
        _shutdown_timeout: float = _SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        self._target = _target
        self._target_args = tuple(_target_args)
        self._parent_timeout = float(_parent_timeout)
        self._startup_timeout = float(
            _parent_timeout if _startup_timeout is None else _startup_timeout
        )
        self._shutdown_timeout = float(_shutdown_timeout)
        self._ctx = multiprocessing.get_context("spawn")
        self._owner_pid = os.getpid()
        self._process: Any = None
        self._conn: Any = None
        self._lock = threading.RLock()
        self._next_request_id = 1
        self._send_bytes = _connection_send_bytes

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ctx"] = None
        state["_process"] = None
        state["_conn"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._ctx = multiprocessing.get_context("spawn")
        self._owner_pid = os.getpid()
        self._process = None
        self._conn = None
        self._lock = threading.RLock()

    def _adopt_current_process(self) -> None:
        if self._owner_pid == os.getpid():
            return
        # These are inherited duplicates.  Killing the recorded process here
        # would kill the original owner's healthy worker.
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._process = None
        self._conn = None
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()

    def _serialized(self) -> Any:
        """Return the current-process lock, adopting safely before exposure."""
        while True:
            # Check before acquiring: after fork an inherited lock may record a
            # vanished owning thread and can never be acquired in this child.
            if self._owner_pid != os.getpid():
                self._adopt_current_process()
                continue
            lock = self._lock
            lock.acquire()
            if self._owner_pid == os.getpid() and lock is self._lock:
                return lock
            lock.release()
            self._adopt_current_process()

    def _await_message(
        self,
        deadline_seconds: float,
        *,
        conn: Any,
        process: Any,
    ) -> dict[str, Any]:
        if conn is None or process is None:
            raise _RetryableInfrastructureError("worker is not running")
        deadline = time.monotonic() + deadline_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _RetryableInfrastructureError("worker operation timed out")
            try:
                ready = conn.poll(min(remaining, 0.05))
            except (EOFError, OSError) as exc:
                raise _RetryableInfrastructureError("worker pipe failed") from exc
            if ready:
                return _recv_message(conn)
            if not process.is_alive():
                raise _RetryableInfrastructureError("worker exited")

    def _start(self) -> None:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=self._target,
            args=(child_conn, *self._target_args),
            daemon=True,
            name="math-verify-worker",
        )
        self._conn = parent_conn
        self._process = process
        try:
            process.start()
            child_conn.close()
            hello = self._await_message(
                self._startup_timeout,
                conn=parent_conn,
                process=process,
            )
            if hello.get("kind") == "startup_error":
                error_type = hello.get("error_type")
                raise _RetryableInfrastructureError(
                    f"Math-Verify worker startup failed ({error_type})"
                )
            if (
                hello.get("protocol") != _PROTOCOL_VERSION
                or hello.get("kind") != "hello"
                or not isinstance(hello.get("pid"), int)
                or hello.get("pid") != process.pid
                or hello.get("versions") != _EXPECTED_VERSIONS
            ):
                raise _RetryableInfrastructureError("worker handshake mismatch")
        except BaseException as exc:
            try:
                child_conn.close()
            except OSError:
                pass
            self._discard_worker(terminate=True)
            if isinstance(exc, _RetryableInfrastructureError):
                raise
            if isinstance(exc, Exception):
                raise _RetryableInfrastructureError(
                    "could not start Math-Verify worker"
                ) from exc
            raise

    def _discard_worker(self, *, terminate: bool) -> None:
        conn, process = self._conn, self._process
        self._conn = None
        self._process = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is None:
            return
        try:
            process.join(timeout=0)
        except (AssertionError, ValueError):
            return
        if terminate and process.is_alive():
            process.terminate()
            process.join(timeout=self._shutdown_timeout)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=self._shutdown_timeout)
        else:
            process.join(timeout=self._shutdown_timeout)

    def _request_once(self, request: dict[str, Any]) -> Any:
        if self._process is None or self._conn is None or not self._process.is_alive():
            self._discard_worker(terminate=True)
            self._start()
        request_id = self._next_request_id
        self._next_request_id += 1
        wire_request = {
            "protocol": _PROTOCOL_VERSION,
            "kind": "request",
            "id": request_id,
            **request,
        }
        payload = _json_bytes(wire_request)
        try:
            assert self._conn is not None
            assert self._process is not None
            # Bind the whole request to one immutable worker generation.  A
            # timed-out daemon sender may resume only after retry has installed
            # a new self._conn; it must still touch solely this old endpoint.
            conn = self._conn
            process = self._process
            send_bytes = self._send_bytes
            started = time.monotonic()
            failure: list[BaseException] = []

            def send() -> None:
                try:
                    send_bytes(conn, payload)
                except BaseException as exc:
                    failure.append(exc)

            send_thread = threading.Thread(
                target=send,
                name="math-verify-pipe-send",
                daemon=True,
            )
            send_thread.start()
            send_thread.join(timeout=self._parent_timeout)
            if send_thread.is_alive():
                raise _RetryableInfrastructureError("worker request send timed out")
            if failure:
                raise _RetryableInfrastructureError("worker pipe send failed") from failure[0]
            remaining = self._parent_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise _RetryableInfrastructureError("worker operation timed out")
            response = self._await_message(
                remaining,
                conn=conn,
                process=process,
            )
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise _RetryableInfrastructureError("worker pipe failed") from exc
        if (
            response.get("protocol") != _PROTOCOL_VERSION
            or response.get("kind") != "response"
            or response.get("id") != request_id
        ):
            raise _RetryableInfrastructureError("worker response mismatch")
        status = response.get("status")
        if status == "gold_error":
            raise MathVerifyDataError("Math-Verify could not parse benchmark gold")
        if status == "operation_timeout":
            raise _RetryableInfrastructureError("Math-Verify operation timed out")
        if status != "ok" or "result" not in response:
            raise _RetryableInfrastructureError("worker response has invalid status")
        return response["result"]

    def _request_locked(
        self,
        request: dict[str, Any],
        validator: Optional[Callable[[Any], Any]] = None,
    ) -> Any:
        failures: list[str] = []
        for _attempt in range(2):
            try:
                result = self._request_once(request)
                return validator(result) if validator is not None else result
            except MathVerifyDataError:
                raise
            except _RetryableInfrastructureError as exc:
                failures.append(str(exc))
                self._discard_worker(terminate=True)
        detail = failures[-1] if failures else "unknown worker failure"
        raise GraderInfrastructureError(
            f"Math-Verify worker failed twice: {detail}"
        )

    def is_parseable(self, candidate: str) -> bool:
        _expression_bytes(candidate, field="candidate")

        def validate(result: Any) -> bool:
            if type(result) is not bool:
                raise _RetryableInfrastructureError(
                    "Math-Verify returned invalid parseability"
                )
            return result

        lock = self._serialized()
        try:
            result = self._request_locked(
                {"op": "is_parseable", "candidate": candidate}, validate
            )
        finally:
            lock.release()
        return result

    def grade(self, gold: str, candidate: str) -> Optional[bool]:
        _expression_bytes(gold, field="gold")
        _expression_bytes(candidate, field="candidate")

        def validate(result: Any) -> Optional[bool]:
            if result is not None and type(result) is not bool:
                raise _RetryableInfrastructureError(
                    "Math-Verify returned invalid grade"
                )
            return result

        lock = self._serialized()
        try:
            result = self._request_locked(
                {"op": "grade", "gold": gold, "candidate": candidate}, validate
            )
        finally:
            lock.release()
        return result

    def grade_many(
        self, items: list[tuple[str, str]]
    ) -> list[Optional[bool]]:
        validated: list[list[str]] = []
        for index, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(f"items[{index}] must be a (gold, candidate) tuple")
            gold, candidate = item
            _expression_bytes(gold, field=f"items[{index}].gold")
            _expression_bytes(candidate, field=f"items[{index}].candidate")
            validated.append([gold, candidate])
        if not validated:
            return []

        # Bound every wire message as well as each expression.  Keep one lock
        # across all chunks so concurrent callers cannot interleave batches.
        chunks: list[list[list[str]]] = []
        chunk: list[list[str]] = []
        for item in validated:
            trial = chunk + [item]
            probe = {
                "protocol": _PROTOCOL_VERSION,
                "kind": "request",
                "id": self._next_request_id,
                "op": "grade_many",
                "items": trial,
            }
            try:
                _json_bytes(probe)
            except ValueError:
                if not chunk:
                    raise
                chunks.append(chunk)
                chunk = [item]
            else:
                chunk = trial
            if len(chunk) >= _MAX_BATCH_ITEMS:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)

        results: list[Optional[bool]] = []
        lock = self._serialized()
        try:
            for current in chunks:
                def validate(batch: Any) -> list[Optional[bool]]:
                    if (
                        not isinstance(batch, list)
                        or len(batch) != len(current)
                        or any(
                            value is not None and type(value) is not bool
                            for value in batch
                        )
                    ):
                        raise _RetryableInfrastructureError(
                            "Math-Verify returned an invalid batch grade"
                        )
                    return batch

                batch = self._request_locked(
                    {"op": "grade_many", "items": current}, validate
                )
                results.extend(batch)
        finally:
            lock.release()
        return results

    def close(self) -> None:
        lock = self._serialized()
        try:
            if self._process is None or self._conn is None:
                self._discard_worker(terminate=True)
                return
            process = self._process
            try:
                request_id = self._next_request_id
                self._next_request_id += 1
                _send_message(
                    self._conn,
                    {
                        "protocol": _PROTOCOL_VERSION,
                        "kind": "request",
                        "id": request_id,
                        "op": "shutdown",
                    },
                )
                if self._conn.poll(self._shutdown_timeout):
                    response = _recv_message(self._conn)
                    if response.get("id") != request_id:
                        raise _RetryableInfrastructureError(
                            "shutdown response mismatch"
                        )
                process.join(timeout=self._shutdown_timeout)
            except (BaseException,):
                # close() is cleanup; it must remain idempotent even when the
                # child is already corrupt or gone.
                pass
            finally:
                self._discard_worker(terminate=True)
        finally:
            lock.release()


__all__ = [
    "MathVerifyDataError",
    "MathVerifyWorker",
    "MATH_VERIFY_WORKER_PROTOCOL",
    "CANONICALIZATION_PROTOCOL",
    "EXACT_CANONICALIZATIONS",
    "canonicalize_math_expression",
]
