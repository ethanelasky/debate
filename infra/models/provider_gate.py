"""Process-wide per-provider concurrency / deadline / retry gate.

Salvaged from ~/ai-debate/ai_infra/models/provider_gate.py, slimmed. One
registry keyed by provider name replaces the hand-rolled per-wrapper
ThreadPoolExecutors, backoff decorators, and the module-global DashScope rate
limiter. Wrappers call ``run_batched_predict`` for batch fan-out and
``call_with_retry`` around the raw API call, supplying their own retryability
predicates; only the budgets come from here. Dropped: token-fidelity hooks,
provenance/metrics callbacks.

The defaults reproduce the old wrappers' effective limits: fan-out widths per
provider, unbounded process-wide concurrency (the old executors were built per
predict() call, so they never capped anything across rounds), no driver-level
deadline (timeouts lived in the HTTP clients and stay there), and each
wrapper's backoff budget verbatim. SDK-internal retries/timeouts are
deliberately not modeled here.

Thread-safety: plain ``threading`` primitives throughout. No asyncio.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

import backoff


class ProviderGateDeadlineExceeded(Exception):
    """Raised when a gated call exceeds its hard per-call deadline."""


@dataclass(frozen=True)
class GateConfig:
    """Per-provider gate parameters.

    max_concurrency=None is unbounded. When it is finite, note that a
    deadline-abandoned call keeps holding its slot until the underlying call
    actually returns (a sync call cannot be cancelled), so the cap needs
    straggler headroom and every gated call needs its own inner timeout.

    call_deadline_s starts counting AFTER the concurrency slot is acquired, so
    it bounds call latency, not queue+call latency.

    backoff_factor / backoff_max_value of None mean "don't pass the kwarg", so
    backoff.expo's own defaults apply (factor=1, no ceiling).

    min_interval_s spaces attempt starts across all threads for the provider
    (token-bucket of one); 0.0 disables.
    """

    batch_fanout: int = 16
    max_concurrency: Optional[int] = None
    call_deadline_s: Optional[float] = None
    max_tries: int = 1
    backoff_factor: Optional[float] = None
    backoff_max_value: Optional[float] = None
    min_interval_s: float = 0.0


#: Pinned to the pre-refactor wrappers' live behavior; re-derive before editing.
DEFAULT_GATE_CONFIGS: dict[str, GateConfig] = {
    "openai": GateConfig(batch_fanout=16, max_tries=4, backoff_factor=2),
    "anthropic": GateConfig(batch_fanout=16, max_tries=1),
    "google": GateConfig(batch_fanout=8, max_tries=4, backoff_factor=2),
    "tinker": GateConfig(batch_fanout=64, max_tries=4, backoff_factor=None),
    "alibaba": GateConfig(
        batch_fanout=16,
        max_tries=6,
        backoff_factor=5,
        backoff_max_value=60,
        min_interval_s=0.1,  # DashScope ~10 req/s, global, per attempt
    ),
    "openrouter": GateConfig(
        batch_fanout=16, max_tries=6, backoff_factor=4, backoff_max_value=60
    ),
    "fireworks": GateConfig(batch_fanout=16, max_tries=1),
    "local": GateConfig(batch_fanout=16, max_tries=1),
}

#: Auto-registered for provider names not in the table: no retry, no cap.
_UNKNOWN_PROVIDER_CONFIG = GateConfig(batch_fanout=16, max_tries=1)


class _GateEntry:
    """Runtime state for one provider: semaphore + rate-slot bookkeeping."""

    def __init__(self, provider: str, config: GateConfig):
        self.provider = provider
        self.config = config
        self.semaphore: Optional[threading.Semaphore] = (
            threading.Semaphore(config.max_concurrency)
            if config.max_concurrency is not None
            else None
        )
        self._rate_lock = threading.Lock()
        self._rate_next_fire = 0.0
        self._straggler_lock = threading.Lock()
        self._straggler_count = 0

    @property
    def straggler_count(self) -> int:
        """Deadline-abandoned calls still occupying a concurrency slot."""
        with self._straggler_lock:
            return self._straggler_count

    def _note_straggler_started(self) -> None:
        with self._straggler_lock:
            self._straggler_count += 1

    def _note_straggler_finished(self) -> None:
        with self._straggler_lock:
            self._straggler_count -= 1

    def wait_rate_slot(self) -> None:
        """Block until this provider's next rate slot, then reserve it."""
        interval = self.config.min_interval_s
        if interval <= 0:
            return
        # Lock held only to advance the timestamp; the sleep is outside it.
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._rate_next_fire - now)
            self._rate_next_fire = max(self._rate_next_fire, now) + interval
        if wait > 0:
            time.sleep(wait)


class ProviderGate:
    """Process-wide registry of per-provider gate entries (thread-safe)."""

    def __init__(self, configs: Optional[dict[str, GateConfig]] = None):
        self._lock = threading.Lock()
        self._entries: dict[str, _GateEntry] = {
            provider: _GateEntry(provider, config)
            for provider, config in (configs or DEFAULT_GATE_CONFIGS).items()
        }

    def entry(self, provider: str) -> _GateEntry:
        with self._lock:
            entry = self._entries.get(provider)
            if entry is None:
                entry = _GateEntry(provider, _UNKNOWN_PROVIDER_CONFIG)
                self._entries[provider] = entry
            return entry

    def configure(self, provider: str, config: GateConfig) -> None:
        """Replace a provider's entry (training profiles override the
        inference defaults here). Reconfigure BEFORE launching work: in-flight
        calls keep the entry they acquired, and the replacement starts with
        fresh runtime state — including _rate_next_fire=0.0, so a mid-run swap
        on a rate-limited provider can transiently double its request rate."""
        with self._lock:
            self._entries[provider] = _GateEntry(provider, config)

    def update(self, provider: str, **overrides: Any) -> None:
        """Copy the provider's config with field overrides and reconfigure.
        Same before-launch caveat as :meth:`configure`."""
        with self._lock:
            current = self._entries.get(provider)
            base = current.config if current is not None else _UNKNOWN_PROVIDER_CONFIG
            self._entries[provider] = _GateEntry(provider, replace(base, **overrides))

    def straggler_counts(self) -> dict[str, int]:
        with self._lock:
            entries = list(self._entries.values())
        return {e.provider: e.straggler_count for e in entries}

    def reset_defaults(self) -> None:
        with self._lock:
            self._entries = {
                provider: _GateEntry(provider, config)
                for provider, config in DEFAULT_GATE_CONFIGS.items()
            }


_GATE: Optional[ProviderGate] = None
_GATE_INIT_LOCK = threading.Lock()


def get_gate() -> ProviderGate:
    """Return the process-wide ProviderGate singleton."""
    global _GATE
    if _GATE is None:
        with _GATE_INIT_LOCK:
            if _GATE is None:
                _GATE = ProviderGate()
    return _GATE


#: Seam for tests to simulate thread-spawn failure ("can't start new thread").
_THREAD_FACTORY = threading.Thread


def _gated_call(
    entry: _GateEntry,
    call_one: Callable[[int], Any],
    idx: int,
    deadline_result_factory: Optional[Callable[[], Any]],
) -> Any:
    """Run one single-input call through the gate: acquire a concurrency slot,
    enforce the hard deadline if configured, release when the call finishes."""
    config = entry.config
    semaphore = entry.semaphore
    if semaphore is not None:
        semaphore.acquire()

    if config.call_deadline_s is None:
        # Fast path (all inference defaults): no watchdog thread.
        try:
            return call_one(idx)
        finally:
            if semaphore is not None:
                semaphore.release()

    # Deadline path: run on a daemon thread and join with timeout. A sync call
    # cannot be cancelled from outside, so on expiry it is ABANDONED — it keeps
    # running and releases its slot only when it truly finishes.
    box: dict[str, Any] = {}
    # done is set by the runner before it releases the slot; abandoned is set by
    # the caller at expiry. Both under state_lock so exactly one side wins and
    # the straggler counter's increment/decrement pair up.
    state = {"done": False, "abandoned": False}
    state_lock = threading.Lock()

    def _runner() -> None:
        try:
            box["value"] = call_one(idx)
        except BaseException as exc:  # re-raised on the caller thread
            box["error"] = exc
        finally:
            with state_lock:
                state["done"] = True
                was_abandoned = state["abandoned"]
            if was_abandoned:
                entry._note_straggler_finished()
            if semaphore is not None:
                semaphore.release()

    thread = _THREAD_FACTORY(
        target=_runner, daemon=True, name=f"provider-gate-{entry.provider}-{idx}"
    )
    try:
        thread.start()
    except RuntimeError:
        # start() can fail after the slot was acquired; the runner owns the
        # release and never ran, so release here.
        if semaphore is not None:
            semaphore.release()
        raise

    thread.join(config.call_deadline_s)
    with state_lock:
        if not state["done"]:
            state["abandoned"] = True
            entry._note_straggler_started()
    if state["abandoned"]:
        if deadline_result_factory is not None:
            return deadline_result_factory()
        raise ProviderGateDeadlineExceeded(
            f"Call to provider '{entry.provider}' exceeded hard deadline of "
            f"{config.call_deadline_s}s (item {idx}); abandoned as a straggler."
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_batched_predict(
    provider: str,
    num_items: int,
    call_one: Callable[[int], Any],
    logger: Any = None,
    deadline_result_factory: Optional[Callable[[], Any]] = None,
) -> list[Any]:
    """Shared predict driver: fan a batch of single-input calls out on a
    bounded executor, routing each through the provider's gate.

    ``call_one`` maps an index in 0..num_items-1 to that item's result (a
    ModelResponse or a list of them); exceptions propagate as from
    ``future.result()``. ``deadline_result_factory`` supplies the substitute
    result when a configured hard deadline expires, so a deadline degrades
    per-item like an API failure instead of failing the batch; without it,
    expiry raises :class:`ProviderGateDeadlineExceeded`. Results come back in
    input order.
    """
    entry = get_gate().entry(provider)
    try:
        with ThreadPoolExecutor(max_workers=entry.config.batch_fanout) as executor:
            futures = [
                executor.submit(
                    _gated_call, entry, call_one, idx, deadline_result_factory
                )
                for idx in range(num_items)
            ]
            return [future.result() for future in futures]
    except RuntimeError as e:
        # If a straggler outlives orchestrator shutdown, atexit sets
        # concurrent.futures' module-level _shutdown and any new
        # ThreadPoolExecutor rejects .submit(). Fall back to serial.
        if "cannot schedule new futures after interpreter shutdown" not in str(e):
            raise
        if logger is not None:
            logger.warning(
                "Interpreter shutting down; falling back to sequential predict."
            )
        return [
            _gated_call(entry, call_one, idx, deadline_result_factory)
            for idx in range(num_items)
        ]


def call_with_retry(
    provider: str,
    fn: Callable[[], Any],
    *,
    exception: Any = Exception,
    giveup: Optional[Callable[[Exception], bool]] = None,
) -> Any:
    """Execute a raw provider call under the gate's retry budget.

    Wraps the same scope the old per-wrapper ``@backoff.on_exception``
    decorators did: ``exception`` (what is retryable) and ``giveup`` come from
    the wrapper, only max_tries/factor/max_value come from the gate config. The
    provider's rate slot is acquired once per attempt, initial fire or retry.
    On an exhausted budget the last exception is re-raised.
    """
    entry = get_gate().entry(provider)
    config = entry.config

    def _attempt() -> Any:
        entry.wait_rate_slot()
        return fn()

    if config.max_tries <= 1:
        return _attempt()

    wait_gen_kwargs: dict[str, Any] = {}
    if config.backoff_factor is not None:
        wait_gen_kwargs["factor"] = config.backoff_factor
    if config.backoff_max_value is not None:
        wait_gen_kwargs["max_value"] = config.backoff_max_value

    retrying = backoff.on_exception(
        backoff.expo,
        exception,
        max_tries=config.max_tries,
        giveup=giveup if giveup is not None else (lambda _e: False),
        **wait_gen_kwargs,
    )(_attempt)
    return retrying()
