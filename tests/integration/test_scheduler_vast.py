"""Opt-in, read-only Vast integration contract.

Set ``VAST_INTEGRATION=1`` to authenticate, list instances, and search offers.
Paid Vast lifecycle and garbage-collection integration is deferred until the
adapter, watchdog, and ownership contracts are approved. This test never reads
``~/code/.env`` or puts credentials in command arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import os
import re
import subprocess

import pytest


_ENABLED = os.environ.get("VAST_INTEGRATION") == "1"
_PROFILE = os.environ.get("SCHEDULER_VAST_PROFILE", "vast:b300x1")
_OFFER_QUERY = os.environ.get("SCHEDULER_VAST_OFFER_QUERY", "gpu_name = B300")
_PROFILE_SYNTAX = re.compile(r"vast:[a-z0-9][a-z0-9._-]*")
_TIMEOUT_SECONDS = 120
_CREDENTIAL_WORDS = frozenset(
    {"apikey", "authorization", "bearer", "credential", "password", "secret", "token"}
)
_SAFE_SUBPROCESS_KWARGS = {
    "capture_output": True,
    "text": True,
    "check": True,
}


def _words(value: str) -> set[str]:
    """Split punctuation and camelCase before credential-word checks."""
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    return {part for part in re.split(r"[^a-z0-9]+", value.lower()) if part}


class _ReadOnlyRunner:
    """Allow only the two Vast CLI reads used by this integration smoke."""

    def __init__(self, *, offer_query: str) -> None:
        if _words(offer_query) & _CREDENTIAL_WORDS:
            raise AssertionError("credential-bearing offer query refused")
        self._inventory_command = ("vastai", "show", "instances", "--raw")
        self._offer_command = (
            "vastai",
            "search",
            "offers",
            offer_query,
            "--raw",
        )
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[object], **kwargs: object) -> object:
        command = tuple(str(part) for part in argv)
        if command not in {self._inventory_command, self._offer_command}:
            # Do not interpolate rejected argv: it may contain a credential.
            raise AssertionError("provider command is not an approved read-only form")
        if kwargs != _SAFE_SUBPROCESS_KWARGS:
            # This rejects shell, executable, preexec_fn, env overrides, and all
            # other subprocess controls instead of trying to sanitize them.
            raise AssertionError("provider subprocess options are not approved")

        self.calls.append(command)
        return subprocess.run(
            command,
            shell=False,
            timeout=_TIMEOUT_SECONDS,
            **_SAFE_SUBPROCESS_KWARGS,
        )


def _require_api_key_name() -> None:
    if "VAST_API_KEY" not in os.environ:
        pytest.fail(
            "VAST_API_KEY is missing; export it into this shell before pytest",
            pytrace=False,
        )


def _build_harness(runner: _ReadOnlyRunner) -> object:
    try:
        module = importlib.import_module("tools.scheduler.vast")
    except ModuleNotFoundError:
        pytest.fail("`tools.scheduler.vast` is not implemented", pytrace=False)
    factory = getattr(module, "create_integration_harness", None)
    if not callable(factory):
        pytest.fail("`create_integration_harness` is missing", pytrace=False)
    return factory(
        profile=_PROFILE,
        command_runner=runner,
        timeout_seconds=_TIMEOUT_SECONDS,
    )


def _inventory(harness: object) -> list[tuple[str, str, str]]:
    records = harness.inventory()
    assert isinstance(records, Sequence) and not isinstance(records, (str, bytes))
    snapshot: list[tuple[str, str, str]] = []
    for record in records:
        assert isinstance(record, Mapping), "inventory entry must be a mapping"
        instance_id = str(record.get("id") or "")
        assert instance_id, "inventory entry needs an id"
        snapshot.append(
            (
                instance_id,
                str(record.get("name") or ""),
                str(record.get("status") or ""),
            )
        )
    return sorted(snapshot)


@pytest.mark.skipif(
    not _ENABLED, reason="set VAST_INTEGRATION=1 for the read-only Vast smoke"
)
def test_vast_authenticated_inventory_and_offer_search_are_read_only() -> None:
    _require_api_key_name()
    assert _PROFILE_SYNTAX.fullmatch(_PROFILE), (
        f"profile must use `vast:<flavor>` syntax: {_PROFILE!r}"
    )
    assert _OFFER_QUERY.strip(), "the configured offer query cannot be empty"

    runner = _ReadOnlyRunner(offer_query=_OFFER_QUERY)
    harness = _build_harness(runner)
    before = _inventory(harness)
    try:
        report = harness.preflight(offer_query=_OFFER_QUERY)
    finally:
        after = _inventory(harness)
        assert before == after, "read-only preflight changed instance id/name/status"

    assert isinstance(report, Mapping), "preflight must return a mapping"
    assert report.get("offer_query") == _OFFER_QUERY
    offers = report.get("offers")
    assert isinstance(offers, Sequence) and not isinstance(offers, (str, bytes))
    for offer in offers:
        assert isinstance(offer, Mapping), "offer entry must be a mapping"
        assert offer.get("id") is not None, "offer entry needs an id"
        assert str(offer.get("gpu_name") or ""), "offer entry needs a gpu name"

    assert runner.calls == [
        ("vastai", "show", "instances", "--raw"),
        ("vastai", "search", "offers", _OFFER_QUERY, "--raw"),
        ("vastai", "show", "instances", "--raw"),
    ], "harness must perform exactly two inventories around one offer search"
