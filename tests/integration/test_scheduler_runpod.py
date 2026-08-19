"""Retirement regression for Debate's obsolete provider integration sketches.

Live provider proofs belong to the independent ``ethanelasky/job-scheduler``
repository.  Debate may eventually invoke the installed scheduler over its
published process/CLI protocol, but it must not import a scheduler package or
own a provider harness.  The historical RunPod and Vast settings are therefore
inert here: this module contains no environment lookup, provider import,
subprocess runner, direct call, SDK hook, or paid/read-only integration body
for any setting to enable.

The scheduler repository's retired-integration-sketch inventory owns the
corresponding RUNPOD-001/VAST-001/REPO-001 dispositions and future live-proof
gates.  Paid or destructive Vast paths remain blocked by VAST-001.
"""

from __future__ import annotations


def test_obsolete_provider_harness_surfaces_are_absent() -> None:
    """Keep retired flags and direct Python harnesses disconnected."""

    retired_settings = {
        "RUNPOD_INTEGRATION",
        "RUNPOD_PAID_INTEGRATION",
        "VAST_INTEGRATION",
        "SCHEDULER_VAST_PROFILE",
        "SCHEDULER_VAST_OFFER_QUERY",
    }
    obsolete_python_surface = {
        "_READ_ENABLED",
        "_load_harness_factory",
        "_require_read_only_prerequisites",
        "_AuditingRunner",
        "test_runpod_authenticated_preflight_is_read_only",
        "test_paid_runpod_owned_worker_lifecycle_smoke",
        "_ReadOnlyRunner",
        "_build_harness",
        "test_vast_authenticated_inventory_and_offer_search_are_read_only",
    }

    assert retired_settings.isdisjoint(globals())
    assert obsolete_python_surface.isdisjoint(globals())
    local_callables = {
        name
        for name, value in globals().items()
        if callable(value) and getattr(value, "__module__", None) == __name__
    }
    assert local_callables == {"test_obsolete_provider_harness_surfaces_are_absent"}
