"""Retirement regression for Debate's obsolete RunPod integration sketch.

Live provider proofs belong to the independent ``ethanelasky/job-scheduler``
repository.  Debate may eventually invoke the installed scheduler over its
published process/CLI protocol, but it must not import a scheduler package or
own a provider harness.  The historical ``RUNPOD_INTEGRATION`` and
``RUNPOD_PAID_INTEGRATION`` settings are therefore inert here: this module
contains no environment lookup, provider import, subprocess runner, direct
call, SDK hook, or paid/read-only integration body for either setting to
enable.

The scheduler repository's retired-integration-sketch inventory owns the
corresponding RUNPOD-001/REPO-001 disposition and the future live-proof gate.
"""

from __future__ import annotations


def test_obsolete_runpod_harness_surface_is_absent() -> None:
    """Keep the retired flags and direct Python harness disconnected."""

    retired_enable_flags = {
        "RUNPOD_INTEGRATION",
        "RUNPOD_PAID_INTEGRATION",
    }
    obsolete_python_surface = {
        "_READ_ENABLED",
        "_load_harness_factory",
        "_require_read_only_prerequisites",
        "_AuditingRunner",
        "test_runpod_authenticated_preflight_is_read_only",
        "test_paid_runpod_owned_worker_lifecycle_smoke",
    }

    assert retired_enable_flags.isdisjoint(globals())
    assert obsolete_python_surface.isdisjoint(globals())
    local_callables = {
        name
        for name, value in globals().items()
        if callable(value) and getattr(value, "__module__", None) == __name__
    }
    assert local_callables == {"test_obsolete_runpod_harness_surface_is_absent"}
