"""Test-only ownership guard for datasets' process-global tqdm monitor."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def dataset_tqdm_monitor_owner() -> Iterator[Callable[..., Any]]:
    """Stop only the datasets tqdm monitor created by a tracked call."""
    from datasets.utils.tqdm import tqdm as datasets_tqdm

    monitor_before = datasets_tqdm.monitor
    owned_monitor = None

    def call(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        nonlocal owned_monitor
        try:
            return function(*args, **kwargs)
        finally:
            monitor_after = datasets_tqdm.monitor
            if (
                monitor_before is None
                and owned_monitor is None
                and monitor_after is not None
            ):
                owned_monitor = monitor_after

    try:
        yield call
    finally:
        if owned_monitor is not None and datasets_tqdm.monitor is owned_monitor:
            owned_monitor.exit()
            owned_monitor.join()
            if datasets_tqdm.monitor is owned_monitor:
                datasets_tqdm.monitor = None
