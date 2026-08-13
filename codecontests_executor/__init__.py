"""Authenticated remote executor for CodeContests Python candidates.

The service-side package intentionally uses only the Python standard library so
it can be installed on the small, credential-free executor guest without the
rest of the training repository.
"""

from .protocol import (
    PROTOCOL_VERSION,
    ExecutorProtocolError,
    derive_limits,
    outputs_match,
)

__all__ = [
    "PROTOCOL_VERSION",
    "ExecutorProtocolError",
    "derive_limits",
    "outputs_match",
]
