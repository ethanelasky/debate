#!/usr/bin/env python3
"""Emit frozen client/protocol/verifier provenance for executor deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = "palaestra.codecontests.client-provenance.v1"


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe/missing provenance file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "format": FORMAT,
        "client_sha256": _digest(args.client),
        "protocol_sha256": _digest(args.protocol),
        "verifier_sha256": _digest(args.verifier),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
