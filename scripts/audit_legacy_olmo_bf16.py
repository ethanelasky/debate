#!/usr/bin/env python3
"""Create or verify a fail-closed snapshot of the legacy OLMo-32B BF16 model."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from infra.legacy_olmo_bf16 import (
    AUDIT_MARKER_NAME,
    AUDITED_MODEL_PATH,
    LEGACY_SOURCE_PATH,
    LegacyArtifactError,
    audit_legacy_artifact,
    validate_audited_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(LEGACY_SOURCE_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(AUDITED_MODEL_PATH))
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "auto"),
        default="hardlink",
        help=(
            "hardlink is fast and storage-free; auto copies only if linking "
            "is unsupported"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-hash an existing committed audited directory instead of creating it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            validate_audited_artifact(args.output_dir, progress=print)
        else:
            audit_legacy_artifact(
                args.source_dir,
                args.output_dir,
                link_mode=args.link_mode,
                progress=print,
            )
        marker = args.output_dir / AUDIT_MARKER_NAME
        digest = hashlib.sha256(marker.read_bytes()).hexdigest()
        print(f"audit_marker={marker}")
        print(f"audit_marker_sha256={digest}")
    except (LegacyArtifactError, OSError) as exc:
        print(f"legacy BF16 audit refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
