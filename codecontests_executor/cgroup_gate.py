"""Pinned pre-exec gate used to place rootful runsc without a fork race."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit(125)
    try:
        gate_fd = int(sys.argv[1], 10)
    except ValueError:
        raise SystemExit(125) from None
    if os.read(gate_fd, 1) != b"G":
        raise SystemExit(125)
    os.close(gate_fd)
    command = sys.argv[3:]
    os.execve(command[0], command, dict(os.environ))


if __name__ == "__main__":
    main()
