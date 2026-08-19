"""One immutable, path-safe namespace for every process launch.

Production entry points resolve the namespace once and pass the resulting
string to every sink.  Scheduler launches provide it through
``DEBATE_LAUNCH_NAMESPACE``; manual launches get a canonical UUID4.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path


ENV_VAR = "DEBATE_LAUNCH_NAMESPACE"
_NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_CLAIM_LOCK = threading.Lock()


@dataclass(frozen=True)
class _DirectoryAuthority:
    fd: int
    device: int
    inode: int


_PROCESS_CLAIMS: dict[tuple[int, str], _DirectoryAuthority] = {}


def _clear_claims_after_fork() -> None:
    """A child is a different launch authority even before it execs."""
    global _CLAIM_LOCK
    for authority in _PROCESS_CLAIMS.values():
        try:
            os.close(authority.fd)
        except OSError:
            pass
    _CLAIM_LOCK = threading.Lock()
    _PROCESS_CLAIMS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_clear_claims_after_fork)


def validate_launch_namespace(value: str) -> str:
    """Return *value* unchanged when it is an approved path component."""
    if not isinstance(value, str) or _NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError(
            f"{ENV_VAR} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}, "
            f"got {value!r}"
        )
    return value


def resolve_launch_namespace(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one scheduler-supplied namespace or a manual UUID4 fallback.

    Presence and value are deliberately distinct: an explicitly empty
    environment value is invalid and must not silently turn into a new manual
    launch identity.
    """
    if value is not None:
        return validate_launch_namespace(value)
    source = os.environ if environ is None else environ
    if ENV_VAR in source:
        return validate_launch_namespace(source[ENV_VAR])
    return str(uuid.uuid4())


def safe_path_component(value: str, *, fallback: str) -> str:
    """Return a readable safe component, hashing values that need slugging."""
    if (
        isinstance(value, str)
        and len(value.encode("utf-8")) <= 128
        and _SAFE_COMPONENT_RE.fullmatch(value) is not None
    ):
        return value
    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("._-")[:80]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{slug or fallback}-{digest}"


def _claim_key(path: Path) -> str:
    """Lexical absolute key; unlike resolve(), this never follows symlinks."""
    return os.path.abspath(os.fspath(path))


def _process_claim_key(path: Path) -> tuple[int, str]:
    return os.getpid(), _claim_key(path)


def _path_walk(path: str | os.PathLike[str]) -> tuple[Path, str, tuple[str, ...]]:
    claimed = Path(path)
    raw = os.fspath(path)
    if not raw or raw == ".":
        raise ValueError("launch destination must name a leaf directory")
    parts = claimed.parts
    if claimed.is_absolute():
        anchor = claimed.anchor
        components = parts[1:]
    else:
        anchor = "."
        components = parts
    if not components or any(part in {"", ".", ".."} for part in components):
        raise ValueError(
            f"launch destination must not contain empty, dot, or parent components: {claimed}"
        )
    return claimed, anchor, tuple(components)


def _open_directory(parent_fd: int, component: str, *, display: Path) -> int:
    try:
        return os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"refusing unsafe directory component while claiming {display}: {component}",
        ) from exc


def _verify_links(links: list[tuple[int, str, int]], *, display: Path) -> None:
    """Ensure every opened child is still the no-follow entry under its parent."""
    for parent_fd, component, child_fd in links:
        try:
            linked = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
        except OSError as exc:
            raise RuntimeError(
                f"directory ancestry changed while claiming {display}"
            ) from exc
        if (
            not stat.S_ISDIR(linked.st_mode)
            or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RuntimeError(
                f"directory ancestry changed while claiming {display}"
            )


def _existing_directory_identity(path: str | os.PathLike[str]) -> tuple[int, int]:
    """Open an existing path component-by-component without following links."""
    claimed, anchor, components = _path_walk(path)
    fds = [os.open(anchor, _DIRECTORY_FLAGS)]
    links: list[tuple[int, str, int]] = []
    try:
        for component in components:
            child_fd = _open_directory(fds[-1], component, display=claimed)
            links.append((fds[-1], component, child_fd))
            fds.append(child_fd)
        _verify_links(links, display=claimed)
        opened = os.fstat(fds[-1])
        return opened.st_dev, opened.st_ino
    finally:
        for fd in reversed(fds):
            os.close(fd)


def claim_directory(path: str | os.PathLike[str]) -> Path:
    """Claim a leaf below a trusted, stable parent without following links.

    Directory descriptors anchor the walk and an open descriptor for the leaf
    is retained for later writes. ``mkdirat`` does not return an inode on either
    macOS or Linux, so a hostile actor with rename permission on a newly created
    parent can still win the mkdir-to-open window with another real directory.
    This API therefore requires its configured root/parents to be owner-trusted
    and stable during claiming. A consumer that later hands a pathname to an
    external API (rather than using the retained directory descriptor) must
    keep that trusted-root condition for the entire external-consumption
    lifetime too. This API does protect against symlink traversal, detects
    observed ancestry replacement, and makes same-leaf creation an atomic
    one-winner operation.
    """
    claimed, anchor, components = _path_walk(path)
    fds = [os.open(anchor, _DIRECTORY_FLAGS)]
    links: list[tuple[int, str, int]] = []
    try:
        for component in components[:-1]:
            try:
                child_fd = os.open(
                    component, _DIRECTORY_FLAGS, dir_fd=fds[-1]
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, dir_fd=fds[-1])
                except FileExistsError:
                    pass
                child_fd = _open_directory(
                    fds[-1], component, display=claimed
                )
            except OSError as exc:
                raise OSError(
                    exc.errno,
                    "refusing unsafe directory component while claiming "
                    f"{claimed}: {component}",
                ) from exc
            links.append((fds[-1], component, child_fd))
            fds.append(child_fd)

        leaf = components[-1]
        try:
            os.mkdir(leaf, dir_fd=fds[-1])
        except FileExistsError:
            raise FileExistsError(
                f"refusing existing launch destination: {claimed}"
            ) from None
        leaf_fd = _open_directory(fds[-1], leaf, display=claimed)
        links.append((fds[-1], leaf, leaf_fd))
        fds.append(leaf_fd)
        _verify_links(links, display=claimed)
        opened = os.fstat(leaf_fd)
        authority_fd = os.dup(leaf_fd)
        with _CLAIM_LOCK:
            key = _process_claim_key(claimed)
            if key in _PROCESS_CLAIMS:
                os.close(authority_fd)
                raise FileExistsError(
                    f"refusing destination already claimed by this process: {claimed}"
                )
            _PROCESS_CLAIMS[key] = _DirectoryAuthority(
                fd=authority_fd,
                device=opened.st_dev,
                inode=opened.st_ino,
            )
        return claimed
    finally:
        for fd in reversed(fds):
            os.close(fd)


def require_claimed_directory(path: str | os.PathLike[str]) -> Path:
    """Accept only a still-stable directory claimed earlier by this process."""
    claimed = Path(path)
    with _CLAIM_LOCK:
        expected = _PROCESS_CLAIMS.get(_process_claim_key(claimed))
    if expected is None:
        raise ValueError(
            f"directory was not claimed by this process: {claimed}"
        )
    try:
        actual = _existing_directory_identity(claimed)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"claimed directory is no longer safely reachable: {claimed}"
        ) from exc
    if actual != (expected.device, expected.inode):
        raise ValueError(
            f"claimed directory identity changed after reservation: {claimed}"
        )
    return claimed


def open_claimed_text_file(
    directory: str | os.PathLike[str],
    filename: str,
) -> TextIOWrapper:
    """Exclusively create a regular file through retained directory authority.

    The pathname is checked against the retained inode before opening. If an
    ancestor changes after that check, ``openat`` still targets the retained
    directory rather than following the replacement.
    """
    if not isinstance(filename, str) or _FILE_COMPONENT_RE.fullmatch(filename) is None:
        raise ValueError(f"output filename must be one safe component, got {filename!r}")
    claimed = Path(directory)
    with _CLAIM_LOCK:
        authority = _PROCESS_CLAIMS.get(_process_claim_key(claimed))
        if authority is None:
            raise ValueError(
                f"directory was not claimed by this process: {claimed}"
            )
        authority_fd = os.dup(authority.fd)
        expected = authority.device, authority.inode
    try:
        try:
            actual = _existing_directory_identity(claimed)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"claimed directory is no longer safely reachable: {claimed}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"claimed directory identity changed after reservation: {claimed}"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            file_fd = os.open(filename, flags, 0o666, dir_fd=authority_fd)
        except FileExistsError:
            raise FileExistsError(
                f"refusing existing launch output: {claimed / filename}"
            ) from None
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            os.close(file_fd)
            raise RuntimeError(
                f"claimed output is not a regular file: {claimed / filename}"
            )
        return os.fdopen(file_fd, "w", encoding="utf-8")
    finally:
        os.close(authority_fd)


def open_claimed_read_fd(
    directory: str | os.PathLike[str],
    filename: str,
) -> int:
    """Open one completed, immutable-looking file through retained authority.

    The returned descriptor is caller-owned, read-only, blocking, and
    positioned at byte zero.  Its open file description pins the validated
    inode even if the directory entry is subsequently renamed or replaced.
    """
    if (
        not isinstance(filename, str)
        or _FILE_COMPONENT_RE.fullmatch(filename) is None
    ):
        raise ValueError(
            f"input filename must be one safe component, got {filename!r}"
        )
    claimed = Path(directory)
    with _CLAIM_LOCK:
        authority = _PROCESS_CLAIMS.get(_process_claim_key(claimed))
        if authority is None:
            raise ValueError(
                f"directory was not claimed by this process: {claimed}"
            )
        authority_fd = os.dup(authority.fd)
        expected = authority.device, authority.inode
    file_fd = -1
    try:
        try:
            actual = _existing_directory_identity(claimed)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"claimed directory is no longer safely reachable: {claimed}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"claimed directory identity changed after reservation: {claimed}"
            )

        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
            | os.O_NONBLOCK
        )
        try:
            file_fd = os.open(filename, flags, dir_fd=authority_fd)
        except OSError as exc:
            raise RuntimeError(
                f"refusing unsafe claimed input: {claimed / filename}"
            ) from exc

        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(
                f"claimed input is not a regular file: {claimed / filename}"
            )
        if opened.st_uid != os.geteuid():
            raise RuntimeError(
                f"claimed input is not owned by euid {os.geteuid()}: "
                f"{claimed / filename}"
            )
        if opened.st_nlink != 1:
            raise RuntimeError(
                f"claimed input must have exactly one hard link: "
                f"{claimed / filename}"
            )
        if opened.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"claimed input must not be group- or other-writable: "
                f"{claimed / filename}"
            )

        try:
            named = os.stat(filename, dir_fd=authority_fd, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"claimed input changed while validating: {claimed / filename}"
            ) from exc
        if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(
                f"claimed input changed while validating: {claimed / filename}"
            )

        status_flags = fcntl.fcntl(file_fd, fcntl.F_GETFL)
        if status_flags & os.O_ACCMODE != os.O_RDONLY:
            raise RuntimeError(
                f"claimed input did not open read-only: {claimed / filename}"
            )
        if status_flags & os.O_NONBLOCK:
            fcntl.fcntl(file_fd, fcntl.F_SETFL, status_flags & ~os.O_NONBLOCK)
        final_flags = fcntl.fcntl(file_fd, fcntl.F_GETFL)
        if final_flags & os.O_ACCMODE != os.O_RDONLY or final_flags & os.O_NONBLOCK:
            raise RuntimeError(
                f"claimed input descriptor flags are unsafe: {claimed / filename}"
            )
        if os.lseek(file_fd, 0, os.SEEK_SET) != 0:
            raise RuntimeError(
                f"claimed input is not positioned at byte zero: "
                f"{claimed / filename}"
            )

        result = file_fd
        file_fd = -1
        return result
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(authority_fd)
