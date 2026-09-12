"""Single-read, identity-bound YAML snapshots for immutable gate inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

import yaml


@dataclass(frozen=True)
class ImmutableYamlSnapshot:
    """The exact configuration bytes consumed by a gate entry point.

    The digest intentionally covers the original YAML bytes, not its parsed
    representation.  Thus comments, duplicate-key spelling, and any other
    byte-level substitution cannot be hidden by YAML's semantic decoding.
    """

    path: Path
    raw_bytes: bytes
    sha256: str
    document: Any
    device: int
    inode: int
    mode: int


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return every stable regular-file fact relevant while this fd is open."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_immutable_yaml_snapshot(path: Path, label: str) -> ImmutableYamlSnapshot:
    """Read a canonical, single-link YAML file exactly once and bind its bytes.

    A separate pathname hash would introduce a config-byte TOCTOU window.  This
    helper instead opens with ``O_NOFOLLOW``, checks the descriptor identity
    before and after its only read, verifies the pathname still names that same
    canonical file, then hashes and parses precisely those bytes.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeError(f"{label} must be an absolute non-symlink path")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError(f"{label} requires an O_NOFOLLOW-capable platform")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is unreadable or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} must be a single-link regular file")
        # Configurations are deliberately small.  A one-shot read both avoids a
        # second content observation and rejects any abnormal short read rather
        # than silently stitching together bytes from a changing object.
        raw_bytes = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{label} is unreadable or unsafe") from exc
    finally:
        os.close(descriptor)

    if _identity(before) != _identity(after) or len(raw_bytes) != before.st_size:
        raise RuntimeError(f"{label} mutated while it was read")
    try:
        path_metadata = os.lstat(candidate)
        canonical = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} no longer has a canonical pathname") from exc
    if (
        canonical != candidate
        or _identity(path_metadata) != _identity(after)
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
    ):
        raise RuntimeError(f"{label} pathname changed while it was read")
    try:
        document = yaml.safe_load(raw_bytes)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"{label} is unreadable") from exc
    return ImmutableYamlSnapshot(
        path=candidate,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        document=document,
        device=after.st_dev,
        inode=after.st_ino,
        mode=stat.S_IMODE(after.st_mode),
    )


__all__ = ["ImmutableYamlSnapshot", "read_immutable_yaml_snapshot"]
