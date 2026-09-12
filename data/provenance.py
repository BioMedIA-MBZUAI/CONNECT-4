"""Canonical content hashes used to bind CONNECT-4 caches to their sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"provenance source not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_file_sha256(path: str | Path) -> dict[str, str]:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"provenance directory not found: {directory}")
    return {
        file.relative_to(directory).as_posix(): sha256_file(file)
        for file in sorted(directory.rglob("*"))
        if file.is_file()
    }


def validate_sha256_map(
    recorded: Mapping[str, str], current: Mapping[str, str], *, context: str
) -> None:
    if dict(recorded) != dict(current):
        missing = sorted(set(current) - set(recorded))
        extra = sorted(set(recorded) - set(current))
        changed = sorted(
            key
            for key in set(recorded) & set(current)
            if recorded[key] != current[key]
        )
        raise RuntimeError(
            f"{context} source fingerprint mismatch; missing={missing[:10]}, "
            f"extra={extra[:10]}, changed={changed[:10]}"
        )


__all__ = [
    "canonical_sha256",
    "directory_file_sha256",
    "sha256_file",
    "validate_sha256_map",
]
