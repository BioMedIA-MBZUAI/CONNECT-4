"""Deterministic content hashes for CONNECT-4 QA and rendering sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


IMPLEMENTATION_SOURCE_PATHS = {
    "quality_evaluator": "eval/quality.py",
    "paper_metrics": "eval/postseal_metrics.py",
    "quality_cli": "scripts/check_4d_quality.py",
    "visualizer": "scripts/visualize_4d_comparison.py",
    "spatial_detail_helper": "utils/spatial_detail.py",
    "source_provenance_helper": "utils/source_provenance.py",
}


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_source_records(
    repository_root: str | Path | None = None,
) -> Dict[str, Dict[str, str | int]]:
    """Return exact paths, sizes, and SHA-256 hashes for QA/rendering code."""
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    records: Dict[str, Dict[str, str | int]] = {}
    for role, relative in IMPLEMENTATION_SOURCE_PATHS.items():
        path = (root / relative).resolve(strict=True)
        records[role] = {
            "relative_path": relative,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return records


__all__ = [
    "IMPLEMENTATION_SOURCE_PATHS",
    "implementation_source_records",
    "sha256_file",
]
