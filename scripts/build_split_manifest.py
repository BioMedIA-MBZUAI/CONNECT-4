#!/usr/bin/env python3
"""Explicit, write-once builders for immutable CONNECT-4 split authorities."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.provenance import sha256_file  # noqa: E402
from data.split import (  # noqa: E402
    build_recovery_split_manifest,
    build_seeded_split_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new immutable split authority offline. Existing outputs "
            "are never replaced. Runtime train/preprocess/infer commands cannot "
            "invoke this builder."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    recovery = subparsers.add_parser(
        "recovery-half", help="derive the exact root-reviewed half-cohort split"
    )
    recovery.add_argument("--selection-bundle", required=True)
    recovery.add_argument("--selection-root-review", required=True)
    recovery.add_argument("--output", required=True)

    paper = subparsers.add_parser(
        "paper-seeded",
        help="build a fixed paper-profile split from a pinned cohort CSV",
    )
    paper.add_argument("--cohort-manifest", required=True)
    paper.add_argument("--cohort-manifest-sha256", required=True)
    paper.add_argument("--output", required=True)
    paper.add_argument("--val-frac", type=float, default=0.15)
    paper.add_argument("--test-frac", type=float, default=0.15)
    paper.add_argument("--seed", type=int, default=42)
    return parser


def _paper_records(path: Path, expected_sha256: str) -> tuple[list[str], dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("paper cohort-manifest SHA-256 differs")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"scan_id", "patient_id"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("paper cohort manifest requires scan_id and patient_id")
        rows = list(reader)
    scans = [str(row["scan_id"]).strip() for row in rows]
    patients = {
        str(row["scan_id"]).strip(): str(row["patient_id"]).strip()
        for row in rows
    }
    return scans, patients


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "recovery-half":
        manifest = build_recovery_split_manifest(
            selection_bundle=args.selection_bundle,
            selection_root_review=args.selection_root_review,
            output_path=args.output,
        )
    else:
        scans, patients = _paper_records(
            Path(args.cohort_manifest).expanduser(), args.cohort_manifest_sha256
        )
        manifest = build_seeded_split_manifest(
            scans,
            patients,
            output_path=args.output,
            cohort_manifest_sha256=args.cohort_manifest_sha256,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    print(f"published={Path(args.output).expanduser().resolve()}")
    print(f"file_sha256={sha256_file(args.output)}")
    print(f"record_sha256={manifest['record_sha256']}")


if __name__ == "__main__":
    main()
