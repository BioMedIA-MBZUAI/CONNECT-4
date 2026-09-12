#!/usr/bin/env python3
"""Run fail-closed CONNECT-4 geometry, texture, and dynamics QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.quality import (
    QualityContractError,
    QualityPolicy,
    evaluate_4d_pair_quality,
)
from utils.source_provenance import implementation_source_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a real/predicted 4D fMRI pair and reject registration, "
            "texture, field-of-view, or temporal-collapse failures."
        )
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument(
        "--roi-labels",
        type=Path,
        default=None,
        help=(
            "3D anatomical integer label map on the exact fMRI grid; enables "
            "phase-invariant ROI FC and spectral gates"
        ),
    )
    parser.add_argument(
        "--require-structured-temporal",
        action="store_true",
        help=(
            "fail final validation when usable ROI labels or either structured "
            "temporal metric are unavailable"
        ),
    )
    parser.add_argument(
        "--min-structured-temporal-rois",
        type=int,
        default=None,
        help="override the minimum number of usable anatomical ROIs",
    )
    parser.add_argument(
        "--min-roi-fc-correlation",
        type=float,
        default=None,
        help="override the minimum upper-triangle ROI FC matrix correlation",
    )
    parser.add_argument(
        "--min-roi-power-spectrum-correlation",
        type=float,
        default=None,
        help="override the minimum normalized ROI power-spectrum correlation",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy_overrides = {
            name: value
            for name, value in (
                (
                    "min_structured_temporal_rois",
                    args.min_structured_temporal_rois,
                ),
                ("min_roi_fc_correlation", args.min_roi_fc_correlation),
                (
                    "min_roi_power_spectrum_correlation",
                    args.min_roi_power_spectrum_correlation,
                ),
            )
            if value is not None
        }
        policy = QualityPolicy(**policy_overrides)
        report = evaluate_4d_pair_quality(
            args.real,
            args.predicted,
            mask_path=args.mask,
            roi_labels_path=args.roi_labels,
            require_structured_temporal=args.require_structured_temporal,
            policy=policy,
        )
    except (QualityContractError, ValueError, RuntimeError) as exc:
        report = {
            "schema": "connect4-paired-4d-quality-v1",
            "passed": False,
            "verdict": "fail_contract",
            "error": str(exc),
            "implementation_sources": implementation_source_records(),
            "sources": {
                "real": str(args.real),
                "predicted": str(args.predicted),
                "mask": None if args.mask is None else str(args.mask),
                "roi_labels": (
                    None if args.roi_labels is None else str(args.roi_labels)
                ),
            },
        }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
