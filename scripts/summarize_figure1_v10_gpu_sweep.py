#!/usr/bin/env python3
"""Select the largest A100-safe v10 D core and retain release stops on failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.figure1_gpu_gate import (  # noqa: E402
    EXPECTED_CORES,
    MINIMUM_SAFETY_MARGIN_GIB,
    PAPER_PROFILE_BLOCK_STATUS,
    SWEEP_SUMMARY_SCHEMA,
    atomic_write_json_no_replace,
    canonical_sha256,
    seal_sweep_summary,
    sha256_file,
    validate_benchmark_record,
    validate_sweep_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = {}
    all_bindings = []
    for profile, expected_cores in EXPECTED_CORES.items():
        if profile == "paper":
            profiles[profile] = {
                "status": PAPER_PROFILE_BLOCK_STATUS,
                "attempted_core_sizes": [],
                "largest_safe_core_size": None,
                "selected_peak_allocated_gib": None,
                "selected_peak_reserved_gib": None,
                "selected_forward_seconds": None,
                "selected_backward_seconds": None,
                "selected_step_seconds": None,
                "memory_gate_passed": False,
                "four_gpu_ddp_smoke_attested": False,
                "release_stop": True,
                "record_order_sha256": canonical_sha256([]),
                "record_bindings": [],
                "records": [],
                "blocking_contract": {
                    "required_shape_bctdhw": [1, 1, 128, 128, 128, 128],
                    "perceptual_loss_required": True,
                    "available_brainlm_authority_grid": [64, 80, 64],
                    "reviewed_128_grid_brainlm_authority_available": False,
                    "synthetic_or_omitted_brainlm_forbidden": True,
                },
            }
            continue
        records = []
        record_bindings = []
        for core in expected_cores:
            path = args.results_dir / f"figure1_v10_{profile}_slab{core}.json"
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            try:
                record = validate_benchmark_record(
                    value,
                    expected_profile=profile,
                    expected_core=core,
                )
            except ValueError as exc:
                raise ValueError(f"invalid v10 benchmark record {path}: {exc}") from exc
            records.append(record)
            binding = {
                "order": len(record_bindings),
                "profile": profile,
                "depth_slab_size": core,
                "filename": path.name,
                "file_sha256": sha256_file(path),
                "canonical_record_sha256": record["canonical_record_sha256"],
                "slurm_job_id": record["execution"]["slurm_job_id"],
            }
            record_bindings.append(binding)
            all_bindings.append(binding)
        attempted = [record["depth_slab_size"] for record in records]
        if attempted != list(expected_cores[: len(attempted)]):
            raise RuntimeError(
                f"{profile} sweep is not a contiguous core-size prefix"
            )
        safe = [record for record in records if record.get("passed") is True]
        selected = max(safe, key=lambda record: record["depth_slab_size"]) if safe else None
        profiles[profile] = {
            "attempted_core_sizes": attempted,
            "largest_safe_core_size": (
                selected["depth_slab_size"] if selected is not None else None
            ),
            "selected_peak_allocated_gib": (
                selected["peak_allocated_gib"] if selected is not None else None
            ),
            "selected_peak_reserved_gib": (
                selected["peak_reserved_gib"] if selected is not None else None
            ),
            "selected_forward_seconds": (
                selected["forward_seconds"] if selected is not None else None
            ),
            "selected_backward_seconds": (
                selected["backward_seconds"] if selected is not None else None
            ),
            "selected_step_seconds": (
                selected["step_seconds"] if selected is not None else None
            ),
            "memory_gate_passed": selected is not None,
            "four_gpu_ddp_smoke_attested": False,
            "release_stop": True,
            "record_order_sha256": canonical_sha256(record_bindings),
            "record_bindings": record_bindings,
            "records": records,
        }

    summary = {
        "schema": SWEEP_SUMMARY_SCHEMA,
        "passed": False,
        "minimum_safety_margin_gib": MINIMUM_SAFETY_MARGIN_GIB,
        "selection_rule": (
            "largest core whose exact per-rank full Connect4 forward/backward/"
            "gradient-clip/AdamW step stays four GiB below A100-40GB memory"
        ),
        "four_gpu_ddp_smoke_required": True,
        "four_gpu_ddp_smoke_attested": False,
        "record_file_order_sha256": canonical_sha256(all_bindings),
        "paper_profile_release_stop": True,
        "profiles": profiles,
    }
    summary = seal_sweep_summary(summary)
    validate_sweep_summary(summary)
    atomic_write_json_no_replace(args.output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
