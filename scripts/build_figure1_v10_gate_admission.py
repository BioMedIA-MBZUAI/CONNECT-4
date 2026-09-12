#!/usr/bin/env python3
"""Build one immutable, target-array-free recovery admission for the GPU sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.cohort_admission import cohort_admission_identity  # noqa: E402
from scripts.benchmark_figure1_v10_gpu import (  # noqa: E402
    _admission_runtime_identity,
    _execution_identity,
)
from training.train import _build_rank_zero_training_admission  # noqa: E402
from utils.config import validate_figure1_recovery_gate_config  # noqa: E402
from utils.immutable_yaml import read_immutable_yaml_snapshot  # noqa: E402
from utils.figure1_gpu_gate import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_sha256,
    source_tree_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(args: argparse.Namespace) -> None:
    config_snapshot = read_immutable_yaml_snapshot(
        args.config, "gate admission config"
    )
    source_root = args.source_root.expanduser()
    if source_root.resolve(strict=True) != REPOSITORY_ROOT:
        raise RuntimeError("gate admission source-root differs from executing stage")
    config = validate_figure1_recovery_gate_config(config_snapshot.document)
    if config["data"].get("protocol_profile") != "a4-native-recovery-v1":
        raise RuntimeError("Figure-1 real-batch gate admission is recovery-only")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("gate admission requires its allocated A100 CUDA device")
    properties = torch.cuda.get_device_properties(0)
    execution = _execution_identity(properties)
    source_sha256, source_count = source_tree_identity(source_root)
    runtime = _admission_runtime_identity(
        config,
        execution=execution,
        source_tree_sha256=source_sha256,
        source_file_count=source_count,
        world_size=1,
    )
    admission = _build_rank_zero_training_admission(config, 1, runtime)
    wrapper = {
        "schema": "connect4-figure1-v10-gate-cohort-admission-v1",
        "status": "ADMITTED_REAL_RECOVERY_TRAINING_COHORT",
        "source_tree_sha256": source_sha256,
        "source_file_count": source_count,
        "config_sha256": canonical_sha256(config),
        "config_file_sha256": config_snapshot.sha256,
        "runtime_identity": runtime,
        "cohort_admission": admission,
        "cohort_admission_identity": cohort_admission_identity(admission),
    }
    wrapper["record_sha256"] = canonical_sha256(wrapper)
    atomic_write_json_no_replace(args.output, wrapper)
    print(json.dumps(wrapper["cohort_admission_identity"], indent=2), flush=True)


if __name__ == "__main__":
    main(_parser().parse_args())
