import json
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from architecture_contract import SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
import models.brainlm_context as brainlm_context
from models.brainlm_context import (
    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    _expected_official_content_identity,
)
import utils.figure1_gpu_gate as gpu_gate
from utils.figure1_gpu_gate import (
    BENCHMARK_RECORD_SCHEMA,
    DDP_SMOKE_SCHEMA,
    PAPER_PROFILE_BLOCK_STATUS,
    PROFILE_SHAPES,
    SWEEP_SUMMARY_SCHEMA,
    atomic_write_json_no_replace,
    authenticate_configured_depth_slab_selection,
    benchmark_model_config_identity,
    canonical_sha256,
    scheduler_allocation_sha256,
    seal_benchmark_record,
    seal_sweep_summary,
    sha256_file,
    validate_benchmark_record,
    validate_ddp_smoke_attestation,
    validate_sweep_summary,
)


SOURCE_SHA = "a" * 64
SOURCE_COUNT = 7


def _brainlm_identity() -> dict:
    ordered_scan_ids = [f"A4-TRAIN-{index}" for index in range(4)]
    authority = {
        "schema": brainlm_context.BRAINLM_AUTHORITY_SCHEMA,
        "file_sha256": "4" * 64,
        "size_bytes": 1234,
        "authority_record_sha256": "5" * 64,
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "ordered_scan_ids": ordered_scan_ids,
        "ordered_scan_ids_sha256": canonical_sha256(ordered_scan_ids),
        "scan_count": len(ordered_scan_ids),
        "all_scans_explicitly_structurally_reviewed": True,
        "dense_nonlinear_pull_field_required": True,
        "affine_only_fallback_forbidden": True,
    }
    authority["content_record_sha256"] = canonical_sha256(authority)
    identity = {
        "schema": "connect4-brainlm-a424-perceptual-identity-v2",
        "name": "brainlm",
        "model_artifacts": _expected_official_content_identity(),
        "mni_authority": authority,
        "adapter_contract": brainlm_context.BRAINLM_CONTEXT_ADAPTER_CONTRACT,
        "feature_contract": brainlm_context.BRAINLM_FEATURE_SCHEMA,
        "feature_layers": (
            "last four encoder hidden states, CLS token, layer-normalized"
        ),
        "deterministic_identical_prediction_target_mask": True,
        "input_frames": brainlm_context.INPUT_FRAMES,
        "input_tr_seconds": brainlm_context.INPUT_TR_SECONDS,
        "temporal_adapter": "linear 128 to 200 samples, align_corners=True",
        "scaler_adapter": "per-scan/per-parcel median-IQR then clamp [-6,6]",
        "normalization_and_tr_match_pretraining_exactly": False,
        "pretraining_domain_exactness_claimed": False,
        "domain_shift_reason": (
            "TR=3 s differs from BrainLM training acquisitions and public "
            "population parcel median/IQR vectors are unavailable"
        ),
        "projection": brainlm_context._expected_mapping_contract(),
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return identity


def _fixture_sha(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _cohort_identity(*, world_size: int) -> dict:
    value = {
        "schema": "connect4-distributed-cohort-admission-identity-v1",
        "authority_record_sha256": _fixture_sha(
            f"cohort-authority-world-{world_size}"
        ),
        "authority_canonical_bytes_sha256": _fixture_sha(
            f"cohort-bytes-world-{world_size}"
        ),
        "data_root_sha256": _fixture_sha("cohort-stable-data-root"),
        "runtime_identity_sha256": _fixture_sha(
            f"cohort-runtime-world-{world_size}"
        ),
        "config_sha256": _fixture_sha("recovery-config"),
        "ordered_scan_ids_sha256": _fixture_sha("ordered-scan-ids"),
        "partitions_sha256": _fixture_sha("fixed-partitions"),
        "split_identity_sha256": _fixture_sha("fixed-split"),
        "artifact_identity_sha256": _fixture_sha("run-artifact-v5"),
        "structural_artifact_identities_sha256": _fixture_sha(
            "structural-artifacts"
        ),
        "dataset_state_sha256": _fixture_sha("dataset-state-payload"),
        "world_size": world_size,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _brainlm_batch_context_identity(
    scan_index: int,
    *,
    cohort_identity: dict,
) -> dict:
    scan_id = f"A4-TRAIN-{scan_index}"
    configured = _brainlm_identity()
    value = {
        "schema": brainlm_context.BRAINLM_BATCH_CONTEXT_IDENTITY_SCHEMA,
        "brainlm_perceptual_identity_sha256": configured["record_sha256"],
        "authority_file_sha256": configured["mni_authority"]["file_sha256"],
        "authority_content_record_sha256": configured["mni_authority"][
            "content_record_sha256"
        ],
        "authority_record_sha256": configured["mni_authority"][
            "authority_record_sha256"
        ],
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "run_artifact_identity_sha256": cohort_identity[
            "artifact_identity_sha256"
        ],
        "cohort_admission_identity_record_sha256": cohort_identity[
            "record_sha256"
        ],
        "dataset_state_record_sha256": _fixture_sha("dataset-state-record"),
        "dataset_state_sha256": cohort_identity["dataset_state_sha256"],
        "structural_artifact_identities_sha256": cohort_identity[
            "structural_artifact_identities_sha256"
        ],
        "scan_ids": [scan_id],
        "roles": ["train"],
        "prepared_t1_sha256": [_fixture_sha(f"{scan_id}-prepared-t1")],
        "prepared_mask_sha256": [_fixture_sha(f"{scan_id}-prepared-mask")],
        "padded_t1_artifact_descriptor_sha256": [
            _fixture_sha(f"{scan_id}-padded-t1")
        ],
        "padded_mask_artifact_descriptor_sha256": [
            _fixture_sha(f"{scan_id}-padded-mask")
        ],
        "cache_metadata_artifact_descriptor_sha256": [
            _fixture_sha(f"{scan_id}-metadata")
        ],
        "structural_source_identity_sha256": [
            _fixture_sha(f"{scan_id}-structural-source")
        ],
        "native_alignment_authority_sha256": [
            _fixture_sha("native-alignment-authority")
        ],
        "target_artifact_identity_sha256": [
            _fixture_sha(f"{scan_id}-target-artifact")
        ],
        "support_tensor_sha256": [_fixture_sha(f"{scan_id}-brain-support")],
        "support_foreground_voxels": [12345 + scan_index],
        "admitted_scan_context_record_sha256": [
            _fixture_sha(f"{scan_id}-admitted-context")
        ],
        "authority_scan_record_sha256": [
            _fixture_sha(f"{scan_id}-authority-context")
        ],
        "displacement_sha256": [_fixture_sha(f"{scan_id}-displacement")],
        "displacement_size_bytes": [1024 + scan_index],
        "padded_shape": list(brainlm_context.PADDED_SHAPE),
        "native_shape": list(brainlm_context.NATIVE_SHAPE),
        "padding_before": list(brainlm_context.PADDING_BEFORE),
        "padding_after": list(brainlm_context.PADDING_AFTER),
        "padded_affine_ras_mm": [
            [
                [3.0, 0.0, 0.0, -90.0],
                [0.0, 3.0, 0.0, -126.0],
                [0.0, 0.0, 3.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        "runtime_mapping_contract": brainlm_context._expected_mapping_contract(),
        "prediction_target_context_shared": True,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _training_batch_identity(
    scan_index: int = 0,
    *,
    admission_world_size: int = 1,
) -> dict:
    cohort_identity = _cohort_identity(world_size=admission_world_size)
    context_identity = _brainlm_batch_context_identity(
        scan_index, cohort_identity=cohort_identity
    )
    value = {
        "schema": "connect4-figure1-v10-real-training-batch-v2",
        "protocol_profile": "a4-native-recovery-v1",
        "scan_ids": [f"A4-TRAIN-{scan_index}"],
        "roles": ["train"],
        "local_batch_size": 1,
        "cohort_admission_identity": cohort_identity,
        "dataset_state_record_sha256": context_identity[
            "dataset_state_record_sha256"
        ],
        "sample_artifacts_sha256": _fixture_sha(
            f"A4-TRAIN-{scan_index}-sample-artifacts"
        ),
        "target_artifact_identities_sha256": _fixture_sha(
            f"A4-TRAIN-{scan_index}-target-artifacts"
        ),
        "target_tensor_identity": {
            "schema": "connect4-recovery-training-target-tensor-v1",
            "dtype": "torch.float32",
            "shape_bctdhw": [1, 1, 128, 64, 80, 64],
            "sha256": _fixture_sha(f"A4-TRAIN-{scan_index}-target-tensor"),
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "brainlm_batch_context_identity": context_identity,
        "brainlm_context_source": (
            "exact-model-context-preflight; forward-equality-required-on-success"
        ),
        "admitted_unsealed_training_target_loaded": True,
        "synthetic_inputs_used": False,
        "sealed_target_voxel_data_opened": False,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _config(profile: str, core: int) -> dict:
    protocol = {
        "paper": "paper-a4-adni-fmriprep-v1",
        "recovery": "a4-native-recovery-v1",
    }[profile]
    shape = list(PROFILE_SHAPES[profile])
    return {
        "data": {"protocol_profile": protocol, "architecture_shape": shape},
        "models": {
            "dit": {
                "input_size": shape,
                "patch_size": [16, 16, 16],
                "spatial_window_size": [8, 8, 8],
            },
            "unet": {
                "depth_slab_size": core,
                "depth_slab_sweep_summary_path": None,
                "depth_slab_sweep_summary_sha256": None,
                "depth_slab_sweep_summary_sha256_env": "",
                "depth_slab_ddp_smoke_path": None,
                "depth_slab_ddp_smoke_sha256": None,
                "depth_slab_ddp_smoke_sha256_env": "",
            },
        },
        "training": {
            "loss": {
                "perceptual_extractor": {
                    "checkpoint": "/immutable/brainlm.pt",
                    "checkpoint_env": "",
                    "checkpoint_sha256": "e" * 64,
                    "checkpoint_sha256_env": "",
                    "source_revision": "1" * 40,
                    "source_revision_env": "",
                }
            }
        },
    }


def _record(profile: str, core: int, *, passed: bool) -> dict:
    peak_reserved = 20.0 + core / 16.0 if passed else 38.0
    brainlm_identity = _brainlm_identity()
    value = {
        "schema": BENCHMARK_RECORD_SCHEMA,
        "gate_scope": "exact-full-connect4-per-rank-real-training-step-memory",
        "passed": passed,
        "failure_kind": None if passed else "CUDA_OUT_OF_MEMORY",
        "failure": None if passed else "synthetic CUDA OOM fixture",
        "profile": profile,
        "shape_bctdhw": [1, 1, 128, *PROFILE_SHAPES[profile]],
        "depth_slab_size": core,
        "derived_depth_halo": 14,
        "source_tree_sha256": SOURCE_SHA,
        "source_file_count": SOURCE_COUNT,
        "architecture_contract_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "base_config_file_sha256": ("b" if profile == "paper" else "c") * 64,
        "benchmark_model_config_sha256": benchmark_model_config_identity(
            _config(profile, core), profile
        ),
        "brainlm_identity": brainlm_identity,
        "brainlm_identity_sha256": brainlm_identity["record_sha256"],
        "training_batch_identity": _training_batch_identity(),
        "scheduler_alphas_cumprod_sha256": "d" * 64,
        "composite_loss_configured": True,
        "all_roi_masks_configured": True,
        "composite_loss_completed": passed,
        "four_gpu_ddp_smoke_required": True,
        "four_gpu_ddp_smoke_attested": False,
        "loss_weights": {
            "voxel": 1.0,
            "ssim": 0.5,
            "fc": 0.3,
            "temporal": 0.2,
            "perceptual": 0.1,
            "volume": 0.5,
            "region_hist": 0.5,
        },
        "amp": {
            "enabled": True,
            "dtype": "bfloat16",
            "gradient_scaler_enabled": False,
            "parameter_dtype": "float32",
        },
        "batch_contract": {
            "production_global_batch_size": 4,
            "executed_world_size": 1,
            "executed_local_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "ddp_wrapped": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "DDIM",
            "num_diffusion_steps": 1000,
            "beta_schedule": "linear",
            "num_inference_steps": 50,
            "eta": 0.0,
            "training_objective": "epsilon",
        },
        "execution": {
            "slurm_job_id": "12345",
            "slurm_partition": "cscc-gpu-p",
            "slurm_qos": "cscc-gpu-qos",
            "slurm_num_nodes": 1,
            "allocated_gpu_count": 1,
            "hostname": "compute-fixture",
            "allocated_hostnames": ["compute-fixture"],
            "cuda_visible_devices": "0",
            "cuda_device_count": 1,
            "gpu_uuid": "GPU-memory-fixture",
            "nvidia_smi_name": "NVIDIA A100-SXM4-40GB",
            "nvidia_smi_memory_mib": 40960,
            "nvidia_driver_version": "550.54.15",
            "python_executable": "/fixture/qualified-connect4-v10/bin/python3.10",
            "python_executable_sha256": "2" * 64,
            "isolated_python": True,
            "scontrol_record_sha256": scheduler_allocation_sha256(
                slurm_job_id="12345",
                slurm_partition="cscc-gpu-p",
                slurm_qos="cscc-gpu-qos",
                allocated_hostnames=["compute-fixture"],
                slurm_num_nodes=1,
                allocated_gpu_count=1,
            ),
        },
        "gpu": "NVIDIA A100-SXM4-40GB",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "cudnn_version": 90100,
        "gpu_total_memory_gib": 40.0,
        "safety_margin_gib": 4.0,
        "safe_reserved_limit_gib": 36.0,
        "peak_allocated_gib": peak_reserved - 1.0,
        "peak_reserved_gib": peak_reserved,
        "forward_seconds": 1.0 if passed else None,
        "backward_seconds": 2.0 if passed else None,
        "step_seconds": 3.0 if passed else None,
        "adamw_step_included": passed,
        "gradient_clip_included": passed,
        "required_gradient_groups_finite": passed,
        "required_gradient_groups_nonzero": passed,
        "full_target_blind_inference_completed": passed,
        "inference_autocast_dtype": "bfloat16",
        "inference_num_steps": 50,
        "inference_seconds": 4.0 if passed else None,
        "inference_peak_reserved_gib": peak_reserved if passed else 0.0,
    }
    return seal_benchmark_record(value)


def _profile_summary(profile: str, records: list[dict]) -> dict:
    safe = [record for record in records if record["passed"]]
    selected = max(safe, key=lambda record: record["depth_slab_size"])
    bindings = [
        {
            "order": order,
            "profile": profile,
            "depth_slab_size": record["depth_slab_size"],
            "filename": (
                f"figure1_v10_{profile}_slab{record['depth_slab_size']}.json"
            ),
            "file_sha256": f"{order + 4:x}" * 64,
            "canonical_record_sha256": record["canonical_record_sha256"],
            "slurm_job_id": record["execution"]["slurm_job_id"],
        }
        for order, record in enumerate(records)
    ]
    return {
        "attempted_core_sizes": [record["depth_slab_size"] for record in records],
        "largest_safe_core_size": selected["depth_slab_size"],
        "selected_peak_allocated_gib": selected["peak_allocated_gib"],
        "selected_peak_reserved_gib": selected["peak_reserved_gib"],
        "selected_forward_seconds": selected["forward_seconds"],
        "selected_backward_seconds": selected["backward_seconds"],
        "selected_step_seconds": selected["step_seconds"],
        "memory_gate_passed": True,
        "four_gpu_ddp_smoke_attested": False,
        "release_stop": True,
        "record_order_sha256": canonical_sha256(bindings),
        "record_bindings": bindings,
        "records": records,
    }


def _paper_block() -> dict:
    return {
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


def _summary_from_records(recovery: list[dict]) -> dict:
    profiles = {
        "recovery": _profile_summary("recovery", recovery),
        "paper": _paper_block(),
    }
    all_bindings = [
        binding
        for profile in ("recovery", "paper")
        for binding in profiles[profile]["record_bindings"]
    ]
    return seal_sweep_summary(
        {
            "schema": SWEEP_SUMMARY_SCHEMA,
            "passed": False,
            "minimum_safety_margin_gib": 4.0,
            "selection_rule": "largest safe exact full-step core",
            "four_gpu_ddp_smoke_required": True,
            "four_gpu_ddp_smoke_attested": False,
            "record_file_order_sha256": canonical_sha256(all_bindings),
            "paper_profile_release_stop": True,
            "profiles": profiles,
        }
    )


def _summary() -> dict:
    recovery = [_record("recovery", core, passed=core < 8) for core in (1, 2, 4, 8)]
    return _summary_from_records(recovery)


def _ddp_attestation(selected: dict, summary_sha256: str) -> dict:
    synchronized_state = "8" * 64
    record = {
        "schema": DDP_SMOKE_SCHEMA,
        "gate_scope": "exact-full-connect4-four-rank-ddp-real-training-step",
        "passed": True,
        "profile": selected["profile"],
        "shape_bctdhw": [1, 1, 128, *PROFILE_SHAPES[selected["profile"]]],
        "depth_slab_size": selected["depth_slab_size"],
        "derived_depth_halo": 14,
        "source_file_count": SOURCE_COUNT,
        "memory_sweep_summary_sha256": summary_sha256,
        "selected_memory_record_sha256": selected["canonical_record_sha256"],
        **{
            key: selected[key]
            for key in (
                "source_tree_sha256",
                "architecture_contract_sha256",
                "base_config_file_sha256",
                "benchmark_model_config_sha256",
                "brainlm_identity",
                "brainlm_identity_sha256",
                "scheduler_alphas_cumprod_sha256",
                "loss_weights",
                "amp",
                "optimizer",
                "scheduler",
            )
        },
        "initial_model_state_sha256": "7" * 64,
        "batch_contract": {
            "global_batch_size": 4,
            "world_size": 4,
            "local_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "ddp_wrapped": True,
        },
        "ddp_gradient_reduction_attested": True,
        "all_ranks_post_step_state_equal": True,
        "synchronized_post_step_model_state_sha256": synchronized_state,
        "execution": {
            "slurm_job_id": "54321",
            "slurm_partition": "cscc-gpu-p",
            "slurm_qos": "cscc-gpu-qos",
            "slurm_num_nodes": 1,
            "allocated_gpu_count": 4,
            "allocated_hostnames": ["compute-fixture"],
            "hostname": "compute-fixture",
            "cuda_visible_devices": "0,1,2,3",
            "cuda_device_count": 4,
            "scontrol_record_sha256": scheduler_allocation_sha256(
                slurm_job_id="54321",
                slurm_partition="cscc-gpu-p",
                slurm_qos="cscc-gpu-qos",
                allocated_hostnames=["compute-fixture"],
                slurm_num_nodes=1,
                allocated_gpu_count=4,
            ),
            "python_executable": "/fixture/qualified-connect4-v10/bin/python3.10",
            "python_executable_sha256": "2" * 64,
            "isolated_python": True,
            "torch_version": selected["torch_version"],
            "cuda_version": selected["cuda_version"],
            "cudnn_version": selected["cudnn_version"],
        },
        "rank_records": [
            {
                "rank": rank,
                "world_size": 4,
                "local_batch_size": 1,
                "gpu_uuid": f"GPU-ddp-fixture-{rank}",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "cuda_visible_device": str(rank),
                "torch_cuda_device_index": rank,
                "nvidia_driver_version": "550.54.15",
                "gpu_total_memory_gib": 40.0,
                "peak_allocated_gib": 30.0,
                "peak_reserved_gib": 31.0,
                "forward_seconds": 1.0,
                "backward_seconds": 2.0,
                "step_seconds": 3.0,
                "composite_loss_completed": True,
                "required_gradient_groups_finite": True,
                "required_gradient_groups_nonzero": True,
                "gradient_clip_included": True,
                "adamw_step_included": True,
                "post_step_model_state_sha256": synchronized_state,
                "training_batch_identity": _training_batch_identity(
                    rank, admission_world_size=4
                ),
            }
            for rank in range(4)
        ],
    }
    record["global_training_batch_identity_sha256"] = canonical_sha256(
        [rank["training_batch_identity"] for rank in record["rank_records"]]
    )
    return seal_benchmark_record(record)


def _write_gate_pair(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    summary = validate_sweep_summary(_summary())
    selected = next(
        record
        for record in summary["profiles"]["recovery"]["records"]
        if record["depth_slab_size"] == 4
    )
    summary_path = tmp_path / "figure1_v10_sweep_summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    summary_sha256 = sha256_file(summary_path)
    ddp_path = tmp_path / "figure1_v10_ddp.json"
    ddp_path.write_text(
        json.dumps(_ddp_attestation(selected, summary_sha256)) + "\n",
        encoding="utf-8",
    )
    config = _config("recovery", 4)
    config["models"]["unet"].update(
        {
            "depth_slab_sweep_summary_path": str(summary_path),
            "depth_slab_sweep_summary_sha256": summary_sha256,
            "depth_slab_ddp_smoke_path": str(ddp_path),
            "depth_slab_ddp_smoke_sha256": sha256_file(ddp_path),
        }
    )
    return config, summary, summary_path, ddp_path


def test_sweep_selects_largest_safe_and_requires_matching_four_gpu_ddp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, summary, _, _ = _write_gate_pair(tmp_path)
    monkeypatch.setattr(
        gpu_gate, "source_tree_identity", lambda _root: (SOURCE_SHA, SOURCE_COUNT)
    )
    monkeypatch.setattr(
        brainlm_context,
        "configured_brainlm_identity",
        lambda _spec: _brainlm_identity(),
    )
    selection = authenticate_configured_depth_slab_selection(config)
    assert selection["depth_slab_size"] == 4
    assert summary["profiles"]["paper"]["status"] == PAPER_PROFILE_BLOCK_STATUS
    assert summary["profiles"]["paper"]["records"] == []

    config["models"]["unet"]["depth_slab_size"] = 2
    with pytest.raises(RuntimeError, match="not the authenticated largest-safe"):
        authenticate_configured_depth_slab_selection(config)


def test_one_gpu_sweep_alone_cannot_clear_release_stop(tmp_path: Path):
    config, _, _, _ = _write_gate_pair(tmp_path)
    config["models"]["unet"]["depth_slab_ddp_smoke_path"] = None
    config["models"]["unet"]["depth_slab_ddp_smoke_sha256"] = None
    with pytest.raises(RuntimeError, match="separate four-GPU DDP"):
        authenticate_configured_depth_slab_selection(config)


def test_gate_rejects_source_or_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _, _, _ = _write_gate_pair(tmp_path)
    monkeypatch.setattr(
        brainlm_context,
        "configured_brainlm_identity",
        lambda _spec: _brainlm_identity(),
    )
    monkeypatch.setattr(
        gpu_gate, "source_tree_identity", lambda _root: ("9" * 64, SOURCE_COUNT)
    )
    with pytest.raises(RuntimeError, match="current source"):
        authenticate_configured_depth_slab_selection(config)

    monkeypatch.setattr(
        gpu_gate, "source_tree_identity", lambda _root: (SOURCE_SHA, SOURCE_COUNT)
    )
    config["models"]["dit"]["spatial_window_size"] = [9, 9, 9]
    with pytest.raises(RuntimeError, match="config differs"):
        authenticate_configured_depth_slab_selection(config)


def test_sweep_rejects_runtime_drift_between_core_measurements():
    recovery = [_record("recovery", core, passed=core < 8) for core in (1, 2, 4, 8)]
    recovery[1]["execution"] = dict(recovery[1]["execution"])
    recovery[1]["execution"]["nvidia_driver_version"] = "551.00"
    recovery[1] = seal_benchmark_record(recovery[1])
    with pytest.raises(ValueError, match="runtime identity drifted"):
        validate_sweep_summary(_summary_from_records(recovery))


def test_ddp_attestation_rejects_false_scheduler_and_rank_gpu_mapping():
    selected = _record("recovery", 4, passed=True)
    record = _ddp_attestation(selected, "f" * 64)
    record["execution"] = dict(record["execution"])
    record["execution"]["allocated_gpu_count"] = 1
    with pytest.raises(ValueError, match="scheduler allocation differs"):
        validate_ddp_smoke_attestation(seal_benchmark_record(record))

    record = _ddp_attestation(selected, "f" * 64)
    record["rank_records"] = [dict(rank) for rank in record["rank_records"]]
    record["rank_records"][1]["cuda_visible_device"] = "0"
    with pytest.raises(ValueError, match="rank-to-GPU mapping differs"):
        validate_ddp_smoke_attestation(seal_benchmark_record(record))

    record = _ddp_attestation(selected, "f" * 64)
    record["execution"] = dict(record["execution"])
    record["execution"]["scontrol_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scheduler-allocation digest differs"):
        validate_ddp_smoke_attestation(seal_benchmark_record(record))


def test_live_scheduler_parser_requires_all_gpu_counts_to_agree():
    from scripts.benchmark_figure1_v10_gpu import (
        _require_slurm_allocation_shape,
    )

    _require_slurm_allocation_shape(
        "JobId=123 NumNodes=1 TresPerNode=gres:gpu:a100:1 "
        "AllocTRES=cpu=8,gres/gpu=1",
        {"NumNodes": "1"},
        expected_nodes=1,
        expected_gpus=1,
    )
    with pytest.raises(RuntimeError, match="exactly 1 allocated GPU"):
        _require_slurm_allocation_shape(
            "JobId=123 NumNodes=1 TresPerNode=gres:gpu:a100:1 "
            "AllocTRES=cpu=8,gres/gpu=4",
            {"NumNodes": "1"},
            expected_nodes=1,
            expected_gpus=1,
        )


def test_gate_rejects_ddp_driver_drift_from_memory_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, _, _, ddp_path = _write_gate_pair(tmp_path)
    ddp = json.loads(ddp_path.read_text(encoding="utf-8"))
    ddp["rank_records"] = [dict(rank) for rank in ddp["rank_records"]]
    for rank in ddp["rank_records"]:
        rank["nvidia_driver_version"] = "551.00"
    ddp_path.write_text(
        json.dumps(seal_benchmark_record(ddp)) + "\n", encoding="utf-8"
    )
    config["models"]["unet"]["depth_slab_ddp_smoke_sha256"] = sha256_file(
        ddp_path
    )
    monkeypatch.setattr(
        gpu_gate, "source_tree_identity", lambda _root: (SOURCE_SHA, SOURCE_COUNT)
    )
    with pytest.raises(RuntimeError, match="NVIDIA driver differs"):
        authenticate_configured_depth_slab_selection(config)


def test_sweep_rejects_early_stop_tampering_and_unsafe_measurements():
    summary = _summary()
    summary["profiles"]["paper"]["status"] = "FALSE_PAPER_PASS"
    summary = seal_sweep_summary(summary)
    with pytest.raises(ValueError):
        validate_sweep_summary(summary)

    record = _record("recovery", 1, passed=True)
    record["safety_margin_gib"] = 1.0
    record = seal_benchmark_record(record)
    with pytest.raises(ValueError, match="below four GiB"):
        validate_benchmark_record(record, expected_profile="recovery", expected_core=1)

    record = _record("recovery", 1, passed=True)
    record["required_gradient_groups_nonzero"] = False
    record = seal_benchmark_record(record)
    with pytest.raises(ValueError, match="zero gradient"):
        validate_benchmark_record(record, expected_profile="recovery", expected_core=1)

    with pytest.raises(ValueError, match=PAPER_PROFILE_BLOCK_STATUS):
        validate_benchmark_record(
            _record("paper", 1, passed=True),
            expected_profile="paper",
            expected_core=1,
        )


def _resign_training_batch(record: dict) -> dict:
    batch = record["training_batch_identity"]
    batch["record_sha256"] = canonical_sha256(
        {key: value for key, value in batch.items() if key != "record_sha256"}
    )
    return seal_benchmark_record(record)


def test_real_batch_gate_rejects_context_authority_and_target_tampering():
    record = deepcopy(_record("recovery", 1, passed=True))
    context = record["training_batch_identity"][
        "brainlm_batch_context_identity"
    ]
    context["support_tensor_sha256"] = [_fixture_sha("forged-support")]
    with pytest.raises(ValueError, match="batch-context identity is invalid"):
        validate_benchmark_record(
            _resign_training_batch(record),
            expected_profile="recovery",
            expected_core=1,
        )

    record = deepcopy(_record("recovery", 1, passed=True))
    context = record["training_batch_identity"][
        "brainlm_batch_context_identity"
    ]
    context["cohort_admission_identity_record_sha256"] = _fixture_sha(
        "different-cohort"
    )
    context["record_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="cohort/run authority"):
        validate_benchmark_record(
            _resign_training_batch(record),
            expected_profile="recovery",
            expected_core=1,
        )

    record = deepcopy(_record("recovery", 1, passed=True))
    target = record["training_batch_identity"]["target_tensor_identity"]
    target["maximum"] = 1.01
    with pytest.raises(ValueError, match="target tensor contract differs"):
        validate_benchmark_record(
            _resign_training_batch(record),
            expected_profile="recovery",
            expected_core=1,
        )

    record = deepcopy(_record("recovery", 1, passed=True))
    record["training_batch_identity"]["synthetic_inputs_used"] = True
    with pytest.raises(ValueError, match="admitted unsealed training scans"):
        validate_benchmark_record(
            _resign_training_batch(record),
            expected_profile="recovery",
            expected_core=1,
        )


def test_oom_record_remains_bound_to_preflighted_real_brainlm_context():
    record = _record("recovery", 1, passed=False)
    validated = validate_benchmark_record(
        record, expected_profile="recovery", expected_core=1
    )
    batch = validated["training_batch_identity"]
    assert batch["synthetic_inputs_used"] is False
    assert batch["admitted_unsealed_training_target_loaded"] is True
    assert batch["brainlm_batch_context_identity"]["support_tensor_sha256"]


def test_atomic_gpu_artifacts_are_no_replace_and_reject_symlink_parent(tmp_path: Path):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "record.json"
    atomic_write_json_no_replace(path, {"record": 1})
    with pytest.raises(FileExistsError):
        atomic_write_json_no_replace(path, {"record": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"record": 1}

    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent is unsafe"):
        atomic_write_json_no_replace(linked_parent / "record.json", {"record": 1})



