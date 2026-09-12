import csv
import numpy as np
import pytest
import torch
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache

from architecture_contract import (
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    synthesis_architecture_contract,
)
from data.protocol import (
    CONDITIONING_ARTIFACT_SCHEMA,
    RUN_ARTIFACT_IDENTITY_SCHEMA,
    TRAINING_CHECKPOINT_FORMAT,
    build_split_identity,
    load_completed_synthesis_checkpoint,
)
from data.cohort_admission import (
    COHORT_ADMISSION_IDENTITY_SCHEMA,
    build_validation_shard_contract,
    validate_production_training_runtime_identity,
)
from data.provenance import canonical_sha256
from eval.development_quality import (
    QUALITY_CHECK_NAMES,
    SELECTION_TIER,
    checkpoint_candidate_identity,
    evaluated_tensor_set_identity,
    model_state_sha256,
    seal_development_quality_record,
)
from eval.quality import DEVELOPMENT_REQUIRED_CHECKS, POLICY_NOTE, QualityPolicy
from models import brainlm_context
from utils.config import PAPER_PROTOCOL_PROFILE
from eval.downstream import (
    DOWNSTREAM_FOLD_ALLOCATOR_ID,
    DOWNSTREAM_GAT_SETTINGS_FORMAT,
    DOWNSTREAM_METRICS,
    _build_gat_settings_identity,
    compare_downstream_gat_results,
    evaluate_downstream_gat,
    five_fold_downstream_splits,
)


def _example_data():
    patient_ids = np.repeat([f"P{i:02d}" for i in range(30)], 2)
    labels = np.repeat([i % 3 for i in range(30)], 2)
    cohorts = np.repeat(["OASIS-3" if i % 2 else "ANMerge" for i in range(30)], 2)
    scan_ids = np.asarray([f"SCAN-{index:03d}" for index in range(len(labels))])
    return labels, patient_ids, cohorts, scan_ids


def test_downstream_settings_identity_names_custom_fold_allocator():
    settings = _build_gat_settings_identity(
        seed=4,
        hidden_channels=8,
        heads=2,
        dropout=0.1,
        epochs=3,
        batch_size=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        resolved_device="cpu",
    )
    assert settings["format"] == DOWNSTREAM_GAT_SETTINGS_FORMAT
    assert settings["format"] == "connect4_downstream_gat_settings_v2"
    assert settings["fold_allocator"] == DOWNSTREAM_FOLD_ALLOCATOR_ID
    assert settings["fold_allocator"] == (
        "connect4_seeded_greedy_single_label_patient_group_scan_balance_5fold_v1"
    )
    assert "splitter" not in settings
    payload = dict(settings)
    claimed_sha256 = payload.pop("sha256")
    assert claimed_sha256 == canonical_sha256(payload)


def _brainlm_identity():
    authority = {
        "schema": brainlm_context.BRAINLM_AUTHORITY_SCHEMA,
        "file_sha256": "4" * 64,
        "size_bytes": 1234,
        "authority_record_sha256": "5" * 64,
        "native_preprocessing_source_sha256": (
            brainlm_context.CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "ordered_scan_ids": ["B100_001"],
        "ordered_scan_ids_sha256": canonical_sha256(["B100_001"]),
        "scan_count": 1,
        "all_scans_explicitly_structurally_reviewed": True,
        "dense_nonlinear_pull_field_required": True,
        "affine_only_fallback_forbidden": True,
    }
    authority["content_record_sha256"] = canonical_sha256(authority)
    identity = {
        "schema": "connect4-brainlm-a424-perceptual-identity-v2",
        "name": "brainlm",
        "model_artifacts": brainlm_context._expected_official_content_identity(),
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
    return brainlm_context.validate_configured_brainlm_identity(identity)


def _runtime_identity():
    scheduler_digest = canonical_sha256(
        {
            "schema": "connect4-ciai-scontrol-allocation-identity-v1",
            "slurm_job_id": "12345",
            "slurm_partition": "cscc-gpu-p",
            "slurm_qos": "cscc-gpu-qos",
            "allocated_hostnames": ["compute01"],
            "slurm_num_nodes": 1,
            "allocated_gpu_count": 4,
        }
    )
    runtime = {
        "schema": "connect4-v10-ciai-four-a100-training-runtime-v1",
        "synthesis_architecture_schema": SYNTHESIS_ARCHITECTURE_SCHEMA,
        "synthesis_architecture_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "training_checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "slurm_job_id": "12345",
        "slurm_partition": "cscc-gpu-p",
        "slurm_qos": "cscc-gpu-qos",
        "allocated_hostnames": ["compute01"],
        "hostname": "compute01",
        "scontrol_record_sha256": scheduler_digest,
        "source_tree_sha256": "9" * 64,
        "source_file_count": 100,
        "python_executable": "/fixture/qualified-connect4-v10/bin/python3.10",
        "python_executable_sha256": "a" * 64,
        "isolated_python": True,
        "torch_version": "2.7.0",
        "cuda_version": "12.8",
        "cudnn_version": 90100,
        "amp_dtype": "bfloat16",
        "world_size": 4,
        "gpu_records": [
            {
                "rank": rank,
                "local_rank": rank,
                "gpu_uuid": f"GPU-{rank:04d}",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_gib": 39.5,
                "nvidia_driver_version": "570.00",
                "cuda_visible_device": str(rank),
            }
            for rank in range(4)
        ],
        "slab_selection": {
            "profile": "recovery",
            "depth_slab_size": 4,
            "summary_path": "/evidence/sweep.json",
            "summary_sha256": "b" * 64,
            "ddp_smoke_path": "/evidence/ddp.json",
            "ddp_smoke_sha256": "c" * 64,
            "measured_ddp_step_seconds": 12.5,
            "recovery_half_epoch_eta_seconds": 6487.5,
        },
    }
    runtime["canonical_sha256"] = canonical_sha256(runtime)
    return validate_production_training_runtime_identity(runtime)


def _write_unseen_manifest(tmp_path, scan_ids, patient_ids, cohorts):
    path = tmp_path / "downstream-cohort.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["scan_id", "patient_id", "cohort"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "scan_id": scan_id,
                "patient_id": patient_id,
                "cohort": cohort,
            }
            for scan_id, patient_id, cohort in zip(
                scan_ids, patient_ids, cohorts
            )
        )
    return path


@lru_cache(maxsize=4)
def _paper_checkpoint_split(train_patient):
    scans = [f"A_{index:05d}" for index in range(6290)] + [
        f"D_{index:05d}" for index in range(700)
    ]
    cohorts = {
        scan: ("A4" if scan.startswith("A_") else "ADNI") for scan in scans
    }
    patients = {scan: f"SYNTH-{scan}" for scan in scans}
    patients[scans[0]] = train_patient
    return build_split_identity(
        scans,
        list(range(6000)),
        list(range(6000, 6590)),
        list(range(6590, len(scans))),
        cohorts,
        patients,
    )


def _write_completed_checkpoint(tmp_path, *, train_patient="TRAIN-A", mutate=None):
    split_identity = deepcopy(_paper_checkpoint_split(train_patient))
    authenticated_targets = sorted(
        record["scan_id"]
        for split_name in ("train", "validation")
        for record in split_identity["partitions"][split_name]
    )
    sealed_targets = sorted(
        record["scan_id"] for record in split_identity["partitions"]["test"]
    )
    target_access = {
        "schema": "connect4-training-target-access-v1",
        "authenticated_target_scan_ids": authenticated_targets,
        "authenticated_target_scan_ids_sha256": canonical_sha256(
            authenticated_targets
        ),
        "sealed_target_scan_ids": sealed_targets,
        "sealed_target_scan_ids_sha256": canonical_sha256(sealed_targets),
        "sealed_targets_opened": False,
    }
    conditioning_identity = {
        "format": CONDITIONING_ARTIFACT_SCHEMA,
        "brainiac": {"checkpoint_sha256": "1" * 64},
        "modernbert": {"fingerprint_sha256": "2" * 64},
        "roi_feature_extractor": {"fingerprint_sha256": "3" * 64},
    }
    spatial_authority = {
        "profile": "paper-common-grid",
        "common_grid_contract_sha256": "c" * 64,
    }
    quality_policy = asdict(QualityPolicy())
    config = {
        "data": {
            "protocol_profile": PAPER_PROTOCOL_PROFILE,
            "architecture_shape": [128, 128, 128],
            "num_frames": 128,
            "tr_seconds": 3.0,
        },
        "training": {
            "epochs": 200,
            "val_batches": 4,
            "development_quality_gate": {"policy": quality_policy},
        },
    }
    artifact_identity = {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "sha256": "a" * 64,
        "scaler_identity_sha256": "b" * 64,
        "common_grid_contract_sha256": "c" * 64,
        "native_alignment_authority_sha256": None,
        "spatial_authority": spatial_authority,
        "spatial_authority_sha256": canonical_sha256(spatial_authority),
        "conditioning_identity": conditioning_identity,
        "conditioning_identity_sha256": canonical_sha256(
            conditioning_identity
        ),
        "evaluation_extractor": {
            "status": "deferred-unopened-by-training",
            "purpose": "final-test-only",
        },
        "perceptual_extractor": _brainlm_identity(),
        "target_access_contract": target_access,
        "target_artifact_identities_sha256": "d" * 64,
        "structural_artifact_identities_sha256": "e" * 64,
        "num_authenticated_targets": len(authenticated_targets),
        "num_sealed_targets_unopened": len(sealed_targets),
        "num_scans": 6990,
    }
    runtime_identity = _runtime_identity()
    admission_identity = {
        "schema": COHORT_ADMISSION_IDENTITY_SCHEMA,
        "authority_record_sha256": "0" * 64,
        "authority_canonical_bytes_sha256": "1" * 64,
        "data_root_sha256": "2" * 64,
        "runtime_identity_sha256": runtime_identity["canonical_sha256"],
        "config_sha256": canonical_sha256(config),
        "ordered_scan_ids_sha256": "3" * 64,
        "partitions_sha256": "4" * 64,
        "split_identity_sha256": canonical_sha256(split_identity),
        "artifact_identity_sha256": canonical_sha256(artifact_identity),
        "structural_artifact_identities_sha256": artifact_identity[
            "structural_artifact_identities_sha256"
        ],
        "dataset_state_sha256": "5" * 64,
        "world_size": 4,
    }
    admission_identity["record_sha256"] = canonical_sha256(admission_identity)
    validation_shard_contract = build_validation_shard_contract(
        admission_identity,
        development_partition_size=590,
        global_limit=4,
        world_size=4,
    )
    development_records = split_identity["partitions"]["validation"]
    selection_validation_shard_contract = build_validation_shard_contract(
        admission_identity,
        development_partition_size=len(development_records),
        global_limit=len(development_records),
        world_size=4,
    )
    model_state = {"weight": torch.ones(1)}
    model_digest = model_state_sha256(model_state)
    candidate = checkpoint_candidate_identity(
        model_state_digest=model_digest,
        completed_batches=200,
        next_epoch=200,
        next_batch_index=0,
        partial=False,
    )
    checks = {
        name: {"passed": True, "value": 1.0, "policy": "fixture"}
        for name in QUALITY_CHECK_NAMES
    }
    affine = np.diag([3.0, 3.0, 3.0, 1.0]).tolist()
    geometry_shape = [128, 128, 128, 128]
    checks["grid_affine_tr_contract"]["value"] = {
        "shape": geometry_shape,
        "affine_max_abs_difference": 0.0,
        "tr_seconds": 3.0,
    }
    checks["near_static_voxel_fraction"]["value"] = 0.0
    checks["structured_temporal_available"]["value"] = 4
    rows = []
    for position, scan_record in enumerate(development_records):
        scan_id = scan_record["scan_id"]
        source_identity = {
            "schema": "connect4-development-source-identity-v1",
            "scan_id": scan_id,
            "scan_role": "development-validation",
            "brainlm_target_artifact_identity_sha256": canonical_sha256(
                ["target-artifact", scan_id]
            ),
            "brainlm_scan_context_record_sha256": canonical_sha256(
                ["scan-context", scan_id]
            ),
            "brainlm_dataset_state_record_sha256": "5" * 64,
            "brainlm_dataset_state_sha256": "6" * 64,
            "brainlm_support_tensor_sha256": canonical_sha256(
                ["support", scan_id]
            ),
            "brainlm_prepared_mask_sha256": canonical_sha256(
                ["prepared-mask", scan_id]
            ),
            "brainlm_padded_mask_artifact_descriptor_sha256": canonical_sha256(
                ["padded-mask-descriptor", scan_id]
            ),
            "brainlm_structural_source_identity_sha256": canonical_sha256(
                ["structural-source", scan_id]
            ),
            "brainlm_native_preprocessing_source_sha256": "7" * 64,
            "brainlm_native_alignment_authority_sha256": "8" * 64,
            "roi_mapping_sha256": canonical_sha256(["roi-mapping"]),
            "padded_shape": [128, 128, 128],
            "native_shape": [128, 128, 128],
            "padding_before": [0, 0, 0],
            "padding_after": [0, 0, 0],
            "padded_affine_ras_mm": deepcopy(affine),
            "padded_affine_sha256": canonical_sha256(affine),
            "native_affine_ras_mm": deepcopy(affine),
            "native_affine_sha256": canonical_sha256(affine),
        }
        source_identity["record_sha256"] = canonical_sha256(source_identity)
        tensor_shape = [128, 128, 128, 128]
        target_identity = {
            "schema": "connect4-development-tensor-identity-v1",
            "kind": "target",
            "sha256": canonical_sha256(["target", scan_id]),
            "dtype": "torch.float32",
            "shape": tensor_shape,
            "finite": True,
            "in_unit_interval": True,
        }
        prediction_identity = {
            "schema": "connect4-development-tensor-identity-v1",
            "kind": "prediction",
            "sha256": canonical_sha256(["prediction", scan_id]),
            "dtype": "torch.float32",
            "shape": tensor_shape,
            "finite": True,
            "in_unit_interval": True,
        }
        brain_mask_identity = {
            "schema": "connect4-development-tensor-identity-v1",
            "kind": "brain_mask",
            "sha256": canonical_sha256(["brain-mask", scan_id]),
            "dtype": "torch.float32",
            "shape": [128, 128, 128],
            "finite": True,
            "binary": True,
            "positive_voxels": 128**3,
            "native_binary_sha256": source_identity[
                "brainlm_support_tensor_sha256"
            ],
            "padded_binary_sha256": source_identity[
                "brainlm_support_tensor_sha256"
            ],
            "source_support_tensor_sha256": source_identity[
                "brainlm_support_tensor_sha256"
            ],
            "padding_lineage_sha256": canonical_sha256(
                {
                    "padded_shape": [128, 128, 128],
                    "native_shape": [128, 128, 128],
                    "padding_before": [0, 0, 0],
                    "padding_after": [0, 0, 0],
                }
            ),
        }
        target_validity_mask_identity = {
            "schema": "connect4-development-tensor-identity-v1",
            "kind": "target_validity_mask",
            "sha256": canonical_sha256(["target-validity-mask", scan_id]),
            "dtype": "torch.float32",
            "shape": [128, 128, 128],
            "finite": True,
            "binary": True,
            "positive_voxels": 128**3,
            "binary_sha256": brain_mask_identity["native_binary_sha256"],
            "brain_mask_binary_sha256": brain_mask_identity[
                "native_binary_sha256"
            ],
            "outside_brain_voxels": 0,
            "target_tensor_sha256": target_identity["sha256"],
            "derivation_contract": "connect4-observed-bold-support-v1",
        }
        roi_masks_identity = {
            "schema": "connect4-development-tensor-identity-v1",
            "kind": "roi_masks",
            "sha256": canonical_sha256(["roi-masks", scan_id]),
            "dtype": "torch.float32",
            "shape": [4, 128, 128, 128],
            "finite": True,
            "binary": True,
            "binary_sha256": canonical_sha256(["roi-binary", scan_id]),
            "channel_binary_sha256": [
                canonical_sha256(["roi-binary", scan_id, label])
                for label in range(1, 5)
            ],
            "roi_union_binary_sha256": canonical_sha256(
                ["roi-union", scan_id]
            ),
            "brain_mask_binary_sha256": brain_mask_identity[
                "native_binary_sha256"
            ],
            "nonoverlapping": True,
            "outside_brain_voxels": 0,
            "positive_voxels": 128**3,
            "nonempty_roi_labels": [1, 2, 3, 4],
            "channel_voxel_counts": [128**3 // 4] * 4,
            "source_prepared_mask_sha256": source_identity[
                "brainlm_prepared_mask_sha256"
            ],
            "source_padded_mask_descriptor_sha256": source_identity[
                "brainlm_padded_mask_artifact_descriptor_sha256"
            ],
            "source_structural_identity_sha256": source_identity[
                "brainlm_structural_source_identity_sha256"
            ],
            "roi_mapping_sha256": source_identity["roi_mapping_sha256"],
        }
        evaluated_tensors = evaluated_tensor_set_identity(
            target=target_identity,
            prediction=prediction_identity,
            brain_mask=brain_mask_identity,
            target_validity_mask=target_validity_mask_identity,
            roi_masks=roi_masks_identity,
        )
        quality = {
            "schema": "connect4-paired-4d-quality-v1",
            "policy_note": POLICY_NOTE,
            "policy": quality_policy,
            "implementation_sources": {
                "quality_evaluator": {
                    "relative_path": "eval/quality.py",
                    "path": "/fixture/eval/quality.py",
                    "size_bytes": 1,
                    "sha256": "9" * 64,
                }
            },
            "sources": {
                "array_identity": source_identity,
                "evaluated_tensor_set_identity": evaluated_tensors,
            },
            "geometry": {
                "shape": geometry_shape,
                "tr_seconds": 3.0,
                "affine": deepcopy(affine),
                "affine_max_abs_difference": 0.0,
                "canonical_axis_codes": ["R", "A", "S"],
                "original_axis_codes": {
                    "real": ["R", "A", "S"],
                    "predicted": ["R", "A", "S"],
                },
                "voxel_sizes_mm": [3.0, 3.0, 3.0],
                "mask_voxels": 128**3,
                "mask_derived_from_real": False,
                "array_entry_point": True,
                "resampling_performed": False,
            },
            "support": {},
            "outside_mask": {},
            "spatial": {
                "high_frequency_rms_ratio": 1.0,
                "temporal_mean_high_frequency_correlation": 1.0,
                "dynamic_high_frequency_rms_ratio": 1.0,
                "dynamic_high_frequency_correlation": 1.0,
            },
            "temporal": {
                "temporal_variance_ratio": 1.0,
                "dvars_ratio": 1.0,
                "dynamic_power_ratio": 1.0,
                "effective_rank_ratio": 1.0,
                "near_static_voxel_fraction": 0.0,
            },
            "structured_temporal": {
                "required": True,
                "roi_labels_supplied": True,
                "phase_invariant": True,
                "roi_count": 4,
                "roi_labels": [1, 2, 3, 4],
                "roi_voxel_counts": {
                    str(label): 128**3 // 4 for label in range(1, 5)
                },
                "power_spectrum_non_dc_bins": 64,
                "issues": [],
                "temporally_valid_real_roi_count": 4,
                "excluded_constant_real_roi_labels": [],
                "roi_mean_time_series_shape": [4, 128],
                "available": True,
                "unavailable_reason": None,
                "constant_predicted_roi_labels": [],
                "fc_upper_triangle_edges": 6,
                "fc_matrix_correlation": 1.0,
                "power_spectrum_correlation": 1.0,
            },
            "paper_metrics": {},
            "checks": checks,
            "failed_checks": [],
            "failed_temporal_checks": [],
            "failed_temporal_collapse_checks": [],
            "failed_structured_temporal_checks": [],
            "temporal_quality_mismatch_detected": False,
            "temporal_collapse_detected": False,
            "passed": True,
            "verdict": "pass",
        }
        rows.append(
            {
                "global_position": position,
                "scan_id": scan_id,
                "scan_role": "development-validation",
                "source_identity": source_identity,
                "target_identity": target_identity,
                "prediction_identity": prediction_identity,
                "brain_mask_identity": brain_mask_identity,
                "target_validity_mask_identity": target_validity_mask_identity,
                "roi_masks_identity": roi_masks_identity,
                "quality": quality,
                "quality_sha256": canonical_sha256(quality),
                "passed": True,
            }
        )
    prediction_identities = [row["prediction_identity"] for row in rows]
    selection_qa = seal_development_quality_record(
        {
            "schema": "connect4-development-quality-attestation-v1",
            "tier": SELECTION_TIER,
            "status": "PASS",
            "passed": True,
            "selectable": True,
            "all_development_evaluated": True,
            "policy_note": POLICY_NOTE,
            "policy": quality_policy,
            "policy_sha256": canonical_sha256(quality_policy),
            "required_checks": list(DEVELOPMENT_REQUIRED_CHECKS),
            "step": 200,
            "development_partition_size": len(development_records),
            "global_limit": len(development_records),
            "ordered_global_positions": list(range(len(development_records))),
            "ordered_scan_ids": [row["scan_id"] for row in rows],
            "ordered_scan_ids_sha256": canonical_sha256(
                [row["scan_id"] for row in rows]
            ),
            "per_scan_records": rows,
            "per_scan_records_sha256": canonical_sha256(rows),
            "prediction_set_sha256": canonical_sha256(prediction_identities),
            "model_state_sha256": model_digest,
            "checkpoint_candidate_identity": candidate,
            "checkpoint_candidate_identity_sha256": candidate["record_sha256"],
            "config_sha256": canonical_sha256(config),
            "cohort_admission_identity_record_sha256": admission_identity[
                "record_sha256"
            ],
            "cohort_admission_data_root_sha256": admission_identity[
                "data_root_sha256"
            ],
            "runtime_identity_sha256": runtime_identity["canonical_sha256"],
            "artifact_identity_sha256": canonical_sha256(artifact_identity),
            "validation_shard_contract_sha256": (
                selection_validation_shard_contract["record_sha256"]
            ),
            "no_best_subject_selection": True,
            "sealed_test_targets_opened": False,
            "exact_native_crop_no_resampling": True,
        }
    )
    checkpoint = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "architecture_contract": synthesis_architecture_contract(),
        "architecture_contract_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "model": model_state,
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": {},
        "rng_states": [{}, {}, {}, {}],
        "world_size": 4,
        "config": config,
        "split_identity": split_identity,
        "artifact_identity": artifact_identity,
        "runtime_identity": runtime_identity,
        "cohort_admission_identity": admission_identity,
        "validation_shard_contract": validation_shard_contract,
        "selection_validation_shard_contract": (
            selection_validation_shard_contract
        ),
        "latest_development_qa": selection_qa,
        "latest_development_qa_file_sha256": "9" * 64,
        "selection_development_qa": selection_qa,
        "selection_development_qa_file_sha256": "9" * 64,
        "selection_eligible": True,
        "partial": False,
        "next_batch_index": 0,
        "next_epoch": 200,
        "completed_batches": 200,
    }
    if mutate is not None:
        mutate(checkpoint)
    path = tmp_path / "completed.pt"
    torch.save(checkpoint, path)
    return path


def test_downstream_protocol_has_exactly_five_patient_disjoint_folds(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    checkpoint = _write_completed_checkpoint(tmp_path)
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    folds = five_fold_downstream_splits(
        labels,
        patients,
        scan_ids=scans,
        cohort_ids=cohorts,
        unseen_cohort_manifest_path=manifest,
        synthesis_checkpoint_path=checkpoint,
        seed=12,
    )
    assert len(folds) == 5
    all_test = []
    for train_indices, test_indices in folds:
        assert set(patients[train_indices]).isdisjoint(set(patients[test_indices]))
        all_test.extend(test_indices.tolist())
    assert sorted(all_test) == list(range(len(labels)))


@pytest.mark.parametrize("mutation", ["missing", "failed", "model_drift"])
def test_completed_checkpoint_requires_exact_full_development_qa(tmp_path, mutation):
    def mutate(checkpoint):
        if mutation == "missing":
            checkpoint.pop("selection_development_qa")
        elif mutation == "model_drift":
            checkpoint["model"]["weight"] = torch.zeros(1)
        else:
            qa = deepcopy(checkpoint["selection_development_qa"])
            row = qa["per_scan_records"][0]
            row["quality"]["passed"] = False
            row["quality"]["checks"]["dynamic_power_ratio"]["passed"] = False
            row["quality_sha256"] = canonical_sha256(row["quality"])
            row["passed"] = False
            qa["passed"] = False
            qa["status"] = "FAIL"
            qa["selectable"] = False
            qa["per_scan_records_sha256"] = canonical_sha256(
                qa["per_scan_records"]
            )
            qa = seal_development_quality_record(qa)
            checkpoint["selection_development_qa"] = qa
            checkpoint["latest_development_qa"] = qa

    path = _write_completed_checkpoint(tmp_path, mutate=mutate)
    with pytest.raises(RuntimeError, match="development|all-development|differs"):
        load_completed_synthesis_checkpoint(path)


def test_downstream_protocol_rejects_seen_patient_or_cohort(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    checkpoint = _write_completed_checkpoint(tmp_path, train_patient="P00")
    with pytest.raises(ValueError, match="patients used"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )
    checkpoint = _write_completed_checkpoint(tmp_path)
    cohorts = cohorts.copy()
    cohorts[:2] = "A4"
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    with pytest.raises(ValueError, match="synthesis-cohort identities"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_requires_exact_manifest_identity(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    checkpoint = _write_completed_checkpoint(tmp_path)
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    changed_patients = patients.copy()
    changed_patients[0] = "CALLER-SUBSTITUTED"
    with pytest.raises(ValueError, match="differ from the exact unseen-cohort manifest"):
        five_fold_downstream_splits(
            labels,
            changed_patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_requires_completed_checkpoint_evidence(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=tmp_path / "missing.pt",
        )
    checkpoint = _write_completed_checkpoint(
        tmp_path,
        mutate=lambda value: value.update(partial=True, next_epoch=199),
    )
    with pytest.raises(RuntimeError, match="completed synthesis-training"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_rejects_detail_v1_checkpoint_before_use(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)

    def make_legacy(value):
        value["format"] = "connect4_iteration_exact_v2"
        value.pop("architecture_contract")
        value.pop("architecture_contract_sha256")

    checkpoint = _write_completed_checkpoint(tmp_path, mutate=make_legacy)
    with pytest.raises(RuntimeError, match="pre-v10"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_rejects_run_artifact_v4_checkpoint(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    checkpoint = _write_completed_checkpoint(
        tmp_path,
        mutate=lambda value: value["artifact_identity"].update(
            format="connect4_run_artifacts_v4"
        ),
    )
    with pytest.raises(RuntimeError, match="invalid run-artifact identity"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_rejects_smaller_resigned_validation_extent(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)

    def shrink_validation_extent(value):
        contract = value["validation_shard_contract"]
        contract["global_limit"] = 3
        contract["global_first_n_positions"] = [0, 1, 2]
        contract["rank_to_global_positions"] = {
            "0": [0],
            "1": [1],
            "2": [2],
            "3": [],
        }
        contract.pop("record_sha256")
        contract["record_sha256"] = canonical_sha256(contract)

    checkpoint = _write_completed_checkpoint(
        tmp_path,
        mutate=shrink_validation_extent,
    )
    with pytest.raises(RuntimeError, match="validation-shard extent differs"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_protocol_rejects_resigned_revoked_brainlm_publication(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)

    def substitute_revoked_publication(value):
        identity = value["artifact_identity"]["perceptual_extractor"]
        artifacts = identity["model_artifacts"]
        publication = artifacts["immutable_publication"]
        publication.update(
            {
                "published_source_root": (
                    "/srv/connect4/connect4_validation_20260829/"
                    "official_brainlm_"
                    "eded39c86c27e03f5ead1d6a14311e92d1305e5"
                ),
                "purpose": (
                    "connect4-official-brainlm-eded39c86c27-immutable-source"
                ),
                "sha256": (
                    "c70b6dcde2a14181b6b969b726f47dd1177bae8711a32d3f6d060fbff9189e7f"
                ),
            }
        )
        publication.pop("content_record_sha256")
        publication["content_record_sha256"] = canonical_sha256(publication)
        artifacts.pop("content_record_sha256")
        artifacts["content_record_sha256"] = canonical_sha256(artifacts)
        identity.pop("record_sha256")
        identity["record_sha256"] = canonical_sha256(identity)

    checkpoint = _write_completed_checkpoint(
        tmp_path,
        mutate=substitute_revoked_publication,
    )
    with pytest.raises(RuntimeError, match="invalid run-artifact identity"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_downstream_checkpoint_exclusion_evidence_cannot_be_partially_replaced(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    checkpoint = _write_completed_checkpoint(
        tmp_path,
        mutate=lambda value: value["split_identity"].update(
            synthesis_patient_ids=["FABRICATED"]
        ),
    )
    with pytest.raises(RuntimeError, match="exclusion identity is invalid"):
        five_fold_downstream_splits(
            labels,
            patients,
            scan_ids=scans,
            cohort_ids=cohorts,
            unseen_cohort_manifest_path=manifest,
            synthesis_checkpoint_path=checkpoint,
        )


def test_actual_gat_evaluator_reports_cross_domain_metrics_and_statistics(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    checkpoint = _write_completed_checkpoint(tmp_path)
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    node_features = []
    edge_indices = []
    for label in labels:
        # A real graph input with two directed edges; the class offset merely
        # keeps the one-epoch smoke test numerically well-conditioned.
        node_features.append(
            np.asarray([[float(label), 0.0], [float(label), 1.0], [float(label), 2.0]])
        )
        edge_indices.append(np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]]))

    result = evaluate_downstream_gat(
        node_features,
        edge_indices,
        labels,
        patients,
        scan_ids=scans,
        graph_scan_ids=scans,
        cohort_ids=cohorts,
        unseen_cohort_manifest_path=manifest,
        synthesis_checkpoint_path=checkpoint,
        training_node_features=[features + 0.25 for features in node_features],
        training_edge_indices=edge_indices,
        training_graph_scan_ids=scans,
        hidden_channels=4,
        heads=1,
        dropout=0.0,
        epochs=1,
        batch_size=16,
        seed=4,
        device="cpu",
    )
    assert result["num_folds"] == 5
    assert result["domain_protocol"] == "cross_domain"
    assert len(result["synthesis_checkpoint_sha256"]) == 64
    assert result["gat_settings"]["seed"] == 4
    assert result["gat_settings"]["hidden_channels"] == 4
    assert result["gat_settings"]["heads"] == 1
    assert result["gat_settings"]["epochs"] == 1
    assert result["gat_settings"]["optimizer"] == "AdamW"
    assert result["gat_settings"]["format"] == DOWNSTREAM_GAT_SETTINGS_FORMAT
    assert result["gat_settings"]["fold_allocator"] == DOWNSTREAM_FOLD_ALLOCATOR_ID
    assert "splitter" not in result["gat_settings"]
    assert len(result["gat_settings"]["sha256"]) == 64
    stream_identity = result["graph_stream_identity"]
    assert [
        record["scan_id"] for record in stream_identity["evaluation"]["records"]
    ] == scans.tolist()
    assert [
        record["scan_id"] for record in stream_identity["training"]["records"]
    ] == scans.tolist()
    assert all(
        len(record["graph_sha256"]) == 64
        for role in ("evaluation", "training")
        for record in stream_identity[role]["records"]
    )
    assert (
        stream_identity["evaluation"]["sha256"]
        != stream_identity["training"]["sha256"]
    )
    assert len(stream_identity["sha256"]) == 64
    assert len(result["protocol_identity"]["sha256"]) == 64
    assert len(result["folds"]) == 5
    assert set(result["summary"]) == set(DOWNSTREAM_METRICS)
    for metric in DOWNSTREAM_METRICS:
        assert 0.0 <= result["summary"][metric]["mean"] <= 1.0
        assert result["summary"][metric]["std"] >= 0.0
    comparison = compare_downstream_gat_results(result, result)
    assert comparison["correction"] == "Bonferroni"
    assert comparison["pairing_unit"] == "held-out patient fold"
    assert comparison["paper_significance_reproducible"] is False
    assert comparison["minimum_attainable_bonferroni_p"] == 0.3125
    assert (
        comparison["graph_stream_identity_sha256"]
        == result["graph_stream_identity"]["sha256"]
    )
    assert comparison["gat_settings_sha256"] == result["gat_settings"]["sha256"]
    assert set(comparison["metrics"]) == set(DOWNSTREAM_METRICS)
    assert all(
        values["bonferroni_p"] == 1.0
        for values in comparison["metrics"].values()
    )
    assert all(
        values["paper_significance"] is None
        for values in comparison["metrics"].values()
    )

    wrong_checkpoint = deepcopy(result)
    wrong_checkpoint["synthesis_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different synthesis checkpoints"):
        compare_downstream_gat_results(result, wrong_checkpoint)

    repeated_fold = deepcopy(result)
    repeated_fold["folds"][1]["fold_identity_sha256"] = repeated_fold["folds"][0][
        "fold_identity_sha256"
    ]
    with pytest.raises(ValueError, match="invalid or repeated"):
        compare_downstream_gat_results(result, repeated_fold)

    different_settings = deepcopy(result)
    different_settings["gat_settings"]["seed"] = 5
    settings_payload = dict(different_settings["gat_settings"])
    settings_payload.pop("sha256")
    different_settings["gat_settings"]["sha256"] = canonical_sha256(
        settings_payload
    )
    with pytest.raises(ValueError, match="different GAT settings or seeds"):
        compare_downstream_gat_results(result, different_settings)

    reordered_graphs = deepcopy(result)
    evaluation_stream = reordered_graphs["graph_stream_identity"]["evaluation"]
    evaluation_stream["records"][0], evaluation_stream["records"][1] = (
        evaluation_stream["records"][1],
        evaluation_stream["records"][0],
    )
    stream_payload = dict(evaluation_stream)
    stream_payload.pop("sha256")
    evaluation_stream["sha256"] = canonical_sha256(stream_payload)
    streams_payload = dict(reordered_graphs["graph_stream_identity"])
    streams_payload.pop("sha256")
    reordered_graphs["graph_stream_identity"]["sha256"] = canonical_sha256(
        streams_payload
    )
    with pytest.raises(ValueError, match="not aligned to the protocol scan order"):
        compare_downstream_gat_results(result, reordered_graphs)

    changed_graph = deepcopy(result)
    changed_stream = changed_graph["graph_stream_identity"]["evaluation"]
    changed_stream["records"][0]["graph_sha256"] = "e" * 64
    changed_stream_payload = dict(changed_stream)
    changed_stream_payload.pop("sha256")
    changed_stream["sha256"] = canonical_sha256(changed_stream_payload)
    changed_streams_payload = dict(changed_graph["graph_stream_identity"])
    changed_streams_payload.pop("sha256")
    changed_graph["graph_stream_identity"]["sha256"] = canonical_sha256(
        changed_streams_payload
    )
    with pytest.raises(ValueError, match="different or reordered graph streams"):
        compare_downstream_gat_results(result, changed_graph)


def test_cross_domain_graph_streams_require_independent_scan_bindings(tmp_path):
    labels, patients, cohorts, scans = _example_data()
    checkpoint = _write_completed_checkpoint(tmp_path)
    manifest = _write_unseen_manifest(tmp_path, scans, patients, cohorts)
    node_features = [np.ones((2, 2), dtype=np.float32) for _ in labels]
    edge_indices = [np.asarray([[0, 1], [1, 0]]) for _ in labels]
    common = {
        "scan_ids": scans,
        "cohort_ids": cohorts,
        "unseen_cohort_manifest_path": manifest,
        "synthesis_checkpoint_path": checkpoint,
        "training_node_features": node_features,
        "training_edge_indices": edge_indices,
        "epochs": 1,
        "device": "cpu",
    }
    reordered = np.roll(scans, 1)
    with pytest.raises(ValueError, match="graph_scan_ids must exactly match"):
        evaluate_downstream_gat(
            node_features,
            edge_indices,
            labels,
            patients,
            graph_scan_ids=reordered,
            training_graph_scan_ids=scans,
            **common,
        )
    with pytest.raises(ValueError, match="training_graph_scan_ids must exactly match"):
        evaluate_downstream_gat(
            node_features,
            edge_indices,
            labels,
            patients,
            graph_scan_ids=scans,
            training_graph_scan_ids=reordered,
            **common,
        )
    with pytest.raises(ValueError, match="training_graph_scan_ids are required"):
        evaluate_downstream_gat(
            node_features,
            edge_indices,
            labels,
            patients,
            graph_scan_ids=scans,
            **common,
        )
