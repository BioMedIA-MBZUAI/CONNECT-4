from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.ndimage import gaussian_filter

from data.provenance import canonical_sha256
from eval.development_quality import (
    QUALITY_CHECK_NAMES,
    ROUTINE_TIER,
    SELECTION_TIER,
    checkpoint_candidate_identity,
    development_tensor_identity,
    evaluated_tensor_set_identity,
    model_state_sha256,
    publish_development_quality_record,
    seal_development_quality_record,
    validate_development_quality_record,
    validate_progress_checkpoint_development_qa,
    validate_selectable_checkpoint_development_qa,
)
from eval.quality import (
    DEVELOPMENT_REQUIRED_CHECKS,
    POLICY_NOTE,
    QualityPolicy,
    evaluate_4d_pair_quality_arrays,
    require_release_quality_policy,
)
from training.train import (
    _run_rank_zero_synchronized,
    publish_selectable_checkpoint,
    require_passing_final_development_qa,
)
from utils.config import validate_paper_config


def _dynamic_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    shape = (12, 12, 12)
    frames = 32
    mask = np.zeros(shape, dtype=np.float32)
    mask[2:10, 2:10, 2:10] = 1
    roi_masks = np.zeros((4, *shape), dtype=np.float32)
    roi_masks[0, 2:6, 2:6, 2:10] = 1
    roi_masks[1, 6:10, 2:6, 2:10] = 1
    roi_masks[2, 2:6, 6:10, 2:10] = 1
    roi_masks[3, 6:10, 6:10, 2:10] = 1
    coordinates = np.indices(shape)
    checker = ((coordinates.sum(axis=0) % 2) * 2 - 1).astype(np.float32)
    time = np.arange(frames, dtype=np.float32)
    real = np.zeros((*shape, frames), dtype=np.float32)
    for roi_index, frequency in enumerate((1.0, 2.0, 3.0, 5.0)):
        temporal = np.sin(2 * np.pi * frequency * time / frames + roi_index * 0.31)
        temporal += 0.35 * np.cos(
            2 * np.pi * (frequency + 1.0) * time / frames + roi_index
        )
        roi = roi_masks[roi_index] > 0
        real[roi] = (
            0.50
            + 0.10 * checker[roi, None]
            + 0.12 * temporal[None, :]
            + rng.normal(0.0, 0.015, size=(int(roi.sum()), frames))
        )
    real *= mask[..., None]
    real = np.clip(real, 0.0, 1.0)
    return real, real.copy(), mask, roi_masks


def test_array_quality_gate_passes_identical_dynamic_native_grid_and_rejects_collapse():
    real, predicted, mask, roi_masks = _dynamic_pair()
    identity = {"record_sha256": "a" * 64, "scan_role": "development-validation"}
    passed = evaluate_4d_pair_quality_arrays(
        real,
        predicted,
        mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=3.0,
        roi_masks=roi_masks,
        require_structured_temporal=True,
        source_identity=identity,
    )
    assert passed["passed"] is True
    assert passed["geometry"]["resampling_performed"] is False
    assert all(passed["checks"][name]["passed"] for name in DEVELOPMENT_REQUIRED_CHECKS)

    static = np.repeat(real.mean(axis=-1, keepdims=True), real.shape[-1], axis=-1)
    collapsed = evaluate_4d_pair_quality_arrays(
        real,
        static,
        mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=3.0,
        roi_masks=roi_masks,
        require_structured_temporal=True,
    )
    assert collapsed["passed"] is False
    assert collapsed["temporal_collapse_detected"] is True
    assert collapsed["checks"]["dynamic_power_ratio"]["passed"] is False

    smoothed = np.stack(
        [
            gaussian_filter(real[..., frame], sigma=2.0)
            for frame in range(real.shape[-1])
        ],
        axis=-1,
    )
    smoothed *= mask[..., None]
    over_smoothed = evaluate_4d_pair_quality_arrays(
        real,
        smoothed,
        mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=3.0,
        roi_masks=roi_masks,
        require_structured_temporal=True,
    )
    assert over_smoothed["passed"] is False
    assert any(
        over_smoothed["checks"][name]["passed"] is False
        for name in (
            "spatial_high_frequency_ratio",
            "dynamic_high_frequency_ratio",
            "dynamic_high_frequency_correlation",
        )
    )


def test_default_gate_rejects_half_amplitude_dynamic_regression():
    """A variance-quartering mean regression must never look non-collapsed."""

    real, _, mask, roi_masks = _dynamic_pair()
    temporal_mean = real.mean(axis=-1, keepdims=True)
    half_amplitude = temporal_mean + 0.5 * (real - temporal_mean)
    result = evaluate_4d_pair_quality_arrays(
        real,
        half_amplitude,
        mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=3.0,
        roi_masks=roi_masks,
        require_structured_temporal=True,
    )

    assert result["passed"] is False
    assert result["temporal_collapse_detected"] is True
    assert result["checks"]["dynamic_high_frequency_ratio"]["passed"] is False
    assert result["checks"]["temporal_variance_ratio"]["passed"] is False
    assert result["checks"]["dvars_ratio"]["passed"] is False
    assert result["checks"]["dynamic_power_ratio"]["passed"] is False


def test_release_policy_cannot_be_relaxed_or_redefine_metrics():
    baseline = asdict(QualityPolicy())
    assert require_release_quality_policy(baseline) == QualityPolicy()

    weaker_minimum = copy.deepcopy(baseline)
    weaker_minimum["min_dynamic_power_ratio"] = 0.25
    with pytest.raises(ValueError, match="weakens frozen bounds"):
        require_release_quality_policy(weaker_minimum)

    weaker_maximum = copy.deepcopy(baseline)
    weaker_maximum["max_near_static_voxel_fraction"] = 0.50
    with pytest.raises(ValueError, match="weakens frozen bounds"):
        require_release_quality_policy(weaker_maximum)

    changed_metric = copy.deepcopy(baseline)
    changed_metric["high_frequency_sigma_voxels"] = 2.0
    with pytest.raises(ValueError, match="changes frozen metric definitions"):
        require_release_quality_policy(changed_metric)


def test_attestation_accepts_actual_array_quality_and_tensor_commitment():
    record, config = _qa_record(
        tier=SELECTION_TIER,
        passed=True,
        model_state={"weight": torch.ones(1)},
        development_size=1,
    )
    real, predicted, mask, roi_masks = _dynamic_pair()
    source = copy.deepcopy(record["per_scan_records"][0]["source_identity"])
    affine = np.diag([3.0, 3.0, 3.0, 1.0]).tolist()
    source.update(
        {
            "padded_shape": [12, 12, 12],
            "native_shape": [12, 12, 12],
            "padding_before": [0, 0, 0],
            "padding_after": [0, 0, 0],
            "padded_affine_ras_mm": copy.deepcopy(affine),
            "padded_affine_sha256": canonical_sha256(affine),
            "native_affine_ras_mm": copy.deepcopy(affine),
            "native_affine_sha256": canonical_sha256(affine),
            "brainlm_support_tensor_sha256": hashlib.sha256(
                np.asarray(mask > 0.5, dtype=np.uint8).tobytes(order="C")
            ).hexdigest(),
        }
    )
    source.pop("record_sha256")
    source["record_sha256"] = canonical_sha256(source)
    target_tensor = torch.from_numpy(real).permute(3, 0, 1, 2).contiguous()
    prediction_tensor = torch.from_numpy(predicted).permute(3, 0, 1, 2).contiguous()
    mask_tensor = torch.from_numpy(mask)
    target_validity_tensor = torch.from_numpy(
        np.any(real != 0.0, axis=-1).astype(np.float32)
    )
    roi_tensor = torch.from_numpy(roi_masks) * target_validity_tensor
    target_identity = development_tensor_identity(target_tensor, kind="target")
    prediction_identity = development_tensor_identity(
        prediction_tensor, kind="prediction"
    )
    brain_identity = development_tensor_identity(
        mask_tensor,
        kind="brain_mask",
        source_identity=source,
    )
    target_validity_identity = development_tensor_identity(
        target_validity_tensor,
        kind="target_validity_mask",
        brain_mask=mask_tensor,
        target=target_tensor,
    )
    roi_identity = development_tensor_identity(
        roi_tensor,
        kind="roi_masks",
        brain_mask=mask_tensor,
        source_identity=source,
    )
    tensor_set = evaluated_tensor_set_identity(
        target=target_identity,
        prediction=prediction_identity,
        brain_mask=brain_identity,
        target_validity_mask=target_validity_identity,
        roi_masks=roi_identity,
    )
    quality = evaluate_4d_pair_quality_arrays(
        real,
        predicted,
        target_validity_tensor.numpy(),
        affine=np.asarray(affine),
        tr_seconds=3.0,
        roi_masks=roi_tensor.numpy(),
        require_structured_temporal=True,
        source_identity=source,
        evaluated_tensor_set_identity=tensor_set,
        structural_mask=mask,
    )
    row = record["per_scan_records"][0]
    row.update(
        {
            "source_identity": source,
            "target_identity": target_identity,
            "prediction_identity": prediction_identity,
            "brain_mask_identity": brain_identity,
            "target_validity_mask_identity": target_validity_identity,
            "roi_masks_identity": roi_identity,
            "quality": quality,
            "quality_sha256": canonical_sha256(quality),
            "passed": True,
        }
    )
    config["data"] = {
        "architecture_shape": [12, 12, 12],
        "num_frames": 32,
        "tr_seconds": 3.0,
    }
    record["config_sha256"] = canonical_sha256(config)
    record = _resign_after_rows(record)
    assert (
        validate_development_quality_record(record, require_selection_pass=True)
        == record
    )


def _qa_record(
    *,
    tier: str,
    passed: bool,
    model_state: dict[str, torch.Tensor],
    development_size: int = 3,
) -> tuple[dict, dict]:
    selection = tier == SELECTION_TIER
    limit = development_size if selection else development_size - 1
    policy = asdict(QualityPolicy())
    config = {
        "data": {
            "architecture_shape": [8, 8, 8],
            "num_frames": 8,
            "tr_seconds": 3.0,
        },
        "training": {"development_quality_gate": {"policy": policy}},
    }
    model_digest = model_state_sha256(model_state)
    candidate = checkpoint_candidate_identity(
        model_state_digest=model_digest,
        completed_batches=17,
        next_epoch=4,
        next_batch_index=0,
        partial=not selection,
    )
    rows = []
    for position in range(limit):
        scan_id = f"DEV-{position:03d}"
        affine = np.diag([3.0, 3.0, 3.0, 1.0]).tolist()
        source = {
            "schema": "connect4-development-source-identity-v1",
            "scan_id": scan_id,
            "scan_role": "development-validation",
            "brainlm_target_artifact_identity_sha256": "a" * 64,
            "brainlm_scan_context_record_sha256": "b" * 64,
            "brainlm_dataset_state_record_sha256": "c" * 64,
            "brainlm_dataset_state_sha256": "d" * 64,
            "brainlm_support_tensor_sha256": hashlib.sha256(
                np.ones((8, 8, 8), dtype=np.uint8).tobytes(order="C")
            ).hexdigest(),
            "brainlm_prepared_mask_sha256": "e" * 64,
            "brainlm_padded_mask_artifact_descriptor_sha256": "1" * 64,
            "brainlm_structural_source_identity_sha256": "2" * 64,
            "brainlm_native_preprocessing_source_sha256": "f" * 64,
            "brainlm_native_alignment_authority_sha256": "0" * 64,
            "roi_mapping_sha256": "3" * 64,
            "padded_shape": [8, 8, 8],
            "native_shape": [8, 8, 8],
            "padding_before": [0, 0, 0],
            "padding_after": [0, 0, 0],
            "padded_affine_ras_mm": copy.deepcopy(affine),
            "padded_affine_sha256": canonical_sha256(affine),
            "native_affine_ras_mm": copy.deepcopy(affine),
            "native_affine_sha256": canonical_sha256(affine),
        }
        source["record_sha256"] = canonical_sha256(source)
        target_tensor = torch.full((8, 8, 8, 8), 0.5)
        prediction_tensor = target_tensor.clone()
        mask_tensor = torch.ones((8, 8, 8))
        target_validity_tensor = mask_tensor.clone()
        roi_tensor = torch.zeros((4, 8, 8, 8))
        for roi_index in range(4):
            roi_tensor[roi_index, roi_index * 2 : (roi_index + 1) * 2] = 1
        target_identity = development_tensor_identity(target_tensor, kind="target")
        prediction_identity = development_tensor_identity(
            prediction_tensor, kind="prediction"
        )
        brain_mask_identity = development_tensor_identity(
            mask_tensor,
            kind="brain_mask",
            source_identity=source,
        )
        target_validity_mask_identity = development_tensor_identity(
            target_validity_tensor,
            kind="target_validity_mask",
            brain_mask=mask_tensor,
            target=target_tensor,
        )
        roi_masks_identity = development_tensor_identity(
            roi_tensor,
            kind="roi_masks",
            brain_mask=mask_tensor,
            source_identity=source,
        )
        evaluated_tensors = evaluated_tensor_set_identity(
            target=target_identity,
            prediction=prediction_identity,
            brain_mask=brain_mask_identity,
            target_validity_mask=target_validity_mask_identity,
            roi_masks=roi_masks_identity,
        )
        checks = {
            name: {
                "passed": bool(passed or name != "dynamic_high_frequency_ratio"),
                "value": 1.0 if passed else 0.0,
                "policy": "fixture",
            }
            for name in QUALITY_CHECK_NAMES
        }
        checks["grid_affine_tr_contract"]["passed"] = True
        checks["grid_affine_tr_contract"]["value"] = {
            "shape": [8, 8, 8, 8],
            "affine_max_abs_difference": 0.0,
            "tr_seconds": 3.0,
        }
        checks["near_static_voxel_fraction"]["value"] = 0.0
        checks["structured_temporal_available"]["value"] = 4
        checks["roi_fc_correlation"]["value"] = 1.0
        checks["roi_power_spectrum_correlation"]["value"] = 1.0
        failed_checks = [
            name for name in QUALITY_CHECK_NAMES if not checks[name]["passed"]
        ]
        quality = {
            "schema": "connect4-paired-4d-quality-v1",
            "policy_note": POLICY_NOTE,
            "policy": policy,
            "implementation_sources": {
                "quality_evaluator": {
                    "relative_path": "eval/quality.py",
                    "path": "/fixture/eval/quality.py",
                    "size_bytes": 1,
                    "sha256": "9" * 64,
                }
            },
            "sources": {
                "array_identity": source,
                "evaluated_tensor_set_identity": evaluated_tensors,
            },
            "geometry": {
                "shape": [8, 8, 8, 8],
                "tr_seconds": 3.0,
                "affine": copy.deepcopy(affine),
                "affine_max_abs_difference": 0.0,
                "canonical_axis_codes": ["R", "A", "S"],
                "original_axis_codes": {
                    "real": ["R", "A", "S"],
                    "predicted": ["R", "A", "S"],
                },
                "voxel_sizes_mm": [3.0, 3.0, 3.0],
                "mask_voxels": 512,
                "mask_derived_from_real": False,
                "array_entry_point": True,
                "resampling_performed": False,
            },
            "support": {},
            "outside_mask": {},
            "spatial": {
                "high_frequency_rms_ratio": 1.0,
                "temporal_mean_high_frequency_correlation": 1.0,
                "dynamic_high_frequency_rms_ratio": 1.0 if passed else 0.0,
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
                "roi_voxel_counts": {str(label): 128 for label in range(1, 5)},
                "power_spectrum_non_dc_bins": 4,
                "issues": [],
                "temporally_valid_real_roi_count": 4,
                "excluded_constant_real_roi_labels": [],
                "roi_mean_time_series_shape": [4, 8],
                "available": True,
                "unavailable_reason": None,
                "constant_predicted_roi_labels": [],
                "fc_upper_triangle_edges": 6,
                "fc_matrix_correlation": 1.0,
                "power_spectrum_correlation": 1.0,
            },
            "paper_metrics": {},
            "checks": checks,
            "failed_checks": failed_checks,
            "failed_temporal_checks": [],
            "failed_temporal_collapse_checks": [],
            "failed_structured_temporal_checks": [],
            "temporal_quality_mismatch_detected": False,
            "temporal_collapse_detected": False,
            "passed": passed,
            "verdict": "pass" if passed else "fail_quality_gate",
        }
        rows.append(
            {
                "global_position": position,
                "scan_id": scan_id,
                "scan_role": "development-validation",
                "source_identity": source,
                "target_identity": target_identity,
                "prediction_identity": prediction_identity,
                "brain_mask_identity": brain_mask_identity,
                "target_validity_mask_identity": target_validity_mask_identity,
                "roi_masks_identity": roi_masks_identity,
                "quality": quality,
                "quality_sha256": canonical_sha256(quality),
                "passed": passed,
            }
        )
    scan_ids = [row["scan_id"] for row in rows]
    prediction_identities = [row["prediction_identity"] for row in rows]
    qa = seal_development_quality_record(
        {
            "schema": "connect4-development-quality-attestation-v1",
            "tier": tier,
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "selectable": bool(selection and passed),
            "all_development_evaluated": selection,
            "policy_note": POLICY_NOTE,
            "policy": policy,
            "policy_sha256": canonical_sha256(policy),
            "required_checks": list(DEVELOPMENT_REQUIRED_CHECKS),
            "step": 17,
            "development_partition_size": development_size,
            "global_limit": limit,
            "ordered_global_positions": list(range(limit)),
            "ordered_scan_ids": scan_ids,
            "ordered_scan_ids_sha256": canonical_sha256(scan_ids),
            "per_scan_records": rows,
            "per_scan_records_sha256": canonical_sha256(rows),
            "prediction_set_sha256": canonical_sha256(prediction_identities),
            "model_state_sha256": model_digest,
            "checkpoint_candidate_identity": candidate,
            "checkpoint_candidate_identity_sha256": candidate["record_sha256"],
            "config_sha256": canonical_sha256(config),
            "cohort_admission_identity_record_sha256": "1" * 64,
            "cohort_admission_data_root_sha256": "2" * 64,
            "runtime_identity_sha256": "3" * 64,
            "artifact_identity_sha256": canonical_sha256({"artifact": 1}),
            "validation_shard_contract_sha256": "4" * 64,
            "no_best_subject_selection": True,
            "sealed_test_targets_opened": False,
            "exact_native_crop_no_resampling": True,
        }
    )
    return qa, config


def _checkpoint(*, passed: bool) -> dict:
    model_state = {"weight": torch.tensor([1.0, 2.0])}
    qa, config = _qa_record(
        tier=SELECTION_TIER,
        passed=passed,
        model_state=model_state,
    )
    return {
        "model": model_state,
        "config": config,
        "artifact_identity": {"artifact": 1},
        "runtime_identity": {"canonical_sha256": "3" * 64},
        "cohort_admission_identity": {
            "record_sha256": "1" * 64,
            "data_root_sha256": "2" * 64,
        },
        "selection_validation_shard_contract": {"record_sha256": "4" * 64},
        "completed_batches": 17,
        "next_epoch": 4,
        "next_batch_index": 0,
        "partial": False,
        "selection_eligible": passed,
        "latest_development_qa": qa,
        "latest_development_qa_file_sha256": "5" * 64,
        "selection_development_qa": qa,
        "selection_development_qa_file_sha256": "5" * 64,
    }


def _resign_after_rows(record: dict) -> dict:
    value = copy.deepcopy(record)
    value["per_scan_records_sha256"] = canonical_sha256(value["per_scan_records"])
    value["prediction_set_sha256"] = canonical_sha256(
        [row["prediction_identity"] for row in value["per_scan_records"]]
    )
    return seal_development_quality_record(value)


def _resign_first_row(record: dict) -> dict:
    value = copy.deepcopy(record)
    row = value["per_scan_records"][0]
    row["quality_sha256"] = canonical_sha256(row["quality"])
    return _resign_after_rows(value)


def test_qa_attestation_rejects_rank_order_and_missing_identity_even_when_resigned():
    record, _ = _qa_record(
        tier=SELECTION_TIER,
        passed=True,
        model_state={"weight": torch.ones(1)},
        development_size=43,
    )
    assert validate_development_quality_record(record, require_selection_pass=True)
    crossed = copy.deepcopy(record)
    crossed["per_scan_records"][0], crossed["per_scan_records"][1] = (
        crossed["per_scan_records"][1],
        crossed["per_scan_records"][0],
    )
    crossed = _resign_after_rows(crossed)
    with pytest.raises(RuntimeError, match="per-scan identity"):
        validate_development_quality_record(crossed)

    missing = copy.deepcopy(record)
    missing["per_scan_records"][0].pop("source_identity")
    missing = _resign_after_rows(missing)
    with pytest.raises(RuntimeError, match="per-scan"):
        validate_development_quality_record(missing)

    real, predicted, mask, _ = _dynamic_pair()
    no_roi = evaluate_4d_pair_quality_arrays(
        real,
        predicted,
        mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=3.0,
        roi_masks=None,
        require_structured_temporal=True,
    )
    assert no_roi["checks"]["structured_temporal_available"]["passed"] is False


def test_qa_rejects_crossed_source_minimal_grid_and_mask_roi_swaps():
    record, _ = _qa_record(
        tier=SELECTION_TIER,
        passed=True,
        model_state={"weight": torch.ones(1)},
    )

    crossed = copy.deepcopy(record)
    row = crossed["per_scan_records"][0]
    source = row["source_identity"]
    source["scan_id"] = "CROSSED-SOURCE"
    source["record_sha256"] = canonical_sha256(
        {key: value for key, value in source.items() if key != "record_sha256"}
    )
    row["quality"]["sources"]["array_identity"] = source
    crossed = _resign_first_row(crossed)
    with pytest.raises(RuntimeError, match="source identity differs"):
        validate_development_quality_record(crossed)

    minimal = copy.deepcopy(record)
    minimal["per_scan_records"][0]["quality"]["geometry"] = {
        "shape": [8, 8, 8, 8],
        "resampling_performed": False,
    }
    minimal = _resign_first_row(minimal)
    with pytest.raises(RuntimeError, match="grid/source evidence is incomplete"):
        validate_development_quality_record(minimal)

    affine_swap = copy.deepcopy(record)
    affine_swap["per_scan_records"][0]["quality"]["geometry"]["affine"][0][3] = 9.0
    affine_swap = _resign_first_row(affine_swap)
    with pytest.raises(RuntimeError, match="target/prediction/mask grid differs"):
        validate_development_quality_record(affine_swap)

    mask_swap = copy.deepcopy(record)
    first = mask_swap["per_scan_records"][0]
    first["brain_mask_identity"], first["roi_masks_identity"] = (
        first["roi_masks_identity"],
        first["brain_mask_identity"],
    )
    mask_swap = _resign_first_row(mask_swap)
    with pytest.raises(RuntimeError, match="brain_mask tensor identity"):
        validate_development_quality_record(mask_swap)

    roi_mismatch = copy.deepcopy(record)
    roi_row = roi_mismatch["per_scan_records"][0]
    roi_row["quality"]["structured_temporal"]["roi_labels"] = [1, 2, 3, 9]
    roi_mismatch = _resign_first_row(roi_mismatch)
    with pytest.raises(RuntimeError, match="ROI/structured-temporal evidence differs"):
        validate_development_quality_record(roi_mismatch)


@pytest.mark.parametrize(
    ("identity_name", "mutations"),
    [
        ("brain_mask_identity", {"sha256": "f" * 64}),
        ("brain_mask_identity", {"native_binary_sha256": "f" * 64}),
        ("roi_masks_identity", {"sha256": "f" * 64}),
        (
            "roi_masks_identity",
            {
                "binary_sha256": "f" * 64,
                "channel_binary_sha256": ["f" * 64] * 4,
            },
        ),
    ],
)
def test_qa_rejects_resigned_same_kind_mask_or_roi_content_substitution(
    identity_name,
    mutations,
):
    record, _ = _qa_record(
        tier=SELECTION_TIER,
        passed=True,
        model_state={"weight": torch.ones(1)},
    )
    record["per_scan_records"][0][identity_name].update(mutations)
    record = _resign_first_row(record)
    with pytest.raises(RuntimeError, match="grid/source evidence"):
        validate_development_quality_record(record, require_selection_pass=True)


def test_selectable_checkpoint_rejects_resigned_tr_mismatch():
    checkpoint = _checkpoint(passed=True)
    qa = copy.deepcopy(checkpoint["selection_development_qa"])
    for row in qa["per_scan_records"]:
        row["quality"]["geometry"]["tr_seconds"] = 2.0
        row["quality"]["checks"]["grid_affine_tr_contract"]["value"]["tr_seconds"] = 2.0
        row["quality_sha256"] = canonical_sha256(row["quality"])
    qa = _resign_after_rows(qa)
    checkpoint["selection_development_qa"] = qa
    checkpoint["latest_development_qa"] = qa
    with pytest.raises(RuntimeError, match="differs from its all-development"):
        validate_selectable_checkpoint_development_qa(checkpoint)


def test_rank_zero_publication_failure_converges_before_any_rank_continues(
    monkeypatch,
):
    captured: dict[str, dict] = {}

    def gather(statuses, local_status):
        if local_status["rank"] == 0:
            captured["root"] = copy.deepcopy(local_status)
            nonroot = {
                "rank": 1,
                "attempted": False,
                "ok": True,
                "result": None,
                "error": None,
            }
            statuses[:] = [local_status, nonroot]
        else:
            statuses[:] = [captured["root"], local_status]

    monkeypatch.setattr("training.train.dist.all_gather_object", gather)

    def fail_publication():
        raise OSError("no-replace publication failed")

    with pytest.raises(RuntimeError, match="failed on rank zero.*no-replace"):
        _run_rank_zero_synchronized(
            fail_publication,
            rank=0,
            world_size=2,
            distributed=True,
            label="fixture publication",
        )
    nonroot_action_called = False

    def forbidden_nonroot_action():
        nonlocal nonroot_action_called
        nonroot_action_called = True

    with pytest.raises(RuntimeError, match="failed on rank zero.*no-replace"):
        _run_rank_zero_synchronized(
            forbidden_nonroot_action,
            rank=1,
            world_size=2,
            distributed=True,
            label="fixture publication",
        )
    assert nonroot_action_called is False


def test_failed_routine_is_resumable_but_never_selectable():
    failed, _ = _qa_record(
        tier=ROUTINE_TIER,
        passed=False,
        model_state={"weight": torch.ones(1)},
    )
    progress = {
        "partial": True,
        "selection_eligible": False,
        "selection_development_qa": None,
        "selection_development_qa_file_sha256": None,
        "latest_development_qa": failed,
        "latest_development_qa_file_sha256": "a" * 64,
    }
    observed = validate_progress_checkpoint_development_qa(progress)
    assert observed["passed"] is False
    with pytest.raises(RuntimeError, match="not development-QA selectable"):
        validate_selectable_checkpoint_development_qa(progress)
    with pytest.raises(RuntimeError, match="configured horizon"):
        require_passing_final_development_qa({"record": failed})


def test_selectable_publication_requires_pass_and_exact_model_state(tmp_path: Path):
    failed = _checkpoint(passed=False)
    rejected = tmp_path / "failed.pt"
    with pytest.raises(RuntimeError, match="not development-QA selectable"):
        publish_selectable_checkpoint(failed, rejected)
    assert not rejected.exists()

    passed = _checkpoint(passed=True)
    candidate = tmp_path / "passed.pt"
    publish_selectable_checkpoint(passed, candidate)
    assert candidate.is_file()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_selectable_checkpoint(passed, candidate)

    mismatched = copy.deepcopy(passed)
    mismatched["model"]["weight"] = torch.tensor([9.0, 9.0])
    with pytest.raises(RuntimeError, match="differs from its all-development"):
        validate_selectable_checkpoint_development_qa(mismatched)


def test_validation_qa_record_is_signed_atomic_and_never_replaced(tmp_path: Path):
    record, _ = _qa_record(
        tier=ROUTINE_TIER,
        passed=False,
        model_state={"weight": torch.ones(1)},
    )
    path = tmp_path / "routine.json"
    observed, digest = publish_development_quality_record(path, record)
    assert observed == record
    assert len(digest) == 64
    # An identical resume may reuse immutable evidence, but cannot rewrite it.
    assert publish_development_quality_record(path, record)[1] == digest
    changed, _ = _qa_record(
        tier=ROUTINE_TIER,
        passed=False,
        model_state={"weight": torch.zeros(1)},
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_development_quality_record(path, changed)


def test_config_keeps_routine_first_n_distinct_from_all_development_selection():
    path = Path(__file__).resolve().parents[1] / "configs" / "connect4.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Keep this test scoped to development-selection semantics; the committed
    # production template's unresolved ModernBERT revision is tested as a
    # mandatory release stop in test_config_contract.py.
    config["models"]["modernbert_revision"] = "a" * 40
    assert validate_paper_config(config)

    conflated = copy.deepcopy(config)
    conflated["training"]["development_quality_gate"]["selection"]["num_samples"] = 4
    with pytest.raises(ValueError, match="extents distinct"):
        validate_paper_config(conflated)

    incomplete = copy.deepcopy(config)
    incomplete["training"]["development_quality_gate"]["required_checks"].pop()
    with pytest.raises(ValueError, match="extents distinct"):
        validate_paper_config(incomplete)

    invalid = copy.deepcopy(config)
    invalid["training"]["development_quality_gate"]["policy"][
        "min_dynamic_power_ratio"
    ] = -1.0
    with pytest.raises(ValueError, match="policy is invalid"):
        validate_paper_config(invalid)

    weakened = copy.deepcopy(config)
    weakened["training"]["development_quality_gate"]["policy"][
        "min_dynamic_power_ratio"
    ] = 0.25
    with pytest.raises(ValueError, match="weaker than the frozen pre-seal"):
        validate_paper_config(weakened)
