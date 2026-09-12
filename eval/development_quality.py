"""Authenticated development-only quality attestations for synthesis training.

These gates are implementation release policy, not manuscript metrics or
published thresholds.  They consume only the fixed patient-disjoint
development partition.  A routine first-N record is diagnostic; only a passing
all-development record bound to the exact model state can authorize a
selectable completed checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_MAPPING_SHA256,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
)
from data.provenance import canonical_sha256, sha256_file
from eval.quality import (
    DEVELOPMENT_REQUIRED_CHECKS,
    POLICY_NOTE,
    require_release_quality_policy,
)
from utils.figure1_gpu_gate import atomic_write_json_no_replace
from utils.compat import strict_zip


DEVELOPMENT_QUALITY_CONFIG_SCHEMA = "connect4-development-quality-policy-v2"
DEVELOPMENT_QUALITY_RECORD_SCHEMA = "connect4-development-quality-attestation-v1"
CHECKPOINT_CANDIDATE_SCHEMA = "connect4-checkpoint-candidate-identity-v1"
ROUTINE_TIER = "routine-first-n"
SELECTION_TIER = "selection-all-development"

QUALITY_CHECK_NAMES = (
    "grid_affine_tr_contract",
    "support_dice",
    "support_center_of_mass",
    "mask_field_of_view",
    "outside_mask_leakage",
    "spatial_robust_range_ratio",
    "spatial_gradient_ratio",
    "spatial_laplacian_ratio",
    "spatial_high_frequency_ratio",
    "spatial_high_frequency_correlation",
    "dynamic_high_frequency_ratio",
    "dynamic_high_frequency_correlation",
    "temporal_variance_ratio",
    "dvars_ratio",
    "dynamic_power_ratio",
    "effective_rank_ratio",
    "near_static_voxel_fraction",
    "structured_temporal_available",
    "roi_fc_correlation",
    "roi_power_spectrum_correlation",
)

_SOURCE_SHA_FIELDS = (
    "brainlm_target_artifact_identity_sha256",
    "brainlm_scan_context_record_sha256",
    "brainlm_dataset_state_record_sha256",
    "brainlm_dataset_state_sha256",
    "brainlm_support_tensor_sha256",
    "brainlm_prepared_mask_sha256",
    "brainlm_padded_mask_artifact_descriptor_sha256",
    "brainlm_structural_source_identity_sha256",
    "brainlm_native_preprocessing_source_sha256",
    "brainlm_native_alignment_authority_sha256",
    "roi_mapping_sha256",
)

_QUALITY_FIELDS = {
    "schema",
    "policy_note",
    "policy",
    "implementation_sources",
    "sources",
    "geometry",
    "support",
    "outside_mask",
    "spatial",
    "temporal",
    "structured_temporal",
    "paper_metrics",
    "checks",
    "failed_checks",
    "failed_temporal_checks",
    "failed_temporal_collapse_checks",
    "failed_structured_temporal_checks",
    "temporal_quality_mismatch_detected",
    "temporal_collapse_detected",
    "passed",
    "verdict",
}

_STRUCTURED_TEMPORAL_FIELDS = {
    "required",
    "roi_labels_supplied",
    "phase_invariant",
    "roi_count",
    "roi_labels",
    "roi_voxel_counts",
    "power_spectrum_non_dc_bins",
    "issues",
    "temporally_valid_real_roi_count",
    "excluded_constant_real_roi_labels",
    "roi_mean_time_series_shape",
    "available",
    "unavailable_reason",
    "constant_predicted_roi_labels",
    "fc_upper_triangle_edges",
    "fc_matrix_correlation",
    "power_spectrum_correlation",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a tensor's exact dtype, shape, and contiguous bytes."""

    if not torch.is_tensor(value):
        raise TypeError("tensor identity requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _binary_sha256(value: torch.Tensor) -> str:
    binary = (value.detach().cpu() > 0.5).to(torch.uint8).contiguous()
    return hashlib.sha256(binary.numpy().tobytes(order="C")).hexdigest()


def evaluated_tensor_set_identity(
    *,
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    brain_mask: Mapping[str, Any],
    target_validity_mask: Mapping[str, Any],
    roi_masks: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit every tensor consumed by one paired-quality evaluation."""

    record = {
        "schema": "connect4-development-evaluated-tensor-set-v2",
        "target_identity_sha256": canonical_sha256(target),
        "prediction_identity_sha256": canonical_sha256(prediction),
        "brain_mask_identity_sha256": canonical_sha256(brain_mask),
        "target_validity_mask_identity_sha256": canonical_sha256(
            target_validity_mask
        ),
        "roi_masks_identity_sha256": canonical_sha256(roi_masks),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def development_tensor_identity(
    value: torch.Tensor,
    *,
    kind: str,
    brain_mask: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    source_identity: Mapping[str, Any] | None = None,
    target_validity_contract: str | None = None,
) -> dict[str, Any]:
    """Describe exact QA tensors so a checkpoint cannot substitute ROI/grid data."""

    if not torch.is_tensor(value):
        raise TypeError("development QA identity requires a tensor")
    tensor = value.detach().cpu().contiguous()
    identity: dict[str, Any] = {
        "schema": "connect4-development-tensor-identity-v1",
        "kind": kind,
        "sha256": tensor_sha256(tensor),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all()),
    }
    if kind in {"target", "prediction"}:
        if tensor.ndim != 4:
            raise ValueError(f"development {kind} must be [T,D,H,W]")
        identity["in_unit_interval"] = bool(
            identity["finite"] and (tensor >= 0).all() and (tensor <= 1).all()
        )
    elif kind == "brain_mask":
        if tensor.ndim != 3:
            raise ValueError("development brain mask must be [D,H,W]")
        if not isinstance(source_identity, Mapping):
            raise ValueError("development brain-mask identity requires source lineage")
        binary = bool(((tensor == 0) | (tensor == 1)).all())
        native_shape = list(tensor.shape)
        padded_shape = source_identity.get("padded_shape")
        source_native_shape = source_identity.get("native_shape")
        padding_before = source_identity.get("padding_before")
        padding_after = source_identity.get("padding_after")
        if (
            source_native_shape != native_shape
            or not isinstance(padded_shape, list)
            or not isinstance(padding_before, list)
            or not isinstance(padding_after, list)
            or len(padded_shape) != 3
            or len(padding_before) != 3
            or len(padding_after) != 3
            or padded_shape
            != [
                padding_before[index] + native_shape[index] + padding_after[index]
                for index in range(3)
            ]
        ):
            raise ValueError("development brain-mask crop lineage is invalid")
        padded = torch.zeros(tuple(padded_shape), dtype=torch.uint8)
        slices = tuple(
            slice(start, start + size)
            for start, size in strict_zip(padding_before, native_shape)
        )
        padded[slices] = (tensor > 0.5).to(torch.uint8)
        padded_binary_sha256 = hashlib.sha256(
            padded.contiguous().numpy().tobytes(order="C")
        ).hexdigest()
        source_support_sha256 = source_identity.get("brainlm_support_tensor_sha256")
        if padded_binary_sha256 != source_support_sha256:
            raise ValueError(
                "development brain mask differs from admitted padded support"
            )
        identity.update(
            {
                "binary": binary,
                "positive_voxels": int((tensor > 0).sum()),
                "native_binary_sha256": _binary_sha256(tensor),
                "padded_binary_sha256": padded_binary_sha256,
                "source_support_tensor_sha256": source_support_sha256,
                "padding_lineage_sha256": canonical_sha256(
                    {
                        "padded_shape": padded_shape,
                        "native_shape": native_shape,
                        "padding_before": padding_before,
                        "padding_after": padding_after,
                    }
                ),
            }
        )
    elif kind == "target_validity_mask":
        if tensor.ndim != 3:
            raise ValueError(
                "development target-validity mask must be [D,H,W]"
            )
        if brain_mask is None or not torch.is_tensor(brain_mask):
            raise ValueError(
                "development target-validity identity requires the brain mask"
            )
        if target is None or not torch.is_tensor(target):
            raise ValueError(
                "development target-validity identity requires the paired target"
            )
        structural = brain_mask.detach().cpu().contiguous()
        paired_target = target.detach().cpu().contiguous()
        if tuple(structural.shape) != tuple(tensor.shape):
            raise ValueError("development validity/structural-mask grids differ")
        if paired_target.ndim != 4 or tuple(paired_target.shape[1:]) != tuple(
            tensor.shape
        ):
            raise ValueError("development validity/target grids differ")
        binary = bool(((tensor == 0) | (tensor == 1)).all())
        observed = paired_target.ne(0).any(dim=0)
        if not binary or not torch.equal(tensor.bool(), observed):
            raise ValueError(
                "development target-validity mask differs from exact target support"
            )
        if target_validity_contract is None:
            # Retain the established standalone diagnostic default while every
            # production training caller supplies its admitted profile value.
            target_validity_contract = (
                TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[
                    "paper-a4-adni-fmriprep-v1"
                ]
            )
        if target_validity_contract not in frozenset(
            TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.values()
        ):
            raise ValueError(
                "development target-validity identity has an unknown protocol "
                "contract"
            )
        identity.update(
            {
                "binary": binary,
                "positive_voxels": int(tensor.bool().sum()),
                "binary_sha256": _binary_sha256(tensor),
                "brain_mask_binary_sha256": _binary_sha256(structural),
                "outside_brain_voxels": int(
                    (tensor.bool() & ~structural.bool()).sum()
                ),
                "target_tensor_sha256": tensor_sha256(paired_target),
                "derivation_contract": target_validity_contract,
            }
        )
    elif kind == "roi_masks":
        if tensor.ndim != 4:
            raise ValueError("development ROI masks must be [R,D,H,W]")
        if brain_mask is None or not torch.is_tensor(brain_mask):
            raise ValueError("development ROI identity requires the brain mask")
        if not isinstance(source_identity, Mapping):
            raise ValueError("development ROI identity requires source lineage")
        mask = brain_mask.detach().cpu().contiguous()
        if tuple(mask.shape) != tuple(tensor.shape[1:]):
            raise ValueError("development ROI and brain-mask grids differ")
        binary = bool(((tensor == 0) | (tensor == 1)).all())
        roi_binary = tensor > 0
        counts = roi_binary.reshape(tensor.shape[0], -1).sum(dim=1)
        overlap = roi_binary.sum(dim=0)
        roi_mapping_sha256 = source_identity.get("roi_mapping_sha256")
        if (
            roi_mapping_sha256 == CANONICAL_ROI_MAPPING_SHA256
            and tensor.shape[0] != len(CANONICAL_ROI_LABEL_IDS)
        ):
            raise ValueError(
                "development ROI tensor differs from the canonical 32-ROI extent"
            )
        channel_label_ids = (
            tuple(CANONICAL_ROI_LABEL_IDS)
            if roi_mapping_sha256 == CANONICAL_ROI_MAPPING_SHA256
            else tuple(range(1, tensor.shape[0] + 1))
        )
        nonempty_labels = [
            label_id
            for label_id, count in strict_zip(channel_label_ids, counts.tolist())
            if count > 0
        ]
        identity.update(
            {
                "binary": binary,
                "binary_sha256": _binary_sha256(tensor),
                "channel_binary_sha256": [
                    _binary_sha256(tensor[index]) for index in range(tensor.shape[0])
                ],
                "roi_union_binary_sha256": _binary_sha256(roi_binary.any(dim=0)),
                "brain_mask_binary_sha256": _binary_sha256(mask),
                "nonoverlapping": bool((overlap <= 1).all()),
                "outside_brain_voxels": int(
                    (roi_binary.any(dim=0) & ~(mask > 0)).sum()
                ),
                "positive_voxels": int(counts.sum()),
                "nonempty_roi_labels": nonempty_labels,
                "channel_voxel_counts": [int(count) for count in counts.tolist()],
                "source_prepared_mask_sha256": source_identity.get(
                    "brainlm_prepared_mask_sha256"
                ),
                "source_padded_mask_descriptor_sha256": source_identity.get(
                    "brainlm_padded_mask_artifact_descriptor_sha256"
                ),
                "source_structural_identity_sha256": source_identity.get(
                    "brainlm_structural_source_identity_sha256"
                ),
                "roi_mapping_sha256": roi_mapping_sha256,
            }
        )
    else:
        raise ValueError(f"unsupported development tensor identity kind {kind!r}")
    return identity


def model_state_sha256(model_or_state: torch.nn.Module | Mapping[str, Any]) -> str:
    """Hash the exact state dictionary independently of serialization order."""

    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, torch.nn.Module)
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TypeError("model state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_candidate_identity(
    *,
    model_state_digest: str,
    completed_batches: int,
    next_epoch: int,
    next_batch_index: int,
    partial: bool,
) -> dict[str, Any]:
    if not _is_sha256(model_state_digest):
        raise ValueError("checkpoint candidate model-state digest is invalid")
    for label, value in (
        ("completed_batches", completed_batches),
        ("next_epoch", next_epoch),
        ("next_batch_index", next_batch_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint candidate {label} is invalid")
    record = {
        "schema": CHECKPOINT_CANDIDATE_SCHEMA,
        "model_state_sha256": model_state_digest,
        "completed_batches": completed_batches,
        "next_epoch": next_epoch,
        "next_batch_index": next_batch_index,
        "partial": bool(partial),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def seal_development_quality_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record.pop("record_sha256", None)
    record["record_sha256"] = canonical_sha256(record)
    return record


def _validate_candidate(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("development QA has no checkpoint-candidate identity")
    candidate = dict(value)
    claimed = candidate.pop("record_sha256", None)
    if (
        candidate.get("schema") != CHECKPOINT_CANDIDATE_SCHEMA
        or not _is_sha256(claimed)
        or canonical_sha256(candidate) != claimed
        or not _is_sha256(candidate.get("model_state_sha256"))
        or not isinstance(candidate.get("partial"), bool)
    ):
        raise RuntimeError("development QA checkpoint-candidate identity is invalid")
    for key in ("completed_batches", "next_epoch", "next_batch_index"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("development QA checkpoint counters are invalid")
    candidate["record_sha256"] = claimed
    return candidate


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _validate_tensor_identity(value: object, *, kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"development QA {kind} identity is missing")
    identity = dict(value)
    common = {"schema", "kind", "sha256", "dtype", "shape", "finite"}
    extra = {
        "target": {"in_unit_interval"},
        "prediction": {"in_unit_interval"},
        "brain_mask": {
            "binary",
            "positive_voxels",
            "native_binary_sha256",
            "padded_binary_sha256",
            "source_support_tensor_sha256",
            "padding_lineage_sha256",
        },
        "target_validity_mask": {
            "binary",
            "positive_voxels",
            "binary_sha256",
            "brain_mask_binary_sha256",
            "outside_brain_voxels",
            "target_tensor_sha256",
            "derivation_contract",
        },
        "roi_masks": {
            "binary",
            "binary_sha256",
            "channel_binary_sha256",
            "roi_union_binary_sha256",
            "brain_mask_binary_sha256",
            "nonoverlapping",
            "outside_brain_voxels",
            "positive_voxels",
            "nonempty_roi_labels",
            "channel_voxel_counts",
            "source_prepared_mask_sha256",
            "source_padded_mask_descriptor_sha256",
            "source_structural_identity_sha256",
            "roi_mapping_sha256",
        },
    }[kind]
    shape = identity.get("shape")
    expected_ndim = 3 if kind in {"brain_mask", "target_validity_mask"} else 4
    if (
        set(identity) != common | extra
        or identity.get("schema") != "connect4-development-tensor-identity-v1"
        or identity.get("kind") != kind
        or not _is_sha256(identity.get("sha256"))
        or not isinstance(identity.get("dtype"), str)
        or not identity["dtype"].startswith("torch.")
        or identity.get("finite") is not True
        or not isinstance(shape, list)
        or len(shape) != expected_ndim
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in shape
        )
    ):
        raise RuntimeError(f"development QA {kind} tensor identity is invalid")
    if kind in {"target", "prediction"}:
        if identity.get("in_unit_interval") is not True:
            raise RuntimeError(f"development QA {kind} is outside [0,1]")
        return identity
    if identity.get("binary") is not True:
        raise RuntimeError(f"development QA {kind} is not exactly binary")
    positive = identity.get("positive_voxels")
    if isinstance(positive, bool) or not isinstance(positive, int) or positive < 1:
        raise RuntimeError(f"development QA {kind} has no positive voxels")
    if kind == "brain_mask":
        if (
            positive > int(np.prod(shape))
            or any(
                not _is_sha256(identity.get(field))
                for field in (
                    "native_binary_sha256",
                    "padded_binary_sha256",
                    "source_support_tensor_sha256",
                    "padding_lineage_sha256",
                )
            )
            or identity["padded_binary_sha256"]
            != identity["source_support_tensor_sha256"]
        ):
            raise RuntimeError("development QA brain-mask voxel count is invalid")
        return identity
    if kind == "target_validity_mask":
        if (
            positive > int(np.prod(shape))
            or any(
                not _is_sha256(identity.get(field))
                for field in (
                    "binary_sha256",
                    "brain_mask_binary_sha256",
                    "target_tensor_sha256",
                )
            )
            or identity.get("outside_brain_voxels") != 0
            or identity.get("derivation_contract")
            not in frozenset(
                TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.values()
            )
        ):
            raise RuntimeError(
                "development QA target-validity identity is invalid"
            )
        return identity
    counts = identity.get("channel_voxel_counts")
    labels = identity.get("nonempty_roi_labels")
    channel_label_ids = (
        tuple(CANONICAL_ROI_LABEL_IDS)
        if identity.get("roi_mapping_sha256") == CANONICAL_ROI_MAPPING_SHA256
        else tuple(range(1, shape[0] + 1))
    )
    expected_labels = (
        [
            label_id
            for label_id, count in strict_zip(channel_label_ids, counts)
            if count > 0
        ]
        if isinstance(counts, list)
        and len(counts) == shape[0]
        and all(
            not isinstance(count, bool) and isinstance(count, int) and count >= 0
            for count in counts
        )
        else None
    )
    if (
        any(
            not _is_sha256(identity.get(field))
            for field in (
                "binary_sha256",
                "roi_union_binary_sha256",
                "brain_mask_binary_sha256",
                "source_prepared_mask_sha256",
                "source_padded_mask_descriptor_sha256",
                "source_structural_identity_sha256",
                "roi_mapping_sha256",
            )
        )
        or not isinstance(identity.get("channel_binary_sha256"), list)
        or len(identity["channel_binary_sha256"]) != shape[0]
        or any(not _is_sha256(item) for item in identity["channel_binary_sha256"])
        or identity.get("nonoverlapping") is not True
        or identity.get("outside_brain_voxels") != 0
        or (
            identity.get("roi_mapping_sha256") == CANONICAL_ROI_MAPPING_SHA256
            and shape[0] != len(CANONICAL_ROI_LABEL_IDS)
        )
        or expected_labels is None
        or labels != expected_labels
        or sum(counts) != positive
        or not labels
    ):
        raise RuntimeError("development QA ROI tensor identity is invalid")
    return identity


def _validate_source_identity(
    value: object,
    *,
    scan_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("development QA source identity is missing")
    source = dict(value)
    claimed = source.pop("record_sha256", None)
    expected_fields = {
        "schema",
        "scan_id",
        "scan_role",
        *_SOURCE_SHA_FIELDS,
        "padded_shape",
        "native_shape",
        "padding_before",
        "padding_after",
        "padded_affine_ras_mm",
        "padded_affine_sha256",
        "native_affine_ras_mm",
        "native_affine_sha256",
    }
    if (
        set(source) != expected_fields
        or source.get("schema") != "connect4-development-source-identity-v1"
        or source.get("scan_id") != scan_id
        or source.get("scan_role") != "development-validation"
        or not _is_sha256(claimed)
        or canonical_sha256(source) != claimed
        or any(not _is_sha256(source.get(field)) for field in _SOURCE_SHA_FIELDS)
    ):
        raise RuntimeError("development QA source identity differs from its row")
    triplets: dict[str, list[int]] = {}
    for field, positive in (
        ("padded_shape", True),
        ("native_shape", True),
        ("padding_before", False),
        ("padding_after", False),
    ):
        item = source.get(field)
        minimum = 1 if positive else 0
        if (
            not isinstance(item, list)
            or len(item) != 3
            or any(
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < minimum
                for number in item
            )
        ):
            raise RuntimeError("development QA source crop geometry is invalid")
        triplets[field] = item
    if triplets["padded_shape"] != [
        triplets["padding_before"][index]
        + triplets["native_shape"][index]
        + triplets["padding_after"][index]
        for index in range(3)
    ]:
        raise RuntimeError("development QA source crop does not reconstruct its grid")
    affines: dict[str, np.ndarray] = {}
    for field in ("padded_affine_ras_mm", "native_affine_ras_mm"):
        affine = np.asarray(source.get(field), dtype=np.float64)
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12
        ):
            raise RuntimeError("development QA source affine is invalid")
        affines[field] = affine
    if (
        source.get("padded_affine_sha256")
        != canonical_sha256(source["padded_affine_ras_mm"])
        or source.get("native_affine_sha256")
        != canonical_sha256(source["native_affine_ras_mm"])
        or not _is_sha256(source.get("padded_affine_sha256"))
        or not _is_sha256(source.get("native_affine_sha256"))
    ):
        raise RuntimeError("development QA source affine digest is invalid")
    expected_native = affines["padded_affine_ras_mm"].copy()
    expected_native[:3, 3] += expected_native[:3, :3] @ np.asarray(
        triplets["padding_before"], dtype=np.float64
    )
    if not np.allclose(
        expected_native,
        affines["native_affine_ras_mm"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("development QA native affine is not the exact inverse crop")
    source["record_sha256"] = claimed
    return source


def _validate_quality_payload(
    quality: object,
    *,
    policy: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    brain_mask: Mapping[str, Any],
    target_validity_mask: Mapping[str, Any],
    roi_masks: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if not isinstance(quality, Mapping) or set(quality) != _QUALITY_FIELDS:
        raise RuntimeError("development QA paired-quality payload is incomplete")
    result = dict(quality)
    geometry = result.get("geometry")
    geometry_fields = {
        "shape",
        "tr_seconds",
        "affine",
        "affine_max_abs_difference",
        "canonical_axis_codes",
        "original_axis_codes",
        "voxel_sizes_mm",
        "mask_voxels",
        "mask_derived_from_real",
        "array_entry_point",
        "resampling_performed",
    }
    if (
        result.get("schema") != "connect4-paired-4d-quality-v1"
        or result.get("policy_note") != POLICY_NOTE
        or result.get("policy") != policy
        or not isinstance(result.get("implementation_sources"), Mapping)
        or not result["implementation_sources"]
        or result.get("sources")
        != {
            "array_identity": dict(source),
            "evaluated_tensor_set_identity": evaluated_tensor_set_identity(
                target=target,
                prediction=prediction,
                brain_mask=brain_mask,
                target_validity_mask=target_validity_mask,
                roi_masks=roi_masks,
            ),
        }
        or not isinstance(geometry, Mapping)
        or set(geometry) != geometry_fields
        or geometry.get("mask_derived_from_real") is not False
        or geometry.get("array_entry_point") is not True
        or geometry.get("resampling_performed") is not False
        or geometry.get("affine_max_abs_difference") != 0.0
    ):
        raise RuntimeError("development QA grid/source evidence is incomplete")
    implementation_sources = result["implementation_sources"]
    for item in implementation_sources.values():
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("relative_path"), str)
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 1
        ):
            raise RuntimeError(
                "development QA implementation-source evidence is invalid"
            )
    geometry_shape = geometry.get("shape")
    affine = np.asarray(geometry.get("affine"), dtype=np.float64)
    tr_seconds = _finite_number(geometry.get("tr_seconds"))
    voxel_sizes = geometry.get("voxel_sizes_mm")
    if (
        not isinstance(geometry_shape, list)
        or len(geometry_shape) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in geometry_shape
        )
        or affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12
        or tr_seconds is None
        or tr_seconds <= 0.0
        or not isinstance(voxel_sizes, list)
        or len(voxel_sizes) != 3
        or any(
            _finite_number(value) is None or float(value) <= 0 for value in voxel_sizes
        )
        or not isinstance(geometry.get("canonical_axis_codes"), list)
        or len(geometry["canonical_axis_codes"]) != 3
        or not isinstance(geometry.get("original_axis_codes"), Mapping)
    ):
        raise RuntimeError("development QA affine/TR/grid evidence is invalid")
    expected_native_shape = source["native_shape"]
    expected_target_shape = [geometry_shape[3], *geometry_shape[:3]]
    if (
        geometry_shape[:3] != expected_native_shape
        or target["shape"] != expected_target_shape
        or prediction["shape"] != expected_target_shape
        or brain_mask["shape"] != geometry_shape[:3]
        or target_validity_mask["shape"] != geometry_shape[:3]
        or roi_masks["shape"][1:] != geometry_shape[:3]
        or geometry.get("mask_voxels")
        != target_validity_mask["positive_voxels"]
        or target_validity_mask["positive_voxels"]
        > brain_mask["positive_voxels"]
        or roi_masks["positive_voxels"]
        > target_validity_mask["positive_voxels"]
        or target_validity_mask["brain_mask_binary_sha256"]
        != brain_mask["native_binary_sha256"]
        or target_validity_mask["target_tensor_sha256"] != target["sha256"]
        or brain_mask["padded_binary_sha256"] != source["brainlm_support_tensor_sha256"]
        or brain_mask["source_support_tensor_sha256"]
        != source["brainlm_support_tensor_sha256"]
        or roi_masks["brain_mask_binary_sha256"] != brain_mask["native_binary_sha256"]
        or roi_masks["source_prepared_mask_sha256"]
        != source["brainlm_prepared_mask_sha256"]
        or roi_masks["source_padded_mask_descriptor_sha256"]
        != source["brainlm_padded_mask_artifact_descriptor_sha256"]
        or roi_masks["source_structural_identity_sha256"]
        != source["brainlm_structural_source_identity_sha256"]
        or roi_masks["roi_mapping_sha256"] != source["roi_mapping_sha256"]
        or brain_mask["padding_lineage_sha256"]
        != canonical_sha256(
            {
                "padded_shape": source["padded_shape"],
                "native_shape": source["native_shape"],
                "padding_before": source["padding_before"],
                "padding_after": source["padding_after"],
            }
        )
        or canonical_sha256(geometry["affine"]) != source["native_affine_sha256"]
    ):
        raise RuntimeError("development QA target/prediction/mask grid differs")
    checks = result.get("checks")
    if not isinstance(checks, Mapping) or tuple(checks) != QUALITY_CHECK_NAMES:
        raise RuntimeError("development QA quality-check payload is incomplete")
    for name, check in checks.items():
        if (
            not isinstance(check, Mapping)
            or not isinstance(check.get("passed"), bool)
            or "value" not in check
            or not isinstance(check.get("policy"), str)
            or not check["policy"]
        ):
            raise RuntimeError(f"development QA check {name!r} is incomplete")
    grid_value = checks["grid_affine_tr_contract"].get("value")
    if checks["grid_affine_tr_contract"].get("passed") is not True or grid_value != {
        "shape": geometry_shape,
        "affine_max_abs_difference": 0.0,
        "tr_seconds": tr_seconds,
    }:
        raise RuntimeError("development QA affine/TR contract was not passed")
    failed_checks = [name for name in QUALITY_CHECK_NAMES if not checks[name]["passed"]]
    if result.get("failed_checks") != failed_checks:
        raise RuntimeError("development QA failed-check inventory differs")
    passed = not failed_checks
    if (
        result.get("passed") is not passed
        or result.get("verdict")
        not in {"pass", "fail_quality_gate", "fail_temporal_collapse"}
        or (passed and result.get("verdict") != "pass")
    ):
        raise RuntimeError("development QA paired-quality verdict differs")
    structured = result.get("structured_temporal")
    if not isinstance(structured, Mapping):
        raise RuntimeError("development QA structured-temporal evidence is missing")
    if passed:
        if set(structured) != _STRUCTURED_TEMPORAL_FIELDS:
            raise RuntimeError(
                "development QA structured-temporal payload is incomplete"
            )
        labels = roi_masks["nonempty_roi_labels"]
        counts = roi_masks["channel_voxel_counts"]
        valid_count = structured.get("temporally_valid_real_roi_count")
        fc = _finite_number(structured.get("fc_matrix_correlation"))
        spectrum = _finite_number(structured.get("power_spectrum_correlation"))
        if roi_masks["roi_mapping_sha256"] == CANONICAL_ROI_MAPPING_SHA256:
            expected_counts = {
                str(label): count
                for label, count in strict_zip(CANONICAL_ROI_LABEL_IDS, counts)
                if count > 0
            }
        else:
            expected_counts = {str(label): counts[label - 1] for label in labels}
        if (
            structured.get("required") is not True
            or structured.get("roi_labels_supplied") is not True
            or structured.get("phase_invariant") is not True
            or structured.get("available") is not True
            or structured.get("unavailable_reason") is not None
            or structured.get("issues") != []
            or structured.get("roi_count") != len(labels)
            or structured.get("roi_labels") != labels
            or structured.get("roi_voxel_counts") != expected_counts
            or isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or not policy["min_structured_temporal_rois"] <= valid_count <= len(labels)
            or structured.get("roi_mean_time_series_shape")
            != [valid_count, geometry_shape[3]]
            or structured.get("fc_upper_triangle_edges")
            != valid_count * (valid_count - 1) // 2
            or structured.get("power_spectrum_non_dc_bins") != geometry_shape[3] // 2
            or fc is None
            or fc < float(policy["min_roi_fc_correlation"])
            or spectrum is None
            or spectrum < float(policy["min_roi_power_spectrum_correlation"])
            or checks["structured_temporal_available"].get("value") != valid_count
            or checks["roi_fc_correlation"].get("value") != fc
            or checks["roi_power_spectrum_correlation"].get("value") != spectrum
        ):
            raise RuntimeError(
                "development QA ROI/structured-temporal evidence differs"
            )
        metric_sources = {
            "spatial_high_frequency_ratio": ("spatial", "high_frequency_rms_ratio"),
            "spatial_high_frequency_correlation": (
                "spatial",
                "temporal_mean_high_frequency_correlation",
            ),
            "dynamic_high_frequency_ratio": (
                "spatial",
                "dynamic_high_frequency_rms_ratio",
            ),
            "dynamic_high_frequency_correlation": (
                "spatial",
                "dynamic_high_frequency_correlation",
            ),
            "temporal_variance_ratio": ("temporal", "temporal_variance_ratio"),
            "dvars_ratio": ("temporal", "dvars_ratio"),
            "dynamic_power_ratio": ("temporal", "dynamic_power_ratio"),
            "effective_rank_ratio": ("temporal", "effective_rank_ratio"),
            "near_static_voxel_fraction": (
                "temporal",
                "near_static_voxel_fraction",
            ),
        }
        for check_name, (section_name, metric_name) in metric_sources.items():
            section = result.get(section_name)
            if (
                not isinstance(section, Mapping)
                or _finite_number(section.get(metric_name)) is None
                or checks[check_name].get("value") != section.get(metric_name)
            ):
                raise RuntimeError(
                    f"development QA check {check_name!r} is not tied to its metric"
                )
    return result, passed


def validate_development_quality_record(
    value: object,
    *,
    require_selection_pass: bool = False,
) -> dict[str, Any]:
    """Validate a complete signed routine or all-development QA record."""

    if not isinstance(value, Mapping):
        raise RuntimeError("development QA record is missing")
    record = dict(value)
    claimed = record.pop("record_sha256", None)
    if (
        record.get("schema") != DEVELOPMENT_QUALITY_RECORD_SCHEMA
        or not _is_sha256(claimed)
        or canonical_sha256(record) != claimed
    ):
        raise RuntimeError("development QA record signature is invalid")
    tier = record.get("tier")
    if tier not in {ROUTINE_TIER, SELECTION_TIER}:
        raise RuntimeError("development QA tier is invalid")
    if record.get("policy_note") != POLICY_NOTE:
        raise RuntimeError("development QA misstates the implementation-only policy")
    required = record.get("required_checks")
    if required != list(DEVELOPMENT_REQUIRED_CHECKS):
        raise RuntimeError("development QA required-check set is incomplete")
    policy = record.get("policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("development QA policy is missing")
    try:
        require_release_quality_policy(policy)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development QA policy is invalid") from exc
    if record.get("policy_sha256") != canonical_sha256(policy) or not all(
        _is_sha256(record.get(key))
        for key in (
            "config_sha256",
            "cohort_admission_identity_record_sha256",
            "cohort_admission_data_root_sha256",
            "runtime_identity_sha256",
            "artifact_identity_sha256",
            "validation_shard_contract_sha256",
            "model_state_sha256",
            "ordered_scan_ids_sha256",
            "per_scan_records_sha256",
            "prediction_set_sha256",
        )
    ):
        raise RuntimeError("development QA identity bindings are invalid")
    candidate = _validate_candidate(record.get("checkpoint_candidate_identity"))
    if (
        record.get("checkpoint_candidate_identity_sha256") != candidate["record_sha256"]
        or candidate["model_state_sha256"] != record["model_state_sha256"]
        or candidate["completed_batches"] != record.get("step")
    ):
        raise RuntimeError("development QA candidate/model-state binding differs")
    development_size = record.get("development_partition_size")
    global_limit = record.get("global_limit")
    if (
        isinstance(development_size, bool)
        or not isinstance(development_size, int)
        or development_size < 1
        or isinstance(global_limit, bool)
        or not isinstance(global_limit, int)
        or not 1 <= global_limit <= development_size
        or record.get("step") != candidate["completed_batches"]
    ):
        raise RuntimeError("development QA extent or step is invalid")
    positions = record.get("ordered_global_positions")
    scan_ids = record.get("ordered_scan_ids")
    rows = record.get("per_scan_records")
    if (
        positions != list(range(global_limit))
        or not isinstance(scan_ids, list)
        or len(scan_ids) != global_limit
        or len(set(scan_ids)) != global_limit
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in scan_ids)
        or not isinstance(rows, list)
        or len(rows) != global_limit
        or record["ordered_scan_ids_sha256"] != canonical_sha256(scan_ids)
        or record["per_scan_records_sha256"] != canonical_sha256(rows)
    ):
        raise RuntimeError("development QA scan ordering/extent is invalid")
    prediction_identities: list[dict[str, Any]] = []
    row_passes: list[bool] = []
    for position, scan_id, row in strict_zip(positions, scan_ids, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "global_position",
            "scan_id",
            "scan_role",
            "source_identity",
            "target_identity",
            "prediction_identity",
            "brain_mask_identity",
            "target_validity_mask_identity",
            "roi_masks_identity",
            "quality",
            "quality_sha256",
            "passed",
        }:
            raise RuntimeError("development QA per-scan record is invalid")
        if (
            row.get("global_position") != position
            or row.get("scan_id") != scan_id
            or row.get("scan_role") != "development-validation"
        ):
            raise RuntimeError("development QA per-scan identity differs")
        source_identity = _validate_source_identity(
            row.get("source_identity"), scan_id=scan_id
        )
        target = _validate_tensor_identity(row.get("target_identity"), kind="target")
        prediction = _validate_tensor_identity(
            row.get("prediction_identity"), kind="prediction"
        )
        brain_mask = _validate_tensor_identity(
            row.get("brain_mask_identity"), kind="brain_mask"
        )
        target_validity_mask = _validate_tensor_identity(
            row.get("target_validity_mask_identity"),
            kind="target_validity_mask",
        )
        roi_masks = _validate_tensor_identity(
            row.get("roi_masks_identity"), kind="roi_masks"
        )
        quality, quality_passed = _validate_quality_payload(
            row.get("quality"),
            policy=policy,
            source=source_identity,
            target=target,
            prediction=prediction,
            brain_mask=brain_mask,
            target_validity_mask=target_validity_mask,
            roi_masks=roi_masks,
        )
        if row.get("quality_sha256") != canonical_sha256(quality):
            raise RuntimeError("development QA paired-quality digest differs")
        check_values = quality["checks"]
        required_passed = all(
            check_values[name].get("passed") is True
            for name in DEVELOPMENT_REQUIRED_CHECKS
        )
        row_passed = bool(quality_passed and required_passed)
        if row.get("passed") is not row_passed:
            raise RuntimeError("development QA per-scan verdict differs")
        row_passes.append(row_passed)
        prediction_identities.append(dict(prediction))
    passed = bool(all(row_passes))
    if (
        record.get("passed") is not passed
        or record.get("status") != ("PASS" if passed else "FAIL")
        or record["prediction_set_sha256"] != canonical_sha256(prediction_identities)
        or record.get("no_best_subject_selection") is not True
        or record.get("sealed_test_targets_opened") is not False
        or record.get("exact_native_crop_no_resampling") is not True
    ):
        raise RuntimeError("development QA aggregate verdict/boundary is invalid")
    selection = tier == SELECTION_TIER
    if (
        record.get("all_development_evaluated") is not selection
        or record.get("selectable") is not (selection and passed)
        or (selection and global_limit != development_size)
        or (selection and candidate["partial"] is not False)
        or (not selection and candidate["partial"] is not True)
    ):
        raise RuntimeError("development QA tier/selection extent is invalid")
    if require_selection_pass and not (selection and passed):
        raise RuntimeError("an all-development passing QA attestation is required")
    record["record_sha256"] = claimed
    return record


def publish_development_quality_record(
    path: str | Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Publish once, or accept an already published byte-identical record."""

    validated = validate_development_quality_record(record)
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError("development QA destination is not a regular file")
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing development QA record is unreadable") from exc
        if existing != validated:
            raise FileExistsError(
                f"refusing to replace a different development QA record {destination}"
            )
    else:
        atomic_write_json_no_replace(destination, validated)
    return validated, sha256_file(destination)


def validate_selectable_checkpoint_development_qa(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an exact-state, all-development pass in a completed checkpoint."""

    if checkpoint.get("selection_eligible") is not True:
        raise RuntimeError("synthesis checkpoint is not development-QA selectable")
    qa = validate_development_quality_record(
        checkpoint.get("selection_development_qa"),
        require_selection_pass=True,
    )
    candidate = qa["checkpoint_candidate_identity"]
    config = checkpoint.get("config")
    data_config = config.get("data", {}) if isinstance(config, Mapping) else {}
    expected_frames = data_config.get("num_frames")
    expected_tr = data_config.get("tr_seconds")
    expected_padded_shape = data_config.get("architecture_shape")
    grid_matches_config = (
        not isinstance(expected_frames, bool)
        and isinstance(expected_frames, int)
        and expected_frames > 1
        and _finite_number(expected_tr) is not None
        and float(expected_tr) > 0.0
        and isinstance(expected_padded_shape, list)
        and len(expected_padded_shape) == 3
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item > 0
            for item in expected_padded_shape
        )
        and all(
            row["quality"]["geometry"]["shape"][3] == expected_frames
            and row["quality"]["geometry"]["tr_seconds"] == float(expected_tr)
            and row["source_identity"]["padded_shape"] == expected_padded_shape
            for row in qa["per_scan_records"]
        )
    )
    if (
        checkpoint.get("partial") is not False
        or not grid_matches_config
        or candidate["partial"] is not False
        or candidate["completed_batches"] != checkpoint.get("completed_batches")
        or candidate["next_epoch"] != checkpoint.get("next_epoch")
        or candidate["next_batch_index"] != checkpoint.get("next_batch_index")
        or qa["model_state_sha256"] != model_state_sha256(checkpoint.get("model"))
        or qa["config_sha256"] != canonical_sha256(config)
        or qa["policy"]
        != checkpoint.get("config", {})
        .get("training", {})
        .get("development_quality_gate", {})
        .get("policy")
        or qa["cohort_admission_identity_record_sha256"]
        != checkpoint.get("cohort_admission_identity", {}).get("record_sha256")
        or qa["cohort_admission_data_root_sha256"]
        != checkpoint.get("cohort_admission_identity", {}).get("data_root_sha256")
        or qa["runtime_identity_sha256"]
        != checkpoint.get("runtime_identity", {}).get("canonical_sha256")
        or qa["artifact_identity_sha256"]
        != canonical_sha256(checkpoint.get("artifact_identity"))
        or qa["validation_shard_contract_sha256"]
        != checkpoint.get("selection_validation_shard_contract", {}).get(
            "record_sha256"
        )
        or not _is_sha256(checkpoint.get("selection_development_qa_file_sha256"))
        or checkpoint.get("latest_development_qa") != qa
        or checkpoint.get("latest_development_qa_file_sha256")
        != checkpoint.get("selection_development_qa_file_sha256")
    ):
        raise RuntimeError(
            "completed checkpoint differs from its all-development QA attestation"
        )
    return qa


def validate_progress_checkpoint_development_qa(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow failed/latest QA on a resumable, explicitly non-selectable state."""

    if (
        checkpoint.get("partial") is not True
        or checkpoint.get("selection_eligible") is not False
        or checkpoint.get("selection_development_qa") is not None
        or checkpoint.get("selection_development_qa_file_sha256") is not None
    ):
        raise RuntimeError("progress checkpoint falsely claims selection eligibility")
    latest = checkpoint.get("latest_development_qa")
    file_digest = checkpoint.get("latest_development_qa_file_sha256")
    if latest is None:
        if file_digest is not None:
            raise RuntimeError("progress checkpoint has an orphaned QA file digest")
        return None
    record = validate_development_quality_record(latest)
    if not _is_sha256(file_digest):
        raise RuntimeError("progress checkpoint latest QA file digest is invalid")
    return record


__all__ = [
    "CHECKPOINT_CANDIDATE_SCHEMA",
    "DEVELOPMENT_QUALITY_CONFIG_SCHEMA",
    "DEVELOPMENT_QUALITY_RECORD_SCHEMA",
    "QUALITY_CHECK_NAMES",
    "ROUTINE_TIER",
    "SELECTION_TIER",
    "checkpoint_candidate_identity",
    "development_tensor_identity",
    "evaluated_tensor_set_identity",
    "model_state_sha256",
    "publish_development_quality_record",
    "seal_development_quality_record",
    "tensor_sha256",
    "validate_development_quality_record",
    "validate_progress_checkpoint_development_qa",
    "validate_selectable_checkpoint_development_qa",
]
