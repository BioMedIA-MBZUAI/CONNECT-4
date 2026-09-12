"""Fail-closed data and dynamics QA for paired real/predicted 4D fMRI.

This module is deliberately separate from the paper-reported synthesis metrics.
Its thresholds are configurable implementation policy used to catch broken
grids, content/support displacement, spatial smoothing collapse, and nearly
static 4D predictions before figures or scientific metrics are trusted.

The evaluator never registers or resamples an fMRI volume or mask. A grid,
affine, timing, or mask mismatch is a contract error. Images that share a valid
native grid are reoriented together to canonical RAS+ only for calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import torch

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_MAPPING_SHA256,
)
from .postseal_metrics import (
    METRIC_DEFINITIONS_SOURCE,
    frame_to_frame_correlation,
    mse,
    psnr,
    roi_correlation,
    ssim3d,
    voxel_correlation,
)
from utils.source_provenance import implementation_source_records
from utils.spatial_detail import mask_normalized_high_pass


POLICY_NOTE = (
    "CONNECT-4 implementation QA thresholds; configurable and not "
    "paper-reported synthesis metrics"
)

# The development gate deliberately names the scientific failure modes it must
# enforce.  Keeping this list next to the metric definitions prevents a caller
# from treating an incomplete subset of the paired-quality report as a pass.
DEVELOPMENT_REQUIRED_CHECKS = (
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


class QualityContractError(ValueError):
    """Raised when paired data cannot be safely assessed."""


@dataclass(frozen=True)
class QualityPolicy:
    """Configurable guardrails for a paired 4D fMRI QA run.

    Defaults are conservative engineering guardrails, not manuscript claims.
    Production thresholds must be frozen from the patient-disjoint development
    partition before sealed-target inference; sealed held-out targets must never
    be used to calibrate or relax them.
    """

    affine_atol_mm: float = 1e-4
    tr_atol_seconds: float = 1e-6
    support_fraction_of_real_q99: float = 0.10
    min_support_dice: float = 0.85
    max_support_com_distance_mm: float = 8.0
    max_mask_boundary_fraction: float = 0.01
    max_outside_mask_leakage_ratio: float = 0.01
    min_spatial_robust_range_ratio: float = 0.70
    max_spatial_robust_range_ratio: float = 2.00
    min_spatial_gradient_ratio: float = 0.80
    max_spatial_gradient_ratio: float = 2.00
    min_spatial_laplacian_ratio: float = 0.80
    max_spatial_laplacian_ratio: float = 2.00
    min_spatial_high_frequency_ratio: float = 0.80
    max_spatial_high_frequency_ratio: float = 2.00
    min_spatial_high_frequency_correlation: float = 0.40
    min_dynamic_high_frequency_ratio: float = 0.70
    max_dynamic_high_frequency_ratio: float = 4.00
    min_dynamic_high_frequency_correlation: float = 0.40
    high_frequency_sigma_voxels: float = 1.0
    spatial_sample_frames: int = 8
    min_temporal_variance_ratio: float = 0.50
    max_temporal_variance_ratio: float = 4.00
    min_dvars_ratio: float = 0.70
    max_dvars_ratio: float = 4.00
    min_dynamic_power_ratio: float = 0.50
    max_dynamic_power_ratio: float = 4.00
    min_effective_rank_ratio: float = 0.70
    max_effective_rank_ratio: float = 2.00
    near_static_relative_variance: float = 0.25
    max_near_static_voxel_fraction: float = 0.25
    min_structured_temporal_rois: int = 4
    min_roi_fc_correlation: float = 0.40
    min_roi_power_spectrum_correlation: float = 0.40
    temporal_failures_for_collapse: int = 3
    reference_epsilon: float = 1e-12
    temporal_chunk_voxels: int = 16_384

    def __post_init__(self) -> None:
        positive = (
            "high_frequency_sigma_voxels",
            "min_spatial_robust_range_ratio",
            "max_spatial_robust_range_ratio",
            "min_spatial_gradient_ratio",
            "max_spatial_gradient_ratio",
            "min_spatial_laplacian_ratio",
            "max_spatial_laplacian_ratio",
            "min_spatial_high_frequency_ratio",
            "max_spatial_high_frequency_ratio",
            "min_dynamic_high_frequency_ratio",
            "max_dynamic_high_frequency_ratio",
            "min_temporal_variance_ratio",
            "max_temporal_variance_ratio",
            "min_dvars_ratio",
            "max_dvars_ratio",
            "min_dynamic_power_ratio",
            "max_dynamic_power_ratio",
            "min_effective_rank_ratio",
            "max_effective_rank_ratio",
            "near_static_relative_variance",
            "reference_epsilon",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"QualityPolicy.{name} must be positive and finite")
        nonnegative = (
            "affine_atol_mm",
            "tr_atol_seconds",
            "max_support_com_distance_mm",
            "max_outside_mask_leakage_ratio",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"QualityPolicy.{name} must be nonnegative and finite")
        unit_interval = (
            "support_fraction_of_real_q99",
            "min_support_dice",
            "max_mask_boundary_fraction",
            "max_near_static_voxel_fraction",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"QualityPolicy.{name} must lie in [0, 1]")
        for name in (
            "min_spatial_high_frequency_correlation",
            "min_dynamic_high_frequency_correlation",
            "min_roi_fc_correlation",
            "min_roi_power_spectrum_correlation",
        ):
            correlation = float(getattr(self, name))
            if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
                raise ValueError(f"QualityPolicy.{name} must lie in [-1, 1]")
        for lower, upper in (
            ("min_spatial_robust_range_ratio", "max_spatial_robust_range_ratio"),
            ("min_spatial_gradient_ratio", "max_spatial_gradient_ratio"),
            ("min_spatial_laplacian_ratio", "max_spatial_laplacian_ratio"),
            (
                "min_spatial_high_frequency_ratio",
                "max_spatial_high_frequency_ratio",
            ),
            (
                "min_dynamic_high_frequency_ratio",
                "max_dynamic_high_frequency_ratio",
            ),
            ("min_temporal_variance_ratio", "max_temporal_variance_ratio"),
            ("min_dvars_ratio", "max_dvars_ratio"),
            ("min_dynamic_power_ratio", "max_dynamic_power_ratio"),
            ("min_effective_rank_ratio", "max_effective_rank_ratio"),
        ):
            if float(getattr(self, lower)) > float(getattr(self, upper)):
                raise ValueError(f"QualityPolicy.{lower} exceeds {upper}")
        if (
            isinstance(self.spatial_sample_frames, bool)
            or not isinstance(self.spatial_sample_frames, Integral)
            or self.spatial_sample_frames < 1
        ):
            raise ValueError(
                "QualityPolicy.spatial_sample_frames must be a positive integer"
            )
        if (
            isinstance(self.temporal_chunk_voxels, bool)
            or not isinstance(self.temporal_chunk_voxels, Integral)
            or self.temporal_chunk_voxels < 1
        ):
            raise ValueError(
                "QualityPolicy.temporal_chunk_voxels must be a positive integer"
            )
        if (
            isinstance(self.min_structured_temporal_rois, bool)
            or not isinstance(self.min_structured_temporal_rois, Integral)
            or self.min_structured_temporal_rois < 3
        ):
            raise ValueError(
                "QualityPolicy.min_structured_temporal_rois must be an integer >= 3"
            )
        if (
            isinstance(self.temporal_failures_for_collapse, bool)
            or not isinstance(self.temporal_failures_for_collapse, Integral)
            or not 1 <= self.temporal_failures_for_collapse <= 5
        ):
            raise ValueError(
                "QualityPolicy.temporal_failures_for_collapse must be in [1, 5]"
            )


_RELEASE_POLICY_LOWER_BOUNDS = (
    "min_support_dice",
    "min_spatial_robust_range_ratio",
    "min_spatial_gradient_ratio",
    "min_spatial_laplacian_ratio",
    "min_spatial_high_frequency_ratio",
    "min_spatial_high_frequency_correlation",
    "min_dynamic_high_frequency_ratio",
    "min_dynamic_high_frequency_correlation",
    "min_temporal_variance_ratio",
    "min_dvars_ratio",
    "min_dynamic_power_ratio",
    "min_effective_rank_ratio",
    "near_static_relative_variance",
    "min_structured_temporal_rois",
    "min_roi_fc_correlation",
    "min_roi_power_spectrum_correlation",
)
_RELEASE_POLICY_UPPER_BOUNDS = (
    "affine_atol_mm",
    "tr_atol_seconds",
    "max_support_com_distance_mm",
    "max_mask_boundary_fraction",
    "max_outside_mask_leakage_ratio",
    "max_spatial_robust_range_ratio",
    "max_spatial_gradient_ratio",
    "max_spatial_laplacian_ratio",
    "max_spatial_high_frequency_ratio",
    "max_dynamic_high_frequency_ratio",
    "max_temporal_variance_ratio",
    "max_dvars_ratio",
    "max_dynamic_power_ratio",
    "max_effective_rank_ratio",
    "max_near_static_voxel_fraction",
    "temporal_failures_for_collapse",
)
_RELEASE_POLICY_EXACT_MEASUREMENT_FIELDS = (
    "support_fraction_of_real_q99",
    "high_frequency_sigma_voxels",
    "spatial_sample_frames",
    "reference_epsilon",
    "temporal_chunk_voxels",
)


def require_release_quality_policy(value: Mapping[str, Any]) -> QualityPolicy:
    """Reject a production policy weaker than the frozen development baseline.

    Threshold direction is checked explicitly. Fields that change the metric
    definition rather than simply tightening a bound must remain byte-value
    equivalent to the baseline. This prevents a runtime config or re-signed QA
    record from silently relaxing the pre-seal policy.
    """

    if not isinstance(value, Mapping):
        raise ValueError("release quality policy must be a mapping")
    try:
        policy = QualityPolicy(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("release quality policy is invalid") from exc
    baseline = QualityPolicy()
    weaker = [
        name
        for name in _RELEASE_POLICY_LOWER_BOUNDS
        if float(getattr(policy, name)) < float(getattr(baseline, name))
    ]
    weaker.extend(
        name
        for name in _RELEASE_POLICY_UPPER_BOUNDS
        if float(getattr(policy, name)) > float(getattr(baseline, name))
    )
    if weaker:
        raise ValueError(
            "release quality policy weakens frozen bounds: " + ", ".join(sorted(weaker))
        )
    changed_measurements = [
        name
        for name in _RELEASE_POLICY_EXACT_MEASUREMENT_FIELDS
        if getattr(policy, name) != getattr(baseline, name)
    ]
    if changed_measurements:
        raise ValueError(
            "release quality policy changes frozen metric definitions: "
            + ", ".join(sorted(changed_measurements))
        )
    return policy


def _finite_affine(image: nib.spatialimages.SpatialImage, label: str) -> None:
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise QualityContractError(f"{label} affine must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise QualityContractError(f"{label} affine is singular")


def _header_tr_seconds(image: nib.spatialimages.SpatialImage, label: str) -> float:
    zooms = image.header.get_zooms()
    if len(zooms) < 4:
        raise QualityContractError(f"{label} header has no temporal sampling interval")
    value = float(zooms[3])
    if not np.isfinite(value) or value <= 0:
        raise QualityContractError(f"{label} header TR must be positive and finite")
    unit = image.header.get_xyzt_units()[1]
    factors = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
    if unit not in factors:
        raise QualityContractError(
            f"{label} header must declare sec, msec, or usec time units; got {unit!r}"
        )
    return value * factors[unit]


def _load_pair(
    real_path: str | Path,
    predicted_path: str | Path,
    mask_path: Optional[str | Path],
    policy: QualityPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    real_image = nib.load(str(real_path))
    predicted_image = nib.load(str(predicted_path))
    for label, image in (("real", real_image), ("predicted", predicted_image)):
        if image.ndim != 4:
            raise QualityContractError(f"{label} fMRI must be 4D, got {image.shape}")
        if image.shape[-1] < 2:
            raise QualityContractError(f"{label} fMRI must contain at least two frames")
        _finite_affine(image, label)
    if tuple(real_image.shape) != tuple(predicted_image.shape):
        raise QualityContractError(
            "real and predicted fMRI shapes differ: "
            f"{real_image.shape} vs {predicted_image.shape}"
        )
    affine_difference = float(
        np.max(np.abs(real_image.affine - predicted_image.affine))
    )
    if affine_difference > policy.affine_atol_mm:
        raise QualityContractError(
            "real and predicted fMRI affines differ "
            f"(max absolute difference {affine_difference:g})"
        )
    real_tr = _header_tr_seconds(real_image, "real")
    predicted_tr = _header_tr_seconds(predicted_image, "predicted")
    if abs(real_tr - predicted_tr) > policy.tr_atol_seconds:
        raise QualityContractError(
            f"real and predicted fMRI TR differ: {real_tr:g}s vs {predicted_tr:g}s"
        )

    original_axis_codes = {
        "real": list(nib.aff2axcodes(real_image.affine)),
        "predicted": list(nib.aff2axcodes(predicted_image.affine)),
    }
    canonical_real = nib.as_closest_canonical(real_image)
    canonical_predicted = nib.as_closest_canonical(predicted_image)
    if tuple(canonical_real.shape) != tuple(
        canonical_predicted.shape
    ) or not np.allclose(
        canonical_real.affine,
        canonical_predicted.affine,
        rtol=0.0,
        atol=policy.affine_atol_mm,
    ):
        raise QualityContractError(
            "paired images diverged during canonical orientation"
        )
    real = canonical_real.get_fdata(dtype=np.float32)
    predicted = canonical_predicted.get_fdata(dtype=np.float32)
    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise QualityContractError("real or predicted fMRI contains NaN or infinity")

    mask_derived = mask_path is None
    if mask_path is None:
        mask = np.any(np.abs(real) > 1e-8, axis=-1)
    else:
        mask_image = nib.load(str(mask_path))
        if mask_image.ndim != 3:
            raise QualityContractError(f"brain mask must be 3D, got {mask_image.shape}")
        _finite_affine(mask_image, "brain mask")
        canonical_mask = nib.as_closest_canonical(mask_image)
        if tuple(canonical_mask.shape) != tuple(canonical_real.shape[:3]):
            raise QualityContractError(
                "brain mask spatial shape differs from the paired fMRI grid"
            )
        mask_affine_difference = float(
            np.max(np.abs(canonical_mask.affine - canonical_real.affine))
        )
        if mask_affine_difference > policy.affine_atol_mm:
            raise QualityContractError(
                "brain mask affine differs from the paired fMRI grid "
                f"(max absolute difference {mask_affine_difference:g})"
            )
        mask_values = canonical_mask.get_fdata(dtype=np.float32)
        if not np.isfinite(mask_values).all():
            raise QualityContractError("brain mask contains NaN or infinity")
        if not np.logical_or(mask_values == 0.0, mask_values == 1.0).all():
            raise QualityContractError("brain mask must be exactly binary")
        mask = mask_values.astype(bool, copy=False)
    if not bool(mask.any()):
        raise QualityContractError("brain mask is empty")

    geometry = {
        "shape": list(real.shape),
        "tr_seconds": real_tr,
        "affine": np.asarray(canonical_real.affine, dtype=np.float64).tolist(),
        "affine_max_abs_difference": affine_difference,
        "canonical_axis_codes": list(nib.aff2axcodes(canonical_real.affine)),
        "original_axis_codes": original_axis_codes,
        "voxel_sizes_mm": [
            float(value) for value in nib.affines.voxel_sizes(canonical_real.affine)
        ],
        "mask_voxels": int(mask.sum()),
        "mask_derived_from_real": bool(mask_derived),
    }
    return (
        real,
        predicted,
        mask,
        np.asarray(canonical_real.affine, dtype=np.float64),
        real_tr,
        geometry,
    )


def _load_structural_mask(
    structural_mask_path: str | Path,
    *,
    spatial_shape: tuple[int, int, int],
    affine: np.ndarray,
    policy: QualityPolicy,
) -> np.ndarray:
    """Load an exact independent structural support without resampling it."""
    image = nib.load(str(structural_mask_path))
    if image.ndim != 3:
        raise QualityContractError(
            f"structural brain mask must be 3D, got {image.shape}"
        )
    _finite_affine(image, "structural brain mask")
    canonical = nib.as_closest_canonical(image)
    if tuple(canonical.shape) != spatial_shape:
        raise QualityContractError(
            "structural brain mask spatial shape differs from the paired fMRI grid"
        )
    affine_difference = float(np.max(np.abs(canonical.affine - affine)))
    if affine_difference > policy.affine_atol_mm:
        raise QualityContractError(
            "structural brain mask affine differs from the paired fMRI grid "
            f"(max absolute difference {affine_difference:g})"
        )
    values = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise QualityContractError(
            "structural brain mask contains NaN or infinity"
        )
    if not np.logical_or(values == 0.0, values == 1.0).all():
        raise QualityContractError("structural brain mask must be exactly binary")
    mask = values.astype(bool, copy=False)
    if not bool(mask.any()):
        raise QualityContractError("structural brain mask is empty")
    return mask


def _load_roi_labels(
    roi_labels_path: str | Path,
    spatial_shape: tuple[int, int, int],
    affine: np.ndarray,
    policy: QualityPolicy,
) -> np.ndarray:
    """Load a discrete anatomical parcellation without resampling it."""
    label_image = nib.load(str(roi_labels_path))
    if label_image.ndim != 3:
        raise QualityContractError(f"ROI label map must be 3D, got {label_image.shape}")
    _finite_affine(label_image, "ROI label map")
    canonical_labels = nib.as_closest_canonical(label_image)
    if tuple(canonical_labels.shape) != tuple(spatial_shape):
        raise QualityContractError(
            "ROI label map spatial shape differs from the paired fMRI grid"
        )
    affine_difference = float(np.max(np.abs(canonical_labels.affine - affine)))
    if affine_difference > policy.affine_atol_mm:
        raise QualityContractError(
            "ROI label map affine differs from the paired fMRI grid "
            f"(max absolute difference {affine_difference:g})"
        )
    values = canonical_labels.get_fdata(dtype=np.float64)
    if not np.isfinite(values).all():
        raise QualityContractError("ROI label map contains NaN or infinity")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-5):
        raise QualityContractError("ROI label map must contain discrete integer labels")
    if float(rounded.min()) < 0.0:
        raise QualityContractError(
            "ROI label map must use zero for background and positive ROI labels"
        )
    if float(rounded.max()) > float(np.iinfo(np.int32).max):
        raise QualityContractError("ROI label map contains labels outside int32 range")
    return rounded.astype(np.int32, copy=False)


def _vector_pearson(
    left: np.ndarray,
    right: np.ndarray,
    policy: QualityPolicy,
) -> Optional[float]:
    """Return Pearson correlation for equal finite vectors, or ``None``."""
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if (
        left_values.shape != right_values.shape
        or left_values.size < 2
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        return None
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= policy.reference_epsilon:
        return None
    correlation = float(np.dot(left_values, right_values) / denominator)
    if not np.isfinite(correlation):
        return None
    return float(np.clip(correlation, -1.0, 1.0))


def _required_roi_masks(
    roi_labels: np.ndarray,
    target_validity_mask: np.ndarray,
    *,
    configured_label_ids: Optional[tuple[int, ...]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return configured anatomical ROIs only when every ROI is supervised."""
    observed_labels = np.unique(roi_labels[roi_labels > 0]).astype(np.int64)
    if configured_label_ids is None:
        configured_labels = observed_labels
    else:
        configured_labels = np.asarray(configured_label_ids, dtype=np.int64)
        if not np.array_equal(observed_labels, np.sort(configured_labels)):
            missing = sorted(set(configured_labels.tolist()) - set(observed_labels.tolist()))
            extra = sorted(set(observed_labels.tolist()) - set(configured_labels.tolist()))
            raise QualityContractError(
                "structural ROI labels differ from the canonical 32-ROI set; "
                f"missing={missing}, extra={extra}"
            )
    if configured_labels.size < 1:
        raise QualityContractError("structural ROI label map has no configured ROI")
    roi_masks = np.stack(
        [target_validity_mask & (roi_labels == label) for label in configured_labels],
        axis=0,
    )
    counts = roi_masks.reshape(roi_masks.shape[0], -1).sum(axis=1)
    if np.any(counts < 1):
        missing = [
            int(label)
            for label, count in zip(configured_labels.tolist(), counts.tolist())
            if int(count) < 1
        ]
        raise QualityContractError(
            "every configured ROI must have non-empty target-validity support; "
            f"missing labels: {missing}"
        )
    return configured_labels, roi_masks


def _structured_temporal_quality(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    roi_labels: Optional[np.ndarray],
    policy: QualityPolicy,
    *,
    required: bool,
    configured_roi_label_ids: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Compare phase-invariant temporal structure in anatomical ROIs.

    FC is compared by correlating the strict upper triangles of the real and
    predicted ROI correlation matrices. Spectral structure is compared after
    normalizing each ROI's non-DC periodogram, so the metric assesses spectral
    shape rather than merely reproducing global variance. Neither metric
    requires frame-wise phase agreement between the paired scans.
    """
    if roi_labels is None:
        return {
            "available": False,
            "required": bool(required),
            "roi_labels_supplied": False,
            "unavailable_reason": "roi_label_map_not_supplied",
            "phase_invariant": True,
            "roi_count": 0,
            "roi_labels": [],
            "roi_voxel_counts": {},
            "temporally_valid_real_roi_count": 0,
            "excluded_constant_real_roi_labels": [],
            "roi_mean_time_series_shape": None,
            "fc_upper_triangle_edges": 0,
            "fc_matrix_correlation": None,
            "power_spectrum_non_dc_bins": 0,
            "power_spectrum_correlation": None,
            "issues": ["ROI label map was not supplied"],
        }

    label_ids, roi_masks = _required_roi_masks(
        roi_labels,
        mask,
        configured_label_ids=configured_roi_label_ids,
    )
    voxel_counts: Dict[str, int] = {
        str(int(label)): int(roi.sum())
        for label, roi in zip(label_ids.tolist(), roi_masks)
    }
    real_series = [
        np.mean(np.asarray(real[roi], dtype=np.float64), axis=0)
        for roi in roi_masks
    ]
    predicted_series = [
        np.mean(np.asarray(predicted[roi], dtype=np.float64), axis=0)
        for roi in roi_masks
    ]

    base: Dict[str, Any] = {
        "required": bool(required),
        "roi_labels_supplied": True,
        "phase_invariant": True,
        "roi_count": len(real_series),
        "roi_labels": [int(value) for value in label_ids],
        "roi_voxel_counts": voxel_counts,
        "power_spectrum_non_dc_bins": int(real.shape[-1] // 2),
        "issues": [],
    }
    if len(real_series) < policy.min_structured_temporal_rois:
        base.update(
            {
                "available": False,
                "unavailable_reason": (
                    "too_few_nonempty_rois: "
                    f"found {len(real_series)}, require at least "
                    f"{policy.min_structured_temporal_rois}"
                ),
                "temporally_valid_real_roi_count": 0,
                "excluded_constant_real_roi_labels": [],
                "roi_mean_time_series_shape": None,
                "fc_upper_triangle_edges": 0,
                "fc_matrix_correlation": None,
                "power_spectrum_correlation": None,
            }
        )
        base["issues"].append(base["unavailable_reason"])
        return base

    real_means = np.stack(real_series, axis=0)
    predicted_means = np.stack(predicted_series, axis=0)
    real_centered = real_means - real_means.mean(axis=1, keepdims=True)
    predicted_centered = predicted_means - predicted_means.mean(axis=1, keepdims=True)
    real_norms = np.linalg.norm(real_centered, axis=1)
    real_valid = real_norms > policy.reference_epsilon
    excluded_labels = label_ids[~real_valid]
    real_centered = real_centered[real_valid]
    predicted_centered = predicted_centered[real_valid]
    valid_labels = label_ids[real_valid]
    base.update(
        {
            "temporally_valid_real_roi_count": int(real_centered.shape[0]),
            "excluded_constant_real_roi_labels": [
                int(value) for value in excluded_labels
            ],
            "roi_mean_time_series_shape": [
                int(real_centered.shape[0]),
                int(real_centered.shape[1]),
            ],
        }
    )
    if real_centered.shape[0] < policy.min_structured_temporal_rois:
        base.update(
            {
                "available": False,
                "unavailable_reason": (
                    "too_few_temporally_valid_real_rois: "
                    f"found {real_centered.shape[0]}, require at least "
                    f"{policy.min_structured_temporal_rois}"
                ),
                "fc_upper_triangle_edges": 0,
                "fc_matrix_correlation": None,
                "power_spectrum_correlation": None,
            }
        )
        base["issues"].append(base["unavailable_reason"])
        return base

    base["available"] = True
    base["unavailable_reason"] = None
    predicted_norms = np.linalg.norm(predicted_centered, axis=1)
    collapsed_prediction = predicted_norms <= policy.reference_epsilon
    collapsed_labels = valid_labels[collapsed_prediction]
    base["constant_predicted_roi_labels"] = [int(value) for value in collapsed_labels]

    roi_count = int(real_centered.shape[0])
    triangle = np.triu_indices(roi_count, k=1)
    base["fc_upper_triangle_edges"] = int(triangle[0].size)
    if bool(collapsed_prediction.any()):
        fc_correlation = None
        base["issues"].append(
            "predicted ROI mean time series is constant for labels "
            + ", ".join(str(int(value)) for value in collapsed_labels)
        )
    else:
        real_unit = real_centered / np.linalg.norm(real_centered, axis=1, keepdims=True)
        predicted_unit = predicted_centered / predicted_norms[:, None]
        # Keep this small ROI-by-time contraction on NumPy's deterministic
        # elementwise reduction path.  Accelerate-backed ``matmul`` has emitted
        # spurious floating-point overflow/invalid warnings for finite,
        # unit-normalized low-amplitude fMRI traces on macOS, even though the
        # resulting matrix was finite.  ``einsum`` is equivalent here and
        # avoids making a valid quality report platform-warning dependent.
        real_fc = np.einsum("it,jt->ij", real_unit, real_unit, optimize=False)
        predicted_fc = np.einsum(
            "it,jt->ij", predicted_unit, predicted_unit, optimize=False
        )
        fc_correlation = _vector_pearson(
            real_fc[triangle], predicted_fc[triangle], policy
        )
        if fc_correlation is None:
            base["issues"].append(
                "ROI FC upper triangle has insufficient variation for correlation"
            )
    base["fc_matrix_correlation"] = fc_correlation

    real_power = np.abs(np.fft.rfft(real_centered, axis=1)[:, 1:]) ** 2
    predicted_power = np.abs(np.fft.rfft(predicted_centered, axis=1)[:, 1:]) ** 2
    real_power_sum = real_power.sum(axis=1, keepdims=True)
    predicted_power_sum = predicted_power.sum(axis=1, keepdims=True)
    if (
        real_power.shape[1] < 2
        or bool((real_power_sum <= policy.reference_epsilon).any())
        or bool((predicted_power_sum <= policy.reference_epsilon).any())
    ):
        spectrum_correlation = None
        base["issues"].append(
            "ROI periodograms have insufficient non-DC power or frequency bins"
        )
    else:
        real_power /= real_power_sum
        predicted_power /= predicted_power_sum
        spectrum_correlation = _vector_pearson(real_power, predicted_power, policy)
        if spectrum_correlation is None:
            base["issues"].append(
                "normalized ROI periodograms have insufficient variation for "
                "correlation"
            )
    base["power_spectrum_correlation"] = spectrum_correlation
    return base


def _paper_metric_value(value: float) -> Dict[str, Any]:
    numeric = float(value)
    if np.isfinite(numeric):
        return {"available": True, "value": numeric, "nonfinite_value": None}
    if np.isposinf(numeric):
        return {
            "available": True,
            "value": None,
            "nonfinite_value": "positive_infinity",
        }
    if np.isneginf(numeric):
        return {
            "available": True,
            "value": None,
            "nonfinite_value": "negative_infinity",
        }
    return {"available": False, "value": None, "reason": "metric_returned_nan"}


def _paper_metric_report(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    roi_labels: Optional[np.ndarray],
    *,
    configured_roi_label_ids: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Compute paired metrics from the isolated post-seal metric module."""
    target_tensor = torch.from_numpy(
        np.ascontiguousarray(real.transpose(3, 0, 1, 2))
    ).unsqueeze(0)
    predicted_tensor = torch.from_numpy(
        np.ascontiguousarray(predicted.transpose(3, 0, 1, 2))
    ).unsqueeze(0)
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
    metrics: Dict[str, Dict[str, Any]] = {
        "mse": _paper_metric_value(mse(predicted_tensor, target_tensor, mask_tensor)),
        "ssim": _paper_metric_value(
            ssim3d(predicted_tensor, target_tensor, mask_tensor)
        ),
        "voxel_corr": _paper_metric_value(
            voxel_correlation(predicted_tensor, target_tensor, mask_tensor)
        ),
        "f2f_corr": _paper_metric_value(
            frame_to_frame_correlation(predicted_tensor, target_tensor, mask_tensor)
        ),
        "psnr": _paper_metric_value(psnr(predicted_tensor, target_tensor, mask_tensor)),
    }
    if roi_labels is None:
        metrics["roi_corr"] = {
            "available": False,
            "value": None,
            "reason": "exact_grid_roi_labels_not_supplied",
        }
        roi_count = 0
    else:
        label_ids, roi_masks = _required_roi_masks(
            roi_labels,
            mask,
            configured_label_ids=configured_roi_label_ids,
        )
        roi_tensor = torch.from_numpy(np.ascontiguousarray(roi_masks)).unsqueeze(0)
        metrics["roi_corr"] = _paper_metric_value(
            roi_correlation(
                predicted_tensor,
                target_tensor,
                roi_tensor,
                mask=mask_tensor,
            )
        )
        roi_count = int(label_ids.size)
    return {
        "schema": "connect4-single-subject-paper-metrics-v1",
        "reporting_only_no_quality_thresholds": True,
        "definitions_source": METRIC_DEFINITIONS_SOURCE,
        "paired_subject_count": 1,
        "roi_count": roi_count,
        "metrics": metrics,
        "distributional_metrics": {
            "fid": {
                "available": False,
                "value": None,
                "reason": "requires_at_least_two_subject_feature_samples_per_distribution",
            },
            "inception_score": {
                "available": False,
                "value": None,
                "reason": "distributional_metric_not_defined_for_one_subject",
            },
        },
    }


def _world_com(support: np.ndarray, affine: np.ndarray) -> Optional[list[float]]:
    coordinates = np.argwhere(support)
    if coordinates.size == 0:
        return None
    voxel_com = coordinates.astype(np.float64).mean(axis=0)
    return [float(value) for value in nib.affines.apply_affine(affine, voxel_com)]


def _support_quality(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    policy: QualityPolicy,
) -> Dict[str, Any]:
    real_rms = np.sqrt(np.mean(np.square(real, dtype=np.float64), axis=-1))
    predicted_rms = np.sqrt(np.mean(np.square(predicted, dtype=np.float64), axis=-1))
    real_q99 = float(np.percentile(real_rms[mask], 99.0))
    if not np.isfinite(real_q99) or real_q99 <= policy.reference_epsilon:
        raise QualityContractError("real fMRI has no measurable in-mask signal support")
    threshold = policy.support_fraction_of_real_q99 * real_q99
    real_support = mask & (real_rms >= threshold)
    predicted_support = mask & (predicted_rms >= threshold)
    intersection = int(np.logical_and(real_support, predicted_support).sum())
    denominator = int(real_support.sum() + predicted_support.sum())
    dice = float(2.0 * intersection / denominator) if denominator else 0.0
    real_com = _world_com(real_support, affine)
    predicted_com = _world_com(predicted_support, affine)
    if real_com is None or predicted_com is None:
        com_distance = None
    else:
        com_distance = float(
            np.linalg.norm(np.asarray(real_com) - np.asarray(predicted_com))
        )

    face_supports = {
        "x_min": mask[0, :, :],
        "x_max": mask[-1, :, :],
        "y_min": mask[:, 0, :],
        "y_max": mask[:, -1, :],
        "z_min": mask[:, :, 0],
        "z_max": mask[:, :, -1],
    }
    face_fractions = {
        name: float(face.sum() / mask.sum()) for name, face in face_supports.items()
    }
    maximum_face_fraction = max(face_fractions.values())
    return {
        "threshold": threshold,
        "threshold_fraction_of_real_q99": policy.support_fraction_of_real_q99,
        "real_rms_q99": real_q99,
        "real_support_voxels": int(real_support.sum()),
        "predicted_support_voxels": int(predicted_support.sum()),
        "intersection_voxels": intersection,
        "dice": dice,
        "real_center_of_mass_world_mm": real_com,
        "predicted_center_of_mass_world_mm": predicted_com,
        "center_of_mass_distance_mm": com_distance,
        "mask_boundary_face_fractions": face_fractions,
        "maximum_mask_boundary_fraction": maximum_face_fraction,
    }


def _outside_mask_quality(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    *,
    reference_real_rms_q99: float,
    policy: QualityPolicy,
) -> Dict[str, Any]:
    """Measure prediction leakage that masked analyses and figures would hide.

    CONNECT-4 targets are zero outside their structural support.  The primary
    leakage score therefore uses the maximum absolute predicted exterior value,
    normalized by the real target's robust in-mask RMS scale.  A maximum-based
    score deliberately catches even an isolated distant corner artifact; RMS
    and energy summaries are retained to describe distributed leakage.
    """
    outside = np.logical_not(mask)
    outside_voxels = int(outside.sum())
    if outside_voxels < 1:
        return {
            "outside_mask_voxels": 0,
            "reference_real_in_mask_rms_q99": float(reference_real_rms_q99),
            "real_outside_mask_max_abs": 0.0,
            "predicted_outside_mask_max_abs": 0.0,
            "predicted_outside_mask_rms": 0.0,
            "predicted_inside_mask_rms": float(
                np.sqrt(np.mean(np.square(predicted[mask], dtype=np.float64)))
            ),
            "predicted_outside_to_inside_rms_ratio": 0.0,
            "predicted_outside_mask_energy_fraction": 0.0,
            "predicted_outside_mask_leakage_ratio": 0.0,
        }

    real_outside_values = np.asarray(real[outside], dtype=np.float64)
    outside_values = np.asarray(predicted[outside], dtype=np.float64)
    inside_values = np.asarray(predicted[mask], dtype=np.float64)
    outside_square_sum = float(np.sum(outside_values * outside_values))
    inside_square_sum = float(np.sum(inside_values * inside_values))
    outside_rms = float(np.sqrt(outside_square_sum / outside_values.size))
    inside_rms = float(np.sqrt(inside_square_sum / inside_values.size))
    outside_max = float(np.max(np.abs(outside_values)))
    total_energy = outside_square_sum + inside_square_sum
    energy_fraction = (
        float(outside_square_sum / total_energy)
        if total_energy > policy.reference_epsilon
        else 0.0
    )
    rms_ratio = (
        float(outside_rms / inside_rms)
        if inside_rms > policy.reference_epsilon
        else (0.0 if outside_rms <= policy.reference_epsilon else None)
    )
    leakage_ratio = float(outside_max / reference_real_rms_q99)
    if not all(
        np.isfinite(value)
        for value in (outside_max, outside_rms, energy_fraction, leakage_ratio)
    ):
        raise QualityContractError("outside-mask leakage metric is non-finite")
    return {
        "outside_mask_voxels": outside_voxels,
        "reference_real_in_mask_rms_q99": float(reference_real_rms_q99),
        "real_outside_mask_max_abs": float(np.max(np.abs(real_outside_values))),
        "predicted_outside_mask_max_abs": outside_max,
        "predicted_outside_mask_rms": outside_rms,
        "predicted_inside_mask_rms": inside_rms,
        "predicted_outside_to_inside_rms_ratio": rms_ratio,
        "predicted_outside_mask_energy_fraction": energy_fraction,
        "predicted_outside_mask_leakage_ratio": leakage_ratio,
    }


def _sample_frames(num_frames: int, requested: int) -> np.ndarray:
    count = min(int(num_frames), int(requested))
    return np.unique(np.rint(np.linspace(0, num_frames - 1, count)).astype(int))


def _spatial_gradient_rms(
    data: np.ndarray, mask: np.ndarray, frames: np.ndarray
) -> float:
    square_sum = 0.0
    count = 0
    for frame_index in frames:
        volume = np.asarray(data[..., frame_index], dtype=np.float64)
        for axis in range(3):
            difference = np.diff(volume, axis=axis)
            left = np.take(mask, np.arange(mask.shape[axis] - 1), axis=axis)
            right = np.take(mask, np.arange(1, mask.shape[axis]), axis=axis)
            values = difference[np.logical_and(left, right)]
            square_sum += float(np.sum(values * values))
            count += int(values.size)
    if count < 1:
        raise QualityContractError("brain mask has no valid adjacent spatial voxels")
    return float(np.sqrt(square_sum / count))


def _spatial_robust_range(
    data: np.ndarray, mask: np.ndarray, frames: np.ndarray
) -> float:
    """Return an outlier-resistant in-mask q95-q05 spatial signal range."""
    values = np.asarray(data[..., frames][mask], dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise QualityContractError("robust spatial range has insufficient values")
    lower, upper = np.percentile(values, (5.0, 95.0))
    result = float(upper - lower)
    if not np.isfinite(result):
        raise QualityContractError("robust spatial range is non-finite")
    return result


def _spatial_laplacian_rms(
    data: np.ndarray, mask: np.ndarray, frames: np.ndarray
) -> float:
    """Return the RMS discrete 3D Laplacian on boundary-safe brain voxels."""
    valid = np.asarray(mask, dtype=bool).copy()
    for axis in range(3):
        centre = [slice(None)] * 3
        previous = [slice(None)] * 3
        following = [slice(None)] * 3
        centre[axis] = slice(1, -1)
        previous[axis] = slice(None, -2)
        following[axis] = slice(2, None)
        supported = np.zeros_like(valid)
        supported[tuple(centre)] = (
            mask[tuple(previous)] & mask[tuple(centre)] & mask[tuple(following)]
        )
        valid &= supported
    count = int(valid.sum()) * int(frames.size)
    if count < 1:
        raise QualityContractError(
            "brain mask has no boundary-safe voxels for a 3D Laplacian"
        )
    square_sum = 0.0
    for frame_index in frames:
        volume = np.asarray(data[..., frame_index], dtype=np.float64)
        laplacian = np.zeros(mask.shape, dtype=np.float64)
        for axis in range(3):
            centre = [slice(None)] * 3
            previous = [slice(None)] * 3
            following = [slice(None)] * 3
            centre[axis] = slice(1, -1)
            previous[axis] = slice(None, -2)
            following[axis] = slice(2, None)
            laplacian[tuple(centre)] += (
                volume[tuple(previous)]
                - 2.0 * volume[tuple(centre)]
                + volume[tuple(following)]
            )
        values = laplacian[valid]
        square_sum += float(np.sum(values * values))
    return float(np.sqrt(square_sum / count))


def _spatial_high_frequency_rms(
    data: np.ndarray,
    mask: np.ndarray,
    frames: np.ndarray,
    sigma: float,
) -> float:
    square_sum = 0.0
    count = 0
    for frame_index in frames:
        volume = np.asarray(data[..., frame_index], dtype=np.float64)
        detail, _interior = mask_normalized_high_pass(volume, mask, sigma)
        values = detail[mask]
        square_sum += float(np.sum(values * values))
        count += int(values.size)
    return float(np.sqrt(square_sum / count))


def _dynamic_high_frequency_summary(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    frames: np.ndarray,
    sigma: float,
    policy: QualityPolicy,
) -> Dict[str, Optional[float]]:
    """Measure frame-varying fine detail, excluding static anatomy.

    A prediction can reproduce the temporal mean's edge energy by adding one
    static anatomical texture to every frame.  It can also preserve aggregate
    variance by moving unrelated temporal traces between voxels.  Neither is
    faithful 4D texture.  This statistic first removes each volume's temporal
    mean, applies the same mask-normalized high-pass operator to deterministic
    paired frames, and then accumulates one correlation over the complete
    boundary-safe spatiotemporal sample.
    """
    real_mean = np.asarray(real.mean(axis=-1), dtype=np.float64)
    predicted_mean = np.asarray(predicted.mean(axis=-1), dtype=np.float64)
    count = 0
    real_sum = 0.0
    predicted_sum = 0.0
    real_square_sum = 0.0
    predicted_square_sum = 0.0
    cross_sum = 0.0
    reference_interior: Optional[np.ndarray] = None
    for frame_index in frames:
        real_detail, real_interior = mask_normalized_high_pass(
            np.asarray(real[..., frame_index], dtype=np.float64) - real_mean,
            mask,
            sigma,
        )
        predicted_detail, predicted_interior = mask_normalized_high_pass(
            np.asarray(predicted[..., frame_index], dtype=np.float64) - predicted_mean,
            mask,
            sigma,
        )
        if not np.array_equal(real_interior, predicted_interior):
            raise QualityContractError("paired dynamic high-pass supports diverged")
        if reference_interior is None:
            reference_interior = real_interior
        elif not np.array_equal(reference_interior, real_interior):
            raise QualityContractError(
                "dynamic high-pass support changed across sampled frames"
            )
        real_values = np.asarray(real_detail[real_interior], dtype=np.float64)
        predicted_values = np.asarray(
            predicted_detail[predicted_interior], dtype=np.float64
        )
        if real_values.size < 1 or not (
            np.isfinite(real_values).all() and np.isfinite(predicted_values).all()
        ):
            raise QualityContractError(
                "dynamic high-frequency detail is empty or non-finite"
            )
        count += int(real_values.size)
        real_sum += float(real_values.sum())
        predicted_sum += float(predicted_values.sum())
        real_square_sum += float(np.dot(real_values, real_values))
        predicted_square_sum += float(np.dot(predicted_values, predicted_values))
        cross_sum += float(np.dot(real_values, predicted_values))
    if count < 2:
        raise QualityContractError(
            "dynamic high-frequency detail has insufficient paired samples"
        )
    real_rms = float(np.sqrt(real_square_sum / count))
    predicted_rms = float(np.sqrt(predicted_square_sum / count))
    ratio = _safe_reference_ratio(
        predicted_rms,
        real_rms,
        "dynamic spatial high-frequency RMS",
        policy,
    )
    real_centered_square = real_square_sum - real_sum * real_sum / count
    predicted_centered_square = (
        predicted_square_sum - predicted_sum * predicted_sum / count
    )
    centered_cross = cross_sum - real_sum * predicted_sum / count
    denominator = float(
        np.sqrt(max(real_centered_square, 0.0) * max(predicted_centered_square, 0.0))
    )
    correlation = (
        None
        if denominator <= policy.reference_epsilon
        else float(centered_cross / denominator)
    )
    if correlation is not None and not np.isfinite(correlation):
        raise QualityContractError(
            "dynamic spatial high-frequency correlation is non-finite"
        )
    return {
        "rms_real": real_rms,
        "rms_predicted": predicted_rms,
        "rms_ratio": ratio,
        "correlation": correlation,
    }


def _masked_pearson(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    policy: QualityPolicy,
) -> Optional[float]:
    """Return a finite masked Pearson coefficient or ``None`` if undefined."""
    left_values = np.asarray(left[mask], dtype=np.float64)
    right_values = np.asarray(right[mask], dtype=np.float64)
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= policy.reference_epsilon:
        return None
    correlation = float(np.dot(left_values, right_values) / denominator)
    return correlation if np.isfinite(correlation) else None


def _safe_reference_ratio(
    predicted_value: float,
    real_value: float,
    label: str,
    policy: QualityPolicy,
) -> float:
    if not np.isfinite(real_value) or real_value <= policy.reference_epsilon:
        raise QualityContractError(
            f"real reference {label} is too small or non-finite for QA"
        )
    ratio = float(predicted_value / real_value)
    if not np.isfinite(ratio):
        raise QualityContractError(f"predicted/reference {label} ratio is non-finite")
    return ratio


def _spatial_quality(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    policy: QualityPolicy,
) -> Dict[str, Any]:
    frames = _sample_frames(real.shape[-1], policy.spatial_sample_frames)
    real_robust_range = _spatial_robust_range(real, mask, frames)
    predicted_robust_range = _spatial_robust_range(predicted, mask, frames)
    real_gradient = _spatial_gradient_rms(real, mask, frames)
    predicted_gradient = _spatial_gradient_rms(predicted, mask, frames)
    real_laplacian = _spatial_laplacian_rms(real, mask, frames)
    predicted_laplacian = _spatial_laplacian_rms(predicted, mask, frames)
    real_high_frequency = _spatial_high_frequency_rms(
        real, mask, frames, policy.high_frequency_sigma_voxels
    )
    predicted_high_frequency = _spatial_high_frequency_rms(
        predicted, mask, frames, policy.high_frequency_sigma_voxels
    )
    # Energy ratios alone can be fooled by spatially unrelated noise or by the
    # wrong anatomical texture. Compare a boundary-safe temporal-mean high-pass
    # pattern inside an eroded brain support as a separate alignment/detail gate.
    real_mean = np.asarray(real.mean(axis=-1), dtype=np.float64)
    predicted_mean = np.asarray(predicted.mean(axis=-1), dtype=np.float64)
    real_mean_high_frequency, interior = mask_normalized_high_pass(
        real_mean,
        mask,
        policy.high_frequency_sigma_voxels,
    )
    predicted_mean_high_frequency, predicted_interior = mask_normalized_high_pass(
        predicted_mean,
        mask,
        policy.high_frequency_sigma_voxels,
    )
    if not np.array_equal(interior, predicted_interior):
        raise QualityContractError("paired high-pass interior supports diverged")
    high_frequency_correlation = _masked_pearson(
        real_mean_high_frequency,
        predicted_mean_high_frequency,
        interior,
        policy,
    )
    dynamic_high_frequency = _dynamic_high_frequency_summary(
        real,
        predicted,
        mask,
        frames,
        policy.high_frequency_sigma_voxels,
        policy,
    )
    return {
        "sampled_frames": [int(value) for value in frames],
        "robust_range_quantiles": [5.0, 95.0],
        "robust_range_real": real_robust_range,
        "robust_range_predicted": predicted_robust_range,
        "robust_range_ratio": _safe_reference_ratio(
            predicted_robust_range,
            real_robust_range,
            "spatial q95-q05 range",
            policy,
        ),
        "gradient_rms_real": real_gradient,
        "gradient_rms_predicted": predicted_gradient,
        "gradient_rms_ratio": _safe_reference_ratio(
            predicted_gradient, real_gradient, "spatial gradient RMS", policy
        ),
        "laplacian_rms_real": real_laplacian,
        "laplacian_rms_predicted": predicted_laplacian,
        "laplacian_rms_ratio": _safe_reference_ratio(
            predicted_laplacian, real_laplacian, "spatial Laplacian RMS", policy
        ),
        "high_frequency_sigma_voxels": policy.high_frequency_sigma_voxels,
        "high_frequency_rms_real": real_high_frequency,
        "high_frequency_rms_predicted": predicted_high_frequency,
        "high_frequency_rms_ratio": _safe_reference_ratio(
            predicted_high_frequency,
            real_high_frequency,
            "spatial high-frequency RMS",
            policy,
        ),
        "temporal_mean_high_frequency_correlation": high_frequency_correlation,
        "dynamic_high_frequency_rms_real": dynamic_high_frequency["rms_real"],
        "dynamic_high_frequency_rms_predicted": dynamic_high_frequency["rms_predicted"],
        "dynamic_high_frequency_rms_ratio": dynamic_high_frequency["rms_ratio"],
        "dynamic_high_frequency_correlation": dynamic_high_frequency["correlation"],
    }


def _effective_rank_and_dynamic_power(
    values: np.ndarray, policy: QualityPolicy
) -> tuple[float, float]:
    num_frames = int(values.shape[1])
    gram = np.zeros((num_frames, num_frames), dtype=np.float64)
    power_sum = 0.0
    voxel_count = 0
    chunk = int(policy.temporal_chunk_voxels)
    for start in range(0, values.shape[0], chunk):
        block = np.asarray(values[start : start + chunk], dtype=np.float64)
        block -= block.mean(axis=1, keepdims=True)
        # Some Accelerate-backed NumPy builds leave floating-point status flags
        # set after finite BLAS products. Validate the product explicitly rather
        # than emitting spurious overflow warnings for normal fMRI values.
        with np.errstate(all="ignore"):
            contribution = block.T @ block
        if not np.isfinite(contribution).all():
            raise QualityContractError("temporal Gram matrix is non-finite")
        gram += contribution
        spectrum = np.fft.rfft(block, axis=1)
        power_sum += float(np.sum(np.abs(spectrum[:, 1:]) ** 2))
        voxel_count += int(block.shape[0])
    eigenvalues = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= policy.reference_epsilon:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    dynamic_power = float(power_sum / max(voxel_count, 1))
    return effective_rank, dynamic_power


def _temporal_quality(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    policy: QualityPolicy,
) -> Dict[str, Any]:
    real_values = np.asarray(real[mask], dtype=np.float32)
    predicted_values = np.asarray(predicted[mask], dtype=np.float32)
    real_variance_by_voxel = np.var(real_values, axis=1, dtype=np.float64)
    predicted_variance_by_voxel = np.var(predicted_values, axis=1, dtype=np.float64)
    real_variance = float(real_variance_by_voxel.mean())
    predicted_variance = float(predicted_variance_by_voxel.mean())

    real_difference = np.diff(real_values.astype(np.float64), axis=1)
    predicted_difference = np.diff(predicted_values.astype(np.float64), axis=1)
    real_dvars_by_frame = np.sqrt(np.mean(real_difference * real_difference, axis=0))
    predicted_dvars_by_frame = np.sqrt(
        np.mean(predicted_difference * predicted_difference, axis=0)
    )
    real_dvars = float(real_dvars_by_frame.mean())
    predicted_dvars = float(predicted_dvars_by_frame.mean())

    real_rank, real_power = _effective_rank_and_dynamic_power(real_values, policy)
    predicted_rank, predicted_power = _effective_rank_and_dynamic_power(
        predicted_values, policy
    )
    if real_rank <= policy.reference_epsilon:
        raise QualityContractError(
            "real reference effective temporal rank is zero; collapse QA is undefined"
        )
    relative_variance = predicted_variance_by_voxel / np.maximum(
        real_variance_by_voxel, policy.reference_epsilon
    )
    near_static_fraction = float(
        np.mean(relative_variance < policy.near_static_relative_variance)
    )
    return {
        "mean_voxelwise_variance_real": real_variance,
        "mean_voxelwise_variance_predicted": predicted_variance,
        "temporal_variance_ratio": _safe_reference_ratio(
            predicted_variance, real_variance, "temporal variance", policy
        ),
        "mean_dvars_real": real_dvars,
        "mean_dvars_predicted": predicted_dvars,
        "dvars_ratio": _safe_reference_ratio(
            predicted_dvars, real_dvars, "DVARS", policy
        ),
        "dynamic_power_real": real_power,
        "dynamic_power_predicted": predicted_power,
        "dynamic_power_ratio": _safe_reference_ratio(
            predicted_power, real_power, "dynamic spectral power", policy
        ),
        "effective_rank_real": real_rank,
        "effective_rank_predicted": predicted_rank,
        "effective_rank_ratio": _safe_reference_ratio(
            predicted_rank, real_rank, "effective temporal rank", policy
        ),
        "relative_variance_threshold": policy.near_static_relative_variance,
        "near_static_voxel_fraction": near_static_fraction,
    }


def _range_check(value: float, lower: float, upper: float) -> bool:
    return bool(np.isfinite(value) and lower <= value <= upper)


def _check(passed: bool, value: Any, policy_description: str) -> Dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "policy": policy_description,
    }


def _unavailable_check(
    policy_description: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "passed": None,
        "value": None,
        "available": False,
        "reason": reason,
        "policy": policy_description,
    }


def _evaluate_loaded_pair(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    tr: float,
    geometry: Mapping[str, Any],
    roi_labels: Optional[np.ndarray],
    *,
    roi_labels_supplied: bool,
    require_structured_temporal: bool,
    policy: QualityPolicy,
    sources: Mapping[str, Any],
    structural_mask: Optional[np.ndarray] = None,
    configured_roi_label_ids: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Apply the single paired-quality implementation to validated arrays."""
    support = _support_quality(real, predicted, mask, affine, policy)
    leakage_mask = mask if structural_mask is None else structural_mask
    outside_mask = _outside_mask_quality(
        real,
        predicted,
        leakage_mask,
        reference_real_rms_q99=support["real_rms_q99"],
        policy=policy,
    )
    spatial = _spatial_quality(real, predicted, mask, policy)
    temporal = _temporal_quality(real, predicted, mask, policy)
    structured_temporal = _structured_temporal_quality(
        real,
        predicted,
        mask,
        roi_labels,
        policy,
        required=require_structured_temporal,
        configured_roi_label_ids=configured_roi_label_ids,
    )
    paper_metrics = _paper_metric_report(
        real,
        predicted,
        mask,
        roi_labels,
        configured_roi_label_ids=configured_roi_label_ids,
    )

    checks: Dict[str, Dict[str, Any]] = {
        "grid_affine_tr_contract": _check(
            True,
            {
                "shape": geometry["shape"],
                "affine_max_abs_difference": geometry["affine_max_abs_difference"],
                "tr_seconds": tr,
            },
            "exact shape; affine and TR within configured tolerances",
        ),
        "support_dice": _check(
            support["dice"] >= policy.min_support_dice,
            support["dice"],
            f">= {policy.min_support_dice:g}",
        ),
        "support_center_of_mass": _check(
            support["center_of_mass_distance_mm"] is not None
            and support["center_of_mass_distance_mm"]
            <= policy.max_support_com_distance_mm,
            support["center_of_mass_distance_mm"],
            f"<= {policy.max_support_com_distance_mm:g} mm",
        ),
        "mask_field_of_view": _check(
            support["maximum_mask_boundary_fraction"]
            <= policy.max_mask_boundary_fraction,
            support["maximum_mask_boundary_fraction"],
            f"maximum boundary-face fraction <= {policy.max_mask_boundary_fraction:g}",
        ),
        "outside_mask_leakage": _check(
            outside_mask["predicted_outside_mask_leakage_ratio"]
            <= policy.max_outside_mask_leakage_ratio,
            {
                "max_abs_to_real_rms_q99_ratio": outside_mask[
                    "predicted_outside_mask_leakage_ratio"
                ],
                "max_abs": outside_mask["predicted_outside_mask_max_abs"],
                "real_max_abs": outside_mask["real_outside_mask_max_abs"],
                "rms": outside_mask["predicted_outside_mask_rms"],
                "energy_fraction": outside_mask[
                    "predicted_outside_mask_energy_fraction"
                ],
            },
            "maximum absolute predicted exterior signal / real in-mask RMS q99 "
            f"<= {policy.max_outside_mask_leakage_ratio:g}",
        ),
        "spatial_robust_range_ratio": _check(
            _range_check(
                spatial["robust_range_ratio"],
                policy.min_spatial_robust_range_ratio,
                policy.max_spatial_robust_range_ratio,
            ),
            spatial["robust_range_ratio"],
            f"in [{policy.min_spatial_robust_range_ratio:g}, "
            f"{policy.max_spatial_robust_range_ratio:g}]",
        ),
        "spatial_gradient_ratio": _check(
            _range_check(
                spatial["gradient_rms_ratio"],
                policy.min_spatial_gradient_ratio,
                policy.max_spatial_gradient_ratio,
            ),
            spatial["gradient_rms_ratio"],
            f"in [{policy.min_spatial_gradient_ratio:g}, "
            f"{policy.max_spatial_gradient_ratio:g}]",
        ),
        "spatial_laplacian_ratio": _check(
            _range_check(
                spatial["laplacian_rms_ratio"],
                policy.min_spatial_laplacian_ratio,
                policy.max_spatial_laplacian_ratio,
            ),
            spatial["laplacian_rms_ratio"],
            f"in [{policy.min_spatial_laplacian_ratio:g}, "
            f"{policy.max_spatial_laplacian_ratio:g}]",
        ),
        "spatial_high_frequency_ratio": _check(
            _range_check(
                spatial["high_frequency_rms_ratio"],
                policy.min_spatial_high_frequency_ratio,
                policy.max_spatial_high_frequency_ratio,
            ),
            spatial["high_frequency_rms_ratio"],
            f"in [{policy.min_spatial_high_frequency_ratio:g}, "
            f"{policy.max_spatial_high_frequency_ratio:g}]",
        ),
        "spatial_high_frequency_correlation": _check(
            spatial["temporal_mean_high_frequency_correlation"] is not None
            and spatial["temporal_mean_high_frequency_correlation"]
            >= policy.min_spatial_high_frequency_correlation,
            spatial["temporal_mean_high_frequency_correlation"],
            f">= {policy.min_spatial_high_frequency_correlation:g}",
        ),
        "dynamic_high_frequency_ratio": _check(
            _range_check(
                spatial["dynamic_high_frequency_rms_ratio"],
                policy.min_dynamic_high_frequency_ratio,
                policy.max_dynamic_high_frequency_ratio,
            ),
            spatial["dynamic_high_frequency_rms_ratio"],
            f"in [{policy.min_dynamic_high_frequency_ratio:g}, "
            f"{policy.max_dynamic_high_frequency_ratio:g}]",
        ),
        "dynamic_high_frequency_correlation": _check(
            spatial["dynamic_high_frequency_correlation"] is not None
            and spatial["dynamic_high_frequency_correlation"]
            >= policy.min_dynamic_high_frequency_correlation,
            spatial["dynamic_high_frequency_correlation"],
            f">= {policy.min_dynamic_high_frequency_correlation:g}",
        ),
        "temporal_variance_ratio": _check(
            _range_check(
                temporal["temporal_variance_ratio"],
                policy.min_temporal_variance_ratio,
                policy.max_temporal_variance_ratio,
            ),
            temporal["temporal_variance_ratio"],
            f"in [{policy.min_temporal_variance_ratio:g}, "
            f"{policy.max_temporal_variance_ratio:g}]",
        ),
        "dvars_ratio": _check(
            _range_check(
                temporal["dvars_ratio"],
                policy.min_dvars_ratio,
                policy.max_dvars_ratio,
            ),
            temporal["dvars_ratio"],
            f"in [{policy.min_dvars_ratio:g}, {policy.max_dvars_ratio:g}]",
        ),
        "dynamic_power_ratio": _check(
            _range_check(
                temporal["dynamic_power_ratio"],
                policy.min_dynamic_power_ratio,
                policy.max_dynamic_power_ratio,
            ),
            temporal["dynamic_power_ratio"],
            f"in [{policy.min_dynamic_power_ratio:g}, "
            f"{policy.max_dynamic_power_ratio:g}]",
        ),
        "effective_rank_ratio": _check(
            _range_check(
                temporal["effective_rank_ratio"],
                policy.min_effective_rank_ratio,
                policy.max_effective_rank_ratio,
            ),
            temporal["effective_rank_ratio"],
            f"in [{policy.min_effective_rank_ratio:g}, "
            f"{policy.max_effective_rank_ratio:g}]",
        ),
        "near_static_voxel_fraction": _check(
            temporal["near_static_voxel_fraction"]
            <= policy.max_near_static_voxel_fraction,
            temporal["near_static_voxel_fraction"],
            f"<= {policy.max_near_static_voxel_fraction:g} at relative variance "
            f"< {policy.near_static_relative_variance:g}",
        ),
    }
    structured_requested = bool(roi_labels_supplied)
    if structured_temporal["available"]:
        checks["structured_temporal_available"] = _check(
            True,
            structured_temporal["temporally_valid_real_roi_count"],
            f">= {policy.min_structured_temporal_rois:g} usable anatomical ROIs",
        )
        checks["roi_fc_correlation"] = _check(
            structured_temporal["fc_matrix_correlation"] is not None
            and structured_temporal["fc_matrix_correlation"]
            >= policy.min_roi_fc_correlation,
            structured_temporal["fc_matrix_correlation"],
            f">= {policy.min_roi_fc_correlation:g}",
        )
        checks["roi_power_spectrum_correlation"] = _check(
            structured_temporal["power_spectrum_correlation"] is not None
            and structured_temporal["power_spectrum_correlation"]
            >= policy.min_roi_power_spectrum_correlation,
            structured_temporal["power_spectrum_correlation"],
            f">= {policy.min_roi_power_spectrum_correlation:g}",
        )
    else:
        availability_reason = str(structured_temporal["unavailable_reason"])
        availability_policy = (
            f">= {policy.min_structured_temporal_rois:g} usable anatomical ROIs"
        )
        if require_structured_temporal or structured_requested:
            checks["structured_temporal_available"] = _check(
                False,
                structured_temporal["temporally_valid_real_roi_count"],
                availability_policy,
            )
            checks["structured_temporal_available"]["reason"] = availability_reason
        else:
            checks["structured_temporal_available"] = _unavailable_check(
                "optional unless ROI labels are supplied or explicitly required",
                availability_reason,
            )
        checks["roi_fc_correlation"] = _unavailable_check(
            f">= {policy.min_roi_fc_correlation:g}", availability_reason
        )
        checks["roi_power_spectrum_correlation"] = _unavailable_check(
            f">= {policy.min_roi_power_spectrum_correlation:g}",
            availability_reason,
        )
    temporal_check_names = (
        "temporal_variance_ratio",
        "dvars_ratio",
        "dynamic_power_ratio",
        "effective_rank_ratio",
        "near_static_voxel_fraction",
    )
    failed_temporal = [
        name for name in temporal_check_names if not checks[name]["passed"]
    ]
    structured_check_names = (
        "structured_temporal_available",
        "roi_fc_correlation",
        "roi_power_spectrum_correlation",
    )
    failed_checks = [name for name, value in checks.items() if value["passed"] is False]
    failed_structured_temporal = [
        name for name in structured_check_names if checks[name]["passed"] is False
    ]
    # A ratio can fail because the prediction is too weak *or* too strong.  It
    # can also fail because its temporal rank is much higher than the reference
    # (for example when an unfiltered real series is dominated by one slow
    # drift).  Those are temporal mismatches, but they are not evidence that the
    # prediction collapsed.  Preserve the same fail-closed gates while reserving
    # the stronger collapse label for multiple low-dynamics/static indicators.
    collapse_failures = []
    if temporal["temporal_variance_ratio"] < policy.min_temporal_variance_ratio:
        collapse_failures.append("temporal_variance_ratio")
    if temporal["dvars_ratio"] < policy.min_dvars_ratio:
        collapse_failures.append("dvars_ratio")
    if temporal["dynamic_power_ratio"] < policy.min_dynamic_power_ratio:
        collapse_failures.append("dynamic_power_ratio")
    if temporal["effective_rank_ratio"] < policy.min_effective_rank_ratio:
        collapse_failures.append("effective_rank_ratio")
    if temporal["near_static_voxel_fraction"] > policy.max_near_static_voxel_fraction:
        collapse_failures.append("near_static_voxel_fraction")
    # A high-rank/noisy or over-dynamic prediction is biologically wrong and
    # remains a hard quality failure, but it is not a static/low-dimensional
    # collapse. Guard against a scale-mismatched low-rank reference causing
    # three low-side ratios to mislabel visibly high-rank dynamics as collapse.
    high_side_mismatch = any(
        (
            temporal["temporal_variance_ratio"]
            > policy.max_temporal_variance_ratio,
            temporal["dvars_ratio"] > policy.max_dvars_ratio,
            temporal["dynamic_power_ratio"] > policy.max_dynamic_power_ratio,
            temporal["effective_rank_ratio"] > policy.max_effective_rank_ratio,
        )
    )
    nontrivial_dynamic_energy = any(
        (
            temporal["temporal_variance_ratio"]
            >= policy.min_temporal_variance_ratio,
            temporal["dvars_ratio"] >= policy.min_dvars_ratio,
            temporal["dynamic_power_ratio"] >= policy.min_dynamic_power_ratio,
        )
    )
    collapse_detected = (
        len(collapse_failures) >= policy.temporal_failures_for_collapse
        and not (high_side_mismatch and nontrivial_dynamic_energy)
    )
    if collapse_detected:
        verdict = "fail_temporal_collapse"
    elif failed_checks:
        verdict = "fail_quality_gate"
    else:
        verdict = "pass"
    return {
        "schema": "connect4-paired-4d-quality-v1",
        "policy_note": POLICY_NOTE,
        "policy": asdict(policy),
        "implementation_sources": implementation_source_records(),
        "sources": dict(sources),
        "geometry": dict(geometry),
        "support": support,
        "outside_mask": outside_mask,
        "spatial": spatial,
        "temporal": temporal,
        "structured_temporal": structured_temporal,
        "paper_metrics": paper_metrics,
        "checks": checks,
        "failed_checks": failed_checks,
        "failed_temporal_checks": failed_temporal,
        "failed_temporal_collapse_checks": collapse_failures,
        "failed_structured_temporal_checks": failed_structured_temporal,
        "temporal_quality_mismatch_detected": bool(failed_temporal),
        "temporal_collapse_detected": collapse_detected,
        "passed": not failed_checks,
        "verdict": verdict,
    }


def evaluate_4d_pair_quality(
    real_path: str | Path,
    predicted_path: str | Path,
    *,
    mask_path: Optional[str | Path] = None,
    structural_mask_path: Optional[str | Path] = None,
    roi_labels_path: Optional[str | Path] = None,
    require_structured_temporal: bool = False,
    require_canonical_roi_mapping: bool = False,
    policy: Optional[QualityPolicy] = None,
) -> Dict[str, Any]:
    """Evaluate a paired 4D fMRI result and return a JSON-serializable record.

    Invalid geometry, timing, mask, values, or reference dynamics raise
    :class:`QualityContractError`. The returned verdict fails when any configured
    gate fails and labels temporal collapse only after the configured number of
    independent temporal gates fail. Structured temporal gates are optional
    when ``roi_labels_path`` is absent. Supplying a label map enables them;
    ``require_structured_temporal=True`` additionally makes absence of a usable
    map a failed quality gate.
    """
    resolved_policy = policy or QualityPolicy()
    real, predicted, mask, affine, tr, geometry = _load_pair(
        real_path, predicted_path, mask_path, resolved_policy
    )
    structural_mask = (
        mask
        if structural_mask_path is None
        else _load_structural_mask(
            structural_mask_path,
            spatial_shape=tuple(int(value) for value in real.shape[:3]),
            affine=affine,
            policy=resolved_policy,
        )
    )
    if bool((mask & ~structural_mask).any()):
        raise QualityContractError(
            "target-validity mask lies outside structural brain support"
        )
    roi_labels = (
        None
        if roi_labels_path is None
        else _load_roi_labels(
            roi_labels_path,
            tuple(int(value) for value in real.shape[:3]),
            affine,
            resolved_policy,
        )
    )
    configured_roi_label_ids = None
    if require_canonical_roi_mapping:
        if roi_labels is None:
            raise QualityContractError(
                "canonical 32-ROI evaluation requires an exact structural label map"
            )
        configured_roi_label_ids = tuple(CANONICAL_ROI_LABEL_IDS)
        _required_roi_masks(
            roi_labels,
            mask,
            configured_label_ids=configured_roi_label_ids,
        )
    return _evaluate_loaded_pair(
        real,
        predicted,
        mask,
        affine,
        tr,
        geometry,
        roi_labels,
        roi_labels_supplied=roi_labels_path is not None,
        require_structured_temporal=require_structured_temporal,
        policy=resolved_policy,
        sources={
            "real": str(Path(real_path)),
            "predicted": str(Path(predicted_path)),
            "mask": None if mask_path is None else str(Path(mask_path)),
            "structural_mask": (
                None
                if structural_mask_path is None
                else str(Path(structural_mask_path))
            ),
            "roi_labels": (
                None if roi_labels_path is None else str(Path(roi_labels_path))
            ),
        },
        structural_mask=structural_mask,
        configured_roi_label_ids=configured_roi_label_ids,
    )


def _roi_masks_to_labels(
    roi_masks: Optional[np.ndarray],
    spatial_shape: tuple[int, int, int],
    *,
    roi_label_ids: Optional[Sequence[int]] = None,
    require_canonical_roi_mapping: bool = False,
) -> Optional[np.ndarray]:
    if roi_masks is None:
        return None
    values = np.asarray(roi_masks)
    if values.ndim != 4 or tuple(values.shape[1:]) != spatial_shape:
        raise QualityContractError(
            "ROI masks must be [R,D,H,W] on the exact paired native grid"
        )
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise QualityContractError("ROI masks are empty or non-finite")
    if roi_label_ids is None:
        label_ids = tuple(range(1, values.shape[0] + 1))
    else:
        try:
            label_ids = tuple(roi_label_ids)
        except TypeError as exc:
            raise QualityContractError(
                "ROI label IDs must be one ordered integer sequence"
            ) from exc
        if (
            len(label_ids) != values.shape[0]
            or any(
                isinstance(label, bool)
                or not isinstance(label, (int, np.integer))
                or int(label) < 1
                for label in label_ids
            )
            or len(set(map(int, label_ids))) != len(label_ids)
        ):
            raise QualityContractError(
                "ROI label IDs must be unique positive integers matching the channel count"
            )
        label_ids = tuple(map(int, label_ids))
    if require_canonical_roi_mapping and label_ids != tuple(CANONICAL_ROI_LABEL_IDS):
        raise QualityContractError(
            "ROI channel order differs from the canonical 32-ROI mapping "
            f"{CANONICAL_ROI_MAPPING_SHA256}"
        )
    binary = values > 0.5
    if not np.all((values == 0) | (values == 1)):
        raise QualityContractError("ROI masks must be exactly binary")
    counts = binary.reshape(binary.shape[0], -1).sum(axis=1)
    if np.any(counts < 1):
        missing = np.flatnonzero(counts < 1).astype(int).tolist()
        raise QualityContractError(
            "every configured ROI must have non-empty target-validity support; "
            f"missing ROI channels: {missing}"
        )
    overlap = np.sum(binary, axis=0, dtype=np.int16)
    if bool((overlap > 1).any()):
        raise QualityContractError("ROI masks overlap and cannot form a label map")
    labels = np.zeros(spatial_shape, dtype=np.int32)
    for index, label_id in enumerate(label_ids):
        labels[binary[index]] = label_id
    return labels


def evaluate_4d_pair_quality_arrays(
    real: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    *,
    affine: np.ndarray,
    tr_seconds: float,
    roi_masks: Optional[np.ndarray] = None,
    roi_label_ids: Optional[Sequence[int]] = None,
    require_structured_temporal: bool = False,
    require_canonical_roi_mapping: bool = False,
    policy: Optional[QualityPolicy] = None,
    source_identity: Optional[Mapping[str, Any]] = None,
    evaluated_tensor_set_identity: Optional[Mapping[str, Any]] = None,
    structural_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate already aligned native-grid arrays without any resampling.

    Arrays are strictly ``[D,H,W,T]`` with a ``[D,H,W]`` mask and optional
    non-overlapping ``[R,D,H,W]`` ROI masks.  Unlike the NIfTI entry point this
    function never changes orientation: callers must supply the exact native
    crop and its affine.  It is used by the authenticated development gate so
    the evaluated voxels are precisely the voxels eligible for publication.
    """

    resolved_policy = policy or QualityPolicy()
    real_values = np.asarray(real, dtype=np.float32)
    predicted_values = np.asarray(predicted, dtype=np.float32)
    mask_values = np.asarray(mask)
    if real_values.ndim != 4 or real_values.shape[-1] < 2:
        raise QualityContractError("real fMRI array must be [D,H,W,T] with T >= 2")
    if predicted_values.shape != real_values.shape:
        raise QualityContractError("real and predicted array shapes differ")
    if not np.isfinite(real_values).all() or not np.isfinite(predicted_values).all():
        raise QualityContractError("real or predicted fMRI contains NaN or infinity")
    if mask_values.shape != real_values.shape[:3] or not np.isfinite(mask_values).all():
        raise QualityContractError("brain mask differs from the paired native grid")
    binary_mask = mask_values > 0.5
    if not np.logical_or(mask_values == 0, mask_values == 1).all():
        raise QualityContractError("target-validity mask must be exactly binary")
    if not bool(binary_mask.any()):
        raise QualityContractError("brain mask is empty")
    if structural_mask is None:
        binary_structural_mask = binary_mask
    else:
        structural_values = np.asarray(structural_mask)
        if (
            structural_values.shape != real_values.shape[:3]
            or not np.isfinite(structural_values).all()
            or not np.logical_or(
                structural_values == 0, structural_values == 1
            ).all()
        ):
            raise QualityContractError(
                "structural brain mask must be exactly binary on the paired grid"
            )
        binary_structural_mask = structural_values.astype(bool, copy=False)
        if not bool(binary_structural_mask.any()):
            raise QualityContractError("structural brain mask is empty")
        if bool((binary_mask & ~binary_structural_mask).any()):
            raise QualityContractError(
                "target-validity mask lies outside structural brain support"
            )
    affine_values = np.asarray(affine, dtype=np.float64)
    if (
        affine_values.shape != (4, 4)
        or not np.isfinite(affine_values).all()
        or abs(float(np.linalg.det(affine_values[:3, :3]))) <= 1e-12
    ):
        raise QualityContractError("native affine must be finite and nonsingular")
    tr_value = float(tr_seconds)
    if not np.isfinite(tr_value) or tr_value <= 0:
        raise QualityContractError("TR must be positive and finite")
    labels = _roi_masks_to_labels(
        roi_masks,
        tuple(int(value) for value in real_values.shape[:3]),
        roi_label_ids=roi_label_ids,
        require_canonical_roi_mapping=require_canonical_roi_mapping,
    )
    configured_roi_label_ids = (
        tuple(CANONICAL_ROI_LABEL_IDS)
        if require_canonical_roi_mapping
        else (None if roi_label_ids is None else tuple(map(int, roi_label_ids)))
    )
    geometry = {
        "shape": list(real_values.shape),
        "tr_seconds": tr_value,
        "affine": affine_values.tolist(),
        "affine_max_abs_difference": 0.0,
        "canonical_axis_codes": list(nib.aff2axcodes(affine_values)),
        "original_axis_codes": {
            "real": list(nib.aff2axcodes(affine_values)),
            "predicted": list(nib.aff2axcodes(affine_values)),
        },
        "voxel_sizes_mm": [
            float(value) for value in nib.affines.voxel_sizes(affine_values)
        ],
        "mask_voxels": int(binary_mask.sum()),
        "mask_derived_from_real": False,
        "array_entry_point": True,
        "resampling_performed": False,
    }
    return _evaluate_loaded_pair(
        real_values,
        predicted_values,
        binary_mask,
        affine_values,
        tr_value,
        geometry,
        labels,
        roi_labels_supplied=roi_masks is not None,
        require_structured_temporal=require_structured_temporal,
        policy=resolved_policy,
        sources={
            "array_identity": dict(source_identity or {}),
            **(
                {}
                if evaluated_tensor_set_identity is None
                else {
                    "evaluated_tensor_set_identity": dict(evaluated_tensor_set_identity)
                }
            ),
        },
        structural_mask=binary_structural_mask,
        configured_roi_label_ids=configured_roi_label_ids,
    )


__all__ = [
    "DEVELOPMENT_REQUIRED_CHECKS",
    "POLICY_NOTE",
    "QualityContractError",
    "QualityPolicy",
    "evaluate_4d_pair_quality",
    "evaluate_4d_pair_quality_arrays",
    "require_release_quality_policy",
]
