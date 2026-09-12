#!/usr/bin/env python3
"""Post-inference spatial-texture and temporal-detail audit for paired 4D fMRI.

This evaluator is intentionally separate from inference.  It verifies a
target-blind prediction manifest before opening the real target (when one is
provided), then uses the real target only for frozen, post-hoc visual QA.  It
never registers, resamples, filters, or overwrites either input NIfTI.

The primary PNG puts the evidence that is easy to hide with independent image
windows into one figure:

* one shared raw-signal window for real and predicted slices;
* one shared voxel-scale high-pass window for real and predicted detail;
* signed raw and high-pass residuals on their own symmetric windows;
* gradient/Laplacian magnitude quantiles and local radial power spectra;
* sampled-frame detail energy; and
* a real-selected ROI/voxel temporal trace, first differences, and residual.

The companion JSON binds inputs, parameters, code, metrics, and the PNG by
SHA-256.  Thresholds are descriptive by default.  Optional release enforcement,
including amplitude-gaming guards, is implementation policy and is never
presented as a paper-reported metric.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nibabel as nib
import numpy as np
from scipy import ndimage

from scripts.visualize_4d_comparison import (
    ComparisonVolume,
    _comparison_rendering,
    _plane,
    _plane_extent_mm,
    _residual_colormap,
    compute_scale_limits,
    load_comparison,
    select_display_slices,
)
from utils.source_provenance import sha256_file
from utils.spatial_detail import mask_normalized_high_pass


SCHEMA = "connect4-post-inference-texture-audit-v3"
DEFAULT_SAMPLE_FRAME_COUNT = 8
DEFAULT_DETAIL_SIGMA_VOXELS = 0.7
DEFAULT_INTERIOR_EROSION_VOXELS = 2
DEFAULT_PATCH_SIZE_VOXELS = 9
DEFAULT_MAX_PATCHES = 48
DEFAULT_SPECTRUM_BINS = 10
DEFAULT_HIGH_FREQUENCY_CYCLES_PER_VOXEL = 0.35
DEFAULT_NEAR_NYQUIST_CYCLES_PER_VOXEL = 0.60
DEFAULT_TEXTURE_RETENTION_REFERENCE = 0.80
DEFAULT_MINIMUM_DETAIL_CORRELATION = 0.40
DEFAULT_MAXIMUM_NEAR_NYQUIST_TAIL_POWER_RATIO = 1.25
TEXTURE_GATE_FAILURE_EXIT_CODE = 2
ROBUST_PERCENTILE = 99.0

TEXTURE_GATE_METRICS = {
    "voxel_scale_detail_rms_ratio": "voxel-scale detail RMS retention",
    "gradient_rms_ratio": "gradient RMS retention",
    "laplacian_rms_ratio": "Laplacian RMS retention",
    "local_high_frequency_power_ratio": "local high-frequency power retention",
}

ANTI_GAMING_GUARD_METRICS = {
    "aggregate_detail_correlation": "voxel-scale detail correlation",
    "near_nyquist_tail_power_ratio": (
        "predicted/real near-Nyquist tail-power ratio"
    ),
}

REAL_COLOR = "#225ea8"
PREDICTED_COLOR = "#e67e22"
RESIDUAL_COLOR = "#7a5195"
NEUTRAL_COLOR = "#444444"


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    a = a[finite] - a[finite].mean()
    b = b[finite] - b[finite].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else float("nan")


def _sample_frames(num_frames: int, count: int) -> np.ndarray:
    if num_frames < 2:
        raise ValueError("texture audit requires at least two temporal frames")
    if isinstance(count, bool) or int(count) != count or int(count) < 2:
        raise ValueError("sample frame count must be an integer >= 2")
    count = min(int(count), int(num_frames))
    return np.unique(np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int32))


def _interior_mask(mask: np.ndarray, erosion_voxels: int) -> np.ndarray:
    if (
        isinstance(erosion_voxels, bool)
        or int(erosion_voxels) != erosion_voxels
        or int(erosion_voxels) < 0
    ):
        raise ValueError("interior erosion must be a nonnegative integer")
    support = np.asarray(mask, dtype=bool)
    if int(erosion_voxels) == 0:
        return support.copy()
    interior = ndimage.binary_erosion(support, iterations=int(erosion_voxels))
    if not bool(interior.any()):
        raise ValueError("interior erosion removed the complete brain mask")
    return interior


def _prediction_manifest_contract(
    manifest_path: str | Path,
    predicted_path: str | Path,
) -> Dict[str, Any]:
    """Fail closed on the upstream target-blind prediction record.

    This check occurs before ``load_comparison`` opens the real target.  It does
    not claim to reproduce inference; it binds this audit to the already-saved
    prediction and to the upstream record's explicit ordering assertions.
    """
    path = Path(manifest_path).expanduser().resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("prediction manifest must contain a JSON object")
    if record.get("schema") != "connect4-target-blind-finetuned-prediction-v1":
        raise ValueError("unsupported target-blind prediction manifest schema")
    stored_hash = record.get("record_sha256")
    if not isinstance(stored_hash, str):
        raise ValueError("prediction manifest lacks record_sha256")
    hash_payload = dict(record)
    hash_payload.pop("record_sha256", None)
    computed_record_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_record_hash:
        raise ValueError("prediction manifest record_sha256 mismatch")

    actual_prediction_hash = sha256_file(predicted_path)
    recorded_prediction = record.get("prediction")
    if not isinstance(recorded_prediction, dict):
        raise ValueError("prediction manifest lacks prediction metadata")
    if recorded_prediction.get("sha256") != actual_prediction_hash:
        raise ValueError("prediction NIfTI SHA-256 differs from prediction manifest")
    ordering = record.get("target_blind_ordering")
    required_ordering = {
        "real_target_opened_before_prediction_saved": False,
        "real_target_hashed_before_prediction_saved": False,
        "prediction_saved_before_target_metrics": True,
    }
    if not isinstance(ordering, dict) or any(
        ordering.get(key) is not value for key, value in required_ordering.items()
    ):
        raise ValueError("prediction manifest does not certify target-blind ordering")
    return {
        "verified": True,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "record_sha256": stored_hash,
        "schema": record["schema"],
        "scan_id": record.get("scan_id"),
        "evaluation_role": record.get("evaluation_role"),
        "prediction_sha256": actual_prediction_hash,
        "sampling": record.get("sampling"),
        "target_blind_ordering": required_ordering,
        "prediction_verified_before_real_opened": True,
    }


def _figure_target_usage_note(
    target_blind_record: Optional[Dict[str, Any]],
    *,
    same_case_target_used_for_optimization: bool = False,
) -> str:
    """Describe only target-use ordering actually authenticated by this audit."""
    if not isinstance(same_case_target_used_for_optimization, bool):
        raise TypeError("same-case target-use flag must be boolean")
    if same_case_target_used_for_optimization:
        return (
            "Real target was used for same-case optimization and is compared here "
            "post hoc; this is a reconstruction-capacity check, not generalization"
        )
    if target_blind_record is not None:
        return (
            "Authenticated target-blind prediction; real target opened only for this "
            "post-inference audit"
        )
    return (
        "Upstream target use is not certified by this audit; consult the run "
        "provenance before making a generalization claim"
    )


def _farthest_patch_centers(
    mask: np.ndarray,
    patch_size: int,
    max_patches: int,
) -> np.ndarray:
    """Select deterministic, well-spaced centers for complete interior cubes.

    Every voxel in a selected ``patch_size ** 3`` cube must lie inside the
    binary brain support.  In particular, this deliberately supplies a full
    cubic structuring element to SciPy: its default connectivity-one element
    would only prove that a cross/diamond around the center is interior and
    could silently admit background voxels at patch corners.
    """
    if (
        isinstance(patch_size, bool)
        or int(patch_size) != patch_size
        or int(patch_size) < 5
        or int(patch_size) % 2 != 1
    ):
        raise ValueError("patch size must be an odd integer >= 5")
    if (
        isinstance(max_patches, bool)
        or int(max_patches) != max_patches
        or int(max_patches) < 1
    ):
        raise ValueError("maximum patch count must be a positive integer")
    size = int(patch_size)
    support = np.asarray(mask, dtype=bool)
    eligible = ndimage.binary_erosion(
        support,
        structure=np.ones((size, size, size), dtype=bool),
        iterations=1,
        border_value=0,
    )
    coordinates = np.argwhere(eligible)
    if coordinates.size == 0:
        raise ValueError(
            f"brain mask contains no complete {patch_size}x{patch_size}x{patch_size} patch"
        )
    requested = int(max_patches)
    if int(coordinates.shape[0]) < requested:
        raise ValueError(
            f"brain mask contains only {int(coordinates.shape[0])} complete "
            f"{patch_size}x{patch_size}x{patch_size} patch centers; "
            f"{requested} are required"
        )
    values = coordinates.astype(np.float64)
    centre_of_mass = np.asarray(ndimage.center_of_mass(support), dtype=np.float64)
    first = int(np.sum((values - centre_of_mass[None, :]) ** 2, axis=1).argmin())
    selected = [first]
    minimum_distance = np.sum((values - values[first][None, :]) ** 2, axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < requested:
        next_index = int(minimum_distance.argmax())
        selected.append(next_index)
        distance = np.sum((values - values[next_index][None, :]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -1.0
    return coordinates[np.asarray(selected, dtype=np.int64)]


def _local_patch_spectrum(
    volume: np.ndarray,
    frames: np.ndarray,
    centers: np.ndarray,
    *,
    patch_size: int,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean Hann-windowed local 3D power, avoiding brain/exterior edges."""
    if isinstance(bins, bool) or int(bins) != bins or int(bins) < 4:
        raise ValueError("spectrum bin count must be an integer >= 4")
    size = int(patch_size)
    radius = size // 2
    window_1d = np.hanning(size)
    window = (
        window_1d[:, None, None]
        * window_1d[None, :, None]
        * window_1d[None, None, :]
    )
    window_normalizer = float(np.sum(window * window) * window.size)
    frequency = np.fft.fftfreq(size)
    grids = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    radial_frequency = np.sqrt(sum(grid * grid for grid in grids))
    positive = radial_frequency[radial_frequency > 0]
    edges = np.linspace(
        float(positive.min()) * (1.0 - 1e-9),
        float(positive.max()) + 1e-9,
        int(bins) + 1,
    )
    shell_masks = [
        (radial_frequency >= edges[index])
        & (radial_frequency < edges[index + 1])
        for index in range(int(bins))
    ]
    shell_counts = np.asarray([int(shell.sum()) for shell in shell_masks], dtype=np.int64)
    if np.any(shell_counts == 0):
        raise RuntimeError("local spectrum construction produced an empty radial shell")
    shell_frequency = np.asarray(
        [float(radial_frequency[shell].mean()) for shell in shell_masks],
        dtype=np.float64,
    )
    accumulated = np.zeros(int(bins), dtype=np.float64)
    observations = 0
    for frame in np.asarray(frames, dtype=np.int64):
        frame_values = np.asarray(volume[..., int(frame)], dtype=np.float64)
        for coordinate in centers:
            slices = tuple(
                slice(int(axis_value) - radius, int(axis_value) + radius + 1)
                for axis_value in coordinate
            )
            patch = frame_values[slices].copy()
            patch -= patch.mean()
            power = np.abs(np.fft.fftn(patch * window)) ** 2 / window_normalizer
            accumulated += np.asarray(
                [float(power[shell].mean()) for shell in shell_masks]
            )
            observations += 1
    if observations < 1:
        raise RuntimeError("local spectrum received no patch/frame observations")
    return shell_frequency, accumulated / observations, shell_counts


def _derivative_diagnostics(
    comparison: ComparisonVolume,
    frames: np.ndarray,
    interior: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    real_gradient = []
    predicted_gradient = []
    real_laplacian = []
    predicted_laplacian = []
    for frame in np.asarray(frames, dtype=np.int64):
        per_source = []
        for volume in (comparison.real, comparison.predicted):
            values = np.asarray(volume[..., int(frame)], dtype=np.float64)
            derivatives = np.gradient(values)
            gradient_magnitude = np.sqrt(sum(item * item for item in derivatives))
            laplacian_magnitude = np.abs(ndimage.laplace(values, mode="nearest"))
            per_source.append((gradient_magnitude[interior], laplacian_magnitude[interior]))
        real_gradient.append(per_source[0][0])
        real_laplacian.append(per_source[0][1])
        predicted_gradient.append(per_source[1][0])
        predicted_laplacian.append(per_source[1][1])
    arrays = {
        "real_gradient": np.concatenate(real_gradient),
        "predicted_gradient": np.concatenate(predicted_gradient),
        "real_laplacian": np.concatenate(real_laplacian),
        "predicted_laplacian": np.concatenate(predicted_laplacian),
    }
    probabilities = np.asarray(
        [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
        dtype=np.float64,
    )
    result: Dict[str, np.ndarray | float] = {
        "quantile_percent": probabilities,
    }
    for name, values in arrays.items():
        result[f"{name}_quantiles"] = np.percentile(values, probabilities)
        result[f"{name}_rms"] = _rms(values)
    result["gradient_rms_ratio"] = _safe_ratio(
        float(result["predicted_gradient_rms"]),
        float(result["real_gradient_rms"]),
    )
    result["laplacian_rms_ratio"] = _safe_ratio(
        float(result["predicted_laplacian_rms"]),
        float(result["real_laplacian_rms"]),
    )
    return result


def _temporal_trace(comparison: ComparisonVolume, interior: np.ndarray) -> Dict[str, Any]:
    if comparison.roi_labels is not None:
        labels = np.unique(comparison.roi_labels[(comparison.roi_labels > 0) & interior])
        if labels.size < 1:
            raise ValueError("ROI label map contains no positive interior labels")
        real_series = []
        predicted_series = []
        for label in labels:
            roi = interior & (comparison.roi_labels == label)
            real_series.append(
                np.asarray(comparison.real[roi], dtype=np.float64).mean(axis=0)
            )
            predicted_series.append(
                np.asarray(comparison.predicted[roi], dtype=np.float64).mean(axis=0)
            )
        real_stack = np.stack(real_series)
        predicted_stack = np.stack(predicted_series)
        selected_index = int(np.var(real_stack, axis=1).argmax())
        real = real_stack[selected_index]
        predicted = predicted_stack[selected_index]
        selection = {
            "kind": "real-selected exact-label ROI",
            "roi_label": int(labels[selected_index]),
            "selection_rule": "largest real ROI-mean temporal variance",
        }
    else:
        coordinates = np.argwhere(interior)
        real_voxels = np.asarray(comparison.real[interior], dtype=np.float64)
        selected_index = int(np.var(real_voxels, axis=1).argmax())
        coordinate = coordinates[selected_index]
        real = real_voxels[selected_index]
        predicted = np.asarray(
            comparison.predicted[tuple(int(value) for value in coordinate)],
            dtype=np.float64,
        )
        selection = {
            "kind": "real-selected interior voxel",
            "voxel_xyz": [int(value) for value in coordinate],
            "selection_rule": "largest real voxelwise temporal variance",
        }
    real_centered = real - real.mean()
    predicted_centered = predicted - predicted.mean()
    real_difference = np.diff(real_centered)
    predicted_difference = np.diff(predicted_centered)
    residual = predicted_centered - real_centered
    return {
        "selection": selection,
        "time_seconds": np.arange(comparison.num_frames, dtype=np.float64)
        * comparison.tr_seconds,
        "real_centered": real_centered,
        "predicted_centered": predicted_centered,
        "residual": residual,
        "real_first_difference": real_difference,
        "predicted_first_difference": predicted_difference,
        "variance_ratio": _safe_ratio(
            float(np.var(predicted_centered)), float(np.var(real_centered))
        ),
        "trace_correlation": _finite_pearson(real_centered, predicted_centered),
        "first_difference_rms_ratio": _safe_ratio(
            _rms(predicted_difference), _rms(real_difference)
        ),
        "first_difference_correlation": _finite_pearson(
            real_difference, predicted_difference
        ),
        "residual_rms": _rms(residual),
    }


def compute_texture_diagnostics(
    comparison: ComparisonVolume,
    *,
    sample_frame_count: int = DEFAULT_SAMPLE_FRAME_COUNT,
    detail_sigma_voxels: float = DEFAULT_DETAIL_SIGMA_VOXELS,
    interior_erosion_voxels: int = DEFAULT_INTERIOR_EROSION_VOXELS,
    patch_size_voxels: int = DEFAULT_PATCH_SIZE_VOXELS,
    max_patches: int = DEFAULT_MAX_PATCHES,
    spectrum_bins: int = DEFAULT_SPECTRUM_BINS,
    high_frequency_cycles_per_voxel: float = DEFAULT_HIGH_FREQUENCY_CYCLES_PER_VOXEL,
    near_nyquist_cycles_per_voxel: float = DEFAULT_NEAR_NYQUIST_CYCLES_PER_VOXEL,
    texture_retention_reference: float = DEFAULT_TEXTURE_RETENTION_REFERENCE,
) -> Dict[str, Any]:
    """Compute display evidence without modifying either paired input array."""
    sigma = float(detail_sigma_voxels)
    high_frequency = float(high_frequency_cycles_per_voxel)
    near_nyquist = float(near_nyquist_cycles_per_voxel)
    reference = float(texture_retention_reference)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("detail sigma must be positive and finite")
    if not np.isfinite(high_frequency) or high_frequency <= 0:
        raise ValueError("high-frequency boundary must be positive and finite")
    if not np.isfinite(near_nyquist) or near_nyquist <= 0:
        raise ValueError("near-Nyquist boundary must be positive and finite")
    if near_nyquist < high_frequency:
        raise ValueError(
            "near-Nyquist boundary must be greater than or equal to the "
            "high-frequency boundary"
        )
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("texture retention reference must be positive and finite")
    frames = _sample_frames(comparison.num_frames, sample_frame_count)
    interior = _interior_mask(comparison.mask, interior_erosion_voxels)

    real_detail_values = []
    predicted_detail_values = []
    detail_rms_real = []
    detail_rms_predicted = []
    detail_correlation = []
    for frame in frames:
        real_detail, real_support = mask_normalized_high_pass(
            comparison.real[..., int(frame)],
            comparison.mask,
            sigma,
            erosion_iterations=int(interior_erosion_voxels),
        )
        predicted_detail, predicted_support = mask_normalized_high_pass(
            comparison.predicted[..., int(frame)],
            comparison.mask,
            sigma,
            erosion_iterations=int(interior_erosion_voxels),
        )
        if not np.array_equal(real_support, predicted_support) or not np.array_equal(
            real_support, interior
        ):
            raise RuntimeError("paired high-pass interior supports diverged")
        real_values = real_detail[interior]
        predicted_values = predicted_detail[interior]
        real_detail_values.append(real_values)
        predicted_detail_values.append(predicted_values)
        detail_rms_real.append(_rms(real_values))
        detail_rms_predicted.append(_rms(predicted_values))
        detail_correlation.append(_finite_pearson(real_values, predicted_values))
    all_real_detail = np.concatenate(real_detail_values)
    all_predicted_detail = np.concatenate(predicted_detail_values)
    detail_rms_real_array = np.asarray(detail_rms_real, dtype=np.float64)
    detail_rms_predicted_array = np.asarray(detail_rms_predicted, dtype=np.float64)
    selected_frame_index = int(detail_rms_real_array.argmax())
    selected_frame = int(frames[selected_frame_index])
    selected_real_detail, _ = mask_normalized_high_pass(
        comparison.real[..., selected_frame],
        comparison.mask,
        sigma,
        erosion_iterations=int(interior_erosion_voxels),
    )
    selected_predicted_detail, _ = mask_normalized_high_pass(
        comparison.predicted[..., selected_frame],
        comparison.mask,
        sigma,
        erosion_iterations=int(interior_erosion_voxels),
    )
    shared_detail_limit = float(
        np.percentile(
            np.abs(np.concatenate((all_real_detail, all_predicted_detail))),
            ROBUST_PERCENTILE,
        )
    )
    detail_residual_limit = float(
        np.percentile(
            np.abs(all_predicted_detail - all_real_detail), ROBUST_PERCENTILE
        )
    )
    shared_detail_limit = max(shared_detail_limit, 1e-8)
    detail_residual_limit = max(detail_residual_limit, 1e-8)

    derivative = _derivative_diagnostics(comparison, frames, interior)
    centers = _farthest_patch_centers(
        comparison.mask, int(patch_size_voxels), int(max_patches)
    )
    frequency, real_power, shell_counts = _local_patch_spectrum(
        comparison.real,
        frames,
        centers,
        patch_size=int(patch_size_voxels),
        bins=int(spectrum_bins),
    )
    predicted_frequency, predicted_power, predicted_counts = _local_patch_spectrum(
        comparison.predicted,
        frames,
        centers,
        patch_size=int(patch_size_voxels),
        bins=int(spectrum_bins),
    )
    if not np.array_equal(shell_counts, predicted_counts) or not np.allclose(
        frequency, predicted_frequency, rtol=0.0, atol=0.0
    ):
        raise RuntimeError("paired local spectrum grids diverged")
    high_band = frequency >= high_frequency
    if not bool(high_band.any()):
        raise ValueError(
            "high-frequency boundary lies above all local radial spectrum bins"
        )
    near_nyquist_band = frequency >= near_nyquist
    real_high_power = float(np.sum(real_power[high_band] * shell_counts[high_band]))
    predicted_high_power = float(
        np.sum(predicted_power[high_band] * shell_counts[high_band])
    )
    high_frequency_power_ratio = _safe_ratio(predicted_high_power, real_high_power)
    if bool(near_nyquist_band.any()):
        real_near_nyquist_power = float(
            np.sum(real_power[near_nyquist_band] * shell_counts[near_nyquist_band])
        )
        predicted_near_nyquist_power = float(
            np.sum(
                predicted_power[near_nyquist_band]
                * shell_counts[near_nyquist_band]
            )
        )
        near_nyquist_tail_power_ratio = _safe_ratio(
            predicted_near_nyquist_power, real_near_nyquist_power
        )
    else:
        # Descriptive audits remain backward-compatible when a custom patch
        # spectrum has no shell at this boundary. Explicit anti-gaming
        # enforcement consumes the non-finite ratio and fails closed.
        real_near_nyquist_power = float("nan")
        predicted_near_nyquist_power = float("nan")
        near_nyquist_tail_power_ratio = float("nan")
    aggregate_detail_rms_real = _rms(all_real_detail)
    aggregate_detail_rms_predicted = _rms(all_predicted_detail)
    aggregate_detail_ratio = _safe_ratio(
        aggregate_detail_rms_predicted, aggregate_detail_rms_real
    )
    diagnostic_ratios = {
        "voxel_scale_detail_rms_ratio": aggregate_detail_ratio,
        "gradient_rms_ratio": float(derivative["gradient_rms_ratio"]),
        "laplacian_rms_ratio": float(derivative["laplacian_rms_ratio"]),
        "local_high_frequency_power_ratio": high_frequency_power_ratio,
    }
    attenuated_metrics = [
        name
        for name, value in diagnostic_ratios.items()
        if np.isfinite(value) and value < reference
    ]
    return {
        "sampled_frames": frames,
        "selected_frame": selected_frame,
        "selected_frame_rule": (
            "largest real voxel-scale high-pass RMS among evenly spaced sampled frames"
        ),
        "interior": interior,
        "selected_real_detail": selected_real_detail,
        "selected_predicted_detail": selected_predicted_detail,
        "detail_sigma_voxels": sigma,
        "interior_erosion_voxels": int(interior_erosion_voxels),
        "detail_rms_real_by_frame": detail_rms_real_array,
        "detail_rms_predicted_by_frame": detail_rms_predicted_array,
        "detail_correlation_by_frame": np.asarray(detail_correlation, dtype=np.float64),
        "aggregate_detail_rms_real": aggregate_detail_rms_real,
        "aggregate_detail_rms_predicted": aggregate_detail_rms_predicted,
        "aggregate_detail_rms_ratio": aggregate_detail_ratio,
        "aggregate_detail_correlation": _finite_pearson(
            all_real_detail, all_predicted_detail
        ),
        "shared_detail_absolute_limit": shared_detail_limit,
        "detail_residual_absolute_limit": detail_residual_limit,
        "derivative": derivative,
        "patch_centers_xyz": centers,
        "patch_count": int(centers.shape[0]),
        "patch_count_requested": int(max_patches),
        "patch_size_voxels": int(patch_size_voxels),
        "patch_support_contract": (
            "every voxel in each full odd-sized cubic patch lies inside the "
            "binary brain support; FOV exterior is ineligible"
        ),
        "patch_eligibility_structuring_element": "full cubic ones",
        "local_spectrum_frequency_cycles_per_voxel": frequency,
        "local_spectrum_shell_counts": shell_counts,
        "local_spectrum_real_power": real_power,
        "local_spectrum_predicted_power": predicted_power,
        "high_frequency_boundary_cycles_per_voxel": high_frequency,
        "local_high_frequency_power_real": real_high_power,
        "local_high_frequency_power_predicted": predicted_high_power,
        "local_high_frequency_power_ratio": high_frequency_power_ratio,
        "near_nyquist_boundary_cycles_per_voxel": near_nyquist,
        "local_near_nyquist_tail_power_real": real_near_nyquist_power,
        "local_near_nyquist_tail_power_predicted": predicted_near_nyquist_power,
        "near_nyquist_tail_power_ratio": near_nyquist_tail_power_ratio,
        "temporal_trace": _temporal_trace(comparison, interior),
        "texture_retention_reference": reference,
        "diagnostic_ratios": diagnostic_ratios,
        "attenuated_metrics_below_reference": attenuated_metrics,
        "spatial_texture_attenuation_flag": len(attenuated_metrics) >= 2,
        "flag_policy": (
            "descriptive visualization flag only: at least two of four independent "
            "texture-retention ratios fall below the stated reference; not a paper "
            "metric or release threshold"
        ),
    }


def _format_metric(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "undefined"


def build_texture_quality_gate(
    diagnostics: Dict[str, Any],
    *,
    enforce: bool,
    enforce_anti_gaming: bool = False,
    minimum_detail_correlation: float = DEFAULT_MINIMUM_DETAIL_CORRELATION,
    maximum_near_nyquist_tail_power_ratio: float = (
        DEFAULT_MAXIMUM_NEAR_NYQUIST_TAIL_POWER_RATIO
    ),
) -> Dict[str, Any]:
    """Build explicit amplitude-retention and anti-gaming policy records.

    Descriptive mode records the exact result but never changes the process
    exit status. Amplitude enforcement preserves the historical behavior:
    every finite predicted/real retention ratio must meet the reference.
    Anti-gaming enforcement is opt-in and additionally requires correlated
    detail and rejects excess near-Nyquist tail power. Non-finite values fail
    every enforced policy closed.
    """
    ratios = diagnostics.get("diagnostic_ratios")
    if not isinstance(ratios, dict):
        raise ValueError("texture diagnostics lack diagnostic_ratios")
    reference = float(diagnostics.get("texture_retention_reference", float("nan")))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("texture diagnostics contain an invalid retention reference")
    detail_correlation_minimum = float(minimum_detail_correlation)
    tail_power_ratio_maximum = float(maximum_near_nyquist_tail_power_ratio)
    if (
        not np.isfinite(detail_correlation_minimum)
        or detail_correlation_minimum < -1.0
        or detail_correlation_minimum > 1.0
    ):
        raise ValueError("minimum detail correlation must be finite and in [-1, 1]")
    if not np.isfinite(tail_power_ratio_maximum) or tail_power_ratio_maximum <= 0.0:
        raise ValueError(
            "maximum near-Nyquist tail-power ratio must be positive and finite"
        )
    results: Dict[str, Dict[str, Any]] = {}
    for name, display_name in TEXTURE_GATE_METRICS.items():
        if name not in ratios:
            raise ValueError(f"texture diagnostics lack required gate metric {name!r}")
        value = float(ratios[name])
        passed = bool(np.isfinite(value) and value >= reference)
        results[name] = {
            "display_name": display_name,
            "value": value if np.isfinite(value) else None,
            "minimum_inclusive": reference,
            "passed": passed,
        }
    all_passed = bool(all(result["passed"] for result in results.values()))
    detail_correlation = float(
        diagnostics.get("aggregate_detail_correlation", float("nan"))
    )
    near_nyquist_tail_power_ratio = float(
        diagnostics.get("near_nyquist_tail_power_ratio", float("nan"))
    )
    near_nyquist_boundary = float(
        diagnostics.get(
            "near_nyquist_boundary_cycles_per_voxel", float("nan")
        )
    )
    detail_correlation_passed = bool(
        np.isfinite(detail_correlation)
        and detail_correlation >= detail_correlation_minimum
    )
    near_nyquist_tail_passed = bool(
        np.isfinite(near_nyquist_tail_power_ratio)
        and near_nyquist_tail_power_ratio <= tail_power_ratio_maximum
    )
    anti_gaming_results = {
        "aggregate_detail_correlation": {
            "display_name": ANTI_GAMING_GUARD_METRICS[
                "aggregate_detail_correlation"
            ],
            "value": detail_correlation if np.isfinite(detail_correlation) else None,
            "operator": ">=",
            "minimum_inclusive": detail_correlation_minimum,
            "passed": detail_correlation_passed,
        },
        "near_nyquist_tail_power_ratio": {
            "display_name": ANTI_GAMING_GUARD_METRICS[
                "near_nyquist_tail_power_ratio"
            ],
            "value": (
                near_nyquist_tail_power_ratio
                if np.isfinite(near_nyquist_tail_power_ratio)
                else None
            ),
            "operator": "<=",
            "maximum_inclusive": tail_power_ratio_maximum,
            "passed": near_nyquist_tail_passed,
        },
    }
    anti_gaming_passed = bool(
        all(result["passed"] for result in anti_gaming_results.values())
    )
    amplitude_enforced = bool(enforce)
    anti_gaming_enforced = bool(enforce_anti_gaming)
    any_enforced = amplitude_enforced or anti_gaming_enforced
    active_policies_passed = bool(
        (not amplitude_enforced or all_passed)
        and (not anti_gaming_enforced or anti_gaming_passed)
    )
    if any_enforced:
        verdict = "pass" if active_policies_passed else "fail"
        release_gate_passed: Optional[bool] = active_policies_passed
        exit_code = 0 if active_policies_passed else TEXTURE_GATE_FAILURE_EXIT_CODE
        if amplitude_enforced and anti_gaming_enforced:
            mode = "combined-release-gate"
            figure_label = (
                f"ENFORCED TEXTURE + ANTI-GAMING GATE: {verdict.upper()}"
            )
        elif amplitude_enforced:
            mode = "enforced-release-gate"
            figure_label = f"ENFORCED TEXTURE RETENTION GATE: {verdict.upper()}"
        else:
            mode = "anti-gaming-release-gate"
            figure_label = (
                f"ENFORCED TEXTURE ANTI-GAMING GUARDS: {verdict.upper()}"
            )
    else:
        verdict = "not-enforced"
        release_gate_passed = None
        exit_code = 0
        mode = "descriptive-only"
        figure_label = "DESCRIPTIVE TEXTURE AUDIT: NOT ENFORCED"
    return {
        "schema": "connect4-texture-retention-quality-gate-v3",
        "mode": mode,
        "enforced": any_enforced,
        "amplitude_retention_enforced": amplitude_enforced,
        "anti_gaming_guards_enforced": anti_gaming_enforced,
        "policy": {
            "comparison": "predicted_to_real_retention_ratio",
            "operator": ">=",
            "minimum_inclusive": reference,
            "all_metrics_required": True,
            "required_metrics": list(TEXTURE_GATE_METRICS),
            "nonfinite_values_fail": True,
            "paper_reported_metric": False,
            "scope": "implementation release policy for post-inference spatial texture",
        },
        "results": results,
        "anti_gaming_policy": {
            "all_guards_required_when_enforced": True,
            "required_metrics": list(ANTI_GAMING_GUARD_METRICS),
            "aggregate_detail_correlation": {
                "operator": ">=",
                "minimum_inclusive": detail_correlation_minimum,
                "support": "aggregate boundary-safe high-pass interior samples",
            },
            "near_nyquist_tail_power_ratio": {
                "comparison": "predicted_to_real_power_ratio",
                "operator": "<=",
                "maximum_inclusive": tail_power_ratio_maximum,
                "boundary_cycles_per_voxel": (
                    near_nyquist_boundary
                    if np.isfinite(near_nyquist_boundary)
                    else None
                ),
                "support": "deterministic complete-mask local cubic patches",
                "cutoff_application": (
                    "include radial spectrum shells whose mean mode frequency is "
                    "greater than or equal to the boundary; every mode assigned "
                    "to an included shell contributes"
                ),
            },
            "nonfinite_values_fail": True,
            "paper_reported_metric": False,
            "scope": (
                "implementation release policy guarding amplitude-only texture "
                "gate gaming"
            ),
            "calibration_note": (
                "initial bounds selected from the B915 blind-validation "
                "amplitude-gaming negative control; not a paper metric"
            ),
        },
        "anti_gaming_results": anti_gaming_results,
        "all_metrics_meet_reference": all_passed,
        "anti_gaming_guards_passed": anti_gaming_passed,
        "would_pass_if_enforced": all_passed,
        "would_pass_if_anti_gaming_enforced": anti_gaming_passed,
        "would_pass_if_all_enforced": bool(all_passed and anti_gaming_passed),
        "release_gate_passed": release_gate_passed,
        "verdict": verdict,
        "figure_label": figure_label,
        "exit_code": exit_code,
        "failure_exit_code": TEXTURE_GATE_FAILURE_EXIT_CODE,
    }


def plot_texture_audit(
    comparison: ComparisonVolume,
    diagnostics: Dict[str, Any],
    out_path: str | Path,
    *,
    title_prefix: str = "CONNECT-4 post-inference texture audit",
    target_usage_note: str,
    _directory_fd: int | None = None,
) -> Path:
    """Render one evidence-dense, matched-scale static diagnostic."""
    out_path = Path(out_path)
    signal_limit, raw_residual_limit = compute_scale_limits(comparison)
    signal_cmap, signal_norm = _comparison_rendering(comparison, signal_limit)
    residual_cmap = _residual_colormap()
    raw_residual_norm = TwoSlopeNorm(
        vmin=-raw_residual_limit, vcenter=0.0, vmax=raw_residual_limit
    )
    detail_limit = float(diagnostics["shared_detail_absolute_limit"])
    detail_residual_limit = float(diagnostics["detail_residual_absolute_limit"])
    detail_norm = TwoSlopeNorm(vmin=-detail_limit, vcenter=0.0, vmax=detail_limit)
    detail_residual_norm = TwoSlopeNorm(
        vmin=-detail_residual_limit, vcenter=0.0, vmax=detail_residual_limit
    )
    frame = int(diagnostics["selected_frame"])
    slices = select_display_slices(comparison.mask)
    axial = "axial"
    extent = _plane_extent_mm(comparison.real.shape, comparison.affine, axial)

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(17.2, 19.5),
        facecolor="white",
        constrained_layout=True,
    )
    raw_volumes = (
        comparison.real[..., frame],
        comparison.predicted[..., frame],
        comparison.predicted[..., frame] - comparison.real[..., frame],
    )
    raw_titles = (
        "Real signal · shared window",
        "Predicted signal · shared window",
        "Predicted − real · signed residual",
    )
    raw_artists = []
    for column, (volume, title) in enumerate(zip(raw_volumes, raw_titles)):
        artist = axes[0, column].imshow(
            _plane(volume, comparison.mask, axial, slices),
            cmap=signal_cmap if column < 2 else residual_cmap,
            norm=signal_norm if column < 2 else raw_residual_norm,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        axes[0, column].set_title(title)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        axes[0, column].set_facecolor("#111111")
        raw_artists.append(artist)
    fig.colorbar(
        raw_artists[0],
        ax=axes[0, :2],
        location="right",
        shrink=0.72,
        label="BOLD signal (input units; shared real/predicted scale)",
    )
    fig.colorbar(
        raw_artists[2],
        ax=axes[0, 2],
        location="right",
        shrink=0.72,
        label="Signed raw residual",
    )

    real_detail = np.asarray(diagnostics["selected_real_detail"], dtype=np.float64)
    predicted_detail = np.asarray(
        diagnostics["selected_predicted_detail"], dtype=np.float64
    )
    interior = np.asarray(diagnostics["interior"], dtype=bool)
    detail_volumes = (
        real_detail,
        predicted_detail,
        predicted_detail - real_detail,
    )
    detail_titles = (
        "Real voxel-scale detail · shared window",
        "Predicted voxel-scale detail · shared window",
        "Predicted − real detail · signed residual",
    )
    detail_artists = []
    for column, (volume, title) in enumerate(zip(detail_volumes, detail_titles)):
        artist = axes[1, column].imshow(
            _plane(volume, interior, axial, slices),
            cmap=residual_cmap,
            norm=detail_norm if column < 2 else detail_residual_norm,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        axes[1, column].set_title(title)
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
        axes[1, column].set_facecolor("#111111")
        detail_artists.append(artist)
    fig.colorbar(
        detail_artists[0],
        ax=axes[1, :2],
        location="right",
        shrink=0.72,
        label="High-pass detail (input units; shared real/predicted scale)",
    )
    fig.colorbar(
        detail_artists[2],
        ax=axes[1, 2],
        location="right",
        shrink=0.72,
        label="Signed high-pass residual",
    )

    derivative = diagnostics["derivative"]
    quantiles = np.asarray(derivative["quantile_percent"], dtype=np.float64)
    axes[2, 0].plot(
        quantiles,
        derivative["real_gradient_quantiles"],
        color=REAL_COLOR,
        lw=2.0,
        label="Real gradient",
    )
    axes[2, 0].plot(
        quantiles,
        derivative["predicted_gradient_quantiles"],
        color=PREDICTED_COLOR,
        lw=1.8,
        ls="--",
        label="Predicted gradient",
    )
    axes[2, 0].plot(
        quantiles,
        derivative["real_laplacian_quantiles"],
        color=REAL_COLOR,
        lw=1.5,
        ls=":",
        label="Real |Laplacian|",
    )
    axes[2, 0].plot(
        quantiles,
        derivative["predicted_laplacian_quantiles"],
        color=PREDICTED_COLOR,
        lw=1.5,
        ls="-.",
        label="Predicted |Laplacian|",
    )
    axes[2, 0].set_title(
        "Interior spatial-derivative magnitude quantiles\n"
        f"gradient RMS ratio={_format_metric(float(derivative['gradient_rms_ratio']))} · "
        f"Laplacian RMS ratio={_format_metric(float(derivative['laplacian_rms_ratio']))}"
    )
    axes[2, 0].set_xlabel("Masked sample percentile")
    axes[2, 0].set_ylabel("Magnitude (input units/voxel)")
    axes[2, 0].legend(frameon=False, fontsize=8, ncol=2)

    frequency = np.asarray(
        diagnostics["local_spectrum_frequency_cycles_per_voxel"], dtype=np.float64
    )
    real_power = np.asarray(diagnostics["local_spectrum_real_power"], dtype=np.float64)
    predicted_power = np.asarray(
        diagnostics["local_spectrum_predicted_power"], dtype=np.float64
    )
    boundary = float(diagnostics["high_frequency_boundary_cycles_per_voxel"])
    axes[2, 1].plot(
        frequency, real_power, color=REAL_COLOR, lw=2.0, marker="o", label="Real"
    )
    axes[2, 1].plot(
        frequency,
        predicted_power,
        color=PREDICTED_COLOR,
        lw=1.8,
        ls="--",
        marker="s",
        markerfacecolor="white",
        label="Predicted",
    )
    axes[2, 1].axvspan(
        boundary,
        float(frequency.max()) * 1.01,
        color="#dddddd",
        alpha=0.55,
        label="High-frequency band",
    )
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_xlim(float(frequency.min()) * 0.95, float(frequency.max()) * 1.01)
    axes[2, 1].set_title(
        f"Complete-mask {int(diagnostics['patch_size_voxels'])}³ Hann-windowed "
        "radial spatial power\n"
        f"high-frequency predicted/real power="
        f"{_format_metric(float(diagnostics['local_high_frequency_power_ratio']))}"
    )
    axes[2, 1].set_xlabel("Radial frequency (cycles/voxel)")
    axes[2, 1].set_ylabel("Mean local power (input units²)")
    axes[2, 1].legend(frameon=False, fontsize=8)

    frames = np.asarray(diagnostics["sampled_frames"], dtype=np.int32)
    time_sampled = frames.astype(np.float64) * comparison.tr_seconds
    axes[2, 2].plot(
        time_sampled,
        diagnostics["detail_rms_real_by_frame"],
        color=REAL_COLOR,
        lw=2.0,
        marker="o",
        label="Real",
    )
    axes[2, 2].plot(
        time_sampled,
        diagnostics["detail_rms_predicted_by_frame"],
        color=PREDICTED_COLOR,
        lw=1.8,
        ls="--",
        marker="s",
        markerfacecolor="white",
        label="Predicted",
    )
    axes[2, 2].axvline(frame * comparison.tr_seconds, color=NEUTRAL_COLOR, lw=1.0, ls=":")
    axes[2, 2].set_title(
        "Voxel-scale detail RMS across sampled frames\n"
        f"aggregate predicted/real="
        f"{_format_metric(float(diagnostics['aggregate_detail_rms_ratio']))} · "
        f"detail r={_format_metric(float(diagnostics['aggregate_detail_correlation']))}"
    )
    axes[2, 2].set_xlabel("Time (seconds)")
    axes[2, 2].set_ylabel("High-pass RMS (input units)")
    axes[2, 2].legend(frameon=False)

    temporal = diagnostics["temporal_trace"]
    time = np.asarray(temporal["time_seconds"], dtype=np.float64)
    axes[3, 0].plot(
        time, temporal["real_centered"], color=REAL_COLOR, lw=1.8, label="Real"
    )
    axes[3, 0].plot(
        time,
        temporal["predicted_centered"],
        color=PREDICTED_COLOR,
        lw=1.5,
        ls="--",
        label="Predicted",
    )
    selection = temporal["selection"]
    selection_text = (
        f"ROI label {selection['roi_label']}"
        if "roi_label" in selection
        else f"voxel {selection['voxel_xyz']}"
    )
    axes[3, 0].set_title(
        f"Demeaned temporal trace · {selection_text}\n"
        f"real-only selection · r={_format_metric(float(temporal['trace_correlation']))} · "
        f"variance ratio={_format_metric(float(temporal['variance_ratio']))}"
    )
    axes[3, 0].set_xlabel("Time (seconds)")
    axes[3, 0].set_ylabel("Demeaned BOLD signal (input units)")
    axes[3, 0].legend(frameon=False)

    difference_time = time[1:]
    axes[3, 1].plot(
        difference_time,
        temporal["real_first_difference"],
        color=REAL_COLOR,
        lw=1.6,
        label="Real Δ",
    )
    axes[3, 1].plot(
        difference_time,
        temporal["predicted_first_difference"],
        color=PREDICTED_COLOR,
        lw=1.4,
        ls="--",
        label="Predicted Δ",
    )
    axes[3, 1].axhline(0.0, color="#aaaaaa", lw=0.8)
    axes[3, 1].set_title(
        "First temporal differences\n"
        f"RMS ratio={_format_metric(float(temporal['first_difference_rms_ratio']))} · "
        f"difference r={_format_metric(float(temporal['first_difference_correlation']))}"
    )
    axes[3, 1].set_xlabel("Time (seconds)")
    axes[3, 1].set_ylabel("Frame difference (input units)")
    axes[3, 1].legend(frameon=False)

    axes[3, 2].plot(
        time, temporal["residual"], color=RESIDUAL_COLOR, lw=1.5
    )
    axes[3, 2].axhline(0.0, color="#aaaaaa", lw=0.8)
    ratios = diagnostics["diagnostic_ratios"]
    quality_gate = diagnostics.get("quality_gate")
    if not isinstance(quality_gate, dict):
        quality_gate = build_texture_quality_gate(diagnostics, enforce=False)
    metric_results = quality_gate["results"]
    result_labels = {
        name: "PASS" if metric_results[name]["passed"] else "FAIL"
        for name in TEXTURE_GATE_METRICS
    }
    anti_gaming_results = quality_gate.get("anti_gaming_results")
    if not isinstance(anti_gaming_results, dict):
        anti_gaming_results = build_texture_quality_gate(
            diagnostics, enforce=False
        )["anti_gaming_results"]
    anti_gaming_labels = {
        name: "PASS" if anti_gaming_results[name]["passed"] else "FAIL"
        for name in ANTI_GAMING_GUARD_METRICS
    }
    detail_guard = anti_gaming_results["aggregate_detail_correlation"]
    tail_guard = anti_gaming_results["near_nyquist_tail_power_ratio"]
    anti_gaming_mode_label = (
        "ENFORCED"
        if quality_gate.get("anti_gaming_guards_enforced")
        else "DESCRIPTIVE"
    )
    attenuation_text = (
        "multi-metric attenuation detected"
        if diagnostics["spatial_texture_attenuation_flag"]
        else "multi-metric attenuation not detected"
    )
    axes[3, 2].set_title(
        "Demeaned temporal residual (predicted − real)\n"
        f"residual RMS={_format_metric(float(temporal['residual_rms']))}"
    )
    axes[3, 2].set_xlabel("Time (seconds)")
    axes[3, 2].set_ylabel("Residual (input units)")
    axes[3, 2].text(
        0.02,
        0.98,
        str(quality_gate["figure_label"])
        + "\n"
        + f"detail RMS   {ratios['voxel_scale_detail_rms_ratio']:.3f} "
        + f"{result_labels['voxel_scale_detail_rms_ratio']}\n"
        + f"gradient     {ratios['gradient_rms_ratio']:.3f} "
        + f"{result_labels['gradient_rms_ratio']}\n"
        + f"Laplacian    {ratios['laplacian_rms_ratio']:.3f} "
        + f"{result_labels['laplacian_rms_ratio']}\n"
        + f"high-band    {ratios['local_high_frequency_power_ratio']:.3f} "
        + f"{result_labels['local_high_frequency_power_ratio']}\n"
        + f"minimum >=   {diagnostics['texture_retention_reference']:.2f}\n"
        + f"anti-game    {anti_gaming_mode_label}\n"
        + "detail r     "
        + f"{_format_metric(float(detail_guard['value']) if detail_guard['value'] is not None else float('nan'))} "
        + f"{anti_gaming_labels['aggregate_detail_correlation']} >= "
        + f"{detail_guard['minimum_inclusive']:.2f}\n"
        + "Nyquist P-r  "
        + f"{_format_metric(float(tail_guard['value']) if tail_guard['value'] is not None else float('nan'))} "
        + f"{anti_gaming_labels['near_nyquist_tail_power_ratio']} <= "
        + f"{tail_guard['maximum_inclusive']:.2f}\n"
        + attenuation_text,
        transform=axes[3, 2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.90},
    )

    for axis in axes[2:, :].reshape(-1):
        axis.grid(axis="y", color="#dddddd", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{title_prefix}\n"
        f"frame {frame} ({frame * comparison.tr_seconds:g} s) · axial z-index "
        f"{slices['axial']} · paired canonical RAS+ display · TR="
        f"{comparison.tr_seconds:g} s\n"
        "Nearest-neighbor display; no fMRI registration/resampling\n"
        f"{target_usage_note}\n"
        f"{quality_gate['figure_label']} · all four ratios must be >= "
        f"{quality_gate['policy']['minimum_inclusive']:.2f} when enforced\n"
        "Anti-gaming implementation guards: detail r >= "
        f"{detail_guard['minimum_inclusive']:.2f}; near-Nyquist predicted/real "
        f"tail power <= {tail_guard['maximum_inclusive']:.2f} when explicitly enforced",
        fontsize=14,
    )
    buffer = io.BytesIO()
    try:
        fig.savefig(
            buffer,
            format="png",
            dpi=170,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(fig)
    _write_output_new(out_path, buffer.getvalue(), directory_fd=_directory_fd)
    return out_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _open_new_output_directory(path: Path) -> tuple[Path, int, tuple[int, int]]:
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = Path(os.path.abspath(requested))
    parent = requested.parent
    if parent.resolve(strict=True) != parent or parent.is_symlink():
        raise ValueError("texture output parent must be canonical and non-symlinked")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.mkdir(requested.name, 0o700, dir_fd=parent_fd)
        directory_fd = os.open(
            requested.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(directory_fd)
        current = requested.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            os.close(directory_fd)
            raise RuntimeError("texture output directory changed while opening")
        os.fsync(parent_fd)
        return requested, directory_fd, (opened.st_dev, opened.st_ino)
    finally:
        os.close(parent_fd)


def _assert_output_directory(
    path: Path, directory_fd: int, identity: tuple[int, int]
) -> None:
    opened = os.fstat(directory_fd)
    current = path.lstat()
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (opened.st_dev, opened.st_ino) != identity
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RuntimeError("texture output directory was replaced")


def _write_output_new(
    path: Path, payload: bytes, *, directory_fd: int | None = None
) -> None:
    owned_directory_fd = directory_fd is None
    if owned_directory_fd:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent or parent.is_symlink():
            raise ValueError("texture output parent aliases another directory")
        directory_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    assert directory_fd is not None
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError("texture output is not a new single-link regular file")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise RuntimeError("texture output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        final = os.fstat(descriptor)
        if final.st_nlink != 1 or final.st_size != len(payload):
            raise RuntimeError("texture output link or size changed while writing")
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if owned_directory_fd:
            os.close(directory_fd)


def _hash_output_at(directory_fd: int, name: str) -> str:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError("texture output binding is not a single-link file")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        final = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(opened, field) != getattr(final, field)
            or getattr(opened, field) != getattr(current, field)
            for field in stable
        ):
            raise RuntimeError("texture output changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _public_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {
        "interior",
        "selected_real_detail",
        "selected_predicted_detail",
        "quality_gate",
    }
    return {
        key: _json_safe(value)
        for key, value in diagnostics.items()
        if key not in excluded
    }


def generate_texture_audit(
    real_path: str | Path,
    predicted_path: str | Path,
    mask_path: str | Path,
    out_dir: str | Path,
    *,
    roi_labels_path: Optional[str | Path] = None,
    prediction_manifest_path: Optional[str | Path] = None,
    same_case_target_used_for_optimization: bool = False,
    prefix: str = "paired",
    sample_frame_count: int = DEFAULT_SAMPLE_FRAME_COUNT,
    detail_sigma_voxels: float = DEFAULT_DETAIL_SIGMA_VOXELS,
    interior_erosion_voxels: int = DEFAULT_INTERIOR_EROSION_VOXELS,
    patch_size_voxels: int = DEFAULT_PATCH_SIZE_VOXELS,
    max_patches: int = DEFAULT_MAX_PATCHES,
    spectrum_bins: int = DEFAULT_SPECTRUM_BINS,
    high_frequency_cycles_per_voxel: float = DEFAULT_HIGH_FREQUENCY_CYCLES_PER_VOXEL,
    near_nyquist_cycles_per_voxel: float = DEFAULT_NEAR_NYQUIST_CYCLES_PER_VOXEL,
    texture_retention_reference: float = DEFAULT_TEXTURE_RETENTION_REFERENCE,
    enforce_texture_retention: bool = False,
    minimum_detail_correlation: float = DEFAULT_MINIMUM_DETAIL_CORRELATION,
    maximum_near_nyquist_tail_power_ratio: float = (
        DEFAULT_MAXIMUM_NEAR_NYQUIST_TAIL_POWER_RATIO
    ),
    enforce_texture_anti_gaming: bool = False,
) -> Dict[str, Path]:
    """Verify provenance, load a strict pair, and write PNG plus bound JSON."""
    prefix_value = str(prefix).strip()
    if not prefix_value or Path(prefix_value).name != prefix_value:
        raise ValueError("output prefix must be one non-empty filename component")
    target_blind_record = None
    if prediction_manifest_path is not None:
        target_blind_record = _prediction_manifest_contract(
            prediction_manifest_path, predicted_path
        )
    target_usage_note = _figure_target_usage_note(
        target_blind_record,
        same_case_target_used_for_optimization=(
            same_case_target_used_for_optimization
        ),
    )
    comparison = load_comparison(
        real_path,
        predicted_path,
        mask_path=mask_path,
        roi_labels_path=roi_labels_path,
    )
    diagnostics = compute_texture_diagnostics(
        comparison,
        sample_frame_count=sample_frame_count,
        detail_sigma_voxels=detail_sigma_voxels,
        interior_erosion_voxels=interior_erosion_voxels,
        patch_size_voxels=patch_size_voxels,
        max_patches=max_patches,
        spectrum_bins=spectrum_bins,
        high_frequency_cycles_per_voxel=high_frequency_cycles_per_voxel,
        near_nyquist_cycles_per_voxel=near_nyquist_cycles_per_voxel,
        texture_retention_reference=texture_retention_reference,
    )
    quality_gate = build_texture_quality_gate(
        diagnostics,
        enforce=bool(enforce_texture_retention),
        enforce_anti_gaming=bool(enforce_texture_anti_gaming),
        minimum_detail_correlation=minimum_detail_correlation,
        maximum_near_nyquist_tail_power_ratio=(
            maximum_near_nyquist_tail_power_ratio
        ),
    )
    diagnostics["quality_gate"] = quality_gate
    out_dir, output_directory_fd, output_directory_identity = (
        _open_new_output_directory(Path(out_dir))
    )
    try:
        png_path = plot_texture_audit(
            comparison,
            diagnostics,
            out_dir / f"{prefix_value}_texture_audit.png",
            target_usage_note=target_usage_note,
            _directory_fd=output_directory_fd,
        )
        _assert_output_directory(
            out_dir, output_directory_fd, output_directory_identity
        )
    except Exception:
        os.close(output_directory_fd)
        raise
    signal_limit, residual_limit = compute_scale_limits(comparison)
    script_path = Path(__file__).resolve(strict=True)
    visualizer_path = Path(__file__).with_name("visualize_4d_comparison.py").resolve(
        strict=True
    )
    helper_path = Path(__file__).resolve().parents[1] / "utils" / "spatial_detail.py"
    inputs = {
        "real": Path(real_path).expanduser().resolve(strict=True),
        "predicted": Path(predicted_path).expanduser().resolve(strict=True),
        "mask": Path(mask_path).expanduser().resolve(strict=True),
        "roi_labels": (
            None
            if roi_labels_path is None
            else Path(roi_labels_path).expanduser().resolve(strict=True)
        ),
    }
    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "frozen post-inference visual QA; no model fitting or inference",
        "target_blind_provenance": target_blind_record,
        "target_blind_note": (
            "The evaluator opens the real target only after verifying the optional "
            "upstream target-blind prediction record. It cannot certify an upstream "
            "run when no prediction manifest is supplied."
        ),
        "same_case_target_used_for_optimization": bool(
            same_case_target_used_for_optimization
        ),
        "figure_target_usage_note": target_usage_note,
        "quality_gate": _json_safe(quality_gate),
        "inputs": {
            name: (
                None
                if path is None
                else {"path": str(path), "sha256": sha256_file(path)}
            )
            for name, path in inputs.items()
        },
        "geometry": {
            "shape_xyzt": [int(value) for value in comparison.real.shape],
            "affine": np.asarray(comparison.affine, dtype=np.float64).tolist(),
            "voxel_sizes_mm": [
                float(value) for value in nib.affines.voxel_sizes(comparison.affine)
            ],
            "tr_seconds": float(comparison.tr_seconds),
            "space": comparison.space_description,
            "mask_voxels": int(comparison.mask.sum()),
            "mask_description": comparison.mask_description,
            "mask_resampled": bool(comparison.mask_resampled),
        },
        "display_contract": {
            "raw_real_predicted_shared_absolute_limit": float(signal_limit),
            "raw_residual_absolute_limit": float(residual_limit),
            "robust_percentile": ROBUST_PERCENTILE,
            "high_pass_real_predicted_shared_absolute_limit": float(
                diagnostics["shared_detail_absolute_limit"]
            ),
            "high_pass_residual_absolute_limit": float(
                diagnostics["detail_residual_absolute_limit"]
            ),
            "image_interpolation": "nearest",
            "fMRI_registration_or_resampling": False,
            "slice_selection": "mask-only largest axial support",
            "frame_selection": diagnostics["selected_frame_rule"],
            "selection_uses_prediction": False,
            "local_spectrum_patch_selection": diagnostics[
                "patch_support_contract"
            ],
            "local_spectrum_patch_count": int(diagnostics["patch_count"]),
            "near_nyquist_boundary_cycles_per_voxel": float(
                diagnostics["near_nyquist_boundary_cycles_per_voxel"]
            ),
            "local_spectrum_frequency_coordinate": (
                "mean radial cycles/voxel of the Fourier modes assigned to each shell"
            ),
            "near_nyquist_cutoff_application": (
                "shell-mean cutoff: include shells whose mean radial frequency is "
                "greater than or equal to the configured boundary"
            ),
        },
        "diagnostics": _public_diagnostics(diagnostics),
        "implementation_sources": {
            "texture_audit": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "strict_pair_visualizer": {
                "path": str(visualizer_path),
                "sha256": sha256_file(visualizer_path),
            },
            "boundary_safe_high_pass": {
                "path": str(helper_path.resolve(strict=True)),
                "sha256": sha256_file(helper_path),
            },
        },
        "outputs": {
            "texture_audit_png": {
                "path": str(png_path),
                "sha256": _hash_output_at(
                    output_directory_fd, png_path.name
                ),
            }
        },
    }
    manifest["record_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    json_path = out_dir / f"{prefix_value}_texture_audit.json"
    _assert_output_directory(out_dir, output_directory_fd, output_directory_identity)
    _write_output_new(
        json_path,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        directory_fd=output_directory_fd,
    )
    _assert_output_directory(out_dir, output_directory_fd, output_directory_identity)
    os.close(output_directory_fd)
    return {"png": png_path, "manifest": json_path}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a matched-scale, post-inference spatial-texture and temporal-detail "
            "audit for one strict real/predicted 4D fMRI pair."
        )
    )
    parser.add_argument("--real", required=True, help="real 4D fMRI NIfTI")
    parser.add_argument("--pred", required=True, help="predicted 4D fMRI NIfTI")
    parser.add_argument("--mask", required=True, help="paired-grid 3D brain mask")
    parser.add_argument(
        "--roi-labels",
        default=None,
        help="optional paired-grid integer ROI labels for the temporal trace",
    )
    parser.add_argument(
        "--prediction-manifest",
        default=None,
        help="optional target-blind prediction manifest verified before real is opened",
    )
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--prefix", default="paired", help="output filename prefix")
    parser.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAME_COUNT)
    parser.add_argument(
        "--detail-sigma-voxels", type=float, default=DEFAULT_DETAIL_SIGMA_VOXELS
    )
    parser.add_argument(
        "--interior-erosion-voxels",
        type=int,
        default=DEFAULT_INTERIOR_EROSION_VOXELS,
    )
    parser.add_argument(
        "--patch-size-voxels", type=int, default=DEFAULT_PATCH_SIZE_VOXELS
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=DEFAULT_MAX_PATCHES,
        help=(
            "required count of deterministic, complete-mask cubic patches; "
            "the audit fails if the mask cannot supply this many"
        ),
    )
    parser.add_argument("--spectrum-bins", type=int, default=DEFAULT_SPECTRUM_BINS)
    parser.add_argument(
        "--high-frequency-cycles-per-voxel",
        type=float,
        default=DEFAULT_HIGH_FREQUENCY_CYCLES_PER_VOXEL,
    )
    parser.add_argument(
        "--near-nyquist-cycles-per-voxel",
        type=float,
        default=DEFAULT_NEAR_NYQUIST_CYCLES_PER_VOXEL,
        help=(
            "lower radial-frequency boundary for the complete-patch "
            "near-Nyquist tail-power guard"
        ),
    )
    parser.add_argument(
        "--texture-retention-reference",
        type=float,
        default=DEFAULT_TEXTURE_RETENTION_REFERENCE,
        help=(
            "minimum predicted/real retention ratio for all four metrics; "
            "descriptive by default and enforced only with "
            "--enforce-texture-retention; not a paper metric"
        ),
    )
    parser.add_argument(
        "--enforce-texture-retention",
        action="store_true",
        help=(
            "enforce all four texture-retention ratios at the configured minimum; "
            f"write the audit artifacts, then exit {TEXTURE_GATE_FAILURE_EXIT_CODE} "
            "when any metric fails"
        ),
    )
    parser.add_argument(
        "--minimum-detail-correlation",
        type=float,
        default=DEFAULT_MINIMUM_DETAIL_CORRELATION,
        help=(
            "minimum aggregate real/predicted boundary-safe detail correlation "
            "for the optional anti-gaming release policy; not a paper metric"
        ),
    )
    parser.add_argument(
        "--maximum-near-nyquist-tail-power-ratio",
        type=float,
        default=DEFAULT_MAXIMUM_NEAR_NYQUIST_TAIL_POWER_RATIO,
        help=(
            "maximum predicted/real complete-patch near-Nyquist tail-power ratio "
            "for the optional anti-gaming release policy; not a paper metric"
        ),
    )
    parser.add_argument(
        "--enforce-texture-anti-gaming",
        action="store_true",
        help=(
            "explicitly enforce both implementation anti-gaming guards; write "
            "the audit artifacts, then exit "
            f"{TEXTURE_GATE_FAILURE_EXIT_CODE} when either guard fails"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    outputs = generate_texture_audit(
        args.real,
        args.pred,
        args.mask,
        args.out_dir,
        roi_labels_path=args.roi_labels,
        prediction_manifest_path=args.prediction_manifest,
        prefix=args.prefix,
        sample_frame_count=args.sample_frames,
        detail_sigma_voxels=args.detail_sigma_voxels,
        interior_erosion_voxels=args.interior_erosion_voxels,
        patch_size_voxels=args.patch_size_voxels,
        max_patches=args.max_patches,
        spectrum_bins=args.spectrum_bins,
        high_frequency_cycles_per_voxel=args.high_frequency_cycles_per_voxel,
        near_nyquist_cycles_per_voxel=args.near_nyquist_cycles_per_voxel,
        texture_retention_reference=args.texture_retention_reference,
        enforce_texture_retention=args.enforce_texture_retention,
        minimum_detail_correlation=args.minimum_detail_correlation,
        maximum_near_nyquist_tail_power_ratio=(
            args.maximum_near_nyquist_tail_power_ratio
        ),
        enforce_texture_anti_gaming=args.enforce_texture_anti_gaming,
    )
    for name, path in outputs.items():
        print(f"[texture-audit] {name}: {path}")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    quality_gate = manifest["quality_gate"]
    print(
        "[texture-audit] quality gate: "
        f"{quality_gate['verdict']} · mode={quality_gate['mode']} · "
        f"all_metrics_meet_reference={quality_gate['all_metrics_meet_reference']} · "
        "anti_gaming_guards_passed="
        f"{quality_gate['anti_gaming_guards_passed']}"
    )
    return int(quality_gate["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
