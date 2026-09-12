"""
Visualisation of real vs predicted rs-fMRI (magma colormap).

`plot_real_vs_synthetic` renders a compact temporal-mean comparison retained
for backwards compatibility.  `plot_real_vs_synthetic_4d` is the diagnostic
figure used by training and inference: it keeps individual frames visible and
adds spatial-high-pass, temporal-STD, DVARS, and spectrum panels so a collapsed
or over-smoothed output cannot look acceptable merely because of temporal
averaging.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence
import warnings

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.spatial_detail import mask_normalized_high_pass


def _to_numpy_tdhw(x) -> np.ndarray:
    """Accept only singleton-batch/channel forms and return ``[T,D,H,W]``."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 6:
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError("6D fMRI input must have singleton batch and channel axes")
        x = x[0, 0]
    elif x.ndim == 5:
        if x.shape[0] != 1:
            raise ValueError("5D fMRI input must have a singleton batch axis")
        x = x[0]
    if x.ndim == 3:  # [D,H,W] single frame
        x = x[None]
    if x.ndim != 4:
        raise ValueError(f"fMRI input must resolve to [T,D,H,W], got shape {x.shape}")
    return x


def _robust_window(arr: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0):
    """Robust (percentile) intensity window, ignoring non-finite values."""
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(a, lo_p)), float(np.percentile(a, hi_p))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _tri_planes(vol: np.ndarray, idx):
    """
    Extract axial / sagittal / coronal slices from a [D, H, W] volume at the
    given (d, h, w) indices. Returns (axial, sagittal, coronal), each oriented
    upright for display.
    """
    # vol axes follow the MNI grid: axis0 = X (L-R), axis1 = Y (A-P), axis2 = Z (I-S).
    d, h, w = idx  # d: X-index, h: Y-index, w: Z-index
    axial = np.rot90(vol[:, :, w])  # constant Z (axis2) -> [X, Y]
    sagittal = np.rot90(vol[d, :, :])  # constant X (axis0) -> [Y, Z]
    coronal = np.rot90(vol[:, h, :])  # constant Y (axis1) -> [X, Z]
    return axial, sagittal, coronal


def _to_numpy_mask(mask, spatial_shape: Sequence[int]) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    values = np.asarray(mask)
    while values.ndim > 3:
        if values.shape[0] != 1:
            raise ValueError("brain mask may contain only singleton leading axes")
        values = values[0]
    if values.shape != tuple(spatial_shape):
        raise ValueError(
            f"brain mask shape {values.shape} differs from fMRI grid {tuple(spatial_shape)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("brain mask contains NaN or infinity")
    if not np.logical_or(values == 0, values == 1).all():
        raise ValueError("brain mask must be exactly binary")
    support = values.astype(bool, copy=False)
    if not support.any():
        raise ValueError("brain mask is empty")
    return support


def _largest_mask_slice(mask: np.ndarray, axis: int) -> int:
    reduce_axes = tuple(value for value in range(3) if value != axis)
    counts = mask.sum(axis=reduce_axes)
    candidates = np.flatnonzero(counts == counts.max())
    centre = float(np.average(np.arange(counts.size), weights=counts))
    return int(candidates[np.abs(candidates - centre).argmin()])


def _sample_masked_tdhw(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    max_samples: int = 1_000_000,
) -> np.ndarray:
    spatial = np.flatnonzero(mask.reshape(-1))
    temporal = int(array.shape[0])
    total = int(spatial.size * temporal)
    count = min(total, int(max_samples))
    positions = np.linspace(0, total - 1, count, dtype=np.int64)
    selected_spatial = spatial[positions // temporal]
    selected_time = positions % temporal
    matrix = array.reshape(temporal, -1)
    return np.asarray(matrix[selected_time, selected_spatial], dtype=np.float64)


def _masked_axial(volume: np.ndarray, mask: np.ndarray, z_index: int) -> np.ndarray:
    plane = np.rot90(volume[:, :, z_index])
    support = np.rot90(mask[:, :, z_index])
    return np.where(support, plane, np.nan)


def _finite_percentile(values: np.ndarray, percentile: float, fallback: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    result = float(np.percentile(finite, percentile))
    if not np.isfinite(result) or result <= 0.0:
        return fallback
    return result


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": values.dtype.str, "shape": list(values.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if values.ndim == 0:
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        for first_axis_index in range(values.shape[0]):
            digest.update(np.ascontiguousarray(values[first_axis_index]).tobytes())
    return digest.hexdigest()


def _display_image(
    axis,
    image: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str | None = None,
    ylabel: str | None = None,
):
    plotted = axis.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title, fontsize=10)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=10)
    return plotted


def plot_real_vs_synthetic_4d(
    real,
    synthetic,
    out_path: str,
    slice_idx: Optional[Sequence[int]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    brain_mask=None,
    target_validity_mask=None,
    repetition_time: float = 3.0,
    high_pass_sigma_voxels: float = 1.0,
    scale_limits: Optional[Mapping[str, float]] = None,
    title: str = "CONNECT-4: real vs predicted 4D rs-fMRI",
):
    """Render a fail-loud 4D comparison without temporal-mean concealment.

    The first six rows show four identical time frames on one axial slice:
    mean-centred signal and its mask-normalized 3D Gaussian high-pass residual,
    predicted, and absolute-difference rows.  Three additional rows expose the
    temporal-mean high-pass texture that demeaning would otherwise conceal.
    The final three rows show tri-planar temporal-STD maps.  The remaining cells
    contain whole-brain global-signal, DVARS, and calibrated
    mean-power-spectrum comparisons.

    Every corresponding real/predicted panel shares an intensity window.
    Frame selection is deterministic and independent of either signal unless
    explicitly supplied.  This is a visualization, not a metric or a means to
    tune a held-out prediction.  Optional ``scale_limits`` allows a frozen
    multi-candidate display contract; every render writes a canonical-hashed
    JSON sidecar with slices, structural-mask, optional paired target-validity,
    input digests, sampling, scales, and output identity. The structural mask
    alone controls all display context; target validity is authenticated only
    to bind this development visualization to the paired scoring support.
    """
    real = _to_numpy_tdhw(real).astype(np.float64, copy=False)
    syn = _to_numpy_tdhw(synthetic).astype(np.float64, copy=False)
    if real.shape != syn.shape:
        raise ValueError(
            "real and synthetic 4D arrays must have exactly matching shapes; "
            f"got {real.shape} and {syn.shape}"
        )
    if real.ndim != 4 or real.shape[0] < 4:
        raise ValueError("4D visualization requires at least four temporal frames")
    if not np.isfinite(real).all() or not np.isfinite(syn).all():
        raise ValueError("real or synthetic fMRI contains NaN or infinity")
    if not np.isfinite(repetition_time) or repetition_time <= 0.0:
        raise ValueError("repetition_time must be positive and finite")
    sigma = float(high_pass_sigma_voxels)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("high_pass_sigma_voxels must be positive and finite")

    frame_count, depth, height, width = real.shape
    if brain_mask is None:
        raise ValueError("an independent structural brain_mask is required")
    support = _to_numpy_mask(brain_mask, (depth, height, width))
    paired_validity = None
    if target_validity_mask is not None:
        paired_validity = _to_numpy_mask(
            target_validity_mask,
            (depth, height, width),
        )
        if bool((paired_validity & ~support).any()):
            raise ValueError(
                "target-validity mask lies outside independent structural support"
            )
        exact_target_support = np.any(real != 0.0, axis=0)
        if not np.array_equal(paired_validity, exact_target_support):
            raise ValueError(
                "target-validity mask differs from exact paired real-fMRI support"
            )
    real_mean = np.mean(real, axis=0)
    syn_mean = np.mean(syn, axis=0)
    real_dynamic = real - real_mean[None, ...]
    syn_dynamic = syn - syn_mean[None, ...]
    real_std = np.nanstd(real_dynamic, axis=0)
    syn_std = np.nanstd(syn_dynamic, axis=0)

    if slice_idx is None:
        # Slice choice is independent of real and predicted fMRI values.
        d = _largest_mask_slice(support, 0)
        h = _largest_mask_slice(support, 1)
        w = _largest_mask_slice(support, 2)
    else:
        if len(slice_idx) != 3:
            raise ValueError("slice_idx must contain x, y, and z indices")
        d, h, w = (int(value) for value in slice_idx)
        if not (0 <= d < depth and 0 <= h < height and 0 <= w < width):
            raise ValueError("slice_idx lies outside the 4D spatial grid")
    index = (d, h, w)

    if frame_indices is None:
        selected = np.linspace(0, frame_count - 1, 4, dtype=int).tolist()
    else:
        if len(frame_indices) != 4:
            raise ValueError("frame_indices must contain exactly four indices")
        selected = [int(value) for value in frame_indices]
        if len(set(selected)) != 4 or not all(
            0 <= value < frame_count for value in selected
        ):
            raise ValueError("frame_indices must be four distinct in-range indices")

    real_frames = [_masked_axial(real_dynamic[t], support, w) for t in selected]
    syn_frames = [_masked_axial(syn_dynamic[t], support, w) for t in selected]
    difference_frames = [b - a for a, b in zip(real_frames, syn_frames)]
    real_high_by_frame = {}
    syn_high_by_frame = {}
    real_high_samples = []
    syn_high_samples = []
    sample_spatial = None
    high_interior = None
    for temporal_index in range(frame_count):
        real_detail, real_interior = mask_normalized_high_pass(
            real_dynamic[temporal_index], support, sigma
        )
        syn_detail, syn_interior = mask_normalized_high_pass(
            syn_dynamic[temporal_index], support, sigma
        )
        if not np.array_equal(real_interior, syn_interior):
            raise RuntimeError("paired high-pass interior supports diverged")
        if high_interior is not None and not np.array_equal(
            high_interior, real_interior
        ):
            raise RuntimeError("high-pass interior support changed across frames")
        high_interior = real_interior
        if sample_spatial is None:
            sample_spatial = np.flatnonzero(real_interior.reshape(-1))
            per_frame_samples = min(
                sample_spatial.size, max(1, 1_000_000 // frame_count)
            )
            sample_spatial = sample_spatial[
                np.linspace(
                    0, sample_spatial.size - 1, per_frame_samples, dtype=np.int64
                )
            ]
        real_high_samples.append(real_detail.reshape(-1)[sample_spatial])
        syn_high_samples.append(syn_detail.reshape(-1)[sample_spatial])
        if temporal_index in selected:
            real_high_by_frame[temporal_index] = _masked_axial(
                real_detail, real_interior, w
            )
            syn_high_by_frame[temporal_index] = _masked_axial(
                syn_detail, syn_interior, w
            )
    if high_interior is None:
        raise RuntimeError("high-pass calculation produced no support")
    real_high = [real_high_by_frame[value] for value in selected]
    syn_high = [syn_high_by_frame[value] for value in selected]
    high_difference = [b - a for a, b in zip(real_high, syn_high)]

    # Independently demeaned dynamic panels cannot reveal a smoothed or biased
    # temporal-mean anatomy.  Preserve those useful dynamic panels, but add a
    # separate temporal-mean texture audit on the exact same boundary-safe
    # support.  This remains a display-only comparison and does not change the
    # raw prediction.
    real_mean_high, real_mean_interior = mask_normalized_high_pass(
        real_mean, support, sigma
    )
    syn_mean_high, syn_mean_interior = mask_normalized_high_pass(
        syn_mean, support, sigma
    )
    if not np.array_equal(real_mean_interior, syn_mean_interior):
        raise RuntimeError("paired temporal-mean high-pass supports diverged")
    if not np.array_equal(real_mean_interior, high_interior):
        raise RuntimeError(
            "temporal-mean and dynamic high-pass interior supports diverged"
        )
    mean_texture_index = tuple(
        _largest_mask_slice(real_mean_interior, axis) for axis in range(3)
    )
    real_mean_high_planes = [
        np.where(mask_plane, value_plane, np.nan)
        for value_plane, mask_plane in zip(
            _tri_planes(real_mean_high, mean_texture_index),
            _tri_planes(real_mean_interior, mean_texture_index),
        )
    ]
    syn_mean_high_planes = [
        np.where(mask_plane, value_plane, np.nan)
        for value_plane, mask_plane in zip(
            _tri_planes(syn_mean_high, mean_texture_index),
            _tri_planes(real_mean_interior, mean_texture_index),
        )
    ]
    mean_high_difference_planes = [
        syn_plane - real_plane
        for real_plane, syn_plane in zip(real_mean_high_planes, syn_mean_high_planes)
    ]

    real_dynamic_sample = _sample_masked_tdhw(real_dynamic, support)
    syn_dynamic_sample = _sample_masked_tdhw(syn_dynamic, support)
    computed_limits = {
        "dynamic_abs_limit": _finite_percentile(
            np.abs(np.concatenate([real_dynamic_sample, syn_dynamic_sample])),
            98.0,
            1e-6,
        ),
        "dynamic_difference_limit": _finite_percentile(
            np.abs(real_dynamic_sample - syn_dynamic_sample), 98.0, 1e-6
        ),
    }
    real_high_sample = np.concatenate(real_high_samples)
    syn_high_sample = np.concatenate(syn_high_samples)
    computed_limits["high_pass_abs_limit"] = _finite_percentile(
        np.abs(np.concatenate([real_high_sample, syn_high_sample])),
        98.0,
        1e-6,
    )
    computed_limits["high_pass_difference_limit"] = _finite_percentile(
        np.abs(real_high_sample - syn_high_sample), 98.0, 1e-6
    )
    mean_high_real_values = real_mean_high[real_mean_interior]
    mean_high_syn_values = syn_mean_high[real_mean_interior]
    computed_limits["temporal_mean_high_pass_abs_limit"] = _finite_percentile(
        np.abs(np.concatenate([mean_high_real_values, mean_high_syn_values])),
        98.0,
        1e-6,
    )
    computed_limits["temporal_mean_high_pass_difference_limit"] = _finite_percentile(
        np.abs(mean_high_real_values - mean_high_syn_values),
        98.0,
        1e-6,
    )

    real_std_planes = [
        np.where(mask_plane, value_plane, np.nan)
        for value_plane, mask_plane in zip(
            _tri_planes(real_std, index), _tri_planes(support, index)
        )
    ]
    syn_std_planes = [
        np.where(mask_plane, value_plane, np.nan)
        for value_plane, mask_plane in zip(
            _tri_planes(syn_std, index), _tri_planes(support, index)
        )
    ]
    std_difference_planes = [b - a for a, b in zip(real_std_planes, syn_std_planes)]
    computed_limits["temporal_std_limit"] = _finite_percentile(
        np.concatenate([real_std[support], syn_std[support]]),
        98.0,
        1e-6,
    )
    computed_limits["temporal_std_difference_limit"] = _finite_percentile(
        np.abs(real_std[support] - syn_std[support]),
        98.0,
        1e-6,
    )

    required_scale_keys = tuple(computed_limits)
    if scale_limits is None:
        limits = computed_limits
        scale_source = "computed_joint_real_prediction_single_selected_output_only"
    else:
        if set(scale_limits) != set(required_scale_keys):
            raise ValueError(
                "scale_limits keys must be exactly " + ", ".join(required_scale_keys)
            )
        limits = {key: float(scale_limits[key]) for key in required_scale_keys}
        if not all(np.isfinite(value) and value > 0.0 for value in limits.values()):
            raise ValueError("every frozen scale limit must be positive and finite")
        scale_source = "caller_supplied_frozen_multi_candidate_contract"
    dynamic_limit = limits["dynamic_abs_limit"]
    difference_limit = limits["dynamic_difference_limit"]
    high_limit = limits["high_pass_abs_limit"]
    high_difference_limit = limits["high_pass_difference_limit"]
    mean_high_limit = limits["temporal_mean_high_pass_abs_limit"]
    mean_high_difference_limit = limits["temporal_mean_high_pass_difference_limit"]
    std_limit = limits["temporal_std_limit"]
    std_difference_limit = limits["temporal_std_difference_limit"]

    fig, axes = plt.subplots(12, 4, figsize=(16.0, 33.0), layout="constrained")
    frame_titles = [f"frame {frame}" for frame in selected]
    image_rows = (
        ("real dynamic", real_frames, "coolwarm", -dynamic_limit, dynamic_limit),
        ("predicted dynamic", syn_frames, "coolwarm", -dynamic_limit, dynamic_limit),
        (
            "frame-aligned dynamic residual (predicted - real)*",
            difference_frames,
            "coolwarm",
            -difference_limit,
            difference_limit,
        ),
        ("real spatial high-pass", real_high, "coolwarm", -high_limit, high_limit),
        ("predicted spatial high-pass", syn_high, "coolwarm", -high_limit, high_limit),
        (
            "frame-aligned high-pass residual (predicted - real)*",
            high_difference,
            "coolwarm",
            -high_difference_limit,
            high_difference_limit,
        ),
    )
    for row_index, (label, images, color_map, low, high) in enumerate(image_rows):
        plotted = None
        for column_index, image in enumerate(images):
            plotted = _display_image(
                axes[row_index, column_index],
                image,
                cmap=color_map,
                vmin=low,
                vmax=high,
                title=frame_titles[column_index] if row_index in (0, 3) else None,
                ylabel=label if column_index == 0 else None,
            )
        fig.colorbar(plotted, ax=axes[row_index, :], shrink=0.78, pad=0.012)

    mean_d, mean_h, mean_w = mean_texture_index
    mean_texture_plane_titles = (
        f"axial z={mean_w}",
        f"sagittal x={mean_d}",
        f"coronal y={mean_h}",
    )
    mean_texture_rows = (
        (
            "real temporal-mean high-pass",
            real_mean_high_planes,
            "coolwarm",
            -mean_high_limit,
            mean_high_limit,
        ),
        (
            "predicted temporal-mean high-pass",
            syn_mean_high_planes,
            "coolwarm",
            -mean_high_limit,
            mean_high_limit,
        ),
        (
            "temporal-mean high-pass residual (predicted - real)",
            mean_high_difference_planes,
            "coolwarm",
            -mean_high_difference_limit,
            mean_high_difference_limit,
        ),
    )
    for offset, (label, images, color_map, low, high) in enumerate(mean_texture_rows):
        row_index = 6 + offset
        plotted = None
        for column_index, image in enumerate(images):
            plotted = _display_image(
                axes[row_index, column_index],
                image,
                cmap=color_map,
                vmin=low,
                vmax=high,
                title=(
                    mean_texture_plane_titles[column_index] if row_index == 6 else None
                ),
                ylabel=label if column_index == 0 else None,
            )
        fig.colorbar(plotted, ax=axes[row_index, :3], shrink=0.78, pad=0.012)
        axes[row_index, 3].axis("off")

    plane_titles = (f"axial z={w}", f"sagittal x={d}", f"coronal y={h}")
    std_rows = (
        ("real temporal STD", real_std_planes, "magma", 0.0, std_limit),
        ("predicted temporal STD", syn_std_planes, "magma", 0.0, std_limit),
        (
            "temporal STD residual (predicted - real)",
            std_difference_planes,
            "coolwarm",
            -std_difference_limit,
            std_difference_limit,
        ),
    )
    for offset, (label, images, color_map, low, high) in enumerate(std_rows):
        row_index = 9 + offset
        plotted = None
        for column_index, image in enumerate(images):
            plotted = _display_image(
                axes[row_index, column_index],
                image,
                cmap=color_map,
                vmin=low,
                vmax=high,
                title=plane_titles[column_index] if row_index == 9 else None,
                ylabel=label if column_index == 0 else None,
            )
        fig.colorbar(plotted, ax=axes[row_index, :3], shrink=0.78, pad=0.012)

    real_matrix = real_dynamic[:, support]
    syn_matrix = syn_dynamic[:, support]
    time_seconds = np.arange(frame_count, dtype=np.float64) * repetition_time

    trace_axis = axes[9, 3]
    trace_axis.plot(
        time_seconds, np.mean(real_matrix, axis=1), color="#1f5f99", label="real"
    )
    trace_axis.plot(
        time_seconds,
        np.mean(syn_matrix, axis=1),
        color="#d17a22",
        linestyle="--",
        label="predicted",
    )
    trace_axis.set_title("brain-support mean signal", fontsize=10)
    trace_axis.set_xlabel("time (s)")
    trace_axis.legend(frameon=False, fontsize=8)

    real_dvars = np.sqrt(np.mean(np.diff(real_matrix, axis=0) ** 2, axis=1))
    syn_dvars = np.sqrt(np.mean(np.diff(syn_matrix, axis=0) ** 2, axis=1))
    dvars_axis = axes[10, 3]
    dvars_axis.plot(time_seconds[1:], real_dvars, color="#1f5f99", label="real")
    dvars_axis.plot(
        time_seconds[1:], syn_dvars, color="#d17a22", linestyle="--", label="predicted"
    )
    dvars_axis.set_title("DVARS", fontsize=10)
    dvars_axis.set_xlabel("time (s)")

    sample_count = min(4096, real_matrix.shape[1])
    sample_indices = np.linspace(0, real_matrix.shape[1] - 1, sample_count, dtype=int)
    frequencies = np.fft.rfftfreq(frame_count, d=repetition_time)
    real_fft = np.fft.rfft(real_matrix[:, sample_indices], axis=0)
    syn_fft = np.fft.rfft(syn_matrix[:, sample_indices], axis=0)
    # One-sided periodogram density: units of input-signal squared per Hz.
    periodogram_scale = repetition_time / frame_count
    real_density = periodogram_scale * np.abs(real_fft) ** 2
    syn_density = periodogram_scale * np.abs(syn_fft) ** 2
    one_sided_stop = -1 if frame_count % 2 == 0 else None
    real_density[1:one_sided_stop] *= 2.0
    syn_density[1:one_sided_stop] *= 2.0
    real_power = np.mean(real_density, axis=1)
    syn_power = np.mean(syn_density, axis=1)
    spectrum_axis = axes[11, 3]
    spectrum_axis.plot(frequencies[1:], real_power[1:], color="#1f5f99", label="real")
    spectrum_axis.plot(
        frequencies[1:],
        syn_power[1:],
        color="#d17a22",
        linestyle="--",
        label="predicted",
    )
    passband = (frequencies >= 0.01) & (frequencies <= 0.10)
    # ``np.trapz`` is available throughout the declared NumPy>=1.24 support
    # range; ``np.trapezoid`` was introduced only in NumPy 2.x.
    real_band = float(np.trapz(real_power[passband], frequencies[passband]))
    syn_band = float(np.trapz(syn_power[passband], frequencies[passband]))
    spectrum_axis.set_title(
        "mean one-sided periodogram (signal²/Hz)\n"
        f"integrated 0.01–0.10 Hz power: real={real_band:.3g}, predicted={syn_band:.3g}",
        fontsize=9,
    )
    spectrum_axis.set_xlabel("frequency (Hz)")

    for axis in (trace_axis, dvars_axis, spectrum_axis):
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    exterior = ~support
    outside_max = float(np.max(np.abs(syn[:, exterior]))) if exterior.any() else 0.0
    real_outside_max = (
        float(np.max(np.abs(real[:, exterior]))) if exterior.any() else 0.0
    )
    real_reference = float(np.percentile(np.abs(real[:, support]), 99.0))
    leakage_ratio = outside_max / max(real_reference, 1e-12)
    fig.suptitle(
        f"{title}\nindependent-mask slices and whole-mask shared scales; "
        f"z={w}; TR={repetition_time:g} s; outside max real/predicted="
        f"{real_outside_max:.3g}/{outside_max:.3g}; predicted outside / real in-mask q99="
        f"{leakage_ratio:.3g}\n"
        "*Frame-aligned residuals are descriptive only: resting-state phase is not "
        "identifiable from structure and these panels must not rank or gate models.\n"
        "Temporal-mean high-pass rows expose static spatial texture that the "
        "demeaned dynamic rows intentionally remove.",
        fontsize=14,
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    display_record = {
        "schema": "connect4-in-memory-4d-display-contract-v1",
        "inputs": {
            "real_array_sha256": _array_sha256(real),
            "synthetic_array_sha256": _array_sha256(syn),
            "second_input_semantic_role": "predicted",
            "structural_mask_sha256": _array_sha256(support.astype(np.uint8)),
            "target_validity_mask_sha256": (
                None
                if paired_validity is None
                else _array_sha256(paired_validity.astype(np.uint8))
            ),
            "structural_mask_semantic_role": (
                "target-independent-display-context-and-output-support"
            ),
            "target_validity_mask_semantic_role": (
                "paired-scoring-support-not-used-for-display"
                if paired_validity is not None
                else "not-supplied-diagnostic-display-only"
            ),
            "shape_tdhw": list(real.shape),
        },
        "display": {
            "slice_indices_xyz": [d, h, w],
            "slice_selection": "largest independent structural-mask area"
            if slice_idx is None
            else "caller supplied",
            "frame_indices": selected,
            "frame_selection": "deterministic evenly spaced"
            if frame_indices is None
            else "caller supplied",
            "repetition_time_seconds": repetition_time,
            "image_interpolation": "nearest",
            "high_pass": {
                "method": "mask-normalized 3D Gaussian minus low-pass",
                "sigma_voxels": sigma,
                "erosion_iterations": 1,
                "scale_support": "eroded interior",
            },
            "temporal_mean_texture": {
                "slice_indices_xyz": list(mean_texture_index),
                "slice_selection": "largest eroded structural-mask area",
                "shared_real_prediction_scale": True,
                "display_only_not_model_selection": True,
            },
            "scale_percentile": 98.0,
            "scale_sample_rule": "deterministic linspace, at most 1000000 values",
            "scale_source": scale_source,
            "scale_limits": limits,
            "phase_aligned_residuals_report_only": True,
            "frame_aligned_residuals_report_only": True,
            "residual_semantic": "predicted_minus_real_signed",
        },
        "output": {
            "path": str(output.resolve(strict=True)),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
    }
    display_record["record_sha256"] = hashlib.sha256(
        json.dumps(display_record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".display.json")
    sidecar.write_text(json.dumps(display_record, indent=2) + "\n", encoding="utf-8")
    return out_path


def plot_real_vs_synthetic(
    real,
    synthetic,
    out_path: str,
    slice_idx: Optional[Sequence[int]] = None,
    cmap: str = "magma",
    title: str = "CONNECT-4: real vs predicted rs-fMRI",
    allow_temporal_mean_legacy: bool = False,
    **_,
):
    """
    Save a legacy temporal-mean figure comparing real and predicted fMRI.
    anatomical planes.

    Rows:    real / predicted / |difference|
    Columns: Axial / Sagittal / Coronal

    The temporal-mean volume is shown. The most informative slice per axis
    (max variance of the real mean) is chosen automatically. Real and predicted
    data share one robust intensity window so attenuation, bias, and smoothing
    remain visually apparent. The non-negative absolute-difference row uses a
    separate zero-anchored window.
    This view is not 4D QA and is fail-closed unless the caller explicitly
    acknowledges the legacy-only diagnostic through
    ``allow_temporal_mean_legacy=True``.
    """
    if not allow_temporal_mean_legacy:
        raise RuntimeError(
            "legacy temporal-mean visualization is disabled; use "
            "plot_real_vs_synthetic_4d with an independent structural mask"
        )
    warnings.warn(
        "temporal-mean-only visualization is deprecated and is not 4D QA",
        DeprecationWarning,
        stacklevel=2,
    )
    real = _to_numpy_tdhw(real)
    syn = _to_numpy_tdhw(synthetic)
    if real.shape != syn.shape:
        raise ValueError("legacy real and synthetic arrays must have matching shapes")
    if not np.isfinite(real).all() or not np.isfinite(syn).all():
        raise ValueError("legacy real or synthetic array contains NaN or infinity")
    T = real.shape[0]
    rmean = real[:T].mean(0)  # [D, H, W]
    smean = syn[:T].mean(0)
    D, H, W = rmean.shape

    # informative slice per axis = max variance of the real mean along that axis
    if slice_idx is None:
        d = int(rmean.reshape(D, -1).var(axis=1).argmax())
        h = int(rmean.transpose(1, 0, 2).reshape(H, -1).var(axis=1).argmax())
        w = int(rmean.transpose(2, 0, 1).reshape(W, -1).var(axis=1).argmax())
    else:
        d, h, w = slice_idx
    idx = (d, h, w)

    real_row = list(_tri_planes(rmean, idx))
    syn_row = list(_tri_planes(smean, idx))
    diff_row = [np.abs(a - b) for a, b in zip(real_row, syn_row)]

    shared_signal = np.concatenate([plane.ravel() for plane in (*real_row, *syn_row)])
    signal_lo, signal_hi = _robust_window(shared_signal)
    difference_values = np.concatenate([plane.ravel() for plane in diff_row])
    finite_difference = difference_values[np.isfinite(difference_values)]
    difference_hi = (
        float(np.percentile(finite_difference, 98.0)) if finite_difference.size else 1.0
    )
    if not np.isfinite(difference_hi) or difference_hi <= 0.0:
        difference_hi = 1e-6
    rows = [
        ("real", real_row, signal_lo, signal_hi),
        ("predicted", syn_row, signal_lo, signal_hi),
        ("|difference|", diff_row, 0.0, difference_hi),
    ]
    col_titles = [f"Axial (z={w})", f"Sagittal (x={d})", f"Coronal (y={h})"]

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.0))
    for ri, (name, imgs, lo, hi) in enumerate(rows):
        im = None
        for ci, img in enumerate(imgs):
            ax = axes[ri, ci]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(name, fontsize=12)
        fig.colorbar(im, ax=axes[ri, -1], fraction=0.046, pad=0.02)

    fig.suptitle(f"{title}  (LEGACY TEMPORAL-MEAN ONLY — NOT 4D QA)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fmri_nifti(
    vol,
    out_path: str,
    affine: Optional[np.ndarray] = None,
    repetition_time: float = 3.0,
):
    """Save a 4D fMRI volume [T,D,H,W] (or [B,C,T,D,H,W]) as a NIfTI (.nii.gz)."""
    import nibabel as nib

    v = _to_numpy_tdhw(vol)  # [T, D, H, W]
    v = np.transpose(v, (1, 2, 3, 0))  # NIfTI wants [D, H, W, T]
    if affine is None:
        raise ValueError("an anatomical reference affine is required")
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not np.isfinite(repetition_time) or repetition_time <= 0:
        raise ValueError("repetition_time must be positive and finite")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(v.astype(np.float32), affine)
    image.header.set_zooms((*image.header.get_zooms()[:3], float(repetition_time)))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, out_path)
    return out_path
