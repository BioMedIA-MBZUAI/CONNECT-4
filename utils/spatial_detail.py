"""Boundary-safe spatial-detail primitives shared by QA and visualisation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


def mask_normalized_high_pass(
    volume: np.ndarray,
    mask: np.ndarray,
    sigma_voxels: float,
    *,
    erosion_iterations: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a Gaussian high-pass image and a conservative interior support.

    The low-pass image is a normalized convolution: both the zero-filled signal
    and the binary support are smoothed, then divided.  Consequently, the zero
    exterior cannot create a bright high-frequency rim at the mask boundary.
    Values outside ``mask`` are returned as zero.  The second result is a
    one-or-more-voxel eroded support suitable for correlations; it falls back to
    the full mask only when erosion would remove the entire support.
    """
    values = np.asarray(volume, dtype=np.float64)
    support = np.asarray(mask, dtype=bool)
    sigma = float(sigma_voxels)
    if values.ndim != 3:
        raise ValueError(f"high-pass volume must be 3D, got shape {values.shape}")
    if support.shape != values.shape:
        raise ValueError(
            f"high-pass mask shape {support.shape} differs from {values.shape}"
        )
    if not bool(support.any()):
        raise ValueError("high-pass mask is empty")
    if not np.isfinite(values).all():
        raise ValueError("high-pass volume contains NaN or infinity")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("high-pass sigma must be positive and finite")
    if (
        isinstance(erosion_iterations, bool)
        or int(erosion_iterations) != erosion_iterations
        or int(erosion_iterations) < 0
    ):
        raise ValueError("erosion_iterations must be a nonnegative integer")

    support_values = support.astype(np.float64)
    weights = ndimage.gaussian_filter(
        support_values, sigma=sigma, mode="constant", cval=0.0
    )
    numerator = ndimage.gaussian_filter(
        np.where(support, values, 0.0),
        sigma=sigma,
        mode="constant",
        cval=0.0,
    )
    valid = support & (weights > np.finfo(np.float64).eps)
    low_pass = np.divide(
        numerator,
        weights,
        out=np.zeros_like(numerator),
        where=weights > np.finfo(np.float64).eps,
    )
    detail = np.zeros_like(values)
    detail[valid] = values[valid] - low_pass[valid]

    interior = support
    if int(erosion_iterations) > 0:
        candidate = ndimage.binary_erosion(
            support, iterations=int(erosion_iterations)
        )
        if bool(candidate.any()):
            interior = candidate
    return detail, interior


__all__ = ["mask_normalized_high_pass"]
