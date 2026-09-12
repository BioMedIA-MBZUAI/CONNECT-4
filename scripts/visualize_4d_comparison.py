#!/usr/bin/env python3
"""Strict real-versus-predicted 4D fMRI visualisation in a paired image space.

The input images must already be paired on the same spatial grid, affine, and
temporal grid.  The script never registers or resamples either fMRI volume.  It
only reorients both arrays together to canonical RAS+ for display.  An optional
3D brain mask may be nearest-neighbour resampled when it spans the same paired
world geometry; a shifted, cropped, rotated, or otherwise incompatible mask is
rejected.

Outputs for each subject are:

* a representative-frame montage with one shared real/predicted scale;
* temporal diagnostics (global signal, framewise spatial r, and RMSE); and
* whole-FOV exterior-artifact maximum projections;
* optional exact-label ROI-FC and normalized spectral diagnostics; and
* a fixed-scale tri-planar animation, MP4 when ffmpeg is available and GIF
  otherwise.

Example:

    python -m scripts.visualize_4d_comparison \
        --real /data/fMRI/S001_fMRI.nii.gz \
        --pred /outputs/S001_predicted.nii.gz \
        --mask /data/Masks/S001_mask.nii.gz \
        --out-dir /outputs/S001_visualisation
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Dict, Optional, Sequence, Tuple
import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation as mpl_animation
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from utils.source_provenance import implementation_source_records, sha256_file
from utils.spatial_detail import mask_normalized_high_pass


AFFINE_ATOL_MM = 1e-4
WORLD_GEOMETRY_ATOL_MM = 1e-3
TR_ATOL_SECONDS = 1e-6
ROBUST_PERCENTILE = 99.0
SPACE_DESCRIPTION = "paired input space; canonical RAS+ display"


@dataclass(frozen=True)
class _OutputDirectory:
    """A held, canonical output root used for descriptor-relative publication."""

    path: Path
    descriptor: int
    identity: tuple[int, int]
    bindings: dict[str, dict[str, object]] = field(default_factory=dict)


def _open_output_directory(path: Path) -> _OutputDirectory:
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = Path(os.path.abspath(requested))
    parent = requested.parent
    if parent.resolve(strict=True) != parent or parent.is_symlink():
        raise ValueError("visualization output parent must be canonical and non-symlinked")
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    created = False
    try:
        try:
            os.mkdir(requested.name, 0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        descriptor = os.open(
            requested.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        current = requested.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            os.close(descriptor)
            raise RuntimeError("visualization output directory changed while opening")
        if created:
            os.fsync(parent_descriptor)
        return _OutputDirectory(
            path=requested,
            descriptor=descriptor,
            identity=(int(opened.st_dev), int(opened.st_ino)),
        )
    finally:
        os.close(parent_descriptor)


def _assert_output_directory(output: _OutputDirectory) -> None:
    opened = os.fstat(output.descriptor)
    current = output.path.lstat()
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (opened.st_dev, opened.st_ino) != output.identity
        or (current.st_dev, current.st_ino) != output.identity
    ):
        raise RuntimeError("visualization output directory was replaced")


def _validate_output_path(path: Path, output: _OutputDirectory) -> None:
    if path.parent != output.path or path.name in {"", ".", ".."}:
        raise ValueError("visualization output escaped its held output directory")


def _write_output_new(path: Path, payload: bytes, output: _OutputDirectory) -> None:
    _validate_output_path(path, output)
    _assert_output_directory(output)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=output.descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError(
                "visualization output is not a new single-link regular file"
            )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise RuntimeError("visualization output write made no progress")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if final.st_nlink != 1 or final.st_size != len(payload):
            raise RuntimeError("visualization output link or size changed while writing")
        current = os.stat(
            path.name,
            dir_fd=output.descriptor,
            follow_symlinks=False,
        )
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        if any(
            getattr(final, field_name) != getattr(current, field_name)
            for field_name in stable_fields
        ):
            raise RuntimeError("visualization output path changed while writing")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        read_size = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            read_size += len(block)
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        if read_size != len(payload) or digest.hexdigest() != expected_sha256:
            raise RuntimeError("visualization output bytes differ after durable write")
        output.bindings[path.name] = {
            field_name: int(getattr(final, field_name))
            for field_name in stable_fields
        } | {"sha256": expected_sha256}
        os.fsync(output.descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _assert_output_directory(output)


def _output_snapshot(path: Path, output: _OutputDirectory) -> dict[str, object]:
    _validate_output_path(path, output)
    _assert_output_directory(output)
    descriptor = os.open(
        path.name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=output.descriptor,
    )
    digest = hashlib.sha256()
    expected = output.bindings.get(path.name)
    if expected is None:
        os.close(descriptor)
        raise RuntimeError("visualization output lacks its creation-time binding")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError("visualization output is not a single-link file")
        size = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            size += len(block)
        final = os.fstat(descriptor)
        current = os.stat(
            path.name,
            dir_fd=output.descriptor,
            follow_symlinks=False,
        )
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
        if any(
            getattr(opened, field) != getattr(final, field)
            or getattr(opened, field) != getattr(current, field)
            for field in stable_fields
        ) or size != opened.st_size:
            raise RuntimeError("visualization output changed while hashing")
        for field_name in stable_fields:
            if int(getattr(opened, field_name)) != int(expected[field_name]):
                raise RuntimeError(
                    "visualization output identity changed after its creation"
                )
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError("visualization output bytes changed after its creation")
    finally:
        os.close(descriptor)
    _assert_output_directory(output)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
    }


def _save_figure_new(
    figure: matplotlib.figure.Figure,
    path: Path,
    *,
    output: _OutputDirectory | None,
    dpi: int,
) -> Path:
    owned = output is None
    if owned:
        output = _open_output_directory(path.parent)
        path = output.path / path.name
    assert output is not None
    buffer = io.BytesIO()
    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        _write_output_new(path, buffer.getvalue(), output)
        return path
    finally:
        plt.close(figure)
        if owned:
            os.close(output.descriptor)


@dataclass(frozen=True)
class ComparisonVolume:
    """Validated and canonically oriented paired 4D fMRI data."""

    real: np.ndarray
    predicted: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    tr_seconds: float
    mask_description: str
    mask_resampled: bool
    roi_labels: Optional[np.ndarray]
    roi_description: Optional[str]
    space_description: str = SPACE_DESCRIPTION

    @property
    def num_frames(self) -> int:
        return int(self.real.shape[-1])


def _nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _validate_affine(affine: np.ndarray, label: str) -> None:
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{label} affine must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise ValueError(f"{label} affine is singular")


def _header_tr_seconds(image: nib.spatialimages.SpatialImage, label: str) -> float:
    zooms = image.header.get_zooms()
    if len(zooms) < 4:
        raise ValueError(f"{label} NIfTI header has no temporal sampling interval")
    value = float(zooms[3])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} NIfTI header TR must be positive and finite")
    time_unit = image.header.get_xyzt_units()[1]
    unit_scale = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}
    if time_unit not in unit_scale:
        raise ValueError(
            f"{label} NIfTI header must declare sec, msec, or usec time units; "
            f"got {time_unit!r}"
        )
    return value * unit_scale[time_unit]


def _world_corners(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """World coordinates of the eight voxel-edge corners of the image FOV."""
    shape = tuple(int(size) for size in image.shape[:3])
    voxel_corners = np.asarray(
        list(product(*((-0.5, float(size) - 0.5) for size in shape))),
        dtype=np.float64,
    )
    return nib.affines.apply_affine(image.affine, voxel_corners)


def _axis_directions(affine: np.ndarray) -> np.ndarray:
    axes = np.asarray(affine, dtype=np.float64)[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if np.any(lengths <= 1e-12):
        raise ValueError("image affine contains a degenerate spatial axis")
    return axes / lengths[None, :]


def _same_world_geometry(
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> bool:
    """Whether two canonical grids span the same oriented world support."""
    directions_match = np.allclose(
        _axis_directions(source.affine),
        _axis_directions(reference.affine),
        rtol=0.0,
        atol=1e-5,
    )
    corners_match = np.allclose(
        _world_corners(source),
        _world_corners(reference),
        rtol=0.0,
        atol=WORLD_GEOMETRY_ATOL_MM,
    )
    return bool(directions_match and corners_match)


def _load_mask(
    mask_path: Optional[str | Path],
    reference: nib.spatialimages.SpatialImage,
) -> Tuple[np.ndarray, str, bool]:
    if mask_path is None:
        raise ValueError(
            "an independent structural brain mask is required; target-derived "
            "or prediction-derived display support is forbidden"
        )

    mask_image = nib.load(str(mask_path))
    if mask_image.ndim != 3:
        raise ValueError(f"brain mask must be a 3D NIfTI, got shape {mask_image.shape}")
    _validate_affine(mask_image.affine, "brain mask")
    canonical_mask = nib.as_closest_canonical(mask_image)
    same_grid = (
        tuple(canonical_mask.shape) == tuple(reference.shape[:3])
        and np.allclose(
            canonical_mask.affine,
            reference.affine,
            rtol=0.0,
            atol=AFFINE_ATOL_MM,
        )
    )
    resampled = False
    if not same_grid:
        if not _same_world_geometry(canonical_mask, reference):
            raise ValueError(
                "brain mask does not span the same oriented world geometry "
                "as the paired fMRI volumes; refusing implicit registration"
            )
        canonical_mask = resample_from_to(
            canonical_mask,
            (reference.shape[:3], reference.affine),
            order=0,
            # Same voxel-edge FOV has already been established.  ``nearest``
            # prevents SciPy from eroding an edge voxel merely because the
            # target centre lies beyond the source centre but inside its cell.
            mode="nearest",
            cval=0.0,
        )
        resampled = True

    mask_values = canonical_mask.get_fdata(dtype=np.float32)
    if not np.isfinite(mask_values).all():
        raise ValueError("brain mask contains NaN or infinity")
    mask = mask_values > 0
    if not bool(mask.any()):
        raise ValueError("brain mask is empty after canonicalisation/resampling")
    description = f"{Path(mask_path).name}"
    if resampled:
        description += " (nearest-neighbour resampled on matching world support)"
    return mask, description, resampled


def _load_roi_labels(
    roi_labels_path: Optional[str | Path],
    reference: nib.spatialimages.SpatialImage,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load exact-grid anatomical labels without registration or resampling."""
    if roi_labels_path is None:
        return None, None
    label_image = nib.load(str(roi_labels_path))
    if label_image.ndim != 3:
        raise ValueError(
            f"ROI label map must be a 3D NIfTI, got shape {label_image.shape}"
        )
    _validate_affine(label_image.affine, "ROI label map")
    canonical_labels = nib.as_closest_canonical(label_image)
    if tuple(canonical_labels.shape) != tuple(reference.shape[:3]):
        raise ValueError("ROI label map spatial shape differs from the paired fMRI grid")
    if not np.allclose(
        canonical_labels.affine,
        reference.affine,
        rtol=0.0,
        atol=AFFINE_ATOL_MM,
    ):
        raise ValueError("ROI label map affine differs from the paired fMRI grid")
    values = canonical_labels.get_fdata(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ROI label map contains NaN or infinity")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-5):
        raise ValueError("ROI label map must contain discrete integer labels")
    if float(rounded.min()) < 0.0 or float(rounded.max()) > np.iinfo(np.int32).max:
        raise ValueError("ROI labels must be zero-background positive int32 values")
    labels = rounded.astype(np.int32, copy=False)
    if np.unique(labels[labels > 0]).size < 2:
        raise ValueError("ROI label map must contain at least two positive regions")
    return labels, Path(roi_labels_path).name


def load_comparison(
    real_path: str | Path,
    predicted_path: str | Path,
    *,
    mask_path: Optional[str | Path] = None,
    roi_labels_path: Optional[str | Path] = None,
    tr_seconds: Optional[float] = None,
) -> ComparisonVolume:
    """Load and strictly validate a paired real/predicted 4D NIfTI comparison.

    ``tr_seconds`` is a validation override: it must exactly match the TR stored
    in both headers (within one microsecond).  It never replaces header timing.
    """
    real_image = nib.load(str(real_path))
    predicted_image = nib.load(str(predicted_path))
    for label, image in (("real", real_image), ("predicted", predicted_image)):
        if image.ndim != 4:
            raise ValueError(f"{label} fMRI must be 4D, got shape {image.shape}")
        _validate_affine(image.affine, label)
    if tuple(real_image.shape) != tuple(predicted_image.shape):
        raise ValueError(
            "real and predicted fMRI shapes differ: "
            f"{real_image.shape} vs {predicted_image.shape}"
        )
    if not np.allclose(
        real_image.affine,
        predicted_image.affine,
        rtol=0.0,
        atol=AFFINE_ATOL_MM,
    ):
        raise ValueError(
            "real and predicted fMRI affines differ; refusing to resample either volume"
        )

    real_tr = _header_tr_seconds(real_image, "real")
    predicted_tr = _header_tr_seconds(predicted_image, "predicted")
    if not np.isclose(real_tr, predicted_tr, rtol=0.0, atol=TR_ATOL_SECONDS):
        raise ValueError(
            f"real and predicted fMRI TR differ: {real_tr:g}s vs {predicted_tr:g}s"
        )
    if tr_seconds is not None:
        requested_tr = float(tr_seconds)
        if not np.isfinite(requested_tr) or requested_tr <= 0:
            raise ValueError("TR validation override must be positive and finite")
        if not np.isclose(
            requested_tr, real_tr, rtol=0.0, atol=TR_ATOL_SECONDS
        ):
            raise ValueError(
                f"TR validation override {requested_tr:g}s does not match "
                f"the paired NIfTI headers ({real_tr:g}s)"
            )

    canonical_real = nib.as_closest_canonical(real_image)
    canonical_predicted = nib.as_closest_canonical(predicted_image)
    if tuple(canonical_real.shape) != tuple(canonical_predicted.shape) or not np.allclose(
        canonical_real.affine,
        canonical_predicted.affine,
        rtol=0.0,
        atol=AFFINE_ATOL_MM,
    ):
        raise RuntimeError("paired images diverged during canonical orientation")
    axis_codes = nib.aff2axcodes(canonical_real.affine)
    if axis_codes != ("R", "A", "S"):
        raise RuntimeError(f"canonical display orientation is {axis_codes}, expected RAS+")

    real = canonical_real.get_fdata(dtype=np.float32)
    predicted = canonical_predicted.get_fdata(dtype=np.float32)
    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise ValueError("real or predicted fMRI contains NaN or infinity")
    mask, mask_description, mask_resampled = _load_mask(mask_path, canonical_real)
    roi_labels, roi_description = _load_roi_labels(
        roi_labels_path, canonical_real
    )
    return ComparisonVolume(
        real=real,
        predicted=predicted,
        mask=mask,
        affine=np.asarray(canonical_real.affine, dtype=np.float64),
        tr_seconds=real_tr,
        mask_description=mask_description,
        mask_resampled=mask_resampled,
        roi_labels=roi_labels,
        roi_description=roi_description,
    )


def _sample_masked_values(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    max_samples: int = 1_000_000,
) -> np.ndarray:
    """Deterministically sample a masked 4D array without a full masked copy."""
    spatial_indices = np.flatnonzero(mask.reshape(-1))
    num_frames = int(array.shape[-1])
    total = int(spatial_indices.size * num_frames)
    count = min(total, int(max_samples))
    if count < 1:
        raise ValueError("cannot sample an empty brain mask")
    positions = np.linspace(0, total - 1, num=count, dtype=np.int64)
    selected_spatial = spatial_indices[positions // num_frames]
    selected_time = positions % num_frames
    coordinates = np.unravel_index(selected_spatial, mask.shape)
    return np.asarray(
        array[coordinates[0], coordinates[1], coordinates[2], selected_time],
        dtype=np.float32,
    )


def compute_scale_limits(
    comparison: ComparisonVolume,
    *,
    percentile: float = ROBUST_PERCENTILE,
) -> Tuple[float, float]:
    """Fixed symmetric limits for signals and signed residuals."""
    if not 50.0 < float(percentile) <= 100.0:
        raise ValueError("robust scale percentile must be in (50, 100]")
    real_values = _sample_masked_values(comparison.real, comparison.mask)
    predicted_values = _sample_masked_values(comparison.predicted, comparison.mask)
    signal_values = np.concatenate((np.abs(real_values), np.abs(predicted_values)))
    residual_values = np.abs(predicted_values - real_values)
    signal_limit = float(np.percentile(signal_values, percentile))
    residual_limit = float(np.percentile(residual_values, percentile))
    if not np.isfinite(signal_limit) or signal_limit <= 1e-8:
        signal_limit = 1.0
    if not np.isfinite(residual_limit) or residual_limit <= 1e-8:
        residual_limit = max(signal_limit * 0.05, 1e-3)
    return signal_limit, residual_limit


def _largest_mask_slice(mask: np.ndarray, axis: int) -> int:
    reduce_axes = tuple(index for index in range(3) if index != axis)
    counts = mask.sum(axis=reduce_axes)
    candidates = np.flatnonzero(counts == counts.max())
    if candidates.size == 0:
        raise ValueError("cannot select a slice from an empty brain mask")
    centre = float(np.average(np.arange(counts.size), weights=counts))
    return int(candidates[np.abs(candidates - centre).argmin()])


def select_display_slices(mask: np.ndarray) -> Dict[str, int]:
    """Mask-only slice selection, independent of prediction quality."""
    return {
        "sagittal": _largest_mask_slice(mask, 0),
        "coronal": _largest_mask_slice(mask, 1),
        "axial": _largest_mask_slice(mask, 2),
    }


def _plane(
    volume: np.ndarray,
    mask: np.ndarray,
    plane_name: str,
    indices: Dict[str, int],
) -> np.ndarray:
    if plane_name == "axial":
        data = volume[:, :, indices[plane_name]]
        support = mask[:, :, indices[plane_name]]
    elif plane_name == "sagittal":
        data = volume[indices[plane_name], :, :]
        support = mask[indices[plane_name], :, :]
    elif plane_name == "coronal":
        data = volume[:, indices[plane_name], :]
        support = mask[:, indices[plane_name], :]
    else:
        raise ValueError(f"unknown anatomical plane {plane_name!r}")
    displayed = np.rot90(np.asarray(data, dtype=np.float32))
    displayed_mask = np.rot90(np.asarray(support, dtype=bool))
    return np.where(displayed_mask, displayed, np.nan)


def _plane_extent_mm(
    shape: Sequence[int], affine: np.ndarray, plane_name: str
) -> Tuple[float, float, float, float]:
    """Matplotlib extent with one display unit per physical millimetre.

    ``_plane`` rotates each slice by 90 degrees, so its displayed horizontal
    and vertical axes correspond to (X,Y), (Y,Z), or (X,Z), respectively.
    Voxel-edge lengths are used to avoid the familiar one-voxel FOV error.
    """
    spatial_shape = tuple(int(value) for value in shape[:3])
    if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
        raise ValueError("shape must contain three positive spatial dimensions")
    voxel_sizes = nib.affines.voxel_sizes(np.asarray(affine, dtype=np.float64))
    axes = {
        "axial": (0, 1),
        "sagittal": (1, 2),
        "coronal": (0, 2),
    }
    if plane_name not in axes:
        raise ValueError(f"unknown anatomical plane {plane_name!r}")
    horizontal, vertical = axes[plane_name]
    width = spatial_shape[horizontal] * float(voxel_sizes[horizontal])
    height = spatial_shape[vertical] * float(voxel_sizes[vertical])
    return (0.0, width, 0.0, height)


def representative_frames(num_frames: int, count: int = 6) -> np.ndarray:
    if num_frames < 1:
        raise ValueError("4D fMRI must contain at least one temporal frame")
    count = min(int(count), int(num_frames))
    return np.unique(np.rint(np.linspace(0, num_frames - 1, count)).astype(int))


def _comparison_rendering(
    comparison: ComparisonVolume,
    signal_limit: float,
) -> Tuple[matplotlib.colors.Colormap, Normalize]:
    """Select a shared signal scale without assuming z-scored input units."""
    real_values = _sample_masked_values(comparison.real, comparison.mask)
    predicted_values = _sample_masked_values(comparison.predicted, comparison.mask)
    signed = float(min(real_values.min(), predicted_values.min())) < -1e-6
    signal = plt.colormaps["coolwarm" if signed else "viridis"].copy()
    signal_norm: Normalize
    if signed:
        signal_norm = TwoSlopeNorm(
            vmin=-signal_limit, vcenter=0.0, vmax=signal_limit
        )
    else:
        signal_norm = Normalize(vmin=0.0, vmax=signal_limit)
    signal.set_bad("#111111")
    return signal, signal_norm


def _residual_colormap():
    residual = plt.colormaps["PuOr"].copy()
    residual.set_bad("#111111")
    return residual


def plot_frame_montage(
    comparison: ComparisonVolume,
    out_path: str | Path,
    *,
    scale_limits: Optional[Tuple[float, float]] = None,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Save an evenly spaced real/predicted/signed-residual frame montage."""
    out_path = Path(out_path)
    signal_limit, residual_limit = scale_limits or compute_scale_limits(comparison)
    signal_cmap, signal_norm = _comparison_rendering(comparison, signal_limit)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    residual_cmap = _residual_colormap()
    frames = representative_frames(comparison.num_frames, count=6)
    slices = select_display_slices(comparison.mask)
    axial_index = slices["axial"]

    fig, axes = plt.subplots(
        3,
        len(frames),
        figsize=(3.0 * len(frames), 8.7),
        squeeze=False,
        facecolor="white",
        constrained_layout=True,
    )
    signal_image = residual_image = None
    for column, frame in enumerate(frames):
        real_frame = comparison.real[..., frame]
        predicted_frame = comparison.predicted[..., frame]
        images = (
            _plane(real_frame, comparison.mask, "axial", slices),
            _plane(predicted_frame, comparison.mask, "axial", slices),
            _plane(
                predicted_frame - real_frame,
                comparison.mask,
                "axial",
                slices,
            ),
        )
        for row, image in enumerate(images):
            cmap = signal_cmap if row < 2 else residual_cmap
            norm = signal_norm if row < 2 else residual_norm
            rendered = axes[row, column].imshow(
                image,
                cmap=cmap,
                norm=norm,
                origin="upper",
                extent=_plane_extent_mm(
                    comparison.real.shape, comparison.affine, "axial"
                ),
                aspect="equal",
                interpolation="nearest",
            )
            if row < 2:
                signal_image = rendered
            else:
                residual_image = rendered
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_facecolor("#111111")
        axes[0, column].set_title(
            f"Frame {frame}\n{frame * comparison.tr_seconds:g} s", fontsize=10
        )
    row_labels = ("Real signal", "Predicted signal", "Predicted − real")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11)
    if signal_image is not None:
        fig.colorbar(
            signal_image,
            ax=axes[:2, :],
            location="right",
            shrink=0.82,
            label="BOLD signal (input units; shared scale)",
        )
    if residual_image is not None:
        fig.colorbar(
            residual_image,
            ax=axes[2, :],
            location="right",
            shrink=0.82,
            label="Signed residual (predicted − real)",
        )
    fig.suptitle(
        "Real and predicted 4D fMRI representative frames\n"
        f"Axial slice z-index {axial_index} · {comparison.space_description} · "
        f"TR={comparison.tr_seconds:g} s",
        fontsize=13,
    )
    return _save_figure_new(
        fig,
        out_path,
        output=_output_directory,
        dpi=160,
    )


def compute_temporal_diagnostics(comparison: ComparisonVolume) -> Dict[str, np.ndarray]:
    """Compute global curves, framewise spatial Pearson r, and masked RMSE."""
    global_real = np.empty(comparison.num_frames, dtype=np.float64)
    global_predicted = np.empty(comparison.num_frames, dtype=np.float64)
    spatial_correlation = np.empty(comparison.num_frames, dtype=np.float64)
    rmse = np.empty(comparison.num_frames, dtype=np.float64)
    for frame in range(comparison.num_frames):
        real_values = np.asarray(comparison.real[..., frame][comparison.mask], dtype=np.float64)
        predicted_values = np.asarray(
            comparison.predicted[..., frame][comparison.mask], dtype=np.float64
        )
        global_real[frame] = real_values.mean()
        global_predicted[frame] = predicted_values.mean()
        real_centered = real_values - global_real[frame]
        predicted_centered = predicted_values - global_predicted[frame]
        denominator = float(
            np.sqrt(np.dot(real_centered, real_centered) * np.dot(
                predicted_centered, predicted_centered
            ))
        )
        spatial_correlation[frame] = (
            float(np.dot(real_centered, predicted_centered) / denominator)
            if denominator > 1e-12
            else np.nan
        )
        rmse[frame] = float(np.sqrt(np.mean((predicted_values - real_values) ** 2)))
    return {
        "time_seconds": np.arange(comparison.num_frames, dtype=np.float64)
        * comparison.tr_seconds,
        "global_real": global_real,
        "global_predicted": global_predicted,
        "spatial_correlation": spatial_correlation,
        "rmse": rmse,
    }


def plot_temporal_diagnostics(
    comparison: ComparisonVolume,
    out_path: str | Path,
    *,
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Save temporal curves that retain the 4D comparison over all frames."""
    out_path = Path(out_path)
    values = diagnostics or compute_temporal_diagnostics(comparison)
    time = values["time_seconds"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 9.0),
        sharex=True,
        facecolor="white",
        constrained_layout=True,
    )
    axes[0].plot(time, values["global_real"], color="#225ea8", lw=1.8, label="Real")
    axes[0].plot(
        time,
        values["global_predicted"],
        color="#e67e22",
        lw=1.6,
        ls="--",
        label="Predicted",
    )
    axes[0].set_ylabel("Mean BOLD signal (input units)")
    axes[0].set_title("Brain-masked global signal")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    finite_correlations = np.isfinite(values["spatial_correlation"])
    mean_correlation = (
        float(np.mean(values["spatial_correlation"][finite_correlations]))
        if bool(finite_correlations.any())
        else float("nan")
    )
    axes[1].plot(time, values["spatial_correlation"], color="#7a5195", lw=1.6)
    axes[1].axhline(mean_correlation, color="#444444", lw=1.0, ls=":")
    # Leave a small visual margin so a perfect r=1 trace is not clipped into
    # the top axis boundary.
    axes[1].set_ylim(-1.02, 1.02)
    axes[1].set_ylabel("Pearson r")
    mean_correlation_text = (
        f"{mean_correlation:.3f}" if np.isfinite(mean_correlation) else "undefined"
    )
    undefined_count = int((~finite_correlations).sum())
    axes[1].set_title(
        "Framewise real–predicted spatial correlation "
        f"(mean r={mean_correlation_text}; undefined frames={undefined_count})"
    )

    mean_rmse = float(values["rmse"].mean())
    axes[2].plot(time, values["rmse"], color="#d95f02", lw=1.6)
    axes[2].axhline(mean_rmse, color="#444444", lw=1.0, ls=":")
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_ylabel("RMSE")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title(f"Brain-masked framewise RMSE (mean={mean_rmse:.3f})")
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Real and predicted 4D fMRI temporal diagnostics\n"
        f"{comparison.space_description} · TR={comparison.tr_seconds:g} s · "
        f"mask: {comparison.mask_description}",
        fontsize=13,
    )
    return _save_figure_new(
        fig,
        out_path,
        output=_output_directory,
        dpi=160,
    )


def plot_spatial_detail_comparison(
    comparison: ComparisonVolume,
    out_path: str | Path,
    *,
    sigma_voxels: float = 1.0,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Render static and frame-varying high-pass detail without changing data.

    The top three rows use the temporal mean, matching the implementation QA
    ``spatial_high_frequency_correlation`` gate.  The bottom three rows use the
    fixed midpoint frame after removing each volume's temporal mean, exposing
    the frame-varying detail that a static texture overlay can conceal.  The
    real and predicted panels in each section share one symmetric scale; each
    signed residual has its own symmetric scale.
    """
    sigma = float(sigma_voxels)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("detail sigma must be positive and finite")
    out_path = Path(out_path)
    real_mean = comparison.real.mean(axis=-1, dtype=np.float64)
    predicted_mean = comparison.predicted.mean(axis=-1, dtype=np.float64)
    real_detail, interior = mask_normalized_high_pass(
        real_mean, comparison.mask, sigma
    )
    predicted_detail, predicted_interior = mask_normalized_high_pass(
        predicted_mean, comparison.mask, sigma
    )
    if not np.array_equal(interior, predicted_interior):
        raise RuntimeError("paired high-pass interior supports diverged")
    detail_residual = predicted_detail - real_detail
    dynamic_frame = comparison.num_frames // 2
    real_dynamic_detail, real_dynamic_interior = mask_normalized_high_pass(
        np.asarray(comparison.real[..., dynamic_frame], dtype=np.float64)
        - real_mean,
        comparison.mask,
        sigma,
    )
    predicted_dynamic_detail, predicted_dynamic_interior = (
        mask_normalized_high_pass(
            np.asarray(comparison.predicted[..., dynamic_frame], dtype=np.float64)
            - predicted_mean,
            comparison.mask,
            sigma,
        )
    )
    if not (
        np.array_equal(interior, real_dynamic_interior)
        and np.array_equal(interior, predicted_dynamic_interior)
    ):
        raise RuntimeError("static and dynamic high-pass supports diverged")
    dynamic_residual = predicted_dynamic_detail - real_dynamic_detail
    real_values = real_detail[interior]
    predicted_values = predicted_detail[interior]
    real_centered = real_values - real_values.mean()
    predicted_centered = predicted_values - predicted_values.mean()
    denominator = float(
        np.linalg.norm(real_centered) * np.linalg.norm(predicted_centered)
    )
    correlation = (
        float(np.dot(real_centered, predicted_centered) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    real_dynamic_values = real_dynamic_detail[interior]
    predicted_dynamic_values = predicted_dynamic_detail[interior]
    real_dynamic_centered = real_dynamic_values - real_dynamic_values.mean()
    predicted_dynamic_centered = (
        predicted_dynamic_values - predicted_dynamic_values.mean()
    )
    dynamic_denominator = float(
        np.linalg.norm(real_dynamic_centered)
        * np.linalg.norm(predicted_dynamic_centered)
    )
    dynamic_correlation = (
        float(
            np.dot(real_dynamic_centered, predicted_dynamic_centered)
            / dynamic_denominator
        )
        if dynamic_denominator > 1e-12
        else float("nan")
    )
    real_dynamic_rms = float(np.sqrt(np.mean(real_dynamic_values**2)))
    predicted_dynamic_rms = float(np.sqrt(np.mean(predicted_dynamic_values**2)))
    dynamic_rms_ratio = (
        predicted_dynamic_rms / real_dynamic_rms
        if real_dynamic_rms > 1e-12
        else float("nan")
    )
    shared_limit = float(
        np.percentile(
            np.abs(np.concatenate((real_values, predicted_values))),
            ROBUST_PERCENTILE,
        )
    )
    residual_limit = float(
        np.percentile(np.abs(detail_residual[interior]), ROBUST_PERCENTILE)
    )
    dynamic_shared_limit = float(
        np.percentile(
            np.abs(
                np.concatenate((real_dynamic_values, predicted_dynamic_values))
            ),
            ROBUST_PERCENTILE,
        )
    )
    dynamic_residual_limit = float(
        np.percentile(np.abs(dynamic_residual[interior]), ROBUST_PERCENTILE)
    )
    if not np.isfinite(shared_limit) or shared_limit <= 1e-12:
        shared_limit = 1.0
    if not np.isfinite(residual_limit) or residual_limit <= 1e-12:
        residual_limit = max(shared_limit * 0.05, 1e-3)
    if not np.isfinite(dynamic_shared_limit) or dynamic_shared_limit <= 1e-12:
        dynamic_shared_limit = 1.0
    if (
        not np.isfinite(dynamic_residual_limit)
        or dynamic_residual_limit <= 1e-12
    ):
        dynamic_residual_limit = max(dynamic_shared_limit * 0.05, 1e-3)
    detail_norm = TwoSlopeNorm(
        vmin=-shared_limit, vcenter=0.0, vmax=shared_limit
    )
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    dynamic_norm = TwoSlopeNorm(
        vmin=-dynamic_shared_limit,
        vcenter=0.0,
        vmax=dynamic_shared_limit,
    )
    dynamic_residual_norm = TwoSlopeNorm(
        vmin=-dynamic_residual_limit,
        vcenter=0.0,
        vmax=dynamic_residual_limit,
    )
    cmap = _residual_colormap()
    # Keep the rendered diagnostic on the exact same conservative support used
    # for its scale and correlation.  Showing the one-voxel rim would reintroduce
    # boundary values that the honest quantitative comparison deliberately
    # excludes.
    slices = select_display_slices(interior)
    planes = ("axial", "sagittal", "coronal")
    volumes = (
        real_detail,
        predicted_detail,
        detail_residual,
        real_dynamic_detail,
        predicted_dynamic_detail,
        dynamic_residual,
    )
    row_labels = (
        "Real temporal-mean detail",
        "Predicted temporal-mean detail",
        "Predicted − real detail",
        f"Real dynamic detail (frame {dynamic_frame})",
        f"Predicted dynamic detail (frame {dynamic_frame})",
        "Predicted − real dynamic detail",
    )
    fig, axes = plt.subplots(
        6,
        3,
        figsize=(11.8, 17.2),
        facecolor="white",
        constrained_layout=True,
    )
    artists = []
    for row, volume in enumerate(volumes):
        row_artists = []
        for column, plane_name in enumerate(planes):
            artist = axes[row, column].imshow(
                _plane(volume, interior, plane_name, slices),
                cmap=cmap,
                norm=(
                    detail_norm
                    if row < 2
                    else (
                        residual_norm
                        if row == 2
                        else dynamic_norm if row < 5 else dynamic_residual_norm
                    )
                ),
                origin="upper",
                extent=_plane_extent_mm(
                    comparison.real.shape, comparison.affine, plane_name
                ),
                aspect="equal",
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_facecolor("#111111")
            if row in (0, 3):
                axes[row, column].set_title(
                    (
                        f"{plane_name.capitalize()} · index {slices[plane_name]}"
                        if row == 0
                        else plane_name.capitalize()
                    )
                )
            row_artists.append(artist)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=10)
        artists.append(row_artists)
    fig.colorbar(
        artists[0][-1],
        ax=axes[:2, :],
        location="right",
        shrink=0.84,
        label="High-pass detail (input units; shared real/predicted scale)",
    )
    fig.colorbar(
        artists[2][-1],
        ax=axes[2, :],
        location="right",
        shrink=0.84,
        label="Signed high-pass detail residual",
    )
    fig.colorbar(
        artists[3][-1],
        ax=axes[3:5, :],
        location="right",
        shrink=0.84,
        label=(
            "Temporally demeaned high-pass detail "
            "(input units; shared real/predicted scale)"
        ),
    )
    fig.colorbar(
        artists[5][-1],
        ax=axes[5, :],
        location="right",
        shrink=0.84,
        label="Signed dynamic high-pass residual",
    )
    correlation_text = "undefined" if not np.isfinite(correlation) else f"{correlation:.3f}"
    dynamic_correlation_text = (
        "undefined"
        if not np.isfinite(dynamic_correlation)
        else f"{dynamic_correlation:.3f}"
    )
    dynamic_ratio_text = (
        "undefined"
        if not np.isfinite(dynamic_rms_ratio)
        else f"{dynamic_rms_ratio:.3f}"
    )
    fig.suptitle(
        "Static and frame-varying spatial-detail audit "
        "(diagnostic high-pass; raw images unchanged)\n"
        f"Gaussian sigma={sigma:g} voxel · temporal-mean r={correlation_text} · "
        f"fixed midpoint frame {dynamic_frame} dynamic r={dynamic_correlation_text}, "
        f"RMS ratio={dynamic_ratio_text} · "
        f"{comparison.space_description}",
        fontsize=13,
    )
    return _save_figure_new(
        fig,
        out_path,
        output=_output_directory,
        dpi=160,
    )


def _maximum_projection(volume: np.ndarray, plane_name: str) -> np.ndarray:
    if plane_name == "axial":
        projection = np.max(volume, axis=2)
    elif plane_name == "sagittal":
        projection = np.max(volume, axis=0)
    elif plane_name == "coronal":
        projection = np.max(volume, axis=1)
    else:
        raise ValueError(f"unknown anatomical plane {plane_name!r}")
    return np.rot90(np.asarray(projection))


def compute_outside_mask_diagnostics(
    comparison: ComparisonVolume,
) -> Dict[str, float]:
    """Summarize prediction content hidden by a conventional brain-mask view."""
    outside = np.logical_not(comparison.mask)
    real_rms = np.sqrt(
        np.mean(np.square(comparison.real, dtype=np.float64), axis=-1)
    )
    reference = float(np.percentile(real_rms[comparison.mask], 99.0))
    if not np.isfinite(reference) or reference <= 1e-12:
        raise ValueError("real in-mask RMS q99 is unavailable for leakage display")
    if not bool(outside.any()):
        return {
            "outside_mask_voxels": 0.0,
            "reference_real_in_mask_rms_q99": reference,
            "real_outside_mask_max_abs": 0.0,
            "predicted_outside_mask_max_abs": 0.0,
            "predicted_outside_mask_rms": 0.0,
            "predicted_outside_mask_energy_fraction": 0.0,
            "predicted_outside_mask_leakage_ratio": 0.0,
        }
    real_outside_values = np.asarray(comparison.real[outside], dtype=np.float64)
    outside_values = np.asarray(comparison.predicted[outside], dtype=np.float64)
    inside_values = np.asarray(comparison.predicted[comparison.mask], dtype=np.float64)
    outside_energy = float(np.sum(outside_values * outside_values))
    inside_energy = float(np.sum(inside_values * inside_values))
    total_energy = outside_energy + inside_energy
    outside_max = float(np.max(np.abs(outside_values)))
    return {
        "outside_mask_voxels": float(outside.sum()),
        "reference_real_in_mask_rms_q99": reference,
        "real_outside_mask_max_abs": float(np.max(np.abs(real_outside_values))),
        "predicted_outside_mask_max_abs": outside_max,
        "predicted_outside_mask_rms": float(
            np.sqrt(outside_energy / outside_values.size)
        ),
        "predicted_outside_mask_energy_fraction": (
            float(outside_energy / total_energy) if total_energy > 1e-12 else 0.0
        ),
        "predicted_outside_mask_leakage_ratio": outside_max / reference,
    }


def plot_outside_mask_leakage(
    comparison: ComparisonVolume,
    out_path: str | Path,
    *,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Render whole-FOV maximum projections so exterior artifacts stay visible."""
    out_path = Path(out_path)
    diagnostics = compute_outside_mask_diagnostics(comparison)
    outside = np.logical_not(comparison.mask)
    maximum_over_time = np.max(np.abs(comparison.predicted), axis=-1)
    exterior = np.where(outside, maximum_over_time, 0.0)
    support = comparison.mask.astype(np.float32)
    limit = float(diagnostics["predicted_outside_mask_max_abs"])
    if not np.isfinite(limit) or limit <= 1e-12:
        limit = max(
            float(diagnostics["reference_real_in_mask_rms_q99"]) * 0.01,
            1e-6,
        )
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#111111")
    norm = Normalize(vmin=0.0, vmax=limit)
    planes = ("axial", "sagittal", "coronal")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.3),
        facecolor="white",
        constrained_layout=True,
    )
    artist = None
    for axis, plane_name in zip(axes, planes):
        extent = _plane_extent_mm(comparison.real.shape, comparison.affine, plane_name)
        artist = axis.imshow(
            _maximum_projection(exterior, plane_name),
            cmap=cmap,
            norm=norm,
            origin="upper",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
        )
        axis.contour(
            _maximum_projection(support, plane_name),
            levels=[0.5],
            colors=["#4dd0e1"],
            linewidths=0.9,
            origin="upper",
            extent=extent,
        )
        axis.set_title(f"{plane_name.capitalize()} maximum projection")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#111111")
    assert artist is not None
    fig.colorbar(
        artist,
        ax=axes,
        location="right",
        shrink=0.82,
        label="Maximum |predicted signal| outside supplied mask",
    )
    fig.suptitle(
        "Whole-FOV outside-mask artifact audit (cyan = supplied brain support)\n"
        f"predicted exterior max={diagnostics['predicted_outside_mask_max_abs']:.6g} · "
        f"real exterior max={diagnostics['real_outside_mask_max_abs']:.6g} · "
        "max / real in-mask RMS q99="
        f"{diagnostics['predicted_outside_mask_leakage_ratio']:.6g} · "
        f"energy fraction={diagnostics['predicted_outside_mask_energy_fraction']:.6g}",
        fontsize=12.5,
    )
    return _save_figure_new(
        fig,
        out_path,
        output=_output_directory,
        dpi=160,
    )


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if int(finite.sum()) < 2:
        return float("nan")
    left_values = left_values[finite]
    right_values = right_values[finite]
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    return (
        float(np.dot(left_values, right_values) / denominator)
        if denominator > 1e-12
        else float("nan")
    )


def compute_structured_dynamics(
    comparison: ComparisonVolume,
) -> Dict[str, np.ndarray | float]:
    """Compute exact-label ROI FC and normalized non-DC spectra for display."""
    if comparison.roi_labels is None:
        raise ValueError("structured dynamics require an exact-grid ROI label map")
    label_ids = np.unique(
        comparison.roi_labels[comparison.roi_labels > 0]
    ).astype(np.int32)
    if label_ids.size < 2:
        raise ValueError("structured dynamics require at least two nonempty ROIs")
    real_series = []
    predicted_series = []
    for label in label_ids:
        roi = comparison.mask & (comparison.roi_labels == label)
        if not bool(roi.any()):
            raise ValueError(
                "every configured ROI must have non-empty target-validity "
                f"support; missing label {int(label)}"
            )
        real_series.append(
            np.mean(np.asarray(comparison.real[roi], dtype=np.float64), axis=0)
        )
        predicted_series.append(
            np.mean(np.asarray(comparison.predicted[roi], dtype=np.float64), axis=0)
        )
    real_centered = np.stack(real_series) - np.mean(real_series, axis=1)[:, None]
    predicted_centered = np.stack(predicted_series) - np.mean(
        predicted_series, axis=1
    )[:, None]

    def correlation_matrix(centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(centered, axis=1)
        valid = norms > 1e-12
        unit = np.zeros_like(centered)
        unit[valid] = centered[valid] / norms[valid, None]
        # Keep this small deterministic ROI contraction on NumPy's scalar
        # einsum path.  On macOS, Accelerate's BLAS dispatcher can emit
        # spurious overflow/invalid warnings for otherwise finite, normalized
        # matrices (the same platform issue guarded against in eval.quality).
        matrix = np.einsum("rt,st->rs", unit, unit, optimize=False)
        matrix[~valid, :] = np.nan
        matrix[:, ~valid] = np.nan
        return matrix, valid

    real_fc, real_valid = correlation_matrix(real_centered)
    predicted_fc, predicted_valid = correlation_matrix(predicted_centered)
    triangle = np.triu_indices(label_ids.size, k=1)
    fc_correlation = (
        _finite_pearson(real_fc[triangle], predicted_fc[triangle])
        if bool(real_valid.all() and predicted_valid.all())
        else float("nan")
    )

    real_power = np.abs(np.fft.rfft(real_centered, axis=1)[:, 1:]) ** 2
    predicted_power = np.abs(np.fft.rfft(predicted_centered, axis=1)[:, 1:]) ** 2
    real_power_sum = real_power.sum(axis=1, keepdims=True)
    predicted_power_sum = predicted_power.sum(axis=1, keepdims=True)
    real_spectral_valid = real_power_sum[:, 0] > 1e-12
    predicted_spectral_valid = predicted_power_sum[:, 0] > 1e-12
    real_normalized_power = np.full_like(real_power, np.nan)
    predicted_normalized_power = np.full_like(predicted_power, np.nan)
    real_normalized_power[real_spectral_valid] = (
        real_power[real_spectral_valid] / real_power_sum[real_spectral_valid]
    )
    predicted_normalized_power[predicted_spectral_valid] = (
        predicted_power[predicted_spectral_valid]
        / predicted_power_sum[predicted_spectral_valid]
    )
    spectrum_correlation = (
        _finite_pearson(real_normalized_power, predicted_normalized_power)
        if bool(real_spectral_valid.all() and predicted_spectral_valid.all())
        else float("nan")
    )
    frequencies = np.fft.rfftfreq(
        comparison.num_frames, d=comparison.tr_seconds
    )[1:]
    return {
        "label_ids": label_ids,
        "real_fc": real_fc,
        "predicted_fc": predicted_fc,
        "fc_difference": predicted_fc - real_fc,
        "real_fc_valid": real_valid,
        "predicted_fc_valid": predicted_valid,
        "fc_correlation": fc_correlation,
        "frequencies_hz": frequencies,
        "real_normalized_power": real_normalized_power,
        "predicted_normalized_power": predicted_normalized_power,
        "spectrum_correlation": spectrum_correlation,
    }


def plot_structured_dynamics(
    comparison: ComparisonVolume,
    out_path: str | Path,
    *,
    diagnostics: Optional[Dict[str, np.ndarray | float]] = None,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Render matched-scale ROI-FC matrices and normalized ROI spectra."""
    out_path = Path(out_path)
    values = diagnostics or compute_structured_dynamics(comparison)
    labels = np.asarray(values["label_ids"], dtype=np.int32)
    frequencies = np.asarray(values["frequencies_hz"], dtype=np.float64)
    real_power = np.asarray(values["real_normalized_power"], dtype=np.float64)
    predicted_power = np.asarray(
        values["predicted_normalized_power"], dtype=np.float64
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 10.0),
        facecolor="white",
        constrained_layout=True,
    )
    fc_cmap = plt.colormaps["coolwarm"].copy()
    fc_cmap.set_bad("#333333")
    difference_cmap = plt.colormaps["PuOr"].copy()
    difference_cmap.set_bad("#333333")
    fc_norm = Normalize(vmin=-1.0, vmax=1.0)
    difference_norm = Normalize(vmin=-2.0, vmax=2.0)
    fc_arrays = (
        np.asarray(values["real_fc"], dtype=np.float64),
        np.asarray(values["predicted_fc"], dtype=np.float64),
        np.asarray(values["fc_difference"], dtype=np.float64),
    )
    fc_titles = (
        "Real ROI functional connectivity",
        "Predicted ROI functional connectivity",
        "Predicted − real ROI FC",
    )
    fc_artists = []
    tick_step = max(1, int(np.ceil(labels.size / 16)))
    ticks = np.arange(0, labels.size, tick_step)
    for column, (matrix, title) in enumerate(zip(fc_arrays, fc_titles)):
        artist = axes[0, column].imshow(
            matrix,
            cmap=fc_cmap if column < 2 else difference_cmap,
            norm=fc_norm if column < 2 else difference_norm,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )
        axes[0, column].set_title(title)
        axes[0, column].set_xticks(ticks, labels[ticks], rotation=90, fontsize=7)
        axes[0, column].set_yticks(ticks, labels[ticks], fontsize=7)
        axes[0, column].set_xlabel("ROI label")
        axes[0, column].set_ylabel("ROI label")
        fc_artists.append(artist)
    fig.colorbar(
        fc_artists[0],
        ax=axes[0, :2],
        location="right",
        shrink=0.78,
        label="Pearson correlation (shared [−1, 1] scale)",
    )
    fig.colorbar(
        fc_artists[2],
        ax=axes[0, 2],
        location="right",
        shrink=0.78,
        label="FC difference (fixed [−2, 2] scale)",
    )

    real_aggregate = np.nanmean(real_power, axis=0)
    predicted_aggregate = np.nanmean(predicted_power, axis=0)
    axes[1, 0].plot(
        frequencies, real_aggregate, color="#225ea8", lw=1.8, label="Real"
    )
    axes[1, 0].plot(
        frequencies,
        predicted_aggregate,
        color="#e67e22",
        lw=1.6,
        ls="--",
        label="Predicted",
    )
    axes[1, 0].set_xlim(float(frequencies[0]), float(frequencies[-1]))
    aggregate_limit = float(
        np.nanmax(np.concatenate((real_aggregate, predicted_aggregate)))
    )
    axes[1, 0].set_ylim(0.0, max(aggregate_limit * 1.05, 1e-6))
    axes[1, 0].set_title("Mean of per-ROI normalized non-DC spectra")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Normalized power")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", color="#dddddd", lw=0.7)
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    finite_power = np.concatenate(
        (real_power[np.isfinite(real_power)], predicted_power[np.isfinite(predicted_power)])
    )
    power_limit = float(finite_power.max()) if finite_power.size else 1.0
    power_norm = Normalize(vmin=0.0, vmax=max(power_limit, 1e-12))
    power_cmap = plt.colormaps["viridis"].copy()
    power_cmap.set_bad("#333333")
    spectral_artists = []
    for column, (power, title) in enumerate(
        (
            (real_power, "Real per-ROI normalized spectrum"),
            (predicted_power, "Predicted per-ROI normalized spectrum"),
        ),
        start=1,
    ):
        artist = axes[1, column].imshow(
            power,
            cmap=power_cmap,
            norm=power_norm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                float(frequencies[0]),
                float(frequencies[-1]),
                -0.5,
                labels.size - 0.5,
            ),
        )
        axes[1, column].set_xlim(float(frequencies[0]), float(frequencies[-1]))
        axes[1, column].set_ylim(-0.5, labels.size - 0.5)
        axes[1, column].set_yticks(ticks, labels[ticks], fontsize=7)
        axes[1, column].set_title(title)
        axes[1, column].set_xlabel("Frequency (Hz)")
        axes[1, column].set_ylabel("ROI label")
        spectral_artists.append(artist)
    fig.colorbar(
        spectral_artists[-1],
        ax=axes[1, 1:],
        location="right",
        shrink=0.78,
        label="Per-ROI normalized non-DC power (shared scale)",
    )
    fc_text = (
        f"{float(values['fc_correlation']):.3f}"
        if np.isfinite(float(values["fc_correlation"]))
        else "undefined"
    )
    spectrum_text = (
        f"{float(values['spectrum_correlation']):.3f}"
        if np.isfinite(float(values["spectrum_correlation"]))
        else "undefined"
    )
    fig.suptitle(
        "Anatomical structured-dynamics audit\n"
        f"ROI labels: {comparison.roi_description} · regions={labels.size} · "
        f"FC upper-triangle r={fc_text} · normalized-spectrum r={spectrum_text} · "
        f"TR={comparison.tr_seconds:g} s",
        fontsize=13,
    )
    return _save_figure_new(
        fig,
        out_path,
        output=_output_directory,
        dpi=160,
    )


class _BytesFFMpegWriter(mpl_animation.FFMpegWriter):
    """FFmpeg writer whose MP4 exists only on an anonymous held descriptor."""

    payload: bytes

    def setup(self, fig, outfile, dpi=None):
        del outfile
        self._output_stream = tempfile.TemporaryFile(mode="w+b")
        try:
            super().setup(fig, "pipe:1", dpi=dpi)
        except Exception:
            self._output_stream.close()
            raise

    def _run(self):
        self._proc = subprocess.Popen(
            self._args(),
            stdin=subprocess.PIPE,
            stdout=self._output_stream,
            stderr=subprocess.PIPE,
        )

    def finish(self):
        _output, error = self._proc.communicate()
        try:
            if self._proc.returncode:
                detail = error.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"ffmpeg exited {self._proc.returncode}: {detail}"
                )
            self._output_stream.flush()
            self._output_stream.seek(0)
            self.payload = self._output_stream.read()
            if not self.payload:
                raise RuntimeError("ffmpeg produced an empty MP4")
        finally:
            self._output_stream.close()


class _BytesPillowWriter(mpl_animation.PillowWriter):
    """Pillow writer that serializes GIF frames directly into memory."""

    payload: bytes

    def setup(self, fig, outfile, dpi=None):
        del outfile
        self.fig = fig
        self.dpi = fig.dpi if dpi is None else dpi
        self._frames = []

    def finish(self):
        if not self._frames:
            raise RuntimeError("Pillow produced no animation frames")
        buffer = io.BytesIO()
        self._frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
        )
        self.payload = buffer.getvalue()
        if not self.payload:
            raise RuntimeError("Pillow produced an empty GIF")


def _render_animation_payload(
    animation: mpl_animation.FuncAnimation,
    *,
    writer: _BytesFFMpegWriter | _BytesPillowWriter,
    dpi: int,
) -> bytes:
    """Render without any named output path and return encoded bytes."""
    animation.save("descriptor-only-animation", writer=writer, dpi=dpi)
    return writer.payload


def _save_animation_with_fallback(
    animation: mpl_animation.FuncAnimation,
    output_base: Path,
    *,
    fps: int,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Write a new durable MP4 through ffmpeg, falling back to a new GIF."""
    owned = _output_directory is None
    output = _output_directory or _open_output_directory(output_base.parent)
    output_base = output.path / output_base.name
    _validate_output_path(output_base, output)
    ffmpeg = shutil.which("ffmpeg")
    try:
        if ffmpeg is not None:
            mp4_path = output_base.with_suffix(".mp4")
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
            writer = _BytesFFMpegWriter(
                fps=fps,
                codec="libx264",
                bitrate=2400,
                # libx264 with yuv420p requires even pixel dimensions. Matplotlib's
                # constrained layout can round an otherwise even figure height to
                # an odd raster size, so pad by at most one pixel instead of
                # changing or cropping the anatomical panels.
                extra_args=[
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                    "-f",
                    "mp4",
                ],
            )
            try:
                payload = _render_animation_payload(
                    animation,
                    writer=writer,
                    dpi=100,
                )
            except Exception as exc:
                warnings.warn(
                    f"ffmpeg MP4 encoding failed ({exc}); falling back to GIF",
                    RuntimeWarning,
                )
            else:
                _write_output_new(mp4_path, payload, output)
                return mp4_path

        gif_path = output_base.with_suffix(".gif")
        payload = _render_animation_payload(
            animation,
            writer=_BytesPillowWriter(fps=fps),
            dpi=85,
        )
        _write_output_new(gif_path, payload, output)
        return gif_path
    finally:
        if owned:
            os.close(output.descriptor)


def save_comparison_animation(
    comparison: ComparisonVolume,
    output_base: str | Path,
    *,
    fps: int = 8,
    scale_limits: Optional[Tuple[float, float]] = None,
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
    _output_directory: _OutputDirectory | None = None,
) -> Path:
    """Save a fixed-scale tri-planar real/predicted/residual animation."""
    if isinstance(fps, bool) or int(fps) != fps or int(fps) < 1:
        raise ValueError("animation fps must be a positive integer")
    fps = int(fps)
    signal_limit, residual_limit = scale_limits or compute_scale_limits(comparison)
    values = diagnostics or compute_temporal_diagnostics(comparison)
    signal_cmap, signal_norm = _comparison_rendering(comparison, signal_limit)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    residual_cmap = _residual_colormap()
    slices = select_display_slices(comparison.mask)
    planes = ("axial", "sagittal", "coronal")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.8, 9.2),
        facecolor="white",
        constrained_layout=True,
    )
    image_artists = []
    for row in range(3):
        row_artists = []
        for column, plane_name in enumerate(planes):
            real_frame = comparison.real[..., 0]
            predicted_frame = comparison.predicted[..., 0]
            volumes = (real_frame, predicted_frame, predicted_frame - real_frame)
            cmap = signal_cmap if row < 2 else residual_cmap
            norm = signal_norm if row < 2 else residual_norm
            artist = axes[row, column].imshow(
                _plane(volumes[row], comparison.mask, plane_name, slices),
                cmap=cmap,
                norm=norm,
                origin="upper",
                extent=_plane_extent_mm(
                    comparison.real.shape, comparison.affine, plane_name
                ),
                aspect="equal",
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_facecolor("#111111")
            if row == 0:
                axis_letter = {"axial": "z", "sagittal": "x", "coronal": "y"}[
                    plane_name
                ]
                axes[row, column].set_title(
                    f"{plane_name.capitalize()} ({axis_letter}={slices[plane_name]})"
                )
            row_artists.append(artist)
        image_artists.append(row_artists)
    for row, label in enumerate(("Real", "Predicted", "Predicted − real")):
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.colorbar(
        image_artists[0][-1],
        ax=axes[:2, :],
        location="right",
        shrink=0.84,
        label="BOLD signal (input units; shared scale)",
    )
    fig.colorbar(
        image_artists[2][-1],
        ax=axes[2, :],
        location="right",
        shrink=0.84,
        label="Signed residual",
    )
    status = fig.suptitle("", fontsize=13)

    def update(frame: int):
        real_frame = comparison.real[..., frame]
        predicted_frame = comparison.predicted[..., frame]
        volumes = (real_frame, predicted_frame, predicted_frame - real_frame)
        for row in range(3):
            for column, plane_name in enumerate(planes):
                image_artists[row][column].set_data(
                    _plane(volumes[row], comparison.mask, plane_name, slices)
                )
        frame_correlation = float(values["spatial_correlation"][frame])
        correlation_text = (
            f"{frame_correlation:.3f}"
            if np.isfinite(frame_correlation)
            else "undefined"
        )
        status.set_text(
            "Real and predicted 4D fMRI · "
            f"frame {frame + 1}/{comparison.num_frames} · "
            f"t={frame * comparison.tr_seconds:g} s · "
            f"spatial r={correlation_text}\n"
            f"{comparison.space_description}; fixed scales across all frames"
        )
        return [
            artist for row_artists in image_artists for artist in row_artists
        ] + [status]

    movie = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=comparison.num_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    try:
        return _save_animation_with_fallback(
            movie,
            Path(output_base),
            fps=fps,
            _output_directory=_output_directory,
        )
    finally:
        plt.close(fig)


def _validated_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("output prefix must be a non-empty filename component")
    return value


def _generate_comparison_outputs_open(
    real_path: str | Path,
    predicted_path: str | Path,
    out_dir: str | Path,
    *,
    mask_path: str | Path,
    roi_labels_path: Optional[str | Path] = None,
    tr_seconds: Optional[float] = None,
    prefix: Optional[str] = None,
    fps: int = 8,
    _output_directory: _OutputDirectory,
) -> Dict[str, Path]:
    """Generate outputs while the exact publication directory remains held."""
    comparison = load_comparison(
        real_path,
        predicted_path,
        mask_path=mask_path,
        roi_labels_path=roi_labels_path,
        tr_seconds=tr_seconds,
    )
    out_dir = _output_directory.path
    if prefix is None:
        prefix = _nifti_stem(Path(predicted_path))
        if prefix.endswith("_synthetic"):
            prefix = prefix[: -len("_synthetic")]
    prefix = _validated_prefix(prefix)
    limits = compute_scale_limits(comparison)
    diagnostics = compute_temporal_diagnostics(comparison)
    montage = plot_frame_montage(
        comparison,
        out_dir / f"{prefix}_frames_montage.png",
        scale_limits=limits,
        _output_directory=_output_directory,
    )
    temporal = plot_temporal_diagnostics(
        comparison,
        out_dir / f"{prefix}_temporal_diagnostics.png",
        diagnostics=diagnostics,
        _output_directory=_output_directory,
    )
    spatial_detail = plot_spatial_detail_comparison(
        comparison,
        out_dir / f"{prefix}_spatial_detail.png",
        _output_directory=_output_directory,
    )
    outside_mask = plot_outside_mask_leakage(
        comparison,
        out_dir / f"{prefix}_outside_mask_leakage.png",
        _output_directory=_output_directory,
    )
    structured = None
    if comparison.roi_labels is not None:
        structured = plot_structured_dynamics(
            comparison,
            out_dir / f"{prefix}_structured_dynamics.png",
            _output_directory=_output_directory,
        )
    animation = save_comparison_animation(
        comparison,
        out_dir / f"{prefix}_4d_comparison",
        fps=fps,
        scale_limits=limits,
        diagnostics=diagnostics,
        _output_directory=_output_directory,
    )
    outputs = {
        "montage": montage,
        "temporal": temporal,
        "spatial_detail": spatial_detail,
        "outside_mask": outside_mask,
        "animation": animation,
    }
    if structured is not None:
        outputs["structured_dynamics"] = structured
    input_paths = {
        "real": Path(real_path).expanduser().resolve(strict=True),
        "predicted": Path(predicted_path).expanduser().resolve(strict=True),
        "mask": (
            None
            if mask_path is None
            else Path(mask_path).expanduser().resolve(strict=True)
        ),
        "roi_labels": (
            None
            if roi_labels_path is None
            else Path(roi_labels_path).expanduser().resolve(strict=True)
        ),
    }
    structured_values = (
        None if comparison.roi_labels is None else compute_structured_dynamics(comparison)
    )
    manifest = {
        "schema": "connect4-4d-visualization-manifest-v1",
        "implementation_sources": implementation_source_records(),
        "inputs": {
            name: (
                None
                if path is None
                else {"path": str(path), "sha256": sha256_file(path)}
            )
            for name, path in input_paths.items()
        },
        "display_contract": {
            "space": comparison.space_description,
            "tr_seconds": comparison.tr_seconds,
            "shared_signal_absolute_limit": float(limits[0]),
            "shared_signal_limit_percentile": ROBUST_PERCENTILE,
            "signed_residual_absolute_limit": float(limits[1]),
            "signed_residual_limit_percentile": ROBUST_PERCENTILE,
            "fixed_scales_across_all_animation_frames": True,
            "image_interpolation": "nearest",
            "mask_description": comparison.mask_description,
            "mask_resampled": comparison.mask_resampled,
            "display_slices": select_display_slices(comparison.mask),
            "spatial_detail_diagnostic": {
                "temporal_mean_and_temporally_demeaned_detail_shown": True,
                "dynamic_frame_index": comparison.num_frames // 2,
                "dynamic_frame_selection": (
                    "fixed integer midpoint; independent of real and predicted values"
                ),
                "gaussian_sigma_voxels": 1.0,
                "shared_real_predicted_scales": True,
            },
            "outside_mask_diagnostics": compute_outside_mask_diagnostics(comparison),
            "structured_dynamics": (
                None
                if structured_values is None
                else {
                    "roi_count": int(
                        np.asarray(structured_values["label_ids"]).size
                    ),
                    "fc_correlation": (
                        float(structured_values["fc_correlation"])
                        if np.isfinite(float(structured_values["fc_correlation"]))
                        else None
                    ),
                    "power_spectrum_correlation": (
                        float(structured_values["spectrum_correlation"])
                        if np.isfinite(
                            float(structured_values["spectrum_correlation"])
                        )
                        else None
                    ),
                    "roi_labels_source": comparison.roi_description,
                }
            ),
        },
        "outputs": {
            name: _output_snapshot(Path(path), _output_directory)
            for name, path in outputs.items()
        },
    }
    manifest["record_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = out_dir / f"{prefix}_visualization_manifest.json"
    _write_output_new(
        manifest_path,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        _output_directory,
    )
    outputs["manifest"] = manifest_path
    for path in outputs.values():
        _output_snapshot(Path(path), _output_directory)
    _assert_output_directory(_output_directory)
    return outputs


def generate_comparison_outputs(
    real_path: str | Path,
    predicted_path: str | Path,
    out_dir: str | Path,
    *,
    mask_path: str | Path,
    roi_labels_path: Optional[str | Path] = None,
    tr_seconds: Optional[float] = None,
    prefix: Optional[str] = None,
    fps: int = 8,
) -> Dict[str, Path]:
    """Validate one pair and durably publish new static and animated outputs."""
    output_directory = _open_output_directory(Path(out_dir))
    try:
        return _generate_comparison_outputs_open(
            real_path,
            predicted_path,
            output_directory.path,
            mask_path=mask_path,
            roi_labels_path=roi_labels_path,
            tr_seconds=tr_seconds,
            prefix=prefix,
            fps=fps,
            _output_directory=output_directory,
        )
    finally:
        os.close(output_directory.descriptor)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create strict real-versus-predicted 4D fMRI visualisations in the "
            "existing paired image space. The fMRI volumes are never resampled."
        )
    )
    parser.add_argument("--real", required=True, help="real 4D fMRI NIfTI")
    parser.add_argument("--pred", required=True, help="predicted 4D fMRI NIfTI")
    parser.add_argument(
        "--mask",
        required=True,
        help=(
            "required independent 3D structural brain/segmentation mask; may be nearest-neighbour "
            "resampled only when its oriented world support matches the fMRI"
        ),
    )
    parser.add_argument(
        "--roi-labels",
        default=None,
        help=(
            "optional exact-grid 3D anatomical integer label map; enables "
            "matched ROI-FC and normalized spectral diagnostics"
        ),
    )
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument(
        "--prefix",
        default=None,
        help="output filename prefix (defaults to predicted NIfTI stem)",
    )
    parser.add_argument(
        "--tr-seconds",
        type=float,
        default=None,
        help=(
            "optional exact TR validation override in seconds; must match both "
            "NIfTI headers and never replaces header timing"
        ),
    )
    parser.add_argument("--fps", type=int, default=8, help="animation frames/second")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    outputs = generate_comparison_outputs(
        args.real,
        args.pred,
        args.out_dir,
        mask_path=args.mask,
        roi_labels_path=args.roi_labels,
        tr_seconds=args.tr_seconds,
        prefix=args.prefix,
        fps=args.fps,
    )
    print(f"[visualize-4d] space: {SPACE_DESCRIPTION}")
    for name, path in outputs.items():
        print(f"[visualize-4d] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
