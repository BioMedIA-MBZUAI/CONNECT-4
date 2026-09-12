"""Versioned spatial semantics for structural patches and graph caches.

NIfTI arrays are indexed in canonical RAS+ ``(X, Y, Z)`` order.  Integer
indices denote voxel centres.  When a grid is resized with PyTorch
``align_corners=False``, a target-grid coordinate maps back to the source as
``(target + 0.5) * source_size / target_size - 0.5``.  Keeping these details in
one hash-bound contract prevents a cache built with swapped X/Z semantics or a
corner-based coordinate convention from being reused silently.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

from architecture_contract import (
    PATCH_CENTER_CONVENTION,
    geometric_patch_center_voxel as _canonical_patch_center_voxel,
)


PATCH_COORDINATE_CONTRACT = {
    "schema": "connect4-patch-coordinate-contract-v3",
    "array_axis_order": ["X", "Y", "Z"],
    "canonical_world_orientation": "RAS+",
    "anatomical_axis_meaning": {
        "X": "left-to-right",
        "Y": "posterior-to-anterior",
        "Z": "inferior-to-superior",
    },
    "flattening_order": "C order; Z varies fastest, then Y, then X",
    "voxel_coordinate_convention": "integer indices are voxel centres",
    "patch_center_convention": PATCH_CENTER_CONVENTION,
    "resize_coordinate_mapping": (
        "source=(target+0.5)*source_size/target_size-0.5 "
        "(align_corners=False)"
    ),
}


def align_corners_false_source_index(
    target_index: Sequence[float],
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> np.ndarray:
    """Map a target voxel-centre index to the corresponding source index."""
    index = np.asarray(target_index, dtype=np.float64)
    source = np.asarray(source_shape, dtype=np.float64)
    target = np.asarray(target_shape, dtype=np.float64)
    if index.shape != source.shape or source.shape != target.shape:
        raise ValueError("index, source_shape, and target_shape must have equal rank")
    if np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("source_shape and target_shape must be positive")
    return (index + 0.5) * source / target - 0.5


def patch_grid_xyz(
    patch_index: int,
    target_shape: Sequence[int],
    patch_size: Sequence[int],
) -> tuple[int, int, int]:
    """Return the canonical ``(X, Y, Z)`` patch-grid coordinate.

    Flattening follows NumPy/PyTorch C order, so Z is the fastest-varying
    coordinate.  The names deliberately match canonical NIfTI array axes.
    """
    shape = tuple(int(value) for value in target_shape)
    patch = tuple(int(value) for value in patch_size)
    if len(shape) != 3 or len(patch) != 3:
        raise ValueError("target_shape and patch_size must contain X, Y, Z")
    if any(value <= 0 for value in shape + patch):
        raise ValueError("target_shape and patch_size entries must be positive")
    if any(size % width for size, width in zip(shape, patch)):
        raise ValueError("target_shape must be divisible by patch_size")
    nx, ny, nz = (size // width for size, width in zip(shape, patch))
    count = nx * ny * nz
    index = int(patch_index)
    if isinstance(patch_index, bool) or index != patch_index or not 0 <= index < count:
        raise IndexError(f"patch index {patch_index!r} is outside [0,{count})")
    x_index = index // (ny * nz)
    remainder = index % (ny * nz)
    y_index = remainder // nz
    z_index = remainder % nz
    return x_index, y_index, z_index


def geometric_patch_center_voxel(
    patch_xyz: Sequence[int], patch_size: Sequence[int]
) -> np.ndarray:
    """Return the geometric centre in voxel-centre coordinates.

    A two-voxel interval starts at centres 0 and 1, hence its centre is 0.5,
    not 1.0.  The explicit half-voxel term is the same centre convention used
    by ``align_corners=False``.
    """
    center = _canonical_patch_center_voxel(tuple(patch_xyz), tuple(patch_size))
    if len(center) != 3:
        raise ValueError("patch_xyz and patch_size must be XYZ triplets")
    return np.asarray(center, dtype=np.float64)


def nifti_geometry_fingerprint(path: str | Path) -> dict[str, object]:
    """Return canonical grid geometry suitable for a cache source fingerprint."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    if image.ndim != 3:
        raise ValueError(f"structural source must be 3D, got {image.shape}")
    if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError("canonical structural source is not RAS+")
    shape = tuple(int(value) for value in image.shape)
    edge_indices = np.asarray(
        list(product(*[(-0.5, float(size) - 0.5) for size in shape])),
        dtype=np.float64,
    )
    edge_world = nib.affines.apply_affine(image.affine, edge_indices)
    return {
        "shape_xyz": list(shape),
        "affine": np.asarray(image.affine, dtype=np.float64).tolist(),
        "voxel_sizes_mm_xyz": [
            float(value) for value in nib.affines.voxel_sizes(image.affine)
        ],
        "axis_codes": ["R", "A", "S"],
        "voxel_edge_bounds_mm": {
            "minimum": edge_world.min(axis=0).tolist(),
            "maximum": edge_world.max(axis=0).tolist(),
        },
    }


__all__ = [
    "PATCH_COORDINATE_CONTRACT",
    "align_corners_false_source_index",
    "geometric_patch_center_voxel",
    "nifti_geometry_fingerprint",
    "patch_grid_xyz",
]
