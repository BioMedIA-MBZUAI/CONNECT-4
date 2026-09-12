"""Build a T1-only, hash-bound cohort-common 3-mm grid contract.

The source reference must be a cohort anatomical reference selected without
looking at functional data. The accompanying manifest binds every cohort T1w
input by SHA-256. Matrix dimensions are derived from the structural reference
bounds and then symmetrically padded only to the requested patch multiple.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .conform import (
    ARCHITECTURE_PADDING_SCHEMA_VERSION,
    COMMON_GRID_SCHEMA_VERSION,
    STRUCTURAL_MANIFEST_SCHEMA_VERSION,
    TARGET_VOXEL,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
)


def _edge_corners(shape: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (-0.5, float(shape[0]) - 0.5)
            for y in (-0.5, float(shape[1]) - 0.5)
            for z in (-0.5, float(shape[2]) - 0.5)
        ],
        dtype=np.float64,
    )


def _validate_structural_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid structural cohort manifest: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != STRUCTURAL_MANIFEST_SCHEMA_VERSION
        or payload.get("functional_data_used") is not False
        or not isinstance(payload.get("scans"), list)
        or not payload["scans"]
    ):
        raise ValueError("structural cohort manifest must be non-empty and T1-only")
    reference = payload.get("structural_reference")
    if (
        not isinstance(reference, dict)
        or reference.get("modality") != "T1w"
        or reference.get("cohort_derived") is not True
        or reference.get("functional_data_used") is not False
        or not str(reference.get("derivation_method", "")).strip()
        or not _is_sha256(reference.get("sha256"))
        or not str(reference.get("path", "")).strip()
    ):
        raise ValueError(
            "structural manifest requires explicit T1w cohort-reference evidence"
        )
    reference_path = Path(str(reference["path"])).expanduser()
    if not reference_path.is_absolute():
        reference_path = path.parent / reference_path
    reference_path = reference_path.resolve(strict=True)
    if _sha256_file(reference_path) != reference["sha256"]:
        raise ValueError("structural manifest reference SHA-256 mismatch")
    seen: set[str] = set()
    for row in payload["scans"]:
        if not isinstance(row, dict):
            raise ValueError("structural cohort manifest rows must be objects")
        if any(
            token in str(key).lower()
            for key in row
            for token in ("fmri", "bold", "functional")
        ):
            raise ValueError("functional fields are forbidden in a grid manifest")
        scan_id = str(row.get("scan_id", "")).strip()
        recorded = row.get("t1w_sha256")
        raw_path = str(row.get("t1w_path", "")).strip()
        if not scan_id or scan_id in seen or not _is_sha256(recorded) or not raw_path:
            raise ValueError("each structural row requires unique scan_id/path/SHA-256")
        seen.add(scan_id)
        t1_path = Path(raw_path).expanduser()
        if not t1_path.is_absolute():
            t1_path = path.parent / t1_path
        t1_path = t1_path.resolve(strict=True)
        if _sha256_file(t1_path) != recorded:
            raise ValueError(f"T1w SHA-256 mismatch for {scan_id}")
        image = nib.load(str(t1_path))
        if image.ndim != 3 or not np.isfinite(image.affine).all():
            raise ValueError(f"invalid structural image for {scan_id}")
    return payload


def build_common_grid_contract(
    structural_reference_path: str | Path,
    structural_manifest_path: str | Path,
    output_contract_path: str | Path,
    *,
    patch_multiple: Sequence[int] = (16, 16, 16),
) -> Path:
    """Create the reference NIfTI and its immutable JSON contract atomically."""
    source_path = Path(structural_reference_path).expanduser().resolve(strict=True)
    manifest_path = Path(structural_manifest_path).expanduser().resolve(strict=True)
    manifest = _validate_structural_manifest(manifest_path)
    recorded_reference = manifest["structural_reference"]
    manifest_reference_path = Path(str(recorded_reference["path"])).expanduser()
    if not manifest_reference_path.is_absolute():
        manifest_reference_path = manifest_path.parent / manifest_reference_path
    if manifest_reference_path.resolve(strict=True) != source_path or (
        recorded_reference["sha256"] != _sha256_file(source_path)
    ):
        raise ValueError(
            "--structural-reference must match the manifest's hash-bound "
            "T1w cohort reference"
        )
    multiple = tuple(int(value) for value in patch_multiple)
    if len(multiple) != 3 or any(value < 1 for value in multiple):
        raise ValueError("patch_multiple must contain three positive integers")

    source = nib.as_closest_canonical(nib.load(str(source_path)))
    if source.ndim != 3 or not np.isfinite(source.get_fdata(dtype=np.float32)).all():
        raise ValueError("structural reference must be a finite 3D NIfTI")
    world_edges = nib.affines.apply_affine(
        source.affine, _edge_corners(source.shape[:3])
    )
    minimum = world_edges.min(axis=0)
    maximum = world_edges.max(axis=0)
    spacing = float(TARGET_VOXEL[0])
    anatomical_shape = tuple(
        max(1, int(math.ceil((upper - lower) / spacing)))
        for lower, upper in zip(minimum, maximum)
    )
    anatomical_affine = np.diag((spacing, spacing, spacing, 1.0))
    anatomical_affine[:3, 3] = minimum + spacing / 2.0

    architecture_shape = tuple(
        int(math.ceil(size / patch) * patch)
        for size, patch in zip(anatomical_shape, multiple)
    )
    extra = tuple(
        architecture - anatomical
        for architecture, anatomical in zip(architecture_shape, anatomical_shape)
    )
    before = tuple(value // 2 for value in extra)
    after = tuple(value - start for value, start in zip(extra, before))
    architecture_affine = anatomical_affine.copy()
    architecture_affine[:3, 3] -= architecture_affine[:3, :3] @ np.asarray(before)

    output_contract = Path(output_contract_path).expanduser()
    output_contract.parent.mkdir(parents=True, exist_ok=True)
    if output_contract.exists():
        raise FileExistsError(f"refusing to overwrite grid contract: {output_contract}")
    reference_path = output_contract.with_name(output_contract.stem + "_reference.nii.gz")
    if reference_path.exists():
        raise FileExistsError(f"refusing to overwrite grid reference: {reference_path}")

    with tempfile.TemporaryDirectory(
        prefix=".connect4-grid-", dir=output_contract.parent
    ) as temporary:
        temporary_path = Path(temporary)
        staged_reference = temporary_path / reference_path.name
        reference = resample_from_to(
            source,
            (architecture_shape, architecture_affine),
            order=1,
            mode="constant",
            cval=0.0,
        )
        reference.header.set_xyzt_units(xyz="mm")
        reference.header.set_data_dtype(np.float32)
        nib.save(reference, staged_reference)
        os.replace(staged_reference, reference_path)

        record = {
            "schema": COMMON_GRID_SCHEMA_VERSION,
            "cohort_common": True,
            "derivation": {
                "source": "structural-reference-and-cohort-T1-manifest",
                "functional_data_used": False,
                "matrix_size_reported_by_paper": False,
                "voxel_size_mm_reported_by_paper": 3.0,
                "unpublished_recovery_choice": True,
            },
            "structural_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "scan_count": len(manifest["scans"]),
            },
            "source_structural_reference": {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "modality": "T1w",
                "cohort_derived": True,
                "derivation_method": recorded_reference["derivation_method"],
                "world_edge_minimum_mm": minimum.tolist(),
                "world_edge_maximum_mm": maximum.tolist(),
            },
            "reference": {
                "path": str(reference_path),
                "sha256": _sha256_file(reference_path),
                "affine": architecture_affine.tolist(),
            },
            "voxel_size_mm": list(TARGET_VOXEL),
            "anatomical_shape": list(anatomical_shape),
            "anatomical_affine": anatomical_affine.tolist(),
            "architecture_shape": list(architecture_shape),
            "architecture_padding": {
                "schema": ARCHITECTURE_PADDING_SCHEMA_VERSION,
                "purpose": "patch-multiple-only-not-paper-anatomy",
                "multiple": list(multiple),
                "before": list(before),
                "after": list(after),
            },
        }
        record["record_sha256"] = _canonical_sha256(record)
        staged_contract = temporary_path / output_contract.name
        staged_contract.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staged_contract, output_contract)
    return output_contract.resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-reference", required=True)
    parser.add_argument("--structural-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch-multiple", nargs=3, type=int, default=(16, 16, 16))
    args = parser.parse_args()
    result = build_common_grid_contract(
        args.structural_reference,
        args.structural_manifest,
        args.out,
        patch_multiple=args.patch_multiple,
    )
    print(result)


if __name__ == "__main__":
    main()
