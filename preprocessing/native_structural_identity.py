"""Target-blind identity for the non-certified native-grid recovery profile.

The manuscript reports 3-mm output but does not specify the 61x73x61 recovery
matrix. This module never labels that matrix paper-certified. Its cohort and
per-scan authorities contain only T1/SynthSeg inputs and structural outputs.
BOLD belongs to a separate allowlisted target receipt and is never resolved,
stat'ed, hashed, or opened while structural caches are admitted.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from .conform import zscore_inbrain
from .preprocess_t1 import T1_INTENSITY_NORMALIZATION
from .source_acquisition_identity import (
    SourceAcquisitionError,
    canonical_sha256,
    snapshot_binary_artifact,
    snapshot_json_artifact,
)


NATIVE_STRUCTURAL_ALIGNMENT_SCHEMA = "connect4-native-structural-alignment-binding-v2"
NATIVE_T1_PREPROCESSING_SCHEMA = "connect4-t1-native-aligned-padding-v1"
NATIVE_SEG_PREPROCESSING_SCHEMA = "connect4-seg-native-aligned-padding-v1"
NATIVE_STRUCTURAL_SCAN_PROVENANCE_SCHEMA = "connect4-native-structural-scan-v4"
NATIVE_STRUCTURAL_BATCH_MANIFEST_SCHEMA = "connect4-native-structural-batch-v4"
NATIVE_SCAN_PROVENANCE_SCHEMA = NATIVE_STRUCTURAL_SCAN_PROVENANCE_SCHEMA
NATIVE_BATCH_MANIFEST_SCHEMA = NATIVE_STRUCTURAL_BATCH_MANIFEST_SCHEMA
NATIVE_SPATIAL_PROFILE = "non-certified-per-scan-t1-grid-identical-zero-padding-v1"
NATIVE_CERTIFICATION_STATUS = "NON_CERTIFIED_RECOVERY_PREPROCESSING"
NATIVE_STRUCTURAL_BATCH_PURPOSE = (
    "target-blind-native-structural-cohort-authority"
)
NATIVE_STRUCTURAL_SCAN_PURPOSE = "target-blind-native-structural-scan-authority"
NATIVE_SPATIAL_JOIN_SCHEMA = "connect4-native-spatial-target-join-v1"
NATIVE_SHAPE = (61, 73, 61)
ARCHITECTURE_SHAPE = (64, 80, 64)
PADDING_BEFORE = (1, 3, 1)
PADDING_AFTER = (2, 4, 2)
IDENTICALLY_PADDED_MODALITIES = ("t1w", "segmentation")


def padded_affine(native_affine: np.ndarray) -> np.ndarray:
    """Shift the origin so the unpadded crop retains exact native world space."""
    affine = np.asarray(native_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise SourceAcquisitionError("native structural affine is invalid")
    result = affine.copy()
    result[:3, 3] -= result[:3, :3] @ np.asarray(PADDING_BEFORE, dtype=np.float64)
    return result


def zero_pad_native(array: np.ndarray) -> np.ndarray:
    """Apply the one allowed 61x73x61 -> 64x80x64 spatial transform."""
    values = np.asarray(array)
    if values.shape[:3] != NATIVE_SHAPE:
        raise ValueError(f"native array must begin with shape {NATIVE_SHAPE}")
    trailing = ((0, 0),) * (values.ndim - 3)
    return np.pad(
        values,
        tuple(zip(PADDING_BEFORE, PADDING_AFTER)) + trailing,
        mode="constant",
        constant_values=0,
    )


def _signed_record(
    path: Path, *, expected_sha256: str, expected_schema: str, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    value, evidence = snapshot_json_artifact(path, label=label)
    if evidence["sha256"] != expected_sha256:
        raise SourceAcquisitionError(f"{label} SHA-256 differs")
    unsigned = dict(value)
    recorded = unsigned.pop("record_sha256", None)
    if value.get("schema") != expected_schema or recorded != canonical_sha256(unsigned):
        raise SourceAcquisitionError(f"{label} signed record differs")
    return value, evidence


def _artifact(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise SourceAcquisitionError(f"{label} artifact fields differ")
    try:
        _payload, evidence = snapshot_binary_artifact(
            Path(str(value["path"])),
            expected_sha256=str(value["sha256"]),
            label=label,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceAcquisitionError(f"{label} artifact is malformed") from exc
    if evidence != value:
        raise SourceAcquisitionError(f"{label} size/path evidence differs")
    return dict(value)


def build_native_structural_alignment_binding(
    batch_manifest_path: Path,
    *,
    expected_batch_sha256: str,
    scan_id: str,
) -> dict[str, object]:
    """Authenticate a BOLD-free native batch/scan and construct its join."""
    batch, batch_evidence = _signed_record(
        Path(batch_manifest_path),
        expected_sha256=expected_batch_sha256,
        expected_schema=NATIVE_BATCH_MANIFEST_SCHEMA,
        label="native structural batch manifest",
    )
    if set(batch) != {
        "schema",
        "purpose",
        "paper_certified",
        "certification_status",
        "spatial_profile",
        "preprocessing_contract",
        "scans",
        "record_sha256",
    }:
        raise SourceAcquisitionError(
            "native structural batch fields differ or include target data"
        )
    if (
        batch.get("purpose") != NATIVE_STRUCTURAL_BATCH_PURPOSE
        or batch.get("paper_certified") is not False
        or batch.get("certification_status") != NATIVE_CERTIFICATION_STATUS
        or batch.get("spatial_profile") != NATIVE_SPATIAL_PROFILE
    ):
        raise SourceAcquisitionError(
            "native structural batch purpose/certification differs"
        )
    contract = batch.get("preprocessing_contract")
    if (
        not isinstance(contract, dict)
        or set(contract)
        != {
            "native_shape",
            "architecture_shape",
            "voxel_size_mm",
            "padding_before",
            "padding_after",
            "padding_mode",
            "interpolation_after_native_preprocessing",
        }
        or contract.get("native_shape") != list(NATIVE_SHAPE)
        or contract.get("architecture_shape") != list(ARCHITECTURE_SHAPE)
        or float(contract.get("voxel_size_mm", 0.0)) != 3.0
        or contract.get("padding_before") != list(PADDING_BEFORE)
        or contract.get("padding_after") != list(PADDING_AFTER)
        or contract.get("padding_mode") != "constant-zero-no-interpolation"
        or contract.get("interpolation_after_native_preprocessing") is not False
    ):
        raise SourceAcquisitionError("native structural batch spatial contract differs")
    rows = batch.get("scans")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict)
            or set(row)
            != {
                "scan_id",
                "structural_provenance_path",
                "structural_provenance_sha256",
            }
            for row in rows
        )
        or [row["scan_id"] for row in rows]
        != sorted({str(row["scan_id"]) for row in rows})
    ):
        raise SourceAcquisitionError(
            "native structural batch scan rows differ or include target data"
        )
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("scan_id") == scan_id]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        raise SourceAcquisitionError(
            "native batch does not contain exactly one requested scan"
        )
    row = matches[0]
    provenance, provenance_evidence = _signed_record(
        Path(str(row.get("structural_provenance_path", ""))),
        expected_sha256=str(row.get("structural_provenance_sha256", "")),
        expected_schema=NATIVE_SCAN_PROVENANCE_SCHEMA,
        label="native structural scan provenance",
    )
    if set(provenance) != {
        "schema",
        "purpose",
        "scan_id",
        "paper_certified",
        "certification_status",
        "spatial_profile",
        "native_grid",
        "inputs",
        "outputs",
        "record_sha256",
    }:
        raise SourceAcquisitionError(
            "native structural scan fields differ or include target data"
        )
    if (
        provenance.get("scan_id") != scan_id
        or provenance.get("purpose") != NATIVE_STRUCTURAL_SCAN_PURPOSE
        or provenance.get("paper_certified") is not False
        or provenance.get("certification_status") != NATIVE_CERTIFICATION_STATUS
        or provenance.get("spatial_profile") != NATIVE_SPATIAL_PROFILE
    ):
        raise SourceAcquisitionError(
            "native structural scan identity/certification differs"
        )
    grid = provenance.get("native_grid")
    if (
        not isinstance(grid, dict)
        or set(grid) != {"shape", "voxel_size_mm", "affine"}
        or grid.get("shape") != list(NATIVE_SHAPE)
        or float(grid.get("voxel_size_mm", 0.0)) != 3.0
    ):
        raise SourceAcquisitionError("native per-scan T1 grid differs")
    native_affine = np.asarray(grid.get("affine"), dtype=np.float64)
    expected_padded_affine = padded_affine(native_affine)

    outputs = provenance.get("outputs")
    inputs = provenance.get("inputs")
    if (
        not isinstance(outputs, dict)
        or set(outputs) != {"t1_common", "mask_common"}
        or not isinstance(inputs, dict)
        or set(inputs) != {"raw_t1", "raw_synthseg_mask"}
    ):
        raise SourceAcquisitionError(
            "native structural input/output evidence differs or includes target data"
        )
    input_artifacts = {
        "raw_t1": _artifact(inputs.get("raw_t1"), label="native raw T1 input"),
        "raw_synthseg_mask": _artifact(
            inputs.get("raw_synthseg_mask"),
            label="native raw SynthSeg input",
        ),
    }
    output_artifacts = {
        "t1w": _artifact(outputs.get("t1_common"), label="native T1 output"),
        "segmentation": _artifact(
            outputs.get("mask_common"), label="native segmentation output"
        ),
    }
    spatial_join = {
        "schema": NATIVE_SPATIAL_JOIN_SCHEMA,
        "scan_id": scan_id,
        "native_shape": list(NATIVE_SHAPE),
        "native_affine": native_affine.tolist(),
        "architecture_shape": list(ARCHITECTURE_SHAPE),
        "padding_before": list(PADDING_BEFORE),
        "padding_after": list(PADDING_AFTER),
        "padded_affine": expected_padded_affine.tolist(),
        "padding_mode": "constant-zero-no-interpolation",
        "interpolation_after_native_preprocessing": False,
    }

    binding: dict[str, object] = {
        "schema": NATIVE_STRUCTURAL_ALIGNMENT_SCHEMA,
        "scan_id": scan_id,
        "paper_certified": False,
        "certification_status": NATIVE_CERTIFICATION_STATUS,
        "spatial_profile": NATIVE_SPATIAL_PROFILE,
        "batch_authority": {
            **batch_evidence,
            "record_sha256": batch["record_sha256"],
        },
        "scan_provenance": {
            **provenance_evidence,
            "record_sha256": provenance["record_sha256"],
        },
        "native_inputs": input_artifacts,
        "native_outputs": output_artifacts,
        "native_shape": list(NATIVE_SHAPE),
        "architecture_shape": list(ARCHITECTURE_SHAPE),
        "padding_before": list(PADDING_BEFORE),
        "padding_after": list(PADDING_AFTER),
        "native_affine": native_affine.tolist(),
        "padded_affine": expected_padded_affine.tolist(),
        "padding_mode": "constant-zero-no-interpolation",
        "interpolation_after_native_preprocessing": False,
        "identically_padded_modalities": list(IDENTICALLY_PADDED_MODALITIES),
        "spatial_target_join": spatial_join,
        "spatial_target_join_sha256": canonical_sha256(spatial_join),
        "matrix_size_reported_by_paper": False,
    }
    binding["fingerprint_sha256"] = canonical_sha256(binding)
    return binding


def load_native_structural_alignment_binding(
    value: object,
    *,
    expected_scan_id: str,
    expected_batch_sha256: str,
    expected_batch_path: Path | None = None,
) -> dict[str, object]:
    """Rebuild and compare an externally batch-pinned native alignment join."""
    if not isinstance(value, dict):
        raise SourceAcquisitionError("native structural alignment binding is missing")
    batch = value.get("batch_authority")
    if not isinstance(batch, dict) or batch.get("sha256") != expected_batch_sha256:
        raise SourceAcquisitionError("native batch authority SHA-256 differs")
    if expected_batch_path is not None:
        configured = Path(expected_batch_path)
        if (
            not configured.is_absolute()
            or configured != Path(os.path.abspath(configured))
            or batch.get("path") != str(configured)
        ):
            raise SourceAcquisitionError("native batch authority path differs")
    observed = build_native_structural_alignment_binding(
        Path(str(batch.get("path", ""))),
        expected_batch_sha256=expected_batch_sha256,
        scan_id=expected_scan_id,
    )
    if observed != value:
        raise SourceAcquisitionError(
            "native structural alignment differs from authenticated evidence"
        )
    return observed


def _nifti_from_artifact(artifact: Mapping[str, object], *, label: str):
    content, _ = snapshot_binary_artifact(
        Path(str(artifact["path"])),
        expected_sha256=str(artifact["sha256"]),
        label=label,
    )
    try:
        nifti_bytes = (
            gzip.decompress(content)
            if str(artifact["path"]).endswith(".gz")
            else content
        )
        return nib.Nifti1Image.from_bytes(nifti_bytes)
    except Exception as exc:
        raise SourceAcquisitionError(f"{label} is not a valid NIfTI") from exc


def _sidecar_path(path: Path) -> Path:
    return path.with_name(
        path.name[:-7] + ".json"
        if path.name.endswith(".nii.gz")
        else path.stem + ".json"
    )


def materialize_native_padded_structural(
    batch_manifest_path: Path,
    *,
    expected_batch_sha256: str,
    scan_id: str,
    t1_output_path: Path,
    segmentation_output_path: Path,
) -> dict[str, object]:
    """Normalize and identically pad native T1/mask without interpolation.

    A separate allowlisted target receipt must reproduce the binding's signed
    ``spatial_target_join`` before padding BOLD. This structural function and
    its authorities contain no target path, bytes, or digest.
    """
    binding = build_native_structural_alignment_binding(
        Path(batch_manifest_path),
        expected_batch_sha256=expected_batch_sha256,
        scan_id=scan_id,
    )
    t1_artifact = binding["native_outputs"]["t1w"]
    segmentation_artifact = binding["native_outputs"]["segmentation"]
    native_t1 = _nifti_from_artifact(t1_artifact, label="native structural T1")
    native_segmentation = _nifti_from_artifact(
        segmentation_artifact, label="native structural segmentation"
    )
    expected_affine = np.asarray(binding["native_affine"], dtype=np.float64)
    if (
        tuple(native_t1.shape) != NATIVE_SHAPE
        or tuple(native_segmentation.shape) != NATIVE_SHAPE
        or not np.allclose(native_t1.affine, expected_affine, rtol=0.0, atol=1e-6)
        or not np.allclose(
            native_segmentation.affine, expected_affine, rtol=0.0, atol=1e-6
        )
    ):
        raise SourceAcquisitionError("native T1/segmentation geometry differs")
    t1_values = np.asarray(native_t1.dataobj, dtype=np.float32)
    segmentation_values = np.asarray(native_segmentation.dataobj, dtype=np.float32)
    if not np.isfinite(t1_values).all():
        raise SourceAcquisitionError("native T1 contains NaN or infinity")
    if not np.isfinite(segmentation_values).all() or not np.allclose(
        segmentation_values,
        np.rint(segmentation_values),
        rtol=0.0,
        atol=1e-6,
    ):
        raise SourceAcquisitionError("native segmentation is not a finite label map")
    support = segmentation_values > 0
    if not np.any(support):
        raise SourceAcquisitionError("native segmentation support is empty")
    normalized_t1 = zscore_inbrain(t1_values, support=support)
    padded_t1 = zero_pad_native(normalized_t1).astype(np.float32, copy=False)
    padded_segmentation = zero_pad_native(np.rint(segmentation_values).astype(np.int16))
    output_affine = np.asarray(binding["padded_affine"], dtype=np.float64)
    t1_output = Path(t1_output_path)
    segmentation_output = Path(segmentation_output_path)
    t1_sidecar = _sidecar_path(t1_output)
    segmentation_sidecar = _sidecar_path(segmentation_output)
    for path in (t1_output, segmentation_output, t1_sidecar, segmentation_sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to replace native materialization: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(padded_segmentation, output_affine), segmentation_output)
    segmentation_output_sha256 = hashlib.sha256(
        segmentation_output.read_bytes()
    ).hexdigest()
    segmentation_metadata = {
        "schema": NATIVE_SEG_PREPROCESSING_SCHEMA,
        "scan_id": scan_id,
        "paper_certified": False,
        "native_alignment_authority_sha256": expected_batch_sha256,
        "source_sha256": segmentation_artifact["sha256"],
        "output_sha256": segmentation_output_sha256,
        "native_alignment": binding,
        "architecture_shape": list(ARCHITECTURE_SHAPE),
        "anatomical_shape": list(NATIVE_SHAPE),
        "architecture_padding": {
            "before": list(PADDING_BEFORE),
            "after": list(PADDING_AFTER),
            "mode": "constant-zero-no-interpolation",
            "identically_applied_to": list(IDENTICALLY_PADDED_MODALITIES),
        },
    }
    segmentation_sidecar.write_text(
        json.dumps(segmentation_metadata, indent=2) + "\n", encoding="utf-8"
    )
    nib.save(nib.Nifti1Image(padded_t1, output_affine), t1_output)
    t1_sha256 = hashlib.sha256(t1_output.read_bytes()).hexdigest()
    segmentation_sidecar_sha256 = hashlib.sha256(
        segmentation_sidecar.read_bytes()
    ).hexdigest()
    t1_metadata = {
        "schema": NATIVE_T1_PREPROCESSING_SCHEMA,
        "scan_id": scan_id,
        "paper_certified": False,
        "native_alignment_authority_sha256": expected_batch_sha256,
        "source_sha256": t1_artifact["sha256"],
        "output_sha256": t1_sha256,
        "segmentation_output_sha256": segmentation_metadata["output_sha256"],
        "segmentation_sidecar_sha256": segmentation_sidecar_sha256,
        "native_alignment": binding,
        "architecture_shape": list(ARCHITECTURE_SHAPE),
        "anatomical_shape": list(NATIVE_SHAPE),
        "architecture_padding": segmentation_metadata["architecture_padding"],
        "versioned_recovery_choices": {
            "intensity_normalization": T1_INTENSITY_NORMALIZATION,
            "paper_spatial_matrix_claimed": False,
        },
    }
    t1_sidecar.write_text(json.dumps(t1_metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "binding": binding,
        "t1": str(t1_output),
        "t1_sidecar": str(t1_sidecar),
        "segmentation": str(segmentation_output),
        "segmentation_sidecar": str(segmentation_sidecar),
    }


__all__ = [
    "ARCHITECTURE_SHAPE",
    "IDENTICALLY_PADDED_MODALITIES",
    "NATIVE_CERTIFICATION_STATUS",
    "NATIVE_BATCH_MANIFEST_SCHEMA",
    "NATIVE_SEG_PREPROCESSING_SCHEMA",
    "NATIVE_SHAPE",
    "NATIVE_SCAN_PROVENANCE_SCHEMA",
    "NATIVE_SPATIAL_PROFILE",
    "NATIVE_SPATIAL_JOIN_SCHEMA",
    "NATIVE_STRUCTURAL_ALIGNMENT_SCHEMA",
    "NATIVE_STRUCTURAL_BATCH_MANIFEST_SCHEMA",
    "NATIVE_STRUCTURAL_BATCH_PURPOSE",
    "NATIVE_STRUCTURAL_SCAN_PROVENANCE_SCHEMA",
    "NATIVE_STRUCTURAL_SCAN_PURPOSE",
    "NATIVE_T1_PREPROCESSING_SCHEMA",
    "PADDING_AFTER",
    "PADDING_BEFORE",
    "build_native_structural_alignment_binding",
    "load_native_structural_alignment_binding",
    "materialize_native_padded_structural",
    "padded_affine",
    "zero_pad_native",
]
