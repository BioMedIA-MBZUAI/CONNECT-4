#!/usr/bin/env python3
"""Publish reviewed current-ec46 nonlinear MNI authority for BrainLM.

This builder does not estimate registration and cannot create or infer a human
review decision.  It consumes one or more signed structural-only registration
manifests and one independently hash-pinned review record, revalidates the
current native-preprocessing lineage and every runtime artifact, and emits the
only authority schema accepted by ``models.brainlm_context``.

No fMRI, prediction, or target voxel file is opened by this program.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from data.provenance import canonical_sha256, sha256_file
from utils.compat import strict_zip
from models.brainlm_context import (
    A424_AFFINE_RAS_MM,
    A424_SHAPE,
    BRAINLM_AUTHORITY_SCAN_SCHEMA,
    BRAINLM_AUTHORITY_SCHEMA,
    BRAINLM_RUNTIME_MAPPING_SCHEMA,
    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    MAXIMUM_DENSE_DISPLACEMENT_MM,
    NATIVE_SHAPE,
    OFFICIAL_A424_ATLAS_SHA256,
    OFFICIAL_A424_COORDINATES_SHA256,
    OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
    OFFICIAL_BRAINLM_CONFIG_SHA256,
    OFFICIAL_BRAINLM_REPOSITORY,
    OFFICIAL_BRAINLM_SOURCE_REVISION,
    PADDED_SHAPE,
    PADDING_AFTER,
    PADDING_BEFORE,
    _expected_mapping_contract,
    _read_bound_json,
    _read_regular_file,
    _record_sha256,
    _support_sha256,
    load_brainlm_authority,
)


REGISTRATION_SCHEMA = "connect4-structural-mni-registration-evidence-v1"
REGISTRATION_SCAN_SCHEMA = "connect4-structural-mni-registration-scan-v1"
REGISTRATION_MAPPING_SCHEMA = "connect4-mni-dense-runtime-application-v1"
NATIVE_SCAN_SCHEMA = "connect4-native-preprocessing-scan-v3"
HUMAN_REVIEW_SCHEMA = "connect4-structural-mni-a424-human-review-v2"
HUMAN_REVIEW_SCAN_SCHEMA = "connect4-structural-mni-a424-human-review-scan-v2"
REVIEW_PROTOCOL_SCHEMA = "connect4-structural-mni-a424-review-protocol-v2"

NATIVE_AFFINE_RAS_MM = np.asarray(
    [
        [3.0, 0.0, 0.0, -90.5],
        [0.0, 3.0, 0.0, -125.5],
        [0.0, 0.0, 3.0, -71.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
PADDED_AFFINE_RAS_MM = NATIVE_AFFINE_RAS_MM.copy()
PADDED_AFFINE_RAS_MM[:3, 3] -= NATIVE_AFFINE_RAS_MM[:3, :3] @ np.asarray(
    PADDING_BEFORE, dtype=np.float64
)

_REVIEWER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:+-]{2,127}$")
_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z$"
)
_SCAN_ID = re.compile(r"^B[0-9]+_[0-9]+$")
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_NIFTI_BYTES = 128 * 1024 * 1024
_MAX_DECOMPRESSED_NIFTI_BYTES = 32 * 1024 * 1024


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _is_sha256(digest):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")
    return digest


def _decompress_nifti(payload: bytes, *, label: str) -> bytes:
    if payload[:2] != b"\x1f\x8b":
        if len(payload) > _MAX_DECOMPRESSED_NIFTI_BYTES:
            raise RuntimeError(f"{label} uncompressed payload is too large")
        return payload
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            if output.tell() + len(block) > _MAX_DECOMPRESSED_NIFTI_BYTES:
                raise RuntimeError(f"{label} decompressed payload is too large")
            output.write(block)
    return output.getvalue()


def _nifti_from_authenticated_bytes(payload: bytes, *, label: str) -> nib.Nifti1Image:
    try:
        image = nib.Nifti1Image.from_bytes(_decompress_nifti(payload, label=label))
    except Exception as exc:
        raise RuntimeError(f"{label} is not a valid NIfTI-1 image") from exc
    return image


def _artifact_bytes(
    artifact: Mapping[str, Any],
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[Path, bytes]:
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"{label} artifact is missing")
    expected_size = artifact.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or expected_size > maximum_bytes
    ):
        raise RuntimeError(f"{label} artifact size differs")
    path, payload = _read_regular_file(
        Path(str(artifact.get("path", ""))),
        expected_sha256=_require_sha256(
            artifact.get("sha256"), label=f"{label} SHA-256"
        ),
        expected_size_bytes=expected_size,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    return path, payload


def _expected_review_protocol() -> dict[str, Any]:
    return {
        "schema": REVIEW_PROTOCOL_SCHEMA,
        "protocol_id": "CONNECT4_CURRENT_EC46_STRUCTURAL_MNI_A424_VISUAL_QC_V2",
        "structural_only": True,
        "prediction_or_functional_target_used": False,
        "all_three_orthogonal_planes_reviewed": True,
        "template_subject_overlay_reviewed": True,
        "a424_coverage_overlay_reviewed": True,
        "gross_misalignment_absent": True,
        "left_right_flip_absent": True,
        "non_anatomical_warp_absent": True,
        "brain_coverage_acceptable": True,
        "review_decision_not_inferred_by_software": True,
    }


def _validate_registration_mapping(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("registration runtime mapping is missing")
    expected = {
        "schema": REGISTRATION_MAPPING_SCHEMA,
        "mapping_kind": "dense_displacement_pull_field",
        "domain": "official A424 canonical RAS grid",
        "domain_shape": list(A424_SHAPE),
        "domain_affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "domain_atlas_sha256": OFFICIAL_A424_ATLAS_SHA256,
        "range": "prepared structural common-grid RAS world millimetres",
        "range_grid_shape": list(NATIVE_SHAPE),
        "range_grid_affine_ras_mm": NATIVE_AFFINE_RAS_MM.tolist(),
        "component_order": ["right", "anterior", "superior"],
        "vector_units": "millimetres",
        "equation": (
            "prepared_source_world_ras = a424_mni_world_ras + "
            "dense_displacement_ras_mm[a424_voxel]"
        ),
        "source_fmri_sampling": "trilinear, zero padding, align_corners=True",
        "source_support_sampling": (
            "trilinear, clamped to [0,1], zero padding"
        ),
        "inverse_field_required": False,
        "affine_only_fallback_forbidden": True,
        "prewarped_fmri_required": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("registration runtime mapping contract differs")
    grid = value.get("pytorch_grid_sample_contract")
    if grid != {
        "align_corners": True,
        "grid_component_order": ["x", "y", "z"],
        "input_tensor_axis_order": ["batch", "channel", "z", "y", "x"],
        "mode": "bilinear (trilinear for 5D input)",
        "padding_mode": "zeros",
        "voxel_to_normalized_equation": (
            "normalized = 2 * voxel / (size - 1) - 1"
        ),
    }:
        raise RuntimeError("registration PyTorch sampling contract differs")


def _validate_native_provenance(
    evidence: Mapping[str, Any],
    *,
    scan_id: str,
    role: str,
    prepared_t1: Mapping[str, Any],
    prepared_mask: Mapping[str, Any],
) -> dict[str, Any]:
    provenance, loaded = _read_bound_json(
        Path(str(evidence.get("path", ""))),
        expected_sha256=_require_sha256(
            evidence.get("sha256"), label=f"{scan_id} native provenance SHA-256"
        ),
        expected_size_bytes=evidence.get("size_bytes"),
        expected_schema=NATIVE_SCAN_SCHEMA,
        label=f"{scan_id} current native preprocessing provenance",
    )
    if loaded["record_sha256"] != evidence.get("record_sha256"):
        raise RuntimeError(f"{scan_id} native provenance record binding differs")
    implementation = provenance.get("implementation")
    outputs = provenance.get("outputs")
    target_access = provenance.get("target_access_contract")
    if (
        provenance.get("scan_id") != scan_id
        or provenance.get("role") != role
        or provenance.get("paper_certified") is not False
        or not isinstance(implementation, Mapping)
        or implementation.get("source_sha256")
        != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        or not isinstance(outputs, Mapping)
        or outputs.get("t1_common", {}).get("sha256") != prepared_t1.get("sha256")
        or outputs.get("mask_common", {}).get("sha256")
        != prepared_mask.get("sha256")
        or not isinstance(target_access, Mapping)
        or target_access.get("sealed_test_opened") is not False
    ):
        raise RuntimeError(f"{scan_id} current native preprocessing binding differs")
    return {
        "path": loaded["path"],
        "sha256": loaded["sha256"],
        "size_bytes": loaded["size_bytes"],
        "record_sha256": loaded["record_sha256"],
        "source_sha256": CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    }


def _validate_prepared_structural(
    artifact: Mapping[str, Any],
    *,
    scan_id: str,
    label: str,
    is_mask: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    path, payload = _artifact_bytes(
        artifact,
        label=f"{scan_id} prepared {label}",
        maximum_bytes=_MAX_NIFTI_BYTES,
    )
    image = _nifti_from_authenticated_bytes(payload, label=f"{scan_id} prepared {label}")
    if (
        image.ndim != 3
        or tuple(image.shape) != NATIVE_SHAPE
        or not np.allclose(image.affine, NATIVE_AFFINE_RAS_MM, rtol=0.0, atol=1e-6)
    ):
        raise RuntimeError(f"{scan_id} prepared {label} geometry differs")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{scan_id} prepared {label} is non-finite")
    if is_mask:
        if not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6):
            raise RuntimeError(f"{scan_id} prepared mask is not an integer label map")
        support = values > 0
        if not support.any():
            raise RuntimeError(f"{scan_id} prepared mask support is empty")
    return {
        "path": str(path),
        "sha256": str(artifact["sha256"]),
        "size_bytes": int(artifact["size_bytes"]),
    }, values


def _validate_dense_displacement(
    artifact: Mapping[str, Any], *, scan_id: str
) -> dict[str, Any]:
    expected_metadata = {
        "shape": [*A424_SHAPE, 3],
        "affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "dtype": "float32",
        "nifti_intent": "vector",
        "component_order": ["right", "anterior", "superior"],
        "runtime_mapping_schema": REGISTRATION_MAPPING_SCHEMA,
        "stored_values_exactly_verified": True,
    }
    if any(artifact.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError(f"{scan_id} dense displacement metadata differs")
    path, payload = _artifact_bytes(
        artifact,
        label=f"{scan_id} A424 dense displacement",
        maximum_bytes=_MAX_NIFTI_BYTES,
    )
    image = _nifti_from_authenticated_bytes(
        payload, label=f"{scan_id} A424 dense displacement"
    )
    if (
        tuple(image.shape) != (*A424_SHAPE, 3)
        or not np.allclose(image.affine, A424_AFFINE_RAS_MM, rtol=0.0, atol=1e-5)
        or image.header.get_intent()[0] != "vector"
        or int(image.header["qform_code"]) != 4
        or int(image.header["sform_code"]) != 4
        or np.dtype(image.get_data_dtype()) != np.dtype(np.float32)
    ):
        raise RuntimeError(f"{scan_id} dense displacement NIfTI contract differs")
    values = np.asarray(image.dataobj, dtype=np.float32)
    magnitudes = np.linalg.norm(values.astype(np.float64), axis=-1)
    if (
        not np.isfinite(values).all()
        or float(magnitudes.max()) > MAXIMUM_DENSE_DISPLACEMENT_MM
    ):
        raise RuntimeError(f"{scan_id} dense displacement is invalid or unbounded")
    return {
        "path": str(path),
        "sha256": str(artifact["sha256"]),
        "size_bytes": int(artifact["size_bytes"]),
        "shape": [*A424_SHAPE, 3],
        "affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "dtype": "float32",
        "nifti_intent": "vector",
        "component_order": ["right", "anterior", "superior"],
        "runtime_mapping_schema": BRAINLM_RUNTIME_MAPPING_SCHEMA,
    }


def _validate_qc_png(artifact: Mapping[str, Any], *, scan_id: str) -> dict[str, Any]:
    path, payload = _artifact_bytes(
        artifact,
        label=f"{scan_id} structural MNI visual QC",
        maximum_bytes=32 * 1024 * 1024,
    )
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.width < 1 or image.height < 1:
                raise RuntimeError("PNG dimensions/format differ")
            image.verify()
    except Exception as exc:
        raise RuntimeError(f"{scan_id} visual QC is not a valid PNG") from exc
    return {
        "path": str(path),
        "sha256": str(artifact["sha256"]),
        "size_bytes": int(artifact["size_bytes"]),
    }


def _validate_registration_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_builder_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, evidence = _read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_size_bytes=None,
        expected_schema=REGISTRATION_SCHEMA,
        label="current-ec46 structural MNI registration evidence",
    )
    role = str(manifest.get("role", ""))
    scans = manifest.get("scans")
    ordered_ids = manifest.get("source_ordered_scan_ids")
    upstream = manifest.get("upstream")
    implementation = manifest.get("implementation")
    atlas = manifest.get("a424_atlas")
    downstream = manifest.get("downstream_authorization")
    if (
        role not in {"train", "development-validation"}
        or manifest.get("structural_only") is not True
        or manifest.get("sealed_target_voxel_data_opened") is not False
        or manifest.get("fMRI_voxel_files_opened") != 0
        or manifest.get("fMRI_voxel_files_hashed") != 0
        or manifest.get("prediction_or_target_files_opened") != 0
        or not isinstance(scans, list)
        or not scans
        or not isinstance(ordered_ids, list)
        or len(scans) != len(ordered_ids)
        or manifest.get("scan_count") != len(scans)
        or manifest.get("source_scan_count") != len(scans)
        or manifest.get("source_ordered_scan_ids_sha256")
        != canonical_sha256(ordered_ids)
        or manifest.get("ordered_scan_ids_sha256") != canonical_sha256(ordered_ids)
        or not isinstance(upstream, Mapping)
        or upstream.get("input_kind") != "native-v3-structural-projection"
        or upstream.get("role") != role
        or upstream.get("native_source_sha256")
        != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        or upstream.get("ordered_scan_ids") != ordered_ids
        or upstream.get("ordered_scan_ids_sha256") != canonical_sha256(ordered_ids)
        or not isinstance(implementation, Mapping)
        or implementation.get("source", {}).get("sha256")
        != expected_builder_sha256
        or implementation.get("executed_private_source_snapshot", {}).get("sha256")
        != expected_builder_sha256
        or implementation.get("source_copy_hash_identical") is not True
        or not isinstance(atlas, Mapping)
        or atlas.get("sha256") != OFFICIAL_A424_ATLAS_SHA256
        or atlas.get("canonical_geometry", {}).get("shape") != list(A424_SHAPE)
        or not np.allclose(
            np.asarray(atlas.get("canonical_geometry", {}).get("affine")),
            A424_AFFINE_RAS_MM,
            rtol=0.0,
            atol=1e-5,
        )
        or not isinstance(downstream, Mapping)
        or downstream.get("status")
        != "EVIDENCE_ONLY_NOT_YET_BRAINLM_RUNTIME_AUTHORITY"
        or downstream.get("requires_explicit_human_review_wrapper") is not True
        or downstream.get("affine_only_consumer_forbidden") is not True
    ):
        raise RuntimeError("current-ec46 registration top-level contract differs")

    records: list[dict[str, Any]] = []
    for expected_id, scan in strict_zip(ordered_ids, scans):
        if not isinstance(scan, Mapping):
            raise RuntimeError("registration scan record is not an object")
        scan_id = str(scan.get("scan_id", ""))
        registration_record_sha = _record_sha256(
            scan, label=f"registration scan {scan_id}"
        )
        lineage = scan.get("lineage")
        transform = scan.get("transform")
        qc = scan.get("qc")
        if (
            scan_id != expected_id
            or not _SCAN_ID.fullmatch(scan_id)
            or scan.get("schema") != REGISTRATION_SCAN_SCHEMA
            or scan.get("role") != role
            or scan.get("structural_only") is not True
            or scan.get("fMRI_voxel_data_read") is not False
            or scan.get("prediction_or_target_data_read") is not False
            or not isinstance(lineage, Mapping)
            or lineage.get("fMRI_voxel_data_read") is not False
            or lineage.get("prediction_or_target_data_read") is not False
            or not isinstance(transform, Mapping)
            or transform.get("representation")
            != "nonlinear dense pull mapping; no lossless 4x4 representation"
            or transform.get("semantic_direction")
            != "MNI-template RAS world -> prepared-source RAS world"
            or transform.get("estimated_from_t1_intensity") is not True
            or transform.get("identity_inferred_from_nifti_header") is not False
            or transform.get("affine_only_runtime_authorized") is not False
            or not isinstance(qc, Mapping)
            or qc.get("passed") is not True
            or qc.get("visual_review")
            != {
                "artifact_key": "structural_registration_visual_qc_png",
                "automatic_visual_pass_claimed": False,
                "status": "PENDING_HUMAN_REVIEW",
            }
        ):
            raise RuntimeError(f"{scan_id} registration scan contract differs")
        _validate_registration_mapping(transform.get("runtime_application_contract"))
        artifacts = transform.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(f"{scan_id} registration artifacts are missing")
        prepared_t1 = lineage.get("prepared_t1")
        prepared_mask = lineage.get("prepared_structural_mask")
        if not isinstance(prepared_t1, Mapping) or not isinstance(
            prepared_mask, Mapping
        ):
            raise RuntimeError(f"{scan_id} prepared structural lineage is missing")
        t1_evidence, _ = _validate_prepared_structural(
            prepared_t1, scan_id=scan_id, label="T1", is_mask=False
        )
        mask_evidence, mask_values = _validate_prepared_structural(
            prepared_mask, scan_id=scan_id, label="mask", is_mask=True
        )
        native_provenance = _validate_native_provenance(
            lineage.get("preprocessing_provenance", {}),
            scan_id=scan_id,
            role=role,
            prepared_t1=prepared_t1,
            prepared_mask=prepared_mask,
        )
        dense = _validate_dense_displacement(
            artifacts.get("a424_mni_to_prepared_displacement_ras_mm", {}),
            scan_id=scan_id,
        )
        visual = _validate_qc_png(
            artifacts.get("structural_registration_visual_qc_png", {}),
            scan_id=scan_id,
        )
        padded_support = np.pad(
            mask_values > 0,
            tuple(strict_zip(PADDING_BEFORE, PADDING_AFTER)),
            mode="constant",
            constant_values=False,
        )
        if tuple(padded_support.shape) != PADDED_SHAPE:
            raise RuntimeError(f"{scan_id} padded support shape differs")
        records.append(
            {
                "scan_id": scan_id,
                "role": role,
                "registration_scan_record_sha256": registration_record_sha,
                "native_preprocessing_provenance": native_provenance,
                "prepared_t1": t1_evidence,
                "prepared_mask": mask_evidence,
                "support_tensor_sha256": _support_sha256(
                    torch.from_numpy(padded_support)
                ),
                "support_foreground_voxels": int(np.count_nonzero(padded_support)),
                "a424_dense_displacement_artifact": dense,
                "visual_qc_artifact": visual,
            }
        )
    descriptor = {
        "schema": REGISTRATION_SCHEMA,
        "sha256": evidence["sha256"],
        "record_sha256": evidence["record_sha256"],
        "role": role,
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "registration_builder_source_sha256": expected_builder_sha256,
    }
    return records, descriptor


def _validate_review(
    review: Mapping[str, Any],
    *,
    review_evidence: Mapping[str, Any],
    registration_descriptors: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    builder_source_sha256: str,
) -> list[dict[str, Any]]:
    scan_ids = [str(record["scan_id"]) for record in records]
    reviewer = review.get("reviewer")
    review_scans = review.get("scans")
    if (
        review.get("decision") != "PASS"
        or review.get("structural_only") is not True
        or review.get("prediction_or_functional_target_used") is not False
        or review.get("authority_builder_source_sha256") != builder_source_sha256
        or review.get("registration_manifests")
        != [dict(item) for item in registration_descriptors]
        or review.get("scan_count") != len(records)
        or review.get("ordered_scan_ids") != scan_ids
        or review.get("ordered_scan_ids_sha256") != canonical_sha256(scan_ids)
        or review.get("review_protocol") != _expected_review_protocol()
        or not isinstance(reviewer, Mapping)
        or set(reviewer) != {"reviewer_id", "qualification", "organization"}
        or not _REVIEWER_ID.fullmatch(str(reviewer.get("reviewer_id", "")))
        or not str(reviewer.get("qualification", "")).strip()
        or not str(reviewer.get("organization", "")).strip()
        or not _UTC.fullmatch(str(review.get("review_timestamp_utc", "")))
        or not isinstance(review_scans, list)
        or len(review_scans) != len(records)
    ):
        raise RuntimeError("explicit current-ec46 human-review contract differs")
    validated: list[dict[str, Any]] = []
    for expected, item in strict_zip(records, review_scans):
        if not isinstance(item, Mapping):
            raise RuntimeError("human-review scan record is not an object")
        review_record_sha = _record_sha256(
            item, label=f"human review scan {expected['scan_id']}"
        )
        if (
            item.get("schema") != HUMAN_REVIEW_SCAN_SCHEMA
            or item.get("scan_id") != expected["scan_id"]
            or item.get("role") != expected["role"]
            or item.get("decision") != "PASS"
            or item.get("registration_scan_record_sha256")
            != expected["registration_scan_record_sha256"]
            or item.get("visual_qc_png_sha256")
            != expected["visual_qc_artifact"]["sha256"]
            or item.get("a424_dense_displacement_sha256")
            != expected["a424_dense_displacement_artifact"]["sha256"]
            or item.get("gross_misalignment_absent") is not True
            or item.get("left_right_flip_absent") is not True
            or item.get("non_anatomical_warp_absent") is not True
            or item.get("brain_coverage_acceptable") is not True
        ):
            raise RuntimeError(
                f"{expected['scan_id']} explicit human visual review differs"
            )
        validated.append(
            {
                "decision": "PASS",
                "structural_only": True,
                "prediction_or_functional_target_used": False,
                "reviewed_displacement_sha256": expected[
                    "a424_dense_displacement_artifact"
                ]["sha256"],
                "reviewed_visual_qc_png_sha256": expected["visual_qc_artifact"][
                    "sha256"
                ],
                "review_record_sha256": review_record_sha,
                "review_file_sha256": review_evidence["sha256"],
            }
        )
    return validated


def _publish_authority(path: Path, value: Mapping[str, Any]) -> str:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise RuntimeError("BrainLM authority output must be an absolute path")
    if os.path.lexists(requested):
        raise FileExistsError(f"BrainLM authority output already exists: {requested}")
    parent = requested.parent.resolve(strict=True)
    if parent != requested.parent or not parent.is_dir():
        raise RuntimeError("BrainLM authority output parent must be canonical")
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=requested.name + ".tmp-", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise RuntimeError("BrainLM authority write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.close(descriptor)
        descriptor = -1
        digest = hashlib.sha256(payload).hexdigest()
        load_brainlm_authority(temporary, expected_sha256=digest)
        os.link(temporary, requested, follow_symlinks=False)
        published = True
        metadata = requested.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or sha256_file(requested) != digest
        ):
            raise RuntimeError("published BrainLM authority verification failed")
        return digest
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if published and (not requested.exists() or sha256_file(requested) != hashlib.sha256(payload).hexdigest()):
            raise RuntimeError("published BrainLM authority pathname changed")


def build_authority(
    *,
    registration_manifests: Sequence[Path],
    registration_manifest_sha256: Sequence[str],
    registration_builder_sha256: str,
    human_review: Path,
    human_review_sha256: str,
    output: Path,
) -> tuple[Path, str]:
    if not registration_manifests or len(registration_manifests) != len(
        registration_manifest_sha256
    ):
        raise ValueError("registration manifest paths/digests must be non-empty and paired")
    registration_builder_sha256 = _require_sha256(
        registration_builder_sha256,
        label="registration builder SHA-256",
    )
    builder_source_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    all_records: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for path, digest in strict_zip(
        registration_manifests, registration_manifest_sha256
    ):
        records, descriptor = _validate_registration_manifest(
            path,
            expected_sha256=_require_sha256(
                digest, label="registration manifest SHA-256"
            ),
            expected_builder_sha256=registration_builder_sha256,
        )
        all_records.extend(records)
        descriptors.append(descriptor)
    scan_ids = [str(record["scan_id"]) for record in all_records]
    if len(scan_ids) != len(set(scan_ids)):
        raise RuntimeError("registration manifests contain duplicate scan IDs")
    review, review_evidence = _read_bound_json(
        human_review,
        expected_sha256=_require_sha256(
            human_review_sha256, label="human review SHA-256"
        ),
        expected_size_bytes=None,
        expected_schema=HUMAN_REVIEW_SCHEMA,
        label="explicit current-ec46 structural MNI human review",
    )
    reviewed = _validate_review(
        review,
        review_evidence=review_evidence,
        registration_descriptors=descriptors,
        records=all_records,
        builder_source_sha256=builder_source_sha256,
    )
    scans: list[dict[str, Any]] = []
    for evidence, human in strict_zip(all_records, reviewed):
        record = {
            "schema": BRAINLM_AUTHORITY_SCAN_SCHEMA,
            "scan_id": evidence["scan_id"],
            "role": evidence["role"],
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "prepared_t1_sha256": evidence["prepared_t1"]["sha256"],
            "prepared_mask_sha256": evidence["prepared_mask"]["sha256"],
            "support_tensor_sha256": evidence["support_tensor_sha256"],
            "support_foreground_voxels": evidence[
                "support_foreground_voxels"
            ],
            "padded_shape": list(PADDED_SHAPE),
            "native_shape": list(NATIVE_SHAPE),
            "padding_before": list(PADDING_BEFORE),
            "padding_after": list(PADDING_AFTER),
            "padded_affine_ras_mm": PADDED_AFFINE_RAS_MM.tolist(),
            "native_affine_ras_mm": NATIVE_AFFINE_RAS_MM.tolist(),
            "registration_scan_record_sha256": evidence[
                "registration_scan_record_sha256"
            ],
            "native_preprocessing_provenance": evidence[
                "native_preprocessing_provenance"
            ],
            "prepared_t1_artifact": evidence["prepared_t1"],
            "prepared_mask_artifact": evidence["prepared_mask"],
            "visual_qc_artifact": evidence["visual_qc_artifact"],
            "human_review": human,
            "a424_dense_displacement_artifact": evidence[
                "a424_dense_displacement_artifact"
            ],
        }
        record["record_sha256"] = canonical_sha256(record)
        scans.append(record)
    authority: dict[str, Any] = {
        "schema": BRAINLM_AUTHORITY_SCHEMA,
        "status": "AUTHORIZED_AFTER_EXPLICIT_HUMAN_STRUCTURAL_REVIEW",
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "structural_only": True,
        "functional_target_used_for_registration": False,
        "sealed_target_voxel_data_opened": False,
        "authorized_roles": ["train", "development-validation"],
        "authorizes_prediction_emission": False,
        "authorizes_final_evaluation": False,
        "official_brainlm_artifacts": {
            "repository": OFFICIAL_BRAINLM_REPOSITORY,
            "source_revision": OFFICIAL_BRAINLM_SOURCE_REVISION,
            "checkpoint_sha256": OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
            "config_sha256": OFFICIAL_BRAINLM_CONFIG_SHA256,
            "a424_atlas_sha256": OFFICIAL_A424_ATLAS_SHA256,
            "a424_coordinates_sha256": OFFICIAL_A424_COORDINATES_SHA256,
        },
        "source_grid": {
            "architecture_shape": list(PADDED_SHAPE),
            "native_shape": list(NATIVE_SHAPE),
            "padding_before": list(PADDING_BEFORE),
            "padding_after": list(PADDING_AFTER),
            "frames": 128,
            "tr_seconds": 3.0,
        },
        "runtime_mapping_contract": _expected_mapping_contract(),
        "authority_builder_source_sha256": builder_source_sha256,
        "registration_manifests": descriptors,
        "human_review_evidence": {
            "schema": HUMAN_REVIEW_SCHEMA,
            **review_evidence,
            "decision": "PASS",
            "reviewer": dict(review["reviewer"]),
            "review_timestamp_utc": review["review_timestamp_utc"],
            "software_did_not_infer_review_decision": True,
        },
        "scan_count": len(scans),
        "ordered_scan_ids": scan_ids,
        "ordered_scan_ids_sha256": canonical_sha256(scan_ids),
        "scans": scans,
    }
    authority["record_sha256"] = canonical_sha256(authority)
    digest = _publish_authority(output, authority)
    return output.resolve(strict=True), digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration-manifest", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--registration-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--registration-builder-sha256", required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--human-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    path, digest = build_authority(
        registration_manifests=arguments.registration_manifest,
        registration_manifest_sha256=arguments.registration_manifest_sha256,
        registration_builder_sha256=arguments.registration_builder_sha256,
        human_review=arguments.human_review,
        human_review_sha256=arguments.human_review_sha256,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "authority": str(path),
                "sha256": digest,
                "schema": BRAINLM_AUTHORITY_SCHEMA,
                "native_preprocessing_source_sha256": (
                    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
