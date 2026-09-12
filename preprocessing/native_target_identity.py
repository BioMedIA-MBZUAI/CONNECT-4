"""Fail-closed admission for non-certified native Stage-B BOLD targets.

This module is deliberately separate from the paper-profile fMRIPrep loader.
It authenticates the complete train/development controller chain before it
returns an identity for one zero-padded native BOLD target.  It never supports
sealed-test targets and never turns recovery evidence into paper certification.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from .native_structural_identity import (
    ARCHITECTURE_SHAPE,
    NATIVE_CERTIFICATION_STATUS,
    NATIVE_SHAPE,
    NATIVE_SPATIAL_JOIN_SCHEMA,
    PADDING_AFTER,
    PADDING_BEFORE,
    build_native_structural_alignment_binding,
)
from .source_acquisition_identity import (
    SourceAcquisitionError,
    canonical_sha256,
    load_authenticated_json_artifact,
    snapshot_binary_artifact,
    snapshot_json_artifact,
)


TARGET_SIDECAR_SCHEMA = "connect4-native-padded-bold-target-v1"
TARGET_PUBLICATION_SCHEMA = "connect4-native-padded-bold-publication-v1"
TARGET_IDENTITY_SCHEMA = "connect4-native-target-artifact-identity-v1"
SUCCESS_RECEIPT_SCHEMA = "connect4-half-native-success-receipt-v1"
COMPLETED_SET_SCHEMA = "connect4-half-native-completed-set-authority-v1"
NATIVE_BATCH_SCHEMA = "connect4-native-preprocessing-batch-v3"
NATIVE_SCAN_SCHEMA = "connect4-native-preprocessing-scan-v3"
NO_REPLACE_COMMIT_SCHEMA = "connect4-native-nfs-no-replace-commit-v2"
OUTPUT_AUTHORITY_SCHEMA = "connect4-half-native-output-authority-v1"
SELECTION_SCHEMA = "connect4-half-training-patient-selection-v1"
ROOT_REVIEW_SCHEMA = "connect4-half-training-patient-selection-root-review-v1"

NATIVE_SOURCE_SHA256 = (
    "ec46e437f4180b818e517b4593a398c0892f2108931cb87be2f707c54b2f222f"
)
RUNTIME_ATTESTER_SHA256 = (
    "0a397a2adeef472f252a91e5947ce0b9e616a4cc764b7a6cdf3ac0d8a72ed638"
)
NATIVE_VERIFIER_SHA256 = (
    "3cbc410f0095911f812c7639965d0e01d50d1d7b9e16db2b10da7729a8b65679"
)
CORRECTED_V3_OUTPUT_ROOT = Path(
    os.environ.get("CONNECT4_NATIVE_OUTPUT_ROOT", "data/native_preprocessing")
)
CORRECTED_V3_STAGE_MANIFEST_SHA256 = (
    "254b1c40f7dfaab663b59008bb0384417bb25b0587bf58cd97ed26a79ec6fe65"
)
CORRECTED_V3_CONTROLLER_SHA256 = (
    "ad572e5f4b0905845315cbdc9e090970320912b1cd59680d198ffed9a474f3c3"
)
CORRECTED_V3_OUTPUT_AUTHORITY_SHA256 = (
    "7200d934e2a9ec27d01d072cf4cb45c233f6725198cf4dab8e8d86351572a4b6"
)
CORRECTED_V3_OUTPUT_AUTHORITY_RECORD_SHA256 = (
    "ca22c180ea8436e154a05fe17503abc08c4fc843c9a9b2695d930be470730425"
)
CORRECTED_V3_WORKLIST_SHA256 = (
    "74119fb690684e11a717b6db2be5f581adf47456d67f5d0fe4f6ba20cd5fa264"
)
CORRECTED_V3_WORKLIST_RECORD_SHA256 = (
    "e0043a60b75133a5265395142d74f8c33e0b2fff3c861d8017749ba521cc7290"
)
CORRECTED_V3_SELECTION_MANIFEST_SHA256 = (
    "834f7bb1954bd36111f17cd886e7ab9a8ca4d80aebf25ffd6d00962ed4f1c547"
)
CORRECTED_V3_ROOT_REVIEW_SHA256 = (
    "64063fb5e2cad94dd7fc5fddbc9b1a34c5e5ff9e32d64cecf715cc6f5618d1f2"
)
PROTOCOL_PROVENANCE_SHA256 = (
    "52e2d57aa5500c09e13bc6a7061494f765a624bc4b5ef444688f8f837eae7d72"
)
PROTOCOL_RECORD_SHA256 = (
    "8868580703978455454fb08af5365a614eb4dc14fac81f4d061c8ec4788067de"
)
SOURCE_ROLE_MANIFEST_SHA256 = {
    "train": "338e591ae4ee42e9c9e834a6dea0bd253b65aed5a12d468cc0e167733f5275a9",
    "development-validation": (
        "37eaf9d2330a30d39f0dda2ec312cea546fbd69d76ecf50b1b958e7dcb71e142"
    ),
}
TARGET_FRAMES = 128
TARGET_TR_SECONDS = 3.0
VOXEL_SIZE_MM = 3.0
PADDING_MODE = "constant-zero-no-interpolation"
TARGET_ROLES = frozenset({"train", "development-validation"})
EXPECTED_TRAIN_COUNT = 2078
EXPECTED_DEVELOPMENT_COUNT = 43
EXPECTED_TOTAL_COUNT = EXPECTED_TRAIN_COUNT + EXPECTED_DEVELOPMENT_COUNT
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_MAX_TRANSIENT_SAME_SCAN_RETRIES = 1


class NativeTargetError(RuntimeError):
    """Raised when a recovery target cannot be authenticated exactly."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARACTERS)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise NativeTargetError(f"{label} is not a lowercase SHA-256")
    return str(value)


def _canonical_existing_path(path: Path, *, label: str, directory: bool = False) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise NativeTargetError(f"{label} must be an absolute canonical path")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise NativeTargetError(f"{label} is missing: {requested}") from exc
    if resolved != requested:
        raise NativeTargetError(f"{label} aliases another path")
    if directory and not requested.is_dir():
        raise NativeTargetError(f"{label} is not a directory")
    return requested


def _signed_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    expected_schema: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    source = _canonical_existing_path(path, label=label)
    try:
        if expected_sha256 is None:
            value, evidence = snapshot_json_artifact(source, label=label)
        else:
            value, evidence = load_authenticated_json_artifact(
                source,
                expected_sha256=_require_sha256(
                    expected_sha256, f"expected {label} SHA-256"
                ),
                expected_size=expected_size,
                label=label,
            )
    except SourceAcquisitionError as exc:
        raise NativeTargetError(str(exc)) from exc
    unsigned = dict(value)
    recorded = unsigned.pop("record_sha256", None)
    if (
        value.get("schema") != expected_schema
        or not _is_sha256(recorded)
        or recorded != canonical_sha256(unsigned)
    ):
        raise NativeTargetError(f"{label} signed record differs")
    return value, evidence


def _corrected_v3_task_paths(global_index: int) -> dict[str, Path]:
    if isinstance(global_index, bool) or not isinstance(global_index, int):
        raise NativeTargetError("corrected-v3 task index is invalid")
    if global_index < 0 or global_index >= EXPECTED_TOTAL_COUNT:
        raise NativeTargetError("corrected-v3 task index is outside the fixed worklist")
    root = _canonical_existing_path(
        CORRECTED_V3_OUTPUT_ROOT,
        label="corrected-v3 half-native output root",
        directory=True,
    )
    task_name = f"task_{global_index:04d}"
    return {
        "root": root,
        "output_authority": root / "output_authority.json",
        "shard": root / "shards" / task_name,
        "shard_marker": root / "shard_markers" / f"{task_name}.json",
        "success_receipt": root / "receipts" / "success" / f"{task_name}.json",
        "success_marker": (
            root / "receipts" / "success_markers" / f"{task_name}.json"
        ),
    }


def _load_corrected_v3_output_authority(
    paths: Mapping[str, Path], *, selection: Mapping[str, object]
) -> dict[str, Any]:
    authority_path = paths["output_authority"]
    authority, evidence = _signed_json(
        authority_path,
        label="corrected-v3 half-native output authority",
        expected_sha256=CORRECTED_V3_OUTPUT_AUTHORITY_SHA256,
        expected_size=1119,
        expected_schema=OUTPUT_AUTHORITY_SCHEMA,
    )
    expected = {
        "schema": OUTPUT_AUTHORITY_SCHEMA,
        "stage_manifest_sha256": CORRECTED_V3_STAGE_MANIFEST_SHA256,
        "worklist_sha256": CORRECTED_V3_WORKLIST_SHA256,
        "worklist_record_sha256": CORRECTED_V3_WORKLIST_RECORD_SHA256,
        "selection_manifest_sha256": CORRECTED_V3_SELECTION_MANIFEST_SHA256,
        "root_review_sha256": CORRECTED_V3_ROOT_REVIEW_SHA256,
        "native_source_sha256": NATIVE_SOURCE_SHA256,
        "expected_task_count": EXPECTED_TOTAL_COUNT,
        "worker_ranges": [
            {
                "worker_index": 0,
                "start_inclusive": 0,
                "end_exclusive": 1061,
            },
            {
                "worker_index": 1,
                "start_inclusive": 1061,
                "end_exclusive": EXPECTED_TOTAL_COUNT,
            },
        ],
        "no_replacement": True,
        "no_refill": True,
        "paper_certified": False,
        "sealed_target_opened": False,
        "initialized_without_target_payload_access": True,
        "record_sha256": CORRECTED_V3_OUTPUT_AUTHORITY_RECORD_SHA256,
    }
    if (
        authority != expected
        or evidence
        != {
            "path": str(authority_path),
            "sha256": CORRECTED_V3_OUTPUT_AUTHORITY_SHA256,
            "size_bytes": 1119,
        }
        or stat.S_IMODE(os.lstat(authority_path).st_mode) != 0o400
        or selection.get("selection_manifest", {}).get("sha256")
        != CORRECTED_V3_SELECTION_MANIFEST_SHA256
        or selection.get("root_review", {}).get("sha256")
        != CORRECTED_V3_ROOT_REVIEW_SHA256
    ):
        raise NativeTargetError("corrected-v3 output authority contract differs")
    return authority


def _unsigned_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    try:
        return load_authenticated_json_artifact(
            _canonical_existing_path(path, label=label),
            expected_sha256=_require_sha256(
                expected_sha256, f"expected {label} SHA-256"
            ),
            expected_size=expected_size,
            label=label,
        )
    except SourceAcquisitionError as exc:
        raise NativeTargetError(str(exc)) from exc


def _artifact(value: object, *, label: str) -> tuple[dict[str, object], bytes]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise NativeTargetError(f"{label} artifact fields differ")
    digest = _require_sha256(value.get("sha256"), f"{label} SHA-256")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise NativeTargetError(f"{label} size is invalid")
    try:
        payload, evidence = snapshot_binary_artifact(
            Path(str(value.get("path", ""))),
            expected_sha256=digest,
            label=label,
        )
    except SourceAcquisitionError as exc:
        raise NativeTargetError(str(exc)) from exc
    if evidence != value:
        raise NativeTargetError(f"{label} artifact identity differs")
    return dict(evidence), payload


def _json_artifact(
    value: object, *, label: str, expected_schema: str
) -> tuple[dict[str, Any], dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "record_sha256",
    }:
        raise NativeTargetError(f"{label} artifact fields differ")
    record, evidence = _signed_json(
        Path(str(value.get("path", ""))),
        label=label,
        expected_sha256=str(value.get("sha256", "")),
        expected_size=value.get("size_bytes"),
        expected_schema=expected_schema,
    )
    if (
        evidence
        != {key: value[key] for key in ("path", "sha256", "size_bytes")}
        or record.get("record_sha256") != value.get("record_sha256")
    ):
        raise NativeTargetError(f"{label} recorded identity differs")
    return record, evidence


def _nifti(payload: bytes, path: Path, *, label: str) -> nib.Nifti1Image:
    try:
        content = gzip.decompress(payload) if path.name.endswith(".gz") else payload
        return nib.Nifti1Image.from_bytes(content)
    except Exception as exc:
        raise NativeTargetError(f"{label} is not a valid NIfTI") from exc


def _nifti_artifact(
    value: object, *, label: str
) -> tuple[nib.Nifti1Image, dict[str, object]]:
    evidence, payload = _artifact(value, label=label)
    return _nifti(payload, Path(str(evidence["path"])), label=label), evidence


def _validate_nifti_geometry(
    image: nib.Nifti1Image,
    *,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    label: str,
    frames: int | None = None,
) -> None:
    expected_shape = (*shape, frames) if frames is not None else shape
    if tuple(image.shape) != expected_shape:
        raise NativeTargetError(f"{label} shape differs: {image.shape}")
    observed_affine = np.asarray(image.affine, dtype=np.float64)
    if (
        observed_affine.shape != (4, 4)
        or not np.isfinite(observed_affine).all()
        or not np.allclose(observed_affine, affine, rtol=0.0, atol=1e-6)
    ):
        raise NativeTargetError(f"{label} affine differs")
    zooms = np.asarray(image.header.get_zooms()[: (4 if frames is not None else 3)])
    expected_zooms = (
        (VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM, TARGET_TR_SECONDS)
        if frames is not None
        else (VOXEL_SIZE_MM, VOXEL_SIZE_MM, VOXEL_SIZE_MM)
    )
    if zooms.shape != (len(expected_zooms),) or not np.allclose(
        zooms, expected_zooms, rtol=0.0, atol=1e-6
    ):
        raise NativeTargetError(f"{label} voxel size/TR differs")


def _file_commit_marker(
    path: Path,
    *,
    expected_sha256: str,
    destination: Path,
    destination_evidence: Mapping[str, object],
    expected_record_sha256: str | None = None,
) -> dict[str, Any]:
    marker, marker_evidence = _signed_json(
        path,
        label="no-replace file commit marker",
        expected_sha256=expected_sha256,
        expected_schema=NO_REPLACE_COMMIT_SCHEMA,
    )
    expected_protocol = {
        "descriptor_rooted": True,
        "node_local_to_nfs_temporary_copy": "openat-O_CREAT|O_EXCL",
        "final_publication": "linkat-no-replace",
        "file_rehashed_frozen_and_fsynced_before_marker": True,
        "commit_marker_published_last": True,
        "overwrite_capable_rename_used": False,
    }
    metadata = os.lstat(destination)
    file_identity = marker.get("file_identity")
    if (
        set(marker)
        != {
            "schema",
            "kind",
            "status",
            "purpose",
            "destination",
            "commit_marker",
            "mode",
            "source_was_node_local",
            "file_identity",
            "publication_protocol",
            "sha256",
            "size_bytes",
            "record_sha256",
        }
        or marker.get("kind") != "single-file"
        or marker.get("status") != "COMMITTED"
        or not str(marker.get("purpose", "")).strip()
        or marker.get("destination") != str(destination)
        or marker.get("commit_marker") != str(path)
        or marker.get("mode") != 0o400
        or marker.get("source_was_node_local") is not True
        or marker.get("publication_protocol") != expected_protocol
        or marker.get("sha256") != destination_evidence.get("sha256")
        or marker.get("size_bytes") != destination_evidence.get("size_bytes")
        or not isinstance(file_identity, dict)
        or set(file_identity) != {"device", "inode"}
        or file_identity.get("device") != metadata.st_dev
        or file_identity.get("inode") != metadata.st_ino
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or marker_evidence["path"] != str(path)
        or (
            expected_record_sha256 is not None
            and marker.get("record_sha256") != expected_record_sha256
        )
    ):
        raise NativeTargetError("no-replace file commit-marker binding differs")
    return marker


def _directory_commit_marker(
    path: Path,
    *,
    expected_sha256: str,
    destination: Path,
    expected_record_sha256: str,
    expected_purpose: str,
    preverified_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    marker, marker_evidence = _signed_json(
        path,
        label="native shard commit marker",
        expected_sha256=expected_sha256,
        expected_schema=NO_REPLACE_COMMIT_SCHEMA,
    )
    root = _canonical_existing_path(
        destination, label="native committed shard", directory=True
    )
    root_metadata = os.lstat(root)
    expected_protocol = {
        "descriptor_rooted": True,
        "directory_reservation": "mkdirat-exclusive",
        "leaf_publication": "openat-O_CREAT|O_EXCL",
        "tree_rehashed_frozen_and_fsynced_before_marker": True,
        "commit_marker_published_last": True,
        "directory_rename_noreplace_used": False,
        "overwrite_capable_rename_used": False,
        "failed_reservations_are_never_cleaned_or_reused": True,
    }
    expected_fields = {
        "schema",
        "kind",
        "status",
        "purpose",
        "destination",
        "commit_marker",
        "source_was_node_local",
        "root_mode",
        "root_identity",
        "publication_protocol",
        "inventory",
        "inventory_sha256",
        "directories",
        "directories_sha256",
        "record_sha256",
    }
    if (
        set(marker) != expected_fields
        or marker.get("kind") != "directory-tree"
        or marker.get("status") != "COMMITTED"
        or marker.get("purpose") != expected_purpose
        or marker.get("destination") != str(root)
        or marker.get("commit_marker") != str(path)
        or marker.get("source_was_node_local") is not True
        or marker.get("root_mode") != 0o500
        or marker.get("root_identity")
        != {"device": root_metadata.st_dev, "inode": root_metadata.st_ino}
        or marker.get("publication_protocol") != expected_protocol
        or marker.get("record_sha256") != expected_record_sha256
        or marker_evidence["path"] != str(path)
        or stat.S_IMODE(root_metadata.st_mode) != 0o500
        or stat.S_IMODE(os.lstat(path).st_mode) != 0o400
    ):
        raise NativeTargetError("native shard commit-marker binding differs")
    inventory = marker.get("inventory")
    directories = marker.get("directories")
    if not isinstance(inventory, list) or not isinstance(directories, list):
        raise NativeTargetError("native shard commit-marker inventory differs")
    inventory_by_path: dict[str, Mapping[str, object]] = {}
    for item in inventory:
        if (
            not isinstance(item, dict)
            or set(item) != {"relative_path", "sha256", "size_bytes", "mode"}
            or not isinstance(item.get("relative_path"), str)
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item.get("size_bytes", -1) < 0
            or item.get("mode") != 0o400
            or item["relative_path"] in inventory_by_path
        ):
            raise NativeTargetError("native shard commit-marker inventory differs")
        inventory_by_path[item["relative_path"]] = item
    expected_inventory: list[dict[str, object]] = []
    expected_directories: list[str] = []
    consumed_preverified: set[str] = set()
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        for name in sorted(directory_names):
            child = current / name
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise NativeTargetError("native shard contains an unsafe directory")
            if stat.S_IMODE(metadata.st_mode) != 0o500:
                raise NativeTargetError("native shard directory mode differs")
            expected_directories.append(str(child.relative_to(root)))
        for name in sorted(file_names):
            child = current / name
            relative_path = str(child.relative_to(root))
            metadata = os.lstat(child)
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o400
            ):
                raise NativeTargetError("native shard contains an unsafe file")
            recorded = inventory_by_path.get(relative_path)
            if recorded is None:
                raise NativeTargetError("native shard contains an unrecorded file")
            artifact = preverified_artifacts.get(relative_path)
            if artifact is not None:
                if (
                    artifact.get("path") != str(child)
                    or artifact.get("sha256") != recorded["sha256"]
                    or artifact.get("size_bytes") != recorded["size_bytes"]
                ):
                    raise NativeTargetError(
                        "native shard preverified artifact binding differs"
                    )
                consumed_preverified.add(relative_path)
            else:
                try:
                    _payload, artifact = snapshot_binary_artifact(
                        child,
                        expected_sha256=str(recorded["sha256"]),
                        label="native shard committed artifact",
                    )
                except SourceAcquisitionError as exc:
                    raise NativeTargetError(str(exc)) from exc
            expected_inventory.append(
                {
                    "relative_path": relative_path,
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                    "mode": 0o400,
                }
            )
    expected_inventory.sort(key=lambda item: str(item["relative_path"]))
    expected_directories.sort()
    current_root_metadata = os.lstat(root)
    if (
        inventory != expected_inventory
        or directories != expected_directories
        or marker.get("inventory_sha256") != canonical_sha256(expected_inventory)
        or marker.get("directories_sha256")
        != canonical_sha256(expected_directories)
        or (current_root_metadata.st_dev, current_root_metadata.st_ino)
        != (root_metadata.st_dev, root_metadata.st_ino)
        or stat.S_IMODE(current_root_metadata.st_mode) != 0o500
        or consumed_preverified != set(preverified_artifacts)
    ):
        raise NativeTargetError("native shard committed inventory/bytes differ")
    return marker


def _target_role_manifest_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Bind both unsealed target roles to the signed selection manifest."""
    target_role_artifacts = {
        "train": (
            "selected_train_scans",
            "selected_train_scans.csv",
            EXPECTED_TRAIN_COUNT,
        ),
        "development-validation": (
            "development_validation",
            "development_validation_scans.csv",
            EXPECTED_DEVELOPMENT_COUNT,
        ),
    }
    role_manifest_sha256_by_role: dict[str, str] = {}
    for target_role, (target_key, target_filename, target_count) in (
        target_role_artifacts.items()
    ):
        target_record = manifest.get("output_artifacts", {}).get(target_key)
        if (
            not isinstance(target_record, dict)
            or set(target_record)
            != {"filename", "record_unit", "row_count", "sha256", "size_bytes"}
            or target_record.get("filename") != target_filename
            or target_record.get("record_unit") != "csv-data-row"
            or target_record.get("row_count") != target_count
            or not _is_sha256(target_record.get("sha256"))
            or isinstance(target_record.get("size_bytes"), bool)
            or not isinstance(target_record.get("size_bytes"), int)
            or target_record.get("size_bytes") < 1
        ):
            raise NativeTargetError(
                f"{target_role} selection artifact declaration differs"
            )
        role_manifest_sha256_by_role[target_role] = str(
            target_record["sha256"]
        )
    return role_manifest_sha256_by_role


def _load_selection(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    root_review_path: Path,
    root_review_sha256: str,
    scan_id: str,
    role: str,
) -> dict[str, object]:
    manifest, manifest_evidence = _signed_json(
        manifest_path,
        label="half selection manifest",
        expected_sha256=manifest_sha256,
        expected_schema=SELECTION_SCHEMA,
    )
    review, review_evidence = _unsigned_json(
        root_review_path,
        label="half selection root review",
        expected_sha256=root_review_sha256,
    )
    bundle = review.get("selection_bundle")
    scope = review.get("scope")
    if (
        set(review)
        != {
            "schema",
            "decision",
            "reviewed_at_utc",
            "scope",
            "selection_bundle",
            "source_authority",
            "implementation_identity",
            "independent_review_evidence",
            "review_note",
        }
        or review.get("schema") != ROOT_REVIEW_SCHEMA
        or review.get("decision")
        != "APPROVED_AS_TARGET_BLIND_HALF_SELECTION_INPUT"
        or not isinstance(bundle, dict)
        or bundle.get("selection_manifest_sha256") != manifest_evidence["sha256"]
        or not isinstance(scope, dict)
        or "sealed-target access before a prediction seal"
        not in scope.get("not_authorized", [])
    ):
        raise NativeTargetError("root review does not authorize this selection input")
    _target_role_manifest_digests(manifest)
    artifact_key = {
        "train": "selected_train_scans",
        "development-validation": "development_validation",
    }[role]
    filename = {
        "train": "selected_train_scans.csv",
        "development-validation": "development_validation_scans.csv",
    }[role]
    record = manifest.get("output_artifacts", {}).get(artifact_key)
    if (
        not isinstance(record, dict)
        or record.get("filename") != filename
        or not _is_sha256(record.get("sha256"))
    ):
        raise NativeTargetError("selection role artifact declaration differs")
    csv_path = manifest_path.parent / filename
    try:
        payload, role_evidence = snapshot_binary_artifact(
            csv_path,
            expected_sha256=str(record["sha256"]),
            label=f"{role} selection CSV",
        )
    except SourceAcquisitionError as exc:
        raise NativeTargetError(str(exc)) from exc
    if role_evidence.get("size_bytes") != record.get("size_bytes"):
        raise NativeTargetError("selection role artifact size differs")
    try:
        import csv

        reader = csv.DictReader(payload.decode("utf-8").splitlines())
        if reader.fieldnames != [
            "scan_id",
            "BID",
            "role",
            "t1_filename",
            "mask_filename",
            "fmri_filename",
            "graph_cache_complete",
            "hypergraph_cache_complete",
            "legacy_preprocessed_present",
            "common_grid_complete",
        ]:
            raise NativeTargetError("selection CSV fields differ")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise NativeTargetError("selection CSV is not UTF-8") from exc
    if len(rows) != record.get("row_count"):
        raise NativeTargetError("selection CSV row count differs")
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("scan_id") == scan_id
    ]
    if len(matches) != 1:
        raise NativeTargetError("selection does not contain exactly one requested scan")
    role_index, row = matches[0]
    bid = scan_id.split("_", 1)[0]
    if (
        row.get("BID") != bid
        or row.get("role") != role
        or row.get("t1_filename") != f"{scan_id}_T1.nii.gz"
        or row.get("mask_filename") != f"{scan_id}_mask.nii.gz"
        or row.get("fmri_filename") != f"{scan_id}_fMRI.nii.gz"
    ):
        raise NativeTargetError("selection row identity differs")
    return {
        "selection_manifest": {
            **manifest_evidence,
            "record_sha256": manifest["record_sha256"],
        },
        "root_review": review_evidence,
        "role_manifest": role_evidence,
        "role": role,
        "scan_id": scan_id,
        "BID": bid,
        "role_index": role_index,
    }


def _load_completed_set(
    *,
    path: Path,
    expected_sha256: str,
    marker_path: Path,
    marker_sha256: str,
    selection: Mapping[str, object],
    scan_id: str,
    role: str,
) -> tuple[dict[str, Any], dict[str, object], dict[str, Any], dict[str, Any]]:
    completed, evidence = _signed_json(
        path,
        label="half native completed-set authority",
        expected_sha256=expected_sha256,
        expected_schema=COMPLETED_SET_SCHEMA,
    )
    marker = _file_commit_marker(
        marker_path,
        expected_sha256=marker_sha256,
        destination=path,
        destination_evidence=evidence,
    )
    selection_manifest, _ = _signed_json(
        Path(str(selection["selection_manifest"]["path"])),
        label="half selection manifest",
        expected_sha256=str(selection["selection_manifest"]["sha256"]),
        expected_schema=SELECTION_SCHEMA,
    )
    role_manifest_sha256_by_role = _target_role_manifest_digests(
        selection_manifest
    )
    expected_selection = {
        "selection_manifest_sha256": selection["selection_manifest"]["sha256"],
        "root_review_sha256": selection["root_review"]["sha256"],
        "selected_train_scans_sha256": role_manifest_sha256_by_role["train"],
        "development_validation_scans_sha256": (
            role_manifest_sha256_by_role["development-validation"]
        ),
        "worklist_sha256": completed.get("selection", {}).get("worklist_sha256"),
        "worklist_record_sha256": completed.get("selection", {}).get(
            "worklist_record_sha256"
        ),
    }
    completed_selection = completed.get("selection")
    if (
        set(completed)
        != {
            "schema",
            "role",
            "roles",
            "selection",
            "implementation",
            "expected_count",
            "success_count",
            "quarantine_count",
            "success_receipts",
            "quarantine_receipts",
            "no_refill",
            "no_replacement",
            "complete",
            "paper_certified",
            "sealed_target_opened",
            "record_sha256",
        }
        or completed.get("role") != "train+development-validation"
        or completed.get("roles")
        != {
            "train_expected": EXPECTED_TRAIN_COUNT,
            "development_validation_expected": EXPECTED_DEVELOPMENT_COUNT,
        }
        or completed_selection != expected_selection
        or any(
            not _is_sha256(expected_selection[key])
            for key in (
                "selected_train_scans_sha256",
                "development_validation_scans_sha256",
                "worklist_sha256",
                "worklist_record_sha256",
            )
        )
        or completed.get("implementation")
        != {"native_source_sha256": NATIVE_SOURCE_SHA256}
        or completed.get("expected_count") != EXPECTED_TOTAL_COUNT
        or isinstance(completed.get("success_count"), bool)
        or not isinstance(completed.get("success_count"), int)
        or isinstance(completed.get("quarantine_count"), bool)
        or not isinstance(completed.get("quarantine_count"), int)
        or completed.get("success_count") + completed.get("quarantine_count")
        != EXPECTED_TOTAL_COUNT
        or completed.get("no_refill") is not True
        or completed.get("no_replacement") is not True
        or completed.get("complete") is not True
        or completed.get("paper_certified") is not False
        or completed.get("sealed_target_opened") is not False
    ):
        raise NativeTargetError("completed-set semantic contract differs")
    entry_fields = {
        "global_index",
        "role_index",
        "scan_id",
        "role",
        "path",
        "sha256",
        "record_sha256",
        "commit_marker_path",
        "commit_marker_sha256",
        "commit_marker_record_sha256",
    }
    successes = completed.get("success_receipts")
    quarantines = completed.get("quarantine_receipts")
    if (
        not isinstance(successes, list)
        or not isinstance(quarantines, list)
        or len(successes) != completed["success_count"]
        or len(quarantines) != completed["quarantine_count"]
        or any(not isinstance(item, dict) or set(item) != entry_fields for item in successes)
        or any(
            not isinstance(item, dict) or set(item) != entry_fields
            for item in quarantines
        )
    ):
        raise NativeTargetError("completed-set receipt inventories differ")
    all_entries = successes + quarantines
    indices = [item.get("global_index") for item in all_entries]
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or set(indices) != set(range(EXPECTED_TOTAL_COUNT))
        or len(indices) != len(set(indices))
        or len({item.get("scan_id") for item in all_entries}) != EXPECTED_TOTAL_COUNT
    ):
        raise NativeTargetError("completed-set task coverage differs")
    role_index = int(selection["role_index"])
    global_index = role_index if role == "train" else EXPECTED_TRAIN_COUNT + role_index
    matches = [
        item
        for item in successes
        if item.get("scan_id") == scan_id
        and item.get("role") == role
        and item.get("role_index") == role_index
        and item.get("global_index") == global_index
    ]
    if len(matches) != 1 or any(
        item.get("scan_id") == scan_id for item in quarantines
    ):
        raise NativeTargetError("requested scan is not one unique completed success")
    return completed, evidence, matches[0], marker


def _load_success_receipt(
    entry: Mapping[str, Any],
    *,
    completed_selection: Mapping[str, Any],
    selection: Mapping[str, object],
    expected_runtime_attester_sha256: str,
    expected_native_verifier_sha256: str,
) -> tuple[dict[str, Any], dict[str, object], dict[str, Path]]:
    paths = _corrected_v3_task_paths(entry.get("global_index"))
    if (
        entry.get("path") != str(paths["success_receipt"])
        or entry.get("commit_marker_path") != str(paths["success_marker"])
    ):
        raise NativeTargetError("success receipt is not from the corrected-v3 output")
    _load_corrected_v3_output_authority(paths, selection=selection)
    receipt, evidence = _signed_json(
        paths["success_receipt"],
        label="half native success receipt",
        expected_sha256=str(entry["sha256"]),
        expected_schema=SUCCESS_RECEIPT_SCHEMA,
    )
    if (
        evidence["sha256"] != entry["sha256"]
        or receipt.get("record_sha256") != entry["record_sha256"]
    ):
        raise NativeTargetError("completed-set success receipt identity differs")
    marker = _file_commit_marker(
        paths["success_marker"],
        expected_sha256=str(entry["commit_marker_sha256"]),
        destination=paths["success_receipt"],
        destination_evidence=evidence,
        expected_record_sha256=str(entry["commit_marker_record_sha256"]),
    )
    expected_task = {
        "global_index": entry["global_index"],
        "worker_index": 0 if entry["global_index"] < 1061 else 1,
        "lane_index": (
            entry["global_index"]
            if entry["global_index"] < 1061
            else entry["global_index"] - 1061
        )
        % 4,
        "role_index": entry["role_index"],
        "scan_id": entry["scan_id"],
        "BID": str(entry["scan_id"]).split("_", 1)[0],
        "role": entry["role"],
    }
    strict = receipt.get("strict_v2")
    implementation = receipt.get("implementation")
    execution = receipt.get("execution")
    if (
        set(receipt)
        != {
            "schema",
            "task",
            "selection",
            "strict_v2",
            "implementation",
            "outcome",
            "paper_certified",
            "sealed_target_opened",
            "no_replacement",
            "no_refill",
            "execution",
            "publication",
            "record_sha256",
        }
        or receipt.get("task") != expected_task
        or receipt.get("selection") != completed_selection
        or receipt.get("selection", {}).get("selection_manifest_sha256")
        != selection["selection_manifest"]["sha256"]
        or receipt.get("selection", {}).get("root_review_sha256")
        != selection["root_review"]["sha256"]
        or receipt.get("selection", {}).get("worklist_sha256")
        != CORRECTED_V3_WORKLIST_SHA256
        or receipt.get("selection", {}).get("worklist_record_sha256")
        != CORRECTED_V3_WORKLIST_RECORD_SHA256
        or strict
        != {
            "protocol_provenance_sha256": PROTOCOL_PROVENANCE_SHA256,
            "protocol_record_sha256": PROTOCOL_RECORD_SHA256,
            "source_role_manifest_sha256": SOURCE_ROLE_MANIFEST_SHA256[
                str(entry["role"])
            ],
        }
        or implementation
        != {
            "native_source_sha256": NATIVE_SOURCE_SHA256,
            "runtime_attester_sha256": expected_runtime_attester_sha256,
            "native_verifier_sha256": expected_native_verifier_sha256,
        }
        or receipt.get("outcome") != "SUCCESS"
        or receipt.get("paper_certified") is not False
        or receipt.get("sealed_target_opened") is not False
        or receipt.get("no_replacement") is not True
        or receipt.get("no_refill") is not True
        or not isinstance(execution, dict)
        or set(execution)
        != {
            "slurm_job_id",
            "slurm_job_nodelist",
            "slurm_partition",
            "slurm_qos",
            "allocated_gpu_count",
            "cuda_visible_devices",
            "controller_gpu_token",
            "worker_index",
            "lane_index",
            "same_scan_attempt_count",
            "same_scan_transient_retry_limit",
        }
        or execution.get("slurm_partition") != "cscc-gpu-p"
        or execution.get("slurm_qos") != "cscc-gpu-qos"
        or execution.get("allocated_gpu_count") != 4
        or execution.get("worker_index") != expected_task["worker_index"]
        or execution.get("lane_index") != expected_task["lane_index"]
        or isinstance(execution.get("same_scan_attempt_count"), bool)
        or execution.get("same_scan_attempt_count") not in {None, 1, 2}
        or execution.get("same_scan_transient_retry_limit")
        != _MAX_TRANSIENT_SAME_SCAN_RETRIES
        or str(execution.get("controller_gpu_token", ""))
        != str(execution.get("cuda_visible_devices", ""))
        or not str(execution.get("cuda_visible_devices", "")).strip()
        or "," in str(execution.get("cuda_visible_devices", ""))
        or marker.get("destination") != entry["path"]
    ):
        raise NativeTargetError("success receipt semantic/GPU binding differs")
    return receipt, evidence, paths


def _load_native_target_bundle(
    receipt: Mapping[str, Any],
    *,
    scan_id: str,
    role: str,
    controller_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    publication = receipt.get("publication")
    expected_keys = {
        "shard_path",
        "shard_commit_marker_path",
        "shard_commit_marker_sha256",
        "shard_commit_marker_record_sha256",
        "native_batch_manifest",
        "scan_provenance",
        "preprocessed_fmri",
        "t1_common",
        "mask_common",
        "full_native_verification",
    }
    if not isinstance(publication, dict) or set(publication) != expected_keys:
        raise NativeTargetError("success publication fields differ")
    expected_shard = controller_paths["shard"]
    expected_shard_marker = controller_paths["shard_marker"]
    task = receipt["task"]
    if (
        publication.get("shard_path") != str(expected_shard)
        or publication.get("shard_commit_marker_path")
        != str(expected_shard_marker)
        or task.get("scan_id") != scan_id
        or task.get("role") != role
    ):
        raise NativeTargetError("success publication is not the corrected-v3 task shard")
    _canonical_existing_path(
        expected_shard, label="corrected-v3 native task shard", directory=True
    )
    expected_scan_root = expected_shard / scan_id
    batch_artifact = publication.get("native_batch_manifest")
    provenance_artifact = publication.get("scan_provenance")
    if not isinstance(batch_artifact, dict) or not isinstance(
        provenance_artifact, dict
    ):
        raise NativeTargetError("corrected-v3 native manifest artifacts differ")
    if batch_artifact.get("path") != str(
        expected_shard / "native_preprocessing_batch_manifest.json"
    ) or provenance_artifact.get("path") != str(
        expected_scan_root / "preprocessing_provenance.json"
    ):
        raise NativeTargetError("corrected-v3 native manifest paths differ")
    batch, batch_evidence = _json_artifact(
        publication["native_batch_manifest"],
        label="ec46 native batch manifest",
        expected_schema=NATIVE_BATCH_SCHEMA,
    )
    provenance, provenance_evidence = _json_artifact(
        publication["scan_provenance"],
        label="ec46 native scan provenance",
        expected_schema=NATIVE_SCAN_SCHEMA,
    )
    strict = receipt["strict_v2"]
    if (
        batch.get("role") != role
        or batch.get("paper_certified") is not False
        or batch.get("sealed_test_opened") is not False
        or batch.get("implementation", {}).get("source_sha256")
        != NATIVE_SOURCE_SHA256
        or batch.get("protocol_provenance", {}).get("sha256")
        != strict["protocol_provenance_sha256"]
        or batch.get("protocol_provenance", {}).get("record_sha256")
        != strict["protocol_record_sha256"]
        or batch.get("source_role_manifest", {}).get("sha256")
        != SOURCE_ROLE_MANIFEST_SHA256[role]
        or batch.get("selection", {}).get("scan_id") != scan_id
        or batch.get("selection", {}).get("selected_count") != 1
        or provenance.get("scan_id") != scan_id
        or provenance.get("role") != role
        or provenance.get("paper_certified") is not False
        or provenance.get("fmriprep_claimed") is not False
        or provenance.get("implementation", {}).get("source_sha256")
        != NATIVE_SOURCE_SHA256
    ):
        raise NativeTargetError("native target batch/provenance authority differs")
    contract = batch.get("preprocessing_contract")
    scans = batch.get("scans")
    if (
        not isinstance(contract, dict)
        or contract.get("shape") != [*NATIVE_SHAPE, TARGET_FRAMES]
        or float(contract.get("voxel_size_mm", 0.0)) != VOXEL_SIZE_MM
        or float(contract.get("tr_seconds", 0.0)) != TARGET_TR_SECONDS
        or not isinstance(scans, list)
        or len(scans) != 1
        or scans[0].get("scan_id") != scan_id
        or scans[0].get("BID") != scan_id.split("_", 1)[0]
        or scans[0].get("role") != role
    ):
        raise NativeTargetError("native target batch contract/row differs")
    expected_full_verification = {
        "verified": True,
        "role": role,
        "scan_count": 1,
        "patient_count": 1,
        "record_sha256": batch["record_sha256"],
        "manifest_sha256": publication["native_batch_manifest"]["sha256"],
        "scans": [
            {
                "verified": True,
                "scan_id": scan_id,
                "role": role,
                "record_sha256": provenance["record_sha256"],
                "provenance_sha256": publication["scan_provenance"]["sha256"],
                "paper_certified": False,
            }
        ],
    }
    full_verification = publication.get("full_native_verification")
    full_scans = (
        full_verification.get("scans")
        if isinstance(full_verification, dict)
        else None
    )
    full_scan = full_scans[0] if isinstance(full_scans, list) and full_scans else None
    if (
        full_verification != expected_full_verification
        or full_verification.get("verified") is not True
        or isinstance(full_verification.get("scan_count"), bool)
        or not isinstance(full_verification.get("scan_count"), int)
        or isinstance(full_verification.get("patient_count"), bool)
        or not isinstance(full_verification.get("patient_count"), int)
        or not isinstance(full_scan, dict)
        or full_scan.get("verified") is not True
        or full_scan.get("paper_certified") is not False
    ):
        raise NativeTargetError("corrected-v3 full native verification differs")
    outputs = provenance.get("outputs")
    if not isinstance(outputs, dict):
        raise NativeTargetError("native target provenance output inventory is missing")
    verified: dict[str, object] = {}
    for key in ("preprocessed_fmri", "t1_common", "mask_common"):
        artifact, _payload = _artifact(publication[key], label=f"native target {key}")
        if (
            Path(str(artifact["path"])).parent != expected_scan_root
            or
            outputs.get(key) != artifact
            or scans[0].get(f"{key}_path") != artifact["path"]
            or scans[0].get(f"{key}_sha256") != artifact["sha256"]
            or scans[0].get(f"{key}_size_bytes") != artifact["size_bytes"]
        ):
            raise NativeTargetError(f"native target {key} authorities differ")
        verified[key] = artifact
    preverified_artifacts = {
        str(Path(str(value["path"])).relative_to(expected_shard)): value
        for value in (
            batch_evidence,
            provenance_evidence,
            *(
                verified[key]
                for key in ("preprocessed_fmri", "t1_common", "mask_common")
            ),
        )
    }
    _directory_commit_marker(
        expected_shard_marker,
        expected_sha256=str(publication["shard_commit_marker_sha256"]),
        destination=expected_shard,
        expected_record_sha256=str(
            publication["shard_commit_marker_record_sha256"]
        ),
        expected_purpose=(
            f"half-native-task_{int(task['global_index']):04d}-{role}-ec46"
        ),
        preverified_artifacts=preverified_artifacts,
    )
    return batch, provenance, verified


def _compare_native_structural_target(
    structural_binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    target_artifacts: Mapping[str, object],
) -> np.ndarray:
    grid = provenance.get("common_grid")
    inputs = provenance.get("inputs")
    if (
        not isinstance(grid, dict)
        or grid.get("shape") != list(NATIVE_SHAPE)
        or float(grid.get("voxel_size_mm", 0.0)) != VOXEL_SIZE_MM
        or not isinstance(inputs, dict)
        or not {"t1", "structural_mask", "native_fmri"}.issubset(inputs)
    ):
        raise NativeTargetError("native target spatial/source evidence is missing")
    native_affine = np.asarray(grid.get("affine"), dtype=np.float64)
    join = structural_binding["spatial_target_join"]
    if (
        native_affine.shape != (4, 4)
        or not np.isfinite(native_affine).all()
        or join.get("native_affine") != native_affine.tolist()
    ):
        raise NativeTargetError("target native grid does not join structural authority")
    for native_name, structural_name in (
        ("t1", "raw_t1"),
        ("structural_mask", "raw_synthseg_mask"),
    ):
        native_input = inputs.get(native_name)
        structural_input = structural_binding["native_inputs"][structural_name]
        if (
            not isinstance(native_input, dict)
            or native_input.get("path") != structural_input["path"]
            or native_input.get("sha256") != structural_input["sha256"]
        ):
            raise NativeTargetError("target and structural raw-source identities differ")
    native_t1, _ = _nifti_artifact(
        target_artifacts["t1_common"], label="target-bundle native T1"
    )
    native_mask, _ = _nifti_artifact(
        target_artifacts["mask_common"], label="target-bundle native mask"
    )
    structural_t1, _ = _nifti_artifact(
        structural_binding["native_outputs"]["t1w"],
        label="structural-authority native T1",
    )
    structural_mask, _ = _nifti_artifact(
        structural_binding["native_outputs"]["segmentation"],
        label="structural-authority native mask",
    )
    for image, label in (
        (native_t1, "target-bundle native T1"),
        (native_mask, "target-bundle native mask"),
        (structural_t1, "structural-authority native T1"),
        (structural_mask, "structural-authority native mask"),
    ):
        _validate_nifti_geometry(
            image, shape=NATIVE_SHAPE, affine=native_affine, label=label
        )
    if not np.array_equal(
        np.asarray(native_t1.dataobj, dtype=np.float32),
        np.asarray(structural_t1.dataobj, dtype=np.float32),
    ) or not np.array_equal(
        np.asarray(native_mask.dataobj), np.asarray(structural_mask.dataobj)
    ):
        raise NativeTargetError("target-bundle and structural native arrays differ")
    bold, _ = _nifti_artifact(
        target_artifacts["preprocessed_fmri"], label="native BOLD target"
    )
    _validate_nifti_geometry(
        bold,
        shape=NATIVE_SHAPE,
        affine=native_affine,
        label="native BOLD target",
        frames=TARGET_FRAMES,
    )
    values = np.asarray(bold.dataobj, dtype=np.float32)
    mask = np.asarray(native_mask.dataobj) > 0
    final_qc = provenance.get("final_qc")
    normalization = provenance.get("normalization")
    if (
        not np.isfinite(values).all()
        or not isinstance(final_qc, dict)
        or final_qc.get("shape") != [*NATIVE_SHAPE, TARGET_FRAMES]
        or final_qc.get("outside_mask_nonzero_voxels") != 0
        or not isinstance(normalization, dict)
        or normalization.get("applied") is not True
        or float(values.min()) < 0.0
        or float(values.max()) > 1.0
        or np.any(values[~mask, :] != 0.0)
    ):
        raise NativeTargetError("native BOLD normalization/QC contract differs")
    return values


def validate_native_padded_target(
    publication_directory: Path,
    *,
    scan_id: str,
    expected_role: str,
    expected_structural_binding: Mapping[str, Any],
    structural_batch_path: Path,
    structural_batch_sha256: str,
    selection_manifest_path: Path,
    selection_manifest_sha256: str,
    selection_root_review_path: Path,
    selection_root_review_sha256: str,
    completed_set_path: Path,
    completed_set_sha256: str,
    completed_set_commit_marker_path: Path,
    completed_set_commit_marker_sha256: str,
    reviewed_native_source_path: Path,
    reviewed_native_source_sha256: str,
    runtime_attester_sha256: str,
    native_verifier_sha256: str,
) -> dict[str, object]:
    """Authenticate one train/development Stage-B target and return its identity."""
    if expected_role not in TARGET_ROLES:
        # Fail before resolving any target directory or target authority.
        raise NativeTargetError("native target admission is train/development only")
    if reviewed_native_source_sha256 != NATIVE_SOURCE_SHA256:
        raise NativeTargetError("reviewed native source pin is not ec46")
    if runtime_attester_sha256 != RUNTIME_ATTESTER_SHA256:
        raise NativeTargetError("runtime-attester pin differs from the reviewed controller")
    if native_verifier_sha256 != NATIVE_VERIFIER_SHA256:
        raise NativeTargetError("native-verifier pin differs from the reviewed controller")
    try:
        _source_bytes, source_evidence = snapshot_binary_artifact(
            reviewed_native_source_path,
            expected_sha256=reviewed_native_source_sha256,
            label="reviewed ec46 native implementation",
        )
    except SourceAcquisitionError as exc:
        raise NativeTargetError(str(exc)) from exc
    rebuilt_binding = build_native_structural_alignment_binding(
        structural_batch_path,
        expected_batch_sha256=structural_batch_sha256,
        scan_id=scan_id,
    )
    if rebuilt_binding != expected_structural_binding:
        raise NativeTargetError("target structural join differs from admitted structural data")
    directory = _canonical_existing_path(
        publication_directory, label="padded target publication", directory=True
    )
    if directory.name != scan_id:
        raise NativeTargetError("padded target publication directory identifies another scan")
    expected_names = {
        f"{scan_id}_fMRI.nii.gz",
        f"{scan_id}_fMRI.json",
        "padded_target_publication.json",
    }
    if {path.name for path in directory.iterdir()} != expected_names:
        raise NativeTargetError("padded target publication inventory differs")
    manifest_path = directory / "padded_target_publication.json"
    manifest, manifest_evidence = _signed_json(
        manifest_path,
        label="padded target publication manifest",
        expected_schema=TARGET_PUBLICATION_SCHEMA,
    )
    if (
        set(manifest)
        != {
            "schema",
            "scan_id",
            "role",
            "paper_certified",
            "selection_manifest_sha256",
            "completed_set_sha256",
            "success_receipt_sha256",
            "structural_alignment_authority_sha256",
            "spatial_target_join_sha256",
            "artifacts",
            "no_refill",
            "no_replacement",
            "record_sha256",
        }
        or manifest.get("scan_id") != scan_id
        or manifest.get("role") != expected_role
        or manifest.get("paper_certified") is not False
        or manifest.get("selection_manifest_sha256") != selection_manifest_sha256
        or manifest.get("completed_set_sha256") != completed_set_sha256
        or manifest.get("structural_alignment_authority_sha256")
        != structural_batch_sha256
        or manifest.get("no_refill") is not True
        or manifest.get("no_replacement") is not True
    ):
        raise NativeTargetError("padded target publication identity differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"bold", "bold_sidecar"}:
        raise NativeTargetError("padded target artifact inventory differs")
    bold_path = directory / f"{scan_id}_fMRI.nii.gz"
    sidecar_path = directory / f"{scan_id}_fMRI.json"
    for key, path in (("bold", bold_path), ("bold_sidecar", sidecar_path)):
        value = artifacts[key]
        if not isinstance(value, dict) or value.get("path") != str(path):
            raise NativeTargetError(f"padded target {key} path differs")
        _artifact(value, label=f"padded target {key}")
    sidecar, sidecar_evidence = _signed_json(
        sidecar_path,
        label="padded target sidecar",
        expected_sha256=str(artifacts["bold_sidecar"]["sha256"]),
        expected_size=artifacts["bold_sidecar"]["size_bytes"],
        expected_schema=TARGET_SIDECAR_SCHEMA,
    )
    expected_sidecar_fields = {
        "schema",
        "scan_id",
        "BID",
        "role",
        "paper_certified",
        "certification_status",
        "selection",
        "controller_authority",
        "native_preprocessing",
        "structural_alignment_authority_sha256",
        "structural_alignment_authority",
        "native_source_identity",
        "source_fmri",
        "source_sha256",
        "output_sha256",
        "spatial_target_join",
        "spatial_target_join_sha256",
        "architecture_padding",
        "identically_padded_modalities",
        "native_normalization",
        "normalization_preserved_without_recomputation",
        "interpolation_after_native_preprocessing",
        "record_sha256",
    }
    join = sidecar.get("spatial_target_join")
    expected_join = expected_structural_binding["spatial_target_join"]
    if (
        set(sidecar) != expected_sidecar_fields
        or sidecar.get("scan_id") != scan_id
        or sidecar.get("BID") != scan_id.split("_", 1)[0]
        or sidecar.get("role") != expected_role
        or sidecar.get("paper_certified") is not False
        or sidecar.get("certification_status") != NATIVE_CERTIFICATION_STATUS
        or not isinstance(join, dict)
        or set(join)
        != {
            "schema",
            "scan_id",
            "native_shape",
            "native_affine",
            "architecture_shape",
            "padding_before",
            "padding_after",
            "padded_affine",
            "padding_mode",
            "interpolation_after_native_preprocessing",
        }
        or join.get("schema") != NATIVE_SPATIAL_JOIN_SCHEMA
        or join != expected_join
        or sidecar.get("spatial_target_join_sha256") != canonical_sha256(join)
        or sidecar.get("spatial_target_join_sha256")
        != expected_structural_binding["spatial_target_join_sha256"]
        or sidecar.get("structural_alignment_authority_sha256")
        != structural_batch_sha256
        or sidecar.get("source_sha256")
        != sidecar.get("native_preprocessing", {}).get("preprocessed_fmri_sha256")
        or sidecar.get("output_sha256") != artifacts["bold"]["sha256"]
        or sidecar.get("architecture_padding")
        != {
            "before": list(PADDING_BEFORE),
            "after": list(PADDING_AFTER),
            "mode": PADDING_MODE,
            "identically_applied_to": ["t1w", "segmentation", "bold"],
        }
        or sidecar.get("identically_padded_modalities")
        != ["t1w", "segmentation", "bold"]
        or sidecar.get("normalization_preserved_without_recomputation") is not True
        or sidecar.get("interpolation_after_native_preprocessing") is not False
    ):
        raise NativeTargetError("padded target sidecar spatial/identity contract differs")
    structural_authority = sidecar.get("structural_alignment_authority")
    if (
        not isinstance(structural_authority, dict)
        or structural_authority != expected_structural_binding["batch_authority"]
    ):
        raise NativeTargetError("padded target structural authority differs")
    native_source_identity = sidecar.get("native_source_identity")
    source_fmri = sidecar.get("source_fmri")
    if (
        not isinstance(native_source_identity, dict)
        or set(native_source_identity)
        != {
            "schema",
            "scan_id",
            "raw_t1_sha256",
            "raw_synthseg_mask_sha256",
            "raw_fmri_sha256",
        }
        or native_source_identity.get("schema")
        != "connect4-native-target-source-identity-v1"
        or native_source_identity.get("scan_id") != scan_id
        or native_source_identity.get("raw_t1_sha256")
        != expected_structural_binding["native_inputs"]["raw_t1"]["sha256"]
        or native_source_identity.get("raw_synthseg_mask_sha256")
        != expected_structural_binding["native_inputs"]["raw_synthseg_mask"][
            "sha256"
        ]
        or not _is_sha256(native_source_identity.get("raw_fmri_sha256"))
        or not isinstance(source_fmri, dict)
        or set(source_fmri) != {"path", "sha256"}
        or source_fmri.get("sha256") != native_source_identity.get("raw_fmri_sha256")
    ):
        raise NativeTargetError("padded target raw-source identity differs")
    selection = _load_selection(
        manifest_path=selection_manifest_path,
        manifest_sha256=selection_manifest_sha256,
        root_review_path=selection_root_review_path,
        root_review_sha256=selection_root_review_sha256,
        scan_id=scan_id,
        role=expected_role,
    )
    if sidecar.get("selection") != selection:
        raise NativeTargetError("padded target selection evidence differs")
    completed, completed_evidence, entry, completed_marker = _load_completed_set(
        path=completed_set_path,
        expected_sha256=completed_set_sha256,
        marker_path=completed_set_commit_marker_path,
        marker_sha256=completed_set_commit_marker_sha256,
        selection=selection,
        scan_id=scan_id,
        role=expected_role,
    )
    receipt, receipt_evidence, controller_paths = _load_success_receipt(
        entry,
        completed_selection=completed["selection"],
        selection=selection,
        expected_runtime_attester_sha256=runtime_attester_sha256,
        expected_native_verifier_sha256=native_verifier_sha256,
    )
    controller = sidecar.get("controller_authority")
    expected_controller = {
        "completed_set": {
            **completed_evidence,
            "record_sha256": completed["record_sha256"],
        },
        "completed_set_commit_marker": {
            "path": str(completed_set_commit_marker_path),
            "sha256": completed_set_commit_marker_sha256,
            "record_sha256": completed_marker["record_sha256"],
        },
        "success_receipt": {
            **receipt_evidence,
            "record_sha256": receipt["record_sha256"],
        },
        "success_receipt_commit_marker": {
            "path": entry["commit_marker_path"],
            "sha256": entry["commit_marker_sha256"],
            "record_sha256": entry["commit_marker_record_sha256"],
        },
        "no_refill": True,
        "no_replacement": True,
    }
    if controller != expected_controller:
        raise NativeTargetError("padded target controller authority differs")
    if manifest.get("success_receipt_sha256") != receipt_evidence["sha256"]:
        raise NativeTargetError("padded target publication success receipt differs")
    _batch, provenance, native_artifacts = _load_native_target_bundle(
        receipt,
        scan_id=scan_id,
        role=expected_role,
        controller_paths=controller_paths,
    )
    native_values = _compare_native_structural_target(
        expected_structural_binding, provenance, native_artifacts
    )
    native_preprocessing = sidecar.get("native_preprocessing")
    if (
        not isinstance(native_preprocessing, dict)
        or native_preprocessing
        != {
            "success_receipt_sha256": receipt_evidence["sha256"],
            "success_receipt_record_sha256": receipt["record_sha256"],
            "native_source_sha256": source_evidence["sha256"],
            "scan_provenance_sha256": receipt["publication"]["scan_provenance"][
                "sha256"
            ],
            "scan_provenance_record_sha256": receipt["publication"][
                "scan_provenance"
            ]["record_sha256"],
            "preprocessed_fmri_sha256": native_artifacts["preprocessed_fmri"][
                "sha256"
            ],
        }
        or sidecar.get("native_normalization") != provenance.get("normalization")
    ):
        raise NativeTargetError("padded target native-preprocessing identity differs")
    bold_artifact, bold_payload = _artifact(artifacts["bold"], label="padded target BOLD")
    image = _nifti(bold_payload, bold_path, label="padded target BOLD")
    padded_affine = np.asarray(join["padded_affine"], dtype=np.float64)
    _validate_nifti_geometry(
        image,
        shape=ARCHITECTURE_SHAPE,
        affine=padded_affine,
        label="padded target BOLD",
        frames=TARGET_FRAMES,
    )
    values = np.asarray(image.dataobj, dtype=np.float32)
    crop3 = tuple(
        slice(start, start + size) for start, size in zip(PADDING_BEFORE, NATIVE_SHAPE)
    )
    crop4 = (*crop3, slice(None))
    padding_mask = np.ones(ARCHITECTURE_SHAPE, dtype=bool)
    padding_mask[crop3] = False
    if (
        not np.isfinite(values).all()
        or float(values.min()) < 0.0
        or float(values.max()) > 1.0
        or np.any(values[padding_mask, :] != 0.0)
        or not np.array_equal(values[crop4], native_values)
    ):
        raise NativeTargetError(
            "padded target values/crop differ from receipt-bound native BOLD"
        )
    identity: dict[str, object] = {
        "format": TARGET_IDENTITY_SCHEMA,
        "scan_id": scan_id,
        "role": expected_role,
        "paper_certified": False,
        "certification_status": NATIVE_CERTIFICATION_STATUS,
        "publication_sha256": manifest_evidence["sha256"],
        "publication_record_sha256": manifest["record_sha256"],
        "target_sidecar_sha256": sidecar_evidence["sha256"],
        "target_sidecar_record_sha256": sidecar["record_sha256"],
        "target_output_sha256": bold_artifact["sha256"],
        "native_bold_sha256": native_artifacts["preprocessed_fmri"]["sha256"],
        "raw_source_identity": native_source_identity,
        "spatial_target_join_sha256": sidecar["spatial_target_join_sha256"],
        "structural_alignment_authority_sha256": structural_batch_sha256,
        "completed_set_sha256": completed_evidence["sha256"],
        "completed_set_record_sha256": completed["record_sha256"],
        "completed_set_commit_marker_sha256": completed_set_commit_marker_sha256,
        "success_receipt_sha256": receipt_evidence["sha256"],
        "success_receipt_record_sha256": receipt["record_sha256"],
        "success_receipt_commit_marker_sha256": entry["commit_marker_sha256"],
        "native_batch_sha256": receipt["publication"]["native_batch_manifest"][
            "sha256"
        ],
        "native_scan_provenance_sha256": receipt["publication"]["scan_provenance"][
            "sha256"
        ],
        "reviewed_native_source_sha256": source_evidence["sha256"],
        "architecture_shape": list(ARCHITECTURE_SHAPE),
        "num_frames": TARGET_FRAMES,
        "tr_seconds": TARGET_TR_SECONDS,
    }
    identity["fingerprint_sha256"] = canonical_sha256(identity)
    return identity


__all__ = [
    "COMPLETED_SET_SCHEMA",
    "CORRECTED_V3_CONTROLLER_SHA256",
    "CORRECTED_V3_OUTPUT_AUTHORITY_RECORD_SHA256",
    "CORRECTED_V3_OUTPUT_AUTHORITY_SHA256",
    "CORRECTED_V3_OUTPUT_ROOT",
    "CORRECTED_V3_STAGE_MANIFEST_SHA256",
    "CORRECTED_V3_WORKLIST_RECORD_SHA256",
    "CORRECTED_V3_WORKLIST_SHA256",
    "NATIVE_SOURCE_SHA256",
    "NATIVE_VERIFIER_SHA256",
    "NativeTargetError",
    "RUNTIME_ATTESTER_SHA256",
    "SUCCESS_RECEIPT_SCHEMA",
    "TARGET_IDENTITY_SCHEMA",
    "TARGET_PUBLICATION_SCHEMA",
    "TARGET_ROLES",
    "TARGET_SIDECAR_SCHEMA",
    "validate_native_padded_target",
]
