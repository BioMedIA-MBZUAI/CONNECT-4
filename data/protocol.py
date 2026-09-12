"""Executable provenance checks for the manuscript evaluation protocol.

The paper identifies the synthesis-training data as exactly the A4 and ADNI
paired cohorts.  File names alone cannot establish that provenance, so training
and test-time metric evaluation require a complete scan-to-cohort manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import torch

from architecture_contract import (
    REJECTED_LEGACY_CHECKPOINT_FORMATS,
    TRAINING_CHECKPOINT_FORMAT,
    require_current_synthesis_architecture,
)
from .dataset import Connect4Dataset
from .cohort_admission import (
    validate_cohort_admission_identity,
    validate_production_training_runtime_identity,
    validate_validation_shard_contract,
)
from .provenance import canonical_sha256, directory_file_sha256, sha256_file
from .spatial_contract import PATCH_COORDINATE_CONTRACT
from utils.config import A4_RECOVERY_PROTOCOL_PROFILE, PAPER_PROTOCOL_PROFILE
from eval.development_quality import (
    validate_selectable_checkpoint_development_qa,
)


PAPER_TRAINING_COHORTS: Tuple[str, str] = ("A4", "ADNI")
PAPER_COHORT_SCAN_COUNTS = {"A4": 6290, "ADNI": 700}
COHORT_MANIFEST_SCHEMA = "connect4_scan_cohorts_v1"
SPLIT_IDENTITY_SCHEMA = "connect4_split_identity_v2"
RUN_ARTIFACT_IDENTITY_SCHEMA = "connect4_run_artifacts_v6"
EXTERNAL_CACHE_PROTOCOL_SCHEMA = "connect4_external_cache_protocol_v2"
CONDITIONING_ARTIFACT_SCHEMA = "connect4_conditioning_artifacts_v3"
PAPER_VOXEL_SIZE_MM = 3.0
CLINICAL_MODERNBERT_MODEL_ID = "Simonlee711/Clinical_ModernBERT"
EXTRACTOR_ADAPTER_CONTRACTS = {
    "brainlm": "connect4-brainlm-a424-contextual-perceptual-v2",
    "slimbrain": "connect4-slimbrain-4d-adapter-v1",
}
DEFERRED_EVALUATION_EXTRACTOR_IDENTITY = {
    "status": "deferred-unopened-by-training",
    "purpose": "final-test-only",
}
NATIVE_TARGET_ARTIFACT_IDENTITY_SCHEMA = (
    "connect4-native-target-artifact-identity-v1"
)
NATIVE_RECOVERY_CERTIFICATION_STATUS = "NON_CERTIFIED_RECOVERY_PREPROCESSING"
PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA = (
    "connect4-paper-target-artifact-identity-v2"
)
TARGET_VALIDITY_MASK_CONTRACT = "connect4-observed-bold-support-v1"


@dataclass(frozen=True)
class TrainingCohortEvidence:
    cohort_by_scan: Dict[str, str]
    patient_by_scan: Dict[str, str]


@dataclass(frozen=True)
class SynthesisExclusionEvidence:
    """Checkpoint-bound identities that an external cohort must not contain."""

    patient_ids: FrozenSet[str]
    cohorts: FrozenSet[str]


def _text_field(value: object) -> str:
    return "" if value is None else str(value)


def _canonical_cohort(value: object) -> str:
    cohort = str(value).strip().upper()
    if not cohort:
        raise ValueError("cohort names must be non-empty")
    return cohort


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_recorded_extractor_fingerprint(value: object, expected_name: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if expected_name == "brainlm":
        # Completed checkpoints and runtime records must pass the same complete
        # path-independent validator as a newly constructed official module.
        # This rejects a re-signed old NFS publication, generic ViT-MAE state,
        # source/config/atlas drift, or a weakened reviewed authority.
        try:
            from models.brainlm_context import (
                validate_configured_brainlm_identity,
            )

            validate_configured_brainlm_identity(value)
        except (ImportError, RuntimeError, TypeError, ValueError):
            return False
        return True
    revision = str(value.get("source_revision", ""))
    return (
        value.get("name") == expected_name
        and _is_sha256(value.get("checkpoint_sha256"))
        and len(revision) in (40, 64)
        and all(character in "0123456789abcdef" for character in revision)
        and value.get("adapter_contract") == EXTRACTOR_ADAPTER_CONTRACTS[expected_name]
    )


def _validated_training_target_access(value: object, *, num_scans: int) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("training target-access identity must be a mapping")
    authenticated = value.get("authenticated_target_scan_ids")
    sealed = value.get("sealed_target_scan_ids")
    if (
        value.get("schema") != "connect4-training-target-access-v1"
        or value.get("sealed_targets_opened") is not False
        or not isinstance(authenticated, list)
        or not isinstance(sealed, list)
        or authenticated != sorted(set(map(str, authenticated)))
        or sealed != sorted(set(map(str, sealed)))
        or set(authenticated) & set(sealed)
        or len(authenticated) + len(sealed) != num_scans
        or value.get("authenticated_target_scan_ids_sha256")
        != canonical_sha256(authenticated)
        or value.get("sealed_target_scan_ids_sha256")
        != canonical_sha256(sealed)
    ):
        raise ValueError("training target-access identity is invalid")
    return dict(value)


def _validated_native_target_artifact_identity(
    value: object, *, scan_id: str
) -> dict:
    """Require the complete recursively admitted Stage-B identity shape."""
    expected_fields = {
        "format",
        "scan_id",
        "role",
        "paper_certified",
        "certification_status",
        "publication_sha256",
        "publication_record_sha256",
        "target_sidecar_sha256",
        "target_sidecar_record_sha256",
        "target_output_sha256",
        "native_bold_sha256",
        "raw_source_identity",
        "spatial_target_join_sha256",
        "structural_alignment_authority_sha256",
        "completed_set_sha256",
        "completed_set_record_sha256",
        "completed_set_commit_marker_sha256",
        "success_receipt_sha256",
        "success_receipt_record_sha256",
        "success_receipt_commit_marker_sha256",
        "native_batch_sha256",
        "native_scan_provenance_sha256",
        "reviewed_native_source_sha256",
        "architecture_shape",
        "num_frames",
        "tr_seconds",
        "fingerprint_sha256",
    }
    sha_fields = expected_fields - {
        "format",
        "scan_id",
        "role",
        "paper_certified",
        "certification_status",
        "raw_source_identity",
        "architecture_shape",
        "num_frames",
        "tr_seconds",
    }
    raw_source = value.get("raw_source_identity") if isinstance(value, Mapping) else None
    tr_seconds = value.get("tr_seconds") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("format") != NATIVE_TARGET_ARTIFACT_IDENTITY_SCHEMA
        or value.get("scan_id") != scan_id
        or value.get("role") not in {"train", "development-validation"}
        or value.get("paper_certified") is not False
        or value.get("certification_status")
        != NATIVE_RECOVERY_CERTIFICATION_STATUS
        or value.get("architecture_shape") != [64, 80, 64]
        or value.get("num_frames") != 128
        or isinstance(tr_seconds, bool)
        or not isinstance(tr_seconds, (int, float))
        or not math.isfinite(float(tr_seconds))
        or float(tr_seconds) != 3.0
        or any(not _is_sha256(value.get(field)) for field in sha_fields)
        or not isinstance(raw_source, Mapping)
        or set(raw_source)
        != {
            "schema",
            "scan_id",
            "raw_t1_sha256",
            "raw_synthseg_mask_sha256",
            "raw_fmri_sha256",
        }
        or raw_source.get("schema")
        != "connect4-native-target-source-identity-v1"
        or raw_source.get("scan_id") != scan_id
        or any(
            not _is_sha256(raw_source.get(field))
            for field in (
                "raw_t1_sha256",
                "raw_synthseg_mask_sha256",
                "raw_fmri_sha256",
            )
        )
        or value.get("fingerprint_sha256")
        != canonical_sha256(
            {key: item for key, item in value.items() if key != "fingerprint_sha256"}
        )
    ):
        raise RuntimeError(
            f"{scan_id} has no authenticated native target identity"
        )
    return dict(value)


def validate_fixed_protocol_config(config: Mapping) -> None:
    """Require configured assertions that are fixed by the manuscript contract."""
    if not isinstance(config, Mapping) or not isinstance(config.get("data"), Mapping):
        raise ValueError("configuration must contain a data mapping")
    value = config["data"].get("voxel_size_mm")
    if isinstance(value, bool):
        raise ValueError("data.voxel_size_mm is a fixed 3.0 mm protocol assertion")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data.voxel_size_mm is a fixed 3.0 mm protocol assertion"
        ) from exc
    if not math.isfinite(numeric) or numeric != PAPER_VOXEL_SIZE_MM:
        raise ValueError("paper-faithful execution requires data.voxel_size_mm=3.0")


def _read_cohort_rows(path: Path) -> Sequence[Tuple[str, str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"scan-to-cohort manifest not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON cohort manifest: {path}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("format") != COHORT_MANIFEST_SCHEMA
        ):
            raise ValueError(
                f"JSON cohort manifest must declare format={COHORT_MANIFEST_SCHEMA!r}"
            )
        payload = payload.get("scans")
        if not isinstance(payload, dict):
            raise ValueError("JSON cohort manifest must contain a scans mapping")
        rows = []
        for scan_id, record in payload.items():
            if not isinstance(record, dict):
                raise ValueError(
                    "each JSON scans entry must contain patient_id and cohort fields"
                )
            rows.append(
                (
                    _text_field(scan_id),
                    _text_field(record.get("patient_id", "")),
                    _text_field(record.get("cohort", "")),
                )
            )
        return rows

    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("CSV cohort manifest has no header")
            columns = {str(name).strip().lower(): name for name in reader.fieldnames}
            if not {"scan_id", "patient_id", "cohort"}.issubset(columns):
                raise ValueError(
                    "CSV cohort manifest must contain scan_id, patient_id, and cohort columns"
                )
            return [
                (
                    _text_field(row.get(columns["scan_id"], "")),
                    _text_field(row.get(columns["patient_id"], "")),
                    _text_field(row.get(columns["cohort"], "")),
                )
                for row in reader
            ]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid CSV cohort manifest: {path}") from exc


def conditioning_identity_from_cache_sources(source_fingerprint: Mapping) -> dict:
    """Extract exact shared conditioning artifacts from validated cache sources."""
    if not isinstance(source_fingerprint, Mapping):
        raise RuntimeError("cache source fingerprint is missing")
    dataset_sources = source_fingerprint.get("dataset")
    brainiac = source_fingerprint.get("brainiac")
    modernbert = source_fingerprint.get("modernbert")
    if not isinstance(dataset_sources, Mapping):
        raise RuntimeError("cache dataset source fingerprint is missing")
    common_grid_digest = dataset_sources.get("common_grid_contract_sha256")
    if common_grid_digest is not None and not _is_sha256(common_grid_digest):
        raise RuntimeError("cache common-grid conditioning identity is invalid")
    if (
        not isinstance(brainiac, Mapping)
        or brainiac.get("implementation") != "BrainIAC"
        or not _is_sha256(brainiac.get("checkpoint_sha256"))
        or not isinstance(brainiac.get("source_files_sha256"), Mapping)
        or not brainiac.get("source_files_sha256")
        or not all(
            isinstance(name, str) and name and _is_sha256(digest)
            for name, digest in brainiac.get("source_files_sha256", {}).items()
        )
        or "brainiac/load_brainiac.py" not in brainiac.get("source_files_sha256", {})
        or not _is_sha256(brainiac.get("source_files_fingerprint_sha256"))
        or brainiac.get("source_files_fingerprint_sha256")
        != canonical_sha256(brainiac.get("source_files_sha256"))
        or not isinstance(brainiac.get("adapter"), Mapping)
        or brainiac.get("adapter", {}).get("input_channels") != 1
        or brainiac.get("adapter", {}).get("resize_shape") != [96, 96, 96]
        or brainiac.get("adapter", {}).get("resize_mode") != "trilinear"
        or brainiac.get("adapter", {}).get("align_corners") is not False
        or brainiac.get("adapter", {}).get("output") != "CLS token embedding"
        or isinstance(brainiac.get("adapter", {}).get("embedding_dim"), bool)
        or not isinstance(brainiac.get("adapter", {}).get("embedding_dim"), int)
        or brainiac.get("adapter", {}).get("embedding_dim") < 1
    ):
        raise RuntimeError("cache BrainIAC conditioning identity is invalid")
    if not isinstance(modernbert, Mapping):
        raise RuntimeError("cache ModernBERT conditioning identity is invalid")
    modernbert_without_digest = dict(modernbert)
    modernbert_digest = modernbert_without_digest.pop("fingerprint_sha256", None)
    modernbert_revision = str(modernbert.get("revision", "")).strip().lower()
    modernbert_local_files = modernbert.get("local_files_sha256")
    if (
        modernbert.get("implementation") != "Clinical-ModernBERT"
        or modernbert.get("upstream_model_id") != CLINICAL_MODERNBERT_MODEL_ID
        or not str(modernbert.get("model_name", "")).strip()
        or len(modernbert_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in modernbert_revision)
        or not isinstance(modernbert_local_files, Mapping)
        or (
            modernbert.get("model_name") != CLINICAL_MODERNBERT_MODEL_ID
            and not modernbert_local_files
        )
        or any(
            not isinstance(name, str) or not name or not _is_sha256(digest)
            for name, digest in modernbert_local_files.items()
        )
        or not _is_sha256(modernbert_digest)
        or modernbert_digest != canonical_sha256(modernbert_without_digest)
    ):
        raise RuntimeError("cache ModernBERT conditioning fingerprint is invalid")
    dwi_sha256 = dataset_sources.get("dwi")
    workbook_sha256 = dataset_sources.get("normative_workbook")
    roi_features = dataset_sources.get("roi_feature_extractor_identity")
    connectivity_sha256 = source_fingerprint.get("functional_connectivity_sha256")
    if not all(
        _is_sha256(value)
        for value in (dwi_sha256, workbook_sha256, connectivity_sha256)
    ):
        raise RuntimeError(
            "cache DWI/normative/text conditioning fingerprint is invalid"
        )
    if (
        not isinstance(roi_features, Mapping)
        or roi_features.get("format")
        != Connect4Dataset.ROI_FEATURE_CONDITIONING_SCHEMA
        or not _is_sha256(roi_features.get("fingerprint_sha256"))
    ):
        raise RuntimeError("cache AnatCL/PyRadiomics conditioning identity is invalid")
    roi_without_digest = dict(roi_features)
    roi_digest = roi_without_digest.pop("fingerprint_sha256", None)
    if roi_digest != canonical_sha256(roi_without_digest):
        raise RuntimeError("cache ROI-feature conditioning fingerprint is invalid")
    roi_specs = roi_features.get("roi_specs")
    anatcl = roi_features.get("anatcl")
    radiomics = roi_features.get("pyradiomics")
    concatenation = roi_features.get("concatenation")
    if (
        not isinstance(roi_specs, list)
        or len(roi_specs) != 32
        or [
            item.get("index") if isinstance(item, Mapping) else None
            for item in roi_specs
        ]
        != list(range(32))
        or any(
            not isinstance(item, Mapping)
            or not str(item.get("slug", "")).strip()
            or isinstance(item.get("label_id"), bool)
            or not isinstance(item.get("label_id"), int)
            for item in roi_specs
        )
        or len({item["slug"] for item in roi_specs}) != 32
        or len({item["label_id"] for item in roi_specs}) != 32
        or not isinstance(anatcl, Mapping)
        or anatcl.get("schema") != Connect4Dataset.ANATCL_PROVENANCE_SCHEMA
        or anatcl.get("embedding_dim") != 512
        or anatcl.get("model") != Connect4Dataset.ANATCL_MODEL_CONTRACT
        or anatcl.get("extraction") != Connect4Dataset.ANATCL_EXTRACTION_CONTRACT
        or not isinstance(anatcl.get("cat12_input_protocol"), Mapping)
        or set(anatcl.get("cat12_input_protocol", {})) != {"schema", "pipeline"}
        or anatcl.get("cat12_input_protocol", {}).get("schema")
        != Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA
        or anatcl.get("cat12_input_protocol", {}).get("pipeline")
        != Connect4Dataset.CAT12_PREPROCESSING_CONTRACT
        or not isinstance(radiomics, Mapping)
        or radiomics.get("schema") != Connect4Dataset.RADIOMICS_PROVENANCE_SCHEMA
        or not isinstance(radiomics.get("extractor"), Mapping)
        or any(
            radiomics.get("extractor", {}).get(key) != expected
            for key, expected in Connect4Dataset.PYRADIOMICS_EXTRACTOR_CONTRACT.items()
        )
        or not _is_sha256(radiomics.get("extractor", {}).get("runtime_lock_sha256"))
        or not _is_sha256(radiomics.get("extractor", {}).get("runtime_tree_sha256"))
        or not isinstance(radiomics.get("feature_columns"), list)
        or not radiomics.get("feature_columns")
        or radiomics.get("feature_columns")
        != sorted(set(map(str, radiomics.get("feature_columns"))))
        or radiomics.get("feature_dim") != len(radiomics.get("feature_columns"))
        or concatenation
        != {
            "order": ["anatcl", "pyradiomics"],
            "output_dim": 512 + len(radiomics.get("feature_columns")),
        }
        or not _is_sha256(roi_features.get("extraction_implementation_sha256"))
    ):
        raise RuntimeError("cache ROI-feature protocol identity is invalid")
    schema = str(source_fingerprint.get("patch_description_schema_version", ""))
    coordinate_contract = source_fingerprint.get("patch_coordinate_contract")
    target_shape = source_fingerprint.get("target_shape")
    patch_size = source_fingerprint.get("patch_size")
    if (
        not schema
        or coordinate_contract != PATCH_COORDINATE_CONTRACT
        or not isinstance(target_shape, list)
        or len(target_shape) != 3
        or not isinstance(patch_size, list)
        or len(patch_size) != 3
    ):
        raise RuntimeError("cache text/grid conditioning identity is invalid")
    identity = {
        "format": CONDITIONING_ARTIFACT_SCHEMA,
        "brainiac": dict(brainiac),
        "modernbert": dict(modernbert),
        "dwi_sha256": dwi_sha256,
        "normative_workbook_sha256": workbook_sha256,
        "roi_feature_extractor": dict(roi_features),
        "patch_description_schema_version": schema,
        "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
        "functional_connectivity_sha256": connectivity_sha256,
        "target_shape": list(target_shape),
        "patch_size": list(patch_size),
    }
    if common_grid_digest is not None:
        identity["common_grid_contract_sha256"] = common_grid_digest
    return identity


def validate_training_cohort_manifest(
    manifest_path: str,
    scan_ids: Sequence[str],
    *,
    expected_cohorts: Sequence[str] = PAPER_TRAINING_COHORTS,
    expected_scan_counts: Optional[Mapping[str, int]] = None,
    protocol_profile: Optional[str] = None,
) -> TrainingCohortEvidence:
    """Return a complete canonical scan-to-cohort map or fail closed.

    The paper profile requires exactly A4+ADNI and the published 6,290/700
    counts.  The explicitly non-certified recovery profile requires A4 only
    and an exact positive configured count.  The profiles cannot masquerade as
    each other.
    """
    scans = [str(scan_id).strip() for scan_id in scan_ids]
    if not scans or any(not scan_id for scan_id in scans):
        raise ValueError("scan_ids must be a non-empty sequence of non-empty strings")
    if len(set(scans)) != len(scans):
        raise ValueError("the discovered scan cohort contains duplicate scan IDs")

    path = Path(manifest_path).expanduser()
    rows = _read_cohort_rows(path)
    mapping: Dict[str, str] = {}
    patient_by_scan: Dict[str, str] = {}
    for raw_scan_id, raw_patient_id, raw_cohort in rows:
        scan_id = raw_scan_id.strip()
        if not scan_id:
            raise ValueError("cohort manifest contains an empty scan_id")
        if scan_id in mapping:
            raise ValueError(f"cohort manifest contains duplicate scan_id {scan_id!r}")
        patient_id = raw_patient_id.strip()
        if not patient_id:
            raise ValueError(f"cohort manifest has no patient_id for scan {scan_id!r}")
        mapping[scan_id] = _canonical_cohort(raw_cohort)
        patient_by_scan[scan_id] = patient_id

    expected_scans = set(scans)
    recorded_scans = set(mapping)
    if recorded_scans != expected_scans:
        missing = sorted(expected_scans - recorded_scans)
        extra = sorted(recorded_scans - expected_scans)
        raise ValueError(
            "scan-to-cohort manifest does not exactly match the discovered cohort; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    profile = str(protocol_profile or PAPER_PROTOCOL_PROFILE).strip()
    if profile == PAPER_PROTOCOL_PROFILE:
        profile_expected_cohorts = PAPER_TRAINING_COHORTS
    elif profile == A4_RECOVERY_PROTOCOL_PROFILE:
        profile_expected_cohorts = ("A4",)
    else:
        raise ValueError("unknown synthesis cohort protocol profile")
    caller_expected = tuple(expected_cohorts)
    if (
        protocol_profile is not None
        and caller_expected != PAPER_TRAINING_COHORTS
        and {_canonical_cohort(value) for value in caller_expected}
        != {_canonical_cohort(value) for value in profile_expected_cohorts}
    ):
        raise ValueError("expected cohorts differ from the selected protocol profile")
    expected = {
        _canonical_cohort(cohort) for cohort in profile_expected_cohorts
    }
    if not expected:
        raise ValueError("expected_cohorts must not be empty")
    observed = set(mapping.values())
    if observed != expected:
        raise ValueError(
            f"{profile} requires exactly cohorts {sorted(expected)}, but the "
            f"manifest contains {sorted(observed)}"
        )

    counts = Counter(mapping.values())
    if profile == PAPER_PROTOCOL_PROFILE:
        configured_counts = (
            PAPER_COHORT_SCAN_COUNTS
            if expected_scan_counts is None
            else expected_scan_counts
        )
    else:
        if expected_scan_counts is None:
            raise ValueError(
                "A4 recovery profile requires an exact configured A4 scan count"
            )
        configured_counts = expected_scan_counts
    canonical_counts: Dict[str, int] = {}
    for cohort, count in configured_counts.items():
        name = _canonical_cohort(cohort)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("expected cohort scan counts must be positive integers")
        canonical_counts[name] = count
    if (
        profile == PAPER_PROTOCOL_PROFILE
        and canonical_counts != PAPER_COHORT_SCAN_COUNTS
    ):
        raise ValueError(
            "paper training cohort counts are fixed at "
            f"{PAPER_COHORT_SCAN_COUNTS}; they cannot be overridden or disabled"
        )
    if profile == A4_RECOVERY_PROTOCOL_PROFILE and set(canonical_counts) != {"A4"}:
        raise ValueError("A4 recovery profile count must define A4 only")
    if set(canonical_counts) != expected:
        raise ValueError(
            "paper cohort counts must define every and only the expected cohorts"
        )
    actual_counts = {cohort: counts[cohort] for cohort in sorted(expected)}
    if actual_counts != dict(sorted(canonical_counts.items())):
        raise ValueError(
            f"cohort scan counts differ from the authenticated profile counts: "
            f"expected={dict(sorted(canonical_counts.items()))}, actual={actual_counts}"
        )
    patient_cohorts: Dict[str, str] = {}
    for scan_id, patient_id in patient_by_scan.items():
        cohort = mapping[scan_id]
        previous = patient_cohorts.setdefault(patient_id, cohort)
        if previous != cohort:
            raise ValueError(
                f"patient {patient_id!r} is assigned to multiple cohorts: "
                f"{previous!r} and {cohort!r}"
            )
    return TrainingCohortEvidence(mapping, patient_by_scan)


def validate_synthesis_split_identity(
    split_identity: Mapping,
    *,
    expected_cohorts: Sequence[str] = PAPER_TRAINING_COHORTS,
    expected_scan_counts: Optional[Mapping[str, int]] = None,
) -> SynthesisExclusionEvidence:
    """Validate the exclusion evidence embedded by ``build_split_identity``.

    Downstream callers must consume this complete checkpoint field rather than
    providing ad-hoc lists of supposedly seen patients or cohorts. The
    evidence digest makes accidental or partial edits fail closed, while the
    split digest remains the checkpoint's identity for the full assignment.
    """
    if not isinstance(split_identity, Mapping):
        raise ValueError("checkpoint split_identity must be a mapping")
    if split_identity.get("format") != SPLIT_IDENTITY_SCHEMA:
        raise ValueError(
            f"checkpoint split identity must use {SPLIT_IDENTITY_SCHEMA!r}"
        )
    if not _is_sha256(split_identity.get("sha256")):
        raise ValueError("checkpoint split identity has no valid SHA-256")
    num_scans = split_identity.get("num_scans")
    if isinstance(num_scans, bool) or not isinstance(num_scans, int) or num_scans < 3:
        raise ValueError("checkpoint split identity has an invalid scan count")
    partition_counts = split_identity.get("partition_counts")
    if (
        not isinstance(partition_counts, Mapping)
        or set(partition_counts) != {"train", "validation", "test"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in partition_counts.values()
        )
        or sum(partition_counts.values()) != num_scans
    ):
        raise ValueError("checkpoint split identity has invalid partition counts")

    partitions = split_identity.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("checkpoint split identity has no complete partition records")
    canonical_partitions: Dict[str, list] = {}
    all_scan_ids = []
    partition_patients: Dict[str, set] = {}
    for split_name in ("train", "validation", "test"):
        records = partitions[split_name]
        if (
            not isinstance(records, list)
            or len(records) != partition_counts[split_name]
        ):
            raise ValueError("checkpoint partition records disagree with their counts")
        canonical_records = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("checkpoint partition records must be mappings")
            scan_id = str(record.get("scan_id", "")).strip()
            patient_id = str(record.get("patient_id", "")).strip()
            cohort = _canonical_cohort(record.get("cohort", ""))
            if not scan_id or not patient_id:
                raise ValueError("checkpoint partition record identity is empty")
            canonical_records.append(
                {"scan_id": scan_id, "patient_id": patient_id, "cohort": cohort}
            )
            all_scan_ids.append(scan_id)
        canonical_partitions[split_name] = canonical_records
        partition_patients[split_name] = {
            record["patient_id"] for record in canonical_records
        }
    if len(all_scan_ids) != num_scans or len(set(all_scan_ids)) != num_scans:
        raise ValueError("checkpoint partition records do not identify unique scans")
    if (
        partition_patients["train"] & partition_patients["validation"]
        or partition_patients["train"] & partition_patients["test"]
        or partition_patients["validation"] & partition_patients["test"]
    ):
        raise ValueError("checkpoint partition records are not patient-disjoint")
    partition_patient_counts = split_identity.get("partition_patient_counts")
    if partition_patient_counts != {
        split_name: len(partition_patients[split_name])
        for split_name in ("train", "validation", "test")
    }:
        raise ValueError("checkpoint patient counts disagree with partition records")
    if canonical_sha256(canonical_partitions) != split_identity["sha256"]:
        raise ValueError("checkpoint partition records do not match the split digest")

    raw_patients = split_identity.get("synthesis_patient_ids")
    raw_cohorts = split_identity.get("synthesis_cohorts")
    if isinstance(raw_patients, (str, bytes)) or not isinstance(
        raw_patients, (list, tuple)
    ):
        raise ValueError("checkpoint split identity has invalid synthesis patients")
    if isinstance(raw_cohorts, (str, bytes)) or not isinstance(
        raw_cohorts, (list, tuple)
    ):
        raise ValueError("checkpoint split identity has invalid synthesis cohorts")
    patients = [str(value).strip() for value in raw_patients]
    cohorts = [_canonical_cohort(value) for value in raw_cohorts]
    if not patients or any(not value for value in patients):
        raise ValueError("checkpoint synthesis patient evidence must be non-empty")
    if not cohorts:
        raise ValueError("checkpoint synthesis cohort evidence must be non-empty")
    if patients != sorted(set(patients)) or cohorts != sorted(set(cohorts)):
        raise ValueError(
            "checkpoint synthesis patient/cohort evidence must be sorted and unique"
        )
    derived_patients = sorted(
        {
            record["patient_id"]
            for split_name in ("train", "validation")
            for record in canonical_partitions[split_name]
        }
    )
    derived_cohorts = sorted(
        {
            record["cohort"]
            for records in canonical_partitions.values()
            for record in records
        }
    )
    if patients != derived_patients or cohorts != derived_cohorts:
        raise ValueError(
            "checkpoint synthesis exclusions are not derived from its partitions"
        )
    expected = sorted({_canonical_cohort(value) for value in expected_cohorts})
    if cohorts != expected:
        raise ValueError(
            "checkpoint synthesis cohort evidence differs from the paper training "
            f"cohorts: expected={expected}, actual={cohorts}"
        )
    cohort_counts = split_identity.get("cohort_counts")
    if (
        not isinstance(cohort_counts, Mapping)
        or {_canonical_cohort(value) for value in cohort_counts} != set(expected)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in cohort_counts.values()
        )
        or sum(cohort_counts.values()) != num_scans
    ):
        raise ValueError("checkpoint split identity has invalid cohort counts")
    derived_cohort_counts = Counter(
        record["cohort"]
        for records in canonical_partitions.values()
        for record in records
    )
    if dict(sorted(derived_cohort_counts.items())) != dict(
        sorted((_canonical_cohort(key), value) for key, value in cohort_counts.items())
    ):
        raise ValueError("checkpoint cohort counts disagree with partition records")
    observed_counts = dict(sorted(derived_cohort_counts.items()))
    if expected_scan_counts is None:
        if expected == sorted(PAPER_TRAINING_COHORTS):
            required_counts = PAPER_COHORT_SCAN_COUNTS
        else:
            raise ValueError(
                "non-paper checkpoint validation requires exact cohort scan counts"
            )
    else:
        required_counts = {
            _canonical_cohort(cohort): count
            for cohort, count in expected_scan_counts.items()
        }
        if set(required_counts) != set(expected) or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in required_counts.values()
        ):
            raise ValueError("expected checkpoint cohort counts are invalid")
    if observed_counts != dict(sorted(required_counts.items())):
        raise ValueError(
            "checkpoint cohort counts differ from the configured exact counts "
            f"{dict(sorted(required_counts.items()))}"
        )
    evidence_payload = {
        "synthesis_patient_ids": patients,
        "synthesis_cohorts": cohorts,
    }
    if split_identity.get("synthesis_evidence_sha256") != canonical_sha256(
        evidence_payload
    ):
        raise ValueError("checkpoint synthesis exclusion evidence digest is invalid")
    return SynthesisExclusionEvidence(frozenset(patients), frozenset(cohorts))


def validate_unseen_cohort_manifest(
    manifest_path: str,
    scan_ids: Sequence[str],
    synthesis_split_identity: Mapping,
) -> TrainingCohortEvidence:
    """Validate complete external-cohort coverage and checkpoint-level novelty."""
    exclusion = validate_synthesis_split_identity(synthesis_split_identity)
    scans = [str(scan_id).strip() for scan_id in scan_ids]
    if not scans or any(not scan_id for scan_id in scans):
        raise ValueError("scan_ids must be a non-empty sequence of non-empty strings")
    if len(set(scans)) != len(scans):
        raise ValueError("the discovered external cohort contains duplicate scan IDs")

    mapping: Dict[str, str] = {}
    patient_by_scan: Dict[str, str] = {}
    for raw_scan_id, raw_patient_id, raw_cohort in _read_cohort_rows(
        Path(manifest_path).expanduser()
    ):
        scan_id = raw_scan_id.strip()
        if not scan_id:
            raise ValueError("external cohort manifest contains an empty scan_id")
        if scan_id in mapping:
            raise ValueError(
                f"external cohort manifest contains duplicate scan_id {scan_id!r}"
            )
        patient_id = raw_patient_id.strip()
        if not patient_id:
            raise ValueError(
                f"external cohort manifest has no patient_id for scan {scan_id!r}"
            )
        mapping[scan_id] = _canonical_cohort(raw_cohort)
        patient_by_scan[scan_id] = patient_id

    expected_scans = set(scans)
    recorded_scans = set(mapping)
    if recorded_scans != expected_scans:
        missing = sorted(expected_scans - recorded_scans)
        extra = sorted(recorded_scans - expected_scans)
        raise ValueError(
            "external scan-to-cohort manifest does not exactly match the discovered "
            f"cohort; missing={missing[:10]}, extra={extra[:10]}"
        )

    patient_cohorts: Dict[str, str] = {}
    for scan_id, patient_id in patient_by_scan.items():
        cohort = mapping[scan_id]
        previous = patient_cohorts.setdefault(patient_id, cohort)
        if previous != cohort:
            raise ValueError(
                f"external patient {patient_id!r} is assigned to multiple cohorts: "
                f"{previous!r} and {cohort!r}"
            )
    patient_overlap = sorted(set(patient_cohorts) & exclusion.patient_ids)
    if patient_overlap:
        raise ValueError(
            "external data contain patients used by synthesis fitting: "
            f"{patient_overlap[:10]}"
        )
    cohort_overlap = sorted(set(mapping.values()) & exclusion.cohorts)
    if cohort_overlap:
        raise ValueError(
            f"external data contain synthesis-cohort identities: {cohort_overlap}"
        )
    return TrainingCohortEvidence(mapping, patient_by_scan)


def load_completed_synthesis_checkpoint(
    checkpoint_path: str | PathLike[str],
) -> Tuple[Mapping, SynthesisExclusionEvidence, str]:
    """Safely load the completed paper checkpoint used as unseen-data evidence."""
    if isinstance(checkpoint_path, (bytes, bytearray)):
        raise TypeError("synthesis checkpoint must be a filesystem path")
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"synthesis training checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"cannot safely load synthesis checkpoint {path}: {exc}"
        ) from exc
    observed_format = (
        checkpoint.get("format") if isinstance(checkpoint, Mapping) else None
    )
    if observed_format in REJECTED_LEGACY_CHECKPOINT_FORMATS:
        raise RuntimeError(
            "legacy synthesis checkpoint is forbidden: pre-v10 raw-epsilon or target-leaking "
            "structural identity, detail/SSIM, detached full-DDIM, or unweighted "
            "one-step composite-gradient contracts cannot authorize evaluation"
        )
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("format") != TRAINING_CHECKPOINT_FORMAT
        or not isinstance(checkpoint.get("model"), Mapping)
        or not checkpoint.get("model")
        or not isinstance(checkpoint.get("optimizer"), Mapping)
        or not isinstance(checkpoint.get("scaler"), Mapping)
        or not isinstance(checkpoint.get("rng_states"), list)
        or not checkpoint.get("rng_states")
        or isinstance(checkpoint.get("world_size"), bool)
        or not isinstance(checkpoint.get("world_size"), int)
        or checkpoint.get("world_size") < 1
        or len(checkpoint.get("rng_states", [])) != checkpoint.get("world_size")
        or not isinstance(checkpoint.get("config"), Mapping)
    ):
        raise RuntimeError(
            "a complete current synthesis-training checkpoint is required"
        )
    require_current_synthesis_architecture(
        checkpoint.get("architecture_contract"),
        checkpoint.get("architecture_contract_sha256"),
    )
    try:
        configured_epochs = checkpoint["config"]["training"]["epochs"]
    except Exception as exc:
        raise RuntimeError(
            "synthesis checkpoint has no configured training horizon"
        ) from exc
    if (
        isinstance(configured_epochs, bool)
        or not isinstance(configured_epochs, int)
        or configured_epochs < 1
        or checkpoint.get("partial") is not False
        or checkpoint.get("next_batch_index") != 0
        or checkpoint.get("next_epoch") != configured_epochs
        or isinstance(checkpoint.get("completed_batches"), bool)
        or not isinstance(checkpoint.get("completed_batches"), int)
        or checkpoint.get("completed_batches") < 1
    ):
        raise RuntimeError("a completed synthesis-training checkpoint is required")
    artifact = checkpoint.get("artifact_identity")
    common_grid_digest = (
        artifact.get("common_grid_contract_sha256")
        if isinstance(artifact, Mapping)
        else None
    )
    native_alignment_digest = (
        artifact.get("native_alignment_authority_sha256")
        if isinstance(artifact, Mapping)
        else None
    )
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
        or not _is_sha256(artifact.get("sha256"))
        or not _is_sha256(artifact.get("scaler_identity_sha256"))
        or (_is_sha256(common_grid_digest) == _is_sha256(native_alignment_digest))
        or not _is_sha256(artifact.get("spatial_authority_sha256"))
        or artifact.get("spatial_authority_sha256")
        != canonical_sha256(artifact.get("spatial_authority"))
        or not _is_sha256(artifact.get("target_artifact_identities_sha256"))
        or not _is_sha256(
            artifact.get("structural_artifact_identities_sha256")
        )
        or not isinstance(artifact.get("conditioning_identity"), Mapping)
        or not _is_sha256(artifact.get("conditioning_identity_sha256"))
        or artifact.get("conditioning_identity_sha256")
        != canonical_sha256(artifact.get("conditioning_identity"))
        or artifact.get("conditioning_identity", {}).get("format")
        != CONDITIONING_ARTIFACT_SCHEMA
        or artifact.get("evaluation_extractor")
        != DEFERRED_EVALUATION_EXTRACTOR_IDENTITY
        or not _is_recorded_extractor_fingerprint(
            artifact.get("perceptual_extractor"), "brainlm"
        )
        or isinstance(artifact.get("num_scans"), bool)
        or not isinstance(artifact.get("num_scans"), int)
        or artifact.get("num_scans") < 1
    ):
        raise RuntimeError("synthesis checkpoint has invalid run-artifact identity")
    try:
        runtime_identity = validate_production_training_runtime_identity(
            checkpoint.get("runtime_identity")
        )
        admission_identity = validate_cohort_admission_identity(
            checkpoint.get("cohort_admission_identity")
        )
        validation_shards = validate_validation_shard_contract(
            checkpoint.get("validation_shard_contract"),
            expected_cohort_admission_identity=admission_identity,
            expected_world_size=checkpoint["world_size"],
        )
        selection_validation_shards = validate_validation_shard_contract(
            checkpoint.get("selection_validation_shard_contract"),
            expected_cohort_admission_identity=admission_identity,
            expected_world_size=checkpoint["world_size"],
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"synthesis checkpoint distributed-admission identity is invalid: {exc}"
        ) from exc
    if (
        runtime_identity.get("canonical_sha256")
        != admission_identity["runtime_identity_sha256"]
        or admission_identity["config_sha256"]
        != canonical_sha256(checkpoint["config"])
        or admission_identity["artifact_identity_sha256"]
        != canonical_sha256(artifact)
        or admission_identity["structural_artifact_identities_sha256"]
        != artifact["structural_artifact_identities_sha256"]
        or admission_identity["world_size"] != checkpoint["world_size"]
        or validation_shards["world_size"] != checkpoint["world_size"]
        or selection_validation_shards["world_size"]
        != checkpoint["world_size"]
    ):
        raise RuntimeError(
            "synthesis checkpoint distributed-admission bindings differ"
        )
    try:
        target_access = _validated_training_target_access(
            artifact.get("target_access_contract"),
            num_scans=artifact["num_scans"],
        )
    except ValueError as exc:
        raise RuntimeError(
            f"synthesis checkpoint target-access identity is invalid: {exc}"
        ) from exc
    if (
        artifact.get("num_authenticated_targets")
        != len(target_access["authenticated_target_scan_ids"])
        or artifact.get("num_sealed_targets_unopened")
        != len(target_access["sealed_target_scan_ids"])
    ):
        raise RuntimeError("synthesis checkpoint target-access counts differ")
    checkpoint_data = checkpoint.get("config", {}).get("data", {})
    profile = checkpoint_data.get("protocol_profile")
    if profile == PAPER_PROTOCOL_PROFILE:
        expected_cohorts = PAPER_TRAINING_COHORTS
        expected_counts = PAPER_COHORT_SCAN_COUNTS
    elif profile == A4_RECOVERY_PROTOCOL_PROFILE:
        expected_cohorts = ("A4",)
        expected_counts = checkpoint_data.get("expected_cohort_scan_counts")
    else:
        raise RuntimeError("synthesis checkpoint protocol profile is invalid")
    try:
        exclusion = validate_synthesis_split_identity(
            checkpoint.get("split_identity"),
            expected_cohorts=expected_cohorts,
            expected_scan_counts=expected_counts,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"synthesis checkpoint exclusion identity is invalid: {exc}"
        ) from exc
    if artifact["num_scans"] != checkpoint["split_identity"]["num_scans"]:
        raise RuntimeError("checkpoint split and run-artifact scan counts differ")
    if admission_identity["split_identity_sha256"] != canonical_sha256(
        checkpoint["split_identity"]
    ):
        raise RuntimeError("checkpoint split and cohort-admission roots differ")
    development_size = len(
        checkpoint["split_identity"]["partitions"]["validation"]
    )
    configured_val_batches = checkpoint["config"]["training"].get(
        "val_batches", 4
    )
    if (
        isinstance(configured_val_batches, bool)
        or not isinstance(configured_val_batches, int)
        or configured_val_batches < 1
        or validation_shards["development_partition_size"] != development_size
        or validation_shards["global_limit"]
        != min(configured_val_batches, development_size)
        or selection_validation_shards["development_partition_size"]
        != development_size
        or selection_validation_shards["global_limit"] != development_size
    ):
        raise RuntimeError(
            "checkpoint validation-shard extent differs from the configured "
            "development first-N set"
        )
    selection_qa = validate_selectable_checkpoint_development_qa(checkpoint)
    expected_development_scan_ids = [
        str(record["scan_id"])
        for record in checkpoint["split_identity"]["partitions"]["validation"]
    ]
    if (
        selection_qa["development_partition_size"] != development_size
        or selection_qa["global_limit"] != development_size
        or selection_qa["ordered_scan_ids"] != expected_development_scan_ids
        or selection_qa["ordered_global_positions"]
        != list(range(development_size))
        or selection_qa["validation_shard_contract_sha256"]
        != selection_validation_shards["record_sha256"]
    ):
        raise RuntimeError(
            "completed checkpoint QA does not cover the fixed full development set"
        )
    expected_authenticated = sorted(
        record["scan_id"]
        for split_name in ("train", "validation")
        for record in checkpoint["split_identity"]["partitions"][split_name]
    )
    expected_sealed = sorted(
        record["scan_id"]
        for record in checkpoint["split_identity"]["partitions"]["test"]
    )
    if (
        target_access["authenticated_target_scan_ids"] != expected_authenticated
        or target_access["sealed_target_scan_ids"] != expected_sealed
    ):
        raise RuntimeError(
            "checkpoint target access does not equal train+development versus "
            "sealed-test split identities"
        )
    return checkpoint, exclusion, sha256_file(path)


def build_external_cache_protocol_context(
    *,
    manifest_path: str,
    scan_ids: Sequence[str],
    synthesis_split_identity: Mapping,
    synthesis_artifact_identity: Mapping,
    synthesis_checkpoint_sha256: str,
) -> Tuple[TrainingCohortEvidence, dict]:
    """Build the exact external-cohort context embedded in every graph cache."""
    if not _is_sha256(synthesis_checkpoint_sha256):
        raise ValueError("synthesis checkpoint SHA-256 is invalid")
    evidence = validate_unseen_cohort_manifest(
        manifest_path, scan_ids, synthesis_split_identity
    )
    conditioning_identity = synthesis_artifact_identity.get("conditioning_identity")
    conditioning_digest = synthesis_artifact_identity.get(
        "conditioning_identity_sha256"
    )
    if (
        not isinstance(conditioning_identity, Mapping)
        or conditioning_identity.get("format") != CONDITIONING_ARTIFACT_SCHEMA
        or not _is_sha256(conditioning_digest)
        or conditioning_digest != canonical_sha256(conditioning_identity)
    ):
        raise ValueError("synthesis checkpoint conditioning identity is invalid")
    common_grid_digest = synthesis_artifact_identity.get("common_grid_contract_sha256")
    if not _is_sha256(common_grid_digest):
        raise ValueError(
            "external-cache preprocessing currently requires a paper common-grid "
            "checkpoint; native per-scan recovery authorities are cohort-specific"
        )
    context = {
        "format": EXTERNAL_CACHE_PROTOCOL_SCHEMA,
        "external_cohort_manifest_sha256": sha256_file(
            Path(manifest_path).expanduser()
        ),
        "synthesis_checkpoint_sha256": synthesis_checkpoint_sha256,
        "synthesis_split_sha256": synthesis_split_identity["sha256"],
        "synthesis_exclusion_sha256": synthesis_split_identity[
            "synthesis_evidence_sha256"
        ],
        "conditioning_identity": dict(conditioning_identity),
        "conditioning_identity_sha256": conditioning_digest,
        "synthesis_common_grid_contract_sha256": common_grid_digest,
        "num_scans": len(evidence.cohort_by_scan),
        "num_patients": len(set(evidence.patient_by_scan.values())),
        "external_cohorts": sorted(set(evidence.cohort_by_scan.values())),
    }
    return evidence, context


def validate_external_scaler_provenance(
    scaler_dir: str,
    external_scan_ids: Sequence[str],
    checkpoint_artifact_identity: Mapping,
    checkpoint_split_identity: Mapping,
) -> dict:
    """Prove external preprocessing reuses the checkpoint-bound train scalers."""
    if (
        not isinstance(checkpoint_artifact_identity, Mapping)
        or checkpoint_artifact_identity.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
        or not _is_sha256(checkpoint_artifact_identity.get("sha256"))
        or not _is_sha256(checkpoint_artifact_identity.get("scaler_identity_sha256"))
    ):
        raise RuntimeError("checkpoint run-artifact/scaler identity is invalid")
    validate_synthesis_split_identity(checkpoint_split_identity)
    if not scaler_dir:
        raise ValueError("data.scaler_dir is required in external preprocessing mode")
    directory = Path(scaler_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"synthesis-training scaler directory not found: {directory}"
        )
    provenance_path = directory / "training_partition.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"missing or invalid synthesis-training scaler provenance: {provenance_path}"
        ) from exc
    training_ids = provenance.get("training_partition_scan_ids")
    fitted_ids = provenance.get("fitted_scan_ids")
    if (
        provenance.get("schema") != "connect4-training-only-scalers-v1"
        or not isinstance(training_ids, list)
        or not training_ids
        or training_ids != sorted(set(map(str, training_ids)))
        or not isinstance(fitted_ids, list)
        or fitted_ids != training_ids
    ):
        raise RuntimeError("synthesis-training scaler provenance is invalid")
    expected_training_ids = sorted(
        record["scan_id"] for record in checkpoint_split_identity["partitions"]["train"]
    )
    if sorted(map(str, training_ids)) != expected_training_ids:
        raise RuntimeError(
            "scaler training partition does not match the synthesis checkpoint"
        )
    overlap = sorted(set(map(str, external_scan_ids)) & set(map(str, training_ids)))
    if overlap:
        raise RuntimeError(
            f"external scans overlap the scaler-fitting partition: {overlap[:10]}"
        )
    expected_digest = checkpoint_artifact_identity["scaler_identity_sha256"]
    current_digest = canonical_sha256(directory_file_sha256(directory))
    if not _is_sha256(expected_digest) or current_digest != expected_digest:
        raise RuntimeError(
            "external preprocessing scalers do not match the synthesis checkpoint"
        )
    return provenance


def patient_level_split_from_manifest(
    scan_ids: Sequence[str],
    patient_by_scan: Mapping[str, str],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
    manifest_path: Optional[str],
    manifest_sha256: Optional[str] = None,
    protocol_profile: str = "paper-a4-adni-fmriprep-v1",
):
    """Load a fixed authority using explicit patient IDs.

    The fraction and seed parameters remain in the public signature for caller
    compatibility, but runtime never uses them to choose or write a split.
    """
    scans = [str(scan_id) for scan_id in scan_ids]
    if set(patient_by_scan) != set(scans):
        raise ValueError("patient map must cover exactly the scans being split")
    from .split import load_immutable_split_indices

    train, validation, test = load_immutable_split_indices(
        scans,
        patient_by_scan,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        protocol_profile=protocol_profile,
        expected_val_frac=float(val_frac),
        expected_test_frac=float(test_frac),
        expected_seed=int(seed),
    )
    partitions = (train, validation, test)
    if any(not partition for partition in partitions):
        raise ValueError("train, validation, and test partitions must all be non-empty")
    flattened = [index for partition in partitions for index in partition]
    if (
        len(flattened) != len(scans)
        or len(set(flattened)) != len(flattened)
        or set(flattened) != set(range(len(scans)))
    ):
        raise ValueError("patient split must cover every scan exactly once")
    patient_sets = [
        {str(patient_by_scan[scans[index]]).strip() for index in partition}
        for partition in partitions
    ]
    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise ValueError("patient split partitions are not patient-disjoint")
    return train, validation, test


def _configured_extractor_fingerprint(spec: Mapping, *, expected_name: str) -> dict:
    if not isinstance(spec, Mapping) or not bool(spec.get("enabled", True)):
        raise ValueError(f"the paper's {expected_name} extractor must be enabled")
    name = str(spec.get("name", "")).strip().lower()
    if name != expected_name:
        raise ValueError(f"extractor must be {expected_name}, got {name!r}")
    if expected_name == "brainlm":
        # Import lazily so dataset/protocol import does not initialize the large
        # frozen encoder. Construction validates the real BrainLM state shape,
        # official source/config/atlas bytes, and every reviewed displacement
        # authority record before its path-independent identity is checkpointed.
        from models.brainlm_context import configured_brainlm_identity

        identity = configured_brainlm_identity(spec)
        if not _is_recorded_extractor_fingerprint(identity, "brainlm"):
            raise RuntimeError("configured contextual BrainLM identity is invalid")
        return identity
    checkpoint = spec.get("checkpoint")
    environment_name = str(spec.get("checkpoint_env", "")).strip()
    if not checkpoint and environment_name:
        checkpoint = os.environ.get(environment_name)
    if not checkpoint:
        raise FileNotFoundError(
            f"{expected_name} checkpoint is required to fingerprint the paper protocol"
        )
    path = Path(str(checkpoint)).expanduser()
    actual_sha256 = sha256_file(path)
    expected_sha256 = spec.get("checkpoint_sha256")
    sha_environment_name = str(spec.get("checkpoint_sha256_env", "")).strip()
    if not expected_sha256 and sha_environment_name:
        expected_sha256 = os.environ.get(sha_environment_name)
    if not _is_sha256(expected_sha256):
        raise ValueError(
            f"{expected_name} checkpoint_sha256 must be configured as lowercase SHA-256"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{expected_name} checkpoint SHA-256 mismatch: expected "
            f"{expected_sha256}, found {actual_sha256}"
        )
    source_revision = spec.get("source_revision")
    revision_environment_name = str(spec.get("source_revision_env", "")).strip()
    if not source_revision and revision_environment_name:
        source_revision = os.environ.get(revision_environment_name)
    source_revision = str(source_revision or "").strip().lower()
    if len(source_revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError(
            f"{expected_name} source_revision must be a full immutable source commit"
        )
    adapter_contract = EXTRACTOR_ADAPTER_CONTRACTS[expected_name]
    configured_contract = spec.get("adapter_contract")
    if configured_contract is not None and configured_contract != adapter_contract:
        raise ValueError(
            f"{expected_name} adapter_contract must be {adapter_contract!r}"
        )
    fingerprint = {
        "name": expected_name,
        "checkpoint_sha256": actual_sha256,
        "source_revision": source_revision,
        "adapter_contract": adapter_contract,
    }
    if "source" in spec:
        source = str(spec.get("source", "")).strip()
        if not source:
            raise ValueError(f"{expected_name} source provenance must be non-empty")
        fingerprint["source"] = source
    return fingerprint


def _evaluation_extractor_fingerprint(spec: Mapping) -> dict:
    return _configured_extractor_fingerprint(spec, expected_name="slimbrain")


def build_run_artifact_identity(
    dataset,
    *,
    cohort_manifest_path: str,
    evaluation_extractor_spec: Optional[Mapping] = None,
    perceptual_extractor_spec: Optional[Mapping] = None,
    brainlm_authority_scope: Optional[Mapping] = None,
    target_scan_ids: Optional[Sequence[str]] = None,
    require_dataset_target_match: bool = True,
) -> dict:
    """Bind resume to all structural caches and only authorized target bytes.

    ``Connect4PrecomputedDataset`` validates every embedded source/artifact hash
    before this function is called. Hashing each complete cache metadata record
    binds the checkpoint to those validated contents without rereading all node
    arrays. The dataset supplies a recursively validated target identity only
    for the exact target allowlist. Every other scan is recorded as
    target-unopened, allowing this identity to reproduce when sealed targets do
    not exist on the training host. Evaluation-extractor identity may be deferred
    so training never opens or initializes the final-test SLIM-Brain artifact.
    """
    scans = [str(scan_id) for scan_id in dataset.scan_ids]
    if target_scan_ids is None:
        dataset_targets = getattr(dataset, "target_scan_ids", None)
        targets = set(scans if dataset_targets is None else dataset_targets)
    else:
        if isinstance(target_scan_ids, (str, bytes)):
            raise TypeError("target_scan_ids must be a sequence of scan IDs")
        target_values = [str(scan_id).strip() for scan_id in target_scan_ids]
        if any(not scan_id for scan_id in target_values):
            raise ValueError("target_scan_ids cannot contain an empty scan ID")
        if len(set(target_values)) != len(target_values):
            raise ValueError("target_scan_ids cannot contain duplicate scan IDs")
        targets = set(target_values)
    unknown_targets = sorted(targets - set(scans))
    if unknown_targets:
        raise ValueError(
            "target_scan_ids contains scans outside the structural cohort: "
            f"{unknown_targets[:10]}"
        )
    dataset_targets = getattr(dataset, "target_scan_ids", None)
    if (
        require_dataset_target_match
        and dataset_targets is not None
        and set(dataset_targets) != targets
    ):
        raise RuntimeError(
            "artifact target allowlist differs from the dataset target-access "
            "contract"
        )

    per_scan = {}
    structural_artifact_identities = {}
    target_artifact_identities = {}
    conditioning_identity = None
    for scan_id in scans:
        metadata_path = (
            dataset.precomputed_dir / "hypergraphs" / f"{scan_id}_metadata.json"
        )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"cannot read validated structural artifact provenance for {scan_id}"
            ) from exc
        record = {
            "cache_metadata_sha256": canonical_sha256(metadata),
            "target_access": "sealed-unopened",
        }
        admitted_metadata = getattr(
            dataset, "_validated_cache_metadata_sha256", None
        )
        if (
            isinstance(admitted_metadata, Mapping)
            and admitted_metadata.get(scan_id)
            != record["cache_metadata_sha256"]
        ):
            raise RuntimeError(
                f"{scan_id} structural metadata changed after validation"
            )
        structural_artifact_identities[scan_id] = {
            "cache_metadata_sha256": record["cache_metadata_sha256"],
        }
        if scan_id in targets:
            recovery_identities = getattr(dataset, "target_artifact_identities", {})
            if getattr(dataset, "recovery_profile", False):
                target_identity = _validated_native_target_artifact_identity(
                    recovery_identities.get(scan_id), scan_id=scan_id
                )
                if target_identity[
                    "structural_alignment_authority_sha256"
                ] != getattr(dataset, "native_alignment_authority_sha256", None):
                    raise RuntimeError(
                        f"{scan_id} native target identity uses another structural "
                        "alignment authority"
                    )
                target_roles = getattr(dataset, "target_scan_roles", None)
                if isinstance(target_roles, Mapping) and target_roles.get(
                    scan_id
                ) != target_identity["role"]:
                    raise RuntimeError(
                        f"{scan_id} native target identity uses another split role"
                    )
                target_artifact_identities[scan_id] = target_identity
                record.update(
                    {
                        "target_access": "authenticated-allowlist",
                        "target_identity_sha256": target_identity[
                            "fingerprint_sha256"
                        ],
                        "target_output_sha256": target_identity[
                            "target_output_sha256"
                        ],
                    }
                )
            else:
                target_identity = recovery_identities.get(scan_id)
                if not isinstance(target_identity, Mapping):
                    raise RuntimeError(
                        f"{scan_id} has no admitted paper-target identity"
                    )
                unsigned_identity = dict(target_identity)
                recorded_fingerprint = unsigned_identity.pop(
                    "fingerprint_sha256", None
                )
                if (
                    set(target_identity)
                    != {
                        "format",
                        "scan_id",
                        "target_sidecar_sha256",
                        "target_output_sha256",
                        "functional_validity_mask_path",
                        "functional_validity_mask_sha256",
                        "functional_validity_mask_size_bytes",
                        "functional_validity_mask_contract",
                        "fingerprint_sha256",
                    }
                    or target_identity.get("format")
                    != PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA
                    or target_identity.get("scan_id") != scan_id
                    or not _is_sha256(
                        target_identity.get("target_sidecar_sha256")
                    )
                    or not _is_sha256(
                        target_identity.get("target_output_sha256")
                    )
                    or not isinstance(
                        target_identity.get("functional_validity_mask_path"), str
                    )
                    or not Path(
                        target_identity["functional_validity_mask_path"]
                    ).is_absolute()
                    or not _is_sha256(
                        target_identity.get("functional_validity_mask_sha256")
                    )
                    or isinstance(
                        target_identity.get("functional_validity_mask_size_bytes"),
                        bool,
                    )
                    or not isinstance(
                        target_identity.get("functional_validity_mask_size_bytes"),
                        int,
                    )
                    or target_identity["functional_validity_mask_size_bytes"] < 1
                    or target_identity.get("functional_validity_mask_contract")
                    != TARGET_VALIDITY_MASK_CONTRACT
                    or recorded_fingerprint
                    != canonical_sha256(unsigned_identity)
                ):
                    raise RuntimeError(
                        f"{scan_id} admitted paper-target identity differs"
                    )
                target_identity = dict(target_identity)
                output_sha256 = target_identity["target_output_sha256"]
                target_artifact_identities[scan_id] = target_identity
                record.update(
                    {
                        "target_access": "authenticated-allowlist",
                        "target_identity_sha256": target_identity[
                            "fingerprint_sha256"
                        ],
                        "target_output_sha256": output_sha256,
                        "functional_validity_mask_sha256": target_identity[
                            "functional_validity_mask_sha256"
                        ],
                    }
                )
        per_scan[scan_id] = record
        scan_conditioning = conditioning_identity_from_cache_sources(
            metadata.get("source_fingerprint")
        )
        if conditioning_identity is None:
            conditioning_identity = scan_conditioning
        elif scan_conditioning != conditioning_identity:
            raise RuntimeError(
                "validated cache cohort uses inconsistent conditioning artifacts"
            )
    if conditioning_identity is None:
        raise RuntimeError("validated cache cohort has no conditioning identity")
    conditioning_identity_sha256 = canonical_sha256(conditioning_identity)
    grid_contract = getattr(dataset, "common_grid_contract", None)
    common_grid_digest = (
        grid_contract.get("contract_sha256")
        if isinstance(grid_contract, Mapping)
        else None
    )
    native_alignment_digest = getattr(
        dataset, "native_alignment_authority_sha256", None
    )
    if _is_sha256(common_grid_digest) == _is_sha256(native_alignment_digest):
        raise RuntimeError(
            "validated training dataset must have exactly one spatial authority"
        )
    spatial_authority = (
        {
            "profile": "paper-common-grid",
            "common_grid_contract_sha256": common_grid_digest,
        }
        if _is_sha256(common_grid_digest)
        else {
            "profile": "non-certified-native-stage-b",
            "native_alignment_authority_sha256": native_alignment_digest,
        }
    )
    spatial_authority_sha256 = canonical_sha256(spatial_authority)
    scaler_dir = getattr(dataset, "scaler_dir", None)
    if scaler_dir is None:
        raise RuntimeError("validated training dataset has no scaler directory")
    scaler_identity_sha256 = canonical_sha256(directory_file_sha256(scaler_dir))
    perceptual_fingerprint = _configured_extractor_fingerprint(
        perceptual_extractor_spec, expected_name="brainlm"
    )
    recovery_profile = bool(getattr(dataset, "recovery_profile", False))
    if recovery_profile and targets:
        if not isinstance(brainlm_authority_scope, Mapping):
            raise RuntimeError(
                "recovery training requires a split-bound BrainLM authority scope"
            )
        scope = dict(brainlm_authority_scope)
        claimed_scope_sha256 = scope.pop("record_sha256", None)
        target_roles = getattr(dataset, "target_scan_roles", None)
        if not isinstance(target_roles, Mapping):
            raise RuntimeError("recovery dataset has no target role authority")
        expected_role_items = [
            {"scan_id": scan_id, "role": target_roles.get(scan_id)}
            for scan_id in sorted(targets)
        ]
        mni_authority = perceptual_fingerprint.get("mni_authority")
        if (
            set(brainlm_authority_scope)
            != {
                "schema",
                "authority_file_sha256",
                "authority_record_sha256",
                "authority_ordered_scan_ids_sha256",
                "split_assignment_sha256",
                "split_identity_sha256",
                "target_scan_roles_sha256",
                "authorized_scan_count",
                "train_scan_count",
                "development_scan_count",
                "sealed_scan_count",
                "sealed_scan_ids_sha256",
                "sealed_scans_authorized",
                "functional_target_bytes_opened",
                "record_sha256",
            }
            or scope.get("schema")
            != "connect4-brainlm-authority-split-scope-v1"
            or claimed_scope_sha256 != canonical_sha256(scope)
            or scope.get("authorized_scan_count") != len(targets)
            or scope.get("train_scan_count")
            != sum(item["role"] == "train" for item in expected_role_items)
            or scope.get("development_scan_count")
            != sum(
                item["role"] == "development-validation"
                for item in expected_role_items
            )
            or scope.get("sealed_scan_count") != len(scans) - len(targets)
            or scope.get("target_scan_roles_sha256")
            != canonical_sha256(expected_role_items)
            or scope.get("sealed_scans_authorized") is not False
            or scope.get("functional_target_bytes_opened") is not False
            or not isinstance(mni_authority, Mapping)
            or scope.get("authority_file_sha256")
            != mni_authority.get("file_sha256")
            or scope.get("authority_record_sha256")
            != mni_authority.get("authority_record_sha256")
            or scope.get("authority_ordered_scan_ids_sha256")
            != mni_authority.get("ordered_scan_ids_sha256")
        ):
            raise RuntimeError(
                "split-bound BrainLM authority scope differs from the run artifact"
            )
        resolved_brainlm_scope = dict(brainlm_authority_scope)
    elif brainlm_authority_scope is not None:
        raise RuntimeError(
            "BrainLM authority scope is permitted only for recovery target admission"
        )
    else:
        resolved_brainlm_scope = None
    if evaluation_extractor_spec is None:
        evaluation_fingerprint = dict(DEFERRED_EVALUATION_EXTRACTOR_IDENTITY)
    else:
        evaluation_fingerprint = _evaluation_extractor_fingerprint(
            evaluation_extractor_spec
        )
    ordered_targets = sorted(targets)
    ordered_unopened = sorted(set(scans) - targets)
    target_access_contract = {
        "schema": "connect4-training-target-access-v1",
        "authenticated_target_scan_ids": ordered_targets,
        "authenticated_target_scan_ids_sha256": canonical_sha256(ordered_targets),
        "sealed_target_scan_ids": ordered_unopened,
        "sealed_target_scan_ids_sha256": canonical_sha256(ordered_unopened),
        "sealed_targets_opened": False,
    }
    target_artifact_identities_sha256 = canonical_sha256(
        {scan_id: target_artifact_identities[scan_id] for scan_id in ordered_targets}
    )
    structural_artifact_identities_sha256 = canonical_sha256(
        structural_artifact_identities
    )
    payload = {
        "cohort_manifest_sha256": sha256_file(Path(cohort_manifest_path).expanduser()),
        "scaler_identity_sha256": scaler_identity_sha256,
        "conditioning_identity": conditioning_identity,
        "conditioning_identity_sha256": conditioning_identity_sha256,
        "common_grid_contract_sha256": common_grid_digest,
        "native_alignment_authority_sha256": native_alignment_digest,
        "spatial_authority": spatial_authority,
        "spatial_authority_sha256": spatial_authority_sha256,
        "evaluation_extractor": evaluation_fingerprint,
        "perceptual_extractor": perceptual_fingerprint,
        "brainlm_authority_scope": resolved_brainlm_scope,
        "target_access_contract": target_access_contract,
        "target_artifact_identities_sha256": (
            target_artifact_identities_sha256
        ),
        # Training and sealed inference have different target-access maps, but
        # they must consume byte-identical structural cache records.
        "structural_artifact_identities_sha256": (
            structural_artifact_identities_sha256
        ),
        "scans": per_scan,
    }
    return {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "sha256": canonical_sha256(payload),
        "num_scans": len(per_scan),
        "num_authenticated_targets": len(ordered_targets),
        "num_sealed_targets_unopened": len(ordered_unopened),
        "scaler_identity_sha256": scaler_identity_sha256,
        "common_grid_contract_sha256": common_grid_digest,
        "native_alignment_authority_sha256": native_alignment_digest,
        "spatial_authority": spatial_authority,
        "spatial_authority_sha256": spatial_authority_sha256,
        "conditioning_identity": conditioning_identity,
        "conditioning_identity_sha256": conditioning_identity_sha256,
        "evaluation_extractor": payload["evaluation_extractor"],
        "perceptual_extractor": payload["perceptual_extractor"],
        "brainlm_authority_scope": resolved_brainlm_scope,
        "target_access_contract": target_access_contract,
        "target_artifact_identities_sha256": (
            target_artifact_identities_sha256
        ),
        "structural_artifact_identities_sha256": (
            structural_artifact_identities_sha256
        ),
    }


def validate_training_scaler_provenance(
    scaler_dir: str,
    training_scan_ids: Sequence[str],
) -> dict:
    """Require scaler provenance to name the exact fixed training partition."""
    if not scaler_dir:
        raise ValueError(
            "data.scaler_dir is required for precomputed feature provenance"
        )
    provenance_path = Path(scaler_dir).expanduser() / "training_partition.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"missing or invalid training-only scaler provenance: {provenance_path}"
        ) from exc
    expected = sorted(map(str, training_scan_ids))
    if not expected or any(not scan_id for scan_id in expected):
        raise ValueError("training scaler partition must contain non-empty scan IDs")
    if len(set(expected)) != len(expected):
        raise ValueError("training scaler partition contains duplicate scan IDs")
    fitted = provenance.get("fitted_scan_ids")
    if (
        provenance.get("schema") != "connect4-training-only-scalers-v1"
        or provenance.get("training_partition_scan_ids") != expected
        or fitted != expected
    ):
        raise RuntimeError(
            "feature scalers were not fitted exclusively against the exact current "
            "fixed training partition"
        )
    return provenance


def build_split_identity(
    scan_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    cohort_by_scan: Mapping[str, str],
    patient_by_scan: Mapping[str, str],
) -> dict:
    """Create a stable identity for the exact cohort and three-way assignment."""
    scans = [str(scan_id) for scan_id in scan_ids]
    partitions = {
        "train": [int(index) for index in train_indices],
        "validation": [int(index) for index in validation_indices],
        "test": [int(index) for index in test_indices],
    }
    if any(not indices for indices in partitions.values()):
        raise ValueError("train, validation, and test partitions must all be non-empty")
    flattened = [index for indices in partitions.values() for index in indices]
    if (
        len(flattened) != len(scans)
        or len(set(flattened)) != len(flattened)
        or set(flattened) != set(range(len(scans)))
    ):
        raise ValueError("split indices must cover every scan exactly once")
    if set(cohort_by_scan) != set(scans):
        raise ValueError("cohort map must cover exactly the scans used by the split")
    if set(patient_by_scan) != set(scans):
        raise ValueError("patient map must cover exactly the scans used by the split")
    if any(not str(patient_by_scan[scan_id]).strip() for scan_id in scans):
        raise ValueError("patient IDs used by the split must be non-empty")
    partition_patients = {
        split: {str(patient_by_scan[scans[index]]).strip() for index in indices}
        for split, indices in partitions.items()
    }
    if (
        partition_patients["train"] & partition_patients["validation"]
        or partition_patients["train"] & partition_patients["test"]
        or partition_patients["validation"] & partition_patients["test"]
    ):
        raise ValueError("split partitions are not patient-disjoint")

    payload = {
        split: [
            {
                "scan_id": scans[index],
                "patient_id": str(patient_by_scan[scans[index]]).strip(),
                "cohort": _canonical_cohort(cohort_by_scan[scans[index]]),
            }
            for index in sorted(indices, key=lambda item: scans[item])
        ]
        for split, indices in partitions.items()
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    cohort_counts = Counter(
        _canonical_cohort(value) for value in cohort_by_scan.values()
    )
    patient_counts = {
        split: len({record["patient_id"] for record in records})
        for split, records in payload.items()
    }
    synthesis_patient_ids = sorted(
        {
            record["patient_id"]
            for split in ("train", "validation")
            for record in payload[split]
        }
    )
    synthesis_cohorts = sorted(cohort_counts)
    synthesis_evidence = {
        "synthesis_patient_ids": synthesis_patient_ids,
        "synthesis_cohorts": synthesis_cohorts,
    }
    return {
        "format": SPLIT_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "num_scans": len(scans),
        "partition_counts": {name: len(values) for name, values in partitions.items()},
        "partition_patient_counts": patient_counts,
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "partitions": payload,
        **synthesis_evidence,
        "synthesis_evidence_sha256": canonical_sha256(synthesis_evidence),
    }


__all__ = [
    "PAPER_TRAINING_COHORTS",
    "PAPER_COHORT_SCAN_COUNTS",
    "COHORT_MANIFEST_SCHEMA",
    "SPLIT_IDENTITY_SCHEMA",
    "TRAINING_CHECKPOINT_FORMAT",
    "RUN_ARTIFACT_IDENTITY_SCHEMA",
    "EXTERNAL_CACHE_PROTOCOL_SCHEMA",
    "CONDITIONING_ARTIFACT_SCHEMA",
    "PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA",
    "TARGET_VALIDITY_MASK_CONTRACT",
    "PAPER_VOXEL_SIZE_MM",
    "CLINICAL_MODERNBERT_MODEL_ID",
    "EXTRACTOR_ADAPTER_CONTRACTS",
    "TrainingCohortEvidence",
    "SynthesisExclusionEvidence",
    "conditioning_identity_from_cache_sources",
    "validate_fixed_protocol_config",
    "validate_training_cohort_manifest",
    "validate_synthesis_split_identity",
    "validate_unseen_cohort_manifest",
    "load_completed_synthesis_checkpoint",
    "build_external_cache_protocol_context",
    "validate_external_scaler_provenance",
    "patient_level_split_from_manifest",
    "build_run_artifact_identity",
    "validate_training_scaler_provenance",
    "build_split_identity",
]
