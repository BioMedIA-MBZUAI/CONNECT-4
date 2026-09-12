"""One-way, post-seal evaluation of the fixed CONNECT-4 held-out cohort.

This module deliberately has no dependency on CONNECT-4 training, inference,
checkpoint loading, or the synthesis model.  Importing it is also deliberately
free of Torch and metric/model modules.  Its trust boundary has two phases:

1. authenticate the complete target-blind prediction-set seal, all 34 subject
   records, and every prediction byte; then
2. authenticate a closed externally pinned execution authority and all 34
   Stage-B target publications/bytes, then validate every exact native pair.

Only after the complete cohort crosses both phases may metric, feature-model,
quality, texture, or visualization code be imported.  Production execution is
implemented in the separately staged :mod:`eval.postseal_execution` module;
this public module retains the byte-verifier and publication primitives.

The evaluator never crops, registers, resamples, or otherwise changes an fMRI
volume.  The recovery protocol's 64x80x64 grid is training padding only; final
paired evaluation is on the exact unpadded 61x73x61x128, 3-mm, TR=3-s grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass, field as dataclass_field
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any
import zlib

import nibabel as nib
import numpy as np

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_MAPPING_SHA256,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
)

# These literal protocol identifiers are needed to parse evidence without
# importing the Torch-backed metric module.  The first metric import validates
# that the staged metric module exposes exactly the same identifiers.
METRIC_DEFINITIONS_SOURCE = (
    "eval/postseal_metrics.py::{mse,ssim3d,voxel_correlation,roi_correlation,"
    "frame_to_frame_correlation,psnr}"
)
SLIMBRAIN_AUTHORITY_SCHEMA = "connect4-postseal-slimbrain-evaluation-authority-v1"
SLIMBRAIN_AVAILABLE_METRICS_SCHEMA = (
    "connect4-postseal-distributional-metrics-available-v2"
)
SLIMBRAIN_ADAPTER_CONTRACT = "connect4-slimbrain-full-volume-bctdhw-v2"
SLIMBRAIN_INPUT_SEMANTICS = {
    "axes": "B,C,T,D,H,W",
    "channels": 1,
    "value_domain": "stored-normalized-[0,1]",
    "spatial_or_temporal_resampling": False,
    "masking": "none-full-authenticated-volume",
    "batching": "one-subject-at-a-time-after-complete-cohort-authentication",
}

PREDICTION_SET_FORMAT = "connect4-target-blind-prediction-set-v1"
PREDICTION_FORMAT = "connect4-target-blind-prediction-v1"
EVALUATION_SCHEMA = "connect4-postseal-heldout-evaluation-v5"
SUBJECT_EVALUATION_SCHEMA = "connect4-postseal-heldout-subject-evaluation-v2"
PUBLICATION_RECEIPT_SCHEMA = "connect4-postseal-heldout-publication-receipt-v1"
DATA_QUALITY_SCHEMA = "connect4-postseal-heldout-data-quality-v4"
TEXTURE_AUDIT_SCHEMA = "connect4-post-inference-texture-audit-v3"
TEXTURE_GATE_SCHEMA = "connect4-texture-retention-quality-gate-v3"
STAGE_B_VERIFICATION_SCHEMA = (
    "connect4-native-stage-b-sealed-target-completed-set-verification-v1"
)
STAGE_B_COMPLETED_SET_SCHEMA = "connect4-native-stage-b-sealed-target-completed-set-v1"
STAGE_B_WORKLIST_SCHEMA = "connect4-native-stage-b-sealed-target-worklist-v1"
TARGET_VALIDITY_MASK_CONTRACT = "connect4-observed-bold-support-v1"

EXPECTED_SHAPE = (61, 73, 61, 128)
EXPECTED_SPATIAL_SHAPE = EXPECTED_SHAPE[:3]
EXPECTED_VOXEL_SIZE_MM = (3.0, 3.0, 3.0)
EXPECTED_TR_SECONDS = 3.0
EXPECTED_AXIS_CODES = ("R", "A", "S")
_TEXTURE_RETENTION_METRICS = (
    "voxel_scale_detail_rms_ratio",
    "gradient_rms_ratio",
    "laplacian_rms_ratio",
    "local_high_frequency_power_ratio",
)
_TEXTURE_RETENTION_DISPLAY_NAMES = {
    "voxel_scale_detail_rms_ratio": "voxel-scale detail RMS retention",
    "gradient_rms_ratio": "gradient RMS retention",
    "laplacian_rms_ratio": "Laplacian RMS retention",
    "local_high_frequency_power_ratio": "local high-frequency power retention",
}
_TEXTURE_ANTI_GAMING_METRICS = (
    "aggregate_detail_correlation",
    "near_nyquist_tail_power_ratio",
)
_TEXTURE_ANTI_GAMING_DISPLAY_NAMES = {
    "aggregate_detail_correlation": "voxel-scale detail correlation",
    "near_nyquist_tail_power_ratio": ("predicted/real near-Nyquist tail-power ratio"),
}
# This release policy is part of the authenticated evaluator source, rather than
# accepted from a self-signed per-result manifest.  Any policy change therefore
# requires a newly staged runtime/source authority.
_TEXTURE_RETENTION_MINIMUM = 0.80
_TEXTURE_DETAIL_CORRELATION_MINIMUM = 0.40
_TEXTURE_NEAR_NYQUIST_MAXIMUM = 1.25
_TEXTURE_NEAR_NYQUIST_BOUNDARY = 0.60
_TEXTURE_FAILURE_EXIT_CODE = 2
_MIN_STRUCTURAL_BRAIN_SUPPORT_FRACTION = 0.05
_MAX_STRUCTURAL_BRAIN_SUPPORT_FRACTION = 0.60
_MAX_STRUCTURAL_BRAIN_BOUNDARY_FRACTION = 0.01
_MIN_TEXTURE_PATCH_EXTENT_VOXELS = 9
_DISTRIBUTIONAL_UNAVAILABLE_SCHEMA = (
    "connect4-postseal-distributional-metrics-unavailable-v1"
)
_DISTRIBUTIONAL_UNAVAILABLE_REASON = (
    "a closed pinned evaluation feature-model loader/runtime is not yet qualified; "
    "FID and Inception Score are unavailable and were not substituted or fabricated"
)
_DISTRIBUTIONAL_AVAILABLE_FIELDS = {
    "schema",
    "available",
    "evaluation_feature_model_loaded",
    "fid",
    "inception_score",
    "sample_count",
    "ordered_scan_ids_sha256",
    "input_bindings",
    "model_authority",
    "feature_commitments",
    "feature_artifacts",
    "feature_mask",
    "spatial_or_temporal_resampling_performed",
    "loaded_after_complete_prediction_and_target_authentication",
    "evaluation_only",
    "authorizes_training",
    "authorizes_model_selection",
    "authorizes_candidate_selection",
    "authorizes_checkpoint_selection",
    "record_sha256",
}
_DISTRIBUTIONAL_TENSOR_ARTIFACT_FIELDS = {
    "relative_path",
    "sha256",
    "size_bytes",
    "tensor_sha256",
    "shape",
    "dtype",
}
_DISTRIBUTIONAL_TENSOR_PATHS = {
    "generated_features": "distributional_evidence/generated_features.f32le",
    "real_features": "distributional_evidence/real_features.f32le",
    "generated_logits": "distributional_evidence/generated_logits.f32le",
}
_DISTRIBUTIONAL_TENSOR_DTYPE = "float32-little-endian-c-order"
_SUBJECT_EVALUATION_FIELDS = {
    "schema",
    "scan_id",
    "prediction",
    "prediction_record",
    "target",
    "target_validity_mask",
    "structural_brain_mask",
    "structural_labels",
    "stage_b_publication",
    "stage_b_success_receipt",
    "native_preprocessing_receipt",
    "pair_contract",
    "quality",
    "visualization_outputs",
    "texture_audit",
    "fmri_resampling_performed",
    "evaluation_can_authorize_training_or_selection",
    "record_sha256",
}
_ROOT_EVALUATION_FIELDS = {
    "schema",
    "status",
    "role",
    "scan_count",
    "ordered_scan_ids",
    "ordered_scan_ids_sha256",
    "sealed_role_csv_sha256",
    "prediction_set_seal",
    "stage_b_completed_set",
    "stage_b_verifier_source",
    "stage_b_verifier_dependencies",
    "stage_b_verification",
    "checkpoint_bindings",
    "subjects",
    "paper_metrics_cohort_mean",
    "distributional_metrics",
    "data_quality_summary",
    "fitness_for_use",
    "texture_release_gate",
    "quality_pass_count",
    "quality_fail_count",
    "texture_release_gate_pass_count",
    "texture_release_gate_fail_count",
    "output_inventory",
    "output_inventory_sha256",
    "implementation_sources",
    "production_runtime_and_scheduler_evidence",
    "production_evaluator_authenticated_before_import",
    "prediction_set_fully_authenticated_before_any_target_access",
    "all_target_publications_and_bytes_authenticated_before_metrics",
    "all_targets_authenticated_before_texture_audits",
    "all_34_texture_retention_and_anti_gaming_gates_passed",
    "exact_native_grid_no_fmri_resampling",
    "independent_structural_mask_and_roi_labels_only",
    "synthesis_model_or_checkpoint_loaded",
    "evaluation_feature_model_loaded",
    "training_or_inference_entrypoint_imported",
    "authorizes_training",
    "authorizes_model_selection",
    "authorizes_candidate_selection",
    "authorizes_checkpoint_selection",
    "authorizes_prediction",
    "authorizes_inference",
    "can_change_checkpoint_or_prediction_set",
    "record_sha256",
}
_PUBLICATION_RECEIPT_FIELDS = {
    "schema",
    "status",
    "destination",
    "evaluation",
    "output_inventory_sha256",
    "all_34_subjects_complete",
    "all_34_texture_retention_and_anti_gaming_gates_passed",
    "published_after_all_subjects_complete",
    "no_overwrite",
    "evaluation_feedback_to_training_or_selection",
    "record_sha256",
}
EXPECTED_SCAN_IDS = (
    "B17815888_016",
    "B17815888_027",
    "B17815888_066",
    "B17815888_072",
    "B18042045_009",
    "B18042045_027",
    "B18042045_048",
    "B34710358_004",
    "B34710358_009",
    "B34710358_027",
    "B38881515_004",
    "B38881515_009",
    "B38881515_999",
    "B45524671_004",
    "B46475534_048",
    "B46475534_066",
    "B50601165_009",
    "B50601165_027",
    "B50601165_072",
    "B63018964_027",
    "B65964797_004",
    "B65964797_047",
    "B65964797_066",
    "B67864280_009",
    "B67864280_027",
    "B67864280_048",
    "B68798784_004",
    "B79109115_004",
    "B79109115_027",
    "B79109115_999",
    "B85876120_004",
    "B85876120_009",
    "B85876120_027",
    "B85876120_999",
)
EXPECTED_SCAN_IDS_SHA256 = (
    "d1021320cda7c0eb0b54270a768cdf58e77859e11f6291a02eb163a4de41a8ae"
)
SEALED_ROLE_CSV_SHA256 = (
    "b5c04f0ccf4dac74b266233b4705be85c49dbbe589b64e6166aaa8449c0ecfa8"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_NATIVE_PATH_TYPE = type(Path())
_QUALITY_RATIO_SPECS = (
    (
        "robust_range_ratio",
        "spatial",
        "robust_range_ratio",
        "spatial_robust_range_ratio",
        "Robust q95-q05 range",
    ),
    (
        "gradient_rms_ratio",
        "spatial",
        "gradient_rms_ratio",
        "spatial_gradient_ratio",
        "Gradient RMS",
    ),
    (
        "laplacian_rms_ratio",
        "spatial",
        "laplacian_rms_ratio",
        "spatial_laplacian_ratio",
        "Laplacian RMS",
    ),
    (
        "high_frequency_rms_ratio",
        "spatial",
        "high_frequency_rms_ratio",
        "spatial_high_frequency_ratio",
        "High-pass RMS",
    ),
    (
        "dynamic_high_frequency_rms_ratio",
        "spatial",
        "dynamic_high_frequency_rms_ratio",
        "dynamic_high_frequency_ratio",
        "Frame-varying high-pass RMS",
    ),
    (
        "temporal_variance_ratio",
        "temporal",
        "temporal_variance_ratio",
        "temporal_variance_ratio",
        "Temporal variance",
    ),
    (
        "dvars_ratio",
        "temporal",
        "dvars_ratio",
        "dvars_ratio",
        "DVARS",
    ),
    (
        "non_dc_power_ratio",
        "temporal",
        "dynamic_power_ratio",
        "dynamic_power_ratio",
        "Non-DC power",
    ),
    (
        "effective_rank_ratio",
        "temporal",
        "effective_rank_ratio",
        "effective_rank_ratio",
        "Effective rank",
    ),
)
_QUALITY_RATIO_POLICY_FIELDS = {
    "robust_range_ratio": (
        "min_spatial_robust_range_ratio",
        "max_spatial_robust_range_ratio",
    ),
    "gradient_rms_ratio": (
        "min_spatial_gradient_ratio",
        "max_spatial_gradient_ratio",
    ),
    "laplacian_rms_ratio": (
        "min_spatial_laplacian_ratio",
        "max_spatial_laplacian_ratio",
    ),
    "high_frequency_rms_ratio": (
        "min_spatial_high_frequency_ratio",
        "max_spatial_high_frequency_ratio",
    ),
    "dynamic_high_frequency_rms_ratio": (
        "min_dynamic_high_frequency_ratio",
        "max_dynamic_high_frequency_ratio",
    ),
    "temporal_variance_ratio": (
        "min_temporal_variance_ratio",
        "max_temporal_variance_ratio",
    ),
    "dvars_ratio": ("min_dvars_ratio", "max_dvars_ratio"),
    "non_dc_power_ratio": ("min_dynamic_power_ratio", "max_dynamic_power_ratio"),
    "effective_rank_ratio": (
        "min_effective_rank_ratio",
        "max_effective_rank_ratio",
    ),
}
_ROOT_SEAL_FIELDS = {
    "format",
    "status",
    "split",
    "num_subjects",
    "subjects",
    "subjects_sha256",
    "sealed_targets_opened",
    "paired_quality_gate_run",
    "synthesis_checkpoint_sha256",
    "training_run_artifact_identity_sha256",
    "training_target_artifact_identities_sha256",
    "spatial_authority_sha256",
    "record_sha256",
}
_SUBJECT_RECORD_FIELDS = {
    "format",
    "status",
    "scan_id",
    "protocol_profile",
    "target_opened_before_prediction",
    "paired_quality_gate_run",
    "prediction",
    "synthesis_checkpoint_sha256",
    "training_run_artifact_identity_sha256",
    "training_target_artifact_identities_sha256",
    "spatial_authority_sha256",
    "record_sha256",
}
_CHECKPOINT_BINDING_FIELDS = (
    "synthesis_checkpoint_sha256",
    "training_run_artifact_identity_sha256",
    "training_target_artifact_identities_sha256",
    "spatial_authority_sha256",
)


class PostsealEvaluationError(RuntimeError):
    """Raised when post-seal evidence or publication is not exact."""


class PostsealPublicationUncertainError(PostsealEvaluationError):
    """A no-replace rename happened but its durable commit could not be proven."""


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the repository's signed-record convention."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARACTERS)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise PostsealEvaluationError(f"{label} is not a lowercase SHA-256")
    assert type(value) is str
    return value


def _float32_tensor_sha256(value: np.ndarray) -> str:
    """Hash a finite FP32 matrix with an explicit shape/dtype domain separator."""

    array = np.asarray(value, dtype=np.float32, order="C")
    if array.ndim != 2 or not np.isfinite(array).all():
        raise PostsealEvaluationError(
            "distributional evidence tensor must be one finite FP32 matrix"
        )
    header = json.dumps(
        {"shape": list(array.shape), "dtype": "float32"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _safe_scan_id(value: object, *, label: str = "scan ID") -> str:
    if type(value) is not str:
        raise PostsealEvaluationError(f"{label} is not a built-in string")
    scan_id = value
    if (
        scan_id not in EXPECTED_SCAN_IDS
        or Path(scan_id).name != scan_id
        or "/" in scan_id
        or "\\" in scan_id
    ):
        raise PostsealEvaluationError(f"{label} is not in the fixed sealed cohort")
    return scan_id


def _canonical_path(
    path: str | Path,
    *,
    label: str,
    directory: bool = False,
    must_exist: bool = True,
) -> Path:
    if type(path) not in {str, _NATIVE_PATH_TYPE}:
        raise PostsealEvaluationError(
            f"{label} must use an exact built-in string or native Path type"
        )
    raw_path = str(path)
    if not raw_path or "$" in raw_path or "~" in raw_path:
        raise PostsealEvaluationError(
            f"{label} must be direct and cannot use environment or user indirection"
        )
    requested = Path(raw_path)
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise PostsealEvaluationError(f"{label} must be an absolute lexical path")
    if not must_exist:
        parent = requested.parent
        canonical_parent = _canonical_path(
            parent, label=f"{label} parent", directory=True
        )
        if requested.parent != canonical_parent or requested.name in {"", ".", ".."}:
            raise PostsealEvaluationError(f"{label} aliases another path")
        try:
            requested.lstat()
        except FileNotFoundError:
            return requested
        raise FileExistsError(f"{label} already exists: {requested}")
    try:
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PostsealEvaluationError(f"{label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or resolved != requested:
        raise PostsealEvaluationError(f"{label} aliases another path")
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise PostsealEvaluationError(f"{label} is not a directory")
    elif not stat.S_ISREG(metadata.st_mode):
        raise PostsealEvaluationError(f"{label} is not a regular file")
    return requested


@dataclass(frozen=True)
class FileSnapshot:
    """Content identity captured through one stable, no-follow file descriptor."""

    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int

    def descriptor(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _snapshot_file(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    capture: bool = False,
    maximum_bytes: int | None = None,
) -> tuple[FileSnapshot, bytes | None]:
    source = _canonical_path(path, label=label)
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(expected_sha256, label=f"expected {label} SHA-256")
    )
    try:
        initial = source.lstat()
    except OSError as exc:
        raise PostsealEvaluationError(f"{label} cannot be inspected") from exc
    if (
        not stat.S_ISREG(initial.st_mode)
        or stat.S_ISLNK(initial.st_mode)
        or initial.st_nlink != 1
    ):
        raise PostsealEvaluationError(
            f"{label} must be one non-symlink, non-hardlinked regular file"
        )
    if initial.st_size < 1:
        raise PostsealEvaluationError(f"{label} is empty")
    if expected_size is not None and initial.st_size != expected_size:
        raise PostsealEvaluationError(f"{label} size differs")
    if maximum_bytes is not None and initial.st_size > maximum_bytes:
        raise PostsealEvaluationError(f"{label} exceeds its byte limit")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino, before.st_size)
            != (initial.st_dev, initial.st_ino, initial.st_size)
        ):
            raise PostsealEvaluationError(f"{label} changed before authentication")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            total += len(block)
            if maximum_bytes is not None and total > maximum_bytes:
                raise PostsealEvaluationError(f"{label} exceeds its byte limit")
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        after = os.fstat(descriptor)
        current = source.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if total != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(current, field)
        for field in stable_fields
    ):
        raise PostsealEvaluationError(f"{label} changed during authentication")
    observed = digest.hexdigest()
    if expected is not None and observed != expected:
        raise PostsealEvaluationError(f"{label} SHA-256 differs")
    snapshot = FileSnapshot(
        path=source,
        sha256=observed,
        size_bytes=total,
        device=before.st_dev,
        inode=before.st_ino,
    )
    return snapshot, None if chunks is None else b"".join(chunks)


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostsealEvaluationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PostsealEvaluationError(
                    f"{label} contains non-finite JSON constant {token}"
                )
            ),
        )
    except PostsealEvaluationError:
        raise
    except Exception as exc:
        raise PostsealEvaluationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PostsealEvaluationError(f"{label} must be a JSON object")
    return value


def _signed_json_snapshot(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    expected_schema_key: str | None = None,
    expected_schema: str | None = None,
) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot, payload = _snapshot_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        capture=True,
        maximum_bytes=64 * 1024 * 1024,
    )
    assert payload is not None
    value = _parse_json_object(payload, label=label)
    unsigned = dict(value)
    recorded = unsigned.pop("record_sha256", None)
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise PostsealEvaluationError(f"{label} signed record differs")
    if (
        expected_schema_key is not None
        and value.get(expected_schema_key) != expected_schema
    ):
        raise PostsealEvaluationError(f"{label} schema/format differs")
    return value, snapshot


@dataclass(frozen=True)
class EvaluationPins:
    """Externally frozen prediction/checkpoint bindings fixed before target access."""

    prediction_set_seal_sha256: str
    synthesis_checkpoint_sha256: str
    training_run_artifact_identity_sha256: str
    training_target_artifact_identities_sha256: str
    spatial_authority_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "prediction_set_seal_sha256",
            *_CHECKPOINT_BINDING_FIELDS,
        ):
            value = object.__getattribute__(self, name)
            _require_sha256(value, label=name)

    def checkpoint_bindings(self) -> dict[str, str]:
        return {
            name: object.__getattribute__(self, name)
            for name in _CHECKPOINT_BINDING_FIELDS
        }


@dataclass(frozen=True)
class _EvaluationPinSnapshot:
    prediction_set_seal_sha256: str
    checkpoint_items: tuple[tuple[str, str], ...]

    def checkpoint_bindings(self) -> dict[str, str]:
        return dict(self.checkpoint_items)


@dataclass(frozen=True)
class _PublicationAuthorityPins:
    prediction_set_seal_path: str
    prediction_set_seal_sha256: str
    stage_b_completed_set_path: str
    stage_b_completed_set_sha256: str
    stage_b_verifier_source_path: str
    stage_b_verifier_source_sha256: str
    stage_b_verifier_dependency_paths: tuple[str, ...]
    stage_b_verifier_dependency_sha256s: tuple[str, ...]
    execution_authority_path: str
    execution_authority_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("prediction-set seal", self.prediction_set_seal_sha256),
            ("Stage-B completed set", self.stage_b_completed_set_sha256),
            ("Stage-B verifier source", self.stage_b_verifier_source_sha256),
            ("production execution authority", self.execution_authority_sha256),
        ):
            _require_sha256(value, label=f"external {label} SHA-256")
        for label, value in (
            ("prediction-set seal", self.prediction_set_seal_path),
            ("Stage-B completed set", self.stage_b_completed_set_path),
            ("Stage-B verifier source", self.stage_b_verifier_source_path),
            ("production execution authority", self.execution_authority_path),
        ):
            if (
                type(value) is not str
                or not value.startswith("/")
                or "$" in value
                or "~" in value
                or Path(value) != Path(os.path.abspath(value))
            ):
                raise PostsealEvaluationError(f"external {label} path type differs")
        if (
            type(self.stage_b_verifier_dependency_paths) is not tuple
            or len(self.stage_b_verifier_dependency_paths) != 2
            or type(self.stage_b_verifier_dependency_sha256s) is not tuple
            or len(self.stage_b_verifier_dependency_paths)
            != len(self.stage_b_verifier_dependency_sha256s)
        ):
            raise PostsealEvaluationError(
                "external Stage-B verifier dependency pins are incomplete"
            )
        for index, (path, digest) in enumerate(
            zip(
                self.stage_b_verifier_dependency_paths,
                self.stage_b_verifier_dependency_sha256s,
            )
        ):
            if type(path) is not str:
                raise PostsealEvaluationError(
                    f"external Stage-B verifier dependency {index} path type differs"
                )
            if (
                not path.startswith("/")
                or "$" in path
                or "~" in path
                or Path(path) != Path(os.path.abspath(path))
            ):
                raise PostsealEvaluationError(
                    f"external Stage-B verifier dependency {index} path differs"
                )
            _require_sha256(
                digest,
                label=f"external Stage-B verifier dependency {index} SHA-256",
            )


def _snapshot_evaluation_pins(pins: EvaluationPins) -> _EvaluationPinSnapshot:
    """Copy exact immutable scalar pins without executing caller-defined code."""
    if type(pins) is not EvaluationPins:
        raise PostsealEvaluationError(
            "evaluation pins must be an exact EvaluationPins instance"
        )
    values: dict[str, str] = {}
    for name in ("prediction_set_seal_sha256", *_CHECKPOINT_BINDING_FIELDS):
        value = object.__getattribute__(pins, name)
        if type(value) is not str:
            raise PostsealEvaluationError(f"{name} must be an exact built-in string")
        values[name] = _require_sha256(value, label=name)
    return _EvaluationPinSnapshot(
        prediction_set_seal_sha256=values["prediction_set_seal_sha256"],
        checkpoint_items=tuple(
            (name, values[name]) for name in _CHECKPOINT_BINDING_FIELDS
        ),
    )


def _validate_checkpoint_bindings(
    value: Mapping[str, Any], pins: _EvaluationPinSnapshot, *, label: str
) -> None:
    expected = pins.checkpoint_bindings()
    for name, digest in expected.items():
        observed = _require_sha256(value.get(name), label=f"{label} {name}")
        if observed != digest:
            raise PostsealEvaluationError(f"{label} {name} differs")


def _authenticate_prediction_set_impl(
    prediction_root: Path,
    *,
    pins: _EvaluationPinSnapshot,
) -> None:
    """Authenticate every sealed prediction byte without exposing an intermediate."""
    root = _canonical_path(prediction_root, label="prediction root", directory=True)
    root_children = {path.name: path for path in root.iterdir()}
    if set(root_children) != {"subjects", "prediction_set_seal.json"}:
        raise PostsealEvaluationError(
            "prediction root is partial or contains unsealed extra entries"
        )
    for name, path in root_children.items():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PostsealEvaluationError(
                f"prediction root entry {name!r} is a symlink"
            )
    seal_path = root / "prediction_set_seal.json"
    seal, _ = _signed_json_snapshot(
        seal_path,
        label="prediction-set seal",
        expected_sha256=pins.prediction_set_seal_sha256,
        expected_schema_key="format",
        expected_schema=PREDICTION_SET_FORMAT,
    )
    if set(seal) != _ROOT_SEAL_FIELDS:
        raise PostsealEvaluationError("prediction-set seal fields differ")
    subject_bindings = seal.get("subjects")
    if not isinstance(subject_bindings, list):
        raise PostsealEvaluationError("prediction-set subjects are not a list")
    scan_ids = [
        item.get("scan_id") if isinstance(item, dict) else None
        for item in subject_bindings
    ]
    if (
        seal.get("status") != "SEALED_TARGET_BLIND_PREDICTION_SET"
        or seal.get("split") != "test"
        or seal.get("num_subjects") != len(EXPECTED_SCAN_IDS)
        or tuple(scan_ids) != EXPECTED_SCAN_IDS
        or seal.get("subjects_sha256") != canonical_sha256(subject_bindings)
        or canonical_sha256(scan_ids) != EXPECTED_SCAN_IDS_SHA256
        or seal.get("sealed_targets_opened") is not False
        or seal.get("paired_quality_gate_run") is not False
    ):
        raise PostsealEvaluationError(
            "prediction-set seal is partial, reordered, replaced, or not target blind"
        )
    _validate_checkpoint_bindings(seal, pins, label="prediction-set seal")
    for expected_scan_id, binding in zip(EXPECTED_SCAN_IDS, subject_bindings):
        if not isinstance(binding, dict) or set(binding) != {
            "scan_id",
            "prediction_sha256",
            "subject_record_sha256",
        }:
            raise PostsealEvaluationError("prediction subject binding fields differ")
        scan_id = _safe_scan_id(binding.get("scan_id"))
        if scan_id != expected_scan_id:
            raise PostsealEvaluationError("prediction subject order differs")
        prediction_sha = _require_sha256(
            binding.get("prediction_sha256"),
            label=f"{scan_id} bound prediction SHA-256",
        )
        record_sha = _require_sha256(
            binding.get("subject_record_sha256"),
            label=f"{scan_id} bound subject-record SHA-256",
        )
        subject_dir = _canonical_path(
            root / "subjects" / scan_id,
            label=f"{scan_id} prediction directory",
            directory=True,
        )
        record, _ = _signed_json_snapshot(
            subject_dir / "prediction.json",
            label=f"{scan_id} prediction record",
            expected_schema_key="format",
            expected_schema=PREDICTION_FORMAT,
        )
        if set(record) != _SUBJECT_RECORD_FIELDS:
            raise PostsealEvaluationError(f"{scan_id} prediction-record fields differ")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict) or set(prediction) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "shape",
            "repetition_time_seconds",
        }:
            raise PostsealEvaluationError(f"{scan_id} prediction descriptor differs")
        expected_relative = f"subjects/{scan_id}/prediction.nii.gz"
        if (
            record.get("status") != "SEALED_TARGET_BLIND_PREDICTION"
            or record.get("scan_id") != scan_id
            or not isinstance(record.get("protocol_profile"), str)
            or not record.get("protocol_profile")
            or record.get("target_opened_before_prediction") is not False
            or record.get("paired_quality_gate_run") is not False
            or record.get("record_sha256") != record_sha
            or prediction.get("relative_path") != expected_relative
            or prediction.get("sha256") != prediction_sha
            or prediction.get("shape") != list(EXPECTED_SHAPE)
            or prediction.get("repetition_time_seconds") != EXPECTED_TR_SECONDS
            or isinstance(prediction.get("size_bytes"), bool)
            or not isinstance(prediction.get("size_bytes"), int)
            or prediction.get("size_bytes") < 1
        ):
            raise PostsealEvaluationError(
                f"{scan_id} prediction record is inconsistent with the root seal"
            )
        _validate_checkpoint_bindings(record, pins, label=f"{scan_id} prediction")
        prediction_path = root / expected_relative
        _snapshot_file(
            prediction_path,
            label=f"{scan_id} prediction NIfTI",
            expected_sha256=prediction_sha,
            expected_size=prediction["size_bytes"],
            capture=False,
        )
    # Reject extra entries anywhere under subjects, including alternate records
    # or replacement predictions that are not committed by the root seal.
    expected_relative_files = {
        f"{scan_id}/prediction.json" for scan_id in EXPECTED_SCAN_IDS
    } | {f"{scan_id}/prediction.nii.gz" for scan_id in EXPECTED_SCAN_IDS}
    observed_relative_files: set[str] = set()
    subjects_root = _canonical_path(
        root / "subjects", label="prediction subjects root", directory=True
    )
    observed_subject_directories = {
        path.name
        for path in subjects_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if observed_subject_directories != set(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError(
            "prediction subject directories are partial, extra, or replaced"
        )
    for path in subjects_root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PostsealEvaluationError("prediction tree contains a symbolic link")
        if path.is_file():
            observed_relative_files.add(path.relative_to(subjects_root).as_posix())
        elif not path.is_dir():
            raise PostsealEvaluationError("prediction tree contains a special file")
    if observed_relative_files != expected_relative_files:
        raise PostsealEvaluationError(
            "prediction tree is partial or contains unsealed extra/replacement files"
        )
    return None


def _header_tr_seconds(image: nib.spatialimages.SpatialImage, *, label: str) -> float:
    zooms = image.header.get_zooms()
    if len(zooms) < 4:
        raise PostsealEvaluationError(f"{label} has no temporal zoom")
    unit = image.header.get_xyzt_units()[1]
    if unit != "sec":
        raise PostsealEvaluationError(f"{label} time unit is not seconds")
    value = float(zooms[3])
    if value != EXPECTED_TR_SECONDS:
        raise PostsealEvaluationError(f"{label} TR differs from 3 seconds")
    return value


def _strict_nifti_values(
    path: Path,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    is_labels: bool = False,
    is_binary_mask: bool = False,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    if is_labels and is_binary_mask:
        raise PostsealEvaluationError(f"{label} cannot be both labels and mask")
    if type(path) is not _NATIVE_PATH_TYPE or not path.name.endswith(".nii.gz"):
        raise PostsealEvaluationError(
            f"{label} must be an exact gzip-compressed single-file NIfTI path"
        )
    snapshot, compressed = _snapshot_file(
        path,
        label=f"{label} container",
        capture=True,
        maximum_bytes=512 * 1024 * 1024,
    )
    assert compressed is not None
    if compressed[:3] != b"\x1f\x8b\x08":
        raise PostsealEvaluationError(f"{label} is not an exact gzip container")
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        uncompressed = decompressor.decompress(compressed)
        uncompressed += decompressor.flush()
    except zlib.error as exc:
        raise PostsealEvaluationError(f"{label} gzip stream is invalid") from exc
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise PostsealEvaluationError(
            f"{label} must contain exactly one complete gzip member"
        )
    try:
        image = nib.Nifti1Image.from_bytes(uncompressed)
    except Exception as exc:
        raise PostsealEvaluationError(f"{label} is not a valid NIfTI") from exc
    if type(image) is not nib.Nifti1Image or len(image.header.extensions) != 0:
        raise PostsealEvaluationError(
            f"{label} must be one extension-free NIfTI-1 image"
        )
    expected_dtype = (
        np.dtype(np.int16)
        if is_labels
        else np.dtype(np.uint8)
        if is_binary_mask
        else np.dtype(np.float32)
    )
    if np.dtype(image.header.get_data_dtype()) != expected_dtype:
        raise PostsealEvaluationError(f"{label} stored data type differs")
    expected_payload_size = int(image.dataobj.offset) + (
        int(np.prod(expected_shape, dtype=np.int64)) * expected_dtype.itemsize
    )
    if len(uncompressed) != expected_payload_size:
        raise PostsealEvaluationError(
            f"{label} NIfTI payload has trailing or missing bytes"
        )
    slope = float(getattr(image.dataobj, "slope", 1.0))
    intercept = float(getattr(image.dataobj, "inter", 0.0))
    if slope != 1.0 or intercept != 0.0:
        raise PostsealEvaluationError(f"{label} uses ambiguous header scaling")
    if tuple(int(value) for value in image.shape) != expected_shape:
        raise PostsealEvaluationError(f"{label} shape differs: {image.shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12
    ):
        raise PostsealEvaluationError(f"{label} affine is invalid")
    axis_codes = tuple(nib.aff2axcodes(affine))
    if axis_codes != EXPECTED_AXIS_CODES:
        raise PostsealEvaluationError(
            f"{label} orientation {axis_codes} differs from RAS+"
        )
    zooms = tuple(float(value) for value in image.header.get_zooms())
    if zooms[:3] != EXPECTED_VOXEL_SIZE_MM:
        raise PostsealEvaluationError(f"{label} is not exactly 3-mm isotropic")
    if image.header.get_xyzt_units()[0] != "mm":
        raise PostsealEvaluationError(f"{label} spatial unit is not millimetres")
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if (
        int(qform_code) > 0
        and int(sform_code) > 0
        and not np.array_equal(
            np.asarray(qform, dtype=np.float64), np.asarray(sform, dtype=np.float64)
        )
    ):
        raise PostsealEvaluationError(f"{label} qform/sform affines disagree")
    if len(expected_shape) == 4:
        _header_tr_seconds(image, label=label)
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise PostsealEvaluationError(f"{label} contains NaN or infinity")
    minimum = float(values.min())
    maximum = float(values.max())
    if is_labels:
        rounded = np.rint(values)
        if (
            minimum < 0.0
            or maximum > float(np.iinfo(np.int16).max)
            or not np.array_equal(values, rounded.astype(np.float32))
        ):
            raise PostsealEvaluationError(
                f"{label} is not a nonnegative integer structural label map"
            )
        values = rounded.astype(np.int16, copy=False)
        positive_labels = np.unique(values[values > 0])
        if positive_labels.size < 2:
            raise PostsealEvaluationError(
                f"{label} has fewer than two nonempty structural regions"
            )
    elif is_binary_mask:
        if not np.logical_or(values == 0.0, values == 1.0).all():
            raise PostsealEvaluationError(f"{label} is not exactly binary")
        values = values.astype(bool, copy=False)
    elif minimum < 0.0 or maximum > 1.0:
        raise PostsealEvaluationError(f"{label} is outside the [0,1] domain")
    return (
        image,
        values,
        {
            "shape": list(expected_shape),
            "affine": affine.tolist(),
            "axis_codes": list(axis_codes),
            "voxel_sizes_mm": list(zooms[:3]),
            "tr_seconds": EXPECTED_TR_SECONDS if len(expected_shape) == 4 else None,
            "minimum": minimum,
            "maximum": maximum,
            "container": "gzip-nifti1-single-file-no-extensions",
            "container_sha256": snapshot.sha256,
            "gzip_member_count": 1,
            "uncompressed_size_bytes": len(uncompressed),
            "stored_dtype": expected_dtype.name,
            "header_scaling": {"slope": slope, "intercept": intercept},
        },
    )


def _derive_exact_target_validity_mask(
    real: np.ndarray,
    structural_brain_mask: np.ndarray,
) -> np.ndarray:
    """Derive the one permitted post-seal target-supervision support.

    The held-out target is available only after the caller has authenticated the
    complete target-blind prediction set and every held-out target byte.  This
    helper has no pathname access and cannot open a target by itself.
    """
    values = np.asarray(real)
    structural = np.asarray(structural_brain_mask)
    if values.ndim != 4 or structural.shape != values.shape[:3]:
        raise PostsealEvaluationError(
            "target-validity inputs differ from the paired native grid"
        )
    if not np.isfinite(values).all():
        raise PostsealEvaluationError(
            "target-validity derivation received non-finite fMRI"
        )
    if not np.logical_or(structural == 0, structural == 1).all():
        raise PostsealEvaluationError(
            "target-validity structural support is not exactly binary"
        )
    validity = np.any(values != 0.0, axis=-1)
    structural = structural.astype(bool, copy=False)
    if not bool(validity.any()):
        raise PostsealEvaluationError("held-out target has empty observed BOLD support")
    if bool((validity & ~structural).any()):
        raise PostsealEvaluationError(
            "held-out target has observed BOLD outside structural brain support"
        )
    return validity


def validate_exact_pair(
    real_path: str | Path,
    predicted_path: str | Path,
    structural_brain_mask_path: str | Path,
    structural_labels_path: str | Path,
    *,
    protocol_profile: str,
) -> dict[str, Any]:
    """Enforce the no-resampling paired geometry and [0,1] data contract."""
    real_image, real_values, real = _strict_nifti_values(
        Path(real_path), label="real fMRI", expected_shape=EXPECTED_SHAPE
    )
    predicted_image, _, predicted = _strict_nifti_values(
        Path(predicted_path), label="predicted fMRI", expected_shape=EXPECTED_SHAPE
    )
    mask_image, brain_mask, structural_brain_mask = _strict_nifti_values(
        Path(structural_brain_mask_path),
        label="independent structural brain mask",
        expected_shape=EXPECTED_SPATIAL_SHAPE,
        is_binary_mask=True,
    )
    labels_image, labels, structural = _strict_nifti_values(
        Path(structural_labels_path),
        label="independent structural labels",
        expected_shape=EXPECTED_SPATIAL_SHAPE,
        is_labels=True,
    )
    real_affine = np.asarray(real_image.affine, dtype=np.float64)
    predicted_affine = np.asarray(predicted_image.affine, dtype=np.float64)
    mask_affine = np.asarray(mask_image.affine, dtype=np.float64)
    labels_affine = np.asarray(labels_image.affine, dtype=np.float64)
    if not np.array_equal(real_affine, predicted_affine):
        raise PostsealEvaluationError(
            "real/predicted affines are not exactly equal; fMRI resampling is forbidden"
        )
    if not np.array_equal(real_affine, mask_affine):
        raise PostsealEvaluationError(
            "structural-mask affine is not exactly equal to the paired fMRI affine"
        )
    if not np.array_equal(real_affine, labels_affine):
        raise PostsealEvaluationError(
            "structural-label affine is not exactly equal to the paired fMRI affine"
        )
    foreground_count = int(np.count_nonzero(brain_mask))
    foreground_fraction = foreground_count / float(brain_mask.size)
    boundary = np.zeros(brain_mask.shape, dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    boundary_count = int(np.count_nonzero(brain_mask & boundary))
    boundary_fraction = (
        float(boundary_count) / float(foreground_count) if foreground_count else 1.0
    )
    coordinates = np.argwhere(brain_mask)
    extents = (
        tuple(int(value) for value in (coordinates.max(0) - coordinates.min(0) + 1))
        if coordinates.size
        else (0, 0, 0)
    )
    minimum_extents = tuple(
        min(_MIN_TEXTURE_PATCH_EXTENT_VOXELS, max(1, dimension - 2))
        for dimension in EXPECTED_SPATIAL_SHAPE
    )
    roi_foreground = labels > 0
    roi_outside_support = int(np.count_nonzero(roi_foreground & ~brain_mask))
    if (
        foreground_fraction < _MIN_STRUCTURAL_BRAIN_SUPPORT_FRACTION
        or foreground_fraction > _MAX_STRUCTURAL_BRAIN_SUPPORT_FRACTION
        or boundary_fraction > _MAX_STRUCTURAL_BRAIN_BOUNDARY_FRACTION
        or any(value < minimum for value, minimum in zip(extents, minimum_extents))
        or roi_outside_support != 0
    ):
        raise PostsealEvaluationError(
            "independent structural brain support is incomplete, implausible, or "
            "does not contain every ROI label voxel"
        )
    structural_brain_mask.update(
        {
            "foreground_voxel_count": foreground_count,
            "foreground_fraction": foreground_fraction,
            "boundary_foreground_voxel_count": boundary_count,
            "boundary_foreground_fraction": boundary_fraction,
            "foreground_extent_voxels": list(extents),
            "roi_foreground_outside_mask_voxel_count": roi_outside_support,
            "binary_exact": True,
        }
    )
    target_validity = _derive_exact_target_validity_mask(real_values, brain_mask)
    target_validity_count = int(np.count_nonzero(target_validity))
    configured_roi_labels = np.unique(labels[labels > 0])
    if type(protocol_profile) is not str or not protocol_profile:
        raise PostsealEvaluationError("post-seal protocol profile is invalid")
    target_validity_contract_name = (
        TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(protocol_profile)
    )
    if target_validity_contract_name is None:
        raise PostsealEvaluationError(
            f"post-seal protocol profile {protocol_profile!r} has no target-validity "
            "contract"
        )
    if tuple(int(value) for value in configured_roi_labels) != tuple(
        CANONICAL_ROI_LABEL_IDS
    ):
        missing = sorted(
            set(CANONICAL_ROI_LABEL_IDS) - set(map(int, configured_roi_labels.tolist()))
        )
        extra = sorted(
            set(map(int, configured_roi_labels.tolist())) - set(CANONICAL_ROI_LABEL_IDS)
        )
        raise PostsealEvaluationError(
            "post-seal structural labels differ from the canonical 32-ROI mapping; "
            f"missing={missing}, extra={extra}"
        )
    missing_validity_roi_labels = [
        int(label)
        for label in configured_roi_labels
        if not bool((target_validity & (labels == label)).any())
    ]
    if missing_validity_roi_labels:
        raise PostsealEvaluationError(
            "every configured structural ROI must have non-empty target-validity "
            f"support; missing labels: {missing_validity_roi_labels}"
        )
    target_validity_contract = {
        "contract": target_validity_contract_name,
        "derivation": "exact-nonzero-support-across-final-stored-target-time-series",
        "shape": list(EXPECTED_SPATIAL_SHAPE),
        "foreground_voxel_count": target_validity_count,
        "foreground_fraction_of_structural_support": float(
            target_validity_count / foreground_count
        ),
        "structural_foreground_voxel_count": foreground_count,
        "structural_voxels_excluded_from_target_metrics": (
            foreground_count - target_validity_count
        ),
        "outside_structural_brain_voxel_count": 0,
        "configured_structural_roi_count": int(configured_roi_labels.size),
        "all_configured_structural_rois_have_target_validity_support": True,
        "equals_exact_final_nonzero_support": True,
        "exactly_binary": True,
        "fmri_resampled": False,
    }
    return {
        "real": real,
        "predicted": predicted,
        "structural_brain_mask": structural_brain_mask,
        "target_validity_mask": target_validity_contract,
        "structural_labels": structural,
        "affines_exactly_equal": True,
        "orientations_exactly_equal": True,
        "shape_exactly_equal": True,
        "tr_exactly_equal_seconds": EXPECTED_TR_SECONDS,
        "fmri_resampled": False,
        "mask_or_labels_derived_from_real_or_prediction": False,
        "positive_structural_label_count": int(np.unique(labels[labels > 0]).size),
        "protocol_profile": protocol_profile,
        "canonical_roi_mapping_sha256": CANONICAL_ROI_MAPPING_SHA256,
    }


def _copy_authenticated_file(
    snapshot: FileSnapshot, destination: Path, *, label: str
) -> None:
    current, payload = _snapshot_file(
        snapshot.path,
        label=label,
        expected_sha256=snapshot.sha256,
        expected_size=snapshot.size_bytes,
        capture=True,
    )
    if (current.device, current.inode) != (snapshot.device, snapshot.inode):
        raise PostsealEvaluationError(f"{label} inode changed after set authentication")
    assert payload is not None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed, _ = _snapshot_file(
        destination,
        label=f"private {label} copy",
        expected_sha256=snapshot.sha256,
        expected_size=snapshot.size_bytes,
    )
    if observed.sha256 != snapshot.sha256:
        raise PostsealEvaluationError(f"private {label} copy differs")


def _distributional_metrics_unavailable() -> dict[str, Any]:
    """Return exact evidence for an intentionally omitted SLIM-Brain authority."""
    return {
        "schema": _DISTRIBUTIONAL_UNAVAILABLE_SCHEMA,
        "available": False,
        "evaluation_feature_model_loaded": False,
        "fid": None,
        "inception_score": None,
        "reason": _DISTRIBUTIONAL_UNAVAILABLE_REASON,
    }


def _load_metric_module_after_complete_pair_preflight():
    """Load the Torch-backed metric surface only after the cohort barrier."""

    from . import postseal_metrics

    expected_constants = {
        "METRIC_DEFINITIONS_SOURCE": METRIC_DEFINITIONS_SOURCE,
        "SLIMBRAIN_ADAPTER_CONTRACT": SLIMBRAIN_ADAPTER_CONTRACT,
        "SLIMBRAIN_AUTHORITY_SCHEMA": SLIMBRAIN_AUTHORITY_SCHEMA,
        "SLIMBRAIN_AVAILABLE_METRICS_SCHEMA": SLIMBRAIN_AVAILABLE_METRICS_SCHEMA,
        "SLIMBRAIN_INPUT_SEMANTICS": SLIMBRAIN_INPUT_SEMANTICS,
    }
    if any(
        getattr(postseal_metrics, name, None) != expected
        for name, expected in expected_constants.items()
    ):
        raise PostsealEvaluationError("post-seal metric protocol identity differs")
    return postseal_metrics


def _distributional_metrics_available(
    *,
    accumulator: Any,
    artifact_root: Path,
    prediction_set_seal_sha256: str,
    stage_b_completed_set_sha256: str,
) -> dict[str, Any]:
    """Persist and sign evidence from the exact authenticated accumulator only."""

    metric_module = _load_metric_module_after_complete_pair_preflight()
    if type(accumulator) is not metric_module.SynthesisMetricAccumulator:
        raise PostsealEvaluationError(
            "distributional evidence requires the exact authenticated accumulator"
        )
    root = _canonical_path(
        artifact_root, label="distributional artifact root", directory=True
    )
    authority = accumulator.authenticated_authority
    tensors = accumulator.distributional_tensors()
    if accumulator.num_samples != len(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError(
            "distributional evidence requires the complete sealed cohort"
        )
    commitments = accumulator.distributional_commitments()
    metrics = accumulator.compute()
    artifact_directory = root / "distributional_evidence"
    try:
        artifact_directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise PostsealEvaluationError(
            "distributional evidence directory already exists"
        ) from exc
    feature_artifacts: dict[str, dict[str, Any]] = {}
    for name, relative_path in _DISTRIBUTIONAL_TENSOR_PATHS.items():
        tensor = tensors[name].detach().to(device="cpu").float().contiguous()
        array = np.asarray(tensor.numpy(), dtype=np.dtype("<f4"), order="C")
        payload = array.tobytes(order="C")
        snapshot = _write_bytes_new(root / relative_path, payload)
        tensor_digest = _float32_tensor_sha256(array)
        commitment_name = f"{name}_sha256"
        if tensor_digest != commitments[commitment_name]:
            raise PostsealEvaluationError(
                f"distributional {name} tensor commitment differs before publication"
            )
        feature_artifacts[name] = {
            "relative_path": relative_path,
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "tensor_sha256": tensor_digest,
            "shape": list(array.shape),
            "dtype": _DISTRIBUTIONAL_TENSOR_DTYPE,
        }

    body = {
        "schema": SLIMBRAIN_AVAILABLE_METRICS_SCHEMA,
        "available": True,
        "evaluation_feature_model_loaded": True,
        "fid": float(metrics["fid"]),
        "inception_score": float(metrics["is"]),
        "sample_count": len(EXPECTED_SCAN_IDS),
        "ordered_scan_ids_sha256": EXPECTED_SCAN_IDS_SHA256,
        "input_bindings": {
            "prediction_set_seal_sha256": _require_sha256(
                prediction_set_seal_sha256,
                label="distributional prediction-set seal SHA-256",
            ),
            "stage_b_completed_set_sha256": _require_sha256(
                stage_b_completed_set_sha256,
                label="distributional Stage-B completed-set SHA-256",
            ),
        },
        "model_authority": dict(authority.evidence),
        "feature_commitments": dict(commitments),
        "feature_artifacts": feature_artifacts,
        "feature_mask": "none-full-authenticated-volume",
        "spatial_or_temporal_resampling_performed": False,
        "loaded_after_complete_prediction_and_target_authentication": True,
        "evaluation_only": True,
        "authorizes_training": False,
        "authorizes_model_selection": False,
        "authorizes_candidate_selection": False,
        "authorizes_checkpoint_selection": False,
    }
    result = dict(body)
    result["record_sha256"] = canonical_sha256(body)
    authority_file = result["model_authority"].get("authority", {})
    _validate_distributional_evidence(
        result,
        expected_authority_sha256=authority_file.get("raw_sha256"),
        artifact_root=root,
    )
    return result


def _validate_artifact_descriptor(value: object, *, label: str) -> None:
    if type(value) is not dict or set(value) != {"path", "sha256", "size_bytes"}:
        raise PostsealEvaluationError(f"{label} descriptor fields differ")
    path = value.get("path")
    if type(path) is not str or not path or not Path(path).is_absolute():
        raise PostsealEvaluationError(f"{label} path differs")
    _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    size = value.get("size_bytes")
    if type(size) is not int or size < 1:
        raise PostsealEvaluationError(f"{label} size differs")


def _immutable_distributional_artifact(
    descriptor: Mapping[str, Any], *, label: str
) -> FileSnapshot:
    """Reauthenticate one authority artifact without executing its model."""

    _validate_artifact_descriptor(descriptor, label=label)
    path_text = descriptor["path"]
    if "$" in path_text or "~" in path_text:
        raise PostsealEvaluationError(
            f"{label} path uses forbidden environment indirection"
        )
    path = Path(path_text)
    if path != Path(os.path.abspath(path)):
        raise PostsealEvaluationError(f"{label} path is not lexical")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PostsealEvaluationError(f"{label} cannot be inspected") from exc
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise PostsealEvaluationError(f"{label} is not frozen in mode 0444")
    snapshot, _ = _snapshot_file(
        path,
        label=label,
        expected_sha256=descriptor["sha256"],
        expected_size=descriptor["size_bytes"],
    )
    return snapshot


def _reauthenticate_distributional_authority(
    authority: Mapping[str, Any], *, expected_authority_sha256: str
) -> None:
    """Bind publication evidence to the actual externally pinned authority."""

    authority_file = authority["authority"]
    path_text = authority_file["path"]
    if "$" in path_text or "~" in path_text:
        raise PostsealEvaluationError(
            "SLIM-Brain authority path uses forbidden environment indirection"
        )
    path = Path(path_text)
    if path != Path(os.path.abspath(path)):
        raise PostsealEvaluationError("SLIM-Brain authority path is not lexical")
    if authority_file["size_bytes"] > 256 * 1024:
        raise PostsealEvaluationError("SLIM-Brain authority exceeds its byte limit")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PostsealEvaluationError(
            "SLIM-Brain authority cannot be inspected"
        ) from exc
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise PostsealEvaluationError("SLIM-Brain authority is not frozen in mode 0444")
    record, snapshot = _signed_json_snapshot(
        path,
        label="SLIM-Brain authority",
        expected_sha256=expected_authority_sha256,
        expected_size=authority_file["size_bytes"],
        expected_schema_key="schema",
        expected_schema=SLIMBRAIN_AUTHORITY_SCHEMA,
    )
    expected_record_fields = {
        "schema",
        "status",
        "model_name",
        "checkpoint",
        "source",
        "adapter",
        "feature_output",
        "logits_output",
        "runtime",
        "restrictions",
        "record_sha256",
    }
    if (
        set(record) != expected_record_fields
        or snapshot.sha256 != authority_file["raw_sha256"]
        or record.get("record_sha256") != authority_file["record_sha256"]
        or record.get("status") != "QUALIFIED_IMMUTABLE_EVALUATION_ONLY"
        or record.get("model_name") != "slimbrain"
        or record.get("checkpoint") != authority["checkpoint"]
        or record.get("source") != authority["source"]
        or record.get("adapter") != authority["adapter"]
        or record.get("feature_output") != authority["feature_output"]
        or record.get("logits_output") != authority["logits_output"]
        or record.get("runtime") != authority["runtime"]
        or record.get("restrictions")
        != {
            "environment_indirection": False,
            "generic_or_pickle_model_loading": False,
            "data_derived_feature_mask": False,
            "single_subject_distributional_metrics": False,
            "training_or_selection_feedback": False,
            "target_access_before_prediction_set_seal": False,
        }
    ):
        raise PostsealEvaluationError(
            "SLIM-Brain authority/publication cross-binding differs"
        )
    checkpoint = _immutable_distributional_artifact(
        authority["checkpoint"], label="SLIM-Brain checkpoint"
    )
    source_descriptor = {
        key: authority["source"][key] for key in ("path", "sha256", "size_bytes")
    }
    source = _immutable_distributional_artifact(
        source_descriptor, label="SLIM-Brain source manifest"
    )
    if (checkpoint.device, checkpoint.inode) == (source.device, source.inode):
        raise PostsealEvaluationError("SLIM-Brain source and checkpoint alias")
    adapter_source, _ = _snapshot_file(
        Path(__file__).with_name("postseal_metrics.py"),
        label="SLIM-Brain adapter implementation source",
    )
    if adapter_source.sha256 != authority["adapter"]["implementation_source_sha256"]:
        raise PostsealEvaluationError("SLIM-Brain adapter implementation differs")
    final_snapshot, _ = _snapshot_file(
        path,
        label="SLIM-Brain authority final authentication",
        expected_sha256=expected_authority_sha256,
        expected_size=authority_file["size_bytes"],
    )
    if (snapshot.device, snapshot.inode) != (
        final_snapshot.device,
        final_snapshot.inode,
    ):
        raise PostsealEvaluationError(
            "SLIM-Brain authority changed during verification"
        )


def _validate_distributional_artifact_descriptors(
    value: object,
    *,
    feature_dimension: int,
    class_count: int,
) -> dict[str, dict[str, Any]]:
    if type(value) is not dict or set(value) != set(_DISTRIBUTIONAL_TENSOR_PATHS):
        raise PostsealEvaluationError(
            "distributional feature-artifact role set differs"
        )
    expected_shapes = {
        "generated_features": [len(EXPECTED_SCAN_IDS), feature_dimension],
        "real_features": [len(EXPECTED_SCAN_IDS), feature_dimension],
        "generated_logits": [len(EXPECTED_SCAN_IDS), class_count],
    }
    result: dict[str, dict[str, Any]] = {}
    for name, relative_path in _DISTRIBUTIONAL_TENSOR_PATHS.items():
        descriptor = value.get(name)
        if (
            type(descriptor) is not dict
            or set(descriptor) != _DISTRIBUTIONAL_TENSOR_ARTIFACT_FIELDS
            or descriptor.get("relative_path") != relative_path
            or descriptor.get("shape") != expected_shapes[name]
            or descriptor.get("dtype") != _DISTRIBUTIONAL_TENSOR_DTYPE
        ):
            raise PostsealEvaluationError(
                f"distributional {name} artifact descriptor differs"
            )
        expected_size = int(np.prod(expected_shapes[name], dtype=np.int64)) * 4
        if descriptor.get("size_bytes") != expected_size:
            raise PostsealEvaluationError(
                f"distributional {name} artifact size differs"
            )
        _require_sha256(
            descriptor.get("sha256"),
            label=f"distributional {name} artifact SHA-256",
        )
        _require_sha256(
            descriptor.get("tensor_sha256"),
            label=f"distributional {name} tensor SHA-256",
        )
        result[name] = dict(descriptor)
    return result


def _recompute_distributional_metrics_from_artifacts(
    value: Mapping[str, Any], *, artifact_root: Path
) -> None:
    """Reopen immutable arrays and independently recompute all FID/IS evidence."""

    root = _canonical_path(
        artifact_root, label="distributional artifact root", directory=True
    )
    authority = value["model_authority"]
    feature_dimension = authority["feature_output"]["dimension"]
    class_count = authority["logits_output"]["class_count"]
    descriptors = _validate_distributional_artifact_descriptors(
        value.get("feature_artifacts"),
        feature_dimension=feature_dimension,
        class_count=class_count,
    )
    arrays: dict[str, np.ndarray] = {}
    for name, relative_path in _DISTRIBUTIONAL_TENSOR_PATHS.items():
        descriptor = descriptors[name]
        snapshot, payload = _snapshot_file(
            root / relative_path,
            label=f"published distributional {name}",
            expected_sha256=descriptor["sha256"],
            expected_size=descriptor["size_bytes"],
            capture=True,
            maximum_bytes=descriptor["size_bytes"],
        )
        assert payload is not None
        if snapshot.path != root / relative_path:
            raise PostsealEvaluationError(
                f"distributional {name} artifact escaped its publication root"
            )
        array = np.frombuffer(payload, dtype=np.dtype("<f4")).astype(
            np.float32, copy=True
        )
        array = array.reshape(tuple(descriptor["shape"]))
        if not np.isfinite(array).all():
            raise PostsealEvaluationError(
                f"distributional {name} artifact contains NaN or infinity"
            )
        observed_tensor_sha = _float32_tensor_sha256(array)
        commitment_name = f"{name}_sha256"
        if (
            observed_tensor_sha != descriptor["tensor_sha256"]
            or observed_tensor_sha != value["feature_commitments"][commitment_name]
        ):
            raise PostsealEvaluationError(
                f"distributional {name} tensor commitment differs"
            )
        arrays[name] = array
    metric_module = _load_metric_module_after_complete_pair_preflight()
    observed_fid = metric_module.frechet_distance_from_features(
        arrays["generated_features"], arrays["real_features"]
    )
    observed_is = metric_module.inception_score_from_logits(arrays["generated_logits"])
    if not np.isclose(observed_fid, float(value["fid"]), rtol=1e-12, atol=1e-12):
        raise PostsealEvaluationError("published FID differs from immutable features")
    if not np.isclose(
        observed_is,
        float(value["inception_score"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise PostsealEvaluationError(
            "published Inception Score differs from immutable logits"
        )


def _validate_distributional_evidence(
    value: object,
    *,
    expected_authority_sha256: str | None = None,
    artifact_root: Path | None = None,
) -> bool:
    """Validate unavailable or externally pinned available FID/IS evidence."""

    unavailable = _distributional_metrics_unavailable()
    if type(value) is dict and value == unavailable:
        if expected_authority_sha256 is not None:
            raise PostsealEvaluationError(
                "SLIM-Brain authority pin was supplied for unavailable metrics"
            )
        return False
    if type(value) is not dict or set(value) != _DISTRIBUTIONAL_AVAILABLE_FIELDS:
        raise PostsealEvaluationError("distributional metric evidence fields differ")
    unsigned = dict(value)
    recorded = unsigned.pop("record_sha256", None)
    if (
        value.get("schema") != SLIMBRAIN_AVAILABLE_METRICS_SCHEMA
        or value.get("available") is not True
        or value.get("evaluation_feature_model_loaded") is not True
        or recorded != canonical_sha256(unsigned)
    ):
        raise PostsealEvaluationError("available distributional metric record differs")
    fid = value.get("fid")
    inception_score = value.get("inception_score")
    if (
        type(fid) not in {int, float}
        or type(inception_score) not in {int, float}
        or not np.isfinite(float(fid))
        or not np.isfinite(float(inception_score))
        or float(fid) < 0.0
        or float(inception_score) < 1.0
        or value.get("sample_count") != len(EXPECTED_SCAN_IDS)
        or value.get("ordered_scan_ids_sha256") != EXPECTED_SCAN_IDS_SHA256
        or value.get("feature_mask") != "none-full-authenticated-volume"
        or value.get("spatial_or_temporal_resampling_performed") is not False
        or value.get("loaded_after_complete_prediction_and_target_authentication")
        is not True
        or value.get("evaluation_only") is not True
        or any(
            value.get(field) is not False
            for field in (
                "authorizes_training",
                "authorizes_model_selection",
                "authorizes_candidate_selection",
                "authorizes_checkpoint_selection",
            )
        )
    ):
        raise PostsealEvaluationError(
            "available distributional metric contract differs"
        )
    bindings = value.get("input_bindings")
    if type(bindings) is not dict or set(bindings) != {
        "prediction_set_seal_sha256",
        "stage_b_completed_set_sha256",
    }:
        raise PostsealEvaluationError("distributional input bindings differ")
    for name, digest in bindings.items():
        _require_sha256(digest, label=f"distributional {name}")
    authority = value.get("model_authority")
    if type(authority) is not dict or set(authority) != {
        "authority",
        "model_name",
        "checkpoint",
        "source",
        "adapter",
        "feature_output",
        "logits_output",
        "runtime",
        "frozen_eval",
        "torchscript_only",
    }:
        raise PostsealEvaluationError("SLIM-Brain model-authority evidence differs")
    authority_file = authority.get("authority")
    if type(authority_file) is not dict or set(authority_file) != {
        "path",
        "raw_sha256",
        "record_sha256",
        "size_bytes",
    }:
        raise PostsealEvaluationError("SLIM-Brain authority-file evidence differs")
    raw_authority_sha = _require_sha256(
        authority_file.get("raw_sha256"), label="SLIM-Brain raw authority SHA-256"
    )
    _require_sha256(
        authority_file.get("record_sha256"),
        label="SLIM-Brain authority record SHA-256",
    )
    if expected_authority_sha256 is not None and raw_authority_sha != _require_sha256(
        expected_authority_sha256,
        label="external SLIM-Brain authority SHA-256",
    ):
        raise PostsealEvaluationError("SLIM-Brain authority external pin differs")
    if expected_authority_sha256 is None:
        raise PostsealEvaluationError(
            "available distributional metrics require an external authority pin"
        )
    if (
        type(authority_file.get("path")) is not str
        or not Path(authority_file["path"]).is_absolute()
        or type(authority_file.get("size_bytes")) is not int
        or authority_file["size_bytes"] < 1
        or authority.get("model_name") != "slimbrain"
        or authority.get("frozen_eval") is not True
        or authority.get("torchscript_only") is not True
    ):
        raise PostsealEvaluationError("SLIM-Brain authority identity differs")
    _validate_artifact_descriptor(
        authority.get("checkpoint"), label="SLIM-Brain checkpoint evidence"
    )
    source = authority.get("source")
    if type(source) is not dict or set(source) != {
        "path",
        "sha256",
        "size_bytes",
        "revision",
    }:
        raise PostsealEvaluationError("SLIM-Brain source evidence differs")
    _validate_artifact_descriptor(
        {key: source[key] for key in ("path", "sha256", "size_bytes")},
        label="SLIM-Brain source evidence",
    )
    revision = source.get("revision")
    if (
        type(revision) is not str
        or len(revision) not in {40, 64}
        or not all(character in _SHA256_CHARACTERS for character in revision)
    ):
        raise PostsealEvaluationError("SLIM-Brain source revision differs")
    adapter = authority.get("adapter")
    if (
        type(adapter) is not dict
        or set(adapter)
        != {"contract", "implementation_source_sha256", "input_semantics"}
        or adapter.get("contract") != SLIMBRAIN_ADAPTER_CONTRACT
        or adapter.get("input_semantics") != SLIMBRAIN_INPUT_SEMANTICS
        or not _is_sha256(adapter.get("implementation_source_sha256"))
    ):
        raise PostsealEvaluationError("SLIM-Brain adapter evidence differs")
    feature_output = authority.get("feature_output")
    logits_output = authority.get("logits_output")
    if (
        type(feature_output) is not dict
        or set(feature_output)
        != {"output_key", "layer_name", "aggregation", "dimension"}
        or feature_output.get("output_key") != "features"
        or type(feature_output.get("layer_name")) is not str
        or not feature_output["layer_name"]
        or type(feature_output.get("aggregation")) is not str
        or not feature_output["aggregation"]
        or type(feature_output.get("dimension")) is not int
        or feature_output["dimension"] < 1
        or type(logits_output) is not dict
        or set(logits_output)
        != {
            "output_key",
            "layer_name",
            "class_count",
            "trained",
            "class_semantics_sha256",
        }
        or logits_output.get("output_key") != "logits"
        or type(logits_output.get("layer_name")) is not str
        or not logits_output["layer_name"]
        or logits_output.get("trained") is not True
        or type(logits_output.get("class_count")) is not int
        or logits_output["class_count"] < 2
        or not _is_sha256(logits_output.get("class_semantics_sha256"))
    ):
        raise PostsealEvaluationError("SLIM-Brain layer/logit semantics differ")
    runtime = authority.get("runtime")
    if type(runtime) is not dict or set(runtime) != {
        "dependency_authority_sha256",
        "python_executable_sha256",
        "torch_version",
        "device_type",
    }:
        raise PostsealEvaluationError("SLIM-Brain runtime evidence differs")
    _require_sha256(
        runtime.get("dependency_authority_sha256"),
        label="SLIM-Brain dependency-authority SHA-256",
    )
    _require_sha256(
        runtime.get("python_executable_sha256"),
        label="SLIM-Brain Python SHA-256",
    )
    if (
        type(runtime.get("torch_version")) is not str
        or not runtime["torch_version"]
        or runtime.get("device_type") not in {"cpu", "cuda"}
    ):
        raise PostsealEvaluationError("SLIM-Brain runtime semantics differ")
    commitments = value.get("feature_commitments")
    if type(commitments) is not dict or set(commitments) != {
        "sample_count",
        "feature_dimension",
        "class_count",
        "generated_features_sha256",
        "real_features_sha256",
        "generated_logits_sha256",
    }:
        raise PostsealEvaluationError("SLIM-Brain feature commitments differ")
    if (
        commitments.get("sample_count") != len(EXPECTED_SCAN_IDS)
        or commitments.get("feature_dimension") != feature_output["dimension"]
        or commitments.get("class_count") != logits_output["class_count"]
    ):
        raise PostsealEvaluationError("SLIM-Brain feature dimensions/count differ")
    for name in (
        "generated_features_sha256",
        "real_features_sha256",
        "generated_logits_sha256",
    ):
        _require_sha256(commitments.get(name), label=f"SLIM-Brain {name}")
    _validate_distributional_artifact_descriptors(
        value.get("feature_artifacts"),
        feature_dimension=feature_output["dimension"],
        class_count=logits_output["class_count"],
    )
    _reauthenticate_distributional_authority(
        authority, expected_authority_sha256=raw_authority_sha
    )
    if artifact_root is not None:
        _recompute_distributional_metrics_from_artifacts(
            value, artifact_root=artifact_root
        )
    return True


def _validate_distributional_summary_mode(
    summary: Mapping[str, Any], *, available: bool
) -> None:
    """Reject a re-signed quality report that misstates FID/IS availability."""

    findings = summary.get("findings")
    if not isinstance(findings, list):
        raise PostsealEvaluationError("data-quality findings differ")
    distributional_findings = [
        finding
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("dimension") == "fid_and_inception_score"
    ]
    expected_finding = {
        "dimension": "fid_and_inception_score",
        "status": "pass" if available else "unavailable",
        "severity": "none" if available else "medium",
        "confidence": "high",
        "passing_scan_count": len(EXPECTED_SCAN_IDS) if available else 0,
        "passing_scan_rate": 1.0 if available else 0.0,
        "failed_scan_ids": [],
        "risk": None if available else "full paper metric coverage is incomplete",
        "required_action": (
            None
            if available
            else "qualify the closed pinned evaluation feature-model authority; never substitute"
        ),
    }
    if distributional_findings != [expected_finding]:
        raise PostsealEvaluationError("FID/IS data-quality finding differs")
    coverage = summary.get("required_input_file_hash_coverage")
    roles = coverage.get("required_by_role") if isinstance(coverage, dict) else None
    if not isinstance(roles, dict):
        raise PostsealEvaluationError("data-quality hash coverage differs")
    model_roles = {
        "slimbrain_evaluation_authority",
        "slimbrain_checkpoint",
        "slimbrain_source_manifest",
    }
    if any((role in roles) is not available for role in model_roles) or (
        available and any(roles[role] != 1 for role in model_roles)
    ):
        raise PostsealEvaluationError("SLIM-Brain hash-coverage mode differs")
    expected_hash_count = sum(
        count for count in roles.values() if type(count) is int and count >= 0
    )
    if (
        len(roles) != sum(type(count) is int and count >= 0 for count in roles.values())
        or coverage.get("required_file_count") != expected_hash_count
        or coverage.get("hash_authenticated_file_count") != expected_hash_count
        or coverage.get("coverage_rate") != 1.0
        or coverage.get("missing_or_unhashed") != []
    ):
        raise PostsealEvaluationError("data-quality hash totals differ")
    rates = summary.get("cohort_rates")
    fitness = summary.get("fitness_for_use")
    if not isinstance(rates, dict) or not isinstance(fitness, dict):
        raise PostsealEvaluationError("data-quality fitness evidence differs")
    collapse_count = rates.get("temporal_collapse_count")
    quality_fail_count = rates.get("quality_fail_count")
    if (
        type(collapse_count) is not int
        or collapse_count < 0
        or type(quality_fail_count) is not int
        or quality_fail_count < 0
    ):
        raise PostsealEvaluationError("data-quality failure counts differ")
    if collapse_count:
        expected_status, expected_severity = "NOT_FIT", "critical"
    elif quality_fail_count:
        expected_status, expected_severity = "NOT_FIT", "high"
    elif available:
        expected_status, expected_severity = "FIT", "none"
    else:
        expected_status, expected_severity = "CONDITIONALLY_FIT", "medium"
    blocked_uses = fitness.get("blocked_uses")
    unavailable_block = "FID or Inception Score claims without pinned SLIM-Brain"
    if (
        fitness.get("status") != expected_status
        or fitness.get("severity") != expected_severity
        or not isinstance(blocked_uses, list)
        or ((unavailable_block in blocked_uses) is available)
    ):
        raise PostsealEvaluationError("data-quality FID/IS fitness mode differs")


def _signed_record(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema": schema, **dict(body)}
    value["record_sha256"] = canonical_sha256(value)
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> FileSnapshot:
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    snapshot, _ = _snapshot_file(path, label=f"published {path.name}")
    return snapshot


def _write_bytes_new(path: Path, payload: bytes) -> FileSnapshot:
    """Write one non-empty binary artifact exactly once and durably."""

    if type(payload) is not bytes or not payload:
        raise PostsealEvaluationError(
            "published binary artifact must be non-empty bytes"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _snapshot_file(path, label=f"published {path.name}")[0]


@dataclass
class _HeldOutputDirectory:
    """Descriptor-held append-only publication root with retained bindings."""

    path: Path
    descriptor: int
    identity: tuple[int, int]
    bindings: dict[str, FileSnapshot] = dataclass_field(default_factory=dict)

    @classmethod
    def open(cls, path: Path, *, create: bool = False) -> "_HeldOutputDirectory":
        requested = Path(path)
        if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
            raise PostsealEvaluationError(
                "held publication directory must be an absolute lexical path"
            )
        parent = _canonical_path(
            requested.parent,
            label="held publication parent",
            directory=True,
        )
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        created = False
        try:
            if create:
                os.mkdir(requested.name, 0o700, dir_fd=parent_descriptor)
                created = True
            descriptor = os.open(
                requested.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            current = os.stat(
                requested.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                os.close(descriptor)
                raise PostsealEvaluationError(
                    "held publication directory changed while opening"
                )
            if created:
                os.fsync(parent_descriptor)
            return cls(
                path=requested,
                descriptor=descriptor,
                identity=(int(opened.st_dev), int(opened.st_ino)),
            )
        finally:
            os.close(parent_descriptor)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "_HeldOutputDirectory":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def assert_identity(self) -> None:
        opened = os.fstat(self.descriptor)
        current = self.path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise PostsealEvaluationError("held publication directory was replaced")

    def write_new(self, name: str, payload: bytes) -> FileSnapshot:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PostsealEvaluationError("held publication filename is invalid")
        self.assert_identity()
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=self.descriptor,
        )
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
                raise PostsealEvaluationError(
                    "held publication output is not a new single-link file"
                )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise PostsealEvaluationError(
                        "held publication write made no progress"
                    )
                view = view[written:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            final = os.fstat(descriptor)
            current = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                final.st_nlink != 1
                or stat.S_IMODE(final.st_mode) != 0o600
                or final.st_size != len(payload)
                or not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or stat.S_IMODE(current.st_mode) != 0o600
                or any(
                    getattr(initial, key) != getattr(final, key)
                    for key in ("st_dev", "st_ino")
                )
                or any(getattr(final, key) != getattr(current, key) for key in stable)
            ):
                raise PostsealEvaluationError(
                    "held publication output identity/mode/size changed"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            total = 0
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
                total += len(block)
            expected_digest = hashlib.sha256(payload).hexdigest()
            if total != len(payload) or digest.hexdigest() != expected_digest:
                raise PostsealEvaluationError(
                    "held publication output bytes differ after write"
                )
            snapshot = FileSnapshot(
                path=self.path / name,
                sha256=expected_digest,
                size_bytes=total,
                device=int(final.st_dev),
                inode=int(final.st_ino),
            )
            self.bindings[name] = snapshot
            os.fsync(self.descriptor)
        finally:
            os.close(descriptor)
        self.assert_identity()
        self.reauthenticate(snapshot)
        return snapshot

    def reauthenticate(self, expected: FileSnapshot) -> None:
        self.assert_identity()
        if expected.path.parent != self.path or expected.path.name not in self.bindings:
            raise PostsealEvaluationError("held publication binding escaped its root")
        descriptor = os.open(
            expected.path.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self.descriptor,
        )
        digest = hashlib.sha256()
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
            ):
                raise PostsealEvaluationError("held publication file mode/link differs")
            total = 0
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
                total += len(block)
            after = os.fstat(descriptor)
            current = os.stat(
                expected.path.name,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or stat.S_IMODE(after.st_mode) != 0o600
                or not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or stat.S_IMODE(current.st_mode) != 0o600
                or any(
                    getattr(before, key) != getattr(after, key)
                    or getattr(before, key) != getattr(current, key)
                    for key in stable
                )
            ):
                raise PostsealEvaluationError("held publication file changed")
        finally:
            os.close(descriptor)
        if (
            (before.st_dev, before.st_ino) != (expected.device, expected.inode)
            or total != expected.size_bytes
            or digest.hexdigest() != expected.sha256
        ):
            raise PostsealEvaluationError(
                "held publication file identity/size/hash differs"
            )
        self.assert_identity()

    def reauthenticate_all(self) -> None:
        for expected in self.bindings.values():
            self.reauthenticate(expected)


def _held_json_new(
    output: _HeldOutputDirectory,
    name: str,
    value: Mapping[str, Any],
) -> FileSnapshot:
    payload = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return output.write_new(name, payload)


def _source_inventory() -> dict[str, dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "postseal_evaluator": root / "eval" / "postseal.py",
        "postseal_executor": root / "eval" / "postseal_execution.py",
        "quality_evaluator": root / "eval" / "quality.py",
        "paper_metrics": root / "eval" / "postseal_metrics.py",
        "visualizer": root / "scripts" / "visualize_4d_comparison.py",
        "texture_auditor": root / "scripts" / "visualize_texture_audit.py",
        "postseal_cli": root / "scripts" / "evaluate_postseal_heldout.py",
        "postseal_slurm_launcher": (
            root / "scripts" / "run_postseal_heldout_evaluation.slurm"
        ),
        "spatial_detail": root / "utils" / "spatial_detail.py",
    }
    return {
        role: _snapshot_file(path, label=f"{role} implementation")[0].descriptor()
        for role, path in paths.items()
    }


def _relative_inventory(root: Path, *, excluded: set[str]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PostsealEvaluationError("evaluation bundle contains a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        snapshot, _ = _snapshot_file(path, label=f"evaluation output {relative}")
        result.append(
            {
                "relative_path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        )
    return result


def _publish_visualization_outputs(
    generated_outputs: Mapping[str, str | Path],
    *,
    publication: _HeldOutputDirectory,
    output_root: Path,
    subject_relative_dir: Path,
    temporary_inputs: Mapping[str, Path],
    input_descriptors: Mapping[str, FileSnapshot],
) -> tuple[dict[str, Path], FileSnapshot]:
    """Publish visual artifacts and their final bound manifest exactly once."""
    if "manifest" not in generated_outputs:
        raise PostsealEvaluationError("visualization manifest is missing")
    expected_inputs = {"real", "predicted", "mask", "roi_labels"}
    if (
        set(temporary_inputs) != expected_inputs
        or set(input_descriptors) != expected_inputs
    ):
        raise PostsealEvaluationError("visualization input set differs")
    manifest_path = _canonical_path(
        generated_outputs["manifest"],
        label="private visualization manifest",
    )
    manifest, _manifest_snapshot = _signed_json_snapshot(
        manifest_path,
        label="private visualization manifest",
        expected_schema_key="schema",
        expected_schema="connect4-4d-visualization-manifest-v1",
    )
    unsigned = dict(manifest)
    unsigned.pop("record_sha256", None)
    inputs = unsigned.get("inputs")
    outputs = unsigned.get("outputs")
    display = unsigned.get("display_contract")
    if (
        not isinstance(inputs, dict)
        or not isinstance(outputs, dict)
        or not isinstance(display, dict)
    ):
        raise PostsealEvaluationError("visualization manifest structure differs")
    if set(inputs) != expected_inputs:
        raise PostsealEvaluationError("visualization manifest input set differs")
    for name, snapshot in input_descriptors.items():
        generated_input = inputs[name]
        if not isinstance(generated_input, dict) or set(generated_input) != {
            "path",
            "sha256",
        }:
            raise PostsealEvaluationError(f"visualization input {name} differs")
        temporary_snapshot, _ = _snapshot_file(
            temporary_inputs[name],
            label=f"temporary visualization input {name}",
            expected_sha256=str(generated_input["sha256"]),
        )
        if (
            generated_input["path"] != str(temporary_snapshot.path)
            or temporary_snapshot.sha256 != snapshot.sha256
            or temporary_snapshot.size_bytes != snapshot.size_bytes
        ):
            raise PostsealEvaluationError(
                f"visualization input {name} is not the authenticated source"
            )
        inputs[name] = {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
        }
    expected_output_names = set(generated_outputs) - {"manifest"}
    if set(outputs) != expected_output_names:
        raise PostsealEvaluationError("visualization manifest output set differs")
    published: dict[str, Path] = {}
    for name, descriptor in outputs.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise PostsealEvaluationError("visualization output descriptor differs")
        private_path = _canonical_path(
            generated_outputs[name],
            label=f"private visualization output {name}",
        )
        if private_path.parent != manifest_path.parent or descriptor["path"] != str(
            private_path
        ):
            raise PostsealEvaluationError(
                "visualization output escaped its private directory"
            )
        actual, payload = _snapshot_file(
            private_path,
            label=f"private visualization output {name}",
            expected_sha256=str(descriptor["sha256"]),
            capture=True,
        )
        assert payload is not None
        public_snapshot = publication.write_new(private_path.name, payload)
        if public_snapshot.sha256 != actual.sha256:
            raise PostsealEvaluationError("visualization publication bytes differ")
        descriptor["path"] = str(output_root / subject_relative_dir / private_path.name)
        descriptor["sha256"] = public_snapshot.sha256
        published[name] = public_snapshot.path
    if (
        display.get("fixed_scales_across_all_animation_frames") is not True
        or display.get("image_interpolation") != "nearest"
        or display.get("mask_resampled") is not False
    ):
        raise PostsealEvaluationError("visualization display contract differs")
    encoded_unsigned = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if str(manifest_path.parent) in encoded_unsigned or ".authenticated_inputs" in (
        encoded_unsigned
    ):
        raise PostsealEvaluationError(
            "bound visualization manifest retained a private staged path"
        )
    unsigned["record_sha256"] = canonical_sha256(unsigned)
    bound_snapshot = _held_json_new(publication, manifest_path.name, unsigned)
    published["manifest"] = bound_snapshot.path
    publication.reauthenticate_all()
    return published, bound_snapshot


def _recompute_texture_gate(gate: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Recompute a texture verdict from raw values and frozen release policy.

    Per-result and aggregate booleans are evidence to cross-check, never an
    authority.  The policy below is frozen in this authenticated evaluator
    source; a self-consistent, re-signed manifest cannot weaken a bound.
    """
    expected_gate_fields = {
        "schema",
        "mode",
        "enforced",
        "amplitude_retention_enforced",
        "anti_gaming_guards_enforced",
        "policy",
        "results",
        "anti_gaming_policy",
        "anti_gaming_results",
        "all_metrics_meet_reference",
        "anti_gaming_guards_passed",
        "would_pass_if_enforced",
        "would_pass_if_anti_gaming_enforced",
        "would_pass_if_all_enforced",
        "release_gate_passed",
        "verdict",
        "figure_label",
        "exit_code",
        "failure_exit_code",
    }
    if set(gate) != expected_gate_fields:
        raise PostsealEvaluationError(f"{label} texture gate fields differ")
    policy = gate.get("policy")
    anti_policy = gate.get("anti_gaming_policy")
    results = gate.get("results")
    anti_results = gate.get("anti_gaming_results")
    expected_policy = {
        "comparison": "predicted_to_real_retention_ratio",
        "operator": ">=",
        "minimum_inclusive": _TEXTURE_RETENTION_MINIMUM,
        "all_metrics_required": True,
        "required_metrics": list(_TEXTURE_RETENTION_METRICS),
        "nonfinite_values_fail": True,
        "paper_reported_metric": False,
        "scope": "implementation release policy for post-inference spatial texture",
    }
    expected_anti_policy = {
        "all_guards_required_when_enforced": True,
        "required_metrics": list(_TEXTURE_ANTI_GAMING_METRICS),
        "aggregate_detail_correlation": {
            "operator": ">=",
            "minimum_inclusive": _TEXTURE_DETAIL_CORRELATION_MINIMUM,
            "support": "aggregate boundary-safe high-pass interior samples",
        },
        "near_nyquist_tail_power_ratio": {
            "comparison": "predicted_to_real_power_ratio",
            "operator": "<=",
            "maximum_inclusive": _TEXTURE_NEAR_NYQUIST_MAXIMUM,
            "boundary_cycles_per_voxel": _TEXTURE_NEAR_NYQUIST_BOUNDARY,
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
            "implementation release policy guarding amplitude-only texture gate gaming"
        ),
        "calibration_note": (
            "initial bounds selected from the B915 blind-validation amplitude-gaming "
            "negative control; not a paper metric"
        ),
    }
    if (
        policy != expected_policy
        or anti_policy != expected_anti_policy
        or not isinstance(results, dict)
        or set(results) != set(_TEXTURE_RETENTION_METRICS)
        or not isinstance(anti_results, dict)
        or set(anti_results) != set(_TEXTURE_ANTI_GAMING_METRICS)
    ):
        raise PostsealEvaluationError(f"{label} frozen texture policy differs")

    recomputed_results: dict[str, dict[str, Any]] = {}
    retention_passes: list[bool] = []
    for name in _TEXTURE_RETENTION_METRICS:
        result = results[name]
        if not isinstance(result, dict) or set(result) != {
            "display_name",
            "value",
            "minimum_inclusive",
            "passed",
        }:
            raise PostsealEvaluationError(f"{label} texture result {name} differs")
        raw_value = result.get("value")
        finite = type(raw_value) in {int, float} and np.isfinite(float(raw_value))
        expected_pass = bool(finite and float(raw_value) >= _TEXTURE_RETENTION_MINIMUM)
        if (
            result.get("display_name") != _TEXTURE_RETENTION_DISPLAY_NAMES[name]
            or result.get("minimum_inclusive") != _TEXTURE_RETENTION_MINIMUM
            or result.get("passed") is not expected_pass
        ):
            raise PostsealEvaluationError(
                f"{label} texture result {name} contradicts raw metric/policy"
            )
        retention_passes.append(expected_pass)
        recomputed_results[name] = dict(result)

    anti_specs = {
        "aggregate_detail_correlation": (
            ">=",
            "minimum_inclusive",
            _TEXTURE_DETAIL_CORRELATION_MINIMUM,
        ),
        "near_nyquist_tail_power_ratio": (
            "<=",
            "maximum_inclusive",
            _TEXTURE_NEAR_NYQUIST_MAXIMUM,
        ),
    }
    recomputed_anti_results: dict[str, dict[str, Any]] = {}
    anti_passes: list[bool] = []
    for name in _TEXTURE_ANTI_GAMING_METRICS:
        operator, bound_name, bound = anti_specs[name]
        result = anti_results[name]
        if not isinstance(result, dict) or set(result) != {
            "display_name",
            "value",
            "operator",
            bound_name,
            "passed",
        }:
            raise PostsealEvaluationError(
                f"{label} texture anti-gaming result {name} differs"
            )
        raw_value = result.get("value")
        finite = type(raw_value) in {int, float} and np.isfinite(float(raw_value))
        expected_pass = bool(
            finite
            and (
                float(raw_value) >= bound
                if operator == ">="
                else float(raw_value) <= bound
            )
        )
        if (
            result.get("display_name") != _TEXTURE_ANTI_GAMING_DISPLAY_NAMES[name]
            or result.get("operator") != operator
            or result.get(bound_name) != bound
            or result.get("passed") is not expected_pass
        ):
            raise PostsealEvaluationError(
                f"{label} texture anti-gaming result {name} contradicts raw "
                "metric/policy"
            )
        anti_passes.append(expected_pass)
        recomputed_anti_results[name] = dict(result)

    retention_passed = bool(all(retention_passes))
    anti_passed = bool(all(anti_passes))
    combined_passed = bool(retention_passed and anti_passed)
    expected_exit = 0 if combined_passed else _TEXTURE_FAILURE_EXIT_CODE
    expected_verdict = "pass" if combined_passed else "fail"
    expected_label = f"ENFORCED TEXTURE + ANTI-GAMING GATE: {expected_verdict.upper()}"
    aggregate_expectations = {
        "all_metrics_meet_reference": retention_passed,
        "anti_gaming_guards_passed": anti_passed,
        "would_pass_if_enforced": retention_passed,
        "would_pass_if_anti_gaming_enforced": anti_passed,
        "would_pass_if_all_enforced": combined_passed,
        "release_gate_passed": combined_passed,
        "verdict": expected_verdict,
        "figure_label": expected_label,
        "exit_code": expected_exit,
    }
    if (
        gate.get("schema") != TEXTURE_GATE_SCHEMA
        or gate.get("mode") != "combined-release-gate"
        or gate.get("enforced") is not True
        or gate.get("amplitude_retention_enforced") is not True
        or gate.get("anti_gaming_guards_enforced") is not True
        or gate.get("failure_exit_code") != _TEXTURE_FAILURE_EXIT_CODE
        or any(
            gate.get(name) != value for name, value in aggregate_expectations.items()
        )
    ):
        raise PostsealEvaluationError(
            f"{label} texture aggregate verdict contradicts recomputed metrics"
        )
    return {
        "passed": combined_passed,
        "verdict": expected_verdict,
        "exit_code": expected_exit,
        "policy": dict(policy),
        "anti_gaming_policy": dict(anti_policy),
        "results": recomputed_results,
        "anti_gaming_results": recomputed_anti_results,
    }


def _publish_texture_audit(
    generated_outputs: Mapping[str, str | Path],
    *,
    publication: _HeldOutputDirectory,
    output_root: Path,
    subject_relative_dir: Path,
    temporary_inputs: Mapping[str, Path],
    input_descriptors: Mapping[str, FileSnapshot],
    prediction: Any,
    predictions: Any,
) -> dict[str, Any]:
    """Authenticate and publish one enforced texture audit append-only.

    The texture implementation necessarily receives private copies while the
    append-only bundle is being assembled.  This boundary validates those
    temporary inputs and every generated byte, then writes one final manifest
    whose input paths refer to the authenticated prediction/Stage-B artifacts.
    The raw generator manifest remains private and is never overwritten.
    """
    if set(generated_outputs) != {"png", "manifest"}:
        raise PostsealEvaluationError("texture audit output set differs")
    expected_input_names = {"real", "predicted", "mask", "roi_labels"}
    if (
        set(temporary_inputs) != expected_input_names
        or set(input_descriptors) != expected_input_names
    ):
        raise PostsealEvaluationError("texture audit input set differs")
    texture_dir = subject_relative_dir / "texture"
    manifest_path = _canonical_path(
        Path(generated_outputs["manifest"]),
        label=f"{prediction.scan_id} texture audit manifest",
    )
    png_path = _canonical_path(
        Path(generated_outputs["png"]),
        label=f"{prediction.scan_id} texture audit PNG",
    )
    staged_root = temporary_inputs["real"].parents[2]
    expected_private_directory = temporary_inputs["real"].parent / "texture_raw"
    expected_texture_directory = staged_root / texture_dir
    if (
        manifest_path.parent != png_path.parent
        or manifest_path.parent != expected_private_directory
        or publication.path != expected_texture_directory
        or staged_root.parent != output_root.parent
    ):
        raise PostsealEvaluationError("texture audit outputs span directories")
    if manifest_path.name != f"{prediction.scan_id}_texture_audit.json" or (
        png_path.name != f"{prediction.scan_id}_texture_audit.png"
    ):
        raise PostsealEvaluationError("texture audit output names differ")

    manifest, original_snapshot = _signed_json_snapshot(
        manifest_path,
        label=f"{prediction.scan_id} texture audit manifest",
        expected_schema_key="schema",
        expected_schema=TEXTURE_AUDIT_SCHEMA,
    )
    unsigned = dict(manifest)
    unsigned.pop("record_sha256", None)
    gate = unsigned.get("quality_gate")
    inputs = unsigned.get("inputs")
    geometry = unsigned.get("geometry")
    display = unsigned.get("display_contract")
    implementation_sources = unsigned.get("implementation_sources")
    manifest_outputs = unsigned.get("outputs")
    if not all(
        isinstance(value, dict)
        for value in (
            gate,
            inputs,
            geometry,
            display,
            implementation_sources,
            manifest_outputs,
        )
    ):
        raise PostsealEvaluationError("texture audit manifest structure differs")
    if unsigned.get("target_blind_provenance") is not None:
        raise PostsealEvaluationError(
            "texture audit unexpectedly claimed independent prediction provenance"
        )

    recomputed_gate = _recompute_texture_gate(
        gate, label=f"{prediction.scan_id} generated"
    )
    if recomputed_gate["passed"] is not True:
        raise PostsealEvaluationError(
            f"{prediction.scan_id} texture retention/anti-gaming verdict failed"
        )
    exit_code = recomputed_gate["exit_code"]
    policy = recomputed_gate["policy"]
    anti_gaming_policy = recomputed_gate["anti_gaming_policy"]
    results = recomputed_gate["results"]
    anti_gaming_results = recomputed_gate["anti_gaming_results"]

    if set(inputs) != expected_input_names:
        raise PostsealEvaluationError("texture audit input set differs")
    for name in sorted(inputs):
        generated = inputs[name]
        if not isinstance(generated, dict) or set(generated) != {"path", "sha256"}:
            raise PostsealEvaluationError(f"texture audit input {name} differs")
        temporary_snapshot, _ = _snapshot_file(
            temporary_inputs[name],
            label=f"{prediction.scan_id} temporary texture input {name}",
            expected_sha256=str(generated.get("sha256", "")),
        )
        authenticated = input_descriptors[name]
        if (
            generated.get("path") != str(temporary_snapshot.path)
            or temporary_snapshot.sha256 != authenticated.sha256
            or temporary_snapshot.size_bytes != authenticated.size_bytes
        ):
            raise PostsealEvaluationError(
                f"texture audit input {name} is not the authenticated source"
            )
        inputs[name] = authenticated.descriptor()

    shape = geometry.get("shape_xyzt")
    voxel_sizes = geometry.get("voxel_sizes_mm")
    tr_seconds = geometry.get("tr_seconds")
    if (
        shape != list(EXPECTED_SHAPE)
        or not isinstance(voxel_sizes, list)
        or len(voxel_sizes) != 3
        or any(
            type(value) not in {int, float}
            or not np.isclose(
                float(value), expected, rtol=0.0, atol=np.finfo(np.float32).eps
            )
            for value, expected in zip(voxel_sizes, EXPECTED_VOXEL_SIZE_MM)
        )
        or type(tr_seconds) not in {int, float}
        or not np.isclose(float(tr_seconds), EXPECTED_TR_SECONDS, rtol=0.0, atol=1e-6)
        or geometry.get("mask_resampled") is not False
        or display.get("image_interpolation") != "nearest"
        or display.get("fMRI_registration_or_resampling") is not False
        or display.get("selection_uses_prediction") is not False
    ):
        raise PostsealEvaluationError(
            "texture audit display/grid contract permits resampling or smoothing"
        )

    source_root = Path(__file__).resolve().parents[1]
    expected_sources = {
        "texture_audit": source_root / "scripts" / "visualize_texture_audit.py",
        "strict_pair_visualizer": (
            source_root / "scripts" / "visualize_4d_comparison.py"
        ),
        "boundary_safe_high_pass": source_root / "utils" / "spatial_detail.py",
    }
    if set(implementation_sources) != set(expected_sources):
        raise PostsealEvaluationError("texture audit implementation sources differ")
    for name, expected_path in expected_sources.items():
        descriptor = implementation_sources[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise PostsealEvaluationError(
                f"texture audit implementation source {name} differs"
            )
        source_snapshot, _ = _snapshot_file(
            expected_path,
            label=f"texture audit implementation source {name}",
            expected_sha256=str(descriptor.get("sha256", "")),
        )
        if descriptor.get("path") != str(source_snapshot.path):
            raise PostsealEvaluationError(
                f"texture audit implementation source {name} path differs"
            )
        implementation_sources[name] = source_snapshot.descriptor()

    if set(manifest_outputs) != {"texture_audit_png"}:
        raise PostsealEvaluationError("texture audit manifest output set differs")
    png_binding = manifest_outputs["texture_audit_png"]
    if not isinstance(png_binding, dict) or set(png_binding) != {"path", "sha256"}:
        raise PostsealEvaluationError("texture audit PNG descriptor differs")
    png_snapshot, png_payload = _snapshot_file(
        png_path,
        label=f"{prediction.scan_id} texture audit PNG",
        expected_sha256=str(png_binding.get("sha256", "")),
        capture=True,
    )
    if png_binding.get("path") != str(png_snapshot.path):
        raise PostsealEvaluationError("texture audit PNG path differs")
    expected_png_relative = texture_dir / png_path.name
    assert png_payload is not None
    published_png = publication.write_new(png_path.name, png_payload)
    if published_png.sha256 != png_snapshot.sha256:
        raise PostsealEvaluationError("texture audit PNG publication bytes differ")
    png_binding.clear()
    png_binding.update(
        {
            "path": str(output_root / expected_png_relative),
            "sha256": published_png.sha256,
            "size_bytes": published_png.size_bytes,
        }
    )

    unsigned["target_blind_provenance"] = {
        "verified": True,
        "prediction_set_seal": predictions.seal_snapshot.descriptor()
        | {"record_sha256": predictions.seal["record_sha256"]},
        "prediction_subject_record": prediction.record_snapshot.descriptor()
        | {"record_sha256": prediction.record["record_sha256"]},
        "prediction": prediction.prediction_snapshot.descriptor(),
        "prediction_set_fully_authenticated_before_any_target_access": True,
        "texture_audit_run_only_after_all_targets_authenticated": True,
    }
    unsigned["target_blind_note"] = (
        "The complete 34-scan target-blind prediction set was authenticated before "
        "the Stage-B target boundary; this audit is post-seal evaluation only."
    )
    encoded_unsigned = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if (
        str(staged_root) in encoded_unsigned
        or ".authenticated_inputs" in encoded_unsigned
    ):
        raise PostsealEvaluationError(
            "bound texture manifest retained a private staged input path"
        )
    unsigned["record_sha256"] = canonical_sha256(unsigned)

    current_snapshot, _ = _snapshot_file(
        manifest_path,
        label=f"{prediction.scan_id} private texture manifest before publication",
        expected_sha256=original_snapshot.sha256,
        expected_size=original_snapshot.size_bytes,
    )
    if (current_snapshot.device, current_snapshot.inode) != (
        original_snapshot.device,
        original_snapshot.inode,
    ):
        raise PostsealEvaluationError("private texture manifest changed before binding")
    bound_snapshot = _held_json_new(publication, manifest_path.name, unsigned)
    publication.reauthenticate_all()
    manifest_relative = texture_dir / manifest_path.name
    return {
        "schema": TEXTURE_AUDIT_SCHEMA,
        "quality_gate_schema": TEXTURE_GATE_SCHEMA,
        "quality_gate": dict(gate),
        "mode": gate["mode"],
        "verdict": gate["verdict"],
        "exit_code": exit_code,
        "release_gate_passed": True,
        "amplitude_retention_enforced": True,
        "anti_gaming_guards_enforced": True,
        "policy": dict(policy),
        "anti_gaming_policy": dict(anti_gaming_policy),
        "results": dict(results),
        "anti_gaming_results": dict(anti_gaming_results),
        "display_contract": {
            "image_interpolation": "nearest",
            "fmri_registration_or_resampling": False,
            "mask_resampled": False,
            "selection_uses_prediction": False,
        },
        "outputs": {
            "png": {
                "relative_path": expected_png_relative.as_posix(),
                "sha256": published_png.sha256,
                "size_bytes": published_png.size_bytes,
            },
            "manifest": {
                "relative_path": manifest_relative.as_posix(),
                "sha256": bound_snapshot.sha256,
                "size_bytes": bound_snapshot.size_bytes,
                "record_sha256": unsigned["record_sha256"],
            },
        },
        "prediction_set_fully_authenticated_before_any_target_access": True,
        "all_targets_authenticated_before_texture_audit": True,
        "fmri_resampling_performed": False,
        "can_authorize_training_or_selection": False,
    }


def _aggregate_paper_metrics(
    subject_records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    names = ("mse", "voxel_corr", "roi_corr", "f2f_corr", "ssim", "psnr")
    aggregate: dict[str, float] = {}
    for name in names:
        values = []
        for record in subject_records:
            paper_report = record.get("quality", {}).get("paper_metrics", {})
            metric = paper_report.get("metrics", {}).get(name)
            report_schema = paper_report.get("schema")
            production_provenance_valid = (
                report_schema is None and "definitions_source" not in paper_report
            ) or (
                report_schema == "connect4-single-subject-paper-metrics-v1"
                and paper_report.get("definitions_source") == METRIC_DEFINITIONS_SOURCE
            )
            if (
                not isinstance(paper_report, dict)
                or not production_provenance_valid
                or not isinstance(metric, dict)
                or metric.get("available") is not True
                or isinstance(metric.get("value"), bool)
                or not isinstance(metric.get("value"), (int, float))
                or not np.isfinite(float(metric["value"]))
            ):
                raise PostsealEvaluationError(
                    f"paper metric {name} is unavailable/non-finite"
                )
            values.append(float(metric["value"]))
        aggregate[name] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return aggregate


def _finite_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise PostsealEvaluationError(f"{label} is missing or non-finite")
    return float(value)


def _optional_finite_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        return None
    return float(value)


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise PostsealEvaluationError("cohort summary received invalid numeric values")
    q25, median, q75 = np.percentile(array, (25.0, 50.0, 75.0))
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def _optional_distribution_summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None]
    return {
        "available_count": len(finite),
        "unavailable_count": len(values) - len(finite),
        "summary": None if not finite else _distribution_summary(finite),
    }


def _compact_texture_gate(scan_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the per-scan enforced texture evidence used for cohort fitness."""
    expected_evidence_fields = {
        "schema",
        "quality_gate_schema",
        "quality_gate",
        "mode",
        "verdict",
        "exit_code",
        "release_gate_passed",
        "amplitude_retention_enforced",
        "anti_gaming_guards_enforced",
        "policy",
        "anti_gaming_policy",
        "results",
        "anti_gaming_results",
        "display_contract",
        "outputs",
        "prediction_set_fully_authenticated_before_any_target_access",
        "all_targets_authenticated_before_texture_audit",
        "fmri_resampling_performed",
        "can_authorize_training_or_selection",
    }
    if set(evidence) != expected_evidence_fields:
        raise PostsealEvaluationError(f"{scan_id} texture evidence fields differ")
    raw_gate = evidence.get("quality_gate")
    if not isinstance(raw_gate, dict):
        raise PostsealEvaluationError(f"{scan_id} texture gate is missing")
    recomputed = _recompute_texture_gate(raw_gate, label=f"{scan_id} subject evidence")
    policy = recomputed["policy"]
    anti_gaming_policy = recomputed["anti_gaming_policy"]
    results = recomputed["results"]
    anti_gaming_results = recomputed["anti_gaming_results"]
    outputs = evidence.get("outputs")
    display = evidence.get("display_contract")
    if (
        evidence.get("schema") != TEXTURE_AUDIT_SCHEMA
        or evidence.get("quality_gate_schema") != TEXTURE_GATE_SCHEMA
        or recomputed["passed"] is not True
        or evidence.get("mode") != raw_gate["mode"]
        or evidence.get("verdict") != recomputed["verdict"]
        or evidence.get("exit_code") != recomputed["exit_code"]
        or evidence.get("release_gate_passed") is not recomputed["passed"]
        or evidence.get("amplitude_retention_enforced") is not True
        or evidence.get("anti_gaming_guards_enforced") is not True
        or evidence.get("prediction_set_fully_authenticated_before_any_target_access")
        is not True
        or evidence.get("all_targets_authenticated_before_texture_audit") is not True
        or evidence.get("fmri_resampling_performed") is not False
        or evidence.get("can_authorize_training_or_selection") is not False
        or evidence.get("policy") != policy
        or evidence.get("anti_gaming_policy") != anti_gaming_policy
        or evidence.get("results") != results
        or evidence.get("anti_gaming_results") != anti_gaming_results
        or not all(isinstance(value, dict) for value in (outputs, display))
        or display.get("image_interpolation") != "nearest"
        or display.get("fmri_registration_or_resampling") is not False
        or display.get("mask_resampled") is not False
        or display.get("selection_uses_prediction") is not False
    ):
        raise PostsealEvaluationError(
            f"{scan_id} enforced texture evidence is not passing"
        )
    if set(outputs) != {"png", "manifest"}:
        raise PostsealEvaluationError(f"{scan_id} texture output evidence differs")
    expected_paths = {
        "png": f"subjects/{scan_id}/texture/{scan_id}_texture_audit.png",
        "manifest": f"subjects/{scan_id}/texture/{scan_id}_texture_audit.json",
    }
    for name, expected_relative in expected_paths.items():
        descriptor = outputs[name]
        expected_fields = {"relative_path", "sha256", "size_bytes"}
        if name == "manifest":
            expected_fields.add("record_sha256")
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != expected_fields
            or descriptor.get("relative_path") != expected_relative
            or not _is_sha256(descriptor.get("sha256"))
            or isinstance(descriptor.get("size_bytes"), bool)
            or not isinstance(descriptor.get("size_bytes"), int)
            or descriptor.get("size_bytes") < 1
            or (name == "manifest" and not _is_sha256(descriptor.get("record_sha256")))
        ):
            raise PostsealEvaluationError(
                f"{scan_id} texture {name} descriptor differs"
            )
    return {
        "schema": TEXTURE_GATE_SCHEMA,
        "mode": "combined-release-gate",
        "verdict": recomputed["verdict"],
        "exit_code": recomputed["exit_code"],
        "passed": recomputed["passed"],
        "policy": dict(policy),
        "anti_gaming_policy": dict(anti_gaming_policy),
        "results": dict(results),
        "anti_gaming_results": dict(anti_gaming_results),
        "outputs": {name: dict(value) for name, value in outputs.items()},
        "nearest_neighbor_display": True,
        "fmri_resampled": False,
    }


def _compact_data_quality_row(
    scan_id: str,
    quality: Mapping[str, Any],
    pair_contract: Mapping[str, Any],
    texture_audit: Mapping[str, Any],
) -> dict[str, Any]:
    spatial = quality.get("spatial")
    temporal = quality.get("temporal")
    support = quality.get("support")
    outside = quality.get("outside_mask")
    structured = quality.get("structured_temporal")
    checks = quality.get("checks")
    if not all(
        isinstance(value, dict)
        for value in (spatial, temporal, support, outside, structured, checks)
    ):
        raise PostsealEvaluationError(f"{scan_id} quality record structure differs")
    ratios: dict[str, float] = {}
    ratio_gate_pass: dict[str, bool] = {}
    for key, section_name, value_name, check_name, _label in _QUALITY_RATIO_SPECS:
        section = spatial if section_name == "spatial" else temporal
        ratios[key] = _finite_number(section.get(value_name), label=f"{scan_id} {key}")
        check = checks.get(check_name)
        if not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
            raise PostsealEvaluationError(f"{scan_id} {check_name} gate differs")
        ratio_gate_pass[key] = bool(check["passed"])
    failed_checks = quality.get("failed_checks")
    if not isinstance(failed_checks, list) or not all(
        isinstance(value, str) for value in failed_checks
    ):
        raise PostsealEvaluationError(f"{scan_id} failed-check record differs")
    passed = quality.get("passed") is True
    texture_gate = _compact_texture_gate(scan_id, texture_audit)
    collapse = quality.get("temporal_collapse_detected") is True
    severity = "none" if passed else ("critical" if collapse else "high")
    fitness = (
        "FIT_FOR_PAIRED_QA"
        if passed
        else ("NOT_FIT_TEMPORAL_COLLAPSE" if collapse else "NOT_FIT_QUALITY_GATES")
    )
    real = pair_contract.get("real")
    predicted = pair_contract.get("predicted")
    brain_mask = pair_contract.get("structural_brain_mask")
    target_validity = pair_contract.get("target_validity_mask")
    labels = pair_contract.get("structural_labels")
    if not all(
        isinstance(value, dict)
        for value in (real, predicted, brain_mask, target_validity, labels)
    ):
        raise PostsealEvaluationError(f"{scan_id} pair contract structure differs")
    expected_target_validity_fields = {
        "contract",
        "derivation",
        "shape",
        "foreground_voxel_count",
        "foreground_fraction_of_structural_support",
        "structural_foreground_voxel_count",
        "structural_voxels_excluded_from_target_metrics",
        "outside_structural_brain_voxel_count",
        "configured_structural_roi_count",
        "all_configured_structural_rois_have_target_validity_support",
        "equals_exact_final_nonzero_support",
        "exactly_binary",
        "fmri_resampled",
    }
    target_validity_count = target_validity.get("foreground_voxel_count")
    structural_count = target_validity.get("structural_foreground_voxel_count")
    excluded_count = target_validity.get(
        "structural_voxels_excluded_from_target_metrics"
    )
    protocol_profile = pair_contract.get("protocol_profile")
    expected_target_validity_contract = (
        TARGET_VALIDITY_MASK_CONTRACT
        if protocol_profile is None
        else TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(protocol_profile)
    )
    if (
        set(target_validity) != expected_target_validity_fields
        or expected_target_validity_contract is None
        or target_validity.get("contract") != expected_target_validity_contract
        or (
            protocol_profile is not None
            and (
                pair_contract.get("canonical_roi_mapping_sha256")
                != CANONICAL_ROI_MAPPING_SHA256
                or pair_contract.get("positive_structural_label_count")
                != len(CANONICAL_ROI_LABEL_IDS)
            )
        )
        or target_validity.get("derivation")
        != "exact-nonzero-support-across-final-stored-target-time-series"
        or target_validity.get("shape") != list(EXPECTED_SPATIAL_SHAPE)
        or type(target_validity_count) is not int
        or type(structural_count) is not int
        or type(excluded_count) is not int
        or target_validity_count < 1
        or structural_count < target_validity_count
        or excluded_count != structural_count - target_validity_count
        or target_validity.get("outside_structural_brain_voxel_count") != 0
        or target_validity.get("configured_structural_roi_count")
        != pair_contract.get("positive_structural_label_count")
        or target_validity.get(
            "all_configured_structural_rois_have_target_validity_support"
        )
        is not True
        or target_validity.get("equals_exact_final_nonzero_support") is not True
        or target_validity.get("exactly_binary") is not True
        or target_validity.get("fmri_resampled") is not False
        or brain_mask.get("foreground_voxel_count") != structural_count
        or not np.isclose(
            _finite_number(
                target_validity.get("foreground_fraction_of_structural_support"),
                label=f"{scan_id} target-validity support fraction",
            ),
            target_validity_count / structural_count,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        )
    ):
        raise PostsealEvaluationError(
            f"{scan_id} target-validity pair contract differs"
        )
    return {
        "scan_id": scan_id,
        "patient_id": scan_id.rsplit("_", 1)[0],
        "domain": {
            "real_minimum": _finite_number(
                real.get("minimum"), label=f"{scan_id} real minimum"
            ),
            "real_maximum": _finite_number(
                real.get("maximum"), label=f"{scan_id} real maximum"
            ),
            "predicted_minimum": _finite_number(
                predicted.get("minimum"), label=f"{scan_id} prediction minimum"
            ),
            "predicted_maximum": _finite_number(
                predicted.get("maximum"), label=f"{scan_id} prediction maximum"
            ),
            "finite_and_in_unit_interval": True,
        },
        "geometry": {
            "shape": list(EXPECTED_SHAPE),
            "axis_codes": list(EXPECTED_AXIS_CODES),
            "voxel_sizes_mm": list(EXPECTED_VOXEL_SIZE_MM),
            "tr_seconds": EXPECTED_TR_SECONDS,
            "shape_affine_orientation_tr_exact": True,
            "fmri_resampled": False,
            "structural_label_count": int(
                pair_contract["positive_structural_label_count"]
            ),
            "structural_labels_independent": True,
            "structural_brain_mask_independent": True,
            "structural_brain_support_fraction": _finite_number(
                brain_mask.get("foreground_fraction"),
                label=f"{scan_id} structural brain support fraction",
            ),
            "target_validity_support_fraction_of_structural": _finite_number(
                target_validity.get("foreground_fraction_of_structural_support"),
                label=f"{scan_id} target-validity support fraction",
            ),
            "structural_voxels_excluded_from_target_metrics": excluded_count,
            "target_validity_equals_exact_final_nonzero_support": True,
            "roi_labels_subset_of_brain_support": (
                brain_mask.get("roi_foreground_outside_mask_voxel_count") == 0
            ),
        },
        "spatial_support": {
            "dice": _finite_number(
                support.get("dice"), label=f"{scan_id} support Dice"
            ),
            "center_of_mass_distance_mm": _optional_finite_number(
                support.get("center_of_mass_distance_mm")
            ),
            "outside_mask_leakage_ratio": _finite_number(
                outside.get("predicted_outside_mask_leakage_ratio"),
                label=f"{scan_id} leakage ratio",
            ),
        },
        "texture_ratios": {
            key: ratios[key] for key, *_rest in _QUALITY_RATIO_SPECS[:5]
        },
        "temporal_ratios": {
            key: ratios[key] for key, *_rest in _QUALITY_RATIO_SPECS[5:]
        },
        "texture_alignment": {
            "temporal_mean_high_pass_correlation": _optional_finite_number(
                spatial.get("temporal_mean_high_frequency_correlation")
            ),
            "dynamic_high_pass_correlation": _optional_finite_number(
                spatial.get("dynamic_high_frequency_correlation")
            ),
        },
        "texture_release_gate": texture_gate,
        "temporal_collapse": {
            "near_static_voxel_fraction": _finite_number(
                temporal.get("near_static_voxel_fraction"),
                label=f"{scan_id} near-static fraction",
            ),
            "detected": collapse,
        },
        "structured_dynamics": {
            "available": structured.get("available") is True,
            "roi_fc_correlation": _optional_finite_number(
                structured.get("fc_matrix_correlation")
            ),
            "roi_non_dc_spectrum_correlation": _optional_finite_number(
                structured.get("power_spectrum_correlation")
            ),
        },
        "ratios": ratios,
        "ratio_gate_pass": ratio_gate_pass,
        "quality_gate": {
            "passed": passed,
            "verdict": str(quality.get("verdict")),
            "failed_checks": failed_checks,
            "severity": severity,
            "confidence": {
                "measurement": "high",
                "engineering_threshold_interpretation": "moderate",
            },
            "fitness_for_use": fitness,
        },
    }


def _ratio_thresholds(quality: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    policy = quality.get("policy")
    if not isinstance(policy, dict):
        raise PostsealEvaluationError("quality policy is missing")
    result: dict[str, dict[str, float]] = {}
    for key, fields in _QUALITY_RATIO_POLICY_FIELDS.items():
        result[key] = {
            "minimum": _finite_number(
                policy.get(fields[0]), label=f"{key} minimum policy"
            ),
            "maximum": _finite_number(
                policy.get(fields[1]), label=f"{key} maximum policy"
            ),
        }
    return result


def _scan_failure_ids(
    rows: Sequence[Mapping[str, Any]], check_names: set[str]
) -> list[str]:
    return [
        str(row["scan_id"])
        for row in rows
        if any(name in check_names for name in row["quality_gate"]["failed_checks"])
    ]


def _finding(
    *,
    dimension: str,
    failed_scan_ids: Sequence[str],
    failure_severity: str,
    risk: str,
    required_action: str,
) -> dict[str, Any]:
    failed = list(failed_scan_ids)
    count = len(EXPECTED_SCAN_IDS) - len(failed)
    return {
        "dimension": dimension,
        "status": "pass" if not failed else "fail",
        "severity": "none" if not failed else failure_severity,
        "confidence": "high",
        "passing_scan_count": count,
        "passing_scan_rate": float(count / len(EXPECTED_SCAN_IDS)),
        "failed_scan_ids": failed,
        "risk": None if not failed else risk,
        "required_action": None if not failed else required_action,
    }


def _build_data_quality_summary(
    subject_records: Sequence[Mapping[str, Any]],
    *,
    distributional_evidence: Mapping[str, Any],
    verifier_dependency_count: int,
) -> dict[str, Any]:
    authority_pin = None
    if (
        type(distributional_evidence) is dict
        and distributional_evidence.get("available") is True
    ):
        model_authority = distributional_evidence.get("model_authority")
        authority_file = (
            model_authority.get("authority")
            if isinstance(model_authority, dict)
            else None
        )
        authority_pin = (
            authority_file.get("raw_sha256")
            if isinstance(authority_file, dict)
            else None
        )
    distributional_available = _validate_distributional_evidence(
        distributional_evidence,
        expected_authority_sha256=authority_pin,
    )
    if [record.get("scan_id") for record in subject_records] != list(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError("data-quality subject order/set differs")
    rows = [
        _compact_data_quality_row(
            str(record["scan_id"]),
            record["quality"],
            record["pair_contract"],
            record["texture_audit"],
        )
        for record in subject_records
    ]
    thresholds = _ratio_thresholds(subject_records[0]["quality"])
    for record in subject_records[1:]:
        if _ratio_thresholds(record["quality"]) != thresholds:
            raise PostsealEvaluationError("per-subject quality policies differ")
    texture_policy = rows[0]["texture_release_gate"]["policy"]
    texture_anti_gaming_policy = rows[0]["texture_release_gate"]["anti_gaming_policy"]
    if any(
        row["texture_release_gate"]["policy"] != texture_policy
        or row["texture_release_gate"]["anti_gaming_policy"]
        != texture_anti_gaming_policy
        for row in rows[1:]
    ):
        raise PostsealEvaluationError("per-subject texture policies differ")
    texture_failures = [
        str(row["scan_id"])
        for row in rows
        if row["texture_release_gate"]["passed"] is not True
    ]
    if texture_failures:
        raise PostsealEvaluationError(
            "a failed texture release gate reached cohort publication"
        )
    ratio_summaries = {}
    for key, _section, _value, _check, label in _QUALITY_RATIO_SPECS:
        values = [float(row["ratios"][key]) for row in rows]
        passing = sum(bool(row["ratio_gate_pass"][key]) for row in rows)
        ratio_summaries[key] = {
            "label": label,
            "predicted_over_real": True,
            "target_ratio": 1.0,
            "guardrail": thresholds[key],
            "distribution": _distribution_summary(values),
            "passing_scan_count": passing,
            "passing_scan_rate": float(passing / len(rows)),
        }
    support_values = [float(row["spatial_support"]["dice"]) for row in rows]
    leakage_values = [
        float(row["spatial_support"]["outside_mask_leakage_ratio"]) for row in rows
    ]
    high_pass_correlations = [
        row["texture_alignment"]["temporal_mean_high_pass_correlation"] for row in rows
    ]
    dynamic_high_pass_correlations = [
        row["texture_alignment"]["dynamic_high_pass_correlation"] for row in rows
    ]
    roi_fc = [row["structured_dynamics"]["roi_fc_correlation"] for row in rows]
    roi_spectrum = [
        row["structured_dynamics"]["roi_non_dc_spectrum_correlation"] for row in rows
    ]
    spatial_checks = {
        "support_dice",
        "support_center_of_mass",
        "mask_field_of_view",
        "outside_mask_leakage",
        "spatial_robust_range_ratio",
        "spatial_gradient_ratio",
        "spatial_laplacian_ratio",
        "spatial_high_frequency_ratio",
        "spatial_high_frequency_correlation",
        "dynamic_high_frequency_ratio",
        "dynamic_high_frequency_correlation",
    }
    temporal_checks = {
        "temporal_variance_ratio",
        "dvars_ratio",
        "dynamic_power_ratio",
        "effective_rank_ratio",
        "near_static_voxel_fraction",
    }
    structured_checks = {
        "structured_temporal_available",
        "roi_fc_correlation",
        "roi_power_spectrum_correlation",
    }
    spatial_failures = _scan_failure_ids(rows, spatial_checks)
    temporal_failures = _scan_failure_ids(rows, temporal_checks)
    structured_failures = _scan_failure_ids(rows, structured_checks)
    collapse_ids = [
        str(row["scan_id"])
        for row in rows
        if row["temporal_collapse"]["detected"] is True
    ]
    quality_failures = [
        str(row["scan_id"]) for row in rows if row["quality_gate"]["passed"] is not True
    ]
    required_files_by_role = {
        "prediction_set_seal": 1,
        "prediction_subject_record": len(EXPECTED_SCAN_IDS),
        "prediction_nifti": len(EXPECTED_SCAN_IDS),
        "stage_b_completed_set": 1,
        "native_real_bold": len(EXPECTED_SCAN_IDS),
        "native_structural_brain_mask": len(EXPECTED_SCAN_IDS),
        "native_structural_labels": len(EXPECTED_SCAN_IDS),
        "stage_b_publication": len(EXPECTED_SCAN_IDS),
        "stage_b_success_receipt": len(EXPECTED_SCAN_IDS),
        "native_preprocessing_receipt": len(EXPECTED_SCAN_IDS),
        "stage_b_verifier_source": 1,
        "stage_b_verifier_dependencies": int(verifier_dependency_count),
    }
    if distributional_available:
        required_files_by_role.update(
            {
                "slimbrain_evaluation_authority": 1,
                "slimbrain_checkpoint": 1,
                "slimbrain_source_manifest": 1,
            }
        )
    hash_count = int(sum(required_files_by_role.values()))
    if collapse_ids:
        fitness_status = "NOT_FIT"
        fitness_reason = "one or more predictions exhibit multi-gate temporal collapse"
        overall_severity = "critical"
    elif quality_failures:
        fitness_status = "NOT_FIT"
        fitness_reason = "one or more scans fail paired spatial/temporal quality gates"
        overall_severity = "high"
    elif not distributional_available:
        fitness_status = "CONDITIONALLY_FIT"
        fitness_reason = (
            "all paired contracts, engineering quality gates, and enforced texture "
            "retention/anti-gaming gates pass, but exact paper-wide FID/IS are "
            "unavailable without pinned SLIM-Brain"
        )
        overall_severity = "medium"
    else:
        fitness_status = "FIT"
        fitness_reason = (
            "all paired contracts, quality gates, enforced texture retention/"
            "anti-gaming gates, and paper metric paths complete"
        )
        overall_severity = "none"
    findings = [
        {
            "dimension": "sealed_identity_completeness_order_and_hash_coverage",
            "status": "pass",
            "severity": "none",
            "confidence": "high",
            "passing_scan_count": len(rows),
            "passing_scan_rate": 1.0,
            "failed_scan_ids": [],
            "risk": None,
            "required_action": None,
        },
        {
            "dimension": "finite_unit_domain_shape_affine_orientation_tr",
            "status": "pass",
            "severity": "none",
            "confidence": "high",
            "passing_scan_count": len(rows),
            "passing_scan_rate": 1.0,
            "failed_scan_ids": [],
            "risk": None,
            "required_action": None,
        },
        _finding(
            dimension="spatial_support_leakage_and_texture",
            failed_scan_ids=spatial_failures,
            failure_severity="high",
            risk="misaligned, leaking, or over-smoothed spatial predictions",
            required_action="treat affected scans as failed; inspect fixed-scale spatial diagnostics",
        ),
        _finding(
            dimension="enforced_texture_retention_and_anti_gaming",
            failed_scan_ids=texture_failures,
            failure_severity="high",
            risk="over-smoothed or amplitude-gamed predictions",
            required_action="reject the complete append-only evaluation publication",
        ),
        _finding(
            dimension="temporal_variance_dvars_non_dc_power_and_effective_rank",
            failed_scan_ids=temporal_failures,
            failure_severity="critical" if collapse_ids else "high",
            risk="temporal mismatch or collapsed 4D dynamics",
            required_action="do not claim temporal fidelity; inspect full traces and spectra",
        ),
        _finding(
            dimension="roi_functional_connectivity_and_spectrum",
            failed_scan_ids=structured_failures,
            failure_severity="high",
            risk="anatomically structured dynamics are not reproduced",
            required_action="inspect fixed ROI FC/spectrum panels and retain failure status",
        ),
        {
            "dimension": "fid_and_inception_score",
            "status": "pass" if distributional_available else "unavailable",
            "severity": "none" if distributional_available else "medium",
            "confidence": "high",
            "passing_scan_count": len(rows) if distributional_available else 0,
            "passing_scan_rate": 1.0 if distributional_available else 0.0,
            "failed_scan_ids": [],
            "risk": (
                None
                if distributional_available
                else "full paper metric coverage is incomplete"
            ),
            "required_action": (
                None
                if distributional_available
                else "qualify the closed pinned evaluation feature-model authority; never substitute"
            ),
        },
    ]
    return {
        "status": "COMPLETE_34_SCAN_DATA_QUALITY_REVIEW",
        "analytical_question": (
            "Does the complete target-blind prediction set reproduce aligned spatial "
            "signal, fine-scale texture, and non-collapsed temporal dynamics?"
        ),
        "dataset_and_grain": {
            "dataset_role": "fixed-sealed-test-final-report-only",
            "scan_row_grain": "one row per sealed visit/scan",
            "scan_row_count": len(rows),
            "patient_count": len({row["patient_id"] for row in rows}),
            "array_grain": "one normalized voxel value per frame within one scan",
            "array_shape_per_scan": list(EXPECTED_SHAPE),
            "spatial_voxels_per_frame": int(np.prod(EXPECTED_SPATIAL_SHAPE)),
            "frames_per_scan": EXPECTED_SHAPE[-1],
            "scalar_values_across_cohort": int(len(rows) * np.prod(EXPECTED_SHAPE)),
            "voxel_sizes_mm": list(EXPECTED_VOXEL_SIZE_MM),
            "tr_seconds": EXPECTED_TR_SECONDS,
        },
        "completeness_identity_and_order": {
            "expected_scan_count": len(EXPECTED_SCAN_IDS),
            "observed_scan_count": len(rows),
            "complete_scan_rate": 1.0,
            "unique_scan_id_rate": 1.0,
            "fixed_order_exact": True,
            "ordered_scan_ids_sha256": EXPECTED_SCAN_IDS_SHA256,
            "no_replacement": True,
            "no_refill": True,
            "prediction_set_fully_authenticated_before_target_access": True,
        },
        "required_input_file_hash_coverage": {
            "required_by_role": required_files_by_role,
            "required_file_count": hash_count,
            "hash_authenticated_file_count": hash_count,
            "coverage_rate": 1.0,
            "missing_or_unhashed": [],
        },
        "contract_rates": {
            "finite_unit_interval_rate": 1.0,
            "exact_shape_rate": 1.0,
            "exact_affine_rate": 1.0,
            "exact_orientation_rate": 1.0,
            "exact_tr_rate": 1.0,
            "no_fmri_resampling_rate": 1.0,
            "independent_structural_support_rate": 1.0,
        },
        "cohort_distributions": {
            "support_dice": _distribution_summary(support_values),
            "outside_mask_leakage_ratio": _distribution_summary(leakage_values),
            "temporal_mean_high_pass_correlation": (
                _optional_distribution_summary(high_pass_correlations)
            ),
            "dynamic_high_pass_correlation": (
                _optional_distribution_summary(dynamic_high_pass_correlations)
            ),
            "roi_fc_correlation": _optional_distribution_summary(roi_fc),
            "roi_non_dc_spectrum_correlation": (
                _optional_distribution_summary(roi_spectrum)
            ),
            "predicted_over_real_ratios": ratio_summaries,
        },
        "cohort_rates": {
            "quality_pass_count": len(rows) - len(quality_failures),
            "quality_fail_count": len(quality_failures),
            "quality_pass_rate": float((len(rows) - len(quality_failures)) / len(rows)),
            "temporal_collapse_count": len(collapse_ids),
            "temporal_collapse_rate": float(len(collapse_ids) / len(rows)),
            "spatial_quality_failure_rate": float(len(spatial_failures) / len(rows)),
            "temporal_quality_failure_rate": float(len(temporal_failures) / len(rows)),
            "structured_dynamics_failure_rate": float(
                len(structured_failures) / len(rows)
            ),
            "texture_release_gate_pass_count": len(rows) - len(texture_failures),
            "texture_release_gate_fail_count": len(texture_failures),
            "texture_release_gate_pass_rate": float(
                (len(rows) - len(texture_failures)) / len(rows)
            ),
        },
        "texture_release_gate": {
            "schema": TEXTURE_GATE_SCHEMA,
            "mode": "combined-release-gate",
            "required_for_publication": True,
            "verdict": "pass",
            "exit_code": 0,
            "scan_count": len(rows),
            "passing_scan_count": len(rows) - len(texture_failures),
            "failing_scan_count": len(texture_failures),
            "failed_scan_ids": texture_failures,
            "policy": dict(texture_policy),
            "anti_gaming_policy": dict(texture_anti_gaming_policy),
            "subject_outputs": [
                {
                    "scan_id": row["scan_id"],
                    "verdict": row["texture_release_gate"]["verdict"],
                    "exit_code": row["texture_release_gate"]["exit_code"],
                    "outputs": row["texture_release_gate"]["outputs"],
                }
                for row in rows
            ],
            "nearest_neighbor_display": True,
            "fmri_resampled": False,
            "evaluation_feedback_to_training_or_selection": False,
        },
        "findings": findings,
        "fitness_for_use": {
            "use_case": "faithful post-seal held-out real-versus-predicted evaluation",
            "status": fitness_status,
            "severity": overall_severity,
            "confidence": {
                "sealed_contract_measurements": "high",
                "complete_cohort_rates": "high",
                "engineering_gate_interpretation": "moderate",
            },
            "reason": fitness_reason,
            "texture_release_gate_required": True,
            "texture_release_gate_verdict": "pass",
            "texture_release_gate_pass_count": len(rows),
            "texture_release_gate_fail_count": 0,
            "allowed_uses": (
                [
                    "diagnostic visualization and failure analysis",
                    "reporting paired metrics with explicit failed quality status",
                ]
                if fitness_status == "NOT_FIT"
                else [
                    "fixed-cohort paired quality reporting",
                    "fixed-scale real/predicted/residual visualization",
                    "paper paired-metric reporting",
                ]
            ),
            "blocked_uses": [
                "training or checkpoint/candidate selection",
                "additional prediction or inference authorization",
                "claiming paper numerical reproduction without comparing authenticated results",
            ]
            + (
                []
                if distributional_available
                else ["FID or Inception Score claims without pinned SLIM-Brain"]
            ),
        },
        "ratio_thresholds": thresholds,
        "per_scan_rows": rows,
        "threshold_note": (
            "Quality gates are predeclared broad engineering guardrails, not "
            "paper-reported synthesis thresholds."
        ),
        "distributional_metrics": dict(distributional_evidence),
        "evaluation_feedback_to_training_or_selection": False,
    }


def _render_cohort_quality_chart(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    publication: _HeldOutputDirectory | None = None,
) -> FileSnapshot:
    """Render the predeclared two-panel 34-scan ratio distribution chart."""
    owned_publication = publication is None
    if publication is not None and output_path.parent != publication.path:
        raise PostsealEvaluationError("cohort chart escaped its held output root")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    rows = summary["per_scan_rows"]
    thresholds = summary["ratio_thresholds"]
    if len(rows) != len(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError("cohort chart requires exactly 34 scan rows")
    figure, axes = plt.subplots(2, 1, figsize=(12.4, 8.4), constrained_layout=True)
    blue = "#225ea8"
    orange = "#e67e22"
    ink = "#263238"
    neutral = "#b0bec5"
    jitter = np.linspace(-0.22, 0.22, len(rows))
    panels = (_QUALITY_RATIO_SPECS[:5], _QUALITY_RATIO_SPECS[5:])
    panel_titles = (
        "Spatial texture ratios (prediction / real)",
        "Temporal dynamics ratios (prediction / real)",
    )
    for axis, specs, title in zip(axes, panels, panel_titles):
        plotted_values: list[float] = []
        plotted_limits: list[float] = [1.0]
        for position, (key, _section, _value, _check, label) in enumerate(specs):
            values = np.asarray(
                [float(row["ratios"][key]) for row in rows], dtype=np.float64
            )
            passes = np.asarray(
                [bool(row["ratio_gate_pass"][key]) for row in rows], dtype=bool
            )
            lower = float(thresholds[key]["minimum"])
            upper = float(thresholds[key]["maximum"])
            plotted_values.extend(values.tolist())
            plotted_limits.extend((lower, upper))
            axis.plot(
                [lower, upper],
                [position, position],
                color=neutral,
                lw=7.0,
                solid_capstyle="round",
                zorder=1,
            )
            if bool(passes.any()):
                axis.scatter(
                    values[passes],
                    position + jitter[passes],
                    s=22,
                    marker="o",
                    facecolors="white",
                    edgecolors=blue,
                    linewidths=0.9,
                    zorder=3,
                )
            if bool((~passes).any()):
                axis.scatter(
                    values[~passes],
                    position + jitter[~passes],
                    s=28,
                    marker="x",
                    color=orange,
                    linewidths=1.3,
                    zorder=4,
                )
            axis.scatter(
                [float(np.median(values))],
                [position],
                s=34,
                marker="D",
                color=ink,
                zorder=5,
            )
        all_values = np.asarray(plotted_values + plotted_limits, dtype=np.float64)
        minimum = min(0.0, float(all_values.min()))
        maximum = float(all_values.max())
        padding = max(0.08, 0.06 * max(maximum - minimum, 1.0))
        axis.set_xlim(minimum - padding, maximum + padding)
        axis.set_yticks(np.arange(len(specs)), [str(spec[-1]) for spec in specs])
        axis.set_ylim(len(specs) - 0.55, -0.55)
        axis.axvline(1.0, color=ink, lw=1.1, ls="--", zorder=0)
        axis.set_title(title, loc="left", color=ink, fontsize=11)
        axis.set_xlabel("Ratio; 1.0 means matched magnitude")
        axis.grid(axis="x", color="#e6e9eb", lw=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=blue,
            label="scan passes metric guardrail",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=orange,
            lw=0,
            label="scan fails metric guardrail",
        ),
        Line2D([0], [0], marker="D", color=ink, lw=0, label="cohort median"),
        Line2D([0], [0], color=neutral, lw=7, label="predeclared guardrail"),
        Line2D([0], [0], color=ink, lw=1.1, ls="--", label="matched ratio = 1"),
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Post-seal paired quality ratios — fixed 34-scan cohort\n"
        "61x73x61x128 native grid · 3-mm RAS+ · TR=3 s · one dot per scan",
        fontsize=14,
        color=ink,
    )
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=180, facecolor="white")
    finally:
        plt.close(figure)
    if publication is None:
        publication = _HeldOutputDirectory.open(output_path.parent)
        output_path = publication.path / output_path.name
    try:
        snapshot = publication.write_new(output_path.name, buffer.getvalue())
        publication.reauthenticate_all()
        return snapshot
    finally:
        if owned_publication:
            publication.close()


def _data_quality_markdown(summary: Mapping[str, Any]) -> str:
    dataset = summary["dataset_and_grain"]
    rates = summary["cohort_rates"]
    fitness = summary["fitness_for_use"]
    ratios = summary["cohort_distributions"]["predicted_over_real_ratios"]
    lines = [
        "# CONNECT-4 post-seal held-out data-quality report",
        "",
        f"**Fitness for use: {fitness['status']} ({fitness['severity']}).** "
        f"{fitness['reason']}",
        "",
        "## Dataset and grain",
        "",
        f"The fixed sealed cohort contains {dataset['scan_row_count']} scans from "
        f"{dataset['patient_count']} patients. The reporting grain is one row per "
        f"scan plus an authenticated {tuple(dataset['array_shape_per_scan'])} 4D "
        "real/predicted array pair. No fMRI resampling is permitted.",
        "",
        "Completeness, uniqueness, fixed order, no-replacement/no-refill, required "
        "file hashes, finite [0,1] domain, shape, affine, RAS+ orientation, and "
        "TR=3 s all pass at 34/34; otherwise this bundle could not be published.",
        "",
        "## Cohort quality rates",
        "",
        "| Measure | Count/rate |",
        "|---|---:|",
        f"| Overall paired quality pass | {rates['quality_pass_count']}/34 "
        f"({rates['quality_pass_rate']:.1%}) |",
        f"| Enforced texture + anti-gaming gate pass | "
        f"{rates['texture_release_gate_pass_count']}/34 "
        f"({rates['texture_release_gate_pass_rate']:.1%}) |",
        f"| Temporal collapse | {rates['temporal_collapse_count']}/34 "
        f"({rates['temporal_collapse_rate']:.1%}) |",
        f"| Spatial support/texture failure | {rates['spatial_quality_failure_rate']:.1%} |",
        f"| Temporal dynamics failure | {rates['temporal_quality_failure_rate']:.1%} |",
        f"| ROI structured-dynamics failure | {rates['structured_dynamics_failure_rate']:.1%} |",
        "",
        "## Predicted/real ratio summary",
        "",
        "| Ratio | Median | IQR | Range | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, _section, _value, _check, label in _QUALITY_RATIO_SPECS:
        metric = ratios[key]
        distribution = metric["distribution"]
        lines.append(
            f"| {label} | {distribution['median']:.4g} | "
            f"{distribution['q25']:.4g}-{distribution['q75']:.4g} | "
            f"{distribution['minimum']:.4g}-{distribution['maximum']:.4g} | "
            f"{metric['passing_scan_rate']:.1%} |"
        )
    distributional = summary["distributional_metrics"]
    distributional_note = (
        "- FID and Inception Score were computed over all 34 authenticated stored "
        "prediction/target volumes with the externally pinned frozen SLIM-Brain "
        "authority; no feature mask or resampling was used."
        if distributional.get("available") is True
        else "- FID and Inception Score are unavailable until a closed pinned "
        "evaluation feature-model authority is qualified; no substitute is allowed."
    )
    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )
    for finding in summary["findings"]:
        lines.append(
            f"- **{finding['dimension']}** — {finding['status']}; severity "
            f"{finding['severity']}; confidence {finding['confidence']}; "
            f"passing rate {finding['passing_scan_rate']:.1%}."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            f"- {summary['threshold_note']}",
            "- A passing quality gate does not itself reproduce or equal the paper's "
            "reported numerical results.",
            distributional_note,
            "- This evaluation cannot feed back into training, prediction, candidate, "
            "model, or checkpoint selection.",
            "",
            "See `cohort_quality_summary.png` and each subject directory for the "
            "fixed-scale real/predicted/residual, temporal, high-pass, leakage, "
            "ROI-FC/spectrum, and all-frame animation outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_new(path: Path, value: str) -> FileSnapshot:
    payload = value.encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _snapshot_file(path, label=f"published {path.name}")[0]


def _freeze_staged_tree(root: Path) -> tuple[dict[str, Any], ...]:
    """Make a completed staging tree read-only and capture every inode/hash."""
    entries = sorted(root.rglob("*"), key=lambda path: (len(path.parts), str(path)))
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PostsealEvaluationError("staged publication contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise PostsealEvaluationError(
                    "staged publication contains a hard-linked file"
                )
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o400)
            finally:
                os.close(descriptor)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PostsealEvaluationError(
                "staged publication contains a non-file filesystem object"
            )
    for path in sorted(
        [root, *(item for item in entries if item.is_dir())],
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(path, 0o500)
    state: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            state.append(
                {
                    "relative_path": relative,
                    "kind": "directory",
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mode": stat.S_IMODE(metadata.st_mode),
                }
            )
        else:
            snapshot, _ = _snapshot_file(path, label=f"frozen {relative}")
            state.append(
                {
                    "relative_path": relative,
                    "kind": "file",
                    "device": snapshot.device,
                    "inode": snapshot.inode,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "sha256": snapshot.sha256,
                    "size_bytes": snapshot.size_bytes,
                }
            )
    return tuple(state)


def _reauthenticate_frozen_tree(
    root: Path, expected: Sequence[Mapping[str, Any]]
) -> None:
    observed_paths = [
        path.relative_to(root).as_posix() for path in sorted(root.rglob("*"))
    ]
    expected_paths = [str(row.get("relative_path")) for row in expected]
    if observed_paths != expected_paths:
        raise PostsealEvaluationError("frozen publication tree set changed")
    for path, row in zip(sorted(root.rglob("*")), expected):
        metadata = path.lstat()
        expected_kind = row.get("kind")
        if (
            stat.S_ISLNK(metadata.st_mode)
            or (
                expected_kind == "file"
                and (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o400
                )
            )
            or (
                expected_kind == "directory"
                and (
                    not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o500
                )
            )
        ):
            raise PostsealEvaluationError("frozen publication tree type/mode changed")
        if (metadata.st_dev, metadata.st_ino) != (
            row.get("device"),
            row.get("inode"),
        ):
            raise PostsealEvaluationError("frozen publication inode changed")
        if expected_kind == "file":
            _snapshot_file(
                path,
                label=f"precommit {row['relative_path']}",
                expected_sha256=str(row.get("sha256", "")),
                expected_size=row.get("size_bytes"),
            )


def _remove_staged_tree(root: Path) -> None:
    """Best-effort removal for a private staging tree, including frozen trees."""
    for directory, child_directories, file_names in os.walk(root, topdown=False):
        for file_name in file_names:
            try:
                os.chmod(Path(directory) / file_name, 0o600)
            except OSError:
                pass
        for child in child_directories:
            try:
                os.chmod(Path(directory) / child, 0o700)
            except OSError:
                pass
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    shutil.rmtree(root)


@dataclass
class _HeldStagingRoot:
    """Retain the staging root and parent identities through durable publication."""

    source: Path
    destination: Path
    parent_descriptor: int
    root_descriptor: int
    parent_identity: tuple[int, int]
    root_identity: tuple[int, int]
    renamed: bool = False

    @classmethod
    def open(cls, source: Path, destination: Path) -> "_HeldStagingRoot":
        if source.parent != destination.parent:
            raise PostsealEvaluationError(
                "staging and destination must share one retained parent"
            )
        parent = _canonical_path(
            source.parent, label="evaluation staging parent", directory=True
        )
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        root_descriptor = -1
        try:
            root_descriptor = os.open(
                source.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            parent_metadata = os.fstat(parent_descriptor)
            root_metadata = os.fstat(root_descriptor)
            held = cls(
                source=source,
                destination=destination,
                parent_descriptor=parent_descriptor,
                root_descriptor=root_descriptor,
                parent_identity=(
                    int(parent_metadata.st_dev),
                    int(parent_metadata.st_ino),
                ),
                root_identity=(int(root_metadata.st_dev), int(root_metadata.st_ino)),
            )
            held.assert_source_identity()
            return held
        except Exception:
            if root_descriptor >= 0:
                os.close(root_descriptor)
            os.close(parent_descriptor)
            raise

    def close(self) -> None:
        if self.root_descriptor >= 0:
            os.close(self.root_descriptor)
            self.root_descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1

    def _assert_parent_identity(self) -> None:
        try:
            opened = os.fstat(self.parent_descriptor)
            current = self.source.parent.lstat()
        except OSError as exc:
            raise PostsealEvaluationError(
                "evaluation staging parent was replaced"
            ) from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != self.parent_identity
            or (current.st_dev, current.st_ino) != self.parent_identity
        ):
            raise PostsealEvaluationError("evaluation staging parent was replaced")

    def _assert_named_identity(self, name: str, *, label: str) -> None:
        self._assert_parent_identity()
        try:
            opened = os.fstat(self.root_descriptor)
            current = os.stat(
                name, dir_fd=self.parent_descriptor, follow_symlinks=False
            )
        except OSError as exc:
            raise PostsealEvaluationError(f"{label} was replaced") from exc
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (opened.st_dev, opened.st_ino) != self.root_identity
            or (current.st_dev, current.st_ino) != self.root_identity
        ):
            raise PostsealEvaluationError(f"{label} was replaced")
        self._assert_parent_identity()

    def assert_source_identity(self) -> None:
        if self.renamed:
            raise PostsealEvaluationError("staging root was already published")
        self._assert_named_identity(self.source.name, label="evaluation staging root")

    def assert_destination_identity(self) -> None:
        if not self.renamed:
            raise PostsealEvaluationError("evaluation destination is not committed")
        self._assert_named_identity(
            self.destination.name, label="committed evaluation root"
        )

    def cleanup_owned_source(self) -> bool:
        """Remove only the still-bound source; preserve any attacker-detached root."""
        if self.renamed:
            return False
        try:
            self.assert_source_identity()
        except PostsealEvaluationError:
            return False
        _remove_staged_tree(self.source)
        return True


def _renameat_noreplace(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    """Invoke the audited descriptor-relative no-replace rename primitive."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        function = libc.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    else:  # pragma: no cover - production platforms are Linux and macOS
        raise PostsealEvaluationError(
            "this platform has no audited atomic no-replace directory rename"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"evaluation output already exists: {destination_name}"
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _publish_held_staging_root(
    held: _HeldStagingRoot,
    expected_frozen_tree: Sequence[Mapping[str, Any]],
) -> None:
    """Reauthenticate and durably publish the exact descriptor-held root."""
    held.assert_source_identity()
    _reauthenticate_frozen_tree(held.source, expected_frozen_tree)
    held.assert_source_identity()
    try:
        _renameat_noreplace(
            held.parent_descriptor, held.source.name, held.destination.name
        )
    except BaseException as exc:
        # If control was interrupted after the kernel committed the rename, the
        # retained root descriptor lets us classify it as committed/uncertain
        # instead of an ordinary precommit failure.
        try:
            held._assert_named_identity(  # noqa: SLF001
                held.destination.name, label="committed evaluation root"
            )
        except PostsealEvaluationError:
            raise exc
        held.renamed = True
        raise PostsealPublicationUncertainError(
            "evaluation directory rename committed no-replace, but completion "
            f"of the publication primitive is uncertain: {held.destination}"
        ) from exc
    held.renamed = True
    try:
        held.assert_destination_identity()
        try:
            os.fsync(held.parent_descriptor)
        except OSError as exc:
            raise PostsealPublicationUncertainError(
                "evaluation directory was atomically committed no-replace, but "
                f"parent-directory durability is uncertain: {held.destination}"
            ) from exc
        held.assert_destination_identity()
        _reauthenticate_frozen_tree(held.destination, expected_frozen_tree)
        held.assert_destination_identity()
    except PostsealPublicationUncertainError:
        raise
    except Exception as exc:
        raise PostsealPublicationUncertainError(
            "evaluation directory was atomically committed no-replace, but the "
            f"committed root identity/tree is uncertain: {held.destination}"
        ) from exc


def _reopen_external_artifact(
    value: object,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_path: str | None = None,
) -> FileSnapshot:
    if type(value) is not dict or set(value) != {"path", "sha256", "size_bytes"}:
        raise PostsealEvaluationError(f"{label} descriptor fields differ")
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise PostsealEvaluationError(f"{label} external pin differs")
    size = value.get("size_bytes")
    if type(size) is not int or size < 1:
        raise PostsealEvaluationError(f"{label} size differs")
    snapshot, _ = _snapshot_file(
        value.get("path", ""),
        label=label,
        expected_sha256=digest,
        expected_size=size,
    )
    if expected_path is not None and snapshot.path != Path(expected_path):
        raise PostsealEvaluationError(f"{label} external path pin differs")
    return snapshot


def _reopen_external_signed_artifact(
    value: object,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_path: str | None = None,
) -> tuple[dict[str, Any], FileSnapshot]:
    if type(value) is not dict or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "record_sha256",
    }:
        raise PostsealEvaluationError(f"{label} signed descriptor fields differ")
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise PostsealEvaluationError(f"{label} external pin differs")
    record_digest = _require_sha256(
        value.get("record_sha256"), label=f"{label} record SHA-256"
    )
    size = value.get("size_bytes")
    if type(size) is not int or size < 1:
        raise PostsealEvaluationError(f"{label} size differs")
    record, snapshot = _signed_json_snapshot(
        value.get("path", ""),
        label=label,
        expected_sha256=digest,
        expected_size=size,
    )
    if record.get("record_sha256") != record_digest:
        raise PostsealEvaluationError(f"{label} record SHA-256 differs")
    if expected_path is not None and snapshot.path != Path(expected_path):
        raise PostsealEvaluationError(f"{label} external path pin differs")
    return record, snapshot


def _reopen_prediction_publication_authority(
    evaluation: Mapping[str, Any], *, pins: _PublicationAuthorityPins
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    descriptor = evaluation.get("prediction_set_seal")
    seal, seal_snapshot = _reopen_external_signed_artifact(
        descriptor,
        label="external prediction-set seal",
        expected_sha256=pins.prediction_set_seal_sha256,
        expected_path=pins.prediction_set_seal_path,
    )
    if seal_snapshot.path.name != "prediction_set_seal.json":
        raise PostsealEvaluationError("external prediction-set seal path differs")
    checkpoint_bindings = evaluation.get("checkpoint_bindings")
    if type(checkpoint_bindings) is not dict or set(checkpoint_bindings) != set(
        _CHECKPOINT_BINDING_FIELDS
    ):
        raise PostsealEvaluationError("evaluation checkpoint bindings differ")
    frozen_pins = _EvaluationPinSnapshot(
        prediction_set_seal_sha256=pins.prediction_set_seal_sha256,
        checkpoint_items=tuple(
            (name, _require_sha256(checkpoint_bindings[name], label=name))
            for name in _CHECKPOINT_BINDING_FIELDS
        ),
    )
    _authenticate_prediction_set_impl(seal_snapshot.path.parent, pins=frozen_pins)
    if descriptor != seal_snapshot.descriptor() | {
        "record_sha256": seal["record_sha256"]
    }:
        raise PostsealEvaluationError("prediction-set seal publication binding differs")
    predictions: dict[str, dict[str, Any]] = {}
    for binding in seal["subjects"]:
        scan_id = _safe_scan_id(binding.get("scan_id"))
        subject_root = seal_snapshot.path.parent / "subjects" / scan_id
        record, record_snapshot = _signed_json_snapshot(
            subject_root / "prediction.json",
            label=f"{scan_id} external prediction record",
            expected_schema_key="format",
            expected_schema=PREDICTION_FORMAT,
        )
        prediction_descriptor = record["prediction"]
        prediction_snapshot, _ = _snapshot_file(
            subject_root / prediction_descriptor["relative_path"].split("/")[-1],
            label=f"{scan_id} external prediction",
            expected_sha256=prediction_descriptor["sha256"],
            expected_size=prediction_descriptor["size_bytes"],
        )
        predictions[scan_id] = {
            "prediction": prediction_snapshot.descriptor(),
            "prediction_record": record_snapshot.descriptor()
            | {"record_sha256": record["record_sha256"]},
            "protocol_profile": record["protocol_profile"],
        }
    return seal, predictions


def _stage_b_completed_subjects(
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required_fields = {
        "schema",
        "prediction_set_seal",
        "checkpoint_bindings",
        "sealed_target_worklist",
        "expected_count",
        "success_count",
        "quarantine_count",
        "ordered_scan_ids",
        "ordered_scan_ids_sha256",
        "subjects",
        "quarantines",
        "complete",
        "all_34_successful",
        "no_replacement",
        "no_refill",
        "evaluation_only",
        "authorizes_model_training",
        "authorizes_checkpoint_selection",
        "authorizes_model_or_candidate_selection",
        "authorizes_prediction_emission",
        "authorizes_additional_inference",
        "authorizes_training_or_model_feedback",
        "paper_certified",
        "record_sha256",
    }
    subjects = record.get("subjects")
    expected = list(EXPECTED_SCAN_IDS)
    if (
        type(record) is not dict
        or set(record) != required_fields
        or record.get("schema") != STAGE_B_COMPLETED_SET_SCHEMA
        or record.get("expected_count") != len(EXPECTED_SCAN_IDS)
        or record.get("success_count") != len(EXPECTED_SCAN_IDS)
        or record.get("quarantine_count") != 0
        or record.get("ordered_scan_ids") != expected
        or record.get("ordered_scan_ids_sha256") != EXPECTED_SCAN_IDS_SHA256
        or type(subjects) is not list
        or len(subjects) != len(EXPECTED_SCAN_IDS)
        or [item.get("scan_id") if type(item) is dict else None for item in subjects]
        != expected
        or record.get("quarantines") != []
        or record.get("complete") is not True
        or record.get("all_34_successful") is not True
        or record.get("no_replacement") is not True
        or record.get("no_refill") is not True
        or record.get("evaluation_only") is not True
        or record.get("paper_certified") is not False
        or any(
            record.get(field) is not False
            for field in (
                "authorizes_model_training",
                "authorizes_checkpoint_selection",
                "authorizes_model_or_candidate_selection",
                "authorizes_prediction_emission",
                "authorizes_additional_inference",
                "authorizes_training_or_model_feedback",
            )
        )
    ):
        raise PostsealEvaluationError(
            "Stage-B completed-set schema/fields/semantics differ"
        )
    return [dict(item) for item in subjects]


def _reopen_stage_b_publication_authority(
    evaluation: Mapping[str, Any],
    *,
    pins: _PublicationAuthorityPins,
    prediction_seal: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    completed_descriptor = evaluation.get("stage_b_completed_set")
    completed, completed_snapshot = _reopen_external_signed_artifact(
        completed_descriptor,
        label="external Stage-B completed set",
        expected_sha256=pins.stage_b_completed_set_sha256,
        expected_path=pins.stage_b_completed_set_path,
    )
    completed_subjects = _stage_b_completed_subjects(completed)
    if completed.get("prediction_set_seal") != evaluation.get(
        "prediction_set_seal"
    ) or completed.get("checkpoint_bindings") != evaluation.get("checkpoint_bindings"):
        raise PostsealEvaluationError(
            "Stage-B completed-set prediction/checkpoint binding differs"
        )
    worklist, _worklist_snapshot = _reopen_external_signed_artifact(
        completed.get("sealed_target_worklist"),
        label="external Stage-B sealed-target worklist",
    )
    if worklist.get("schema") != STAGE_B_WORKLIST_SCHEMA:
        raise PostsealEvaluationError("Stage-B sealed-target worklist schema differs")
    prediction_bindings = {
        item["scan_id"]: item for item in prediction_seal["subjects"]
    }

    source = _reopen_external_artifact(
        evaluation.get("stage_b_verifier_source"),
        label="external Stage-B verifier source",
        expected_sha256=pins.stage_b_verifier_source_sha256,
        expected_path=pins.stage_b_verifier_source_path,
    )
    dependency_values = evaluation.get("stage_b_verifier_dependencies")
    if type(dependency_values) is not list or len(dependency_values) != len(
        pins.stage_b_verifier_dependency_sha256s
    ):
        raise PostsealEvaluationError(
            "external Stage-B verifier dependency order/count differs"
        )
    dependencies = [
        _reopen_external_artifact(
            descriptor,
            label=f"external Stage-B verifier dependency {index}",
            expected_sha256=pins.stage_b_verifier_dependency_sha256s[index],
            expected_path=pins.stage_b_verifier_dependency_paths[index],
        )
        for index, descriptor in enumerate(dependency_values)
    ]
    authority_paths = [source.path, *(item.path for item in dependencies)]
    if len(authority_paths) != len(set(authority_paths)):
        raise PostsealEvaluationError("Stage-B verifier authority paths alias")

    verification = evaluation.get("stage_b_verification")
    required_fields = {
        "schema",
        "completed_set",
        "prediction_set_seal_sha256",
        "prediction_set_seal_record_sha256",
        "checkpoint_bindings",
        "ordered_scan_ids",
        "ordered_scan_ids_sha256",
        "subjects",
        "complete",
        "no_replacement",
        "no_refill",
        "evaluation_only",
        "authorizes_model_training",
        "authorizes_checkpoint_selection",
        "authorizes_model_or_candidate_selection",
        "authorizes_prediction_emission",
        "authorizes_additional_inference",
        "record_sha256",
    }
    if type(verification) is not dict or set(verification) != required_fields:
        raise PostsealEvaluationError("published Stage-B verification fields differ")
    unsigned_verification = dict(verification)
    recorded_verification_sha = unsigned_verification.pop("record_sha256", None)
    prediction_descriptor = evaluation["prediction_set_seal"]
    if (
        verification.get("schema") != STAGE_B_VERIFICATION_SCHEMA
        or recorded_verification_sha != canonical_sha256(unsigned_verification)
        or verification.get("completed_set") != completed_descriptor
        or completed_snapshot.sha256 != pins.stage_b_completed_set_sha256
        or verification.get("prediction_set_seal_sha256")
        != pins.prediction_set_seal_sha256
        or verification.get("prediction_set_seal_record_sha256")
        != prediction_descriptor["record_sha256"]
        or verification.get("checkpoint_bindings")
        != evaluation.get("checkpoint_bindings")
        or verification.get("ordered_scan_ids") != list(EXPECTED_SCAN_IDS)
        or verification.get("ordered_scan_ids_sha256") != EXPECTED_SCAN_IDS_SHA256
        or verification.get("complete") is not True
        or verification.get("no_replacement") is not True
        or verification.get("no_refill") is not True
        or verification.get("evaluation_only") is not True
        or any(
            verification.get(field) is not False
            for field in (
                "authorizes_model_training",
                "authorizes_checkpoint_selection",
                "authorizes_model_or_candidate_selection",
                "authorizes_prediction_emission",
                "authorizes_additional_inference",
            )
        )
    ):
        raise PostsealEvaluationError("published Stage-B verification differs")
    subject_values = verification.get("subjects")
    if type(subject_values) is not list or len(subject_values) != len(
        EXPECTED_SCAN_IDS
    ):
        raise PostsealEvaluationError("published Stage-B subject set differs")
    stage_subjects: dict[str, dict[str, Any]] = {}
    seen_paths: dict[str, set[Path]] = {
        name: set()
        for name in (
            "native_bold",
            "native_structural_brain_mask",
            "native_structural_labels",
            "publication",
            "success_receipt",
            "native_preprocessing_receipt",
        )
    }
    all_subject_paths: set[Path] = set()
    completed_subject_fields = {
        "global_index",
        "scan_id",
        "BID",
        "role",
        "success_receipt",
        "success_receipt_commit_marker",
        "native_preprocessing_receipt",
        "publication",
        "native_bold",
        "native_structural_labels",
        "padded_bold",
        "sidecar",
        "prediction_subject_record_sha256",
        "prediction_sha256",
    }
    for index, (expected_scan_id, value, completed_value) in enumerate(
        zip(EXPECTED_SCAN_IDS, subject_values, completed_subjects)
    ):
        fields = {"scan_id", *seen_paths}
        if type(value) is not dict or set(value) != fields:
            raise PostsealEvaluationError("published Stage-B subject fields differ")
        scan_id = _safe_scan_id(value.get("scan_id"), label="Stage-B scan ID")
        if scan_id != expected_scan_id:
            raise PostsealEvaluationError("published Stage-B subject order differs")
        prediction_binding = prediction_bindings[scan_id]
        if (
            type(completed_value) is not dict
            or set(completed_value) != completed_subject_fields
            or completed_value.get("global_index") != index
            or type(completed_value.get("BID")) is not str
            or not completed_value["BID"]
            or completed_value.get("role") != "sealed-test"
            or completed_value.get("scan_id") != scan_id
            or completed_value.get("prediction_subject_record_sha256")
            != prediction_binding["subject_record_sha256"]
            or completed_value.get("prediction_sha256")
            != prediction_binding["prediction_sha256"]
            or completed_value.get("native_bold") != value.get("native_bold")
            or completed_value.get("native_structural_labels")
            != value.get("native_structural_labels")
            or completed_value.get("publication") != value.get("publication")
            or completed_value.get("success_receipt") != value.get("success_receipt")
            or completed_value.get("native_preprocessing_receipt")
            != value.get("native_preprocessing_receipt")
        ):
            raise PostsealEvaluationError(
                f"{scan_id} Stage-B completed/verification binding differs"
            )
        for extra_role, signed in (
            ("success_receipt_commit_marker", True),
            ("padded_bold", False),
            ("sidecar", False),
        ):
            extra_descriptor = completed_value.get(extra_role)
            if signed:
                _record, extra_snapshot = _reopen_external_signed_artifact(
                    extra_descriptor,
                    label=f"{scan_id} external Stage-B {extra_role}",
                )
            else:
                extra_snapshot = _reopen_external_artifact(
                    extra_descriptor,
                    label=f"{scan_id} external Stage-B {extra_role}",
                )
            if extra_snapshot.path in all_subject_paths:
                raise PostsealEvaluationError(
                    f"Stage-B external artifact path is reused: {scan_id} {extra_role}"
                )
            all_subject_paths.add(extra_snapshot.path)
        for role in (
            "native_bold",
            "native_structural_brain_mask",
            "native_structural_labels",
        ):
            snapshot = _reopen_external_artifact(
                value.get(role), label=f"{scan_id} external Stage-B {role}"
            )
            if snapshot.path in seen_paths[role]:
                raise PostsealEvaluationError(f"Stage-B {role} path is reused")
            if snapshot.path in all_subject_paths:
                raise PostsealEvaluationError(
                    f"Stage-B external artifact path is reused: {scan_id} {role}"
                )
            seen_paths[role].add(snapshot.path)
            all_subject_paths.add(snapshot.path)
        for role in (
            "publication",
            "success_receipt",
            "native_preprocessing_receipt",
        ):
            _record, snapshot = _reopen_external_signed_artifact(
                value.get(role), label=f"{scan_id} external Stage-B {role}"
            )
            if snapshot.path in seen_paths[role]:
                raise PostsealEvaluationError(f"Stage-B {role} path is reused")
            if snapshot.path in all_subject_paths:
                raise PostsealEvaluationError(
                    f"Stage-B external artifact path is reused: {scan_id} {role}"
                )
            seen_paths[role].add(snapshot.path)
            all_subject_paths.add(snapshot.path)
        stage_subjects[scan_id] = dict(value)
    return stage_subjects


def _verify_evaluation_publication_impl(
    root: Path,
    *,
    evaluation_pin: str,
    receipt_pin: str,
    logical_root: Path,
    authority_pins: _PublicationAuthorityPins,
    preflight_pair_contracts: Mapping[str, Mapping[str, Any]],
    preflight_execution_evidence: Mapping[str, Any],
    distributional_authority_pin: str | None = None,
) -> dict[str, Any]:
    """Internal verifier with an already authenticated physical/logical root."""
    evaluation, evaluation_snapshot = _signed_json_snapshot(
        root / "evaluation.json",
        label="evaluation root record",
        expected_sha256=evaluation_pin,
        expected_schema_key="schema",
        expected_schema=EVALUATION_SCHEMA,
    )
    receipt, receipt_snapshot = _signed_json_snapshot(
        root / "publication_receipt.json",
        label="evaluation publication receipt",
        expected_sha256=receipt_pin,
        expected_schema_key="schema",
        expected_schema=PUBLICATION_RECEIPT_SCHEMA,
    )
    evaluation_binding = receipt.get("evaluation")
    distributional = evaluation.get("distributional_metrics")
    evaluation_feature_model_loaded = evaluation.get("evaluation_feature_model_loaded")
    if (
        set(evaluation) != _ROOT_EVALUATION_FIELDS
        or set(receipt) != _PUBLICATION_RECEIPT_FIELDS
        or evaluation_snapshot.sha256 != evaluation_pin
        or receipt_snapshot.sha256 != receipt_pin
        or not isinstance(evaluation_binding, dict)
        or set(evaluation_binding)
        != {
            "relative_path",
            "sha256",
            "size_bytes",
            "record_sha256",
        }
    ):
        raise PostsealEvaluationError("evaluation receipt binding fields differ")
    if (
        receipt.get("status") != "COMMITTED_NO_REPLACE"
        or receipt.get("destination") != str(logical_root)
        or evaluation_binding.get("relative_path") != "evaluation.json"
        or evaluation_binding.get("sha256") != evaluation_snapshot.sha256
        or evaluation_binding.get("size_bytes") != evaluation_snapshot.size_bytes
        or evaluation_binding.get("record_sha256") != evaluation.get("record_sha256")
        or receipt.get("output_inventory_sha256")
        != evaluation.get("output_inventory_sha256")
        or receipt.get("all_34_subjects_complete") is not True
        or receipt.get("all_34_texture_retention_and_anti_gaming_gates_passed")
        is not True
        or receipt.get("published_after_all_subjects_complete") is not True
        or receipt.get("no_overwrite") is not True
        or receipt.get("evaluation_feedback_to_training_or_selection") is not False
        or evaluation.get("status") != "COMPLETE_POSTSEAL_HELDOUT_EVALUATION"
        or evaluation.get("role") != "fixed-sealed-test-final-report-only"
        or evaluation.get("sealed_role_csv_sha256") != SEALED_ROLE_CSV_SHA256
        or evaluation.get("production_evaluator_authenticated_before_import")
        is not True
        or evaluation.get("prediction_set_fully_authenticated_before_any_target_access")
        is not True
        or evaluation.get(
            "all_target_publications_and_bytes_authenticated_before_metrics"
        )
        is not True
        or evaluation.get("all_targets_authenticated_before_texture_audits") is not True
        or evaluation.get("exact_native_grid_no_fmri_resampling") is not True
        or evaluation.get("independent_structural_mask_and_roi_labels_only") is not True
        or evaluation.get("synthesis_model_or_checkpoint_loaded") is not False
        or evaluation.get("training_or_inference_entrypoint_imported") is not False
        or any(
            evaluation.get(field) is not False
            for field in (
                "authorizes_training",
                "authorizes_model_selection",
                "authorizes_candidate_selection",
                "authorizes_checkpoint_selection",
                "authorizes_prediction",
                "authorizes_inference",
                "can_change_checkpoint_or_prediction_set",
            )
        )
    ):
        raise PostsealEvaluationError("evaluation publication receipt differs")
    prediction_seal, prediction_subjects = _reopen_prediction_publication_authority(
        evaluation, pins=authority_pins
    )
    stage_b_subjects = _reopen_stage_b_publication_authority(
        evaluation, pins=authority_pins, prediction_seal=prediction_seal
    )
    if set(preflight_pair_contracts) != set(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError("external complete-pair preflight set differs")
    independently_derived_pair_contracts: dict[str, dict[str, Any]] = {}
    for scan_id in EXPECTED_SCAN_IDS:
        prediction = prediction_subjects[scan_id]
        target = stage_b_subjects[scan_id]
        contract = validate_exact_pair(
            Path(target["native_bold"]["path"]),
            Path(prediction["prediction"]["path"]),
            Path(target["native_structural_brain_mask"]["path"]),
            Path(target["native_structural_labels"]["path"]),
            protocol_profile=prediction["protocol_profile"],
        )
        if contract != preflight_pair_contracts[scan_id]:
            raise PostsealEvaluationError(
                f"{scan_id} exact-pair contract changed after cohort preflight"
            )
        independently_derived_pair_contracts[scan_id] = contract
    from .postseal_execution import authenticate_published_execution_evidence

    published_execution_evidence = authenticate_published_execution_evidence(
        evaluation.get("production_runtime_and_scheduler_evidence"),
        authority_path=authority_pins.execution_authority_path,
        authority_sha256=authority_pins.execution_authority_sha256,
    )
    if published_execution_evidence != preflight_execution_evidence:
        raise PostsealEvaluationError(
            "published execution evidence differs from the cohort preflight"
        )
    distributional_available = _validate_distributional_evidence(
        distributional,
        expected_authority_sha256=distributional_authority_pin,
        artifact_root=root,
    )
    if evaluation_feature_model_loaded is not distributional_available:
        raise PostsealEvaluationError(
            "evaluation feature-model status differs from distributional evidence"
        )
    if distributional_available:
        distributional_bindings = distributional["input_bindings"]
        prediction_set_binding = evaluation.get("prediction_set_seal")
        stage_b_binding = evaluation.get("stage_b_completed_set")
        if (
            not isinstance(prediction_set_binding, dict)
            or not isinstance(stage_b_binding, dict)
            or distributional_bindings["prediction_set_seal_sha256"]
            != prediction_set_binding.get("sha256")
            or distributional_bindings["stage_b_completed_set_sha256"]
            != stage_b_binding.get("sha256")
        ):
            raise PostsealEvaluationError(
                "distributional prediction/target set bindings differ"
            )
    if (
        prediction_seal.get("record_sha256")
        != evaluation["prediction_set_seal"]["record_sha256"]
    ):
        raise PostsealEvaluationError("prediction-set seal record binding differs")
    inventory = evaluation.get("output_inventory")
    if not isinstance(inventory, list):
        raise PostsealEvaluationError("evaluation output inventory is missing")
    observed = _relative_inventory(
        root, excluded={"evaluation.json", "publication_receipt.json"}
    )
    if (
        inventory != observed
        or evaluation.get("output_inventory_sha256") != canonical_sha256(observed)
        or evaluation.get("ordered_scan_ids") != list(EXPECTED_SCAN_IDS)
        or evaluation.get("ordered_scan_ids_sha256") != EXPECTED_SCAN_IDS_SHA256
        or evaluation.get("scan_count") != len(EXPECTED_SCAN_IDS)
    ):
        raise PostsealEvaluationError("evaluation bundle inventory/cohort differs")
    if evaluation.get("implementation_sources") != _source_inventory():
        raise PostsealEvaluationError(
            "evaluation implementation-source bindings differ"
        )
    subjects = evaluation.get("subjects")
    if not isinstance(subjects, list) or [
        item.get("scan_id") if isinstance(item, dict) else None for item in subjects
    ] != list(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError("evaluation subject order/set differs")
    for subject in subjects:
        scan_id = _safe_scan_id(subject.get("scan_id"))
        descriptor = subject.get("record")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "record_sha256",
        }:
            raise PostsealEvaluationError(f"{scan_id} evaluation descriptor differs")
        expected_relative = f"subjects/{scan_id}/evaluation.json"
        if descriptor.get("relative_path") != expected_relative:
            raise PostsealEvaluationError(f"{scan_id} evaluation path differs")
        record, snapshot = _signed_json_snapshot(
            root / expected_relative,
            label=f"{scan_id} subject evaluation",
            expected_sha256=str(descriptor.get("sha256", "")),
            expected_size=descriptor.get("size_bytes"),
            expected_schema_key="schema",
            expected_schema=SUBJECT_EVALUATION_SCHEMA,
        )
        if (
            snapshot.sha256 != descriptor.get("sha256")
            or record.get("record_sha256") != descriptor.get("record_sha256")
            or record.get("scan_id") != scan_id
            or set(record) != _SUBJECT_EVALUATION_FIELDS
            or record.get("fmri_resampling_performed") is not False
            or record.get("evaluation_can_authorize_training_or_selection") is not False
        ):
            raise PostsealEvaluationError(f"{scan_id} evaluation binding differs")
        prediction_authority = prediction_subjects[scan_id]
        stage_b_authority = stage_b_subjects[scan_id]
        if (
            record.get("prediction") != prediction_authority["prediction"]
            or record.get("prediction_record")
            != prediction_authority["prediction_record"]
            or record.get("target") != stage_b_authority["native_bold"]
            or record.get("structural_brain_mask")
            != stage_b_authority["native_structural_brain_mask"]
            or record.get("structural_labels")
            != stage_b_authority["native_structural_labels"]
            or record.get("stage_b_publication") != stage_b_authority["publication"]
            or record.get("stage_b_success_receipt")
            != stage_b_authority["success_receipt"]
            or record.get("native_preprocessing_receipt")
            != stage_b_authority["native_preprocessing_receipt"]
        ):
            raise PostsealEvaluationError(
                f"{scan_id} external prediction/Stage-B binding differs"
            )
        target_validity_descriptor = record.get("target_validity_mask")
        pair_contract = record.get("pair_contract")
        pair_protocol_profile = (
            pair_contract.get("protocol_profile")
            if isinstance(pair_contract, dict)
            else None
        )
        expected_target_validity_contract = (
            TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(pair_protocol_profile)
            if type(pair_protocol_profile) is str
            else None
        )
        target_validity_relative = f"subjects/{scan_id}/target_validity_mask.nii.gz"
        if (
            pair_contract != independently_derived_pair_contracts[scan_id]
            or pair_protocol_profile != prediction_authority["protocol_profile"]
            or pair_contract.get("canonical_roi_mapping_sha256")
            != CANONICAL_ROI_MAPPING_SHA256
            or not isinstance(target_validity_descriptor, dict)
            or set(target_validity_descriptor)
            != {
                "path",
                "sha256",
                "size_bytes",
                "contract",
                "derivation",
                "equals_exact_final_nonzero_support",
            }
            or target_validity_descriptor.get("path")
            != str(logical_root / target_validity_relative)
            or expected_target_validity_contract is None
            or target_validity_descriptor.get("contract")
            != expected_target_validity_contract
            or target_validity_descriptor.get("derivation")
            != (
                "exact-nonzero-support-across-final-stored-target-time-series-"
                "after-complete-target-authentication"
            )
            or target_validity_descriptor.get("equals_exact_final_nonzero_support")
            is not True
        ):
            raise PostsealEvaluationError(
                f"{scan_id} target-validity artifact binding differs"
            )
        validity_snapshot, _ = _snapshot_file(
            root / target_validity_relative,
            label=f"{scan_id} published target-validity mask",
            expected_sha256=str(target_validity_descriptor.get("sha256", "")),
            expected_size=target_validity_descriptor.get("size_bytes"),
        )
        _validity_image, target_validity_values, _validity_geometry = (
            _strict_nifti_values(
                validity_snapshot.path,
                label=f"{scan_id} published target-validity mask",
                expected_shape=EXPECTED_SPATIAL_SHAPE,
                is_binary_mask=True,
            )
        )
        pair_target_validity = (
            pair_contract.get("target_validity_mask")
            if isinstance(pair_contract, dict)
            else None
        )
        if (
            not isinstance(pair_target_validity, dict)
            or pair_target_validity.get("foreground_voxel_count")
            != int(np.count_nonzero(target_validity_values))
            or pair_target_validity.get("contract") != expected_target_validity_contract
            or pair_target_validity.get("equals_exact_final_nonzero_support")
            is not True
        ):
            raise PostsealEvaluationError(
                f"{scan_id} target-validity artifact/pair contract differs"
            )
        texture_evidence = record.get("texture_audit")
        if not isinstance(texture_evidence, dict):
            raise PostsealEvaluationError(f"{scan_id} texture evidence is missing")
        _compact_texture_gate(scan_id, texture_evidence)
        texture_outputs = texture_evidence["outputs"]
        png_descriptor = texture_outputs["png"]
        manifest_descriptor = texture_outputs["manifest"]
        _snapshot_file(
            root / str(png_descriptor["relative_path"]),
            label=f"{scan_id} published texture PNG",
            expected_sha256=str(png_descriptor["sha256"]),
            expected_size=png_descriptor["size_bytes"],
        )
        texture_manifest, texture_manifest_snapshot = _signed_json_snapshot(
            root / str(manifest_descriptor["relative_path"]),
            label=f"{scan_id} published texture manifest",
            expected_sha256=str(manifest_descriptor["sha256"]),
            expected_size=manifest_descriptor["size_bytes"],
            expected_schema_key="schema",
            expected_schema=TEXTURE_AUDIT_SCHEMA,
        )
        texture_gate = texture_manifest.get("quality_gate")
        texture_inputs = texture_manifest.get("inputs")
        texture_display = texture_manifest.get("display_contract")
        texture_manifest_outputs = texture_manifest.get("outputs")
        if not isinstance(texture_gate, dict):
            raise PostsealEvaluationError(
                f"{scan_id} published texture gate is missing"
            )
        recomputed_manifest_gate = _recompute_texture_gate(
            texture_gate, label=f"{scan_id} published manifest"
        )
        if (
            texture_manifest_snapshot.sha256 != manifest_descriptor["sha256"]
            or texture_manifest.get("record_sha256")
            != manifest_descriptor["record_sha256"]
            or recomputed_manifest_gate["passed"] is not True
            or texture_gate != texture_evidence.get("quality_gate")
            or texture_gate.get("policy") != texture_evidence["policy"]
            or texture_gate.get("anti_gaming_policy")
            != texture_evidence["anti_gaming_policy"]
            or texture_gate.get("verdict") != recomputed_manifest_gate["verdict"]
            or texture_gate.get("exit_code") != recomputed_manifest_gate["exit_code"]
            or not isinstance(texture_display, dict)
            or texture_display.get("image_interpolation") != "nearest"
            or texture_display.get("fMRI_registration_or_resampling") is not False
            or not isinstance(texture_manifest_outputs, dict)
            or texture_manifest_outputs.get("texture_audit_png")
            != {
                "path": str(logical_root / str(png_descriptor["relative_path"])),
                "sha256": png_descriptor["sha256"],
                "size_bytes": png_descriptor["size_bytes"],
            }
            or not isinstance(texture_inputs, dict)
            or texture_inputs.get("real") != record.get("target")
            or texture_inputs.get("predicted") != record.get("prediction")
            or texture_inputs.get("mask")
            != {
                key: target_validity_descriptor[key]
                for key in ("path", "sha256", "size_bytes")
            }
            or texture_inputs.get("roi_labels") != record.get("structural_labels")
        ):
            raise PostsealEvaluationError(
                f"{scan_id} published texture manifest binding differs"
            )
    data_quality_binding = evaluation.get("data_quality_summary")
    if not isinstance(data_quality_binding, dict) or set(data_quality_binding) != {
        "relative_path",
        "sha256",
        "size_bytes",
        "record_sha256",
    }:
        raise PostsealEvaluationError("data-quality summary binding differs")
    if data_quality_binding.get("relative_path") != "data_quality_summary.json":
        raise PostsealEvaluationError("data-quality summary path differs")
    data_quality, data_quality_snapshot = _signed_json_snapshot(
        root / "data_quality_summary.json",
        label="data-quality summary",
        expected_sha256=str(data_quality_binding.get("sha256", "")),
        expected_size=data_quality_binding.get("size_bytes"),
        expected_schema_key="schema",
        expected_schema=DATA_QUALITY_SCHEMA,
    )
    per_scan_rows = data_quality.get("per_scan_rows")
    cohort_texture_gate = data_quality.get("texture_release_gate")
    _validate_distributional_summary_mode(
        data_quality, available=distributional_available
    )
    if (
        data_quality_snapshot.sha256 != data_quality_binding.get("sha256")
        or data_quality.get("record_sha256")
        != data_quality_binding.get("record_sha256")
        or evaluation.get("fitness_for_use") != data_quality.get("fitness_for_use")
        or data_quality.get("distributional_metrics") != distributional
        or _validate_distributional_evidence(
            data_quality.get("distributional_metrics"),
            expected_authority_sha256=distributional_authority_pin,
            artifact_root=root,
        )
        is not distributional_available
        or not isinstance(per_scan_rows, list)
        or [
            row.get("scan_id") if isinstance(row, dict) else None
            for row in per_scan_rows
        ]
        != list(EXPECTED_SCAN_IDS)
        or not isinstance(cohort_texture_gate, dict)
        or evaluation.get("texture_release_gate") != cohort_texture_gate
        or cohort_texture_gate.get("schema") != TEXTURE_GATE_SCHEMA
        or cohort_texture_gate.get("verdict") != "pass"
        or type(cohort_texture_gate.get("exit_code")) is not int
        or cohort_texture_gate.get("exit_code") != 0
        or cohort_texture_gate.get("passing_scan_count") != len(EXPECTED_SCAN_IDS)
        or cohort_texture_gate.get("failing_scan_count") != 0
        or evaluation.get("texture_release_gate_pass_count") != len(EXPECTED_SCAN_IDS)
        or evaluation.get("texture_release_gate_fail_count") != 0
        or evaluation.get("all_34_texture_retention_and_anti_gaming_gates_passed")
        is not True
    ):
        raise PostsealEvaluationError("data-quality summary cohort/binding differs")
    subject_texture_by_scan = {
        str(subject["scan_id"]): _signed_json_snapshot(
            root / str(subject["record"]["relative_path"]),
            label=f"{subject['scan_id']} subject texture cross-binding",
            expected_sha256=str(subject["record"]["sha256"]),
            expected_size=subject["record"]["size_bytes"],
            expected_schema_key="schema",
            expected_schema=SUBJECT_EVALUATION_SCHEMA,
        )[0]["texture_audit"]
        for subject in subjects
    }
    for row in per_scan_rows:
        scan_id = _safe_scan_id(row.get("scan_id"))
        if row.get("texture_release_gate") != _compact_texture_gate(
            scan_id, subject_texture_by_scan[scan_id]
        ):
            raise PostsealEvaluationError(
                f"{scan_id} cohort texture evidence differs from subject record"
            )
    expected_subject_texture_outputs = [
        {
            "scan_id": row["scan_id"],
            "verdict": row["texture_release_gate"]["verdict"],
            "exit_code": row["texture_release_gate"]["exit_code"],
            "outputs": row["texture_release_gate"]["outputs"],
        }
        for row in per_scan_rows
    ]
    if (
        cohort_texture_gate.get("subject_outputs") != expected_subject_texture_outputs
        or cohort_texture_gate.get("policy")
        != per_scan_rows[0]["texture_release_gate"]["policy"]
        or cohort_texture_gate.get("anti_gaming_policy")
        != per_scan_rows[0]["texture_release_gate"]["anti_gaming_policy"]
        or cohort_texture_gate.get("nearest_neighbor_display") is not True
        or cohort_texture_gate.get("fmri_resampled") is not False
        or cohort_texture_gate.get("evaluation_feedback_to_training_or_selection")
        is not False
    ):
        raise PostsealEvaluationError("cohort texture output/policy binding differs")
    for field, expected_relative in (
        ("cohort_chart", "cohort_quality_summary.png"),
        ("human_readable_report", "DATA_QUALITY_REPORT.md"),
    ):
        artifact = data_quality.get(field)
        if not isinstance(artifact, dict) or set(artifact) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise PostsealEvaluationError(f"data-quality {field} binding differs")
        if artifact.get("relative_path") != expected_relative:
            raise PostsealEvaluationError(f"data-quality {field} path differs")
        _snapshot_file(
            root / expected_relative,
            label=f"data-quality {field}",
            expected_sha256=str(artifact.get("sha256", "")),
            expected_size=artifact.get("size_bytes"),
        )
    return evaluation


def verify_evaluation_publication(
    output_root: str | Path,
    *,
    expected_evaluation_sha256: str,
    expected_receipt_sha256: str,
    expected_prediction_set_seal_path: str,
    expected_prediction_set_seal_sha256: str,
    expected_stage_b_completed_set_path: str,
    expected_stage_b_completed_set_sha256: str,
    expected_stage_b_verifier_source_path: str,
    expected_stage_b_verifier_source_sha256: str,
    expected_stage_b_verifier_dependency_paths: Sequence[str],
    expected_stage_b_verifier_dependency_sha256s: Sequence[str],
    expected_execution_authority_path: str,
    expected_execution_authority_sha256: str,
    expected_slimbrain_authority_sha256: str | None = None,
) -> dict[str, Any]:
    """Reauthenticate a committed bundle after the complete external preflight."""
    if type(output_root) not in {str, _NATIVE_PATH_TYPE}:
        raise PostsealEvaluationError("evaluation output path type differs")
    if (
        type(expected_evaluation_sha256) is not str
        or type(expected_receipt_sha256) is not str
        or type(expected_prediction_set_seal_path) is not str
        or type(expected_prediction_set_seal_sha256) is not str
        or type(expected_stage_b_completed_set_path) is not str
        or type(expected_stage_b_completed_set_sha256) is not str
        or type(expected_stage_b_verifier_source_path) is not str
        or type(expected_stage_b_verifier_source_sha256) is not str
        or type(expected_execution_authority_path) is not str
        or type(expected_execution_authority_sha256) is not str
        or type(expected_stage_b_verifier_dependency_paths) not in {list, tuple}
        or type(expected_stage_b_verifier_dependency_sha256s) not in {list, tuple}
        or any(
            type(value) is not str
            for value in expected_stage_b_verifier_dependency_paths
        )
        or any(
            type(value) is not str
            for value in expected_stage_b_verifier_dependency_sha256s
        )
        or type(expected_slimbrain_authority_sha256) not in {str, type(None)}
    ):
        raise PostsealEvaluationError("publication pins must be exact built-in strings")
    evaluation_pin = _require_sha256(
        expected_evaluation_sha256, label="external evaluation SHA-256"
    )
    receipt_pin = _require_sha256(
        expected_receipt_sha256, label="external publication-receipt SHA-256"
    )
    authority_pins = _PublicationAuthorityPins(
        prediction_set_seal_path=expected_prediction_set_seal_path,
        prediction_set_seal_sha256=expected_prediction_set_seal_sha256,
        stage_b_completed_set_path=expected_stage_b_completed_set_path,
        stage_b_completed_set_sha256=expected_stage_b_completed_set_sha256,
        stage_b_verifier_source_path=expected_stage_b_verifier_source_path,
        stage_b_verifier_source_sha256=expected_stage_b_verifier_source_sha256,
        stage_b_verifier_dependency_paths=tuple(
            expected_stage_b_verifier_dependency_paths
        ),
        stage_b_verifier_dependency_sha256s=tuple(
            expected_stage_b_verifier_dependency_sha256s
        ),
        execution_authority_path=expected_execution_authority_path,
        execution_authority_sha256=expected_execution_authority_sha256,
    )
    distributional_authority_pin = (
        None
        if expected_slimbrain_authority_sha256 is None
        else _require_sha256(
            expected_slimbrain_authority_sha256,
            label="external SLIM-Brain authority SHA-256",
        )
    )
    # This import is deliberately metric/Torch-free.  The external prediction
    # seal, every prediction byte, Stage-B authority, every target byte, and all
    # 34 exact native NIfTI pairs are authenticated before the output bundle is
    # opened or any distributional/feature-model surface can be imported.
    from .postseal_execution import preflight_external_publication_authorities

    preflight = preflight_external_publication_authorities(
        prediction_set_seal_path=authority_pins.prediction_set_seal_path,
        prediction_set_seal_sha256=authority_pins.prediction_set_seal_sha256,
        stage_b_completed_set_path=authority_pins.stage_b_completed_set_path,
        stage_b_completed_set_sha256=authority_pins.stage_b_completed_set_sha256,
        stage_b_verifier_source_path=authority_pins.stage_b_verifier_source_path,
        stage_b_verifier_source_sha256=(authority_pins.stage_b_verifier_source_sha256),
        stage_b_verifier_dependency_paths=(
            authority_pins.stage_b_verifier_dependency_paths
        ),
        stage_b_verifier_dependency_sha256s=(
            authority_pins.stage_b_verifier_dependency_sha256s
        ),
        execution_authority_path=authority_pins.execution_authority_path,
        execution_authority_sha256=authority_pins.execution_authority_sha256,
        require_live_scheduler=False,
    )
    root = _canonical_path(output_root, label="evaluation output", directory=True)
    return _verify_evaluation_publication_impl(
        root,
        evaluation_pin=evaluation_pin,
        receipt_pin=receipt_pin,
        logical_root=root,
        authority_pins=authority_pins,
        preflight_pair_contracts=preflight.pair_contract_by_scan,
        preflight_execution_evidence=preflight.execution_evidence,
        distributional_authority_pin=distributional_authority_pin,
    )


def _verify_held_staged_publication(
    held: _HeldStagingRoot,
    *,
    evaluation_pin: str,
    receipt_pin: str,
    authority_pins: _PublicationAuthorityPins,
    preflight_pair_contracts: Mapping[str, Mapping[str, Any]],
    preflight_execution_evidence: Mapping[str, Any],
    distributional_authority_pin: str | None = None,
) -> dict[str, Any]:
    """Verify the staged bytes using the destination bound into the held root."""
    if type(held) is not _HeldStagingRoot or held.renamed:
        raise PostsealEvaluationError("held staging verifier authority differs")
    held.assert_source_identity()
    result = _verify_evaluation_publication_impl(
        held.source,
        evaluation_pin=evaluation_pin,
        receipt_pin=receipt_pin,
        logical_root=held.destination,
        authority_pins=authority_pins,
        preflight_pair_contracts=preflight_pair_contracts,
        preflight_execution_evidence=preflight_execution_evidence,
        distributional_authority_pin=distributional_authority_pin,
    )
    held.assert_source_identity()
    return result


def _verify_held_committed_publication(
    held: _HeldStagingRoot,
    *,
    evaluation_pin: str,
    receipt_pin: str,
    authority_pins: _PublicationAuthorityPins,
    preflight_pair_contracts: Mapping[str, Mapping[str, Any]],
    preflight_execution_evidence: Mapping[str, Any],
    distributional_authority_pin: str | None = None,
) -> dict[str, Any]:
    """Verify committed bytes while retaining the exact moved root descriptor."""
    if type(held) is not _HeldStagingRoot or not held.renamed:
        raise PostsealEvaluationError("held committed verifier authority differs")
    held.assert_destination_identity()
    result = _verify_evaluation_publication_impl(
        held.destination,
        evaluation_pin=evaluation_pin,
        receipt_pin=receipt_pin,
        logical_root=held.destination,
        authority_pins=authority_pins,
        preflight_pair_contracts=preflight_pair_contracts,
        preflight_execution_evidence=preflight_execution_evidence,
        distributional_authority_pin=distributional_authority_pin,
    )
    held.assert_destination_identity()
    return result


def run_postseal_evaluation(
    *,
    prediction_root: str | Path,
    pins: EvaluationPins,
    stage_b_completed_set: str | Path,
    stage_b_completed_set_sha256: str,
    output_root: str | Path,
    execution_authority_path: str | Path | None = None,
    execution_authority_sha256: str | None = None,
    fps: int = 8,
    slimbrain_authority: str | Path | None = None,
    slimbrain_authority_sha256: str | None = None,
    slimbrain_dependency_authority_sha256: str | None = None,
    slimbrain_python_executable_sha256: str | None = None,
    slimbrain_device: str | None = None,
) -> dict[str, Any]:
    """Run the staged production two-pass evaluator with a frozen authority."""

    # Importing this staged executor is safe: it imports no metric/model/Torch
    # surface.  Its first filesystem phase authenticates all prediction bytes;
    # it opens Stage-B targets only through the externally pinned authority.
    from .postseal_execution import run_postseal_evaluation as execute

    return execute(
        prediction_root=prediction_root,
        pins=pins,
        stage_b_completed_set=stage_b_completed_set,
        stage_b_completed_set_sha256=stage_b_completed_set_sha256,
        output_root=output_root,
        execution_authority_path=execution_authority_path,
        execution_authority_sha256=execution_authority_sha256,
        fps=fps,
        slimbrain_authority=slimbrain_authority,
        slimbrain_authority_sha256=slimbrain_authority_sha256,
        slimbrain_dependency_authority_sha256=(slimbrain_dependency_authority_sha256),
        slimbrain_python_executable_sha256=slimbrain_python_executable_sha256,
        slimbrain_device=slimbrain_device,
    )


__all__ = [
    "EXPECTED_SCAN_IDS",
    "EXPECTED_SCAN_IDS_SHA256",
    "EXPECTED_SHAPE",
    "EvaluationPins",
    "PostsealEvaluationError",
    "PostsealPublicationUncertainError",
    "canonical_sha256",
    "run_postseal_evaluation",
    "validate_exact_pair",
    "verify_evaluation_publication",
]
