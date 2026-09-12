"""Strict loader for paper-certified CONNECT-4 training artifacts.

All expensive foundation-model features are computed offline, but anatomical
and functional volumes are loaded without any online registration, resampling,
normalisation, temporal padding, or subject substitution.  Those operations
belong to the audited preprocessing stage and are proven by per-run provenance.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Collection, Dict, List, Mapping, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_LABEL_TO_CHANNEL,
    CANONICAL_ROI_MAPPING_SHA256,
    CANONICAL_ROI_SPECS,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
)
from .patch_descriptions import (
    KNOWN_FUNCTIONAL_CONNECTIVITY,
    PATCH_DESCRIPTION_SCHEMA_VERSION,
    build_patch_description,
)
from .dataset import Connect4Dataset
from .provenance import canonical_sha256, directory_file_sha256, sha256_file
from .spatial_contract import PATCH_COORDINATE_CONTRACT
from .protocol import (
    EXTERNAL_CACHE_PROTOCOL_SCHEMA,
    PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA,
    TARGET_VALIDITY_MASK_CONTRACT,
    _validated_native_target_artifact_identity,
    conditioning_identity_from_cache_sources,
)
from preprocessing.preprocess_fmri import (
    DEFAULT_HIGH_PASS_HZ,
    DEFAULT_LOW_PASS_HZ,
    DEFAULT_SMOOTHING_FWHM_MM,
    DEFAULT_SLICE_TIMING_REFERENCE,
    FMRI_INTENSITY_NORMALIZATION_CONTRACT,
    FMRI_INTENSITY_NORMALIZATION_DOMAIN,
    FMRI_INTENSITY_NORMALIZATION_METHOD,
    FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE,
    FMRI_PREPROCESSING_SCHEMA_VERSION,
    FMRIPREP_EXECUTION_SCHEMA_VERSION,
    FMRIPREP_EVIDENCE_SCHEMA_VERSION,
    MIN_SMOOTHING_GRADIENT_RETENTION,
    _validated_fmriprep_extra_arguments,
    _validated_fmriprep_version_probe,
)
from preprocessing.conform import load_common_grid_contract
from preprocessing.source_acquisition_identity import (
    SourceAcquisitionError,
    derivative_source_relative_paths,
    load_authenticated_json_artifact,
    load_fmriprep_runtime_identity,
    load_source_acquisition_identity,
    snapshot_binary_artifact,
    structural_source_projection,
)
from preprocessing.native_target_identity import (
    NATIVE_SOURCE_SHA256,
    NativeTargetError,
    validate_native_padded_target,
)
from preprocessing.native_structural_identity import (
    ARCHITECTURE_SHAPE,
    NATIVE_CERTIFICATION_STATUS,
    NATIVE_SHAPE,
    NATIVE_SPATIAL_PROFILE,
    NATIVE_STRUCTURAL_ALIGNMENT_SCHEMA,
    PADDING_AFTER,
    PADDING_BEFORE,
)
from utils.compat import strict_zip


RECOVERY_TARGET_VALIDITY_MASK_CONTRACT = (
    "connect4-noncertified-native-observed-support-derived-v1"
)


def target_validity_mask_from_fmri(
    fmri: np.ndarray,
    structural_brain_mask: np.ndarray,
    *,
    scan_id: str = "fMRI target",
) -> np.ndarray:
    """Return target-observed support without reinterpreting quiet BOLD signal.

    Native preprocessing writes unsupported/padded voxels as exact zero across
    every frame.  A voxel is therefore eligible for supervision when at least
    one stored frame is non-zero.  This deliberately does *not* threshold
    temporal variance: a finite, constant, non-zero series is observed data and
    remains valid.  Structural support is an independent conditioning/output
    authority and is only used here to reject impossible target foreground.
    """

    values = np.asarray(fmri)
    structural = np.asarray(structural_brain_mask)
    if values.ndim != 4:
        raise ValueError(f"{scan_id} must be [D,H,W,T], got {values.shape}")
    if structural.shape != values.shape[:3]:
        raise ValueError(
            f"{scan_id} structural mask {structural.shape} differs from "
            f"target grid {values.shape[:3]}"
        )
    if not np.isfinite(values).all() or not np.isfinite(structural).all():
        raise ValueError(f"{scan_id} target/mask contains NaN or infinity")
    if not np.logical_or(structural == 0, structural == 1).all():
        raise ValueError(f"{scan_id} structural brain mask must be exactly binary")
    observed = np.any(values != 0.0, axis=-1)
    structural_binary = structural.astype(bool, copy=False)
    outside = observed & ~structural_binary
    if bool(outside.any()):
        raise ValueError(
            f"{scan_id} has observed fMRI outside the structural brain mask"
        )
    if not bool(observed.any()):
        raise ValueError(f"{scan_id} target-validity mask is empty")
    return observed.astype(np.float32, copy=False)


def discover_structural_cache_scan_ids(
    root_dir: str | Path,
    precomputed_dir: str | Path,
    structural_stage_root: str | Path | None = None,
) -> List[str]:
    """Return the exact structural/cache cohort without touching fMRI targets.

    Training uses this inventory to authenticate the patient split before it
    constructs a dataset that is allowed to see any target.  In particular,
    no fMRI directory is listed, stat'ed, or opened here.
    """

    root = Path(root_dir)
    precomputed = Path(precomputed_dir)
    graphs_dir = precomputed / "graphs"
    if structural_stage_root is None:
        t1_dir = root / "T1"
        mask_dir = root / "Masks"
        t1_paths = [*t1_dir.glob("*_T1.nii.gz"), *t1_dir.glob("*_T1.nii")]
        mask_paths = [
            *mask_dir.glob("*_mask.nii.gz"),
            *mask_dir.glob("*_mask.nii"),
        ]
    else:
        stage = Path(structural_stage_root)
        t1_paths = list(stage.glob("*/*_T1.nii.gz"))
        mask_paths = list(stage.glob("*/*_mask.nii.gz"))
        for path, suffix in [
            *((path, "_T1.nii.gz") for path in t1_paths),
            *((path, "_mask.nii.gz") for path in mask_paths),
        ]:
            if path.parent.name != path.name.removesuffix(suffix):
                raise RuntimeError(
                    "native Stage-B structural artifact is outside its exact "
                    f"per-scan directory: {path}"
                )
    t1_ids = {
        path.name.removesuffix("_T1.nii.gz").removesuffix("_T1.nii")
        for path in t1_paths
    }
    mask_ids = {
        path.name.removesuffix("_mask.nii.gz").removesuffix("_mask.nii")
        for path in mask_paths
    }
    cache_ids = {
        path.name.removesuffix("_image_nodes.npy")
        for path in graphs_dir.glob("*_image_nodes.npy")
    }
    if not t1_ids or not mask_ids or not cache_ids:
        raise RuntimeError(
            "structural/cache discovery requires non-empty T1, Masks, and "
            "image-node cohorts"
        )
    if not (t1_ids == mask_ids == cache_ids):
        raise RuntimeError(
            "T1, mask, and image-node cache scan identities differ; refusing "
            "to resolve a target-access split from an incomplete cohort"
        )
    return sorted(cache_ids)


class Connect4PrecomputedDataset(Dataset):
    """Load aligned volumes and frozen node features for one fixed cohort."""

    # One canonical ordered ROI mapping is shared by raw preprocessing and the
    # certified cache loader; its exact contents are hashed into provenance.
    ROI_SPECS = CANONICAL_ROI_SPECS
    NUM_ROIS = len(ROI_SPECS)
    ID_TO_SLUG = {label_id: slug for slug, label_id in ROI_SPECS}
    ROI_LABEL_IDS = CANONICAL_ROI_LABEL_IDS
    ROI_LABEL_TO_CHANNEL = dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    ROI_MAPPING_SHA256 = CANONICAL_ROI_MAPPING_SHA256

    _PREPROCESSING_STEPS = [
        "T1w co-registration",
        "slice-timing correction",
        "rigid motion correction",
        "single-pass spatial harmonisation",
        "spatial smoothing",
        "temporal filtering",
        "TR/frame harmonisation",
        "recovery intensity normalization",
    ]
    _TEXT_COMPONENTS = [
        "roi_distribution_and_patch_center",
        "known_functional_connectivity",
        "subject_specific_normative_volume",
    ]
    _DISTRIBUTED_STATE_SCHEMA = "connect4-precomputed-rank-zero-admission-state-v1"
    _BRAINLM_SCAN_CONTEXT_SCHEMA = "connect4-brainlm-admitted-scan-context-v1"
    _BRAINLM_SCAN_CONTEXT_FIELDS = frozenset(
        {
            "schema",
            "scan_id",
            "role",
            "native_preprocessing_source_sha256",
            "prepared_t1_sha256",
            "prepared_mask_sha256",
            "padded_t1_artifact_descriptor_sha256",
            "padded_mask_artifact_descriptor_sha256",
            "cache_metadata_artifact_descriptor_sha256",
            "structural_source_identity_sha256",
            "native_alignment_authority_sha256",
            "target_artifact_identity_sha256",
            "padded_shape",
            "native_shape",
            "padding_before",
            "padding_after",
            "padded_affine_ras_mm",
            "support_tensor_sha256",
            "support_foreground_voxels",
            "record_sha256",
        }
    )
    _DISTRIBUTED_STATE_FIELDS = frozenset(
        {
            "schema",
            "root",
            "precomputed_dir",
            "fmri_dir",
            "require_paper_preprocessing",
            "recovery_profile",
            "structural_stage_root",
            "native_alignment_authority_path",
            "native_alignment_authority_sha256",
            "common_grid_contract",
            "target_shape",
            "patch_size",
            "num_frames",
            "expected_node_dims",
            "normalize_intensity",
            "scaler_dir",
            "scan_ids",
            "scan_roles",
            "target_scan_ids",
            "target_scan_roles",
            "target_artifact_identities",
            "sample_artifacts",
            "brainlm_context_records",
            "dwi_matrix",
            "record_sha256",
        }
    )
    _ADMITTED_ARTIFACT_FIELDS = frozenset(
        {
            "path",
            "sha256",
            "size_bytes",
            "device",
            "inode",
            "mode",
            "nlink",
            "mtime_ns",
            "ctime_ns",
        }
    )

    def __init__(
        self,
        root_dir: str,
        precomputed_dir: str,
        target_shape: Optional[Tuple[int, int, int]] = None,
        num_frames: int = 128,
        validate_files: bool = True,
        normalize_intensity: bool = True,
        register_fmri_to_t1w: bool = False,
        fmri_scale_factor: float = 1.0,
        dwi_matrix_path: Optional[str] = None,
        fmri_dir: Optional[str] = None,
        scaler_dir: Optional[str] = None,
        normative_csv_path: Optional[str] = None,
        cohort_manifest_path: Optional[str] = None,
        brainiac_model_path: Optional[str] = None,
        brainiac_checkpoint_sha256: Optional[str] = None,
        brainiac_source_sha256: Optional[str] = None,
        modernbert_model_name: Optional[str] = None,
        modernbert_revision: Optional[str] = None,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        image_node_dim: int = 768,
        mask_node_dim: int = 768,
        roi_node_dim: int = 619,
        load_fmri_targets: bool = True,
        target_scan_ids: Optional[Collection[str]] = None,
        external_protocol_context: Optional[Mapping] = None,
        require_paper_text_schema: bool = True,
        require_paper_preprocessing: bool = True,
        common_grid_contract_path: Optional[str] = None,
        common_grid_contract_sha256: Optional[str] = None,
        structural_stage_root: Optional[str] = None,
        native_alignment_authority_path: Optional[str] = None,
        native_alignment_authority_sha256: Optional[str] = None,
        native_selection_manifest_path: Optional[str] = None,
        native_selection_manifest_sha256: Optional[str] = None,
        native_selection_root_review_path: Optional[str] = None,
        native_selection_root_review_sha256: Optional[str] = None,
        native_completed_set_path: Optional[str] = None,
        native_completed_set_sha256: Optional[str] = None,
        native_completed_set_commit_marker_path: Optional[str] = None,
        native_completed_set_commit_marker_sha256: Optional[str] = None,
        native_reviewed_source_path: Optional[str] = None,
        native_reviewed_source_sha256: Optional[str] = None,
        native_runtime_attester_sha256: Optional[str] = None,
        native_verifier_sha256: Optional[str] = None,
        target_scan_roles: Optional[Mapping[str, str]] = None,
        sealed_prediction_mode: bool = False,
        allow_synthetic_grid_override: bool = False,
    ) -> None:
        super().__init__()
        self._require_canonical_roi_contract()
        self.root = Path(root_dir)
        self.precomputed_dir = Path(precomputed_dir)
        self.fmri_dir = Path(fmri_dir) if fmri_dir else self.root / "fMRI"
        self.require_paper_preprocessing = bool(require_paper_preprocessing)
        self.recovery_profile = not self.require_paper_preprocessing
        self.sealed_prediction_mode = bool(sealed_prediction_mode)
        self.structural_stage_root = (
            Path(structural_stage_root) if structural_stage_root is not None else None
        )
        self.native_alignment_authority_path = (
            Path(native_alignment_authority_path)
            if native_alignment_authority_path is not None
            else None
        )
        self.native_alignment_authority_sha256 = (
            str(native_alignment_authority_sha256).strip().lower()
            if native_alignment_authority_sha256 is not None
            else None
        )
        native_target_values = {
            "selection_manifest_path": native_selection_manifest_path,
            "selection_manifest_sha256": native_selection_manifest_sha256,
            "selection_root_review_path": native_selection_root_review_path,
            "selection_root_review_sha256": native_selection_root_review_sha256,
            "completed_set_path": native_completed_set_path,
            "completed_set_sha256": native_completed_set_sha256,
            "completed_set_commit_marker_path": (
                native_completed_set_commit_marker_path
            ),
            "completed_set_commit_marker_sha256": (
                native_completed_set_commit_marker_sha256
            ),
            "reviewed_source_path": native_reviewed_source_path,
            "reviewed_source_sha256": native_reviewed_source_sha256,
            "runtime_attester_sha256": native_runtime_attester_sha256,
            "native_verifier_sha256": native_verifier_sha256,
        }
        self.native_target_authority = {
            key: (
                Path(value)
                if key.endswith("_path") and value is not None
                else str(value).strip().lower()
                if value is not None
                else None
            )
            for key, value in native_target_values.items()
        }
        native_constructor_fields = {
            "structural_stage_root": self.structural_stage_root,
            "native_alignment_authority_path": self.native_alignment_authority_path,
            "native_alignment_authority_sha256": (
                self.native_alignment_authority_sha256
            ),
            **self.native_target_authority,
        }
        if self.require_paper_preprocessing and any(
            value is not None for value in native_constructor_fields.values()
        ):
            raise ValueError(
                "paper preprocessing rejects every native-recovery authority field"
            )
        if self.require_paper_preprocessing and target_scan_roles is not None:
            raise ValueError("paper preprocessing rejects native target role bindings")
        if self.recovery_profile:
            required_structural = {
                "structural_stage_root": self.structural_stage_root,
                "native_alignment_authority_path": (
                    self.native_alignment_authority_path
                ),
                "native_alignment_authority_sha256": (
                    self.native_alignment_authority_sha256
                ),
            }
            missing_structural = [
                key for key, value in required_structural.items() if value is None
            ]
            if missing_structural:
                raise ValueError(
                    "native recovery structural authority is incomplete: "
                    f"{missing_structural}"
                )
            if load_fmri_targets:
                missing_target = [
                    key
                    for key, value in self.native_target_authority.items()
                    if value is None
                ]
                if missing_target:
                    raise ValueError(
                        "native recovery target authority is incomplete: "
                        f"{missing_target}"
                    )
            elif any(
                value is not None for value in self.native_target_authority.values()
            ):
                raise ValueError(
                    "structural-only recovery loading rejects target authority fields"
                )
        self.common_grid_contract = None
        if common_grid_contract_path is not None:
            if not common_grid_contract_sha256:
                raise ValueError(
                    "production common-grid evidence requires an externally "
                    "configured SHA-256 pin"
                )
            self.common_grid_contract = load_common_grid_contract(
                common_grid_contract_path,
                expected_sha256=common_grid_contract_sha256,
            )
            contract_shape = tuple(
                int(value) for value in self.common_grid_contract["architecture_shape"]
            )
            if target_shape is not None and tuple(target_shape) != contract_shape:
                raise ValueError(
                    "configured architecture shape differs from the common-grid contract"
                )
            self.target_shape = contract_shape
        elif self.recovery_profile:
            if common_grid_contract_sha256 is not None:
                raise ValueError("native recovery cannot use a common-grid SHA-256")
            if target_shape is None or tuple(target_shape) != ARCHITECTURE_SHAPE:
                raise ValueError(
                    "native recovery requires the exact 64x80x64 Stage-B grid"
                )
            if allow_synthetic_grid_override:
                raise ValueError(
                    "native recovery is authority-bound, not a synthetic grid override"
                )
            self.target_shape = ARCHITECTURE_SHAPE
        elif allow_synthetic_grid_override:
            if target_shape is None:
                raise ValueError(
                    "synthetic grid override requires an explicit target_shape"
                )
            self.target_shape = tuple(int(value) for value in target_shape)
        else:
            raise ValueError(
                "production dataset loading requires a hash-bound common-grid contract"
            )
        self.allow_synthetic_grid_override = bool(allow_synthetic_grid_override)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.num_frames = int(num_frames)
        self.expected_node_dims = {
            "image": int(image_node_dim),
            "mask": int(mask_node_dim),
            "ROI": int(roi_node_dim),
        }
        self.load_fmri_targets = bool(load_fmri_targets)
        if self.sealed_prediction_mode and self.load_fmri_targets:
            raise ValueError(
                "sealed prediction mode is structural-only and cannot load targets"
            )
        self.external_protocol_context = (
            dict(external_protocol_context)
            if external_protocol_context is not None
            else None
        )
        self.normalize_intensity = bool(normalize_intensity)
        self.require_paper_text_schema = bool(require_paper_text_schema)
        if not validate_files:
            raise ValueError(
                "production cohort validation cannot be disabled for the "
                "certified CONNECT-4 loader"
            )
        if not self.require_paper_text_schema:
            raise ValueError(
                "Connect4PrecomputedDataset requires the reconciled patch-text schema"
            )
        if self.require_paper_preprocessing and self.common_grid_contract is None:
            if not self.allow_synthetic_grid_override:
                raise ValueError("paper preprocessing requires common-grid evidence")
        if self.load_fmri_targets and self.external_protocol_context is not None:
            raise ValueError(
                "external cache context is valid only when fMRI targets are disabled"
            )
        if (
            not self.load_fmri_targets
            and self.external_protocol_context is None
            and not self.sealed_prediction_mode
        ):
            raise ValueError(
                "structural-only loading requires checkpoint-derived external "
                "cache protocol context"
            )
        if self.sealed_prediction_mode and self.external_protocol_context is not None:
            raise ValueError(
                "sealed synthesis-cohort prediction cannot use external cache context"
            )
        if register_fmri_to_t1w:
            raise ValueError(
                "online fMRI registration is forbidden; supply the certified "
                "fMRIPrep/CONNECT-4 derivative"
            )
        if not np.isclose(float(fmri_scale_factor), 1.0):
            raise ValueError(
                "online fMRI spatial scaling violates the certified contract"
            )
        self.scaler_dir = Path(scaler_dir) if scaler_dir is not None else None
        if len(self.target_shape) != 3 or any(value < 1 for value in self.target_shape):
            raise ValueError("target_shape must contain three positive integers")
        if self.num_frames != 128:
            raise ValueError(
                "the paper-certified dataset requires exactly 128 fMRI frames"
            )
        if any(size % patch for size, patch in zip(self.target_shape, self.patch_size)):
            raise ValueError("target_shape must be divisible by patch_size")
        if any(value < 1 for value in self.expected_node_dims.values()):
            raise ValueError("expected node dimensions must be positive")
        if self.recovery_profile:
            path_fields = {
                "structural_stage_root": self.structural_stage_root,
                "native_alignment_authority_path": (
                    self.native_alignment_authority_path
                ),
            }
            if self.load_fmri_targets:
                path_fields.update(
                    {
                        key: value
                        for key, value in self.native_target_authority.items()
                        if key.endswith("_path")
                    }
                )
                path_fields["target_stage_root"] = self.fmri_dir
            for label, raw_path in path_fields.items():
                path = Path(raw_path)
                if not path.is_absolute() or path != Path(path.absolute()):
                    raise ValueError(f"{label} must be an absolute canonical path")
                try:
                    resolved = path.resolve(strict=True)
                except OSError as exc:
                    raise FileNotFoundError(f"{label} is missing: {path}") from exc
                if resolved != path:
                    raise ValueError(f"{label} aliases another path")
                if label in {"structural_stage_root", "target_stage_root"}:
                    if not path.is_dir():
                        raise ValueError(f"{label} must be a directory")
                elif not path.is_file():
                    raise ValueError(f"{label} must be a file")
            digest_fields = {
                "native_alignment_authority_sha256": (
                    self.native_alignment_authority_sha256
                ),
            }
            if self.load_fmri_targets:
                digest_fields.update(
                    {
                        key: value
                        for key, value in self.native_target_authority.items()
                        if key.endswith("_sha256")
                    }
                )
            for label, value in digest_fields.items():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{label} must be lowercase SHA-256")

        self.structure_to_roi_idx = dict(self.ROI_LABEL_TO_CHANNEL)
        graphs_dir = self.precomputed_dir / "graphs"
        if not graphs_dir.is_dir():
            raise RuntimeError(f"Precomputed graphs directory not found: {graphs_dir}")
        self.scan_ids = sorted(
            path.name.removesuffix("_image_nodes.npy")
            for path in graphs_dir.glob("*_image_nodes.npy")
        )
        if not self.scan_ids:
            raise RuntimeError(f"No image-node caches found in {graphs_dir}")
        if target_scan_ids is None:
            resolved_target_scan_ids = (
                frozenset(self.scan_ids) if self.load_fmri_targets else frozenset()
            )
        else:
            if isinstance(target_scan_ids, (str, bytes)):
                raise TypeError("target_scan_ids must be a collection of scan IDs")
            target_values = [str(value).strip() for value in target_scan_ids]
            if any(not value for value in target_values):
                raise ValueError("target_scan_ids cannot contain an empty scan ID")
            if len(set(target_values)) != len(target_values):
                raise ValueError("target_scan_ids cannot contain duplicate scan IDs")
            unknown_targets = sorted(set(target_values) - set(self.scan_ids))
            if unknown_targets:
                raise ValueError(
                    "target_scan_ids contains scans outside the structural cohort: "
                    f"{unknown_targets[:10]}"
                )
            resolved_target_scan_ids = frozenset(target_values)
        if not self.load_fmri_targets and resolved_target_scan_ids:
            raise ValueError(
                "target_scan_ids must be empty when load_fmri_targets is false"
            )
        self.target_scan_ids = resolved_target_scan_ids
        if self.recovery_profile and self.load_fmri_targets:
            if not isinstance(target_scan_roles, Mapping):
                raise ValueError(
                    "native recovery target loading requires explicit scan roles"
                )
            normalized_roles = {
                str(scan_id): str(role) for scan_id, role in target_scan_roles.items()
            }
            if set(normalized_roles) != set(self.target_scan_ids):
                raise ValueError(
                    "native recovery target roles must cover exactly the target allowlist"
                )
            if any(
                role not in {"train", "development-validation"}
                for role in normalized_roles.values()
            ):
                raise ValueError(
                    "native recovery targets are restricted to train/development"
                )
            self.target_scan_roles = normalized_roles
        else:
            if target_scan_roles not in (None, {}):
                raise ValueError(
                    "target_scan_roles is valid only for recovery target loading"
                )
            self.target_scan_roles = {}
        self.scan_roles = {
            scan_id: self.target_scan_roles.get(
                scan_id,
                "sealed-test" if scan_id not in self.target_scan_ids else "train",
            )
            for scan_id in self.scan_ids
        }
        if self.external_protocol_context is not None:
            context = self.external_protocol_context
            if (
                context.get("format") != EXTERNAL_CACHE_PROTOCOL_SCHEMA
                or context.get("num_scans") != len(self.scan_ids)
                or not isinstance(context.get("external_cohorts"), list)
                or not context.get("external_cohorts")
                or not isinstance(
                    context.get("synthesis_common_grid_contract_sha256"), str
                )
                or len(context["synthesis_common_grid_contract_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in context["synthesis_common_grid_contract_sha256"]
                )
                or (
                    self.common_grid_contract is None
                    and not self.allow_synthetic_grid_override
                )
                or (
                    self.common_grid_contract is not None
                    and context.get("synthesis_common_grid_contract_sha256")
                    != self.common_grid_contract.get("contract_sha256")
                )
            ):
                raise ValueError("external cache protocol context is invalid")

        self._validate_complete_cohort()
        dwi_path = (
            Path(dwi_matrix_path)
            if dwi_matrix_path is not None
            else self.root / "dwi_matrix.csv"
        )
        if not dwi_path.is_file():
            raise FileNotFoundError(f"DWI connectivity matrix not found: {dwi_path}")
        dwi_array = np.loadtxt(dwi_path, delimiter=",", dtype=np.float32)
        self.dwi_matrix = torch.from_numpy(np.asarray(dwi_array, dtype=np.float32))
        if self.dwi_matrix.shape != (self.NUM_ROIS, self.NUM_ROIS):
            raise ValueError(
                f"DWI matrix must be {self.NUM_ROIS}x{self.NUM_ROIS}, "
                f"got {tuple(self.dwi_matrix.shape)}"
            )
        if not torch.isfinite(self.dwi_matrix).all():
            raise ValueError("DWI matrix contains NaN or infinity")
        if (self.dwi_matrix < 0).any():
            raise ValueError(
                "DWI structural-connectivity coefficients must be non-negative"
            )

        if normative_csv_path is None:
            raise ValueError(
                "normative_csv_path is required to verify cached Potvin text"
            )
        if brainiac_model_path is None:
            raise ValueError(
                "brainiac_model_path is required to verify image-node provenance"
            )
        if modernbert_model_name is None:
            raise ValueError(
                "modernbert_model_name is required to verify text-node provenance"
            )
        self.brainiac_model_path = Path(brainiac_model_path)
        self.brainiac_checkpoint_sha256 = (
            str(brainiac_checkpoint_sha256 or "").strip().lower()
        )
        self.brainiac_source_sha256 = str(brainiac_source_sha256 or "").strip().lower()
        for label, digest in (
            ("BrainIAC checkpoint", self.brainiac_checkpoint_sha256),
            ("BrainIAC source", self.brainiac_source_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} SHA-256 pin is required")
        self.modernbert_model_name = str(modernbert_model_name)
        self.modernbert_revision = (
            str(modernbert_revision).strip() if modernbert_revision else None
        )
        self.source_dataset = Connect4Dataset(
            root_dir=str(self.root),
            patch_size=self.patch_size,
            target_shape=self.target_shape,
            dwi_matrix_path=str(dwi_path),
            normative_csv_path=str(normative_csv_path),
            cohort_manifest_path=cohort_manifest_path,
            scaler_dir=None,
            common_grid_contract_sha256=(
                self.common_grid_contract["contract_sha256"]
                if self.common_grid_contract is not None
                else None
            ),
            native_alignment_authority_path=(
                str(self.native_alignment_authority_path)
                if self.native_alignment_authority_path is not None
                else None
            ),
            native_alignment_authority_sha256=(self.native_alignment_authority_sha256),
            structural_stage_root=(
                str(self.structural_stage_root)
                if self.structural_stage_root is not None
                else None
            ),
        )
        if self.source_dataset.scan_ids != self.scan_ids:
            raise RuntimeError("source and precomputed scan order/cohort differ")
        if (
            tuple(self.source_dataset.ROI_SPECS) != self.ROI_SPECS
            or self.source_dataset.structure_to_roi_idx != self.ROI_LABEL_TO_CHANNEL
        ):
            raise RuntimeError(
                "structural source dataset differs from the canonical 32-ROI "
                f"mapping {self.ROI_MAPPING_SHA256}"
            )
        self._validated_cache_metadata_sha256: Dict[str, str] = {}
        for scan_id in self.scan_ids:
            self._validate_cache_provenance(scan_id)
        # Target admission happens only after structural cache/source identity
        # has been authenticated.  The loop is strictly allowlisted, so sealed
        # fMRI paths and sidecars are still never touched by training.
        self.target_artifact_identities: Dict[str, dict] = {}
        for scan_id in sorted(self.target_scan_ids):
            identity = self._validate_preprocessed_target(scan_id)
            if identity is not None:
                self.target_artifact_identities[scan_id] = identity

    @staticmethod
    def _valid_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @classmethod
    def _require_canonical_roi_contract(cls) -> None:
        """Reject class/runtime mutation of the architecture-bound ROI mapping."""

        if (
            tuple(cls.ROI_SPECS) != tuple(CANONICAL_ROI_SPECS)
            or cls.NUM_ROIS != 32
            or tuple(cls.ROI_LABEL_IDS) != tuple(CANONICAL_ROI_LABEL_IDS)
            or dict(cls.ROI_LABEL_TO_CHANNEL) != dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
            or cls.ROI_MAPPING_SHA256 != CANONICAL_ROI_MAPPING_SHA256
            or cls.ID_TO_SLUG
            != {label_id: name for name, label_id in CANONICAL_ROI_SPECS}
            or TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(
                "paper-a4-adni-fmriprep-v1"
            )
            != TARGET_VALIDITY_MASK_CONTRACT
            or TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(
                "a4-native-recovery-v1"
            )
            != RECOVERY_TARGET_VALIDITY_MASK_CONTRACT
        ):
            raise RuntimeError(
                "precomputed dataset ROI semantics differ from the canonical "
                f"32-ROI mapping {CANONICAL_ROI_MAPPING_SHA256}"
            )

    @classmethod
    def _admitted_artifact_descriptor(
        cls,
        path: Path,
        *,
        expected_sha256: str,
    ) -> dict[str, object]:
        """Bind a validated artifact to its rank-zero inode/stat snapshot."""

        if not cls._valid_sha256(expected_sha256):
            raise RuntimeError(f"admitted artifact has no valid SHA-256: {path}")
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise RuntimeError(f"admitted artifact path is not canonical: {path}")
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
        ):
            raise RuntimeError(
                f"admitted artifact must be one non-linked regular file: {path}"
            )
        return {
            "path": str(path),
            "sha256": expected_sha256,
            "size_bytes": int(metadata.st_size),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mode": int(metadata.st_mode),
            "nlink": int(metadata.st_nlink),
            "mtime_ns": int(metadata.st_mtime_ns),
            "ctime_ns": int(metadata.st_ctime_ns),
        }

    def _rank_zero_sample_artifacts(self, scan_id: str) -> dict[str, dict]:
        """Return hashes plus inode snapshots for every file read by __getitem__."""

        graphs = self.precomputed_dir / "graphs"
        hypergraphs = self.precomputed_dir / "hypergraphs"
        metadata_path = hypergraphs / f"{scan_id}_metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dataset_sources = metadata["source_fingerprint"]["dataset"]
            artifact_hashes = metadata["artifact_sha256"]
        except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{scan_id} validated cache metadata cannot form admission state"
            ) from exc
        validated_metadata_sha256 = self._validated_cache_metadata_sha256.get(scan_id)
        if (
            not self._valid_sha256(validated_metadata_sha256)
            or canonical_sha256(metadata) != validated_metadata_sha256
        ):
            raise RuntimeError(
                f"{scan_id} cache metadata changed after rank-zero validation"
            )
        paths_and_hashes = {
            "t1w": (self._t1_path(scan_id), dataset_sources.get("t1w")),
            "segmentation": (
                self._segmentation_path(scan_id),
                dataset_sources.get("segmentation"),
            ),
            "image_nodes": (
                graphs / f"{scan_id}_image_nodes.npy",
                artifact_hashes.get(f"{scan_id}_image_nodes.npy"),
            ),
            "mask_nodes": (
                graphs / f"{scan_id}_mask_nodes.npy",
                artifact_hashes.get(f"{scan_id}_mask_nodes.npy"),
            ),
            "roi_nodes": (
                graphs / f"{scan_id}_roi_nodes.npy",
                artifact_hashes.get(f"{scan_id}_roi_nodes.npy"),
            ),
            "metadata": (metadata_path, sha256_file(metadata_path)),
            "patch_distributions": (
                hypergraphs / f"{scan_id}_patch_distributions.json",
                artifact_hashes.get(f"{scan_id}_patch_distributions.json"),
            ),
        }
        if scan_id in self.target_scan_ids:
            identity = self.target_artifact_identities.get(scan_id)
            if not isinstance(identity, Mapping):
                raise RuntimeError(f"{scan_id} admitted target identity is missing")
            paths_and_hashes["target"] = (
                self._target_path(scan_id),
                identity.get("target_output_sha256"),
            )
            if not self.recovery_profile:
                paths_and_hashes["target_validity_mask"] = (
                    self._target_validity_mask_path(scan_id),
                    identity.get("functional_validity_mask_sha256"),
                )
        return {
            label: self._admitted_artifact_descriptor(
                path, expected_sha256=str(digest or "")
            )
            for label, (path, digest) in paths_and_hashes.items()
        }

    @classmethod
    def _payload_from_admitted_descriptor(
        cls,
        path: Path,
        descriptor: Mapping,
        *,
        label: str,
    ) -> bytes:
        """Read the exact rank-zero inode, never a path reopened after hashing."""

        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != cls._ADMITTED_ARTIFACT_FIELDS
            or descriptor.get("path") != str(path)
            or not path.is_absolute()
            or path != Path(os.path.abspath(path))
        ):
            raise RuntimeError(f"{label} admitted path/descriptor differs")
        try:
            path_before = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(f"{label} admitted artifact is missing") from exc
        if (
            stat.S_ISLNK(path_before.st_mode)
            or not stat.S_ISREG(path_before.st_mode)
            or not cls._stat_matches_descriptor(path_before, descriptor)
        ):
            raise RuntimeError(f"{label} admitted inode/stat snapshot changed")

        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError(f"{label} cannot enforce O_NOFOLLOW")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
        try:
            file_descriptor = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(f"{label} admitted descriptor cannot be opened") from exc
        try:
            opened_before = os.fstat(file_descriptor)
            if not stat.S_ISREG(
                opened_before.st_mode
            ) or not cls._stat_matches_descriptor(opened_before, descriptor):
                raise RuntimeError(
                    f"{label} admitted opened inode/stat snapshot changed"
                )
            expected_size = int(descriptor["size_bytes"])
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(file_descriptor, min(1024 * 1024, expected_size + 1))
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise RuntimeError(f"{label} admitted size changed")
                digest.update(block)
                chunks.append(block)
            opened_after = os.fstat(file_descriptor)
        finally:
            os.close(file_descriptor)
        try:
            path_after = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(f"{label} admitted path changed after read") from exc
        if (
            total != int(descriptor["size_bytes"])
            or not cls._stat_matches_descriptor(opened_after, descriptor)
            or not cls._stat_matches_descriptor(path_after, descriptor)
        ):
            raise RuntimeError(f"{label} admitted inode/stat snapshot changed")
        if digest.hexdigest() != descriptor.get("sha256"):
            raise RuntimeError(f"{label} admitted hash/stable snapshot changed")
        return b"".join(chunks)

    @staticmethod
    def _brainlm_support_sha256(support: np.ndarray | torch.Tensor) -> str:
        if torch.is_tensor(support):
            values = (
                (support.detach().to(device="cpu") > 0.5)
                .to(torch.uint8)
                .contiguous()
                .numpy()
            )
        else:
            values = np.ascontiguousarray(np.asarray(support) > 0, dtype=np.uint8)
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    @classmethod
    def _validate_structural_source_identity(
        cls,
        value: object,
        *,
        scan_id: str,
    ) -> dict:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{scan_id} structural source identity is missing")
        identity = dict(value)
        unsigned = dict(identity)
        fingerprint = unsigned.pop("fingerprint_sha256", None)
        native_outputs = identity.get("native_outputs")
        padded_affine = np.asarray(identity.get("padded_affine"), dtype=np.float64)
        native_affine = np.asarray(identity.get("native_affine"), dtype=np.float64)
        if (
            identity.get("schema") != NATIVE_STRUCTURAL_ALIGNMENT_SCHEMA
            or identity.get("scan_id") != scan_id
            or identity.get("paper_certified") is not False
            or identity.get("certification_status") != NATIVE_CERTIFICATION_STATUS
            or identity.get("spatial_profile") != NATIVE_SPATIAL_PROFILE
            or fingerprint != canonical_sha256(unsigned)
            or identity.get("native_shape") != list(NATIVE_SHAPE)
            or identity.get("architecture_shape") != list(ARCHITECTURE_SHAPE)
            or identity.get("padding_before") != list(PADDING_BEFORE)
            or identity.get("padding_after") != list(PADDING_AFTER)
            or identity.get("padding_mode") != "constant-zero-no-interpolation"
            or identity.get("interpolation_after_native_preprocessing") is not False
            or padded_affine.shape != (4, 4)
            or native_affine.shape != (4, 4)
            or not np.isfinite(padded_affine).all()
            or not np.isfinite(native_affine).all()
            or not np.allclose(
                padded_affine[:3, :3], native_affine[:3, :3], rtol=0.0, atol=1e-6
            )
            or not np.allclose(
                padded_affine[:3, 3],
                native_affine[:3, 3]
                - native_affine[:3, :3] @ np.asarray(PADDING_BEFORE),
                rtol=0.0,
                atol=1e-6,
            )
            or not isinstance(native_outputs, Mapping)
            or set(native_outputs) != {"t1w", "segmentation"}
        ):
            raise RuntimeError(f"{scan_id} structural source identity differs")
        for name in ("t1w", "segmentation"):
            artifact = native_outputs[name]
            if (
                not isinstance(artifact, Mapping)
                or set(artifact) != {"path", "sha256", "size_bytes"}
                or not isinstance(artifact.get("path"), str)
                or not Path(artifact["path"]).is_absolute()
                or not cls._valid_sha256(artifact.get("sha256"))
                or isinstance(artifact.get("size_bytes"), bool)
                or not isinstance(artifact.get("size_bytes"), int)
                or artifact["size_bytes"] < 1
            ):
                raise RuntimeError(
                    f"{scan_id} prepared structural artifact identity differs"
                )
        return identity

    def _rank_zero_brainlm_context_record(
        self,
        scan_id: str,
        *,
        role: str,
        artifacts: Mapping[str, Mapping],
    ) -> dict[str, object]:
        """Bind BrainLM context to admitted structure without opening fMRI."""

        if not self.recovery_profile or role not in {
            "train",
            "development-validation",
        }:
            raise RuntimeError("BrainLM context is restricted to recovery train/dev")
        target_identity = _validated_native_target_artifact_identity(
            self.target_artifact_identities.get(scan_id), scan_id=scan_id
        )
        if (
            target_identity["role"] != role
            or target_identity["reviewed_native_source_sha256"] != NATIVE_SOURCE_SHA256
            or target_identity["structural_alignment_authority_sha256"]
            != self.native_alignment_authority_sha256
        ):
            raise RuntimeError(f"{scan_id} BrainLM target authority binding differs")

        metadata_path = (
            self.precomputed_dir / "hypergraphs" / f"{scan_id}_metadata.json"
        )
        try:
            metadata = json.loads(
                self._payload_from_admitted_descriptor(
                    metadata_path,
                    artifacts["metadata"],
                    label=f"{scan_id} BrainLM metadata",
                ).decode("utf-8")
            )
            structural = self._validate_structural_source_identity(
                metadata["source_fingerprint"]["dataset"]["structural_source_identity"],
                scan_id=scan_id,
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{scan_id} BrainLM structural cache binding is invalid"
            ) from exc

        t1_path = Path(str(artifacts["t1w"]["path"]))
        mask_path = Path(str(artifacts["segmentation"]["path"]))
        t1_image = self._nifti_from_admitted_payload(
            self._payload_from_admitted_descriptor(
                t1_path, artifacts["t1w"], label=f"{scan_id} BrainLM padded T1"
            ),
            t1_path,
        )
        mask_image = self._nifti_from_admitted_payload(
            self._payload_from_admitted_descriptor(
                mask_path,
                artifacts["segmentation"],
                label=f"{scan_id} BrainLM padded mask",
            ),
            mask_path,
        )
        padded_affine = np.asarray(structural["padded_affine"], dtype=np.float64)
        t1_values = np.asarray(t1_image.dataobj, dtype=np.float32)
        mask_values = np.asarray(mask_image.dataobj, dtype=np.float32)
        rounded_mask = np.rint(mask_values)
        if (
            tuple(t1_image.shape) != ARCHITECTURE_SHAPE
            or tuple(mask_image.shape) != ARCHITECTURE_SHAPE
            or not np.allclose(t1_image.affine, padded_affine, rtol=0.0, atol=1e-6)
            or not np.allclose(mask_image.affine, padded_affine, rtol=0.0, atol=1e-6)
            or not np.isfinite(t1_values).all()
            or not np.isfinite(mask_values).all()
            or not np.allclose(mask_values, rounded_mask, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError(f"{scan_id} BrainLM padded structural grid differs")
        native_crop = tuple(
            slice(before, before + size)
            for before, size in strict_zip(PADDING_BEFORE, NATIVE_SHAPE)
        )
        outside = np.ones(ARCHITECTURE_SHAPE, dtype=bool)
        outside[native_crop] = False
        if np.any(rounded_mask[outside] != 0):
            raise RuntimeError(f"{scan_id} BrainLM structural padding is nonzero")
        support = rounded_mask > 0
        support_count = int(np.count_nonzero(support))
        if support_count < 1:
            raise RuntimeError(f"{scan_id} BrainLM structural support is empty")
        native_outputs = structural["native_outputs"]
        record: dict[str, object] = {
            "schema": self._BRAINLM_SCAN_CONTEXT_SCHEMA,
            "scan_id": scan_id,
            "role": role,
            "native_preprocessing_source_sha256": NATIVE_SOURCE_SHA256,
            "prepared_t1_sha256": native_outputs["t1w"]["sha256"],
            "prepared_mask_sha256": native_outputs["segmentation"]["sha256"],
            "padded_t1_artifact_descriptor_sha256": canonical_sha256(artifacts["t1w"]),
            "padded_mask_artifact_descriptor_sha256": canonical_sha256(
                artifacts["segmentation"]
            ),
            "cache_metadata_artifact_descriptor_sha256": canonical_sha256(
                artifacts["metadata"]
            ),
            "structural_source_identity_sha256": structural["fingerprint_sha256"],
            "native_alignment_authority_sha256": (
                self.native_alignment_authority_sha256
            ),
            "target_artifact_identity_sha256": target_identity["fingerprint_sha256"],
            "padded_shape": list(ARCHITECTURE_SHAPE),
            "native_shape": list(NATIVE_SHAPE),
            "padding_before": list(PADDING_BEFORE),
            "padding_after": list(PADDING_AFTER),
            "padded_affine_ras_mm": padded_affine.tolist(),
            "support_tensor_sha256": self._brainlm_support_sha256(support),
            "support_foreground_voxels": support_count,
        }
        record["record_sha256"] = canonical_sha256(record)
        return record

    def build_rank_zero_admission_state(
        self,
        *,
        target_scan_roles: Optional[Mapping[str, str]] = None,
    ) -> dict[str, object]:
        """Freeze the fully validated dataset into a target-array-free state."""

        self._require_canonical_roi_contract()
        if self.structure_to_roi_idx != self.ROI_LABEL_TO_CHANNEL:
            raise RuntimeError("rank-zero dataset ROI mapping was mutated")

        admitted_roles = dict(
            self.target_scan_roles if target_scan_roles is None else target_scan_roles
        )
        if set(admitted_roles) != set(self.target_scan_ids) or any(
            role not in {"train", "development-validation"}
            for role in admitted_roles.values()
        ):
            raise RuntimeError(
                "rank-zero admission roles must cover exactly the target allowlist"
            )
        sample_artifacts = {
            scan_id: self._rank_zero_sample_artifacts(scan_id)
            for scan_id in self.scan_ids
        }
        scan_roles = {
            scan_id: admitted_roles.get(scan_id, "sealed-test")
            for scan_id in self.scan_ids
        }
        if (
            set(scan_roles) != set(self.scan_ids)
            or any(
                role not in {"train", "development-validation", "sealed-test"}
                for role in scan_roles.values()
            )
            or {
                scan_id for scan_id, role in scan_roles.items() if role != "sealed-test"
            }
            != set(self.target_scan_ids)
        ):
            raise RuntimeError("rank-zero admission all-scan roles differ")
        brainlm_context_records = (
            {
                scan_id: self._rank_zero_brainlm_context_record(
                    scan_id,
                    role=admitted_roles[scan_id],
                    artifacts=sample_artifacts[scan_id],
                )
                for scan_id in sorted(self.target_scan_ids)
            }
            if self.recovery_profile
            else {}
        )
        state: dict[str, object] = {
            "schema": self._DISTRIBUTED_STATE_SCHEMA,
            "root": str(self.root),
            "precomputed_dir": str(self.precomputed_dir),
            "fmri_dir": str(self.fmri_dir),
            "require_paper_preprocessing": self.require_paper_preprocessing,
            "recovery_profile": self.recovery_profile,
            "structural_stage_root": (
                str(self.structural_stage_root)
                if self.structural_stage_root is not None
                else None
            ),
            "native_alignment_authority_path": (
                str(self.native_alignment_authority_path)
                if self.native_alignment_authority_path is not None
                else None
            ),
            "native_alignment_authority_sha256": (
                self.native_alignment_authority_sha256
            ),
            "common_grid_contract": self.common_grid_contract,
            "target_shape": list(self.target_shape),
            "patch_size": list(self.patch_size),
            "num_frames": self.num_frames,
            "expected_node_dims": dict(self.expected_node_dims),
            "normalize_intensity": self.normalize_intensity,
            "scaler_dir": str(self.scaler_dir) if self.scaler_dir is not None else None,
            "scan_ids": list(self.scan_ids),
            "scan_roles": dict(sorted(scan_roles.items())),
            "target_scan_ids": sorted(self.target_scan_ids),
            "target_scan_roles": dict(sorted(admitted_roles.items())),
            "target_artifact_identities": {
                scan_id: self.target_artifact_identities[scan_id]
                for scan_id in sorted(self.target_artifact_identities)
            },
            "sample_artifacts": sample_artifacts,
            "brainlm_context_records": brainlm_context_records,
            "dwi_matrix": self.dwi_matrix.cpu().tolist(),
        }
        state["record_sha256"] = canonical_sha256(state)
        validated = self._validate_rank_zero_admission_state(state)
        self._admitted_sample_artifacts = validated["sample_artifacts"]
        self._brainlm_context_records = validated["brainlm_context_records"]
        self._rank_zero_admission_state_sha256 = validated["record_sha256"]
        self._rank_zero_admission_payload_sha256 = canonical_sha256(validated)
        return validated

    @classmethod
    def _validate_rank_zero_admission_state(cls, value: Mapping) -> dict[str, object]:
        cls._require_canonical_roi_contract()
        if (
            not isinstance(value, Mapping)
            or set(value) != cls._DISTRIBUTED_STATE_FIELDS
        ):
            raise RuntimeError("rank-zero dataset-admission state fields differ")
        state = dict(value)
        unsigned = dict(state)
        recorded = unsigned.pop("record_sha256", None)
        if (
            state.get("schema") != cls._DISTRIBUTED_STATE_SCHEMA
            or not cls._valid_sha256(recorded)
            or canonical_sha256(unsigned) != recorded
        ):
            raise RuntimeError("rank-zero dataset-admission state digest differs")
        require_paper = state.get("require_paper_preprocessing")
        recovery = state.get("recovery_profile")
        normalize = state.get("normalize_intensity")
        if (
            not isinstance(require_paper, bool)
            or not isinstance(recovery, bool)
            or recovery is require_paper
            or not isinstance(normalize, bool)
        ):
            raise RuntimeError("rank-zero dataset-admission profile booleans differ")
        for field in ("root", "precomputed_dir", "fmri_dir"):
            raw_path = state.get(field)
            if (
                not isinstance(raw_path, str)
                or not Path(raw_path).is_absolute()
                or Path(raw_path) != Path(os.path.abspath(raw_path))
            ):
                raise RuntimeError(f"rank-zero dataset-admission {field} path differs")
        structural_root = state.get("structural_stage_root")
        native_authority_path = state.get("native_alignment_authority_path")
        native_authority_sha = state.get("native_alignment_authority_sha256")
        common_grid = state.get("common_grid_contract")
        if recovery:
            for field, raw_path in (
                ("structural_stage_root", structural_root),
                ("native_alignment_authority_path", native_authority_path),
            ):
                if (
                    not isinstance(raw_path, str)
                    or not Path(raw_path).is_absolute()
                    or Path(raw_path) != Path(os.path.abspath(raw_path))
                ):
                    raise RuntimeError(f"rank-zero recovery {field} path differs")
            if not cls._valid_sha256(native_authority_sha) or common_grid is not None:
                raise RuntimeError("rank-zero recovery spatial authority differs")
        elif (
            structural_root is not None
            or native_authority_path is not None
            or native_authority_sha is not None
            or not isinstance(common_grid, Mapping)
        ):
            raise RuntimeError("rank-zero paper spatial authority differs")
        scaler_dir = state.get("scaler_dir")
        if scaler_dir is not None and (
            not isinstance(scaler_dir, str)
            or not Path(scaler_dir).is_absolute()
            or Path(scaler_dir) != Path(os.path.abspath(scaler_dir))
        ):
            raise RuntimeError("rank-zero scaler directory path differs")
        scan_ids = state.get("scan_ids")
        scan_roles = state.get("scan_roles")
        targets = state.get("target_scan_ids")
        roles = state.get("target_scan_roles")
        identities = state.get("target_artifact_identities")
        sample_artifacts = state.get("sample_artifacts")
        brainlm_records = state.get("brainlm_context_records")
        if (
            not isinstance(scan_ids, list)
            or not scan_ids
            or scan_ids != sorted(set(scan_ids))
            or not isinstance(targets, list)
            or targets != sorted(set(targets))
            or not set(targets).issubset(scan_ids)
            or not isinstance(scan_roles, Mapping)
            or set(scan_roles) != set(scan_ids)
            or any(
                role not in {"train", "development-validation", "sealed-test"}
                for role in scan_roles.values()
            )
            or not isinstance(roles, Mapping)
            or set(roles) != set(targets)
            or any(
                role not in {"train", "development-validation"}
                for role in roles.values()
            )
            or any(scan_roles[scan_id] != roles[scan_id] for scan_id in targets)
            or any(
                scan_roles[scan_id] != "sealed-test"
                for scan_id in set(scan_ids) - set(targets)
            )
            or not isinstance(identities, Mapping)
            or set(identities) != set(targets)
            or not isinstance(sample_artifacts, Mapping)
            or set(sample_artifacts) != set(scan_ids)
            or not isinstance(brainlm_records, Mapping)
            or set(brainlm_records)
            != (set(targets) if state.get("recovery_profile") is True else set())
        ):
            raise RuntimeError("rank-zero dataset-admission cohort/target map differs")
        required_base = {
            "t1w",
            "segmentation",
            "image_nodes",
            "mask_nodes",
            "roi_nodes",
            "metadata",
            "patch_distributions",
        }
        for scan_id in scan_ids:
            artifacts = sample_artifacts[scan_id]
            expected_keys = required_base | (
                ({"target"} | ({"target_validity_mask"} if not recovery else set()))
                if scan_id in targets
                else set()
            )
            if not isinstance(artifacts, Mapping) or set(artifacts) != expected_keys:
                raise RuntimeError(
                    f"{scan_id} rank-zero admitted sample inventory differs"
                )
            for label, descriptor in artifacts.items():
                if (
                    not isinstance(descriptor, Mapping)
                    or set(descriptor) != cls._ADMITTED_ARTIFACT_FIELDS
                    or not cls._valid_sha256(descriptor.get("sha256"))
                    or not isinstance(descriptor.get("path"), str)
                    or not Path(descriptor["path"]).is_absolute()
                    or Path(descriptor["path"])
                    != Path(os.path.abspath(descriptor["path"]))
                    or any(
                        isinstance(descriptor.get(field), bool)
                        or not isinstance(descriptor.get(field), int)
                        or descriptor[field] < minimum
                        for field, minimum in (
                            ("size_bytes", 1),
                            ("device", 0),
                            ("inode", 1),
                            ("mode", 1),
                            ("nlink", 1),
                            ("mtime_ns", 0),
                            ("ctime_ns", 0),
                        )
                    )
                    or not stat.S_ISREG(int(descriptor.get("mode", 0)))
                    or descriptor.get("nlink") != 1
                ):
                    raise RuntimeError(f"{scan_id} admitted {label} descriptor differs")
            if scan_id in targets:
                identity = identities[scan_id]
                try:
                    validated_identity = (
                        _validated_native_target_artifact_identity(
                            identity, scan_id=scan_id
                        )
                        if recovery
                        else dict(identity)
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{scan_id} admitted target binding differs"
                    ) from exc
                if (
                    not isinstance(identity, Mapping)
                    or validated_identity.get("target_output_sha256")
                    != artifacts["target"]["sha256"]
                    or (
                        not recovery
                        and (
                            set(validated_identity)
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
                            or validated_identity.get("format")
                            != PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA
                            or validated_identity.get("scan_id") != scan_id
                            or validated_identity.get("functional_validity_mask_path")
                            != artifacts["target_validity_mask"]["path"]
                            or validated_identity.get("functional_validity_mask_sha256")
                            != artifacts["target_validity_mask"]["sha256"]
                            or validated_identity.get(
                                "functional_validity_mask_size_bytes"
                            )
                            != artifacts["target_validity_mask"]["size_bytes"]
                            or validated_identity.get(
                                "functional_validity_mask_contract"
                            )
                            != TARGET_VALIDITY_MASK_CONTRACT
                            or validated_identity.get("fingerprint_sha256")
                            != canonical_sha256(
                                {
                                    key: value
                                    for key, value in validated_identity.items()
                                    if key != "fingerprint_sha256"
                                }
                            )
                        )
                    )
                    or (recovery and roles[scan_id] != validated_identity.get("role"))
                    or (
                        recovery
                        and validated_identity.get(
                            "structural_alignment_authority_sha256"
                        )
                        != native_authority_sha
                    )
                    or (
                        recovery
                        and validated_identity.get("reviewed_native_source_sha256")
                        != NATIVE_SOURCE_SHA256
                    )
                ):
                    raise RuntimeError(f"{scan_id} admitted target binding differs")
                if recovery:
                    context = brainlm_records[scan_id]
                    context_unsigned = (
                        dict(context) if isinstance(context, Mapping) else {}
                    )
                    context_digest = context_unsigned.pop("record_sha256", None)
                    affine = np.asarray(
                        context.get("padded_affine_ras_mm")
                        if isinstance(context, Mapping)
                        else None,
                        dtype=np.float64,
                    )
                    support_count = (
                        context.get("support_foreground_voxels")
                        if isinstance(context, Mapping)
                        else None
                    )
                    if (
                        not isinstance(context, Mapping)
                        or set(context) != cls._BRAINLM_SCAN_CONTEXT_FIELDS
                        or context.get("schema") != cls._BRAINLM_SCAN_CONTEXT_SCHEMA
                        or context.get("scan_id") != scan_id
                        or context.get("role") != roles[scan_id]
                        or context.get("native_preprocessing_source_sha256")
                        != NATIVE_SOURCE_SHA256
                        or any(
                            not cls._valid_sha256(context.get(field))
                            for field in (
                                "prepared_t1_sha256",
                                "prepared_mask_sha256",
                                "padded_t1_artifact_descriptor_sha256",
                                "padded_mask_artifact_descriptor_sha256",
                                "cache_metadata_artifact_descriptor_sha256",
                                "structural_source_identity_sha256",
                                "native_alignment_authority_sha256",
                                "target_artifact_identity_sha256",
                                "support_tensor_sha256",
                                "record_sha256",
                            )
                        )
                        or context_digest != canonical_sha256(context_unsigned)
                        or context.get("padded_t1_artifact_descriptor_sha256")
                        != canonical_sha256(artifacts["t1w"])
                        or context.get("padded_mask_artifact_descriptor_sha256")
                        != canonical_sha256(artifacts["segmentation"])
                        or context.get("cache_metadata_artifact_descriptor_sha256")
                        != canonical_sha256(artifacts["metadata"])
                        or context.get("native_alignment_authority_sha256")
                        != native_authority_sha
                        or context.get("target_artifact_identity_sha256")
                        != validated_identity.get("fingerprint_sha256")
                        or context.get("padded_shape") != list(ARCHITECTURE_SHAPE)
                        or context.get("native_shape") != list(NATIVE_SHAPE)
                        or context.get("padding_before") != list(PADDING_BEFORE)
                        or context.get("padding_after") != list(PADDING_AFTER)
                        or affine.shape != (4, 4)
                        or not np.isfinite(affine).all()
                        or isinstance(support_count, bool)
                        or not isinstance(support_count, int)
                        or support_count < 1
                        or support_count > int(np.prod(ARCHITECTURE_SHAPE))
                    ):
                        raise RuntimeError(
                            f"{scan_id} admitted BrainLM context differs"
                        )
        dwi = np.asarray(state.get("dwi_matrix"), dtype=np.float32)
        if (
            dwi.shape != (cls.NUM_ROIS, cls.NUM_ROIS)
            or not np.isfinite(dwi).all()
            or np.any(dwi < 0)
        ):
            raise RuntimeError("rank-zero admitted DWI matrix differs")
        for name in ("target_shape", "patch_size"):
            item = state.get(name)
            if (
                not isinstance(item, list)
                or len(item) != 3
                or any(
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or number < 1
                    for number in item
                )
            ):
                raise RuntimeError(f"rank-zero admitted {name} differs")
        expected_dims = state.get("expected_node_dims")
        target_shape = state["target_shape"]
        patch_size = state["patch_size"]
        if (
            not isinstance(expected_dims, Mapping)
            or set(expected_dims) != {"image", "mask", "ROI"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in expected_dims.values()
            )
            or any(
                target % patch
                for target, patch in strict_zip(target_shape, patch_size)
            )
            or (recovery and target_shape != list(ARCHITECTURE_SHAPE))
        ):
            raise RuntimeError("rank-zero admitted grid/node dimensions differ")
        if state.get("num_frames") != 128:
            raise RuntimeError("rank-zero admitted frame count differs")
        return state

    @classmethod
    def from_rank_zero_admission_state(
        cls, value: Mapping
    ) -> "Connect4PrecomputedDataset":
        """Hydrate a worker rank only from the authenticated rank-zero state."""

        state = cls._validate_rank_zero_admission_state(value)
        self = cls.__new__(cls)
        Dataset.__init__(self)
        self.root = Path(state["root"])
        self.precomputed_dir = Path(state["precomputed_dir"])
        self.fmri_dir = Path(state["fmri_dir"])
        self.require_paper_preprocessing = bool(state["require_paper_preprocessing"])
        self.recovery_profile = bool(state["recovery_profile"])
        self.sealed_prediction_mode = False
        self.structural_stage_root = (
            Path(state["structural_stage_root"])
            if state["structural_stage_root"] is not None
            else None
        )
        self.native_alignment_authority_path = (
            Path(state["native_alignment_authority_path"])
            if state["native_alignment_authority_path"] is not None
            else None
        )
        self.native_alignment_authority_sha256 = state[
            "native_alignment_authority_sha256"
        ]
        self.native_target_authority = {}
        self.common_grid_contract = state["common_grid_contract"]
        self.target_shape = tuple(state["target_shape"])
        self.patch_size = tuple(state["patch_size"])
        self.num_frames = int(state["num_frames"])
        self.expected_node_dims = dict(state["expected_node_dims"])
        self.load_fmri_targets = True
        self.external_protocol_context = None
        self.normalize_intensity = bool(state["normalize_intensity"])
        self.require_paper_text_schema = True
        self.allow_synthetic_grid_override = False
        self.scaler_dir = Path(state["scaler_dir"]) if state["scaler_dir"] else None
        self.structure_to_roi_idx = dict(self.ROI_LABEL_TO_CHANNEL)
        self.scan_ids = list(state["scan_ids"])
        self.scan_roles = dict(state["scan_roles"])
        self.target_scan_ids = frozenset(state["target_scan_ids"])
        self.target_scan_roles = dict(state["target_scan_roles"])
        self.target_artifact_identities = dict(state["target_artifact_identities"])
        self._admitted_sample_artifacts = dict(state["sample_artifacts"])
        self._brainlm_context_records = dict(state["brainlm_context_records"])
        self._rank_zero_admission_state_sha256 = state["record_sha256"]
        self._rank_zero_admission_payload_sha256 = canonical_sha256(state)
        self.dwi_matrix = torch.from_numpy(
            np.asarray(state["dwi_matrix"], dtype=np.float32)
        )
        # Deliberately absent: constructing Connect4Dataset would repeat all
        # structural/radiomics admission on every rank. __getitem__ consumes only
        # the closed, hash-bound precomputed state above.
        self.source_dataset = None
        return self

    @staticmethod
    def _stat_matches_descriptor(metadata: os.stat_result, descriptor: Mapping) -> bool:
        return all(
            int(observed) == int(descriptor[name])
            for name, observed in (
                ("device", metadata.st_dev),
                ("inode", metadata.st_ino),
                ("mode", metadata.st_mode),
                ("nlink", metadata.st_nlink),
                ("size_bytes", metadata.st_size),
                ("mtime_ns", metadata.st_mtime_ns),
                ("ctime_ns", metadata.st_ctime_ns),
            )
        )

    def _admitted_payload(self, scan_id: str, label: str, path: Path) -> bytes:
        artifacts = getattr(self, "_admitted_sample_artifacts", None)
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("rank-zero admitted sample map is missing")
        descriptor = artifacts.get(scan_id, {}).get(label)
        if not isinstance(descriptor, Mapping) or descriptor.get("path") != str(path):
            raise RuntimeError(f"{scan_id} admitted {label} path differs")
        return self._payload_from_admitted_descriptor(
            path, descriptor, label=f"{scan_id} admitted {label}"
        )

    @staticmethod
    def _nifti_from_admitted_payload(payload: bytes, path: Path) -> nib.Nifti1Image:
        content = gzip.decompress(payload) if path.name.endswith(".gz") else payload
        return nib.Nifti1Image.from_bytes(content)

    def __len__(self) -> int:
        return len(self.scan_ids)

    def _t1_path(self, scan_id: str) -> Path:
        structural_root = getattr(self, "structural_stage_root", None)
        if structural_root is not None:
            return structural_root / scan_id / f"{scan_id}_T1.nii.gz"
        return self.root / "T1" / f"{scan_id}_T1.nii.gz"

    def _segmentation_path(self, scan_id: str) -> Path:
        structural_root = getattr(self, "structural_stage_root", None)
        if structural_root is not None:
            return structural_root / scan_id / f"{scan_id}_mask.nii.gz"
        return self.root / "Masks" / f"{scan_id}_mask.nii.gz"

    def _target_directory(self, scan_id: str) -> Path:
        return (
            self.fmri_dir / scan_id
            if getattr(self, "recovery_profile", False)
            else self.fmri_dir
        )

    def _target_path(self, scan_id: str) -> Path:
        return self._target_directory(scan_id) / f"{scan_id}_fMRI.nii.gz"

    def _target_sidecar_path(self, scan_id: str) -> Path:
        return self._target_directory(scan_id) / f"{scan_id}_fMRI.json"

    def _target_validity_mask_path(self, scan_id: str) -> Path:
        return (
            self._target_directory(scan_id)
            / f"{scan_id}_fMRI_functional_validity_mask.nii.gz"
        )

    def _load_target_image(self, scan_id: str) -> nib.Nifti1Image:
        path = self._target_path(scan_id)
        if isinstance(getattr(self, "_admitted_sample_artifacts", None), Mapping):
            payload = self._admitted_payload(scan_id, "target", path)
            return self._nifti_from_admitted_payload(payload, path)
        if not getattr(self, "recovery_profile", False):
            return nib.load(str(path))
        identity = self.target_artifact_identities.get(scan_id)
        if not isinstance(identity, Mapping):
            raise RuntimeError(f"{scan_id} has no admitted recovery-target identity")
        try:
            payload, evidence = snapshot_binary_artifact(
                path,
                expected_sha256=str(identity["target_output_sha256"]),
                label=f"{scan_id} admitted padded recovery target",
            )
            if evidence["sha256"] != identity["target_output_sha256"]:
                raise RuntimeError("target identity changed after admission")
            content = gzip.decompress(payload) if path.name.endswith(".gz") else payload
            return nib.Nifti1Image.from_bytes(content)
        except (KeyError, OSError, SourceAcquisitionError) as exc:
            raise RuntimeError(
                f"{scan_id} recovery target changed after admission"
            ) from exc

    def _load_target_validity_mask_image(self, scan_id: str) -> nib.Nifti1Image:
        """Load the rank-zero admitted certified functional-support artifact."""

        if getattr(self, "recovery_profile", False):
            raise RuntimeError(
                "non-certified native recovery has no functional-validity artifact"
            )
        path = self._target_validity_mask_path(scan_id)
        if isinstance(getattr(self, "_admitted_sample_artifacts", None), Mapping):
            payload = self._admitted_payload(scan_id, "target_validity_mask", path)
            return self._nifti_from_admitted_payload(payload, path)
        return nib.load(str(path))

    def _required_files(self, scan_id: str) -> List[Path]:
        graphs = self.precomputed_dir / "graphs"
        hypergraphs = self.precomputed_dir / "hypergraphs"
        image_embeddings = self.precomputed_dir / "image_emb"
        text_embeddings = self.precomputed_dir / "modernbert_emb"
        required = [
            self._t1_path(scan_id),
            self._segmentation_path(scan_id),
            graphs / f"{scan_id}_image_nodes.npy",
            graphs / f"{scan_id}_mask_nodes.npy",
            graphs / f"{scan_id}_roi_nodes.npy",
            image_embeddings / f"{scan_id}_image_patch_embeddings.npy",
            text_embeddings / f"{scan_id}_mask_patch_embeddings.npy",
            hypergraphs / f"{scan_id}_patch_distributions.json",
            hypergraphs / f"{scan_id}_patch_descriptions.json",
            hypergraphs / f"{scan_id}_hyperedge_index.npy",
            hypergraphs / f"{scan_id}_hyperedge_weights.npy",
            hypergraphs / f"{scan_id}_metadata.json",
        ]
        if scan_id in self.target_scan_ids:
            required.extend(
                [
                    self._target_path(scan_id),
                    self._target_sidecar_path(scan_id),
                ]
            )
            if self.recovery_profile:
                required.append(
                    self._target_directory(scan_id) / "padded_target_publication.json"
                )
            else:
                required.append(self._target_validity_mask_path(scan_id))
        return required

    def _validate_complete_cohort(self) -> None:
        if self.structural_stage_root is None:
            t1_dir = self.root / "T1"
            t1_paths = [*t1_dir.glob("*_T1.nii.gz"), *t1_dir.glob("*_T1.nii")]
        else:
            t1_dir = self.structural_stage_root
            t1_paths = list(t1_dir.glob("*/*_T1.nii.gz"))
        source_ids = {
            path.name.removesuffix("_T1.nii.gz").removesuffix("_T1.nii")
            for path in t1_paths
        }
        if not source_ids:
            raise RuntimeError(f"No T1w cohort files found in {t1_dir}")
        cached_ids = set(self.scan_ids)
        if cached_ids != source_ids:
            missing = sorted(source_ids - cached_ids)
            extra = sorted(cached_ids - source_ids)
            raise RuntimeError(
                "precomputed cache cohort differs from the complete T1w cohort; "
                f"missing={missing[:10]}, extra={extra[:10]}. Refusing to train "
                "on a silently reduced or substituted cohort."
            )

        incomplete: Dict[str, List[str]] = {}
        for scan_id in self.scan_ids:
            invalid = []
            for path in self._required_files(scan_id):
                try:
                    if path.stat().st_size == 0:
                        invalid.append(str(path))
                except OSError:
                    invalid.append(str(path))
            if invalid:
                incomplete[scan_id] = invalid
        if incomplete:
            preview = "; ".join(
                f"{scan}: {paths[0]}" for scan, paths in list(incomplete.items())[:5]
            )
            raise FileNotFoundError(
                f"{len(incomplete)} discovered scan(s) have incomplete artifacts "
                f"({preview}). Refusing to alter the fixed cohort by dropping them."
            )

    @staticmethod
    def _finite_float(value: object) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return result if np.isfinite(result) else float("nan")

    @staticmethod
    def _require_matching_structural_fmri_source_identity(
        structural_identity: object,
        fmri_source_identity: Mapping,
        fmri_source_evidence: Mapping,
    ) -> dict:
        """Prove structural caches and an allowlisted fMRI share one acquisition."""

        if not isinstance(structural_identity, Mapping):
            raise SourceAcquisitionError(
                "structural cache has no authenticated source-acquisition binding"
            )
        # The structural side is a target-blind v2 authority. The allowlisted
        # fMRI side may authenticate its full raw-BOLD identity here, then only
        # the T1/SynthSeg projection is joined. No full-identity artifact hash
        # is ever required by sealed structural/cache admission.
        _ = fmri_source_evidence
        observed = structural_source_projection(fmri_source_identity)
        expected = structural_source_projection(structural_identity)
        if expected != observed:
            raise SourceAcquisitionError(
                "structural cache and fMRI structural-source projections differ"
            )
        return observed

    def _validate_preprocessed_target(self, scan_id: str) -> Optional[dict]:
        if getattr(self, "recovery_profile", False):
            return self._validate_native_padded_target(scan_id)
        fmri_path = self._target_path(scan_id)
        sidecar_path = self._target_sidecar_path(scan_id)
        try:
            with sidecar_path.open() as stream:
                provenance = json.load(stream)
        except Exception as exc:
            raise RuntimeError(
                f"Missing/invalid preprocessing provenance for {scan_id}: "
                f"{sidecar_path}"
            ) from exc

        failures: List[str] = []
        if (
            provenance.get("PreprocessingSchemaVersion")
            != FMRI_PREPROCESSING_SCHEMA_VERSION
        ):
            failures.append("wrong preprocessing schema")
        if provenance.get("PaperRequiredStepsComplete") is not True:
            failures.append("PaperRequiredStepsComplete is not true")
        if provenance.get("PreprocessingBackend") != "fmriprep":
            failures.append("production target was not sourced from fMRIPrep")
        manuscript_claims = provenance.get("ManuscriptClaims")
        if manuscript_claims != {
            "Frames": 128,
            "RepetitionTimeSeconds": 3.0,
            "VoxelSizeMM": [3.0, 3.0, 3.0],
            "SpatialMatrix": None,
            "SmoothingFWHMMM": None,
            "TemporalFilterCutoffsHz": None,
            "IntensityNormalization": None,
        }:
            failures.append(
                "manuscript-claim boundary is missing or misrepresents an "
                "unreported hyperparameter"
            )
        recovery_choices = provenance.get("VersionedRecoveryChoices")
        if not isinstance(recovery_choices, Mapping):
            failures.append("versioned recovery choices are missing")
        else:
            if recovery_choices.get("SpatialSmoothingMethod") != (
                "Gaussian signal/mask division within functional-validity mask"
            ):
                failures.append("recovery smoothing method is inconsistent")
            if not np.isclose(
                self._finite_float(recovery_choices.get("SpatialSmoothingFWHMMM")),
                DEFAULT_SMOOTHING_FWHM_MM,
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append("recovery smoothing FWHM is inconsistent")
            if not np.isclose(
                self._finite_float(
                    recovery_choices.get("MinimumGradientRetentionRatioExclusive")
                ),
                MIN_SMOOTHING_GRADIENT_RETENTION,
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append("recovery gradient-retention policy is inconsistent")
            if recovery_choices.get("TemporalFilterPreservesVoxelMean") is not True:
                failures.append(
                    "recovery temporal filter does not preserve voxel means"
                )
            if not np.isclose(
                self._finite_float(recovery_choices.get("TemporalHighPassHz")),
                DEFAULT_HIGH_PASS_HZ,
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append("recovery temporal high-pass cutoff differs")
            if not np.isclose(
                self._finite_float(recovery_choices.get("TemporalLowPassHz")),
                DEFAULT_LOW_PASS_HZ,
                rtol=0.0,
                atol=1e-12,
            ):
                failures.append("recovery temporal low-pass cutoff differs")
            if (
                self.normalize_intensity
                and recovery_choices.get("IntensityNormalizationContract")
                != FMRI_INTENSITY_NORMALIZATION_CONTRACT
            ):
                failures.append("recovery intensity-normalization contract differs")
            if recovery_choices.get("PostFMRIPrepTargetInterpolationCount") != 1:
                failures.append(
                    "post-fMRIPrep target data were not spatially interpolated "
                    "exactly once"
                )
            if (
                recovery_choices.get("PostFMRIPrepIntermediateT1GridResamplingApplied")
                is not False
            ):
                failures.append(
                    "post-fMRIPrep target data used a forbidden intermediate "
                    "T1-grid resampling"
                )
            if recovery_choices.get("PostFMRIPrepSpatialInterpolationOrder") != 1:
                failures.append("post-fMRIPrep spatial interpolation order differs")
            if recovery_choices.get("SpatialResamplingRoute") != (
                "fMRIPrep T1w-space BOLD grid directly to cohort-common grid"
            ):
                failures.append("post-fMRIPrep spatial resampling route differs")
        grid_evidence = provenance.get("CommonGridContract")
        if self.common_grid_contract is not None:
            expected_grid = self.common_grid_contract
            if (
                not isinstance(recovery_choices, Mapping)
                or recovery_choices.get("SpatialGrid") != expected_grid["schema"]
            ):
                failures.append("recovery grid choice differs from configured evidence")
            if not isinstance(grid_evidence, Mapping):
                failures.append("common-grid evidence is missing")
            else:
                if grid_evidence.get("Schema") != expected_grid["schema"]:
                    failures.append(
                        "common-grid schema differs from configured evidence"
                    )
                if grid_evidence.get("SHA256") != expected_grid["contract_sha256"]:
                    failures.append("common-grid contract SHA-256 differs")
                if (
                    grid_evidence.get("ReferenceSHA256")
                    != expected_grid["reference"]["sha256"]
                ):
                    failures.append("common-grid reference SHA-256 differs")
                if (
                    grid_evidence.get("ArchitectureShape")
                    != expected_grid["architecture_shape"]
                ):
                    failures.append("common-grid architecture shape differs")
                if (
                    grid_evidence.get("AnatomicalShape")
                    != expected_grid["anatomical_shape"]
                ):
                    failures.append("common-grid anatomical crop differs")
                if (
                    grid_evidence.get("ArchitecturePadding")
                    != expected_grid["architecture_padding"]
                ):
                    failures.append("common-grid architecture padding differs")
                if grid_evidence.get("MatrixSizeReportedByPaper") is not False:
                    failures.append(
                        "spatial matrix is incorrectly represented as a paper fact"
                    )
        if provenance.get("FMRIPrepApplied") is not True:
            failures.append("fMRIPrep provenance missing")
        generated_by = provenance.get("FMRIPrepGeneratedBy")
        if not isinstance(generated_by, dict):
            failures.append("fMRIPrep GeneratedBy snapshot missing")
        elif (
            str(generated_by.get("Name", "")).strip().lower() != "fmriprep"
            or not str(generated_by.get("Version", "")).strip()
        ):
            failures.append("fMRIPrep name/version provenance invalid")
        evidence = provenance.get("FMRIPrepEvidence")
        artifacts: dict[str, Mapping] = {}
        if not isinstance(evidence, dict):
            failures.append("hash-bound fMRIPrep evidence missing")
        else:
            expected_evidence_fields = {
                "EvidenceSchemaVersion",
                "ParticipantLabel",
                "ExecutionWrapper",
                "ExecutionSuccess",
                "ExecutionReturnCode",
                "ExecutionStartedAtUTC",
                "ExecutionCompletedAtUTC",
                "ScanIdentity",
                "OutputSpaces",
                "ExtraArguments",
                "IgnoredFeatures",
                "SliceTimingEnabledByWrapper",
                "SliceTimingReference",
                "DerivativeSliceTimingCorrected",
                "MotionConfoundsRows",
                "FreshDerivativesGeneration",
                "BIDSInputInventory",
                "ExecutableIdentity",
                "RuntimeIdentity",
                "VersionProbeBeforeExecution",
                "VersionProbeAfterExecution",
                "CoregistrationBinding",
                "SourceAcquisitionIdentity",
                "SourceAcquisition",
                "ExecutionRecord",
                "DerivativeDatasetDescription",
                "BOLDDerivative",
                "BOLDBrainMask",
                "BOLDMetadata",
                "MotionConfounds",
                "BOLDToT1wTransform",
                "T1wReference",
                "T1wMetadata",
                "StdoutLog",
                "StderrLog",
            }
            if set(evidence) != expected_evidence_fields:
                failures.append("fMRIPrep evidence fields differ from schema v4")
            participant = scan_id.split("_", 1)[0].removeprefix("sub-")
            if (
                evidence.get("EvidenceSchemaVersion")
                != FMRIPREP_EVIDENCE_SCHEMA_VERSION
            ):
                failures.append("wrong fMRIPrep evidence schema")
            if evidence.get("ParticipantLabel") != participant:
                failures.append("fMRIPrep evidence identifies another participant")
            if evidence.get("ExecutionWrapper") != "preprocessing.run_fmriprep":
                failures.append("fMRIPrep evidence was not produced by the wrapper")
            if evidence.get("ExecutionSuccess") is not True:
                failures.append("fMRIPrep wrapper execution was not successful")
            if evidence.get("ExecutionReturnCode") != 0:
                failures.append("fMRIPrep wrapper return code is not zero")
            for timestamp in ("ExecutionStartedAtUTC", "ExecutionCompletedAtUTC"):
                value = evidence.get(timestamp)
                if not isinstance(value, str) or not value.endswith("Z"):
                    failures.append(f"{timestamp} is not a UTC timestamp")
            if evidence.get("OutputSpaces") != ["T1w"]:
                failures.append("fMRIPrep evidence does not fix output space to T1w")
            try:
                _validated_fmriprep_extra_arguments(evidence.get("ExtraArguments"))
            except ValueError as exc:
                failures.append(f"fMRIPrep extra-argument evidence is invalid: {exc}")
            ignored = evidence.get("IgnoredFeatures")
            if not isinstance(ignored, list) or not all(
                isinstance(value, str) for value in ignored
            ):
                failures.append("fMRIPrep ignored-feature evidence is invalid")
            elif any(
                value.lower().replace("-", "").replace("_", "") == "slicetiming"
                for value in ignored
            ):
                failures.append("fMRIPrep evidence disabled slice timing")
            if evidence.get("SliceTimingEnabledByWrapper") is not True:
                failures.append("slice-timing wrapper evidence missing")
            slice_reference = self._finite_float(evidence.get("SliceTimingReference"))
            if not np.isclose(slice_reference, 0.5, rtol=0.0, atol=1e-12):
                failures.append("slice-timing reference evidence differs")
            if evidence.get("DerivativeSliceTimingCorrected") is not True:
                failures.append("fMRIPrep derivative does not confirm slice timing")
            motion_rows = evidence.get("MotionConfoundsRows")
            if (
                isinstance(motion_rows, bool)
                or not isinstance(motion_rows, int)
                or motion_rows < 5
            ):
                failures.append("motion-confounds frame evidence is invalid")

            artifact_names = (
                "ExecutionRecord",
                "DerivativeDatasetDescription",
                "BOLDDerivative",
                "BOLDBrainMask",
                "BOLDMetadata",
                "MotionConfounds",
                "BOLDToT1wTransform",
                "T1wReference",
                "T1wMetadata",
                "StdoutLog",
                "StderrLog",
            )
            for artifact_name in artifact_names:
                artifact = evidence.get(artifact_name)
                if not isinstance(artifact, dict):
                    failures.append(f"{artifact_name} hash evidence missing")
                    continue
                path = artifact.get("Path")
                relative = artifact.get("RelativePath")
                digest = artifact.get("SHA256")
                size = artifact.get("SizeBytes")
                if not isinstance(path, str) or not Path(path).is_absolute():
                    failures.append(f"{artifact_name} absolute path is invalid")
                if (
                    not isinstance(relative, str)
                    or not relative
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                ):
                    failures.append(
                        f"{artifact_name} derivative-relative path is invalid"
                    )
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    failures.append(f"{artifact_name} SHA-256 is invalid")
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    or (artifact_name not in {"StdoutLog", "StderrLog"} and size == 0)
                ):
                    failures.append(f"{artifact_name} size evidence is invalid")
                if isinstance(path, str) and Path(path).is_absolute():
                    try:
                        source_path = Path(path).resolve(strict=True)
                        if str(source_path) != path:
                            failures.append(f"{artifact_name} path is not canonical")
                        if source_path.stat().st_size != size:
                            failures.append(
                                f"{artifact_name} recorded size no longer matches"
                            )
                        if sha256_file(source_path) != digest:
                            failures.append(
                                f"{artifact_name} recorded SHA-256 no longer matches"
                            )
                    except Exception as exc:
                        failures.append(f"{artifact_name} cannot be revalidated: {exc}")
                artifacts[artifact_name] = artifact

            source_identity = None
            observed_source_artifact = None
            source_artifact = evidence.get("SourceAcquisitionIdentity")
            source_record = evidence.get("SourceAcquisition")
            if not isinstance(source_artifact, dict) or set(source_artifact) != {
                "Path",
                "SHA256",
                "SizeBytes",
                "RecordSHA256",
            }:
                failures.append("source-acquisition identity hash evidence missing")
            elif not isinstance(source_record, dict):
                failures.append("source-acquisition signed record missing")
            else:
                try:
                    source_identity, observed_source_artifact = (
                        load_source_acquisition_identity(
                            Path(str(source_artifact["Path"])),
                            expected_sha256=str(source_artifact["SHA256"]),
                            expected_participant=participant,
                            expected_scan_id=scan_id,
                        )
                    )
                    expected_source_artifact = {
                        "Path": observed_source_artifact["path"],
                        "SHA256": observed_source_artifact["sha256"],
                        "SizeBytes": observed_source_artifact["size_bytes"],
                        "RecordSHA256": observed_source_artifact["record_sha256"],
                    }
                    if source_artifact != expected_source_artifact:
                        failures.append(
                            "source-acquisition identity artifact evidence differs"
                        )
                    if source_identity != source_record:
                        failures.append(
                            "source-acquisition signed record differs from evidence"
                        )
                    source_entities = source_identity["bids_entities"]["raw_bold"]
                    source_inventory = source_identity["bids_input_inventory"]
                    expected_scan_identity = {
                        "ScanID": scan_id,
                        "ParticipantLabel": participant,
                        "Session": source_entities.get("ses"),
                        "Task": source_entities.get("task"),
                        "Run": source_entities.get("run"),
                        "Acquisition": source_entities.get("acq"),
                        "Direction": source_entities.get("dir"),
                        "Echo": source_entities.get("echo"),
                        "RawBOLDRelativePath": source_inventory[
                            "raw_bold_relative_path"
                        ],
                        "RawT1wRelativePath": source_inventory["raw_t1_relative_path"],
                    }
                    if evidence.get("ScanIdentity") != expected_scan_identity:
                        failures.append(
                            "fMRIPrep evidence identifies another exact scan"
                        )
                    if evidence.get("BIDSInputInventory") != {
                        "Identity": source_inventory,
                        "VerifiedBeforeExecution": True,
                        "VerifiedAfterExecution": True,
                    }:
                        failures.append(
                            "fMRIPrep evidence BIDS input inventory differs"
                        )
                except (KeyError, OSError, SourceAcquisitionError) as exc:
                    failures.append(
                        f"source-acquisition identity cannot be revalidated: {exc}"
                    )

            if (
                self.common_grid_contract is not None
                and source_identity is not None
                and observed_source_artifact is not None
            ):
                try:
                    structural_source_identity = self.source_dataset.source_fingerprint(
                        scan_id
                    ).get("structural_source_identity")
                    self._require_matching_structural_fmri_source_identity(
                        structural_source_identity,
                        source_identity,
                        observed_source_artifact,
                    )
                except (KeyError, OSError, SourceAcquisitionError) as exc:
                    failures.append(
                        "structural/fMRI source-acquisition join cannot be "
                        f"authenticated: {exc}"
                    )

            derivative_description = artifacts.get("DerivativeDatasetDescription")
            derivative_root = None
            if derivative_description is not None:
                description_path = derivative_description.get("Path")
                if (
                    isinstance(description_path, str)
                    and Path(description_path).is_absolute()
                ):
                    derivative_root = Path(description_path).parent
                    if (
                        derivative_description.get("RelativePath")
                        != "dataset_description.json"
                    ):
                        failures.append(
                            "derivative dataset description is not at its root"
                        )
                    for artifact_name, artifact in artifacts.items():
                        relative = artifact.get("RelativePath")
                        path = artifact.get("Path")
                        if isinstance(relative, str) and isinstance(path, str):
                            try:
                                expected = (derivative_root / relative).resolve(
                                    strict=True
                                )
                                if expected != Path(path).resolve(strict=True):
                                    failures.append(
                                        f"{artifact_name} path disagrees with derivatives root"
                                    )
                            except Exception:
                                # The concrete missing/unreadable-artifact failure was
                                # already recorded above.
                                pass

            fresh_generation = evidence.get("FreshDerivativesGeneration")
            generation_id = (
                fresh_generation.get("GenerationID")
                if isinstance(fresh_generation, Mapping)
                else None
            )
            if (
                not isinstance(fresh_generation, Mapping)
                or set(fresh_generation)
                != {
                    "GenerationID",
                    "ScanID",
                    "Path",
                    "PathAbsentBeforeCreation",
                    "CreatedExclusively",
                    "InitiallyEmpty",
                    "NoReuse",
                }
                or not isinstance(generation_id, str)
                or len(generation_id) != 64
                or any(
                    character not in "0123456789abcdef" for character in generation_id
                )
                or fresh_generation.get("ScanID") != scan_id
                or derivative_root is None
                or fresh_generation.get("Path") != str(derivative_root)
                or fresh_generation.get("PathAbsentBeforeCreation") is not True
                or fresh_generation.get("CreatedExclusively") is not True
                or fresh_generation.get("InitiallyEmpty") is not True
                or fresh_generation.get("NoReuse") is not True
            ):
                failures.append("fMRIPrep fresh-derivatives generation differs")

            executable_identity = evidence.get("ExecutableIdentity")
            runtime_binding = evidence.get("RuntimeIdentity")
            if (
                not isinstance(executable_identity, Mapping)
                or not isinstance(runtime_binding, Mapping)
                or set(runtime_binding)
                != {
                    "Artifact",
                    "Identity",
                    "VerifiedBeforeExecution",
                    "VerifiedAfterExecution",
                }
                or not isinstance(runtime_binding.get("Artifact"), Mapping)
                or set(runtime_binding.get("Artifact", {}))
                != {"Path", "SHA256", "SizeBytes", "RecordSHA256"}
            ):
                failures.append("fMRIPrep runtime evidence is malformed")
            else:
                runtime_artifact = runtime_binding["Artifact"]
                try:
                    runtime_identity, runtime_evidence = load_fmriprep_runtime_identity(
                        Path(str(runtime_artifact["Path"])),
                        expected_sha256=str(runtime_artifact["SHA256"]),
                        expected_executable=Path(
                            str(executable_identity.get("path", ""))
                        ),
                    )
                    expected_runtime_artifact = {
                        "Path": runtime_evidence["path"],
                        "SHA256": runtime_evidence["sha256"],
                        "SizeBytes": runtime_evidence["size_bytes"],
                        "RecordSHA256": runtime_evidence["record_sha256"],
                    }
                    if (
                        runtime_artifact != expected_runtime_artifact
                        or runtime_binding.get("Identity") != runtime_identity
                        or runtime_binding.get("VerifiedBeforeExecution") is not True
                        or runtime_binding.get("VerifiedAfterExecution") is not True
                        or executable_identity != runtime_identity.get("executable")
                        or not Path(str(executable_identity.get("path", ""))).is_file()
                        or not os.access(
                            Path(str(executable_identity.get("path", ""))),
                            os.X_OK,
                        )
                    ):
                        failures.append("fMRIPrep runtime evidence differs")
                    runtime_version = str(runtime_identity["fmriprep_version"])
                    for probe_key, probe_label in (
                        (
                            "VersionProbeBeforeExecution",
                            "pre-execution fMRIPrep version probe",
                        ),
                        (
                            "VersionProbeAfterExecution",
                            "post-execution fMRIPrep version probe",
                        ),
                    ):
                        _validated_fmriprep_version_probe(
                            evidence.get(probe_key),
                            executable=Path(str(executable_identity["path"])),
                            expected_version=runtime_version,
                            label=probe_label,
                        )
                    if (
                        isinstance(generated_by, Mapping)
                        and generated_by.get("Version") != runtime_version
                    ):
                        failures.append(
                            "fMRIPrep runtime version differs from GeneratedBy"
                        )
                except (
                    KeyError,
                    OSError,
                    ValueError,
                    SourceAcquisitionError,
                ) as exc:
                    failures.append(
                        f"fMRIPrep runtime evidence cannot be revalidated: {exc}"
                    )

            binding = evidence.get("CoregistrationBinding")
            if not isinstance(binding, dict):
                failures.append("BOLD-to-T1w transform binding missing")
            elif all(
                name in artifacts
                for name in ("BOLDDerivative", "BOLDToT1wTransform", "T1wReference")
            ):
                run_prefix = binding.get("BOLDRunPrefix")
                if (
                    not isinstance(run_prefix, str)
                    or run_prefix.split("_", 1)[0] != f"sub-{participant}"
                ):
                    failures.append(
                        "BOLD-to-T1w transform binding identifies another run"
                    )
                if binding.get("TransformFrom") not in {"boldref", "scanner"}:
                    failures.append("BOLD-to-T1w transform source is invalid")
                if binding.get("TransformTo") != "T1w":
                    failures.append("BOLD-to-T1w transform target is invalid")
                if (
                    binding.get("TransformSemantic")
                    != "BOLD-to-subject-T1w registration"
                ):
                    failures.append("BOLD-to-T1w transform semantics are invalid")
                if binding.get("T1wSubject") != participant:
                    failures.append("BOLD-to-T1w target identifies another participant")
                for binding_key, artifact_name in (
                    ("BOLDDerivativeSHA256", "BOLDDerivative"),
                    ("BOLDBrainMaskSHA256", "BOLDBrainMask"),
                    ("TransformArtifactSHA256", "BOLDToT1wTransform"),
                    ("TargetT1wArtifactSHA256", "T1wReference"),
                ):
                    if binding.get(binding_key) != artifacts[artifact_name].get(
                        "SHA256"
                    ):
                        failures.append(f"{binding_key} does not match its artifact")
                if source_identity is not None:
                    for binding_key, source_value in (
                        ("RawT1wSHA256", source_identity["raw_t1"]["sha256"]),
                        ("RawBOLDSHA256", source_identity["raw_bold"]["sha256"]),
                        (
                            "SynthSegMaskSHA256",
                            source_identity["synthseg"]["mask"]["sha256"],
                        ),
                    ):
                        if binding.get(binding_key) != source_value:
                            failures.append(
                                f"{binding_key} does not match source acquisition"
                            )

            if source_identity is not None and all(
                name in artifacts
                for name in ("ExecutionRecord", "BOLDMetadata", "T1wMetadata")
            ):
                expected_source_binding = {
                    "Artifact": source_artifact,
                    "Identity": source_identity,
                    "VerifiedBeforeExecution": True,
                    "VerifiedAfterExecution": True,
                }
                try:
                    execution_record, _ = load_authenticated_json_artifact(
                        Path(artifacts["ExecutionRecord"]["Path"]),
                        expected_sha256=str(artifacts["ExecutionRecord"]["SHA256"]),
                        expected_size=int(artifacts["ExecutionRecord"]["SizeBytes"]),
                        label="fMRIPrep execution record",
                    )
                    if not isinstance(execution_record, dict):
                        failures.append(
                            "fMRIPrep execution record is not a JSON object"
                        )
                        execution_record = {}
                    expected_execution_fields = {
                        "SchemaVersion",
                        "Wrapper",
                        "Success",
                        "ReturnCode",
                        "StartedAtUTC",
                        "CompletedAtUTC",
                        "Command",
                        "BIDSRoot",
                        "DerivativesRoot",
                        "WorkRoot",
                        "ScanIdentity",
                        "ParticipantLabel",
                        "OutputSpaces",
                        "ExtraArguments",
                        "IgnoredFeatures",
                        "SliceTimingEnabledByWrapper",
                        "SliceTimingReference",
                        "FMRIPrepGeneratedBy",
                        "FreshDerivativesGeneration",
                        "BIDSInputInventory",
                        "ExecutableIdentity",
                        "RuntimeIdentity",
                        "VersionProbeBeforeExecution",
                        "VersionProbeAfterExecution",
                        "SourceAcquisitionIdentity",
                        "StdoutLog",
                        "StderrLog",
                        "Artifacts",
                    }
                    if (
                        set(execution_record) != expected_execution_fields
                        or execution_record.get("SchemaVersion")
                        != FMRIPREP_EXECUTION_SCHEMA_VERSION
                        or execution_record.get("Wrapper")
                        != "preprocessing.run_fmriprep"
                        or execution_record.get("Success") is not True
                        or execution_record.get("ReturnCode") != 0
                        or execution_record.get("FMRIPrepGeneratedBy") != generated_by
                    ):
                        failures.append("fMRIPrep execution record schema differs")
                    if (
                        execution_record.get("SourceAcquisitionIdentity")
                        != expected_source_binding
                    ):
                        failures.append(
                            "fMRIPrep execution record source-acquisition binding differs"
                        )
                    for field in (
                        "ScanIdentity",
                        "OutputSpaces",
                        "ExtraArguments",
                        "IgnoredFeatures",
                        "SliceTimingEnabledByWrapper",
                        "SliceTimingReference",
                        "FreshDerivativesGeneration",
                        "BIDSInputInventory",
                        "ExecutableIdentity",
                        "RuntimeIdentity",
                        "VersionProbeBeforeExecution",
                        "VersionProbeAfterExecution",
                    ):
                        if execution_record.get(field) != evidence.get(field):
                            failures.append(
                                f"fMRIPrep execution/evidence {field} differs"
                            )
                    if derivative_root is not None:
                        executable_path = Path(
                            str(
                                execution_record.get("ExecutableIdentity", {}).get(
                                    "path", ""
                                )
                            )
                        )
                        expected_work_root = derivative_root / "connect4-work"
                        try:
                            recorded_extra_arguments = (
                                _validated_fmriprep_extra_arguments(
                                    execution_record.get("ExtraArguments")
                                )
                            )
                        except ValueError as exc:
                            failures.append(
                                f"fMRIPrep execution extra arguments differ: {exc}"
                            )
                            recorded_extra_arguments = []
                        expected_command = [
                            str(executable_path),
                            str(Path(source_identity["bids_root"])),
                            str(derivative_root),
                            "participant",
                            "--participant-label",
                            participant,
                            "--output-spaces",
                            "T1w",
                            "--slice-time-ref",
                            "0.5",
                            "--level",
                            "full",
                            "--work-dir",
                            str(expected_work_root),
                            *recorded_extra_arguments,
                        ]
                        if (
                            execution_record.get("Command") != expected_command
                            or execution_record.get("BIDSRoot")
                            != str(Path(source_identity["bids_root"]))
                            or execution_record.get("DerivativesRoot")
                            != str(derivative_root)
                            or execution_record.get("WorkRoot")
                            != str(expected_work_root)
                            or execution_record.get("ParticipantLabel") != participant
                        ):
                            failures.append(
                                "fMRIPrep execution command/root identity differs"
                            )
                    bids_root = Path(source_identity["bids_root"])
                    raw_bold_relative = (
                        Path(source_identity["raw_bold"]["path"])
                        .relative_to(bids_root)
                        .as_posix()
                    )
                    raw_t1_relative = (
                        Path(source_identity["raw_t1"]["path"])
                        .relative_to(bids_root)
                        .as_posix()
                    )
                    bold_metadata, _ = load_authenticated_json_artifact(
                        Path(artifacts["BOLDMetadata"]["Path"]),
                        expected_sha256=str(artifacts["BOLDMetadata"]["SHA256"]),
                        expected_size=int(artifacts["BOLDMetadata"]["SizeBytes"]),
                        label="fMRIPrep BOLD metadata",
                    )
                    t1_metadata, _ = load_authenticated_json_artifact(
                        Path(artifacts["T1wMetadata"]["Path"]),
                        expected_sha256=str(artifacts["T1wMetadata"]["SHA256"]),
                        expected_size=int(artifacts["T1wMetadata"]["SizeBytes"]),
                        label="fMRIPrep T1w metadata",
                    )
                    if raw_bold_relative not in derivative_source_relative_paths(
                        bold_metadata
                    ):
                        failures.append(
                            "BOLD derivative Sources omit authenticated raw BOLD"
                        )
                    if raw_t1_relative not in derivative_source_relative_paths(
                        t1_metadata
                    ):
                        failures.append(
                            "T1w derivative Sources omit authenticated raw T1w"
                        )
                except (
                    KeyError,
                    OSError,
                    ValueError,
                    SourceAcquisitionError,
                ) as exc:
                    failures.append(
                        "source-acquisition derivative binding cannot be "
                        f"revalidated: {exc}"
                    )

            if artifacts.get("ExecutionRecord", {}).get("RelativePath") != (
                f"logs/connect4-fmriprep_{scan_id}_execution.json"
            ):
                failures.append("fMRIPrep execution-record path is invalid")
            for provenance_key, artifact_name in (
                ("CoregistrationTransform", "BOLDToT1wTransform"),
                ("MotionParameters", "MotionConfounds"),
                ("SourceMetadata", "BOLDMetadata"),
            ):
                if artifact_name in artifacts and provenance.get(
                    provenance_key
                ) != artifacts[artifact_name].get("Path"):
                    failures.append(
                        f"{provenance_key} does not match fMRIPrep evidence"
                    )
            for provenance_key, artifact_name in (
                ("CoregistrationTransformSHA256", "BOLDToT1wTransform"),
                ("MotionParametersSHA256", "MotionConfounds"),
            ):
                if artifact_name in artifacts and provenance.get(
                    provenance_key
                ) != artifacts[artifact_name].get("SHA256"):
                    failures.append(
                        f"{provenance_key} does not match fMRIPrep evidence"
                    )
        functional_validity = provenance.get("FunctionalValidityMask")
        functional_validity_path = self._target_validity_mask_path(scan_id)
        expected_functional_fields = {
            "Path",
            "SHA256",
            "SizeBytes",
            "Role",
            "Derivation",
            "CertifiedFMRIPrepCoverage",
            "SourceBOLDBrainMaskSHA256",
            "Shape",
            "ValidVoxelCount",
            "StructuralVoxelCount",
            "BOLDSupportVoxelCount",
            "StructuralVoxelsExcludedFromLoss",
            "AppliedToSpatialSmoothing",
            "AppliedToIntensityNormalization",
            "OutsideMaskForcedZero",
            "EqualsExactFinalNonzeroSupport",
        }
        if (
            not isinstance(functional_validity, Mapping)
            or set(functional_validity) != expected_functional_fields
        ):
            failures.append("functional-validity mask evidence is incomplete")
        else:
            try:
                resolved_validity_path = functional_validity_path.resolve(strict=True)
                validity_digest = sha256_file(resolved_validity_path)
                validity_size = resolved_validity_path.stat().st_size
            except Exception as exc:
                failures.append(
                    f"functional-validity mask cannot be revalidated: {exc}"
                )
                resolved_validity_path = functional_validity_path
                validity_digest = None
                validity_size = None
            bold_mask_digest = artifacts.get("BOLDBrainMask", {}).get("SHA256")
            integer_fields = (
                "ValidVoxelCount",
                "StructuralVoxelCount",
                "BOLDSupportVoxelCount",
                "StructuralVoxelsExcludedFromLoss",
            )
            if (
                functional_validity.get("Path") != str(resolved_validity_path)
                or functional_validity.get("SHA256") != validity_digest
                or functional_validity.get("SizeBytes") != validity_size
                or functional_validity.get("Role")
                != "fMRI spatial support and model-loss validity"
                or functional_validity.get("Derivation")
                != (
                    "nearest-neighbour common-grid fMRIPrep BOLD brain mask "
                    "intersected with nonzero structural-mask support"
                )
                or functional_validity.get("CertifiedFMRIPrepCoverage") is not True
                or functional_validity.get("SourceBOLDBrainMaskSHA256")
                != bold_mask_digest
                or functional_validity.get("Shape") != list(self.target_shape)
                or any(
                    isinstance(functional_validity.get(name), bool)
                    or not isinstance(functional_validity.get(name), int)
                    or functional_validity[name] < 0
                    for name in integer_fields
                )
                or functional_validity.get("ValidVoxelCount", 0) < 1
                or functional_validity.get("AppliedToSpatialSmoothing") is not True
                or functional_validity.get("AppliedToIntensityNormalization")
                is not bool(self.normalize_intensity)
                or functional_validity.get("OutsideMaskForcedZero") is not True
                or functional_validity.get("EqualsExactFinalNonzeroSupport") is not True
            ):
                failures.append("functional-validity mask provenance differs")
            elif (
                functional_validity["ValidVoxelCount"]
                + functional_validity["StructuralVoxelsExcludedFromLoss"]
                != functional_validity["StructuralVoxelCount"]
                or functional_validity["ValidVoxelCount"]
                > functional_validity["BOLDSupportVoxelCount"]
            ):
                failures.append("functional-validity mask voxel counts differ")

        if provenance.get("CoregistrationEvidenceValidated") is not True:
            failures.append("hash-bound T1w co-registration evidence not validated")
        output_digest = provenance.get("OutputSHA256")
        if (
            not isinstance(output_digest, str)
            or len(output_digest) != 64
            or any(character not in "0123456789abcdef" for character in output_digest)
        ):
            failures.append("output fMRI SHA-256 provenance is invalid")
        else:
            try:
                if sha256_file(fmri_path) != output_digest:
                    failures.append("output fMRI SHA-256 no longer matches")
            except Exception as exc:
                failures.append(f"output fMRI cannot be hash-validated: {exc}")
        pipeline = provenance.get("PipelineDescription", {})
        if pipeline.get("Name") != "fMRIPrep + CONNECT-4 harmonisation":
            failures.append("pipeline identity does not match fMRIPrep + CONNECT-4")
        if pipeline.get("Steps") != self._PREPROCESSING_STEPS:
            failures.append(
                "pipeline steps/order do not match the certified preprocessing contract"
            )
        for key, description in (
            ("SliceTimingApplied", "slice-timing correction missing"),
            ("MotionCorrectionApplied", "motion correction missing"),
            ("SpatialSmoothingApplied", "spatial smoothing missing"),
            ("TemporalFilteringApplied", "temporal filtering missing"),
            ("CoregistrationValidated", "T1w co-registration not validated"),
        ):
            if provenance.get(key) is not True:
                failures.append(description)
        if not np.isclose(
            self._finite_float(provenance.get("SpatialSmoothingFWHM")),
            DEFAULT_SMOOTHING_FWHM_MM,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append(
                "spatial smoothing is not the versioned 3-mm recovery choice"
            )
        if provenance.get("SpatialSmoothingMethod") != (
            "Gaussian signal/mask division within functional-validity mask"
        ):
            failures.append("spatial smoothing was not mask-normalized")
        gradient_retention = self._finite_float(
            provenance.get("SmoothingGradientRetentionRatio")
        )
        if gradient_retention <= MIN_SMOOTHING_GRADIENT_RETENTION:
            failures.append("spatial smoothing failed gradient-retention policy")
        if provenance.get("TemporalPaddingApplied") is not False:
            failures.append("temporal padding was applied or not explicitly excluded")
        recorded_mask_digest = provenance.get("AnatomicalMaskSHA256")
        expected_mask_path = self._segmentation_path(scan_id)
        try:
            if recorded_mask_digest != sha256_file(expected_mask_path):
                failures.append(
                    "smoothing-mask SHA-256 differs from the certified mask"
                )
        except Exception as exc:
            failures.append(f"certified smoothing mask cannot be hashed: {exc}")
        if "fmriprep" not in str(provenance.get("CoregistrationBackend", "")).lower():
            failures.append("co-registration backend is not fMRIPrep")
        if provenance.get("OutputShape") != [*self.target_shape, self.num_frames]:
            failures.append("provenance output shape is not the configured target")
        if not np.isclose(self._finite_float(provenance.get("RepetitionTime")), 3.0):
            failures.append("output TR is not 3 seconds")
        voxel_size = np.asarray(provenance.get("VoxelSize", []), dtype=np.float64)
        if voxel_size.shape != (3,) or not np.allclose(voxel_size, 3.0):
            failures.append("output voxels are not 3 mm isotropic")
        if provenance.get("TemporalFilterPreservesVoxelMean") is not True:
            failures.append("temporal filtering did not preserve voxel means")
        if not np.isclose(
            self._finite_float(provenance.get("TemporalHighPassHz")),
            DEFAULT_HIGH_PASS_HZ,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append("top-level temporal high-pass cutoff differs")
        if not np.isclose(
            self._finite_float(provenance.get("TemporalLowPassHz")),
            DEFAULT_LOW_PASS_HZ,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append("top-level temporal low-pass cutoff differs")
        if not np.isclose(
            self._finite_float(provenance.get("SliceTimingReference")),
            DEFAULT_SLICE_TIMING_REFERENCE,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append("top-level slice-timing reference differs")
        if provenance.get("TemporalZScore") is not False:
            failures.append("forbidden temporal voxel z-score was applied")
        if self.normalize_intensity:
            intensity = provenance.get("IntensityNormalization")
            expected_intensity_fields = {
                "Contract",
                "ReportedByPaper",
                "Method",
                "Domain",
                "UpperQuantile",
                "FunctionalValidityQuantileBeforeClip",
                "CeilingAndDivisor",
                "MinimumDivisor",
                "FunctionalValidityValueCount",
                "FractionNegativeBeforeClipInFunctionalValidity",
                "FractionClippedAtCeilingInFunctionalValidity",
                "LowerQuantileSubtraction",
                "TemporalVoxelZScore",
                "OutsideMaskForcedZero",
                "OutputMinimum",
                "OutputMaximum",
            }
            if provenance.get("IntensityNormalizationApplied") is not True:
                failures.append("configured intensity normalization missing")
            elif not isinstance(intensity, Mapping):
                failures.append("intensity-normalization evidence missing")
            else:
                quantile = self._finite_float(
                    intensity.get("FunctionalValidityQuantileBeforeClip")
                )
                ceiling = self._finite_float(intensity.get("CeilingAndDivisor"))
                negative_fraction = self._finite_float(
                    intensity.get("FractionNegativeBeforeClipInFunctionalValidity")
                )
                clipped_fraction = self._finite_float(
                    intensity.get("FractionClippedAtCeilingInFunctionalValidity")
                )
                output_minimum = self._finite_float(intensity.get("OutputMinimum"))
                output_maximum = self._finite_float(intensity.get("OutputMaximum"))
                value_count = intensity.get("FunctionalValidityValueCount")
                expected_value_count = (
                    int(functional_validity.get("ValidVoxelCount", -1))
                    * self.num_frames
                    if isinstance(functional_validity, Mapping)
                    else -1
                )
                intensity_differs = (
                    set(intensity) != expected_intensity_fields
                    or intensity.get("Contract")
                    != FMRI_INTENSITY_NORMALIZATION_CONTRACT
                    or intensity.get("ReportedByPaper") is not False
                    or intensity.get("Method") != FMRI_INTENSITY_NORMALIZATION_METHOD
                    or intensity.get("Domain") != FMRI_INTENSITY_NORMALIZATION_DOMAIN
                    or not np.isclose(
                        self._finite_float(intensity.get("UpperQuantile")),
                        FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    or quantile < 0.0
                    or not np.isclose(ceiling, max(quantile, 1.0), rtol=0.0, atol=1e-12)
                    or not np.isclose(
                        self._finite_float(intensity.get("MinimumDivisor")),
                        1.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    or isinstance(value_count, bool)
                    or not isinstance(value_count, int)
                    or value_count != expected_value_count
                    or not 0.0 <= negative_fraction <= 1.0
                    or not 0.0 <= clipped_fraction <= 1.0
                    or intensity.get("LowerQuantileSubtraction") is not False
                    or intensity.get("TemporalVoxelZScore") is not False
                    or intensity.get("OutsideMaskForcedZero") is not True
                    or not 0.0 <= output_minimum <= output_maximum <= 1.0
                )
                if intensity_differs:
                    failures.append("intensity-normalization evidence differs")

        reference_affine = (
            nib.load(self.common_grid_contract["reference_path"]).affine
            if self.common_grid_contract is not None
            else None
        )
        try:
            image = nib.load(str(fmri_path))
            if tuple(image.shape) != (*self.target_shape, self.num_frames):
                failures.append(f"NIfTI shape is {image.shape}")
            zooms = np.asarray(image.header.get_zooms()[:4], dtype=np.float64)
            if zooms.shape != (4,) or not np.allclose(zooms, (3.0, 3.0, 3.0, 3.0)):
                failures.append(f"NIfTI voxel sizes/TR are {tuple(zooms)}")
            spatial_unit, temporal_unit = image.header.get_xyzt_units()
            if (spatial_unit, temporal_unit) != ("mm", "sec"):
                failures.append(
                    "NIfTI spatial/time units are "
                    f"{(spatial_unit, temporal_unit)!r}, expected ('mm', 'sec')"
                )
            if reference_affine is not None and not np.allclose(
                image.affine, reference_affine, rtol=0.0, atol=1e-6
            ):
                failures.append("fMRI affine differs from the common-grid reference")
            elif reference_affine is None:
                reference_affine = image.affine
            if self.normalize_intensity:
                values = np.asarray(image.dataobj, dtype=np.float32)
                if not np.isfinite(values).all():
                    failures.append("normalized fMRI contains NaN or infinity")
                elif float(values.min()) < -1e-6 or float(values.max()) > 1.0 + 1e-6:
                    failures.append(
                        "normalized fMRI lies outside the certified [0,1] range"
                    )
                elif isinstance(intensity, Mapping) and (
                    not np.isclose(
                        float(values.min()),
                        self._finite_float(intensity.get("OutputMinimum")),
                        rtol=0.0,
                        atol=1e-7,
                    )
                    or not np.isclose(
                        float(values.max()),
                        self._finite_float(intensity.get("OutputMaximum")),
                        rtol=0.0,
                        atol=1e-7,
                    )
                ):
                    failures.append("normalized fMRI extrema differ from provenance")
        except Exception as exc:
            failures.append(f"fMRI header cannot be validated: {exc}")

        for label, path in (
            ("T1w", self._t1_path(scan_id)),
            ("segmentation", self._segmentation_path(scan_id)),
        ):
            try:
                aligned = nib.load(str(path))
                if aligned.ndim != 3 or tuple(aligned.shape) != self.target_shape:
                    failures.append(f"{label} shape is {aligned.shape}")
                zooms = np.asarray(aligned.header.get_zooms()[:3], dtype=np.float64)
                if zooms.shape != (3,) or not np.allclose(zooms, 3.0):
                    failures.append(f"{label} is not 3 mm isotropic")
                if reference_affine is not None and not np.allclose(
                    aligned.affine, reference_affine, rtol=0.0, atol=1e-3
                ):
                    failures.append(f"{label} affine is not aligned with fMRI")
            except Exception as exc:
                failures.append(f"{label} header cannot be validated: {exc}")

        try:
            target_image = nib.load(str(fmri_path))
            target_values = target_image.get_fdata(dtype=np.float32)
            structural_image = nib.load(str(self._segmentation_path(scan_id)))
            structural_values = structural_image.get_fdata(dtype=np.float32) > 0.5
            validity_image = nib.load(str(functional_validity_path))
            validity_values = validity_image.get_fdata(dtype=np.float32)
            if (
                validity_image.ndim != 3
                or tuple(validity_image.shape) != self.target_shape
            ):
                failures.append("functional-validity mask grid differs")
            elif (
                not np.isfinite(validity_values).all()
                or not np.logical_or(
                    validity_values == 0.0, validity_values == 1.0
                ).all()
            ):
                failures.append("functional-validity mask is not exactly binary")
            elif not np.allclose(
                validity_image.affine,
                target_image.affine,
                rtol=0.0,
                atol=1e-6,
            ):
                failures.append("functional-validity mask affine differs")
            else:
                expected_validity = target_validity_mask_from_fmri(
                    target_values,
                    structural_values,
                    scan_id=scan_id,
                )
                if not np.array_equal(validity_values, expected_validity):
                    failures.append(
                        "functional-validity mask differs from exact stored "
                        "nonzero target support"
                    )
                elif isinstance(functional_validity, Mapping) and (
                    int(np.count_nonzero(validity_values))
                    != functional_validity.get("ValidVoxelCount")
                    or int(np.count_nonzero(structural_values))
                    != functional_validity.get("StructuralVoxelCount")
                ):
                    failures.append(
                        "functional-validity mask content/count evidence differs"
                    )
        except Exception as exc:
            failures.append(f"functional-validity mask cannot be validated: {exc}")

        if failures:
            raise RuntimeError(
                f"{scan_id} is not a certified preprocessed target: "
                + "; ".join(failures)
            )
        target_identity = {
            "format": PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA,
            "scan_id": scan_id,
            "target_sidecar_sha256": canonical_sha256(provenance),
            "target_output_sha256": output_digest,
            "functional_validity_mask_path": str(
                functional_validity_path.resolve(strict=True)
            ),
            "functional_validity_mask_sha256": functional_validity["SHA256"],
            "functional_validity_mask_size_bytes": functional_validity["SizeBytes"],
            "functional_validity_mask_contract": TARGET_VALIDITY_MASK_CONTRACT,
        }
        target_identity["fingerprint_sha256"] = canonical_sha256(target_identity)
        return target_identity

    def _validate_native_padded_target(self, scan_id: str) -> dict:
        """Authenticate one allowlisted recovery target before NIfTI admission."""
        if scan_id not in self.target_scan_ids:
            raise RuntimeError(
                "native target validation attempted outside the target allowlist"
            )
        role = self.target_scan_roles.get(scan_id)
        if role not in {"train", "development-validation"}:
            raise RuntimeError(
                "native target validation requires an explicit train/development role"
            )
        structural_identity = self.source_dataset.source_fingerprint(scan_id).get(
            "structural_source_identity"
        )
        if not isinstance(structural_identity, Mapping):
            raise RuntimeError(
                f"{scan_id} has no authenticated native structural binding"
            )
        authority = self.native_target_authority
        try:
            return validate_native_padded_target(
                self._target_directory(scan_id),
                scan_id=scan_id,
                expected_role=role,
                expected_structural_binding=structural_identity,
                structural_batch_path=Path(self.native_alignment_authority_path),
                structural_batch_sha256=str(self.native_alignment_authority_sha256),
                selection_manifest_path=Path(authority["selection_manifest_path"]),
                selection_manifest_sha256=str(authority["selection_manifest_sha256"]),
                selection_root_review_path=Path(
                    authority["selection_root_review_path"]
                ),
                selection_root_review_sha256=str(
                    authority["selection_root_review_sha256"]
                ),
                completed_set_path=Path(authority["completed_set_path"]),
                completed_set_sha256=str(authority["completed_set_sha256"]),
                completed_set_commit_marker_path=Path(
                    authority["completed_set_commit_marker_path"]
                ),
                completed_set_commit_marker_sha256=str(
                    authority["completed_set_commit_marker_sha256"]
                ),
                reviewed_native_source_path=Path(authority["reviewed_source_path"]),
                reviewed_native_source_sha256=str(authority["reviewed_source_sha256"]),
                runtime_attester_sha256=str(authority["runtime_attester_sha256"]),
                native_verifier_sha256=str(authority["native_verifier_sha256"]),
            )
        except (NativeTargetError, SourceAcquisitionError) as exc:
            raise RuntimeError(
                f"{scan_id} native recovery target admission failed: {exc}"
            ) from exc

    def _scaler_fingerprint(self) -> Dict:
        if self.scaler_dir is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "files_sha256": directory_file_sha256(self.scaler_dir),
        }

    def _modernbert_fingerprint(self, recorded: object) -> Dict:
        if not isinstance(recorded, dict):
            raise RuntimeError("Clinical-ModernBERT source provenance is missing")
        if recorded.get("implementation") != "Clinical-ModernBERT":
            raise RuntimeError("text cache does not identify Clinical-ModernBERT")
        if recorded.get("upstream_model_id") != "Simonlee711/Clinical_ModernBERT":
            raise RuntimeError(
                "text cache does not identify the required upstream model"
            )
        if recorded.get("model_name") != self.modernbert_model_name:
            raise RuntimeError(
                "configured Clinical-ModernBERT model differs from the cache source"
            )
        revision = str(recorded.get("revision", "")).strip().lower()
        if (
            len(revision) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in revision)
            or self.modernbert_revision != revision
        ):
            raise RuntimeError(
                "models.modernbert_revision must equal the immutable revision "
                f"used for preprocessing ({revision})"
            )
        local_path = Path(self.modernbert_model_name).expanduser()
        if local_path.is_dir():
            current_hashes = directory_file_sha256(local_path)
            if (
                not current_hashes
                or recorded.get("local_files_sha256") != current_hashes
            ):
                raise RuntimeError("local Clinical-ModernBERT files changed")
        else:
            if self.modernbert_model_name != "Simonlee711/Clinical_ModernBERT":
                raise RuntimeError("remote text cache uses the wrong model ID")
        without_digest = dict(recorded)
        recorded_digest = without_digest.pop("fingerprint_sha256", None)
        if recorded_digest != canonical_sha256(without_digest):
            raise RuntimeError("Clinical-ModernBERT fingerprint digest is invalid")
        return dict(recorded)

    def _validate_brainiac_fingerprint(self, recorded: object) -> None:
        if not isinstance(recorded, dict):
            raise RuntimeError("BrainIAC source provenance is missing")
        source_files = recorded.get("source_files_sha256")
        adapter = recorded.get("adapter")
        expected_adapter = {
            "input_channels": 1,
            "resize_shape": [96, 96, 96],
            "resize_mode": "trilinear",
            "align_corners": False,
            "output": "CLS token embedding",
            "embedding_dim": self.expected_node_dims["image"],
        }
        wrapper_path = (
            Path(__file__).resolve().parents[1] / "models" / "brainiac_wrapper.py"
        )
        if (
            recorded.get("implementation") != "BrainIAC"
            or recorded.get("checkpoint_sha256")
            != sha256_file(self.brainiac_model_path)
            or recorded.get("checkpoint_sha256") != self.brainiac_checkpoint_sha256
            or not isinstance(source_files, dict)
            or not source_files
            or source_files.get("connect4/models/brainiac_wrapper.py")
            != sha256_file(wrapper_path)
            or "brainiac/load_brainiac.py" not in source_files
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for name, digest in source_files.items()
            )
            or recorded.get("source_files_fingerprint_sha256")
            != canonical_sha256(source_files)
            or recorded.get("source_files_fingerprint_sha256")
            != self.brainiac_source_sha256
            or adapter != expected_adapter
        ):
            raise RuntimeError("BrainIAC checkpoint/source/adapter provenance differs")

    def _validate_cache_provenance(self, scan_id: str) -> None:
        graphs = self.precomputed_dir / "graphs"
        hypergraphs = self.precomputed_dir / "hypergraphs"
        metadata_path = hypergraphs / f"{scan_id}_metadata.json"
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        if metadata.get("scan_id") != scan_id:
            raise RuntimeError(f"{scan_id} cache metadata identifies another scan")
        if metadata.get("spatial_shape") != list(self.target_shape):
            raise RuntimeError(f"{scan_id} cache has the wrong spatial shape")
        if metadata.get("patch_size") != list(self.patch_size):
            raise RuntimeError(f"{scan_id} cache has the wrong structural patch size")
        if metadata.get("num_rois") != self.NUM_ROIS:
            raise RuntimeError(f"{scan_id} cache has the wrong ROI count")
        if (
            metadata.get("patch_description_schema_version")
            != PATCH_DESCRIPTION_SCHEMA_VERSION
        ):
            raise RuntimeError(f"{scan_id} cache uses the wrong patch-text schema")
        if metadata.get("patch_coordinate_contract") != PATCH_COORDINATE_CONTRACT:
            raise RuntimeError(
                f"{scan_id} cache uses the wrong patch-coordinate contract"
            )
        if metadata.get("patch_text_components") != self._TEXT_COMPONENTS:
            raise RuntimeError(f"{scan_id} cache omits a patch-text component")
        normative_subject = self.source_dataset.scan_to_normative_subject[scan_id]
        expected_normative_hash = canonical_sha256(
            {
                "workbook_sha256": self.source_dataset.normative_workbook_sha256,
                "subject_rows": self.source_dataset.normative_source_rows[
                    normative_subject
                ],
            }
        )
        if metadata.get("normative_context_sha256") != expected_normative_hash:
            raise RuntimeError(f"{scan_id} normative context digest differs")
        if metadata.get("potvin_supported_roi_ids") != sorted(
            self.source_dataset.POTVIN_ROI_IDS
        ):
            raise RuntimeError(f"{scan_id} cache misstates Potvin ROI coverage")

        recorded_sources = metadata.get("source_fingerprint")
        if not isinstance(recorded_sources, dict):
            raise RuntimeError(f"{scan_id} source fingerprint is missing")
        expected_dataset = self.source_dataset.source_fingerprint(scan_id)
        if recorded_sources.get("dataset") != expected_dataset:
            raise RuntimeError(f"{scan_id} graph cache is stale for its source data")
        expected_geometry = expected_dataset.get("structural_grid_geometry")
        if metadata.get("structural_grid_geometry") != expected_geometry:
            raise RuntimeError(f"{scan_id} cache structural-grid geometry differs")
        recorded_external = recorded_sources.get("external_protocol")
        if self.external_protocol_context is None:
            if recorded_external is not None:
                raise RuntimeError(
                    f"{scan_id} external graph cache cannot be used as paired training data"
                )
        elif recorded_external != self.external_protocol_context:
            raise RuntimeError(
                f"{scan_id} cache external-cohort/checkpoint identity differs"
            )
        self._validate_brainiac_fingerprint(recorded_sources.get("brainiac"))
        self._modernbert_fingerprint(recorded_sources.get("modernbert"))
        if recorded_sources.get("scalers") != self._scaler_fingerprint():
            raise RuntimeError(f"{scan_id} feature-scaler provenance differs")
        expected_fixed = {
            "target_shape": list(self.target_shape),
            "patch_size": list(self.patch_size),
            "patch_description_schema_version": PATCH_DESCRIPTION_SCHEMA_VERSION,
            "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
            "functional_connectivity_sha256": canonical_sha256(
                KNOWN_FUNCTIONAL_CONNECTIVITY
            ),
        }
        for key, value in expected_fixed.items():
            if recorded_sources.get(key) != value:
                raise RuntimeError(f"{scan_id} source fingerprint field {key} differs")
        if metadata.get("source_fingerprint_sha256") != canonical_sha256(
            recorded_sources
        ):
            raise RuntimeError(f"{scan_id} source fingerprint digest is invalid")
        if self.external_protocol_context is not None:
            if conditioning_identity_from_cache_sources(recorded_sources) != (
                self.external_protocol_context.get("conditioning_identity")
            ):
                raise RuntimeError(
                    f"{scan_id} conditioning artifacts differ from the "
                    "synthesis checkpoint"
                )

        artifact_paths = [
            graphs / f"{scan_id}_image_nodes.npy",
            self.precomputed_dir
            / "image_emb"
            / f"{scan_id}_image_patch_embeddings.npy",
            self.precomputed_dir
            / "modernbert_emb"
            / f"{scan_id}_mask_patch_embeddings.npy",
            graphs / f"{scan_id}_mask_nodes.npy",
            graphs / f"{scan_id}_roi_nodes.npy",
            hypergraphs / f"{scan_id}_hyperedge_index.npy",
            hypergraphs / f"{scan_id}_hyperedge_weights.npy",
            hypergraphs / f"{scan_id}_patch_distributions.json",
            hypergraphs / f"{scan_id}_patch_descriptions.json",
        ]
        expected_artifacts = {path.name: sha256_file(path) for path in artifact_paths}
        if metadata.get("artifact_sha256") != expected_artifacts:
            raise RuntimeError(f"{scan_id} cached artifact hash mismatch")

        descriptions_path = hypergraphs / f"{scan_id}_patch_descriptions.json"
        with descriptions_path.open() as stream:
            descriptions = json.load(stream)
        expected_patches = int(
            np.prod(
                [
                    size // patch
                    for size, patch in zip(self.target_shape, self.patch_size)
                ]
            )
        )
        if (
            not isinstance(descriptions, list)
            or len(descriptions) != expected_patches
            or any(
                not isinstance(text, str) or not text.strip() for text in descriptions
            )
        ):
            raise RuntimeError(f"{scan_id} raw patch-description cache is invalid")

        segmentation, _ = self.source_dataset._load_nifti(
            self._segmentation_path(scan_id), is_mask=True
        )
        current_distributions = [
            self.source_dataset._compute_patch_distribution(segmentation, index)
            for index in range(expected_patches)
        ]
        with (hypergraphs / f"{scan_id}_patch_distributions.json").open() as stream:
            cached_raw = json.load(stream)
        cached_distributions = [
            {int(key): float(value) for key, value in distribution.items()}
            for distribution in cached_raw
        ]
        if cached_distributions != current_distributions:
            raise RuntimeError(
                f"{scan_id} cached ROI coverage no longer matches the segmentation"
            )
        affine = nib.load(str(self._segmentation_path(scan_id))).affine
        expected_descriptions = [
            build_patch_description(
                patch_idx=patch_index,
                center_mm=self.source_dataset._compute_patch_center_mm(
                    patch_index,
                    segmentation=segmentation,
                    affine=affine,
                ),
                distribution=distribution,
                id_to_slug=self.source_dataset.ID_TO_SLUG,
                normative_index=self.source_dataset.normative_index,
                patient_id=normative_subject,
            )
            for patch_index, distribution in enumerate(current_distributions)
        ]
        if descriptions != expected_descriptions:
            raise RuntimeError(
                f"{scan_id} cached patch text differs from the versioned "
                "functional-connectivity/Potvin description"
            )
        if len(current_distributions) != metadata.get("num_patches"):
            raise RuntimeError(f"{scan_id} cache patch count is invalid")

        from graphs.hypergraph import HypergraphBuilder

        builder = HypergraphBuilder(expected_patches, self.NUM_ROIS)
        edge_index, edge_weights = builder.build_hyperedges(
            current_distributions,
            self.structure_to_roi_idx,
            torch.device("cpu"),
        )
        saved_index = torch.from_numpy(
            np.load(hypergraphs / f"{scan_id}_hyperedge_index.npy", allow_pickle=False)
        ).long()
        saved_weights = torch.from_numpy(
            np.load(
                hypergraphs / f"{scan_id}_hyperedge_weights.npy", allow_pickle=False
            )
        ).float()
        if not torch.equal(saved_index, edge_index.cpu()) or not torch.allclose(
            saved_weights, edge_weights.cpu(), rtol=0.0, atol=1e-7
        ):
            raise RuntimeError(f"{scan_id} saved hypergraph differs from ROI coverage")
        self._validated_cache_metadata_sha256[scan_id] = canonical_sha256(metadata)

    def _validate_patch_distributions(
        self,
        scan_id: str,
        patch_distributions: List[Dict[int, float]],
        expected_patches: int,
    ) -> None:
        """Validate absolute mapped-ROI fractions without renormalising them."""
        if len(patch_distributions) != expected_patches:
            raise ValueError(
                f"{scan_id} has {len(patch_distributions)} distributions but "
                f"{expected_patches} image patches"
            )
        total_mapped = 0.0
        for patch_index, distribution in enumerate(patch_distributions):
            if not isinstance(distribution, dict):
                raise TypeError(
                    f"{scan_id} patch {patch_index} distribution is not a dict"
                )
            unknown = set(distribution) - set(self.structure_to_roi_idx)
            if unknown:
                raise ValueError(
                    f"{scan_id} patch {patch_index} contains unmapped cached labels: "
                    f"{sorted(unknown)}"
                )
            values = np.asarray(list(distribution.values()), dtype=np.float64)
            if values.size and (
                not np.isfinite(values).all()
                or (values < 0.0).any()
                or (values > 1.0 + 1e-4).any()
            ):
                raise ValueError(
                    f"Invalid ROI coverage in {scan_id} patch {patch_index}"
                )
            patch_total = float(values.sum()) if values.size else 0.0
            if patch_total > 1.0 + 1e-4:
                raise ValueError(
                    f"ROI coverages for {scan_id} patch {patch_index} sum to "
                    f"{patch_total}, which exceeds the full patch fraction 1"
                )
            total_mapped += patch_total
        if total_mapped <= 0.0:
            raise ValueError(f"{scan_id} has no mapped foreground ROI coverage")

    @staticmethod
    def _load_nodes(path: Path, label: str) -> torch.Tensor:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(f"Cannot load {label} node cache: {path}") from exc
        nodes = torch.from_numpy(np.asarray(array)).float()
        if nodes.ndim != 2:
            raise ValueError(f"{label} node cache must be a matrix, got {nodes.shape}")
        if not torch.isfinite(nodes).all():
            raise ValueError(f"{label} node cache contains NaN or infinity")
        return nodes

    def __getitem__(self, index: int) -> Dict:
        self._require_canonical_roi_contract()
        if self.structure_to_roi_idx != self.ROI_LABEL_TO_CHANNEL:
            raise RuntimeError("dataset ROI mapping was mutated after admission")
        scan_id = self.scan_ids[index]
        t1_path = self._t1_path(scan_id)
        segmentation_path = self._segmentation_path(scan_id)
        admitted = isinstance(
            getattr(self, "_admitted_sample_artifacts", None), Mapping
        )
        if admitted:
            t1_image = self._nifti_from_admitted_payload(
                self._admitted_payload(scan_id, "t1w", t1_path), t1_path
            )
            segmentation_image = self._nifti_from_admitted_payload(
                self._admitted_payload(scan_id, "segmentation", segmentation_path),
                segmentation_path,
            )
        else:
            t1_image = nib.load(str(t1_path))
            segmentation_image = nib.load(str(segmentation_path))
        t1_array = t1_image.get_fdata(dtype=np.float32)
        label_array = segmentation_image.get_fdata(dtype=np.float32)
        arrays = [("T1w", t1_array), ("segmentation", label_array)]
        fmri_image = None
        fmri_array = None
        if scan_id in self.target_scan_ids:
            fmri_image = self._load_target_image(scan_id)
            fmri_array = fmri_image.get_fdata(dtype=np.float32)
            arrays.append(("fMRI", fmri_array))
        for label, array in arrays:
            if not np.isfinite(array).all():
                raise ValueError(f"{scan_id} {label} contains NaN or infinity")
        if (
            t1_array.shape != self.target_shape
            or label_array.shape != self.target_shape
        ):
            raise ValueError(f"{scan_id} T1w/segmentation left the certified grid")
        if fmri_array is not None and fmri_array.shape != (
            *self.target_shape,
            self.num_frames,
        ):
            raise ValueError(f"{scan_id} fMRI left the certified grid")
        rounded_labels = np.rint(label_array)
        if not np.allclose(label_array, rounded_labels, rtol=0.0, atol=1e-4):
            raise ValueError(f"{scan_id} segmentation contains non-integer labels")
        labels = torch.from_numpy(rounded_labels.astype(np.int64, copy=False))
        if not self.allow_synthetic_grid_override:
            raise RuntimeError(
                "V10 production dataset materialization is blocked until a fresh "
                "Stage-B v2 authority supplies an independently authenticated "
                "structural brain mask; deriving brain_mask from segmentation "
                "labels is forbidden"
            )
        # Synthetic-grid unit fixtures may use segmentation support as an explicit
        # test mask. This branch is never admissible to production training/eval.
        brain_mask = (labels > 0).float().unsqueeze(0)
        if not bool(brain_mask.any()):
            raise ValueError(f"{scan_id} segmentation/brain mask is empty")
        roi_masks = {
            roi_index: (labels == label_id).float()
            for label_id, roi_index in self.structure_to_roi_idx.items()
        }

        graphs = self.precomputed_dir / "graphs"
        node_paths = {
            "image": graphs / f"{scan_id}_image_nodes.npy",
            "mask": graphs / f"{scan_id}_mask_nodes.npy",
            "ROI": graphs / f"{scan_id}_roi_nodes.npy",
        }
        if admitted:
            loaded_nodes = {}
            for display_name, path in node_paths.items():
                authority_name = f"{display_name.lower()}_nodes"
                payload = self._admitted_payload(scan_id, authority_name, path)
                try:
                    array = np.load(io.BytesIO(payload), allow_pickle=False)
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot decode admitted {display_name} node cache: {path}"
                    ) from exc
                nodes = torch.from_numpy(np.asarray(array)).float()
                if nodes.ndim != 2 or not torch.isfinite(nodes).all():
                    raise ValueError(
                        f"admitted {display_name} node cache is invalid: {nodes.shape}"
                    )
                loaded_nodes[display_name] = nodes
            image_nodes = loaded_nodes["image"]
            mask_nodes = loaded_nodes["mask"]
            roi_nodes = loaded_nodes["ROI"]
        else:
            image_nodes = self._load_nodes(node_paths["image"], "image")
            mask_nodes = self._load_nodes(node_paths["mask"], "mask")
            roi_nodes = self._load_nodes(node_paths["ROI"], "ROI")
        for label, nodes in (
            ("image", image_nodes),
            ("mask", mask_nodes),
            ("ROI", roi_nodes),
        ):
            if nodes.shape[1] != self.expected_node_dims[label]:
                raise ValueError(
                    f"{scan_id} {label} node dimension is {nodes.shape[1]}, "
                    f"expected {self.expected_node_dims[label]}"
                )
        if image_nodes.shape[0] != mask_nodes.shape[0]:
            raise ValueError(f"{scan_id} image/mask patch counts differ")
        if roi_nodes.shape[0] != self.NUM_ROIS:
            raise ValueError(
                f"{scan_id} has {roi_nodes.shape[0]} ROI nodes, expected {self.NUM_ROIS}"
            )

        hypergraphs = self.precomputed_dir / "hypergraphs"
        metadata_path = hypergraphs / f"{scan_id}_metadata.json"
        distributions_path = hypergraphs / f"{scan_id}_patch_distributions.json"
        if admitted:
            try:
                metadata = json.loads(
                    self._admitted_payload(scan_id, "metadata", metadata_path).decode(
                        "utf-8"
                    )
                )
                raw_distributions = json.loads(
                    self._admitted_payload(
                        scan_id, "patch_distributions", distributions_path
                    ).decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{scan_id} admitted cache JSON cannot be decoded"
                ) from exc
        else:
            with metadata_path.open() as stream:
                metadata = json.load(stream)
            with distributions_path.open() as stream:
                raw_distributions = json.load(stream)
        brainlm_record = getattr(self, "_brainlm_context_records", {}).get(scan_id)
        if brainlm_record is not None:
            structural = self._validate_structural_source_identity(
                metadata.get("source_fingerprint", {})
                .get("dataset", {})
                .get("structural_source_identity"),
                scan_id=scan_id,
            )
            observed_support_sha = self._brainlm_support_sha256(brain_mask[0])
            observed_support_count = int(brain_mask.sum().item())
            if (
                structural["fingerprint_sha256"]
                != brainlm_record["structural_source_identity_sha256"]
                or structural["native_outputs"]["t1w"]["sha256"]
                != brainlm_record["prepared_t1_sha256"]
                or structural["native_outputs"]["segmentation"]["sha256"]
                != brainlm_record["prepared_mask_sha256"]
                or observed_support_sha != brainlm_record["support_tensor_sha256"]
                or observed_support_count != brainlm_record["support_foreground_voxels"]
                or not np.allclose(
                    t1_image.affine,
                    np.asarray(
                        brainlm_record["padded_affine_ras_mm"], dtype=np.float64
                    ),
                    rtol=0.0,
                    atol=1e-6,
                )
            ):
                raise RuntimeError(
                    f"{scan_id} admitted BrainLM context changed at materialization"
                )
        if (
            metadata.get("patch_description_schema_version")
            != PATCH_DESCRIPTION_SCHEMA_VERSION
        ):
            raise RuntimeError(f"{scan_id} mask-text cache uses the wrong schema")
        if metadata.get("patch_text_components") != self._TEXT_COMPONENTS:
            raise RuntimeError(f"{scan_id} patch text is missing a required component")
        normative_hash = str(metadata.get("normative_context_sha256", ""))
        if len(normative_hash) != 64 or any(
            char not in "0123456789abcdef" for char in normative_hash
        ):
            raise RuntimeError(f"{scan_id} normative-context hash is invalid")
        if int(metadata.get("num_patches", -1)) != image_nodes.shape[0]:
            raise ValueError(f"{scan_id} metadata patch count disagrees with caches")
        if not isinstance(raw_distributions, list):
            raise TypeError(f"{scan_id} patch distributions must be a JSON list")
        patch_distributions = [
            {int(key): float(value) for key, value in distribution.items()}
            for distribution in raw_distributions
        ]
        self._validate_patch_distributions(
            scan_id, patch_distributions, int(image_nodes.shape[0])
        )

        from graphs.hypergraph import HypergraphBuilder

        hypergraph_builder = HypergraphBuilder(
            num_patches=int(image_nodes.shape[0]), num_rois=self.NUM_ROIS
        )
        hyperedge_index, hyperedge_weights = hypergraph_builder.build_hyperedges(
            patch_distributions, self.structure_to_roi_idx, torch.device("cpu")
        )
        if hyperedge_weights.numel() == 0:
            raise ValueError(f"{scan_id} has no ROI-coverage hyperedges")

        t1w = torch.from_numpy(t1_array.copy()).unsqueeze(0)
        result = {
            "t1w": t1w,
            "brain_mask": brain_mask,
            "t1w_affine": torch.from_numpy(t1_image.affine.copy()).float(),
            "roi_masks": roi_masks,
            "image_patch_embeddings": image_nodes,
            "mask_patch_embeddings": mask_nodes,
            "roi_embeddings": roi_nodes,
            "patch_distributions": patch_distributions,
            "structure_to_roi_idx": self.structure_to_roi_idx,
            "roi_label_ids": self.ROI_LABEL_IDS,
            "roi_mapping_sha256": self.ROI_MAPPING_SHA256,
            "dwi_matrix": self.dwi_matrix,
            "scan_id": scan_id,
            "scan_role": self.scan_roles.get(scan_id, "sealed-test"),
            "image_nodes": image_nodes,
            "mask_nodes": mask_nodes,
            "roi_nodes": roi_nodes,
            "hyperedge_index": hyperedge_index,
            "hyperedge_weights": hyperedge_weights,
        }
        if brainlm_record is not None:
            result.update(
                {
                    "brainlm_context_schema": brainlm_record["schema"],
                    "brainlm_role": brainlm_record["role"],
                    "brainlm_native_preprocessing_source_sha256": (
                        brainlm_record["native_preprocessing_source_sha256"]
                    ),
                    "brainlm_prepared_t1_sha256": brainlm_record["prepared_t1_sha256"],
                    "brainlm_prepared_mask_sha256": brainlm_record[
                        "prepared_mask_sha256"
                    ],
                    "brainlm_padded_t1_artifact_descriptor_sha256": (
                        brainlm_record["padded_t1_artifact_descriptor_sha256"]
                    ),
                    "brainlm_padded_mask_artifact_descriptor_sha256": (
                        brainlm_record["padded_mask_artifact_descriptor_sha256"]
                    ),
                    "brainlm_cache_metadata_artifact_descriptor_sha256": (
                        brainlm_record["cache_metadata_artifact_descriptor_sha256"]
                    ),
                    "brainlm_structural_source_identity_sha256": (
                        brainlm_record["structural_source_identity_sha256"]
                    ),
                    "brainlm_native_alignment_authority_sha256": (
                        brainlm_record["native_alignment_authority_sha256"]
                    ),
                    "brainlm_target_artifact_identity_sha256": (
                        brainlm_record["target_artifact_identity_sha256"]
                    ),
                    "brainlm_padded_shape": torch.tensor(
                        brainlm_record["padded_shape"], dtype=torch.int64
                    ),
                    "brainlm_native_shape": torch.tensor(
                        brainlm_record["native_shape"], dtype=torch.int64
                    ),
                    "brainlm_padding_before": torch.tensor(
                        brainlm_record["padding_before"], dtype=torch.int64
                    ),
                    "brainlm_padding_after": torch.tensor(
                        brainlm_record["padding_after"], dtype=torch.int64
                    ),
                    "brainlm_padded_affine_ras_mm": torch.from_numpy(
                        np.asarray(
                            brainlm_record["padded_affine_ras_mm"],
                            dtype=np.float64,
                        )
                    ),
                    "brainlm_support_tensor_sha256": brainlm_record[
                        "support_tensor_sha256"
                    ],
                    "brainlm_support_foreground_voxels": torch.tensor(
                        brainlm_record["support_foreground_voxels"],
                        dtype=torch.int64,
                    ),
                    "brainlm_scan_context_record_sha256": brainlm_record[
                        "record_sha256"
                    ],
                    "brainlm_dataset_state_record_sha256": (
                        self._rank_zero_admission_state_sha256
                    ),
                    "brainlm_dataset_state_sha256": (
                        self._rank_zero_admission_payload_sha256
                    ),
                }
            )
        if fmri_array is not None:
            fmri = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(fmri_array, -1, 0))
            ).unsqueeze(0)
            expected_validity = target_validity_mask_from_fmri(
                fmri_array,
                brain_mask[0].numpy(),
                scan_id=scan_id,
            )
            if self.recovery_profile:
                identity = self.target_artifact_identities.get(scan_id)
                if (
                    not isinstance(identity, Mapping)
                    or identity.get("paper_certified") is not False
                    or identity.get("certification_status")
                    != NATIVE_CERTIFICATION_STATUS
                ):
                    raise RuntimeError(
                        f"{scan_id} recovery validity derivation lacks its "
                        "explicit non-certified provenance contract"
                    )
                validity_values = expected_validity
                validity_contract = RECOVERY_TARGET_VALIDITY_MASK_CONTRACT
            else:
                validity_image = self._load_target_validity_mask_image(scan_id)
                validity_values = validity_image.get_fdata(dtype=np.float32)
                if (
                    validity_image.ndim != 3
                    or tuple(validity_values.shape) != self.target_shape
                    or not np.allclose(
                        validity_image.affine,
                        fmri_image.affine,
                        rtol=0.0,
                        atol=1e-6,
                    )
                    or not np.isfinite(validity_values).all()
                    or not np.logical_or(
                        validity_values == 0.0, validity_values == 1.0
                    ).all()
                    or not np.array_equal(validity_values, expected_validity)
                ):
                    raise RuntimeError(
                        f"{scan_id} admitted functional-validity artifact differs"
                    )
                validity_contract = TARGET_VALIDITY_MASK_CONTRACT
            target_validity_mask = torch.from_numpy(
                np.ascontiguousarray(validity_values, dtype=np.float32)
            ).unsqueeze(0)
            result["fmri"] = fmri
            result["fmri_mean"] = fmri.mean(dim=1)
            result["target_validity_mask"] = target_validity_mask
            result["target_validity_mask_contract"] = validity_contract
        return result


__all__ = [
    "Connect4PrecomputedDataset",
    "RECOVERY_TARGET_VALIDITY_MASK_CONTRACT",
    "TARGET_VALIDITY_MASK_CONTRACT",
    "discover_structural_cache_scan_ids",
    "target_validity_mask_from_fmri",
]
