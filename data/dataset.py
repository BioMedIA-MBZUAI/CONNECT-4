"""Strict structural-feature dataset used only for offline graph preprocessing.

This loader never reads or modifies fMRI. It consumes already aligned T1w and
SynthSeg volumes, subject-specific Potvin inputs, subject-specific radiomics,
and provenance-bound AnatCL embeddings. The training loader is
``Connect4PrecomputedDataset``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.patch_descriptions import (
    build_normative_index,
)
from data.patchify import Patchify3D
from data.provenance import canonical_sha256, sha256_file
from data.spatial_contract import (
    PATCH_COORDINATE_CONTRACT,
    geometric_patch_center_voxel,
    nifti_geometry_fingerprint,
    patch_grid_xyz,
)
from normative.atrophy import ASEG_TO_NORM_LABEL, AtrophyDescriber
from preprocessing.preprocess_seg import SEG_PREPROCESSING_SCHEMA_VERSION
from preprocessing.preprocess_t1 import (
    T1_INTENSITY_NORMALIZATION as PRODUCTION_T1_INTENSITY_NORMALIZATION,
    T1_PREPROCESSING_SCHEMA_VERSION,
)
from preprocessing.conform import zscore_inbrain
from preprocessing.native_structural_identity import (
    IDENTICALLY_PADDED_MODALITIES,
    NATIVE_SEG_PREPROCESSING_SCHEMA,
    NATIVE_T1_PREPROCESSING_SCHEMA,
    load_native_structural_alignment_binding,
)
from preprocessing.source_acquisition_identity import (
    SourceAcquisitionError,
    load_structural_source_binding,
    snapshot_binary_artifact,
)
from utils.scalers import FeatureScalerManager


class Connect4Dataset(Dataset):
    """Load paper inputs for deterministic, offline node-feature extraction."""

    ROI_SPECS: List[Tuple[str, int]] = [
        ("left_cerebral_white_matter", 2),
        ("left_cerebral_cortex", 3),
        ("left_lateral_ventricle", 4),
        ("left_inferior_lateral_ventricle", 5),
        ("left_cerebellum_white_matter", 7),
        ("left_cerebellum_cortex", 8),
        ("left_thalamus", 10),
        ("left_caudate", 11),
        ("left_putamen", 12),
        ("left_pallidum", 13),
        ("third_ventricle", 14),
        ("fourth_ventricle", 15),
        ("brain_stem", 16),
        ("left_hippocampus", 17),
        ("left_amygdala", 18),
        ("csf", 24),
        ("left_accumbens_area", 26),
        ("left_ventral_dc", 28),
        ("right_cerebral_white_matter", 41),
        ("right_cerebral_cortex", 42),
        ("right_lateral_ventricle", 43),
        ("right_inferior_lateral_ventricle", 44),
        ("right_cerebellum_white_matter", 46),
        ("right_cerebellum_cortex", 47),
        ("right_thalamus", 49),
        ("right_caudate", 50),
        ("right_putamen", 51),
        ("right_pallidum", 52),
        ("right_hippocampus", 53),
        ("right_amygdala", 54),
        ("right_accumbens_area", 58),
        ("right_ventral_dc", 60),
    ]
    NUM_ROIS = len(ROI_SPECS)
    SLUG_TO_ID = dict(ROI_SPECS)
    ID_TO_SLUG = {label_id: slug for slug, label_id in ROI_SPECS}
    ROI_LABEL_TO_INDEX = {
        0: -1,
        **{label_id: index for index, (_, label_id) in enumerate(ROI_SPECS)},
    }
    POTVIN_ROI_IDS = frozenset(ASEG_TO_NORM_LABEL)
    ANATCL_PROVENANCE_SCHEMA = "connect4-anatcl-roi-features-v2"
    CAT12_INPUT_PROVENANCE_SCHEMA = "connect4-cat12-anatcl-input-v1"
    RADIOMICS_PROVENANCE_SCHEMA = "connect4-pyradiomics-roi-features-v2"
    ROI_FEATURE_CONDITIONING_SCHEMA = "connect4_roi_feature_conditioning_v3"
    ROI_FEATURE_ARTIFACT_SCHEMA = "connect4_roi_feature_artifact_identity_v1"
    ANATCL_EMBEDDING_DIM = 512
    ANATCL_MODEL_CONTRACT = {
        "package": "anatcl",
        "package_version": "0.0.2",
        "source_revision": "62e344627fe2f5685054802d7e6c6c98b572c205",
        "source_files_sha256": {
            "__init__.py": (
                "5f3e8e4dacdd60eeb81d3cbacf060d0a8580169c6cd0223213c52e547f71c9b0"
            ),
            "anatcl.py": (
                "505dd542452ec91db74effc8d9333ae0347f618669dacf6827952d09c3cf9633"
            ),
            "models/__init__.py": (
                "b95e008fab63fd31d74b0c0ea7d2879b8df55b7982b330502ac34a715bbdcce6"
            ),
            "models/resnet3d.py": (
                "bca448a5e9e9e4ec5f86ecb5e5077640aa50b90721959701472c0068de1a6bf5"
            ),
        },
        "source_tree_sha256": (
            "ed74362e12f574d71f7c7aba638a89f4c09fe15de36663f8e523125fd71ac638"
        ),
        "architecture": "resnet18",
        "descriptor": "global",
        "fold": 0,
        "use_head": False,
        "pretrained": True,
        "weights_sha256": (
            "77f6d44d0317ffc704ffc751db7eab349a8109e2cd596ab06070f1aec7a55d64"
        ),
        "state_dict_sha256": (
            "ef695e5926cb5a09867d8bbd61b781ed873b50571bc058ad746a78b95ff00639"
        ),
    }
    CAT12_PREPROCESSING_CONTRACT = {
        "input_kind": "raw T1w",
        "output_kind": "CAT12 modulated normalized gray-matter VBM (mwp1/mwp1u)",
        "voxel_size_mm": [1.5, 1.5, 1.5],
        "source_shape": [121, 145, 121],
        "brainprep_source_revision": ("afdd3c10be6540d671c1764b1cb4133a4858ee27"),
        "container_image": "docker://neurospin/brainprep-anat:v1",
        "container_manifest_sha256": (
            "f999dfb4f6fa0c043a2f624314c5598765d2cb1e5ecc86c2874cfa53a44cd88d"
        ),
        "cat12_version": "12.8.2",
        "cat12_revision": "r2166",
        "spm12_revision": "r7771",
        "matlab_compiler_runtime": "R2017b",
    }
    ANATCL_EXTRACTION_CONTRACT = {
        "input": "ROI-masked CAT12 mwp1/mwp1u VBM",
        "roi_mask": (
            "SynthSeg label resampled by the CAT12 authority onto the exact VBM grid; "
            "voxels outside one ROI are set to zero"
        ),
        "source_shape": [121, 145, 121],
        "center_crop": {
            "shape": [121, 128, 121],
            "slices": [[0, 121], [8, 136], [0, 121]],
        },
        "zero_pad": {
            "shape": [128, 128, 128],
            "before": [3, 0, 3],
            "after": [4, 0, 4],
        },
        "tensor_shape": [1, 1, 128, 128, 128],
        "normalization": {"mean": 0.0, "std": 1.0},
        "aggregation": "none; one 512-D global descriptor per ROI",
    }
    PYRADIOMICS_EXTRACTOR_CONTRACT = {
        "package": "PyRadiomics",
        "distribution": "pyradiomics",
        "package_version": "3.0.1",
        "upstream_revision": "08bea7067e350303eead533b471aa60697b3b8c3",
        "source_distribution_sha256": (
            "47c57f441d6cb7973fa3b2ea48d3948df78e3348e1c69e1e2ff19001601fc2f5"
        ),
        "configuration": "RadiomicsFeatureExtractor defaults",
    }
    T1_PREPROCESSING_SCHEMA = T1_PREPROCESSING_SCHEMA_VERSION
    SEG_PREPROCESSING_SCHEMA = SEG_PREPROCESSING_SCHEMA_VERSION
    T1_PREPROCESSING_IDENTITY_SCHEMA = (
        "connect4_structural_preprocessing_artifact_identity_v2"
    )
    T1_INTENSITY_NORMALIZATION = PRODUCTION_T1_INTENSITY_NORMALIZATION

    _NORMATIVE_COLUMNS = {
        "PatientID",
        "StructureID",
        "MeasuredVolume",
        "Age",
        "Sex",
        "FieldStrength",
        "Manufacturer",
        "ICV",
    }

    def __init__(
        self,
        root_dir: str,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        target_shape: Optional[Tuple[int, int, int]] = None,
        dwi_matrix_path: str = "/path/to/data/dwi_matrix.csv",
        normative_csv_path: str = "/path/to/data/patient_roi_normative_inputs.csv",
        cohort_manifest_path: Optional[str] = None,
        scaler_dir: Optional[str] = None,
        common_grid_contract_sha256: Optional[str] = None,
        native_alignment_authority_path: Optional[str] = None,
        native_alignment_authority_sha256: Optional[str] = None,
        structural_stage_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root_dir)
        self.patch_size = tuple(int(value) for value in patch_size)
        if target_shape is None:
            raise ValueError(
                "target_shape must come from a validated common-grid contract; "
                "there is no manuscript-derived spatial-matrix default"
            )
        self.target_shape = tuple(int(value) for value in target_shape)
        self.common_grid_contract_sha256 = (
            str(common_grid_contract_sha256).strip().lower()
            if common_grid_contract_sha256 is not None
            else None
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
        self.structural_stage_root = (
            Path(structural_stage_root) if structural_stage_root is not None else None
        )
        for label, digest in (
            ("common-grid contract", self.common_grid_contract_sha256),
            ("native-alignment authority", self.native_alignment_authority_sha256),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} SHA-256 is invalid")
        if (
            self.common_grid_contract_sha256 is not None
            and self.native_alignment_authority_sha256 is not None
        ):
            raise ValueError(
                "paper-profile common-grid and non-certified native-alignment "
                "authorities are mutually exclusive"
            )
        native_fields = (
            self.native_alignment_authority_path,
            self.native_alignment_authority_sha256,
            self.structural_stage_root,
        )
        if any(value is not None for value in native_fields) and not all(
            value is not None for value in native_fields
        ):
            raise ValueError(
                "native recovery requires structural_stage_root and both the "
                "native-alignment authority path and SHA-256"
            )
        if (
            self.common_grid_contract_sha256 is not None
            and self.structural_stage_root is not None
        ):
            raise ValueError(
                "paper common-grid loading cannot use a native Stage-B structural root"
            )
        for path, label, directory in (
            (
                self.native_alignment_authority_path,
                "native-alignment authority",
                False,
            ),
            (self.structural_stage_root, "native Stage-B structural root", True),
        ):
            if path is None:
                continue
            if not path.is_absolute() or path != Path(os.path.abspath(path)):
                raise ValueError(f"{label} path must be absolute and canonical")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise FileNotFoundError(f"{label} path is missing: {path}") from exc
            if resolved != path or (directory and not path.is_dir()):
                raise ValueError(f"{label} path aliases another path or has wrong type")
        if len(self.patch_size) != 3 or any(value < 1 for value in self.patch_size):
            raise ValueError("patch_size must have three positive entries")
        if len(self.target_shape) != 3 or any(value < 1 for value in self.target_shape):
            raise ValueError("target_shape must have three positive entries")
        if any(size % patch for size, patch in zip(self.target_shape, self.patch_size)):
            raise ValueError("target_shape must be divisible by patch_size")
        self.patchifier = Patchify3D(self.patch_size)
        self.structure_to_roi_idx = {
            label_id: index for index, (_, label_id) in enumerate(self.ROI_SPECS)
        }

        self.scan_ids = self._discover_scan_ids()
        if not self.scan_ids:
            raise RuntimeError(f"No T1w scans found under {self._t1_cohort_root()}")
        self._validate_structural_cohort()

        self.dwi_matrix_path = Path(dwi_matrix_path)
        self.dwi_matrix = self._load_dwi_matrix(self.dwi_matrix_path)
        self.cohort_manifest_path = (
            Path(cohort_manifest_path).expanduser()
            if cohort_manifest_path is not None
            else None
        )
        self.patient_id_by_scan = self._load_manifest_patient_mapping(
            self.cohort_manifest_path
        )
        self.normative_csv_path = Path(normative_csv_path)
        (
            self.normative_descriptions,
            self.normative_source_rows,
            self.scan_to_normative_subject,
            self.normative_workbook_sha256,
        ) = self._load_normative_descriptions(self.normative_csv_path)
        self.normative_index = build_normative_index(self.normative_descriptions)

        self.radiomics_csv_path = self.root / "all_radiomics.csv"
        self.radiomics_provenance_path = self.root / "radiomics_provenance.json"
        (
            self.radiomics_by_subject,
            self.scan_to_radiomics_subject,
            self.radiomics_feature_names,
        ) = self._load_radiomics_table(self.radiomics_csv_path)
        self._validate_radiomics_provenance(self.radiomics_provenance_path)
        self._roi_feature_identity_cache: Dict[str, dict] = {}
        self._roi_feature_artifact_identity_cache: Dict[str, dict] = {}

        self.scaler_manager: Optional[FeatureScalerManager] = None
        if scaler_dir is not None:
            scaler_path = Path(scaler_dir)
            if not scaler_path.is_dir():
                raise FileNotFoundError(f"scaler directory not found: {scaler_path}")
            self.scaler_manager = FeatureScalerManager(scaler_path)
            self.scaler_manager.load_scalers(scaler_path)
            required = {"radiomics", "anatcl"}
            if set(self.scaler_manager.scalers) != required:
                raise RuntimeError(
                    f"scaler directory must contain exactly {sorted(required)} scalers"
                )
        self.anatcl_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _load_manifest_patient_mapping(
        self, path: Optional[Path]
    ) -> Optional[Dict[str, str]]:
        """Load an explicit scan-to-patient mapping for normative evidence only.

        Filename prefixes are never interpreted as patient identities.  A
        patient-level Potvin table is accepted only when this manifest binds
        each discovered scan to one exact patient ID.
        """
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"cohort manifest not found: {path}")
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid JSON cohort manifest: {path}") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("format") != "connect4_scan_cohorts_v1"
                or not isinstance(payload.get("scans"), dict)
            ):
                raise ValueError(
                    "JSON cohort manifest must use connect4_scan_cohorts_v1 "
                    "with a scans mapping"
                )
            raw_rows = [
                (
                    scan_id,
                    record.get("patient_id") if isinstance(record, dict) else None,
                )
                for scan_id, record in payload["scans"].items()
            ]
        else:
            try:
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    reader = csv.DictReader(stream)
                    columns = {
                        str(name).strip().lower(): name
                        for name in (reader.fieldnames or [])
                    }
                    if not {"scan_id", "patient_id"}.issubset(columns):
                        raise ValueError(
                            "cohort manifest requires scan_id and patient_id columns"
                        )
                    raw_rows = [
                        (
                            row.get(columns["scan_id"]),
                            row.get(columns["patient_id"]),
                        )
                        for row in reader
                    ]
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"invalid CSV cohort manifest: {path}") from exc
        mapping: Dict[str, str] = {}
        for row_index, (raw_scan, raw_patient) in enumerate(raw_rows):
            scan_id = str(raw_scan or "").strip()
            patient_id = str(raw_patient or "").strip()
            if not scan_id or not patient_id:
                raise ValueError(
                    f"cohort manifest row {row_index} has an empty scan/patient ID"
                )
            if scan_id in mapping:
                raise ValueError(f"cohort manifest duplicates scan_id {scan_id!r}")
            mapping[scan_id] = patient_id
        expected = set(self.scan_ids)
        if set(mapping) != expected:
            raise ValueError(
                "cohort manifest scan coverage differs from structural scans: "
                f"missing={sorted(expected - set(mapping))[:10]}, "
                f"extra={sorted(set(mapping) - expected)[:10]}"
            )
        return mapping

    def _t1_cohort_root(self) -> Path:
        return self.structural_stage_root or self.root / "T1"

    def _t1_path(self, scan_id: str) -> Path:
        if self.structural_stage_root is not None:
            return self.structural_stage_root / scan_id / f"{scan_id}_T1.nii.gz"
        return self.root / "T1" / f"{scan_id}_T1.nii.gz"

    def _segmentation_path(self, scan_id: str) -> Path:
        if self.structural_stage_root is not None:
            return self.structural_stage_root / scan_id / f"{scan_id}_mask.nii.gz"
        return self.root / "Masks" / f"{scan_id}_mask.nii.gz"

    def _t1_sidecar_path(self, scan_id: str) -> Path:
        if self.structural_stage_root is not None:
            return self.structural_stage_root / scan_id / f"{scan_id}_T1.json"
        return self.root / "T1" / f"{scan_id}_T1.json"

    def _segmentation_sidecar_path(self, scan_id: str) -> Path:
        if self.structural_stage_root is not None:
            return self.structural_stage_root / scan_id / f"{scan_id}_mask.json"
        return self.root / "Masks" / f"{scan_id}_mask.json"

    def _discover_scan_ids(self) -> List[str]:
        if self.structural_stage_root is None:
            t1_paths = list((self.root / "T1").glob("*_T1.nii.gz"))
            t1_paths.extend((self.root / "T1").glob("*_T1.nii"))
        else:
            t1_paths = list(self.structural_stage_root.glob("*/*_T1.nii.gz"))
        ids = set()
        for path in t1_paths:
            suffix = "_T1.nii.gz" if path.name.endswith("_T1.nii.gz") else "_T1.nii"
            scan_id = path.name.removesuffix(suffix)
            if self.structural_stage_root is not None and path.parent.name != scan_id:
                raise RuntimeError(
                    "native Stage-B structural file is outside its exact per-scan "
                    f"directory: {path}"
                )
            ids.add(scan_id)
        return sorted(ids)

    def _validate_structural_cohort(self) -> None:
        failures = []
        for scan_id in self.scan_ids:
            t1_path = self._t1_path(scan_id)
            mask_path = self._segmentation_path(scan_id)
            try:
                t1 = nib.load(str(t1_path))
                mask = nib.load(str(mask_path))
                if tuple(t1.shape) != self.target_shape:
                    failures.append(f"{scan_id}: T1 shape {t1.shape}")
                if tuple(mask.shape) != self.target_shape:
                    failures.append(f"{scan_id}: mask shape {mask.shape}")
                if not np.allclose(t1.affine, mask.affine, rtol=0.0, atol=1e-3):
                    failures.append(f"{scan_id}: T1/mask affine mismatch")
                if not np.allclose(t1.header.get_zooms()[:3], (3.0, 3.0, 3.0)):
                    failures.append(f"{scan_id}: T1 is not 3 mm isotropic")
                if not np.allclose(mask.header.get_zooms()[:3], (3.0, 3.0, 3.0)):
                    failures.append(f"{scan_id}: mask is not 3 mm isotropic")
                if (
                    self.common_grid_contract_sha256 is not None
                    or self.native_alignment_authority_sha256 is not None
                ):
                    self._validate_production_t1_artifact(
                        scan_id,
                        t1_image=t1,
                        segmentation_image=mask,
                    )
            except Exception as exc:
                failures.append(f"{scan_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "structural preprocessing contract failed: " + "; ".join(failures[:10])
            )

    def _validate_production_t1_artifact(
        self,
        scan_id: str,
        *,
        t1_image: Optional[nib.spatialimages.SpatialImage] = None,
        segmentation_image: Optional[nib.spatialimages.SpatialImage] = None,
    ) -> Dict[str, object]:
        """Validate and identify one production T1 artifact without caching.

        A configured common-grid digest is the production boundary.  At that
        boundary, a NIfTI with plausible geometry is insufficient: the exact
        v4 T1 and v3 segmentation sidecars must bind the bytes and declare the recovery
        normalization used to create them.  Both derivative sidecars and both
        output NIfTIs are read once
        through descriptor-held, no-follow snapshots.  The descriptors and
        paths must remain unchanged until validation finishes, so the parsed
        sidecar, hashed NIfTIs, decoded arrays, and recorded geometry cannot be
        assembled from different versions of an artifact.
        """
        if (
            self.common_grid_contract_sha256 is None
            and self.native_alignment_authority_sha256 is None
        ):
            raise RuntimeError(
                "production T1 validation requires exactly one spatial-authority pin"
            )

        t1_path = self._t1_path(scan_id)
        segmentation_path = self._segmentation_path(scan_id)
        sidecar_path = self._t1_sidecar_path(scan_id)
        segmentation_sidecar_path = self._segmentation_sidecar_path(scan_id)
        # These compatibility parameters used to carry already-opened images.
        # Production validation intentionally ignores them because their bytes
        # were not acquired under this method's stable snapshot boundary.
        _ = t1_image, segmentation_image
        with (
            self._stable_regular_file_snapshot(
                sidecar_path, f"{scan_id} production T1 sidecar"
            ) as sidecar_snapshot,
            self._stable_regular_file_snapshot(
                segmentation_sidecar_path,
                f"{scan_id} production segmentation sidecar",
            ) as segmentation_sidecar_snapshot,
            self._stable_regular_file_snapshot(
                t1_path, f"{scan_id} production T1 NIfTI"
            ) as t1_snapshot,
            self._stable_regular_file_snapshot(
                segmentation_path, f"{scan_id} 32-ROI segmentation"
            ) as segmentation_snapshot,
        ):
            try:
                sidecar_bytes = sidecar_snapshot["bytes"]
                if not isinstance(sidecar_bytes, bytes):
                    raise TypeError("sidecar snapshot is not bytes")
                payload = json.loads(sidecar_bytes.decode("utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"{scan_id} production T1 sidecar is invalid: {sidecar_path}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"{scan_id} production T1 sidecar is not an object")
            try:
                segmentation_sidecar_bytes = segmentation_sidecar_snapshot["bytes"]
                if not isinstance(segmentation_sidecar_bytes, bytes):
                    raise TypeError("segmentation sidecar snapshot is not bytes")
                segmentation_payload = json.loads(
                    segmentation_sidecar_bytes.decode("utf-8")
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{scan_id} production segmentation sidecar is invalid: "
                    f"{segmentation_sidecar_path}"
                ) from exc
            if not isinstance(segmentation_payload, dict):
                raise RuntimeError(
                    f"{scan_id} production segmentation sidecar is not an object"
                )

            if self.native_alignment_authority_sha256 is not None:
                return self._validate_native_structural_artifact(
                    scan_id=scan_id,
                    t1_payload=payload,
                    segmentation_payload=segmentation_payload,
                    t1_sidecar_snapshot=sidecar_snapshot,
                    segmentation_sidecar_snapshot=segmentation_sidecar_snapshot,
                    t1_snapshot=t1_snapshot,
                    segmentation_snapshot=segmentation_snapshot,
                )

            failures = []
            if payload.get("schema") != self.T1_PREPROCESSING_SCHEMA:
                failures.append(f"schema is not {self.T1_PREPROCESSING_SCHEMA}")
            if segmentation_payload.get("schema") != self.SEG_PREPROCESSING_SCHEMA:
                failures.append(
                    f"segmentation schema is not {self.SEG_PREPROCESSING_SCHEMA}"
                )
            if payload.get("scan_id") != scan_id:
                failures.append("T1 sidecar identifies another scan")
            if segmentation_payload.get("scan_id") != scan_id:
                failures.append("segmentation sidecar identifies another scan")
            recorded_output_sha256 = payload.get("output_sha256")
            if not self._valid_sha256(recorded_output_sha256):
                failures.append("output_sha256 is invalid")
            elif recorded_output_sha256 != t1_snapshot["sha256"]:
                failures.append("output_sha256 does not bind the T1 NIfTI")
            recorded_segmentation_sha256 = segmentation_payload.get("output_sha256")
            if not self._valid_sha256(recorded_segmentation_sha256):
                failures.append("segmentation output_sha256 is invalid")
            elif recorded_segmentation_sha256 != segmentation_snapshot["sha256"]:
                failures.append(
                    "segmentation output_sha256 does not bind the segmentation NIfTI"
                )
            if payload.get("segmentation_output_sha256") != (
                recorded_segmentation_sha256
            ):
                failures.append(
                    "T1 sidecar does not bind the conformed segmentation bytes"
                )
            if (
                payload.get("segmentation_sidecar_sha256")
                != (segmentation_sidecar_snapshot["sha256"])
            ):
                failures.append(
                    "T1 sidecar does not bind the segmentation sidecar bytes"
                )
            if (
                payload.get("common_grid_contract_sha256")
                != self.common_grid_contract_sha256
            ):
                failures.append("common-grid SHA-256 differs from the configured pin")
            if payload.get("architecture_shape") != list(self.target_shape):
                failures.append("architecture_shape differs from the configured shape")
            if (
                segmentation_payload.get("common_grid_contract_sha256")
                != self.common_grid_contract_sha256
            ):
                failures.append(
                    "segmentation common-grid SHA-256 differs from the configured pin"
                )
            if segmentation_payload.get("architecture_shape") != list(
                self.target_shape
            ):
                failures.append(
                    "segmentation architecture_shape differs from the configured shape"
                )
            recovery = payload.get("versioned_recovery_choices")
            if not isinstance(recovery, dict):
                failures.append("versioned_recovery_choices is missing")
            elif (
                recovery.get("intensity_normalization")
                != self.T1_INTENSITY_NORMALIZATION
            ):
                failures.append(
                    "intensity normalization is not the declared authenticated-"
                    "segmentation-support in-brain z-score with exact-zero exterior"
                )

            structural_source = None
            try:
                t1_source, _ = load_structural_source_binding(
                    payload.get("structural_source"),
                    expected_scan_id=scan_id,
                )
                segmentation_source, _ = load_structural_source_binding(
                    segmentation_payload.get("structural_source"),
                    expected_scan_id=scan_id,
                )
                if t1_source != segmentation_source:
                    failures.append(
                        "T1 and segmentation structural-source identities differ"
                    )
                else:
                    structural_source = t1_source
                    if (
                        payload.get("source_sha256")
                        != structural_source["raw_t1_sha256"]
                    ):
                        failures.append(
                            "T1 source_sha256 differs from authenticated raw T1w"
                        )
                    if (
                        segmentation_payload.get("source_sha256")
                        != (structural_source["synthseg_mask_sha256"])
                    ):
                        failures.append(
                            "segmentation source_sha256 differs from authenticated "
                            "SynthSeg mask"
                        )
            except SourceAcquisitionError as exc:
                failures.append(
                    "target-blind structural source identity cannot be authenticated: "
                    f"{exc}"
                )

            try:
                t1 = self._nifti_from_snapshot(t1_snapshot, "T1")
                segmentation = self._nifti_from_snapshot(
                    segmentation_snapshot, "32-ROI segmentation"
                )
                if tuple(t1.shape) != self.target_shape:
                    failures.append(f"T1 shape {t1.shape}")
                if tuple(segmentation.shape) != self.target_shape:
                    failures.append(f"segmentation shape {segmentation.shape}")
                if not np.allclose(t1.affine, segmentation.affine, rtol=0.0, atol=1e-3):
                    failures.append("T1/segmentation affine mismatch")
                if not np.allclose(t1.header.get_zooms()[:3], (3.0, 3.0, 3.0)):
                    failures.append("T1 is not 3 mm isotropic")
                if not np.allclose(
                    segmentation.header.get_zooms()[:3], (3.0, 3.0, 3.0)
                ):
                    failures.append("segmentation is not 3 mm isotropic")
                if (
                    tuple(t1.shape) == self.target_shape
                    and tuple(segmentation.shape) == self.target_shape
                ):
                    t1_values = np.asarray(t1.dataobj, dtype=np.float32)
                    segmentation_values = np.asarray(
                        segmentation.dataobj, dtype=np.float32
                    )
                    if not np.isfinite(t1_values).all():
                        failures.append("T1 contains NaN or infinity")
                    elif np.any(t1_values < -3.0) or np.any(t1_values > 3.0):
                        failures.append("T1 lies outside the required [-3,3] range")
                    if not np.isfinite(segmentation_values).all():
                        failures.append("32-ROI segmentation contains NaN or infinity")
                    else:
                        if not np.allclose(
                            segmentation_values,
                            np.rint(segmentation_values),
                            rtol=0.0,
                            atol=1e-6,
                        ):
                            failures.append(
                                "32-ROI segmentation contains non-integer labels"
                            )
                        segmentation_support = segmentation_values > 0
                        if not np.any(segmentation_support):
                            failures.append("32-ROI segmentation support is empty")
                        if np.any(t1_values[~segmentation_support] != 0.0):
                            failures.append(
                                "T1 exterior is not exactly zero outside the 32-ROI "
                                "segmentation support"
                            )
            except Exception as exc:
                failures.append(
                    f"T1/segmentation snapshot values cannot be validated: {exc}"
                )

            if failures:
                raise RuntimeError(
                    f"{scan_id} production T1 preprocessing contract failed: "
                    + "; ".join(failures)
                )

            structural_geometry = {
                "t1w": self._nifti_geometry_from_image(t1),
                "segmentation": self._nifti_geometry_from_image(segmentation),
            }
            identity: Dict[str, object] = {
                "format": self.T1_PREPROCESSING_IDENTITY_SCHEMA,
                "t1_sidecar_sha256": sidecar_snapshot["sha256"],
                "t1_sidecar_canonical_sha256": canonical_sha256(payload),
                "segmentation_sidecar_sha256": segmentation_sidecar_snapshot["sha256"],
                "segmentation_sidecar_canonical_sha256": canonical_sha256(
                    segmentation_payload
                ),
                "schema": payload["schema"],
                "segmentation_schema": segmentation_payload["schema"],
                "scan_id": scan_id,
                "output_sha256": recorded_output_sha256,
                "segmentation_sha256": recorded_segmentation_sha256,
                "structural_source_identity": structural_source,
                "common_grid_contract_sha256": payload["common_grid_contract_sha256"],
                "architecture_shape": list(payload["architecture_shape"]),
                "intensity_normalization": recovery["intensity_normalization"],
                "structural_grid_geometry": structural_geometry,
            }
            identity["fingerprint_sha256"] = canonical_sha256(identity)
            return identity

    def _validate_native_structural_artifact(
        self,
        *,
        scan_id: str,
        t1_payload: Dict[str, object],
        segmentation_payload: Dict[str, object],
        t1_sidecar_snapshot: Dict[str, object],
        segmentation_sidecar_snapshot: Dict[str, object],
        t1_snapshot: Dict[str, object],
        segmentation_snapshot: Dict[str, object],
    ) -> Dict[str, object]:
        """Admit the explicitly non-certified per-scan native padding profile."""
        authority_sha256 = self.native_alignment_authority_sha256
        if authority_sha256 is None:
            raise RuntimeError("native alignment authority is not configured")
        failures: List[str] = []
        if t1_payload.get("schema") != NATIVE_T1_PREPROCESSING_SCHEMA:
            failures.append(f"schema is not {NATIVE_T1_PREPROCESSING_SCHEMA}")
        if segmentation_payload.get("schema") != NATIVE_SEG_PREPROCESSING_SCHEMA:
            failures.append(
                f"segmentation schema is not {NATIVE_SEG_PREPROCESSING_SCHEMA}"
            )
        if t1_payload.get("scan_id") != scan_id:
            failures.append("T1 sidecar identifies another scan")
        if segmentation_payload.get("scan_id") != scan_id:
            failures.append("segmentation sidecar identifies another scan")
        if (
            t1_payload.get("paper_certified") is not False
            or segmentation_payload.get("paper_certified") is not False
        ):
            failures.append("native recovery sidecars claim paper certification")
        if (
            t1_payload.get("native_alignment_authority_sha256") != authority_sha256
            or segmentation_payload.get("native_alignment_authority_sha256")
            != authority_sha256
        ):
            failures.append("native-alignment authority SHA-256 differs")
        recorded_t1_sha256 = t1_payload.get("output_sha256")
        recorded_segmentation_sha256 = segmentation_payload.get("output_sha256")
        if (
            not self._valid_sha256(recorded_t1_sha256)
            or recorded_t1_sha256 != t1_snapshot["sha256"]
        ):
            failures.append("native-profile T1 output hash differs")
        if (
            not self._valid_sha256(recorded_segmentation_sha256)
            or recorded_segmentation_sha256 != segmentation_snapshot["sha256"]
        ):
            failures.append("native-profile segmentation output hash differs")
        if t1_payload.get("segmentation_output_sha256") != (
            recorded_segmentation_sha256
        ):
            failures.append("T1 sidecar does not bind the padded segmentation")
        if (
            t1_payload.get("segmentation_sidecar_sha256")
            != (segmentation_sidecar_snapshot["sha256"])
        ):
            failures.append("T1 sidecar does not bind the segmentation sidecar")

        native_alignment = None
        try:
            t1_alignment = load_native_structural_alignment_binding(
                t1_payload.get("native_alignment"),
                expected_scan_id=scan_id,
                expected_batch_sha256=authority_sha256,
                expected_batch_path=self.native_alignment_authority_path,
            )
            segmentation_alignment = load_native_structural_alignment_binding(
                segmentation_payload.get("native_alignment"),
                expected_scan_id=scan_id,
                expected_batch_sha256=authority_sha256,
                expected_batch_path=self.native_alignment_authority_path,
            )
            if t1_alignment != segmentation_alignment:
                failures.append("T1/segmentation native alignment identities differ")
            else:
                native_alignment = t1_alignment
        except SourceAcquisitionError as exc:
            failures.append(
                f"native structural alignment cannot be authenticated: {exc}"
            )

        if native_alignment is not None:
            if list(self.target_shape) != native_alignment["architecture_shape"]:
                failures.append("configured shape differs from native padded shape")
            for payload, label in (
                (t1_payload, "T1"),
                (segmentation_payload, "segmentation"),
            ):
                if (
                    payload.get("architecture_shape")
                    != native_alignment["architecture_shape"]
                ):
                    failures.append(f"{label} architecture shape differs")
                if payload.get("anatomical_shape") != native_alignment["native_shape"]:
                    failures.append(f"{label} native anatomical shape differs")
                if payload.get("architecture_padding") != {
                    "before": native_alignment["padding_before"],
                    "after": native_alignment["padding_after"],
                    "mode": "constant-zero-no-interpolation",
                    "identically_applied_to": list(IDENTICALLY_PADDED_MODALITIES),
                }:
                    failures.append(f"{label} native padding declaration differs")
            if (
                t1_payload.get("source_sha256")
                != native_alignment["native_outputs"]["t1w"]["sha256"]
            ):
                failures.append("T1 source hash differs from native preprocessing")
            if (
                segmentation_payload.get("source_sha256")
                != native_alignment["native_outputs"]["segmentation"]["sha256"]
            ):
                failures.append(
                    "segmentation source hash differs from native preprocessing"
                )
        recovery = t1_payload.get("versioned_recovery_choices")
        if (
            not isinstance(recovery, dict)
            or recovery.get("intensity_normalization")
            != self.T1_INTENSITY_NORMALIZATION
            or recovery.get("paper_spatial_matrix_claimed") is not False
        ):
            failures.append("native T1 normalization/certification declaration differs")

        try:
            t1_image = self._nifti_from_snapshot(t1_snapshot, "padded native T1")
            segmentation_image = self._nifti_from_snapshot(
                segmentation_snapshot, "padded native segmentation"
            )
            t1_values = np.asarray(t1_image.dataobj, dtype=np.float32)
            segmentation_values = np.asarray(
                segmentation_image.dataobj, dtype=np.float32
            )
            if (
                tuple(t1_image.shape) != self.target_shape
                or tuple(segmentation_image.shape) != self.target_shape
            ):
                failures.append("native structural outputs have the wrong padded shape")
            if not np.allclose(
                t1_image.affine,
                segmentation_image.affine,
                rtol=0.0,
                atol=1e-6,
            ):
                failures.append("native padded T1/segmentation affines differ")
            if not np.isfinite(t1_values).all():
                failures.append("native padded T1 contains NaN or infinity")
            if not np.isfinite(segmentation_values).all() or not np.allclose(
                segmentation_values,
                np.rint(segmentation_values),
                rtol=0.0,
                atol=1e-6,
            ):
                failures.append("native padded segmentation is not a finite label map")

            if native_alignment is not None:
                expected_affine = np.asarray(
                    native_alignment["padded_affine"], dtype=np.float64
                )
                if not np.allclose(
                    t1_image.affine, expected_affine, rtol=0.0, atol=1e-6
                ):
                    failures.append("native padded affine differs")
                before = tuple(
                    int(value) for value in native_alignment["padding_before"]
                )
                native_shape = tuple(
                    int(value) for value in native_alignment["native_shape"]
                )
                crop = tuple(
                    slice(start, start + size)
                    for start, size in zip(before, native_shape)
                )
                padding_mask = np.ones(self.target_shape, dtype=bool)
                padding_mask[crop] = False
                if np.any(t1_values[padding_mask] != 0.0) or np.any(
                    segmentation_values[padding_mask] != 0.0
                ):
                    failures.append("native architecture padding is not exactly zero")
                native_t1_artifact = native_alignment["native_outputs"]["t1w"]
                native_segmentation_artifact = native_alignment["native_outputs"][
                    "segmentation"
                ]
                native_t1_bytes, _ = snapshot_binary_artifact(
                    Path(str(native_t1_artifact["path"])),
                    expected_sha256=str(native_t1_artifact["sha256"]),
                    label="native unpadded T1",
                )
                native_segmentation_bytes, _ = snapshot_binary_artifact(
                    Path(str(native_segmentation_artifact["path"])),
                    expected_sha256=str(native_segmentation_artifact["sha256"]),
                    label="native unpadded segmentation",
                )
                native_t1 = self._nifti_from_snapshot(
                    {
                        "bytes": native_t1_bytes,
                        "path": native_t1_artifact["path"],
                    },
                    "native unpadded T1",
                )
                native_segmentation = self._nifti_from_snapshot(
                    {
                        "bytes": native_segmentation_bytes,
                        "path": native_segmentation_artifact["path"],
                    },
                    "native unpadded segmentation",
                )
                native_t1_values = np.asarray(native_t1.dataobj, dtype=np.float32)
                native_segmentation_values = np.asarray(
                    native_segmentation.dataobj, dtype=np.float32
                )
                native_affine = np.asarray(
                    native_alignment["native_affine"], dtype=np.float64
                )
                if (
                    tuple(native_t1.shape) != native_shape
                    or tuple(native_segmentation.shape) != native_shape
                    or not np.allclose(
                        native_t1.affine, native_affine, rtol=0.0, atol=1e-6
                    )
                    or not np.allclose(
                        native_segmentation.affine,
                        native_affine,
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    failures.append("native unpadded structural geometry differs")
                elif not np.array_equal(
                    segmentation_values[crop], native_segmentation_values
                ):
                    failures.append(
                        "padded segmentation crop differs from native bytes"
                    )
                else:
                    support = native_segmentation_values > 0
                    expected_t1 = zscore_inbrain(native_t1_values, support=support)
                    if not np.allclose(
                        t1_values[crop], expected_t1, rtol=0.0, atol=1e-6
                    ):
                        failures.append(
                            "padded T1 is not the authenticated mask-supported "
                            "normalization of native T1"
                        )
            segmentation_support = segmentation_values > 0
            if not np.any(segmentation_support):
                failures.append("native padded segmentation support is empty")
            if np.any(t1_values[~segmentation_support] != 0.0):
                failures.append("native padded T1 exterior is not exactly zero")
        except Exception as exc:
            failures.append(f"native structural values cannot be validated: {exc}")

        if failures:
            raise RuntimeError(
                f"{scan_id} native structural preprocessing contract failed: "
                + "; ".join(failures)
            )
        geometry = {
            "t1w": self._nifti_geometry_from_image(t1_image),
            "segmentation": self._nifti_geometry_from_image(segmentation_image),
        }
        identity: Dict[str, object] = {
            "format": self.T1_PREPROCESSING_IDENTITY_SCHEMA,
            "spatial_profile": native_alignment["spatial_profile"],
            "paper_certified": False,
            "scan_id": scan_id,
            "t1_sidecar_sha256": t1_sidecar_snapshot["sha256"],
            "t1_sidecar_canonical_sha256": canonical_sha256(t1_payload),
            "segmentation_sidecar_sha256": segmentation_sidecar_snapshot["sha256"],
            "segmentation_sidecar_canonical_sha256": canonical_sha256(
                segmentation_payload
            ),
            "schema": t1_payload["schema"],
            "segmentation_schema": segmentation_payload["schema"],
            "output_sha256": recorded_t1_sha256,
            "segmentation_sha256": recorded_segmentation_sha256,
            "structural_source_identity": native_alignment,
            "native_alignment_authority_sha256": authority_sha256,
            "architecture_shape": list(self.target_shape),
            "intensity_normalization": recovery["intensity_normalization"],
            "structural_grid_geometry": geometry,
        }
        identity["fingerprint_sha256"] = canonical_sha256(identity)
        return identity

    @staticmethod
    def _stat_signature(metadata: os.stat_result) -> Tuple[int, ...]:
        """Return mutation-sensitive metadata for one descriptor/path binding."""
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )

    @staticmethod
    @contextmanager
    def _stable_regular_file_snapshot(
        path: Path, label: str
    ) -> Iterator[Dict[str, object]]:
        """Yield exact bytes while holding a stable, unique regular-file descriptor."""

        def checked_path_stat() -> os.stat_result:
            try:
                metadata = os.lstat(path)
            except FileNotFoundError as exc:
                raise RuntimeError(f"{label} is missing: {path}") from exc
            except OSError as exc:
                raise RuntimeError(f"{label} cannot be inspected: {path}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"{label} must not be a symbolic link: {path}")
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"{label} is not a regular file: {path}")
            if metadata.st_nlink != 1:
                raise RuntimeError(f"{label} must not have hard links: {path}")
            return metadata

        path_before = checked_path_stat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(
                f"{label} cannot be opened without following links: {path}"
            ) from exc
        try:
            descriptor_before = os.fstat(descriptor)
            expected = Connect4Dataset._stat_signature(descriptor_before)
            if expected != Connect4Dataset._stat_signature(path_before):
                raise RuntimeError(
                    f"{label} changed during production admission: {path}"
                )
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise RuntimeError(f"{label} is not a regular file: {path}")
            if descriptor_before.st_nlink != 1:
                raise RuntimeError(f"{label} must not have hard links: {path}")

            chunks = []
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)

            descriptor_after_read = os.fstat(descriptor)
            path_after_read = checked_path_stat()
            if (
                Connect4Dataset._stat_signature(descriptor_after_read) != expected
                or Connect4Dataset._stat_signature(path_after_read) != expected
            ):
                raise RuntimeError(
                    f"{label} changed during production admission: {path}"
                )
            snapshot: Dict[str, object] = {
                "path": str(path),
                "bytes": b"".join(chunks),
                "sha256": digest.hexdigest(),
            }
            try:
                yield snapshot
            finally:
                descriptor_final = os.fstat(descriptor)
                path_final = checked_path_stat()
                if (
                    Connect4Dataset._stat_signature(descriptor_final) != expected
                    or Connect4Dataset._stat_signature(path_final) != expected
                ):
                    raise RuntimeError(
                        f"{label} changed during production admission: {path}"
                    )
        finally:
            os.close(descriptor)

    @staticmethod
    def _nifti_from_snapshot(
        snapshot: Dict[str, object], label: str
    ) -> nib.Nifti1Image:
        """Decode a NIfTI solely from bytes captured by the stable snapshot."""
        content = snapshot.get("bytes")
        path = str(snapshot.get("path", ""))
        if not isinstance(content, bytes):
            raise RuntimeError(f"{label} snapshot contains no bytes")
        try:
            nifti_bytes = gzip.decompress(content) if path.endswith(".gz") else content
            return nib.Nifti1Image.from_bytes(nifti_bytes)
        except Exception as exc:
            raise RuntimeError(f"{label} snapshot is not a valid NIfTI") from exc

    @staticmethod
    def _nifti_geometry_from_image(image: nib.spatialimages.SpatialImage) -> dict:
        """Fingerprint geometry without reopening a validated snapshot path."""
        canonical = nib.as_closest_canonical(image)
        if canonical.ndim != 3:
            raise ValueError(f"structural source must be 3D, got {canonical.shape}")
        if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
            raise ValueError("canonical structural source is not RAS+")
        shape = tuple(int(value) for value in canonical.shape)
        edge_indices = np.asarray(
            list(product(*[(-0.5, float(size) - 0.5) for size in shape])),
            dtype=np.float64,
        )
        edge_world = nib.affines.apply_affine(canonical.affine, edge_indices)
        return {
            "shape_xyz": list(shape),
            "affine": np.asarray(canonical.affine, dtype=np.float64).tolist(),
            "voxel_sizes_mm_xyz": [
                float(value) for value in nib.affines.voxel_sizes(canonical.affine)
            ],
            "axis_codes": ["R", "A", "S"],
            "voxel_edge_bounds_mm": {
                "minimum": edge_world.min(axis=0).tolist(),
                "maximum": edge_world.max(axis=0).tolist(),
            },
        }

    def _load_dwi_matrix(self, path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"DWI matrix not found: {path}")
        array = np.loadtxt(path, delimiter=",", dtype=np.float32)
        matrix = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if matrix.shape != (self.NUM_ROIS, self.NUM_ROIS):
            raise ValueError(f"DWI matrix must be {self.NUM_ROIS}x{self.NUM_ROIS}")
        if not torch.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("DWI matrix must contain finite non-negative coefficients")
        return matrix

    def _load_normative_descriptions(
        self, path: Path
    ) -> Tuple[Dict[Tuple[str, str], str], Dict[str, list], Dict[str, str], str]:
        if not path.is_file():
            raise FileNotFoundError(f"Potvin input CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = self._NORMATIVE_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                "Potvin input CSV must contain raw measured volume and all model "
                f"covariates; missing={sorted(missing)}"
            )
        if frame.empty:
            raise ValueError("Potvin input CSV is empty")

        describer = AtrophyDescriber()
        workbook_sha = sha256_file(describer.norms.workbook_path)
        descriptions: Dict[Tuple[str, str], str] = {}
        source_rows: Dict[str, list] = {}
        seen: set[Tuple[str, int]] = set()
        for row_index, row in frame.iterrows():
            patient = str(row["PatientID"]).strip()
            if not patient or patient == "nan":
                raise ValueError(f"normative row {row_index} has invalid PatientID")
            try:
                structure_id = int(row["StructureID"])
                measured = float(row["MeasuredVolume"])
                age = float(row["Age"])
                sex = int(row["Sex"])
                field = int(row["FieldStrength"])
                manufacturer = int(row["Manufacturer"])
                icv = float(row["ICV"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"normative row {row_index} has invalid values"
                ) from exc
            if structure_id not in self.POTVIN_ROI_IDS:
                raise ValueError(
                    f"normative row {row_index} uses ROI {structure_id}, which has "
                    "no individual Potvin workbook model"
                )
            numeric = np.asarray((measured, age, icv), dtype=np.float64)
            if not np.isfinite(numeric).all() or measured <= 0 or icv <= 0:
                raise ValueError(
                    f"normative row {row_index} has non-finite/non-positive data"
                )
            key = (patient, structure_id)
            if key in seen:
                raise ValueError(
                    f"duplicate normative row for {patient}, ROI {structure_id}"
                )
            seen.add(key)
            norm_label = ASEG_TO_NORM_LABEL[structure_id]
            description = describer.describe_roi(
                norm_label,
                measured,
                age,
                sex,
                field,
                manufacturer,
                icv,
            )
            slug = self.ID_TO_SLUG[structure_id]
            descriptions[(patient, slug)] = description
            source_rows.setdefault(patient, []).append(
                {
                    "structure_id": structure_id,
                    "measured_volume": measured,
                    "age": age,
                    "sex": sex,
                    "field_strength": field,
                    "manufacturer": manufacturer,
                    "icv": icv,
                    "norm_label": norm_label,
                }
            )

        available = set(source_rows)
        scan_mapping: Dict[str, str] = {}
        for scan_id in self.scan_ids:
            if scan_id in available:
                scan_mapping[scan_id] = scan_id
                continue
            if self.patient_id_by_scan is not None:
                patient_id = self.patient_id_by_scan[scan_id]
                if patient_id in available:
                    scan_mapping[scan_id] = patient_id
                    continue
            raise ValueError(
                f"Potvin input {path} has no exact row for scan {scan_id!r}. "
                "Patient-level rows require an explicit cohort manifest mapping; "
                "filename-prefix inference is forbidden."
            )
        for scan_id, subject in scan_mapping.items():
            ids = {row["structure_id"] for row in source_rows[subject]}
            if ids != self.POTVIN_ROI_IDS:
                raise ValueError(
                    f"{scan_id} must provide all {len(self.POTVIN_ROI_IDS)} "
                    f"individual Potvin ROIs; missing={sorted(self.POTVIN_ROI_IDS - ids)}"
                )
            source_rows[subject].sort(key=lambda item: item["structure_id"])
        return descriptions, source_rows, scan_mapping, workbook_sha

    def _load_radiomics_table(
        self, path: Path
    ) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, str], List[str]]:
        if not path.is_file():
            raise FileNotFoundError(f"subject-specific radiomics CSV not found: {path}")
        frame = pd.read_csv(path)
        if "scan_id" not in frame.columns or "structure_id" not in frame.columns:
            raise ValueError("radiomics CSV requires scan_id and structure_id")
        feature_names = sorted(
            column for column in frame.columns if column.startswith("radiomics_")
        )
        if not feature_names:
            raise ValueError(
                "radiomics CSV has no 'radiomics_' feature columns; use the "
                "CONNECT-4 subject-specific extractor"
            )
        by_subject: Dict[str, Dict[int, torch.Tensor]] = {}
        for row_index, row in frame.iterrows():
            subject = str(row["scan_id"]).strip()
            if not subject or subject == "nan":
                raise ValueError(f"radiomics row {row_index} has invalid scan_id")
            try:
                structure_id = int(row["structure_id"])
                values = row[feature_names].to_numpy(dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"radiomics row {row_index} is invalid") from exc
            if structure_id not in self.ID_TO_SLUG:
                raise ValueError(
                    f"radiomics row {row_index} has unknown ROI {structure_id}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"radiomics row {row_index} contains NaN or infinity")
            subject_rows = by_subject.setdefault(subject, {})
            if structure_id in subject_rows:
                raise ValueError(
                    f"duplicate radiomics row for {subject}, ROI {structure_id}"
                )
            subject_rows[structure_id] = torch.from_numpy(values.copy())

        available = set(by_subject)
        expected_scans = set(self.scan_ids)
        if available != expected_scans:
            raise ValueError(
                "radiomics CSV scan_id coverage must exactly match structural scans; "
                f"missing={sorted(expected_scans - available)[:10]}, "
                f"extra={sorted(available - expected_scans)[:10]}. Patient or "
                "filename-prefix fallback is forbidden across sessions."
            )
        scan_mapping = {scan_id: scan_id for scan_id in self.scan_ids}
        expected = set(self.ID_TO_SLUG)
        for scan_id, subject in scan_mapping.items():
            ids = set(by_subject[subject])
            if ids != expected:
                raise ValueError(
                    f"{scan_id} must have exactly {self.NUM_ROIS} subject-specific "
                    f"radiomics rows; missing={sorted(expected - ids)}, "
                    f"extra={sorted(ids - expected)}"
                )
        return by_subject, scan_mapping, feature_names

    def radiomics_for_scan(self, scan_id: str) -> Dict[int, torch.Tensor]:
        return self.radiomics_by_subject[self.scan_to_radiomics_subject[scan_id]]

    def _validate_radiomics_provenance(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                "PyRadiomics provenance is required for subject-specific ROI "
                f"features: {path}"
            )
        try:
            with path.open() as stream:
                provenance = json.load(stream)
        except Exception as exc:
            raise RuntimeError(f"invalid PyRadiomics provenance: {path}") from exc
        failures = []
        if provenance.get("schema") != self.RADIOMICS_PROVENANCE_SCHEMA:
            failures.append("wrong schema")
        extractor = provenance.get("extractor", {})
        expected_extractor = self.PYRADIOMICS_EXTRACTOR_CONTRACT
        if not isinstance(extractor, dict):
            failures.append("extractor identity is missing")
        else:
            for key, expected in expected_extractor.items():
                if extractor.get(key) != expected:
                    failures.append(f"extractor {key} differs")
            if extractor.get("distribution") == "pyradiomics-cuda":
                failures.append("CUDA fork is not official PyRadiomics")
            if not self._valid_sha256(extractor.get("runtime_lock_sha256")):
                failures.append("official runtime lock SHA-256 is missing")
            if not self._valid_sha256(extractor.get("runtime_tree_sha256")):
                failures.append("official runtime tree SHA-256 is missing")
        if provenance.get("feature_columns") != self.radiomics_feature_names:
            failures.append("feature-column schema differs")
        if provenance.get("row_count") != self.NUM_ROIS * len(self.scan_ids):
            failures.append("row count differs from the complete cohort")
        if provenance.get("output_sha256") != sha256_file(self.radiomics_csv_path):
            failures.append("radiomics CSV hash mismatch")
        expected_sources = {
            scan_id: {
                "t1w": sha256_file(self._t1_path(scan_id)),
                "segmentation": sha256_file(self._segmentation_path(scan_id)),
            }
            for scan_id in self.scan_ids
        }
        if provenance.get("source_sha256") != expected_sources:
            failures.append("T1w/segmentation source hashes differ")
        if failures:
            raise RuntimeError(
                "PyRadiomics provenance is invalid: " + "; ".join(failures)
            )
        self._radiomics_provenance = provenance

    @staticmethod
    def _valid_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def roi_feature_extractor_identity(self, scan_id: str) -> dict:
        """Return the portable, scan-independent ROI conditioning protocol."""
        if scan_id in self._roi_feature_identity_cache:
            return dict(self._roi_feature_identity_cache[scan_id])
        provenance_path = self.root / "AnatCL" / scan_id / "provenance.json"
        try:
            with provenance_path.open() as stream:
                anatcl_provenance = json.load(stream)
        except Exception as exc:
            raise RuntimeError(
                f"invalid AnatCL provenance for {scan_id}: {provenance_path}"
            ) from exc
        model = anatcl_provenance.get("model")
        extraction = anatcl_provenance.get("extraction")
        cat12_input = anatcl_provenance.get("cat12_input")
        if (
            anatcl_provenance.get("schema") != self.ANATCL_PROVENANCE_SCHEMA
            or anatcl_provenance.get("scan_id") != scan_id
            or not isinstance(model, dict)
            or model != self.ANATCL_MODEL_CONTRACT
            or extraction != self.ANATCL_EXTRACTION_CONTRACT
            or not isinstance(cat12_input, dict)
            or cat12_input.get("schema") != self.CAT12_INPUT_PROVENANCE_SCHEMA
            or cat12_input.get("pipeline") != self.CAT12_PREPROCESSING_CONTRACT
            or not self._valid_sha256(cat12_input.get("authority_manifest_sha256"))
            or not self._valid_sha256(cat12_input.get("scan_record_sha256"))
            or not self._valid_sha256(cat12_input.get("cat12_mwp1_sha256"))
            or not self._valid_sha256(
                cat12_input.get("segmentation_in_cat12_vbm_sha256")
            )
        ):
            raise RuntimeError(
                f"{scan_id} AnatCL model/extraction provenance is invalid"
            )
        radiomics = self._radiomics_provenance
        extractor = radiomics.get("extractor")
        if (
            radiomics.get("schema") != self.RADIOMICS_PROVENANCE_SCHEMA
            or not isinstance(extractor, dict)
            or any(
                extractor.get(key) != expected
                for key, expected in self.PYRADIOMICS_EXTRACTOR_CONTRACT.items()
            )
            or not self._valid_sha256(extractor.get("runtime_lock_sha256"))
            or not self._valid_sha256(extractor.get("runtime_tree_sha256"))
            or radiomics.get("feature_columns") != self.radiomics_feature_names
        ):
            raise RuntimeError("PyRadiomics extractor/feature schema is invalid")
        implementation_path = (
            Path(__file__).resolve().parents[1]
            / "preprocessing"
            / "extract_roi_features.py"
        )
        payload = {
            "format": self.ROI_FEATURE_CONDITIONING_SCHEMA,
            "roi_specs": [
                {"index": index, "slug": slug, "label_id": label_id}
                for index, (slug, label_id) in enumerate(self.ROI_SPECS)
            ],
            "anatcl": {
                "schema": self.ANATCL_PROVENANCE_SCHEMA,
                "model": dict(model),
                "extraction": dict(extraction),
                "cat12_input_protocol": {
                    "schema": self.CAT12_INPUT_PROVENANCE_SCHEMA,
                    "pipeline": dict(self.CAT12_PREPROCESSING_CONTRACT),
                },
                "embedding_dim": self.ANATCL_EMBEDDING_DIM,
            },
            "pyradiomics": {
                "schema": self.RADIOMICS_PROVENANCE_SCHEMA,
                "extractor": dict(extractor),
                "feature_columns": list(self.radiomics_feature_names),
                "feature_dim": len(self.radiomics_feature_names),
            },
            "concatenation": {
                "order": ["anatcl", "pyradiomics"],
                "output_dim": self.ANATCL_EMBEDDING_DIM
                + len(self.radiomics_feature_names),
            },
            "extraction_implementation_sha256": sha256_file(implementation_path),
        }
        payload["fingerprint_sha256"] = canonical_sha256(payload)
        artifact = {
            "format": self.ROI_FEATURE_ARTIFACT_SCHEMA,
            "scan_id": scan_id,
            "cat12_input": dict(cat12_input),
            "anatcl_provenance_sha256": sha256_file(provenance_path),
        }
        artifact["fingerprint_sha256"] = canonical_sha256(artifact)
        self._roi_feature_identity_cache[scan_id] = payload
        self._roi_feature_artifact_identity_cache[scan_id] = artifact
        return dict(payload)

    def roi_feature_artifact_identity(self, scan_id: str) -> dict:
        """Return per-scan CAT12 inputs separately from the shared protocol."""
        if scan_id not in self._roi_feature_artifact_identity_cache:
            self.roi_feature_extractor_identity(scan_id)
        return dict(self._roi_feature_artifact_identity_cache[scan_id])

    def _load_anatcl_embeddings(self, scan_id: str) -> Dict[str, torch.Tensor]:
        if scan_id in self.anatcl_cache:
            return self.anatcl_cache[scan_id]
        folder = self.root / "AnatCL" / scan_id
        provenance_path = folder / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"AnatCL provenance not found for {scan_id}: {provenance_path}"
            )
        with provenance_path.open() as stream:
            provenance = json.load(stream)
        self.roi_feature_extractor_identity(scan_id)
        if provenance.get("schema") != self.ANATCL_PROVENANCE_SCHEMA:
            raise RuntimeError(f"{scan_id} AnatCL provenance schema is invalid")
        expected_sources = {
            "t1w": sha256_file(self._t1_path(scan_id)),
            "segmentation": sha256_file(self._segmentation_path(scan_id)),
        }
        if provenance.get("source_sha256") != expected_sources:
            raise RuntimeError(f"{scan_id} AnatCL embeddings are stale for T1/mask")
        if provenance.get("model", {}).get("pretrained") is not True:
            raise RuntimeError(
                f"{scan_id} AnatCL provenance does not identify pretrained weights"
            )
        output_hashes = provenance.get("embedding_sha256")
        if not isinstance(output_hashes, dict):
            raise RuntimeError(f"{scan_id} AnatCL embedding hashes are missing")

        embeddings: Dict[str, torch.Tensor] = {}
        for slug, _label_id in self.ROI_SPECS:
            path = folder / f"{scan_id}_{slug}.pth"
            if output_hashes.get(slug) != sha256_file(path):
                raise RuntimeError(f"{scan_id} AnatCL hash mismatch for {slug}")
            try:
                embedding = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                embedding = torch.load(path, map_location="cpu")
            embedding = torch.as_tensor(embedding).reshape(-1).float()
            if (
                embedding.numel() != self.ANATCL_EMBEDDING_DIM
                or not torch.isfinite(embedding).all()
            ):
                raise ValueError(
                    f"{scan_id} {slug} AnatCL embedding must be finite "
                    f"{self.ANATCL_EMBEDDING_DIM}-D"
                )
            embeddings[slug] = embedding
        if set(output_hashes) != {slug for slug, _ in self.ROI_SPECS}:
            raise RuntimeError(
                f"{scan_id} AnatCL provenance contains unexpected outputs"
            )
        self.anatcl_cache[scan_id] = embeddings
        return embeddings

    def _build_roi_embeddings(self, scan_id: str) -> torch.Tensor:
        anatcl = self._load_anatcl_embeddings(scan_id)
        radiomics = self.radiomics_for_scan(scan_id)
        rows = []
        for slug, label_id in self.ROI_SPECS:
            anatcl_row = anatcl[slug]
            radiomics_row = radiomics[label_id]
            if self.scaler_manager is not None:
                anatcl_row = self.scaler_manager.scale_anatcl(anatcl_row)
                radiomics_row = self.scaler_manager.scale_radiomics(radiomics_row)
            rows.append(torch.cat((anatcl_row, radiomics_row)))
        features = torch.stack(rows)
        if not torch.isfinite(features).all():
            raise ValueError(f"{scan_id} ROI features contain NaN or infinity")
        return features

    def _load_nifti(
        self,
        path: Path,
        is_fmri: bool = False,
        is_mask: bool = False,
        skip_resample: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        if is_fmri or skip_resample:
            raise ValueError(
                "graph preprocessing loader accepts structural volumes only"
            )
        image = nib.load(str(path))
        if image.ndim != 3 or tuple(image.shape) != self.target_shape:
            raise ValueError(f"{path} is not on the certified structural grid")
        array = image.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains NaN or infinity")
        if is_mask:
            rounded = np.rint(array)
            if not np.allclose(array, rounded, rtol=0.0, atol=1e-4):
                raise ValueError(f"{path} contains non-integer segmentation labels")
            array = rounded
        tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
        return tensor.float(), image.affine.copy()

    def _compute_patch_distribution(
        self, mask: torch.Tensor, patch_idx: int
    ) -> Dict[int, float]:
        x_index, y_index, z_index = patch_grid_xyz(
            patch_idx, self.target_shape, self.patch_size
        )
        starts = [
            x_index * self.patch_size[0],
            y_index * self.patch_size[1],
            z_index * self.patch_size[2],
        ]
        patch = mask[
            0,
            0,
            starts[0] : starts[0] + self.patch_size[0],
            starts[1] : starts[1] + self.patch_size[1],
            starts[2] : starts[2] + self.patch_size[2],
        ]
        total = int(patch.numel())
        distribution = {}
        for label, count in zip(*torch.unique(patch, return_counts=True)):
            label_id = int(label.item())
            if label_id in self.ID_TO_SLUG and int(count.item()) > 0:
                distribution[label_id] = float(count.item()) / total
        return distribution

    def _compute_patch_center_mm(
        self,
        patch_idx: int,
        segmentation: torch.Tensor,
        affine: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Foreground center of mass in physical space, with geometric empty fallback."""
        patch_xyz = patch_grid_xyz(patch_idx, self.target_shape, self.patch_size)
        start = np.asarray(patch_xyz, dtype=np.float64) * np.asarray(
            self.patch_size, dtype=np.float64
        )
        patch = segmentation[
            0,
            0,
            int(start[0]) : int(start[0]) + self.patch_size[0],
            int(start[1]) : int(start[1]) + self.patch_size[1],
            int(start[2]) : int(start[2]) + self.patch_size[2],
        ]
        foreground = torch.nonzero(patch > 0, as_tuple=False).cpu().numpy()
        if len(foreground):
            voxel = start + foreground.mean(axis=0)
        else:
            voxel = geometric_patch_center_voxel(patch_xyz, self.patch_size)
        physical = np.asarray(affine, dtype=np.float64) @ np.append(voxel, 1.0)
        return tuple(float(value) for value in physical[:3])

    def source_fingerprint(self, scan_id: str) -> Dict[str, object]:
        normative_subject = self.scan_to_normative_subject[scan_id]
        radiomics_subject = self.scan_to_radiomics_subject[scan_id]
        anatcl_folder = self.root / "AnatCL" / scan_id
        paths = {
            "t1w": self._t1_path(scan_id),
            "segmentation": self._segmentation_path(scan_id),
            "dwi": self.dwi_matrix_path,
            "normative_csv": self.normative_csv_path,
            "radiomics_csv": self.radiomics_csv_path,
            "radiomics_provenance": self.radiomics_provenance_path,
            "normative_workbook": Path(__file__).resolve().parents[1]
            / "normative"
            / "mmc2.xlsm",
            "anatcl_provenance": anatcl_folder / "provenance.json",
        }
        t1_identity = (
            self._validate_production_t1_artifact(scan_id)
            if (
                self.common_grid_contract_sha256 is not None
                or self.native_alignment_authority_sha256 is not None
            )
            else None
        )
        fingerprint = {
            name: (
                t1_identity["output_sha256"]
                if name == "t1w" and t1_identity is not None
                else t1_identity["segmentation_sha256"]
                if name == "segmentation" and t1_identity is not None
                else sha256_file(path)
            )
            for name, path in paths.items()
        }
        fingerprint["patch_coordinate_contract"] = dict(PATCH_COORDINATE_CONTRACT)
        if t1_identity is not None:
            if self.common_grid_contract_sha256 is not None:
                fingerprint["common_grid_contract_sha256"] = (
                    self.common_grid_contract_sha256
                )
            else:
                fingerprint["native_alignment_authority_sha256"] = (
                    self.native_alignment_authority_sha256
                )
            fingerprint["t1_preprocessing_sidecar_sha256"] = t1_identity[
                "t1_sidecar_sha256"
            ]
            fingerprint["segmentation_preprocessing_sidecar_sha256"] = t1_identity[
                "segmentation_sidecar_sha256"
            ]
            fingerprint["structural_source_identity"] = t1_identity[
                "structural_source_identity"
            ]
            fingerprint["t1_preprocessing_identity"] = t1_identity
        fingerprint["structural_grid_geometry"] = (
            t1_identity["structural_grid_geometry"]
            if t1_identity is not None
            else {
                "t1w": nifti_geometry_fingerprint(paths["t1w"]),
                "segmentation": nifti_geometry_fingerprint(paths["segmentation"]),
            }
        )
        if self.cohort_manifest_path is not None:
            fingerprint["cohort_manifest"] = sha256_file(self.cohort_manifest_path)
            fingerprint["normative_scan_to_patient"] = canonical_sha256(
                {
                    scan_id: self.patient_id_by_scan[scan_id]
                    for scan_id in sorted(self.patient_id_by_scan or {})
                }
            )
        fingerprint["normative_subject_rows"] = canonical_sha256(
            self.normative_source_rows[normative_subject]
        )
        fingerprint["radiomics_subject_rows"] = canonical_sha256(
            {
                str(label): self.radiomics_by_subject[radiomics_subject][label].tolist()
                for label in sorted(self.radiomics_by_subject[radiomics_subject])
            }
        )
        fingerprint["roi_feature_extractor_identity"] = (
            self.roi_feature_extractor_identity(scan_id)
        )
        fingerprint["roi_feature_artifact_identity"] = (
            self.roi_feature_artifact_identity(scan_id)
        )
        for slug, _ in self.ROI_SPECS:
            fingerprint[f"anatcl:{slug}"] = sha256_file(
                anatcl_folder / f"{scan_id}_{slug}.pth"
            )
        return fingerprint

    def __len__(self) -> int:
        return len(self.scan_ids)

    def __getitem__(self, index: int) -> Dict:
        scan_id = self.scan_ids[index]
        t1_path = self._t1_path(scan_id)
        mask_path = self._segmentation_path(scan_id)
        t1w, affine = self._load_nifti(t1_path)
        segmentation, mask_affine = self._load_nifti(mask_path, is_mask=True)
        if not np.allclose(affine, mask_affine, rtol=0.0, atol=1e-3):
            raise ValueError(f"{scan_id} T1/mask affines changed after validation")
        if not bool((segmentation > 0).any()):
            raise ValueError(f"{scan_id} segmentation is empty")
        return {
            "t1w": t1w,
            "mask": (segmentation > 0).float(),
            "segmentation": segmentation,
            "roi_embeddings": self._build_roi_embeddings(scan_id),
            "structure_to_roi_idx": self.structure_to_roi_idx,
            "dwi_matrix": self.dwi_matrix,
            "scan_id": scan_id,
            "affine": torch.from_numpy(affine).float(),
        }


__all__ = ["Connect4Dataset"]
