"""
T1w structural-MRI preprocessing (CONNECT-4, Figure 1A input).

Steps:
  1. authenticate the same-scan raw T1w and SynthSeg source identity,
  2. reorient to RAS+ and resample to a hash-bound structural common grid,
  3. z-score inside the authenticated conformed segmentation support and keep
     the exterior exactly zero.

    python -m preprocessing.preprocess_t1 --in sub-01_T1w.nii.gz --out sub-01_T1w_128.nii.gz

Segmentation is produced separately with SynthSeg (run externally by you); run
`preprocess_seg.py` first so its authenticated derivative defines the support.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from .conform import (
    _sha256_file,
    conform_volume,
    load_common_grid_contract,
    zscore_inbrain,
)
from .preprocess_seg import SEG_PREPROCESSING_SCHEMA_VERSION
from .source_acquisition_identity import (
    SourceAcquisitionError,
    load_structural_source_authority,
    load_structural_source_binding,
    snapshot_binary_artifact,
    snapshot_json_artifact,
    structural_source_binding,
)


T1_PREPROCESSING_SCHEMA_VERSION = "connect4-t1-common-grid-v4"
T1_INTENSITY_NORMALIZATION = (
    "authenticated-segmentation-support in-brain z-score with exact zero exterior"
)


def _nifti_from_snapshot(content: bytes, path: Path, *, label: str) -> nib.Nifti1Image:
    try:
        nifti_bytes = gzip.decompress(content) if path.name.endswith(".gz") else content
        return nib.Nifti1Image.from_bytes(nifti_bytes)
    except Exception as exc:
        raise ValueError(f"{label} is not a valid NIfTI image") from exc


def preprocess_t1(
    in_path: str,
    out_path: str,
    normalize: bool = True,
    *,
    common_grid_contract_path: str | None = None,
    common_grid_contract_sha256: str | None = None,
    segmentation_path: str | None = None,
    segmentation_sidecar_path: str | None = None,
    structural_source_authority_path: str | None = None,
    structural_source_authority_sha256: str | None = None,
    source_acquisition_identity_path: str | None = None,
    source_acquisition_identity_sha256: str | None = None,
    synthetic_target=None,
) -> str:
    if normalize is not True:
        raise ValueError(
            "the production T1w pipeline requires in-brain z-score normalization"
        )
    contract = None
    source_binding = None
    segmentation = None
    segmentation_sidecar_sha256 = None
    segmentation_output_sha256 = None
    if common_grid_contract_path is not None:
        if not common_grid_contract_sha256:
            raise ValueError(
                "production common-grid evidence requires an externally configured "
                "SHA-256 pin"
            )
        contract = load_common_grid_contract(
            common_grid_contract_path,
            expected_sha256=common_grid_contract_sha256,
        )
        if (
            source_acquisition_identity_path is not None
            or source_acquisition_identity_sha256 is not None
        ):
            raise ValueError(
                "legacy BOLD-bearing SourceAcquisition authority is forbidden for "
                "structural preprocessing; provide the target-blind v2 authority"
            )
        if (
            not segmentation_path
            or not structural_source_authority_path
            or not structural_source_authority_sha256
        ):
            raise ValueError(
                "production T1 preprocessing requires the authenticated conformed "
                "segmentation and externally SHA-pinned target-blind structural "
                "source authority"
            )
        try:
            source_identity, source_evidence = load_structural_source_authority(
                Path(structural_source_authority_path),
                expected_sha256=structural_source_authority_sha256,
                expected_raw_t1=Path(in_path),
            )
            source_binding = structural_source_binding(source_identity, source_evidence)
            segmentation_output = Path(segmentation_path)
            if segmentation_sidecar_path is None:
                segmentation_sidecar = segmentation_output.with_name(
                    segmentation_output.name[:-7] + ".json"
                    if segmentation_output.name.endswith(".nii.gz")
                    else segmentation_output.stem + ".json"
                )
            else:
                segmentation_sidecar = Path(segmentation_sidecar_path)
            segmentation_metadata, segmentation_sidecar_evidence = (
                snapshot_json_artifact(
                    segmentation_sidecar,
                    label="conformed segmentation sidecar",
                )
            )
            segmentation_sidecar_sha256 = segmentation_sidecar_evidence["sha256"]
            segmentation_output_sha256 = segmentation_metadata.get("output_sha256")
            if (
                segmentation_metadata.get("schema") != SEG_PREPROCESSING_SCHEMA_VERSION
                or segmentation_metadata.get("scan_id") != source_binding["scan_id"]
                or segmentation_metadata.get("source_sha256")
                != source_binding["synthseg_mask_sha256"]
                or segmentation_metadata.get("common_grid_contract_sha256")
                != contract["contract_sha256"]
                or segmentation_metadata.get("architecture_shape")
                != contract["architecture_shape"]
            ):
                raise SourceAcquisitionError(
                    "conformed segmentation provenance differs from T1 source/grid"
                )
            admitted_binding, _ = load_structural_source_binding(
                segmentation_metadata.get("structural_source"),
                expected_scan_id=str(source_binding["scan_id"]),
            )
            if admitted_binding != source_binding:
                raise SourceAcquisitionError(
                    "T1 and segmentation source-acquisition identities differ"
                )
            segmentation_bytes, _ = snapshot_binary_artifact(
                segmentation_output,
                expected_sha256=str(segmentation_output_sha256 or ""),
                label="conformed segmentation",
            )
            segmentation = _nifti_from_snapshot(
                segmentation_bytes,
                segmentation_output,
                label="conformed segmentation",
            )
        except SourceAcquisitionError as exc:
            raise ValueError(
                "T1/segmentation source-acquisition admission failed"
            ) from exc
    elif synthetic_target is None:
        raise ValueError("production T1 preprocessing requires a common-grid contract")
    if source_binding is not None:
        try:
            t1_bytes, _ = snapshot_binary_artifact(
                Path(in_path),
                expected_sha256=str(source_binding["raw_t1_sha256"]),
                label="raw T1w preprocessing source",
            )
        except SourceAcquisitionError as exc:
            raise ValueError("raw T1w changed after source admission") from exc
        img = _nifti_from_snapshot(t1_bytes, Path(in_path), label="source T1w")
    else:
        img = nib.load(in_path)
    if not np.isfinite(img.get_fdata(dtype=np.float32)).all():
        raise ValueError("source T1w contains NaN or infinity")
    img = conform_volume(
        img, order=1, grid_contract=contract, synthetic_target=synthetic_target
    )
    data = img.get_fdata().astype(np.float32)
    if segmentation is not None:
        if tuple(segmentation.shape) != tuple(img.shape) or not np.allclose(
            segmentation.affine, img.affine, rtol=0.0, atol=1e-5
        ):
            raise ValueError("authenticated segmentation and conformed T1 grids differ")
        segmentation_values = segmentation.get_fdata(dtype=np.float32)
        if not np.isfinite(segmentation_values).all() or not np.allclose(
            segmentation_values,
            np.rint(segmentation_values),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("authenticated segmentation is not a finite label map")
        support = segmentation_values > 0
        if not np.any(support):
            raise ValueError("authenticated segmentation has empty brain support")
        data = zscore_inbrain(data, support=support)
    else:
        data = zscore_inbrain(data)
    if not np.isfinite(data).all():
        raise RuntimeError("T1w normalization produced NaN or infinity")
    output = Path(out_path)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), output)
    if contract is not None:
        sidecar = output.with_name(
            output.name[:-7] + ".json"
            if output.name.endswith(".nii.gz")
            else output.stem + ".json"
        )
        sidecar.write_text(
            json.dumps(
                {
                    "schema": T1_PREPROCESSING_SCHEMA_VERSION,
                    "scan_id": source_binding["scan_id"],
                    "source_sha256": source_binding["raw_t1_sha256"],
                    "output_sha256": _sha256_file(output),
                    "segmentation_output_sha256": segmentation_output_sha256,
                    "segmentation_sidecar_sha256": segmentation_sidecar_sha256,
                    "structural_source": source_binding,
                    "common_grid_contract_path": contract["contract_path"],
                    "common_grid_contract_sha256": contract["contract_sha256"],
                    "architecture_shape": contract["architecture_shape"],
                    "anatomical_shape": contract["anatomical_shape"],
                    "architecture_padding": contract["architecture_padding"],
                    "matrix_size_reported_by_paper": False,
                    "manuscript_claims": {
                        "voxel_size_mm": [3.0, 3.0, 3.0],
                        "spatial_matrix": None,
                        "intensity_normalization": None,
                    },
                    "versioned_recovery_choices": {
                        "common_grid_schema": contract["schema"],
                        "intensity_normalization": (T1_INTENSITY_NORMALIZATION),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"[t1] {in_path} -> {out_path}  shape={data.shape}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--common-grid-contract", required=True)
    ap.add_argument("--common-grid-contract-sha256", required=True)
    ap.add_argument("--segmentation", required=True)
    ap.add_argument("--segmentation-sidecar", default=None)
    ap.add_argument("--structural-source-authority", required=True)
    ap.add_argument("--structural-source-authority-sha256", required=True)
    a = ap.parse_args()
    preprocess_t1(
        a.in_path,
        a.out_path,
        common_grid_contract_path=a.common_grid_contract,
        common_grid_contract_sha256=a.common_grid_contract_sha256,
        segmentation_path=a.segmentation,
        segmentation_sidecar_path=a.segmentation_sidecar,
        structural_source_authority_path=a.structural_source_authority,
        structural_source_authority_sha256=(
            a.structural_source_authority_sha256
        ),
    )
