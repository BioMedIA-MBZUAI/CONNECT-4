"""
Segmentation conforming (CONNECT-4, Figure 1A "3D Segmentation").

Segmentation is **run externally by the user with SynthSeg**
(Billot et al., https://github.com/BBillot/SynthSeg). This step only brings the
externally-produced label map onto the hash-bound structural common grid
(3 mm isotropic, nearest-neighbour so labels are preserved).

    python -m preprocessing.preprocess_seg --in sub-01_synthseg.nii.gz --out seg_128.nii.gz

The conformed labels drive the mask patches / ROI statistics (Fig 1A), the ROI
graph and ROI-coverage hyperedges (Fig 1B), and the measured ROI volumes used by
the normative / atrophy module (`normative/`).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .conform import _sha256_file, conform_volume, load_common_grid_contract
from .source_acquisition_identity import (
    SourceAcquisitionError,
    load_structural_source_authority,
    snapshot_binary_artifact,
    structural_source_binding,
)


SEG_PREPROCESSING_SCHEMA_VERSION = "connect4-seg-common-grid-v3"


def preprocess_seg(
    in_path: str,
    out_path: str,
    *,
    common_grid_contract_path: str | None = None,
    common_grid_contract_sha256: str | None = None,
    structural_source_authority_path: str | None = None,
    structural_source_authority_sha256: str | None = None,
    source_acquisition_identity_path: str | None = None,
    source_acquisition_identity_sha256: str | None = None,
    synthetic_target=None,
) -> str:
    """Conform an external SynthSeg map to one explicit common grid."""
    contract = None
    source_binding = None
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
            not structural_source_authority_path
            or not structural_source_authority_sha256
        ):
            raise ValueError(
                "production segmentation preprocessing requires an externally "
                "SHA-pinned target-blind structural source authority"
            )
        try:
            source_identity, source_evidence = load_structural_source_authority(
                Path(structural_source_authority_path),
                expected_sha256=structural_source_authority_sha256,
                expected_synthseg_mask=Path(in_path),
            )
        except SourceAcquisitionError as exc:
            raise ValueError(
                "segmentation structural-source authority admission failed"
            ) from exc
        source_binding = structural_source_binding(source_identity, source_evidence)
    elif synthetic_target is None:
        raise ValueError(
            "production segmentation preprocessing requires a common-grid contract"
        )
    if source_binding is not None:
        try:
            source_bytes, _ = snapshot_binary_artifact(
                Path(in_path),
                expected_sha256=str(source_binding["synthseg_mask_sha256"]),
                label="raw SynthSeg preprocessing source",
            )
            nifti_bytes = (
                gzip.decompress(source_bytes)
                if str(in_path).endswith(".gz")
                else source_bytes
            )
            seg = nib.Nifti1Image.from_bytes(nifti_bytes)
        except SourceAcquisitionError as exc:
            raise ValueError(
                "raw SynthSeg mask changed after source admission"
            ) from exc
        except Exception as exc:
            raise ValueError("raw SynthSeg source is not a valid NIfTI image") from exc
    else:
        seg = nib.load(in_path)
    source_values = seg.get_fdata(dtype=np.float32)
    if not np.isfinite(source_values).all():
        raise ValueError("source SynthSeg segmentation contains NaN or infinity")
    if not np.allclose(source_values, np.rint(source_values), rtol=0.0, atol=1e-6):
        raise ValueError("source SynthSeg segmentation contains non-integer labels")
    seg = conform_volume(
        seg, order=0, grid_contract=contract, synthetic_target=synthetic_target
    )
    output_values = seg.get_fdata(dtype=np.float32)
    if not np.isfinite(output_values).all() or not np.allclose(
        output_values, np.rint(output_values), rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("conformed SynthSeg segmentation is not a finite label map")
    if not np.any(output_values > 0):
        raise RuntimeError("conformed SynthSeg segmentation has empty brain support")
    output = Path(out_path)
    nib.save(seg, output)
    if contract is not None:
        sidecar = output.with_name(
            output.name[:-7] + ".json"
            if output.name.endswith(".nii.gz")
            else output.stem + ".json"
        )
        sidecar.write_text(
            json.dumps(
                {
                    "schema": SEG_PREPROCESSING_SCHEMA_VERSION,
                    "scan_id": source_binding["scan_id"],
                    "source_sha256": source_binding["synthseg_mask_sha256"],
                    "output_sha256": _sha256_file(output),
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
                    },
                    "versioned_recovery_choices": {
                        "common_grid_schema": contract["schema"],
                        "interpolation": "nearest-neighbour",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"[seg] {in_path} -> {out_path}  shape={seg.shape}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path", required=True, help="external SynthSeg label map"
    )
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--common-grid-contract", required=True)
    ap.add_argument("--common-grid-contract-sha256", required=True)
    ap.add_argument("--structural-source-authority", required=True)
    ap.add_argument("--structural-source-authority-sha256", required=True)
    a = ap.parse_args()
    preprocess_seg(
        a.in_path,
        a.out_path,
        common_grid_contract_path=a.common_grid_contract,
        common_grid_contract_sha256=a.common_grid_contract_sha256,
        structural_source_authority_path=a.structural_source_authority,
        structural_source_authority_sha256=(
            a.structural_source_authority_sha256
        ),
    )
