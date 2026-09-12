import json

import nibabel as nib
import numpy as np
import pytest

from data.provenance import sha256_file
from preprocessing.build_common_grid_contract import build_common_grid_contract
from preprocessing.conform import (
    COMMON_GRID_SCHEMA_VERSION,
    STRUCTURAL_MANIFEST_SCHEMA_VERSION,
    conform_volume,
    crop_architecture_array,
    load_common_grid_contract,
)


def _build_contract(tmp_path):
    t1_path = tmp_path / "sub-01_T1w.nii.gz"
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(
        nib.Nifti1Image(
            np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7), affine
        ),
        t1_path,
    )
    manifest_path = tmp_path / "structural_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": STRUCTURAL_MANIFEST_SCHEMA_VERSION,
                "functional_data_used": False,
                "structural_reference": {
                    "path": str(t1_path),
                    "sha256": sha256_file(t1_path),
                    "modality": "T1w",
                    "cohort_derived": True,
                    "functional_data_used": False,
                    "derivation_method": "synthetic-test-reference",
                },
                "scans": [
                    {
                        "scan_id": "sub-01",
                        "t1w_path": str(t1_path),
                        "t1w_sha256": sha256_file(t1_path),
                    }
                ],
            }
        )
    )
    contract_path = build_common_grid_contract(
        t1_path,
        manifest_path,
        tmp_path / "common_grid.json",
        patch_multiple=(4, 4, 4),
    )
    return t1_path, manifest_path, contract_path


def test_structural_grid_is_derived_padded_hash_bound_and_crop_reversible(tmp_path):
    _t1_path, _manifest_path, contract_path = _build_contract(tmp_path)
    contract = load_common_grid_contract(
        contract_path, expected_sha256=sha256_file(contract_path)
    )

    assert contract["schema"] == COMMON_GRID_SCHEMA_VERSION
    assert contract["anatomical_shape"] == [5, 6, 7]
    assert contract["architecture_shape"] == [8, 8, 8]
    assert contract["architecture_padding"]["purpose"] == (
        "patch-multiple-only-not-paper-anatomy"
    )
    assert contract["derivation"]["matrix_size_reported_by_paper"] is False
    architecture = np.arange(8**3).reshape(8, 8, 8)
    cropped = crop_architecture_array(architecture, contract)
    assert cropped.shape == (5, 6, 7)
    starts = contract["architecture_padding"]["before"]
    assert cropped[0, 0, 0] == architecture[tuple(starts)]


def test_common_grid_rejects_external_pin_or_structural_source_tampering(tmp_path):
    t1_path, _manifest_path, contract_path = _build_contract(tmp_path)
    with pytest.raises(ValueError, match="does not match configuration"):
        load_common_grid_contract(contract_path, expected_sha256="0" * 64)

    with t1_path.open("ab") as stream:
        stream.write(b"tampered structural source")
    with pytest.raises(ValueError, match="source structural reference SHA-256"):
        load_common_grid_contract(contract_path)


def test_grid_builder_rejects_functional_evidence_in_structural_manifest(tmp_path):
    t1_path = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), t1_path)
    manifest_path = tmp_path / "bad_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": STRUCTURAL_MANIFEST_SCHEMA_VERSION,
                "functional_data_used": False,
                "structural_reference": {
                    "path": str(t1_path),
                    "sha256": sha256_file(t1_path),
                    "modality": "T1w",
                    "cohort_derived": True,
                    "functional_data_used": False,
                    "derivation_method": "synthetic-test-reference",
                },
                "scans": [
                    {
                        "scan_id": "sub-01",
                        "t1w_path": str(t1_path),
                        "t1w_sha256": sha256_file(t1_path),
                        "bold_frames": 128,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="functional fields are forbidden"):
        build_common_grid_contract(
            t1_path, manifest_path, tmp_path / "grid.json"
        )


def test_spatial_conforming_has_no_implicit_matrix_default():
    image = nib.Nifti1Image(np.ones((2, 3, 4)), np.eye(4))
    with pytest.raises(ValueError, match="validated common-grid contract"):
        conform_volume(image)
