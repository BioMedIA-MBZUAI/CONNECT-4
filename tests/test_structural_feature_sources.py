import json
import os
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import Connect4Dataset
from data import dataset as dataset_module
from data.provenance import canonical_sha256, sha256_file
from models.brainiac_wrapper import BrainIACWrapper
from preprocessing.native_structural_identity import (
    NATIVE_CERTIFICATION_STATUS,
    NATIVE_SPATIAL_PROFILE,
    NATIVE_STRUCTURAL_BATCH_MANIFEST_SCHEMA,
    NATIVE_STRUCTURAL_BATCH_PURPOSE,
    NATIVE_STRUCTURAL_SCAN_PROVENANCE_SCHEMA,
    NATIVE_STRUCTURAL_SCAN_PURPOSE,
    build_native_structural_alignment_binding,
    materialize_native_padded_structural,
)
from preprocessing.source_acquisition_identity import (
    load_structural_source_binding,
)
from tests.support import (
    write_source_acquisition_fixture,
    write_structural_sources,
)


def _dataset(tmp_path, scan_ids=("sub-01_run-1", "sub-02_run-1")):
    sources = write_structural_sources(tmp_path, scan_ids=scan_ids)
    dataset = Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
    )
    return dataset, sources


def _write_production_t1_fixture(tmp_path):
    scan_id = "sub-01_run-1"
    common_grid_sha256 = "c" * 64
    sources = write_structural_sources(tmp_path, scan_ids=(scan_id,))
    t1_path = tmp_path / "T1" / f"{scan_id}_T1.nii.gz"
    mask_path = tmp_path / "Masks" / f"{scan_id}_mask.nii.gz"
    affine = np.diag([3.0, 3.0, 3.0, 1.0])

    segmentation = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation[1:, :, :] = 2
    t1_values = np.zeros((4, 4, 4), dtype=np.float32)
    in_brain = np.linspace(-2.0, 2.0, int((segmentation > 0).sum())).astype(np.float32)
    t1_values[segmentation > 0] = in_brain
    nib.save(nib.Nifti1Image(t1_values, affine), t1_path)
    nib.save(nib.Nifti1Image(segmentation, affine), mask_path)

    source_hashes = {
        "t1w": sha256_file(t1_path),
        "segmentation": sha256_file(mask_path),
    }
    anatcl_path = tmp_path / "AnatCL" / scan_id / "provenance.json"
    anatcl = json.loads(anatcl_path.read_text(encoding="utf-8"))
    anatcl["source_sha256"] = dict(source_hashes)
    anatcl_path.write_text(json.dumps(anatcl, indent=2), encoding="utf-8")
    radiomics_path = tmp_path / "radiomics_provenance.json"
    radiomics = json.loads(radiomics_path.read_text(encoding="utf-8"))
    radiomics["source_sha256"][scan_id] = dict(source_hashes)
    radiomics_path.write_text(json.dumps(radiomics, indent=2), encoding="utf-8")

    source_acquisition = write_source_acquisition_fixture(
        tmp_path,
        scan_id=scan_id,
        t1_source=t1_path,
        synthseg_source=mask_path,
    )
    sources["source_acquisition"] = source_acquisition

    segmentation_sidecar_path = tmp_path / "Masks" / f"{scan_id}_mask.json"
    segmentation_sidecar = {
        "schema": Connect4Dataset.SEG_PREPROCESSING_SCHEMA,
        "scan_id": scan_id,
        "source_sha256": source_acquisition["binding"]["synthseg_mask_sha256"],
        "output_sha256": sha256_file(mask_path),
        "structural_source": source_acquisition["binding"],
        "common_grid_contract_path": "/authority/common-grid.json",
        "common_grid_contract_sha256": common_grid_sha256,
        "architecture_shape": [4, 4, 4],
        "anatomical_shape": [4, 4, 4],
        "architecture_padding": None,
        "matrix_size_reported_by_paper": False,
        "manuscript_claims": {
            "voxel_size_mm": [3.0, 3.0, 3.0],
            "spatial_matrix": None,
        },
        "versioned_recovery_choices": {
            "common_grid_schema": "connect4-common-grid-v1",
            "interpolation": "nearest-neighbour",
        },
    }
    segmentation_sidecar_path.write_text(
        json.dumps(segmentation_sidecar, indent=2), encoding="utf-8"
    )

    sidecar_path = tmp_path / "T1" / f"{scan_id}_T1.json"
    sidecar = {
        "schema": Connect4Dataset.T1_PREPROCESSING_SCHEMA,
        "scan_id": scan_id,
        "source_sha256": source_acquisition["binding"]["raw_t1_sha256"],
        "output_sha256": sha256_file(t1_path),
        "segmentation_output_sha256": sha256_file(mask_path),
        "segmentation_sidecar_sha256": sha256_file(segmentation_sidecar_path),
        "structural_source": source_acquisition["binding"],
        "common_grid_contract_path": "/authority/common-grid.json",
        "common_grid_contract_sha256": common_grid_sha256,
        "architecture_shape": [4, 4, 4],
        "anatomical_shape": [4, 4, 4],
        "architecture_padding": None,
        "matrix_size_reported_by_paper": False,
        "manuscript_claims": {
            "voxel_size_mm": [3.0, 3.0, 3.0],
            "spatial_matrix": None,
            "intensity_normalization": None,
        },
        "versioned_recovery_choices": {
            "common_grid_schema": "connect4-common-grid-v1",
            "intensity_normalization": (Connect4Dataset.T1_INTENSITY_NORMALIZATION),
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sources, scan_id, common_grid_sha256, sidecar_path


def _production_dataset(tmp_path, sources, common_grid_sha256):
    return Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
        common_grid_contract_sha256=common_grid_sha256,
    )


def _write_native_production_fixture(tmp_path):
    scan_id = "sub-01_run-1"
    native_shape = (61, 73, 61)
    architecture_shape = (64, 80, 64)
    native_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    native_affine[:3, 3] = [-90.5, -125.5, -71.5]
    sources = write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=architecture_shape
    )
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native_t1_path = native_dir / "t1_common3mm.nii.gz"
    native_mask_path = native_dir / "mask_common3mm.nii.gz"
    native_t1_values = np.linspace(
        0.0, 100.0, int(np.prod(native_shape)), dtype=np.float32
    ).reshape(native_shape)
    native_mask_values = np.zeros(native_shape, dtype=np.int16)
    native_mask_values[2:-2, 2:-2, 2:-2] = 2
    nib.save(nib.Nifti1Image(native_t1_values, native_affine), native_t1_path)
    nib.save(nib.Nifti1Image(native_mask_values, native_affine), native_mask_path)

    def artifact(path):
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    provenance_path = native_dir / "preprocessing_provenance.json"
    provenance = {
        "schema": NATIVE_STRUCTURAL_SCAN_PROVENANCE_SCHEMA,
        "purpose": NATIVE_STRUCTURAL_SCAN_PURPOSE,
        "scan_id": scan_id,
        "paper_certified": False,
        "certification_status": NATIVE_CERTIFICATION_STATUS,
        "spatial_profile": NATIVE_SPATIAL_PROFILE,
        "native_grid": {
            "shape": list(native_shape),
            "voxel_size_mm": 3.0,
            "affine": native_affine.tolist(),
        },
        "inputs": {
            "raw_t1": artifact(native_t1_path),
            "raw_synthseg_mask": artifact(native_mask_path),
        },
        "outputs": {
            "t1_common": artifact(native_t1_path),
            "mask_common": artifact(native_mask_path),
        },
    }
    provenance["record_sha256"] = canonical_sha256(provenance)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    batch_path = native_dir / "native_preprocessing_batch_manifest.json"
    batch = {
        "schema": NATIVE_STRUCTURAL_BATCH_MANIFEST_SCHEMA,
        "purpose": NATIVE_STRUCTURAL_BATCH_PURPOSE,
        "paper_certified": False,
        "certification_status": NATIVE_CERTIFICATION_STATUS,
        "spatial_profile": NATIVE_SPATIAL_PROFILE,
        "preprocessing_contract": {
            "native_shape": list(native_shape),
            "architecture_shape": list(architecture_shape),
            "voxel_size_mm": 3.0,
            "padding_before": [1, 3, 1],
            "padding_after": [2, 4, 2],
            "padding_mode": "constant-zero-no-interpolation",
            "interpolation_after_native_preprocessing": False,
        },
        "scans": [
            {
                "scan_id": scan_id,
                "structural_provenance_path": str(provenance_path.resolve()),
                "structural_provenance_sha256": sha256_file(provenance_path),
            }
        ],
    }
    batch["record_sha256"] = canonical_sha256(batch)
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    batch_sha256 = sha256_file(batch_path)
    binding = build_native_structural_alignment_binding(
        batch_path,
        expected_batch_sha256=batch_sha256,
        scan_id=scan_id,
    )
    structural_stage_root = tmp_path / "padded_structural"
    scan_stage = structural_stage_root / scan_id
    scan_stage.mkdir(parents=True)
    t1_path = scan_stage / f"{scan_id}_T1.nii.gz"
    mask_path = scan_stage / f"{scan_id}_mask.nii.gz"
    materialize_native_padded_structural(
        batch_path,
        expected_batch_sha256=batch_sha256,
        scan_id=scan_id,
        t1_output_path=t1_path,
        segmentation_output_path=mask_path,
    )
    source_hashes = {
        "t1w": sha256_file(t1_path),
        "segmentation": sha256_file(mask_path),
    }
    anatcl_path = tmp_path / "AnatCL" / scan_id / "provenance.json"
    anatcl = json.loads(anatcl_path.read_text(encoding="utf-8"))
    anatcl["source_sha256"] = dict(source_hashes)
    anatcl_path.write_text(json.dumps(anatcl), encoding="utf-8")
    radiomics_path = tmp_path / "radiomics_provenance.json"
    radiomics = json.loads(radiomics_path.read_text(encoding="utf-8"))
    radiomics["source_sha256"][scan_id] = dict(source_hashes)
    radiomics_path.write_text(json.dumps(radiomics), encoding="utf-8")

    return sources, scan_id, batch_sha256, binding


def _native_production_dataset(tmp_path, sources, batch_sha256):
    return Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(16, 16, 16),
        target_shape=(64, 80, 64),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
        native_alignment_authority_path=str(
            tmp_path / "native" / "native_preprocessing_batch_manifest.json"
        ),
        native_alignment_authority_sha256=batch_sha256,
        structural_stage_root=str(tmp_path / "padded_structural"),
    )


def test_native_structural_profile_is_explicitly_noncertified_and_exactly_padded(
    tmp_path,
):
    sources, scan_id, batch_sha256, binding = _write_native_production_fixture(tmp_path)
    dataset = _native_production_dataset(tmp_path, sources, batch_sha256)
    fingerprint = dataset.source_fingerprint(scan_id)
    identity = fingerprint["t1_preprocessing_identity"]
    assert identity["paper_certified"] is False
    assert identity["structural_source_identity"] == binding
    assert fingerprint["native_alignment_authority_sha256"] == batch_sha256
    assert "common_grid_contract_sha256" not in fingerprint
    assert binding["native_shape"] == [61, 73, 61]
    assert binding["architecture_shape"] == [64, 80, 64]
    assert binding["identically_padded_modalities"] == [
        "t1w",
        "segmentation",
    ]
    assert binding["interpolation_after_native_preprocessing"] is False
    assert binding["matrix_size_reported_by_paper"] is False
    assert "bold" not in json.dumps(binding).lower()


def test_native_structural_authority_rejects_bold_bearing_legacy_fields(tmp_path):
    _sources, scan_id, _batch_sha256, binding = _write_native_production_fixture(
        tmp_path
    )
    batch_path = Path(binding["batch_authority"]["path"])
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch.pop("record_sha256")
    batch["raw_bold_sha256"] = "0" * 64
    batch["record_sha256"] = canonical_sha256(batch)
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="fields differ or include target data",
    ):
        build_native_structural_alignment_binding(
            batch_path,
            expected_batch_sha256=sha256_file(batch_path),
            scan_id=scan_id,
        )


def test_native_structural_profile_rejects_padding_or_normalization_tampering(
    tmp_path,
):
    sources, scan_id, batch_sha256, _binding = _write_native_production_fixture(
        tmp_path
    )
    t1_path = tmp_path / "padded_structural" / scan_id / f"{scan_id}_T1.nii.gz"
    t1_sidecar_path = (
        tmp_path / "padded_structural" / scan_id / f"{scan_id}_T1.json"
    )
    image = nib.load(str(t1_path))
    values = image.get_fdata(dtype=np.float32)
    values[0, 0, 0] = 0.25
    nib.save(nib.Nifti1Image(values, image.affine, image.header), t1_path)
    sidecar = json.loads(t1_sidecar_path.read_text(encoding="utf-8"))
    sidecar["output_sha256"] = sha256_file(t1_path)
    t1_sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(RuntimeError, match="padding is not exactly zero"):
        _native_production_dataset(tmp_path, sources, batch_sha256)


def test_paper_and_native_spatial_authorities_are_mutually_exclusive(tmp_path):
    sources = write_structural_sources(tmp_path, scan_ids=("sub-01_run-1",))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Connect4Dataset(
            root_dir=str(tmp_path),
            patch_size=(2, 2, 2),
            target_shape=(4, 4, 4),
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
            common_grid_contract_sha256="c" * 64,
            native_alignment_authority_sha256="d" * 64,
        )


def test_radiomics_and_normative_context_are_subject_specific(tmp_path):
    dataset, _ = _dataset(tmp_path)
    first = dataset.radiomics_for_scan("sub-01_run-1")[2]
    second = dataset.radiomics_for_scan("sub-02_run-1")[2]
    assert not np.allclose(first.numpy(), second.numpy())
    assert len(dataset.normative_source_rows["sub-01_run-1"]) == 23
    assert (
        "demographic-matched normative volume"
        in dataset.normative_index[("sub-01_run-1", "left_hippocampus")]
    )
    assert dataset._build_roi_embeddings("sub-01_run-1").shape == (32, 515)


def test_production_structural_sidecars_are_bound_into_source_fingerprint(tmp_path):
    sources, scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    # Sealed structural/cache validation cannot depend on target bytes or on a
    # BOLD-bearing authority record. The independently signed structural v2
    # authority remains sufficient.
    sources["source_acquisition"]["raw_bold"].unlink()
    sources["source_acquisition"]["identity_path"].write_bytes(
        b"sealed BOLD-bearing identity: must never be opened"
    )
    dataset = _production_dataset(tmp_path, sources, common_grid_sha256)
    first = dataset.source_fingerprint(scan_id)
    identity = first["t1_preprocessing_identity"]
    assert first["t1_preprocessing_sidecar_sha256"] == sha256_file(sidecar_path)
    assert identity["format"] == Connect4Dataset.T1_PREPROCESSING_IDENTITY_SCHEMA
    assert identity["schema"] == Connect4Dataset.T1_PREPROCESSING_SCHEMA
    assert identity["segmentation_schema"] == (Connect4Dataset.SEG_PREPROCESSING_SCHEMA)
    assert identity["output_sha256"] == first["t1w"]
    assert identity["segmentation_sha256"] == first["segmentation"]
    assert (
        first["structural_source_identity"] == sources["source_acquisition"]["binding"]
    )
    assert identity["common_grid_contract_sha256"] == common_grid_sha256
    assert identity["architecture_shape"] == [4, 4, 4]
    assert identity["intensity_normalization"] == (
        Connect4Dataset.T1_INTENSITY_NORMALIZATION
    )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_path.write_text(
        json.dumps(sidecar, separators=(",", ":")), encoding="utf-8"
    )
    second = dataset.source_fingerprint(scan_id)
    assert second["t1_preprocessing_sidecar_sha256"] == sha256_file(sidecar_path)
    assert (
        second["t1_preprocessing_sidecar_sha256"]
        != first["t1_preprocessing_sidecar_sha256"]
    )
    assert (
        second["t1_preprocessing_identity"]["fingerprint_sha256"]
        != identity["fingerprint_sha256"]
    )
    assert (
        second["t1_preprocessing_identity"]["t1_sidecar_canonical_sha256"]
        == (identity["t1_sidecar_canonical_sha256"])
    )
    assert canonical_sha256(second) != canonical_sha256(first)

    sidecar["schema"] = "connect4-t1-common-grid-v1"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema is not"):
        dataset.source_fingerprint(scan_id)


def test_bold_bearing_v1_structural_binding_is_rejected_without_target_access(
    tmp_path,
):
    sources, scan_id, _common_grid_sha256, _sidecar_path = (
        _write_production_t1_fixture(tmp_path)
    )
    legacy = dict(sources["source_acquisition"]["binding"])
    legacy["schema"] = "connect4-structural-source-binding-v1"
    legacy["raw_bold_sha256"] = "0" * 64
    sources["source_acquisition"]["raw_bold"].unlink()
    sources["source_acquisition"]["identity_path"].write_bytes(
        b"sealed target sentinel"
    )
    with pytest.raises(RuntimeError, match="structural source-binding fields differ"):
        load_structural_source_binding(legacy, expected_scan_id=scan_id)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "connect4-t1-common-grid-v1", "schema is not"),
        ("output_sha256", "0" * 64, "does not bind the T1 NIfTI"),
        (
            "common_grid_contract_sha256",
            "d" * 64,
            "common-grid SHA-256 differs",
        ),
        ("architecture_shape", [4, 4, 8], "architecture_shape differs"),
        (
            "versioned_recovery_choices",
            {
                "common_grid_schema": "connect4-common-grid-v1",
                "intensity_normalization": "whole-volume z-score",
            },
            "intensity normalization is not",
        ),
    ],
)
def test_production_t1_rejects_forged_or_superseded_sidecar(
    tmp_path, field, value, error
):
    sources, _scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = value
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match=error):
        _production_dataset(tmp_path, sources, common_grid_sha256)


def test_production_t1_rejects_missing_sidecar_but_synthetic_path_remains(tmp_path):
    sources = write_structural_sources(tmp_path, scan_ids=("sub-01_run-1",))
    synthetic = Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
    )
    assert len(synthetic) == 1
    with pytest.raises(RuntimeError, match="production T1 sidecar is missing"):
        _production_dataset(tmp_path, sources, "c" * 64)


def test_production_t1_rejects_stale_sidecar_after_nifti_mutation(tmp_path):
    sources, scan_id, common_grid_sha256, _sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    t1_path = tmp_path / "T1" / f"{scan_id}_T1.nii.gz"
    image = nib.load(str(t1_path))
    values = image.get_fdata(dtype=np.float32)
    values[1, 0, 0] += np.float32(0.125)
    nib.save(nib.Nifti1Image(values, image.affine, image.header), t1_path)
    with pytest.raises(RuntimeError, match="does not bind the T1 NIfTI"):
        _production_dataset(tmp_path, sources, common_grid_sha256)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("out-of-range", r"outside the required \[-3,3\] range"),
        ("nonzero-exterior", "exterior is not exactly zero"),
        ("nonfinite", "contains NaN or infinity"),
    ],
)
def test_production_t1_directly_validates_values_and_segmentation_support(
    tmp_path, mutation, error
):
    sources, scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    t1_path = tmp_path / "T1" / f"{scan_id}_T1.nii.gz"
    image = nib.load(str(t1_path))
    values = image.get_fdata(dtype=np.float32)
    if mutation == "out-of-range":
        values[1, 0, 0] = 3.01
    elif mutation == "nonzero-exterior":
        values[0, 0, 0] = 0.25
    else:
        values[1, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(values, image.affine, image.header), t1_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output_sha256"] = sha256_file(t1_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match=error):
        _production_dataset(tmp_path, sources, common_grid_sha256)


@pytest.mark.parametrize(
    "artifact", ["sidecar", "segmentation_sidecar", "t1", "segmentation"]
)
def test_production_t1_rejects_symbolic_link_artifacts(tmp_path, artifact):
    sources, scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    dataset = _production_dataset(tmp_path, sources, common_grid_sha256)
    paths = {
        "sidecar": sidecar_path,
        "segmentation_sidecar": tmp_path / "Masks" / f"{scan_id}_mask.json",
        "t1": tmp_path / "T1" / f"{scan_id}_T1.nii.gz",
        "segmentation": tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    }
    path = paths[artifact]
    original = path.with_name(f"original-{path.name}")
    path.rename(original)
    path.symlink_to(original.name)
    with pytest.raises(RuntimeError, match="must not be a symbolic link"):
        dataset.source_fingerprint(scan_id)


@pytest.mark.parametrize(
    "artifact", ["sidecar", "segmentation_sidecar", "t1", "segmentation"]
)
def test_production_t1_rejects_hard_link_artifacts(tmp_path, artifact):
    sources, scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    dataset = _production_dataset(tmp_path, sources, common_grid_sha256)
    paths = {
        "sidecar": sidecar_path,
        "segmentation_sidecar": tmp_path / "Masks" / f"{scan_id}_mask.json",
        "t1": tmp_path / "T1" / f"{scan_id}_T1.nii.gz",
        "segmentation": tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    }
    path = paths[artifact]
    os.link(path, path.with_name(f"hardlink-{path.name}"))
    with pytest.raises(RuntimeError, match="must not have hard links"):
        dataset.source_fingerprint(scan_id)


@pytest.mark.parametrize(
    "artifact", ["sidecar", "segmentation_sidecar", "t1", "segmentation"]
)
def test_production_t1_rejects_concurrent_path_substitution(
    tmp_path, monkeypatch, artifact
):
    sources, scan_id, common_grid_sha256, sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    dataset = _production_dataset(tmp_path, sources, common_grid_sha256)
    paths = {
        "sidecar": sidecar_path,
        "segmentation_sidecar": tmp_path / "Masks" / f"{scan_id}_mask.json",
        "t1": tmp_path / "T1" / f"{scan_id}_T1.nii.gz",
        "segmentation": tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    }
    path = paths[artifact]
    replacement = path.with_name(f"replacement-{path.name}")
    if artifact in {"sidecar", "segmentation_sidecar"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["audit_note"] = "adversarial descriptor/path substitution"
        replacement.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        shutil.copyfile(path, replacement)

    target = os.lstat(path)
    target_inode = (target.st_dev, target.st_ino)
    original_read = dataset_module.os.read
    substituted = False

    def substituting_read(descriptor, count):
        nonlocal substituted
        chunk = original_read(descriptor, count)
        opened = dataset_module.os.fstat(descriptor)
        if not substituted and (opened.st_dev, opened.st_ino) == target_inode:
            os.replace(replacement, path)
            substituted = True
        return chunk

    monkeypatch.setattr(dataset_module.os, "read", substituting_read)
    with pytest.raises(RuntimeError, match="changed during production admission"):
        dataset.source_fingerprint(scan_id)
    assert substituted is True


@pytest.mark.parametrize(
    ("source_key", "error"),
    [
        ("raw_t1", "structural-authority raw T1w bytes differ"),
        ("raw_synthseg", "structural-authority SynthSeg mask bytes differ"),
    ],
)
def test_production_structural_admission_rehashes_exact_raw_sources(
    tmp_path, source_key, error
):
    sources, _scan_id, common_grid_sha256, _sidecar_path = _write_production_t1_fixture(
        tmp_path
    )
    source_path = sources["source_acquisition"][source_key]
    with source_path.open("ab") as stream:
        stream.write(b"post-signing-source-substitution")
    with pytest.raises(RuntimeError, match=error):
        _production_dataset(tmp_path, sources, common_grid_sha256)


def test_production_structural_admission_rejects_cross_scan_source_identity(
    tmp_path,
):
    sources, scan_id, common_grid_sha256, t1_sidecar_path = (
        _write_production_t1_fixture(tmp_path)
    )
    other = write_source_acquisition_fixture(
        tmp_path / "other",
        scan_id="sub-02_run-1",
        t1_source=tmp_path / "T1" / f"{scan_id}_T1.nii.gz",
        synthseg_source=tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    )
    segmentation_sidecar_path = tmp_path / "Masks" / f"{scan_id}_mask.json"
    segmentation_sidecar = json.loads(
        segmentation_sidecar_path.read_text(encoding="utf-8")
    )
    segmentation_sidecar["structural_source"] = other["binding"]
    segmentation_sidecar["source_sha256"] = other["binding"]["synthseg_mask_sha256"]
    segmentation_sidecar_path.write_text(
        json.dumps(segmentation_sidecar, indent=2), encoding="utf-8"
    )
    t1_sidecar = json.loads(t1_sidecar_path.read_text(encoding="utf-8"))
    t1_sidecar["structural_source"] = other["binding"]
    t1_sidecar["source_sha256"] = other["binding"]["raw_t1_sha256"]
    t1_sidecar["segmentation_sidecar_sha256"] = sha256_file(segmentation_sidecar_path)
    t1_sidecar_path.write_text(json.dumps(t1_sidecar, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identifies another scan"):
        _production_dataset(tmp_path, sources, common_grid_sha256)


def test_roi_feature_conditioning_is_shared_and_rejects_anatcl_drift(tmp_path):
    dataset, sources = _dataset(tmp_path)
    first = dataset.roi_feature_extractor_identity("sub-01_run-1")
    second = dataset.roi_feature_extractor_identity("sub-02_run-1")
    assert first == second
    first_artifact = dataset.roi_feature_artifact_identity("sub-01_run-1")
    second_artifact = dataset.roi_feature_artifact_identity("sub-02_run-1")
    assert first_artifact != second_artifact
    assert first_artifact["cat12_input"] != second_artifact["cat12_input"]
    assert first["concatenation"] == {
        "order": ["anatcl", "pyradiomics"],
        "output_dim": 515,
    }
    assert len(first["roi_specs"]) == 32

    path = tmp_path / "AnatCL" / "sub-02_run-1" / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["model"]["architecture"] = "unrecorded-architecture"
    path.write_text(json.dumps(provenance))
    changed = Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
    )
    with pytest.raises(RuntimeError, match="model/extraction provenance is invalid"):
        changed.roi_feature_extractor_identity("sub-02_run-1")


def test_external_cat12_authority_may_differ_while_protocol_stays_identical(tmp_path):
    training, _ = _dataset(tmp_path / "training", scan_ids=("sub-train_run-1",))
    external, _ = _dataset(tmp_path / "external", scan_ids=("sub-external_run-1",))
    training_protocol = training.roi_feature_extractor_identity("sub-train_run-1")
    external_protocol = external.roi_feature_extractor_identity("sub-external_run-1")
    assert training_protocol == external_protocol
    assert (
        training.roi_feature_artifact_identity("sub-train_run-1")["cat12_input"][
            "authority_manifest_sha256"
        ]
        != external.roi_feature_artifact_identity("sub-external_run-1")["cat12_input"][
            "authority_manifest_sha256"
        ]
    )


def test_superseded_three_plane_anatcl_v1_cache_is_rejected(tmp_path):
    sources = write_structural_sources(tmp_path, scan_ids=("sub-01_run-1",))
    path = tmp_path / "AnatCL" / "sub-01_run-1" / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["schema"] = "connect4-anatcl-roi-features-v1"
    provenance["extraction"] = {
        "views": ["axis-0", "axis-1", "axis-2"],
        "input_size": [224, 224],
        "aggregation": "mean of three global descriptors",
    }
    path.write_text(json.dumps(provenance))
    dataset = Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
    )
    with pytest.raises(RuntimeError, match="model/extraction provenance is invalid"):
        dataset.roi_feature_extractor_identity("sub-01_run-1")


def test_normative_loader_rejects_arbitrary_description_only_csv(tmp_path):
    sources = write_structural_sources(tmp_path, scan_ids=("sub-01_run-1",))
    pd.DataFrame(
        [
            {
                "PatientID": "sub-01_run-1",
                "Structure": "Hippocampus L",
                "full_description": "trust me",
            }
        ]
    ).to_csv(sources["normative"], index=False)
    with pytest.raises(ValueError, match="raw measured volume"):
        Connect4Dataset(
            root_dir=str(tmp_path),
            patch_size=(2, 2, 2),
            target_shape=(4, 4, 4),
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
        )


def test_radiomics_requires_exact_scan_id_without_patient_prefix_fallback(tmp_path):
    scan_id = "sub-01_run-1"
    sources = write_structural_sources(tmp_path, scan_ids=(scan_id,))
    frame = pd.read_csv(tmp_path / "all_radiomics.csv")
    frame["scan_id"] = "sub-01"
    frame.to_csv(tmp_path / "all_radiomics.csv", index=False)
    with pytest.raises(ValueError, match="fallback is forbidden"):
        Connect4Dataset(
            root_dir=str(tmp_path),
            patch_size=(2, 2, 2),
            target_shape=(4, 4, 4),
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
        )


def test_normative_patient_rows_require_explicit_manifest_mapping(tmp_path):
    scan_id = "sub-01_run-1"
    patient_id = "participant-01"
    sources = write_structural_sources(tmp_path, scan_ids=(scan_id,))
    frame = pd.read_csv(sources["normative"])
    frame["PatientID"] = patient_id
    frame.to_csv(sources["normative"], index=False)
    with pytest.raises(ValueError, match="filename-prefix inference is forbidden"):
        Connect4Dataset(
            root_dir=str(tmp_path),
            patch_size=(2, 2, 2),
            target_shape=(4, 4, 4),
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
        )

    manifest = tmp_path / "cohort_manifest.csv"
    manifest.write_text(
        f"scan_id,patient_id,cohort\n{scan_id},{patient_id},A4\n",
        encoding="utf-8",
    )
    dataset = Connect4Dataset(
        root_dir=str(tmp_path),
        patch_size=(2, 2, 2),
        target_shape=(4, 4, 4),
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
        cohort_manifest_path=str(manifest),
    )
    assert dataset.scan_to_normative_subject == {scan_id: patient_id}


def test_anatcl_cache_is_bound_to_t1_and_mask_hashes(tmp_path):
    dataset, _ = _dataset(tmp_path, scan_ids=("sub-01_run-1",))
    provenance_path = tmp_path / "AnatCL" / "sub-01_run-1" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["source_sha256"]["t1w"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="stale"):
        dataset._load_anatcl_embeddings("sub-01_run-1")


def test_radiomics_csv_is_bound_to_extractor_and_source_hashes(tmp_path):
    sources = write_structural_sources(tmp_path, scan_ids=("sub-01_run-1",))
    frame = pd.read_csv(tmp_path / "all_radiomics.csv")
    frame.loc[0, "radiomics_feature_0"] += 1.0
    frame.to_csv(tmp_path / "all_radiomics.csv", index=False)
    with pytest.raises(RuntimeError, match="radiomics CSV hash mismatch"):
        Connect4Dataset(
            root_dir=str(tmp_path),
            patch_size=(2, 2, 2),
            target_shape=(4, 4, 4),
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
        )


def test_patch_center_is_foreground_center_of_mass_not_fixed_midpoint(tmp_path):
    dataset, _ = _dataset(tmp_path, scan_ids=("sub-01_run-1",))
    segmentation, affine = dataset._load_nifti(
        tmp_path / "Masks" / "sub-01_run-1_mask.nii.gz", is_mask=True
    )
    center = dataset._compute_patch_center_mm(0, segmentation, affine)
    # First 2x2x2 patch is all foreground. Voxel COM is (0.5,0.5,0.5),
    # therefore 1.5 mm on a 3 mm grid.
    assert center == pytest.approx((1.5, 1.5, 1.5))


class _WrongBatchBrainIAC(torch.nn.Module):
    def forward(self, patches):
        return torch.ones(1, 3, device=patches.device)


def _wrapper_without_external_checkpoint(model):
    wrapper = BrainIACWrapper.__new__(BrainIACWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.embed_dim = 3
    wrapper.device = "cpu"
    return wrapper


def test_brainiac_wrapper_rejects_duplicated_or_multichannel_patch_outputs():
    wrapper = _wrapper_without_external_checkpoint(_WrongBatchBrainIAC())
    with pytest.raises(RuntimeError, match="one configured embedding per patch"):
        wrapper.encode(torch.zeros(2, 1, 4, 4, 4))
    with pytest.raises(ValueError, match="single-channel"):
        wrapper.encode(torch.zeros(1, 2, 4, 4, 4))
