from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from data.provenance import canonical_sha256, sha256_file
from models.brainlm_context import (
    A424_AFFINE_RAS_MM,
    A424_SHAPE,
    BRAINLM_AUTHORITY_SCHEMA,
    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    NATIVE_SHAPE,
    OFFICIAL_A424_ATLAS_SHA256,
    load_brainlm_authority,
)


def _builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_brainlm_mni_authority_v2.py"
    specification = importlib.util.spec_from_file_location(
        "_test_brainlm_authority_builder", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


builder = _builder_module()


def _signed(value: dict) -> dict:
    result = dict(value)
    result["record_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def _artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _registration_mapping() -> dict:
    return {
        "schema": builder.REGISTRATION_MAPPING_SCHEMA,
        "mapping_kind": "dense_displacement_pull_field",
        "domain": "official A424 canonical RAS grid",
        "domain_shape": list(A424_SHAPE),
        "domain_affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "domain_atlas_sha256": OFFICIAL_A424_ATLAS_SHA256,
        "range": "prepared structural common-grid RAS world millimetres",
        "range_grid_shape": list(NATIVE_SHAPE),
        "range_grid_affine_ras_mm": builder.NATIVE_AFFINE_RAS_MM.tolist(),
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
        "pytorch_grid_sample_contract": {
            "align_corners": True,
            "grid_component_order": ["x", "y", "z"],
            "input_tensor_axis_order": ["batch", "channel", "z", "y", "x"],
            "mode": "bilinear (trilinear for 5D input)",
            "padding_mode": "zeros",
            "voxel_to_normalized_equation": (
                "normalized = 2 * voxel / (size - 1) - 1"
            ),
        },
    }


@pytest.fixture()
def authority_inputs(tmp_path: Path):
    scan_id = "B100_001"
    role = "train"
    registration_builder_sha = "7" * 64

    t1_path = tmp_path / "t1_common3mm.nii.gz"
    mask_path = tmp_path / "mask_common3mm.nii.gz"
    displacement_path = tmp_path / "a424_displacement.nii.gz"
    qc_path = tmp_path / "registration_qc.png"

    t1 = np.zeros(NATIVE_SHAPE, dtype=np.float32)
    t1[10:50, 10:60, 10:50] = 1.0
    mask = np.zeros(NATIVE_SHAPE, dtype=np.int16)
    mask[10:50, 10:60, 10:50] = 1
    nib.save(nib.Nifti1Image(t1, builder.NATIVE_AFFINE_RAS_MM), t1_path)
    nib.save(nib.Nifti1Image(mask, builder.NATIVE_AFFINE_RAS_MM), mask_path)

    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header.set_intent("vector")
    displacement_image = nib.Nifti1Image(
        np.zeros((*A424_SHAPE, 3), dtype=np.float32),
        A424_AFFINE_RAS_MM,
        header,
    )
    displacement_image.set_qform(A424_AFFINE_RAS_MM, code=4)
    displacement_image.set_sform(A424_AFFINE_RAS_MM, code=4)
    nib.save(displacement_image, displacement_path)
    Image.new("RGB", (24, 24), color=(30, 60, 90)).save(qc_path)

    t1_artifact = _artifact(t1_path)
    t1_artifact["canonical_geometry"] = {
        "shape": list(NATIVE_SHAPE),
        "affine": builder.NATIVE_AFFINE_RAS_MM.tolist(),
    }
    mask_artifact = _artifact(mask_path)
    mask_artifact["canonical_geometry"] = {
        "shape": list(NATIVE_SHAPE),
        "affine": builder.NATIVE_AFFINE_RAS_MM.tolist(),
    }

    native_provenance = _signed(
        {
            "schema": builder.NATIVE_SCAN_SCHEMA,
            "scan_id": scan_id,
            "role": role,
            "paper_certified": False,
            "implementation": {
                "source_sha256": CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            },
            "outputs": {
                "t1_common": dict(t1_artifact),
                "mask_common": dict(mask_artifact),
            },
            "target_access_contract": {"sealed_test_opened": False},
        }
    )
    provenance_path = tmp_path / "preprocessing_provenance.json"
    provenance_sha = _write_json(provenance_path, native_provenance)
    provenance_evidence = {
        "path": str(provenance_path.resolve()),
        "sha256": provenance_sha,
        "size_bytes": provenance_path.stat().st_size,
        "record_sha256": native_provenance["record_sha256"],
    }

    displacement_artifact = {
        **_artifact(displacement_path),
        "shape": [*A424_SHAPE, 3],
        "affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "dtype": "float32",
        "nifti_intent": "vector",
        "component_order": ["right", "anterior", "superior"],
        "runtime_mapping_schema": builder.REGISTRATION_MAPPING_SCHEMA,
        "stored_values_exactly_verified": True,
    }
    qc_artifact = _artifact(qc_path)
    scan = _signed(
        {
            "schema": builder.REGISTRATION_SCAN_SCHEMA,
            "scan_id": scan_id,
            "role": role,
            "structural_only": True,
            "fMRI_voxel_data_read": False,
            "prediction_or_target_data_read": False,
            "lineage": {
                "fMRI_voxel_data_read": False,
                "prediction_or_target_data_read": False,
                "prepared_t1": t1_artifact,
                "prepared_structural_mask": mask_artifact,
                "preprocessing_provenance": provenance_evidence,
            },
            "transform": {
                "representation": (
                    "nonlinear dense pull mapping; no lossless 4x4 representation"
                ),
                "semantic_direction": (
                    "MNI-template RAS world -> prepared-source RAS world"
                ),
                "estimated_from_t1_intensity": True,
                "identity_inferred_from_nifti_header": False,
                "affine_only_runtime_authorized": False,
                "runtime_application_contract": _registration_mapping(),
                "artifacts": {
                    "a424_mni_to_prepared_displacement_ras_mm": (
                        displacement_artifact
                    ),
                    "structural_registration_visual_qc_png": qc_artifact,
                },
            },
            "qc": {
                "passed": True,
                "visual_review": {
                    "artifact_key": "structural_registration_visual_qc_png",
                    "automatic_visual_pass_claimed": False,
                    "status": "PENDING_HUMAN_REVIEW",
                },
            },
        }
    )
    ordered = [scan_id]
    registration = _signed(
        {
            "schema": builder.REGISTRATION_SCHEMA,
            "role": role,
            "structural_only": True,
            "sealed_target_voxel_data_opened": False,
            "fMRI_voxel_files_opened": 0,
            "fMRI_voxel_files_hashed": 0,
            "prediction_or_target_files_opened": 0,
            "source_ordered_scan_ids": ordered,
            "source_ordered_scan_ids_sha256": canonical_sha256(ordered),
            "ordered_scan_ids_sha256": canonical_sha256(ordered),
            "source_scan_count": 1,
            "scan_count": 1,
            "upstream": {
                "input_kind": "native-v3-structural-projection",
                "role": role,
                "native_source_sha256": (
                    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
                ),
                "ordered_scan_ids": ordered,
                "ordered_scan_ids_sha256": canonical_sha256(ordered),
            },
            "implementation": {
                "source": {"sha256": registration_builder_sha},
                "executed_private_source_snapshot": {
                    "sha256": registration_builder_sha
                },
                "source_copy_hash_identical": True,
            },
            "a424_atlas": {
                "sha256": OFFICIAL_A424_ATLAS_SHA256,
                "canonical_geometry": {
                    "shape": list(A424_SHAPE),
                    "affine": A424_AFFINE_RAS_MM.tolist(),
                },
            },
            "downstream_authorization": {
                "status": "EVIDENCE_ONLY_NOT_YET_BRAINLM_RUNTIME_AUTHORITY",
                "requires_explicit_human_review_wrapper": True,
                "affine_only_consumer_forbidden": True,
            },
            "scans": [scan],
        }
    )
    registration_path = tmp_path / "registration.json"
    registration_sha = _write_json(registration_path, registration)
    descriptor = {
        "schema": builder.REGISTRATION_SCHEMA,
        "sha256": registration_sha,
        "record_sha256": registration["record_sha256"],
        "role": role,
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "registration_builder_source_sha256": registration_builder_sha,
    }
    review_scan = _signed(
        {
            "schema": builder.HUMAN_REVIEW_SCAN_SCHEMA,
            "scan_id": scan_id,
            "role": role,
            "decision": "PASS",
            "registration_scan_record_sha256": scan["record_sha256"],
            "visual_qc_png_sha256": qc_artifact["sha256"],
            "a424_dense_displacement_sha256": displacement_artifact["sha256"],
            "gross_misalignment_absent": True,
            "left_right_flip_absent": True,
            "non_anatomical_warp_absent": True,
            "brain_coverage_acceptable": True,
        }
    )
    review = _signed(
        {
            "schema": builder.HUMAN_REVIEW_SCHEMA,
            "decision": "PASS",
            "structural_only": True,
            "prediction_or_functional_target_used": False,
            "authority_builder_source_sha256": sha256_file(Path(builder.__file__)),
            "registration_manifests": [descriptor],
            "scan_count": 1,
            "ordered_scan_ids": ordered,
            "ordered_scan_ids_sha256": canonical_sha256(ordered),
            "review_protocol": builder._expected_review_protocol(),
            "reviewer": {
                "reviewer_id": "reviewer-001",
                "qualification": "neuroimaging registration reviewer",
                "organization": "CONNECT-4 recovery review",
            },
            "review_timestamp_utc": "2026-09-01T18:00:00Z",
            "scans": [review_scan],
        }
    )
    review_path = tmp_path / "review.json"
    review_sha = _write_json(review_path, review)
    return {
        "scan_id": scan_id,
        "registration": registration,
        "registration_path": registration_path,
        "registration_sha": registration_sha,
        "registration_builder_sha": registration_builder_sha,
        "review": review,
        "review_path": review_path,
        "review_sha": review_sha,
        "displacement_path": displacement_path,
    }


def _build(inputs: dict, output: Path):
    return builder.build_authority(
        registration_manifests=[inputs["registration_path"]],
        registration_manifest_sha256=[inputs["registration_sha"]],
        registration_builder_sha256=inputs["registration_builder_sha"],
        human_review=inputs["review_path"],
        human_review_sha256=inputs["review_sha"],
        output=output,
    )


def test_builds_runtime_loadable_current_ec46_authority(authority_inputs, tmp_path):
    output = tmp_path / "brainlm_authority_v2.json"
    path, digest = _build(authority_inputs, output)
    assert path == output.resolve()
    assert digest == sha256_file(output)
    records, identity = load_brainlm_authority(output, expected_sha256=digest)
    assert set(records) == {authority_inputs["scan_id"]}
    assert identity["schema"] == BRAINLM_AUTHORITY_SCHEMA
    assert identity["native_preprocessing_source_sha256"] == (
        CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
    )
    assert records[authority_inputs["scan_id"]]["human_review"]["decision"] == "PASS"
    assert records[authority_inputs["scan_id"]]["support_foreground_voxels"] == (
        40 * 50 * 40
    )
    assert records[authority_inputs["scan_id"]][
        "a424_dense_displacement_artifact"
    ]["runtime_mapping_schema"] == builder.BRAINLM_RUNTIME_MAPPING_SCHEMA


def test_runtime_loader_rejects_nonpositive_support_count(authority_inputs, tmp_path):
    output = tmp_path / "brainlm_authority_v2.json"
    _path, _digest = _build(authority_inputs, output)
    authority = json.loads(output.read_text(encoding="utf-8"))
    authority["scans"][0]["support_foreground_voxels"] = 0
    authority["scans"][0].pop("record_sha256")
    authority["scans"][0] = _signed(authority["scans"][0])
    authority.pop("record_sha256")
    authority = _signed(authority)
    invalid = tmp_path / "invalid-support-authority.json"
    digest = _write_json(invalid, authority)
    with pytest.raises(RuntimeError, match="scan contract differs"):
        load_brainlm_authority(invalid, expected_sha256=digest)


def test_rejects_historical_native_registration_even_when_resigned(
    authority_inputs, tmp_path
):
    historical = "8beee80eddc3da357229fdbc8ba57b7eab286cbdb34bf2941490f534372e631a"
    registration = dict(authority_inputs["registration"])
    registration["upstream"] = dict(registration["upstream"])
    registration["upstream"]["native_source_sha256"] = historical
    registration.pop("record_sha256")
    registration = _signed(registration)
    path = tmp_path / "historical_registration.json"
    digest = _write_json(path, registration)
    inputs = dict(authority_inputs, registration_path=path, registration_sha=digest)
    with pytest.raises(RuntimeError, match="current-ec46 registration"):
        _build(inputs, tmp_path / "must_not_exist.json")


def test_rejects_review_decision_or_dense_field_byte_drift(authority_inputs, tmp_path):
    review = dict(authority_inputs["review"])
    review["decision"] = "FAIL"
    review.pop("record_sha256")
    review = _signed(review)
    review_path = tmp_path / "failed_review.json"
    review_sha = _write_json(review_path, review)
    changed = dict(authority_inputs, review_path=review_path, review_sha=review_sha)
    with pytest.raises(RuntimeError, match="human-review contract"):
        _build(changed, tmp_path / "review_must_not_exist.json")

    authority_inputs["displacement_path"].write_bytes(
        authority_inputs["displacement_path"].read_bytes() + b"drift"
    )
    with pytest.raises(
        RuntimeError, match="SHA-256/size|not the exact expected regular file"
    ):
        _build(authority_inputs, tmp_path / "drift_must_not_exist.json")
