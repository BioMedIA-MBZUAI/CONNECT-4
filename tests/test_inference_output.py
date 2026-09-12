import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from inference import infer as inference_module
from inference import runtime as inference_runtime_module
from eval.visualize import save_fmri_nifti
from data.provenance import canonical_sha256, sha256_file
from data.protocol import (
    RUN_ARTIFACT_IDENTITY_SCHEMA,
    TrainingCohortEvidence,
    build_split_identity,
)
from data.split import PAPER_PROFILE, build_seeded_split_manifest
from inference.infer import POSTSEAL_EVALUATOR_GUIDANCE
from inference.runtime import (
    _generation_config,
    _inverse_pad_native_prediction,
    _require_prediction_domain,
    _target_blind_prediction_record,
    _target_blind_prediction_set_record,
    fixed_inference_indices,
    require_completed_training_checkpoint,
    require_structural_artifact_compatibility,
    validate_inference_mode,
    validate_subject_limit,
)


def test_inference_cli_directs_paired_work_to_postseal_evaluator():
    assert "--visualize --metrics" not in inference_module.__doc__
    assert "POSTSEAL_HELDOUT_EVALUATION.md" in POSTSEAL_EVALUATOR_GUIDANCE
    assert "will not open held-out targets" in POSTSEAL_EVALUATOR_GUIDANCE


def test_inference_entrypoint_exposes_target_blind_cli():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "inference.infer", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--config" in completed.stdout
    source_path = Path(inference_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    assert "from .runtime import" in source


def test_subject_limit_is_external_candidate_only():
    validate_subject_limit("external", None)
    validate_subject_limit("external", 1)
    validate_subject_limit("test", None)
    validate_subject_limit("all", None)

    for invalid in (True, False, 0, -1, 1.0, "1"):
        with pytest.raises(ValueError, match="positive integer"):
            validate_subject_limit("external", invalid)

    for sealed_split in ("test", "all"):
        with pytest.raises(ValueError, match="complete selected split"):
            validate_subject_limit(sealed_split, 1)


def test_target_blind_sealing_rejects_out_of_domain_without_clamping():
    valid = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 1, 1, 3)
    assert _require_prediction_domain(valid) is valid
    with pytest.raises(RuntimeError, match=r"\[0,1\].*refusing to clamp"):
        _require_prediction_domain(valid - 0.01)
    with pytest.raises(RuntimeError, match=r"\[0,1\].*refusing to clamp"):
        _require_prediction_domain(valid + 0.01)
    invalid = valid.clone()
    invalid[..., 1] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        _require_prediction_domain(invalid)


def test_v9_native_prediction_inverse_padding_is_exact_shape_affine_and_tr(tmp_path):
    native_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    native_affine[:3, 3] = (-90.0, -126.0, -72.0)
    before = np.asarray([1, 3, 1])
    padded_affine = native_affine.copy()
    padded_affine[:3, 3] -= native_affine[:3, :3] @ before
    binding = {
        "native_shape": [61, 73, 61],
        "architecture_shape": [64, 80, 64],
        "padding_before": [1, 3, 1],
        "padding_after": [2, 4, 2],
        "native_affine": native_affine.tolist(),
        "padded_affine": padded_affine.tolist(),
        "padding_mode": "constant-zero-no-interpolation",
        "interpolation_after_native_preprocessing": False,
    }
    prediction = torch.arange(
        2 * 64 * 80 * 64, dtype=torch.float32
    ).reshape(1, 1, 2, 64, 80, 64)
    cropped, affine = _inverse_pad_native_prediction(
        prediction, binding, padded_affine
    )
    assert cropped.shape == (1, 1, 2, 61, 73, 61)
    assert torch.equal(cropped, prediction[..., 1:62, 3:76, 1:62])
    np.testing.assert_array_equal(affine, native_affine)

    path = tmp_path / "sealed_native_prediction.nii.gz"
    save_fmri_nifti(cropped, str(path), affine=affine, repetition_time=3.0)
    image = nib.load(path)
    assert image.shape == (61, 73, 61, 2)
    np.testing.assert_array_equal(image.affine, native_affine)
    assert image.header.get_zooms() == pytest.approx((3.0, 3.0, 3.0, 3.0))

    changed_affine = padded_affine.copy()
    changed_affine[0, 3] += 3.0
    with pytest.raises(ValueError, match="signed padded affine"):
        _inverse_pad_native_prediction(prediction, binding, changed_affine)

    malformed = dict(binding)
    malformed["padding_after"] = [1, 4, 2]
    with pytest.raises(ValueError, match="reconstruct architecture_shape"):
        _inverse_pad_native_prediction(prediction, malformed, padded_affine)


def test_generation_compatibility_fingerprint_includes_sampling_seed():
    base = {
        "data": {"architecture_shape": [8, 8, 8]},
        "models": {"dit": {"num_inference_steps": 2}},
        "training": {"seed": 7},
    }
    changed = {
        "data": dict(base["data"]),
        "models": base["models"],
        "training": {"seed": 8},
    }
    assert _generation_config(base)["sampling_seed"] == 7
    assert _generation_config(base) != _generation_config(changed)

    profile_changed = {
        "data": {
            **base["data"],
            "protocol_profile": "a4-native-recovery-v1",
            "preprocessing_evidence_status": (
                "NON_CERTIFIED_RECOVERY_PREPROCESSING"
            ),
            "require_paper_preprocessing": False,
            "native_alignment_authority_sha256": "a" * 64,
        },
        "models": base["models"],
        "training": base["training"],
    }
    assert _generation_config(base) != _generation_config(profile_changed)


def test_synthetic_nifti_preserves_anatomical_grid_and_paper_tr(tmp_path):
    volume = torch.zeros((1, 1, 7, 4, 5, 6), dtype=torch.float32)
    affine = np.array(
        [
            [3.0, 0.0, 0.0, -30.0],
            [0.0, 3.0, 0.0, -40.0],
            [0.0, 0.0, 3.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "synthetic.nii.gz"
    save_fmri_nifti(volume, str(path), affine=affine, repetition_time=3.0)

    image = nib.load(path)
    assert image.shape == (4, 5, 6, 7)
    assert np.allclose(image.affine, affine)
    assert image.header.get_zooms() == pytest.approx((3.0, 3.0, 3.0, 3.0))
    assert image.header.get_xyzt_units() == ("mm", "sec")


def test_synthetic_nifti_refuses_identity_affine_guess(tmp_path):
    with pytest.raises(ValueError, match="reference affine"):
        save_fmri_nifti(
            torch.zeros((1, 1, 2, 2, 2, 2)),
            str(tmp_path / "invalid.nii.gz"),
        )


def test_metric_inference_uses_complete_persisted_test_split(tmp_path):
    scans = [f"P{index:02d}_001" for index in range(10)]
    cohorts = {
        scan: ("A4" if index < 5 else "ADNI")
        for index, scan in enumerate(scans)
    }
    patients = {scan: scan.rsplit("_", 1)[0] for scan in scans}
    manifest = tmp_path / "split.json"
    build_seeded_split_manifest(
        scans,
        patients,
        output_path=manifest,
        cohort_manifest_sha256="a" * 64,
        val_frac=0.2,
        test_frac=0.2,
        seed=7,
    )
    payload = json.loads(manifest.read_text())
    role_by_scan = {
        record["scan_id"]: role
        for role, records in payload["partitions"].items()
        for record in records
    }
    train = [index for index, scan in enumerate(scans) if role_by_scan[scan] == "train"]
    validation = [
        index
        for index, scan in enumerate(scans)
        if role_by_scan[scan] == "development_validation"
    ]
    test = [
        index for index, scan in enumerate(scans) if role_by_scan[scan] == "sealed_test"
    ]
    identity = build_split_identity(
        scans, train, validation, test, cohorts, patients
    )
    config = {
        "training": {
            "val_frac": 0.2,
            "test_frac": 0.2,
            "seed": 7,
            "split_manifest": str(manifest),
            "split_manifest_sha256": sha256_file(manifest),
            "split_manifest_sha256_env": "",
        },
        "data": {"protocol_profile": PAPER_PROFILE},
    }
    selected, name = fixed_inference_indices(
        scans,
        config,
        TrainingCohortEvidence(cohorts, patients),
        requested_split=None,
        metrics=True,
        checkpoint_split_identity=identity,
    )
    assert name == "test"
    assert selected == test


def test_metric_inference_rejects_all_split_or_checkpoint_split_drift(tmp_path):
    scans = [f"P{index:02d}_001" for index in range(10)]
    cohorts = {
        scan: ("A4" if index < 5 else "ADNI")
        for index, scan in enumerate(scans)
    }
    patients = {scan: scan.rsplit("_", 1)[0] for scan in scans}
    manifest = tmp_path / "split.json"
    build_seeded_split_manifest(
        scans,
        patients,
        output_path=manifest,
        cohort_manifest_sha256="a" * 64,
        val_frac=0.2,
        test_frac=0.2,
    )
    config = {
        "training": {
            "val_frac": 0.2,
            "test_frac": 0.2,
            "seed": 42,
            "split_manifest": str(manifest),
            "split_manifest_sha256": sha256_file(manifest),
            "split_manifest_sha256_env": "",
        },
        "data": {"protocol_profile": PAPER_PROFILE},
    }
    with pytest.raises(ValueError, match="only on the fixed disjoint test"):
        fixed_inference_indices(
            scans,
            config,
            TrainingCohortEvidence(cohorts, patients),
            requested_split="all",
            metrics=True,
            checkpoint_split_identity=None,
        )
    with pytest.raises(RuntimeError, match="identity does not match"):
        fixed_inference_indices(
            scans,
            config,
            TrainingCohortEvidence(cohorts, patients),
            requested_split="test",
            metrics=False,
            checkpoint_split_identity={"sha256": "wrong"},
        )


def test_test_inference_rejects_partial_or_intermediate_checkpoint(monkeypatch):
    complete = {
        "config": {"training": {"epochs": 200}},
        "partial": False,
        "next_batch_index": 0,
        "next_epoch": 200,
    }
    with pytest.raises(RuntimeError, match="development-QA selectable"):
        require_completed_training_checkpoint(complete)
    monkeypatch.setattr(
        inference_runtime_module,
        "validate_selectable_checkpoint_development_qa",
        lambda _checkpoint: {},
    )
    require_completed_training_checkpoint(complete)
    for changed_field, value in (
        ("partial", True),
        ("next_batch_index", 4),
        ("next_epoch", 199),
    ):
        checkpoint = dict(complete)
        checkpoint[changed_field] = value
        with pytest.raises(RuntimeError, match="completed-run checkpoint"):
            require_completed_training_checkpoint(checkpoint)


def test_external_inference_is_strictly_structural_only():
    assert validate_inference_mode(
        "external", metrics=False, visualize=False
    ) == "external"
    with pytest.raises(ValueError, match="structural-only"):
        validate_inference_mode("external", metrics=True, visualize=False)
    with pytest.raises(ValueError, match="structural-only"):
        validate_inference_mode("external", metrics=False, visualize=True)


def test_visualization_defaults_to_fixed_test_and_rejects_non_test():
    assert validate_inference_mode(
        None, metrics=False, visualize=True
    ) == "test"
    with pytest.raises(ValueError, match="only after sealing"):
        validate_inference_mode("all", metrics=False, visualize=True)


def test_structural_compatibility_ignores_only_target_bound_fields():
    spatial = {
        "profile": "non-certified-native-stage-b",
        "native_alignment_authority_sha256": "a" * 64,
    }
    base = {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "num_scans": 3,
        "scaler_identity_sha256": "b" * 64,
        "conditioning_identity": {"format": "conditioning"},
        "conditioning_identity_sha256": "c" * 64,
        "common_grid_contract_sha256": None,
        "native_alignment_authority_sha256": "a" * 64,
        "spatial_authority": spatial,
        "spatial_authority_sha256": canonical_sha256(spatial),
        "structural_artifact_identities_sha256": "1" * 64,
        "target_artifact_identities_sha256": "d" * 64,
        "target_access_contract": {"authenticated_target_scan_ids": ["scan-a"]},
        "sha256": "e" * 64,
    }
    target_blind = {
        **base,
        "target_artifact_identities_sha256": canonical_sha256({}),
        "target_access_contract": {"authenticated_target_scan_ids": []},
        "sha256": "f" * 64,
    }
    require_structural_artifact_compatibility(base, target_blind)

    changed = dict(target_blind)
    changed["native_alignment_authority_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="structural/conditioning artifacts"):
        require_structural_artifact_compatibility(base, changed)

    changed_cache = dict(target_blind)
    changed_cache["structural_artifact_identities_sha256"] = "8" * 64
    with pytest.raises(RuntimeError, match="structural/conditioning artifacts"):
        require_structural_artifact_compatibility(base, changed_cache)

    stale = dict(target_blind)
    stale["format"] = "connect4_run_artifacts_v4"
    with pytest.raises(RuntimeError, match="stale run-artifact identity"):
        require_structural_artifact_compatibility(base, stale)

    missing_root = dict(target_blind)
    missing_root.pop("structural_artifact_identities_sha256")
    with pytest.raises(RuntimeError, match="target-independent structural root"):
        require_structural_artifact_compatibility(base, missing_root)


def test_prediction_seal_binds_outputs_without_target_access(tmp_path):
    prediction_path = tmp_path / "prediction.nii.gz"
    save_fmri_nifti(
        torch.zeros((1, 1, 128, 4, 5, 6)),
        str(prediction_path),
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        repetition_time=3.0,
    )
    artifact_identity = {
        "sha256": "a" * 64,
        "target_artifact_identities_sha256": "b" * 64,
        "spatial_authority_sha256": "c" * 64,
    }
    record = _target_blind_prediction_record(
        scan_id="scan-001",
        prediction_path=prediction_path,
        prediction_shape=(4, 5, 6, 128),
        checkpoint_sha256="d" * 64,
        checkpoint_artifact_identity=artifact_identity,
        protocol_profile="a4-native-recovery-v1",
    )
    assert record["target_opened_before_prediction"] is False
    assert record["training_target_artifact_identities_sha256"] == "b" * 64
    assert record["record_sha256"] == canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )

    sealed = _target_blind_prediction_set_record(
        split_name="test",
        subject_records=[record],
        expected_scan_ids=["scan-001"],
        checkpoint_sha256="d" * 64,
        checkpoint_artifact_identity=artifact_identity,
    )
    assert sealed["sealed_targets_opened"] is False
    assert sealed["subjects"][0]["prediction_sha256"] == record["prediction"][
        "sha256"
    ]

    tampered = dict(record)
    tampered["target_opened_before_prediction"] = True
    with pytest.raises(ValueError, match="authenticated records"):
        _target_blind_prediction_set_record(
            split_name="test",
            subject_records=[tampered],
            expected_scan_ids=["scan-001"],
            checkpoint_sha256="d" * 64,
            checkpoint_artifact_identity=artifact_identity,
        )

    with pytest.raises(ValueError, match="authenticated records"):
        _target_blind_prediction_set_record(
            split_name="test",
            subject_records=[record],
            expected_scan_ids=["scan-001", "scan-002"],
            checkpoint_sha256="d" * 64,
            checkpoint_artifact_identity=artifact_identity,
        )

    with pytest.raises(ValueError, match="unique expected scan IDs"):
        _target_blind_prediction_set_record(
            split_name="test",
            subject_records=[record],
            expected_scan_ids=["scan-001", "scan-001"],
            checkpoint_sha256="d" * 64,
            checkpoint_artifact_identity=artifact_identity,
        )


def test_prediction_record_rejects_unsafe_scan_id_before_path_binding(tmp_path):
    prediction_path = tmp_path / "prediction.nii.gz"
    prediction_path.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="unsafe inference scan ID"):
        _target_blind_prediction_record(
            scan_id="..",
            prediction_path=prediction_path,
            prediction_shape=(4, 5, 6, 128),
            checkpoint_sha256="d" * 64,
            checkpoint_artifact_identity={},
            protocol_profile="a4-native-recovery-v1",
        )
