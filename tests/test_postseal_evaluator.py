from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
import pytest
from scipy import ndimage

from eval import postseal
from eval import postseal_metrics
from eval import quality as paired_quality
from scripts import visualize_4d_comparison as visualizer
from scripts import visualize_texture_audit as texture_audit
from eval import postseal_execution as test_postseal


_FAKE_STAGE_B_RESULT: dict[str, Any] | None = None
_TEST_ACCESS_OBSERVER = None
_TEST_EXECUTION_BINDINGS: dict[str, dict[str, Any]] | None = None


def verify_postseal_target_completed_set(
    completed_set_path: Path,
    expected_sha256: str,
    expected_prediction_seal_sha256: str,
    expected_checkpoint_bindings: dict[str, str],
) -> dict[str, Any]:
    """Test double with the exact frozen adapter signature."""
    del (
        completed_set_path,
        expected_sha256,
        expected_prediction_seal_sha256,
        expected_checkpoint_bindings,
    )
    assert _FAKE_STAGE_B_RESULT is not None
    return _FAKE_STAGE_B_RESULT


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = postseal.canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, *, signed: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }
    if signed:
        record = json.loads(path.read_text(encoding="utf-8"))
        value["record_sha256"] = record["record_sha256"]
    return value


def _pins(seal_sha: str) -> postseal.EvaluationPins:
    return postseal.EvaluationPins(
        prediction_set_seal_sha256=seal_sha,
        synthesis_checkpoint_sha256="1" * 64,
        training_run_artifact_identity_sha256="2" * 64,
        training_target_artifact_identities_sha256="3" * 64,
        spatial_authority_sha256="4" * 64,
    )


def _install_test_execution_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    def acquire(*_args, **_kwargs):
        assert _TEST_EXECUTION_BINDINGS is not None
        evidence = {
            "schema": test_postseal.PRODUCTION_EXECUTION_EVIDENCE_SCHEMA,
            "authority": {
                "path": "/unit/execution-authority.json",
                "sha256": "6" * 64,
                "size_bytes": 1,
                "record_sha256": "7" * 64,
            },
            "runtime_manifest": {
                "path": "/unit/postseal_runtime.sha256",
                "sha256": "8" * 64,
                "size_bytes": 1,
            },
            "dependency_runtime_authority": {
                "path": "/unit/dependency-authority.json",
                "sha256": "9" * 64,
                "size_bytes": 1,
                "record_sha256": "a" * 64,
            },
            "python_executable": {
                "path": "/unit/python",
                "sha256": "b" * 64,
                "size_bytes": 1,
            },
            "slurm_launcher": {
                "path": "/unit/launcher.slurm",
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
            "stage_b_verifier": {
                "source": {
                    "path": "/unit/source.py",
                    "sha256": "d" * 64,
                    "size_bytes": 1,
                },
                "dependencies": [],
                "function_name": "verify_postseal_target_completed_set",
            },
            "scheduler": {"schema": "unit-ciai-one-gpu-attestation-v2"},
            "runtime_closure": {
                "source_runtime_files_sha256": "e" * 64,
                "source_runtime_file_count": 15,
                "dependency_runtime_files_sha256": "f" * 64,
                "dependency_runtime_file_count": 1,
            },
            "bindings": dict(_TEST_EXECUTION_BINDINGS),
            "restrictions": {"evaluation_only": True},
            "barrier_transitions": [
                "ALL_34_PREDICTION_RECORDS_AND_BYTES_AUTHENTICATED",
                "EXTERNAL_EXECUTION_AUTHORITY_AUTHENTICATED",
            ],
        }
        evidence["record_sha256"] = postseal.canonical_sha256(evidence)
        return test_postseal._ExecutionBoundaryAuthority(  # noqa: SLF001
            evidence=evidence,
            stage_b_verifier_authority=_test_verifier_authority(),
            capability=test_postseal._EXECUTION_BOUNDARY_CAPABILITY,  # noqa: SLF001
        )

    monkeypatch.setattr(
        test_postseal,
        "_acquire_execution_boundary",
        acquire,
    )

    def authenticate_evidence(value, **_kwargs):
        if type(value) is not dict:
            raise postseal.PostsealEvaluationError("fixture execution evidence differs")
        return dict(value)

    monkeypatch.setattr(
        test_postseal,
        "authenticate_published_execution_evidence",
        authenticate_evidence,
    )

    def open_fixture_targets(grant, completed_set_path, completed_set_sha256):
        observer = _TEST_ACCESS_OBSERVER
        if observer is not None:
            observer("prediction_set_fully_authenticated")
            observer("target_verifier_invoked")
        assert _FAKE_STAGE_B_RESULT is not None
        authority = grant.stage_b_verifier_authority
        verifier_source, _ = test_postseal._snapshot_file(  # noqa: SLF001
            authority.source_path,
            label="fixture Stage-B source",
            expected_sha256=authority.source_sha256,
        )
        dependencies = tuple(
            test_postseal._snapshot_file(  # noqa: SLF001
                path,
                label=f"fixture Stage-B dependency {path.name}",
                expected_sha256=digest,
            )[0]
            for path, digest in authority.dependency_sources
        )
        targets = test_postseal._authenticate_stage_b_result(  # noqa: SLF001
            completed_set_path,
            completed_set_sha256=completed_set_sha256,
            predictions=grant.predictions,
            pins=grant.pins,
            raw_result=_FAKE_STAGE_B_RESULT,
            verifier_source=verifier_source,
            verifier_dependencies=dependencies,
        )
        if observer is not None:
            for scan_id in postseal.EXPECTED_SCAN_IDS:
                observer(f"target_bytes:{scan_id}")
        return targets

    monkeypatch.setattr(
        test_postseal, "_open_authenticated_stage_b_targets", open_fixture_targets
    )
    return {"schema": "unit-test-execution-boundary-v1"}


def _authenticate_prediction_set_for_test(
    prediction_root: Path,
    *,
    pins: postseal.EvaluationPins,
    access_observer,
) -> test_postseal.AuthenticatedPredictionSet:
    original_snapshot = test_postseal._snapshot_file  # noqa: SLF001

    def observed_snapshot(path, *, label, **kwargs):
        if label == "prediction-set seal":
            access_observer("prediction_set_seal")
        elif label.endswith(" prediction record"):
            access_observer(f"prediction_record:{label.split()[0]}")
        elif label.endswith(" prediction NIfTI"):
            access_observer(f"prediction_bytes:{label.split()[0]}")
        return original_snapshot(path, label=label, **kwargs)

    test_postseal._snapshot_file = observed_snapshot  # type: ignore[attr-defined]  # noqa: SLF001
    try:
        return test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    finally:
        test_postseal._snapshot_file = original_snapshot  # type: ignore[attr-defined]  # noqa: SLF001


def _authenticate_stage_b_targets_for_test(
    completed_set_path: Path,
    *,
    completed_set_sha256: str,
    predictions: test_postseal.AuthenticatedPredictionSet,
    pins: postseal.EvaluationPins,
    verifier,
    verifier_authority: test_postseal._StageBVerifierAuthority,
    access_observer=None,
) -> test_postseal.AuthenticatedTargetSet:
    if access_observer is not None:
        access_observer("target_verifier_invoked")
    raw_result = verifier(
        completed_set_path,
        completed_set_sha256,
        predictions.seal_snapshot.sha256,
        pins.checkpoint_bindings(),
    )
    verifier_source, _ = test_postseal._snapshot_file(  # noqa: SLF001
        verifier_authority.source_path,
        label="fixture Stage-B source",
        expected_sha256=verifier_authority.source_sha256,
    )
    dependencies = tuple(
        test_postseal._snapshot_file(  # noqa: SLF001
            path,
            label=f"fixture Stage-B dependency {path.name}",
            expected_sha256=digest,
        )[0]
        for path, digest in verifier_authority.dependency_sources
    )
    result = test_postseal._authenticate_stage_b_result(  # noqa: SLF001
        completed_set_path,
        completed_set_sha256=completed_set_sha256,
        predictions=predictions,
        pins=test_postseal._snapshot_evaluation_pins(pins),  # noqa: SLF001
        raw_result=raw_result,
        verifier_source=verifier_source,
        verifier_dependencies=dependencies,
    )
    if access_observer is not None:
        for scan_id in postseal.EXPECTED_SCAN_IDS:
            access_observer(f"target_bytes:{scan_id}")
    return result


def _run_postseal_evaluation_for_test(**kwargs):
    global _TEST_ACCESS_OBSERVER, _TEST_EXECUTION_BINDINGS
    kwargs.pop("stage_b_verifier")
    kwargs.pop("stage_b_verifier_authority", None)
    observer = kwargs.pop("access_observer", None)
    if kwargs.pop("evaluation_feature_extractor", None) is not None:
        raise AssertionError("feature-model fixtures are not part of publication tests")
    _TEST_ACCESS_OBSERVER = observer
    prediction_seal = Path(kwargs["prediction_root"]) / "prediction_set_seal.json"
    completed_set = Path(kwargs["stage_b_completed_set"])
    _TEST_EXECUTION_BINDINGS = {
        "raw_config": {
            "path": "/unit/raw-config.yaml",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "immutable_split": {
            "path": "/unit/immutable-split.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "prediction_set_seal": _artifact(prediction_seal, signed=True),
        "stage_b_completed_set": _artifact(completed_set, signed=True),
    }
    kwargs.setdefault(
        "execution_authority_path",
        Path(kwargs["prediction_root"]).parent / "execution-authority.json",
    )
    kwargs.setdefault("execution_authority_sha256", "6" * 64)
    try:
        return test_postseal.run_postseal_evaluation(**kwargs)
    finally:
        _TEST_ACCESS_OBSERVER = None


def _test_verifier_authority() -> test_postseal._StageBVerifierAuthority:
    source_path = Path(__file__).resolve()
    evaluator_source = Path(postseal.__file__).resolve()
    package_init = evaluator_source.parent / "__init__.py"
    return test_postseal._StageBVerifierAuthority(  # noqa: SLF001
        source_path=source_path,
        source_sha256=_sha(source_path),
        dependency_sources=(
            (evaluator_source, _sha(evaluator_source)),
            (package_init, _sha(package_init)),
        ),
    )


def _prediction_fixture(tmp_path: Path) -> tuple[Path, postseal.EvaluationPins]:
    root = tmp_path / "predictions"
    (root / "subjects").mkdir(parents=True)
    bindings = []
    for scan_id in postseal.EXPECTED_SCAN_IDS:
        subject = root / "subjects" / scan_id
        subject.mkdir()
        prediction_path = subject / "prediction.nii.gz"
        prediction_path.write_bytes(f"prediction:{scan_id}".encode())
        record = _signed(
            {
                "format": postseal.PREDICTION_FORMAT,
                "status": "SEALED_TARGET_BLIND_PREDICTION",
                "scan_id": scan_id,
                "protocol_profile": "a4-native-recovery-v1",
                "target_opened_before_prediction": False,
                "paired_quality_gate_run": False,
                "prediction": {
                    "relative_path": f"subjects/{scan_id}/prediction.nii.gz",
                    "sha256": _sha(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                    "shape": list(postseal.EXPECTED_SHAPE),
                    "repetition_time_seconds": 3.0,
                },
                "synthesis_checkpoint_sha256": "1" * 64,
                "training_run_artifact_identity_sha256": "2" * 64,
                "training_target_artifact_identities_sha256": "3" * 64,
                "spatial_authority_sha256": "4" * 64,
            }
        )
        _write_json(subject / "prediction.json", record)
        bindings.append(
            {
                "scan_id": scan_id,
                "prediction_sha256": record["prediction"]["sha256"],
                "subject_record_sha256": record["record_sha256"],
            }
        )
    seal = _signed(
        {
            "format": postseal.PREDICTION_SET_FORMAT,
            "status": "SEALED_TARGET_BLIND_PREDICTION_SET",
            "split": "test",
            "num_subjects": len(bindings),
            "subjects": bindings,
            "subjects_sha256": postseal.canonical_sha256(bindings),
            "sealed_targets_opened": False,
            "paired_quality_gate_run": False,
            "synthesis_checkpoint_sha256": "1" * 64,
            "training_run_artifact_identity_sha256": "2" * 64,
            "training_target_artifact_identities_sha256": "3" * 64,
            "spatial_authority_sha256": "4" * 64,
        }
    )
    seal_path = root / "prediction_set_seal.json"
    _write_json(seal_path, seal)
    return root, _pins(_sha(seal_path))


def _resign_prediction_seal(root: Path) -> postseal.EvaluationPins:
    path = root / "prediction_set_seal.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("record_sha256")
    value = _signed(value)
    _write_json(path, value)
    return _pins(_sha(path))


def _signed_artifact(path: Path, schema: str, scan_id: str) -> dict[str, Any]:
    _write_json(path, _signed({"schema": schema, "scan_id": scan_id}))
    return _artifact(path, signed=True)


def _stage_b_fixture(
    tmp_path: Path,
    predictions: test_postseal.AuthenticatedPredictionSet,
    pins: postseal.EvaluationPins,
) -> tuple[Path, str, test_postseal._StageBVerifierAuthority]:
    global _FAKE_STAGE_B_RESULT
    root = tmp_path / "stage_b"
    root.mkdir()
    subjects = []
    completed_subjects = []
    prediction_by_scan = predictions.by_scan
    for scan_id in postseal.EXPECTED_SCAN_IDS:
        subject = root / scan_id
        subject.mkdir()
        bold = subject / "native_bold.nii.gz"
        brain_mask = subject / "native_brain_mask.nii.gz"
        labels = subject / "native_labels.nii.gz"
        padded_bold = subject / "padded_bold.nii.gz"
        bold.write_bytes(f"target:{scan_id}".encode())
        brain_mask.write_bytes(f"brain-mask:{scan_id}".encode())
        labels.write_bytes(f"labels:{scan_id}".encode())
        padded_bold.write_bytes(f"padded-target:{scan_id}".encode())
        publication = _signed_artifact(
            subject / "publication.json", "unit-publication", scan_id
        )
        success_receipt = _signed_artifact(
            subject / "success.json", "unit-success", scan_id
        )
        success_marker = _signed_artifact(
            subject / "success.commit.json", "unit-success-marker", scan_id
        )
        native_receipt = _signed_artifact(
            subject / "native.json", "unit-native", scan_id
        )
        sidecar_path = subject / "sidecar.json"
        _write_json(
            sidecar_path,
            _signed({"schema": "unit-sidecar", "scan_id": scan_id}),
        )
        target_subject = {
            "scan_id": scan_id,
            "native_bold": _artifact(bold),
            "native_structural_brain_mask": _artifact(brain_mask),
            "native_structural_labels": _artifact(labels),
            "publication": publication,
            "success_receipt": success_receipt,
            "native_preprocessing_receipt": native_receipt,
        }
        subjects.append(target_subject)
        prediction = prediction_by_scan[scan_id]
        completed_subjects.append(
            {
                "global_index": len(completed_subjects),
                "scan_id": scan_id,
                "BID": scan_id.split("_", 1)[0],
                "role": "sealed-test",
                "success_receipt": success_receipt,
                "success_receipt_commit_marker": success_marker,
                "native_preprocessing_receipt": native_receipt,
                "publication": publication,
                "native_bold": target_subject["native_bold"],
                "native_structural_labels": target_subject["native_structural_labels"],
                "padded_bold": _artifact(padded_bold),
                "sidecar": _artifact(sidecar_path),
                "prediction_subject_record_sha256": prediction.record["record_sha256"],
                "prediction_sha256": prediction.prediction_snapshot.sha256,
            }
        )
    worklist_path = root / "sealed_target_worklist.json"
    _write_json(
        worklist_path,
        _signed(
            {
                "schema": postseal.STAGE_B_WORKLIST_SCHEMA,
                "ordered_scan_ids": list(postseal.EXPECTED_SCAN_IDS),
            }
        ),
    )
    completed_path = root / "completed.json"
    completed = _signed(
        {
            "schema": postseal.STAGE_B_COMPLETED_SET_SCHEMA,
            "prediction_set_seal": predictions.seal_snapshot.descriptor()
            | {"record_sha256": predictions.seal["record_sha256"]},
            "checkpoint_bindings": pins.checkpoint_bindings(),
            "sealed_target_worklist": _artifact(worklist_path, signed=True),
            "expected_count": len(postseal.EXPECTED_SCAN_IDS),
            "success_count": len(postseal.EXPECTED_SCAN_IDS),
            "quarantine_count": 0,
            "ordered_scan_ids": list(postseal.EXPECTED_SCAN_IDS),
            "ordered_scan_ids_sha256": postseal.EXPECTED_SCAN_IDS_SHA256,
            "subjects": completed_subjects,
            "quarantines": [],
            "complete": True,
            "all_34_successful": True,
            "no_replacement": True,
            "no_refill": True,
            "evaluation_only": True,
            "authorizes_model_training": False,
            "authorizes_checkpoint_selection": False,
            "authorizes_model_or_candidate_selection": False,
            "authorizes_prediction_emission": False,
            "authorizes_additional_inference": False,
            "authorizes_training_or_model_feedback": False,
            "paper_certified": False,
        }
    )
    _write_json(completed_path, completed)
    result = {
        "schema": postseal.STAGE_B_VERIFICATION_SCHEMA,
        "completed_set": _artifact(completed_path, signed=True),
        "prediction_set_seal_sha256": predictions.seal_snapshot.sha256,
        "prediction_set_seal_record_sha256": predictions.seal["record_sha256"],
        "checkpoint_bindings": pins.checkpoint_bindings(),
        "ordered_scan_ids": list(postseal.EXPECTED_SCAN_IDS),
        "ordered_scan_ids_sha256": postseal.EXPECTED_SCAN_IDS_SHA256,
        "subjects": subjects,
        "complete": True,
        "no_replacement": True,
        "no_refill": True,
        "evaluation_only": True,
        "authorizes_model_training": False,
        "authorizes_checkpoint_selection": False,
        "authorizes_model_or_candidate_selection": False,
        "authorizes_prediction_emission": False,
        "authorizes_additional_inference": False,
    }
    result["record_sha256"] = postseal.canonical_sha256(result)
    _FAKE_STAGE_B_RESULT = result
    authority = _test_verifier_authority()
    return completed_path, _sha(completed_path), authority


def _closed_stage_b_package(
    tmp_path: Path,
    *,
    marker: Path,
) -> tuple[Path, test_postseal._StageBVerifierAuthority]:
    package = tmp_path / "closed_stage_b"
    package.mkdir()
    init_path = package / "__init__.py"
    controller = package / "stage_b_controller.py"
    source = package / "sealed_target_stage_b.py"
    init_path.write_text('"""Closed unit package."""\n', encoding="utf-8")
    controller.write_text("VALUE = 1\n", encoding="utf-8")
    source.write_text(
        "from pathlib import Path\n"
        "def verify_postseal_target_completed_set(*_args):\n"
        f"    Path({str(marker)!r}).write_text('original', encoding='utf-8')\n"
        "    return {}\n",
        encoding="utf-8",
    )
    return package, test_postseal._StageBVerifierAuthority(  # noqa: SLF001
        source_path=source,
        source_sha256=_sha(source),
        dependency_sources=(
            (init_path, _sha(init_path)),
            (controller, _sha(controller)),
        ),
    )


def _write_nifti(
    path: Path,
    data: np.ndarray,
    *,
    affine: np.ndarray | None = None,
    tr: float | None = None,
) -> None:
    affine = np.diag([3.0, 3.0, 3.0, 1.0]) if affine is None else affine
    image = nib.Nifti1Image(data, affine)
    if data.ndim == 4:
        image.header.set_zooms((3.0, 3.0, 3.0, 3.0 if tr is None else tr))
        image.header.set_xyzt_units("mm", "sec")
    else:
        image.header.set_zooms((3.0, 3.0, 3.0))
        image.header.set_xyzt_units("mm")
    nib.save(image, str(path))


def _install_test_native_contract(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    *,
    roi_labels: tuple[int, ...] = (1, 2),
) -> None:
    """Keep tiny synthetic fixtures exact without weakening production constants."""

    for module in (postseal, test_postseal):
        monkeypatch.setattr(module, "EXPECTED_SHAPE", shape)
        monkeypatch.setattr(module, "EXPECTED_SPATIAL_SHAPE", shape[:3])
    monkeypatch.setattr(postseal, "CANONICAL_ROI_LABEL_IDS", roi_labels)
    monkeypatch.setattr(postseal, "CANONICAL_ROI_MAPPING_SHA256", "d" * 64)


def _interior_brain_mask(shape: tuple[int, int, int], margin: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[margin:-margin, margin:-margin, margin:-margin] = 1
    return mask


def _quality_record(*, passed: bool = True, collapse: bool = False) -> dict[str, Any]:
    ratio_checks = {
        "spatial_robust_range_ratio": {"passed": passed},
        "spatial_gradient_ratio": {"passed": passed},
        "spatial_laplacian_ratio": {"passed": passed},
        "spatial_high_frequency_ratio": {"passed": passed},
        "dynamic_high_frequency_ratio": {"passed": passed},
        "temporal_variance_ratio": {"passed": passed},
        "dvars_ratio": {"passed": passed},
        "dynamic_power_ratio": {"passed": passed},
        "effective_rank_ratio": {"passed": passed},
    }
    checks = {
        "grid_affine_tr_contract": {"passed": True},
        "support_dice": {"passed": passed},
        "support_center_of_mass": {"passed": passed},
        "mask_field_of_view": {"passed": passed},
        "outside_mask_leakage": {"passed": passed},
        **ratio_checks,
        "spatial_high_frequency_correlation": {"passed": passed},
        "dynamic_high_frequency_correlation": {"passed": passed},
        "near_static_voxel_fraction": {"passed": passed},
        "structured_temporal_available": {"passed": passed},
        "roi_fc_correlation": {"passed": passed},
        "roi_power_spectrum_correlation": {"passed": passed},
    }
    failed_checks = (
        []
        if passed
        else [
            "spatial_high_frequency_ratio",
            "dynamic_high_frequency_ratio",
            "dynamic_high_frequency_correlation",
            "temporal_variance_ratio",
            "dvars_ratio",
            "dynamic_power_ratio",
            "effective_rank_ratio",
            "near_static_voxel_fraction",
        ]
    )
    paper_values = {
        "mse": 0.01,
        "voxel_corr": 0.8,
        "roi_corr": 0.7,
        "f2f_corr": 0.75,
        "ssim": 0.85,
        "psnr": 20.0,
    }
    return {
        "policy": {
            "min_spatial_robust_range_ratio": 0.5,
            "max_spatial_robust_range_ratio": 2.0,
            "min_spatial_gradient_ratio": 0.5,
            "max_spatial_gradient_ratio": 2.0,
            "min_spatial_laplacian_ratio": 0.5,
            "max_spatial_laplacian_ratio": 2.0,
            "min_spatial_high_frequency_ratio": 0.5,
            "max_spatial_high_frequency_ratio": 2.0,
            "min_dynamic_high_frequency_ratio": 0.25,
            "max_dynamic_high_frequency_ratio": 4.0,
            "min_temporal_variance_ratio": 0.25,
            "max_temporal_variance_ratio": 4.0,
            "min_dvars_ratio": 0.25,
            "max_dvars_ratio": 4.0,
            "min_dynamic_power_ratio": 0.25,
            "max_dynamic_power_ratio": 4.0,
            "min_effective_rank_ratio": 0.5,
            "max_effective_rank_ratio": 2.0,
        },
        "geometry": {
            "shape": list(postseal.EXPECTED_SHAPE),
            "canonical_axis_codes": list(postseal.EXPECTED_AXIS_CODES),
            "tr_seconds": postseal.EXPECTED_TR_SECONDS,
            "mask_derived_from_real": False,
        },
        "support": {"dice": 0.95, "center_of_mass_distance_mm": 1.0},
        "outside_mask": {"predicted_outside_mask_leakage_ratio": 0.001},
        "spatial": {
            "robust_range_ratio": 0.98,
            "gradient_rms_ratio": 0.92,
            "laplacian_rms_ratio": 0.88,
            "high_frequency_rms_ratio": 0.84 if passed else 0.2,
            "temporal_mean_high_frequency_correlation": 0.7,
            "dynamic_high_frequency_rms_ratio": 0.82 if passed else 0.1,
            "dynamic_high_frequency_correlation": 0.66 if passed else 0.0,
        },
        "temporal": {
            "temporal_variance_ratio": 0.91 if passed else 0.01,
            "dvars_ratio": 0.9 if passed else 0.01,
            "dynamic_power_ratio": 0.87 if passed else 0.01,
            "effective_rank_ratio": 0.95 if passed else 0.1,
            "near_static_voxel_fraction": 0.04 if passed else 0.95,
        },
        "structured_temporal": {
            "available": True,
            "fc_matrix_correlation": 0.72,
            "power_spectrum_correlation": 0.68,
        },
        "paper_metrics": {
            "metrics": {
                key: {"available": True, "value": value}
                for key, value in paper_values.items()
            }
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "temporal_collapse_detected": collapse,
        "passed": passed,
        "verdict": "pass"
        if passed
        else ("fail_temporal_collapse" if collapse else "fail_quality_gate"),
    }


def _pair_contract() -> dict[str, Any]:
    image = {
        "minimum": 0.0,
        "maximum": 1.0,
    }
    return {
        "real": dict(image),
        "predicted": dict(image),
        "structural_brain_mask": {
            "minimum": 0.0,
            "maximum": 1.0,
            "foreground_fraction": 0.25,
            "foreground_voxel_count": 120,
            "roi_foreground_outside_mask_voxel_count": 0,
        },
        "target_validity_mask": {
            "contract": postseal.TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[
                "a4-native-recovery-v1"
            ],
            "derivation": (
                "exact-nonzero-support-across-final-stored-target-time-series"
            ),
            "shape": list(postseal.EXPECTED_SPATIAL_SHAPE),
            "foreground_voxel_count": 100,
            "foreground_fraction_of_structural_support": 100 / 120,
            "structural_foreground_voxel_count": 120,
            "structural_voxels_excluded_from_target_metrics": 20,
            "outside_structural_brain_voxel_count": 0,
            "configured_structural_roi_count": len(postseal.CANONICAL_ROI_LABEL_IDS),
            "all_configured_structural_rois_have_target_validity_support": True,
            "equals_exact_final_nonzero_support": True,
            "exactly_binary": True,
            "fmri_resampled": False,
        },
        "structural_labels": {
            "minimum": 0.0,
            "maximum": float(max(postseal.CANONICAL_ROI_LABEL_IDS)),
        },
        "positive_structural_label_count": len(postseal.CANONICAL_ROI_LABEL_IDS),
        "protocol_profile": "a4-native-recovery-v1",
        "canonical_roi_mapping_sha256": postseal.CANONICAL_ROI_MAPPING_SHA256,
    }


def _passing_texture_gate(*, forged_values: bool = False) -> dict[str, Any]:
    diagnostics = {
        "diagnostic_ratios": {name: 1.0 for name in texture_audit.TEXTURE_GATE_METRICS},
        "texture_retention_reference": (
            texture_audit.DEFAULT_TEXTURE_RETENTION_REFERENCE
        ),
        "aggregate_detail_correlation": 1.0,
        "near_nyquist_tail_power_ratio": 1.0,
        "near_nyquist_boundary_cycles_per_voxel": (
            texture_audit.DEFAULT_NEAR_NYQUIST_CYCLES_PER_VOXEL
        ),
    }
    gate = texture_audit.build_texture_quality_gate(
        diagnostics,
        enforce=True,
        enforce_anti_gaming=True,
    )
    if forged_values:
        for result in gate["results"].values():
            result["value"] = 0.0
            result["passed"] = True
        gate["anti_gaming_results"]["aggregate_detail_correlation"]["value"] = -999.0
        gate["anti_gaming_results"]["aggregate_detail_correlation"]["passed"] = True
        gate["anti_gaming_results"]["near_nyquist_tail_power_ratio"]["value"] = 999.0
        gate["anti_gaming_results"]["near_nyquist_tail_power_ratio"]["passed"] = True
    return gate


def _synthetic_texture_evidence(
    scan_id: str, *, forged_values: bool = False
) -> dict[str, Any]:
    gate = _passing_texture_gate(forged_values=forged_values)
    return {
        "schema": postseal.TEXTURE_AUDIT_SCHEMA,
        "quality_gate_schema": postseal.TEXTURE_GATE_SCHEMA,
        "quality_gate": gate,
        "mode": gate["mode"],
        "verdict": gate["verdict"],
        "exit_code": gate["exit_code"],
        "release_gate_passed": True,
        "amplitude_retention_enforced": True,
        "anti_gaming_guards_enforced": True,
        "policy": gate["policy"],
        "anti_gaming_policy": gate["anti_gaming_policy"],
        "results": gate["results"],
        "anti_gaming_results": gate["anti_gaming_results"],
        "display_contract": {
            "image_interpolation": "nearest",
            "fmri_registration_or_resampling": False,
            "mask_resampled": False,
            "selection_uses_prediction": False,
        },
        "outputs": {
            "png": {
                "relative_path": (
                    f"subjects/{scan_id}/texture/{scan_id}_texture_audit.png"
                ),
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "manifest": {
                "relative_path": (
                    f"subjects/{scan_id}/texture/{scan_id}_texture_audit.json"
                ),
                "sha256": "b" * 64,
                "size_bytes": 1,
                "record_sha256": "c" * 64,
            },
        },
        "prediction_set_fully_authenticated_before_any_target_access": True,
        "all_targets_authenticated_before_texture_audit": True,
        "fmri_resampling_performed": False,
        "can_authorize_training_or_selection": False,
    }


def _quality_subject_records() -> list[dict[str, Any]]:
    return [
        {
            "scan_id": scan_id,
            "quality": _quality_record(),
            "pair_contract": _pair_contract(),
            "texture_audit": _synthetic_texture_evidence(scan_id),
        }
        for scan_id in postseal.EXPECTED_SCAN_IDS
    ]


def _publication_fixture(
    tmp_path: Path, *, forged_texture_values: bool = False
) -> Path:
    prediction_root, pins = _prediction_fixture(tmp_path)
    authenticated_predictions = test_postseal.authenticate_prediction_set(
        prediction_root, pins=pins
    )
    completed_path, _completed_sha, _runtime_verifier_authority = _stage_b_fixture(
        tmp_path, authenticated_predictions, pins
    )
    _package, verifier_authority = _closed_stage_b_package(
        tmp_path, marker=tmp_path / "unused-stage-b-marker"
    )
    assert _FAKE_STAGE_B_RESULT is not None
    stage_b_verification = copy.deepcopy(_FAKE_STAGE_B_RESULT)
    stage_b_by_scan = {
        item["scan_id"]: item for item in stage_b_verification["subjects"]
    }
    root = tmp_path / "published-evaluation"
    root.mkdir()
    subjects = []
    texture_rows = []
    for scan_id in postseal.EXPECTED_SCAN_IDS:
        subject_dir = root / "subjects" / scan_id
        texture_dir = subject_dir / "texture"
        texture_dir.mkdir(parents=True)
        target_validity_path = subject_dir / "target_validity_mask.nii.gz"
        target_validity = np.zeros(postseal.EXPECTED_SPATIAL_SHAPE, dtype=np.uint8)
        target_validity.reshape(-1)[:100] = 1
        _write_nifti(target_validity_path, target_validity)
        prediction_record_path = (
            prediction_root / "subjects" / scan_id / "prediction.json"
        )
        stage_b_subject = stage_b_by_scan[scan_id]
        input_descriptors = {
            "real": dict(stage_b_subject["native_bold"]),
            "predicted": _artifact(
                prediction_root / "subjects" / scan_id / "prediction.nii.gz"
            ),
            "structural_brain_mask": dict(
                stage_b_subject["native_structural_brain_mask"]
            ),
            "structural_labels": dict(stage_b_subject["native_structural_labels"]),
        }
        input_descriptors["target_validity_mask"] = {
            "path": str(target_validity_path),
            "sha256": _sha(target_validity_path),
            "size_bytes": target_validity_path.stat().st_size,
            "contract": postseal.TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[
                "a4-native-recovery-v1"
            ],
            "derivation": (
                "exact-nonzero-support-across-final-stored-target-time-series-"
                "after-complete-target-authentication"
            ),
            "equals_exact_final_nonzero_support": True,
        }
        png_path = texture_dir / f"{scan_id}_texture_audit.png"
        png_path.write_bytes(f"{scan_id}:texture".encode("ascii"))
        gate = _passing_texture_gate(forged_values=forged_texture_values)
        texture_manifest = _signed(
            {
                "schema": postseal.TEXTURE_AUDIT_SCHEMA,
                "quality_gate": gate,
                "inputs": {
                    "real": input_descriptors["real"],
                    "predicted": input_descriptors["predicted"],
                    "mask": {
                        key: input_descriptors["target_validity_mask"][key]
                        for key in ("path", "sha256", "size_bytes")
                    },
                    "roi_labels": input_descriptors["structural_labels"],
                },
                "display_contract": {
                    "image_interpolation": "nearest",
                    "fMRI_registration_or_resampling": False,
                },
                "outputs": {
                    "texture_audit_png": {
                        "path": str(png_path),
                        "sha256": _sha(png_path),
                        "size_bytes": png_path.stat().st_size,
                    }
                },
            }
        )
        texture_manifest_path = texture_dir / f"{scan_id}_texture_audit.json"
        _write_json(texture_manifest_path, texture_manifest)
        texture_evidence = _synthetic_texture_evidence(
            scan_id, forged_values=forged_texture_values
        )
        texture_evidence["outputs"] = {
            "png": {
                "relative_path": png_path.relative_to(root).as_posix(),
                "sha256": _sha(png_path),
                "size_bytes": png_path.stat().st_size,
            },
            "manifest": {
                "relative_path": texture_manifest_path.relative_to(root).as_posix(),
                "sha256": _sha(texture_manifest_path),
                "size_bytes": texture_manifest_path.stat().st_size,
                "record_sha256": texture_manifest["record_sha256"],
            },
        }
        record = _signed(
            {
                "schema": postseal.SUBJECT_EVALUATION_SCHEMA,
                "scan_id": scan_id,
                "target": input_descriptors["real"],
                "target_validity_mask": input_descriptors["target_validity_mask"],
                "prediction": input_descriptors["predicted"],
                "prediction_record": _artifact(prediction_record_path, signed=True),
                "structural_brain_mask": input_descriptors["structural_brain_mask"],
                "structural_labels": input_descriptors["structural_labels"],
                "stage_b_publication": dict(stage_b_subject["publication"]),
                "stage_b_success_receipt": dict(stage_b_subject["success_receipt"]),
                "native_preprocessing_receipt": dict(
                    stage_b_subject["native_preprocessing_receipt"]
                ),
                "pair_contract": _pair_contract(),
                "quality": _quality_record(),
                "visualization_outputs": [],
                "texture_audit": texture_evidence,
                "fmri_resampling_performed": False,
                "evaluation_can_authorize_training_or_selection": False,
            }
        )
        path = subject_dir / "evaluation.json"
        _write_json(path, record)
        compact_texture = (
            {
                "schema": postseal.TEXTURE_GATE_SCHEMA,
                "mode": "combined-release-gate",
                "verdict": "pass",
                "exit_code": 0,
                "passed": True,
                "policy": texture_evidence["policy"],
                "anti_gaming_policy": texture_evidence["anti_gaming_policy"],
                "results": texture_evidence["results"],
                "anti_gaming_results": texture_evidence["anti_gaming_results"],
                "outputs": texture_evidence["outputs"],
                "nearest_neighbor_display": True,
                "fmri_resampled": False,
            }
            if forged_texture_values
            else postseal._compact_texture_gate(  # noqa: SLF001
                scan_id, texture_evidence
            )
        )
        texture_rows.append(
            {
                "scan_id": scan_id,
                "texture_release_gate": compact_texture,
            }
        )
        subjects.append(
            {
                "scan_id": scan_id,
                "record": {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _sha(path),
                    "size_bytes": path.stat().st_size,
                    "record_sha256": record["record_sha256"],
                },
            }
        )
    chart = root / "cohort_quality_summary.png"
    chart.write_bytes(b"synthetic chart bytes")
    report = root / "DATA_QUALITY_REPORT.md"
    report.write_text("# synthetic report\n", encoding="utf-8")
    fitness = {
        "status": "CONDITIONALLY_FIT",
        "severity": "medium",
        "texture_release_gate_required": True,
        "texture_release_gate_verdict": "pass",
        "texture_release_gate_pass_count": 34,
        "texture_release_gate_fail_count": 0,
        "blocked_uses": [
            "training or checkpoint/candidate selection",
            "additional prediction or inference authorization",
            "claiming paper numerical reproduction without comparing authenticated results",
            "FID or Inception Score claims without pinned SLIM-Brain",
        ],
    }
    cohort_texture_gate = {
        "schema": postseal.TEXTURE_GATE_SCHEMA,
        "mode": "combined-release-gate",
        "required_for_publication": True,
        "verdict": "pass",
        "exit_code": 0,
        "scan_count": len(postseal.EXPECTED_SCAN_IDS),
        "passing_scan_count": len(postseal.EXPECTED_SCAN_IDS),
        "failing_scan_count": 0,
        "failed_scan_ids": [],
        "policy": _passing_texture_gate()["policy"],
        "anti_gaming_policy": _passing_texture_gate()["anti_gaming_policy"],
        "subject_outputs": [
            {
                "scan_id": row["scan_id"],
                "verdict": row["texture_release_gate"]["verdict"],
                "exit_code": row["texture_release_gate"]["exit_code"],
                "outputs": row["texture_release_gate"]["outputs"],
            }
            for row in texture_rows
        ],
        "nearest_neighbor_display": True,
        "fmri_resampled": False,
        "evaluation_feedback_to_training_or_selection": False,
    }
    distributional = postseal._distributional_metrics_unavailable()  # noqa: SLF001
    data_quality = _signed(
        {
            "schema": postseal.DATA_QUALITY_SCHEMA,
            "per_scan_rows": texture_rows,
            "fitness_for_use": fitness,
            "distributional_metrics": distributional,
            "findings": [
                {
                    "dimension": "fid_and_inception_score",
                    "status": "unavailable",
                    "severity": "medium",
                    "confidence": "high",
                    "passing_scan_count": 0,
                    "passing_scan_rate": 0.0,
                    "failed_scan_ids": [],
                    "risk": "full paper metric coverage is incomplete",
                    "required_action": (
                        "qualify the closed pinned evaluation feature-model "
                        "authority; never substitute"
                    ),
                }
            ],
            "required_input_file_hash_coverage": {
                "required_by_role": {},
                "required_file_count": 0,
                "hash_authenticated_file_count": 0,
                "coverage_rate": 1.0,
                "missing_or_unhashed": [],
            },
            "cohort_rates": {
                "quality_fail_count": 0,
                "temporal_collapse_count": 0,
            },
            "texture_release_gate": cohort_texture_gate,
            "cohort_chart": {
                "relative_path": chart.name,
                "sha256": _sha(chart),
                "size_bytes": chart.stat().st_size,
            },
            "human_readable_report": {
                "relative_path": report.name,
                "sha256": _sha(report),
                "size_bytes": report.stat().st_size,
            },
        }
    )
    data_quality_path = root / "data_quality_summary.json"
    _write_json(data_quality_path, data_quality)
    inventory = postseal._relative_inventory(  # noqa: SLF001
        root, excluded={"evaluation.json", "publication_receipt.json"}
    )
    evaluation = _signed(
        {
            "schema": postseal.EVALUATION_SCHEMA,
            "status": "COMPLETE_POSTSEAL_HELDOUT_EVALUATION",
            "role": "fixed-sealed-test-final-report-only",
            "scan_count": len(postseal.EXPECTED_SCAN_IDS),
            "ordered_scan_ids": list(postseal.EXPECTED_SCAN_IDS),
            "ordered_scan_ids_sha256": postseal.EXPECTED_SCAN_IDS_SHA256,
            "sealed_role_csv_sha256": postseal.SEALED_ROLE_CSV_SHA256,
            "prediction_set_seal": _artifact(
                prediction_root / "prediction_set_seal.json", signed=True
            ),
            "stage_b_completed_set": _artifact(completed_path, signed=True),
            "stage_b_verifier_source": _artifact(verifier_authority.source_path),
            "stage_b_verifier_dependencies": [
                _artifact(path)
                for path, _digest in verifier_authority.dependency_sources
            ],
            "stage_b_verification": stage_b_verification,
            "checkpoint_bindings": pins.checkpoint_bindings(),
            "subjects": subjects,
            "paper_metrics_cohort_mean": {},
            "distributional_metrics": distributional,
            "fitness_for_use": fitness,
            "texture_release_gate": cohort_texture_gate,
            "texture_release_gate_pass_count": len(postseal.EXPECTED_SCAN_IDS),
            "texture_release_gate_fail_count": 0,
            "quality_pass_count": len(postseal.EXPECTED_SCAN_IDS),
            "quality_fail_count": 0,
            "all_34_texture_retention_and_anti_gaming_gates_passed": True,
            "data_quality_summary": {
                "relative_path": data_quality_path.name,
                "sha256": _sha(data_quality_path),
                "size_bytes": data_quality_path.stat().st_size,
                "record_sha256": data_quality["record_sha256"],
            },
            "output_inventory": inventory,
            "output_inventory_sha256": postseal.canonical_sha256(inventory),
            "implementation_sources": postseal._source_inventory(),  # noqa: SLF001
            "production_runtime_and_scheduler_evidence": {},
            "production_evaluator_authenticated_before_import": True,
            "prediction_set_fully_authenticated_before_any_target_access": True,
            "all_target_publications_and_bytes_authenticated_before_metrics": True,
            "all_targets_authenticated_before_texture_audits": True,
            "exact_native_grid_no_fmri_resampling": True,
            "independent_structural_mask_and_roi_labels_only": True,
            "synthesis_model_or_checkpoint_loaded": False,
            "evaluation_feature_model_loaded": False,
            "training_or_inference_entrypoint_imported": False,
            "authorizes_training": False,
            "authorizes_model_selection": False,
            "authorizes_candidate_selection": False,
            "authorizes_checkpoint_selection": False,
            "authorizes_prediction": False,
            "authorizes_inference": False,
            "can_change_checkpoint_or_prediction_set": False,
        }
    )
    evaluation_path = root / "evaluation.json"
    _write_json(evaluation_path, evaluation)
    receipt = _signed(
        {
            "schema": postseal.PUBLICATION_RECEIPT_SCHEMA,
            "status": "COMMITTED_NO_REPLACE",
            "destination": str(root),
            "evaluation": {
                "relative_path": evaluation_path.name,
                "sha256": _sha(evaluation_path),
                "size_bytes": evaluation_path.stat().st_size,
                "record_sha256": evaluation["record_sha256"],
            },
            "output_inventory_sha256": evaluation["output_inventory_sha256"],
            "all_34_subjects_complete": True,
            "all_34_texture_retention_and_anti_gaming_gates_passed": True,
            "published_after_all_subjects_complete": True,
            "no_overwrite": True,
            "evaluation_feedback_to_training_or_selection": False,
        }
    )
    _write_json(root / "publication_receipt.json", receipt)
    return root


def _publication_authority_arguments(root: Path) -> dict[str, Any]:
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    return {
        "expected_prediction_set_seal_path": evaluation["prediction_set_seal"]["path"],
        "expected_prediction_set_seal_sha256": evaluation["prediction_set_seal"][
            "sha256"
        ],
        "expected_stage_b_completed_set_path": evaluation["stage_b_completed_set"][
            "path"
        ],
        "expected_stage_b_completed_set_sha256": evaluation["stage_b_completed_set"][
            "sha256"
        ],
        "expected_stage_b_verifier_source_path": evaluation["stage_b_verifier_source"][
            "path"
        ],
        "expected_stage_b_verifier_source_sha256": evaluation[
            "stage_b_verifier_source"
        ]["sha256"],
        "expected_stage_b_verifier_dependency_paths": [
            item["path"] for item in evaluation["stage_b_verifier_dependencies"]
        ],
        "expected_stage_b_verifier_dependency_sha256s": [
            item["sha256"] for item in evaluation["stage_b_verifier_dependencies"]
        ],
        "expected_execution_authority_path": "/unit/execution-authority.json",
        "expected_execution_authority_sha256": "6" * 64,
    }


def _verify_publication(
    root: Path,
    *,
    slimbrain_authority_sha256: str | None = None,
    authority_pins: dict[str, Any] | None = None,
) -> dict[str, Any]:
    authority_arguments = (
        _publication_authority_arguments(root)
        if authority_pins is None
        else {
            "expected_prediction_set_seal_path": authority_pins["prediction_path"],
            "expected_prediction_set_seal_sha256": authority_pins["prediction"],
            "expected_stage_b_completed_set_path": authority_pins["completed_path"],
            "expected_stage_b_completed_set_sha256": authority_pins["completed"],
            "expected_stage_b_verifier_source_path": authority_pins["source_path"],
            "expected_stage_b_verifier_source_sha256": authority_pins["source"],
            "expected_stage_b_verifier_dependency_paths": authority_pins[
                "dependency_paths"
            ],
            "expected_stage_b_verifier_dependency_sha256s": authority_pins[
                "dependencies"
            ],
            "expected_execution_authority_path": ("/unit/execution-authority.json"),
            "expected_execution_authority_sha256": "6" * 64,
        }
    )
    call_arguments = {
        "expected_evaluation_sha256": _sha(root / "evaluation.json"),
        "expected_receipt_sha256": _sha(root / "publication_receipt.json"),
        **authority_arguments,
        "expected_slimbrain_authority_sha256": slimbrain_authority_sha256,
    }
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation["production_runtime_and_scheduler_evidence"] != {}:
        return postseal.verify_evaluation_publication(root, **call_arguments)

    # The hand-built publication fixture predates the externally frozen runtime
    # authority and intentionally uses tiny non-NIfTI placeholders for external
    # target/prediction bytes.  Keep those recursive publication-attack tests
    # scoped to the bundle verifier; dedicated successor regressions below drive
    # the real public cohort preflight with valid native-NIfTI fixtures.
    pair_contracts = {
        scan_id: _pair_contract() for scan_id in postseal.EXPECTED_SCAN_IDS
    }
    original_preflight = test_postseal.preflight_external_publication_authorities
    original_production_validator = postseal.validate_exact_pair
    original_execution_validator = test_postseal.validate_exact_pair
    original_evidence_authenticator = (
        test_postseal.authenticate_published_execution_evidence
    )

    def fixture_preflight(**_kwargs):
        return SimpleNamespace(
            pair_contract_by_scan=pair_contracts,
            execution_evidence={},
        )

    def fixture_pair_contract(*_args, **_kwargs):
        return _pair_contract()

    def fixture_execution_evidence(value, **_kwargs):
        if value != {}:
            raise postseal.PostsealEvaluationError("fixture execution evidence differs")
        return {}

    test_postseal.preflight_external_publication_authorities = fixture_preflight
    postseal.validate_exact_pair = fixture_pair_contract
    test_postseal.validate_exact_pair = fixture_pair_contract
    test_postseal.authenticate_published_execution_evidence = fixture_execution_evidence
    try:
        return postseal.verify_evaluation_publication(root, **call_arguments)
    finally:
        test_postseal.preflight_external_publication_authorities = original_preflight
        postseal.validate_exact_pair = original_production_validator
        test_postseal.validate_exact_pair = original_execution_validator
        test_postseal.authenticate_published_execution_evidence = (
            original_evidence_authenticator
        )


def _resign_root_evaluation_and_receipt(root: Path) -> None:
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256", None)
    _write_json(evaluation_path, _signed(evaluation))
    updated_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt_path = root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256", None)
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": updated_evaluation["record_sha256"],
        }
    )
    receipt["output_inventory_sha256"] = evaluation["output_inventory_sha256"]
    _write_json(receipt_path, _signed(receipt))


def _materialize_prediction_niftis(
    root: Path, predicted: np.ndarray
) -> postseal.EvaluationPins:
    bindings = []
    for scan_id in postseal.EXPECTED_SCAN_IDS:
        subject = root / "subjects" / scan_id
        prediction_path = subject / "prediction.nii.gz"
        _write_nifti(prediction_path, predicted)
        record = json.loads((subject / "prediction.json").read_text(encoding="utf-8"))
        record.pop("record_sha256")
        record["prediction"]["sha256"] = _sha(prediction_path)
        record["prediction"]["size_bytes"] = prediction_path.stat().st_size
        record = _signed(record)
        _write_json(subject / "prediction.json", record)
        bindings.append(
            {
                "scan_id": scan_id,
                "prediction_sha256": record["prediction"]["sha256"],
                "subject_record_sha256": record["record_sha256"],
            }
        )
    seal_path = root / "prediction_set_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal.pop("record_sha256")
    seal["subjects"] = bindings
    seal["subjects_sha256"] = postseal.canonical_sha256(bindings)
    seal = _signed(seal)
    _write_json(seal_path, seal)
    return _pins(_sha(seal_path))


def _materialize_stage_b_niftis(
    real: np.ndarray, labels: np.ndarray, brain_mask: np.ndarray | None = None
) -> str:
    assert _FAKE_STAGE_B_RESULT is not None
    if brain_mask is None:
        brain_mask = labels > 0
    for subject in _FAKE_STAGE_B_RESULT["subjects"]:
        bold_path = Path(subject["native_bold"]["path"])
        mask_path = Path(subject["native_structural_brain_mask"]["path"])
        label_path = Path(subject["native_structural_labels"]["path"])
        _write_nifti(bold_path, real)
        _write_nifti(mask_path, np.asarray(brain_mask, dtype=np.uint8))
        _write_nifti(label_path, labels)
        subject["native_bold"] = _artifact(bold_path)
        subject["native_structural_brain_mask"] = _artifact(mask_path)
        subject["native_structural_labels"] = _artifact(label_path)
    completed_path = Path(_FAKE_STAGE_B_RESULT["completed_set"]["path"])
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed.pop("record_sha256")
    result_by_scan = {
        subject["scan_id"]: subject for subject in _FAKE_STAGE_B_RESULT["subjects"]
    }
    for entry in completed["subjects"]:
        result_subject = result_by_scan[entry["scan_id"]]
        entry["native_bold"] = dict(result_subject["native_bold"])
        entry["native_structural_labels"] = dict(
            result_subject["native_structural_labels"]
        )
    _write_json(completed_path, _signed(completed))
    _FAKE_STAGE_B_RESULT["completed_set"] = _artifact(completed_path, signed=True)
    unsigned = dict(_FAKE_STAGE_B_RESULT)
    unsigned.pop("record_sha256")
    _FAKE_STAGE_B_RESULT["record_sha256"] = postseal.canonical_sha256(unsigned)
    return _sha(completed_path)


def _fake_visual_outputs(
    real_path: Path,
    predicted_path: Path,
    out_dir: Path,
    *,
    mask_path: Path,
    roi_labels_path: Path,
    tr_seconds: float,
    prefix: str,
    fps: int,
) -> dict[str, Path]:
    del tr_seconds, fps
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, suffix in (
        ("montage", "frames_montage.png"),
        ("temporal", "temporal_diagnostics.png"),
        ("spatial_detail", "spatial_detail.png"),
        ("outside_mask", "outside_mask_leakage.png"),
        ("structured_dynamics", "structured_dynamics.png"),
        ("animation", "4d_comparison.gif"),
    ):
        path = out_dir / f"{prefix}_{suffix}"
        path.write_bytes(f"{prefix}:{name}".encode("ascii"))
        outputs[name] = path
    inputs = {
        "real": Path(real_path).resolve(strict=True),
        "predicted": Path(predicted_path).resolve(strict=True),
        "mask": Path(mask_path).resolve(strict=True),
        "roi_labels": Path(roi_labels_path).resolve(strict=True),
    }
    manifest = _signed(
        {
            "schema": "connect4-4d-visualization-manifest-v1",
            "inputs": {
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in inputs.items()
            },
            "display_contract": {
                "fixed_scales_across_all_animation_frames": True,
                "image_interpolation": "nearest",
                "mask_resampled": False,
            },
            "outputs": {
                name: {"path": str(path.resolve(strict=True)), "sha256": _sha(path)}
                for name, path in outputs.items()
            },
        }
    )
    manifest_path = out_dir / f"{prefix}_visualization_manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def _fake_texture_outputs(
    real_path: Path,
    predicted_path: Path,
    mask_path: Path,
    out_dir: Path,
    *,
    roi_labels_path: Path,
    prefix: str,
    enforce_texture_retention: bool,
    enforce_texture_anti_gaming: bool,
) -> dict[str, Path]:
    assert enforce_texture_retention is True
    assert enforce_texture_anti_gaming is True
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{prefix}_texture_audit.png"
    png_path.write_bytes(f"{prefix}:texture".encode("ascii"))
    inputs = {
        "real": Path(real_path).resolve(strict=True),
        "predicted": Path(predicted_path).resolve(strict=True),
        "mask": Path(mask_path).resolve(strict=True),
        "roi_labels": Path(roi_labels_path).resolve(strict=True),
    }
    repository = Path(postseal.__file__).resolve().parents[1]
    sources = {
        "texture_audit": repository / "scripts" / "visualize_texture_audit.py",
        "strict_pair_visualizer": (
            repository / "scripts" / "visualize_4d_comparison.py"
        ),
        "boundary_safe_high_pass": repository / "utils" / "spatial_detail.py",
    }
    manifest = _signed(
        {
            "schema": postseal.TEXTURE_AUDIT_SCHEMA,
            "scope": "synthetic test texture audit",
            "target_blind_provenance": None,
            "target_blind_note": "rewritten only after post-seal authentication",
            "quality_gate": _passing_texture_gate(),
            "inputs": {
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in inputs.items()
            },
            "geometry": {
                "shape_xyzt": list(postseal.EXPECTED_SHAPE),
                "voxel_sizes_mm": list(postseal.EXPECTED_VOXEL_SIZE_MM),
                "tr_seconds": postseal.EXPECTED_TR_SECONDS,
                "mask_resampled": False,
            },
            "display_contract": {
                "image_interpolation": "nearest",
                "fMRI_registration_or_resampling": False,
                "selection_uses_prediction": False,
            },
            "implementation_sources": {
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in sources.items()
            },
            "outputs": {
                "texture_audit_png": {
                    "path": str(png_path.resolve(strict=True)),
                    "sha256": _sha(png_path),
                }
            },
        }
    )
    manifest_path = out_dir / f"{prefix}_texture_audit.json"
    _write_json(manifest_path, manifest)
    return {"png": png_path, "manifest": manifest_path}


def _install_held_publication_attack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    target_suffix: str,
    threat: str,
) -> dict[str, Any]:
    original = postseal._HeldOutputDirectory.write_new  # noqa: SLF001
    state: dict[str, Any] = {"triggered": False}

    def attacked(self, name: str, payload: bytes):
        if not state["triggered"] and name.endswith(target_suffix):
            state["triggered"] = True
            destination = self.path / name
            sentinel_payload = f"sentinel:{target_suffix}:{threat}".encode("ascii")
            if threat in {"symlink", "hardlink"}:
                sentinel = tmp_path / f"sentinel-{threat}-{len(target_suffix)}"
                sentinel.write_bytes(sentinel_payload)
                if threat == "symlink":
                    destination.symlink_to(sentinel)
                else:
                    destination.hardlink_to(sentinel)
                state["sentinel"] = sentinel
                state["sentinel_payload"] = sentinel_payload
            elif threat == "root-swap":
                detached = self.path.with_name(f"{self.path.name}.detached")
                self.path.rename(detached)
                self.path.mkdir(mode=0o700)
                sentinel = self.path / name
                sentinel.write_bytes(sentinel_payload)
                state["sentinel"] = sentinel
                state["sentinel_payload"] = sentinel_payload
            elif threat == "precommit-mutation":
                snapshot = original(self, name, payload)
                with snapshot.path.open("ab") as stream:
                    stream.write(b"injected-precommit-mutation")
                state["mutated_path"] = snapshot.path
                return snapshot
            else:  # pragma: no cover - parametrization is closed
                raise AssertionError(threat)
        return original(self, name, payload)

    monkeypatch.setattr(postseal._HeldOutputDirectory, "write_new", attacked)  # noqa: SLF001
    return state


def test_authenticates_exact_complete_prediction_set(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    events: list[str] = []
    authenticated = _authenticate_prediction_set_for_test(
        root, pins=pins, access_observer=events.append
    )
    assert [item.scan_id for item in authenticated.subjects] == list(
        postseal.EXPECTED_SCAN_IDS
    )
    assert (
        len([event for event in events if event.startswith("prediction_bytes:")]) == 34
    )


def test_public_orchestrator_has_no_callback_seam_or_corrupt_last_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    final_record = (
        root / "subjects" / postseal.EXPECTED_SCAN_IDS[-1] / "prediction.json"
    )
    with final_record.open("ab") as stream:
        stream.write(b"\ncorrupt-last-record")
    events: list[str] = []
    target_path = tmp_path / "must-not-open-target.json"
    original_canonical_path = postseal._canonical_path  # noqa: SLF001

    def observe_target_path(path, **kwargs):
        if path == target_path:
            events.append("target-path-observed")
        return original_canonical_path(path, **kwargs)

    monkeypatch.setattr(postseal, "_canonical_path", observe_target_path)

    def caller_code(event: str) -> None:
        events.append(event)

    assert not hasattr(postseal, "authenticate_prediction_set")
    assert (
        "access_observer"
        not in inspect.signature(postseal.run_postseal_evaluation).parameters
    )
    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=target_path,
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
            access_observer=caller_code,
        )
    assert events == []
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=target_path,
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
        )
    assert events == []


@pytest.mark.parametrize("threat", ["pins-subclass", "str-subclass", "path-subclass"])
def test_corrupt_last_record_rejects_executable_input_subclasses_without_effect(
    tmp_path: Path, threat: str
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    final_record = (
        root / "subjects" / postseal.EXPECTED_SCAN_IDS[-1] / "prediction.json"
    )
    with final_record.open("ab") as stream:
        stream.write(b"\ncorrupt-last-record")
    effects: list[str] = []

    class ExecutablePins(postseal.EvaluationPins):
        def __getattribute__(self, name: str):
            effects.append(f"pins:{name}")
            return super().__getattribute__(name)

    class ExecutableString(str):
        def __str__(self) -> str:
            effects.append("str")
            return super().__str__()

        def __fspath__(self) -> str:
            effects.append("fspath")
            return str(self)

    class ExecutablePath(type(Path())):
        def __str__(self) -> str:
            effects.append("path-str")
            return super().__str__()

        def __fspath__(self) -> str:
            effects.append("path-fspath")
            return super().__fspath__()

    supplied_root: object = root
    supplied_pins: object = pins
    if threat == "pins-subclass":
        supplied_pins = object.__new__(ExecutablePins)
        for field in postseal.EvaluationPins.__dataclass_fields__:
            object.__setattr__(
                supplied_pins, field, object.__getattribute__(pins, field)
            )
    elif threat == "str-subclass":
        supplied_root = ExecutableString(str(root))
    else:
        supplied_root = ExecutablePath(root)
    effects.clear()
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.run_postseal_evaluation(  # type: ignore[arg-type]
            prediction_root=supplied_root,
            pins=supplied_pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
        )
    assert effects == []


def test_private_capability_and_injection_keywords_cannot_enter_public_apis() -> None:
    for removed in (
        "authenticate_stage_b_targets",
        "load_stage_b_verifier",
        "authenticate_slim_brain_binding",
        "_authenticate_prediction_set_for_test",
        "_authenticate_stage_b_targets_for_test",
        "_authenticate_slim_brain_binding_for_test",
        "_run_postseal_evaluation_for_test",
        "_TEST_ONLY_CAPABILITY",
        "_EXECUTION_BOUNDARY_CAPABILITY",
        "_ExecutionBoundaryAuthority",
        "_TargetBoundaryGrant",
        "_mint_target_boundary_grant",
        "_require_target_boundary_grant",
        "_StageBVerifierAuthority",
        "_LoadedStageBVerifier",
        "_load_stage_b_verifier",
        "_open_authenticated_stage_b_targets",
        "_capture_stage_b_source",
        "AuthenticatedPrediction",
        "AuthenticatedPredictionSet",
    ):
        assert not hasattr(postseal, removed)
    public_parameters = {
        postseal.run_postseal_evaluation: {
            "stage_b_verifier",
            "stage_b_verifier_authority",
            "access_observer",
            "slim_brain",
            "evaluation_feature_extractor",
            "test_capability",
            "runtime_evidence",
        },
        postseal.verify_evaluation_publication: {"_logical_root", "logical_root"},
    }
    for function, forbidden in public_parameters.items():
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
        assert "_TEST_ONLY_CAPABILITY" not in inspect.getsource(function)
    production_source = Path(postseal.__file__).read_text(encoding="utf-8")
    assert "_TEST_ONLY_CAPABILITY" not in production_source
    assert "_for_test" not in production_source
    assert "exec(\n            compile(" not in production_source


@pytest.mark.parametrize("mutation", ["partial", "reordered", "extra", "replaced"])
def test_partial_reordered_extra_or_replaced_prediction_sets_fail(
    tmp_path: Path, mutation: str
) -> None:
    root, _ = _prediction_fixture(tmp_path)
    seal_path = root / "prediction_set_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal.pop("record_sha256")
    if mutation == "partial":
        seal["subjects"] = seal["subjects"][:-1]
        seal["num_subjects"] = 33
    elif mutation == "reordered":
        seal["subjects"][0], seal["subjects"][1] = (
            seal["subjects"][1],
            seal["subjects"][0],
        )
    elif mutation == "extra":
        seal["subjects"].append(dict(seal["subjects"][-1]))
        seal["num_subjects"] = 35
    else:
        seal["subjects"][0]["scan_id"] = "B00000000_000"
    seal["subjects_sha256"] = postseal.canonical_sha256(seal["subjects"])
    seal = _signed(seal)
    _write_json(seal_path, seal)
    with pytest.raises(postseal.PostsealEvaluationError):
        test_postseal.authenticate_prediction_set(root, pins=_pins(_sha(seal_path)))


def test_extra_prediction_tree_file_fails(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    (
        root / "subjects" / postseal.EXPECTED_SCAN_IDS[0] / "alternate.nii.gz"
    ).write_bytes(b"replacement")
    with pytest.raises(postseal.PostsealEvaluationError, match="extra"):
        test_postseal.authenticate_prediction_set(root, pins=pins)


@pytest.mark.parametrize("artifact", ["record", "nifti"])
def test_subject_record_or_nifti_tamper_fails(tmp_path: Path, artifact: str) -> None:
    root, pins = _prediction_fixture(tmp_path)
    subject = root / "subjects" / postseal.EXPECTED_SCAN_IDS[-1]
    path = subject / (
        "prediction.json" if artifact == "record" else "prediction.nii.gz"
    )
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(postseal.PostsealEvaluationError):
        test_postseal.authenticate_prediction_set(root, pins=pins)


def test_target_verifier_is_never_invoked_before_full_prediction_authentication(
    tmp_path: Path,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    final_prediction = (
        root / "subjects" / postseal.EXPECTED_SCAN_IDS[-1] / "prediction.nii.gz"
    )
    final_prediction.write_bytes(b"tampered")
    caller_events: list[str] = []

    def forbidden_caller_code(_event: str) -> None:
        caller_events.append(_event)

    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            stage_b_verifier_authority=_test_verifier_authority(),
            output_root=tmp_path / "output",
            access_observer=forbidden_caller_code,
        )
    assert caller_events == []

    def forbidden_verifier(*_args) -> dict[str, Any]:
        caller_events.append("verifier")
        return {}

    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            stage_b_verifier=forbidden_verifier,
            stage_b_verifier_authority=_test_verifier_authority(),
            output_root=tmp_path / "output",
        )
    assert caller_events == []
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
        )


def test_stage_b_verifier_module_is_not_imported_before_prediction_authentication(
    tmp_path: Path,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    prediction = (
        root / "subjects" / postseal.EXPECTED_SCAN_IDS[-1] / "prediction.nii.gz"
    )
    prediction.write_bytes(b"tampered")
    package = tmp_path / "trap_stage_b"
    package.mkdir()
    marker = tmp_path / "forbidden-target-import-marker"
    init_path = package / "__init__.py"
    controller = package / "stage_b_controller.py"
    source = package / "sealed_target_stage_b.py"
    init_path.write_text("", encoding="utf-8")
    controller.write_text("VALUE = 1\n", encoding="utf-8")
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('opened')\n"
        "def verify_postseal_target_completed_set(*args):\n"
        "    raise RuntimeError('must not run')\n",
        encoding="utf-8",
    )
    authority = test_postseal._StageBVerifierAuthority(  # noqa: SLF001
        source,
        _sha(source),
        ((init_path, _sha(init_path)), (controller, _sha(controller))),
    )
    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open.json",
            stage_b_completed_set_sha256="5" * 64,
            stage_b_verifier_authority=authority,
            output_root=tmp_path / "output",
        )
    assert not marker.exists()
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open.json",
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
        )
    assert not marker.exists()


def test_target_cannot_open_without_preimport_runtime_and_one_gpu_evidence(
    tmp_path: Path,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="execution authority path type differs",
    ):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
        )
    assert not (tmp_path / "must-not-open-target.json").exists()


def test_execution_authority_reopens_exact_scientific_input_descriptors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen-authority-inputs"
    root.mkdir()
    raw_config = root / "raw-config.yaml"
    immutable_split = root / "immutable-split.json"
    prediction_seal = root / "prediction_set_seal.json"
    completed_set = root / "stage_b_completed_set.json"
    raw_config.write_bytes(b"schema: exact-test-config\n")
    immutable_split.write_bytes(b'{"split":"immutable-test"}\n')
    _write_json(
        prediction_seal,
        _signed({"format": postseal.PREDICTION_SET_FORMAT}),
    )
    _write_json(
        completed_set,
        _signed({"schema": postseal.STAGE_B_COMPLETED_SET_SCHEMA}),
    )
    for path in (raw_config, immutable_split, prediction_seal, completed_set):
        path.chmod(0o444)
    root.chmod(0o555)
    bindings = {
        "raw_config": _artifact(raw_config),
        "immutable_split": _artifact(immutable_split),
        "prediction_set_seal": _artifact(prediction_seal, signed=True),
        "stage_b_completed_set": _artifact(completed_set, signed=True),
    }
    try:
        observed = test_postseal._authenticate_execution_bindings(  # noqa: SLF001
            bindings
        )
        assert observed == bindings
        forged = copy.deepcopy(bindings)
        forged["raw_config"]["sha256"] = "0" * 64
        with pytest.raises(postseal.PostsealEvaluationError, match="SHA-256 differs"):
            test_postseal._authenticate_execution_bindings(forged)  # noqa: SLF001
        extra = copy.deepcopy(bindings)
        extra["untrusted"] = _artifact(raw_config)
        with pytest.raises(postseal.PostsealEvaluationError, match="fields differ"):
            test_postseal._authenticate_execution_bindings(extra)  # noqa: SLF001
    finally:
        root.chmod(0o700)
        for path in (raw_config, immutable_split, prediction_seal, completed_set):
            path.chmod(0o600)


def test_published_execution_evidence_rejects_resigned_arbitrary_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "schema": test_postseal.PRODUCTION_EXECUTION_EVIDENCE_SCHEMA,
        "authority": {"path": "/authority", "sha256": "1" * 64},
        "scheduler": {"hostname": "gpu-01"},
        "bindings": {"prediction_set_seal": {"sha256": "2" * 64}},
        "restrictions": {"evaluation_only": True},
        "barrier_transitions": [
            "ALL_34_PREDICTION_RECORDS_AND_BYTES_AUTHENTICATED",
            "EXTERNAL_EXECUTION_AUTHORITY_AUTHENTICATED",
        ],
    }
    base["record_sha256"] = postseal.canonical_sha256(base)
    boundary = test_postseal._ExecutionBoundaryAuthority(  # noqa: SLF001
        evidence=base,
        stage_b_verifier_authority=_test_verifier_authority(),
        capability=test_postseal._EXECUTION_BOUNDARY_CAPABILITY,  # noqa: SLF001
    )
    calls: list[tuple[Path, str, bool]] = []

    def acquire(path, digest, *, require_live_scheduler=True):
        calls.append((path, digest, require_live_scheduler))
        return boundary

    monkeypatch.setattr(test_postseal, "_acquire_execution_boundary", acquire)
    expected = dict(base)
    expected["barrier_transitions"] = [
        "ALL_34_PREDICTION_RECORDS_AND_BYTES_AUTHENTICATED",
        "EXTERNAL_EXECUTION_AUTHORITY_AUTHENTICATED",
        "ALL_34_TARGET_PUBLICATIONS_AND_BYTES_AUTHENTICATED",
        "ALL_34_NATIVE_NIFTI_PAIRS_CONTENT_PREFLIGHTED",
        "METRIC_IMPORT_BARRIER_OPEN",
    ]
    expected["record_sha256"] = postseal.canonical_sha256(
        {key: value for key, value in expected.items() if key != "record_sha256"}
    )
    assert (
        test_postseal.authenticate_published_execution_evidence(
            expected,
            authority_path="/frozen/execution-authority.json",
            authority_sha256="3" * 64,
        )
        == expected
    )
    forged = copy.deepcopy(expected)
    forged["scheduler"] = {"hostname": "login-01", "forged": True}
    forged["record_sha256"] = postseal.canonical_sha256(
        {key: value for key, value in forged.items() if key != "record_sha256"}
    )
    with pytest.raises(postseal.PostsealEvaluationError, match="frozen authority"):
        test_postseal.authenticate_published_execution_evidence(
            forged,
            authority_path="/frozen/execution-authority.json",
            authority_sha256="3" * 64,
        )
    assert calls == [
        (Path("/frozen/execution-authority.json"), "3" * 64, False),
        (Path("/frozen/execution-authority.json"), "3" * 64, False),
    ]


def test_publication_verifier_stops_at_pair_preflight_before_metric_bundle_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_final_pair(**_kwargs):
        events.append("complete-cohort-preflight")
        raise postseal.PostsealEvaluationError("corrupt final native pair")

    def forbidden_bundle_verifier(*_args, **_kwargs):
        events.append("metric-or-bundle-verifier")
        raise AssertionError("bundle/metric code ran before complete pair preflight")

    monkeypatch.setattr(
        test_postseal, "preflight_external_publication_authorities", fail_final_pair
    )
    monkeypatch.setattr(
        postseal, "_verify_evaluation_publication_impl", forbidden_bundle_verifier
    )
    with pytest.raises(
        postseal.PostsealEvaluationError, match="corrupt final native pair"
    ):
        postseal.verify_evaluation_publication(
            "/must-not-open/postseal-output",
            expected_evaluation_sha256="1" * 64,
            expected_receipt_sha256="2" * 64,
            expected_prediction_set_seal_path="/sealed/prediction_set_seal.json",
            expected_prediction_set_seal_sha256="3" * 64,
            expected_stage_b_completed_set_path="/sealed/stage_b_completed.json",
            expected_stage_b_completed_set_sha256="4" * 64,
            expected_stage_b_verifier_source_path="/sealed/sealed_target_stage_b.py",
            expected_stage_b_verifier_source_sha256="5" * 64,
            expected_stage_b_verifier_dependency_paths=(
                "/sealed/__init__.py",
                "/sealed/stage_b_controller.py",
            ),
            expected_stage_b_verifier_dependency_sha256s=("6" * 64, "7" * 64),
            expected_execution_authority_path="/sealed/execution-authority.json",
            expected_execution_authority_sha256="8" * 64,
        )
    assert events == ["complete-cohort-preflight"]


def test_importer_cannot_reconstruct_grant_or_execute_chosen_stage_b(
    tmp_path: Path,
) -> None:
    package = tmp_path / "forged-grant-stage-b"
    marker = tmp_path / "forged-grant-executed"
    package.mkdir()
    source = package / "sealed_target_stage_b.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def verify_postseal_target_completed_set(*_args):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    exploit = compile(
        "authority_type = production._StageBVerifierAuthority\n"
        "execution_type = production._ExecutionBoundaryAuthority\n"
        "grant_type = production._TargetBoundaryGrant\n"
        "capability = production._EXECUTION_BOUNDARY_CAPABILITY\n"
        "mint = production._mint_target_boundary_grant\n"
        "open_targets = production._open_authenticated_stage_b_targets\n",
        "ordinary_importer_forgery.py",
        "exec",
    )
    with pytest.raises(AttributeError):
        exec(
            exploit,
            {
                "production": __import__(
                    "eval.postseal", fromlist=["run_postseal_evaluation"]
                )
            },
        )
    assert not marker.exists()
    production_source = Path(postseal.__file__).read_text(encoding="utf-8")
    assert "_EXECUTION_BOUNDARY_CAPABILITY" not in production_source
    assert "_mint_target_boundary_grant" not in production_source
    assert "_load_stage_b_verifier" not in production_source
    assert "sealed_target_stage_b.py" not in production_source


def test_plain_runtime_mapping_cannot_cross_target_boundary(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=tmp_path / "must-not-open-target.json",
            stage_b_completed_set_sha256="5" * 64,
            output_root=tmp_path / "output",
            runtime_evidence={
                "immutable_runtime": {"all_checks": True},
                "scheduler": {"all_checks": True},
            },
        )


def test_staging_setup_failure_leaves_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    _install_test_execution_boundary(monkeypatch)
    original_mkdir = Path.mkdir

    def fail_private_inputs(path: Path, *args, **kwargs) -> None:
        if path.name == ".authenticated_inputs":
            raise OSError("injected private-input setup failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_private_inputs)
    output = tmp_path / "setup-failure-output"
    with pytest.raises(OSError, match="setup failure"):
        _run_postseal_evaluation_for_test(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=verify_postseal_target_completed_set,
            output_root=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.staged-*"))


def test_stage_b_target_and_receipt_tamper_fail(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    assert _FAKE_STAGE_B_RESULT is not None
    subject = _FAKE_STAGE_B_RESULT["subjects"][10]
    Path(subject["success_receipt"]["path"]).write_text("tamper", encoding="utf-8")
    with pytest.raises(postseal.PostsealEvaluationError):
        _authenticate_stage_b_targets_for_test(
            completed,
            completed_set_sha256=completed_sha,
            predictions=predictions,
            pins=pins,
            verifier=verify_postseal_target_completed_set,
            verifier_authority=authority,
        )


def test_stage_b_adapter_authenticates_all_fixed_targets_after_predictions(
    tmp_path: Path,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    events: list[str] = []
    predictions = _authenticate_prediction_set_for_test(
        root, pins=pins, access_observer=events.append
    )
    events.append("prediction_set_fully_authenticated")
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    targets = _authenticate_stage_b_targets_for_test(
        completed,
        completed_set_sha256=completed_sha,
        predictions=predictions,
        pins=pins,
        verifier=verify_postseal_target_completed_set,
        verifier_authority=authority,
        access_observer=events.append,
    )
    assert [item.scan_id for item in targets.subjects] == list(
        postseal.EXPECTED_SCAN_IDS
    )
    boundary = events.index("target_verifier_invoked")
    assert boundary > events.index("prediction_set_fully_authenticated")
    assert all(
        events.index(f"prediction_bytes:{scan_id}") < boundary
        for scan_id in postseal.EXPECTED_SCAN_IDS
    )
    assert len([event for event in events if event.startswith("target_bytes:")]) == 34


def test_stage_b_missing_independent_brain_support_fails_closed(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    assert _FAKE_STAGE_B_RESULT is not None
    _FAKE_STAGE_B_RESULT["subjects"][0].pop("native_structural_brain_mask")
    unsigned = dict(_FAKE_STAGE_B_RESULT)
    unsigned.pop("record_sha256")
    _FAKE_STAGE_B_RESULT["record_sha256"] = postseal.canonical_sha256(unsigned)
    with pytest.raises(
        postseal.PostsealEvaluationError, match="subject evidence fields"
    ):
        _authenticate_stage_b_targets_for_test(
            completed,
            completed_set_sha256=completed_sha,
            predictions=predictions,
            pins=pins,
            verifier=verify_postseal_target_completed_set,
            verifier_authority=authority,
        )


def test_stage_b_source_tamper_fails(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    assert _FAKE_STAGE_B_RESULT is not None
    target = Path(_FAKE_STAGE_B_RESULT["subjects"][20]["native_bold"]["path"])
    target.write_bytes(b"tamper")
    with pytest.raises(postseal.PostsealEvaluationError):
        _authenticate_stage_b_targets_for_test(
            completed,
            completed_set_sha256=completed_sha,
            predictions=predictions,
            pins=pins,
            verifier=verify_postseal_target_completed_set,
            verifier_authority=authority,
        )


@pytest.mark.parametrize("failure", ["shape", "affine", "tr", "domain"])
def test_pair_grid_affine_tr_and_domain_tamper_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _install_test_native_contract(monkeypatch, (3, 4, 5, 4))
    real = np.full((3, 4, 5, 4), 0.25, dtype=np.float32)
    predicted = np.full_like(real, 0.30)
    brain_mask = _interior_brain_mask((3, 4, 5))
    real *= brain_mask[..., None]
    labels = np.zeros((3, 4, 5), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[:, 2:, :][brain_mask[:, 2:, :].astype(bool)] = 2
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    brain_mask_path = tmp_path / "brain_mask.nii.gz"
    labels_path = tmp_path / "labels.nii.gz"
    _write_nifti(real_path, real)
    _write_nifti(brain_mask_path, brain_mask)
    if failure == "shape":
        predicted = predicted[:, :, :-1]
    if failure == "domain":
        predicted.flat[0] = 1.1
    affine = None
    if failure == "affine":
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        affine[0, 3] = 3.0
    _write_nifti(
        predicted_path,
        predicted,
        affine=affine,
        tr=2.0 if failure == "tr" else None,
    )
    _write_nifti(labels_path, labels)
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.validate_exact_pair(
            real_path,
            predicted_path,
            brain_mask_path,
            labels_path,
            protocol_profile="a4-native-recovery-v1",
        )


@pytest.mark.parametrize(
    "attack",
    ("text", "concatenated-gzip", "float64", "extension", "millisecond-unit"),
)
def test_strict_native_nifti_rejects_container_header_and_dtype_ambiguity(
    tmp_path: Path, attack: str
) -> None:
    path = tmp_path / "attacked.nii.gz"
    values = np.full((3, 4, 5, 4), 0.25, dtype=np.float32)
    if attack == "text":
        path.write_bytes(b"ordinary text is not a NIfTI")
    else:
        image = nib.Nifti1Image(
            values.astype(np.float64) if attack == "float64" else values,
            np.diag([3.0, 3.0, 3.0, 1.0]),
        )
        image.header.set_zooms((3.0, 3.0, 3.0, 3.0))
        image.header.set_xyzt_units(
            "mm", "msec" if attack == "millisecond-unit" else "sec"
        )
        if attack == "millisecond-unit":
            image.header.set_zooms((3.0, 3.0, 3.0, 3000.0))
        if attack == "extension":
            image.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"extra"))
        nib.save(image, str(path))
        if attack == "concatenated-gzip":
            path.write_bytes(path.read_bytes() + gzip.compress(b"second member"))
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal._strict_nifti_values(  # noqa: SLF001
            path,
            label="attacked native fMRI",
            expected_shape=(3, 4, 5, 4),
        )


def test_exact_pair_accepts_only_independent_structural_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_test_native_contract(monkeypatch, (3, 4, 5, 4))
    real = np.full((3, 4, 5, 4), 0.25, dtype=np.float32)
    predicted = np.full_like(real, 0.30)
    brain_mask = _interior_brain_mask((3, 4, 5))
    real *= brain_mask[..., None]
    labels = np.zeros((3, 4, 5), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[:, 2:, :][brain_mask[:, 2:, :].astype(bool)] = 2
    paths = [
        tmp_path / name
        for name in (
            "real.nii.gz",
            "pred.nii.gz",
            "brain_mask.nii.gz",
            "labels.nii.gz",
        )
    ]
    _write_nifti(paths[0], real)
    _write_nifti(paths[1], predicted)
    _write_nifti(paths[2], brain_mask)
    _write_nifti(paths[3], labels)
    result = postseal.validate_exact_pair(
        *paths, protocol_profile="a4-native-recovery-v1"
    )
    assert result["fmri_resampled"] is False
    assert result["mask_or_labels_derived_from_real_or_prediction"] is False
    assert result["affines_exactly_equal"] is True
    assert result["structural_brain_mask"]["binary_exact"] is True
    assert result["target_validity_mask"] == {
        "contract": postseal.TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[
            "a4-native-recovery-v1"
        ],
        "derivation": ("exact-nonzero-support-across-final-stored-target-time-series"),
        "shape": [3, 4, 5],
        "foreground_voxel_count": int(brain_mask.sum()),
        "foreground_fraction_of_structural_support": 1.0,
        "structural_foreground_voxel_count": int(brain_mask.sum()),
        "structural_voxels_excluded_from_target_metrics": 0,
        "outside_structural_brain_voxel_count": 0,
        "configured_structural_roi_count": 2,
        "all_configured_structural_rois_have_target_validity_support": True,
        "equals_exact_final_nonzero_support": True,
        "exactly_binary": True,
        "fmri_resampled": False,
    }


def test_sparse_roi_labels_cannot_shrink_independent_brain_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_test_native_contract(monkeypatch, (11, 11, 11, 4))
    real = np.full((11, 11, 11, 4), 0.25, dtype=np.float32)
    brain_mask = _interior_brain_mask((11, 11, 11))
    real *= brain_mask[..., None]
    labels = np.zeros((11, 11, 11), dtype=np.int16)
    labels[3, 3, 3] = 1
    labels[7, 7, 7] = 2
    paths = [
        tmp_path / name
        for name in ("real.nii.gz", "pred.nii.gz", "brain.nii.gz", "labels.nii.gz")
    ]
    for path, values in zip(paths, (real, real, brain_mask, labels)):
        _write_nifti(path, values)
    result = postseal.validate_exact_pair(
        *paths, protocol_profile="a4-native-recovery-v1"
    )
    assert result["positive_structural_label_count"] == 2
    assert result["structural_brain_mask"]["foreground_voxel_count"] == int(
        brain_mask.sum()
    )
    assert (
        result["structural_brain_mask"]["roi_foreground_outside_mask_voxel_count"] == 0
    )


def test_postseal_validity_excludes_structural_zeros_and_rejects_bold_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_test_native_contract(monkeypatch, (11, 11, 11, 4))
    structural = _interior_brain_mask((11, 11, 11))
    real = np.repeat(structural[..., None].astype(np.float32), 4, axis=-1) * 0.25
    excluded = tuple(np.argwhere(structural)[0])
    real[excluded] = 0.0
    labels = np.zeros(structural.shape, dtype=np.int16)
    labels[3, 3, 3] = 1
    labels[7, 7, 7] = 2
    paths = [
        tmp_path / name
        for name in ("real.nii.gz", "pred.nii.gz", "brain.nii.gz", "labels.nii.gz")
    ]
    for path, values in zip(paths, (real, real, structural, labels)):
        _write_nifti(path, values)

    pair = postseal.validate_exact_pair(
        *paths, protocol_profile="a4-native-recovery-v1"
    )
    validity = pair["target_validity_mask"]
    assert validity["foreground_voxel_count"] == int(structural.sum()) - 1
    assert validity["structural_voxels_excluded_from_target_metrics"] == 1
    assert validity["equals_exact_final_nonzero_support"] is True

    real[0, 0, 0, :] = 0.25
    _write_nifti(paths[0], real)
    with pytest.raises(postseal.PostsealEvaluationError, match="BOLD outside"):
        postseal.validate_exact_pair(*paths, protocol_profile="a4-native-recovery-v1")

    real[0, 0, 0, :] = 0.0
    real[3, 3, 3, :] = 0.0
    _write_nifti(paths[0], real)
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match=r"every configured structural ROI.*missing labels: \[1\]",
    ):
        postseal.validate_exact_pair(*paths, protocol_profile="a4-native-recovery-v1")


def test_missing_or_empty_independent_brain_support_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_test_native_contract(monkeypatch, (11, 11, 11, 4))
    real = np.full((11, 11, 11, 4), 0.25, dtype=np.float32)
    labels = np.zeros((11, 11, 11), dtype=np.int16)
    labels[3, 3, 3] = 1
    labels[7, 7, 7] = 2
    paths = [
        tmp_path / name
        for name in ("real.nii.gz", "pred.nii.gz", "empty.nii.gz", "labels.nii.gz")
    ]
    for path, values in zip(
        paths, (real, real, np.zeros((11, 11, 11), dtype=np.uint8), labels)
    ):
        _write_nifti(path, values)
    with pytest.raises(postseal.PostsealEvaluationError, match="brain support"):
        postseal.validate_exact_pair(*paths, protocol_profile="a4-native-recovery-v1")


def test_default_fid_is_are_explicitly_unavailable() -> None:
    evidence = postseal._distributional_metrics_unavailable()  # noqa: SLF001
    assert evidence["schema"] == postseal._DISTRIBUTIONAL_UNAVAILABLE_SCHEMA  # noqa: SLF001
    assert evidence["available"] is False
    assert evidence["evaluation_feature_model_loaded"] is False
    assert evidence["fid"] is None
    assert evidence["inception_score"] is None
    assert "not substituted or fabricated" in evidence["reason"]


def test_cosmetic_in_memory_slimbrain_object_cannot_enter_production() -> None:
    attribute_reads: list[str] = []

    class CosmeticExtractor:
        def __getattribute__(self, name: str):
            attribute_reads.append(name)
            return object.__getattribute__(self, name)

        connect4_model_name = "SLIMBrain"
        connect4_adapter_contract = "connect4-slimbrain-4d-adapter-v1"
        checkpoint_sha256 = "a" * 64

        def forward(self, *_args):
            raise AssertionError("cosmetic extractor must never execute")

    extractor = CosmeticExtractor()
    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root="/not/opened",
            pins=object(),
            stage_b_completed_set="/not/opened",
            stage_b_completed_set_sha256="5" * 64,
            output_root="/not/opened",
            evaluation_feature_extractor=extractor,
        )
    assert attribute_reads == []


def test_no_synthetic_feature_model_capability_exists_in_runtime_source() -> None:
    source = Path(postseal.__file__).read_text(encoding="utf-8")
    assert "test_feature_extractor" not in source
    assert "evaluation_feature_extractor" not in source
    assert "test_only_synthetic_feature_model" not in source


def test_output_alias_and_overwrite_are_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        postseal._canonical_path(  # noqa: SLF001 - adversarial boundary test
            existing, label="output", must_exist=False
        )
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(postseal.PostsealEvaluationError, match="aliases"):
        postseal._canonical_path(  # noqa: SLF001 - adversarial boundary test
            alias_parent / "output", label="output", must_exist=False
        )


@pytest.mark.parametrize("raw_path", ("~/postseal-output", "$TMPDIR/postseal-output"))
def test_canonical_path_rejects_raw_user_or_environment_indirection(
    raw_path: str,
) -> None:
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="cannot use environment or user indirection",
    ):
        postseal._canonical_path(  # noqa: SLF001 - adversarial boundary test
            raw_path, label="postseal output", must_exist=False
        )


def test_module_imports_no_training_inference_or_model_code() -> None:
    source_path = Path(postseal.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {"training", "inference", "models"}
    direct_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            direct_roots.add(node.module.split(".", 1)[0])
    assert direct_roots.isdisjoint(forbidden)
    script = (
        "import sys; import eval.postseal; "
        "assert 'training' not in sys.modules; "
        "assert 'inference' not in sys.modules; "
        "assert 'models.connect4' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_path.parents[1],
        check=True,
        env={**os.environ, "PYTHONPATH": str(source_path.parents[1])},
    )


def test_public_module_and_cli_import_before_barrier_block_all_metric_model_layers() -> (
    None
):
    repository = Path(postseal.__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys

class BlockEarlyMetricModelImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            fullname == 'torch'
            or fullname.startswith('torch.')
            or fullname == 'eval.postseal_metrics'
            or fullname == 'eval.quality'
            or fullname == 'scripts.visualize_4d_comparison'
            or fullname == 'scripts.visualize_texture_audit'
            or fullname == 'models'
            or fullname.startswith('models.')
            or fullname == 'training'
            or fullname.startswith('training.')
            or fullname == 'inference'
            or fullname.startswith('inference.')
        )
        if forbidden:
            raise RuntimeError('EARLY_IMPORT:' + fullname)
        return None

sys.meta_path.insert(0, BlockEarlyMetricModelImport())
sys.path.insert(0, {str(repository)!r})
import eval.postseal
import scripts.evaluate_postseal_heldout
for name in (
    'torch',
    'eval.postseal_metrics',
    'eval.quality',
    'scripts.visualize_4d_comparison',
    'scripts.visualize_texture_audit',
):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=repository,
        check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )


def test_cli_help_does_not_import_model_or_training_code() -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_postseal_heldout",
            "--help",
        ],
        cwd=repository,
        check=True,
        env={**os.environ, "PYTHONPATH": str(repository)},
        capture_output=True,
        text=True,
    )


def test_stage_b_package_with_pycache_or_extra_siblings_fails_closed() -> None:
    repository = Path(postseal.__file__).resolve().parents[2]
    package = (
        repository / "cluster_validation_20260829" / "native_stage_b_orchestration_v1"
    )
    source = package / "sealed_target_stage_b.py"
    init_path = package / "__init__.py"
    controller = package / "stage_b_controller.py"
    authority = test_postseal._StageBVerifierAuthority(  # noqa: SLF001
        source_path=source,
        source_sha256=_sha(source),
        dependency_sources=(
            (init_path, _sha(init_path)),
            (controller, _sha(controller)),
        ),
    )
    with pytest.raises(postseal.PostsealEvaluationError, match="closed exact"):
        test_postseal._load_stage_b_verifier(authority)  # noqa: SLF001


def test_spoofed_stage_b_callable_name_and_filename_cannot_enter_public_path(
    tmp_path: Path,
) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    marker = tmp_path / "spoofed-verifier-called"
    namespace: dict[str, Any] = {}
    exec(
        compile(
            "def verify_postseal_target_completed_set(*_args):\n"
            f"    open({str(marker)!r}, 'w').write('called')\n"
            "    return {}\n",
            str(authority.source_path),
            "exec",
        ),
        namespace,
    )
    spoofed = namespace["verify_postseal_target_completed_set"]
    assert spoofed.__name__ == authority.function_name
    assert Path(spoofed.__code__.co_filename) == authority.source_path
    with pytest.raises(TypeError):
        postseal.run_postseal_evaluation(
            prediction_root=root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=spoofed,
            output_root=tmp_path / "output",
        )
    assert not marker.exists()


def test_stage_b_source_replacement_after_capture_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_marker = tmp_path / "original-stage-b-called"
    malicious_marker = tmp_path / "replacement-stage-b-called"
    package, authority = _closed_stage_b_package(tmp_path, marker=original_marker)
    original_capture = test_postseal._capture_stage_b_source  # noqa: SLF001
    attacked = False

    def replace_after_capture(package_path, descriptor, name, expected_sha256):
        nonlocal attacked
        captured = original_capture(package_path, descriptor, name, expected_sha256)
        if name == "sealed_target_stage_b.py" and not attacked:
            attacked = True
            source = package / name
            source.rename(tmp_path / "captured-stage-b-source")
            source.write_text(
                "from pathlib import Path\n"
                f"Path({str(malicious_marker)!r}).write_text('called')\n"
                "def verify_postseal_target_completed_set(*_args):\n"
                "    return {}\n",
                encoding="utf-8",
            )
        return captured

    monkeypatch.setattr(test_postseal, "_capture_stage_b_source", replace_after_capture)
    with pytest.raises(
        postseal.PostsealEvaluationError, match="identity/bytes changed"
    ):
        test_postseal._load_stage_b_verifier(authority)  # noqa: SLF001
    assert attacked is True
    assert not original_marker.exists()
    assert not malicious_marker.exists()


def test_stage_b_package_root_swap_fails_before_captured_code_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "root-swapped-stage-b-called"
    package, authority = _closed_stage_b_package(tmp_path, marker=marker)
    detached = package.with_name(f"{package.name}.detached")
    original_capture = test_postseal._capture_stage_b_source  # noqa: SLF001
    attacked = False

    def swap_after_capture(package_path, descriptor, name, expected_sha256):
        nonlocal attacked
        captured = original_capture(package_path, descriptor, name, expected_sha256)
        if name == "stage_b_controller.py" and not attacked:
            attacked = True
            package.rename(detached)
            package.mkdir()
        return captured

    monkeypatch.setattr(test_postseal, "_capture_stage_b_source", swap_after_capture)
    with pytest.raises(postseal.PostsealEvaluationError, match="root was replaced"):
        test_postseal._load_stage_b_verifier(authority)  # noqa: SLF001
    assert attacked is True
    assert not marker.exists()


def test_stage_b_function_code_replacement_fails_before_invocation(
    tmp_path: Path,
) -> None:
    original_marker = tmp_path / "original-stage-b-function-called"
    malicious_marker = tmp_path / "replacement-stage-b-function-called"
    _package, authority = _closed_stage_b_package(tmp_path, marker=original_marker)
    loaded = test_postseal._load_stage_b_verifier(authority)  # noqa: SLF001
    try:
        namespace: dict[str, Any] = {}
        exec(
            "def replacement(*_args):\n"
            f"    open({str(malicious_marker)!r}, 'w').write('called')\n"
            "    return {}\n",
            namespace,
        )
        loaded.function.__code__ = namespace["replacement"].__code__
        with pytest.raises(
            postseal.PostsealEvaluationError,
            match="function object was replaced or mutated",
        ):
            loaded.reauthenticate()
    finally:
        loaded.close()
    assert not original_marker.exists()
    assert not malicious_marker.exists()


def test_stage_b_result_cannot_authorize_selection(tmp_path: Path) -> None:
    root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    assert _FAKE_STAGE_B_RESULT is not None
    _FAKE_STAGE_B_RESULT["authorizes_checkpoint_selection"] = True
    unsigned = dict(_FAKE_STAGE_B_RESULT)
    unsigned.pop("record_sha256")
    _FAKE_STAGE_B_RESULT["record_sha256"] = postseal.canonical_sha256(unsigned)
    with pytest.raises(postseal.PostsealEvaluationError, match="contract"):
        _authenticate_stage_b_targets_for_test(
            completed,
            completed_set_sha256=completed_sha,
            predictions=predictions,
            pins=pins,
            verifier=verify_postseal_target_completed_set,
            verifier_authority=authority,
        )


def test_nested_paper_metrics_are_aggregated_without_substitution() -> None:
    records = _quality_subject_records()
    aggregate = postseal._aggregate_paper_metrics(  # noqa: SLF001
        records
    )
    assert aggregate == {
        "mse": pytest.approx(0.01),
        "voxel_corr": pytest.approx(0.8),
        "roi_corr": pytest.approx(0.7),
        "f2f_corr": pytest.approx(0.75),
        "ssim": pytest.approx(0.85),
        "psnr": pytest.approx(20.0),
    }
    records[0]["quality"]["paper_metrics"]["metrics"]["roi_corr"] = {
        "available": False,
        "value": None,
    }
    with pytest.raises(postseal.PostsealEvaluationError, match="roi_corr"):
        postseal._aggregate_paper_metrics(records)  # noqa: SLF001


def test_data_quality_summary_has_complete_grain_rates_and_visual(
    tmp_path: Path,
) -> None:
    records = _quality_subject_records()
    unavailable = postseal._distributional_metrics_unavailable()  # noqa: SLF001
    summary = postseal._build_data_quality_summary(  # noqa: SLF001
        records,
        distributional_evidence=unavailable,
        verifier_dependency_count=2,
    )
    assert summary["dataset_and_grain"]["scan_row_count"] == 34
    assert summary["dataset_and_grain"]["patient_count"] == 13
    assert summary["completeness_identity_and_order"]["fixed_order_exact"] is True
    assert summary["required_input_file_hash_coverage"]["coverage_rate"] == 1.0
    assert summary["contract_rates"] == {
        "finite_unit_interval_rate": 1.0,
        "exact_shape_rate": 1.0,
        "exact_affine_rate": 1.0,
        "exact_orientation_rate": 1.0,
        "exact_tr_rate": 1.0,
        "no_fmri_resampling_rate": 1.0,
        "independent_structural_support_rate": 1.0,
    }
    ratios = summary["cohort_distributions"]["predicted_over_real_ratios"]
    assert set(ratios) == {
        "robust_range_ratio",
        "gradient_rms_ratio",
        "laplacian_rms_ratio",
        "high_frequency_rms_ratio",
        "dynamic_high_frequency_rms_ratio",
        "temporal_variance_ratio",
        "dvars_ratio",
        "non_dc_power_ratio",
        "effective_rank_ratio",
    }
    assert all(metric["distribution"]["count"] == 34 for metric in ratios.values())
    assert summary["fitness_for_use"]["status"] == "CONDITIONALLY_FIT"
    assert summary["fitness_for_use"]["severity"] == "medium"
    assert summary["texture_release_gate"]["verdict"] == "pass"
    assert summary["texture_release_gate"]["passing_scan_count"] == 34
    assert summary["cohort_rates"]["texture_release_gate_pass_count"] == 34
    assert summary["cohort_rates"]["texture_release_gate_fail_count"] == 0
    assert summary["fitness_for_use"]["texture_release_gate_pass_count"] == 34
    chart = postseal._render_cohort_quality_chart(  # noqa: SLF001
        summary, tmp_path / "cohort_quality_summary.png"
    )
    assert chart.size_bytes > 10_000
    with Image.open(chart.path) as rendered:
        assert rendered.width >= 1_800
        assert rendered.height >= 1_200
    report = postseal._data_quality_markdown(summary)  # noqa: SLF001
    assert "Fitness for use: CONDITIONALLY_FIT" in report
    assert "Laplacian RMS" in report
    assert "Non-DC power" in report


def _temporary_and_authenticated_inputs(
    tmp_path: Path,
    *,
    staged: Path,
    scan_id: str,
    prediction: postseal.FileSnapshot | None = None,
) -> tuple[dict[str, Path], dict[str, postseal.FileSnapshot]]:
    private = staged / ".authenticated_inputs" / scan_id
    authenticated = tmp_path / f"authenticated-{scan_id}"
    private.mkdir(parents=True)
    authenticated.mkdir()
    temporary_inputs: dict[str, Path] = {}
    descriptors: dict[str, postseal.FileSnapshot] = {}
    for name in ("real", "predicted", "mask", "roi_labels"):
        if name == "predicted" and prediction is not None:
            payload = prediction.path.read_bytes()
            descriptor = prediction
        else:
            payload = f"authenticated:{scan_id}:{name}".encode("ascii")
            source = authenticated / f"{name}.bin"
            source.write_bytes(payload)
            descriptor = postseal._snapshot_file(  # noqa: SLF001
                source, label=f"authenticated {name}"
            )[0]
        temporary = private / f"{name}.bin"
        temporary.write_bytes(payload)
        temporary_inputs[name] = temporary.resolve(strict=True)
        descriptors[name] = descriptor
    return temporary_inputs, descriptors


@pytest.mark.parametrize(
    "threat", ("symlink", "hardlink", "root-swap", "precommit-mutation")
)
def test_visualization_manifest_production_publisher_rejects_path_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threat: str,
) -> None:
    scan_id = postseal.EXPECTED_SCAN_IDS[0]
    staged = tmp_path / "visual-staged"
    public_subject = staged / "subjects" / scan_id
    public_subject.mkdir(parents=True)
    temporary, descriptors = _temporary_and_authenticated_inputs(
        tmp_path, staged=staged, scan_id=scan_id
    )
    generated = _fake_visual_outputs(
        temporary["real"],
        temporary["predicted"],
        temporary["real"].parent / "visualization_raw",
        mask_path=temporary["mask"],
        roi_labels_path=temporary["roi_labels"],
        tr_seconds=postseal.EXPECTED_TR_SECONDS,
        prefix=scan_id,
        fps=2,
    )
    state = _install_held_publication_attack(
        monkeypatch,
        tmp_path,
        target_suffix="_visualization_manifest.json",
        threat=threat,
    )
    with postseal._HeldOutputDirectory.open(public_subject) as publication:  # noqa: SLF001
        with pytest.raises((FileExistsError, postseal.PostsealEvaluationError)):
            postseal._publish_visualization_outputs(  # noqa: SLF001
                generated,
                publication=publication,
                output_root=tmp_path / "visual-final",
                subject_relative_dir=Path("subjects") / scan_id,
                temporary_inputs=temporary,
                input_descriptors=descriptors,
            )
    assert state["triggered"] is True
    if "sentinel" in state:
        assert state["sentinel"].read_bytes() == state["sentinel_payload"]


@pytest.mark.parametrize(
    "threat", ("symlink", "hardlink", "root-swap", "precommit-mutation")
)
def test_texture_manifest_production_publisher_rejects_path_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threat: str,
) -> None:
    scan_id = postseal.EXPECTED_SCAN_IDS[0]
    prediction_root, pins = _prediction_fixture(tmp_path)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    prediction = predictions.by_scan[scan_id]
    staged = tmp_path / "texture-staged"
    (staged / "subjects" / scan_id).mkdir(parents=True)
    temporary, descriptors = _temporary_and_authenticated_inputs(
        tmp_path,
        staged=staged,
        scan_id=scan_id,
        prediction=prediction.prediction_snapshot,
    )
    generated = _fake_texture_outputs(
        temporary["real"],
        temporary["predicted"],
        temporary["mask"],
        temporary["real"].parent / "texture_raw",
        roi_labels_path=temporary["roi_labels"],
        prefix=scan_id,
        enforce_texture_retention=True,
        enforce_texture_anti_gaming=True,
    )
    state = _install_held_publication_attack(
        monkeypatch,
        tmp_path,
        target_suffix="_texture_audit.json",
        threat=threat,
    )
    with postseal._HeldOutputDirectory.open(  # noqa: SLF001
        staged / "subjects" / scan_id / "texture", create=True
    ) as publication:
        with pytest.raises((FileExistsError, postseal.PostsealEvaluationError)):
            postseal._publish_texture_audit(  # noqa: SLF001
                generated,
                publication=publication,
                output_root=tmp_path / "texture-final",
                subject_relative_dir=Path("subjects") / scan_id,
                temporary_inputs=temporary,
                input_descriptors=descriptors,
                prediction=prediction,
                predictions=predictions,
            )
    assert state["triggered"] is True
    if "sentinel" in state:
        assert state["sentinel"].read_bytes() == state["sentinel_payload"]


@pytest.mark.parametrize(
    "threat", ("symlink", "hardlink", "root-swap", "precommit-mutation")
)
def test_cohort_chart_production_publisher_rejects_path_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threat: str,
) -> None:
    summary = postseal._build_data_quality_summary(  # noqa: SLF001
        _quality_subject_records(),
        distributional_evidence=postseal._distributional_metrics_unavailable(),  # noqa: SLF001
        verifier_dependency_count=2,
    )
    public_root = tmp_path / "cohort-publication"
    public_root.mkdir()
    state = _install_held_publication_attack(
        monkeypatch,
        tmp_path,
        target_suffix="cohort_quality_summary.png",
        threat=threat,
    )
    with postseal._HeldOutputDirectory.open(public_root) as publication:  # noqa: SLF001
        with pytest.raises((FileExistsError, postseal.PostsealEvaluationError)):
            postseal._render_cohort_quality_chart(  # noqa: SLF001
                summary,
                public_root / "cohort_quality_summary.png",
                publication=publication,
            )
    assert state["triggered"] is True
    if "sentinel" in state:
        assert state["sentinel"].read_bytes() == state["sentinel_payload"]


@pytest.mark.parametrize(
    ("target_kind", "target_suffix"),
    (
        ("visualization_manifest", "_visualization_manifest.json"),
        ("texture_manifest", "_texture_audit.json"),
        ("cohort_chart", "cohort_quality_summary.png"),
    ),
)
@pytest.mark.parametrize(
    "threat", ("symlink", "hardlink", "root-swap", "precommit-mutation")
)
def test_postseal_entrypoint_rejects_held_publication_path_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    target_suffix: str,
    threat: str,
) -> None:
    """Exercise each protected output through the complete production entrypoint."""
    _install_test_native_contract(monkeypatch, (5, 6, 7, 8))
    coordinates = np.indices((5, 6, 7, 8), dtype=np.float32)
    real = (0.2 + 0.01 * coordinates.sum(axis=0)).astype(np.float32)
    predicted = (0.9 * real + 0.03).astype(np.float32)
    brain_mask = _interior_brain_mask((5, 6, 7))
    real *= brain_mask[..., None]
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[3:, :, :][brain_mask[3:, :, :].astype(bool)] = 2
    prediction_root, _ = _prediction_fixture(tmp_path)
    pins = _materialize_prediction_niftis(prediction_root, predicted)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    completed_sha = _materialize_stage_b_niftis(real, labels, brain_mask)
    monkeypatch.setattr(
        paired_quality,
        "evaluate_4d_pair_quality",
        lambda *_args, **_kwargs: copy.deepcopy(_quality_record()),
    )
    monkeypatch.setattr(visualizer, "generate_comparison_outputs", _fake_visual_outputs)
    monkeypatch.setattr(texture_audit, "generate_texture_audit", _fake_texture_outputs)
    _install_test_execution_boundary(monkeypatch)
    state = _install_held_publication_attack(
        monkeypatch,
        tmp_path,
        target_suffix=target_suffix,
        threat=threat,
    )
    output = tmp_path / f"attacked-{target_kind}-{threat}"
    with pytest.raises((FileExistsError, postseal.PostsealEvaluationError)):
        _run_postseal_evaluation_for_test(
            prediction_root=prediction_root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=verify_postseal_target_completed_set,
            stage_b_verifier_authority=authority,
            output_root=output,
        )
    assert state["triggered"] is True
    assert not output.exists()
    staged_residue = list(tmp_path.glob(f".{output.name}.staged-*"))
    if threat == "root-swap" and target_kind == "cohort_chart":
        # The adversary moved the descriptor-held staging root itself.  It is
        # deliberately preserved rather than following an attacker-selected
        # pathname for deletion; critically, it is never published as output.
        assert len(staged_residue) == 2
        assert any(path.name.endswith(".detached") for path in staged_residue)
        assert any(not path.name.endswith(".detached") for path in staged_residue)
    else:
        assert not staged_residue
    if "sentinel" in state and state["sentinel"].exists():
        assert state["sentinel"].read_bytes() == state["sentinel_payload"]


def test_data_quality_temporal_collapse_is_critical_and_not_fit() -> None:
    records = _quality_subject_records()
    records[0] = copy.deepcopy(records[0])
    records[0]["quality"] = _quality_record(passed=False, collapse=True)
    summary = postseal._build_data_quality_summary(  # noqa: SLF001
        records,
        distributional_evidence=postseal._distributional_metrics_unavailable(),  # noqa: SLF001
        verifier_dependency_count=2,
    )
    assert summary["cohort_rates"]["quality_fail_count"] == 1
    assert summary["cohort_rates"]["temporal_collapse_count"] == 1
    assert summary["fitness_for_use"]["status"] == "NOT_FIT"
    assert summary["fitness_for_use"]["severity"] == "critical"
    assert (
        summary["per_scan_rows"][0]["quality_gate"]["fitness_for_use"]
        == "NOT_FIT_TEMPORAL_COLLAPSE"
    )


def test_atomic_directory_publication_never_overwrites_or_follows_alias(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "payload").write_bytes(b"first")
    destination = tmp_path / "published"
    held = postseal._HeldStagingRoot.open(staged, destination)  # noqa: SLF001
    try:
        frozen = postseal._freeze_staged_tree(staged)  # noqa: SLF001
        postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert (destination / "payload").read_bytes() == b"first"

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "payload").write_bytes(b"replacement")
    held = postseal._HeldStagingRoot.open(replacement, destination)  # noqa: SLF001
    try:
        frozen = postseal._freeze_staged_tree(replacement)  # noqa: SLF001
        with pytest.raises(FileExistsError):
            postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert (destination / "payload").read_bytes() == b"first"
    assert (replacement / "payload").read_bytes() == b"replacement"

    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    another = tmp_path / "another"
    another.mkdir()
    (another / "payload").write_bytes(b"another")
    held = postseal._HeldStagingRoot.open(another, alias)  # noqa: SLF001
    try:
        frozen = postseal._freeze_staged_tree(another)  # noqa: SLF001
        with pytest.raises(FileExistsError):
            postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert (destination / "payload").read_bytes() == b"first"


def test_frozen_tree_inode_replacement_fails_before_commit(tmp_path: Path) -> None:
    staged = tmp_path / "staged-race"
    staged.mkdir()
    payload = staged / "payload"
    payload.write_bytes(b"authenticated")
    destination = tmp_path / "must-not-commit"
    held = postseal._HeldStagingRoot.open(staged, destination)  # noqa: SLF001
    frozen = postseal._freeze_staged_tree(staged)  # noqa: SLF001
    os.chmod(staged, 0o700)
    os.chmod(payload, 0o600)
    payload.unlink()
    payload.write_bytes(b"replacement")
    os.chmod(payload, 0o400)
    os.chmod(staged, 0o500)
    try:
        with pytest.raises(postseal.PostsealEvaluationError, match="inode changed"):
            postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert not destination.exists()
    os.chmod(staged, 0o700)


def test_postrename_parent_fsync_failure_is_explicit_committed_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "staged-fsync"
    staged.mkdir()
    (staged / "payload").write_bytes(b"complete")
    destination = tmp_path / "committed"
    held = postseal._HeldStagingRoot.open(staged, destination)  # noqa: SLF001
    frozen = postseal._freeze_staged_tree(staged)  # noqa: SLF001

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(postseal.os, "fsync", fail_fsync)
    try:
        with pytest.raises(
            postseal.PostsealPublicationUncertainError, match="committed"
        ):
            postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert destination.is_dir()
    assert (destination / "payload").read_bytes() == b"complete"


def test_descriptor_held_publication_rejects_root_swap_at_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "held-staged"
    staged.mkdir()
    (staged / "payload").write_bytes(b"authenticated")
    destination = tmp_path / "must-be-uncertain"
    detached = tmp_path / "held-staged.detached"
    held = postseal._HeldStagingRoot.open(staged, destination)  # noqa: SLF001
    frozen = postseal._freeze_staged_tree(staged)  # noqa: SLF001
    original_rename = postseal._renameat_noreplace  # noqa: SLF001

    def swap_then_rename(parent_descriptor, source_name, destination_name):
        staged.rename(detached)
        staged.mkdir()
        (staged / "attacker").write_bytes(b"replacement")
        return original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(postseal, "_renameat_noreplace", swap_then_rename)
    try:
        with pytest.raises(
            postseal.PostsealPublicationUncertainError,
            match="committed root identity/tree is uncertain",
        ):
            postseal._publish_held_staging_root(held, frozen)  # noqa: SLF001
    finally:
        held.close()
    assert detached.is_dir()
    assert (detached / "payload").read_bytes() == b"authenticated"
    assert destination.is_dir()
    assert (destination / "attacker").read_bytes() == b"replacement"


@pytest.mark.parametrize(
    "with_slimbrain", [False, True], ids=["unavailable", "available"]
)
def test_complete_synthetic_run_publishes_and_reauthenticates_all_34(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_slimbrain: bool
) -> None:
    _install_test_native_contract(monkeypatch, (5, 6, 7, 8))
    coordinates = np.indices((5, 6, 7, 8), dtype=np.float32)
    real = (0.2 + 0.01 * coordinates.sum(axis=0)).astype(np.float32)
    predicted = (0.9 * real + 0.03).astype(np.float32)
    brain_mask = _interior_brain_mask((5, 6, 7))
    real *= brain_mask[..., None]
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[3:, :, :][brain_mask[3:, :, :].astype(bool)] = 2
    prediction_root, _ = _prediction_fixture(tmp_path)
    pins = _materialize_prediction_niftis(prediction_root, predicted)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    completed_sha = _materialize_stage_b_niftis(real, labels, brain_mask)
    monkeypatch.setattr(
        paired_quality,
        "evaluate_4d_pair_quality",
        lambda *_args, **_kwargs: copy.deepcopy(_quality_record()),
    )
    monkeypatch.setattr(visualizer, "generate_comparison_outputs", _fake_visual_outputs)
    texture_calls: list[str] = []
    boundary_events: list[str] = []

    def recording_texture_outputs(*args, **kwargs):
        texture_calls.append(str(kwargs["prefix"]))
        boundary_events.append(f"texture_audit:{kwargs['prefix']}")
        return _fake_texture_outputs(*args, **kwargs)

    monkeypatch.setattr(
        texture_audit, "generate_texture_audit", recording_texture_outputs
    )
    _install_test_execution_boundary(monkeypatch)
    output = tmp_path / "evaluation-output"
    distributional_arguments: dict[str, Any] = {}
    slimbrain_authority_sha256 = None
    if with_slimbrain:
        from tests.test_postseal_runtime import _slimbrain_authority_fixture

        (
            slimbrain_authority,
            slimbrain_authority_sha256,
            dependency_authority_sha256,
            python_executable_sha256,
        ) = _slimbrain_authority_fixture(tmp_path)
        distributional_arguments = {
            "slimbrain_authority": slimbrain_authority,
            "slimbrain_authority_sha256": slimbrain_authority_sha256,
            "slimbrain_dependency_authority_sha256": (dependency_authority_sha256),
            "slimbrain_python_executable_sha256": python_executable_sha256,
            "slimbrain_device": "cpu",
        }
        original_slimbrain_loader = postseal_metrics.load_authenticated_slimbrain

        def observed_slimbrain_loader(*args, **kwargs):
            boundary_events.append("slimbrain_loaded")
            return original_slimbrain_loader(*args, **kwargs)

        monkeypatch.setattr(
            postseal_metrics,
            "load_authenticated_slimbrain",
            observed_slimbrain_loader,
        )
    record = _run_postseal_evaluation_for_test(
        prediction_root=prediction_root,
        pins=pins,
        stage_b_completed_set=completed,
        stage_b_completed_set_sha256=completed_sha,
        stage_b_verifier=verify_postseal_target_completed_set,
        stage_b_verifier_authority=authority,
        output_root=output,
        access_observer=boundary_events.append,
        **distributional_arguments,
    )
    assert record["scan_count"] == 34
    assert record["quality_pass_count"] == 34
    assert record["quality_fail_count"] == 0
    assert record["texture_release_gate_pass_count"] == 34
    assert record["texture_release_gate_fail_count"] == 0
    assert record["all_34_texture_retention_and_anti_gaming_gates_passed"] is True
    assert texture_calls == list(postseal.EXPECTED_SCAN_IDS)
    first_texture_event = boundary_events.index(
        f"texture_audit:{postseal.EXPECTED_SCAN_IDS[0]}"
    )
    assert boundary_events.index("prediction_set_fully_authenticated") < (
        boundary_events.index("target_verifier_invoked")
    )
    assert all(
        boundary_events.index(f"target_bytes:{scan_id}") < first_texture_event
        for scan_id in postseal.EXPECTED_SCAN_IDS
    )
    if with_slimbrain:
        assert all(
            boundary_events.index(f"target_bytes:{scan_id}")
            < boundary_events.index("slimbrain_loaded")
            for scan_id in postseal.EXPECTED_SCAN_IDS
        )
    assert record["production_evaluator_authenticated_before_import"] is True
    assert record["synthesis_model_or_checkpoint_loaded"] is False
    assert record["evaluation_feature_model_loaded"] is with_slimbrain
    assert (
        record["distributional_metrics"]["evaluation_feature_model_loaded"]
        is with_slimbrain
    )
    assert record["distributional_metrics"]["available"] is with_slimbrain
    if with_slimbrain:
        assert record["distributional_metrics"]["sample_count"] == 34
        assert record["distributional_metrics"]["fid"] >= 0.0
        assert record["distributional_metrics"]["inception_score"] >= 1.0
        assert record["fitness_for_use"]["status"] == "FIT"
    else:
        assert record["fitness_for_use"]["status"] == "CONDITIONALLY_FIT"
    assert "model_or_checkpoint_loaded" not in record
    assert record["authorizes_training"] is False
    assert record["authorizes_checkpoint_selection"] is False
    assert record["can_change_checkpoint_or_prediction_set"] is False
    assert (output / "DATA_QUALITY_REPORT.md").is_file()
    assert (output / "cohort_quality_summary.png").is_file()
    assert len(list((output / "subjects").glob("*/evaluation.json"))) == 34
    first_subject_record = json.loads(
        (
            output / "subjects" / postseal.EXPECTED_SCAN_IDS[0] / "evaluation.json"
        ).read_text(encoding="utf-8")
    )
    texture_evidence = first_subject_record["texture_audit"]
    assert texture_evidence["verdict"] == "pass"
    assert texture_evidence["exit_code"] == 0
    assert texture_evidence["display_contract"]["image_interpolation"] == "nearest"
    assert texture_evidence["fmri_resampling_performed"] is False
    texture_manifest = json.loads(
        (output / texture_evidence["outputs"]["manifest"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert ".authenticated_inputs" not in json.dumps(texture_manifest)
    assert texture_manifest["inputs"]["real"] == first_subject_record["target"]
    assert texture_manifest["inputs"]["predicted"] == first_subject_record["prediction"]
    assert texture_manifest["inputs"]["mask"] == {
        key: first_subject_record["target_validity_mask"][key]
        for key in ("path", "sha256", "size_bytes")
    }
    assert (
        first_subject_record["target_validity_mask"][
            "equals_exact_final_nonzero_support"
        ]
        is True
    )
    assert first_subject_record["pair_contract"]["target_validity_mask"][
        "foreground_voxel_count"
    ] == int(brain_mask.sum())
    verified = _verify_publication(
        output, slimbrain_authority_sha256=slimbrain_authority_sha256
    )
    assert verified["record_sha256"] == record["record_sha256"]
    if with_slimbrain:
        with pytest.raises(
            postseal.PostsealEvaluationError, match="external authority pin"
        ):
            _verify_publication(output)
        with pytest.raises(
            postseal.PostsealEvaluationError, match="authority external pin"
        ):
            _verify_publication(output, slimbrain_authority_sha256="f" * 64)
    with pytest.raises(FileExistsError):
        test_postseal.run_postseal_evaluation(
            prediction_root=prediction_root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            output_root=output,
            execution_authority_path=tmp_path / "execution-authority.json",
            execution_authority_sha256="6" * 64,
        )
    if with_slimbrain:
        frozen_authorities = _publication_authority_arguments(output)
        output.chmod(0o700)
        for name in (
            "evaluation.json",
            "data_quality_summary.json",
            "publication_receipt.json",
        ):
            (output / name).chmod(0o600)
        evaluation_path = output / "evaluation.json"
        quality_path = output / "data_quality_summary.json"
        receipt_path = output / "publication_receipt.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        forged_distributional = copy.deepcopy(evaluation["distributional_metrics"])
        forged_distributional.pop("record_sha256")
        forged_distributional["fid"] = 0.0
        forged_distributional["inception_score"] = 999.0
        forged_distributional = _signed(forged_distributional)
        quality.pop("record_sha256")
        quality["distributional_metrics"] = copy.deepcopy(forged_distributional)
        _write_json(quality_path, _signed(quality))
        signed_quality = json.loads(quality_path.read_text(encoding="utf-8"))
        evaluation.pop("record_sha256")
        evaluation["distributional_metrics"] = forged_distributional
        evaluation["data_quality_summary"].update(
            {
                "sha256": _sha(quality_path),
                "size_bytes": quality_path.stat().st_size,
                "record_sha256": signed_quality["record_sha256"],
            }
        )
        inventory = postseal._relative_inventory(  # noqa: SLF001
            output, excluded={"evaluation.json", "publication_receipt.json"}
        )
        evaluation["output_inventory"] = inventory
        evaluation["output_inventory_sha256"] = postseal.canonical_sha256(inventory)
        _write_json(evaluation_path, _signed(evaluation))
        signed_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.pop("record_sha256")
        receipt["evaluation"].update(
            {
                "sha256": _sha(evaluation_path),
                "size_bytes": evaluation_path.stat().st_size,
                "record_sha256": signed_evaluation["record_sha256"],
            }
        )
        receipt["output_inventory_sha256"] = evaluation["output_inventory_sha256"]
        _write_json(receipt_path, _signed(receipt))
        with pytest.raises(
            postseal.PostsealEvaluationError,
            match="published (FID|Inception Score) differs",
        ):
            postseal.verify_evaluation_publication(
                output,
                expected_evaluation_sha256=_sha(evaluation_path),
                expected_receipt_sha256=_sha(receipt_path),
                **frozen_authorities,
                expected_slimbrain_authority_sha256=slimbrain_authority_sha256,
            )


@pytest.mark.parametrize("corruption", ("shape", "affine", "tr", "value"))
def test_final_pair_preflight_failure_precedes_all_models_metrics_and_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    _install_test_native_contract(monkeypatch, (5, 6, 7, 8))
    real = np.full((5, 6, 7, 8), 0.25, dtype=np.float32)
    brain_mask = _interior_brain_mask((5, 6, 7))
    real *= brain_mask[..., None]
    predicted = (real * 0.9 + 0.02 * brain_mask[..., None]).astype(np.float32)
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[3:, :, :][brain_mask[3:, :, :].astype(bool)] = 2
    prediction_root, _ = _prediction_fixture(tmp_path)
    pins = _materialize_prediction_niftis(prediction_root, predicted)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    completed_sha = _materialize_stage_b_niftis(real, labels, brain_mask)
    assert _FAKE_STAGE_B_RESULT is not None
    final_subject = _FAKE_STAGE_B_RESULT["subjects"][-1]
    final_target = Path(final_subject["native_bold"]["path"])
    corrupted = real.copy()
    affine = None
    tr = None
    if corruption == "shape":
        corrupted = corrupted[..., :-1]
    elif corruption == "affine":
        affine = np.diag([3.0, 3.0, 3.0, 1.0])
        affine[0, 3] = 3.0
    elif corruption == "tr":
        tr = 2.0
    else:
        corrupted[2, 2, 2, 0] = 1.25
    _write_nifti(final_target, corrupted, affine=affine, tr=tr)
    final_subject["native_bold"] = _artifact(final_target)
    unsigned_result = dict(_FAKE_STAGE_B_RESULT)
    unsigned_result.pop("record_sha256")
    _FAKE_STAGE_B_RESULT["record_sha256"] = postseal.canonical_sha256(unsigned_result)

    calls = {"extractor": 0, "quality": 0, "texture": 0, "visualization": 0}

    def forbidden_extractor(*_args, **_kwargs):
        calls["extractor"] += 1
        raise AssertionError("feature model loaded before complete pair preflight")

    def forbidden_quality(*_args, **_kwargs):
        calls["quality"] += 1
        raise AssertionError("quality fitting ran before complete pair preflight")

    def forbidden_texture(*_args, **_kwargs):
        calls["texture"] += 1
        raise AssertionError("texture audit ran before complete pair preflight")

    def forbidden_visualization(*_args, **_kwargs):
        calls["visualization"] += 1
        raise AssertionError("visualization ran before complete pair preflight")

    monkeypatch.setattr(
        postseal_metrics, "load_authenticated_slimbrain", forbidden_extractor
    )
    monkeypatch.setattr(paired_quality, "evaluate_4d_pair_quality", forbidden_quality)
    monkeypatch.setattr(texture_audit, "generate_texture_audit", forbidden_texture)
    monkeypatch.setattr(
        visualizer, "generate_comparison_outputs", forbidden_visualization
    )
    _install_test_execution_boundary(monkeypatch)
    output = tmp_path / f"final-{corruption}-failure"
    with pytest.raises(postseal.PostsealEvaluationError):
        _run_postseal_evaluation_for_test(
            prediction_root=prediction_root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=verify_postseal_target_completed_set,
            stage_b_verifier_authority=authority,
            output_root=output,
            slimbrain_authority=completed,
            slimbrain_authority_sha256="a" * 64,
            slimbrain_dependency_authority_sha256="b" * 64,
            slimbrain_python_executable_sha256="c" * 64,
            slimbrain_device="cpu",
        )
    assert calls == {
        "extractor": 0,
        "quality": 0,
        "texture": 0,
        "visualization": 0,
    }
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.staged-*"))


def test_smoothed_sealed_prediction_cannot_publish_a_fit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real texture math while keeping the integration fixture small."""
    _install_test_native_contract(monkeypatch, (15, 15, 15, 8))
    x, y, z, t = np.indices((15, 15, 15, 8), dtype=np.float32)
    checkerboard = ((x.astype(int) + y.astype(int) + z.astype(int)) % 2) * 2 - 1
    phase = 2.0 * np.pi * t / 8.0
    real = (
        0.50
        + 0.18 * checkerboard * (1.0 + 0.10 * np.sin(phase))
        + 0.03 * np.sin(0.45 * x + phase)
    ).astype(np.float32)
    predicted = np.stack(
        [
            ndimage.gaussian_filter(real[..., frame], sigma=1.0, mode="nearest")
            for frame in range(real.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)
    brain_mask = _interior_brain_mask(real.shape[:3], margin=2)
    real *= brain_mask[..., None]
    labels = np.zeros(real.shape[:3], dtype=np.int16)
    labels[4:6, 4:6, 4:6] = 1
    labels[9:11, 9:11, 9:11] = 2
    prediction_root, _ = _prediction_fixture(tmp_path)
    pins = _materialize_prediction_niftis(prediction_root, predicted)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    completed_sha = _materialize_stage_b_niftis(real, labels, brain_mask)
    monkeypatch.setattr(
        paired_quality,
        "evaluate_4d_pair_quality",
        lambda *_args, **_kwargs: copy.deepcopy(_quality_record()),
    )

    original_texture_audit = texture_audit.generate_texture_audit
    audited_scan_ids: list[str] = []

    def lightweight_real_texture_audit(
        real_path,
        predicted_path,
        mask_path,
        out_dir,
        *,
        roi_labels_path,
        prefix,
        enforce_texture_retention,
        enforce_texture_anti_gaming,
    ):
        audited_scan_ids.append(prefix)
        return original_texture_audit(
            real_path,
            predicted_path,
            mask_path,
            out_dir,
            roi_labels_path=roi_labels_path,
            prefix=prefix,
            sample_frame_count=8,
            detail_sigma_voxels=0.7,
            interior_erosion_voxels=2,
            patch_size_voxels=5,
            max_patches=8,
            spectrum_bins=6,
            high_frequency_cycles_per_voxel=0.30,
            enforce_texture_retention=enforce_texture_retention,
            enforce_texture_anti_gaming=enforce_texture_anti_gaming,
        )

    monkeypatch.setattr(
        texture_audit, "generate_texture_audit", lightweight_real_texture_audit
    )
    monkeypatch.setattr(
        visualizer,
        "generate_comparison_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed texture gate must stop before visualization")
        ),
    )
    _install_test_execution_boundary(monkeypatch)
    output = tmp_path / "smoothed-evaluation"
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="texture retention/anti-gaming verdict failed",
    ):
        _run_postseal_evaluation_for_test(
            prediction_root=prediction_root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=verify_postseal_target_completed_set,
            stage_b_verifier_authority=authority,
            output_root=output,
        )
    assert audited_scan_ids == [postseal.EXPECTED_SCAN_IDS[0]]
    assert not output.exists()
    assert not (output / "DATA_QUALITY_REPORT.md").exists()
    assert not (output / "data_quality_summary.json").exists()
    assert not list(tmp_path.glob(".smoothed-evaluation.staged-*"))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (("verdict", "fail"), ("exit_code", 2)),
)
def test_texture_verdict_and_exit_code_each_fail_publication_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
) -> None:
    _install_test_native_contract(monkeypatch, (5, 6, 7, 8))
    coordinates = np.indices((5, 6, 7, 8), dtype=np.float32)
    real = (0.2 + 0.01 * coordinates.sum(axis=0)).astype(np.float32)
    brain_mask = _interior_brain_mask((5, 6, 7))
    real *= brain_mask[..., None]
    labels = np.zeros((5, 6, 7), dtype=np.int16)
    labels[brain_mask.astype(bool)] = 1
    labels[3:, :, :][brain_mask[3:, :, :].astype(bool)] = 2
    prediction_root, _ = _prediction_fixture(tmp_path)
    pins = _materialize_prediction_niftis(prediction_root, real)
    predictions = test_postseal.authenticate_prediction_set(prediction_root, pins=pins)
    completed, completed_sha, authority = _stage_b_fixture(tmp_path, predictions, pins)
    completed_sha = _materialize_stage_b_niftis(real, labels, brain_mask)
    monkeypatch.setattr(
        paired_quality,
        "evaluate_4d_pair_quality",
        lambda *_args, **_kwargs: copy.deepcopy(_quality_record()),
    )

    def invalid_texture_output(*args, **kwargs):
        outputs = _fake_texture_outputs(*args, **kwargs)
        manifest_path = outputs["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("record_sha256")
        manifest["quality_gate"][field] = invalid_value
        _write_json(manifest_path, _signed(manifest))
        return outputs

    monkeypatch.setattr(texture_audit, "generate_texture_audit", invalid_texture_output)
    _install_test_execution_boundary(monkeypatch)
    output = tmp_path / f"invalid-texture-{field}"
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="texture aggregate verdict contradicts",
    ):
        _run_postseal_evaluation_for_test(
            prediction_root=prediction_root,
            pins=pins,
            stage_b_completed_set=completed,
            stage_b_completed_set_sha256=completed_sha,
            stage_b_verifier=verify_postseal_target_completed_set,
            stage_b_verifier_authority=authority,
            output_root=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.staged-*"))


@pytest.mark.parametrize(
    "mutation", ["chart", "subject", "data_quality", "receipt", "extra"]
)
def test_publication_verifier_recursively_rejects_output_or_receipt_tamper(
    tmp_path: Path, mutation: str
) -> None:
    root = _publication_fixture(tmp_path)
    verified = _verify_publication(root)
    assert verified["scan_count"] == 34
    if mutation == "chart":
        target = root / "cohort_quality_summary.png"
    elif mutation == "subject":
        target = root / "subjects" / postseal.EXPECTED_SCAN_IDS[0] / "evaluation.json"
    elif mutation == "data_quality":
        target = root / "data_quality_summary.json"
    elif mutation == "receipt":
        target = root / "publication_receipt.json"
    else:
        (root / "unbound-extra-file").write_bytes(b"extra")
        target = None
    if target is not None:
        with target.open("ab") as stream:
            stream.write(b"tamper")
    with pytest.raises(postseal.PostsealEvaluationError):
        _verify_publication(root)


@pytest.mark.parametrize(
    "authority_role",
    ("prediction", "completed", "source", "dependency_0", "dependency_1"),
)
@pytest.mark.parametrize(
    "attack", ("mutate", "delete", "replace", "hardlink", "symlink")
)
def test_publication_verifier_reopens_every_external_authority_fail_closed(
    tmp_path: Path, authority_role: str, attack: str
) -> None:
    root = _publication_fixture(tmp_path)
    frozen = _publication_authority_arguments(root)
    role_paths = {
        "prediction": Path(frozen["expected_prediction_set_seal_path"]),
        "completed": Path(frozen["expected_stage_b_completed_set_path"]),
        "source": Path(frozen["expected_stage_b_verifier_source_path"]),
        "dependency_0": Path(frozen["expected_stage_b_verifier_dependency_paths"][0]),
        "dependency_1": Path(frozen["expected_stage_b_verifier_dependency_paths"][1]),
    }
    target = role_paths[authority_role]
    if attack == "mutate":
        target.write_bytes(target.read_bytes() + b"\nexternal-authority-tamper\n")
    elif attack == "delete":
        target.unlink()
    elif attack == "replace":
        detached = target.with_name(f"{target.name}.detached")
        target.rename(detached)
        target.write_bytes(b"external-authority-replacement\n")
    elif attack == "hardlink":
        os.link(target, target.with_name(f"{target.name}.second-link"))
    else:
        detached = target.with_name(f"{target.name}.detached")
        target.rename(detached)
        target.symlink_to(detached)

    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(root / "evaluation.json"),
            expected_receipt_sha256=_sha(root / "publication_receipt.json"),
            **frozen,
        )


@pytest.mark.parametrize(
    "descriptor_attack", ("path", "size", "hash", "record", "extra", "reorder")
)
def test_coordinated_resigned_external_authority_descriptor_attack_fails_pins(
    tmp_path: Path, descriptor_attack: str
) -> None:
    root = _publication_fixture(tmp_path)
    frozen = _publication_authority_arguments(root)
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    if descriptor_attack == "path":
        source = Path(evaluation["stage_b_verifier_source"]["path"])
        substitute = source.with_name("same-bytes-substitute.py")
        substitute.write_bytes(source.read_bytes())
        evaluation["stage_b_verifier_source"]["path"] = str(substitute)
    elif descriptor_attack == "size":
        evaluation["stage_b_verifier_source"]["size_bytes"] += 1
    elif descriptor_attack == "hash":
        evaluation["stage_b_verifier_source"]["sha256"] = "0" * 64
    elif descriptor_attack == "record":
        evaluation["stage_b_completed_set"]["record_sha256"] = "0" * 64
    elif descriptor_attack == "extra":
        evaluation["stage_b_verifier_source"]["attacker_field"] = False
    else:
        evaluation["stage_b_verifier_dependencies"].reverse()
    _write_json(evaluation_path, _signed(evaluation))
    _resign_root_evaluation_and_receipt(root)

    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(evaluation_path),
            expected_receipt_sha256=_sha(root / "publication_receipt.json"),
            **frozen,
        )


@pytest.mark.parametrize(
    "semantic_attack", ("schema", "extra", "prediction", "checkpoint", "subject")
)
def test_attacker_pinned_resigned_stage_b_authority_still_fails_semantics(
    tmp_path: Path, semantic_attack: str
) -> None:
    root = _publication_fixture(tmp_path)
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    completed_path = Path(evaluation["stage_b_completed_set"]["path"])
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed.pop("record_sha256")
    if semantic_attack == "schema":
        completed["schema"] = "attacker-stage-b-completed-set"
    elif semantic_attack == "extra":
        completed["attacker_field"] = False
    elif semantic_attack == "prediction":
        completed["prediction_set_seal"]["sha256"] = "f" * 64
    elif semantic_attack == "checkpoint":
        completed["checkpoint_bindings"]["synthesis_checkpoint_sha256"] = "f" * 64
    else:
        completed["subjects"][0]["prediction_sha256"] = "f" * 64
    _write_json(completed_path, _signed(completed))
    completed_descriptor = _artifact(completed_path, signed=True)

    evaluation.pop("record_sha256")
    evaluation["stage_b_completed_set"] = completed_descriptor
    verification = evaluation["stage_b_verification"]
    verification.pop("record_sha256")
    verification["completed_set"] = completed_descriptor
    evaluation["stage_b_verification"] = _signed(verification)
    _write_json(evaluation_path, _signed(evaluation))
    _resign_root_evaluation_and_receipt(root)

    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(evaluation_path),
            expected_receipt_sha256=_sha(root / "publication_receipt.json"),
            **_publication_authority_arguments(root),
        )


def test_forged_raw_texture_values_cannot_pass_compaction_or_publication(
    tmp_path: Path,
) -> None:
    scan_id = postseal.EXPECTED_SCAN_IDS[0]
    forged = _synthetic_texture_evidence(scan_id, forged_values=True)
    with pytest.raises(
        postseal.PostsealEvaluationError, match="contradicts raw metric/policy"
    ):
        postseal._compact_texture_gate(scan_id, forged)  # noqa: SLF001

    root = _publication_fixture(tmp_path, forged_texture_values=True)
    with pytest.raises(
        postseal.PostsealEvaluationError, match="contradicts raw metric/policy"
    ):
        _verify_publication(root)


def test_resigned_authorization_flip_fails_external_publication_pins(
    tmp_path: Path,
) -> None:
    root = _publication_fixture(tmp_path)
    evaluation_path = root / "evaluation.json"
    receipt_path = root / "publication_receipt.json"
    evaluation_pin = _sha(evaluation_path)
    receipt_pin = _sha(receipt_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    evaluation["authorizes_training"] = True
    _write_json(evaluation_path, _signed(evaluation))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256")
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": json.loads(evaluation_path.read_text(encoding="utf-8"))[
                "record_sha256"
            ],
        }
    )
    _write_json(receipt_path, _signed(receipt))
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=evaluation_pin,
            expected_receipt_sha256=receipt_pin,
            **_publication_authority_arguments(root),
        )
    with pytest.raises(postseal.PostsealEvaluationError):
        _verify_publication(root)


@pytest.mark.parametrize(
    "field", ("synthesis_model_or_checkpoint_loaded", "evaluation_feature_model_loaded")
)
def test_resigned_model_loading_evidence_flip_fails_even_with_attacker_pins(
    tmp_path: Path, field: str
) -> None:
    root = _publication_fixture(tmp_path)
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    evaluation[field] = True
    _write_json(evaluation_path, _signed(evaluation))

    receipt_path = root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256")
    updated_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": updated_evaluation["record_sha256"],
        }
    )
    _write_json(receipt_path, _signed(receipt))
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(evaluation_path),
            expected_receipt_sha256=_sha(receipt_path),
            **_publication_authority_arguments(root),
        )


@pytest.mark.parametrize("attack", ["fabricated-loaded", "quality-cross-bind"])
def test_coordinated_resigned_distributional_metrics_cannot_bypass_false_only_gate(
    tmp_path: Path, attack: str
) -> None:
    root = _publication_fixture(tmp_path)
    required = postseal._distributional_metrics_unavailable()  # noqa: SLF001
    fabricated = (
        {
            "schema": required["schema"],
            "available": True,
            "evaluation_feature_model_loaded": True,
            "fid": 0.0,
            "inception_score": 999.0,
            "reason": "attacker fabricated a coordinated feature-model record",
        }
        if attack == "fabricated-loaded"
        else {**required, "reason": "attacker changed only data-quality evidence"}
    )
    quality_path = root / "data_quality_summary.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality.pop("record_sha256")
    quality["distributional_metrics"] = fabricated
    _write_json(quality_path, _signed(quality))

    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    if attack == "fabricated-loaded":
        evaluation["evaluation_feature_model_loaded"] = True
        evaluation["distributional_metrics"] = fabricated
    updated_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    evaluation["data_quality_summary"].update(
        {
            "sha256": _sha(quality_path),
            "size_bytes": quality_path.stat().st_size,
            "record_sha256": updated_quality["record_sha256"],
        }
    )
    inventory = postseal._relative_inventory(  # noqa: SLF001
        root, excluded={"evaluation.json", "publication_receipt.json"}
    )
    evaluation["output_inventory"] = inventory
    evaluation["output_inventory_sha256"] = postseal.canonical_sha256(inventory)
    _write_json(evaluation_path, _signed(evaluation))

    receipt_path = root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256")
    updated_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": updated_evaluation["record_sha256"],
        }
    )
    receipt["output_inventory_sha256"] = evaluation["output_inventory_sha256"]
    _write_json(receipt_path, _signed(receipt))
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(evaluation_path),
            expected_receipt_sha256=_sha(receipt_path),
            **_publication_authority_arguments(root),
        )


def test_resigned_unavailable_distributional_mode_cannot_claim_fit(
    tmp_path: Path,
) -> None:
    root = _publication_fixture(tmp_path)
    quality_path = root / "data_quality_summary.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality.pop("record_sha256")
    quality["fitness_for_use"]["status"] = "FIT"
    quality["fitness_for_use"]["severity"] = "none"
    quality["fitness_for_use"]["blocked_uses"].remove(
        "FID or Inception Score claims without pinned SLIM-Brain"
    )
    _write_json(quality_path, _signed(quality))

    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    updated_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    evaluation["fitness_for_use"] = updated_quality["fitness_for_use"]
    evaluation["data_quality_summary"].update(
        {
            "sha256": _sha(quality_path),
            "size_bytes": quality_path.stat().st_size,
            "record_sha256": updated_quality["record_sha256"],
        }
    )
    inventory = postseal._relative_inventory(  # noqa: SLF001
        root, excluded={"evaluation.json", "publication_receipt.json"}
    )
    evaluation["output_inventory"] = inventory
    evaluation["output_inventory_sha256"] = postseal.canonical_sha256(inventory)
    _write_json(evaluation_path, _signed(evaluation))

    receipt_path = root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256")
    updated_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": updated_evaluation["record_sha256"],
        }
    )
    receipt["output_inventory_sha256"] = evaluation["output_inventory_sha256"]
    _write_json(receipt_path, _signed(receipt))
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="FID/IS fitness mode",
    ):
        _verify_publication(root)


def test_resigned_subject_authorization_flip_fails_even_attacker_selected_pins(
    tmp_path: Path,
) -> None:
    root = _publication_fixture(tmp_path)
    scan_id = postseal.EXPECTED_SCAN_IDS[0]
    subject_path = root / "subjects" / scan_id / "evaluation.json"
    subject = json.loads(subject_path.read_text(encoding="utf-8"))
    subject.pop("record_sha256")
    subject["evaluation_can_authorize_training_or_selection"] = True
    _write_json(subject_path, _signed(subject))

    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("record_sha256")
    subject_binding = evaluation["subjects"][0]["record"]
    updated_subject = json.loads(subject_path.read_text(encoding="utf-8"))
    subject_binding.update(
        {
            "sha256": _sha(subject_path),
            "size_bytes": subject_path.stat().st_size,
            "record_sha256": updated_subject["record_sha256"],
        }
    )
    inventory = postseal._relative_inventory(  # noqa: SLF001
        root, excluded={"evaluation.json", "publication_receipt.json"}
    )
    evaluation["output_inventory"] = inventory
    evaluation["output_inventory_sha256"] = postseal.canonical_sha256(inventory)
    _write_json(evaluation_path, _signed(evaluation))

    receipt_path = root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("record_sha256")
    updated_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    receipt["evaluation"].update(
        {
            "sha256": _sha(evaluation_path),
            "size_bytes": evaluation_path.stat().st_size,
            "record_sha256": updated_evaluation["record_sha256"],
        }
    )
    receipt["output_inventory_sha256"] = evaluation["output_inventory_sha256"]
    _write_json(receipt_path, _signed(receipt))
    with pytest.raises(postseal.PostsealEvaluationError):
        postseal.verify_evaluation_publication(
            root,
            expected_evaluation_sha256=_sha(evaluation_path),
            expected_receipt_sha256=_sha(receipt_path),
            **_publication_authority_arguments(root),
        )
