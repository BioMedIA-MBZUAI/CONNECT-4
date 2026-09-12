from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from data.dataset_precomputed import Connect4PrecomputedDataset
from data.provenance import canonical_sha256, sha256_file
from preprocessing import native_target_identity as native_target_identity_module
from preprocessing.native_structural_identity import (
    build_native_structural_alignment_binding,
)
from preprocessing.native_target_identity import (
    CORRECTED_V3_CONTROLLER_SHA256,
    CORRECTED_V3_OUTPUT_AUTHORITY_RECORD_SHA256,
    CORRECTED_V3_OUTPUT_AUTHORITY_SHA256,
    CORRECTED_V3_STAGE_MANIFEST_SHA256,
    CORRECTED_V3_WORKLIST_RECORD_SHA256,
    CORRECTED_V3_WORKLIST_SHA256,
    NATIVE_SOURCE_SHA256,
    NATIVE_VERIFIER_SHA256,
    NativeTargetError,
    RUNTIME_ATTESTER_SHA256,
    TARGET_IDENTITY_SCHEMA,
    validate_native_padded_target,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

stage_b = importlib.import_module(
    "cluster_validation_20260829.native_stage_b_materializer_v1."
    "materialize_native_stage_b"
)
stage_b_tests = importlib.import_module(
    "cluster_validation_20260829.native_stage_b_materializer_v1."
    "test_materialize_native_stage_b"
)


SCAN_ID = stage_b_tests.SCAN_ID
ROLE = stage_b_tests.ROLE
SELECTION_MANIFEST = stage_b_tests.HALF_BUNDLE / "selection_manifest.json"
ROOT_REVIEW = stage_b_tests.ROOT_REVIEW
NATIVE_SOURCE = stage_b_tests.NATIVE_SOURCE


@pytest.fixture(autouse=True)
def _bind_local_corrected_v3_output_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        native_target_identity_module,
        "CORRECTED_V3_OUTPUT_ROOT",
        (tmp_path / "corrected_v3_half_native_output").resolve(),
    )


def _build_stage_b_target(tmp_path: Path) -> dict[str, object]:
    structural = stage_b_tests._structural_pipeline(tmp_path)
    authorities = stage_b_tests._target_authorities(tmp_path, structural)
    target_root = (tmp_path / "padded_targets").resolve()
    publication = stage_b.pad_target(
        SimpleNamespace(
            scan_id=SCAN_ID,
            role=ROLE,
            selection_manifest=str(SELECTION_MANIFEST.resolve()),
            selection_manifest_sha256=sha256_file(SELECTION_MANIFEST),
            root_review=str(ROOT_REVIEW.resolve()),
            root_review_sha256=sha256_file(ROOT_REVIEW),
            structural_batch=str(structural["batch"]),
            structural_batch_sha256=structural["batch_sha256"],
            completed_set=str(authorities["completed"]),
            completed_set_sha256=authorities["completed_sha256"],
            completed_set_commit_marker=str(authorities["completed_marker"]),
            completed_set_commit_marker_sha256=authorities[
                "completed_marker_sha256"
            ],
            reviewed_native_source=str(NATIVE_SOURCE.resolve()),
            output_root=str(target_root),
        )
    )
    _upgrade_target_fixture_to_corrected_v3(
        tmp_path=tmp_path,
        authorities=authorities,
        padded_publication=publication,
    )
    binding = build_native_structural_alignment_binding(
        Path(structural["batch"]),
        expected_batch_sha256=str(structural["batch_sha256"]),
        scan_id=SCAN_ID,
    )
    return {
        "structural": structural,
        "authorities": authorities,
        "target_root": target_root,
        "publication": publication,
        "binding": binding,
    }


def _validation_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    structural = fixture["structural"]
    authorities = fixture["authorities"]
    return {
        "scan_id": SCAN_ID,
        "expected_role": ROLE,
        "expected_structural_binding": fixture["binding"],
        "structural_batch_path": Path(structural["batch"]),
        "structural_batch_sha256": structural["batch_sha256"],
        "selection_manifest_path": SELECTION_MANIFEST.resolve(),
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "selection_root_review_path": ROOT_REVIEW.resolve(),
        "selection_root_review_sha256": sha256_file(ROOT_REVIEW),
        "completed_set_path": Path(authorities["completed"]),
        "completed_set_sha256": authorities["completed_sha256"],
        "completed_set_commit_marker_path": Path(authorities["completed_marker"]),
        "completed_set_commit_marker_sha256": authorities[
            "completed_marker_sha256"
        ],
        "reviewed_native_source_path": NATIVE_SOURCE.resolve(),
        "reviewed_native_source_sha256": NATIVE_SOURCE_SHA256,
        "runtime_attester_sha256": RUNTIME_ATTESTER_SHA256,
        "native_verifier_sha256": NATIVE_VERIFIER_SHA256,
    }


def _rewrite_signed(path: Path, value: dict) -> None:
    os.chmod(path, 0o600)
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    unsigned["record_sha256"] = canonical_sha256(unsigned)
    path.write_text(
        json.dumps(unsigned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(path, 0o400)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _rewrite_commit_marker(marker_path: Path, destination_path: Path) -> dict:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["sha256"] = sha256_file(destination_path)
    marker["size_bytes"] = destination_path.stat().st_size
    _rewrite_signed(marker_path, marker)
    return json.loads(marker_path.read_text(encoding="utf-8"))


def _write_signed_new(path: Path, value: dict) -> dict:
    path.write_text("{}\n", encoding="utf-8")
    _rewrite_signed(path, value)
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze_tree_and_build_marker(
    shard: Path, marker_path: Path, *, purpose: str
) -> dict:
    directories = []
    inventory = []
    for current_root, directory_names, file_names in os.walk(shard, topdown=True):
        current = Path(current_root)
        for name in sorted(directory_names):
            child = current / name
            directories.append(str(child.relative_to(shard)))
        for name in sorted(file_names):
            child = current / name
            os.chmod(child, 0o400)
            inventory.append(
                {
                    "relative_path": str(child.relative_to(shard)),
                    "sha256": sha256_file(child),
                    "size_bytes": child.stat().st_size,
                    "mode": 0o400,
                }
            )
    for relative in sorted(directories, key=lambda item: len(Path(item).parts), reverse=True):
        os.chmod(shard / relative, 0o500)
    os.chmod(shard, 0o500)
    inventory.sort(key=lambda item: item["relative_path"])
    directories.sort()
    metadata = shard.stat()
    return _write_signed_new(
        marker_path,
        {
            "schema": stage_b.NO_REPLACE_COMMIT_SCHEMA,
            "kind": "directory-tree",
            "status": "COMMITTED",
            "purpose": purpose,
            "destination": str(shard),
            "commit_marker": str(marker_path),
            "source_was_node_local": True,
            "root_mode": 0o500,
            "root_identity": {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            },
            "publication_protocol": {
                "descriptor_rooted": True,
                "directory_reservation": "mkdirat-exclusive",
                "leaf_publication": "openat-O_CREAT|O_EXCL",
                "tree_rehashed_frozen_and_fsynced_before_marker": True,
                "commit_marker_published_last": True,
                "directory_rename_noreplace_used": False,
                "overwrite_capable_rename_used": False,
                "failed_reservations_are_never_cleaned_or_reused": True,
            },
            "inventory": inventory,
            "inventory_sha256": canonical_sha256(inventory),
            "directories": directories,
            "directories_sha256": canonical_sha256(directories),
        },
    )


def _upgrade_target_fixture_to_corrected_v3(
    *, tmp_path: Path, authorities: dict[str, object], padded_publication: Path
) -> None:
    """Upgrade the older materializer fixture after it has built its target.

    The production materializer fixture intentionally models the predecessor
    receipt.  Main-loader tests need the exact corrected-v3 controller chain,
    so this helper moves the immutable artifacts into the v3 layout and
    re-signs every dependent test authority.
    """

    output_root = (tmp_path / "corrected_v3_half_native_output").resolve()
    for relative in (
        "shards",
        "shard_markers",
        "receipts/success",
        "receipts/success_markers",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    output_authority = {
        "schema": "connect4-half-native-output-authority-v1",
        "stage_manifest_sha256": CORRECTED_V3_STAGE_MANIFEST_SHA256,
        "worklist_sha256": CORRECTED_V3_WORKLIST_SHA256,
        "worklist_record_sha256": CORRECTED_V3_WORKLIST_RECORD_SHA256,
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "root_review_sha256": sha256_file(ROOT_REVIEW),
        "native_source_sha256": NATIVE_SOURCE_SHA256,
        "expected_task_count": 2121,
        "worker_ranges": [
            {
                "worker_index": 0,
                "start_inclusive": 0,
                "end_exclusive": 1061,
            },
            {
                "worker_index": 1,
                "start_inclusive": 1061,
                "end_exclusive": 2121,
            },
        ],
        "no_replacement": True,
        "no_refill": True,
        "paper_certified": False,
        "sealed_target_opened": False,
        "initialized_without_target_payload_access": True,
    }
    authority_path = output_root / "output_authority.json"
    authority_record = _write_signed_new(authority_path, output_authority)
    os.chmod(authority_path, 0o400)
    assert authority_record["record_sha256"] == (
        CORRECTED_V3_OUTPUT_AUTHORITY_RECORD_SHA256
    )
    assert sha256_file(authority_path) == CORRECTED_V3_OUTPUT_AUTHORITY_SHA256

    old_receipt_path = Path(authorities["receipt"])
    receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_shard = Path(receipt["publication"]["shard_path"])
    old_shard_marker = Path(receipt["publication"]["shard_commit_marker_path"])
    shard = output_root / "shards" / "task_0000"
    shard_marker_path = output_root / "shard_markers" / "task_0000.json"
    old_shard.rename(shard)
    old_shard_marker.unlink()
    scan_root = shard / SCAN_ID

    provenance_path = scan_root / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    outputs = {}
    for key, value in provenance["outputs"].items():
        output_path = scan_root / Path(value["path"]).name
        outputs[key] = _artifact(output_path)
    provenance["outputs"] = outputs
    _rewrite_signed(provenance_path, provenance)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    batch_path = shard / "native_preprocessing_batch_manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    scan = batch["scans"][0]
    for key, value in outputs.items():
        scan[f"{key}_path"] = value["path"]
        scan[f"{key}_sha256"] = value["sha256"]
        scan[f"{key}_size_bytes"] = value["size_bytes"]
    provenance_artifact = _artifact(provenance_path)
    scan["preprocessing_provenance_path"] = provenance_artifact["path"]
    scan["preprocessing_provenance_sha256"] = provenance_artifact["sha256"]
    scan["preprocessing_provenance_size_bytes"] = provenance_artifact["size_bytes"]
    _rewrite_signed(batch_path, batch)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))

    shard_marker = _freeze_tree_and_build_marker(
        shard,
        shard_marker_path,
        purpose="half-native-task_0000-train-ec46",
    )
    os.chmod(shard_marker_path, 0o400)
    publication = {
        "shard_path": str(shard),
        "shard_commit_marker_path": str(shard_marker_path),
        "shard_commit_marker_sha256": sha256_file(shard_marker_path),
        "shard_commit_marker_record_sha256": shard_marker["record_sha256"],
        "native_batch_manifest": {
            **_artifact(batch_path),
            "record_sha256": batch["record_sha256"],
        },
        "scan_provenance": {
            **provenance_artifact,
            "record_sha256": provenance["record_sha256"],
        },
        "preprocessed_fmri": outputs["preprocessed_fmri"],
        "t1_common": outputs["t1_common"],
        "mask_common": outputs["mask_common"],
        "full_native_verification": {
            "verified": True,
            "role": ROLE,
            "scan_count": 1,
            "patient_count": 1,
            "record_sha256": batch["record_sha256"],
            "manifest_sha256": sha256_file(batch_path),
            "scans": [
                {
                    "verified": True,
                    "scan_id": SCAN_ID,
                    "role": ROLE,
                    "record_sha256": provenance["record_sha256"],
                    "provenance_sha256": sha256_file(provenance_path),
                    "paper_certified": False,
                }
            ],
        },
    }

    receipt_selection = receipt["selection"]
    receipt_selection["worklist_sha256"] = CORRECTED_V3_WORKLIST_SHA256
    receipt_selection["worklist_record_sha256"] = (
        CORRECTED_V3_WORKLIST_RECORD_SHA256
    )
    receipt["execution"]["same_scan_attempt_count"] = 1
    receipt["execution"]["same_scan_transient_retry_limit"] = 1
    receipt["publication"] = publication
    receipt_path = output_root / "receipts" / "success" / "task_0000.json"
    receipt_marker_path = (
        output_root / "receipts" / "success_markers" / "task_0000.json"
    )
    old_receipt_path.rename(receipt_path)
    old_receipt_marker = Path(
        json.loads(Path(authorities["completed"]).read_text(encoding="utf-8"))[
            "success_receipts"
        ][0]["commit_marker_path"]
    )
    old_receipt_marker.rename(receipt_marker_path)
    _rewrite_signed(receipt_path, receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_marker = json.loads(receipt_marker_path.read_text(encoding="utf-8"))
    receipt_metadata = receipt_path.stat()
    receipt_marker.update(
        {
            "destination": str(receipt_path),
            "commit_marker": str(receipt_marker_path),
            "file_identity": {
                "device": receipt_metadata.st_dev,
                "inode": receipt_metadata.st_ino,
            },
            "sha256": sha256_file(receipt_path),
            "size_bytes": receipt_metadata.st_size,
        }
    )
    _rewrite_signed(receipt_marker_path, receipt_marker)
    receipt_marker = json.loads(receipt_marker_path.read_text(encoding="utf-8"))

    completed_path = Path(authorities["completed"])
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed["selection"]["worklist_sha256"] = CORRECTED_V3_WORKLIST_SHA256
    completed["selection"]["worklist_record_sha256"] = (
        CORRECTED_V3_WORKLIST_RECORD_SHA256
    )
    completed["success_receipts"][0].update(
        {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "record_sha256": receipt["record_sha256"],
            "commit_marker_path": str(receipt_marker_path),
            "commit_marker_sha256": sha256_file(receipt_marker_path),
            "commit_marker_record_sha256": receipt_marker["record_sha256"],
        }
    )
    _rewrite_signed(completed_path, completed)
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed_marker_path = Path(authorities["completed_marker"])
    completed_marker = _rewrite_commit_marker(completed_marker_path, completed_path)

    sidecar_path = padded_publication / f"{SCAN_ID}_fMRI.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller_authority"]["completed_set"] = {
        **_artifact(completed_path),
        "record_sha256": completed["record_sha256"],
    }
    sidecar["controller_authority"]["completed_set_commit_marker"] = {
        "path": str(completed_marker_path),
        "sha256": sha256_file(completed_marker_path),
        "record_sha256": completed_marker["record_sha256"],
    }
    sidecar["controller_authority"]["success_receipt"] = {
        **_artifact(receipt_path),
        "record_sha256": receipt["record_sha256"],
    }
    sidecar["controller_authority"]["success_receipt_commit_marker"] = {
        "path": str(receipt_marker_path),
        "sha256": sha256_file(receipt_marker_path),
        "record_sha256": receipt_marker["record_sha256"],
    }
    sidecar["native_preprocessing"].update(
        {
            "success_receipt_sha256": sha256_file(receipt_path),
            "success_receipt_record_sha256": receipt["record_sha256"],
            "scan_provenance_sha256": sha256_file(provenance_path),
            "scan_provenance_record_sha256": provenance["record_sha256"],
        }
    )
    _rewrite_signed(sidecar_path, sidecar)

    manifest_path = padded_publication / "padded_target_publication.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_set_sha256"] = sha256_file(completed_path)
    manifest["success_receipt_sha256"] = sha256_file(receipt_path)
    manifest["artifacts"]["bold_sidecar"] = _artifact(sidecar_path)
    _rewrite_signed(manifest_path, manifest)

    authorities.update(
        {
            "receipt": receipt_path,
            "completed_sha256": sha256_file(completed_path),
            "completed_marker_sha256": sha256_file(completed_marker_path),
            "v3_output_root": output_root,
        }
    )


def _resign_controller_chain(
    fixture: dict[str, object], receipt: dict
) -> None:
    authorities = fixture["authorities"]
    receipt_path = Path(authorities["receipt"])
    _rewrite_signed(receipt_path, receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    completed_path = Path(authorities["completed"])
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    entry = completed["success_receipts"][0]
    receipt_marker_path = Path(entry["commit_marker_path"])
    receipt_marker = _rewrite_commit_marker(receipt_marker_path, receipt_path)
    entry.update(
        {
            "sha256": sha256_file(receipt_path),
            "record_sha256": receipt["record_sha256"],
            "commit_marker_sha256": sha256_file(receipt_marker_path),
            "commit_marker_record_sha256": receipt_marker["record_sha256"],
        }
    )
    _rewrite_signed(completed_path, completed)
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed_marker_path = Path(authorities["completed_marker"])
    completed_marker = _rewrite_commit_marker(completed_marker_path, completed_path)

    publication = Path(fixture["publication"])
    sidecar_path = publication / f"{SCAN_ID}_fMRI.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller_authority"]["completed_set"] = {
        **_artifact(completed_path),
        "record_sha256": completed["record_sha256"],
    }
    sidecar["controller_authority"]["completed_set_commit_marker"].update(
        {
            "sha256": sha256_file(completed_marker_path),
            "record_sha256": completed_marker["record_sha256"],
        }
    )
    sidecar["controller_authority"]["success_receipt"] = {
        **_artifact(receipt_path),
        "record_sha256": receipt["record_sha256"],
    }
    sidecar["controller_authority"]["success_receipt_commit_marker"].update(
        {
            "sha256": sha256_file(receipt_marker_path),
            "record_sha256": receipt_marker["record_sha256"],
        }
    )
    sidecar["native_preprocessing"].update(
        {
            "success_receipt_sha256": sha256_file(receipt_path),
            "success_receipt_record_sha256": receipt["record_sha256"],
        }
    )
    _rewrite_signed(sidecar_path, sidecar)

    manifest_path = publication / "padded_target_publication.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_set_sha256"] = sha256_file(completed_path)
    manifest["success_receipt_sha256"] = sha256_file(receipt_path)
    manifest["artifacts"]["bold_sidecar"] = _artifact(sidecar_path)
    _rewrite_signed(manifest_path, manifest)
    authorities["completed_sha256"] = sha256_file(completed_path)
    authorities["completed_marker_sha256"] = sha256_file(completed_marker_path)


def _resign_padded_output(
    publication: Path,
    *,
    values: np.ndarray,
    affine: np.ndarray,
) -> None:
    bold_path = publication / f"{SCAN_ID}_fMRI.nii.gz"
    sidecar_path = publication / f"{SCAN_ID}_fMRI.json"
    manifest_path = publication / "padded_target_publication.json"
    image = nib.load(str(bold_path))
    os.chmod(bold_path, 0o600)
    nib.save(nib.Nifti1Image(values, affine, image.header), bold_path)
    os.chmod(bold_path, 0o400)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output_sha256"] = sha256_file(bold_path)
    _rewrite_signed(sidecar_path, sidecar)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["bold"] = _artifact(bold_path)
    manifest["artifacts"]["bold_sidecar"] = _artifact(sidecar_path)
    _rewrite_signed(manifest_path, manifest)


def test_main_loader_accepts_exact_materializer_target_and_binds_identity(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    identity = validate_native_padded_target(
        Path(fixture["publication"]), **_validation_kwargs(fixture)
    )
    assert identity["format"] == TARGET_IDENTITY_SCHEMA
    assert identity["scan_id"] == SCAN_ID
    assert identity["role"] == "train"
    assert identity["paper_certified"] is False
    assert identity["structural_alignment_authority_sha256"] == fixture[
        "structural"
    ]["batch_sha256"]
    assert identity["fingerprint_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in identity.items()
            if key != "fingerprint_sha256"
        }
    )
    receipt = json.loads(Path(fixture["authorities"]["receipt"]).read_text())
    assert set(receipt["publication"]["full_native_verification"]) == {
        "verified",
        "role",
        "scan_count",
        "patient_count",
        "record_sha256",
        "manifest_sha256",
        "scans",
    }
    assert receipt["execution"]["same_scan_attempt_count"] == 1
    assert receipt["execution"]["same_scan_transient_retry_limit"] == 1

    dataset = object.__new__(Connect4PrecomputedDataset)
    dataset.recovery_profile = True
    dataset.fmri_dir = Path(fixture["target_root"])
    dataset.target_scan_ids = frozenset({SCAN_ID})
    dataset.target_scan_roles = {SCAN_ID: "train"}
    dataset.native_alignment_authority_path = Path(
        fixture["structural"]["batch"]
    )
    dataset.native_alignment_authority_sha256 = fixture["structural"][
        "batch_sha256"
    ]
    dataset.native_target_authority = {
        "selection_manifest_path": SELECTION_MANIFEST.resolve(),
        "selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "selection_root_review_path": ROOT_REVIEW.resolve(),
        "selection_root_review_sha256": sha256_file(ROOT_REVIEW),
        "completed_set_path": Path(fixture["authorities"]["completed"]),
        "completed_set_sha256": fixture["authorities"]["completed_sha256"],
        "completed_set_commit_marker_path": Path(
            fixture["authorities"]["completed_marker"]
        ),
        "completed_set_commit_marker_sha256": fixture["authorities"][
            "completed_marker_sha256"
        ],
        "reviewed_source_path": NATIVE_SOURCE.resolve(),
        "reviewed_source_sha256": NATIVE_SOURCE_SHA256,
        "runtime_attester_sha256": RUNTIME_ATTESTER_SHA256,
        "native_verifier_sha256": NATIVE_VERIFIER_SHA256,
    }
    dataset.source_dataset = SimpleNamespace(
        source_fingerprint=lambda _scan_id: {
            "structural_source_identity": fixture["binding"]
        }
    )
    assert dataset._validate_preprocessed_target(SCAN_ID) == identity


def test_corrected_v3_stage_pin_binds_exact_controller_and_verifier():
    stage_manifest_path = WORKSPACE_ROOT / (
        "cluster_results_20260829/"
        "half_native_stage_v3_node_local_publisher_diagnostics/stage_manifest.json"
    )
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(stage_manifest)
    record_sha256 = unsigned.pop("record_sha256")
    inventory = {
        item["relative_path"]: item for item in stage_manifest["inventory"]
    }
    assert sha256_file(stage_manifest_path) == CORRECTED_V3_STAGE_MANIFEST_SHA256
    assert record_sha256 == canonical_sha256(unsigned)
    assert inventory["half_native_controller.py"]["sha256"] == (
        CORRECTED_V3_CONTROLLER_SHA256
    )
    assert inventory["native/verify_native_preprocessing.py"]["sha256"] == (
        NATIVE_VERIFIER_SHA256
    )
    assert inventory["native/native_preprocessing.py"]["sha256"] == (
        NATIVE_SOURCE_SHA256
    )
    assert inventory["worklist.json"]["sha256"] == CORRECTED_V3_WORKLIST_SHA256


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "success publication fields differ"),
        ("tampered", "full native verification differs"),
        ("extra", "full native verification differs"),
        ("historical-receipt", "semantic/GPU binding differs"),
    ],
)
def test_corrected_v3_verification_and_retry_fields_are_closed(
    tmp_path, case, message
):
    fixture = _build_stage_b_target(tmp_path)
    receipt_path = Path(fixture["authorities"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if case == "missing":
        receipt["publication"].pop("full_native_verification")
    elif case == "tampered":
        receipt["publication"]["full_native_verification"]["scans"][0][
            "provenance_sha256"
        ] = "0" * 64
    elif case == "extra":
        receipt["publication"]["full_native_verification"][
            "unreviewed_extension"
        ] = False
    else:
        receipt["execution"].pop("same_scan_attempt_count")
        receipt["execution"].pop("same_scan_transient_retry_limit")
    _resign_controller_chain(fixture, receipt)
    with pytest.raises(NativeTargetError, match=message):
        validate_native_padded_target(
            Path(fixture["publication"]), **_validation_kwargs(fixture)
        )


def test_historical_v2_stage_authority_is_rejected_even_when_resigned(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    authority_path = Path(fixture["authorities"]["v3_output_root"]) / (
        "output_authority.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["stage_manifest_sha256"] = (
        "13a927160e3fa1582e9cf42dd704b72dd3340f38319a0ef4c12a3ab68d0c9194"
    )
    _rewrite_signed(authority_path, authority)
    with pytest.raises(NativeTargetError, match="corrected-v3.*output authority"):
        validate_native_padded_target(
            Path(fixture["publication"]), **_validation_kwargs(fixture)
        )


def test_corrected_v3_shard_inventory_rejects_unrecorded_file(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    receipt = json.loads(Path(fixture["authorities"]["receipt"]).read_text())
    shard = Path(receipt["publication"]["shard_path"])
    os.chmod(shard, 0o700)
    extra = shard / "unrecorded.bin"
    extra.write_bytes(b"not in the immutable v3 shard")
    os.chmod(extra, 0o400)
    os.chmod(shard, 0o500)
    with pytest.raises(NativeTargetError, match="unrecorded file"):
        validate_native_padded_target(
            Path(fixture["publication"]), **_validation_kwargs(fixture)
        )


def test_resigned_sidecar_cannot_substitute_raw_structural_identity(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    publication = Path(fixture["publication"])
    sidecar_path = publication / f"{SCAN_ID}_fMRI.json"
    manifest_path = publication / "padded_target_publication.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["native_source_identity"]["raw_t1_sha256"] = "0" * 64
    _rewrite_signed(sidecar_path, sidecar)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["bold_sidecar"] = _artifact(sidecar_path)
    _rewrite_signed(manifest_path, manifest)
    with pytest.raises(NativeTargetError, match="raw-source identity differs"):
        validate_native_padded_target(publication, **_validation_kwargs(fixture))


def test_resigned_output_cannot_replace_receipt_bound_native_bold_crop(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    publication = Path(fixture["publication"])
    bold_path = publication / f"{SCAN_ID}_fMRI.nii.gz"
    image = nib.load(str(bold_path))
    values = np.asarray(image.dataobj, dtype=np.float32)
    values[1, 3, 1, 0] = min(1.0, float(values[1, 3, 1, 0]) + 0.125)
    _resign_padded_output(publication, values=values, affine=image.affine)
    with pytest.raises(NativeTargetError, match="receipt-bound native BOLD"):
        validate_native_padded_target(publication, **_validation_kwargs(fixture))


def test_resigned_output_cannot_hide_nonzero_architecture_padding(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    publication = Path(fixture["publication"])
    image = nib.load(str(publication / f"{SCAN_ID}_fMRI.nii.gz"))
    values = np.asarray(image.dataobj, dtype=np.float32)
    values[0, 0, 0, 0] = 0.5
    _resign_padded_output(publication, values=values, affine=image.affine)
    with pytest.raises(NativeTargetError, match="receipt-bound native BOLD"):
        validate_native_padded_target(publication, **_validation_kwargs(fixture))


def test_resigned_output_cannot_change_padded_affine(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    publication = Path(fixture["publication"])
    image = nib.load(str(publication / f"{SCAN_ID}_fMRI.nii.gz"))
    values = np.asarray(image.dataobj, dtype=np.float32)
    changed_affine = image.affine.copy()
    changed_affine[0, 3] += 3.0
    _resign_padded_output(
        publication, values=values, affine=changed_affine
    )
    with pytest.raises(NativeTargetError, match="affine differs"):
        validate_native_padded_target(publication, **_validation_kwargs(fixture))


def test_fully_resigned_completed_set_cannot_substitute_other_role_digest(tmp_path):
    fixture = _build_stage_b_target(tmp_path)
    authorities = fixture["authorities"]
    completed_path = Path(authorities["completed"])
    completed_marker_path = Path(authorities["completed_marker"])
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    entry = completed["success_receipts"][0]
    receipt_path = Path(entry["path"])
    receipt_marker_path = Path(entry["commit_marker_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    substituted_digest = "0" * 64
    receipt["selection"]["development_validation_scans_sha256"] = (
        substituted_digest
    )
    _rewrite_signed(receipt_path, receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_marker = _rewrite_commit_marker(receipt_marker_path, receipt_path)
    entry.update(
        {
            "sha256": sha256_file(receipt_path),
            "record_sha256": receipt["record_sha256"],
            "commit_marker_sha256": sha256_file(receipt_marker_path),
            "commit_marker_record_sha256": receipt_marker["record_sha256"],
        }
    )
    completed["selection"]["development_validation_scans_sha256"] = (
        substituted_digest
    )
    _rewrite_signed(completed_path, completed)
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed_marker = _rewrite_commit_marker(
        completed_marker_path, completed_path
    )

    publication = Path(fixture["publication"])
    sidecar_path = publication / f"{SCAN_ID}_fMRI.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller_authority"]["completed_set"].update(
        {
            "sha256": sha256_file(completed_path),
            "size_bytes": completed_path.stat().st_size,
            "record_sha256": completed["record_sha256"],
        }
    )
    sidecar["controller_authority"]["completed_set_commit_marker"].update(
        {
            "sha256": sha256_file(completed_marker_path),
            "record_sha256": completed_marker["record_sha256"],
        }
    )
    sidecar["controller_authority"]["success_receipt"].update(
        {
            "sha256": sha256_file(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "record_sha256": receipt["record_sha256"],
        }
    )
    sidecar["controller_authority"]["success_receipt_commit_marker"].update(
        {
            "sha256": sha256_file(receipt_marker_path),
            "record_sha256": receipt_marker["record_sha256"],
        }
    )
    sidecar["native_preprocessing"].update(
        {
            "success_receipt_sha256": sha256_file(receipt_path),
            "success_receipt_record_sha256": receipt["record_sha256"],
        }
    )
    _rewrite_signed(sidecar_path, sidecar)
    manifest_path = publication / "padded_target_publication.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_set_sha256"] = sha256_file(completed_path)
    manifest["success_receipt_sha256"] = sha256_file(receipt_path)
    manifest["artifacts"]["bold_sidecar"] = _artifact(sidecar_path)
    _rewrite_signed(manifest_path, manifest)

    kwargs = _validation_kwargs(fixture)
    kwargs["completed_set_sha256"] = sha256_file(completed_path)
    kwargs["completed_set_commit_marker_sha256"] = sha256_file(
        completed_marker_path
    )
    with pytest.raises(NativeTargetError, match="completed-set semantic"):
        validate_native_padded_target(publication, **kwargs)


def test_sealed_role_fails_before_any_target_path_is_resolved(tmp_path, monkeypatch):
    touched = False

    def forbidden(*_args, **_kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("target path was resolved")

    monkeypatch.setattr(
        "preprocessing.native_target_identity._canonical_existing_path", forbidden
    )
    with pytest.raises(NativeTargetError, match="train/development only"):
        validate_native_padded_target(
            tmp_path / "sealed-target",
            scan_id=SCAN_ID,
            expected_role="sealed-test",
            expected_structural_binding={},
            structural_batch_path=tmp_path / "structural.json",
            structural_batch_sha256="0" * 64,
            selection_manifest_path=tmp_path / "selection.json",
            selection_manifest_sha256="0" * 64,
            selection_root_review_path=tmp_path / "review.json",
            selection_root_review_sha256="0" * 64,
            completed_set_path=tmp_path / "completed.json",
            completed_set_sha256="0" * 64,
            completed_set_commit_marker_path=tmp_path / "completed.commit.json",
            completed_set_commit_marker_sha256="0" * 64,
            reviewed_native_source_path=tmp_path / "native.py",
            reviewed_native_source_sha256=NATIVE_SOURCE_SHA256,
            runtime_attester_sha256=RUNTIME_ATTESTER_SHA256,
            native_verifier_sha256=NATIVE_VERIFIER_SHA256,
        )
    assert touched is False
