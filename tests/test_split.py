import copy
import json
from pathlib import Path

import pytest

from data.provenance import canonical_sha256, sha256_file
from data.split import (
    PAPER_PROFILE,
    RECOVERY_PROFILE,
    build_recovery_split_manifest,
    build_seeded_split_manifest,
    load_immutable_split_indices,
    patient_id,
    patient_level_split,
    patient_level_train_val_test_split,
    validate_immutable_split_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_MANIFEST = (
    REPOSITORY_ROOT
    / "cluster_results_20260829"
    / "immutable_recovery_split_authority_v1"
    / "connect4_half_recovery_split.json"
)


def _patients(scan_ids, indices):
    return {patient_id(scan_ids[index]) for index in indices}


def _paper_authority(tmp_path, *, patients=20):
    scans = [
        f"P{patient:02d}_{session:03d}"
        for patient in range(patients)
        for session in (1, 2)
    ]
    patient_map = {scan: patient_id(scan) for scan in scans}
    path = tmp_path / "paper-split.json"
    build_seeded_split_manifest(
        scans,
        patient_map,
        output_path=path,
        cohort_manifest_sha256="a" * 64,
        val_frac=0.2,
        test_frac=0.2,
        seed=7,
    )
    return scans, patient_map, path, sha256_file(path)


def _resign(payload):
    unsigned = copy.deepcopy(payload)
    unsigned.pop("record_sha256", None)
    return {**unsigned, "record_sha256": canonical_sha256(unsigned)}


def _write_payload(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _recovery_payload():
    return json.loads(RECOVERY_MANIFEST.read_text(encoding="utf-8"))


def _recompute_derived(payload):
    payload["partition_counts"] = {
        role: {
            "patients": len({record["patient_id"] for record in records}),
            "scans": len(records),
        }
        for role, records in payload["partitions"].items()
    }
    payload["partition_record_sha256"] = {
        role: canonical_sha256(records)
        for role, records in payload["partitions"].items()
    }
    payload["total_patient_count"] = len(
        {
            record["patient_id"]
            for records in payload["partitions"].values()
            for record in records
        }
    )
    payload["total_scan_count"] = sum(
        len(records) for records in payload["partitions"].values()
    )
    return _resign(payload)


def test_explicit_paper_split_is_patient_disjoint_and_read_only(tmp_path):
    scans, patient_map, manifest, digest = _paper_authority(tmp_path)
    before = manifest.read_bytes()
    train, validation, test = load_immutable_split_indices(
        scans,
        patient_map,
        manifest_path=manifest,
        manifest_sha256=digest,
        protocol_profile=PAPER_PROFILE,
    )
    train_patients = _patients(scans, train)
    validation_patients = _patients(scans, validation)
    test_patients = _patients(scans, test)
    assert train_patients.isdisjoint(validation_patients | test_patients)
    assert validation_patients.isdisjoint(test_patients)
    assert set(train + validation + test) == set(range(len(scans)))
    assert len(validation_patients) == len(test_patients) == 4
    assert manifest.read_bytes() == before


def test_runtime_missing_manifest_fails_without_writing(tmp_path):
    path = tmp_path / "missing" / "split.json"
    with pytest.raises(FileNotFoundError, match="runtime generation is forbidden"):
        patient_level_train_val_test_split(
            [f"P{index:02d}_001" for index in range(10)],
            manifest_path=str(path),
            manifest_sha256="a" * 64,
        )
    assert not path.exists()
    assert not path.parent.exists()


def test_runtime_rejects_tampered_file_before_json_use(tmp_path):
    scans, patient_map, manifest, digest = _paper_authority(tmp_path)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_immutable_split_indices(
            scans,
            patient_map,
            manifest_path=manifest,
            manifest_sha256=digest,
            protocol_profile=PAPER_PROFILE,
        )


def test_recovery_manifest_is_exact_authorized_half_cohort():
    payload = validate_immutable_split_manifest(
        _recovery_payload(), protocol_profile=RECOVERY_PROFILE
    )
    assert payload["partition_counts"] == {
        "train": {"patients": 533, "scans": 2078},
        "development_validation": {"patients": 13, "scans": 43},
        "sealed_test": {"patients": 13, "scans": 34},
    }
    assert payload["total_patient_count"] == 559
    assert payload["total_scan_count"] == 2155
    assert "fmri_filename" not in RECOVERY_MANIFEST.read_text(encoding="utf-8")

    records = [
        record
        for role in ("sealed_test", "train", "development_validation")
        for record in reversed(payload["partitions"][role])
    ]
    scans = [record["scan_id"] for record in records]
    patients = {record["scan_id"]: record["patient_id"] for record in records}
    train, development, sealed = load_immutable_split_indices(
        scans,
        patients,
        manifest_path=RECOVERY_MANIFEST,
        manifest_sha256=sha256_file(RECOVERY_MANIFEST),
        protocol_profile=RECOVERY_PROFILE,
    )
    assert (len(train), len(development), len(sealed)) == (2078, 43, 34)


def test_recovery_builder_rejects_tampered_root_review_without_output(tmp_path):
    source_review = (
        REPOSITORY_ROOT
        / "cluster_results_20260829"
        / "half_training_patient_selection_v1_root_review.json"
    )
    tampered_review = tmp_path / "root-review.json"
    tampered_review.write_bytes(source_review.read_bytes() + b"\n")
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="root-review SHA-256 differs"):
        build_recovery_split_manifest(
            selection_bundle=(
                REPOSITORY_ROOT
                / "cluster_results_20260829"
                / "half_training_patient_selection_v1_published"
            ),
            selection_root_review=tampered_review,
            output_path=output,
        )
    assert not output.exists()


def test_recovery_rejects_resigned_reordered_records():
    payload = _recovery_payload()
    payload["partitions"]["train"][:2] = reversed(
        payload["partitions"]["train"][:2]
    )
    payload = _recompute_derived(payload)
    with pytest.raises(ValueError, match="not canonical"):
        validate_immutable_split_manifest(payload, protocol_profile=RECOVERY_PROFILE)


def test_recovery_rejects_resigned_role_crossing():
    payload = _recovery_payload()
    train = payload["partitions"]["train"]
    development = payload["partitions"]["development_validation"]
    train[0], development[0] = development[0], train[0]
    train.sort(key=lambda value: (value["patient_id"], value["scan_id"]))
    development.sort(key=lambda value: (value["patient_id"], value["scan_id"]))
    payload = _recompute_derived(payload)
    with pytest.raises(ValueError, match="multiple split partitions"):
        validate_immutable_split_manifest(payload, protocol_profile=RECOVERY_PROFILE)


def test_recovery_rejects_resigned_patient_overlap():
    payload = _recovery_payload()
    payload["partitions"]["development_validation"][0]["patient_id"] = payload[
        "partitions"
    ]["train"][0]["patient_id"]
    payload["partitions"]["development_validation"].sort(
        key=lambda value: (value["patient_id"], value["scan_id"])
    )
    payload = _recompute_derived(payload)
    with pytest.raises(ValueError, match="multiple split partitions"):
        validate_immutable_split_manifest(payload, protocol_profile=RECOVERY_PROFILE)


def test_recovery_rejects_resigned_count_drift():
    payload = _recovery_payload()
    payload["partitions"]["train"].pop()
    payload = _recompute_derived(payload)
    with pytest.raises(ValueError, match="exact authorized half cohort"):
        validate_immutable_split_manifest(payload, protocol_profile=RECOVERY_PROFILE)


def test_runtime_rejects_patient_identity_drift(tmp_path):
    scans, patient_map, manifest, digest = _paper_authority(tmp_path)
    patient_map[scans[0]] = "WRONG"
    with pytest.raises(ValueError, match="patient identity differs"):
        load_immutable_split_indices(
            scans,
            patient_map,
            manifest_path=manifest,
            manifest_sha256=digest,
            protocol_profile=PAPER_PROFILE,
        )


def test_offline_builder_is_no_overwrite(tmp_path):
    scans, patient_map, manifest, _ = _paper_authority(tmp_path)
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        build_seeded_split_manifest(
            scans,
            patient_map,
            output_path=manifest,
            cohort_manifest_sha256="a" * 64,
        )
    assert manifest.read_bytes() == original


def test_two_way_helper_remains_patient_disjoint_and_does_not_write():
    scans = ["A_1", "A_2", "B_1", "C_1", "D_1"]
    train, validation = patient_level_split(scans, val_frac=0.25, seed=3)
    assert _patients(scans, train).isdisjoint(_patients(scans, validation))
