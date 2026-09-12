import csv
import json
from copy import deepcopy
from functools import lru_cache
from types import SimpleNamespace

import pytest

from data.dataset import Connect4Dataset
from data.protocol import (
    CONDITIONING_ARTIFACT_SCHEMA,
    EXTERNAL_CACHE_PROTOCOL_SCHEMA,
    PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA,
    RUN_ARTIFACT_IDENTITY_SCHEMA,
    TARGET_VALIDITY_MASK_CONTRACT,
    build_external_cache_protocol_context,
    build_run_artifact_identity,
    build_split_identity,
    conditioning_identity_from_cache_sources,
    patient_level_split_from_manifest,
    validate_synthesis_split_identity,
    validate_external_scaler_provenance,
    validate_fixed_protocol_config,
    validate_training_cohort_manifest,
    validate_training_scaler_provenance,
    validate_unseen_cohort_manifest,
)
from data.provenance import canonical_sha256, directory_file_sha256, sha256_file
from data.split import PAPER_PROFILE, build_seeded_split_manifest
from data.patch_descriptions import PATCH_DESCRIPTION_SCHEMA_VERSION
from data.spatial_contract import PATCH_COORDINATE_CONTRACT
from utils.config import A4_RECOVERY_PROTOCOL_PROFILE


def _conditioning_identity():
    modernbert = {
        "implementation": "Clinical-ModernBERT",
        "upstream_model_id": "Simonlee711/Clinical_ModernBERT",
        "model_name": "/models/modernbert",
        "revision": "a" * 40,
        "local_files_sha256": {"weights.bin": "2" * 64},
    }
    modernbert["fingerprint_sha256"] = canonical_sha256(modernbert)
    brainiac_sources = {
        "connect4/models/brainiac_wrapper.py": "6" * 64,
        "brainiac/load_brainiac.py": "7" * 64,
    }
    roi_features = {
        "format": Connect4Dataset.ROI_FEATURE_CONDITIONING_SCHEMA,
        "roi_specs": [
            {"index": index, "slug": f"roi_{index}", "label_id": index + 1}
            for index in range(32)
        ],
        "anatcl": {
            "schema": Connect4Dataset.ANATCL_PROVENANCE_SCHEMA,
            "model": dict(Connect4Dataset.ANATCL_MODEL_CONTRACT),
            "extraction": dict(Connect4Dataset.ANATCL_EXTRACTION_CONTRACT),
            "cat12_input_protocol": {
                "schema": Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA,
                "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
            },
            "embedding_dim": 512,
        },
        "pyradiomics": {
            "schema": Connect4Dataset.RADIOMICS_PROVENANCE_SCHEMA,
            "extractor": {
                **Connect4Dataset.PYRADIOMICS_EXTRACTOR_CONTRACT,
                "runtime_lock_sha256": "c" * 64,
                "runtime_tree_sha256": "d" * 64,
            },
            "feature_columns": ["radiomics_a"],
            "feature_dim": 1,
        },
        "concatenation": {
            "order": ["anatcl", "pyradiomics"],
            "output_dim": 513,
        },
        "extraction_implementation_sha256": "9" * 64,
    }
    roi_features["fingerprint_sha256"] = canonical_sha256(roi_features)
    return {
        "format": CONDITIONING_ARTIFACT_SCHEMA,
        "brainiac": {
            "implementation": "BrainIAC",
            "checkpoint_sha256": "1" * 64,
            "source_files_sha256": brainiac_sources,
            "source_files_fingerprint_sha256": canonical_sha256(brainiac_sources),
            "adapter": {
                "input_channels": 1,
                "resize_shape": [96, 96, 96],
                "resize_mode": "trilinear",
                "align_corners": False,
                "output": "CLS token embedding",
                "embedding_dim": 768,
            },
        },
        "modernbert": modernbert,
        "dwi_sha256": "3" * 64,
        "normative_workbook_sha256": "4" * 64,
        "roi_feature_extractor": roi_features,
        "patch_description_schema_version": PATCH_DESCRIPTION_SCHEMA_VERSION,
        "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
        "functional_connectivity_sha256": "5" * 64,
        "target_shape": [64, 80, 64],
        "patch_size": [16, 16, 16],
    }


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scan_id", "patient_id", "cohort"])
        writer.writeheader()
        writer.writerows(rows)


def _paper_rows():
    return [
        {
            "scan_id": f"A_{index:05d}",
            "patient_id": f"A-P{index:05d}",
            "cohort": "A4",
        }
        for index in range(6290)
    ] + [
        {
            "scan_id": f"D_{index:05d}",
            "patient_id": f"D-P{index:05d}",
            "cohort": "ADNI",
        }
        for index in range(700)
    ]


@lru_cache(maxsize=1)
def _base_paper_split_identity():
    rows = _paper_rows()
    scans = [row["scan_id"] for row in rows]
    cohort_by_scan = {row["scan_id"]: row["cohort"] for row in rows}
    patient_by_scan = {row["scan_id"]: row["patient_id"] for row in rows}
    return build_split_identity(
        scans,
        list(range(6000)),
        list(range(6000, 6590)),
        list(range(6590, len(scans))),
        cohort_by_scan,
        patient_by_scan,
    )


def test_training_cohort_manifest_is_complete_and_exact(tmp_path):
    rows = _paper_rows()
    scans = [row["scan_id"] for row in rows]
    path = tmp_path / "cohorts.csv"
    _write_manifest(path, rows)
    mapping = validate_training_cohort_manifest(str(path), scans)
    assert len(mapping.cohort_by_scan) == 6990
    assert mapping.cohort_by_scan["A_00000"] == "A4"
    assert mapping.patient_by_scan["D_00699"] == "D-P00699"
    with pytest.raises(ValueError, match="cannot be overridden or disabled"):
        validate_training_cohort_manifest(
            str(path), scans, expected_scan_counts={"A4": 1, "ADNI": 1}
        )


def test_a4_recovery_manifest_uses_exact_configured_count_without_paper_claim(tmp_path):
    rows = [
        {"scan_id": "A_001", "patient_id": "P1", "cohort": "A4"},
        {"scan_id": "A_002", "patient_id": "P2", "cohort": "A4"},
    ]
    path = tmp_path / "a4_recovery.csv"
    _write_manifest(path, rows)
    evidence = validate_training_cohort_manifest(
        str(path),
        ["A_001", "A_002"],
        expected_scan_counts={"A4": 2},
        protocol_profile=A4_RECOVERY_PROTOCOL_PROFILE,
    )
    assert set(evidence.cohort_by_scan.values()) == {"A4"}

    with pytest.raises(ValueError, match="authenticated profile counts"):
        validate_training_cohort_manifest(
            str(path),
            ["A_001", "A_002"],
            expected_scan_counts={"A4": 3},
            protocol_profile=A4_RECOVERY_PROTOCOL_PROFILE,
        )
    with pytest.raises(ValueError, match="A4 only"):
        validate_training_cohort_manifest(
            str(path),
            ["A_001", "A_002"],
            expected_scan_counts={"A4": 2, "ADNI": 1},
            protocol_profile=A4_RECOVERY_PROTOCOL_PROFILE,
        )


def test_training_cohort_manifest_rejects_missing_extra_or_wrong_cohort(tmp_path):
    scans = ["A_001", "D_001"]
    path = tmp_path / "cohorts.csv"
    _write_manifest(
        path,
        [
            {"scan_id": "A_001", "patient_id": "A", "cohort": "A4"},
            {"scan_id": "OTHER", "patient_id": "O", "cohort": "OASIS-3"},
        ],
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_training_cohort_manifest(str(path), scans)

    _write_manifest(
        path,
        [
            {"scan_id": "A_001", "patient_id": "A", "cohort": "A4"},
            {"scan_id": "D_001", "patient_id": "D", "cohort": "OASIS-3"},
        ],
    )
    with pytest.raises(ValueError, match="requires exactly cohorts"):
        validate_training_cohort_manifest(str(path), scans)


def test_training_cohort_manifest_rejects_duplicate_scan(tmp_path):
    path = tmp_path / "cohorts.csv"
    _write_manifest(
        path,
        [
            {"scan_id": "A_001", "patient_id": "A", "cohort": "A4"},
            {"scan_id": "A_001", "patient_id": "A", "cohort": "ADNI"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate scan_id"):
        validate_training_cohort_manifest(str(path), ["A_001"])


def test_versioned_json_manifest_and_split_identity(tmp_path):
    rows = _paper_rows()
    path = tmp_path / "cohorts.json"
    path.write_text(
        json.dumps(
            {
                "format": "connect4_scan_cohorts_v1",
                "scans": {
                    row["scan_id"]: {
                        "patient_id": row["patient_id"],
                        "cohort": row["cohort"],
                    }
                    for row in rows
                },
            }
        )
    )
    scans = [row["scan_id"] for row in rows]
    validate_training_cohort_manifest(str(path), scans)
    identity = _paper_split_identity()
    assert identity["format"] == "connect4_split_identity_v2"
    assert len(identity["sha256"]) == 64
    assert identity["partition_counts"] == {
        "train": 6000,
        "validation": 590,
        "test": 400,
    }
    exclusion = validate_synthesis_split_identity(identity)
    assert len(exclusion.patient_ids) == 6590
    assert exclusion.cohorts == frozenset({"A4", "ADNI"})


def _paper_split_identity():
    return deepcopy(_base_paper_split_identity())


def test_external_manifest_requires_exact_coverage_and_unseen_identity(tmp_path):
    path = tmp_path / "external.csv"
    rows = [
        {"scan_id": "E_001", "patient_id": "EXT-1", "cohort": "ideas"},
        {"scan_id": "E_002", "patient_id": "EXT-2", "cohort": "OASIS-3"},
    ]
    _write_manifest(path, rows)
    evidence = validate_unseen_cohort_manifest(
        str(path), ["E_001", "E_002"], _paper_split_identity()
    )
    assert evidence.cohort_by_scan == {"E_001": "IDEAS", "E_002": "OASIS-3"}

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_unseen_cohort_manifest(str(path), ["E_001"], _paper_split_identity())


@pytest.mark.parametrize(
    "patient_id,cohort,message",
    [
        ("A-P00000", "IDEAS", "patients used by synthesis"),
        ("EXT-1", "ADNI", "synthesis-cohort identities"),
    ],
)
def test_external_manifest_rejects_checkpoint_patient_or_cohort(
    tmp_path, patient_id, cohort, message
):
    path = tmp_path / "external.csv"
    _write_manifest(
        path,
        [{"scan_id": "E_001", "patient_id": patient_id, "cohort": cohort}],
    )
    with pytest.raises(ValueError, match=message):
        validate_unseen_cohort_manifest(str(path), ["E_001"], _paper_split_identity())


def test_synthesis_exclusion_identity_rejects_tampering():
    identity = _paper_split_identity()
    identity["synthesis_patient_ids"] = ["FABRICATED"]
    with pytest.raises(
        ValueError, match="derived from its partitions|digest is invalid"
    ):
        validate_synthesis_split_identity(identity)


def test_external_cache_context_binds_manifest_checkpoint_and_exclusion(tmp_path):
    manifest = tmp_path / "external.csv"
    _write_manifest(
        manifest,
        [{"scan_id": "E_001", "patient_id": "EXT-1", "cohort": "IDEAS"}],
    )
    conditioning = _conditioning_identity()
    evidence, context = build_external_cache_protocol_context(
        manifest_path=str(manifest),
        scan_ids=["E_001"],
        synthesis_split_identity=_paper_split_identity(),
        synthesis_artifact_identity={
            "conditioning_identity": conditioning,
            "conditioning_identity_sha256": canonical_sha256(conditioning),
            "common_grid_contract_sha256": "f" * 64,
        },
        synthesis_checkpoint_sha256="e" * 64,
    )
    assert evidence.patient_by_scan == {"E_001": "EXT-1"}
    assert context["format"] == EXTERNAL_CACHE_PROTOCOL_SCHEMA
    assert context["synthesis_common_grid_contract_sha256"] == "f" * 64
    assert context["external_cohorts"] == ["IDEAS"]
    assert context["synthesis_checkpoint_sha256"] == "e" * 64


def test_conditioning_identity_binds_brainiac_source_and_roi_pipeline():
    conditioning = _conditioning_identity()
    sources = {
        "dataset": {
            "dwi": conditioning["dwi_sha256"],
            "normative_workbook": conditioning["normative_workbook_sha256"],
            "roi_feature_extractor_identity": conditioning["roi_feature_extractor"],
        },
        "brainiac": conditioning["brainiac"],
        "modernbert": conditioning["modernbert"],
        "target_shape": conditioning["target_shape"],
        "patch_size": conditioning["patch_size"],
        "patch_description_schema_version": conditioning[
            "patch_description_schema_version"
        ],
        "patch_coordinate_contract": conditioning["patch_coordinate_contract"],
        "functional_connectivity_sha256": conditioning[
            "functional_connectivity_sha256"
        ],
    }
    conditioning_identity_from_cache_sources(sources)
    changed = deepcopy(sources)
    roi_identity = changed["dataset"]["roi_feature_extractor_identity"]
    roi_identity["anatcl"]["model"]["state_dict_sha256"] = "0" * 64
    without_digest = dict(roi_identity)
    without_digest.pop("fingerprint_sha256")
    roi_identity["fingerprint_sha256"] = canonical_sha256(without_digest)
    with pytest.raises(RuntimeError, match="ROI-feature protocol identity is invalid"):
        conditioning_identity_from_cache_sources(changed)

    invalid = deepcopy(sources)
    invalid["brainiac"]["source_files_sha256"]["brainiac/load_brainiac.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="BrainIAC conditioning identity"):
        conditioning_identity_from_cache_sources(invalid)


def test_external_scalers_must_match_checkpoint_and_exclude_external_scans(tmp_path):
    split_identity = _paper_split_identity()
    training_ids = sorted(
        record["scan_id"] for record in split_identity["partitions"]["train"]
    )
    scaler_dir = tmp_path / "training-scalers"
    scaler_dir.mkdir()
    provenance_path = scaler_dir / "training_partition.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": training_ids,
                "fitted_scan_ids": training_ids,
            }
        )
    )
    (scaler_dir / "radiomics_scaler.pkl").write_bytes(b"radiomics")
    (scaler_dir / "anatcl_scaler.pkl").write_bytes(b"anatcl")
    artifact = {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "sha256": "f" * 64,
        "scaler_identity_sha256": canonical_sha256(directory_file_sha256(scaler_dir)),
    }
    validate_external_scaler_provenance(
        str(scaler_dir), ["E_001"], artifact, split_identity
    )

    provenance_path.write_text(
        json.dumps(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": training_ids,
                "fitted_scan_ids": training_ids[:-1],
            }
        )
    )
    partial_artifact = {
        **artifact,
        "scaler_identity_sha256": canonical_sha256(directory_file_sha256(scaler_dir)),
    }
    with pytest.raises(RuntimeError, match="scaler provenance is invalid"):
        validate_external_scaler_provenance(
            str(scaler_dir), ["E_001"], partial_artifact, split_identity
        )
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": training_ids,
                "fitted_scan_ids": training_ids,
            }
        )
    )
    artifact["scaler_identity_sha256"] = canonical_sha256(
        directory_file_sha256(scaler_dir)
    )

    with pytest.raises(RuntimeError, match="overlap the scaler-fitting"):
        validate_external_scaler_provenance(
            str(scaler_dir), [training_ids[0]], artifact, split_identity
        )
    (scaler_dir / "anatcl_scaler.pkl").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="do not match"):
        validate_external_scaler_provenance(
            str(scaler_dir), ["E_001"], artifact, split_identity
        )


def test_split_identity_rejects_empty_or_incomplete_partitions():
    scans = ["A_001", "D_001", "D_002"]
    mapping = {"A_001": "A4", "D_001": "ADNI", "D_002": "ADNI"}
    patients = {"A_001": "A", "D_001": "D1", "D_002": "D2"}
    with pytest.raises(ValueError, match="must all be non-empty"):
        build_split_identity(scans, [0, 1], [], [2], mapping, patients)
    with pytest.raises(ValueError, match="cover every scan exactly once"):
        build_split_identity(scans, [0], [1], [1], mapping, patients)


def test_fixed_voxel_size_is_a_validated_protocol_assertion():
    validate_fixed_protocol_config({"data": {"voxel_size_mm": 3.0}})
    with pytest.raises(ValueError, match="requires data.voxel_size_mm=3.0"):
        validate_fixed_protocol_config({"data": {"voxel_size_mm": 2.0}})


def test_explicit_patient_manifest_keeps_multi_part_scan_ids_together(tmp_path):
    scans = [
        "sub-01_ses-1_run-1",
        "sub-01_ses-2_run-1",
        *[f"sub-{index:02d}_ses-1_run-1" for index in range(2, 11)],
    ]
    patients = {
        scan: ("sub-01" if scan.startswith("sub-01_") else scan.split("_")[0])
        for scan in scans
    }
    manifest = tmp_path / "split.json"
    build_seeded_split_manifest(
        scans,
        patients,
        output_path=manifest,
        cohort_manifest_sha256="a" * 64,
        val_frac=0.2,
        test_frac=0.2,
        seed=5,
    )
    train, validation, test = patient_level_split_from_manifest(
        scans,
        patients,
        val_frac=0.2,
        test_frac=0.2,
        seed=5,
        manifest_path=str(manifest),
        manifest_sha256=sha256_file(manifest),
        protocol_profile=PAPER_PROFILE,
    )
    partitions = [train, validation, test]
    containing = [
        partition for partition in partitions if 0 in partition or 1 in partition
    ]
    assert len(containing) == 1
    assert {0, 1}.issubset(set(containing[0]))


def test_explicit_patient_split_rejects_empty_persisted_partition(tmp_path):
    scans = [f"S{index}" for index in range(10)]
    patients = {scan: f"P{index}" for index, scan in enumerate(scans)}
    manifest = tmp_path / "empty-split.json"
    build_seeded_split_manifest(
        scans,
        patients,
        output_path=manifest,
        cohort_manifest_sha256="a" * 64,
        val_frac=0.2,
        test_frac=0.2,
    )
    payload = json.loads(manifest.read_text())
    payload["partitions"]["development_validation"] = []
    unsigned = {key: value for key, value in payload.items() if key != "record_sha256"}
    payload = {**unsigned, "record_sha256": canonical_sha256(unsigned)}
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="must be a non-empty list"):
        patient_level_split_from_manifest(
            scans,
            patients,
            val_frac=0.2,
            test_frac=0.2,
            seed=1,
            manifest_path=str(manifest),
            manifest_sha256=sha256_file(manifest),
            protocol_profile=PAPER_PROFILE,
        )


def test_run_artifact_identity_binds_cache_target_and_evaluation_weights(
    tmp_path, monkeypatch
):
    precomputed = tmp_path / "precomputed"
    fmri = tmp_path / "fmri"
    (precomputed / "hypergraphs").mkdir(parents=True)
    fmri.mkdir()
    conditioning = _conditioning_identity()
    cache_sources = {
        "dataset": {
            "dwi": conditioning["dwi_sha256"],
            "normative_workbook": conditioning["normative_workbook_sha256"],
            "roi_feature_extractor_identity": conditioning["roi_feature_extractor"],
        },
        "brainiac": conditioning["brainiac"],
        "modernbert": conditioning["modernbert"],
        "target_shape": conditioning["target_shape"],
        "patch_size": conditioning["patch_size"],
        "patch_description_schema_version": conditioning[
            "patch_description_schema_version"
        ],
        "patch_coordinate_contract": conditioning["patch_coordinate_contract"],
        "functional_connectivity_sha256": conditioning[
            "functional_connectivity_sha256"
        ],
    }
    for index, scan_id in enumerate(("S1", "S2"), start=1):
        scan_sources = deepcopy(cache_sources)
        cat12_input = {
            "schema": Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA,
            "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
            "authority_manifest_sha256": f"{index}" * 64,
            "scan_record_sha256": f"{index + 2}" * 64,
            "cat12_mwp1_sha256": f"{index + 4}" * 64,
            "segmentation_in_cat12_vbm_sha256": f"{index + 6}" * 64,
        }
        roi_artifact = {
            "format": Connect4Dataset.ROI_FEATURE_ARTIFACT_SCHEMA,
            "scan_id": scan_id,
            "cat12_input": cat12_input,
            "anatcl_provenance_sha256": format(index + 8, "x") * 64,
        }
        roi_artifact["fingerprint_sha256"] = canonical_sha256(roi_artifact)
        scan_sources["dataset"]["roi_feature_artifact_identity"] = roi_artifact
        (precomputed / "hypergraphs" / f"{scan_id}_metadata.json").write_text(
            json.dumps(
                {
                    "source_fingerprint": scan_sources,
                    "source_fingerprint_sha256": canonical_sha256(scan_sources),
                }
            )
        )
        (fmri / f"{scan_id}_fMRI.json").write_text(
            json.dumps({"OutputSHA256": chr(ord("a") + index) * 64})
        )
    cohort_manifest = tmp_path / "cohorts.csv"
    cohort_manifest.write_text(
        "scan_id,patient_id,cohort\nS1,P1,A4\nS2,P2,A4\n"
    )
    extractor = tmp_path / "slimbrain.pt"
    extractor.write_bytes(b"official slimbrain artifact")
    perceptual_extractor = tmp_path / "brainlm.pt"
    perceptual_extractor.write_bytes(b"official brainlm artifact")
    scaler_dir = tmp_path / "scalers"
    scaler_dir.mkdir()
    (scaler_dir / "training_partition.json").write_text(
        json.dumps(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": ["S1", "S2"],
                "fitted_scan_ids": ["S1", "S2"],
            }
        )
    )
    dataset = SimpleNamespace(
        scan_ids=["S1", "S2"],
        precomputed_dir=precomputed,
        fmri_dir=fmri,
        scaler_dir=scaler_dir,
        common_grid_contract={"contract_sha256": "d" * 64},
    )

    def paper_target_identity(scan_id):
        sidecar = json.loads((fmri / f"{scan_id}_fMRI.json").read_text())
        validity_path = fmri / f"{scan_id}_functional_validity_mask.nii.gz"
        if not validity_path.exists():
            validity_path.write_bytes(f"validity:{scan_id}".encode("ascii"))
        identity = {
            "format": PAPER_TARGET_ARTIFACT_IDENTITY_SCHEMA,
            "scan_id": scan_id,
            "target_sidecar_sha256": canonical_sha256(sidecar),
            "target_output_sha256": sidecar["OutputSHA256"],
            "functional_validity_mask_path": str(validity_path),
            "functional_validity_mask_sha256": sha256_file(validity_path),
            "functional_validity_mask_size_bytes": validity_path.stat().st_size,
            "functional_validity_mask_contract": TARGET_VALIDITY_MASK_CONTRACT,
        }
        identity["fingerprint_sha256"] = canonical_sha256(identity)
        return identity

    dataset.target_artifact_identities = {
        scan_id: paper_target_identity(scan_id) for scan_id in dataset.scan_ids
    }
    spec = {
        "enabled": True,
        "name": "slimbrain",
        "checkpoint": str(extractor),
        "checkpoint_sha256": directory_file_sha256(tmp_path)["slimbrain.pt"],
        "source_revision": "a" * 40,
    }
    perceptual_spec = {
        "enabled": True,
        "name": "brainlm",
        "checkpoint": str(perceptual_extractor),
        "checkpoint_sha256": directory_file_sha256(tmp_path)["brainlm.pt"],
        "source_revision": "b" * 40,
    }
    # This unit isolates run-artifact target/structural binding from the much
    # larger official BrainLM loader. Strict official construction, complete
    # identity validation, and generic-ViTMAE rejection have dedicated tests.
    recorded_brainlm_identity = {
        "schema": "test-recorded-brainlm-identity-v1",
        "name": "brainlm",
        "mni_authority": {
            "file_sha256": "a" * 64,
            "authority_record_sha256": "b" * 64,
            "ordered_scan_ids_sha256": canonical_sha256(["S1"]),
        },
        "record_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        "models.brainlm_context.configured_brainlm_identity",
        lambda _spec: deepcopy(recorded_brainlm_identity),
    )
    monkeypatch.setattr(
        "data.protocol._is_recorded_extractor_fingerprint",
        lambda value, expected_name: (
            expected_name == "brainlm" and value == recorded_brainlm_identity
        ),
    )
    first = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=spec,
        perceptual_extractor_spec=perceptual_spec,
    )
    (fmri / "S1_fMRI.json").write_text(json.dumps({"OutputSHA256": "c" * 64}))
    dataset.target_artifact_identities["S1"] = paper_target_identity("S1")
    second = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=spec,
        perceptual_extractor_spec=perceptual_spec,
    )
    assert first["sha256"] != second["sha256"]
    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        build_run_artifact_identity(
            dataset,
            cohort_manifest_path=str(cohort_manifest),
            evaluation_extractor_spec={**spec, "checkpoint_sha256": "0" * 64},
            perceptual_extractor_spec=perceptual_spec,
        )

    # Training authenticates only its train+development target allowlist.  A
    # sealed target may be absent or replaced by unreadable sentinel bytes and
    # must not change, or even be needed to reproduce, the run identity.
    dataset.target_scan_ids = frozenset({"S1"})
    (fmri / "S2_fMRI.json").unlink()
    sealed_absent = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_spec,
        target_scan_ids=["S1"],
    )
    (fmri / "S2_fMRI.json").write_bytes(b"sealed sentinel: must never be opened")
    sealed_sentinel = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_spec,
        target_scan_ids=["S1"],
    )
    assert sealed_absent == sealed_sentinel
    assert sealed_absent["num_authenticated_targets"] == 1
    assert sealed_absent["num_sealed_targets_unopened"] == 1
    assert sealed_absent["target_access_contract"]["sealed_targets_opened"] is False
    assert sealed_absent["evaluation_extractor"] == {
        "status": "deferred-unopened-by-training",
        "purpose": "final-test-only",
    }

    def native_identity(output_sha256):
        identity = {
            "format": "connect4-native-target-artifact-identity-v1",
            "scan_id": "S1",
            "role": "train",
            "paper_certified": False,
            "certification_status": "NON_CERTIFIED_RECOVERY_PREPROCESSING",
            "publication_sha256": "1" * 64,
            "publication_record_sha256": "2" * 64,
            "target_sidecar_sha256": "3" * 64,
            "target_sidecar_record_sha256": "4" * 64,
            "target_output_sha256": output_sha256,
            "native_bold_sha256": "6" * 64,
            "raw_source_identity": {
                "schema": "connect4-native-target-source-identity-v1",
                "scan_id": "S1",
                "raw_t1_sha256": "7" * 64,
                "raw_synthseg_mask_sha256": "8" * 64,
                "raw_fmri_sha256": "9" * 64,
            },
            "spatial_target_join_sha256": "a" * 64,
            "structural_alignment_authority_sha256": "b" * 64,
            "completed_set_sha256": "c" * 64,
            "completed_set_record_sha256": "d" * 64,
            "completed_set_commit_marker_sha256": "e" * 64,
            "success_receipt_sha256": "f" * 64,
            "success_receipt_record_sha256": "0" * 64,
            "success_receipt_commit_marker_sha256": "1" * 64,
            "native_batch_sha256": "2" * 64,
            "native_scan_provenance_sha256": "3" * 64,
            "reviewed_native_source_sha256": "4" * 64,
            "architecture_shape": [64, 80, 64],
            "num_frames": 128,
            "tr_seconds": 3.0,
        }
        identity["fingerprint_sha256"] = canonical_sha256(identity)
        return identity

    # Recovery run identity consumes the already recursively validated identity
    # object and never falls back to opening a target sidecar. The sealed scan's
    # sentinel remains irrelevant, while any admitted-target change is bound.
    dataset.recovery_profile = True
    dataset.common_grid_contract = None
    dataset.native_alignment_authority_sha256 = "b" * 64
    dataset.target_scan_roles = {"S1": "train"}
    dataset.target_artifact_identities = {"S1": native_identity("5" * 64)}
    (fmri / "S1_fMRI.json").write_bytes(b"must not be opened in recovery")
    role_items = [{"scan_id": "S1", "role": "train"}]
    brainlm_scope = {
        "schema": "connect4-brainlm-authority-split-scope-v1",
        "authority_file_sha256": "a" * 64,
        "authority_record_sha256": "b" * 64,
        "authority_ordered_scan_ids_sha256": canonical_sha256(["S1"]),
        "split_assignment_sha256": "c" * 64,
        "split_identity_sha256": "d" * 64,
        "target_scan_roles_sha256": canonical_sha256(role_items),
        "authorized_scan_count": 1,
        "train_scan_count": 1,
        "development_scan_count": 0,
        "sealed_scan_count": 1,
        "sealed_scan_ids_sha256": canonical_sha256(["S2"]),
        "sealed_scans_authorized": False,
        "functional_target_bytes_opened": False,
    }
    brainlm_scope["record_sha256"] = canonical_sha256(brainlm_scope)
    with pytest.raises(RuntimeError, match="requires a split-bound BrainLM"):
        build_run_artifact_identity(
            dataset,
            cohort_manifest_path=str(cohort_manifest),
            evaluation_extractor_spec=None,
            perceptual_extractor_spec=perceptual_spec,
            target_scan_ids=["S1"],
        )
    recovery_first = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_spec,
        brainlm_authority_scope=brainlm_scope,
        target_scan_ids=["S1"],
    )
    dataset.target_artifact_identities = {"S1": native_identity("6" * 64)}
    recovery_changed = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_spec,
        brainlm_authority_scope=brainlm_scope,
        target_scan_ids=["S1"],
    )
    assert recovery_first["sha256"] != recovery_changed["sha256"]
    assert recovery_first["target_artifact_identities_sha256"] != (
        recovery_changed["target_artifact_identities_sha256"]
    )
    assert recovery_first["spatial_authority"] == {
        "profile": "non-certified-native-stage-b",
        "native_alignment_authority_sha256": "b" * 64,
    }

    sealed_metadata_path = precomputed / "hypergraphs" / "S2_metadata.json"
    sealed_metadata = json.loads(sealed_metadata_path.read_text())
    sealed_metadata["sealed_structural_marker"] = "changed"
    sealed_metadata_path.write_text(json.dumps(sealed_metadata))
    structurally_changed = build_run_artifact_identity(
        dataset,
        cohort_manifest_path=str(cohort_manifest),
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_spec,
        brainlm_authority_scope=brainlm_scope,
        target_scan_ids=["S1"],
    )
    assert structurally_changed["sha256"] != sealed_sentinel["sha256"]
    assert structurally_changed[
        "structural_artifact_identities_sha256"
    ] != sealed_sentinel["structural_artifact_identities_sha256"]


def test_recursive_directory_fingerprint_binds_nested_files(tmp_path):
    source = tmp_path / "model"
    nested = source / "snapshots" / "revision"
    nested.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    weight = nested / "model.safetensors"
    weight.write_bytes(b"weights-v1")
    first = directory_file_sha256(source)
    assert set(first) == {"config.json", "snapshots/revision/model.safetensors"}
    weight.write_bytes(b"weights-v2")
    assert directory_file_sha256(source) != first


def test_scaler_provenance_must_name_exact_training_partition(tmp_path):
    scaler_dir = tmp_path / "scalers"
    scaler_dir.mkdir()
    path = scaler_dir / "training_partition.json"
    path.write_text(
        json.dumps(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": ["S1", "S2"],
                "fitted_scan_ids": ["S1", "S2"],
            }
        )
    )
    validate_training_scaler_provenance(str(scaler_dir), ["S2", "S1"])
    payload = json.loads(path.read_text())
    payload["fitted_scan_ids"] = ["S1"]
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="exact current fixed training partition"):
        validate_training_scaler_provenance(str(scaler_dir), ["S1", "S2"])
    payload["fitted_scan_ids"] = ["S1", "S2"]
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="exact current fixed training partition"):
        validate_training_scaler_provenance(str(scaler_dir), ["S1", "S3"])
