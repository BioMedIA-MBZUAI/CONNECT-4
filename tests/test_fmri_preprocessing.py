import numpy as np
import hashlib
import json
import importlib
import nibabel as nib
from nibabel.processing import resample_from_to
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from architecture_contract import synthesis_architecture_contract
from preprocessing.conform import (
    STRUCTURAL_MANIFEST_SCHEMA_VERSION,
    conform_4d,
    load_common_grid_contract,
    set_num_frames,
    zscore_inbrain,
)
from preprocessing.build_common_grid_contract import build_common_grid_contract
from data.provenance import sha256_file
from preprocessing.preprocess_fmri import (
    FMRI_INTENSITY_NORMALIZATION_CONTRACT,
    FMRI_INTENSITY_NORMALIZATION_DOMAIN,
    FMRIPREP_EXECUTION_SCHEMA_VERSION,
    FMRIPREP_EVIDENCE_SCHEMA_VERSION,
    _apply_rigid,
    _artifact_evidence,
    _coregister_data,
    _estimate_rigid_mutual_information,
    _infer_slice_times,
    _normalised_mutual_information,
    _resample_to_tr,
    _recovery_intensity_normalize,
    _source_tr_seconds,
    _slice_timing_correct,
    _spatial_smooth,
    _temporal_filter,
)
from preprocessing.preprocess_t1 import preprocess_t1
from preprocessing.preprocess_seg import preprocess_seg
from preprocessing.source_acquisition_identity import (
    SOURCE_ACQUISITION_PURPOSE,
    SOURCE_ACQUISITION_SCHEMA,
    STRUCTURAL_SOURCE_AUTHORITY_PURPOSE,
    STRUCTURAL_SOURCE_AUTHORITY_SCHEMA,
    SYNTHSEG_PROVENANCE_SCHEMA,
    SourceAcquisitionError,
    canonical_sha256,
    load_source_acquisition_identity,
    snapshot_bids_input_inventory,
    structural_source_projection,
)
from tests.support import (
    write_fmriprep_runtime_fixture,
    write_source_acquisition_fixture,
)


def _signed_record(value):
    record = dict(value)
    record["record_sha256"] = canonical_sha256(record)
    return record


def _source_artifact(path: Path):
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _write_fmriprep_derivatives(
    tmp_path: Path,
    *,
    frames: int = 8,
    motion_columns=("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"),
    transform_source: str = "boldref",
    slice_timing_corrected: bool = True,
):
    bids_root = tmp_path / "bids"
    derivatives = tmp_path / "fmriprep"
    func = derivatives / "sub-01" / "func"
    anat = derivatives / "sub-01" / "anat"
    logs = derivatives / "logs"
    raw_anat = bids_root / "sub-01" / "anat"
    raw_func = bids_root / "sub-01" / "func"
    raw_anat.mkdir(parents=True)
    raw_func.mkdir(parents=True)
    func.mkdir(parents=True)
    anat.mkdir(parents=True)
    logs.mkdir(parents=True)

    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    bold = func / "sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    bold_brain_mask = func / "sub-01_task-rest_space-T1w_desc-brain_mask.nii.gz"
    bold_metadata = func / "sub-01_task-rest_space-T1w_desc-preproc_bold.json"
    confounds = func / "sub-01_task-rest_desc-confounds_timeseries.tsv"
    transform_suffix = (
        "mode-image_desc-coreg_xfm.txt"
        if transform_source == "boldref"
        else "mode-image_xfm.txt"
    )
    transform = func / (
        f"sub-01_task-rest_from-{transform_source}_to-T1w_{transform_suffix}"
    )
    t1 = anat / "sub-01_desc-preproc_T1w.nii.gz"
    t1_metadata = anat / "sub-01_desc-preproc_T1w.json"
    raw_t1 = raw_anat / "sub-01_T1w.nii.gz"
    raw_bold = raw_func / "sub-01_task-rest_bold.nii.gz"
    raw_bold_metadata = raw_func / "sub-01_task-rest_bold.json"
    synthseg_mask = tmp_path / "sub-01_synthseg.nii.gz"
    synthseg_provenance = tmp_path / "sub-01_synthseg_provenance.json"
    source_identity = tmp_path / "sub-01_source_acquisition.json"
    dataset_description = derivatives / "dataset_description.json"
    stdout_log = logs / "connect4-fmriprep_sub-01.stdout.log"
    stderr_log = logs / "connect4-fmriprep_sub-01.stderr.log"

    bold_image = nib.Nifti1Image(np.ones((4, 4, 4, frames), dtype=np.float32), affine)
    bold_image.header.set_zooms((3.0, 3.0, 3.0, 3.0))
    nib.save(bold_image, bold)
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), affine),
        bold_brain_mask,
    )
    nib.save(bold_image, raw_bold)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), affine), t1)
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), affine), raw_t1)
    (bids_root / "dataset_description.json").write_text(
        json.dumps({"Name": "CONNECT-4 test input", "BIDSVersion": "1.10.0"})
    )
    raw_bold_metadata.write_text(
        json.dumps(
            {
                "TaskName": "rest",
                "RepetitionTime": 3.0,
                "SliceTiming": [0.0, 0.75, 1.5, 2.25],
            }
        )
    )
    nib.save(
        nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), affine),
        synthseg_mask,
    )
    bold_metadata.write_text(
        json.dumps(
            {
                "SpatialReference": "T1w",
                "RepetitionTime": 3.0,
                "SliceEncodingDirection": "k",
                "SliceTimingCorrected": slice_timing_corrected,
                "StartTime": 0.375,
                "Sources": ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"],
            }
        )
    )
    t1_metadata.write_text(
        json.dumps({"Sources": ["bids::sub-01/anat/sub-01_T1w.nii.gz"]})
    )
    confounds.write_text(
        "\t".join(motion_columns)
        + "\n"
        + "\n".join("\t".join("0" for _ in motion_columns) for _ in range(frames))
        + "\n"
    )
    transform.write_text(
        "#Insight Transform File V1.0\n"
        "Transform: AffineTransform_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n"
        "FixedParameters: 0 0 0\n"
    )
    dataset_description.write_text(
        json.dumps(
            {
                "Name": "fMRIPrep outputs",
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "fMRIPrep", "Version": "24.1.1"}],
            }
        )
    )
    stdout_log.write_text("fMRIPrep completed\n")
    stderr_log.write_text("")
    producer = {
        "name": "SynthSeg",
        "version": "2.0",
        "source_revision": "a" * 40,
        "model_sha256": "b" * 64,
        "container_image": "docker://freesurfer/synthseg:test",
        "container_manifest_sha256": "c" * 64,
    }
    synthseg_record = _signed_record(
        {
            "schema": SYNTHSEG_PROVENANCE_SCHEMA,
            "scan_id": "sub-01_task-rest",
            "producer": producer,
            "input_t1": _source_artifact(raw_t1),
            "source_t1_sha256": sha256_file(raw_t1),
            "output_mask": _source_artifact(synthseg_mask),
        }
    )
    synthseg_provenance.write_text(json.dumps(synthseg_record))
    provenance_artifact = _source_artifact(synthseg_provenance)
    provenance_artifact["record_sha256"] = synthseg_record["record_sha256"]
    source_record = _signed_record(
        {
            "schema": SOURCE_ACQUISITION_SCHEMA,
            "purpose": SOURCE_ACQUISITION_PURPOSE,
            "scan_id": "sub-01_task-rest",
            "participant_label": "01",
            "bids_root": str(bids_root.resolve()),
            "bids_entities": {
                "raw_t1": {"suffix": "T1w", "sub": "01"},
                "raw_bold": {
                    "suffix": "bold",
                    "sub": "01",
                    "task": "rest",
                },
            },
            "bids_input_inventory": snapshot_bids_input_inventory(
                bids_root.resolve(),
                participant="01",
                raw_t1=raw_t1.resolve(),
                raw_bold=raw_bold.resolve(),
            ),
            "raw_t1": _source_artifact(raw_t1),
            "raw_bold": _source_artifact(raw_bold),
            "synthseg": {
                "mask": _source_artifact(synthseg_mask),
                "provenance": provenance_artifact,
                "producer": producer,
            },
        }
    )
    source_identity.write_text(json.dumps(source_record))
    structural_authority = tmp_path / "sub-01_structural_source_authority.json"
    structural_record = _signed_record(
        {
            "schema": STRUCTURAL_SOURCE_AUTHORITY_SCHEMA,
            "purpose": STRUCTURAL_SOURCE_AUTHORITY_PURPOSE,
            "scan_id": "sub-01_task-rest",
            "raw_t1": _source_artifact(raw_t1),
            "synthseg": {
                "mask": _source_artifact(synthseg_mask),
                "provenance": provenance_artifact,
                "producer": producer,
            },
        }
    )
    structural_authority.write_text(json.dumps(structural_record))
    return {
        "bids_root": bids_root,
        "derivatives": derivatives,
        "bold": bold,
        "bold_brain_mask": bold_brain_mask,
        "metadata": bold_metadata,
        "confounds": confounds,
        "transform": transform,
        "t1": t1,
        "t1_metadata": t1_metadata,
        "raw_t1": raw_t1,
        "raw_bold": raw_bold,
        "raw_bold_metadata": raw_bold_metadata,
        "synthseg_mask": synthseg_mask,
        "synthseg_provenance": synthseg_provenance,
        "source_identity": source_identity,
        "source_identity_sha256": sha256_file(source_identity),
        "source_identity_record_sha256": source_record["record_sha256"],
        "structural_authority": structural_authority,
        "structural_authority_sha256": sha256_file(structural_authority),
        "dataset_description": dataset_description,
        "stdout": stdout_log,
        "stderr": stderr_log,
    }


def _source_identity_arguments(paths):
    return {
        "source_acquisition_identity_path": str(paths["source_identity"]),
        "source_acquisition_identity_sha256": paths["source_identity_sha256"],
    }


def _runtime_identity_arguments(runtime):
    return {
        "fmriprep_runtime_identity_path": str(runtime["path"]),
        "fmriprep_runtime_identity_sha256": runtime["sha256"],
    }


def _version_probe_result(version="24.1.1"):
    return SimpleNamespace(
        returncode=0,
        stdout=f"fMRIPrep v{version}\n".encode(),
        stderr=b"",
    )


def _publish_fmriprep_fixture(paths, fresh_derivatives: Path):
    shutil.copyfile(
        paths["dataset_description"], fresh_derivatives / "dataset_description.json"
    )
    shutil.copytree(paths["derivatives"] / "sub-01", fresh_derivatives / "sub-01")
    (fresh_derivatives / "connect4-work").mkdir()


def _write_execution_record(paths, *, ignored_features=()):
    ignored = list(ignored_features)
    executable = paths["bids_root"].parent / "fmriprep-consumer-fixture"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        paths["bids_root"].parent,
        executable=executable,
        version="24.1.1",
    )
    source_identity = json.loads(paths["source_identity"].read_text())
    input_inventory = source_identity["bids_input_inventory"]
    command = [
        str(executable.resolve()),
        str(paths["bids_root"]),
        str(paths["derivatives"]),
        "participant",
        "--participant-label",
        "01",
        "--output-spaces",
        "T1w",
        "--slice-time-ref",
        "0.5",
        "--level",
        "full",
        "--work-dir",
        str(paths["derivatives"] / "connect4-work"),
    ]
    if ignored:
        command += ["--ignore", *ignored]
    artifact_paths = [
        paths["dataset_description"],
        paths["bold"],
        paths["bold_brain_mask"],
        paths["metadata"],
        paths["confounds"],
        paths["transform"],
        paths["t1"],
        paths["t1_metadata"],
        paths["stdout"],
        paths["stderr"],
    ]
    probe_stdout = "fMRIPrep v24.1.1\n"
    version_probe = {
        "Command": [str(executable.resolve()), "--version"],
        "ReturnCode": 0,
        "ObservedVersion": "24.1.1",
        "StdoutSHA256": hashlib.sha256(probe_stdout.encode()).hexdigest(),
        "StderrSHA256": hashlib.sha256(b"").hexdigest(),
        "Stdout": probe_stdout,
        "Stderr": "",
    }
    runtime_identity = runtime["identity"]
    record = {
        "SchemaVersion": FMRIPREP_EXECUTION_SCHEMA_VERSION,
        "Wrapper": "preprocessing.run_fmriprep",
        "Success": True,
        "ReturnCode": 0,
        "StartedAtUTC": "2026-08-29T10:00:00Z",
        "CompletedAtUTC": "2026-08-29T10:01:00Z",
        "Command": command,
        "BIDSRoot": str(paths["bids_root"]),
        "DerivativesRoot": str(paths["derivatives"]),
        "WorkRoot": str(paths["derivatives"] / "connect4-work"),
        "ScanIdentity": {
            "ScanID": "sub-01_task-rest",
            "ParticipantLabel": "01",
            "Session": None,
            "Task": "rest",
            "Run": None,
            "Acquisition": None,
            "Direction": None,
            "Echo": None,
            "RawBOLDRelativePath": input_inventory["raw_bold_relative_path"],
            "RawT1wRelativePath": input_inventory["raw_t1_relative_path"],
        },
        "ParticipantLabel": "01",
        "OutputSpaces": ["T1w"],
        "ExtraArguments": [],
        "IgnoredFeatures": ignored,
        "SliceTimingEnabledByWrapper": True,
        "SliceTimingReference": 0.5,
        "FMRIPrepGeneratedBy": {"Name": "fMRIPrep", "Version": "24.1.1"},
        "FreshDerivativesGeneration": {
            "GenerationID": "d" * 64,
            "ScanID": "sub-01_task-rest",
            "Path": str(paths["derivatives"]),
            "PathAbsentBeforeCreation": True,
            "CreatedExclusively": True,
            "InitiallyEmpty": True,
            "NoReuse": True,
        },
        "BIDSInputInventory": {
            "Identity": input_inventory,
            "VerifiedBeforeExecution": True,
            "VerifiedAfterExecution": True,
        },
        "ExecutableIdentity": runtime_identity["executable"],
        "RuntimeIdentity": {
            "Artifact": {
                "Path": str(runtime["path"]),
                "SHA256": runtime["sha256"],
                "SizeBytes": runtime["path"].stat().st_size,
                "RecordSHA256": runtime_identity["record_sha256"],
            },
            "Identity": runtime_identity,
            "VerifiedBeforeExecution": True,
            "VerifiedAfterExecution": True,
        },
        "VersionProbeBeforeExecution": version_probe,
        "VersionProbeAfterExecution": version_probe,
        "SourceAcquisitionIdentity": {
            "Artifact": {
                "Path": str(paths["source_identity"].resolve()),
                "SHA256": paths["source_identity_sha256"],
                "SizeBytes": paths["source_identity"].stat().st_size,
                "RecordSHA256": paths["source_identity_record_sha256"],
            },
            "Identity": json.loads(paths["source_identity"].read_text()),
            "VerifiedBeforeExecution": True,
            "VerifiedAfterExecution": True,
        },
        "StdoutLog": _artifact_evidence(paths["stdout"], paths["derivatives"]),
        "StderrLog": _artifact_evidence(paths["stderr"], paths["derivatives"]),
        "Artifacts": [
            _artifact_evidence(path, paths["derivatives"]) for path in artifact_paths
        ],
    }
    record_path = (
        paths["derivatives"]
        / "logs"
        / "connect4-fmriprep_sub-01_task-rest_execution.json"
    )
    record_path.write_text(json.dumps(record))
    paths["record"] = record_path
    paths["runtime_identity"] = runtime["path"]
    paths["runtime_executable"] = executable
    return record_path


def _patch_fast_postprocessing(module, monkeypatch, calls):
    def fast_masked_smoothing(array, mask, *_args):
        calls.append("spatial smoothing")
        result = np.asarray(array).copy()
        result[~np.asarray(mask, dtype=bool), :] = 0.0
        return result

    monkeypatch.setattr(
        module,
        "_spatial_smooth",
        fast_masked_smoothing,
    )
    monkeypatch.setattr(
        module,
        "_temporal_filter",
        lambda array, **_kwargs: calls.append("temporal filtering") or array,
    )
    monkeypatch.setattr(
        module,
        "conform_4d",
        lambda image, order, **_kwargs: calls.append("harmonisation") or image,
    )
    monkeypatch.setattr(module, "_masked_spatial_gradient_energy", lambda *_args: 1.0)
    monkeypatch.setattr(module, "_resample_to_tr", lambda array, *_args: array)
    monkeypatch.setattr(module, "set_num_frames", lambda array, _frames: array)
    monkeypatch.setattr(module, "TARGET_FRAMES", 8)


def _mean_absolute_spatial_gradient(array):
    values = np.asarray(array, dtype=np.float64)
    return float(
        np.mean(
            np.concatenate(
                [np.abs(np.diff(values, axis=axis)).ravel() for axis in range(3)]
            )
        )
    )


def test_sequential_linear_grid_resampling_erases_texture_vs_one_composed_pass():
    indices = np.indices((32, 32, 32))
    checkerboard = (indices.sum(axis=0) % 2).astype(np.float32)
    source = nib.Nifti1Image(checkerboard, np.eye(4))
    intermediate_affine = np.eye(4)
    intermediate_affine[:3, 3] = 0.4
    target_affine = np.eye(4)
    target_affine[:3, 3] = 0.8

    direct = resample_from_to(source, (source.shape, target_affine), order=1).get_fdata(
        dtype=np.float32
    )
    intermediate = resample_from_to(
        source, (source.shape, intermediate_affine), order=1
    )
    sequential = resample_from_to(
        intermediate, (source.shape, target_affine), order=1
    ).get_fdata(dtype=np.float32)

    direct_gradient = _mean_absolute_spatial_gradient(direct[2:-2, 2:-2, 2:-2])
    sequential_gradient = _mean_absolute_spatial_gradient(sequential[2:-2, 2:-2, 2:-2])
    assert direct_gradient > 0.20
    assert sequential_gradient / direct_gradient < 0.001


def _production_grid_arguments(tmp_path, paths):
    mask_path = tmp_path / "anatomical_mask.nii.gz"
    t1_image = nib.load(paths["t1"])
    nib.save(
        nib.Nifti1Image(np.ones(t1_image.shape, dtype=np.uint8), t1_image.affine),
        mask_path,
    )
    manifest_path = tmp_path / "structural_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": STRUCTURAL_MANIFEST_SCHEMA_VERSION,
                "functional_data_used": False,
                "structural_reference": {
                    "path": str(paths["t1"]),
                    "sha256": sha256_file(paths["t1"]),
                    "modality": "T1w",
                    "cohort_derived": True,
                    "functional_data_used": False,
                    "derivation_method": "synthetic-test-reference",
                },
                "scans": [
                    {
                        "scan_id": "sub-01",
                        "t1w_path": str(paths["t1"]),
                        "t1w_sha256": sha256_file(paths["t1"]),
                    }
                ],
            }
        )
    )
    contract_path = build_common_grid_contract(
        paths["t1"],
        manifest_path,
        tmp_path / "common_grid.json",
        patch_multiple=(1, 1, 1),
    )
    return {
        "brain_mask_path": str(mask_path),
        "common_grid_contract_path": str(contract_path),
        "common_grid_contract_sha256": sha256_file(contract_path),
    }


def test_source_acquisition_identity_binds_raw_bids_and_synthseg(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    identity, evidence = load_source_acquisition_identity(
        paths["source_identity"],
        expected_sha256=paths["source_identity_sha256"],
        expected_bids_root=paths["bids_root"],
        expected_participant="sub-01",
        expected_raw_t1=paths["raw_t1"],
        expected_raw_bold=paths["raw_bold"],
        expected_synthseg_mask=paths["synthseg_mask"],
    )
    assert identity["bids_entities"]["raw_t1"] == {
        "suffix": "T1w",
        "sub": "01",
    }
    assert identity["bids_entities"]["raw_bold"] == {
        "suffix": "bold",
        "sub": "01",
        "task": "rest",
    }
    assert evidence["record_sha256"] == paths["source_identity_record_sha256"]


def test_signed_synthseg_provenance_cannot_name_another_t1(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    provenance = json.loads(paths["synthseg_provenance"].read_text())
    provenance.pop("record_sha256")
    provenance["input_t1"] = _source_artifact(paths["raw_bold"])
    provenance = _signed_record(provenance)
    paths["synthseg_provenance"].write_text(json.dumps(provenance))

    identity = json.loads(paths["source_identity"].read_text())
    identity.pop("record_sha256")
    provenance_artifact = _source_artifact(paths["synthseg_provenance"])
    provenance_artifact["record_sha256"] = provenance["record_sha256"]
    identity["synthseg"]["provenance"] = provenance_artifact
    identity = _signed_record(identity)
    paths["source_identity"].write_text(json.dumps(identity))

    with pytest.raises(
        SourceAcquisitionError,
        match="source-T1/output-mask binding differs",
    ):
        load_source_acquisition_identity(
            paths["source_identity"],
            expected_sha256=sha256_file(paths["source_identity"]),
        )


def test_source_acquisition_identity_rejects_postsigning_source_swap(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    with paths["raw_bold"].open("ab") as stream:
        stream.write(b"swapped-after-signing")
    with pytest.raises(SourceAcquisitionError, match="raw BIDS BOLD bytes differ"):
        load_source_acquisition_identity(
            paths["source_identity"],
            expected_sha256=paths["source_identity_sha256"],
        )


def test_source_acquisition_identity_rejects_inherited_metadata_mutation(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    inherited = paths["bids_root"] / "task-rest_bold.json"
    inherited.write_text(json.dumps({"TaskName": "rest", "FlipAngle": 60}))
    identity = json.loads(paths["source_identity"].read_text())
    identity.pop("record_sha256")
    identity["bids_input_inventory"] = snapshot_bids_input_inventory(
        paths["bids_root"],
        participant="01",
        raw_t1=paths["raw_t1"],
        raw_bold=paths["raw_bold"],
    )
    identity = _signed_record(identity)
    paths["source_identity"].write_text(json.dumps(identity))
    expected_sha256 = sha256_file(paths["source_identity"])

    inherited.write_text(json.dumps({"TaskName": "rest", "FlipAngle": 90}))
    with pytest.raises(SourceAcquisitionError, match="BIDS input inventory differs"):
        load_source_acquisition_identity(
            paths["source_identity"], expected_sha256=expected_sha256
        )


def test_bids_input_inventory_binds_fieldmaps_sbref_and_participant_tables(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    participant_root = paths["bids_root"] / "sub-01"
    fmap = participant_root / "fmap" / "sub-01_dir-AP_epi.nii.gz"
    fmap.parent.mkdir()
    fmap.write_bytes(b"field-map-bytes")
    fmap.with_name(fmap.name.removesuffix(".nii.gz") + ".json").write_text(
        json.dumps({"PhaseEncodingDirection": "j-"})
    )
    sbref = participant_root / "func" / "sub-01_task-rest_sbref.nii.gz"
    sbref.write_bytes(b"sbref-bytes")
    scans = participant_root / "sub-01_scans.tsv"
    scans.write_text("filename\nfunc/sub-01_task-rest_bold.nii.gz\n")

    inventory = snapshot_bids_input_inventory(
        paths["bids_root"],
        participant="01",
        raw_t1=paths["raw_t1"],
        raw_bold=paths["raw_bold"],
    )
    relative_paths = {value["relative_path"] for value in inventory["files"]}
    assert fmap.relative_to(paths["bids_root"]).as_posix() in relative_paths
    assert sbref.relative_to(paths["bids_root"]).as_posix() in relative_paths
    assert scans.relative_to(paths["bids_root"]).as_posix() in relative_paths


def test_bids_input_inventory_rejects_second_participant_bold_run(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    other_bold = paths["raw_bold"].with_name("sub-01_task-rest_run-2_bold.nii.gz")
    shutil.copyfile(paths["raw_bold"], other_bold)
    with pytest.raises(
        SourceAcquisitionError, match="exactly the authenticated raw BOLD"
    ):
        snapshot_bids_input_inventory(
            paths["bids_root"],
            participant="01",
            raw_t1=paths["raw_t1"],
            raw_bold=paths["raw_bold"],
        )


def test_signed_source_identity_rejects_cross_subject_bold(tmp_path):
    paths = _write_fmriprep_derivatives(tmp_path)
    other_bold = paths["bids_root"] / "sub-02" / "func" / "sub-02_task-rest_bold.nii.gz"
    other_bold.parent.mkdir(parents=True)
    shutil.copyfile(paths["raw_bold"], other_bold)
    identity = json.loads(paths["source_identity"].read_text())
    identity.pop("record_sha256")
    identity["raw_bold"] = _source_artifact(other_bold)
    identity["bids_entities"]["raw_bold"]["sub"] = "02"
    identity = _signed_record(identity)
    paths["source_identity"].write_text(json.dumps(identity))
    with pytest.raises(SourceAcquisitionError, match="BIDS entities differ"):
        load_source_acquisition_identity(
            paths["source_identity"],
            expected_sha256=sha256_file(paths["source_identity"]),
        )


def test_preprocess_subject_authenticates_sources_before_any_processing(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_subject")
    paths = _write_fmriprep_derivatives(tmp_path)
    calls = []

    def fake_seg(*_args, **kwargs):
        calls.append("seg")
        assert kwargs["structural_source_authority_path"] == str(
            paths["structural_authority"]
        )
        return "seg_common.nii.gz"

    def fake_t1(*_args, **kwargs):
        calls.append("t1")
        assert kwargs["segmentation_path"] == "seg_common.nii.gz"
        assert (
            kwargs["structural_source_authority_sha256"]
            == paths["structural_authority_sha256"]
        )
        return "T1w_common.nii.gz"

    monkeypatch.setattr(module, "preprocess_t1", fake_t1)
    monkeypatch.setattr(module, "preprocess_seg", fake_seg)

    def fake_fmri(*_args, **kwargs):
        calls.append("fmri")
        assert kwargs["source_acquisition_identity_path"] == str(
            paths["source_identity"]
        )
        assert (
            kwargs["source_acquisition_identity_sha256"]
            == paths["source_identity_sha256"]
        )

    monkeypatch.setattr(module, "preprocess_fmri", fake_fmri)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_subject",
            "--t1",
            str(paths["raw_t1"]),
            "--seg",
            str(paths["synthseg_mask"]),
            "--fmri",
            str(paths["bold"]),
            "--fmriprep-t1",
            str(paths["t1"]),
            "--structural-source-authority",
            str(paths["structural_authority"]),
            "--structural-source-authority-sha256",
            paths["structural_authority_sha256"],
            "--source-acquisition-identity",
            str(paths["source_identity"]),
            "--source-acquisition-identity-sha256",
            paths["source_identity_sha256"],
            "--common-grid-contract",
            str(tmp_path / "grid.json"),
            "--common-grid-contract-sha256",
            "d" * 64,
            "--out_dir",
            str(tmp_path / "subject-output"),
        ],
    )
    module.main()
    assert calls == ["seg", "t1", "fmri"]


def test_preprocess_subject_rejects_t1_swap_before_creating_outputs(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_subject")
    paths = _write_fmriprep_derivatives(tmp_path)
    swapped_t1 = tmp_path / "sub-01_T1w_swapped.nii.gz"
    shutil.copyfile(paths["raw_t1"], swapped_t1)
    calls = []
    monkeypatch.setattr(
        module,
        "preprocess_t1",
        lambda *_args, **_kwargs: calls.append("t1"),
    )
    output = tmp_path / "subject-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_subject",
            "--t1",
            str(swapped_t1),
            "--seg",
            str(paths["synthseg_mask"]),
            "--fmri",
            str(paths["bold"]),
            "--fmriprep-t1",
            str(paths["t1"]),
            "--structural-source-authority",
            str(paths["structural_authority"]),
            "--structural-source-authority-sha256",
            paths["structural_authority_sha256"],
            "--source-acquisition-identity",
            str(paths["source_identity"]),
            "--source-acquisition-identity-sha256",
            paths["source_identity_sha256"],
            "--common-grid-contract",
            str(tmp_path / "grid.json"),
            "--common-grid-contract-sha256",
            "d" * 64,
            "--out_dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert calls == []
    assert not output.exists()


def test_interleaved_slice_times_are_indexed_by_slice():
    times = _infer_slice_times(5, tr=3.0, order="interleaved-ascending")
    np.testing.assert_allclose(times, [0.0, 1.8, 0.6, 2.4, 1.2])


def test_conform_4d_preserves_or_explicitly_sets_each_file_tr(monkeypatch):
    conform_module = importlib.import_module("preprocessing.conform")
    monkeypatch.setattr(conform_module, "conform_volume", lambda image, order=1: image)
    image = nib.Nifti1Image(np.zeros((2, 3, 4, 2), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm", "msec")
    image.header.set_zooms((1.0, 1.0, 1.0, 3520.0))

    preserved = conform_4d(image)
    assert preserved.header.get_xyzt_units() == ("mm", "msec")
    assert preserved.header.get_zooms()[3] == pytest.approx(3520.0)

    explicit = conform_4d(image, tr_seconds=3.52)
    assert explicit.header.get_xyzt_units() == ("mm", "sec")
    assert explicit.header.get_zooms()[3] == pytest.approx(3.52)


def test_slice_timing_at_reference_is_identity():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(3, 4, 2, 20)).astype(np.float32)
    corrected = _slice_timing_correct(
        data,
        tr=3.0,
        slice_times=[1.5, 1.5],
        slice_axis=2,
        reference_fraction=0.5,
    )
    np.testing.assert_allclose(corrected, data, atol=1e-6)


def test_spatial_smoothing_does_not_mix_time():
    data = np.zeros((9, 9, 9, 2), dtype=np.float32)
    data[4, 4, 4, 0] = 1.0
    mask = np.ones(data.shape[:3], dtype=bool)
    smoothed = _spatial_smooth(data, mask, np.diag([3.0, 3.0, 3.0, 1.0]), fwhm_mm=3.0)
    assert 0.0 < smoothed[4, 4, 4, 0] < 1.0
    assert np.count_nonzero(smoothed[..., 0]) > 1
    assert np.count_nonzero(smoothed[..., 1]) == 0


def test_mask_normalized_smoothing_preserves_constant_edge_intensity():
    data = np.zeros((9, 9, 9, 2), dtype=np.float32)
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    data[mask, :] = 1.0
    smoothed = _spatial_smooth(data, mask, np.diag([3.0, 3.0, 3.0, 1.0]), fwhm_mm=3.0)
    np.testing.assert_allclose(smoothed[mask, :], 1.0, atol=1e-6)
    assert np.count_nonzero(smoothed[~mask, :]) == 0


def test_bandpass_preserves_resting_frequency_and_rejects_high_frequency():
    tr = 3.0
    frame_times = np.arange(256) * tr
    pass_signal = np.sin(2.0 * np.pi * 0.03 * frame_times)
    stop_signal = np.sin(2.0 * np.pi * 0.14 * frame_times)
    data = (pass_signal + stop_signal).reshape(1, 1, 1, -1).astype(np.float32)
    filtered = _temporal_filter(data, tr, 0.01, 0.10).reshape(-1)
    interior = slice(24, -24)
    pass_projection = abs(np.dot(filtered[interior], pass_signal[interior]))
    stop_projection = abs(np.dot(filtered[interior], stop_signal[interior]))
    assert pass_projection > 10.0 * stop_projection


def test_temporal_filter_preserves_each_voxel_mean():
    tr = 3.0
    frame_times = np.arange(256) * tr
    oscillation = np.sin(2.0 * np.pi * 0.03 * frame_times)
    data = (
        np.stack((100.0 + oscillation, 7.5 + 2.0 * oscillation))
        .reshape(1, 1, 2, -1)
        .astype(np.float32)
    )
    filtered = _temporal_filter(data, tr, 0.01, 0.10)
    np.testing.assert_allclose(
        filtered.mean(axis=-1), data.mean(axis=-1), rtol=0.0, atol=2e-6
    )


def test_recovery_intensity_scale_preserves_nonzero_temporal_means():
    data = np.array([[[[10.0, 11.0, 12.0], [-2.0, 2.0, 3.0]]]], dtype=np.float32)
    mask = np.ones(data.shape[:3], dtype=bool)
    normalized, record = _recovery_intensity_normalize(data, mask)
    assert record["Contract"] == FMRI_INTENSITY_NORMALIZATION_CONTRACT
    assert record["ReportedByPaper"] is False
    assert record["TemporalVoxelZScore"] is False
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0
    assert np.all(normalized.mean(axis=-1)[mask] > 0.0)


def test_recovery_intensity_contract_matches_synthesis_architecture():
    assert (
        synthesis_architecture_contract()["fmri_intensity_contract"]
        == FMRI_INTENSITY_NORMALIZATION_CONTRACT
    )


def test_recovery_intensity_scale_matches_functional_validity_fixture_bitwise():
    data = np.full((2, 2, 2, 128), 2.0, dtype=np.float32)
    data[0, 0, 0, 0] = -4.0
    data[0, 0, 0, 1] = 1.0
    mask = np.ones(data.shape[:3], dtype=bool)
    actual, record = _recovery_intensity_normalize(data, mask)
    expected = np.ones_like(data)
    expected[0, 0, 0, 0] = 0.0
    expected[0, 0, 0, 1] = 0.5
    assert record["Domain"] == FMRI_INTENSITY_NORMALIZATION_DOMAIN
    assert record["FunctionalValidityQuantileBeforeClip"] == 2.0
    assert record["CeilingAndDivisor"] == 2.0
    assert record["FunctionalValidityValueCount"] == data.size
    assert record["FractionNegativeBeforeClipInFunctionalValidity"] == pytest.approx(
        1.0 / data.size
    )
    assert record["FractionClippedAtCeilingInFunctionalValidity"] == 0.0
    assert np.array_equal(actual, expected)


def test_recovery_intensity_scale_is_invariant_to_background_and_padding():
    valid_signal = np.asarray(
        [[-2.0, 2.0, 4.0, 8.0], [1.0, 3.0, 5.0, 9.0]],
        dtype=np.float32,
    )
    compact = np.full((2, 2, 2, 4), 1_000.0, dtype=np.float32)
    compact_mask = np.zeros((2, 2, 2), dtype=bool)
    compact_mask[0, 0, 0] = True
    compact_mask[0, 0, 1] = True
    compact[compact_mask, :] = valid_signal

    padded = np.full((8, 8, 8, 4), -1_000.0, dtype=np.float32)
    padded_mask = np.zeros((8, 8, 8), dtype=bool)
    padded_mask[3, 3, 3] = True
    padded_mask[4, 4, 4] = True
    padded[padded_mask, :] = valid_signal

    compact_output, compact_record = _recovery_intensity_normalize(
        compact, compact_mask
    )
    padded_output, padded_record = _recovery_intensity_normalize(padded, padded_mask)
    assert np.array_equal(
        compact_output[compact_mask, :], padded_output[padded_mask, :]
    )
    assert compact_record["FunctionalValidityQuantileBeforeClip"] == pytest.approx(
        padded_record["FunctionalValidityQuantileBeforeClip"]
    )
    assert compact_record["CeilingAndDivisor"] == pytest.approx(
        padded_record["CeilingAndDivisor"]
    )
    assert np.count_nonzero(compact_output[~compact_mask, :]) == 0
    assert np.count_nonzero(padded_output[~padded_mask, :]) == 0


def test_temporal_filter_rejects_nonfinite_input_instead_of_repairing_it():
    data = np.zeros((1, 1, 1, 32), dtype=np.float32)
    data[..., 4] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        _temporal_filter(data, 3.0, 0.01, 0.10)


def test_source_tr_override_cannot_disagree_with_recorded_evidence():
    image = nib.Nifti1Image(np.zeros((2, 2, 2, 4), dtype=np.float32), np.eye(4))
    image.header.set_zooms((1.0, 1.0, 1.0, 3.0))
    image.header.set_xyzt_units("mm", "sec")
    assert _source_tr_seconds(image, {"RepetitionTime": 3.0}, 3.0) == 3.0
    with pytest.raises(ValueError, match="never override"):
        _source_tr_seconds(image, {"RepetitionTime": 3.0}, 2.0)


def test_t1_production_normalization_cannot_be_disabled(tmp_path):
    with pytest.raises(ValueError, match="requires in-brain z-score"):
        preprocess_t1(
            str(tmp_path / "missing.nii.gz"),
            str(tmp_path / "out.nii.gz"),
            normalize=False,
        )


def test_t1_inbrain_zscore_preserves_exact_pre_normalization_zero_exterior():
    volume = np.zeros((6, 7, 8), dtype=np.float32)
    foreground = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + 10.0
    volume[1:4, 2:6, 1:6] = foreground
    support = volume != 0

    normalized = zscore_inbrain(volume)

    assert normalized.dtype == np.float32
    assert np.array_equal(normalized[~support], np.zeros_like(normalized[~support]))
    assert np.isfinite(normalized).all()
    assert normalized[support].std() == pytest.approx(1.0, abs=1e-5)
    assert np.any(normalized[support] != 0)


def test_t1_zscore_uses_authenticated_segmentation_support_not_intensity_support():
    volume = np.zeros((2, 2, 2), dtype=np.float32)
    volume[0, 0, 1] = 2.0
    volume[0, 1, 0] = 4.0
    segmentation_support = np.zeros_like(volume, dtype=bool)
    segmentation_support[0] = True

    normalized = zscore_inbrain(volume, support=segmentation_support)

    assert np.all(normalized[~segmentation_support] == 0.0)
    assert normalized[0, 0, 0] != 0.0
    assert normalized[segmentation_support].std() == pytest.approx(1.0, abs=1e-5)


def test_official_structural_preprocessing_binds_sources_and_masks_t1(tmp_path):
    scan_id = "sub-01_run-1"
    input_dir = tmp_path / "structural-inputs"
    input_dir.mkdir()
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    t1_values = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    t1_values[1, 0, 0] = 0.0
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[1:] = 2
    t1_source = input_dir / "source_t1.nii.gz"
    segmentation_source = input_dir / "source_seg.nii.gz"
    nib.save(nib.Nifti1Image(t1_values, affine), t1_source)
    nib.save(nib.Nifti1Image(labels, affine), segmentation_source)
    acquisition = write_source_acquisition_fixture(
        tmp_path / "acquisition",
        scan_id=scan_id,
        t1_source=t1_source,
        synthseg_source=segmentation_source,
    )

    manifest_path = tmp_path / "structural_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": STRUCTURAL_MANIFEST_SCHEMA_VERSION,
                "functional_data_used": False,
                "structural_reference": {
                    "path": str(acquisition["raw_t1"]),
                    "sha256": sha256_file(acquisition["raw_t1"]),
                    "modality": "T1w",
                    "cohort_derived": True,
                    "functional_data_used": False,
                    "derivation_method": "synthetic-test-reference",
                },
                "scans": [
                    {
                        "scan_id": scan_id,
                        "t1w_path": str(acquisition["raw_t1"]),
                        "t1w_sha256": sha256_file(acquisition["raw_t1"]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    contract_path = build_common_grid_contract(
        acquisition["raw_t1"],
        manifest_path,
        tmp_path / "common-grid.json",
        patch_multiple=(1, 1, 1),
    )
    contract_sha256 = sha256_file(contract_path)
    segmentation_output = tmp_path / "Masks" / f"{scan_id}_mask.nii.gz"
    t1_output = tmp_path / "T1" / f"{scan_id}_T1.nii.gz"
    segmentation_output.parent.mkdir()
    t1_output.parent.mkdir()

    # Structural preprocessing must remain valid when every BOLD-bearing
    # artifact is absent or unreadable. Only the separate v2 structural
    # authority is admissible on this path.
    acquisition["raw_bold"].unlink()
    acquisition["identity_path"].write_bytes(
        b"sealed BOLD-bearing identity: must never be opened"
    )

    preprocess_seg(
        str(acquisition["raw_synthseg"]),
        str(segmentation_output),
        common_grid_contract_path=str(contract_path),
        common_grid_contract_sha256=contract_sha256,
        structural_source_authority_path=str(acquisition["structural_authority_path"]),
        structural_source_authority_sha256=acquisition["structural_authority_sha256"],
    )
    preprocess_t1(
        str(acquisition["raw_t1"]),
        str(t1_output),
        common_grid_contract_path=str(contract_path),
        common_grid_contract_sha256=contract_sha256,
        segmentation_path=str(segmentation_output),
        structural_source_authority_path=str(acquisition["structural_authority_path"]),
        structural_source_authority_sha256=acquisition["structural_authority_sha256"],
    )

    normalized = nib.load(str(t1_output)).get_fdata(dtype=np.float32)
    conformed_labels = nib.load(str(segmentation_output)).get_fdata(dtype=np.float32)
    support = conformed_labels > 0
    assert np.all(normalized[~support] == 0.0)
    assert normalized[1, 0, 0] != 0.0
    segmentation_metadata = json.loads(
        segmentation_output.with_name(f"{scan_id}_mask.json").read_text()
    )
    t1_metadata = json.loads(t1_output.with_name(f"{scan_id}_T1.json").read_text())
    assert segmentation_metadata["structural_source"] == acquisition["binding"]
    assert t1_metadata["structural_source"] == acquisition["binding"]
    assert t1_metadata["segmentation_output_sha256"] == sha256_file(segmentation_output)


def test_tr_resampling_changes_samples_not_only_header():
    data = np.arange(20, dtype=np.float32).reshape(1, 1, 1, 20)
    resampled = _resample_to_tr(data, source_tr=1.5, target_tr=3.0)
    assert resampled.shape == (1, 1, 1, 10)
    assert not np.array_equal(resampled.reshape(-1), data.reshape(-1)[:10])


def test_short_runs_are_rejected_instead_of_fabricated():
    data = np.array([0.0, 1.0, 2.0], dtype=np.float32).reshape(1, 1, 1, 3)
    with pytest.raises(ValueError, match="temporal padding is forbidden"):
        set_num_frames(data, 8)


def test_mutual_information_coregistration_estimates_non_identity_motion():
    grid = np.indices((24, 24, 24), dtype=np.float32)
    fixed = (
        np.exp(
            -sum((grid[i] - center) ** 2 for i, center in enumerate((8, 11, 13))) / 18.0
        )
        + 0.7
        * np.exp(
            -sum((grid[i] - center) ** 2 for i, center in enumerate((16, 7, 9))) / 10.0
        )
    ).astype(np.float32)
    moving = _apply_rigid(fixed, [0, 0, 0, 2.0, -1.0, 1.0])
    before = _normalised_mutual_information(fixed, moving)
    estimated = _estimate_rigid_mutual_information(fixed, moving, max_iterations=25)
    after = _normalised_mutual_information(fixed, _apply_rigid(moving, estimated))
    assert np.linalg.norm(estimated[3:]) > 0.5
    assert after > before


def test_python_coregistration_is_estimated_and_validated():
    grid = np.indices((20, 20, 20), dtype=np.float32)
    fixed = (
        np.exp(
            -sum((grid[i] - center) ** 2 for i, center in enumerate((7, 10, 11))) / 14.0
        )
        + 0.6
        * np.exp(
            -sum((grid[i] - center) ** 2 for i, center in enumerate((14, 6, 8))) / 8.0
        )
    ).astype(np.float32)
    moving = _apply_rigid(fixed, [0, 0, 0, 2.0, -1.0, 1.0]).astype(np.float32)
    fmri = nib.Nifti1Image(
        np.stack([moving, moving * 1.01, moving * 0.99], axis=-1), np.eye(4)
    )
    t1 = nib.Nifti1Image(fixed, np.eye(4))
    registered, transform, backend, _convention, quality = _coregister_data(
        fmri, t1, "python"
    )
    assert registered.shape == (20, 20, 20, 3)
    assert backend == "SciPy six-DOF mutual information"
    assert not np.allclose(transform, np.eye(4))
    assert quality["CoregistrationValidated"] is True
    assert quality["CoregistrationNMIFinal"] > quality["CoregistrationNMIInitial"]


def test_custom_debug_entrypoint_runs_steps_but_is_not_paper_certified(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    data = np.ones((4, 4, 4, 8), dtype=np.float32)
    fmri = nib.Nifti1Image(data, np.diag([3.0, 3.0, 3.0, 1.0]))
    fmri.header.set_zooms((3.0, 3.0, 3.0, 3.0))
    t1 = nib.Nifti1Image(
        np.ones((4, 4, 4), dtype=np.float32), np.diag([3.0, 3.0, 3.0, 1.0])
    )
    fmri_path = tmp_path / "sub-01_bold.nii.gz"
    t1_path = tmp_path / "sub-01_T1w.nii.gz"
    out_path = tmp_path / "sub-01_bold_128.nii.gz"
    nib.save(fmri, fmri_path)
    nib.save(t1, t1_path)
    (tmp_path / "sub-01_bold.json").write_text(
        json.dumps(
            {
                "RepetitionTime": 3.0,
                "SliceEncodingDirection": "k",
                "SliceTiming": [0.0, 0.75, 1.5, 2.25],
            }
        )
    )

    calls = []

    def coreg(fmri_img, _t1_img, _backend):
        calls.append("coregistration")
        return (
            fmri_img.get_fdata(dtype=np.float32),
            np.eye(4),
            "test MI",
            "test convention",
            {
                "CoregistrationValidated": True,
                "CoregistrationNMIInitial": 1.0,
                "CoregistrationNMIFinal": 1.1,
            },
        )

    def slice_timing(array, **_kwargs):
        calls.append("slice timing")
        return array

    def motion(array, _affine, _header, backend):
        calls.append("motion correction")
        return array, np.zeros((array.shape[-1], 6)), backend

    def smooth(array, *_args):
        calls.append("spatial smoothing")
        return array

    def temporal(array, **_kwargs):
        calls.append("temporal filtering")
        return array

    def conform(image, order, **_kwargs):
        calls.append("harmonisation")
        return image

    monkeypatch.setattr(module, "_coregister_data", coreg)
    monkeypatch.setattr(module, "_slice_timing_correct", slice_timing)
    monkeypatch.setattr(module, "_motion_correct_data", motion)
    monkeypatch.setattr(module, "_spatial_smooth", smooth)
    monkeypatch.setattr(module, "_temporal_filter", temporal)
    monkeypatch.setattr(module, "conform_4d", conform)
    monkeypatch.setattr(module, "_masked_spatial_gradient_energy", lambda *_args: 1.0)
    monkeypatch.setattr(module, "_resample_to_tr", lambda array, *_args: array)
    monkeypatch.setattr(module, "set_num_frames", lambda array, _frames: array)
    monkeypatch.setattr(module, "TARGET_FRAMES", 8)

    module.preprocess_fmri(
        str(fmri_path),
        str(t1_path),
        str(out_path),
        normalize=False,
        preprocessing_backend="custom-debug",
        coregistration_backend="python",
        motion_backend="python",
        allow_synthetic_grid_override=True,
    )
    assert calls == [
        "coregistration",
        "slice timing",
        "motion correction",
        "harmonisation",
        "spatial smoothing",
        "temporal filtering",
    ]
    sidecar = json.loads((tmp_path / "sub-01_bold_128.json").read_text())
    assert sidecar["PaperRequiredStepsComplete"] is False
    assert sidecar["FMRIPrepApplied"] is False
    assert sidecar["PipelineDescription"] == {
        "Name": "CONNECT-4 custom-debug preprocessing",
        "Steps": [
            "debug T1w co-registration",
            "debug slice-timing correction",
            "debug rigid motion correction",
            "debug spatial harmonisation",
            "debug spatial smoothing",
            "debug temporal filtering",
            "debug TR/frame harmonisation",
        ],
    }
    assert sidecar["CoregistrationTransform"] == "sub-01_bold_128_coreg.mat"
    assert sidecar["CoregistrationValidated"] is True
    assert sidecar["MotionParameters"] == "sub-01_bold_128_motion.tsv"


def test_fmriprep_derivative_is_required_for_paper_certification(tmp_path, monkeypatch):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths)
    out_path = tmp_path / "sub-01_fMRI.nii.gz"

    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    grid_arguments = _production_grid_arguments(tmp_path, paths)

    module.preprocess_fmri(
        str(paths["bold"]),
        str(paths["t1"]),
        str(out_path),
        fmriprep_dataset_description_path=str(paths["dataset_description"]),
        fmriprep_execution_record_path=str(paths["record"]),
        motion_confounds_path=str(paths["confounds"]),
        coregistration_transform_path=str(paths["transform"]),
        **_source_identity_arguments(paths),
        **grid_arguments,
    )
    assert calls == ["harmonisation", "spatial smoothing", "temporal filtering"]
    sidecar = json.loads((tmp_path / "sub-01_fMRI.json").read_text())
    assert nib.load(out_path).header.get_xyzt_units() == ("mm", "sec")
    assert sidecar["PaperRequiredStepsComplete"] is True
    assert sidecar["PreprocessingBackend"] == "fmriprep"
    assert sidecar["FMRIPrepGeneratedBy"] == {
        "Name": "fMRIPrep",
        "Version": "24.1.1",
    }
    assert sidecar["CoregistrationEvidenceValidated"] is True
    assert (
        sidecar["VersionedRecoveryChoices"]["PostFMRIPrepTargetInterpolationCount"] == 1
    )
    assert (
        sidecar["VersionedRecoveryChoices"][
            "PostFMRIPrepIntermediateT1GridResamplingApplied"
        ]
        is False
    )
    assert (
        sidecar["VersionedRecoveryChoices"]["SpatialResamplingRoute"]
        == "fMRIPrep T1w-space BOLD grid directly to cohort-common grid"
    )
    assert (
        sidecar["FMRIPrepEvidence"]["EvidenceSchemaVersion"]
        == FMRIPREP_EVIDENCE_SCHEMA_VERSION
    )
    evidence = sidecar["FMRIPrepEvidence"]
    execution = json.loads(paths["record"].read_text())
    for field in (
        "ScanIdentity",
        "FreshDerivativesGeneration",
        "BIDSInputInventory",
        "ExecutableIdentity",
        "RuntimeIdentity",
        "VersionProbeBeforeExecution",
        "VersionProbeAfterExecution",
    ):
        assert evidence[field] == execution[field]
    assert evidence["ScanIdentity"]["ScanID"] == "sub-01_task-rest"
    assert evidence["FreshDerivativesGeneration"]["NoReuse"] is True
    assert evidence["BIDSInputInventory"]["VerifiedAfterExecution"] is True
    assert evidence["RuntimeIdentity"]["VerifiedAfterExecution"] is True
    assert evidence["MotionConfoundsRows"] == 8
    assert evidence["DerivativeSliceTimingCorrected"] is True
    assert evidence["CoregistrationBinding"] == {
        "BOLDRunPrefix": "sub-01_task-rest",
        "TransformFrom": "boldref",
        "TransformTo": "T1w",
        "TransformSemantic": "BOLD-to-subject-T1w registration",
        "T1wSubject": "01",
        "T1wSession": None,
        "BOLDDerivativeSHA256": module._sha256_file(paths["bold"]),
        "BOLDBrainMaskSHA256": module._sha256_file(paths["bold_brain_mask"]),
        "TransformArtifactSHA256": module._sha256_file(paths["transform"]),
        "TargetT1wArtifactSHA256": module._sha256_file(paths["t1"]),
        "RawT1wSHA256": sha256_file(paths["raw_t1"]),
        "RawBOLDSHA256": sha256_file(paths["raw_bold"]),
        "SynthSegMaskSHA256": sha256_file(paths["synthseg_mask"]),
    }
    assert sidecar["CoregistrationTransform"] == str(paths["transform"].resolve())
    assert sidecar["MotionParameters"] == str(paths["confounds"].resolve())
    assert sidecar["CoregistrationTransformSHA256"] == module._sha256_file(
        paths["transform"]
    )
    assert sidecar["MotionParametersSHA256"] == module._sha256_file(paths["confounds"])
    for artifact_name in (
        "ExecutionRecord",
        "DerivativeDatasetDescription",
        "BOLDDerivative",
        "BOLDBrainMask",
        "BOLDMetadata",
        "MotionConfounds",
        "BOLDToT1wTransform",
        "T1wReference",
        "T1wMetadata",
        "StdoutLog",
        "StderrLog",
    ):
        artifact = sidecar["FMRIPrepEvidence"][artifact_name]
        assert set(("Path", "RelativePath", "SHA256", "SizeBytes")) <= set(artifact)


def test_fmriprep_target_is_interpolated_directly_from_bold_to_common_grid(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)

    # A T1w-space derivative need not share the sampled voxel grid of the T1w
    # reference. Make that distinction observable so a reintroduced
    # BOLD->T1-grid->common-grid chain cannot hide behind a no-op fixture.
    t1_image = nib.load(paths["t1"])
    t1_affine = np.asarray(t1_image.affine).copy()
    t1_affine[:3, 3] += 0.4
    nib.save(
        nib.Nifti1Image(
            t1_image.get_fdata(dtype=np.float32), t1_affine, t1_image.header
        ),
        paths["t1"],
    )
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    source_bold_affine = np.asarray(nib.load(paths["bold"]).affine)
    seen_affines = []

    def capture_direct_source(image, order, **_kwargs):
        assert order == 1
        seen_affines.append(np.asarray(image.affine).copy())
        return image

    monkeypatch.setattr(module, "conform_4d", capture_direct_source)
    output = tmp_path / "sub-01_fMRI.nii.gz"
    module.preprocess_fmri(
        str(paths["bold"]),
        str(paths["t1"]),
        str(output),
        fmriprep_dataset_description_path=str(paths["dataset_description"]),
        fmriprep_execution_record_path=str(paths["record"]),
        motion_confounds_path=str(paths["confounds"]),
        coregistration_transform_path=str(paths["transform"]),
        **_source_identity_arguments(paths),
        **grid_arguments,
    )

    assert len(seen_affines) == 1
    np.testing.assert_array_equal(seen_affines[0], source_bold_affine)
    assert not np.array_equal(seen_affines[0], t1_affine)
    sidecar = json.loads(output.with_suffix("").with_suffix(".json").read_text())
    choices = sidecar["VersionedRecoveryChoices"]
    assert choices["PostFMRIPrepTargetInterpolationCount"] == 1
    assert choices["PostFMRIPrepIntermediateT1GridResamplingApplied"] is False
    functional_mask = tmp_path / "sub-01_fMRI_functional_validity_mask.nii.gz"
    assert functional_mask.is_file()
    assert sidecar["FunctionalValidityMask"]["Path"] == str(functional_mask.resolve())
    assert sidecar["FunctionalValidityMask"]["SHA256"] == sha256_file(functional_mask)
    assert sidecar["FunctionalValidityMask"]["SourceBOLDBrainMaskSHA256"] == (
        sha256_file(paths["bold_brain_mask"])
    )
    assert sidecar["AnatomicalMaskRole"] == (
        "structural labels and model conditioning only"
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("legacy_schema", "unsupported schema"),
        ("scan_identity", "another exact scan"),
        ("bids_inventory", "complete BIDS input inventory"),
        ("fresh_generation", "fresh exact-scan generation"),
        ("runtime_binding", "runtime identity differs"),
        ("version_probe", "does not attest the pinned fMRIPrep executable"),
        ("version_probe_hash", "output/hash evidence differs"),
    ],
)
def test_fmriprep_consumer_rejects_resigned_v3_or_incomplete_v4_execution(
    tmp_path, tamper, message
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    record_path = _write_execution_record(paths)
    record = json.loads(record_path.read_text())
    if tamper == "legacy_schema":
        record["SchemaVersion"] = "connect4-fmriprep-execution-v3"
    elif tamper == "scan_identity":
        record["ScanIdentity"]["ScanID"] = "sub-02_task-rest"
    elif tamper == "bids_inventory":
        record["BIDSInputInventory"]["VerifiedAfterExecution"] = False
    elif tamper == "fresh_generation":
        record["FreshDerivativesGeneration"]["NoReuse"] = False
    elif tamper == "runtime_binding":
        record["RuntimeIdentity"]["VerifiedAfterExecution"] = False
    elif tamper == "version_probe":
        record["VersionProbeAfterExecution"]["ObservedVersion"] = "0.0.0"
    elif tamper == "version_probe_hash":
        record["VersionProbeAfterExecution"]["Stdout"] = "forged output\n"
    record_path.write_text(json.dumps(record))

    source_identity, source_evidence = load_source_acquisition_identity(
        paths["source_identity"],
        expected_sha256=paths["source_identity_sha256"],
    )
    with pytest.raises(ValueError, match=message):
        module._validate_fmriprep_evidence(
            fmri_path=str(paths["bold"]),
            t1_path=str(paths["t1"]),
            metadata=json.loads(paths["metadata"].read_text()),
            metadata_path=paths["metadata"],
            dataset_description_path=str(paths["dataset_description"]),
            execution_record_path=str(record_path),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            num_frames=8,
            source_acquisition_identity=source_identity,
            source_acquisition_evidence=source_evidence,
        )


def test_certified_functional_mask_excludes_unobserved_structural_voxels(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    bold_support = np.ones((4, 4, 4), dtype=np.uint8)
    bold_support[0, 0, 0] = 0
    bold_support[3, 3, 3] = 0
    nib.save(
        nib.Nifti1Image(bold_support, np.diag([3.0, 3.0, 3.0, 1.0])),
        paths["bold_brain_mask"],
    )
    _write_execution_record(paths)
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    structural_path = Path(grid_arguments["brain_mask_path"])
    labels = np.resize(np.arange(1, 33, dtype=np.int16), (4, 4, 4))
    nib.save(
        nib.Nifti1Image(labels, np.diag([3.0, 3.0, 3.0, 1.0])),
        structural_path,
    )
    structural_sha256 = sha256_file(structural_path)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    output = tmp_path / "sub-01_fMRI.nii.gz"

    module.preprocess_fmri(
        str(paths["bold"]),
        str(paths["t1"]),
        str(output),
        fmriprep_dataset_description_path=str(paths["dataset_description"]),
        fmriprep_execution_record_path=str(paths["record"]),
        motion_confounds_path=str(paths["confounds"]),
        coregistration_transform_path=str(paths["transform"]),
        **_source_identity_arguments(paths),
        **grid_arguments,
    )

    assert sha256_file(structural_path) == structural_sha256
    assert set(np.unique(nib.load(structural_path).get_fdata()).astype(int)) == set(
        range(1, 33)
    )
    functional_path = tmp_path / "sub-01_fMRI_functional_validity_mask.nii.gz"
    functional = nib.load(functional_path).get_fdata(dtype=np.float32)
    assert set(np.unique(functional)) == {0.0, 1.0}
    assert functional[0, 0, 0] == 0.0
    assert functional[3, 3, 3] == 0.0
    assert np.count_nonzero(functional) == 62
    output_data = nib.load(output).get_fdata(dtype=np.float32)
    assert np.all(output_data[functional == 0.0, :] == 0.0)
    sidecar = json.loads((tmp_path / "sub-01_fMRI.json").read_text())
    mask_evidence = sidecar["FunctionalValidityMask"]
    assert mask_evidence["ValidVoxelCount"] == 62
    assert mask_evidence["StructuralVoxelCount"] == 64
    assert mask_evidence["StructuralVoxelsExcludedFromLoss"] == 2
    assert mask_evidence["CertifiedFMRIPrepCoverage"] is True
    assert mask_evidence["AppliedToSpatialSmoothing"] is True
    assert mask_evidence["AppliedToIntensityNormalization"] is True
    assert mask_evidence["EqualsExactFinalNonzeroSupport"] is True


def test_certified_bold_brain_mask_hash_tampering_is_rejected(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths)
    with paths["bold_brain_mask"].open("ab") as stream:
        stream.write(b"post-wrapper-mask-tamper")

    with pytest.raises(ValueError, match="BOLD brain mask SHA-256 no longer matches"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


def test_certified_functional_mask_rejects_all_zero_voxel_inside_support(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    bold_img = nib.load(paths["bold"])
    bold_data = bold_img.get_fdata(dtype=np.float32)
    bold_data[1, 1, 1, :] = 0.0
    nib.save(
        nib.Nifti1Image(bold_data, bold_img.affine, bold_img.header),
        paths["bold"],
    )
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    grid_arguments = _production_grid_arguments(tmp_path, paths)

    with pytest.raises(
        RuntimeError,
        match="functional-validity mask differs from exact final nonzero fMRI support",
    ):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
            **grid_arguments,
        )


def test_production_rejects_source_run_shorter_than_required_frame_count(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path, frames=7)
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    # The fast fixture uses eight as its stand-in for the paper's 128 frames.
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    with pytest.raises(ValueError, match="at least 8 acquired frames"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
            **grid_arguments,
        )
    assert not (tmp_path / "out.nii.gz").exists()


def test_production_rejects_smoothing_that_crosses_gradient_retention_floor(
    tmp_path, monkeypatch
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    energies = iter((1.0, 0.30))
    monkeypatch.setattr(
        module,
        "_masked_spatial_gradient_energy",
        lambda *_args: next(energies),
    )
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    with pytest.raises(RuntimeError, match="gradient retention gate"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
            **grid_arguments,
        )
    assert not (tmp_path / "out.nii.gz").exists()


@pytest.mark.parametrize(
    "override,message",
    (
        ({"high_pass_hz": 0.02}, "versioned 0.01-Hz high-pass"),
        ({"low_pass_hz": 0.08}, "versioned 0.1-Hz low-pass"),
        ({"high_pass_hz": None}, "versioned 0.01-Hz high-pass"),
        ({"low_pass_hz": None}, "versioned 0.1-Hz low-pass"),
        (
            {"slice_timing_reference": 0.25},
            "forbids slice-timing-reference overrides",
        ),
    ),
)
def test_production_rejects_changed_filter_or_slice_reference(
    tmp_path, override, message
):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    with pytest.raises(ValueError, match=message):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            **_source_identity_arguments(paths),
            **override,
        )
    assert not (tmp_path / "out.nii.gz").exists()


def test_production_rejects_nondefault_slice_reference_in_derivative_metadata(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["SliceTimingReference"] = 0.25
    paths["metadata"].write_text(json.dumps(metadata))
    _write_execution_record(paths)
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    with pytest.raises(
        ValueError,
        match="derivative metadata must fix SliceTimingReference to 0.5",
    ):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
            **grid_arguments,
        )
    assert not (tmp_path / "out.nii.gz").exists()


def test_unverified_bold_cannot_be_certified_as_fmriprep(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    with np.testing.assert_raises_regex(ValueError, "complete wrapper evidence chain"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            **_source_identity_arguments(paths),
        )


def test_fmriprep_transform_hash_tampering_is_rejected(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths)
    paths["transform"].write_text(paths["transform"].read_text() + "tampered\n")
    with pytest.raises(ValueError, match="SHA-256 no longer matches"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


def test_legacy_fmriprep_scanner_to_t1w_transform_is_bound(tmp_path, monkeypatch):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path, transform_source="scanner")
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    output = tmp_path / "out.nii.gz"
    module.preprocess_fmri(
        str(paths["bold"]),
        str(paths["t1"]),
        str(output),
        fmriprep_dataset_description_path=str(paths["dataset_description"]),
        fmriprep_execution_record_path=str(paths["record"]),
        motion_confounds_path=str(paths["confounds"]),
        coregistration_transform_path=str(paths["transform"]),
        **_source_identity_arguments(paths),
        **grid_arguments,
    )
    evidence = json.loads((tmp_path / "out.json").read_text())["FMRIPrepEvidence"]
    assert evidence["CoregistrationBinding"]["TransformFrom"] == "scanner"


def test_fmriprep_command_cannot_ignore_slice_timing(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths, ignored_features=("slicetiming",))
    with pytest.raises(ValueError, match="disabled the required slice-timing"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


def test_fmriprep_derivative_must_confirm_slice_timing_correction(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path, slice_timing_corrected=False)
    _write_execution_record(paths)
    with pytest.raises(ValueError, match="SliceTimingCorrected=true"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


def test_fmriprep_motion_confounds_require_all_six_rigid_parameters(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(
        tmp_path,
        motion_columns=("trans_x", "trans_y", "trans_z", "rot_x", "rot_y"),
    )
    _write_execution_record(paths)
    with pytest.raises(ValueError, match="lacks motion columns"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


def test_fmriprep_wrapper_rejects_prepopulated_derivatives_and_fake_executable(
    tmp_path, monkeypatch
):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o750)
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))
    executed = []
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: executed.append(True),
    )
    with pytest.raises(FileExistsError, match="must not already exist"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(paths["derivatives"]),
            "sub-01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            fmriprep_executable=str(executable),
            extra_args=("--", "--fs-no-reconall"),
        )
    assert executed == []


def test_fmriprep_wrapper_writes_closed_fresh_runtime_bound_success_record(
    tmp_path, monkeypatch
):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "fresh-fmriprep"
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        tmp_path, executable=executable, version="24.1.1"
    )
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))
    observed = {}

    def completed(command, stdout, stderr, **kwargs):
        if command == [str(executable), "--version"]:
            return _version_probe_result()
        observed["command"] = command
        observed["kwargs"] = kwargs
        _publish_fmriprep_fixture(paths, fresh_derivatives)
        stdout.write("successful fMRIPrep test run\n")
        stderr.write("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", completed)
    record_path = Path(
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "sub-01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            **_runtime_identity_arguments(runtime),
            fmriprep_executable=str(executable),
            extra_args=("--", "--fs-no-reconall", "--nthreads=4"),
        )
    )
    record = json.loads(record_path.read_text())
    assert observed["command"][3:] == [
        "participant",
        "--participant-label",
        "01",
        "--output-spaces",
        "T1w",
        "--slice-time-ref",
        "0.5",
        "--level",
        "full",
        "--work-dir",
        str(fresh_derivatives / "connect4-work"),
        "--fs-no-reconall",
        "--nthreads",
        "4",
    ]
    assert record["SchemaVersion"] == wrapper.FMRIPREP_EXECUTION_SCHEMA_VERSION
    assert record["Success"] is True
    assert record["ReturnCode"] == 0
    assert record["OutputSpaces"] == ["T1w"]
    assert record["IgnoredFeatures"] == []
    assert record["SliceTimingEnabledByWrapper"] is True
    assert record["SliceTimingReference"] == 0.5
    assert record["ScanIdentity"]["ScanID"] == "sub-01_task-rest"
    assert record["FreshDerivativesGeneration"]["NoReuse"] is True
    assert (
        record["BIDSInputInventory"]["Identity"]
        == json.loads(paths["source_identity"].read_text())["bids_input_inventory"]
    )
    assert record["ExecutableIdentity"] == runtime["identity"]["executable"]
    assert record["RuntimeIdentity"]["Identity"] == runtime["identity"]
    assert record["VersionProbeBeforeExecution"]["ObservedVersion"] == "24.1.1"
    assert record["VersionProbeAfterExecution"]["ObservedVersion"] == "24.1.1"
    assert record["SourceAcquisitionIdentity"]["VerifiedBeforeExecution"] is True
    assert record["SourceAcquisitionIdentity"]["VerifiedAfterExecution"] is True
    assert (
        record["SourceAcquisitionIdentity"]["Artifact"]["SHA256"]
        == paths["source_identity_sha256"]
    )
    artifact_names = {entry["RelativePath"] for entry in record["Artifacts"]}
    fresh_bold = fresh_derivatives / paths["bold"].relative_to(paths["derivatives"])
    fresh_mask = fresh_derivatives / paths["bold_brain_mask"].relative_to(
        paths["derivatives"]
    )
    fresh_transform = fresh_derivatives / paths["transform"].relative_to(
        paths["derivatives"]
    )
    assert fresh_bold.relative_to(fresh_derivatives).as_posix() in artifact_names
    assert fresh_mask.relative_to(fresh_derivatives).as_posix() in artifact_names
    assert fresh_transform.relative_to(fresh_derivatives).as_posix() in artifact_names
    assert record_path.name == "connect4-fmriprep_sub-01_task-rest_execution.json"
    assert (
        record["StdoutLog"]["SHA256"]
        == _artifact_evidence(
            record_path.with_name("connect4-fmriprep_sub-01_task-rest.stdout.log")
        )["SHA256"]
    )


def test_fmriprep_wrapper_rejects_unpinned_executable_before_generation(
    tmp_path, monkeypatch
):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "fresh-fmriprep"
    executable = tmp_path / "fake-fmriprep"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))
    with pytest.raises(ValueError, match="SHA-pinned runtime identity"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            fmriprep_executable=str(executable),
        )
    assert not fresh_derivatives.exists()


def test_fmriprep_wrapper_rejects_runtime_mutation_during_execution(
    tmp_path, monkeypatch
):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "fresh-fmriprep"
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        tmp_path, executable=executable, version="24.1.1"
    )
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))

    def completed(command, stdout, stderr, **_kwargs):
        if command == [str(executable), "--version"]:
            return _version_probe_result()
        stdout.write("runtime changed after the process started\n")
        stderr.write("")
        runtime["runtime_payload"].write_bytes(b"mutated runtime payload\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", completed)
    with pytest.raises(RuntimeError, match="runtime identity changed"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            **_runtime_identity_arguments(runtime),
            fmriprep_executable=str(executable),
        )
    assert not (
        fresh_derivatives / "logs" / "connect4-fmriprep_sub-01_task-rest_execution.json"
    ).exists()


def test_fmriprep_wrapper_requires_exact_same_run_bold_brain_mask(tmp_path):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    paths["bold_brain_mask"].unlink()
    other_run_mask = paths["bold"].with_name(
        "sub-01_task-other_space-T1w_desc-brain_mask.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(
            np.ones((4, 4, 4), dtype=np.uint8),
            np.diag([3.0, 3.0, 3.0, 1.0]),
        ),
        other_run_mask,
    )
    with pytest.raises(RuntimeError, match="exact-run evidence cardinality"):
        wrapper._required_derivative_artifacts(
            paths["derivatives"],
            "01",
            paths["raw_bold"],
        )


def test_fmriprep_wrapper_rejects_source_change_during_execution(tmp_path, monkeypatch):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "fresh-fmriprep"
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        tmp_path, executable=executable, version="24.1.1"
    )
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))

    def completed(command, stdout, stderr, **_kwargs):
        if command == [str(executable), "--version"]:
            return _version_probe_result()
        stdout.write("fMRIPrep completed before source mutation was detected\n")
        stderr.write("")
        metadata = json.loads(paths["raw_bold_metadata"].read_text())
        metadata["SliceTiming"][0] = 0.125
        paths["raw_bold_metadata"].write_text(json.dumps(metadata))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", completed)
    with pytest.raises(RuntimeError, match="changed during fMRIPrep execution"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            **_runtime_identity_arguments(runtime),
            fmriprep_executable=str(executable),
        )
    assert not (
        fresh_derivatives / "logs" / "connect4-fmriprep_sub-01_task-rest_execution.json"
    ).exists()


def test_fmriprep_wrapper_rejects_source_swap_before_subprocess(tmp_path, monkeypatch):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "fresh-fmriprep"
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("must not execute\n")
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))
    executed = []
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: executed.append(True),
    )
    with paths["raw_bold"].open("ab") as stream:
        stream.write(b"swapped-before-fmriprep")
    with pytest.raises(ValueError, match="identity admission failed"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            fmriprep_executable=str(executable),
        )
    assert executed == []
    assert not fresh_derivatives.exists()


def test_preprocess_fmri_rejects_derivative_that_omits_raw_bold_source(tmp_path):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    paths = _write_fmriprep_derivatives(tmp_path)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["Sources"] = ["bids::sub-01/func/sub-01_task-other_bold.nii.gz"]
    paths["metadata"].write_text(json.dumps(metadata))
    _write_execution_record(paths)
    with pytest.raises(ValueError, match="Sources omit the authenticated raw BOLD"):
        module.preprocess_fmri(
            str(paths["bold"]),
            str(paths["t1"]),
            str(tmp_path / "out.nii.gz"),
            fmriprep_dataset_description_path=str(paths["dataset_description"]),
            fmriprep_execution_record_path=str(paths["record"]),
            motion_confounds_path=str(paths["confounds"]),
            coregistration_transform_path=str(paths["transform"]),
            **_source_identity_arguments(paths),
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--ignore", "slicetiming"),
        ("--ignore=slicetiming",),
        ("--ignore=fieldmaps,slice-timing",),
        ("--ignore", "fieldmaps,slice_timing"),
        ("--output-spaces", "MNI152NLin2009cAsym"),
        ("--participant-label=02",),
        ("--reports-only",),
        ("--boilerplate-only",),
        ("--anat-only",),
        ("--level", "minimal"),
        ("--sloppy",),
        ("--config-file", "/tmp/fmriprep.toml"),
        ("--derivatives", "fmriprep=/tmp/stale"),
        ("--bids-filter-file", "/tmp/filter.json"),
        ("--slice-time-ref", "0"),
        ("--dummy-scans", "5"),
        ("--task-id", "rest"),
        ("--work-dir", "/tmp/reused-work"),
    ],
)
def test_fmriprep_wrapper_rejects_contract_overrides(arguments):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    with pytest.raises(ValueError):
        wrapper._validate_extra_args(arguments)


def test_fmriprep_wrapper_accepts_only_normalized_resource_arguments():
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    assert wrapper._validate_extra_args(
        (
            "--",
            "--fs-no-reconall",
            "--notrack",
            "--nthreads=8",
            "--omp-nthreads",
            "4",
            "--mem-mb",
            "32000",
        )
    ) == [
        "--fs-no-reconall",
        "--notrack",
        "--nthreads",
        "8",
        "--omp-nthreads",
        "4",
        "--mem-mb",
        "32000",
    ]


def test_fmriprep_wrapper_does_not_emit_success_record_on_failure(
    tmp_path, monkeypatch
):
    wrapper = importlib.import_module("preprocessing.run_fmriprep")
    paths = _write_fmriprep_derivatives(tmp_path)
    fresh_derivatives = tmp_path / "failed-fmriprep"
    executable = tmp_path / "fmriprep-bin"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        tmp_path, executable=executable, version="24.1.1"
    )
    monkeypatch.setattr(wrapper.shutil, "which", lambda _name: str(executable))

    def completed(command, **_kwargs):
        if command == [str(executable), "--version"]:
            return _version_probe_result()
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(wrapper.subprocess, "run", completed)
    with pytest.raises(RuntimeError, match="failed with return code 2"):
        wrapper.run_fmriprep(
            str(paths["bids_root"]),
            str(fresh_derivatives),
            "01",
            scan_id="sub-01_task-rest",
            **_source_identity_arguments(paths),
            **_runtime_identity_arguments(runtime),
            fmriprep_executable=str(executable),
        )
    record_path = (
        fresh_derivatives / "logs" / "connect4-fmriprep_sub-01_task-rest_execution.json"
    )
    assert not record_path.exists()


def test_training_contract_rejects_harmonized_fmri_tampering(tmp_path, monkeypatch):
    module = importlib.import_module("preprocessing.preprocess_fmri")
    dataset_module = importlib.import_module("data.dataset_precomputed")
    paths = _write_fmriprep_derivatives(tmp_path)
    _write_execution_record(paths)
    calls = []
    _patch_fast_postprocessing(module, monkeypatch, calls)
    scan_id = "sub-01_task-rest"
    output = tmp_path / f"{scan_id}_fMRI.nii.gz"
    grid_arguments = _production_grid_arguments(tmp_path, paths)
    module.preprocess_fmri(
        str(paths["bold"]),
        str(paths["t1"]),
        str(output),
        fmriprep_dataset_description_path=str(paths["dataset_description"]),
        fmriprep_execution_record_path=str(paths["record"]),
        motion_confounds_path=str(paths["confounds"]),
        coregistration_transform_path=str(paths["transform"]),
        **_source_identity_arguments(paths),
        **grid_arguments,
    )
    (tmp_path / "T1").mkdir()
    (tmp_path / "Masks").mkdir()
    aligned = nib.Nifti1Image(
        np.zeros((4, 4, 4), dtype=np.float32),
        np.diag([3.0, 3.0, 3.0, 1.0]),
    )
    nib.save(aligned, tmp_path / "T1" / f"{scan_id}_T1.nii.gz")
    shutil.copyfile(
        grid_arguments["brain_mask_path"],
        tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    )

    dataset = object.__new__(dataset_module.Connect4PrecomputedDataset)
    dataset.root = tmp_path
    dataset.fmri_dir = tmp_path
    dataset.target_shape = (4, 4, 4)
    dataset.num_frames = 8
    dataset.normalize_intensity = True
    dataset.common_grid_contract = load_common_grid_contract(
        grid_arguments["common_grid_contract_path"],
        expected_sha256=grid_arguments["common_grid_contract_sha256"],
    )
    source_identity, _source_evidence = load_source_acquisition_identity(
        paths["source_identity"],
        expected_sha256=paths["source_identity_sha256"],
    )
    source_binding = structural_source_projection(source_identity)
    dataset.source_dataset = SimpleNamespace(
        source_fingerprint=lambda _scan_id: {
            "structural_source_identity": source_binding
        }
    )
    dataset._validate_preprocessed_target(scan_id)

    with output.open("ab") as stream:
        stream.write(b"post-preprocessing-tamper")
    with pytest.raises(RuntimeError, match="output fMRI SHA-256 no longer matches"):
        dataset._validate_preprocessed_target(scan_id)
