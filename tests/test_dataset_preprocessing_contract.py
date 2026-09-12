import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from data.dataset_precomputed import (
    Connect4PrecomputedDataset,
    TARGET_VALIDITY_MASK_CONTRACT,
    target_validity_mask_from_fmri,
)
from data.protocol import conditioning_identity_from_cache_sources
from data.protocol import EXTERNAL_CACHE_PROTOCOL_SCHEMA
from data.provenance import canonical_sha256, sha256_file
from preprocessing.preprocess_fmri import (
    FMRI_INTENSITY_NORMALIZATION_CONTRACT,
    FMRI_INTENSITY_NORMALIZATION_DOMAIN,
    FMRI_INTENSITY_NORMALIZATION_METHOD,
    FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE,
    FMRI_PREPROCESSING_SCHEMA_VERSION,
    FMRIPREP_EXECUTION_SCHEMA_VERSION,
    FMRIPREP_EVIDENCE_SCHEMA_VERSION,
)
from preprocessing.source_acquisition_identity import (
    SOURCE_ACQUISITION_PURPOSE,
    SOURCE_ACQUISITION_SCHEMA,
    SourceAcquisitionError,
    SYNTHSEG_PROVENANCE_SCHEMA,
    load_source_acquisition_identity,
    snapshot_bids_input_inventory,
    structural_source_projection,
)
from tests.support import (
    write_certified_graph_cache,
    write_fmriprep_runtime_fixture,
    write_source_acquisition_fixture,
    write_structural_sources,
)


def _contract_dataset(tmp_path):
    dataset = object.__new__(Connect4PrecomputedDataset)
    dataset.root = tmp_path
    dataset.fmri_dir = tmp_path
    dataset.target_shape = (4, 5, 6)
    dataset.num_frames = 7
    dataset.normalize_intensity = True
    dataset.common_grid_contract = None
    return dataset


def _write_valid_target(
    tmp_path,
    scan_id="sub-01_run-1",
    *,
    shape=(4, 5, 6),
    num_frames=7,
    write_structural=True,
):
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    if write_structural:
        (tmp_path / "T1").mkdir(exist_ok=True)
        (tmp_path / "Masks").mkdir(exist_ok=True)
        t1 = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
        structural = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
        nib.save(t1, tmp_path / "T1" / f"{scan_id}_T1.nii.gz")
        nib.save(structural, tmp_path / "Masks" / f"{scan_id}_mask.nii.gz")
    structural_values = (
        nib.load(str(tmp_path / "Masks" / f"{scan_id}_mask.nii.gz")).get_fdata(
            dtype=np.float32
        )
        > 0.5
    )
    target_values = np.zeros((*shape, num_frames), dtype=np.float32)
    target_values[structural_values, :] = 0.5
    image = nib.Nifti1Image(target_values, affine)
    image.header.set_xyzt_units(xyz="mm", t="sec")
    image.header.set_zooms((3.0, 3.0, 3.0, 3.0))
    nib.save(image, tmp_path / f"{scan_id}_fMRI.nii.gz")
    functional_validity_path = (
        tmp_path / f"{scan_id}_fMRI_functional_validity_mask.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(structural_values.astype(np.uint8), affine),
        functional_validity_path,
    )

    participant = scan_id.split("_", 1)[0].removeprefix("sub-")
    derivative_root = tmp_path / "fmriprep-fixture"
    func = derivative_root / f"sub-{participant}" / "func"
    anat = derivative_root / f"sub-{participant}" / "anat"
    logs = derivative_root / "logs"
    for folder in (func, anat, logs):
        folder.mkdir(parents=True, exist_ok=True)
    run_prefix = f"sub-{participant}_task-rest"
    bold = func / f"{run_prefix}_space-T1w_desc-preproc_bold.nii.gz"
    bold_brain_mask = func / f"{run_prefix}_space-T1w_desc-brain_mask.nii.gz"
    bold_metadata = func / f"{run_prefix}_space-T1w_desc-preproc_bold.json"
    motion = func / f"{run_prefix}_desc-confounds_timeseries.tsv"
    transform = func / f"{run_prefix}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
    t1_reference = anat / f"sub-{participant}_desc-preproc_T1w.nii.gz"
    t1_metadata = anat / f"sub-{participant}_desc-preproc_T1w.json"
    dataset_description = derivative_root / "dataset_description.json"
    stdout = logs / f"connect4-fmriprep_{scan_id}.stdout.log"
    stderr = logs / f"connect4-fmriprep_{scan_id}.stderr.log"
    execution_record = logs / f"connect4-fmriprep_{scan_id}_execution.json"
    bids_root = tmp_path / "raw-bids"
    raw_anat = bids_root / f"sub-{participant}" / "anat"
    raw_func = bids_root / f"sub-{participant}" / "func"
    raw_anat.mkdir(parents=True, exist_ok=True)
    raw_func.mkdir(parents=True, exist_ok=True)
    raw_t1 = raw_anat / f"sub-{participant}_T1w.nii.gz"
    raw_bold = raw_func / f"sub-{participant}_task-rest_bold.nii.gz"
    synthseg_mask = tmp_path / f"sub-{participant}_synthseg_source.nii.gz"
    synthseg_provenance = tmp_path / f"sub-{participant}_synthseg_provenance.json"
    source_identity_path = tmp_path / f"sub-{participant}_source_acquisition.json"
    nib.save(image, bold)
    nib.save(
        nib.Nifti1Image(structural_values.astype(np.uint8), affine),
        bold_brain_mask,
    )
    bold_metadata.write_text(
        json.dumps(
            {
                "SpatialReference": "T1w",
                "RepetitionTime": 3.0,
                "SliceTimingCorrected": True,
                "StartTime": 0.5,
                "Sources": [
                    f"bids::sub-{participant}/func/"
                    f"sub-{participant}_task-rest_bold.nii.gz"
                ],
            }
        )
    )
    motion.write_text(
        "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n"
        + "\n".join("0\t0\t0\t0\t0\t0" for _ in range(num_frames))
        + "\n"
    )
    transform.write_text("# test BOLD-reference to T1w transform\n")
    source_t1 = nib.load(str(tmp_path / "T1" / f"{scan_id}_T1.nii.gz"))
    nib.save(source_t1, t1_reference)
    nib.save(source_t1, raw_t1)
    nib.save(image, raw_bold)
    (bids_root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "CONNECT-4 certified unit-test BIDS input",
                "BIDSVersion": "1.10.0",
            }
        )
    )
    raw_bold.with_name(raw_bold.name.removesuffix(".nii.gz") + ".json").write_text(
        json.dumps(
            {
                "TaskName": "rest",
                "RepetitionTime": 3.0,
                "SliceTiming": [3.0 * index / shape[2] for index in range(shape[2])],
            }
        )
    )
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.int16), affine),
        synthseg_mask,
    )
    t1_metadata.write_text(
        json.dumps(
            {"Sources": [f"bids::sub-{participant}/anat/sub-{participant}_T1w.nii.gz"]}
        )
    )
    dataset_description.write_text(
        json.dumps(
            {
                "Name": "fMRIPrep outputs",
                "GeneratedBy": [{"Name": "fMRIPrep", "Version": "24.1.1"}],
            }
        )
    )
    stdout.write_text("fMRIPrep completed\n")
    stderr.write_text("")

    def source_artifact(path: Path):
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    producer = {
        "name": "SynthSeg",
        "version": "2.0",
        "source_revision": "a" * 40,
        "model_sha256": "b" * 64,
        "container_image": "docker://freesurfer/synthseg:test",
        "container_manifest_sha256": "c" * 64,
    }
    synthseg_record = {
        "schema": SYNTHSEG_PROVENANCE_SCHEMA,
        "scan_id": scan_id,
        "producer": producer,
        "input_t1": source_artifact(raw_t1),
        "source_t1_sha256": sha256_file(raw_t1),
        "output_mask": source_artifact(synthseg_mask),
    }
    synthseg_record["record_sha256"] = canonical_sha256(synthseg_record)
    synthseg_provenance.write_text(json.dumps(synthseg_record))
    provenance_artifact = source_artifact(synthseg_provenance)
    provenance_artifact["record_sha256"] = synthseg_record["record_sha256"]
    source_identity = {
        "schema": SOURCE_ACQUISITION_SCHEMA,
        "purpose": SOURCE_ACQUISITION_PURPOSE,
        "scan_id": scan_id,
        "participant_label": participant,
        "bids_root": str(bids_root.resolve()),
        "bids_entities": {
            "raw_t1": {"suffix": "T1w", "sub": participant},
            "raw_bold": {
                "suffix": "bold",
                "sub": participant,
                "task": "rest",
            },
        },
        "bids_input_inventory": snapshot_bids_input_inventory(
            bids_root,
            participant=participant,
            raw_t1=raw_t1,
            raw_bold=raw_bold,
        ),
        "raw_t1": source_artifact(raw_t1),
        "raw_bold": source_artifact(raw_bold),
        "synthseg": {
            "mask": source_artifact(synthseg_mask),
            "provenance": provenance_artifact,
            "producer": producer,
        },
    }
    source_identity["record_sha256"] = canonical_sha256(source_identity)
    source_identity_path.write_text(json.dumps(source_identity))
    source_identity_artifact = {
        "Path": str(source_identity_path.resolve()),
        "SHA256": sha256_file(source_identity_path),
        "SizeBytes": source_identity_path.stat().st_size,
        "RecordSHA256": source_identity["record_sha256"],
    }
    source_binding = {
        "Artifact": source_identity_artifact,
        "Identity": source_identity,
        "VerifiedBeforeExecution": True,
        "VerifiedAfterExecution": True,
    }

    def artifact(path: Path):
        return {
            "Path": str(path.resolve()),
            "RelativePath": path.relative_to(derivative_root).as_posix(),
            "SHA256": sha256_file(path),
            "SizeBytes": path.stat().st_size,
        }

    executable = tmp_path / "fmriprep-consumer-fixture"
    executable.write_text("#!/bin/sh\necho 'fMRIPrep v24.1.1'\n")
    executable.chmod(0o750)
    runtime = write_fmriprep_runtime_fixture(
        tmp_path,
        executable=executable,
        version="24.1.1",
    )
    runtime_identity = runtime["identity"]
    runtime_binding = {
        "Artifact": {
            "Path": str(runtime["path"]),
            "SHA256": runtime["sha256"],
            "SizeBytes": runtime["path"].stat().st_size,
            "RecordSHA256": runtime_identity["record_sha256"],
        },
        "Identity": runtime_identity,
        "VerifiedBeforeExecution": True,
        "VerifiedAfterExecution": True,
    }
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
    input_inventory_binding = {
        "Identity": source_identity["bids_input_inventory"],
        "VerifiedBeforeExecution": True,
        "VerifiedAfterExecution": True,
    }
    scan_identity = {
        "ScanID": scan_id,
        "ParticipantLabel": participant,
        "Session": None,
        "Task": "rest",
        "Run": None,
        "Acquisition": None,
        "Direction": None,
        "Echo": None,
        "RawBOLDRelativePath": source_identity["bids_input_inventory"][
            "raw_bold_relative_path"
        ],
        "RawT1wRelativePath": source_identity["bids_input_inventory"][
            "raw_t1_relative_path"
        ],
    }
    fresh_generation = {
        "GenerationID": "d" * 64,
        "ScanID": scan_id,
        "Path": str(derivative_root.resolve()),
        "PathAbsentBeforeCreation": True,
        "CreatedExclusively": True,
        "InitiallyEmpty": True,
        "NoReuse": True,
    }
    command = [
        str(executable.resolve()),
        str(bids_root.resolve()),
        str(derivative_root.resolve()),
        "participant",
        "--participant-label",
        participant,
        "--output-spaces",
        "T1w",
        "--slice-time-ref",
        "0.5",
        "--level",
        "full",
        "--work-dir",
        str(derivative_root.resolve() / "connect4-work"),
    ]
    artifact_paths = (
        dataset_description,
        bold,
        bold_brain_mask,
        bold_metadata,
        motion,
        transform,
        t1_reference,
        t1_metadata,
        stdout,
        stderr,
    )
    execution_record.write_text(
        json.dumps(
            {
                "SchemaVersion": FMRIPREP_EXECUTION_SCHEMA_VERSION,
                "Wrapper": "preprocessing.run_fmriprep",
                "Success": True,
                "ReturnCode": 0,
                "StartedAtUTC": "2026-08-29T10:00:00Z",
                "CompletedAtUTC": "2026-08-29T10:01:00Z",
                "Command": command,
                "BIDSRoot": str(bids_root.resolve()),
                "DerivativesRoot": str(derivative_root.resolve()),
                "WorkRoot": str(derivative_root.resolve() / "connect4-work"),
                "ScanIdentity": scan_identity,
                "ParticipantLabel": participant,
                "OutputSpaces": ["T1w"],
                "ExtraArguments": [],
                "IgnoredFeatures": [],
                "SliceTimingEnabledByWrapper": True,
                "SliceTimingReference": 0.5,
                "FMRIPrepGeneratedBy": {
                    "Name": "fMRIPrep",
                    "Version": "24.1.1",
                },
                "FreshDerivativesGeneration": fresh_generation,
                "BIDSInputInventory": input_inventory_binding,
                "ExecutableIdentity": runtime_identity["executable"],
                "RuntimeIdentity": runtime_binding,
                "VersionProbeBeforeExecution": version_probe,
                "VersionProbeAfterExecution": version_probe,
                "SourceAcquisitionIdentity": source_binding,
                "StdoutLog": artifact(stdout),
                "StderrLog": artifact(stderr),
                "Artifacts": [artifact(path) for path in artifact_paths],
            }
        )
    )

    artifacts = {
        "ExecutionRecord": artifact(execution_record),
        "DerivativeDatasetDescription": artifact(dataset_description),
        "BOLDDerivative": artifact(bold),
        "BOLDBrainMask": artifact(bold_brain_mask),
        "BOLDMetadata": artifact(bold_metadata),
        "MotionConfounds": artifact(motion),
        "BOLDToT1wTransform": artifact(transform),
        "T1wReference": artifact(t1_reference),
        "T1wMetadata": artifact(t1_metadata),
        "StdoutLog": artifact(stdout),
        "StderrLog": artifact(stderr),
    }
    evidence = {
        "EvidenceSchemaVersion": FMRIPREP_EVIDENCE_SCHEMA_VERSION,
        "ParticipantLabel": participant,
        "ExecutionWrapper": "preprocessing.run_fmriprep",
        "ExecutionSuccess": True,
        "ExecutionReturnCode": 0,
        "ExecutionStartedAtUTC": "2026-08-29T10:00:00Z",
        "ExecutionCompletedAtUTC": "2026-08-29T10:01:00Z",
        "ScanIdentity": scan_identity,
        "OutputSpaces": ["T1w"],
        "ExtraArguments": [],
        "IgnoredFeatures": [],
        "SliceTimingEnabledByWrapper": True,
        "SliceTimingReference": 0.5,
        "DerivativeSliceTimingCorrected": True,
        "MotionConfoundsRows": num_frames,
        "FreshDerivativesGeneration": fresh_generation,
        "BIDSInputInventory": input_inventory_binding,
        "ExecutableIdentity": runtime_identity["executable"],
        "RuntimeIdentity": runtime_binding,
        "VersionProbeBeforeExecution": version_probe,
        "VersionProbeAfterExecution": version_probe,
        "CoregistrationBinding": {
            "BOLDRunPrefix": run_prefix,
            "TransformFrom": "boldref",
            "TransformTo": "T1w",
            "TransformSemantic": "BOLD-to-subject-T1w registration",
            "T1wSubject": participant,
            "T1wSession": None,
            "BOLDDerivativeSHA256": artifacts["BOLDDerivative"]["SHA256"],
            "BOLDBrainMaskSHA256": artifacts["BOLDBrainMask"]["SHA256"],
            "TransformArtifactSHA256": artifacts["BOLDToT1wTransform"]["SHA256"],
            "TargetT1wArtifactSHA256": artifacts["T1wReference"]["SHA256"],
            "RawT1wSHA256": source_identity["raw_t1"]["sha256"],
            "RawBOLDSHA256": source_identity["raw_bold"]["sha256"],
            "SynthSegMaskSHA256": source_identity["synthseg"]["mask"]["sha256"],
        },
        "SourceAcquisitionIdentity": source_identity_artifact,
        "SourceAcquisition": source_identity,
        **artifacts,
    }
    metadata = {
        "PreprocessingSchemaVersion": FMRI_PREPROCESSING_SCHEMA_VERSION,
        "PaperRequiredStepsComplete": True,
        "ManuscriptClaims": {
            "Frames": 128,
            "RepetitionTimeSeconds": 3.0,
            "VoxelSizeMM": [3.0, 3.0, 3.0],
            "SpatialMatrix": None,
            "SmoothingFWHMMM": None,
            "TemporalFilterCutoffsHz": None,
            "IntensityNormalization": None,
        },
        "VersionedRecoveryChoices": {
            "SpatialGrid": "synthetic-debug",
            "SpatialSmoothingFWHMMM": 3.0,
            "SpatialSmoothingMethod": (
                "Gaussian signal/mask division within functional-validity mask"
            ),
            "MinimumGradientRetentionRatioExclusive": 0.30,
            "TemporalHighPassHz": 0.01,
            "TemporalLowPassHz": 0.10,
            "TemporalFilterPreservesVoxelMean": True,
            "IntensityNormalizationContract": (FMRI_INTENSITY_NORMALIZATION_CONTRACT),
            "PostFMRIPrepTargetInterpolationCount": 1,
            "PostFMRIPrepIntermediateT1GridResamplingApplied": False,
            "PostFMRIPrepSpatialInterpolationOrder": 1,
            "SpatialResamplingRoute": (
                "fMRIPrep T1w-space BOLD grid directly to cohort-common grid"
            ),
        },
        "PreprocessingBackend": "fmriprep",
        "FMRIPrepApplied": True,
        "FMRIPrepGeneratedBy": {"Name": "fMRIPrep", "Version": "24.1.1"},
        "FMRIPrepEvidence": evidence,
        "PipelineDescription": {
            "Name": "fMRIPrep + CONNECT-4 harmonisation",
            "Steps": [
                "T1w co-registration",
                "slice-timing correction",
                "rigid motion correction",
                "single-pass spatial harmonisation",
                "spatial smoothing",
                "temporal filtering",
                "TR/frame harmonisation",
                "recovery intensity normalization",
            ],
        },
        "CoregistrationBackend": "fMRIPrep BOLD-to-T1w co-registration",
        "CoregistrationValidated": True,
        "CoregistrationEvidenceValidated": True,
        "CoregistrationTransform": artifacts["BOLDToT1wTransform"]["Path"],
        "CoregistrationTransformSHA256": artifacts["BOLDToT1wTransform"]["SHA256"],
        "SliceTimingApplied": True,
        "MotionCorrectionApplied": True,
        "MotionParameters": artifacts["MotionConfounds"]["Path"],
        "MotionParametersSHA256": artifacts["MotionConfounds"]["SHA256"],
        "SpatialSmoothingApplied": True,
        "SpatialSmoothingFWHM": 3.0,
        "SpatialSmoothingMethod": (
            "Gaussian signal/mask division within functional-validity mask"
        ),
        "SmoothingGradientRetentionRatio": 0.75,
        "AnatomicalMaskSHA256": sha256_file(
            tmp_path / "Masks" / f"{scan_id}_mask.nii.gz"
        ),
        "FunctionalValidityMask": {
            "Path": str(functional_validity_path.resolve()),
            "SHA256": sha256_file(functional_validity_path),
            "SizeBytes": functional_validity_path.stat().st_size,
            "Role": "fMRI spatial support and model-loss validity",
            "Derivation": (
                "nearest-neighbour common-grid fMRIPrep BOLD brain mask "
                "intersected with nonzero structural-mask support"
            ),
            "CertifiedFMRIPrepCoverage": True,
            "SourceBOLDBrainMaskSHA256": artifacts["BOLDBrainMask"]["SHA256"],
            "Shape": list(shape),
            "ValidVoxelCount": int(structural_values.sum()),
            "StructuralVoxelCount": int(structural_values.sum()),
            "BOLDSupportVoxelCount": int(structural_values.sum()),
            "StructuralVoxelsExcludedFromLoss": 0,
            "AppliedToSpatialSmoothing": True,
            "AppliedToIntensityNormalization": True,
            "OutsideMaskForcedZero": True,
            "EqualsExactFinalNonzeroSupport": True,
        },
        "TemporalFilteringApplied": True,
        "TemporalFilterPreservesVoxelMean": True,
        "TemporalHighPassHz": 0.01,
        "TemporalLowPassHz": 0.10,
        "SliceTimingReference": 0.5,
        "TemporalPaddingApplied": False,
        "OutputShape": [*shape, num_frames],
        "RepetitionTime": 3.0,
        "VoxelSize": [3.0, 3.0, 3.0],
        "TemporalZScore": False,
        "IntensityNormalizationApplied": True,
        "IntensityNormalization": {
            "Contract": FMRI_INTENSITY_NORMALIZATION_CONTRACT,
            "ReportedByPaper": False,
            "Method": FMRI_INTENSITY_NORMALIZATION_METHOD,
            "Domain": FMRI_INTENSITY_NORMALIZATION_DOMAIN,
            "UpperQuantile": FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE,
            "FunctionalValidityQuantileBeforeClip": 0.5,
            "CeilingAndDivisor": 1.0,
            "MinimumDivisor": 1.0,
            "FunctionalValidityValueCount": int(structural_values.sum()) * num_frames,
            "FractionNegativeBeforeClipInFunctionalValidity": 0.0,
            "FractionClippedAtCeilingInFunctionalValidity": 0.0,
            "LowerQuantileSubtraction": False,
            "TemporalVoxelZScore": False,
            "OutsideMaskForcedZero": True,
            "OutputMinimum": 0.5,
            "OutputMaximum": 0.5,
        },
        "SourceMetadata": artifacts["BOLDMetadata"]["Path"],
        "OutputSHA256": sha256_file(tmp_path / f"{scan_id}_fMRI.nii.gz"),
    }
    sidecar = tmp_path / f"{scan_id}_fMRI.json"
    sidecar.write_text(json.dumps(metadata))
    return sidecar, metadata


def test_dataset_accepts_certified_paper_preprocessing(tmp_path):
    dataset = _contract_dataset(tmp_path)
    _write_valid_target(tmp_path)
    dataset._validate_preprocessed_target("sub-01_run-1")


@pytest.mark.parametrize(
    "field,value,message",
    (
        (
            "PostFMRIPrepTargetInterpolationCount",
            2,
            "spatially interpolated exactly once",
        ),
        (
            "PostFMRIPrepIntermediateT1GridResamplingApplied",
            True,
            "forbidden intermediate T1-grid resampling",
        ),
        (
            "PostFMRIPrepSpatialInterpolationOrder",
            3,
            "spatial interpolation order differs",
        ),
        (
            "SpatialResamplingRoute",
            "BOLD to T1 grid to cohort-common grid",
            "spatial resampling route differs",
        ),
    ),
)
def test_dataset_rejects_texture_erasing_spatial_resampling_contract(
    tmp_path, field, value, message
):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    metadata["VersionedRecoveryChoices"][field] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match=message):
        dataset._validate_preprocessed_target("sub-01_run-1")


@pytest.mark.parametrize(
    "scope,field,value,message",
    (
        (
            "VersionedRecoveryChoices",
            "TemporalHighPassHz",
            0.02,
            "recovery temporal high-pass cutoff differs",
        ),
        (
            "VersionedRecoveryChoices",
            "TemporalLowPassHz",
            0.08,
            "recovery temporal low-pass cutoff differs",
        ),
        (None, "TemporalHighPassHz", 0.02, "top-level temporal high-pass cutoff"),
        (None, "TemporalLowPassHz", 0.08, "top-level temporal low-pass cutoff"),
        (None, "SliceTimingReference", 0.25, "slice-timing reference differs"),
    ),
)
def test_dataset_rejects_changed_recovery_filter_or_slice_reference(
    tmp_path, scope, field, value, message
):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    destination = metadata if scope is None else metadata[scope]
    destination[field] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match=message):
        dataset._validate_preprocessed_target("sub-01_run-1")


@pytest.mark.parametrize(
    "tamper",
    (
        "legacy_evidence",
        "scan_identity",
        "bids_inventory",
        "fresh",
        "runtime",
        "version_probe_hash",
    ),
)
def test_dataset_rejects_incomplete_fmriprep_v4_evidence(tmp_path, tamper):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    evidence = metadata["FMRIPrepEvidence"]
    if tamper == "legacy_evidence":
        evidence["EvidenceSchemaVersion"] = "connect4-fmriprep-evidence-v3"
    elif tamper == "scan_identity":
        evidence["ScanIdentity"]["ScanID"] = "sub-02_run-1"
    elif tamper == "bids_inventory":
        evidence["BIDSInputInventory"]["VerifiedAfterExecution"] = False
    elif tamper == "fresh":
        evidence["FreshDerivativesGeneration"]["NoReuse"] = False
    elif tamper == "runtime":
        evidence["RuntimeIdentity"]["VerifiedAfterExecution"] = False
    elif tamper == "version_probe_hash":
        evidence["VersionProbeAfterExecution"]["Stdout"] = "forged output\n"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_runtime_artifact_changed_after_preprocessing(tmp_path):
    dataset = _contract_dataset(tmp_path)
    _sidecar, metadata = _write_valid_target(tmp_path)
    runtime = metadata["FMRIPrepEvidence"]["RuntimeIdentity"]["Identity"]
    executable_path = runtime["executable"]["path"]
    payload_path = next(
        Path(artifact["path"])
        for artifact in runtime["runtime_artifacts"]
        if artifact["path"] != executable_path
    )
    with payload_path.open("ab") as stream:
        stream.write(b"changed-after-certified-preprocessing")
    with pytest.raises(RuntimeError, match="runtime evidence"):
        dataset._validate_preprocessed_target("sub-01_run-1")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("Domain", "whole 4D volume including zero background"),
        ("FunctionalValidityValueCount", 1),
        ("FractionClippedAtCeilingInFunctionalValidity", float("nan")),
    ],
)
def test_dataset_rejects_invalid_functional_validity_normalization_evidence(
    tmp_path, field, replacement
):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    metadata["IntensityNormalization"][field] = replacement
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="intensity-normalization evidence differs"):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_legacy_whole_volume_normalization_record(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    normalization = metadata["IntensityNormalization"]
    normalization["Contract"] = "connect4-unpublished-v54b-q995-nonnegative-v1"
    normalization["WholeVolumeQuantileBeforeClip"] = normalization.pop(
        "FunctionalValidityQuantileBeforeClip"
    )
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="intensity-normalization evidence differs"):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_raw_bold_changed_after_certification(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    raw_bold = Path(
        metadata["FMRIPrepEvidence"]["SourceAcquisition"]["raw_bold"]["path"]
    )
    with raw_bold.open("ab") as stream:
        stream.write(b"post-certification-source-swap")
    with pytest.raises(RuntimeError, match="source-acquisition identity"):
        dataset._validate_preprocessed_target("sub-01_run-1")
    assert sidecar.exists()


def test_dataset_rejects_forged_execution_source_binding(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    evidence = metadata["FMRIPrepEvidence"]
    execution_path = Path(evidence["ExecutionRecord"]["Path"])
    execution = json.loads(execution_path.read_text())
    execution["SourceAcquisitionIdentity"]["Identity"]["scan_id"] = "sub-02"
    execution_path.write_text(json.dumps(execution))
    evidence["ExecutionRecord"]["SHA256"] = sha256_file(execution_path)
    evidence["ExecutionRecord"]["SizeBytes"] = execution_path.stat().st_size
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(
        RuntimeError,
        match="execution record source-acquisition binding differs",
    ):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_derivative_sources_substitution(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    evidence = metadata["FMRIPrepEvidence"]
    bold_metadata_path = Path(evidence["BOLDMetadata"]["Path"])
    bold_metadata = json.loads(bold_metadata_path.read_text())
    bold_metadata["Sources"] = ["bids::sub-01/func/sub-01_task-other_bold.nii.gz"]
    bold_metadata_path.write_text(json.dumps(bold_metadata))
    evidence["BOLDMetadata"]["SHA256"] = sha256_file(bold_metadata_path)
    evidence["BOLDMetadata"]["SizeBytes"] = bold_metadata_path.stat().st_size
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(
        RuntimeError,
        match="BOLD derivative Sources omit authenticated raw BOLD",
    ):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_temporal_zscore_or_out_of_range_target(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    metadata["TemporalZScore"] = True
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="forbidden temporal voxel z-score"):
        dataset._validate_preprocessed_target("sub-01_run-1")

    metadata["TemporalZScore"] = False
    image_path = tmp_path / "sub-01_run-1_fMRI.nii.gz"
    image = nib.load(image_path)
    invalid = image.get_fdata(dtype=np.float32)
    invalid[0, 0, 0, 0] = 1.5
    nib.save(nib.Nifti1Image(invalid, image.affine, image.header), image_path)
    metadata["OutputSHA256"] = sha256_file(image_path)
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match=r"outside the certified \[0,1\] range"):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_three_milliseconds_masquerading_as_three_seconds(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    image_path = tmp_path / "sub-01_run-1_fMRI.nii.gz"
    image = nib.load(image_path)
    image.header.set_xyzt_units(xyz="mm", t="msec")
    # pixdim[4] remains numerically 3.0; without checking units this is an
    # actual 0.003-second TR that looks like the required 3-second value.
    nib.save(image, image_path)
    metadata["OutputSHA256"] = sha256_file(image_path)
    sidecar.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="expected \\('mm', 'sec'\\)"):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_dataset_rejects_legacy_target_missing_required_step(tmp_path):
    dataset = _contract_dataset(tmp_path)
    sidecar, metadata = _write_valid_target(tmp_path)
    metadata["SliceTimingApplied"] = False
    metadata["PaperRequiredStepsComplete"] = False
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="slice-timing correction missing"):
        dataset._validate_preprocessed_target("sub-01_run-1")


def test_patch_roi_coverage_is_absolute_not_renormalized():
    dataset = object.__new__(Connect4PrecomputedDataset)
    dataset.structure_to_roi_idx = {2: 0, 3: 1}

    # The unlisted 60% may be background or an unmapped SynthSeg structure.
    # The stored 25%/15% values must remain the hyperedge coefficients.
    distributions = [{2: 0.25, 3: 0.15}, {}]
    dataset._validate_patch_distributions("scan", distributions, expected_patches=2)


@pytest.mark.parametrize(
    "distribution, message",
    [
        ({2: -0.1}, "Invalid ROI coverage"),
        ({2: float("nan")}, "Invalid ROI coverage"),
        ({2: 0.7, 3: 0.4}, "exceeds the full patch fraction"),
    ],
)
def test_patch_roi_coverage_rejects_invalid_fractions(distribution, message):
    dataset = object.__new__(Connect4PrecomputedDataset)
    dataset.structure_to_roi_idx = {2: 0, 3: 1}
    with pytest.raises(ValueError, match=message):
        dataset._validate_patch_distributions(
            "scan", [distribution], expected_patches=1
        )


def test_target_validity_uses_observed_support_not_temporal_variance():
    fmri = np.zeros((2, 2, 2, 4), dtype=np.float32)
    fmri[0, 0, 0] = 0.25  # constant non-zero is valid observed signal
    fmri[1, 1, 1, 2] = 0.5
    structural = np.ones((2, 2, 2), dtype=np.float32)
    validity = target_validity_mask_from_fmri(fmri, structural, scan_id="scan")
    assert validity.dtype == np.float32
    assert validity[0, 0, 0] == 1
    assert validity[1, 1, 1] == 1
    assert validity.sum() == 2


def test_target_validity_rejects_observed_signal_outside_structure():
    fmri = np.zeros((2, 2, 2, 4), dtype=np.float32)
    fmri[0, 0, 0, 0] = 1
    structural = np.zeros((2, 2, 2), dtype=np.float32)
    structural[1, 1, 1] = 1
    with pytest.raises(ValueError, match="outside the structural brain mask"):
        target_validity_mask_from_fmri(fmri, structural, scan_id="scan")


def test_certified_dataset_loads_without_resampling_or_subject_substitution(tmp_path):
    scan_id = "sub-01_run-1"
    sources = write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=(4, 4, 4), radiomics_dim=3
    )
    _write_valid_target(
        tmp_path,
        scan_id,
        shape=(4, 4, 4),
        num_frames=128,
        write_structural=False,
    )
    precomputed = write_certified_graph_cache(tmp_path, sources, scan_id)

    dataset = Connect4PrecomputedDataset(
        root_dir=str(tmp_path),
        precomputed_dir=str(precomputed),
        target_shape=(4, 4, 4),
        patch_size=(2, 4, 4),
        num_frames=128,
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
        brainiac_model_path=str(sources["brainiac"]),
        brainiac_checkpoint_sha256=sources["brainiac_fingerprint"]["checkpoint_sha256"],
        brainiac_source_sha256=sources["brainiac_fingerprint"][
            "source_files_fingerprint_sha256"
        ],
        modernbert_model_name=str(sources["modernbert"]),
        modernbert_revision="a" * 40,
        roi_node_dim=515,
        fmri_dir=str(tmp_path),
        allow_synthetic_grid_override=True,
    )
    sample = dataset[0]
    assert sample["fmri"].shape == (1, 128, 4, 4, 4)
    assert sample["t1w"].shape == (1, 4, 4, 4)
    assert sample["brain_mask"].shape == (1, 4, 4, 4)
    assert sample["target_validity_mask"].shape == (1, 4, 4, 4)
    assert torch.equal(sample["target_validity_mask"], sample["brain_mask"])
    assert sample["target_validity_mask_contract"] == (TARGET_VALIDITY_MASK_CONTRACT)
    assert len(sample["roi_masks"]) == 32
    assert sample["hyperedge_weights"].tolist() == pytest.approx([1.0, 1.0])


def test_exact_target_allowlist_never_touches_sealed_fmri_or_sidecar(
    tmp_path, monkeypatch
):
    scan_id = "sub-sealed_run-1"
    sources = write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=(4, 4, 4), radiomics_dim=3
    )
    precomputed = write_certified_graph_cache(tmp_path, sources, scan_id)
    fmri_dir = tmp_path / "sealed-fmri"
    fmri_dir.mkdir()

    def build():
        return Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(precomputed),
            target_shape=(4, 4, 4),
            patch_size=(2, 4, 4),
            num_frames=128,
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
            brainiac_model_path=str(sources["brainiac"]),
            brainiac_checkpoint_sha256=sources["brainiac_fingerprint"][
                "checkpoint_sha256"
            ],
            brainiac_source_sha256=sources["brainiac_fingerprint"][
                "source_files_fingerprint_sha256"
            ],
            modernbert_model_name=str(sources["modernbert"]),
            modernbert_revision="a" * 40,
            roi_node_dim=515,
            fmri_dir=str(fmri_dir),
            target_scan_ids=[],
            allow_synthetic_grid_override=True,
        )

    absent = build()[0]
    assert "fmri" not in absent and "fmri_mean" not in absent
    assert "target_validity_mask" not in absent

    # Both files are deliberately invalid. Any stat/open/NIfTI load of either
    # would fail construction or __getitem__, so successful loading proves the
    # sealed scan is outside the target-access path.
    (fmri_dir / f"{scan_id}_fMRI.nii.gz").write_bytes(b"sealed sentinel")
    (fmri_dir / f"{scan_id}_fMRI.json").write_bytes(b"sealed sentinel")
    original_stat = Path.stat
    original_open = Path.open

    def guarded_stat(path, *args, **kwargs):
        if path.parent == fmri_dir:
            raise AssertionError(f"sealed target was stat'ed: {path}")
        return original_stat(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if path.parent == fmri_dir:
            raise AssertionError(f"sealed target was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    sentinel = build()[0]
    assert "fmri" not in sentinel and "fmri_mean" not in sentinel
    assert "target_validity_mask" not in sentinel


def test_structural_and_fmri_source_acquisition_must_match_exactly(tmp_path):
    scan_id = "sub-join_run-1"
    write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=(4, 4, 4), radiomics_dim=3
    )
    fixture = write_source_acquisition_fixture(
        tmp_path,
        scan_id=scan_id,
        t1_source=tmp_path / "T1" / f"{scan_id}_T1.nii.gz",
        synthseg_source=tmp_path / "Masks" / f"{scan_id}_mask.nii.gz",
    )
    identity, evidence = load_source_acquisition_identity(
        fixture["identity_path"], expected_sha256=fixture["identity_sha256"]
    )
    admitted = (
        Connect4PrecomputedDataset._require_matching_structural_fmri_source_identity(
            fixture["binding"], identity, evidence
        )
    )
    assert admitted == structural_source_projection(fixture["binding"])

    # The target-side raw BOLD is authenticated by its own full identity, but
    # is intentionally absent from the structural join.
    alternate_bold = dict(identity)
    alternate_bold["raw_bold"] = dict(identity["raw_bold"])
    alternate_bold["raw_bold"]["sha256"] = "0" * 64
    assert Connect4PrecomputedDataset._require_matching_structural_fmri_source_identity(
        fixture["binding"], alternate_bold, evidence
    ) == structural_source_projection(fixture["binding"])

    mismatched = dict(fixture["binding"])
    mismatched["raw_t1_sha256"] = "0" * 64
    with pytest.raises(
        SourceAcquisitionError,
        match="structural cache and fMRI structural-source projections differ",
    ):
        Connect4PrecomputedDataset._require_matching_structural_fmri_source_identity(
            mismatched, identity, evidence
        )


def test_structural_only_dataset_never_requires_or_returns_fmri(tmp_path):
    scan_id = "sub-external_run-1"
    sources = write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=(4, 4, 4), radiomics_dim=3
    )
    precomputed = write_certified_graph_cache(tmp_path, sources, scan_id)
    external_context = {
        "format": EXTERNAL_CACHE_PROTOCOL_SCHEMA,
        "external_cohort_manifest_sha256": "a" * 64,
        "synthesis_checkpoint_sha256": "b" * 64,
        "synthesis_split_sha256": "c" * 64,
        "synthesis_exclusion_sha256": "d" * 64,
        "synthesis_common_grid_contract_sha256": "f" * 64,
        "num_scans": 1,
        "num_patients": 1,
        "external_cohorts": ["IDEAS"],
    }
    metadata_path = precomputed / "hypergraphs" / f"{scan_id}_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    external_context["conditioning_identity"] = (
        conditioning_identity_from_cache_sources(metadata["source_fingerprint"])
    )
    external_context["conditioning_identity_sha256"] = canonical_sha256(
        external_context["conditioning_identity"]
    )
    metadata["source_fingerprint"]["external_protocol"] = external_context
    metadata["source_fingerprint_sha256"] = canonical_sha256(
        metadata["source_fingerprint"]
    )
    metadata_path.write_text(json.dumps(metadata))

    dataset = Connect4PrecomputedDataset(
        root_dir=str(tmp_path),
        precomputed_dir=str(precomputed),
        target_shape=(4, 4, 4),
        patch_size=(2, 4, 4),
        num_frames=128,
        dwi_matrix_path=str(sources["dwi"]),
        normative_csv_path=str(sources["normative"]),
        brainiac_model_path=str(sources["brainiac"]),
        brainiac_checkpoint_sha256=sources["brainiac_fingerprint"]["checkpoint_sha256"],
        brainiac_source_sha256=sources["brainiac_fingerprint"][
            "source_files_fingerprint_sha256"
        ],
        modernbert_model_name=str(sources["modernbert"]),
        modernbert_revision="a" * 40,
        roi_node_dim=515,
        fmri_dir=str(tmp_path / "missing-fmri"),
        load_fmri_targets=False,
        external_protocol_context=external_context,
        allow_synthetic_grid_override=True,
    )
    sample = dataset[0]
    assert "fmri" not in sample
    assert "fmri_mean" not in sample
    assert sample["t1w"].shape == (1, 4, 4, 4)
    assert sample["brain_mask"].shape == (1, 4, 4, 4)

    wrong_context = dict(external_context)
    wrong_context["synthesis_checkpoint_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="checkpoint identity differs"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(precomputed),
            target_shape=(4, 4, 4),
            patch_size=(2, 4, 4),
            num_frames=128,
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
            brainiac_model_path=str(sources["brainiac"]),
            brainiac_checkpoint_sha256=sources["brainiac_fingerprint"][
                "checkpoint_sha256"
            ],
            brainiac_source_sha256=sources["brainiac_fingerprint"][
                "source_files_fingerprint_sha256"
            ],
            modernbert_model_name=str(sources["modernbert"]),
            modernbert_revision="a" * 40,
            roi_node_dim=515,
            fmri_dir=str(tmp_path / "missing-fmri"),
            load_fmri_targets=False,
            external_protocol_context=wrong_context,
            allow_synthetic_grid_override=True,
        )

    provenance_path = tmp_path / "AnatCL" / scan_id / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["cat12_input"]["cat12_mwp1_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="graph cache is stale"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(precomputed),
            target_shape=(4, 4, 4),
            patch_size=(2, 4, 4),
            num_frames=128,
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
            brainiac_model_path=str(sources["brainiac"]),
            brainiac_checkpoint_sha256=sources["brainiac_fingerprint"][
                "checkpoint_sha256"
            ],
            brainiac_source_sha256=sources["brainiac_fingerprint"][
                "source_files_fingerprint_sha256"
            ],
            modernbert_model_name=str(sources["modernbert"]),
            modernbert_revision="a" * 40,
            roi_node_dim=515,
            fmri_dir=str(tmp_path / "missing-fmri"),
            load_fmri_targets=False,
            external_protocol_context=external_context,
            allow_synthetic_grid_override=True,
        )


def test_production_loader_rejects_disabling_cohort_validation(tmp_path):
    with pytest.raises(ValueError, match="cannot be disabled"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(tmp_path / "precomputed"),
            validate_files=False,
            target_shape=(4, 4, 4),
            allow_synthetic_grid_override=True,
        )


def test_native_recovery_loader_requires_complete_explicit_authorities(tmp_path):
    with pytest.raises(ValueError, match="structural authority is incomplete"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(tmp_path / "precomputed"),
            require_paper_preprocessing=False,
            target_shape=(4, 4, 4),
            allow_synthetic_grid_override=True,
        )


def test_production_loader_rejects_non_paper_frame_count(tmp_path):
    with pytest.raises(ValueError, match="exactly 128 fMRI frames"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(tmp_path / "precomputed"),
            num_frames=127,
            target_shape=(4, 4, 4),
            allow_synthetic_grid_override=True,
        )


def test_production_loader_rejects_a_silently_missing_cached_scan(tmp_path):
    # Build one complete cached scan, then add a second source T1/fMRI target
    # without graph artifacts. Training must not reinterpret that failure as a
    # smaller cohort.
    scan_id = "sub-01_run-1"
    scan_ids = (scan_id, "sub-02_run-1")
    sources = write_structural_sources(
        tmp_path, scan_ids=scan_ids, shape=(4, 4, 4), radiomics_dim=3
    )
    for source_scan_id in scan_ids:
        _write_valid_target(
            tmp_path,
            source_scan_id,
            shape=(4, 4, 4),
            num_frames=128,
            write_structural=False,
        )
    precomputed = write_certified_graph_cache(tmp_path, sources, scan_id)

    with pytest.raises(RuntimeError, match="cache cohort differs"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(precomputed),
            target_shape=(4, 4, 4),
            patch_size=(2, 4, 4),
            num_frames=128,
            dwi_matrix_path=str(sources["dwi"]),
            fmri_dir=str(tmp_path),
            allow_synthetic_grid_override=True,
        )


def test_cache_cannot_self_attest_arbitrary_patch_text(tmp_path):
    scan_id = "sub-01_run-1"
    sources = write_structural_sources(
        tmp_path, scan_ids=(scan_id,), shape=(4, 4, 4), radiomics_dim=3
    )
    _write_valid_target(
        tmp_path,
        scan_id,
        shape=(4, 4, 4),
        num_frames=128,
        write_structural=False,
    )
    precomputed = write_certified_graph_cache(tmp_path, sources, scan_id)
    descriptions_path = (
        precomputed / "hypergraphs" / f"{scan_id}_patch_descriptions.json"
    )
    descriptions = json.loads(descriptions_path.read_text())
    descriptions[0] = "arbitrary text that omits connectivity and Potvin context"
    descriptions_path.write_text(json.dumps(descriptions))
    metadata_path = precomputed / "hypergraphs" / f"{scan_id}_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Even if a stale-cache author updates the artifact's self-reported hash,
    # the loader independently reconstructs the paper-defined source text.
    metadata["artifact_sha256"][descriptions_path.name] = sha256_file(descriptions_path)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="cached patch text differs"):
        Connect4PrecomputedDataset(
            root_dir=str(tmp_path),
            precomputed_dir=str(precomputed),
            target_shape=(4, 4, 4),
            patch_size=(2, 4, 4),
            num_frames=128,
            dwi_matrix_path=str(sources["dwi"]),
            normative_csv_path=str(sources["normative"]),
            brainiac_model_path=str(sources["brainiac"]),
            brainiac_checkpoint_sha256=sources["brainiac_fingerprint"][
                "checkpoint_sha256"
            ],
            brainiac_source_sha256=sources["brainiac_fingerprint"][
                "source_files_fingerprint_sha256"
            ],
            modernbert_model_name=str(sources["modernbert"]),
            modernbert_revision="a" * 40,
            roi_node_dim=515,
            fmri_dir=str(tmp_path),
            allow_synthetic_grid_override=True,
        )
