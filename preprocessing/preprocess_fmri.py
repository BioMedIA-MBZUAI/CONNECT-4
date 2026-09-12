"""fMRIPrep-backed rs-fMRI preprocessing for CONNECT-4.

The paper explicitly cites fMRIPrep [Esteban et al., 2019]. Production inputs
to this module must therefore be fMRIPrep ``space-T1w_desc-preproc_bold``
derivatives with generated-by provenance. This module applies the remaining
smoothing/filtering and CONNECT-4 harmonisation steps. A custom numerical
implementation is retained only as an explicit debug backend and is marked
non-paper-faithful in its output provenance.

The required processing contract is:

1. authenticate fMRIPrep's T1w co-registration, slice-timing correction, and
   rigid motion correction;
2. resample the already co-registered BOLD derivative directly to a hash-bound
   T1-only cohort-common 3-mm grid in one interpolation, with any architecture
   padding explicitly recorded and cropped at publication;
3. apply spatial smoothing on that common grid;
4. apply temporal filtering;
5. harmonise to the paper-reported 128 frames and TR=3 s; and
6. apply the explicitly unpublished recovery intensity normalization.

For the debug backend only, FSL MCFLIRT/FLIRT or deterministic SciPy routines
implement the named corrections. Those outputs cannot pass the production
dataset validator because reproducing operations is not the same as using the
paper's cited pipeline.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import warnings

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage, optimize, signal

from utils.registration import register_fmri_to_t1w_nib
from .conform import (
    conform_4d,
    conform_volume,
    load_common_grid_contract,
    set_num_frames,
    TARGET_FRAMES,
    TR_SECONDS,
)
from .source_acquisition_identity import (
    SourceAcquisitionError,
    derivative_source_relative_paths,
    load_authenticated_json_artifact,
    load_fmriprep_runtime_identity,
    load_source_acquisition_identity,
    snapshot_json_artifact,
)


DEFAULT_SMOOTHING_FWHM_MM = 3.0
MIN_SMOOTHING_GRADIENT_RETENTION = 0.30
DEFAULT_HIGH_PASS_HZ = 0.01
DEFAULT_LOW_PASS_HZ = 0.10
DEFAULT_SLICE_TIMING_REFERENCE = 0.5
FMRI_PREPROCESSING_SCHEMA_VERSION = "connect4-fmriprep-preprocessing-v9"
FMRIPREP_EXECUTION_SCHEMA_VERSION = "connect4-fmriprep-execution-v4"
FMRIPREP_EVIDENCE_SCHEMA_VERSION = "connect4-fmriprep-evidence-v4"
FMRI_INTENSITY_NORMALIZATION_CONTRACT = (
    "connect4-unpublished-v54b-functional-validity-q995-nonnegative-v2"
)
FMRI_INTENSITY_NORMALIZATION_DOMAIN = "functional-validity-supported 4D values only"
FMRI_INTENSITY_NORMALIZATION_METHOD = (
    "clip nonnegative signal to max(functional-validity q99.5,1) and divide"
)
FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE = 0.995

_MOTION_COLUMNS = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
)


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def _sidecar_path(path: Path) -> Path:
    return path.with_name(_nifti_stem(path) + ".json")


def _bold_brain_mask_path(bold_path: Path) -> Path:
    """Resolve the exact same-run fMRIPrep brain mask for one BOLD derivative."""
    bold_stem = _nifti_stem(bold_path)
    suffix = "_desc-preproc_bold"
    if not bold_stem.endswith(suffix):
        raise ValueError("production BOLD name cannot identify its fMRIPrep brain mask")
    mask_stem = bold_stem[: -len(suffix)] + "_desc-brain_mask"
    candidates = [
        path
        for path in (
            bold_path.with_name(mask_stem + ".nii.gz"),
            bold_path.with_name(mask_stem + ".nii"),
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "production fMRIPrep evidence requires exactly one same-run "
            f"T1w-space BOLD brain mask for {bold_path}"
        )
    return candidates[0].resolve(strict=True)


def _functional_validity_mask_path(output_path: Path) -> Path:
    extension = ".nii.gz" if output_path.name.endswith(".nii.gz") else ".nii"
    return output_path.with_name(
        _nifti_stem(output_path) + "_functional_validity_mask" + extension
    )


def _load_bids_metadata(
    fmri_path: str, metadata_path: Optional[str]
) -> Tuple[Dict[str, Any], Optional[Path]]:
    path = Path(metadata_path) if metadata_path else _sidecar_path(Path(fmri_path))
    if not path.exists():
        return {}, None
    with open(path) as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"BIDS metadata must be a JSON object: {path}")
    return metadata, path


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        with path.open() as stream:
            value = json.load(stream)
    except Exception as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    return parsed


def _validated_fmriprep_extra_arguments(value: object) -> List[str]:
    """Validate the wrapper's closed production argument suffix."""
    if not isinstance(value, list) or not all(
        isinstance(token, str) and token for token in value
    ):
        raise ValueError("fMRIPrep execution extra arguments are malformed")
    allowed_flags = {"--fs-no-reconall", "--notrack"}
    allowed_values = {"--mem-mb", "--nthreads", "--omp-nthreads"}
    seen: set[str] = set()
    index = 0
    while index < len(value):
        option = value[index]
        if option in allowed_flags:
            if option in seen:
                raise ValueError("fMRIPrep execution repeats an extra argument")
            seen.add(option)
            index += 1
            continue
        if option not in allowed_values or option in seen:
            raise ValueError("fMRIPrep execution has a non-allowlisted extra argument")
        seen.add(option)
        index += 1
        if index >= len(value) or re.fullmatch(r"[1-9][0-9]*", value[index]) is None:
            raise ValueError("fMRIPrep execution extra-argument value is invalid")
        index += 1
    return list(value)


def _validated_fmriprep_version_probe(
    value: object,
    *,
    executable: Path,
    expected_version: str,
    label: str,
) -> Dict[str, Any]:
    """Validate a bounded, hash-described wrapper version probe."""
    fields = {
        "Command",
        "ReturnCode",
        "ObservedVersion",
        "StdoutSHA256",
        "StderrSHA256",
        "Stdout",
        "Stderr",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields differ")
    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    if (
        value.get("Command") != [str(executable), "--version"]
        or isinstance(value.get("ReturnCode"), bool)
        or not isinstance(value.get("ReturnCode"), int)
        or value.get("ReturnCode") != 0
        or value.get("ObservedVersion") != expected_version
        or sha256_pattern.fullmatch(str(value.get("StdoutSHA256", ""))) is None
        or sha256_pattern.fullmatch(str(value.get("StderrSHA256", ""))) is None
        or not isinstance(value.get("Stdout"), str)
        or not isinstance(value.get("Stderr"), str)
    ):
        raise ValueError(f"{label} does not attest the pinned fMRIPrep executable")
    stdout = value["Stdout"].encode("utf-8")
    stderr = value["Stderr"].encode("utf-8")
    if (
        len(stdout) > 65536
        or len(stderr) > 65536
        or hashlib.sha256(stdout).hexdigest() != value["StdoutSHA256"]
        or hashlib.sha256(stderr).hexdigest() != value["StderrSHA256"]
    ):
        raise ValueError(f"{label} output/hash evidence differs")
    return dict(value)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one immutable provenance artifact without loading it into memory."""
    if not path.is_file():
        raise FileNotFoundError(f"Provenance artifact not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _artifact_evidence(
    path: Path, derivatives_root: Optional[Path] = None
) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    evidence: Dict[str, Any] = {
        "Path": str(resolved),
        "SHA256": _sha256_file(resolved),
        "SizeBytes": int(resolved.stat().st_size),
    }
    if derivatives_root is not None:
        root = derivatives_root.resolve(strict=True)
        try:
            evidence["RelativePath"] = resolved.relative_to(root).as_posix()
        except ValueError:
            pass
    return evidence


def _fmriprep_generated_by(dataset_description: Dict[str, Any]) -> Dict[str, str]:
    """Return fMRIPrep provenance from derivative-root dataset_description.json."""
    generated_by = dataset_description.get("GeneratedBy", [])
    if isinstance(generated_by, dict):
        generated_by = [generated_by]
    if not isinstance(generated_by, list):
        raise ValueError("derivative dataset_description GeneratedBy must be a list")
    for entry in generated_by:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("Name", "")).strip().lower() != "fmriprep":
            continue
        version = str(entry.get("Version", "")).strip()
        if not version:
            raise ValueError("fMRIPrep GeneratedBy entry is missing Version")
        return {"Name": "fMRIPrep", "Version": version}
    raise ValueError(
        "derivative-root dataset_description.json does not identify fMRIPrep"
    )


def _command_option_values(command: Sequence[object], option: str) -> List[str]:
    """Extract all values following a repeatable fMRIPrep CLI option."""
    tokens = [str(token) for token in command]
    values: List[str] = []
    for index, token in enumerate(tokens):
        if token.startswith(option + "="):
            values.extend(
                value for value in re.split(r"[\s,]+", token.split("=", 1)[1]) if value
            )
            continue
        if token != option:
            continue
        cursor = index + 1
        while cursor < len(tokens) and not tokens[cursor].startswith("--"):
            values.extend(
                value for value in re.split(r"[\s,]+", tokens[cursor]) if value
            )
            cursor += 1
    return values


def _bids_entities(path: Path) -> Dict[str, str]:
    stem = _nifti_stem(path)
    entities: Dict[str, str] = {}
    for component in stem.split("_"):
        if "-" not in component:
            continue
        key, value = component.split("-", 1)
        if key and value:
            entities[key.lower()] = value
    return entities


def _run_prefix(path: Path) -> str:
    stem = _nifti_stem(path)
    match = re.search(r"_space-[^_]+", stem, flags=re.IGNORECASE)
    return stem[: match.start()] if match else stem


def _manifest_artifact_map(
    execution_record: Dict[str, Any],
    derivatives_root: Path,
) -> Dict[str, Dict[str, Any]]:
    artifacts = execution_record.get("Artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("fMRIPrep execution record has no hashed artifact inventory")
    mapped: Dict[str, Dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("fMRIPrep artifact inventory entries must be objects")
        relative = str(entry.get("RelativePath", "")).strip()
        digest = str(entry.get("SHA256", "")).strip().lower()
        size = entry.get("SizeBytes")
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid artifact inventory entry: {entry!r}")
        target = (derivatives_root / relative).resolve()
        try:
            target.relative_to(derivatives_root)
        except ValueError as exc:
            raise ValueError(f"Artifact escapes derivatives root: {relative}") from exc
        if relative in mapped:
            raise ValueError(f"Duplicate artifact in execution record: {relative}")
        mapped[relative] = entry
    return mapped


def _verify_manifest_artifact(
    path: Path,
    derivatives_root: Path,
    artifact_map: Dict[str, Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(derivatives_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{label} is outside the recorded derivatives root: {path}"
        ) from exc
    recorded = artifact_map.get(relative)
    if recorded is None:
        raise ValueError(
            f"{label} is absent from the successful wrapper artifact inventory: {relative}"
        )
    actual = _artifact_evidence(resolved, derivatives_root)
    if actual["SHA256"] != str(recorded.get("SHA256", "")).lower():
        raise ValueError(
            f"{label} SHA-256 no longer matches the execution record: {relative}"
        )
    if actual["SizeBytes"] != int(recorded.get("SizeBytes", -1)):
        raise ValueError(
            f"{label} size no longer matches the execution record: {relative}"
        )
    return actual


def _validate_motion_confounds(path: Path, expected_frames: int) -> int:
    """Require fMRIPrep's six finite rigid-motion traces, one row per frame."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = set(_MOTION_COLUMNS) - columns
        if missing:
            raise ValueError(
                f"fMRIPrep confounds file lacks motion columns {sorted(missing)}: {path}"
            )
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            for column in _MOTION_COLUMNS:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid motion value at {path}:{row_number} ({column})"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite motion value at {path}:{row_number} ({column})"
                    )
            rows += 1
    if rows != int(expected_frames):
        raise ValueError(
            f"fMRIPrep confounds has {rows} rows but BOLD has {expected_frames} frames"
        )
    return rows


def _validate_fmriprep_evidence(
    *,
    fmri_path: str,
    t1_path: str,
    metadata: Dict[str, Any],
    metadata_path: Optional[Path],
    dataset_description_path: str,
    execution_record_path: str,
    motion_confounds_path: str,
    coregistration_transform_path: str,
    num_frames: int,
    source_acquisition_identity: Mapping[str, Any],
    source_acquisition_evidence: Mapping[str, object],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Validate a wrapper-produced, hash-bound fMRIPrep evidence chain."""
    bold = Path(fmri_path).resolve(strict=True)
    t1 = Path(t1_path).resolve(strict=True)
    bold_brain_mask = _bold_brain_mask_path(bold)
    if metadata_path is None:
        raise ValueError("production fMRIPrep evidence requires the BOLD JSON sidecar")
    bold_metadata = metadata_path.resolve(strict=True)
    dataset_description = Path(dataset_description_path).resolve(strict=True)
    execution_path = Path(execution_record_path).resolve(strict=True)
    confounds = Path(motion_confounds_path).resolve(strict=True)
    transform = Path(coregistration_transform_path).resolve(strict=True)

    try:
        execution, execution_snapshot = snapshot_json_artifact(
            execution_path, label="fMRIPrep execution record"
        )
    except SourceAcquisitionError as exc:
        raise ValueError("fMRIPrep execution record cannot be authenticated") from exc
    expected_execution_fields = {
        "SchemaVersion",
        "Wrapper",
        "Success",
        "ReturnCode",
        "StartedAtUTC",
        "CompletedAtUTC",
        "Command",
        "BIDSRoot",
        "DerivativesRoot",
        "WorkRoot",
        "ScanIdentity",
        "ParticipantLabel",
        "OutputSpaces",
        "ExtraArguments",
        "IgnoredFeatures",
        "SliceTimingEnabledByWrapper",
        "SliceTimingReference",
        "FMRIPrepGeneratedBy",
        "FreshDerivativesGeneration",
        "BIDSInputInventory",
        "ExecutableIdentity",
        "RuntimeIdentity",
        "VersionProbeBeforeExecution",
        "VersionProbeAfterExecution",
        "SourceAcquisitionIdentity",
        "StdoutLog",
        "StderrLog",
        "Artifacts",
    }
    if set(execution) != expected_execution_fields:
        raise ValueError("fMRIPrep execution record fields differ from schema v4")
    if execution.get("SchemaVersion") != FMRIPREP_EXECUTION_SCHEMA_VERSION:
        raise ValueError("fMRIPrep execution record uses an unsupported schema")
    if execution.get("Wrapper") != "preprocessing.run_fmriprep":
        raise ValueError(
            "fMRIPrep execution record was not emitted by the CONNECT-4 wrapper"
        )
    if execution.get("Success") is not True or execution.get("ReturnCode") != 0:
        raise ValueError("fMRIPrep wrapper execution did not complete successfully")
    started_at = _utc_timestamp(execution.get("StartedAtUTC"), "StartedAtUTC")
    completed_at = _utc_timestamp(execution.get("CompletedAtUTC"), "CompletedAtUTC")
    if completed_at < started_at:
        raise ValueError("fMRIPrep execution completed before it started")
    command = execution.get("Command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise ValueError("fMRIPrep execution record has an invalid command")
    if len(command) < 4 or str(command[3]).lower() != "participant":
        raise ValueError(
            "fMRIPrep wrapper did not execute participant-level preprocessing"
        )
    output_spaces = _command_option_values(command, "--output-spaces")
    if "t1w" not in {value.lower() for value in output_spaces}:
        raise ValueError("fMRIPrep wrapper command did not request --output-spaces T1w")
    ignored_features = _command_option_values(command, "--ignore")
    if "slicetiming" in {
        value.strip().lower().replace("-", "").replace("_", "")
        for value in ignored_features
    }:
        raise ValueError(
            "fMRIPrep wrapper disabled the required slice-timing correction"
        )

    participant = str(execution.get("ParticipantLabel", "")).removeprefix("sub-")
    command_participants = [
        value.removeprefix("sub-")
        for value in _command_option_values(command, "--participant-label")
    ]
    if command_participants != [participant] or not participant:
        raise ValueError(
            "fMRIPrep command participant does not match its execution record"
        )
    recorded_spaces = execution.get("OutputSpaces")
    if recorded_spaces != ["T1w"] or [value.lower() for value in output_spaces] != [
        "t1w"
    ]:
        raise ValueError("fMRIPrep execution record must fix OutputSpaces to ['T1w']")
    if execution.get("IgnoredFeatures") != ignored_features:
        raise ValueError("fMRIPrep command and recorded ignored features disagree")
    if execution.get("SliceTimingEnabledByWrapper") is not True:
        raise ValueError(
            "fMRIPrep execution record does not affirm the wrapper constraint"
        )
    slice_timing_reference = execution.get("SliceTimingReference")
    if (
        isinstance(slice_timing_reference, bool)
        or not isinstance(slice_timing_reference, (int, float))
        or not np.isclose(float(slice_timing_reference), 0.5, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("fMRIPrep execution record does not fix slice-time reference")
    extra_arguments = _validated_fmriprep_extra_arguments(
        execution.get("ExtraArguments")
    )

    derivatives_root_value = str(execution.get("DerivativesRoot", "")).strip()
    if not derivatives_root_value:
        raise ValueError("fMRIPrep execution record lacks DerivativesRoot")
    derivatives_root = Path(derivatives_root_value).resolve(strict=True)
    if Path(command[2]).resolve() != derivatives_root:
        raise ValueError("recorded fMRIPrep command and DerivativesRoot disagree")
    bids_root_value = str(execution.get("BIDSRoot", "")).strip()
    if not bids_root_value:
        raise ValueError("fMRIPrep execution record lacks BIDSRoot")
    bids_root = Path(bids_root_value).resolve(strict=True)
    if not bids_root.is_dir() or Path(command[1]).resolve() != bids_root:
        raise ValueError("recorded fMRIPrep command and BIDSRoot disagree")
    if bids_root != Path(str(source_acquisition_identity.get("bids_root", ""))):
        raise ValueError("fMRIPrep BIDS root differs from source-acquisition identity")
    if (
        str(source_acquisition_identity.get("participant_label", "")).removeprefix(
            "sub-"
        )
        != participant
    ):
        raise ValueError(
            "fMRIPrep participant differs from source-acquisition identity"
        )
    source_scan_id = str(source_acquisition_identity.get("scan_id", ""))
    source_bold_entities = dict(
        source_acquisition_identity.get("bids_entities", {}).get("raw_bold", {})
    )
    source_input_inventory = source_acquisition_identity.get("bids_input_inventory")
    if not source_scan_id or not isinstance(source_input_inventory, dict):
        raise ValueError("source-acquisition scan/input inventory is incomplete")
    expected_scan_identity = {
        "ScanID": source_scan_id,
        "ParticipantLabel": participant,
        "Session": source_bold_entities.get("ses"),
        "Task": source_bold_entities.get("task"),
        "Run": source_bold_entities.get("run"),
        "Acquisition": source_bold_entities.get("acq"),
        "Direction": source_bold_entities.get("dir"),
        "Echo": source_bold_entities.get("echo"),
        "RawBOLDRelativePath": source_input_inventory.get("raw_bold_relative_path"),
        "RawT1wRelativePath": source_input_inventory.get("raw_t1_relative_path"),
    }
    if execution.get("ScanIdentity") != expected_scan_identity:
        raise ValueError("fMRIPrep execution identifies another exact scan")
    input_inventory_binding = execution.get("BIDSInputInventory")
    if (
        not isinstance(input_inventory_binding, dict)
        or set(input_inventory_binding)
        != {"Identity", "VerifiedBeforeExecution", "VerifiedAfterExecution"}
        or input_inventory_binding.get("Identity") != source_input_inventory
        or input_inventory_binding.get("VerifiedBeforeExecution") is not True
        or input_inventory_binding.get("VerifiedAfterExecution") is not True
    ):
        raise ValueError(
            "fMRIPrep execution does not bind the complete BIDS input inventory"
        )
    source_binding = execution.get("SourceAcquisitionIdentity")
    expected_source_artifact = {
        "Path": source_acquisition_evidence.get("path"),
        "SHA256": source_acquisition_evidence.get("sha256"),
        "SizeBytes": source_acquisition_evidence.get("size_bytes"),
        "RecordSHA256": source_acquisition_evidence.get("record_sha256"),
    }
    if (
        not isinstance(source_binding, dict)
        or set(source_binding)
        != {
            "Artifact",
            "Identity",
            "VerifiedBeforeExecution",
            "VerifiedAfterExecution",
        }
        or source_binding.get("Artifact") != expected_source_artifact
        or source_binding.get("Identity") != dict(source_acquisition_identity)
        or source_binding.get("VerifiedBeforeExecution") is not True
        or source_binding.get("VerifiedAfterExecution") is not True
    ):
        raise ValueError(
            "fMRIPrep execution does not bind the exact source-acquisition identity"
        )

    fresh_generation = execution.get("FreshDerivativesGeneration")
    if (
        not isinstance(fresh_generation, dict)
        or set(fresh_generation)
        != {
            "GenerationID",
            "ScanID",
            "Path",
            "PathAbsentBeforeCreation",
            "CreatedExclusively",
            "InitiallyEmpty",
            "NoReuse",
        }
        or re.fullmatch(r"[0-9a-f]{64}", str(fresh_generation.get("GenerationID", "")))
        is None
        or fresh_generation.get("ScanID") != source_scan_id
        or fresh_generation.get("Path") != str(derivatives_root)
        or fresh_generation.get("PathAbsentBeforeCreation") is not True
        or fresh_generation.get("CreatedExclusively") is not True
        or fresh_generation.get("InitiallyEmpty") is not True
        or fresh_generation.get("NoReuse") is not True
    ):
        raise ValueError("fMRIPrep derivatives are not a fresh exact-scan generation")

    executable_identity = execution.get("ExecutableIdentity")
    runtime_binding = execution.get("RuntimeIdentity")
    if (
        not isinstance(executable_identity, dict)
        or not isinstance(runtime_binding, dict)
        or set(runtime_binding)
        != {
            "Artifact",
            "Identity",
            "VerifiedBeforeExecution",
            "VerifiedAfterExecution",
        }
        or not isinstance(runtime_binding.get("Artifact"), dict)
        or set(runtime_binding["Artifact"])
        != {"Path", "SHA256", "SizeBytes", "RecordSHA256"}
    ):
        raise ValueError("fMRIPrep execution runtime binding is malformed")
    recorded_runtime_artifact = runtime_binding["Artifact"]
    try:
        executable_path = Path(str(executable_identity.get("path", "")))
        runtime_identity, runtime_evidence = load_fmriprep_runtime_identity(
            Path(str(recorded_runtime_artifact["Path"])),
            expected_sha256=str(recorded_runtime_artifact["SHA256"]),
            expected_executable=executable_path,
        )
    except (KeyError, OSError, SourceAcquisitionError) as exc:
        raise ValueError("fMRIPrep runtime identity cannot be authenticated") from exc
    expected_runtime_artifact = {
        "Path": runtime_evidence["path"],
        "SHA256": runtime_evidence["sha256"],
        "SizeBytes": runtime_evidence["size_bytes"],
        "RecordSHA256": runtime_evidence["record_sha256"],
    }
    if (
        recorded_runtime_artifact != expected_runtime_artifact
        or runtime_binding.get("Identity") != runtime_identity
        or runtime_binding.get("VerifiedBeforeExecution") is not True
        or runtime_binding.get("VerifiedAfterExecution") is not True
        or executable_identity != runtime_identity.get("executable")
        or not executable_path.is_file()
        or not os.access(executable_path, os.X_OK)
    ):
        raise ValueError("fMRIPrep execution runtime identity differs")
    runtime_version = str(runtime_identity["fmriprep_version"])
    version_probe_before = _validated_fmriprep_version_probe(
        execution.get("VersionProbeBeforeExecution"),
        executable=executable_path,
        expected_version=runtime_version,
        label="pre-execution fMRIPrep version probe",
    )
    version_probe_after = _validated_fmriprep_version_probe(
        execution.get("VersionProbeAfterExecution"),
        executable=executable_path,
        expected_version=runtime_version,
        label="post-execution fMRIPrep version probe",
    )

    expected_work_root = derivatives_root / "connect4-work"
    fixed_command = [
        str(executable_path),
        str(bids_root),
        str(derivatives_root),
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
        str(expected_work_root),
    ]
    if command != fixed_command + extra_arguments or execution.get("WorkRoot") != str(
        expected_work_root
    ):
        raise ValueError("fMRIPrep command differs from the closed v4 execution")
    expected_execution_path = (
        derivatives_root / "logs" / f"connect4-fmriprep_{source_scan_id}_execution.json"
    )
    if execution_path != expected_execution_path.resolve(strict=True):
        raise ValueError("fMRIPrep execution record is not at the wrapper-defined path")
    try:
        execution_path.relative_to(derivatives_root)
    except ValueError as exc:
        raise ValueError(
            "execution record must be stored inside the derivatives root"
        ) from exc
    expected_dataset_description = derivatives_root / "dataset_description.json"
    if dataset_description != expected_dataset_description.resolve(strict=True):
        raise ValueError(
            "fMRIPrep provenance must come from derivative-root dataset_description.json"
        )

    artifact_map = _manifest_artifact_map(execution, derivatives_root)
    dataset_description_evidence = _verify_manifest_artifact(
        dataset_description,
        derivatives_root,
        artifact_map,
        "derivative dataset_description.json",
    )
    try:
        description, _ = load_authenticated_json_artifact(
            dataset_description,
            expected_sha256=dataset_description_evidence["SHA256"],
            expected_size=dataset_description_evidence["SizeBytes"],
            label="fMRIPrep dataset_description.json",
        )
    except SourceAcquisitionError as exc:
        raise ValueError(
            "fMRIPrep dataset_description cannot be authenticated"
        ) from exc
    if str(description.get("DatasetType", "")).lower() != "derivative":
        raise ValueError(
            "fMRIPrep dataset_description DatasetType must be 'derivative'"
        )
    generated_by = _fmriprep_generated_by(description)
    if (
        execution.get("FMRIPrepGeneratedBy") != generated_by
        or generated_by.get("Version") != runtime_version
    ):
        raise ValueError(
            "fMRIPrep execution/runtime does not match derivative GeneratedBy provenance"
        )

    execution_logs: Dict[str, Dict[str, Any]] = {}
    for record_field, evidence_field in (
        ("StdoutLog", "StdoutLog"),
        ("StderrLog", "StderrLog"),
    ):
        recorded_log = execution.get(record_field)
        if not isinstance(recorded_log, dict):
            raise ValueError(f"fMRIPrep execution record lacks {record_field} evidence")
        recorded_log_path = str(recorded_log.get("Path", "")).strip()
        if not recorded_log_path:
            raise ValueError(
                f"fMRIPrep execution record has invalid {record_field} evidence"
            )
        actual_log = _verify_manifest_artifact(
            Path(recorded_log_path),
            derivatives_root,
            artifact_map,
            record_field,
        )
        if any(
            recorded_log.get(key) != actual_log[key]
            for key in ("Path", "RelativePath", "SHA256", "SizeBytes")
        ):
            raise ValueError(
                f"fMRIPrep {record_field} evidence disagrees with its inventory"
            )
        execution_logs[evidence_field] = actual_log

    bold_entities = _bids_entities(bold)
    subject = bold_entities.get("sub", "")
    if not subject or participant != subject:
        raise ValueError("BOLD subject does not match the wrapper participant")
    for entity in ("sub", "ses", "task", "run", "acq", "dir", "echo"):
        if bold_entities.get(entity) != source_bold_entities.get(entity):
            raise ValueError(
                "selected fMRIPrep BOLD entities differ from the exact source scan"
            )
    if (
        "space-t1w" not in _nifti_stem(bold).lower()
        or "desc-preproc_bold" not in _nifti_stem(bold).lower()
    ):
        raise ValueError(
            "production BOLD must be a space-T1w desc-preproc_bold derivative"
        )
    if bold.relative_to(derivatives_root).parts[0] != f"sub-{subject}":
        raise ValueError(
            "BOLD derivative is not in its standard fMRIPrep subject directory"
        )
    t1_entities = _bids_entities(t1)
    if (
        t1_entities.get("sub") != subject
        or "desc-preproc_t1w" not in _nifti_stem(t1).lower()
    ):
        raise ValueError(
            "supplied T1 must be the same-subject fMRIPrep desc-preproc_T1w derivative"
        )
    bold_session = bold_entities.get("ses")
    if t1_entities.get("ses") not in {None, bold_session}:
        raise ValueError("BOLD and T1 session entities disagree")
    if t1.relative_to(derivatives_root).parts[0] != f"sub-{subject}":
        raise ValueError(
            "T1w reference is not in the selected fMRIPrep subject directory"
        )

    prefix = _run_prefix(bold)
    expected_metadata = _sidecar_path(bold).resolve(strict=True)
    if bold_metadata != expected_metadata:
        raise ValueError("BOLD metadata sidecar does not match the selected derivative")
    expected_mask_stem = (
        _nifti_stem(bold).removesuffix("_desc-preproc_bold") + "_desc-brain_mask"
    )
    if (
        bold_brain_mask.parent != bold.parent
        or _nifti_stem(bold_brain_mask) != expected_mask_stem
    ):
        raise ValueError("BOLD brain mask does not match the selected BOLD run")
    if (
        confounds.parent != bold.parent
        or not confounds.name.startswith(prefix + "_desc-confounds_")
        or confounds.suffix != ".tsv"
    ):
        raise ValueError("motion confounds do not match the selected BOLD run")
    transform_lower = transform.name.lower()
    transform_from = next(
        (
            source
            for source in ("boldref", "scanner")
            if transform.name.startswith(f"{prefix}_from-{source}_to-T1w_")
        ),
        None,
    )
    if (
        transform.parent != bold.parent
        or transform_from is None
        or "mode-image" not in transform_lower
        or "xfm" not in transform_lower
    ):
        raise ValueError(
            "coregistration transform is not this run's explicit fMRIPrep "
            "BOLD-to-T1w transform"
        )
    if transform.stat().st_size < 1:
        raise ValueError("BOLD-to-T1w transform is empty")
    if int(num_frames) < 5:
        raise ValueError(
            "fMRIPrep disables slice-timing correction for very short runs; "
            "production evidence requires at least five frames"
        )

    if metadata.get("SliceTimingCorrected") is not True:
        raise ValueError(
            "fMRIPrep BOLD metadata does not confirm SliceTimingCorrected=true"
        )
    motion_rows = _validate_motion_confounds(confounds, num_frames)

    bold_evidence = _verify_manifest_artifact(
        bold,
        derivatives_root,
        artifact_map,
        "BOLD derivative",
    )
    bold_brain_mask_evidence = _verify_manifest_artifact(
        bold_brain_mask,
        derivatives_root,
        artifact_map,
        "BOLD brain mask",
    )
    metadata_evidence = _verify_manifest_artifact(
        bold_metadata,
        derivatives_root,
        artifact_map,
        "BOLD metadata",
    )
    confounds_evidence = _verify_manifest_artifact(
        confounds,
        derivatives_root,
        artifact_map,
        "motion confounds",
    )
    transform_evidence = _verify_manifest_artifact(
        transform,
        derivatives_root,
        artifact_map,
        "BOLD-to-T1w transform",
    )
    t1_evidence = _verify_manifest_artifact(
        t1,
        derivatives_root,
        artifact_map,
        "T1w reference",
    )
    t1_metadata = _sidecar_path(t1).resolve(strict=True)
    t1_metadata_evidence = _verify_manifest_artifact(
        t1_metadata,
        derivatives_root,
        artifact_map,
        "T1w metadata",
    )
    try:
        authenticated_bold_metadata, _ = load_authenticated_json_artifact(
            bold_metadata,
            expected_sha256=metadata_evidence["SHA256"],
            expected_size=metadata_evidence["SizeBytes"],
            label="fMRIPrep BOLD metadata",
        )
        authenticated_t1_metadata, _ = load_authenticated_json_artifact(
            t1_metadata,
            expected_sha256=t1_metadata_evidence["SHA256"],
            expected_size=t1_metadata_evidence["SizeBytes"],
            label="fMRIPrep T1w metadata",
        )
    except SourceAcquisitionError as exc:
        raise ValueError(
            "fMRIPrep derivative metadata cannot be authenticated"
        ) from exc
    if authenticated_bold_metadata != metadata:
        raise ValueError("selected BOLD metadata changed during validation")
    raw_t1 = source_acquisition_identity["raw_t1"]
    raw_bold = source_acquisition_identity["raw_bold"]
    raw_t1_relative = Path(str(raw_t1["path"])).relative_to(bids_root).as_posix()
    raw_bold_relative = Path(str(raw_bold["path"])).relative_to(bids_root).as_posix()
    try:
        bold_sources = derivative_source_relative_paths(authenticated_bold_metadata)
        t1_sources = derivative_source_relative_paths(authenticated_t1_metadata)
    except SourceAcquisitionError as exc:
        raise ValueError("fMRIPrep derivative source binding is invalid") from exc
    if raw_bold_relative not in bold_sources:
        raise ValueError("BOLD derivative Sources omit the authenticated raw BOLD")
    if raw_t1_relative not in t1_sources:
        raise ValueError("T1w derivative Sources omit the authenticated raw T1w")

    evidence = {
        "EvidenceSchemaVersion": FMRIPREP_EVIDENCE_SCHEMA_VERSION,
        "ParticipantLabel": subject,
        "ExecutionWrapper": "preprocessing.run_fmriprep",
        "ExecutionSuccess": True,
        "ExecutionReturnCode": 0,
        "ExecutionStartedAtUTC": execution["StartedAtUTC"],
        "ExecutionCompletedAtUTC": execution["CompletedAtUTC"],
        "ScanIdentity": expected_scan_identity,
        "OutputSpaces": output_spaces,
        "ExtraArguments": extra_arguments,
        "IgnoredFeatures": ignored_features,
        "SliceTimingEnabledByWrapper": True,
        "SliceTimingReference": 0.5,
        "DerivativeSliceTimingCorrected": True,
        "MotionConfoundsRows": motion_rows,
        "FreshDerivativesGeneration": dict(fresh_generation),
        "BIDSInputInventory": dict(input_inventory_binding),
        "ExecutableIdentity": dict(executable_identity),
        "RuntimeIdentity": dict(runtime_binding),
        "VersionProbeBeforeExecution": version_probe_before,
        "VersionProbeAfterExecution": version_probe_after,
        "CoregistrationBinding": {
            "BOLDRunPrefix": prefix,
            "TransformFrom": transform_from,
            "TransformTo": "T1w",
            "TransformSemantic": "BOLD-to-subject-T1w registration",
            "T1wSubject": subject,
            "T1wSession": t1_entities.get("ses"),
            "BOLDDerivativeSHA256": bold_evidence["SHA256"],
            "BOLDBrainMaskSHA256": bold_brain_mask_evidence["SHA256"],
            "TransformArtifactSHA256": transform_evidence["SHA256"],
            "TargetT1wArtifactSHA256": t1_evidence["SHA256"],
            "RawT1wSHA256": raw_t1["sha256"],
            "RawBOLDSHA256": raw_bold["sha256"],
            "SynthSegMaskSHA256": source_acquisition_identity["synthseg"]["mask"][
                "sha256"
            ],
        },
        "SourceAcquisitionIdentity": expected_source_artifact,
        "SourceAcquisition": dict(source_acquisition_identity),
        "ExecutionRecord": {
            "Path": execution_snapshot["path"],
            "RelativePath": execution_path.relative_to(derivatives_root).as_posix(),
            "SHA256": execution_snapshot["sha256"],
            "SizeBytes": execution_snapshot["size_bytes"],
        },
        "DerivativeDatasetDescription": dataset_description_evidence,
        "BOLDDerivative": bold_evidence,
        "BOLDBrainMask": bold_brain_mask_evidence,
        "BOLDMetadata": metadata_evidence,
        "MotionConfounds": confounds_evidence,
        "BOLDToT1wTransform": transform_evidence,
        "T1wReference": t1_evidence,
        "T1wMetadata": t1_metadata_evidence,
        **execution_logs,
    }
    return generated_by, evidence


def _validate_fmriprep_derivative(
    fmri_path: str,
    metadata: Dict[str, Any],
) -> None:
    """Validate file-level BOLD identity; execution evidence is checked separately."""
    stem = _nifti_stem(Path(fmri_path)).lower()
    spatial_reference = metadata.get("SpatialReference", "")
    if isinstance(spatial_reference, (list, tuple)):
        spatial_reference = " ".join(str(value) for value in spatial_reference)
    in_t1_space = "space-t1w" in stem or "t1w" in str(spatial_reference).lower()
    if not in_t1_space:
        raise ValueError(
            "fMRIPrep BOLD derivative must be in subject T1w space; use the "
            "space-T1w desc-preproc_bold output"
        )
    if metadata.get("SliceTimingCorrected") is not True:
        raise ValueError(
            "fMRIPrep derivative sidecar must set SliceTimingCorrected=true so "
            "the paper-required slice-timing correction is auditable"
        )


def _source_tr_seconds(
    img: nib.Nifti1Image,
    metadata: Dict[str, Any],
    override: Optional[float] = None,
) -> float:
    metadata_tr = (
        float(metadata["RepetitionTime"])
        if metadata.get("RepetitionTime") is not None
        else None
    )
    header_tr = None
    if img.ndim == 4 and len(img.header.get_zooms()) >= 4:
        time_unit = img.header.get_xyzt_units()[1]
        if time_unit in {"sec", "msec", "usec"}:
            header_tr = float(img.header.get_zooms()[3])
            if time_unit == "msec":
                header_tr /= 1_000.0
            elif time_unit == "usec":
                header_tr /= 1_000_000.0
    recorded = [value for value in (metadata_tr, header_tr) if value is not None]
    if any(not np.isfinite(value) or value <= 0 for value in recorded):
        raise ValueError(f"Invalid recorded source repetition time: {recorded!r}")
    if len(recorded) == 2 and not np.isclose(
        recorded[0], recorded[1], rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"BIDS and NIfTI source TR disagree: {recorded[0]} vs {recorded[1]}"
        )
    if override is not None:
        tr = float(override)
        if recorded and not all(
            np.isclose(tr, value, rtol=0.0, atol=1e-6) for value in recorded
        ):
            raise ValueError(
                "source_tr may only restate, never override, BIDS/NIfTI TR evidence"
            )
    elif recorded:
        tr = recorded[0]
    else:
        raise ValueError("source TR is absent from both BIDS metadata and NIfTI header")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"Invalid source repetition time: {tr!r}")
    return tr


def _slice_axis(
    img: nib.Nifti1Image,
    metadata: Dict[str, Any],
    override: Optional[int] = None,
) -> int:
    if override is not None:
        axis = int(override)
    else:
        direction = str(metadata.get("SliceEncodingDirection", ""))
        axis_for_letter = {"i": 0, "j": 1, "k": 2}
        axis = axis_for_letter.get(direction[:1].lower(), -1)
        if axis < 0:
            try:
                axis = img.header.get_dim_info()[2]
            except Exception:
                axis = None
            axis = 2 if axis is None else int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1, or 2; got {axis}")
    return axis


def _infer_slice_times(num_slices: int, tr: float, order: str) -> np.ndarray:
    """Return per-slice acquisition times for an explicitly declared order."""
    if num_slices < 1:
        raise ValueError("num_slices must be positive")
    ascending = list(range(num_slices))
    if order == "ascending":
        acquisition_order = ascending
    elif order == "descending":
        acquisition_order = ascending[::-1]
    elif order == "interleaved-ascending":
        acquisition_order = ascending[::2] + ascending[1::2]
    elif order == "interleaved-descending":
        descending = ascending[::-1]
        acquisition_order = descending[::2] + descending[1::2]
    else:
        raise ValueError(f"Unknown slice acquisition order: {order!r}")
    times = np.empty(num_slices, dtype=np.float64)
    for rank, slice_index in enumerate(acquisition_order):
        times[slice_index] = rank * float(tr) / float(num_slices)
    return times


def _resample_slice_times(slice_times: np.ndarray, num_slices: int) -> np.ndarray:
    """Map native-grid slice times to a co-registered grid if counts differ."""
    if slice_times.size == num_slices:
        return slice_times.astype(np.float64, copy=False)
    if slice_times.size < 2:
        return np.repeat(slice_times, num_slices).astype(np.float64)
    old_positions = np.linspace(0.0, 1.0, slice_times.size)
    new_positions = np.linspace(0.0, 1.0, num_slices)
    return np.interp(new_positions, old_positions, slice_times).astype(np.float64)


def _resolve_slice_times(
    img: nib.Nifti1Image,
    metadata: Dict[str, Any],
    num_slices: int,
    tr: float,
    slice_axis: int,
    fallback_order: str,
) -> Tuple[np.ndarray, str]:
    if metadata.get("SliceTiming") is not None:
        times = np.asarray(metadata["SliceTiming"], dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
            raise ValueError("BIDS SliceTiming must be a finite one-dimensional list")
        direction = str(metadata.get("SliceEncodingDirection", ""))
        if direction.endswith("-"):
            times = times[::-1]
        source = "BIDS SliceTiming"
    else:
        times = np.asarray([], dtype=np.float64)
        try:
            header_times = img.header.get_slice_times()
            if header_times and all(value is not None for value in header_times):
                times = np.asarray(header_times, dtype=np.float64)
        except Exception:
            pass
        if times.size:
            source = "NIfTI slice timing"
        else:
            times = _infer_slice_times(img.shape[slice_axis], tr, fallback_order)
            source = f"inferred {fallback_order}"
            warnings.warn(
                "No BIDS/NIfTI slice timing metadata found; using declared "
                f"fallback order {fallback_order!r}. Supply the BIDS JSON sidecar "
                "for acquisition-specific correction.",
                RuntimeWarning,
            )
    if np.any(times < 0) or np.any(times >= tr + 1e-7):
        raise ValueError(
            f"SliceTiming values must lie in [0, TR); TR={tr}, range={times.min()}..{times.max()}"
        )
    if times.size != num_slices:
        source += (
            f" (interpolated {times.size}->{num_slices} slices after co-registration)"
        )
        times = _resample_slice_times(times, num_slices)
    return times, source


def _slice_timing_correct(
    data4d: np.ndarray,
    tr: float,
    slice_times: Sequence[float],
    slice_axis: int,
    reference_fraction: float = DEFAULT_SLICE_TIMING_REFERENCE,
) -> np.ndarray:
    """Correct each slice to a common within-TR reference using cubic interpolation."""
    data = np.asarray(data4d, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected [X,Y,Z,T] fMRI, got {data.shape}")
    if not 0.0 <= reference_fraction <= 1.0:
        raise ValueError("slice timing reference must be a fraction in [0, 1]")
    times = np.asarray(slice_times, dtype=np.float64)
    if times.shape != (data.shape[slice_axis],):
        raise ValueError(
            f"Expected {data.shape[slice_axis]} slice times for axis {slice_axis}, got {times.shape}"
        )
    corrected = np.empty_like(data, dtype=np.float32)
    reference_time = reference_fraction * float(tr)
    for slice_index, acquisition_time in enumerate(times):
        selector = [slice(None)] * 4
        selector[slice_axis] = slice_index
        block = data[tuple(selector)]
        temporal_shift = (float(acquisition_time) - reference_time) / float(tr)
        shift_vector = [0.0] * block.ndim
        shift_vector[-1] = temporal_shift
        corrected[tuple(selector)] = ndimage.shift(
            block,
            shift=shift_vector,
            order=3,
            mode="nearest",
            prefilter=True,
        ).astype(np.float32, copy=False)
    return corrected


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _apply_rigid(
    volume: np.ndarray, parameters: Sequence[float], order: int = 1
) -> np.ndarray:
    """Apply moving-to-reference rotations (rad) and translations (voxels)."""
    rx, ry, rz, tx, ty, tz = [float(value) for value in parameters]
    rotation = _rotation_matrix(rx, ry, rz)
    inverse = rotation.T
    center = (np.asarray(volume.shape, dtype=np.float64) - 1.0) / 2.0
    translation = np.asarray([tx, ty, tz], dtype=np.float64)
    offset = center - inverse @ (center + translation)
    return ndimage.affine_transform(
        volume,
        matrix=inverse,
        offset=offset,
        output_shape=volume.shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def _rigid_forward_matrix(
    parameters: Sequence[float], shape: Sequence[int]
) -> np.ndarray:
    """Moving-to-reference homogeneous transform in voxel coordinates."""
    rx, ry, rz, tx, ty, tz = [float(value) for value in parameters]
    rotation = _rotation_matrix(rx, ry, rz)
    center = (np.asarray(shape[:3], dtype=np.float64) - 1.0) / 2.0
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center + np.asarray([tx, ty, tz]) - rotation @ center
    return matrix


def _robust_unit_scale(volume: np.ndarray) -> np.ndarray:
    if not np.isfinite(volume).all():
        raise ValueError("registration volume contains NaN or infinity")
    values = volume[volume != 0]
    if values.size < 32:
        return np.asarray(volume, dtype=np.float32).copy()
    lower, upper = np.percentile(values, [2.0, 98.0])
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def _normalised_mutual_information(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Histogram NMI, robust to the different T1w and mean-BOLD contrasts."""
    valid = np.isfinite(fixed) & np.isfinite(moving) & ((fixed > 0) | (moving > 0))
    if valid.sum() < 64:
        return 0.0
    joint, _, _ = np.histogram2d(
        fixed[valid], moving[valid], bins=32, range=((0, 1), (0, 1))
    )
    total = joint.sum()
    if total <= 0:
        return 0.0
    probability = joint / total
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    nonzero_joint = probability > 0
    entropy_joint = -np.sum(
        probability[nonzero_joint] * np.log(probability[nonzero_joint])
    )
    nonzero_x = px > 0
    nonzero_y = py > 0
    entropy_x = -np.sum(px[nonzero_x] * np.log(px[nonzero_x]))
    entropy_y = -np.sum(py[nonzero_y] * np.log(py[nonzero_y]))
    return (
        float((entropy_x + entropy_y) / entropy_joint) if entropy_joint > 1e-12 else 0.0
    )


def _estimate_rigid_mutual_information(
    fixed_t1: np.ndarray,
    moving_mean_bold: np.ndarray,
    max_iterations: int = 60,
) -> np.ndarray:
    """Estimate a six-DOF moving-to-fixed transform with mutual information."""
    stride = max(1, int(math.ceil(max(fixed_t1.shape) / 48.0)))
    fixed = _robust_unit_scale(
        ndimage.gaussian_filter(fixed_t1[::stride, ::stride, ::stride], sigma=0.8)
    )
    moving = _robust_unit_scale(
        ndimage.gaussian_filter(
            moving_mean_bold[::stride, ::stride, ::stride], sigma=0.8
        )
    )
    fixed_weight = (
        fixed > np.percentile(fixed[fixed > 0], 15.0)
        if np.any(fixed > 0)
        else fixed > 0
    )
    moving_weight = (
        moving > np.percentile(moving[moving > 0], 15.0)
        if np.any(moving > 0)
        else moving > 0
    )
    fixed_center = np.asarray(ndimage.center_of_mass(fixed_weight.astype(np.float32)))
    moving_center = np.asarray(ndimage.center_of_mass(moving_weight.astype(np.float32)))
    if not np.isfinite(fixed_center).all() or not np.isfinite(moving_center).all():
        initial_translation = np.zeros(3, dtype=np.float64)
    else:
        initial_translation = np.clip(fixed_center - moving_center, -8.0, 8.0)
    initial = np.concatenate([np.zeros(3), initial_translation])

    def objective(candidate: np.ndarray) -> float:
        warped = _apply_rigid(moving, candidate, order=1)
        return -_normalised_mutual_information(fixed, warped)

    result = optimize.minimize(
        objective,
        initial,
        method="Powell",
        bounds=[(-0.262, 0.262)] * 3 + [(-12.0, 12.0)] * 3,
        options={"maxiter": int(max_iterations), "xtol": 5e-4, "ftol": 1e-5},
    )
    coarse = result.x if np.isfinite(result.fun) else initial
    full = np.asarray(coarse, dtype=np.float64).copy()
    full[3:] *= stride
    return full


def _header_resample_to_t1(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    t1_canonical = nib.as_closest_canonical(t1_img)
    registered = (
        register_fmri_to_t1w_nib(fmri_img, t1_canonical)[0].detach().cpu().numpy()
    )
    data = np.transpose(registered, (1, 2, 3, 0)).astype(np.float32, copy=False)
    return data, t1_canonical


def _header_resample_mean_to_t1(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> np.ndarray:
    """Memory-bounded header initialisation of only the temporal mean."""
    fmri_canonical = nib.as_closest_canonical(fmri_img)
    t1_canonical = nib.as_closest_canonical(t1_img)
    accumulator = np.zeros(fmri_canonical.shape[:3], dtype=np.float64)
    for frame_index in range(fmri_canonical.shape[-1]):
        accumulator += np.asanyarray(
            fmri_canonical.dataobj[..., frame_index], dtype=np.float32
        )
    accumulator /= float(fmri_canonical.shape[-1])
    mean_img = nib.Nifti1Image(accumulator.astype(np.float32), fmri_canonical.affine)
    try:
        resampled = resample_from_to(
            mean_img,
            t1_canonical,
            order=1,
            mode="constant",
            cval=0.0,
            force_resample=True,
            copy_header=True,
        )
    except TypeError:
        resampled = resample_from_to(
            mean_img, t1_canonical, order=1, mode="constant", cval=0.0
        )
    return resampled.get_fdata(dtype=np.float32)


def _coregister_scipy(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Header initialisation followed by actual six-DOF T1/BOLD MI fitting."""
    initial_data, t1_canonical = _header_resample_to_t1(fmri_img, t1_img)
    fixed = t1_canonical.get_fdata(dtype=np.float32)
    moving_mean = initial_data.mean(axis=-1, dtype=np.float64).astype(np.float32)
    parameters = _estimate_rigid_mutual_information(fixed, moving_mean)
    registered = np.empty_like(initial_data, dtype=np.float32)
    for frame_index in range(initial_data.shape[-1]):
        registered[..., frame_index] = _apply_rigid(
            initial_data[..., frame_index], parameters, order=1
        ).astype(np.float32, copy=False)
    voxel_transform = _rigid_forward_matrix(parameters, fixed.shape)
    world_transform = (
        t1_canonical.affine @ voxel_transform @ np.linalg.inv(t1_canonical.affine)
    )
    return registered, world_transform, "world-space RAS+ moving-to-T1"


def _coregister_flirt(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Estimate mean-BOLD-to-T1 six-DOF registration with FSL FLIRT."""
    flirt = shutil.which("flirt")
    if flirt is None:
        raise RuntimeError("FSL FLIRT was requested but 'flirt' is not on PATH")
    t1_canonical = nib.as_closest_canonical(t1_img)
    fmri_canonical = nib.as_closest_canonical(fmri_img)
    fmri_data = fmri_canonical.get_fdata(dtype=np.float32)
    mean_bold = fmri_data.mean(axis=-1, dtype=np.float64).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="connect4_flirt_") as temp_dir:
        temp = Path(temp_dir)
        fmri_path = temp / "bold.nii.gz"
        mean_path = temp / "mean_bold.nii.gz"
        t1_path = temp / "t1.nii.gz"
        matrix_path = temp / "bold_to_t1.mat"
        registered_path = temp / "registered.nii.gz"
        nib.save(fmri_canonical, fmri_path)
        nib.save(nib.Nifti1Image(mean_bold, fmri_canonical.affine), mean_path)
        nib.save(t1_canonical, t1_path)
        estimate = subprocess.run(
            [
                flirt,
                "-in",
                str(mean_path),
                "-ref",
                str(t1_path),
                "-omat",
                str(matrix_path),
                "-dof",
                "6",
                "-cost",
                "normmi",
                "-searchrx",
                "-30",
                "30",
                "-searchry",
                "-30",
                "30",
                "-searchrz",
                "-30",
                "30",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if estimate.returncode != 0:
            raise RuntimeError(f"FLIRT registration failed: {estimate.stderr.strip()}")
        applyxfm4d = shutil.which("applyxfm4D")
        if applyxfm4d:
            applied = subprocess.run(
                [
                    applyxfm4d,
                    str(fmri_path),
                    str(t1_path),
                    str(registered_path),
                    str(matrix_path),
                    "-singlematrix",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if applied.returncode != 0:
                raise RuntimeError(f"applyxfm4D failed: {applied.stderr.strip()}")
            registered = nib.load(registered_path).get_fdata(dtype=np.float32)
        else:
            frames = []
            for frame_index in range(fmri_data.shape[-1]):
                frame_in = temp / f"frame_{frame_index:04d}.nii.gz"
                frame_out = temp / f"frame_{frame_index:04d}_reg.nii.gz"
                nib.save(
                    nib.Nifti1Image(fmri_data[..., frame_index], fmri_canonical.affine),
                    frame_in,
                )
                applied = subprocess.run(
                    [
                        flirt,
                        "-in",
                        str(frame_in),
                        "-ref",
                        str(t1_path),
                        "-out",
                        str(frame_out),
                        "-applyxfm",
                        "-init",
                        str(matrix_path),
                        "-interp",
                        "trilinear",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if applied.returncode != 0:
                    raise RuntimeError(
                        f"FLIRT frame application failed: {applied.stderr.strip()}"
                    )
                frames.append(nib.load(frame_out).get_fdata(dtype=np.float32))
            registered = np.stack(frames, axis=-1)
        transform = np.loadtxt(matrix_path)
    return (
        registered.astype(np.float32, copy=False),
        transform,
        "FSL FLIRT scaled-mm matrix",
    )


def _validate_coregistration(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    registered: np.ndarray,
    transform: np.ndarray,
) -> Dict[str, Any]:
    """Require finite geometry and no material NMI regression before saving."""
    t1_canonical = nib.as_closest_canonical(t1_img)
    expected_shape = t1_canonical.shape[:3] + (fmri_img.shape[-1],)
    if registered.shape != expected_shape:
        raise RuntimeError(
            f"Co-registration produced shape {registered.shape}; expected {expected_shape}"
        )
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("Co-registration transform must be a finite 4x4 matrix")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
        raise RuntimeError("Co-registration transform is singular")
    fixed = _robust_unit_scale(t1_canonical.get_fdata(dtype=np.float32))
    initial_mean = _robust_unit_scale(
        _header_resample_mean_to_t1(fmri_img, t1_canonical)
    )
    registered_mean = _robust_unit_scale(
        registered.mean(axis=-1, dtype=np.float64).astype(np.float32)
    )
    nmi_initial = _normalised_mutual_information(fixed, initial_mean)
    nmi_final = _normalised_mutual_information(fixed, registered_mean)
    validated = bool(np.isfinite(nmi_final) and nmi_final + 0.01 >= nmi_initial)
    if not validated:
        raise RuntimeError(
            "Co-registration validation failed: normalized mutual information "
            f"regressed from {nmi_initial:.6f} to {nmi_final:.6f}"
        )
    return {
        "CoregistrationValidated": True,
        "CoregistrationNMIInitial": float(nmi_initial),
        "CoregistrationNMIFinal": float(nmi_final),
    }


def _coregister_data(
    fmri_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    backend: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, Dict[str, Any]]:
    requested = backend.lower()
    if requested not in {"auto", "flirt", "python", "header"}:
        raise ValueError(
            "coregistration backend must be 'auto', 'flirt', 'python', or 'header'"
        )
    if requested == "flirt" or (requested == "auto" and shutil.which("flirt")):
        try:
            data, transform, convention = _coregister_flirt(fmri_img, t1_img)
            quality = _validate_coregistration(fmri_img, t1_img, data, transform)
            return (
                data,
                transform,
                "FSL FLIRT six-DOF mutual information",
                convention,
                quality,
            )
        except Exception:
            if requested == "flirt":
                raise
            warnings.warn(
                "FLIRT failed; falling back to SciPy mutual-information registration",
                RuntimeWarning,
            )
    if requested in {"auto", "python"}:
        data, transform, convention = _coregister_scipy(fmri_img, t1_img)
        quality = _validate_coregistration(fmri_img, t1_img, data, transform)
        return data, transform, "SciPy six-DOF mutual information", convention, quality
    data, _ = _header_resample_to_t1(fmri_img, t1_img)
    quality = _validate_coregistration(fmri_img, t1_img, data, np.eye(4))
    return (
        data,
        np.eye(4),
        "header-only resampling (debug)",
        "identity world-space RAS+",
        quality,
    )


def _motion_correct_python(
    data4d: np.ndarray,
    voxel_sizes: Sequence[float],
    max_iterations: int = 35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Six-DOF volume-to-middle-volume motion correction using SciPy."""
    data = np.asarray(data4d, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected [X,Y,Z,T] fMRI, got {data.shape}")
    num_frames = data.shape[-1]
    parameters = np.zeros((num_frames, 6), dtype=np.float64)
    if num_frames <= 1:
        return data.copy(), parameters

    reference_index = num_frames // 2
    reference = data[..., reference_index]
    stride = max(1, int(math.ceil(max(reference.shape) / 48.0)))
    ref_small = ndimage.gaussian_filter(
        reference[::stride, ::stride, ::stride], sigma=0.75
    )
    nonzero = np.abs(ref_small[np.isfinite(ref_small) & (ref_small != 0)])
    threshold = np.percentile(nonzero, 20.0) if nonzero.size else 0.0
    ref_mask = np.isfinite(ref_small) & (np.abs(ref_small) > threshold)
    reference_center = np.asarray(ndimage.center_of_mass(np.abs(ref_small)))
    if not np.isfinite(reference_center).all():
        reference_center = (np.asarray(ref_small.shape) - 1.0) / 2.0

    corrected = np.empty_like(data, dtype=np.float32)
    for frame_index in range(num_frames):
        moving = data[..., frame_index]
        if frame_index == reference_index:
            corrected[..., frame_index] = moving
            continue
        moving_small = ndimage.gaussian_filter(
            moving[::stride, ::stride, ::stride], sigma=0.75
        )
        moving_center = np.asarray(ndimage.center_of_mass(np.abs(moving_small)))
        if not np.isfinite(moving_center).all():
            moving_center = reference_center
        initial = np.zeros(6, dtype=np.float64)
        initial[3:] = np.clip(reference_center - moving_center, -4.0, 4.0)

        def objective(candidate: np.ndarray) -> float:
            warped = _apply_rigid(moving_small, candidate, order=1)
            valid = ref_mask & np.isfinite(warped)
            if valid.sum() < 32:
                return 2.0
            fixed_values = ref_small[valid].astype(np.float64)
            moving_values = warped[valid].astype(np.float64)
            fixed_values -= fixed_values.mean()
            moving_values -= moving_values.mean()
            denominator = np.linalg.norm(fixed_values) * np.linalg.norm(moving_values)
            if denominator <= 1e-12:
                return 2.0
            return 1.0 - float(np.dot(fixed_values, moving_values) / denominator)

        result = optimize.minimize(
            objective,
            initial,
            method="Powell",
            bounds=[(-0.175, 0.175)] * 3 + [(-6.0, 6.0)] * 3,
            options={"maxiter": int(max_iterations), "xtol": 1e-3, "ftol": 1e-4},
        )
        coarse_parameters = result.x if np.isfinite(result.fun) else initial
        full_parameters = np.asarray(coarse_parameters, dtype=np.float64)
        full_parameters[3:] *= stride
        parameters[frame_index] = full_parameters
        corrected[..., frame_index] = _apply_rigid(
            moving, full_parameters, order=1
        ).astype(np.float32)

    parameters[:, 3:] *= np.asarray(voxel_sizes[:3], dtype=np.float64)
    return corrected, parameters


def _motion_correct_mcflirt(img: nib.Nifti1Image) -> Tuple[np.ndarray, np.ndarray]:
    executable = shutil.which("mcflirt")
    if executable is None:
        raise RuntimeError("FSL MCFLIRT was requested but 'mcflirt' is not on PATH")
    with tempfile.TemporaryDirectory(prefix="connect4_mcflirt_") as temp_dir:
        input_path = Path(temp_dir) / "coreg_stc.nii.gz"
        output_base = Path(temp_dir) / "mc"
        nib.save(img, input_path)
        command = [
            executable,
            "-in",
            str(input_path),
            "-out",
            str(output_base),
            "-refvol",
            str(img.shape[-1] // 2),
            "-plots",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"MCFLIRT failed: {completed.stderr.strip()}")
        output_path = Path(str(output_base) + ".nii.gz")
        if not output_path.exists():
            output_path = output_base
        corrected = nib.load(output_path).get_fdata(dtype=np.float32)
        parameter_path = Path(str(output_base) + ".par")
        parameters = (
            np.loadtxt(parameter_path, ndmin=2)
            if parameter_path.exists()
            else np.zeros((img.shape[-1], 6))
        )
        return corrected.astype(np.float32, copy=False), np.asarray(
            parameters, dtype=np.float64
        )


def _motion_correct_data(
    data4d: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    backend: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    requested = backend.lower()
    if requested not in {"auto", "mcflirt", "python"}:
        raise ValueError("motion backend must be 'auto', 'mcflirt', or 'python'")
    if requested == "mcflirt" or (requested == "auto" and shutil.which("mcflirt")):
        image = nib.Nifti1Image(data4d.astype(np.float32), affine, header)
        corrected, parameters = _motion_correct_mcflirt(image)
        return corrected, parameters, "FSL MCFLIRT"
    corrected, parameters = _motion_correct_python(
        data4d, nib.affines.voxel_sizes(affine)
    )
    return corrected, parameters, "SciPy six-DOF rigid fallback"


def _masked_spatial_gradient_energy(data4d: np.ndarray, mask3d: np.ndarray) -> float:
    """Mean absolute neighbor gradient where both voxels are inside the mask."""
    data = np.asarray(data4d, dtype=np.float32)
    mask = np.asarray(mask3d, dtype=bool)
    values = []
    for axis in range(3):
        difference = np.diff(data, axis=axis)
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        valid = mask[tuple(low)] & mask[tuple(high)]
        if valid.any():
            values.append(float(np.mean(np.abs(difference[valid, :]))))
    if not values:
        return 0.0
    return float(np.mean(values))


def _spatial_smooth(
    data4d: np.ndarray,
    mask3d: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float,
) -> np.ndarray:
    """Mask-normalized Gaussian smoothing without edge darkening or signal bleed."""
    if fwhm_mm < 0:
        raise ValueError("smoothing FWHM must be non-negative")
    data = np.asarray(data4d, dtype=np.float32)
    mask = np.asarray(mask3d, dtype=bool)
    if data.ndim != 4 or data.shape[:3] != mask.shape or not mask.any():
        raise ValueError("spatial smoothing requires a non-empty aligned 3D mask")
    output = np.zeros_like(data, dtype=np.float32)
    if fwhm_mm == 0:
        output[mask, :] = data[mask, :]
        return output
    voxel_sizes = nib.affines.voxel_sizes(affine)
    sigma_mm = float(fwhm_mm) / math.sqrt(8.0 * math.log(2.0))
    sigma_vox = tuple(sigma_mm / float(size) for size in voxel_sizes[:3])
    weights = ndimage.gaussian_filter(
        mask.astype(np.float32), sigma=sigma_vox, mode="constant", cval=0.0
    )
    numerator = ndimage.gaussian_filter(
        data * mask[..., None],
        sigma=sigma_vox + (0.0,),
        mode="constant",
        cval=0.0,
    )
    safe = np.maximum(weights, np.finfo(np.float32).eps)
    output[mask, :] = (numerator / safe[..., None])[mask, :]
    if not np.isfinite(output).all() or np.any(output[~mask, :] != 0):
        raise RuntimeError("mask-normalized smoothing violated its support contract")
    return output


def _temporal_filter(
    data4d: np.ndarray,
    tr: float,
    high_pass_hz: Optional[float],
    low_pass_hz: Optional[float],
    order: int = 2,
    voxel_batch_size: int = 65536,
) -> np.ndarray:
    """Mean-preserving zero-phase Butterworth filtering with bounded memory.

    A conventional high-pass/band-pass removes the voxel DC component. That
    makes the manuscript's temporal-mean intensity comparisons degenerate, so
    this recovery implementation filters only the centered residual and then
    restores the exact pre-filter voxel mean.
    """
    if not np.isfinite(data4d).all():
        raise ValueError("temporal filter input contains NaN or infinity")
    high = None if high_pass_hz is None or high_pass_hz <= 0 else float(high_pass_hz)
    low = None if low_pass_hz is None or low_pass_hz <= 0 else float(low_pass_hz)
    if high is None and low is None:
        return np.asarray(data4d, dtype=np.float32).copy()
    nyquist = 0.5 / float(tr)
    if low is not None and low >= nyquist:
        raise ValueError(
            f"low-pass cutoff {low} Hz must be below Nyquist {nyquist:.6g} Hz"
        )
    if high is not None and high >= nyquist:
        raise ValueError(
            f"high-pass cutoff {high} Hz must be below Nyquist {nyquist:.6g} Hz"
        )
    if high is not None and low is not None and high >= low:
        raise ValueError("high-pass cutoff must be lower than low-pass cutoff")
    if high is not None and low is not None:
        cutoff: Any = (high, low)
        filter_type = "bandpass"
    elif high is not None:
        cutoff = high
        filter_type = "highpass"
    else:
        cutoff = low
        filter_type = "lowpass"
    sos = signal.butter(
        order, cutoff, btype=filter_type, fs=1.0 / float(tr), output="sos"
    )
    num_frames = data4d.shape[-1]
    padlen = min(num_frames - 1, 3 * (2 * len(sos) + 1))
    if padlen < 1:
        raise ValueError("Temporal filtering requires at least two fMRI frames")
    flat = np.asarray(data4d, dtype=np.float32).reshape(-1, num_frames)
    filtered = np.empty_like(flat, dtype=np.float32)
    for start in range(0, flat.shape[0], voxel_batch_size):
        stop = min(start + voxel_batch_size, flat.shape[0])
        block = flat[start:stop].astype(np.float64)
        original_mean = block.mean(axis=-1, keepdims=True)
        residual = block - original_mean
        cleaned = signal.sosfiltfilt(sos, residual, axis=-1, padlen=padlen)
        cleaned -= cleaned.mean(axis=-1, keepdims=True)
        filtered[start:stop] = (original_mean + cleaned).astype(np.float32, copy=False)
    filtered = filtered.reshape(data4d.shape)
    if not np.isfinite(filtered).all():
        raise RuntimeError("temporal filtering produced NaN or infinity")
    return filtered


def _resample_to_tr(
    data4d: np.ndarray,
    source_tr: float,
    target_tr: float = TR_SECONDS,
    voxel_batch_size: int = 65536,
) -> np.ndarray:
    """Resample the temporal axis to the target TR with anti-alias filtering."""
    if not np.isfinite(data4d).all():
        raise ValueError("TR-resampling input contains NaN or infinity")
    if np.isclose(source_tr, target_tr, rtol=0.0, atol=1e-6):
        return np.asarray(data4d, dtype=np.float32).copy()
    ratio = Fraction(float(source_tr) / float(target_tr)).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    num_frames = data4d.shape[-1]
    output_frames = int(math.ceil(num_frames * up / down))
    flat = np.asarray(data4d, dtype=np.float32).reshape(-1, num_frames)
    resampled = np.empty((flat.shape[0], output_frames), dtype=np.float32)
    for start in range(0, flat.shape[0], voxel_batch_size):
        stop = min(start + voxel_batch_size, flat.shape[0])
        block = signal.resample_poly(flat[start:stop], up, down, axis=-1)
        resampled[start:stop] = block[..., :output_frames].astype(
            np.float32, copy=False
        )
    resampled = resampled.reshape(data4d.shape[:-1] + (output_frames,))
    if not np.isfinite(resampled).all():
        raise RuntimeError("TR resampling produced NaN or infinity")
    return resampled


def _recovery_intensity_normalize(
    data4d: np.ndarray,
    mask3d: np.ndarray,
    *,
    upper_quantile: float = FMRI_INTENSITY_NORMALIZATION_UPPER_QUANTILE,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Apply the declared unpublished v54b-compatible per-scan scale.

    The paper does not disclose an intensity normalization. This is therefore
    recorded as a recovery choice, not a manuscript claim. Unlike the removed
    temporal z-score, it does not force every voxel to zero temporal mean and
    unit variance. The quantile domain is the functional-validity mask so the
    result cannot change merely because the common grid has more background.
    """
    data = np.asarray(data4d, dtype=np.float32)
    mask = np.asarray(mask3d, dtype=bool)
    if data.ndim != 4 or data.shape[:3] != mask.shape or not mask.any():
        raise ValueError(
            "intensity normalization requires aligned 4D data and a non-empty mask"
        )
    if not np.isfinite(data).all():
        raise ValueError("intensity-normalization input contains NaN or infinity")
    if not 0.0 < float(upper_quantile) < 1.0:
        raise ValueError("intensity upper quantile must lie strictly inside (0,1)")
    nonnegative = np.maximum(data, 0.0).astype(np.float32, copy=False)
    nonnegative[~mask, :] = 0.0
    functional_values = nonnegative[mask, :]
    quantile = float(np.quantile(functional_values, upper_quantile))
    if not np.isfinite(quantile):
        raise RuntimeError("functional-validity intensity quantile is non-finite")
    ceiling = max(quantile, 1.0)
    output = (np.clip(nonnegative, 0.0, ceiling) / ceiling).astype(
        np.float32, copy=False
    )
    output[~mask, :] = 0.0
    record = {
        "Contract": FMRI_INTENSITY_NORMALIZATION_CONTRACT,
        "ReportedByPaper": False,
        "Method": FMRI_INTENSITY_NORMALIZATION_METHOD,
        "Domain": FMRI_INTENSITY_NORMALIZATION_DOMAIN,
        "UpperQuantile": float(upper_quantile),
        "FunctionalValidityQuantileBeforeClip": quantile,
        "CeilingAndDivisor": float(ceiling),
        "MinimumDivisor": 1.0,
        "FunctionalValidityValueCount": int(functional_values.size),
        "FractionNegativeBeforeClipInFunctionalValidity": float(
            np.mean(data[mask, :] < 0.0)
        ),
        "FractionClippedAtCeilingInFunctionalValidity": float(
            np.mean(functional_values > ceiling)
        ),
        "LowerQuantileSubtraction": False,
        "TemporalVoxelZScore": False,
        "OutsideMaskForcedZero": True,
        "OutputMinimum": float(output.min()),
        "OutputMaximum": float(output.max()),
    }
    return output, record


def _write_motion_parameters(out_path: Path, parameters: np.ndarray) -> Path:
    path = out_path.with_name(_nifti_stem(out_path) + "_motion.tsv")
    header = "rotation_x_rad\trotation_y_rad\trotation_z_rad\ttranslation_x_mm\ttranslation_y_mm\ttranslation_z_mm"
    np.savetxt(path, np.asarray(parameters), delimiter="\t", header=header, comments="")
    return path


def preprocess_fmri(
    fmri_path: str,
    t1_path: str,
    out_path: str,
    mcflirt: bool = False,
    normalize: bool = True,
    *,
    metadata_path: Optional[str] = None,
    fmriprep_dataset_description_path: Optional[str] = None,
    fmriprep_execution_record_path: Optional[str] = None,
    source_acquisition_identity_path: Optional[str] = None,
    source_acquisition_identity_sha256: Optional[str] = None,
    motion_confounds_path: Optional[str] = None,
    coregistration_transform_path: Optional[str] = None,
    brain_mask_path: Optional[str] = None,
    common_grid_contract_path: Optional[str] = None,
    common_grid_contract_sha256: Optional[str] = None,
    source_tr: Optional[float] = None,
    slice_axis: Optional[int] = None,
    slice_order: str = "interleaved-ascending",
    slice_timing_reference: Optional[float] = None,
    preprocessing_backend: str = "fmriprep",
    coregistration_backend: str = "auto",
    motion_backend: str = "auto",
    apply_slice_timing: bool = True,
    apply_motion_correction: bool = True,
    smoothing_fwhm_mm: float = DEFAULT_SMOOTHING_FWHM_MM,
    high_pass_hz: Optional[float] = DEFAULT_HIGH_PASS_HZ,
    low_pass_hz: Optional[float] = DEFAULT_LOW_PASS_HZ,
    allow_synthetic_grid_override: bool = False,
) -> str:
    """Postprocess a verified fMRIPrep derivative for CONNECT-4.

    ``preprocessing_backend='custom-debug'`` exercises the local correction
    routines for tests/diagnostics, but its provenance deliberately cannot pass
    the production dataset contract.
    """
    if preprocessing_backend not in {"fmriprep", "custom-debug"}:
        raise ValueError("preprocessing_backend must be 'fmriprep' or 'custom-debug'")
    if preprocessing_backend == "fmriprep" and source_tr is not None:
        raise ValueError(
            "production fMRIPrep preprocessing forbids source_tr overrides; "
            "TR must come from hash-bound BIDS/NIfTI evidence"
        )
    if preprocessing_backend == "fmriprep" and slice_timing_reference is not None:
        raise ValueError(
            "production fMRIPrep preprocessing forbids slice-timing-reference "
            "overrides; the wrapper evidence fixes it to 0.5"
        )
    source_acquisition_identity = None
    source_acquisition_evidence = None
    if preprocessing_backend == "fmriprep":
        if (
            not source_acquisition_identity_path
            or not source_acquisition_identity_sha256
        ):
            raise ValueError(
                "production fMRIPrep preprocessing requires an externally "
                "SHA-pinned source-acquisition identity"
            )
        source_acquisition_identity, source_acquisition_evidence = (
            load_source_acquisition_identity(
                Path(source_acquisition_identity_path),
                expected_sha256=source_acquisition_identity_sha256,
            )
        )
    if preprocessing_backend == "fmriprep" and normalize is not True:
        raise ValueError(
            "certified CONNECT-4 targets require the declared unpublished "
            "nonnegative functional-validity q99.5 recovery scaling"
        )
    if preprocessing_backend == "fmriprep" and not np.isclose(
        float(smoothing_fwhm_mm), DEFAULT_SMOOTHING_FWHM_MM, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "production uses the versioned 3-mm mask-normalized smoothing recovery "
            "choice; this value is not claimed as a manuscript hyperparameter"
        )
    if preprocessing_backend == "fmriprep":
        for value, expected, label in (
            (high_pass_hz, DEFAULT_HIGH_PASS_HZ, "high-pass"),
            (low_pass_hz, DEFAULT_LOW_PASS_HZ, "low-pass"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isclose(float(value), expected, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    f"production uses the versioned {expected:g}-Hz {label} "
                    "recovery choice; this value is not claimed as a manuscript "
                    "hyperparameter"
                )
    if (
        preprocessing_backend == "fmriprep"
        and common_grid_contract_path is not None
        and not common_grid_contract_sha256
    ):
        raise ValueError(
            "production common-grid evidence requires an externally configured "
            "SHA-256 pin"
        )
    common_grid = None
    if common_grid_contract_path is not None:
        common_grid = load_common_grid_contract(
            common_grid_contract_path,
            expected_sha256=common_grid_contract_sha256,
        )
    fmri_img = nib.load(fmri_path)
    t1_img = nib.load(t1_path)
    if fmri_img.ndim != 4 or fmri_img.shape[-1] < 2:
        raise ValueError(
            f"rs-fMRI must be a 4D run with at least two frames; got {fmri_img.shape}"
        )
    if not np.isfinite(fmri_img.get_fdata(dtype=np.float32)).all():
        raise ValueError("source fMRI contains NaN or infinity")
    if not np.isfinite(t1_img.get_fdata(dtype=np.float32)).all():
        raise ValueError("T1w reference contains NaN or infinity")
    if brain_mask_path is not None:
        structural_mask_img = nib.load(brain_mask_path)
        mask_values = structural_mask_img.get_fdata(dtype=np.float32)
        if structural_mask_img.ndim != 3 or not np.isfinite(mask_values).all():
            raise ValueError("anatomical structural mask must be a finite 3D NIfTI")
        if not np.any(mask_values > 0):
            raise ValueError("anatomical structural mask is empty")
    elif allow_synthetic_grid_override:
        # Explicitly test-only: production is rejected above.  This keeps tiny
        # numerical fixtures independent of a full SynthSeg artifact.
        structural_mask_img = nib.Nifti1Image(
            (t1_img.get_fdata(dtype=np.float32) != 0).astype(np.uint8),
            t1_img.affine,
        )
    else:
        structural_mask_img = None
    metadata, resolved_metadata_path = _load_bids_metadata(fmri_path, metadata_path)
    source_tr_value = _source_tr_seconds(fmri_img, metadata, source_tr)
    source_slice_axis = _slice_axis(fmri_img, metadata, slice_axis)
    if preprocessing_backend == "fmriprep":
        metadata_slice_reference = metadata.get(
            "SliceTimingReference", DEFAULT_SLICE_TIMING_REFERENCE
        )
        if (
            isinstance(metadata_slice_reference, bool)
            or not isinstance(metadata_slice_reference, (int, float))
            or not np.isclose(
                float(metadata_slice_reference),
                DEFAULT_SLICE_TIMING_REFERENCE,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "production fMRIPrep derivative metadata must fix "
                "SliceTimingReference to 0.5"
            )
        resolved_slice_timing_reference = DEFAULT_SLICE_TIMING_REFERENCE
    else:
        resolved_slice_timing_reference = float(
            slice_timing_reference
            if slice_timing_reference is not None
            else metadata.get("SliceTimingReference", DEFAULT_SLICE_TIMING_REFERENCE)
        )
    t1_canonical = nib.as_closest_canonical(t1_img)
    synthetic_target = None
    if common_grid is None:
        synthetic_target = (t1_canonical.shape[:3], t1_canonical.affine)
    fmriprep_info = None
    fmriprep_evidence = None
    bold_brain_mask_img = None
    coregistration_transform = None
    motion_parameters = None

    if preprocessing_backend == "fmriprep":
        if not apply_slice_timing or not apply_motion_correction:
            raise ValueError(
                "fMRIPrep production inputs cannot disable the manuscript's "
                "slice-timing or motion-correction requirements"
            )
        _validate_fmriprep_derivative(fmri_path, metadata)
        evidence_arguments = {
            "fmriprep_dataset_description_path": fmriprep_dataset_description_path,
            "fmriprep_execution_record_path": fmriprep_execution_record_path,
            "motion_confounds_path": motion_confounds_path,
            "coregistration_transform_path": coregistration_transform_path,
        }
        missing_evidence = [
            name for name, value in evidence_arguments.items() if not value
        ]
        if missing_evidence:
            raise ValueError(
                "production fMRIPrep preprocessing requires the complete wrapper "
                f"evidence chain; missing {', '.join(missing_evidence)}"
            )
        fmriprep_info, fmriprep_evidence = _validate_fmriprep_evidence(
            fmri_path=fmri_path,
            t1_path=t1_path,
            metadata=metadata,
            metadata_path=resolved_metadata_path,
            dataset_description_path=str(fmriprep_dataset_description_path),
            execution_record_path=str(fmriprep_execution_record_path),
            motion_confounds_path=str(motion_confounds_path),
            coregistration_transform_path=str(coregistration_transform_path),
            num_frames=int(fmri_img.shape[-1]),
            source_acquisition_identity=source_acquisition_identity,
            source_acquisition_evidence=source_acquisition_evidence,
        )
        bold_brain_mask_img = nib.load(str(fmriprep_evidence["BOLDBrainMask"]["Path"]))
        bold_brain_mask_values = bold_brain_mask_img.get_fdata(dtype=np.float32)
        if (
            bold_brain_mask_img.ndim != 3
            or not np.isfinite(bold_brain_mask_values).all()
        ):
            raise ValueError("fMRIPrep BOLD brain mask must be a finite 3D NIfTI")
        if not np.all(
            (bold_brain_mask_values == 0.0) | (bold_brain_mask_values == 1.0)
        ):
            raise ValueError("fMRIPrep BOLD brain mask must be strictly binary")
        if not np.any(bold_brain_mask_values > 0.5):
            raise ValueError("fMRIPrep BOLD brain mask is empty")
        if tuple(bold_brain_mask_img.shape) != tuple(fmri_img.shape[:3]) or not (
            np.allclose(
                bold_brain_mask_img.affine,
                fmri_img.affine,
                rtol=0.0,
                atol=1e-5,
            )
        ):
            raise ValueError(
                "fMRIPrep BOLD brain mask geometry does not match the selected BOLD"
            )
        print(
            f"[fmri] 1-3/6 verified fMRIPrep {fmriprep_info['Version']} "
            "T1w-space derivative and hash-bound execution evidence "
            "(co-registration, slice timing, motion)",
            flush=True,
        )
        # The derivative is already co-registered to subject T1w space by
        # fMRIPrep.  Preserve its native sampled grid here and compose the only
        # CONNECT-4 spatial interpolation directly into the cohort-common grid
        # below.  Resampling first to the fMRIPrep T1 image and then again to
        # the common grid destroys fine spatial texture whenever those grids
        # differ.
        registered = fmri_img.get_fdata(dtype=np.float32)
        registered_affine = np.asarray(fmri_img.affine, dtype=np.float64)
        post_fmriprep_target_interpolation_count = 1
        spatial_resampling_route = (
            "fMRIPrep T1w-space BOLD grid directly to cohort-common grid"
        )
        resolved_coregistration_backend = (
            "fMRIPrep explicit BOLD-to-T1w image transform"
        )
        transform_convention = "fMRIPrep BOLD-to-subject-T1w image transform"
        coregistration_quality = {
            "CoregistrationValidated": True,
            "CoregistrationEvidenceValidated": True,
        }
        slice_timing_source = (
            "fMRIPrep derivative SliceTimingCorrected=true + successful "
            "wrapper command with slice timing enabled"
        )
        resolved_motion_backend = (
            "fMRIPrep head-motion correction with six-parameter confounds evidence"
        )
    else:
        print(
            "[fmri] DEBUG 1/6 estimating six-DOF mean-BOLD-to-T1w "
            "co-registration outside fMRIPrep",
            flush=True,
        )
        (
            registered,
            coregistration_transform,
            resolved_coregistration_backend,
            transform_convention,
            coregistration_quality,
        ) = _coregister_data(fmri_img, t1_canonical, coregistration_backend)
        registered_affine = np.asarray(t1_canonical.affine, dtype=np.float64)
        post_fmriprep_target_interpolation_count = None
        spatial_resampling_route = (
            "debug coregistration grid to explicit synthetic/common grid"
        )

    registered_header = fmri_img.header.copy()
    if int(fmri_img.shape[-1]) < int(TARGET_FRAMES):
        raise ValueError(
            f"fMRI source run has {fmri_img.shape[-1]} frames; at least "
            f"{TARGET_FRAMES} acquired frames are required before TR "
            "harmonisation and temporal padding is forbidden"
        )
    registered_header.set_data_dtype(np.float32)
    registered_header.set_data_shape(registered.shape)
    registered_header.set_zooms(
        tuple(nib.affines.voxel_sizes(registered_affine)) + (source_tr_value,)
    )

    if preprocessing_backend == "custom-debug":
        slice_timing_source = "disabled"
        if apply_slice_timing:
            print("[fmri] DEBUG 2/6 applying slice-timing correction", flush=True)
            registered_slice_axis = source_slice_axis
            slice_times, slice_timing_source = _resolve_slice_times(
                fmri_img,
                metadata,
                registered.shape[registered_slice_axis],
                source_tr_value,
                source_slice_axis,
                slice_order,
            )
            registered = _slice_timing_correct(
                registered,
                tr=source_tr_value,
                slice_times=slice_times,
                slice_axis=registered_slice_axis,
                reference_fraction=resolved_slice_timing_reference,
            )

        motion_parameters = np.zeros((registered.shape[-1], 6), dtype=np.float64)
        resolved_motion_backend = "disabled"
        if apply_motion_correction:
            print("[fmri] DEBUG 3/6 applying rigid motion correction", flush=True)
            if mcflirt:
                motion_backend = "mcflirt"
            registered, motion_parameters, resolved_motion_backend = (
                _motion_correct_data(
                    registered,
                    t1_canonical.affine,
                    registered_header,
                    backend=motion_backend,
                )
            )

    if common_grid is None and not allow_synthetic_grid_override:
        raise ValueError(
            "production fMRI preprocessing requires a hash-bound structural "
            "common-grid contract"
        )
    if structural_mask_img is None:
        raise ValueError(
            "production fMRI preprocessing requires a hash-bound anatomical mask"
        )

    print(
        "[fmri] 4/6 resampling to the hash-bound structural common grid",
        flush=True,
    )
    registered_img = nib.Nifti1Image(registered, registered_affine, registered_header)
    conformed = conform_4d(
        registered_img,
        order=1,
        tr_seconds=source_tr_value,
        grid_contract=common_grid,
        synthetic_target=synthetic_target,
    )
    data = conformed.get_fdata(dtype=np.float32)
    conformed_structural_mask = conform_volume(
        structural_mask_img,
        order=0,
        grid_contract=common_grid,
        synthetic_target=synthetic_target,
    )
    structural_support = conformed_structural_mask.get_fdata(dtype=np.float32) > 0.5
    if structural_support.shape != data.shape[:3] or not structural_support.any():
        raise RuntimeError("common-grid anatomical structural mask is invalid")
    if preprocessing_backend == "fmriprep":
        if bold_brain_mask_img is None:
            raise RuntimeError("authenticated fMRIPrep BOLD brain mask is unavailable")
        conformed_bold_brain_mask = conform_volume(
            bold_brain_mask_img,
            order=0,
            grid_contract=common_grid,
            synthetic_target=synthetic_target,
        )
        bold_support = conformed_bold_brain_mask.get_fdata(dtype=np.float32) > 0.5
        if bold_support.shape != data.shape[:3] or not bold_support.any():
            raise RuntimeError("common-grid fMRIPrep BOLD brain mask is invalid")
        functional_validity_mask = structural_support & bold_support
    else:
        # Debug-only data have no certified fMRIPrep coverage evidence.
        bold_support = structural_support.copy()
        functional_validity_mask = structural_support.copy()
    if not functional_validity_mask.any():
        raise RuntimeError(
            "structural and functional masks have no common-grid intersection"
        )
    smoothing_mask = functional_validity_mask

    print(
        f"[fmri] 4/6 mask-normalized spatial smoothing ({smoothing_fwhm_mm:g} mm FWHM)",
        flush=True,
    )
    gradient_before = _masked_spatial_gradient_energy(data, smoothing_mask)
    data = _spatial_smooth(data, smoothing_mask, conformed.affine, smoothing_fwhm_mm)
    gradient_after = _masked_spatial_gradient_energy(data, smoothing_mask)
    gradient_retention = (
        float(gradient_after / gradient_before) if gradient_before > 0 else 0.0
    )
    if gradient_retention <= MIN_SMOOTHING_GRADIENT_RETENTION:
        raise RuntimeError(
            "3-mm smoothing failed the mandatory spatial-gradient retention gate: "
            f"{gradient_retention:.6g} <= {MIN_SMOOTHING_GRADIENT_RETENTION:.2f}"
        )
    if not np.isfinite(data).all() or np.any(data[~smoothing_mask, :] != 0):
        raise RuntimeError("spatial smoothing produced invalid or out-of-mask signal")

    print(
        f"[fmri] 5/6 temporal filtering ({high_pass_hz!r}-{low_pass_hz!r} Hz)",
        flush=True,
    )
    data = _temporal_filter(
        data,
        tr=source_tr_value,
        high_pass_hz=high_pass_hz,
        low_pass_hz=low_pass_hz,
    )

    print(
        "[fmri] 6/6 harmonising to 128 real frames and TR=3 s (no temporal padding)",
        flush=True,
    )
    data = _resample_to_tr(data, source_tr_value, TR_SECONDS)
    data = set_num_frames(data, TARGET_FRAMES).astype(np.float32, copy=False)
    normalization_record: Optional[Dict[str, Any]] = None
    if normalize:
        data, normalization_record = _recovery_intensity_normalize(data, smoothing_mask)
    data[~smoothing_mask, :] = 0.0
    if not np.isfinite(data).all():
        raise RuntimeError("certified fMRI output contains NaN or infinity")
    observed_output_support = np.any(data != 0.0, axis=-1)
    if not np.array_equal(functional_validity_mask, observed_output_support):
        missing_signal = int(
            np.count_nonzero(functional_validity_mask & ~observed_output_support)
        )
        unexpected_signal = int(
            np.count_nonzero(~functional_validity_mask & observed_output_support)
        )
        raise RuntimeError(
            "functional-validity mask differs from exact final nonzero fMRI "
            f"support (masked all-zero voxels={missing_signal}, "
            f"out-of-mask signal voxels={unexpected_signal})"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(data, conformed.affine, conformed.header)
    out_img.header.set_data_dtype(np.float32)
    out_img.header.set_xyzt_units(xyz="mm", t="sec")
    out_img.header.set_zooms(tuple(out_img.header.get_zooms()[:3]) + (TR_SECONDS,))
    nib.save(out_img, out)
    output_sha256 = _sha256_file(out.resolve(strict=True))
    functional_validity_path = _functional_validity_mask_path(out)
    functional_mask_header = conformed_structural_mask.header.copy()
    functional_mask_header.set_data_dtype(np.uint8)
    functional_mask_img = nib.Nifti1Image(
        functional_validity_mask.astype(np.uint8),
        conformed_structural_mask.affine,
        functional_mask_header,
    )
    functional_mask_img.header.set_xyzt_units(xyz="mm")
    nib.save(functional_mask_img, functional_validity_path)
    functional_mask_evidence = _artifact_evidence(functional_validity_path)

    coregistration_path: Optional[Path] = None
    if preprocessing_backend == "fmriprep" and fmriprep_evidence is not None:
        coregistration_path = Path(fmriprep_evidence["BOLDToT1wTransform"]["Path"])
    elif coregistration_transform is not None:
        coregistration_path = out.with_name(_nifti_stem(out) + "_coreg.mat")
        np.savetxt(coregistration_path, coregistration_transform, fmt="%.12g")
    motion_path: Optional[Path] = None
    if preprocessing_backend == "fmriprep" and fmriprep_evidence is not None:
        motion_path = Path(fmriprep_evidence["MotionConfounds"]["Path"])
    elif motion_parameters is not None and apply_motion_correction:
        motion_path = _write_motion_parameters(out, motion_parameters)
    smoothing_applied = float(smoothing_fwhm_mm) > 0.0
    temporal_filtering_applied = (
        high_pass_hz is not None and float(high_pass_hz) > 0.0
    ) or (low_pass_hz is not None and float(low_pass_hz) > 0.0)
    production_pipeline_steps = [
        "T1w co-registration",
        "slice-timing correction",
        "rigid motion correction",
        "single-pass spatial harmonisation",
        "spatial smoothing",
        "temporal filtering",
        "TR/frame harmonisation",
        "recovery intensity normalization",
    ]
    if preprocessing_backend == "fmriprep":
        pipeline_name = "fMRIPrep + CONNECT-4 harmonisation"
        pipeline_steps = production_pipeline_steps
    else:
        pipeline_name = "CONNECT-4 custom-debug preprocessing"
        pipeline_steps = ["debug T1w co-registration"]
        if apply_slice_timing:
            pipeline_steps.append("debug slice-timing correction")
        if apply_motion_correction:
            pipeline_steps.append("debug rigid motion correction")
        pipeline_steps.append("debug spatial harmonisation")
        if smoothing_applied:
            pipeline_steps.append("debug spatial smoothing")
        if temporal_filtering_applied:
            pipeline_steps.append("debug temporal filtering")
        pipeline_steps.append("debug TR/frame harmonisation")
        if normalize:
            pipeline_steps.append("debug intensity normalization")
    paper_steps_complete = (
        preprocessing_backend == "fmriprep"
        and common_grid is not None
        and brain_mask_path is not None
        and fmriprep_info is not None
        and fmriprep_evidence is not None
        and fmriprep_evidence.get("EvidenceSchemaVersion")
        == FMRIPREP_EVIDENCE_SCHEMA_VERSION
        and isinstance(fmriprep_evidence.get("BOLDBrainMask"), dict)
        and fmriprep_evidence.get("SliceTimingEnabledByWrapper") is True
        and int(fmriprep_evidence.get("MotionConfoundsRows", -1))
        == int(fmri_img.shape[-1])
        and bool(apply_slice_timing)
        and bool(apply_motion_correction)
        and smoothing_applied
        and np.isclose(
            float(smoothing_fwhm_mm),
            DEFAULT_SMOOTHING_FWHM_MM,
            rtol=0.0,
            atol=1e-12,
        )
        and gradient_retention > MIN_SMOOTHING_GRADIENT_RETENTION
        and temporal_filtering_applied
        and np.isclose(
            float(high_pass_hz),
            DEFAULT_HIGH_PASS_HZ,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(low_pass_hz),
            DEFAULT_LOW_PASS_HZ,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            resolved_slice_timing_reference,
            DEFAULT_SLICE_TIMING_REFERENCE,
            rtol=0.0,
            atol=1e-12,
        )
        and post_fmriprep_target_interpolation_count == 1
    )
    provenance = {
        "PreprocessingSchemaVersion": FMRI_PREPROCESSING_SCHEMA_VERSION,
        "PaperRequiredStepsComplete": paper_steps_complete,
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
            "SpatialGrid": (
                common_grid["schema"] if common_grid is not None else "synthetic-debug"
            ),
            "SpatialSmoothingFWHMMM": float(smoothing_fwhm_mm),
            "SpatialSmoothingMethod": (
                "Gaussian signal/mask division within functional-validity mask"
            ),
            "MinimumGradientRetentionRatioExclusive": (
                MIN_SMOOTHING_GRADIENT_RETENTION
            ),
            "TemporalHighPassHz": high_pass_hz,
            "TemporalLowPassHz": low_pass_hz,
            "TemporalFilterPreservesVoxelMean": True,
            "IntensityNormalizationContract": (
                FMRI_INTENSITY_NORMALIZATION_CONTRACT if normalize else None
            ),
            "PostFMRIPrepTargetInterpolationCount": (
                post_fmriprep_target_interpolation_count
            ),
            "PostFMRIPrepIntermediateT1GridResamplingApplied": False,
            "PostFMRIPrepSpatialInterpolationOrder": 1,
            "SpatialResamplingRoute": spatial_resampling_route,
        },
        "PreprocessingBackend": preprocessing_backend,
        "FMRIPrepApplied": preprocessing_backend == "fmriprep",
        "FMRIPrepGeneratedBy": fmriprep_info,
        "FMRIPrepEvidence": fmriprep_evidence,
        "PipelineDescription": {
            "Name": pipeline_name,
            "Steps": pipeline_steps,
        },
        "SourceRepetitionTime": source_tr_value,
        "CoregistrationBackend": resolved_coregistration_backend,
        "CoregistrationTransform": (
            (
                str(coregistration_path)
                if preprocessing_backend == "fmriprep"
                else coregistration_path.name
            )
            if coregistration_path
            else None
        ),
        "CoregistrationTransformSHA256": (
            fmriprep_evidence["BOLDToT1wTransform"]["SHA256"]
            if fmriprep_evidence is not None
            else None
        ),
        "CoregistrationTransformConvention": transform_convention,
        **coregistration_quality,
        "SliceTimingApplied": bool(apply_slice_timing),
        "SliceTimingSource": slice_timing_source,
        "SliceTimingReference": resolved_slice_timing_reference,
        "MotionCorrectionApplied": bool(apply_motion_correction),
        "MotionCorrectionBackend": resolved_motion_backend,
        "MotionParameters": (
            (
                str(motion_path)
                if preprocessing_backend == "fmriprep"
                else motion_path.name
            )
            if motion_path
            else None
        ),
        "MotionParametersSHA256": (
            fmriprep_evidence["MotionConfounds"]["SHA256"]
            if fmriprep_evidence is not None
            else None
        ),
        "SpatialSmoothingApplied": smoothing_applied,
        "SpatialSmoothingFWHM": float(smoothing_fwhm_mm),
        "SpatialSmoothingMethod": (
            "Gaussian signal/mask division within functional-validity mask"
        ),
        "SmoothingGradientBefore": gradient_before,
        "SmoothingGradientAfter": gradient_after,
        "SmoothingGradientRetentionRatio": gradient_retention,
        "MinimumSmoothingGradientRetentionRatioExclusive": (
            MIN_SMOOTHING_GRADIENT_RETENTION
        ),
        "AnatomicalMask": (
            str(Path(brain_mask_path).expanduser().resolve(strict=True))
            if brain_mask_path is not None
            else None
        ),
        "AnatomicalMaskSHA256": (
            _sha256_file(Path(brain_mask_path).expanduser().resolve(strict=True))
            if brain_mask_path is not None
            else None
        ),
        "AnatomicalMaskRole": "structural labels and model conditioning only",
        "FunctionalValidityMask": {
            **functional_mask_evidence,
            "Role": "fMRI spatial support and model-loss validity",
            "Derivation": (
                "nearest-neighbour common-grid fMRIPrep BOLD brain mask "
                "intersected with nonzero structural-mask support"
                if preprocessing_backend == "fmriprep"
                else "debug-only nonzero structural-mask support"
            ),
            "CertifiedFMRIPrepCoverage": preprocessing_backend == "fmriprep",
            "SourceBOLDBrainMaskSHA256": (
                fmriprep_evidence["BOLDBrainMask"]["SHA256"]
                if fmriprep_evidence is not None
                else None
            ),
            "Shape": list(functional_validity_mask.shape),
            "ValidVoxelCount": int(np.count_nonzero(functional_validity_mask)),
            "StructuralVoxelCount": int(np.count_nonzero(structural_support)),
            "BOLDSupportVoxelCount": int(np.count_nonzero(bold_support)),
            "StructuralVoxelsExcludedFromLoss": int(
                np.count_nonzero(structural_support & ~functional_validity_mask)
            ),
            "AppliedToSpatialSmoothing": True,
            "AppliedToIntensityNormalization": bool(normalize),
            "OutsideMaskForcedZero": True,
            "EqualsExactFinalNonzeroSupport": True,
        },
        "CommonGridContract": (
            {
                "Path": common_grid["contract_path"],
                "SHA256": common_grid["contract_sha256"],
                "Schema": common_grid["schema"],
                "ReferencePath": common_grid["reference_path"],
                "ReferenceSHA256": common_grid["reference"]["sha256"],
                "AnatomicalShape": common_grid["anatomical_shape"],
                "ArchitectureShape": common_grid["architecture_shape"],
                "ArchitecturePadding": common_grid["architecture_padding"],
                "MatrixSizeReportedByPaper": False,
            }
            if common_grid is not None
            else None
        ),
        "TemporalFilteringApplied": temporal_filtering_applied,
        "TemporalFilterPreservesVoxelMean": True,
        "TemporalHighPassHz": high_pass_hz,
        "TemporalLowPassHz": low_pass_hz,
        "OutputShape": list(data.shape),
        "OutputSHA256": output_sha256,
        "RepetitionTime": TR_SECONDS,
        "VoxelSize": [3.0, 3.0, 3.0],
        "TemporalZScore": False,
        "IntensityNormalizationApplied": bool(normalize),
        "IntensityNormalization": normalization_record,
        "TemporalPaddingApplied": False,
        "SourceMetadata": (
            fmriprep_evidence["BOLDMetadata"]["Path"]
            if fmriprep_evidence is not None
            else str(resolved_metadata_path)
            if resolved_metadata_path
            else None
        ),
    }
    with open(_sidecar_path(out), "w") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    print(
        f"[fmri] {fmri_path} -> {out} shape={data.shape} TR={TR_SECONDS}s", flush=True
    )
    return str(out)


def _optional_frequency(value: str) -> Optional[float]:
    lowered = value.strip().lower()
    return None if lowered in {"none", "off", "null"} else float(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fmri",
        required=True,
        help="fMRIPrep space-T1w desc-preproc_bold derivative",
    )
    parser.add_argument(
        "--t1",
        required=True,
        help="same-subject fMRIPrep desc-preproc_T1w transform target",
    )
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--metadata",
        default=None,
        help="BIDS JSON sidecar (auto-discovered by default)",
    )
    parser.add_argument(
        "--fmriprep-dataset-description",
        default=None,
        help="derivative-root dataset_description.json (required in production)",
    )
    parser.add_argument(
        "--fmriprep-execution-record",
        default=None,
        help="successful preprocessing.run_fmriprep execution JSON (required in production)",
    )
    parser.add_argument(
        "--source-acquisition-identity",
        default=None,
        help=(
            "externally SHA-pinned raw T1w/BOLD/SynthSeg identity JSON "
            "(required in production)"
        ),
    )
    parser.add_argument(
        "--source-acquisition-identity-sha256",
        default=None,
        help="expected SHA-256 of --source-acquisition-identity",
    )
    parser.add_argument(
        "--motion-confounds",
        default=None,
        help="matching fMRIPrep desc-confounds_timeseries.tsv (required in production)",
    )
    parser.add_argument(
        "--coregistration-transform",
        default=None,
        help=(
            "matching fMRIPrep from-boldref_to-T1w (or legacy from-scanner_to-T1w) "
            "image transform (required in production)"
        ),
    )
    parser.add_argument(
        "--brain-mask",
        default=None,
        help=(
            "anatomical label/mask NIfTI required for structural support and "
            "model conditioning; production smoothing/loss use the separately "
            "authenticated fMRIPrep BOLD brain mask intersection"
        ),
    )
    parser.add_argument(
        "--common-grid-contract",
        default=None,
        help="hash-bound T1-only cohort-common grid JSON required in production",
    )
    parser.add_argument(
        "--common-grid-contract-sha256",
        default=None,
        help="expected SHA-256 of the common-grid contract",
    )
    parser.add_argument(
        "--source-tr",
        type=float,
        default=None,
        help="debug-only TR restatement; production reads hash-bound metadata/header",
    )
    parser.add_argument(
        "--preprocessing-backend",
        choices=("fmriprep", "custom-debug"),
        default="fmriprep",
        help="production requires fmriprep; custom-debug outputs are rejected by training",
    )
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument(
        "--slice-order",
        choices=(
            "ascending",
            "descending",
            "interleaved-ascending",
            "interleaved-descending",
        ),
        default="interleaved-ascending",
        help="fallback used only when BIDS/NIfTI SliceTiming is unavailable",
    )
    parser.add_argument(
        "--slice-timing-reference",
        type=float,
        default=None,
        help="fraction of TR (defaults to BIDS metadata, then 0.5)",
    )
    parser.add_argument(
        "--coregistration-backend",
        choices=("auto", "flirt", "python", "header"),
        default="auto",
        help="FLIRT or SciPy MI estimation; header is an explicit debug-only fallback",
    )
    parser.add_argument(
        "--no-slice-timing", action="store_true", help="explicit ablation only"
    )
    parser.add_argument(
        "--no-motion-correction", action="store_true", help="explicit ablation only"
    )
    parser.add_argument(
        "--motion-backend", choices=("auto", "mcflirt", "python"), default="auto"
    )
    parser.add_argument(
        "--mcflirt",
        action="store_true",
        help="backward-compatible alias for --motion-backend mcflirt",
    )
    parser.add_argument(
        "--smoothing-fwhm-mm", type=float, default=DEFAULT_SMOOTHING_FWHM_MM
    )
    parser.add_argument(
        "--high-pass-hz", type=_optional_frequency, default=DEFAULT_HIGH_PASS_HZ
    )
    parser.add_argument(
        "--low-pass-hz", type=_optional_frequency, default=DEFAULT_LOW_PASS_HZ
    )
    arguments = parser.parse_args()
    preprocess_fmri(
        arguments.fmri,
        arguments.t1,
        arguments.out_path,
        mcflirt=arguments.mcflirt,
        normalize=True,
        metadata_path=arguments.metadata,
        fmriprep_dataset_description_path=arguments.fmriprep_dataset_description,
        fmriprep_execution_record_path=arguments.fmriprep_execution_record,
        source_acquisition_identity_path=arguments.source_acquisition_identity,
        source_acquisition_identity_sha256=(
            arguments.source_acquisition_identity_sha256
        ),
        motion_confounds_path=arguments.motion_confounds,
        coregistration_transform_path=arguments.coregistration_transform,
        brain_mask_path=arguments.brain_mask,
        common_grid_contract_path=arguments.common_grid_contract,
        common_grid_contract_sha256=arguments.common_grid_contract_sha256,
        source_tr=arguments.source_tr,
        slice_axis=arguments.slice_axis,
        slice_order=arguments.slice_order,
        slice_timing_reference=arguments.slice_timing_reference,
        preprocessing_backend=arguments.preprocessing_backend,
        coregistration_backend=arguments.coregistration_backend,
        motion_backend=arguments.motion_backend,
        apply_slice_timing=not arguments.no_slice_timing,
        apply_motion_correction=not arguments.no_motion_correction,
        smoothing_fwhm_mm=arguments.smoothing_fwhm_mm,
        high_pass_hz=arguments.high_pass_hz,
        low_pass_hz=arguments.low_pass_hz,
    )
