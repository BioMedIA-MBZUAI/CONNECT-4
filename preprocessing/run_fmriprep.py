"""Run fMRIPrep with CONNECT-4's auditable production requirements.

This wrapper is intentionally small: it fixes the subject-T1w output space,
forbids disabling slice-timing correction, records the exact successful
command, and hashes the fMRIPrep artifacts consumed by CONNECT-4.  The record
is written only after fMRIPrep exits successfully and the required derivative
classes are present. An externally SHA-pinned source-acquisition record is
authenticated before execution and after success; derivative ``Sources`` must
bind the exact raw T1w and BOLD named by that record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .preprocess_fmri import (
    _artifact_evidence,
    _fmriprep_generated_by,
    _load_json_object,
    _nifti_stem,
)
from .source_acquisition_identity import (
    SourceAcquisitionError,
    derivative_source_relative_paths,
    load_authenticated_json_artifact,
    load_fmriprep_runtime_identity,
    load_source_acquisition_identity,
)


WRAPPER_NAME = "preprocessing.run_fmriprep"
FMRIPREP_EXECUTION_SCHEMA_VERSION = "connect4-fmriprep-execution-v4"
FIXED_SLICE_TIMING_REFERENCE = 0.5
_ALLOWED_FLAG_ARGUMENTS = {"--fs-no-reconall", "--notrack"}
_ALLOWED_POSITIVE_INTEGER_ARGUMENTS = {
    "--mem-mb",
    "--nthreads",
    "--omp-nthreads",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _participant_label(value: str) -> str:
    label = str(value).strip()
    if label.startswith("sub-"):
        label = label[4:]
    if re.fullmatch(r"[A-Za-z0-9]+", label) is None:
        raise ValueError(f"Invalid BIDS participant label: {value!r}")
    return label


def _scan_id(value: str) -> str:
    scan = str(value).strip()
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,253}[A-Za-z0-9])?", scan) is None:
        raise ValueError(f"Invalid CONNECT-4 scan ID: {value!r}")
    return scan


def _validate_extra_args(extra_args: Sequence[str]) -> List[str]:
    """Parse a positive grammar containing resource-only production options."""

    arguments = [str(value) for value in extra_args]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    normalized: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(
                f"fMRIPrep production arguments may not contain positional token {token!r}"
            )
        option, separator, inline_value = token.partition("=")
        if option in seen:
            raise ValueError(f"duplicate fMRIPrep production option: {option}")
        seen.add(option)
        if option in _ALLOWED_FLAG_ARGUMENTS:
            if separator:
                raise ValueError(f"{option} is a flag and cannot take a value")
            normalized.append(option)
            index += 1
            continue
        if option not in _ALLOWED_POSITIVE_INTEGER_ARGUMENTS:
            raise ValueError(
                f"fMRIPrep option is not in the production allowlist: {option}"
            )
        if separator:
            value = inline_value
        else:
            index += 1
            if index >= len(arguments) or arguments[index].startswith("--"):
                raise ValueError(f"{option} requires one positive integer")
            value = arguments[index]
        if re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise ValueError(f"{option} requires one positive integer")
        normalized.extend((option, value))
        index += 1
    return normalized


def _fresh_derivatives_path(value: str, bids_root: Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise ValueError("derivatives root must be an absolute canonical new path")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("derivatives parent directory does not exist") from exc
    if parent != requested.parent:
        raise ValueError("derivatives parent directory aliases another path")
    if os.path.lexists(requested):
        raise FileExistsError(
            f"certified derivatives generation must not already exist: {requested}"
        )
    if requested == bids_root or bids_root in requested.parents:
        raise ValueError("fresh derivatives root must be outside the raw BIDS root")
    return requested


def _version_probe(executable: Path, expected_version: str) -> dict[str, object]:
    command = [str(executable), "--version"]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    stdout = bytes(completed.stdout or b"")
    stderr = bytes(completed.stderr or b"")
    if len(stdout) > 65536 or len(stderr) > 65536:
        raise RuntimeError("fMRIPrep version probe exceeded its bounded output")
    combined = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    match = re.search(
        r"(?i)\bfmriprep(?:\s+version)?\s+v?"
        r"([0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9._+-]*)?)",
        combined,
    )
    observed = match.group(1) if match is not None else None
    if completed.returncode != 0 or observed != expected_version:
        raise RuntimeError(
            "fMRIPrep executable version probe does not match the runtime authority"
        )
    return {
        "Command": command,
        "ReturnCode": int(completed.returncode),
        "ObservedVersion": observed,
        "StdoutSHA256": hashlib.sha256(stdout).hexdigest(),
        "StderrSHA256": hashlib.sha256(stderr).hexdigest(),
        "Stdout": stdout.decode("utf-8", errors="replace"),
        "Stderr": stderr.decode("utf-8", errors="replace"),
    }


def _required_derivative_artifacts(
    derivatives_root: Path,
    participant: str,
    raw_bold: Path,
) -> Dict[str, List[Path]]:
    subject_root = derivatives_root / f"sub-{participant}"
    if not subject_root.is_dir():
        raise RuntimeError(
            f"fMRIPrep succeeded but produced no sub-{participant} derivative directory"
        )
    subject_directories = sorted(
        path.name for path in derivatives_root.glob("sub-*") if path.is_dir()
    )
    if subject_directories != [f"sub-{participant}"]:
        raise RuntimeError(
            "fresh fMRIPrep derivatives contain an unexpected participant tree"
        )
    files = [path for path in subject_root.rglob("*") if path.is_file()]
    raw_stem = _nifti_stem(raw_bold)
    if not raw_stem.endswith("_bold"):
        raise RuntimeError("authenticated raw BOLD has an invalid suffix")
    run_prefix = raw_stem.removesuffix("_bold")

    all_preprocessed_bold = [
        path
        for path in files
        if "space-T1w" in path.name and "desc-preproc_bold.nii" in path.name
    ]
    if any(
        not path.name.startswith(run_prefix + "_") for path in all_preprocessed_bold
    ):
        raise RuntimeError(
            "fresh fMRIPrep derivatives contain output for another BOLD run"
        )

    def matching(predicate: Any) -> List[Path]:
        return sorted((path for path in files if predicate(path)), key=str)

    groups = {
        "T1w-space preprocessed BOLD": matching(
            lambda path: (
                path.name.startswith(run_prefix + "_")
                and "space-T1w" in path.name
                and "desc-preproc_bold.nii" in path.name
            )
        ),
        "T1w-space BOLD metadata": matching(
            lambda path: (
                path.name.startswith(run_prefix + "_")
                and "space-T1w" in path.name
                and path.name.endswith("desc-preproc_bold.json")
            )
        ),
        "T1w-space BOLD brain mask": matching(
            lambda path: (
                path.name.startswith(run_prefix + "_")
                and "space-T1w" in path.name
                and "desc-brain_mask.nii" in path.name
            )
        ),
        "motion confounds": matching(
            lambda path: (
                path.name.startswith(run_prefix + "_")
                and (
                    path.name.endswith("desc-confounds_timeseries.tsv")
                    or path.name.endswith("desc-confounds_regressors.tsv")
                )
            )
        ),
        "BOLDref-to-T1w transform": matching(
            lambda path: (
                path.name.startswith(run_prefix + "_")
                and (
                    "_from-boldref_to-T1w_" in path.name
                    or "_from-scanner_to-T1w_" in path.name
                )
                and "mode-image" in path.name.lower()
                and "xfm" in path.name.lower()
            )
        ),
        "preprocessed T1w reference": matching(
            lambda path: (
                path.name.endswith("desc-preproc_T1w.nii.gz")
                or path.name.endswith("desc-preproc_T1w.nii")
            )
        ),
        "preprocessed T1w metadata": matching(
            lambda path: path.name.endswith("desc-preproc_T1w.json")
        ),
    }
    invalid_counts = [
        f"{label}={len(matches)}"
        for label, matches in groups.items()
        if len(matches) != 1
    ]
    if invalid_counts:
        raise RuntimeError(
            "fMRIPrep returned success but the exact-run evidence cardinality differs: "
            + ", ".join(invalid_counts)
        )
    bold_masks = groups["T1w-space BOLD brain mask"]
    for bold_path in groups["T1w-space preprocessed BOLD"]:
        bold_stem = _nifti_stem(bold_path)
        suffix = "_desc-preproc_bold"
        if not bold_stem.endswith(suffix):
            raise RuntimeError(
                f"unexpected fMRIPrep preprocessed BOLD name: {bold_path}"
            )
        expected_mask_stem = bold_stem[: -len(suffix)] + "_desc-brain_mask"
        paired_masks = [
            path
            for path in bold_masks
            if path.parent == bold_path.parent
            and _nifti_stem(path) == expected_mask_stem
        ]
        if len(paired_masks) != 1:
            raise RuntimeError(
                "fMRIPrep returned success but the exact same-run T1w-space "
                f"BOLD brain mask is missing or ambiguous for {bold_path}"
            )
    corrected_metadata = [
        path
        for path in groups["T1w-space BOLD metadata"]
        if _load_json_object(path, "fMRIPrep BOLD metadata").get("SliceTimingCorrected")
        is True
    ]
    if not corrected_metadata:
        raise RuntimeError(
            "fMRIPrep returned success but no T1w-space BOLD metadata confirms "
            "SliceTimingCorrected=true"
        )
    return groups


def _verify_derivative_source_bindings(
    groups: Dict[str, List[Path]],
    *,
    source_acquisition_identity: Dict[str, Any],
    bids_root: Path,
    artifact_by_path: Dict[str, Dict[str, Any]],
) -> None:
    """Require fMRIPrep derivative metadata to name the exact raw inputs."""
    raw_t1 = Path(str(source_acquisition_identity["raw_t1"]["path"]))
    raw_bold = Path(str(source_acquisition_identity["raw_bold"]["path"]))
    try:
        raw_t1_relative = raw_t1.relative_to(bids_root).as_posix()
        raw_bold_relative = raw_bold.relative_to(bids_root).as_posix()
    except ValueError as exc:
        raise RuntimeError("source-acquisition inputs escape the BIDS root") from exc

    def exact_sources(paths: Sequence[Path], label: str) -> set[str]:
        if len(paths) != 1:
            raise RuntimeError(f"fMRIPrep {label} cardinality is not exact")
        for path in paths:
            try:
                evidence = artifact_by_path[str(path.resolve(strict=True))]
                metadata, _ = load_authenticated_json_artifact(
                    path,
                    expected_sha256=str(evidence["SHA256"]),
                    expected_size=int(evidence["SizeBytes"]),
                    label=f"fMRIPrep {label}",
                )
                if (
                    label == "BOLD metadata"
                    and metadata.get("SliceTimingCorrected") is not True
                ):
                    raise SourceAcquisitionError(
                        "fMRIPrep BOLD metadata does not confirm slice timing"
                    )
                return derivative_source_relative_paths(metadata)
            except (KeyError, SourceAcquisitionError) as exc:
                raise RuntimeError(
                    f"fMRIPrep {label} has invalid BIDS Sources: {path}"
                ) from exc
        raise AssertionError("unreachable empty derivative metadata group")

    bold_sources = exact_sources(groups["T1w-space BOLD metadata"], "BOLD metadata")
    t1_sources = exact_sources(groups["preprocessed T1w metadata"], "T1w metadata")
    if bold_sources != {raw_bold_relative}:
        raise RuntimeError(
            "fMRIPrep BOLD derivative metadata does not name exactly the authenticated raw BOLD"
        )
    if t1_sources != {raw_t1_relative}:
        raise RuntimeError(
            "fMRIPrep T1w derivative metadata does not name exactly the authenticated raw T1w"
        )


def _deduplicate(paths: Iterable[Path]) -> List[Path]:
    unique: Dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve(strict=True)
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for execution-record publication")
    descriptor = os.open(temporary, flags | nofollow, 0o440)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to replace execution record: {path}")
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(descriptor)


def run_fmriprep(
    bids_root: str,
    derivatives_root: str,
    participant_label: str,
    *,
    scan_id: str | None = None,
    source_acquisition_identity_path: str | None = None,
    source_acquisition_identity_sha256: str | None = None,
    fmriprep_runtime_identity_path: str | None = None,
    fmriprep_runtime_identity_sha256: str | None = None,
    fmriprep_executable: str = "fmriprep",
    extra_args: Sequence[str] = (),
) -> str:
    """Run participant-level fMRIPrep and return its execution-record path."""
    bids = Path(bids_root).resolve(strict=True)
    if not bids.is_dir():
        raise NotADirectoryError(f"BIDS root is not a directory: {bids}")
    participant = _participant_label(participant_label)
    if scan_id is None:
        raise ValueError("paper-certified fMRIPrep requires an exact scan ID")
    scan = _scan_id(scan_id)
    derivatives = _fresh_derivatives_path(derivatives_root, bids)
    additions = _validate_extra_args(extra_args)
    if not source_acquisition_identity_path or not source_acquisition_identity_sha256:
        raise ValueError(
            "paper-certified fMRIPrep requires an externally SHA-pinned "
            "source-acquisition identity"
        )
    try:
        source_identity_before, source_evidence_before = (
            load_source_acquisition_identity(
                Path(source_acquisition_identity_path),
                expected_sha256=source_acquisition_identity_sha256,
                expected_bids_root=bids,
                expected_participant=participant,
                expected_scan_id=scan,
            )
        )
    except SourceAcquisitionError as exc:
        raise ValueError("source-acquisition identity admission failed") from exc

    executable_value = shutil.which(fmriprep_executable)
    if executable_value is None:
        explicit = Path(fmriprep_executable).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(
                f"Cannot find fMRIPrep executable: {fmriprep_executable}"
            )
        executable_value = str(explicit.resolve(strict=True))
    executable = Path(executable_value).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PermissionError(f"fMRIPrep executable is not executable: {executable}")
    if not fmriprep_runtime_identity_path or not fmriprep_runtime_identity_sha256:
        raise ValueError(
            "paper-certified fMRIPrep requires an externally SHA-pinned runtime identity"
        )
    try:
        runtime_before, runtime_evidence_before = load_fmriprep_runtime_identity(
            Path(fmriprep_runtime_identity_path),
            expected_sha256=fmriprep_runtime_identity_sha256,
            expected_executable=executable,
        )
    except SourceAcquisitionError as exc:
        raise ValueError("fMRIPrep runtime identity admission failed") from exc
    version_probe_before = _version_probe(
        executable, str(runtime_before["fmriprep_version"])
    )

    bold_entities = dict(source_identity_before["bids_entities"]["raw_bold"])
    input_inventory = dict(source_identity_before["bids_input_inventory"])
    scan_identity = {
        "ScanID": scan,
        "ParticipantLabel": participant,
        "Session": bold_entities.get("ses"),
        "Task": bold_entities["task"],
        "Run": bold_entities.get("run"),
        "Acquisition": bold_entities.get("acq"),
        "Direction": bold_entities.get("dir"),
        "Echo": bold_entities.get("echo"),
        "RawBOLDRelativePath": input_inventory["raw_bold_relative_path"],
        "RawT1wRelativePath": input_inventory["raw_t1_relative_path"],
    }

    os.mkdir(derivatives, mode=0o750)
    if list(derivatives.iterdir()):
        raise RuntimeError(
            "fresh derivatives root was not empty after exclusive creation"
        )
    generation_id = secrets.token_hex(32)
    work_path = derivatives / "connect4-work"

    command = [
        str(executable),
        str(bids),
        str(derivatives),
        "participant",
        "--participant-label",
        participant,
        "--output-spaces",
        "T1w",
        "--slice-time-ref",
        str(FIXED_SLICE_TIMING_REFERENCE),
        "--level",
        "full",
        "--work-dir",
        str(work_path),
        *additions,
    ]
    logs = derivatives / "logs"
    logs.mkdir(mode=0o750, exist_ok=False)
    base = f"connect4-fmriprep_{scan}"
    stdout_path = logs / f"{base}.stdout.log"
    stderr_path = logs / f"{base}.stderr.log"
    record_path = logs / f"{base}_execution.json"

    started = _utc_now()
    with stdout_path.open("x") as stdout_stream, stderr_path.open("x") as stderr_stream:
        completed = subprocess.run(
            command,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            check=False,
        )
    finished = _utc_now()
    if completed.returncode != 0:
        raise RuntimeError(
            "fMRIPrep failed with return code "
            f"{completed.returncode}; see {stderr_path}"
        )

    try:
        source_identity_after, source_evidence_after = load_source_acquisition_identity(
            Path(source_acquisition_identity_path),
            expected_sha256=source_acquisition_identity_sha256,
            expected_bids_root=bids,
            expected_participant=participant,
            expected_scan_id=scan,
        )
    except SourceAcquisitionError as exc:
        raise RuntimeError(
            "source-acquisition identity changed during fMRIPrep execution"
        ) from exc
    if (
        source_identity_after != source_identity_before
        or source_evidence_after != source_evidence_before
    ):
        raise RuntimeError("source-acquisition identity changed during execution")
    try:
        runtime_after, runtime_evidence_after = load_fmriprep_runtime_identity(
            Path(fmriprep_runtime_identity_path),
            expected_sha256=fmriprep_runtime_identity_sha256,
            expected_executable=executable,
        )
    except SourceAcquisitionError as exc:
        raise RuntimeError(
            "fMRIPrep runtime identity changed during execution"
        ) from exc
    if (
        runtime_after != runtime_before
        or runtime_evidence_after != runtime_evidence_before
    ):
        raise RuntimeError("fMRIPrep runtime identity changed during execution")
    version_probe_after = _version_probe(
        executable, str(runtime_after["fmriprep_version"])
    )

    dataset_description_path = derivatives / "dataset_description.json"
    description = _load_json_object(
        dataset_description_path, "fMRIPrep derivative dataset_description.json"
    )
    if str(description.get("DatasetType", "")).lower() != "derivative":
        raise RuntimeError(
            "fMRIPrep dataset_description DatasetType is not 'derivative'"
        )
    generated_by = _fmriprep_generated_by(description)
    if str(generated_by.get("Version", "")) != str(runtime_after["fmriprep_version"]):
        raise RuntimeError(
            "fMRIPrep derivative GeneratedBy version differs from the pinned runtime"
        )
    raw_bold = Path(str(source_identity_after["raw_bold"]["path"]))
    groups = _required_derivative_artifacts(derivatives, participant, raw_bold)
    evidence_paths = _deduplicate(
        [dataset_description_path, stdout_path, stderr_path]
        + [path for matches in groups.values() for path in matches]
    )
    artifacts = [_artifact_evidence(path, derivatives) for path in evidence_paths]
    artifact_by_path = {str(artifact["Path"]): artifact for artifact in artifacts}
    _verify_derivative_source_bindings(
        groups,
        source_acquisition_identity=source_identity_after,
        bids_root=bids,
        artifact_by_path=artifact_by_path,
    )

    record: Dict[str, Any] = {
        "SchemaVersion": FMRIPREP_EXECUTION_SCHEMA_VERSION,
        "Wrapper": WRAPPER_NAME,
        "Success": True,
        "ReturnCode": int(completed.returncode),
        "StartedAtUTC": started,
        "CompletedAtUTC": finished,
        "Command": command,
        "BIDSRoot": str(bids),
        "DerivativesRoot": str(derivatives),
        "WorkRoot": str(work_path),
        "ScanIdentity": scan_identity,
        "ParticipantLabel": participant,
        "OutputSpaces": ["T1w"],
        "ExtraArguments": additions,
        "IgnoredFeatures": [],
        "SliceTimingEnabledByWrapper": True,
        "SliceTimingReference": FIXED_SLICE_TIMING_REFERENCE,
        "FMRIPrepGeneratedBy": generated_by,
        "FreshDerivativesGeneration": {
            "GenerationID": generation_id,
            "ScanID": scan,
            "Path": str(derivatives),
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
        "ExecutableIdentity": runtime_after["executable"],
        "RuntimeIdentity": {
            "Artifact": {
                "Path": runtime_evidence_after["path"],
                "SHA256": runtime_evidence_after["sha256"],
                "SizeBytes": runtime_evidence_after["size_bytes"],
                "RecordSHA256": runtime_evidence_after["record_sha256"],
            },
            "Identity": runtime_after,
            "VerifiedBeforeExecution": True,
            "VerifiedAfterExecution": True,
        },
        "VersionProbeBeforeExecution": version_probe_before,
        "VersionProbeAfterExecution": version_probe_after,
        "SourceAcquisitionIdentity": {
            "Artifact": {
                "Path": source_evidence_after["path"],
                "SHA256": source_evidence_after["sha256"],
                "SizeBytes": source_evidence_after["size_bytes"],
                "RecordSHA256": source_evidence_after["record_sha256"],
            },
            "Identity": source_identity_after,
            "VerifiedBeforeExecution": True,
            "VerifiedAfterExecution": True,
        },
        "StdoutLog": _artifact_evidence(stdout_path, derivatives),
        "StderrLog": _artifact_evidence(stderr_path, derivatives),
        "Artifacts": artifacts,
    }
    _write_new_json(record_path, record)
    print(f"[fmriprep] verified execution record -> {record_path}", flush=True)
    return str(record_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--derivatives-root", required=True)
    parser.add_argument("--participant-label", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--source-acquisition-identity", required=True)
    parser.add_argument("--source-acquisition-identity-sha256", required=True)
    parser.add_argument("--fmriprep-runtime-identity", required=True)
    parser.add_argument("--fmriprep-runtime-identity-sha256", required=True)
    parser.add_argument("--fmriprep-executable", default="fmriprep")
    parser.add_argument(
        "fmriprep_args",
        nargs=argparse.REMAINDER,
        help="additional fMRIPrep arguments after '--'",
    )
    arguments = parser.parse_args()
    run_fmriprep(
        arguments.bids_root,
        arguments.derivatives_root,
        arguments.participant_label,
        scan_id=arguments.scan_id,
        source_acquisition_identity_path=arguments.source_acquisition_identity,
        source_acquisition_identity_sha256=(
            arguments.source_acquisition_identity_sha256
        ),
        fmriprep_runtime_identity_path=arguments.fmriprep_runtime_identity,
        fmriprep_runtime_identity_sha256=(arguments.fmriprep_runtime_identity_sha256),
        fmriprep_executable=arguments.fmriprep_executable,
        extra_args=arguments.fmriprep_args,
    )


if __name__ == "__main__":
    main()
