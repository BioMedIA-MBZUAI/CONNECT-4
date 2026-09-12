"""Fail-closed source-acquisition identity for paper-certified preprocessing.

The contract joins the exact raw BIDS T1w and BOLD bytes to an externally
produced SynthSeg parcellation.  It is deliberately independent of fMRIPrep:
the same externally SHA-pinned record is authenticated before fMRIPrep starts,
again when its derivatives are consumed, and before structural/SynthSeg
preprocessing begins.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping


SOURCE_ACQUISITION_SCHEMA = "connect4-source-acquisition-identity-v2"
BIDS_INPUT_INVENTORY_SCHEMA = "connect4-bids-input-inventory-v1"
BIDS_INPUT_INVENTORY_SCOPE = "dataset-root-files-and-exact-participant-tree"
FMRIPREP_RUNTIME_IDENTITY_SCHEMA = "connect4-fmriprep-runtime-identity-v1"
FMRIPREP_RUNTIME_IDENTITY_PURPOSE = "paper-certified-fmriprep-runtime"
SYNTHSEG_PROVENANCE_SCHEMA = "connect4-synthseg-parcellation-provenance-v1"
SOURCE_ACQUISITION_PURPOSE = "paper-certified-fmriprep-source-acquisition"
STRUCTURAL_SOURCE_AUTHORITY_SCHEMA = "connect4-structural-source-authority-v2"
STRUCTURAL_SOURCE_AUTHORITY_PURPOSE = "target-blind-t1-synthseg-source-authority"
STRUCTURAL_SOURCE_BINDING_SCHEMA = "connect4-structural-source-binding-v2"
STRUCTURAL_SOURCE_PROJECTION_SCHEMA = "connect4-structural-source-projection-v2"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024 * 1024
MAX_BIDS_INPUT_FILES = 100_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SCAN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,253}[A-Za-z0-9])?$")
FMRIPREP_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[A-Za-z0-9._+-]*)?$")


class SourceAcquisitionError(RuntimeError):
    """Raised when source bytes cannot share one authenticated identity."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _canonical_path(path: Path, *, label: str, directory: bool = False) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise SourceAcquisitionError(f"{label} must be an absolute canonical path")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise SourceAcquisitionError(f"{label} is missing: {requested}") from exc
    if resolved != requested:
        raise SourceAcquisitionError(f"{label} aliases another path")
    if directory and not requested.is_dir():
        raise SourceAcquisitionError(f"{label} is not a directory")
    return requested


def _snapshot_file_payload(
    path: Path,
    *,
    label: str,
    limit: int,
    capture_payload: bool,
    allow_empty: bool = False,
) -> tuple[dict[str, object], bytes | None]:
    source = _canonical_path(path, label=label)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise SourceAcquisitionError("O_NOFOLLOW is required for source admission")
    descriptor = os.open(source, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < (0 if allow_empty else 1)
            or before.st_size > limit
        ):
            raise SourceAcquisitionError(
                f"{label} must be one bounded, non-hardlinked regular file"
            )
        digest = hashlib.sha256()
        payload = bytearray() if capture_payload else None
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise SourceAcquisitionError(f"{label} was truncated")
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SourceAcquisitionError(f"{label} grew during authentication")
        after = os.fstat(descriptor)
        current = os.lstat(source)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
            )

        if identity(before) != identity(after) or identity(after) != identity(current):
            raise SourceAcquisitionError(f"{label} changed during authentication")
        return (
            {
                "path": str(source),
                "sha256": digest.hexdigest(),
                "size_bytes": int(after.st_size),
            },
            bytes(payload) if payload is not None else None,
        )
    finally:
        os.close(descriptor)


def _snapshot_file(
    path: Path, *, label: str, limit: int, allow_empty: bool = False
) -> dict[str, object]:
    evidence, _ = _snapshot_file_payload(
        path,
        label=label,
        limit=limit,
        capture_payload=False,
        allow_empty=allow_empty,
    )
    return evidence


def snapshot_file_artifact(path: Path, *, label: str) -> dict[str, object]:
    """Return the stable byte identity of one bounded, canonical runtime file."""

    return _snapshot_file(Path(path), label=label, limit=MAX_SOURCE_BYTES)


def _bids_input_paths(bids_root: Path, participant: str) -> list[Path]:
    """Enumerate every file fMRIPrep may read for one isolated participant.

    BIDS inheritance can draw metadata from regular files at the dataset root;
    field maps, SBRefs, scans/events tables, and modality sidecars live below the
    participant directory.  Both scopes are closed here.  Other participant
    trees are deliberately excluded because the wrapper fixes one participant.
    """

    root = _canonical_path(bids_root, label="BIDS input root", directory=True)
    subject_root = _canonical_path(
        root / f"sub-{participant}",
        label="BIDS participant input root",
        directory=True,
    )
    paths: list[Path] = []
    for child in root.iterdir():
        info = os.lstat(child)
        if stat.S_ISLNK(info.st_mode):
            raise SourceAcquisitionError(
                f"BIDS dataset-root input may not be a symlink: {child}"
            )
        if stat.S_ISREG(info.st_mode):
            paths.append(child)

    for directory, directory_names, file_names in os.walk(
        subject_root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        for name in list(directory_names):
            candidate = current / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SourceAcquisitionError(
                    f"BIDS participant tree contains an unsafe directory: {candidate}"
                )
        for name in file_names:
            candidate = current / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SourceAcquisitionError(
                    f"BIDS participant tree contains an unsafe file: {candidate}"
                )
            paths.append(candidate)
    unique = sorted(set(paths), key=lambda value: value.relative_to(root).as_posix())
    if not unique or len(unique) > MAX_BIDS_INPUT_FILES:
        raise SourceAcquisitionError(
            "BIDS input inventory is empty or exceeds its bounded file count"
        )
    return unique


def _nifti_relative_matches(relative: str, suffix: str) -> bool:
    return relative.endswith(f"_{suffix}.nii") or relative.endswith(f"_{suffix}.nii.gz")


def _effective_bold_metadata(
    bids_root: Path,
    participant: str,
    raw_bold: Path,
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Resolve and bind the BIDS-inherited metadata needed for slice timing."""

    bold_entities = bids_entities(raw_bold, suffix="bold")
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for relative, evidence in artifacts.items():
        if not relative.endswith("_bold.json"):
            continue
        path = bids_root / relative
        try:
            entities = bids_entities(path, suffix="bold")
        except SourceAcquisitionError:
            continue
        candidate_entities = {
            key: value for key, value in entities.items() if key != "suffix"
        }
        if any(
            bold_entities.get(key) != value for key, value in candidate_entities.items()
        ):
            continue
        try:
            metadata, _ = load_authenticated_json_artifact(
                path,
                expected_sha256=str(evidence["sha256"]),
                expected_size=int(evidence["size_bytes"]),
                label=f"BIDS inherited BOLD metadata {relative}",
            )
        except (KeyError, SourceAcquisitionError) as exc:
            raise SourceAcquisitionError(
                f"BIDS inherited BOLD metadata is invalid: {relative}"
            ) from exc
        candidates.append(
            (
                len(candidate_entities),
                len(PurePosixPath(relative).parts),
                relative,
                metadata,
            )
        )
    if not candidates:
        raise SourceAcquisitionError(
            "BIDS inputs provide no metadata sidecar applicable to the exact raw BOLD"
        )
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    effective: dict[str, Any] = {}
    sources: list[str] = []
    for _specificity, _depth, relative, metadata in candidates:
        effective.update(metadata)
        sources.append(relative)
    repetition_time = effective.get("RepetitionTime")
    slice_timing = effective.get("SliceTiming")
    if (
        isinstance(repetition_time, bool)
        or not isinstance(repetition_time, (int, float))
        or not float(repetition_time) > 0
        or not isinstance(slice_timing, list)
        or len(slice_timing) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) < float(repetition_time)
            for value in slice_timing
        )
    ):
        raise SourceAcquisitionError(
            "effective BIDS BOLD metadata must provide valid RepetitionTime and SliceTiming"
        )
    return {
        "source_files": sources,
        "repetition_time_seconds": float(repetition_time),
        "slice_timing_sha256": canonical_sha256(
            [float(value) for value in slice_timing]
        ),
        "slice_count": len(slice_timing),
        "phase_encoding_direction": effective.get("PhaseEncodingDirection"),
        "total_readout_time_seconds": effective.get("TotalReadoutTime"),
    }


def snapshot_bids_input_inventory(
    bids_root: Path,
    *,
    participant: str,
    raw_t1: Path,
    raw_bold: Path,
) -> dict[str, Any]:
    """Hash a complete, stable fMRIPrep input scope for one exact BOLD run."""

    root = _canonical_path(bids_root, label="BIDS input root", directory=True)
    participant_value = str(participant).removeprefix("sub-")
    if re.fullmatch(r"[A-Za-z0-9]+", participant_value) is None:
        raise SourceAcquisitionError("BIDS input participant label is invalid")
    t1 = _canonical_path(Path(raw_t1), label="inventory raw T1w")
    bold = _canonical_path(Path(raw_bold), label="inventory raw BOLD")
    first_paths = _bids_input_paths(root, participant_value)
    files: list[dict[str, object]] = []
    artifacts: dict[str, Mapping[str, object]] = {}
    for path in first_paths:
        evidence = _snapshot_file(
            path,
            label=f"BIDS input {path.relative_to(root).as_posix()}",
            limit=MAX_SOURCE_BYTES,
            allow_empty=True,
        )
        relative = path.relative_to(root).as_posix()
        artifact = {
            "relative_path": relative,
            "sha256": evidence["sha256"],
            "size_bytes": evidence["size_bytes"],
        }
        files.append(artifact)
        artifacts[relative] = artifact
    second_paths = _bids_input_paths(root, participant_value)
    if [str(path) for path in second_paths] != [str(path) for path in first_paths]:
        raise SourceAcquisitionError(
            "BIDS input file set changed during inventory authentication"
        )

    t1_relative = t1.relative_to(root).as_posix()
    bold_relative = bold.relative_to(root).as_posix()
    if t1_relative not in artifacts or bold_relative not in artifacts:
        raise SourceAcquisitionError(
            "exact raw T1w/BOLD is absent from the BIDS input inventory"
        )
    participant_prefix = f"sub-{participant_value}/"
    bold_images = [
        relative
        for relative in artifacts
        if relative.startswith(participant_prefix)
        and _nifti_relative_matches(relative, "bold")
    ]
    t1_images = [
        relative
        for relative in artifacts
        if relative.startswith(participant_prefix)
        and _nifti_relative_matches(relative, "T1w")
    ]
    if bold_images != [bold_relative]:
        raise SourceAcquisitionError(
            "certified per-run BIDS scope must contain exactly the authenticated raw BOLD"
        )
    if t1_images != [t1_relative]:
        raise SourceAcquisitionError(
            "certified per-run BIDS scope must contain exactly the authenticated raw T1w"
        )
    if "dataset_description.json" not in artifacts:
        raise SourceAcquisitionError(
            "BIDS input inventory lacks dataset_description.json"
        )
    effective_metadata = _effective_bold_metadata(
        root, participant_value, bold, artifacts
    )
    unsigned: dict[str, Any] = {
        "schema": BIDS_INPUT_INVENTORY_SCHEMA,
        "scope": BIDS_INPUT_INVENTORY_SCOPE,
        "bids_root": str(root),
        "participant_label": participant_value,
        "raw_t1_relative_path": t1_relative,
        "raw_bold_relative_path": bold_relative,
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(int(value["size_bytes"]) for value in files),
        "effective_bold_metadata": effective_metadata,
    }
    return {**unsigned, "record_sha256": canonical_sha256(unsigned)}


def _read_signed_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    value, evidence = load_authenticated_json_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        label=label,
    )
    recorded = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    if not _valid_sha256(recorded) or canonical_sha256(unsigned) != recorded:
        raise SourceAcquisitionError(f"{label} signed record differs")
    return value, evidence


def load_authenticated_json_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Parse exactly the descriptor-held bytes authenticated by SHA and size."""
    if not _valid_sha256(expected_sha256):
        raise SourceAcquisitionError(f"{label} expected SHA-256 is invalid")
    if expected_size is not None and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or expected_size > MAX_JSON_BYTES
    ):
        raise SourceAcquisitionError(f"{label} expected size is invalid")
    value, evidence = snapshot_json_artifact(path, label=label)
    if evidence["sha256"] != expected_sha256:
        raise SourceAcquisitionError(f"{label} SHA-256 differs")
    if expected_size is not None and evidence["size_bytes"] != expected_size:
        raise SourceAcquisitionError(f"{label} size differs")
    return value, evidence


def snapshot_json_artifact(
    path: Path, *, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    """Read and parse one stable, descriptor-held, non-linked JSON snapshot."""
    evidence, payload = _snapshot_file_payload(
        path, label=label, limit=MAX_JSON_BYTES, capture_payload=True
    )
    assert payload is not None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAcquisitionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SourceAcquisitionError(f"{label} must be a JSON object")
    return value, evidence


def snapshot_binary_artifact(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[bytes, dict[str, object]]:
    """Capture exact descriptor-held source bytes under an expected SHA-256."""
    if not _valid_sha256(expected_sha256):
        raise SourceAcquisitionError(f"{label} expected SHA-256 is invalid")
    evidence, payload = _snapshot_file_payload(
        path, label=label, limit=MAX_SOURCE_BYTES, capture_payload=True
    )
    assert payload is not None
    if evidence["sha256"] != expected_sha256:
        raise SourceAcquisitionError(f"{label} SHA-256 differs")
    return payload, evidence


def _canonical_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SourceAcquisitionError(f"{label} relative path is missing")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SourceAcquisitionError(f"{label} relative path is not canonical")
    return relative


def _nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def bids_entities(path: Path, *, suffix: str) -> dict[str, str]:
    tokens = _nifti_stem(path).split("_")
    if not tokens or tokens[-1].lower() != suffix.lower():
        raise SourceAcquisitionError(f"BIDS source is not a {suffix} image: {path}")
    entities: dict[str, str] = {}
    for token in tokens[:-1]:
        if "-" not in token:
            raise SourceAcquisitionError(f"BIDS source has an invalid entity: {path}")
        key, value = token.split("-", 1)
        if not key or not value or key in entities:
            raise SourceAcquisitionError(f"BIDS source entities are ambiguous: {path}")
        entities[key] = value
    if "sub" not in entities:
        raise SourceAcquisitionError(f"BIDS source has no subject entity: {path}")
    return {"suffix": tokens[-1], **dict(sorted(entities.items()))}


def derivative_source_relative_paths(metadata: Mapping[str, Any]) -> set[str]:
    """Normalize exact BIDS ``Sources`` URIs without accepting path traversal."""
    values = metadata.get("Sources")
    if not isinstance(values, list) or not values:
        raise SourceAcquisitionError("derivative metadata has no BIDS Sources")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise SourceAcquisitionError("derivative BIDS Sources are malformed")
        if value.startswith("bids::"):
            candidate = value[len("bids::") :]
        elif value.startswith("bids:"):
            pieces = value.split(":", 2)
            if len(pieces) != 3 or not pieces[1]:
                raise SourceAcquisitionError("derivative BIDS URI is malformed")
            candidate = pieces[2]
        else:
            candidate = value
        relative = _canonical_relative(candidate, label="derivative BIDS source")
        result.add(relative.as_posix())
    return result


def _validate_artifact(
    value: object,
    *,
    label: str,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        raise SourceAcquisitionError(f"{label} artifact fields differ")
    if (
        not _valid_sha256(value.get("sha256"))
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
    ):
        raise SourceAcquisitionError(f"{label} artifact identity is malformed")
    observed = _snapshot_file(
        Path(str(value.get("path", ""))), label=label, limit=MAX_SOURCE_BYTES
    )
    if observed != value:
        raise SourceAcquisitionError(f"{label} bytes differ")
    if expected is not None and observed != dict(expected):
        raise SourceAcquisitionError(f"{label} is not the expected source artifact")
    return observed


def load_fmriprep_runtime_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_executable: Path,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Authenticate and rehash an externally pinned complete runtime inventory."""

    identity, evidence = _read_signed_json(
        Path(path),
        expected_sha256=expected_sha256,
        label="fMRIPrep runtime identity",
    )
    if set(identity) != {
        "schema",
        "purpose",
        "fmriprep_version",
        "runtime_kind",
        "complete_runtime_inventory",
        "executable",
        "runtime_artifacts",
        "record_sha256",
    }:
        raise SourceAcquisitionError("fMRIPrep runtime identity fields differ")
    version = identity.get("fmriprep_version")
    runtime_kind = identity.get("runtime_kind")
    if (
        identity.get("schema") != FMRIPREP_RUNTIME_IDENTITY_SCHEMA
        or identity.get("purpose") != FMRIPREP_RUNTIME_IDENTITY_PURPOSE
        or FMRIPREP_VERSION.fullmatch(str(version)) is None
        or not isinstance(runtime_kind, str)
        or not runtime_kind.strip()
        or identity.get("complete_runtime_inventory") is not True
    ):
        raise SourceAcquisitionError("fMRIPrep runtime identity contract differs")
    executable = _validate_artifact(
        identity.get("executable"), label="fMRIPrep executable"
    )
    expected = _canonical_path(
        Path(expected_executable), label="expected fMRIPrep executable"
    )
    if Path(str(executable["path"])) != expected:
        raise SourceAcquisitionError("fMRIPrep runtime executable path differs")
    artifacts = identity.get("runtime_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SourceAcquisitionError("fMRIPrep runtime artifact inventory is empty")
    observed_artifacts: list[dict[str, object]] = []
    for index, artifact in enumerate(artifacts):
        observed_artifacts.append(
            _validate_artifact(
                artifact,
                label=f"fMRIPrep runtime artifact {index}",
            )
        )
    paths = [str(value["path"]) for value in observed_artifacts]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SourceAcquisitionError(
            "fMRIPrep runtime artifacts must be uniquely path-sorted"
        )
    if executable not in observed_artifacts:
        raise SourceAcquisitionError(
            "fMRIPrep executable is absent from the complete runtime inventory"
        )
    if observed_artifacts != artifacts:
        raise SourceAcquisitionError("fMRIPrep runtime artifact bytes differ")
    return identity, {**evidence, "record_sha256": identity["record_sha256"]}


def _validate_synthseg_provenance(
    provenance: Mapping[str, Any],
    *,
    scan_id: str,
    raw_t1: Mapping[str, object],
    mask: Mapping[str, object],
) -> dict[str, Any]:
    if set(provenance) != {
        "schema",
        "scan_id",
        "producer",
        "input_t1",
        "source_t1_sha256",
        "output_mask",
        "record_sha256",
    }:
        raise SourceAcquisitionError("SynthSeg provenance fields differ")
    producer = provenance.get("producer")
    if (
        provenance.get("schema") != SYNTHSEG_PROVENANCE_SCHEMA
        or provenance.get("scan_id") != scan_id
        or provenance.get("input_t1") != dict(raw_t1)
        or provenance.get("source_t1_sha256") != raw_t1.get("sha256")
        or provenance.get("output_mask") != dict(mask)
        or not isinstance(producer, dict)
        or set(producer)
        != {
            "name",
            "version",
            "source_revision",
            "model_sha256",
            "container_image",
            "container_manifest_sha256",
        }
        or producer.get("name") != "SynthSeg"
        or not str(producer.get("version", "")).strip()
        or REVISION.fullmatch(str(producer.get("source_revision", ""))) is None
        or not _valid_sha256(producer.get("model_sha256"))
        or not str(producer.get("container_image", "")).strip()
        or not _valid_sha256(producer.get("container_manifest_sha256"))
    ):
        raise SourceAcquisitionError(
            "SynthSeg producer or source-T1/output-mask binding differs"
        )
    return dict(producer)


def load_source_acquisition_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_bids_root: Path | None = None,
    expected_participant: str | None = None,
    expected_scan_id: str | None = None,
    expected_raw_t1: Path | None = None,
    expected_raw_bold: Path | None = None,
    expected_synthseg_mask: Path | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Authenticate the signed identity and rehash all four source artifacts."""
    identity, identity_evidence = _read_signed_json(
        Path(path),
        expected_sha256=expected_sha256,
        label="source-acquisition identity",
    )
    if set(identity) != {
        "schema",
        "purpose",
        "scan_id",
        "participant_label",
        "bids_root",
        "bids_entities",
        "bids_input_inventory",
        "raw_t1",
        "raw_bold",
        "synthseg",
        "record_sha256",
    }:
        raise SourceAcquisitionError("source-acquisition identity fields differ")
    scan_id = str(identity.get("scan_id", ""))
    participant = str(identity.get("participant_label", "")).removeprefix("sub-")
    if (
        identity.get("schema") != SOURCE_ACQUISITION_SCHEMA
        or identity.get("purpose") != SOURCE_ACQUISITION_PURPOSE
        or SCAN_ID.fullmatch(scan_id) is None
        or not participant
        or re.fullmatch(r"[A-Za-z0-9]+", participant) is None
    ):
        raise SourceAcquisitionError("source-acquisition schema/identity differs")
    if expected_scan_id is not None and scan_id != str(expected_scan_id):
        raise SourceAcquisitionError("source-acquisition scan ID differs")
    bids_root = _canonical_path(
        Path(str(identity.get("bids_root", ""))),
        label="source-acquisition BIDS root",
        directory=True,
    )
    if expected_bids_root is not None and bids_root != _canonical_path(
        expected_bids_root, label="expected BIDS root", directory=True
    ):
        raise SourceAcquisitionError("source-acquisition BIDS root differs")
    if expected_participant is not None and participant != str(
        expected_participant
    ).removeprefix("sub-"):
        raise SourceAcquisitionError("source-acquisition participant differs")

    raw_t1 = _validate_artifact(identity.get("raw_t1"), label="raw BIDS T1w")
    raw_bold = _validate_artifact(identity.get("raw_bold"), label="raw BIDS BOLD")
    for label, artifact in (("raw BIDS T1w", raw_t1), ("raw BIDS BOLD", raw_bold)):
        source = Path(str(artifact["path"]))
        try:
            relative = source.relative_to(bids_root)
        except ValueError as exc:
            raise SourceAcquisitionError(f"{label} escapes the BIDS root") from exc
        if _canonical_relative(relative.as_posix(), label=label) != PurePosixPath(
            relative.as_posix()
        ):
            raise AssertionError("unreachable relative-path normalization mismatch")
    t1_entities = bids_entities(Path(str(raw_t1["path"])), suffix="T1w")
    bold_entities = bids_entities(Path(str(raw_bold["path"])), suffix="bold")
    for artifact, entities, datatype in (
        (raw_t1, t1_entities, "anat"),
        (raw_bold, bold_entities, "func"),
    ):
        relative = Path(str(artifact["path"])).relative_to(bids_root)
        expected_parent = [f"sub-{entities['sub']}"]
        if entities.get("ses") is not None:
            expected_parent.append(f"ses-{entities['ses']}")
        expected_parent.append(datatype)
        if relative.parent.parts != tuple(expected_parent):
            raise SourceAcquisitionError(
                f"raw {datatype} source is outside its exact BIDS entity directory"
            )
    expected_entities = {
        "raw_t1": t1_entities,
        "raw_bold": bold_entities,
    }
    if (
        t1_entities["sub"] != participant
        or bold_entities["sub"] != participant
        or not bold_entities.get("task")
        or (
            t1_entities.get("ses") is not None
            and bold_entities.get("ses") is not None
            and t1_entities["ses"] != bold_entities["ses"]
        )
        or identity.get("bids_entities") != expected_entities
    ):
        raise SourceAcquisitionError("raw T1w/BOLD BIDS entities differ")

    recorded_inventory = identity.get("bids_input_inventory")
    if not isinstance(recorded_inventory, dict):
        raise SourceAcquisitionError(
            "source-acquisition BIDS input inventory is missing"
        )
    observed_inventory = snapshot_bids_input_inventory(
        bids_root,
        participant=participant,
        raw_t1=Path(str(raw_t1["path"])),
        raw_bold=Path(str(raw_bold["path"])),
    )
    if recorded_inventory != observed_inventory:
        raise SourceAcquisitionError(
            "source-acquisition BIDS input inventory differs from live inputs"
        )

    synthseg = identity.get("synthseg")
    if not isinstance(synthseg, dict) or set(synthseg) != {
        "mask",
        "provenance",
        "producer",
    }:
        raise SourceAcquisitionError("SynthSeg acquisition fields differ")
    mask = _validate_artifact(synthseg.get("mask"), label="SynthSeg mask")
    provenance_artifact = synthseg.get("provenance")
    if not isinstance(provenance_artifact, dict) or set(provenance_artifact) != {
        "path",
        "sha256",
        "size_bytes",
        "record_sha256",
    }:
        raise SourceAcquisitionError("SynthSeg provenance artifact fields differ")
    provenance, observed_provenance = _read_signed_json(
        Path(str(provenance_artifact.get("path", ""))),
        expected_sha256=str(provenance_artifact.get("sha256", "")),
        label="SynthSeg provenance",
    )
    if observed_provenance != {
        key: provenance_artifact[key] for key in ("path", "sha256", "size_bytes")
    } or provenance.get("record_sha256") != provenance_artifact.get("record_sha256"):
        raise SourceAcquisitionError("SynthSeg provenance artifact differs")
    producer = _validate_synthseg_provenance(
        provenance,
        scan_id=scan_id,
        raw_t1=raw_t1,
        mask=mask,
    )
    if synthseg.get("producer") != producer:
        raise SourceAcquisitionError("SynthSeg producer identity differs")

    for expected_path, observed, label in (
        (expected_raw_t1, raw_t1, "raw T1w"),
        (expected_raw_bold, raw_bold, "raw BOLD"),
        (expected_synthseg_mask, mask, "SynthSeg mask"),
    ):
        if expected_path is not None and _canonical_path(
            expected_path, label=f"expected {label}"
        ) != Path(str(observed["path"])):
            raise SourceAcquisitionError(f"source-acquisition {label} path differs")
    return identity, {
        **identity_evidence,
        "record_sha256": identity["record_sha256"],
    }


def structural_source_projection(value: Mapping[str, Any]) -> dict[str, object]:
    """Project a structural or full target identity onto target-blind fields.

    The projection deliberately excludes raw BOLD and the digest/path of the
    full SourceAcquisition record.  It is therefore safe to compare against an
    allowlisted train/development fMRI identity without making a sealed
    structural cache depend on target bytes.
    """
    try:
        schema = value.get("schema")
        if schema == STRUCTURAL_SOURCE_BINDING_SCHEMA:
            if set(value) != {
                "schema",
                "scan_id",
                "structural_source_authority_path",
                "structural_source_authority_sha256",
                "structural_source_authority_size_bytes",
                "structural_source_authority_record_sha256",
                "raw_t1_sha256",
                "synthseg_mask_sha256",
                "synthseg_provenance_sha256",
                "synthseg_provenance_record_sha256",
            }:
                raise SourceAcquisitionError(
                    "structural source binding fields differ or include target data"
                )
            projection = {
                "schema": STRUCTURAL_SOURCE_PROJECTION_SCHEMA,
                "scan_id": value["scan_id"],
                "raw_t1_sha256": value["raw_t1_sha256"],
                "synthseg_mask_sha256": value["synthseg_mask_sha256"],
                "synthseg_provenance_sha256": value["synthseg_provenance_sha256"],
                "synthseg_provenance_record_sha256": value[
                    "synthseg_provenance_record_sha256"
                ],
            }
        elif schema == STRUCTURAL_SOURCE_PROJECTION_SCHEMA:
            if set(value) != {
                "schema",
                "scan_id",
                "raw_t1_sha256",
                "synthseg_mask_sha256",
                "synthseg_provenance_sha256",
                "synthseg_provenance_record_sha256",
            }:
                raise SourceAcquisitionError(
                    "structural source projection fields differ"
                )
            projection = dict(value)
        else:
            if schema not in {
                SOURCE_ACQUISITION_SCHEMA,
                STRUCTURAL_SOURCE_AUTHORITY_SCHEMA,
            }:
                raise SourceAcquisitionError(
                    "unsupported structural/full source identity schema"
                )
            synthseg = value["synthseg"]
            provenance = synthseg["provenance"]
            projection = {
                "schema": STRUCTURAL_SOURCE_PROJECTION_SCHEMA,
                "scan_id": value["scan_id"],
                "raw_t1_sha256": value["raw_t1"]["sha256"],
                "synthseg_mask_sha256": synthseg["mask"]["sha256"],
                "synthseg_provenance_sha256": provenance["sha256"],
                "synthseg_provenance_record_sha256": provenance["record_sha256"],
            }
    except (KeyError, TypeError) as exc:
        raise SourceAcquisitionError(
            "target-blind structural source fields are missing"
        ) from exc
    if (
        not isinstance(projection["scan_id"], str)
        or not projection["scan_id"]
        or any(
            not _valid_sha256(projection[key])
            for key in (
                "raw_t1_sha256",
                "synthseg_mask_sha256",
                "synthseg_provenance_sha256",
                "synthseg_provenance_record_sha256",
            )
        )
    ):
        raise SourceAcquisitionError("structural source projection is malformed")
    return projection


def load_structural_source_authority(
    path: Path,
    *,
    expected_sha256: str,
    expected_scan_id: str | None = None,
    expected_raw_t1: Path | None = None,
    expected_synthseg_mask: Path | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Authenticate a signed T1/SynthSeg-only authority without touching BOLD."""
    authority, authority_evidence = _read_signed_json(
        Path(path),
        expected_sha256=expected_sha256,
        label="target-blind structural source authority",
    )
    if set(authority) != {
        "schema",
        "purpose",
        "scan_id",
        "raw_t1",
        "synthseg",
        "record_sha256",
    }:
        raise SourceAcquisitionError("structural source authority fields differ")
    scan_id = authority.get("scan_id")
    if (
        authority.get("schema") != STRUCTURAL_SOURCE_AUTHORITY_SCHEMA
        or authority.get("purpose") != STRUCTURAL_SOURCE_AUTHORITY_PURPOSE
        or not isinstance(scan_id, str)
        or not scan_id
        or (expected_scan_id is not None and scan_id != str(expected_scan_id))
    ):
        raise SourceAcquisitionError("structural source authority identity differs")
    raw_t1 = _validate_artifact(
        authority.get("raw_t1"), label="structural-authority raw T1w"
    )
    synthseg = authority.get("synthseg")
    if not isinstance(synthseg, dict) or set(synthseg) != {
        "mask",
        "provenance",
        "producer",
    }:
        raise SourceAcquisitionError("structural-authority SynthSeg fields differ")
    mask = _validate_artifact(
        synthseg.get("mask"), label="structural-authority SynthSeg mask"
    )
    provenance_artifact = synthseg.get("provenance")
    if not isinstance(provenance_artifact, dict) or set(provenance_artifact) != {
        "path",
        "sha256",
        "size_bytes",
        "record_sha256",
    }:
        raise SourceAcquisitionError(
            "structural-authority SynthSeg provenance fields differ"
        )
    provenance, observed_provenance = _read_signed_json(
        Path(str(provenance_artifact.get("path", ""))),
        expected_sha256=str(provenance_artifact.get("sha256", "")),
        label="structural-authority SynthSeg provenance",
    )
    if observed_provenance != {
        key: provenance_artifact[key] for key in ("path", "sha256", "size_bytes")
    } or provenance.get("record_sha256") != provenance_artifact.get("record_sha256"):
        raise SourceAcquisitionError(
            "structural-authority SynthSeg provenance artifact differs"
        )
    producer = _validate_synthseg_provenance(
        provenance,
        scan_id=scan_id,
        raw_t1=raw_t1,
        mask=mask,
    )
    if synthseg.get("producer") != producer:
        raise SourceAcquisitionError(
            "structural-authority SynthSeg producer identity differs"
        )
    for expected_path, observed, label in (
        (expected_raw_t1, raw_t1, "raw T1w"),
        (expected_synthseg_mask, mask, "SynthSeg mask"),
    ):
        if expected_path is not None and _canonical_path(
            expected_path, label=f"expected structural {label}"
        ) != Path(str(observed["path"])):
            raise SourceAcquisitionError(f"structural-authority {label} path differs")
    return authority, {
        **authority_evidence,
        "record_sha256": authority["record_sha256"],
    }


def structural_source_binding(
    authority: Mapping[str, Any], evidence: Mapping[str, object]
) -> dict[str, object]:
    """Bind a T1/SynthSeg-only signed authority into structural derivatives."""
    if authority.get("schema") != STRUCTURAL_SOURCE_AUTHORITY_SCHEMA:
        raise SourceAcquisitionError(
            "legacy/full source-acquisition records cannot authorize structural "
            "preprocessing; use the target-blind v2 structural authority"
        )
    projection = structural_source_projection(authority)
    try:
        binding = {
            "schema": STRUCTURAL_SOURCE_BINDING_SCHEMA,
            "scan_id": projection["scan_id"],
            "structural_source_authority_path": evidence["path"],
            "structural_source_authority_sha256": evidence["sha256"],
            "structural_source_authority_size_bytes": evidence["size_bytes"],
            "structural_source_authority_record_sha256": evidence["record_sha256"],
            "raw_t1_sha256": projection["raw_t1_sha256"],
            "synthseg_mask_sha256": projection["synthseg_mask_sha256"],
            "synthseg_provenance_sha256": projection["synthseg_provenance_sha256"],
            "synthseg_provenance_record_sha256": projection[
                "synthseg_provenance_record_sha256"
            ],
        }
    except (KeyError, TypeError) as exc:
        raise SourceAcquisitionError(
            "structural authority evidence is missing"
        ) from exc
    if (
        not isinstance(binding["structural_source_authority_path"], str)
        or not binding["structural_source_authority_path"]
        or isinstance(binding["structural_source_authority_size_bytes"], bool)
        or not isinstance(binding["structural_source_authority_size_bytes"], int)
        or binding["structural_source_authority_size_bytes"] < 1
        or any(
            not _valid_sha256(binding[key])
            for key in (
                "structural_source_authority_sha256",
                "structural_source_authority_record_sha256",
                "raw_t1_sha256",
                "synthseg_mask_sha256",
                "synthseg_provenance_sha256",
                "synthseg_provenance_record_sha256",
            )
        )
    ):
        raise SourceAcquisitionError("structural source binding is malformed")
    return binding


def load_structural_source_binding(
    value: object, *, expected_scan_id: str | None = None
) -> tuple[dict[str, object], dict[str, Any]]:
    """Authenticate a v2 structural binding without resolving any BOLD path."""
    expected_fields = {
        "schema",
        "scan_id",
        "structural_source_authority_path",
        "structural_source_authority_sha256",
        "structural_source_authority_size_bytes",
        "structural_source_authority_record_sha256",
        "raw_t1_sha256",
        "synthseg_mask_sha256",
        "synthseg_provenance_sha256",
        "synthseg_provenance_record_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise SourceAcquisitionError("structural source-binding fields differ")
    if expected_scan_id is not None and value.get("scan_id") != expected_scan_id:
        raise SourceAcquisitionError(
            "structural source binding identifies another scan"
        )
    authority, evidence = load_structural_source_authority(
        Path(str(value.get("structural_source_authority_path", ""))),
        expected_sha256=str(value.get("structural_source_authority_sha256", "")),
        expected_scan_id=expected_scan_id,
    )
    observed = structural_source_binding(authority, evidence)
    if observed != value:
        raise SourceAcquisitionError(
            "structural source binding differs from the target-blind authority"
        )
    return observed, authority


__all__ = [
    "BIDS_INPUT_INVENTORY_SCHEMA",
    "BIDS_INPUT_INVENTORY_SCOPE",
    "FMRIPREP_RUNTIME_IDENTITY_PURPOSE",
    "FMRIPREP_RUNTIME_IDENTITY_SCHEMA",
    "SOURCE_ACQUISITION_PURPOSE",
    "SOURCE_ACQUISITION_SCHEMA",
    "STRUCTURAL_SOURCE_AUTHORITY_PURPOSE",
    "STRUCTURAL_SOURCE_AUTHORITY_SCHEMA",
    "STRUCTURAL_SOURCE_BINDING_SCHEMA",
    "STRUCTURAL_SOURCE_PROJECTION_SCHEMA",
    "SYNTHSEG_PROVENANCE_SCHEMA",
    "SourceAcquisitionError",
    "bids_entities",
    "canonical_sha256",
    "derivative_source_relative_paths",
    "load_authenticated_json_artifact",
    "load_fmriprep_runtime_identity",
    "load_source_acquisition_identity",
    "load_structural_source_authority",
    "load_structural_source_binding",
    "snapshot_json_artifact",
    "snapshot_binary_artifact",
    "snapshot_bids_input_inventory",
    "snapshot_file_artifact",
    "structural_source_binding",
    "structural_source_projection",
]
