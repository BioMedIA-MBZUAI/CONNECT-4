# ruff: noqa: F401, F821
"""Non-staged synthetic fixture evaluator for post-seal tests only.

This module deliberately owns executable Stage-B doubles and intermediate
fixture objects that are forbidden from the immutable production runtime.
It is excluded from ``scripts.stage_postseal_runtime.RUNTIME_FILES``.
"""

from __future__ import annotations

import types
import tempfile

from eval import postseal as _production

# Reuse the production evaluator’s inert validation and descriptor-safe
# publisher primitives without adding a callable fixture seam to that module.
globals().update(
    {
        name: value
        for name, value in vars(_production).items()
        if not name.startswith("__")
    }
)


@dataclass(frozen=True)
class _ExecutionBoundaryAuthority:
    evidence: Mapping[str, Any]
    stage_b_verifier_authority: _StageBVerifierAuthority
    capability: object


_EXECUTION_BOUNDARY_CAPABILITY = object()


def _acquire_execution_boundary() -> _ExecutionBoundaryAuthority:
    """Acquire, rather than accept, the target-boundary execution capability.

    The current source-only runtime v1 is revoked: it does not authenticate its
    site-packages, native libraries, or import machinery.  A fresh closed v2
    runtime must replace this hard stop; caller-provided booleans are never an
    alternative authority.
    """
    if not os.environ.get("SLURM_JOB_ID") or "login" in os.uname().nodename.lower():
        raise PostsealEvaluationError(
            "postseal target access requires internally acquired CIAI Slurm GPU "
            "authority; caller-supplied runtime evidence is forbidden"
        )
    raise PostsealEvaluationError(
        "postseal runtime v1 is revoked: a closed versioned dependency/native-library "
        "inventory with disabled .pth/import hooks and an internally pinned Stage-B "
        "authority is required before target access"
    )


@dataclass(frozen=True)
class AuthenticatedPrediction:
    scan_id: str
    record: dict[str, Any]
    record_snapshot: FileSnapshot
    prediction_snapshot: FileSnapshot


@dataclass(frozen=True)
class AuthenticatedPredictionSet:
    root: Path
    seal: dict[str, Any]
    seal_snapshot: FileSnapshot
    subjects: tuple[AuthenticatedPrediction, ...]

    @property
    def by_scan(self) -> dict[str, AuthenticatedPrediction]:
        return {item.scan_id: item for item in self.subjects}


def _validate_checkpoint_bindings(
    value: Mapping[str, Any], pins: _EvaluationPinSnapshot, *, label: str
) -> None:
    expected = pins.checkpoint_bindings()
    for name, digest in expected.items():
        observed = _require_sha256(value.get(name), label=f"{label} {name}")
        if observed != digest:
            raise PostsealEvaluationError(f"{label} {name} differs")


def _authenticate_prediction_set_impl(
    prediction_root: Path,
    *,
    pins: _EvaluationPinSnapshot,
) -> AuthenticatedPredictionSet:
    """Authenticate every sealed prediction byte before target authorization."""
    root = _canonical_path(prediction_root, label="prediction root", directory=True)
    root_children = {path.name: path for path in root.iterdir()}
    if set(root_children) != {"subjects", "prediction_set_seal.json"}:
        raise PostsealEvaluationError(
            "prediction root is partial or contains unsealed extra entries"
        )
    for name, path in root_children.items():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PostsealEvaluationError(
                f"prediction root entry {name!r} is a symlink"
            )
    seal_path = root / "prediction_set_seal.json"
    seal, seal_snapshot = _signed_json_snapshot(
        seal_path,
        label="prediction-set seal",
        expected_sha256=pins.prediction_set_seal_sha256,
        expected_schema_key="format",
        expected_schema=PREDICTION_SET_FORMAT,
    )
    if set(seal) != _ROOT_SEAL_FIELDS:
        raise PostsealEvaluationError("prediction-set seal fields differ")
    subject_bindings = seal.get("subjects")
    if not isinstance(subject_bindings, list):
        raise PostsealEvaluationError("prediction-set subjects are not a list")
    scan_ids = [
        item.get("scan_id") if isinstance(item, dict) else None
        for item in subject_bindings
    ]
    if (
        seal.get("status") != "SEALED_TARGET_BLIND_PREDICTION_SET"
        or seal.get("split") != "test"
        or seal.get("num_subjects") != len(EXPECTED_SCAN_IDS)
        or tuple(scan_ids) != EXPECTED_SCAN_IDS
        or seal.get("subjects_sha256") != canonical_sha256(subject_bindings)
        or canonical_sha256(scan_ids) != EXPECTED_SCAN_IDS_SHA256
        or seal.get("sealed_targets_opened") is not False
        or seal.get("paired_quality_gate_run") is not False
    ):
        raise PostsealEvaluationError(
            "prediction-set seal is partial, reordered, replaced, or not target blind"
        )
    _validate_checkpoint_bindings(seal, pins, label="prediction-set seal")
    subjects: list[AuthenticatedPrediction] = []
    for expected_scan_id, binding in zip(EXPECTED_SCAN_IDS, subject_bindings):
        if not isinstance(binding, dict) or set(binding) != {
            "scan_id",
            "prediction_sha256",
            "subject_record_sha256",
        }:
            raise PostsealEvaluationError("prediction subject binding fields differ")
        scan_id = _safe_scan_id(binding.get("scan_id"))
        if scan_id != expected_scan_id:
            raise PostsealEvaluationError("prediction subject order differs")
        prediction_sha = _require_sha256(
            binding.get("prediction_sha256"),
            label=f"{scan_id} bound prediction SHA-256",
        )
        record_sha = _require_sha256(
            binding.get("subject_record_sha256"),
            label=f"{scan_id} bound subject-record SHA-256",
        )
        subject_dir = _canonical_path(
            root / "subjects" / scan_id,
            label=f"{scan_id} prediction directory",
            directory=True,
        )
        record, record_snapshot = _signed_json_snapshot(
            subject_dir / "prediction.json",
            label=f"{scan_id} prediction record",
            expected_schema_key="format",
            expected_schema=PREDICTION_FORMAT,
        )
        if set(record) != _SUBJECT_RECORD_FIELDS:
            raise PostsealEvaluationError(f"{scan_id} prediction-record fields differ")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict) or set(prediction) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "shape",
            "repetition_time_seconds",
        }:
            raise PostsealEvaluationError(f"{scan_id} prediction descriptor differs")
        expected_relative = f"subjects/{scan_id}/prediction.nii.gz"
        if (
            record.get("status") != "SEALED_TARGET_BLIND_PREDICTION"
            or record.get("scan_id") != scan_id
            or not isinstance(record.get("protocol_profile"), str)
            or not record.get("protocol_profile")
            or record.get("target_opened_before_prediction") is not False
            or record.get("paired_quality_gate_run") is not False
            or record.get("record_sha256") != record_sha
            or prediction.get("relative_path") != expected_relative
            or prediction.get("sha256") != prediction_sha
            or prediction.get("shape") != list(EXPECTED_SHAPE)
            or prediction.get("repetition_time_seconds") != EXPECTED_TR_SECONDS
            or isinstance(prediction.get("size_bytes"), bool)
            or not isinstance(prediction.get("size_bytes"), int)
            or prediction.get("size_bytes") < 1
        ):
            raise PostsealEvaluationError(
                f"{scan_id} prediction record is inconsistent with the root seal"
            )
        _validate_checkpoint_bindings(record, pins, label=f"{scan_id} prediction")
        prediction_path = root / expected_relative
        prediction_snapshot, _ = _snapshot_file(
            prediction_path,
            label=f"{scan_id} prediction NIfTI",
            expected_sha256=prediction_sha,
            expected_size=prediction["size_bytes"],
            capture=False,
        )
        subjects.append(
            AuthenticatedPrediction(
                scan_id=scan_id,
                record=record,
                record_snapshot=record_snapshot,
                prediction_snapshot=prediction_snapshot,
            )
        )
    # Reject extra entries anywhere under subjects, including alternate records
    # or replacement predictions that are not committed by the root seal.
    expected_relative_files = {
        f"{scan_id}/prediction.json" for scan_id in EXPECTED_SCAN_IDS
    } | {f"{scan_id}/prediction.nii.gz" for scan_id in EXPECTED_SCAN_IDS}
    observed_relative_files: set[str] = set()
    subjects_root = _canonical_path(
        root / "subjects", label="prediction subjects root", directory=True
    )
    observed_subject_directories = {
        path.name
        for path in subjects_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if observed_subject_directories != set(EXPECTED_SCAN_IDS):
        raise PostsealEvaluationError(
            "prediction subject directories are partial, extra, or replaced"
        )
    for path in subjects_root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PostsealEvaluationError("prediction tree contains a symbolic link")
        if path.is_file():
            observed_relative_files.add(path.relative_to(subjects_root).as_posix())
        elif not path.is_dir():
            raise PostsealEvaluationError("prediction tree contains a special file")
    if observed_relative_files != expected_relative_files:
        raise PostsealEvaluationError(
            "prediction tree is partial or contains unsealed extra/replacement files"
        )
    return AuthenticatedPredictionSet(
        root=root,
        seal=seal,
        seal_snapshot=seal_snapshot,
        subjects=tuple(subjects),
    )


def authenticate_prediction_set(
    prediction_root: str | Path,
    *,
    pins: EvaluationPins,
) -> AuthenticatedPredictionSet:
    """Authenticate the fixed complete prediction set without caller callbacks."""
    root = _canonical_path(prediction_root, label="prediction root", directory=True)
    frozen_pins = _snapshot_evaluation_pins(pins)
    return _authenticate_prediction_set_impl(
        root,
        pins=frozen_pins,
    )


@dataclass(frozen=True)
class _StageBVerifierAuthority:
    source_path: Path
    source_sha256: str
    dependency_sources: tuple[tuple[Path, str], ...]
    function_name: str = "verify_postseal_target_completed_set"

    def __post_init__(self) -> None:
        source_path = object.__getattribute__(self, "source_path")
        source_sha256 = object.__getattribute__(self, "source_sha256")
        dependencies = object.__getattribute__(self, "dependency_sources")
        function_name = object.__getattribute__(self, "function_name")
        if type(source_path) is not _NATIVE_PATH_TYPE:
            raise PostsealEvaluationError("Stage-B source path type differs")
        _require_sha256(source_sha256, label="Stage-B verifier source SHA-256")
        if type(function_name) is not str or function_name != (
            "verify_postseal_target_completed_set"
        ):
            raise PostsealEvaluationError("Stage-B verifier function name differs")
        if type(dependencies) is not tuple or len(dependencies) != 2:
            raise PostsealEvaluationError(
                "Stage-B verifier must pin exactly its package initializer and controller"
            )
        if any(type(item) is not tuple or len(item) != 2 for item in dependencies):
            raise PostsealEvaluationError("Stage-B dependency declaration differs")
        paths = [item[0] for item in dependencies]
        if any(type(path) is not _NATIVE_PATH_TYPE for path in paths):
            raise PostsealEvaluationError("Stage-B dependency path type differs")
        if len(set(paths)) != len(paths) or source_path in paths:
            raise PostsealEvaluationError("Stage-B verifier dependency paths differ")
        for path, digest in dependencies:
            if not path.is_absolute():
                raise PostsealEvaluationError(
                    "Stage-B verifier dependency path is not absolute"
                )
            _require_sha256(digest, label="Stage-B verifier dependency SHA-256")


@dataclass(frozen=True)
class TargetSubjectEvidence:
    scan_id: str
    native_bold: FileSnapshot
    native_structural_brain_mask: FileSnapshot
    native_structural_labels: FileSnapshot
    publication: FileSnapshot
    success_receipt: FileSnapshot
    native_preprocessing_receipt: FileSnapshot
    raw: dict[str, Any]


@dataclass(frozen=True)
class AuthenticatedTargetSet:
    completed_set: FileSnapshot
    completed_set_record_sha256: str
    verification: dict[str, Any]
    verifier_source: FileSnapshot
    verifier_dependencies: tuple[FileSnapshot, ...]
    subjects: tuple[TargetSubjectEvidence, ...]

    @property
    def by_scan(self) -> dict[str, TargetSubjectEvidence]:
        return {item.scan_id: item for item in self.subjects}


def _artifact_from_verifier(
    value: object,
    *,
    label: str,
    json_record: bool = False,
) -> tuple[FileSnapshot, dict[str, Any] | None]:
    expected_fields = {"path", "sha256", "size_bytes"}
    if json_record:
        expected_fields.add("record_sha256")
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise PostsealEvaluationError(f"{label} descriptor fields differ")
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise PostsealEvaluationError(f"{label} size differs")
    if json_record:
        record, snapshot = _signed_json_snapshot(
            Path(str(value.get("path", ""))),
            label=label,
            expected_sha256=digest,
            expected_size=size,
        )
        if record.get("record_sha256") != value.get("record_sha256"):
            raise PostsealEvaluationError(f"{label} record SHA-256 differs")
        return snapshot, record
    snapshot, _ = _snapshot_file(
        Path(str(value.get("path", ""))),
        label=label,
        expected_sha256=digest,
        expected_size=size,
    )
    return snapshot, None


def _authenticate_stage_b_result(
    completed_set_path: Path,
    *,
    completed_set_sha256: str,
    predictions: AuthenticatedPredictionSet,
    pins: _EvaluationPinSnapshot,
    raw_result: object,
    verifier_source: FileSnapshot,
    verifier_dependencies: tuple[FileSnapshot, ...],
) -> AuthenticatedTargetSet:
    """Authenticate the inert result returned by the pinned Stage-B function."""
    completed_digest = _require_sha256(
        completed_set_sha256, label="Stage-B completed-set SHA-256"
    )
    if type(raw_result) is not dict:
        raise PostsealEvaluationError("Stage-B verifier returned no exact dictionary")
    result = raw_result
    required_fields = {
        "schema",
        "completed_set",
        "prediction_set_seal_sha256",
        "prediction_set_seal_record_sha256",
        "checkpoint_bindings",
        "ordered_scan_ids",
        "ordered_scan_ids_sha256",
        "subjects",
        "complete",
        "no_replacement",
        "no_refill",
        "evaluation_only",
        "authorizes_model_training",
        "authorizes_checkpoint_selection",
        "authorizes_model_or_candidate_selection",
        "authorizes_prediction_emission",
        "authorizes_additional_inference",
        "record_sha256",
    }
    if set(result) != required_fields:
        raise PostsealEvaluationError("Stage-B verifier result fields differ")
    unsigned_result = dict(result)
    result_record_sha = unsigned_result.pop("record_sha256", None)
    bindings = result.get("checkpoint_bindings")
    subject_values = result.get("subjects")
    if (
        result.get("schema") != STAGE_B_VERIFICATION_SCHEMA
        or result_record_sha != canonical_sha256(unsigned_result)
        or result.get("prediction_set_seal_sha256") != predictions.seal_snapshot.sha256
        or result.get("prediction_set_seal_record_sha256")
        != predictions.seal.get("record_sha256")
        or bindings != pins.checkpoint_bindings()
        or result.get("ordered_scan_ids") != list(EXPECTED_SCAN_IDS)
        or result.get("ordered_scan_ids_sha256") != EXPECTED_SCAN_IDS_SHA256
        or not isinstance(subject_values, list)
        or len(subject_values) != len(EXPECTED_SCAN_IDS)
        or result.get("complete") is not True
        or result.get("no_replacement") is not True
        or result.get("no_refill") is not True
        or result.get("evaluation_only") is not True
        or any(
            result.get(name) is not False
            for name in (
                "authorizes_model_training",
                "authorizes_checkpoint_selection",
                "authorizes_model_or_candidate_selection",
                "authorizes_prediction_emission",
                "authorizes_additional_inference",
            )
        )
    ):
        raise PostsealEvaluationError("Stage-B verification contract differs")
    completed_descriptor = result.get("completed_set")
    completed_snapshot, completed_record = _artifact_from_verifier(
        completed_descriptor,
        label="Stage-B postseal completed set",
        json_record=True,
    )
    if (
        completed_snapshot.path
        != _canonical_path(completed_set_path, label="Stage-B completed-set path")
        or completed_snapshot.sha256 != completed_digest
        or completed_record is None
    ):
        raise PostsealEvaluationError("Stage-B completed-set binding differs")
    authenticated_subjects: list[TargetSubjectEvidence] = []
    seen_artifact_paths: dict[str, set[Path]] = {
        "native_bold": set(),
        "native_structural_brain_mask": set(),
        "native_structural_labels": set(),
        "publication": set(),
        "success_receipt": set(),
        "native_preprocessing_receipt": set(),
    }
    for expected_scan_id, value in zip(EXPECTED_SCAN_IDS, subject_values):
        expected_subject_fields = {
            "scan_id",
            "native_bold",
            "native_structural_brain_mask",
            "native_structural_labels",
            "publication",
            "success_receipt",
            "native_preprocessing_receipt",
        }
        if not isinstance(value, dict) or set(value) != expected_subject_fields:
            raise PostsealEvaluationError("Stage-B subject evidence fields differ")
        scan_id = _safe_scan_id(value.get("scan_id"), label="Stage-B scan ID")
        if scan_id != expected_scan_id:
            raise PostsealEvaluationError("Stage-B subject order differs")
        publication, _ = _artifact_from_verifier(
            value.get("publication"),
            label=f"{scan_id} Stage-B publication",
            json_record=True,
        )
        success_receipt, _ = _artifact_from_verifier(
            value.get("success_receipt"),
            label=f"{scan_id} Stage-B success receipt",
            json_record=True,
        )
        native_receipt, _ = _artifact_from_verifier(
            value.get("native_preprocessing_receipt"),
            label=f"{scan_id} native preprocessing receipt",
            json_record=True,
        )
        native_bold, _ = _artifact_from_verifier(
            value.get("native_bold"), label=f"{scan_id} native BOLD target"
        )
        structural_brain_mask, _ = _artifact_from_verifier(
            value.get("native_structural_brain_mask"),
            label=f"{scan_id} native structural brain mask",
        )
        structural_labels, _ = _artifact_from_verifier(
            value.get("native_structural_labels"),
            label=f"{scan_id} native structural labels",
        )
        if (
            structural_brain_mask.path == structural_labels.path
            or structural_brain_mask.sha256 == structural_labels.sha256
        ):
            raise PostsealEvaluationError(
                f"{scan_id} brain support is not independent of ROI labels"
            )
        subject_artifacts = {
            "native_bold": native_bold,
            "native_structural_brain_mask": structural_brain_mask,
            "native_structural_labels": structural_labels,
            "publication": publication,
            "success_receipt": success_receipt,
            "native_preprocessing_receipt": native_receipt,
        }
        for role, artifact in subject_artifacts.items():
            if artifact.path in seen_artifact_paths[role]:
                raise PostsealEvaluationError(
                    f"Stage-B {role} artifact is reused across sealed scans"
                )
            seen_artifact_paths[role].add(artifact.path)
        authenticated_subjects.append(
            TargetSubjectEvidence(
                scan_id=scan_id,
                native_bold=native_bold,
                native_structural_brain_mask=structural_brain_mask,
                native_structural_labels=structural_labels,
                publication=publication,
                success_receipt=success_receipt,
                native_preprocessing_receipt=native_receipt,
                raw=dict(value),
            )
        )
    return AuthenticatedTargetSet(
        completed_set=completed_snapshot,
        completed_set_record_sha256=completed_record["record_sha256"],
        verification=result,
        verifier_source=verifier_source,
        verifier_dependencies=verifier_dependencies,
        subjects=tuple(authenticated_subjects),
    )


_STAGE_B_PACKAGE_NAMES = frozenset(
    {"__init__.py", "stage_b_controller.py", "sealed_target_stage_b.py"}
)


@dataclass(frozen=True)
class _CapturedStageBSource:
    snapshot: FileSnapshot
    payload: bytes
    mode: int
    mtime_ns: int
    ctime_ns: int


def _assert_stage_b_package_binding(
    package: Path,
    descriptor: int,
    identity: tuple[int, int],
) -> None:
    try:
        opened = os.fstat(descriptor)
        current = package.lstat()
    except OSError as exc:
        raise PostsealEvaluationError("Stage-B package root is unavailable") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (opened.st_dev, opened.st_ino) != identity
        or (current.st_dev, current.st_ino) != identity
    ):
        raise PostsealEvaluationError("Stage-B package root was replaced")


def _capture_stage_b_source(
    package: Path,
    package_descriptor: int,
    name: str,
    expected_sha256: str,
) -> _CapturedStageBSource:
    if name not in _STAGE_B_PACKAGE_NAMES:
        raise PostsealEvaluationError("Stage-B source name is outside the closed set")
    expected = _require_sha256(expected_sha256, label=f"Stage-B {name} SHA-256")
    try:
        initial = os.stat(name, dir_fd=package_descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=package_descriptor,
        )
    except OSError as exc:
        raise PostsealEvaluationError(
            f"Stage-B source {name} cannot be opened"
        ) from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or stat.S_ISLNK(initial.st_mode)
            or initial.st_nlink != 1
            or stat.S_IMODE(initial.st_mode) & 0o022
            or initial.st_size < 1
            or initial.st_size > 16 * 1024 * 1024
            or (before.st_dev, before.st_ino, before.st_size)
            != (initial.st_dev, initial.st_ino, initial.st_size)
        ):
            raise PostsealEvaluationError(
                f"Stage-B source {name} type/link/mode/size differs"
            )
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            total += len(block)
            if total > 16 * 1024 * 1024:
                raise PostsealEvaluationError(f"Stage-B source {name} is too large")
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=package_descriptor, follow_symlinks=False)
    finally:
        os.close(descriptor)
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if total != before.st_size or any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(current, field)
        for field in stable
    ):
        raise PostsealEvaluationError(f"Stage-B source {name} changed while captured")
    observed = digest.hexdigest()
    if observed != expected:
        raise PostsealEvaluationError(f"Stage-B source {name} SHA-256 differs")
    return _CapturedStageBSource(
        snapshot=FileSnapshot(
            path=package / name,
            sha256=observed,
            size_bytes=total,
            device=int(before.st_dev),
            inode=int(before.st_ino),
        ),
        payload=b"".join(chunks),
        mode=stat.S_IMODE(before.st_mode),
        mtime_ns=int(before.st_mtime_ns),
        ctime_ns=int(before.st_ctime_ns),
    )


def _reauthenticate_stage_b_source(
    package_descriptor: int,
    expected: _CapturedStageBSource,
) -> None:
    name = expected.snapshot.path.name
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=package_descriptor,
        )
    except OSError as exc:
        raise PostsealEvaluationError(f"Stage-B source {name} was replaced") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected.mode
        ):
            raise PostsealEvaluationError(f"Stage-B source {name} mode/link differs")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=package_descriptor, follow_symlinks=False)
    finally:
        os.close(descriptor)
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(
            getattr(before, field) != getattr(after, field)
            or getattr(before, field) != getattr(current, field)
            for field in stable
        )
        or (before.st_dev, before.st_ino)
        != (expected.snapshot.device, expected.snapshot.inode)
        or total != expected.snapshot.size_bytes
        or digest.hexdigest() != expected.snapshot.sha256
        or before.st_mtime_ns != expected.mtime_ns
        or before.st_ctime_ns != expected.ctime_ns
    ):
        raise PostsealEvaluationError(f"Stage-B source {name} identity/bytes changed")


@dataclass(frozen=True)
class _LoadedStageBVerifier:
    function: types.FunctionType
    function_code: types.CodeType
    function_globals: dict[str, Any]
    source: FileSnapshot
    dependencies: tuple[FileSnapshot, ...]
    package: Path
    package_descriptor: int
    package_identity: tuple[int, int]
    captured: tuple[_CapturedStageBSource, ...]
    module_names: tuple[str, ...]

    def reauthenticate(self) -> None:
        if (
            self.function.__code__ is not self.function_code
            or self.function.__globals__ is not self.function_globals
            or self.function.__defaults__ is not None
            or self.function.__kwdefaults__ is not None
            or self.function.__closure__ is not None
        ):
            raise PostsealEvaluationError(
                "Stage-B verifier function object was replaced or mutated"
            )
        _assert_stage_b_package_binding(
            self.package, self.package_descriptor, self.package_identity
        )
        if set(os.listdir(self.package_descriptor)) != _STAGE_B_PACKAGE_NAMES:
            raise PostsealEvaluationError(
                "Stage-B package inventory changed after capture"
            )
        for expected in self.captured:
            _reauthenticate_stage_b_source(self.package_descriptor, expected)
        _assert_stage_b_package_binding(
            self.package, self.package_descriptor, self.package_identity
        )

    def close(self) -> None:
        if self.package_descriptor < 0:
            return
        for name in self.module_names:
            sys.modules.pop(name, None)
        os.close(self.package_descriptor)
        object.__setattr__(self, "package_descriptor", -1)

    def __enter__(self) -> "_LoadedStageBVerifier":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _load_stage_b_verifier(
    authority: _StageBVerifierAuthority,
) -> _LoadedStageBVerifier:
    """Execute only descriptor-captured bytes from a closed Stage-B package."""
    source = _canonical_path(authority.source_path, label="Stage-B verifier source")
    package = _canonical_path(
        source.parent, label="Stage-B verifier package", directory=True
    )
    declared_by_name = {source.name: (source, authority.source_sha256)}
    for path, digest in authority.dependency_sources:
        canonical = _canonical_path(path, label="Stage-B verifier dependency")
        if canonical.name in declared_by_name:
            raise PostsealEvaluationError("Stage-B verifier source declaration repeats")
        declared_by_name[canonical.name] = (canonical, digest)
    if set(declared_by_name) != _STAGE_B_PACKAGE_NAMES or any(
        path.parent != package for path, _ in declared_by_name.values()
    ):
        raise PostsealEvaluationError(
            "Stage-B package is not a closed exact source inventory"
        )
    package_descriptor = os.open(
        package,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(package_descriptor)
    package_identity = (int(opened.st_dev), int(opened.st_ino))
    module_names: tuple[str, ...] = ()
    try:
        _assert_stage_b_package_binding(package, package_descriptor, package_identity)
        if set(os.listdir(package_descriptor)) != _STAGE_B_PACKAGE_NAMES:
            raise PostsealEvaluationError(
                "Stage-B package is not a closed exact source inventory"
            )
        captured_by_name = {
            name: _capture_stage_b_source(package, package_descriptor, name, digest)
            for name, (_path, digest) in declared_by_name.items()
        }
        captured = tuple(
            captured_by_name[name] for name in sorted(_STAGE_B_PACKAGE_NAMES)
        )
        _assert_stage_b_package_binding(package, package_descriptor, package_identity)
        if set(os.listdir(package_descriptor)) != _STAGE_B_PACKAGE_NAMES:
            raise PostsealEvaluationError(
                "Stage-B package inventory changed before execution"
            )
        for expected in captured:
            _reauthenticate_stage_b_source(package_descriptor, expected)
        _assert_stage_b_package_binding(package, package_descriptor, package_identity)
        source_binding = captured_by_name["sealed_target_stage_b.py"]
        dependency_bindings = tuple(
            captured_by_name[name].snapshot
            for name in ("__init__.py", "stage_b_controller.py")
        )
        module_prefix = (
            f"_connect4_closed_stage_b_{authority.source_sha256[:16]}_"
            f"{package_identity[1]:x}_{id(captured_by_name):x}"
        )
        controller_name = f"{module_prefix}.stage_b_controller"
        verifier_name = f"{module_prefix}.sealed_target_stage_b"
        module_names = (verifier_name, controller_name, module_prefix)
        if any(name in sys.modules for name in module_names):
            raise PostsealEvaluationError("Stage-B private module identity collides")
        package_module = types.ModuleType(module_prefix)
        package_module.__file__ = str(package / "__init__.py")
        package_module.__package__ = module_prefix
        package_module.__path__ = []
        controller_module = types.ModuleType(controller_name)
        controller_module.__file__ = str(package / "stage_b_controller.py")
        controller_module.__package__ = module_prefix
        verifier_module = types.ModuleType(verifier_name)
        verifier_module.__file__ = str(source)
        verifier_module.__package__ = module_prefix
        sys.modules[module_prefix] = package_module
        sys.modules[controller_name] = controller_module
        sys.modules[verifier_name] = verifier_module
        exec(
            compile(
                captured_by_name["__init__.py"].payload,
                str(package / "__init__.py"),
                "exec",
            ),
            package_module.__dict__,
        )
        exec(
            compile(
                captured_by_name["stage_b_controller.py"].payload,
                str(package / "stage_b_controller.py"),
                "exec",
            ),
            controller_module.__dict__,
        )
        setattr(package_module, "stage_b_controller", controller_module)
        exec(
            compile(source_binding.payload, str(source), "exec"),
            verifier_module.__dict__,
        )
        verifier = verifier_module.__dict__.get(authority.function_name)
        if (
            not isinstance(verifier, types.FunctionType)
            or verifier.__globals__ is not verifier_module.__dict__
        ):
            raise PostsealEvaluationError(
                "Stage-B captured module did not define the exact verifier function"
            )
        for name in module_names:
            sys.modules.pop(name, None)
        loaded = _LoadedStageBVerifier(
            function=verifier,
            function_code=verifier.__code__,
            function_globals=verifier.__globals__,
            source=source_binding.snapshot,
            dependencies=dependency_bindings,
            package=package,
            package_descriptor=package_descriptor,
            package_identity=package_identity,
            captured=captured,
            module_names=module_names,
        )
        loaded.reauthenticate()
        return loaded
    except Exception:
        for name in module_names:
            sys.modules.pop(name, None)
        os.close(package_descriptor)
        raise


@dataclass(frozen=True)
class _TargetBoundaryGrant:
    predictions: AuthenticatedPredictionSet
    pins: _EvaluationPinSnapshot
    stage_b_verifier_authority: _StageBVerifierAuthority
    runtime_evidence: Mapping[str, Any]
    capability: object


def _build_target_boundary_authorizers():
    grant_nonce = object()

    def mint(
        predictions: AuthenticatedPredictionSet,
        pins: _EvaluationPinSnapshot,
        execution_authority: _ExecutionBoundaryAuthority,
    ) -> _TargetBoundaryGrant:
        if (
            type(predictions) is not AuthenticatedPredictionSet
            or type(pins) is not _EvaluationPinSnapshot
            or type(execution_authority) is not _ExecutionBoundaryAuthority
            or execution_authority.capability is not _EXECUTION_BOUNDARY_CAPABILITY
            or type(execution_authority.stage_b_verifier_authority)
            is not _StageBVerifierAuthority
            or type(execution_authority.evidence) is not dict
        ):
            raise PostsealEvaluationError("execution/target-boundary authority differs")
        return _TargetBoundaryGrant(
            predictions=predictions,
            pins=pins,
            stage_b_verifier_authority=execution_authority.stage_b_verifier_authority,
            runtime_evidence=dict(execution_authority.evidence),
            capability=grant_nonce,
        )

    def require(grant: object) -> _TargetBoundaryGrant:
        if (
            type(grant) is not _TargetBoundaryGrant
            or object.__getattribute__(grant, "capability") is not grant_nonce
            or type(object.__getattribute__(grant, "predictions"))
            is not AuthenticatedPredictionSet
            or type(object.__getattribute__(grant, "pins"))
            is not _EvaluationPinSnapshot
            or type(object.__getattribute__(grant, "stage_b_verifier_authority"))
            is not _StageBVerifierAuthority
        ):
            raise PostsealEvaluationError("target-boundary grant differs")
        return grant

    return mint, require


_mint_target_boundary_grant, _require_target_boundary_grant = (
    _build_target_boundary_authorizers()
)
del _build_target_boundary_authorizers


def _open_authenticated_stage_b_targets(
    grant: _TargetBoundaryGrant,
    completed_set_path: Path,
    completed_set_sha256: str,
) -> AuthenticatedTargetSet:
    """Open targets only through a freshly minted, internally pinned grant."""
    grant = _require_target_boundary_grant(grant)
    completed_digest = _require_sha256(
        completed_set_sha256, label="Stage-B completed-set SHA-256"
    )
    authority = grant.stage_b_verifier_authority
    with _load_stage_b_verifier(authority) as loaded:
        loaded.reauthenticate()
        try:
            raw_result = loaded.function(
                completed_set_path,
                completed_digest,
                grant.predictions.seal_snapshot.sha256,
                grant.pins.checkpoint_bindings(),
            )
        except Exception as exc:
            raise PostsealEvaluationError(
                "pinned Stage-B target verification failed"
            ) from exc
        finally:
            loaded.reauthenticate()
        return _authenticate_stage_b_result(
            completed_set_path,
            completed_set_sha256=completed_digest,
            predictions=grant.predictions,
            pins=grant.pins,
            raw_result=raw_result,
            verifier_source=loaded.source,
            verifier_dependencies=loaded.dependencies,
        )


def _materialize_postseal_target_validity_mask(
    real_path: Path,
    structural_brain_mask_path: Path,
    destination: Path,
) -> FileSnapshot:
    """Materialize exact observed BOLD support after target-set authentication."""
    if destination.exists():
        raise FileExistsError(destination)
    real_image, real, _ = _strict_nifti_values(
        real_path,
        label="postseal target for validity derivation",
        expected_shape=EXPECTED_SHAPE,
    )
    mask_image, structural, _ = _strict_nifti_values(
        structural_brain_mask_path,
        label="postseal structural support for validity derivation",
        expected_shape=EXPECTED_SPATIAL_SHAPE,
        is_binary_mask=True,
    )
    if not np.array_equal(real_image.affine, mask_image.affine):
        raise PostsealEvaluationError(
            "target-validity structural affine differs from held-out target"
        )
    validity = _derive_exact_target_validity_mask(real, structural)
    header = mask_image.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(
        validity.astype(np.uint8, copy=False),
        np.asarray(real_image.affine, dtype=np.float64),
        header=header,
    )
    nib.save(image, str(destination))
    snapshot, _ = _snapshot_file(
        destination,
        label="derived postseal target-validity mask",
    )
    return snapshot


def _run_postseal_evaluation_impl(
    *,
    prediction_root: Path,
    pins: _EvaluationPinSnapshot,
    stage_b_completed_set: Path,
    stage_b_completed_set_sha256: str,
    output_root: Path,
    fps: int = 8,
    slimbrain_authority: Path | None = None,
    slimbrain_authority_sha256: str | None = None,
    slimbrain_dependency_authority_sha256: str | None = None,
    slimbrain_python_executable_sha256: str | None = None,
    slimbrain_device: str | None = None,
) -> dict[str, Any]:
    """Run after public inputs have been reduced to inert exact snapshots."""
    output = _canonical_path(output_root, label="evaluation output", must_exist=False)
    predictions = _authenticate_prediction_set_impl(prediction_root, pins=pins)
    execution_authority = _acquire_execution_boundary()
    target_grant = _mint_target_boundary_grant(predictions, pins, execution_authority)
    # The completed-set pathname is first resolved only after the full prediction
    # set and internally acquired execution/Stage-B authority are authenticated.
    completed_set = _canonical_path(
        stage_b_completed_set, label="Stage-B completed-set path"
    )
    targets = _open_authenticated_stage_b_targets(
        target_grant, completed_set, stage_b_completed_set_sha256
    )
    runtime_evidence = dict(target_grant.runtime_evidence)
    distributional_evidence = _distributional_metrics_unavailable()
    distributional_accumulator = None
    distributional_authority_pin = None
    distributional_arguments = (
        slimbrain_authority,
        slimbrain_authority_sha256,
        slimbrain_dependency_authority_sha256,
        slimbrain_python_executable_sha256,
        slimbrain_device,
    )
    distributional_requested = any(
        value is not None for value in distributional_arguments
    )
    if distributional_requested:
        if any(value is None for value in distributional_arguments):
            raise PostsealEvaluationError(
                "the complete explicit SLIM-Brain authority/runtime binding is required"
            )
    parent = output.parent
    staged: Path | None = None
    held_staging: _HeldStagingRoot | None = None
    try:
        staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.staged-", dir=parent))
        held_staging = _HeldStagingRoot.open(staged, output)
        os.chmod(staged, 0o700)
        private_inputs = staged / ".authenticated_inputs"
        private_inputs.mkdir(mode=0o700)
        prediction_by_scan = predictions.by_scan
        target_by_scan = targets.by_scan
        preflighted_pairs: list[dict[str, Any]] = []
        # This first pass is deliberately limited to authenticated private copies,
        # exact NIfTI/value validation and target-validity derivation.  No feature
        # model, quality metric, texture audit or visualization code is imported or
        # invoked until all 34 pairs have passed this barrier.
        for scan_id in EXPECTED_SCAN_IDS:
            prediction = prediction_by_scan[scan_id]
            target = target_by_scan[scan_id]
            private_subject = private_inputs / scan_id
            private_subject.mkdir(mode=0o700)
            real_copy = private_subject / "real.nii.gz"
            prediction_copy = private_subject / "prediction.nii.gz"
            brain_mask_copy = private_subject / "structural_brain_mask.nii.gz"
            target_validity_copy = private_subject / "target_validity_mask.nii.gz"
            labels_copy = private_subject / "structural_labels.nii.gz"
            _copy_authenticated_file(
                target.native_bold, real_copy, label=f"{scan_id} target"
            )
            _copy_authenticated_file(
                prediction.prediction_snapshot,
                prediction_copy,
                label=f"{scan_id} prediction",
            )
            _copy_authenticated_file(
                target.native_structural_brain_mask,
                brain_mask_copy,
                label=f"{scan_id} structural brain mask",
            )
            _copy_authenticated_file(
                target.native_structural_labels,
                labels_copy,
                label=f"{scan_id} structural labels",
            )
            private_target_validity = _materialize_postseal_target_validity_mask(
                real_copy,
                brain_mask_copy,
                target_validity_copy,
            )
            pair_contract = validate_exact_pair(
                real_copy, prediction_copy, brain_mask_copy, labels_copy
            )
            preflighted_pairs.append(
                {
                    "scan_id": scan_id,
                    "prediction": prediction,
                    "target": target,
                    "private_subject": private_subject,
                    "real_copy": real_copy,
                    "prediction_copy": prediction_copy,
                    "brain_mask_copy": brain_mask_copy,
                    "target_validity_copy": target_validity_copy,
                    "labels_copy": labels_copy,
                    "private_target_validity": private_target_validity,
                    "pair_contract": pair_contract,
                }
            )
        if [item["scan_id"] for item in preflighted_pairs] != list(EXPECTED_SCAN_IDS):
            raise PostsealEvaluationError("complete pair preflight order differs")

        # Lazy imports and the optional feature-model load are now behind the
        # complete exact-pair barrier above.
        from eval.quality import evaluate_4d_pair_quality
        from scripts.visualize_4d_comparison import generate_comparison_outputs
        from scripts.visualize_texture_audit import generate_texture_audit

        if distributional_requested:
            from eval.postseal_metrics import (
                SlimBrainAuthorityError,
                SynthesisMetricAccumulator,
                load_authenticated_slimbrain,
            )
            import torch

            try:
                extractor, _slimbrain_model_authority = load_authenticated_slimbrain(
                    slimbrain_authority,
                    expected_authority_sha256=slimbrain_authority_sha256,
                    dependency_authority_sha256=(slimbrain_dependency_authority_sha256),
                    python_executable_sha256=slimbrain_python_executable_sha256,
                    device=torch.device(slimbrain_device),
                )
            except (SlimBrainAuthorityError, ValueError, TypeError) as exc:
                raise PostsealEvaluationError(
                    f"SLIM-Brain evaluation authority failed closed: {exc}"
                ) from exc
            distributional_accumulator = SynthesisMetricAccumulator(extractor)
            distributional_authority_pin = slimbrain_authority_sha256

        subject_records: list[dict[str, Any]] = []
        for preflighted in preflighted_pairs:
            scan_id = preflighted["scan_id"]
            prediction = preflighted["prediction"]
            target = preflighted["target"]
            private_subject = preflighted["private_subject"]
            real_copy = preflighted["real_copy"]
            prediction_copy = preflighted["prediction_copy"]
            brain_mask_copy = preflighted["brain_mask_copy"]
            target_validity_copy = preflighted["target_validity_copy"]
            labels_copy = preflighted["labels_copy"]
            private_target_validity = preflighted["private_target_validity"]
            pair_contract = preflighted["pair_contract"]
            quality = evaluate_4d_pair_quality(
                real_copy,
                prediction_copy,
                mask_path=target_validity_copy,
                structural_mask_path=brain_mask_copy,
                roi_labels_path=labels_copy,
                require_structured_temporal=True,
            )
            if distributional_accumulator is not None:
                prediction_values = nib.load(str(prediction_copy)).get_fdata(
                    dtype=np.float32
                )
                target_values = nib.load(str(real_copy)).get_fdata(dtype=np.float32)
                prediction_tensor = (
                    torch.from_numpy(np.moveaxis(prediction_values, -1, 0).copy())
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                target_tensor = (
                    torch.from_numpy(np.moveaxis(target_values, -1, 0).copy())
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                distributional_accumulator.update_distributional(
                    prediction_tensor.to(slimbrain_device),
                    target_tensor.to(slimbrain_device),
                )
            quality["sources"] = {
                "real": str(target.native_bold.path),
                "predicted": str(prediction.prediction_snapshot.path),
                "mask": "derived-after-complete-target-authentication",
                "structural_mask": str(target.native_structural_brain_mask.path),
                "roi_labels": str(target.native_structural_labels.path),
            }
            geometry = quality.get("geometry", {})
            if (
                geometry.get("shape") != list(EXPECTED_SHAPE)
                or geometry.get("canonical_axis_codes") != list(EXPECTED_AXIS_CODES)
                or geometry.get("tr_seconds") != EXPECTED_TR_SECONDS
                or geometry.get("mask_derived_from_real") is not False
            ):
                raise PostsealEvaluationError(f"{scan_id} quality geometry differs")
            subject_relative = Path("subjects") / scan_id
            subject_output = staged / subject_relative
            subject_output.mkdir(parents=True, mode=0o700)
            with _HeldOutputDirectory.open(subject_output) as mask_publication:
                _private_snapshot, validity_payload = _snapshot_file(
                    target_validity_copy,
                    label=f"{scan_id} private target-validity mask",
                    expected_sha256=private_target_validity.sha256,
                    expected_size=private_target_validity.size_bytes,
                    capture=True,
                )
                assert validity_payload is not None
                published_target_validity = mask_publication.write_new(
                    "target_validity_mask.nii.gz", validity_payload
                )
            target_validity_descriptor = FileSnapshot(
                path=output / subject_relative / "target_validity_mask.nii.gz",
                sha256=published_target_validity.sha256,
                size_bytes=published_target_validity.size_bytes,
                device=published_target_validity.device,
                inode=published_target_validity.inode,
            )
            texture_outputs = generate_texture_audit(
                real_copy,
                prediction_copy,
                target_validity_copy,
                private_subject / "texture_raw",
                roi_labels_path=labels_copy,
                prefix=scan_id,
                enforce_texture_retention=True,
                enforce_texture_anti_gaming=True,
            )
            with _HeldOutputDirectory.open(
                subject_output / "texture", create=True
            ) as texture_publication:
                texture_evidence = _publish_texture_audit(
                    texture_outputs,
                    publication=texture_publication,
                    output_root=output,
                    subject_relative_dir=subject_relative,
                    temporary_inputs={
                        "real": real_copy,
                        "predicted": prediction_copy,
                        "mask": target_validity_copy,
                        "roi_labels": labels_copy,
                    },
                    input_descriptors={
                        "real": target.native_bold,
                        "predicted": prediction.prediction_snapshot,
                        "mask": target_validity_descriptor,
                        "roi_labels": target.native_structural_labels,
                    },
                    prediction=prediction,
                    predictions=predictions,
                )
            raw_visualization_outputs = generate_comparison_outputs(
                real_copy,
                prediction_copy,
                private_subject / "visualization_raw",
                mask_path=target_validity_copy,
                roi_labels_path=labels_copy,
                tr_seconds=EXPECTED_TR_SECONDS,
                prefix=scan_id,
                fps=fps,
            )
            with _HeldOutputDirectory.open(subject_output) as visual_publication:
                outputs, rewritten_manifest = _publish_visualization_outputs(
                    raw_visualization_outputs,
                    publication=visual_publication,
                    output_root=output,
                    subject_relative_dir=subject_relative,
                    temporary_inputs={
                        "real": real_copy,
                        "predicted": prediction_copy,
                        "mask": target_validity_copy,
                        "roi_labels": labels_copy,
                    },
                    input_descriptors={
                        "real": target.native_bold,
                        "predicted": prediction.prediction_snapshot,
                        "mask": target_validity_descriptor,
                        "roi_labels": target.native_structural_labels,
                    },
                )
            output_inventory = []
            for name, path in sorted(outputs.items()):
                snapshot, _ = _snapshot_file(
                    path,
                    label=f"{scan_id} visualization {name}",
                )
                if name == "manifest":
                    snapshot = rewritten_manifest
                output_inventory.append(
                    {
                        "name": name,
                        "relative_path": path.relative_to(staged).as_posix(),
                        "sha256": snapshot.sha256,
                        "size_bytes": snapshot.size_bytes,
                    }
                )
            subject_record = _signed_record(
                SUBJECT_EVALUATION_SCHEMA,
                {
                    "scan_id": scan_id,
                    "prediction": prediction.prediction_snapshot.descriptor(),
                    "prediction_record": prediction.record_snapshot.descriptor()
                    | {"record_sha256": prediction.record["record_sha256"]},
                    "target": target.native_bold.descriptor(),
                    "target_validity_mask": target_validity_descriptor.descriptor()
                    | {
                        "contract": TARGET_VALIDITY_MASK_CONTRACT,
                        "derivation": (
                            "exact-nonzero-support-across-final-stored-target-"
                            "time-series-after-complete-target-authentication"
                        ),
                        "equals_exact_final_nonzero_support": True,
                    },
                    "structural_brain_mask": (
                        target.native_structural_brain_mask.descriptor()
                    ),
                    "structural_labels": target.native_structural_labels.descriptor(),
                    "stage_b_publication": dict(target.raw["publication"]),
                    "stage_b_success_receipt": dict(target.raw["success_receipt"]),
                    "native_preprocessing_receipt": dict(
                        target.raw["native_preprocessing_receipt"]
                    ),
                    "pair_contract": pair_contract,
                    "quality": quality,
                    "visualization_outputs": output_inventory,
                    "texture_audit": texture_evidence,
                    "fmri_resampling_performed": False,
                    "evaluation_can_authorize_training_or_selection": False,
                },
            )
            record_snapshot = _write_json_new(
                subject_output / "evaluation.json", subject_record
            )
            subject_records.append(
                {
                    "scan_id": scan_id,
                    "pair_contract": pair_contract,
                    "quality": quality,
                    "texture_audit": texture_evidence,
                    "record": {
                        "relative_path": (subject_output / "evaluation.json")
                        .relative_to(staged)
                        .as_posix(),
                        "sha256": record_snapshot.sha256,
                        "size_bytes": record_snapshot.size_bytes,
                        "record_sha256": subject_record["record_sha256"],
                    },
                }
            )
            shutil.rmtree(private_subject)
        paper_metrics = _aggregate_paper_metrics(subject_records)
        if distributional_accumulator is not None:
            if distributional_accumulator.num_samples != len(EXPECTED_SCAN_IDS):
                raise PostsealEvaluationError(
                    "distributional metrics did not consume the complete sealed cohort"
                )
            distributional_evidence = _distributional_metrics_available(
                accumulator=distributional_accumulator,
                artifact_root=staged,
                prediction_set_seal_sha256=predictions.seal_snapshot.sha256,
                stage_b_completed_set_sha256=targets.completed_set.sha256,
            )
        shutil.rmtree(private_inputs)
        data_quality_body = _build_data_quality_summary(
            subject_records,
            distributional_evidence=distributional_evidence,
            verifier_dependency_count=len(targets.verifier_dependencies),
        )
        with _HeldOutputDirectory.open(staged) as cohort_publication:
            cohort_chart = _render_cohort_quality_chart(
                data_quality_body,
                staged / "cohort_quality_summary.png",
                publication=cohort_publication,
            )
        data_quality_body["chart_contract"] = {
            "analytical_question": data_quality_body["analytical_question"],
            "supported_takeaway": (
                "the complete cohort distribution and every scan's pass/fail status "
                "for four spatial-texture and four temporal-dynamics ratios"
            ),
            "canonical_family": "distribution and benchmark",
            "variant": "two-panel horizontal cohort strip with guardrail intervals",
            "observation_count": len(EXPECTED_SCAN_IDS),
            "observation_grain": "one dot per scan per ratio",
            "renderer": "static Matplotlib PNG",
            "palette_policy": "hard two-root cap plus neutrals",
            "palette": {
                "passing_open_circle": "#225ea8",
                "failing_x": "#e67e22",
                "median_diamond": "#263238",
                "guardrail": "#b0bec5",
            },
            "non_color_distinction": (
                "open circle=pass, x=fail, diamond=median, thick bar=guardrail"
            ),
            "no_scan_slice_frame_or_window_visual_selection": True,
        }
        data_quality_body["cohort_chart"] = {
            "relative_path": "cohort_quality_summary.png",
            "sha256": cohort_chart.sha256,
            "size_bytes": cohort_chart.size_bytes,
        }
        report_snapshot = _write_text_new(
            staged / "DATA_QUALITY_REPORT.md",
            _data_quality_markdown(data_quality_body),
        )
        data_quality_body["human_readable_report"] = {
            "relative_path": "DATA_QUALITY_REPORT.md",
            "sha256": report_snapshot.sha256,
            "size_bytes": report_snapshot.size_bytes,
        }
        data_quality_record = _signed_record(DATA_QUALITY_SCHEMA, data_quality_body)
        data_quality_snapshot = _write_json_new(
            staged / "data_quality_summary.json", data_quality_record
        )
        implementation_sources = _source_inventory()
        bundle_inventory = _relative_inventory(
            staged,
            excluded={"evaluation.json", "publication_receipt.json"},
        )
        evaluation_record = _signed_record(
            EVALUATION_SCHEMA,
            {
                "status": "COMPLETE_POSTSEAL_HELDOUT_EVALUATION",
                "role": "fixed-sealed-test-final-report-only",
                "scan_count": len(EXPECTED_SCAN_IDS),
                "ordered_scan_ids": list(EXPECTED_SCAN_IDS),
                "ordered_scan_ids_sha256": EXPECTED_SCAN_IDS_SHA256,
                "sealed_role_csv_sha256": SEALED_ROLE_CSV_SHA256,
                "prediction_set_seal": predictions.seal_snapshot.descriptor()
                | {"record_sha256": predictions.seal["record_sha256"]},
                "stage_b_completed_set": targets.completed_set.descriptor()
                | {"record_sha256": targets.completed_set_record_sha256},
                "stage_b_verifier_source": targets.verifier_source.descriptor(),
                "stage_b_verifier_dependencies": [
                    snapshot.descriptor() for snapshot in targets.verifier_dependencies
                ],
                "stage_b_verification": dict(targets.verification),
                "checkpoint_bindings": pins.checkpoint_bindings(),
                "subjects": [
                    {"scan_id": record["scan_id"], "record": record["record"]}
                    for record in subject_records
                ],
                "paper_metrics_cohort_mean": paper_metrics,
                "distributional_metrics": distributional_evidence,
                "data_quality_summary": {
                    "relative_path": "data_quality_summary.json",
                    "sha256": data_quality_snapshot.sha256,
                    "size_bytes": data_quality_snapshot.size_bytes,
                    "record_sha256": data_quality_record["record_sha256"],
                },
                "fitness_for_use": data_quality_body["fitness_for_use"],
                "texture_release_gate": data_quality_body["texture_release_gate"],
                "quality_pass_count": sum(
                    record["quality"].get("passed") is True
                    for record in subject_records
                ),
                "quality_fail_count": sum(
                    record["quality"].get("passed") is not True
                    for record in subject_records
                ),
                "texture_release_gate_pass_count": len(subject_records),
                "texture_release_gate_fail_count": 0,
                "output_inventory": bundle_inventory,
                "output_inventory_sha256": canonical_sha256(bundle_inventory),
                "implementation_sources": implementation_sources,
                "production_runtime_and_scheduler_evidence": dict(runtime_evidence),
                "production_evaluator_authenticated_before_import": True,
                "prediction_set_fully_authenticated_before_any_target_access": True,
                "all_target_publications_and_bytes_authenticated_before_metrics": True,
                "all_targets_authenticated_before_texture_audits": True,
                "all_34_texture_retention_and_anti_gaming_gates_passed": True,
                "exact_native_grid_no_fmri_resampling": True,
                "independent_structural_mask_and_roi_labels_only": True,
                "synthesis_model_or_checkpoint_loaded": False,
                "evaluation_feature_model_loaded": (
                    distributional_evidence.get("available") is True
                ),
                "training_or_inference_entrypoint_imported": False,
                "authorizes_training": False,
                "authorizes_model_selection": False,
                "authorizes_candidate_selection": False,
                "authorizes_checkpoint_selection": False,
                "authorizes_prediction": False,
                "authorizes_inference": False,
                "can_change_checkpoint_or_prediction_set": False,
            },
        )
        evaluation_snapshot = _write_json_new(
            staged / "evaluation.json", evaluation_record
        )
        receipt = _signed_record(
            PUBLICATION_RECEIPT_SCHEMA,
            {
                "status": "COMMITTED_NO_REPLACE",
                "destination": str(output),
                "evaluation": evaluation_snapshot.descriptor()
                | {"record_sha256": evaluation_record["record_sha256"]},
                "output_inventory_sha256": evaluation_record["output_inventory_sha256"],
                "all_34_subjects_complete": True,
                "all_34_texture_retention_and_anti_gaming_gates_passed": True,
                "published_after_all_subjects_complete": True,
                "no_overwrite": True,
                "evaluation_feedback_to_training_or_selection": False,
            },
        )
        receipt["evaluation"] = {
            "relative_path": "evaluation.json",
            "sha256": evaluation_snapshot.sha256,
            "size_bytes": evaluation_snapshot.size_bytes,
            "record_sha256": evaluation_record["record_sha256"],
        }
        receipt["record_sha256"] = canonical_sha256(
            {key: value for key, value in receipt.items() if key != "record_sha256"}
        )
        receipt_snapshot = _write_json_new(staged / "publication_receipt.json", receipt)
        frozen_tree = _freeze_staged_tree(staged)
        publication_authority_pins = _PublicationAuthorityPins(
            prediction_set_seal_path=str(predictions.seal_snapshot.path),
            prediction_set_seal_sha256=predictions.seal_snapshot.sha256,
            stage_b_completed_set_path=str(targets.completed_set.path),
            stage_b_completed_set_sha256=targets.completed_set.sha256,
            stage_b_verifier_source_path=str(targets.verifier_source.path),
            stage_b_verifier_source_sha256=targets.verifier_source.sha256,
            stage_b_verifier_dependency_paths=tuple(
                str(snapshot.path) for snapshot in targets.verifier_dependencies
            ),
            stage_b_verifier_dependency_sha256s=tuple(
                snapshot.sha256 for snapshot in targets.verifier_dependencies
            ),
        )
        assert held_staging is not None
        held_staging.assert_source_identity()
        _reauthenticate_frozen_tree(staged, frozen_tree)
        _verify_held_staged_publication(
            held_staging,
            evaluation_pin=evaluation_snapshot.sha256,
            receipt_pin=receipt_snapshot.sha256,
            authority_pins=publication_authority_pins,
            distributional_authority_pin=distributional_authority_pin,
        )
        held_staging.assert_source_identity()
        _reauthenticate_frozen_tree(staged, frozen_tree)
        held_staging.assert_source_identity()
        _publish_held_staging_root(held_staging, frozen_tree)
        try:
            _verify_held_committed_publication(
                held_staging,
                evaluation_pin=evaluation_snapshot.sha256,
                receipt_pin=receipt_snapshot.sha256,
                authority_pins=publication_authority_pins,
                distributional_authority_pin=distributional_authority_pin,
            )
        except Exception as exc:
            raise PostsealPublicationUncertainError(
                "evaluation was committed no-replace but committed-tree "
                f"verification failed: {output}"
            ) from exc
        return evaluation_record
    except Exception:
        if held_staging is not None:
            held_staging.cleanup_owned_source()
        elif staged is not None and staged.exists():
            _remove_staged_tree(staged)
        raise
    finally:
        if held_staging is not None:
            held_staging.close()


def run_postseal_evaluation(
    *,
    prediction_root: str | Path,
    pins: EvaluationPins,
    stage_b_completed_set: str | Path,
    stage_b_completed_set_sha256: str,
    output_root: str | Path,
    fps: int = 8,
    slimbrain_authority: str | Path | None = None,
    slimbrain_authority_sha256: str | None = None,
    slimbrain_dependency_authority_sha256: str | None = None,
    slimbrain_python_executable_sha256: str | None = None,
    slimbrain_device: str | None = None,
) -> dict[str, Any]:
    """Run the production path with inert exact inputs and no executable seams."""
    if type(prediction_root) not in {str, _NATIVE_PATH_TYPE}:
        raise PostsealEvaluationError("prediction root path type differs")
    if type(stage_b_completed_set) not in {str, _NATIVE_PATH_TYPE}:
        raise PostsealEvaluationError("Stage-B completed-set path type differs")
    if type(output_root) not in {str, _NATIVE_PATH_TYPE}:
        raise PostsealEvaluationError("evaluation output path type differs")
    if type(stage_b_completed_set_sha256) is not str:
        raise PostsealEvaluationError(
            "Stage-B completed-set SHA-256 must be an exact built-in string"
        )
    completed_digest = _require_sha256(
        stage_b_completed_set_sha256, label="Stage-B completed-set SHA-256"
    )
    if type(fps) is not int or fps < 1:
        raise PostsealEvaluationError(
            "visualization FPS must be a positive built-in int"
        )
    frozen_pins = _snapshot_evaluation_pins(pins)
    prediction_path = _canonical_path(
        prediction_root, label="prediction root", directory=True
    )
    # Deliberately convert, but do not inspect, the target pathname before the
    # prediction set and private target-boundary grant are complete.
    completed_path = Path(stage_b_completed_set)
    output_path = Path(output_root)
    return _run_postseal_evaluation_impl(
        prediction_root=prediction_path,
        pins=frozen_pins,
        stage_b_completed_set=completed_path,
        stage_b_completed_set_sha256=completed_digest,
        output_root=output_path,
        fps=fps,
        slimbrain_authority=(
            None if slimbrain_authority is None else Path(slimbrain_authority)
        ),
        slimbrain_authority_sha256=slimbrain_authority_sha256,
        slimbrain_dependency_authority_sha256=(slimbrain_dependency_authority_sha256),
        slimbrain_python_executable_sha256=slimbrain_python_executable_sha256,
        slimbrain_device=slimbrain_device,
    )


_fixture_authenticate_prediction_set = authenticate_prediction_set
_fixture_run_postseal_evaluation = run_postseal_evaluation


def _sync_fixture_contract() -> None:
    """Mirror only test-monkeypatched geometry constants from production."""
    global EXPECTED_SHAPE, EXPECTED_SPATIAL_SHAPE
    EXPECTED_SHAPE = _production.EXPECTED_SHAPE
    EXPECTED_SPATIAL_SHAPE = _production.EXPECTED_SPATIAL_SHAPE


def authenticate_prediction_set(*args, **kwargs):
    """Test-only access to the production prediction validator."""
    _sync_fixture_contract()
    return _fixture_authenticate_prediction_set(*args, **kwargs)


def run_postseal_evaluation(*args, **kwargs):
    """Test-only synthetic evaluator; never part of the staged runtime."""
    _sync_fixture_contract()
    return _fixture_run_postseal_evaluation(*args, **kwargs)
