"""Read-only, patient-disjoint CONNECT-4 split authorities.

Production entry points never choose or persist a split. They accept only an
existing immutable manifest whose file SHA-256 is pinned by configuration and
whose canonical self-signature, complete scan inventory, role counts and
patient-disjointness all validate. Creation is deliberately confined to the
explicit offline builders in this module and ``scripts/build_split_manifest.py``.
"""
from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import canonical_sha256, sha256_file


IMMUTABLE_SPLIT_SCHEMA = "connect4-immutable-patient-split-v2"
PAPER_PROFILE = "paper-a4-adni-fmriprep-v1"
RECOVERY_PROFILE = "a4-native-recovery-v1"
RECOVERY_AUTHORITY_KIND = "target-blind-published-half-cohort"
SEEDED_AUTHORITY_KIND = "offline-seeded-patient-split"
PARTITION_NAMES = ("train", "development_validation", "sealed_test")

# Exact, independently root-reviewed target-blind half-cohort source authority.
RECOVERY_SOURCE_SHA256 = {
    "selection_manifest_sha256": (
        "834f7bb1954bd36111f17cd886e7ab9a8ca4d80aebf25ffd6d00962ed4f1c547"
    ),
    "selected_train_scans_sha256": (
        "967115c03ed8cb242cbaec2d07f452ad3072dd52181f12eaaf056c7d27bf70ed"
    ),
    "development_validation_scans_sha256": (
        "37eaf9d2330a30d39f0dda2ec312cea546fbd69d76ecf50b1b958e7dcb71e142"
    ),
    "sealed_test_scans_sha256": (
        "b5c04f0ccf4dac74b266233b4705be85c49dbbe589b64e6166aaa8449c0ecfa8"
    ),
    "selection_root_review_sha256": (
        "64063fb5e2cad94dd7fc5fddbc9b1a34c5e5ff9e32d64cecf715cc6f5618d1f2"
    ),
}
RECOVERY_PARTITION_COUNTS = {
    "train": {"patients": 533, "scans": 2078},
    "development_validation": {"patients": 13, "scans": 43},
    "sealed_test": {"patients": 13, "scans": 34},
}
RECOVERY_PARTITION_RECORD_SHA256 = {
    "train": "b47e640750aae8df5d888f3b0a9f28e5fdae245c3703ab869b05146a6c6c0a0b",
    "development_validation": (
        "e65e591fe4681ee47ca1f49e3e622d663fea2e140cfe96711522241fd5b9b75a"
    ),
    "sealed_test": (
        "a3739185d2a3ac4e53d2db06ff62be94ac863916a1457eca99a429fe3ba017b3"
    ),
}


def patient_id(scan_id: str) -> str:
    """Return the patient component of ``<patient>_<session>`` scan IDs."""
    return scan_id.rsplit("_", 1)[0] if "_" in scan_id else scan_id


def _patient_groups(scan_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, scan_id_value in enumerate(scan_ids):
        groups[patient_id(str(scan_id_value))].append(index)
    return dict(groups)


def _split_counts(num_patients: int, val_frac: float, test_frac: float) -> tuple[int, int]:
    if num_patients < 1:
        raise ValueError("scan_ids must contain at least one patient")
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be less than 1")
    n_val = max(1, int(round(num_patients * val_frac))) if val_frac else 0
    n_test = max(1, int(round(num_patients * test_frac))) if test_frac else 0
    if n_val + n_test >= num_patients:
        raise ValueError(
            f"{num_patients} patients are insufficient for non-empty partitions"
        )
    return n_val, n_test


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing split authority: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing split authority: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_records(
    records: object, *, partition: str
) -> list[dict[str, str]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"split partition {partition!r} must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "scan_id",
            "patient_id",
        }:
            raise ValueError(f"split partition {partition!r} record fields differ")
        scan = str(raw_record["scan_id"]).strip()
        patient = str(raw_record["patient_id"]).strip()
        if not scan or not patient:
            raise ValueError(f"split partition {partition!r} has an empty identity")
        normalized.append({"scan_id": scan, "patient_id": patient})
    expected_order = sorted(
        normalized, key=lambda value: (value["patient_id"], value["scan_id"])
    )
    if normalized != expected_order:
        raise ValueError(f"split partition {partition!r} records are not canonical")
    return normalized


def validate_immutable_split_manifest(
    payload: Mapping[str, Any],
    *,
    protocol_profile: str,
) -> dict[str, Any]:
    """Validate a closed, self-signed split record without reading data payloads."""
    if not isinstance(payload, Mapping):
        raise ValueError("split manifest must be a JSON object")
    manifest = dict(payload)
    required_fields = {
        "schema",
        "protocol_profile",
        "authority_kind",
        "source_authority",
        "generation",
        "partitions",
        "partition_counts",
        "partition_record_sha256",
        "total_patient_count",
        "total_scan_count",
        "complete",
        "patient_disjoint",
        "no_replacement",
        "target_payload_access",
        "record_sha256",
    }
    if set(manifest) != required_fields:
        raise ValueError("split manifest fields differ from the closed schema")
    claimed_signature = manifest.pop("record_sha256", None)
    if not _is_sha256(claimed_signature) or canonical_sha256(manifest) != claimed_signature:
        raise ValueError("split manifest canonical signature differs")
    manifest["record_sha256"] = claimed_signature
    if (
        manifest["schema"] != IMMUTABLE_SPLIT_SCHEMA
        or manifest["protocol_profile"] != protocol_profile
        or manifest["complete"] is not True
        or manifest["patient_disjoint"] is not True
        or manifest["no_replacement"] is not True
        or manifest["target_payload_access"] != "ZERO"
    ):
        raise ValueError("split manifest protocol or closed-state assertion differs")
    partitions = manifest["partitions"]
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITION_NAMES):
        raise ValueError("split manifest partition names differ")
    normalized = {
        name: _normalized_records(partitions[name], partition=name)
        for name in PARTITION_NAMES
    }
    all_scans: dict[str, str] = {}
    patient_roles: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    record_digests: dict[str, str] = {}
    for name, records in normalized.items():
        patients = {record["patient_id"] for record in records}
        counts[name] = {"patients": len(patients), "scans": len(records)}
        record_digests[name] = canonical_sha256(records)
        for record in records:
            scan = record["scan_id"]
            patient = record["patient_id"]
            if scan in all_scans:
                raise ValueError(f"scan {scan!r} occurs in multiple split partitions")
            if patient in patient_roles and patient_roles[patient] != name:
                raise ValueError(f"patient {patient!r} occurs in multiple split partitions")
            all_scans[scan] = patient
            patient_roles[patient] = name
    if dict(manifest["partition_counts"]) != counts:
        raise ValueError("split manifest partition count drift")
    if dict(manifest["partition_record_sha256"]) != record_digests:
        raise ValueError("split manifest partition identity differs")
    if (
        manifest["total_patient_count"] != len(patient_roles)
        or manifest["total_scan_count"] != len(all_scans)
    ):
        raise ValueError("split manifest total count drift")

    if protocol_profile == RECOVERY_PROFILE:
        if (
            manifest["authority_kind"] != RECOVERY_AUTHORITY_KIND
            or dict(manifest["source_authority"]) != RECOVERY_SOURCE_SHA256
            or counts != RECOVERY_PARTITION_COUNTS
            or record_digests != RECOVERY_PARTITION_RECORD_SHA256
            or manifest["total_patient_count"] != 559
            or manifest["total_scan_count"] != 2155
            or manifest["generation"]
            != {
                "method": "published-role-lists-no-randomization",
                "selected_training_fraction": "533/1066 patients",
            }
        ):
            raise ValueError("recovery split differs from the exact authorized half cohort")
    elif protocol_profile == PAPER_PROFILE:
        source_authority = manifest["source_authority"]
        generation = manifest["generation"]
        if (
            manifest["authority_kind"] != SEEDED_AUTHORITY_KIND
            or not isinstance(source_authority, Mapping)
            or set(source_authority) != {"cohort_manifest_sha256"}
            or not _is_sha256(source_authority.get("cohort_manifest_sha256"))
            or not isinstance(generation, Mapping)
            or set(generation) != {"method", "seed", "val_frac", "test_frac"}
            or generation.get("method") != "seeded-patient-shuffle"
            or isinstance(generation.get("seed"), bool)
            or not isinstance(generation.get("seed"), int)
        ):
            raise ValueError("paper split must be an explicit offline-seeded authority")
        n_validation, n_test = _split_counts(
            len(patient_roles),
            float(generation["val_frac"]),
            float(generation["test_frac"]),
        )
        if (
            counts["development_validation"]["patients"] != n_validation
            or counts["sealed_test"]["patients"] != n_test
        ):
            raise ValueError("paper split count differs from its signed generation policy")
    else:
        raise ValueError(f"unsupported split protocol profile {protocol_profile!r}")
    return manifest


def load_immutable_split_indices(
    scan_ids: Sequence[str],
    patient_by_scan: Mapping[str, str],
    *,
    manifest_path: str | os.PathLike[str] | None,
    manifest_sha256: str | None,
    protocol_profile: str,
    expected_val_frac: float | None = None,
    expected_test_frac: float | None = None,
    expected_seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Load an existing digest-pinned authority; this function never writes."""
    if manifest_path is None or not str(manifest_path).strip():
        raise FileNotFoundError("an existing immutable split manifest is required")
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"immutable split manifest not found; runtime generation is forbidden: {path}"
        )
    if not _is_sha256(manifest_sha256):
        raise ValueError("immutable split manifest requires a lowercase SHA-256 pin")
    if sha256_file(path) != manifest_sha256:
        raise ValueError("immutable split manifest file SHA-256 differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"immutable split manifest is invalid JSON: {path}") from exc
    manifest = validate_immutable_split_manifest(
        payload, protocol_profile=protocol_profile
    )
    if protocol_profile == PAPER_PROFILE and any(
        value is not None
        for value in (expected_val_frac, expected_test_frac, expected_seed)
    ):
        expected_generation = {
            "method": "seeded-patient-shuffle",
            "seed": expected_seed,
            "val_frac": expected_val_frac,
            "test_frac": expected_test_frac,
        }
        if manifest["generation"] != expected_generation:
            raise ValueError("paper split signed generation policy differs from config")
    scans = [str(value).strip() for value in scan_ids]
    if not scans or any(not value for value in scans) or len(scans) != len(set(scans)):
        raise ValueError("runtime scan inventory must contain unique non-empty IDs")
    if set(patient_by_scan) != set(scans):
        raise ValueError("patient map must cover exactly the scans being split")
    manifest_assignment: dict[str, tuple[str, str]] = {}
    for partition in PARTITION_NAMES:
        for record in manifest["partitions"][partition]:
            manifest_assignment[record["scan_id"]] = (partition, record["patient_id"])
    if set(manifest_assignment) != set(scans):
        missing = sorted(set(scans) - set(manifest_assignment))
        extra = sorted(set(manifest_assignment) - set(scans))
        raise ValueError(
            "immutable split manifest does not match the runtime scan inventory; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    indices = {name: [] for name in PARTITION_NAMES}
    for index, scan in enumerate(scans):
        partition, authorized_patient = manifest_assignment[scan]
        runtime_patient = str(patient_by_scan[scan]).strip()
        if runtime_patient != authorized_patient:
            raise ValueError(f"patient identity differs for authorized scan {scan!r}")
        indices[partition].append(index)
    return (
        indices["train"],
        indices["development_validation"],
        indices["sealed_test"],
    )


def patient_level_train_val_test_split(
    scan_ids: Sequence[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    manifest_path: str | None = None,
    *,
    manifest_sha256: str | None = None,
    protocol_profile: str = PAPER_PROFILE,
) -> tuple[list[int], list[int], list[int]]:
    """Compatibility wrapper around the production read-only authority loader.

    ``val_frac``, ``test_frac`` and ``seed`` remain accepted so old callers fail
    closed at the manifest boundary rather than silently changing semantics.
    They do not generate a split. Use :func:`build_seeded_split_manifest`
    explicitly before runtime.
    """
    scans = [str(value) for value in scan_ids]
    patients = {scan: patient_id(scan) for scan in scans}
    return load_immutable_split_indices(
        scans,
        patients,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        protocol_profile=protocol_profile,
        expected_val_frac=float(val_frac),
        expected_test_frac=float(test_frac),
        expected_seed=int(seed),
    )


def patient_level_split(
    scan_ids: Sequence[str], val_frac: float = 0.15, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Pure in-memory compatibility helper; never used by production runtime."""
    groups = _patient_groups(scan_ids)
    patients = sorted(groups)
    n_val, _ = _split_counts(len(patients), val_frac, 0.0)
    shuffled = patients.copy()
    random.Random(seed).shuffle(shuffled)
    validation_patients = set(shuffled[:n_val])
    train: list[int] = []
    validation: list[int] = []
    for patient in patients:
        (validation if patient in validation_patients else train).extend(groups[patient])
    return sorted(train), sorted(validation)


def _signed_manifest(unsigned: dict[str, Any]) -> dict[str, Any]:
    return {**unsigned, "record_sha256": canonical_sha256(unsigned)}


def build_seeded_split_manifest(
    scan_ids: Sequence[str],
    patient_by_scan: Mapping[str, str],
    *,
    output_path: str | os.PathLike[str],
    cohort_manifest_sha256: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Explicit offline paper-profile builder with write-once publication."""
    if not _is_sha256(cohort_manifest_sha256):
        raise ValueError("offline builder requires the source cohort-manifest SHA-256")
    scans = [str(value).strip() for value in scan_ids]
    if not scans or len(scans) != len(set(scans)) or set(patient_by_scan) != set(scans):
        raise ValueError("offline builder requires one patient identity per unique scan")
    patient_to_scans: dict[str, list[str]] = defaultdict(list)
    for scan in scans:
        patient = str(patient_by_scan[scan]).strip()
        if not patient:
            raise ValueError("offline builder patient identities must be non-empty")
        patient_to_scans[patient].append(scan)
    patients = sorted(patient_to_scans)
    n_validation, n_test = _split_counts(len(patients), val_frac, test_frac)
    shuffled = patients.copy()
    random.Random(seed).shuffle(shuffled)
    sealed = set(shuffled[:n_test])
    development = set(shuffled[n_test : n_test + n_validation])
    role_by_patient = {
        patient: (
            "sealed_test"
            if patient in sealed
            else "development_validation"
            if patient in development
            else "train"
        )
        for patient in patients
    }
    partitions = {name: [] for name in PARTITION_NAMES}
    for scan in scans:
        patient = str(patient_by_scan[scan]).strip()
        partitions[role_by_patient[patient]].append(
            {"scan_id": scan, "patient_id": patient}
        )
    for records in partitions.values():
        records.sort(key=lambda value: (value["patient_id"], value["scan_id"]))
    counts = {
        name: {
            "patients": len({record["patient_id"] for record in records}),
            "scans": len(records),
        }
        for name, records in partitions.items()
    }
    unsigned = {
        "schema": IMMUTABLE_SPLIT_SCHEMA,
        "protocol_profile": PAPER_PROFILE,
        "authority_kind": SEEDED_AUTHORITY_KIND,
        "source_authority": {"cohort_manifest_sha256": cohort_manifest_sha256},
        "generation": {
            "method": "seeded-patient-shuffle",
            "seed": int(seed),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
        },
        "partitions": partitions,
        "partition_counts": counts,
        "partition_record_sha256": {
            name: canonical_sha256(records) for name, records in partitions.items()
        },
        "total_patient_count": len(patients),
        "total_scan_count": len(scans),
        "complete": True,
        "patient_disjoint": True,
        "no_replacement": True,
        "target_payload_access": "ZERO",
    }
    manifest = _signed_manifest(unsigned)
    validate_immutable_split_manifest(manifest, protocol_profile=PAPER_PROFILE)
    _write_new_json(Path(output_path).expanduser(), manifest)
    return manifest


def _read_authorized_role_csv(
    path: Path, *, expected_sha256: str, expected_role: str
) -> list[dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"published role-list SHA-256 differs: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"scan_id", "BID", "role"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"published role-list fields differ: {path}")
        rows = list(reader)
    records = []
    for row in rows:
        if str(row["role"]).strip() != expected_role:
            raise ValueError(f"published role-list role differs: {path}")
        records.append(
            {
                "scan_id": str(row["scan_id"]).strip(),
                "patient_id": str(row["BID"]).strip(),
            }
        )
    return _normalized_records(records, partition=expected_role)


def build_recovery_split_manifest(
    *,
    selection_bundle: str | os.PathLike[str],
    selection_root_review: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Derive the exact authorized half split from metadata-only role lists."""
    bundle = Path(selection_bundle).expanduser()
    review_path = Path(selection_root_review).expanduser()
    if sha256_file(review_path) != RECOVERY_SOURCE_SHA256["selection_root_review_sha256"]:
        raise ValueError("half-cohort root-review SHA-256 differs")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("decision") != "APPROVED_AS_TARGET_BLIND_HALF_SELECTION_INPUT":
        raise ValueError("half-cohort source is not independently approved")
    selection_manifest_path = bundle / "selection_manifest.json"
    if sha256_file(selection_manifest_path) != RECOVERY_SOURCE_SHA256[
        "selection_manifest_sha256"
    ]:
        raise ValueError("half-cohort selection-manifest SHA-256 differs")
    partitions = {
        "train": _read_authorized_role_csv(
            bundle / "selected_train_scans.csv",
            expected_sha256=RECOVERY_SOURCE_SHA256["selected_train_scans_sha256"],
            expected_role="train",
        ),
        "development_validation": _read_authorized_role_csv(
            bundle / "development_validation_scans.csv",
            expected_sha256=RECOVERY_SOURCE_SHA256[
                "development_validation_scans_sha256"
            ],
            expected_role="development-validation",
        ),
        "sealed_test": _read_authorized_role_csv(
            bundle / "sealed_test_scans.csv",
            expected_sha256=RECOVERY_SOURCE_SHA256["sealed_test_scans_sha256"],
            expected_role="sealed-test",
        ),
    }
    counts = {
        name: {
            "patients": len({record["patient_id"] for record in records}),
            "scans": len(records),
        }
        for name, records in partitions.items()
    }
    unsigned = {
        "schema": IMMUTABLE_SPLIT_SCHEMA,
        "protocol_profile": RECOVERY_PROFILE,
        "authority_kind": RECOVERY_AUTHORITY_KIND,
        "source_authority": dict(RECOVERY_SOURCE_SHA256),
        "generation": {
            "method": "published-role-lists-no-randomization",
            "selected_training_fraction": "533/1066 patients",
        },
        "partitions": partitions,
        "partition_counts": counts,
        "partition_record_sha256": {
            name: canonical_sha256(records) for name, records in partitions.items()
        },
        "total_patient_count": len(
            {
                record["patient_id"]
                for records in partitions.values()
                for record in records
            }
        ),
        "total_scan_count": sum(len(records) for records in partitions.values()),
        "complete": True,
        "patient_disjoint": True,
        "no_replacement": True,
        "target_payload_access": "ZERO",
    }
    manifest = _signed_manifest(unsigned)
    validate_immutable_split_manifest(manifest, protocol_profile=RECOVERY_PROFILE)
    _write_new_json(Path(output_path).expanduser(), manifest)
    return manifest


__all__ = [
    "IMMUTABLE_SPLIT_SCHEMA",
    "PAPER_PROFILE",
    "RECOVERY_PROFILE",
    "RECOVERY_SOURCE_SHA256",
    "build_recovery_split_manifest",
    "build_seeded_split_manifest",
    "load_immutable_split_indices",
    "patient_id",
    "patient_level_split",
    "patient_level_train_val_test_split",
    "validate_immutable_split_manifest",
]
