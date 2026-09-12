"""Canonical rank-zero admission authority for distributed CONNECT-4 loading.

Only rank zero is permitted to perform the expensive whole-cohort validation.
The resulting closed JSON authority is broadcast as bounded canonical bytes over
tensor collectives; pickle/object collectives are deliberately not used.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from architecture_contract import (
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    TRAINING_CHECKPOINT_FORMAT,
)
from .provenance import canonical_sha256


COHORT_ADMISSION_SCHEMA = "connect4-distributed-cohort-admission-v1"
COHORT_ADMISSION_STATUS = "ADMITTED_COMPLETE_FIXED_COHORT"
COHORT_ADMISSION_IDENTITY_SCHEMA = (
    "connect4-distributed-cohort-admission-identity-v1"
)
PRODUCTION_TRAINING_RUNTIME_SCHEMA = (
    "connect4-v10-ciai-four-a100-training-runtime-v1"
)
VALIDATION_SHARD_CONTRACT_SCHEMA = (
    "connect4-development-first-n-ddp-validation-shards-v1"
)
MAX_COHORT_ADMISSION_BYTES = 64 * 1024 * 1024
_PARTITION_KEYS = frozenset({"train", "development_validation", "sealed_test"})
_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "authority_record_sha256",
        "authority_canonical_bytes_sha256",
        "data_root_sha256",
        "runtime_identity_sha256",
        "config_sha256",
        "ordered_scan_ids_sha256",
        "partitions_sha256",
        "split_identity_sha256",
        "artifact_identity_sha256",
        "structural_artifact_identities_sha256",
        "dataset_state_sha256",
        "world_size",
        "record_sha256",
    }
)
_AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "status",
        "protocol_profile",
        "world_size",
        "config_sha256",
        "runtime_identity",
        "runtime_identity_sha256",
        "ordered_scan_ids",
        "ordered_scan_ids_sha256",
        "partitions",
        "partitions_sha256",
        "split_identity",
        "split_identity_sha256",
        "artifact_identity",
        "artifact_identity_sha256",
        "structural_artifact_identities_sha256",
        "dataset_state",
        "dataset_state_sha256",
        "complete",
        "no_replacement",
        "no_refill",
        "nonzero_rank_target_array_materialization_during_admission",
        "record_sha256",
    }
)
_PRODUCTION_RUNTIME_FIELDS = frozenset(
    {
        "schema",
        "synthesis_architecture_schema",
        "synthesis_architecture_sha256",
        "training_checkpoint_format",
        "slurm_job_id",
        "slurm_partition",
        "slurm_qos",
        "allocated_hostnames",
        "hostname",
        "scontrol_record_sha256",
        "source_tree_sha256",
        "source_file_count",
        "python_executable",
        "python_executable_sha256",
        "isolated_python",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "amp_dtype",
        "world_size",
        "gpu_records",
        "slab_selection",
        "canonical_sha256",
    }
)
_PRODUCTION_GPU_FIELDS = frozenset(
    {
        "rank",
        "local_rank",
        "gpu_uuid",
        "gpu_name",
        "gpu_total_memory_gib",
        "nvidia_driver_version",
        "cuda_visible_device",
    }
)
_VALIDATION_SHARD_FIELDS = frozenset(
    {
        "schema",
        "cohort_admission_identity_record_sha256",
        "cohort_admission_data_root_sha256",
        "ordered_scan_ids_sha256",
        "partitions_sha256",
        "development_partition_size",
        "global_limit",
        "global_first_n_positions",
        "world_size",
        "rank_to_global_positions",
        "complete",
        "disjoint",
        "no_replacement",
        "deterministic_assignment",
        "record_sha256",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_identity_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("canonical_sha256", None)
    if not _is_sha256(claimed) or canonical_sha256(unsigned) != claimed:
        raise RuntimeError("production runtime identity digest differs")
    return claimed


def validate_training_runtime_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("production runtime identity is missing")
    identity = dict(value)
    _runtime_identity_sha256(identity)
    return identity


def validate_production_training_runtime_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact four-rank CIAI A100 production runtime record."""

    identity = validate_training_runtime_identity(value)
    if set(identity) != _PRODUCTION_RUNTIME_FIELDS:
        raise RuntimeError("production runtime identity fields differ")
    if (
        identity.get("schema") != PRODUCTION_TRAINING_RUNTIME_SCHEMA
        or identity.get("synthesis_architecture_schema")
        != SYNTHESIS_ARCHITECTURE_SCHEMA
        or identity.get("synthesis_architecture_sha256")
        != SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
        or identity.get("training_checkpoint_format")
        != TRAINING_CHECKPOINT_FORMAT
        or not str(identity.get("slurm_job_id", "")).isdigit()
        or identity.get("slurm_partition") != "cscc-gpu-p"
        or identity.get("slurm_qos") != "cscc-gpu-qos"
        or identity.get("world_size") != 4
        or identity.get("amp_dtype") != "bfloat16"
        or identity.get("isolated_python") is not True
        or not isinstance(identity.get("python_executable"), str)
        or not identity["python_executable"].startswith("/")
        or "mica" in identity["python_executable"].lower()
    ):
        raise RuntimeError("production runtime class differs")
    hosts = identity.get("allocated_hostnames")
    hostname = identity.get("hostname")
    if (
        not isinstance(hosts, list)
        or len(hosts) != 1
        or any(not isinstance(host, str) or not host for host in hosts)
        or not isinstance(hostname, str)
        or hostname not in hosts
        or any("login" in host.lower() for host in [*hosts, hostname])
    ):
        raise RuntimeError("production runtime compute-host identity differs")
    if any(
        not _is_sha256(identity.get(field))
        for field in (
            "scontrol_record_sha256",
            "source_tree_sha256",
            "python_executable_sha256",
        )
    ):
        raise RuntimeError("production runtime content digest differs")
    expected_scheduler_digest = canonical_sha256(
        {
            "schema": "connect4-ciai-scontrol-allocation-identity-v1",
            "slurm_job_id": identity["slurm_job_id"],
            "slurm_partition": identity["slurm_partition"],
            "slurm_qos": identity["slurm_qos"],
            "allocated_hostnames": hosts,
            "slurm_num_nodes": 1,
            "allocated_gpu_count": 4,
        }
    )
    if identity["scontrol_record_sha256"] != expected_scheduler_digest:
        raise RuntimeError("production scheduler-allocation digest differs")
    if (
        isinstance(identity.get("source_file_count"), bool)
        or not isinstance(identity.get("source_file_count"), int)
        or identity["source_file_count"] < 1
        or not isinstance(identity.get("torch_version"), str)
        or not identity["torch_version"]
        or not isinstance(identity.get("cuda_version"), str)
        or not identity["cuda_version"]
        or isinstance(identity.get("cudnn_version"), bool)
        or not isinstance(identity.get("cudnn_version"), int)
        or identity["cudnn_version"] < 1
    ):
        raise RuntimeError("production runtime software identity differs")

    gpu_records = identity.get("gpu_records")
    if not isinstance(gpu_records, list) or len(gpu_records) != 4:
        raise RuntimeError("production runtime requires four GPU records")
    normalized_gpu_records: list[dict[str, Any]] = []
    for expected_rank, raw_record in enumerate(gpu_records):
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("production runtime GPU record is malformed")
        record = dict(raw_record)
        if set(record) != _PRODUCTION_GPU_FIELDS:
            raise RuntimeError("production runtime GPU record fields differ")
        memory_gib = record.get("gpu_total_memory_gib")
        if (
            record.get("rank") != expected_rank
            or record.get("local_rank") != expected_rank
            or not isinstance(record.get("gpu_uuid"), str)
            or not record["gpu_uuid"].startswith("GPU-")
            or not isinstance(record.get("gpu_name"), str)
            or "A100" not in record["gpu_name"]
            or isinstance(memory_gib, bool)
            or not isinstance(memory_gib, (int, float))
            or not 39.0 <= float(memory_gib) <= 41.0
            or not isinstance(record.get("nvidia_driver_version"), str)
            or not record["nvidia_driver_version"]
            or not isinstance(record.get("cuda_visible_device"), str)
            or not record["cuda_visible_device"]
        ):
            raise RuntimeError("production runtime GPU identity differs")
        normalized_gpu_records.append(record)
    if (
        len({record["gpu_uuid"] for record in normalized_gpu_records}) != 4
        or len(
            {record["cuda_visible_device"] for record in normalized_gpu_records}
        )
        != 4
        or len(
            {record["nvidia_driver_version"] for record in normalized_gpu_records}
        )
        != 1
    ):
        raise RuntimeError("production runtime rank/device mapping differs")

    slab = identity.get("slab_selection")
    if not isinstance(slab, Mapping):
        raise RuntimeError("production runtime slab-selection identity is missing")
    required_slab_fields = {
        "profile",
        "depth_slab_size",
        "summary_path",
        "summary_sha256",
        "ddp_smoke_path",
        "ddp_smoke_sha256",
        "measured_ddp_step_seconds",
        "recovery_half_epoch_eta_seconds",
    }
    if not required_slab_fields.issubset(slab):
        raise RuntimeError("production runtime slab-selection fields differ")
    core = slab.get("depth_slab_size")
    measured_seconds = slab.get("measured_ddp_step_seconds")
    if (
        slab.get("profile") not in {"recovery", "paper"}
        or isinstance(core, bool)
        or not isinstance(core, int)
        or core < 1
        or not isinstance(slab.get("summary_path"), str)
        or not slab["summary_path"].startswith("/")
        or not _is_sha256(slab.get("summary_sha256"))
        or not isinstance(slab.get("ddp_smoke_path"), str)
        or not slab["ddp_smoke_path"].startswith("/")
        or not _is_sha256(slab.get("ddp_smoke_sha256"))
        or isinstance(measured_seconds, bool)
        or not isinstance(measured_seconds, (int, float))
        or float(measured_seconds) <= 0.0
    ):
        raise RuntimeError("production runtime slab-selection identity differs")
    recovery_eta = slab.get("recovery_half_epoch_eta_seconds")
    if slab["profile"] == "recovery":
        if (
            isinstance(recovery_eta, bool)
            or not isinstance(recovery_eta, (int, float))
            or float(recovery_eta) <= 0.0
        ):
            raise RuntimeError("production recovery runtime ETA identity differs")
    elif recovery_eta is not None:
        raise RuntimeError("production paper runtime has a recovery ETA")
    return identity


def build_validation_shard_contract(
    cohort_admission_identity_value: Mapping[str, Any],
    *,
    development_partition_size: int,
    global_limit: int,
    world_size: int,
) -> dict[str, Any]:
    """Build the closed global first-N development-validation shard map."""

    admission = validate_cohort_admission_identity(
        cohort_admission_identity_value
    )
    if (
        isinstance(development_partition_size, bool)
        or not isinstance(development_partition_size, int)
        or development_partition_size < 1
        or isinstance(global_limit, bool)
        or not isinstance(global_limit, int)
        or not 1 <= global_limit <= development_partition_size
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or admission["world_size"] != world_size
    ):
        raise RuntimeError("validation-shard construction arguments differ")
    first_n = list(range(global_limit))
    rank_to_positions = {
        str(rank): [position for position in first_n if position % world_size == rank]
        for rank in range(world_size)
    }
    contract = {
        "schema": VALIDATION_SHARD_CONTRACT_SCHEMA,
        "cohort_admission_identity_record_sha256": admission["record_sha256"],
        "cohort_admission_data_root_sha256": admission["data_root_sha256"],
        "ordered_scan_ids_sha256": admission["ordered_scan_ids_sha256"],
        "partitions_sha256": admission["partitions_sha256"],
        "development_partition_size": development_partition_size,
        "global_limit": global_limit,
        "global_first_n_positions": first_n,
        "world_size": world_size,
        "rank_to_global_positions": rank_to_positions,
        "complete": True,
        "disjoint": True,
        "no_replacement": True,
        "deterministic_assignment": "development-position-mod-world-size",
    }
    contract["record_sha256"] = canonical_sha256(contract)
    return validate_validation_shard_contract(
        contract,
        expected_cohort_admission_identity=admission,
        expected_world_size=world_size,
    )


def validate_validation_shard_contract(
    value: Mapping[str, Any],
    *,
    expected_cohort_admission_identity: Mapping[str, Any] | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    """Validate a complete/disjoint deterministic validation work partition."""

    if not isinstance(value, Mapping) or set(value) != _VALIDATION_SHARD_FIELDS:
        raise RuntimeError("validation-shard contract fields differ")
    contract = dict(value)
    claimed = contract.get("record_sha256")
    unsigned = {key: item for key, item in contract.items() if key != "record_sha256"}
    if (
        contract.get("schema") != VALIDATION_SHARD_CONTRACT_SCHEMA
        or contract.get("complete") is not True
        or contract.get("disjoint") is not True
        or contract.get("no_replacement") is not True
        or contract.get("deterministic_assignment")
        != "development-position-mod-world-size"
        or not _is_sha256(claimed)
        or canonical_sha256(unsigned) != claimed
    ):
        raise RuntimeError("validation-shard contract status or digest differs")
    for field in (
        "cohort_admission_identity_record_sha256",
        "cohort_admission_data_root_sha256",
        "ordered_scan_ids_sha256",
        "partitions_sha256",
    ):
        if not _is_sha256(contract.get(field)):
            raise RuntimeError("validation-shard admission binding differs")
    development_size = contract.get("development_partition_size")
    global_limit = contract.get("global_limit")
    world_size = contract.get("world_size")
    if (
        isinstance(development_size, bool)
        or not isinstance(development_size, int)
        or development_size < 1
        or isinstance(global_limit, bool)
        or not isinstance(global_limit, int)
        or not 1 <= global_limit <= development_size
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise RuntimeError("validation-shard dimensions differ")
    first_n = contract.get("global_first_n_positions")
    rank_to_positions = contract.get("rank_to_global_positions")
    expected_first_n = list(range(global_limit))
    expected_mapping = {
        str(rank): [
            position
            for position in expected_first_n
            if position % world_size == rank
        ]
        for rank in range(world_size)
    }
    if first_n != expected_first_n or rank_to_positions != expected_mapping:
        raise RuntimeError("validation-shard deterministic partition differs")
    flattened = [
        position
        for rank in range(world_size)
        for position in rank_to_positions[str(rank)]
    ]
    if sorted(flattened) != expected_first_n or len(flattened) != len(set(flattened)):
        raise RuntimeError("validation-shard partition is incomplete or overlapping")
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError("validation-shard world size differs")
    if expected_cohort_admission_identity is not None:
        admission = validate_cohort_admission_identity(
            expected_cohort_admission_identity
        )
        if (
            admission["world_size"] != world_size
            or contract["cohort_admission_identity_record_sha256"]
            != admission["record_sha256"]
            or contract["cohort_admission_data_root_sha256"]
            != admission["data_root_sha256"]
            or contract["ordered_scan_ids_sha256"]
            != admission["ordered_scan_ids_sha256"]
            or contract["partitions_sha256"] != admission["partitions_sha256"]
        ):
            raise RuntimeError("validation-shard cohort-admission binding differs")
    return contract


def seal_cohort_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    authority = dict(value)
    authority.pop("record_sha256", None)
    authority["record_sha256"] = canonical_sha256(authority)
    return authority


def validate_cohort_admission(
    value: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_world_size: int,
    expected_runtime_identity_sha256: str,
) -> dict[str, Any]:
    """Validate the complete closed authority without opening cohort artifacts."""

    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise RuntimeError("distributed cohort-admission fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != COHORT_ADMISSION_SCHEMA
        or authority.get("status") != COHORT_ADMISSION_STATUS
        or authority.get("complete") is not True
        or authority.get("no_replacement") is not True
        or authority.get("no_refill") is not True
        or authority.get("nonzero_rank_target_array_materialization_during_admission")
        is not False
    ):
        raise RuntimeError("distributed cohort-admission status differs")
    if (
        not _is_sha256(authority.get("record_sha256"))
        or canonical_sha256(
            {key: item for key, item in authority.items() if key != "record_sha256"}
        )
        != authority["record_sha256"]
    ):
        raise RuntimeError("distributed cohort-admission record SHA-256 differs")
    if (
        not _is_sha256(expected_config_sha256)
        or authority.get("config_sha256") != expected_config_sha256
    ):
        raise RuntimeError("distributed cohort-admission config differs")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size < 1
        or authority.get("world_size") != expected_world_size
    ):
        raise RuntimeError("distributed cohort-admission world size differs")
    runtime_identity = authority.get("runtime_identity")
    if (
        not isinstance(runtime_identity, Mapping)
        or not _is_sha256(expected_runtime_identity_sha256)
        or authority.get("runtime_identity_sha256")
        != expected_runtime_identity_sha256
    ):
        raise RuntimeError("distributed cohort-admission runtime identity differs")
    try:
        observed_runtime_identity_sha256 = _runtime_identity_sha256(runtime_identity)
    except RuntimeError as exc:
        raise RuntimeError(
            "distributed cohort-admission runtime identity differs"
        ) from exc
    if observed_runtime_identity_sha256 != expected_runtime_identity_sha256:
        raise RuntimeError("distributed cohort-admission runtime identity differs")
    if not isinstance(authority.get("protocol_profile"), str) or not authority[
        "protocol_profile"
    ]:
        raise RuntimeError("distributed cohort-admission profile is missing")

    scan_ids = authority.get("ordered_scan_ids")
    if (
        not isinstance(scan_ids, list)
        or not scan_ids
        or scan_ids != sorted(set(scan_ids))
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in scan_ids)
        or authority.get("ordered_scan_ids_sha256") != canonical_sha256(scan_ids)
    ):
        raise RuntimeError("distributed cohort-admission scan order differs")
    partitions = authority.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != _PARTITION_KEYS:
        raise RuntimeError("distributed cohort-admission partitions differ")
    normalized_partitions: dict[str, list[int]] = {}
    all_indices: list[int] = []
    for name in sorted(_PARTITION_KEYS):
        raw = partitions[name]
        if (
            not isinstance(raw, list)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(scan_ids)
                for index in raw
            )
            or len(set(raw)) != len(raw)
        ):
            raise RuntimeError(f"distributed cohort-admission {name} indices differ")
        normalized_partitions[name] = list(raw)
        all_indices.extend(raw)
    if sorted(all_indices) != list(range(len(scan_ids))):
        raise RuntimeError("distributed cohort-admission partitions do not cover cohort")
    if authority.get("partitions_sha256") != canonical_sha256(normalized_partitions):
        raise RuntimeError("distributed cohort-admission partition digest differs")

    split_identity = authority.get("split_identity")
    artifact_identity = authority.get("artifact_identity")
    dataset_state = authority.get("dataset_state")
    if (
        not isinstance(split_identity, Mapping)
        or authority.get("split_identity_sha256")
        != canonical_sha256(split_identity)
    ):
        raise RuntimeError("distributed cohort-admission split identity differs")
    if (
        not isinstance(artifact_identity, Mapping)
        or authority.get("artifact_identity_sha256")
        != canonical_sha256(artifact_identity)
    ):
        raise RuntimeError("distributed cohort-admission artifact identity differs")
    structural_digest = authority.get("structural_artifact_identities_sha256")
    if (
        not _is_sha256(structural_digest)
        or artifact_identity.get("structural_artifact_identities_sha256")
        != structural_digest
    ):
        raise RuntimeError("distributed cohort-admission structural root differs")
    if (
        not isinstance(dataset_state, Mapping)
        or authority.get("dataset_state_sha256") != canonical_sha256(dataset_state)
    ):
        raise RuntimeError("distributed cohort-admission dataset state differs")
    state_scans = dataset_state.get("scan_ids")
    state_scan_roles = dataset_state.get("scan_roles")
    state_targets = dataset_state.get("target_scan_ids")
    state_roles = dataset_state.get("target_scan_roles")
    expected_targets = {
        scan_ids[index]
        for partition in ("train", "development_validation")
        for index in normalized_partitions[partition]
    }
    expected_roles = {
        **{
            scan_ids[index]: "train"
            for index in normalized_partitions["train"]
        },
        **{
            scan_ids[index]: "development-validation"
            for index in normalized_partitions["development_validation"]
        },
    }
    expected_scan_roles = {
        **expected_roles,
        **{
            scan_ids[index]: "sealed-test"
            for index in normalized_partitions["sealed_test"]
        },
    }
    if (
        state_scans != scan_ids
        or state_scan_roles != dict(sorted(expected_scan_roles.items()))
        or state_targets != sorted(expected_targets)
        or state_roles != dict(sorted(expected_roles.items()))
    ):
        raise RuntimeError("distributed cohort-admission all-scan roles differ")
    brainlm_scope = artifact_identity.get("brainlm_authority_scope")
    if authority.get("protocol_profile") == "a4-native-recovery-v1":
        expected_role_items = [
            {"scan_id": scan_id, "role": expected_roles[scan_id]}
            for scan_id in sorted(expected_roles)
        ]
        sealed_ids = sorted(
            scan_ids[index] for index in normalized_partitions["sealed_test"]
        )
        if (
            not isinstance(brainlm_scope, Mapping)
            or brainlm_scope.get("schema")
            != "connect4-brainlm-authority-split-scope-v1"
            or brainlm_scope.get("split_identity_sha256")
            != canonical_sha256(split_identity)
            or brainlm_scope.get("split_assignment_sha256")
            != split_identity.get("sha256")
            or brainlm_scope.get("target_scan_roles_sha256")
            != canonical_sha256(expected_role_items)
            or brainlm_scope.get("authorized_scan_count") != len(expected_roles)
            or brainlm_scope.get("sealed_scan_count") != len(sealed_ids)
            or brainlm_scope.get("sealed_scan_ids_sha256")
            != canonical_sha256(sealed_ids)
            or brainlm_scope.get("sealed_scans_authorized") is not False
            or brainlm_scope.get("functional_target_bytes_opened") is not False
        ):
            raise RuntimeError(
                "distributed cohort-admission BrainLM split scope differs"
            )
    elif brainlm_scope is not None:
        raise RuntimeError(
            "paper cohort-admission cannot carry a recovery BrainLM split scope"
        )
    split_partitions = split_identity.get("partitions")
    expected_split_scan_sets = {
        "train": {
            scan_ids[index] for index in normalized_partitions["train"]
        },
        "validation": {
            scan_ids[index]
            for index in normalized_partitions["development_validation"]
        },
        "test": {
            scan_ids[index] for index in normalized_partitions["sealed_test"]
        },
    }
    if not isinstance(split_partitions, Mapping) or set(split_partitions) != set(
        expected_split_scan_sets
    ):
        raise RuntimeError("distributed cohort-admission split partitions differ")
    for name, expected_scan_set in expected_split_scan_sets.items():
        records = split_partitions[name]
        if (
            not isinstance(records, list)
            or any(not isinstance(record, Mapping) for record in records)
            or {record.get("scan_id") for record in records} != expected_scan_set
            or len(records) != len(expected_scan_set)
        ):
            raise RuntimeError(
                f"distributed cohort-admission {name} split binding differs"
            )
    return authority


def build_cohort_admission(
    *,
    protocol_profile: str,
    world_size: int,
    config_sha256: str,
    runtime_identity: Mapping[str, Any],
    ordered_scan_ids: Sequence[str],
    train_indices: Sequence[int],
    development_indices: Sequence[int],
    sealed_indices: Sequence[int],
    split_identity: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    dataset_state: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_digest = _runtime_identity_sha256(runtime_identity)
    partitions = {
        "development_validation": list(development_indices),
        "sealed_test": list(sealed_indices),
        "train": list(train_indices),
    }
    scans = list(ordered_scan_ids)
    authority = seal_cohort_admission(
        {
            "schema": COHORT_ADMISSION_SCHEMA,
            "status": COHORT_ADMISSION_STATUS,
            "protocol_profile": str(protocol_profile),
            "world_size": int(world_size),
            "config_sha256": config_sha256,
            "runtime_identity": dict(runtime_identity),
            "runtime_identity_sha256": runtime_digest,
            "ordered_scan_ids": scans,
            "ordered_scan_ids_sha256": canonical_sha256(scans),
            "partitions": partitions,
            "partitions_sha256": canonical_sha256(partitions),
            "split_identity": dict(split_identity),
            "split_identity_sha256": canonical_sha256(split_identity),
            "artifact_identity": dict(artifact_identity),
            "artifact_identity_sha256": canonical_sha256(artifact_identity),
            "structural_artifact_identities_sha256": artifact_identity.get(
                "structural_artifact_identities_sha256"
            ),
            "dataset_state": dict(dataset_state),
            "dataset_state_sha256": canonical_sha256(dataset_state),
            "complete": True,
            "no_replacement": True,
            "no_refill": True,
            "nonzero_rank_target_array_materialization_during_admission": False,
        }
    )
    return validate_cohort_admission(
        authority,
        expected_config_sha256=config_sha256,
        expected_world_size=world_size,
        expected_runtime_identity_sha256=runtime_digest,
    )


def cohort_admission_identity(authority: Mapping[str, Any]) -> dict[str, Any]:
    payload = canonical_json_bytes(authority)
    stable_authority = {
        key: value
        for key, value in authority.items()
        if key not in {"runtime_identity", "runtime_identity_sha256", "record_sha256"}
    }
    identity = {
        "schema": COHORT_ADMISSION_IDENTITY_SCHEMA,
        "authority_record_sha256": authority["record_sha256"],
        "authority_canonical_bytes_sha256": hashlib.sha256(payload).hexdigest(),
        "data_root_sha256": canonical_sha256(stable_authority),
        "runtime_identity_sha256": authority["runtime_identity_sha256"],
        "config_sha256": authority["config_sha256"],
        "ordered_scan_ids_sha256": authority["ordered_scan_ids_sha256"],
        "partitions_sha256": authority["partitions_sha256"],
        "split_identity_sha256": authority["split_identity_sha256"],
        "artifact_identity_sha256": authority["artifact_identity_sha256"],
        "structural_artifact_identities_sha256": authority[
            "structural_artifact_identities_sha256"
        ],
        "dataset_state_sha256": authority["dataset_state_sha256"],
        "world_size": authority["world_size"],
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return identity


def validate_cohort_admission_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise RuntimeError("cohort-admission identity fields differ")
    identity = dict(value)
    unsigned = dict(identity)
    recorded = unsigned.pop("record_sha256", None)
    digest_fields = set(_IDENTITY_FIELDS) - {"schema", "world_size"}
    if (
        identity.get("schema") != COHORT_ADMISSION_IDENTITY_SCHEMA
        or not _is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
        or any(not _is_sha256(identity.get(field)) for field in digest_fields)
        or isinstance(identity.get("world_size"), bool)
        or not isinstance(identity.get("world_size"), int)
        or identity["world_size"] < 1
    ):
        raise RuntimeError("cohort-admission identity differs")
    return identity


def _collective_device() -> torch.device:
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL cohort admission requires a CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_rank_zero_cohort_admission(
    builder: Callable[[], Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    expected_config_sha256: str,
    expected_runtime_identity_sha256: str,
) -> dict[str, Any]:
    """Build only on rank zero, then broadcast bounded canonical JSON tensors."""

    if world_size == 1:
        return validate_cohort_admission(
            builder(),
            expected_config_sha256=expected_config_sha256,
            expected_world_size=1,
            expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        )
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() != rank
        or dist.get_world_size() != world_size
    ):
        raise RuntimeError("distributed cohort admission requires the active process group")
    device = _collective_device()
    status = 0
    payload = b""
    digest = b"\0" * 32
    if rank == 0:
        try:
            authority = validate_cohort_admission(
                builder(),
                expected_config_sha256=expected_config_sha256,
                expected_world_size=world_size,
                expected_runtime_identity_sha256=(
                    expected_runtime_identity_sha256
                ),
            )
            payload = canonical_json_bytes(authority)
            if not 1 <= len(payload) <= MAX_COHORT_ADMISSION_BYTES:
                raise RuntimeError("cohort-admission authority exceeds the byte limit")
            status = 1
            digest = hashlib.sha256(bytes([status]) + payload).digest()
        except Exception as exc:  # every other rank must receive one terminal state
            failure = {
                "error_type": type(exc).__name__,
                "message": str(exc)[:16_384],
            }
            payload = canonical_json_bytes(failure)
            status = 0
            digest = hashlib.sha256(bytes([status]) + payload).digest()

    header = torch.tensor(
        [status, len(payload)], dtype=torch.int64, device=device
    ) if rank == 0 else torch.empty(2, dtype=torch.int64, device=device)
    dist.broadcast(header, src=0)
    received_status, payload_size = (int(value) for value in header.cpu().tolist())
    if received_status not in {0, 1}:
        raise RuntimeError("rank-zero cohort-admission status is invalid")
    if not 1 <= payload_size <= MAX_COHORT_ADMISSION_BYTES:
        raise RuntimeError("rank-zero cohort-admission payload length is invalid")

    digest_tensor = (
        torch.frombuffer(bytearray(digest), dtype=torch.uint8).clone().to(device)
        if rank == 0
        else torch.empty(32, dtype=torch.uint8, device=device)
    )
    dist.broadcast(digest_tensor, src=0)
    payload_tensor = (
        torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().to(device)
        if rank == 0
        else torch.empty(payload_size, dtype=torch.uint8, device=device)
    )
    dist.broadcast(payload_tensor, src=0)
    received = payload_tensor.cpu().numpy().tobytes()
    received_digest = digest_tensor.cpu().numpy().tobytes()
    if (
        hashlib.sha256(bytes([received_status]) + received).digest()
        != received_digest
    ):
        raise RuntimeError("rank-zero cohort-admission transport digest differs")
    try:
        value = json.loads(received.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("rank-zero cohort-admission payload is invalid JSON") from exc
    if received_status != 1:
        if not isinstance(value, Mapping):
            raise RuntimeError("rank-zero cohort admission failed without evidence")
        raise RuntimeError(
            "rank-zero cohort admission failed: "
            f"{value.get('error_type', 'Error')}: {value.get('message', '')}"
        )
    return validate_cohort_admission(
        value,
        expected_config_sha256=expected_config_sha256,
        expected_world_size=world_size,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
    )


__all__ = [
    "COHORT_ADMISSION_IDENTITY_SCHEMA",
    "COHORT_ADMISSION_SCHEMA",
    "COHORT_ADMISSION_STATUS",
    "MAX_COHORT_ADMISSION_BYTES",
    "PRODUCTION_TRAINING_RUNTIME_SCHEMA",
    "VALIDATION_SHARD_CONTRACT_SCHEMA",
    "broadcast_rank_zero_cohort_admission",
    "build_cohort_admission",
    "build_validation_shard_contract",
    "canonical_json_bytes",
    "cohort_admission_identity",
    "seal_cohort_admission",
    "validate_cohort_admission",
    "validate_cohort_admission_identity",
    "validate_production_training_runtime_identity",
    "validate_training_runtime_identity",
    "validate_validation_shard_contract",
]
