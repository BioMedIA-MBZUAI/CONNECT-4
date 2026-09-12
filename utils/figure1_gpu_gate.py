"""Authenticated A100 selection for the Figure-1 v10 depth-slab core.

The manuscript does not specify a memory-slab schedule.  Production therefore
binds the largest core that completes the exact training step under a measured
four-GiB A100 safety margin; it never treats a convenient hand-written value as
paper evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


BENCHMARK_RECORD_SCHEMA = (
    "connect4-figure1-v10-a100-real-train-step-benchmark-v5"
)
SWEEP_SUMMARY_SCHEMA = "connect4-figure1-v10-a100-real-train-sweep-summary-v5"
DDP_SMOKE_SCHEMA = "connect4-figure1-v10-four-a100-real-train-smoke-v4"
PAPER_PROFILE_BLOCK_STATUS = (
    "PAPER_PROFILE_BLOCKED_MISSING_REVIEWED_128_GRID_BRAINLM_AUTHORITY"
)
MINIMUM_SAFETY_MARGIN_GIB = 4.0
DERIVED_DEPTH_HALO = 14
PROFILE_SHAPES = {
    "paper": (128, 128, 128),
    "recovery": (64, 80, 64),
}
EXPECTED_CORES = {
    "recovery": (1, 2, 4, 8, 16, 32, 64),
    "paper": (1, 2, 4, 8, 16, 32, 64, 128),
}
PROTOCOL_TO_BENCHMARK_PROFILE = {
    "paper-a4-adni-fmriprep-v1": "paper",
    "a4-native-recovery-v1": "recovery",
}
SOURCE_EXCLUSIONS = frozenset(
    {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scheduler_allocation_sha256(
    *,
    slurm_job_id: str,
    slurm_partition: str,
    slurm_qos: str,
    allocated_hostnames: list[str],
    slurm_num_nodes: int,
    allocated_gpu_count: int,
) -> str:
    """Hash only immutable scheduler-allocation facts from ``scontrol``.

    Raw ``scontrol show job`` output contains changing runtime/accounting fields,
    so hashing the whole line makes four ranks or sequential slab attempts claim
    different runtimes despite sharing one allocation.  Callers must first parse
    and verify the live record; this digest binds its stable CIAI allocation facts.
    """

    hosts = list(allocated_hostnames)
    if (
        not str(slurm_job_id).isdigit()
        or not isinstance(slurm_partition, str)
        or not slurm_partition
        or not isinstance(slurm_qos, str)
        or not slurm_qos
        or not hosts
        or any(not isinstance(host, str) or not host for host in hosts)
        or isinstance(slurm_num_nodes, bool)
        or not isinstance(slurm_num_nodes, int)
        or slurm_num_nodes < 1
        or len(hosts) != slurm_num_nodes
        or isinstance(allocated_gpu_count, bool)
        or not isinstance(allocated_gpu_count, int)
        or allocated_gpu_count < 1
    ):
        raise RuntimeError("scheduler allocation identity is incomplete")
    return canonical_sha256(
        {
            "schema": "connect4-ciai-scontrol-allocation-identity-v1",
            "slurm_job_id": str(slurm_job_id),
            "slurm_partition": slurm_partition,
            "slurm_qos": slurm_qos,
            "allocated_hostnames": hosts,
            "slurm_num_nodes": slurm_num_nodes,
            "allocated_gpu_count": allocated_gpu_count,
        }
    )


def source_tree_identity(root: Path) -> tuple[str, int]:
    """Hash one immutable, canonical source tree with a closed inventory rule."""

    root = root.expanduser()
    if not root.is_absolute() or root.is_symlink() or root.resolve(strict=True) != root:
        raise RuntimeError("source root must be an absolute canonical directory")
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise RuntimeError("source root permissions are unsafe")
    entries: list[dict[str, Any]] = []

    def walk(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            child_metadata = os.lstat(child)
            if stat.S_ISLNK(child_metadata.st_mode):
                raise RuntimeError(f"source inventory rejects symlink {child}")
            if stat.S_ISDIR(child_metadata.st_mode):
                if child.name not in SOURCE_EXCLUSIONS:
                    walk(child)
                continue
            if not stat.S_ISREG(child_metadata.st_mode):
                raise RuntimeError(f"source inventory rejects non-regular {child}")
            if child.suffix in {".pyc", ".pyo"} or child.name == ".DS_Store":
                continue
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "size_bytes": child_metadata.st_size,
                    "sha256": sha256_file(child),
                }
            )

    walk(root)
    if not entries:
        raise RuntimeError("source inventory is empty")
    return canonical_sha256(entries), len(entries)


def benchmark_model_config_identity(
    config: Mapping[str, Any], profile: str
) -> str:
    """Return the core-neutral, gate-neutral exact benchmark model identity."""

    if profile not in PROFILE_SHAPES:
        raise RuntimeError(f"unsupported Figure-1 profile {profile!r}")
    value = deepcopy(dict(config))
    try:
        data = value["data"]
        dit = value["models"]["dit"]
        unet = value["models"]["unet"]
        perceptual = value["training"]["loss"]["perceptual_extractor"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Figure-1 benchmark config structure is incomplete") from exc
    shape = list(PROFILE_SHAPES[profile])
    data["protocol_profile"] = "figure1-v10-a100-gpu-benchmark-only"
    data["architecture_shape"] = shape
    dit["input_size"] = shape
    unet["depth_slab_size"] = "SWEEP_CORE"
    for key, empty in (
        ("depth_slab_sweep_summary_path", None),
        ("depth_slab_sweep_summary_sha256", None),
        ("depth_slab_sweep_summary_sha256_env", ""),
        ("depth_slab_ddp_smoke_path", None),
        ("depth_slab_ddp_smoke_sha256", None),
        ("depth_slab_ddp_smoke_sha256_env", ""),
    ):
        unet[key] = empty
    for direct, environment in (
        ("source_root", "source_root_env"),
        ("checkpoint", "checkpoint_env"),
        ("config", "config_env"),
        ("atlas", "atlas_env"),
        ("coordinates", "coordinates_env"),
        ("authority", "authority_env"),
    ):
        resolved = perceptual.get(direct)
        if not resolved:
            environment_name = str(perceptual.get(environment) or "").strip()
            resolved = os.environ.get(environment_name) if environment_name else None
        perceptual[direct] = "<AUTHENTICATED_DEPLOYMENT_PATH>" if resolved else None
        perceptual[environment] = ""
    for direct, environment in (
        ("checkpoint_sha256", "checkpoint_sha256_env"),
        ("config_sha256", "config_sha256_env"),
        ("source_revision", "source_revision_env"),
        ("authority_sha256", "authority_sha256_env"),
    ):
        resolved = perceptual.get(direct)
        if not resolved:
            environment_name = str(perceptual.get(environment) or "").strip()
            resolved = os.environ.get(environment_name) if environment_name else None
        perceptual[direct] = str(resolved).strip().lower() if resolved else None
        perceptual[environment] = ""
    return canonical_sha256(value)


def seal_benchmark_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record.pop("canonical_record_sha256", None)
    record["canonical_record_sha256"] = canonical_sha256(record)
    return record


def seal_sweep_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(value)
    summary.pop("canonical_summary_sha256", None)
    summary["canonical_summary_sha256"] = canonical_sha256(summary)
    return summary


def atomic_write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one JSON file without following links or replacing."""

    path = path.expanduser()
    if not path.is_absolute():
        raise RuntimeError("authenticated GPU artifacts require an absolute path")
    parent = path.parent
    parent_metadata = os.lstat(parent)
    if (
        stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent.resolve(strict=True) != parent
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise RuntimeError("authenticated GPU artifact parent is unsafe")
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"refusing to replace existing GPU artifact {path}")

    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary = parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_descriptor = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{label} must be {qualifier}finite")
    return number


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _validate_recorded_brainlm_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = record.get("brainlm_identity")
    digest = _require_sha256(
        record.get("brainlm_identity_sha256"), "brainlm_identity_sha256"
    )
    try:
        from models.brainlm_context import validate_configured_brainlm_identity

        validated = validate_configured_brainlm_identity(identity)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Figure-1 BrainLM v2 identity is invalid: {exc}") from exc
    if validated.get("record_sha256") != digest:
        raise ValueError("Figure-1 BrainLM v2 identity SHA-256 differs")
    return validated


def _validate_real_training_batch_identity(
    value: Any,
    *,
    expected_local_batch_size: int,
    expected_admission_world_size: int,
    expected_brainlm_identity: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "protocol_profile",
        "scan_ids",
        "roles",
        "local_batch_size",
        "cohort_admission_identity",
        "dataset_state_record_sha256",
        "sample_artifacts_sha256",
        "target_artifact_identities_sha256",
        "target_tensor_identity",
        "brainlm_batch_context_identity",
        "brainlm_context_source",
        "admitted_unsealed_training_target_loaded",
        "synthetic_inputs_used",
        "sealed_target_voxel_data_opened",
        "record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Figure-1 real training-batch identity fields differ")
    identity = dict(value)
    claimed = _require_sha256(identity.pop("record_sha256"), "training batch record")
    if canonical_sha256(identity) != claimed:
        raise ValueError("Figure-1 real training-batch identity digest differs")
    scans = identity.get("scan_ids")
    roles = identity.get("roles")
    if (
        identity.get("schema") != "connect4-figure1-v10-real-training-batch-v2"
        or identity.get("protocol_profile") != "a4-native-recovery-v1"
        or identity.get("local_batch_size") != expected_local_batch_size
        or not isinstance(scans, list)
        or len(scans) != expected_local_batch_size
        or len(set(scans)) != len(scans)
        or any(not isinstance(scan, str) or not scan for scan in scans)
        or roles != ["train"] * expected_local_batch_size
        or identity.get("brainlm_context_source")
        != "exact-model-context-preflight; forward-equality-required-on-success"
        or identity.get("admitted_unsealed_training_target_loaded") is not True
        or identity.get("synthetic_inputs_used") is not False
        or identity.get("sealed_target_voxel_data_opened") is not False
    ):
        raise ValueError("Figure-1 gate did not use admitted unsealed training scans")
    for key in ("dataset_state_record_sha256", "sample_artifacts_sha256", "target_artifact_identities_sha256"):
        _require_sha256(identity.get(key), key)

    try:
        from data.cohort_admission import validate_cohort_admission_identity
        from models.brainlm_context import validate_brainlm_batch_context_identity

        admission = validate_cohort_admission_identity(
            identity.get("cohort_admission_identity")
        )
        context = validate_brainlm_batch_context_identity(
            identity.get("brainlm_batch_context_identity"),
            expected_brainlm_identity_sha256=expected_brainlm_identity[
                "record_sha256"
            ],
            expected_dataset_state_sha256=admission["dataset_state_sha256"],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Figure-1 admitted BrainLM batch-context identity is invalid: {exc}"
        ) from exc
    if (
        admission["world_size"] != expected_admission_world_size
        or context["scan_ids"] != scans
        or context["roles"] != roles
        or context["cohort_admission_identity_record_sha256"]
        != admission["record_sha256"]
        or context["dataset_state_record_sha256"]
        != identity["dataset_state_record_sha256"]
        or context["dataset_state_sha256"] != admission["dataset_state_sha256"]
        or context["run_artifact_identity_sha256"]
        != admission["artifact_identity_sha256"]
        or context["structural_artifact_identities_sha256"]
        != admission["structural_artifact_identities_sha256"]
        or context["authority_file_sha256"]
        != expected_brainlm_identity["mni_authority"]["file_sha256"]
        or context["authority_content_record_sha256"]
        != expected_brainlm_identity["mni_authority"]["content_record_sha256"]
        or context["authority_record_sha256"]
        != expected_brainlm_identity["mni_authority"][
            "authority_record_sha256"
        ]
        or not set(scans).issubset(
            set(expected_brainlm_identity["mni_authority"]["ordered_scan_ids"])
        )
    ):
        raise ValueError(
            "Figure-1 BrainLM context differs from its cohort/run authority"
        )

    target = identity.get("target_tensor_identity")
    expected_shape = [
        expected_local_batch_size,
        1,
        128,
        *PROFILE_SHAPES["recovery"],
    ]
    if not isinstance(target, Mapping) or set(target) != {
        "schema",
        "dtype",
        "shape_bctdhw",
        "sha256",
        "minimum",
        "maximum",
    }:
        raise ValueError("Figure-1 target-tensor identity fields differ")
    minimum = _finite_number(target.get("minimum"), "target minimum")
    maximum = _finite_number(target.get("maximum"), "target maximum")
    if (
        target.get("schema") != "connect4-recovery-training-target-tensor-v1"
        or target.get("dtype") != "torch.float32"
        or target.get("shape_bctdhw") != expected_shape
        or not 0.0 <= minimum <= maximum <= 1.0
    ):
        raise ValueError("Figure-1 recovery target tensor contract differs")
    _require_sha256(target.get("sha256"), "target tensor SHA-256")

    brainlm = context
    recorded_brainlm = identity["brainlm_batch_context_identity"]
    if (
        recorded_brainlm != brainlm
        or brainlm["record_sha256"]
        != recorded_brainlm.get("record_sha256")
    ):
        raise ValueError("Figure-1 BrainLM batch-context canonical record differs")
    return dict(value)


def validate_benchmark_record(
    value: Mapping[str, Any],
    *,
    expected_profile: str,
    expected_core: int,
) -> dict[str, Any]:
    """Validate one release-grade full DiT + decoder A100 measurement."""

    if not isinstance(value, Mapping):
        raise ValueError("Figure-1 benchmark record must be a mapping")
    record = dict(value)
    if expected_profile not in PROFILE_SHAPES:
        raise ValueError(f"unsupported benchmark profile {expected_profile!r}")
    if record.get("schema") != BENCHMARK_RECORD_SCHEMA:
        raise ValueError("Figure-1 benchmark record schema differs")
    claimed_record_digest = _require_sha256(
        record.get("canonical_record_sha256"), "canonical_record_sha256"
    )
    unsigned_record = dict(record)
    unsigned_record.pop("canonical_record_sha256")
    if canonical_sha256(unsigned_record) != claimed_record_digest:
        raise ValueError("Figure-1 canonical record SHA-256 differs")
    if record.get("gate_scope") != (
        "exact-full-connect4-per-rank-real-training-step-memory"
    ):
        raise ValueError("Figure-1 benchmark omitted the full per-rank train step")
    if record.get("profile") != expected_profile:
        raise ValueError("Figure-1 benchmark profile differs")
    if expected_profile != "recovery":
        raise ValueError(
            f"{PAPER_PROFILE_BLOCK_STATUS}: a paper-grid composite record is forbidden"
        )
    if record.get("depth_slab_size") != expected_core:
        raise ValueError("Figure-1 benchmark slab core differs")
    if record.get("derived_depth_halo") != DERIVED_DEPTH_HALO:
        raise ValueError("Figure-1 benchmark did not use the derived 14-voxel halo")
    expected_shape = [1, 1, 128, *PROFILE_SHAPES[expected_profile]]
    if record.get("shape_bctdhw") != expected_shape:
        raise ValueError(
            f"Figure-1 benchmark shape must be the full {expected_shape} profile"
        )
    for key in (
        "source_tree_sha256",
        "architecture_contract_sha256",
        "base_config_file_sha256",
        "benchmark_model_config_sha256",
        "brainlm_identity_sha256",
        "scheduler_alphas_cumprod_sha256",
    ):
        _require_sha256(record.get(key), key)
    if not isinstance(record.get("source_file_count"), int) or record[
        "source_file_count"
    ] < 1:
        raise ValueError("Figure-1 source inventory is empty")
    brainlm_identity = _validate_recorded_brainlm_identity(record)
    _validate_real_training_batch_identity(
        record.get("training_batch_identity"),
        expected_local_batch_size=1,
        expected_admission_world_size=1,
        expected_brainlm_identity=brainlm_identity,
    )
    if record.get("composite_loss_configured") is not True:
        raise ValueError("Figure-1 full gate omitted the composite biological loss")
    if record.get("all_roi_masks_configured") is not True:
        raise ValueError("Figure-1 full gate omitted the 32 ROI masks")
    expected_weights = {
        "voxel": 1.0,
        "ssim": 0.5,
        "fc": 0.3,
        "temporal": 0.2,
        "perceptual": 0.1,
        "volume": 0.5,
        "region_hist": 0.5,
    }
    if record.get("loss_weights") != expected_weights:
        raise ValueError("Figure-1 full gate loss weights differ")

    amp = record.get("amp")
    if amp != {
        "enabled": True,
        "dtype": "bfloat16",
        "gradient_scaler_enabled": False,
        "parameter_dtype": "float32",
    }:
        raise ValueError("Figure-1 full gate must use the production A100 bf16 path")
    batch = record.get("batch_contract")
    if batch != {
        "production_global_batch_size": 4,
        "executed_world_size": 1,
        "executed_local_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "ddp_wrapped": False,
    }:
        raise ValueError("Figure-1 full gate batch contract differs")
    if (
        record.get("four_gpu_ddp_smoke_required") is not True
        or record.get("four_gpu_ddp_smoke_attested") is not False
    ):
        raise ValueError("one-GPU memory record must retain the four-GPU DDP gate")
    optimizer = record.get("optimizer")
    if optimizer != {
        "name": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "gradient_clip_norm": 1.0,
    }:
        raise ValueError("Figure-1 full gate optimizer contract differs")
    scheduler = record.get("scheduler")
    if scheduler != {
        "name": "DDIM",
        "num_diffusion_steps": 1000,
        "beta_schedule": "linear",
        "num_inference_steps": 50,
        "eta": 0.0,
        "training_objective": "epsilon",
    }:
        raise ValueError("Figure-1 full gate scheduler contract differs")

    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Figure-1 benchmark execution attestation is missing")
    if execution.get("slurm_partition") != "cscc-gpu-p":
        raise ValueError("Figure-1 benchmark ran on the wrong partition")
    if execution.get("slurm_qos") != "cscc-gpu-qos":
        raise ValueError("Figure-1 benchmark ran under the wrong QOS")
    if (
        execution.get("slurm_num_nodes") != 1
        or execution.get("allocated_gpu_count") != 1
    ):
        raise ValueError("Figure-1 benchmark scheduler shape differs")
    job_id = str(execution.get("slurm_job_id") or "")
    if not job_id.isdigit():
        raise ValueError("Figure-1 benchmark Slurm job ID is invalid")
    if not str(execution.get("cuda_visible_devices") or "").strip():
        raise ValueError("Figure-1 benchmark CUDA visibility is unbound")
    allocated_hosts = execution.get("allocated_hostnames")
    hostname = str(execution.get("hostname") or "")
    if (
        not hostname
        or "login" in hostname.lower()
        or not isinstance(allocated_hosts, list)
        or len(allocated_hosts) != 1
        or hostname not in allocated_hosts
    ):
        raise ValueError("Figure-1 benchmark did not attest a compute node")
    if not str(execution.get("gpu_uuid") or "").startswith("GPU-"):
        raise ValueError("Figure-1 benchmark GPU UUID is missing")
    _require_sha256(execution.get("python_executable_sha256"), "python executable")
    _require_sha256(execution.get("scontrol_record_sha256"), "scontrol record")
    if execution["scontrol_record_sha256"] != scheduler_allocation_sha256(
        slurm_job_id=job_id,
        slurm_partition=execution["slurm_partition"],
        slurm_qos=execution["slurm_qos"],
        allocated_hostnames=allocated_hosts,
        slurm_num_nodes=execution["slurm_num_nodes"],
        allocated_gpu_count=execution["allocated_gpu_count"],
    ):
        raise ValueError("Figure-1 scheduler-allocation digest differs")
    python_executable = execution.get("python_executable")
    if (
        not isinstance(python_executable, str)
        or not python_executable.startswith("/")
        or "mica" in python_executable.lower()
    ):
        raise ValueError("Figure-1 benchmark runtime is missing or revoked")
    visible_devices = [
        token.strip()
        for token in str(execution.get("cuda_visible_devices") or "").split(",")
        if token.strip()
    ]
    if execution.get("cuda_device_count") != 1 or len(visible_devices) != 1:
        raise ValueError("Figure-1 benchmark must expose exactly one GPU")
    if execution.get("isolated_python") is not True:
        raise ValueError("Figure-1 benchmark Python was not isolated")
    if "A100" not in str(record.get("gpu", "")):
        raise ValueError("Figure-1 release gate requires an A100 measurement")
    if (
        "A100" not in str(execution.get("nvidia_smi_name") or "")
        or not isinstance(execution.get("nvidia_smi_memory_mib"), int)
        or not 39_000 <= execution["nvidia_smi_memory_mib"] <= 42_000
        or not str(execution.get("nvidia_driver_version") or "").strip()
    ):
        raise ValueError("Figure-1 nvidia-smi A100-40GB identity differs")
    if (
        not str(record.get("torch_version") or "").strip()
        or not str(record.get("cuda_version") or "").strip()
        or record.get("cudnn_version") is None
    ):
        raise ValueError("Figure-1 framework runtime identity is incomplete")
    passed = record.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("Figure-1 benchmark passed flag must be boolean")

    total = _finite_number(
        record.get("gpu_total_memory_gib"), "gpu_total_memory_gib", positive=True
    )
    if not 39.0 <= total <= 41.0:
        raise ValueError("Figure-1 release gate requires the A100-40GB class")
    margin = _finite_number(
        record.get("safety_margin_gib"), "safety_margin_gib", positive=True
    )
    if margin < MINIMUM_SAFETY_MARGIN_GIB:
        raise ValueError("Figure-1 benchmark safety margin is below four GiB")
    safe_limit = _finite_number(
        record.get("safe_reserved_limit_gib"),
        "safe_reserved_limit_gib",
        positive=True,
    )
    if not math.isclose(safe_limit, total - margin, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Figure-1 benchmark safety limit is inconsistent")
    peak_allocated = _finite_number(
        record.get("peak_allocated_gib"), "peak_allocated_gib"
    )
    peak_reserved = _finite_number(
        record.get("peak_reserved_gib"), "peak_reserved_gib"
    )
    if peak_allocated < 0 or peak_reserved < 0 or peak_allocated > peak_reserved + 1e-6:
        raise ValueError("Figure-1 benchmark CUDA peak-memory values are inconsistent")

    if passed:
        if record.get("adamw_step_included") is not True:
            raise ValueError("Figure-1 passing benchmark must complete an AdamW step")
        if record.get("gradient_clip_included") is not True:
            raise ValueError("Figure-1 passing benchmark must clip gradients")
        if record.get("composite_loss_completed") is not True:
            raise ValueError("Figure-1 passing benchmark did not complete all losses")
        if record.get("required_gradient_groups_finite") is not True:
            raise ValueError("Figure-1 passing benchmark missed required gradients")
        if record.get("required_gradient_groups_nonzero") is not True:
            raise ValueError("Figure-1 passing benchmark has a zero gradient group")
        if record.get("full_target_blind_inference_completed") is not True:
            raise ValueError("Figure-1 passing benchmark omitted full DDIM inference")
        if record.get("inference_autocast_dtype") != "bfloat16":
            raise ValueError("Figure-1 inference gate must use A100 bfloat16")
        if record.get("inference_num_steps") != 50:
            raise ValueError("Figure-1 inference gate must execute all 50 DDIM steps")
        if peak_reserved > safe_limit + 1e-6:
            raise ValueError("Figure-1 passing record violates its safety margin")
        forward = _finite_number(
            record.get("forward_seconds"), "forward_seconds", positive=True
        )
        backward = _finite_number(
            record.get("backward_seconds"), "backward_seconds", positive=True
        )
        step = _finite_number(
            record.get("step_seconds"), "step_seconds", positive=True
        )
        _finite_number(
            record.get("inference_seconds"), "inference_seconds", positive=True
        )
        inference_peak = _finite_number(
            record.get("inference_peak_reserved_gib"),
            "inference_peak_reserved_gib",
        )
        if inference_peak < 0 or inference_peak > safe_limit + 1e-6:
            raise ValueError("Figure-1 inference violates its safety margin")
        if step + 1e-9 < forward or step + 1e-9 < backward:
            raise ValueError("Figure-1 benchmark step timing is inconsistent")
    else:
        failure_kind = record.get("failure_kind")
        if failure_kind not in {"CUDA_OUT_OF_MEMORY", "SAFETY_MARGIN"}:
            raise ValueError("Figure-1 failed record has an unsupported failure kind")
        if not str(record.get("failure", "")).strip():
            raise ValueError("Figure-1 failed record must explain its failure")
    return record


def validate_sweep_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete prefix sweeps and their largest-safe-core selection."""

    if not isinstance(value, Mapping):
        raise ValueError("Figure-1 sweep summary must be a mapping")
    summary = dict(value)
    if summary.get("schema") != SWEEP_SUMMARY_SCHEMA:
        raise ValueError("Figure-1 sweep summary schema differs")
    claimed_summary_digest = _require_sha256(
        summary.get("canonical_summary_sha256"), "canonical_summary_sha256"
    )
    unsigned_summary = dict(summary)
    unsigned_summary.pop("canonical_summary_sha256")
    if canonical_sha256(unsigned_summary) != claimed_summary_digest:
        raise ValueError("Figure-1 canonical summary SHA-256 differs")
    if _finite_number(
        summary.get("minimum_safety_margin_gib"),
        "minimum_safety_margin_gib",
        positive=True,
    ) != MINIMUM_SAFETY_MARGIN_GIB:
        raise ValueError("Figure-1 sweep summary must enforce a four-GiB margin")
    profiles = summary.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(EXPECTED_CORES):
        raise ValueError("Figure-1 sweep summary profiles differ")

    common_identity = None
    all_bindings = []
    for profile, expected_cores in EXPECTED_CORES.items():
        profile_value = profiles[profile]
        if not isinstance(profile_value, Mapping):
            raise ValueError(f"Figure-1 {profile} sweep must be a mapping")
        if profile == "paper":
            expected_block = {
                "status": PAPER_PROFILE_BLOCK_STATUS,
                "attempted_core_sizes": [],
                "largest_safe_core_size": None,
                "selected_peak_allocated_gib": None,
                "selected_peak_reserved_gib": None,
                "selected_forward_seconds": None,
                "selected_backward_seconds": None,
                "selected_step_seconds": None,
                "memory_gate_passed": False,
                "four_gpu_ddp_smoke_attested": False,
                "release_stop": True,
                "record_order_sha256": canonical_sha256([]),
                "record_bindings": [],
                "records": [],
                "blocking_contract": {
                    "required_shape_bctdhw": [1, 1, 128, 128, 128, 128],
                    "perceptual_loss_required": True,
                    "available_brainlm_authority_grid": [64, 80, 64],
                    "reviewed_128_grid_brainlm_authority_available": False,
                    "synthetic_or_omitted_brainlm_forbidden": True,
                },
            }
            if dict(profile_value) != expected_block:
                raise ValueError(
                    f"Figure-1 paper profile must remain {PAPER_PROFILE_BLOCK_STATUS}"
                )
            continue
        records_value = profile_value.get("records")
        if not isinstance(records_value, list):
            raise ValueError(f"Figure-1 {profile} records must be a list")
        records = []
        for raw_record in records_value:
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"Figure-1 {profile} record must be a mapping")
            core = raw_record.get("depth_slab_size")
            if isinstance(core, bool) or not isinstance(core, int):
                raise ValueError(f"Figure-1 {profile} slab core must be an integer")
            records.append(
                validate_benchmark_record(
                    raw_record,
                    expected_profile=profile,
                    expected_core=core,
                )
            )
        identity_fields = (
            "source_tree_sha256",
            "source_file_count",
            "architecture_contract_sha256",
            "brainlm_identity_sha256",
            "scheduler_alphas_cumprod_sha256",
        )
        for record in records:
            execution = record["execution"]
            identity = {
                "record_fields": {
                    field: record[field] for field in identity_fields
                },
                "brainlm_identity": record["brainlm_identity"],
                "training_batch_identity": record["training_batch_identity"],
                "amp": record["amp"],
                "batch_contract": record["batch_contract"],
                "loss_weights": record["loss_weights"],
                "optimizer": record["optimizer"],
                "scheduler": record["scheduler"],
                "torch_version": record["torch_version"],
                "cuda_version": record["cuda_version"],
                "cudnn_version": record["cudnn_version"],
                "execution": {
                    key: execution[key]
                    for key in (
                        "slurm_job_id",
                        "slurm_partition",
                        "slurm_qos",
                        "slurm_num_nodes",
                        "allocated_gpu_count",
                        "allocated_hostnames",
                        "hostname",
                        "cuda_visible_devices",
                        "cuda_device_count",
                        "gpu_uuid",
                        "nvidia_smi_name",
                        "nvidia_smi_memory_mib",
                        "nvidia_driver_version",
                        "python_executable",
                        "python_executable_sha256",
                        "isolated_python",
                        "scontrol_record_sha256",
                    )
                },
            }
            if common_identity is None:
                common_identity = identity
            elif identity != common_identity:
                raise ValueError("Figure-1 sweep source/config/runtime identity drifted")
        model_identities = {
            record["benchmark_model_config_sha256"] for record in records
        }
        if len(model_identities) != 1:
            raise ValueError(f"Figure-1 {profile} benchmark model config drifted")
        attempted = [record["depth_slab_size"] for record in records]
        if attempted != list(expected_cores[: len(attempted)]):
            raise ValueError(
                f"Figure-1 {profile} sweep must be an ordered contiguous prefix"
            )
        if attempted[:3] != [1, 2, 4]:
            raise ValueError(
                f"Figure-1 {profile} sweep did not complete mandatory cores 1,2,4"
            )
        if len(attempted) < len(expected_cores):
            if records[-1].get("failure_kind") != "CUDA_OUT_OF_MEMORY":
                raise ValueError(
                    f"Figure-1 {profile} sweep stopped before full depth without OOM"
                )
        safe = [record for record in records if record["passed"]]
        selected = max(safe, key=lambda record: record["depth_slab_size"]) if safe else None
        selected_core = selected["depth_slab_size"] if selected is not None else None
        if profile_value.get("attempted_core_sizes") != attempted:
            raise ValueError(f"Figure-1 {profile} attempted-core summary differs")
        if profile_value.get("largest_safe_core_size") != selected_core:
            raise ValueError(f"Figure-1 {profile} did not select the largest safe core")
        bindings = profile_value.get("record_bindings")
        if not isinstance(bindings, list) or len(bindings) != len(records):
            raise ValueError(f"Figure-1 {profile} record-file bindings differ")
        expected_bindings = []
        for order, (record, binding) in enumerate(zip(records, bindings)):
            if not isinstance(binding, Mapping):
                raise ValueError(f"Figure-1 {profile} record binding is invalid")
            _require_sha256(binding.get("file_sha256"), "record file SHA-256")
            expected = {
                "order": order,
                "profile": profile,
                "depth_slab_size": record["depth_slab_size"],
                "filename": (
                    f"figure1_v10_{profile}_slab{record['depth_slab_size']}.json"
                ),
                "file_sha256": binding["file_sha256"],
                "canonical_record_sha256": record["canonical_record_sha256"],
                "slurm_job_id": record["execution"]["slurm_job_id"],
            }
            if dict(binding) != expected:
                raise ValueError(f"Figure-1 {profile} record binding differs")
            expected_bindings.append(expected)
        if profile_value.get("record_order_sha256") != canonical_sha256(
            expected_bindings
        ):
            raise ValueError(f"Figure-1 {profile} record order digest differs")
        all_bindings.extend(expected_bindings)
        selected_fields = {
            "selected_peak_allocated_gib": "peak_allocated_gib",
            "selected_peak_reserved_gib": "peak_reserved_gib",
            "selected_forward_seconds": "forward_seconds",
            "selected_backward_seconds": "backward_seconds",
            "selected_step_seconds": "step_seconds",
        }
        for summary_key, record_key in selected_fields.items():
            expected = selected.get(record_key) if selected is not None else None
            if profile_value.get(summary_key) != expected:
                raise ValueError(f"Figure-1 {profile} {summary_key} differs")
        if profile_value.get("memory_gate_passed") is not (selected is not None):
            raise ValueError(f"Figure-1 {profile} memory-gate flag differs")
        if profile_value.get("four_gpu_ddp_smoke_attested") is not False:
            raise ValueError(f"Figure-1 {profile} memory sweep cannot attest DDP")
        if profile_value.get("release_stop") is not True:
            raise ValueError(f"Figure-1 {profile} release-stop flag differs")
    if summary.get("record_file_order_sha256") != canonical_sha256(all_bindings):
        raise ValueError("Figure-1 sweep record-file order digest differs")
    if (
        summary.get("four_gpu_ddp_smoke_required") is not True
        or summary.get("four_gpu_ddp_smoke_attested") is not False
        or summary.get("passed") is not False
    ):
        raise ValueError("one-GPU sweep must retain the separate four-GPU DDP gate")
    if summary.get("paper_profile_release_stop") is not True:
        raise ValueError("Figure-1 paper release-stop flag differs")
    return summary


def validate_ddp_smoke_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the separate exact four-rank production-batch smoke record."""

    if not isinstance(value, Mapping):
        raise ValueError("Figure-1 DDP smoke attestation must be a mapping")
    record = dict(value)
    if record.get("schema") != DDP_SMOKE_SCHEMA:
        raise ValueError("Figure-1 DDP smoke schema differs")
    claimed = _require_sha256(
        record.get("canonical_record_sha256"), "DDP canonical record SHA-256"
    )
    unsigned = dict(record)
    unsigned.pop("canonical_record_sha256")
    if canonical_sha256(unsigned) != claimed:
        raise ValueError("Figure-1 DDP canonical record SHA-256 differs")
    if record.get("passed") is not True:
        raise ValueError("Figure-1 four-GPU DDP smoke did not pass")
    if record.get("gate_scope") != (
        "exact-full-connect4-four-rank-ddp-real-training-step"
    ):
        raise ValueError("Figure-1 DDP smoke omitted the exact full training step")
    if record.get("profile") != "recovery":
        raise ValueError(
            f"{PAPER_PROFILE_BLOCK_STATUS}: only recovery has a reviewed BrainLM gate"
        )
    if record.get("shape_bctdhw") != [
        1, 1, 128, *PROFILE_SHAPES[record["profile"]]
    ]:
        raise ValueError("Figure-1 DDP smoke used the wrong full profile shape")
    if record.get("derived_depth_halo") != DERIVED_DEPTH_HALO:
        raise ValueError("Figure-1 DDP smoke did not use the derived depth halo")
    if not isinstance(record.get("source_file_count"), int) or record[
        "source_file_count"
    ] < 1:
        raise ValueError("Figure-1 DDP source inventory is empty")
    core = record.get("depth_slab_size")
    if isinstance(core, bool) or not isinstance(core, int) or core < 1:
        raise ValueError("Figure-1 DDP slab core is invalid")
    for key in (
        "memory_sweep_summary_sha256",
        "selected_memory_record_sha256",
        "source_tree_sha256",
        "architecture_contract_sha256",
        "base_config_file_sha256",
        "benchmark_model_config_sha256",
        "brainlm_identity_sha256",
        "scheduler_alphas_cumprod_sha256",
        "initial_model_state_sha256",
        "global_training_batch_identity_sha256",
    ):
        _require_sha256(record.get(key), key)
    brainlm_identity = _validate_recorded_brainlm_identity(record)
    if record.get("batch_contract") != {
        "global_batch_size": 4,
        "world_size": 4,
        "local_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "ddp_wrapped": True,
    }:
        raise ValueError("Figure-1 DDP smoke batch contract differs")
    if record.get("amp") != {
        "enabled": True,
        "dtype": "bfloat16",
        "gradient_scaler_enabled": False,
        "parameter_dtype": "float32",
    }:
        raise ValueError("Figure-1 DDP smoke must use production A100 bf16")
    if record.get("loss_weights") != {
        "voxel": 1.0,
        "ssim": 0.5,
        "fc": 0.3,
        "temporal": 0.2,
        "perceptual": 0.1,
        "volume": 0.5,
        "region_hist": 0.5,
    }:
        raise ValueError("Figure-1 DDP smoke loss weights differ")
    if record.get("optimizer") != {
        "name": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "gradient_clip_norm": 1.0,
    }:
        raise ValueError("Figure-1 DDP smoke optimizer differs")
    if record.get("scheduler") != {
        "name": "DDIM",
        "num_diffusion_steps": 1000,
        "beta_schedule": "linear",
        "num_inference_steps": 50,
        "eta": 0.0,
        "training_objective": "epsilon",
    }:
        raise ValueError("Figure-1 DDP smoke scheduler differs")
    if (
        record.get("ddp_gradient_reduction_attested") is not True
        or record.get("all_ranks_post_step_state_equal") is not True
    ):
        raise ValueError("Figure-1 DDP synchronization was not attested")
    synchronized_state = _require_sha256(
        record.get("synchronized_post_step_model_state_sha256"),
        "DDP synchronized post-step model state",
    )
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Figure-1 DDP execution attestation is missing")
    if (
        execution.get("slurm_partition") != "cscc-gpu-p"
        or execution.get("slurm_qos") != "cscc-gpu-qos"
        or execution.get("slurm_num_nodes") != 1
        or execution.get("allocated_gpu_count") != 4
        or execution.get("cuda_device_count") != 4
        or not str(execution.get("slurm_job_id") or "").isdigit()
    ):
        raise ValueError("Figure-1 DDP scheduler allocation differs")
    _require_sha256(execution.get("scontrol_record_sha256"), "DDP scontrol record")
    _require_sha256(execution.get("python_executable_sha256"), "DDP Python executable")
    python_executable = execution.get("python_executable")
    if (
        not isinstance(python_executable, str)
        or not python_executable.startswith("/")
        or "mica" in python_executable.lower()
        or execution.get("isolated_python") is not True
    ):
        raise ValueError("Figure-1 DDP runtime is missing, revoked, or non-isolated")
    hosts = execution.get("allocated_hostnames")
    hostname = str(execution.get("hostname") or "")
    if (
        not isinstance(hosts, list)
        or len(hosts) != 1
        or hostname not in hosts
        or any(not str(host) or "login" in str(host).lower() for host in hosts)
    ):
        raise ValueError("Figure-1 DDP smoke did not attest compute hosts")
    if execution["scontrol_record_sha256"] != scheduler_allocation_sha256(
        slurm_job_id=str(execution["slurm_job_id"]),
        slurm_partition=execution["slurm_partition"],
        slurm_qos=execution["slurm_qos"],
        allocated_hostnames=hosts,
        slurm_num_nodes=execution["slurm_num_nodes"],
        allocated_gpu_count=execution["allocated_gpu_count"],
    ):
        raise ValueError("Figure-1 DDP scheduler-allocation digest differs")
    visible_devices = [
        token.strip()
        for token in str(execution.get("cuda_visible_devices") or "").split(",")
        if token.strip()
    ]
    if len(visible_devices) != 4 or len(set(visible_devices)) != 4:
        raise ValueError("Figure-1 DDP CUDA visibility differs")
    if (
        not str(execution.get("torch_version") or "").strip()
        or not str(execution.get("cuda_version") or "").strip()
        or execution.get("cudnn_version") is None
    ):
        raise ValueError("Figure-1 DDP framework runtime identity is incomplete")
    ranks = record.get("rank_records")
    if not isinstance(ranks, list) or len(ranks) != 4:
        raise ValueError("Figure-1 DDP smoke needs four rank records")
    uuids = set()
    rank_visible_devices = set()
    drivers = set()
    training_batch_identities = []
    for expected_rank, rank in enumerate(ranks):
        if not isinstance(rank, Mapping) or rank.get("rank") != expected_rank:
            raise ValueError("Figure-1 DDP rank order differs")
        if rank.get("world_size") != 4 or rank.get("local_batch_size") != 1:
            raise ValueError("Figure-1 DDP rank batch contract differs")
        uuid = str(rank.get("gpu_uuid") or "")
        if not uuid.startswith("GPU-") or uuid in uuids:
            raise ValueError("Figure-1 DDP GPU UUIDs are invalid or repeated")
        uuids.add(uuid)
        visible_device = str(rank.get("cuda_visible_device") or "").strip()
        if (
            visible_device != visible_devices[expected_rank]
            or visible_device in rank_visible_devices
            or rank.get("torch_cuda_device_index") != expected_rank
        ):
            raise ValueError("Figure-1 DDP rank-to-GPU mapping differs")
        rank_visible_devices.add(visible_device)
        driver = str(rank.get("nvidia_driver_version") or "").strip()
        if not driver:
            raise ValueError("Figure-1 DDP rank driver identity is missing")
        drivers.add(driver)
        training_batch_identities.append(
            _validate_real_training_batch_identity(
                rank.get("training_batch_identity"),
                expected_local_batch_size=1,
                expected_admission_world_size=4,
                expected_brainlm_identity=brainlm_identity,
            )
        )
        if "A100" not in str(rank.get("gpu_name") or ""):
            raise ValueError("Figure-1 DDP rank did not use A100")
        total = _finite_number(
            rank.get("gpu_total_memory_gib"), "DDP GPU memory", positive=True
        )
        if not 39.0 <= total <= 41.0:
            raise ValueError("Figure-1 DDP rank did not use A100-40GB")
        peak_allocated = _finite_number(
            rank.get("peak_allocated_gib"), "DDP peak allocated"
        )
        peak_reserved = _finite_number(
            rank.get("peak_reserved_gib"), "DDP peak reserved"
        )
        if (
            peak_allocated < 0
            or peak_allocated > peak_reserved + 1e-6
            or peak_reserved > total - MINIMUM_SAFETY_MARGIN_GIB + 1e-6
        ):
            raise ValueError("Figure-1 DDP rank violates the four-GiB margin")
        forward_seconds = _finite_number(
            rank.get("forward_seconds"), "DDP forward_seconds", positive=True
        )
        backward_seconds = _finite_number(
            rank.get("backward_seconds"), "DDP backward_seconds", positive=True
        )
        step_seconds = _finite_number(
            rank.get("step_seconds"), "DDP step_seconds", positive=True
        )
        if step_seconds + 1e-9 < max(forward_seconds, backward_seconds):
            raise ValueError("Figure-1 DDP step timing is inconsistent")
        if (
            rank.get("composite_loss_completed") is not True
            or rank.get("required_gradient_groups_finite") is not True
            or rank.get("required_gradient_groups_nonzero") is not True
            or rank.get("gradient_clip_included") is not True
            or rank.get("adamw_step_included") is not True
        ):
            raise ValueError("Figure-1 DDP rank did not complete the exact step")
        if _require_sha256(
            rank.get("post_step_model_state_sha256"),
            "DDP rank post-step model state",
        ) != synchronized_state:
            raise ValueError("Figure-1 DDP post-step model states diverged")
    if len(drivers) != 1:
        raise ValueError("Figure-1 DDP ranks used different NVIDIA drivers")
    if len(
        {
            identity["scan_ids"][0]
            for identity in training_batch_identities
        }
    ) != 4:
        raise ValueError("Figure-1 DDP ranks did not use four distinct training scans")
    if record["global_training_batch_identity_sha256"] != canonical_sha256(
        training_batch_identities
    ):
        raise ValueError("Figure-1 DDP global training-batch identity differs")
    return record


def authenticate_configured_depth_slab_selection(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Authenticate and return the configured profile's selected slab core."""

    data = config.get("data")
    models = config.get("models")
    if not isinstance(data, Mapping) or not isinstance(models, Mapping):
        raise RuntimeError("production slab selection requires data/models config")
    unet = models.get("unet")
    if not isinstance(unet, Mapping):
        raise RuntimeError("production slab selection requires models.unet config")
    protocol = str(data.get("protocol_profile", "")).strip()
    profile = PROTOCOL_TO_BENCHMARK_PROFILE.get(protocol)
    if profile is None:
        raise RuntimeError("production slab selection has an unknown protocol profile")
    core = unet.get("depth_slab_size")
    if isinstance(core, bool) or not isinstance(core, int) or core < 1:
        raise RuntimeError(
            "models.unet.depth_slab_size is unresolved; bind the authenticated "
            "Figure-1 v10 A100 sweep's largest safe core before model construction"
        )
    raw_path = str(unet.get("depth_slab_sweep_summary_path") or "").strip()
    if not raw_path:
        raise RuntimeError("production slab selection is missing its sweep summary")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = path.resolve()
    direct_digest = unet.get("depth_slab_sweep_summary_sha256")
    environment_name = str(
        unet.get("depth_slab_sweep_summary_sha256_env") or ""
    ).strip()
    expected_digest = (
        str(direct_digest).strip().lower()
        if direct_digest is not None
        else str(os.environ.get(environment_name, "")).strip().lower()
    )
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise RuntimeError("production slab selection has no valid SHA-256 pin")
    if not path.is_file() or sha256_file(path) != expected_digest:
        raise RuntimeError("Figure-1 v10 A100 sweep summary is missing or digest-mismatched")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Figure-1 v10 A100 sweep summary is unreadable") from exc
    try:
        validated = validate_sweep_summary(summary)
    except ValueError as exc:
        raise RuntimeError(f"Figure-1 v10 A100 sweep summary is invalid: {exc}") from exc
    profile_value = validated["profiles"][profile]
    if not profile_value["memory_gate_passed"]:
        raise RuntimeError(f"Figure-1 v10 {profile} per-rank memory gate did not pass")
    selected = profile_value["largest_safe_core_size"]
    if core != selected:
        raise RuntimeError(
            f"configured slab core {core} is not the authenticated largest-safe "
            f"{profile} core {selected}"
        )
    raw_ddp_path = str(unet.get("depth_slab_ddp_smoke_path") or "").strip()
    if not raw_ddp_path:
        raise RuntimeError(
            "production slab selection is missing the separate four-GPU DDP smoke"
        )
    ddp_path = Path(raw_ddp_path).expanduser()
    if not ddp_path.is_absolute() and base_dir is not None:
        ddp_path = base_dir / ddp_path
    ddp_path = ddp_path.resolve()
    ddp_direct_digest = unet.get("depth_slab_ddp_smoke_sha256")
    ddp_environment_name = str(
        unet.get("depth_slab_ddp_smoke_sha256_env") or ""
    ).strip()
    ddp_expected_digest = (
        str(ddp_direct_digest).strip().lower()
        if ddp_direct_digest is not None
        else str(os.environ.get(ddp_environment_name, "")).strip().lower()
    )
    if len(ddp_expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in ddp_expected_digest
    ):
        raise RuntimeError("four-GPU DDP smoke has no valid SHA-256 pin")
    if not ddp_path.is_file() or sha256_file(ddp_path) != ddp_expected_digest:
        raise RuntimeError("four-GPU DDP smoke is missing or digest-mismatched")
    try:
        ddp_value = json.loads(ddp_path.read_text(encoding="utf-8"))
        ddp = validate_ddp_smoke_attestation(ddp_value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"four-GPU DDP smoke is invalid: {exc}") from exc
    if ddp["profile"] != profile or ddp["depth_slab_size"] != core:
        raise RuntimeError("four-GPU DDP smoke profile/core differs from selection")
    selected_record = next(
        record
        for record in profile_value["records"]
        if record["depth_slab_size"] == core
    )
    if (
        ddp["memory_sweep_summary_sha256"] != expected_digest
        or ddp["selected_memory_record_sha256"]
        != selected_record["canonical_record_sha256"]
    ):
        raise RuntimeError("four-GPU DDP smoke is not bound to this memory sweep")
    for field in (
        "source_tree_sha256",
        "architecture_contract_sha256",
        "base_config_file_sha256",
        "benchmark_model_config_sha256",
        "brainlm_identity_sha256",
        "scheduler_alphas_cumprod_sha256",
    ):
        if ddp[field] != selected_record[field]:
            raise RuntimeError(f"four-GPU DDP smoke {field} differs from memory gate")
    if ddp["source_file_count"] != selected_record["source_file_count"]:
        raise RuntimeError("four-GPU DDP smoke source inventory count differs")
    if ddp["brainlm_identity"] != selected_record["brainlm_identity"]:
        raise RuntimeError("four-GPU DDP BrainLM v2 identity differs from memory gate")
    selected_execution = selected_record["execution"]
    ddp_execution = ddp["execution"]
    for field in (
        "python_executable",
        "python_executable_sha256",
        "torch_version",
        "cuda_version",
        "cudnn_version",
    ):
        memory_value = (
            selected_record[field]
            if field in {"torch_version", "cuda_version", "cudnn_version"}
            else selected_execution[field]
        )
        if ddp_execution[field] != memory_value:
            raise RuntimeError(f"four-GPU DDP runtime {field} differs from memory gate")
    memory_driver = selected_execution["nvidia_driver_version"]
    if any(
        rank["nvidia_driver_version"] != memory_driver
        for rank in ddp["rank_records"]
    ):
        raise RuntimeError("four-GPU DDP NVIDIA driver differs from memory gate")
    source_root = Path(__file__).resolve().parents[1]
    try:
        current_source_sha256, current_source_count = source_tree_identity(source_root)
        current_model_config_sha256 = benchmark_model_config_identity(config, profile)
        from architecture_contract import SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
        from models.brainlm_context import configured_brainlm_identity

        current_brainlm_identity = configured_brainlm_identity(
            config["training"]["loss"]["perceptual_extractor"]
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"cannot attest current Figure-1 source/config: {exc}") from exc
    if (
        selected_record["source_tree_sha256"] != current_source_sha256
        or selected_record["source_file_count"] != current_source_count
    ):
        raise RuntimeError("Figure-1 GPU evidence predates or differs from current source")
    if (
        selected_record["architecture_contract_sha256"]
        != SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
    ):
        raise RuntimeError("Figure-1 GPU evidence architecture contract is stale")
    if (
        selected_record["benchmark_model_config_sha256"]
        != current_model_config_sha256
    ):
        raise RuntimeError("Figure-1 GPU evidence config differs from current config")
    if selected_record["brainlm_identity"] != current_brainlm_identity:
        raise RuntimeError("Figure-1 GPU evidence BrainLM authority/artifacts differ")
    measured_ddp_step_seconds = max(
        float(rank["step_seconds"]) for rank in ddp["rank_records"]
    )
    return {
        "profile": profile,
        "depth_slab_size": core,
        "summary_path": str(path),
        "summary_sha256": expected_digest,
        "ddp_smoke_path": str(ddp_path),
        "ddp_smoke_sha256": ddp_expected_digest,
        "measured_ddp_step_seconds": measured_ddp_step_seconds,
        # The authenticated recovery subset contains 2,078 visits and uses
        # drop_last at the paper global batch of four: 519 optimizer steps.
        "recovery_half_epoch_eta_seconds": (
            measured_ddp_step_seconds * (2078 // 4)
            if profile == "recovery"
            else None
        ),
    }


__all__ = [
    "BENCHMARK_RECORD_SCHEMA",
    "DDP_SMOKE_SCHEMA",
    "DERIVED_DEPTH_HALO",
    "EXPECTED_CORES",
    "MINIMUM_SAFETY_MARGIN_GIB",
    "PAPER_PROFILE_BLOCK_STATUS",
    "PROFILE_SHAPES",
    "PROTOCOL_TO_BENCHMARK_PROFILE",
    "SWEEP_SUMMARY_SCHEMA",
    "atomic_write_json_no_replace",
    "authenticate_configured_depth_slab_selection",
    "benchmark_model_config_identity",
    "canonical_sha256",
    "seal_benchmark_record",
    "seal_sweep_summary",
    "scheduler_allocation_sha256",
    "sha256_file",
    "source_tree_identity",
    "validate_benchmark_record",
    "validate_ddp_smoke_attestation",
    "validate_sweep_summary",
]
