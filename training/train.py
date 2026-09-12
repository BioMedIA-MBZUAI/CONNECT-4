"""
CONNECT-4 production training (exact four-GPU DistributedDataParallel).

Features
--------
* externally fixed patient-level train/development/sealed split (no leakage),
* the Figure-1E objective plus explicit temporal coherence,
* periodic validation with per-subject evaluation metrics printed
  on rank 0,
* periodic real-vs-synthetic magma visualisation (`eval/visualize.py`),
* bf16 + gradient accumulation + per-epoch checkpoints (rank 0).

This module is training-only by construction.  It never creates a sealed-test
loader, opens a sealed fMRI target/sidecar, initializes SLIM-Brain, or publishes
final paired metrics.  Final test inference is a separate explicit entry point.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data.dataset_precomputed import (
    Connect4PrecomputedDataset,
    discover_structural_cache_scan_ids,
)
from data.cohort_admission import (
    broadcast_rank_zero_cohort_admission,
    build_cohort_admission,
    build_validation_shard_contract,
    cohort_admission_identity,
    validate_cohort_admission_identity,
    validate_production_training_runtime_identity,
    validate_validation_shard_contract,
)
from data.collate import connect4_collate_fn
from data.protocol import (
    RUN_ARTIFACT_IDENTITY_SCHEMA,
    TRAINING_CHECKPOINT_FORMAT,
    build_run_artifact_identity,
    build_split_identity,
    patient_level_split_from_manifest,
    validate_fixed_protocol_config,
    validate_training_cohort_manifest,
    validate_training_scaler_provenance,
)
from models.connect4 import Connect4Model
from models.brainlm_context import validate_configured_brainlm_authority_scope
from eval.development_quality import (
    ROUTINE_TIER,
    SELECTION_TIER,
    checkpoint_candidate_identity,
    development_tensor_identity,
    evaluated_tensor_set_identity,
    model_state_sha256,
    publish_development_quality_record,
    seal_development_quality_record,
    validate_development_quality_record,
    validate_progress_checkpoint_development_qa,
    validate_selectable_checkpoint_development_qa,
)
from eval.metrics import SynthesisMetricAccumulator
from eval.quality import (
    DEVELOPMENT_REQUIRED_CHECKS,
    QualityContractError,
    QualityPolicy,
    evaluate_4d_pair_quality_arrays,
)
from eval.visualize import plot_real_vs_synthetic_4d
from preprocessing.conform import anatomical_affine, crop_architecture_array
from utils.config import load_config, resolve_configured_value
from utils.figure1_gpu_gate import (
    PROTOCOL_TO_BENCHMARK_PROFILE,
    authenticate_configured_depth_slab_selection,
    canonical_sha256,
)
from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_LABEL_TO_CHANNEL,
    CANONICAL_ROI_MAPPING_SHA256,
    REJECTED_LEGACY_CHECKPOINT_FORMATS,
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
    require_current_synthesis_architecture,
    require_production_token_codec_shape,
    synthesis_architecture_contract,
)


PAPER_EFFECTIVE_BATCH_SIZE = 4
PRODUCTION_WORLD_SIZE = 4
PRODUCTION_DISTRIBUTED_TIMEOUT = timedelta(hours=6)


def _production_token_codec_shape_from_config(
    config: Mapping[str, Any],
) -> Optional[dict[str, int]]:
    """Return exact V18 shape evidence for production profiles only."""

    data = config.get("data")
    models = config.get("models")
    if not isinstance(data, Mapping) or not isinstance(models, Mapping):
        return None
    profile = str(data.get("protocol_profile", "")).strip()
    if profile not in TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE:
        return None
    dit = models.get("dit")
    if not isinstance(dit, Mapping):
        raise RuntimeError("V18 production checkpoint config has no DiT mapping")
    try:
        patch_size = tuple(dit["patch_size"])
        patch_value_dim = (
            int(data["out_channels"])
            * int(dit["temporal_patch_size"])
            * math.prod(patch_size)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "V18 production checkpoint config has incomplete token-codec shape"
        ) from exc
    return require_production_token_codec_shape(
        protocol_profile=profile,
        hidden_size=dit.get("hidden_size"),
        patch_value_dim=patch_value_dim,
    )


# --------------------------------------------------------------------------- #
def setup_distributed():
    """Initialize the exact four-rank NCCL world with a six-hour timeout."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=PRODUCTION_DISTRIBUTED_TIMEOUT,
        )
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def require_production_training_runtime(
    config: dict,
    *,
    is_distributed: bool,
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict:
    """Load and bind the externally qualified four-A100 runtime identity."""

    if not is_distributed or world_size != PRODUCTION_WORLD_SIZE:
        raise RuntimeError("production training requires exactly four DDP ranks")
    identity_path_value = os.environ.get("CONNECT4_RUNTIME_IDENTITY", "").strip()
    if not identity_path_value:
        raise RuntimeError(
            "CONNECT4_RUNTIME_IDENTITY must point to the qualified runtime JSON"
        )
    identity_path = Path(identity_path_value)
    if not identity_path.is_absolute() or identity_path.is_symlink():
        raise RuntimeError(
            "CONNECT4_RUNTIME_IDENTITY must be an absolute non-symlink path"
        )
    raw = identity_path.read_bytes()
    if not raw or len(raw) > 1024 * 1024:
        raise RuntimeError("runtime identity JSON is empty or unexpectedly large")
    try:
        identity = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime identity is not canonical UTF-8 JSON") from exc
    runtime = validate_production_training_runtime_identity(identity)
    if runtime["world_size"] != world_size:
        raise RuntimeError("runtime identity world size differs from this launch")
    if runtime["gpu_records"][rank]["local_rank"] != local_rank:
        raise RuntimeError("runtime identity rank-to-GPU mapping differs")
    if runtime["slurm_job_id"] != os.environ.get("SLURM_JOB_ID", ""):
        raise RuntimeError("runtime identity is bound to another Slurm job")
    return runtime


def is_main(rank: int) -> bool:
    return rank == 0


def _run_rank_zero_synchronized(
    action: Callable[[], Any],
    *,
    rank: int,
    world_size: int,
    distributed: bool,
    label: str,
) -> Any:
    """Run a root-only publication while every rank exchanges its outcome."""

    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or rank not in range(world_size)
        or not isinstance(distributed, bool)
        or distributed != (world_size > 1)
        or not isinstance(label, str)
        or not label
    ):
        raise ValueError("rank-zero synchronized action arguments are invalid")
    local_status = {
        "rank": rank,
        "attempted": rank == 0,
        "ok": True,
        "result": None,
        "error": None,
    }
    if rank == 0:
        try:
            local_status["result"] = action()
        except BaseException as exc:  # coordinate failures before re-raising
            local_status["ok"] = False
            local_status["error"] = (
                f"{type(exc).__name__}: {str(exc)[:2000]}"
            )
    if distributed:
        statuses: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(statuses, local_status)
    else:
        statuses = [local_status]
    if (
        any(not isinstance(status, Mapping) for status in statuses)
        or [status["rank"] for status in statuses] != list(range(world_size))
        or statuses[0].get("attempted") is not True
        or any(status.get("attempted") is not False for status in statuses[1:])
        or any(status.get("ok") is not True for status in statuses[1:])
    ):
        raise RuntimeError(f"{label} rank-status exchange is invalid")
    root_status = statuses[0]
    if root_status.get("ok") is not True:
        raise RuntimeError(f"{label} failed on rank zero: {root_status.get('error')}")
    return root_status.get("result")


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _require_supervised_batch_contracts(
    batch: Mapping[str, Any],
    *,
    protocol_profile: str,
) -> None:
    """Authenticate ROI order and paired-support semantics before model use."""

    fmri = batch.get("fmri")
    structural = batch.get("brain_mask")
    target_validity = batch.get("target_validity_mask")
    if (
        not torch.is_tensor(fmri)
        or not torch.is_tensor(structural)
        or not torch.is_tensor(target_validity)
    ):
        raise RuntimeError(
            "supervised train/development batches require fMRI plus separate "
            "structural and target-validity masks"
        )
    batch_size = int(fmri.shape[0]) if fmri.ndim >= 1 else 0
    if (
        batch_size < 1
        or fmri.ndim != 6
        or fmri.shape[1] != 1
        or structural.ndim != 5
        or target_validity.ndim != 5
        or structural.shape != target_validity.shape
        or tuple(target_validity.shape[:2]) != (batch_size, 1)
        or tuple(target_validity.shape[-3:]) != tuple(fmri.shape[-3:])
    ):
        raise RuntimeError(
            "supervised batch target/structural/validity extents differ"
        )
    if (
        not bool(torch.isfinite(structural).all())
        or not bool(torch.isfinite(target_validity).all())
        or not bool(((structural == 0) | (structural == 1)).all())
        or not bool(((target_validity == 0) | (target_validity == 1)).all())
    ):
        raise RuntimeError(
            "supervised structural and target-validity masks must be finite and "
            "exactly binary"
        )
    structural_binary = structural.bool()
    validity_binary = target_validity.bool()
    if not bool(validity_binary.flatten(1).any(dim=1).all()):
        raise RuntimeError("supervised target-validity mask is empty")
    if bool((validity_binary & ~structural_binary).any()):
        raise RuntimeError(
            "supervised target-validity mask lies outside structural support; "
            "the two masks may have been swapped"
        )
    exact_target_support = fmri.ne(0).any(dim=2, keepdim=False)
    if not torch.equal(validity_binary, exact_target_support):
        raise RuntimeError(
            "supervised target-validity mask differs from exact paired-fMRI "
            "support"
        )

    expected_contract = TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(
        protocol_profile
    )
    if expected_contract is None:
        raise RuntimeError(
            f"no target-validity contract is bound to {protocol_profile!r}"
        )
    contracts = batch.get("target_validity_mask_contract")
    if isinstance(contracts, str):
        contracts = [contracts]
    if (
        not isinstance(contracts, (list, tuple))
        or len(contracts) != batch_size
        or any(value != expected_contract for value in contracts)
    ):
        raise RuntimeError(
            "supervised batch target-validity contract differs from protocol "
            f"profile {protocol_profile!r}"
        )

    roi_mapping = batch.get("structure_to_roi_idx")
    if roi_mapping != dict(CANONICAL_ROI_LABEL_TO_CHANNEL):
        raise RuntimeError("supervised batch differs from the canonical 32-ROI mapping")
    mapping_digests = batch.get("roi_mapping_sha256")
    if isinstance(mapping_digests, str):
        mapping_digests = [mapping_digests]
    if (
        not isinstance(mapping_digests, (list, tuple))
        or len(mapping_digests) != batch_size
        or any(value != CANONICAL_ROI_MAPPING_SHA256 for value in mapping_digests)
    ):
        raise RuntimeError("supervised batch canonical ROI digest is missing or differs")
    label_orders = batch.get("roi_label_ids")
    if (
        not isinstance(label_orders, (list, tuple))
        or len(label_orders) != batch_size
        or any(tuple(value) != tuple(CANONICAL_ROI_LABEL_IDS) for value in label_orders)
    ):
        raise RuntimeError("supervised batch canonical ROI label order differs")


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    """Convert the manuscript's global batch size to a DDP local batch size."""
    if isinstance(global_batch_size, bool) or not isinstance(global_batch_size, int):
        raise TypeError("training.batch_size must be an integer global batch size")
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an integer")
    if global_batch_size < 1 or world_size < 1:
        raise ValueError("global batch size and world size must both be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"global training batch size {global_batch_size} is not divisible by "
            f"DDP world size {world_size}; use a divisible world size so the "
            "effective batch remains exactly the configured value"
        )
    return global_batch_size // world_size


def validate_paper_effective_batch(
    global_micro_batch_size: int,
    world_size: int,
    grad_accum_steps: int,
) -> int:
    """Return the local micro-batch after proving effective batch size is four."""
    local_batch = per_rank_batch_size(global_micro_batch_size, world_size)
    if isinstance(grad_accum_steps, bool) or not isinstance(grad_accum_steps, int):
        raise TypeError("training.grad_accum_steps must be an integer")
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be positive")
    effective = global_micro_batch_size * grad_accum_steps
    if effective != PAPER_EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            "paper-faithful optimization requires effective global batch size 4; "
            f"got {global_micro_batch_size} x {grad_accum_steps} accumulation = {effective}"
        )
    return local_batch


def seed_process(seed: int, rank: int) -> None:
    """Seed model-side randomness independently on every DDP rank."""
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        # Keep the checkpoint compatible with torch.load(weights_only=True): a
        # raw NumPy ndarray would require unsafe pickle-global allowlisting.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    required = {"python", "numpy", "torch"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise RuntimeError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict) or not {
        "bit_generator", "keys", "position", "has_gauss", "cached_gaussian"
    }.issubset(numpy_state):
        raise RuntimeError("checkpoint NumPy RNG state is incomplete")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        if "cuda" not in state:
            raise RuntimeError("CUDA resume requires saved CUDA RNG state")
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot leave `latest.pt` torn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_torch_save_no_replace(payload: dict, path: Path) -> None:
    """Atomically publish one immutable selectable checkpoint."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    original_parent = path.parent.absolute()
    parent = path.parent.resolve(strict=True)
    if path.parent.is_symlink() or parent != original_parent:
        raise RuntimeError("selectable-checkpoint parent is unsafe")
    path = parent / path.name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace selectable checkpoint {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_selectable_checkpoint(payload: dict, path: Path) -> None:
    """Validate the all-development attestation before any candidate write."""

    validate_selectable_checkpoint_development_qa(payload)
    _atomic_torch_save_no_replace(payload, path)


def require_passing_final_development_qa(result: Mapping[str, Any]) -> dict:
    """Fail the configured training horizon unless the all-dev gate passed."""

    if not isinstance(result, Mapping) or not isinstance(result.get("record"), Mapping):
        raise RuntimeError("training completion has no development-QA record")
    try:
        return validate_development_quality_record(
            result["record"],
            require_selection_pass=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "training reached its configured horizon without an authenticated "
            "all-development quality pass; no selectable checkpoint was published"
        ) from exc


def _checkpoint_config(cfg: dict) -> dict:
    """Deep-copy the exact run config to make resume drift detectable."""
    return copy.deepcopy(cfg)


def _runtime_resume_contract(runtime_identity: dict) -> dict:
    """Drop job-instance IDs while preserving the authenticated runtime class."""

    runtime = validate_production_training_runtime_identity(runtime_identity)
    gpu_records = runtime.get("gpu_records")
    gpu_contract = []
    if isinstance(gpu_records, list):
        gpu_contract = [
            {
                "gpu_name": record.get("gpu_name"),
                "gpu_total_memory_gib": record.get("gpu_total_memory_gib"),
                "nvidia_driver_version": record.get("nvidia_driver_version"),
            }
            for record in gpu_records
            if isinstance(record, dict)
        ]
    return {
        key: runtime.get(key)
        for key in (
            "schema",
            "slurm_partition",
            "slurm_qos",
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
            "slab_selection",
        )
    } | {"gpu_hardware": gpu_contract}


def _require_resume_admission_compatibility(
    checkpoint_identity: dict,
    current_identity: dict,
) -> None:
    checkpoint_admission = validate_cohort_admission_identity(
        checkpoint_identity
    )
    current_admission = validate_cohort_admission_identity(current_identity)
    stable_fields = (
        "data_root_sha256",
        "config_sha256",
        "ordered_scan_ids_sha256",
        "partitions_sha256",
        "split_identity_sha256",
        "artifact_identity_sha256",
        "structural_artifact_identities_sha256",
        "dataset_state_sha256",
        "world_size",
    )
    differences = [
        field
        for field in stable_fields
        if checkpoint_admission[field] != current_admission[field]
    ]
    if differences:
        raise RuntimeError(
            "checkpoint cohort-admission root differs from the current run: "
            f"{differences}"
        )


def _require_admission_payload_bindings(
    identity: dict,
    *,
    runtime_identity: dict,
    config: dict,
    split_identity: dict,
    artifact_identity: dict,
    world_size: int,
    label: str,
) -> dict:
    """Bind a compact admission receipt back to this checkpoint/run payload."""

    admission = validate_cohort_admission_identity(identity)
    runtime = validate_production_training_runtime_identity(runtime_identity)
    if (
        admission["runtime_identity_sha256"] != runtime["canonical_sha256"]
        or admission["config_sha256"] != canonical_sha256(config)
        or admission["split_identity_sha256"] != canonical_sha256(split_identity)
        or admission["artifact_identity_sha256"]
        != canonical_sha256(artifact_identity)
        or admission["structural_artifact_identities_sha256"]
        != artifact_identity.get("structural_artifact_identities_sha256")
        or admission["world_size"] != world_size
    ):
        raise RuntimeError(f"{label} cohort-admission payload bindings differ")
    return admission


def validate_resume_checkpoint(
    checkpoint: dict,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: dict,
    split_identity: dict,
    artifact_identity: dict,
    runtime_identity: dict,
    cohort_admission_identity: dict,
    validation_shard_contract: dict,
    world_size: int,
    rank: int,
    batches_per_epoch: int,
    grad_accum_steps: int,
) -> tuple[int, int, int]:
    """Strictly restore a checkpoint and return (epoch, batch index, step).

    Partial checkpoints are saved only immediately after an optimizer update.
    The deterministic sampler can therefore replay the epoch order and skip the
    already completed batches without duplicating a gradient update.
    """
    observed_format = checkpoint.get("format") if isinstance(checkpoint, dict) else None
    if observed_format in REJECTED_LEGACY_CHECKPOINT_FORMATS:
        raise RuntimeError(
            "pre-v18 synthesis checkpoints, including unbounded, independently "
            "scaled, or "
            "bias-shifted token codecs, raw-space epsilon, or target-leaking structural "
            "identities, detached/full-DDIM composite-loss gradients, and "
            "unweighted legacy one-step estimates, cannot resume"
        )
    if not isinstance(checkpoint, dict) or observed_format != TRAINING_CHECKPOINT_FORMAT:
        raise RuntimeError(
            "checkpoint is not an iteration-exact "
            f"{TRAINING_CHECKPOINT_FORMAT!r} training state"
        )
    require_current_synthesis_architecture(
        checkpoint.get("architecture_contract"),
        checkpoint.get("architecture_contract_sha256"),
    )
    if checkpoint.get("config") != _checkpoint_config(config):
        raise RuntimeError("checkpoint config differs from the current training config")
    expected_codec_shape = _production_token_codec_shape_from_config(config)
    if expected_codec_shape is not None:
        if checkpoint.get("production_token_codec_shape") != expected_codec_shape:
            raise RuntimeError(
                "checkpoint V18 production token-codec shape evidence differs"
            )
        shape_evidence = getattr(
            model,
            "production_token_codec_shape_evidence",
            None,
        )
        if not callable(shape_evidence) or shape_evidence() != expected_codec_shape:
            raise RuntimeError(
                "instantiated model V18 production token-codec shape differs"
            )
    if checkpoint.get("split_identity") != split_identity:
        raise RuntimeError("checkpoint cohort/split identity differs from the current run")
    checkpoint_artifact = checkpoint.get("artifact_identity")
    if (
        not isinstance(checkpoint_artifact, dict)
        or checkpoint_artifact.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
        or not isinstance(artifact_identity, dict)
        or artifact_identity.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
    ):
        raise RuntimeError(
            "stale run-artifact identity is forbidden; "
            f"{RUN_ARTIFACT_IDENTITY_SCHEMA} is required"
        )
    if checkpoint.get("artifact_identity") != artifact_identity:
        raise RuntimeError("checkpoint data/evaluation artifacts differ from the current run")
    checkpoint_runtime = checkpoint.get("runtime_identity")
    try:
        runtime_contract_differs = (
            not isinstance(checkpoint_runtime, dict)
            or _runtime_resume_contract(checkpoint_runtime)
            != _runtime_resume_contract(runtime_identity)
        )
    except RuntimeError as exc:
        raise RuntimeError("checkpoint production runtime contract differs") from exc
    if runtime_contract_differs:
        raise RuntimeError("checkpoint production runtime contract differs")
    checkpoint_admission_value = checkpoint.get("cohort_admission_identity")
    if not isinstance(checkpoint_admission_value, dict):
        raise RuntimeError("checkpoint has no cohort-admission identity")
    checkpoint_admission = _require_admission_payload_bindings(
        checkpoint_admission_value,
        runtime_identity=checkpoint_runtime,
        config=checkpoint["config"],
        split_identity=checkpoint["split_identity"],
        artifact_identity=checkpoint_artifact,
        world_size=world_size,
        label="checkpoint",
    )
    current_admission = _require_admission_payload_bindings(
        cohort_admission_identity,
        runtime_identity=runtime_identity,
        config=config,
        split_identity=split_identity,
        artifact_identity=artifact_identity,
        world_size=world_size,
        label="current run",
    )
    _require_resume_admission_compatibility(
        checkpoint_admission, current_admission
    )
    try:
        current_validation_shards = validate_validation_shard_contract(
            validation_shard_contract,
            expected_cohort_admission_identity=current_admission,
            expected_world_size=world_size,
        )
        checkpoint_validation_shards = validate_validation_shard_contract(
            checkpoint.get("validation_shard_contract"),
            expected_cohort_admission_identity=checkpoint_admission,
            expected_world_size=world_size,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint validation-shard contract is invalid: {exc}"
        ) from exc
    job_specific_validation_fields = {
        "cohort_admission_identity_record_sha256",
        "record_sha256",
    }
    checkpoint_stable_shards = {
        key: value
        for key, value in checkpoint_validation_shards.items()
        if key not in job_specific_validation_fields
    }
    current_stable_shards = {
        key: value
        for key, value in current_validation_shards.items()
        if key not in job_specific_validation_fields
    }
    if checkpoint_stable_shards != current_stable_shards:
        raise RuntimeError("checkpoint validation-shard contract differs")
    development_size = checkpoint_validation_shards["development_partition_size"]
    current_selection_value = build_validation_shard_contract(
        current_admission,
        development_partition_size=development_size,
        global_limit=development_size,
        world_size=world_size,
    )
    try:
        checkpoint_selection_shards = validate_validation_shard_contract(
            checkpoint.get("selection_validation_shard_contract"),
            expected_cohort_admission_identity=checkpoint_admission,
            expected_world_size=world_size,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint all-development shard contract is invalid: {exc}"
        ) from exc
    checkpoint_stable_selection = {
        key: value
        for key, value in checkpoint_selection_shards.items()
        if key not in job_specific_validation_fields
    }
    current_stable_selection = {
        key: value
        for key, value in current_selection_value.items()
        if key not in job_specific_validation_fields
    }
    if (
        checkpoint_selection_shards["global_limit"] != development_size
        or checkpoint_stable_selection != current_stable_selection
    ):
        raise RuntimeError("checkpoint all-development shard contract differs")
    if int(checkpoint.get("world_size", -1)) != int(world_size):
        raise RuntimeError("checkpoint DDP world size differs from the current run")
    for key in ("model", "optimizer", "scaler", "rng_states"):
        if key not in checkpoint:
            raise RuntimeError(f"checkpoint is missing required state {key!r}")

    next_epoch = checkpoint.get("next_epoch")
    next_batch = checkpoint.get("next_batch_index")
    completed_batches = checkpoint.get("completed_batches")
    for name, value in (
        ("next_epoch", next_epoch),
        ("next_batch_index", next_batch),
        ("completed_batches", completed_batches),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"checkpoint {name} must be a non-negative integer")
    if next_batch >= batches_per_epoch:
        if next_batch != 0:
            raise RuntimeError("checkpoint next_batch_index is outside the epoch")
    if next_batch % grad_accum_steps:
        raise RuntimeError("checkpoint resumes inside an unsaved accumulation window")
    expected_batches = next_epoch * batches_per_epoch + next_batch
    if completed_batches != expected_batches:
        raise RuntimeError(
            "checkpoint iteration counters are inconsistent with the current loader"
        )

    partial = checkpoint.get("partial")
    eligible = checkpoint.get("selection_eligible")
    if not isinstance(partial, bool) or not isinstance(eligible, bool):
        raise RuntimeError("checkpoint QA selection state is missing")
    latest_qa = checkpoint.get("latest_development_qa")
    latest_file_sha = checkpoint.get("latest_development_qa_file_sha256")
    if latest_qa is None:
        if latest_file_sha is not None:
            raise RuntimeError("checkpoint latest development-QA digest is orphaned")
    else:
        validate_development_quality_record(latest_qa)
        if (
            not isinstance(latest_file_sha, str)
            or len(latest_file_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in latest_file_sha
            )
        ):
            raise RuntimeError("checkpoint latest development-QA file digest is invalid")
    if partial:
        validate_progress_checkpoint_development_qa(checkpoint)
    else:
        validate_selectable_checkpoint_development_qa(checkpoint)

    rng_states = checkpoint["rng_states"]
    if not isinstance(rng_states, list) or len(rng_states) != world_size:
        raise RuntimeError("checkpoint does not contain one RNG state per DDP rank")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng_state(rng_states[rank])
    except Exception as exc:
        raise RuntimeError(
            f"checkpoint state cannot be restored exactly: {type(exc).__name__}: {exc}"
        ) from exc
    return next_epoch, next_batch, completed_batches


def _precomputed_dataset_kwargs(cfg, target_scan_ids, target_scan_roles):
    d = cfg["data"]
    recovery_profile = not d.get("require_paper_preprocessing", True)
    return {
        "root_dir": d["root_dir"],
        "precomputed_dir": d["precomputed_dir"],
        "target_shape": tuple(d["architecture_shape"]),
        "num_frames": int(d["num_frames"]),
        "normalize_intensity": d.get("normalize_intensity", True),
        "require_paper_preprocessing": d.get(
            "require_paper_preprocessing", True
        ),
        "dwi_matrix_path": d.get("dwi_matrix_path"),
        "fmri_dir": d.get("fmri_dir"),
        "scaler_dir": d.get("scaler_dir"),
        "normative_csv_path": d.get("normative_csv_path"),
        "cohort_manifest_path": d.get("cohort_manifest"),
        "brainiac_model_path": cfg["models"].get("brainiac_path"),
        "brainiac_checkpoint_sha256": resolve_configured_value(
            cfg["models"],
            "brainiac_checkpoint_sha256",
            "brainiac_checkpoint_sha256_env",
        ),
        "brainiac_source_sha256": resolve_configured_value(
            cfg["models"],
            "brainiac_source_sha256",
            "brainiac_source_sha256_env",
        ),
        "modernbert_model_name": cfg["models"].get("modernbert_name"),
        "modernbert_revision": cfg["models"].get("modernbert_revision"),
        "patch_size": tuple(cfg["models"]["graphs"]["patch_size"]),
        "image_node_dim": int(cfg["models"]["fusion"]["image_embed_dim"]),
        "mask_node_dim": int(cfg["models"]["fusion"]["mask_embed_dim"]),
        "roi_node_dim": int(cfg["models"]["fusion"]["roi_embed_dim"]),
        "target_scan_ids": target_scan_ids,
        "common_grid_contract_path": d.get("common_grid_contract_path"),
        "common_grid_contract_sha256": resolve_configured_value(
            d,
            "common_grid_contract_sha256",
            "common_grid_contract_sha256_env",
        ),
        "structural_stage_root": d.get("structural_stage_root"),
        "native_alignment_authority_path": d.get(
            "native_alignment_authority_path"
        ),
        "native_alignment_authority_sha256": resolve_configured_value(
            d,
            "native_alignment_authority_sha256",
            "native_alignment_authority_sha256_env",
        ),
        "native_selection_manifest_path": d.get(
            "native_selection_manifest_path"
        ),
        "native_selection_manifest_sha256": resolve_configured_value(
            d,
            "native_selection_manifest_sha256",
            "native_selection_manifest_sha256_env",
        ),
        "native_selection_root_review_path": d.get(
            "native_selection_root_review_path"
        ),
        "native_selection_root_review_sha256": resolve_configured_value(
            d,
            "native_selection_root_review_sha256",
            "native_selection_root_review_sha256_env",
        ),
        "native_completed_set_path": d.get("native_completed_set_path"),
        "native_completed_set_sha256": resolve_configured_value(
            d,
            "native_completed_set_sha256",
            "native_completed_set_sha256_env",
        ),
        "native_completed_set_commit_marker_path": d.get(
            "native_completed_set_commit_marker_path"
        ),
        "native_completed_set_commit_marker_sha256": resolve_configured_value(
            d,
            "native_completed_set_commit_marker_sha256",
            "native_completed_set_commit_marker_sha256_env",
        ),
        "native_reviewed_source_path": d.get("native_reviewed_source_path"),
        "native_reviewed_source_sha256": resolve_configured_value(
            d,
            "native_reviewed_source_sha256",
            "native_reviewed_source_sha256_env",
        ),
        "native_runtime_attester_sha256": resolve_configured_value(
            d,
            "native_runtime_attester_sha256",
            "native_runtime_attester_sha256_env",
        ),
        "native_verifier_sha256": resolve_configured_value(
            d,
            "native_verifier_sha256",
            "native_verifier_sha256_env",
        ),
        "target_scan_roles": target_scan_roles if recovery_profile else None,
        "allow_synthetic_grid_override": d.get(
            "allow_synthetic_grid_override", False
        ),
    }


def _build_rank_zero_training_admission(cfg, world, runtime_identity):
    """Perform the one whole-cohort admission permitted for a DDP run."""

    d = cfg["data"]
    scan_ids = discover_structural_cache_scan_ids(
        d["root_dir"],
        d["precomputed_dir"],
        structural_stage_root=d.get("structural_stage_root"),
    )
    cohort_manifest = d.get("cohort_manifest")
    if not cohort_manifest:
        raise ValueError(
            "data.cohort_manifest is required to prove the complete synthesis cohort"
        )
    cohort_evidence = validate_training_cohort_manifest(
        cohort_manifest,
        scan_ids,
        expected_scan_counts=d.get("expected_cohort_scan_counts"),
        protocol_profile=d.get("protocol_profile"),
    )
    tr_idx, va_idx, te_idx = patient_level_split_from_manifest(
        scan_ids,
        cohort_evidence.patient_by_scan,
        val_frac=cfg["training"].get("val_frac", 0.15),
        test_frac=cfg["training"].get("test_frac", 0.15),
        seed=cfg["training"].get("seed", 42),
        manifest_path=cfg["training"].get("split_manifest"),
        manifest_sha256=resolve_configured_value(
            cfg["training"],
            "split_manifest_sha256",
            "split_manifest_sha256_env",
        ),
        protocol_profile=str(d.get("protocol_profile", "")),
    )
    split_identity = build_split_identity(
        scan_ids,
        tr_idx,
        va_idx,
        te_idx,
        cohort_evidence.cohort_by_scan,
        cohort_evidence.patient_by_scan,
    )
    target_scan_ids = {
        scan_ids[index] for index in [*tr_idx, *va_idx]
    }
    target_scan_roles = {
        **{scan_ids[index]: "train" for index in tr_idx},
        **{
            scan_ids[index]: "development-validation"
            for index in va_idx
        },
    }
    perceptual_extractor_spec = (
        cfg.get("training", {}).get("loss", {}).get("perceptual_extractor")
    )
    brainlm_authority_scope = None
    if d.get("protocol_profile") == "a4-native-recovery-v1":
        brainlm_authority_scope = validate_configured_brainlm_authority_scope(
            perceptual_extractor_spec,
            expected_scan_roles=target_scan_roles,
            split_identity=split_identity,
        )
    full = Connect4PrecomputedDataset(
        **_precomputed_dataset_kwargs(
            cfg, target_scan_ids, target_scan_roles
        )
    )
    if full.scan_ids != scan_ids:
        raise RuntimeError(
            "validated dataset order differs from the structural split inventory"
        )
    validate_training_scaler_provenance(
        d.get("scaler_dir"),
        [full.scan_ids[index] for index in tr_idx],
    )
    artifact_identity = build_run_artifact_identity(
        full,
        cohort_manifest_path=cohort_manifest,
        evaluation_extractor_spec=None,
        perceptual_extractor_spec=perceptual_extractor_spec,
        brainlm_authority_scope=brainlm_authority_scope,
        target_scan_ids=sorted(target_scan_ids),
    )
    dataset_state = full.build_rank_zero_admission_state(
        target_scan_roles=target_scan_roles
    )
    return build_cohort_admission(
        protocol_profile=d.get("protocol_profile"),
        world_size=world,
        config_sha256=canonical_sha256(cfg),
        runtime_identity=runtime_identity,
        ordered_scan_ids=scan_ids,
        train_indices=tr_idx,
        development_indices=va_idx,
        sealed_indices=te_idx,
        split_identity=split_identity,
        artifact_identity=artifact_identity,
        dataset_state=dataset_state,
    )


def build_loaders(cfg, world, rank, runtime_identity=None):
    validate_fixed_protocol_config(cfg)
    d = cfg["data"]
    if d.get("protocol_profile") in PROTOCOL_TO_BENCHMARK_PROFILE:
        try:
            authenticate_configured_depth_slab_selection(cfg)
        except RuntimeError as exc:
            raise RuntimeError(
                "RELEASE STOP: Figure-1 V18 training requires both the current-"
                "source-bound per-rank A100-40GB largest-safe-core sweep and a "
                "matching exact four-GPU DDP full-step attestation"
            ) from exc
    if not isinstance(runtime_identity, dict):
        raise RuntimeError("training runtime identity is required before data admission")
    runtime_identity = validate_production_training_runtime_identity(runtime_identity)
    runtime_identity_sha256 = runtime_identity["canonical_sha256"]
    admission = broadcast_rank_zero_cohort_admission(
        lambda: _build_rank_zero_training_admission(
            cfg, world, runtime_identity
        ),
        rank=rank,
        world_size=world,
        expected_config_sha256=canonical_sha256(cfg),
        expected_runtime_identity_sha256=runtime_identity_sha256,
    )
    full = Connect4PrecomputedDataset.from_rank_zero_admission_state(
        admission["dataset_state"]
    )
    partitions = admission["partitions"]
    tr_idx = partitions["train"]
    va_idx = partitions["development_validation"]
    te_idx = partitions["sealed_test"]
    split_identity = admission["split_identity"]
    artifact_identity = admission["artifact_identity"]
    admission_identity = cohort_admission_identity(admission)
    if is_main(rank):
        n_pat = len(
            {
                record["patient_id"]
                for records in split_identity["partitions"].values()
                for record in records
            }
        )
        print(
            f"[data] rank-zero admitted {len(full.scan_ids)} scans / {n_pat} patients "
            f"-> train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} "
            f"(authority {admission_identity['record_sha256']})",
            flush=True,
        )
    # Fail closed on a corrupt sample. Randomly substituting another subject
    # changes the fixed patient-wise sampling distribution and can conceal a
    # preprocessing-contract violation.
    train_ds = Subset(full, tr_idx)
    val_ds = Subset(full, va_idx)

    global_batch = cfg["training"]["batch_size"]
    accumulation = cfg["training"].get("grad_accum_steps", 1)
    local_batch = validate_paper_effective_batch(global_batch, world, accumulation)
    if is_main(rank):
        print(
            f"[data] global training batch {global_batch} -> "
            f"{local_batch} sample(s) per rank across {world} rank(s), "
            f"accumulation {accumulation} -> effective batch {PAPER_EFFECTIVE_BATCH_SIZE}",
            flush=True,
        )
    # Use the epoch-seeded sampler even on one rank. Reconstructing its order and
    # skipping completed batches is what makes partial-checkpoint resume exact.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=int(cfg["training"].get("seed", 42)),
        drop_last=True,
    )
    loader_generator = torch.Generator().manual_seed(
        int(cfg["training"].get("seed", 42)) + int(rank)
    )
    train_loader = DataLoader(
        train_ds, batch_size=local_batch, shuffle=False,
        sampler=train_sampler, num_workers=cfg["training"].get("num_workers", 2),
        collate_fn=connect4_collate_fn, pin_memory=True, drop_last=True,
        generator=loader_generator,
    )
    val_batches = cfg["training"].get("val_batches", 4)
    global_validation_limit = min(val_batches, len(val_ds))
    validation_shards = build_validation_shard_contract(
        admission_identity,
        development_partition_size=len(val_ds),
        global_limit=global_validation_limit,
        world_size=world,
    )
    selection_validation_shards = build_validation_shard_contract(
        admission_identity,
        development_partition_size=len(val_ds),
        global_limit=len(val_ds),
        world_size=world,
    )
    rank_val_ds = build_rank_local_validation_subset(
        val_ds,
        limit=val_batches,
        world_size=world,
        rank=rank,
        validation_shard_contract=validation_shards,
    )
    rank_val_ds.connect4_selection_validation_shard_contract = (
        selection_validation_shards
    )
    val_loader = DataLoader(
        rank_val_ds, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=connect4_collate_fn, pin_memory=True,
    )
    if len(train_loader) < 1:
        raise ValueError("training partition is too small for one complete global batch")
    if len(train_loader) % accumulation:
        raise ValueError(
            f"{len(train_loader)} training micro-batches are not divisible by "
            f"grad_accum_steps={accumulation}; refusing a smaller final optimizer batch"
        )
    return (
        train_loader,
        val_loader,
        train_sampler,
        split_identity,
        artifact_identity,
        admission_identity,
    )


def validation_shard_indices(limit: int, world_size: int, rank: int) -> tuple[int, ...]:
    """Return a complete, disjoint, balanced validation work partition."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or world_size < 1
        or rank not in range(world_size)
    ):
        raise ValueError("validation shard arguments are invalid")
    return tuple(index for index in range(limit) if index % world_size == rank)


def build_rank_local_validation_subset(
    dataset,
    *,
    limit: int,
    world_size: int,
    rank: int,
    validation_shard_contract: dict,
) -> Subset:
    """Restrict target materialization to this rank's global first-N shard."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("training.val_batches must be a positive integer")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank not in range(world_size)
    ):
        raise ValueError("validation rank/world size is invalid")
    global_limit = min(limit, len(dataset))
    if global_limit < 1:
        raise ValueError("development-validation partition is empty")
    contract = validate_validation_shard_contract(
        validation_shard_contract,
        expected_world_size=world_size,
    )
    if (
        contract["global_limit"] != global_limit
        or contract["development_partition_size"] != len(dataset)
    ):
        raise RuntimeError("validation-shard dataset extent differs")
    indices = tuple(contract["rank_to_global_positions"][str(rank)])
    subset = Subset(dataset, indices)
    subset.connect4_validation_shard = {
        "rank": rank,
        "world_size": world_size,
        "global_limit": global_limit,
        "global_indices": indices,
        "contract_record_sha256": contract["record_sha256"],
    }
    subset.connect4_validation_shard_contract = contract
    return subset


def build_selection_validation_loader(val_loader, *, world_size: int, rank: int):
    """Build the all-development rank shard without changing loader contracts."""

    routine_dataset = val_loader.dataset
    development_dataset = getattr(routine_dataset, "dataset", None)
    selection_contract = getattr(
        routine_dataset,
        "connect4_selection_validation_shard_contract",
        None,
    )
    if development_dataset is None:
        raise RuntimeError("routine validation loader has no development dataset")
    selection_contract = validate_validation_shard_contract(
        selection_contract,
        expected_world_size=world_size,
    )
    if (
        selection_contract["development_partition_size"]
        != len(development_dataset)
        or selection_contract["global_limit"] != len(development_dataset)
    ):
        raise RuntimeError("selection validation does not cover all development visits")
    rank_dataset = build_rank_local_validation_subset(
        development_dataset,
        limit=len(development_dataset),
        world_size=world_size,
        rank=rank,
        validation_shard_contract=selection_contract,
    )
    return DataLoader(
        rank_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=connect4_collate_fn,
        pin_memory=True,
    )


def _single_batch_value(batch: Mapping[str, Any], key: str) -> Any:
    if key not in batch:
        raise RuntimeError(f"development QA batch is missing {key!r}")
    value = batch[key]
    if torch.is_tensor(value):
        if value.shape[0] != 1:
            raise RuntimeError("development QA requires validation batch size one")
        value = value[0].detach().cpu()
        return value.item() if value.ndim == 0 else value
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError("development QA requires validation batch size one")
        return value[0]
    return value


def _development_source_identity(
    batch: Mapping[str, Any],
    *,
    scan_id: str,
    native_affine: np.ndarray,
) -> dict[str, Any]:
    """Bind QA to the admitted target/structural context, never a test target."""

    fields = (
        "brainlm_target_artifact_identity_sha256",
        "brainlm_scan_context_record_sha256",
        "brainlm_dataset_state_record_sha256",
        "brainlm_dataset_state_sha256",
        "brainlm_support_tensor_sha256",
        "brainlm_prepared_mask_sha256",
        "brainlm_padded_mask_artifact_descriptor_sha256",
        "brainlm_structural_source_identity_sha256",
        "brainlm_native_preprocessing_source_sha256",
        "brainlm_native_alignment_authority_sha256",
    )
    values = {key: str(_single_batch_value(batch, key)) for key in fields}
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise RuntimeError("development QA batch identity has an invalid SHA-256")
    role = str(_single_batch_value(batch, "scan_role"))
    brainlm_role = str(_single_batch_value(batch, "brainlm_role"))
    if role != "development-validation" or brainlm_role != role:
        raise RuntimeError(
            "development QA may open only admitted development-validation targets"
        )
    roi_mapping = batch.get("structure_to_roi_idx")
    if (
        not isinstance(roi_mapping, Mapping)
        or dict(roi_mapping) != dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    ):
        raise RuntimeError("development QA ROI mapping is not the canonical 32-ROI map")
    mapping_digests = batch.get("roi_mapping_sha256")
    label_orders = batch.get("roi_label_ids")
    if (
        mapping_digests != [CANONICAL_ROI_MAPPING_SHA256]
        or not isinstance(label_orders, list)
        or len(label_orders) != 1
        or tuple(label_orders[0]) != tuple(CANONICAL_ROI_LABEL_IDS)
    ):
        raise RuntimeError("development QA canonical ROI identity differs")
    affine = _single_batch_value(batch, "brainlm_padded_affine_ras_mm")
    if not torch.is_tensor(affine) or tuple(affine.shape) != (4, 4):
        raise RuntimeError("development QA batch has no signed padded affine")
    padded_affine = np.asarray(affine, dtype=np.float64)
    native_affine = np.asarray(native_affine, dtype=np.float64)
    if (
        not np.isfinite(padded_affine).all()
        or not np.isfinite(native_affine).all()
        or abs(float(np.linalg.det(padded_affine[:3, :3]))) <= 1e-12
        or abs(float(np.linalg.det(native_affine[:3, :3]))) <= 1e-12
    ):
        raise RuntimeError("development QA batch affine is invalid")

    def triplet(key: str, *, positive: bool) -> list[int]:
        raw = _single_batch_value(batch, key)
        if not torch.is_tensor(raw) or tuple(raw.shape) != (3,):
            raise RuntimeError(f"development QA batch lacks {key}")
        result = [int(item) for item in raw.tolist()]
        minimum = 1 if positive else 0
        if any(item < minimum for item in result):
            raise RuntimeError(f"development QA batch has invalid {key}")
        return result

    padded_shape = triplet("brainlm_padded_shape", positive=True)
    native_shape = triplet("brainlm_native_shape", positive=True)
    padding_before = triplet("brainlm_padding_before", positive=False)
    padding_after = triplet("brainlm_padding_after", positive=False)
    if padded_shape != [
        padding_before[index] + native_shape[index] + padding_after[index]
        for index in range(3)
    ]:
        raise RuntimeError("development QA batch crop does not reconstruct its grid")
    expected_native_affine = padded_affine.copy()
    expected_native_affine[:3, 3] += expected_native_affine[:3, :3] @ np.asarray(
        padding_before, dtype=np.float64
    )
    if not np.allclose(
        expected_native_affine, native_affine, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "development QA crop affine differs from the admitted padded affine"
        )
    record = {
        "schema": "connect4-development-source-identity-v1",
        "scan_id": scan_id,
        "scan_role": role,
        **values,
        "roi_mapping_sha256": CANONICAL_ROI_MAPPING_SHA256,
        "padded_shape": padded_shape,
        "native_shape": native_shape,
        "padding_before": padding_before,
        "padding_after": padding_after,
        "padded_affine_ras_mm": padded_affine.tolist(),
        "padded_affine_sha256": canonical_sha256(padded_affine.tolist()),
        "native_affine_ras_mm": native_affine.tolist(),
        "native_affine_sha256": canonical_sha256(native_affine.tolist()),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _crop_validation_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    roi_masks: torch.Tensor,
    brain_mask: torch.Tensor,
    target_validity_mask: torch.Tensor,
    batch: Mapping[str, Any],
    grid_contract: Mapping[str, Any] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """Return the exact native/anatomical crop and affine, without resampling."""

    arrays = (
        prediction,
        target,
        roi_masks,
        brain_mask,
        target_validity_mask,
    )
    if any(value is None or not torch.is_tensor(value) for value in arrays):
        raise RuntimeError(
            "development QA requires target, prediction, ROI, structural mask, "
            "and target-validity mask"
        )
    if grid_contract is not None:
        cropped = tuple(crop_architecture_array(value, grid_contract) for value in arrays)
        return (*cropped, anatomical_affine(grid_contract))

    def triplet(key: str, *, positive: bool) -> tuple[int, int, int]:
        raw = _single_batch_value(batch, key)
        if not torch.is_tensor(raw) or tuple(raw.shape) != (3,):
            raise RuntimeError(f"development QA native binding lacks {key}")
        result = tuple(int(value) for value in raw.tolist())
        minimum = 1 if positive else 0
        if any(value < minimum for value in result):
            raise RuntimeError(f"development QA native binding has invalid {key}")
        return result

    native = triplet("brainlm_native_shape", positive=True)
    padded = triplet("brainlm_padded_shape", positive=True)
    before = triplet("brainlm_padding_before", positive=False)
    after = triplet("brainlm_padding_after", positive=False)
    if tuple(before[i] + native[i] + after[i] for i in range(3)) != padded:
        raise RuntimeError("development QA native padding does not reconstruct grid")
    if any(tuple(value.shape[-3:]) != padded for value in arrays):
        raise RuntimeError("development QA tensor differs from signed padded grid")
    slices = tuple(slice(start, start + size) for start, size in zip(before, native))
    cropped = tuple(value[(..., *slices)] for value in arrays)
    if any(tuple(value.shape[-3:]) != native for value in cropped):
        raise RuntimeError("development QA inverse crop did not restore native grid")
    padded_affine = _single_batch_value(batch, "brainlm_padded_affine_ras_mm")
    if not torch.is_tensor(padded_affine) or tuple(padded_affine.shape) != (4, 4):
        raise RuntimeError("development QA native affine is missing")
    native_affine = np.asarray(padded_affine, dtype=np.float64).copy()
    native_affine[:3, 3] += native_affine[:3, :3] @ np.asarray(before)
    return (*cropped, native_affine)


@torch.no_grad()
def _validate_local_shard(model, val_loader, device, cfg, rank, world_size):
    model.eval()
    shard = getattr(val_loader.dataset, "connect4_validation_shard", None)
    contract_value = getattr(
        val_loader.dataset, "connect4_validation_shard_contract", None
    )
    if not isinstance(shard, dict):
        raise RuntimeError("validation loader is not the authenticated rank-local shard")
    try:
        contract = validate_validation_shard_contract(
            contract_value,
            expected_world_size=world_size,
        )
        expected_indices = tuple(
            contract["rank_to_global_positions"][str(rank)]
        )
        observed_indices = tuple(shard["global_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "validation loader is not the authenticated rank-local shard"
        ) from exc
    if (
        shard.get("rank") != rank
        or shard.get("world_size") != world_size
        or shard.get("global_limit") != contract["global_limit"]
        or shard.get("contract_record_sha256") != contract["record_sha256"]
        or observed_indices != expected_indices
        or len(val_loader.dataset) != len(observed_indices)
    ):
        raise RuntimeError("validation loader is not the authenticated rank-local shard")
    metrics = SynthesisMetricAccumulator()
    qa_rows: list[dict[str, Any]] = []
    first = None
    grid_contract = _dataset_common_grid(val_loader)
    policy = QualityPolicy(
        **dict(cfg["training"]["development_quality_gate"]["policy"])
    )
    try:
        for local_index, batch in enumerate(val_loader):
            if local_index >= len(observed_indices):
                raise RuntimeError("development QA opened an unassigned target")
            global_position = observed_indices[local_index]
            scan_id = str(_single_batch_value(batch, "scan_id"))
            _require_supervised_batch_contracts(
                batch,
                protocol_profile=str(cfg["data"]["protocol_profile"]),
            )
            batch = move(batch, device)
            amp_on = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
            use_bf16 = amp_on and torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                out = model(batch)
            if "target" not in out:
                raise RuntimeError("development QA model omitted its paired target")
            (
                prediction,
                target,
                roi_masks,
                brain_mask,
                target_validity_mask,
                native_affine,
            ) = (
                _crop_validation_sample(
                    out["prediction"],
                    out["target"],
                    out.get("supervision_roi_masks"),
                    out.get("brain_mask"),
                    out.get("target_validity_mask"),
                    batch,
                    grid_contract,
                )
            )
            source_identity = _development_source_identity(
                batch,
                scan_id=scan_id,
                native_affine=native_affine,
            )
            metrics.update(
                prediction,
                target,
                roi_masks=roi_masks,
                mask=target_validity_mask,
            )
            if (
                tuple(prediction.shape[:3])
                != (1, 1, int(cfg["data"]["num_frames"]))
                or target.shape != prediction.shape
                or roi_masks.ndim != 5
                or roi_masks.shape[0] != 1
                or tuple(brain_mask.shape[:2]) != (1, 1)
                or tuple(target_validity_mask.shape[:2]) != (1, 1)
            ):
                raise RuntimeError("development QA tensors have an invalid 4D shape")
            target_quality_tensor = target[0, 0].detach().float().cpu().contiguous()
            prediction_quality_tensor = (
                prediction[0, 0].detach().float().cpu().contiguous()
            )
            mask_quality_tensor = (
                brain_mask[0, 0].detach().float().cpu().contiguous()
            )
            target_validity_quality_tensor = (
                target_validity_mask[0, 0]
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
            roi_quality_tensor = roi_masks[0].detach().float().cpu().contiguous()
            real_array = target_quality_tensor.permute(1, 2, 3, 0).numpy()
            predicted_array = prediction_quality_tensor.permute(1, 2, 3, 0).numpy()
            mask_array = target_validity_quality_tensor.numpy()
            structural_mask_array = mask_quality_tensor.numpy()
            roi_array = roi_quality_tensor.numpy()
            target_identity = development_tensor_identity(
                target_quality_tensor, kind="target"
            )
            prediction_identity = development_tensor_identity(
                prediction_quality_tensor, kind="prediction"
            )
            brain_mask_identity = development_tensor_identity(
                mask_quality_tensor,
                kind="brain_mask",
                source_identity=source_identity,
            )
            target_validity_mask_identity = development_tensor_identity(
                target_validity_quality_tensor,
                kind="target_validity_mask",
                brain_mask=mask_quality_tensor,
                target=target_quality_tensor,
                target_validity_contract=str(
                    _single_batch_value(batch, "target_validity_mask_contract")
                ),
            )
            roi_masks_identity = development_tensor_identity(
                roi_quality_tensor,
                kind="roi_masks",
                brain_mask=mask_quality_tensor,
                source_identity=source_identity,
            )
            evaluated_tensors = evaluated_tensor_set_identity(
                target=target_identity,
                prediction=prediction_identity,
                brain_mask=brain_mask_identity,
                target_validity_mask=target_validity_mask_identity,
                roi_masks=roi_masks_identity,
            )
            try:
                quality = evaluate_4d_pair_quality_arrays(
                    real_array,
                    predicted_array,
                    mask_array,
                    affine=native_affine,
                    tr_seconds=float(cfg["data"]["tr_seconds"]),
                    roi_masks=roi_array,
                    roi_label_ids=CANONICAL_ROI_LABEL_IDS,
                    require_structured_temporal=True,
                    require_canonical_roi_mapping=True,
                    policy=policy,
                    source_identity=source_identity,
                    evaluated_tensor_set_identity=evaluated_tensors,
                    structural_mask=structural_mask_array,
                )
            except QualityContractError as exc:
                raise RuntimeError(
                    f"development QA contract failed for {scan_id}: {exc}"
                ) from exc
            required_passed = all(
                isinstance(quality["checks"].get(name), Mapping)
                and quality["checks"][name].get("passed") is True
                for name in DEVELOPMENT_REQUIRED_CHECKS
            )
            qa_rows.append(
                {
                    "global_position": global_position,
                    "scan_id": scan_id,
                    "scan_role": "development-validation",
                    "source_identity": source_identity,
                    "target_identity": target_identity,
                    "prediction_identity": prediction_identity,
                    "brain_mask_identity": brain_mask_identity,
                    "target_validity_mask_identity": (
                        target_validity_mask_identity
                    ),
                    "roi_masks_identity": roi_masks_identity,
                    "quality": quality,
                    "quality_sha256": canonical_sha256(quality),
                    "passed": bool(quality["passed"] and required_passed),
                }
            )
            if first is None and rank == 0:
                first = (
                    target.detach(),
                    prediction.detach(),
                    brain_mask.detach(),
                    target_validity_mask.detach(),
                )
    finally:
        model.train()
    if len(qa_rows) != len(observed_indices):
        raise RuntimeError("development QA did not evaluate every assigned target")
    return {
        "rank": rank,
        "num_samples": metrics.num_samples,
        "metric_sums": dict(metrics._sums),
        "qa_rows": qa_rows,
        "model_state_sha256": model_state_sha256(model),
    }, first


@torch.no_grad()
def validate(
    model,
    val_loader,
    device,
    cfg,
    rank,
    step,
    out_dir,
    *,
    tier: str,
    split_identity: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    cohort_admission_identity: Mapping[str, Any],
    candidate_next_epoch: int,
    candidate_next_batch_index: int,
):
    """Run deterministic rank-sharded DDIM QA on the selected dev extent."""

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    core = model.module if isinstance(model, DDP) else model
    failure: Exception | None = None
    local_payload = None
    first = None
    try:
        local_payload, first = _validate_local_shard(
            core, val_loader, device, cfg, rank, world_size
        )
    except Exception as exc:  # every surviving rank still joins failure exchange
        failure = exc
    status = {
        "rank": rank,
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
        "metrics": local_payload,
    }
    if distributed:
        statuses: list[dict | None] = [None for _ in range(world_size)]
        dist.all_gather_object(statuses, status)
    else:
        statuses = [status]
    if any(not isinstance(item, Mapping) for item in statuses):
        raise RuntimeError("distributed validation omitted a rank status")
    failures = [item["failure"] for item in statuses if item["failure"] is not None]
    if failures:
        raise RuntimeError(f"distributed validation failed: {failures}") from failure
    payloads = [item["metrics"] for item in statuses]
    if any(payload is None for payload in payloads):
        raise RuntimeError("distributed validation omitted a rank's metric payload")
    if sorted(payload["rank"] for payload in payloads) != list(range(world_size)):
        raise RuntimeError("distributed validation rank aggregation differs")
    total_samples = sum(payload["num_samples"] for payload in payloads)
    if total_samples < 1:
        raise RuntimeError("development QA evaluated no samples")
    sums: dict[str, float] = {}
    for payload in payloads:
        for key, value in payload["metric_sums"].items():
            sums[key] = sums.get(key, 0.0) + float(value)
    model_digests = {payload["model_state_sha256"] for payload in payloads}
    if len(model_digests) != 1:
        raise RuntimeError("DDP model states differ during development QA")
    model_digest = next(iter(model_digests))
    contract = validate_validation_shard_contract(
        getattr(
            val_loader.dataset,
            "connect4_validation_shard_contract",
            None,
        ),
        expected_cohort_admission_identity=cohort_admission_identity,
        expected_world_size=world_size,
    )
    if tier not in {ROUTINE_TIER, SELECTION_TIER}:
        raise ValueError("development QA tier is invalid")
    development_records = split_identity["partitions"]["validation"]
    expected_scan_ids = [
        str(record["scan_id"])
        for record in development_records[: contract["global_limit"]]
    ]
    rows = sorted(
        [row for payload in payloads for row in payload["qa_rows"]],
        key=lambda row: row["global_position"],
    )
    if (
        total_samples != contract["global_limit"]
        or [row["global_position"] for row in rows]
        != list(range(contract["global_limit"]))
        or [row["scan_id"] for row in rows] != expected_scan_ids
    ):
        raise RuntimeError(
            "development QA rank aggregation differs from the fixed split order"
        )
    selection = tier == SELECTION_TIER
    expected_limit = (
        len(development_records)
        if selection
        else min(int(cfg["training"]["val_batches"]), len(development_records))
    )
    if contract["global_limit"] != expected_limit:
        raise RuntimeError("development QA tier is conflated with the wrong extent")
    candidate = checkpoint_candidate_identity(
        model_state_digest=model_digest,
        completed_batches=int(step),
        next_epoch=int(candidate_next_epoch),
        next_batch_index=int(candidate_next_batch_index),
        partial=not selection,
    )
    policy = dict(cfg["training"]["development_quality_gate"]["policy"])
    prediction_identities = [dict(row["prediction_identity"]) for row in rows]
    passed = bool(all(row["passed"] for row in rows))
    qa_record = seal_development_quality_record(
        {
            "schema": "connect4-development-quality-attestation-v1",
            "tier": tier,
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "selectable": bool(selection and passed),
            "all_development_evaluated": selection,
            "policy_note": cfg["training"]["development_quality_gate"][
                "policy_note"
            ],
            "policy": policy,
            "policy_sha256": canonical_sha256(policy),
            "required_checks": list(DEVELOPMENT_REQUIRED_CHECKS),
            "step": int(step),
            "development_partition_size": len(development_records),
            "global_limit": contract["global_limit"],
            "ordered_global_positions": list(range(contract["global_limit"])),
            "ordered_scan_ids": expected_scan_ids,
            "ordered_scan_ids_sha256": canonical_sha256(expected_scan_ids),
            "per_scan_records": rows,
            "per_scan_records_sha256": canonical_sha256(rows),
            "prediction_set_sha256": canonical_sha256(prediction_identities),
            "model_state_sha256": model_digest,
            "checkpoint_candidate_identity": candidate,
            "checkpoint_candidate_identity_sha256": candidate["record_sha256"],
            "config_sha256": canonical_sha256(_checkpoint_config(cfg)),
            "cohort_admission_identity_record_sha256": (
                cohort_admission_identity["record_sha256"]
            ),
            "cohort_admission_data_root_sha256": cohort_admission_identity[
                "data_root_sha256"
            ],
            "runtime_identity_sha256": runtime_identity["canonical_sha256"],
            "artifact_identity_sha256": canonical_sha256(artifact_identity),
            "validation_shard_contract_sha256": contract["record_sha256"],
            "no_best_subject_selection": True,
            "sealed_test_targets_opened": False,
            "exact_native_crop_no_resampling": True,
        }
    )
    validate_development_quality_record(qa_record)
    qa_root = Path(out_dir).resolve() / "development_qa"
    record_path = qa_root / f"{tier}_step{int(step):012d}.json"

    def publish_record() -> str:
        qa_root.mkdir(parents=True, exist_ok=True)
        _, digest = publish_development_quality_record(
            record_path,
            qa_record,
        )
        return digest

    record_file_sha256 = _run_rank_zero_synchronized(
        publish_record,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        label="development QA atomic publication",
    )
    if not isinstance(record_file_sha256, str) or len(record_file_sha256) != 64:
        raise RuntimeError("development QA publication returned an invalid digest")

    def report_record() -> bool:
        aggregate = {key: value / total_samples for key, value in sums.items()}
        print(
            f"[val step {step}] "
            + "  ".join(f"{key}={value:.4f}" for key, value in aggregate.items()),
            flush=True,
        )
        if first is not None and cfg["training"].get("visualize", True):
            path = Path(out_dir) / f"val_step{step}_real_vs_synthetic.png"
            plot_real_vs_synthetic_4d(
                first[0],
                first[1],
                str(path),
                brain_mask=first[2],
                target_validity_mask=first[3],
                repetition_time=3.0,
            )
            print(f"[val step {step}] viz -> {path}", flush=True)
        print(
            f"[development QA {tier} step {step}] "
            f"status={qa_record['status']} samples={total_samples} -> {record_path}",
            flush=True,
        )
        return True

    _run_rank_zero_synchronized(
        report_record,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        label="development QA reporting",
    )
    return {
        "record": qa_record,
        "record_path": str(record_path),
        "record_file_sha256": record_file_sha256,
    }


@torch.no_grad()
def _dataset_common_grid(loader):
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return getattr(dataset, "common_grid_contract", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/connect4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    is_dist, rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    runtime_identity = require_production_training_runtime(
        cfg,
        is_distributed=is_dist,
        rank=rank,
        world_size=world,
        local_rank=local,
    )
    if is_main(rank):
        print(
            "[runtime] authenticated CIAI four-A100 identity "
            f"{runtime_identity['canonical_sha256']}",
            flush=True,
        )
    seed_process(int(cfg["training"].get("seed", 42)), rank)

    (
        train_loader,
        val_loader,
        train_sampler,
        split_identity,
        artifact_identity,
        admission_identity,
    ) = build_loaders(
        cfg,
        world,
        rank,
        runtime_identity=runtime_identity,
    )
    validation_shard_contract = validate_validation_shard_contract(
        getattr(val_loader.dataset, "connect4_validation_shard_contract", None),
        expected_cohort_admission_identity=admission_identity,
        expected_world_size=world,
    )
    selection_val_loader = build_selection_validation_loader(
        val_loader,
        world_size=world,
        rank=rank,
    )
    selection_validation_shard_contract = validate_validation_shard_contract(
        getattr(
            selection_val_loader.dataset,
            "connect4_validation_shard_contract",
            None,
        ),
        expected_cohort_admission_identity=admission_identity,
        expected_world_size=world,
    )

    model = Connect4Model(cfg).to(device)
    model.bind_training_authorities(
        artifact_identity=artifact_identity,
        cohort_admission_identity=admission_identity,
    )
    model.validate_model_state_invariants()
    if is_dist:
        model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=True)
    core = model.module if is_dist else model

    optimizer_name = str(cfg["training"].get("optimizer", "AdamW"))
    if optimizer_name.lower() != "adamw":
        raise ValueError(f"paper-faithful training requires AdamW, got {optimizer_name!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])

    epochs = cfg["training"].get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 200:
        raise ValueError("paper-faithful training.epochs must be an integer from 1 to 200")

    # Prefer bf16 on Ampere+ (A100): same exponent range as fp32, so the FFT /
    # attention / conv ops don't overflow -> no GradScaler and no skipped steps
    # (fp16 was overflowing, causing the recurring "non-finite grad" skips).
    amp_on = cfg["training"].get("amp", True)
    use_bf16 = amp_on and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_on and not use_bf16))
    if is_main(rank):
        print(f"[amp] dtype={'bf16' if use_bf16 else ('fp16' if amp_on else 'fp32')} "
              f"grad_scaler={'on' if (amp_on and not use_bf16) else 'off'}", flush=True)
    accum = cfg["training"].get("grad_accum_steps", 1)
    log_every = cfg["training"].get("log_every", 20)
    eval_every = cfg["training"].get("eval_every", 200)
    ckpt_every = cfg["training"].get("ckpt_every", 100)
    for name, interval in (
        ("log_every", log_every),
        ("eval_every", eval_every),
        ("ckpt_every", ckpt_every),
    ):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(f"training.{name} must be a positive integer")

    ckpt_dir = Path(cfg["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["training"].get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(
        path,
        next_epoch,
        next_batch_index,
        completed_batches,
        partial,
        *,
        latest_qa,
        selection_qa=None,
        selection_eligible=False,
        immutable_selection=False,
    ):
        local_rng = _rng_state()
        if is_dist:
            rng_states = [None for _ in range(world)]
            dist.all_gather_object(rng_states, local_rng)
        else:
            rng_states = [local_rng]

        def publish_checkpoint() -> bool:
            # A checkpoint can never publish a codec state that crossed the
            # V18 bounded-gauge/DC/tied-decoder and model-wide finite-state
            # boundary since the last optimizer update.
            core.validate_model_state_invariants()
            payload = {
                "format": TRAINING_CHECKPOINT_FORMAT,
                "architecture_contract": synthesis_architecture_contract(),
                "architecture_contract_sha256": (
                    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
                ),
                "production_token_codec_shape": (
                    core.production_token_codec_shape_evidence()
                ),
                "model": core.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": _checkpoint_config(cfg),
                "split_identity": split_identity,
                "artifact_identity": artifact_identity,
                "runtime_identity": runtime_identity,
                "cohort_admission_identity": admission_identity,
                "validation_shard_contract": validation_shard_contract,
                "selection_validation_shard_contract": (
                    selection_validation_shard_contract
                ),
                "world_size": world,
                "next_epoch": int(next_epoch),
                "next_batch_index": int(next_batch_index),
                "completed_batches": int(completed_batches),
                "partial": bool(partial),
                "latest_development_qa": (
                    None if latest_qa is None else latest_qa["record"]
                ),
                "latest_development_qa_file_sha256": (
                    None if latest_qa is None else latest_qa["record_file_sha256"]
                ),
                "selection_development_qa": (
                    None if selection_qa is None else selection_qa["record"]
                ),
                "selection_development_qa_file_sha256": (
                    None
                    if selection_qa is None
                    else selection_qa["record_file_sha256"]
                ),
                "selection_eligible": bool(selection_eligible),
                "rng_states": rng_states,
            }
            if selection_eligible:
                validate_selectable_checkpoint_development_qa(payload)
            elif partial:
                validate_progress_checkpoint_development_qa(payload)
            elif not partial:
                raise RuntimeError(
                    "a non-partial checkpoint requires passing all-development QA"
                )
            if immutable_selection:
                publish_selectable_checkpoint(payload, Path(path))
            else:
                _atomic_torch_save(payload, Path(path))
            return True

        _run_rank_zero_synchronized(
            publish_checkpoint,
            rank=rank,
            world_size=world,
            distributed=is_dist,
            label=f"checkpoint publication {Path(path).name}",
        )

    # Resume only from a complete, protocol-identical training state. The epoch
    # sampler order is deterministic, and the saved batch is the next one to run.
    start_epoch, resume_batch, step = 0, 0, 0
    latest_development_qa = None
    latest = ckpt_dir / "latest.pt"
    epoch_ckpts = sorted(ckpt_dir.glob("connect4_epoch*.pt"))
    resume_path = latest if latest.exists() else (epoch_ckpts[-1] if epoch_ckpts else None)
    if resume_path is not None:
        try:
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=True
            )
        except Exception as exc:
            raise RuntimeError(f"cannot read resume checkpoint {resume_path}: {exc}") from exc
        start_epoch, resume_batch, step = validate_resume_checkpoint(
            checkpoint,
            model=core,
            optimizer=opt,
            scaler=scaler,
            config=cfg,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=runtime_identity,
            cohort_admission_identity=admission_identity,
            validation_shard_contract=validation_shard_contract,
            world_size=world,
            rank=rank,
            batches_per_epoch=len(train_loader),
            grad_accum_steps=accum,
        )
        if checkpoint.get("latest_development_qa") is not None:
            latest_development_qa = {
                "record": validate_development_quality_record(
                    checkpoint["latest_development_qa"]
                ),
                "record_path": None,
                "record_file_sha256": checkpoint.get(
                    "latest_development_qa_file_sha256"
                ),
            }
        if start_epoch > epochs:
            raise RuntimeError(
                f"checkpoint resumes at epoch {start_epoch}, beyond configured epochs={epochs}"
            )
        if is_main(rank):
            print(
                f"[resume] {resume_path.name} -> epoch {start_epoch}, "
                f"next batch {resume_batch}, completed batches {step}",
                flush=True,
            )
        if start_epoch == epochs and checkpoint.get("selection_eligible") is True:
            if is_dist:
                dist.barrier()
                dist.destroy_process_group()
            if is_main(rank):
                print(
                    "[training complete] existing exact-state all-development "
                    "QA-passing checkpoint retained without replacement",
                    flush=True,
                )
            return

    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        core.set_training_sampling_epoch(epoch)
        train_loader.generator.manual_seed(
            int(cfg["training"].get("seed", 42)) + rank + epoch * 1_000_003
        )
        first_batch = resume_batch if epoch == start_epoch else 0
        for it, batch in enumerate(train_loader):
            if it < first_batch:
                continue
            _require_supervised_batch_contracts(
                batch,
                protocol_profile=str(cfg["data"]["protocol_profile"]),
            )
            batch = move(batch, device)
            with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                out = model(batch)
                loss = out["losses"]["total"] / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}, batch {it}")
            scaler.scale(loss).backward()
            optimizer_boundary = (it + 1) % accum == 0
            if optimizer_boundary:
                scaler.unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"].get("grad_clip", 1.0))
                if not torch.isfinite(gnorm):
                    raise RuntimeError(
                        f"non-finite gradient norm at epoch {epoch}, batch {it}"
                    )
                scaler.step(opt)
                core.validate_model_state_invariants()
                scaler.update()
                opt.zero_grad(set_to_none=True)

            step += 1
            if is_main(rank) and step % log_every == 0:
                terms = {k: round(float(v), 4) for k, v in out["losses"].items()}
                print(f"epoch {epoch} step {step} | {terms}", flush=True)
            next_batch = it + 1
            next_epoch = epoch
            if next_batch == len(train_loader):
                next_epoch, next_batch = epoch + 1, 0
            if optimizer_boundary and step % eval_every == 0:
                latest_development_qa = validate(
                    model,
                    val_loader,
                    device,
                    cfg,
                    rank,
                    step,
                    out_dir,
                    tier=ROUTINE_TIER,
                    split_identity=split_identity,
                    artifact_identity=artifact_identity,
                    runtime_identity=runtime_identity,
                    cohort_admission_identity=admission_identity,
                    candidate_next_epoch=next_epoch,
                    candidate_next_batch_index=next_batch,
                )
            if optimizer_boundary and step % ckpt_every == 0:
                save_ckpt(
                    ckpt_dir / "latest.pt",
                    next_epoch,
                    next_batch,
                    step,
                    partial=True,
                    latest_qa=latest_development_qa,
                )
                if is_main(rank):
                    print(f"[ckpt] latest saved at step {step}", flush=True)

        # Epoch/progress checkpoints remain resumable but explicitly
        # non-selectable. A failed routine QA never aborts learning.
        if epoch + 1 < epochs:
            save_ckpt(
                ckpt_dir / f"connect4_epoch{epoch:03d}.pt",
                epoch + 1,
                0,
                step,
                partial=True,
                latest_qa=latest_development_qa,
            )
        save_ckpt(
            ckpt_dir / "latest.pt",
            epoch + 1,
            0,
            step,
            partial=True,
            latest_qa=latest_development_qa,
        )
        if is_main(rank):
            print(f"[ckpt] saved non-selectable progress epoch {epoch}", flush=True)

    # A completed/selectable checkpoint is a separate immutable publication.
    # It evaluates the exact final model state over every authenticated
    # development visit; the routine first-N record can never authorize it.
    selection_qa = validate(
        model,
        selection_val_loader,
        device,
        cfg,
        rank,
        step,
        out_dir,
        tier=SELECTION_TIER,
        split_identity=split_identity,
        artifact_identity=artifact_identity,
        runtime_identity=runtime_identity,
        cohort_admission_identity=admission_identity,
        candidate_next_epoch=epochs,
        candidate_next_batch_index=0,
    )
    latest_development_qa = selection_qa
    if selection_qa["record"]["passed"] is not True:
        save_ckpt(
            ckpt_dir / "latest.pt",
            epochs,
            0,
            step,
            partial=True,
            latest_qa=latest_development_qa,
        )
    require_passing_final_development_qa(selection_qa)
    completed_path = ckpt_dir / f"connect4_epoch{epochs - 1:03d}.pt"
    save_ckpt(
        completed_path,
        epochs,
        0,
        step,
        partial=False,
        latest_qa=selection_qa,
        selection_qa=selection_qa,
        selection_eligible=True,
        immutable_selection=True,
    )
    save_ckpt(
        ckpt_dir / "latest.pt",
        epochs,
        0,
        step,
        partial=False,
        latest_qa=selection_qa,
        selection_qa=selection_qa,
        selection_eligible=True,
    )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    if is_main(rank):
        print(
            "[training complete] checkpoints written; sealed-test targets remained "
            "unopened. Run the separate inference/evaluation entry point only after "
            "checkpoint selection is frozen.",
            flush=True,
        )


if __name__ == "__main__":
    main()
