#!/usr/bin/env python3
"""Authenticated per-rank A100 gate for one full Figure-1 V14 train step.

One rank-zero-admitted, unsealed recovery training sample drives production
fusion, token-latent DiT, exact-halo TC-UNet, every composite loss (with
the official contextual BrainLM), gradient clipping, AdamW, and full target-
blind DDIM. This one-GPU memory gate does not claim four-rank DDP execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from architecture_contract import (  # noqa: E402
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    require_production_token_codec_shape,
)
from data.cohort_admission import (  # noqa: E402
    cohort_admission_identity,
    validate_cohort_admission,
    validate_cohort_admission_identity,
)
from data.collate import connect4_collate_fn  # noqa: E402
from data.dataset_precomputed import Connect4PrecomputedDataset  # noqa: E402
from models.connect4 import Connect4Model  # noqa: E402
from models.brainlm_context import (  # noqa: E402
    configured_brainlm_identity,
    validate_brainlm_batch_context_identity,
)
from models.dit4d_temporal import DDIMScheduler  # noqa: E402
from utils.config import validate_figure1_recovery_gate_config  # noqa: E402
from utils.immutable_yaml import read_immutable_yaml_snapshot  # noqa: E402
from utils.figure1_gpu_gate import (  # noqa: E402
    BENCHMARK_RECORD_SCHEMA,
    DERIVED_DEPTH_HALO,
    PAPER_PROFILE_BLOCK_STATUS,
    PROFILE_SHAPES,
    atomic_write_json_no_replace,
    benchmark_model_config_identity,
    canonical_sha256,
    scheduler_allocation_sha256,
    seal_benchmark_record,
    sha256_file,
    source_tree_identity,
    validate_benchmark_record,
)


RUNTIME_AUTHORITY_RELEASE_STOP = (
    "V14 GPU gates are blocked until a fresh immutable BrainLM428/CONNECT-4 "
    "runtime authority binds the exact Python path and bytes"
)
EXPECTED_PARTITION = "cscc-gpu-p"
EXPECTED_QOS = "cscc-gpu-qos"


def _require_slurm_allocation_shape(
    scontrol: str,
    scheduler: dict[str, str],
    *,
    expected_nodes: int,
    expected_gpus: int,
) -> None:
    """Reject scheduler records that do not prove the requested allocation.

    CIAI Slurm versions have emitted both ``gres/gpu=4`` and
    ``gres:gpu:a100:4`` spellings.  The complete raw record is separately
    hashed; this parser accepts those spelling variants without trusting an
    unverified caller-provided GPU count.
    """

    if scheduler.get("NumNodes") != str(expected_nodes):
        raise RuntimeError(
            f"benchmark requires exactly {expected_nodes} allocated node(s)"
        )
    gpu_counts = {
        int(match.group(1))
        for match in re.finditer(
            r"(?:gres[/:])?gpu(?::[A-Za-z0-9_.-]+)?[=:](\d+)(?!\d)",
            scontrol,
            flags=re.IGNORECASE,
        )
    }
    if gpu_counts != {expected_gpus}:
        raise RuntimeError(
            f"scheduler record does not prove exactly {expected_gpus} allocated GPU(s)"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_SHAPES), required=True)
    parser.add_argument("--slab-size", type=int, required=True)
    parser.add_argument("--safety-margin-gib", type=float, default=4.0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cohort-admission", type=Path, required=True)
    parser.add_argument("--cohort-admission-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _gib(value: int) -> float:
    return float(value) / 1024.0**3


def _regular_canonical_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError(f"{label} must be an absolute non-symlink path")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must be a single-link regular file")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"{label} path is not canonical")
    return path


def _command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    return completed.stdout.strip()


def _execution_identity(properties: torch.cuda.DeviceProperties) -> dict:
    raise RuntimeError(RUNTIME_AUTHORITY_RELEASE_STOP)


def _unreleased_execution_identity(
    properties: torch.cuda.DeviceProperties,
) -> dict:
    """Reserved implementation; unreachable until a reviewed authority exists."""

    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not job_id.isdigit():
        raise RuntimeError("benchmark must run inside a numeric Slurm job")
    scontrol = _command(["/usr/bin/scontrol", "show", "job", job_id, "-o"])
    scheduler = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in scontrol.split()
        if "=" in token
    }
    partition = scheduler.get("Partition")
    qos = scheduler.get("QOS")
    if partition != EXPECTED_PARTITION or qos != EXPECTED_QOS:
        raise RuntimeError(f"unexpected Slurm partition/QOS: {partition}/{qos}")
    _require_slurm_allocation_shape(
        scontrol,
        scheduler,
        expected_nodes=1,
        expected_gpus=1,
    )
    node_list = scheduler.get("NodeList", "")
    allocated_hosts = _command(
        ["/usr/bin/scontrol", "show", "hostnames", node_list]
    ).splitlines()
    hostname = platform.node().split(".", 1)[0]
    allocated_hosts = [value.split(".", 1)[0] for value in allocated_hosts if value]
    if (
        len(allocated_hosts) != 1
        or not hostname
        or "login" in hostname.lower()
        or hostname not in allocated_hosts
    ):
        raise RuntimeError("benchmark is not running on an allocated compute node")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(visible_tokens) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark must expose exactly one allocated CUDA device")
    gpu_rows = _command(
        [
            "/usr/bin/nvidia-smi",
            f"--id={visible_tokens[0]}",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(gpu_rows) != 1:
        raise RuntimeError("nvidia-smi must attest exactly one visible GPU")
    gpu_uuid, gpu_name, memory_mib, driver = (
        value.strip() for value in gpu_rows[0].split(",")
    )
    if (
        not gpu_uuid.startswith("GPU-")
        or "A100" not in gpu_name
        or not 39_000 <= int(memory_mib) <= 42_000
        or "A100" not in properties.name
        or not 39.0 <= _gib(properties.total_memory) <= 41.0
    ):
        raise RuntimeError("Figure-1 release gate requires one A100-40GB GPU")
    python = _regular_canonical_file(Path(sys.executable), "Python executable")
    if "mica" in str(python).lower():
        raise RuntimeError("the V14 benchmark cannot use the revoked mica runtime")
    return {
        "slurm_job_id": job_id,
        "slurm_partition": partition,
        "slurm_qos": qos,
        "slurm_node_list": node_list,
        "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
        "hostname": hostname,
        "allocated_hostnames": allocated_hosts,
        "slurm_num_nodes": 1,
        "allocated_gpu_count": 1,
        "cuda_visible_devices": visible,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_uuid": gpu_uuid,
        "nvidia_smi_name": gpu_name,
        "nvidia_smi_memory_mib": int(memory_mib),
        "nvidia_driver_version": driver,
        "python_executable": str(python),
        "python_executable_sha256": sha256_file(python),
        "python_version": platform.python_version(),
        "isolated_python": bool(sys.flags.isolated),
        "scontrol_record_sha256": scheduler_allocation_sha256(
            slurm_job_id=job_id,
            slurm_partition=partition,
            slurm_qos=qos,
            allocated_hostnames=allocated_hosts,
            slurm_num_nodes=1,
            allocated_gpu_count=1,
        ),
    }


def _benchmark_config(config: dict, profile: str, core: int) -> tuple[dict, str]:
    shape = PROFILE_SHAPES[profile]
    value = deepcopy(config)
    value["data"]["protocol_profile"] = "figure1-v10-a100-gpu-benchmark-only"
    value["data"]["architecture_shape"] = list(shape)
    value["models"]["dit"]["input_size"] = list(shape)
    value["models"]["unet"].update(
        {
            "depth_slab_size": core,
            "depth_slab_sweep_summary_path": None,
            "depth_slab_sweep_summary_sha256": None,
            "depth_slab_sweep_summary_sha256_env": "",
            "depth_slab_ddp_smoke_path": None,
            "depth_slab_ddp_smoke_sha256": None,
            "depth_slab_ddp_smoke_sha256_env": "",
        }
    )
    return value, benchmark_model_config_identity(value, profile)


def _admission_runtime_identity(
    config: dict,
    *,
    execution: dict,
    source_tree_sha256: str,
    source_file_count: int,
    world_size: int,
) -> dict:
    """Bind rank-zero cohort admission to this immutable gate execution."""

    value = {
        "schema": "connect4-figure1-v10-real-batch-gate-runtime-v1",
        "protocol_profile": config["data"]["protocol_profile"],
        "world_size": world_size,
        "config_sha256": canonical_sha256(config),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": source_file_count,
        "slurm_job_id": execution["slurm_job_id"],
        "slurm_partition": execution["slurm_partition"],
        "slurm_qos": execution["slurm_qos"],
        "allocated_hostnames": execution["allocated_hostnames"],
        "scontrol_record_sha256": execution["scontrol_record_sha256"],
        "python_executable": execution["python_executable"],
        "python_executable_sha256": execution["python_executable_sha256"],
        "isolated_python": execution["isolated_python"],
    }
    value["canonical_sha256"] = canonical_sha256(value)
    return value


def _admitted_training_batch(
    config: dict,
    *,
    execution: dict,
    source_tree_sha256: str,
    source_file_count: int,
    admission_path: Path,
    admission_sha256: str,
    config_file_sha256: str,
) -> tuple[dict, dict, dict]:
    """Materialize only the first fixed admitted recovery-training sample."""

    if config["data"].get("protocol_profile") != "a4-native-recovery-v1":
        raise RuntimeError("real-batch Figure-1 gate is recovery-only")
    runtime = _admission_runtime_identity(
        config,
        execution=execution,
        source_tree_sha256=source_tree_sha256,
        source_file_count=source_file_count,
        world_size=1,
    )
    path = _regular_canonical_file(admission_path, "gate cohort admission")
    if stat.S_IMODE(os.lstat(path).st_mode) & 0o222:
        raise RuntimeError("gate cohort admission must be immutable")
    expected_file_sha256 = str(admission_sha256).strip().lower()
    if sha256_file(path) != expected_file_sha256:
        raise RuntimeError("gate cohort-admission file digest differs")
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("gate cohort-admission file is unreadable") from exc
    required_wrapper_fields = {
        "schema",
        "status",
        "source_tree_sha256",
        "source_file_count",
        "config_sha256",
        "config_file_sha256",
        "runtime_identity",
        "cohort_admission",
        "cohort_admission_identity",
        "record_sha256",
    }
    if not isinstance(wrapper, dict) or set(wrapper) != required_wrapper_fields:
        raise RuntimeError("gate cohort-admission wrapper fields differ")
    claimed_wrapper_sha256 = wrapper.get("record_sha256")
    if (
        wrapper.get("schema") != "connect4-figure1-v10-gate-cohort-admission-v1"
        or wrapper.get("status") != "ADMITTED_REAL_RECOVERY_TRAINING_COHORT"
        or wrapper.get("source_tree_sha256") != source_tree_sha256
        or wrapper.get("source_file_count") != source_file_count
        or wrapper.get("config_sha256") != canonical_sha256(config)
        or wrapper.get("config_file_sha256") != config_file_sha256
        or wrapper.get("runtime_identity") != runtime
        or not isinstance(claimed_wrapper_sha256, str)
        or canonical_sha256(
            {
                key: value
                for key, value in wrapper.items()
                if key != "record_sha256"
            }
        )
        != claimed_wrapper_sha256
    ):
        raise RuntimeError("gate cohort-admission wrapper identity differs")
    admission = validate_cohort_admission(
        wrapper["cohort_admission"],
        expected_config_sha256=canonical_sha256(config),
        expected_world_size=1,
        expected_runtime_identity_sha256=runtime["canonical_sha256"],
    )
    identity = validate_cohort_admission_identity(
        wrapper["cohort_admission_identity"]
    )
    if identity != cohort_admission_identity(admission):
        raise RuntimeError("gate cohort-admission checkpoint identity differs")
    train_indices = admission["partitions"]["train"]
    if not isinstance(train_indices, list) or not train_indices:
        raise RuntimeError("recovery admission has no training sample")
    dataset = Connect4PrecomputedDataset.from_rank_zero_admission_state(
        admission["dataset_state"]
    )
    sample = dataset[int(train_indices[0])]
    if (
        sample.get("scan_id") not in admission["dataset_state"]["target_scan_roles"]
        or admission["dataset_state"]["target_scan_roles"][sample["scan_id"]]
        != "train"
        or "fmri" not in sample
    ):
        raise RuntimeError("gate sample is not an admitted unsealed training target")
    return connect4_collate_fn([sample]), admission, identity


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _real_training_batch_identity(
    batch: dict,
    *,
    admission: dict,
    admission_identity: dict,
    brainlm_batch_context_identity: dict,
) -> dict:
    scan_ids = [str(value) for value in batch.get("scan_id", [])]
    if len(scan_ids) != 1:
        raise RuntimeError("per-rank gate requires exactly one admitted training scan")
    roles = [
        str(admission["dataset_state"]["target_scan_roles"].get(scan_id, ""))
        for scan_id in scan_ids
    ]
    if roles != ["train"]:
        raise RuntimeError("per-rank gate target role differs from fixed train split")
    try:
        context_identity = validate_brainlm_batch_context_identity(
            brainlm_batch_context_identity,
            expected_brainlm_identity_sha256=admission["artifact_identity"][
                "perceptual_extractor"
            ]["record_sha256"],
            expected_dataset_state_sha256=admission_identity[
                "dataset_state_sha256"
            ],
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"model did not expose a valid signed BrainLM batch context: {exc}"
        ) from exc
    state = admission["dataset_state"]
    selected_artifacts = {
        scan_id: state["sample_artifacts"][scan_id] for scan_id in scan_ids
    }
    selected_targets = {
        scan_id: state["target_artifact_identities"][scan_id]
        for scan_id in scan_ids
    }
    target = batch.get("fmri")
    if not torch.is_tensor(target):
        raise RuntimeError("real training-batch identity requires an fMRI target")
    target_minimum = float(target.detach().amin().cpu().item())
    target_maximum = float(target.detach().amax().cpu().item())
    identity = {
        "schema": "connect4-figure1-v10-real-training-batch-v2",
        "protocol_profile": "a4-native-recovery-v1",
        "scan_ids": scan_ids,
        "roles": roles,
        "local_batch_size": 1,
        "cohort_admission_identity": admission_identity,
        "dataset_state_record_sha256": state["record_sha256"],
        "sample_artifacts_sha256": canonical_sha256(selected_artifacts),
        "target_artifact_identities_sha256": canonical_sha256(selected_targets),
        "target_tensor_identity": {
            "schema": "connect4-recovery-training-target-tensor-v1",
            "dtype": str(target.dtype),
            "shape_bctdhw": list(target.shape),
            "sha256": _tensor_sha256(target),
            "minimum": target_minimum,
            "maximum": target_maximum,
        },
        "brainlm_batch_context_identity": context_identity,
        "brainlm_context_source": (
            "exact-model-context-preflight; forward-equality-required-on-success"
        ),
        "admitted_unsealed_training_target_loaded": True,
        "synthetic_inputs_used": False,
        "sealed_target_voxel_data_opened": False,
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return identity


def _base_record(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    if args.slab_size < 1 or args.safety_margin_gib < 4.0:
        raise ValueError("slab-size must be positive and safety margin at least four GiB")
    if args.profile != "recovery":
        raise RuntimeError(PAPER_PROFILE_BLOCK_STATUS)
    config_snapshot = read_immutable_yaml_snapshot(args.config, "benchmark config")
    source_root = args.source_root.expanduser()
    if source_root.resolve(strict=True) != REPOSITORY_ROOT:
        raise RuntimeError("benchmark source-root differs from the executing stage")
    # Validate the complete paper/recovery schema but deliberately do not run
    # production gate authentication: this measurement creates that evidence.
    config = validate_figure1_recovery_gate_config(config_snapshot.document)
    model_config, model_config_sha256 = _benchmark_config(
        config, args.profile, args.slab_size
    )
    source_sha256, source_count = source_tree_identity(source_root)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A100 CUDA with native bfloat16 support is required")
    properties = torch.cuda.get_device_properties(0)
    execution = _execution_identity(properties)
    scheduler = DDIMScheduler(num_train_timesteps=1_000, schedule="linear")
    scheduler_digest = hashlib.sha256(
        scheduler.alphas_cumprod.cpu().numpy().tobytes()
    ).hexdigest()
    total_gib = _gib(properties.total_memory)
    brainlm_identity = configured_brainlm_identity(
        model_config["training"]["loss"]["perceptual_extractor"]
    )
    record = {
        "schema": BENCHMARK_RECORD_SCHEMA,
        "gate_scope": "exact-full-connect4-per-rank-real-training-step-memory",
        "passed": False,
        "profile": args.profile,
        "shape_bctdhw": [1, 1, 128, *PROFILE_SHAPES[args.profile]],
        "depth_slab_size": args.slab_size,
        "derived_depth_halo": DERIVED_DEPTH_HALO,
        "source_tree_sha256": source_sha256,
        "source_file_count": source_count,
        "architecture_contract_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "base_config_file_sha256": config_snapshot.sha256,
        "benchmark_model_config_sha256": model_config_sha256,
        "brainlm_identity": brainlm_identity,
        "brainlm_identity_sha256": brainlm_identity["record_sha256"],
        "training_batch_identity": None,
        "scheduler_alphas_cumprod_sha256": scheduler_digest,
        "composite_loss_configured": True,
        "all_roi_masks_configured": True,
        "composite_loss_completed": False,
        "four_gpu_ddp_smoke_required": True,
        "four_gpu_ddp_smoke_attested": False,
        "loss_weights": {
            "voxel": 1.0,
            "ssim": 0.5,
            "fc": 0.3,
            "temporal": 0.2,
            "perceptual": 0.1,
            "volume": 0.5,
            "region_hist": 0.5,
        },
        "amp": {
            "enabled": True,
            "dtype": "bfloat16",
            "gradient_scaler_enabled": False,
            "parameter_dtype": "float32",
        },
        "batch_contract": {
            "production_global_batch_size": 4,
            "executed_world_size": 1,
            "executed_local_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "ddp_wrapped": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "DDIM",
            "num_diffusion_steps": 1000,
            "beta_schedule": "linear",
            "num_inference_steps": 50,
            "eta": 0.0,
            "training_objective": "epsilon",
        },
        "execution": execution,
        "gpu": properties.name,
        "gpu_total_memory_gib": total_gib,
        "safety_margin_gib": args.safety_margin_gib,
        "safe_reserved_limit_gib": total_gib - args.safety_margin_gib,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "forward_seconds": None,
        "backward_seconds": None,
        "step_seconds": None,
        "adamw_step_included": False,
        "gradient_clip_included": False,
        "required_gradient_groups_finite": False,
        "required_gradient_groups_nonzero": False,
        "full_target_blind_inference_completed": False,
        "inference_autocast_dtype": "bfloat16",
        "inference_num_steps": 50,
        "inference_seconds": None,
        "inference_peak_reserved_gib": 0.0,
    }
    return record, model_config, config


def _publish(args: argparse.Namespace, record: dict) -> None:
    sealed = seal_benchmark_record(record)
    validate_benchmark_record(
        sealed, expected_profile=args.profile, expected_core=args.slab_size
    )
    atomic_write_json_no_replace(args.output, sealed)
    print(json.dumps(sealed, indent=2), flush=True)


def main(args: argparse.Namespace) -> None:
    record, config, admission_config = _base_record(args)
    device = torch.device("cuda", 0)
    exit_status = 0
    try:
        torch.manual_seed(20260901)
        torch.cuda.manual_seed_all(20260901)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        batch, admission, admission_identity = _admitted_training_batch(
            admission_config,
            execution=record["execution"],
            source_tree_sha256=record["source_tree_sha256"],
            source_file_count=record["source_file_count"],
            admission_path=args.cohort_admission,
            admission_sha256=args.cohort_admission_sha256,
            config_file_sha256=record["base_config_file_sha256"],
        )
        model = Connect4Model(config, build_loss=True).to(device).train()
        require_production_token_codec_shape(
            protocol_profile=admission_config["data"]["protocol_profile"],
            hidden_size=model.dit.hidden_size,
            patch_value_dim=model.dit.patch_value_dim,
        )
        model.bind_training_authorities(
            artifact_identity=admission["artifact_identity"],
            cohort_admission_identity=admission_identity,
        )
        model.validate_model_state_invariants()
        if model.decoder.depth_receptive_field_radius != DERIVED_DEPTH_HALO:
            raise RuntimeError("TC-UNet derived halo changed")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        batch = _move_batch(batch, device)
        brain_support = model._normalise_brain_mask(
            batch.get("brain_mask"), 1, device
        )
        if brain_support is None:
            raise RuntimeError("real benchmark batch omitted structural brain support")
        preflight_context_identity = (
            model.preflight_brainlm_batch_context_identity(batch, brain_support)
        )
        record["training_batch_identity"] = _real_training_batch_identity(
            batch,
            admission=admission,
            admission_identity=admission_identity,
            brainlm_batch_context_identity=preflight_context_identity,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(batch)
            objective = output["losses"]["total"]
        if output.get("brainlm_batch_context_identity") != preflight_context_identity:
            raise RuntimeError(
                "composite loss used a different BrainLM context than preflight"
            )
        required_losses = {
            "ssim", "voxel", "volume", "region_hist", "temporal",
            "perceptual", "fc", "diffusion", "token_codec_reconstruction", "total",
        }
        if (
            not torch.isfinite(objective)
            or not required_losses.issubset(output["losses"])
            or not all(torch.isfinite(output["losses"][key]) for key in required_losses)
        ):
            raise RuntimeError("full Connect4 composite objective is incomplete/non-finite")
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - started

        backward_started = time.perf_counter()
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gradient_parameters = {
            "fusion": next(model.fusion.parameters()),
            "dit_tied_token_codec": model.dit.token_codec_raw_detail_rows,
            "dit_epsilon_head": model.dit.epsilon_head.linear.weight,
            "projection": model.dit_to_unet.projection.weight,
            "decoder": model.decoder.output_head.weight,
        }
        if (
            not torch.isfinite(gradient_norm)
            or any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in gradient_parameters.values()
            )
            or any(
                not bool(torch.count_nonzero(parameter.grad).item())
                for parameter in gradient_parameters.values()
            )
        ):
            raise RuntimeError("full Connect4 step missed a finite nonzero gradient group")
        optimizer.step()
        model.validate_model_state_invariants()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        step_seconds = time.perf_counter() - started

        inference_started = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16
        ):
            sampled = model.eval().sampled_ddim_decode(
                batch,
                sampling_context="evaluation",
            )["prediction"]
        if (
            not torch.isfinite(sampled).all()
            or bool((sampled < 0).any())
            or bool((sampled > 1).any())
        ):
            raise RuntimeError("full target-blind DDIM prediction is outside [0,1]")
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        inference_peak_reserved = _gib(torch.cuda.max_memory_reserved(device))
        record.update(
            {
                "passed": True,
                "composite_loss_completed": True,
                "adamw_step_included": True,
                "gradient_clip_included": True,
                "required_gradient_groups_finite": True,
                "required_gradient_groups_nonzero": True,
                "full_target_blind_inference_completed": True,
                "gradient_norm": float(gradient_norm),
                "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "step_seconds": step_seconds,
                "inference_seconds": inference_seconds,
                "inference_peak_reserved_gib": inference_peak_reserved,
            }
        )
        if record["peak_reserved_gib"] > record["safe_reserved_limit_gib"]:
            record.update(
                {
                    "passed": False,
                    "failure_kind": "SAFETY_MARGIN",
                    "failure": (
                        f"peak reserved {record['peak_reserved_gib']:.3f} GiB exceeds "
                        f"safe limit {record['safe_reserved_limit_gib']:.3f} GiB"
                    ),
                }
            )
            exit_status = 2
    except torch.cuda.OutOfMemoryError as error:
        try:
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            pass
        record.update(
            {
                "failure_kind": "CUDA_OUT_OF_MEMORY",
                "failure": str(error) or "CUDA out of memory",
                "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
            }
        )
        exit_status = 3
    if record.get("passed"):
        model.validate_model_state_invariants()
    _publish(args, record)
    if exit_status:
        raise SystemExit(exit_status)


if __name__ == "__main__":
    main(_parser().parse_args())
