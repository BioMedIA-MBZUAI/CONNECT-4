#!/usr/bin/env python3
"""Exact four-rank DDP smoke required after the one-GPU V14 memory sweep."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from architecture_contract import (  # noqa: E402
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    require_production_token_codec_shape,
)
from data.cohort_admission import (  # noqa: E402
    broadcast_rank_zero_cohort_admission,
    cohort_admission_identity,
)
from data.collate import connect4_collate_fn  # noqa: E402
from data.dataset_precomputed import Connect4PrecomputedDataset  # noqa: E402
from models.connect4 import Connect4Model  # noqa: E402
from models.brainlm_context import configured_brainlm_identity  # noqa: E402
from scripts.benchmark_figure1_v10_gpu import (  # noqa: E402
    EXPECTED_PARTITION,
    EXPECTED_QOS,
    RUNTIME_AUTHORITY_RELEASE_STOP,
    _admission_runtime_identity,
    _benchmark_config,
    _gib,
    _move_batch,
    _real_training_batch_identity,
    _regular_canonical_file,
    _require_slurm_allocation_shape,
)
from training.train import _build_rank_zero_training_admission  # noqa: E402
from data.provenance import canonical_sha256  # noqa: E402
from utils.config import validate_figure1_recovery_gate_config  # noqa: E402
from utils.immutable_yaml import read_immutable_yaml_snapshot  # noqa: E402
from utils.figure1_gpu_gate import (  # noqa: E402
    DDP_SMOKE_SCHEMA,
    MINIMUM_SAFETY_MARGIN_GIB,
    PROFILE_SHAPES,
    atomic_write_json_no_replace,
    scheduler_allocation_sha256,
    seal_benchmark_record,
    sha256_file,
    source_tree_identity,
    validate_ddp_smoke_attestation,
    validate_sweep_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILE_SHAPES), required=True)
    parser.add_argument("--slab-size", type=int, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--memory-summary", type=Path, required=True)
    parser.add_argument("--memory-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    return completed.stdout.strip()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    if args.profile != "recovery":
        raise RuntimeError(
            "PAPER_PROFILE_BLOCKED_MISSING_REVIEWED_128_GRID_BRAINLM_AUTHORITY"
        )
    config_snapshot = read_immutable_yaml_snapshot(
        args.config, "DDP benchmark config"
    )
    source_root = args.source_root.expanduser()
    if source_root.resolve(strict=True) != REPOSITORY_ROOT:
        raise RuntimeError("DDP source-root differs from the executing stage")
    source_sha256, source_count = source_tree_identity(source_root)
    config = validate_figure1_recovery_gate_config(config_snapshot.document)
    summary_path = _regular_canonical_file(args.memory_summary, "memory summary")
    summary_digest = str(args.memory_summary_sha256).strip().lower()
    if sha256_file(summary_path) != summary_digest:
        raise RuntimeError("DDP memory summary digest differs")
    try:
        summary = validate_sweep_summary(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"DDP memory summary is invalid: {exc}") from exc
    profile = summary["profiles"][args.profile]
    selected_core = profile["largest_safe_core_size"]
    if profile["memory_gate_passed"] is not True or selected_core is None:
        raise RuntimeError("DDP profile has no memory-safe core")
    if args.slab_size is not None and args.slab_size != selected_core:
        raise RuntimeError("DDP core is not the memory sweep's largest safe core")
    args.slab_size = int(selected_core)
    model_config, model_config_sha256 = _benchmark_config(
        config, args.profile, args.slab_size
    )
    selected = next(
        record
        for record in profile["records"]
        if record["depth_slab_size"] == args.slab_size
    )
    expected = {
        "source_tree_sha256": source_sha256,
        "source_file_count": source_count,
        "architecture_contract_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "base_config_file_sha256": config_snapshot.sha256,
        "benchmark_model_config_sha256": model_config_sha256,
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise RuntimeError("DDP source/config differs from the selected memory record")
    brainlm_identity = configured_brainlm_identity(
        model_config["training"]["loss"]["perceptual_extractor"]
    )
    if brainlm_identity != selected.get("brainlm_identity"):
        raise RuntimeError("DDP configured BrainLM authority differs from memory gate")
    return model_config, config, selected, {
        "memory_sweep_summary_sha256": summary_digest,
        "source_file_count": source_count,
    }


def _execution_identity() -> dict:
    raise RuntimeError(RUNTIME_AUTHORITY_RELEASE_STOP)


def _unreleased_execution_identity() -> dict:
    """Reserved implementation; unreachable until a reviewed authority exists."""

    job_id = str(os.environ.get("SLURM_JOB_ID") or "")
    if not job_id.isdigit():
        raise RuntimeError("DDP smoke must run inside a numeric Slurm job")
    scontrol = _command(["/usr/bin/scontrol", "show", "job", job_id, "-o"])
    scheduler = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in scontrol.split()
        if "=" in token
    }
    if (
        scheduler.get("Partition") != EXPECTED_PARTITION
        or scheduler.get("QOS") != EXPECTED_QOS
    ):
        raise RuntimeError("DDP smoke ran under the wrong partition/QOS")
    _require_slurm_allocation_shape(
        scontrol,
        scheduler,
        expected_nodes=1,
        expected_gpus=4,
    )
    hosts = _command(
        [
            "/usr/bin/scontrol",
            "show",
            "hostnames",
            scheduler.get("NodeList", ""),
        ]
    ).splitlines()
    hosts = [host.split(".", 1)[0] for host in hosts if host]
    hostname = platform.node().split(".", 1)[0]
    if (
        len(hosts) != 1
        or hostname not in hosts
        or any("login" in host.lower() for host in hosts)
    ):
        raise RuntimeError("DDP smoke did not attest allocated compute hosts")
    python = _regular_canonical_file(Path(sys.executable), "Python executable")
    if "mica" in str(python).lower() or not sys.flags.isolated:
        raise RuntimeError("DDP smoke requires a non-mica isolated Python")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "")
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(visible_tokens) != 4 or len(set(visible_tokens)) != 4:
        raise RuntimeError("DDP smoke must expose four distinct allocated GPUs")
    return {
        "slurm_job_id": job_id,
        "slurm_partition": scheduler["Partition"],
        "slurm_qos": scheduler["QOS"],
        "allocated_gpu_count": 4,
        "slurm_num_nodes": 1,
        "allocated_hostnames": hosts,
        "hostname": hostname,
        "cuda_visible_devices": visible,
        "cuda_device_count": torch.cuda.device_count(),
        "scontrol_record_sha256": scheduler_allocation_sha256(
            slurm_job_id=job_id,
            slurm_partition=scheduler["Partition"],
            slurm_qos=scheduler["QOS"],
            allocated_hostnames=hosts,
            slurm_num_nodes=1,
            allocated_gpu_count=4,
        ),
        "python_executable": str(python),
        "python_executable_sha256": sha256_file(python),
        "isolated_python": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def main(args: argparse.Namespace) -> None:
    if args.slab_size is not None and args.slab_size < 1:
        raise ValueError("slab-size must be positive")
    config, admission_config, selected, input_identity = _read_inputs(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("four A100 GPUs with native bfloat16 are required")

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != 4 or local_rank not in range(4) or torch.cuda.device_count() != 4:
        raise RuntimeError("DDP smoke requires one torchrun world exposing four GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    if "A100" not in properties.name or not 39.0 <= _gib(properties.total_memory) <= 41.0:
        raise RuntimeError("each DDP rank requires an A100-40GB GPU")
    visible_tokens = [
        token.strip()
        for token in str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        if token.strip()
    ]
    if len(visible_tokens) != 4 or len(set(visible_tokens)) != 4:
        raise RuntimeError("DDP rank cannot map four distinct visible GPU identities")
    gpu_row = _command(
        [
            "/usr/bin/nvidia-smi",
            f"--id={visible_tokens[local_rank]}",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = [row for row in gpu_row.splitlines() if row.strip()]
    if len(gpu_rows) != 1:
        raise RuntimeError("each DDP rank must attest exactly one mapped GPU")
    gpu_uuid, gpu_name, gpu_memory_mib, gpu_driver = (
        value.strip() for value in gpu_rows[0].split(",")
    )
    if (
        not gpu_uuid.startswith("GPU-")
        or "A100" not in gpu_name
        or not 39_000 <= int(gpu_memory_mib) <= 42_000
    ):
        raise RuntimeError("mapped DDP GPU is not an A100-40GB device")

    execution_values = [_execution_identity() if rank == 0 else None]
    dist.broadcast_object_list(execution_values, src=0)
    execution = execution_values[0]
    if not isinstance(execution, dict):
        raise RuntimeError("DDP execution identity broadcast failed")
    admission_runtime = _admission_runtime_identity(
        admission_config,
        execution=execution,
        source_tree_sha256=selected["source_tree_sha256"],
        source_file_count=input_identity["source_file_count"],
        world_size=world,
    )
    admission = broadcast_rank_zero_cohort_admission(
        lambda: _build_rank_zero_training_admission(
            admission_config, world, admission_runtime
        ),
        rank=rank,
        world_size=world,
        expected_config_sha256=canonical_sha256(admission_config),
        expected_runtime_identity_sha256=admission_runtime["canonical_sha256"],
    )
    admission_identity = cohort_admission_identity(admission)
    train_indices = admission["partitions"]["train"]
    if not isinstance(train_indices, list) or len(train_indices) < world:
        raise RuntimeError("DDP recovery admission has fewer than four train scans")
    dataset = Connect4PrecomputedDataset.from_rank_zero_admission_state(
        admission["dataset_state"]
    )
    sample = dataset[int(train_indices[rank])]
    if (
        admission["dataset_state"]["target_scan_roles"].get(sample.get("scan_id"))
        != "train"
        or "fmri" not in sample
    ):
        raise RuntimeError("DDP rank did not receive an admitted training target")
    batch = connect4_collate_fn([sample])

    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    core = Connect4Model(config, build_loss=True).to(device).train()
    require_production_token_codec_shape(
        protocol_profile=admission_config["data"]["protocol_profile"],
        hidden_size=core.dit.hidden_size,
        patch_value_dim=core.dit.patch_value_dim,
    )
    core.bind_training_authorities(
        artifact_identity=admission["artifact_identity"],
        cohort_admission_identity=admission_identity,
    )
    core.validate_model_state_invariants()
    if core.decoder.depth_receptive_field_radius != 14:
        raise RuntimeError("DDP TC-UNet derived halo changed")
    scheduler_digest = hashlib.sha256(
        core.dit.scheduler.alphas_cumprod.cpu().numpy().tobytes()
    ).hexdigest()
    if scheduler_digest != selected["scheduler_alphas_cumprod_sha256"]:
        raise RuntimeError("DDP scheduler tensor differs from the memory gate")
    model = DDP(
        core,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        find_unused_parameters=True,
    )
    initial_state = _state_sha256(core)
    initial_states = [None for _ in range(world)]
    dist.all_gather_object(initial_states, initial_state)
    if len(set(initial_states)) != 1:
        raise RuntimeError("DDP ranks did not start from the same model state")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    torch.manual_seed(20260901 + rank)
    torch.cuda.manual_seed_all(20260901 + rank)
    batch = _move_batch(batch, device)
    brain_support = core._normalise_brain_mask(batch.get("brain_mask"), 1, device)
    if brain_support is None:
        raise RuntimeError("real DDP benchmark batch omitted structural brain support")
    preflight_context_identity = core.preflight_brainlm_batch_context_identity(
        batch, brain_support
    )
    training_batch_identity = _real_training_batch_identity(
        batch,
        admission=admission,
        admission_identity=admission_identity,
        brainlm_batch_context_identity=preflight_context_identity,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        output = model(batch)
        objective = output["losses"]["total"]
    if output.get("brainlm_batch_context_identity") != preflight_context_identity:
        raise RuntimeError(
            "DDP composite loss used a different BrainLM context than preflight"
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
        raise RuntimeError("DDP composite objective is incomplete/non-finite")
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started

    backward_started = time.perf_counter()
    objective.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gradient_parameters = {
        "fusion": next(core.fusion.parameters()),
        "dit_tied_token_codec": core.dit.token_codec_raw_detail_rows,
        "dit_epsilon_head": core.dit.epsilon_head.linear.weight,
        "projection": core.dit_to_unet.projection.weight,
        "decoder": core.decoder.output_head.weight,
    }
    if (
        not torch.isfinite(gradient_norm)
        or any(
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or not bool(torch.count_nonzero(parameter.grad).item())
            for parameter in gradient_parameters.values()
        )
    ):
        raise RuntimeError("DDP smoke missed a finite nonzero gradient group")
    optimizer.step()
    core.validate_model_state_invariants()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    step_seconds = time.perf_counter() - started
    peak_allocated = _gib(torch.cuda.max_memory_allocated(device))
    peak_reserved = _gib(torch.cuda.max_memory_reserved(device))
    if peak_reserved > _gib(properties.total_memory) - MINIMUM_SAFETY_MARGIN_GIB:
        raise RuntimeError("DDP rank violates the four-GiB A100 safety margin")

    post_state = _state_sha256(core)
    post_states = [None for _ in range(world)]
    dist.all_gather_object(post_states, post_state)
    if len(set(post_states)) != 1:
        raise RuntimeError("DDP optimizer step diverged across ranks")
    rank_record = {
        "rank": rank,
        "world_size": world,
        "local_batch_size": 1,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "cuda_visible_device": visible_tokens[local_rank],
        "torch_cuda_device_index": local_rank,
        "nvidia_driver_version": gpu_driver,
        "gpu_total_memory_gib": _gib(properties.total_memory),
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "step_seconds": step_seconds,
        "composite_loss_completed": True,
        "required_gradient_groups_finite": True,
        "required_gradient_groups_nonzero": True,
        "gradient_clip_included": True,
        "adamw_step_included": True,
        "post_step_model_state_sha256": post_state,
        "training_batch_identity": training_batch_identity,
    }
    rank_records = [None for _ in range(world)]
    dist.all_gather_object(rank_records, rank_record)

    if rank == 0:
        record = seal_benchmark_record(
            {
                "schema": DDP_SMOKE_SCHEMA,
                "gate_scope": (
                    "exact-full-connect4-four-rank-ddp-real-training-step"
                ),
                "passed": True,
                "profile": args.profile,
                "shape_bctdhw": [1, 1, 128, *PROFILE_SHAPES[args.profile]],
                "depth_slab_size": args.slab_size,
                "derived_depth_halo": 14,
                "source_file_count": input_identity["source_file_count"],
                "memory_sweep_summary_sha256": input_identity[
                    "memory_sweep_summary_sha256"
                ],
                "selected_memory_record_sha256": selected[
                    "canonical_record_sha256"
                ],
                **{
                    key: selected[key]
                    for key in (
                        "source_tree_sha256",
                        "architecture_contract_sha256",
                        "base_config_file_sha256",
                        "benchmark_model_config_sha256",
                        "brainlm_identity",
                        "brainlm_identity_sha256",
                        "scheduler_alphas_cumprod_sha256",
                        "loss_weights",
                        "amp",
                        "optimizer",
                        "scheduler",
                    )
                },
                "initial_model_state_sha256": initial_state,
                "batch_contract": {
                    "global_batch_size": 4,
                    "world_size": 4,
                    "local_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "ddp_wrapped": True,
                },
                "ddp_gradient_reduction_attested": True,
                "all_ranks_post_step_state_equal": True,
                "synchronized_post_step_model_state_sha256": post_state,
                "execution": execution_values[0],
                "rank_records": rank_records,
                "global_training_batch_identity_sha256": canonical_sha256(
                    [record["training_batch_identity"] for record in rank_records]
                ),
            }
        )
        validate_ddp_smoke_attestation(record)
        atomic_write_json_no_replace(args.output, record)
        print(json.dumps(record, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(_parser().parse_args())
