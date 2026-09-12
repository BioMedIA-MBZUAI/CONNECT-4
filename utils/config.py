"""Strict, shared configuration contract for paper-protocol entry points."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from architecture_contract import require_production_token_codec_shape
from eval.development_quality import DEVELOPMENT_QUALITY_CONFIG_SCHEMA
from eval.quality import (
    DEVELOPMENT_REQUIRED_CHECKS,
    POLICY_NOTE,
    QualityPolicy,
    require_release_quality_policy,
)
from utils.figure1_gpu_gate import authenticate_configured_depth_slab_selection


_LEAF = None
_EXTRACTOR_SCHEMA = {
    "enabled": _LEAF,
    "name": _LEAF,
    "checkpoint": _LEAF,
    "checkpoint_env": _LEAF,
    "checkpoint_sha256": _LEAF,
    "checkpoint_sha256_env": _LEAF,
    "source_revision": _LEAF,
    "source_revision_env": _LEAF,
    "adapter_contract": _LEAF,
    "source_root": _LEAF,
    "source_root_env": _LEAF,
    "source_publication_marker": _LEAF,
    "source_publication_marker_env": _LEAF,
    "source_publication_marker_sha256": _LEAF,
    "config": _LEAF,
    "config_env": _LEAF,
    "config_sha256": _LEAF,
    "config_sha256_env": _LEAF,
    "atlas": _LEAF,
    "atlas_env": _LEAF,
    "coordinates": _LEAF,
    "coordinates_env": _LEAF,
    "authority": _LEAF,
    "authority_env": _LEAF,
    "authority_sha256": _LEAF,
    "authority_sha256_env": _LEAF,
    "resample_chunk_frames": _LEAF,
}
_CONFIG_SCHEMA = {
    "data": {
        "protocol_profile": _LEAF,
        "preprocessing_evidence_status": _LEAF,
        "root_dir": _LEAF,
        "precomputed_dir": _LEAF,
        "dwi_matrix_path": _LEAF,
        "normative_csv_path": _LEAF,
        "cohort_manifest": _LEAF,
        "expected_cohort_scan_counts": _LEAF,
        "fmri_dir": _LEAF,
        "scaler_dir": _LEAF,
        "register_fmri_to_t1w": _LEAF,
        "require_paper_preprocessing": _LEAF,
        "common_grid_contract_path": _LEAF,
        "common_grid_contract_sha256": _LEAF,
        "common_grid_contract_sha256_env": _LEAF,
        "native_alignment_authority_sha256": _LEAF,
        "native_alignment_authority_sha256_env": _LEAF,
        "native_alignment_authority_path": _LEAF,
        "structural_stage_root": _LEAF,
        "native_selection_manifest_path": _LEAF,
        "native_selection_manifest_sha256": _LEAF,
        "native_selection_manifest_sha256_env": _LEAF,
        "native_selection_root_review_path": _LEAF,
        "native_selection_root_review_sha256": _LEAF,
        "native_selection_root_review_sha256_env": _LEAF,
        "native_completed_set_path": _LEAF,
        "native_completed_set_sha256": _LEAF,
        "native_completed_set_sha256_env": _LEAF,
        "native_completed_set_commit_marker_path": _LEAF,
        "native_completed_set_commit_marker_sha256": _LEAF,
        "native_completed_set_commit_marker_sha256_env": _LEAF,
        "native_reviewed_source_path": _LEAF,
        "native_reviewed_source_sha256": _LEAF,
        "native_reviewed_source_sha256_env": _LEAF,
        "native_runtime_attester_sha256": _LEAF,
        "native_runtime_attester_sha256_env": _LEAF,
        "native_verifier_sha256": _LEAF,
        "native_verifier_sha256_env": _LEAF,
        "architecture_shape": _LEAF,
        "allow_synthetic_grid_override": _LEAF,
        "num_frames": _LEAF,
        "voxel_size_mm": _LEAF,
        "tr_seconds": _LEAF,
        "out_channels": _LEAF,
        "normalize_intensity": _LEAF,
    },
    "models": {
        "brainiac_path": _LEAF,
        "brainiac_checkpoint_sha256": _LEAF,
        "brainiac_checkpoint_sha256_env": _LEAF,
        "brainiac_source_sha256": _LEAF,
        "brainiac_source_sha256_env": _LEAF,
        "modernbert_name": _LEAF,
        "modernbert_revision": _LEAF,
        "graphs": {"patch_size": _LEAF, "k_neighbors": _LEAF},
        "fusion": {
            "image_embed_dim": _LEAF,
            "mask_embed_dim": _LEAF,
            "roi_embed_dim": _LEAF,
            "hidden_dim": _LEAF,
            "output_dim": _LEAF,
            "num_attention_layers": _LEAF,
            "num_heads": _LEAF,
            "dropout": _LEAF,
            "num_rois": _LEAF,
        },
        "dit": {
            "diffusion_domain": _LEAF,
            "input_size": _LEAF,
            "patch_size": _LEAF,
            "temporal_patch_size": _LEAF,
            "hidden_size": _LEAF,
            "depth": _LEAF,
            "num_heads": _LEAF,
            "mlp_ratio": _LEAF,
            "num_diffusion_steps": _LEAF,
            "beta_schedule": _LEAF,
            "num_inference_steps": _LEAF,
            "eta": _LEAF,
            "spatial_window_size": _LEAF,
        },
        "unet": {
            "base_channels": _LEAF,
            "num_levels": _LEAF,
            "use_temporal_attn": _LEAF,
            "temporal_attn_heads": _LEAF,
            "use_checkpoint": _LEAF,
            "temporal_chunk": _LEAF,
            "depth_slab_size": _LEAF,
            "depth_slab_sweep_summary_path": _LEAF,
            "depth_slab_sweep_summary_sha256": _LEAF,
            "depth_slab_sweep_summary_sha256_env": _LEAF,
            "depth_slab_ddp_smoke_path": _LEAF,
            "depth_slab_ddp_smoke_sha256": _LEAF,
            "depth_slab_ddp_smoke_sha256_env": _LEAF,
            "detail_path_enabled": _LEAF,
        },
    },
    "training": {
        "batch_size": _LEAF,
        "num_workers": _LEAF,
        "optimizer": _LEAF,
        "learning_rate": _LEAF,
        "weight_decay": _LEAF,
        "epochs": _LEAF,
        "amp": _LEAF,
        "grad_accum_steps": _LEAF,
        "grad_clip": _LEAF,
        "diffusion_weight": _LEAF,
        "token_codec_reconstruction_weight": _LEAF,
        "ckpt_dir": _LEAF,
        "out_dir": _LEAF,
        "val_frac": _LEAF,
        "test_frac": _LEAF,
        "split_manifest": _LEAF,
        "split_manifest_sha256": _LEAF,
        "split_manifest_sha256_env": _LEAF,
        "seed": _LEAF,
        "log_every": _LEAF,
        "eval_every": _LEAF,
        "ckpt_every": _LEAF,
        "val_batches": _LEAF,
        "visualize": _LEAF,
        "development_quality_gate": {
            "schema": _LEAF,
            "enabled": _LEAF,
            "policy_note": _LEAF,
            "required_checks": _LEAF,
            "routine": {
                "extent": _LEAF,
                "num_samples": _LEAF,
            },
            "selection": {
                "extent": _LEAF,
                "num_samples": _LEAF,
                "final_only": _LEAF,
            },
            "policy": {key: _LEAF for key in asdict(QualityPolicy())},
        },
        "loss": {
            "voxel": _LEAF,
            "ssim": _LEAF,
            "fc": _LEAF,
            "temporal": _LEAF,
            "perceptual": _LEAF,
            "volume": _LEAF,
            "region_hist": _LEAF,
            "perceptual_extractor": _EXTRACTOR_SCHEMA,
        },
    },
    "evaluation": {
        "batch_size": _LEAF,
        "num_workers": _LEAF,
        "feature_extractor": _EXTRACTOR_SCHEMA,
        "quality_gates": {
            "paired_4d_required": _LEAF,
            "texture_retention_required": _LEAF,
            "texture_anti_gaming_required": _LEAF,
            "require_structured_temporal": _LEAF,
        },
    },
}

_PUBLISHED_LOSS_WEIGHTS = {
    "voxel": 1.0,
    "ssim": 0.5,
    "fc": 0.3,
    "temporal": 0.2,
    "perceptual": 0.1,
}
_PUBLISHED_COHORT_COUNTS = {"A4": 6290, "ADNI": 700}
PAPER_PROTOCOL_PROFILE = "paper-a4-adni-fmriprep-v1"
A4_RECOVERY_PROTOCOL_PROFILE = "a4-native-recovery-v1"
PAPER_PREPROCESSING_EVIDENCE_STATUS = "PAPER_CERTIFIED_FMRIPREP"
RECOVERY_PREPROCESSING_EVIDENCE_STATUS = "NON_CERTIFIED_RECOVERY_PREPROCESSING"
_NATIVE_RECOVERY_PATH_FIELDS = (
    "structural_stage_root",
    "native_alignment_authority_path",
    "native_selection_manifest_path",
    "native_selection_root_review_path",
    "native_completed_set_path",
    "native_completed_set_commit_marker_path",
    "native_reviewed_source_path",
)
_NATIVE_RECOVERY_SHA_PAIRS = (
    ("native_alignment_authority_sha256", "native_alignment_authority_sha256_env"),
    ("native_selection_manifest_sha256", "native_selection_manifest_sha256_env"),
    (
        "native_selection_root_review_sha256",
        "native_selection_root_review_sha256_env",
    ),
    ("native_completed_set_sha256", "native_completed_set_sha256_env"),
    (
        "native_completed_set_commit_marker_sha256",
        "native_completed_set_commit_marker_sha256_env",
    ),
    ("native_reviewed_source_sha256", "native_reviewed_source_sha256_env"),
    ("native_runtime_attester_sha256", "native_runtime_attester_sha256_env"),
    ("native_verifier_sha256", "native_verifier_sha256_env"),
)


def _reject_unknown_keys(value: Any, schema: Any, path: str = "config") -> None:
    if schema is _LEAF:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = sorted(set(value) - set(schema))
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")
    for key, child in value.items():
        _reject_unknown_keys(child, schema[key], f"{path}.{key}")


def _required(mapping: Mapping, key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required by the CONNECT-4 protocol")
    return mapping[key]


def _fixed_number(value: Any, expected: float, path: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{path} must equal {expected}")
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must equal {expected}") from exc
    if not math.isfinite(actual) or not math.isclose(
        actual, float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"paper-faithful configuration requires {path}={expected}")


def _positive_triplet(value: Any, path: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{path} must contain three positive integers")
    return tuple(int(item) for item in value)


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return int(value)


def _finite_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    include_minimum: bool = True,
    include_maximum: bool = True,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be a finite number")
    if minimum is not None and (
        numeric < minimum or (numeric == minimum and not include_minimum)
    ):
        comparator = ">=" if include_minimum else ">"
        raise ValueError(f"{path} must be {comparator} {minimum}")
    if maximum is not None and (
        numeric > maximum or (numeric == maximum and not include_maximum)
    ):
        comparator = "<=" if include_maximum else "<"
        raise ValueError(f"{path} must be {comparator} {maximum}")
    return numeric


def validate_paper_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the one production schema shared by preprocessing/train/infer."""
    if not isinstance(config, Mapping):
        raise TypeError("configuration root must be a mapping")
    _reject_unknown_keys(config, _CONFIG_SCHEMA)
    for section in _CONFIG_SCHEMA:
        _required(config, section, "config")

    data = config["data"]
    models = config["models"]
    training = config["training"]
    evaluation = config["evaluation"]
    for key in ("brainiac_path", "modernbert_name", "modernbert_revision"):
        if not str(_required(models, key, "config.models")).strip():
            raise ValueError(f"models.{key} must be non-empty")
    if str(models["modernbert_revision"]).strip() == (
        "<full immutable Hugging Face commit SHA>"
    ):
        raise ValueError(
            "models.modernbert_revision is an unresolved placeholder; provide "
            "the authenticated immutable Hugging Face commit SHA"
        )
    for direct_key, environment_key in (
        ("brainiac_checkpoint_sha256", "brainiac_checkpoint_sha256_env"),
        ("brainiac_source_sha256", "brainiac_source_sha256_env"),
    ):
        if (
            not models.get(direct_key)
            and not str(models.get(environment_key, "")).strip()
        ):
            raise ValueError(
                f"models.{direct_key} or models.{environment_key} is required"
            )
        direct_value = models.get(direct_key)
        if direct_value is not None:
            digest = str(direct_value).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"models.{direct_key} must be lowercase SHA-256")
    for key in (
        "protocol_profile",
        "preprocessing_evidence_status",
        "common_grid_contract_path",
        "architecture_shape",
        "allow_synthetic_grid_override",
        "num_frames",
        "voxel_size_mm",
        "tr_seconds",
        "out_channels",
        "require_paper_preprocessing",
        "expected_cohort_scan_counts",
    ):
        _required(data, key, "config.data")
    if data["allow_synthetic_grid_override"] is not False:
        raise ValueError(
            "data.allow_synthetic_grid_override must be false in production"
        )
    _fixed_number(data["num_frames"], 128, "data.num_frames")
    _fixed_number(data["voxel_size_mm"], 3.0, "data.voxel_size_mm")
    _fixed_number(data["tr_seconds"], 3.0, "data.tr_seconds")
    _fixed_number(data["out_channels"], 1, "data.out_channels")
    protocol_profile = str(data["protocol_profile"]).strip()
    preprocessing_status = str(data["preprocessing_evidence_status"]).strip()
    if data.get("register_fmri_to_t1w", False) is not False:
        raise ValueError("online registration is forbidden by the production protocol")
    if data.get("normalize_intensity", True) is not True:
        raise ValueError(
            "certified targets must retain the versioned unpublished intensity "
            "normalization recovery choice"
        )
    configured_counts = data["expected_cohort_scan_counts"]
    if not isinstance(configured_counts, Mapping):
        raise ValueError("data.expected_cohort_scan_counts must be a mapping")
    if protocol_profile == PAPER_PROTOCOL_PROFILE:
        if data["require_paper_preprocessing"] is not True:
            raise ValueError(
                "paper A4+ADNI profile requires require_paper_preprocessing=true"
            )
        if preprocessing_status != PAPER_PREPROCESSING_EVIDENCE_STATUS:
            raise ValueError(
                "paper A4+ADNI profile requires PAPER_CERTIFIED_FMRIPREP evidence"
            )
        if dict(configured_counts) != _PUBLISHED_COHORT_COUNTS:
            raise ValueError(
                "data.expected_cohort_scan_counts must equal the published "
                f"{_PUBLISHED_COHORT_COUNTS} for the paper profile"
            )
        if not str(data["common_grid_contract_path"]).strip():
            raise ValueError("paper profile common-grid contract path is required")
        if (
            not data.get("common_grid_contract_sha256")
            and not str(data.get("common_grid_contract_sha256_env", "")).strip()
        ):
            raise ValueError(
                "paper profile common-grid SHA-256 pin or environment key is required"
            )
        direct_grid_digest = data.get("common_grid_contract_sha256")
        if direct_grid_digest is not None:
            digest = str(direct_grid_digest).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "data.common_grid_contract_sha256 must be lowercase SHA-256"
                )
        if any(
            data.get(key) is not None for key in _NATIVE_RECOVERY_PATH_FIELDS
        ) or any(
            data.get(direct) is not None or str(data.get(environment, "")).strip()
            for direct, environment in _NATIVE_RECOVERY_SHA_PAIRS
        ):
            raise ValueError(
                "paper profile cannot use any non-certified native Stage-B field"
            )
    elif protocol_profile == A4_RECOVERY_PROTOCOL_PROFILE:
        if data["require_paper_preprocessing"] is not False:
            raise ValueError(
                "A4 recovery profile must set require_paper_preprocessing=false"
            )
        if preprocessing_status != RECOVERY_PREPROCESSING_EVIDENCE_STATUS:
            raise ValueError(
                "A4 recovery profile must declare NON_CERTIFIED_RECOVERY_PREPROCESSING"
            )
        if set(configured_counts) != {"A4"}:
            raise ValueError(
                "A4 recovery profile requires exactly one configured A4 count"
            )
        recovery_count = configured_counts["A4"]
        if (
            isinstance(recovery_count, bool)
            or not isinstance(recovery_count, int)
            or recovery_count < 1
        ):
            raise ValueError(
                "A4 recovery profile count must be a positive exact integer"
            )
        if (
            data.get("common_grid_contract_path") is not None
            or data.get("common_grid_contract_sha256") is not None
            or str(data.get("common_grid_contract_sha256_env", "")).strip()
        ):
            raise ValueError(
                "A4 recovery profile uses per-scan T1-aligned native evidence, "
                "not one cohort-common reference affine"
            )
        missing_paths = [
            key
            for key in _NATIVE_RECOVERY_PATH_FIELDS
            if not str(data.get(key, "")).strip()
        ]
        if missing_paths:
            raise ValueError(
                "A4 recovery profile requires all native Stage-B authority paths: "
                f"{missing_paths}"
            )
        for direct, environment in _NATIVE_RECOVERY_SHA_PAIRS:
            direct_value = data.get(direct)
            environment_name = str(data.get(environment, "")).strip()
            if direct_value is None and not environment_name:
                raise ValueError(
                    f"A4 recovery profile requires {direct} or {environment}"
                )
            if direct_value is not None:
                digest = str(direct_value).strip().lower()
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError(f"data.{direct} must be lowercase SHA-256")
        if not str(data.get("fmri_dir", "")).strip():
            raise ValueError("A4 recovery profile requires the Stage-B target root")
    else:
        raise ValueError(
            "data.protocol_profile must be the paper A4+ADNI profile or the "
            "authenticated A4 native-recovery profile"
        )

    target_shape = _positive_triplet(
        data["architecture_shape"], "data.architecture_shape"
    )
    graph_patch = _positive_triplet(
        _required(models["graphs"], "patch_size", "config.models.graphs"),
        "models.graphs.patch_size",
    )
    dit_input = _positive_triplet(
        _required(models["dit"], "input_size", "config.models.dit"),
        "models.dit.input_size",
    )
    dit_patch = _positive_triplet(
        _required(models["dit"], "patch_size", "config.models.dit"),
        "models.dit.patch_size",
    )
    spatial_window = _positive_triplet(
        _required(models["dit"], "spatial_window_size", "config.models.dit"),
        "models.dit.spatial_window_size",
    )
    if protocol_profile == PAPER_PROTOCOL_PROFILE:
        if target_shape != (128, 128, 128) or graph_patch != (16, 16, 16):
            raise ValueError(
                "paper profile requires the Figure-1 128x128x128 structural "
                "tensor and 16x16x16 structural patches"
            )
    elif target_shape != (64, 80, 64) or graph_patch != (16, 16, 16):
        raise ValueError(
            "A4 native-recovery profile requires the declared 64x80x64 "
            "zero-padded architecture grid and 16x16x16 patches"
        )
    if any(size % patch for size, patch in zip(target_shape, graph_patch)):
        raise ValueError(
            "data.architecture_shape must be divisible by models.graphs.patch_size"
        )
    if any(size % patch for size, patch in zip(dit_input, dit_patch)):
        raise ValueError(
            "models.dit.input_size must be divisible by models.dit.patch_size"
        )
    graph_grid = tuple(size // patch for size, patch in zip(target_shape, graph_patch))
    if models["dit"].get("diffusion_domain") != ("connect4-token-latent-diffusion-v1"):
        raise ValueError(
            "models.dit.diffusion_domain must be exactly "
            "'connect4-token-latent-diffusion-v1'; raw-direct voxel diffusion "
            "is forbidden by the v18 synthesis contract"
        )
    if dit_input != target_shape:
        raise ValueError(
            "Figure-1 v14 requires models.dit.input_size to equal the complete "
            "data.architecture_shape; low-resolution reconstruction is forbidden"
        )
    if dit_patch != graph_patch:
        raise ValueError(
            "Figure-1 v14 requires DiT patches to equal the aligned structural "
            "graph patches"
        )
    dit_grid = tuple(size // patch for size, patch in zip(dit_input, dit_patch))
    if spatial_window != (8, 8, 8):
        raise ValueError(
            "models.dit.spatial_window_size must be exactly [8,8,8] in v14; "
            "this closed implementation choice reconciles Figure-1 3D windows "
            "with the manuscript's global-interaction prose"
        )
    if any(window < grid for window, grid in zip(spatial_window, dit_grid)):
        raise ValueError(
            "the fixed v14 8x8x8 attention window must cover the complete patch grid"
        )
    graph_tokens = math.prod(graph_grid)
    k_neighbors = _positive_int(
        _required(models["graphs"], "k_neighbors", "config.models.graphs"),
        "models.graphs.k_neighbors",
    )
    if graph_tokens < 2 or k_neighbors >= graph_tokens:
        raise ValueError(
            "models.graphs.k_neighbors must be smaller than the number of patches"
        )

    fusion = models["fusion"]
    for key in (
        "image_embed_dim",
        "mask_embed_dim",
        "roi_embed_dim",
        "hidden_dim",
        "output_dim",
        "num_attention_layers",
        "num_heads",
    ):
        _positive_int(
            _required(fusion, key, "config.models.fusion"), f"models.fusion.{key}"
        )
    if fusion["hidden_dim"] % fusion["num_heads"]:
        raise ValueError("models.fusion.hidden_dim must be divisible by num_heads")
    _finite_number(
        _required(fusion, "dropout", "config.models.fusion"),
        "models.fusion.dropout",
        minimum=0.0,
        maximum=1.0,
        include_maximum=False,
    )

    dit = models["dit"]
    for key in (
        "hidden_size",
        "depth",
        "num_heads",
        "num_diffusion_steps",
        "num_inference_steps",
    ):
        _positive_int(_required(dit, key, "config.models.dit"), f"models.dit.{key}")
    if dit["hidden_size"] % dit["num_heads"]:
        raise ValueError("models.dit.hidden_size must be divisible by num_heads")
    _finite_number(
        _required(dit, "mlp_ratio", "config.models.dit"),
        "models.dit.mlp_ratio",
        minimum=0.0,
        include_minimum=False,
    )
    if dit["num_inference_steps"] > dit["num_diffusion_steps"]:
        raise ValueError(
            "models.dit.num_inference_steps cannot exceed num_diffusion_steps"
        )
    if dit.get("beta_schedule") not in {"linear", "scaled_linear"}:
        raise ValueError("models.dit.beta_schedule is unsupported")
    if dit["num_diffusion_steps"] < 2:
        raise ValueError("models.dit.num_diffusion_steps must be at least 2")
    _finite_number(
        _required(dit, "eta", "config.models.dit"),
        "models.dit.eta",
        minimum=0.0,
        maximum=1.0,
    )
    temporal_patch = _required(
        models["dit"], "temporal_patch_size", "config.models.dit"
    )
    if (
        isinstance(temporal_patch, bool)
        or not isinstance(temporal_patch, int)
        or temporal_patch < 1
        or 128 % temporal_patch
    ):
        raise ValueError("models.dit.temporal_patch_size must divide 128 frames")
    if protocol_profile in {PAPER_PROTOCOL_PROFILE, A4_RECOVERY_PROTOCOL_PROFILE} and (
        temporal_patch != 1
    ):
        raise ValueError(
            "v14 production models.dit.temporal_patch_size must be 1; the v9 "
            "16-frame/128x codec is forbidden"
        )
    require_production_token_codec_shape(
        protocol_profile=protocol_profile,
        hidden_size=dit["hidden_size"],
        patch_value_dim=(
            int(data["out_channels"]) * temporal_patch * math.prod(dit_patch)
        ),
    )
    _fixed_number(
        _required(fusion, "num_rois", "config.models.fusion"),
        32,
        "models.fusion.num_rois",
    )
    unet = models["unet"]
    if unet.get("use_temporal_attn") is not True:
        raise ValueError("models.unet.use_temporal_attn must be true")
    _fixed_number(unet.get("temporal_chunk", 0), 0, "models.unet.temporal_chunk")
    for key in ("base_channels", "num_levels", "temporal_attn_heads"):
        _positive_int(_required(unet, key, "config.models.unet"), f"models.unet.{key}")
    slab_core = _required(unet, "depth_slab_size", "config.models.unet")
    summary_path = _required(
        unet, "depth_slab_sweep_summary_path", "config.models.unet"
    )
    summary_digest = _required(
        unet, "depth_slab_sweep_summary_sha256", "config.models.unet"
    )
    summary_digest_env = _required(
        unet, "depth_slab_sweep_summary_sha256_env", "config.models.unet"
    )
    ddp_smoke_path = _required(unet, "depth_slab_ddp_smoke_path", "config.models.unet")
    ddp_smoke_digest = _required(
        unet, "depth_slab_ddp_smoke_sha256", "config.models.unet"
    )
    ddp_smoke_digest_env = _required(
        unet, "depth_slab_ddp_smoke_sha256_env", "config.models.unet"
    )
    if slab_core is None:
        if protocol_profile != PAPER_PROTOCOL_PROFILE:
            raise ValueError(
                "A4 recovery models.unet.depth_slab_size must be selected by "
                "an authenticated Figure-1 V14 GPU sweep"
            )
        if (
            summary_path is not None
            or summary_digest is not None
            or str(summary_digest_env).strip()
            or ddp_smoke_path is not None
            or ddp_smoke_digest is not None
            or str(ddp_smoke_digest_env).strip()
        ):
            raise ValueError(
                "unresolved paper depth_slab_size must not claim a sweep binding"
            )
    else:
        slab_core = _positive_int(slab_core, "models.unet.depth_slab_size")
        if slab_core > target_shape[0]:
            raise ValueError("models.unet.depth_slab_size cannot exceed grid depth")
        if not str(summary_path or "").strip():
            raise ValueError(
                "a selected depth_slab_size requires depth_slab_sweep_summary_path"
            )
        if summary_digest is None and not str(summary_digest_env).strip():
            raise ValueError(
                "a selected depth_slab_size requires its sweep-summary SHA-256 pin"
            )
        if summary_digest is not None:
            digest = str(summary_digest).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "models.unet.depth_slab_sweep_summary_sha256 must be lowercase SHA-256"
                )
        if not str(ddp_smoke_path or "").strip():
            raise ValueError(
                "a selected depth_slab_size requires depth_slab_ddp_smoke_path"
            )
        if ddp_smoke_digest is None and not str(ddp_smoke_digest_env).strip():
            raise ValueError(
                "a selected depth_slab_size requires its four-GPU DDP SHA-256 pin"
            )
        if ddp_smoke_digest is not None:
            digest = str(ddp_smoke_digest).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "models.unet.depth_slab_ddp_smoke_sha256 must be lowercase SHA-256"
                )
    _fixed_number(unet["base_channels"], 64, "models.unet.base_channels")
    _fixed_number(unet["num_levels"], 4, "models.unet.num_levels")
    if any(value % 8 for value in target_shape[1:]):
        raise ValueError("Figure-1 TC-UNet requires H,W divisible by eight")
    if unet["base_channels"] % unet["temporal_attn_heads"]:
        raise ValueError(
            "models.unet.base_channels must be divisible by temporal_attn_heads"
        )
    if not isinstance(unet.get("use_checkpoint"), bool):
        raise ValueError("models.unet.use_checkpoint must be boolean")
    detail_path_enabled = _required(unet, "detail_path_enabled", "config.models.unet")
    if not isinstance(detail_path_enabled, bool):
        raise ValueError("models.unet.detail_path_enabled must be boolean")
    if detail_path_enabled:
        raise ValueError(
            "models.unet.detail_path_enabled must be false: the v18 synthesis "
            "contract forbids copying T1 texture into predicted fMRI"
        )

    if str(_required(training, "optimizer", "config.training")).lower() != "adamw":
        raise ValueError("training.optimizer must be AdamW")
    if not str(_required(training, "split_manifest", "config.training")).strip():
        raise ValueError(
            "training.split_manifest must name an existing offline authority"
        )
    split_digest = _required(training, "split_manifest_sha256", "config.training")
    split_digest_env = _required(
        training, "split_manifest_sha256_env", "config.training"
    )
    if split_digest is None and not str(split_digest_env).strip():
        raise ValueError(
            "training split authority requires split_manifest_sha256 or its environment key"
        )
    if split_digest is not None:
        normalized_split_digest = str(split_digest).strip().lower()
        if len(normalized_split_digest) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_split_digest
        ):
            raise ValueError("training.split_manifest_sha256 must be lowercase SHA-256")
    _finite_number(
        _required(training, "learning_rate", "config.training"),
        "training.learning_rate",
        minimum=0.0,
        include_minimum=False,
    )
    _finite_number(
        _required(training, "weight_decay", "config.training"),
        "training.weight_decay",
        minimum=0.0,
    )
    _finite_number(
        _required(training, "diffusion_weight", "config.training"),
        "training.diffusion_weight",
        minimum=0.0,
        include_minimum=False,
    )
    _fixed_number(
        _required(
            training,
            "token_codec_reconstruction_weight",
            "config.training",
        ),
        1.0,
        "training.token_codec_reconstruction_weight",
    )
    _finite_number(
        _required(training, "grad_clip", "config.training"),
        "training.grad_clip",
        minimum=0.0,
        include_minimum=False,
    )
    _nonnegative_int(
        _required(training, "num_workers", "config.training"),
        "training.num_workers",
    )
    epochs = _required(training, "epochs", "config.training")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epochs <= 200
    ):
        raise ValueError("training.epochs must be an integer from 1 to 200")
    batch_size = _required(training, "batch_size", "config.training")
    accumulation = training.get("grad_accum_steps", 1)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or batch_size < 1
        or accumulation < 1
        or batch_size * accumulation != 4
    ):
        raise ValueError("training batch_size * grad_accum_steps must equal 4")
    for key in ("log_every", "eval_every", "ckpt_every", "val_batches"):
        _positive_int(
            _required(training, key, "config.training"),
            f"training.{key}",
        )
    if not isinstance(_required(training, "visualize", "config.training"), bool):
        raise ValueError("training.visualize must be boolean")
    gate = _required(
        training,
        "development_quality_gate",
        "config.training",
    )
    for key in (
        "schema",
        "enabled",
        "policy_note",
        "required_checks",
        "routine",
        "selection",
        "policy",
    ):
        _required(gate, key, "config.training.development_quality_gate")
    routine = gate["routine"]
    selection = gate["selection"]
    if (
        gate["schema"] != DEVELOPMENT_QUALITY_CONFIG_SCHEMA
        or gate["enabled"] is not True
        or gate["policy_note"] != POLICY_NOTE
        or gate["required_checks"] != list(DEVELOPMENT_REQUIRED_CHECKS)
        or routine.get("extent") != "fixed-authenticated-first-n"
        or routine.get("num_samples") != training["val_batches"]
        or selection.get("extent") != "all-authenticated-development"
        or selection.get("num_samples") != "all"
        or selection.get("final_only") is not True
    ):
        raise ValueError(
            "training.development_quality_gate must keep routine first-N and "
            "final all-development selection extents distinct and mandatory"
        )
    configured_policy = gate["policy"]
    expected_policy_fields = set(asdict(QualityPolicy()))
    if not isinstance(configured_policy, Mapping) or set(configured_policy) != (
        expected_policy_fields
    ):
        raise ValueError(
            "training.development_quality_gate.policy must explicitly configure "
            "every QualityPolicy threshold"
        )
    try:
        require_release_quality_policy(configured_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.development_quality_gate.policy is invalid or weaker "
            "than the frozen pre-seal baseline"
        ) from exc
    loss = _required(training, "loss", "config.training")
    for name, expected in _PUBLISHED_LOSS_WEIGHTS.items():
        _fixed_number(
            _required(loss, name, "config.training.loss"),
            expected,
            f"training.loss.{name}",
        )
    for name in ("volume", "region_hist"):
        _finite_number(
            _required(loss, name, "config.training.loss"),
            f"training.loss.{name}",
            minimum=0.0,
        )
    perceptual = _required(loss, "perceptual_extractor", "config.training.loss")
    if (
        perceptual.get("enabled") is not True
        or str(perceptual.get("name", "")).lower() != "brainlm"
        or perceptual.get("adapter_contract")
        != "connect4-brainlm-a424-contextual-perceptual-v2"
        or perceptual.get("source_revision")
        != "eded39c86c27e03f5ead1d6a14311e92d1305e5e"
        or (
            not perceptual.get("source_root")
            and perceptual.get("source_root_env") != "CONNECT4_BRAINLM_SOURCE_ROOT"
        )
        or (
            not perceptual.get("source_publication_marker")
            and perceptual.get("source_publication_marker_env")
            != "CONNECT4_BRAINLM_PUBLICATION_MARKER"
        )
        or perceptual.get("checkpoint_sha256")
        != "e647c70b2af023d945bfd61b4187c4fdf1700fbb4433a0818e982520564493c5"
        or perceptual.get("config_sha256")
        != "03142b047b43e174a1f2d9f64c43396916740abf014f9a404e33487575714091"
        or perceptual.get("source_publication_marker_sha256")
        != "af79bf37838aeb2f67005deb6750e3ad62eeb8fe65ff5190c0b304b4fdc23912"
    ):
        raise ValueError(
            "training perceptual extractor must be the pinned official GitHub "
            "BrainLM checkpoint with contextual A424/MNI adapter"
        )
    feature = _required(evaluation, "feature_extractor", "config.evaluation")
    if (
        feature.get("enabled") is not True
        or str(feature.get("name", "")).lower() != "slimbrain"
    ):
        raise ValueError("evaluation feature extractor must be enabled SLIM-Brain")
    _positive_int(
        _required(evaluation, "batch_size", "config.evaluation"),
        "evaluation.batch_size",
    )
    _nonnegative_int(
        _required(evaluation, "num_workers", "config.evaluation"),
        "evaluation.num_workers",
    )
    gates = _required(evaluation, "quality_gates", "config.evaluation")
    for key in (
        "paired_4d_required",
        "texture_retention_required",
        "texture_anti_gaming_required",
        "require_structured_temporal",
    ):
        if gates.get(key) is not True:
            raise ValueError(f"evaluation.quality_gates.{key} must be true")
    return dict(config)


def _require_gate_absolute_path(value: Any, path: str) -> None:
    text = str(value or "").strip()
    # G0 inputs must describe their authority without consulting the launcher
    # environment.  In particular, ``Path.expanduser`` would make a literal
    # ``~`` depend on HOME and could bind different artifacts on different
    # ranks or nodes.
    if not text or "~" in text or not Path(text).is_absolute():
        raise ValueError(f"{path} must be a direct absolute path for the G0 gate")


def _require_gate_direct_sha256(
    mapping: Mapping[str, Any],
    direct_key: str,
    environment_key: str,
    *,
    prefix: str,
) -> None:
    digest = mapping.get(direct_key)
    environment_name = str(mapping.get(environment_key, "")).strip()
    if environment_name:
        raise ValueError(
            f"{prefix}.{environment_key} must be empty for the immutable G0 gate"
        )
    normalized = str(digest or "").strip()
    if (
        len(normalized) != 64
        or normalized != normalized.lower()
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(
            f"{prefix}.{direct_key} must be a direct lowercase SHA-256 for the G0 gate"
        )


def validate_figure1_recovery_gate_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the pre-selection recovery gate without circular slab evidence.

    G0 is the only configuration state allowed to execute the one-GPU sweep,
    its admission builder, or the four-GPU smoke.  It keeps the slab selection
    and both pieces of evidence unresolved, while requiring every input
    authority to be an absolute, directly digest-pinned value.  The ordinary
    production validator remains unchanged and continues to reject this state.
    """
    if not isinstance(config, Mapping):
        raise TypeError("configuration root must be a mapping")
    try:
        data = config["data"]
        models = config["models"]
        training = config["training"]
        unet = models["unet"]
        perceptual = training["loss"]["perceptual_extractor"]
    except (KeyError, TypeError) as exc:
        raise ValueError("G0 gate configuration is missing a required section") from exc

    unresolved = {
        "depth_slab_size": None,
        "depth_slab_sweep_summary_path": None,
        "depth_slab_sweep_summary_sha256": None,
        "depth_slab_sweep_summary_sha256_env": "",
        "depth_slab_ddp_smoke_path": None,
        "depth_slab_ddp_smoke_sha256": None,
        "depth_slab_ddp_smoke_sha256_env": "",
    }
    for key, expected in unresolved.items():
        if unet.get(key) != expected:
            raise ValueError(
                f"models.unet.{key} must be {expected!r} in the unresolved G0 gate"
            )

    # Reuse the complete production schema and all non-slab protocol checks by
    # validating a private resolved surrogate.  The caller's G0 mapping is never
    # mutated and can therefore be sealed into pre-selection evidence as-is.
    surrogate = deepcopy(config)
    surrogate_unet = surrogate["models"]["unet"]
    surrogate_unet.update(
        {
            "depth_slab_size": 1,
            "depth_slab_sweep_summary_path": "/__connect4_g0__/sweep.json",
            "depth_slab_sweep_summary_sha256": "0" * 64,
            "depth_slab_sweep_summary_sha256_env": "",
            "depth_slab_ddp_smoke_path": "/__connect4_g0__/ddp.json",
            "depth_slab_ddp_smoke_sha256": "1" * 64,
            "depth_slab_ddp_smoke_sha256_env": "",
        }
    )
    validate_paper_config(surrogate)

    if data.get("protocol_profile") != A4_RECOVERY_PROTOCOL_PROFILE:
        raise ValueError("Figure-1 G0 gate is restricted to A4 native recovery")
    if data.get("expected_cohort_scan_counts") != {"A4": 2155}:
        raise ValueError(
            "Figure-1 G0 gate requires exactly 2,155 authenticated A4 scans"
        )

    for key in (
        "root_dir",
        "precomputed_dir",
        "dwi_matrix_path",
        "normative_csv_path",
        "cohort_manifest",
        "fmri_dir",
        "scaler_dir",
        *_NATIVE_RECOVERY_PATH_FIELDS,
    ):
        _require_gate_absolute_path(data.get(key), f"data.{key}")
    for direct_key, environment_key in _NATIVE_RECOVERY_SHA_PAIRS:
        _require_gate_direct_sha256(
            data,
            direct_key,
            environment_key,
            prefix="data",
        )

    _require_gate_absolute_path(models.get("brainiac_path"), "models.brainiac_path")
    for direct_key, environment_key in (
        ("brainiac_checkpoint_sha256", "brainiac_checkpoint_sha256_env"),
        ("brainiac_source_sha256", "brainiac_source_sha256_env"),
    ):
        _require_gate_direct_sha256(
            models,
            direct_key,
            environment_key,
            prefix="models",
        )

    _require_gate_absolute_path(
        training.get("split_manifest"), "training.split_manifest"
    )
    _require_gate_direct_sha256(
        training,
        "split_manifest_sha256",
        "split_manifest_sha256_env",
        prefix="training",
    )

    for direct_key, environment_key in (
        ("source_root", "source_root_env"),
        ("source_publication_marker", "source_publication_marker_env"),
        ("authority", "authority_env"),
    ):
        if str(perceptual.get(environment_key, "")).strip():
            raise ValueError(
                "training.loss.perceptual_extractor."
                f"{environment_key} must be empty for the immutable G0 gate"
            )
        _require_gate_absolute_path(
            perceptual.get(direct_key),
            f"training.loss.perceptual_extractor.{direct_key}",
        )
    # These four assets may be omitted because the authenticated official
    # source root resolves their fixed reviewed paths.  If a gate config does
    # override one explicitly, the override must still be a direct absolute
    # path; relative overrides would otherwise depend on the launcher's CWD.
    for direct_key in ("checkpoint", "config", "atlas", "coordinates"):
        if perceptual.get(direct_key) is not None:
            _require_gate_absolute_path(
                perceptual.get(direct_key),
                f"training.loss.perceptual_extractor.{direct_key}",
            )
    for environment_key in sorted(
        key for key in _EXTRACTOR_SCHEMA if key.endswith("_env")
    ):
        if str(perceptual.get(environment_key, "")).strip():
            raise ValueError(
                "training.loss.perceptual_extractor."
                f"{environment_key} must be empty for the immutable G0 gate"
            )
    for direct_key, environment_key in (
        ("source_publication_marker_sha256", "source_publication_marker_sha256_env"),
        ("authority_sha256", "authority_sha256_env"),
    ):
        _require_gate_direct_sha256(
            perceptual,
            direct_key,
            environment_key,
            prefix="training.loss.perceptual_extractor",
        )
    return dict(config)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML and enforce the shared paper-protocol configuration schema."""
    cfg_path = Path(path).expanduser()
    with cfg_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config is None:
        config = {}
    validated = validate_paper_config(config)
    unet = validated["models"]["unet"]
    if unet["depth_slab_size"] is not None:
        selection = authenticate_configured_depth_slab_selection(
            validated,
            base_dir=cfg_path.parent,
        )
        # Bind the resolved absolute path and the environment-resolved digest
        # into the exact config subsequently stored in checkpoints.
        unet["depth_slab_sweep_summary_path"] = selection["summary_path"]
        unet["depth_slab_sweep_summary_sha256"] = selection["summary_sha256"]
        unet["depth_slab_sweep_summary_sha256_env"] = ""
        unet["depth_slab_ddp_smoke_path"] = selection["ddp_smoke_path"]
        unet["depth_slab_ddp_smoke_sha256"] = selection["ddp_smoke_sha256"]
        unet["depth_slab_ddp_smoke_sha256_env"] = ""
    return validated


def resolve_configured_value(
    mapping: Mapping[str, Any], direct_key: str, environment_key: str
) -> Any:
    """Resolve a direct config value or its explicitly named environment value."""
    value = mapping.get(direct_key)
    if value:
        return value
    environment_name = str(mapping.get(environment_key, "")).strip()
    return os.environ.get(environment_name) if environment_name else None


__all__ = [
    "A4_RECOVERY_PROTOCOL_PROFILE",
    "PAPER_PREPROCESSING_EVIDENCE_STATUS",
    "PAPER_PROTOCOL_PROFILE",
    "RECOVERY_PREPROCESSING_EVIDENCE_STATUS",
    "load_config",
    "resolve_configured_value",
    "validate_figure1_recovery_gate_config",
    "validate_paper_config",
]
