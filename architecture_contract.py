"""Closed synthesis-architecture identity for training and inference checkpoints."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import product
from numbers import Integral
from types import MappingProxyType
from typing import Any


TRAINING_CHECKPOINT_FORMAT = "connect4_iteration_exact_v18"
REJECTED_LEGACY_CHECKPOINT_FORMATS = frozenset(
    {
        "connect4_iteration_exact_v2",
        "connect4_iteration_exact_v3",
        "connect4_iteration_exact_v4",
        "connect4_iteration_exact_v5",
        "connect4_iteration_exact_v6",
        "connect4_iteration_exact_v7",
        "connect4_iteration_exact_v8",
        "connect4_iteration_exact_v9",
        "connect4_iteration_exact_v10",
        "connect4_iteration_exact_v11",
        "connect4_iteration_exact_v12",
        "connect4_iteration_exact_v13",
        "connect4_iteration_exact_v14",
        "connect4_iteration_exact_v15",
        "connect4_iteration_exact_v16",
        "connect4_iteration_exact_v17",
    }
)
SYNTHESIS_ARCHITECTURE_SCHEMA = "connect4-synthesis-architecture-contract-v18"
SYNTHESIS_LAUNCH_ADMISSION_SCHEMA = (
    "connect4-synthesis-v18-launch-admission-v1"
)
REJECTED_SYNTHESIS_LAUNCH_SCHEMAS = frozenset(
    {
        "connect4-half-native-v9-static-launch-envelope-v1",
        "connect4-half-native-v9-launch-admission-v1",
        "connect4-synthesis-v10-launch-admission-v1",
        "connect4-synthesis-v11-launch-admission-v1",
        "connect4-synthesis-v12-launch-admission-v1",
        "connect4-synthesis-v13-launch-admission-v1",
        "connect4-synthesis-v14-launch-admission-v1",
        "connect4-synthesis-v15-launch-admission-v1",
        "connect4-synthesis-v16-launch-admission-v1",
        "connect4-synthesis-v17-launch-admission-v1",
    }
)
FULL_RESOLUTION_DETAIL_CONTRACT = (
    "connect4-full-resolution-t1-detail-recovery-v3"
)
FRAMEWISE_SSIM_CONTRACT = "connect4-framewise-3d-ssim-full-sequence-v2"
PATCH_CENTER_CONVENTION = (
    "start + patch_width / 2 - 0.5 in voxel-centre coordinates"
)
TOKEN_LATENT_NORMALIZATION_CONTRACT = (
    "connect4-tied-dc-detail-row-orthonormal-projection-codec-v3"
)
TOKEN_CODEC_STATE_VERSION = 18
TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE = 0.05
TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE = 16.0
TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER = 20.0
TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE = 4.0
TOKEN_CODEC_ORTHONORMAL_ATOL = 2e-5
PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE = 512
PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM = 4096
PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO = 8

# FreeSurfer aseg labels in the one permitted graph/DWI/ROI-channel order.  The
# order is part of the architecture, not dataset metadata that a caller may
# reinterpret at runtime.
CANONICAL_ROI_SPECS = (
    ("left_cerebral_white_matter", 2),
    ("left_cerebral_cortex", 3),
    ("left_lateral_ventricle", 4),
    ("left_inferior_lateral_ventricle", 5),
    ("left_cerebellum_white_matter", 7),
    ("left_cerebellum_cortex", 8),
    ("left_thalamus", 10),
    ("left_caudate", 11),
    ("left_putamen", 12),
    ("left_pallidum", 13),
    ("third_ventricle", 14),
    ("fourth_ventricle", 15),
    ("brain_stem", 16),
    ("left_hippocampus", 17),
    ("left_amygdala", 18),
    ("csf", 24),
    ("left_accumbens_area", 26),
    ("left_ventral_dc", 28),
    ("right_cerebral_white_matter", 41),
    ("right_cerebral_cortex", 42),
    ("right_lateral_ventricle", 43),
    ("right_inferior_lateral_ventricle", 44),
    ("right_cerebellum_white_matter", 46),
    ("right_cerebellum_cortex", 47),
    ("right_thalamus", 49),
    ("right_caudate", 50),
    ("right_putamen", 51),
    ("right_pallidum", 52),
    ("right_hippocampus", 53),
    ("right_amygdala", 54),
    ("right_accumbens_area", 58),
    ("right_ventral_dc", 60),
)
CANONICAL_ROI_LABEL_IDS = tuple(label_id for _, label_id in CANONICAL_ROI_SPECS)
CANONICAL_ROI_LABEL_TO_CHANNEL = MappingProxyType(
    {label_id: channel for channel, label_id in enumerate(CANONICAL_ROI_LABEL_IDS)}
)

TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE = MappingProxyType(
    {
        "paper-a4-adni-fmriprep-v1": "connect4-observed-bold-support-v1",
        "a4-native-recovery-v1": (
            "connect4-noncertified-native-observed-support-derived-v1"
        ),
    }
)
PRODUCTION_SYNTHESIS_PROTOCOL_PROFILES = frozenset(
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE
)


def require_production_token_codec_shape(
    *,
    protocol_profile: object,
    hidden_size: object,
    patch_value_dim: object,
) -> dict[str, int]:
    """Bind paper/recovery claims to the one V18 4096-to-512 codec shape.

    Direct research construction remains configurable, but neither production
    protocol profile may carry a different width or compression ratio into
    configuration, model, architecture, or checkpoint evidence.
    """

    profile = str(protocol_profile).strip()
    if profile not in PRODUCTION_SYNTHESIS_PROTOCOL_PROFILES:
        raise ValueError(
            "V18 production token-codec admission requires a paper or A4 "
            "recovery protocol profile"
        )
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, Integral)
        or int(hidden_size) != PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE
    ):
        raise ValueError(
            "V18 production models.dit.hidden_size must be exactly "
            f"{PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE}"
        )
    if (
        isinstance(patch_value_dim, bool)
        or not isinstance(patch_value_dim, Integral)
        or int(patch_value_dim) != PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM
    ):
        raise ValueError(
            "V18 production token patch_value_dim must be exactly "
            f"{PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM}"
        )
    compression = int(patch_value_dim) // int(hidden_size)
    if (
        int(patch_value_dim) % int(hidden_size)
        or compression != PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO
    ):
        raise ValueError(
            "V18 production token codec must use exactly 8x compression"
        )
    return {
        "hidden_size": PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
        "patch_value_dim": PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
        "compression_ratio": PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
    }


def _positive_integer_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty integer sequence") from exc
    if not raw or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item < 1
        for item in raw
    ):
        raise ValueError(f"{label} must be a non-empty positive-integer sequence")
    return tuple(int(item) for item in raw)


def geometric_patch_center_voxel(
    patch_grid_index: Any,
    patch_size: Any,
) -> tuple[float, ...]:
    """Return one geometric patch centre on a voxel-centre coordinate grid."""

    try:
        grid = tuple(patch_grid_index)
    except TypeError as exc:
        raise ValueError("patch_grid_index must be an integer sequence") from exc
    width = _positive_integer_tuple(patch_size, label="patch_size")
    if len(grid) != len(width) or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item < 0
        for item in grid
    ):
        raise ValueError(
            "patch_grid_index must be a non-negative integer sequence with the "
            "same rank as patch_size"
        )
    return tuple(
        int(index) * size + size / 2.0 - 0.5
        for index, size in zip(grid, width)
    )


def geometric_patch_centers_voxel(
    target_shape: Any,
    patch_size: Any,
) -> tuple[tuple[float, ...], ...]:
    """Return all patch centres in C order (last axis varies fastest)."""

    shape = _positive_integer_tuple(target_shape, label="target_shape")
    width = _positive_integer_tuple(patch_size, label="patch_size")
    if len(shape) != len(width):
        raise ValueError("target_shape and patch_size must have equal rank")
    if any(size % step for size, step in zip(shape, width)):
        raise ValueError("target_shape must be divisible by patch_size")
    grid_shape = tuple(size // step for size, step in zip(shape, width))
    return tuple(
        geometric_patch_center_voxel(index, width)
        for index in product(*(range(count) for count in grid_shape))
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_roi_records() -> list[dict[str, Any]]:
    return [
        {
            "channel_index": channel,
            "label_id": label_id,
            "name": name,
        }
        for channel, (name, label_id) in enumerate(CANONICAL_ROI_SPECS)
    ]


CANONICAL_ROI_MAPPING_SHA256 = _canonical_sha256(_canonical_roi_records())


def synthesis_architecture_contract() -> dict[str, Any]:
    """Return a fresh, weights-only-safe copy of the current closed contract."""

    return {
        "schema": SYNTHESIS_ARCHITECTURE_SCHEMA,
        "checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "required_external_synthesis_launch_admission_schema": (
            SYNTHESIS_LAUNCH_ADMISSION_SCHEMA
        ),
        "external_synthesis_launch_admission_status": "unresolved-release-stop",
        "rejected_synthesis_launch_schemas": sorted(
            REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
        ),
        "preprocessing_authority_contract": (
            "separate-current-authority-with-independent-terminal-GO-required"
        ),
        "preprocessing_authority_required_for_model_admission_roles": [
            "exploratory",
            "production",
        ],
        "preprocessing_authority_version_preauthorization": "forbidden-none",
        "full_resolution_detail_contract": FULL_RESOLUTION_DETAIL_CONTRACT,
        "full_resolution_detail_path": "forbidden-in-synthesis",
        "t1_high_pass_detail_injection": "forbidden-painted-anatomy",
        "prediction_texture_source": (
            "learned-token-diffusion-x0-decoder-plus-tc-unet-only"
        ),
        "detail_v1_checkpoint_accepted": False,
        "paper_figure_structural_shape": [128, 128, 128],
        "paper_profile_latent_contract": (
            "tied-orthogonal-token-latent-dit-a100-40gb-benchmark-release-stop-v3"
        ),
        "a4_recovery_output_shape": [64, 80, 64],
        "a4_recovery_latent_shape": [64, 80, 64],
        "rejected_v8_a4_recovery_latent_shape": [16, 20, 16],
        "a4_recovery_spatial_claim_status": "unpublished-non-certified",
        "latent_spatial_resolution_contract": (
            "token-x0-decoded-once-to-complete-configured-spatial-grid"
        ),
        "diffusion_domain": "connect4-token-latent-diffusion-v1",
        "token_latent_normalization_contract": (
            TOKEN_LATENT_NORMALIZATION_CONTRACT
        ),
        "token_codec_state_version": TOKEN_CODEC_STATE_VERSION,
        "token_codec_effective_encoder": (
            "fp32-fixed-dc-row-plus-bounded-qr-orthonormal-dc-complement-detail-rows"
        ),
        "token_latent_scale_interpretation": (
            "orthonormal-coordinate-projection-with-fixed-dc-amplitude"
        ),
        "token_codec_hidden_size_constraint": (
            "two-or-more-and-no-greater-than-patch-value-dimension"
        ),
        "token_codec_raw_detail_parameterization": (
            "full-row-rank-[hidden-size-minus-one,patch-value-dimension-minus-one]-"
            "coordinates-in-fixed-dc-orthogonal-householder-basis"
        ),
        "token_codec_raw_detail_minimum_singular_value": (
            TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
        ),
        "token_codec_raw_detail_maximum_singular_value": (
            TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE
        ),
        "token_codec_raw_detail_maximum_condition_number": (
            TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER
        ),
        "token_codec_raw_detail_maximum_rms_singular_value": (
            TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE
        ),
        "token_codec_raw_detail_gauge_contract": (
            "finite-full-rank-min-max-singular-condition-and-rms-norm-bounded"
        ),
        "token_codec_effective_encoder_orthonormality": (
            "E-times-E-transpose-equals-identity-in-fp32"
        ),
        "token_codec_compute_dtype": (
            "fp32-autocast-disabled-for-orthogonalization-encode-and-tied-decode"
        ),
        "token_codec_orthonormality_runtime_atol": TOKEN_CODEC_ORTHONORMAL_ATOL,
        "token_codec_dc_row": (
            "exact-fixed-all-ones-divided-by-sqrt-patch-value-dimension-row-zero"
        ),
        "token_codec_encoder_bias": "forbidden-no-parameter",
        "token_codec_decoder_bias": "forbidden-no-parameter",
        "legacy_independent_token_decoder_parameters_accepted": False,
        "encoder_decoder_inverse_scale_symmetry_accepted": False,
        "raw_voxel_space_epsilon_prediction_accepted": False,
        "clean_target_tokenization": (
            "aligned-4d-patch-tied-orthonormal-projection-once-before-forward-noising"
        ),
        "production_temporal_patch_size": 1,
        "production_raw_values_per_token": PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
        "production_token_width": PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
        "production_token_codec_compression_ratio": (
            PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO
        ),
        "production_token_codec_shape_admission": (
            "paper-and-a4-recovery-profiles-require-exactly-4096-to-512-8x"
        ),
        "production_token_codec_shape_record": {
            "hidden_size": PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
            "patch_value_dim": PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
            "compression_ratio": PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
        },
        "rejected_v9_temporal_patch_size": 16,
        "rejected_v9_token_codec_compression_ratio": 128.0,
        "epsilon_target_shape": "[B,temporal_groups*spatial_patches,hidden_size]",
        "epsilon_head": "hidden-size-to-hidden-size-token-projection",
        "epsilon_target_output_dimension_equality_required": True,
        "token_noise_coordinate_contract": "iid-gaussian-all-hidden-coordinates",
        "token_to_patch_decoder": (
            "exact-transpose-of-same-effective-encoder-no-parameters-no-bias"
        ),
        "token_to_patch_decoder_operator_norm": 1.0,
        "token_codec_roundtrip_semantics": (
            "orthogonal-projection-onto-fixed-dc-plus-learned-detail-row-subspace"
        ),
        "token_codec_dc_mean_preservation": "exact-in-real-arithmetic",
        "token_codec_detail_preservation": (
            "exact-only-within-learned-orthonormal-detail-subspace"
        ),
        "token_codec_full_patch_invertibility_when_compressed_claimed": False,
        "token_codec_runtime_invariant_validator": (
            "Connect4Model.validate_model_state_invariants-before-and-after-load-"
            "after-optimizer-step-and-before-checkpoint-publication"
        ),
        "checkpoint_model_state_finiteness_contract": (
            "all-floating-and-complex-parameters-and-buffers-recursively-finite"
        ),
        "checkpoint_state_tree_contract": (
            "cycle-safe-full-path-flat-string-to-tensor-mapping-only-v3"
        ),
        "checkpoint_codec_state_path_contract": (
            "exact-flat-surface-specific-marker-and-raw-detail-paths-v1"
        ),
        "checkpoint_codec_preflight_contract": (
            "full-flat-model-and-direct-dit-before-any-super-load-v2"
        ),
        "nested_dit_failure_after_parent_mutation_accepted": False,
        "checkpoint_state_mapping_valued_entries_accepted": False,
        "unexpected_codec_state_subtrees_accepted": False,
        "checkpoint_state_nested_metadata_accepted": False,
        "checkpoint_state_unsupported_container_leaves_accepted": False,
        "nested_legacy_codec_state_accepted": False,
        "token_codec_manuscript_status": (
            "implementation-choice-latent-codec-underspecified"
        ),
        "token_codec_reconstruction_objective": (
            "target-validity-supported-voxel-mse-plus-mean-adjacent-target-"
            "validity-supported-t-d-h-w-first-difference-mse"
        ),
        "token_codec_mask_source": "paired-target-validity-mask-training-only",
        "token_diffusion_mask_source": "paired-target-validity-mask-training-only",
        "token_diffusion_patch_weighting": (
            "mean-target-validity-fraction-per-spatiotemporal-token-then-"
            "support-normalized-mean-over-token-and-hidden-coordinates"
        ),
        "target_validity_mask_shape": "[B,1,D,H,W]-constant-over-target-frames",
        "target_validity_mask_contract_by_protocol_profile": dict(
            TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE
        ),
        "target_validity_mask_derivation": (
            "exact-nonzero-over-time-paired-fmri-support-no-error-dependent-subset"
        ),
        "target_validity_mask_without_paired_target_accepted": False,
        "structural_brain_mask_roles": [
            "target-blind-conditioning",
            "target-blind-output-support",
            "upper-bound-on-paired-target-validity",
        ],
        "inference_output_mask_source": (
            "target-independent-authenticated-structural-brain-mask"
        ),
        "explicit_structural_brain_mask_required": True,
        "roi_union_brain_mask_fallback_accepted": False,
        "roi_mask_value_contract": "finite-exact-binary-no-thresholding",
        "roi_mask_key_contract": "canonical-contiguous-zero-based-exact-count",
        "canonical_roi_count": len(CANONICAL_ROI_SPECS),
        "canonical_roi_records": _canonical_roi_records(),
        "canonical_roi_mapping_sha256": CANONICAL_ROI_MAPPING_SHA256,
        "roi_mask_overlap_contract": "spatially-disjoint",
        "roi_foreground_outside_structural_brain_mask_accepted": False,
        "token_codec_reconstruction_weight": 1.0,
        "token_codec_reconstruction_manuscript_status": (
            "implementation-only-safeguard-not-paper-reported-loss"
        ),
        "dit_to_tc_unet_projection": (
            "per-frame-1x1x1-channel-only-full-grid-1-to-64"
        ),
        "dit_to_tc_unet_projection_manuscript_status": (
            "implementation-choice-projection-operator-underspecified"
        ),
        "dit_figure1c_spatial_attention": (
            "3d-window-covering-complete-8x8x8-paper-patch-grid"
        ),
        "dit_spatial_window_size": [8, 8, 8],
        "dit_attention_factorization": (
            "per-frame-3d-spatial-window-then-per-location-global-temporal"
        ),
        "dit_temporal_attention": "global-over-all-temporal-patch-groups",
        "dit_attention_manuscript_ambiguity_resolution": (
            "figure-labels-3d-window-prose-says-global-use-one-full-grid-window"
        ),
        "flattened_joint_spatiotemporal_sdpa_accepted": False,
        "tc_unet_figure1_shapes": [
            "[B,T,64,D,H,W]",
            "[B,T,128,D,H/2,W/2]",
            "[B,T,256,D,H/4,W/4]",
            "[B,T,512,D,H/8,W/8]",
        ],
        "tc_unet_depth_pooling": "forbidden-D-preserved-at-every-level",
        "tc_unet_hw_pooling": "maxpool3d-kernel-and-stride-1x2x2",
        "tc_unet_hw_upsampling": "convtranspose3d-kernel-and-stride-1x2x2",
        "tc_unet_final_head": "per-frame-stride-one-1x1x1-conv3d",
        "tc_unet_output_parameterization": "differentiable-sigmoid-after-final-head",
        "tc_unet_output_parameterization_manuscript_status": (
            "implementation-choice-output-domain-underspecified"
        ),
        "inference_output_clamping": "forbidden-fail-closed-outside-zero-one",
        "public_caller_supplied_ddim_initial_noise_accepted": False,
        "production_ddim_initial_noise": (
            "deterministic-scan-and-context-bound-token-latent-only"
        ),
        "v18_inference_launch_status": (
            "blocked-pending-external-signed-scheduler-runtime-admission"
        ),
        "tc_unet_interpolation": "forbidden",
        "tc_unet_depth_slab_halo": 14,
        "tc_unet_depth_slab_contract": (
            "derived-longest-path-radius-exact-overlap-crop-no-stitch-average-v1"
        ),
        "tc_unet_temporal_attention_kernel": (
            "scaled-dot-product-attention-no-materialized-t-by-t-score-contract-v1"
        ),
        "tc_unet_depth_slab_selection_contract": (
            "current-source-config-bound-largest-safe-per-rank-a100-core-plus-"
            "matching-four-rank-ddp-full-step-four-gib-margin-v2"
        ),
        "unmeasured_depth_slab_core_accepted": False,
        "tc_unet_spatial_normalization": (
            "voxelwise-channel-layernorm-finite-depth-receptive-field"
        ),
        "tc_unet_spatial_normalization_manuscript_status": (
            "implementation-choice-normalization-underspecified"
        ),
        "graph_dit_spatial_token_alignment": "one-to-one",
        "ssim_contract": FRAMEWISE_SSIM_CONTRACT,
        "ssim_aggregation": (
            "mean-brain-masked-framewise-3d-ssim-over-complete-4d-sequence"
        ),
        "temporal_mean_only_ssim_accepted": False,
        "temporal_coherence_contract": "connect4-target-delta-relative-l1-v1",
        "static_temporal_mean_relative_loss": 1.0,
        "decoder_training_latent_contract": (
            "tied-orthogonal-projection-single-noise-level-differentiable-token-"
            "x0-recovery-surrogate-v4"
        ),
        "target_derived_one_step_x0_decoder_training_accepted": True,
        "training_composite_gradient_contract": (
            "biological-losses-update-fusion-dit-and-tc-unet-v1"
        ),
        "training_biological_noise_weighting": (
            "mean-alpha-cumprod-bounds-x0-epsilon-sensitivity-v1"
        ),
        "full_ddim_backpropagation_claimed": False,
        "decoder_sampler_gradient_contract": (
            "training-differentiable-x0-surrogate-evaluation-no-grad-full-ddim"
        ),
        "sampling_noise_contract": (
            "sha256-per-scan-per-context-iid-token-coordinate-noise-v2"
        ),
        "sampling_noise_domain": (
            "connect4-target-blind-token-ddim-initial-noise-v2"
        ),
        "generation_compatibility_includes_sampling_seed": True,
        "training_entrypoint_contract": "train-and-development-only-v3",
        "sealed_test_target_access_during_training": "never-opened",
        "run_artifact_identity_contract": "connect4_run_artifacts_v6",
        "rejected_run_artifact_identity_contracts": [
            "connect4_run_artifacts_v4",
            "connect4_run_artifacts_v5",
        ],
        "structural_source_authority_schema": (
            "connect4-structural-source-authority-v2"
        ),
        "structural_source_binding_schema": (
            "connect4-structural-source-binding-v2"
        ),
        "rejected_structural_source_binding_schemas": [
            "connect4-structural-source-binding-v1"
        ],
        "native_structural_batch_schema": "connect4-native-structural-batch-v4",
        "native_structural_alignment_schema": (
            "connect4-native-structural-alignment-binding-v2"
        ),
        "rejected_native_structural_alignment_schemas": [
            "connect4-native-structural-alignment-binding-v1"
        ],
        "sealed_structural_identity_may_bind_bold": False,
        "patch_center_convention": PATCH_CENTER_CONVENTION,
        "fmri_intensity_contract": (
            "connect4-unpublished-v54b-functional-validity-q995-nonnegative-v2"
        ),
        "temporal_voxel_zscore_accepted": False,
        "region_histogram_support": [0.0, 1.0],
    }


SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256 = _canonical_sha256(
    synthesis_architecture_contract()
)


def require_current_synthesis_architecture(
    value: object,
    claimed_sha256: object,
) -> dict[str, Any]:
    """Reject absent, legacy, partially asserted, or mutated architecture claims."""

    expected = synthesis_architecture_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(
            "checkpoint synthesis architecture differs; pre-v18, independently "
            "scaled/bias-shifted latent-codec, raw-epsilon, target-leaking "
            "structural identity, detached-gradient, T1-painted-detail, "
            "low-resolution/all-axis-pooled, corner-shifted, or temporal-mean-only "
            "checkpoints are forbidden"
        )
    if (
        not isinstance(claimed_sha256, str)
        or claimed_sha256 != SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
        or _canonical_sha256(dict(value)) != claimed_sha256
    ):
        raise RuntimeError("checkpoint synthesis architecture digest differs")
    return expected


def require_current_synthesis_launch_schema(value: object) -> str:
    """Reject pre-V18 launch identities for V18 synthesis roles."""

    if value in REJECTED_SYNTHESIS_LAUNCH_SCHEMAS:
        raise RuntimeError(
            "pre-V18 signed launch identities cannot authorize V18 model gates, "
            "training, or inference"
        )
    if value != SYNTHESIS_LAUNCH_ADMISSION_SCHEMA:
        raise RuntimeError("current V18 synthesis launch admission is required")
    return SYNTHESIS_LAUNCH_ADMISSION_SCHEMA


__all__ = [
    "CANONICAL_ROI_LABEL_IDS",
    "CANONICAL_ROI_LABEL_TO_CHANNEL",
    "CANONICAL_ROI_MAPPING_SHA256",
    "CANONICAL_ROI_SPECS",
    "FRAMEWISE_SSIM_CONTRACT",
    "FULL_RESOLUTION_DETAIL_CONTRACT",
    "PATCH_CENTER_CONVENTION",
    "PRODUCTION_SYNTHESIS_PROTOCOL_PROFILES",
    "PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO",
    "PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE",
    "PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM",
    "REJECTED_LEGACY_CHECKPOINT_FORMATS",
    "REJECTED_SYNTHESIS_LAUNCH_SCHEMAS",
    "SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256",
    "SYNTHESIS_ARCHITECTURE_SCHEMA",
    "SYNTHESIS_LAUNCH_ADMISSION_SCHEMA",
    "TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE",
    "TOKEN_CODEC_STATE_VERSION",
    "TOKEN_CODEC_ORTHONORMAL_ATOL",
    "TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER",
    "TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE",
    "TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE",
    "TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE",
    "TOKEN_LATENT_NORMALIZATION_CONTRACT",
    "TRAINING_CHECKPOINT_FORMAT",
    "geometric_patch_center_voxel",
    "geometric_patch_centers_voxel",
    "require_production_token_codec_shape",
    "require_current_synthesis_architecture",
    "require_current_synthesis_launch_schema",
    "synthesis_architecture_contract",
]
