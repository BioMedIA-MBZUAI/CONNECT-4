from pathlib import Path

import pytest

from architecture_contract import (
    REJECTED_LEGACY_CHECKPOINT_FORMATS,
    REJECTED_SYNTHESIS_LAUNCH_SCHEMAS,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    TRAINING_CHECKPOINT_FORMAT,
    require_current_synthesis_launch_schema,
    synthesis_architecture_contract,
)


def test_v18_contract_closes_token_diffusion_texture_gradient_and_identity():
    contract = synthesis_architecture_contract()

    assert TRAINING_CHECKPOINT_FORMAT == "connect4_iteration_exact_v18"
    assert SYNTHESIS_ARCHITECTURE_SCHEMA == (
        "connect4-synthesis-architecture-contract-v18"
    )
    assert "connect4_iteration_exact_v6" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v7" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v8" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v9" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4-half-native-v9-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4_iteration_exact_v11" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v12" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v13" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v14" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v15" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v16" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4_iteration_exact_v17" in REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert "connect4-synthesis-v11-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v12-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v13-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v14-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v15-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v16-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    assert "connect4-synthesis-v17-launch-admission-v1" in (
        REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    )
    with pytest.raises(RuntimeError, match="pre-V18 signed launch identities"):
        require_current_synthesis_launch_schema(
            "connect4-half-native-v9-launch-admission-v1"
        )
    assert contract["full_resolution_detail_path"] == "forbidden-in-synthesis"
    assert contract["t1_high_pass_detail_injection"] == (
        "forbidden-painted-anatomy"
    )
    assert contract["paper_figure_structural_shape"] == [128, 128, 128]
    assert "benchmark-release-stop" in contract["paper_profile_latent_contract"]
    assert contract["a4_recovery_output_shape"] == [64, 80, 64]
    assert contract["a4_recovery_latent_shape"] == [64, 80, 64]
    assert contract["rejected_v8_a4_recovery_latent_shape"] == [16, 20, 16]
    assert contract["a4_recovery_spatial_claim_status"] == (
        "unpublished-non-certified"
    )
    assert contract["diffusion_domain"] == (
        "connect4-token-latent-diffusion-v1"
    )
    assert contract["raw_voxel_space_epsilon_prediction_accepted"] is False
    assert contract["epsilon_target_output_dimension_equality_required"] is True
    assert contract["epsilon_head"] == (
        "hidden-size-to-hidden-size-token-projection"
    )
    assert contract["production_temporal_patch_size"] == 1
    assert contract["production_token_codec_compression_ratio"] == 8.0
    assert contract["rejected_v9_token_codec_compression_ratio"] == 128.0
    assert contract["token_codec_reconstruction_weight"] == 1.0
    assert contract["explicit_structural_brain_mask_required"] is True
    assert contract["roi_union_brain_mask_fallback_accepted"] is False
    assert contract["roi_mask_value_contract"] == (
        "finite-exact-binary-no-thresholding"
    )
    assert contract["roi_mask_overlap_contract"] == "spatially-disjoint"
    assert contract["roi_foreground_outside_structural_brain_mask_accepted"] is False
    assert contract["public_caller_supplied_ddim_initial_noise_accepted"] is False
    assert contract["v18_inference_launch_status"].startswith("blocked-pending")
    assert contract["checkpoint_state_mapping_valued_entries_accepted"] is False
    assert "flat-string-to-tensor" in contract["checkpoint_state_tree_contract"]
    assert contract["token_codec_manuscript_status"].startswith(
        "implementation-choice"
    )
    assert contract["training_composite_gradient_contract"] == (
        "biological-losses-update-fusion-dit-and-tc-unet-v1"
    )
    assert contract["full_ddim_backpropagation_claimed"] is False
    assert contract["generation_compatibility_includes_sampling_seed"] is True
    assert contract["sealed_structural_identity_may_bind_bold"] is False
    assert contract["structural_source_binding_schema"] == (
        "connect4-structural-source-binding-v2"
    )
    assert contract["rejected_structural_source_binding_schemas"] == [
        "connect4-structural-source-binding-v1"
    ]
    assert contract["native_structural_alignment_schema"] == (
        "connect4-native-structural-alignment-binding-v2"
    )
    assert contract["rejected_native_structural_alignment_schemas"] == [
        "connect4-native-structural-alignment-binding-v1"
    ]
    assert "complete-configured-spatial-grid" in contract[
        "latent_spatial_resolution_contract"
    ]
    assert contract["tc_unet_depth_pooling"].startswith("forbidden")
    assert contract["tc_unet_depth_slab_halo"] == 14
    assert contract["tc_unet_final_head"].endswith("1x1x1-conv3d")
    assert contract["tc_unet_interpolation"] == "forbidden"
    assert contract["tc_unet_output_parameterization"] == (
        "differentiable-sigmoid-after-final-head"
    )
    assert contract["inference_output_clamping"].startswith("forbidden")
    assert contract["flattened_joint_spatiotemporal_sdpa_accepted"] is False
    assert contract["dit_figure1c_spatial_attention"].startswith("3d-window")
    assert contract["unmeasured_depth_slab_core_accepted"] is False
    assert "current-source-config-bound-largest-safe" in contract[
        "tc_unet_depth_slab_selection_contract"
    ]
    assert "matching-four-rank-ddp" in contract[
        "tc_unet_depth_slab_selection_contract"
    ]
    assert contract["dit_to_tc_unet_projection_manuscript_status"].startswith(
        "implementation-choice"
    )
    assert contract["tc_unet_spatial_normalization_manuscript_status"].startswith(
        "implementation-choice"
    )
    assert contract["run_artifact_identity_contract"] == "connect4_run_artifacts_v6"
    assert contract["rejected_run_artifact_identity_contracts"] == [
        "connect4_run_artifacts_v4",
        "connect4_run_artifacts_v5",
    ]
    assert contract["temporal_coherence_contract"] == (
        "connect4-target-delta-relative-l1-v1"
    )
    assert contract["static_temporal_mean_relative_loss"] == 1.0


def test_v18_runtime_steps_and_checkpoint_publication_fail_closed_on_model_state():
    root = Path(__file__).resolve().parents[1]
    training = (root / "training" / "train.py").read_text(encoding="utf-8")
    one_gpu = (
        root / "scripts" / "benchmark_figure1_v10_gpu.py"
    ).read_text(encoding="utf-8")
    four_gpu = (
        root / "scripts" / "benchmark_figure1_v10_ddp.py"
    ).read_text(encoding="utf-8")

    assert (
        "scaler.step(opt)\n                core.validate_model_state_invariants()"
        in training
    )
    publish = training.index("def publish_checkpoint() -> bool:")
    validate = training.index("core.validate_model_state_invariants()", publish)
    payload = training.index("payload = {", publish)
    assert publish < validate < payload
    assert '"production_token_codec_shape"' in training
    assert "pre-v18 synthesis checkpoints" in training

    assert (
        "optimizer.step()\n        model.validate_model_state_invariants()"
        in one_gpu
    )
    assert (
        "optimizer.step()\n    core.validate_model_state_invariants()"
        in four_gpu
    )
    for source in (one_gpu, four_gpu):
        assert "require_production_token_codec_shape(" in source
        assert "dit_tied_token_codec" in source
        assert "dit_clean_token_encoder" not in source
        assert "dit_token_decoder" not in source
