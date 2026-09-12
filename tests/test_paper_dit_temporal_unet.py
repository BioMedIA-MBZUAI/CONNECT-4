import torch
import pytest
from torch.nn import functional as F

from graphs.image_graph import ImageGraphBuilder
from models.dit4d import DiTBlock4D
from models.dit4d_temporal import (
    DDIMScheduler,
    DiT4DTemporal,
    TOKEN_LATENT_DIFFUSION_DOMAIN,
)
from models.tc_film_unet import ChannelLayerNorm, TCUNet4DFiLM, TemporalAttention


def test_ddim_forward_noise_and_x0_recovery_are_inverse():
    scheduler = DDIMScheduler(num_train_timesteps=20)
    clean = torch.randn(2, 1, 3, 4, 4, 4)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 17])
    noisy = scheduler.add_noise(clean, noise, timesteps)
    recovered = scheduler.predict_original_sample(noisy, noise, timesteps)
    assert torch.allclose(recovered, clean, atol=1e-5, rtol=1e-5)


def test_dit_block_cross_attention_uses_condition_tokens():
    torch.manual_seed(1)
    block = DiTBlock4D(hidden_size=12, num_heads=3, mlp_ratio=2.0).eval()
    with torch.no_grad():
        # Enable only the adaLN-Zero cross-attention residual gate.
        block.adaLN_modulation[-1].bias[8 * 12 : 9 * 12].fill_(1.0)
    x = torch.randn(1, 4, 12)
    timestep_condition = torch.randn(1, 12)
    context_a = torch.randn(1, 3, 12)
    context_b = context_a + 2.0
    shape = (2, 1, 1, 2)
    out_a = block(x, timestep_condition, context=context_a, spatiotemporal_shape=shape)
    out_b = block(x, timestep_condition, context=context_b, spatiotemporal_shape=shape)
    assert not torch.allclose(out_a, out_b)


def _tiny_dit():
    model = DiT4DTemporal(
        input_size=(4, 4, 4),
        in_channels=1,
        patch_size=(2, 2, 2),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        input_dim=7,
        t1_token_dim=5,
        num_temporal_frames=2,
        num_diffusion_steps=20,
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    ).eval()
    # Standard DiT starts as a zero denoiser. Enable the output and
    # cross-attention gate so this unit test can observe conditioning flow.
    with torch.no_grad():
        torch.nn.init.normal_(model.epsilon_head.linear.weight, std=0.05)
        for block in model.blocks:
            width = model.hidden_size
            block.adaLN_modulation[-1].bias[8 * width : 9 * width].fill_(1.0)
    return model


def test_dit_uses_same_even_patch_geometric_centres_as_image_graph():
    model = DiT4DTemporal(
        input_size=(4, 8, 12),
        in_channels=1,
        patch_size=(2, 4, 6),
        hidden_size=12,
        depth=1,
        num_heads=3,
        input_dim=7,
        t1_token_dim=5,
        num_temporal_frames=4,
        temporal_patch_size=2,
        num_diffusion_steps=20,
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    ).eval()
    spatial = ImageGraphBuilder(
        patch_size=(2, 4, 6)
    ).get_patch_positions((4, 8, 12))
    centers = model.spatiotemporal_patch_centers

    assert centers.shape == (16, 4)
    assert torch.equal(centers[:8, 0], torch.full((8,), 0.5))
    assert torch.equal(centers[8:, 0], torch.full((8,), 2.5))
    assert torch.equal(centers[:8, 1:], spatial)
    assert torch.equal(centers[8:, 1:], spatial)


def test_temporal_dit_requires_noisy_token_latent_and_hypergraph_context():
    torch.manual_seed(2)
    model = _tiny_dit()
    assert not hasattr(model, "t1_volume_embed")
    assert not hasattr(model, "t1_condition_proj")
    noisy = torch.randn(1, 16, 8)
    diffusion_t = torch.tensor([7])
    graph = torch.randn(1, 8, 7)
    out = model(noisy, diffusion_t, graph)
    out_graph_changed = model(noisy, diffusion_t, graph + 1.0)
    assert out.shape == noisy.shape
    assert not torch.allclose(out, out_graph_changed)
    with pytest.raises(ValueError, match="noisy_latent"):
        model(torch.randn(1, 1, 2, 4, 4, 4), diffusion_t, graph)


def test_temporal_dit_uses_aligned_graph_tokens_in_noisy_sequence():
    torch.manual_seed(21)
    model = _tiny_dit()
    noisy = torch.randn(1, 16, 8)
    diffusion_t = torch.tensor([5])
    graph = torch.randn(1, 8, 7)
    output_a = model(noisy, diffusion_t, graph)
    output_b = model(noisy, diffusion_t, graph + 0.5)
    assert not torch.allclose(output_a, output_b)
    with pytest.raises(ValueError, match="align one-to-one"):
        model(noisy, diffusion_t, graph[:, :-1])


def test_dit_uses_figure1_3d_window_then_global_temporal_attention():
    torch.manual_seed(22)
    block = DiTBlock4D(hidden_size=4, num_heads=1, mlp_ratio=1.0).eval()
    with torch.no_grad():
        # Enable only the self-attention residual and make q/k constant, so
        # every query directly averages values from all four joint tokens.
        block.adaLN_modulation[-1].weight.zero_()
        block.adaLN_modulation[-1].bias.zero_()
        block.adaLN_modulation[-1].bias[2 * 4 : 3 * 4].fill_(1.0)
        block.adaLN_modulation[-1].bias[5 * 4 : 6 * 4].fill_(1.0)
        for qkv in (block.spatial_qkv, block.temporal_qkv):
            qkv.weight.zero_()
            qkv.weight[2 * 4 : 3 * 4].copy_(torch.eye(4))
        for projection in (block.spatial_out_proj, block.temporal_out_proj):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
    original = torch.zeros(1, 4, 4)
    changed = original.clone()
    # Token 3 differs in both its temporal and spatial index from token 0
    # Spatial attention moves the signal within frame 1, then global temporal
    # attention moves it to frame 0 at the corresponding spatial patch.
    changed[:, 3, 0] = 4.0
    condition = torch.zeros(1, 4)
    shape = (2, 1, 1, 2)
    output_a = block(original, condition, spatiotemporal_shape=shape)
    output_b = block(changed, condition, spatiotemporal_shape=shape)
    assert not torch.allclose(output_a[:, 0], output_b[:, 0])
    with pytest.raises(ValueError, match="spatiotemporal_shape"):
        block(original, condition, spatiotemporal_shape=(2, 2))


def test_diffusion_and_reconstructed_x0_paths_both_backpropagate():
    torch.manual_seed(3)
    model = _tiny_dit().train()
    clean = torch.randn(1, 1, 2, 4, 4, 4)
    graph = torch.randn(1, 8, 7, requires_grad=True)
    result = model.diffusion_loss(
        clean,
        graph,
        timesteps=torch.tensor([9]),
        noise=torch.randn(1, 16, 8),
        target_validity_mask=torch.ones(1, 1, 4, 4, 4),
    )
    decoded = result["pred_original_volume"]
    biological = (decoded - clean.sigmoid()).square().mean()
    total = result["loss"] + biological + result["codec_reconstruction_loss"]
    total.backward()
    assert graph.grad is not None and graph.grad.abs().sum() > 0
    for parameter in (
        model.token_codec_raw_detail_rows,
        model.epsilon_head.linear.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_diffusion_epsilon_loss_uses_exact_patchwise_target_validity_weights():
    torch.manual_seed(31)
    model = _tiny_dit().train()
    clean = torch.randn(1, 1, 2, 4, 4, 4)
    graph = torch.randn(1, 8, 7)
    noise = torch.randn(1, 16, 8)
    validity = torch.ones(1, 1, 4, 4, 4)
    validity[..., :2, :2, :2] = 0

    result = model.diffusion_loss(
        clean,
        graph,
        timesteps=torch.tensor([7]),
        noise=noise,
        target_validity_mask=validity,
    )
    expanded = validity.unsqueeze(2).expand(1, 1, 2, 4, 4, 4)
    expected_weights = model._patchify_volume(expanded).mean(dim=-1)
    torch.testing.assert_close(result["token_validity"], expected_weights)
    assert (expected_weights == 0).sum() == 2
    assert (expected_weights == 1).sum() == 14

    squared_error = (result["predicted_noise"] - noise).square()
    expected_loss = (
        squared_error * expected_weights.unsqueeze(-1)
    ).sum() / (expected_weights.sum() * squared_error.shape[-1])
    torch.testing.assert_close(result["loss"], expected_loss)


def test_ddim_sampling_runs_reverse_trajectory():
    torch.manual_seed(4)
    model = _tiny_dit()
    graph = torch.randn(1, 8, 7)
    output = model.sample(
        graph,
        shape=(1, 16, 8),
        num_inference_steps=3,
    )
    assert output.shape == (1, 16, 8)
    assert model.decode_tokens(output).shape == (1, 1, 2, 4, 4, 4)
    assert torch.isfinite(output).all()


def test_ddim_initial_noise_override_is_exact_and_fail_closed():
    model = _tiny_dit()
    graph = torch.randn(1, 8, 7)
    noise = torch.randn(1, 16, 8)
    first = model.sample(
        graph, num_inference_steps=3, initial_noise=noise
    )
    second = model.sample(
        graph, num_inference_steps=3, initial_noise=noise
    )
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match="initial DDIM noise must have shape"):
        model.sample(
            graph, num_inference_steps=3, initial_noise=noise[..., :-1]
        )


def test_token_codec_and_epsilon_regression_remove_raw_patch_noise_bottleneck():
    """V14 predicts every tied orthonormal token coordinate, never raw epsilon."""

    model = DiT4DTemporal(
        input_size=(4, 4, 4),
        in_channels=1,
        patch_size=(4, 4, 4),
        hidden_size=12,
        depth=1,
        num_heads=3,
        input_dim=7,
        t1_token_dim=5,
        num_temporal_frames=2,
        temporal_patch_size=2,
        num_diffusion_steps=20,
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    ).eval()
    clean = torch.randn(2, 1, 2, 4, 4, 4)
    clean_tokens = model.encode_clean_volume(clean)
    assert model.patch_value_dim == 128
    assert clean_tokens.shape == (2, 1, 12)
    assert model.token_codec_raw_detail_rows.shape == (11, 127)
    assert not hasattr(model, "clean_token_encoder")
    assert not hasattr(model, "token_decoder")
    assert model.validate_token_codec_invariants()["decoder_parameter_count"] == 0
    assert model.epsilon_head.linear.in_features == 12
    assert model.epsilon_head.linear.out_features == 12
    result = model.diffusion_loss(
        clean,
        torch.randn(2, 1, 7),
        timesteps=torch.tensor([3, 11]),
        noise=torch.randn_like(clean_tokens),
        target_validity_mask=torch.ones(2, 1, 4, 4, 4),
    )
    assert result["target_noise"].shape == (2, 1, 12)
    assert result["predicted_noise"].shape == result["target_noise"].shape
    assert result["pred_original_sample"].shape == clean_tokens.shape
    assert result["pred_original_volume"].shape == clean.shape
    assert not hasattr(model, "noisy_latent_embed")
    assert not hasattr(model, "final_layer")


def test_token_latent_noise_occupies_every_denoiser_coordinate():
    model = _tiny_dit()
    noise = torch.randn(256, model.hidden_size)
    # With many independent draws the empirical covariance is full-rank H. This
    # would be impossible for the old raw-patch target's 116-dimensional
    # nullspace in this tiny 128-value-patch/H=12 analogue.
    assert torch.linalg.matrix_rank(noise - noise.mean(dim=0)) == model.hidden_size


def test_codec_objective_learns_checkerboard_and_temporal_detail():
    torch.manual_seed(41)
    model = DiT4DTemporal(
        input_size=(4, 4, 4),
        in_channels=1,
        patch_size=(4, 4, 4),
        hidden_size=16,
        depth=1,
        num_heads=4,
        input_dim=7,
        t1_token_dim=5,
        num_temporal_frames=2,
        temporal_patch_size=2,
        num_diffusion_steps=20,
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    )
    coordinates = torch.meshgrid(
        torch.arange(2),
        torch.arange(4),
        torch.arange(4),
        torch.arange(4),
        indexing="ij",
    )
    checkerboard = sum(coordinates).remainder(2).float()[None, None]
    optimizer = torch.optim.Adam([model.token_codec_raw_detail_rows], lr=0.03)
    brain_mask = torch.ones(1, 1, 4, 4, 4)
    initial = model.codec_reconstruction_loss(
        checkerboard, target_validity_mask=brain_mask
    )["loss"].detach()
    first_gradients = None
    for step in range(120):
        optimizer.zero_grad(set_to_none=True)
        result = model.codec_reconstruction_loss(
            checkerboard, target_validity_mask=brain_mask
        )
        result["loss"].backward()
        if step == 0:
            first_gradients = model.token_codec_raw_detail_rows.grad.detach().abs().sum()
        optimizer.step()
        model.validate_token_codec_invariants()
    final_result = model.codec_reconstruction_loss(
        checkerboard, target_validity_mask=brain_mask
    )
    reconstruction = final_result["reconstruction"].detach()
    assert first_gradients is not None
    assert first_gradients > 0
    assert final_result["loss"] < 0.02 * initial
    assert torch.corrcoef(
        torch.stack(
            (
                torch.diff(checkerboard, dim=2).flatten(),
                torch.diff(reconstruction, dim=2).flatten(),
            )
        )
    )[0, 1] > 0.98
    assert torch.corrcoef(
        torch.stack(
            (
                torch.diff(checkerboard, dim=3).flatten(),
                torch.diff(reconstruction, dim=3).flatten(),
            )
        )
    )[0, 1] > 0.98


def test_codec_mask_prevents_zero_background_from_hiding_in_brain_detail_failure():
    model = DiT4DTemporal(
        input_size=(8, 8, 8),
        in_channels=1,
        patch_size=(4, 4, 4),
        hidden_size=16,
        depth=1,
        num_heads=4,
        input_dim=7,
        t1_token_dim=5,
        num_temporal_frames=2,
        temporal_patch_size=2,
        num_diffusion_steps=20,
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    )
    target = torch.zeros(1, 1, 2, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[..., 2:6, 2:6, 2:6] = 1
    local = torch.meshgrid(
        torch.arange(2),
        torch.arange(4),
        torch.arange(4),
        torch.arange(4),
        indexing="ij",
    )
    target[..., 2:6, 2:6, 2:6] = sum(local).remainder(2).float()
    result = model.codec_reconstruction_loss(
        target, target_validity_mask=mask
    )
    expanded_mask = mask.unsqueeze(2).expand_as(target)
    expected_voxel = (
        (result["reconstruction"] - target).square() * expanded_mask
    ).sum() / expanded_mask.sum()
    torch.testing.assert_close(result["voxel"], expected_voxel)
    assert result["voxel"] > 0
    assert result["spatiotemporal_first_difference"] > 0
    assert not torch.allclose(
        result["voxel"], F.mse_loss(result["reconstruction"], target)
    )


def _tiny_unet(*, depth=20, levels=2, use_checkpoint=False):
    return TCUNet4DFiLM(
        in_channels=2,
        out_channels=1,
        base_channels=2,
        num_levels=levels,
        temporal_cond_dim=8,
        graph_cond_dim=6,
        use_checkpoint=use_checkpoint,
        use_temporal_attn=True,
        temporal_attn_heads=1,
        input_spatial_size=(depth, 8, 8),
    )


def test_final_decoder_spatial_schedule_defaults_to_single_frame():
    model = _tiny_unet()
    assert model.final_decoder_frame_chunk_size == 1


def test_final_decoder_live_input_offload_is_cuda_only_and_fail_closed():
    model = _tiny_unet()
    with pytest.raises(RuntimeError, match="only CUDA tensors"):
        model._differentiable_pinned_cpu_copy(
            torch.randn(1, 2, 2, 4, 4, 4, requires_grad=True)
        )


@pytest.mark.parametrize("shape", [(2, 4, 3, 5, 6), (2, 4, 3, 2, 5, 6)])
def test_channel_layer_norm_elides_only_the_redundant_contiguous_copy(shape):
    """The memory-safe view must be exactly the former materialised tensor."""
    torch.manual_seed(60)
    norm = ChannelLayerNorm(4)
    x = torch.randn(*shape, requires_grad=True)
    channel_last = (0, *range(2, len(shape)), 1)
    channel_first = (0, len(shape) - 1, *range(1, len(shape) - 1))
    actual = norm(x)
    legacy = F.layer_norm(
        x.permute(channel_last),
        (4,),
        norm.weight,
        norm.bias,
        norm.eps,
    ).permute(channel_first).contiguous()

    assert torch.equal(actual, legacy)
    assert not actual.is_contiguous()
    assert actual.permute(channel_last).is_contiguous()

    probe = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * probe).sum(), x, retain_graph=True)[0]
    legacy_gradient = torch.autograd.grad((legacy * probe).sum(), x)[0]
    assert torch.equal(actual_gradient, legacy_gradient)


def test_temporal_unet_preserves_depth_and_halves_only_height_width():
    model = _tiny_unet(depth=10, levels=3).eval()
    pooled_shapes = []
    hooks = [
        pool.register_forward_hook(
            lambda _module, _inputs, output: pooled_shapes.append(output.shape[-3:])
        )
        for pool in model.encoder_pool
    ]
    x = torch.randn(1, 2, 2, 10, 8, 8)
    output = model(x, torch.randn(1, 6))
    for hook in hooks:
        hook.remove()
    assert pooled_shapes == [(10, 4, 4), (10, 2, 2)]
    assert output.shape == (1, 2, 10, 8, 8)
    assert all(pool.kernel_size == (1, 2, 2) for pool in model.encoder_pool)
    assert all(
        upsample.kernel_size == (1, 2, 2)
        and upsample.stride == (1, 2, 2)
        for upsample in model.decoder_upsample
    )


def test_temporal_channel_attention_couples_frames():
    torch.manual_seed(5)
    model = _tiny_unet(depth=4).eval()
    temporal_modules = [
        module for module in model.modules() if isinstance(module, TemporalAttention)
    ]
    assert temporal_modules
    assert all(module.chunk_size == 256 for module in temporal_modules)
    x = torch.randn(1, 2, 2, 4, 8, 8, requires_grad=True)
    output = model(x, torch.randn(1, 6))
    output[:, 0].sum().backward()
    assert x.grad[:, :, 1].abs().sum() > 0


def test_temporal_unet_uses_unchanged_grid_1x1x1_final_head_and_exact_mask():
    torch.manual_seed(6)
    model = _tiny_unet(depth=8).eval()
    assert isinstance(model.output_head, torch.nn.Conv3d)
    assert model.output_head.kernel_size == (1, 1, 1)
    assert model.output_head.stride == (1, 1, 1)
    x = torch.randn(1, 2, 2, 8, 8, 8, requires_grad=True)
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[..., 1:7, 1:7, 1:7] = 1
    output = model(x, torch.randn(1, 6), mask=mask)
    assert output.shape == (1, 2, 8, 8, 8)
    assert bool(((0 <= output) & (output <= 1)).all())
    assert torch.count_nonzero(output[..., 0, :, :]) == 0
    assert torch.count_nonzero(output[..., -1, :, :]) == 0
    output.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="resampling is forbidden"):
        model(x.detach(), torch.randn(1, 6), mask=mask[..., :-1, :, :])


def test_dit_projection_changes_channels_only_on_the_complete_grid():
    from models.tc_film_unet import DiTToTCUNetProjection

    projection = DiTToTCUNetProjection(1, 2, (8, 8, 8))
    x = torch.randn(1, 1, 2, 8, 8, 8)
    output = projection(x)
    assert output.shape == (1, 2, 2, 8, 8, 8)
    assert projection.projection.kernel_size == (1, 1, 1)
    assert projection.projection.stride == (1, 1, 1)
    with pytest.raises(ValueError, match="configured full D grid"):
        projection(x[..., :-1, :, :])
    with pytest.raises(ValueError, match="configured full H,W grid"):
        projection(x[..., :, :-1, :])


def test_derived_halo_is_exact_and_slabbed_output_and_gradients_have_no_seams():
    from copy import deepcopy
    from models.tc_film_unet import DiTToTCUNetProjection

    torch.manual_seed(61)
    full_model = _tiny_unet(depth=20).eval()
    slab_model = deepcopy(full_model).train()
    slab_model.use_checkpoint = True
    full_projection = DiTToTCUNetProjection(1, 2, (20, 8, 8)).eval()
    slab_projection = deepcopy(full_projection).eval()
    assert full_model.depth_receptive_field_radius == 6

    latent_full = torch.randn(
        1, 1, 2, 20, 8, 8, requires_grad=True
    )
    latent_slab = latent_full.detach().clone().requires_grad_(True)
    graph_full = torch.randn(1, 6, requires_grad=True)
    graph_slab = graph_full.detach().clone().requires_grad_(True)
    mask = torch.ones(1, 1, 20, 8, 8)

    whole = full_model.decode_dit_volume(
        latent_full,
        full_projection,
        graph_full,
        mask=mask,
        depth_slab_size=0,
    )
    slabbed = slab_model.decode_dit_volume(
        latent_slab,
        slab_projection,
        graph_slab,
        mask=mask,
        depth_slab_size=3,
    )
    torch.testing.assert_close(slabbed, whole, rtol=2e-5, atol=2e-6)
    assert bool(((0 <= slabbed) & (slabbed <= 1)).all())

    whole.square().mean().backward()
    slabbed.square().mean().backward()
    torch.testing.assert_close(latent_slab.grad, latent_full.grad, rtol=5e-5, atol=5e-7)
    torch.testing.assert_close(graph_slab.grad, graph_full.grad, rtol=5e-5, atol=5e-7)
    torch.testing.assert_close(
        slab_projection.projection.weight.grad,
        full_projection.projection.weight.grad,
        rtol=5e-5,
        atol=5e-7,
    )


def test_decoder_skip_concat_is_backed_by_contiguous_frame_major_storage():
    """The decoder's first spatial block must not duplicate its large FP32 cat."""
    model = _tiny_unet(depth=8).eval()
    observed = []

    def capture_layout(_module, inputs):
        value = inputs[0]
        observed.append(
            {
                "logical_shape": tuple(value.shape),
                "frame_major_contiguous": value.permute(
                    0, 2, 1, 3, 4, 5
                ).is_contiguous(),
            }
        )

    hook = model.decoder[0].register_forward_pre_hook(capture_layout)
    output = model(torch.randn(1, 2, 2, 8, 8, 8), torch.randn(1, 6))
    hook.remove()

    assert output.shape == (1, 2, 8, 8, 8)
    assert observed
    assert all(item["frame_major_contiguous"] for item in observed)
    assert all(item["logical_shape"][2] == 2 for item in observed)


def test_four_level_figure1_halo_is_derived_as_fourteen_voxels():
    model = _tiny_unet(depth=32, levels=4)
    assert model.depth_receptive_field_radius == 14


def test_odd_height_width_are_rejected_instead_of_interpolated():
    with pytest.raises(ValueError, match="interpolation-based skip repair is forbidden"):
        TCUNet4DFiLM(
            in_channels=2,
            out_channels=1,
            base_channels=2,
            num_levels=3,
            temporal_cond_dim=8,
            graph_cond_dim=6,
            temporal_attn_heads=1,
            input_spatial_size=(8, 10, 12),
        )
