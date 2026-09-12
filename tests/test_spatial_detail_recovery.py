import torch

from architecture_contract import FULL_RESOLUTION_DETAIL_CONTRACT
from models.spatial_detail import FullResolutionT1DetailPath


def _checkerboard(size: int = 8) -> torch.Tensor:
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size) for _ in range(3)), indexing="ij"
        )
    )
    values = (coordinates.sum(dim=0) % 2).float() * 2.0 - 1.0
    return values[None, None]


def test_full_resolution_detail_path_preserves_high_pass_capacity_and_subject_identity():
    torch.manual_seed(81)
    path = FullResolutionT1DetailPath(
        hidden_channels=3,
        highpass_sigma_voxels=0.8,
        gain_limit=0.2,
    ).eval()
    t1_a = _checkerboard()
    t1_b = -t1_a
    latent = torch.randn(1, 1, 3, 2, 2, 2)
    mask = torch.ones_like(t1_a)
    with torch.no_grad():
        path.gain_parameter.fill_(1.0)

    residual_a = path(t1_a, latent, mask=mask)
    residual_b = path(t1_b, latent, mask=mask)

    assert residual_a.shape == (1, 1, 3, 8, 8, 8)
    assert residual_a.abs().mean() > 1e-3
    assert not torch.allclose(residual_a, residual_b)
    # An alternating input must remain expressible at the native output grid;
    # no 16^3 (or other fixed) spatial bottleneck is traversed by this path.
    neighbor_gradient = (residual_a[..., 1:, :, :] - residual_a[..., :-1, :, :]).abs()
    assert neighbor_gradient.mean() > 1e-3


def test_detail_gate_uses_each_frames_spatial_latent_field_not_a_global_mean():
    assert FullResolutionT1DetailPath.contract == FULL_RESOLUTION_DETAIL_CONTRACT
    path = FullResolutionT1DetailPath(
        hidden_channels=2,
        highpass_sigma_voxels=0.8,
        gain_limit=0.2,
    ).eval()
    # The local modulation is deliberately zero until it is learned.
    assert torch.count_nonzero(path.temporal_gate.weight) == 0
    assert torch.count_nonzero(path.temporal_gate.bias) == 0
    with torch.no_grad():
        path.gain_parameter.fill_(1.0)
        path.temporal_gate.weight[0, 0, 1, 1, 1] = 1.0

    t1 = _checkerboard()
    latent = torch.zeros(1, 1, 2, 2, 2, 2)
    # Equal global means: the removed scalar-per-frame implementation produced
    # identical detail for these two frames despite opposite spatial locations.
    latent[0, 0, 0, 0, 0, 0] = 1.0
    latent[0, 0, 1, 1, 1, 1] = 1.0
    residual = path(t1, latent, mask=torch.ones_like(t1))

    assert residual.shape == (1, 1, 2, 8, 8, 8)
    assert not torch.allclose(residual[:, :, 0], residual[:, :, 1])
    assert (residual[:, :, 0] - residual[:, :, 1]).abs().mean() > 1e-4


def test_mask_normalized_high_pass_has_no_constant_brain_boundary_rim():
    path = FullResolutionT1DetailPath(
        hidden_channels=2,
        highpass_sigma_voxels=1.0,
        gain_limit=0.2,
    ).eval()
    mask = torch.zeros(1, 1, 10, 10, 10)
    mask[..., 2:8, 1:9, 3:9] = 1.0
    constant_t1 = 0.73 * mask
    dirty_exterior_t1 = constant_t1 + (1.0 - mask) * 1e6

    high_pass = path.high_pass(constant_t1, mask=mask)
    dirty_high_pass = path.high_pass(dirty_exterior_t1, mask=mask)
    assert torch.count_nonzero(high_pass * (1.0 - mask)) == 0
    assert high_pass[mask.bool()].abs().max() <= 1e-6
    assert torch.allclose(dirty_high_pass, high_pass, atol=1e-6, rtol=0.0)

    # Even trained refinement/gating parameters may only modulate supported
    # anatomy; they cannot turn a zero high pass into an anatomical boundary.
    with torch.no_grad():
        path.gain_parameter.fill_(1.0)
        path.spatial_refinement[-1].weight.fill_(0.2)
        path.spatial_refinement[-1].bias.fill_(3.0)
        path.temporal_gate.weight.fill_(0.1)
        path.temporal_gate.bias.fill_(1.0)
    residual = path(
        constant_t1,
        torch.randn(1, 1, 2, 3, 3, 3),
        mask=mask,
    )
    assert torch.count_nonzero(residual * (1.0 - mask.unsqueeze(2))) == 0
    assert residual.abs().max() <= 1e-6


def test_detail_path_initial_output_is_zero_with_first_step_gain_gradient():
    torch.manual_seed(820)
    path = FullResolutionT1DetailPath(
        hidden_channels=2,
        highpass_sigma_voxels=1.0,
        gain_limit=0.15,
    ).train()
    t1 = _checkerboard()
    latent = torch.randn(1, 1, 2, 2, 2, 2)
    mask = torch.ones_like(t1)

    residual = path(t1, latent, mask=mask)
    assert torch.count_nonzero(residual) == 0
    target = path.high_pass(t1, mask=mask).unsqueeze(2).expand_as(residual)
    (residual - target).square().mean().backward()

    assert path.gain_parameter.grad is not None
    assert path.gain_parameter.grad.abs() > 0


def test_full_resolution_detail_path_is_bounded_masked_and_trainable():
    torch.manual_seed(82)
    path = FullResolutionT1DetailPath(
        hidden_channels=2,
        highpass_sigma_voxels=1.0,
        gain_limit=0.15,
    ).train()
    t1 = _checkerboard().requires_grad_(True)
    latent = torch.randn(1, 1, 2, 2, 2, 2, requires_grad=True)
    mask = torch.zeros_like(t1)
    mask[..., 1:-1, 1:-1, 1:-1] = 1.0
    with torch.no_grad():
        # Represents the state after the zero-initialized gain's first update.
        path.gain_parameter.fill_(0.5)

    residual = path(t1, latent, mask=mask)
    assert torch.count_nonzero(residual * (1.0 - mask.unsqueeze(2))) == 0
    assert residual.abs().max() <= path.gain_limit * 1.25 + 1e-6

    residual.square().mean().backward()
    assert t1.grad is not None and t1.grad.abs().sum() > 0
    assert path.gain_parameter.grad is not None
    assert path.gain_parameter.grad.abs() > 0
    # Local parameters train after the bounded zero-initial gain moves away
    # from zero on its first paired-target update.
    assert path.temporal_gate.weight.grad is not None
    assert path.temporal_gate.weight.grad.abs().sum() > 0
    final_convolution = path.spatial_refinement[-1]
    assert final_convolution.weight.grad is not None
    assert final_convolution.weight.grad.abs().sum() > 0
