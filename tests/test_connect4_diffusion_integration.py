import copy
import inspect
import math

import pytest
import torch
from torch import nn

from utils.config import PAPER_PROTOCOL_PROFILE
from data.collate import connect4_collate_fn
from data.dataset_precomputed import TARGET_VALIDITY_MASK_CONTRACT
from models.connect4 import Connect4Model, roi_masks_to_tensor
from models.losses import Connect4Loss


def _tiny_config():
    return {
        "data": {
            "architecture_shape": [8, 8, 8],
            "num_frames": 2,
            "out_channels": 1,
        },
        "models": {
            "graphs": {"patch_size": [4, 4, 4], "k_neighbors": 2},
            "fusion": {
                "image_embed_dim": 3,
                "mask_embed_dim": 4,
                "roi_embed_dim": 5,
                "hidden_dim": 8,
                "output_dim": 6,
                "num_attention_layers": 2,
                "num_heads": 2,
                "dropout": 0.0,
                "num_rois": 2,
            },
            "dit": {
                "diffusion_domain": "connect4-token-latent-diffusion-v1",
                "input_size": [8, 8, 8],
                "patch_size": [4, 4, 4],
                "hidden_size": 12,
                "depth": 1,
                "num_heads": 3,
                "mlp_ratio": 2.0,
                "num_diffusion_steps": 20,
                "num_inference_steps": 2,
                "eta": 0.0,
            },
            "unet": {
                "base_channels": 2,
                "num_levels": 2,
                "use_temporal_attn": True,
                "temporal_attn_heads": 1,
                "use_checkpoint": False,
                "temporal_chunk": 0,
                "depth_slab_size": 0,
            },
        },
        "training": {
            "diffusion_weight": 1.0,
            "token_codec_reconstruction_weight": 1.0,
            "loss": {},
        },
    }


def _tiny_batch():
    patch_distributions = [
        {1: 0.75, 0: 0.25} if index % 2 == 0 else {2: 0.5, 0: 0.5}
        for index in range(8)
    ]
    roi_masks = torch.zeros(1, 2, 8, 8, 8)
    roi_masks[:, 0, :4] = 1
    roi_masks[:, 1, 4:] = 1
    return {
        "scan_id": ["tiny-scan-001"],
        "image_nodes": torch.randn(1, 8, 3),
        "mask_nodes": torch.randn(1, 8, 4),
        "roi_nodes": torch.randn(1, 2, 5),
        "dwi_matrix": torch.tensor([[[1.0, 0.3], [0.3, 1.0]]]),
        "structure_to_roi_idx": {1: 0, 2: 1},
        "patch_distributions": [patch_distributions],
        "roi_masks": roi_masks,
        "brain_mask": torch.ones(1, 1, 8, 8, 8),
        "target_validity_mask": torch.ones(1, 1, 8, 8, 8),
        "target_validity_mask_contract": [TARGET_VALIDITY_MASK_CONTRACT],
        "t1w": torch.randn(1, 1, 8, 8, 8),
        "fmri": torch.randn(1, 1, 2, 8, 8, 8),
    }


@pytest.mark.parametrize("strict", [True, False])
def test_v14_connect4_load_rejects_nonfinite_state_outside_dit(strict):
    source = Connect4Model(_tiny_config(), build_loss=False)
    state = copy.deepcopy(source.state_dict())
    state["decoder.output_head.weight"][0, 0, 0, 0, 0] = float("nan")
    destination = Connect4Model(_tiny_config(), build_loss=False)

    with pytest.raises(RuntimeError, match="decoder.output_head.weight"):
        destination.load_state_dict(state, strict=strict)
    assert destination.validate_model_state_invariants()["model_state_finite"]


def test_v14_connect4_runtime_gate_recursively_rejects_nested_decoder_state():
    model = Connect4Model(_tiny_config(), build_loss=False)
    model.auxiliary = nn.Module()
    model.auxiliary.deep = nn.Module()
    model.auxiliary.deep.token_decoder = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="recursively forbids"):
        model.validate_model_state_invariants()


def test_production_model_rejects_unmeasured_slab_before_construction():
    config = _tiny_config()
    config["data"]["protocol_profile"] = PAPER_PROTOCOL_PROFILE
    config["models"]["unet"].update(
        {
            "depth_slab_size": None,
            "depth_slab_sweep_summary_path": None,
            "depth_slab_sweep_summary_sha256": None,
            "depth_slab_sweep_summary_sha256_env": "",
            "depth_slab_ddp_smoke_path": None,
            "depth_slab_ddp_smoke_sha256": None,
            "depth_slab_ddp_smoke_sha256_env": "",
        }
    )
    with pytest.raises(RuntimeError, match="unresolved.*authenticated.*A100 sweep"):
        Connect4Model(config, build_loss=False)


class _DummyBiologicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = nn.Module()
        self.perceptual.fe = nn.Identity()

    def forward(self, prediction, target, **_):
        total = (prediction - target).square().mean()
        return {"dummy": total, "total": total}


class _TinyFrozenPerceptualExtractor(nn.Module):
    def forward(self, volume):
        return volume.mean(dim=(2, 3, 4, 5))


def test_connect4_tiny_training_and_ddim_inference_end_to_end():
    torch.manual_seed(12)
    model = Connect4Model(_tiny_config(), build_loss=False)
    model.loss_fn = _DummyBiologicalLoss()
    batch = _tiny_batch()

    model.train()
    model.set_training_sampling_epoch(0)
    output = model(batch)
    assert output["prediction"].shape == (1, 1, 2, 8, 8, 8)
    assert set(("biological_total", "diffusion", "total")) <= set(
        output["losses"]
    )
    output["losses"]["total"].backward()
    assert model.fusion.shared_encoder.convs[0].att_src.grad is not None
    for parameter in (
        model.dit.token_codec_raw_detail_rows,
        model.dit.epsilon_head.linear.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert isinstance(model.dit_to_unet.projection, nn.Conv3d)
    assert model.dit_to_unet.projection.kernel_size == (1, 1, 1)
    assert model.dit_to_unet.projection.weight.grad is not None
    assert model.decoder.output_head.weight.grad is not None

    model.eval()
    with torch.no_grad():
        patch_tokens = model.build_patch_tokens(batch, torch.device("cpu"))
        prediction = model.generate(
            patch_tokens,
            batch["t1w"],
            mask=output["brain_mask"],
            num_inference_steps=2,
            scan_ids=batch["scan_id"],
        )
    assert prediction.shape == (1, 1, 2, 8, 8, 8)
    assert torch.isfinite(prediction).all()


def test_t1_high_pass_detail_path_is_forbidden_in_synthesis():
    config = _tiny_config()
    config["models"]["unet"]["detail_path_enabled"] = True
    with pytest.raises(ValueError, match="T1 high-pass detail injection is forbidden"):
        Connect4Model(config, build_loss=False)

    config["models"]["unet"]["detail_path_enabled"] = False
    model = Connect4Model(config, build_loss=False).eval()
    batch = _tiny_batch()
    decoded = model.sampled_ddim_decode(
        batch,
        sampling_context="evaluation",
    )["prediction"]
    assert decoded.shape == batch["fmri"].shape
    assert torch.isfinite(decoded).all()


def test_roi_masks_cannot_substitute_for_explicit_structural_brain_mask():
    config = _tiny_config()
    config["models"]["fusion"]["num_rois"] = 32
    model = Connect4Model(config, build_loss=False).train()
    batch = _tiny_batch()
    batch["roi_masks"] = batch["roi_masks"].repeat(1, 16, 1, 1, 1)
    batch["roi_nodes"] = batch["roi_nodes"].repeat(1, 16, 1)
    batch.pop("brain_mask")
    assert batch["roi_masks"].shape[1] == 32
    with pytest.raises(ValueError, match="explicit authenticated structural"):
        model(batch)
    with pytest.raises(ValueError, match="missing fields.*brain_mask"):
        model.sampled_ddim_decode(batch, sampling_context="evaluation")


def test_sampling_rejects_absent_or_nonbinary_structural_brain_mask():
    model = Connect4Model(_tiny_config(), build_loss=False).eval()
    batch = _tiny_batch()
    patch_tokens = model.build_patch_tokens(batch, torch.device("cpu"))
    with pytest.raises(ValueError, match="explicit authenticated structural"):
        model.generate(
            patch_tokens,
            batch["t1w"],
            None,
            num_inference_steps=2,
            scan_ids=batch["scan_id"],
        )
    batch["brain_mask"] = torch.full((1, 1, 8, 8, 8), 0.5)
    with pytest.raises(ValueError, match="exactly binary"):
        model.sampled_ddim_decode(batch, sampling_context="evaluation")


def test_paired_target_requires_exact_target_validity_support():
    target = torch.ones(1, 1, 2, 8, 8, 8)
    target[..., 0, 0, 0] = 0
    structural = torch.ones(1, 1, 8, 8, 8)
    expected = torch.ones_like(structural)
    expected[..., 0, 0, 0] = 0
    observed = Connect4Model._normalise_target_validity_mask(
        expected, target, structural
    )
    assert torch.equal(observed, expected)

    with pytest.raises(ValueError, match="exact observed target support"):
        Connect4Model._normalise_target_validity_mask(
            torch.ones_like(expected), target, structural
        )
    outside_structural = structural.clone()
    outside_structural[..., 0, 0, 1] = 0
    with pytest.raises(ValueError, match="outside the structural"):
        Connect4Model._normalise_target_validity_mask(
            expected, target, outside_structural
        )


def test_model_rejects_validity_mask_without_target():
    model = Connect4Model(_tiny_config(), build_loss=False).eval()
    batch = _tiny_batch()
    batch.pop("fmri")
    with pytest.raises(ValueError, match="forbidden without a paired"):
        model(batch)


def test_model_requires_target_validity_support_in_every_configured_roi():
    model = Connect4Model(_tiny_config(), build_loss=False).train()
    batch = _tiny_batch()
    batch["fmri"][..., 4:, :, :] = 0
    batch["target_validity_mask"][..., 4:, :, :] = 0
    with pytest.raises(
        ValueError,
        match="every configured ROI must have non-empty target_validity_mask",
    ):
        model(batch)


def test_roi_foreground_outside_structural_brain_mask_is_rejected():
    model = Connect4Model(_tiny_config(), build_loss=False).train()
    batch = _tiny_batch()
    batch["brain_mask"][..., 0, 0, 0] = 0
    assert batch["roi_masks"][..., 0, 0, 0].any()
    with pytest.raises(ValueError, match="ROI foreground lies outside"):
        model(batch)
    with pytest.raises(ValueError, match="ROI foreground lies outside"):
        model.sampled_ddim_decode(batch, sampling_context="evaluation")


def test_target_blind_full_ddim_latent_is_independent_of_available_target():
    torch.manual_seed(13)
    model = Connect4Model(_tiny_config(), build_loss=False)
    batch = _tiny_batch()

    model.train()
    model.set_training_sampling_epoch(7)
    training = model.sampled_ddim_decode(batch)

    substituted = dict(batch)
    substituted["fmri"] = torch.full_like(batch["fmri"], 1234.0)
    substituted_training = model.sampled_ddim_decode(
        substituted
    )
    assert torch.equal(training["sampling_noise"], substituted_training["sampling_noise"])
    assert torch.equal(training["sampling_schedule"], substituted_training["sampling_schedule"])
    assert torch.equal(training["sampled_latent"], substituted_training["sampled_latent"])

    model.eval()
    inference = model.sampled_ddim_decode(
        substituted,
        sampling_context="train-epoch-00000007",
    )
    assert torch.equal(training["sampling_schedule"], inference["sampling_schedule"])
    assert torch.equal(training["sampling_noise"], inference["sampling_noise"])
    assert torch.equal(training["sampled_latent"], inference["sampled_latent"])
    assert "target" not in inspect.signature(model.sampled_ddim_decode).parameters
    assert "initial_noise" not in inspect.signature(
        model.sampled_ddim_decode
    ).parameters
    assert "initial_noise" not in inspect.signature(model.generate).parameters


def test_sampled_decoder_phase_detaches_denoiser_but_trains_tied_codec_and_unet():
    torch.manual_seed(14)
    model = Connect4Model(_tiny_config(), build_loss=False).train()
    model.set_training_sampling_epoch(2)
    batch = _tiny_batch()
    sampled = model.sampled_ddim_decode(batch)
    sampled["prediction"].square().mean().backward()

    decoder_grad = model.decoder.output_head.weight.grad
    assert decoder_grad is not None and torch.isfinite(decoder_grad).all()
    codec_grad = model.dit.token_codec_raw_detail_rows.grad
    assert codec_grad is not None and torch.isfinite(codec_grad).all()
    assert codec_grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.dit.blocks.parameters())
    assert all(
        parameter.grad is None for parameter in model.dit.epsilon_head.parameters()
    )
    assert all(parameter.grad is None for parameter in model.fusion.parameters())


def test_biological_objective_reaches_fusion_dit_and_decoder_parameter_groups():
    """Regression: the old no-grad sampled path left DiT/fusion gradients absent."""
    torch.manual_seed(140)
    model = Connect4Model(_tiny_config(), build_loss=False).train()
    model.loss_fn = Connect4Loss(
        perceptual_feature_extractor=_TinyFrozenPerceptualExtractor()
    )
    batch = _tiny_batch()
    batch["fmri"] = batch["fmri"].sigmoid()
    output = model(batch)
    assert {
        "ssim",
        "voxel",
        "volume",
        "region_hist",
        "temporal",
        "perceptual",
        "fc",
    } <= output["losses"].keys()
    assert all(
        torch.isfinite(output["losses"][name])
        for name in (
            "ssim",
            "voxel",
            "volume",
            "region_hist",
            "temporal",
            "perceptual",
            "fc",
        )
    )

    model.zero_grad(set_to_none=True)
    output["losses"]["weighted_biological_total"].backward(retain_graph=True)

    def has_finite_nonzero_gradient(module):
        return any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(parameter.grad.abs().sum() > 0)
            for parameter in module.parameters()
        )

    assert has_finite_nonzero_gradient(model.fusion)
    assert has_finite_nonzero_gradient(model.dit)
    assert has_finite_nonzero_gradient(model.decoder)
    for parameter in (
        model.dit.token_codec_raw_detail_rows,
        model.dit.epsilon_head.linear.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    # The anti-collapse temporal term by itself must reach the same three
    # learned parameter groups, not merely contribute a logged scalar.
    model.zero_grad(set_to_none=True)
    temporal_objective = (
        output["losses"]["biological_noise_weight"]
        * model.loss_fn.w["temporal"]
        * output["losses"]["temporal"]
    )
    temporal_objective.backward()
    assert has_finite_nonzero_gradient(model.fusion)
    assert has_finite_nonzero_gradient(model.dit)
    assert has_finite_nonzero_gradient(model.decoder)


def test_training_uses_bounded_single_step_x0_not_full_ddim(monkeypatch):
    torch.manual_seed(141)
    model = Connect4Model(_tiny_config(), build_loss=False).train()
    model.loss_fn = _DummyBiologicalLoss()

    def forbidden_sampler(*_args, **_kwargs):
        raise AssertionError("full DDIM must not run inside the training graph")

    monkeypatch.setattr(model, "sampled_ddim_decode", forbidden_sampler)
    output = model(_tiny_batch())
    weight = output["losses"]["biological_noise_weight"]
    assert weight.ndim == 0
    assert 0 < float(weight) <= 1
    assert torch.allclose(
        output["losses"]["weighted_biological_total"],
        weight * output["losses"]["biological_total"],
    )


def test_model_rejects_any_low_resolution_dit_reconstruction_bottleneck():
    config = _tiny_config()
    config["data"]["architecture_shape"] = [16, 16, 16]
    config["models"]["graphs"]["patch_size"] = [8, 8, 8]
    config["models"]["dit"]["input_size"] = [2, 2, 2]
    config["models"]["dit"]["patch_size"] = [2, 2, 2]
    config["models"]["dit"]["hidden_size"] = 8
    config["models"]["dit"]["num_heads"] = 2
    with pytest.raises(ValueError, match="complete configured D,H,W grid"):
        Connect4Model(config, build_loss=False)


def test_model_rejects_graph_dit_spatial_token_misalignment():
    config = _tiny_config()
    config["models"]["dit"]["patch_size"] = [8, 8, 8]
    with pytest.raises(ValueError, match="align one-to-one with graph patches"):
        Connect4Model(config, build_loss=False)


def test_training_noise_changes_by_epoch_but_is_fixed_per_scan_and_context():
    model = Connect4Model(_tiny_config(), build_loss=False)
    shape = (2, 16, 12)
    ids = ("scan-a", "scan-b")
    epoch_zero = model._deterministic_ddim_noise(
        ids, shape, device=torch.device("cpu"), dtype=torch.float32,
        sampling_context="train-epoch-00000000",
    )
    epoch_zero_again = model._deterministic_ddim_noise(
        ids, shape, device=torch.device("cpu"), dtype=torch.float32,
        sampling_context="train-epoch-00000000",
    )
    epoch_one = model._deterministic_ddim_noise(
        ids, shape, device=torch.device("cpu"), dtype=torch.float32,
        sampling_context="train-epoch-00000001",
    )
    reversed_ids = model._deterministic_ddim_noise(
        tuple(reversed(ids)), shape, device=torch.device("cpu"), dtype=torch.float32,
        sampling_context="train-epoch-00000000",
    )
    assert torch.equal(epoch_zero, epoch_zero_again)
    assert not torch.equal(epoch_zero, epoch_one)
    assert torch.equal(epoch_zero[0], reversed_ids[1])
    assert torch.equal(epoch_zero[1], reversed_ids[0])


def test_roi_masks_are_never_resampled_online():
    roi_masks = torch.zeros(1, 2, 7, 8, 8)
    with pytest.raises(ValueError, match="online resampling is forbidden"):
        roi_masks_to_tensor(roi_masks, (8, 8, 8), torch.device("cpu"))


def test_roi_mask_singleton_axes_are_not_silently_squeezed():
    malformed = [{0: torch.zeros(1, 8, 8, 8)}]
    with pytest.raises(ValueError, match=r"exactly \[D,H,W\]"):
        roi_masks_to_tensor(malformed, (8, 8, 8), torch.device("cpu"))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 2.0, 0.7, -3.0])
def test_roi_masks_reject_nonfinite_and_nonbinary_values(invalid):
    roi_masks = torch.zeros(1, 2, 8, 8, 8)
    roi_masks[0, 0, 0, 0, 0] = invalid
    message = "NaN or infinity" if not math.isfinite(invalid) else "exactly binary"
    with pytest.raises(ValueError, match=message):
        roi_masks_to_tensor(
            roi_masks,
            (8, 8, 8),
            torch.device("cpu"),
            expected_num_rois=2,
        )


def test_roi_mask_mapping_keys_are_canonical_and_consistent():
    volume = torch.zeros(8, 8, 8)
    with pytest.raises(ValueError, match="contiguous canonical"):
        roi_masks_to_tensor(
            [{0: volume, 2: volume}],
            (8, 8, 8),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="key sets differ"):
        roi_masks_to_tensor(
            [{0: volume}, {0: volume, 1: volume}],
            (8, 8, 8),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="exactly 0..1"):
        roi_masks_to_tensor(
            [{0: volume}],
            (8, 8, 8),
            torch.device("cpu"),
            expected_num_rois=2,
        )


def test_model_bound_roi_masks_must_be_spatially_disjoint():
    roi_masks = torch.zeros(1, 2, 8, 8, 8)
    roi_masks[:, :, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="spatially disjoint"):
        roi_masks_to_tensor(
            roi_masks,
            (8, 8, 8),
            torch.device("cpu"),
            expected_num_rois=2,
        )


def test_collate_rejects_nonuniform_functional_target_shapes():
    batch = [
        {"fmri": torch.zeros(1, 2, 8, 8, 8), "scan_id": "a"},
        {"fmri": torch.zeros(1, 3, 8, 8, 8), "scan_id": "b"},
    ]
    with pytest.raises(RuntimeError, match="stack expects each tensor"):
        connect4_collate_fn(batch)


def test_model_rejects_legacy_extra_singleton_conditioning_axes():
    with pytest.raises(ValueError, match="exactly"):
        Connect4Model._normalise_t1(
            torch.zeros(1, 1, 1, 8, 8, 8), 1, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="exactly"):
        Connect4Model._normalise_brain_mask(
            torch.ones(1, 1, 1, 8, 8, 8), 1, torch.device("cpu")
        )


def test_collate_rejects_mixed_sample_schemas_and_roi_maps():
    with pytest.raises(ValueError, match="different fields"):
        connect4_collate_fn([{"scan_id": "a"}, {"scan_id": "b", "x": 1}])
    with pytest.raises(ValueError, match="differs between subjects"):
        connect4_collate_fn([
            {"structure_to_roi_idx": {1: 0}},
            {"structure_to_roi_idx": {1: 1}},
        ])
