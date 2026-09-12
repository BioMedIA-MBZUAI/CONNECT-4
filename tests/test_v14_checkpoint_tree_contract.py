import copy

import pytest
import torch

from models.connect4 import Connect4Model
from models.dit4d_temporal import (
    DiT4DTemporal,
    TOKEN_LATENT_DIFFUSION_DOMAIN,
    _DIRECT_DIT_CODEC_STATE_PATHS,
    _validate_checkpoint_state_tree,
)


def _tiny_dit() -> DiT4DTemporal:
    return DiT4DTemporal(
        input_size=(2, 2, 2),
        in_channels=1,
        patch_size=(2, 2, 2),
        hidden_size=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        input_dim=4,
        t1_token_dim=4,
        num_temporal_frames=2,
        temporal_patch_size=1,
        num_diffusion_steps=4,
        spatial_window_size=(8, 8, 8),
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    )


def _tiny_connect4() -> Connect4Model:
    config = {
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
                "diffusion_domain": TOKEN_LATENT_DIFFUSION_DOMAIN,
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
    return Connect4Model(config, build_loss=False)


@pytest.fixture(params=("dit", "connect4"))
def model_factory(request):
    return _tiny_dit if request.param == "dit" else _tiny_connect4


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "legacy_component",
    ["clean_token_encoder", "token_decoder"],
)
def test_v18_both_model_surfaces_reject_nested_legacy_tensor_before_load(
    model_factory,
    strict,
    legacy_component,
):
    source = model_factory()
    state = copy.deepcopy(source.state_dict())
    state["unexpected_nested"] = {
        "safe": {legacy_component: {"weight": torch.ones(2, 2)}}
    }
    destination = model_factory()
    before = next(destination.parameters()).detach().clone()

    with pytest.raises(
        RuntimeError,
        match=(
            r"forbidden legacy independent token-codec state at "
            rf"'unexpected_nested\.safe\.{legacy_component}\.weight'"
        ),
    ):
        destination.load_state_dict(state, strict=strict)
    assert torch.equal(next(destination.parameters()), before)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_both_model_surfaces_reject_nested_non_tensor_metadata_before_load(
    model_factory,
    strict,
):
    source = model_factory()
    state = copy.deepcopy(source.state_dict())
    state["unexpected_nested"] = {"metadata": {"revision": "legacy"}}
    destination = model_factory()
    before = next(destination.parameters()).detach().clone()

    with pytest.raises(
        RuntimeError,
        match=r"unsupported str leaf at 'unexpected_nested\.metadata\.revision'",
    ):
        destination.load_state_dict(state, strict=strict)
    assert torch.equal(next(destination.parameters()), before)


@pytest.mark.parametrize(
    "nested,expected",
    [
        ({7: torch.ones(1)}, r"mapping key at 'extra' must be a string"),
        ({"items": [torch.ones(1)]}, r"unsupported list leaf at 'extra\.items'"),
        (
            {"deep": {"value": torch.tensor(float("nan"))}},
            r"non-finite tensor at 'extra\.deep\.value'",
        ),
    ],
)
def test_v18_shared_walker_rejects_nested_key_leaf_and_nonfinite_attacks(
    nested,
    expected,
):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state["extra"] = nested

    with pytest.raises(RuntimeError, match=expected):
        _validate_checkpoint_state_tree(
            state,
            label="V18 adversarial state",
            allowed_codec_state_paths=_DIRECT_DIT_CODEC_STATE_PATHS,
        )


def test_v18_shared_walker_rejects_mapping_cycles_with_full_path():
    state = copy.deepcopy(_tiny_dit().state_dict())
    cycle = {}
    cycle["self"] = cycle
    state["extra"] = cycle

    with pytest.raises(RuntimeError, match=r"cyclic mapping at 'extra\.self'"):
        _validate_checkpoint_state_tree(
            state,
            label="V18 adversarial state",
            allowed_codec_state_paths=_DIRECT_DIT_CODEC_STATE_PATHS,
        )
