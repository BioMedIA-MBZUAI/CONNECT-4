import copy

import pytest
import torch
import torch.nn as nn

from architecture_contract import (
    PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
    PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
    PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE,
    TOKEN_CODEC_STATE_VERSION,
    require_production_token_codec_shape,
    synthesis_architecture_contract,
)
from models.dit4d_temporal import (
    DiT4DTemporal,
    TOKEN_LATENT_DIFFUSION_DOMAIN,
    _validate_checkpoint_state_tree,
)
from training.train import _production_token_codec_shape_from_config


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


def test_v18_contract_binds_bounded_gauge_and_exact_production_shape():
    contract = synthesis_architecture_contract()

    assert TOKEN_CODEC_STATE_VERSION == 18
    assert contract["token_codec_state_version"] == 18
    assert contract["token_codec_raw_detail_maximum_singular_value"] == (
        TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE
    )
    assert contract["token_codec_raw_detail_maximum_condition_number"] == (
        TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER
    )
    assert contract["token_codec_raw_detail_maximum_rms_singular_value"] == (
        TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE
    )
    assert contract["production_token_codec_shape_record"] == {
        "hidden_size": PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
        "patch_value_dim": PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
        "compression_ratio": PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
    }
    assert require_production_token_codec_shape(
        protocol_profile="paper-a4-adni-fmriprep-v1",
        hidden_size=512,
        patch_value_dim=4096,
    ) == contract["production_token_codec_shape_record"]


def test_v18_checkpoint_shape_evidence_is_derived_from_exact_config():
    config = {
        "data": {
            "protocol_profile": "a4-native-recovery-v1",
            "out_channels": 1,
        },
        "models": {
            "dit": {
                "hidden_size": 512,
                "patch_size": [16, 16, 16],
                "temporal_patch_size": 1,
            }
        },
    }
    assert _production_token_codec_shape_from_config(config) == {
        "hidden_size": 512,
        "patch_value_dim": 4096,
        "compression_ratio": 8,
    }
    config["models"]["dit"]["hidden_size"] = 256
    with pytest.raises(ValueError, match="hidden_size must be exactly 512"):
        _production_token_codec_shape_from_config(config)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("scale", [1e6, 1e18])
def test_v18_rejects_uniform_unbounded_qr_gauge_in_strict_and_nonstrict_loads(
    strict,
    scale,
):
    source = _tiny_dit()
    attacked = copy.deepcopy(source.state_dict())
    attacked["token_codec_raw_detail_rows"].mul_(scale)

    with pytest.raises(RuntimeError, match="maximum singular value"):
        _tiny_dit().load_state_dict(attacked, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_rejects_anisotropic_qr_gauge_attack(strict):
    source = _tiny_dit()
    attacked = copy.deepcopy(source.state_dict())
    detail = torch.zeros_like(attacked["token_codec_raw_detail_rows"])
    detail[0, 0] = 1e12
    detail[1, 1] = 1.0
    detail[2, 2] = 0.0501
    attacked["token_codec_raw_detail_rows"] = detail

    with pytest.raises(RuntimeError, match="maximum singular value|condition number"):
        _tiny_dit().load_state_dict(attacked, strict=strict)


@pytest.mark.parametrize("attack", ["uniform-1e6", "uniform-1e18", "anisotropic"])
def test_v18_post_adamw_gate_rejects_every_historical_qr_gauge_attack(attack):
    model = _tiny_dit()
    with torch.no_grad():
        if attack.startswith("uniform"):
            scale = 1e6 if attack.endswith("1e6") else 1e18
            model.token_codec_raw_detail_rows.mul_(scale)
        else:
            detail = torch.zeros_like(model.token_codec_raw_detail_rows)
            detail[0, 0] = 1e12
            detail[1, 1] = 1.0
            detail[2, 2] = 0.0501
            model.token_codec_raw_detail_rows.copy_(detail)
    optimizer = torch.optim.AdamW(
        [model.token_codec_raw_detail_rows],
        lr=1e-3,
        weight_decay=1e-2,
    )
    model.token_codec_raw_detail_rows.grad = torch.ones_like(
        model.token_codec_raw_detail_rows
    )
    optimizer.step()

    with pytest.raises(RuntimeError, match="singular value|condition number"):
        model.validate_model_state_invariants()


@pytest.mark.parametrize(
    "diagonal,match",
    [
        ([17.0, 1.0, 1.0], "maximum singular value"),
        ([2.0, 0.09, 0.09], "condition number"),
        ([5.0, 5.0, 5.0], "RMS singular value"),
    ],
)
def test_v18_each_maximum_gauge_bound_is_independently_executable(
    diagonal,
    match,
):
    state = copy.deepcopy(_tiny_dit().state_dict())
    detail = torch.zeros_like(state["token_codec_raw_detail_rows"])
    for index, value in enumerate(diagonal):
        detail[index, index] = value
    state["token_codec_raw_detail_rows"] = detail

    with pytest.raises(RuntimeError, match=match):
        _tiny_dit().load_state_dict(state, strict=False)


def test_v18_bounded_adamw_step_produces_nonzero_effective_encoder_update():
    torch.manual_seed(1301)
    model = _tiny_dit()
    optimizer = torch.optim.AdamW(
        [model.token_codec_raw_detail_rows],
        lr=1e-2,
        weight_decay=1e-2,
    )
    before = model.effective_token_codec_encoder().detach().clone()
    probe = torch.randn_like(before)

    objective = (model.effective_token_codec_encoder() * probe).sum()
    objective.backward()
    assert model.token_codec_raw_detail_rows.grad is not None
    assert bool(torch.count_nonzero(model.token_codec_raw_detail_rows.grad))
    optimizer.step()
    report = model.validate_model_state_invariants()
    after = model.effective_token_codec_encoder().detach()

    assert report["model_state_finite"]
    assert not torch.equal(after, before)
    assert float((after - before).abs().amax()) > 0.0


@pytest.mark.parametrize("legacy_version", [10, 11, 12, 13, 14, 15, 16, 17])
@pytest.mark.parametrize("strict", [True, False])
def test_v18_rejects_all_legacy_codec_markers_even_nonstrict(
    legacy_version,
    strict,
):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state["_token_codec_state_version"].fill_(legacy_version)
    with pytest.raises(RuntimeError, match="V10/V11/V12/V13/V14/V15/V16/V17"):
        _tiny_dit().load_state_dict(state, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_rejects_nested_legacy_decoder_checkpoint_state(strict):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state["auxiliary.deep.token_decoder.weight"] = torch.ones(2, 2)
    with pytest.raises(RuntimeError, match="forbidden legacy independent"):
        _tiny_dit().load_state_dict(state, strict=strict)


@pytest.mark.parametrize(
    "kind,name",
    [
        ("module", "token_decoder"),
        ("parameter", "clean_token_encoder_weight"),
        ("buffer", "token_decoder_scale"),
    ],
)
def test_v18_recursively_rejects_nested_runtime_modules_parameters_and_buffers(
    kind,
    name,
):
    model = _tiny_dit()
    auxiliary = nn.Module()
    deep = nn.Module()
    auxiliary.add_module("deep", deep)
    model.add_module("auxiliary", auxiliary)
    if kind == "module":
        deep.add_module(name, nn.Linear(2, 2))
    elif kind == "parameter":
        deep.register_parameter(name, nn.Parameter(torch.ones(1)))
    else:
        deep.register_buffer(name, torch.ones(1))

    with pytest.raises(RuntimeError, match="recursively forbids"):
        model.validate_model_state_invariants()


@pytest.mark.parametrize("strict", [True, False])
def test_v18_model_wide_finite_gate_rejects_nan_epsilon_head_before_load(strict):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state["epsilon_head.linear.weight"][0, 0] = float("nan")
    destination = _tiny_dit()
    before = destination.epsilon_head.linear.weight.detach().clone()

    with pytest.raises(RuntimeError, match="non-finite.*epsilon_head.linear.weight"):
        destination.load_state_dict(state, strict=strict)
    assert torch.equal(destination.epsilon_head.linear.weight, before)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_finite_gate_recurses_into_unexpected_nested_checkpoint_state(strict):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state["unexpected_container"] = {
        "nested": {"value": torch.tensor([1.0, float("nan")])}
    }

    with pytest.raises(
        RuntimeError,
        match=r"unexpected_container\.nested\.value",
    ):
        _tiny_dit().load_state_dict(state, strict=strict)


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"integer_only": torch.tensor(14, dtype=torch.int64)},
        {"zero_length_float": torch.empty(0)},
    ],
)
def test_v18_empty_or_non_tensor_only_state_cannot_claim_finite_audit(state):
    with pytest.raises(RuntimeError, match=r"no (floating or complex )?tensor state"):
        _validate_checkpoint_state_tree(state, label="adversarial empty audit")


def test_v18_non_tensor_only_state_fails_at_the_exact_nested_path():
    state = {"metadata": {"items": "finite"}}
    with pytest.raises(
        RuntimeError,
        match=r"unsupported str leaf at 'metadata\.items'",
    ):
        _validate_checkpoint_state_tree(state, label="adversarial metadata audit")


def test_v18_cyclic_nested_checkpoint_state_fails_closed():
    state = copy.deepcopy(_tiny_dit().state_dict())
    cycle = {}
    cycle["self"] = cycle
    state["unexpected_cycle"] = cycle

    with pytest.raises(
        RuntimeError,
        match=r"cyclic mapping at 'unexpected_cycle\.self'",
    ):
        _tiny_dit().load_state_dict(state, strict=False)


def test_v18_model_wide_runtime_gate_rejects_nan_outside_codec():
    model = _tiny_dit()
    with torch.no_grad():
        model.epsilon_head.linear.bias[0] = float("nan")
    with pytest.raises(RuntimeError, match="epsilon_head.linear.bias"):
        model.validate_model_state_invariants()


def test_v18_checkpoint_roundtrip_retains_finite_bounded_codec_state():
    source = _tiny_dit()
    destination = _tiny_dit()
    result = destination.load_state_dict(source.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(
        destination.effective_token_codec_encoder(),
        source.effective_token_codec_encoder(),
    )
    assert destination.validate_model_state_invariants()["model_state_finite"]
