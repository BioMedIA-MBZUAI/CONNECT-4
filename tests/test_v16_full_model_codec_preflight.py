import copy

import pytest
import torch

from architecture_contract import TOKEN_CODEC_STATE_VERSION
from tests.test_v14_checkpoint_tree_contract import _tiny_connect4, _tiny_dit


_MARKER_PATH = "dit._token_codec_state_version"
_RAW_DETAIL_PATH = "dit.token_codec_raw_detail_rows"


def _snapshot(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def _assert_snapshot(model, expected):
    observed = model.state_dict()
    assert tuple(observed) == tuple(expected)
    for name, tensor in expected.items():
        assert torch.equal(observed[name], tensor), name


def _make_parent_mutation_visible(state, destination):
    destination_state = destination.state_dict()
    fusion_names = [name for name in state if name.startswith("fusion.")]
    assert len(fusion_names) >= 20
    for index, name in enumerate(fusion_names):
        state[name] = torch.full_like(state[name], float(index + 1))
    assert any(
        not torch.equal(state[name], destination_state[name])
        for name in fusion_names
    )


def _malform_full_model_state(state, attack):
    if attack == "missing-marker":
        state.pop(_MARKER_PATH)
    elif attack == "missing-raw-detail":
        state.pop(_RAW_DETAIL_PATH)
    elif attack == "wrong-shape":
        state[_RAW_DETAIL_PATH] = state[_RAW_DETAIL_PATH][:, :-1].clone()
    elif attack == "integer-dtype":
        state[_RAW_DETAIL_PATH] = torch.ones_like(
            state[_RAW_DETAIL_PATH],
            dtype=torch.int64,
        )
    elif attack == "rank-deficient":
        state[_RAW_DETAIL_PATH][1].copy_(state[_RAW_DETAIL_PATH][0])
    elif attack == "unbounded-uniform-gauge":
        state[_RAW_DETAIL_PATH].mul_(1e6)
    elif attack == "anisotropic-rms":
        detail = torch.zeros_like(state[_RAW_DETAIL_PATH])
        diagonal = torch.full(
            (detail.shape[0],),
            5.0,
            dtype=detail.dtype,
            device=detail.device,
        )
        diagonal[-3:] = 0.2
        detail[
            torch.arange(detail.shape[0]),
            torch.arange(detail.shape[0]),
        ] = diagonal
        state[_RAW_DETAIL_PATH] = detail
    else:  # pragma: no cover - test helper contract
        raise AssertionError(attack)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize(
    "attack,expected",
    [
        ("missing-marker", "state marker"),
        ("missing-raw-detail", "no token-codec raw detail rows"),
        ("wrong-shape", "must have shape"),
        ("integer-dtype", "real floating-point dtype"),
        ("rank-deficient", "minimum singular value"),
        ("unbounded-uniform-gauge", "maximum singular value"),
        ("anisotropic-rms", "condition number|RMS singular value"),
    ],
)
def test_v18_full_connect4_codec_preflight_rejects_before_any_parent_mutation(
    strict,
    attack,
    expected,
):
    source = _tiny_connect4()
    state = copy.deepcopy(source.state_dict())
    destination = _tiny_connect4()
    before = _snapshot(destination)
    _make_parent_mutation_visible(state, destination)
    _malform_full_model_state(state, attack)

    with pytest.raises(RuntimeError, match=expected):
        destination.load_state_dict(state, strict=strict)
    _assert_snapshot(destination, before)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_direct_dit_codec_preflight_rejects_before_any_mutation(strict):
    state = copy.deepcopy(_tiny_dit().state_dict())
    state.pop("_token_codec_state_version")
    destination = _tiny_dit()
    before = _snapshot(destination)
    state["condition_type"] = torch.full_like(state["condition_type"], 19.0)
    assert not torch.equal(state["condition_type"], before["condition_type"])

    with pytest.raises(RuntimeError, match="state marker"):
        destination.load_state_dict(state, strict=strict)
    _assert_snapshot(destination, before)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("factory", [_tiny_dit, _tiny_connect4])
def test_v18_valid_codec_states_load_on_both_public_surfaces(factory, strict):
    source = factory()
    state = copy.deepcopy(source.state_dict())
    destination = factory()

    result = destination.load_state_dict(state, strict=strict)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert destination.validate_token_codec_invariants()["state_version"] == (
        TOKEN_CODEC_STATE_VERSION
    )
    for name, expected in state.items():
        assert torch.equal(destination.state_dict()[name], expected), name
