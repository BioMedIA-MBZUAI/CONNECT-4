import copy

import pytest
import torch

from architecture_contract import TOKEN_CODEC_STATE_VERSION
from tests.test_v14_checkpoint_tree_contract import _tiny_connect4, _tiny_dit


_CODEC_STATE_SUFFIXES = (
    "_token_codec_state_version",
    "token_codec_raw_detail_rows",
)


def _complete_snapshot(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def _assert_complete_snapshot(model, expected):
    observed = model.state_dict()
    assert tuple(observed) == tuple(expected)
    for name, tensor in expected.items():
        assert torch.equal(observed[name], tensor), name


def _make_preflight_mutation_visible(state, destination):
    destination_state = destination.state_dict()
    changed = []
    for index, (name, tensor) in enumerate(tuple(state.items())):
        if name.endswith(_CODEC_STATE_SUFFIXES) or not (
            tensor.is_floating_point() or tensor.is_complex()
        ):
            continue
        replacement = torch.full_like(tensor, float(index + 101))
        state[name] = replacement
        if not torch.equal(replacement, destination_state[name]):
            changed.append(name)
    assert changed
    return tuple(changed)


@pytest.mark.parametrize("factory", [_tiny_dit, _tiny_connect4])
@pytest.mark.parametrize("strict", [True, False])
def test_v18_finite_tensor_only_nested_mapping_rejects_before_any_mutation(
    factory,
    strict,
):
    state = copy.deepcopy(factory().state_dict())
    destination = factory()
    before = _complete_snapshot(destination)
    changed_source_names = _make_preflight_mutation_visible(state, destination)
    state["unexpected_nested"] = {
        "safe": {"weight": torch.ones(2, 2, dtype=torch.float32)}
    }

    with pytest.raises(
        RuntimeError,
        match=(
            r"mapping-valued state entry at 'unexpected_nested\.safe'; "
            r"supported PyTorch state dicts must be flat string-to-tensor mappings"
        ),
    ):
        destination.load_state_dict(state, strict=strict)

    _assert_complete_snapshot(destination, before)
    assert all(
        not torch.equal(state[name], before[name])
        for name in changed_source_names
    )


@pytest.mark.parametrize("factory", [_tiny_dit, _tiny_connect4])
@pytest.mark.parametrize("strict", [True, False])
def test_v18_valid_flat_state_dict_loads_on_both_public_surfaces(factory, strict):
    state = copy.deepcopy(factory().state_dict())
    assert state
    assert all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state.items()
    )
    destination = factory()

    result = destination.load_state_dict(state, strict=strict)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert destination.validate_token_codec_invariants()["state_version"] == (
        TOKEN_CODEC_STATE_VERSION
    )
    _assert_complete_snapshot(destination, state)
