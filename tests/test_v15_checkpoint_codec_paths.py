import copy

import pytest
import torch

from architecture_contract import TOKEN_CODEC_STATE_VERSION
from tests.test_v14_checkpoint_tree_contract import _tiny_connect4, _tiny_dit


@pytest.fixture(params=("dit", "connect4"))
def model_surface(request):
    if request.param == "dit":
        return _tiny_dit, "", "token_codec_raw_detail_rows"
    return _tiny_connect4, "dit.", "dit.token_codec_raw_detail_rows"


def _state_snapshot(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def _assert_state_unchanged(model, snapshot):
    observed = model.state_dict()
    assert tuple(observed) == tuple(snapshot)
    for name, expected in snapshot.items():
        assert torch.equal(observed[name], expected), name


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("encoding", ["dotted", "nested"])
def test_v18_rejects_unexpected_finite_v13_codec_subtree_before_mutation(
    model_surface,
    strict,
    encoding,
):
    factory, _, raw_detail_path = model_surface
    source = factory()
    state = copy.deepcopy(source.state_dict())
    raw_detail = state[raw_detail_path].detach().clone()
    if encoding == "dotted":
        state["unexpected_codec._token_codec_state_version"] = torch.tensor(
            13,
            dtype=torch.int64,
        )
        state["unexpected_codec.token_codec_raw_detail_rows"] = raw_detail
    else:
        state["unexpected_codec"] = {
            "_token_codec_state_version": torch.tensor(13, dtype=torch.int64),
            "token_codec_raw_detail_rows": raw_detail,
        }

    destination = factory()
    before = _state_snapshot(destination)
    with pytest.raises(
        RuntimeError,
        match=(
            r"token-codec state at unexpected path "
            r"'unexpected_codec\._token_codec_state_version'"
        ),
    ):
        destination.load_state_dict(state, strict=strict)
    _assert_state_unchanged(destination, before)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("legacy_version", [10, 11, 12, 13, 14, 15, 16, 17])
def test_v18_rejects_wrong_marker_at_allowed_surface_path_before_mutation(
    model_surface,
    strict,
    legacy_version,
):
    factory, prefix, _ = model_surface
    state = copy.deepcopy(factory().state_dict())
    marker_path = f"{prefix}_token_codec_state_version"
    state[marker_path] = torch.tensor(legacy_version, dtype=torch.int64)

    destination = factory()
    before = _state_snapshot(destination)
    with pytest.raises(
        RuntimeError,
        match=r"not V18.*V10/V11/V12/V13/V14/V15/V16/V17",
    ):
        destination.load_state_dict(state, strict=strict)
    _assert_state_unchanged(destination, before)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("encoding", ["dotted", "nested"])
def test_v18_rejects_unexpected_current_codec_marker_and_raw_rows(
    model_surface,
    strict,
    encoding,
):
    factory, _, raw_detail_path = model_surface
    state = copy.deepcopy(factory().state_dict())
    raw_detail = state[raw_detail_path].detach().clone()
    if encoding == "dotted":
        state["shadow._token_codec_state_version"] = torch.tensor(
            TOKEN_CODEC_STATE_VERSION,
            dtype=torch.int64,
        )
        state["shadow.token_codec_raw_detail_rows"] = raw_detail
    else:
        state["shadow"] = {
            "_token_codec_state_version": torch.tensor(
                TOKEN_CODEC_STATE_VERSION,
                dtype=torch.int64,
            ),
            "token_codec_raw_detail_rows": raw_detail,
        }

    destination = factory()
    before = _state_snapshot(destination)
    with pytest.raises(RuntimeError, match="token-codec state at unexpected path"):
        destination.load_state_dict(state, strict=strict)
    _assert_state_unchanged(destination, before)


def test_v18_rejects_unexpected_raw_detail_without_marker_on_both_surfaces(
    model_surface,
):
    factory, _, raw_detail_path = model_surface
    state = copy.deepcopy(factory().state_dict())
    state["shadow.token_codec_raw_detail_rows"] = state[
        raw_detail_path
    ].detach().clone()

    destination = factory()
    before = _state_snapshot(destination)
    with pytest.raises(
        RuntimeError,
        match=r"unexpected path 'shadow\.token_codec_raw_detail_rows'",
    ):
        destination.load_state_dict(state, strict=False)
    _assert_state_unchanged(destination, before)


@pytest.mark.parametrize("strict", [True, False])
def test_v18_connect4_rejects_nested_encoding_of_logically_allowed_dit_paths(
    strict,
):
    state = copy.deepcopy(_tiny_connect4().state_dict())
    state["dit"] = {
        "_token_codec_state_version": torch.tensor(
            TOKEN_CODEC_STATE_VERSION,
            dtype=torch.int64,
        ),
        "token_codec_raw_detail_rows": state[
            "dit.token_codec_raw_detail_rows"
        ].detach().clone(),
    }

    destination = _tiny_connect4()
    before = _state_snapshot(destination)
    with pytest.raises(
        RuntimeError,
        match=r"unexpected path 'dit\._token_codec_state_version'",
    ):
        destination.load_state_dict(state, strict=strict)
    _assert_state_unchanged(destination, before)
