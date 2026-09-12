import torch
import pytest

import models.fusion as fusion_module
from graphs.hypergraph import HypergraphBuilder
from graphs.image_graph import ImageGraphBuilder
from graphs.mask_graph import MaskGraphBuilder
from graphs.roi_graph import ROIGraphBuilder
from models.fusion import (
    GroupAwareNodes2Token,
    MultiModalFusion,
    Sinusoidal3DPositionEncoding,
)


def test_image_graph_uses_raw_chebyshev_distance_weights():
    builder = ImageGraphBuilder(k_neighbors=1)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    adjacency = builder.build_adjacency(positions, torch.device("cpu"))

    assert adjacency[0, 1].item() == pytest.approx(1.0)
    assert adjacency[1, 2].item() == pytest.approx(2.0)
    assert adjacency[0, 2].item() == 0.0
    assert torch.equal(adjacency, adjacency.T)


def test_image_graph_and_nodes2token_share_even_patch_geometric_centres():
    positions = ImageGraphBuilder(patch_size=(2, 4, 6)).get_patch_positions(
        (4, 8, 12)
    )
    expected = torch.tensor(
        [
            [0.5, 1.5, 2.5],
            [0.5, 1.5, 8.5],
            [0.5, 5.5, 2.5],
            [0.5, 5.5, 8.5],
            [2.5, 1.5, 2.5],
            [2.5, 1.5, 8.5],
            [2.5, 5.5, 2.5],
            [2.5, 5.5, 8.5],
        ]
    )
    assert torch.equal(positions, expected)

    nodes2token = GroupAwareNodes2Token(4, 6, dropout=0.0).eval()
    empty_patch_tokens = nodes2token(
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        num_patches=positions.shape[0],
        patch_positions=positions,
    )
    assert torch.equal(
        empty_patch_tokens,
        nodes2token.position_encoding(expected),
    )


def test_roi_graph_uses_dwi_matrix_without_rewriting_edges():
    dwi = torch.tensor([[0.0, 0.2], [0.7, 0.0]])
    _, adjacency = ROIGraphBuilder(num_rois=2)(
        torch.randn(1, 2, 3), dwi_matrix=dwi,
    )
    assert torch.equal(adjacency, dwi)


def test_mask_graph_does_not_invent_non_dwi_self_edges():
    dwi = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    adjacency = MaskGraphBuilder().build_adjacency_from_dwi(
        2, [{1: 1.0}, {2: 1.0}], dwi, {1: 0, 2: 1},
    )
    assert adjacency[0, 1].item() == pytest.approx(0.5)
    assert torch.equal(adjacency.diagonal(), torch.zeros(2))


def test_mask_graph_preserves_directed_dwi_coefficients():
    dwi = torch.tensor([[0.0, 0.2], [0.8, 0.0]])
    adjacency = MaskGraphBuilder().build_adjacency_from_dwi(
        2, [{10: 1.0}, {20: 1.0}], dwi, {10: 0, 20: 1},
    )
    assert adjacency[0, 1].item() == pytest.approx(0.2)
    assert adjacency[1, 0].item() == pytest.approx(0.8)


def test_hypergraph_contains_only_positive_foreground_roi_edges():
    builder = HypergraphBuilder(num_patches=3, num_rois=3)
    distributions = [
        {0: 0.6, 1: 0.4, 2: 0.0},
        {0: 1.0},
        {2: 0.2},
    ]
    index, weights = builder.build_hyperedges(
        distributions, {1: 0, 2: 1, 3: 2}, torch.device("cpu")
    )

    assert torch.allclose(weights, torch.tensor([0.4, 0.2]))
    assert torch.equal(builder.patch_ids_from_index(index), torch.tensor([0, 2]))
    assert torch.equal(index[:, index[1] == 0][0], torch.tensor([0, 3, 6]))
    assert torch.equal(index[:, index[1] == 1][0], torch.tensor([2, 5, 7]))


def test_hypergraph_rejects_nonfinite_coverage_before_filtering():
    builder = HypergraphBuilder(num_patches=1, num_rois=1)
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        builder.build_hyperedges(
            [{0: float("nan")}], {1: 0}, torch.device("cpu")
        )


def test_hypergraph_rejects_negative_fractional_coverage():
    builder = HypergraphBuilder(num_patches=1, num_rois=1)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        builder.build_hyperedges(
            [{1: -0.1}], {1: 0}, torch.device("cpu")
        )


def test_nodes2token_is_group_softmax_equation_and_keeps_empty_patch():
    module = GroupAwareNodes2Token(2, 2, dropout=0.0).eval()
    with torch.no_grad():
        module.projection.weight.copy_(torch.eye(2))
        module.projection.bias.zero_()
        module.attention_score.weight.copy_(torch.tensor([[1.0, 0.0]]))

    h = torch.tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
    patch_ids = torch.tensor([0, 0, 2])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    )
    tokens = module(
        h,
        patch_ids,
        num_patches=3,
        patch_positions=positions,
    )
    content = tokens - module.position_encoding(positions)

    scores = torch.tanh(h[:2])[:, 0]
    expected_patch_zero = (torch.softmax(scores, dim=0)[:, None] * h[:2]).sum(0)
    assert torch.allclose(content[0], expected_patch_zero)
    assert torch.equal(content[1], torch.zeros(2))
    assert torch.equal(content[2], h[2])


def test_hyperedge_reductions_accept_fp32_softmax_inside_bf16_execution(
    monkeypatch,
):
    """PyG fallbacks may promote segment-softmax output under autocast."""
    original_segment_softmax = fusion_module.segment_softmax

    def fp32_segment_softmax(*args, **kwargs):
        return original_segment_softmax(*args, **kwargs).float()

    monkeypatch.setattr(
        fusion_module,
        "segment_softmax",
        fp32_segment_softmax,
    )

    nodes2token = GroupAwareNodes2Token(2, 2, dropout=0.0).to(
        dtype=torch.bfloat16
    )
    hyperedge_nodes = torch.tensor(
        [[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    patch_ids = torch.tensor([0, 0, 2])
    positions = torch.zeros(3, 3)
    tokens = nodes2token(
        hyperedge_nodes,
        patch_ids,
        num_patches=3,
        patch_positions=positions,
    )
    assert tokens.dtype == torch.bfloat16
    assert torch.isfinite(tokens).all()

    fusion = _tiny_fusion().to(dtype=torch.bfloat16)
    nodes = torch.tensor(
        [[0.0] * 8, [1.0] * 8, [2.0] * 8],
        dtype=torch.bfloat16,
    )
    incidence = torch.tensor([[0, 1, 2], [0, 0, 0]])
    representations = fusion._hyperedge_representations(
        nodes,
        incidence,
        num_hyperedges=1,
    )
    assert representations.dtype == torch.bfloat16
    assert torch.isfinite(representations).all()


def test_3d_position_is_final_token_for_patch_without_roi():
    module = GroupAwareNodes2Token(4, 6, dropout=0.0).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]
    )
    tokens = module(
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        num_patches=3,
        patch_positions=positions,
    )
    expected = module.position_encoding(positions)
    assert torch.equal(tokens, expected)
    assert not torch.equal(tokens[0], tokens[1])


def _tiny_fusion():
    return MultiModalFusion(
        image_embed_dim=3,
        mask_embed_dim=4,
        roi_embed_dim=5,
        hidden_dim=8,
        output_dim=6,
        num_attention_layers=2,
        num_heads=2,
        dropout=0.0,
        num_rois=1,
    ).eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_full_fusion_cuda_bf16_forward_backward_and_optimizer_step(monkeypatch):
    """Qualify the exact job-215808 mixed-precision failure path on CUDA."""
    original_segment_softmax = fusion_module.segment_softmax

    def fp32_segment_softmax(*args, **kwargs):
        return original_segment_softmax(*args, **kwargs).float()

    monkeypatch.setattr(
        fusion_module,
        "segment_softmax",
        fp32_segment_softmax,
    )
    device = torch.device("cuda", 0)
    fusion = _tiny_fusion().train().to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-4)
    image = torch.randn(1, 2, 3, device=device)
    mask = torch.randn(1, 2, 4, device=device)
    roi = torch.randn(1, 1, 5, device=device)
    adjacency = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0]]],
        device=device,
    )
    roi_adjacency = torch.zeros(1, 1, 1, device=device)
    incidence = torch.tensor(
        [[0, 2, 4, 0, 2, 4], [0, 0, 0, 1, 1, 1]],
        device=device,
    )
    weights = torch.tensor([0.25, 0.75], device=device)
    patch_ids = torch.tensor([0, 0], device=device)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        output = fusion(
            image,
            adjacency,
            mask,
            adjacency,
            roi,
            roi_adjacency,
            incidence,
            weights,
            patch_ids,
            patch_positions=positions,
        )
        loss = output.float().square().mean()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in fusion.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in fusion.parameters())


def test_fusion_has_one_shared_encoder_and_hyperedge_incident_messages():
    fusion = _tiny_fusion()
    assert fusion.image_encoder is fusion.mask_encoder is fusion.roi_encoder
    state_keys = tuple(fusion.state_dict())
    assert any(key.startswith("shared_encoder.") for key in state_keys)
    assert not any(key.startswith("image_encoder.") for key in state_keys)
    assert not any(key.startswith("mask_encoder.") for key in state_keys)
    assert not any(key.startswith("roi_encoder.") for key in state_keys)

    zero_patch_adj = torch.zeros(1, 1, 1)
    zero_roi_adj = torch.zeros(1, 1, 1)
    incidence = torch.tensor([[0, 1, 2], [0, 0, 0]])
    edge_index, edge_weight, _ = fusion._build_shared_edges(
        zero_patch_adj,
        zero_patch_adj,
        zero_roi_adj,
        incidence,
        torch.tensor([0.25]),
        batch_size=1,
    )
    pairs = set(map(tuple, edge_index.T.tolist()))
    assert pairs == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    assert torch.allclose(edge_weight, torch.full((6,), 0.25))


def test_fusion_derives_real_patch_groups_and_emits_all_patch_tokens():
    torch.manual_seed(0)
    fusion = _tiny_fusion()
    image = torch.randn(1, 2, 3)
    mask = torch.randn(1, 2, 4)
    roi = torch.randn(1, 1, 5)
    adjacency = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    roi_adjacency = torch.zeros(1, 1, 1)
    # Two ROI hyperedges both belong to patch 0. Patch 1 has no ROI.
    incidence = torch.tensor(
        [[0, 2, 4, 0, 2, 4], [0, 0, 0, 1, 1, 1]]
    )
    weights = torch.tensor([0.25, 0.75])
    wrong_historical_ids = torch.arange(2)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="disagree"):
        fusion(
            image,
            adjacency,
            mask,
            adjacency,
            roi,
            roi_adjacency,
            incidence,
            weights,
            wrong_historical_ids,
            patch_positions=positions,
        )
    out_derived = fusion(
        image,
        adjacency,
        mask,
        adjacency,
        roi,
        roi_adjacency,
        incidence,
        weights,
        torch.tensor([0, 0]),
        patch_positions=positions,
    )
    assert out_derived.shape == (1, 2, 6)
    expected_empty = Sinusoidal3DPositionEncoding(6)(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )[1]
    assert torch.allclose(out_derived[0, 1], expected_empty)
