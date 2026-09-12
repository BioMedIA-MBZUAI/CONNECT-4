"""Paper-faithful multimodal graph and hyperedge fusion.

The three modalities first enter a common hidden space. One *shared*
multi-head graph-attention encoder is then applied to a unified edge set made
from both within-modality graph edges and pairwise messages induced by every
incident hyperedge. Finally, group-aware Nodes2Token implements

    z_i = sum_{r in E_i} softmax_r(s_r) h_r + PE_3D(i)

for every image patch, including positional-only tokens for patches that have
no foreground ROI hyperedge.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, softmax as segment_softmax, to_dense_batch


class GraphAttentionEncoder(nn.Module):
    """A residual multi-layer, multi-head graph-attention encoder.

    ``edge_weight`` is passed as a scalar edge feature. In particular, raw
    Chebyshev distances from the image graph are not converted to inverse
    similarities or otherwise hand-normalised before the learned edge map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if num_layers > 1 and hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_layers = num_layers
        self.dropout = float(dropout)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            last = layer == num_layers - 1
            concat = not last
            if concat:
                if out_dim % num_heads:
                    raise ValueError(
                        f"layer {layer} output dimension {out_dim} is not divisible "
                        f"by {num_heads} heads"
                    )
                out_channels = out_dim // num_heads
            else:
                out_channels = out_dim
            self.convs.append(
                GATConv(
                    in_dim,
                    out_channels,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))

    @staticmethod
    def _expand_single_graph_edges(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_size: int,
        nodes_per_graph: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size == 1 or edge_index.numel() == 0:
            return edge_index, edge_weight
        if int(edge_index.max().item()) >= nodes_per_graph:
            return edge_index, edge_weight
        indices = [edge_index + b * nodes_per_graph for b in range(batch_size)]
        return torch.cat(indices, dim=1), edge_weight.repeat(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        was_dense = x.ndim == 3
        if x.ndim not in (2, 3):
            raise ValueError("x must have shape [N,D] or [B,N,D]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must have one scalar per edge")
        if not torch.isfinite(x).all() or not torch.isfinite(edge_weight).all():
            raise ValueError("graph features and edge weights must be finite")

        if was_dense:
            batch_size, nodes_per_graph, feature_dim = x.shape
            x = x.reshape(batch_size * nodes_per_graph, feature_dim)
            edge_index, edge_weight = self._expand_single_graph_edges(
                edge_index, edge_weight, batch_size, nodes_per_graph
            )
            if batch is None:
                batch = torch.arange(batch_size, device=x.device).repeat_interleave(
                    nodes_per_graph
                )

        if edge_index.numel() == 0:
            node_ids = torch.arange(x.shape[0], device=x.device)
            edge_index = torch.stack((node_ids, node_ids))
            edge_weight = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        elif int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= x.shape[0]:
            raise ValueError("edge_index contains an out-of-range node id")

        edge_attr = edge_weight.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            if x.shape == residual.shape:
                x = x + residual
            x = norm(x)
            if layer != self.num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if was_dense:
            x, valid = to_dense_batch(x, batch)
            if not valid.all():
                raise ValueError("batched graph contains unequal node counts")
        return x


class Sinusoidal3DPositionEncoding(nn.Module):
    """Deterministic sinusoidal encoding that uses all three spatial axes."""

    def __init__(self, dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        pairs = (dim + 1) // 2
        pair_ids = torch.arange(pairs)
        axis = pair_ids.remainder(3)
        frequency_band = torch.div(pair_ids, 3, rounding_mode="floor")
        bands = max(1, math.ceil(pairs / 3))
        denominator = max(1, bands - 1)
        frequency = torch.exp(
            -math.log(temperature) * frequency_band.float() / denominator
        )
        self.register_buffer("axis", axis, persistent=False)
        self.register_buffer("frequency", frequency, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [P,3]")
        positions = positions.to(device=self.frequency.device, dtype=self.frequency.dtype)
        angles = positions[:, self.axis] * self.frequency
        encoded = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return encoded[:, : self.dim]


class GroupAwareNodes2Token(nn.Module):
    """Attention-pool hyperedge representations within their source patch.

    The learned scalar ``s_r`` is computed from each projected hyperedge
    representation. Softmax is applied only over hyperedges sharing the same
    patch, exactly as in the paper equation. Fractional coverage is already
    used by the shared graph encoder's incident-hyperedge messages and is not
    incorrectly substituted for the learned Nodes2Token score.
    """

    def __init__(
        self,
        node_dim: int,
        token_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.token_dim = token_dim
        self.projection = nn.Linear(node_dim, token_dim)
        self.attention_score = nn.Linear(token_dim, 1, bias=False)
        self.position_encoding = Sinusoidal3DPositionEncoding(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hyperedge_nodes: torch.Tensor,
        patch_ids: torch.Tensor,
        *,
        num_patches: int,
        patch_positions: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hyperedge_nodes.ndim not in (2, 3):
            raise ValueError("hyperedge_nodes must have shape [E,D] or [E,M,D]")
        num_hyperedges = hyperedge_nodes.shape[0]
        if patch_ids.shape != (num_hyperedges,):
            raise ValueError("patch_ids must have shape [E]")
        if not torch.isfinite(hyperedge_nodes).all():
            raise ValueError("hyperedge representations must be finite")

        if hyperedge_nodes.ndim == 3:
            if node_mask is None:
                node_mask = torch.ones(
                    hyperedge_nodes.shape[:2],
                    device=hyperedge_nodes.device,
                    dtype=torch.bool,
                )
            if node_mask.shape != hyperedge_nodes.shape[:2]:
                raise ValueError("node_mask must have shape [E,M]")
            denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
            hyperedge_nodes = (
                hyperedge_nodes * node_mask.unsqueeze(-1)
            ).sum(dim=1) / denominator

        if num_patches < 1:
            raise ValueError("num_patches must be positive")
        if num_hyperedges and (
            int(patch_ids.min().item()) < 0
            or int(patch_ids.max().item()) >= num_patches
        ):
            raise ValueError("patch_ids contains an out-of-range patch id")

        h = self.projection(hyperedge_nodes)
        scores = self.attention_score(torch.tanh(h)).squeeze(-1)
        alpha = segment_softmax(scores, patch_ids, num_nodes=num_patches)
        tokens = h.new_zeros((num_patches, self.token_dim))
        # PyG's fallback segment softmax can return FP32 probabilities inside
        # a BF16 autocast region.  ``index_add_`` requires its source and
        # destination to have exactly the same dtype, so retain the weighted
        # values while casting only the accumulation source to the token dtype.
        weighted_h = (alpha.unsqueeze(-1) * h).to(dtype=tokens.dtype)
        tokens.index_add_(0, patch_ids, weighted_h)
        tokens = self.dropout(tokens)

        if patch_positions.shape != (num_patches, 3):
            raise ValueError("patch_positions must have shape [num_patches,3]")
        return tokens + self.position_encoding(
            patch_positions.to(device=tokens.device)
        ).to(dtype=tokens.dtype)


class MultiModalFusion(nn.Module):
    """Shared graph/hyperedge attention followed by group-aware tokenisation."""

    def __init__(
        self,
        image_embed_dim: int,
        mask_embed_dim: int,
        roi_embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_rois: int = 32,
    ) -> None:
        super().__init__()
        self.image_input_proj = nn.Linear(image_embed_dim, hidden_dim)
        self.mask_input_proj = nn.Linear(mask_embed_dim, hidden_dim)
        self.roi_input_proj = nn.Linear(roi_embed_dim, hidden_dim)
        self.shared_encoder = GraphAttentionEncoder(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            num_layers=num_attention_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.hyperedge_node_score = nn.Linear(hidden_dim, 1, bias=False)
        self.nodes2token = GroupAwareNodes2Token(
            hidden_dim, output_dim, dropout=dropout
        )
        self.num_rois = num_rois

    @property
    def image_encoder(self) -> GraphAttentionEncoder:
        return self.shared_encoder

    @property
    def mask_encoder(self) -> GraphAttentionEncoder:
        return self.shared_encoder

    @property
    def roi_encoder(self) -> GraphAttentionEncoder:
        return self.shared_encoder

    @staticmethod
    def _normalise_adjacency(adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        if adj.ndim == 2:
            return adj.unsqueeze(0).expand(batch_size, -1, -1)
        if adj.ndim != 3 or adj.shape[0] != batch_size:
            raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
        return adj

    @staticmethod
    def _derive_patch_ids(
        hyperedge_index: torch.Tensor, num_hyperedges: int, num_patches: int
    ) -> torch.Tensor:
        if num_hyperedges == 0:
            return torch.empty(
                0, dtype=torch.long, device=hyperedge_index.device
            )
        patch_ids = torch.full(
            (num_hyperedges,), -1, dtype=torch.long, device=hyperedge_index.device
        )
        node_ids, edge_ids = hyperedge_index
        image_members = node_ids < num_patches
        image_counts = torch.zeros(
            num_hyperedges, dtype=torch.long, device=hyperedge_index.device
        )
        image_counts.scatter_add_(
            0, edge_ids[image_members], torch.ones_like(edge_ids[image_members])
        )
        if (image_counts != 1).any():
            bad = torch.nonzero(image_counts != 1, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"hyperedges must contain exactly one image-patch node: {bad}"
            )
        patch_ids[edge_ids[image_members]] = node_ids[image_members]
        return patch_ids

    def _build_shared_edges(
        self,
        image_adj: torch.Tensor,
        mask_adj: torch.Tensor,
        roi_adj: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weights: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine within-graph and incident-hyperedge messages."""
        image_adj = self._normalise_adjacency(image_adj, batch_size)
        mask_adj = self._normalise_adjacency(mask_adj, batch_size)
        roi_adj = self._normalise_adjacency(roi_adj, batch_size)
        num_patches = image_adj.shape[1]
        if mask_adj.shape[1:] != (num_patches, num_patches):
            raise ValueError("image and mask graphs must use the same patch nodes")
        num_rois = roi_adj.shape[1]
        nodes_per_subject = 2 * num_patches + num_rois
        num_hyperedges = hyperedge_weights.shape[0]
        if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
            raise ValueError("hyperedge_index must have shape [2,I]")
        if hyperedge_index.numel():
            if int(hyperedge_index[0].max().item()) >= nodes_per_subject:
                raise ValueError("hyperedge incidence has an out-of-range node id")
            if int(hyperedge_index[1].max().item()) >= num_hyperedges:
                raise ValueError("hyperedge incidence has an out-of-range edge id")

        all_indices: List[torch.Tensor] = []
        all_weights: List[torch.Tensor] = []
        modality_offsets = (0, num_patches, 2 * num_patches)

        induced = torch.empty(
            (2, 0), dtype=torch.long, device=hyperedge_index.device
        )
        induced_weights = hyperedge_weights.new_empty(0)
        if hyperedge_index.numel():
            node_ids, edge_ids = hyperedge_index
            order = torch.argsort(edge_ids, stable=True)
            dense_members, valid_members = to_dense_batch(
                node_ids[order], edge_ids[order], batch_size=num_hyperedges,
                fill_value=-1,
            )
            width = dense_members.shape[1]
            src = dense_members[:, :, None].expand(-1, width, width)
            dst = dense_members[:, None, :].expand(-1, width, width)
            pair_valid = (
                valid_members[:, :, None]
                & valid_members[:, None, :]
                & (src != dst)
            )
            induced = torch.stack((src[pair_valid], dst[pair_valid]))
            induced_weights = hyperedge_weights[:, None, None].expand(
                -1, width, width
            )[pair_valid]

        for subject in range(batch_size):
            subject_offset = subject * nodes_per_subject
            for adjacency, modality_offset in zip(
                (image_adj, mask_adj, roi_adj), modality_offsets
            ):
                edge_index, edge_weight = dense_to_sparse(adjacency[subject])
                if edge_index.numel():
                    all_indices.append(edge_index + subject_offset + modality_offset)
                    all_weights.append(edge_weight)

            if induced.numel():
                all_indices.append(induced + subject_offset)
                all_weights.append(induced_weights)

        if all_indices:
            edge_index = torch.cat(all_indices, dim=1)
            edge_weight = torch.cat(all_weights).to(dtype=image_adj.dtype)
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=image_adj.device
            )
            edge_weight = torch.empty(0, dtype=image_adj.dtype, device=image_adj.device)
        batch = torch.arange(batch_size, device=image_adj.device).repeat_interleave(
            nodes_per_subject
        )
        return edge_index, edge_weight, batch

    def _hyperedge_representations(
        self,
        nodes: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_hyperedges: int,
    ) -> torch.Tensor:
        if num_hyperedges == 0:
            return nodes.new_empty((0, nodes.shape[-1]))
        node_ids, edge_ids = hyperedge_index
        members = nodes[node_ids]
        scores = self.hyperedge_node_score(torch.tanh(members)).squeeze(-1)
        attention = segment_softmax(scores, edge_ids, num_nodes=num_hyperedges)
        representations = nodes.new_zeros((num_hyperedges, nodes.shape[-1]))
        weighted_members = (attention.unsqueeze(-1) * members).to(
            dtype=representations.dtype
        )
        representations.index_add_(0, edge_ids, weighted_members)
        return representations

    def forward(
        self,
        image_nodes: torch.Tensor,
        image_adj: torch.Tensor,
        mask_nodes: torch.Tensor,
        mask_adj: torch.Tensor,
        roi_nodes: torch.Tensor,
        roi_adj: torch.Tensor,
        hyperedge_index: torch.Tensor,
        hyperedge_weights: torch.Tensor,
        patch_ids: Optional[torch.Tensor] = None,
        *,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        if image_nodes.ndim != 3 or mask_nodes.ndim != 3 or roi_nodes.ndim != 3:
            raise ValueError("node tensors must have shape [B,N,D]")
        batch_size, num_patches = image_nodes.shape[:2]
        if mask_nodes.shape[:2] != (batch_size, num_patches):
            raise ValueError("image and mask streams must have aligned patch nodes")
        if roi_nodes.shape[0] != batch_size:
            raise ValueError("all modalities must have the same batch size")
        tensors = (
            image_nodes, mask_nodes, roi_nodes, image_adj, mask_adj, roi_adj,
            hyperedge_weights,
        )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("fusion inputs must be finite")

        projected = torch.cat(
            (
                self.image_input_proj(image_nodes),
                self.mask_input_proj(mask_nodes),
                self.roi_input_proj(roi_nodes),
            ),
            dim=1,
        )
        edge_index, edge_weight, batch = self._build_shared_edges(
            image_adj, mask_adj, roi_adj, hyperedge_index, hyperedge_weights,
            batch_size,
        )
        updated_nodes = self.shared_encoder(projected, edge_index, edge_weight, batch)

        num_hyperedges = hyperedge_weights.shape[0]
        derived_patch_ids = self._derive_patch_ids(
            hyperedge_index, num_hyperedges, num_patches
        )
        if patch_ids is not None:
            if not torch.is_tensor(patch_ids) or patch_ids.shape != derived_patch_ids.shape:
                raise ValueError(
                    "supplied patch_ids must have exactly one entry per hyperedge"
                )
            supplied_patch_ids = patch_ids.to(
                device=derived_patch_ids.device, dtype=torch.long
            )
            if not torch.equal(supplied_patch_ids, derived_patch_ids):
                raise ValueError(
                    "supplied patch_ids disagree with the hyperedge incidence map"
                )

        subject_tokens = []
        for subject in range(batch_size):
            h = self._hyperedge_representations(
                updated_nodes[subject], hyperedge_index, num_hyperedges
            )
            tokens = self.nodes2token(
                h,
                derived_patch_ids,
                num_patches=num_patches,
                patch_positions=patch_positions,
            )
            subject_tokens.append(tokens)
        return torch.stack(subject_tokens, dim=0)


__all__ = [
    "GraphAttentionEncoder",
    "Sinusoidal3DPositionEncoding",
    "GroupAwareNodes2Token",
    "MultiModalFusion",
]
