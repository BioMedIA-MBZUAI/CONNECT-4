"""
Fusion Module using PyTorch Geometric
Graph attention encoder + hypergraph message passing.
Weighted average per patch for hyperedges with same patch ID.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from torch_geometric.nn import GATConv, HypergraphConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse


class GraphAttentionEncoder(nn.Module):
    """
    Multi-layer graph attention encoder using PyTorch Geometric.
    
    Implements step (1): Graph step within each modality
    - IMG graph GNN on P*_img
    - Mask graph GNN on P*_mask  
    - ROI graph GNN on S*
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            # Calculate actual output dimension after GATConv
            if i < num_layers - 1:
                # Intermediate layers: concat=True, so output = (dims[i+1] // num_heads) * num_heads
                # But we need to ensure dims[i+1] is divisible by num_heads
                out_per_head = dims[i + 1] // num_heads
                actual_out_dim = out_per_head * num_heads  # Actual output after concat
            else:
                # Last layer: concat=False, output = dims[i + 1]
                out_per_head = dims[i + 1]  # Not used, but defined for consistency
                actual_out_dim = dims[i + 1]
            
            self.convs.append(
                GATConv(
                    dims[i],
                    out_per_head if i < num_layers - 1 else dims[i + 1],
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    edge_dim=1,  # Edge weights
                    concat=True if i < num_layers - 1 else False,
                )
            )
            if i < num_layers - 1:
                # Use actual output dimension for LayerNorm
                self.norms.append(nn.LayerNorm(actual_out_dim))
    
    def forward(
        self,
        x: torch.Tensor,  # [B, N, input_dim]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_weight: torch.Tensor,  # [num_edges] - edge weights
        batch: Optional[torch.Tensor] = None,  # [num_nodes] - batch assignment
    ) -> torch.Tensor:
        """
        Encode graph with multi-layer GAT.
        
        Args:
            x: Node features [B, N, input_dim] or [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
            batch: Batch assignment [num_nodes] (optional)
        
        Returns:
            Encoded features [B, N, output_dim] or [num_nodes, output_dim]
        """
        # Track if input was in batch format
        was_batch_format = x.dim() == 3
        original_B = x.shape[0] if was_batch_format else None
        
        # Handle batch format: convert from [B, N, D] to [num_nodes, D]
        if was_batch_format:
            B, N, D = x.shape
            x = x.view(B * N, D)
            if batch is None:
                batch = torch.arange(B, device=x.device).repeat_interleave(N)
        
        # Check inputs for NaN/Inf
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf in GraphAttentionEncoder input x (shape={x.shape}, min={x.min()}, max={x.max()})")
        if not torch.isfinite(edge_weight).all():
            raise RuntimeError(f"NaN/Inf in GraphAttentionEncoder edge_weight (shape={edge_weight.shape}, min={edge_weight.min()}, max={edge_weight.max()})")
        
        # Check edge_index validity
        if edge_index.shape[0] != 2:
            raise RuntimeError(f"Invalid edge_index shape: {edge_index.shape}, expected [2, num_edges]")
        if edge_index.shape[1] > 0:
            max_node_idx = edge_index.max().item()
            num_nodes = x.shape[0]
            if max_node_idx >= num_nodes:
                raise RuntimeError(f"Edge index out of bounds: max_node_idx={max_node_idx}, num_nodes={num_nodes}")
        
        # Normalize edge weights to prevent extreme values
        # Use tanh normalization to keep weights in reasonable range
        edge_weight_normalized = torch.tanh(edge_weight / 10.0) * 10.0  # Scale to roughly [-10, 10]
        # Clamp as backup
        edge_weight_normalized = torch.clamp(edge_weight_normalized, min=-10.0, max=10.0)
        
        # Ensure edge_weight has correct shape for GATConv
        edge_attr = edge_weight_normalized.unsqueeze(-1)  # [num_edges, 1] for edge_dim=1
        
        # Check edge_attr
        if not torch.isfinite(edge_attr).all():
            raise RuntimeError(f"NaN/Inf in edge_attr after normalization (shape={edge_attr.shape}, min={edge_attr.min()}, max={edge_attr.max()})")
        
        # Process through all layers
        for i, conv in enumerate(self.convs):
            residual = x if i > 0 and x.shape[-1] == conv.out_channels else None
            
            # Detailed checks before conv
            if not torch.isfinite(x).all():
                finite_mask = torch.isfinite(x)
                finite_count = finite_mask.sum().item()
                total_count = x.numel()
                finite_min = x[finite_mask].min() if finite_count > 0 else torch.tensor(float('nan'), device=x.device)
                finite_max = x[finite_mask].max() if finite_count > 0 else torch.tensor(float('nan'), device=x.device)
                raise RuntimeError(
                    f"NaN/Inf in x BEFORE layer {i} (shape={x.shape}, finite_count={finite_count}/{total_count}, "
                    f"finite_min={finite_min}, finite_max={finite_max})"
                )
            
            # Check edge_index validity
            if edge_index.shape[1] > 0:
                max_node_idx = edge_index.max().item()
                num_nodes = x.shape[0]
                if max_node_idx >= num_nodes:
                    raise RuntimeError(
                        f"Invalid edge_index at layer {i}: max_node_idx={max_node_idx} >= num_nodes={num_nodes}, "
                        f"edge_index_shape={edge_index.shape}, x_shape={x.shape}"
                    )
                min_node_idx = edge_index.min().item()
                if min_node_idx < 0:
                    raise RuntimeError(
                        f"Invalid edge_index at layer {i}: min_node_idx={min_node_idx} < 0"
                    )
            
            # Check edge_attr
            if not torch.isfinite(edge_attr).all():
                finite_mask = torch.isfinite(edge_attr)
                finite_count = finite_mask.sum().item()
                total_count = edge_attr.numel()
                raise RuntimeError(
                    f"NaN/Inf in edge_attr BEFORE layer {i} (shape={edge_attr.shape}, "
                    f"finite_count={finite_count}/{total_count}, "
                    f"edge_attr_min={edge_attr.min()}, edge_attr_max={edge_attr.max()})"
                )
            
            # Check for isolated nodes (nodes with no edges)
            if edge_index.shape[1] == 0:
                # No edges - need to transform x to correct output dimension
                # Calculate expected output dimension from conv layer
                conv = self.convs[i]
                num_heads = getattr(conv, 'heads', 1)
                out_channels = conv.out_channels
                concat = getattr(conv, 'concat', False)
                
                if concat and num_heads > 1:
                    # Intermediate layer: concat=True, output = heads * out_channels
                    expected_out_dim = num_heads * out_channels
                else:
                    # Last layer: concat=False, output = out_channels
                    expected_out_dim = out_channels
                
                if x.shape[-1] != expected_out_dim:
                    # Use conv's weight matrix to transform (without attention)
                    # GATConv has a linear transformation in its forward
                    # For empty edges, we can use the conv's weight directly
                    # But this is complex, so instead we'll create a dummy edge for each node
                    # Create self-loops: each node connects to itself
                    num_nodes = x.shape[0]
                    self_loops = torch.arange(num_nodes, device=x.device)
                    edge_index_self = torch.stack([self_loops, self_loops], dim=0)  # [2, num_nodes]
                    edge_attr_self = torch.ones(num_nodes, 1, device=x.device) * 0.1  # Small weight for self-loops
                    x_out = conv(x, edge_index_self, edge_attr=edge_attr_self)
                else:
                    x_out = x
            else:
                # Try to catch NaN during forward pass
                try:
                    x_out = conv(x, edge_index, edge_attr=edge_attr)
                except Exception as e:
                    raise RuntimeError(
                        f"Error in GATConv forward pass at layer {i}: {e}\n"
                        f"  x shape: {x.shape}, x stats: min={x.min():.6f}, max={x.max():.6f}\n"
                        f"  edge_index shape: {edge_index.shape}, num_edges: {edge_index.shape[1]}\n"
                        f"  edge_attr shape: {edge_attr.shape}, edge_attr stats: min={edge_attr.min():.6f}, max={edge_attr.max():.6f}"
                    ) from e
            
            # Check after convolution with detailed info
            if not torch.isfinite(x_out).all():
                finite_mask = torch.isfinite(x_out)
                finite_count = finite_mask.sum().item()
                total_count = x_out.numel()
                finite_min = x_out[finite_mask].min() if finite_count > 0 else torch.tensor(float('nan'), device=x_out.device)
                finite_max = x_out[finite_mask].max() if finite_count > 0 else torch.tensor(float('nan'), device=x_out.device)
                
                # Additional debugging: check which nodes/edges might be problematic
                nan_mask = ~torch.isfinite(x_out)
                nan_nodes = torch.any(nan_mask, dim=1)  # [num_nodes] - which nodes have NaN
                num_nan_nodes = nan_nodes.sum().item()
                
                # Check if specific edges are problematic
                if edge_index.shape[1] > 0:
                    # Find edges connected to NaN nodes
                    nan_edge_mask = nan_nodes[edge_index[0]] | nan_nodes[edge_index[1]]
                    num_nan_edges = nan_edge_mask.sum().item()
                    
                    raise RuntimeError(
                        f"NaN/Inf in GraphAttentionEncoder AFTER layer {i}\n"
                        f"  Output shape: {x_out.shape}, finite_count: {finite_count}/{total_count}\n"
                        f"  Finite values: min={finite_min}, max={finite_max}\n"
                        f"  NaN nodes: {num_nan_nodes}/{x_out.shape[0]} ({100*num_nan_nodes/x_out.shape[0]:.1f}%)\n"
                        f"  NaN edges: {num_nan_edges}/{edge_index.shape[1]} ({100*num_nan_edges/edge_index.shape[1]:.1f}%)\n"
                        f"  Input x stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}\n"
                        f"  edge_index shape: {edge_index.shape}, num_edges: {edge_index.shape[1]}\n"
                        f"  edge_attr stats: min={edge_attr.min():.6f}, max={edge_attr.max():.6f}, mean={edge_attr.mean():.6f}\n"
                        f"  Conv config: in_channels={conv.in_channels}, out_channels={conv.out_channels}, heads={getattr(conv, 'heads', 'N/A')}"
                    )
            
            x = x_out
            
            if residual is not None:
                x = x + residual
                # Check after residual
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"NaN/Inf in GraphAttentionEncoder after residual at layer {i}")
            
            if i < len(self.norms):
                # Verify x shape matches LayerNorm expected shape
                norm = self.norms[i]
                expected_norm_shape = norm.normalized_shape[0] if isinstance(norm.normalized_shape, tuple) else norm.normalized_shape
                if x.shape[-1] != expected_norm_shape:
                    raise RuntimeError(
                        f"LayerNorm shape mismatch at layer {i}: "
                        f"x shape: {x.shape}, expected normalized_shape: {norm.normalized_shape}, "
                        f"x last dim: {x.shape[-1]}, expected: {expected_norm_shape}"
                    )
                x = self.norms[i](x)
                x = F.relu(x)
                # Check after norm and relu
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"NaN/Inf in GraphAttentionEncoder after norm/relu at layer {i}")
        
        # Convert back to batch format if input was batch format
        if was_batch_format:
            if batch is not None and len(batch.unique()) > 1:
                x, mask = to_dense_batch(x, batch)
                return x  # [B, N, output_dim]
            else:
                # Single batch: ensure output is [B, N, output_dim]
                N = x.shape[0]
                return x.unsqueeze(0)  # [1, N, output_dim]
        
        return x




class HypergraphEncoder(nn.Module):
    """
    Hypergraph encoder using PyG's HypergraphConv with attention.
    
    Implements step (2): Hypergraph step across modalities.
    - Hyperedges e_Pi aggregate from {P_i_img, P_i_mask, S_roi(i)}
    - Nodes {P_i_img, P_i_mask, S*} get messages back from their incident e_Pi
    
    Uses HypergraphConv with attention mechanism for better feature aggregation.
    """
    
    def __init__(
        self,
        node_dim: int,
        hyperedge_dim: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
        attention_mode: str = 'node',
        num_rois: int = 32,  # Number of ROIs for hyperedge_attr projection
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hyperedge_dim = hyperedge_dim
        self.num_rois = num_rois
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Projection layers for hyperedge_attr to match each layer's input dimension
        # Each layer expects hyperedge_attr to match its in_channels
        self.hyperedge_projs = nn.ModuleList()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Calculate actual dimensions accounting for head concatenation
        # First layer: node_dim -> hyperedge_dim (with possible concatenation)
        # Middle layers: previous_output_dim -> hyperedge_dim (with possible concatenation)
        # Last layer: previous_output_dim -> node_dim (no concatenation)
        
        current_dim = node_dim
        for i in range(num_layers):
            heads_i = num_heads if i < num_layers - 1 else 1
            concat_i = True if i < num_layers - 1 and num_heads > 1 else False
            
            if i < num_layers - 1:
                # Intermediate layers: output to hyperedge_dim
                out_dim = hyperedge_dim
            else:
                # Last layer: output to node_dim
                out_dim = node_dim
            
            # When concat=True, output dim = heads * out_channels
            actual_out_dim = out_dim * heads_i if concat_i else out_dim
            
            self.convs.append(
                HypergraphConv(
                    in_channels=current_dim,  # Use actual current dimension
                    out_channels=out_dim,
                    use_attention=True,  # Enable attention mechanism
                    attention_mode=attention_mode,  # 'node' or 'edge'
                    heads=heads_i,
                    concat=concat_i,
                    negative_slope=0.2,
                    dropout=dropout,
                    bias=True,
                )
            )
            
            # Create projection layer for hyperedge_attr to match this layer's input dimension
            # hyperedge_attr comes in as [num_hyperedges, num_rois], needs to be projected to current_dim
            layer_input_dim = current_dim  # Store before updating
            self.hyperedge_projs.append(
                nn.Linear(num_rois, layer_input_dim) if num_rois != layer_input_dim else nn.Identity()
            )
            
            # Update current_dim for next layer
            current_dim = actual_out_dim
            
            if i < num_layers - 1:
                # LayerNorm expects the actual output dimension after concatenation
                self.norms.append(nn.LayerNorm(actual_out_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,  # [B, num_nodes, node_dim] or [num_nodes, node_dim]
        hyperedge_index: torch.Tensor,  # [2, num_edges] - (node_idx, hyperedge_idx)
        hyperedge_attr: Optional[torch.Tensor] = None,  # [num_hyperedges, hyperedge_dim] - structure proportions
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hypergraph message passing with attention.
        
        Args:
            node_features: [B, num_nodes, node_dim] or [num_nodes, node_dim]
            hyperedge_index: [2, num_edges] - sparse hyperedge connectivity
            hyperedge_attr: [num_hyperedges, hyperedge_dim] - structure proportions (0-1) per hyperedge
            batch: Batch assignment (optional)
        
        Returns:
            Updated node features [B, num_nodes, node_dim] or [num_nodes, node_dim]
        """
        # Handle batch format
        if node_features.dim() == 3:
            B, N, D = node_features.shape
            x = node_features.view(B * N, D)
            if batch is None:
                batch = torch.arange(B, device=x.device).repeat_interleave(N)
        else:
            x = node_features
        
        # HypergraphConv expects hyperedge_index in format [2, num_edges]
        # where hyperedge_index[0] = node indices, hyperedge_index[1] = hyperedge indices
        # This matches our format, so we can use it directly
        
        # Check input for NaN/Inf
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf in HypergraphEncoder input x")
        
        # Ensure hyperedge_attr is provided (required for attention)
        # hyperedge_attr should be [num_hyperedges, num_rois] with structure proportions
        if hyperedge_attr is None:
            # Create default hyperedge attributes (all ones)
            if hyperedge_index.shape[1] > 0:
                num_hyperedges = hyperedge_index[1].max().item() + 1
                # Use num_rois as default dimension
                hyperedge_attr = torch.ones(num_hyperedges, self.num_rois, device=x.device, dtype=x.dtype)
            else:
                # No hyperedges, return input unchanged
                return x
        
        # Check hyperedge_attr for NaN/Inf
        if not torch.isfinite(hyperedge_attr).all():
            raise RuntimeError(f"NaN/Inf in HypergraphEncoder hyperedge_attr (min={hyperedge_attr.min()}, max={hyperedge_attr.max()})")
        
        # Process through layers
        for i, conv in enumerate(self.convs):
            # Store input for residual connection
            residual = x if i > 0 else None
            
            # Project hyperedge_attr to match this layer's input dimension
            # Get the input dimension for this layer (before head concatenation)
            if i == 0:
                layer_input_dim = self.node_dim
            else:
                # Previous layer's output dimension (after concatenation)
                prev_heads = self.num_heads if i - 1 < self.num_layers - 1 else 1
                prev_concat = True if i - 1 < self.num_layers - 1 and self.num_heads > 1 else False
                prev_out_dim = self.hyperedge_dim if i - 1 < self.num_layers - 1 else self.node_dim
                layer_input_dim = prev_out_dim * prev_heads if prev_concat else prev_out_dim
            
            # Project hyperedge_attr to match layer's input dimension
            hyperedge_attr_proj = self.hyperedge_projs[i](hyperedge_attr)  # [num_hyperedges, layer_input_dim]
            
            # Check projection output
            if not torch.isfinite(hyperedge_attr_proj).all():
                raise RuntimeError(f"NaN/Inf in hyperedge_attr_proj after projection at layer {i}")
            
            # HypergraphConv forward pass with attention
            # The attention mechanism (attention_mode='node') computes attention
            # scores among nodes within the same hyperedge, which effectively
            # weights the aggregation based on node importance.
            # hyperedge_attr contains structure proportions (0-1) for each hyperedge
            x = conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr_proj)
            
            # Check after convolution
            if not torch.isfinite(x).all():
                raise RuntimeError(f"NaN/Inf in HypergraphEncoder after layer {i}")
            
            # Handle residual connection - only add if dimensions match
            if residual is not None and x.shape[-1] == residual.shape[-1]:
                x = x + residual
                # Check after residual
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"NaN/Inf in HypergraphEncoder after residual at layer {i}")
            
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
                # Check after norm/relu/dropout
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"NaN/Inf in HypergraphEncoder after norm/relu/dropout at layer {i}")
        
        # Convert back to batch format if needed
        # Track if input was in batch format
        was_batch_format = node_features.dim() == 3
        
        if was_batch_format:
            if batch is not None and len(batch.unique()) > 1:
                x, mask = to_dense_batch(x, batch)
                return x  # [B, N, D]
            else:
                # Single batch: ensure output is [1, N, D]
                N = x.shape[0]
                D = x.shape[1]
                return x.unsqueeze(0)  # [1, N, D]
        
        return x


class GroupAwareNodes2Token(nn.Module):
    """
    Group-aware nodes2token tokenization with learned attention.
    For hyperedges with same patch ID, create attention-weighted average based on learned attention
    and ROI distribution (structure proportions) in that patch.
    """
    
    def __init__(
        self,
        node_dim: int,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        
        # Projection from node_dim to token_dim
        self.projection = nn.Linear(node_dim, token_dim)
        
        # Learned attention for nodes within each hyperedge
        # This learns which nodes (image patch, mask patch, ROI nodes) are important
        self.node_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learned attention for hyperedges within each patch
        # This learns which hyperedges (structures) are important for each patch
        self.hyperedge_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Projection to incorporate ROI distribution as bias/weight
        # ROI distribution is used as additional context for attention
        self.roi_dist_proj = nn.Linear(1, token_dim)  # Projects single weight to token_dim for bias
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
    
    def forward(
        self,
        hyperedge_nodes: torch.Tensor,  # [num_hyperedges, max_nodes_per_hyperedge, node_dim]
        hyperedge_weights: torch.Tensor,  # [num_hyperedges] - ROI distribution weights (structure coverage)
        patch_ids: torch.Tensor,  # [num_hyperedges] - patch ID for each hyperedge
    ) -> torch.Tensor:
        """
        Convert hyperedge nodes to tokens using learned attention.
        
        Args:
            hyperedge_nodes: Node features in hyperedge [num_hyperedges, max_nodes, node_dim]
            hyperedge_weights: ROI distribution weights (structure coverage) [num_hyperedges]
            patch_ids: Patch ID for each hyperedge [num_hyperedges]
        
        Returns:
            tokens: [num_unique_patches, token_dim]
        """
        # Check inputs for NaN/Inf
        if not torch.isfinite(hyperedge_nodes).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_nodes input to GroupAwareNodes2Token")
        if not torch.isfinite(hyperedge_weights).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_weights input to GroupAwareNodes2Token (min={hyperedge_weights.min()}, max={hyperedge_weights.max()})")
        
        # Clamp weights to prevent extreme values
        hyperedge_weights = torch.clamp(hyperedge_weights, min=-1e6, max=1e6)
        
        num_hyperedges, max_nodes, _ = hyperedge_nodes.shape
        
        # Validate input shapes
        if hyperedge_weights.shape[0] != num_hyperedges:
            raise RuntimeError(
                f"Shape mismatch: hyperedge_nodes has {num_hyperedges} hyperedges, "
                f"but hyperedge_weights has {hyperedge_weights.shape[0]} weights"
            )
        if patch_ids.shape[0] != num_hyperedges:
            raise RuntimeError(
                f"Shape mismatch: hyperedge_nodes has {num_hyperedges} hyperedges, "
                f"but patch_ids has {patch_ids.shape[0]} IDs"
            )
        
        # Step 1: Project nodes to token dimension
        tokens = self.projection(hyperedge_nodes)  # [num_hyperedges, max_nodes, token_dim]
        
        # Check projection output
        if not torch.isfinite(tokens).all():
            raise RuntimeError(f"NaN/Inf in tokens after projection in GroupAwareNodes2Token")
        
        # Step 2: Learned attention within each hyperedge (weight nodes)
        # Use ROI distribution weights as attention bias
        # MEMORY OPTIMIZATION: Process in chunks and clear cache aggressively
        hyperedge_tokens_list = []
        chunk_size = min(32, num_hyperedges)  # Process 32 hyperedges at a time to reduce memory
        
        for chunk_start in range(0, num_hyperedges, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_hyperedges)
            chunk_tokens_list = []
            
            for he_idx in range(chunk_start, chunk_end):
                he_tokens = tokens[he_idx:he_idx+1]  # [1, max_nodes, token_dim]
                
                # Incorporate ROI distribution weight as bias
                # Project weight to token_dim and add as bias to query/key/value
                roi_weight = hyperedge_weights[he_idx].unsqueeze(0).unsqueeze(0)  # [1, 1]
                roi_bias = self.roi_dist_proj(roi_weight)  # [1, 1, token_dim]
                
                # Add ROI distribution bias to tokens (acts as positional/importance bias)
                he_tokens_with_bias = he_tokens + roi_bias
                
                # Self-attention within hyperedge to learn which nodes are important
                he_tokens_attended, _ = self.node_attention(
                    he_tokens_with_bias,  # query
                    he_tokens_with_bias,  # key
                    he_tokens_with_bias,  # value
                )
                
                # Residual connection and normalization
                he_tokens_attended = self.norm1(he_tokens_attended + he_tokens)
                he_tokens_attended = self.dropout(he_tokens_attended)
                
                # Average over nodes in hyperedge (after attention weighting)
                # Attention already weighted the nodes, so simple mean is appropriate
                he_token = he_tokens_attended.mean(dim=1)  # [1, token_dim]
                chunk_tokens_list.append(he_token)
                
                # Clear intermediate tensors to save memory
                del he_tokens, roi_bias, he_tokens_with_bias, he_tokens_attended
            
            # Concatenate chunk and add to main list
            if chunk_tokens_list:
                chunk_tokens = torch.cat(chunk_tokens_list, dim=0)  # [chunk_size, token_dim]
                hyperedge_tokens_list.append(chunk_tokens)
                del chunk_tokens_list, chunk_tokens
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # CRITICAL: Ensure we concatenate correctly
        if len(hyperedge_tokens_list) == 0:
            raise RuntimeError("No hyperedge tokens created - num_hyperedges must be > 0")
        
        hyperedge_tokens = torch.cat(hyperedge_tokens_list, dim=0)  # [num_hyperedges, token_dim]
        
        # Validate shape after concatenation
        if hyperedge_tokens.shape[0] != num_hyperedges:
            raise RuntimeError(
                f"Shape mismatch after concatenation: expected {num_hyperedges} hyperedges, "
                f"got {hyperedge_tokens.shape[0]} (shape: {hyperedge_tokens.shape})"
            )
        
        # Check after node attention
        if not torch.isfinite(hyperedge_tokens).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_tokens after node attention in GroupAwareNodes2Token")
        
        # Step 3: Group by patch ID and learned attention across hyperedges
        # MEMORY OPTIMIZATION: Process patches in smaller groups and clear cache
        unique_patch_ids = torch.unique(patch_ids)
        num_patches = len(unique_patch_ids)
        
        patch_tokens = []
        patch_chunk_size = min(16, num_patches)  # Process 16 patches at a time
        
        for chunk_start in range(0, num_patches, patch_chunk_size):
            chunk_end = min(chunk_start + patch_chunk_size, num_patches)
            chunk_patch_tokens = []
            
            for patch_idx in range(chunk_start, chunk_end):
                patch_id = unique_patch_ids[patch_idx]
                mask = (patch_ids == patch_id)
                
                # Validate mask shape matches hyperedge_tokens
                if mask.shape[0] != hyperedge_tokens.shape[0]:
                    raise RuntimeError(
                        f"Mask shape mismatch: mask has {mask.shape[0]} elements, "
                        f"but hyperedge_tokens has {hyperedge_tokens.shape[0]} rows. "
                        f"patch_ids shape: {patch_ids.shape}, hyperedge_tokens shape: {hyperedge_tokens.shape}"
                    )
                
                patch_hyperedge_tokens = hyperedge_tokens[mask]  # [num_hyperedges_for_patch, token_dim]
                patch_hyperedge_weights = hyperedge_weights[mask]  # [num_hyperedges_for_patch]
                
                if len(patch_hyperedge_tokens) > 0:
                    # Incorporate ROI distribution weights as bias for hyperedge attention
                    # Project weights to token_dim and add as bias
                    patch_weights_expanded = patch_hyperedge_weights.unsqueeze(-1)  # [num_hyperedges, 1]
                    patch_weights_bias = self.roi_dist_proj(patch_weights_expanded)  # [num_hyperedges, token_dim]
                    
                    # Add ROI distribution bias to hyperedge tokens
                    patch_tokens_with_bias = patch_hyperedge_tokens + patch_weights_bias
                    
                    # Learned attention across hyperedges for this patch
                    # This learns which structures (hyperedges) are important for this patch
                    patch_tokens_attended, _ = self.hyperedge_attention(
                        patch_tokens_with_bias.unsqueeze(0),  # [1, num_hyperedges, token_dim] - query
                        patch_tokens_with_bias.unsqueeze(0),  # [1, num_hyperedges, token_dim] - key
                        patch_tokens_with_bias.unsqueeze(0),  # [1, num_hyperedges, token_dim] - value
                    )
                    
                    # Residual connection and normalization
                    patch_tokens_attended = self.norm2(
                        patch_tokens_attended + patch_hyperedge_tokens.unsqueeze(0)
                    )
                    patch_tokens_attended = self.dropout(patch_tokens_attended)
                    
                    # Attention-weighted average: attention already weighted hyperedges
                    # Use ROI distribution weights as additional weighting factor
                    # Combine learned attention with ROI distribution importance
                    patch_weights_normalized = F.softmax(
                        torch.clamp(patch_hyperedge_weights, min=-50, max=50), 
                        dim=0
                    )  # [num_hyperedges]
                    
                    # Weight by both learned attention (already in attended tokens) and ROI distribution
                    patch_token = (patch_tokens_attended.squeeze(0) * patch_weights_normalized.unsqueeze(-1)).sum(dim=0)  # [token_dim]
                    
                    # Clear intermediate tensors
                    del patch_weights_expanded, patch_weights_bias, patch_tokens_with_bias
                    del patch_tokens_attended, patch_weights_normalized
                else:
                    patch_token = torch.zeros(self.token_dim, device=hyperedge_tokens.device)
                
                # Check patch token
                if not torch.isfinite(patch_token).all():
                    raise RuntimeError(f"NaN/Inf in patch_token for patch_id={patch_id}")
                
                chunk_patch_tokens.append(patch_token)
            
            # Add chunk tokens to main list
            patch_tokens.extend(chunk_patch_tokens)
            del chunk_patch_tokens
            
            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        patch_tokens = torch.stack(patch_tokens)  # [num_patches, token_dim]
        
        # Final check
        if not torch.isfinite(patch_tokens).all():
            raise RuntimeError(f"NaN/Inf in final patch_tokens from GroupAwareNodes2Token")
        
        return patch_tokens


class MultiModalFusion(nn.Module):
    """
    Complete fusion module using PyTorch Geometric:
    1. Graph attention encoder for each graph (PyG GATConv)
    2. Hypergraph encoder across hyperedges (PyG MessagePassing)
    3. Group-aware nodes2token per hyperedge
    4. Weighted average per patch
    """
    
    def __init__(
        self,
        image_embed_dim: int,
        mask_embed_dim: int,
        roi_embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_attention_layers: int = 3,
        num_hypergraph_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_rois: int = 32,  # Number of ROIs for hyperedge_attr projection
    ):
        super().__init__()
        
        # Step (1): Graph encoders for each modality (within-modality GNN)
        self.image_encoder = GraphAttentionEncoder(
            image_embed_dim, hidden_dim, hidden_dim, num_attention_layers, num_heads, dropout
        )
        self.mask_encoder = GraphAttentionEncoder(
            mask_embed_dim, hidden_dim, hidden_dim, num_attention_layers, num_heads, dropout
        )
        self.roi_encoder = GraphAttentionEncoder(
            roi_embed_dim, hidden_dim, hidden_dim, num_attention_layers, num_heads, dropout
        )
        
        # Step (2): Hypergraph encoder (across-modality message passing)
        self.hypergraph_encoder = HypergraphEncoder(
            node_dim=hidden_dim,
            hyperedge_dim=hidden_dim,
            num_layers=num_hypergraph_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_mode='node',  # Attention among nodes within same hyperedge
            num_rois=num_rois,  # For hyperedge_attr projection
        )
        
        # Nodes2token: Convert hyperedge-aggregated features to patch tokens
        # Use learned attention with ROI distribution weighting
        self.nodes2token = GroupAwareNodes2Token(
            node_dim=hidden_dim,
            token_dim=output_dim,
            num_heads=num_heads,  # Use same number of heads as other attention layers
            dropout=dropout,
        )
        
        # Final projection
        self.final_proj = nn.Linear(output_dim, output_dim)
        
        # Cache scaler files to avoid file system operations in forward pass (for torch.compile compatibility)
        scalers_dir = Path("/path/to/data/reconstructed_graphs/scalers")
        self.has_image_scalers = (scalers_dir / "image_mean.npy").exists() and (scalers_dir / "image_std.npy").exists()
        self.has_mask_scalers = (scalers_dir / "mask_mean.npy").exists() and (scalers_dir / "mask_std.npy").exists()
        self.has_roi_scalers = (scalers_dir / "roi_mean.npy").exists() and (scalers_dir / "roi_std.npy").exists()
        
        # Pre-load scalers if they exist (to avoid file I/O in forward pass)
        if self.has_image_scalers:
            self.image_mean = torch.from_numpy(np.load(scalers_dir / "image_mean.npy"))
            self.image_std = torch.from_numpy(np.load(scalers_dir / "image_std.npy"))
        else:
            self.image_mean = None
            self.image_std = None
            
        if self.has_mask_scalers:
            self.mask_mean = torch.from_numpy(np.load(scalers_dir / "mask_mean.npy"))
            self.mask_std = torch.from_numpy(np.load(scalers_dir / "mask_std.npy"))
        else:
            self.mask_mean = None
            self.mask_std = None
            
        if self.has_roi_scalers:
            self.roi_mean = torch.from_numpy(np.load(scalers_dir / "roi_mean.npy"))
            self.roi_std = torch.from_numpy(np.load(scalers_dir / "roi_std.npy"))
        else:
            self.roi_mean = None
            self.roi_std = None
    
    def _adjacency_to_edge_index(
        self,
        adj: torch.Tensor,  # [B, N, N] or [N, N]
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert dense adjacency matrix to PyG edge_index and edge_weight format.
        
        Returns:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            batch: [num_nodes] batch assignment
        """
        # Check adjacency matrix for NaN/Inf
        if not torch.isfinite(adj).all():
            finite_mask = torch.isfinite(adj)
            finite_count = finite_mask.sum().item()
            total_count = adj.numel()
            finite_min = adj[finite_mask].min() if finite_count > 0 else torch.tensor(float('nan'), device=adj.device)
            finite_max = adj[finite_mask].max() if finite_count > 0 else torch.tensor(float('nan'), device=adj.device)
            raise RuntimeError(
                f"NaN/Inf in adjacency matrix (shape={adj.shape}, "
                f"finite_count={finite_count}/{total_count}, "
                f"finite_min={finite_min}, finite_max={finite_max})"
            )
        
        if adj.dim() == 3:
            # Batch format: [B, N, N]
            B, N, _ = adj.shape
            edge_indices = []
            edge_weights = []
            batches = []
            
            for b in range(B):
                adj_b = adj[b]
                edge_idx, edge_w = dense_to_sparse(adj_b)
                
                # Check edge weights for NaN/Inf
                if not torch.isfinite(edge_w).all():
                    raise RuntimeError(f"NaN/Inf in edge weights for batch {b} (shape={edge_w.shape}, min={edge_w.min()}, max={edge_w.max()})")
                
                edge_indices.append(edge_idx + b * N)  # Offset for batch
                edge_weights.append(edge_w)
                batches.append(torch.full((N,), b, device=adj.device))
            
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights)
            batch = torch.cat(batches)
        else:
            # Single graph: [N, N]
            N = adj.shape[0]
            edge_index, edge_weight = dense_to_sparse(adj)
            
            # Check edge weights for NaN/Inf
            if not torch.isfinite(edge_weight).all():
                raise RuntimeError(f"NaN/Inf in edge weights (shape={edge_weight.shape}, min={edge_weight.min()}, max={edge_weight.max()})")
            
            batch = torch.zeros(N, dtype=torch.long, device=adj.device)
        
        # Final check
        if not torch.isfinite(edge_weight).all():
            raise RuntimeError(f"NaN/Inf in final edge_weight (shape={edge_weight.shape}, min={edge_weight.min()}, max={edge_weight.max()})")
        
        return edge_index, edge_weight, batch
    
    def forward(
        self,
        image_nodes: torch.Tensor,  # [B, num_patches, image_embed_dim]
        image_adj: torch.Tensor,  # [B, num_patches, num_patches] - weighted adjacency
        mask_nodes: torch.Tensor,  # [B, num_patches, mask_embed_dim]
        mask_adj: torch.Tensor,  # [B, num_patches, num_patches] - weighted adjacency
        roi_nodes: torch.Tensor,  # [B, num_rois, roi_embed_dim]
        roi_adj: torch.Tensor,  # [B, num_rois, num_rois] - weighted adjacency
        hyperedge_index: torch.Tensor,  # [2, num_edges] - (node_idx, hyperedge_idx)
        hyperedge_weights: torch.Tensor,  # [num_hyperedges] - weights
        patch_ids: torch.Tensor,  # [num_hyperedges] - patch ID for each hyperedge
        patch_distributions: Optional[List[Dict[int, float]]] = None,  # Per-patch structure distributions
        structure_to_roi_idx: Optional[Dict[int, int]] = None,  # Map structure ID to ROI index
    ) -> torch.Tensor:
        """
        Complete fusion pipeline using PyTorch Geometric.
        
        Returns:
            fused_tokens: [B, num_patches, output_dim]
        """
        B = image_nodes.shape[0]
        num_patches = image_nodes.shape[1]
        num_rois = roi_nodes.shape[1]
        device = image_nodes.device
        
        # Step (1): Graph step within each modality
        # Check inputs for NaN/Inf
        if not torch.isfinite(image_nodes).all():
            raise RuntimeError(f"NaN/Inf in image_nodes input to MultiModalFusion")
        if not torch.isfinite(mask_nodes).all():
            raise RuntimeError(f"NaN/Inf in mask_nodes input to MultiModalFusion")
        if not torch.isfinite(roi_nodes).all():
            raise RuntimeError(f"NaN/Inf in roi_nodes input to MultiModalFusion")
        if not torch.isfinite(image_adj).all():
            raise RuntimeError(f"NaN/Inf in image_adj input to MultiModalFusion")
        if not torch.isfinite(mask_adj).all():
            raise RuntimeError(f"NaN/Inf in mask_adj input to MultiModalFusion")
        if not torch.isfinite(roi_adj).all():
            raise RuntimeError(f"NaN/Inf in roi_adj input to MultiModalFusion")
        
        # Convert adjacency matrices to PyG format
        img_edge_index, img_edge_weight, img_batch = self._adjacency_to_edge_index(image_adj, B)
        mask_edge_index, mask_edge_weight, mask_batch = self._adjacency_to_edge_index(mask_adj, B)
        roi_edge_index, roi_edge_weight, roi_batch = self._adjacency_to_edge_index(roi_adj, B)
        
        # Check edge weights after conversion
        if not torch.isfinite(img_edge_weight).all():
            raise RuntimeError(f"NaN/Inf in img_edge_weight after conversion (shape={img_edge_weight.shape}, min={img_edge_weight.min()}, max={img_edge_weight.max()})")
        if not torch.isfinite(mask_edge_weight).all():
            raise RuntimeError(f"NaN/Inf in mask_edge_weight after conversion (shape={mask_edge_weight.shape}, min={mask_edge_weight.min()}, max={mask_edge_weight.max()})")
        if not torch.isfinite(roi_edge_weight).all():
            raise RuntimeError(f"NaN/Inf in roi_edge_weight after conversion (shape={roi_edge_weight.shape}, min={roi_edge_weight.min()}, max={roi_edge_weight.max()})")
        
        # Normalize edge weights to prevent extreme values (tanh normalization)
        img_edge_weight = torch.tanh(img_edge_weight / 10.0) * 10.0
        mask_edge_weight = torch.tanh(mask_edge_weight / 10.0) * 10.0
        roi_edge_weight = torch.tanh(roi_edge_weight / 10.0) * 10.0
        
        # Clamp as backup
        img_edge_weight = torch.clamp(img_edge_weight, min=-10.0, max=10.0)
        mask_edge_weight = torch.clamp(mask_edge_weight, min=-10.0, max=10.0)
        roi_edge_weight = torch.clamp(roi_edge_weight, min=-10.0, max=10.0)
        
        # Process each modality graph
        image_encoded = self.image_encoder(image_nodes, img_edge_index, img_edge_weight, img_batch)
        mask_encoded = self.mask_encoder(mask_nodes, mask_edge_index, mask_edge_weight, mask_batch)
        
        # Normalize embeddings using saved scalers (if available)
        # Note: Embeddings are already normalized during reconstruction, but we apply normalization
        # here as well in case raw embeddings are loaded (for consistency)
        # Scalers are pre-loaded in __init__ to avoid file I/O in forward pass (torch.compile compatibility)
        
        # Normalize image nodes (3-sigma clipping preserves ~99.7% of data)
        if self.has_image_scalers:
            image_mean = self.image_mean.to(image_nodes.device).to(image_nodes.dtype)
            image_std = self.image_std.to(image_nodes.device).to(image_nodes.dtype)
            image_nodes = (image_nodes - image_mean) / image_std
            image_nodes = torch.clamp(image_nodes, min=-3.0, max=3.0)  # Clip only extreme outliers
        
        # Normalize mask nodes (3-sigma clipping)
        if self.has_mask_scalers:
            mask_mean = self.mask_mean.to(mask_nodes.device).to(mask_nodes.dtype)
            mask_std = self.mask_std.to(mask_nodes.device).to(mask_nodes.dtype)
            mask_nodes = (mask_nodes - mask_mean) / mask_std
            mask_nodes = torch.clamp(mask_nodes, min=-3.0, max=3.0)  # Clip only extreme outliers
        
        # Normalize ROI nodes (3-sigma clipping)
        if self.has_roi_scalers:
            roi_mean = self.roi_mean.to(roi_nodes.device).to(roi_nodes.dtype)
            roi_std = self.roi_std.to(roi_nodes.device).to(roi_nodes.dtype)
            roi_nodes = (roi_nodes - roi_mean) / roi_std
            roi_nodes = torch.clamp(roi_nodes, min=-3.0, max=3.0)  # Clip only extreme outliers
        
        # Debug ROI inputs before encoding
        roi_encoded = self.roi_encoder(roi_nodes, roi_edge_index, roi_edge_weight, roi_batch)
        
        # Check encoded outputs
        if not torch.isfinite(image_encoded).all():
            raise RuntimeError(f"NaN/Inf in image_encoded after GraphAttentionEncoder")
        if not torch.isfinite(mask_encoded).all():
            raise RuntimeError(f"NaN/Inf in mask_encoded after GraphAttentionEncoder")
        
        # Detailed check for ROI encoder (this is where the error occurs)
        if not torch.isfinite(roi_encoded).all():
            finite_mask = torch.isfinite(roi_encoded)
            finite_count = finite_mask.sum().item()
            total_count = roi_encoded.numel()
            finite_min = roi_encoded[finite_mask].min() if finite_count > 0 else torch.tensor(float('nan'), device=roi_encoded.device)
            finite_max = roi_encoded[finite_mask].max() if finite_count > 0 else torch.tensor(float('nan'), device=roi_encoded.device)
            
            # Check ROI inputs before encoding
            roi_finite_before = torch.isfinite(roi_nodes).all()
            roi_adj_finite_before = torch.isfinite(roi_adj).all()
            
            raise RuntimeError(
                f"NaN/Inf in roi_encoded after GraphAttentionEncoder\n"
                f"  roi_encoded shape: {roi_encoded.shape}, finite_count: {finite_count}/{total_count}\n"
                f"  Finite values: min={finite_min}, max={finite_max}\n"
                f"  ROI nodes finite before encoding: {roi_finite_before}\n"
                f"  ROI adj finite before encoding: {roi_adj_finite_before}\n"
                f"  ROI nodes stats: min={roi_nodes.min():.6f}, max={roi_nodes.max():.6f}, mean={roi_nodes.mean():.6f}, std={roi_nodes.std():.6f}\n"
                f"  ROI adj stats: min={roi_adj.min():.6f}, max={roi_adj.max():.6f}, mean={roi_adj.mean():.6f}, std={roi_adj.std():.6f}"
            )
        
        # Step (2): Hypergraph step across modalities
        # Combine all nodes: [image_patches, mask_patches, roi_nodes]
        all_nodes = torch.cat([
            image_encoded,  # [B, num_patches, hidden_dim]
            mask_encoded,   # [B, num_patches, hidden_dim]
            roi_encoded,    # [B, num_rois, hidden_dim]
        ], dim=1)  # [B, 2*num_patches + num_rois, hidden_dim]
        
        # Check hyperedge_weights for NaN/Inf
        if not torch.isfinite(hyperedge_weights).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_weights input to MultiModalFusion (min={hyperedge_weights.min()}, max={hyperedge_weights.max()})")
        
        # Clamp hyperedge_weights to prevent extreme values
        hyperedge_weights = torch.clamp(hyperedge_weights, min=-1e6, max=1e6)
        
        # Build hyperedge_attr from patch_distributions
        # Each hyperedge corresponds to a patch, and hyperedge_attr represents structure proportions
        num_hyperedges = hyperedge_weights.shape[0]
        if patch_distributions is not None and structure_to_roi_idx is not None:
            # Create hyperedge_attr: [num_hyperedges, num_rois]
            # Each row represents structure proportions for that patch/hyperedge
            hyperedge_attr = torch.zeros(num_hyperedges, num_rois, device=device, dtype=torch.float32)
            
            for he_idx in range(num_hyperedges):
                patch_id = patch_ids[he_idx].item()
                if patch_id < len(patch_distributions):
                    patch_dist = patch_distributions[patch_id]
                    # Map structure proportions to ROI indices
                    for structure_id, proportion in patch_dist.items():
                        if structure_id != 0 and structure_id in structure_to_roi_idx:
                            roi_idx = structure_to_roi_idx[structure_id]
                            # Check proportion for NaN/Inf
                            prop_val = float(proportion)
                            if not (torch.isfinite(torch.tensor(prop_val))):
                                raise RuntimeError(f"NaN/Inf in proportion for patch_id={patch_id}, structure_id={structure_id}, proportion={proportion}")
                            hyperedge_attr[he_idx, roi_idx] = prop_val
        else:
            # Fallback: use hyperedge_weights as single attribute dimension
            hyperedge_attr = hyperedge_weights.unsqueeze(-1)  # [num_hyperedges, 1]
        
        # Check hyperedge_attr for NaN/Inf
        if not torch.isfinite(hyperedge_attr).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_attr after construction (min={hyperedge_attr.min()}, max={hyperedge_attr.max()})")
        
        # Adjust hyperedge_index for batch processing
        # If hyperedge_index is for single graph, expand for batch
        if hyperedge_index.shape[1] > 0:
            # Check if hyperedge_index needs batch offset
            max_node_idx = hyperedge_index[0].max().item()
            if max_node_idx < 2 * num_patches + num_rois:
                # Single graph, need to expand for batch
                batch_hyperedge_indices = []
                batch_hyperedge_attrs = []
                for b in range(B):
                    offset = b * (2 * num_patches + num_rois)
                    batch_hyperedge_indices.append(hyperedge_index + offset)
                    batch_hyperedge_attrs.append(hyperedge_attr)
                hyperedge_index_batch = torch.cat(batch_hyperedge_indices, dim=1)
                hyperedge_attr_batch = torch.cat(batch_hyperedge_attrs, dim=0)  # [B*num_hyperedges, num_rois]
                all_batch = torch.arange(B, device=device).repeat_interleave(2 * num_patches + num_rois)
            else:
                # Already batched
                hyperedge_index_batch = hyperedge_index
                hyperedge_attr_batch = hyperedge_attr
                all_batch = torch.arange(B, device=device).repeat_interleave(2 * num_patches + num_rois)
        else:
            hyperedge_index_batch = hyperedge_index
            hyperedge_attr_batch = hyperedge_attr
            all_batch = None
        
        # Check all_nodes before hypergraph encoder
        if not torch.isfinite(all_nodes).all():
            raise RuntimeError(f"NaN/Inf in all_nodes before HypergraphEncoder")
        
        # Check hyperedge_attr_batch
        if not torch.isfinite(hyperedge_attr_batch).all():
            raise RuntimeError(f"NaN/Inf in hyperedge_attr_batch before HypergraphEncoder (min={hyperedge_attr_batch.min()}, max={hyperedge_attr_batch.max()})")
        
        # Hypergraph message passing
        updated_nodes = self.hypergraph_encoder(
            all_nodes,
            hyperedge_index_batch,
            hyperedge_attr_batch,  # Pass hyperedge_attr instead of weights
            all_batch,
        )  # [B, 2*num_patches + num_rois, hidden_dim]
        
        # Check updated_nodes after hypergraph encoder
        if not torch.isfinite(updated_nodes).all():
            raise RuntimeError(f"NaN/Inf in updated_nodes after HypergraphEncoder")
        
        # Step (3): Group-aware nodes2token tokenization
        # Extract hyperedge node features for each hyperedge
        # Use the original (non-batched) hyperedge_index to extract nodes per batch
        num_hyperedges = hyperedge_weights.shape[0]
        
        # Find max nodes per hyperedge (using original index)
        max_nodes_per_hyperedge = 0
        if hyperedge_index.shape[1] > 0:
            for he_idx in range(num_hyperedges):
                he_mask = (hyperedge_index[1] == he_idx)
                num_nodes_in_he = he_mask.sum().item()
                max_nodes_per_hyperedge = max(max_nodes_per_hyperedge, num_nodes_in_he)
        else:
            max_nodes_per_hyperedge = 1  # At least 1 for empty case
        
        # Extract hyperedge node features per batch
        fused_tokens_list = []
        for b in range(B):
            # Extract hyperedge nodes for this batch
            hyperedge_nodes_b = []
            
            # Handle case when num_hyperedges is 0
            if num_hyperedges == 0:
                # Create dummy hyperedge with zero features
                # Ensure max_nodes_per_hyperedge is at least 1
                max_nodes = max(max_nodes_per_hyperedge, 1)
                dummy_feats = torch.zeros((1, max_nodes, updated_nodes.shape[2]), device=updated_nodes.device)
                hyperedge_nodes_b = dummy_feats
                # Create dummy weights and patch_ids
                dummy_weights = torch.zeros(1, device=updated_nodes.device)
                dummy_patch_ids = torch.zeros(1, device=updated_nodes.device, dtype=torch.long)
                patch_tokens_b = self.nodes2token(
                    hyperedge_nodes_b,
                    dummy_weights,
                    dummy_patch_ids,
                )
            else:
                for he_idx in range(num_hyperedges):
                    if hyperedge_index.shape[1] > 0:
                        he_mask = (hyperedge_index[1] == he_idx)
                        connected_node_indices = hyperedge_index[0][he_mask]  # Node indices in single graph
                        
                        if len(connected_node_indices) > 0:
                            # Get node features from this batch's updated nodes
                            he_node_feats = updated_nodes[b, connected_node_indices, :]  # [num_nodes_in_he, hidden_dim]
                        else:
                            # Empty hyperedge
                            he_node_feats = torch.zeros((1, updated_nodes.shape[2]), device=updated_nodes.device)
                    else:
                        # No hyperedges
                        he_node_feats = torch.zeros((1, updated_nodes.shape[2]), device=updated_nodes.device)
                    
                    # Pad to max_nodes_per_hyperedge (INSIDE the for loop!)
                    if he_node_feats.shape[0] < max_nodes_per_hyperedge:
                        padding = torch.zeros(
                            (max_nodes_per_hyperedge - he_node_feats.shape[0], he_node_feats.shape[1]),
                            device=he_node_feats.device
                        )
                        he_node_feats = torch.cat([he_node_feats, padding], dim=0)
                    elif he_node_feats.shape[0] > max_nodes_per_hyperedge:
                        he_node_feats = he_node_feats[:max_nodes_per_hyperedge]
                    
                    hyperedge_nodes_b.append(he_node_feats)
            
            hyperedge_nodes_b = torch.stack(hyperedge_nodes_b, dim=0)  # [num_hyperedges, max_nodes, hidden_dim]
            
            # Apply nodes2token for this batch
            patch_tokens_b = self.nodes2token(
                hyperedge_nodes_b,  # [num_hyperedges, max_nodes, hidden_dim]
                hyperedge_weights,  # [num_hyperedges]
                patch_ids,  # [num_hyperedges]
            )  # [num_unique_patches, output_dim]
            
            fused_tokens_list.append(patch_tokens_b)
        
        # Stack and ensure consistent shape
        # Pad to num_patches (expected number of patches)
        fused_tokens_padded = []
        for tokens in fused_tokens_list:
            if tokens.shape[0] < num_patches:
                padding = torch.zeros(
                    (num_patches - tokens.shape[0], tokens.shape[1]),
                    device=tokens.device
                )
                tokens = torch.cat([tokens, padding], dim=0)
            elif tokens.shape[0] > num_patches:
                # Truncate if somehow we have more patches
                tokens = tokens[:num_patches]
            fused_tokens_padded.append(tokens)
        
        fused = torch.stack(fused_tokens_padded, dim=0)  # [B, num_patches, output_dim]
        
        # Check before final projection
        if not torch.isfinite(fused).all():
            raise RuntimeError(f"NaN/Inf in fused tokens before final projection (min={fused.min()}, max={fused.max()})")
        
        # Final projection
        fused = self.final_proj(fused)  # [B, num_patches, output_dim]
        
        # Check after final projection
        if not torch.isfinite(fused).all():
            raise RuntimeError(f"NaN/Inf in fused tokens after final projection (min={fused.min()}, max={fused.max()})")
        
        return fused
