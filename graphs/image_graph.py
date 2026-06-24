"""
Image Graph Construction Module
Constructs graph from T1w patches with BMMAE embeddings.
Edges weighted by Chebyshev distance, connected to nearest k neighbors.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


def chebyshev_distance(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Compute Chebyshev distance between patch positions."""
    return torch.max(torch.abs(pos1 - pos2), dim=-1)[0]


class ImageGraphBuilder(nn.Module):
    """
    Builds graph from T1w image patches.
    - Nodes: patches with BMMAE embeddings
    - Edges: weighted by Chebyshev distance, k nearest neighbors
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        k_neighbors: int = 5,
        edge_weight_threshold: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.k_neighbors = k_neighbors
        self.edge_weight_threshold = edge_weight_threshold
    
    def get_patch_positions(
        self,
        spatial_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Get 3D positions of each patch center."""
        d_patches = spatial_shape[0] // self.patch_size[0]
        h_patches = spatial_shape[1] // self.patch_size[1]
        w_patches = spatial_shape[2] // self.patch_size[2]
        
        positions = []
        for d in range(d_patches):
            for h in range(h_patches):
                for w in range(w_patches):
                    center_d = d * self.patch_size[0] + self.patch_size[0] // 2
                    center_h = h * self.patch_size[1] + self.patch_size[1] // 2
                    center_w = w * self.patch_size[2] + self.patch_size[2] // 2
                    positions.append([center_d, center_h, center_w])
        
        return torch.tensor(positions, dtype=torch.float32)
    
    def build_adjacency(
        self,
        patch_positions: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Build adjacency matrix with Chebyshev distance weights.
        Returns: [num_patches, num_patches] adjacency matrix
        """
        num_patches = patch_positions.shape[0]
        adj = torch.zeros(num_patches, num_patches, device=device)
        
        # Compute pairwise Chebyshev distances
        for i in range(num_patches):
            distances = []
            for j in range(num_patches):
                if i == j:
                    dist = 0.0
                else:
                    dist = chebyshev_distance(
                        patch_positions[i:i+1],
                        patch_positions[j:j+1]
                    ).item()
                distances.append(dist)
            
            distances = torch.tensor(distances, device=device)
            # Get k nearest neighbors (excluding self)
            _, topk_indices = torch.topk(distances, k=min(self.k_neighbors + 1, num_patches), largest=False)
            topk_indices = topk_indices[topk_indices != i]  # Remove self
            
            # Set edge weights (inverse distance, normalized)
            for idx in topk_indices:
                dist = distances[idx].item()
                if dist > self.edge_weight_threshold:
                    weight = 1.0 / (1.0 + dist)  # Inverse distance weighting
                    adj[i, idx] = weight
                    adj[idx, i] = weight  # Symmetric
        
        return adj
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,  # [B, num_patches, embed_dim]
        spatial_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build image graph.
        
        Args:
            patch_embeddings: BMMAE embeddings for each patch [B, num_patches, embed_dim]
            spatial_shape: Spatial dimensions of original image (D, H, W)
        
        Returns:
            node_features: [B, num_patches, embed_dim]
            adjacency: [B, num_patches, num_patches] or [num_patches, num_patches]
        """
        batch_size = patch_embeddings.shape[0]
        device = patch_embeddings.device
        
        # Get patch positions
        patch_positions = self.get_patch_positions(spatial_shape).to(device)
        
        # Build adjacency matrix
        adj = self.build_adjacency(patch_positions, device)
        
        # Expand to batch if needed
        if batch_size > 1:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return patch_embeddings, adj

