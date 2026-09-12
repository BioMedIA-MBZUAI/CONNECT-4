"""
Image Graph Construction Module
Constructs graph from T1w patches with frozen BrainIAC embeddings.
Edges weighted by Chebyshev distance, connected to nearest k neighbors.
"""
import torch
import torch.nn as nn
from typing import Tuple

from architecture_contract import geometric_patch_centers_voxel


def chebyshev_distance(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """Compute Chebyshev distance between patch positions."""
    return torch.max(torch.abs(pos1 - pos2), dim=-1)[0]


class ImageGraphBuilder(nn.Module):
    """
    Builds graph from T1w image patches.
    - Nodes: patches with BrainIAC embeddings
    - Edges: weighted by Chebyshev distance, k nearest neighbors
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        k_neighbors: int = 5,
        edge_weight_threshold: float = 0.0,
    ):
        super().__init__()
        if (
            len(patch_size) != 3
            or any(isinstance(value, bool) or int(value) < 1 for value in patch_size)
        ):
            raise ValueError("patch_size must contain three positive integers")
        if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, int) or k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")
        if not torch.isfinite(torch.tensor(float(edge_weight_threshold))):
            raise ValueError("edge_weight_threshold must be finite")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.k_neighbors = int(k_neighbors)
        self.edge_weight_threshold = float(edge_weight_threshold)
    
    def get_patch_positions(
        self,
        spatial_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Get canonical geometric XYZ patch centres in C-order."""
        if len(spatial_shape) != 3:
            raise ValueError("spatial_shape must be an XYZ triplet")
        positions = geometric_patch_centers_voxel(spatial_shape, self.patch_size)
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
        positions = patch_positions.to(device=device)
        num_patches = positions.shape[0]
        if num_patches < 2 or self.k_neighbors >= num_patches:
            raise ValueError(
                "image graph requires at least two patches and k_neighbors < num_patches"
            )

        distances = (positions[:, None, :] - positions[None, :, :]).abs().amax(dim=-1)
        candidates = distances.clone()
        candidates.fill_diagonal_(torch.inf)
        k = min(self.k_neighbors, max(0, num_patches - 1))
        neighbours = torch.topk(candidates, k=k, largest=False).indices
        directed = torch.zeros(
            num_patches, num_patches, dtype=torch.bool, device=device
        )
        directed.scatter_(1, neighbours, True)
        undirected = directed | directed.T
        undirected &= distances > self.edge_weight_threshold

        # The paper defines the edge attribute as the Chebyshev distance itself
        # (not a similarity obtained by inverting that distance).
        return torch.where(undirected, distances, torch.zeros_like(distances))
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,  # [B, num_patches, embed_dim]
        spatial_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build image graph.
        
        Args:
            patch_embeddings: BrainIAC embeddings for each patch [B, num_patches, embed_dim]
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
