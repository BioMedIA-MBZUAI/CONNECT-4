"""
ROI Graph Construction Module
Each node is a structure, embeddings from radiomics + AnatCL.
Edges weighted by DWI matrix.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class ROIGraphBuilder(nn.Module):
    """
    Builds ROI-level graph.
    - Nodes: brain structures (ROIs)
    - Node features: radiomics features + AnatCL embeddings
    - Edges: weighted by DWI connectivity matrix
    """
    
    def __init__(
        self,
        num_rois: int = 32,
        dwi_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_rois = num_rois
        
        if dwi_matrix is not None:
            self.register_buffer('dwi_matrix', dwi_matrix)
        else:
            self.dwi_matrix = None
    
    def forward(
        self,
        roi_embeddings: torch.Tensor,  # [B, num_rois, embed_dim]
        dwi_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build ROI graph.
        
        Args:
            roi_embeddings: Combined radiomics + AnatCL embeddings [B, num_rois, embed_dim]
            dwi_matrix: DWI connectivity matrix [num_rois, num_rois] (required)
        
        Returns:
            node_features: [B, num_rois, embed_dim]
            adjacency: [B, num_rois, num_rois] or [num_rois, num_rois]
        """
        batch_size = roi_embeddings.shape[0]
        device = roi_embeddings.device
        
        # Use provided DWI matrix or instance variable
        if dwi_matrix is None:
            dwi_matrix = self.dwi_matrix
        
        if dwi_matrix is None:
            raise ValueError(
                "DWI matrix is required for ROI graph construction. "
                "Provide it via dwi_matrix parameter or __init__."
            )
        
        adj = dwi_matrix.clone().to(device)
        if adj.shape != (self.num_rois, self.num_rois):
            raise ValueError(
                f"DWI matrix must be {self.num_rois}x{self.num_rois}, got {adj.shape}"
            )
        if not torch.isfinite(adj).all():
            raise ValueError("DWI matrix contains NaN or infinity")
        if (adj < 0).any():
            raise ValueError("DWI structural-connectivity coefficients must be non-negative")
        # The manuscript says ROI edges are taken directly from the fixed DWI
        # matrix. Do not symmetrise, normalise, threshold, or overwrite its
        # diagonal here. GATConv supplies computational self-loops separately.
        
        # Expand to batch if needed
        if batch_size > 1:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return roi_embeddings, adj
