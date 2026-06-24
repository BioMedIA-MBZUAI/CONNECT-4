"""
Mask Graph Construction Module
Constructs graph from mask patches with modernBERT text embeddings.
Edges based on DWI connectivity matrix.
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import numpy as np


class MaskGraphBuilder(nn.Module):
    """
    Builds graph from mask patches.
    - Nodes: patches with modernBERT text embeddings
    - Edges: based on DWI connectivity matrix
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        dwi_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        if dwi_matrix is not None:
            self.register_buffer('dwi_matrix', dwi_matrix)
        else:
            self.dwi_matrix = None
    
    def compute_patch_structure_distribution(
        self,
        mask: torch.Tensor,  # [1, 1, D, H, W]
        patch_idx: int,
        spatial_shape: Tuple[int, int, int],
        structure_labels: Dict[int, str],
    ) -> Dict[int, float]:
        """
        Compute distribution of structures in a patch.
        Returns: {structure_id: percentage_coverage}
        """
        d_patches = spatial_shape[0] // self.patch_size[0]
        h_patches = spatial_shape[1] // self.patch_size[1]
        w_patches = spatial_shape[2] // self.patch_size[2]
        
        d_idx = patch_idx // (h_patches * w_patches)
        h_idx = (patch_idx // w_patches) % h_patches
        w_idx = patch_idx % w_patches
        
        d_start = d_idx * self.patch_size[0]
        d_end = min(d_start + self.patch_size[0], spatial_shape[0])
        h_start = h_idx * self.patch_size[1]
        h_end = min(h_start + self.patch_size[1], spatial_shape[1])
        w_start = w_idx * self.patch_size[2]
        w_end = min(w_start + self.patch_size[2], spatial_shape[2])
        
        patch_mask = mask[0, 0, d_start:d_end, h_start:h_end, w_start:w_end]
        total_voxels = patch_mask.numel()
        
        if total_voxels == 0:
            return {}
        
        # Count each structure
        unique_labels, counts = torch.unique(patch_mask, return_counts=True)
        distribution = {}
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            label_int = int(label)
            if label_int in structure_labels:
                distribution[label_int] = count / total_voxels
        
        return distribution
    
    def compute_patch_center_of_mass(
        self,
        patch_idx: int,
        spatial_shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """Compute center of mass of a patch."""
        d_patches = spatial_shape[0] // self.patch_size[0]
        h_patches = spatial_shape[1] // self.patch_size[1]
        w_patches = spatial_shape[2] // self.patch_size[2]
        
        d_idx = patch_idx // (h_patches * w_patches)
        h_idx = (patch_idx // w_patches) % h_patches
        w_idx = patch_idx % w_patches
        
        center_d = d_idx * self.patch_size[0] + self.patch_size[0] / 2.0
        center_h = h_idx * self.patch_size[1] + self.patch_size[1] / 2.0
        center_w = w_idx * self.patch_size[2] + self.patch_size[2] / 2.0
        
        return (center_d, center_h, center_w)
    
    def create_text_description(
        self,
        patch_idx: int,
        distribution: Dict[int, float],
        center_of_mass: Tuple[float, float, float],
        structure_labels: Dict[int, str],
        normative_values: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> str:
        """
        Create textual description for a patch.
        Format: "Patch {idx} at ({d}, {h}, {w}) contains {num_structures} structures: 
                 {structure_name} ({coverage}%), ..."
        """
        num_structures = len(distribution)
        com_d, com_h, com_w = center_of_mass
        
        desc_parts = [
            f"Patch {patch_idx} at center ({com_d:.1f}, {com_h:.1f}, {com_w:.1f}) "
            f"contains {num_structures} structures:"
        ]
        
        # Sort by coverage (descending)
        sorted_structures = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        for structure_id, coverage in sorted_structures:
            structure_name = structure_labels.get(structure_id, f"structure_{structure_id}")
            desc_parts.append(f"{structure_name} ({coverage*100:.1f}%)")
            
            # Add normative values if available
            if normative_values and structure_id in normative_values:
                norms = normative_values[structure_id]
                norm_str = ", ".join([f"{k}={v:.2f}" for k, v in norms.items()])
                desc_parts[-1] += f" [normative: {norm_str}]"
        
        return " ".join(desc_parts)
    
    def build_adjacency_from_dwi(
        self,
        num_patches: int,
        patch_distributions: List[Dict[int, float]],
        dwi_matrix: torch.Tensor,
        structure_to_roi_idx: Dict[int, int],
    ) -> torch.Tensor:
        """
        Build adjacency matrix based on DWI connectivity.
        For patches with multiple structures, use weighted average based on distribution.
        
        For each pair of patches (i, j):
        - Compute weighted average of DWI connectivity between all structures in patch i and patch j
        - Weight by the distribution (coverage percentage) of each structure in its respective patch
        """
        device = dwi_matrix.device
        adj = torch.zeros(num_patches, num_patches, device=device)
        
        for i in range(num_patches):
            for j in range(i + 1, num_patches):
                # Handle case where patch_distributions might not have entries for all patches
                if i >= len(patch_distributions) or j >= len(patch_distributions):
                    continue
                dist_i = patch_distributions[i] if isinstance(patch_distributions[i], dict) else {}
                dist_j = patch_distributions[j] if isinstance(patch_distributions[j], dict) else {}
                
                # Skip if either patch has no structures
                if not dist_i or not dist_j:
                    continue
                
                # Compute weighted average DWI connectivity
                # For each structure in patch i, compute weighted connectivity to all structures in patch j
                total_weight = 0.0
                weighted_sum = 0.0
                
                for struct_i_id, coverage_i in dist_i.items():
                    if struct_i_id not in structure_to_roi_idx:
                        continue
                    roi_i = structure_to_roi_idx[struct_i_id]
                    
                    # For each structure in patch j, weight by both distributions
                    for struct_j_id, coverage_j in dist_j.items():
                        if struct_j_id not in structure_to_roi_idx:
                            continue
                        roi_j = structure_to_roi_idx[struct_j_id]
                        
                        # DWI connectivity between these two structures
                        dwi_weight = dwi_matrix[roi_i, roi_j].item()
                        
                        # Weight by product of coverages (both patches contribute)
                        weight = coverage_i * coverage_j
                        weighted_sum += dwi_weight * weight
                        total_weight += weight
                
                if total_weight > 0:
                    avg_weight = weighted_sum / total_weight
                    adj[i, j] = avg_weight
                    adj[j, i] = avg_weight
        
        # Add self-loops (patches are connected to themselves)
        adj.fill_diagonal_(1.0)
        
        return adj
    
    def forward(
        self,
        patch_text_embeddings: torch.Tensor,  # [B, num_patches, embed_dim]
        mask: torch.Tensor,  # [B, 1, D, H, W]
        spatial_shape: Tuple[int, int, int],
        structure_labels: Dict[int, str],
        dwi_matrix: Optional[torch.Tensor] = None,
        structure_to_roi_idx: Optional[Dict[int, int]] = None,
        normative_values: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build mask graph.
        
        Args:
            patch_text_embeddings: modernBERT embeddings [B, num_patches, embed_dim]
            mask: Segmentation mask [B, 1, D, H, W]
            spatial_shape: Spatial dimensions (D, H, W)
            structure_labels: Mapping from structure ID to name
            dwi_matrix: DWI connectivity matrix [num_rois, num_rois]
            structure_to_roi_idx: Mapping from structure ID to ROI index
            normative_values: Optional normative values per structure
        
        Returns:
            node_features: [B, num_patches, embed_dim]
            adjacency: [B, num_patches, num_patches] or [num_patches, num_patches]
        """
        batch_size = patch_text_embeddings.shape[0]
        num_patches = patch_text_embeddings.shape[1]
        device = patch_text_embeddings.device
        
        # Use provided DWI matrix or instance variable
        if dwi_matrix is None:
            dwi_matrix = self.dwi_matrix
        
        if dwi_matrix is None:
            raise ValueError(
                "DWI matrix is required for mask graph construction. "
                "Provide it via dwi_matrix parameter or __init__."
            )
        
        if structure_to_roi_idx is None:
            raise ValueError(
                "structure_to_roi_idx is required for mask graph construction. "
                "It maps structure IDs to ROI indices for DWI matrix lookup."
            )
        
        # Compute patch distributions for first sample in batch
        patch_distributions = []
        for patch_idx in range(num_patches):
            dist = self.compute_patch_structure_distribution(
                mask[0:1], patch_idx, spatial_shape, structure_labels
            )
            patch_distributions.append(dist)
        
        # Build adjacency from DWI using weighted average based on distribution
        adj = self.build_adjacency_from_dwi(
            num_patches, patch_distributions, dwi_matrix, structure_to_roi_idx
        )
        
        # Expand to batch if needed
        if batch_size > 1:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return patch_text_embeddings, adj

