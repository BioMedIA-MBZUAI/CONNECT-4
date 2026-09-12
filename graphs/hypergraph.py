"""
Hypergraph Construction Module
Creates hyperedges connecting:
- Patch node from image graph
- Patch node from mask graph  
- Corresponding ROI nodes based on structures present in patch (if any)
Weighted by structure distribution in patch.
Background-only patches do not create hyperedges: the paper defines one
hyperedge only for a non-zero-coverage ROI in a patch.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class HypergraphBuilder(nn.Module):
    """
    Builds hypergraph connecting image patches, mask patches, and ROI nodes.
    """
    
    def __init__(
        self,
        num_patches: int,
        num_rois: int,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_rois = num_rois
        # Parameter-free modules otherwise cannot observe ``module.to(device)``.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
    
    def build_hyperedges(
        self,
        patch_distributions: List[Dict[int, float]],  # Per-patch structure distributions
        structure_to_roi_idx: Dict[int, int],  # Map structure ID to ROI index
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build hyperedge connectivity.
        
        Returns:
            hyperedge_index: [2, num_hyperedges] - (node_idx, hyperedge_idx)
            hyperedge_weights: [num_hyperedges] - fractional ROI coverages
        """
        hyperedges = []
        hyperedge_weights = []
        
        # For each patch, create one hyperedge per structure present in the patch.
        # Each hyperedge connects:
        # - Image patch node (patch_idx)
        # - Mask patch node (patch_idx + num_patches)
        # - ROI node for that specific structure
        # Patches containing no foreground ROI deliberately contribute no
        # hyperedge. Nodes2Token still emits their tokens by adding the 3D
        # positional encoding to a zero pooled representation.
        
        for patch_idx in range(self.num_patches):
            dist = patch_distributions[patch_idx]
            
            # Validate that dist is a dict
            if not isinstance(dist, dict):
                raise TypeError(
                    f"Expected patch_distributions[{patch_idx}] to be a dict, but got {type(dist).__name__}: {dist}. "
                    f"patch_distributions type: {type(patch_distributions)}, length: {len(patch_distributions) if hasattr(patch_distributions, '__len__') else 'N/A'}. "
                    f"First few entries: {[type(patch_distributions[i]).__name__ if i < len(patch_distributions) else 'N/A' for i in range(min(5, len(patch_distributions) if hasattr(patch_distributions, '__len__') else 0))]}"
                )
            
            # Validate values before filtering so a corrupt background or
            # unmapped entry cannot be silently hidden by the selection logic.
            for structure_id, coverage in dist.items():
                if not np.isfinite(float(coverage)):
                    raise RuntimeError(
                        f"NaN/Inf in coverage for patch {patch_idx}, "
                        f"structure {structure_id}: {coverage}"
                    )
                if not 0.0 <= float(coverage) <= 1.0:
                    raise ValueError(
                        f"ROI coverage must lie in [0,1], got {coverage} for "
                        f"patch {patch_idx}, structure {structure_id}"
                    )

            # The paper creates an edge only for a foreground ROI with strictly
            # positive (non-zero) fractional coverage.
            valid_dist = {
                structure_id: float(coverage)
                for structure_id, coverage in dist.items()
                if structure_id != 0
                and structure_id in structure_to_roi_idx
                and float(coverage) > 0.0
            }
            
            # Create one hyperedge per structure in this patch
            if valid_dist:
                # Create a separate hyperedge for each structure
                for structure_id, coverage in valid_dist.items():
                    # Create hyperedge connecting image patch, mask patch, and this ROI
                    hyperedge_nodes = []
                    
                    # Always add image patch node
                    image_node_idx = patch_idx
                    hyperedge_nodes.append(image_node_idx)
                    
                    # Always add mask patch node
                    mask_node_idx = patch_idx + self.num_patches
                    hyperedge_nodes.append(mask_node_idx)
                    
                    # Add ROI node for this specific structure
                    roi_idx = structure_to_roi_idx[structure_id]
                    roi_node_idx = roi_idx + 2 * self.num_patches
                    hyperedge_nodes.append(roi_node_idx)
                    
                    # Weight hyperedge by this structure's coverage
                    hyperedge_weight = float(coverage)
                    
                    hyperedges.append({
                        'nodes': hyperedge_nodes,
                        'weight': hyperedge_weight,
                    })
                    hyperedge_weights.append(hyperedge_weight)
        
        # Convert to edge index format
        # We have one hyperedge per non-background structure per patch.
        edge_list = []
        
        for hyperedge_idx, hyperedge in enumerate(hyperedges):
            for node_idx in hyperedge['nodes']:
                edge_list.append([node_idx, hyperedge_idx])
        
        if edge_list:
            hyperedge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
        else:
            # A scan can validly contain no mapped foreground ROI (for example,
            # an empty/cropped mask); represent that as an empty incidence map.
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        hyperedge_weights_tensor = torch.tensor(hyperedge_weights, dtype=torch.float32, device=device)
        
        return hyperedge_index, hyperedge_weights_tensor

    def patch_ids_from_index(self, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """Return the source patch id for every real hyperedge.

        Each paper-defined hyperedge contains exactly one image-patch node in
        ``[0, num_patches)``. Deriving group ids from incidence avoids relying
        on hyperedge enumeration (a patch may own zero, one, or many edges).
        """
        if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
            raise ValueError("hyperedge_index must have shape [2, num_incidents]")
        if hyperedge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=hyperedge_index.device)
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        patch_ids = torch.full(
            (num_hyperedges,), -1, dtype=torch.long, device=hyperedge_index.device
        )
        node_ids, edge_ids = hyperedge_index
        image_incidence = node_ids < self.num_patches
        image_counts = torch.zeros(
            num_hyperedges, dtype=torch.long, device=hyperedge_index.device
        )
        image_counts.scatter_add_(
            0, edge_ids[image_incidence], torch.ones_like(edge_ids[image_incidence])
        )
        if (image_counts != 1).any():
            bad = torch.nonzero(image_counts != 1, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Hyperedges must contain exactly one image-patch node: {bad}"
            )
        patch_ids[edge_ids[image_incidence]] = node_ids[image_incidence]
        return patch_ids
    
    def forward(
        self,
        patch_distributions: List[Dict[int, float]],
        structure_to_roi_idx: Dict[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build hypergraph.
        
        Args:
            patch_distributions: List of structure distributions per patch
            structure_to_roi_idx: Mapping from structure ID to ROI index
        
        Returns:
            hyperedge_index: [2, num_hyperedges]
            hyperedge_weights: [num_hyperedges]
        """
        return self.build_hyperedges(
            patch_distributions,
            structure_to_roi_idx,
            self._device_anchor.device,
        )
