"""
Hypergraph Construction Module
Creates hyperedges connecting:
- Patch node from image graph
- Patch node from mask graph  
- Corresponding ROI nodes based on structures present in patch (if any)
Weighted by structure distribution in patch.
For background-only patches, only connects image and mask patch nodes.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
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
            hyperedge_weights: [num_hyperedges] - weights based on distribution
        """
        hyperedges = []
        hyperedge_weights = []
        
        # For each patch, create one hyperedge per structure present in the patch.
        # Each hyperedge connects:
        # - Image patch node (patch_idx)
        # - Mask patch node (patch_idx + num_patches)
        # - ROI node for that specific structure
        # For background-only patches, create one hyperedge with just image and mask nodes
        
        for patch_idx in range(self.num_patches):
            dist = patch_distributions[patch_idx]
            
            # Validate that dist is a dict
            if not isinstance(dist, dict):
                raise TypeError(
                    f"Expected patch_distributions[{patch_idx}] to be a dict, but got {type(dist).__name__}: {dist}. "
                    f"patch_distributions type: {type(patch_distributions)}, length: {len(patch_distributions) if hasattr(patch_distributions, '__len__') else 'N/A'}. "
                    f"First few entries: {[type(patch_distributions[i]).__name__ if i < len(patch_distributions) else 'N/A' for i in range(min(5, len(patch_distributions) if hasattr(patch_distributions, '__len__') else 0))]}"
                )
            
            # Filter out background (label 0) and only keep structures in structure_to_roi_idx
            valid_dist = {
                structure_id: coverage 
                for structure_id, coverage in dist.items() 
                if structure_id != 0 and structure_id in structure_to_roi_idx
            }
            
            # Get background coverage
            background_coverage = dist.get(0, 0.0)
            
            # Check for NaN/Inf in background_coverage
            if not np.isfinite(background_coverage):
                raise RuntimeError(f"NaN/Inf in background_coverage for patch {patch_idx}: {background_coverage}")
            
            # Create one hyperedge per structure in this patch
            if valid_dist:
                # Check for NaN/Inf in coverages
                for sid, cov in valid_dist.items():
                    if not np.isfinite(cov):
                        raise RuntimeError(f"NaN/Inf in coverage for patch {patch_idx}, structure {sid}: {cov}")
                
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
            else:
                # Background-only patch: create one hyperedge with just image and mask nodes
                hyperedge_nodes = []
                
                # Always add image patch node
                image_node_idx = patch_idx
                hyperedge_nodes.append(image_node_idx)
                
                # Always add mask patch node
                mask_node_idx = patch_idx + self.num_patches
                hyperedge_nodes.append(mask_node_idx)
                
                # Weight by background coverage or small default
                hyperedge_weight = float(background_coverage) if background_coverage > 0 else 0.1
                
                hyperedges.append({
                    'nodes': hyperedge_nodes,
                    'weight': hyperedge_weight,
                })
                hyperedge_weights.append(hyperedge_weight)
        
        # Convert to edge index format
        # We have one hyperedge per structure per patch (plus one for background-only patches)
        # So we'll have >= num_patches hyperedges (more if patches have multiple structures)
        num_hyperedges = len(hyperedges)
        edge_list = []
        
        for hyperedge_idx, hyperedge in enumerate(hyperedges):
            for node_idx in hyperedge['nodes']:
                edge_list.append([node_idx, hyperedge_idx])
        
        if edge_list:
            hyperedge_index = torch.tensor(edge_list, dtype=torch.long, device=device).T
        else:
            # This should not happen since we create at least one hyperedge per patch
            print(f"WARNING: No hyperedges created despite processing {self.num_patches} patches. "
                  f"This indicates a bug in the hypergraph construction.")
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        hyperedge_weights_tensor = torch.tensor(hyperedge_weights, dtype=torch.float32, device=device)
        
        # Validate: we should have at least num_patches hyperedges (one per patch minimum)
        if len(hyperedge_weights_tensor) < self.num_patches:
            print(f"WARNING: Expected at least {self.num_patches} hyperedges (one per patch) but got {len(hyperedge_weights_tensor)}. "
                  f"This may indicate a data preprocessing issue.")
        
        return hyperedge_index, hyperedge_weights_tensor
    
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
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        return self.build_hyperedges(
            patch_distributions,
            structure_to_roi_idx,
            device,
        )

