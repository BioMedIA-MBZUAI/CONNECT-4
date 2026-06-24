"""
Optimized on-the-fly registration for fMRI to T1w space.
Memory-efficient implementation that processes temporal frames in batches.
"""
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional
import torch.nn.functional as F


def register_fmri_to_t1w_optimized(
    fmri: torch.Tensor,  # [1, T, D, H, W] or [T, D, H, W]
    t1w: torch.Tensor,  # [1, 1, D, H, W] or [1, D, H, W]
    fmri_affine: np.ndarray,  # 4x4 affine matrix from fMRI NIfTI
    t1w_affine: np.ndarray,  # 4x4 affine matrix from T1w NIfTI
    target_shape: Optional[Tuple[int, int, int]] = None,
    batch_temporal: int = 10,  # Process this many temporal frames at once
) -> torch.Tensor:
    """
    Memory-efficient registration of fMRI to T1w space.
    Processes temporal frames in batches to reduce memory usage.
    
    Args:
        fmri: fMRI volume [1, T, D, H, W] or [T, D, H, W]
        t1w: T1w structural volume [1, 1, D, H, W] or [1, D, H, W]
        fmri_affine: Affine matrix from fMRI NIfTI header
        t1w_affine: Affine matrix from T1w NIfTI header
        target_shape: Target spatial shape (D, H, W). If None, uses T1w shape.
        batch_temporal: Number of temporal frames to process at once
    
    Returns:
        registered_fmri: Registered fMRI [1, T, D, H, W] in T1w space
    """
    # Ensure batch dimension
    if fmri.ndim == 4:
        fmri = fmri.unsqueeze(0)  # [1, T, D, H, W]
    
    if t1w.ndim == 3:
        t1w = t1w.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    elif t1w.ndim == 4:
        t1w = t1w.unsqueeze(0)  # [1, 1, D, H, W]
    
    B, T, D_fmri, H_fmri, W_fmri = fmri.shape
    _, _, D_t1w, H_t1w, W_t1w = t1w.shape
    
    # Use T1w shape as target if not specified
    if target_shape is None:
        target_shape = (D_t1w, H_t1w, W_t1w)
    
    D_target, H_target, W_target = target_shape
    
    # Compute transformation from T1w voxel space to fMRI voxel space
    fmri_affine_inv = np.linalg.inv(fmri_affine)
    transform_matrix = fmri_affine_inv @ t1w_affine  # [4, 4]
    
    device = fmri.device
    
    # Create coordinate grid in target (T1w) voxel space (compute once, reuse for all frames)
    # NIfTI convention: (x, y, z) = (width, height, depth)
    d_indices = torch.arange(D_target, dtype=torch.float32, device=device)
    h_indices = torch.arange(H_target, dtype=torch.float32, device=device)
    w_indices = torch.arange(W_target, dtype=torch.float32, device=device)
    
    D_grid, H_grid, W_grid = torch.meshgrid(d_indices, h_indices, w_indices, indexing='ij')
    
    # Convert to homogeneous coordinates using NIfTI convention: (x, y, z) = (width, height, depth)
    coords_t1w_vox = torch.stack([
        W_grid.flatten(),  # x = width (NIfTI convention)
        H_grid.flatten(),  # y = height
        D_grid.flatten(),  # z = depth
        torch.ones(D_target * H_target * W_target, device=device)
    ], dim=0)  # [4, D*H*W]
    
    # Transform to fMRI voxel space
    transform_matrix_torch = torch.from_numpy(transform_matrix).float().to(device)
    coords_fmri_vox = transform_matrix_torch @ coords_t1w_vox  # [4, D*H*W]
    coords_fmri_vox = coords_fmri_vox[:3, :]  # [3, D*H*W] where [x, y, z] = [width, height, depth]
    
    # Reshape to [D_target, H_target, W_target, 3]
    coords_fmri_vox = coords_fmri_vox.permute(1, 0).reshape(D_target, H_target, W_target, 3)
    # coords_fmri_vox[:, :, :, 0] = x (width), [1] = y (height), [2] = z (depth)
    
    # Clamp coordinates to valid range
    coords_fmri_vox_clamped = coords_fmri_vox.clone()
    coords_fmri_vox_clamped[:, :, :, 0] = torch.clamp(coords_fmri_vox[:, :, :, 0], 0, W_fmri - 1)  # x (width)
    coords_fmri_vox_clamped[:, :, :, 1] = torch.clamp(coords_fmri_vox[:, :, :, 1], 0, H_fmri - 1)  # y (height)
    coords_fmri_vox_clamped[:, :, :, 2] = torch.clamp(coords_fmri_vox[:, :, :, 2], 0, D_fmri - 1)  # z (depth)
    
    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects (x, y, z) = (width, height, depth) in [-1, 1] range
    coords_normalized = torch.zeros_like(coords_fmri_vox_clamped)
    
    if W_fmri > 1:
        coords_normalized[:, :, :, 0] = 2.0 * coords_fmri_vox_clamped[:, :, :, 0] / (W_fmri - 1) - 1.0  # x (width)
    else:
        coords_normalized[:, :, :, 0] = 0.0
    
    if H_fmri > 1:
        coords_normalized[:, :, :, 1] = 2.0 * coords_fmri_vox_clamped[:, :, :, 1] / (H_fmri - 1) - 1.0  # y (height)
    else:
        coords_normalized[:, :, :, 1] = 0.0
    
    if D_fmri > 1:
        coords_normalized[:, :, :, 2] = 2.0 * coords_fmri_vox_clamped[:, :, :, 2] / (D_fmri - 1) - 1.0  # z (depth)
    else:
        coords_normalized[:, :, :, 2] = 0.0
    
    coords_5d = coords_normalized.unsqueeze(0)  # [1, D, H, W, 3] - reuse for all frames
    
    # Process temporal frames in batches for memory efficiency
    registered_frames = []
    for t_start in range(0, T, batch_temporal):
        t_end = min(t_start + batch_temporal, T)
        fmri_batch = fmri[0, t_start:t_end, :, :, :]  # [batch_size, D_fmri, H_fmri, W_fmri]
        
        # Add channel dimension: [batch_size, 1, D, H, W]
        fmri_batch_5d = fmri_batch.unsqueeze(1)
        
        # Expand coords for batch: [batch_size, D, H, W, 3]
        batch_size = t_end - t_start
        coords_batch = coords_5d.expand(batch_size, -1, -1, -1, -1)
        
        # Apply grid_sample to entire batch at once
        registered_batch = F.grid_sample(
            fmri_batch_5d,
            coords_batch,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [batch_size, 1, D_target, H_target, W_target]
        
        # Remove channel dimension and add to list
        for t_idx in range(batch_size):
            registered_frames.append(registered_batch[t_idx, 0, :, :, :])
    
    # Stack temporal frames: [T, D_target, H_target, W_target]
    registered_fmri = torch.stack(registered_frames, dim=0)
    
    # Add batch dimension: [1, T, D_target, H_target, W_target]
    registered_fmri = registered_fmri.unsqueeze(0)
    
    return registered_fmri

