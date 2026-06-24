"""
Lightweight on-the-fly registration helpers.
"""
from typing import Optional
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import torch


def register_fmri_to_t1w_nib(
    fmri_img: nib.Nifti1Image,
    t1_ref_img: nib.Nifti1Image,
    batch_temporal: int = 8,
    order: int = 1,
    cval: float = 0.0,
) -> torch.Tensor:
    """
    Register a 4D fMRI NIfTI to T1 reference space using nibabel's resample_from_to.
    Processes temporal frames in batches to avoid high memory use.
    Returns a torch tensor shaped [1, T, D, H, W] in float32.
    """
    # Reorient both images to canonical (RAS+) to avoid axis flips
    fmri_img = nib.as_closest_canonical(fmri_img)
    t1_ref_img = nib.as_closest_canonical(t1_ref_img)

    fmri_data = fmri_img.get_fdata().astype(np.float32)
    T = fmri_data.shape[3] if fmri_data.ndim == 4 else 1
    frames = []
    for t in range(0, T, batch_temporal):
        chunk = fmri_data[..., t : t + batch_temporal]  # [X,Y,Z,k]
        out_chunk = []
        for i in range(chunk.shape[3]):
            frame_img = nib.Nifti1Image(chunk[..., i], fmri_img.affine)
            # Use kwargs only if supported; fallback for older nibabel versions
            try:
                res = resample_from_to(
                    frame_img,
                    t1_ref_img,
                    order=order,
                    mode="constant",
                    cval=cval,
                    force_resample=True,
                    copy_header=True,
                )
            except TypeError:
                # Older nibabel without force_resample/copy_header
                res = resample_from_to(
                    frame_img,
                    t1_ref_img,
                    order=order,
                    mode="constant",
                    cval=cval,
                )
            out_chunk.append(res.get_fdata().astype(np.float32))
        frames.append(np.stack(out_chunk, axis=0))  # [k, D, H, W]
    registered = np.concatenate(frames, axis=0)  # [T, D, H, W]
    return torch.from_numpy(registered).unsqueeze(0)  # [1, T, D, H, W]
"""
Registration utilities for aligning fMRI to structural MRI (T1w) while preserving temporal dimension.
"""
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional
import torch.nn.functional as F


def register_fmri_to_t1w(
    fmri: torch.Tensor,  # [1, T, D, H, W] or [T, D, H, W]
    t1w: torch.Tensor,  # [1, 1, D, H, W] or [1, D, H, W]
    fmri_affine: np.ndarray,  # 4x4 affine matrix from fMRI NIfTI
    t1w_affine: np.ndarray,  # 4x4 affine matrix from T1w NIfTI
    target_shape: Optional[Tuple[int, int, int]] = None,
    method: str = 'affine',  # 'affine' or 'rigid'
) -> torch.Tensor:
    """
    Register fMRI to T1w space while preserving temporal dimension.
    
    Args:
        fmri: fMRI volume [1, T, D, H, W] or [T, D, H, W]
        t1w: T1w structural volume [1, 1, D, H, W] or [1, D, H, W]
        fmri_affine: Affine matrix from fMRI NIfTI header
        t1w_affine: Affine matrix from T1w NIfTI header
        target_shape: Target spatial shape (D, H, W). If None, uses T1w shape.
        method: Registration method ('affine' or 'rigid')
    
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
    # Step 1: Transform T1w voxel coords -> world coords: p_world = A_t1w @ p_t1w_vox
    # Step 2: Transform world coords -> fMRI voxel coords: p_fmri_vox = A_fmri^-1 @ p_world
    # Combined: p_fmri_vox = A_fmri^-1 @ A_t1w @ p_t1w_vox
    
    fmri_affine_inv = np.linalg.inv(fmri_affine)
    transform_matrix = fmri_affine_inv @ t1w_affine  # [4, 4]
    
    # Debug output
    print(f"[registration] T1w shape: {t1w.shape}, fMRI shape: {fmri.shape}")
    print(f"[registration] Target shape: {target_shape}")
    print(f"[registration] T1w affine origin: {t1w_affine[:3, 3]}")
    print(f"[registration] fMRI affine origin: {fmri_affine[:3, 3]}")
    
    device = fmri.device
    
    # Create coordinate grid in target (T1w) voxel space
    # grid_sample expects coordinates in (x, y, z) = (width, height, depth) order
    # We need to create a grid in T1w space and transform to fMRI space
    
    # Create meshgrid in T1w voxel space: (depth, height, width) = (D, H, W)
    d_indices = torch.arange(D_target, dtype=torch.float32, device=device)
    h_indices = torch.arange(H_target, dtype=torch.float32, device=device)
    w_indices = torch.arange(W_target, dtype=torch.float32, device=device)
    
    D_grid, H_grid, W_grid = torch.meshgrid(d_indices, h_indices, w_indices, indexing='ij')
    # Shape: [D_target, H_target, W_target]
    
    # Convert to homogeneous coordinates: [D, H, W, 1]
    # Note: NIfTI uses (i, j, k) = (x, y, z) = (width, height, depth) in voxel space
    # But our tensors are [D, H, W] = (depth, height, width)
    # For affine transform, we use (x, y, z) = (width, height, depth) = (W, H, D)
    coords_t1w_vox = torch.stack([
        W_grid.flatten(),  # x = width (NIfTI convention)
        H_grid.flatten(),  # y = height
        D_grid.flatten(),  # z = depth
        torch.ones(D_target * H_target * W_target, device=device)  # homogeneous
    ], dim=0)  # [4, D*H*W]
    
    # Transform to fMRI voxel space
    transform_matrix_torch = torch.from_numpy(transform_matrix).float().to(device)
    coords_fmri_vox = transform_matrix_torch @ coords_t1w_vox  # [4, D*H*W]
    
    # Extract x, y, z (remove homogeneous coordinate)
    coords_fmri_vox = coords_fmri_vox[:3, :]  # [3, D*H*W] where [x, y, z] = [width, height, depth]
    
    # Reshape to [D_target, H_target, W_target, 3]
    # coords_fmri_vox[0] = x (width), [1] = y (height), [2] = z (depth)
    coords_fmri_vox = coords_fmri_vox.permute(1, 0).reshape(D_target, H_target, W_target, 3)
    # Now coords_fmri_vox[:, :, :, 0] = x (width), [1] = y (height), [2] = z (depth)
    
    # Clamp coordinates to valid range
    coords_fmri_vox_clamped = coords_fmri_vox.clone()
    # Clamp x (width) to [0, W_fmri-1]
    coords_fmri_vox_clamped[:, :, :, 0] = torch.clamp(coords_fmri_vox[:, :, :, 0], 0, W_fmri - 1)
    # Clamp y (height) to [0, H_fmri-1]
    coords_fmri_vox_clamped[:, :, :, 1] = torch.clamp(coords_fmri_vox[:, :, :, 1], 0, H_fmri - 1)
    # Clamp z (depth) to [0, D_fmri-1]
    coords_fmri_vox_clamped[:, :, :, 2] = torch.clamp(coords_fmri_vox[:, :, :, 2], 0, D_fmri - 1)
    
    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects (x, y, z) = (width, height, depth) in [-1, 1] range
    coords_normalized = torch.zeros_like(coords_fmri_vox_clamped)
    
    # Normalize x (width) dimension
    if W_fmri > 1:
        coords_normalized[:, :, :, 0] = 2.0 * coords_fmri_vox_clamped[:, :, :, 0] / (W_fmri - 1) - 1.0
    else:
        coords_normalized[:, :, :, 0] = 0.0
    
    # Normalize y (height) dimension
    if H_fmri > 1:
        coords_normalized[:, :, :, 1] = 2.0 * coords_fmri_vox_clamped[:, :, :, 1] / (H_fmri - 1) - 1.0
    else:
        coords_normalized[:, :, :, 1] = 0.0
    
    # Normalize z (depth) dimension
    if D_fmri > 1:
        coords_normalized[:, :, :, 2] = 2.0 * coords_fmri_vox_clamped[:, :, :, 2] / (D_fmri - 1) - 1.0
    else:
        coords_normalized[:, :, :, 2] = 0.0
    
    coords_fmri_space = coords_normalized
    
    # Process each temporal frame
    registered_frames = []
    for t in range(T):
        fmri_frame = fmri[0, t, :, :, :]  # [D_fmri, H_fmri, W_fmri]
        
        # Add channel dimension for grid_sample: [1, 1, D, H, W]
        fmri_frame_5d = fmri_frame.unsqueeze(0).unsqueeze(0)
        
        # Reshape coords for grid_sample: [1, D, H, W, 3]
        coords_5d = coords_fmri_space.unsqueeze(0)
        
        # Apply grid_sample (expects [N, C, D, H, W] and [N, D, H, W, 3])
        registered_frame = F.grid_sample(
            fmri_frame_5d,
            coords_5d,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [1, 1, D_target, H_target, W_target]
        
        registered_frames.append(registered_frame.squeeze(0).squeeze(0))  # [D_target, H_target, W_target]
    
    # Stack temporal frames: [T, D_target, H_target, W_target]
    registered_fmri = torch.stack(registered_frames, dim=0)
    
    # Add batch dimension: [1, T, D_target, H_target, W_target]
    registered_fmri = registered_fmri.unsqueeze(0)
    
    return registered_fmri


def register_fmri_to_t1w_from_paths(
    fmri_path: Path,
    t1w_path: Path,
    target_shape: Optional[Tuple[int, int, int]] = None,
    method: str = 'affine',
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and register fMRI to T1w from file paths.
    
    Args:
        fmri_path: Path to fMRI NIfTI file
        t1w_path: Path to T1w NIfTI file
        target_shape: Target spatial shape (D, H, W). If None, uses T1w shape.
        method: Registration method ('affine' or 'rigid')
        normalize: Whether to normalize intensities
    
    Returns:
        registered_fmri: Registered fMRI [1, T, D, H, W]
        t1w: T1w volume [1, 1, D, H, W]
    """
    # Load NIfTI files
    fmri_img = nib.load(str(fmri_path))
    t1w_img = nib.load(str(t1w_path))
    
    # Get data and affines
    fmri_data = fmri_img.get_fdata().astype(np.float32)
    t1w_data = t1w_img.get_fdata().astype(np.float32)
    
    fmri_affine = fmri_img.affine
    t1w_affine = t1w_img.affine
    
    # Convert to torch tensors
    if fmri_data.ndim == 4:
        # 4D fMRI: [D, H, W, T] -> [1, T, D, H, W]
        fmri_tensor = torch.from_numpy(fmri_data).permute(3, 0, 1, 2).unsqueeze(0)
    elif fmri_data.ndim == 3:
        # 3D: [D, H, W] -> [1, 1, D, H, W]
        fmri_tensor = torch.from_numpy(fmri_data).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D fMRI, got {fmri_data.shape}")
    
    # T1w is always 3D: [D, H, W] -> [1, 1, D, H, W]
    t1w_tensor = torch.from_numpy(t1w_data).unsqueeze(0).unsqueeze(0)
    
    # Normalize if requested
    if normalize:
        fmri_mean = fmri_tensor.mean()
        fmri_std = fmri_tensor.std()
        if fmri_std > 0:
            fmri_tensor = (fmri_tensor - fmri_mean) / (fmri_std + 1e-6)
        
        t1w_mean = t1w_tensor.mean()
        t1w_std = t1w_tensor.std()
        if t1w_std > 0:
            t1w_tensor = (t1w_tensor - t1w_mean) / (t1w_std + 1e-6)
    
    # Register fMRI to T1w
    registered_fmri = register_fmri_to_t1w(
        fmri_tensor,
        t1w_tensor,
        fmri_affine,
        t1w_affine,
        target_shape=target_shape,
        method=method,
    )
    
    # Debug: Print shapes to verify alignment
    print(f"[registration] T1w shape: {t1w_tensor.shape}")
    print(f"[registration] Original fMRI shape: {fmri_tensor.shape}")
    print(f"[registration] Registered fMRI shape: {registered_fmri.shape}")
    print(f"[registration] T1w affine:\n{t1w_affine}")
    print(f"[registration] fMRI affine:\n{fmri_affine}")
    print(f"[registration] Transform matrix:\n{transform_matrix}")
    
    return registered_fmri, t1w_tensor

