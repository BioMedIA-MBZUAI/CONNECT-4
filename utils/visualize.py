"""
Visualization utilities for brain scan predictions using nilearn.

This matches the standalone debug script the user validated:
  - Column 1: T1
  - Column 2: mean target fMRI (over time)
  - Column 3: first frame of 4D prediction (DiT or decoder output)

All three columns share the same coordinate system and slice selection,
using nilearn.image.mean_img and nilearn.plotting.plot_epi.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:
    import nibabel as nib
    from nilearn import image
    from nilearn.plotting import plot_epi, find_xyz_cut_coords
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("[visualize] Warning: nilearn/matplotlib not available; visualizations disabled", flush=True)


def _tensor_to_3d_t1(t1w: Tensor, patient_idx: int) -> np.ndarray:
    """
    Convert T1 tensor [B, 1, D, H, W] (or [B, D, H, W]) to 3D volume [D, H, W].
    """
    t = t1w[patient_idx].detach().cpu().float()
    if t.ndim == 4:  # [C, D, H, W]
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"[visualize] T1 expected 3D/4D, got {t.shape}")
    return t.numpy()


def _tensor_to_4d_fmri(vol: Tensor, patient_idx: int, name: str) -> np.ndarray:
    """
    Convert a tensor to 4D fMRI volume [D, H, W, T] in numpy, for target or prediction.

    Accepts:
      - [B, 1, T, D, H, W]
      - [B, T, D, H, W]
      - [B, C, T, D, H, W] (takes C=0)
    """
    v = vol[patient_idx].detach().cpu().float()

    # Remove channel if present
    if v.ndim == 5:  # [C, T, D, H, W]
        v = v[0]
    elif v.ndim == 4:
        # [T, D, H, W] already
        pass
    else:
        raise ValueError(f"[visualize] {name} expected [C,T,D,H,W] or [T,D,H,W], got {v.shape}")

    if v.ndim != 4:
        raise ValueError(f"[visualize] {name} expected 4D after squeeze, got {v.shape}")

    # v: [T, D, H, W] -> [D, H, W, T]
    v_np = v.numpy()
    return np.transpose(v_np, (1, 2, 3, 0))


def _compute_vmin_vmax(data: np.ndarray, default_vmin: float = 0.0, default_vmax: float = 1.0):
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return default_vmin, default_vmax
    vmin = np.percentile(vals, 1)
    vmax = np.percentile(vals, 99)
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _get_cut_coords(ref_img: "nib.Nifti1Image", num_slices: int = 7, view: str = "axial"):
    """
    Compute cut_coords in mm, ensuring coordinates are within valid image bounds.
    Matches the debug_viz_playground.py logic:
      - Use central line in the orthogonal axes
      - Spread slices between 15% and 85% of the FOV in the chosen axis
    """
    # This also verifies the image is valid in nilearn's sense
    _ = find_xyz_cut_coords(ref_img)

    data = ref_img.get_fdata()
    affine = ref_img.affine

    if view == "axial":
        display_mode = "z"
        z_voxels = np.linspace(int(data.shape[2] * 0.15), int(data.shape[2] * 0.85), num_slices)
        coords = []
        for z_vox in z_voxels:
            vox_coord = np.array([data.shape[0] // 2, data.shape[1] // 2, z_vox, 1])
            mm_coord = affine @ vox_coord
            coords.append(mm_coord[2])  # z in mm
    elif view == "coronal":
        display_mode = "y"
        y_voxels = np.linspace(int(data.shape[1] * 0.15), int(data.shape[1] * 0.85), num_slices)
        coords = []
        for y_vox in y_voxels:
            vox_coord = np.array([data.shape[0] // 2, y_vox, data.shape[2] // 2, 1])
            mm_coord = affine @ vox_coord
            coords.append(mm_coord[1])  # y in mm
    elif view == "sagittal":
        display_mode = "x"
        x_voxels = np.linspace(int(data.shape[0] * 0.15), int(data.shape[0] * 0.85), num_slices)
        coords = []
        for x_vox in x_voxels:
            vox_coord = np.array([x_vox, data.shape[1] // 2, data.shape[2] // 2, 1])
            mm_coord = affine @ vox_coord
            coords.append(mm_coord[0])  # x in mm
    else:
        raise ValueError(f"Unknown view: {view}")

    return np.array(coords), display_mode


def visualize_brain_scans(
    t1w: Tensor,           # [B, 1, D, H, W]
    target: Tensor,        # [B, 1, T, D, H, W] or [B, T, D, H, W]
    pred: Tensor,          # [B, C, T, D, H, W] or [B, 1, T, D, H, W]
    output_dir: Path,
    epoch: int,
    step: int,
    view: str = "axial",   # 'axial', 'coronal', 'sagittal'
    num_slices: int = 7,
    patient_idx: int = 0,
    mask: Optional[Tensor] = None,
    affine: Optional[np.ndarray] = None,
    **_: dict,
):
    """
    Main visualization entrypoint used by training.

    Columns:
      - Column 1: T1
      - Column 2: mean target fMRI over time
      - Column 3: first frame of 4D prediction (DiT or decoder output)

    All columns share the same display_mode and cut_coords, as in the
    debug_viz_playground.py script.
    """
    if not NILEARN_AVAILABLE:
        print("[visualize] nilearn/matplotlib not available; skipping visualization.", flush=True)
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------
    # SIMPLIFIED: Convert tensors to numpy volumes - NO AFFINE, NO REORIENTATION
    # Just extract raw data and display as-is
    # --------------------------
    t1_vol = _tensor_to_3d_t1(t1w, patient_idx=patient_idx)  # [D, H, W]

    # Target 4D: [D, H, W, T]
    target_4d = _tensor_to_4d_fmri(target, patient_idx=patient_idx, name="target")  # [D, H, W, T]
    # Mean target over time
    target_mean = target_4d.mean(axis=3)  # [D, H, W]

    # Prediction 4D: [D, H, W, T]
    pred_4d = _tensor_to_4d_fmri(pred, patient_idx=patient_idx, name="pred")  # [D, H, W, T]
    # Mean prediction over time
    pred_mean = pred_4d.mean(axis=3)  # [D, H, W]

    # --------------------------
    # Build simple NIfTI images with identity affine - NO TRANSFORMATIONS
    # --------------------------
    identity_affine = np.eye(4)
    t1_img = nib.Nifti1Image(t1_vol.astype(np.float32), identity_affine)
    target_mean_img = nib.Nifti1Image(target_mean.astype(np.float32), identity_affine)
    pred_mean_img = nib.Nifti1Image(pred_mean.astype(np.float32), identity_affine)

    # --------------------------
    # Compute cut_coords from target_mean_img (fMRI) - use its own coordinate system
    # --------------------------
    cut_coords, display_mode = _get_cut_coords(target_mean_img, num_slices=num_slices, view=view)
    actual_num_slices = len(cut_coords)
    
    # DEBUG: Print slice information
    target_shape = target_mean.shape
    t1_shape = t1_vol.shape
    print(f"[visualize] T1 shape: {t1_shape}, Target fMRI shape: {target_shape}, Requested slices: {num_slices}, Using {actual_num_slices} slices", flush=True)

    # --------------------------
    # Compute vmin/vmax per image
    # --------------------------
    t1_vmin, t1_vmax = _compute_vmin_vmax(t1_img.get_fdata(), 0.0, 1.0)
    tgt_vmin, tgt_vmax = _compute_vmin_vmax(target_mean_img.get_fdata(), 0.0, 1.0)
    pred_vmin, pred_vmax = _compute_vmin_vmax(pred_mean_img.get_fdata(), 0.0, 1.0)

    # --------------------------
    # Create figure: T1 / mean target / first-frame prediction
    # Use actual_num_slices instead of num_slices to avoid IndexError
    # --------------------------
    fig = plt.figure(figsize=(15, 5 * actual_num_slices), facecolor="black")
    gs = gridspec.GridSpec(actual_num_slices, 3, figure=fig, hspace=0.1, wspace=0.05)

    titles = ["T1", "Mean Target fMRI", "First Prediction Frame"]

    for slice_idx in range(actual_num_slices):
        coord = cut_coords[slice_idx]

        # Column 0: T1
        ax1 = fig.add_subplot(gs[slice_idx, 0])
        ax1.set_facecolor("black")
        plot_epi(
            t1_img,
            cut_coords=[coord],
            display_mode=display_mode,
            cmap="gray",
            black_bg=True,
            axes=ax1,
            title="" if slice_idx > 0 else titles[0],
            output_file=None,
            figure=fig,
            vmin=t1_vmin,
            vmax=t1_vmax,
        )
        ax1.axis("off")

        # Column 1: mean target
        ax2 = fig.add_subplot(gs[slice_idx, 1])
        ax2.set_facecolor("black")
        plot_epi(
            target_mean_img,
            cut_coords=[coord],
            display_mode=display_mode,
            cmap="magma",
            black_bg=True,
            axes=ax2,
            title="" if slice_idx > 0 else titles[1],
            output_file=None,
            figure=fig,
            vmin=tgt_vmin,
            vmax=tgt_vmax,
        )
        ax2.axis("off")

        # Column 2: mean prediction
        ax3 = fig.add_subplot(gs[slice_idx, 2])
        ax3.set_facecolor("black")
        plot_epi(
            pred_mean_img,
            cut_coords=[coord],
            display_mode=display_mode,
            cmap="magma",
            black_bg=True,
            axes=ax3,
            title="" if slice_idx > 0 else titles[2],
            output_file=None,
            figure=fig,
            vmin=pred_vmin,
            vmax=pred_vmax,
        )
        ax3.axis("off")

    fig.suptitle(
        f"Epoch {epoch:03d} Step {step:05d} – {view.capitalize()} view",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.995,
    )

    filename = f"sample_epoch{epoch:03d}_step{step:05d}_p{patient_idx:02d}_{view}_nilearn.png"
    filepath = output_dir / filename
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return filepath


# Alias for backward compatibility
visualize_brain_scans_nilearn = visualize_brain_scans


