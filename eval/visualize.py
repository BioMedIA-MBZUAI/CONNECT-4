"""
Visualisation of real vs synthetic rs-fMRI (magma colormap).

`plot_real_vs_synthetic` renders, for a chosen axial slice and a set of time
frames, the real and synthetic volumes side by side in the **magma** colormap,
plus their temporal-mean and absolute-difference maps. Designed to drop straight
into a paper figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_numpy_tdhw(x) -> np.ndarray:
    """Accept [B,C,T,D,H,W]/[B,T,D,H,W]/[T,D,H,W]; return [T,D,H,W] (first item)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 4:
        x = x[0]
    if x.ndim == 3:                       # [D,H,W] single frame
        x = x[None]
    return x


def _robust_window(arr: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0):
    """Robust (percentile) intensity window, ignoring non-finite values."""
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(a, lo_p)), float(np.percentile(a, hi_p))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _tri_planes(vol: np.ndarray, idx):
    """
    Extract axial / sagittal / coronal slices from a [D, H, W] volume at the
    given (d, h, w) indices. Returns (axial, sagittal, coronal), each oriented
    upright for display.
    """
    # vol axes follow the MNI grid: axis0 = X (L-R), axis1 = Y (A-P), axis2 = Z (I-S).
    d, h, w = idx                          # d: X-index, h: Y-index, w: Z-index
    axial = np.rot90(vol[:, :, w])         # constant Z (axis2) -> [X, Y]
    sagittal = np.rot90(vol[d, :, :])      # constant X (axis0) -> [Y, Z]
    coronal = np.rot90(vol[:, h, :])       # constant Y (axis1) -> [X, Z]
    return axial, sagittal, coronal


def plot_real_vs_synthetic(
    real,
    synthetic,
    out_path: str,
    slice_idx: Optional[Sequence[int]] = None,
    cmap: str = "magma",
    title: str = "CONNECT-4: real vs synthetic rs-fMRI",
    **_,
):
    """
    Save a magma figure comparing real and synthetic fMRI in the three
    anatomical planes.

    Rows:    real / synthetic (fake) / |difference|
    Columns: Axial / Sagittal / Coronal

    The temporal-mean volume is shown. The most informative slice per axis
    (max variance of the real mean) is chosen automatically. Each ROW uses its
    own robust intensity window (2-98th percentile) so real, fake and difference
    are all clearly visible even when their numeric ranges differ.
    """
    real = _to_numpy_tdhw(real)
    syn = _to_numpy_tdhw(synthetic)
    T = min(real.shape[0], syn.shape[0])
    rmean = real[:T].mean(0)               # [D, H, W]
    smean = syn[:T].mean(0)
    D, H, W = rmean.shape

    # informative slice per axis = max variance of the real mean along that axis
    if slice_idx is None:
        d = int(rmean.reshape(D, -1).var(axis=1).argmax())
        h = int(rmean.transpose(1, 0, 2).reshape(H, -1).var(axis=1).argmax())
        w = int(rmean.transpose(2, 0, 1).reshape(W, -1).var(axis=1).argmax())
    else:
        d, h, w = slice_idx
    idx = (d, h, w)

    real_row = list(_tri_planes(rmean, idx))
    syn_row = list(_tri_planes(smean, idx))
    diff_row = [np.abs(a - b) for a, b in zip(real_row, syn_row)]

    rows = [
        ("real", real_row, *_robust_window(np.concatenate([x.ravel() for x in real_row]))),
        ("synthetic (fake)", syn_row, *_robust_window(np.concatenate([x.ravel() for x in syn_row]))),
        ("|difference|", diff_row, *_robust_window(np.concatenate([x.ravel() for x in diff_row]))),
    ]
    col_titles = [f"Axial (z={w})", f"Sagittal (x={d})", f"Coronal (y={h})"]

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.0))
    for ri, (name, imgs, lo, hi) in enumerate(rows):
        im = None
        for ci, img in enumerate(imgs):
            ax = axes[ri, ci]
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.set_ylabel(name, fontsize=12)
        fig.colorbar(im, ax=axes[ri, -1], fraction=0.046, pad=0.02)

    fig.suptitle(f"{title}  (temporal mean)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_fmri_nifti(vol, out_path: str, affine: Optional[np.ndarray] = None):
    """Save a 4D fMRI volume [T,D,H,W] (or [B,C,T,D,H,W]) as a NIfTI (.nii.gz)."""
    import nibabel as nib
    v = _to_numpy_tdhw(vol)                 # [T, D, H, W]
    v = np.transpose(v, (1, 2, 3, 0))       # NIfTI wants [D, H, W, T]
    if affine is None:
        affine = np.eye(4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v.astype(np.float32), affine), out_path)
    return out_path
