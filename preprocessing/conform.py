"""
Shared spatial conforming for CONNECT-4.

Per the paper, **both** T1w and rs-fMRI are brought to a common grid:
    shape       = 128 x 128 x 128
    voxel size  = 3 mm isotropic
and rs-fMRI has **128 frames** acquired at **TR = 3 s**.

`conform_volume` resamples a 3D NIfTI to that grid (trilinear for intensity
images, nearest-neighbour for label maps). `conform_4d` applies it per frame.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.processing import conform as _nib_conform

TARGET_SHAPE = (128, 128, 128)     # D, H, W
TARGET_VOXEL = (3.0, 3.0, 3.0)     # mm, isotropic
TARGET_FRAMES = 128                # T
TR_SECONDS = 3.0                   # repetition time


def conform_volume(img: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """Resample a 3D image to 128^3 @ 3 mm isotropic (order=0 for labels)."""
    img = nib.as_closest_canonical(img)
    return _nib_conform(
        img, out_shape=TARGET_SHAPE, voxel_size=TARGET_VOXEL,
        order=order, cval=0.0, orientation="RAS",
    )


def conform_4d(img: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """Conform a 4D image frame-by-frame; sets TR=3 s in the header."""
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    if data.ndim == 3:
        data = data[..., None]
    T = data.shape[3]
    frames = []
    for t in range(T):
        f = nib.Nifti1Image(np.asarray(data[..., t], dtype=np.float32), img.affine, img.header)
        frames.append(conform_volume(f, order=order).get_fdata().astype(np.float32))
    out = np.stack(frames, axis=-1)                      # [128,128,128,T]
    ref = conform_volume(nib.Nifti1Image(data[..., 0].astype(np.float32), img.affine), order=order)
    new = nib.Nifti1Image(out, ref.affine, ref.header)
    zooms = list(new.header.get_zooms()[:3]) + [TR_SECONDS]
    new.header.set_zooms(zooms)
    return new


def set_num_frames(data4d: np.ndarray, n: int = TARGET_FRAMES) -> np.ndarray:
    """Force the temporal dimension (last axis) to exactly `n` frames.

    Truncates if longer; reflect-pads if shorter.
    """
    T = data4d.shape[-1]
    if T == n:
        return data4d
    if T > n:
        return data4d[..., :n]
    # shorter: tile (wrap) until we reach exactly n frames
    reps = int(np.ceil(n / T))
    return np.concatenate([data4d] * reps, axis=-1)[..., :n]


def zscore_inbrain(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score using statistics inside the (nonzero) brain, clipped to +/-3."""
    brain = vol[np.abs(vol) > 0]
    mu = brain.mean() if brain.size else vol.mean()
    sd = brain.std() if brain.size else vol.std()
    return np.clip((vol - mu) / (sd + eps), -3.0, 3.0)
