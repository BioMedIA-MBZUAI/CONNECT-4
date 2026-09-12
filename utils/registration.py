"""Debug-only header resampling used by the non-production preprocessing path.

Production CONNECT-4 targets must instead carry the verified fMRIPrep
BOLD-to-T1w transform and evidence chain. Keeping only this narrowly used
helper avoids exposing the obsolete on-the-fly registration API to training.
"""
from __future__ import annotations

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch


def register_fmri_to_t1w_nib(
    fmri_img: nib.Nifti1Image,
    t1_ref_img: nib.Nifti1Image,
    batch_temporal: int = 8,
    order: int = 1,
    cval: float = 0.0,
) -> torch.Tensor:
    """Header-resample a 4D image to a T1 grid for explicit diagnostics only.

    Returns ``[1,T,D,H,W]`` float32. This function does not estimate or prove
    anatomical registration and therefore cannot create a production-eligible
    target.
    """
    if batch_temporal < 1:
        raise ValueError("batch_temporal must be positive")
    fmri_img = nib.as_closest_canonical(fmri_img)
    t1_ref_img = nib.as_closest_canonical(t1_ref_img)
    fmri_data = fmri_img.get_fdata(dtype=np.float32)
    if fmri_data.ndim != 4 or fmri_data.shape[3] < 1:
        raise ValueError(f"debug fMRI input must be non-empty 4D, got {fmri_data.shape}")

    chunks = []
    for start in range(0, fmri_data.shape[3], batch_temporal):
        frames = []
        for offset in range(min(batch_temporal, fmri_data.shape[3] - start)):
            frame = nib.Nifti1Image(
                fmri_data[..., start + offset], fmri_img.affine, fmri_img.header
            )
            try:
                result = resample_from_to(
                    frame,
                    t1_ref_img,
                    order=order,
                    mode="constant",
                    cval=cval,
                    force_resample=True,
                    copy_header=True,
                )
            except TypeError:  # nibabel < 5.2 compatibility
                result = resample_from_to(
                    frame, t1_ref_img, order=order, mode="constant", cval=cval
                )
            frames.append(result.get_fdata(dtype=np.float32))
        chunks.append(np.stack(frames, axis=0))
    return torch.from_numpy(np.concatenate(chunks, axis=0)).unsqueeze(0)


__all__ = ["register_fmri_to_t1w_nib"]
