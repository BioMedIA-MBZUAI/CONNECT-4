"""
T1w structural-MRI preprocessing (CONNECT-4, Figure 1A input).

Steps:
  1. reorient to RAS+ and conform to 128^3 @ 3 mm isotropic,
  2. z-score intensity normalisation inside the brain.

    python -m preprocessing.preprocess_t1 --in sub-01_T1w.nii.gz --out sub-01_T1w_128.nii.gz

Segmentation is produced separately with SynthSeg (run externally by you); use
`preprocess_seg.py` to conform that label map onto the same grid.
"""
from __future__ import annotations

import argparse

import numpy as np
import nibabel as nib

from .conform import conform_volume, zscore_inbrain


def preprocess_t1(in_path: str, out_path: str, normalize: bool = True) -> str:
    img = nib.load(in_path)
    img = conform_volume(img, order=1)                       # 128^3 @ 3 mm
    data = img.get_fdata().astype(np.float32)
    if normalize:
        data = zscore_inbrain(data)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
    print(f"[t1] {in_path} -> {out_path}  shape={data.shape}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--no-normalize", action="store_true")
    a = ap.parse_args()
    preprocess_t1(a.in_path, a.out_path, normalize=not a.no_normalize)
