"""
rs-fMRI preprocessing (CONNECT-4, generation target).

Brings 4D rs-fMRI onto the same grid as the T1w and to the paper's acquisition
spec: **128^3 voxels, 3 mm isotropic, 128 frames, TR = 3 s**.

Steps:
  1. (optional) rigid motion correction across frames,
  2. register the run to the (already conformed) T1w reference,
  3. conform every frame to 128^3 @ 3 mm isotropic,
  4. force exactly 128 frames (truncate / reflect-pad) and write TR = 3 s,
  5. per-voxel temporal z-score normalisation.

    python -m preprocessing.preprocess_fmri --fmri sub-01_bold.nii.gz \
        --t1 sub-01_T1w_128.nii.gz --out sub-01_bold_128.nii.gz

Motion correction is best done with FSL `mcflirt` or SynthMorph beforehand; the
`--mcflirt` flag will call it if available, otherwise step 1 is skipped.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

from utils.registration import register_fmri_to_t1w_nib
from .conform import conform_4d, set_num_frames, TARGET_FRAMES, TR_SECONDS


def _motion_correct(fmri_path: str) -> str:
    """Rigid motion correction with FSL mcflirt if present, else passthrough."""
    if shutil.which("mcflirt") is None:
        print("[fmri] mcflirt not found — skipping motion correction")
        return fmri_path
    out = str(Path(tempfile.mkdtemp()) / "mc")
    subprocess.run(["mcflirt", "-in", fmri_path, "-out", out], check=True)
    out_nii = out + ".nii.gz"
    print(f"[fmri] motion-corrected -> {out_nii}")
    return out_nii


def _temporal_zscore(data4d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = data4d.mean(axis=-1, keepdims=True)
    sd = data4d.std(axis=-1, keepdims=True)
    return np.clip((data4d - mu) / (sd + eps), -3.0, 3.0).astype(np.float32)


def preprocess_fmri(fmri_path: str, t1_path: str, out_path: str,
                    mcflirt: bool = False, normalize: bool = True) -> str:
    if mcflirt:
        fmri_path = _motion_correct(fmri_path)

    fmri_img = nib.load(fmri_path)
    t1_img = nib.load(t1_path)

    # 2) register the run into T1 space (returns [1, T, D, H, W])
    reg = register_fmri_to_t1w_nib(fmri_img, t1_img)[0].numpy()        # [T, D, H, W]
    reg = np.transpose(reg, (1, 2, 3, 0))                              # [D, H, W, T]
    reg_img = nib.Nifti1Image(reg.astype(np.float32), nib.as_closest_canonical(t1_img).affine)

    # 3) conform to 128^3 @ 3 mm isotropic, TR=3 s
    conf = conform_4d(reg_img, order=1)
    data = conf.get_fdata().astype(np.float32)                        # [128,128,128,T]

    # 4) exactly 128 frames
    data = set_num_frames(data, TARGET_FRAMES)

    # 5) per-voxel temporal z-score
    if normalize:
        data = _temporal_zscore(data)

    out_img = nib.Nifti1Image(data, conf.affine, conf.header)
    out_img.header.set_zooms(list(out_img.header.get_zooms()[:3]) + [TR_SECONDS])
    nib.save(out_img, out_path)
    print(f"[fmri] {fmri_path} -> {out_path}  shape={data.shape}  TR={TR_SECONDS}s")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri", required=True)
    ap.add_argument("--t1", required=True, help="conformed T1w (128^3 @ 3mm)")
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--mcflirt", action="store_true", help="run FSL mcflirt motion correction")
    ap.add_argument("--no-normalize", action="store_true")
    a = ap.parse_args()
    preprocess_fmri(a.fmri, a.t1, a.out_path, mcflirt=a.mcflirt, normalize=not a.no_normalize)
