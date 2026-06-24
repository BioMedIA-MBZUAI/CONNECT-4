"""
Offline native-fMRI -> MNI registration (CONNECT-4).

The 4D rs-fMRI in `fMRI_norm/` is in native scanner space; its NIfTI header
affine does NOT map to MNI (so resampling by header alone is badly misaligned
with the T1/ROI masks). The registered MNI **mean** exists for every subject
(`fMRI_mean_norm/`, same grid as T1/ROI), so we recover the per-subject
native->MNI affine by intensity-registering the native fMRI mean to that
registered mean (dipy, mutual information). We cache the 4x4 world-space
transform; the dataset then applies it cheaply (one grid_sample) to all frames.

    # one subject
    python -m preprocessing.register_fmri_mni --sid B10081264_004
    # a slice of the cohort (for a SLURM array)
    python -m preprocessing.register_fmri_mni --start 0 --stop 500

Transforms are written to  <fMRI_norm>/../mni_xfm/<sid>.npy  (4x4, world space:
native_world = M @ mni_world). Already-done subjects are skipped.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import nibabel as nib

G = Path("/path/to/data")
XFM_DIR = G / "mni_xfm"


def _downsample(data, affine, factor=2):
    """Cheap stride downsample (keeps world affine correct) to speed registration."""
    d = data[::factor, ::factor, ::factor]
    aff = affine.copy()
    aff[:3, :3] = aff[:3, :3] * factor
    return np.ascontiguousarray(d.astype(np.float32)), aff


def register_subject(sid: str, overwrite: bool = False) -> bool:
    from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                     transform_centers_of_mass)
    from dipy.align.transforms import RigidTransform3D, AffineTransform3D

    out = XFM_DIR / f"{sid}.npy"
    if out.exists() and not overwrite:
        return True
    fmri_p = G / "fMRI_norm" / f"{sid}_fMRI.nii.gz"
    fixed_p = G / "fMRI_mean_norm" / f"{sid}_fMRI.nii.gz"
    if not (fmri_p.exists() and fixed_p.exists()):
        return False
    try:
        fm = nib.load(str(fmri_p))
        mov = np.moveaxis(fm.get_fdata().astype(np.float32), 3, 0).mean(0)   # native mean
        mov_aff = fm.affine
        fix_img = nib.load(str(fixed_p))
        fixed = fix_img.get_fdata().astype(np.float32)
        fix_aff = fix_img.affine
        # register at half resolution for speed (transform is world-space, res-free)
        fxd, fxd_aff = _downsample(fixed, fix_aff, factor=2)

        com = transform_centers_of_mass(fxd, fxd_aff, mov, mov_aff)
        metric = MutualInformationMetric(32, None)
        areg = AffineRegistration(metric=metric, level_iters=[100, 50, 10],
                                  sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)
        rig = areg.optimize(fxd, mov, RigidTransform3D(), None, fxd_aff, mov_aff,
                            starting_affine=com.affine)
        aff = areg.optimize(fxd, mov, AffineTransform3D(), None, fxd_aff, mov_aff,
                            starting_affine=rig.affine)
        XFM_DIR.mkdir(parents=True, exist_ok=True)
        # aff.affine: world-space map fixed(MNI) -> moving(native)
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, aff.affine.astype(np.float64))
        os.replace(tmp, out)
        return True
    except Exception as e:  # leave uncached -> dataset falls back to rough reg
        print(f"[reg] {sid} failed: {type(e).__name__}: {e}", flush=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default=None)
    ap.add_argument("--list", default=None, help="text file of scan ids to register")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if a.sid:
        sids = [a.sid]
    elif a.list:
        sids = sorted(Path(a.list).read_text().split())[a.start:a.stop]
    else:
        sids = sorted({os.path.basename(f).replace("_fMRI.nii.gz", "")
                       for f in glob.glob(str(G / "fMRI_norm" / "*_fMRI.nii.gz"))})
        sids = sids[a.start:a.stop]

    done = 0
    for i, sid in enumerate(sids):
        if register_subject(sid, overwrite=a.overwrite):
            done += 1
        if i % 25 == 0:
            print(f"[reg] {i+1}/{len(sids)} processed ({done} cached)", flush=True)
    print(f"[reg] finished: {done}/{len(sids)} transforms cached", flush=True)


if __name__ == "__main__":
    main()
