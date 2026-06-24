"""
Segmentation conforming (CONNECT-4, Figure 1A "3D Segmentation").

Segmentation is **run externally by the user with SynthSeg**
(Billot et al., https://github.com/BBillot/SynthSeg). This step only brings the
externally-produced label map onto the CONNECT-4 grid (128^3 @ 3 mm isotropic,
nearest-neighbour so labels are preserved).

    python -m preprocessing.preprocess_seg --in sub-01_synthseg.nii.gz --out seg_128.nii.gz

The conformed labels drive the mask patches / ROI statistics (Fig 1A), the ROI
graph and ROI-coverage hyperedges (Fig 1B), and the measured ROI volumes used by
the normative / atrophy module (`normative/`).
"""
from __future__ import annotations

import argparse

import nibabel as nib

from .conform import conform_volume


def preprocess_seg(in_path: str, out_path: str) -> str:
    """Conform an external SynthSeg label map to 128^3 @ 3 mm (nearest-neighbour)."""
    seg = nib.load(in_path)
    seg = conform_volume(seg, order=0)                       # order=0 -> preserve labels
    nib.save(seg, out_path)
    print(f"[seg] {in_path} -> {out_path}  shape={seg.shape}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="external SynthSeg label map")
    ap.add_argument("--out", dest="out_path", required=True)
    a = ap.parse_args()
    preprocess_seg(a.in_path, a.out_path)
