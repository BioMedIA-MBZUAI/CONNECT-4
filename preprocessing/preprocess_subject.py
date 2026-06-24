"""
One-shot per-subject preprocessing driver (CONNECT-4).

Runs the full chain for a subject:
    T1w        -> conform 128^3 @ 3 mm + z-score          (preprocess_t1)
    SynthSeg   -> conform external label map to the grid   (preprocess_seg)
    rs-fMRI    -> register to T1, conform, 128 frames@TR3  (preprocess_fmri)

Segmentation must be produced **externally** with SynthSeg
(https://github.com/BBillot/SynthSeg) and passed via `--seg`.

    python -m preprocessing.preprocess_subject \
        --t1 sub-01_T1w.nii.gz --seg sub-01_synthseg.nii.gz \
        --fmri sub-01_bold.nii.gz --out_dir derivatives/sub-01

Outputs in `--out_dir`: T1w_128.nii.gz, seg_128.nii.gz, bold_128.nii.gz
(all 128^3 @ 3 mm; bold is 128 frames at TR = 3 s).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .preprocess_t1 import preprocess_t1
from .preprocess_seg import preprocess_seg
from .preprocess_fmri import preprocess_fmri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--seg", required=True, help="external SynthSeg label map for this subject")
    ap.add_argument("--fmri", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mcflirt", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t1_128 = preprocess_t1(a.t1, str(out / "T1w_128.nii.gz"))
    preprocess_seg(a.seg, str(out / "seg_128.nii.gz"))
    preprocess_fmri(a.fmri, t1_128, str(out / "bold_128.nii.gz"), mcflirt=a.mcflirt)
    print(f"[done] subject preprocessed -> {out}")


if __name__ == "__main__":
    main()
