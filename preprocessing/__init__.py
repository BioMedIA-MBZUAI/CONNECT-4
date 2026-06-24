"""
CONNECT-4 preprocessing.

Brings both modalities onto the paper's common grid:
    128 x 128 x 128 voxels, 3 mm isotropic; rs-fMRI: 128 frames, TR = 3 s.

  preprocess_t1     T1w  -> conform 128^3 @ 3 mm + intensity z-score
  preprocess_seg    SynthSeg labels (run externally) -> conformed to the grid
  preprocess_fmri   bold -> register to T1, conform, 128 frames, temporal z-score

Segmentation itself is produced **externally** with SynthSeg
(https://github.com/BBillot/SynthSeg); `preprocess_seg` only conforms its output.
"""
from .conform import conform_volume, conform_4d, TARGET_SHAPE, TARGET_VOXEL, TARGET_FRAMES, TR_SECONDS
from .preprocess_t1 import preprocess_t1
from .preprocess_seg import preprocess_seg
from .preprocess_fmri import preprocess_fmri

__all__ = [
    "conform_volume", "conform_4d",
    "TARGET_SHAPE", "TARGET_VOXEL", "TARGET_FRAMES", "TR_SECONDS",
    "preprocess_t1", "preprocess_seg", "preprocess_fmri",
]
