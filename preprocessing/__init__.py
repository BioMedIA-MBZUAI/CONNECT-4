"""
CONNECT-4 preprocessing.

Brings both modalities onto an explicit, hash-bound, T1-only cohort-common
grid. The paper specifies 3 mm isotropic resolution and rs-fMRI with 128 frames
at TR = 3 s; it does not specify a spatial matrix.

  preprocess_seg    authenticated SynthSeg labels -> conformed to the grid
  preprocess_t1     T1w -> grid + z-score inside authenticated segmentation
  preprocess_fmri   verified fMRIPrep T1w-space BOLD -> smoothing, temporal
                    filtering, 128 frames @ TR 3 s / 3 mm

Segmentation itself is produced **externally** with SynthSeg
(https://github.com/BBillot/SynthSeg); `preprocess_seg` only conforms its output.
"""
from .conform import (
    COMMON_GRID_SCHEMA_VERSION,
    TARGET_VOXEL,
    TARGET_FRAMES,
    TR_SECONDS,
    conform_4d,
    conform_volume,
    load_common_grid_contract,
)
from .preprocess_t1 import preprocess_t1
from .preprocess_seg import preprocess_seg
from .preprocess_fmri import FMRI_PREPROCESSING_SCHEMA_VERSION, preprocess_fmri

__all__ = [
    "conform_volume", "conform_4d",
    "COMMON_GRID_SCHEMA_VERSION", "TARGET_VOXEL", "TARGET_FRAMES", "TR_SECONDS",
    "load_common_grid_contract",
    "FMRI_PREPROCESSING_SCHEMA_VERSION",
    "preprocess_t1", "preprocess_seg", "preprocess_fmri",
]
