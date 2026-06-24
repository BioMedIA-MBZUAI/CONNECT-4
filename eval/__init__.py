"""Evaluation: synthetic-vs-real rs-fMRI metrics and visualisation."""
from .metrics import compute_all
from .visualize import plot_real_vs_synthetic, save_fmri_nifti

__all__ = ["compute_all", "plot_real_vs_synthetic", "save_fmri_nifti"]
