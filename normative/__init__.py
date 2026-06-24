"""
External normative regional-volume modelling for CONNECT-4 (Figure 1A).

Wraps the Potvin et al. subcortical-volume norms (mmc2.xlsm) and converts
measured ROI volumes into per-subject structural-atrophy descriptions.
"""
from .subcortical_norms import SubcorticalNorms
from .atrophy import AtrophyDescriber, ASEG_TO_NORM_LABEL

__all__ = ["SubcorticalNorms", "AtrophyDescriber", "ASEG_TO_NORM_LABEL"]
