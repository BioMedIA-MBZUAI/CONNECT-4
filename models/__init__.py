"""
CONNECT-4 model modules, organised by the panels of Figure 1.

A) Input encoders : BrainIACWrapper (ViT), ModernBERTWrapper (text)
B) Graph fusion   : MultiModalFusion (graph-attention + hyperedge + nodes2token)
C) DiT generation : DiT4DTemporal (conditional 4D DDIM denoiser)
D) Temporal UNet  : TCUNet4DFiLM
E) Losses         : Connect4Loss, Figure-1E components, and temporal coherence
"""
from .brainiac_wrapper import BrainIACWrapper
from .modernbert_wrapper import ModernBERTWrapper
from .fusion import MultiModalFusion
from .dit4d_temporal import DDIMScheduler, DiT4DTemporal
from .tc_film_unet import DiTToTCUNetProjection, TCUNet4DFiLM
from .losses import (
    Connect4Loss,
    SSIM3DLoss,
    VoxelIntensityLoss,
    VolumeLoss,
    RegionHistogramLoss,
    TemporalCoherenceLoss,
    PerceptualLoss,
    FCMatrixLoss,
)

__all__ = [
    "BrainIACWrapper",
    "ModernBERTWrapper",
    "MultiModalFusion",
    "DiT4DTemporal",
    "DDIMScheduler",
    "TCUNet4DFiLM",
    "DiTToTCUNetProjection",
    "Connect4Loss",
    "SSIM3DLoss",
    "VoxelIntensityLoss",
    "VolumeLoss",
    "RegionHistogramLoss",
    "TemporalCoherenceLoss",
    "PerceptualLoss",
    "FCMatrixLoss",
]
