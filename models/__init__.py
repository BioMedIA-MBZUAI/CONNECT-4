"""
CONNECT-4 model modules, organised by the panels of Figure 1.

A) Input encoders : BrainIACWrapper (ViT), ModernBERTWrapper (text)
B) Graph fusion   : MultiModalFusion (graph-attention + hyperedge + nodes2token)
C) DiT generation : DiT4D / DiT4DTemporal
D) Temporal UNet  : TCUNet4DFiLM
E) Losses         : Connect4Loss and its six components
"""
from .brainiac_wrapper import BrainIACWrapper
from .modernbert_wrapper import ModernBERTWrapper
from .fusion import MultiModalFusion
from .dit4d import DiT4D
from .dit4d_temporal import DiT4DTemporal
from .tc_film_unet import TCUNet4DFiLM
from .slimbrain_wrapper import SlimBrainEncoder
from .losses import (
    Connect4Loss,
    SSIM3DLoss,
    VoxelIntensityLoss,
    VolumeLoss,
    RegionHistogramLoss,
    PerceptualLoss,
    FCMatrixLoss,
)

__all__ = [
    "BrainIACWrapper",
    "ModernBERTWrapper",
    "MultiModalFusion",
    "DiT4D",
    "DiT4DTemporal",
    "TCUNet4DFiLM",
    "SlimBrainEncoder",
    "Connect4Loss",
    "SSIM3DLoss",
    "VoxelIntensityLoss",
    "VolumeLoss",
    "RegionHistogramLoss",
    "PerceptualLoss",
    "FCMatrixLoss",
]
