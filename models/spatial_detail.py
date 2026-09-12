"""Quarantined legacy T1-conditioned detail experiment.

This module is retained only so regression tests can characterize and reject
the former recovery path.  Production CONNECT-4 must never instantiate it:
injecting a T1-derived high pass can make a prediction look anatomically sharp
while repeating essentially static texture through time. The Figure-1 v10
decoder must learn spatial detail from the diffusion latent and TC-UNet itself;
``models.connect4`` and configuration validation therefore fail closed if this
path is enabled.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from architecture_contract import FULL_RESOLUTION_DETAIL_CONTRACT


class FullResolutionT1DetailPath(nn.Module):
    """Legacy test fixture; forbidden in every production synthesis profile."""

    contract = FULL_RESOLUTION_DETAIL_CONTRACT

    def __init__(
        self,
        *,
        hidden_channels: int = 8,
        highpass_sigma_voxels: float = 1.0,
        gain_limit: float = 0.25,
        frame_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("detail hidden_channels must be positive")
        if not math.isfinite(highpass_sigma_voxels) or highpass_sigma_voxels <= 0:
            raise ValueError("detail high-pass sigma must be positive and finite")
        if not math.isfinite(gain_limit) or not 0 < gain_limit <= 1:
            raise ValueError("detail gain_limit must lie in (0,1]")
        if (
            isinstance(frame_chunk_size, bool)
            or not isinstance(frame_chunk_size, int)
            or frame_chunk_size < 1
        ):
            raise ValueError("detail frame_chunk_size must be a positive integer")
        radius = max(1, int(math.ceil(3.0 * highpass_sigma_voxels)))
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(
            -0.5 * (coordinate / float(highpass_sigma_voxels)).square()
        )
        kernel_1d /= kernel_1d.sum()
        kernel_3d = (
            kernel_1d[:, None, None]
            * kernel_1d[None, :, None]
            * kernel_1d[None, None, :]
        )
        self.register_buffer(
            "gaussian_kernel",
            kernel_3d[None, None],
            persistent=True,
        )
        self.radius = radius
        self.gain_limit = float(gain_limit)
        self.frame_chunk_size = int(frame_chunk_size)
        self.spatial_refinement = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
        )
        # The spatial template starts as the analytic high pass. The separate
        # zero-initialized gain below keeps the complete residual at zero until
        # paired-target training moves it away from zero.
        nn.init.zeros_(self.spatial_refinement[-1].weight)
        nn.init.zeros_(self.spatial_refinement[-1].bias)
        # Each low-resolution latent frame is projected to the full output grid
        # and receives its own spatial gate.  The former Conv1d gate first
        # averaged over every spatial location and could only scale one static
        # T1 texture map by one scalar per frame.
        self.temporal_gate = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)
        # The complete residual is exactly zero at initialization.  A bounded
        # tanh gain receives the first paired-target gradient; after it moves
        # away from zero, the local refinement and temporal gate also train.
        self.gain_parameter = nn.Parameter(torch.tensor(0.0))

    def high_pass(
        self,
        t1: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a support-normalized high pass with an exactly zero exterior.

        The numerator and valid-support mass are filtered with the same
        zero-padded Gaussian.  This avoids treating the zero exterior as dark
        anatomy at a brain-mask boundary, which otherwise creates a synthetic
        bright rim even when the in-mask T1 is constant.
        """
        if t1.ndim != 5 or t1.shape[1] != 1:
            raise ValueError(f"T1 detail input must be [B,1,D,H,W], got {t1.shape}")
        if not t1.is_floating_point():
            raise ValueError("T1 detail input must be floating point")
        if not torch.isfinite(t1).all():
            raise ValueError("T1 detail input contains NaN or infinity")
        if mask is None:
            support = torch.ones_like(t1)
        else:
            if mask.ndim != 5 or mask.shape != t1.shape:
                raise ValueError("detail mask must match full-resolution T1")
            if not torch.isfinite(mask).all():
                raise ValueError("detail mask contains NaN or infinity")
            support = (mask > 0.5).to(device=t1.device, dtype=t1.dtype)

        masked_t1 = t1 * support
        kernel = self.gaussian_kernel.to(device=t1.device, dtype=t1.dtype)
        numerator = F.conv3d(masked_t1, kernel, padding=self.radius)
        support_mass = F.conv3d(support, kernel, padding=self.radius)
        blurred = numerator / support_mass.clamp_min(torch.finfo(t1.dtype).eps)
        return (masked_t1 - blurred) * support

    def forward(
        self,
        t1: torch.Tensor,
        latent: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 6:
            raise ValueError(
                f"detail temporal source must be [B,C,T,D,H,W], got {latent.shape}"
            )
        if t1.shape[0] != latent.shape[0]:
            raise ValueError("T1/detail latent batch sizes differ")
        if mask is None:
            support = torch.ones_like(t1)
        else:
            if mask.ndim != 5 or mask.shape != t1.shape:
                raise ValueError("detail mask must match full-resolution T1")
            if not torch.isfinite(mask).all():
                raise ValueError("detail mask contains NaN or infinity")
            support = (mask > 0.5).to(device=t1.device, dtype=t1.dtype)
        masked_t1 = t1 * support
        high_pass = self.high_pass(t1, mask=support)
        learned = self.spatial_refinement(
            torch.cat((masked_t1, high_pass), dim=1)
        )
        # Refinement can modulate supported anatomical detail but cannot invent
        # a rim or texture where the mask-normalized high pass is zero.
        spatial = torch.tanh(
            high_pass * (1.0 + 0.25 * torch.tanh(learned))
        ) * support
        gain = self.gain_limit * torch.tanh(self.gain_parameter)
        batch, _, frames, _, _, _ = latent.shape

        def detail_chunk(
            latent_frames: torch.Tensor,
            spatial_map: torch.Tensor,
        ) -> torch.Tensor:
            chunk_frames = latent_frames.shape[2]
            temporal_source = latent_frames.mean(dim=1).reshape(
                batch * chunk_frames, 1, *latent_frames.shape[-3:]
            )
            if temporal_source.shape[-3:] != t1.shape[-3:]:
                temporal_source = F.interpolate(
                    temporal_source,
                    size=t1.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            temporal = self.temporal_gate(temporal_source)
            temporal = temporal.reshape(
                batch, chunk_frames, 1, *t1.shape[-3:]
            ).permute(0, 2, 1, 3, 4, 5)
            temporal = 1.0 + 0.25 * torch.tanh(temporal)
            return gain * spatial_map.unsqueeze(2) * temporal

        chunks = []
        for start in range(0, frames, self.frame_chunk_size):
            latent_chunk = latent[:, :, start : start + self.frame_chunk_size]
            if torch.is_grad_enabled() and (
                latent_chunk.requires_grad or spatial.requires_grad
            ):
                residual_chunk = checkpoint(
                    detail_chunk,
                    latent_chunk,
                    spatial,
                    use_reentrant=False,
                )
            else:
                residual_chunk = detail_chunk(latent_chunk, spatial)
            chunks.append(residual_chunk)
        residual = torch.cat(chunks, dim=2)
        residual = residual * support.to(residual.dtype).unsqueeze(2)
        if not torch.isfinite(residual).all():
            raise RuntimeError("full-resolution detail path produced non-finite values")
        return residual


__all__ = ["FullResolutionT1DetailPath"]
