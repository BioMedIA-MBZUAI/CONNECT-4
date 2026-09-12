"""Reusable blocks for the paper's conditional 4D Diffusion Transformer.

The complete denoiser and DDIM trajectory live in :mod:`dit4d_temporal`.
Keeping only its building blocks here prevents an alternative non-diffusion
generator from silently becoming a second model path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiTBlock4D(nn.Module):
    """Figure-1 3D-window attention, global time attention, cross-attention, MLP.

    Figure 1C labels 3D Window Attention while the prose calls the spatial and
    temporal interactions global. The explicit v10 interpretation uses one
    configured 3D window large enough to cover the complete paper patch grid,
    followed by global temporal attention at every spatial patch. Smaller grids
    are validity-masked inside that window rather than interpolated.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        spatial_window_size: Tuple[int, int, int] = (8, 8, 8),
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.spatial_window_size = tuple(int(value) for value in spatial_window_size)
        if len(self.spatial_window_size) != 3 or any(
            value < 1 for value in self.spatial_window_size
        ):
            raise ValueError("spatial_window_size must be a positive 3D tuple")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm_spatial = nn.LayerNorm(hidden_size, eps=1e-6)
        self.spatial_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.spatial_out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_temporal = nn.LayerNorm(hidden_size, eps=1e-6)
        self.temporal_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.temporal_out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )

    @property
    def qkv(self) -> nn.Linear:
        """Compatibility view of the Figure-1 spatial-window QKV projection."""

        return self.spatial_qkv

    @property
    def out_proj(self) -> nn.Linear:
        """Compatibility view of the Figure-1 spatial-window output projection."""

        return self.spatial_out_proj

    @staticmethod
    def _modulate(
        norm: nn.LayerNorm,
        values: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        dtype = values.dtype
        values = norm(values.float()).to(dtype)
        return values * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _attention(
        self,
        tokens: torch.Tensor,
        qkv_projection: nn.Linear,
        output_projection: nn.Linear,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, length, dimension = tokens.shape
        query, key, value = qkv_projection(tokens).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            heads(query),
            heads(key),
            heads(value),
            attn_mask=attention_mask,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return output_projection(attended)

    def _spatial_window_attention(
        self,
        tokens: torch.Tensor,
        shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        batch, _, channels = tokens.shape
        time, depth, height, width = shape
        wd, wh, ww = self.spatial_window_size
        if depth > wd or height > wh or width > ww:
            raise ValueError(
                "configured 3D attention window must cover the complete spatial "
                "patch grid to retain the manuscript's stated global interaction"
            )
        values = tokens.reshape(batch, time, depth, height, width, channels)
        padded = values.new_zeros((batch, time, wd, wh, ww, channels))
        padded[:, :, :depth, :height, :width] = values
        windows = padded.reshape(batch * time, wd * wh * ww, channels)
        valid = torch.zeros(
            (wd, wh, ww), device=tokens.device, dtype=torch.bool
        )
        valid[:depth, :height, :width] = True
        valid = valid.reshape(1, 1, 1, -1).expand(batch * time, -1, -1, -1)
        attended = self._attention(
            windows,
            self.spatial_qkv,
            self.spatial_out_proj,
            attention_mask=valid,
        )
        attended = attended * valid.reshape(batch * time, -1, 1).to(attended.dtype)
        attended = attended.reshape(batch, time, wd, wh, ww, channels)
        return attended[:, :, :depth, :height, :width].reshape_as(tokens)

    def _temporal_attention(
        self,
        tokens: torch.Tensor,
        shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        batch, _, channels = tokens.shape
        time, depth, height, width = shape
        spatial = depth * height * width
        sequences = tokens.reshape(batch, time, spatial, channels).permute(0, 2, 1, 3)
        sequences = sequences.reshape(batch * spatial, time, channels)
        attended = self._attention(
            sequences,
            self.temporal_qkv,
            self.temporal_out_proj,
        )
        return attended.reshape(batch, spatial, time, channels).permute(
            0, 2, 1, 3
        ).reshape_as(tokens)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        spatiotemporal_shape: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or c.shape != (x.shape[0], self.hidden_size):
            raise ValueError("x must be [B,N,H] and c must be [B,H]")
        (
            shift_spatial,
            scale_spatial,
            gate_spatial,
            shift_temporal,
            scale_temporal,
            gate_temporal,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(12, dim=1)

        if spatiotemporal_shape is None or (
            len(spatiotemporal_shape) != 4
            or int(np.prod(spatiotemporal_shape)) != x.shape[1]
        ):
            raise ValueError(
                "spatiotemporal_shape [T,D,H,W] must exactly match the token count"
            )
        spatial_input = self._modulate(
            self.norm_spatial, x, shift_spatial, scale_spatial
        )
        x = x + gate_spatial.unsqueeze(1) * self._spatial_window_attention(
            spatial_input, spatiotemporal_shape
        )
        temporal_input = self._modulate(
            self.norm_temporal, x, shift_temporal, scale_temporal
        )
        x = x + gate_temporal.unsqueeze(1) * self._temporal_attention(
            temporal_input, spatiotemporal_shape
        )

        if context is not None:
            if (
                context.ndim != 3
                or context.shape[0] != x.shape[0]
                or context.shape[2] != self.hidden_size
            ):
                raise ValueError("context must have shape [B,M,H]")
            query = self._modulate(
                self.norm_cross, x, shift_cross, scale_cross
            )
            cross, _ = self.cross_attn(
                query, context, context, need_weights=False
            )
            x = x + gate_cross.unsqueeze(1) * cross

        mlp_input = self._modulate(self.norm2, x, shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)


class FinalLayer4D(nn.Module):
    """adaLN output projection from tokens to four-dimensional patches."""

    def __init__(
        self, hidden_size: int, patch_size: tuple, out_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(
            hidden_size, int(np.prod(patch_size)) * int(out_channels)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        dtype = x.dtype
        x = self.norm_final(x.float()).to(dtype)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal diffusion-timestep embedding followed by a learned MLP."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 2 or hidden_size % 2:
            raise ValueError("hidden_size must be an even integer of at least two")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        half = hidden_size // 2
        denominator = max(1, half - 1)
        frequencies = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * (-np.log(10_000.0) / denominator)
        )
        self.register_buffer("timestep_embedding", frequencies)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = timestep.float().unsqueeze(-1) * self.timestep_embedding.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.mlp(embedding)


__all__ = ["DiTBlock4D", "FinalLayer4D", "TimestepEmbedder"]
