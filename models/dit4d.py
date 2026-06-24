"""
4D DiT Blocks for Temporal Synthesis
Adapted from the Guide implementation for graph token inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DiTBlock4D(nn.Module):
    """
    DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Adapted for 4D (temporal) synthesis.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        enable_flash_attn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        # Use SDPA (scaled_dot_product_attention) instead of MultiheadAttention for memory efficiency
        # SDPA uses Flash Attention when available and doesn't materialize attention weights
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # adaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def modulate(self, norm_func, x, shift, scale):
        """
        Modulate features with shift and scale (matches reference implementation).
        Applies norm first, then modulates - with float32 casting for numerical stability.
        """
        dtype = x.dtype
        x = norm_func(x.to(torch.float32)).to(dtype)
        x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        return x
    
    def forward(
        self,
        x: torch.Tensor,  # [B, num_patches, hidden_size]
        c: torch.Tensor,  # [B, hidden_size] - conditioning (timestep + text)
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tokens [B, num_patches, hidden_size]
            c: Conditioning vector [B, hidden_size]
        
        Returns:
            Output tokens [B, num_patches, hidden_size]
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation using SDPA (Flash Attention)
        # Use modulate function that applies norm inside (matches reference)
        x_modulated = self.modulate(self.norm1, x, shift_msa, scale_msa)
        
        # Compute Q, K, V
        B, N, D = x_modulated.shape
        qkv = self.qkv(x_modulated)  # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, N, D]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        
        # Use scaled_dot_product_attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, H, N, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        attn_out = self.out_proj(attn_out)  # [B, N, D]
        
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation (matches reference pattern)
        x_modulated = self.modulate(self.norm2, x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class FinalLayer4D(nn.Module):
    """Final layer for DiT 4D."""
    
    def __init__(
        self,
        hidden_size: int,
        patch_size: tuple,
        out_channels: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, np.prod(patch_size) * out_channels)
    
    def modulate(self, norm_func, x, shift, scale):
        """Modulate features (matches reference implementation)."""
        dtype = x.dtype
        x = norm_func(x.to(torch.float32)).to(dtype)
        x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        return x
    
    def forward(
        self,
        x: torch.Tensor,  # [B, num_patches, hidden_size]
        c: torch.Tensor,  # [B, hidden_size]
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """Timestep embedding for diffusion."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.register_buffer('timestep_embedding', self._build_timestep_embedding(hidden_size))
    
    def _build_timestep_embedding(self, dim: int):
        """Build sinusoidal timestep embedding base frequencies."""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        return emb
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps using sinusoidal positional encoding.
        
        Args:
            t: Timesteps [B] - can be any positive integer values
        
        Returns:
            Embeddings [B, hidden_size]
        """
        # Convert timesteps to float and compute sinusoidal embedding
        # t_emb shape: [B, half_dim]
        t_emb = t.float().unsqueeze(-1) * self.timestep_embedding.unsqueeze(0)  # [B, half_dim]
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)  # [B, dim]
        return self.mlp(t_emb)


class DiT4D(nn.Module):
    """
    4D DiT model for temporal synthesis.
    Takes graph tokens and produces 4D output.
    """
    
    def __init__(
        self,
        input_size: tuple = (16, 128, 128, 128),  # (T, D, H, W) for 4D volume
        in_channels: int = 1,
        patch_size: tuple = (1, 1, 16, 16),  # (T, D, H, W) patch size
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        input_dim: Optional[int] = None,  # Input dimension (e.g., from fusion output)
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        
        # Validate that input_size and patch_size are 4D
        if len(input_size) != 4:
            raise ValueError(f"input_size must be 4D (T, D, H, W), got {len(input_size)}D: {input_size}")
        if len(patch_size) != 4:
            raise ValueError(f"patch_size must be 4D (T, D, H, W), got {len(patch_size)}D: {patch_size}")
        
        # Compute number of patches for output volume
        # input_size is the output volume dimensions in patch space [T, D, H, W]
        # patch_size is the size of each output patch [T, D, H, W]
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(4)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0] if patch_size[0] > 0 else input_size[0]
        self.num_spatial = num_patches // self.num_temporal if self.num_temporal > 0 else num_patches
        
        # Input projection layer if input_dim != hidden_size
        if input_dim is not None and input_dim != hidden_size:
            self.input_proj = nn.Linear(input_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock4D(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        
        # Final layer
        # Output dimension: patch_size * out_channels
        # For patch_size=[1,2,2] and out_channels=2: 1*2*2*2 = 8
        self.final_layer = FinalLayer4D(
            hidden_size,
            patch_size,
            self.out_channels,
        )
        
        # Verify final layer output dimension
        expected_final_dim = np.prod(patch_size) * self.out_channels
        if self.final_layer.linear.out_features != expected_final_dim:
            raise ValueError(
                f"Final layer output dimension mismatch: "
                f"expected {expected_final_dim}, got {self.final_layer.linear.out_features}"
            )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(
        self,
        x: torch.Tensor,  # [B, num_patches, patch_size * out_channels]
    ) -> torch.Tensor:
        """
        Convert patches back to 4D volume.
        
        Returns:
            4D volume [B, out_channels, T, D, H, W]
        """
        B = x.shape[0]
        c = self.out_channels
        # input_size is [T, D, H, W], patch_size is [T, D, H, W]
        t, d, h, w = [self.input_size[i] // self.patch_size[i] for i in range(4)]
        pt, pd, ph, pw = self.patch_size
        
        # Check input shape (using .size() instead of .shape for torch.compile compatibility)
        expected_patch_dim = np.prod(self.patch_size) * c
        if x.size(-1) != expected_patch_dim:
            raise ValueError(
                f"Expected last dimension to be {expected_patch_dim} (patch_size * out_channels), "
                f"but got {x.size(-1)}. Input shape: {x.shape}"
            )
        
        if x.size(1) != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, but got {x.size(1)}. Input shape: {x.shape}"
            )
        
        # Reshape: [B, num_patches, patch_size * out_channels] -> [B, t, d, h, w, pt, pd, ph, pw, c]
        x = x.reshape(B, t, d, h, w, pt, pd, ph, pw, c)
        # Permute to [B, c, t, pt, d, pd, h, ph, w, pw]
        x = x.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8)
        # Reshape to [B, c, T, D, H, W] where T=t*pt, D=d*pd, H=h*ph, W=w*pw
        x = x.reshape(B, c, t * pt, d * pd, h * ph, w * pw)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,  # [B, num_patches, hidden_size] - graph tokens
        t: torch.Tensor,  # [B] - timesteps
        y: Optional[torch.Tensor] = None,  # [B, hidden_size] - optional conditioning
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Graph tokens [B, num_patches, hidden_size]
            t: Timesteps [B]
            y: Optional conditioning [B, hidden_size]
        
        Returns:
            4D output [B, out_channels, T, D, H, W]
        """
        # Project input to hidden_size if needed
        x = self.input_proj(x)  # [B, num_fusion_patches, hidden_size]
        
        # Handle mismatch between fusion patches and DiT expected patches
        # Fusion outputs 512 patches (8×8×8 spatial), DiT may expect different number
        # Use tensor operations instead of shape checks to avoid graph breaks in torch.compile
        num_fusion_patches = x.size(1)  # Get as int (causes graph break but necessary)
        if num_fusion_patches != self.num_patches:
            if num_fusion_patches < self.num_patches:
                # Expand: repeat patches to match DiT expected number
                repeat_factor = (self.num_patches + num_fusion_patches - 1) // num_fusion_patches
                x = x.repeat(1, repeat_factor, 1)
                x = x[:, :self.num_patches, :]  # Trim to exact size
            else:  # num_fusion_patches > self.num_patches
                # Downsample: take first num_patches
                x = x[:, :self.num_patches, :]
        
        # Timestep embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]
        
        # Combine with optional conditioning
        if y is not None:
            c = t_emb + y
        else:
            c = t_emb
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # [B, num_patches, patch_size * out_channels]
        
        # Debug: Check shape before unpatchify
        expected_patch_dim = np.prod(self.patch_size) * self.out_channels
        if x.shape[-1] != expected_patch_dim:
            raise RuntimeError(
                f"Final layer output shape mismatch: got {x.shape}, "
                f"expected [B, {self.num_patches}, {expected_patch_dim}]. "
                f"patch_size={self.patch_size}, out_channels={self.out_channels}"
            )
        
        # Unpatchify to 4D
        x = self.unpatchify(x)  # [B, out_channels, T, D, H, W]
        
        return x

