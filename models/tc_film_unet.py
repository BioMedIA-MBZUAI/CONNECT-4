"""
TC-UNet Style 4D UNet with Temporal + Graph Conditioning
Adapted from TC-UNet for 4D fMRI generation [B, C, T, D, H, W]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for temporal conditioning (from TC-UNet)."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, D, H, W] or [B, C, D, H, W]
        gamma: torch.Tensor,  # [B, C] or [B, C, T]
        beta: torch.Tensor,  # [B, C] or [B, C, T]
    ) -> torch.Tensor:
        """Apply FiLM modulation."""
        # Reshape gamma and beta to match spatial dimensions
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        return gamma * x + beta


class TemporalGraphFiLMGenerator(nn.Module):
    """Generate FiLM parameters from temporal + graph conditioning."""
    
    def __init__(
        self,
        temporal_cond_dim: int,
        graph_cond_dim: int,
        num_features: int,
    ):
        super().__init__()
        # Combine temporal and graph conditioning
        combined_dim = temporal_cond_dim + graph_cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),
            nn.GELU(),
            nn.Linear(combined_dim * 2, num_features * 2),
        )
    
    def forward(
        self, 
        temporal_cond: torch.Tensor,  # [B, temporal_cond_dim] or [B, T, temporal_cond_dim]
        graph_cond: torch.Tensor,  # [B, graph_cond_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters from combined conditioning.
        
        Returns:
            gamma: [B, num_features] or [B, T, num_features]
            beta: [B, num_features] or [B, T, num_features]
        """
        # Expand graph_cond to match temporal_cond if needed
        if temporal_cond.dim() == 3:  # [B, T, temporal_cond_dim]
            B, T, _ = temporal_cond.shape
            graph_cond_expanded = graph_cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, graph_cond_dim]
            combined = torch.cat([temporal_cond, graph_cond_expanded], dim=-1)  # [B, T, temporal_cond_dim + graph_cond_dim]
        else:  # temporal_cond is [B, temporal_cond_dim]
            combined = torch.cat([temporal_cond, graph_cond], dim=-1)  # [B, temporal_cond_dim + graph_cond_dim]
        
        out = self.mlp(combined)  # [B, num_features * 2] or [B, T, num_features * 2]
        gamma, beta = out.chunk(2, dim=-1)  # Each [B, num_features] or [B, T, num_features]
        return gamma, beta


class Conv4DBlock(nn.Module):
    """4D convolution block (3D spatial + temporal) with temporal + graph FiLM conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_film: bool = False,
        temporal_cond_dim: Optional[int] = None,
        graph_cond_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()
        self.use_film = use_film
        
        # 3D conv for spatial dimensions (D, H, W), temporal handled separately
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Group normalization
        num_groups = min(groups, out_channels)
        while num_groups > 1 and (out_channels % num_groups) != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()
        
        if use_film:
            assert temporal_cond_dim is not None and graph_cond_dim is not None
            self.film_gen = TemporalGraphFiLMGenerator(temporal_cond_dim, graph_cond_dim, out_channels)
            self.film = FiLM(out_channels)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, D, H, W]
        temporal_cond: Optional[torch.Tensor] = None,  # [B, temporal_cond_dim] or [B, T, temporal_cond_dim]
        graph_cond: Optional[torch.Tensor] = None,  # [B, graph_cond_dim]
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, C, T, D, H, W]
            temporal_cond: Temporal conditioning [B, temporal_cond_dim] or [B, T, temporal_cond_dim]
            graph_cond: Graph conditioning [B, graph_cond_dim]
        """
        B, C, T, D, H, W = x.shape
        
        # Process each temporal frame independently with shared conv weights
        # Reshape: [B, C, T, D, H, W] -> [B*T, C, D, H, W]
        x_reshaped = x.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, T, C, D, H, W]
        x_reshaped = x_reshaped.view(B * T, C, D, H, W)  # [B*T, C, D, H, W]
        
        # Apply 3D conv
        x_reshaped = self.conv(x_reshaped)  # [B*T, out_channels, D, H, W]
        x_reshaped = self.norm(x_reshaped)
        
        # Apply FiLM conditioning if enabled
        if self.use_film and temporal_cond is not None and graph_cond is not None:
            # Expand graph_cond for batched processing
            graph_cond_expanded = graph_cond.unsqueeze(1).expand(-1, T, -1).contiguous()  # [B, T, graph_cond_dim]
            graph_cond_expanded = graph_cond_expanded.view(B * T, -1)  # [B*T, graph_cond_dim]
            
            # Handle temporal_cond
            if temporal_cond.dim() == 2:  # [B, temporal_cond_dim]
                temporal_cond_expanded = temporal_cond.unsqueeze(1).expand(-1, T, -1).contiguous()  # [B, T, temporal_cond_dim]
            else:  # [B, T, temporal_cond_dim]
                temporal_cond_expanded = temporal_cond
            temporal_cond_expanded = temporal_cond_expanded.view(B * T, -1)  # [B*T, temporal_cond_dim]
            
            # Generate FiLM parameters
            gamma, beta = self.film_gen(temporal_cond_expanded, graph_cond_expanded)  # Each [B*T, out_channels]
            
            # Reshape for FiLM: [B*T, out_channels] -> [B*T, out_channels, 1, 1, 1]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            x_reshaped = self.film(x_reshaped, gamma, beta)
        
        x_reshaped = self.act(x_reshaped)
        
        # Reshape back: [B*T, out_channels, D, H, W] -> [B, out_channels, T, D, H, W]
        x_reshaped = x_reshaped.view(B, T, -1, D, H, W)  # [B, T, out_channels, D, H, W]
        x = x_reshaped.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, out_channels, T, D, H, W]
        
        return x


class TemporalAttention(nn.Module):
    """Temporal attention across T dimension (adapted from TC-UNet)."""
    
    def __init__(self, dim, heads=4, dim_head=32, chunk_size: int = 2048):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
        # voxel-chunk size: attention is O(n_voxels * T^2), so we process voxels in
        # chunks (checkpointed) to bound memory for long sequences (T up to 128).
        self.chunk_size = chunk_size

    def _attn_chunk(self, xr: torch.Tensor) -> torch.Tensor:
        # xr: [c, C, T, 1, 1] -> [c, C, T]
        qkv = self.to_qkv(xr).squeeze(-1).squeeze(-1).permute(0, 2, 1)   # [c, T, hidden*3]
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.heads), [q, k, v])
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = out.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)          # [c, hidden, T, 1, 1]
        return self.to_out(out).squeeze(-1).squeeze(-1)                 # [c, C, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, D, H, W]
        Returns:
            [B, C, T, D, H, W]
        """
        B, C, T, D, H, W = x.shape
        n = B * D * H * W
        xr = x.permute(0, 3, 4, 5, 1, 2).reshape(n, C, T, 1, 1)        # [n, C, T, 1, 1]

        outs = []
        for i in range(0, n, self.chunk_size):
            xc = xr[i:i + self.chunk_size]
            if self.training and xc.requires_grad:
                oc = checkpoint.checkpoint(self._attn_chunk, xc, use_reentrant=False)
            else:
                oc = self._attn_chunk(xc)
            outs.append(oc)
        out = torch.cat(outs, dim=0)                                    # [n, C, T]
        return out.reshape(B, D, H, W, C, T).permute(0, 4, 5, 1, 2, 3).contiguous()


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # Simple layer norm for 4D
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm with 1 group = LayerNorm

    def forward(self, x, **kwargs):
        # Normalize across the channel dimension (GroupNorm with 1 group),
        # which correctly handles the [B, C, T, D, H, W] layout.
        return self.fn(self.norm(x), **kwargs)


class TCUNet4DFiLM(nn.Module):
    """
    TC-UNet Style 4D UNet with Temporal + Graph Conditioning for fMRI.
    
    Processes [B, C, T, D, H, W] where:
    - T: temporal dimension (time frames)
    - D, H, W: spatial dimensions (depth, height, width)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        num_levels: int = 4,
        temporal_cond_dim: int = 128,
        graph_cond_dim: int = 512,
        mask_cond_dim: int = 1,
        use_checkpoint: bool = False,
        checkpoint_reentrant: bool = False,
        use_temporal_attn: bool = True,
        temporal_attn_heads: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.use_checkpoint = use_checkpoint
        self.checkpoint_reentrant = checkpoint_reentrant
        
        # Temporal conditioning MLP (from TC-UNet)
        time_dim = temporal_cond_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(temporal_cond_dim),
            nn.Linear(temporal_cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.temporal_cond_dim = time_dim
        
        # Initial conv
        self.init_conv = Conv4DBlock(
            in_channels, base_channels,
            use_film=True,
            temporal_cond_dim=self.temporal_cond_dim,
            graph_cond_dim=graph_cond_dim,
        )
        
        # Initial temporal attention
        if use_temporal_attn:
            self.init_temporal_attn = Residual(PreNorm(
                base_channels,
                TemporalAttention(base_channels, heads=temporal_attn_heads)
            ))
        else:
            self.init_temporal_attn = nn.Identity()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_pool = nn.ModuleList()
        
        in_ch = base_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(
                Conv4DBlock(
                    in_ch, out_ch,
                    use_film=True,
                    temporal_cond_dim=self.temporal_cond_dim,
                    graph_cond_dim=graph_cond_dim,
                )
            )
            self.encoder.append(
                Conv4DBlock(
                    out_ch, out_ch,
                    use_film=True,
                    temporal_cond_dim=self.temporal_cond_dim,
                    graph_cond_dim=graph_cond_dim,
                )
            )
            if use_temporal_attn:
                self.encoder.append(
                    Residual(PreNorm(
                        out_ch,
                        TemporalAttention(out_ch, heads=temporal_attn_heads)
                    ))
                )
            else:
                self.encoder.append(nn.Identity())
            
            if i < num_levels - 1:
                # Downsample spatial dimensions only (D, H, W), keep T
                self.encoder_pool.append(nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)))  # Only downsample H, W
            in_ch = out_ch
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i in range(num_levels - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            # Upsample spatial dimensions only
            self.decoder_upsample.append(
                nn.ConvTranspose3d(in_ch, in_ch, (1, 2, 2), stride=(1, 2, 2))
            )
            self.decoder.append(
                Conv4DBlock(
                    in_ch + out_ch, out_ch,
                    use_film=True,
                    temporal_cond_dim=self.temporal_cond_dim,
                    graph_cond_dim=graph_cond_dim,
                )
            )
            self.decoder.append(
                Conv4DBlock(
                    out_ch, out_ch,
                    use_film=True,
                    temporal_cond_dim=self.temporal_cond_dim,
                    graph_cond_dim=graph_cond_dim,
                )
            )
            if use_temporal_attn:
                self.decoder.append(
                    Residual(PreNorm(
                        out_ch,
                        TemporalAttention(out_ch, heads=temporal_attn_heads)
                    ))
                )
            else:
                self.decoder.append(nn.Identity())
        
        # Final output
        self.final_conv = nn.Conv3d(base_channels, out_channels, 1)
        self.out_channels = out_channels
        
        # Mask conditioning (applied as additional input)
        self.mask_proj = nn.Conv3d(mask_cond_dim, in_channels, 1)
    
    @staticmethod
    def _apply_per_frame(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a 3D (D,H,W) module independently to every temporal frame."""
        B, C, T, D, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W)
        y = module(y)
        Dn, Hn, Wn = y.shape[-3:]
        return y.reshape(B, T, C, Dn, Hn, Wn).permute(0, 2, 1, 3, 4, 5).contiguous()

    def _run_block(self, block: nn.Module, x: torch.Tensor, temporal_cond: torch.Tensor, graph_cond: torch.Tensor) -> torch.Tensor:
        """Optionally checkpoint a block to save activation memory."""
        if self.use_checkpoint:
            return checkpoint.checkpoint(
                lambda inp, t_cond, g_cond: block(inp, t_cond, g_cond),
                x,
                temporal_cond,
                graph_cond,
                use_reentrant=self.checkpoint_reentrant,
            )
        return block(x, temporal_cond, graph_cond)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, in_channels, T, D, H, W]
        graph_cond: torch.Tensor,  # [B, graph_cond_dim] - graph conditioning
        temporal_indices: Optional[torch.Tensor] = None,  # [B] or [B, T] - frame indices (0 to T-1)
        mask: Optional[torch.Tensor] = None,  # [B, 1, T, D, H, W] or [B, 1, D, H, W] - mask conditioning
    ) -> torch.Tensor:
        """
        Forward pass with temporal + graph conditioning.
        
        Args:
            x: Input [B, in_channels, T, D, H, W]
            graph_cond: Graph conditioning [B, graph_cond_dim]
            temporal_indices: Frame indices [B] or [B, T] (if None, uses sequential 0..T-1)
            mask: Optional mask [B, 1, T, D, H, W] or [B, 1, D, H, W]
        
        Returns:
            Output [B, T, D, H, W] - 4D fMRI format (channel dimension removed)
        """
        B, C, T, D, H, W = x.shape
        
        # Generate temporal embeddings
        if temporal_indices is None:
            # Use sequential indices 0 to T-1
            temporal_indices = torch.arange(T, device=x.device, dtype=torch.float32).expand(B, -1)  # [B, T]
        elif temporal_indices.dim() == 1:
            # [B] -> [B, T] (broadcast same time to all frames)
            temporal_indices = temporal_indices.unsqueeze(1).expand(-1, T).float()  # [B, T]
        else:
            temporal_indices = temporal_indices.float()  # [B, T]
        
        # Generate temporal conditioning: [B, T] -> [B, T, temporal_cond_dim]
        # Process each time point independently
        temporal_indices_flat = temporal_indices.view(-1)  # [B*T]
        temporal_cond_flat = self.time_mlp(temporal_indices_flat)  # [B*T, temporal_cond_dim]
        temporal_cond = temporal_cond_flat.view(B, T, -1)  # [B, T, temporal_cond_dim]
        
        # Add mask as additional channel if provided
        if mask is not None:
            if mask.dim() == 5:  # [B, 1, D, H, W]
                mask = mask.unsqueeze(2).expand(-1, -1, T, -1, -1, -1)  # [B, 1, T, D, H, W]
            # Project mask to match input channels
            # Process each temporal frame
            mask_reshaped = mask.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, T, 1, D, H, W]
            mask_reshaped = mask_reshaped.view(B * T, 1, D, H, W)  # [B*T, 1, D, H, W]
            mask_feat = self.mask_proj(mask_reshaped)  # [B*T, in_channels, D, H, W]
            mask_feat = mask_feat.view(B, T, C, D, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, in_channels, T, D, H, W]
            x = x + mask_feat
        
        # Initial conv and temporal attention
        x = self.init_conv(x, temporal_cond, graph_cond)
        x = self.init_temporal_attn(x)
        
        r = x.clone()  # Residual connection
        
        # Encoder
        encoder_outputs = []
        idx = 0
        for i in range(self.num_levels):
            x = self._run_block(self.encoder[idx], x, temporal_cond, graph_cond)
            idx += 1
            x = self._run_block(self.encoder[idx], x, temporal_cond, graph_cond)
            idx += 1
            encoder_outputs.append(x)
            # Temporal attention
            x = self.encoder[idx](x)
            idx += 1
            if i < self.num_levels - 1:
                x = self._apply_per_frame(self.encoder_pool[i], x)
        
        # Decoder
        decoder_idx = 0
        for i, upsample in enumerate(self.decoder_upsample):
            x = self._apply_per_frame(upsample, x)
            x = torch.cat([x, encoder_outputs[-(i+2)]], dim=1)
            x = self._run_block(self.decoder[decoder_idx], x, temporal_cond, graph_cond)
            decoder_idx += 1
            x = self._run_block(self.decoder[decoder_idx], x, temporal_cond, graph_cond)
            decoder_idx += 1
            # Temporal attention
            x = self.decoder[decoder_idx](x)
            decoder_idx += 1
        
        # Final output (process each temporal frame)
        B, C_final, T, D, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B, T, C_final, D, H, W]
        x_reshaped = x_reshaped.view(B * T, C_final, D, H, W)  # [B*T, C_final, D, H, W]
        x_reshaped = self.final_conv(x_reshaped)  # [B*T, out_channels, D, H, W]
        
        # Reshape to 4D: [B, T, D, H, W] (remove channel dimension)
        if self.out_channels == 1:
            # Squeeze channel dimension for 4D output
            x_reshaped = x_reshaped.squeeze(1)  # [B*T, D, H, W]
            x_reshaped = x_reshaped.view(B, T, D, H, W)  # [B, T, D, H, W]
        else:
            # If multiple channels, keep as [B, T, out_channels, D, H, W]
            x_reshaped = x_reshaped.view(B, T, self.out_channels, D, H, W)  # [B, T, out_channels, D, H, W]
            # For 4D output, average or select first channel
            x_reshaped = x_reshaped.mean(dim=2)  # [B, T, D, H, W]
        
        return x_reshaped  # [B, T, D, H, W] - 4D fMRI output


if __name__ == "__main__":
    # Test the model
    model = TCUNet4DFiLM(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        num_levels=3,
        temporal_cond_dim=64,
        graph_cond_dim=256,
    ).cuda()
    
    B, C, T, D, H, W = 2, 1, 10, 32, 32, 32
    x = torch.randn(B, C, T, D, H, W).cuda()
    graph_cond = torch.randn(B, 256).cuda()
    temporal_indices = torch.arange(T, dtype=torch.float32).expand(B, -1).cuda()
    
    out = model(x, graph_cond, temporal_indices)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Model test passed!")

