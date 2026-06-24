"""
Novel DiT for Temporal Generation: Using Diffusion Timesteps as Temporal Dimension

KEY POINT: The hypergraph token generation (Steps 1-5) remains COMPLETELY UNCHANGED.
Only the DiT forward pass interprets timesteps as temporal frame indices instead of noise levels.

Input: fused_tokens from hypergraph [B, num_patches, output_dim] - UNCHANGED
Output: Single temporal frame [B, out_channels, D, H, W] instead of full 4D volume
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from models.dit4d import DiT4D, DiTBlock4D, FinalLayer4D, TimestepEmbedder


class TemporalContextEncoder(nn.Module):
    """
    Encodes previous temporal frames for conditioning.
    This allows the model to use temporal context when generating frame t.
    """
    
    def __init__(self, in_channels: int, hidden_size: int, num_frames: int = 1):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        
        # 3D CNN to encode previous frames
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels * num_frames, hidden_size // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size // 4),
            nn.GELU(),
            nn.Conv3d(hidden_size // 4, hidden_size // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_size // 2),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
    
    def forward(self, prev_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_frames: [B, C, num_frames, D, H, W] - previous temporal frames
        Returns:
            temporal_tokens: [B, hidden_size] - encoded temporal context
        """
        B, C, T_prev, D, H, W = prev_frames.shape
        # Reshape to [B, C*T_prev, D, H, W]
        x = prev_frames.reshape(B, C * T_prev, D, H, W)
        return self.encoder(x)  # [B, hidden_size]


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention to temporal context (previous frames).
    Allows current frame generation to attend to previous frames.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        x: torch.Tensor,  # [B, num_patches, hidden_size] - current tokens
        temporal_context: torch.Tensor,  # [B, hidden_size] - encoded previous frames
    ) -> torch.Tensor:
        """
        Cross-attention: current tokens attend to temporal context.
        """
        B, N, D = x.shape
        
        # Expand temporal context to match number of patches
        # Simple approach: broadcast temporal context to all patches
        temporal_expanded = temporal_context.unsqueeze(1).expand(B, N, D)  # [B, N, D]
        
        # Self-attention with temporal bias
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(temporal_expanded)  # [B, N, D]
        v = self.v_proj(temporal_expanded)  # [B, N, D]
        
        # Multi-head attention
        q = q.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, N, d]
        k = k.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, N, d]
        v = v.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B, H, N, d]
        
        attn = (q @ k.transpose(-2, -1)) / np.sqrt(D // self.num_heads)  # [B, H, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.out_proj(out)
        
        return x + out  # Residual connection


class DiT4DTemporal(DiT4D):
    """
    Modified DiT that uses timesteps as temporal frame indices.
    
    IMPORTANT: Input tokens from hypergraph fusion remain UNCHANGED.
    This class only modifies how timesteps are interpreted and what is output.
    
    Key differences from standard DiT:
    1. Timesteps represent temporal frames (0 to T-1), NOT noise levels
    2. Outputs single frame [B, C, D, H, W] instead of full 4D volume [B, C, T, D, H, W]
    3. Can condition on previous frames for temporal context
    4. Input: fused_tokens [B, num_patches, output_dim] - SAME AS BEFORE
    """
    
    def __init__(
        self,
        input_size: tuple = (128, 128, 128),  # (D, H, W) - single frame spatial dimensions
        in_channels: int = 1,
        patch_size: tuple = (16, 16, 16),  # (D, H, W) patch size for single frame
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        input_dim: Optional[int] = None,
        num_temporal_frames: int = 16,  # Number of temporal frames in sequence
        use_temporal_context: bool = True,  # Whether to use previous frames
        num_context_frames: int = 1,  # Number of previous frames to use
    ):
        # Initialize base DiT with 3D input (single frame)
        # Note: We override input_size to be 3D (D, H, W) instead of 4D (T, D, H, W)
        super().__init__(
            input_size=(1, *input_size),  # Add dummy temporal dim for compatibility
            in_channels=in_channels,
            patch_size=(1, *patch_size),  # Add dummy temporal dim
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
            input_dim=input_dim,
        )
        
        # Override input_size and patch_size for 3D (single frame)
        self.input_size = input_size  # (D, H, W)
        self.patch_size = patch_size  # (D, H, W)
        self.num_temporal_frames = num_temporal_frames
        self.use_temporal_context = use_temporal_context
        
        # Recompute num_patches for 3D
        self.num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        
        # Temporal context encoder (if using previous frames)
        if use_temporal_context:
            self.temporal_encoder = TemporalContextEncoder(
                in_channels=in_channels,
                hidden_size=hidden_size,
                num_frames=num_context_frames,
            )
            self.temporal_cross_attn = TemporalCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
        else:
            self.temporal_encoder = None
            self.temporal_cross_attn = None
        
        # Override final layer for 3D output
        self.final_layer = FinalLayer4D(
            hidden_size,
            patch_size,
            self.out_channels,
        )
    
    def unpatchify_3d(
        self,
        x: torch.Tensor,  # [B, num_patches, patch_size * out_channels]
    ) -> torch.Tensor:
        """
        Convert patches back to 3D volume (single frame).
        
        Returns:
            3D volume [B, out_channels, D, H, W]
        """
        B = x.shape[0]
        c = self.out_channels
        # input_size is [D, H, W], patch_size is [D, H, W]
        d, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pd, ph, pw = self.patch_size
        
        # Reshape: [B, num_patches, patch_size * out_channels] -> [B, d, h, w, pd, ph, pw, c]
        x = x.reshape(B, d, h, w, pd, ph, pw, c)
        # Permute to [B, c, d, pd, h, ph, w, pw]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # Reshape to [B, c, D, H, W] where D=d*pd, H=h*ph, W=w*pw
        x = x.reshape(B, c, d * pd, h * ph, w * pw)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,  # [B, num_patches, hidden_size] - graph tokens
        t: torch.Tensor,  # [B] - temporal frame indices (0 to T-1), NOT noise levels!
        y: Optional[torch.Tensor] = None,  # [B, hidden_size] - optional conditioning (T1)
        temporal_context: Optional[torch.Tensor] = None,  # [B, C, num_prev, D, H, W] - previous frames
    ) -> torch.Tensor:
        """
        Forward pass with temporal frame generation.
        
        Args:
            x: Graph tokens [B, num_patches, hidden_size]
            t: Temporal frame indices [B] (0 to num_temporal_frames-1)
            y: Optional conditioning [B, hidden_size] (e.g., T1)
            temporal_context: Previous frames [B, C, num_prev, D, H, W] (optional)
        
        Returns:
            Single frame [B, out_channels, D, H, W]
        """
        # Project input to hidden_size if needed
        x = self.input_proj(x)  # [B, num_patches, hidden_size]
        
        # Handle mismatch between fusion patches and DiT expected patches
        num_fusion_patches = x.size(1)
        if num_fusion_patches != self.num_patches:
            if num_fusion_patches < self.num_patches:
                repeat_factor = (self.num_patches + num_fusion_patches - 1) // num_fusion_patches
                x = x.repeat(1, repeat_factor, 1)
                x = x[:, :self.num_patches, :]
            else:
                x = x[:, :self.num_patches, :]
        
        # Timestep embedding - NOW REPRESENTS TEMPORAL POSITION, NOT NOISE LEVEL
        # Normalize temporal index to [0, 1] range for better embedding
        t_normalized = t.float() / max(1, self.num_temporal_frames - 1)  # [B]
        t_emb = self.t_embedder(t_normalized.long())  # [B, hidden_size]
        
        # Combine with optional conditioning
        if y is not None:
            c = t_emb + y
        else:
            c = t_emb
        
        # Encode temporal context if available
        if self.use_temporal_context and temporal_context is not None:
            temporal_tokens = self.temporal_encoder(temporal_context)  # [B, hidden_size]
            # Add temporal context to conditioning
            c = c + 0.1 * temporal_tokens  # Small weight to not overwhelm T1 conditioning
            # Cross-attention to temporal context
            x = self.temporal_cross_attn(x, temporal_tokens)
        
        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # [B, num_patches, patch_size * out_channels]
        
        # Unpatchify to 3D (single frame)
        x = self.unpatchify_3d(x)  # [B, out_channels, D, H, W]
        
        return x


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing DiT4DTemporal (temporal frame generation)...")
    
    model = DiT4DTemporal(
        input_size=(128, 128, 128),  # Single frame spatial dimensions
        in_channels=1,
        patch_size=(16, 16, 16),
        hidden_size=1024,
        depth=12,  # Smaller for testing
        num_heads=16,
        num_temporal_frames=16,
        use_temporal_context=True,
    ).to(device)
    
    B = 2
    num_patches = 512  # From fusion
    num_temporal = 16
    
    # Graph tokens
    fused_tokens = torch.randn(B, num_patches, 768).to(device)
    
    # Temporal frame indices (0 to 15)
    temporal_indices = torch.randint(0, num_temporal, (B,), device=device)
    
    # T1 conditioning
    t1_cond = torch.randn(B, 1024).to(device)
    
    # Previous frames (for temporal context)
    prev_frames = torch.randn(B, 1, 1, 128, 128, 128).to(device)  # [B, C, num_prev, D, H, W]
    
    # Forward pass
    output = model(
        fused_tokens,
        t=temporal_indices,
        y=t1_cond,
        temporal_context=prev_frames,
    )
    
    print(f"Input tokens: {fused_tokens.shape}")
    print(f"Temporal indices: {temporal_indices}")
    print(f"Output frame: {output.shape}")  # Should be [B, out_channels, D, H, W]
    
    # Test sequential generation
    print("\nTesting sequential generation...")
    all_frames = []
    for t in range(num_temporal):
        t_tensor = torch.full((B,), t, device=device)
        frame_t = model(fused_tokens, t=t_tensor, y=t1_cond, temporal_context=prev_frames)
        all_frames.append(frame_t)
        # Update previous frames for next iteration
        prev_frames = frame_t.unsqueeze(2)  # [B, C, 1, D, H, W]
    
    all_frames = torch.stack(all_frames, dim=2)  # [B, C, T, D, H, W]
    print(f"Full sequence: {all_frames.shape}")
    
    print("\n✓ DiT4DTemporal test passed!")

