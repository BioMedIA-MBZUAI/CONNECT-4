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


class ChannelLayerNorm(nn.Module):
    """Layer-normalise channels independently at every (T,D,H,W) location.

    Standard ``GroupNorm`` also reduces over the spatial axes.  That would make
    an exact finite-halo depth-slab execution impossible because every output
    voxel would depend on the complete D axis. Figure 1 does not prescribe a
    normaliser. This explicit implementation choice uses local channel norm so
    the network retains a finite receptive field and admits exact depth slabs.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {5, 6} or x.shape[1] != self.channels:
            raise ValueError(
                f"channel norm expects channel-first 5D/6D input with "
                f"C={self.channels}, got {tuple(x.shape)}"
            )
        permutation = (0, *range(2, x.ndim), 1)
        inverse = (0, x.ndim - 1, *range(1, x.ndim - 1))
        normalised = torch.nn.functional.layer_norm(
            x.permute(permutation),
            (self.channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        # ``layer_norm`` returns a contiguous channel-last tensor.  The
        # permutation below is therefore a value-identical channel-first view.
        # Do not immediately copy it back to contiguous channel-first storage:
        # Conv4DBlock's pointwise FiLM/SiLU operations accept the view and the
        # block already materialises its one required channel-first result at
        # its output boundary.  Avoiding this redundant transient copy is
        # important for exact depth-slab execution (several GiB at T=128) and
        # changes neither values nor the resulting gradients.
        return normalised.permute(inverse)


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
        
        # Local channel normalisation keeps the D receptive field finite, which
        # is required for mathematically exact halo execution at 128^3.
        del groups
        self.norm = ChannelLayerNorm(out_channels)
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
        # ``x_reshaped`` already owns contiguous frame-major storage. Return a
        # channel-first view of those exact bytes: every following spatial
        # block/pool permutes back to frame-major, while temporal attention uses
        # indexed gathers. Materialising this view used to request a complete
        # FP32 [B,C,T,D,H,W] copy (3.44--4.69 GiB at the real slab depths).
        x = x_reshaped.permute(0, 2, 1, 3, 4, 5)  # [B, out_channels, T, D, H, W]
        
        return x


class TemporalAttention(nn.Module):
    """Multi-head temporal-channel attention.

    Time is the attended sequence axis and channels are projected into each
    head's query/key/value feature space, so every voxel can mix information
    from all frames and channels.
    """
    
    def __init__(self, dim, heads=4, dim_head=32, chunk_size: int = 256):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
        # Voxel-chunk size: attention is O(n_voxels * T^2), so process independent
        # voxels in bounded groups while every voxel still attends over the full
        # temporal sequence.  A 256-voxel group also bounds A100 recomputation
        # workspace during the full 128-frame backward pass.
        self.chunk_size = chunk_size

    def _attn_chunk(self, xr: torch.Tensor) -> torch.Tensor:
        # xr: [c, C, T, 1, 1] -> [c, C, T]
        qkv = self.to_qkv(xr).squeeze(-1).squeeze(-1).permute(0, 2, 1)   # [c, T, hidden*3]
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=self.heads), [q, k, v])
        # PyTorch selects Flash/memory-efficient SDPA on supported A100 dtypes,
        # avoiding the explicit [voxels, heads, T, T] score tensor. Its CPU
        # fallback implements the same scaled dot-product attention equation.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
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
        voxels_per_batch = D * H * W
        n = B * voxels_per_batch
        output = None
        for i in range(0, n, self.chunk_size):
            # Gather only this global [B,D,H,W] voxel interval.  The previous
            # whole-grid permute+reshape materialised a full [n,C,T] copy even
            # though attention is independent between voxels.  Explicit global
            # indices preserve the exact legacy row order, including chunks
            # crossing a batch boundary, while bounding this reorder to
            # ``chunk_size`` voxels and retaining every one of the T frames.
            flat_indices = torch.arange(
                i,
                min(i + self.chunk_size, n),
                device=x.device,
                dtype=torch.long,
            )
            batch_indices = torch.div(
                flat_indices, voxels_per_batch, rounding_mode="floor"
            )
            voxel_indices = torch.remainder(flat_indices, voxels_per_batch)
            depth_indices = torch.div(
                voxel_indices, H * W, rounding_mode="floor"
            )
            plane_indices = torch.remainder(voxel_indices, H * W)
            height_indices = torch.div(plane_indices, W, rounding_mode="floor")
            width_indices = torch.remainder(plane_indices, W)
            xc = x[
                batch_indices,
                :,
                :,
                depth_indices,
                height_indices,
                width_indices,
            ].unsqueeze(-1).unsqueeze(-1)                                # [c, C, T, 1, 1]

            if self.training and xc.requires_grad:
                # ``to_qkv`` is this chunk's first value-changing operation and
                # CUDA autocast already converts its FP32 input to BF16. Move
                # that identical cast before the checkpoint boundary. Store the
                # nested checkpoint input on pinned CPU so an enclosing slab
                # replay cannot accumulate a second full-grid CUDA copy.
                checkpoint_input = xc
                if xc.is_cuda and torch.is_autocast_enabled("cuda"):
                    checkpoint_dtype = torch.get_autocast_dtype("cuda")
                    if (
                        xc.dtype == torch.float32
                        and checkpoint_dtype in {torch.float16, torch.bfloat16}
                    ):
                        checkpoint_input = xc.to(dtype=checkpoint_dtype)
                if checkpoint_input.is_cuda:
                    with torch.autograd.graph.save_on_cpu(
                        pin_memory=True,
                        device_type="cuda",
                    ):
                        oc = checkpoint.checkpoint(
                            self._attn_chunk,
                            checkpoint_input,
                            use_reentrant=False,
                        )
                else:
                    oc = checkpoint.checkpoint(
                        self._attn_chunk,
                        checkpoint_input,
                        use_reentrant=False,
                    )
            else:
                oc = self._attn_chunk(xc)
            if output is None:
                # One canonical allocation replaces both the full list+cat and
                # the final full-grid contiguous copy.  IndexPutBackward routes
                # gradients to each disjoint chunk; no value is detached or
                # recomputed, and every output location is written exactly once.
                output = oc.new_empty((B, C, T, D, H, W))
            output[
                batch_indices,
                :,
                :,
                depth_indices,
                height_indices,
                width_indices,
            ] = oc
        if output is None:
            raise RuntimeError("temporal attention received an empty voxel grid")
        return output

    def forward_pre_norm_residual_head(
        self,
        x: torch.Tensor,
        normalizer: nn.Module,
        output_head: nn.Conv3d,
    ) -> torch.Tensor:
        """Stream the final pointwise chain without a full C-channel transient.

        Channel normalization, temporal attention, the residual addition, and
        the 1x1x1 output head are all pointwise in D,H,W. Evaluating independent
        voxels in bounded groups therefore preserves every operator and all T
        frames while materializing only the final output-channel volume.
        """
        if x.ndim != 6:
            raise ValueError("fused final chain expects [B,C,T,D,H,W]")
        B, C, T, D, H, W = x.shape
        if not isinstance(normalizer, ChannelLayerNorm) or normalizer.channels != C:
            raise RuntimeError("fused final chain normalization contract changed")
        if (
            not isinstance(output_head, nn.Conv3d)
            or output_head.in_channels != C
            or output_head.kernel_size != (1, 1, 1)
            or output_head.stride != (1, 1, 1)
            or output_head.padding != (0, 0, 0)
            or output_head.dilation != (1, 1, 1)
            or output_head.groups != 1
        ):
            raise RuntimeError("fused final chain requires the exact 1x1x1 head")

        voxels_per_batch = D * H * W
        output = None
        for i in range(0, B * voxels_per_batch, self.chunk_size):
            flat_indices = torch.arange(
                i,
                min(i + self.chunk_size, B * voxels_per_batch),
                device=x.device,
                dtype=torch.long,
            )
            batch_indices = torch.div(
                flat_indices, voxels_per_batch, rounding_mode="floor"
            )
            voxel_indices = torch.remainder(flat_indices, voxels_per_batch)
            depth_indices = torch.div(
                voxel_indices, H * W, rounding_mode="floor"
            )
            plane_indices = torch.remainder(voxel_indices, H * W)
            height_indices = torch.div(
                plane_indices, W, rounding_mode="floor"
            )
            width_indices = torch.remainder(plane_indices, W)
            raw_chunk = x[
                batch_indices,
                :,
                :,
                depth_indices,
                height_indices,
                width_indices,
            ].unsqueeze(-1).unsqueeze(-1)

            def run_final_chunk(value: torch.Tensor) -> torch.Tensor:
                normalised = normalizer(value)
                if (
                    normalised.is_cuda
                    and torch.is_autocast_enabled("cuda")
                    and normalised.dtype == torch.float32
                ):
                    autocast_dtype = torch.get_autocast_dtype("cuda")
                    if autocast_dtype in {torch.float16, torch.bfloat16}:
                        # to_qkv performs this same autocast at its boundary.
                        normalised = normalised.to(dtype=autocast_dtype)
                attended = self._attn_chunk(normalised)
                residual = attended + value.squeeze(-1).squeeze(-1)
                return output_head(
                    residual.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)

            if self.training and torch.is_grad_enabled():
                if raw_chunk.is_cuda:
                    with torch.autograd.graph.save_on_cpu(
                        pin_memory=True,
                        device_type="cuda",
                    ):
                        head_chunk = checkpoint.checkpoint(
                            run_final_chunk,
                            raw_chunk,
                            use_reentrant=False,
                        )
                else:
                    head_chunk = checkpoint.checkpoint(
                        run_final_chunk,
                        raw_chunk,
                        use_reentrant=False,
                    )
            else:
                head_chunk = run_final_chunk(raw_chunk)

            if output is None:
                output = head_chunk.new_empty(
                    B, output_head.out_channels, T, D, H, W
                )
            output[
                batch_indices,
                :,
                :,
                depth_indices,
                height_indices,
                width_indices,
            ] = head_chunk
        if output is None:
            raise RuntimeError("fused final chain received an empty voxel grid")
        return output

    def forward_pre_norm_residual_head_from_pinned_cpu(
        self,
        x: torch.Tensor,
        normalizer: nn.Module,
        output_head: nn.Conv3d,
        *,
        execution_device: torch.device,
    ) -> torch.Tensor:
        """Run the unchanged final pointwise chain from pinned host backing.

        The host tensor is laid out [B,D,H,W,C,T], so every contiguous slice is
        the same global voxel row interval used by the CUDA path and contains
        all T frames. Only bounded rows are resident on CUDA at once.
        """
        if x.ndim != 6:
            raise ValueError("host-backed fused final chain expects [B,D,H,W,C,T]")
        B, D, H, W, C, T = x.shape
        execution_device = torch.device(execution_device)
        if (
            x.device.type != "cpu"
            or not x.is_pinned()
            or not x.is_contiguous()
            or execution_device.type != "cuda"
        ):
            raise RuntimeError(
                "host-backed fused final chain requires contiguous pinned CPU "
                "storage and CUDA execution"
            )
        if not isinstance(normalizer, ChannelLayerNorm) or normalizer.channels != C:
            raise RuntimeError("host-backed final normalization contract changed")
        if (
            not isinstance(output_head, nn.Conv3d)
            or output_head.in_channels != C
            or output_head.kernel_size != (1, 1, 1)
            or output_head.stride != (1, 1, 1)
            or output_head.padding != (0, 0, 0)
            or output_head.dilation != (1, 1, 1)
            or output_head.groups != 1
            or any(
                parameter.device != execution_device
                for module in (self, normalizer, output_head)
                for parameter in module.parameters()
            )
        ):
            raise RuntimeError("host-backed fused final chain device/operator changed")

        def run_final_chunk(value: torch.Tensor) -> torch.Tensor:
            normalised = normalizer(value)
            if (
                normalised.is_cuda
                and torch.is_autocast_enabled("cuda")
                and normalised.dtype == torch.float32
            ):
                autocast_dtype = torch.get_autocast_dtype("cuda")
                if autocast_dtype in {torch.float16, torch.bfloat16}:
                    normalised = normalised.to(dtype=autocast_dtype)
            attended = self._attn_chunk(normalised)
            residual = attended + value.squeeze(-1).squeeze(-1)
            return output_head(
                residual.unsqueeze(-1).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)

        voxel_count = B * D * H * W
        flat_host = x.reshape(voxel_count, C, T)
        # One split node owns all disjoint row views. Its backward reassembles a
        # single host gradient instead of creating one full CUDA scatter-gradient
        # base for every voxel group.
        host_chunks = flat_host.split(self.chunk_size, dim=0)
        flat_output = None
        row_start = 0
        for host_chunk in host_chunks:
            row_end = row_start + host_chunk.shape[0]
            raw_chunk = host_chunk.to(
                device=execution_device,
                non_blocking=False,
                memory_format=torch.preserve_format,
            ).unsqueeze(-1).unsqueeze(-1)
            if (
                raw_chunk.device != execution_device
                or raw_chunk.shape != (row_end - row_start, C, T, 1, 1)
                or raw_chunk.dtype != x.dtype
                or raw_chunk.requires_grad != x.requires_grad
            ):
                raise RuntimeError("host-backed final chunk restore changed identity")
            if self.training and torch.is_grad_enabled():
                with torch.autograd.graph.save_on_cpu(
                    pin_memory=True,
                    device_type="cuda",
                ):
                    head_chunk = checkpoint.checkpoint(
                        run_final_chunk,
                        raw_chunk,
                        use_reentrant=False,
                    )
            else:
                head_chunk = run_final_chunk(raw_chunk)
            if flat_output is None:
                flat_output = head_chunk.new_empty(
                    voxel_count, output_head.out_channels, T
                )
            flat_output[row_start:row_end].copy_(head_chunk)
            del raw_chunk, head_chunk
            row_start = row_end

        if flat_output is None or row_start != voxel_count:
            raise RuntimeError("host-backed fused final chain received no voxels")
        return flat_output.reshape(
            B, D, H, W, output_head.out_channels, T
        ).permute(0, 4, 5, 1, 2, 3).contiguous()

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, output_head=None, **kwargs):
        if output_head is not None:
            if (
                args
                or kwargs
                or not isinstance(self.fn, PreNorm)
                or not isinstance(self.fn.fn, TemporalAttention)
            ):
                raise RuntimeError("fused output head requires PreNorm attention")
            return self.fn.fn.forward_pre_norm_residual_head(
                x,
                self.fn.norm,
                output_head,
            )
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChannelLayerNorm(dim)
        self.offload_norm_saved_tensors = False

    def forward(self, x, **kwargs):
        if (
            self.offload_norm_saved_tensors
            and self.training
            and x.is_cuda
            and x.requires_grad
            and torch.is_grad_enabled()
        ):
            # The final full-resolution normalisation retains one mean/rstd
            # field for backward. Move only that operator's byte-exact saved
            # state to host memory during one-slab replay; the normalised value,
            # residual input, attention equation, and gradients are unchanged.
            with torch.autograd.graph.save_on_cpu(
                pin_memory=True,
                device_type="cuda",
            ):
                normalised = self.norm(x)
        else:
            normalised = self.norm(x)
        return self.fn(normalised, **kwargs)


class DiTToTCUNetProjection(nn.Module):
    """Pointwise full-grid projection from scalar DiT x0 to TC-UNet features.

    Figure 1 labels the full-grid 64-channel handoff but not its projection
    operator or latent channelization. This explicit implementation choice
    requires a scalar DiT volume already on D,H,W and uses a per-frame 1x1x1
    projection. It cannot resize, spatially smooth, or copy T1 detail.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_size = tuple(int(value) for value in spatial_size)
        if self.in_channels < 1 or self.out_channels < 1:
            raise ValueError("DiT-to-TC-UNet projection channels must be positive")
        if len(self.spatial_size) != 3 or any(value < 1 for value in self.spatial_size):
            raise ValueError("projection spatial_size must be a positive D,H,W triplet")
        self.projection = nn.Conv3d(self.in_channels, self.out_channels, 1)

    def _project(self, x: torch.Tensor, *, require_full_depth: bool) -> torch.Tensor:
        if x.ndim != 6 or x.shape[1] != self.in_channels:
            raise ValueError(
                "DiT-to-TC-UNet projection expects [B,C,T,D,H,W] with "
                f"C={self.in_channels}, got {tuple(x.shape)}"
            )
        batch, channels, time, depth, height, width = x.shape
        expected_depth, expected_height, expected_width = self.spatial_size
        if (height, width) != (expected_height, expected_width):
            raise ValueError(
                "DiT output must already match the configured full H,W grid; "
                f"expected {(expected_height, expected_width)}, got {(height, width)}"
            )
        if require_full_depth and depth != expected_depth:
            raise ValueError(
                "DiT output must already match the configured full D grid; "
                f"expected {expected_depth}, got {depth}"
            )
        if not 1 <= depth <= expected_depth:
            raise ValueError("projected depth slab lies outside the configured D grid")
        frames = x.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * time, channels, depth, height, width
        )
        features = self.projection(frames)
        return features.reshape(
            batch, time, self.out_channels, depth, height, width
        ).permute(0, 2, 1, 3, 4, 5).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(x, require_full_depth=True)

    def project_depth_slab(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(x, require_full_depth=False)


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
        input_spatial_size: Optional[Tuple[int, int, int]] = None,
        final_decoder_frame_chunk_size: int = 1,
    ):
        super().__init__()
        if int(out_channels) != 1:
            raise ValueError("CONNECT-4 temporal UNet requires out_channels=1")
        if int(in_channels) != int(base_channels):
            raise ValueError(
                "Figure-1 TC-UNet input channels must equal base_channels "
                f"({in_channels} != {base_channels})"
            )
        if isinstance(num_levels, bool) or int(num_levels) < 2:
            raise ValueError("TC-UNet num_levels must be at least two")
        if use_temporal_attn is not True:
            raise ValueError(
                "paper-faithful temporal UNet requires temporal attention enabled"
            )
        self.num_levels = int(num_levels)
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_reentrant = bool(checkpoint_reentrant)
        self.in_channels = int(in_channels)
        if (
            isinstance(final_decoder_frame_chunk_size, bool)
            or int(final_decoder_frame_chunk_size) < 1
        ):
            raise ValueError("final_decoder_frame_chunk_size must be a positive integer")
        # The final full-resolution spatial chain is frame-local.  A one-frame
        # default bounds its checkpoint-replay LayerNorm scratch allocation on
        # 40-GiB A100s without changing operator equations, parameters, or the
        # complete temporal extent reassembled before attention.
        self.final_decoder_frame_chunk_size = int(final_decoder_frame_chunk_size)
        # Conv4DBlock never mixes temporal rows: its Conv3d, channel norm, FiLM,
        # and SiLU tail are all frame-local.  Large whole-T checkpoint replays
        # therefore use bounded frame groups and one differentiable assembled
        # backing allocation.  This preserves every operator and all 128 frames
        # while avoiding a second multi-GiB Conv3d/LayerNorm result.
        self.block_frame_chunk_size = 8
        self.block_stream_threshold_bytes = 1 << 29
        self.input_spatial_size = (
            tuple(int(value) for value in input_spatial_size)
            if input_spatial_size is not None
            else None
        )
        if self.input_spatial_size is not None:
            if len(self.input_spatial_size) != 3 or any(
                value < 1 for value in self.input_spatial_size
            ):
                raise ValueError("input_spatial_size must be a positive D,H,W triplet")
            downscale = 2 ** (self.num_levels - 1)
            if any(value % downscale for value in self.input_spatial_size[1:]):
                raise ValueError(
                    "TC-UNet H and W must be divisible by 2^(num_levels-1); "
                    "interpolation-based skip repair is forbidden"
                )
        
        # Temporal conditioning MLP (from TC-UNet)
        time_dim = temporal_cond_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(temporal_cond_dim),
            nn.Linear(temporal_cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.temporal_cond_dim = time_dim
        
        # The separate DiTToTCUNetProjection has already produced the Figure-1
        # [B,T,64,D,H,W] tensor.  No low-grid reconstruction occurs here.
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
                # Figure 1 preserves D and downsamples H,W only.
                self.encoder_pool.append(
                    nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
                )
            in_ch = out_ch
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for i in range(num_levels - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            # Symmetric Figure-1 path: preserve D and double H,W only.
            self.decoder_upsample.append(
                nn.ConvTranspose3d(
                    in_ch,
                    in_ch,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                )
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

        final_attention = self.decoder[-1]
        if not isinstance(final_attention, Residual) or not isinstance(
            final_attention.fn, PreNorm
        ):
            raise RuntimeError("final decoder temporal attention contract changed")
        final_attention.fn.offload_norm_saved_tensors = True
        
        # Figure 1 labels a per-timepoint 1x1x1 final head at unchanged D,H,W.
        self.output_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels
        
        # Mask conditioning (applied as additional input)
        self.mask_proj = nn.Conv3d(mask_cond_dim, base_channels, 1)
        self.depth_receptive_field_radius = self._derive_depth_receptive_radius()
    
    @staticmethod
    def _apply_per_frame(
        module: nn.Module,
        x: torch.Tensor,
        *,
        keep_frame_major_storage: bool = False,
    ) -> torch.Tensor:
        """Apply a 3D (D,H,W) module independently to every temporal frame."""
        B, C, T, D, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W)
        y = module(y)
        Cn = y.shape[1]
        Dn, Hn, Wn = y.shape[-3:]
        channel_first = y.reshape(B, T, Cn, Dn, Hn, Wn).permute(
            0, 2, 1, 3, 4, 5
        )
        if (
            keep_frame_major_storage
            and x.is_cuda
            and torch.is_autocast_enabled("cuda")
        ):
            # The decoder immediately permutes this upsample result back to
            # [B,T,C,D,H,W] for skip concatenation.  Preserve that already-
            # contiguous backing storage instead of allocating a redundant
            # channel-first copy that can exceed 2.5 GiB at T=128.  Tensor
            # values, axes, and the transpose-convolution gradient are unchanged.
            return channel_first
        return channel_first.contiguous()

    @staticmethod
    def _conv_depth_radius(module: nn.Conv3d) -> int:
        kernel = module.kernel_size[0]
        stride = module.stride[0]
        dilation = module.dilation[0]
        if stride != 1 or kernel % 2 != 1:
            raise RuntimeError(
                "finite symmetric D-halo derivation requires odd Conv3d kernels "
                "with D stride one"
            )
        expected_padding = dilation * (kernel - 1) // 2
        if module.padding[0] != expected_padding:
            raise RuntimeError(
                "finite symmetric D-halo derivation requires same Conv3d D padding"
            )
        return expected_padding

    def _derive_depth_receptive_radius(self) -> int:
        """Derive the exact longest-path D radius from instantiated operators."""

        radius = 0
        skip_radii = []
        encoder_index = 0
        for level in range(self.num_levels):
            for _ in range(2):
                block = self.encoder[encoder_index]
                if not isinstance(block, Conv4DBlock):
                    raise RuntimeError("TC-UNet encoder convolution contract changed")
                radius += self._conv_depth_radius(block.conv)
                encoder_index += 1
            encoder_index += 1  # temporal attention is spatially pointwise
            skip_radii.append(radius)
            if level < self.num_levels - 1:
                pool = self.encoder_pool[level]
                kernel = pool.kernel_size
                stride = pool.stride
                kernel_d = kernel if isinstance(kernel, int) else kernel[0]
                stride_d = stride if isinstance(stride, int) else stride[0]
                if kernel_d != 1 or stride_d != 1:
                    raise RuntimeError("Figure-1 TC-UNet must not pool D")

        decoder_index = 0
        for stage, upsample in enumerate(self.decoder_upsample):
            if (
                upsample.kernel_size[0] != 1
                or upsample.stride[0] != 1
                or upsample.padding[0] != 0
                or upsample.output_padding[0] != 0
            ):
                raise RuntimeError("Figure-1 TC-UNet must not upsample D")
            radius = max(radius, skip_radii[-(stage + 2)])
            for _ in range(2):
                block = self.decoder[decoder_index]
                if not isinstance(block, Conv4DBlock):
                    raise RuntimeError("TC-UNet decoder convolution contract changed")
                radius += self._conv_depth_radius(block.conv)
                decoder_index += 1
            decoder_index += 1  # temporal attention is spatially pointwise

        if (
            self.output_head.kernel_size != (1, 1, 1)
            or self.output_head.stride != (1, 1, 1)
        ):
            raise RuntimeError("Figure-1 final head must be a stride-one 1x1x1 Conv3d")
        return radius

    def _run_block(self, block: nn.Module, x: torch.Tensor, temporal_cond: torch.Tensor, graph_cond: torch.Tensor) -> torch.Tensor:
        """Optionally checkpoint a block to save activation memory."""
        if self.use_checkpoint and self.training and x.requires_grad:
            # Conv4DBlock's first value-changing operator is an autocast Conv3d.
            # Store exactly the dtype/value that Conv3d already consumes at the
            # checkpoint boundary.  This commutes only across its preceding
            # permute/contiguous view operations and preserves the cast-backward
            # path, while avoiding retained FP32 checkpoint inputs.
            checkpoint_dtype = None
            if x.is_cuda and torch.is_autocast_enabled("cuda"):
                checkpoint_dtype = torch.get_autocast_dtype("cuda")
            estimated_output_bytes = 0
            if isinstance(block, Conv4DBlock) and checkpoint_dtype is not None:
                estimated_output_bytes = (
                    x.shape[0]
                    * x.shape[2]
                    * block.conv.out_channels
                    * math.prod(x.shape[-3:])
                    * torch.empty((), dtype=checkpoint_dtype).element_size()
                )
            if (
                isinstance(block, Conv4DBlock)
                and checkpoint_dtype is not None
                and x.shape[2] > self.block_frame_chunk_size
                and estimated_output_bytes >= self.block_stream_threshold_bytes
            ):
                assembled = None
                time = x.shape[2]
                for frame_start in range(0, time, self.block_frame_chunk_size):
                    frame_end = min(
                        time, frame_start + self.block_frame_chunk_size
                    )
                    checkpoint_input = x[:, :, frame_start:frame_end]
                    if (
                        checkpoint_input.dtype == torch.float32
                        and checkpoint_dtype in {torch.float16, torch.bfloat16}
                    ):
                        checkpoint_input = checkpoint_input.to(
                            dtype=checkpoint_dtype
                        )
                    temporal_chunk = (
                        temporal_cond[:, frame_start:frame_end]
                        if temporal_cond.ndim == 3
                        else temporal_cond
                    )
                    with torch.autograd.graph.save_on_cpu(
                        pin_memory=True,
                        device_type="cuda",
                    ):
                        decoded_chunk = checkpoint.checkpoint(
                            lambda inp, t_cond, g_cond: block(
                                inp, t_cond, g_cond
                            ),
                            checkpoint_input,
                            temporal_chunk,
                            graph_cond,
                            use_reentrant=self.checkpoint_reentrant,
                        )
                    frame_major = decoded_chunk.permute(0, 2, 3, 4, 5, 1)
                    if not frame_major.is_contiguous():
                        raise RuntimeError(
                            "streamed Conv4DBlock output lost frame-major backing"
                        )
                    if assembled is None:
                        assembled = frame_major.new_empty(
                            frame_major.shape[0],
                            time,
                            *frame_major.shape[2:],
                        )
                    assembled[:, frame_start:frame_end].copy_(frame_major)
                    del decoded_chunk, frame_major
                if assembled is None:
                    raise RuntimeError("streamed Conv4DBlock received an empty T axis")
                return assembled.permute(0, 5, 1, 2, 3, 4)

            checkpoint_input = x
            if checkpoint_dtype is not None:
                if (
                    x.dtype == torch.float32
                    and checkpoint_dtype in {torch.float16, torch.bfloat16}
                ):
                    checkpoint_input = x.to(dtype=checkpoint_dtype)
            if checkpoint_input.is_cuda:
                # A complete slab is itself replayed during backward. Keep the
                # fine checkpoint input's exact bytes in pinned CPU memory so
                # nested block boundaries do not reconstruct another full set
                # of CUDA activations before the final attention residual.
                with torch.autograd.graph.save_on_cpu(
                    pin_memory=True,
                    device_type="cuda",
                ):
                    return checkpoint.checkpoint(
                        lambda inp, t_cond, g_cond: block(inp, t_cond, g_cond),
                        checkpoint_input,
                        temporal_cond,
                        graph_cond,
                        use_reentrant=self.checkpoint_reentrant,
                    )
            return checkpoint.checkpoint(
                lambda inp, t_cond, g_cond: block(inp, t_cond, g_cond),
                checkpoint_input,
                temporal_cond,
                graph_cond,
                use_reentrant=self.checkpoint_reentrant,
            )
        return block(x, temporal_cond, graph_cond)

    def _run_final_decoder_spatial_chunks(
        self,
        upsample: nn.ConvTranspose3d,
        first_block: Conv4DBlock,
        second_block: Conv4DBlock,
        x: torch.Tensor,
        skip: torch.Tensor,
        temporal_cond: torch.Tensor,
        graph_cond: torch.Tensor,
        *,
        execution_device: torch.device,
        host_resident_replay: bool,
    ) -> Tuple[torch.Tensor, bool]:
        """Checkpoint the final per-frame spatial chain in bounded T groups.

        These four spatial operations do not mix T. The reduced-channel result
        is reassembled over complete T before unchanged global attention.
        """
        if x.ndim != 6 or skip.ndim != 6:
            raise ValueError("decoder tensors must be [B,C,T,D,H,W]")
        if x.shape[0] != skip.shape[0] or x.shape[2] != skip.shape[2]:
            raise RuntimeError("decoder and skip batch/time axes differ")
        if temporal_cond.ndim != 3 or temporal_cond.shape[:2] != (
            x.shape[0], x.shape[2]
        ):
            raise RuntimeError("temporal conditioning must be [B,T,C]")
        execution_device = torch.device(execution_device)
        if temporal_cond.device != execution_device or graph_cond.device != execution_device:
            raise RuntimeError(
                "final decoder conditioning and execution devices must match"
            )
        if not isinstance(host_resident_replay, bool):
            raise RuntimeError("final decoder residency flag must be boolean")
        if x.device != skip.device:
            raise RuntimeError("decoder and skip residency must match")
        if host_resident_replay:
            if (
                x.device.type != "cpu"
                or not x.is_pinned()
                or not skip.is_pinned()
                or execution_device.type != "cuda"
            ):
                raise RuntimeError(
                    "host-resident replay requires pinned CPU inputs and CUDA execution"
                )
        elif x.device != execution_device:
            raise RuntimeError("decoder inputs must use the execution device")
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            # The streamed path bypasses the legacy pre-upsample cache release.
            # Reclaim inactive blocks before its one all-T reduced-channel
            # backing allocation; live tensors and autograd edges are untouched.
            torch.cuda.empty_cache()

        assembled = None
        time = x.shape[2]
        frame_chunk = min(self.final_decoder_frame_chunk_size, time)
        for frame_start in range(0, time, frame_chunk):
            frame_end = min(time, frame_start + frame_chunk)
            x_chunk = x[:, :, frame_start:frame_end]
            skip_chunk = skip[:, :, frame_start:frame_end]
            temporal_chunk = temporal_cond[:, frame_start:frame_end]
            if x_chunk.device != execution_device:
                # During the grad-enabled final-stage replay, the complete x and
                # skip tensors live in pinned CPU memory. Restore only this
                # frame-local slice. The synchronous differentiable copies retain
                # the exact dtype, value, and upstream gradient edges. PyTorch may
                # preserve either dense channel-first or dense frame-major storage
                # here, depending on the incoming decoder stage; both layouts are
                # accepted by the unchanged per-frame operators below.
                x_shape, x_dtype, x_requires_grad = (
                    x_chunk.shape,
                    x_chunk.dtype,
                    x_chunk.requires_grad,
                )
                skip_shape, skip_dtype, skip_requires_grad = (
                    skip_chunk.shape,
                    skip_chunk.dtype,
                    skip_chunk.requires_grad,
                )
                x_chunk = x_chunk.to(
                    device=execution_device,
                    non_blocking=False,
                    memory_format=torch.preserve_format,
                )
                skip_chunk = skip_chunk.to(
                    device=execution_device,
                    non_blocking=False,
                    memory_format=torch.preserve_format,
                )
                if (
                    x_chunk.device != execution_device
                    or skip_chunk.device != execution_device
                    or x_chunk.shape != x_shape
                    or skip_chunk.shape != skip_shape
                    or x_chunk.dtype != x_dtype
                    or skip_chunk.dtype != skip_dtype
                    or x_chunk.requires_grad != x_requires_grad
                    or skip_chunk.requires_grad != skip_requires_grad
                    or not (
                        x_chunk.is_contiguous()
                        or x_chunk.permute(0, 2, 1, 3, 4, 5).is_contiguous()
                    )
                    or not (
                        skip_chunk.is_contiguous()
                        or skip_chunk.permute(0, 2, 1, 3, 4, 5).is_contiguous()
                    )
                ):
                    raise RuntimeError(
                        "final decoder frame restore changed tensor identity/layout"
                    )

            def run_spatial_chunk(
                x_arg: torch.Tensor,
                skip_arg: torch.Tensor,
                temporal_arg: torch.Tensor,
                graph_arg: torch.Tensor,
            ) -> torch.Tensor:
                upsampled = self._apply_per_frame(
                    upsample, x_arg, keep_frame_major_storage=True
                )
                if upsampled.shape[-3:] != skip_arg.shape[-3:]:
                    raise RuntimeError(
                        "TC-UNet symmetric skip shapes differ; interpolation-based "
                        f"repair is forbidden ({upsampled.shape[-3:]} != "
                        f"{skip_arg.shape[-3:]})"
                    )
                if skip_arg.dtype != upsampled.dtype:
                    skip_arg = skip_arg.to(dtype=upsampled.dtype)
                merged = torch.cat(
                    [
                        upsampled.permute(0, 2, 1, 3, 4, 5),
                        skip_arg.permute(0, 2, 1, 3, 4, 5),
                    ],
                    dim=2,
                ).permute(0, 2, 1, 3, 4, 5)
                if not merged.permute(0, 2, 1, 3, 4, 5).is_contiguous():
                    raise RuntimeError("TC-UNet frame-major skip layout changed")
                decoded = first_block(merged, temporal_arg, graph_arg)
                return second_block(decoded, temporal_arg, graph_arg)

            if self.training and self.use_checkpoint and x_chunk.requires_grad:
                if x_chunk.is_cuda:
                    # Explicitly retain nested checkpoint inputs in pinned CPU
                    # memory during the enclosing slab replay. Without this
                    # inner scope, all frame-chunk views keep their complete
                    # x/skip CUDA base allocations alive until attention.
                    with torch.autograd.graph.save_on_cpu(
                        pin_memory=True,
                        device_type="cuda",
                    ):
                        decoded_chunk = checkpoint.checkpoint(
                            run_spatial_chunk,
                            x_chunk,
                            skip_chunk,
                            temporal_chunk,
                            graph_cond,
                            use_reentrant=self.checkpoint_reentrant,
                        )
                else:
                    decoded_chunk = checkpoint.checkpoint(
                        run_spatial_chunk,
                        x_chunk,
                        skip_chunk,
                        temporal_chunk,
                        graph_cond,
                        use_reentrant=self.checkpoint_reentrant,
                    )
            else:
                decoded_chunk = run_spatial_chunk(
                    x_chunk, skip_chunk, temporal_chunk, graph_cond
                )
            if host_resident_replay:
                # Reorder only this bounded frame group to the exact global
                # voxel-major layout consumed by attention, then copy it into
                # one disjoint slice of a single pinned host allocation. These
                # differentiable CopySlices retain every chunk's autograd edge
                # without the former list + cat + second pinned copy, which
                # transiently held three complete decoder results in host RAM.
                # No complete C-channel decoder result or base gradient is
                # materialized on CUDA during the reentrant backward replay.
                voxel_major_chunk = decoded_chunk.permute(0, 3, 4, 5, 1, 2)
                if assembled is None:
                    assembled = torch.empty(
                        voxel_major_chunk.shape[0],
                        *voxel_major_chunk.shape[1:-1],
                        time,
                        dtype=voxel_major_chunk.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                assembled[..., frame_start:frame_end].copy_(
                    voxel_major_chunk,
                    non_blocking=False,
                )
                if (
                    not assembled.is_pinned()
                    or not assembled.is_contiguous()
                    or assembled.shape[-1] != time
                    or assembled.dtype != voxel_major_chunk.dtype
                    or assembled.requires_grad != voxel_major_chunk.requires_grad
                    or (voxel_major_chunk.requires_grad and assembled.grad_fn is None)
                ):
                    raise RuntimeError(
                        "final decoder host assembly changed tensor/autograd identity"
                    )
                del decoded_chunk, voxel_major_chunk
            else:
                if assembled is None:
                    assembled = decoded_chunk.new_empty(
                        decoded_chunk.shape[0],
                        decoded_chunk.shape[1],
                        time,
                        *decoded_chunk.shape[-3:],
                    )
                # Disjoint differentiable copies preserve one backing allocation.
                assembled[:, :, frame_start:frame_end].copy_(decoded_chunk)
                del decoded_chunk

        if host_resident_replay:
            if assembled is None:
                raise RuntimeError("TC-UNet final decoder received an empty T axis")
            if (
                not assembled.is_pinned()
                or not assembled.is_contiguous()
                or assembled.shape[-1] != time
                or (assembled.requires_grad and assembled.grad_fn is None)
            ):
                raise RuntimeError("final decoder full host assembly changed identity")
            return assembled, True
        if assembled is None:
            raise RuntimeError("TC-UNet final decoder received an empty T axis")
        return assembled, False

    @staticmethod
    def _differentiable_pinned_cpu_copy(value: torch.Tensor) -> torch.Tensor:
        """Move live CUDA bytes to pinned CPU without severing autograd."""
        if not value.is_cuda:
            raise RuntimeError("only CUDA tensors may enter the pinned offload path")
        copied = torch.empty_like(
            value,
            device="cpu",
            pin_memory=True,
            memory_format=torch.preserve_format,
        ).copy_(value, non_blocking=False)
        if (
            copied.device.type != "cpu"
            or not copied.is_pinned()
            or copied.shape != value.shape
            or copied.stride() != value.stride()
            or copied.dtype != value.dtype
            or copied.requires_grad != value.requires_grad
            or (value.requires_grad and copied.grad_fn is None)
        ):
            raise RuntimeError(
                "final decoder pinned offload changed tensor/autograd identity"
            )
        return copied
    
    @staticmethod
    def _normalise_temporal_indices(
        temporal_indices: Optional[torch.Tensor],
        *,
        batch: int,
        time: int,
        device: torch.device,
    ) -> torch.Tensor:
        if temporal_indices is None:
            return torch.arange(time, device=device, dtype=torch.float32).expand(
                batch, -1
            )
        temporal_indices = temporal_indices.to(device=device)
        if temporal_indices.ndim == 1:
            if temporal_indices.shape != (batch,):
                raise ValueError("one-dimensional temporal_indices must have shape [B]")
            temporal_indices = temporal_indices.unsqueeze(1).expand(-1, time)
        if temporal_indices.shape != (batch, time):
            raise ValueError("temporal_indices must have shape [B] or [B,T]")
        return temporal_indices.float()

    def _forward_features(
        self,
        x: torch.Tensor,
        graph_cond: torch.Tensor,
        temporal_indices: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        *,
        require_full_depth: bool,
    ) -> torch.Tensor:
        if x.ndim != 6 or x.shape[1] != self.in_channels:
            raise ValueError(
                "TC-UNet expects Figure-1 features [B,C,T,D,H,W] with "
                f"C={self.in_channels}, got {tuple(x.shape)}"
            )
        batch, _, time, depth, height, width = x.shape
        if self.input_spatial_size is not None:
            expected_depth, expected_height, expected_width = self.input_spatial_size
            if (height, width) != (expected_height, expected_width):
                raise ValueError("TC-UNet H,W must equal the configured full grid")
            if require_full_depth and depth != expected_depth:
                raise ValueError("TC-UNet D must equal the configured full grid")
            if not 1 <= depth <= expected_depth:
                raise ValueError("TC-UNet depth slab is outside the configured grid")
        downscale = 2 ** (self.num_levels - 1)
        if height % downscale or width % downscale:
            raise ValueError(
                "TC-UNet H,W must be divisible by 2^(num_levels-1); "
                "interpolation is forbidden"
            )
        if graph_cond.ndim != 2 or graph_cond.shape[0] != batch:
            raise ValueError("graph_cond must have shape [B,C_graph]")

        temporal_indices = self._normalise_temporal_indices(
            temporal_indices,
            batch=batch,
            time=time,
            device=x.device,
        )
        temporal_cond = self.time_mlp(temporal_indices.reshape(-1)).reshape(
            batch, time, -1
        )

        output_mask = None
        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.ndim == 5:
                if tuple(mask.shape) != (batch, 1, depth, height, width):
                    raise ValueError(
                        "static mask must exactly match [B,1,D,H,W]; online "
                        f"resampling is forbidden, got {tuple(mask.shape)}"
                    )
                x = x + self.mask_proj(mask).unsqueeze(2)
                output_mask = mask.unsqueeze(2)
            elif mask.ndim == 6:
                if tuple(mask.shape) != (
                    batch,
                    1,
                    time,
                    depth,
                    height,
                    width,
                ):
                    raise ValueError(
                        "temporal mask must exactly match [B,1,T,D,H,W]; "
                        f"online resampling is forbidden, got {tuple(mask.shape)}"
                    )
                mask_frames = mask.permute(0, 2, 1, 3, 4, 5).reshape(
                    batch * time, 1, depth, height, width
                )
                mask_features = self.mask_proj(mask_frames).reshape(
                    batch, time, self.in_channels, depth, height, width
                ).permute(0, 2, 1, 3, 4, 5)
                x = x + mask_features
                output_mask = mask
            else:
                raise ValueError("mask must be [B,1,D,H,W] or [B,1,T,D,H,W]")

        x = self.init_temporal_attn(x)
        encoder_outputs = []
        encoder_index = 0
        for level in range(self.num_levels):
            x = self._run_block(
                self.encoder[encoder_index], x, temporal_cond, graph_cond
            )
            encoder_index += 1
            x = self._run_block(
                self.encoder[encoder_index], x, temporal_cond, graph_cond
            )
            encoder_index += 1
            x = self.encoder[encoder_index](x)
            encoder_index += 1
            if (
                level < self.num_levels - 1
                and x.is_cuda
                and torch.is_autocast_enabled("cuda")
                and x.dtype == torch.float32
            ):
                skip_dtype = torch.get_autocast_dtype("cuda")
                if skip_dtype in {torch.float16, torch.bfloat16}:
                    # The decoder already casts this skip to the BF16 upsample
                    # dtype immediately before concatenation.  Store that same
                    # value now while the unchanged FP32 ``x`` continues into
                    # max-pooling.  This shortens FP32 skip residency without
                    # changing either branch's values or gradients.
                    encoder_outputs.append(x.to(dtype=skip_dtype))
                else:
                    encoder_outputs.append(x)
            else:
                encoder_outputs.append(x)
            if level < self.num_levels - 1:
                x = self._apply_per_frame(self.encoder_pool[level], x)

        # The final encoder value is also held by ``x``.  Remove the redundant
        # list reference before decoding, then consume every remaining skip
        # exactly once.  Keeping all of the full-resolution 128-frame skips in
        # the list until the end needlessly extends their CUDA lifetimes.
        bottleneck = encoder_outputs.pop()
        if bottleneck is not x:
            raise RuntimeError("TC-UNet bottleneck identity changed")
        del bottleneck

        decoder_index = 0
        output_head_applied = False
        for decoder_stage, upsample in enumerate(self.decoder_upsample):
            is_final_decoder_stage = (
                decoder_stage == len(self.decoder_upsample) - 1
            )
            if not encoder_outputs:
                raise RuntimeError("TC-UNet decoder has no symmetric skip")
            skip = encoder_outputs.pop()
            if (
                is_final_decoder_stage
                and self.training
                and self.use_checkpoint
            ):
                first_block = self.decoder[decoder_index]
                second_block = self.decoder[decoder_index + 1]
                if not isinstance(first_block, Conv4DBlock) or not isinstance(
                    second_block, Conv4DBlock
                ):
                    raise RuntimeError("TC-UNet decoder convolution contract changed")
                execution_device = x.device
                host_resident_final_decoder = False
                if (
                    torch.is_grad_enabled()
                    and x.is_cuda
                    and skip.is_cuda
                ):
                    # In an enclosing slab checkpoint's backward replay these two
                    # complete-T inputs otherwise remain live while the final
                    # decoder forms its FP32 result. At production D=26 they
                    # occupy 2.03125 GiB each. Rebind both caller references to
                    # differentiable pinned-CPU copies; individual frames are
                    # restored for the unchanged spatial blocks, and the complete
                    # result remains host-backed through bounded full-T attention.
                    # This changes residency only, never values, dtypes, operators,
                    # temporal extent, or gradients.
                    x = self._differentiable_pinned_cpu_copy(x)
                    skip = self._differentiable_pinned_cpu_copy(skip)
                    host_resident_final_decoder = True
                    if x.is_cuda or skip.is_cuda:
                        raise RuntimeError("final decoder live-input offload failed")
                    with torch.cuda.device(execution_device):
                        torch.cuda.empty_cache()
                x, host_resident_final_decoder = (
                    self._run_final_decoder_spatial_chunks(
                        upsample,
                        first_block,
                        second_block,
                        x,
                        skip,
                        temporal_cond,
                        graph_cond,
                        execution_device=execution_device,
                        host_resident_replay=host_resident_final_decoder,
                    )
                )
                del skip
                decoder_index += 2
                final_attention = self.decoder[decoder_index]
                if (
                    not isinstance(final_attention, Residual)
                    or not isinstance(final_attention.fn, PreNorm)
                    or not isinstance(final_attention.fn.fn, TemporalAttention)
                ):
                    raise RuntimeError(
                        "final decoder temporal attention contract changed"
                    )
                if host_resident_final_decoder:
                    x = (
                        final_attention.fn.fn
                        .forward_pre_norm_residual_head_from_pinned_cpu(
                            x,
                            final_attention.fn.norm,
                            self.output_head,
                            execution_device=execution_device,
                        )
                    )
                else:
                    x = final_attention(x, output_head=self.output_head)
                if x.device != execution_device or x.shape != (
                    batch,
                    self.out_channels,
                    time,
                    depth,
                    height,
                    width,
                ):
                    raise RuntimeError(
                        "final decoder head changed output device or full-grid shape"
                    )
                output_head_applied = True
                decoder_index += 1
                continue

            if (
                is_final_decoder_stage
                and self.training
                and torch.is_grad_enabled()
                and x.is_cuda
                and torch.is_autocast_enabled("cuda")
            ):
                # ConvTranspose3d must allocate its complete frame-major output
                # before the subsequent concatenation boundary.  Backward slab
                # recomputation can leave differently-sized, unused allocator
                # blocks cached from the encoder and prior decoder level; on a
                # 40-GiB A100 those inert blocks alone can exceed the remaining
                # margin.  Return only allocator-cached free blocks before the
                # transpose convolution.  Live tensors, values, autograd edges,
                # and the mathematical convolution are unchanged.
                torch.cuda.empty_cache()
            x = self._apply_per_frame(
                upsample, x, keep_frame_major_storage=True
            )
            if x.shape[-3:] != skip.shape[-3:]:
                raise RuntimeError(
                    "TC-UNet symmetric skip shapes differ; interpolation-based "
                    f"repair is forbidden ({x.shape[-3:]} != {skip.shape[-3:]})"
                )
            # The upsample branch already has the dtype that autocast will feed
            # to the following Conv3d.  Match the skip before concatenation so
            # PyTorch does not promote a BF16 upsample to a giant FP32 tensor
            # only for Conv3d to cast it straight back to BF16.  This preserves
            # the exact convolution input values and the cast-gradient path.
            if skip.dtype != x.dtype:
                skip = skip.to(dtype=x.dtype)
            if (
                self.training
                and torch.is_grad_enabled()
                and x.is_cuda
                and torch.is_autocast_enabled("cuda")
            ):
                # The following frame-major concatenation is a single large
                # allocation.  Return only allocator-cached free blocks first;
                # live tensors are untouched and the numerical graph is
                # unchanged.  This prevents prior differently-sized slab
                # workspaces from making a physically sufficient A100 appear
                # fragmented at the final decoder level.
                torch.cuda.empty_cache()
            # Concatenate the same channels in frame-major storage.  The first
            # Conv4DBlock immediately views its channel-first input as
            # [B,T,C,D,H,W]; materialising the concatenation in that order
            # prevents a second multi-gigabyte contiguous copy without changing
            # a value, operator, channel order, or temporal receptive field.
            x = torch.cat(
                [
                    x.permute(0, 2, 1, 3, 4, 5),
                    skip.permute(0, 2, 1, 3, 4, 5),
                ],
                dim=2,
            ).permute(0, 2, 1, 3, 4, 5)
            if not x.permute(0, 2, 1, 3, 4, 5).is_contiguous():
                raise RuntimeError("TC-UNet frame-major skip layout changed")
            del skip
            x = self._run_block(
                self.decoder[decoder_index], x, temporal_cond, graph_cond
            )
            decoder_index += 1
            x = self._run_block(
                self.decoder[decoder_index], x, temporal_cond, graph_cond
            )
            decoder_index += 1
            x = self.decoder[decoder_index](x)
            decoder_index += 1

        # The manuscript does not publish an output-domain parameterisation,
        # while the authenticated target contract is finite [0,1]. A
        # differentiable sigmoid is therefore an explicit v10 implementation
        # choice shared by training and inference; inference never clamps.
        logits = (
            x
            if output_head_applied
            else self._apply_per_frame(self.output_head, x)
        )
        output = torch.sigmoid(logits)
        if output.shape[1] != 1 or output.shape[-3:] != (depth, height, width):
            raise RuntimeError(
                "Figure-1 1x1x1 final head changed channels or spatial shape"
            )
        if output_mask is not None:
            output = output * output_mask.to(output.dtype)
        return output[:, 0]

    def forward(
        self,
        x: torch.Tensor,
        graph_cond: torch.Tensor,
        temporal_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode an already full-grid [B,C,T,D,H,W] Figure-1 feature tensor."""

        return self._forward_features(
            x,
            graph_cond,
            temporal_indices,
            mask,
            require_full_depth=True,
        )

    def decode_dit_volume(
        self,
        dit_volume: torch.Tensor,
        projection: DiTToTCUNetProjection,
        graph_cond: torch.Tensor,
        temporal_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        depth_slab_size: int = 0,
    ) -> torch.Tensor:
        """Project and decode either whole-grid or with exact derived D halos."""

        if not isinstance(projection, DiTToTCUNetProjection):
            raise TypeError("projection must be DiTToTCUNetProjection")
        if projection.out_channels != self.in_channels:
            raise ValueError("projection output channels differ from TC-UNet input")
        if projection.spatial_size != self.input_spatial_size:
            raise ValueError("projection and TC-UNet configured grids differ")
        if isinstance(depth_slab_size, bool) or int(depth_slab_size) < 0:
            raise ValueError("depth_slab_size must be a non-negative integer")
        slab_size = int(depth_slab_size)
        if dit_volume.ndim != 6:
            raise ValueError("DiT volume must be [B,C,T,D,H,W]")
        full_depth = dit_volume.shape[-3]
        if slab_size == 0 or slab_size >= full_depth:
            return self.forward(
                projection(dit_volume),
                graph_cond,
                temporal_indices,
                mask,
            )

        expected = projection.spatial_size
        if tuple(dit_volume.shape[-3:]) != expected:
            raise ValueError(
                "slab execution requires a DiT volume on the complete configured grid"
            )
        if mask is not None and tuple(mask.shape[-3:]) != expected:
            raise ValueError("slab mask must already match the complete configured grid")
        temporal_indices = self._normalise_temporal_indices(
            temporal_indices,
            batch=dit_volume.shape[0],
            time=dit_volume.shape[2],
            device=dit_volume.device,
        )
        halo = self.depth_receptive_field_radius
        output_slabs = []
        for core_start in range(0, full_depth, slab_size):
            core_end = min(full_depth, core_start + slab_size)
            extended_start = max(0, core_start - halo)
            extended_end = min(full_depth, core_end + halo)
            latent_slab = dit_volume[..., extended_start:extended_end, :, :]
            mask_slab = (
                None
                if mask is None
                else mask[..., extended_start:extended_end, :, :]
            )

            def run_slab(
                latent_arg: torch.Tensor,
                graph_arg: torch.Tensor,
                temporal_arg: torch.Tensor,
                mask_arg: Optional[torch.Tensor],
            ) -> torch.Tensor:
                def replay_complete_slab() -> torch.Tensor:
                    features = projection.project_depth_slab(latent_arg)
                    return self._forward_features(
                        features,
                        graph_arg,
                        temporal_arg,
                        mask_arg,
                        require_full_depth=False,
                    )

                if (
                    self.training
                    and self.use_checkpoint
                    and torch.is_grad_enabled()
                    and latent_arg.is_cuda
                ):
                    # Reentrant checkpoint backward replays this function after
                    # the caller's original saved-tensor hook scope has ended.
                    # Re-establish byte-exact pinned-CPU residency for every
                    # replay-saved tensor across the complete slab. This changes
                    # storage location only; autocast state, operators, inputs,
                    # full T, halo, and gradient equations remain unchanged.
                    with torch.autograd.graph.save_on_cpu(
                        pin_memory=True,
                        device_type="cuda",
                    ):
                        return replay_complete_slab()
                return replay_complete_slab()

            # The outer slab boundary is deliberately reentrant: its original
            # forward runs without autograd recording, so fine checkpoint inputs
            # cannot accumulate across all 32 overlapping slabs. During backward
            # exactly one slab is replayed with gradients enabled; the inner
            # non-reentrant block/attention checkpoints then keep their exact
            # BF16 inputs on pinned CPU. This changes residency only, not the slab
            # operator, halo, full T=128 extent, or gradient equation.
            if self.training and self.use_checkpoint and latent_slab.requires_grad:
                if mask_slab is None:
                    decoded_slab = checkpoint.checkpoint(
                        lambda latent_arg, graph_arg, temporal_arg: run_slab(
                            latent_arg, graph_arg, temporal_arg, None
                        ),
                        latent_slab,
                        graph_cond,
                        temporal_indices,
                        use_reentrant=True,
                    )
                else:
                    decoded_slab = checkpoint.checkpoint(
                        run_slab,
                        latent_slab,
                        graph_cond,
                        temporal_indices,
                        mask_slab,
                        use_reentrant=True,
                    )
            else:
                decoded_slab = run_slab(
                    latent_slab, graph_cond, temporal_indices, mask_slab
                )
            local_start = core_start - extended_start
            local_end = local_start + (core_end - core_start)
            # A narrow view would retain the complete haloed decoder allocation
            # for every prior slab.  Clone only the exact core so completed
            # halos can be released while preserving values and gradients.
            output_slabs.append(
                decoded_slab[:, :, local_start:local_end].clone(
                    memory_format=torch.contiguous_format
                )
            )
            del decoded_slab
        output = torch.cat(output_slabs, dim=2)
        if tuple(output.shape[-3:]) != expected:
            raise RuntimeError("depth-slab assembly did not restore the full grid")
        return output


if __name__ == "__main__":
    # Test the model
    model = TCUNet4DFiLM(
        in_channels=32,
        out_channels=1,
        base_channels=32,
        num_levels=3,
        temporal_cond_dim=64,
        graph_cond_dim=256,
    ).cuda()
    
    B, C, T, D, H, W = 2, 32, 10, 32, 32, 32
    x = torch.randn(B, C, T, D, H, W).cuda()
    graph_cond = torch.randn(B, 256).cuda()
    temporal_indices = torch.arange(T, dtype=torch.float32).expand(B, -1).cuda()
    
    out = model(x, graph_cond, temporal_indices)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Model test passed!")
