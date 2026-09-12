"""Conditional 4D Diffusion Transformer with a DDIM training/sampling path.

Unlike the historical raw-volume denoiser, ``DiT4DTemporal`` is a token-latent
denoiser. A clean 4D target is patch-encoded once to ``[B,N,H]`` and forward
noise is added in that complete token coordinate system. The DiT receives the
noisy tokens and a diffusion noise level ``t``; it never compresses a raw
65,536-value production patch while trying to predict raw iid epsilon. Each
block applies validity-
masked per-frame 3D window attention over the complete 8x8x8 paper patch grid,
global temporal attention at each spatial location, timestep modulation,
cross-attention to separate T1-derived and graph/ROI tokens, and a position-wise
MLP. ``DDIMScheduler`` provides forward noising, x0 recovery, and deterministic/
stochastic DDIM reverse steps. Recovered x0 tokens are mapped back to scalar
4D patches by the exact transpose of the same row-orthonormal projection,
with no independently trainable decoder scale or bias.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture_contract import (
    TOKEN_CODEC_STATE_VERSION,
    TOKEN_CODEC_ORTHONORMAL_ATOL,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE,
    TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE,
    TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE,
    TOKEN_LATENT_NORMALIZATION_CONTRACT,
    geometric_patch_centers_voxel,
)
from models.dit4d import DiTBlock4D, TimestepEmbedder


TOKEN_LATENT_DIFFUSION_DOMAIN = "connect4-token-latent-diffusion-v1"

_FORBIDDEN_LEGACY_CODEC_COMPONENT_PREFIXES = (
    "clean_token_encoder",
    "token_decoder",
)
_TOKEN_CODEC_STATE_MARKER_COMPONENT = "_token_codec_state_version"
_TOKEN_CODEC_RAW_DETAIL_COMPONENT = "token_codec_raw_detail_rows"
_TOKEN_CODEC_STATE_COMPONENTS = frozenset(
    {
        _TOKEN_CODEC_STATE_MARKER_COMPONENT,
        _TOKEN_CODEC_RAW_DETAIL_COMPONENT,
    }
)
_DIRECT_DIT_CODEC_STATE_PATHS = frozenset(_TOKEN_CODEC_STATE_COMPONENTS)


def _contains_forbidden_legacy_codec_component(name: str) -> bool:
    return any(
        forbidden in component
        for component in name.split(".")
        if component
        for forbidden in _FORBIDDEN_LEGACY_CODEC_COMPONENT_PREFIXES
    )


def _finite_tensor_values(value: torch.Tensor) -> bool:
    if not (value.is_floating_point() or value.is_complex()):
        return True
    if value.layout != torch.strided:
        value = value.values()
    return bool(torch.isfinite(value.detach()).all())


def _validate_checkpoint_state_tree(
    state: object,
    *,
    label: str,
    prefix: str = "",
    allowed_codec_state_paths: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    """Audit one state tree before load or publication.

    The root must be a mapping with string keys and tensor values. Nested
    mappings are traversed only to preserve cycle-safe, complete-path
    diagnostics for forbidden codec state, non-string keys, unsupported leaves,
    and non-finite tensors; every nested mapping is then rejected because the
    supported PyTorch state-dict surface is flat. This deliberately rejects
    state that ``strict=False`` could otherwise ignore.
    """

    if not isinstance(state, Mapping):
        raise RuntimeError(f"{label} must be a tensor-state mapping")
    if any(
        not isinstance(path, str) or not path or path.startswith(".")
        for path in allowed_codec_state_paths
    ):
        raise RuntimeError(f"{label} has an invalid allowed token-codec path")
    checked = 0
    values = 0
    tensor_leaves = 0
    active_mappings: set[int] = set()

    def visit(
        value: object,
        path: str,
        *,
        forbidden_ancestor: bool = False,
        nested_mapping_depth: int = 0,
    ) -> None:
        nonlocal checked, tensor_leaves, values
        codec_components = tuple(
            component
            for component in path.split(".")
            if component in _TOKEN_CODEC_STATE_COMPONENTS
        )
        if codec_components and (
            path not in allowed_codec_state_paths or nested_mapping_depth
        ):
            raise RuntimeError(
                f"{label} contains token-codec state at unexpected path {path!r}"
            )
        forbidden_path = (
            forbidden_ancestor
            or _contains_forbidden_legacy_codec_component(path)
        )
        if torch.is_tensor(value):
            if forbidden_path:
                raise RuntimeError(
                    f"{label} contains forbidden legacy independent token-codec "
                    f"state at {path!r}"
                )
            tensor_leaves += 1
            if value.is_floating_point() or value.is_complex():
                checked += 1
                values += int(value.numel())
                if not _finite_tensor_values(value):
                    raise RuntimeError(
                        f"{label} contains a non-finite tensor at {path!r}"
                    )
            if path.split(".")[-1] == _TOKEN_CODEC_STATE_MARKER_COMPONENT:
                if (
                    value.numel() != 1
                    or value.dtype != torch.int64
                    or int(value.detach().cpu().item())
                    != TOKEN_CODEC_STATE_VERSION
                ):
                    raise RuntimeError(
                        f"{label} token-codec state marker at {path!r} is "
                        f"not V{TOKEN_CODEC_STATE_VERSION}; "
                        "V10/V11/V12/V13/V14/V15/V16/V17 state is forbidden"
                    )
            return
        if path in allowed_codec_state_paths:
            raise RuntimeError(
                f"{label} token-codec state at {path!r} must be a tensor"
            )
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active_mappings:
                raise RuntimeError(
                    f"{label} contains a cyclic mapping at {path!r}"
                )
            if not value:
                if forbidden_path:
                    raise RuntimeError(
                        f"{label} contains forbidden legacy independent "
                        f"token-codec state at {path!r}"
                    )
                raise RuntimeError(
                    f"{label} contains an empty mapping at {path!r}"
                )
            active_mappings.add(identity)
            try:
                for nested_key, nested_value in value.items():
                    if not isinstance(nested_key, str):
                        raise RuntimeError(
                            f"{label} mapping key at {path!r} must be a string; "
                            f"got {nested_key!r}"
                        )
                    nested_path = f"{path}.{nested_key}" if path else nested_key
                    visit(
                        nested_value,
                        nested_path,
                        forbidden_ancestor=forbidden_path,
                        nested_mapping_depth=nested_mapping_depth + 1,
                    )
            finally:
                active_mappings.remove(identity)
            raise RuntimeError(
                f"{label} contains a mapping-valued state entry at {path!r}; "
                "supported PyTorch state dicts must be flat string-to-tensor "
                "mappings"
            )
        if forbidden_path:
            raise RuntimeError(
                f"{label} contains forbidden legacy independent token-codec "
                f"state at {path!r}"
            )
        raise RuntimeError(
            f"{label} has unsupported {type(value).__name__} leaf at {path!r}; "
            "only tensors may be state-dict leaves"
        )

    root_identity = id(state)
    active_mappings.add(root_identity)
    try:
        for key, value in state.items():
            if not isinstance(key, str):
                raise RuntimeError(
                    f"{label} mapping key at '<root>' must be a string; got {key!r}"
                )
            if prefix and not key.startswith(prefix):
                continue
            visit(value, key)
    finally:
        active_mappings.remove(root_identity)
    if tensor_leaves == 0:
        raise RuntimeError(f"{label} contains no tensor state to audit")
    if checked == 0 or values == 0:
        raise RuntimeError(
            f"{label} contains no floating or complex tensor state to audit"
        )
    return checked, values


def _preflight_token_codec_checkpoint_state(
    state: object,
    *,
    label: str,
    codec_prefix: str,
    audit_prefix: str,
    expected_raw_detail_shape: tuple[int, int],
    raw_detail_validator,
) -> None:
    """Validate the exact codec subtree before any ``load_state_dict`` mutation."""

    allowed_codec_state_paths = frozenset(
        f"{codec_prefix}{component}"
        for component in _TOKEN_CODEC_STATE_COMPONENTS
    )
    _validate_checkpoint_state_tree(
        state,
        label=label,
        prefix=audit_prefix,
        allowed_codec_state_paths=allowed_codec_state_paths,
    )
    marker_path = f"{codec_prefix}{_TOKEN_CODEC_STATE_MARKER_COMPONENT}"
    raw_detail_path = f"{codec_prefix}{_TOKEN_CODEC_RAW_DETAIL_COMPONENT}"
    marker = state.get(marker_path) if isinstance(state, Mapping) else None
    if (
        not torch.is_tensor(marker)
        or marker.numel() != 1
        or marker.dtype != torch.int64
        or int(marker.detach().cpu().item()) != TOKEN_CODEC_STATE_VERSION
    ):
        raise RuntimeError(
            f"{label} token-codec state marker at {marker_path!r} is missing "
            f"or not V{TOKEN_CODEC_STATE_VERSION}; "
            "V10/V11/V12/V13/V14/V15/V16/V17 state is forbidden"
        )
    raw_detail = (
        state.get(raw_detail_path) if isinstance(state, Mapping) else None
    )
    if not torch.is_tensor(raw_detail):
        raise RuntimeError(
            f"{label} has no token-codec raw detail rows at {raw_detail_path!r}"
        )
    raw_detail_validator(
        raw_detail,
        expected_shape=expected_raw_detail_shape,
        label=f"{label} token-codec raw detail rows at {raw_detail_path!r}",
    )


class TokenFinalLayer(nn.Module):
    """adaLN epsilon head whose output stays in the token coordinate system."""

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        if hidden_size < 1 or output_size < 1:
            raise ValueError("token head dimensions must be positive")
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        dtype = tokens.dtype
        tokens = self.norm_final(tokens.float()).to(dtype)
        tokens = tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(tokens)


def _coordinate_encoding(coordinates: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding for 1D or multi-axis coordinates."""
    if coordinates.ndim == 1:
        coordinates = coordinates.unsqueeze(-1)
    num_axes = coordinates.shape[-1]
    pairs = (dim + 1) // 2
    pair_ids = torch.arange(pairs, device=coordinates.device)
    axis = pair_ids.remainder(num_axes)
    band = torch.div(pair_ids, num_axes, rounding_mode="floor")
    num_bands = max(1, math.ceil(pairs / num_axes))
    denominator = max(1, num_bands - 1)
    frequency = torch.exp(
        -math.log(10_000.0) * band.float() / denominator
    ).to(dtype=coordinates.dtype)
    angles = coordinates[:, axis] * frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :dim]


class DDIMStepOutput(NamedTuple):
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class DDIMScheduler(nn.Module):
    """Minimal epsilon-prediction DDIM schedule used for training and sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        clip_sample: bool = False,
    ) -> None:
        super().__init__()
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2")
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif schedule == "scaled_linear":
            betas = torch.linspace(
                math.sqrt(beta_start), math.sqrt(beta_end), num_train_timesteps
            ).square()
        else:
            raise ValueError(f"unsupported beta schedule: {schedule}")
        alphas = 1.0 - betas
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def _extract(
        self, values: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=values.device, dtype=torch.long)
        selected = values.index_select(0, timesteps)
        return selected.to(device=sample.device, dtype=sample.dtype).reshape(
            (-1,) + (1,) * (sample.ndim - 1)
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if original_samples.shape != noise.shape:
            raise ValueError("original_samples and noise must have identical shapes")
        sqrt_alpha = self._extract(
            self.alphas_cumprod.sqrt(), timesteps, original_samples
        )
        sqrt_one_minus = self._extract(
            (1.0 - self.alphas_cumprod).sqrt(), timesteps, original_samples
        )
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def predict_original_sample(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod, timesteps, sample)
        prediction = (sample - (1.0 - alpha).sqrt() * model_output) / alpha.sqrt()
        return prediction.clamp(-1.0, 1.0) if self.clip_sample else prediction

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
        previous_timestep: torch.Tensor | int,
        *,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> DDIMStepOutput:
        batch_size = sample.shape[0]

        def _batch_time(value: torch.Tensor | int) -> torch.Tensor:
            if torch.is_tensor(value):
                value = value.to(device=sample.device, dtype=torch.long).reshape(-1)
                if value.numel() == 1:
                    value = value.expand(batch_size)
                if value.numel() != batch_size:
                    raise ValueError("timestep must be scalar or have shape [B]")
                return value
            return torch.full(
                (batch_size,), int(value), device=sample.device, dtype=torch.long
            )

        timestep = _batch_time(timestep)
        previous_timestep = _batch_time(previous_timestep)
        alpha_t = self._extract(self.alphas_cumprod, timestep, sample)
        prev_index = previous_timestep.clamp_min(0)
        alpha_prev = self._extract(self.alphas_cumprod, prev_index, sample)
        terminal = (previous_timestep < 0).reshape(
            (-1,) + (1,) * (sample.ndim - 1)
        )
        alpha_prev = torch.where(terminal, torch.ones_like(alpha_prev), alpha_prev)

        pred_original = self.predict_original_sample(sample, model_output, timestep)
        variance = (
            (1.0 - alpha_prev)
            / (1.0 - alpha_t).clamp_min(torch.finfo(sample.dtype).eps)
            * (1.0 - alpha_t / alpha_prev)
        ).clamp_min(0.0)
        std = float(eta) * variance.sqrt()
        direction = (1.0 - alpha_prev - std.square()).clamp_min(0.0).sqrt()
        prev_sample = alpha_prev.sqrt() * pred_original + direction * model_output
        if eta:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
            prev_sample = prev_sample + std * noise
        return DDIMStepOutput(prev_sample, pred_original)

    def inference_timesteps(
        self, num_inference_steps: int, device: torch.device
    ) -> torch.Tensor:
        if not 1 <= num_inference_steps <= self.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be between 1 and num_train_timesteps"
            )
        return torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).round().long()


class DiT4DTemporal(nn.Module):
    """Token-latent epsilon denoiser conditioned only on fused hypergraph tokens."""

    def __init__(
        self,
        input_size: Optional[tuple] = None,
        in_channels: int = 1,
        patch_size: Optional[tuple] = None,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        input_dim: Optional[int] = None,
        num_temporal_frames: int = 128,
        t1_token_dim: Optional[int] = None,
        t1_channels: int = 1,
        temporal_patch_size: int = 1,
        num_diffusion_steps: int = 1_000,
        beta_schedule: str = "linear",
        spatial_window_size: Tuple[int, int, int] = (8, 8, 8),
        diffusion_domain: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Accepted only for checkpoint/config API compatibility. Raw T1 is
        # already represented by the image nodes in the fused hypergraph and
        # is never consumed as a separate denoiser input.
        del t1_token_dim, t1_channels
        if diffusion_domain != TOKEN_LATENT_DIFFUSION_DOMAIN:
            raise ValueError(
                "models.dit.diffusion_domain must explicitly select "
                f"{TOKEN_LATENT_DIFFUSION_DOMAIN!r}; raw-direct voxel diffusion "
                "is forbidden by the v18 synthesis contract"
            )
        if input_size is None or patch_size is None:
            raise ValueError(
                "input_size and patch_size must be explicit versioned choices; "
                "the manuscript does not define a spatial matrix"
            )
        if len(input_size) != 3 or len(patch_size) != 3:
            raise ValueError("input_size and patch_size must both be 3D")
        if any(value < 1 for value in (*input_size, *patch_size)):
            raise ValueError("input_size and patch_size entries must be positive")
        if any(size % patch for size, patch in zip(input_size, patch_size)):
            raise ValueError("input_size must be divisible by patch_size on every axis")
        if (
            depth < 1
            or num_heads < 1
            or hidden_size < 1
            or in_channels < 1
            or num_temporal_frames < 1
        ):
            raise ValueError(
                "depth, heads, channels, frames, and hidden_size must be positive"
            )
        if num_temporal_frames < 2 or any(size < 2 for size in input_size):
            raise ValueError(
                "token codec first-difference support requires T,D,H,W greater than one"
            )
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not math.isfinite(float(mlp_ratio)) or float(mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must be positive and finite")

        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels
        self.learn_sigma = bool(learn_sigma)
        self.hidden_size = int(hidden_size)
        self.epsilon_target_dim = self.hidden_size
        self.denoiser_output_dim = self.hidden_size * (2 if learn_sigma else 1)
        self.diffusion_domain = diffusion_domain
        self.token_latent_normalization_contract = (
            TOKEN_LATENT_NORMALIZATION_CONTRACT
        )
        self.register_buffer(
            "_token_codec_state_version",
            torch.tensor(TOKEN_CODEC_STATE_VERSION, dtype=torch.int64),
            persistent=True,
        )
        self.num_temporal_frames = int(num_temporal_frames)
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size < 1:
            raise ValueError("temporal_patch_size must be positive")
        if self.num_temporal_frames % self.temporal_patch_size:
            raise ValueError(
                "num_temporal_frames must be divisible by temporal_patch_size"
            )
        self.temporal_grid = self.num_temporal_frames // self.temporal_patch_size
        self.spatial_grid = tuple(
            size // patch for size, patch in zip(input_size, patch_size)
        )
        self.num_patches = math.prod(self.spatial_grid)
        self.spatial_window_size = tuple(int(value) for value in spatial_window_size)
        if self.spatial_window_size != (8, 8, 8):
            raise ValueError(
                "Figure-1 v18 spatial_window_size is the closed 8x8x8 choice"
            )
        if any(
            value < grid
            for value, grid in zip(self.spatial_window_size, self.spatial_grid)
        ):
            raise ValueError(
                "spatial_window_size must cover the complete 3D patch grid"
            )
        self.register_buffer(
            "spatiotemporal_patch_centers",
            torch.tensor(
                geometric_patch_centers_voxel(
                    (self.num_temporal_frames, *self.input_size),
                    (self.temporal_patch_size, *self.patch_size),
                ),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "spatiotemporal_position_encoding",
            _coordinate_encoding(
                self.spatiotemporal_patch_centers,
                self.hidden_size,
            ),
            persistent=False,
        )

        self.patch_value_dim = (
            self.in_channels
            * self.temporal_patch_size
            * math.prod(self.patch_size)
        )
        if self.hidden_size < 2 or self.hidden_size > self.patch_value_dim:
            raise ValueError(
                "V18 tied projection codec requires 2 <= hidden_size <= "
                f"patch_value_dim ({self.hidden_size} versus {self.patch_value_dim})"
            )
        self.token_codec_raw_detail_rows = nn.Parameter(
            torch.empty(self.hidden_size - 1, self.patch_value_dim - 1)
        )
        graph_dim = input_dim if input_dim is not None else hidden_size
        self.graph_condition_proj = (
            nn.Linear(graph_dim, hidden_size)
            if graph_dim != hidden_size
            else nn.Identity()
        )
        self.condition_type = nn.Parameter(torch.zeros(1, hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock4D(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    spatial_window_size=self.spatial_window_size,
                )
                for _ in range(depth)
            ]
        )
        self.epsilon_head = TokenFinalLayer(
            self.hidden_size,
            self.denoiser_output_dim,
        )
        if self.epsilon_target_dim != self.hidden_size:
            raise RuntimeError("epsilon target and token latent dimensions differ")
        if self.denoiser_output_dim not in {
            self.epsilon_target_dim,
            2 * self.epsilon_target_dim,
        }:
            raise RuntimeError("epsilon target and denoiser output dimensions differ")
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            schedule=beta_schedule,
        )
        self.initialize_weights()
        self.validate_token_codec_invariants()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        with torch.no_grad():
            nn.init.normal_(self.token_codec_raw_detail_rows)
            detail_basis, _ = torch.linalg.qr(
                self.token_codec_raw_detail_rows.float().transpose(0, 1),
                mode="reduced",
            )
            self.token_codec_raw_detail_rows.copy_(detail_basis.transpose(0, 1))
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.epsilon_head.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.epsilon_head.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.epsilon_head.linear.weight)
        nn.init.zeros_(self.epsilon_head.linear.bias)

    @staticmethod
    def _validated_raw_detail_rows(
        value: torch.Tensor,
        *,
        expected_shape: tuple[int, int],
        label: str,
    ) -> tuple[torch.Tensor, float, float, float, float, float]:
        """Validate the full-rank, bounded-gauge V18 coordinates in FP32."""

        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"{label} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise RuntimeError(f"{label} must use a real floating-point dtype")
        # This invariant must mean the same thing inside a BF16/FP16 training
        # autocast region as it does in an isolated FP32 checkpoint audit.
        with torch.autocast(device_type=value.device.type, enabled=False):
            value_fp32 = value.float()
            if not bool(torch.isfinite(value_fp32).all()):
                raise RuntimeError(f"{label} contains NaN or infinity")
            singular_values = torch.linalg.svdvals(value_fp32.detach())
        if singular_values.numel() != expected_shape[0] or not bool(
            torch.isfinite(singular_values).all()
        ):
            raise RuntimeError(f"{label} singular values are invalid")
        minimum = float(singular_values.amin().item())
        maximum = float(singular_values.amax().item())
        condition_number = maximum / minimum if minimum > 0 else math.inf
        singular_values_fp64 = singular_values.double()
        frobenius_norm = float(
            torch.linalg.vector_norm(singular_values_fp64).item()
        )
        rms_singular_value = float(
            (frobenius_norm / math.sqrt(singular_values.numel()))
        )
        if minimum < TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE:
            raise RuntimeError(
                f"{label} minimum singular value {minimum:.9g} is below the V18 "
                f"floor {TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE}"
            )
        if maximum > TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE:
            raise RuntimeError(
                f"{label} maximum singular value {maximum:.9g} exceeds the V18 "
                f"ceiling {TOKEN_CODEC_RAW_DETAIL_MAXIMUM_SINGULAR_VALUE}"
            )
        if condition_number > TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER:
            raise RuntimeError(
                f"{label} condition number {condition_number:.9g} exceeds the "
                f"V18 ceiling {TOKEN_CODEC_RAW_DETAIL_MAXIMUM_CONDITION_NUMBER}"
            )
        if rms_singular_value > TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE:
            raise RuntimeError(
                f"{label} RMS singular value {rms_singular_value:.9g} exceeds "
                "the V18 normalized Frobenius-norm ceiling "
                f"{TOKEN_CODEC_RAW_DETAIL_MAXIMUM_RMS_SINGULAR_VALUE}"
            )
        return (
            value_fp32,
            minimum,
            maximum,
            condition_number,
            frobenius_norm,
            rms_singular_value,
        )

    def _fixed_dc_householder(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Construct the exact V18 DC row and its orthogonal-basis transform."""

        dc_row = torch.full(
            (self.patch_value_dim,),
            1.0 / math.sqrt(self.patch_value_dim),
            device=device,
            dtype=torch.float32,
        )
        coordinate_axis = torch.zeros(
            self.patch_value_dim,
            device=device,
            dtype=torch.float32,
        )
        coordinate_axis[0] = 1.0
        householder = coordinate_axis - dc_row
        householder = householder / torch.linalg.vector_norm(householder)
        return dc_row, householder

    def _validated_effective_token_codec(
        self,
    ) -> tuple[torch.Tensor, float, float, float, float, float, float]:
        """Return the tied FP32 encoder after all V18 runtime invariants."""

        marker = self._token_codec_state_version
        if (
            marker.numel() != 1
            or marker.dtype != torch.int64
            or int(marker.detach().cpu().item()) != TOKEN_CODEC_STATE_VERSION
        ):
            raise RuntimeError("V18 token-codec runtime state marker differs")
        forbidden_runtime_names = tuple(
            sorted(
                {
                    f"{kind}:{name}"
                    for kind, named_values in (
                        ("module", self.named_modules()),
                        ("parameter", self.named_parameters(recurse=True)),
                        ("buffer", self.named_buffers(recurse=True)),
                    )
                    for name, _ in named_values
                    if _contains_forbidden_legacy_codec_component(name)
                }
            )
        )
        if forbidden_runtime_names:
            raise RuntimeError(
                "V18 recursively forbids independent clean-token encoder/decoder "
                f"modules, parameters, or buffers: {forbidden_runtime_names}"
            )
        (
            raw,
            minimum_singular_value,
            maximum_singular_value,
            condition_number,
            frobenius_norm,
            rms_singular_value,
        ) = self._validated_raw_detail_rows(
            self.token_codec_raw_detail_rows,
            expected_shape=(self.hidden_size - 1, self.patch_value_dim - 1),
            label="token-codec raw detail rows",
        )
        with torch.autocast(device_type=raw.device.type, enabled=False):
            coordinate_columns, triangular = torch.linalg.qr(
                raw.transpose(0, 1),
                mode="reduced",
            )
            diagonal = torch.diagonal(triangular).detach()
            orientation = torch.where(
                diagonal < 0,
                -torch.ones_like(diagonal),
                torch.ones_like(diagonal),
            )
            coordinate_detail_rows = (
                coordinate_columns * orientation.unsqueeze(0)
            ).transpose(0, 1)
            coordinate_detail_rows = F.pad(coordinate_detail_rows, (1, 0))

            dc_row, householder = self._fixed_dc_householder(raw.device)
            detail_rows = coordinate_detail_rows - 2.0 * (
                coordinate_detail_rows @ householder
            ).unsqueeze(1) * householder.unsqueeze(0)
            effective_encoder = torch.cat(
                (dc_row.unsqueeze(0), detail_rows), dim=0
            )
            expected_shape = (self.hidden_size, self.patch_value_dim)
            if tuple(effective_encoder.shape) != expected_shape:
                raise RuntimeError(
                    "V18 effective token-codec encoder shape differs from "
                    f"{expected_shape}"
                )
            if not torch.equal(effective_encoder[0], dc_row):
                raise RuntimeError("V18 token-codec DC row differs")
            gram = effective_encoder @ effective_encoder.transpose(0, 1)
            identity = torch.eye(
                self.hidden_size,
                device=gram.device,
                dtype=gram.dtype,
            )
            gram_error = float((gram - identity).detach().abs().amax().item())
        if not math.isfinite(gram_error) or gram_error > TOKEN_CODEC_ORTHONORMAL_ATOL:
            raise RuntimeError(
                f"V18 token-codec E E^T error {gram_error:.9g} exceeds "
                f"{TOKEN_CODEC_ORTHONORMAL_ATOL}"
            )
        return (
            effective_encoder,
            minimum_singular_value,
            maximum_singular_value,
            condition_number,
            frobenius_norm,
            rms_singular_value,
            gram_error,
        )

    def effective_token_codec_encoder(self) -> torch.Tensor:
        """Return the differentiable V18 encoder shared by encode and decode."""

        effective_encoder, *_ = self._validated_effective_token_codec()
        return effective_encoder

    def validate_token_codec_invariants(
        self,
    ) -> Dict[str, bool | float | int | str]:
        """Fail closed on V18 runtime/post-load/post-optimizer invariants."""

        (
            _,
            minimum_singular_value,
            maximum_singular_value,
            condition_number,
            frobenius_norm,
            rms_singular_value,
            gram_error,
        ) = (
            self._validated_effective_token_codec()
        )
        return {
            "contract": TOKEN_LATENT_NORMALIZATION_CONTRACT,
            "state_version": TOKEN_CODEC_STATE_VERSION,
            "minimum_raw_detail_singular_value": minimum_singular_value,
            "maximum_raw_detail_singular_value": maximum_singular_value,
            "raw_detail_condition_number": condition_number,
            "raw_detail_frobenius_norm": frobenius_norm,
            "raw_detail_rms_singular_value": rms_singular_value,
            "effective_encoder_gram_max_abs_error": gram_error,
            "effective_encoder_finite_fp32": True,
            "fixed_dc_row_exact": True,
            "encoder_bias_parameter_present": False,
            "decoder_is_exact_transpose": True,
            "decoder_parameter_count": 0,
            "decoder_bias_parameter_present": False,
            "decoder_operator_norm": 1.0,
        }

    def validate_model_state_invariants(
        self,
    ) -> Dict[str, bool | float | int | str]:
        """Validate the codec and every recursively registered tensor state."""

        report = self.validate_token_codec_invariants()
        named_state = {
            **dict(self.named_parameters(recurse=True)),
            **dict(self.named_buffers(recurse=True)),
        }
        tensor_count, value_count = _validate_checkpoint_state_tree(
            named_state,
            label="V18 DiT runtime model state",
            allowed_codec_state_paths=_DIRECT_DIT_CODEC_STATE_PATHS,
        )
        return {
            **report,
            "model_state_finite": True,
            "model_state_floating_tensor_count": tensor_count,
            "model_state_floating_value_count": value_count,
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Reject non-finite or pre-V18 state before and after every load."""

        _preflight_token_codec_checkpoint_state(
            state_dict,
            label="V18 DiT checkpoint state",
            codec_prefix="",
            audit_prefix="",
            expected_raw_detail_shape=(
                self.hidden_size - 1,
                self.patch_value_dim - 1,
            ),
            raw_detail_validator=self._validated_raw_detail_rows,
        )
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        self.validate_model_state_invariants()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Reject pre-V18 or malformed codec state, including non-strict loads."""

        _preflight_token_codec_checkpoint_state(
            state_dict,
            label="V18 DiT checkpoint state",
            codec_prefix=prefix,
            audit_prefix=prefix,
            expected_raw_detail_shape=(
                self.hidden_size - 1,
                self.patch_value_dim - 1,
            ),
            raw_detail_validator=self._validated_raw_detail_rows,
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.validate_token_codec_invariants()

    def _patchify_volume(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 6:
            raise ValueError("4D volume must have shape [B,C,T,D,H,W]")
        batch, channels, time, depth, height, width = volume.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} volume channels")
        if (depth, height, width) != self.input_size:
            raise ValueError(
                f"expected volume spatial shape {self.input_size}, got "
                f"{(depth, height, width)}"
            )
        if time != self.num_temporal_frames:
            raise ValueError(
                f"expected {self.num_temporal_frames} frames, got {time}"
            )
        temporal = self.temporal_grid
        gd, gh, gw = self.spatial_grid
        pt = self.temporal_patch_size
        pd, ph, pw = self.patch_size
        patches = volume.reshape(
            batch,
            channels,
            temporal,
            pt,
            gd,
            pd,
            gh,
            ph,
            gw,
            pw,
        )
        patches = patches.permute(0, 2, 4, 6, 8, 1, 3, 5, 7, 9).reshape(
            batch, temporal * self.num_patches, -1
        )
        if patches.shape[-1] != self.patch_value_dim:
            raise RuntimeError("patch vector dimension differs from the token codec")
        return patches

    def encode_clean_volume(self, clean_volume: torch.Tensor) -> torch.Tensor:
        """Encode the clean scalar 4D grid once, before forward noising."""

        effective_encoder = self.effective_token_codec_encoder()
        return self._encode_clean_volume_with_encoder(
            clean_volume,
            effective_encoder,
        )

    def _encode_clean_volume_with_encoder(
        self,
        clean_volume: torch.Tensor,
        effective_encoder: torch.Tensor,
    ) -> torch.Tensor:
        patches = self._patchify_volume(clean_volume)
        with torch.autocast(device_type=patches.device.type, enabled=False):
            weight = effective_encoder.to(
                device=patches.device,
                dtype=torch.float32,
            )
            return F.linear(patches.float(), weight, bias=None)

    def _spatiotemporal_tokens(
        self,
        noisy_latent: torch.Tensor,
        graph_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        if noisy_latent.ndim != 3:
            raise ValueError("noisy_latent must have token shape [B,N,H]")
        batch, token_count, width = noisy_latent.shape
        if width != self.hidden_size:
            raise ValueError(
                f"noisy token width must be {self.hidden_size}, got {width}"
            )
        expected_tokens = self.temporal_grid * self.num_patches
        if token_count != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} noisy spatiotemporal tokens, got "
                f"{token_count}"
            )
        if graph_tokens.ndim != 3 or graph_tokens.shape[0] != batch:
            raise ValueError("graph_tokens must have shape [B,N,D]")
        if graph_tokens.shape[1] != self.num_patches:
            raise ValueError(
                "graph tokens must align one-to-one with the latent spatial "
                f"patches: expected {self.num_patches}, got {graph_tokens.shape[1]}"
            )

        temporal = self.temporal_grid
        embedded = noisy_latent

        # Connectivity-aware graph tokens are aligned to every temporal group
        # and also remain available as a separate cross-attention context.
        aligned_graph = self.graph_condition_proj(graph_tokens)
        aligned_graph = aligned_graph[:, None].expand(
            batch, temporal, self.num_patches, self.hidden_size
        ).reshape(batch, temporal * self.num_patches, self.hidden_size)
        embedded = embedded + aligned_graph + self.condition_type[0]

        position_encoding = self.spatiotemporal_position_encoding.to(
            device=noisy_latent.device, dtype=embedded.dtype
        )
        if position_encoding.shape != (embedded.shape[1], self.hidden_size):
            raise RuntimeError(
                "cached canonical position encoding does not match latent tokens"
            )
        embedded = embedded + position_encoding.unsqueeze(0)
        return embedded, temporal

    def _conditioning_tokens(
        self,
        graph_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if graph_tokens.ndim != 3:
            raise ValueError("graph_tokens must have shape [B,N,D]")
        if not torch.isfinite(graph_tokens).all():
            raise ValueError("graph conditioning contains NaN or infinity")
        return self.graph_condition_proj(graph_tokens) + self.condition_type[0]

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        d, h, w = self.spatial_grid
        pt = self.temporal_patch_size
        pd, ph, pw = self.patch_size
        channels = self.in_channels
        expected = self.temporal_grid * self.num_patches
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
            expected,
            self.patch_value_dim,
        ):
            raise ValueError(
                f"decoded patches must have shape [B,{expected},"
                f"{self.patch_value_dim}]"
            )
        patches = patches.reshape(
            batch, self.temporal_grid, d, h, w, channels, pt, pd, ph, pw
        )
        patches = patches.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9)
        return patches.reshape(
            batch,
            channels,
            self.temporal_grid * pt,
            d * pd,
            h * ph,
            w * pw,
        )

    def decode_tokens(self, clean_tokens: torch.Tensor) -> torch.Tensor:
        """Decode with the exact transpose of the V18 effective encoder."""

        effective_encoder = self.effective_token_codec_encoder()
        return self._decode_tokens_with_encoder(clean_tokens, effective_encoder)

    def _decode_tokens_with_encoder(
        self,
        clean_tokens: torch.Tensor,
        effective_encoder: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.temporal_grid * self.num_patches, self.hidden_size)
        if clean_tokens.ndim != 3 or tuple(clean_tokens.shape[1:]) != expected:
            raise ValueError(
                f"clean tokens must have shape [B,{expected[0]},{expected[1]}]"
            )
        with torch.autocast(device_type=clean_tokens.device.type, enabled=False):
            effective_decoder = effective_encoder.transpose(0, 1).to(
                device=clean_tokens.device,
                dtype=torch.float32,
            )
            decoded = F.linear(
                clean_tokens.float(), effective_decoder, bias=None
            )
        return self._unpatchify(decoded)

    def codec_reconstruction_loss(
        self,
        clean_volume: torch.Tensor,
        *,
        target_validity_mask: torch.Tensor,
        clean_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Preserve voxel values and temporal/spatial first differences.

        This is a v18 implementation-only codec constraint, not a manuscript
        loss. The tied orthogonal projection preserves the DC component exactly
        and preserves detail only in its learned compressed row subspace.
        """

        effective_encoder = self.effective_token_codec_encoder()
        if clean_tokens is None:
            clean_tokens = self._encode_clean_volume_with_encoder(
                clean_volume,
                effective_encoder,
            )
        return self._codec_reconstruction_loss_with_encoder(
            clean_volume,
            target_validity_mask=target_validity_mask,
            clean_tokens=clean_tokens,
            effective_encoder=effective_encoder,
        )

    def _codec_reconstruction_loss_with_encoder(
        self,
        clean_volume: torch.Tensor,
        *,
        target_validity_mask: torch.Tensor,
        clean_tokens: torch.Tensor,
        effective_encoder: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        reconstruction = self._decode_tokens_with_encoder(
            clean_tokens,
            effective_encoder,
        )
        if reconstruction.shape != clean_volume.shape:
            raise RuntimeError("token codec reconstruction shape differs")
        batch, _, time, depth, height, width = clean_volume.shape
        expected_mask = (batch, 1, depth, height, width)
        if target_validity_mask is None:
            raise ValueError(
                "token codec loss requires an explicit target-validity mask"
            )
        if tuple(target_validity_mask.shape) != expected_mask:
            raise ValueError(
                "target_validity_mask must have shape "
                f"{expected_mask} for token codec loss"
            )
        if not torch.isfinite(target_validity_mask).all():
            raise ValueError(
                "token codec target_validity_mask contains NaN or infinity"
            )
        if not bool(
            ((target_validity_mask == 0) | (target_validity_mask == 1)).all()
        ):
            raise ValueError(
                "token codec target_validity_mask must be exactly binary"
            )
        mask = target_validity_mask.to(
            device=clean_volume.device,
            dtype=clean_volume.dtype,
        )
        mask = mask.unsqueeze(2).expand(batch, 1, time, depth, height, width)
        if not bool(mask.flatten(1).any(dim=1).all()):
            raise ValueError("token codec target_validity_mask is empty")

        squared_error = (reconstruction - clean_volume).square()
        voxel_support = mask.flatten(1).sum(dim=1)
        voxel = (
            (squared_error * mask).flatten(1).sum(dim=1) / voxel_support
        ).mean()
        differences = []
        for axis in (2, 3, 4, 5):
            reconstructed_delta = torch.diff(reconstruction, dim=axis)
            clean_delta = torch.diff(clean_volume, dim=axis)
            size = mask.shape[axis]
            if size < 2:
                raise ValueError("token codec delta axes must have length at least two")
            left = mask.narrow(axis, 0, size - 1)
            right = mask.narrow(axis, 1, size - 1)
            adjacent_brain = left * right
            support = adjacent_brain.flatten(1).sum(dim=1)
            if not bool((support > 0).all()):
                raise ValueError(
                    "a token codec target-validity mask has no adjacent support on "
                    f"axis {axis}"
                )
            differences.append(
                (
                    (
                        (reconstructed_delta - clean_delta).square()
                        * adjacent_brain
                    )
                    .flatten(1)
                    .sum(dim=1)
                    / support
                ).mean()
            )
        detail = torch.stack(differences).mean()
        return {
            "loss": voxel + detail,
            "voxel": voxel,
            "spatiotemporal_first_difference": detail,
            "reconstruction": reconstruction,
        }

    def forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        graph_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise using the fused three-stream hypergraph as conditioning."""
        tokens, temporal_patches = self._spatiotemporal_tokens(
            noisy_latent, graph_tokens
        )
        if t.ndim != 1 or t.shape[0] != noisy_latent.shape[0]:
            raise ValueError("diffusion timestep t must have shape [B]")
        condition = self.t_embedder(t)
        context = self._conditioning_tokens(graph_tokens)
        for block in self.blocks:
            tokens = block(
                tokens,
                condition,
                context=context,
                spatiotemporal_shape=(temporal_patches, *self.spatial_grid),
            )
        output = self.epsilon_head(tokens, condition)
        if output.shape[-1] != self.denoiser_output_dim:
            raise RuntimeError("DiT epsilon head emitted the wrong token dimension")
        return output

    def diffusion_loss(
        self,
        clean_latent: torch.Tensor,
        graph_tokens: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        target_validity_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode a clean target, then train epsilon prediction in token space."""
        effective_encoder = self.effective_token_codec_encoder()
        clean_tokens = self._encode_clean_volume_with_encoder(
            clean_latent,
            effective_encoder,
        )
        codec = self._codec_reconstruction_loss_with_encoder(
            clean_latent,
            target_validity_mask=target_validity_mask,
            clean_tokens=clean_tokens,
            effective_encoder=effective_encoder,
        )
        batch = clean_tokens.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (batch,),
                device=clean_tokens.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_tokens)
        if noise.shape != clean_tokens.shape:
            raise ValueError(
                "diffusion noise must have the exact encoded token-latent shape"
            )
        noisy = self.scheduler.add_noise(clean_tokens, noise, timesteps)
        prediction = self(noisy, timesteps, graph_tokens)
        predicted_noise = (
            prediction.chunk(2, dim=-1)[0] if self.learn_sigma else prediction
        )
        if predicted_noise.shape != noise.shape:
            raise RuntimeError("epsilon target and denoiser output shapes differ")
        pred_original = self.scheduler.predict_original_sample(
            noisy, predicted_noise, timesteps
        )
        pred_original_volume = self._decode_tokens_with_encoder(
            pred_original,
            effective_encoder,
        )
        spatial_validity = target_validity_mask.to(
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        ).unsqueeze(2).expand(
            clean_latent.shape[0],
            1,
            clean_latent.shape[2],
            *clean_latent.shape[-3:],
        )
        token_validity = self._patchify_volume(spatial_validity).mean(dim=-1)
        if tuple(token_validity.shape) != tuple(predicted_noise.shape[:2]):
            raise RuntimeError("target-validity token support shape differs")
        support = token_validity.sum(dim=1) * predicted_noise.shape[-1]
        if not bool((support > 0).all()):
            raise ValueError("target-validity mask has no diffusion token support")
        noise_squared_error = (predicted_noise - noise).square()
        diffusion_loss = (
            (
                noise_squared_error
                * token_validity.unsqueeze(-1).to(noise_squared_error.dtype)
            ).sum(dim=(1, 2))
            / support.to(noise_squared_error.dtype)
        ).mean()
        return {
            "loss": diffusion_loss,
            "predicted_noise": predicted_noise,
            "target_noise": noise,
            "clean_latent": clean_tokens,
            "codec_reconstruction_loss": codec["loss"],
            "codec_voxel_loss": codec["voxel"],
            "codec_first_difference_loss": codec[
                "spatiotemporal_first_difference"
            ],
            "noisy_latent": noisy,
            "pred_original_sample": pred_original,
            "pred_original_volume": pred_original_volume,
            "timesteps": timesteps,
            "token_validity": token_validity,
        }

    @torch.no_grad()
    def sample(
        self,
        graph_tokens: torch.Tensor,
        *,
        shape: Optional[Sequence[int]] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate clean latent tokens by iterating the DDIM trajectory."""
        batch = graph_tokens.shape[0]
        if shape is None:
            shape = (
                batch,
                self.temporal_grid * self.num_patches,
                self.hidden_size,
            )
        shape = tuple(int(value) for value in shape)
        expected = (
            batch,
            self.temporal_grid * self.num_patches,
            self.hidden_size,
        )
        if shape != expected:
            raise ValueError(f"sample shape must be {expected} token coordinates")
        if initial_noise is None:
            sample = torch.randn(
                shape,
                device=graph_tokens.device,
                dtype=graph_tokens.dtype,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial DDIM noise must have shape {shape}, got "
                    f"{tuple(initial_noise.shape)}"
                )
            if initial_noise.device != graph_tokens.device:
                raise ValueError(
                    "initial DDIM noise and conditioning must be on the same device"
                )
            if initial_noise.dtype != graph_tokens.dtype:
                raise ValueError(
                    "initial DDIM noise and conditioning must have the same dtype"
                )
            if not torch.isfinite(initial_noise).all():
                raise ValueError("initial DDIM noise contains NaN or infinity")
            sample = initial_noise.clone()
        timesteps = self.scheduler.inference_timesteps(
            num_inference_steps, graph_tokens.device
        )
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            batch_t = timestep.expand(batch)
            model_output = self(sample, batch_t, graph_tokens)
            predicted_noise = (
                model_output.chunk(2, dim=-1)[0]
                if self.learn_sigma
                else model_output
            )
            sample = self.scheduler.step(
                predicted_noise,
                timestep,
                sample,
                previous,
                eta=eta,
                generator=generator,
            ).prev_sample
        return sample


__all__ = [
    "DDIMScheduler",
    "DDIMStepOutput",
    "DiT4DTemporal",
    "TOKEN_CODEC_ORTHONORMAL_ATOL",
    "TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE",
    "TOKEN_CODEC_STATE_VERSION",
    "TOKEN_LATENT_DIFFUSION_DOMAIN",
    "TOKEN_LATENT_NORMALIZATION_CONTRACT",
    "TokenFinalLayer",
]
