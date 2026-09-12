"""
CONNECT-4 end-to-end model (Figure 1, panels A -> E).

This is the figure-faithful orchestrator. It wires the graph tokens into a
genuine conditional graph-token diffusion trajectory, decodes the recovered
clean tokens to a full scalar 4D grid once, and then applies the temporal UNet.

Data flow
---------
A) Input processing      : per-subject patch/ROI node features are produced by
                           frozen foundation models (BrainIAC for image patches,
                           AnatCL for ROIs, Clinical-ModernBERT for the
                           biological / atrophy text).  These arrive in the batch
                           as `image_nodes`, `mask_nodes`, `roi_nodes`.
B) Graph construction    : ImageGraph (Chebyshev), MaskGraph (DWI prior),
   + hyperedge fusion       ROIGraph (radiomics + AnatCL), then a hypergraph whose
                           hyperedges are weighted by ROI coverage, fused by
                           MultiModalFusion -> connectivity-aware patch tokens.
C) DiT generation        : DiT4DTemporal denoises noisy latent tokens with DDIM,
                           timestep modulation, and T1/graph cross-attention.
D) Temporal UNet         : decoded full-grid x0 is projected to 64 channels,
                           then TCUNet4DFiLM refines it without resizing D.
E) Losses                : Connect4Loss (3D SSIM, voxel intensity, volume,
                           region-histogram, temporal coherence, BrainLM
                           perceptual features, FC-matrix).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from architecture_contract import (
    CANONICAL_ROI_LABEL_TO_CHANNEL,
    CANONICAL_ROI_SPECS,
    PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
    PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
    PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
    TOKEN_LATENT_NORMALIZATION_CONTRACT,
    require_production_token_codec_shape,
)
from graphs import (
    ImageGraphBuilder,
    MaskGraphBuilder,
    ROIGraphBuilder,
    HypergraphBuilder,
)
from .fusion import MultiModalFusion
from .dit4d_temporal import (
    DiT4DTemporal,
    TOKEN_LATENT_DIFFUSION_DOMAIN,
    _TOKEN_CODEC_STATE_COMPONENTS,
    _contains_forbidden_legacy_codec_component,
    _preflight_token_codec_checkpoint_state,
    _validate_checkpoint_state_tree,
)
from .tc_film_unet import DiTToTCUNetProjection, TCUNet4DFiLM
from .losses import Connect4Loss
from .feature_extractors import build_pretrained_4d_extractor
from .brainlm_context import (
    BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA,
    BRAINLM_CONTEXT_SCHEMA,
    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    validate_configured_brainlm_identity,
)
from data.cohort_admission import validate_cohort_admission_identity
from data.dataset_precomputed import (
    RECOVERY_TARGET_VALIDITY_MASK_CONTRACT,
    TARGET_VALIDITY_MASK_CONTRACT,
)
from data.provenance import canonical_sha256
from utils.figure1_gpu_gate import (
    PROTOCOL_TO_BENCHMARK_PROFILE,
    authenticate_configured_depth_slab_selection,
)


PAPER_LOSS_WEIGHTS = {
    "voxel": 1.0,
    "ssim": 0.5,
    "fc": 0.3,
    "temporal": 0.2,
    "perceptual": 0.1,
}

SINGLE_CASE_DEVELOPMENT_SCAN_ID = "B34764547_004"
SINGLE_CASE_DEVELOPMENT_PROTOCOL_PROFILE = (
    "connect4-v18-single-case-development-proof-v1"
)
SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_SCHEMA = (
    "connect4-v18-single-case-development-brainlm-binding-v1"
)
SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_STATUS = (
    "NONPRODUCTION_DEVELOPMENT_RECONSTRUCTION_ONLY"
)
_SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_FIELDS = frozenset(
    {
        "schema",
        "status",
        "protocol_profile",
        "scan_id",
        "role",
        "same_case_target_used_for_optimization",
        "generalization_claim",
        "paper_certified",
        "sealed_holdout_opened",
        "ddim_sampler_target_blind",
        "paper_loss_weights_unchanged",
        "run_artifact_identity_sha256",
        "proof_admission_identity_record_sha256",
        "dataset_state_record_sha256",
        "dataset_state_sha256",
        "structural_artifact_identities_sha256",
        "native_alignment_authority_sha256",
        "brainlm_perceptual_identity_sha256",
        "record_sha256",
    }
)


def _is_complete_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_single_case_development_brainlm_binding(
    value: Mapping[str, Any],
    *,
    expected_brainlm_perceptual_identity_sha256: str,
) -> Dict[str, Any]:
    """Validate the explicitly non-production one-case BrainLM binding.

    This authority is deliberately distinct from the production cohort
    admission.  It permits a development-target reconstruction-capacity run
    without claiming either held-out generalisation or production readiness.
    """

    if not isinstance(value, Mapping) or set(value) != set(
        _SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_FIELDS
    ):
        raise RuntimeError("single-case development BrainLM binding fields differ")
    binding = dict(value)
    unsigned = dict(binding)
    recorded = unsigned.pop("record_sha256", None)
    digest_fields = {
        "run_artifact_identity_sha256",
        "proof_admission_identity_record_sha256",
        "dataset_state_record_sha256",
        "dataset_state_sha256",
        "structural_artifact_identities_sha256",
        "native_alignment_authority_sha256",
        "brainlm_perceptual_identity_sha256",
    }
    if (
        binding.get("schema") != SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_SCHEMA
        or binding.get("status") != SINGLE_CASE_DEVELOPMENT_BRAINLM_BINDING_STATUS
        or binding.get("protocol_profile") != SINGLE_CASE_DEVELOPMENT_PROTOCOL_PROFILE
        or binding.get("scan_id") != SINGLE_CASE_DEVELOPMENT_SCAN_ID
        or binding.get("role") != "development-validation"
        or binding.get("same_case_target_used_for_optimization") is not True
        or binding.get("generalization_claim") is not False
        or binding.get("paper_certified") is not False
        or binding.get("sealed_holdout_opened") is not False
        or binding.get("ddim_sampler_target_blind") is not True
        or binding.get("paper_loss_weights_unchanged") is not True
        or any(not _is_complete_sha256(binding.get(field)) for field in digest_fields)
        or binding.get("brainlm_perceptual_identity_sha256")
        != expected_brainlm_perceptual_identity_sha256
        or not _is_complete_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise RuntimeError("single-case development BrainLM binding differs")
    return binding


_TARGET_BLIND_SAMPLER_KEYS = (
    "image_nodes",
    "mask_nodes",
    "roi_nodes",
    "dwi_matrix",
    "structure_to_roi_idx",
    "patch_distributions",
    "t1w",
    "brain_mask",
    "roi_masks",
    "scan_id",
)
_DDIM_NOISE_DOMAIN = "connect4-target-blind-token-ddim-initial-noise-v2"


def _require_canonical_roi_mapping(value: object) -> dict[int, int]:
    """Return the one V18 aseg-label/channel mapping or fail closed."""

    expected = dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    if not isinstance(value, Mapping):
        raise ValueError("structure_to_roi_idx must be the canonical ROI mapping")
    observed = dict(value)
    if any(
        isinstance(label, bool)
        or not isinstance(label, int)
        or isinstance(channel, bool)
        or not isinstance(channel, int)
        for label, channel in observed.items()
    ):
        raise ValueError("canonical ROI labels and channel indices must be integers")
    if observed != expected:
        raise ValueError(
            "structure_to_roi_idx differs from the V18 canonical 32-ROI "
            "aseg-label/channel order"
        )
    return expected


def _require_contiguous_test_roi_mapping(
    value: object,
    *,
    expected_num_rois: int,
) -> dict[int, int]:
    """Keep reduced synthetic fixtures usable without weakening production."""

    if not isinstance(value, Mapping):
        raise ValueError("structure_to_roi_idx must be an integer mapping")
    observed = dict(value)
    if (
        len(observed) != expected_num_rois
        or any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or isinstance(channel, bool)
            or not isinstance(channel, int)
            for label, channel in observed.items()
        )
        or sorted(observed.values()) != list(range(expected_num_rois))
    ):
        raise ValueError("synthetic ROI mapping must use contiguous channel indices")
    return observed


def _required_target_validity_contract(protocol_profile: str) -> Optional[str]:
    """Bind supervised support semantics to a production protocol profile."""

    expected = TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE.get(protocol_profile)
    if expected is None:
        # Synthetic/unit profiles remain constructible, but production profiles
        # (the keys in PROTOCOL_TO_BENCHMARK_PROFILE) are always closed below.
        if protocol_profile in PROTOCOL_TO_BENCHMARK_PROFILE:
            raise RuntimeError(
                f"no target-validity contract is bound to {protocol_profile!r}"
            )
        return None
    executable = {
        "paper-a4-adni-fmriprep-v1": TARGET_VALIDITY_MASK_CONTRACT,
        "a4-native-recovery-v1": RECOVERY_TARGET_VALIDITY_MASK_CONTRACT,
    }.get(protocol_profile)
    if executable != expected:
        raise RuntimeError(
            "architecture and executable target-validity contracts differ for "
            f"{protocol_profile!r}"
        )
    return expected


def _require_batch_target_validity_contract(
    value: object,
    *,
    batch_size: int,
    protocol_profile: str,
    required_contract: Optional[str],
) -> tuple[str, ...]:
    """Validate one uniform, profile-bound supervised-support assertion."""

    if isinstance(value, str):
        value = [value]
    allowed = (
        {TARGET_VALIDITY_MASK_CONTRACT, RECOVERY_TARGET_VALIDITY_MASK_CONTRACT}
        if required_contract is None
        else {required_contract}
    )
    if (
        not isinstance(value, (list, tuple))
        or len(value) != batch_size
        or any(not isinstance(item, str) or item not in allowed for item in value)
        or len(set(value)) != 1
    ):
        raise ValueError(
            "target_validity_mask derivation contract is missing or differs "
            f"for protocol profile {protocol_profile!r}; expected one of "
            f"{sorted(allowed)!r}"
        )
    return tuple(value)


def validate_published_loss_weights(spec: Dict) -> None:
    """Reject production objectives that silently drift from published lambdas."""
    for name, expected in PAPER_LOSS_WEIGHTS.items():
        try:
            actual = float(spec.get(name, expected))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.loss.{name} must be numeric") from exc
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"paper-faithful training requires training.loss.{name}={expected}, "
                f"got {actual}"
            )


# --------------------------------------------------------------------------- #
# batch-tensor helpers (also used by training/inference for metrics)
# --------------------------------------------------------------------------- #
def roi_masks_to_tensor(
    roi_masks,
    target_spatial,
    device,
    *,
    expected_num_rois: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Normalise the dataset's ROI masks into a dense [B, R, D, H, W] tensor.

    The precomputed dataset yields ROI masks as a (collated) list of dicts
    ``{roi_id: mask[D,H,W]}``; this stacks them (sorted by id).  Spatial shape
    mismatches are rejected because online mask resampling would conceal a
    broken subject/preprocessing binding. Returns None if unavailable.
    """
    if roi_masks is None:
        return None
    if expected_num_rois is not None and (
        isinstance(expected_num_rois, bool) or expected_num_rois < 1
    ):
        raise ValueError("expected_num_rois must be a positive integer")
    if torch.is_tensor(roi_masks):
        if roi_masks.ndim == 4:
            t = roi_masks.unsqueeze(0)
        elif roi_masks.ndim == 5:
            t = roi_masks
        else:
            raise ValueError(
                "tensor ROI masks must be exactly [R,D,H,W] or [B,R,D,H,W], "
                f"got {roi_masks.shape}"
            )
    else:
        samples = roi_masks if isinstance(roi_masks, list) else [roi_masks]
        out = []
        observed_key_set = None
        for sample_index, d in enumerate(samples):
            if isinstance(d, Mapping):
                keys = list(d.keys())
                if not keys:
                    raise ValueError(f"ROI mask sample {sample_index} is empty")
                if any(
                    isinstance(key, bool) or not isinstance(key, int) for key in keys
                ):
                    raise ValueError("ROI mask keys must be canonical integer indices")
                canonical_keys = list(range(len(keys)))
                if sorted(keys) != canonical_keys:
                    raise ValueError(
                        "ROI mask keys must be the contiguous canonical set "
                        f"{canonical_keys}"
                    )
                if observed_key_set is None:
                    observed_key_set = tuple(canonical_keys)
                elif tuple(canonical_keys) != observed_key_set:
                    raise ValueError("ROI mask key sets differ between subjects")
                if expected_num_rois is not None and canonical_keys != list(
                    range(expected_num_rois)
                ):
                    raise ValueError(
                        f"ROI mask keys must be exactly 0..{expected_num_rois - 1}"
                    )
                vols = []
                for key in canonical_keys:
                    volume = torch.as_tensor(d[key]).float()
                    if volume.ndim != 3:
                        raise ValueError(
                            f"ROI mask {key!r} in sample {sample_index} must be "
                            f"exactly [D,H,W], got {volume.shape}"
                        )
                    vols.append(volume)
                stk = torch.stack(vols, dim=0)  # [R,D,H,W]
            else:
                stk = torch.as_tensor(d).float()
                if stk.ndim != 4:
                    raise ValueError(
                        f"ROI mask sample {sample_index} must be exactly "
                        f"[R,D,H,W], got {stk.shape}"
                    )
            out.append(stk)
        t = torch.stack(out, dim=0)
    t = t.to(device).float()  # [B,R,D,H,W]
    if t.dim() != 5:
        raise ValueError(f"ROI masks must resolve to [B,R,D,H,W], got {t.shape}")
    if tuple(t.shape[2:]) != tuple(target_spatial):
        raise ValueError(
            "ROI masks must already match the certified target grid; online "
            f"resampling is forbidden ({tuple(t.shape[2:])} != "
            f"{tuple(target_spatial)})"
        )
    if expected_num_rois is not None and t.shape[1] != expected_num_rois:
        raise ValueError(f"ROI masks must contain exactly {expected_num_rois} channels")
    if not torch.isfinite(t).all():
        raise ValueError("ROI masks contain NaN or infinity")
    if not bool(((t == 0) | (t == 1)).all()):
        raise ValueError("ROI masks must be exactly binary; thresholding is forbidden")
    if expected_num_rois is not None and bool((t.sum(dim=1) > 1).any()):
        raise ValueError("canonical ROI masks must be spatially disjoint")
    return t


class Connect4Model(nn.Module):
    def __init__(self, cfg: Dict, *, build_loss: bool = True):
        super().__init__()
        self.cfg = cfg

        d = cfg["data"]
        m = cfg["models"]
        u = m["unet"]
        protocol_profile = str(d.get("protocol_profile", "")).strip()
        self.protocol_profile = protocol_profile
        self._requires_canonical_roi_semantics = protocol_profile in (
            TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE
        )
        self.required_target_validity_mask_contract = (
            _required_target_validity_contract(protocol_profile)
        )
        if protocol_profile in PROTOCOL_TO_BENCHMARK_PROFILE:
            slab_selection = authenticate_configured_depth_slab_selection(cfg)
            self.depth_slab_size = int(slab_selection["depth_slab_size"])
        else:
            raw_slab_size = u.get("depth_slab_size", 0)
            if (
                isinstance(raw_slab_size, bool)
                or not isinstance(raw_slab_size, int)
                or raw_slab_size < 0
            ):
                raise ValueError(
                    "models.unet.depth_slab_size must be a non-negative integer"
                )
            self.depth_slab_size = raw_slab_size
        shape_value = d.get("architecture_shape")
        if shape_value is None:
            raise ValueError("data.architecture_shape is required")
        self.target_shape = tuple(shape_value)  # padded tensor grid
        self.num_frames = int(d["num_frames"])  # T
        self.out_channels = int(d.get("out_channels", 1))
        if self.out_channels != 1:
            raise ValueError(
                "paper-faithful CONNECT-4 requires one fMRI output channel"
            )
        self.num_rois = int(m["fusion"]["num_rois"])
        if self._requires_canonical_roi_semantics and self.num_rois != len(
            CANONICAL_ROI_SPECS
        ):
            raise ValueError(
                f"CONNECT-4 V18 requires exactly {len(CANONICAL_ROI_SPECS)} "
                "canonical ROI channels"
            )

        # ---- B) graph builders ------------------------------------------------
        gp = m["graphs"]
        self.image_graph_builder = ImageGraphBuilder(
            patch_size=tuple(gp["patch_size"]),
            k_neighbors=gp.get("k_neighbors", 5),
        )
        self.mask_graph_builder = MaskGraphBuilder(patch_size=tuple(gp["patch_size"]))
        self.roi_graph_builder = ROIGraphBuilder(num_rois=self.num_rois)
        num_patches = 1
        for s, p in zip(self.target_shape, gp["patch_size"]):
            num_patches *= s // p
        self.num_patches = num_patches
        self.hypergraph_builder = HypergraphBuilder(
            num_patches=num_patches,
            num_rois=self.num_rois,
        )

        # ---- B) multimodal hypergraph fusion ---------------------------------
        f = m["fusion"]
        self.fusion = MultiModalFusion(
            image_embed_dim=f["image_embed_dim"],
            mask_embed_dim=f["mask_embed_dim"],
            roi_embed_dim=f["roi_embed_dim"],
            hidden_dim=f["hidden_dim"],
            output_dim=f["output_dim"],
            num_attention_layers=f.get("num_attention_layers", 3),
            num_heads=f.get("num_heads", 8),
            dropout=f.get("dropout", 0.1),
            num_rois=self.num_rois,
        )

        # ---- C) DiT generation -----------------------------------------------
        dit = m["dit"]
        temporal_patch_size = dit.get("temporal_patch_size", 1)
        configured_patch_value_dim = (
            self.out_channels
            * temporal_patch_size
            * math.prod(tuple(dit["patch_size"]))
        )
        self._production_token_codec_shape: Optional[dict[str, int]] = None
        if self._requires_canonical_roi_semantics:
            self._production_token_codec_shape = require_production_token_codec_shape(
                protocol_profile=protocol_profile,
                hidden_size=dit.get("hidden_size", 512),
                patch_value_dim=configured_patch_value_dim,
            )
        self.dit = DiT4DTemporal(
            input_size=tuple(dit["input_size"]),  # full (D, H, W)
            in_channels=self.out_channels,
            patch_size=tuple(dit["patch_size"]),
            hidden_size=dit.get("hidden_size", 512),
            depth=dit.get("depth", 8),
            num_heads=dit.get("num_heads", 8),
            mlp_ratio=dit.get("mlp_ratio", 4.0),
            learn_sigma=False,
            input_dim=f["output_dim"],
            num_temporal_frames=self.num_frames,
            temporal_patch_size=temporal_patch_size,
            num_diffusion_steps=dit.get("num_diffusion_steps", 1_000),
            beta_schedule=dit.get("beta_schedule", "linear"),
            spatial_window_size=tuple(dit.get("spatial_window_size", (8, 8, 8))),
            diffusion_domain=dit.get("diffusion_domain"),
        )
        self.dit_spatial_size = tuple(dit["input_size"])
        if self.dit_spatial_size != self.target_shape:
            raise ValueError(
                "Figure-1 v18 requires token decoding on the complete configured "
                "D,H,W grid; low-resolution latent reconstruction is forbidden"
            )
        graph_grid = tuple(
            target // patch
            for target, patch in zip(self.target_shape, gp["patch_size"])
        )
        dit_grid = tuple(
            size // patch
            for size, patch in zip(self.dit_spatial_size, dit["patch_size"])
        )
        if dit_grid != graph_grid:
            raise ValueError(
                "DiT spatial tokens must align one-to-one with graph patches: "
                f"graph grid {graph_grid} differs from DiT grid {dit_grid}"
            )
        if self.dit.diffusion_domain != TOKEN_LATENT_DIFFUSION_DOMAIN:
            raise RuntimeError("raw-direct diffusion escaped v18 model admission")
        if (
            self.dit.token_latent_normalization_contract
            != TOKEN_LATENT_NORMALIZATION_CONTRACT
        ):
            raise RuntimeError(
                "untied or scale-collapsible token codec escaped v18 model admission"
            )
        if self.dit.epsilon_target_dim != self.dit.hidden_size:
            raise RuntimeError("epsilon target width differs from token width")
        if self.dit.denoiser_output_dim != self.dit.epsilon_target_dim:
            raise RuntimeError(
                "CONNECT-4 v18 requires an H-to-H epsilon head when learn_sigma=false"
            )
        self.ddim_steps = int(dit.get("num_inference_steps", 50))
        self.ddim_eta = float(dit.get("eta", 0.0))
        self.ddim_seed = int(cfg["training"].get("seed", 42))
        self._training_sampling_epoch: Optional[int] = None
        self.diffusion_weight = float(cfg["training"].get("diffusion_weight", 1.0))
        self.codec_reconstruction_weight = float(
            cfg["training"].get("token_codec_reconstruction_weight", 1.0)
        )
        if self.ddim_steps < 1:
            raise ValueError("models.dit.num_inference_steps must be positive")
        if self.ddim_eta != 0.0:
            raise ValueError(
                "paper-faithful target-blind synthesis requires deterministic "
                "DDIM eta=0"
            )
        if self.diffusion_weight <= 0:
            raise ValueError("training.diffusion_weight must be positive")
        if not math.isclose(
            self.codec_reconstruction_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "v18 training.token_codec_reconstruction_weight must be 1.0; "
                "this implementation-only codec safeguard does not change the "
                "published biological loss lambdas"
            )

        # Splitting T outside the decoder prevents a voxel from attending to all
        # 128 frames and therefore contradicts the temporal-UNet description.
        decoder_chunk = int(m["unet"].get("temporal_chunk", 0))
        if decoder_chunk:
            raise ValueError(
                "models.unet.temporal_chunk must be 0 for paper-faithful global "
                "temporal-channel attention"
            )

        # ---- D) temporal UNet decoder (TC-UNet: Conv4d temporal+graph cond,
        #         temporal attention, unchanged-grid 1x1x1 output head) ---------
        self.graph_cond_dim = f["output_dim"]
        base_channels = int(u.get("base_channels", 64))
        self.dit_to_unet = DiTToTCUNetProjection(
            self.out_channels,
            base_channels,
            self.target_shape,
        )
        self.decoder = TCUNet4DFiLM(
            in_channels=base_channels,
            out_channels=self.out_channels,
            base_channels=base_channels,
            num_levels=u.get("num_levels", 4),
            graph_cond_dim=self.graph_cond_dim,
            use_temporal_attn=u.get("use_temporal_attn", True),
            temporal_attn_heads=u.get("temporal_attn_heads", 4),
            use_checkpoint=u.get("use_checkpoint", False),
            input_spatial_size=self.target_shape,
        )
        detail_path_enabled = u.get("detail_path_enabled", False)
        if not isinstance(detail_path_enabled, bool):
            raise ValueError("models.unet.detail_path_enabled must be boolean")
        if detail_path_enabled:
            raise ValueError(
                "T1 high-pass detail injection is forbidden: apparent fMRI "
                "texture must be generated by the learned diffusion/TC-UNet path"
            )
        # Keep the public attribute for explicit compatibility inspection, but
        # never construct the historical T1 high-pass residual module.
        self.detail_path = None

        # ---- E) loss ----------------------------------------------------------
        self.loss_fn: Optional[Connect4Loss] = None
        self._brainlm_run_artifact_identity_sha256: Optional[str] = None
        self._brainlm_cohort_admission_identity_record_sha256: Optional[str] = None
        self._brainlm_dataset_state_sha256: Optional[str] = None
        self._brainlm_structural_artifact_identities_sha256: Optional[str] = None
        self._brainlm_native_alignment_authority_sha256: Optional[str] = None
        self._brainlm_single_case_development_scan_id: Optional[str] = None
        self._brainlm_single_case_dataset_state_record_sha256: Optional[str] = None
        if build_loss:
            lw = cfg["training"]["loss"]
            validate_published_loss_weights(lw)
            perceptual_weight = float(lw.get("perceptual", 0.1))
            perceptual_fe = build_pretrained_4d_extractor(
                lw.get("perceptual_extractor"),
                expected_name="brainlm",
                purpose="training perceptual loss",
                device=torch.device("cpu"),
                require_logits=False,
            )
            if perceptual_weight > 0 and perceptual_fe is None:
                raise ValueError(
                    "training.loss.perceptual_extractor must be enabled when the "
                    "paper's non-zero perceptual loss is configured"
                )
            self.loss_fn = Connect4Loss(
                ssim_weight=lw.get("ssim", 0.5),
                voxel_weight=lw.get("voxel", 1.0),
                volume_weight=lw.get("volume", 0.5),
                region_hist_weight=lw.get("region_hist", 0.5),
                perceptual_weight=perceptual_weight,
                fc_weight=lw.get("fc", 0.3),
                temporal_weight=lw.get("temporal", 0.2),
                perceptual_feature_extractor=perceptual_fe,
            )

    def validate_token_codec_invariants(
        self,
    ) -> Dict[str, bool | float | int | str]:
        """Expose V18 codec checks for focused diagnostics."""

        return self.dit.validate_token_codec_invariants()

    def production_token_codec_shape_evidence(self) -> dict[str, int]:
        """Return the exact production codec shape bound into checkpoints."""

        if self._production_token_codec_shape is None:
            raise RuntimeError(
                "non-production research models cannot claim V18 production "
                "token-codec shape evidence"
            )
        observed = require_production_token_codec_shape(
            protocol_profile=self.protocol_profile,
            hidden_size=self.dit.hidden_size,
            patch_value_dim=self.dit.patch_value_dim,
        )
        expected = {
            "hidden_size": PRODUCTION_TOKEN_CODEC_HIDDEN_SIZE,
            "patch_value_dim": PRODUCTION_TOKEN_CODEC_PATCH_VALUE_DIM,
            "compression_ratio": PRODUCTION_TOKEN_CODEC_COMPRESSION_RATIO,
        }
        if observed != expected or self._production_token_codec_shape != expected:
            raise RuntimeError("V18 production token-codec shape evidence differs")
        return dict(expected)

    def validate_model_state_invariants(
        self,
    ) -> Dict[str, bool | float | int | str]:
        """Validate recursive legacy-state, codec, shape, and finiteness gates."""

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
        codec_report = self.dit.validate_token_codec_invariants()
        named_state = {
            **dict(self.named_parameters(recurse=True)),
            **dict(self.named_buffers(recurse=True)),
        }
        tensor_count, value_count = _validate_checkpoint_state_tree(
            named_state,
            label="V18 CONNECT-4 runtime model state",
            allowed_codec_state_paths=frozenset(
                f"dit.{component}" for component in _TOKEN_CODEC_STATE_COMPONENTS
            ),
        )
        if self._requires_canonical_roi_semantics:
            self.production_token_codec_shape_evidence()
        return {
            **codec_report,
            "model_state_finite": True,
            "model_state_floating_tensor_count": tensor_count,
            "model_state_floating_value_count": value_count,
            "production_token_codec_shape_bound": (
                self._requires_canonical_roi_semantics
            ),
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Reject hostile/non-finite state before and after every model load."""

        _preflight_token_codec_checkpoint_state(
            state_dict,
            label="V18 CONNECT-4 checkpoint state",
            codec_prefix="dit.",
            audit_prefix="",
            expected_raw_detail_shape=(
                self.dit.hidden_size - 1,
                self.dit.patch_value_dim - 1,
            ),
            raw_detail_validator=self.dit._validated_raw_detail_rows,
        )
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        self.validate_model_state_invariants()
        return result

    def bind_training_authorities(
        self,
        *,
        artifact_identity: Mapping[str, Any],
        cohort_admission_identity: Mapping[str, Any],
    ) -> None:
        """Bind contextual BrainLM calls to this admitted run/checkpoint root."""

        if self.loss_fn is None:
            raise RuntimeError("BrainLM training authorities require enabled losses")
        extractor = self.loss_fn.perceptual.fe
        if not bool(getattr(extractor, "connect4_requires_context", False)):
            raise RuntimeError("paper perceptual training requires contextual BrainLM")
        recorded_brainlm = validate_configured_brainlm_identity(
            artifact_identity.get("perceptual_extractor", {})
        )
        if recorded_brainlm != getattr(extractor, "connect4_artifact_identity", None):
            raise RuntimeError(
                "constructed BrainLM differs from the run-artifact identity"
            )
        admission = validate_cohort_admission_identity(cohort_admission_identity)
        artifact_sha = canonical_sha256(artifact_identity)
        structural_sha = artifact_identity.get("structural_artifact_identities_sha256")
        native_authority_sha = artifact_identity.get(
            "native_alignment_authority_sha256"
        )
        if (
            admission["artifact_identity_sha256"] != artifact_sha
            or admission["structural_artifact_identities_sha256"] != structural_sha
            or not isinstance(structural_sha, str)
            or len(structural_sha) != 64
            or not isinstance(native_authority_sha, str)
            or len(native_authority_sha) != 64
        ):
            raise RuntimeError("BrainLM run/cohort structural authority differs")
        self._brainlm_run_artifact_identity_sha256 = artifact_sha
        self._brainlm_cohort_admission_identity_record_sha256 = admission[
            "record_sha256"
        ]
        self._brainlm_dataset_state_sha256 = admission["dataset_state_sha256"]
        self._brainlm_structural_artifact_identities_sha256 = structural_sha
        self._brainlm_native_alignment_authority_sha256 = native_authority_sha

    def bind_single_case_development_brainlm_authority(
        self,
        binding: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Bind one authenticated development scan to the unchanged BrainLM loss.

        The production binder above remains the only route for production
        training.  This additive route is intentionally limited to the named
        B347 development reconstruction and preserves explicit non-production
        and no-generalisation labels in its signed binding.
        """

        if self.protocol_profile != SINGLE_CASE_DEVELOPMENT_PROTOCOL_PROFILE:
            raise RuntimeError(
                "single-case BrainLM binding requires the dedicated proof profile"
            )
        if self.loss_fn is None:
            raise RuntimeError("single-case BrainLM binding requires enabled losses")
        if any(
            value is not None
            for value in (
                self._brainlm_run_artifact_identity_sha256,
                self._brainlm_cohort_admission_identity_record_sha256,
                self._brainlm_dataset_state_sha256,
                self._brainlm_structural_artifact_identities_sha256,
                self._brainlm_native_alignment_authority_sha256,
                self._brainlm_single_case_development_scan_id,
                self._brainlm_single_case_dataset_state_record_sha256,
            )
        ):
            raise RuntimeError("BrainLM authorities were already bound")
        extractor = self.loss_fn.perceptual.fe
        if not bool(getattr(extractor, "connect4_requires_context", False)):
            raise RuntimeError("paper perceptual training requires contextual BrainLM")
        artifact_identity = validate_configured_brainlm_identity(
            getattr(extractor, "connect4_artifact_identity", None)
        )
        expected_identity_sha256 = artifact_identity["record_sha256"]
        validated = validate_single_case_development_brainlm_binding(
            binding,
            expected_brainlm_perceptual_identity_sha256=(expected_identity_sha256),
        )
        self._brainlm_run_artifact_identity_sha256 = validated[
            "run_artifact_identity_sha256"
        ]
        self._brainlm_cohort_admission_identity_record_sha256 = validated[
            "proof_admission_identity_record_sha256"
        ]
        self._brainlm_dataset_state_sha256 = validated["dataset_state_sha256"]
        self._brainlm_structural_artifact_identities_sha256 = validated[
            "structural_artifact_identities_sha256"
        ]
        self._brainlm_native_alignment_authority_sha256 = validated[
            "native_alignment_authority_sha256"
        ]
        self._brainlm_single_case_development_scan_id = validated["scan_id"]
        self._brainlm_single_case_dataset_state_record_sha256 = validated[
            "dataset_state_record_sha256"
        ]
        return validated

    @staticmethod
    def _brainlm_string_batch(
        batch: Mapping[str, Any], key: str, batch_size: int
    ) -> list[str]:
        value = batch.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise RuntimeError(f"BrainLM admitted batch field {key} differs")
        result = [str(item) for item in value]
        if any(not item for item in result):
            raise RuntimeError(f"BrainLM admitted batch field {key} is empty")
        return result

    def build_brainlm_batch_context(
        self,
        batch: Mapping[str, Any],
        brain_support: torch.Tensor,
    ) -> dict[str, Any]:
        """Build the sole loss context from a rank-zero admitted real batch."""

        if self.loss_fn is None:
            raise RuntimeError("BrainLM batch context requires enabled losses")
        extractor = self.loss_fn.perceptual.fe
        if not bool(getattr(extractor, "connect4_requires_context", False)):
            raise RuntimeError(
                "configured perceptual extractor is not contextual BrainLM"
            )
        required_roots = (
            self._brainlm_run_artifact_identity_sha256,
            self._brainlm_cohort_admission_identity_record_sha256,
            self._brainlm_dataset_state_sha256,
            self._brainlm_structural_artifact_identities_sha256,
            self._brainlm_native_alignment_authority_sha256,
        )
        if any(value is None for value in required_roots):
            raise RuntimeError("BrainLM training authorities were not bound")
        if not torch.is_tensor(brain_support) or brain_support.ndim != 5:
            raise RuntimeError("BrainLM brain support must be [B,1,D,H,W]")
        batch_size = int(brain_support.shape[0])
        scan_ids = self._brainlm_string_batch(batch, "scan_id", batch_size)
        roles = self._brainlm_string_batch(batch, "brainlm_role", batch_size)
        scan_roles = self._brainlm_string_batch(batch, "scan_role", batch_size)
        if roles != scan_roles or any(
            role not in {"train", "development-validation"} for role in roles
        ):
            raise RuntimeError("BrainLM batch role differs from admitted scan role")
        if self._brainlm_single_case_development_scan_id is not None and (
            scan_ids != [self._brainlm_single_case_development_scan_id]
            or roles != ["development-validation"]
        ):
            raise RuntimeError(
                "single-case BrainLM context escaped its B347 development binding"
            )
        context_schemas = self._brainlm_string_batch(
            batch, "brainlm_context_schema", batch_size
        )
        if any(
            schema != BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA for schema in context_schemas
        ):
            raise RuntimeError("BrainLM admitted scan-context schema differs")
        dataset_state_sha = self._brainlm_string_batch(
            batch, "brainlm_dataset_state_sha256", batch_size
        )
        dataset_state_record_sha = self._brainlm_string_batch(
            batch, "brainlm_dataset_state_record_sha256", batch_size
        )
        if (
            any(
                value != self._brainlm_dataset_state_sha256
                for value in dataset_state_sha
            )
            or len(set(dataset_state_record_sha)) != 1
        ):
            raise RuntimeError("BrainLM batch dataset-state root differs")
        if (
            self._brainlm_single_case_dataset_state_record_sha256 is not None
            and dataset_state_record_sha
            != [self._brainlm_single_case_dataset_state_record_sha256]
        ):
            raise RuntimeError(
                "single-case BrainLM context uses another dataset-state record"
            )
        native_sources = self._brainlm_string_batch(
            batch, "brainlm_native_preprocessing_source_sha256", batch_size
        )
        if any(
            value != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            for value in native_sources
        ):
            raise RuntimeError("BrainLM batch native preprocessing source differs")
        native_authorities = self._brainlm_string_batch(
            batch, "brainlm_native_alignment_authority_sha256", batch_size
        )
        if any(
            value != self._brainlm_native_alignment_authority_sha256
            for value in native_authorities
        ):
            raise RuntimeError("BrainLM batch native alignment authority differs")
        tensor_fields = {
            "source_affine": batch.get("t1w_affine"),
            "admitted_padded_affine": batch.get("brainlm_padded_affine_ras_mm"),
            "padded_shape": batch.get("brainlm_padded_shape"),
            "crop_before": batch.get("brainlm_padding_before"),
            "crop_after": batch.get("brainlm_padding_after"),
            "native_shape": batch.get("brainlm_native_shape"),
        }
        expected_shapes = {
            "source_affine": (batch_size, 4, 4),
            "admitted_padded_affine": (batch_size, 4, 4),
            "padded_shape": (batch_size, 3),
            "crop_before": (batch_size, 3),
            "crop_after": (batch_size, 3),
            "native_shape": (batch_size, 3),
        }
        if any(
            not torch.is_tensor(value) or tuple(value.shape) != expected_shapes[name]
            for name, value in tensor_fields.items()
        ):
            raise RuntimeError("BrainLM admitted affine/grid/crop tensor differs")
        support_counts = batch.get("brainlm_support_foreground_voxels")
        if (
            not torch.is_tensor(support_counts)
            or tuple(support_counts.shape) != (batch_size,)
            or not torch.equal(
                support_counts.detach().cpu().long(),
                (brain_support.detach().cpu() > 0.5).flatten(1).sum(dim=1).long(),
            )
        ):
            raise RuntimeError("BrainLM batch structural-support count differs")
        authority = extractor.authority_identity
        brainlm_identity = extractor.connect4_artifact_identity
        return {
            "schema": BRAINLM_CONTEXT_SCHEMA,
            "brainlm_perceptual_identity_sha256": brainlm_identity["record_sha256"],
            "authority_file_sha256": authority["file_sha256"],
            "authority_content_record_sha256": brainlm_identity["mni_authority"][
                "content_record_sha256"
            ],
            "authority_record_sha256": authority["authority_record_sha256"],
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "run_artifact_identity_sha256": (
                self._brainlm_run_artifact_identity_sha256
            ),
            "cohort_admission_identity_record_sha256": (
                self._brainlm_cohort_admission_identity_record_sha256
            ),
            "dataset_state_record_sha256": dataset_state_record_sha[0],
            "dataset_state_sha256": self._brainlm_dataset_state_sha256,
            "structural_artifact_identities_sha256": (
                self._brainlm_structural_artifact_identities_sha256
            ),
            "scan_ids": scan_ids,
            "roles": roles,
            "prepared_t1_sha256": self._brainlm_string_batch(
                batch, "brainlm_prepared_t1_sha256", batch_size
            ),
            "prepared_mask_sha256": self._brainlm_string_batch(
                batch, "brainlm_prepared_mask_sha256", batch_size
            ),
            "padded_t1_artifact_descriptor_sha256": self._brainlm_string_batch(
                batch,
                "brainlm_padded_t1_artifact_descriptor_sha256",
                batch_size,
            ),
            "padded_mask_artifact_descriptor_sha256": self._brainlm_string_batch(
                batch,
                "brainlm_padded_mask_artifact_descriptor_sha256",
                batch_size,
            ),
            "cache_metadata_artifact_descriptor_sha256": self._brainlm_string_batch(
                batch,
                "brainlm_cache_metadata_artifact_descriptor_sha256",
                batch_size,
            ),
            "structural_source_identity_sha256": self._brainlm_string_batch(
                batch,
                "brainlm_structural_source_identity_sha256",
                batch_size,
            ),
            "native_alignment_authority_sha256": native_authorities,
            "target_artifact_identity_sha256": self._brainlm_string_batch(
                batch,
                "brainlm_target_artifact_identity_sha256",
                batch_size,
            ),
            "support_tensor_sha256": self._brainlm_string_batch(
                batch, "brainlm_support_tensor_sha256", batch_size
            ),
            "support_foreground_voxels": support_counts,
            "admitted_scan_context_record_sha256": self._brainlm_string_batch(
                batch, "brainlm_scan_context_record_sha256", batch_size
            ),
            "admitted_scan_context_schema": context_schemas,
            "source_affine": tensor_fields["source_affine"],
            "admitted_padded_affine": tensor_fields["admitted_padded_affine"],
            "brain_support": brain_support,
            "padded_shape": tensor_fields["padded_shape"],
            "crop_before": tensor_fields["crop_before"],
            "crop_after": tensor_fields["crop_after"],
            "native_shape": tensor_fields["native_shape"],
        }

    def preflight_brainlm_batch_context_identity(
        self,
        batch: Mapping[str, Any],
        brain_support: torch.Tensor,
    ) -> dict[str, Any]:
        """Authenticate the actual batch context before a memory-sweep attempt."""

        context = self.build_brainlm_batch_context(batch, brain_support)
        extractor = self.loss_fn.perceptual.fe if self.loss_fn is not None else None
        prepare = getattr(extractor, "prepare_context_identity", None)
        if not callable(prepare):
            raise RuntimeError("contextual BrainLM has no preflight validator")
        return prepare(
            context,
            device=brain_support.device,
            batch=int(brain_support.shape[0]),
        )

    # ----------------------------------------------------------------------- #
    # B) build the three graphs + hypergraph and fuse them into patch tokens
    # ----------------------------------------------------------------------- #
    def build_patch_tokens(self, batch: Dict, device: torch.device) -> torch.Tensor:
        image_nodes = batch["image_nodes"].to(device)  # [B, P, image_embed_dim]
        mask_nodes = batch["mask_nodes"].to(device)  # [B, P, mask_embed_dim]
        roi_nodes = batch["roi_nodes"].to(device)  # [B, R, roi_embed_dim]
        B = image_nodes.shape[0]
        dwi = batch["dwi_matrix"]
        s2r = (
            _require_canonical_roi_mapping(batch["structure_to_roi_idx"])
            if self._requires_canonical_roi_semantics
            else _require_contiguous_test_roi_mapping(
                batch["structure_to_roi_idx"],
                expected_num_rois=self.num_rois,
            )
        )
        dists = batch["patch_distributions"]

        patch_positions = self.image_graph_builder.get_patch_positions(
            self.target_shape
        ).to(device)
        image_adj = (
            self.image_graph_builder.build_adjacency(patch_positions, device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        fused = []
        for b in range(B):
            dwi_b = dwi[b] if dwi.ndim == 3 else dwi
            dist_b = dists[b] if isinstance(dists, list) else dists
            mask_adj = self.mask_graph_builder.build_adjacency_from_dwi(
                mask_nodes.shape[1], dist_b, dwi_b, s2r
            ).unsqueeze(0)
            _, roi_adj = self.roi_graph_builder(roi_nodes[b : b + 1], dwi_matrix=dwi_b)
            if roi_adj.ndim == 2:
                roi_adj = roi_adj.unsqueeze(0)
            he_index, he_w = self.hypergraph_builder(dist_b, s2r)
            he_index = he_index.to(device)
            he_w = he_w.to(device)
            patch_ids = self.hypergraph_builder.patch_ids_from_index(he_index)
            fused.append(
                self.fusion(
                    image_nodes[b : b + 1],
                    image_adj[b : b + 1],
                    mask_nodes[b : b + 1],
                    mask_adj,
                    roi_nodes[b : b + 1],
                    roi_adj,
                    he_index,
                    he_w,
                    patch_ids,
                    patch_positions=patch_positions,
                )
            )
        return torch.cat(fused, dim=0)  # [B, P, output_dim]

    # ----------------------------------------------------------------------- #
    # C+D) conditional DDIM followed by one full-temporal UNet pass
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _normalise_t1(
        t1: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        t1 = t1.to(device=device, dtype=torch.float32)
        if t1.ndim != 5 or t1.shape[0] != batch_size or t1.shape[1] != 1:
            raise ValueError(f"T1 must be exactly [B,1,D,H,W], got {t1.shape}")
        if not torch.isfinite(t1).all():
            raise ValueError("T1 conditioning contains NaN or infinity")
        return t1

    @staticmethod
    def _normalise_brain_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            raise ValueError(
                "an explicit authenticated structural brain mask is required"
            )
        mask = torch.as_tensor(mask, device=device)
        if mask.ndim != 5 or mask.shape[0] != batch_size or mask.shape[1] != 1:
            raise ValueError(
                f"brain mask must be exactly [B,1,D,H,W], got {mask.shape}"
            )
        if not torch.isfinite(mask).all():
            raise ValueError("brain mask contains NaN or infinity")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("brain mask must be exactly binary")
        if not bool((mask > 0.5).flatten(1).any(dim=1).all()):
            raise ValueError("brain mask is empty for at least one subject")
        return mask.float()

    @staticmethod
    def _normalise_target_validity_mask(
        mask: Optional[torch.Tensor],
        target: torch.Tensor,
        brain_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Validate the exact target-observed support used for supervision.

        Structural ``brain_mask`` remains target-independent and controls
        conditioning and decoded output.  This second mask is permitted only
        when a paired target is already in memory and must equal that target's
        exact non-zero-over-time support; callers cannot choose an easier loss
        subset after seeing prediction error.
        """

        if mask is None:
            raise ValueError("paired fMRI supervision requires target_validity_mask")
        value = torch.as_tensor(mask, device=target.device)
        expected_shape = (
            target.shape[0],
            1,
            *target.shape[-3:],
        )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "target_validity_mask must be exactly [B,1,D,H,W], got "
                f"{tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("target_validity_mask contains NaN or infinity")
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("target_validity_mask must be exactly binary")
        value = value.float()
        if not bool(value.flatten(1).any(dim=1).all()):
            raise ValueError("target_validity_mask is empty for at least one subject")
        if bool((value.bool() & ~brain_mask.bool()).any()):
            raise ValueError(
                "target_validity_mask lies outside the structural brain_mask"
            )
        observed = target.ne(0).any(dim=(1, 2), keepdim=False).unsqueeze(1)
        if not torch.equal(value.bool(), observed):
            raise ValueError(
                "target_validity_mask differs from exact observed target support"
            )
        return value

    @staticmethod
    def _validate_roi_support_within_brain_mask(
        roi_masks: torch.Tensor,
        brain_mask: torch.Tensor,
    ) -> None:
        if roi_masks.ndim != 5 or brain_mask.ndim != 5:
            raise ValueError(
                "ROI and brain masks must be dense five-dimensional tensors"
            )
        roi_support = roi_masks.gt(0.5).any(dim=1, keepdim=True)
        outside = roi_support & brain_mask.eq(0)
        if bool(outside.any()):
            raise ValueError(
                "ROI foreground lies outside the authenticated structural brain_mask"
            )

    def _decode_latent(
        self,
        latent: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode x0 tokens once, then run the UNet while preserving full T."""
        return self._decode_volume(
            self.dit.decode_tokens(latent),
            patch_tokens,
            mask,
        )

    def _decode_volume(
        self,
        volume: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the temporal UNet on one already decoded V18 x0 volume."""
        batch_size, _, time, _, _, _ = volume.shape
        graph_cond = patch_tokens.mean(dim=1)
        temporal_indices = (
            torch.arange(time, device=volume.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        decoded = self.decoder.decode_dit_volume(
            dit_volume=volume,
            projection=self.dit_to_unet,
            graph_cond=graph_cond,
            temporal_indices=temporal_indices,
            mask=mask,
            depth_slab_size=self.depth_slab_size,
        ).unsqueeze(1)
        if mask is not None:
            decoded = decoded * mask.unsqueeze(2).to(decoded.dtype)
        expected = (batch_size, 1, self.num_frames, *self.target_shape)
        if tuple(decoded.shape) != expected:
            raise RuntimeError(
                "temporal UNet did not produce the configured full fMRI volume: "
                f"expected {expected}, got {tuple(decoded.shape)}"
            )
        return decoded

    def set_training_sampling_epoch(self, epoch: int) -> None:
        """Bind reproducible diagnostic full-DDIM noise to one training epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("training sampling epoch must be a non-negative integer")
        self._training_sampling_epoch = epoch

    @staticmethod
    def _validated_scan_ids(scan_ids: object, batch_size: int) -> tuple[str, ...]:
        if not isinstance(scan_ids, (list, tuple)) or len(scan_ids) != batch_size:
            raise ValueError(
                f"target-blind DDIM sampling requires {batch_size} scan IDs"
            )
        result = tuple(str(value).strip() for value in scan_ids)
        if any(not value for value in result):
            raise ValueError("target-blind DDIM scan IDs must be non-empty")
        return result

    def _deterministic_ddim_noise(
        self,
        scan_ids: Sequence[str],
        shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        sampling_context: str,
    ) -> torch.Tensor:
        """Return order-independent Gaussian noise bound to scan and context.

        A diagnostic full-DDIM call made in training mode uses a distinct
        context for every epoch. Development and external inference use the
        fixed ``evaluation`` context, so the same subject/checkpoint is
        reproducible without sharing one seed across all subjects. Optimization
        itself uses the scheduler's randomly noised differentiable x0 path.
        """
        shape = tuple(int(value) for value in shape)
        expected = (
            len(scan_ids),
            self.dit.temporal_grid * self.dit.num_patches,
            self.dit.hidden_size,
        )
        if shape != expected:
            raise ValueError(f"DDIM noise shape must be token latent {expected}")
        context = str(sampling_context).strip()
        if not context:
            raise ValueError("DDIM sampling context must be non-empty")
        samples = []
        for scan_id in scan_ids:
            payload = (
                f"{_DDIM_NOISE_DOMAIN}\0{self.ddim_seed}\0{context}\0{scan_id}"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            seed &= (1 << 63) - 1
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            samples.append(
                torch.randn(shape[1:], generator=generator, dtype=torch.float32)
            )
        return torch.stack(samples).to(device=device, dtype=dtype)

    def _sample_latent(
        self,
        patch_tokens: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self.dit.sample(
            patch_tokens,
            shape=tuple(initial_noise.shape),
            num_inference_steps=(
                self.ddim_steps
                if num_inference_steps is None
                else int(num_inference_steps)
            ),
            eta=self.ddim_eta,
            initial_noise=initial_noise,
        )

    def sampled_ddim_decode(
        self,
        batch: Dict,
        *,
        sampling_context: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode the exact target-blind full-DDIM inference distribution.

        Only structural fields are copied into the sampler input. Conditioning
        and the iterative denoiser run in evaluation mode without gradients.
        This routine is the inference-distribution path; training uses the
        explicitly documented differentiable one-step x0 surrogate instead of
        pretending to backpropagate through this complete DDIM trajectory.
        """
        missing = [key for key in _TARGET_BLIND_SAMPLER_KEYS if key not in batch]
        if missing:
            raise ValueError(f"target-blind DDIM batch is missing fields: {missing}")
        structural_batch = {key: batch[key] for key in _TARGET_BLIND_SAMPLER_KEYS}
        batch_size = int(torch.as_tensor(structural_batch["image_nodes"]).shape[0])
        device = next(self.parameters()).device
        scan_ids = self._validated_scan_ids(structural_batch["scan_id"], batch_size)
        if sampling_context is None:
            if self.training:
                if self._training_sampling_epoch is None:
                    raise RuntimeError(
                        "set_training_sampling_epoch(epoch) is required before "
                        "decoder training"
                    )
                sampling_context = f"train-epoch-{self._training_sampling_epoch:08d}"
            else:
                sampling_context = "evaluation"

        module_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            with torch.no_grad():
                patch_tokens = self.build_patch_tokens(structural_batch, device)
                mask = self._normalise_brain_mask(
                    structural_batch["brain_mask"], batch_size, device
                )
                roi_masks = roi_masks_to_tensor(
                    structural_batch["roi_masks"],
                    self.target_shape,
                    device,
                    expected_num_rois=self.num_rois,
                )
                if roi_masks is None or roi_masks.shape[1] != self.num_rois:
                    raise ValueError(
                        f"target-blind sampling requires {self.num_rois} ROI masks"
                    )
                self._validate_roi_support_within_brain_mask(roi_masks, mask)
                latent_shape = (
                    batch_size,
                    self.dit.temporal_grid * self.dit.num_patches,
                    self.dit.hidden_size,
                )
                noise = self._deterministic_ddim_noise(
                    scan_ids,
                    latent_shape,
                    device=patch_tokens.device,
                    dtype=patch_tokens.dtype,
                    sampling_context=sampling_context,
                )
                latent = self._sample_latent(
                    patch_tokens,
                    initial_noise=noise,
                )
                schedule = self.dit.scheduler.inference_timesteps(
                    self.ddim_steps, patch_tokens.device
                )
        finally:
            # Restore every module individually: frozen feature modules may have
            # intentionally remained in eval mode before this method was called.
            for module, was_training in module_modes:
                module.training = was_training

        prediction = self._decode_latent(latent.detach(), patch_tokens.detach(), mask)
        return {
            "prediction": prediction,
            "sampled_latent": latent.detach(),
            "sampling_noise": noise.detach(),
            "sampling_schedule": schedule.detach(),
        }

    def generate(
        self,
        patch_tokens: torch.Tensor,
        t1: torch.Tensor,
        mask: torch.Tensor,
        *,
        num_inference_steps: Optional[int] = None,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        scan_ids: Optional[Sequence[str]] = None,
        sampling_context: str = "evaluation",
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("generate() requires evaluation mode")
        B = patch_tokens.shape[0]
        device = patch_tokens.device
        self._normalise_t1(t1, B, device)
        mask = self._normalise_brain_mask(mask, B, device)
        if generator is not None:
            raise ValueError(
                "generator-based DDIM noise is forbidden; use scan-bound noise"
            )
        requested_eta = self.ddim_eta if eta is None else float(eta)
        if requested_eta != 0.0:
            raise ValueError("target-blind synthesis requires DDIM eta=0")
        latent_shape = (
            B,
            self.dit.temporal_grid * self.dit.num_patches,
            self.dit.hidden_size,
        )
        ids = self._validated_scan_ids(scan_ids, B)
        initial_noise = self._deterministic_ddim_noise(
            ids,
            latent_shape,
            device=device,
            dtype=patch_tokens.dtype,
            sampling_context=sampling_context,
        )
        latent = self._sample_latent(
            patch_tokens,
            initial_noise=initial_noise,
            num_inference_steps=num_inference_steps,
        )
        return self._decode_latent(latent, patch_tokens, mask)

    def forward(self, batch: Dict) -> Dict:
        device = next(self.parameters()).device
        brain_mask = batch.get("brain_mask")
        if brain_mask is None:
            raise ValueError(
                "paper-faithful execution requires an explicit authenticated "
                "structural brain_mask; an ROI union is not an admissible substitute"
            )
        brain_mask = self._normalise_brain_mask(
            brain_mask, batch["image_nodes"].shape[0], device
        )
        roi_masks = roi_masks_to_tensor(
            batch.get("roi_masks"),
            self.target_shape,
            device,
            expected_num_rois=self.num_rois,
        )
        if roi_masks is None or roi_masks.shape[1] != self.num_rois:
            shape = None if roi_masks is None else tuple(roi_masks.shape)
            raise ValueError(
                f"paper-faithful execution requires {self.num_rois} ROI masks "
                f"per subject, got {shape}"
            )
        self._validate_roi_support_within_brain_mask(roi_masks, brain_mask)

        batch_size = int(batch["image_nodes"].shape[0])
        patch_tokens = self.build_patch_tokens(batch, device) if self.training else None
        target = batch.get("fmri")
        target_validity_mask = None
        if target is not None:
            target = target.to(device)
            if target.ndim == 5:
                target = target.unsqueeze(1)
            if target.ndim != 6:
                raise ValueError(
                    f"fMRI target must resolve to [B,1,T,D,H,W], got {target.shape}"
                )
            expected_target = (
                batch_size,
                self.out_channels,
                self.num_frames,
                *self.target_shape,
            )
            if tuple(target.shape) != expected_target:
                raise ValueError(
                    "fMRI targets must already satisfy the paper harmonisation "
                    f"contract {expected_target}; got {tuple(target.shape)}"
                )
            if not torch.isfinite(target).all():
                raise ValueError("fMRI target contains NaN or infinity")
            target_validity_mask = self._normalise_target_validity_mask(
                batch.get("target_validity_mask"), target, brain_mask
            )
            _require_batch_target_validity_contract(
                batch.get("target_validity_mask_contract"),
                batch_size=batch_size,
                protocol_profile=self.protocol_profile,
                required_contract=self.required_target_validity_mask_contract,
            )
        elif (
            "target_validity_mask" in batch or "target_validity_mask_contract" in batch
        ):
            raise ValueError(
                "target_validity_mask is forbidden without a paired fMRI target"
            )

        supervision_roi_masks = (
            None if target_validity_mask is None else roi_masks * target_validity_mask
        )
        if supervision_roi_masks is not None and not bool(
            supervision_roi_masks.flatten(2).any(dim=2).all()
        ):
            raise ValueError(
                "every configured ROI must have non-empty target_validity_mask "
                "support for every subject"
            )

        diffusion_loss = None
        codec_reconstruction_loss = None
        biological_noise_weight = None
        if self.training:
            if target is None:
                raise ValueError("diffusion training requires a clean fMRI target")
            if patch_tokens is None:
                raise RuntimeError("training patch-token construction was skipped")
            diffusion = self.dit.diffusion_loss(
                target,
                patch_tokens,
                target_validity_mask=target_validity_mask,
            )
            diffusion_loss = diffusion["loss"]
            codec_reconstruction_loss = diffusion["codec_reconstruction_loss"]
            # Recovery-training formulation: decode the differentiable x0
            # estimate from one randomly noised diffusion state. This is the
            # standard memory-bounded denoising surrogate; it is not represented
            # as backpropagation through the 50-step inference trajectory. It
            # lets every composite loss update the DiT, structural fusion, and
            # decoder while full DDIM remains target-blind at evaluation.
            pred = self._decode_volume(
                diffusion["pred_original_volume"],
                patch_tokens,
                brain_mask,
            )
            alpha = self.dit.scheduler.alphas_cumprod.index_select(
                0, diffusion["timesteps"].long()
            )
            # x0 sensitivity to epsilon grows as 1/sqrt(alpha). Multiplying the
            # biological objective by alpha keeps its gradient bounded at very
            # noisy timesteps without changing the published component lambdas.
            biological_noise_weight = alpha.mean().to(
                device=pred.device, dtype=torch.float32
            )
        else:
            sampled = self.sampled_ddim_decode(batch, sampling_context="evaluation")
            pred = sampled["prediction"]

        out = {"prediction": pred}
        if target is not None:
            if self.loss_fn is None:
                raise RuntimeError(
                    "losses were disabled for this model; call generate() for inference"
                )
            perceptual_extractor = self.loss_fn.perceptual.fe
            brainlm_context = (
                self.build_brainlm_batch_context(batch, brain_mask)
                if bool(
                    getattr(perceptual_extractor, "connect4_requires_context", False)
                )
                else None
            )
            losses = self.loss_fn(
                pred,
                target,
                mask=target_validity_mask,
                roi_masks=supervision_roi_masks,
                brainlm_context=brainlm_context,
            )
            if brainlm_context is not None:
                context_identity = self.loss_fn.perceptual.last_context_identity
                if not isinstance(context_identity, Mapping):
                    raise RuntimeError(
                        "BrainLM loss did not expose its authenticated context identity"
                    )
                out["brainlm_batch_context_identity"] = dict(context_identity)
            biological_total = losses["total"]
            if diffusion_loss is None:
                diffusion_loss = biological_total.new_zeros(())
            if codec_reconstruction_loss is None:
                codec_reconstruction_loss = biological_total.new_zeros(())
            if biological_noise_weight is None:
                biological_noise_weight = biological_total.new_ones(())
            losses["biological_total"] = biological_total
            losses["biological_noise_weight"] = biological_noise_weight
            losses["weighted_biological_total"] = (
                biological_noise_weight * biological_total
            )
            losses["diffusion"] = diffusion_loss
            losses["token_codec_reconstruction"] = codec_reconstruction_loss
            losses["total"] = (
                losses["weighted_biological_total"]
                + self.diffusion_weight * diffusion_loss
                + self.codec_reconstruction_weight * codec_reconstruction_loss
            )
            out["losses"] = losses
            out["target"] = target
            out["target_validity_mask"] = target_validity_mask
            out["supervision_roi_masks"] = supervision_roi_masks
        out["roi_masks"] = roi_masks
        out["brain_mask"] = brain_mask
        return out
