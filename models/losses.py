"""
Loss components for CONNECT-4 (Figure 1E).

The six terms drawn in Figure 1E plus the explicit temporal-coherence term in
the method text and implementation details:

    1. SSIM3DLoss              - 3D SSIM Loss              (structural comparison)
    2. VoxelIntensityLoss      - Voxel Intensity Loss      (per-voxel L1)
    3. VolumeLoss              - Volume Loss               (per-ROI integrated signal)
    4. RegionHistogramLoss     - Region Histogram Matching (per-ROI intensity distribution)
    5. TemporalCoherenceLoss   - first-order BOLD-dynamics fidelity
    6. PerceptualLoss          - Perceptual Loss, 4D       (BrainLM features)
    7. FCMatrixLoss            - Functional Connectivity Matrix Loss (ROI x ROI corr.)

All tensors are full 4D volumes [B, C, T, D, H, W] unless stated otherwise.
The combined objective `Connect4Loss` returns every term plus a weighted total.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from architecture_contract import FRAMEWISE_SSIM_CONTRACT

from .feature_extractors import split_feature_output


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, D, H, W] -> [B, C, D, H, W] (mean over time)."""
    return x.mean(dim=2)


def _require_all_roi_support(roi_masks: torch.Tensor) -> torch.Tensor:
    """Validate that every configured ROI exists for every subject."""
    if roi_masks.ndim != 5 or roi_masks.shape[1] < 1:
        raise ValueError("ROI masks must be [B,R,D,H,W] with at least one ROI")
    if not bool(torch.isfinite(roi_masks).all()):
        raise ValueError("ROI masks contain NaN or infinity")
    if not bool(((roi_masks == 0) | (roi_masks == 1)).all()):
        raise ValueError("ROI masks must be exactly binary")
    counts = roi_masks.flatten(2).sum(dim=-1)
    if not bool((counts > 0).all()):
        raise ValueError(
            "every configured ROI must have non-empty target-validity support "
            "for every subject"
        )
    return counts


def _roi_timeseries(vol: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
    """
    Average signal inside each ROI at every time point.

    vol       : [B, C, T, D, H, W]
    roi_masks : [B, R, D, H, W]   (binary)
    returns   : [B, R, T]         (mean signal per ROI per frame)
    """
    B, C, T, D, H, W = vol.shape
    R = roi_masks.shape[1]
    v = vol.mean(dim=1)                                   # [B, T, D, H, W]
    v = v.reshape(B, T, D * H * W)                        # [B, T, P]
    m = roi_masks.reshape(B, R, D * H * W).to(v.dtype)    # [B, R, P]
    denom = _require_all_roi_support(roi_masks).to(v.dtype)  # [B, R]
    ts = torch.einsum("brp,btp->brt", m, v) / denom.unsqueeze(-1)
    return ts                                             # [B, R, T]


def _expand_spatial_mask(mask: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """Broadcast a spatial brain mask to ``vol`` without guessing time as space."""
    m = mask.to(device=vol.device, dtype=vol.dtype)
    if m.ndim == 3:                         # [D,H,W]
        m = m.unsqueeze(0).unsqueeze(0)     # [1,1,D,H,W]
    elif m.ndim == 4:                       # [B,D,H,W]
        m = m.unsqueeze(1)                  # [B,1,D,H,W]
    if m.ndim != 5:
        raise ValueError(f"brain mask must be [D,H,W], [B,D,H,W], or [B,1,D,H,W], got {m.shape}")
    if m.shape[0] not in (1, vol.shape[0]):
        raise ValueError(f"brain-mask batch {m.shape[0]} does not match volume batch {vol.shape[0]}")
    if tuple(m.shape[-3:]) != tuple(vol.shape[-3:]):
        raise ValueError(f"brain-mask spatial shape {m.shape[-3:]} does not match volume {vol.shape[-3:]}")
    if vol.ndim == 6:
        m = m.unsqueeze(2)                  # [B,1,1,D,H,W]
    elif vol.ndim != 5:
        raise ValueError(f"expected a 5D/6D volume, got {vol.shape}")
    return m.expand_as(vol)


# --------------------------------------------------------------------------- #
# 1. 3D SSIM Loss
# --------------------------------------------------------------------------- #
class SSIM3DLoss(nn.Module):
    """One minus mean frame-wise 3D SSIM over the complete 4D sequence.

    The manuscript specifies 3D SSIM for 4D synthesis but does not prescribe a
    temporal-mean reduction.  Scoring corresponding frames prevents a
    temporally collapsed prediction from receiving a perfect SSIM merely
    because its mean volume matches.  Frame chunks and activation
    checkpointing bound the working set for 128-frame training volumes.
    """

    contract = FRAMEWISE_SSIM_CONTRACT

    def __init__(
        self,
        window_size: int = 7,
        c1: float = 0.01 ** 2,
        c2: float = 0.03 ** 2,
        frame_chunk_size: int = 4,
    ):
        super().__init__()
        if (
            isinstance(frame_chunk_size, bool)
            or not isinstance(frame_chunk_size, Integral)
            or frame_chunk_size < 1
        ):
            raise ValueError("SSIM frame_chunk_size must be a positive integer")
        self.window_size = window_size
        self.c1, self.c2 = c1, c2
        self.pad = window_size // 2
        self.frame_chunk_size = int(frame_chunk_size)

    def _ssim(
        self, a: torch.Tensor, b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = min(self.window_size, *a.shape[-3:])
        if k % 2 == 0:
            k -= 1
        k = max(k, 1)
        pad = k // 2

        def avg(x: torch.Tensor) -> torch.Tensor:
            # A cubic box filter is separable. Three 1D pools are
            # mathematically equivalent to one k^3 pool but avoid multiplying
            # the full-frame SSIM cost by k^3 for all 128 frames.
            x = F.avg_pool3d(
                x, (k, 1, 1), stride=1, padding=(pad, 0, 0)
            )
            x = F.avg_pool3d(
                x, (1, k, 1), stride=1, padding=(0, pad, 0)
            )
            return F.avg_pool3d(
                x, (1, 1, k), stride=1, padding=(0, 0, pad)
            )

        mu_a, mu_b = avg(a), avg(b)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        var_a = avg(a * a) - mu_a2
        var_b = avg(b * b) - mu_b2
        cov = avg(a * b) - mu_ab
        num = (2 * mu_ab + self.c1) * (2 * cov + self.c2)
        den = (mu_a2 + mu_b2 + self.c1) * (var_a + var_b + self.c2)
        ssim_map = num / den.clamp_min(1e-8)
        if mask is None:
            return ssim_map.flatten(1).mean(dim=1)
        m = _expand_spatial_mask(mask, a)
        return (ssim_map * m).flatten(1).sum(dim=1) / m.flatten(1).sum(
            dim=1
        ).clamp_min(1.0)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
        mask: Optional[torch.Tensor] = None, **_,
    ) -> torch.Tensor:
        if pred.ndim != 6 or pred.shape != target.shape:
            raise ValueError(
                "SSIM expects matching [B,C,T,D,H,W] prediction and target"
            )
        batch, channels, frames, depth, height, width = pred.shape
        p = pred.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * frames, channels, depth, height, width
        )
        t = target.permute(0, 2, 1, 3, 4, 5).reshape_as(p)
        m = None
        if mask is not None:
            m = _expand_spatial_mask(mask, pred).permute(
                0, 2, 1, 3, 4, 5
            ).reshape_as(p)
        if channels > 1:
            p, t = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
            if m is not None:
                m = m[:, :1]
        if m is not None:
            p, t = p * m, t * m

        scores = []
        for start in range(0, batch * frames, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, batch * frames)
            p_chunk = p[start:stop]
            t_chunk = t[start:stop]
            m_chunk = None if m is None else m[start:stop]

            def score_chunk(
                left: torch.Tensor,
                right: torch.Tensor,
                mask_chunk: Optional[torch.Tensor] = m_chunk,
            ) -> torch.Tensor:
                return self._ssim(left, right, mask=mask_chunk)

            if torch.is_grad_enabled() and p_chunk.requires_grad:
                score = checkpoint(
                    score_chunk,
                    p_chunk,
                    t_chunk,
                    use_reentrant=False,
                )
            else:
                score = score_chunk(p_chunk, t_chunk)
            scores.append(score)
        return 1.0 - torch.cat(scores).mean()


# --------------------------------------------------------------------------- #
# 2. Voxel Intensity Loss
# --------------------------------------------------------------------------- #
class VoxelIntensityLoss(nn.Module):
    """Per-voxel L1 over the full 4D volume (optionally masked to the brain)."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        diff = (pred - target).abs()
        if mask is not None:
            m = _expand_spatial_mask(mask, pred)
            diff = diff * m
            return diff.sum() / m.sum().clamp_min(1.0)
        return diff.mean()


# --------------------------------------------------------------------------- #
# 3. Volume Loss
# --------------------------------------------------------------------------- #
class VolumeLoss(nn.Module):
    """
    Match the integrated (mean) signal inside each ROI -- the "ROI Volume: x%"
    term in Figure 1E.  Compares per-ROI mean intensity of the temporal-mean
    prediction vs. target.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if roi_masks is None:
            raise ValueError("ROI masks are required for the ROI volume loss")
        p = _temporal_mean(pred).mean(dim=1, keepdim=False)   # [B, D, H, W]
        t = _temporal_mean(target).mean(dim=1, keepdim=False)
        B, D, H, W = p.shape
        R = roi_masks.shape[1]
        m = roi_masks.reshape(B, R, D * H * W).to(p.dtype)
        counts = _require_all_roi_support(roi_masks).to(p.dtype)
        denom = counts
        p_roi = (m * p.reshape(B, 1, -1)).sum(dim=2) / denom  # [B, R]
        t_roi = (m * t.reshape(B, 1, -1)).sum(dim=2) / denom
        return (p_roi - t_roi).abs().mean()


# --------------------------------------------------------------------------- #
# 4. Region Histogram Matching Loss
# --------------------------------------------------------------------------- #
class RegionHistogramLoss(nn.Module):
    """
    Differentiable soft-histogram matching of intensity distributions inside
    each ROI (Figure 1E, "Reference vs Predicted Distribution").
    """

    def __init__(self, num_bins: int = 16, vmin: float = 0.0, vmax: float = 1.0, sigma: float = 0.05):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.register_buffer("centers", torch.linspace(vmin, vmax, num_bins))

    def _soft_hist(self, vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # vals/weights: [B, P]   -> hist: [B, num_bins]
        d = vals.unsqueeze(-1) - self.centers.view(1, 1, -1)         # [B, P, K]
        k = torch.exp(-0.5 * (d / self.sigma) ** 2)
        h = (k * weights.unsqueeze(-1)).sum(dim=1)                   # [B, K]
        return h / h.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if roi_masks is None:
            raise ValueError("ROI masks are required for regional histogram loss")
        p = _temporal_mean(pred).mean(dim=1)                        # [B, D, H, W]
        t = _temporal_mean(target).mean(dim=1)
        B, D, H, W = p.shape
        R = roi_masks.shape[1]
        pf = p.reshape(B, -1)
        tf = t.reshape(B, -1)
        m = roi_masks.reshape(B, R, -1).to(p.dtype)
        _require_all_roi_support(roi_masks)
        losses = []
        for r in range(R):
            w = m[:, r, :]                                          # [B, P]
            hp = self._soft_hist(pf, w)
            ht = self._soft_hist(tf, w)
            losses.append(F.l1_loss(hp, ht))
        return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# 5. Temporal-coherence loss
# --------------------------------------------------------------------------- #
class TemporalCoherenceLoss(nn.Module):
    """Match scale-normalized first-order BOLD changes inside the brain.

    The paper names a separate temporal objective but does not publish its
    equation. Raw delta L1 is intensity-scale dominated: with the observed
    recovery data, a static temporal mean incurs only about 0.008 before the
    published 0.2 lambda is applied. Dividing each subject's delta error by its
    real in-mask mean absolute delta makes a static prediction score about one,
    independent of the baseline BOLD intensity. This is a versioned recovery
    definition of the paper-named term, not a reported manuscript equation.
    """

    contract = "connect4-target-delta-relative-l1-v1"

    def __init__(self, minimum_target_delta: float = 1e-4):
        super().__init__()
        if minimum_target_delta <= 0:
            raise ValueError("minimum_target_delta must be positive")
        self.minimum_target_delta = float(minimum_target_delta)

    @staticmethod
    def _per_subject_mean(
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return values.flatten(1).mean(dim=1)
        expanded = _expand_spatial_mask(mask, values)
        numerator = (values * expanded).flatten(1).sum(dim=1)
        denominator = expanded.flatten(1).sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if pred.shape[2] < 2:
            return pred.new_zeros(())
        pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
        target_delta = target[:, :, 1:] - target[:, :, :-1]
        delta_error = self._per_subject_mean(
            (pred_delta - target_delta).abs(), mask
        )
        target_scale = self._per_subject_mean(target_delta.abs(), mask)
        return (
            delta_error / target_scale.clamp_min(self.minimum_target_delta)
        ).mean()


# --------------------------------------------------------------------------- #
# 6. Perceptual Loss (4D, BrainLM features)
# --------------------------------------------------------------------------- #
class PerceptualLoss(nn.Module):
    """
    Deep-feature (perceptual) loss on the *functional* signal. Predicted and
    target 4D fMRI are embedded with a frozen BrainLM reference model and the
    loss is the distance between subject-level embeddings.

    A real, frozen BrainLM extractor is required.  There is intentionally no
    spectral or random-model fallback because such a value is not the
    perceptual objective described in the paper.
    """

    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        super().__init__()
        self.fe = feature_extractor
        self.last_context_identity: Optional[Dict[str, Any]] = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        roi_masks: Optional[torch.Tensor] = None,
        brainlm_context: Optional[Mapping[str, Any]] = None,
        **_,
    ) -> torch.Tensor:
        if self.fe is None:
            raise RuntimeError(
                "PerceptualLoss requires the pretrained BrainLM 4D feature extractor; "
                "no paper-faithful fallback is available"
            )
        self.fe.eval()
        contextual = bool(getattr(self.fe, "connect4_requires_context", False))
        if contextual and not isinstance(brainlm_context, Mapping):
            raise RuntimeError(
                "contextual BrainLM perceptual loss requires scan IDs, roles, "
                "padded affine/crop/support, structural hashes, and reviewed "
                "nonlinear-displacement authority"
            )
        if mask is not None:
            expanded = _expand_spatial_mask(mask, pred)
            pred = pred * expanded
            target = target * expanded
        if contextual:
            pred_output = self.fe(
                pred,
                context=brainlm_context,
                input_role="prediction",
            )
            pred_features, _ = split_feature_output(pred_output)
            with torch.no_grad():
                target_output = self.fe(
                    target,
                    context=brainlm_context,
                    input_role="target",
                )
                target_features, _ = split_feature_output(target_output)
            pred_context = (
                pred_output.get("context_sha256")
                if isinstance(pred_output, Mapping)
                else None
            )
            target_context = (
                target_output.get("context_sha256")
                if isinstance(target_output, Mapping)
                else None
            )
            pred_identity = (
                pred_output.get("context_identity")
                if isinstance(pred_output, Mapping)
                else None
            )
            target_identity = (
                target_output.get("context_identity")
                if isinstance(target_output, Mapping)
                else None
            )
            if (
                not isinstance(pred_context, str)
                or pred_context != target_context
                or not isinstance(pred_identity, Mapping)
                or dict(pred_identity) != dict(target_identity or {})
                or pred_identity.get("record_sha256") != pred_context
                or pred_output.get("input_role") != "prediction"
                or target_output.get("input_role") != "target"
            ):
                raise RuntimeError(
                    "BrainLM prediction and target did not use one identical "
                    "authenticated projection context"
                )
            self.last_context_identity = dict(pred_identity)
        else:
            pred_features, _ = split_feature_output(self.fe(pred))
            with torch.no_grad():
                target_features, _ = split_feature_output(self.fe(target))
        if pred_features.shape != target_features.shape:
            raise ValueError(
                f"BrainLM feature shape mismatch: {pred_features.shape} vs {target_features.shape}"
            )
        if not torch.isfinite(pred_features).all() or not torch.isfinite(
            target_features
        ).all():
            raise RuntimeError("BrainLM perceptual features contain NaN or infinity")
        if torch.is_grad_enabled() and pred.requires_grad and not pred_features.requires_grad:
            raise RuntimeError(
                "BrainLM prediction features are detached; perceptual gradients "
                "must reach the synthesized fMRI"
            )
        return F.l1_loss(pred_features, target_features)


# --------------------------------------------------------------------------- #
# 7. Functional Connectivity Matrix Loss
# --------------------------------------------------------------------------- #
class FCMatrixLoss(nn.Module):
    """
    Pearson correlation matrix over ROI time-series, compared between prediction
    and target (Figure 1E, "Functional Connectivity Matrix Loss").
    """

    @staticmethod
    def _fc(ts: torch.Tensor) -> torch.Tensor:
        # ts: [B, R, T] -> [B, R, R] correlation matrix
        ts = ts - ts.mean(dim=-1, keepdim=True)
        norm = ts.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        ts = ts / norm
        return torch.matmul(ts, ts.transpose(1, 2))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if roi_masks is None:
            raise ValueError("ROI masks are required for functional-connectivity loss")
        _require_all_roi_support(roi_masks)
        fc_p = self._fc(_roi_timeseries(pred, roi_masks))
        fc_t = self._fc(_roi_timeseries(target, roi_masks))
        return F.l1_loss(fc_p, fc_t)


# --------------------------------------------------------------------------- #
# combined objective
# --------------------------------------------------------------------------- #
class Connect4Loss(nn.Module):
    """
    Paper-weighted biologically guided objective.  Returns every individual
    (unweighted) term plus the weighted ``total``. The five published lambdas
    are the defaults below. Figure 1E also names volume and regional-histogram
    auxiliaries but the manuscript publishes no lambdas for them, so their 0.5
    implementation defaults remain explicit/configurable rather than being
    presented as paper-reported values.
    """

    def __init__(
        self,
        ssim_weight: float = 0.5,
        voxel_weight: float = 1.0,
        volume_weight: float = 0.5,
        region_hist_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        fc_weight: float = 0.3,
        temporal_weight: float = 0.2,
        num_hist_bins: int = 16,
        perceptual_feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ssim = SSIM3DLoss()
        self.voxel = VoxelIntensityLoss()
        self.volume = VolumeLoss()
        self.region_hist = RegionHistogramLoss(num_bins=num_hist_bins)
        self.temporal = TemporalCoherenceLoss()
        self.perceptual = PerceptualLoss(perceptual_feature_extractor)
        self.fc = FCMatrixLoss()
        self.w = dict(
            ssim=ssim_weight, voxel=voxel_weight, volume=volume_weight,
            region_hist=region_hist_weight, temporal=temporal_weight,
            perceptual=perceptual_weight, fc=fc_weight,
        )
        if any(float(weight) < 0 for weight in self.w.values()):
            raise ValueError("loss weights must be non-negative")

    def forward(
        self,
        pred: torch.Tensor,                       # [B, C, T, D, H, W]
        target: torch.Tensor,                     # [B, C, T, D, H, W]
        mask: Optional[torch.Tensor] = None,      # [B, 1, D, H, W]
        roi_masks: Optional[torch.Tensor] = None, # [B, R, D, H, W]
        brainlm_context: Optional[Mapping[str, Any]] = None,
        **_,
    ) -> Dict[str, torch.Tensor]:
        if pred.shape != target.shape or pred.ndim != 6:
            raise ValueError(
                f"loss expects matching [B,C,T,D,H,W] tensors, got "
                f"{pred.shape} and {target.shape}"
            )
        if mask is None:
            raise ValueError("a brain mask is required by the biologically guided loss")
        if roi_masks is None:
            raise ValueError(
                "ROI masks are required for regional-distribution and FC losses"
            )
        # Slice the broadcast view back to one spatial plane before validation;
        # never materialize a [B,C,T,D,H,W] mask for full 128-frame volumes.
        spatial_mask = _expand_spatial_mask(mask, pred)[:, 0, 0]
        if not torch.isfinite(spatial_mask).all():
            raise ValueError("target-validity mask contains NaN or infinity")
        if not bool(((spatial_mask == 0) | (spatial_mask == 1)).all()):
            raise ValueError("target-validity mask must be exactly binary")
        if not bool(spatial_mask.flatten(1).any(dim=1).all()):
            raise ValueError("target-validity mask is empty for at least one subject")
        if (
            roi_masks.ndim != 5
            or roi_masks.shape[0] != pred.shape[0]
            or tuple(roi_masks.shape[-3:]) != tuple(pred.shape[-3:])
        ):
            raise ValueError(
                "ROI masks must be [B,R,D,H,W] on the prediction grid"
            )
        if not torch.isfinite(roi_masks).all():
            raise ValueError("ROI masks contain NaN or infinity")
        if not bool(((roi_masks == 0) | (roi_masks == 1)).all()):
            raise ValueError("ROI masks must be exactly binary")
        # Compute losses in fp32 — SSIM/FFT/correlation are numerically unstable
        # under AMP fp16 (produce NaN/Inf). The model forward still runs in AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()
            mask = spatial_mask.unsqueeze(1).float()
            roi_masks = roi_masks.float() * mask
            if not bool(roi_masks.flatten(2).any(dim=2).all()):
                raise ValueError(
                    "every configured ROI must have non-empty target-validity "
                    "support for every subject"
                )
            terms = {
                "ssim":        self.ssim(pred, target, mask=mask),
                "voxel":       self.voxel(pred, target, mask=mask),
                "volume":      self.volume(pred, target, roi_masks=roi_masks),
                "region_hist": self.region_hist(pred, target, roi_masks=roi_masks),
                "temporal":    self.temporal(pred, target, mask=mask),
                "perceptual":  (
                    self.perceptual(
                        pred,
                        target,
                        mask=mask,
                        roi_masks=roi_masks,
                        brainlm_context=brainlm_context,
                    )
                    if self.w["perceptual"] > 0
                    else pred.new_zeros(())
                ),
                "fc":          self.fc(pred, target, roi_masks=roi_masks),
            }
            terms["total"] = sum(self.w[k] * v for k, v in terms.items())
        return terms
