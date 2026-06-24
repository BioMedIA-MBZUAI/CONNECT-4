"""
Loss components for CONNECT-4 (Figure 1E).

Exactly the six terms drawn in the paper, nothing else:

    1. SSIM3DLoss              - 3D SSIM Loss              (structural comparison)
    2. VoxelIntensityLoss      - Voxel Intensity Loss      (per-voxel L1)
    3. VolumeLoss              - Volume Loss               (per-ROI integrated signal)
    4. RegionHistogramLoss     - Region Histogram Matching (per-ROI intensity distribution)
    5. PerceptualLoss          - Perceptual Loss, 4D       (deep features from BrainLM)
    6. FCMatrixLoss            - Functional Connectivity Matrix Loss (ROI x ROI corr.)

All tensors are full 4D volumes [B, C, T, D, H, W] unless stated otherwise.
The combined objective `Connect4Loss` returns every term plus a weighted total.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, D, H, W] -> [B, C, D, H, W] (mean over time)."""
    return x.mean(dim=2)


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
    denom = m.sum(dim=2).clamp_min(1.0)                   # [B, R]
    ts = torch.einsum("brp,btp->brt", m, v) / denom.unsqueeze(-1)
    return ts                                             # [B, R, T]


# --------------------------------------------------------------------------- #
# 1. 3D SSIM Loss
# --------------------------------------------------------------------------- #
class SSIM3DLoss(nn.Module):
    """1 - mean 3D SSIM between the temporal-mean volumes."""

    def __init__(self, window_size: int = 7, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.c1, self.c2 = c1, c2
        self.pad = window_size // 2

    def _ssim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        k = self.window_size
        avg = lambda x: F.avg_pool3d(x, k, stride=1, padding=self.pad)
        mu_a, mu_b = avg(a), avg(b)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        var_a = avg(a * a) - mu_a2
        var_b = avg(b * b) - mu_b2
        cov = avg(a * b) - mu_ab
        num = (2 * mu_ab + self.c1) * (2 * cov + self.c2)
        den = (mu_a2 + mu_b2 + self.c1) * (var_a + var_b + self.c2)
        return (num / den.clamp_min(1e-8)).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
        p = _temporal_mean(pred)                 # [B, C, D, H, W]
        t = _temporal_mean(target)
        if p.shape[1] > 1:                       # collapse channels for SSIM
            p, t = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
        return 1.0 - self._ssim(p, t)


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
            m = mask
            while m.dim() < pred.dim():
                m = m.unsqueeze(2)               # broadcast over T (and C)
            diff = diff * m
            return diff.sum() / m.expand_as(diff).sum().clamp_min(1.0)
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
            return (_temporal_mean(pred).mean() - _temporal_mean(target).mean()).abs()
        p = _temporal_mean(pred).mean(dim=1, keepdim=False)   # [B, D, H, W]
        t = _temporal_mean(target).mean(dim=1, keepdim=False)
        B, D, H, W = p.shape
        R = roi_masks.shape[1]
        m = roi_masks.reshape(B, R, D * H * W).to(p.dtype)
        denom = m.sum(dim=2).clamp_min(1.0)
        p_roi = (m * p.reshape(B, 1, -1)).sum(dim=2) / denom  # [B, R]
        t_roi = (m * t.reshape(B, 1, -1)).sum(dim=2) / denom
        return F.l1_loss(p_roi, t_roi)


# --------------------------------------------------------------------------- #
# 4. Region Histogram Matching Loss
# --------------------------------------------------------------------------- #
class RegionHistogramLoss(nn.Module):
    """
    Differentiable soft-histogram matching of intensity distributions inside
    each ROI (Figure 1E, "Reference vs Predicted Distribution").
    """

    def __init__(self, num_bins: int = 16, vmin: float = -3.0, vmax: float = 3.0, sigma: float = 0.1):
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
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        p = _temporal_mean(pred).mean(dim=1)                        # [B, D, H, W]
        t = _temporal_mean(target).mean(dim=1)
        B, D, H, W = p.shape
        R = roi_masks.shape[1]
        pf = p.reshape(B, -1)
        tf = t.reshape(B, -1)
        m = roi_masks.reshape(B, R, -1).to(p.dtype)
        loss = pred.new_zeros(())
        for r in range(R):
            w = m[:, r, :]                                          # [B, P]
            if w.sum() < 1.0:
                continue
            hp = self._soft_hist(pf, w)
            ht = self._soft_hist(tf, w)
            loss = loss + F.l1_loss(hp, ht)
        return loss / max(R, 1)


# --------------------------------------------------------------------------- #
# 5. Perceptual Loss (4D, BrainLM features)
# --------------------------------------------------------------------------- #
class PerceptualLoss(nn.Module):
    """
    Deep-feature (perceptual) loss on the *functional* signal.  Predicted and
    target fMRI are parcellated into ROI time-series and embedded with a frozen
    4D foundation model (BrainLM); the loss is the distance between embeddings.

    If the BrainLM encoder cannot be loaded, falls back to matching the
    per-ROI temporal power-spectrum (a model-free perceptual proxy) so training
    still runs.  Pass a loaded encoder via `feature_extractor` to use BrainLM.
    """

    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        super().__init__()
        self.fe = feature_extractor

    def _spectral(self, ts: torch.Tensor) -> torch.Tensor:
        # ts: [B, R, T] -> normalised power spectrum [B, R, T//2+1]
        ts = ts - ts.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(ts, dim=-1).abs()
        return spec / spec.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        # Preferred: deep 4D features from a frozen foundation model (SLIM-Brain),
        # computed directly on the full 4D volumes [B, C, T, D, H, W].
        if self.fe is not None:
            return F.l1_loss(self.fe(pred), self.fe(target))
        # model-free perceptual proxy: per-ROI spectral signature
        if roi_masks is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        ts_p = _roi_timeseries(pred, roi_masks)     # [B, R, T]
        ts_t = _roi_timeseries(target, roi_masks)
        return F.l1_loss(self._spectral(ts_p), self._spectral(ts_t))


# --------------------------------------------------------------------------- #
# 6. Functional Connectivity Matrix Loss
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
        std = ts.std(dim=-1, keepdim=True).clamp_min(1e-6)
        ts = ts / std
        T = ts.shape[-1]
        return torch.matmul(ts, ts.transpose(1, 2)) / max(T - 1, 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if roi_masks is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        fc_p = self._fc(_roi_timeseries(pred, roi_masks))
        fc_t = self._fc(_roi_timeseries(target, roi_masks))
        return F.l1_loss(fc_p, fc_t)


# --------------------------------------------------------------------------- #
# combined objective
# --------------------------------------------------------------------------- #
class Connect4Loss(nn.Module):
    """
    Weighted sum of the six Figure-1E components.  Returns a dict with every
    individual (unweighted) term plus the weighted `total`.
    """

    def __init__(
        self,
        ssim_weight: float = 1.0,
        voxel_weight: float = 1.0,
        volume_weight: float = 0.5,
        region_hist_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        fc_weight: float = 0.5,
        num_hist_bins: int = 16,
        perceptual_feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ssim = SSIM3DLoss()
        self.voxel = VoxelIntensityLoss()
        self.volume = VolumeLoss()
        self.region_hist = RegionHistogramLoss(num_bins=num_hist_bins)
        self.perceptual = PerceptualLoss(perceptual_feature_extractor)
        self.fc = FCMatrixLoss()
        self.w = dict(
            ssim=ssim_weight, voxel=voxel_weight, volume=volume_weight,
            region_hist=region_hist_weight, perceptual=perceptual_weight, fc=fc_weight,
        )

    def forward(
        self,
        pred: torch.Tensor,                       # [B, C, T, D, H, W]
        target: torch.Tensor,                     # [B, C, T, D, H, W]
        mask: Optional[torch.Tensor] = None,      # [B, 1, D, H, W]
        roi_masks: Optional[torch.Tensor] = None, # [B, R, D, H, W]
        **_,
    ) -> Dict[str, torch.Tensor]:
        # Compute losses in fp32 — SSIM/FFT/correlation are numerically unstable
        # under AMP fp16 (produce NaN/Inf). The model forward still runs in AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()
            mask = mask.float() if mask is not None else None
            roi_masks = roi_masks.float() if roi_masks is not None else None
            terms = {
                "ssim":        self.ssim(pred, target),
                "voxel":       self.voxel(pred, target, mask=mask),
                "volume":      self.volume(pred, target, roi_masks=roi_masks),
                "region_hist": self.region_hist(pred, target, roi_masks=roi_masks),
                "perceptual":  self.perceptual(pred, target, roi_masks=roi_masks),
                "fc":          self.fc(pred, target, roi_masks=roi_masks),
            }
            terms["total"] = sum(self.w[k] * v for k, v in terms.items())
        return terms
