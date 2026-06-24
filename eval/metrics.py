"""
Synthetic-vs-real rs-fMRI evaluation metrics for CONNECT-4.

All functions accept 4D fMRI as either [B, C, T, D, H, W] or [B, T, D, H, W]
(channel optional, single-channel assumed) and an optional brain `mask`
[B,1,D,H,W] / ROI masks `roi_masks` [B, R, D, H, W].

Provided metrics
----------------
  voxel_correlation     Pearson r between temporal-mean volumes (image fidelity)
  ssim3d                3D SSIM of the temporal-mean volumes
  mse / mae / psnr      voxel error / fidelity
  fc_correlation        correlation of ROI×ROI functional-connectivity matrices
  alff_correlation      ALFF (amplitude of low-frequency fluctuation) agreement
  reho_correlation      ReHo (regional homogeneity) agreement
  spectral_correlation  per-ROI power-spectrum agreement
  slimbrain_distance    cosine distance between SLIM-Brain 4D deep features
  compute_all           dict of everything above
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models.losses import SSIM3DLoss, FCMatrixLoss, _roi_timeseries, _temporal_mean


# --------------------------------------------------------------------------- #
def _ensure_bctdhw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:                 # [B, T, D, H, W]
        x = x.unsqueeze(1)           # [B, 1, T, D, H, W]
    return x


def _flat_masked(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]):
    if mask is not None:
        m = mask
        # drop singleton channel dims until rank <= a's rank
        while m.dim() > a.dim() and 1 in m.shape:
            m = m.squeeze(m.shape.index(1))
        while m.dim() < a.dim():
            m = m.unsqueeze(1)
        m = m.expand_as(a) > 0.5
        return a[m], b[m]
    return a.reshape(-1), b.reshape(-1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a * b).sum() / denom)


# --------------------------------------------------------------------------- #
def voxel_correlation(pred, target, mask=None) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    p, t = _temporal_mean(pred), _temporal_mean(target)
    pa, ta = _flat_masked(p, t, mask)
    return _pearson(pa, ta)


def ssim3d(pred, target) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    return float(1.0 - SSIM3DLoss()(pred, target))


def mse(pred, target, mask=None) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    pa, ta = _flat_masked(pred, target, mask)
    return float(F.mse_loss(pa, ta))


def mae(pred, target, mask=None) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    pa, ta = _flat_masked(pred, target, mask)
    return float(F.l1_loss(pa, ta))


def psnr(pred, target, mask=None) -> float:
    m = mse(pred, target, mask)
    if m <= 0:
        return float("inf")
    rng = float(_ensure_bctdhw(target).max() - _ensure_bctdhw(target).min()) or 1.0
    return float(10.0 * torch.log10(torch.tensor(rng ** 2 / m)))


def fc_correlation(pred, target, roi_masks) -> float:
    """Pearson r between the upper-triangles of the ROI×ROI FC matrices."""
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    fc = FCMatrixLoss._fc
    fp = fc(_roi_timeseries(pred, roi_masks))[0]
    ft = fc(_roi_timeseries(target, roi_masks))[0]
    iu = torch.triu_indices(fp.shape[0], fp.shape[1], offset=1)
    return _pearson(fp[iu[0], iu[1]], ft[iu[0], iu[1]])


def _alff(ts: torch.Tensor) -> torch.Tensor:
    # ts: [B, R, T] -> ALFF [B, R] = sum amplitude in low-freq band (lowest 1/3)
    ts = ts - ts.mean(dim=-1, keepdim=True)
    amp = torch.fft.rfft(ts, dim=-1).abs()
    k = max(1, amp.shape[-1] // 3)
    return amp[..., 1:k + 1].sum(dim=-1)


def alff_correlation(pred, target, roi_masks) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    ap = _alff(_roi_timeseries(pred, roi_masks))[0]
    at = _alff(_roi_timeseries(target, roi_masks))[0]
    return _pearson(ap, at)


def _reho(vol: torch.Tensor) -> torch.Tensor:
    """
    Simplified ReHo per voxel: correlation of each voxel's time-series with the
    mean of its 3x3x3 neighbourhood (Kendall-W approximation), then temporal->
    scalar map. vol: [B, T, D, H, W] -> [B, D, H, W].
    """
    B, T, D, H, W = vol.shape
    v = vol - vol.mean(dim=1, keepdim=True)
    # local-neighbourhood mean time-series via 3D average pooling per frame
    neigh = F.avg_pool3d(v.reshape(B * T, 1, D, H, W), 3, stride=1, padding=1).reshape(B, T, D, H, W)
    num = (v * neigh).sum(dim=1)
    den = (v.norm(dim=1) * neigh.norm(dim=1)).clamp_min(1e-8)
    return num / den


def reho_correlation(pred, target, mask=None) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    rp = _reho(pred[:, 0])
    rt = _reho(target[:, 0])
    pa, ta = _flat_masked(rp, rt, mask)
    return _pearson(pa, ta)


def spectral_correlation(pred, target, roi_masks) -> float:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    def spec(x):
        ts = _roi_timeseries(x, roi_masks)
        ts = ts - ts.mean(dim=-1, keepdim=True)
        s = torch.fft.rfft(ts, dim=-1).abs()
        return (s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8))[0]
    return _pearson(spec(pred).reshape(-1), spec(target).reshape(-1))


def slimbrain_distance(pred, target, encoder=None) -> float:
    """Cosine distance between SLIM-Brain 4D deep features (lower = closer)."""
    from models.slimbrain_wrapper import SlimBrainEncoder
    enc = encoder or SlimBrainEncoder().eval()
    fp = enc(_ensure_bctdhw(pred))
    ft = enc(_ensure_bctdhw(target))
    cos = F.cosine_similarity(fp, ft, dim=-1).mean()
    return float(1.0 - cos)


# --------------------------------------------------------------------------- #
def compute_all(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_masks: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    slimbrain_encoder=None,
) -> Dict[str, float]:
    out = {
        "voxel_corr": voxel_correlation(pred, target, mask),
        "ssim3d": ssim3d(pred, target),
        "mse": mse(pred, target, mask),
        "mae": mae(pred, target, mask),
        "psnr": psnr(pred, target, mask),
        "reho_corr": reho_correlation(pred, target, mask),
    }
    if roi_masks is not None:
        out["fc_corr"] = fc_correlation(pred, target, roi_masks)
        out["alff_corr"] = alff_correlation(pred, target, roi_masks)
        out["spectral_corr"] = spectral_correlation(pred, target, roi_masks)
    try:
        out["slimbrain_dist"] = slimbrain_distance(pred, target, slimbrain_encoder)
    except Exception as e:  # pragma: no cover
        print(f"[metrics] SLIM-Brain distance skipped: {e}")
    return out
