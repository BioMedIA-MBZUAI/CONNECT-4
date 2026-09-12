"""Paper-reported rs-fMRI synthesis metrics for CONNECT-4.

Definitions used for the three explicitly named Pearson metrics are:

``voxel_corr``
    Pearson correlation of the paired real/generated temporal signal at each
    in-brain voxel, averaged over voxels and subjects.
``roi_corr``
    Pearson correlation of the paired real/generated mean ROI time-series,
    averaged over every configured ROI and subject. Every configured ROI must
    have non-empty target-validity support. This is intentionally distinct from
    FC-matrix correlation, which is a training loss.
``f2f_corr``
    Spatial Pearson correlation between each paired real/generated frame inside
    the brain, averaged over frames and subjects.

MSE, SSIM and PSNR are subject-level metrics averaged over a batch. FID and
Inception Score are distributional metrics and therefore live only on
``SynthesisMetricAccumulator``; they are never fabricated for one sample.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.feature_extractors import split_feature_output
from models.losses import SSIM3DLoss, _roi_timeseries


def _ensure_bctdhw(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"fMRI input must be a torch.Tensor, got {type(x)!r}")
    if x.ndim == 5:                     # [B,T,D,H,W]
        x = x.unsqueeze(1)              # [B,1,T,D,H,W]
    if x.ndim != 6:
        raise ValueError(f"expected [B,C,T,D,H,W] or [B,T,D,H,W], got {x.shape}")
    return x.float()


def _paired(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred, target = _ensure_bctdhw(pred), _ensure_bctdhw(target)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {pred.shape} vs {target.shape}")
    return pred, target


def _spatial_mask(mask: Optional[torch.Tensor], vol: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask shaped ``[B,P]`` for a BCTDHW volume."""
    b, _, _, d, h, w = vol.shape
    if mask is None:
        raise ValueError(
            "an explicit paired target-validity mask is required; scoring the "
            "background or deriving support inside a metric is forbidden"
        )
    m = torch.as_tensor(mask, device=vol.device)
    if m.ndim == 3:
        m = m.unsqueeze(0)
    elif m.ndim == 5:
        if m.shape[1] != 1:
            raise ValueError(
                f"target-validity mask channel dimension must be 1, got {m.shape}"
            )
        m = m[:, 0]
    elif m.ndim != 4:
        raise ValueError(
            "target-validity mask must be [D,H,W], [B,D,H,W], or "
            f"[B,1,D,H,W], got {m.shape}"
        )
    if m.shape[0] == 1 and b > 1:
        m = m.expand(b, -1, -1, -1)
    if m.shape[0] != b or tuple(m.shape[-3:]) != (d, h, w):
        raise ValueError(
            f"target-validity mask {m.shape} is incompatible with volume {vol.shape}"
        )
    if not bool(torch.isfinite(m).all()):
        raise ValueError("target-validity mask contains NaN or infinity")
    if not bool(((m == 0) | (m == 1)).all()):
        raise ValueError("target-validity mask must be exactly binary")
    flat = m.reshape(b, -1) > 0.5
    if not bool(flat.any(dim=1).all()):
        raise ValueError("target-validity mask is empty for at least one subject")
    return flat


def _pearson_last(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable Pearson r along the final dimension; undefined constants map to 0."""
    a = a.float() - a.float().mean(dim=-1, keepdim=True)
    b = b.float() - b.float().mean(dim=-1, keepdim=True)
    numerator = (a * b).sum(dim=-1)
    denominator = (a.square().sum(dim=-1) * b.square().sum(dim=-1)).sqrt()
    return torch.where(
        denominator > 1e-12, numerator / denominator.clamp_min(1e-12), 0.0,
    )


def _masked_pearson_last(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Pearson r along the final dimension using a ``[B,N]`` spatial mask."""
    weights = mask.to(a.dtype)
    while weights.ndim < a.ndim:
        weights = weights.unsqueeze(1)
    count = weights.sum(dim=-1).clamp_min(1.0)
    am = (a * weights).sum(dim=-1) / count
    bm = (b * weights).sum(dim=-1) / count
    ac = (a - am.unsqueeze(-1)) * weights
    bc = (b - bm.unsqueeze(-1)) * weights
    numerator = (ac * bc).sum(dim=-1)
    denominator = (ac.square().sum(dim=-1) * bc.square().sum(dim=-1)).sqrt()
    return torch.where(
        denominator > 1e-12, numerator / denominator.clamp_min(1e-12), 0.0,
    )


def voxel_correlation(pred, target, mask=None) -> float:
    """Mean temporal Pearson correlation at each in-brain voxel."""
    pred, target = _paired(pred, target)
    p = pred.mean(dim=1).flatten(2).transpose(1, 2)    # [B,P,T]
    t = target.mean(dim=1).flatten(2).transpose(1, 2)
    corr = _pearson_last(p, t)                        # [B,P]
    valid = _spatial_mask(mask, pred)
    per_subject = (corr * valid).sum(dim=1) / valid.sum(dim=1)
    return float(per_subject.mean())


def roi_correlation(pred, target, roi_masks, mask=None) -> float:
    """Mean temporal Pearson correlation of paired ROI-average time-series."""
    pred, target = _paired(pred, target)
    if roi_masks is None:
        raise ValueError("roi_masks are required to compute the paper's ROI correlation")
    roi_masks = torch.as_tensor(roi_masks, device=pred.device, dtype=pred.dtype)
    if roi_masks.ndim == 4:
        roi_masks = roi_masks.unsqueeze(0)
    if roi_masks.ndim != 5 or roi_masks.shape[0] != pred.shape[0]:
        raise ValueError(f"ROI masks must be [B,R,D,H,W], got {roi_masks.shape}")
    if tuple(roi_masks.shape[-3:]) != tuple(pred.shape[-3:]):
        raise ValueError(
            f"ROI-mask spatial shape {roi_masks.shape[-3:]} != volume {pred.shape[-3:]}"
        )
    if not bool(torch.isfinite(roi_masks).all()):
        raise ValueError("ROI masks contain NaN or infinity")
    if not bool(((roi_masks == 0) | (roi_masks == 1)).all()):
        raise ValueError("ROI masks must be exactly binary")
    spatial = _spatial_mask(mask, pred).reshape(
        pred.shape[0], *pred.shape[-3:]
    )
    roi_masks = roi_masks * spatial.unsqueeze(1).to(roi_masks.dtype)
    corr = _pearson_last(
        _roi_timeseries(pred, roi_masks),
        _roi_timeseries(target, roi_masks),
    )                                                   # [B,R]
    valid = roi_masks.flatten(2).sum(dim=-1) > 0
    if not bool(valid.all()):
        raise ValueError(
            "every configured ROI must have non-empty target-validity support "
            "for every subject"
        )
    per_subject = corr.mean(dim=1)
    return float(per_subject.mean())


def frame_to_frame_correlation(pred, target, mask=None) -> float:
    """Mean spatial Pearson r between corresponding generated and real frames."""
    pred, target = _paired(pred, target)
    p = pred.mean(dim=1).flatten(2)                     # [B,T,P]
    t = target.mean(dim=1).flatten(2)
    corr = _masked_pearson_last(p, t, _spatial_mask(mask, pred))
    return float(corr.mean())


def mse(pred, target, mask=None) -> float:
    pred, target = _paired(pred, target)
    valid = _spatial_mask(mask, pred).to(pred.dtype)[:, None, None, :]
    square_error = (pred - target).square().flatten(3)
    numerator = (square_error * valid).sum(dim=(1, 2, 3))
    denominator = valid.sum(dim=3).flatten() * pred.shape[1] * pred.shape[2]
    return float((numerator / denominator.clamp_min(1.0)).mean())


def ssim3d(pred, target, mask=None) -> float:
    """Mean brain-masked frame-wise 3D SSIM over each 4D subject."""
    pred, target = _paired(pred, target)
    flat_masks = _spatial_mask(mask, pred)
    scores = []
    loss = SSIM3DLoss()
    for index in range(pred.shape[0]):
        subject_mask = flat_masks[index].reshape(
            1, 1, *pred.shape[-3:]
        )
        scores.append(
            1.0 - loss(pred[index:index + 1], target[index:index + 1], mask=subject_mask)
        )
    return float(torch.stack(scores).mean())


def psnr(pred, target, mask=None) -> float:
    """Mean subject PSNR using each target subject's in-brain dynamic range."""
    pred, target = _paired(pred, target)
    valid = _spatial_mask(mask, pred)
    values = []
    for index in range(pred.shape[0]):
        p = pred[index].flatten(2)[..., valid[index]]
        t = target[index].flatten(2)[..., valid[index]]
        error = (p - t).square().mean()
        if float(error) == 0.0:
            values.append(pred.new_tensor(float("inf")))
            continue
        data_range = t.max() - t.min()
        if float(data_range) <= 0.0:
            data_range = t.new_tensor(1.0)
        values.append(10.0 * torch.log10(data_range.square() / error))
    return float(torch.stack(values).mean())


def compute_all(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_masks: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute the six per-subject/batch paper metrics (FID/IS are dataset-only)."""
    if roi_masks is None:
        raise ValueError("roi_masks are required for the paper-reported metric set")
    return {
        "mse": mse(pred, target, mask),
        "ssim": ssim3d(pred, target, mask),
        "voxel_corr": voxel_correlation(pred, target, mask),
        "roi_corr": roi_correlation(pred, target, roi_masks, mask=mask),
        "f2f_corr": frame_to_frame_correlation(pred, target, mask),
        "psnr": psnr(pred, target, mask),
    }


def frechet_distance_from_features(
    generated: torch.Tensor, real: torch.Tensor,
) -> float:
    """Frechet distance between two feature sets, each containing >=2 subjects."""
    generated = torch.as_tensor(generated, dtype=torch.float64)
    real = torch.as_tensor(real, dtype=torch.float64)
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("FID features must be matrices [N,F]")
    if generated.shape[0] < 2 or real.shape[0] < 2:
        raise ValueError("FID requires at least two generated and two real subjects")
    if generated.shape[1] != real.shape[1]:
        raise ValueError(
            f"FID feature dimensions differ: {generated.shape[1]} vs {real.shape[1]}"
        )
    if not bool(torch.isfinite(generated).all() and torch.isfinite(real).all()):
        raise ValueError("FID features contain NaN or infinity")

    def covariance(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=0, keepdim=True)
        return centered.T @ centered / (x.shape[0] - 1)

    mu_generated, mu_real = generated.mean(dim=0), real.mean(dim=0)
    cov_generated, cov_real = covariance(generated), covariance(real)
    # tr(sqrt(Cg Cr)) is evaluated through the similar symmetric PSD matrix
    # sqrt(Cg) Cr sqrt(Cg), avoiding an unstable non-symmetric square root.
    eigvals_g, eigvecs_g = torch.linalg.eigh(cov_generated)
    sqrt_g = (
        eigvecs_g * eigvals_g.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigvecs_g.T
    middle = sqrt_g @ cov_real @ sqrt_g
    trace_sqrt = torch.linalg.eigvalsh(
        (middle + middle.T) * 0.5
    ).clamp_min(0).sqrt().sum()
    fid = (mu_generated - mu_real).square().sum()
    fid = fid + torch.trace(cov_generated) + torch.trace(cov_real) - 2.0 * trace_sqrt
    return float(fid.clamp_min(0.0))


def inception_score_from_logits(logits: torch.Tensor) -> float:
    """Inception Score over generated-subject class logits (dataset-level only)."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    if logits.ndim != 2 or logits.shape[0] < 2 or logits.shape[1] < 2:
        raise ValueError("Inception Score requires logits [N,K] with N>=2 and K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Inception logits contain NaN or infinity")
    log_conditional = torch.log_softmax(logits, dim=-1)
    conditional = log_conditional.exp()
    marginal = conditional.mean(dim=0).clamp_min(torch.finfo(logits.dtype).tiny)
    kl = (conditional * (log_conditional - marginal.log())).sum(dim=-1)
    return float(kl.mean().exp())


class SynthesisMetricAccumulator:
    """Dataset-level, batch-safe accumulator for the complete paper metric set.

    Updates are transactional: tensors and feature outputs are validated before
    any state is committed. Feature tensors are detached to CPU immediately so
    validation cannot retain training graphs or exhaust accelerator memory.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        *,
        require_is_logits: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.require_is_logits = bool(require_is_logits)
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
        self.reset()

    def reset(self) -> None:
        self.num_samples = 0
        self._sums: Dict[str, float] = {}
        self._generated_features: list[torch.Tensor] = []
        self._real_features: list[torch.Tensor] = []
        self._generated_logits: list[torch.Tensor] = []
        self._feature_dim: Optional[int] = None
        self._logit_dim: Optional[int] = None

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        roi_masks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        pred, target = _paired(pred, target)
        batch_size = pred.shape[0]
        batch_metrics = compute_all(pred, target, roi_masks=roi_masks, mask=mask)

        generated_features = real_features = generated_logits = None
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            feature_pred, feature_target = pred, target
            if mask is not None:
                spatial = _spatial_mask(mask, pred).reshape(
                    batch_size, 1, 1, *pred.shape[-3:]
                ).to(pred.dtype)
                feature_pred = pred * spatial
                feature_target = target * spatial
            generated_features, generated_logits = split_feature_output(
                self.feature_extractor(feature_pred)
            )
            real_features, _ = split_feature_output(
                self.feature_extractor(feature_target)
            )
            generated_features = generated_features.detach().float().cpu()
            real_features = real_features.detach().float().cpu()
            generated_logits = (
                generated_logits.detach().float().cpu()
                if generated_logits is not None else None
            )
            if generated_features.shape[0] != batch_size or real_features.shape[0] != batch_size:
                raise ValueError("4D extractor must return exactly one feature row per subject")
            if generated_features.shape != real_features.shape:
                raise ValueError(
                    f"generated/real feature shapes differ: {generated_features.shape} "
                    f"vs {real_features.shape}"
                )
            if not bool(
                torch.isfinite(generated_features).all()
                and torch.isfinite(real_features).all()
            ):
                raise ValueError("4D extractor produced non-finite features")
            if self._feature_dim not in (None, generated_features.shape[1]):
                raise ValueError("4D extractor feature dimension changed between batches")
            if self.require_is_logits and generated_logits is None:
                raise RuntimeError(
                    "The configured SLIM-Brain evaluation extractor has no trained logits "
                    "head; Inception Score cannot be computed from embeddings alone"
                )
            if generated_logits is not None:
                if generated_logits.shape[0] != batch_size:
                    raise ValueError("4D extractor must return exactly one logits row per subject")
                if not bool(torch.isfinite(generated_logits).all()):
                    raise ValueError("4D extractor produced non-finite logits")
                if self._logit_dim not in (None, generated_logits.shape[1]):
                    raise ValueError("4D extractor logits dimension changed between batches")

        # Commit only after all computations and validations succeeded.
        for key, value in batch_metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"metric {key} is not finite: {value}")
        for key, value in batch_metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + value * batch_size
        self.num_samples += batch_size
        if generated_features is not None:
            self._feature_dim = generated_features.shape[1]
            self._generated_features.append(generated_features)
            self._real_features.append(real_features)
            if generated_logits is not None:
                self._logit_dim = generated_logits.shape[1]
                self._generated_logits.append(generated_logits)
        return batch_metrics

    def merge(self, other: "SynthesisMetricAccumulator") -> None:
        """Merge an independently accumulated shard (for distributed evaluation)."""
        if not isinstance(other, SynthesisMetricAccumulator):
            raise TypeError("can only merge another SynthesisMetricAccumulator")
        if (self.feature_extractor is None) != (other.feature_extractor is None):
            raise ValueError("cannot merge feature-enabled and feature-disabled accumulators")
        if self.require_is_logits != other.require_is_logits:
            raise ValueError("cannot merge accumulators with different IS requirements")
        if (
            self._feature_dim not in (None, other._feature_dim)
            or self._logit_dim not in (None, other._logit_dim)
        ):
            raise ValueError(
                "cannot merge accumulators with different feature/logit dimensions"
            )
        for key, value in other._sums.items():
            self._sums[key] = self._sums.get(key, 0.0) + value
        self.num_samples += other.num_samples
        self._feature_dim = self._feature_dim or other._feature_dim
        self._logit_dim = self._logit_dim or other._logit_dim
        self._generated_features.extend(t.clone() for t in other._generated_features)
        self._real_features.extend(t.clone() for t in other._real_features)
        self._generated_logits.extend(t.clone() for t in other._generated_logits)

    def compute(self) -> Dict[str, float]:
        if self.num_samples == 0:
            raise RuntimeError("cannot compute metrics before any subjects are accumulated")
        result = {
            key: value / self.num_samples for key, value in self._sums.items()
        }
        if self.feature_extractor is not None:
            generated = torch.cat(self._generated_features, dim=0)
            real = torch.cat(self._real_features, dim=0)
            if generated.shape[0] < 2:
                raise RuntimeError(
                    "FID/IS are dataset-level metrics and require at least two "
                    "accumulated subjects"
                )
            result["fid"] = frechet_distance_from_features(generated, real)
            if self._generated_logits:
                result["is"] = inception_score_from_logits(
                    torch.cat(self._generated_logits, dim=0)
                )
            elif self.require_is_logits:
                raise RuntimeError(
                    "Inception Score requested but no generated logits were accumulated"
                )
        return result
