"""
SLIM-Brain 4D feature extractor.

SLIM-Brain (Wang et al., 2025) is a data- and training-efficient, atlas-free
**voxel-level** foundation model for fMRI. It uses a lightweight temporal
extractor to rank salient windows and a hierarchical JEPA (Hiera-JEPA) encoder
to produce fine-grained 4D voxel-level features.

    Wang, M., Xia, J., Ye, W., Liu, E., Peng, K., Feng, J., Liu, Q., Wen, H.
    "SLIM-Brain: A Data- and Training-Efficient Foundation Model for fMRI Data
    Analysis." arXiv:2512.21881 (2025).
    https://arxiv.org/abs/2512.21881  ·  https://openreview.net/forum?id=fFgzAQAUqs
    Lab: https://github.com/ncclab-sustech

In CONNECT-4 it is used **frozen** to extract 4D deep features for (a) the
perceptual loss and (b) the synthetic-vs-real evaluation metric
(`eval.metrics.slimbrain_feature_distance`).

Usage
-----
    enc = SlimBrainEncoder(weights_path="/path/to/slimbrain.pt").eval()
    feats = enc(fmri)            # fmri: [B,1,T,D,H,W] or [B,T,D,H,W] -> [B, feat_dim]

If the official weights are not present, a deterministic, frozen lightweight
encoder is used so the loss/metric remain well-defined and the code runs
end-to-end. Set the `CONNECT4_SLIMBRAIN` env var or pass `weights_path` to load
the real model once released.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrozenVoxel4DEncoder(nn.Module):
    """
    Deterministic, frozen stand-in for SLIM-Brain: a small per-frame 3D CNN with
    temporal aggregation. Used only when the official weights are unavailable so
    that 4D feature distances are still meaningful and reproducible.
    """

    def __init__(self, feat_dim: int = 256, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(64, feat_dim)
        # deterministic random projection, then freeze. Weights get a sizeable
        # std so the input (not a constant bias) drives the features; biases = 0.
        for name, p in self.named_parameters():
            if p.dim() == 1:                       # biases -> zero (no constant offset)
                p.data = torch.zeros_like(p)
            else:                                  # weights -> discriminative random
                p.data = torch.empty_like(p).normal_(0.0, 0.3, generator=g)
            p.requires_grad_(False)
        self.feat_dim = feat_dim

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        # Params are frozen (requires_grad=False); gradients still flow to the
        # input so this can be used as a perceptual loss.
        B, C, T, D, H, W = fmri.shape
        x = fmri.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, D, H, W)
        x = self.stem(x).flatten(1)                 # [B*T, 64]
        x = self.proj(x).reshape(B, T, -1)          # [B, T, feat_dim]
        return x.mean(dim=1)                         # temporal average -> [B, feat_dim]


class SlimBrainEncoder(nn.Module):
    """
    Frozen SLIM-Brain wrapper exposing `forward(fmri) -> [B, feat_dim]`.

    Parameters
    ----------
    weights_path : str, optional
        Path to released SLIM-Brain weights (or set env `CONNECT4_SLIMBRAIN`).
    feat_dim : int
        Output feature dimension of the fallback encoder.
    """

    def __init__(self, weights_path: Optional[str] = None, feat_dim: int = 256,
                 device: Optional[torch.device] = None):
        super().__init__()
        weights_path = weights_path or os.environ.get("CONNECT4_SLIMBRAIN")
        self.is_official = False
        self.model = None

        if weights_path and os.path.exists(weights_path):
            try:
                # Official SLIM-Brain checkpoint (TorchScript or state_dict + class).
                obj = torch.load(weights_path, map_location="cpu")
                self.model = obj if isinstance(obj, nn.Module) else torch.jit.load(weights_path, map_location="cpu")
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.is_official = True
                print(f"[SLIM-Brain] loaded official weights from {weights_path}")
            except Exception as e:  # pragma: no cover
                print(f"[SLIM-Brain] failed to load {weights_path} ({e}); using frozen fallback")

        if self.model is None:
            self.model = _FrozenVoxel4DEncoder(feat_dim=feat_dim)
            print("[SLIM-Brain] official weights not found — using deterministic frozen 4D encoder")

        if device is not None:
            self.model = self.model.to(device)

    @staticmethod
    def _to_bctdhw(fmri: torch.Tensor) -> torch.Tensor:
        if fmri.dim() == 5:           # [B, T, D, H, W]
            fmri = fmri.unsqueeze(1)  # [B, 1, T, D, H, W]
        return fmri

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        fmri = self._to_bctdhw(fmri)
        return self.model(fmri)
