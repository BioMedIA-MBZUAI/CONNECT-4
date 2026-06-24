"""
CONNECT-4 end-to-end model (Figure 1, panels A -> E).

This is the clean, figure-faithful orchestrator.  It wires together the genuine
component modules; it deliberately contains none of the diffusion-noise /
GAN / multi-decoder experimentation that accumulated in the research tree.

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
C) DiT generation        : DiT4DTemporal turns the patch tokens into one fMRI
                           frame per temporal index, conditioned (adaLN) on T1.
D) Temporal UNet         : TCUNet4DFiLM refines + upsamples the stacked
                           frames into the full 4D volume with temporal coupling.
E) Losses                : Connect4Loss (3D SSIM, voxel intensity, volume,
                           region-histogram, perceptual, FC-matrix).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from graphs import (
    ImageGraphBuilder, MaskGraphBuilder, ROIGraphBuilder, HypergraphBuilder,
)
from .fusion import MultiModalFusion
from .dit4d_temporal import DiT4DTemporal
from .tc_film_unet import TCUNet4DFiLM
from .losses import Connect4Loss


# --------------------------------------------------------------------------- #
# batch-tensor helpers (also used by training/inference for metrics)
# --------------------------------------------------------------------------- #
def roi_masks_to_tensor(roi_masks, target_spatial, device) -> Optional[torch.Tensor]:
    """
    Normalise the dataset's ROI masks into a dense [B, R, D, H, W] tensor.

    The precomputed dataset yields ROI masks as a (collated) list of dicts
    ``{roi_id: mask[D,H,W]}``; this stacks them (sorted by id) and resizes to the
    target grid with nearest-neighbour. Returns None if unavailable.
    """
    if roi_masks is None:
        return None
    if torch.is_tensor(roi_masks):
        t = roi_masks
        t = t.unsqueeze(0) if t.dim() == 4 else t
        return t.to(device).float()
    samples = roi_masks if isinstance(roi_masks, list) else [roi_masks]
    out = []
    for d in samples:
        if isinstance(d, dict):
            keys = sorted(d.keys())
            vols = [torch.as_tensor(d[k]).float().squeeze() for k in keys]   # each [D,H,W]
            stk = torch.stack(vols, dim=0)                                    # [R, D, H, W]
        else:
            stk = torch.as_tensor(d).float()
        out.append(stk)
    t = torch.stack(out, dim=0).to(device)                                   # [B, R, D, H, W]
    if tuple(t.shape[2:]) != tuple(target_spatial):
        B, R = t.shape[:2]
        t = F.interpolate(t.reshape(B * R, 1, *t.shape[2:]).float(),
                          size=tuple(target_spatial), mode="nearest").reshape(B, R, *target_spatial)
    return (t > 0.5).float()


def brain_mask_from_roi(roi_tensor: torch.Tensor) -> torch.Tensor:
    """Brain mask [B,1,D,H,W] = union of non-background ROIs."""
    fg = roi_tensor[:, 1:] if roi_tensor.shape[1] > 1 else roi_tensor
    return (fg.sum(dim=1, keepdim=True) > 0.5).float()


def resample_4d(vol: torch.Tensor, size_tdhw) -> torch.Tensor:
    """Resample [B,C,T,D,H,W] to a target (T,D,H,W) — spatial then temporal."""
    B, C, T, D, H, W = vol.shape
    Tt, Dt, Ht, Wt = size_tdhw
    if (D, H, W) != (Dt, Ht, Wt):
        vol = F.interpolate(vol.reshape(B * C * T, 1, D, H, W),
                            size=(Dt, Ht, Wt), mode="trilinear", align_corners=False
                            ).reshape(B, C, T, Dt, Ht, Wt)
    if T != Tt:
        # interpolate along time: [B*C, P, T] -> [B*C, P, Tt]
        v = F.interpolate(vol.reshape(B * C, T, -1).permute(0, 2, 1),
                          size=Tt, mode="linear", align_corners=False)
        vol = v.permute(0, 2, 1).reshape(B, C, Tt, Dt, Ht, Wt)
    return vol


class Connect4Model(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

        d = cfg["data"]
        m = cfg["models"]
        self.target_shape = tuple(d["target_shape"])          # (D, H, W)
        self.num_frames = int(d["num_frames"])                # T
        self.out_channels = int(d.get("out_channels", 1))
        self.num_rois = int(m["fusion"]["num_rois"])

        # ---- B) graph builders ------------------------------------------------
        gp = m["graphs"]
        self.image_graph_builder = ImageGraphBuilder(
            patch_size=tuple(gp["patch_size"]), k_neighbors=gp.get("k_neighbors", 5),
        )
        self.mask_graph_builder = MaskGraphBuilder(patch_size=tuple(gp["patch_size"]))
        self.roi_graph_builder = ROIGraphBuilder(num_rois=self.num_rois)
        num_patches = 1
        for s, p in zip(self.target_shape, gp["patch_size"]):
            num_patches *= s // p
        self.num_patches = num_patches
        self.hypergraph_builder = HypergraphBuilder(
            num_patches=num_patches, num_rois=self.num_rois,
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
            num_hypergraph_layers=f.get("num_hypergraph_layers", 2),
            num_heads=f.get("num_heads", 8),
            dropout=f.get("dropout", 0.1),
            num_rois=self.num_rois,
        )

        # ---- C) DiT generation -----------------------------------------------
        dit = m["dit"]
        self.dit = DiT4DTemporal(
            input_size=tuple(dit["input_size"]),              # (D, H, W) low-res
            in_channels=self.out_channels,
            patch_size=tuple(dit["patch_size"]),
            hidden_size=dit.get("hidden_size", 512),
            depth=dit.get("depth", 8),
            num_heads=dit.get("num_heads", 8),
            mlp_ratio=dit.get("mlp_ratio", 4.0),
            learn_sigma=False,
            input_dim=f["output_dim"],
            num_temporal_frames=self.num_frames,
            use_temporal_context=False,
        )
        self.dit_lowres = tuple(dit["input_size"])
        self.dit_checkpoint = bool(dit.get("use_gradient_checkpointing", False))
        self.decoder_chunk = int(m["unet"].get("temporal_chunk", 0))   # 0 = all frames at once
        # project a low-res T1 volume to the DiT conditioning vector (adaLN `y`)
        self.t1_cond = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(1, dit.get("hidden_size", 512)),
        )

        # ---- D) temporal UNet decoder (TC-UNet: Conv4d temporal+graph cond,
        #         temporal attention, MaxPool3d / ConvTranspose3d on H,W) --------
        u = m["unet"]
        self.graph_cond_dim = f["output_dim"]
        self.decoder = TCUNet4DFiLM(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            base_channels=u.get("base_channels", 64),
            num_levels=u.get("num_levels", 4),
            graph_cond_dim=self.graph_cond_dim,
            use_temporal_attn=u.get("use_temporal_attn", True),
            temporal_attn_heads=u.get("temporal_attn_heads", 4),
            use_checkpoint=u.get("use_checkpoint", False),
        )

        # ---- E) loss ----------------------------------------------------------
        lw = cfg["training"]["loss"]
        perceptual_fe = None
        if lw.get("perceptual_use_slimbrain", False):
            from .slimbrain_wrapper import SlimBrainEncoder
            perceptual_fe = SlimBrainEncoder(weights_path=lw.get("slimbrain_weights"))
        self.loss_fn = Connect4Loss(
            ssim_weight=lw.get("ssim", 1.0),
            voxel_weight=lw.get("voxel", 1.0),
            volume_weight=lw.get("volume", 0.5),
            region_hist_weight=lw.get("region_hist", 0.5),
            perceptual_weight=lw.get("perceptual", 0.1),
            fc_weight=lw.get("fc", 0.5),
            perceptual_feature_extractor=perceptual_fe,
        )

    # ----------------------------------------------------------------------- #
    # B) build the three graphs + hypergraph and fuse them into patch tokens
    # ----------------------------------------------------------------------- #
    def build_patch_tokens(self, batch: Dict, device: torch.device) -> torch.Tensor:
        image_nodes = batch["image_nodes"]        # [B, P, image_embed_dim]
        mask_nodes = batch["mask_nodes"]          # [B, P, mask_embed_dim]
        roi_nodes = batch["roi_nodes"]            # [B, R, roi_embed_dim]
        B = image_nodes.shape[0]
        dwi = batch["dwi_matrix"]
        s2r = batch["structure_to_roi_idx"]
        dists = batch["patch_distributions"]

        image_adj = self.image_graph_builder.build_adjacency(
            self.image_graph_builder.get_patch_positions(self.target_shape).to(device), device
        ).unsqueeze(0).expand(B, -1, -1)

        fused = []
        for b in range(B):
            dwi_b = dwi[b] if dwi.ndim == 3 else dwi
            dist_b = dists[b] if isinstance(dists, list) else dists
            mask_adj = self.mask_graph_builder.build_adjacency_from_dwi(
                mask_nodes.shape[1], dist_b, dwi_b, s2r
            ).unsqueeze(0)
            _, roi_adj = self.roi_graph_builder(roi_nodes[b:b + 1], dwi_matrix=dwi_b)
            if roi_adj.ndim == 2:
                roi_adj = roi_adj.unsqueeze(0)
            he_index, he_w = self.hypergraph_builder(dist_b, s2r)
            patch_ids = torch.arange(he_w.shape[0], device=device, dtype=torch.long)
            fused.append(self.fusion(
                image_nodes[b:b + 1], image_adj[b:b + 1],
                mask_nodes[b:b + 1], mask_adj,
                roi_nodes[b:b + 1], roi_adj,
                he_index.to(device), he_w.to(device), patch_ids,
                patch_distributions=dist_b, structure_to_roi_idx=s2r,
            ))
        return torch.cat(fused, dim=0)            # [B, P, output_dim]

    # ----------------------------------------------------------------------- #
    # C+D) generate the 4D volume from patch tokens + T1
    # ----------------------------------------------------------------------- #
    def generate(
        self,
        patch_tokens: torch.Tensor,
        t1: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = patch_tokens.shape[0]
        device = patch_tokens.device
        if t1.ndim == 4:
            t1 = t1.unsqueeze(1)                              # [B, 1, D, H, W]

        t1_low = F.interpolate(t1, size=self.dit_lowres, mode="trilinear", align_corners=False)
        cond = self.t1_cond(t1_low)                          # [B, hidden_size]

        # C) DiT: one frame per temporal index, conditioned on tokens + T1 (adaLN)
        frames = []
        for f_idx in range(self.num_frames):
            t = torch.full((B,), f_idx, device=device, dtype=torch.long)
            if self.dit_checkpoint and self.training:
                frame = torch.utils.checkpoint.checkpoint(
                    self.dit, patch_tokens, t, cond, use_reentrant=False)
            else:
                frame = self.dit(patch_tokens, t, y=cond)     # [B, C, d, h, w] (low-res)
            frames.append(frame)
        x = torch.stack(frames, dim=2)                        # [B, C, T, d, h, w]

        # D) TC-UNet refines at the DiT (low) resolution -- the 4D conv UNet at full
        #    128^3 x T would be far too large; we upsample the refined volume after.
        graph_cond = patch_tokens.mean(dim=1)                 # [B, output_dim] (Fig 1D "Graph Cond")
        mask_low = None
        if mask is not None:
            m = mask if mask.dim() == 5 else mask.unsqueeze(1)
            mask_low = F.interpolate(m.float(), size=self.dit_lowres, mode="nearest")
        # process frames in temporal chunks so the 4D UNet memory scales with the
        # chunk size, not the full 128 frames -> lets us refine at higher spatial
        # resolution (e.g. 32^3) without OOM.
        cs = self.decoder_chunk
        if cs and cs < self.num_frames:
            def _decode_chunk(xc, tidx):
                return self.decoder(x=xc, graph_cond=graph_cond, temporal_indices=tidx, mask=mask_low)
            outs = []
            for s in range(0, self.num_frames, cs):
                xc = x[:, :, s:s + cs]                        # [B, C, cs, d, h, w]
                tidx = torch.arange(s, s + xc.shape[2], device=device).unsqueeze(0).expand(B, -1).float()
                # checkpoint each chunk so backward recomputes it -> memory is bounded
                # by ONE chunk, not all 128 frames (forward-chunking alone keeps every
                # chunk's graph alive through the cat).
                if self.training:
                    oc = torch.utils.checkpoint.checkpoint(_decode_chunk, xc, tidx, use_reentrant=False)
                else:
                    oc = _decode_chunk(xc, tidx)
                outs.append(oc)
            out = torch.cat(outs, dim=1)                      # [B, T, d, h, w]
        else:
            out = self.decoder(x=x, graph_cond=graph_cond, temporal_indices=None, mask=mask_low)

        # upsample the refined 4D volume to the target grid (cheap trilinear)
        D, H, W = self.target_shape
        out = F.interpolate(
            out.reshape(B * self.num_frames, 1, *out.shape[2:]),
            size=self.target_shape, mode="trilinear", align_corners=False,
        ).reshape(B, 1, self.num_frames, D, H, W)
        return out                                            # [B, 1, T, D, H, W]

    def forward(self, batch: Dict) -> Dict:
        device = next(self.parameters()).device
        roi_masks = roi_masks_to_tensor(batch.get("roi_masks"), self.target_shape, device)
        brain_mask = batch.get("brain_mask")
        if brain_mask is None and roi_masks is not None:
            brain_mask = brain_mask_from_roi(roi_masks)

        patch_tokens = self.build_patch_tokens(batch, device)
        pred = self.generate(patch_tokens, batch["t1w"].to(device), mask=brain_mask)

        out = {"prediction": pred}
        if batch.get("fmri") is not None:
            target = batch["fmri"].to(device)
            if target.ndim == 5:
                target = target.unsqueeze(1)                  # [B, 1, T, D, H, W]
            target = resample_4d(target, pred.shape[2:])      # match (T, D, H, W)
            out["losses"] = self.loss_fn(
                pred, target, mask=brain_mask, roi_masks=roi_masks,
            )
            out["target"] = target
        out["roi_masks"] = roi_masks
        out["brain_mask"] = brain_mask
        return out
