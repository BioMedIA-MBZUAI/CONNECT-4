"""
CONNECT-4 inference: synthesise 4D rs-fMRI from structural MRI.

    python -m inference.infer --config configs/connect4.yaml \
        --checkpoint checkpoints/connect4_epoch299.pt \
        --out_dir outputs --num 4 --visualize --metrics

For each subject it:
  1. builds the hypergraph patch tokens from the (frozen-FM) node features,
  2. generates the full 4D volume (DiT -> Temporal UNet),
  3. saves the synthetic volume as NIfTI,
  4. (optional) plots real vs synthetic in magma and prints evaluation metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset_precomputed import Connect4PrecomputedDataset
from data.collate import connect4_collate_fn
from models.connect4 import Connect4Model
from eval.metrics import compute_all
from eval.visualize import plot_real_vs_synthetic, save_fmri_nifti


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/connect4.yaml")
    ap.add_argument("--checkpoint", default=None, help="trained .pt (optional; random weights if omitted)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--num", type=int, default=4, help="number of subjects to synthesise")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Connect4Model(cfg).to(device).eval()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[infer] loaded checkpoint {args.checkpoint}")

    d = cfg["data"]
    ds = Connect4PrecomputedDataset(
        root_dir=d["root_dir"], precomputed_dir=d["precomputed_dir"],
        target_shape=tuple(d["target_shape"]), num_frames=int(d["num_frames"]),
        normalize_intensity=d.get("normalize_intensity", True),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=connect4_collate_fn)

    for i, batch in enumerate(loader):
        if i >= args.num:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        tokens = model.build_patch_tokens(batch, device)
        pred = model.generate(tokens, batch["t1w"], mask=batch.get("brain_mask"))  # [B,1,T,D,H,W]

        sid = batch.get("subject_id", [f"subj{i:03d}"])
        sid = sid[0] if isinstance(sid, (list, tuple)) else f"subj{i:03d}"
        save_fmri_nifti(pred, str(out_dir / f"{sid}_synthetic.nii.gz"))
        print(f"[infer] {sid}: synthesised {tuple(pred.shape)}")

        if (args.visualize or args.metrics) and batch.get("fmri") is not None:
            real = batch["fmri"]
            if args.visualize:
                plot_real_vs_synthetic(real, pred, str(out_dir / f"{sid}_real_vs_synthetic.png"))
            if args.metrics:
                m = compute_all(pred, real, roi_masks=batch.get("roi_masks"),
                                mask=batch.get("brain_mask"))
                print("   metrics:", {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
