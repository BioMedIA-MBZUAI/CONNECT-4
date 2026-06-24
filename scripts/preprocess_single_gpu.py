#!/usr/bin/env python3
"""
Single-GPU preprocessing script (no multiprocessing/spawn).
Processes scans sequentially on one GPU, reusing the same models.
"""
import argparse
import sys
import os
from pathlib import Path
import time

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd

from utils.config import load_config
from data.dataset import Connect4Dataset
from models.brainiac_wrapper import BrainIACWrapper
from models.modernbert_wrapper import ModernBERTWrapper
from graphs.image_graph import ImageGraphBuilder
from graphs.mask_graph import MaskGraphBuilder
from graphs.roi_graph import ROIGraphBuilder
from graphs.hypergraph import HypergraphBuilder
# Import the shared single-sample routine
# Note: we already inserted PROJECT_ROOT at sys.path[0]
from preprocess_graphs import process_single_sample


def main():
    parser = argparse.ArgumentParser(description="Single-GPU preprocessing (no multiprocessing).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--brainiac_batch_size", type=int, default=32, help="Batch size for BrainIAC; lower if OOM.")
    parser.add_argument("--text_batch_size", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=0, help="Start index (inclusive) in scan list.")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive) in scan list.")
    parser.add_argument("--shard_index", type=int, default=0, help="Shard index (0-based) for multi-job splitting.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards (jobs).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_emb").mkdir(exist_ok=True)
    (out_dir / "modernbert_emb").mkdir(exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)
    (out_dir / "hypergraphs").mkdir(exist_ok=True)

    # Speed settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    device = torch.device(args.device)

    # Load config and dataset
    config = load_config(args.config)
    dataset = Connect4Dataset(
        root_dir=config["data"]["root_dir"],
        patch_size=tuple(config["data"]["patch_size"]),
        target_shape=tuple(config["data"]["target_shape"]),
        dwi_matrix_path=config["data"]["dwi_matrix_path"],
        normalize_intensity=config["data"]["normalize_intensity"],
        brainiac_model_path=config["models"].get("brainiac_path"),
        modernbert_model_path=config["models"].get("modernbert_path"),
    )

    # Instantiate models once
    brainiac = BrainIACWrapper(
        model_path=config["models"].get("brainiac_path"),
        embed_dim=config["models"]["brainiac"]["embed_dim"],
        device=str(device),
    ).to(device)
    brainiac.eval()

    modernbert = ModernBERTWrapper(
        model_path=config["models"].get("modernbert_path"),
        embed_dim=config["models"]["modernbert"]["embed_dim"],
    ).to(device)
    modernbert.eval()

    # Builders
    patch_size = tuple(config["data"]["patch_size"])
    num_rois = config["data"]["num_rois"]
    num_patches = np.prod([config["data"]["target_shape"][i] // patch_size[i] for i in range(3)])

    image_graph_builder = ImageGraphBuilder(
        patch_size=patch_size,
        k_neighbors=config["graphs"]["image"]["k_neighbors"],
    )
    mask_graph_builder = MaskGraphBuilder(
        patch_size=patch_size,
    )
    roi_graph_builder = ROIGraphBuilder(
        num_rois=num_rois,
    )
    hypergraph_builder = HypergraphBuilder(
        num_patches=num_patches,
        num_rois=num_rois,
    )

    # Select slice of scans then shard by modulo
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(dataset)
    base_indices = list(range(start, min(end, len(dataset))))
    shard_indices = [
        i for i in base_indices
        if (i % args.num_shards) == args.shard_index
    ]
    print(
        f"Running single-GPU preprocessing on {len(shard_indices)} scans "
        f"(indices {start}..{end - 1}), shard {args.shard_index}/{args.num_shards}",
        flush=True,
    )

    all_texts = []
    t_all_start = time.time()
    for i, idx in enumerate(shard_indices):
        scan_id = dataset.scan_ids[idx]
        res = process_single_sample(
            idx,
            scan_id,
            dataset,
            out_dir,
            brainiac,
            modernbert,
            image_graph_builder,
            mask_graph_builder,
            roi_graph_builder,
            hypergraph_builder,
            device,
            brainiac_batch_size=args.brainiac_batch_size,
            text_batch_size=args.text_batch_size,
        )
        if res:
            all_texts.extend(res)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_all_start
            print(f"Processed {i+1}/{len(shard_indices)} scans in {elapsed/60:.2f} min", flush=True)

    # Save combined text descriptions
    if all_texts:
        df = pd.DataFrame(all_texts)
        df.to_csv(out_dir / "patch_text_descriptions.csv", index=False)
        print(f"Saved {len(all_texts)} text descriptions to patch_text_descriptions.csv", flush=True)

    print(f"Done. Total time: {(time.time() - t_all_start)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()

