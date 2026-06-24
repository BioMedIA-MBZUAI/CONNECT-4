#!/usr/bin/env python3
"""
Precompute hypergraphs for all samples and save them to disk.
This avoids building hypergraphs on-the-fly during training.
"""
import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graphs.hypergraph import HypergraphBuilder
from data.dataset_precomputed import Connect4PrecomputedDataset

def precompute_hypergraph_for_sample(
    scan_id: str,
    precomputed_dir: Path,
    output_dir: Path,
    structure_to_roi_idx: Dict[int, int],
) -> bool:
    """
    Precompute hypergraph for a single sample.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        graphs_dir = precomputed_dir / "graphs"
        hypergraphs_dir = precomputed_dir / "hypergraphs"
        output_graphs_dir = output_dir / "hypergraphs"
        output_graphs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embeddings to get num_patches
        image_nodes_path = graphs_dir / f"{scan_id}_image_nodes.npy"
        if not image_nodes_path.exists():
            print(f"[WARNING] Missing image_nodes for {scan_id}, skipping...")
            return False
        
        image_nodes = np.load(image_nodes_path)
        num_patches = image_nodes.shape[0]
        num_rois = len(structure_to_roi_idx)
        
        # Load patch distributions
        dist_file = hypergraphs_dir / f"{scan_id}_patch_distributions.json"
        if not dist_file.exists():
            print(f"[WARNING] Missing patch_distributions for {scan_id}, skipping...")
            return False
        
        with open(dist_file, 'r') as f:
            patch_distributions = json.load(f)
            # Convert keys back to int
            patch_distributions = [
                {int(k): v for k, v in dist.items()}
                for dist in patch_distributions
            ]
        
        # Build hypergraph
        hypergraph_builder = HypergraphBuilder(
            num_patches=num_patches,
            num_rois=num_rois,
        )
        
        device = torch.device('cpu')
        hyperedge_index, hyperedge_weights = hypergraph_builder.build_hyperedges(
            patch_distributions=patch_distributions,
            structure_to_roi_idx=structure_to_roi_idx,
            device=device,
        )
        
        # Save hyperedge_index and hyperedge_weights
        hyperedge_index_path = output_graphs_dir / f"{scan_id}_hyperedge_index.pt"
        hyperedge_weights_path = output_graphs_dir / f"{scan_id}_hyperedge_weights.pt"
        
        torch.save(hyperedge_index, hyperedge_index_path)
        torch.save(hyperedge_weights, hyperedge_weights_path)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to precompute hypergraph for {scan_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute hypergraphs for all samples")
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default="/path/to/data/reconstructed_graphs",
        help="Directory containing precomputed graphs and embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/data/graphs_final",
        help="Output directory for precomputed hypergraphs",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/path/to/data",
        help="Root directory for data",
    )
    args = parser.parse_args()
    
    precomputed_dir = Path(args.precomputed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get structure_to_roi_idx from dataset
    # We need to instantiate the dataset to get ROI_SPECS
    print("[INFO] Loading dataset to get ROI specifications...")
    dataset = Connect4PrecomputedDataset(
        root_dir=args.root_dir,
        precomputed_dir=str(precomputed_dir),
        register_fmri_to_t1w=False,  # Skip registration
    )
    
    structure_to_roi_idx = dataset.structure_to_roi_idx
    scan_ids = dataset.scan_ids
    
    print(f"[INFO] Found {len(scan_ids)} samples to process")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Process all samples
    success_count = 0
    failed_count = 0
    
    for scan_id in tqdm(scan_ids, desc="Precomputing hypergraphs"):
        success = precompute_hypergraph_for_sample(
            scan_id=scan_id,
            precomputed_dir=precomputed_dir,
            output_dir=output_dir,
            structure_to_roi_idx=structure_to_roi_idx,
        )
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n[INFO] Completed: {success_count} successful, {failed_count} failed")
    print(f"[INFO] Hypergraphs saved to: {output_dir / 'hypergraphs'}")


if __name__ == "__main__":
    main()

