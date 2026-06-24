#!/usr/bin/env python3
"""
Preprocessing script to pre-compute and save all graphs and embeddings.
Multi-GPU support for parallel processing.
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path - MUST be done before any other imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
SCRIPTS_DIR = str(PROJECT_ROOT / "scripts")
NEUROVFM_PATH = str(PROJECT_ROOT / "neurovfm")

# Remove scripts directory from path if it's there (can cause conflicts)
if SCRIPTS_DIR in sys.path:
    sys.path.remove(SCRIPTS_DIR)

# Remove neurovfm from path temporarily (we'll add it back after our imports)
if NEUROVFM_PATH in sys.path:
    sys.path.remove(NEUROVFM_PATH)

# Ensure PROJECT_ROOT is FIRST in path (before neurovfm to avoid namespace conflicts)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

# Add neurovfm to path AFTER project root (so our 'data' package is found first)
if NEUROVFM_PATH not in sys.path:
    sys.path.insert(1, NEUROVFM_PATH)  # Insert at position 1, not 0

# Set multiprocessing start method BEFORE importing torch to avoid TORCH_LIBRARY conflicts
# This must be done before torch is imported
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Set environment variable to prevent triton registration conflicts in spawned processes
# This must be set before torch is imported
os.environ['TRITON_INTERPRET'] = '1'  # Use interpreter mode to avoid registration conflicts

# Now import standard libraries (but delay torch import if we're in a worker process)
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import time
from functools import partial

# Delay torch import - it will be imported in worker_process after CUDA_VISIBLE_DEVICES is set
# For the main process, we'll import it normally
if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.getenv('DELAY_TORCH_IMPORT') != '1':
    import torch
    # Enable faster matmul on supported GPUs (TF32)
    if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
else:
    # We're in a worker process, torch will be imported later
    torch = None

# Verify path setup before importing project modules
dataset_path = os.path.join(PROJECT_ROOT_STR, "data", "dataset.py")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"data/dataset.py not found at {dataset_path}. "
        f"Current working directory: {os.getcwd()}, PROJECT_ROOT: {PROJECT_ROOT_STR}, "
        f"sys.path[0]: {sys.path[0] if sys.path else 'empty'}"
    )

# Now import project modules - import packages first, then modules
try:
    # Import packages first to ensure they're recognized
    import data
    import models
    import graphs
    import utils
    
    # Now import the specific modules
    from data.dataset import Connect4Dataset
    from data.patchify import Patchify3D
    from models.brainiac_wrapper import BrainIACWrapper
    from models.modernbert_wrapper import ModernBERTWrapper
    from graphs.image_graph import ImageGraphBuilder
    from graphs.mask_graph import MaskGraphBuilder
    from graphs.roi_graph import ROIGraphBuilder
    from graphs.hypergraph import HypergraphBuilder
    from utils.config import load_config
    from utils.scalers import FeatureScalerManager
except ImportError as e:
    # Print debug info
    print(f"ERROR: Failed to import modules. PROJECT_ROOT: {PROJECT_ROOT_STR}", file=sys.stderr)
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    print(f"sys.path: {sys.path[:5]}", file=sys.stderr)
    print(f"Files in data/: {os.listdir(os.path.join(PROJECT_ROOT_STR, 'data')) if os.path.exists(os.path.join(PROJECT_ROOT_STR, 'data')) else 'data/ does not exist'}", file=sys.stderr)
    print(f"Trying to import data package directly...", file=sys.stderr)
    try:
        import importlib
        data_spec = importlib.util.find_spec("data")
        print(f"data spec: {data_spec}", file=sys.stderr)
        if data_spec:
            print(f"data spec.origin: {data_spec.origin}", file=sys.stderr)
            print(f"data spec.submodule_search_locations: {data_spec.submodule_search_locations}", file=sys.stderr)
    except Exception as e2:
        print(f"Error checking data spec: {e2}", file=sys.stderr)
    raise


def save_tensor(tensor: torch.Tensor, path: Path):
    """Save tensor as numpy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), tensor.cpu().numpy())


def load_tensor(path: Path) -> torch.Tensor:
    """Load tensor from numpy array."""
    return torch.from_numpy(np.load(str(path)))


def process_single_sample(
    idx: int,
    scan_id: str,
    dataset: Connect4Dataset,
    output_dir: Path,
    brainiac: BrainIACWrapper,
    modernbert: ModernBERTWrapper,
    image_graph_builder: ImageGraphBuilder,
    mask_graph_builder: MaskGraphBuilder,
    roi_graph_builder: ROIGraphBuilder,
    hypergraph_builder: HypergraphBuilder,
    device: torch.device,
    brainiac_batch_size: int = 8,  # Very small default - patches resize from 16^3 to 96^3 (216x memory increase)
    text_batch_size: int = 512,
    scaler_manager: FeatureScalerManager = None,
):
    """Process a single sample."""
    import sys
    try:
        t_total_start = time.time()
        sys.stdout.flush()
        # Create output directories
        image_emb_dir = output_dir / "image_emb"
        modernbert_emb_dir = output_dir / "modernbert_emb"
        graphs_dir = output_dir / "graphs"
        hypergraphs_dir = output_dir / "hypergraphs"
        
        scan_base = scan_id.split("_")[0]
        
        # Check if already processed (skip to speed up)
        image_nodes_path = graphs_dir / f"{scan_id}_image_nodes.npy"
        image_emb_path = image_emb_dir / f"{scan_id}_image_patch_embeddings.npy"
        modernbert_emb_path = modernbert_emb_dir / f"{scan_id}_mask_patch_embeddings.npy"
        if image_nodes_path.exists() and image_emb_path.exists() and modernbert_emb_path.exists():
            mask_nodes_path = graphs_dir / f"{scan_id}_mask_nodes.npy"
            roi_nodes_path = graphs_dir / f"{scan_id}_roi_nodes.npy"
            hyperedge_path = hypergraphs_dir / f"{scan_id}_hyperedge_index.npy"
            if all(p.exists() for p in [mask_nodes_path, roi_nodes_path, hyperedge_path]):
                print(f"  [{scan_id}] ✓ Already processed, skipping...", flush=True)
                return None  # Already processed
        
        # Load sample
        sample = dataset[idx]
        
        t1w = sample['t1w']  # [1, 1, D, H, W]
        mask = sample['mask']  # [1, 1, D, H, W]
        roi_embeddings = sample['roi_embeddings']  # [num_rois, embed_dim]
        patch_distributions = sample['patch_distributions']
        structure_to_roi_idx = sample['structure_to_roi_idx']
        dwi_matrix = sample['dwi_matrix']
        
        # Ensure correct dimensions
        if t1w.dim() == 5 and t1w.shape[0] == 1:
            t1w = t1w.squeeze(0)  # [1, D, H, W]
        if mask.dim() == 5 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # [1, D, H, W]
        
        # Move to device
        print(f"  [{scan_id}] Moving data to device...", flush=True)
        t1w = t1w.to(device)
        mask = mask.to(device)
        
        # 1. Patchify T1w and encode with BrainIAC (batch processing)
        t_image_pipe_start = time.time()
        t_bria_start = time.time()
        # With 16x16x16 patches on 128x128x128 image = 512 patches (8x8x8 = 512)
        print(f"  [{scan_id}] Patchifying T1w image...", flush=True)
        t1w_patches = dataset.patchifier.patchify(t1w.unsqueeze(0))  # List of patches
        num_patches = len(t1w_patches)
        print(f"  [{scan_id}] Created {num_patches} patches", flush=True)
        
        image_patch_embeddings = []
        
        # OPTIMIZED: Process in batches with mixed precision and reduced synchronization
        print(f"  [{scan_id}] Starting BrainIAC batch encoding (batch_size={brainiac_batch_size})...", flush=True)
        
        # Use mixed precision for faster processing and lower memory
        use_amp = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
        
        with torch.no_grad():
            # Process in batches
            num_batches = (num_patches + brainiac_batch_size - 1) // brainiac_batch_size
            for batch_idx, batch_start in enumerate(range(0, num_patches, brainiac_batch_size)):
                if batch_idx % 10 == 0 and batch_idx > 0:
                    print(f"  [{scan_id}] BrainIAC batch {batch_idx+1}/{num_batches}", flush=True)
                
                batch_end = min(batch_start + brainiac_batch_size, num_patches)
                batch_patches_list = t1w_patches[batch_start:batch_end]
                
                # Try to process multiple patches at once if batch_size > 1
                if brainiac_batch_size > 1:
                    try:
                        # Stack patches and process as batch
                        batch_tensor = torch.stack([p.to(device) for p in batch_patches_list])  # [B, C, D, H, W]
                        
                        # Encode batch
                        if use_amp:
                            with torch.autocast("cuda"):
                                batch_embs = brainiac.encode(batch_tensor)  # [B, embed_dim]
                        else:
                            batch_embs = brainiac.encode(batch_tensor)
                        
                        # Move to CPU
                        batch_embs_cpu = batch_embs.cpu()
                        image_patch_embeddings.append(batch_embs_cpu)
                        
                        # Cleanup
                        del batch_tensor, batch_embs, batch_embs_cpu
                        # Only clear cache every 10 batches to reduce overhead
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Fallback to one-by-one if batch fails
                            print(f"  [{scan_id}] OOM with batch_size={brainiac_batch_size}, falling back to one-by-one", flush=True)
                            torch.cuda.empty_cache()
                            for patch in batch_patches_list:
                                patch_tensor = patch.unsqueeze(0).to(device)
                                patch_emb = brainiac.encode(patch_tensor)
                                if patch_emb.dim() == 1:
                                    patch_emb = patch_emb.unsqueeze(0)
                                image_patch_embeddings.append(patch_emb.cpu())
                                del patch_tensor, patch_emb
                        else:
                            raise
                else:
                    # Process one at a time (original approach, but without expensive sync)
                    for patch in batch_patches_list:
                        patch_tensor = patch.unsqueeze(0).to(device)
                        patch_emb = brainiac.encode(patch_tensor)
                        if patch_emb.dim() == 1:
                            patch_emb = patch_emb.unsqueeze(0)
                        image_patch_embeddings.append(patch_emb.cpu())
                        del patch_tensor, patch_emb
                    
                    # Only clear cache every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
        
        # Stack all embeddings
        print(f"  [{scan_id}] BrainIAC encoding complete, stacking embeddings...", flush=True)
        image_patch_embeddings = torch.cat(image_patch_embeddings, dim=0)  # [num_patches, embed_dim]
        
        # Apply scaler if available
        if scaler_manager is not None:
            print(f"  [{scan_id}] Applying BrainIAC scaler...", flush=True)
            image_patch_embeddings = scaler_manager.scale_brainiac(image_patch_embeddings)
        
        print(f"  [{scan_id}] BrainIAC total time: {time.time() - t_bria_start:.2f}s", flush=True)
        
        # Save all patch embeddings efficiently in a single file
        print(f"  [{scan_id}] Saving {num_patches} BrainIAC embeddings (single file)...", flush=True)
        emb_path = image_emb_dir / f"{scan_id}_image_patch_embeddings.npy"
        if not emb_path.exists():
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_path), image_patch_embeddings.numpy())
        
        # 2. Build image graph
        t_img_graph_start = time.time()
        print(f"  [{scan_id}] Building image graph...", flush=True)
        spatial_shape = tuple(dataset.target_shape)
        image_nodes, image_adj = image_graph_builder(
            image_patch_embeddings.unsqueeze(0),
            spatial_shape,
        )
        
        # Save image graph
        print(f"  [{scan_id}] Saving image graph...", flush=True)
        save_tensor(image_nodes.squeeze(0), graphs_dir / f"{scan_id}_image_nodes.npy")
        save_tensor(image_adj.squeeze(0), graphs_dir / f"{scan_id}_image_adj.npy")
        print(f"  [{scan_id}] Image graph time: {time.time() - t_img_graph_start:.2f}s", flush=True)
        print(f"  [{scan_id}] Image pipeline total: {time.time() - t_image_pipe_start:.2f}s", flush=True)
        
        # 3. Process mask patches and create text descriptions
        t_mask_pipe_start = time.time()
        print(f"  [{scan_id}] Patchifying mask and computing distributions...", flush=True)
        t_mask_text_start = time.time()
        mask_patches = dataset.patchifier.patchify(mask.unsqueeze(0))
        
        patch_text_descriptions = []
        mask_patch_embeddings_list = []
        
        # Load mask header for center of mass
        import nibabel as nib
        mask_path = dataset.root / "Masks" / f"{scan_id}_mask.nii.gz"
        mask_img = nib.load(str(mask_path))
        orig_shape = mask_img.shape[:3]
        affine = mask_img.affine
        
        text_descriptions = []
        for patch_idx in range(num_patches):
            dist = patch_distributions[patch_idx]
            
            # Compute center of mass
            x_mm, y_mm, z_mm = dataset._compute_patch_center_mm(
                patch_idx,
                orig_shape=orig_shape,
                affine=affine,
            )
            
            # Create text description
            desc_parts = [
                f"Patch {patch_idx} at center ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm "
                f"contains {len(dist)} structures:"
            ]
            
            for struct_id, coverage in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                struct_slug = dataset.ID_TO_SLUG.get(struct_id, f"structure_{struct_id}")
                struct_name_readable = struct_slug.replace("_", " ")
                base_text = f"{struct_name_readable} ({coverage*100:.1f}%)."
                
                # Append normative description
                norm_key = (scan_base, struct_slug)
                norm_text = dataset.normative_descriptions.get(norm_key, "")
                if norm_text:
                    desc_parts.append(f"{base_text} {norm_text}")
                else:
                    desc_parts.append(base_text)
            
            text_desc = " ".join(desc_parts)
            patch_text_descriptions.append(text_desc)
            
            # Store for CSV
            text_descriptions.append({
                'scan_id': scan_id,
                'patch_idx': patch_idx,
                'description': text_desc,
                'num_structures': len(dist),
                'center_x_mm': x_mm,
                'center_y_mm': y_mm,
                'center_z_mm': z_mm,
            })
        
        # Save raw patch descriptions (pre-ModernBERT) for inspection/reuse
        patch_desc_path = hypergraphs_dir / f"{scan_id}_patch_descriptions.json"
        if not patch_desc_path.exists():
            print(f"  [{scan_id}] Saving raw patch descriptions...", flush=True)
            with open(patch_desc_path, 'w') as f:
                json.dump(patch_text_descriptions, f, indent=2)
        
        # 4. Encode text descriptions with ModernBERT
        print(f"  [{scan_id}] Encoding {len(patch_text_descriptions)} text descriptions with ModernBERT...", flush=True)
        t_modern_start = time.time()
        num_text_batches = (len(patch_text_descriptions) + text_batch_size - 1) // text_batch_size
        with torch.no_grad():
            for i in range(0, len(patch_text_descriptions), text_batch_size):
                batch_idx = i // text_batch_size
                if batch_idx % 5 == 0:
                    print(f"  [{scan_id}] ModernBERT batch {batch_idx+1}/{num_text_batches}", flush=True)
                
                batch_texts = patch_text_descriptions[i:i+text_batch_size]
                if hasattr(modernbert, 'forward'):
                    batch_embs = modernbert.forward(batch_texts)  # [batch_size, embed_dim]
                    mask_patch_embeddings_list.append(batch_embs.cpu())
                else:
                    for desc in batch_texts:
                        emb = modernbert.encode(desc)
                        mask_patch_embeddings_list.append(emb.cpu())
        
        if isinstance(mask_patch_embeddings_list[0], torch.Tensor) and mask_patch_embeddings_list[0].dim() == 2:
            mask_patch_embeddings = torch.cat(mask_patch_embeddings_list, dim=0)
        else:
            mask_patch_embeddings = torch.stack(mask_patch_embeddings_list)
        
        # Apply scaler if available
        if scaler_manager is not None:
            print(f"  [{scan_id}] Applying ModernBERT scaler...", flush=True)
            mask_patch_embeddings = scaler_manager.scale_modernbert(mask_patch_embeddings)
        
        print(f"  [{scan_id}] ModernBERT total time: {time.time() - t_modern_start:.2f}s", flush=True)
        print(f"  [{scan_id}] Mask/text prep total time: {time.time() - t_mask_text_start:.2f}s", flush=True)
        
        # Save all ModernBERT embeddings efficiently in a single file
        print(f"  [{scan_id}] Saving {len(mask_patch_embeddings)} ModernBERT embeddings (single file)...", flush=True)
        modernbert_emb_path = modernbert_emb_dir / f"{scan_id}_mask_patch_embeddings.npy"
        if not modernbert_emb_path.exists():
            modernbert_emb_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_np = mask_patch_embeddings.numpy() if isinstance(mask_patch_embeddings, torch.Tensor) else mask_patch_embeddings
            np.save(str(modernbert_emb_path), embeddings_np)
        
        # 5. Build mask graph
        t_mask_graph_start = time.time()
        print(f"  [{scan_id}] Building mask graph...", flush=True)
        structure_labels = {
            label_id: slug
            for slug, label_id in dataset.ROI_SPECS
        }
        mask_nodes, mask_adj = mask_graph_builder(
            mask_patch_embeddings.unsqueeze(0),
            mask.unsqueeze(0),
            spatial_shape,
            structure_labels,
            dwi_matrix=dwi_matrix,
            structure_to_roi_idx=structure_to_roi_idx,
        )
        
        print(f"  [{scan_id}] Mask graph time: {time.time() - t_mask_graph_start:.2f}s", flush=True)
        # Save mask graph
        save_tensor(mask_nodes.squeeze(0), graphs_dir / f"{scan_id}_mask_nodes.npy")
        save_tensor(mask_adj.squeeze(0), graphs_dir / f"{scan_id}_mask_adj.npy")
        
        # 6. Build ROI graph (roi_embeddings already has scalers applied from dataset)
        t_roi_graph_start = time.time()
        roi_nodes, roi_adj = roi_graph_builder(
            roi_embeddings.unsqueeze(0),
            dwi_matrix=dwi_matrix,
        )
        print(f"  [{scan_id}] ROI graph time: {time.time() - t_roi_graph_start:.2f}s", flush=True)
        
        # Save ROI graph
        save_tensor(roi_nodes.squeeze(0), graphs_dir / f"{scan_id}_roi_nodes.npy")
        save_tensor(roi_adj.squeeze(0), graphs_dir / f"{scan_id}_roi_adj.npy")
        save_tensor(roi_embeddings, graphs_dir / f"{scan_id}_roi_embeddings.npy")
        
        # 7. Build hypergraph
        t_hyper_start = time.time()
        hyperedge_index, hyperedge_weights = hypergraph_builder(
            patch_distributions,
            structure_to_roi_idx,
        )
        print(f"  [{scan_id}] Hypergraph time: {time.time() - t_hyper_start:.2f}s", flush=True)
        
        # Save hypergraph
        print(f"  [{scan_id}] Saving hypergraph...", flush=True)
        save_tensor(hyperedge_index, hypergraphs_dir / f"{scan_id}_hyperedge_index.npy")
        save_tensor(hyperedge_weights, hypergraphs_dir / f"{scan_id}_hyperedge_weights.npy")
        
        # Save patch distributions as JSON
        print(f"  [{scan_id}] Saving patch distributions...", flush=True)
        patch_distributions_json = [
            {str(k): float(v) for k, v in dist.items()}
            for dist in patch_distributions
        ]
        with open(hypergraphs_dir / f"{scan_id}_patch_distributions.json", 'w') as f:
            json.dump(patch_distributions_json, f, indent=2)
        
        # Save metadata
        metadata = {
            'scan_id': scan_id,
            'num_patches': num_patches,
            'num_rois': dataset.NUM_ROIS,
            'spatial_shape': list(spatial_shape),
            'patch_size': list(dataset.patch_size),
        }
        with open(hypergraphs_dir / f"{scan_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  [{scan_id}] ✓ Sample processing complete!", flush=True)
        print(f"  [{scan_id}] Mask/ROI/Hypergraph pipeline total: {time.time() - t_mask_pipe_start:.2f}s", flush=True)
        print(f"  [{scan_id}] Total sample time: {time.time() - t_total_start:.2f}s", flush=True)
        return text_descriptions
        
    except Exception as e:
        print(f"Error processing {scan_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def init_worker(gpu_id):
    """Initialize worker process - sets CUDA_VISIBLE_DEVICES before any torch imports."""
    import os
    # This runs BEFORE the worker_process function, so CUDA_VISIBLE_DEVICES is set
    # before torch gets imported at module level in the spawned process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[Worker init] GPU {gpu_id}: Set CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)


def worker_process(
    gpu_id: int,
    sample_indices: List[int],
    dataset_config: Dict,
    output_dir: Path,
    brainiac_batch_size: int,
    text_batch_size: int,
):
    """Worker process for a single GPU."""
    import sys
    import os
    
    # CUDA_VISIBLE_DEVICES should already be set by init_worker, but verify
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] != str(gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Force flush output immediately
    sys.stdout.flush()
    sys.stderr.flush()
    
    print(f"[GPU {gpu_id}] Starting worker process for {len(sample_indices)} samples", flush=True)
    print(f"[GPU {gpu_id}] PID: {os.getpid()}", flush=True)
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
    
    # In spawn mode, each process starts fresh, so torch will be imported fresh
    # CUDA_VISIBLE_DEVICES is already set above, so torch will only see the assigned GPU
    # DO NOT reload torch - that causes triton registration conflicts
    import torch
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print(f"[GPU {gpu_id}] ERROR: CUDA not available in worker!", flush=True)
        return []
    
    # Set GPU device - in spawn mode with CUDA_VISIBLE_DEVICES set, each process sees only one GPU
    # So we use cuda:0 in each worker (which maps to the actual GPU assigned via CUDA_VISIBLE_DEVICES)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    actual_gpu_id = torch.cuda.current_device()
    print(f"[GPU {gpu_id}] Using device: {device} (CUDA_VISIBLE_DEVICES={gpu_id}, actual device={actual_gpu_id})", flush=True)
    
    # Create dataset (each worker needs its own)
    print(f"[GPU {gpu_id}] Creating dataset...", flush=True)
    dataset = Connect4Dataset(
        root_dir=dataset_config['root_dir'],
        patch_size=tuple(dataset_config['patch_size']),
        target_shape=tuple(dataset_config['target_shape']),
        dwi_matrix_path=dataset_config['dwi_matrix_path'],
        normalize_intensity=dataset_config['normalize_intensity'],
        brainiac_model_path=dataset_config.get('brainiac_model_path'),
        modernbert_model_path=dataset_config['modernbert_model_path'],
    )
    print(f"[GPU {gpu_id}] Dataset created with {len(dataset)} samples", flush=True)
    
    # Initialize models on this GPU
    print(f"[GPU {gpu_id}] Loading BrainIAC model...", flush=True)
    brainiac = BrainIACWrapper(
        model_path=dataset_config.get('brainiac_model_path'),
        embed_dim=dataset_config.get('brainiac_embed_dim', 768),
        device=str(device),
    ).to(device)
    brainiac.eval()  # Set to eval mode
    print(f"[GPU {gpu_id}] BrainIAC loaded", flush=True)
    
    print(f"[GPU {gpu_id}] Loading ModernBERT model...", flush=True)
    modernbert = ModernBERTWrapper(
        model_path=dataset_config['modernbert_model_path'],
        embed_dim=dataset_config['modernbert_embed_dim'],
    ).to(device)
    print(f"[GPU {gpu_id}] ModernBERT loaded", flush=True)
    
    brainiac.eval()
    modernbert.eval()
    
    # Enable GPU optimizations
    torch.backends.cudnn.benchmark = True
    print(f"[GPU {gpu_id}] Models ready, starting processing...", flush=True)
    
    # Initialize graph builders
    image_graph_builder = ImageGraphBuilder(
        patch_size=tuple(dataset_config['patch_size']),
        k_neighbors=dataset_config['k_neighbors'],
    )
    
    mask_graph_builder = MaskGraphBuilder(
        patch_size=tuple(dataset_config['patch_size']),
    )
    
    roi_graph_builder = ROIGraphBuilder(
        num_rois=dataset_config['num_rois'],
    )
    
    num_patches = np.prod([
        dataset_config['target_shape'][i] // dataset_config['patch_size'][i]
        for i in range(3)
    ])
    hypergraph_builder = HypergraphBuilder(
        num_patches=num_patches,
        num_rois=dataset_config['num_rois'],
    )
    
    # Process samples assigned to this GPU
    text_descriptions = []
    total = len(sample_indices)
    for i, idx in enumerate(sample_indices):
        scan_id = dataset.scan_ids[idx]
        print(f"[GPU {gpu_id}] Processing sample {i+1}/{total}: {scan_id}", flush=True)
        
        # Load scaler manager if available
        scaler_manager = None
        if dataset_config.get('scaler_dir'):
            scaler_dir = Path(dataset_config['scaler_dir'])
            if scaler_dir.exists():
                scaler_manager = FeatureScalerManager()
                scaler_manager.load_scalers(scaler_dir)
        
        result = process_single_sample(
            idx, scan_id, dataset, output_dir,
            brainiac, modernbert,
            image_graph_builder, mask_graph_builder,
            roi_graph_builder, hypergraph_builder,
            device, brainiac_batch_size, text_batch_size,
            scaler_manager=scaler_manager,
        )
        if result:
            text_descriptions.extend(result)
        
        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] Completed {i+1}/{total} samples", flush=True)
    
    print(f"[GPU {gpu_id}] Finished processing all {total} samples", flush=True)
    return text_descriptions


def compute_scalers_from_dataset(
    dataset: Connect4Dataset,
    output_dir: Path,
    max_samples: int = None,  # Limit samples for scaler fitting (None = all)
):
    """
    Compute scalers from all training data.
    This must be run before preprocessing to fit scalers on all data.
    """
    print("\n" + "="*80)
    print("Computing feature scalers from training data...")
    print("="*80 + "\n")
    
    scaler_manager = FeatureScalerManager()
    
    # Collect all features
    all_radiomics = []
    all_anatcl_embeddings = []
    all_brainiac_embeddings = []
    all_modernbert_embeddings = []
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Collecting features from {num_samples} samples...", flush=True)
    
    for idx in tqdm(range(num_samples), desc="Collecting features"):
        try:
            scan_id = dataset.scan_ids[idx]
            
            # Collect radiomics features (same for all samples, but collect once per ROI)
            if idx == 0:
                for structure_id, feats in dataset.radiomics_features.items():
                    if feats is not None:
                        all_radiomics.append(feats)
            
            # Collect AnatCL embeddings
            try:
                anatcl_embs = dataset._load_anatcl_embeddings(scan_id)
                for slug, emb in anatcl_embs.items():
                    if emb is not None:
                        all_anatcl_embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Could not load AnatCL embeddings for {scan_id}: {e}", flush=True)
            
            # Note: BrainIAC and ModernBERT embeddings are computed during preprocessing,
            # so we'll collect them during the preprocessing phase
            # For now, we'll fit scalers on a subset during preprocessing
            
        except Exception as e:
            print(f"Warning: Error processing sample {idx} ({scan_id if 'scan_id' in locals() else 'unknown'}): {e}", flush=True)
            continue
    
    # Fit scalers
    print("\nFitting scalers...", flush=True)
    
    if all_radiomics:
        print(f"Fitting radiomics scaler on {len(all_radiomics)} features...", flush=True)
        scaler_manager.scalers['radiomics'] = scaler_manager.fit_radiomics_scaler(
            {i: feats for i, feats in enumerate(all_radiomics)}
        )
    
    if all_anatcl_embeddings:
        print(f"Fitting AnatCL scaler on {len(all_anatcl_embeddings)} embeddings...", flush=True)
        scaler_manager.scalers['anatcl'] = scaler_manager.fit_anatcl_scaler(all_anatcl_embeddings)
    
    # Save scalers
    scaler_dir = output_dir / "scalers"
    scaler_manager.save_scalers(scaler_dir)
    
    print(f"\n✓ Scalers computed and saved to {scaler_dir}")
    return scaler_manager


def preprocess_dataset_multi_gpu(
    dataset: Connect4Dataset,
    output_dir: Path,
    num_gpus: int = 4,
    brainiac_batch_size: int = 4,  # Try batch_size=4, falls back to 1 if OOM
    text_batch_size: int = 512,
    config: Dict = None,
    scaler_manager: FeatureScalerManager = None,
):
    """Preprocess entire dataset using multiple GPUs in parallel."""
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "image_emb").mkdir(exist_ok=True)
    (output_dir / "modernbert_emb").mkdir(exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)
    (output_dir / "hypergraphs").mkdir(exist_ok=True)
    
    # Prepare dataset config for workers
    dataset_config = {
        'root_dir': str(dataset.root),
        'patch_size': list(dataset.patch_size),
        'target_shape': list(dataset.target_shape),
        'dwi_matrix_path': config['data']['dwi_matrix_path'] if config else str(dataset.root / "dwi_matrix.csv"),
        'normalize_intensity': dataset.normalize_intensity,
        'brainiac_model_path': config['models'].get('brainiac_path') if config else None,
        'modernbert_model_path': config['models'].get('modernbert_path') if config else None,
        'brainiac_embed_dim': config['models'].get('brainiac', {}).get('embed_dim', 768) if config else 768,
        'modernbert_embed_dim': config['models']['modernbert']['embed_dim'] if config else 768,
        'num_rois': dataset.NUM_ROIS,
        'k_neighbors': config['graphs']['image']['k_neighbors'] if config else 5,
        'scaler_dir': str(output_dir / "scalers") if scaler_manager else None,
    }
    
    # Split samples across GPUs
    num_samples = len(dataset)
    samples_per_gpu = num_samples // num_gpus
    remainder = num_samples % num_gpus
    
    sample_splits = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        # Distribute remainder samples across first few GPUs
        gpu_sample_count = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + gpu_sample_count
        sample_splits.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    
    print(f"Processing {num_samples} samples across {num_gpus} GPUs")
    for gpu_id, indices in enumerate(sample_splits):
        print(f"  GPU {gpu_id}: {len(indices)} samples (indices {indices[0]}-{indices[-1]})")
    
    # Process in parallel using multiprocessing
    print(f"\nStarting parallel processing on {num_gpus} GPUs...", flush=True)
    print("Note: Each GPU will load models independently, this may take a few minutes...", flush=True)
    print("If you don't see progress after 5 minutes, check GPU utilization with: nvidia-smi", flush=True)
    sys.stdout.flush()
    
    # Use subprocess to set CUDA_VISIBLE_DEVICES before process starts
    # This ensures each process only sees its assigned GPU from the start
    import subprocess
    import pickle
    import tempfile
    
    try:
        print("Launching worker processes...", flush=True)
        sys.stdout.flush()
        
        processes = []
        temp_files = []
        
        # Create a wrapper script that will be executed with CUDA_VISIBLE_DEVICES set
        # CUDA_VISIBLE_DEVICES is set in env before Python starts, so torch will only see that GPU
        wrapper_script = f"""
import sys
import os
import pickle
from pathlib import Path

# Force unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# CUDA_VISIBLE_DEVICES is set by parent process in env before Python starts
gpu_id = int(sys.argv[1])
temp_file = sys.argv[2]

# Set up path before importing - ensure project root is in path
project_root = Path('{PROJECT_ROOT_STR}').resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(str(project_root))

# Import the worker function - torch will be imported at module level,
# but CUDA_VISIBLE_DEVICES is already set, so it will only see the assigned GPU
# Import directly from the module file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "preprocess_graphs",
    str(project_root / "scripts" / "preprocess_graphs.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
worker_process = module.worker_process

# Load the arguments
with open(temp_file, 'rb') as f:
    args = pickle.load(f)

# Run worker
result = worker_process(*args)

# Save result
result_file = temp_file.replace('.pkl', '_result.pkl')
with open(result_file, 'wb') as f:
    pickle.dump((gpu_id, result), f)
"""
        
        # Save wrapper script
        wrapper_path = output_dir / 'worker_wrapper.py'
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_script)
        
        # Start processes with CUDA_VISIBLE_DEVICES set in environment
        for gpu_id, indices in enumerate(sample_splits):
            # Create temp file for arguments
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir=str(output_dir))
            pickle.dump((gpu_id, indices, dataset_config, output_dir, brainiac_batch_size, text_batch_size), temp_file)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Create environment with CUDA_VISIBLE_DEVICES set
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            env['PYTHONPATH'] = PROJECT_ROOT_STR
            
            # Start subprocess with unbuffered output for real-time progress
            p = subprocess.Popen(
                [sys.executable, '-u', str(wrapper_path), str(gpu_id), temp_file.name],  # -u for unbuffered
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            processes.append((gpu_id, p, temp_file.name))
            print(f"Started process for GPU {gpu_id} (PID: {p.pid})", flush=True)
        
        # Stream output from all processes in real-time
        print("Waiting for all worker processes to complete...", flush=True)
        import threading
        import queue
        
        def stream_output(gpu_id, process, output_queue):
            """Stream output from a process in real-time."""
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_queue.put((gpu_id, line.rstrip()))
            process.stdout.close()
            output_queue.put((gpu_id, None))  # Signal completion
        
        # Start output streaming threads
        output_queue = queue.Queue()
        threads = []
        for gpu_id, p, temp_file in processes:
            t = threading.Thread(target=stream_output, args=(gpu_id, p, output_queue))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Collect output and wait for processes
        results = [None] * num_gpus
        completed = set()
        
        while len(completed) < num_gpus:
            try:
                gpu_id, line = output_queue.get(timeout=1.0)
                if line is None:
                    # Process completed
                    completed.add(gpu_id)
                else:
                    # Print output with GPU prefix
                    print(f"[GPU {gpu_id}] {line}", flush=True)
            except queue.Empty:
                # Check if any processes have finished
                for gpu_id, p, temp_file in processes:
                    if p.poll() is not None and gpu_id not in completed:
                        # Process finished, drain remaining output
                        completed.add(gpu_id)
        
        # Wait for all threads to finish
        for t in threads:
            t.join()
        
        # Collect results from completed processes
        for gpu_id, p, temp_file in processes:
            if p.returncode != 0:
                print(f"Warning: GPU {gpu_id} process exited with code {p.returncode}", flush=True)
            else:
                # Load result
                result_file = temp_file.replace('.pkl', '_result.pkl')
                if os.path.exists(result_file):
                    with open(result_file, 'rb') as f:
                        gpu_id_result, result = pickle.load(f)
                        results[gpu_id_result] = result
                    os.remove(result_file)
            os.remove(temp_file)
        
        # Clean up wrapper script
        if wrapper_path.exists():
            wrapper_path.unlink()
        
        print("All worker processes completed!", flush=True)
    except Exception as e:
        print(f"Error setting up subprocess workers: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Collect all text descriptions
    all_text_descriptions = []
    for result in results:
        if result:
            all_text_descriptions.extend(result)
    
    # Save text descriptions CSV
    if all_text_descriptions:
        df_texts = pd.DataFrame(all_text_descriptions)
        df_texts.to_csv(output_dir / "patch_text_descriptions.csv", index=False)
        print(f"\nSaved {len(all_text_descriptions)} text descriptions to patch_text_descriptions.csv")
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='/path/to/data/precomputed_graphs', help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--brainiac_batch_size', type=int, default=16, help='Batch size for BrainIAC encoding (default 16; falls back on OOM)')
    parser.add_argument('--text_batch_size', type=int, default=1024, help='Batch size for ModernBERT encoding')
    parser.add_argument('--skip_scaler_computation', action='store_true', help='Skip scaler computation (use existing scalers)')
    parser.add_argument('--scaler_max_samples', type=int, default=None, help='Max samples to use for scaler fitting (None = all)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create dataset (for getting scan IDs and config)
    dataset = Connect4Dataset(
        root_dir=config['data']['root_dir'],
        patch_size=tuple(config['data']['patch_size']),
        target_shape=tuple(config['data']['target_shape']),
        dwi_matrix_path=config['data']['dwi_matrix_path'],
        normalize_intensity=config['data']['normalize_intensity'],
        brainiac_model_path=config['models'].get('brainiac_path'),
        modernbert_model_path=config['models'].get('modernbert_path'),
    )
    
    # Verify patch size
    num_patches = np.prod([
        config['data']['target_shape'][i] // config['data']['patch_size'][i]
        for i in range(3)
    ])
    print(f"Patch size: {config['data']['patch_size']}")
    print(f"Target shape: {config['data']['target_shape']}")
    print(f"Number of patches per sample: {num_patches} (should be 512 for 16x16x16 patches)")
    
    output_dir = Path(args.output_dir)
    
    # Compute scalers if not skipping
    scaler_manager = None
    if not args.skip_scaler_computation:
        scaler_manager = compute_scalers_from_dataset(
            dataset,
            output_dir,
            max_samples=args.scaler_max_samples,
        )
    else:
        # Load existing scalers
        scaler_dir = output_dir / "scalers"
        if scaler_dir.exists():
            scaler_manager = FeatureScalerManager()
            scaler_manager.load_scalers(scaler_dir)
            print(f"Loaded existing scalers from {scaler_dir}", flush=True)
        else:
            print(f"Warning: Scaler directory not found: {scaler_dir}. Proceeding without scalers.", flush=True)
    
    # Run multi-GPU preprocessing
    preprocess_dataset_multi_gpu(
        dataset,
        output_dir,
        num_gpus=args.num_gpus,
        brainiac_batch_size=args.brainiac_batch_size,
        text_batch_size=args.text_batch_size,
        config=config,
        scaler_manager=scaler_manager,
    )


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
