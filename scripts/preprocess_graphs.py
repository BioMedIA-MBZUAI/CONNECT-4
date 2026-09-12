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
    from data.protocol import (
        build_external_cache_protocol_context,
        conditioning_identity_from_cache_sources,
        load_completed_synthesis_checkpoint,
        patient_level_split_from_manifest,
        validate_external_scaler_provenance,
        validate_fixed_protocol_config,
        validate_training_cohort_manifest,
        validate_training_scaler_provenance,
    )
    from data.patch_descriptions import (
        KNOWN_FUNCTIONAL_CONNECTIVITY,
        PATCH_DESCRIPTION_SCHEMA_VERSION,
        build_normative_index,
        build_patch_description,
    )
    from data.provenance import canonical_sha256, directory_file_sha256, sha256_file
    from data.spatial_contract import PATCH_COORDINATE_CONTRACT
    from preprocessing.conform import load_common_grid_contract
    from models.brainiac_wrapper import BrainIACWrapper
    from models.modernbert_wrapper import ModernBERTWrapper
    from graphs.image_graph import ImageGraphBuilder
    from graphs.mask_graph import MaskGraphBuilder
    from graphs.roi_graph import ROIGraphBuilder
    from graphs.hypergraph import HypergraphBuilder
    from utils.config import load_config, resolve_configured_value
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


def normative_context_hash(dataset: Connect4Dataset, patient_id: str) -> str:
    """Fingerprint the subject text inputs so stale embeddings are not reused."""
    subject = dataset.scan_to_normative_subject[str(patient_id)]
    return canonical_sha256(
        {
            "workbook_sha256": dataset.normative_workbook_sha256,
            "subject_rows": dataset.normative_source_rows[subject],
        }
    )


def scaler_source_fingerprint(scaler_manager: FeatureScalerManager = None) -> Dict:
    if scaler_manager is None or scaler_manager.scaler_dir is None:
        return {"enabled": False}
    directory = Path(scaler_manager.scaler_dir)
    return {
        "enabled": True,
        "files_sha256": directory_file_sha256(directory),
    }


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
    external_protocol_context: Dict = None,
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
        
        normative_subject = dataset.scan_to_normative_subject[scan_id]
        expected_normative_hash = normative_context_hash(dataset, scan_id)
        cache_sources = {
            "dataset": dataset.source_fingerprint(scan_id),
            "brainiac": brainiac.source_fingerprint,
            "modernbert": modernbert.source_fingerprint,
            "scalers": scaler_source_fingerprint(scaler_manager),
            "target_shape": list(dataset.target_shape),
            "patch_size": list(dataset.patch_size),
            "patch_description_schema_version": PATCH_DESCRIPTION_SCHEMA_VERSION,
            "patch_coordinate_contract": dict(PATCH_COORDINATE_CONTRACT),
            "functional_connectivity_sha256": canonical_sha256(
                KNOWN_FUNCTIONAL_CONNECTIVITY
            ),
        }
        if external_protocol_context is not None:
            cache_sources["external_protocol"] = dict(external_protocol_context)
            current_conditioning = conditioning_identity_from_cache_sources(
                cache_sources
            )
            if current_conditioning != external_protocol_context.get(
                "conditioning_identity"
            ):
                raise RuntimeError(
                    "external preprocessing conditioning artifacts differ from "
                    "the completed synthesis checkpoint"
                )
        cache_sources_sha256 = canonical_sha256(cache_sources)
        
        # Check if already processed (skip to speed up).  Text schema changes
        # invalidate ModernBERT embeddings, mask nodes, and downstream graphs.
        image_nodes_path = graphs_dir / f"{scan_id}_image_nodes.npy"
        image_emb_path = image_emb_dir / f"{scan_id}_image_patch_embeddings.npy"
        modernbert_emb_path = modernbert_emb_dir / f"{scan_id}_mask_patch_embeddings.npy"
        metadata_path = hypergraphs_dir / f"{scan_id}_metadata.json"
        patch_desc_path = hypergraphs_dir / f"{scan_id}_patch_descriptions.json"
        description_cache_current = False
        cached_metadata = {}
        if metadata_path.exists() and patch_desc_path.exists():
            try:
                with open(metadata_path) as f:
                    cached_metadata = json.load(f)
                description_cache_current = (
                    cached_metadata.get("patch_description_schema_version")
                    == PATCH_DESCRIPTION_SCHEMA_VERSION
                    and cached_metadata.get("normative_context_sha256")
                    == expected_normative_hash
                    and cached_metadata.get("source_fingerprint") == cache_sources
                    and cached_metadata.get("source_fingerprint_sha256")
                    == cache_sources_sha256
                )
            except (OSError, ValueError, TypeError):
                description_cache_current = False
        if image_nodes_path.exists() and image_emb_path.exists() and modernbert_emb_path.exists():
            mask_nodes_path = graphs_dir / f"{scan_id}_mask_nodes.npy"
            roi_nodes_path = graphs_dir / f"{scan_id}_roi_nodes.npy"
            hyperedge_path = hypergraphs_dir / f"{scan_id}_hyperedge_index.npy"
            hyperedge_weights_path = (
                hypergraphs_dir / f"{scan_id}_hyperedge_weights.npy"
            )
            patch_distribution_path = (
                hypergraphs_dir / f"{scan_id}_patch_distributions.json"
            )
            required_cached = [
                image_nodes_path,
                image_emb_path,
                modernbert_emb_path,
                mask_nodes_path,
                roi_nodes_path,
                hyperedge_path,
                hyperedge_weights_path,
                patch_distribution_path,
                patch_desc_path,
            ]
            artifact_hashes = cached_metadata.get("artifact_sha256", {})
            hashes_current = (
                isinstance(artifact_hashes, dict)
                and set(artifact_hashes)
                == {path.name for path in required_cached}
                and all(
                    artifact_hashes[path.name] == sha256_file(path)
                    for path in required_cached
                )
            )
            if description_cache_current and hashes_current:
                print(f"  [{scan_id}] ✓ Source-bound cache is current, skipping...", flush=True)
                return []
            if not description_cache_current:
                print(
                    f"  [{scan_id}] Cached text predates {PATCH_DESCRIPTION_SCHEMA_VERSION}; "
                    "rebuilding text embeddings and dependent graphs...",
                    flush=True,
                )
        
        # Load sample
        sample = dataset[idx]
        
        t1w = sample['t1w']  # [1, 1, D, H, W]
        mask = sample['mask']  # [1, 1, D, H, W]
        roi_embeddings = sample['roi_embeddings']  # [num_rois, embed_dim]
        # Recomputed below from the label-preserving SynthSeg volume.  The
        # sample's `mask` is a binary loss mask and cannot identify ROIs.
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
        # Patch count comes from the externally validated common-grid shape.
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
        
        print(f"  [{scan_id}] BrainIAC total time: {time.time() - t_bria_start:.2f}s", flush=True)
        
        # Save all patch embeddings efficiently in a single file
        print(f"  [{scan_id}] Saving {num_patches} BrainIAC embeddings (single file)...", flush=True)
        emb_path = image_emb_dir / f"{scan_id}_image_patch_embeddings.npy"
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), image_patch_embeddings.numpy())
        
        # 2. Build image graph
        t_img_graph_start = time.time()
        print(f"  [{scan_id}] Building image graph...", flush=True)
        spatial_shape = tuple(dataset.target_shape)
        image_nodes, _ = image_graph_builder(
            image_patch_embeddings.unsqueeze(0),
            spatial_shape,
        )
        
        # Save image graph
        print(f"  [{scan_id}] Saving image graph...", flush=True)
        save_tensor(image_nodes.squeeze(0), graphs_dir / f"{scan_id}_image_nodes.npy")
        print(f"  [{scan_id}] Image graph time: {time.time() - t_img_graph_start:.2f}s", flush=True)
        print(f"  [{scan_id}] Image pipeline total: {time.time() - t_image_pipe_start:.2f}s", flush=True)
        
        # 3. Process mask patches and create text descriptions
        t_mask_pipe_start = time.time()
        print(f"  [{scan_id}] Patchifying mask and computing distributions...", flush=True)
        t_mask_text_start = time.time()
        patch_text_descriptions = []
        mask_patch_embeddings_list = []
        
        # Reload the label-preserving SynthSeg mask.  Connect4Dataset also
        # exposes a binary brain mask for losses; using that here erases ROI
        # identities and yields empty ROI distributions.
        labelled_mask = sample["segmentation"]
        affine = sample["affine"].cpu().numpy()
        patch_distributions = []
        for patch_idx in range(num_patches):
            raw_distribution = dataset._compute_patch_distribution(labelled_mask, patch_idx)
            patch_distributions.append({
                int(structure_id): float(coverage)
                for structure_id, coverage in raw_distribution.items()
                if int(structure_id) in dataset.ID_TO_SLUG and float(coverage) > 0.0
            })
        # MaskGraphBuilder needs the same labelled mask to construct its DWI
        # adjacency.  Its public shape is [C,D,H,W] before the batch is added.
        mask = labelled_mask.squeeze(0).to(device)
        normative_index = build_normative_index(dataset.normative_descriptions)
        present_slugs = {
            dataset.ID_TO_SLUG[structure_id]
            for distribution in patch_distributions
            for structure_id in distribution
        }
        missing_normative_slugs = sorted(
            slug for slug in present_slugs
            if dataset.SLUG_TO_ID[slug] in dataset.POTVIN_ROI_IDS
            and (normative_subject, slug) not in normative_index
        )
        if missing_normative_slugs:
            raise ValueError(
                f"{scan_id} is missing subject-specific normative descriptions for "
                f"{len(missing_normative_slugs)} present ROIs: {', '.join(missing_normative_slugs)}. "
                "Paper-faithful patch text requires both connectivity and normative context."
            )
        
        text_descriptions = []
        for patch_idx in range(num_patches):
            dist = patch_distributions[patch_idx]
            
            # Compute center of mass
            x_mm, y_mm, z_mm = dataset._compute_patch_center_mm(
                patch_idx,
                segmentation=labelled_mask,
                affine=affine,
            )
            
            # Each ROI summary deliberately contains both known functional
            # connectivity and the subject-specific normative volume context.
            text_desc = build_patch_description(
                patch_idx=patch_idx,
                center_mm=(x_mm, y_mm, z_mm),
                distribution=dist,
                id_to_slug=dataset.ID_TO_SLUG,
                normative_index=normative_index,
                patient_id=normative_subject,
            )
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
                'description_schema_version': PATCH_DESCRIPTION_SCHEMA_VERSION,
            })
        
        # Save raw patch descriptions (pre-ModernBERT) for inspection/reuse
        print(f"  [{scan_id}] Saving raw patch descriptions...", flush=True)
        patch_desc_path.parent.mkdir(parents=True, exist_ok=True)
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
        
        print(f"  [{scan_id}] ModernBERT total time: {time.time() - t_modern_start:.2f}s", flush=True)
        print(f"  [{scan_id}] Mask/text prep total time: {time.time() - t_mask_text_start:.2f}s", flush=True)
        
        # Save all ModernBERT embeddings efficiently in a single file
        print(f"  [{scan_id}] Saving {len(mask_patch_embeddings)} ModernBERT embeddings (single file)...", flush=True)
        modernbert_emb_path = modernbert_emb_dir / f"{scan_id}_mask_patch_embeddings.npy"
        modernbert_emb_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_np = (
            mask_patch_embeddings.detach().cpu().numpy()
            if isinstance(mask_patch_embeddings, torch.Tensor)
            else mask_patch_embeddings
        )
        np.save(str(modernbert_emb_path), embeddings_np)
        
        # 5. Build mask graph
        t_mask_graph_start = time.time()
        print(f"  [{scan_id}] Building mask graph...", flush=True)
        structure_labels = {
            label_id: slug
            for slug, label_id in dataset.ROI_SPECS
        }
        mask_nodes, _ = mask_graph_builder(
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
        
        # 6. Build ROI graph (roi_embeddings already has scalers applied from dataset)
        t_roi_graph_start = time.time()
        roi_nodes, _ = roi_graph_builder(
            roi_embeddings.unsqueeze(0),
            dwi_matrix=dwi_matrix,
        )
        print(f"  [{scan_id}] ROI graph time: {time.time() - t_roi_graph_start:.2f}s", flush=True)
        
        # Save ROI graph
        save_tensor(roi_nodes.squeeze(0), graphs_dir / f"{scan_id}_roi_nodes.npy")
        
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
            'patch_description_schema_version': PATCH_DESCRIPTION_SCHEMA_VERSION,
            'patch_coordinate_contract': dict(PATCH_COORDINATE_CONTRACT),
            'structural_grid_geometry': cache_sources['dataset'][
                'structural_grid_geometry'
            ],
            'normative_context_sha256': expected_normative_hash,
            'source_fingerprint': cache_sources,
            'source_fingerprint_sha256': cache_sources_sha256,
            'patch_text_components': [
                'roi_distribution_and_patch_center',
                'known_functional_connectivity',
                'subject_specific_normative_volume',
            ],
            'potvin_supported_roi_ids': sorted(dataset.POTVIN_ROI_IDS),
        }
        artifact_paths = [
            graphs_dir / f"{scan_id}_image_nodes.npy",
            image_emb_path,
            modernbert_emb_path,
            graphs_dir / f"{scan_id}_mask_nodes.npy",
            graphs_dir / f"{scan_id}_roi_nodes.npy",
            hypergraphs_dir / f"{scan_id}_hyperedge_index.npy",
            hypergraphs_dir / f"{scan_id}_hyperedge_weights.npy",
            hypergraphs_dir / f"{scan_id}_patch_distributions.json",
            patch_desc_path,
        ]
        metadata['artifact_sha256'] = {
            path.name: sha256_file(path) for path in artifact_paths
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
        raise RuntimeError(f"graph preprocessing failed for {scan_id}") from e


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
        raise RuntimeError(f"CUDA is not available in graph worker {gpu_id}")
    
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
        common_grid_contract_sha256=dataset_config.get(
            'common_grid_contract_sha256'
        ),
        dwi_matrix_path=dataset_config['dwi_matrix_path'],
        normative_csv_path=dataset_config['normative_csv_path'],
        cohort_manifest_path=dataset_config['cohort_manifest_path'],
        scaler_dir=dataset_config.get('scaler_dir'),
    )
    print(f"[GPU {gpu_id}] Dataset created with {len(dataset)} samples", flush=True)
    
    # Initialize models on this GPU
    print(f"[GPU {gpu_id}] Loading BrainIAC model...", flush=True)
    brainiac = BrainIACWrapper(
        model_path=dataset_config.get('brainiac_model_path'),
        embed_dim=dataset_config.get('brainiac_embed_dim', 768),
        device=str(device),
        expected_checkpoint_sha256=dataset_config.get(
            'brainiac_checkpoint_sha256'
        ),
        expected_source_fingerprint_sha256=dataset_config.get(
            'brainiac_source_sha256'
        ),
    ).to(device)
    brainiac.eval()  # Set to eval mode
    print(f"[GPU {gpu_id}] BrainIAC loaded", flush=True)
    
    print(f"[GPU {gpu_id}] Loading ModernBERT model...", flush=True)
    modernbert = ModernBERTWrapper(
        model_path=dataset_config['modernbert_model_path'],
        embed_dim=dataset_config['modernbert_embed_dim'],
        revision=dataset_config.get('modernbert_revision'),
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
    completed_scan_ids = []
    total = len(sample_indices)
    for i, idx in enumerate(sample_indices):
        scan_id = dataset.scan_ids[idx]
        print(f"[GPU {gpu_id}] Processing sample {i+1}/{total}: {scan_id}", flush=True)
        
        # Load scaler manager if available
        scaler_path = dataset_config.get('scaler_dir')
        if not scaler_path:
            raise RuntimeError("graph worker has no certified scaler directory")
        scaler_dir = Path(scaler_path)
        if not scaler_dir.is_dir():
            raise FileNotFoundError(
                f"graph worker scaler directory not found: {scaler_dir}"
            )
        scaler_manager = FeatureScalerManager(scaler_dir)
        scaler_manager.load_scalers(scaler_dir)
        if set(scaler_manager.scalers) != {'radiomics', 'anatcl'}:
            raise RuntimeError(
                "graph workers require exactly the radiomics and AnatCL scalers"
            )
        
        result = process_single_sample(
            idx, scan_id, dataset, output_dir,
            brainiac, modernbert,
            image_graph_builder, mask_graph_builder,
            roi_graph_builder, hypergraph_builder,
            device, brainiac_batch_size, text_batch_size,
            scaler_manager=scaler_manager,
            external_protocol_context=dataset_config.get("external_protocol_context"),
        )
        if result:
            text_descriptions.extend(result)
        completed_scan_ids.append(scan_id)
        
        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] Completed {i+1}/{total} samples", flush=True)
    
    print(f"[GPU {gpu_id}] Finished processing all {total} samples", flush=True)
    return {
        "scan_ids": completed_scan_ids,
        "text_descriptions": text_descriptions,
    }


def compute_scalers_from_dataset(
    dataset: Connect4Dataset,
    output_dir: Path,
    sample_indices: List[int] = None,
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
    
    indices = (
        list(range(len(dataset))) if sample_indices is None else list(sample_indices)
    )
    if not indices:
        raise ValueError("the scaler-fitting training partition cannot be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("the scaler-fitting training partition contains duplicates")
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("the scaler-fitting training partition contains an invalid index")
    print(
        f"Collecting features from {len(indices)} training-partition samples...",
        flush=True,
    )
    
    for idx in tqdm(indices, desc="Collecting features"):
        scan_id = dataset.scan_ids[idx]
        for features in dataset.radiomics_for_scan(scan_id).values():
            all_radiomics.append(features)
        anatcl_embeddings = dataset._load_anatcl_embeddings(scan_id)
        all_anatcl_embeddings.extend(anatcl_embeddings.values())
    
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
    scaler_manager.scaler_dir = scaler_dir
    training_scan_ids = sorted(dataset.scan_ids[index] for index in indices)
    with open(scaler_dir / "training_partition.json", "w") as stream:
        json.dump(
            {
                "schema": "connect4-training-only-scalers-v1",
                "training_partition_scan_ids": training_scan_ids,
                "fitted_scan_ids": training_scan_ids,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    validate_training_scaler_provenance(str(scaler_dir), training_scan_ids)
    
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
    external_protocol_context: Dict = None,
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
        'common_grid_contract_sha256': dataset.common_grid_contract_sha256,
        'dwi_matrix_path': config['data']['dwi_matrix_path'] if config else str(dataset.root / "dwi_matrix.csv"),
        'normative_csv_path': (
            config['data'].get('normative_csv_path')
            if config and config['data'].get('normative_csv_path')
            else str(dataset.root / "patient_roi_normative_inputs.csv")
        ),
        'cohort_manifest_path': config['data']['cohort_manifest'],
        'brainiac_model_path': config['models'].get('brainiac_path') if config else None,
        'brainiac_checkpoint_sha256': (
            resolve_configured_value(
                config['models'],
                'brainiac_checkpoint_sha256',
                'brainiac_checkpoint_sha256_env',
            ) if config else None
        ),
        'brainiac_source_sha256': (
            resolve_configured_value(
                config['models'],
                'brainiac_source_sha256',
                'brainiac_source_sha256_env',
            ) if config else None
        ),
        'modernbert_model_path': (
            config['models'].get('modernbert_name')
            or config['models'].get('modernbert_path')
        ) if config else None,
        'modernbert_revision': (
            config['models'].get('modernbert_revision') if config else None
        ),
        'brainiac_embed_dim': (
            config['models']['fusion'].get('image_embed_dim', 768)
            if config else 768
        ),
        'modernbert_embed_dim': (
            config['models']['fusion'].get('mask_embed_dim', 768)
            if config else 768
        ),
        'num_rois': dataset.NUM_ROIS,
        'k_neighbors': config['models']['graphs'].get('k_neighbors', 5) if config else 5,
        'scaler_dir': (
            str(scaler_manager.scaler_dir)
            if scaler_manager is not None and scaler_manager.scaler_dir is not None
            else (str(output_dir / "scalers") if scaler_manager is not None else None)
        ),
        'external_protocol_context': external_protocol_context,
    }
    
    # Split samples across GPUs
    num_samples = len(dataset)
    if num_gpus < 1 or num_gpus > num_samples:
        raise ValueError(
            f"num_gpus must be between 1 and the {num_samples} cohort scans; "
            f"got {num_gpus}"
        )
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
        failed_workers = []
        for gpu_id, p, temp_file in processes:
            if p.returncode != 0:
                failed_workers.append((gpu_id, p.returncode))
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

        missing_results = [index for index, result in enumerate(results) if result is None]
        if failed_workers or missing_results:
            raise RuntimeError(
                "one or more graph-preprocessing workers failed; the fixed cohort "
                f"will not be reduced silently (workers={failed_workers}, "
                f"missing_results={missing_results})"
            )
        
        print("All worker processes completed!", flush=True)
    except Exception as e:
        print(f"Error setting up subprocess workers: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Every worker returns an explicit list of completed scan IDs, including
    # source-bound cache hits. This proves a rerun covered the fixed cohort.
    all_text_descriptions = []
    completed_scan_ids = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("graph worker returned an invalid completion record")
        all_text_descriptions.extend(result.get("text_descriptions", []))
        completed_scan_ids.extend(result.get("scan_ids", []))
    if sorted(completed_scan_ids) != sorted(dataset.scan_ids):
        raise RuntimeError(
            "graph preprocessing did not certify every fixed-cohort scan; "
            f"completed={sorted(completed_scan_ids)[:10]}, "
            f"expected={sorted(dataset.scan_ids)[:10]}"
        )
    
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
    parser.add_argument(
        '--mode',
        choices=('training', 'external'),
        default='training',
        help=(
            'training fits/validates the fixed A4+ADNI preprocessing state; '
            'external prepares a checkpoint-proven unseen structural-only cohort'
        ),
    )
    parser.add_argument(
        '--synthesis_checkpoint',
        default=None,
        help='completed synthesis checkpoint (required only with --mode external)',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (defaults to data.precomputed_dir in the config)',
    )
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--brainiac_batch_size', type=int, default=16, help='Batch size for BrainIAC encoding (default 16; falls back on OOM)')
    parser.add_argument('--text_batch_size', type=int, default=1024, help='Batch size for ModernBERT encoding')
    parser.add_argument(
        '--normative_csv_path',
        type=str,
        default=None,
        help=(
            'Raw per-subject Potvin input CSV (measured volume, age, sex, scanner, '
            'field strength, and ICV). Defaults to data.normative_csv_path in the '
            'config, then <root_dir>/patient_roi_normative_inputs.csv.'
        ),
    )
    parser.add_argument('--skip_scaler_computation', action='store_true', help='Skip scaler computation (use existing scalers)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    validate_fixed_protocol_config(config)
    graph_config = config['models']['graphs']
    fusion_config = config['models']['fusion']
    common_grid = load_common_grid_contract(
        config['data']['common_grid_contract_path'],
        expected_sha256=resolve_configured_value(
            config['data'],
            'common_grid_contract_sha256',
            'common_grid_contract_sha256_env',
        ),
    )
    if tuple(config['data']['architecture_shape']) != tuple(
        common_grid['architecture_shape']
    ):
        raise ValueError(
            "data.architecture_shape differs from the hash-bound common-grid contract"
        )
    cohort_manifest = config['data'].get('cohort_manifest')
    if not cohort_manifest:
        raise ValueError(
            "data.cohort_manifest is required so graph preprocessing proves the "
            "complete explicit patient/cohort identity"
        )
    normative_csv_path = (
        args.normative_csv_path
        or config['data'].get('normative_csv_path')
        or str(Path(config['data']['root_dir']) / 'patient_roi_normative_inputs.csv')
    )
    # Propagate the resolved path to spawned workers without requiring a config
    # schema change for existing runs.
    config['data']['normative_csv_path'] = normative_csv_path
    if not Path(normative_csv_path).is_file():
        raise FileNotFoundError(
            f"Raw Potvin input CSV not found at {normative_csv_path}. It is required "
            "for subject-specific paper patch text."
        )
    
    # Create dataset (for getting scan IDs and config)
    dataset = Connect4Dataset(
        root_dir=config['data']['root_dir'],
        patch_size=tuple(graph_config['patch_size']),
        target_shape=tuple(config['data']['architecture_shape']),
        common_grid_contract_sha256=common_grid['contract_sha256'],
        dwi_matrix_path=config['data']['dwi_matrix_path'],
        normative_csv_path=normative_csv_path,
        cohort_manifest_path=cohort_manifest,
    )
    if not dataset.normative_descriptions:
        raise ValueError(
            f"No valid per-subject normative descriptions were loaded from {normative_csv_path}. "
            "Check PatientID/Structure columns and normative values."
        )
    normalised_norms = build_normative_index(dataset.normative_descriptions)
    missing_roi_norms = [
        (scan_id, slug)
        for scan_id in dataset.scan_ids
        for label_id, slug in dataset.ID_TO_SLUG.items()
        if label_id in dataset.POTVIN_ROI_IDS
        and (dataset.scan_to_normative_subject[scan_id], slug) not in normalised_norms
    ]
    if missing_roi_norms:
        preview = ', '.join(f"{scan}:{slug}" for scan, slug in missing_roi_norms[:10])
        raise ValueError(
            f"Subject-specific Potvin text is missing for {len(missing_roi_norms)} "
            f"scan/ROI pairs (first: {preview}). All 23 individual workbook models "
            "are required; the remaining nine CONNECT-4 ROIs have no individual "
            "Potvin model and are not fabricated."
        )

    output_dir = Path(args.output_dir or config['data']['precomputed_dir'])

    # Lock the manuscript's patient-disjoint partitions before fitting any
    # data-dependent transform. External mode instead proves that every scan is
    # absent from the completed synthesis run and never fits a transform.
    external_protocol_context = None
    completed_checkpoint = None
    if args.mode == 'external':
        if not args.synthesis_checkpoint:
            raise ValueError(
                "--mode external requires --synthesis_checkpoint from a completed run"
            )
        if args.skip_scaler_computation:
            raise ValueError(
                "--skip_scaler_computation is a training-mode option; external mode "
                "always requires and reuses checkpoint-bound training scalers"
            )
        completed_checkpoint, _, checkpoint_sha256 = (
            load_completed_synthesis_checkpoint(args.synthesis_checkpoint)
        )
        cohort_evidence, external_protocol_context = (
            build_external_cache_protocol_context(
                manifest_path=cohort_manifest,
                scan_ids=dataset.scan_ids,
                synthesis_split_identity=completed_checkpoint['split_identity'],
                synthesis_artifact_identity=completed_checkpoint['artifact_identity'],
                synthesis_checkpoint_sha256=checkpoint_sha256,
            )
        )
        print(
            "External structural cohort certified as unseen: "
            f"scans={len(dataset.scan_ids)}, "
            f"patients={len(set(cohort_evidence.patient_by_scan.values()))}, "
            f"cohorts={sorted(set(cohort_evidence.cohort_by_scan.values()))}",
            flush=True,
        )
        train_indices = validation_indices = test_indices = None
    else:
        if args.synthesis_checkpoint:
            raise ValueError(
                "--synthesis_checkpoint is valid only with --mode external"
            )
        cohort_evidence = validate_training_cohort_manifest(
            cohort_manifest,
            dataset.scan_ids,
            expected_scan_counts=config['data'].get('expected_cohort_scan_counts'),
            protocol_profile=config['data'].get('protocol_profile'),
        )
        train_indices, validation_indices, test_indices = (
            patient_level_split_from_manifest(
                dataset.scan_ids,
                cohort_evidence.patient_by_scan,
                val_frac=config['training'].get('val_frac', 0.15),
                test_frac=config['training'].get('test_frac', 0.15),
                seed=config['training'].get('seed', 42),
                manifest_path=config['training'].get('split_manifest'),
                manifest_sha256=resolve_configured_value(
                    config['training'],
                    'split_manifest_sha256',
                    'split_manifest_sha256_env',
                ),
                protocol_profile=str(config['data'].get('protocol_profile', '')),
            )
        )
        print(
            "Patient-disjoint preprocessing partitions: "
            f"train={len(train_indices)}, validation={len(validation_indices)}, "
            f"test={len(test_indices)}",
            flush=True,
        )
    
    # Verify patch size
    num_patches = np.prod([
        config['data']['architecture_shape'][i] // graph_config['patch_size'][i]
        for i in range(3)
    ])
    if int(fusion_config['num_rois']) != dataset.NUM_ROIS:
        raise ValueError(
            f"models.fusion.num_rois={fusion_config['num_rois']} but the "
            f"CONNECT-4 segmentation defines {dataset.NUM_ROIS} ROIs"
        )
    print(f"Patch size: {graph_config['patch_size']}")
    print(f"Architecture shape: {config['data']['architecture_shape']}")
    print(
        f"Number of padded-grid patches per sample: {num_patches} "
        "(a versioned recovery choice, not a paper-reported count)"
    )
    
    # Compute scalers if not skipping
    scaler_manager = None
    if args.mode == 'external':
        scaler_dir = config['data'].get('scaler_dir')
        validate_external_scaler_provenance(
            scaler_dir,
            dataset.scan_ids,
            completed_checkpoint['artifact_identity'],
            completed_checkpoint['split_identity'],
        )
        scaler_manager = FeatureScalerManager(Path(scaler_dir))
        scaler_manager.load_scalers(Path(scaler_dir))
        required_scalers = {'radiomics', 'anatcl'}
        if set(scaler_manager.scalers) != required_scalers:
            raise RuntimeError(
                "external preprocessing requires the checkpoint-bound radiomics "
                "and AnatCL synthesis-training scalers"
            )
        print(
            f"Loaded immutable synthesis-training scalers from {scaler_dir}",
            flush=True,
        )
    elif not args.skip_scaler_computation:
        scaler_manager = compute_scalers_from_dataset(
            dataset,
            output_dir,
            sample_indices=train_indices,
        )
    else:
        # Load existing scalers
        scaler_dir = output_dir / "scalers"
        if scaler_dir.exists():
            expected_scaler_ids = sorted(dataset.scan_ids[index] for index in train_indices)
            try:
                validate_training_scaler_provenance(
                    str(scaler_dir), expected_scaler_ids
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Existing scalers were not fitted on exactly the current fixed "
                    "training partition. Refit them before paper-faithful preprocessing."
                ) from exc
            scaler_manager = FeatureScalerManager(scaler_dir)
            scaler_manager.load_scalers(scaler_dir)
            print(f"Loaded existing scalers from {scaler_dir}", flush=True)
        else:
            raise FileNotFoundError(
                f"--skip_scaler_computation requires the existing, provenance-bound "
                f"training scalers at {scaler_dir}"
            )

    if scaler_manager is None or set(scaler_manager.scalers) != {'radiomics', 'anatcl'}:
        raise RuntimeError(
            "graph preprocessing requires exactly the radiomics and AnatCL scalers"
        )
    
    # Run multi-GPU preprocessing
    preprocess_dataset_multi_gpu(
        dataset,
        output_dir,
        num_gpus=args.num_gpus,
        brainiac_batch_size=args.brainiac_batch_size,
        text_batch_size=args.text_batch_size,
        config=config,
        scaler_manager=scaler_manager,
        external_protocol_context=external_protocol_context,
    )


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
