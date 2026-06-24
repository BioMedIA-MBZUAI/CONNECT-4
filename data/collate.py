"""
Custom collate function for batching Connect4Dataset samples.
Handles non-tensor items like patch_distributions and structure_to_roi_idx.
"""
import torch
from torch.utils.data.dataloader import default_collate
from typing import Dict, List, Any


def connect4_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for Connect4Dataset.
    Handles list of dicts (patch_distributions) and single dict (structure_to_roi_idx).
    Also handles variable-sized tensors like fMRI with different temporal dimensions.
    """
    # Debug statements removed for cleaner output
    # import os
    # rank = os.environ.get('LOCAL_RANK', '0')
    # if len(batch) > 0 and isinstance(batch[0].get('scan_id', ''), str):
    #     scan_id = batch[0].get('scan_id', 'unknown')
    #     print(f"[Collate RANK={rank}] Collating batch of {len(batch)} samples (scan_ids: {[s.get('scan_id', '?') for s in batch]})", flush=True)
    
    # Extract keys
    keys = batch[0].keys()
    
    collated = {}
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Handle non-tensor items
        if key == 'patch_distributions':
            # Keep as list of lists
            collated[key] = values
        elif key == 'structure_to_roi_idx':
            # All samples should have the same mapping, just take first
            collated[key] = values[0]
        elif key == 'scan_id':
            # Keep as list of strings
            collated[key] = values
        elif key in ['hyperedge_index', 'hyperedge_weights']:
            # Variable-sized tensors - keep as list, will be handled in forward pass
            collated[key] = values
        elif key == 'roi_masks':
            # ROI masks: dict of {roi_idx: [D, H, W] tensor}
            # Keep as list of dicts, will be handled in forward pass
            collated[key] = values
            # Debug statements removed for cleaner output
            # if len(batch) > 0:
            #     print(f"[Collate RANK={rank}] Preserved roi_masks: {len(values)} samples, first sample type: {type(values[0]) if len(values) > 0 else 'N/A'}, first sample keys: {list(values[0].keys()) if len(values) > 0 and isinstance(values[0], dict) else 'N/A'}", flush=True)
        elif key == 'fmri_mean':
            # Handle fmri_mean: can be None or tensor [1, 1, D, H, W]
            # Filter out None values and check if all are None
            non_none_values = [v for v in values if v is not None]
            if len(non_none_values) == 0:
                collated[key] = None
            elif len(non_none_values) == len(values):
                # All are tensors, use default_collate
                collated[key] = default_collate(non_none_values)
            else:
                # Some are None - pad None values with zeros matching the shape of non-None tensors
                if len(non_none_values) > 0:
                    # Get shape from first non-None tensor
                    ref_shape = non_none_values[0].shape
                    ref_dtype = non_none_values[0].dtype
                    ref_device = non_none_values[0].device
                    # Replace None with zero tensors
                    padded_values = []
                    for v in values:
                        if v is None:
                            padded_values.append(torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device))
                        else:
                            padded_values.append(v)
                    collated[key] = default_collate(padded_values)
                else:
                    collated[key] = None
        elif key == 'fmri':
            # Handle variable temporal dimensions in fMRI
            # Find max temporal dimension
            max_t = max(v.shape[1] if v.ndim >= 2 else 1 for v in values)
            
            # Pad or truncate each sample to max_t
            padded_values = []
            for v in values:
                if v.ndim == 5:  # [1, T, D, H, W]
                    if v.shape[1] < max_t:
                        # Pad with zeros
                        padding = torch.zeros(
                            (v.shape[0], max_t - v.shape[1], v.shape[2], v.shape[3], v.shape[4]),
                            dtype=v.dtype,
                            device=v.device
                        )
                        v = torch.cat([v, padding], dim=1)
                    elif v.shape[1] > max_t:
                        # Truncate
                        v = v[:, :max_t, :, :, :]
                    padded_values.append(v)
                else:
                    padded_values.append(v)
            
            # Now can use default_collate
            collated[key] = default_collate(padded_values)
        elif isinstance(values[0], torch.Tensor):
            # Use default collate for tensors
            collated[key] = default_collate(values)
        else:
            # For other types, keep as list
            collated[key] = values
    
    # Debug statements removed for cleaner output
    # import os
    # rank = os.environ.get('LOCAL_RANK', '0')
    # if len(batch) > 0:
    #     scan_ids = [s.get('scan_id', '?') for s in batch]
    #     print(f"[Collate RANK={rank}] Collation complete. Batch size: {len(batch)}, scan_ids: {scan_ids}", flush=True)
    #     print(f"[Collate RANK={rank}] Collated keys: {list(collated.keys())}, roi_masks in collated: {'roi_masks' in collated}, roi_masks type: {type(collated.get('roi_masks', None))}", flush=True)
    
    return collated

