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
    Functional targets are deliberately not padded or truncated here.  The
    certified preprocessing contract requires every target to have the exact
    configured temporal and spatial shape, so ``default_collate`` must fail if
    a non-conforming sample reaches the batch boundary.
    """
    # Debug statements removed for cleaner output
    # import os
    # rank = os.environ.get('LOCAL_RANK', '0')
    # if len(batch) > 0 and isinstance(batch[0].get('scan_id', ''), str):
    #     scan_id = batch[0].get('scan_id', 'unknown')
    #     print(f"[Collate RANK={rank}] Collating batch of {len(batch)} samples (scan_ids: {[s.get('scan_id', '?') for s in batch]})", flush=True)
    
    if not batch:
        raise ValueError("cannot collate an empty CONNECT-4 batch")
    expected_keys = set(batch[0])
    for index, sample in enumerate(batch[1:], start=1):
        if set(sample) != expected_keys:
            raise ValueError(
                f"sample {index} has different fields from sample 0; "
                "a mixed data contract cannot be batched"
            )
    keys = batch[0].keys()
    
    collated = {}
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Handle non-tensor items
        if key == 'patch_distributions':
            # Keep as list of lists
            collated[key] = values
        elif key == 'structure_to_roi_idx':
            if any(value != values[0] for value in values[1:]):
                raise ValueError("structure_to_roi_idx differs between subjects")
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
            # A mean target is derived from the certified fMRI itself. Mixed
            # presence or shape would indicate a broken sample contract.
            collated[key] = default_collate(values)
        elif key == 'fmri':
            collated[key] = default_collate(values)
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
