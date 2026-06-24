"""
Patient-level train/val splitting.

Scan IDs follow ``<patientID>_<session>`` (e.g. ``B10081264_004``); a single
patient can contribute several sessions. Splitting **by patient** ensures no
session from a given patient appears in both train and val (no subject leakage).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Tuple


def patient_id(scan_id: str) -> str:
    """`B10081264_004` -> `B10081264` (everything before the last underscore)."""
    return scan_id.rsplit("_", 1)[0] if "_" in scan_id else scan_id


def patient_level_split(
    scan_ids: List[str], val_frac: float = 0.15, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Return (train_indices, val_indices) into `scan_ids`, split by patient.

    All sessions of a patient go to the same side of the split.
    """
    groups = defaultdict(list)
    for i, sid in enumerate(scan_ids):
        groups[patient_id(sid)].append(i)

    patients = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val = max(1, int(round(len(patients) * val_frac)))
    val_patients = set(patients[:n_val])

    train_idx, val_idx = [], []
    for p in patients:
        (val_idx if p in val_patients else train_idx).extend(groups[p])
    return sorted(train_idx), sorted(val_idx)
