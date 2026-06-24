"""
Corruption-tolerant dataset wrapper.

Some NIfTI files in large cohorts are empty/truncated. `SafeDataset` catches any
exception raised by the underlying `__getitem__` (e.g. nibabel `ImageFileError`)
and retries with another random index, so a handful of bad scans cannot crash a
multi-hour distributed run. It transparently exposes `scan_ids` of the base
dataset for patient-level splitting.
"""
from __future__ import annotations

import random

from torch.utils.data import Dataset


class SafeDataset(Dataset):
    # one-time, class-level notice instead of per-sample spam
    _notified = False

    def __init__(self, base: Dataset, retries: int = 20, verbose: bool = False):
        self.base = base
        self.retries = retries
        self.verbose = verbose
        # expose attributes used elsewhere (e.g. scan_ids for the split)
        self.scan_ids = getattr(base, "scan_ids", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        n = len(self.base)
        for attempt in range(self.retries):
            try:
                return self.base[idx]
            except Exception as e:  # corrupt file, missing modality, etc.
                if attempt == 0 and not SafeDataset._notified:
                    SafeDataset._notified = True
                    print("[SafeDataset] some samples are unreadable and are skipped "
                          "silently (pre-filter should have removed most).", flush=True)
                if self.verbose:
                    sid = self.scan_ids[idx] if self.scan_ids else idx
                    print(f"[SafeDataset] skip {sid}: {type(e).__name__}", flush=True)
                idx = random.randint(0, n - 1)
        # final attempt — let it raise if still failing
        return self.base[idx]
