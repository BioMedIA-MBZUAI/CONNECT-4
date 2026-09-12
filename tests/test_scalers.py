import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from utils.scalers import FeatureScalerManager, SCALER_METADATA_SCHEMA


def _fitted_scaler():
    return StandardScaler().fit(np.asarray([[0.0, 1.0], [2.0, 3.0]]))


def test_scaler_loader_is_metadata_bound_and_records_its_directory(tmp_path):
    manager = FeatureScalerManager()
    manager.scalers = {"radiomics": _fitted_scaler(), "anatcl": _fitted_scaler()}
    manager.save_scalers(tmp_path)

    metadata = json.loads((tmp_path / "scaler_metadata.json").read_text())
    assert metadata == {
        "schema": SCALER_METADATA_SCHEMA,
        "scaler_names": ["anatcl", "radiomics"],
        "num_scalers": 2,
    }
    loaded = FeatureScalerManager()
    loaded.load_scalers(tmp_path)
    assert loaded.scaler_dir == tmp_path
    assert set(loaded.scalers) == {"radiomics", "anatcl"}


def test_scaler_loader_rejects_missing_metadata_and_unlisted_stale_files(tmp_path):
    with pytest.raises(RuntimeError, match="metadata is required"):
        FeatureScalerManager().load_scalers(tmp_path)

    manager = FeatureScalerManager()
    manager.scalers = {"radiomics": _fitted_scaler()}
    manager.save_scalers(tmp_path)
    (tmp_path / "stale_scaler.pkl").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="differ from metadata"):
        FeatureScalerManager().load_scalers(tmp_path)


def test_production_scaler_fit_uses_every_fixed_training_scan(tmp_path):
    from scripts.preprocess_graphs import compute_scalers_from_dataset

    class Dataset:
        scan_ids = ["S3", "S1", "S2"]

        def __len__(self):
            return len(self.scan_ids)

        def radiomics_for_scan(self, scan_id):
            value = float(self.scan_ids.index(scan_id))
            return {2: torch.tensor([value, value + 1.0])}

        def _load_anatcl_embeddings(self, scan_id):
            value = float(self.scan_ids.index(scan_id))
            return {"roi": torch.tensor([value + 2.0, value + 3.0])}

    dataset = Dataset()
    compute_scalers_from_dataset(dataset, Path(tmp_path), sample_indices=[2, 0, 1])
    provenance = json.loads(
        (tmp_path / "scalers" / "training_partition.json").read_text()
    )
    assert provenance["training_partition_scan_ids"] == ["S1", "S2", "S3"]
    assert provenance["fitted_scan_ids"] == provenance["training_partition_scan_ids"]

    with pytest.raises(TypeError, match="unexpected keyword argument 'max_samples'"):
        compute_scalers_from_dataset(
            dataset, Path(tmp_path), sample_indices=[0, 1, 2], max_samples=1
        )
