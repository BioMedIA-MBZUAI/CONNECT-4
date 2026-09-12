from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_LABEL_TO_CHANNEL,
    CANONICAL_ROI_MAPPING_SHA256,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
)
from data.dataset_precomputed import Connect4PrecomputedDataset
from eval import metrics, postseal, postseal_metrics
from eval.development_quality import development_tensor_identity
from eval.postseal_metrics import METRIC_DEFINITIONS_SOURCE
from eval.quality import QualityContractError, evaluate_4d_pair_quality_arrays
from eval.visualize import plot_real_vs_synthetic_4d
from training.train import _require_supervised_batch_contracts
from utils.compat import strict_zip


@pytest.mark.parametrize("module", [metrics, postseal_metrics])
@pytest.mark.parametrize(
    "name",
    [
        "voxel_correlation",
        "frame_to_frame_correlation",
        "mse",
        "ssim3d",
        "psnr",
    ],
)
def test_every_paired_metric_rejects_a_missing_target_validity_mask(module, name):
    values = torch.ones((1, 1, 4, 2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="target-validity mask is required"):
        getattr(module, name)(values, values, mask=None)


@pytest.mark.parametrize("module", [metrics, postseal_metrics])
def test_roi_metric_rejects_a_missing_target_validity_mask(module):
    values = torch.ones((1, 1, 4, 2, 2, 2), dtype=torch.float32)
    roi_masks = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="target-validity mask is required"):
        module.roi_correlation(values, values, roi_masks, mask=None)


@pytest.mark.parametrize("module", [metrics, postseal_metrics])
def test_metric_set_and_accumulator_reject_a_missing_target_validity_mask(module):
    values = torch.ones((1, 1, 4, 2, 2, 2), dtype=torch.float32)
    roi_masks = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="target-validity mask is required"):
        module.compute_all(values, values, roi_masks, mask=None)
    accumulator = module.SynthesisMetricAccumulator()
    with pytest.raises(ValueError, match="target-validity mask is required"):
        accumulator.update(values, values, roi_masks, mask=None)
    assert accumulator.num_samples == 0


def _supervised_batch(contract: str) -> dict:
    return {
        "fmri": torch.ones((2, 1, 4, 2, 2, 2)),
        "brain_mask": torch.ones((2, 1, 2, 2, 2)),
        "target_validity_mask": torch.ones((2, 1, 2, 2, 2)),
        "target_validity_mask_contract": [contract, contract],
        "structure_to_roi_idx": dict(CANONICAL_ROI_LABEL_TO_CHANNEL),
        "roi_mapping_sha256": [
            CANONICAL_ROI_MAPPING_SHA256,
            CANONICAL_ROI_MAPPING_SHA256,
        ],
        "roi_label_ids": [
            tuple(CANONICAL_ROI_LABEL_IDS),
            tuple(CANONICAL_ROI_LABEL_IDS),
        ],
    }


def test_training_rejects_wrong_profile_contract_and_roi_reordering():
    profile = "a4-native-recovery-v1"
    expected = TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[profile]
    batch = _supervised_batch(expected)
    _require_supervised_batch_contracts(batch, protocol_profile=profile)

    wrong = dict(batch)
    wrong["target_validity_mask_contract"] = [
        TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE["paper-a4-adni-fmriprep-v1"]
    ] * 2
    with pytest.raises(RuntimeError, match="differs from protocol profile"):
        _require_supervised_batch_contracts(wrong, protocol_profile=profile)

    reordered = dict(batch)
    reordered_mapping = dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    reordered_mapping[2], reordered_mapping[3] = (
        reordered_mapping[3],
        reordered_mapping[2],
    )
    reordered["structure_to_roi_idx"] = reordered_mapping
    with pytest.raises(RuntimeError, match="canonical 32-ROI"):
        _require_supervised_batch_contracts(reordered, protocol_profile=profile)


def test_training_rejects_swapped_structural_and_target_validity_masks():
    profile = "a4-native-recovery-v1"
    batch = _supervised_batch(
        TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[profile]
    )
    batch["fmri"][:, :, :, 0] = 0
    batch["target_validity_mask"][:, :, 0] = 0
    _require_supervised_batch_contracts(batch, protocol_profile=profile)

    swapped = dict(batch)
    swapped["brain_mask"], swapped["target_validity_mask"] = (
        batch["target_validity_mask"],
        batch["brain_mask"],
    )
    with pytest.raises(RuntimeError, match="may have been swapped"):
        _require_supervised_batch_contracts(swapped, protocol_profile=profile)


def test_development_identity_records_the_admitted_recovery_mask_contract():
    profile = "a4-native-recovery-v1"
    target = torch.ones((4, 2, 2, 2), dtype=torch.float32)
    structural = torch.ones((2, 2, 2), dtype=torch.float32)
    validity = torch.ones_like(structural)
    identity = development_tensor_identity(
        validity,
        kind="target_validity_mask",
        brain_mask=structural,
        target=target,
        target_validity_contract=(
            TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[profile]
        ),
    )
    assert (
        identity["derivation_contract"]
        == (TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[profile])
    )


def _roi_quality_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spatial_shape = (4, 4, 4)
    real = np.ones((*spatial_shape, 4), dtype=np.float32)
    predicted = real.copy()
    validity = np.ones(spatial_shape, dtype=np.uint8)
    roi_masks = np.zeros((32, *spatial_shape), dtype=np.uint8)
    for channel, flat_index in enumerate(range(32)):
        roi_masks[channel].reshape(-1)[flat_index] = 1
    return real, predicted, validity, roi_masks


@pytest.mark.parametrize(
    ("channel_count", "label_ids"),
    [
        (32, tuple(reversed(CANONICAL_ROI_LABEL_IDS))),
        (31, CANONICAL_ROI_LABEL_IDS[:-1]),
        (33, (*CANONICAL_ROI_LABEL_IDS, 99)),
    ],
)
def test_quality_rejects_reordered_missing_or_extra_roi_channels(
    channel_count: int,
    label_ids: tuple[int, ...],
):
    real, predicted, validity, roi_masks = _roi_quality_inputs()
    if channel_count < 32:
        roi_masks = roi_masks[:channel_count]
    elif channel_count > 32:
        roi_masks = np.concatenate([roi_masks, roi_masks[:1]], axis=0)
    with pytest.raises(QualityContractError, match="canonical 32-ROI"):
        evaluate_4d_pair_quality_arrays(
            real,
            predicted,
            validity,
            affine=np.diag([3.0, 3.0, 3.0, 1.0]),
            tr_seconds=3.0,
            roi_masks=roi_masks,
            roi_label_ids=label_ids,
            require_canonical_roi_mapping=True,
        )


def test_dataset_rejects_runtime_mutation_of_canonical_roi_semantics(monkeypatch):
    Connect4PrecomputedDataset._require_canonical_roi_contract()
    monkeypatch.setattr(
        Connect4PrecomputedDataset,
        "ROI_LABEL_IDS",
        tuple(reversed(CANONICAL_ROI_LABEL_IDS)),
    )
    with pytest.raises(RuntimeError, match="canonical 32-ROI"):
        Connect4PrecomputedDataset._require_canonical_roi_contract()


def test_development_visualization_binds_both_masks_and_rejects_swapping(
    tmp_path: Path,
):
    real = np.zeros((4, 7, 7, 7), dtype=np.float32)
    real[:, 2:5, 2:5, 2:5] = np.arange(1, 5, dtype=np.float32)[:, None, None, None]
    predicted = real.copy()
    structural = np.zeros((7, 7, 7), dtype=np.uint8)
    structural[1:6, 1:6, 1:6] = 1
    target_validity = np.zeros_like(structural)
    target_validity[2:5, 2:5, 2:5] = 1

    with pytest.raises(ValueError, match="outside independent structural support"):
        plot_real_vs_synthetic_4d(
            real,
            predicted,
            tmp_path / "swapped.png",
            brain_mask=target_validity,
            target_validity_mask=structural,
        )

    output = tmp_path / "development.png"
    plot_real_vs_synthetic_4d(
        real,
        predicted,
        output,
        brain_mask=structural,
        target_validity_mask=target_validity,
    )
    record = json.loads(output.with_suffix(".png.display.json").read_text())
    inputs = record["inputs"]
    assert inputs["structural_mask_sha256"] != inputs["target_validity_mask_sha256"]
    assert inputs["structural_mask_semantic_role"].startswith("target-independent")
    assert inputs["target_validity_mask_semantic_role"].startswith("paired-scoring")
    claimed = record.pop("record_sha256")
    assert (
        claimed
        == hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def test_postseal_metric_aggregation_rejects_definition_source_mismatch():
    values = {
        name: {"available": True, "value": 1.0}
        for name in ("mse", "voxel_corr", "roi_corr", "f2f_corr", "ssim", "psnr")
    }
    record = {
        "quality": {
            "paper_metrics": {
                "schema": "connect4-single-subject-paper-metrics-v1",
                "definitions_source": METRIC_DEFINITIONS_SOURCE,
                "metrics": values,
            }
        }
    }
    assert postseal._aggregate_paper_metrics([record])["mse"] == 1.0
    record["quality"]["paper_metrics"]["definitions_source"] = "eval/metrics.py"
    with pytest.raises(postseal.PostsealEvaluationError, match="mse"):
        postseal._aggregate_paper_metrics([record])


def _write_nifti(path: Path, values: np.ndarray) -> None:
    image = nib.Nifti1Image(values, np.diag([3.0, 3.0, 3.0, 1.0]))
    if values.ndim == 4:
        image.header.set_xyzt_units("mm", "sec")
        image.header.set_zooms((3.0, 3.0, 3.0, 3.0))
    else:
        image.header.set_xyzt_units("mm")
        image.header.set_zooms((3.0, 3.0, 3.0))
    nib.save(image, path)


def test_postseal_pair_binds_protocol_contract_and_exact_roi_set(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(postseal, "EXPECTED_SHAPE", (8, 8, 8, 4))
    monkeypatch.setattr(postseal, "EXPECTED_SPATIAL_SHAPE", (8, 8, 8))
    monkeypatch.setattr(postseal, "_MAX_STRUCTURAL_BRAIN_BOUNDARY_FRACTION", 1.0)

    structural = np.zeros((8, 8, 8), dtype=np.uint8)
    structural[1:7, 1:7, 1:7] = 1
    real = np.zeros((8, 8, 8, 4), dtype=np.float32)
    real[structural.astype(bool)] = 0.5
    predicted = real.copy()
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    coordinates = np.argwhere(structural)
    for label_id, coordinate in strict_zip(
        CANONICAL_ROI_LABEL_IDS,
        coordinates[: len(CANONICAL_ROI_LABEL_IDS)],
    ):
        labels[tuple(coordinate)] = label_id

    paths = [
        tmp_path / name
        for name in ("real.nii.gz", "pred.nii.gz", "mask.nii.gz", "labels.nii.gz")
    ]
    for path, values in strict_zip(paths, (real, predicted, structural, labels)):
        _write_nifti(path, values)

    profile = "a4-native-recovery-v1"
    result = postseal.validate_exact_pair(*paths, protocol_profile=profile)
    assert (
        result["target_validity_mask"]["contract"]
        == (TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE[profile])
    )
    assert result["canonical_roi_mapping_sha256"] == CANONICAL_ROI_MAPPING_SHA256

    labels[tuple(coordinates[0])] = 99
    _write_nifti(paths[-1], labels)
    with pytest.raises(postseal.PostsealEvaluationError, match="canonical 32-ROI"):
        postseal.validate_exact_pair(*paths, protocol_profile=profile)
