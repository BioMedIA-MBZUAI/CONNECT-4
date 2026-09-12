import math
import hashlib

import pytest
import torch
import torch.nn as nn

from architecture_contract import FRAMEWISE_SSIM_CONTRACT
from eval.metrics import (
    SynthesisMetricAccumulator,
    compute_all,
    frechet_distance_from_features,
    roi_correlation,
    ssim3d,
    voxel_correlation,
)
from models.connect4 import validate_published_loss_weights
from models.feature_extractors import (
    Serialized4DFeatureExtractor,
    build_pretrained_4d_extractor,
)
from models.losses import (
    Connect4Loss,
    FCMatrixLoss,
    PerceptualLoss,
    RegionHistogramLoss,
    SSIM3DLoss,
    TemporalCoherenceLoss,
    VolumeLoss,
)


class _ConstantLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, pred, target, **kwargs):
        return pred.new_tensor(self.value)


class _MetricExtractor(nn.Module):
    def forward(self, x):
        flat = x.flatten(1)
        features = torch.stack((flat.mean(1), flat.std(1)), dim=1)
        logits = torch.stack((flat.mean(1), -flat.mean(1)), dim=1)
        return {"features": features, "logits": logits}


class _FeatureOnlyExtractor(nn.Module):
    def forward(self, x):
        return x.flatten(1).mean(1, keepdim=True)


class _StampedBrainLMExtractor(_FeatureOnlyExtractor):
    connect4_model_name = "brainlm"
    connect4_adapter_contract = "connect4-brainlm-4d-adapter-v1"
    connect4_source_revision = "a" * 40


class _ContextualMaskProbe(nn.Module):
    connect4_requires_context = True

    def __init__(self):
        super().__init__()
        self.inputs = {}

    def forward(self, volume, *, context, input_role):
        self.inputs[input_role] = volume.detach().clone()
        identity = {"record_sha256": "a" * 64}
        return {
            "features": volume.flatten(1).sum(dim=1, keepdim=True),
            "context_sha256": identity["record_sha256"],
            "context_identity": identity,
            "input_role": input_role,
        }


def _volumes(batch=2):
    target = torch.arange(batch * 4 * 8, dtype=torch.float32).reshape(
        batch, 1, 4, 2, 2, 2
    )
    pred = target + 0.1
    roi_masks = torch.zeros(batch, 2, 2, 2, 2)
    roi_masks[:, 0, 0] = 1
    roi_masks[:, 1, 1] = 1
    brain_mask = roi_masks.sum(dim=1, keepdim=True)
    return pred, target, roi_masks, brain_mask


def test_temporal_coherence_is_scale_normalized_and_brain_masked():
    target = torch.zeros(1, 1, 3, 1, 1, 2)
    target[0, 0, :, 0, 0, 0] = torch.tensor([0.45, 0.455, 0.46])
    pred = target.clone()
    pred[0, 0, :, 0, 0, 1] = torch.tensor([0.0, 4.0, 8.0])  # outside mask
    mask = torch.tensor([[[[[1.0, 0.0]]]]])
    loss = TemporalCoherenceLoss()
    assert loss.contract == "connect4-target-delta-relative-l1-v1"
    assert loss(pred, target, mask=mask).item() == pytest.approx(0.0)

    # A static temporal mean receives unit loss even though the real BOLD
    # changes are only 0.005 on a baseline near 0.46.
    pred[0, 0, :, 0, 0, 0] = target[0, 0, :, 0, 0, 0].mean()
    assert loss(pred, target, mask=mask).item() == pytest.approx(1.0)

    scaled_target = target.clone()
    scaled_target[0, 0, :, 0, 0, 0] = torch.tensor([0.45, 0.46, 0.47])
    scaled_static = scaled_target.mean(dim=2, keepdim=True).expand_as(scaled_target)
    assert loss(scaled_static, scaled_target, mask=mask).item() == pytest.approx(1.0)


def test_ssim_scores_corresponding_frames_not_only_the_temporal_mean():
    assert SSIM3DLoss.contract == FRAMEWISE_SSIM_CONTRACT
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(7) for _ in range(3)), indexing="ij")
    )
    texture = ((coordinates.sum(dim=0) % 2).float() * 2.0 - 1.0)[
        None, None, None
    ]
    target = torch.cat((texture, -texture), dim=2)
    prediction = target.flip(dims=(2,))
    mask = torch.ones(1, 1, 7, 7, 7)

    # Both 4D sequences have exactly the same temporal-mean volume. A
    # temporal-mean-only SSIM therefore returned a perfect score for this
    # frame-wise inversion.
    torch.testing.assert_close(
        prediction.mean(dim=2), target.mean(dim=2), rtol=0.0, atol=0.0
    )
    loss = SSIM3DLoss(frame_chunk_size=1)(prediction, target, mask=mask)
    assert loss.item() > 0.5
    assert ssim3d(prediction, target, mask) < 0.5


def test_connect4_uses_all_published_weights_in_total():
    loss = Connect4Loss(perceptual_feature_extractor=_FeatureOnlyExtractor())
    assert loss.w["voxel"] == 1.0
    assert loss.w["ssim"] == 0.5
    assert loss.w["fc"] == 0.3
    assert loss.w["temporal"] == 0.2
    assert loss.w["perceptual"] == 0.1

    values = {
        "ssim": 1, "voxel": 2, "volume": 3, "region_hist": 4,
        "temporal": 5, "perceptual": 6, "fc": 7,
    }
    for name, value in values.items():
        setattr(loss, name, _ConstantLoss(value))
    pred, target, roi_masks, brain_mask = _volumes(batch=1)
    terms = loss(pred, target, mask=brain_mask, roi_masks=roi_masks)
    expected = sum(loss.w[name] * value for name, value in values.items())
    assert terms["total"].item() == pytest.approx(expected)


def test_target_validity_masks_every_loss_metric_and_distributional_feature():
    torch.manual_seed(91)
    target = torch.rand(2, 1, 3, 4, 4, 4)
    target[..., 0, :, :] = 0
    validity = torch.ones(2, 1, 4, 4, 4)
    validity[..., 0, :, :] = 0
    roi_masks = torch.zeros(2, 2, 4, 4, 4)
    roi_masks[:, 0, :2] = 1
    roi_masks[:, 1, 2:] = 1
    baseline = target + 0.05 * validity.unsqueeze(2)
    changed = baseline.clone()
    changed[..., 0, :, :] = 1_000.0

    objective = Connect4Loss(
        perceptual_feature_extractor=_FeatureOnlyExtractor()
    )
    baseline_terms = objective(
        baseline, target, mask=validity, roi_masks=roi_masks
    )
    changed_terms = objective(
        changed, target, mask=validity, roi_masks=roi_masks
    )
    for name in baseline_terms:
        torch.testing.assert_close(changed_terms[name], baseline_terms[name])

    baseline_metrics = compute_all(
        baseline, target, roi_masks=roi_masks, mask=validity
    )
    changed_metrics = compute_all(
        changed, target, roi_masks=roi_masks, mask=validity
    )
    assert changed_metrics == pytest.approx(baseline_metrics)

    baseline_accumulator = SynthesisMetricAccumulator(_MetricExtractor())
    changed_accumulator = SynthesisMetricAccumulator(_MetricExtractor())
    baseline_accumulator.update(baseline, target, roi_masks, validity)
    changed_accumulator.update(changed, target, roi_masks, validity)
    torch.testing.assert_close(
        torch.cat(changed_accumulator._generated_features),
        torch.cat(baseline_accumulator._generated_features),
    )
    torch.testing.assert_close(
        torch.cat(changed_accumulator._generated_logits),
        torch.cat(baseline_accumulator._generated_logits),
    )

    differentiated = changed.clone().requires_grad_(True)
    objective(
        differentiated, target, mask=validity, roi_masks=roi_masks
    )["total"].backward()
    assert torch.equal(
        differentiated.grad[..., 0, :, :],
        torch.zeros_like(differentiated.grad[..., 0, :, :]),
    )
    assert differentiated.grad[..., 1:, :, :].abs().sum() > 0


def test_contextual_perceptual_extractor_receives_only_valid_target_support():
    extractor = _ContextualMaskProbe()
    loss = PerceptualLoss(extractor)
    target = torch.ones(1, 1, 2, 2, 2, 2)
    prediction = target.clone().requires_grad_(True)
    prediction.data[..., 0, 0, 0] = 99
    mask = torch.ones(1, 1, 2, 2, 2)
    mask[..., 0, 0, 0] = 0
    value = loss(
        prediction,
        target,
        mask=mask,
        brainlm_context={"authenticated": True},
    )
    assert value == 0
    assert extractor.inputs["prediction"][..., 0, 0, 0].eq(0).all()
    assert extractor.inputs["target"][..., 0, 0, 0].eq(0).all()


def test_every_configured_roi_requires_target_validity_support_per_subject():
    pred, target, roi_masks, validity = _volumes(batch=2)
    roi_masks[1, 1] = 0
    expected_error = "every configured ROI must have non-empty target-validity"

    objective = Connect4Loss(
        perceptual_feature_extractor=_FeatureOnlyExtractor()
    )
    with pytest.raises(ValueError, match=expected_error):
        objective(pred, target, mask=validity, roi_masks=roi_masks)
    for regional_loss in (VolumeLoss(), RegionHistogramLoss(), FCMatrixLoss()):
        with pytest.raises(ValueError, match=expected_error):
            regional_loss(pred, target, roi_masks=roi_masks)
        with pytest.raises(ValueError, match="ROI masks are required"):
            regional_loss(pred, target, roi_masks=None)
    with pytest.raises(ValueError, match=expected_error):
        roi_correlation(pred, target, roi_masks, mask=validity)
    accumulator = SynthesisMetricAccumulator(_MetricExtractor())
    with pytest.raises(ValueError, match=expected_error):
        accumulator.update(pred, target, roi_masks, validity)
    assert accumulator.num_samples == 0


def test_production_loss_config_cannot_drift_from_published_lambdas():
    validate_published_loss_weights({})
    with pytest.raises(ValueError, match=r"training\.loss\.temporal=0\.2"):
        validate_published_loss_weights({"temporal": 0.0})


def test_roi_and_voxel_correlations_average_every_subject_not_only_first():
    target = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1, 1).repeat(2, 1, 1, 1, 1, 1)
    pred = target.clone()
    pred[1] = target[1].flip(1)  # reverse time after dropping batch dimension
    roi_masks = torch.ones(2, 1, 1, 1, 1)
    mask = torch.ones(2, 1, 1, 1, 1)
    assert roi_correlation(pred, target, roi_masks, mask=mask) == pytest.approx(
        0.0, abs=1e-6
    )
    assert voxel_correlation(pred, target, mask) == pytest.approx(0.0, abs=1e-6)


def test_compute_all_reports_exact_non_distributional_paper_set():
    pred, target, roi_masks, brain_mask = _volumes()
    metrics = compute_all(pred, target, roi_masks=roi_masks, mask=brain_mask)
    assert set(metrics) == {"mse", "ssim", "voxel_corr", "roi_corr", "f2f_corr", "psnr"}


def test_dataset_accumulator_requires_multiple_subjects_for_fid_and_is():
    pred, target, roi_masks, brain_mask = _volumes()
    accumulator = SynthesisMetricAccumulator(_MetricExtractor())
    accumulator.update(pred[:1], target[:1], roi_masks[:1], brain_mask[:1])
    with pytest.raises(RuntimeError, match="at least two"):
        accumulator.compute()

    accumulator.update(pred[1:], target[1:], roi_masks[1:], brain_mask[1:])
    metrics = accumulator.compute()
    assert set(metrics) == {
        "mse", "ssim", "voxel_corr", "roi_corr", "f2f_corr", "psnr", "fid", "is"
    }
    assert math.isfinite(metrics["fid"])
    assert math.isfinite(metrics["is"])


def test_accumulator_weights_unequal_batch_sizes_by_subject_count():
    pred, target, roi_masks, brain_mask = _volumes(batch=3)
    pred = target + torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1, 1, 1)
    accumulator = SynthesisMetricAccumulator()
    accumulator.update(pred[:1], target[:1], roi_masks[:1], brain_mask[:1])
    accumulator.update(pred[1:], target[1:], roi_masks[1:], brain_mask[1:])
    assert accumulator.compute()["mse"] == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)


def test_accumulator_does_not_commit_partial_update_when_is_logits_missing():
    pred, target, roi_masks, brain_mask = _volumes(batch=1)
    accumulator = SynthesisMetricAccumulator(
        _FeatureOnlyExtractor(), require_is_logits=True,
    )
    with pytest.raises(RuntimeError, match="logits"):
        accumulator.update(pred, target, roi_masks, brain_mask)
    assert accumulator.num_samples == 0


def test_fid_rejects_single_sample_feature_sets():
    with pytest.raises(ValueError, match="at least two"):
        frechet_distance_from_features(torch.ones(1, 2), torch.ones(1, 2))


def test_roi_union_brain_mask_fallback_api_is_absent():
    import models.connect4 as connect4_module

    assert not hasattr(connect4_module, "brain_mask_from_roi")


def test_legacy_brainlm_adapter_is_rejected_before_missing_weight_fallback(tmp_path):
    spec = {
        "enabled": True,
        "name": "brainlm",
        "checkpoint": str(tmp_path / "missing.pt"),
        "adapter_contract": "connect4-brainlm-4d-adapter-v1",
    }
    with pytest.raises(ValueError, match="contextual-perceptual-v2"):
        build_pretrained_4d_extractor(
            spec, expected_name="brainlm", purpose="test", device=torch.device("cpu")
        )


def test_legacy_brainlm_adapter_cannot_reach_generic_digest_loader(tmp_path):
    checkpoint = tmp_path / "brainlm.pt"
    checkpoint.write_bytes(b"not an authenticated BrainLM module")

    without_digest = {
        "enabled": True,
        "name": "brainlm",
        "checkpoint": str(checkpoint),
        "adapter_contract": "connect4-brainlm-4d-adapter-v1",
    }
    with pytest.raises(ValueError, match="contextual-perceptual-v2"):
        build_pretrained_4d_extractor(
            without_digest,
            expected_name="brainlm",
            purpose="test",
            device=torch.device("cpu"),
        )

    wrong_digest = dict(
        without_digest,
        checkpoint_sha256="0" * 64,
        source_revision="a" * 40,
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != "0" * 64
    with pytest.raises(ValueError, match="contextual-perceptual-v2"):
        build_pretrained_4d_extractor(
            wrong_digest,
            expected_name="brainlm",
            purpose="test",
            device=torch.device("cpu"),
        )


def test_generic_serialized_module_cannot_enter_brainlm_builder(tmp_path):
    checkpoint = tmp_path / "toy.pt"
    torch.save(_FeatureOnlyExtractor(), checkpoint)
    spec = {
        "enabled": True,
        "name": "brainlm",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_revision": "a" * 40,
        "adapter_contract": "connect4-brainlm-4d-adapter-v1",
    }
    with pytest.raises(ValueError, match="contextual-perceptual-v2"):
        build_pretrained_4d_extractor(
            spec,
            expected_name="brainlm",
            purpose="test",
            device=torch.device("cpu"),
        )


def test_even_stamped_serialized_module_is_categorically_rejected_as_brainlm(tmp_path):
    checkpoint = tmp_path / "brainlm_adapter.pt"
    torch.save(_StampedBrainLMExtractor(), checkpoint)
    spec = {
        "enabled": True,
        "name": "brainlm",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_revision": "a" * 40,
        "adapter_contract": "connect4-brainlm-4d-adapter-v1",
    }
    with pytest.raises(RuntimeError, match="generic serialized 4D module"):
        Serialized4DFeatureExtractor.from_checkpoint(
            str(checkpoint),
            name="brainlm",
            expected_sha256=spec["checkpoint_sha256"],
            source_revision=spec["source_revision"],
        )
