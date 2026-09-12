import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from eval.quality import (
    POLICY_NOTE,
    QualityContractError,
    QualityPolicy,
    evaluate_4d_pair_quality,
)
from scripts.check_4d_quality import main as quality_cli_main


def _dynamic_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small positive-valued brain with independently varying modes."""
    shape = (9, 10, 11, 16)
    x, y, z = np.indices(shape[:3], dtype=np.float32)
    mask = ((x - 4.0) / 3.4) ** 2 + ((y - 4.5) / 3.8) ** 2 + (
        (z - 5.0) / 4.2
    ) ** 2 <= 1.0
    anatomy = (
        0.70
        + 0.10 * np.sin(1.31 * x)
        + 0.08 * np.cos(1.17 * y)
        + 0.06 * np.sin(1.43 * z)
    )
    frames = []
    for frame in range(shape[-1]):
        phase = 2.0 * np.pi * frame / shape[-1]
        dynamics = (
            0.13 * np.sin(phase + 0.47 * x + 0.13 * y)
            + 0.08 * np.cos(2.0 * phase + 0.29 * y + 0.19 * z)
            + 0.04 * np.sin(3.0 * phase + 0.37 * z)
        )
        frames.append(np.where(mask, anatomy + dynamics, 0.0))
    real = np.stack(frames, axis=-1).astype(np.float32)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -8.0],
            [0.0, 2.5, 0.0, -11.25],
            [0.0, 0.0, 3.0, -15.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return real, mask, affine


def _write_4d(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    tr_seconds: float = 2.4,
) -> None:
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_zooms((*nib.affines.voxel_sizes(affine), tr_seconds))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)


def _write_mask(path: Path, mask: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)


def _anatomical_labels(mask: np.ndarray) -> np.ndarray:
    """Partition the synthetic brain into six nonempty anatomical ROIs."""
    x, y, _ = np.indices(mask.shape)
    x_group = (x >= mask.shape[0] // 2).astype(np.int16)
    y_group = np.minimum(2, (3 * y // mask.shape[1])).astype(np.int16)
    labels = 1 + 3 * x_group + y_group
    return np.where(mask, labels, 0).astype(np.int16)


def _roi_structured_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create six ROIs with deliberately different spectra and FC."""
    original, mask, affine = _dynamic_pair()
    anatomy = original.mean(axis=-1)
    labels = _anatomical_labels(mask)
    phase = 2.0 * np.pi * np.arange(original.shape[-1]) / original.shape[-1]
    signals = {
        1: np.sin(phase),
        2: 0.8 * np.sin(phase) + 0.6 * np.cos(2.0 * phase),
        3: np.sin(2.0 * phase),
        4: np.sin(3.0 * phase),
        5: np.sin(4.0 * phase),
        6: 0.6 * np.sin(3.0 * phase) + 0.8 * np.cos(5.0 * phase),
    }
    real = np.repeat(anatomy[..., None], original.shape[-1], axis=-1)
    for label, signal in signals.items():
        normalized = signal / np.std(signal)
        real[labels == label] += 0.12 * normalized
    real[~mask] = 0.0
    return real.astype(np.float32), mask, affine, labels


def _independent_temporal_surrogate(
    real: np.ndarray, mask: np.ndarray, seed: int = 27
) -> np.ndarray:
    """Break anatomy/dynamics coupling while preserving every global statistic."""
    mean = np.asarray(real.mean(axis=-1, keepdims=True), dtype=np.float64)
    predicted = np.repeat(mean, real.shape[-1], axis=-1)
    centered = np.asarray(real[mask], dtype=np.float64)
    centered -= centered.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    # Permuting centered temporal traces across anatomy is an independent null.
    # It preserves variance, DVARS, Fourier energy, and the temporal Gram
    # eigenvalues exactly while destroying ROI-specific dynamics.
    predicted[mask] += centered[rng.permutation(centered.shape[0])]
    return predicted.astype(np.float32)


def _write_pair(
    tmp_path: Path,
    predicted: np.ndarray | None = None,
    *,
    mask: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    real, default_mask, affine = _dynamic_pair()
    if predicted is None:
        predicted = real
    if mask is None:
        mask = default_mask
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(predicted_path, predicted, affine)
    _write_mask(mask_path, mask, affine)
    return real_path, predicted_path, mask_path


def test_identical_dynamic_pair_passes_all_configurable_gates(tmp_path):
    real_path, predicted_path, mask_path = _write_pair(tmp_path)

    result = evaluate_4d_pair_quality(real_path, predicted_path, mask_path=mask_path)

    assert result["schema"] == "connect4-paired-4d-quality-v1"
    assert result["policy_note"] == POLICY_NOTE
    assert "not paper-reported" in result["policy_note"]
    assert result["geometry"]["canonical_axis_codes"] == ["R", "A", "S"]
    assert result["geometry"]["tr_seconds"] == pytest.approx(2.4)
    assert result["support"]["dice"] == pytest.approx(1.0)
    assert result["support"]["center_of_mass_distance_mm"] == pytest.approx(0.0)
    assert result["outside_mask"]["predicted_outside_mask_max_abs"] == 0.0
    assert result["outside_mask"]["predicted_outside_mask_rms"] == 0.0
    assert result["outside_mask"]["predicted_outside_mask_energy_fraction"] == 0.0
    assert result["outside_mask"]["predicted_outside_mask_leakage_ratio"] == 0.0
    assert result["checks"]["outside_mask_leakage"]["passed"] is True
    assert result["spatial"]["robust_range_ratio"] == pytest.approx(1.0)
    assert result["spatial"]["gradient_rms_ratio"] == pytest.approx(1.0)
    assert result["spatial"]["laplacian_rms_ratio"] == pytest.approx(1.0)
    assert result["spatial"]["high_frequency_rms_ratio"] == pytest.approx(1.0)
    assert result["checks"]["spatial_robust_range_ratio"]["passed"] is True
    assert result["checks"]["spatial_laplacian_ratio"]["passed"] is True
    assert result["spatial"][
        "temporal_mean_high_frequency_correlation"
    ] == pytest.approx(1.0)
    assert result["spatial"]["dynamic_high_frequency_rms_ratio"] == pytest.approx(
        1.0
    )
    assert result["spatial"][
        "dynamic_high_frequency_correlation"
    ] == pytest.approx(1.0)
    assert result["checks"]["dynamic_high_frequency_ratio"]["passed"] is True
    assert (
        result["checks"]["dynamic_high_frequency_correlation"]["passed"] is True
    )
    assert result["temporal"]["temporal_variance_ratio"] == pytest.approx(1.0)
    assert result["temporal"]["dvars_ratio"] == pytest.approx(1.0)
    assert result["temporal"]["dynamic_power_ratio"] == pytest.approx(1.0)
    assert result["temporal"]["effective_rank_ratio"] == pytest.approx(1.0)
    assert result["failed_checks"] == []
    paper = result["paper_metrics"]
    assert paper["metrics"]["mse"]["value"] == pytest.approx(0.0)
    assert paper["metrics"]["ssim"]["value"] == pytest.approx(1.0)
    assert paper["metrics"]["voxel_corr"]["value"] == pytest.approx(1.0)
    assert paper["metrics"]["f2f_corr"]["value"] == pytest.approx(1.0)
    assert paper["metrics"]["roi_corr"]["available"] is False
    assert paper["metrics"]["psnr"]["value"] is None
    assert paper["metrics"]["psnr"]["nonfinite_value"] == "positive_infinity"
    assert paper["distributional_metrics"]["fid"]["available"] is False
    assert paper["distributional_metrics"]["inception_score"]["available"] is False
    for source in result["implementation_sources"].values():
        path = Path(source["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    assert result["structured_temporal"]["available"] is False
    assert (
        result["structured_temporal"]["unavailable_reason"]
        == "roi_label_map_not_supplied"
    )
    assert result["checks"]["structured_temporal_available"]["passed"] is None
    assert result["temporal_collapse_detected"] is False
    assert result["passed"] is True
    assert result["verdict"] == "pass"


def test_identical_pair_passes_required_structured_temporal_gates(tmp_path):
    real, mask, affine = _dynamic_pair()
    real_path, predicted_path, mask_path = _write_pair(tmp_path, real, mask=mask)
    labels_path = tmp_path / "roi_labels.nii.gz"
    _write_mask(labels_path, _anatomical_labels(mask), affine)

    result = evaluate_4d_pair_quality(
        real_path,
        predicted_path,
        mask_path=mask_path,
        roi_labels_path=labels_path,
        require_structured_temporal=True,
    )

    structured = result["structured_temporal"]
    assert structured["available"] is True
    assert structured["roi_count"] == 6
    assert structured["roi_mean_time_series_shape"] == [6, 16]
    assert structured["fc_upper_triangle_edges"] == 15
    assert structured["fc_matrix_correlation"] == pytest.approx(1.0)
    assert structured["power_spectrum_correlation"] == pytest.approx(1.0)
    assert result["paper_metrics"]["metrics"]["roi_corr"]["value"] == pytest.approx(
        1.0
    )
    assert result["checks"]["roi_fc_correlation"]["passed"] is True
    assert result["checks"]["roi_power_spectrum_correlation"]["passed"] is True
    assert result["failed_structured_temporal_checks"] == []
    assert result["passed"] is True


def test_independent_temporal_noise_fools_global_but_fails_roi_structure(tmp_path):
    real, mask, affine, labels = _roi_structured_pair()
    predicted = _independent_temporal_surrogate(real, mask)
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(predicted_path, predicted, affine)
    _write_mask(mask_path, mask, affine)
    labels_path = tmp_path / "roi_labels.nii.gz"
    _write_mask(labels_path, labels, affine)
    policy = QualityPolicy(
        min_support_dice=0.0,
        max_support_com_distance_mm=100.0,
        max_mask_boundary_fraction=1.0,
        min_spatial_gradient_ratio=0.01,
        max_spatial_gradient_ratio=100.0,
        min_spatial_high_frequency_ratio=0.01,
        max_spatial_high_frequency_ratio=100.0,
        min_spatial_high_frequency_correlation=0.80,
        min_roi_fc_correlation=0.80,
        min_roi_power_spectrum_correlation=0.80,
    )

    result = evaluate_4d_pair_quality(
        real_path,
        predicted_path,
        mask_path=mask_path,
        roi_labels_path=labels_path,
        require_structured_temporal=True,
        policy=policy,
    )

    # The independent null preserves every centered temporal trace, just in the
    # wrong anatomical location, so coarse global gates cannot identify it.
    for name in (
        "temporal_variance_ratio",
        "dvars_ratio",
        "dynamic_power_ratio",
        "effective_rank_ratio",
        "near_static_voxel_fraction",
    ):
        assert result["checks"][name]["passed"] is True
    assert result["temporal"]["temporal_variance_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert result["temporal"]["dynamic_power_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert result["temporal_collapse_detected"] is False

    # The phase-invariant anatomical summaries still reject its wrong network
    # organization and ROI-specific spectral composition.
    assert result["checks"]["roi_fc_correlation"]["passed"] is False
    assert result["checks"]["roi_power_spectrum_correlation"]["passed"] is False
    assert set(result["failed_structured_temporal_checks"]) == {
        "roi_fc_correlation",
        "roi_power_spectrum_correlation",
    }
    assert result["passed"] is False
    assert result["verdict"] == "fail_quality_gate"


def test_static_mean_plus_shuffled_dynamics_fails_dynamic_texture_alignment(
    tmp_path,
):
    """A static anatomical overlay cannot conceal mislocated 4D detail."""
    real, mask, _ = _dynamic_pair()
    predicted = _independent_temporal_surrogate(real, mask)
    real_path, predicted_path, mask_path = _write_pair(
        tmp_path, predicted, mask=mask
    )
    policy = QualityPolicy(
        min_support_dice=0.0,
        max_support_com_distance_mm=100.0,
        max_mask_boundary_fraction=1.0,
        min_spatial_robust_range_ratio=0.01,
        max_spatial_robust_range_ratio=100.0,
        min_spatial_gradient_ratio=0.01,
        max_spatial_gradient_ratio=100.0,
        min_spatial_laplacian_ratio=0.01,
        max_spatial_laplacian_ratio=100.0,
        min_spatial_high_frequency_ratio=0.01,
        max_spatial_high_frequency_ratio=100.0,
        min_spatial_high_frequency_correlation=-1.0,
        min_dynamic_high_frequency_ratio=0.01,
        max_dynamic_high_frequency_ratio=100.0,
        min_dynamic_high_frequency_correlation=0.80,
        min_temporal_variance_ratio=0.01,
        max_temporal_variance_ratio=100.0,
        min_dvars_ratio=0.01,
        max_dvars_ratio=100.0,
        min_dynamic_power_ratio=0.01,
        max_dynamic_power_ratio=100.0,
        min_effective_rank_ratio=0.01,
        max_effective_rank_ratio=100.0,
        max_near_static_voxel_fraction=1.0,
    )

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path, policy=policy
    )

    for name in (
        "spatial_high_frequency_ratio",
        "spatial_high_frequency_correlation",
        "dynamic_high_frequency_ratio",
        "temporal_variance_ratio",
        "dvars_ratio",
        "dynamic_power_ratio",
        "effective_rank_ratio",
        "near_static_voxel_fraction",
    ):
        assert result["checks"][name]["passed"] is True
    assert result["spatial"]["dynamic_high_frequency_correlation"] < 0.80
    assert (
        result["checks"]["dynamic_high_frequency_correlation"]["passed"] is False
    )
    assert result["failed_checks"] == ["dynamic_high_frequency_correlation"]
    assert result["temporal_collapse_detected"] is False
    assert result["verdict"] == "fail_quality_gate"


def test_required_structured_temporal_fails_when_labels_are_absent(tmp_path):
    real_path, predicted_path, mask_path = _write_pair(tmp_path)

    result = evaluate_4d_pair_quality(
        real_path,
        predicted_path,
        mask_path=mask_path,
        require_structured_temporal=True,
    )

    assert result["checks"]["structured_temporal_available"]["passed"] is False
    assert result["failed_structured_temporal_checks"] == [
        "structured_temporal_available"
    ]
    assert result["passed"] is False


def test_cli_require_structured_temporal_exits_two_without_labels(tmp_path, capsys):
    real_path, predicted_path, mask_path = _write_pair(tmp_path)

    exit_code = quality_cli_main(
        [
            "--real",
            str(real_path),
            "--predicted",
            str(predicted_path),
            "--mask",
            str(mask_path),
            "--require-structured-temporal",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["checks"]["structured_temporal_available"]["passed"] is False


def test_cli_required_structured_temporal_accepts_labels_and_overrides(
    tmp_path, capsys
):
    real, mask, affine = _dynamic_pair()
    real_path, predicted_path, mask_path = _write_pair(tmp_path, real, mask=mask)
    labels_path = tmp_path / "roi_labels.nii.gz"
    _write_mask(labels_path, _anatomical_labels(mask), affine)

    exit_code = quality_cli_main(
        [
            "--real",
            str(real_path),
            "--predicted",
            str(predicted_path),
            "--mask",
            str(mask_path),
            "--roi-labels",
            str(labels_path),
            "--require-structured-temporal",
            "--min-structured-temporal-rois",
            "6",
            "--min-roi-fc-correlation",
            "0.95",
            "--min-roi-power-spectrum-correlation",
            "0.95",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["policy"]["min_structured_temporal_rois"] == 6
    assert report["checks"]["roi_fc_correlation"]["passed"] is True
    assert report["checks"]["roi_power_spectrum_correlation"]["passed"] is True


def test_static_prediction_fails_closed_as_temporal_collapse(tmp_path):
    real, mask, _ = _dynamic_pair()
    static_prediction = np.repeat(
        real.mean(axis=-1, keepdims=True), real.shape[-1], axis=-1
    )
    real_path, predicted_path, mask_path = _write_pair(
        tmp_path, static_prediction, mask=mask
    )

    result = evaluate_4d_pair_quality(real_path, predicted_path, mask_path=mask_path)

    assert result["support"]["dice"] == pytest.approx(1.0)
    assert result["temporal"]["temporal_variance_ratio"] < 1e-10
    assert result["temporal"]["dvars_ratio"] < 1e-10
    assert result["temporal"]["dynamic_power_ratio"] < 1e-10
    assert result["temporal"]["effective_rank_ratio"] < 1e-10
    assert result["temporal"]["near_static_voxel_fraction"] == pytest.approx(1.0)
    assert len(result["failed_temporal_checks"]) == 5
    assert result["temporal_collapse_detected"] is True
    assert result["passed"] is False
    assert result["verdict"] == "fail_temporal_collapse"


def test_zero_prediction_has_strict_json_record_and_fails_closed(tmp_path):
    real, mask, _ = _dynamic_pair()
    real_path, predicted_path, mask_path = _write_pair(
        tmp_path, np.zeros_like(real), mask=mask
    )

    result = evaluate_4d_pair_quality(real_path, predicted_path, mask_path=mask_path)

    assert result["support"]["predicted_support_voxels"] == 0
    assert result["support"]["predicted_center_of_mass_world_mm"] is None
    assert result["support"]["center_of_mass_distance_mm"] is None
    assert result["checks"]["support_center_of_mass"]["passed"] is False
    assert result["temporal_collapse_detected"] is True
    assert result["verdict"] == "fail_temporal_collapse"
    json.dumps(result, allow_nan=False)


def test_low_variance_high_rank_mismatch_is_not_mislabeled_as_collapse(tmp_path):
    real, mask, affine = _dynamic_pair()
    frames = real.shape[-1]
    time = np.linspace(-1.0, 1.0, frames, dtype=np.float32)

    # Make the reference overwhelmingly low-rank with a large global drift,
    # while retaining enough local signal for every reference metric to exist.
    real_with_drift = real + 2.5 * time[None, None, None, :]
    rng = np.random.default_rng(91)
    predicted = np.repeat(real.mean(axis=-1, keepdims=True), frames, axis=-1)
    mask_coordinates = np.argwhere(mask)
    dynamic_coordinates = mask_coordinates[
        rng.permutation(mask_coordinates.shape[0])[: int(0.60 * mask_coordinates.shape[0])]
    ]
    dynamic_noise = 0.90 * rng.normal(
        size=(dynamic_coordinates.shape[0], frames)
    ).astype(np.float32)
    predicted[
        dynamic_coordinates[:, 0],
        dynamic_coordinates[:, 1],
        dynamic_coordinates[:, 2],
        :,
    ] += dynamic_noise
    predicted[~mask] = 0.0

    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_4d(real_path, real_with_drift, affine)
    _write_4d(predicted_path, predicted, affine)
    _write_mask(mask_path, mask, affine)

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path
    )

    assert len(result["failed_temporal_checks"]) >= 3
    assert result["temporal"]["effective_rank_ratio"] > 2.0
    assert "effective_rank_ratio" not in result["failed_temporal_collapse_checks"]
    assert result["temporal_quality_mismatch_detected"] is True
    assert result["temporal_collapse_detected"] is False
    assert result["passed"] is False
    assert result["verdict"] == "fail_quality_gate"


@pytest.mark.parametrize("mismatch", ["shape", "affine", "tr"])
def test_grid_affine_and_tr_mismatches_fail_closed(tmp_path, mismatch):
    real, mask, affine = _dynamic_pair()
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_mask(mask_path, mask, affine)

    predicted = real
    predicted_affine = affine.copy()
    predicted_tr = 2.4
    expected = ""
    if mismatch == "shape":
        predicted = real[:-1]
        expected = "shapes differ"
    elif mismatch == "affine":
        predicted_affine[0, 3] += 0.25
        expected = "affines differ"
    else:
        predicted_tr = 2.5
        expected = "TR differ"
    _write_4d(
        predicted_path,
        predicted,
        predicted_affine,
        tr_seconds=predicted_tr,
    )

    with pytest.raises(QualityContractError, match=expected):
        evaluate_4d_pair_quality(real_path, predicted_path, mask_path=mask_path)


def test_mask_grid_mismatch_fails_closed_instead_of_resampling(tmp_path):
    real, mask, affine = _dynamic_pair()
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(predicted_path, real, affine)
    shifted_affine = affine.copy()
    shifted_affine[2, 3] += 1.0
    _write_mask(mask_path, mask, shifted_affine)

    with pytest.raises(QualityContractError, match="mask affine differs"):
        evaluate_4d_pair_quality(real_path, predicted_path, mask_path=mask_path)


def test_support_displacement_and_fov_contact_are_flagged(tmp_path):
    real, mask, _ = _dynamic_pair()
    shifted = ndimage.shift(real, shift=(2, 0, 0, 0), order=0, mode="constant")
    boundary_mask = mask.copy()
    boundary_mask[0, 3:7, 4:8] = True
    real_path, predicted_path, mask_path = _write_pair(
        tmp_path, shifted, mask=boundary_mask
    )
    policy = QualityPolicy(
        min_support_dice=0.0,
        max_support_com_distance_mm=1.0,
        max_mask_boundary_fraction=0.0,
        min_spatial_gradient_ratio=0.01,
        max_spatial_gradient_ratio=100.0,
        min_spatial_high_frequency_ratio=0.01,
        max_spatial_high_frequency_ratio=100.0,
        min_temporal_variance_ratio=0.01,
        max_temporal_variance_ratio=100.0,
        min_dvars_ratio=0.01,
        max_dvars_ratio=100.0,
        min_dynamic_power_ratio=0.01,
        max_dynamic_power_ratio=100.0,
        min_effective_rank_ratio=0.01,
        max_effective_rank_ratio=100.0,
        max_near_static_voxel_fraction=1.0,
    )

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path, policy=policy
    )

    assert result["support"]["center_of_mass_distance_mm"] > 1.0
    assert result["support"]["maximum_mask_boundary_fraction"] > 0.0
    assert result["checks"]["support_center_of_mass"]["passed"] is False
    assert result["checks"]["mask_field_of_view"]["passed"] is False
    assert result["temporal_collapse_detected"] is False
    assert result["verdict"] == "fail_quality_gate"


def test_isolated_distant_outside_mask_artifact_fails_explicit_gate(tmp_path):
    real, mask, _ = _dynamic_pair()
    predicted = real.copy()
    reference = float(
        np.percentile(
            np.sqrt(np.mean(np.square(real, dtype=np.float64), axis=-1))[mask],
            99.0,
        )
    )
    assert not mask[0, 0, 0]
    predicted[0, 0, 0, 0] = 0.02 * reference
    real_path, predicted_path, mask_path = _write_pair(
        tmp_path, predicted, mask=mask
    )

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path
    )

    assert result["outside_mask"][
        "predicted_outside_mask_leakage_ratio"
    ] == pytest.approx(0.02, rel=1e-5)
    assert result["outside_mask"]["predicted_outside_mask_energy_fraction"] < 1e-6
    assert result["checks"]["outside_mask_leakage"]["passed"] is False
    assert result["failed_checks"] == ["outside_mask_leakage"]
    assert result["verdict"] == "fail_quality_gate"


def test_exterior_artifacts_do_not_contaminate_masked_spatial_metrics(tmp_path):
    real, mask, _ = _dynamic_pair()
    (tmp_path / "clean").mkdir()
    clean_real, clean_predicted, clean_mask = _write_pair(
        tmp_path / "clean", real, mask=mask
    )
    clean_result = evaluate_4d_pair_quality(
        clean_real, clean_predicted, mask_path=clean_mask
    )

    dirty = real.copy()
    dirty[0, 0, 0, :] = 100.0
    (tmp_path / "dirty").mkdir()
    dirty_real, dirty_predicted, dirty_mask = _write_pair(
        tmp_path / "dirty", dirty, mask=mask
    )
    dirty_result = evaluate_4d_pair_quality(
        dirty_real, dirty_predicted, mask_path=dirty_mask
    )

    assert dirty_result["checks"]["outside_mask_leakage"]["passed"] is False
    assert dirty_result["spatial"] == clean_result["spatial"]


def test_structural_only_voxels_are_excluded_from_every_target_metric(tmp_path):
    real, target_validity, affine = _dynamic_pair()
    structural = target_validity.copy()
    structural[0, 0, 0] = True
    clean = real.copy()
    dirty = real.copy()
    dirty[0, 0, 0, :] = 100.0

    real_path = tmp_path / "real.nii.gz"
    clean_path = tmp_path / "clean.nii.gz"
    dirty_path = tmp_path / "dirty.nii.gz"
    validity_path = tmp_path / "target_validity_mask.nii.gz"
    structural_path = tmp_path / "structural_brain_mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(clean_path, clean, affine)
    _write_4d(dirty_path, dirty, affine)
    _write_mask(validity_path, target_validity, affine)
    _write_mask(structural_path, structural, affine)

    clean_result = evaluate_4d_pair_quality(
        real_path,
        clean_path,
        mask_path=validity_path,
        structural_mask_path=structural_path,
    )
    dirty_result = evaluate_4d_pair_quality(
        real_path,
        dirty_path,
        mask_path=validity_path,
        structural_mask_path=structural_path,
    )

    for section in ("support", "spatial", "temporal", "paper_metrics"):
        assert dirty_result[section] == clean_result[section]
    assert dirty_result["outside_mask"][
        "predicted_outside_mask_leakage_ratio"
    ] == 0.0
    assert dirty_result["outside_mask"]["predicted_outside_mask_rms"] == 0.0


def test_target_validity_must_be_binary_and_inside_structural_support(tmp_path):
    real, target_validity, affine = _dynamic_pair()
    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    validity_path = tmp_path / "target_validity_mask.nii.gz"
    structural_path = tmp_path / "structural_brain_mask.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(predicted_path, real, affine)

    nonbinary = target_validity.astype(np.float32)
    nonbinary[tuple(np.argwhere(target_validity)[0])] = 0.5
    image = nib.Nifti1Image(nonbinary, affine)
    nib.save(image, validity_path)
    _write_mask(structural_path, target_validity, affine)
    with pytest.raises(QualityContractError, match="exactly binary"):
        evaluate_4d_pair_quality(
            real_path,
            predicted_path,
            mask_path=validity_path,
            structural_mask_path=structural_path,
        )

    validity_outside = target_validity.copy()
    validity_outside[0, 0, 0] = True
    _write_mask(validity_path, validity_outside, affine)
    with pytest.raises(QualityContractError, match="outside structural"):
        evaluate_4d_pair_quality(
            real_path,
            predicted_path,
            mask_path=validity_path,
            structural_mask_path=structural_path,
        )


def test_paired_quality_rejects_any_configured_roi_without_valid_support(tmp_path):
    real, structural, affine = _dynamic_pair()
    labels = _anatomical_labels(structural)
    missing_label = int(labels.max())
    target_validity = structural & (labels != missing_label)
    real[labels == missing_label] = 0.0

    real_path = tmp_path / "real.nii.gz"
    predicted_path = tmp_path / "predicted.nii.gz"
    validity_path = tmp_path / "target_validity_mask.nii.gz"
    structural_path = tmp_path / "structural_brain_mask.nii.gz"
    labels_path = tmp_path / "labels.nii.gz"
    _write_4d(real_path, real, affine)
    _write_4d(predicted_path, real, affine)
    _write_mask(validity_path, target_validity, affine)
    _write_mask(structural_path, structural, affine)
    _write_mask(labels_path, labels, affine)

    with pytest.raises(
        QualityContractError,
        match="every configured ROI must have non-empty target-validity support",
    ):
        evaluate_4d_pair_quality(
            real_path,
            predicted_path,
            mask_path=validity_path,
            structural_mask_path=structural_path,
            roi_labels_path=labels_path,
            require_structured_temporal=True,
        )

def test_spatial_blur_fails_gradient_and_high_frequency_gates(tmp_path):
    real, mask, _ = _dynamic_pair()
    # Smooth within the support using normalized convolution so a hard zero
    # background cannot create a false gradient at the brain boundary.
    weights = ndimage.gaussian_filter(
        mask.astype(np.float32), sigma=1.5, mode="constant"
    )
    blurred = np.zeros_like(real)
    for frame in range(real.shape[-1]):
        numerator = ndimage.gaussian_filter(
            real[..., frame], sigma=1.5, mode="constant"
        )
        blurred[..., frame] = np.where(mask, numerator / np.maximum(weights, 1e-8), 0.0)
    real_path, predicted_path, mask_path = _write_pair(tmp_path, blurred, mask=mask)
    policy = QualityPolicy(
        min_support_dice=0.0,
        min_spatial_gradient_ratio=0.80,
        max_spatial_gradient_ratio=2.0,
        min_spatial_laplacian_ratio=0.80,
        max_spatial_laplacian_ratio=2.0,
        min_spatial_high_frequency_ratio=0.80,
        max_spatial_high_frequency_ratio=2.0,
        min_temporal_variance_ratio=0.001,
        max_temporal_variance_ratio=100.0,
        min_dvars_ratio=0.001,
        max_dvars_ratio=100.0,
        min_dynamic_power_ratio=0.001,
        max_dynamic_power_ratio=100.0,
        min_effective_rank_ratio=0.001,
        max_effective_rank_ratio=100.0,
        near_static_relative_variance=1e-6,
        max_near_static_voxel_fraction=1.0,
    )

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path, policy=policy
    )

    assert result["spatial"]["gradient_rms_ratio"] < 0.80
    assert result["spatial"]["laplacian_rms_ratio"] < 0.80
    assert result["spatial"]["high_frequency_rms_ratio"] < 0.80
    assert result["checks"]["spatial_gradient_ratio"]["passed"] is False
    assert result["checks"]["spatial_laplacian_ratio"]["passed"] is False
    assert result["checks"]["spatial_high_frequency_ratio"]["passed"] is False
    assert result["temporal_collapse_detected"] is False
    assert result["verdict"] == "fail_quality_gate"


def test_unrelated_texture_with_matching_energy_fails_correlation_gate(tmp_path):
    real, mask, _ = _dynamic_pair()
    # A fixed voxel permutation preserves every temporal trajectory and the
    # support-wide signal distribution, but puts fine detail in wrong places.
    scrambled = real.copy()
    rng = np.random.default_rng(17)
    scrambled[mask] = real[mask][rng.permutation(int(mask.sum()))]
    real_path, predicted_path, mask_path = _write_pair(tmp_path, scrambled, mask=mask)
    policy = QualityPolicy(
        min_support_dice=0.0,
        max_support_com_distance_mm=100.0,
        max_mask_boundary_fraction=1.0,
        min_spatial_gradient_ratio=0.01,
        max_spatial_gradient_ratio=100.0,
        min_spatial_high_frequency_ratio=0.01,
        max_spatial_high_frequency_ratio=100.0,
        min_spatial_high_frequency_correlation=0.80,
        min_temporal_variance_ratio=0.001,
        max_temporal_variance_ratio=100.0,
        min_dvars_ratio=0.001,
        max_dvars_ratio=100.0,
        min_dynamic_power_ratio=0.001,
        max_dynamic_power_ratio=100.0,
        min_effective_rank_ratio=0.001,
        max_effective_rank_ratio=100.0,
        near_static_relative_variance=1e-6,
        max_near_static_voxel_fraction=1.0,
    )

    result = evaluate_4d_pair_quality(
        real_path, predicted_path, mask_path=mask_path, policy=policy
    )

    assert 0.01 <= result["spatial"]["high_frequency_rms_ratio"] <= 100.0
    assert result["checks"]["spatial_high_frequency_ratio"]["passed"] is True
    assert result["spatial"]["temporal_mean_high_frequency_correlation"] < 0.80
    assert result["checks"]["spatial_high_frequency_correlation"]["passed"] is False
    assert result["temporal_collapse_detected"] is False
    assert result["verdict"] == "fail_quality_gate"


def test_policy_rejects_invalid_bounds_and_noninteger_counts():
    with pytest.raises(ValueError, match="exceeds"):
        QualityPolicy(
            min_temporal_variance_ratio=2.0,
            max_temporal_variance_ratio=1.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        QualityPolicy(spatial_sample_frames=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"in \[1, 5\]"):
        QualityPolicy(temporal_failures_for_collapse=6)
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        QualityPolicy(min_spatial_high_frequency_correlation=1.1)
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        QualityPolicy(min_dynamic_high_frequency_correlation=1.1)
    with pytest.raises(ValueError, match="exceeds"):
        QualityPolicy(
            min_dynamic_high_frequency_ratio=2.0,
            max_dynamic_high_frequency_ratio=1.0,
        )
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        QualityPolicy(min_roi_fc_correlation=-1.1)
    with pytest.raises(ValueError, match=r"integer >= 3"):
        QualityPolicy(min_structured_temporal_rois=2)
    with pytest.raises(ValueError, match="nonnegative and finite"):
        QualityPolicy(max_outside_mask_leakage_ratio=-0.1)
    with pytest.raises(ValueError, match="nonnegative and finite"):
        QualityPolicy(max_outside_mask_leakage_ratio=float("inf"))
