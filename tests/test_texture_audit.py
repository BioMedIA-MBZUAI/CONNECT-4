import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import pytest
from scipy import ndimage

from scripts import visualize_texture_audit as audit
from scripts.visualize_4d_comparison import ComparisonVolume
from utils.source_provenance import sha256_file


def _comparison(*, smoothed: bool, frames: int = 8) -> ComparisonVolume:
    shape = (15, 15, 15)
    x, y, z = np.indices(shape, dtype=np.float64)
    mask = (x - 7.0) ** 2 + (y - 7.0) ** 2 + (z - 7.0) ** 2 <= 6.0**2
    anatomy = 0.03 * x + 0.02 * y - 0.01 * z
    checkerboard = ((x.astype(int) + y.astype(int) + z.astype(int)) % 2) * 2.0 - 1.0
    real_frames = []
    for frame in range(frames):
        phase = 2.0 * np.pi * frame / frames
        value = anatomy + 0.28 * checkerboard * (1.0 + 0.15 * np.sin(phase))
        value += 0.04 * np.sin(0.6 * x + phase)
        real_frames.append(np.where(mask, value, 0.0))
    real = np.stack(real_frames, axis=-1).astype(np.float32)
    if smoothed:
        predicted = np.stack(
            [
                np.where(
                    mask,
                    ndimage.gaussian_filter(real[..., frame], sigma=1.0),
                    0.0,
                )
                for frame in range(frames)
            ],
            axis=-1,
        ).astype(np.float32)
    else:
        predicted = real.copy()
    labels = np.where(mask, 1 + (x >= 7) + 2 * (y >= 7), 0).astype(np.int32)
    return ComparisonVolume(
        real=real,
        predicted=predicted,
        mask=mask,
        affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        tr_seconds=2.0,
        mask_description="synthetic spherical mask",
        mask_resampled=False,
        roi_labels=labels,
        roi_description="synthetic labels",
    )


def _diagnostics(comparison: ComparisonVolume):
    return audit.compute_texture_diagnostics(
        comparison,
        sample_frame_count=8,
        detail_sigma_voxels=0.7,
        interior_erosion_voxels=2,
        patch_size_voxels=5,
        max_patches=8,
        spectrum_bins=6,
        high_frequency_cycles_per_voxel=0.30,
        texture_retention_reference=0.80,
    )


def _write_4d(path: Path, values: np.ndarray, *, tr: float = 2.0) -> Path:
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
    image.header.set_zooms((3.0, 3.0, 3.0, tr))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)
    return path


def _write_mask(path: Path, values: np.ndarray) -> Path:
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    image = nib.Nifti1Image(np.asarray(values, dtype=np.uint8), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def _prediction_manifest(path: Path, predicted: Path, *, prediction_hash=None) -> Path:
    record = {
        "schema": "connect4-target-blind-finetuned-prediction-v1",
        "scan_id": "B915-test-only",
        "evaluation_role": "blind-validation-model-selection",
        "sampling": {
            "method": "DDIM",
            "steps": 30,
            "target_roi_conditioner": False,
        },
        "target_blind_ordering": {
            "real_target_opened_before_prediction_saved": False,
            "real_target_hashed_before_prediction_saved": False,
            "prediction_saved_before_target_metrics": True,
        },
        "prediction": {
            "path": str(predicted),
            "sha256": prediction_hash or sha256_file(predicted),
        },
    }
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def test_patch_centers_are_deterministic_complete_nine_voxel_cubes():
    mask = np.ones((21, 21, 21), dtype=bool)

    first = audit._farthest_patch_centers(mask, patch_size=9, max_patches=12)
    second = audit._farthest_patch_centers(mask, patch_size=9, max_patches=12)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (12, 3)
    radius = 4
    for center in first:
        patch = tuple(
            slice(int(value) - radius, int(value) + radius + 1)
            for value in center
        )
        assert mask[patch].shape == (9, 9, 9)
        assert mask[patch].all()


def test_patch_centers_exclude_cross_only_boundary_candidate_and_fail_shortfall():
    shape = (31, 31, 31)
    mask = np.zeros(shape, dtype=bool)
    # This exact 9^3 block supplies one and only one complete-cube center.
    mask[2:11, 2:11, 2:11] = True
    complete_center = (6, 6, 6)

    # A taxicab-radius-four diamond survives four iterations of SciPy's
    # default connectivity-one erosion, but it cannot contain a full 9^3 cube.
    cross_only_center = (24, 24, 24)
    coordinates = np.indices(shape)
    taxicab_distance = sum(
        np.abs(coordinates[axis] - cross_only_center[axis]) for axis in range(3)
    )
    mask |= taxicab_distance <= 4
    default_cross_eligible = ndimage.binary_erosion(mask, iterations=4)
    assert default_cross_eligible[cross_only_center]

    centers = audit._farthest_patch_centers(mask, patch_size=9, max_patches=1)
    np.testing.assert_array_equal(centers, np.asarray([complete_center]))
    assert not np.any(np.all(centers == np.asarray(cross_only_center), axis=1))

    with pytest.raises(ValueError, match=r"only 1 complete 9x9x9.*2 are required"):
        audit._farthest_patch_centers(mask, patch_size=9, max_patches=2)


def test_identity_preserves_all_texture_metrics_and_raw_arrays():
    comparison = _comparison(smoothed=False)
    real_before = comparison.real.copy()
    predicted_before = comparison.predicted.copy()

    values = _diagnostics(comparison)

    for ratio in values["diagnostic_ratios"].values():
        assert ratio == pytest.approx(1.0, abs=1e-10)
    assert values["aggregate_detail_correlation"] == pytest.approx(1.0, abs=1e-10)
    assert values["near_nyquist_tail_power_ratio"] == pytest.approx(
        1.0, abs=1e-10
    )
    gate = audit.build_texture_quality_gate(
        values,
        enforce=True,
        enforce_anti_gaming=True,
    )
    assert gate["mode"] == "combined-release-gate"
    assert gate["release_gate_passed"] is True
    assert gate["anti_gaming_guards_passed"] is True
    assert all(result["passed"] for result in gate["anti_gaming_results"].values())
    assert not values["spatial_texture_attenuation_flag"]
    assert values["attenuated_metrics_below_reference"] == []
    np.testing.assert_array_equal(comparison.real, real_before)
    np.testing.assert_array_equal(comparison.predicted, predicted_before)


def test_gaussian_smoothing_is_exposed_by_independent_texture_summaries():
    values = _diagnostics(_comparison(smoothed=True))

    assert values["aggregate_detail_rms_ratio"] < 0.50
    assert values["derivative"]["laplacian_rms_ratio"] < 0.30
    assert values["local_high_frequency_power_ratio"] < 0.10
    assert values["spatial_texture_attenuation_flag"]
    assert len(values["attenuated_metrics_below_reference"]) >= 3
    assert values["selected_frame_rule"].startswith("largest real")
    gate = audit.build_texture_quality_gate(
        values,
        enforce=True,
        enforce_anti_gaming=False,
    )
    assert gate["all_metrics_meet_reference"] is False
    assert gate["release_gate_passed"] is False
    assert gate["exit_code"] == audit.TEXTURE_GATE_FAILURE_EXIT_CODE


def test_b915_gain326_negative_control_fails_both_anti_gaming_guards():
    # Frozen B915 V6-step-8 mean-only sigma=0.5/gain=3.26 negative control,
    # selected NIfTI SHA-256:
    # fe9015152ca321c88c633808f58314212bb66344cb151980c8561a98b1c6a352
    # The scalar regression fixture keeps this unit test portable while binding
    # the policy logic to the measured complete-cube-v2 audit.
    values = _diagnostics(_comparison(smoothed=False))
    values["diagnostic_ratios"] = {
        "voxel_scale_detail_rms_ratio": 1.315141600311002,
        "gradient_rms_ratio": 1.2661060294897721,
        "laplacian_rms_ratio": 1.4260892286084352,
        "local_high_frequency_power_ratio": 0.8024547073229094,
    }
    values["aggregate_detail_correlation"] = 0.3612868093534857
    values["near_nyquist_tail_power_ratio"] = 1.306822633655653
    values["near_nyquist_boundary_cycles_per_voxel"] = 0.60

    amplitude_only_gate = audit.build_texture_quality_gate(
        values,
        enforce=True,
        enforce_anti_gaming=False,
    )
    gate = audit.build_texture_quality_gate(
        values,
        enforce=True,
        enforce_anti_gaming=True,
    )

    # Historical amplitude-only enforcement is deliberately unchanged until
    # the caller explicitly opts into the new guards.
    assert amplitude_only_gate["mode"] == "enforced-release-gate"
    assert amplitude_only_gate["anti_gaming_guards_passed"] is False
    assert amplitude_only_gate["release_gate_passed"] is True
    assert amplitude_only_gate["exit_code"] == 0
    assert gate["all_metrics_meet_reference"] is True
    assert all(result["passed"] for result in gate["results"].values())
    assert gate["would_pass_if_enforced"] is True
    assert gate["anti_gaming_guards_passed"] is False
    assert gate["anti_gaming_results"]["aggregate_detail_correlation"][
        "passed"
    ] is False
    assert gate["anti_gaming_results"]["near_nyquist_tail_power_ratio"][
        "passed"
    ] is False
    assert gate["would_pass_if_all_enforced"] is False
    assert gate["release_gate_passed"] is False
    assert gate["exit_code"] == audit.TEXTURE_GATE_FAILURE_EXIT_CODE


@pytest.mark.parametrize(
    "nonfinite_metric",
    ("aggregate_detail_correlation", "near_nyquist_tail_power_ratio"),
)
def test_anti_gaming_guards_fail_closed_on_nonfinite(nonfinite_metric):
    values = _diagnostics(_comparison(smoothed=False))
    values[nonfinite_metric] = float("nan")

    gate = audit.build_texture_quality_gate(
        values,
        enforce=False,
        enforce_anti_gaming=True,
    )

    result = gate["anti_gaming_results"][nonfinite_metric]
    assert result["value"] is None
    assert result["passed"] is False
    assert gate["anti_gaming_guards_passed"] is False
    assert gate["release_gate_passed"] is False
    assert gate["exit_code"] == audit.TEXTURE_GATE_FAILURE_EXIT_CODE


def test_manifest_is_source_and_output_bound_and_uses_shared_windows(tmp_path):
    comparison = _comparison(smoothed=True)
    real = _write_4d(tmp_path / "real.nii.gz", comparison.real)
    predicted = _write_4d(tmp_path / "predicted.nii.gz", comparison.predicted)
    mask = _write_mask(tmp_path / "mask.nii.gz", comparison.mask)
    prediction_manifest = _prediction_manifest(
        tmp_path / "prediction_manifest.json", predicted
    )

    outputs = audit.generate_texture_audit(
        real,
        predicted,
        mask,
        tmp_path / "audit",
        prediction_manifest_path=prediction_manifest,
        prefix="B915_test",
        sample_frame_count=8,
        detail_sigma_voxels=0.7,
        interior_erosion_voxels=2,
        patch_size_voxels=5,
        max_patches=8,
        spectrum_bins=6,
        high_frequency_cycles_per_voxel=0.30,
    )

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    stored_hash = manifest.pop("record_sha256")
    assert stored_hash == hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest["schema"] == audit.SCHEMA
    assert manifest["target_blind_provenance"]["verified"]
    assert manifest["target_blind_provenance"][
        "prediction_verified_before_real_opened"
    ]
    assert manifest["quality_gate"]["mode"] == "descriptive-only"
    assert manifest["quality_gate"]["enforced"] is False
    assert manifest["quality_gate"]["amplitude_retention_enforced"] is False
    assert manifest["quality_gate"]["anti_gaming_guards_enforced"] is False
    assert manifest["quality_gate"]["verdict"] == "not-enforced"
    assert manifest["quality_gate"]["release_gate_passed"] is None
    assert manifest["quality_gate"]["exit_code"] == 0
    assert not manifest["quality_gate"]["would_pass_if_enforced"]
    assert set(manifest["quality_gate"]["results"]) == set(
        audit.TEXTURE_GATE_METRICS
    )
    assert set(manifest["quality_gate"]["anti_gaming_results"]) == set(
        audit.ANTI_GAMING_GUARD_METRICS
    )
    assert manifest["quality_gate"]["anti_gaming_policy"][
        "paper_reported_metric"
    ] is False
    assert "implementation release policy" in manifest["quality_gate"][
        "anti_gaming_policy"
    ]["scope"]
    assert manifest["display_contract"]["selection_uses_prediction"] is False
    assert manifest["display_contract"]["image_interpolation"] == "nearest"
    assert manifest["display_contract"]["fMRI_registration_or_resampling"] is False
    assert manifest["display_contract"][
        "raw_real_predicted_shared_absolute_limit"
    ] > 0
    assert manifest["display_contract"][
        "high_pass_real_predicted_shared_absolute_limit"
    ] > 0
    assert manifest["display_contract"]["local_spectrum_patch_count"] == 8
    assert manifest["display_contract"][
        "near_nyquist_boundary_cycles_per_voxel"
    ] == pytest.approx(0.60)
    assert "mean radial cycles/voxel" in manifest["display_contract"][
        "local_spectrum_frequency_coordinate"
    ]
    assert "shell-mean cutoff" in manifest["display_contract"][
        "near_nyquist_cutoff_application"
    ]
    assert "mean mode frequency" in manifest["quality_gate"][
        "anti_gaming_policy"
    ]["near_nyquist_tail_power_ratio"]["cutoff_application"]
    assert "every voxel" in manifest["display_contract"][
        "local_spectrum_patch_selection"
    ]
    diagnostics = manifest["diagnostics"]
    assert diagnostics["patch_count"] == diagnostics["patch_count_requested"] == 8
    assert diagnostics["patch_eligibility_structuring_element"] == "full cubic ones"
    assert "FOV exterior is ineligible" in diagnostics["patch_support_contract"]
    for source in manifest["implementation_sources"].values():
        assert sha256_file(source["path"]) == source["sha256"]
    assert sha256_file(outputs["png"]) == manifest["outputs"][
        "texture_audit_png"
    ]["sha256"]
    with Image.open(outputs["png"]) as rendered:
        assert rendered.width > 1200
        assert rendered.height > 1200


def test_bad_prediction_hash_fails_before_real_pair_is_loaded(tmp_path, monkeypatch):
    comparison = _comparison(smoothed=False)
    real = _write_4d(tmp_path / "real.nii.gz", comparison.real)
    predicted = _write_4d(tmp_path / "predicted.nii.gz", comparison.predicted)
    mask = _write_mask(tmp_path / "mask.nii.gz", comparison.mask)
    prediction_manifest = _prediction_manifest(
        tmp_path / "bad_prediction_manifest.json",
        predicted,
        prediction_hash="0" * 64,
    )
    opened_pair = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal opened_pair
        opened_pair = True
        raise AssertionError("real/predicted pair must not be opened")

    monkeypatch.setattr(audit, "load_comparison", forbidden_load)
    with pytest.raises(ValueError, match="prediction NIfTI SHA-256"):
        audit.generate_texture_audit(
            real,
            predicted,
            mask,
            tmp_path / "audit",
            prediction_manifest_path=prediction_manifest,
        )
    assert not opened_pair


def test_figure_target_usage_note_never_claims_unverified_target_blindness():
    unverified = audit._figure_target_usage_note(None)
    verified = audit._figure_target_usage_note({"verified": True})
    same_case = audit._figure_target_usage_note(
        None, same_case_target_used_for_optimization=True
    )

    assert "not certified" in unverified
    assert "generalization claim" in unverified
    assert "opened only for this post-inference audit" not in unverified
    assert "Authenticated target-blind prediction" in verified
    assert "opened only for this post-inference audit" in verified
    assert "same-case optimization" in same_case
    assert "not generalization" in same_case


@pytest.mark.parametrize(
    ("smoothed", "enforced", "expected_exit", "expected_mode", "expected_verdict"),
    (
        (False, True, 0, "enforced-release-gate", "pass"),
        (True, True, audit.TEXTURE_GATE_FAILURE_EXIT_CODE, "enforced-release-gate", "fail"),
        (True, False, 0, "descriptive-only", "not-enforced"),
    ),
)
def test_cli_texture_gate_exit_semantics_and_target_blind_provenance(
    tmp_path,
    smoothed,
    enforced,
    expected_exit,
    expected_mode,
    expected_verdict,
):
    comparison = _comparison(smoothed=smoothed)
    real = _write_4d(tmp_path / "real.nii.gz", comparison.real)
    predicted = _write_4d(tmp_path / "predicted.nii.gz", comparison.predicted)
    mask = _write_mask(tmp_path / "mask.nii.gz", comparison.mask)
    prediction_manifest = _prediction_manifest(
        tmp_path / "prediction_manifest.json", predicted
    )
    out_dir = tmp_path / "cli_audit"
    arguments = [
        "--real",
        str(real),
        "--pred",
        str(predicted),
        "--mask",
        str(mask),
        "--prediction-manifest",
        str(prediction_manifest),
        "--out-dir",
        str(out_dir),
        "--prefix",
        "B915_gate_test",
        "--sample-frames",
        "8",
        "--detail-sigma-voxels",
        "0.7",
        "--interior-erosion-voxels",
        "2",
        "--patch-size-voxels",
        "5",
        "--max-patches",
        "8",
        "--spectrum-bins",
        "6",
        "--high-frequency-cycles-per-voxel",
        "0.30",
    ]
    if enforced:
        arguments.append("--enforce-texture-retention")

    exit_code = audit.main(arguments)

    assert exit_code == expected_exit
    manifest_path = out_dir / "B915_gate_test_texture_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = manifest["quality_gate"]
    assert manifest["schema"] == "connect4-post-inference-texture-audit-v3"
    assert gate["schema"] == "connect4-texture-retention-quality-gate-v3"
    assert gate["mode"] == expected_mode
    assert gate["enforced"] is enforced
    assert gate["amplitude_retention_enforced"] is enforced
    assert gate["anti_gaming_guards_enforced"] is False
    assert gate["verdict"] == expected_verdict
    assert gate["exit_code"] == expected_exit
    assert gate["policy"]["operator"] == ">="
    assert gate["policy"]["minimum_inclusive"] == pytest.approx(0.80)
    assert gate["policy"]["all_metrics_required"]
    assert set(gate["policy"]["required_metrics"]) == set(
        audit.TEXTURE_GATE_METRICS
    )
    assert all(
        result["passed"] for result in gate["results"].values()
    ) is (not smoothed)
    assert gate["would_pass_if_enforced"] is (not smoothed)
    assert gate["release_gate_passed"] is (
        (not smoothed) if enforced else None
    )
    assert gate["anti_gaming_policy"]["aggregate_detail_correlation"][
        "minimum_inclusive"
    ] == pytest.approx(0.40)
    assert gate["anti_gaming_policy"]["near_nyquist_tail_power_ratio"][
        "maximum_inclusive"
    ] == pytest.approx(1.25)
    provenance = manifest["target_blind_provenance"]
    assert provenance["verified"]
    assert provenance["prediction_verified_before_real_opened"]
    assert provenance["prediction_sha256"] == sha256_file(predicted)


def test_cli_explicit_anti_gaming_guards_pass_identity_and_report_policy(tmp_path):
    comparison = _comparison(smoothed=False)
    real = _write_4d(tmp_path / "real.nii.gz", comparison.real)
    predicted = _write_4d(tmp_path / "predicted.nii.gz", comparison.predicted)
    mask = _write_mask(tmp_path / "mask.nii.gz", comparison.mask)
    prediction_manifest = _prediction_manifest(
        tmp_path / "prediction_manifest.json", predicted
    )
    out_dir = tmp_path / "combined_cli_audit"

    exit_code = audit.main(
        [
            "--real",
            str(real),
            "--pred",
            str(predicted),
            "--mask",
            str(mask),
            "--prediction-manifest",
            str(prediction_manifest),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "B915_combined_gate_test",
            "--sample-frames",
            "8",
            "--detail-sigma-voxels",
            "0.7",
            "--interior-erosion-voxels",
            "2",
            "--patch-size-voxels",
            "5",
            "--max-patches",
            "8",
            "--spectrum-bins",
            "6",
            "--high-frequency-cycles-per-voxel",
            "0.30",
            "--near-nyquist-cycles-per-voxel",
            "0.60",
            "--texture-retention-reference",
            "0.80",
            "--minimum-detail-correlation",
            "0.95",
            "--maximum-near-nyquist-tail-power-ratio",
            "1.05",
            "--enforce-texture-retention",
            "--enforce-texture-anti-gaming",
        ]
    )

    assert exit_code == 0
    manifest = json.loads(
        (out_dir / "B915_combined_gate_test_texture_audit.json").read_text(
            encoding="utf-8"
        )
    )
    gate = manifest["quality_gate"]
    assert gate["mode"] == "combined-release-gate"
    assert gate["amplitude_retention_enforced"] is True
    assert gate["anti_gaming_guards_enforced"] is True
    assert gate["all_metrics_meet_reference"] is True
    assert gate["anti_gaming_guards_passed"] is True
    assert gate["release_gate_passed"] is True
    assert gate["anti_gaming_policy"]["aggregate_detail_correlation"][
        "minimum_inclusive"
    ] == pytest.approx(0.95)
    assert gate["anti_gaming_policy"]["near_nyquist_tail_power_ratio"][
        "maximum_inclusive"
    ] == pytest.approx(1.05)
    assert gate["anti_gaming_policy"]["near_nyquist_tail_power_ratio"][
        "boundary_cycles_per_voxel"
    ] == pytest.approx(0.60)
@pytest.mark.parametrize("name", ["scan_texture_audit.png", "scan_texture_audit.json"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_descriptor_safe_texture_write_never_touches_linked_sentinel(
    tmp_path: Path, name: str, link_kind: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"immutable sentinel")
    destination = output / name
    if link_kind == "symlink":
        destination.symlink_to(sentinel)
    else:
        os.link(sentinel, destination)
    directory_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(FileExistsError):
            audit._write_output_new(  # noqa: SLF001
                destination, b"attacker-controlled replacement", directory_fd=directory_fd
            )
    finally:
        os.close(directory_fd)
    assert sentinel.read_bytes() == b"immutable sentinel"
