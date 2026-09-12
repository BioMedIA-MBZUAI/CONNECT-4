import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import pytest

from scripts import visualize_4d_comparison as visual


VISUALIZATION_OUTPUT_NAMES = (
    "paired_frames_montage.png",
    "paired_temporal_diagnostics.png",
    "paired_spatial_detail.png",
    "paired_outside_mask_leakage.png",
    "paired_structured_dynamics.png",
    "paired_4d_comparison.mp4",
    "paired_4d_comparison.gif",
    "paired_visualization_manifest.json",
)


def _write_4d(path: Path, data: np.ndarray, affine: np.ndarray, tr: float = 2.0):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_zooms((*nib.affines.voxel_sizes(affine), float(tr)))
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, path)
    return path


def _write_mask(path: Path, data: np.ndarray, affine: np.ndarray):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


def _roi_labels(mask: np.ndarray) -> np.ndarray:
    x, y, _z = np.indices(mask.shape)
    labels = 1 + 3 * (x >= mask.shape[0] // 2) + np.minimum(
        2, 3 * y // mask.shape[1]
    )
    return np.where(mask, labels, 0).astype(np.uint8)


def _paired_paths(tmp_path: Path, *, affine=None, frames: int = 6):
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    shape = (5, 6, 7)
    x, y, z = np.indices(shape, dtype=np.float32)
    spatial = x + 0.5 * y - 0.25 * z
    real = np.stack(
        [spatial + 0.2 * frame for frame in range(frames)], axis=-1
    ).astype(np.float32)
    predicted = (0.85 * real + 0.1 * np.sin(real)).astype(np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    mask[[0, -1], :, :] = 0
    mask[:, [0, -1], :] = 0
    mask[:, :, [0, -1]] = 0
    return (
        _write_4d(tmp_path / "real.nii.gz", real, affine),
        _write_4d(tmp_path / "pred.nii.gz", predicted, affine),
        _write_mask(tmp_path / "mask.nii.gz", mask, affine),
    )


def test_load_comparison_is_strict_and_canonical(tmp_path):
    # A left-to-right flipped native grid must be displayed as canonical RAS+.
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 8.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 2.0, -6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    real, predicted, mask = _paired_paths(tmp_path, affine=affine)
    comparison = visual.load_comparison(
        real, predicted, mask_path=mask, tr_seconds=2.0
    )

    assert nib.aff2axcodes(comparison.affine) == ("R", "A", "S")
    assert comparison.real.shape == comparison.predicted.shape == (5, 6, 7, 6)
    assert comparison.mask.shape == (5, 6, 7)
    assert comparison.tr_seconds == pytest.approx(2.0)
    assert not comparison.mask_resampled
    assert "paired input space" in comparison.space_description

    with pytest.raises(ValueError, match="TR validation override"):
        visual.load_comparison(
            real, predicted, mask_path=mask, tr_seconds=3.0
        )

    with pytest.raises(ValueError, match="independent structural brain mask"):
        visual.load_comparison(real, predicted)


def test_real_and_predicted_grid_time_and_affine_must_match(tmp_path):
    real, predicted, mask = _paired_paths(tmp_path)
    reference = nib.load(predicted)
    values = reference.get_fdata(dtype=np.float32)

    different_frames = tmp_path / "different_frames.nii.gz"
    _write_4d(
        different_frames, values[..., :-1], reference.affine, tr=2.0
    )
    with pytest.raises(ValueError, match="shapes differ"):
        visual.load_comparison(real, different_frames, mask_path=mask)

    shifted = tmp_path / "shifted.nii.gz"
    shifted_affine = reference.affine.copy()
    shifted_affine[0, 3] += 0.25
    _write_4d(shifted, values, shifted_affine, tr=2.0)
    with pytest.raises(ValueError, match="affines differ"):
        visual.load_comparison(real, shifted, mask_path=mask)

    different_tr = tmp_path / "different_tr.nii.gz"
    _write_4d(different_tr, values, reference.affine, tr=2.5)
    with pytest.raises(ValueError, match="TR differ"):
        visual.load_comparison(real, different_tr, mask_path=mask)


def test_mask_resamples_only_on_matching_world_support(tmp_path):
    shape = (5, 5, 5, 6)
    reference_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    x, y, z = np.indices(shape[:3], dtype=np.float32)
    base = x + y + z + 1.0
    data = np.stack([base + frame for frame in range(shape[-1])], axis=-1)
    real = _write_4d(tmp_path / "real.nii.gz", data, reference_affine)
    predicted = _write_4d(tmp_path / "pred.nii.gz", data, reference_affine)

    # Voxel-edge corners are -1 and 9 mm on both grids, so nearest-neighbour
    # resampling changes resolution without inventing registration.
    coarse_affine = np.diag([10.0 / 3.0, 10.0 / 3.0, 10.0 / 3.0, 1.0])
    coarse_affine[:3, 3] = 2.0 / 3.0
    coarse_mask = _write_mask(
        tmp_path / "coarse_mask.nii.gz",
        np.ones((3, 3, 3), dtype=np.uint8),
        coarse_affine,
    )
    comparison = visual.load_comparison(
        real, predicted, mask_path=coarse_mask
    )
    assert comparison.mask_resampled
    assert comparison.mask.shape == shape[:3]
    assert comparison.mask.all()

    shifted_affine = coarse_affine.copy()
    shifted_affine[0, 3] = 1.0
    shifted_mask = _write_mask(
        tmp_path / "shifted_mask.nii.gz",
        np.ones((3, 3, 3), dtype=np.uint8),
        shifted_affine,
    )
    with pytest.raises(ValueError, match="same oriented world geometry"):
        visual.load_comparison(real, predicted, mask_path=shifted_mask)


def test_plane_extents_preserve_physical_aspect():
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    shape = (5, 6, 7, 2)
    assert visual._plane_extent_mm(shape, affine, "axial") == (0.0, 10.0, 0.0, 18.0)
    assert visual._plane_extent_mm(shape, affine, "sagittal") == (0.0, 18.0, 0.0, 28.0)
    assert visual._plane_extent_mm(shape, affine, "coronal") == (0.0, 10.0, 0.0, 28.0)


def test_montage_and_temporal_diagnostics_are_rendered(tmp_path):
    real, predicted, mask = _paired_paths(tmp_path)
    mask_values = nib.load(mask).get_fdata() > 0
    labels = _write_mask(
        tmp_path / "labels.nii.gz",
        _roi_labels(mask_values),
        nib.load(mask).affine,
    )
    comparison = visual.load_comparison(
        real, predicted, mask_path=mask, roi_labels_path=labels
    )
    limits = visual.compute_scale_limits(comparison)
    diagnostics = visual.compute_temporal_diagnostics(comparison)

    assert limits[0] > 0
    assert limits[1] > 0
    assert diagnostics["time_seconds"].tolist() == [0, 2, 4, 6, 8, 10]
    assert diagnostics["spatial_correlation"].shape == (6,)
    assert diagnostics["rmse"].shape == (6,)
    assert np.all(diagnostics["rmse"] >= 0)

    montage = visual.plot_frame_montage(
        comparison, tmp_path / "montage.png", scale_limits=limits
    )
    temporal = visual.plot_temporal_diagnostics(
        comparison,
        tmp_path / "temporal.png",
        diagnostics=diagnostics,
    )
    detail = visual.plot_spatial_detail_comparison(
        comparison, tmp_path / "detail.png"
    )
    outside = visual.plot_outside_mask_leakage(
        comparison, tmp_path / "outside.png"
    )
    structured_values = visual.compute_structured_dynamics(comparison)
    assert np.asarray(structured_values["real_fc"]).shape == (6, 6)
    assert np.asarray(structured_values["predicted_fc"]).shape == (6, 6)
    assert np.asarray(structured_values["real_normalized_power"]).shape[0] == 6
    structured = visual.plot_structured_dynamics(
        comparison,
        tmp_path / "structured.png",
        diagnostics=structured_values,
    )
    for path in (montage, temporal, detail, outside, structured):
        assert path.is_file() and path.stat().st_size > 0
        with Image.open(path) as rendered:
            assert rendered.width > 500
            assert rendered.height > 500


def test_detail_panel_preserves_raw_arrays_and_uses_physical_aspect(tmp_path):
    real, predicted, mask = _paired_paths(
        tmp_path, affine=np.diag([2.0, 3.0, 4.0, 1.0])
    )
    comparison = visual.load_comparison(real, predicted, mask_path=mask)
    real_before = comparison.real.copy()
    predicted_before = comparison.predicted.copy()
    output = visual.plot_spatial_detail_comparison(
        comparison, tmp_path / "spatial_detail.png", sigma_voxels=1.0
    )
    assert output.is_file() and output.stat().st_size > 0
    np.testing.assert_array_equal(comparison.real, real_before)
    np.testing.assert_array_equal(comparison.predicted, predicted_before)


def test_detail_panel_calls_shared_boundary_safe_high_pass(tmp_path, monkeypatch):
    real, predicted, mask = _paired_paths(tmp_path)
    comparison = visual.load_comparison(real, predicted, mask_path=mask)
    original = visual.mask_normalized_high_pass
    original_plane = visual._plane
    calls = []
    displayed_supports = []

    def recording_high_pass(volume, support, sigma):
        result = original(volume, support, sigma)
        calls.append((np.asarray(volume).copy(), result[1].copy()))
        return result

    def recording_plane(volume, support, plane_name, indices):
        displayed_supports.append(np.asarray(support).copy())
        return original_plane(volume, support, plane_name, indices)

    monkeypatch.setattr(visual, "mask_normalized_high_pass", recording_high_pass)
    monkeypatch.setattr(visual, "_plane", recording_plane)
    visual.plot_spatial_detail_comparison(comparison, tmp_path / "detail_shared.png")

    assert len(calls) == 4
    for call in calls[1:]:
        np.testing.assert_array_equal(calls[0][1], call[1])
    assert np.all(calls[0][1] <= comparison.mask)
    assert len(displayed_supports) == 18
    for displayed_support in displayed_supports:
        np.testing.assert_array_equal(displayed_support, calls[0][1])


def test_whole_fov_leakage_diagnostic_detects_distant_corner(tmp_path):
    real, predicted, mask = _paired_paths(tmp_path)
    predicted_image = nib.load(predicted)
    predicted_values = predicted_image.get_fdata(dtype=np.float32)
    predicted_values[0, 0, 0, :] = 100.0
    dirty = _write_4d(
        tmp_path / "dirty_pred.nii.gz",
        predicted_values,
        predicted_image.affine,
        tr=2.0,
    )
    comparison = visual.load_comparison(real, dirty, mask_path=mask)

    diagnostics = visual.compute_outside_mask_diagnostics(comparison)

    assert diagnostics["predicted_outside_mask_max_abs"] == pytest.approx(100.0)
    assert diagnostics["predicted_outside_mask_leakage_ratio"] > 1.0


def test_generated_visualization_manifest_hash_binds_sources_and_outputs(
    tmp_path, monkeypatch
):
    real, predicted, mask = _paired_paths(tmp_path)
    mask_image = nib.load(mask)
    labels = _write_mask(
        tmp_path / "manifest_labels.nii.gz",
        _roi_labels(mask_image.get_fdata() > 0),
        mask_image.affine,
    )

    safe_writes = []
    original_write = visual._write_output_new

    def recording_write(path, payload, output):
        safe_writes.append(Path(path).name)
        return original_write(path, payload, output)

    def fake_animation(_comparison, output_base, **kwargs):
        path = Path(output_base).with_suffix(".mp4")
        visual._write_output_new(
            path,
            b"fixed animation artifact",
            kwargs["_output_directory"],
        )
        return path

    monkeypatch.setattr(visual, "_write_output_new", recording_write)
    monkeypatch.setattr(visual, "save_comparison_animation", fake_animation)
    outputs = visual.generate_comparison_outputs(
        real,
        predicted,
        tmp_path / "visualizations",
        mask_path=mask,
        roi_labels_path=labels,
        prefix="paired",
    )

    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["schema"] == "connect4-4d-visualization-manifest-v1"
    stored_record_hash = manifest.pop("record_sha256")
    assert stored_record_hash == hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest["display_contract"]["image_interpolation"] == "nearest"
    assert manifest["display_contract"]["structured_dynamics"]["roi_count"] == 6
    for source in manifest["implementation_sources"].values():
        source_path = Path(source["path"])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
    for name, output in manifest["outputs"].items():
        assert hashlib.sha256(Path(output["path"]).read_bytes()).hexdigest() == output[
            "sha256"
        ]
        assert name in outputs
    assert set(safe_writes) == {
        name for name in VISUALIZATION_OUTPUT_NAMES if not name.endswith(".gif")
    }


@pytest.mark.parametrize("output_name", VISUALIZATION_OUTPUT_NAMES)
@pytest.mark.parametrize(
    "threat", ("symlink", "hardlink", "file-replacement", "root-replacement")
)
def test_every_visualization_output_is_descriptor_bound_and_never_replaces_links(
    tmp_path: Path,
    output_name: str,
    threat: str,
) -> None:
    output_path = tmp_path / "held-visualizations"
    output = visual._open_output_directory(output_path)
    sentinel_payload = f"sentinel:{output_name}:{threat}".encode("ascii")
    try:
        if threat == "root-replacement":
            detached = tmp_path / "detached-visualizations"
            output_path.rename(detached)
            output_path.mkdir()
            sentinel = output_path / output_name
            sentinel.write_bytes(sentinel_payload)
            with pytest.raises(RuntimeError, match="directory was replaced"):
                visual._write_output_new(
                    output_path / output_name,
                    b"must-not-write",
                    output,
                )
        elif threat == "file-replacement":
            attacked = output_path / output_name
            visual._write_output_new(attacked, b"original output", output)
            attacked.unlink()
            attacked.write_bytes(sentinel_payload)
            sentinel = attacked
            with pytest.raises(RuntimeError, match="identity changed"):
                visual._output_snapshot(attacked, output)
        else:
            sentinel = tmp_path / f"sentinel-{threat}-{output_name}"
            sentinel.write_bytes(sentinel_payload)
            attacked = output_path / output_name
            if threat == "symlink":
                attacked.symlink_to(sentinel)
            else:
                attacked.hardlink_to(sentinel)
            with pytest.raises(FileExistsError):
                visual._write_output_new(attacked, b"must-not-write", output)
        assert sentinel.read_bytes() == sentinel_payload
    finally:
        os.close(output.descriptor)


def test_animation_uses_mp4_and_falls_back_to_gif(tmp_path, monkeypatch):
    real, predicted, mask = _paired_paths(tmp_path)
    comparison = visual.load_comparison(real, predicted, mask_path=mask)
    saved = []

    def fake_render(animation, *, writer, dpi):
        del dpi
        animation._draw_was_started = True
        saved.append(type(writer))
        if isinstance(writer, visual._BytesFFMpegWriter):
            raise RuntimeError("test ffmpeg failure")
        return b"GIF89a"

    monkeypatch.setattr(visual.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(visual, "_render_animation_payload", fake_render)
    with pytest.warns(RuntimeWarning, match="falling back to GIF"):
        output = visual.save_comparison_animation(
            comparison, tmp_path / "comparison", fps=4
        )

    assert output == tmp_path / "comparison.gif"
    assert output.is_file()
    assert saved == [visual._BytesFFMpegWriter, visual._BytesPillowWriter]


def test_animation_gif_encoder_uses_memory_before_descriptor_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, predicted, mask = _paired_paths(tmp_path, frames=2)
    comparison = visual.load_comparison(real, predicted, mask_path=mask)
    monkeypatch.setattr(visual.shutil, "which", lambda _name: None)

    output = visual.save_comparison_animation(
        comparison,
        tmp_path / "descriptor_animation",
        fps=2,
    )

    assert output == tmp_path / "descriptor_animation.gif"
    assert output.read_bytes().startswith(b"GIF")
    assert not (Path.cwd() / "descriptor-only-animation").exists()


@pytest.mark.skipif(visual.shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_animation_mp4_encoder_uses_anonymous_descriptor_before_publication(
    tmp_path: Path,
) -> None:
    real, predicted, mask = _paired_paths(tmp_path, frames=2)
    comparison = visual.load_comparison(real, predicted, mask_path=mask)

    output = visual.save_comparison_animation(
        comparison,
        tmp_path / "descriptor_animation_mp4",
        fps=2,
    )

    assert output == tmp_path / "descriptor_animation_mp4.mp4"
    assert b"ftyp" in output.read_bytes()[:64]
    assert not (Path.cwd() / "descriptor-only-animation").exists()
