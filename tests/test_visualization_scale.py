from __future__ import annotations

import hashlib
import json
import matplotlib.axes
import numpy as np

import pytest

from eval import visualize as visualization
from eval.visualize import plot_real_vs_synthetic, plot_real_vs_synthetic_4d


def test_real_and_synthetic_use_one_shared_signal_window(tmp_path, monkeypatch):
    real = np.zeros((4, 7, 7, 7), dtype=np.float32)
    synthetic = np.zeros_like(real)
    real[:, 1:6, 1:6, 1:6] = np.linspace(0.2, 1.0, 4)[:, None, None, None]
    synthetic[:, 1:6, 1:6, 1:6] = 0.1 * real[:, 1:6, 1:6, 1:6]

    calls: list[tuple[float | None, float | None]] = []
    original_imshow = matplotlib.axes.Axes.imshow

    def recording_imshow(self, *args, **kwargs):
        calls.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", recording_imshow)
    plot_real_vs_synthetic(
        real,
        synthetic,
        tmp_path / "comparison.png",
        allow_temporal_mean_legacy=True,
    )

    assert len(calls) == 9
    assert len(set(calls[:6])) == 1
    assert calls[6][0] == calls[7][0] == calls[8][0] == 0.0
    assert calls[6][1] == calls[7][1] == calls[8][1]
    assert calls[0][1] > float(synthetic.max())


def test_temporal_mean_legacy_plot_is_fail_closed(tmp_path):
    values = np.ones((4, 5, 5, 5), dtype=np.float32)
    with pytest.raises(RuntimeError, match="legacy temporal-mean"):
        plot_real_vs_synthetic(values, values, tmp_path / "legacy.png")


def test_4d_figure_keeps_frames_and_shares_corresponding_scales(tmp_path, monkeypatch):
    coordinates = np.indices((7, 7, 7), dtype=np.float32)
    texture = ((coordinates[0] + 2 * coordinates[1] + coordinates[2]) % 3) - 1.0
    time = np.arange(8, dtype=np.float32)
    real = np.zeros((8, 7, 7, 7), dtype=np.float32)
    synthetic = np.zeros_like(real)
    real[:, 1:6, 1:6, 1:6] = (
        np.sin(2 * np.pi * time / 8)[:, None, None, None]
        + 0.25
        * texture[1:6, 1:6, 1:6][None, ...]
        * np.cos(2 * np.pi * time / 4)[:, None, None, None]
    )
    synthetic[:, 1:6, 1:6, 1:6] = 0.35 * real[:, 1:6, 1:6, 1:6]

    calls: list[tuple[np.ndarray, float | None, float | None]] = []
    interpolations: list[str | None] = []
    original_imshow = matplotlib.axes.Axes.imshow

    def recording_imshow(self, image, *args, **kwargs):
        calls.append((np.asarray(image), kwargs.get("vmin"), kwargs.get("vmax")))
        interpolations.append(kwargs.get("interpolation"))
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", recording_imshow)
    output = tmp_path / "comparison_4d.png"
    plot_real_vs_synthetic_4d(
        real,
        synthetic,
        output,
        brain_mask=np.pad(np.ones((5, 5, 5)), 1),
        slice_idx=(3, 3, 3),
        frame_indices=(0, 2, 4, 6),
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000
    sidecar = output.with_suffix(output.suffix + ".display.json")
    record = json.loads(sidecar.read_text())
    assert record["schema"] == "connect4-in-memory-4d-display-contract-v1"
    assert record["display"]["phase_aligned_residuals_report_only"] is True
    assert record["display"]["frame_aligned_residuals_report_only"] is True
    assert record["display"]["residual_semantic"] == "predicted_minus_real_signed"
    assert record["display"]["image_interpolation"] == "nearest"
    assert record["display"]["high_pass"]["scale_support"] == "eroded interior"
    assert record["display"]["temporal_mean_texture"] == {
        "slice_indices_xyz": [3, 3, 3],
        "slice_selection": "largest eroded structural-mask area",
        "shared_real_prediction_scale": True,
        "display_only_not_model_selection": True,
    }
    recorded_hash = record.pop("record_sha256")
    assert (
        recorded_hash
        == hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    assert len(calls) == 42
    assert set(interpolations) == {"nearest"}
    # Real and synthetic dynamic panels share one symmetric scale.
    assert {(low, high) for _, low, high in calls[0:8]} == {(calls[0][1], calls[0][2])}
    assert calls[0][1] == -calls[0][2]
    # Signed predicted-minus-real residuals use symmetric limits and retain sign.
    assert {(low, high) for _, low, high in calls[8:12]} == {
        (calls[8][1], calls[8][2])
    }
    assert calls[8][1] == -calls[8][2]
    np.testing.assert_allclose(
        calls[8][0], calls[4][0] - calls[0][0], equal_nan=True
    )
    # Individual displayed frames are not replaced with one temporal mean.
    assert not np.array_equal(calls[0][0], calls[1][0])
    # Real and synthetic high-pass panels likewise share a symmetric scale.
    assert {(low, high) for _, low, high in calls[12:20]} == {
        (calls[12][1], calls[12][2])
    }
    assert calls[12][1] == -calls[12][2]
    assert calls[20][1] == -calls[20][2]
    # Temporal-mean spatial texture is explicit and real/synthetic panels share
    # one symmetric scale instead of disappearing through temporal demeaning.
    assert {(low, high) for _, low, high in calls[24:30]} == {
        (calls[24][1], calls[24][2])
    }
    assert calls[24][1] == -calls[24][2]
    assert calls[30][1] == -calls[30][2]
    # Temporal-STD real and synthetic tri-planes share a zero-anchored scale.
    assert {(low, high) for _, low, high in calls[33:39]} == {(0.0, calls[33][2])}
    assert calls[39][1] == -calls[39][2]


def test_temporal_mean_texture_exposes_static_smoothing_hidden_by_dynamics(
    tmp_path, monkeypatch
):
    coordinates = np.indices((7, 7, 7), dtype=np.float32)
    static_texture = ((coordinates[0] + 2 * coordinates[1] + coordinates[2]) % 3) - 1.0
    temporal = np.asarray([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)
    mask = np.pad(np.ones((5, 5, 5), dtype=np.uint8), 1)
    real = np.zeros((4, 7, 7, 7), dtype=np.float32)
    synthetic = np.zeros_like(real)
    real[:, mask > 0] = temporal[:, None] + static_texture[mask > 0][None, :]
    synthetic[:, mask > 0] = temporal[:, None]

    calls: list[np.ndarray] = []
    original_imshow = matplotlib.axes.Axes.imshow

    def recording_imshow(self, image, *args, **kwargs):
        calls.append(np.asarray(image))
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", recording_imshow)
    plot_real_vs_synthetic_4d(
        real,
        synthetic,
        tmp_path / "static-texture.png",
        brain_mask=mask,
        frame_indices=(0, 1, 2, 3),
    )

    # The demeaned dynamics and their per-frame high-pass detail match exactly.
    for real_panel, synthetic_panel in zip(calls[0:4], calls[4:8]):
        np.testing.assert_allclose(real_panel, synthetic_panel, equal_nan=True)
    for real_panel, synthetic_panel in zip(calls[12:16], calls[16:20]):
        np.testing.assert_allclose(real_panel, synthetic_panel, equal_nan=True)
    # The new temporal-mean texture rows still reveal the missing static detail.
    assert any(
        not np.allclose(real_panel, synthetic_panel, equal_nan=True)
        for real_panel, synthetic_panel in zip(calls[24:27], calls[27:30])
    )
    assert any(float(np.nanmax(panel)) > 0.0 for panel in calls[30:33])


def test_4d_figure_rejects_shape_or_temporal_contract_violations(tmp_path):
    valid = np.ones((4, 5, 5, 5), dtype=np.float32)
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="matching shapes"):
        plot_real_vs_synthetic_4d(
            valid, valid[:, :-1], tmp_path / "bad.png", brain_mask=mask
        )
    with pytest.raises(ValueError, match="at least four"):
        plot_real_vs_synthetic_4d(
            valid[:3], valid[:3], tmp_path / "short.png", brain_mask=mask
        )
    with pytest.raises(ValueError, match="four distinct"):
        plot_real_vs_synthetic_4d(
            valid,
            valid,
            tmp_path / "frames.png",
            brain_mask=mask,
            frame_indices=(0, 0, 1, 2),
        )

    nonfinite = valid.copy()
    nonfinite[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        plot_real_vs_synthetic_4d(
            nonfinite, valid, tmp_path / "nonfinite.png", brain_mask=mask
        )

    with pytest.raises(ValueError, match="singleton batch"):
        plot_real_vs_synthetic_4d(
            np.stack([valid, valid]),
            np.stack([valid, valid]),
            tmp_path / "batch.png",
            brain_mask=mask,
        )


def test_high_pass_scale_uses_only_the_displayed_eroded_interior(tmp_path, monkeypatch):
    values = np.stack(
        [np.full((7, 7, 7), frame, dtype=np.float32) for frame in range(4)]
    )
    mask = np.ones((7, 7, 7), dtype=np.uint8)

    def high_pass_with_extreme_excluded_rim(volume, support, sigma):
        del volume, sigma
        interior = np.asarray(support, dtype=bool).copy()
        interior[[0, -1], :, :] = False
        interior[:, [0, -1], :] = False
        interior[:, :, [0, -1]] = False
        detail = np.ones(interior.shape, dtype=np.float64)
        detail[np.asarray(support, dtype=bool) & ~interior] = 1e9
        return detail, interior

    monkeypatch.setattr(
        visualization, "mask_normalized_high_pass", high_pass_with_extreme_excluded_rim
    )
    output = tmp_path / "rim.png"
    plot_real_vs_synthetic_4d(values, values, output, brain_mask=mask)
    record = json.loads(output.with_suffix(".png.display.json").read_text())
    assert record["display"]["scale_limits"]["high_pass_abs_limit"] == pytest.approx(
        1.0
    )
    assert record["display"]["scale_limits"][
        "temporal_mean_high_pass_abs_limit"
    ] == pytest.approx(1.0)
