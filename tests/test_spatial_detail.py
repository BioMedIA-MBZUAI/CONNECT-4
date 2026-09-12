import numpy as np
import pytest

from utils.spatial_detail import mask_normalized_high_pass


def _support() -> np.ndarray:
    mask = np.zeros((13, 15, 11), dtype=bool)
    mask[2:11, 3:13, 2:9] = True
    return mask


def test_constant_in_mask_has_no_artificial_boundary_detail():
    mask = _support()
    volume = np.where(mask, 0.73, 0.0)

    detail, interior = mask_normalized_high_pass(volume, mask, 1.0)

    np.testing.assert_allclose(detail, 0.0, rtol=0.0, atol=1e-12)
    assert interior.any()
    assert np.all(interior <= mask)


def test_exterior_values_cannot_contaminate_masked_detail():
    mask = _support()
    coordinates = np.indices(mask.shape)
    clean = np.where(
        mask,
        0.4
        + 0.03 * np.sin(1.2 * coordinates[0])
        + 0.02 * np.cos(1.7 * coordinates[1]),
        0.0,
    )
    dirty = clean.copy()
    dirty[~mask] = np.where((coordinates[0][~mask] % 2) == 0, 1e6, -1e6)

    clean_detail, clean_interior = mask_normalized_high_pass(clean, mask, 1.0)
    dirty_detail, dirty_interior = mask_normalized_high_pass(dirty, mask, 1.0)

    np.testing.assert_array_equal(clean_interior, dirty_interior)
    np.testing.assert_allclose(
        clean_detail[mask], dirty_detail[mask], rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(dirty_detail[~mask], 0.0)


@pytest.mark.parametrize(
    ("volume", "mask", "sigma", "erosion", "message"),
    [
        (np.zeros((2, 2)), np.ones((2, 2), bool), 1.0, 1, "must be 3D"),
        (
            np.zeros((2, 2, 2)),
            np.ones((2, 2, 3), bool),
            1.0,
            1,
            "shape",
        ),
        (
            np.zeros((2, 2, 2)),
            np.zeros((2, 2, 2), bool),
            1.0,
            1,
            "empty",
        ),
        (
            np.full((2, 2, 2), np.nan),
            np.ones((2, 2, 2), bool),
            1.0,
            1,
            "NaN",
        ),
        (
            np.zeros((2, 2, 2)),
            np.ones((2, 2, 2), bool),
            0.0,
            1,
            "positive",
        ),
        (
            np.zeros((2, 2, 2)),
            np.ones((2, 2, 2), bool),
            1.0,
            -1,
            "nonnegative integer",
        ),
    ],
)
def test_high_pass_rejects_invalid_inputs(volume, mask, sigma, erosion, message):
    with pytest.raises(ValueError, match=message):
        mask_normalized_high_pass(
            volume, mask, sigma, erosion_iterations=erosion
        )
