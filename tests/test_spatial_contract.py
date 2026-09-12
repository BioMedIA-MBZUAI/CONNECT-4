import nibabel as nib
import numpy as np
import pytest

from architecture_contract import (
    PATCH_CENTER_CONVENTION,
    geometric_patch_centers_voxel,
)
from data.patch_descriptions import build_patch_description
from data.spatial_contract import (
    PATCH_COORDINATE_CONTRACT,
    align_corners_false_source_index,
    geometric_patch_center_voxel,
    nifti_geometry_fingerprint,
    patch_grid_xyz,
)


def test_align_corners_false_mapping_includes_half_voxel_offset():
    mapped = align_corners_false_source_index(
        (0, 0, 0), (182, 218, 182), (128, 128, 128)
    )
    np.testing.assert_allclose(
        mapped, (0.2109375, 0.3515625, 0.2109375), atol=1e-12
    )
    centre = align_corners_false_source_index(
        (63.5, 63.5, 63.5), (182, 218, 182), (128, 128, 128)
    )
    np.testing.assert_allclose(centre, (90.5, 108.5, 90.5), atol=1e-12)


def test_patch_index_is_c_order_xyz_and_geometric_center_is_half_voxel():
    assert patch_grid_xyz(7, (128, 128, 128), (16, 16, 16)) == (0, 0, 7)
    assert patch_grid_xyz(8, (128, 128, 128), (16, 16, 16)) == (0, 1, 0)
    assert patch_grid_xyz(64, (128, 128, 128), (16, 16, 16)) == (1, 0, 0)
    np.testing.assert_allclose(
        geometric_patch_center_voxel((0, 0, 0), (2, 2, 2)),
        (0.5, 0.5, 0.5),
    )


def test_even_patch_centres_are_geometric_in_voxel_and_ras_coordinates():
    centers = np.asarray(
        geometric_patch_centers_voxel((4, 8, 12), (2, 4, 6))
    )
    expected = np.asarray(
        [
            (0.5, 1.5, 2.5),
            (0.5, 1.5, 8.5),
            (0.5, 5.5, 2.5),
            (0.5, 5.5, 8.5),
            (2.5, 1.5, 2.5),
            (2.5, 1.5, 8.5),
            (2.5, 5.5, 2.5),
            (2.5, 5.5, 8.5),
        ]
    )
    np.testing.assert_array_equal(centers, expected)
    np.testing.assert_array_equal(
        geometric_patch_center_voxel((1, 1, 1), (2, 4, 6)),
        expected[-1],
    )

    ras_affine = np.asarray(
        [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 3.0, 0.0, -20.0],
            [0.0, 0.0, 4.0, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ras_centers = nib.affines.apply_affine(ras_affine, centers)
    np.testing.assert_array_equal(ras_centers[0], (-9.0, -15.5, -20.0))
    np.testing.assert_array_equal(ras_centers[-1], (-5.0, -3.5, 4.0))
    assert PATCH_COORDINATE_CONTRACT["patch_center_convention"] == (
        PATCH_CENTER_CONVENTION
    )


def test_patch_text_names_ras_axes_semantically():
    description = build_patch_description(
        patch_idx=1,
        center_mm=(-12.0, 3.0, 24.0),
        distribution={},
        id_to_slug={},
        normative_index={},
        patient_id="S1",
    )
    assert "X=-12.0 mm left-to-right" in description
    assert "Y=3.0 mm posterior-to-anterior" in description
    assert "Z=24.0 mm inferior-to-superior" in description
    assert PATCH_COORDINATE_CONTRACT["array_axis_order"] == ["X", "Y", "Z"]


def test_geometry_fingerprint_uses_voxel_edges_and_canonical_ras(tmp_path):
    data = np.zeros((2, 3, 4), dtype=np.float32)
    affine = np.asarray(
        [[-2.0, 0.0, 0.0, 3.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    path = tmp_path / "structural.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    fingerprint = nifti_geometry_fingerprint(path)
    assert fingerprint["axis_codes"] == ["R", "A", "S"]
    assert fingerprint["shape_xyz"] == [2, 3, 4]
    assert fingerprint["voxel_sizes_mm_xyz"] == pytest.approx([2.0, 3.0, 4.0])
    assert fingerprint["voxel_edge_bounds_mm"]["minimum"] == pytest.approx([0.0, -1.5, -2.0])
    assert fingerprint["voxel_edge_bounds_mm"]["maximum"] == pytest.approx([4.0, 7.5, 14.0])
