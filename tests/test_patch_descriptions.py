from data.patch_descriptions import (
    KNOWN_FUNCTIONAL_CONNECTIVITY,
    biological_roi_description,
    build_normative_index,
    build_patch_description,
)


EXPECTED_ROIS = {
    "left_cerebral_white_matter", "left_cerebral_cortex",
    "left_lateral_ventricle", "left_inferior_lateral_ventricle",
    "left_cerebellum_white_matter", "left_cerebellum_cortex",
    "left_thalamus", "left_caudate", "left_putamen", "left_pallidum",
    "third_ventricle", "fourth_ventricle", "brain_stem",
    "left_hippocampus", "left_amygdala", "csf", "left_accumbens_area",
    "left_ventral_dc", "right_cerebral_white_matter", "right_cerebral_cortex",
    "right_lateral_ventricle", "right_inferior_lateral_ventricle",
    "right_cerebellum_white_matter", "right_cerebellum_cortex",
    "right_thalamus", "right_caudate", "right_putamen", "right_pallidum",
    "right_hippocampus", "right_amygdala", "right_accumbens_area",
    "right_ventral_dc",
}


def test_functional_connectivity_vocabulary_covers_every_connect4_roi():
    assert set(KNOWN_FUNCTIONAL_CONNECTIVITY) == EXPECTED_ROIS


def test_roi_sentence_always_has_connectivity_and_normative_sections():
    description = biological_roi_description(
        "left_hippocampus",
        0.25,
        "The measured volume is below the age-, sex-, scanner-, and field-matched range.",
    )
    assert "Known functional connectivity:" in description
    assert "Normative volume context:" in description
    assert "25.0% patch coverage" in description


def test_potvin_labels_match_synthseg_slugs_in_patch_text():
    normative = build_normative_index({
        ("B001", "Hippocampus L"): "Normative volume 3296 mm^3; measured volume 3120 mm^3."
    })
    description = build_patch_description(
        patch_idx=7,
        center_mm=(1.0, 2.0, 3.0),
        distribution={0: 0.5, 17: 0.25},
        id_to_slug={17: "left_hippocampus"},
        normative_index=normative,
        patient_id="B001",
    )
    assert "contains 1 ROIs" in description
    assert "Known functional connectivity:" in description
    assert "Normative volume 3296" in description
    assert "background" not in description


def test_all_individual_potvin_labels_normalize_to_connect4_roi_slugs():
    from data.dataset import Connect4Dataset
    from normative.atrophy import ASEG_TO_NORM_LABEL

    descriptions = {
        ("subject", label): f"norm for {label}"
        for label in ASEG_TO_NORM_LABEL.values()
    }
    index = build_normative_index(descriptions)
    expected = {
        Connect4Dataset.ID_TO_SLUG[label_id]
        for label_id in ASEG_TO_NORM_LABEL
    }
    assert {structure for patient, structure in index if patient == "subject"} == expected


def test_missing_norm_is_explicit_instead_of_silently_omitted():
    description = biological_roi_description("right_thalamus", 0.1)
    assert "no valid subject-specific normative estimate" in description
