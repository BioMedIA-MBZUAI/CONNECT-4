import math
from pathlib import Path

import pytest

from normative.atrophy import ASEG_TO_NORM_LABEL, AtrophyDescriber
from normative.subcortical_norms import SubcorticalNorms


WORKBOOK = Path(__file__).resolve().parents[1] / "normative" / "mmc2.xlsm"

EXPECTED_REGIONS = {
    "Accumbens L",
    "Accumbens R",
    "Amygdala L",
    "Amygdala R",
    "Brainstem",
    "Caudate L",
    "Caudate R",
    "Hippocampus L",
    "Hippocampus R",
    "Pallidum L",
    "Pallidum R",
    "Putamen L",
    "Putamen R",
    "Thalamus L",
    "Thalamus R",
    "Ventral DC L",
    "Ventral DC R",
    "Ventricles sum",
    "Lateral L",
    "Lateral R",
    "Inferior lateral L",
    "Inferior lateral R",
    "3rd",
    "4th",
    "Corpus callosum",
    "Subcortical GM",
}


@pytest.fixture(scope="module")
def norms():
    return SubcorticalNorms(str(WORKBOOK))


def test_parser_loads_every_displayed_workbook_model(norms):
    assert set(norms.regions()) == EXPECTED_REGIONS
    assert len(norms.region_models) == 26
    assert norms.mean_age == pytest.approx(47.5634617)
    assert norms.mean_tiv == pytest.approx(1521907.28)

    for model in norms.region_models.values():
        dimension = len(model["beta_names"])
        assert dimension == len(model["beta_vals"])
        assert dimension == len(model["M"])
        assert all(len(row) == dimension for row in model["M"])
        assert model["n"] > dimension
        assert model["mse"] > 0
        assert model["t"] > 0
        assert all(math.isfinite(value) for value in model["beta_vals"])


def test_predictions_are_finite_and_ordered_for_all_models(norms):
    predictions = norms.predict_all(
        age=60, sex=1, field_strength=0, manufacturer=3, icv=1_500_000
    )
    assert set(predictions) == EXPECTED_REGIONS
    for prediction in predictions.values():
        assert all(math.isfinite(value) for value in prediction.values())
        assert prediction["lower"] < prediction["pred"] < prediction["upper"]
        assert prediction["se"] > 0


def test_direct_and_log10_models_match_published_workbook_coefficients(norms):
    covariates = dict(age=60, sex=1, field_strength=0, manufacturer=3, icv=1_500_000)

    # Fixed regression results from the named coefficient/covariance ranges in
    # the distributed workbook, independently checked by recalculating that
    # workbook in LibreOffice. One direct and one log10 model cover both paths.
    accumbens = norms.predict_region("Accumbens L", **covariates)
    lateral = norms.predict_region("Lateral L", **covariates)
    assert accumbens["pred"] == pytest.approx(428.369406405649, rel=1e-12)
    assert accumbens["lower"] == pytest.approx(174.783449473830, rel=1e-12)
    assert accumbens["upper"] == pytest.approx(681.955363337467, rel=1e-12)
    assert lateral["pred"] == pytest.approx(9975.07777833955, rel=1e-12)
    assert lateral["lower"] == pytest.approx(4205.78531707564, rel=1e-12)
    assert lateral["upper"] == pytest.approx(23658.4060246588, rel=1e-12)


def test_all_individual_workbook_regions_have_correct_aseg_mapping(norms):
    aggregate_only = {"Ventricles sum", "Corpus callosum", "Subcortical GM"}
    assert len(ASEG_TO_NORM_LABEL) == 23
    assert set(ASEG_TO_NORM_LABEL.values()) == EXPECTED_REGIONS - aggregate_only
    assert set(ASEG_TO_NORM_LABEL.values()) <= set(norms.regions())
    assert ASEG_TO_NORM_LABEL[4] == "Lateral L"
    assert ASEG_TO_NORM_LABEL[43] == "Lateral R"


def test_log_model_z_score_is_computed_on_log10_scale():
    describer = AtrophyDescriber(str(WORKBOOK))
    result = describer.atrophy_for_roi(
        "Lateral L",
        measured_volume=8_000,
        age=60,
        sex=1,
        field_strength=0,
        manufacturer=3,
        icv=1_500_000,
    )
    expected_z = (
        math.log10(result["measured_volume"]) - math.log10(result["normal_volume"])
    ) / describer.norms.predict_region(
        "Lateral L", age=60, sex=1, field_strength=0, manufacturer=3, icv=1_500_000
    )["se"]
    assert result["z"] == pytest.approx(expected_z)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sex", 2, "0=female, 1=male"),
        ("field_strength", 3, "0=3T, 1=1.5T"),
        ("manufacturer", 4, "1=GE, 2=Philips, 3=Siemens"),
    ],
)
def test_workbook_codings_are_validated(norms, field, value, message):
    covariates = dict(age=60, sex=1, field_strength=0, manufacturer=3, icv=1_500_000)
    covariates[field] = value
    with pytest.raises(ValueError, match=message):
        norms.predict_region("Caudate L", **covariates)
