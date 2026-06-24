"""
Structural-atrophy descriptions (Stream A of CONNECT-4, Figure 1A).

Pipeline:
    Age / Gender / Scanner / TIV / Field-strength  ->  External Normative
    Regional Volume (Potvin et al. [19], `subcortical_norms.SubcorticalNorms`)
    ->  Structural Atrophy  =  (normal_volume - actual_volume) / normal_volume
    ->  natural-language "Atrophy Description x s"  ->  Clinical ModernBERT.

The normative model is the *original* mmc2.xlsm workbook from Potvin et al.,
conditioned on subject demographics and acquisition metadata exactly as the
paper describes.  Nothing here is cohort-level: every prediction is per-subject.
"""
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

from .subcortical_norms import SubcorticalNorms


# FreeSurfer / FastSurfer aseg label -> Potvin workbook region label.
# Keys are the integer ROI ids used by the segmentation; values must match the
# row labels in mmc2.xlsm ("Statistics" sheet, column A).
ASEG_TO_NORM_LABEL: Dict[int, str] = {
    10: "Thalamus L",
    49: "Thalamus R",
    11: "Caudate L",
    50: "Caudate R",
    12: "Putamen L",
    51: "Putamen R",
    13: "Pallidum L",
    52: "Pallidum R",
    17: "Hippocampus L",
    53: "Hippocampus R",
    18: "Amygdala L",
    54: "Amygdala R",
    26: "Accumbens L",
    58: "Accumbens R",
    4:  "Lateral Ventricle L",
    43: "Lateral Ventricle R",
}


class AtrophyDescriber:
    """
    Turns measured ROI volumes + demographics into the per-ROI atrophy text that
    feeds the Clinical-ModernBERT text encoder.

    Parameters
    ----------
    workbook_path : str
        Path to ``mmc2.xlsm`` (shipped under ``normative/mmc2.xlsm``).
    """

    def __init__(self, workbook_path: Optional[str] = None):
        if workbook_path is None:
            workbook_path = str(Path(__file__).resolve().parent / "mmc2.xlsm")
        self.norms = SubcorticalNorms(workbook_path)

    def atrophy_for_roi(
        self,
        norm_label: str,
        measured_volume: float,
        age: float,
        sex: int,
        field_strength: int,
        manufacturer: int,
        icv: float,
    ) -> Dict[str, float]:
        """
        Returns the normative prediction and atrophy fraction for one ROI.

        atrophy = (normal - measured) / normal       (positive => volume loss)
        z       = (measured - normal) / se           (model-scale standardised dev.)
        """
        pred = self.norms.predict_region(
            norm_label, age=age, sex=sex, field_strength=field_strength,
            manufacturer=manufacturer, icv=icv,
        )
        normal = pred["pred"]
        atrophy = (normal - measured_volume) / normal if normal else 0.0
        z = (measured_volume - normal) / pred["se"] if pred["se"] else 0.0
        outside = not (pred["lower"] <= measured_volume <= pred["upper"])
        return {
            "normal_volume": normal,
            "measured_volume": measured_volume,
            "atrophy": atrophy,
            "z": z,
            "lower": pred["lower"],
            "upper": pred["upper"],
            "abnormal": float(outside),
        }

    def describe_roi(
        self,
        norm_label: str,
        measured_volume: float,
        age: float,
        sex: int,
        field_strength: int,
        manufacturer: int,
        icv: float,
    ) -> str:
        """One-sentence atrophy description for a single ROI (Figure 1A, yellow box)."""
        a = self.atrophy_for_roi(
            norm_label, measured_volume, age, sex, field_strength, manufacturer, icv
        )
        pct = a["atrophy"] * 100.0
        if pct > 1.0:
            direction = f"an atrophy of {pct:.2f}%"
        elif pct < -1.0:
            direction = f"an enlargement of {-pct:.2f}%"
        else:
            direction = "a volume within the normal range"
        flag = " (outside the 95% normative interval)" if a["abnormal"] else ""
        return (
            f"The {norm_label} exhibits {direction} relative to the demographic-"
            f"matched normative volume of {a['normal_volume']:.0f} mm^3{flag}."
        )

    def describe_subject(
        self,
        roi_volumes: Dict[int, float],
        age: float,
        sex: int,
        field_strength: int,
        manufacturer: int,
        icv: float,
    ) -> Dict[int, str]:
        """
        Atrophy description for every segmented ROI that has a normative model.

        Parameters
        ----------
        roi_volumes : dict[int, float]
            Measured volume (mm^3) keyed by aseg label id.
        """
        out: Dict[int, str] = {}
        for aseg_id, vol in roi_volumes.items():
            label = ASEG_TO_NORM_LABEL.get(int(aseg_id))
            if label is None or label not in self.norms.region_models:
                continue
            out[int(aseg_id)] = self.describe_roi(
                label, vol, age, sex, field_strength, manufacturer, icv
            )
        return out
