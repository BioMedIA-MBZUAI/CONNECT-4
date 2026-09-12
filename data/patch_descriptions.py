"""Versioned biological text for CONNECT-4 mask patches.

The mask stream is intentionally text based.  Each ROI sentence contains two
separate pieces of information:

* a stable, curated anatomy-derived summary of known functional connectivity;
* the subject-specific normative-volume sentence produced from the Potvin
  model (age, sex, scanner manufacturer, field strength, and ICV/TIV).

Keeping this vocabulary deterministic is important: changing these sentences
changes the frozen Clinical-ModernBERT inputs and therefore invalidates cached
mask-patch embeddings. The manuscript specifies the two content categories but
does not publish a source or exact wording for the connectivity sentences; this
module makes that otherwise missing implementation choice explicit and auditable.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping, Tuple


PATCH_DESCRIPTION_SCHEMA_VERSION = "connect4-paper-fc-potvin-v3-ras-xyz"


# These are functional-network summaries, not the DWI matrix used by the graph
# edges.  The distinction matters: DWI is a structural-connectivity prior,
# whereas the paper requires known functional connectivity inside the text.
KNOWN_FUNCTIONAL_CONNECTIVITY: Dict[str, str] = {
    "left_cerebral_white_matter": (
        "supports coupling among left cortical networks and interhemispheric "
        "communication through commissural fibres"
    ),
    "right_cerebral_white_matter": (
        "supports coupling among right cortical networks and interhemispheric "
        "communication through commissural fibres"
    ),
    "left_cerebral_cortex": (
        "participates in distributed sensory, motor, association, and default-mode "
        "networks, with strong homotopic coupling to right cortex"
    ),
    "right_cerebral_cortex": (
        "participates in distributed sensory, motor, association, and default-mode "
        "networks, with strong homotopic coupling to left cortex"
    ),
    "left_lateral_ventricle": (
        "is a CSF space and has no intrinsic neuronal functional connectivity; "
        "its BOLD signal mainly reflects physiological and partial-volume effects"
    ),
    "right_lateral_ventricle": (
        "is a CSF space and has no intrinsic neuronal functional connectivity; "
        "its BOLD signal mainly reflects physiological and partial-volume effects"
    ),
    "left_inferior_lateral_ventricle": (
        "is a CSF space and has no intrinsic neuronal functional connectivity; "
        "it borders medial temporal structures where partial-volume effects may occur"
    ),
    "right_inferior_lateral_ventricle": (
        "is a CSF space and has no intrinsic neuronal functional connectivity; "
        "it borders medial temporal structures where partial-volume effects may occur"
    ),
    "third_ventricle": (
        "is a midline CSF space without intrinsic neuronal functional connectivity; "
        "its signal primarily indexes non-neural physiological variation"
    ),
    "fourth_ventricle": (
        "is a CSF space without intrinsic neuronal functional connectivity; "
        "its signal primarily indexes non-neural physiological variation"
    ),
    "csf": (
        "has no intrinsic neuronal functional connectivity and is commonly used to "
        "characterize non-neural physiological BOLD variation"
    ),
    "left_cerebellum_white_matter": (
        "supports left cerebello-thalamo-cortical communication with motor, cognitive, "
        "and association networks"
    ),
    "right_cerebellum_white_matter": (
        "supports right cerebello-thalamo-cortical communication with motor, cognitive, "
        "and association networks"
    ),
    "left_cerebellum_cortex": (
        "couples with contralateral sensorimotor cortex and distributed frontoparietal "
        "and default-mode networks through cerebello-thalamo-cortical loops"
    ),
    "right_cerebellum_cortex": (
        "couples with contralateral sensorimotor cortex and distributed frontoparietal "
        "and default-mode networks through cerebello-thalamo-cortical loops"
    ),
    "left_thalamus": (
        "is functionally coupled to ipsilateral cortical networks, basal ganglia, "
        "cerebellum, and the contralateral thalamus"
    ),
    "right_thalamus": (
        "is functionally coupled to ipsilateral cortical networks, basal ganglia, "
        "cerebellum, and the contralateral thalamus"
    ),
    "left_caudate": (
        "participates in associative and executive corticostriatal circuits, coupling "
        "with prefrontal cortex, thalamus, and the contralateral striatum"
    ),
    "right_caudate": (
        "participates in associative and executive corticostriatal circuits, coupling "
        "with prefrontal cortex, thalamus, and the contralateral striatum"
    ),
    "left_putamen": (
        "participates in sensorimotor corticostriatal circuits, coupling with motor "
        "cortex, pallidum, thalamus, and the contralateral putamen"
    ),
    "right_putamen": (
        "participates in sensorimotor corticostriatal circuits, coupling with motor "
        "cortex, pallidum, thalamus, and the contralateral putamen"
    ),
    "left_pallidum": (
        "participates in striato-pallido-thalamo-cortical loops and is functionally "
        "coupled to putamen, thalamus, and motor-association cortex"
    ),
    "right_pallidum": (
        "participates in striato-pallido-thalamo-cortical loops and is functionally "
        "coupled to putamen, thalamus, and motor-association cortex"
    ),
    "brain_stem": (
        "couples with cerebellar, thalamic, limbic, sensorimotor, arousal, and autonomic "
        "networks"
    ),
    "left_hippocampus": (
        "participates in medial temporal and default-mode networks, coupling with "
        "entorhinal cortex, posterior cingulate, amygdala, and right hippocampus"
    ),
    "right_hippocampus": (
        "participates in medial temporal and default-mode networks, coupling with "
        "entorhinal cortex, posterior cingulate, amygdala, and left hippocampus"
    ),
    "left_amygdala": (
        "participates in limbic and salience networks, coupling with hippocampus, "
        "ventromedial prefrontal cortex, insula, and right amygdala"
    ),
    "right_amygdala": (
        "participates in limbic and salience networks, coupling with hippocampus, "
        "ventromedial prefrontal cortex, insula, and left amygdala"
    ),
    "left_accumbens_area": (
        "participates in reward and motivational networks, coupling with orbitofrontal "
        "and ventromedial prefrontal cortex, amygdala, hippocampus, and ventral pallidum"
    ),
    "right_accumbens_area": (
        "participates in reward and motivational networks, coupling with orbitofrontal "
        "and ventromedial prefrontal cortex, amygdala, hippocampus, and ventral pallidum"
    ),
    "left_ventral_dc": (
        "participates in thalamic, hypothalamic, subthalamic, limbic, and autonomic "
        "circuits with basal-ganglia and cortical networks"
    ),
    "right_ventral_dc": (
        "participates in thalamic, hypothalamic, subthalamic, limbic, and autonomic "
        "circuits with basal-ganglia and cortical networks"
    ),
}


_NORMATIVE_ALIASES = {
    "lateral_l": "left_lateral_ventricle",
    "lateral_r": "right_lateral_ventricle",
    "thalamus_l": "left_thalamus",
    "thalamus_r": "right_thalamus",
    "caudate_l": "left_caudate",
    "caudate_r": "right_caudate",
    "putamen_l": "left_putamen",
    "putamen_r": "right_putamen",
    "pallidum_l": "left_pallidum",
    "pallidum_r": "right_pallidum",
    "hippocampus_l": "left_hippocampus",
    "hippocampus_r": "right_hippocampus",
    "amygdala_l": "left_amygdala",
    "amygdala_r": "right_amygdala",
    "accumbens_l": "left_accumbens_area",
    "accumbens_r": "right_accumbens_area",
    "lateral_ventricle_l": "left_lateral_ventricle",
    "lateral_ventricle_r": "right_lateral_ventricle",
    "inferior_lateral_l": "left_inferior_lateral_ventricle",
    "inferior_lateral_r": "right_inferior_lateral_ventricle",
    "3rd": "third_ventricle",
    "4th": "fourth_ventricle",
    "brainstem": "brain_stem",
    "ventral_dc_l": "left_ventral_dc",
    "ventral_dc_r": "right_ventral_dc",
}


def normalise_structure_name(name: object) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return _NORMATIVE_ALIASES.get(key, key)


# Private spelling retained for old imports; all production code uses the
# public helper so ROI-name normalization has one auditable definition.
_normalise_structure_name = normalise_structure_name


def build_normative_index(
    descriptions: Mapping[Tuple[str, str], str],
) -> Dict[Tuple[str, str], str]:
    """Normalise patient/ROI keys so Potvin labels and SynthSeg slugs match."""
    index: Dict[Tuple[str, str], str] = {}
    for (patient_id, structure), description in descriptions.items():
        text = str(description).strip()
        if text:
            index[(str(patient_id).strip().lower(), _normalise_structure_name(structure))] = text
    return index


def functional_connectivity_description(structure_slug: str) -> str:
    """Return the curated functional-connectivity sentence for one ROI slug."""
    slug = _normalise_structure_name(structure_slug)
    summary = KNOWN_FUNCTIONAL_CONNECTIVITY.get(slug)
    if summary is None:
        return "Known functional connectivity is not curated for this segmentation label."
    return f"Known functional connectivity: {summary}."


def biological_roi_description(
    structure_slug: str,
    coverage: float,
    normative_description: str = "",
) -> str:
    """Build connectivity context plus an explicit normative-availability section."""
    readable = _normalise_structure_name(structure_slug).replace("_", " ")
    fc_text = functional_connectivity_description(structure_slug)
    norm_text = str(normative_description).strip()
    if norm_text:
        norm_context = f"Normative volume context: {norm_text}"
    else:
        norm_context = (
            "Normative volume context: no valid subject-specific normative estimate "
            "is available for this structure."
        )
    return f"{readable} ({float(coverage) * 100.0:.1f}% patch coverage). {fc_text} {norm_context}"


def build_patch_description(
    patch_idx: int,
    center_mm: Tuple[float, float, float],
    distribution: Mapping[int, float],
    id_to_slug: Mapping[int, str],
    normative_index: Mapping[Tuple[str, str], str],
    patient_id: str,
) -> str:
    """Build versioned patch text with connectivity and normative availability."""
    x_mm, y_mm, z_mm = center_mm
    valid = {
        int(structure_id): float(coverage)
        for structure_id, coverage in distribution.items()
        if int(structure_id) in id_to_slug and float(coverage) > 0.0
    }
    parts = [
        f"Patch {patch_idx} at RAS+ voxel center "
        f"(X={x_mm:.1f} mm left-to-right, "
        f"Y={y_mm:.1f} mm posterior-to-anterior, "
        f"Z={z_mm:.1f} mm inferior-to-superior) "
        f"contains {len(valid)} ROIs with the following structural distribution:"
    ]
    base_patient_id = str(patient_id).strip().lower()
    for structure_id, coverage in sorted(valid.items(), key=lambda item: item[1], reverse=True):
        slug = id_to_slug[structure_id]
        norm_text = normative_index.get(
            (base_patient_id, _normalise_structure_name(slug)), ""
        )
        parts.append(biological_roi_description(slug, coverage, norm_text))
    return " ".join(parts)


__all__ = [
    "PATCH_DESCRIPTION_SCHEMA_VERSION",
    "KNOWN_FUNCTIONAL_CONNECTIVITY",
    "biological_roi_description",
    "build_normative_index",
    "build_patch_description",
    "functional_connectivity_description",
    "normalise_structure_name",
]
