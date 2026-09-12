"""Hash-bound spatial-grid handling for CONNECT-4.

The manuscript fixes 3-mm isotropic voxels, 128 fMRI frames, and TR=3 s. It
does *not* report a spatial matrix. Production code therefore has no implicit
matrix default: callers must supply a validated, T1-only cohort-common grid
contract. Any padding needed by patch/token architectures is recorded in that
contract and is cropped before anatomical publication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


COMMON_GRID_SCHEMA_VERSION = "connect4-structural-common-grid-v2"
STRUCTURAL_MANIFEST_SCHEMA_VERSION = "connect4-structural-cohort-v2"
ARCHITECTURE_PADDING_SCHEMA_VERSION = "connect4-architecture-padding-v1"
TARGET_VOXEL = (3.0, 3.0, 3.0)
TARGET_FRAMES = 128
TR_SECONDS = 3.0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _triplet(
    value: object, name: str, *, positive: bool = True
) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain three integers")
    result = tuple(int(item) for item in value)
    minimum = 1 if positive else 0
    if any(item < minimum for item in result):
        raise ValueError(f"{name} entries must be >= {minimum}")
    return result


def _resolved_bound_path(record_path: Path, value: object, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"common-grid contract is missing {label} path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = record_path.parent / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"common-grid {label} does not exist: {path}") from exc


def load_common_grid_contract(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and independently validate a production common-grid contract."""
    contract_path = Path(path).expanduser().resolve(strict=True)
    contract_digest = _sha256_file(contract_path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if not _is_sha256(expected) or contract_digest != expected:
            raise ValueError(
                "common-grid contract SHA-256 does not match configuration"
            )
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid common-grid JSON: {contract_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("common-grid contract must be a JSON object")
    if payload.get("schema") != COMMON_GRID_SCHEMA_VERSION:
        raise ValueError(
            f"common-grid contract must declare schema={COMMON_GRID_SCHEMA_VERSION!r}"
        )
    recorded_record_digest = payload.get("record_sha256")
    unsigned = dict(payload)
    unsigned.pop("record_sha256", None)
    if not _is_sha256(
        recorded_record_digest
    ) or recorded_record_digest != _canonical_sha256(unsigned):
        raise ValueError("common-grid record_sha256 is invalid")
    if payload.get("cohort_common") is not True:
        raise ValueError("common-grid contract must be cohort-common")
    derivation = payload.get("derivation")
    if (
        not isinstance(derivation, Mapping)
        or derivation.get("source") != "structural-reference-and-cohort-T1-manifest"
        or derivation.get("functional_data_used") is not False
        or derivation.get("matrix_size_reported_by_paper") is not False
    ):
        raise ValueError(
            "common-grid derivation must be T1-only and acknowledge that the paper "
            "does not report a matrix size"
        )

    manifest_record = payload.get("structural_manifest")
    reference_record = payload.get("reference")
    source_reference_record = payload.get("source_structural_reference")
    if (
        not isinstance(manifest_record, Mapping)
        or not isinstance(reference_record, Mapping)
        or not isinstance(source_reference_record, Mapping)
    ):
        raise ValueError(
            "common-grid structural manifest/source/reference evidence is missing"
        )
    if (
        source_reference_record.get("modality") != "T1w"
        or source_reference_record.get("cohort_derived") is not True
        or not str(source_reference_record.get("derivation_method", "")).strip()
    ):
        raise ValueError(
            "common-grid source must be an explicitly cohort-derived T1w reference"
        )
    manifest_path = _resolved_bound_path(
        contract_path, manifest_record.get("path"), "manifest"
    )
    reference_path = _resolved_bound_path(
        contract_path, reference_record.get("path"), "reference"
    )
    source_reference_path = _resolved_bound_path(
        contract_path,
        source_reference_record.get("path"),
        "source structural reference",
    )
    for evidence, artifact, label in (
        (manifest_record, manifest_path, "structural manifest"),
        (
            source_reference_record,
            source_reference_path,
            "source structural reference",
        ),
        (reference_record, reference_path, "reference"),
    ):
        digest = evidence.get("sha256")
        if not _is_sha256(digest) or _sha256_file(artifact) != digest:
            raise ValueError(f"common-grid {label} SHA-256 does not match")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("common-grid structural manifest is invalid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != STRUCTURAL_MANIFEST_SCHEMA_VERSION
        or manifest.get("functional_data_used") is not False
        or not isinstance(manifest.get("scans"), list)
        or not manifest["scans"]
    ):
        raise ValueError(
            "common-grid structural manifest is not a T1-only cohort manifest"
        )
    manifest_reference = manifest.get("structural_reference")
    if (
        not isinstance(manifest_reference, Mapping)
        or manifest_reference.get("modality") != "T1w"
        or manifest_reference.get("cohort_derived") is not True
        or manifest_reference.get("functional_data_used") is not False
        or not str(manifest_reference.get("derivation_method", "")).strip()
        or not _is_sha256(manifest_reference.get("sha256"))
    ):
        raise ValueError(
            "structural manifest lacks explicit T1w cohort-reference evidence"
        )
    bound_manifest_reference = _resolved_bound_path(
        manifest_path,
        manifest_reference.get("path"),
        "manifest structural reference",
    )
    if bound_manifest_reference != source_reference_path or _sha256_file(
        bound_manifest_reference
    ) != manifest_reference.get("sha256"):
        raise ValueError(
            "manifest structural reference does not match the grid source reference"
        )
    forbidden_tokens = ("fmri", "bold", "functional")
    seen_scan_ids: set[str] = set()
    for row in manifest["scans"]:
        if not isinstance(row, Mapping):
            raise ValueError("structural manifest scan rows must be objects")
        if any(token in str(key).lower() for key in row for token in forbidden_tokens):
            raise ValueError(
                "functional-image fields are forbidden in the grid manifest"
            )
        scan_id = str(row.get("scan_id", "")).strip()
        if (
            not scan_id
            or scan_id in seen_scan_ids
            or not _is_sha256(row.get("t1w_sha256"))
        ):
            raise ValueError(
                "each structural manifest row needs a unique scan_id and T1w SHA-256"
            )
        seen_scan_ids.add(scan_id)
    if int(manifest_record.get("scan_count", -1)) != len(manifest["scans"]):
        raise ValueError("common-grid structural manifest scan count is inconsistent")

    anatomical_shape = _triplet(payload.get("anatomical_shape"), "anatomical_shape")
    architecture_shape = _triplet(
        payload.get("architecture_shape"), "architecture_shape"
    )
    padding = payload.get("architecture_padding")
    if (
        not isinstance(padding, Mapping)
        or padding.get("schema") != ARCHITECTURE_PADDING_SCHEMA_VERSION
        or padding.get("purpose") != "patch-multiple-only-not-paper-anatomy"
    ):
        raise ValueError("common-grid architecture-padding evidence is invalid")
    before = _triplet(
        padding.get("before"), "architecture_padding.before", positive=False
    )
    after = _triplet(padding.get("after"), "architecture_padding.after", positive=False)
    multiple = _triplet(padding.get("multiple"), "architecture_padding.multiple")
    smallest_patch_shape = tuple(
        ((size + patch - 1) // patch) * patch
        for size, patch in zip(anatomical_shape, multiple)
    )
    if (
        tuple(a + b + c for a, b, c in zip(anatomical_shape, before, after))
        != architecture_shape
    ):
        raise ValueError(
            "architecture shape does not equal anatomy plus recorded padding"
        )
    if architecture_shape != smallest_patch_shape or any(
        abs(start - end) > 1 for start, end in zip(before, after)
    ):
        raise ValueError(
            "architecture padding must be the smallest symmetric padding to the "
            "recorded patch multiple"
        )

    voxel_size = np.asarray(payload.get("voxel_size_mm"), dtype=np.float64)
    if voxel_size.shape != (3,) or not np.allclose(
        voxel_size, TARGET_VOXEL, rtol=0.0, atol=1e-12
    ):
        raise ValueError("common-grid contract must record 3-mm isotropic voxels")
    source_reference = nib.load(str(source_reference_path))
    if source_reference.ndim != 3 or not np.isfinite(source_reference.affine).all():
        raise ValueError("common-grid source structural reference is invalid")

    reference = nib.load(str(reference_path))
    if reference.ndim != 3 or tuple(reference.shape) != architecture_shape:
        raise ValueError("common-grid reference NIfTI has the wrong architecture shape")
    zooms = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    if zooms.shape != (3,) or not np.allclose(zooms, TARGET_VOXEL, rtol=0.0, atol=1e-6):
        raise ValueError("common-grid reference must have 3-mm isotropic voxels")
    recorded_affine = np.asarray(reference_record.get("affine"), dtype=np.float64)
    if recorded_affine.shape != (4, 4) or not np.allclose(
        recorded_affine, reference.affine, rtol=0.0, atol=1e-6
    ):
        raise ValueError("common-grid reference affine does not match its contract")
    anatomical_affine_value = reference.affine.copy()
    anatomical_affine_value[:3, 3] += reference.affine[:3, :3] @ np.asarray(before)
    recorded_anatomical_affine = np.asarray(
        payload.get("anatomical_affine"), dtype=np.float64
    )
    if recorded_anatomical_affine.shape != (4, 4) or not np.allclose(
        recorded_anatomical_affine,
        anatomical_affine_value,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("common-grid anatomical crop affine is inconsistent")

    validated = dict(payload)
    validated["contract_path"] = str(contract_path)
    validated["contract_sha256"] = contract_digest
    validated["structural_manifest_path"] = str(manifest_path)
    validated["source_structural_reference_path"] = str(source_reference_path)
    validated["reference_path"] = str(reference_path)
    return validated


def contract_reference(contract: Mapping[str, Any]) -> nib.Nifti1Image:
    """Load the already-validated architecture-grid reference."""
    return nib.load(str(contract["reference_path"]))


def conform_volume(
    img: nib.Nifti1Image,
    order: int = 1,
    *,
    grid_contract: Mapping[str, Any] | None = None,
    synthetic_target: tuple[Sequence[int], np.ndarray] | None = None,
) -> nib.Nifti1Image:
    """Resample a 3D image to an explicit grid; there is no production default."""
    if grid_contract is not None and synthetic_target is not None:
        raise ValueError(
            "choose a validated common grid or an explicit synthetic target"
        )
    if grid_contract is not None:
        reference = contract_reference(grid_contract)
        target = (tuple(reference.shape), reference.affine)
    elif synthetic_target is not None:
        shape, affine = synthetic_target
        target = (
            _triplet(tuple(shape), "synthetic target shape"),
            np.asarray(affine),
        )
    else:
        raise ValueError(
            "spatial conforming requires a validated common-grid contract; "
            "tests may pass synthetic_target explicitly"
        )
    canonical = nib.as_closest_canonical(img)
    return resample_from_to(canonical, target, order=order, mode="constant", cval=0.0)


def conform_4d(
    img: nib.Nifti1Image,
    order: int = 1,
    *,
    tr_seconds: float | None = None,
    grid_contract: Mapping[str, Any] | None = None,
    synthetic_target: tuple[Sequence[int], np.ndarray] | None = None,
) -> nib.Nifti1Image:
    """Resample 4D spatial axes to one explicit reference, preserving source TR."""
    input_zooms = img.header.get_zooms()
    input_tr = float(input_zooms[3]) if len(input_zooms) >= 4 else None
    _spatial_unit, input_time_unit = img.header.get_xyzt_units()
    if tr_seconds is None:
        if input_tr is None or not np.isfinite(input_tr) or input_tr <= 0:
            raise ValueError("4D NIfTI header must contain a positive finite TR")
        output_tr = input_tr
        output_time_unit = input_time_unit
    else:
        output_tr = float(tr_seconds)
        if not np.isfinite(output_tr) or output_tr <= 0:
            raise ValueError("tr_seconds must be positive and finite")
        output_time_unit = "sec"
    canonical = nib.as_closest_canonical(img)
    data = canonical.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    frames = []
    reference = None
    for frame_index in range(data.shape[3]):
        frame = nib.Nifti1Image(
            data[..., frame_index], canonical.affine, canonical.header
        )
        if grid_contract is None and synthetic_target is None:
            # Retains a narrow monkeypatch seam for TR-only unit tests; the real
            # conform_volume implementation itself remains fail-closed.
            conformed = conform_volume(frame, order=order)
        else:
            conformed = conform_volume(
                frame,
                order=order,
                grid_contract=grid_contract,
                synthetic_target=synthetic_target,
            )
        reference = conformed
        frames.append(conformed.get_fdata(dtype=np.float32))
    if reference is None:
        raise ValueError("4D image contains zero frames")
    output = np.stack(frames, axis=-1)
    result = nib.Nifti1Image(output, reference.affine, reference.header)
    result.header.set_xyzt_units(xyz="mm", t=output_time_unit or "unknown")
    result.header.set_zooms(tuple(result.header.get_zooms()[:3]) + (output_tr,))
    return result


def set_num_frames(data4d: np.ndarray, n: int = TARGET_FRAMES) -> np.ndarray:
    """Truncate to ``n`` frames and reject shorter runs; never synthesize frames."""
    frames = int(data4d.shape[-1])
    if frames < n:
        raise ValueError(
            f"fMRI run has {frames} frames after TR harmonisation; at least {n} "
            "real frames are required and temporal padding is forbidden"
        )
    return data4d[..., :n]


def anatomical_crop_slices(
    contract: Mapping[str, Any],
) -> tuple[slice, slice, slice]:
    before = tuple(int(value) for value in contract["architecture_padding"]["before"])
    shape = tuple(int(value) for value in contract["anatomical_shape"])
    return tuple(  # type: ignore[return-value]
        slice(start, start + size) for start, size in zip(before, shape)
    )


def crop_architecture_array(array: Any, contract: Mapping[str, Any]) -> Any:
    """Crop the final three spatial axes from architecture padding to anatomy."""
    expected = tuple(int(value) for value in contract["architecture_shape"])
    if tuple(array.shape[-3:]) != expected:
        raise ValueError(
            f"architecture array has spatial shape {tuple(array.shape[-3:])}, "
            f"expected {expected}"
        )
    crop = anatomical_crop_slices(contract)
    return array[(...,) + crop]


def anatomical_affine(contract: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(contract["anatomical_affine"], dtype=np.float64).copy()


def zscore_inbrain(
    vol: np.ndarray,
    eps: float = 1e-6,
    *,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score an explicit brain support and keep its exterior exactly zero.

    Production T1 preprocessing supplies the authenticated, conformed SynthSeg
    support.  The nonzero fallback is retained only for isolated synthetic
    tests and non-production callers.
    """
    values = np.asarray(vol)
    if not np.isfinite(values).all():
        raise ValueError("T1w normalization input contains NaN or infinity")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("T1w normalization epsilon must be positive and finite")
    if support is None:
        normalized_support = np.abs(values) > 0
    else:
        normalized_support = np.asarray(support)
        if normalized_support.shape != values.shape:
            raise ValueError("T1w normalization support shape differs")
        if normalized_support.dtype != np.bool_:
            raise ValueError("T1w normalization support must be boolean")
    output = np.zeros(values.shape, dtype=np.result_type(values.dtype, np.float32))
    if not np.any(normalized_support):
        return output
    brain = values[normalized_support]
    mu = brain.mean()
    sd = brain.std()
    output[normalized_support] = np.clip((brain - mu) / (sd + eps), -3.0, 3.0)
    return output


__all__ = [
    "ARCHITECTURE_PADDING_SCHEMA_VERSION",
    "COMMON_GRID_SCHEMA_VERSION",
    "STRUCTURAL_MANIFEST_SCHEMA_VERSION",
    "TARGET_VOXEL",
    "TARGET_FRAMES",
    "TR_SECONDS",
    "anatomical_affine",
    "anatomical_crop_slices",
    "conform_4d",
    "conform_volume",
    "contract_reference",
    "crop_architecture_array",
    "load_common_grid_contract",
    "set_num_frames",
    "zscore_inbrain",
]
