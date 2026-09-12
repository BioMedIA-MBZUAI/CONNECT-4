"""Atomic, fail-closed publication of paired CONNECT-4 predictions."""
from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

import nibabel as nib
import numpy as np
import torch

from data.provenance import sha256_file
from preprocessing.conform import (
    anatomical_affine,
    crop_architecture_array,
)
from eval.quality import evaluate_4d_pair_quality
from eval.visualize import save_fmri_nifti
from scripts.visualize_texture_audit import generate_texture_audit


PUBLICATION_SCHEMA = "connect4-gated-prediction-bundle-v1"


def _atomic_rename_noreplace(
    parent_fd: int,
    parent_path: Path,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically rename sibling directories without replacing any entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    result: int
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(parent_fd, source, parent_fd, destination, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            parent_fd,
            source,
            parent_fd,
            destination,
            0x00000004,
        )
    else:
        raise RuntimeError(
            "platform lacks an atomic no-replace directory rename primitive"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(parent_path / destination_name),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(parent_path / destination_name),
    )


def _component(value: object) -> str:
    text = str(value).strip()
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError("scan_id must be one safe filename component")
    return text


def _as_tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def _crop(value: Any, grid_contract: Mapping[str, Any] | None) -> Any:
    if grid_contract is None:
        return value
    return crop_architecture_array(value, grid_contract)


def _mask_nifti(value: Any, affine: np.ndarray, path: Path) -> None:
    array = _as_tensor(value).numpy()
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"publication mask must reduce to 3D, got {array.shape}")
    nib.save(nib.Nifti1Image((array > 0.5).astype(np.uint8), affine), path)


def _roi_label_nifti(value: Any, affine: np.ndarray, path: Path) -> None:
    array = _as_tensor(value).numpy()
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[0] < 2:
        raise ValueError(
            f"publication ROI masks must be [1,R,D,H,W] or [R,D,H,W], got {array.shape}"
        )
    foreground = array > 0.5
    labels = np.argmax(array, axis=0).astype(np.int16) + 1
    labels[~foreground.any(axis=0)] = 0
    nib.save(nib.Nifti1Image(labels, affine), path)


def publish_paired_prediction_bundle(
    final_bundle_dir: str | Path,
    *,
    scan_id: str,
    prediction: Any,
    real: Any,
    brain_mask: Any,
    roi_masks: Any,
    architecture_affine: np.ndarray,
    grid_contract: Mapping[str, Any] | None,
    require_structured_temporal: bool = True,
    quality_evaluator: Callable[..., dict] = evaluate_4d_pair_quality,
    texture_evaluator: Callable[..., Mapping[str, Path]] = generate_texture_audit,
) -> Path:
    """Stage one complete bundle, enforce both QA systems, then rename once."""
    identifier = _component(scan_id)
    final_dir = Path(final_bundle_dir).expanduser()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite published bundle: {final_dir}")
    staged = Path(
        tempfile.mkdtemp(prefix=f".{identifier}-gated-", dir=final_dir.parent)
    )
    try:
        prediction = _crop(_as_tensor(prediction), grid_contract)
        real = _crop(_as_tensor(real), grid_contract)
        brain_mask = _crop(_as_tensor(brain_mask), grid_contract)
        roi_masks = _crop(_as_tensor(roi_masks), grid_contract)
        affine = (
            anatomical_affine(grid_contract)
            if grid_contract is not None
            else np.asarray(architecture_affine, dtype=np.float64)
        )
        prediction_path = staged / "synthetic.nii.gz"
        real_path = staged / "real.nii.gz"
        mask_path = staged / "brain_mask.nii.gz"
        roi_path = staged / "roi_labels.nii.gz"
        save_fmri_nifti(
            prediction, str(prediction_path), affine=affine, repetition_time=3.0
        )
        save_fmri_nifti(real, str(real_path), affine=affine, repetition_time=3.0)
        _mask_nifti(brain_mask, affine, mask_path)
        _roi_label_nifti(roi_masks, affine, roi_path)

        quality = quality_evaluator(
            real_path,
            prediction_path,
            mask_path=mask_path,
            roi_labels_path=roi_path,
            require_structured_temporal=require_structured_temporal,
        )
        if quality.get("passed") is not True:
            raise RuntimeError(
                f"paired 4D quality gate rejected {identifier}: "
                f"{quality.get('verdict', 'unknown')}"
            )
        quality_path = staged / "paired_quality.json"
        quality_path.write_text(
            json.dumps(quality, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        texture_outputs = texture_evaluator(
            real_path,
            prediction_path,
            mask_path,
            staged / "texture",
            roi_labels_path=roi_path,
            prefix=identifier,
            enforce_texture_retention=True,
            enforce_texture_anti_gaming=True,
        )
        texture_manifest_path = Path(texture_outputs["manifest"])
        texture_manifest = json.loads(
            texture_manifest_path.read_text(encoding="utf-8")
        )
        gate = texture_manifest.get("quality_gate", {})
        if gate.get("release_gate_passed") is not True or int(
            gate.get("exit_code", 1)
        ) != 0:
            raise RuntimeError(
                f"texture/anti-gaming gate rejected {identifier}: "
                f"{gate.get('verdict', 'unknown')}"
            )

        artifacts = {
            str(path.relative_to(staged)): sha256_file(path)
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema": PUBLICATION_SCHEMA,
            "scan_id": identifier,
            "paired_4d_gate_passed": True,
            "texture_retention_gate_passed": True,
            "texture_anti_gaming_gate_passed": True,
            "architecture_padding_cropped": grid_contract is not None,
            "common_grid_contract_sha256": (
                grid_contract.get("contract_sha256")
                if grid_contract is not None
                else None
            ),
            "artifacts_sha256": artifacts,
        }
        (staged / "publication.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        parent_fd = os.open(
            final_dir.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            _atomic_rename_noreplace(
                parent_fd,
                final_dir.parent,
                staged.name,
                final_dir.name,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite published bundle: {final_dir}"
            ) from error
        finally:
            os.close(parent_fd)
        return final_dir.resolve(strict=True)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


__all__ = ["PUBLICATION_SCHEMA", "publish_paired_prediction_bundle"]
