"""Extract provenance-bound PyRadiomics and 3D AnatCL ROI features.

AnatCL 0.0.2 is a 3D network trained on CAT12 modulated, spatially
normalised gray-matter VBM volumes. Raw/native T1w data and the model's old
three-plane adapter are deliberately rejected. A separately authenticated,
target-free CAT12 authority must provide each ``mwp1`` volume and a SynthSeg
label volume on the exact same 1.5-mm CAT12 grid.

The CONNECT-4 paper says that foundation-model features are ROI-level but does
not specify an adapter. This implementation records one explicit choice: zero
every VBM voxel outside one ROI, apply the upstream deterministic center-crop
and padding transform, and obtain one 512-D descriptor from the 3D network.
PyRadiomics is accepted only from a frozen official ``pyradiomics`` 3.0.1
runtime; the similarly named ``pyradiomics-cuda`` fork is forbidden.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import stat
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from data.dataset import Connect4Dataset
from data.provenance import canonical_sha256, sha256_file


ROI_FEATURE_SCHEMA = Connect4Dataset.ANATCL_PROVENANCE_SCHEMA
CAT12_INPUT_SCHEMA = Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA
RADIOMICS_FEATURE_SCHEMA = Connect4Dataset.RADIOMICS_PROVENANCE_SCHEMA
CAT12_AUTHORITY_SCHEMA = "connect4-cat12-anatcl-authority-v1"
PYRADIOMICS_RUNTIME_LOCK_SCHEMA = "connect4-official-pyradiomics-runtime-lock-v1"
PYRADIOMICS_INSTALLED_FILES_SCHEMA = "connect4-official-pyradiomics-installed-files-v1"

ANATCL_SOURCE_FILES_SHA256 = {
    "__init__.py": "5f3e8e4dacdd60eeb81d3cbacf060d0a8580169c6cd0223213c52e547f71c9b0",
    "anatcl.py": "505dd542452ec91db74effc8d9333ae0347f618669dacf6827952d09c3cf9633",
    "models/__init__.py": (
        "b95e008fab63fd31d74b0c0ea7d2879b8df55b7982b330502ac34a715bbdcce6"
    ),
    "models/resnet3d.py": (
        "bca448a5e9e9e4ec5f86ecb5e5077640aa50b90721959701472c0068de1a6bf5"
    ),
}
ANATCL_SOURCE_TREE_SHA256 = (
    "ed74362e12f574d71f7c7aba638a89f4c09fe15de36663f8e523125fd71ac638"
)
ANATCL_WEIGHTS_SHA256 = (
    "77f6d44d0317ffc704ffc751db7eab349a8109e2cd596ab06070f1aec7a55d64"
)
ANATCL_STATE_DICT_SHA256 = (
    "ef695e5926cb5a09867d8bbd61b781ed873b50571bc058ad746a78b95ff00639"
)
PYRADIOMICS_SDIST_SHA256 = (
    "47c57f441d6cb7973fa3b2ea48d3948df78e3348e1c69e1e2ff19001601fc2f5"
)
PYRADIOMICS_UPSTREAM_REVISION = "08bea7067e350303eead533b471aa60697b3b8c3"
SOURCE_SHAPE = (121, 145, 121)
CROPPED_SHAPE = (121, 128, 121)
MODEL_SHAPE = (128, 128, 128)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_NIFTI_BYTES = 256 * 1024 * 1024

_VERIFIED_PYRADIOMICS_RUNTIME: dict[str, object] | None = None


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_existing_file(path: Path, *, label: str) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    lexical = Path(os.path.abspath(os.fspath(requested)))
    if requested != lexical:
        raise ValueError(f"{label} must be lexically canonical")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing: {requested}") from exc
    if resolved != requested:
        raise ValueError(f"{label} aliases another path")
    return requested


def _snapshot_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    require_frozen: bool,
) -> tuple[bytes, str, int]:
    source = _canonical_existing_file(path, label=label)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for authenticated inputs")
    descriptor = os.open(source, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be one regular, unlinked file")
        if require_frozen and before.st_mode & 0o222:
            raise ValueError(f"{label} must be non-writable")
        if before.st_size < 1 or before.st_size > max_bytes:
            raise ValueError(f"{label} size is outside the authenticated bound")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} was truncated during authentication")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"{label} grew during authentication")
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_mode,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_mode,
            after.st_nlink,
        )
        identity_path = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_mode,
            current.st_nlink,
        )
        if identity_before != identity_after or identity_after != identity_path:
            raise ValueError(f"{label} changed during authentication")
        payload = b"".join(chunks)
        return payload, hashlib.sha256(payload).hexdigest(), int(before.st_size)
    finally:
        os.close(descriptor)


def _read_json_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    require_frozen: bool = True,
) -> tuple[dict[str, Any], str]:
    if not _valid_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA-256 is invalid")
    payload, observed, _ = _snapshot_regular_file(
        path,
        label=label,
        max_bytes=MAX_JSON_BYTES,
        require_frozen=require_frozen,
    )
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, observed


def _validate_signed_record(value: Mapping[str, Any], *, label: str) -> None:
    recorded = value.get("record_sha256")
    if not _valid_sha256(recorded):
        raise ValueError(f"{label} record SHA-256 is invalid")
    unsigned = dict(value)
    del unsigned["record_sha256"]
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"{label} signed record differs")


def _declared_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative POSIX path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError(f"{label} is not a canonical relative POSIX path")
    return relative


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _snapshot_runtime_file_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> dict[str, object]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for runtime authentication")
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(f"{label} must be one non-hardlinked regular file")
    if before.st_mode & 0o222:
        raise RuntimeError(f"{label} must be non-writable")
    descriptor = os.open(name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if _directory_identity(opened) != _directory_identity(before):
            raise RuntimeError(f"{label} changed before authentication")
        digest = hashlib.sha256()
        remaining = int(opened.st_size)
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"{label} was truncated during authentication")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew during authentication")
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not (
            _directory_identity(before)
            == _directory_identity(after)
            == _directory_identity(current)
        ):
            raise RuntimeError(f"{label} changed during authentication")
        return {
            "sha256": digest.hexdigest(),
            "size": int(opened.st_size),
            "mode": stat.S_IMODE(opened.st_mode),
        }
    finally:
        os.close(descriptor)


def _snapshot_frozen_runtime_tree(runtime_root: Path) -> dict[str, object]:
    """Return one descriptor-rooted inventory of an immutable runtime tree."""
    root = _canonical_existing_file(runtime_root, label="PyRadiomics runtime root")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise RuntimeError(
            "O_NOFOLLOW and O_DIRECTORY are required for runtime authentication"
        )
    root_descriptor = os.open(root, os.O_RDONLY | nofollow | directory_flag)
    directories: dict[str, dict[str, int]] = {}
    files: dict[str, dict[str, object]] = {}

    def visit(descriptor: int, relative: PurePosixPath | None) -> None:
        before = os.fstat(descriptor)
        label = "." if relative is None else relative.as_posix()
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError(f"PyRadiomics runtime {label} is not a directory")
        if before.st_mode & 0o222:
            raise RuntimeError(f"PyRadiomics runtime directory {label} is writable")
        directories[label] = {"mode": stat.S_IMODE(before.st_mode)}
        names = sorted(os.listdir(descriptor))
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise RuntimeError("PyRadiomics runtime has an invalid entry name")
            child_relative = (
                PurePosixPath(name) if relative is None else relative / name
            )
            child_label = child_relative.as_posix()
            child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(child.st_mode):
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | nofollow | directory_flag,
                    dir_fd=descriptor,
                )
                try:
                    if _directory_identity(os.fstat(child_descriptor)) != (
                        _directory_identity(child)
                    ):
                        raise RuntimeError(
                            f"PyRadiomics runtime directory {child_label} changed"
                        )
                    visit(child_descriptor, child_relative)
                finally:
                    os.close(child_descriptor)
            elif stat.S_ISREG(child.st_mode):
                files[child_label] = _snapshot_runtime_file_at(
                    descriptor,
                    name,
                    label=f"PyRadiomics runtime file {child_label}",
                )
            else:
                raise RuntimeError(
                    f"PyRadiomics runtime entry {child_label} is a symlink or special file"
                )
        if sorted(os.listdir(descriptor)) != names:
            raise RuntimeError(f"PyRadiomics runtime directory {label} changed")
        after = os.fstat(descriptor)
        if _directory_identity(before) != _directory_identity(after):
            raise RuntimeError(f"PyRadiomics runtime directory {label} changed")

    try:
        root_before = os.lstat(root)
        if _directory_identity(os.fstat(root_descriptor)) != _directory_identity(
            root_before
        ):
            raise RuntimeError("PyRadiomics runtime root changed before authentication")
        visit(root_descriptor, None)
        root_after = os.lstat(root)
        if _directory_identity(os.fstat(root_descriptor)) != _directory_identity(
            root_after
        ):
            raise RuntimeError("PyRadiomics runtime root changed during authentication")
    finally:
        os.close(root_descriptor)
    return {"directories": directories, "files": files}


def _verify_pyradiomics_installed_files(
    lock_path: Path,
    runtime: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Authenticate every byte before resolving or importing ``radiomics``."""
    if set(runtime) != {
        "python_abi",
        "platform",
        "runtime_root_relative_path",
        "site_packages_relative_path",
        "installed_files_manifest_relative_path",
        "runtime_tree_sha256",
        "installed_files_manifest_sha256",
    }:
        raise RuntimeError("PyRadiomics frozen runtime fields differ")
    runtime_relative = _declared_relative_path(
        runtime.get("runtime_root_relative_path"),
        label="PyRadiomics runtime root",
    )
    site_relative = _declared_relative_path(
        runtime.get("site_packages_relative_path"),
        label="PyRadiomics site-packages root",
    )
    manifest_relative = _declared_relative_path(
        runtime.get("installed_files_manifest_relative_path"),
        label="PyRadiomics installed-files manifest",
    )
    lock_parent = _canonical_existing_file(
        lock_path.parent, label="PyRadiomics lock parent"
    )
    runtime_root = lock_parent.joinpath(*runtime_relative.parts)
    site_packages = runtime_root.joinpath(*site_relative.parts)
    manifest_path = lock_parent.joinpath(*manifest_relative.parts)
    manifest, observed_manifest_sha256 = _read_json_snapshot(
        manifest_path,
        expected_sha256=str(runtime.get("installed_files_manifest_sha256", "")),
        label="PyRadiomics installed-files manifest",
    )
    if set(manifest) != {
        "schema",
        "distribution",
        "package_version",
        "directories",
        "files",
        "runtime_tree_sha256",
        "record_sha256",
    }:
        raise RuntimeError("PyRadiomics installed-files manifest fields differ")
    _validate_signed_record(manifest, label="PyRadiomics installed-files manifest")
    declared_inventory = {
        "directories": manifest.get("directories"),
        "files": manifest.get("files"),
    }
    if (
        manifest.get("schema") != PYRADIOMICS_INSTALLED_FILES_SCHEMA
        or manifest.get("distribution") != "pyradiomics"
        or manifest.get("package_version") != "3.0.1"
        or not isinstance(declared_inventory["directories"], dict)
        or not isinstance(declared_inventory["files"], dict)
        or not declared_inventory["files"]
        or canonical_sha256(declared_inventory) != manifest.get("runtime_tree_sha256")
        or manifest.get("runtime_tree_sha256") != runtime.get("runtime_tree_sha256")
    ):
        raise RuntimeError("PyRadiomics installed-files identity differs")
    observed_inventory = _snapshot_frozen_runtime_tree(runtime_root)
    if observed_inventory != declared_inventory:
        raise RuntimeError("PyRadiomics installed runtime bytes or inventory differ")
    if not site_packages.is_dir():
        raise RuntimeError("PyRadiomics site-packages root is missing")
    resolved_runtime = runtime_root.resolve(strict=True)
    resolved_site = site_packages.resolve(strict=True)
    if resolved_runtime != runtime_root or resolved_site != site_packages:
        raise RuntimeError("PyRadiomics runtime paths must not contain aliases")
    if resolved_runtime not in resolved_site.parents:
        raise RuntimeError("PyRadiomics site-packages escapes its runtime root")
    if Path(sys.prefix).resolve(strict=True) != resolved_runtime:
        raise RuntimeError(
            "current interpreter is outside the frozen PyRadiomics runtime"
        )
    return (
        resolved_runtime,
        resolved_site,
        {
            "manifest_sha256": observed_manifest_sha256,
            "runtime_tree_sha256": manifest["runtime_tree_sha256"],
        },
    )


def load_cat12_authority_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_scan_ids: list[str],
) -> tuple[dict[str, Any], str]:
    """Authenticate the externally pinned, target-free CAT12 authority."""
    manifest, observed = _read_json_snapshot(
        path,
        expected_sha256=expected_sha256,
        label="CAT12 authority manifest",
    )
    if set(manifest) != {"schema", "purpose", "pipeline", "scans", "record_sha256"}:
        raise ValueError("CAT12 authority manifest fields differ")
    _validate_signed_record(manifest, label="CAT12 authority manifest")
    if manifest["schema"] != CAT12_AUTHORITY_SCHEMA:
        raise ValueError("CAT12 authority schema differs")
    if manifest["purpose"] != "target-free-train-anatcl-vbm-inputs":
        raise ValueError("CAT12 authority purpose differs")
    if manifest["pipeline"] != Connect4Dataset.CAT12_PREPROCESSING_CONTRACT:
        raise ValueError("CAT12 authority pipeline differs")
    scans = manifest.get("scans")
    if not isinstance(scans, dict) or set(scans) != set(expected_scan_ids):
        raise ValueError("CAT12 authority scan coverage differs")
    for scan_id in expected_scan_ids:
        entry = scans[scan_id]
        if not isinstance(entry, dict) or set(entry) != {
            "record_relative_path",
            "record_file_sha256",
            "record_sha256",
        }:
            raise ValueError(f"CAT12 authority entry differs for {scan_id}")
        if entry["record_relative_path"] != f"{scan_id}/provenance.json":
            raise ValueError(f"CAT12 authority path differs for {scan_id}")
        if not _valid_sha256(entry["record_file_sha256"]) or not _valid_sha256(
            entry["record_sha256"]
        ):
            raise ValueError(f"CAT12 authority record hash is invalid for {scan_id}")
    return manifest, observed


def _load_nifti_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    payload, observed, observed_size = _snapshot_regular_file(
        path,
        label=label,
        max_bytes=MAX_NIFTI_BYTES,
        require_frozen=True,
    )
    if observed != expected_sha256 or observed_size != expected_size:
        raise ValueError(f"{label} content identity differs")
    try:
        if path.name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
                raw = stream.read(MAX_NIFTI_BYTES + 1)
            if len(raw) > MAX_NIFTI_BYTES:
                raise ValueError(f"{label} decompressed size exceeds the bound")
        else:
            raw = payload
        image = nib.Nifti1Image.from_bytes(raw)
        array = image.get_fdata(dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"{label} is not a valid bounded NIfTI image") from exc
    return image, array


def load_cat12_scan_inputs(
    authority_root: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    scan_id: str,
    source_t1_sha256: str,
    source_segmentation_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one scan's authenticated CAT12 VBM and aligned ROI labels."""
    root = Path(authority_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ValueError("CAT12 authority root must be an absolute canonical directory")
    entry = manifest["scans"][scan_id]
    record_path = root / entry["record_relative_path"]
    record, _ = _read_json_snapshot(
        record_path,
        expected_sha256=entry["record_file_sha256"],
        label=f"CAT12 scan record {scan_id}",
    )
    if set(record) != {
        "schema",
        "scan_id",
        "pipeline",
        "source_sha256",
        "outputs",
        "record_sha256",
    }:
        raise ValueError(f"CAT12 scan record fields differ for {scan_id}")
    _validate_signed_record(record, label=f"CAT12 scan record {scan_id}")
    if record["record_sha256"] != entry["record_sha256"]:
        raise ValueError(f"CAT12 scan record binding differs for {scan_id}")
    if record["schema"] != CAT12_INPUT_SCHEMA or record["scan_id"] != scan_id:
        raise ValueError(f"CAT12 scan record identity differs for {scan_id}")
    if record["pipeline"] != Connect4Dataset.CAT12_PREPROCESSING_CONTRACT:
        raise ValueError(f"CAT12 scan pipeline differs for {scan_id}")
    if record["source_sha256"] != {
        "t1w": source_t1_sha256,
        "segmentation": source_segmentation_sha256,
    }:
        raise ValueError(f"CAT12 scan sources differ for {scan_id}")

    outputs = record.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "cat12_mwp1",
        "segmentation_in_cat12_vbm",
    }:
        raise ValueError(f"CAT12 scan outputs differ for {scan_id}")
    expected_names = {
        "cat12_mwp1": f"{scan_id}_mwp1_cat12_vbm.nii.gz",
        "segmentation_in_cat12_vbm": f"{scan_id}_synthseg_in_cat12_vbm.nii.gz",
    }
    loaded: dict[str, tuple[nib.Nifti1Image, np.ndarray]] = {}
    for role, filename in expected_names.items():
        identity = outputs[role]
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {
                "relative_path",
                "sha256",
                "size",
                "shape",
                "voxel_size_mm",
            }
            or identity["relative_path"] != f"{scan_id}/{filename}"
            or not _valid_sha256(identity["sha256"])
            or isinstance(identity["size"], bool)
            or not isinstance(identity["size"], int)
            or identity["size"] < 1
            or identity["shape"] != list(SOURCE_SHAPE)
            or identity["voxel_size_mm"] != [1.5, 1.5, 1.5]
        ):
            raise ValueError(f"CAT12 {role} identity differs for {scan_id}")
        loaded[role] = _load_nifti_snapshot(
            root / identity["relative_path"],
            expected_sha256=identity["sha256"],
            expected_size=identity["size"],
            label=f"CAT12 {role} {scan_id}",
        )

    vbm_image, vbm = loaded["cat12_mwp1"]
    label_image, labels = loaded["segmentation_in_cat12_vbm"]
    if tuple(vbm.shape) != SOURCE_SHAPE or tuple(labels.shape) != SOURCE_SHAPE:
        raise ValueError(f"CAT12 arrays have the wrong shape for {scan_id}")
    if not np.allclose(vbm_image.affine, label_image.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"CAT12 VBM/label affines differ for {scan_id}")
    if not np.allclose(vbm_image.header.get_zooms()[:3], (1.5, 1.5, 1.5)):
        raise ValueError(f"CAT12 VBM is not 1.5-mm isotropic for {scan_id}")
    if not np.allclose(label_image.header.get_zooms()[:3], (1.5, 1.5, 1.5)):
        raise ValueError(f"CAT12 labels are not 1.5-mm isotropic for {scan_id}")
    if not np.isfinite(vbm).all() or bool((vbm < 0).any()):
        raise ValueError(f"CAT12 VBM is non-finite or negative for {scan_id}")
    if not np.isfinite(labels).all():
        raise ValueError(f"CAT12 labels are non-finite for {scan_id}")
    rounded = np.rint(labels)
    if not np.allclose(labels, rounded, rtol=0.0, atol=1e-4):
        raise ValueError(f"CAT12 labels are not integers for {scan_id}")
    labels = rounded.astype(np.int32)
    input_identity = {
        "schema": CAT12_INPUT_SCHEMA,
        "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
        "authority_manifest_sha256": manifest_sha256,
        "scan_record_sha256": record["record_sha256"],
        "cat12_mwp1_sha256": outputs["cat12_mwp1"]["sha256"],
        "segmentation_in_cat12_vbm_sha256": outputs["segmentation_in_cat12_vbm"][
            "sha256"
        ],
    }
    return vbm, labels, input_identity


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Hash model parameters/buffers without serializing a temporary file."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def anatcl_roi_volume_input(
    cat12_vbm: np.ndarray,
    segmentation_in_cat12_vbm: np.ndarray,
    label_id: int,
) -> torch.Tensor:
    """Return the exact 3D AnatCL ROI tensor shaped ``[1,1,128,128,128]``."""
    vbm = np.asarray(cat12_vbm, dtype=np.float32)
    labels = np.asarray(segmentation_in_cat12_vbm)
    if vbm.shape != SOURCE_SHAPE or labels.shape != SOURCE_SHAPE:
        raise ValueError(
            "AnatCL requires authenticated CAT12 VBM/labels shaped 121x145x121; "
            "raw/native T1w arrays are forbidden"
        )
    if not np.isfinite(vbm).all() or bool((vbm < 0).any()):
        raise ValueError("CAT12 VBM contains NaN, infinity, or negative values")
    if not np.issubdtype(labels.dtype, np.integer):
        rounded = np.rint(labels)
        if not np.allclose(labels, rounded, rtol=0.0, atol=1e-4):
            raise ValueError("CAT12 ROI labels must be integers")
        labels = rounded.astype(np.int32)
    roi = labels == int(label_id)
    if not bool(roi.any()):
        raise ValueError(f"CAT12 segmentation contains no voxels for ROI {label_id}")
    masked = np.where(roi, vbm, 0.0)
    cropped = masked[:, 8:136, :]
    if cropped.shape != CROPPED_SHAPE:
        raise AssertionError("internal CAT12 center-crop contract drifted")
    padded = np.pad(cropped, ((3, 4), (0, 0), (3, 4)), mode="constant")
    if padded.shape != MODEL_SHAPE:
        raise AssertionError("internal AnatCL padding contract drifted")
    return torch.from_numpy(padded.copy()).float()[None, None]


def extract_anatcl_embedding(
    model: torch.nn.Module,
    cat12_vbm: np.ndarray,
    segmentation_in_cat12_vbm: np.ndarray,
    label_id: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = anatcl_roi_volume_input(cat12_vbm, segmentation_in_cat12_vbm, label_id).to(
        device
    )
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    if isinstance(output, (tuple, list)):
        output = output[0]
    output = torch.as_tensor(output)
    if output.shape != (1, 512):
        raise ValueError(
            f"pretrained 3D AnatCL must return one 512-D descriptor, got {output.shape}"
        )
    embedding = output[0].float().cpu()
    if not torch.isfinite(embedding).all():
        raise ValueError("AnatCL produced NaN or infinity")
    return embedding


def _anatcl_source_identity(package_root: Path) -> dict[str, Any]:
    root = _canonical_existing_file(
        package_root / "__init__.py", label="AnatCL package initializer"
    ).parent
    observed: dict[str, dict[str, object]] = {}
    regular_python_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.is_file()
    )
    if regular_python_files != sorted(ANATCL_SOURCE_FILES_SHA256):
        raise RuntimeError("AnatCL package source inventory differs")
    for relative_path, expected_sha256 in ANATCL_SOURCE_FILES_SHA256.items():
        payload, observed_sha256, size = _snapshot_regular_file(
            root / relative_path,
            label=f"AnatCL source {relative_path}",
            max_bytes=2 * 1024 * 1024,
            require_frozen=False,
        )
        del payload
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"AnatCL source differs: {relative_path}")
        mode = os.lstat(root / relative_path).st_mode & 0o777
        observed[relative_path] = {
            "sha256": observed_sha256,
            "size": size,
            "mode": mode,
        }
    if canonical_sha256(observed) != ANATCL_SOURCE_TREE_SHA256:
        raise RuntimeError("AnatCL source tree identity differs")
    return {
        "source_files_sha256": dict(ANATCL_SOURCE_FILES_SHA256),
        "source_tree_sha256": ANATCL_SOURCE_TREE_SHA256,
    }


def load_pinned_anatcl_model(
    weights_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the exact official fold-0 model after file/source authentication."""
    try:
        from anatcl import AnatCL, __version__ as anatcl_version
        import anatcl as anatcl_package
    except ImportError as exc:
        raise RuntimeError("official AnatCL 0.0.2 is required") from exc
    if str(anatcl_version) != "0.0.2":
        raise RuntimeError("AnatCL package version differs from 0.0.2")
    source_identity = _anatcl_source_identity(Path(anatcl_package.__file__).parent)
    payload, observed, _ = _snapshot_regular_file(
        weights_path,
        label="official AnatCL weights",
        max_bytes=512 * 1024 * 1024,
        require_frozen=True,
    )
    if observed != ANATCL_WEIGHTS_SHA256:
        raise RuntimeError("official AnatCL weights SHA-256 differs")
    # This exact official artifact predates tensor-only serialization and embeds
    # trainer metadata. It is deserialized only after an exact same-descriptor
    # digest match, and only its state-dict mapping is consumed.
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise RuntimeError("official AnatCL checkpoint structure differs")
    model = AnatCL(
        model="resnet18",
        descriptor="global",
        fold=0,
        use_head=False,
        pretrained=False,
    )
    model.backbone.load_state_dict(checkpoint["model"], strict=True)
    if state_dict_sha256(model) != ANATCL_STATE_DICT_SHA256:
        raise RuntimeError("official AnatCL state-dict SHA-256 differs")
    model = model.to(device).eval()
    model_spec = {
        **Connect4Dataset.ANATCL_MODEL_CONTRACT,
        **source_identity,
    }
    if model_spec != Connect4Dataset.ANATCL_MODEL_CONTRACT:
        raise AssertionError("AnatCL model/source contract drifted")
    return model, model_spec


def _parse_pyradiomics_runtime_lock(
    payload: Mapping[str, Any],
    *,
    lock_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema",
        "production_ready",
        "distribution",
        "package_version",
        "upstream_revision",
        "source_distribution",
        "runtime",
        "rejected_distributions",
        "record_sha256",
    }
    if set(payload) != required:
        raise RuntimeError("PyRadiomics runtime lock fields differ")
    _validate_signed_record(payload, label="PyRadiomics runtime lock")
    source = payload.get("source_distribution")
    runtime = payload.get("runtime")
    if (
        payload.get("schema") != PYRADIOMICS_RUNTIME_LOCK_SCHEMA
        or payload.get("distribution") != "pyradiomics"
        or payload.get("package_version") != "3.0.1"
        or payload.get("upstream_revision") != PYRADIOMICS_UPSTREAM_REVISION
        or payload.get("rejected_distributions") != ["pyradiomics-cuda"]
        or not isinstance(source, dict)
        or source.get("filename") != "pyradiomics-3.0.1.tar.gz"
        or source.get("sha256") != PYRADIOMICS_SDIST_SHA256
        or source.get("size") != 34473030
        or not isinstance(runtime, dict)
    ):
        raise RuntimeError("PyRadiomics official source identity differs")
    if payload.get("production_ready") is not True:
        raise RuntimeError(
            "official PyRadiomics runtime is not frozen; pyradiomics-cuda is forbidden"
        )
    if (
        runtime.get("python_abi") != "cp310"
        or runtime.get("platform") != "linux_x86_64"
        or not isinstance(runtime.get("runtime_root_relative_path"), str)
        or not isinstance(runtime.get("site_packages_relative_path"), str)
        or not isinstance(runtime.get("installed_files_manifest_relative_path"), str)
        or not _valid_sha256(runtime.get("runtime_tree_sha256"))
        or not _valid_sha256(runtime.get("installed_files_manifest_sha256"))
    ):
        raise RuntimeError("PyRadiomics frozen runtime identity is incomplete")
    return {
        **Connect4Dataset.PYRADIOMICS_EXTRACTOR_CONTRACT,
        "runtime_lock_sha256": lock_sha256,
        "runtime_tree_sha256": runtime["runtime_tree_sha256"],
    }


def verify_official_pyradiomics_runtime(
    lock_path: Path,
    *,
    expected_lock_sha256: str,
) -> dict[str, Any]:
    global _VERIFIED_PYRADIOMICS_RUNTIME

    if any(
        name == "radiomics" or name.startswith("radiomics.") for name in sys.modules
    ):
        raise RuntimeError(
            "radiomics was imported before its frozen runtime was authenticated"
        )
    payload, lock_sha256 = _read_json_snapshot(
        lock_path,
        expected_sha256=expected_lock_sha256,
        label="official PyRadiomics runtime lock",
    )
    identity = _parse_pyradiomics_runtime_lock(payload, lock_sha256=lock_sha256)
    runtime = payload["runtime"]
    if (
        sys.implementation.cache_tag != "cpython-310"
        or sys.platform != "linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
    ):
        raise RuntimeError("current interpreter ABI/platform differs from runtime lock")
    runtime_root, site_packages, installed_identity = (
        _verify_pyradiomics_installed_files(lock_path, runtime)
    )
    if installed_identity["runtime_tree_sha256"] != identity["runtime_tree_sha256"]:
        raise RuntimeError("PyRadiomics installed runtime digest differs")
    distributions = importlib.metadata.packages_distributions().get("radiomics", [])
    normalised = {
        str(value).strip().lower().replace("_", "-") for value in distributions
    }
    if "pyradiomics-cuda" in normalised or normalised != {"pyradiomics"}:
        raise RuntimeError(
            "radiomics import must resolve exclusively to official pyradiomics"
        )
    try:
        installed_version = importlib.metadata.version("pyradiomics")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("official pyradiomics distribution is missing") from exc
    if installed_version != "3.0.1":
        raise RuntimeError("official pyradiomics version differs from 3.0.1")
    distribution = importlib.metadata.distribution("pyradiomics")
    distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
    if distribution_root != site_packages:
        raise RuntimeError(
            "official pyradiomics metadata resolves outside frozen site-packages"
        )
    spec = importlib.util.find_spec("radiomics")
    if spec is None or spec.origin is None:
        raise RuntimeError("official radiomics import has no concrete source")
    import_origin = Path(spec.origin).resolve(strict=True)
    if runtime_root not in import_origin.parents or site_packages not in (
        import_origin,
        *import_origin.parents,
    ):
        raise RuntimeError("radiomics import resolves outside frozen runtime")
    runtime_relative_origin = import_origin.relative_to(runtime_root).as_posix()
    installed_manifest_path = lock_path.parent.joinpath(
        *_declared_relative_path(
            runtime["installed_files_manifest_relative_path"],
            label="PyRadiomics installed-files manifest",
        ).parts
    )
    manifest, _ = _read_json_snapshot(
        installed_manifest_path,
        expected_sha256=runtime["installed_files_manifest_sha256"],
        label="PyRadiomics installed-files manifest",
    )
    if runtime_relative_origin not in manifest.get("files", {}):
        raise RuntimeError("radiomics import source is absent from frozen inventory")
    identity["installed_files_manifest_sha256"] = installed_identity["manifest_sha256"]
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise RuntimeError("official PyRadiomics is required") from exc
    loaded_radiomics_modules: dict[str, int] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "radiomics" and not name.startswith("radiomics."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"loaded PyRadiomics module has no source: {name}")
        module_path = Path(module_file).resolve(strict=True)
        if runtime_root not in module_path.parents:
            raise RuntimeError(f"loaded PyRadiomics module escapes runtime: {name}")
        module_relative = module_path.relative_to(runtime_root).as_posix()
        if module_relative not in manifest["files"]:
            raise RuntimeError(
                f"loaded PyRadiomics module is absent from inventory: {name}"
            )
        loaded_radiomics_modules[name] = id(module)
    if "radiomics" not in loaded_radiomics_modules or (
        "radiomics.featureextractor" not in loaded_radiomics_modules
    ):
        raise RuntimeError("official PyRadiomics extractor did not load completely")
    _VERIFIED_PYRADIOMICS_RUNTIME = {
        "runtime_root": runtime_root,
        "runtime_lock_sha256": identity["runtime_lock_sha256"],
        "runtime_tree_sha256": identity["runtime_tree_sha256"],
        "installed_files_manifest_sha256": identity["installed_files_manifest_sha256"],
        "featureextractor_module_id": id(featureextractor),
    }
    return identity


def extract_radiomics_rows(
    t1_path: Path,
    mask_path: Path,
    scan_id: str,
    runtime_identity: Mapping[str, Any],
) -> Tuple[list[dict], list[str]]:
    if any(
        runtime_identity.get(key) != value
        for key, value in Connect4Dataset.PYRADIOMICS_EXTRACTOR_CONTRACT.items()
    ):
        raise RuntimeError("PyRadiomics runtime identity differs")
    if (
        not _valid_sha256(runtime_identity.get("runtime_lock_sha256"))
        or not _valid_sha256(runtime_identity.get("runtime_tree_sha256"))
        or not _valid_sha256(runtime_identity.get("installed_files_manifest_sha256"))
    ):
        raise RuntimeError("PyRadiomics frozen runtime lineage is missing")
    verified = _VERIFIED_PYRADIOMICS_RUNTIME
    if not isinstance(verified, dict) or any(
        verified.get(key) != runtime_identity.get(key)
        for key in (
            "runtime_lock_sha256",
            "runtime_tree_sha256",
            "installed_files_manifest_sha256",
        )
    ):
        raise RuntimeError("PyRadiomics runtime was not authenticated before use")
    featureextractor = sys.modules.get("radiomics.featureextractor")
    if featureextractor is None or id(featureextractor) != verified.get(
        "featureextractor_module_id"
    ):
        raise RuntimeError("authenticated PyRadiomics extractor module was replaced")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    rows = []
    feature_names: list[str] | None = None
    for slug, label_id in Connect4Dataset.ROI_SPECS:
        result = extractor.execute(str(t1_path), str(mask_path), label=int(label_id))
        numeric = {
            key: float(value)
            for key, value in result.items()
            if not key.startswith("diagnostics_")
            and np.asarray(value).size == 1
            and np.issubdtype(np.asarray(value).dtype, np.number)
        }
        names = sorted(numeric)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError(f"PyRadiomics feature schema changed at ROI {slug}")
        values = np.asarray([numeric[name] for name in names], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"PyRadiomics produced non-finite values for {scan_id}/{slug}"
            )
        row = {"scan_id": scan_id, "structure_id": label_id, "structure": slug}
        row.update({f"radiomics_{name}": numeric[name] for name in names})
        rows.append(row)
    return rows, [f"radiomics_{name}" for name in (feature_names or [])]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def extract_scan(
    root: Path,
    scan_id: str,
    anatcl_model: torch.nn.Module,
    device: torch.device,
    model_spec: dict,
    cat12_vbm: np.ndarray,
    segmentation_in_cat12_vbm: np.ndarray,
    cat12_input_identity: dict,
    radiomics_runtime_identity: dict,
) -> list[dict]:
    t1_path = root / "T1" / f"{scan_id}_T1.nii.gz"
    mask_path = root / "Masks" / f"{scan_id}_mask.nii.gz"
    t1_image, mask_image = nib.load(str(t1_path)), nib.load(str(mask_path))
    if t1_image.shape != mask_image.shape or not np.allclose(
        t1_image.affine, mask_image.affine, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"{scan_id} T1w and segmentation are not aligned")

    radiomics_rows, _ = extract_radiomics_rows(
        t1_path,
        mask_path,
        scan_id,
        radiomics_runtime_identity,
    )
    output_dir = root / "AnatCL" / scan_id
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_hashes = {}
    for slug, label_id in Connect4Dataset.ROI_SPECS:
        embedding = extract_anatcl_embedding(
            anatcl_model,
            cat12_vbm,
            segmentation_in_cat12_vbm,
            label_id,
            device,
        )
        output_path = output_dir / f"{scan_id}_{slug}.pth"
        torch.save(embedding, output_path)
        embedding_hashes[slug] = sha256_file(output_path)
    provenance = {
        "schema": ROI_FEATURE_SCHEMA,
        "scan_id": scan_id,
        "source_sha256": {
            "t1w": sha256_file(t1_path),
            "segmentation": sha256_file(mask_path),
        },
        "model": dict(model_spec),
        "extraction": dict(Connect4Dataset.ANATCL_EXTRACTION_CONTRACT),
        "cat12_input": dict(cat12_input_identity),
        "embedding_sha256": embedding_hashes,
    }
    _atomic_json(output_dir / "provenance.json", provenance)
    return radiomics_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--anatcl-weights", type=Path, required=True)
    parser.add_argument("--cat12-authority-root", type=Path, required=True)
    parser.add_argument("--cat12-authority-manifest", type=Path, required=True)
    parser.add_argument("--cat12-authority-manifest-sha256", required=True)
    parser.add_argument("--pyradiomics-runtime-lock", type=Path, required=True)
    parser.add_argument("--pyradiomics-runtime-lock-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    scan_ids = sorted(
        path.name.removesuffix("_T1.nii.gz")
        for path in (root / "T1").glob("*_T1.nii.gz")
    )
    if not scan_ids:
        raise RuntimeError(f"no scans found under {root / 'T1'}")

    cat12_manifest, cat12_manifest_sha256 = load_cat12_authority_manifest(
        args.cat12_authority_manifest,
        expected_sha256=args.cat12_authority_manifest_sha256,
        expected_scan_ids=scan_ids,
    )
    radiomics_runtime_identity = verify_official_pyradiomics_runtime(
        args.pyradiomics_runtime_lock,
        expected_lock_sha256=args.pyradiomics_runtime_lock_sha256,
    )
    device = torch.device(args.device)
    model, model_spec = load_pinned_anatcl_model(args.anatcl_weights, device)

    radiomics_rows = []
    for scan_id in scan_ids:
        t1_path = root / "T1" / f"{scan_id}_T1.nii.gz"
        mask_path = root / "Masks" / f"{scan_id}_mask.nii.gz"
        cat12_vbm, cat12_labels, cat12_input_identity = load_cat12_scan_inputs(
            args.cat12_authority_root,
            cat12_manifest,
            cat12_manifest_sha256,
            scan_id=scan_id,
            source_t1_sha256=sha256_file(t1_path),
            source_segmentation_sha256=sha256_file(mask_path),
        )
        radiomics_rows.extend(
            extract_scan(
                root,
                scan_id,
                model,
                device,
                model_spec,
                cat12_vbm,
                cat12_labels,
                cat12_input_identity,
                radiomics_runtime_identity,
            )
        )
    frame = pd.DataFrame(radiomics_rows)
    temporary = root / "all_radiomics.csv.tmp"
    frame.to_csv(temporary, index=False)
    radiomics_path = root / "all_radiomics.csv"
    os.replace(temporary, radiomics_path)
    feature_columns = sorted(
        column for column in frame.columns if column.startswith("radiomics_")
    )
    radiomics_provenance = {
        "schema": RADIOMICS_FEATURE_SCHEMA,
        "extractor": dict(radiomics_runtime_identity),
        "source_sha256": {
            scan_id: {
                "t1w": sha256_file(root / "T1" / f"{scan_id}_T1.nii.gz"),
                "segmentation": sha256_file(root / "Masks" / f"{scan_id}_mask.nii.gz"),
            }
            for scan_id in scan_ids
        },
        "feature_columns": feature_columns,
        "row_count": int(len(frame)),
        "output_sha256": sha256_file(radiomics_path),
    }
    _atomic_json(root / "radiomics_provenance.json", radiomics_provenance)


if __name__ == "__main__":
    main()


__all__ = [
    "ANATCL_SOURCE_FILES_SHA256",
    "ANATCL_SOURCE_TREE_SHA256",
    "ANATCL_STATE_DICT_SHA256",
    "ANATCL_WEIGHTS_SHA256",
    "CAT12_AUTHORITY_SCHEMA",
    "CAT12_INPUT_SCHEMA",
    "PYRADIOMICS_RUNTIME_LOCK_SCHEMA",
    "RADIOMICS_FEATURE_SCHEMA",
    "ROI_FEATURE_SCHEMA",
    "anatcl_roi_volume_input",
    "extract_anatcl_embedding",
    "extract_radiomics_rows",
    "extract_scan",
    "load_cat12_authority_manifest",
    "load_cat12_scan_inputs",
    "load_pinned_anatcl_model",
    "state_dict_sha256",
    "verify_official_pyradiomics_runtime",
]
