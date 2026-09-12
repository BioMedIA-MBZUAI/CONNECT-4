"""Authenticated, context-aware BrainLM perceptual features for CONNECT-4.

This module intentionally does not accept the historical ``vitmae_111M``
artifact.  That public Hugging Face file is a generic image ViT-MAE state
dictionary (it contains ``patch_embeddings.projection`` and no BrainLM
signal/XYZ projections).  The only accepted checkpoint here is the real
``BrainLMForPretraining`` checkpoint committed in the official BrainLM GitHub
repository at the immutable revision below.

CONNECT-4 volumes live on a zero-padded 64x80x64 architecture grid.  Every
BrainLM call therefore receives an explicit batch context: scan IDs and roles,
the padded affine and crop, the exact structural support, and a reviewed
fixed-MNI-to-subject nonlinear pull field.  The same immutable context is used
for prediction and target.  Projection remains differentiable with respect to
the prediction; registration fields and support are structural evidence and
are detached.

BrainLM was trained on 200 samples at a different acquisition cadence and with
population robust-scaler statistics that are not publicly released.  Linear
128-to-200 interpolation and per-scan/per-parcel median-IQR scaling are
versioned recovery adaptations, not claims of pretraining-domain exactness.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.provenance import canonical_sha256
from utils.compat import strict_zip


# ---------------------------------------------------------------------------
# Immutable upstream and recovery identities
# ---------------------------------------------------------------------------
OFFICIAL_BRAINLM_REPOSITORY = "https://github.com/vandijklab/BrainLM"
OFFICIAL_BRAINLM_SOURCE_REVISION = "eded39c86c27e03f5ead1d6a14311e92d1305e5e"
OFFICIAL_BRAINLM_CHECKPOINT_RELATIVE_PATH = (
    "pretrained_models/2023-06-06-22_15_00-checkpoint-1400/pytorch_model.bin"
)
OFFICIAL_BRAINLM_CONFIG_RELATIVE_PATH = (
    "pretrained_models/2023-06-06-22_15_00-checkpoint-1400/config.json"
)
OFFICIAL_BRAINLM_CHECKPOINT_SHA256 = (
    "e647c70b2af023d945bfd61b4187c4fdf1700fbb4433a0818e982520564493c5"
)
OFFICIAL_BRAINLM_CHECKPOINT_SIZE_BYTES = 30_471_369
OFFICIAL_BRAINLM_CONFIG_SHA256 = (
    "03142b047b43e174a1f2d9f64c43396916740abf014f9a404e33487575714091"
)
OFFICIAL_BRAINLM_CONFIG_SIZE_BYTES = 935
OFFICIAL_BRAINLM_SOURCE_FILES = {
    ".git/refs/heads/main": (
        41,
        "6e92c31ef36be318ec16a2a4e43a2950abf3db45f8416c2a497aca9ddb14bfe6",
    ),
    "brainlm_mae/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "brainlm_mae/configuration_brainlm.py": (
        6_345,
        "41a0f6a8a5e42a45d7f8e9daed64d207f9b19afea0d33686bcff1260ccb320e8",
    ),
    "brainlm_mae/modeling_brainlm.py": (
        25_406,
        "8fa49ecc3a569fb9c655ef47d8a7403a655279ab398c318759b4ccaa8271f4f7",
    ),
    "requirements.txt": (
        2_034,
        "bdd89adb7ba8e646c8b77ce23710e08c6d0d01217762894e0b10cf50c5661e0b",
    ),
}
OFFICIAL_BRAINLM_TRANSFORMERS_VERSION = "4.28.0"
OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT = os.environ.get(
    "CONNECT4_BRAINLM_SOURCE_ROOT", ""
)
OFFICIAL_BRAINLM_PUBLICATION_MARKER_SCHEMA = (
    "connect4-native-nfs-no-replace-commit-v2"
)
OFFICIAL_BRAINLM_PUBLICATION_PURPOSE = (
    "connect4-official-brainlm-eded39c86c27-tracked51-clean-v2"
)
OFFICIAL_BRAINLM_PUBLICATION_MARKER_SHA256 = (
    "af79bf37838aeb2f67005deb6750e3ad62eeb8fe65ff5190c0b304b4fdc23912"
)
OFFICIAL_BRAINLM_PUBLICATION_MARKER_SIZE_BYTES = 17_929
OFFICIAL_BRAINLM_PUBLICATION_RECORD_SHA256 = (
    "49a9fe0759ee0615ec37d23ff20ad75a585030a01d91c979d7a2d3d832da5622"
)
OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256 = (
    "0c71b5b431fe22aee6978506c7402925cdf2232abbdd8e23e111aec6ffe383a3"
)
OFFICIAL_BRAINLM_PUBLICATION_DIRECTORIES_SHA256 = (
    "48201ad1db4e34c009c23b10547a2409cf224864c36724e689289a46bc00abec"
)
OFFICIAL_BRAINLM_PUBLICATION_FILE_COUNT = 80
OFFICIAL_BRAINLM_PUBLICATION_WORKING_TREE_FILE_COUNT = 51
OFFICIAL_BRAINLM_PUBLICATION_DIRECTORY_COUNT = 27

# Categorical deny-list: this is the generic image ViT-MAE artifact previously
# mistaken for the 111M BrainLM model.  It must never initialize a perceptual
# loss, even if a caller supplies its exact digest.
REJECTED_GENERIC_VITMAE_CHECKPOINT_SHA256 = frozenset(
    {"25f5b52178f7409dee4fed2fa3dd18c6692d0764d96996ff0ee4be740520fcd6"}
)

OFFICIAL_A424_ATLAS_RELATIVE_PATH = "toolkit/atlases/A424+2mm.nii.gz"
OFFICIAL_A424_COORDINATES_RELATIVE_PATH = "toolkit/atlases/A424_Coordinates.dat"
OFFICIAL_A424_ATLAS_SHA256 = (
    "8478d7516a2bbf295989835d5033d03e4283143a2955c90428f634951702501b"
)
OFFICIAL_A424_ATLAS_SIZE_BYTES = 90_086
OFFICIAL_A424_COORDINATES_SHA256 = (
    "cdad8f8859c211b1400211ebbad455dedcc4399ac14fd614354fb71b110c892c"
)
OFFICIAL_A424_COORDINATES_SIZE_BYTES = 6_236
A424_SHAPE = (91, 109, 91)
A424_AFFINE_RAS_MM = np.asarray(
    [
        [2.0, 0.0, 0.0, -90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256 = (
    "ec46e437f4180b818e517b4593a398c0892f2108931cb87be2f707c54b2f222f"
)
REJECTED_HISTORICAL_NATIVE_PREPROCESSING_SOURCE_SHA256 = frozenset(
    {
        "8beee80eddc3da357229fdbc8ba57b7eab286cbdb34bf2941490f534372e631a",
        "20ae434f0b23fc7f375048b528d239c21d23c95c505c7819d83a11301edfcbee",
        "4ceee73d71391a3da247ed883afde39022ed1f3c4a07d9ed1df6eb830d9d2d3f",
        "bfb6bfd706e8fa19a4832fd870b6e7aa7e71b478cd4d6ca260859392b7a500d6",
    }
)

BRAINLM_CONTEXT_ADAPTER_CONTRACT = "connect4-brainlm-a424-contextual-perceptual-v2"
BRAINLM_AUTHORITY_SCHEMA = "connect4-brainlm-mni-a424-authority-v2"
BRAINLM_AUTHORITY_SCAN_SCHEMA = "connect4-brainlm-mni-a424-scan-v2"
BRAINLM_AUTHORITY_SPLIT_SCOPE_SCHEMA = (
    "connect4-brainlm-authority-split-scope-v1"
)
RECOVERY_BRAINLM_AUTHORIZED_SCAN_COUNT = 2121
BRAINLM_CONTEXT_SCHEMA = "connect4-brainlm-batch-context-v2"
BRAINLM_BATCH_CONTEXT_IDENTITY_SCHEMA = (
    "connect4-brainlm-batch-context-identity-v1"
)
BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA = (
    "connect4-brainlm-admitted-scan-context-v1"
)
BRAINLM_FEATURE_SCHEMA = "connect4-official-brainlm-perceptual-features-v2"
BRAINLM_RUNTIME_MAPPING_SCHEMA = "connect4-a424-padded-grid-pull-field-v2"

PADDED_SHAPE = (64, 80, 64)
NATIVE_SHAPE = (61, 73, 61)
PADDING_BEFORE = (1, 3, 1)
PADDING_AFTER = (2, 4, 2)
INPUT_FRAMES = 128
INPUT_TR_SECONDS = 3.0
BRAINLM_TIMEPOINTS = 200
BRAINLM_TIMEPOINT_PATCH = 20
BRAINLM_PARCELS = 424
BRAINLM_HIDDEN_SIZE = 256
BRAINLM_HIDDEN_LAYERS = 4
MINIMUM_PARCEL_COVERAGE = 0.50
P5_PARCEL_COVERAGE = 0.80
MAXIMUM_DENSE_DISPLACEMENT_MM = 100.0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _is_sha256(digest):
        raise ValueError(f"{label} must be a complete lowercase SHA-256 digest")
    return digest


def _resolve_spec_path(
    spec: Mapping[str, Any],
    key: str,
    *,
    environment_key: str | None = None,
    default: Path | None = None,
) -> Path:
    value = spec.get(key)
    if not value and environment_key:
        environment_name = str(spec.get(environment_key, "")).strip()
        if environment_name:
            value = os.environ.get(environment_name)
    if not value and default is not None:
        return default
    if not value:
        raise FileNotFoundError(f"BrainLM {key} is not configured")
    return Path(str(value)).expanduser()


def _read_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
    maximum_bytes: int | None = None,
) -> tuple[Path, bytes]:
    """Read immutable bytes without following links or accepting path swaps."""

    candidate = path.expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"{label} must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    expected_digest = _require_sha256(expected_sha256, label=f"{label} SHA-256")
    if isinstance(expected_size_bytes, bool) or expected_size_bytes < 0:
        raise ValueError(f"{label} size is invalid")
    limit = expected_size_bytes if maximum_bytes is None else int(maximum_bytes)
    if expected_size_bytes > limit:
        raise RuntimeError(f"{label} exceeds its configured byte limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_size) != expected_size_bytes:
            raise RuntimeError(f"{label} is not the exact expected regular file")
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise RuntimeError(f"{label} exceeds its configured byte limit")
            digest.update(block)
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        total != expected_size_bytes
        or digest.hexdigest() != expected_digest
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise RuntimeError(f"{label} SHA-256/size changed or differs")
    return resolved, b"".join(blocks)


def _read_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None,
    expected_schema: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser()
    if expected_size_bytes is None:
        try:
            size = resolved.lstat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"{label} is unavailable: {resolved}") from exc
    else:
        size = int(expected_size_bytes)
    file_path, payload = _read_regular_file(
        resolved,
        expected_sha256=expected_sha256,
        expected_size_bytes=size,
        label=label,
        maximum_bytes=64 * 1024 * 1024,
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise RuntimeError(f"{label} schema differs")
    unsigned = dict(value)
    claimed = unsigned.pop("record_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} internal record SHA-256 differs")
    return value, {
        "path": str(file_path),
        "sha256": expected_sha256,
        "size_bytes": size,
        "record_sha256": claimed,
    }


def _record_sha256(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("record_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} internal record SHA-256 differs")
    return str(claimed)


def _content_identity(value: Any) -> Any:
    """Remove deployment paths and re-sign records over the remaining content.

    A record hash computed before removing an absolute ``path`` is itself
    transitively path-dependent.  Such hashes must not be retained under a
    path-independent identity.  Every mapping that carried ``record_sha256``
    therefore receives a freshly computed ``content_record_sha256`` instead.
    File SHA-256 values and signed-source record hashes with distinct field
    names remain intact because they identify content, not its deployment path.
    """

    if isinstance(value, Mapping):
        result = {
            str(key): _content_identity(item)
            for key, item in value.items()
            if key not in {"path", "record_sha256"}
        }
        if "record_sha256" in value:
            result["content_record_sha256"] = canonical_sha256(result)
        return result
    if isinstance(value, list):
        return [_content_identity(item) for item in value]
    if isinstance(value, tuple):
        return [_content_identity(item) for item in value]
    return value


def _load_module(name: str, source_path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import authenticated BrainLM source: {source_path}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _validate_config(payload: Mapping[str, Any]) -> None:
    expected = {
        "architectures": ["BrainLMForPretraining"],
        "model_type": "brainlm_mae",
        "hidden_size": BRAINLM_HIDDEN_SIZE,
        "num_hidden_layers": BRAINLM_HIDDEN_LAYERS,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "decoder_hidden_size": 256,
        "decoder_intermediate_size": 1024,
        "decoder_num_attention_heads": 4,
        "decoder_num_hidden_layers": 2,
        "num_brain_voxels": BRAINLM_PARCELS,
        "num_timepoints_per_voxel": BRAINLM_TIMEPOINTS,
        "timepoint_patching_size": BRAINLM_TIMEPOINT_PATCH,
        "mask_ratio": 0.2,
        "loss_fn": "mse",
    }
    mismatches = {
        key: (payload.get(key), expected_value)
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"official BrainLM config identity differs: {mismatches}")


def _validate_state_dict_identity(state: Any) -> Mapping[str, torch.Tensor]:
    """Reject generic ViT-MAE/base weights and require real BrainLM tensors."""

    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("official BrainLM checkpoint is not a state dictionary")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise RuntimeError("official BrainLM state dictionary contains invalid entries")
    if "vit.embeddings.patch_embeddings.projection.weight" in state:
        raise RuntimeError(
            "generic image ViT-MAE weights are categorically invalid as BrainLM"
        )
    expected_shapes = {
        "vit.embeddings.cls_token": (1, 1, 256),
        "vit.embeddings.signal_embedding_projection.weight": (256, 20),
        "vit.embeddings.xyz_embedding_projection.weight": (256, 3),
        "vit.encoder.layer.3.output.dense.weight": (256, 1024),
        "vit.layernorm.weight": (256,),
    }
    for key, shape in expected_shapes.items():
        value = state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise RuntimeError(f"official BrainLM checkpoint tensor differs: {key}")
    if len(state) != 124 or sum(value.numel() for value in state.values()) != 7_607_724:
        raise RuntimeError("official BrainLM checkpoint parameter inventory differs")
    return state


def _load_official_publication_marker(spec: Mapping[str, Any]) -> dict[str, Any]:
    marker_path = _resolve_spec_path(
        spec,
        "source_publication_marker",
        environment_key="source_publication_marker_env",
    )
    configured_sha = _require_sha256(
        spec.get("source_publication_marker_sha256"),
        label="BrainLM source publication marker SHA-256",
    )
    if configured_sha != OFFICIAL_BRAINLM_PUBLICATION_MARKER_SHA256:
        raise RuntimeError("only the immutable official BrainLM publication marker is accepted")
    marker, evidence = _read_bound_json(
        marker_path,
        expected_sha256=configured_sha,
        expected_size_bytes=OFFICIAL_BRAINLM_PUBLICATION_MARKER_SIZE_BYTES,
        expected_schema=OFFICIAL_BRAINLM_PUBLICATION_MARKER_SCHEMA,
        label="official BrainLM immutable publication marker",
    )
    inventory = marker.get("inventory")
    directories = marker.get("directories")
    root_identity = marker.get("root_identity")
    protocol = marker.get("publication_protocol")
    if (
        evidence["record_sha256"] != OFFICIAL_BRAINLM_PUBLICATION_RECORD_SHA256
        or marker.get("status") != "COMMITTED"
        or marker.get("kind") != "directory-tree"
        or marker.get("purpose") != OFFICIAL_BRAINLM_PUBLICATION_PURPOSE
        or marker.get("destination") != OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT
        or marker.get("commit_marker")
        != OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT + ".commit.json"
        or marker.get("root_mode") != 0o500
        or marker.get("source_was_node_local") is not True
        or not isinstance(root_identity, Mapping)
        or set(root_identity) != {"device", "inode"}
        or any(
            isinstance(root_identity.get(key), bool)
            or not isinstance(root_identity.get(key), int)
            or root_identity[key] < 1
            for key in root_identity
        )
        or protocol
        != {
            "commit_marker_published_last": True,
            "descriptor_rooted": True,
            "directory_rename_noreplace_used": False,
            "directory_reservation": "mkdirat-exclusive",
            "failed_reservations_are_never_cleaned_or_reused": True,
            "leaf_publication": "openat-O_CREAT|O_EXCL",
            "overwrite_capable_rename_used": False,
            "tree_rehashed_frozen_and_fsynced_before_marker": True,
        }
        or not isinstance(inventory, list)
        or len(inventory) != OFFICIAL_BRAINLM_PUBLICATION_FILE_COUNT
        or sum(
            not str(entry.get("relative_path", "")).startswith(".git/")
            for entry in inventory
            if isinstance(entry, Mapping)
        )
        != OFFICIAL_BRAINLM_PUBLICATION_WORKING_TREE_FILE_COUNT
        or any(
            "__pycache__" in str(entry.get("relative_path", ""))
            or str(entry.get("relative_path", "")).endswith(".pyc")
            for entry in inventory
            if isinstance(entry, Mapping)
        )
        or marker.get("inventory_sha256")
        != OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256
        or canonical_sha256(inventory)
        != OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256
        or not isinstance(directories, list)
        or len(directories) != OFFICIAL_BRAINLM_PUBLICATION_DIRECTORY_COUNT
        or marker.get("directories_sha256")
        != OFFICIAL_BRAINLM_PUBLICATION_DIRECTORIES_SHA256
        or canonical_sha256(directories)
        != OFFICIAL_BRAINLM_PUBLICATION_DIRECTORIES_SHA256
    ):
        raise RuntimeError("official BrainLM immutable publication marker differs")
    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in inventory:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"mode", "relative_path", "sha256", "size_bytes"}
            or not isinstance(entry.get("relative_path"), str)
            or not entry["relative_path"]
            or entry["relative_path"] in indexed
        ):
            raise RuntimeError("official BrainLM publication inventory differs")
        indexed[str(entry["relative_path"])] = entry
    required_inventory = {
        **{
            relative: {"size_bytes": size, "sha256": digest}
            for relative, (size, digest) in OFFICIAL_BRAINLM_SOURCE_FILES.items()
        },
        OFFICIAL_BRAINLM_CHECKPOINT_RELATIVE_PATH: {
            "size_bytes": OFFICIAL_BRAINLM_CHECKPOINT_SIZE_BYTES,
            "sha256": OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
        },
        OFFICIAL_BRAINLM_CONFIG_RELATIVE_PATH: {
            "size_bytes": OFFICIAL_BRAINLM_CONFIG_SIZE_BYTES,
            "sha256": OFFICIAL_BRAINLM_CONFIG_SHA256,
        },
        OFFICIAL_A424_ATLAS_RELATIVE_PATH: {
            "size_bytes": OFFICIAL_A424_ATLAS_SIZE_BYTES,
            "sha256": OFFICIAL_A424_ATLAS_SHA256,
        },
        OFFICIAL_A424_COORDINATES_RELATIVE_PATH: {
            "size_bytes": OFFICIAL_A424_COORDINATES_SIZE_BYTES,
            "sha256": OFFICIAL_A424_COORDINATES_SHA256,
        },
    }
    for relative, expected in required_inventory.items():
        entry = indexed.get(relative)
        if (
            not isinstance(entry, Mapping)
            or entry.get("mode") != 0o400
            or entry.get("size_bytes") != expected["size_bytes"]
            or entry.get("sha256") != expected["sha256"]
        ):
            raise RuntimeError(
                f"official BrainLM publication inventory entry differs: {relative}"
            )
    return {
        "schema": OFFICIAL_BRAINLM_PUBLICATION_MARKER_SCHEMA,
        "path": evidence["path"],
        "sha256": evidence["sha256"],
        "size_bytes": evidence["size_bytes"],
        "record_sha256": evidence["record_sha256"],
        "inventory_sha256": OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256,
        "directories_sha256": OFFICIAL_BRAINLM_PUBLICATION_DIRECTORIES_SHA256,
        "published_source_root": OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT,
        "purpose": OFFICIAL_BRAINLM_PUBLICATION_PURPOSE,
        "inventory_file_count": OFFICIAL_BRAINLM_PUBLICATION_FILE_COUNT,
        "working_tree_file_count": (
            OFFICIAL_BRAINLM_PUBLICATION_WORKING_TREE_FILE_COUNT
        ),
        "directory_count": OFFICIAL_BRAINLM_PUBLICATION_DIRECTORY_COUNT,
        "pyc_or_pycache_file_count": 0,
        "status": "COMMITTED",
        "selected_runtime_entries_reverified_against_marker": True,
    }


def _load_official_assets(
    spec: Mapping[str, Any],
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    revision = str(spec.get("source_revision", "")).strip().lower()
    if revision != OFFICIAL_BRAINLM_SOURCE_REVISION:
        raise RuntimeError("only the pinned official GitHub BrainLM revision is accepted")
    source_candidate = _resolve_spec_path(
        spec, "source_root", environment_key="source_root_env"
    ).expanduser()
    if not source_candidate.is_absolute() or source_candidate.is_symlink():
        raise RuntimeError(
            "BrainLM source_root must be an absolute, non-symlink directory"
        )
    if source_candidate != Path(OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT):
        raise RuntimeError(
            "BrainLM source_root must equal the exclusive clean tracked51_v2 "
            "publication path"
        )
    source_root = source_candidate.resolve(strict=True)
    if not source_root.is_dir():
        raise RuntimeError("BrainLM source_root must be a real directory")
    publication = _load_official_publication_marker(spec)
    source_evidence: dict[str, dict[str, Any]] = {}
    for relative, (size, digest) in OFFICIAL_BRAINLM_SOURCE_FILES.items():
        path, _ = _read_regular_file(
            source_root / relative,
            expected_sha256=digest,
            expected_size_bytes=size,
            label=f"official BrainLM source {relative}",
            maximum_bytes=2 * 1024 * 1024,
        )
        source_evidence[relative] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": size,
        }
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("official BrainLM requires transformers==4.28.0") from exc
    if transformers.__version__ != OFFICIAL_BRAINLM_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "official BrainLM runtime requires transformers=="
            f"{OFFICIAL_BRAINLM_TRANSFORMERS_VERSION}, found "
            f"{transformers.__version__}"
        )

    checkpoint = _resolve_spec_path(
        spec,
        "checkpoint",
        environment_key="checkpoint_env",
        default=source_root / OFFICIAL_BRAINLM_CHECKPOINT_RELATIVE_PATH,
    )
    configured_checkpoint_sha = _require_sha256(
        spec.get("checkpoint_sha256"), label="BrainLM checkpoint_sha256"
    )
    if configured_checkpoint_sha in REJECTED_GENERIC_VITMAE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "the configured BrainLM checkpoint is the rejected generic image ViT-MAE artifact"
        )
    if configured_checkpoint_sha != OFFICIAL_BRAINLM_CHECKPOINT_SHA256:
        raise RuntimeError("only the official GitHub BrainLM checkpoint is accepted")
    checkpoint_path, _ = _read_regular_file(
        checkpoint,
        expected_sha256=configured_checkpoint_sha,
        expected_size_bytes=OFFICIAL_BRAINLM_CHECKPOINT_SIZE_BYTES,
        label="official BrainLM checkpoint",
        maximum_bytes=64 * 1024 * 1024,
    )

    config_path = _resolve_spec_path(
        spec,
        "config",
        environment_key="config_env",
        default=source_root / OFFICIAL_BRAINLM_CONFIG_RELATIVE_PATH,
    )
    configured_config_sha = _require_sha256(
        spec.get("config_sha256"), label="BrainLM config_sha256"
    )
    if configured_config_sha != OFFICIAL_BRAINLM_CONFIG_SHA256:
        raise RuntimeError("only the official GitHub BrainLM config is accepted")
    config_file, config_bytes = _read_regular_file(
        config_path,
        expected_sha256=configured_config_sha,
        expected_size_bytes=OFFICIAL_BRAINLM_CONFIG_SIZE_BYTES,
        label="official BrainLM config",
        maximum_bytes=64 * 1024,
    )
    try:
        config_payload = json.loads(config_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("official BrainLM config is invalid JSON") from exc
    if not isinstance(config_payload, dict):
        raise RuntimeError("official BrainLM config is not an object")
    _validate_config(config_payload)

    atlas_path = _resolve_spec_path(
        spec,
        "atlas",
        environment_key="atlas_env",
        default=source_root / OFFICIAL_A424_ATLAS_RELATIVE_PATH,
    )
    coordinates_path = _resolve_spec_path(
        spec,
        "coordinates",
        environment_key="coordinates_env",
        default=source_root / OFFICIAL_A424_COORDINATES_RELATIVE_PATH,
    )
    atlas_file, _ = _read_regular_file(
        atlas_path,
        expected_sha256=OFFICIAL_A424_ATLAS_SHA256,
        expected_size_bytes=OFFICIAL_A424_ATLAS_SIZE_BYTES,
        label="official A424+2mm atlas",
        maximum_bytes=2 * 1024 * 1024,
    )
    coordinate_file, _ = _read_regular_file(
        coordinates_path,
        expected_sha256=OFFICIAL_A424_COORDINATES_SHA256,
        expected_size_bytes=OFFICIAL_A424_COORDINATES_SIZE_BYTES,
        label="official A424 coordinates",
        maximum_bytes=64 * 1024,
    )
    atlas = nib.as_closest_canonical(nib.load(str(atlas_file)))
    labels_float = np.asarray(atlas.dataobj, dtype=np.float32)
    if (
        tuple(atlas.shape) != A424_SHAPE
        or not np.allclose(atlas.affine, A424_AFFINE_RAS_MM, rtol=0.0, atol=1e-5)
        or not np.isfinite(labels_float).all()
        or not np.allclose(labels_float, np.rint(labels_float), rtol=0.0, atol=1e-5)
    ):
        raise RuntimeError("official A424 atlas geometry/labels differ")
    labels = np.rint(labels_float).astype(np.int64)
    if sorted(int(value) for value in np.unique(labels) if value > 0) != list(
        range(1, 425)
    ):
        raise RuntimeError("official A424 atlas does not contain labels 1..424")
    parcel_counts = np.bincount(labels.reshape(-1), minlength=425)[1:]
    if parcel_counts.shape != (424,) or int(parcel_counts.min()) < 1:
        raise RuntimeError("official A424 atlas has an empty parcel")
    coordinates = np.loadtxt(coordinate_file, dtype=np.float32)
    if (
        coordinates.shape != (424, 4)
        or not np.isfinite(coordinates).all()
        or not np.array_equal(coordinates[:, 0].astype(np.int64), np.arange(1, 425))
    ):
        raise RuntimeError("official A424 coordinate order/content differs")

    config_module = _load_module(
        "_connect4_official_brainlm_configuration",
        source_root / "brainlm_mae/configuration_brainlm.py",
    )
    model_module = _load_module(
        "_connect4_official_brainlm_modeling",
        source_root / "brainlm_mae/modeling_brainlm.py",
    )
    configuration = config_module.BrainLMConfig(**config_payload)
    model = model_module.BrainLMForPretraining(configuration)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "BrainLM loading requires torch.load(weights_only=True); unsafe pickle is forbidden"
        ) from exc
    state = _validate_state_dict_identity(state)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("official BrainLM state dictionary did not load strictly")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    identity = {
        "schema": "connect4-official-github-brainlm-artifacts-v1",
        "repository": OFFICIAL_BRAINLM_REPOSITORY,
        "source_revision": OFFICIAL_BRAINLM_SOURCE_REVISION,
        "immutable_publication": publication,
        "source_files": source_evidence,
        "runtime_dependencies": {
            "transformers_version": OFFICIAL_BRAINLM_TRANSFORMERS_VERSION,
            "matches_official_requirements": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "relative_path": OFFICIAL_BRAINLM_CHECKPOINT_RELATIVE_PATH,
            "sha256": OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
            "size_bytes": OFFICIAL_BRAINLM_CHECKPOINT_SIZE_BYTES,
            "state_dict_tensor_count": 124,
            "parameter_count": 7_607_724,
            "has_signal_embedding_projection": True,
            "has_xyz_embedding_projection": True,
            "has_image_patch_projection": False,
        },
        "config": {
            "path": str(config_file),
            "relative_path": OFFICIAL_BRAINLM_CONFIG_RELATIVE_PATH,
            "sha256": OFFICIAL_BRAINLM_CONFIG_SHA256,
            "size_bytes": OFFICIAL_BRAINLM_CONFIG_SIZE_BYTES,
            "architecture": "BrainLMForPretraining",
            "hidden_layers": BRAINLM_HIDDEN_LAYERS,
            "hidden_size": BRAINLM_HIDDEN_SIZE,
            "num_parcels": BRAINLM_PARCELS,
            "num_timepoints": BRAINLM_TIMEPOINTS,
            "timepoint_patch": BRAINLM_TIMEPOINT_PATCH,
        },
        "a424_atlas": {
            "path": str(atlas_file),
            "sha256": OFFICIAL_A424_ATLAS_SHA256,
            "size_bytes": OFFICIAL_A424_ATLAS_SIZE_BYTES,
            "shape": list(A424_SHAPE),
            "affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        },
        "a424_coordinates": {
            "path": str(coordinate_file),
            "sha256": OFFICIAL_A424_COORDINATES_SHA256,
            "size_bytes": OFFICIAL_A424_COORDINATES_SIZE_BYTES,
            "order": "rows 1..424, identical to atlas label/parcel-series order",
        },
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return (
        model,
        torch.from_numpy(labels),
        torch.from_numpy(np.ascontiguousarray(coordinates[:, 1:4])),
        identity,
    )


def _expected_mapping_contract() -> dict[str, Any]:
    return {
        "schema": BRAINLM_RUNTIME_MAPPING_SCHEMA,
        "mapping_kind": "dense_displacement_pull_field",
        "domain": "official A424 canonical RAS grid",
        "domain_shape": list(A424_SHAPE),
        "domain_affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        "domain_atlas_sha256": OFFICIAL_A424_ATLAS_SHA256,
        "range": "current ec46 native prepared structural RAS world millimetres",
        "range_native_shape": list(NATIVE_SHAPE),
        "range_architecture_shape": list(PADDED_SHAPE),
        "padding_before": list(PADDING_BEFORE),
        "padding_after": list(PADDING_AFTER),
        "component_order": ["right", "anterior", "superior"],
        "vector_units": "millimetres",
        "equation": (
            "prepared_source_world_ras = a424_mni_world_ras + "
            "dense_displacement_ras_mm[a424_voxel]"
        ),
        "source_sampling": "trilinear, zero padding, align_corners=True",
        "support_sampling": "trilinear, clamped [0,1], zero padding",
        "inverse_field_applied": False,
        "affine_only_fallback_used": False,
    }


def _resolve_authority_child(authority_path: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"BrainLM authority is missing {label} path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = authority_path.parent / candidate
    return candidate


def load_brainlm_authority(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    authority, evidence = _read_bound_json(
        path,
        expected_sha256=_require_sha256(
            expected_sha256, label="BrainLM authority SHA-256"
        ),
        expected_size_bytes=None,
        expected_schema=BRAINLM_AUTHORITY_SCHEMA,
        label="BrainLM reviewed MNI authority",
    )
    source_sha = str(authority.get("native_preprocessing_source_sha256", ""))
    if source_sha in REJECTED_HISTORICAL_NATIVE_PREPROCESSING_SOURCE_SHA256:
        raise RuntimeError(
            "BrainLM authority uses a categorically rejected historical native source"
        )
    official = authority.get("official_brainlm_artifacts")
    grid = authority.get("source_grid")
    mapping = authority.get("runtime_mapping_contract")
    scans = authority.get("scans")
    ordered_ids = authority.get("ordered_scan_ids")
    if (
        source_sha != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        or authority.get("status")
        != "AUTHORIZED_AFTER_EXPLICIT_HUMAN_STRUCTURAL_REVIEW"
        or authority.get("structural_only") is not True
        or authority.get("functional_target_used_for_registration") is not False
        or authority.get("sealed_target_voxel_data_opened") is not False
        or authority.get("authorized_roles") != ["train", "development-validation"]
        or authority.get("authorizes_prediction_emission") is not False
        or authority.get("authorizes_final_evaluation") is not False
        or official
        != {
            "repository": OFFICIAL_BRAINLM_REPOSITORY,
            "source_revision": OFFICIAL_BRAINLM_SOURCE_REVISION,
            "checkpoint_sha256": OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
            "config_sha256": OFFICIAL_BRAINLM_CONFIG_SHA256,
            "a424_atlas_sha256": OFFICIAL_A424_ATLAS_SHA256,
            "a424_coordinates_sha256": OFFICIAL_A424_COORDINATES_SHA256,
        }
        or grid
        != {
            "architecture_shape": list(PADDED_SHAPE),
            "native_shape": list(NATIVE_SHAPE),
            "padding_before": list(PADDING_BEFORE),
            "padding_after": list(PADDING_AFTER),
            "frames": INPUT_FRAMES,
            "tr_seconds": INPUT_TR_SECONDS,
        }
        or mapping != _expected_mapping_contract()
        or not isinstance(scans, list)
        or not scans
        or not isinstance(ordered_ids, list)
        or len(scans) != len(ordered_ids)
        or authority.get("scan_count") != len(scans)
        or authority.get("ordered_scan_ids_sha256") != canonical_sha256(ordered_ids)
    ):
        raise RuntimeError("BrainLM reviewed MNI authority top-level contract differs")

    records: dict[str, dict[str, Any]] = {}
    for expected_id, raw_record in strict_zip(ordered_ids, scans):
        if not isinstance(raw_record, dict):
            raise RuntimeError("BrainLM authority scan record is not an object")
        scan_id = str(raw_record.get("scan_id", ""))
        _record_sha256(raw_record, label=f"BrainLM authority scan {scan_id}")
        role = raw_record.get("role")
        human = raw_record.get("human_review")
        artifact = raw_record.get("a424_dense_displacement_artifact")
        support_count = raw_record.get("support_foreground_voxels")
        padded_affine = np.asarray(raw_record.get("padded_affine_ras_mm"), dtype=np.float64)
        native_affine = np.asarray(raw_record.get("native_affine_ras_mm"), dtype=np.float64)
        if (
            scan_id != expected_id
            or not scan_id
            or scan_id in records
            or raw_record.get("schema") != BRAINLM_AUTHORITY_SCAN_SCHEMA
            or role not in {"train", "development-validation"}
            or raw_record.get("native_preprocessing_source_sha256") != source_sha
            or not _is_sha256(raw_record.get("prepared_t1_sha256"))
            or not _is_sha256(raw_record.get("prepared_mask_sha256"))
            or not _is_sha256(raw_record.get("support_tensor_sha256"))
            or isinstance(support_count, bool)
            or not isinstance(support_count, int)
            or not 1 <= support_count <= int(np.prod(PADDED_SHAPE))
            or raw_record.get("padded_shape") != list(PADDED_SHAPE)
            or raw_record.get("native_shape") != list(NATIVE_SHAPE)
            or raw_record.get("padding_before") != list(PADDING_BEFORE)
            or raw_record.get("padding_after") != list(PADDING_AFTER)
            or padded_affine.shape != (4, 4)
            or native_affine.shape != (4, 4)
            or not np.isfinite(padded_affine).all()
            or not np.isfinite(native_affine).all()
            or not np.allclose(
                padded_affine[:3, :3], native_affine[:3, :3], rtol=0.0, atol=1e-6
            )
            or not np.allclose(
                padded_affine[:3, 3],
                native_affine[:3, 3]
                - native_affine[:3, :3] @ np.asarray(PADDING_BEFORE),
                rtol=0.0,
                atol=1e-6,
            )
            or not isinstance(human, dict)
            or human.get("decision") != "PASS"
            or human.get("structural_only") is not True
            or human.get("prediction_or_functional_target_used") is not False
            or not _is_sha256(human.get("review_record_sha256"))
            or not isinstance(artifact, dict)
            or artifact.get("sha256") != human.get("reviewed_displacement_sha256")
            or artifact.get("runtime_mapping_schema")
            != BRAINLM_RUNTIME_MAPPING_SCHEMA
        ):
            raise RuntimeError(f"BrainLM authority scan contract differs: {scan_id}")
        records[scan_id] = dict(raw_record)

    authority_identity = {
        "schema": BRAINLM_AUTHORITY_SCHEMA,
        "path": evidence["path"],
        "file_sha256": evidence["sha256"],
        "size_bytes": evidence["size_bytes"],
        "authority_record_sha256": evidence["record_sha256"],
        "native_preprocessing_source_sha256": source_sha,
        "ordered_scan_ids": list(ordered_ids),
        "ordered_scan_ids_sha256": canonical_sha256(ordered_ids),
        "scan_count": len(records),
        "all_scans_explicitly_structurally_reviewed": True,
        "dense_nonlinear_pull_field_required": True,
        "affine_only_fallback_forbidden": True,
    }
    authority_identity["record_sha256"] = canonical_sha256(authority_identity)
    return records, authority_identity


def _load_dense_displacement(
    authority_path: Path,
    artifact: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    required = {
        "path",
        "sha256",
        "size_bytes",
        "shape",
        "affine_ras_mm",
        "dtype",
        "nifti_intent",
        "component_order",
        "runtime_mapping_schema",
    }
    if set(artifact) != required:
        raise RuntimeError("BrainLM dense displacement artifact fields differ")
    path = _resolve_authority_child(
        authority_path, artifact.get("path"), label="dense displacement"
    )
    expected_size = artifact.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or expected_size > 128 * 1024 * 1024
        or artifact.get("shape") != [*A424_SHAPE, 3]
        or artifact.get("affine_ras_mm") != A424_AFFINE_RAS_MM.tolist()
        or artifact.get("dtype") != "float32"
        or artifact.get("nifti_intent") != "vector"
        or artifact.get("component_order") != ["right", "anterior", "superior"]
        or artifact.get("runtime_mapping_schema") != BRAINLM_RUNTIME_MAPPING_SCHEMA
    ):
        raise RuntimeError("BrainLM dense displacement artifact metadata differs")
    displacement_path, payload = _read_regular_file(
        path,
        expected_sha256=_require_sha256(
            artifact.get("sha256"), label="dense displacement SHA-256"
        ),
        expected_size_bytes=expected_size,
        label="BrainLM dense displacement",
        maximum_bytes=128 * 1024 * 1024,
    )
    try:
        if payload[:2] == b"\x1f\x8b":
            output = io.BytesIO()
            with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    if output.tell() + len(block) > 16 * 1024 * 1024:
                        raise RuntimeError(
                            "BrainLM dense displacement decompressed bytes exceed limit"
                        )
                    output.write(block)
            nifti_bytes = output.getvalue()
        else:
            if len(payload) > 16 * 1024 * 1024:
                raise RuntimeError(
                    "BrainLM dense displacement uncompressed bytes exceed limit"
                )
            nifti_bytes = payload
        image = nib.Nifti1Image.from_bytes(nifti_bytes)
    except Exception as exc:
        raise RuntimeError(
            "BrainLM dense displacement authenticated bytes are not a NIfTI"
        ) from exc
    if (
        tuple(image.shape) != (*A424_SHAPE, 3)
        or not np.allclose(image.affine, A424_AFFINE_RAS_MM, rtol=0.0, atol=1e-5)
        or image.header.get_intent()[0] != "vector"
    ):
        raise RuntimeError("BrainLM dense displacement NIfTI geometry differs")
    values = np.asarray(image.dataobj, dtype=np.float32)
    magnitudes = np.linalg.norm(values.astype(np.float64), axis=-1)
    if (
        values.shape != (*A424_SHAPE, 3)
        or not np.isfinite(values).all()
        or float(magnitudes.max()) > MAXIMUM_DENSE_DISPLACEMENT_MM
    ):
        raise RuntimeError("BrainLM dense displacement values are invalid/unbounded")
    evidence = {
        "path": str(displacement_path),
        "sha256": artifact["sha256"],
        "size_bytes": expected_size,
        "maximum_displacement_mm": float(magnitudes.max()),
        "inverse_field_applied": False,
        "affine_only_fallback_used": False,
    }
    return torch.from_numpy(np.ascontiguousarray(values)), evidence


def _support_sha256(support: torch.Tensor) -> str:
    values = (support.detach().to(device="cpu") > 0.5).to(torch.uint8).contiguous()
    return hashlib.sha256(values.numpy().tobytes(order="C")).hexdigest()


def _sequence(value: Any, *, batch: int, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        result = [value]
    elif torch.is_tensor(value):
        result = value.detach().cpu().tolist()
    elif isinstance(value, Sequence):
        result = list(value)
    else:
        raise RuntimeError(f"BrainLM context {label} is not a batch sequence")
    if len(result) != batch:
        raise RuntimeError(f"BrainLM context {label} batch length differs")
    return result


def validate_brainlm_batch_context_identity(
    value: Mapping[str, Any],
    *,
    expected_brainlm_identity_sha256: str | None = None,
    expected_dataset_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete path-independent identity used by one loss call."""

    fields = {
        "schema",
        "brainlm_perceptual_identity_sha256",
        "authority_file_sha256",
        "authority_content_record_sha256",
        "authority_record_sha256",
        "native_preprocessing_source_sha256",
        "run_artifact_identity_sha256",
        "cohort_admission_identity_record_sha256",
        "dataset_state_record_sha256",
        "dataset_state_sha256",
        "structural_artifact_identities_sha256",
        "scan_ids",
        "roles",
        "prepared_t1_sha256",
        "prepared_mask_sha256",
        "padded_t1_artifact_descriptor_sha256",
        "padded_mask_artifact_descriptor_sha256",
        "cache_metadata_artifact_descriptor_sha256",
        "structural_source_identity_sha256",
        "native_alignment_authority_sha256",
        "target_artifact_identity_sha256",
        "support_tensor_sha256",
        "support_foreground_voxels",
        "admitted_scan_context_record_sha256",
        "authority_scan_record_sha256",
        "displacement_sha256",
        "displacement_size_bytes",
        "padded_shape",
        "native_shape",
        "padding_before",
        "padding_after",
        "padded_affine_ras_mm",
        "runtime_mapping_contract",
        "prediction_target_context_shared",
        "record_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("BrainLM batch-context identity fields differ")
    identity = dict(value)
    unsigned = dict(identity)
    recorded = unsigned.pop("record_sha256", None)
    scalar_hashes = (
        "brainlm_perceptual_identity_sha256",
        "authority_file_sha256",
        "authority_content_record_sha256",
        "authority_record_sha256",
        "native_preprocessing_source_sha256",
        "run_artifact_identity_sha256",
        "cohort_admission_identity_record_sha256",
        "dataset_state_record_sha256",
        "dataset_state_sha256",
        "structural_artifact_identities_sha256",
    )
    if (
        identity.get("schema") != BRAINLM_BATCH_CONTEXT_IDENTITY_SCHEMA
        or identity.get("prediction_target_context_shared") is not True
        or identity.get("native_preprocessing_source_sha256")
        != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        or identity.get("runtime_mapping_contract")
        != _expected_mapping_contract()
        or not _is_sha256(recorded)
        or recorded != canonical_sha256(unsigned)
        or any(not _is_sha256(identity.get(field)) for field in scalar_hashes)
    ):
        raise RuntimeError("BrainLM batch-context identity root differs")
    if (
        expected_brainlm_identity_sha256 is not None
        and identity["brainlm_perceptual_identity_sha256"]
        != _require_sha256(
            expected_brainlm_identity_sha256,
            label="expected BrainLM perceptual identity",
        )
    ):
        raise RuntimeError("BrainLM batch-context perceptual identity differs")
    if (
        expected_dataset_state_sha256 is not None
        and identity["dataset_state_sha256"]
        != _require_sha256(
            expected_dataset_state_sha256,
            label="expected dataset-state identity",
        )
    ):
        raise RuntimeError("BrainLM batch-context dataset-state root differs")
    scan_ids = identity.get("scan_ids")
    if (
        not isinstance(scan_ids, list)
        or not scan_ids
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in scan_ids)
    ):
        raise RuntimeError("BrainLM batch-context scan identities differ")
    batch = len(scan_ids)
    sequence_fields = (
        "roles",
        "prepared_t1_sha256",
        "prepared_mask_sha256",
        "padded_t1_artifact_descriptor_sha256",
        "padded_mask_artifact_descriptor_sha256",
        "cache_metadata_artifact_descriptor_sha256",
        "structural_source_identity_sha256",
        "native_alignment_authority_sha256",
        "target_artifact_identity_sha256",
        "support_tensor_sha256",
        "support_foreground_voxels",
        "admitted_scan_context_record_sha256",
        "authority_scan_record_sha256",
        "displacement_sha256",
        "displacement_size_bytes",
        "padded_affine_ras_mm",
    )
    if any(
        not isinstance(identity.get(field), list)
        or len(identity[field]) != batch
        for field in sequence_fields
    ):
        raise RuntimeError("BrainLM batch-context per-scan sequence differs")
    hash_sequences = set(sequence_fields) - {
        "roles",
        "support_foreground_voxels",
        "displacement_size_bytes",
        "padded_affine_ras_mm",
    }
    if (
        any(
            role not in {"train", "development-validation"}
            for role in identity["roles"]
        )
        or any(
            not _is_sha256(item)
            for field in hash_sequences
            for item in identity[field]
        )
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for field in ("support_foreground_voxels", "displacement_size_bytes")
            for item in identity[field]
        )
        or identity.get("padded_shape") != list(PADDED_SHAPE)
        or identity.get("native_shape") != list(NATIVE_SHAPE)
        or identity.get("padding_before") != list(PADDING_BEFORE)
        or identity.get("padding_after") != list(PADDING_AFTER)
    ):
        raise RuntimeError("BrainLM batch-context per-scan contract differs")
    for affine in identity["padded_affine_ras_mm"]:
        array = np.asarray(affine, dtype=np.float64)
        if array.shape != (4, 4) or not np.isfinite(array).all():
            raise RuntimeError("BrainLM batch-context affine differs")
    return identity


class OfficialBrainLMA424Extractor(nn.Module):
    """Frozen official BrainLM with differentiable reviewed A424 projection."""

    connect4_model_name = "brainlm"
    connect4_adapter_contract = BRAINLM_CONTEXT_ADAPTER_CONTRACT
    connect4_source_revision = OFFICIAL_BRAINLM_SOURCE_REVISION
    connect4_requires_context = True

    def __init__(
        self,
        model: nn.Module,
        atlas_labels: torch.Tensor,
        coordinates_xyz: torch.Tensor,
        *,
        authority_records: Mapping[str, Mapping[str, Any]],
        artifact_identity: Mapping[str, Any],
        authority_identity: Mapping[str, Any],
        resample_chunk_frames: int = 4,
    ) -> None:
        super().__init__()
        if tuple(atlas_labels.shape) != A424_SHAPE:
            raise ValueError("A424 atlas labels have the wrong shape")
        if tuple(coordinates_xyz.shape) != (424, 3):
            raise ValueError("A424 coordinates have the wrong shape")
        if resample_chunk_frames < 1:
            raise ValueError("BrainLM resample_chunk_frames must be positive")
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("atlas_labels", atlas_labels.long(), persistent=True)
        self.register_buffer(
            "coordinates_xyz", coordinates_xyz.float(), persistent=True
        )
        counts = torch.bincount(atlas_labels.reshape(-1).long(), minlength=425)[1:]
        if tuple(counts.shape) != (424,) or bool((counts <= 0).any().item()):
            raise RuntimeError("A424 atlas has an empty official parcel")
        self.register_buffer("parcel_counts", counts.float(), persistent=True)
        self.authority_records = {
            str(scan_id): dict(record)
            for scan_id, record in authority_records.items()
        }
        self.artifact_identity = dict(artifact_identity)
        self.authority_identity = dict(authority_identity)
        self.resample_chunk_frames = int(resample_chunk_frames)
        self.last_projection_qc: dict[str, Any] | None = None
        self.last_context_sha256: str | None = None
        self.last_context_identity: dict[str, Any] | None = None
        self.last_preflight_context_identity: dict[str, Any] | None = None
        self.last_input_role: str | None = None

        # Stable pseudo-random mask order, identical for prediction and target.
        token_count = BRAINLM_PARCELS * (BRAINLM_TIMEPOINTS // BRAINLM_TIMEPOINT_PATCH)
        mask_noise = np.asarray(
            [
                int.from_bytes(
                    hashlib.sha256(
                        f"connect4-brainlm-mask-v1:{index}".encode("ascii")
                    ).digest()[:8],
                    "big",
                )
                / float(2**64)
                for index in range(token_count)
            ],
            dtype=np.float32,
        )
        self.register_buffer(
            "deterministic_mask_noise",
            torch.from_numpy(mask_noise).view(1, token_count),
            persistent=True,
        )

        identity = {
            "schema": "connect4-brainlm-a424-perceptual-identity-v2",
            "name": "brainlm",
            "model_artifacts": _content_identity(self.artifact_identity),
            "mni_authority": _content_identity(self.authority_identity),
            "adapter_contract": BRAINLM_CONTEXT_ADAPTER_CONTRACT,
            "feature_contract": BRAINLM_FEATURE_SCHEMA,
            "feature_layers": "last four encoder hidden states, CLS token, layer-normalized",
            "deterministic_identical_prediction_target_mask": True,
            "input_frames": INPUT_FRAMES,
            "input_tr_seconds": INPUT_TR_SECONDS,
            "temporal_adapter": "linear 128 to 200 samples, align_corners=True",
            "scaler_adapter": "per-scan/per-parcel median-IQR then clamp [-6,6]",
            "normalization_and_tr_match_pretraining_exactly": False,
            "pretraining_domain_exactness_claimed": False,
            "domain_shift_reason": (
                "TR=3 s differs from BrainLM training acquisitions and public "
                "population parcel median/IQR vectors are unavailable"
            ),
            "projection": _expected_mapping_contract(),
        }
        identity["record_sha256"] = canonical_sha256(identity)
        self.connect4_artifact_identity = identity
        self.connect4_artifact_identity_sha256 = identity["record_sha256"]

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _physical_grid(
        padded_affine: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(padded_affine.shape[0])
        device = padded_affine.device
        ii, jj, kk = torch.meshgrid(
            torch.arange(A424_SHAPE[0], device=device, dtype=torch.float32),
            torch.arange(A424_SHAPE[1], device=device, dtype=torch.float32),
            torch.arange(A424_SHAPE[2], device=device, dtype=torch.float32),
            indexing="ij",
        )
        atlas_voxels = torch.stack(
            (ii, jj, kk, torch.ones_like(ii)), dim=-1
        ).reshape(-1, 4)
        atlas_affine = torch.as_tensor(
            A424_AFFINE_RAS_MM, device=device, dtype=torch.float32
        )
        mni_world = atlas_voxels @ atlas_affine.T
        source_xyz = mni_world[None, :, :3] + displacement.to(
            device=device, dtype=torch.float32
        ).reshape(batch, -1, 3)
        source_world = torch.cat(
            (
                source_xyz,
                torch.ones(batch, source_xyz.shape[1], 1, device=device),
            ),
            dim=-1,
        )
        source_voxel = torch.einsum(
            "bij,bnj->bni", torch.linalg.inv(padded_affine.float()), source_world
        )[..., :3]
        x = 2.0 * source_voxel[..., 2] / (PADDED_SHAPE[2] - 1) - 1.0
        y = 2.0 * source_voxel[..., 1] / (PADDED_SHAPE[1] - 1) - 1.0
        z = 2.0 * source_voxel[..., 0] / (PADDED_SHAPE[0] - 1) - 1.0
        return torch.stack((x, y, z), dim=-1).reshape(batch, *A424_SHAPE, 3)

    def _prepare_context(
        self,
        context: Mapping[str, Any],
        *,
        device: torch.device,
        batch: int,
    ) -> dict[str, Any]:
        if not isinstance(context, Mapping) or context.get("schema") != BRAINLM_CONTEXT_SCHEMA:
            raise RuntimeError("BrainLM batch context is missing or has the wrong schema")
        if context.get("authority_file_sha256") != self.authority_identity.get(
            "file_sha256"
        ):
            raise RuntimeError("BrainLM batch context uses another MNI authority")
        authority_content_sha = self.connect4_artifact_identity.get(
            "mni_authority", {}
        ).get("content_record_sha256")
        required_roots = {
            "brainlm_perceptual_identity_sha256": (
                self.connect4_artifact_identity_sha256
            ),
            "authority_content_record_sha256": authority_content_sha,
            "authority_record_sha256": self.authority_identity.get(
                "authority_record_sha256"
            ),
        }
        if any(
            not _is_sha256(context.get(field))
            or context.get(field) != expected
            for field, expected in required_roots.items()
        ):
            raise RuntimeError("BrainLM batch context authority/model root differs")
        for field in (
            "run_artifact_identity_sha256",
            "cohort_admission_identity_record_sha256",
            "dataset_state_record_sha256",
            "dataset_state_sha256",
            "structural_artifact_identities_sha256",
        ):
            if not _is_sha256(context.get(field)):
                raise RuntimeError(f"BrainLM batch context {field} is invalid")
        if context.get("native_preprocessing_source_sha256") in (
            REJECTED_HISTORICAL_NATIVE_PREPROCESSING_SOURCE_SHA256
        ):
            raise RuntimeError("BrainLM batch context uses a historical native source")
        if (
            context.get("native_preprocessing_source_sha256")
            != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ):
            raise RuntimeError("BrainLM batch context is not bound to current ec46 source")

        scan_ids = [str(value) for value in _sequence(
            context.get("scan_ids"), batch=batch, label="scan_ids"
        )]
        roles = [str(value) for value in _sequence(
            context.get("roles"), batch=batch, label="roles"
        )]
        t1_hashes = [str(value) for value in _sequence(
            context.get("prepared_t1_sha256"), batch=batch, label="prepared_t1_sha256"
        )]
        mask_hashes = [str(value) for value in _sequence(
            context.get("prepared_mask_sha256"), batch=batch, label="prepared_mask_sha256"
        )]
        sequence_names = (
            "padded_t1_artifact_descriptor_sha256",
            "padded_mask_artifact_descriptor_sha256",
            "cache_metadata_artifact_descriptor_sha256",
            "structural_source_identity_sha256",
            "native_alignment_authority_sha256",
            "target_artifact_identity_sha256",
            "support_tensor_sha256",
            "support_foreground_voxels",
            "admitted_scan_context_record_sha256",
            "admitted_scan_context_schema",
        )
        batch_values = {
            name: _sequence(context.get(name), batch=batch, label=name)
            for name in sequence_names
        }
        source_affine = context.get("source_affine")
        admitted_affine = context.get("admitted_padded_affine")
        brain_support = context.get("brain_support")
        padded_shape = context.get("padded_shape")
        crop_before = context.get("crop_before")
        crop_after = context.get("crop_after")
        native_shape = context.get("native_shape")
        if not torch.is_tensor(source_affine) or tuple(source_affine.shape) != (batch, 4, 4):
            raise RuntimeError("BrainLM source_affine must be [B,4,4]")
        if not torch.is_tensor(admitted_affine) or tuple(admitted_affine.shape) != (
            batch,
            4,
            4,
        ):
            raise RuntimeError("BrainLM admitted_padded_affine must be [B,4,4]")
        if not torch.allclose(
            source_affine.detach().cpu().double(),
            admitted_affine.detach().cpu().double(),
            rtol=0.0,
            atol=1e-5,
        ):
            raise RuntimeError("BrainLM source/admitted padded affine differs")
        if not torch.is_tensor(brain_support):
            raise RuntimeError("BrainLM brain_support tensor is missing")
        if brain_support.ndim == 5 and brain_support.shape[1] == 1:
            support = brain_support[:, 0]
        elif brain_support.ndim == 4:
            support = brain_support
        else:
            raise RuntimeError("BrainLM brain_support must be [B,1,64,80,64]")
        if tuple(support.shape) != (batch, *PADDED_SHAPE):
            raise RuntimeError("BrainLM brain_support grid differs")
        if not torch.isfinite(support).all() or bool(
            ((support != 0) & (support != 1)).any().item()
        ):
            raise RuntimeError("BrainLM brain_support must be a finite binary mask")

        padded_rows = _sequence(padded_shape, batch=batch, label="padded_shape")
        before_rows = _sequence(crop_before, batch=batch, label="crop_before")
        after_rows = _sequence(crop_after, batch=batch, label="crop_after")
        native_rows = _sequence(native_shape, batch=batch, label="native_shape")
        displacements: list[torch.Tensor] = []
        displacement_evidence: list[dict[str, Any]] = []
        scan_record_hashes: list[str] = []
        admitted_record_hashes: list[str] = []
        observed_support_hashes: list[str] = []
        observed_support_counts: list[int] = []
        authority_path = Path(str(self.authority_identity["path"]))
        for index, scan_id in enumerate(scan_ids):
            record = self.authority_records.get(scan_id)
            if record is None:
                raise RuntimeError(f"BrainLM scan is absent from authority: {scan_id}")
            role = roles[index]
            if role not in {"train", "development-validation"} or record.get("role") != role:
                raise RuntimeError(f"BrainLM batch role differs from authority: {scan_id}")
            if (
                not _is_sha256(t1_hashes[index])
                or not _is_sha256(mask_hashes[index])
                or record.get("prepared_t1_sha256") != t1_hashes[index]
                or record.get("prepared_mask_sha256") != mask_hashes[index]
            ):
                raise RuntimeError(f"BrainLM prepared structural hashes differ: {scan_id}")
            if list(before_rows[index]) != list(PADDING_BEFORE):
                raise RuntimeError(f"BrainLM crop-before differs: {scan_id}")
            if list(after_rows[index]) != list(PADDING_AFTER):
                raise RuntimeError(f"BrainLM crop-after differs: {scan_id}")
            if list(native_rows[index]) != list(NATIVE_SHAPE):
                raise RuntimeError(f"BrainLM native shape differs: {scan_id}")
            if list(padded_rows[index]) != list(PADDED_SHAPE):
                raise RuntimeError(f"BrainLM padded shape differs: {scan_id}")
            expected_affine = torch.as_tensor(
                record["padded_affine_ras_mm"], dtype=torch.float32
            )
            if not torch.allclose(
                source_affine[index].detach().cpu().float(),
                expected_affine,
                rtol=0.0,
                atol=1e-5,
            ):
                raise RuntimeError(f"BrainLM padded affine differs: {scan_id}")
            support_digest = _support_sha256(support[index])
            support_count = int((support[index] > 0.5).sum().item())
            declared_support_count = batch_values[
                "support_foreground_voxels"
            ][index]
            context_hash_fields = (
                "padded_t1_artifact_descriptor_sha256",
                "padded_mask_artifact_descriptor_sha256",
                "cache_metadata_artifact_descriptor_sha256",
                "structural_source_identity_sha256",
                "native_alignment_authority_sha256",
                "target_artifact_identity_sha256",
                "support_tensor_sha256",
                "admitted_scan_context_record_sha256",
            )
            if (
                any(
                    not _is_sha256(str(batch_values[field][index]))
                    for field in context_hash_fields
                )
                or batch_values["admitted_scan_context_schema"][index]
                != BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA
                or isinstance(declared_support_count, bool)
                or not isinstance(declared_support_count, int)
                or declared_support_count != support_count
                or declared_support_count
                != record.get("support_foreground_voxels")
                or batch_values["support_tensor_sha256"][index]
                != support_digest
                or support_digest != record.get("support_tensor_sha256")
            ):
                raise RuntimeError(f"BrainLM structural support hash differs: {scan_id}")
            admitted_record = {
                "schema": BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA,
                "scan_id": scan_id,
                "role": role,
                "native_preprocessing_source_sha256": (
                    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
                ),
                "prepared_t1_sha256": t1_hashes[index],
                "prepared_mask_sha256": mask_hashes[index],
                "padded_t1_artifact_descriptor_sha256": str(
                    batch_values["padded_t1_artifact_descriptor_sha256"][index]
                ),
                "padded_mask_artifact_descriptor_sha256": str(
                    batch_values["padded_mask_artifact_descriptor_sha256"][index]
                ),
                "cache_metadata_artifact_descriptor_sha256": str(
                    batch_values["cache_metadata_artifact_descriptor_sha256"][index]
                ),
                "structural_source_identity_sha256": str(
                    batch_values["structural_source_identity_sha256"][index]
                ),
                "native_alignment_authority_sha256": str(
                    batch_values["native_alignment_authority_sha256"][index]
                ),
                "target_artifact_identity_sha256": str(
                    batch_values["target_artifact_identity_sha256"][index]
                ),
                "padded_shape": list(padded_rows[index]),
                "native_shape": list(native_rows[index]),
                "padding_before": list(before_rows[index]),
                "padding_after": list(after_rows[index]),
                "padded_affine_ras_mm": admitted_affine[
                    index
                ].detach().cpu().double().tolist(),
                "support_tensor_sha256": support_digest,
                "support_foreground_voxels": support_count,
            }
            admitted_digest = canonical_sha256(admitted_record)
            if admitted_digest != batch_values[
                "admitted_scan_context_record_sha256"
            ][index]:
                raise RuntimeError(
                    f"BrainLM admitted scan-context record differs: {scan_id}"
                )
            displacement, displacement_qc = _load_dense_displacement(
                authority_path,
                record["a424_dense_displacement_artifact"],
            )
            displacements.append(displacement)
            displacement_evidence.append(displacement_qc)
            scan_record_hashes.append(str(record["record_sha256"]))
            admitted_record_hashes.append(admitted_digest)
            observed_support_hashes.append(support_digest)
            observed_support_counts.append(support_count)

        affine = source_affine.to(device=device, dtype=torch.float32)
        displacement_batch = torch.stack(displacements).to(device=device)
        support_batch = support.detach().to(device=device, dtype=torch.float32)
        grid = self._physical_grid(affine, displacement_batch)
        inbounds = (grid.abs() <= 1.0).all(dim=-1).float()
        sampled_support = F.grid_sample(
            support_batch.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        sampled_support = sampled_support.clamp(0.0, 1.0) * inbounds
        labels_flat = self.atlas_labels.reshape(1, -1).expand(batch, -1)
        support_flat = sampled_support.reshape(batch, -1)
        support_counts = torch.zeros(
            batch, 425, device=device, dtype=torch.float32
        ).scatter_add(1, labels_flat, support_flat)
        inbounds_counts = torch.zeros_like(support_counts).scatter_add(
            1, labels_flat, inbounds.reshape(batch, -1)
        )
        actual_counts = support_counts[:, 1:]
        coverage = actual_counts / self.parcel_counts.to(device).view(1, 424)
        inbounds_fraction = inbounds_counts[:, 1:] / self.parcel_counts.to(device).view(
            1, 424
        )
        sorted_coverage = coverage.sort(dim=1).values
        p5 = sorted_coverage[:, int(np.floor(0.05 * 423))]
        failures = (
            (actual_counts <= 0).any(dim=1)
            | (coverage.min(dim=1).values < MINIMUM_PARCEL_COVERAGE)
            | (p5 < P5_PARCEL_COVERAGE)
            | (inbounds_counts[:, 1:] <= 0).any(dim=1)
        )
        self.last_projection_qc = {
            "scan_ids": scan_ids,
            "minimum_coverage_by_batch": coverage.min(dim=1).values.detach().cpu().tolist(),
            "p5_coverage_by_batch": p5.detach().cpu().tolist(),
            "median_coverage_by_batch": coverage.median(dim=1).values.detach().cpu().tolist(),
            "minimum_inbounds_fraction_by_batch": inbounds_fraction.min(dim=1).values.detach().cpu().tolist(),
            "all_424_parcels_nonempty": bool((actual_counts > 0).all().item()),
            "nonlinear_dense_pull_field_applied": True,
            "inverse_field_applied": False,
            "affine_only_fallback_used": False,
            "prediction_target_context_shared": True,
            "displacement_evidence": displacement_evidence,
            "passed": not bool(failures.any().item()),
        }
        if bool(failures.any().item()):
            raise RuntimeError("BrainLM subject-support/inbounds parcel coverage failed")
        context_identity: dict[str, Any] = {
            "schema": BRAINLM_BATCH_CONTEXT_IDENTITY_SCHEMA,
            "brainlm_perceptual_identity_sha256": context[
                "brainlm_perceptual_identity_sha256"
            ],
            "authority_file_sha256": self.authority_identity["file_sha256"],
            "authority_content_record_sha256": context[
                "authority_content_record_sha256"
            ],
            "authority_record_sha256": context["authority_record_sha256"],
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "run_artifact_identity_sha256": context[
                "run_artifact_identity_sha256"
            ],
            "cohort_admission_identity_record_sha256": context[
                "cohort_admission_identity_record_sha256"
            ],
            "dataset_state_record_sha256": context[
                "dataset_state_record_sha256"
            ],
            "dataset_state_sha256": context["dataset_state_sha256"],
            "structural_artifact_identities_sha256": context[
                "structural_artifact_identities_sha256"
            ],
            "scan_ids": scan_ids,
            "roles": roles,
            "prepared_t1_sha256": t1_hashes,
            "prepared_mask_sha256": mask_hashes,
            "padded_t1_artifact_descriptor_sha256": [
                str(value)
                for value in batch_values[
                    "padded_t1_artifact_descriptor_sha256"
                ]
            ],
            "padded_mask_artifact_descriptor_sha256": [
                str(value)
                for value in batch_values[
                    "padded_mask_artifact_descriptor_sha256"
                ]
            ],
            "cache_metadata_artifact_descriptor_sha256": [
                str(value)
                for value in batch_values[
                    "cache_metadata_artifact_descriptor_sha256"
                ]
            ],
            "structural_source_identity_sha256": [
                str(value)
                for value in batch_values["structural_source_identity_sha256"]
            ],
            "native_alignment_authority_sha256": [
                str(value)
                for value in batch_values["native_alignment_authority_sha256"]
            ],
            "target_artifact_identity_sha256": [
                str(value)
                for value in batch_values["target_artifact_identity_sha256"]
            ],
            "support_tensor_sha256": observed_support_hashes,
            "support_foreground_voxels": observed_support_counts,
            "admitted_scan_context_record_sha256": admitted_record_hashes,
            "authority_scan_record_sha256": scan_record_hashes,
            "displacement_sha256": [item["sha256"] for item in displacement_evidence],
            "displacement_size_bytes": [
                int(item["size_bytes"]) for item in displacement_evidence
            ],
            "padded_shape": list(PADDED_SHAPE),
            "native_shape": list(NATIVE_SHAPE),
            "padding_before": list(PADDING_BEFORE),
            "padding_after": list(PADDING_AFTER),
            "padded_affine_ras_mm": (
                admitted_affine.detach().cpu().double().tolist()
            ),
            "runtime_mapping_contract": _expected_mapping_contract(),
            "prediction_target_context_shared": True,
        }
        context_identity["record_sha256"] = canonical_sha256(context_identity)
        validated_context_identity = validate_brainlm_batch_context_identity(
            context_identity,
            expected_brainlm_identity_sha256=(
                self.connect4_artifact_identity_sha256
            ),
            expected_dataset_state_sha256=context["dataset_state_sha256"],
        )
        context_sha = validated_context_identity["record_sha256"]
        return {
            "grid": grid,
            "support_flat": support_flat,
            "actual_counts": actual_counts,
            "labels": labels_flat.unsqueeze(1),
            "context_sha256": context_sha,
            "context_identity": validated_context_identity,
        }

    def _parcel_timeseries(
        self,
        volume: torch.Tensor,
        prepared: Mapping[str, Any],
    ) -> torch.Tensor:
        batch = volume.shape[0]
        pieces: list[torch.Tensor] = []
        for start in range(0, INPUT_FRAMES, self.resample_chunk_frames):
            chunk = volume[:, 0, start : start + self.resample_chunk_frames].float()
            sampled = F.grid_sample(
                chunk,
                prepared["grid"],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            flat = sampled.reshape(batch, sampled.shape[1], -1)
            indices = prepared["labels"].expand(-1, flat.shape[1], -1)
            weighted = flat * prepared["support_flat"].unsqueeze(1)
            sums = torch.zeros(
                batch, flat.shape[1], 425, device=volume.device, dtype=flat.dtype
            ).scatter_add(2, indices, weighted)
            pieces.append(
                sums[..., 1:]
                / prepared["actual_counts"].to(flat).unsqueeze(1).clamp_min(1e-6)
            )
        series = torch.cat(pieces, dim=1)
        if tuple(series.shape) != (batch, INPUT_FRAMES, BRAINLM_PARCELS):
            raise AssertionError("BrainLM A424 parcel-series shape differs")
        return series

    @torch.no_grad()
    def prepare_context_identity(
        self,
        context: Mapping[str, Any],
        *,
        device: torch.device | None = None,
        batch: int | None = None,
    ) -> dict[str, Any]:
        """Authenticate the exact projection identity before a full model step."""

        support = context.get("brain_support") if isinstance(context, Mapping) else None
        if not torch.is_tensor(support):
            raise RuntimeError("BrainLM context preflight requires brain_support")
        resolved_batch = int(support.shape[0]) if batch is None else int(batch)
        resolved_device = support.device if device is None else torch.device(device)
        prepared = self._prepare_context(
            context, device=resolved_device, batch=resolved_batch
        )
        identity = validate_brainlm_batch_context_identity(
            prepared["context_identity"],
            expected_brainlm_identity_sha256=(
                self.connect4_artifact_identity_sha256
            ),
            expected_dataset_state_sha256=context.get("dataset_state_sha256"),
        )
        self.last_preflight_context_identity = dict(identity)
        return dict(identity)

    @staticmethod
    def _recovery_adapt(series: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(
            series.transpose(1, 2),
            size=BRAINLM_TIMEPOINTS,
            mode="linear",
            align_corners=True,
        )
        q1 = torch.quantile(values, 0.25, dim=-1, keepdim=True)
        median = torch.quantile(values, 0.50, dim=-1, keepdim=True)
        q3 = torch.quantile(values, 0.75, dim=-1, keepdim=True)
        scaled = (values - median) / (q3 - q1).clamp_min(1e-4)
        return scaled.clamp(-6.0, 6.0)

    def forward(
        self,
        fmri: torch.Tensor,
        *,
        context: Mapping[str, Any] | None = None,
        input_role: str | None = None,
    ) -> dict[str, Any]:
        if input_role not in {"prediction", "target"}:
            raise RuntimeError("BrainLM input_role must be prediction or target")
        if (
            fmri.ndim != 6
            or fmri.shape[1] != 1
            or fmri.shape[2] != INPUT_FRAMES
            or tuple(fmri.shape[-3:]) != PADDED_SHAPE
        ):
            raise ValueError("BrainLM input must be [B,1,128,64,80,64]")
        if not torch.isfinite(fmri).all():
            raise ValueError("BrainLM fMRI input contains NaN or infinity")
        prepared = self._prepare_context(
            context or {}, device=fmri.device, batch=int(fmri.shape[0])
        )
        series = self._recovery_adapt(self._parcel_timeseries(fmri, prepared))
        coordinates = self.coordinates_xyz.to(
            device=fmri.device, dtype=torch.float32
        ).unsqueeze(0).expand(fmri.shape[0], -1, -1)
        noise = self.deterministic_mask_noise.to(
            device=fmri.device, dtype=torch.float32
        ).expand(fmri.shape[0], -1)
        output = self.model.vit(
            signal_vectors=series,
            xyz_vectors=coordinates,
            noise=noise,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states
        if not isinstance(hidden, (tuple, list)) or len(hidden) != 5:
            raise RuntimeError("official four-layer BrainLM hidden-state contract differs")
        features = [
            F.layer_norm(value[:, 0].float(), (value.shape[-1],))
            for value in hidden[-4:]
        ]
        concatenated = torch.cat(features, dim=-1)
        if tuple(concatenated.shape) != (fmri.shape[0], 4 * BRAINLM_HIDDEN_SIZE):
            raise RuntimeError("BrainLM perceptual feature shape differs")
        if not torch.isfinite(concatenated).all():
            raise RuntimeError("BrainLM perceptual features are non-finite")
        self.last_context_sha256 = str(prepared["context_sha256"])
        self.last_context_identity = dict(prepared["context_identity"])
        self.last_input_role = input_role
        return {
            "features": concatenated,
            "context_sha256": self.last_context_sha256,
            "context_identity": self.last_context_identity,
            "input_role": input_role,
            "feature_schema": BRAINLM_FEATURE_SCHEMA,
        }


def _resolved_authority_spec(spec: Mapping[str, Any]) -> tuple[Path, str]:
    path = _resolve_spec_path(
        spec, "authority", environment_key="authority_env"
    )
    digest = spec.get("authority_sha256")
    if not digest:
        environment_name = str(spec.get("authority_sha256_env", "")).strip()
        if environment_name:
            digest = os.environ.get(environment_name)
    return path, _require_sha256(digest, label="BrainLM authority_sha256")


def validate_configured_brainlm_authority_scope(
    spec: Mapping[str, Any],
    *,
    expected_scan_roles: Mapping[str, str],
    split_identity: Mapping[str, Any],
    expected_scan_count: int = RECOVERY_BRAINLM_AUTHORIZED_SCAN_COUNT,
) -> dict[str, Any]:
    """Bind reviewed BrainLM authority to exactly train+development scans.

    This early admission authenticates only authority metadata. It deliberately
    does not open a dense displacement child or any functional target bytes.
    Full extractor construction later authenticates every displacement byte.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("BrainLM extractor specification must be a mapping")
    if not isinstance(expected_scan_roles, Mapping):
        raise TypeError("expected BrainLM scan roles must be a mapping")
    if (
        isinstance(expected_scan_count, bool)
        or not isinstance(expected_scan_count, int)
        or expected_scan_count < 1
    ):
        raise ValueError("expected BrainLM authority scan count must be positive")

    role_map = dict(expected_scan_roles)
    allowed_roles = {"train", "development-validation"}
    if (
        len(role_map) != expected_scan_count
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in role_map)
        or any(role not in allowed_roles for role in role_map.values())
        or set(role_map.values()) != allowed_roles
    ):
        raise RuntimeError(
            "expected BrainLM authority scope must be the exact non-empty "
            "train+development target set"
        )
    if not isinstance(split_identity, Mapping):
        raise TypeError("split identity must be a mapping")
    partitions = split_identity.get("partitions")
    if (
        split_identity.get("format") != "connect4_split_identity_v2"
        or not isinstance(split_identity.get("sha256"), str)
        or not isinstance(partitions, Mapping)
        or set(partitions) != {"train", "validation", "test"}
    ):
        raise RuntimeError("BrainLM authority received an invalid split identity")

    split_roles: dict[str, str] = {}
    sealed_ids: list[str] = []
    for partition_name, role in (
        ("train", "train"),
        ("validation", "development-validation"),
        ("test", None),
    ):
        records = partitions[partition_name]
        if not isinstance(records, list) or not records:
            raise RuntimeError(
                f"BrainLM split partition {partition_name} must be non-empty"
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise RuntimeError("BrainLM split partition record is invalid")
            scan_id = record.get("scan_id")
            if not isinstance(scan_id, str) or not scan_id:
                raise RuntimeError("BrainLM split contains an empty scan ID")
            if scan_id in split_roles or scan_id in sealed_ids:
                raise RuntimeError("BrainLM split contains duplicate scan IDs")
            if role is None:
                sealed_ids.append(scan_id)
            else:
                split_roles[scan_id] = role
    if split_roles != role_map:
        raise RuntimeError(
            "expected BrainLM role map differs from the immutable split identity"
        )
    if set(sealed_ids) & set(role_map):
        raise RuntimeError("sealed scans cannot enter the BrainLM authority scope")

    authority_path, authority_sha256 = _resolved_authority_spec(spec)
    authority_records, authority_identity = load_brainlm_authority(
        authority_path, expected_sha256=authority_sha256
    )
    authority_ids = set(authority_records)
    expected_ids = set(role_map)
    missing = sorted(expected_ids - authority_ids)
    extra = sorted(authority_ids - expected_ids)
    if missing or extra:
        raise RuntimeError(
            "BrainLM authority scope differs from immutable train+development "
            f"targets (missing={missing[:10]}, extra={extra[:10]})"
        )
    role_mismatches = sorted(
        scan_id
        for scan_id in expected_ids
        if authority_records[scan_id].get("role") != role_map[scan_id]
    )
    if role_mismatches:
        raise RuntimeError(
            "BrainLM authority role differs from immutable split for scans: "
            f"{role_mismatches[:10]}"
        )

    ordered_role_items = [
        {"scan_id": scan_id, "role": role_map[scan_id]}
        for scan_id in sorted(role_map)
    ]
    scope = {
        "schema": BRAINLM_AUTHORITY_SPLIT_SCOPE_SCHEMA,
        "authority_file_sha256": authority_identity["file_sha256"],
        "authority_record_sha256": authority_identity["authority_record_sha256"],
        "authority_ordered_scan_ids_sha256": authority_identity[
            "ordered_scan_ids_sha256"
        ],
        "split_assignment_sha256": split_identity["sha256"],
        "split_identity_sha256": canonical_sha256(split_identity),
        "target_scan_roles_sha256": canonical_sha256(ordered_role_items),
        "authorized_scan_count": len(role_map),
        "train_scan_count": sum(role == "train" for role in role_map.values()),
        "development_scan_count": sum(
            role == "development-validation" for role in role_map.values()
        ),
        "sealed_scan_count": len(sealed_ids),
        "sealed_scan_ids_sha256": canonical_sha256(sorted(sealed_ids)),
        "sealed_scans_authorized": False,
        "functional_target_bytes_opened": False,
    }
    scope["record_sha256"] = canonical_sha256(scope)
    return scope


def build_official_brainlm_a424_extractor(
    spec: Mapping[str, Any],
    *,
    device: torch.device,
) -> OfficialBrainLMA424Extractor:
    if spec.get("adapter_contract") != BRAINLM_CONTEXT_ADAPTER_CONTRACT:
        raise ValueError(
            "BrainLM requires adapter_contract="
            f"{BRAINLM_CONTEXT_ADAPTER_CONTRACT!r}"
        )
    model, labels, coordinates, artifacts = _load_official_assets(spec)
    authority_path, authority_sha = _resolved_authority_spec(spec)
    authority_records, authority_identity = load_brainlm_authority(
        authority_path, expected_sha256=authority_sha
    )
    module = OfficialBrainLMA424Extractor(
        model,
        labels,
        coordinates,
        authority_records=authority_records,
        artifact_identity=artifacts,
        authority_identity=authority_identity,
        resample_chunk_frames=int(spec.get("resample_chunk_frames", 4)),
    )
    return module.to(device).eval()


def _expected_official_content_identity() -> dict[str, Any]:
    publication: dict[str, Any] = {
        "schema": OFFICIAL_BRAINLM_PUBLICATION_MARKER_SCHEMA,
        "sha256": OFFICIAL_BRAINLM_PUBLICATION_MARKER_SHA256,
        "size_bytes": OFFICIAL_BRAINLM_PUBLICATION_MARKER_SIZE_BYTES,
        "inventory_sha256": OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256,
        "directories_sha256": OFFICIAL_BRAINLM_PUBLICATION_DIRECTORIES_SHA256,
        "published_source_root": OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT,
        "purpose": OFFICIAL_BRAINLM_PUBLICATION_PURPOSE,
        "inventory_file_count": OFFICIAL_BRAINLM_PUBLICATION_FILE_COUNT,
        "working_tree_file_count": (
            OFFICIAL_BRAINLM_PUBLICATION_WORKING_TREE_FILE_COUNT
        ),
        "directory_count": OFFICIAL_BRAINLM_PUBLICATION_DIRECTORY_COUNT,
        "pyc_or_pycache_file_count": 0,
        "status": "COMMITTED",
        "selected_runtime_entries_reverified_against_marker": True,
    }
    publication["content_record_sha256"] = canonical_sha256(publication)
    value: dict[str, Any] = {
        "schema": "connect4-official-github-brainlm-artifacts-v1",
        "repository": OFFICIAL_BRAINLM_REPOSITORY,
        "source_revision": OFFICIAL_BRAINLM_SOURCE_REVISION,
        "immutable_publication": publication,
        "source_files": {
            relative: {"size_bytes": size, "sha256": digest}
            for relative, (size, digest) in OFFICIAL_BRAINLM_SOURCE_FILES.items()
        },
        "runtime_dependencies": {
            "transformers_version": OFFICIAL_BRAINLM_TRANSFORMERS_VERSION,
            "matches_official_requirements": True,
        },
        "checkpoint": {
            "relative_path": OFFICIAL_BRAINLM_CHECKPOINT_RELATIVE_PATH,
            "sha256": OFFICIAL_BRAINLM_CHECKPOINT_SHA256,
            "size_bytes": OFFICIAL_BRAINLM_CHECKPOINT_SIZE_BYTES,
            "state_dict_tensor_count": 124,
            "parameter_count": 7_607_724,
            "has_signal_embedding_projection": True,
            "has_xyz_embedding_projection": True,
            "has_image_patch_projection": False,
        },
        "config": {
            "relative_path": OFFICIAL_BRAINLM_CONFIG_RELATIVE_PATH,
            "sha256": OFFICIAL_BRAINLM_CONFIG_SHA256,
            "size_bytes": OFFICIAL_BRAINLM_CONFIG_SIZE_BYTES,
            "architecture": "BrainLMForPretraining",
            "hidden_layers": BRAINLM_HIDDEN_LAYERS,
            "hidden_size": BRAINLM_HIDDEN_SIZE,
            "num_parcels": BRAINLM_PARCELS,
            "num_timepoints": BRAINLM_TIMEPOINTS,
            "timepoint_patch": BRAINLM_TIMEPOINT_PATCH,
        },
        "a424_atlas": {
            "sha256": OFFICIAL_A424_ATLAS_SHA256,
            "size_bytes": OFFICIAL_A424_ATLAS_SIZE_BYTES,
            "shape": list(A424_SHAPE),
            "affine_ras_mm": A424_AFFINE_RAS_MM.tolist(),
        },
        "a424_coordinates": {
            "sha256": OFFICIAL_A424_COORDINATES_SHA256,
            "size_bytes": OFFICIAL_A424_COORDINATES_SIZE_BYTES,
            "order": "rows 1..424, identical to atlas label/parcel-series order",
        },
    }
    value["content_record_sha256"] = canonical_sha256(value)
    return value


def validate_configured_brainlm_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a recorded v2 identity without loading model or atlas bytes.

    This validates the complete path-independent contract and its internal
    content hashes.  Artifact admission must still construct
    :func:`configured_brainlm_identity` from the configured files and require
    exact equality; this lightweight function alone does not authenticate
    bytes on disk.
    """

    if not isinstance(identity, Mapping):
        raise RuntimeError("recorded BrainLM identity is not an object")
    value = dict(identity)
    claimed = value.pop("record_sha256", None)
    if not _is_sha256(claimed) or claimed != canonical_sha256(value):
        raise RuntimeError("recorded BrainLM identity hash differs")
    model_artifacts = value.get("model_artifacts")
    authority = value.get("mni_authority")
    static_expected = {
        "schema": "connect4-brainlm-a424-perceptual-identity-v2",
        "name": "brainlm",
        "adapter_contract": BRAINLM_CONTEXT_ADAPTER_CONTRACT,
        "feature_contract": BRAINLM_FEATURE_SCHEMA,
        "feature_layers": (
            "last four encoder hidden states, CLS token, layer-normalized"
        ),
        "deterministic_identical_prediction_target_mask": True,
        "input_frames": INPUT_FRAMES,
        "input_tr_seconds": INPUT_TR_SECONDS,
        "temporal_adapter": "linear 128 to 200 samples, align_corners=True",
        "scaler_adapter": "per-scan/per-parcel median-IQR then clamp [-6,6]",
        "normalization_and_tr_match_pretraining_exactly": False,
        "pretraining_domain_exactness_claimed": False,
        "domain_shift_reason": (
            "TR=3 s differs from BrainLM training acquisitions and public "
            "population parcel median/IQR vectors are unavailable"
        ),
        "projection": _expected_mapping_contract(),
    }
    for key, expected in static_expected.items():
        if value.get(key) != expected:
            raise RuntimeError(f"recorded BrainLM identity field differs: {key}")
    if model_artifacts != _expected_official_content_identity():
        raise RuntimeError("recorded BrainLM official artifact identity differs")
    expected_authority_fields = {
        "schema",
        "file_sha256",
        "size_bytes",
        "authority_record_sha256",
        "native_preprocessing_source_sha256",
        "ordered_scan_ids",
        "ordered_scan_ids_sha256",
        "scan_count",
        "all_scans_explicitly_structurally_reviewed",
        "dense_nonlinear_pull_field_required",
        "affine_only_fallback_forbidden",
        "content_record_sha256",
    }
    if not isinstance(authority, Mapping) or set(authority) != expected_authority_fields:
        raise RuntimeError("recorded BrainLM authority identity fields differ")
    authority_unsigned = dict(authority)
    authority_content_hash = authority_unsigned.pop("content_record_sha256", None)
    ordered_ids = authority.get("ordered_scan_ids")
    if (
        authority.get("schema") != BRAINLM_AUTHORITY_SCHEMA
        or not _is_sha256(authority.get("file_sha256"))
        or not _is_sha256(authority.get("authority_record_sha256"))
        or authority.get("native_preprocessing_source_sha256")
        != CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        or not isinstance(authority.get("size_bytes"), int)
        or isinstance(authority.get("size_bytes"), bool)
        or int(authority["size_bytes"]) < 1
        or not isinstance(ordered_ids, list)
        or not ordered_ids
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in ordered_ids)
        or len(ordered_ids) != len(set(ordered_ids))
        or authority.get("scan_count") != len(ordered_ids)
        or authority.get("ordered_scan_ids_sha256") != canonical_sha256(ordered_ids)
        or authority.get("all_scans_explicitly_structurally_reviewed") is not True
        or authority.get("dense_nonlinear_pull_field_required") is not True
        or authority.get("affine_only_fallback_forbidden") is not True
        or authority_content_hash != canonical_sha256(authority_unsigned)
    ):
        raise RuntimeError("recorded BrainLM authority identity differs")
    if set(value) != {*static_expected, "model_artifacts", "mni_authority"}:
        raise RuntimeError("recorded BrainLM identity has unexpected fields")
    return dict(identity)


def configured_brainlm_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every configured artifact and return the checkpoint identity.

    Artifact fingerprinting deliberately performs the same strict construction
    as training.  This is slower than hashing only the checkpoint, but it makes
    resume/checkpoint identity bind the source, real BrainLM architecture,
    atlas, coordinates, reviewed nonlinear authority, and recovery adapter.
    """

    module = build_official_brainlm_a424_extractor(spec, device=torch.device("cpu"))
    return validate_configured_brainlm_identity(module.connect4_artifact_identity)


__all__ = [
    "A424_AFFINE_RAS_MM",
    "A424_SHAPE",
    "BRAINLM_AUTHORITY_SCHEMA",
    "BRAINLM_AUTHORITY_SCAN_SCHEMA",
    "BRAINLM_AUTHORITY_SPLIT_SCOPE_SCHEMA",
    "BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA",
    "BRAINLM_BATCH_CONTEXT_IDENTITY_SCHEMA",
    "BRAINLM_CONTEXT_ADAPTER_CONTRACT",
    "BRAINLM_CONTEXT_SCHEMA",
    "BRAINLM_FEATURE_SCHEMA",
    "BRAINLM_RUNTIME_MAPPING_SCHEMA",
    "CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256",
    "NATIVE_SHAPE",
    "OFFICIAL_A424_ATLAS_SHA256",
    "OFFICIAL_A424_COORDINATES_SHA256",
    "OFFICIAL_BRAINLM_CHECKPOINT_SHA256",
    "OFFICIAL_BRAINLM_CONFIG_SHA256",
    "OFFICIAL_BRAINLM_PUBLICATION_INVENTORY_SHA256",
    "OFFICIAL_BRAINLM_PUBLICATION_MARKER_SHA256",
    "OFFICIAL_BRAINLM_PUBLICATION_PURPOSE",
    "OFFICIAL_BRAINLM_PUBLICATION_RECORD_SHA256",
    "OFFICIAL_BRAINLM_PUBLISHED_SOURCE_ROOT",
    "OFFICIAL_BRAINLM_SOURCE_REVISION",
    "OFFICIAL_BRAINLM_TRANSFORMERS_VERSION",
    "OfficialBrainLMA424Extractor",
    "PADDED_SHAPE",
    "PADDING_AFTER",
    "PADDING_BEFORE",
    "REJECTED_GENERIC_VITMAE_CHECKPOINT_SHA256",
    "REJECTED_HISTORICAL_NATIVE_PREPROCESSING_SOURCE_SHA256",
    "RECOVERY_BRAINLM_AUTHORIZED_SCAN_COUNT",
    "_expected_mapping_contract",
    "_load_dense_displacement",
    "_support_sha256",
    "_validate_state_dict_identity",
    "build_official_brainlm_a424_extractor",
    "configured_brainlm_identity",
    "load_brainlm_authority",
    "validate_configured_brainlm_authority_scope",
    "validate_configured_brainlm_identity",
    "validate_brainlm_batch_context_identity",
]
