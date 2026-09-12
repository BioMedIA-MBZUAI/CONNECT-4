"""Evaluation-only CONNECT-4 paper metrics with no model/training dependency.

This module intentionally duplicates the small mathematical surface needed by
the one-way post-seal evaluator.  Importing it never imports ``models`` or a
training/inference entrypoint.  The paired metric definitions remain identical
to :mod:`eval.metrics`. FID and Inception Score are available only through the
closed, externally hash-pinned TorchScript SLIM-Brain authority implemented
here. Arbitrary in-memory or pickle-backed extractors are never a production
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


METRIC_DEFINITIONS_SOURCE = (
    "eval/postseal_metrics.py::{mse,ssim3d,voxel_correlation,roi_correlation,"
    "frame_to_frame_correlation,psnr}"
)

SLIMBRAIN_AUTHORITY_SCHEMA = "connect4-postseal-slimbrain-evaluation-authority-v1"
SLIMBRAIN_AVAILABLE_METRICS_SCHEMA = (
    "connect4-postseal-distributional-metrics-available-v2"
)
SLIMBRAIN_ADAPTER_CONTRACT = "connect4-slimbrain-full-volume-bctdhw-v2"
SLIMBRAIN_INPUT_SEMANTICS = {
    "axes": "B,C,T,D,H,W",
    "channels": 1,
    "value_domain": "stored-normalized-[0,1]",
    "spatial_or_temporal_resampling": False,
    "masking": "none-full-authenticated-volume",
    "batching": "one-subject-at-a-time-after-complete-cohort-authentication",
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_AUTHORITY_FIELDS = {
    "schema",
    "status",
    "model_name",
    "checkpoint",
    "source",
    "adapter",
    "feature_output",
    "logits_output",
    "runtime",
    "restrictions",
    "record_sha256",
}
_ARTIFACT_FIELDS = {"path", "sha256", "size_bytes"}
_RUNTIME_FIELDS = {
    "dependency_authority_sha256",
    "python_executable_sha256",
    "torch_version",
    "device_type",
}


class SlimBrainAuthorityError(RuntimeError):
    """Raised before an unqualified evaluation feature model can execute."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise SlimBrainAuthorityError(f"{label} is not a lowercase SHA-256")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise SlimBrainAuthorityError(f"{label} must be a positive built-in int")
    return value


def _direct_existing_path(value: object, *, label: str) -> Path:
    """Reject indirection before performing any filesystem lookup."""

    if type(value) not in {str, type(Path())}:
        raise SlimBrainAuthorityError(
            f"{label} path must be an exact built-in string or native Path"
        )
    raw = str(value)
    if not raw or "$" in raw or "~" in raw:
        raise SlimBrainAuthorityError(
            f"{label} path must be explicit and cannot use environment indirection"
        )
    path = Path(raw)
    if not path.is_absolute() or str(path) != os.path.abspath(path):
        raise SlimBrainAuthorityError(f"{label} path is not absolute and lexical")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SlimBrainAuthorityError(f"{label} cannot be resolved") from exc
    if resolved != path:
        raise SlimBrainAuthorityError(f"{label} path aliases another path")
    return path


def _stable_readonly_artifact(
    descriptor: object,
    *,
    label: str,
    expected_path: Path | None = None,
    maximum_bytes: int | None = None,
) -> tuple[Path, bytes]:
    if type(descriptor) is not dict or set(descriptor) != _ARTIFACT_FIELDS:
        raise SlimBrainAuthorityError(f"{label} descriptor fields differ")
    path = _direct_existing_path(descriptor.get("path"), label=label)
    if expected_path is not None and path != expected_path:
        raise SlimBrainAuthorityError(f"{label} path differs from the invoked path")
    try:
        before = path.lstat()
    except OSError as exc:
        raise SlimBrainAuthorityError(f"{label} cannot be inspected") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise SlimBrainAuthorityError(
            f"{label} must be one immutable regular non-symlink file in mode 0444"
        )
    expected_size = _require_positive_int(
        descriptor.get("size_bytes"), label=f"{label} size"
    )
    expected_digest = _require_sha256(
        descriptor.get("sha256"), label=f"{label} SHA-256"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        raise SlimBrainAuthorityError(f"{label} cannot be opened safely") from exc
    chunks: list[bytes] = []
    observed_size = 0
    try:
        opened = os.fstat(file_descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SlimBrainAuthorityError(f"{label} changed while opening")
        while block := os.read(file_descriptor, 1024 * 1024):
            observed_size += len(block)
            if maximum_bytes is not None and observed_size > maximum_bytes:
                raise SlimBrainAuthorityError(f"{label} exceeds its byte limit")
            chunks.append(block)
        after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise SlimBrainAuthorityError(f"{label} changed after reading") from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ):
        raise SlimBrainAuthorityError(f"{label} changed while authenticating")
    payload = b"".join(chunks)
    if observed_size != expected_size or observed_size != before.st_size:
        raise SlimBrainAuthorityError(f"{label} size differs")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise SlimBrainAuthorityError(f"{label} SHA-256 differs")
    return path, payload


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = json.dumps(
        {"shape": list(tensor.shape), "dtype": "float32"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + tensor.numpy().tobytes(order="C")
    ).hexdigest()


@dataclass(frozen=True)
class AuthenticatedSlimBrainAuthority:
    """An exact frozen TorchScript model plus externally pinned semantics."""

    model: nn.Module
    authority_path: Path
    authority_raw_sha256: str
    authority_size_bytes: int
    record: dict[str, Any]
    evidence: dict[str, Any]

    def reauthenticate(self) -> None:
        _stable_readonly_artifact(
            self.record["checkpoint"],
            label="SLIM-Brain checkpoint",
        )
        _stable_readonly_artifact(
            {key: self.record["source"][key] for key in _ARTIFACT_FIELDS},
            label="SLIM-Brain source manifest",
        )
        authority_descriptor = {
            "path": str(self.authority_path),
            "sha256": self.authority_raw_sha256,
            "size_bytes": self.authority_size_bytes,
        }
        _stable_readonly_artifact(
            authority_descriptor,
            label="SLIM-Brain authority",
            expected_path=self.authority_path,
            maximum_bytes=256 * 1024,
        )
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise SlimBrainAuthorityError("SLIM-Brain model is no longer frozen")


class _StrictSlimBrainExtractor(nn.Module):
    """Reject every output except the authority-bound feature/logits mapping."""

    def __init__(self, authority: AuthenticatedSlimBrainAuthority):
        super().__init__()
        self.authority = authority
        self.model = authority.model

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training or self.model.training:
            raise SlimBrainAuthorityError("SLIM-Brain must remain in evaluation mode")
        output = self.model(value)
        if type(output) is not dict or set(output) != {"features", "logits"}:
            raise SlimBrainAuthorityError(
                "SLIM-Brain must return exactly trained features and logits"
            )
        features = output["features"]
        logits = output["logits"]
        if not torch.is_tensor(features) or not torch.is_tensor(logits):
            raise SlimBrainAuthorityError("SLIM-Brain outputs are not tensors")
        features = features.reshape(features.shape[0], -1)
        logits = logits.reshape(logits.shape[0], -1)
        expected_feature_dim = self.authority.record["feature_output"]["dimension"]
        expected_class_count = self.authority.record["logits_output"]["class_count"]
        if features.shape != (value.shape[0], expected_feature_dim):
            raise SlimBrainAuthorityError("SLIM-Brain feature dimensions differ")
        if logits.shape != (value.shape[0], expected_class_count):
            raise SlimBrainAuthorityError("SLIM-Brain trained-logit dimensions differ")
        if not bool(torch.isfinite(features).all() and torch.isfinite(logits).all()):
            raise SlimBrainAuthorityError("SLIM-Brain outputs contain NaN or infinity")
        return {"features": features, "logits": logits}


def load_authenticated_slimbrain(
    authority_path: str | Path,
    *,
    expected_authority_sha256: str,
    dependency_authority_sha256: str,
    python_executable_sha256: str,
    device: torch.device,
) -> tuple[_StrictSlimBrainExtractor, AuthenticatedSlimBrainAuthority]:
    """Load only the exact authority-pinned TorchScript SLIM-Brain evaluator."""

    raw_path = _direct_existing_path(authority_path, label="SLIM-Brain authority")
    expected_raw_digest = _require_sha256(
        expected_authority_sha256, label="SLIM-Brain authority external pin"
    )
    try:
        authority_size = raw_path.lstat().st_size
    except OSError as exc:
        raise SlimBrainAuthorityError(
            "SLIM-Brain authority cannot be inspected"
        ) from exc
    authority_descriptor = {
        "path": str(raw_path),
        "sha256": expected_raw_digest,
        "size_bytes": authority_size,
    }
    authority_file, payload = _stable_readonly_artifact(
        authority_descriptor,
        label="SLIM-Brain authority",
        expected_path=raw_path,
        maximum_bytes=256 * 1024,
    )
    try:
        record = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlimBrainAuthorityError(
            "SLIM-Brain authority is not canonical JSON"
        ) from exc
    if type(record) is not dict or set(record) != _AUTHORITY_FIELDS:
        raise SlimBrainAuthorityError("SLIM-Brain authority fields differ")
    unsigned = dict(record)
    recorded_digest = unsigned.pop("record_sha256", None)
    if (
        record.get("schema") != SLIMBRAIN_AUTHORITY_SCHEMA
        or record.get("status") != "QUALIFIED_IMMUTABLE_EVALUATION_ONLY"
        or record.get("model_name") != "slimbrain"
        or recorded_digest != _canonical_sha256(unsigned)
    ):
        raise SlimBrainAuthorityError(
            "SLIM-Brain authority signature or identity differs"
        )
    checkpoint_path, checkpoint_payload = _stable_readonly_artifact(
        record.get("checkpoint"), label="SLIM-Brain checkpoint"
    )
    source = record.get("source")
    if type(source) is not dict or set(source) != {*_ARTIFACT_FIELDS, "revision"}:
        raise SlimBrainAuthorityError("SLIM-Brain source descriptor fields differ")
    source_path, _ = _stable_readonly_artifact(
        {key: source[key] for key in _ARTIFACT_FIELDS},
        label="SLIM-Brain source manifest",
    )
    if checkpoint_path == source_path:
        raise SlimBrainAuthorityError("SLIM-Brain source and checkpoint alias")
    adapter = record.get("adapter")
    adapter_source = Path(__file__).resolve()
    if type(adapter) is not dict or set(adapter) != {
        "contract",
        "implementation_source_sha256",
        "input_semantics",
    }:
        raise SlimBrainAuthorityError("SLIM-Brain adapter evidence fields differ")
    if (
        adapter.get("contract") != SLIMBRAIN_ADAPTER_CONTRACT
        or adapter.get("input_semantics") != SLIMBRAIN_INPUT_SEMANTICS
        or adapter.get("implementation_source_sha256")
        != hashlib.sha256(adapter_source.read_bytes()).hexdigest()
    ):
        raise SlimBrainAuthorityError("SLIM-Brain adapter identity differs")
    revision = source.get("revision")
    if (
        type(revision) is not str
        or len(revision) not in {40, 64}
        or any(character not in _SHA256_CHARACTERS for character in revision)
    ):
        raise SlimBrainAuthorityError("SLIM-Brain source revision is not immutable")
    feature_output = record.get("feature_output")
    if type(feature_output) is not dict or set(feature_output) != {
        "output_key",
        "layer_name",
        "aggregation",
        "dimension",
    }:
        raise SlimBrainAuthorityError("SLIM-Brain feature semantics fields differ")
    if (
        feature_output.get("output_key") != "features"
        or type(feature_output.get("layer_name")) is not str
        or not feature_output.get("layer_name")
        or type(feature_output.get("aggregation")) is not str
        or not feature_output.get("aggregation")
    ):
        raise SlimBrainAuthorityError("SLIM-Brain feature semantics differ")
    _require_positive_int(
        feature_output.get("dimension"), label="SLIM-Brain feature dimension"
    )
    logits_output = record.get("logits_output")
    if type(logits_output) is not dict or set(logits_output) != {
        "output_key",
        "layer_name",
        "class_count",
        "trained",
        "class_semantics_sha256",
    }:
        raise SlimBrainAuthorityError("SLIM-Brain logits semantics fields differ")
    if (
        logits_output.get("output_key") != "logits"
        or type(logits_output.get("layer_name")) is not str
        or not logits_output.get("layer_name")
        or logits_output.get("trained") is not True
    ):
        raise SlimBrainAuthorityError("SLIM-Brain trained-logit semantics differ")
    _require_positive_int(
        logits_output.get("class_count"), label="SLIM-Brain class count"
    )
    if logits_output["class_count"] < 2:
        raise SlimBrainAuthorityError(
            "SLIM-Brain trained logits require at least two classes"
        )
    _require_sha256(
        logits_output.get("class_semantics_sha256"),
        label="SLIM-Brain class-semantics SHA-256",
    )
    runtime = record.get("runtime")
    if type(runtime) is not dict or set(runtime) != _RUNTIME_FIELDS:
        raise SlimBrainAuthorityError("SLIM-Brain runtime evidence fields differ")
    expected_runtime = {
        "dependency_authority_sha256": _require_sha256(
            dependency_authority_sha256,
            label="closed dependency authority SHA-256",
        ),
        "python_executable_sha256": _require_sha256(
            python_executable_sha256,
            label="Python executable SHA-256",
        ),
        "torch_version": str(torch.__version__),
        "device_type": device.type,
    }
    if runtime != expected_runtime:
        raise SlimBrainAuthorityError("SLIM-Brain runtime identity differs")
    restrictions = record.get("restrictions")
    if restrictions != {
        "environment_indirection": False,
        "generic_or_pickle_model_loading": False,
        "data_derived_feature_mask": False,
        "single_subject_distributional_metrics": False,
        "training_or_selection_feedback": False,
        "target_access_before_prediction_set_seal": False,
    }:
        raise SlimBrainAuthorityError("SLIM-Brain restriction contract differs")
    try:
        model = torch.jit.load(io.BytesIO(checkpoint_payload), map_location=device)
    except Exception as exc:
        raise SlimBrainAuthorityError(
            "SLIM-Brain checkpoint is not a runnable TorchScript module"
        ) from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    required_attributes = {
        "connect4_model_name": "slimbrain",
        "connect4_adapter_contract": SLIMBRAIN_ADAPTER_CONTRACT,
        "connect4_source_revision": revision,
        "connect4_feature_layer": feature_output["layer_name"],
        "connect4_feature_aggregation": feature_output["aggregation"],
        "connect4_feature_dimension": feature_output["dimension"],
        "connect4_logits_layer": logits_output["layer_name"],
        "connect4_logits_trained": True,
        "connect4_class_count": logits_output["class_count"],
        "connect4_class_semantics_sha256": logits_output["class_semantics_sha256"],
    }
    if any(
        getattr(model, key, None) != value for key, value in required_attributes.items()
    ):
        raise SlimBrainAuthorityError(
            "SLIM-Brain TorchScript attributes differ from its pinned authority"
        )
    evidence = {
        "authority": {
            "path": str(authority_file),
            "raw_sha256": expected_raw_digest,
            "record_sha256": recorded_digest,
            "size_bytes": len(payload),
        },
        "model_name": "slimbrain",
        "checkpoint": dict(record["checkpoint"]),
        "source": dict(record["source"]),
        "adapter": dict(adapter),
        "feature_output": dict(feature_output),
        "logits_output": dict(logits_output),
        "runtime": dict(runtime),
        "frozen_eval": True,
        "torchscript_only": True,
    }
    authenticated = AuthenticatedSlimBrainAuthority(
        model=model,
        authority_path=authority_file,
        authority_raw_sha256=expected_raw_digest,
        authority_size_bytes=len(payload),
        record=record,
        evidence=evidence,
    )
    authenticated.reauthenticate()
    extractor = _StrictSlimBrainExtractor(authenticated).to(device).eval()
    return extractor, authenticated


def _ensure_bctdhw(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"fMRI input must be a torch.Tensor, got {type(value)!r}")
    if value.ndim == 5:
        value = value.unsqueeze(1)
    if value.ndim != 6:
        raise ValueError(f"expected [B,C,T,D,H,W] or [B,T,D,H,W], got {value.shape}")
    return value.float()


def _paired(
    predicted: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted, target = _ensure_bctdhw(predicted), _ensure_bctdhw(target)
    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction/target shapes differ: {predicted.shape} vs {target.shape}"
        )
    return predicted, target


def _spatial_mask(mask: Optional[torch.Tensor], volume: torch.Tensor) -> torch.Tensor:
    batch, _, _, depth, height, width = volume.shape
    if mask is None:
        raise ValueError(
            "an explicit paired target-validity mask is required; scoring the "
            "background or deriving support inside a metric is forbidden"
        )
    value = torch.as_tensor(mask, device=volume.device)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    elif value.ndim == 5:
        if value.shape[1] != 1:
            raise ValueError(
                f"target-validity mask channel dimension must be 1, got {value.shape}"
            )
        value = value[:, 0]
    elif value.ndim != 4:
        raise ValueError(
            "target-validity mask must be [D,H,W], [B,D,H,W], or [B,1,D,H,W], "
            f"got {value.shape}"
        )
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1, -1)
    if value.shape[0] != batch or tuple(value.shape[-3:]) != (
        depth,
        height,
        width,
    ):
        raise ValueError(
            "target-validity mask "
            f"{value.shape} is incompatible with volume {volume.shape}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("target-validity mask contains NaN or infinity")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("target-validity mask must be exactly binary")
    flat = value.reshape(batch, -1) > 0.5
    if not bool(flat.any(dim=1).all()):
        raise ValueError("target-validity mask is empty for at least one subject")
    return flat


def _pearson_last(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float() - left.float().mean(dim=-1, keepdim=True)
    right = right.float() - right.float().mean(dim=-1, keepdim=True)
    numerator = (left * right).sum(dim=-1)
    denominator = (left.square().sum(dim=-1) * right.square().sum(dim=-1)).sqrt()
    return torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        0.0,
    )


def _masked_pearson_last(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weights = mask.to(left.dtype)
    while weights.ndim < left.ndim:
        weights = weights.unsqueeze(1)
    count = weights.sum(dim=-1).clamp_min(1.0)
    left_mean = (left * weights).sum(dim=-1) / count
    right_mean = (right * weights).sum(dim=-1) / count
    left_centered = (left - left_mean.unsqueeze(-1)) * weights
    right_centered = (right - right_mean.unsqueeze(-1)) * weights
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = (
        left_centered.square().sum(dim=-1) * right_centered.square().sum(dim=-1)
    ).sqrt()
    return torch.where(
        denominator > 1e-12,
        numerator / denominator.clamp_min(1e-12),
        0.0,
    )


def _roi_timeseries(volume: torch.Tensor, roi_masks: torch.Tensor) -> torch.Tensor:
    batch, _, frames, depth, height, width = volume.shape
    regions = roi_masks.shape[1]
    values = volume.mean(dim=1).reshape(batch, frames, depth * height * width)
    masks = roi_masks.reshape(batch, regions, depth * height * width).to(values.dtype)
    denominator = masks.sum(dim=2)
    if not bool((denominator > 0).all()):
        raise ValueError(
            "every configured ROI must have non-empty target-validity support "
            "for every subject"
        )
    return torch.einsum("brp,btp->brt", masks, values) / denominator.unsqueeze(-1)


def voxel_correlation(predicted, target, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    predicted_flat = predicted.mean(dim=1).flatten(2).transpose(1, 2)
    target_flat = target.mean(dim=1).flatten(2).transpose(1, 2)
    correlation = _pearson_last(predicted_flat, target_flat)
    valid = _spatial_mask(mask, predicted)
    per_subject = (correlation * valid).sum(dim=1) / valid.sum(dim=1)
    return float(per_subject.mean())


def roi_correlation(predicted, target, roi_masks, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    if roi_masks is None:
        raise ValueError("ROI masks are required for the paper ROI correlation")
    roi_masks = torch.as_tensor(
        roi_masks, device=predicted.device, dtype=predicted.dtype
    )
    if roi_masks.ndim == 4:
        roi_masks = roi_masks.unsqueeze(0)
    if roi_masks.ndim != 5 or roi_masks.shape[0] != predicted.shape[0]:
        raise ValueError(f"ROI masks must be [B,R,D,H,W], got {roi_masks.shape}")
    if tuple(roi_masks.shape[-3:]) != tuple(predicted.shape[-3:]):
        raise ValueError("ROI-mask and fMRI spatial shapes differ")
    if not bool(torch.isfinite(roi_masks).all()):
        raise ValueError("ROI masks contain NaN or infinity")
    if not bool(((roi_masks == 0) | (roi_masks == 1)).all()):
        raise ValueError("ROI masks must be exactly binary")
    spatial = _spatial_mask(mask, predicted).reshape(
        predicted.shape[0], *predicted.shape[-3:]
    )
    roi_masks = roi_masks * spatial.unsqueeze(1).to(roi_masks.dtype)
    correlation = _pearson_last(
        _roi_timeseries(predicted, roi_masks),
        _roi_timeseries(target, roi_masks),
    )
    valid = roi_masks.flatten(2).sum(dim=-1) > 0
    if not bool(valid.all()):
        raise ValueError(
            "every configured ROI must have non-empty target-validity support "
            "for every subject"
        )
    per_subject = correlation.mean(dim=1)
    return float(per_subject.mean())


def frame_to_frame_correlation(predicted, target, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    predicted_flat = predicted.mean(dim=1).flatten(2)
    target_flat = target.mean(dim=1).flatten(2)
    return float(
        _masked_pearson_last(
            predicted_flat, target_flat, _spatial_mask(mask, predicted)
        ).mean()
    )


def mse(predicted, target, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    valid = _spatial_mask(mask, predicted).to(predicted.dtype)[:, None, None, :]
    square_error = (predicted - target).square().flatten(3)
    numerator = (square_error * valid).sum(dim=(1, 2, 3))
    denominator = valid.sum(dim=3).flatten() * predicted.shape[1] * predicted.shape[2]
    return float((numerator / denominator.clamp_min(1.0)).mean())


def _expand_frame_masks(
    mask: Optional[torch.Tensor], volume: torch.Tensor
) -> torch.Tensor | None:
    if mask is None:
        return None
    flat = _spatial_mask(mask, volume)
    batch, _, frames, depth, height, width = volume.shape
    return (
        flat.reshape(batch, 1, 1, depth, height, width)
        .expand(batch, 1, frames, depth, height, width)
        .permute(0, 2, 1, 3, 4, 5)
        .reshape(batch * frames, 1, depth, height, width)
        .to(volume.dtype)
    )


def _framewise_ssim(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    window_size: int = 7,
    frame_chunk_size: int = 4,
) -> torch.Tensor:
    batch, channels, frames, depth, height, width = predicted.shape
    predicted_frames = predicted.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * frames, channels, depth, height, width
    )
    target_frames = target.permute(0, 2, 1, 3, 4, 5).reshape_as(predicted_frames)
    masks = _expand_frame_masks(mask, predicted)
    if channels > 1:
        predicted_frames = predicted_frames.mean(dim=1, keepdim=True)
        target_frames = target_frames.mean(dim=1, keepdim=True)
    scores = []
    kernel = min(window_size, depth, height, width)
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    padding = kernel // 2

    def average(value: torch.Tensor) -> torch.Tensor:
        value = F.avg_pool3d(value, (kernel, 1, 1), stride=1, padding=(padding, 0, 0))
        value = F.avg_pool3d(value, (1, kernel, 1), stride=1, padding=(0, padding, 0))
        return F.avg_pool3d(value, (1, 1, kernel), stride=1, padding=(0, 0, padding))

    for start in range(0, batch * frames, frame_chunk_size):
        stop = min(start + frame_chunk_size, batch * frames)
        left = predicted_frames[start:stop]
        right = target_frames[start:stop]
        current_mask = None if masks is None else masks[start:stop]
        if current_mask is not None:
            left = left * current_mask
            right = right * current_mask
        left_mean, right_mean = average(left), average(right)
        left_square, right_square = left_mean.square(), right_mean.square()
        product = left_mean * right_mean
        variance_left = average(left.square()) - left_square
        variance_right = average(right.square()) - right_square
        covariance = average(left * right) - product
        numerator = (2.0 * product + 0.01**2) * (2.0 * covariance + 0.03**2)
        denominator = (left_square + right_square + 0.01**2) * (
            variance_left + variance_right + 0.03**2
        )
        score_map = numerator / denominator.clamp_min(1e-8)
        if current_mask is None:
            scores.append(score_map.flatten(1).mean(dim=1))
        else:
            scores.append(
                (score_map * current_mask).flatten(1).sum(dim=1)
                / current_mask.flatten(1).sum(dim=1).clamp_min(1.0)
            )
    return torch.cat(scores).mean()


def ssim3d(predicted, target, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    flat_masks = _spatial_mask(mask, predicted)
    per_subject = [
        _framewise_ssim(
            predicted[index : index + 1],
            target[index : index + 1],
            flat_masks[index].reshape(1, *predicted.shape[-3:]),
        )
        for index in range(predicted.shape[0])
    ]
    return float(torch.stack(per_subject).mean())


def psnr(predicted, target, mask=None) -> float:
    predicted, target = _paired(predicted, target)
    valid = _spatial_mask(mask, predicted)
    values = []
    for index in range(predicted.shape[0]):
        predicted_values = predicted[index].flatten(2)[..., valid[index]]
        target_values = target[index].flatten(2)[..., valid[index]]
        error = (predicted_values - target_values).square().mean()
        if float(error) == 0.0:
            values.append(predicted.new_tensor(float("inf")))
            continue
        data_range = target_values.max() - target_values.min()
        if float(data_range) <= 0.0:
            data_range = target_values.new_tensor(1.0)
        values.append(10.0 * torch.log10(data_range.square() / error))
    return float(torch.stack(values).mean())


def compute_all(
    predicted: torch.Tensor,
    target: torch.Tensor,
    roi_masks: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if roi_masks is None:
        raise ValueError("ROI masks are required for the paper metric set")
    return {
        "mse": mse(predicted, target, mask),
        "ssim": ssim3d(predicted, target, mask),
        "voxel_corr": voxel_correlation(predicted, target, mask),
        "roi_corr": roi_correlation(predicted, target, roi_masks, mask=mask),
        "f2f_corr": frame_to_frame_correlation(predicted, target, mask),
        "psnr": psnr(predicted, target, mask),
    }


def frechet_distance_from_features(
    generated: torch.Tensor, real: torch.Tensor
) -> float:
    generated = torch.as_tensor(generated, dtype=torch.float64)
    real = torch.as_tensor(real, dtype=torch.float64)
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("FID features must be matrices [N,F]")
    if generated.shape[0] < 2 or real.shape[0] < 2:
        raise ValueError("FID requires at least two samples per distribution")
    if generated.shape[1] != real.shape[1]:
        raise ValueError("FID feature dimensions differ")
    if not bool(torch.isfinite(generated).all() and torch.isfinite(real).all()):
        raise ValueError("FID features contain NaN or infinity")

    def covariance(value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(dim=0, keepdim=True)
        return centered.T @ centered / (value.shape[0] - 1)

    generated_mean, real_mean = generated.mean(dim=0), real.mean(dim=0)
    generated_covariance, real_covariance = covariance(generated), covariance(real)
    eigenvalues, eigenvectors = torch.linalg.eigh(generated_covariance)
    generated_sqrt = (
        eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.T
    middle = generated_sqrt @ real_covariance @ generated_sqrt
    trace_sqrt = (
        torch.linalg.eigvalsh((middle + middle.T) * 0.5).clamp_min(0).sqrt().sum()
    )
    distance = (generated_mean - real_mean).square().sum()
    distance += (
        torch.trace(generated_covariance)
        + torch.trace(real_covariance)
        - 2.0 * trace_sqrt
    )
    return float(distance.clamp_min(0.0))


def inception_score_from_logits(logits: torch.Tensor) -> float:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    if logits.ndim != 2 or logits.shape[0] < 2 or logits.shape[1] < 2:
        raise ValueError("Inception Score requires logits [N,K], N>=2, K>=2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Inception logits contain NaN or infinity")
    log_conditional = torch.log_softmax(logits, dim=-1)
    conditional = log_conditional.exp()
    marginal = conditional.mean(dim=0).clamp_min(torch.finfo(logits.dtype).tiny)
    divergence = (conditional * (log_conditional - marginal.log())).sum(dim=-1)
    return float(divergence.mean().exp())


class SynthesisMetricAccumulator:
    """Evaluation-only cohort metric accumulator for authenticated SLIM-Brain."""

    def __init__(
        self,
        feature_extractor: _StrictSlimBrainExtractor | None = None,
    ) -> None:
        if feature_extractor is not None and type(feature_extractor) is not (
            _StrictSlimBrainExtractor
        ):
            raise SlimBrainAuthorityError(
                "distributional accumulation requires the exact authenticated "
                "SLIM-Brain extractor"
            )
        if feature_extractor is None:
            self.__feature_extractor = None
            self.num_samples = 0
            self._generated_features = []
            self._real_features = []
            self._generated_logits = []
            self._feature_dim = None
            self._logit_dim = None
            return
        if type(feature_extractor.authority) is not AuthenticatedSlimBrainAuthority:
            raise SlimBrainAuthorityError(
                "distributional accumulator authority type differs"
            )
        feature_extractor.authority.reauthenticate()
        self.__feature_extractor = feature_extractor.eval()
        self.num_samples = 0
        self._generated_features: list[torch.Tensor] = []
        self._real_features: list[torch.Tensor] = []
        self._generated_logits: list[torch.Tensor] = []
        self._feature_dim: Optional[int] = None
        self._logit_dim: Optional[int] = None

    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        roi_masks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Fail closed for the removed generic masked accumulation surface."""

        predicted, target = _paired(predicted, target)
        _spatial_mask(mask, predicted)
        del target, roi_masks
        raise SlimBrainAuthorityError(
            "generic or masked post-seal accumulation is forbidden; use the exact "
            "authenticated full-volume SLIM-Brain distributional path"
        )

    @torch.no_grad()
    def update_distributional(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Accumulate full stored volumes without a target-derived feature mask."""

        predicted, target = _paired(predicted, target)
        if type(self.__feature_extractor) is not _StrictSlimBrainExtractor:
            raise SlimBrainAuthorityError(
                "distributional accumulation requires the exact authenticated "
                "SLIM-Brain extractor"
            )
        batch_size = predicted.shape[0]
        self.__feature_extractor.eval()
        generated_output = self.__feature_extractor(predicted)
        real_output = self.__feature_extractor(target)
        generated_features = generated_output["features"]
        generated_logits = generated_output["logits"]
        real_features = real_output["features"]
        generated_features = generated_features.detach().float().cpu()
        real_features = real_features.detach().float().cpu()
        generated_logits = generated_logits.detach().float().cpu()
        if generated_features.shape != real_features.shape:
            raise ValueError("SLIM-Brain generated/real feature shapes differ")
        if generated_features.shape[0] != batch_size:
            raise ValueError("SLIM-Brain must return one feature row per subject")
        if not bool(
            torch.isfinite(generated_features).all()
            and torch.isfinite(real_features).all()
        ):
            raise ValueError("SLIM-Brain returned non-finite features")
        if self._feature_dim not in (None, generated_features.shape[1]):
            raise ValueError("SLIM-Brain feature dimension changed")
        if generated_logits.shape[0] != batch_size:
            raise ValueError("SLIM-Brain must return one logits row per subject")
        if not bool(torch.isfinite(generated_logits).all()):
            raise ValueError("SLIM-Brain returned non-finite logits")
        if self._logit_dim not in (None, generated_logits.shape[1]):
            raise ValueError("SLIM-Brain logits dimension changed")
        self.num_samples += batch_size
        self._feature_dim = generated_features.shape[1]
        self._logit_dim = generated_logits.shape[1]
        self._generated_features.append(generated_features)
        self._real_features.append(real_features)
        self._generated_logits.append(generated_logits)

    @property
    def authenticated_authority(self) -> AuthenticatedSlimBrainAuthority:
        """Return the exact retained authority after fresh byte reauthentication."""

        if type(self.__feature_extractor) is not _StrictSlimBrainExtractor:
            raise SlimBrainAuthorityError("distributional extractor identity changed")
        authority = self.__feature_extractor.authority
        if type(authority) is not AuthenticatedSlimBrainAuthority:
            raise SlimBrainAuthorityError("distributional authority identity changed")
        authority.reauthenticate()
        return authority

    def distributional_tensors(self) -> dict[str, torch.Tensor]:
        """Return immutable-by-copy complete-cohort FP32 evidence tensors."""

        self.authenticated_authority.reauthenticate()
        if self.num_samples < 2:
            raise ValueError("FID requires at least two samples per distribution")
        generated = torch.cat(self._generated_features, dim=0).contiguous()
        real = torch.cat(self._real_features, dim=0).contiguous()
        logits = torch.cat(self._generated_logits, dim=0).contiguous()
        if (
            generated.shape[0] != self.num_samples
            or real.shape != generated.shape
            or logits.shape[0] != self.num_samples
            or not bool(
                torch.isfinite(generated).all()
                and torch.isfinite(real).all()
                and torch.isfinite(logits).all()
            )
        ):
            raise SlimBrainAuthorityError(
                "distributional tensor population is incomplete or non-finite"
            )
        return {
            "generated_features": generated.clone(),
            "real_features": real.clone(),
            "generated_logits": logits.clone(),
        }

    def distributional_commitments(self) -> dict[str, Any]:
        """Hash exact cohort features and logits without publishing their values."""

        tensors = self.distributional_tensors()
        generated = tensors["generated_features"]
        real = tensors["real_features"]
        logits = tensors["generated_logits"]
        return {
            "sample_count": self.num_samples,
            "feature_dimension": generated.shape[1],
            "class_count": logits.shape[1],
            "generated_features_sha256": _tensor_sha256(generated),
            "real_features_sha256": _tensor_sha256(real),
            "generated_logits_sha256": _tensor_sha256(logits),
        }

    def compute(self) -> Dict[str, float]:
        tensors = self.distributional_tensors()
        return {
            "fid": frechet_distance_from_features(
                tensors["generated_features"], tensors["real_features"]
            ),
            "is": inception_score_from_logits(tensors["generated_logits"]),
        }


__all__ = [
    "AuthenticatedSlimBrainAuthority",
    "SLIMBRAIN_ADAPTER_CONTRACT",
    "SLIMBRAIN_AUTHORITY_SCHEMA",
    "SLIMBRAIN_AVAILABLE_METRICS_SCHEMA",
    "SLIMBRAIN_INPUT_SEMANTICS",
    "SlimBrainAuthorityError",
    "SynthesisMetricAccumulator",
    "compute_all",
    "frame_to_frame_correlation",
    "frechet_distance_from_features",
    "inception_score_from_logits",
    "load_authenticated_slimbrain",
    "mse",
    "psnr",
    "roi_correlation",
    "ssim3d",
    "voxel_correlation",
]
