"""Strict adapters for pretrained 4D fMRI feature extractors.

The paper uses different models for different purposes:

* BrainLM (reference [5]) supplies the training perceptual features.
* SLIM-Brain (reference [29]) supplies the evaluation features used by FID/IS.

This module deliberately does not create a random stand-in.  A configured
extractor must be a serialized ``nn.Module``/TorchScript module, its complete
file digest must match an explicitly configured SHA-256, and it must expose one
feature vector per subject. It must also carry the model-specific CONNECT-4
adapter contract, model name, and immutable upstream source revision used when
it was exported. The digest binds a run to exact user-supplied weights; these
provenance assertions do not constitute a cryptographic author signature.
Evaluation extractors may additionally expose classification logits, which are
required for Inception Score.
"""
from __future__ import annotations

import os
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .brainlm_context import (
    BRAINLM_CONTEXT_ADAPTER_CONTRACT,
    build_official_brainlm_a424_extractor,
)


_FEATURE_KEYS = ("features", "feature", "embeddings", "embedding", "pooler_output")
_LOGIT_KEYS = ("logits", "class_logits", "classification_logits")
_MODEL_ADAPTER_CONTRACTS = {
    "brainlm": BRAINLM_CONTEXT_ADAPTER_CONTRACT,
    "slimbrain": "connect4-slimbrain-4d-adapter-v1",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(
            f"{label} must be a complete 64-character SHA-256 digest; exact "
            "pretrained-weight provenance is required for paper-faithful runs"
        )
    return digest


def _require_source_revision(value: Any, *, label: str) -> str:
    revision = str(value or "").strip().lower()
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"{label} must be a full immutable 40- or 64-character source commit"
        )
    return revision


def _validate_adapter_identity(
    module: nn.Module,
    *,
    name: str,
    source_revision: str,
) -> str:
    canonical_name = str(name).strip().lower()
    contract = _MODEL_ADAPTER_CONTRACTS.get(canonical_name)
    if contract is None:
        raise ValueError(f"no 4D adapter contract is registered for {name!r}")
    actual_name = str(getattr(module, "connect4_model_name", "")).strip().lower()
    actual_contract = str(
        getattr(module, "connect4_adapter_contract", "")
    ).strip()
    actual_revision = str(
        getattr(module, "connect4_source_revision", "")
    ).strip().lower()
    if (
        actual_name != canonical_name
        or actual_contract != contract
        or actual_revision != source_revision
    ):
        raise RuntimeError(
            f"{name} artifact does not expose the required model-specific adapter "
            f"identity ({canonical_name}, {contract}, {source_revision}). A generic "
            "serialized nn.Module cannot establish BrainLM/SLIM-Brain provenance."
        )
    return contract


def split_feature_output(output: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(features, logits)`` from a documented extractor output.

    Supported contracts are a feature tensor, ``(features, logits)``, or a
    mapping/object containing a standard feature key and optional logits key.
    Features and logits are flattened after the batch dimension so metric code
    always receives matrices shaped ``[B, F]`` and ``[B, K]``.
    """
    features: Any = None
    logits: Any = None

    if torch.is_tensor(output):
        features = output
    elif isinstance(output, Mapping):
        features = next((output[k] for k in _FEATURE_KEYS if k in output), None)
        logits = next((output[k] for k in _LOGIT_KEYS if k in output), None)
    elif isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        if len(output) == 0:
            raise ValueError("4D feature extractor returned an empty sequence")
        features = output[0]
        logits = output[1] if len(output) > 1 else None
    else:
        features = next((getattr(output, k) for k in _FEATURE_KEYS if hasattr(output, k)), None)
        logits = next((getattr(output, k) for k in _LOGIT_KEYS if hasattr(output, k)), None)

    if not torch.is_tensor(features):
        raise TypeError(
            "4D feature extractor must return a tensor, (features, logits), or "
            "a mapping/object with a 'features' or 'embeddings' tensor"
        )
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim < 2:
        raise ValueError(f"Feature output must include a batch dimension, got {features.shape}")
    features = features.reshape(features.shape[0], -1)

    if logits is not None:
        if not torch.is_tensor(logits):
            raise TypeError("Extractor logits must be a tensor")
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        logits = logits.reshape(logits.shape[0], -1)
        if logits.shape[0] != features.shape[0]:
            raise ValueError(
                f"Feature/logit batch mismatch: {features.shape[0]} vs {logits.shape[0]}"
            )
    return features, logits


class Serialized4DFeatureExtractor(nn.Module):
    """Frozen wrapper around a real serialized 4D fMRI model.

    The checkpoint may be TorchScript or a trusted ``torch.save(nn.Module)``
    artifact.  Plain state dictionaries are rejected because silently guessing
    an architecture would make the reported model provenance false.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        name: str,
        require_logits: bool = False,
        checkpoint_path: Optional[str] = None,
        checkpoint_sha256: Optional[str] = None,
        source_revision: Optional[str] = None,
        adapter_contract: Optional[str] = None,
    ):
        super().__init__()
        self.model = module.eval()
        self.extractor_name = str(name)
        self.require_logits = bool(require_logits)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_revision = source_revision
        self.adapter_contract = adapter_contract
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        name: str,
        require_logits: bool = False,
        expected_sha256: Optional[str] = None,
        source_revision: Optional[str] = None,
        map_location: str | torch.device = "cpu",
    ) -> "Serialized4DFeatureExtractor":
        if str(name).strip().lower() == "brainlm":
            raise RuntimeError(
                "BrainLM cannot be loaded as a generic serialized 4D module. "
                "Training requires the official GitHub BrainLM checkpoint plus "
                "the contextual A424/nonlinear-MNI authority adapter."
            )
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} 4D extractor checkpoint not found: {path}. "
                "Paper-faithful execution requires the real pretrained weights."
            )
        expected_digest = _require_sha256(
            expected_sha256, label=f"{name} checkpoint_sha256"
        )
        immutable_revision = _require_source_revision(
            source_revision, label=f"{name} source_revision"
        )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{name} checkpoint SHA-256 mismatch for {path}: expected "
                f"{expected_digest}, found {actual_digest}"
            )

        module: Optional[nn.Module] = None
        script_error: Optional[Exception] = None
        try:
            module = torch.jit.load(str(path), map_location=map_location)
        except Exception as exc:
            script_error = exc

        if module is None:
            try:
                # Loading a serialized nn.Module can execute code.  This path is
                # intentionally limited to the checkpoint explicitly configured
                # by the user rather than files discovered automatically.
                obj = torch.load(str(path), map_location=map_location, weights_only=False)
            except TypeError:  # PyTorch < 2.0 has no weights_only argument
                obj = torch.load(str(path), map_location=map_location)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load {name} extractor from {path} as TorchScript "
                    f"or a serialized nn.Module (TorchScript error: {script_error})"
                ) from exc
            if isinstance(obj, nn.Module):
                module = obj
            elif isinstance(obj, Mapping) and isinstance(obj.get("model"), nn.Module):
                module = obj["model"]
            else:
                raise TypeError(
                    f"{name} checkpoint {path} contains only weights/state, not a runnable "
                    "4D nn.Module. Export the official model as TorchScript or serialize "
                    "the instantiated module."
                )
        adapter_contract = _validate_adapter_identity(
            module,
            name=name,
            source_revision=immutable_revision,
        )
        return cls(
            module,
            name=name,
            require_logits=require_logits,
            checkpoint_path=str(path.resolve()),
            checkpoint_sha256=actual_digest,
            source_revision=immutable_revision,
            adapter_contract=adapter_contract,
        )

    def train(self, mode: bool = True):
        # The extractor is a fixed reference network even when its parent model
        # is switched into training mode.
        super().train(False)
        self.model.eval()
        return self

    def forward(self, fmri: torch.Tensor):
        output = self.model(fmri)
        features, logits = split_feature_output(output)
        if features.shape[0] != fmri.shape[0]:
            raise ValueError(
                f"{self.extractor_name} returned {features.shape[0]} feature rows "
                f"for an input batch of {fmri.shape[0]}"
            )
        if self.require_logits and logits is None:
            raise RuntimeError(
                f"{self.extractor_name} did not return classification logits. "
                "A trained logits head is required for a mathematically valid "
                "Inception Score; embeddings alone are sufficient only for FID."
            )
        return {"features": features, "logits": logits} if logits is not None else features


def build_pretrained_4d_extractor(
    spec: Optional[Mapping[str, Any]],
    *,
    expected_name: str,
    purpose: str,
    device: torch.device,
    require_logits: bool = False,
) -> Optional[nn.Module]:
    """Build an enabled extractor specification with strict provenance checks."""
    if spec is None or not bool(spec.get("enabled", True)):
        return None

    name = str(spec.get("name", expected_name)).strip().lower()
    if name != expected_name.lower():
        raise ValueError(
            f"{purpose} must use {expected_name} for paper fidelity, got {name!r}"
        )
    if name == "brainlm":
        if require_logits:
            raise ValueError("BrainLM perceptual extraction does not expose class logits")
        return build_official_brainlm_a424_extractor(spec, device=device)
    expected_contract = _MODEL_ADAPTER_CONTRACTS.get(expected_name.lower())
    if spec.get("adapter_contract") != expected_contract:
        raise ValueError(
            f"{purpose} must declare adapter_contract={expected_contract!r}"
        )
    checkpoint = spec.get("checkpoint")
    env_name = str(spec.get("checkpoint_env", "")).strip()
    if not checkpoint and env_name:
        checkpoint = os.environ.get(env_name)
    if not checkpoint:
        raise FileNotFoundError(
            f"No {expected_name} checkpoint configured for {purpose}. Set "
            f"'{env_name}' or the corresponding YAML checkpoint path."
        )
    expected_sha256 = spec.get("checkpoint_sha256")
    sha_environment_name = str(spec.get("checkpoint_sha256_env", "")).strip()
    if not expected_sha256 and sha_environment_name:
        expected_sha256 = os.environ.get(sha_environment_name)
    source_revision = spec.get("source_revision")
    revision_environment_name = str(spec.get("source_revision_env", "")).strip()
    if not source_revision and revision_environment_name:
        source_revision = os.environ.get(revision_environment_name)
    extractor = Serialized4DFeatureExtractor.from_checkpoint(
        str(checkpoint),
        name=expected_name,
        require_logits=require_logits,
        expected_sha256=expected_sha256,
        source_revision=source_revision,
    )
    return extractor.to(device).eval()
