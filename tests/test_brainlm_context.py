from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from data.provenance import canonical_sha256, sha256_file
import models.brainlm_context as brainlm_context
from models.brainlm_context import (
    BRAINLM_CONTEXT_SCHEMA,
    BRAINLM_RUNTIME_MAPPING_SCHEMA,
    CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256,
    OfficialBrainLMA424Extractor,
    _content_identity,
    _expected_official_content_identity,
    _load_official_assets,
    _load_dense_displacement,
    _support_sha256,
    _validate_state_dict_identity,
    validate_configured_brainlm_authority_scope,
    validate_configured_brainlm_identity,
)
from models.losses import PerceptualLoss


class _TinyOfficialVit(torch.nn.Module):
    """Cheap differentiable stand-in for the official encoder's ``vit``."""

    def forward(
        self,
        *,
        signal_vectors: torch.Tensor,
        xyz_vectors: torch.Tensor,
        noise: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        assert signal_vectors.shape[1:] == (424, 200)
        assert xyz_vectors.shape[1:] == (424, 3)
        assert noise.shape[1] == 4240
        assert output_hidden_states is True and return_dict is True
        # One differentiable 256-channel CLS token. Spatially varying temporal
        # samples ensure layer normalization retains a non-zero gradient.
        cls = (
            signal_vectors[:, :256, 0]
            + 0.5 * signal_vectors[:, :256, 37]
            - 0.25 * signal_vectors[:, :256, 113]
        ).unsqueeze(1)
        hidden = tuple(cls * (index + 1.0) + 0.01 * index for index in range(5))
        return SimpleNamespace(hidden_states=hidden)


class _TinyOfficialEncoder(torch.nn.Module):
    """Match the official ``BrainLMForPretraining.vit`` module boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.vit = _TinyOfficialVit()


def _signed(value: dict) -> dict:
    result = dict(value)
    result["record_sha256"] = canonical_sha256(result)
    return result


@pytest.fixture()
def contextual_fixture(tmp_path: Path, monkeypatch):
    shape = (8, 8, 8)
    monkeypatch.setattr(brainlm_context, "A424_SHAPE", shape)
    monkeypatch.setattr(brainlm_context, "A424_AFFINE_RAS_MM", np.eye(4))
    monkeypatch.setattr(brainlm_context, "PADDED_SHAPE", shape)
    monkeypatch.setattr(brainlm_context, "NATIVE_SHAPE", shape)
    monkeypatch.setattr(brainlm_context, "PADDING_BEFORE", (0, 0, 0))
    monkeypatch.setattr(brainlm_context, "PADDING_AFTER", (0, 0, 0))
    monkeypatch.setattr(brainlm_context, "INPUT_FRAMES", 4)

    labels = torch.zeros(shape, dtype=torch.long)
    labels.reshape(-1)[:424] = torch.arange(1, 425)
    coordinates = torch.stack(
        (
            torch.arange(424, dtype=torch.float32),
            torch.arange(424, dtype=torch.float32) * 0.1,
            torch.arange(424, dtype=torch.float32) * -0.1,
        ),
        dim=1,
    )
    displacement_path = tmp_path / "dense_displacement.nii.gz"
    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header.set_intent("vector")
    displacement = np.zeros((*shape, 3), dtype=np.float32)
    image = nib.Nifti1Image(displacement, np.eye(4), header)
    image.set_qform(np.eye(4), code=4)
    image.set_sform(np.eye(4), code=4)
    nib.save(image, displacement_path)

    support = torch.ones(1, 1, *shape)
    scan_id = "B100_001"
    t1_sha = "1" * 64
    mask_sha = "2" * 64
    artifact = {
        "path": str(displacement_path),
        "sha256": sha256_file(displacement_path),
        "size_bytes": displacement_path.stat().st_size,
        "shape": [*shape, 3],
        "affine_ras_mm": np.eye(4).tolist(),
        "dtype": "float32",
        "nifti_intent": "vector",
        "component_order": ["right", "anterior", "superior"],
        "runtime_mapping_schema": BRAINLM_RUNTIME_MAPPING_SCHEMA,
    }
    record = _signed(
        {
            "schema": brainlm_context.BRAINLM_AUTHORITY_SCAN_SCHEMA,
            "scan_id": scan_id,
            "role": "train",
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "prepared_t1_sha256": t1_sha,
            "prepared_mask_sha256": mask_sha,
            "support_tensor_sha256": _support_sha256(support[0, 0]),
            "support_foreground_voxels": int(support.sum().item()),
            "padded_shape": list(shape),
            "native_shape": list(shape),
            "padding_before": [0, 0, 0],
            "padding_after": [0, 0, 0],
            "padded_affine_ras_mm": np.eye(4).tolist(),
            "native_affine_ras_mm": np.eye(4).tolist(),
            "human_review": {
                "decision": "PASS",
                "structural_only": True,
                "prediction_or_functional_target_used": False,
                "reviewed_displacement_sha256": artifact["sha256"],
                "review_record_sha256": "3" * 64,
            },
            "a424_dense_displacement_artifact": artifact,
        }
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}", encoding="utf-8")
    extractor = OfficialBrainLMA424Extractor(
        _TinyOfficialEncoder(),
        labels,
        coordinates,
        authority_records={scan_id: record},
        artifact_identity={"schema": "test-official-assets"},
        authority_identity={
            "schema": brainlm_context.BRAINLM_AUTHORITY_SCHEMA,
            "path": str(authority_path),
            "file_sha256": "a" * 64,
            "authority_record_sha256": "b" * 64,
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "record_sha256": "0" * 64,
        },
        resample_chunk_frames=2,
    )
    context_record = {
        "schema": brainlm_context.BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA,
        "scan_id": scan_id,
        "role": "train",
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "prepared_t1_sha256": t1_sha,
        "prepared_mask_sha256": mask_sha,
        "padded_t1_artifact_descriptor_sha256": "4" * 64,
        "padded_mask_artifact_descriptor_sha256": "5" * 64,
        "cache_metadata_artifact_descriptor_sha256": "6" * 64,
        "structural_source_identity_sha256": "7" * 64,
        "native_alignment_authority_sha256": "8" * 64,
        "target_artifact_identity_sha256": "9" * 64,
        "padded_shape": list(shape),
        "native_shape": list(shape),
        "padding_before": [0, 0, 0],
        "padding_after": [0, 0, 0],
        "padded_affine_ras_mm": np.eye(4).tolist(),
        "support_tensor_sha256": _support_sha256(support[0, 0]),
        "support_foreground_voxels": int(support.sum().item()),
    }
    context_record_sha = canonical_sha256(context_record)
    context = {
        "schema": BRAINLM_CONTEXT_SCHEMA,
        "brainlm_perceptual_identity_sha256": (
            extractor.connect4_artifact_identity_sha256
        ),
        "authority_file_sha256": "a" * 64,
        "authority_content_record_sha256": extractor.connect4_artifact_identity[
            "mni_authority"
        ]["content_record_sha256"],
        "authority_record_sha256": "b" * 64,
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "run_artifact_identity_sha256": "c" * 64,
        "cohort_admission_identity_record_sha256": "d" * 64,
        "dataset_state_record_sha256": "e" * 64,
        "dataset_state_sha256": "f" * 64,
        "structural_artifact_identities_sha256": "0" * 64,
        "scan_ids": [scan_id],
        "roles": ["train"],
        "prepared_t1_sha256": [t1_sha],
        "prepared_mask_sha256": [mask_sha],
        "padded_t1_artifact_descriptor_sha256": ["4" * 64],
        "padded_mask_artifact_descriptor_sha256": ["5" * 64],
        "cache_metadata_artifact_descriptor_sha256": ["6" * 64],
        "structural_source_identity_sha256": ["7" * 64],
        "native_alignment_authority_sha256": ["8" * 64],
        "target_artifact_identity_sha256": ["9" * 64],
        "support_tensor_sha256": [_support_sha256(support[0, 0])],
        "support_foreground_voxels": [int(support.sum().item())],
        "admitted_scan_context_record_sha256": [context_record_sha],
        "admitted_scan_context_schema": [
            brainlm_context.BRAINLM_ADMITTED_SCAN_CONTEXT_SCHEMA
        ],
        "source_affine": torch.eye(4).unsqueeze(0),
        "admitted_padded_affine": torch.eye(4, dtype=torch.float64).unsqueeze(0),
        "brain_support": support,
        "padded_shape": torch.tensor([list(shape)]),
        "crop_before": torch.tensor([[0, 0, 0]]),
        "crop_after": torch.tensor([[0, 0, 0]]),
        "native_shape": torch.tensor([list(shape)]),
    }
    spatial = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    spatial = spatial / spatial.max()
    frames = []
    for time_index in range(4):
        time = float(time_index)
        frames.append(
            0.35
            + 0.08 * spatial
            + 0.015 * time * (0.2 + spatial)
            + 0.004 * time * time * spatial.square()
        )
    target = torch.stack(frames).unsqueeze(0).unsqueeze(0)
    prediction = target.clone()
    prediction[:, :, 1] += 0.02 * spatial
    prediction[:, :, 2] -= 0.015 * spatial.square()
    return extractor, context, prediction, target, displacement_path


def test_contextual_perceptual_loss_has_finite_prediction_gradient_only(
    contextual_fixture,
) -> None:
    extractor, context, prediction, target, _ = contextual_fixture
    prediction = prediction.requires_grad_(True)
    target = target.requires_grad_(True)
    preflight = extractor.prepare_context_identity(context)
    perceptual = PerceptualLoss(extractor)
    loss = perceptual(
        prediction,
        target,
        mask=context["brain_support"],
        brainlm_context=context,
    )
    assert loss.isfinite() and loss.item() > 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0.0
    assert target.grad is None
    assert extractor.last_projection_qc is not None
    assert extractor.last_projection_qc["passed"] is True
    assert extractor.last_projection_qc["nonlinear_dense_pull_field_applied"] is True
    assert extractor.last_projection_qc["prediction_target_context_shared"] is True
    assert extractor.last_input_role == "target"
    assert perceptual.last_context_identity == preflight
    assert extractor.last_context_identity == preflight


def _scope_fixture_authority(
    tmp_path: Path,
    base_record: dict,
    scan_roles: list[tuple[str, str]],
    *,
    ordered_scan_ids: list[str] | None = None,
) -> tuple[Path, str]:
    records = []
    for scan_id, role in scan_roles:
        record = deepcopy(base_record)
        record["scan_id"] = scan_id
        record["role"] = role
        record.pop("record_sha256", None)
        records.append(_signed(record))
    ordered = (
        [scan_id for scan_id, _role in scan_roles]
        if ordered_scan_ids is None
        else ordered_scan_ids
    )
    authority = _signed(
        {
            "schema": brainlm_context.BRAINLM_AUTHORITY_SCHEMA,
            "status": "AUTHORIZED_AFTER_EXPLICIT_HUMAN_STRUCTURAL_REVIEW",
            "structural_only": True,
            "functional_target_used_for_registration": False,
            "sealed_target_voxel_data_opened": False,
            "authorized_roles": ["train", "development-validation"],
            "authorizes_prediction_emission": False,
            "authorizes_final_evaluation": False,
            "native_preprocessing_source_sha256": (
                CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
            ),
            "official_brainlm_artifacts": {
                "repository": brainlm_context.OFFICIAL_BRAINLM_REPOSITORY,
                "source_revision": brainlm_context.OFFICIAL_BRAINLM_SOURCE_REVISION,
                "checkpoint_sha256": (
                    brainlm_context.OFFICIAL_BRAINLM_CHECKPOINT_SHA256
                ),
                "config_sha256": brainlm_context.OFFICIAL_BRAINLM_CONFIG_SHA256,
                "a424_atlas_sha256": brainlm_context.OFFICIAL_A424_ATLAS_SHA256,
                "a424_coordinates_sha256": (
                    brainlm_context.OFFICIAL_A424_COORDINATES_SHA256
                ),
            },
            "source_grid": {
                "architecture_shape": list(brainlm_context.PADDED_SHAPE),
                "native_shape": list(brainlm_context.NATIVE_SHAPE),
                "padding_before": list(brainlm_context.PADDING_BEFORE),
                "padding_after": list(brainlm_context.PADDING_AFTER),
                "frames": brainlm_context.INPUT_FRAMES,
                "tr_seconds": brainlm_context.INPUT_TR_SECONDS,
            },
            "runtime_mapping_contract": brainlm_context._expected_mapping_contract(),
            "scans": records,
            "ordered_scan_ids": ordered,
            "ordered_scan_ids_sha256": canonical_sha256(ordered),
            "scan_count": len(records),
        }
    )
    path = tmp_path / f"scope-authority-{len(scan_roles)}.json"
    path.write_text(json.dumps(authority), encoding="utf-8")
    return path, sha256_file(path)


def _scope_split() -> tuple[dict, dict[str, str]]:
    partitions = {
        "train": [{"scan_id": "B100_001"}],
        "validation": [{"scan_id": "B100_002"}],
        "test": [{"scan_id": "B100_003"}],
    }
    split = {
        "format": "connect4_split_identity_v2",
        "sha256": "a" * 64,
        "partitions": partitions,
    }
    return split, {
        "B100_001": "train",
        "B100_002": "development-validation",
    }


def test_brainlm_authority_scope_is_exact_and_split_bound(
    contextual_fixture, tmp_path: Path
) -> None:
    extractor, _, _, _, _ = contextual_fixture
    base_record = extractor.authority_records["B100_001"]
    authority_path, digest = _scope_fixture_authority(
        tmp_path,
        base_record,
        [("B100_001", "train"), ("B100_002", "development-validation")],
    )
    split, roles = _scope_split()
    scope = validate_configured_brainlm_authority_scope(
        {"authority": str(authority_path), "authority_sha256": digest},
        expected_scan_roles=roles,
        split_identity=split,
        expected_scan_count=2,
    )
    assert scope["authorized_scan_count"] == 2
    assert scope["sealed_scan_count"] == 1
    assert scope["sealed_scans_authorized"] is False
    assert scope["functional_target_bytes_opened"] is False
    assert scope["split_identity_sha256"] == canonical_sha256(split)


@pytest.mark.parametrize(
    "scan_roles",
    [
        [("B100_001", "train")],
        [
            ("B100_001", "train"),
            ("B100_002", "development-validation"),
            ("B100_004", "train"),
        ],
        [("B100_001", "train"), ("B100_002", "train")],
        [
            ("B100_001", "train"),
            ("B100_002", "development-validation"),
            ("B100_003", "train"),
        ],
    ],
)
def test_brainlm_authority_scope_rejects_subset_superset_role_and_sealed_scan(
    contextual_fixture, tmp_path: Path, scan_roles
) -> None:
    extractor, _, _, _, _ = contextual_fixture
    authority_path, digest = _scope_fixture_authority(
        tmp_path, extractor.authority_records["B100_001"], scan_roles
    )
    split, roles = _scope_split()
    with pytest.raises(RuntimeError, match="scope differs|role differs"):
        validate_configured_brainlm_authority_scope(
            {"authority": str(authority_path), "authority_sha256": digest},
            expected_scan_roles=roles,
            split_identity=split,
            expected_scan_count=2,
        )


def test_brainlm_authority_scope_rejects_duplicate_and_split_role_drift(
    contextual_fixture, tmp_path: Path
) -> None:
    extractor, _, _, _, _ = contextual_fixture
    base_record = extractor.authority_records["B100_001"]
    authority_path, digest = _scope_fixture_authority(
        tmp_path,
        base_record,
        [("B100_001", "train"), ("B100_002", "development-validation")],
        ordered_scan_ids=["B100_001", "B100_001"],
    )
    split, roles = _scope_split()
    with pytest.raises(RuntimeError, match="scan contract differs|top-level contract"):
        validate_configured_brainlm_authority_scope(
            {"authority": str(authority_path), "authority_sha256": digest},
            expected_scan_roles=roles,
            split_identity=split,
            expected_scan_count=2,
        )

    split["partitions"]["train"], split["partitions"]["validation"] = (
        split["partitions"]["validation"],
        split["partitions"]["train"],
    )
    with pytest.raises(RuntimeError, match="role map differs"):
        validate_configured_brainlm_authority_scope(
            {"authority": str(authority_path), "authority_sha256": digest},
            expected_scan_roles=roles,
            split_identity=split,
            expected_scan_count=2,
        )


def test_contextual_perceptual_loss_fails_closed_without_context(
    contextual_fixture,
) -> None:
    extractor, _, prediction, target, _ = contextual_fixture
    with pytest.raises(RuntimeError, match="requires scan IDs"):
        PerceptualLoss(extractor)(prediction, target)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("authority_file_sha256", "f" * 64, "another MNI authority"),
        ("roles", ["sealed-test"], "role differs"),
        ("prepared_mask_sha256", ["9" * 64], "structural hashes differ"),
        ("crop_before", torch.tensor([[1, 0, 0]]), "crop-before differs"),
    ],
)
def test_context_identity_and_role_tampering_fail_closed(
    contextual_fixture,
    field,
    replacement,
    message,
) -> None:
    extractor, context, prediction, _, _ = contextual_fixture
    changed = dict(context)
    changed[field] = replacement
    with pytest.raises(RuntimeError, match=message):
        extractor(prediction, context=changed, input_role="prediction")


def test_historical_native_context_is_categorically_rejected(
    contextual_fixture,
) -> None:
    extractor, context, prediction, _, _ = contextual_fixture
    changed = dict(context)
    changed["native_preprocessing_source_sha256"] = (
        "8beee80eddc3da357229fdbc8ba57b7eab286cbdb34bf2941490f534372e631a"
    )
    with pytest.raises(RuntimeError, match="historical native source"):
        extractor(prediction, context=changed, input_role="prediction")


def test_support_and_displacement_byte_drift_fail_closed(
    contextual_fixture,
) -> None:
    extractor, context, prediction, _, displacement_path = contextual_fixture
    changed = dict(context)
    changed["brain_support"] = context["brain_support"].clone()
    changed["brain_support"][0, 0, 0, 0, 0] = 0
    with pytest.raises(RuntimeError, match="support hash differs"):
        extractor(prediction, context=changed, input_role="prediction")

    # The authority remains unchanged while the artifact bytes drift.
    displacement_path.write_bytes(displacement_path.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="exact expected regular file|SHA-256"):
        extractor(prediction, context=context, input_role="prediction")


def test_authority_support_count_drift_fails_closed(contextual_fixture) -> None:
    extractor, context, prediction, _, _ = contextual_fixture
    extractor.authority_records["B100_001"]["support_foreground_voxels"] -= 1
    with pytest.raises(RuntimeError, match="support hash differs"):
        extractor(prediction, context=context, input_role="prediction")


def test_dense_displacement_uses_held_authenticated_bytes_after_path_swap(
    contextual_fixture, monkeypatch
) -> None:
    extractor, _, _, _, displacement_path = contextual_fixture
    artifact = extractor.authority_records["B100_001"][
        "a424_dense_displacement_artifact"
    ]
    original = brainlm_context._read_regular_file

    def read_then_swap(*args, **kwargs):
        result = original(*args, **kwargs)
        displacement_path.write_bytes(b"not-a-nifti-after-authentication")
        return result

    monkeypatch.setattr(brainlm_context, "_read_regular_file", read_then_swap)
    displacement, evidence = _load_dense_displacement(
        Path(extractor.authority_identity["path"]), artifact
    )
    assert tuple(displacement.shape) == (*brainlm_context.A424_SHAPE, 3)
    assert torch.count_nonzero(displacement).item() == 0
    assert evidence["sha256"] == artifact["sha256"]


def test_input_role_cannot_be_missing_or_reused(contextual_fixture) -> None:
    extractor, context, prediction, _, _ = contextual_fixture
    with pytest.raises(RuntimeError, match="input_role"):
        extractor(prediction, context=context)
    with pytest.raises(RuntimeError, match="input_role"):
        extractor(prediction, context=context, input_role="real")


def test_generic_image_vitmae_state_is_not_brainlm() -> None:
    generic = {
        "vit.embeddings.patch_embeddings.projection.weight": torch.zeros(
            768, 3, 16, 16
        )
    }
    with pytest.raises(RuntimeError, match="generic image ViT-MAE"):
        _validate_state_dict_identity(generic)


def test_content_identity_recomputes_hash_after_removing_paths() -> None:
    first = {
        "schema": "example",
        "artifact": {"path": "/stage/a/model.bin", "sha256": "1" * 64},
    }
    first["record_sha256"] = canonical_sha256(first)
    second = {
        "schema": "example",
        "artifact": {"path": "/another/stage/model.bin", "sha256": "1" * 64},
    }
    second["record_sha256"] = canonical_sha256(second)
    assert first["record_sha256"] != second["record_sha256"]
    assert _content_identity(first) == _content_identity(second)
    content = _content_identity(first)
    claimed = content.pop("content_record_sha256")
    assert claimed == canonical_sha256(content)


def _valid_recorded_identity() -> dict:
    authority = {
        "schema": brainlm_context.BRAINLM_AUTHORITY_SCHEMA,
        "file_sha256": "4" * 64,
        "size_bytes": 1234,
        "authority_record_sha256": "5" * 64,
        "native_preprocessing_source_sha256": (
            CURRENT_NATIVE_PREPROCESSING_SOURCE_SHA256
        ),
        "ordered_scan_ids": ["B100_001"],
        "ordered_scan_ids_sha256": canonical_sha256(["B100_001"]),
        "scan_count": 1,
        "all_scans_explicitly_structurally_reviewed": True,
        "dense_nonlinear_pull_field_required": True,
        "affine_only_fallback_forbidden": True,
    }
    authority["content_record_sha256"] = canonical_sha256(authority)
    identity = {
        "schema": "connect4-brainlm-a424-perceptual-identity-v2",
        "name": "brainlm",
        "model_artifacts": _expected_official_content_identity(),
        "mni_authority": authority,
        "adapter_contract": brainlm_context.BRAINLM_CONTEXT_ADAPTER_CONTRACT,
        "feature_contract": brainlm_context.BRAINLM_FEATURE_SCHEMA,
        "feature_layers": (
            "last four encoder hidden states, CLS token, layer-normalized"
        ),
        "deterministic_identical_prediction_target_mask": True,
        "input_frames": brainlm_context.INPUT_FRAMES,
        "input_tr_seconds": brainlm_context.INPUT_TR_SECONDS,
        "temporal_adapter": "linear 128 to 200 samples, align_corners=True",
        "scaler_adapter": "per-scan/per-parcel median-IQR then clamp [-6,6]",
        "normalization_and_tr_match_pretraining_exactly": False,
        "pretraining_domain_exactness_claimed": False,
        "domain_shift_reason": (
            "TR=3 s differs from BrainLM training acquisitions and public "
            "population parcel median/IQR vectors are unavailable"
        ),
        "projection": brainlm_context._expected_mapping_contract(),
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return identity


def test_lightweight_recorded_identity_validation_rejects_drift() -> None:
    identity = _valid_recorded_identity()
    assert validate_configured_brainlm_identity(identity) == identity
    changed = dict(identity)
    changed["model_artifacts"] = dict(identity["model_artifacts"])
    changed["model_artifacts"]["checkpoint"] = dict(
        identity["model_artifacts"]["checkpoint"]
    )
    changed["model_artifacts"]["checkpoint"]["sha256"] = "9" * 64
    changed["record_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="official artifact identity"):
        validate_configured_brainlm_identity(changed)


@pytest.mark.parametrize(
    ("field", "old_value"),
    [
        (
            "purpose",
            "connect4-official-brainlm-eded39c86c27-immutable-source",
        ),
        (
            "published_source_root",
            "/srv/connect4/connect4_validation_20260829/"
            "official_brainlm_eded39c86c27e03f5ead1d6a14311e92d1305e5",
        ),
        (
            "sha256",
            "c70b6dcde2a14181b6b969b726f47dd1177bae8711a32d3f6d060fbff9189e7f",
        ),
    ],
)
def test_resigned_revoked_publication_identity_is_rejected(field, old_value) -> None:
    identity = _valid_recorded_identity()
    model = dict(identity["model_artifacts"])
    publication = dict(model["immutable_publication"])
    publication[field] = old_value
    publication_without_digest = dict(publication)
    publication_without_digest.pop("content_record_sha256", None)
    publication["content_record_sha256"] = canonical_sha256(
        publication_without_digest
    )
    model["immutable_publication"] = publication
    identity["model_artifacts"] = model
    identity["record_sha256"] = canonical_sha256(
        {key: value for key, value in identity.items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="official artifact identity"):
        validate_configured_brainlm_identity(identity)


def test_official_source_root_leaf_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="non-symlink directory"):
        _load_official_assets(
            {
                "source_revision": brainlm_context.OFFICIAL_BRAINLM_SOURCE_REVISION,
                "source_root": str(linked),
            }
        )


def test_official_source_root_byte_clone_is_rejected(tmp_path: Path) -> None:
    byte_clone = tmp_path / "tracked51-byte-clone"
    byte_clone.mkdir()
    with pytest.raises(RuntimeError, match="exclusive clean tracked51_v2"):
        _load_official_assets(
            {
                "source_revision": brainlm_context.OFFICIAL_BRAINLM_SOURCE_REVISION,
                "source_root": str(byte_clone),
            }
        )
