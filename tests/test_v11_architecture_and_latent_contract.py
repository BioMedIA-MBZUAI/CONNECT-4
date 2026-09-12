import copy
import hashlib
import json

import pytest
import torch

from architecture_contract import (
    CANONICAL_ROI_LABEL_IDS,
    CANONICAL_ROI_LABEL_TO_CHANNEL,
    CANONICAL_ROI_MAPPING_SHA256,
    CANONICAL_ROI_SPECS,
    REJECTED_LEGACY_CHECKPOINT_FORMATS,
    REJECTED_SYNTHESIS_LAUNCH_SCHEMAS,
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    SYNTHESIS_LAUNCH_ADMISSION_SCHEMA,
    TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE,
    TOKEN_CODEC_ORTHONORMAL_ATOL,
    TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE,
    TOKEN_CODEC_STATE_VERSION,
    TOKEN_LATENT_NORMALIZATION_CONTRACT,
    TRAINING_CHECKPOINT_FORMAT,
    require_current_synthesis_architecture,
    require_current_synthesis_launch_schema,
    synthesis_architecture_contract,
)
from data.dataset_precomputed import (
    RECOVERY_TARGET_VALIDITY_MASK_CONTRACT,
    Connect4PrecomputedDataset,
)
from data.protocol import RUN_ARTIFACT_IDENTITY_SCHEMA, TARGET_VALIDITY_MASK_CONTRACT
from models.connect4 import (
    _TARGET_BLIND_SAMPLER_KEYS,
    _require_batch_target_validity_contract,
    _require_canonical_roi_mapping,
    _required_target_validity_contract,
)
from models.dit4d_temporal import (
    DiT4DTemporal,
    TOKEN_LATENT_DIFFUSION_DOMAIN,
)
from training.train import validate_resume_checkpoint


EXPECTED_ROI_LABEL_IDS = (
    2,
    3,
    4,
    5,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    24,
    26,
    28,
    41,
    42,
    43,
    44,
    46,
    47,
    49,
    50,
    51,
    52,
    53,
    54,
    58,
    60,
)


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tiny_dit(*, patch_size=(2, 2, 2), hidden_size=4):
    return DiT4DTemporal(
        input_size=(2, 2, 2),
        in_channels=1,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        input_dim=hidden_size,
        t1_token_dim=hidden_size,
        num_temporal_frames=2,
        temporal_patch_size=1,
        num_diffusion_steps=4,
        spatial_window_size=(8, 8, 8),
        diffusion_domain=TOKEN_LATENT_DIFFUSION_DOMAIN,
    )


def test_v18_checkpoint_architecture_launch_and_run_artifact_identity_are_closed():
    contract = synthesis_architecture_contract()

    assert TRAINING_CHECKPOINT_FORMAT == "connect4_iteration_exact_v18"
    assert {
        "connect4_iteration_exact_v10",
        "connect4_iteration_exact_v11",
        "connect4_iteration_exact_v12",
        "connect4_iteration_exact_v13",
        "connect4_iteration_exact_v14",
        "connect4_iteration_exact_v15",
        "connect4_iteration_exact_v16",
        "connect4_iteration_exact_v17",
    } <= REJECTED_LEGACY_CHECKPOINT_FORMATS
    assert SYNTHESIS_ARCHITECTURE_SCHEMA == (
        "connect4-synthesis-architecture-contract-v18"
    )
    assert SYNTHESIS_LAUNCH_ADMISSION_SCHEMA == (
        "connect4-synthesis-v18-launch-admission-v1"
    )
    assert {
        "connect4-synthesis-v10-launch-admission-v1",
        "connect4-synthesis-v11-launch-admission-v1",
        "connect4-synthesis-v12-launch-admission-v1",
        "connect4-synthesis-v13-launch-admission-v1",
        "connect4-synthesis-v14-launch-admission-v1",
        "connect4-synthesis-v15-launch-admission-v1",
        "connect4-synthesis-v16-launch-admission-v1",
        "connect4-synthesis-v17-launch-admission-v1",
    } <= REJECTED_SYNTHESIS_LAUNCH_SCHEMAS
    assert RUN_ARTIFACT_IDENTITY_SCHEMA == "connect4_run_artifacts_v6"
    assert contract["run_artifact_identity_contract"] == RUN_ARTIFACT_IDENTITY_SCHEMA
    assert "connect4_run_artifacts_v5" in contract[
        "rejected_run_artifact_identity_contracts"
    ]
    assert SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256 == (
        "eb4c3feae75a44378360a3d5578f3cf682a61cc3adfab9d548ec2418275e4b9d"
    )
    assert SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256 == _canonical_sha256(contract)
    assert require_current_synthesis_architecture(
        contract, SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
    ) == contract
    assert (
        require_current_synthesis_launch_schema(SYNTHESIS_LAUNCH_ADMISSION_SCHEMA)
        == SYNTHESIS_LAUNCH_ADMISSION_SCHEMA
    )

    for legacy_version in (10, 11, 12, 13, 14, 15, 16, 17):
        legacy = copy.deepcopy(contract)
        legacy["schema"] = (
            f"connect4-synthesis-architecture-contract-v{legacy_version}"
        )
        legacy["checkpoint_format"] = (
            f"connect4_iteration_exact_v{legacy_version}"
        )
        with pytest.raises(RuntimeError, match="pre-v18"):
            require_current_synthesis_architecture(
                legacy, _canonical_sha256(legacy)
            )
        with pytest.raises(RuntimeError, match="pre-V18"):
            require_current_synthesis_launch_schema(
                f"connect4-synthesis-v{legacy_version}-launch-admission-v1"
            )


@pytest.mark.parametrize("legacy_version", [10, 11, 12, 13, 14, 15, 16, 17])
def test_executable_resume_gate_rejects_pre_v18_before_loading_any_state(
    legacy_version,
):
    with pytest.raises(RuntimeError, match="cannot resume"):
        validate_resume_checkpoint(
            {"format": f"connect4_iteration_exact_v{legacy_version}"},
            model=None,
            optimizer=None,
            scaler=None,
            config={},
            split_identity={},
            artifact_identity={},
            runtime_identity={},
            cohort_admission_identity={},
            validation_shard_contract={},
            world_size=1,
            rank=0,
            batches_per_epoch=1,
            grad_accum_steps=1,
        )


def test_v18_contract_closes_two_mask_and_latent_projection_semantics():
    contract = synthesis_architecture_contract()

    assert contract["token_latent_normalization_contract"] == (
        TOKEN_LATENT_NORMALIZATION_CONTRACT
    )
    assert TOKEN_CODEC_STATE_VERSION == 18
    assert contract["token_codec_state_version"] == TOKEN_CODEC_STATE_VERSION
    assert contract["token_codec_raw_detail_minimum_singular_value"] == (
        TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
    )
    assert contract["token_codec_orthonormality_runtime_atol"] == (
        TOKEN_CODEC_ORTHONORMAL_ATOL
    )
    assert contract["token_latent_scale_interpretation"] == (
        "orthonormal-coordinate-projection-with-fixed-dc-amplitude"
    )
    assert contract["token_codec_encoder_bias"] == "forbidden-no-parameter"
    assert contract["token_codec_decoder_bias"] == "forbidden-no-parameter"
    assert contract["token_to_patch_decoder_operator_norm"] == 1.0
    assert not contract["token_codec_full_patch_invertibility_when_compressed_claimed"]
    assert not contract["encoder_decoder_inverse_scale_symmetry_accepted"]
    assert contract["token_codec_mask_source"] == (
        "paired-target-validity-mask-training-only"
    )
    assert contract["token_diffusion_mask_source"] == (
        "paired-target-validity-mask-training-only"
    )
    assert not contract["target_validity_mask_without_paired_target_accepted"]
    assert contract["inference_output_mask_source"] == (
        "target-independent-authenticated-structural-brain-mask"
    )


def test_canonical_roi_names_ids_order_and_digest_are_exact():
    assert len(CANONICAL_ROI_SPECS) == 32
    assert tuple(Connect4PrecomputedDataset.ROI_SPECS) == CANONICAL_ROI_SPECS
    assert CANONICAL_ROI_LABEL_IDS == EXPECTED_ROI_LABEL_IDS
    assert dict(CANONICAL_ROI_LABEL_TO_CHANNEL) == {
        label_id: channel for channel, label_id in enumerate(EXPECTED_ROI_LABEL_IDS)
    }
    records = [
        {"channel_index": channel, "label_id": label_id, "name": name}
        for channel, (name, label_id) in enumerate(CANONICAL_ROI_SPECS)
    ]
    assert CANONICAL_ROI_MAPPING_SHA256 == _canonical_sha256(records)
    assert CANONICAL_ROI_MAPPING_SHA256 == (
        "dad0618b1f82af5ea542c0f2c27941c59a107e91bc3692a94c154372ac99caba"
    )
    contract = synthesis_architecture_contract()
    assert contract["canonical_roi_records"] == records
    assert contract["canonical_roi_mapping_sha256"] == CANONICAL_ROI_MAPPING_SHA256

    assert _require_canonical_roi_mapping(
        dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    ) == dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    swapped = dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(ValueError, match="canonical 32-ROI"):
        _require_canonical_roi_mapping(swapped)


def test_target_validity_is_protocol_specific_and_sampler_stays_target_blind():
    expected = dict(TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE)

    assert expected == {
        "paper-a4-adni-fmriprep-v1": TARGET_VALIDITY_MASK_CONTRACT,
        "a4-native-recovery-v1": RECOVERY_TARGET_VALIDITY_MASK_CONTRACT,
    }
    for profile, contract in expected.items():
        assert _required_target_validity_contract(profile) == contract
        assert _require_batch_target_validity_contract(
            [contract, contract],
            batch_size=2,
            protocol_profile=profile,
            required_contract=contract,
        ) == (contract, contract)
        wrong = (
            RECOVERY_TARGET_VALIDITY_MASK_CONTRACT
            if contract == TARGET_VALIDITY_MASK_CONTRACT
            else TARGET_VALIDITY_MASK_CONTRACT
        )
        with pytest.raises(ValueError, match="differs for protocol profile"):
            _require_batch_target_validity_contract(
                [wrong, wrong],
                batch_size=2,
                protocol_profile=profile,
                required_contract=contract,
            )
    assert _required_target_validity_contract("synthetic-unit-profile") is None

    assert _TARGET_BLIND_SAMPLER_KEYS == (
        "image_nodes",
        "mask_nodes",
        "roi_nodes",
        "dwi_matrix",
        "structure_to_roi_idx",
        "patch_distributions",
        "t1w",
        "brain_mask",
        "roi_masks",
        "scan_id",
    )
    assert "fmri" not in _TARGET_BLIND_SAMPLER_KEYS
    assert "target_validity_mask" not in _TARGET_BLIND_SAMPLER_KEYS
    assert "target_validity_mask_contract" not in _TARGET_BLIND_SAMPLER_KEYS


def test_v18_effective_codec_is_fp32_dc_fixed_orthonormal_and_bias_free():
    torch.manual_seed(91)
    model = _tiny_dit()
    effective = model.effective_token_codec_encoder()

    assert model.token_latent_normalization_contract == (
        TOKEN_LATENT_NORMALIZATION_CONTRACT
    )
    assert effective.dtype == torch.float32
    assert torch.allclose(
        effective @ effective.T,
        torch.eye(model.hidden_size),
        atol=TOKEN_CODEC_ORTHONORMAL_ATOL,
        rtol=0.0,
    )
    expected_dc = torch.full(
        (model.patch_value_dim,),
        1.0 / model.patch_value_dim**0.5,
    )
    assert torch.equal(effective[0], expected_dc)
    assert not hasattr(model, "clean_token_encoder")
    assert not hasattr(model, "token_decoder")
    clean = torch.randn(2, 1, 2, 2, 2, 2)
    latent = model.encode_clean_volume(clean)
    latent.square().mean().backward()
    gradient = model.token_codec_raw_detail_rows.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_v18_orthonormal_codec_blocks_encoder_decoder_inverse_scale_shortcut():
    torch.manual_seed(92)
    model = _tiny_dit().eval()
    clean = torch.randn(2, 1, 2, 2, 2, 2)
    with torch.no_grad():
        baseline_encoder = model.effective_token_codec_encoder().clone()
        baseline_latent = model.encode_clean_volume(clean)
        baseline_reconstruction = model.decode_tokens(baseline_latent)
        model.token_codec_raw_detail_rows.mul_(0.1)
        scaled_encoder = model.effective_token_codec_encoder()
        scaled_latent = model.encode_clean_volume(clean)
        scaled_reconstruction = model.decode_tokens(scaled_latent)

    assert torch.allclose(scaled_encoder, baseline_encoder, atol=2e-6, rtol=2e-6)
    assert torch.allclose(scaled_latent, baseline_latent, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        scaled_reconstruction,
        baseline_reconstruction,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        scaled_latent.square().mean().sqrt(),
        baseline_latent.square().mean().sqrt(),
        atol=2e-6,
        rtol=2e-6,
    )

    epsilon = 1e-3
    attenuated = model.decode_tokens(epsilon * scaled_latent)
    assert torch.allclose(
        attenuated,
        epsilon * scaled_reconstruction,
        atol=1e-7,
        rtol=2e-5,
    )
    attacked_state = copy.deepcopy(model.state_dict())
    attacked_state["token_decoder.weight"] = scaled_encoder.T / epsilon
    with pytest.raises(RuntimeError, match="legacy independent token-codec"):
        _tiny_dit().load_state_dict(attacked_state, strict=False)


def test_v18_square_codec_has_constructive_roundtrip_without_losing_amplitude():
    model = _tiny_dit(patch_size=(2, 2, 2), hidden_size=8).eval()
    clean = torch.tensor(
        [[[[[[0.10, 0.20], [0.30, 0.40]], [[0.50, 0.60], [0.70, 0.80]]],
           [[[0.15, 0.25], [0.35, 0.45]], [[0.55, 0.65], [0.75, 0.85]]]]]],
        dtype=torch.float32,
    )
    reconstructed = model.decode_tokens(model.encode_clean_volume(clean))

    assert torch.allclose(reconstructed, clean, atol=4e-6, rtol=4e-6)
    assert torch.allclose(reconstructed.mean(), clean.mean(), atol=1e-7, rtol=1e-7)
    assert torch.allclose(
        reconstructed.amax() - reconstructed.amin(),
        clean.amax() - clean.amin(),
        atol=4e-6,
        rtol=4e-6,
    )


def test_valid_v18_checkpoint_state_roundtrips_without_legacy_decoder_state():
    torch.manual_seed(93)
    source = _tiny_dit().eval()
    clean = torch.randn(1, 1, 2, 2, 2, 2)
    expected = source.encode_clean_volume(clean)

    destination = _tiny_dit().eval()
    incompatible = destination.load_state_dict(
        copy.deepcopy(source.state_dict()), strict=True
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert not any(
        "clean_token_encoder" in key or "token_decoder" in key
        for key in destination.state_dict()
    )
    assert destination.validate_token_codec_invariants()["state_version"] == 18
    assert torch.equal(destination.encode_clean_volume(clean), expected)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-state-version",
        "v11-state-version",
        "v12-state-version",
        "v13-state-version",
        "v14-state-version",
        "v15-state-version",
        "v16-state-version",
        "v17-state-version",
        "legacy-bias",
        "rank-deficient-detail",
    ],
)
def test_v18_checkpoint_state_rejects_legacy_or_invalid_codec_semantics(failure):
    source = _tiny_dit()
    state = copy.deepcopy(source.state_dict())
    if failure == "missing-state-version":
        state.pop("_token_codec_state_version")
        match = "V18 DiT checkpoint state token-codec state marker"
    elif failure == "v11-state-version":
        state["_token_codec_state_version"].fill_(11)
        match = "not V18"
    elif failure == "v12-state-version":
        state["_token_codec_state_version"].fill_(12)
        match = "not V18"
    elif failure == "v13-state-version":
        state["_token_codec_state_version"].fill_(13)
        match = "not V18"
    elif failure == "v14-state-version":
        state["_token_codec_state_version"].fill_(14)
        match = "not V18"
    elif failure == "v15-state-version":
        state["_token_codec_state_version"].fill_(15)
        match = "not V18"
    elif failure == "v16-state-version":
        state["_token_codec_state_version"].fill_(16)
        match = "not V18"
    elif failure == "v17-state-version":
        state["_token_codec_state_version"].fill_(17)
        match = "not V18"
    elif failure == "legacy-bias":
        state["token_decoder.bias"] = torch.ones(source.patch_value_dim)
        match = "legacy independent token-codec"
    else:
        state["token_codec_raw_detail_rows"][1].copy_(
            state["token_codec_raw_detail_rows"][0]
        )
        match = "minimum singular value"

    destination = _tiny_dit()
    with pytest.raises(RuntimeError, match=match):
        destination.load_state_dict(state, strict=False)
