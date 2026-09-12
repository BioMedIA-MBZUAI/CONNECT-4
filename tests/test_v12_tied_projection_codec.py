import copy
import hashlib
import json

import pytest
import torch
import torch.nn as nn

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
    TARGET_VALIDITY_MASK_CONTRACT,
)
from data.protocol import RUN_ARTIFACT_IDENTITY_SCHEMA
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


def _tiny_dit(*, hidden_size=4):
    return DiT4DTemporal(
        input_size=(2, 2, 2),
        in_channels=1,
        patch_size=(2, 2, 2),
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


def test_v18_architecture_checkpoint_launch_and_artifact_identity_are_closed():
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
    assert contract["v18_inference_launch_status"].startswith("blocked-")
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

    for old_version in (10, 11, 12, 13, 14, 15, 16, 17):
        legacy = copy.deepcopy(contract)
        legacy["schema"] = f"connect4-synthesis-architecture-contract-v{old_version}"
        legacy["checkpoint_format"] = f"connect4_iteration_exact_v{old_version}"
        with pytest.raises(RuntimeError, match="pre-v18"):
            require_current_synthesis_architecture(
                legacy, _canonical_sha256(legacy)
            )
        with pytest.raises(RuntimeError, match="pre-V18"):
            require_current_synthesis_launch_schema(
                f"connect4-synthesis-v{old_version}-launch-admission-v1"
            )


def test_v18_contract_states_honest_projection_codec_semantics():
    contract = synthesis_architecture_contract()

    assert TOKEN_CODEC_STATE_VERSION == 18
    assert contract["token_codec_state_version"] == 18
    assert contract["token_latent_normalization_contract"] == (
        TOKEN_LATENT_NORMALIZATION_CONTRACT
    )
    assert contract["token_codec_dc_mean_preservation"] == (
        "exact-in-real-arithmetic"
    )
    assert contract["token_codec_detail_preservation"] == (
        "exact-only-within-learned-orthonormal-detail-subspace"
    )
    assert not contract["token_codec_full_patch_invertibility_when_compressed_claimed"]
    assert contract["token_to_patch_decoder"] == (
        "exact-transpose-of-same-effective-encoder-no-parameters-no-bias"
    )
    assert contract["token_to_patch_decoder_operator_norm"] == 1.0
    assert contract["token_codec_encoder_bias"] == "forbidden-no-parameter"
    assert contract["token_codec_decoder_bias"] == "forbidden-no-parameter"
    assert not contract["legacy_independent_token_decoder_parameters_accepted"]
    assert not contract["encoder_decoder_inverse_scale_symmetry_accepted"]
    assert contract["token_codec_raw_detail_minimum_singular_value"] == (
        TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
    )
    assert contract["token_codec_orthonormality_runtime_atol"] == (
        TOKEN_CODEC_ORTHONORMAL_ATOL
    )
    assert contract["token_codec_compute_dtype"] == (
        "fp32-autocast-disabled-for-orthogonalization-encode-and-tied-decode"
    )


@pytest.mark.parametrize("legacy_version", [10, 11, 12, 13, 14, 15, 16, 17])
def test_executable_resume_gate_rejects_pre_v18_checkpoint_before_state_load(
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


@pytest.mark.parametrize("hidden_size", [1, 9])
def test_v18_rejects_codec_width_outside_two_through_patch_dimension(hidden_size):
    with pytest.raises(ValueError, match="2 <= hidden_size <= patch_value_dim"):
        _tiny_dit(hidden_size=hidden_size)


def test_effective_encoder_has_fixed_dc_row_orthonormal_rows_and_tied_decoder():
    torch.manual_seed(1201)
    model = _tiny_dit()
    effective = model.effective_token_codec_encoder()
    expected_dc = torch.full(
        (model.patch_value_dim,),
        1.0 / model.patch_value_dim**0.5,
        dtype=torch.float32,
    )

    assert effective.dtype == torch.float32
    assert tuple(effective.shape) == (model.hidden_size, model.patch_value_dim)
    assert torch.equal(effective[0], expected_dc)
    assert torch.allclose(
        effective @ effective.T,
        torch.eye(model.hidden_size),
        atol=TOKEN_CODEC_ORTHONORMAL_ATOL,
        rtol=0.0,
    )
    assert torch.allclose(
        torch.linalg.svdvals(effective.T),
        torch.ones(model.hidden_size),
        atol=TOKEN_CODEC_ORTHONORMAL_ATOL,
        rtol=0.0,
    )
    assert not hasattr(model, "clean_token_encoder")
    assert not hasattr(model, "token_decoder")
    codec_parameters = [
        name
        for name, _ in model.named_parameters()
        if "token_codec" in name
        or "token_decoder" in name
        or "clean_token_encoder" in name
    ]
    assert codec_parameters == ["token_codec_raw_detail_rows"]
    assert not any(
        "token_decoder" in key or "clean_token_encoder" in key
        for key in model.state_dict()
    )

    report = model.validate_token_codec_invariants()
    assert report["contract"] == TOKEN_LATENT_NORMALIZATION_CONTRACT
    assert report["state_version"] == 18
    assert report["minimum_raw_detail_singular_value"] >= (
        TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
    )
    assert report["effective_encoder_gram_max_abs_error"] <= (
        TOKEN_CODEC_ORTHONORMAL_ATOL
    )
    assert report["effective_encoder_finite_fp32"] is True
    assert report["fixed_dc_row_exact"] is True
    assert report["encoder_bias_parameter_present"] is False
    assert report["decoder_is_exact_transpose"] is True
    assert report["decoder_parameter_count"] == 0
    assert report["decoder_bias_parameter_present"] is False
    assert report["decoder_operator_norm"] == 1.0


def test_projection_preserves_dc_and_only_the_learned_detail_subspace():
    torch.manual_seed(1202)
    compressed = _tiny_dit(hidden_size=4).eval()
    constant = torch.empty(3, 1, 2, 2, 2, 2)
    constant[0].fill_(-0.75)
    constant[1].fill_(0.25)
    constant[2].fill_(2.0)
    constant_latent = compressed.encode_clean_volume(constant)
    constant_reconstruction = compressed.decode_tokens(constant_latent)

    assert torch.allclose(
        constant_latent[..., 0],
        constant[:, :, :, 0, 0, 0].squeeze(1)
        * compressed.patch_value_dim**0.5,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        constant_latent[..., 1:],
        torch.zeros_like(constant_latent[..., 1:]),
        atol=2e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        constant_reconstruction, constant, atol=2e-6, rtol=2e-6
    )

    subspace_coordinates = torch.randn(3, 2, compressed.hidden_size)
    subspace_volume = compressed.decode_tokens(subspace_coordinates)
    assert torch.allclose(
        compressed.encode_clean_volume(subspace_volume),
        subspace_coordinates,
        atol=3e-6,
        rtol=3e-6,
    )

    arbitrary = torch.randn(3, 1, 2, 2, 2, 2)
    projected = compressed.decode_tokens(compressed.encode_clean_volume(arbitrary))
    assert torch.allclose(
        projected.mean(dim=(-3, -2, -1)),
        arbitrary.mean(dim=(-3, -2, -1)),
        atol=3e-6,
        rtol=3e-6,
    )
    assert (projected - arbitrary).square().mean() > 1e-3

    square = _tiny_dit(hidden_size=8).eval()
    full_rank_reconstruction = square.decode_tokens(
        square.encode_clean_volume(arbitrary)
    )
    assert torch.allclose(
        full_rank_reconstruction, arbitrary, atol=4e-6, rtol=4e-6
    )


def test_codec_invariants_and_tied_projection_remain_fp32_inside_bf16_autocast():
    model = _tiny_dit().eval()
    constant = torch.ones(1, 1, 2, 2, 2, 2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        report = model.validate_token_codec_invariants()
        latent = model.encode_clean_volume(constant)
        reconstructed = model.decode_tokens(latent)

    assert report["effective_encoder_gram_max_abs_error"] <= (
        TOKEN_CODEC_ORTHONORMAL_ATOL
    )
    assert latent.dtype == torch.float32
    assert reconstructed.dtype == torch.float32
    assert torch.allclose(reconstructed, constant, atol=2e-6, rtol=2e-6)

    bf16_model = _tiny_dit().to(dtype=torch.bfloat16).eval()
    bf16_constant = constant.to(dtype=torch.bfloat16)
    bf16_report = bf16_model.validate_token_codec_invariants()
    bf16_latent = bf16_model.encode_clean_volume(bf16_constant)
    bf16_reconstruction = bf16_model.decode_tokens(bf16_latent)
    assert bf16_report["effective_encoder_gram_max_abs_error"] <= (
        TOKEN_CODEC_ORTHONORMAL_ATOL
    )
    assert bf16_latent.dtype == torch.float32
    assert bf16_reconstruction.dtype == torch.float32
    assert torch.allclose(
        bf16_reconstruction, constant, atol=2e-5, rtol=2e-5
    )
    bf16_objective = bf16_model.codec_reconstruction_loss(
        torch.randn_like(bf16_constant),
        target_validity_mask=torch.ones(1, 1, 2, 2, 2, dtype=torch.bfloat16),
    )["loss"]
    bf16_objective.backward()
    assert bf16_model.token_codec_raw_detail_rows.grad is not None
    assert torch.isfinite(bf16_model.token_codec_raw_detail_rows.grad).all()
    assert bf16_model.token_codec_raw_detail_rows.grad.abs().sum() > 0


def test_raw_epsilon_scaling_cannot_shrink_latents_or_gain_an_inverse_decoder():
    torch.manual_seed(1203)
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

    assert model.validate_token_codec_invariants()[
        "minimum_raw_detail_singular_value"
    ] >= TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
    assert torch.allclose(scaled_encoder, baseline_encoder, atol=2e-6, rtol=2e-6)
    assert torch.allclose(scaled_latent, baseline_latent, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        scaled_reconstruction,
        baseline_reconstruction,
        atol=3e-6,
        rtol=3e-6,
    )

    epsilon = 1e-3
    attenuated = model.decode_tokens(epsilon * scaled_latent)
    assert torch.allclose(
        attenuated,
        epsilon * scaled_reconstruction,
        atol=1e-7,
        rtol=2e-5,
    )
    assert (attenuated - baseline_reconstruction).square().mean() > 1e-2

    attacked_state = copy.deepcopy(model.state_dict())
    attacked_state["token_decoder.weight"] = (
        model.effective_token_codec_encoder().T / epsilon
    )
    with pytest.raises(RuntimeError, match="legacy independent token-codec"):
        _tiny_dit().load_state_dict(attacked_state, strict=False)


def test_constant_patch_bias_attack_is_impossible_and_detected_at_load_or_runtime():
    model = _tiny_dit().eval()
    clean = torch.full((2, 1, 2, 2, 2, 2), 0.625)
    reconstructed = model.decode_tokens(model.encode_clean_volume(clean))
    assert torch.allclose(reconstructed, clean, atol=2e-6, rtol=2e-6)

    state = copy.deepcopy(model.state_dict())
    state["token_decoder.bias"] = torch.full((model.patch_value_dim,), 0.625)
    with pytest.raises(RuntimeError, match="legacy independent token-codec"):
        _tiny_dit().load_state_dict(state, strict=False)

    runtime_attacked = _tiny_dit()
    runtime_attacked.token_decoder = nn.Linear(
        runtime_attacked.hidden_size,
        runtime_attacked.patch_value_dim,
        bias=True,
    )
    with pytest.raises(RuntimeError, match="forbids independent"):
        runtime_attacked.validate_token_codec_invariants()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "malformation",
    [
        "v10-marker",
        "v11-marker",
        "v12-marker",
        "v13-marker",
        "v14-marker",
        "v15-marker",
        "v16-marker",
        "v17-marker",
        "float-marker",
        "missing-marker",
        "missing-raw-detail",
        "wrong-shape",
        "integer-raw-detail",
        "rank-deficient",
        "non-finite",
    ],
)
def test_malformed_or_legacy_codec_state_fails_closed_even_non_strict(
    strict, malformation
):
    state = copy.deepcopy(_tiny_dit().state_dict())
    if malformation == "v10-marker":
        state["_token_codec_state_version"].fill_(10)
        match = "V10/V11"
    elif malformation == "v11-marker":
        state["_token_codec_state_version"].fill_(11)
        match = "V10/V11/V12"
    elif malformation == "v12-marker":
        state["_token_codec_state_version"].fill_(12)
        match = "V10/V11/V12/V13"
    elif malformation == "v13-marker":
        state["_token_codec_state_version"].fill_(13)
        match = "V10/V11/V12/V13"
    elif malformation == "v14-marker":
        state["_token_codec_state_version"].fill_(14)
        match = "V10/V11/V12/V13/V14"
    elif malformation == "v15-marker":
        state["_token_codec_state_version"].fill_(15)
        match = "V10/V11/V12/V13/V14/V15"
    elif malformation == "v16-marker":
        state["_token_codec_state_version"].fill_(16)
        match = "V10/V11/V12/V13/V14/V15/V16"
    elif malformation == "v17-marker":
        state["_token_codec_state_version"].fill_(17)
        match = "V10/V11/V12/V13/V14/V15/V16/V17"
    elif malformation == "float-marker":
        state["_token_codec_state_version"] = torch.tensor(18.0)
        match = "state marker"
    elif malformation == "missing-marker":
        state.pop("_token_codec_state_version")
        match = "state marker"
    elif malformation == "missing-raw-detail":
        state.pop("token_codec_raw_detail_rows")
        match = "no token-codec raw detail rows"
    elif malformation == "wrong-shape":
        state["token_codec_raw_detail_rows"] = torch.ones(2, 7)
        match = "must have shape"
    elif malformation == "integer-raw-detail":
        state["token_codec_raw_detail_rows"] = torch.eye(3, 7, dtype=torch.int64)
        match = "real floating-point dtype"
    elif malformation == "rank-deficient":
        state["token_codec_raw_detail_rows"][1].copy_(
            state["token_codec_raw_detail_rows"][0]
        )
        match = "minimum singular value"
    else:
        state["token_codec_raw_detail_rows"][0, 0] = float("nan")
        match = "non-finite"

    with pytest.raises(RuntimeError, match=match):
        _tiny_dit().load_state_dict(state, strict=strict)


def test_v18_checkpoint_roundtrip_and_post_optimizer_invariants():
    torch.manual_seed(1204)
    source = _tiny_dit().train()
    clean = torch.randn(2, 1, 2, 2, 2, 2)
    validity = torch.ones(2, 1, 2, 2, 2)
    expected_encoder = source.effective_token_codec_encoder().detach().clone()
    expected_latent = source.encode_clean_volume(clean).detach().clone()

    destination = _tiny_dit().train()
    incompatible = destination.load_state_dict(
        copy.deepcopy(source.state_dict()), strict=True
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(
        destination._token_codec_state_version,
        torch.tensor(TOKEN_CODEC_STATE_VERSION, dtype=torch.int64),
    )
    assert torch.equal(destination.effective_token_codec_encoder(), expected_encoder)
    assert torch.equal(destination.encode_clean_volume(clean), expected_latent)

    optimizer = torch.optim.SGD(
        [destination.token_codec_raw_detail_rows], lr=1e-4
    )
    objective = destination.codec_reconstruction_loss(
        clean,
        target_validity_mask=validity,
    )["loss"]
    objective.backward()
    gradient = destination.token_codec_raw_detail_rows.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    optimizer.step()
    report = destination.validate_token_codec_invariants()
    assert report["state_version"] == 18
    assert report["minimum_raw_detail_singular_value"] >= (
        TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE
    )


def test_diffusion_loss_reuses_one_effective_encoder_for_codec_and_x0_decode(
    monkeypatch,
):
    torch.manual_seed(1205)
    model = _tiny_dit().train()
    clean = torch.randn(2, 1, 2, 2, 2, 2)
    graph = torch.randn(2, 1, model.hidden_size)
    validity = torch.ones(2, 1, 2, 2, 2)
    calls = 0
    original = model.effective_token_codec_encoder

    def counted_effective_encoder():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(
        model,
        "effective_token_codec_encoder",
        counted_effective_encoder,
    )
    result = model.diffusion_loss(
        clean,
        graph,
        timesteps=torch.tensor([1, 2]),
        noise=torch.randn(2, 2, model.hidden_size),
        target_validity_mask=validity,
    )

    assert calls == 1
    assert result["pred_original_volume"].shape == clean.shape
    assert torch.isfinite(result["pred_original_volume"]).all()


def test_post_optimizer_gate_rejects_raw_detail_singular_value_below_floor():
    model = _tiny_dit()
    with torch.no_grad():
        model.token_codec_raw_detail_rows.mul_(
            TOKEN_CODEC_RAW_DETAIL_MINIMUM_SINGULAR_VALUE / 2.0
        )
    with pytest.raises(RuntimeError, match="minimum singular value"):
        model.validate_token_codec_invariants()


def test_exact_roi_profile_target_validity_and_target_blind_sampler_positives_hold():
    assert len(CANONICAL_ROI_SPECS) == 32
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
    assert _require_canonical_roi_mapping(
        dict(CANONICAL_ROI_LABEL_TO_CHANNEL)
    ) == dict(CANONICAL_ROI_LABEL_TO_CHANNEL)

    expected_contracts = {
        "paper-a4-adni-fmriprep-v1": TARGET_VALIDITY_MASK_CONTRACT,
        "a4-native-recovery-v1": RECOVERY_TARGET_VALIDITY_MASK_CONTRACT,
    }
    assert dict(TARGET_VALIDITY_MASK_CONTRACT_BY_PROTOCOL_PROFILE) == (
        expected_contracts
    )
    for profile, contract in expected_contracts.items():
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

    assert "fmri" not in _TARGET_BLIND_SAMPLER_KEYS
    assert "target_validity_mask" not in _TARGET_BLIND_SAMPLER_KEYS
    assert "target_validity_mask_contract" not in _TARGET_BLIND_SAMPLER_KEYS
    assert "brain_mask" in _TARGET_BLIND_SAMPLER_KEYS
