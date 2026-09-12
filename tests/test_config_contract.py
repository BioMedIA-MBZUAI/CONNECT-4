from copy import deepcopy
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from utils.config import (
    A4_RECOVERY_PROTOCOL_PROFILE,
    PAPER_PROTOCOL_PROFILE,
    RECOVERY_PREPROCESSING_EVIDENCE_STATUS,
    validate_figure1_recovery_gate_config,
    validate_paper_config,
)
from utils.immutable_yaml import read_immutable_yaml_snapshot


def _raw_repository_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "connect4.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repository_config():
    config = _raw_repository_config()
    # The committed production template must stay blocked until deployment
    # supplies its independently authenticated revision.  Unit fixtures use a
    # syntactically pinned stand-in and never claim that it is an authority.
    config["models"]["modernbert_revision"] = "a" * 40
    return config


def _recovery_config():
    config = _repository_config()
    config["data"].update(
        {
            "protocol_profile": A4_RECOVERY_PROTOCOL_PROFILE,
            "preprocessing_evidence_status": (
                RECOVERY_PREPROCESSING_EVIDENCE_STATUS
            ),
            "require_paper_preprocessing": False,
            "expected_cohort_scan_counts": {"A4": 2155},
            "architecture_shape": [64, 80, 64],
            "common_grid_contract_path": None,
            "common_grid_contract_sha256": None,
            "common_grid_contract_sha256_env": "",
            "native_alignment_authority_sha256": "a" * 64,
            "native_alignment_authority_sha256_env": "",
            "native_alignment_authority_path": "/recovery/structural_batch.json",
            "structural_stage_root": "/recovery/structural",
            "fmri_dir": "/recovery/targets",
            "native_selection_manifest_path": "/recovery/selection.json",
            "native_selection_manifest_sha256": "b" * 64,
            "native_selection_root_review_path": "/recovery/root_review.json",
            "native_selection_root_review_sha256": "c" * 64,
            "native_completed_set_path": "/recovery/completed_set.json",
            "native_completed_set_sha256": "d" * 64,
            "native_completed_set_commit_marker_path": (
                "/recovery/completed_set.commit.json"
            ),
            "native_completed_set_commit_marker_sha256": "e" * 64,
            "native_reviewed_source_path": "/recovery/native_preprocessing.py",
            "native_reviewed_source_sha256": "f" * 64,
            "native_runtime_attester_sha256": "1" * 64,
            "native_verifier_sha256": "2" * 64,
        }
    )
    config["models"]["dit"]["input_size"] = [64, 80, 64]
    config["models"]["dit"]["patch_size"] = [16, 16, 16]
    config["models"]["unet"].update(
        {
            "depth_slab_size": 4,
            "depth_slab_sweep_summary_path": "/recovery/figure1_v10_sweep.json",
            "depth_slab_sweep_summary_sha256": "3" * 64,
            "depth_slab_sweep_summary_sha256_env": "",
            "depth_slab_ddp_smoke_path": "/recovery/figure1_v10_ddp.json",
            "depth_slab_ddp_smoke_sha256": "4" * 64,
            "depth_slab_ddp_smoke_sha256_env": "",
        }
    )
    return config


def _recovery_gate_config():
    config = _recovery_config()
    config["models"].update(
        {
            "brainiac_path": "/recovery/models/BrainIAC.ckpt",
            "brainiac_checkpoint_sha256": "5" * 64,
            "brainiac_checkpoint_sha256_env": "",
            "brainiac_source_sha256": "6" * 64,
            "brainiac_source_sha256_env": "",
        }
    )
    config["training"].update(
        {
            "split_manifest": "/recovery/splits/patient_split.json",
            "split_manifest_sha256": "7" * 64,
            "split_manifest_sha256_env": "",
        }
    )
    perceptual = config["training"]["loss"]["perceptual_extractor"]
    perceptual.update(
        {
            "source_root": "/recovery/brainlm/source",
            "source_root_env": "",
            "source_publication_marker": "/recovery/brainlm/source.commit.json",
            "source_publication_marker_env": "",
            "authority": "/recovery/brainlm/authority.json",
            "authority_env": "",
            "authority_sha256": "8" * 64,
            "authority_sha256_env": "",
        }
    )
    config["models"]["unet"].update(
        {
            "depth_slab_size": None,
            "depth_slab_sweep_summary_path": None,
            "depth_slab_sweep_summary_sha256": None,
            "depth_slab_sweep_summary_sha256_env": "",
            "depth_slab_ddp_smoke_path": None,
            "depth_slab_ddp_smoke_sha256": None,
            "depth_slab_ddp_smoke_sha256_env": "",
        }
    )
    return config


def test_synthetically_pinned_repository_fixture_satisfies_shared_contract():
    config = _repository_config()
    assert config["data"]["architecture_shape"] == [128, 128, 128]
    validate_paper_config(config)


def test_repository_config_rejects_unresolved_modernbert_revision_placeholder():
    config = _raw_repository_config()
    assert config["models"]["modernbert_revision"] == (
        "<full immutable Hugging Face commit SHA>"
    )
    with pytest.raises(ValueError, match="modernbert_revision.*unresolved placeholder"):
        validate_paper_config(config)


@pytest.mark.parametrize("factory", [_repository_config, _recovery_config])
@pytest.mark.parametrize("hidden_size", [256, 4096])
def test_v14_production_profiles_require_exact_512_width_and_8x_codec(
    factory,
    hidden_size,
):
    config = factory()
    config["models"]["dit"]["hidden_size"] = hidden_size
    with pytest.raises(ValueError, match="hidden_size must be exactly 512"):
        validate_paper_config(config)


def test_runtime_split_authority_path_and_digest_pin_are_mandatory():
    missing_path = _repository_config()
    missing_path["training"]["split_manifest"] = ""
    with pytest.raises(ValueError, match="split_manifest must name"):
        validate_paper_config(missing_path)

    missing_pin = _repository_config()
    missing_pin["training"]["split_manifest_sha256"] = None
    missing_pin["training"]["split_manifest_sha256_env"] = ""
    with pytest.raises(ValueError, match="requires split_manifest_sha256"):
        validate_paper_config(missing_pin)

    malformed_pin = _repository_config()
    malformed_pin["training"]["split_manifest_sha256"] = "not-a-digest"
    malformed_pin["training"]["split_manifest_sha256_env"] = ""
    with pytest.raises(ValueError, match="must be lowercase SHA-256"):
        validate_paper_config(malformed_pin)


def test_recovery_dit_is_full_grid_and_exact_depth_slab_bounds_unet_stem():
    config = _recovery_config()
    target = tuple(config["data"]["architecture_shape"])
    dit_grid = tuple(config["models"]["dit"]["input_size"])
    patch = tuple(config["models"]["dit"]["patch_size"])
    temporal_patch = config["models"]["dit"]["temporal_patch_size"]
    frames = config["data"]["num_frames"]

    assert target == (64, 80, 64)
    assert dit_grid == target
    spatial_tokens = 1
    for size, width in zip(dit_grid, patch):
        spatial_tokens *= size // width
    assert spatial_tokens == 80
    assert temporal_patch == 1
    assert (frames // temporal_patch) * spatial_tokens == 10_240

    # Static lower-bound assessment for the largest interior bf16 decoder-stem
    # slab (the fixture's four core slices plus the exact 14-voxel halo on both
    # sides). Production accepts this value only when the digest-bound sweep
    # independently selects it.
    slab_depth = config["models"]["unet"]["depth_slab_size"] + 2 * 14
    decoder_stem_bytes = (
        frames
        * slab_depth
        * target[1]
        * target[2]
        * config["models"]["unet"]["base_channels"]
        * 2
    )
    assert decoder_stem_bytes == 2_684_354_560


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("data", "num_frames", 127, "data.num_frames"),
        ("data", "voxel_size_mm", 2.0, "data.voxel_size_mm"),
        ("data", "tr_seconds", 2.0, "data.tr_seconds"),
        ("data", "out_channels", 2, "data.out_channels"),
    ],
)
def test_fixed_data_invariants_cannot_be_overridden(section, key, value, message):
    config = _repository_config()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_paper_config(config)


def test_temporal_attention_and_published_loss_are_mandatory():
    config = _repository_config()
    config["models"]["unet"]["use_temporal_attn"] = False
    with pytest.raises(ValueError, match="use_temporal_attn"):
        validate_paper_config(config)

    config = _repository_config()
    config["training"]["loss"]["fc"] = 0.0
    with pytest.raises(ValueError, match=r"training\.loss\.fc"):
        validate_paper_config(config)


def test_t1_high_pass_detail_recovery_is_disabled_in_repository_config():
    config = _repository_config()
    assert config["models"]["unet"]["detail_path_enabled"] is False
    validate_paper_config(config)

    config["models"]["unet"]["detail_path_enabled"] = "false"
    with pytest.raises(ValueError, match="detail_path_enabled must be boolean"):
        validate_paper_config(config)

    config = _recovery_config()
    config["models"]["unet"]["detail_path_enabled"] = True
    with pytest.raises(ValueError, match="forbids copying T1 texture"):
        validate_paper_config(config)


def test_recovery_profile_rejects_old_low_resolution_or_axis_permuted_dit_grid():
    config = _recovery_config()
    config["models"]["dit"]["input_size"] = [8, 10, 8]
    config["models"]["dit"]["patch_size"] = [2, 2, 2]
    with pytest.raises(ValueError, match="complete data.architecture_shape"):
        validate_paper_config(config)


def test_v9_rejects_an_unversioned_attention_window_change():
    config = _repository_config()
    config["models"]["dit"]["spatial_window_size"] = [9, 9, 9]
    with pytest.raises(ValueError, match=r"exactly \[8,8,8\]"):
        validate_paper_config(config)

    config = _recovery_config()
    config["models"]["dit"]["input_size"] = [80, 64, 64]
    with pytest.raises(ValueError, match="complete data.architecture_shape"):
        validate_paper_config(config)


def test_cohort_counts_and_unknown_keys_cannot_be_disabled():
    config = _repository_config()
    config["data"]["expected_cohort_scan_counts"] = {"A4": 1, "ADNI": 1}
    with pytest.raises(ValueError, match="published"):
        validate_paper_config(config)

    config = deepcopy(_repository_config())
    config["training"]["silent_override"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        validate_paper_config(config)


def test_authenticated_a4_recovery_profile_is_truthful_and_count_exact():
    config = _recovery_config()
    validate_paper_config(config)

    masquerading = deepcopy(config)
    masquerading["data"]["protocol_profile"] = PAPER_PROTOCOL_PROFILE
    with pytest.raises(ValueError, match=r"paper A4\+ADNI profile"):
        validate_paper_config(masquerading)

    masquerading = deepcopy(config)
    masquerading["data"]["preprocessing_evidence_status"] = (
        "PAPER_CERTIFIED_FMRIPREP"
    )
    with pytest.raises(ValueError, match="NON_CERTIFIED_RECOVERY_PREPROCESSING"):
        validate_paper_config(masquerading)

    masquerading = deepcopy(config)
    masquerading["data"]["expected_cohort_scan_counts"] = {
        "A4": 2155,
        "ADNI": 1,
    }
    with pytest.raises(ValueError, match="exactly one configured A4 count"):
        validate_paper_config(masquerading)


def test_recovery_profile_requires_a_digest_bound_gpu_sweep_selection():
    config = _recovery_config()
    config["models"]["unet"]["depth_slab_size"] = None
    config["models"]["unet"]["depth_slab_sweep_summary_path"] = None
    config["models"]["unet"]["depth_slab_sweep_summary_sha256"] = None
    with pytest.raises(ValueError, match="selected by an authenticated Figure-1"):
        validate_paper_config(config)


def test_recovery_g0_gate_is_non_circular_but_not_production_admissible():
    config = _recovery_gate_config()
    validated = validate_figure1_recovery_gate_config(config)
    assert validated["models"]["unet"]["depth_slab_size"] is None
    assert config["models"]["unet"]["depth_slab_size"] is None
    with pytest.raises(ValueError, match="selected by an authenticated Figure-1"):
        validate_paper_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depth_slab_size", 1),
        ("depth_slab_sweep_summary_path", "/recovery/sweep.json"),
        ("depth_slab_sweep_summary_sha256", "9" * 64),
        ("depth_slab_sweep_summary_sha256_env", "SWEEP_SHA"),
        ("depth_slab_ddp_smoke_path", "/recovery/ddp.json"),
        ("depth_slab_ddp_smoke_sha256", "a" * 64),
        ("depth_slab_ddp_smoke_sha256_env", "DDP_SHA"),
    ],
)
def test_recovery_g0_gate_rejects_any_premature_slab_binding(key, value):
    config = _recovery_gate_config()
    config["models"]["unet"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_figure1_recovery_gate_config(config)


@pytest.mark.parametrize(
    ("path", "direct_key", "environment_key"),
    [
        (("models",), "brainiac_checkpoint_sha256", "brainiac_checkpoint_sha256_env"),
        (("models",), "brainiac_source_sha256", "brainiac_source_sha256_env"),
        (("training",), "split_manifest_sha256", "split_manifest_sha256_env"),
        (
            ("training", "loss", "perceptual_extractor"),
            "authority_sha256",
            "authority_sha256_env",
        ),
    ],
)
def test_recovery_g0_gate_rejects_environment_only_authorities(
    path, direct_key, environment_key
):
    config = _recovery_gate_config()
    target = config
    for key in path:
        target = target[key]
    target[direct_key] = None
    target[environment_key] = "UNTRUSTED_DYNAMIC_VALUE"
    with pytest.raises(ValueError, match=environment_key):
        validate_figure1_recovery_gate_config(config)


@pytest.mark.parametrize(
    "environment_key",
    [
        "checkpoint_env",
        "checkpoint_sha256_env",
        "source_revision_env",
        "source_root_env",
        "source_publication_marker_env",
        "config_env",
        "config_sha256_env",
        "atlas_env",
        "coordinates_env",
        "authority_env",
        "authority_sha256_env",
    ],
)
def test_recovery_g0_gate_rejects_every_brainlm_environment_fallback(
    environment_key,
):
    config = _recovery_gate_config()
    config["training"]["loss"]["perceptual_extractor"][environment_key] = (
        "HIDDEN_DYNAMIC_VALUE"
    )
    with pytest.raises(ValueError, match=environment_key):
        validate_figure1_recovery_gate_config(config)


def test_recovery_g0_gate_rejects_relative_paths_and_count_drift():
    relative = _recovery_gate_config()
    relative["training"]["split_manifest"] = "splits/patient_split.json"
    with pytest.raises(ValueError, match="direct absolute path"):
        validate_figure1_recovery_gate_config(relative)

    subset = _recovery_gate_config()
    subset["data"]["expected_cohort_scan_counts"] = {"A4": 2121}
    with pytest.raises(ValueError, match="2,155"):
        validate_figure1_recovery_gate_config(subset)

    paper = _repository_config()
    with pytest.raises(ValueError, match="A4 native recovery|selected"):
        validate_figure1_recovery_gate_config(paper)

    config = _recovery_config()
    config["models"]["unet"]["depth_slab_sweep_summary_sha256"] = "unsigned"
    with pytest.raises(ValueError, match="must be lowercase SHA-256"):
        validate_paper_config(config)


@pytest.mark.parametrize(
    "path",
    [
        ("data", "root_dir"),
        ("models", "brainiac_path"),
        ("training", "split_manifest"),
        ("training", "loss", "perceptual_extractor", "checkpoint"),
        ("training", "loss", "perceptual_extractor", "config"),
        ("training", "loss", "perceptual_extractor", "atlas"),
        ("training", "loss", "perceptual_extractor", "coordinates"),
    ],
)
def test_recovery_g0_gate_rejects_tilde_paths_without_home_expansion(path):
    config = _recovery_gate_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "~/must-not-resolve"
    with pytest.raises(ValueError, match="direct absolute path"):
        validate_figure1_recovery_gate_config(config)


def test_immutable_yaml_snapshot_reads_once_and_hashes_consumed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "gate.yaml"
    payload = b"answer: 42\n"
    path.write_bytes(payload)
    calls = []
    original_read = os.read

    def counted_read(descriptor: int, count: int) -> bytes:
        calls.append((descriptor, count))
        return original_read(descriptor, count)

    monkeypatch.setattr(os, "read", counted_read)
    snapshot = read_immutable_yaml_snapshot(path, "test config")
    assert calls and len(calls) == 1
    assert snapshot.raw_bytes == payload
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.document == {"answer": 42}


def test_immutable_yaml_snapshot_rejects_a_b_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "gate.yaml"
    replacement = tmp_path / "replacement.yaml"
    path.write_text("identity: A\n", encoding="utf-8")
    replacement.write_text("identity: B\n", encoding="utf-8")
    original_read = os.read

    def replace_after_read(descriptor: int, count: int) -> bytes:
        payload = original_read(descriptor, count)
        os.replace(replacement, path)
        return payload

    monkeypatch.setattr(os, "read", replace_after_read)
    with pytest.raises(RuntimeError, match="mutated while it was read|pathname changed"):
        read_immutable_yaml_snapshot(path, "test config")


@pytest.mark.parametrize(
    "artifact_key",
    [
        "checkpoint",
        "config",
        "atlas",
        "coordinates",
        "authority",
    ],
)
def test_recovery_g0_gate_rejects_relative_brainlm_artifact_paths(
    artifact_key,
):
    config = _recovery_gate_config()
    config["training"]["loss"]["perceptual_extractor"][artifact_key] = (
        f"relative/{artifact_key}"
    )
    with pytest.raises(ValueError, match="direct absolute path"):
        validate_figure1_recovery_gate_config(config)


def test_recovery_profile_requires_native_authority_and_forbids_common_grid():
    config = _recovery_config()
    config["data"]["native_alignment_authority_sha256"] = None
    with pytest.raises(ValueError, match="native_alignment_authority_sha256"):
        validate_paper_config(config)

    config = _recovery_config()
    config["data"]["common_grid_contract_path"] = "/tmp/forbidden.json"
    with pytest.raises(ValueError, match="not one cohort-common reference affine"):
        validate_paper_config(config)

@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("models", "graphs", "k_neighbors"), 0, "k_neighbors"),
        (("models", "dit", "depth"), 0, "depth"),
        (("models", "dit", "num_heads"), 0, "num_heads"),
        (("models", "fusion", "hidden_dim"), 0, "hidden_dim"),
        (("models", "unet", "num_levels"), 0, "num_levels"),
        (("training", "learning_rate"), 0.0, "learning_rate"),
    ],
)
def test_structural_ranges_cannot_collapse_paper_components(path, value, message):
    config = _repository_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_paper_config(config)
