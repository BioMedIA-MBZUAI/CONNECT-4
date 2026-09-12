import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import data.cohort_admission as admission_module
from data.cohort_admission import (
    MAX_COHORT_ADMISSION_BYTES,
    broadcast_rank_zero_cohort_admission,
    build_cohort_admission,
    canonical_json_bytes,
    cohort_admission_identity,
    seal_cohort_admission,
    validate_cohort_admission,
    validate_cohort_admission_identity,
)
from data.dataset_precomputed import Connect4PrecomputedDataset
from data.provenance import canonical_sha256, sha256_file


def _runtime_identity(world_size=2):
    identity = {
        "schema": "test-four-rank-runtime",
        "world_size": world_size,
    }
    identity["canonical_sha256"] = canonical_sha256(identity)
    return identity


def _authority(world_size=2):
    scans = ["S1", "S2", "S3"]
    runtime = _runtime_identity(world_size)
    split = {
        "format": "connect4_split_identity_v2",
        "sha256": "1" * 64,
        "partitions": {
            "train": [{"scan_id": "S1", "patient_id": "P1", "cohort": "A4"}],
            "validation": [
                {"scan_id": "S2", "patient_id": "P2", "cohort": "A4"}
            ],
            "test": [{"scan_id": "S3", "patient_id": "P3", "cohort": "A4"}],
        },
    }
    role_items = [
        {"scan_id": "S1", "role": "train"},
        {"scan_id": "S2", "role": "development-validation"},
    ]
    brainlm_scope = {
        "schema": "connect4-brainlm-authority-split-scope-v1",
        "split_assignment_sha256": split["sha256"],
        "split_identity_sha256": canonical_sha256(split),
        "target_scan_roles_sha256": canonical_sha256(role_items),
        "authorized_scan_count": 2,
        "sealed_scan_count": 1,
        "sealed_scan_ids_sha256": canonical_sha256(["S3"]),
        "sealed_scans_authorized": False,
        "functional_target_bytes_opened": False,
    }
    brainlm_scope["record_sha256"] = canonical_sha256(brainlm_scope)
    artifact = {
        "format": "connect4_run_artifacts_v6",
        "structural_artifact_identities_sha256": "a" * 64,
        "brainlm_authority_scope": brainlm_scope,
    }
    state = {
        "scan_ids": scans,
        "scan_roles": {
            "S1": "train",
            "S2": "development-validation",
            "S3": "sealed-test",
        },
        "target_scan_ids": ["S1", "S2"],
        "target_scan_roles": {
            "S1": "train",
            "S2": "development-validation",
        },
    }
    return build_cohort_admission(
        protocol_profile="a4-native-recovery-v1",
        world_size=world_size,
        config_sha256="b" * 64,
        runtime_identity=runtime,
        ordered_scan_ids=scans,
        train_indices=[0],
        development_indices=[1],
        sealed_indices=[2],
        split_identity=split,
        artifact_identity=artifact,
        dataset_state=state,
    )


def test_closed_authority_and_checkpoint_identity_bind_every_root():
    authority = _authority()
    validated = validate_cohort_admission(
        authority,
        expected_config_sha256="b" * 64,
        expected_world_size=2,
        expected_runtime_identity_sha256=authority["runtime_identity_sha256"],
    )
    assert validated == authority
    assert len(canonical_json_bytes(authority)) < MAX_COHORT_ADMISSION_BYTES
    identity = cohort_admission_identity(authority)
    assert validate_cohort_admission_identity(identity) == identity
    assert identity["structural_artifact_identities_sha256"] == "a" * 64
    assert identity["dataset_state_sha256"] == canonical_sha256(
        authority["dataset_state"]
    )


def test_resealed_role_crossing_and_structural_tamper_are_rejected():
    authority = _authority()
    crossed = copy.deepcopy(authority)
    crossed["dataset_state"]["target_scan_roles"]["S2"] = "train"
    crossed["dataset_state_sha256"] = canonical_sha256(crossed["dataset_state"])
    crossed = seal_cohort_admission(crossed)
    with pytest.raises(RuntimeError, match="all-scan roles differ"):
        validate_cohort_admission(
            crossed,
            expected_config_sha256="b" * 64,
            expected_world_size=2,
            expected_runtime_identity_sha256=authority[
                "runtime_identity_sha256"
            ],
        )

    scope_drift = copy.deepcopy(authority)
    scope_drift["artifact_identity"]["brainlm_authority_scope"][
        "split_identity_sha256"
    ] = "9" * 64
    scope_drift["artifact_identity_sha256"] = canonical_sha256(
        scope_drift["artifact_identity"]
    )
    scope_drift = seal_cohort_admission(scope_drift)
    with pytest.raises(RuntimeError, match="BrainLM split scope differs"):
        validate_cohort_admission(
            scope_drift,
            expected_config_sha256="b" * 64,
            expected_world_size=2,
            expected_runtime_identity_sha256=authority[
                "runtime_identity_sha256"
            ],
        )

    sealed_crossed = copy.deepcopy(authority)
    sealed_crossed["dataset_state"]["scan_roles"]["S3"] = "train"
    sealed_crossed["dataset_state_sha256"] = canonical_sha256(
        sealed_crossed["dataset_state"]
    )
    sealed_crossed = seal_cohort_admission(sealed_crossed)
    with pytest.raises(RuntimeError, match="all-scan roles differ"):
        validate_cohort_admission(
            sealed_crossed,
            expected_config_sha256="b" * 64,
            expected_world_size=2,
            expected_runtime_identity_sha256=authority[
                "runtime_identity_sha256"
            ],
        )

    tampered = copy.deepcopy(authority)
    tampered["structural_artifact_identities_sha256"] = "c" * 64
    tampered = seal_cohort_admission(tampered)
    with pytest.raises(RuntimeError, match="structural root differs"):
        validate_cohort_admission(
            tampered,
            expected_config_sha256="b" * 64,
            expected_world_size=2,
            expected_runtime_identity_sha256=authority[
                "runtime_identity_sha256"
            ],
        )


class _TensorBroadcastBus:
    def __init__(self):
        self.rank = 0
        self.messages = []
        self.read_index = 0

    def set_rank(self, rank):
        self.rank = rank
        self.read_index = 0

    def broadcast(self, tensor, src):
        assert src == 0
        assert torch.is_tensor(tensor)
        if self.rank == 0:
            self.messages.append(tensor.detach().cpu().clone())
        else:
            tensor.copy_(self.messages[self.read_index].to(tensor.device))
            self.read_index += 1


def _install_fake_distributed(monkeypatch, bus):
    monkeypatch.setattr(admission_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(admission_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(admission_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(admission_module.dist, "get_rank", lambda: bus.rank)
    monkeypatch.setattr(admission_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(admission_module.dist, "broadcast", bus.broadcast)


def test_nonzero_rank_receives_only_bounded_canonical_tensors(monkeypatch):
    bus = _TensorBroadcastBus()
    _install_fake_distributed(monkeypatch, bus)
    authority = _authority()
    calls = []

    bus.set_rank(0)
    rank_zero = broadcast_rank_zero_cohort_admission(
        lambda: calls.append(0) or authority,
        rank=0,
        world_size=2,
        expected_config_sha256="b" * 64,
        expected_runtime_identity_sha256=authority["runtime_identity_sha256"],
    )
    assert calls == [0]
    assert len(bus.messages) == 3
    assert all(torch.is_tensor(message) for message in bus.messages)

    bus.set_rank(1)
    rank_one = broadcast_rank_zero_cohort_admission(
        lambda: pytest.fail("rank one constructed the cohort"),
        rank=1,
        world_size=2,
        expected_config_sha256="b" * 64,
        expected_runtime_identity_sha256=authority["runtime_identity_sha256"],
    )
    assert rank_one == rank_zero
    assert canonical_json_bytes(rank_one) == canonical_json_bytes(authority)


def test_rank_zero_failure_is_bounded_and_broadcast_without_worker_build(monkeypatch):
    bus = _TensorBroadcastBus()
    _install_fake_distributed(monkeypatch, bus)

    bus.set_rank(0)
    with pytest.raises(RuntimeError, match="rank-zero cohort admission failed"):
        broadcast_rank_zero_cohort_admission(
            lambda: (_ for _ in ()).throw(ValueError("closed failure")),
            rank=0,
            world_size=2,
            expected_config_sha256="b" * 64,
            expected_runtime_identity_sha256=_runtime_identity()[
                "canonical_sha256"
            ],
        )
    assert len(bus.messages) == 3
    assert int(bus.messages[0][1]) <= MAX_COHORT_ADMISSION_BYTES

    bus.set_rank(1)
    with pytest.raises(RuntimeError, match="ValueError: closed failure"):
        broadcast_rank_zero_cohort_admission(
            lambda: pytest.fail("worker rank executed the failing builder"),
            rank=1,
            world_size=2,
            expected_config_sha256="b" * 64,
            expected_runtime_identity_sha256=_runtime_identity()[
                "canonical_sha256"
            ],
        )


def test_oversize_authority_fails_closed_before_payload_broadcast(monkeypatch):
    bus = _TensorBroadcastBus()
    _install_fake_distributed(monkeypatch, bus)
    monkeypatch.setattr(admission_module, "MAX_COHORT_ADMISSION_BYTES", 512)
    bus.set_rank(0)
    with pytest.raises(RuntimeError, match="exceeds the byte limit"):
        broadcast_rank_zero_cohort_admission(
            _authority,
            rank=0,
            world_size=2,
            expected_config_sha256="b" * 64,
            expected_runtime_identity_sha256=_runtime_identity()[
                "canonical_sha256"
            ],
        )
    assert len(bus.messages) == 3
    assert int(bus.messages[0][1]) <= 512


def _dataset_admission_state(tmp_path: Path):
    root = tmp_path / "root"
    precomputed = tmp_path / "precomputed"
    fmri = tmp_path / "fmri"
    for path in (root, precomputed, fmri):
        path.mkdir()
    structural_stage_root = tmp_path / "structural-stage"
    structural_stage_root.mkdir()
    native_authority = tmp_path / "native-authority.json"
    native_authority.write_text("{}\n", encoding="utf-8")
    labels = {
        "t1w",
        "segmentation",
        "image_nodes",
        "mask_nodes",
        "roi_nodes",
        "metadata",
        "patch_distributions",
    }
    descriptors = {}
    for index, label in enumerate(sorted(labels)):
        path = tmp_path / f"{label}.bin"
        path.write_bytes(f"artifact-{index}".encode())
        descriptors[label] = Connect4PrecomputedDataset._admitted_artifact_descriptor(
            path,
            expected_sha256=sha256_file(path),
        )
    state = {
        "schema": Connect4PrecomputedDataset._DISTRIBUTED_STATE_SCHEMA,
        "root": str(root),
        "precomputed_dir": str(precomputed),
        "fmri_dir": str(fmri),
        "require_paper_preprocessing": False,
        "recovery_profile": True,
        "structural_stage_root": str(structural_stage_root),
        "native_alignment_authority_path": str(native_authority),
        "native_alignment_authority_sha256": sha256_file(native_authority),
        "common_grid_contract": None,
        "target_shape": [64, 80, 64],
        "patch_size": [16, 16, 16],
        "num_frames": 128,
        "expected_node_dims": {"image": 768, "mask": 768, "ROI": 619},
        "normalize_intensity": True,
        "scaler_dir": None,
        "scan_ids": ["S1"],
        "scan_roles": {"S1": "sealed-test"},
        "target_scan_ids": [],
        "target_scan_roles": {},
        "target_artifact_identities": {},
        "sample_artifacts": {"S1": descriptors},
        "brainlm_context_records": {},
        "dwi_matrix": np.zeros(
            (
                Connect4PrecomputedDataset.NUM_ROIS,
                Connect4PrecomputedDataset.NUM_ROIS,
            ),
            dtype=np.float32,
        ).tolist(),
    }
    state["record_sha256"] = canonical_sha256(state)
    return state


def test_worker_hydration_never_constructs_source_dataset_and_rechecks_inode(tmp_path):
    state = _dataset_admission_state(tmp_path)
    dataset = Connect4PrecomputedDataset.from_rank_zero_admission_state(state)
    assert dataset.source_dataset is None
    assert set(dataset._admitted_sample_artifacts["S1"]) == {
        "t1w",
        "segmentation",
        "image_nodes",
        "mask_nodes",
        "roi_nodes",
        "metadata",
        "patch_distributions",
    }
    descriptor = dataset._admitted_sample_artifacts["S1"]["image_nodes"]
    path = Path(descriptor["path"])
    original_size = path.stat().st_size
    path.write_bytes(b"x" * original_size)
    with pytest.raises(RuntimeError, match="inode/stat snapshot changed"):
        dataset._admitted_payload("S1", "image_nodes", path)

    # Even if an attacker could rewrite the in-memory stat snapshot, the
    # independently bound SHA-256 still prevents changed bytes from loading.
    metadata = path.lstat()
    for field, value in (
        ("device", metadata.st_dev),
        ("inode", metadata.st_ino),
        ("mode", metadata.st_mode),
        ("nlink", metadata.st_nlink),
        ("size_bytes", metadata.st_size),
        ("mtime_ns", metadata.st_mtime_ns),
        ("ctime_ns", metadata.st_ctime_ns),
    ):
        descriptor[field] = int(value)
    with pytest.raises(RuntimeError, match="hash/stable snapshot changed"):
        dataset._admitted_payload("S1", "image_nodes", path)


@pytest.mark.parametrize("mutation", ["omitted", "surplus"])
def test_runtime_read_inventory_is_exact_and_closed(tmp_path, mutation):
    state = _dataset_admission_state(tmp_path)
    inventory = state["sample_artifacts"]["S1"]
    if mutation == "omitted":
        inventory.pop("metadata")
    else:
        inventory["unread_cache"] = copy.deepcopy(inventory["metadata"])
    state["record_sha256"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="sample inventory differs"):
        Connect4PrecomputedDataset.from_rank_zero_admission_state(state)


def test_dataset_state_record_tamper_is_rejected(tmp_path):
    state = _dataset_admission_state(tmp_path)
    state["target_shape"] = [64, 64, 64]
    with pytest.raises(RuntimeError, match="state digest differs"):
        Connect4PrecomputedDataset.from_rank_zero_admission_state(state)


def test_transport_digest_is_canonical_sha256():
    payload = canonical_json_bytes(_authority())
    assert len(hashlib.sha256(payload).digest()) == 32
