import ast
import copy
import inspect
from pathlib import Path
import subprocess

import pytest
import torch

from architecture_contract import (
    SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
    SYNTHESIS_ARCHITECTURE_SCHEMA,
    synthesis_architecture_contract,
)
from data.protocol import RUN_ARTIFACT_IDENTITY_SCHEMA, TRAINING_CHECKPOINT_FORMAT
from data.cohort_admission import (
    COHORT_ADMISSION_IDENTITY_SCHEMA,
    build_validation_shard_contract,
    validate_production_training_runtime_identity,
    validate_validation_shard_contract,
)
from data.provenance import canonical_sha256
from training.train import (
    PRODUCTION_DISTRIBUTED_TIMEOUT,
    _rng_state,
    build_rank_local_validation_subset,
    build_loaders,
    per_rank_batch_size,
    require_production_training_runtime,
    validation_shard_indices,
    validate,
    validate_paper_effective_batch,
    validate_resume_checkpoint,
)


def test_training_entrypoint_has_no_final_test_or_slimbrain_execution_path():
    source_path = Path(__file__).resolve().parents[1] / "training" / "train.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_or_defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imported_or_defined.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "evaluate_test" not in imported_or_defined | loaded_names
    assert "build_pretrained_4d_extractor" not in imported_or_defined | loaded_names
    assert "publish_paired_prediction_bundle" not in imported_or_defined | loaded_names
    assert "test_loader" not in loaded_names


def test_paper_profile_stops_until_both_v9_gpu_gates_are_authenticated():
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "connect4.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with pytest.raises(
        RuntimeError,
        match=r"per-rank A100-40GB.*four-GPU DDP",
    ):
        build_loaders(config, world=1, rank=0)


def test_validation_work_is_complete_disjoint_balanced_and_timeout_is_explicit():
    shards = [set(validation_shard_indices(11, 4, rank)) for rank in range(4)]
    assert set.union(*shards) == set(range(11))
    assert sum(len(shard) for shard in shards) == 11
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert PRODUCTION_DISTRIBUTED_TIMEOUT.total_seconds() == 6 * 60 * 60
    validation_source = inspect.getsource(validate)
    assert "_validate_local_shard" in validation_source
    assert "all_gather_object" in validation_source
    assert "broadcast_object_list" not in validation_source

    with pytest.raises(RuntimeError, match="exactly four DDP ranks"):
        require_production_training_runtime(
            {},
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
        )


def test_rank_local_validation_subset_never_opens_unassigned_targets():
    class TargetOpeningDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.opened = []

        def __len__(self):
            return 13

        def __getitem__(self, index):
            self.opened.append(index)
            return index

    admission_identity = _admission_identity_fixture(
        config={"training": {"val_batches": 11}},
        runtime_identity=_production_runtime_identity_fixture(),
    )
    validation_contract = build_validation_shard_contract(
        admission_identity,
        development_partition_size=13,
        global_limit=11,
        world_size=4,
    )
    for rank in range(4):
        dataset = TargetOpeningDataset()
        subset = build_rank_local_validation_subset(
            dataset,
            limit=11,
            world_size=4,
            rank=rank,
            validation_shard_contract=validation_contract,
        )
        observed = [subset[index] for index in range(len(subset))]
        expected = list(validation_shard_indices(11, 4, rank))
        assert observed == expected
        assert dataset.opened == expected
        assert set(dataset.opened).isdisjoint(set(range(11)) - set(expected))
        assert all(index < 11 for index in dataset.opened)


def test_validation_shard_contract_rejects_overlap_even_when_resigned():
    runtime = _production_runtime_identity_fixture()
    admission = _admission_identity_fixture(
        config={"training": {"val_batches": 4}},
        runtime_identity=runtime,
    )
    contract = build_validation_shard_contract(
        admission,
        development_partition_size=8,
        global_limit=4,
        world_size=4,
    )
    assert validate_validation_shard_contract(
        contract,
        expected_cohort_admission_identity=admission,
        expected_world_size=4,
    ) == contract
    tampered = dict(contract)
    tampered["rank_to_global_positions"] = {
        **contract["rank_to_global_positions"],
        "1": [0, 1],
    }
    tampered["record_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="deterministic partition"):
        validate_validation_shard_contract(tampered)


def test_production_launcher_is_portable_and_requires_explicit_inputs():
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "train.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = launcher.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:4" in source
    assert "#SBATCH --partition=cscc-gpu-p" in source
    assert "#SBATCH --qos=cscc-gpu-qos" in source
    assert "#SBATCH --output=slurm-%x-%j.out" in source
    assert "#SBATCH --error=slurm-%x-%j.err" in source
    assert "CONNECT4_CONFIG" in source
    assert "CONNECT4_PYTHON" in source
    assert "mica" not in source.lower()
    assert "torch.distributed.run" in source
    completed = subprocess.run(
        ["bash", str(launcher)], check=False, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "CONNECT4_CONFIG" in completed.stderr


def test_paper_batch_four_is_global_across_four_gpus():
    assert per_rank_batch_size(4, 4) == 1
    assert per_rank_batch_size(4, 1) == 4


def test_global_batch_must_divide_ddp_world_size():
    with pytest.raises(ValueError, match="not divisible"):
        per_rank_batch_size(4, 3)


@pytest.mark.parametrize("batch, world", [(0, 1), (4, 0), (-1, 1)])
def test_global_batch_and_world_must_be_positive(batch, world):
    with pytest.raises(ValueError, match="positive"):
        per_rank_batch_size(batch, world)


def test_gradient_accumulation_is_included_in_effective_paper_batch():
    assert validate_paper_effective_batch(2, 2, 2) == 1
    assert validate_paper_effective_batch(4, 4, 1) == 1
    with pytest.raises(ValueError, match="effective global batch size 4"):
        validate_paper_effective_batch(4, 1, 2)


def _scheduler_digest(job_id: str) -> str:
    return canonical_sha256(
        {
            "schema": "connect4-ciai-scontrol-allocation-identity-v1",
            "slurm_job_id": job_id,
            "slurm_partition": "cscc-gpu-p",
            "slurm_qos": "cscc-gpu-qos",
            "allocated_hostnames": ["compute01"],
            "slurm_num_nodes": 1,
            "allocated_gpu_count": 4,
        }
    )


def _production_runtime_identity_fixture():
    runtime = {
        "schema": "connect4-v10-ciai-four-a100-training-runtime-v1",
        "synthesis_architecture_schema": SYNTHESIS_ARCHITECTURE_SCHEMA,
        "synthesis_architecture_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "training_checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "slurm_job_id": "12345",
        "slurm_partition": "cscc-gpu-p",
        "slurm_qos": "cscc-gpu-qos",
        "allocated_hostnames": ["compute01"],
        "hostname": "compute01",
        "scontrol_record_sha256": _scheduler_digest("12345"),
        "source_tree_sha256": "9" * 64,
        "source_file_count": 100,
        "python_executable": "/fixture/qualified-connect4-v10/bin/python3.10",
        "python_executable_sha256": "a" * 64,
        "isolated_python": True,
        "torch_version": "2.7.0",
        "cuda_version": "12.8",
        "cudnn_version": 90100,
        "amp_dtype": "bfloat16",
        "world_size": 4,
        "gpu_records": [
            {
                "rank": rank,
                "local_rank": rank,
                "gpu_uuid": f"GPU-{rank:04d}",
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "gpu_total_memory_gib": 39.5,
                "nvidia_driver_version": "570.00",
                "cuda_visible_device": str(rank),
            }
            for rank in range(4)
        ],
        "slab_selection": {
            "profile": "recovery",
            "depth_slab_size": 4,
            "summary_path": "/evidence/sweep.json",
            "summary_sha256": "b" * 64,
            "ddp_smoke_path": "/evidence/ddp.json",
            "ddp_smoke_sha256": "c" * 64,
            "measured_ddp_step_seconds": 12.5,
            "recovery_half_epoch_eta_seconds": 6487.5,
        },
    }
    runtime["canonical_sha256"] = canonical_sha256(runtime)
    return validate_production_training_runtime_identity(runtime)


def _admission_identity_fixture(*, config, runtime_identity):
    split_identity = {"format": "connect4_split_identity_v2", "sha256": "a" * 64}
    artifact_identity = {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "sha256": "b" * 64,
        "structural_artifact_identities_sha256": "6" * 64,
    }
    identity = {
        "schema": COHORT_ADMISSION_IDENTITY_SCHEMA,
        "authority_record_sha256": "1" * 64,
        "authority_canonical_bytes_sha256": "2" * 64,
        "data_root_sha256": "3" * 64,
        "runtime_identity_sha256": runtime_identity["canonical_sha256"],
        "config_sha256": canonical_sha256(config),
        "ordered_scan_ids_sha256": "4" * 64,
        "partitions_sha256": "5" * 64,
        "split_identity_sha256": canonical_sha256(split_identity),
        "artifact_identity_sha256": canonical_sha256(artifact_identity),
        "structural_artifact_identities_sha256": "6" * 64,
        "dataset_state_sha256": "7" * 64,
        "world_size": 4,
    }
    identity["record_sha256"] = canonical_sha256(identity)
    return identity


def test_production_runtime_cannot_claim_unsigned_synthesis_launch_identity():
    runtime = _production_runtime_identity_fixture()
    runtime["synthesis_launch_admission_schema"] = (
        "connect4-half-native-v9-launch-admission-v1"
    )
    unsigned = dict(runtime)
    unsigned.pop("canonical_sha256")
    runtime["canonical_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="fields differ"):
        validate_production_training_runtime_identity(runtime)


def test_production_runtime_rejects_revoked_mica_python():
    runtime = _production_runtime_identity_fixture()
    runtime["python_executable"] = "/home/example/.conda/envs/mica/bin/python3.10"
    unsigned = dict(runtime)
    unsigned.pop("canonical_sha256")
    runtime["canonical_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="runtime class differs"):
        validate_production_training_runtime_identity(runtime)


def _resume_fixture():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    config = {"training": {"batch_size": 2, "grad_accum_steps": 2}}
    split_identity = {"format": "connect4_split_identity_v2", "sha256": "a" * 64}
    artifact_identity = {
        "format": RUN_ARTIFACT_IDENTITY_SCHEMA,
        "sha256": "b" * 64,
        "structural_artifact_identities_sha256": "6" * 64,
    }
    runtime_identity = _production_runtime_identity_fixture()
    admission_identity = _admission_identity_fixture(
        config=config,
        runtime_identity=runtime_identity,
    )
    validation_contract = build_validation_shard_contract(
        admission_identity,
        development_partition_size=8,
        global_limit=4,
        world_size=4,
    )
    selection_validation_contract = build_validation_shard_contract(
        admission_identity,
        development_partition_size=8,
        global_limit=8,
        world_size=4,
    )
    checkpoint = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "architecture_contract": synthesis_architecture_contract(),
        "architecture_contract_sha256": SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "split_identity": split_identity,
        "artifact_identity": artifact_identity,
        "runtime_identity": runtime_identity,
        "cohort_admission_identity": admission_identity,
        "validation_shard_contract": validation_contract,
        "selection_validation_shard_contract": selection_validation_contract,
        "world_size": 4,
        "next_epoch": 1,
        "next_batch_index": 2,
        "completed_batches": 12,
        "partial": True,
        "latest_development_qa": None,
        "latest_development_qa_file_sha256": None,
        "selection_development_qa": None,
        "selection_development_qa_file_sha256": None,
        "selection_eligible": False,
        "rng_states": [_rng_state() for _ in range(4)],
    }
    return (
        model,
        optimizer,
        scaler,
        config,
        split_identity,
        artifact_identity,
        checkpoint,
    )


def test_resume_restores_the_exact_next_iteration():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    assert validate_resume_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        split_identity=split_identity,
        artifact_identity=artifact_identity,
        runtime_identity=checkpoint["runtime_identity"],
        cohort_admission_identity=checkpoint["cohort_admission_identity"],
        validation_shard_contract=checkpoint["validation_shard_contract"],
        world_size=4,
        rank=0,
        batches_per_epoch=10,
        grad_accum_steps=2,
    ) == (1, 2, 12)


def test_resume_accepts_a_new_slurm_job_with_the_same_stable_data_root():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    current_runtime = copy.deepcopy(checkpoint["runtime_identity"])
    current_runtime["slurm_job_id"] = "54321"
    current_runtime["scontrol_record_sha256"] = _scheduler_digest("54321")
    current_runtime["canonical_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in current_runtime.items()
            if key != "canonical_sha256"
        }
    )
    current_admission = copy.deepcopy(checkpoint["cohort_admission_identity"])
    current_admission["runtime_identity_sha256"] = current_runtime[
        "canonical_sha256"
    ]
    current_admission["authority_record_sha256"] = "e" * 64
    current_admission["authority_canonical_bytes_sha256"] = "f" * 64
    current_admission["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in current_admission.items()
            if key != "record_sha256"
        }
    )
    current_validation = build_validation_shard_contract(
        current_admission,
        development_partition_size=8,
        global_limit=4,
        world_size=4,
    )

    assert validate_resume_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        split_identity=split_identity,
        artifact_identity=artifact_identity,
        runtime_identity=current_runtime,
        cohort_admission_identity=current_admission,
        validation_shard_contract=current_validation,
        world_size=4,
        rank=0,
        batches_per_epoch=10,
        grad_accum_steps=2,
    ) == (1, 2, 12)


def test_resume_rejects_runtime_not_bound_to_its_admission_receipt():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    tampered = copy.deepcopy(checkpoint)
    tampered["runtime_identity"]["slurm_job_id"] = "54321"
    tampered["runtime_identity"]["scontrol_record_sha256"] = _scheduler_digest(
        "54321"
    )
    tampered["runtime_identity"]["canonical_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered["runtime_identity"].items()
            if key != "canonical_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="admission payload bindings differ"):
        validate_resume_checkpoint(
            tampered,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=tampered["runtime_identity"],
            cohort_admission_identity=tampered["cohort_admission_identity"],
            validation_shard_contract=tampered["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_iteration_checkpoint_is_safe_weights_only_loadable(tmp_path):
    _, _, _, _, _, _, checkpoint = _resume_fixture()
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["format"] == TRAINING_CHECKPOINT_FORMAT
    assert (
        loaded["architecture_contract_sha256"]
        == SYNTHESIS_ARCHITECTURE_CONTRACT_SHA256
    )
    assert loaded["rng_states"][0]["numpy"]["bit_generator"] == "MT19937"


def test_resume_rejects_changed_runtime_contract_and_admission_root():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    changed_runtime = {"schema": "another-runtime", "world_size": 1}
    changed_runtime["canonical_sha256"] = canonical_sha256(changed_runtime)
    with pytest.raises(RuntimeError, match="runtime contract differs"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=changed_runtime,
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )

    changed_admission = dict(checkpoint["cohort_admission_identity"])
    changed_admission["data_root_sha256"] = "8" * 64
    changed_admission["record_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed_admission.items()
            if key != "record_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="admission root differs"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=changed_admission,
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_rejects_stale_v4_run_artifact_identity():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["artifact_identity"] = {
        **checkpoint["artifact_identity"],
        "format": "connect4_run_artifacts_v4",
    }
    with pytest.raises(RuntimeError, match="stale run-artifact identity"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_categorically_rejects_detail_v1_checkpoint():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["format"] = "connect4_iteration_exact_v2"
    checkpoint.pop("architecture_contract")
    checkpoint.pop("architecture_contract_sha256")
    with pytest.raises(RuntimeError, match="pre-v18"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_categorically_rejects_target_derived_decoder_checkpoint():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["format"] = "connect4_iteration_exact_v3"
    with pytest.raises(RuntimeError, match="unweighted legacy one-step"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_rejects_v7_bold_bearing_structural_identity_contract():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["format"] = "connect4_iteration_exact_v7"
    with pytest.raises(RuntimeError, match="target-leaking structural identities"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


@pytest.mark.parametrize(
    "legacy_format",
    (
        "connect4_iteration_exact_v8",
        "connect4_iteration_exact_v9",
        "connect4_iteration_exact_v10",
        "connect4_iteration_exact_v11",
        "connect4_iteration_exact_v12",
        "connect4_iteration_exact_v13",
        "connect4_iteration_exact_v14",
        "connect4_iteration_exact_v15",
        "connect4_iteration_exact_v16",
        "connect4_iteration_exact_v17",
    ),
)
def test_resume_rejects_pre_v18_synthesis_contracts(legacy_format):
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["format"] = legacy_format
    with pytest.raises(RuntimeError, match="pre-v18"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_rejects_self_asserted_or_mutated_detail_v2_contract():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    checkpoint["architecture_contract"]["detail_v1_checkpoint_accepted"] = True
    with pytest.raises(RuntimeError, match="architecture differs"):
        validate_resume_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )


def test_resume_fails_closed_on_config_drift_or_mid_accumulation():
    (
        model, optimizer, scaler, config, split_identity, artifact_identity, checkpoint
    ) = _resume_fixture()
    changed = dict(checkpoint)
    changed["config"] = {"training": {"batch_size": 4, "grad_accum_steps": 1}}
    with pytest.raises(RuntimeError, match="config differs"):
        validate_resume_checkpoint(
            changed,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )

    changed = dict(checkpoint)
    changed["next_batch_index"] = 3
    changed["completed_batches"] = 13
    with pytest.raises(RuntimeError, match="accumulation window"):
        validate_resume_checkpoint(
            changed,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            split_identity=split_identity,
            artifact_identity=artifact_identity,
            runtime_identity=checkpoint["runtime_identity"],
            cohort_admission_identity=checkpoint[
                "cohort_admission_identity"
            ],
            validation_shard_contract=checkpoint["validation_shard_contract"],
            world_size=4,
            rank=0,
            batches_per_epoch=10,
            grad_accum_steps=2,
        )
