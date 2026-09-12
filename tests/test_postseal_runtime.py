from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from eval import postseal
from eval import metrics as general_metrics
from eval import postseal_metrics
from scripts import evaluate_postseal_heldout as cli
from scripts import stage_postseal_runtime as staging


class _TinyScriptedSlimBrain(torch.nn.Module):
    __constants__ = [
        "connect4_model_name",
        "connect4_adapter_contract",
        "connect4_source_revision",
        "connect4_feature_layer",
        "connect4_feature_aggregation",
        "connect4_feature_dimension",
        "connect4_logits_layer",
        "connect4_logits_trained",
        "connect4_class_count",
        "connect4_class_semantics_sha256",
    ]

    def __init__(self, source_revision: str, class_semantics_sha256: str):
        super().__init__()
        self.connect4_model_name = "slimbrain"
        self.connect4_adapter_contract = postseal_metrics.SLIMBRAIN_ADAPTER_CONTRACT
        self.connect4_source_revision = source_revision
        self.connect4_feature_layer = "tiny.features"
        self.connect4_feature_aggregation = "full-volume-test-summary"
        self.connect4_feature_dimension = 4
        self.connect4_logits_layer = "tiny.trained_head"
        self.connect4_logits_trained = True
        self.connect4_class_count = 2
        self.connect4_class_semantics_sha256 = class_semantics_sha256

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        flattened = value.flatten(1)
        mean = flattened.mean(dim=1)
        standard_deviation = flattened.std(dim=1, unbiased=False)
        maximum = flattened.max(dim=1).values
        minimum = flattened.min(dim=1).values
        features = torch.stack((mean, standard_deviation, maximum, minimum), dim=1)
        logits = torch.stack((mean, -mean), dim=1)
        return {"features": features, "logits": logits}


class _TinyFeatureOnlySlimBrain(torch.nn.Module):
    __constants__ = _TinyScriptedSlimBrain.__constants__

    def __init__(self, source_revision: str, class_semantics_sha256: str):
        super().__init__()
        self.connect4_model_name = "slimbrain"
        self.connect4_adapter_contract = postseal_metrics.SLIMBRAIN_ADAPTER_CONTRACT
        self.connect4_source_revision = source_revision
        self.connect4_feature_layer = "tiny.features"
        self.connect4_feature_aggregation = "full-volume-test-summary"
        self.connect4_feature_dimension = 4
        self.connect4_logits_layer = "tiny.trained_head"
        self.connect4_logits_trained = True
        self.connect4_class_count = 2
        self.connect4_class_semantics_sha256 = class_semantics_sha256

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        flattened = value.flatten(1)
        features = torch.stack(
            (
                flattened.mean(dim=1),
                flattened.std(dim=1, unbiased=False),
                flattened.max(dim=1).values,
                flattened.min(dim=1).values,
            ),
            dim=1,
        )
        return {"features": features}


class _SubstituteSlimBrain(torch.nn.Module):
    """Authority-compatible shape whose values expose any path-based reload."""

    __constants__ = _TinyScriptedSlimBrain.__constants__ + [
        "connect4_substitute_marker"
    ]

    def __init__(self, source_revision: str, class_semantics_sha256: str):
        super().__init__()
        self.connect4_model_name = "slimbrain"
        self.connect4_adapter_contract = postseal_metrics.SLIMBRAIN_ADAPTER_CONTRACT
        self.connect4_source_revision = source_revision
        self.connect4_feature_layer = "tiny.features"
        self.connect4_feature_aggregation = "full-volume-test-summary"
        self.connect4_feature_dimension = 4
        self.connect4_logits_layer = "tiny.trained_head"
        self.connect4_logits_trained = True
        self.connect4_class_count = 2
        self.connect4_class_semantics_sha256 = class_semantics_sha256
        self.connect4_substitute_marker = "ATTACKER_CHECKPOINT"

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = value.shape[0]
        features = torch.full((batch, 4), 777.0, device=value.device)
        logits = torch.full((batch, 2), 999.0, device=value.device)
        return {"features": features, "logits": logits}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _slimbrain_authority_fixture(
    tmp_path: Path,
) -> tuple[Path, str, str, str]:
    source_revision = "d" * 40
    class_semantics_sha256 = "e" * 64
    checkpoint = tmp_path / "slimbrain.torchscript"
    torch.jit.script(
        _TinyScriptedSlimBrain(source_revision, class_semantics_sha256)
    ).save(str(checkpoint))
    source_manifest = tmp_path / "slimbrain-source.manifest"
    source_manifest.write_text(f"source_revision={source_revision}\n", encoding="ascii")
    dependency_authority_sha256 = "a" * 64
    python_executable_sha256 = "b" * 64
    adapter_source = Path(postseal_metrics.__file__).resolve()
    record = {
        "schema": postseal_metrics.SLIMBRAIN_AUTHORITY_SCHEMA,
        "status": "QUALIFIED_IMMUTABLE_EVALUATION_ONLY",
        "model_name": "slimbrain",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "source": {
            "path": str(source_manifest),
            "sha256": _sha256(source_manifest),
            "size_bytes": source_manifest.stat().st_size,
            "revision": source_revision,
        },
        "adapter": {
            "contract": postseal_metrics.SLIMBRAIN_ADAPTER_CONTRACT,
            "implementation_source_sha256": _sha256(adapter_source),
            "input_semantics": postseal_metrics.SLIMBRAIN_INPUT_SEMANTICS,
        },
        "feature_output": {
            "output_key": "features",
            "layer_name": "tiny.features",
            "aggregation": "full-volume-test-summary",
            "dimension": 4,
        },
        "logits_output": {
            "output_key": "logits",
            "layer_name": "tiny.trained_head",
            "class_count": 2,
            "trained": True,
            "class_semantics_sha256": class_semantics_sha256,
        },
        "runtime": {
            "dependency_authority_sha256": dependency_authority_sha256,
            "python_executable_sha256": python_executable_sha256,
            "torch_version": str(torch.__version__),
            "device_type": "cpu",
        },
        "restrictions": {
            "environment_indirection": False,
            "generic_or_pickle_model_loading": False,
            "data_derived_feature_mask": False,
            "single_subject_distributional_metrics": False,
            "training_or_selection_feedback": False,
            "target_access_before_prediction_set_seal": False,
        },
    }
    record["record_sha256"] = _canonical_sha256(record)
    authority = tmp_path / "slimbrain-authority.json"
    authority.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    for path in (checkpoint, source_manifest, authority):
        path.chmod(0o444)
    return (
        authority,
        _sha256(authority),
        dependency_authority_sha256,
        python_executable_sha256,
    )


def _rebind_slimbrain_checkpoint(authority: Path) -> str:
    authority.chmod(0o644)
    record = json.loads(authority.read_text(encoding="ascii"))
    checkpoint = Path(record["checkpoint"]["path"])
    record["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    record.pop("record_sha256")
    record["record_sha256"] = _canonical_sha256(record)
    authority.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    authority.chmod(0o444)
    return _sha256(authority)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, child_directories, file_names in os.walk(root, topdown=False):
        os.chmod(directory, 0o700)
        for file_name in file_names:
            os.chmod(Path(directory) / file_name, 0o600)
        for child in child_directories:
            os.chmod(Path(directory) / child, 0o700)


def _preflight(
    launcher: Path,
    runtime_root: Path,
    manifest_sha256: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "CONNECT4_RUNTIME_ROOT": str(runtime_root),
            "CONNECT4_RUNTIME_MANIFEST_SHA256": manifest_sha256,
            "CONNECT4_SLURM_LAUNCHER_SHA256": _sha256(launcher),
            "CONNECT4_RUNTIME_PREFLIGHT_ONLY": "1",
        },
    )


def test_all_closed_runtime_inventories_include_texture_gate_and_dependencies() -> None:
    required = {
        "architecture_contract.py",
        "scripts/visualize_texture_audit.py",
        "scripts/visualize_4d_comparison.py",
        "utils/source_provenance.py",
        "utils/spatial_detail.py",
    }
    assert required <= set(staging.RUNTIME_FILES)
    assert required <= set(cli._EXPECTED_RUNTIME_FILES)  # noqa: SLF001
    assert staging.RUNTIME_FILES == cli._EXPECTED_RUNTIME_FILES  # noqa: SLF001
    assert all(not path.startswith("tests/") for path in staging.RUNTIME_FILES)
    assert "tests/postseal_test_support.py" not in staging.RUNTIME_FILES
    evaluator_source = Path(postseal.__file__).read_text(encoding="utf-8")
    assert "_TEST_ONLY_CAPABILITY" not in evaluator_source
    assert "_for_test" not in evaluator_source
    for removed_stage_b_seam in (
        "_EXECUTION_BOUNDARY_CAPABILITY",
        "_ExecutionBoundaryAuthority",
        "_TargetBoundaryGrant",
        "_mint_target_boundary_grant",
        "_StageBVerifierAuthority",
        "_LoadedStageBVerifier",
        "_load_stage_b_verifier",
        "_open_authenticated_stage_b_targets",
        "sealed_target_stage_b.py",
        "AuthenticatedPredictionSet",
    ):
        assert removed_stage_b_seam not in evaluator_source
    launcher = (
        Path(postseal.__file__).resolve().parents[1]
        / "scripts"
        / ("run_postseal_heldout_evaluation.slurm")
    )
    launcher_source = launcher.read_text(encoding="utf-8")
    for relative_path in required:
        assert relative_path in launcher_source


def test_executable_stage_b_surface_is_isolated_in_staged_production_executor() -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    forbidden = {
        "_EXECUTION_BOUNDARY_CAPABILITY",
        "_ExecutionBoundaryAuthority",
        "_TargetBoundaryGrant",
        "_mint_target_boundary_grant",
        "_StageBVerifierAuthority",
        "_LoadedStageBVerifier",
        "_load_stage_b_verifier",
        "_open_authenticated_stage_b_targets",
        "sealed_target_stage_b.py",
    }
    executor_path = "eval/postseal_execution.py"
    assert executor_path in staging.RUNTIME_FILES
    for relative_path in staging.RUNTIME_FILES:
        if relative_path == executor_path:
            continue
        payload = (repository / relative_path).read_text(encoding="utf-8")
        assert forbidden.isdisjoint(payload.split())
        assert all(token not in payload for token in forbidden)
    executor_source = (repository / executor_path).read_text(encoding="utf-8")
    assert all(
        token in executor_source
        for token in {
            "_EXECUTION_BOUNDARY_CAPABILITY",
            "_ExecutionBoundaryAuthority",
            "_TargetBoundaryGrant",
            "_load_stage_b_verifier",
            "_open_authenticated_stage_b_targets",
        }
    )
    support = repository / "tests" / "postseal_test_support.py"
    assert support.is_file()
    assert support.relative_to(repository).as_posix() not in staging.RUNTIME_FILES


def test_runtime_stager_and_shell_authenticate_exact_read_only_tree_before_python(
    tmp_path: Path,
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "immutable-runtime"
    try:
        receipt = staging.stage_runtime(repository, destination)
        assert receipt["status"] == "STAGED_IMMUTABLE_NO_REPLACE"
        assert receipt["file_count"] == len(staging.RUNTIME_FILES)
        assert destination.stat().st_mode & 0o777 == 0o555
        assert (destination / staging.RUNTIME_MANIFEST).stat().st_mode & 0o777 == 0o444
        assert tuple(item["relative_path"] for item in receipt["files"]) == tuple(
            sorted(staging.RUNTIME_FILES)
        )
        launcher = repository / "scripts" / "run_postseal_heldout_evaluation.slurm"
        result = _preflight(launcher, destination, str(receipt["manifest_sha256"]))
        assert result.returncode == 0, result.stderr
        assert "authenticated runtime" in result.stdout

        with pytest.raises(FileExistsError):
            staging.stage_runtime(repository, destination)

        evaluator = destination / "eval" / "postseal.py"
        os.chmod(evaluator, 0o644)
        with evaluator.open("ab") as stream:
            stream.write(b"\n# tamper\n")
        tampered = _preflight(launcher, destination, str(receipt["manifest_sha256"]))
        assert tampered.returncode == 2
        assert "mutable" in tampered.stderr or "SHA-256 differs" in tampered.stderr
    finally:
        _make_writable(destination)


def test_runtime_stager_rejects_destination_alias(tmp_path: Path) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises((FileExistsError, staging.RuntimeStagingError)):
        staging.stage_runtime(repository, alias)


def test_runtime_staging_is_byte_deterministic(tmp_path: Path) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    first = tmp_path / "runtime-a"
    second = tmp_path / "runtime-b"
    try:
        first_receipt = staging.stage_runtime(repository, first)
        second_receipt = staging.stage_runtime(repository, second)
        assert first_receipt["manifest_sha256"] == second_receipt["manifest_sha256"]
        assert (first / staging.RUNTIME_MANIFEST).read_bytes() == (
            second / staging.RUNTIME_MANIFEST
        ).read_bytes()
        assert first_receipt["files"] == second_receipt["files"]
    finally:
        _make_writable(first)
        _make_writable(second)


def test_runtime_stager_chmod_setup_failure_leaves_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "chmod-failure-runtime"
    original_chmod = staging.os.chmod
    injected = False

    def fail_initial_chmod(path, mode, *args, **kwargs):
        nonlocal injected
        if not injected and Path(path).name.startswith(f".{destination.name}.staged-"):
            injected = True
            raise OSError("injected initial chmod failure")
        return original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(staging.os, "chmod", fail_initial_chmod)
    with pytest.raises(OSError, match="injected initial chmod failure"):
        staging.stage_runtime(repository, destination)

    assert injected
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.staged-*")) == []


def test_runtime_stager_reauthenticates_mutation_immediately_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "precommit-mutation-runtime"
    original_rename = staging._rename_directory_noreplace  # noqa: SLF001

    def mutate_then_rename(source, output, *, expected_tree):
        victim = source / "eval" / "postseal.py"
        original_chmod = staging.os.chmod
        original_chmod(victim, 0o644)
        victim.write_bytes(victim.read_bytes() + b"\n# injected mutation\n")
        original_chmod(victim, 0o444)
        return original_rename(source, output, expected_tree=expected_tree)

    monkeypatch.setattr(
        staging,
        "_rename_directory_noreplace",
        mutate_then_rename,
    )
    with pytest.raises(staging.RuntimeStagingError, match="identity or bytes changed"):
        staging.stage_runtime(repository, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.staged-*")) == []


def test_runtime_stager_preserves_rename_when_parent_fsync_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "uncertain-runtime"

    def fail_parent_fsync(_path: Path) -> None:
        raise OSError("injected parent fsync failure")

    monkeypatch.setattr(staging, "_fsync_parent_directory", fail_parent_fsync)
    try:
        with pytest.raises(
            staging.RuntimePublicationUncertainError,
            match="rename committed no-replace",
        ):
            staging.stage_runtime(repository, destination)
        assert destination.is_dir()
        assert (destination / staging.RUNTIME_MANIFEST).is_file()
        assert list(tmp_path.glob(f".{destination.name}.staged-*")) == []
    finally:
        _make_writable(destination)


def test_runtime_stager_verifies_destination_tree_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "postcommit-mutation-runtime"
    original_reauthenticate = staging._reauthenticate_runtime_tree  # noqa: SLF001
    injected = False

    def mutate_destination(root: Path, expected) -> None:
        nonlocal injected
        if root == destination and not injected:
            injected = True
            victim = root / "eval" / "postseal.py"
            staging.os.chmod(victim, 0o644)
            victim.write_bytes(victim.read_bytes() + b"\n# post-rename mutation\n")
            staging.os.chmod(victim, 0o444)
        original_reauthenticate(root, expected)

    monkeypatch.setattr(
        staging,
        "_reauthenticate_runtime_tree",
        mutate_destination,
    )
    try:
        with pytest.raises(
            staging.RuntimePublicationUncertainError,
            match="rename committed no-replace",
        ):
            staging.stage_runtime(repository, destination)
        assert injected
        assert destination.is_dir()
        assert list(tmp_path.glob(f".{destination.name}.staged-*")) == []
    finally:
        _make_writable(destination)


def test_cli_top_level_is_stdlib_only_and_authenticates_before_project_import() -> None:
    source = Path(cli.__file__).resolve()
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "socket",
        "stat",
        "subprocess",
        "sys",
        "typing",
    }
    main_source = ast.get_source_segment(
        source.read_text(encoding="utf-8"),
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
    )
    assert main_source is not None
    assert main_source.index("_authenticate_runtime(arguments)") < main_source.index(
        "from eval.postseal import"
    )
    assert main_source.index("_require_ciai_gpu_allocation()") < main_source.index(
        "from eval.postseal import"
    )
    assert main_source.index("_authenticate_closed_dependency_runtime(") < (
        main_source.index("from eval.postseal import")
    )
    assert "site.main" not in main_source


@pytest.mark.parametrize("threat", ["pth", "import-hook", "dependency-mutation"])
def test_closed_runtime_stop_precedes_untrusted_dependency_code(
    tmp_path: Path, threat: str
) -> None:
    sentinel = tmp_path / f"{threat}-executed"
    malicious = tmp_path / f"malicious-{threat}.pth"
    malicious.write_text(
        f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    with pytest.raises(cli.BootstrapError, match="closed dependency runtime"):
        cli._authenticate_closed_dependency_runtime(  # noqa: SLF001
            SimpleNamespace(dependency_root=tmp_path),
            {"source_runtime": "authenticated"},
        )
    assert not sentinel.exists()


def test_closed_dependency_runtime_v2_authenticates_exact_no_pth_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dependencies"
    package = root / "site-packages" / "tinydep"
    package.mkdir(parents=True)
    package_init = package / "__init__.py"
    package_init.write_text("VALUE = 1\n", encoding="ascii")
    manifest = root / "dependency-tree.sha256"
    manifest.write_text(
        f"{_sha256(package_init)}  site-packages/tinydep/__init__.py\n",
        encoding="ascii",
    )
    python_sha256 = "b" * 64
    for path in (package_init, manifest):
        path.chmod(0o444)
    for directory in (package, root / "site-packages", root):
        directory.chmod(0o555)
    authority_record = {
        "schema": "connect4-postseal-closed-dependency-runtime-v2",
        "status": "QUALIFIED_IMMUTABLE_NO_SITE_NO_PTH",
        "dependency_root": str(root),
        "tree_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
            "size_bytes": manifest.stat().st_size,
        },
        "import_roots": ["site-packages"],
        "python_executable_sha256": python_sha256,
        "site_enabled": False,
        "pth_execution": False,
        "environment_indirection": False,
    }
    authority_record["record_sha256"] = _canonical_sha256(authority_record)
    authority = tmp_path / "dependency-authority.json"
    authority.write_text(
        json.dumps(authority_record, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    authority.chmod(0o444)
    try:
        evidence = cli._authenticate_closed_dependency_runtime(  # noqa: SLF001
            SimpleNamespace(
                dependency_runtime_authority=authority,
                dependency_runtime_authority_sha256=_sha256(authority),
            ),
            {"python_executable": {"sha256": python_sha256}},
        )
        assert evidence["schema"].endswith("evidence-v2")
        assert evidence["file_count"] == 1
        assert evidence["import_roots"] == [str(root / "site-packages")]
        assert evidence["site_enabled"] is False
        assert evidence["pth_execution"] is False
    finally:
        authority.chmod(0o600)
        _make_writable(root)


def _staged_cli_command(
    destination: Path,
    receipt: dict[str, object],
    *,
    python_sha256: str,
) -> list[str]:
    digest = "a" * 64
    return [
        sys.executable,
        "-B",
        "-I",
        "-S",
        str(destination / "scripts" / "evaluate_postseal_heldout.py"),
        "evaluate",
        "--prediction-root",
        str(destination / "does-not-exist-predictions"),
        "--stage-b-completed-set",
        str(destination / "does-not-exist-targets.json"),
        "--stage-b-completed-set-sha256",
        digest,
        "--runtime-manifest",
        str(destination / staging.RUNTIME_MANIFEST),
        "--runtime-manifest-sha256",
        str(receipt["manifest_sha256"]),
        "--slurm-launcher-sha256",
        next(
            str(binding["sha256"])
            for binding in receipt["files"]  # type: ignore[index,union-attr]
            if binding["relative_path"]  # type: ignore[index]
            == "scripts/run_postseal_heldout_evaluation.slurm"
        ),
        "--python-executable-sha256",
        python_sha256,
        "--dependency-runtime-authority",
        str(destination / "dependency-runtime-authority.json"),
        "--dependency-runtime-authority-sha256",
        digest,
        "--execution-authority",
        str(destination / "execution-authority.json"),
        "--execution-authority-sha256",
        digest,
        "--output-root",
        str(destination.parent / "must-not-exist-output"),
        "--prediction-set-seal-sha256",
        digest,
        "--synthesis-checkpoint-sha256",
        digest,
        "--training-run-artifact-identity-sha256",
        digest,
        "--training-target-artifact-identities-sha256",
        digest,
        "--spatial-authority-sha256",
        digest,
    ]


def test_direct_cli_cannot_bypass_preimport_shell_authentication(
    tmp_path: Path,
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "immutable-runtime"
    try:
        receipt = staging.stage_runtime(repository, destination)
        command = _staged_cli_command(
            destination,
            receipt,
            python_sha256=_sha256(Path(sys.executable).resolve()),
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "CONNECT4_RUNTIME_PREIMPORT_AUTHENTICATED"}
        }
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode != 0
        assert "shell boundary before import" in result.stderr
        assert not (destination.parent / "must-not-exist-output").exists()
    finally:
        _make_writable(destination)


def test_cli_rejects_python_executable_pin_before_project_import(
    tmp_path: Path,
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "immutable-runtime"
    try:
        receipt = staging.stage_runtime(repository, destination)
        command = _staged_cli_command(
            destination,
            receipt,
            python_sha256="0" * 64,
        )
        environment = {
            key: value for key, value in os.environ.items() if key != "PYTHONPATH"
        }
        environment["CONNECT4_RUNTIME_PREIMPORT_AUTHENTICATED"] = "1"
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode != 0
        assert "Python executable differs from its external pin" in result.stderr
        assert not (destination.parent / "must-not-exist-output").exists()
    finally:
        _make_writable(destination)


def test_authenticated_verify_cli_fails_closed_before_site_or_pth_execution(
    tmp_path: Path,
) -> None:
    repository = Path(postseal.__file__).resolve().parents[1]
    destination = tmp_path / "immutable-runtime"
    try:
        receipt = staging.stage_runtime(repository, destination)
        files = receipt["files"]
        assert isinstance(files, list)
        launcher_sha256 = next(
            str(binding["sha256"])
            for binding in files
            if isinstance(binding, dict)
            and binding.get("relative_path")
            == "scripts/run_postseal_heldout_evaluation.slurm"
        )
        command = [
            sys.executable,
            "-B",
            "-I",
            "-S",
            str(destination / "scripts" / "evaluate_postseal_heldout.py"),
            "verify-publication",
            "--output-root",
            str(destination.parent / "does-not-exist-publication"),
            "--expected-evaluation-sha256",
            "a" * 64,
            "--expected-publication-receipt-sha256",
            "b" * 64,
            "--expected-prediction-set-seal-path",
            str(destination / "never-open-prediction-seal.json"),
            "--expected-prediction-set-seal-sha256",
            "c" * 64,
            "--expected-stage-b-completed-set-path",
            str(destination / "never-open-stage-b-completed.json"),
            "--expected-stage-b-completed-set-sha256",
            "d" * 64,
            "--expected-stage-b-verifier-source-path",
            str(destination / "never-open-stage-b-source.py"),
            "--expected-stage-b-verifier-source-sha256",
            "e" * 64,
            "--expected-stage-b-verifier-dependency-path",
            str(destination / "never-open-stage-b-dependency.py"),
            "--expected-stage-b-verifier-dependency-sha256",
            "f" * 64,
            "--runtime-manifest",
            str(destination / staging.RUNTIME_MANIFEST),
            "--runtime-manifest-sha256",
            str(receipt["manifest_sha256"]),
            "--slurm-launcher-sha256",
            launcher_sha256,
            "--python-executable-sha256",
            _sha256(Path(sys.executable).resolve()),
        ]
        environment = {
            key: value for key, value in os.environ.items() if key != "PYTHONPATH"
        }
        environment["CONNECT4_RUNTIME_PREIMPORT_AUTHENTICATED"] = "1"
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode != 0
        assert "closed dependency runtime authority v2 is required" in result.stderr
        assert "No module named" not in result.stderr
    finally:
        _make_writable(destination)


def test_postseal_metric_module_has_no_model_training_or_inference_import() -> None:
    source = Path(postseal.__file__).resolve().with_name("postseal_metrics.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint({"models", "training", "inference"})
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(source.parents[1])!r}); "
                "import eval.postseal_metrics; "
                "assert not any(name == 'models' or name.startswith('models.') "
                "or name == 'training' or name.startswith('training.') "
                "or name == 'inference' or name.startswith('inference.') "
                "for name in sys.modules)"
            ),
        ],
        check=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )


def test_postseal_metric_math_matches_the_general_paper_metric_implementation() -> None:
    generator = torch.Generator().manual_seed(20260901)
    target = torch.rand((2, 6, 7, 8, 9), generator=generator)
    predicted = (target * 0.91 + 0.04).clamp(0.0, 1.0)
    mask = torch.ones((2, 7, 8, 9), dtype=torch.bool)
    mask[:, :1] = False
    roi_masks = torch.zeros((2, 4, 7, 8, 9), dtype=torch.bool)
    roi_masks[:, 0, :, :4, :] = mask[:, :, :4, :]
    roi_masks[:, 1, :, 4:, :] = mask[:, :, 4:, :]
    roi_masks[:, 2, :3, :, :] = mask[:, :3, :, :]
    roi_masks[:, 3, 3:, :, :] = mask[:, 3:, :, :]
    expected = general_metrics.compute_all(
        predicted, target, roi_masks=roi_masks, mask=mask
    )
    observed = postseal_metrics.compute_all(
        predicted, target, roi_masks=roi_masks, mask=mask
    )
    assert set(observed) == set(expected)
    for name in expected:
        assert observed[name] == pytest.approx(expected[name], abs=1e-7, rel=1e-7)


def test_authenticated_slimbrain_computes_complete_cohort_fid_is_and_evidence(
    tmp_path: Path,
) -> None:
    (
        authority_path,
        authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    extractor, authority = postseal_metrics.load_authenticated_slimbrain(
        authority_path,
        expected_authority_sha256=authority_sha256,
        dependency_authority_sha256=dependency_authority_sha256,
        python_executable_sha256=python_executable_sha256,
        device=torch.device("cpu"),
    )
    accumulator = postseal_metrics.SynthesisMetricAccumulator(extractor)
    base = torch.linspace(0.05, 0.95, 2 * 3 * 4 * 5).reshape(1, 1, 2, 3, 4, 5)
    for index in range(len(postseal.EXPECTED_SCAN_IDS)):
        offset = float(index) / 1000.0
        target = (base + offset).clamp(0.0, 1.0)
        predicted = (target * 0.91 + 0.03).clamp(0.0, 1.0)
        accumulator.update_distributional(predicted, target)
    evidence = postseal._distributional_metrics_available(  # noqa: SLF001
        accumulator=accumulator,
        artifact_root=tmp_path,
        prediction_set_seal_sha256="1" * 64,
        stage_b_completed_set_sha256="2" * 64,
    )
    assert evidence["sample_count"] == 34
    assert evidence["feature_mask"] == "none-full-authenticated-volume"
    assert evidence["fid"] >= 0.0
    assert evidence["inception_score"] >= 1.0
    assert set(evidence["feature_artifacts"]) == {
        "generated_features",
        "real_features",
        "generated_logits",
    }
    assert all(
        (tmp_path / descriptor["relative_path"]).is_file()
        for descriptor in evidence["feature_artifacts"].values()
    )
    assert (
        postseal._validate_distributional_evidence(  # noqa: SLF001
            evidence,
            expected_authority_sha256=authority_sha256,
            artifact_root=tmp_path,
        )
        is True
    )
    with pytest.raises(
        postseal.PostsealEvaluationError, match="authority external pin"
    ):
        postseal._validate_distributional_evidence(  # noqa: SLF001
            evidence, expected_authority_sha256="f" * 64
        )
    with pytest.raises(
        postseal.PostsealEvaluationError, match="external authority pin"
    ):
        postseal._validate_distributional_evidence(evidence)  # noqa: SLF001
    forged = json.loads(json.dumps(evidence))
    forged["model_authority"]["feature_output"]["layer_name"] = "forged.layer"
    forged.pop("record_sha256")
    forged["record_sha256"] = _canonical_sha256(forged)
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="authority/publication cross-binding",
    ):
        postseal._validate_distributional_evidence(  # noqa: SLF001
            forged, expected_authority_sha256=authority_sha256
        )
    fabricated_metrics = json.loads(json.dumps(evidence))
    fabricated_metrics["fid"] = 0.0
    fabricated_metrics["inception_score"] = 999.0
    fabricated_metrics.pop("record_sha256")
    fabricated_metrics["record_sha256"] = _canonical_sha256(fabricated_metrics)
    with pytest.raises(
        postseal.PostsealEvaluationError,
        match="published (FID|Inception Score) differs",
    ):
        postseal._validate_distributional_evidence(  # noqa: SLF001
            fabricated_metrics,
            expected_authority_sha256=authority_sha256,
            artifact_root=tmp_path,
        )
    for name, descriptor in evidence["feature_artifacts"].items():
        artifact = tmp_path / descriptor["relative_path"]
        original_payload = artifact.read_bytes()
        tampered_payload = bytearray(original_payload)
        tampered_payload[0] ^= 1
        artifact.write_bytes(tampered_payload)
        with pytest.raises(
            postseal.PostsealEvaluationError,
            match=rf"published distributional {name} SHA-256 differs",
        ):
            postseal._validate_distributional_evidence(  # noqa: SLF001
                evidence,
                expected_authority_sha256=authority_sha256,
                artifact_root=tmp_path,
            )
        artifact.write_bytes(original_payload)
    authority.reauthenticate()


def test_slimbrain_torchscript_load_uses_captured_bytes_during_swap_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        authority_path,
        authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    record = json.loads(authority_path.read_text(encoding="ascii"))
    checkpoint = Path(record["checkpoint"]["path"])
    substitute = tmp_path / "attacker-substitute.pt"
    torch.jit.script(
        _SubstituteSlimBrain(
            record["source"]["revision"],
            record["logits_output"]["class_semantics_sha256"],
        )
    ).save(str(substitute))
    substitute.chmod(0o444)
    detached = tmp_path / "authenticated-checkpoint.detached"
    original_load = postseal_metrics.torch.jit.load
    load_inputs: list[object] = []
    loaded_markers: list[object] = []

    def swap_restore_load(source, *args, **kwargs):
        checkpoint.rename(detached)
        substitute.rename(checkpoint)
        try:
            loaded = original_load(source, *args, **kwargs)
            load_inputs.append(source)
            loaded_markers.append(getattr(loaded, "connect4_substitute_marker", None))
            return loaded
        finally:
            checkpoint.rename(substitute)
            detached.rename(checkpoint)

    monkeypatch.setattr(postseal_metrics.torch.jit, "load", swap_restore_load)
    extractor, authority = postseal_metrics.load_authenticated_slimbrain(
        authority_path,
        expected_authority_sha256=authority_sha256,
        dependency_authority_sha256=dependency_authority_sha256,
        python_executable_sha256=python_executable_sha256,
        device=torch.device("cpu"),
    )
    assert len(load_inputs) == 1
    assert not isinstance(load_inputs[0], (str, Path))
    assert loaded_markers == [None]
    output = extractor(torch.zeros((1, 1, 2, 3, 4, 5)))
    assert torch.equal(output["features"], torch.zeros((1, 4)))
    assert torch.equal(output["logits"], torch.zeros((1, 2)))
    authority.reauthenticate()


def test_distributional_accumulator_rejects_every_generic_extractor(
    tmp_path: Path,
) -> None:
    class GenericExtractor(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "features": value.flatten(1),
                "logits": torch.ones((value.shape[0], 2)),
            }

    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="exact authenticated SLIM-Brain extractor",
    ):
        postseal_metrics.SynthesisMetricAccumulator(GenericExtractor())
    accumulator = postseal_metrics.SynthesisMetricAccumulator()
    volume = torch.zeros((1, 1, 2, 3, 4, 5))
    roi_masks = torch.ones((1, 1, 3, 4, 5))
    mask = torch.ones((1, 3, 4, 5))
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="generic or masked post-seal accumulation is forbidden",
    ):
        accumulator.update(volume, volume, roi_masks, mask=mask)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="exact authenticated SLIM-Brain extractor",
    ):
        accumulator.update_distributional(volume, volume)
    with pytest.raises(postseal_metrics.SlimBrainAuthorityError):
        accumulator.distributional_tensors()
    with pytest.raises(postseal_metrics.SlimBrainAuthorityError):
        postseal._distributional_metrics_available(  # noqa: SLF001
            accumulator=accumulator,
            artifact_root=tmp_path,
            prediction_set_seal_sha256="a" * 64,
            stage_b_completed_set_sha256="b" * 64,
        )
    assert accumulator.num_samples == 0
    assert not (tmp_path / "distributional_evidence").exists()


def test_slimbrain_authority_rejects_resigned_semantic_forgery(
    tmp_path: Path,
) -> None:
    (
        authority_path,
        _authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    authority_path.chmod(0o644)
    record = json.loads(authority_path.read_text(encoding="ascii"))
    record["feature_output"]["layer_name"] = "attacker.substitute"
    record.pop("record_sha256")
    record["record_sha256"] = _canonical_sha256(record)
    authority_path.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    authority_path.chmod(0o444)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="TorchScript attributes differ",
    ):
        postseal_metrics.load_authenticated_slimbrain(
            authority_path,
            expected_authority_sha256=_sha256(authority_path),
            dependency_authority_sha256=dependency_authority_sha256,
            python_executable_sha256=python_executable_sha256,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "indirect_path", ("$CONNECT4_SLIMBRAIN_CHECKPOINT", "~/slimbrain-checkpoint.pt")
)
def test_slimbrain_authority_rejects_environment_checkpoint_indirection(
    tmp_path: Path, indirect_path: str
) -> None:
    (
        authority_path,
        _authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    authority_path.chmod(0o644)
    record = json.loads(authority_path.read_text(encoding="ascii"))
    record["checkpoint"]["path"] = indirect_path
    record.pop("record_sha256")
    record["record_sha256"] = _canonical_sha256(record)
    authority_path.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    authority_path.chmod(0o444)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError, match="environment indirection"
    ):
        postseal_metrics.load_authenticated_slimbrain(
            authority_path,
            expected_authority_sha256=_sha256(authority_path),
            dependency_authority_sha256=dependency_authority_sha256,
            python_executable_sha256=python_executable_sha256,
            device=torch.device("cpu"),
        )


def test_slimbrain_authority_rejects_intermediate_parent_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (
        authority_path,
        authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(real_parent)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError, match="aliases another path"
    ):
        postseal_metrics.load_authenticated_slimbrain(
            alias_parent / authority_path.name,
            expected_authority_sha256=authority_sha256,
            dependency_authority_sha256=dependency_authority_sha256,
            python_executable_sha256=python_executable_sha256,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "authority_path", ("$CONNECT4_SLIMBRAIN_AUTHORITY", "~/slimbrain-authority.json")
)
def test_slimbrain_authority_rejects_indirection_before_preliminary_stat(
    authority_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stat_calls: list[str] = []

    def forbidden_stat(path: Path, *_args, **_kwargs):
        stat_calls.append(str(path))
        raise AssertionError("invalid authority path was inspected before validation")

    monkeypatch.setattr(Path, "stat", forbidden_stat)
    monkeypatch.setattr(Path, "lstat", forbidden_stat)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError, match="environment indirection"
    ):
        postseal_metrics.load_authenticated_slimbrain(
            authority_path,
            expected_authority_sha256="a" * 64,
            dependency_authority_sha256="b" * 64,
            python_executable_sha256="c" * 64,
            device=torch.device("cpu"),
        )
    assert stat_calls == []


def test_slimbrain_rejects_missing_logits_mask_and_single_subject_substitution(
    tmp_path: Path,
) -> None:
    (
        authority_path,
        _authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    record = json.loads(authority_path.read_text(encoding="ascii"))
    checkpoint = Path(record["checkpoint"]["path"])
    checkpoint.chmod(0o644)
    torch.jit.script(
        _TinyFeatureOnlySlimBrain(
            record["source"]["revision"],
            record["logits_output"]["class_semantics_sha256"],
        )
    ).save(str(checkpoint))
    checkpoint.chmod(0o444)
    authority_sha256 = _rebind_slimbrain_checkpoint(authority_path)
    extractor, _authority = postseal_metrics.load_authenticated_slimbrain(
        authority_path,
        expected_authority_sha256=authority_sha256,
        dependency_authority_sha256=dependency_authority_sha256,
        python_executable_sha256=python_executable_sha256,
        device=torch.device("cpu"),
    )
    accumulator = postseal_metrics.SynthesisMetricAccumulator(extractor)
    volume = torch.zeros(1, 1, 2, 3, 4, 5)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="exactly trained features and logits",
    ):
        accumulator.update_distributional(volume, volume)

    valid_authority = tmp_path / "valid"
    valid_authority.mkdir()
    (
        valid_authority_path,
        valid_authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(valid_authority)
    valid_extractor, _authority = postseal_metrics.load_authenticated_slimbrain(
        valid_authority_path,
        expected_authority_sha256=valid_authority_sha256,
        dependency_authority_sha256=dependency_authority_sha256,
        python_executable_sha256=python_executable_sha256,
        device=torch.device("cpu"),
    )
    valid_accumulator = postseal_metrics.SynthesisMetricAccumulator(valid_extractor)
    with pytest.raises(TypeError, match="unexpected keyword argument 'mask'"):
        valid_accumulator.update_distributional(volume, volume, mask=volume)
    valid_accumulator.update_distributional(volume, volume)
    with pytest.raises(ValueError, match="at least two samples"):
        valid_accumulator.compute()


def test_slimbrain_rejects_generic_pickle_checkpoint(tmp_path: Path) -> None:
    (
        authority_path,
        _authority_sha256,
        dependency_authority_sha256,
        python_executable_sha256,
    ) = _slimbrain_authority_fixture(tmp_path)
    record = json.loads(authority_path.read_text(encoding="ascii"))
    checkpoint = Path(record["checkpoint"]["path"])
    checkpoint.chmod(0o644)
    torch.save(
        _TinyScriptedSlimBrain(
            record["source"]["revision"],
            record["logits_output"]["class_semantics_sha256"],
        ),
        checkpoint,
    )
    checkpoint.chmod(0o444)
    authority_sha256 = _rebind_slimbrain_checkpoint(authority_path)
    with pytest.raises(
        postseal_metrics.SlimBrainAuthorityError,
        match="not a runnable TorchScript module",
    ):
        postseal_metrics.load_authenticated_slimbrain(
            authority_path,
            expected_authority_sha256=authority_sha256,
            dependency_authority_sha256=dependency_authority_sha256,
            python_executable_sha256=python_executable_sha256,
            device=torch.device("cpu"),
        )


def _scheduler_line(
    *,
    node: str = "gpu-10",
    gpu_count: str = "1",
    excluded: str = "gpu-[05,50-51,56]",
    minimum_memory: str = "128G",
    per_node_key: str = "TresPerNode",
    per_node_value: str = "gres:gpu:1",
    tres_key: str = "TRES",
    tres_memory: str = "128G",
    gpu_tres_key: str = "gres/gpu",
) -> str:
    return (
        "JobId=123 JobState=RUNNING Partition=cscc-gpu-p QOS=cscc-gpu-qos "
        f"NodeList={node} ExcNodeList={excluded} NumNodes=1 NumCPUs=16 "
        f"NumTasks=1 CPUs/Task=16 MinMemoryNode={minimum_memory} "
        f"{per_node_key}={per_node_value} "
        f"{tres_key}=cpu=16,mem={tres_memory},node=1,billing=16,"
        f"{gpu_tres_key}={gpu_count}"
    )


def test_scheduler_evidence_requires_exact_one_gpu_tres_and_safe_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_output = _scheduler_line()

    def run(command, **_kwargs):
        if command[0] == "/usr/bin/scontrol":
            return SimpleNamespace(stdout=scheduler_output + "\n")
        return SimpleNamespace(stdout="GPU 0: NVIDIA A100-SXM4-40GB\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "gpu-10")
    environment = {
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "cscc-gpu-p",
        "SLURM_CPUS_PER_TASK": "16",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_GPUS_ON_NODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    evidence = cli._require_ciai_gpu_allocation()  # noqa: SLF001
    assert evidence["exact_one_gpu_tres_authenticated"] is True
    assert evidence["allocated_node_not_excluded"] is True
    assert evidence["scheduler_excluded_bad_nodes_authenticated"] is True
    assert evidence["requested_excluded_bad_nodes"] == [
        "gpu-05",
        "gpu-50",
        "gpu-51",
        "gpu-56",
    ]

    scheduler_output = _scheduler_line(gpu_count="2")
    with pytest.raises(cli.BootstrapError, match="exact CIAI"):
        cli._require_ciai_gpu_allocation()  # noqa: SLF001
    scheduler_output = _scheduler_line(node="gpu-51")
    with pytest.raises(cli.BootstrapError, match="excluded"):
        cli._require_ciai_gpu_allocation()  # noqa: SLF001
    scheduler_output = _scheduler_line(node="gpu-51.ciai.local")
    with pytest.raises(cli.BootstrapError, match="excluded"):
        cli._require_ciai_gpu_allocation()  # noqa: SLF001
    scheduler_output = _scheduler_line(excluded="gpu-[05,50-51]")
    with pytest.raises(cli.BootstrapError, match="excluded"):
        cli._require_ciai_gpu_allocation()  # noqa: SLF001


def test_scheduler_evidence_accepts_known_ciai_scontrol_spelling_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_output = _scheduler_line(
        minimum_memory="131072Mn",
        per_node_key="TRESPerNode",
        per_node_value="gres/gpu:a100=1",
        tres_key="AllocTRES",
        tres_memory="131072M",
        gpu_tres_key="gres/gpu:a100",
    )

    def run(command, **_kwargs):
        if command[0] == "/usr/bin/scontrol":
            return SimpleNamespace(stdout=scheduler_output + "\n")
        return SimpleNamespace(stdout="GPU 0: NVIDIA A100-SXM4-40GB\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.socket, "gethostname", lambda: "gpu-10")
    environment = {
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "cscc-gpu-p",
        "SLURM_CPUS_PER_TASK": "16",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_GPUS_ON_NODE": "gpu:a100:1",
        "CUDA_VISIBLE_DEVICES": "GPU-deadbeef",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    evidence = cli._require_ciai_gpu_allocation()  # noqa: SLF001
    assert evidence["exact_one_gpu_tres_authenticated"] is True
    assert evidence["scheduler_tres_per_node"] == "gres/gpu:a100=1"
