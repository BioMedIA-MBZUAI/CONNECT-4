import json
from copy import deepcopy

import nibabel as nib
import numpy as np
import pytest
import torch

from data.dataset import Connect4Dataset
from data.provenance import canonical_sha256, sha256_file
from preprocessing.extract_roi_features import (
    CAT12_AUTHORITY_SCHEMA,
    PYRADIOMICS_INSTALLED_FILES_SCHEMA,
    PYRADIOMICS_RUNTIME_LOCK_SCHEMA,
    _parse_pyradiomics_runtime_lock,
    _snapshot_frozen_runtime_tree,
    _verify_pyradiomics_installed_files,
    anatcl_roi_volume_input,
    extract_anatcl_embedding,
    load_cat12_authority_manifest,
    load_cat12_scan_inputs,
    state_dict_sha256,
)


class _Deterministic3DAnatCL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, images):
        pooled = images.mean(dim=(-3, -2, -1)) * self.scale
        return pooled.repeat(1, 512)


def _signed(payload):
    result = deepcopy(payload)
    result["record_sha256"] = canonical_sha256(result)
    return result


def test_anatcl_uses_exact_3d_cat12_crop_pad_and_roi_mask():
    vbm = np.ones((121, 145, 121), dtype=np.float32)
    labels = np.zeros(vbm.shape, dtype=np.int16)
    labels[10:12, 20:22, 30:32] = 10
    tensor = anatcl_roi_volume_input(vbm, labels, 10)

    assert tensor.shape == (1, 1, 128, 128, 128)
    expected = torch.zeros_like(tensor)
    expected[0, 0, 13:15, 12:14, 33:35] = 1.0
    assert torch.equal(tensor, expected)


def test_anatcl_rejects_raw_or_native_t1_shape_and_absent_roi():
    with pytest.raises(ValueError, match="raw/native T1w"):
        anatcl_roi_volume_input(
            np.zeros((61, 73, 61), np.float32),
            np.zeros((61, 73, 61), np.int16),
            10,
        )
    with pytest.raises(ValueError, match="no voxels"):
        anatcl_roi_volume_input(
            np.zeros((121, 145, 121), np.float32),
            np.zeros((121, 145, 121), np.int16),
            53,
        )


def test_anatcl_embedding_is_one_finite_512d_descriptor():
    model = _Deterministic3DAnatCL()
    vbm = np.ones((121, 145, 121), dtype=np.float32)
    labels = np.zeros(vbm.shape, dtype=np.int16)
    labels[3:8, 20:25, 3:8] = 17
    embedding = extract_anatcl_embedding(model, vbm, labels, 17, torch.device("cpu"))
    assert embedding.shape == (512,)
    assert torch.isfinite(embedding).all()
    assert state_dict_sha256(model) == state_dict_sha256(model)


def test_cat12_authority_rehashes_exact_3d_inputs(tmp_path):
    scan_id = "train-scan-001"
    scan_root = tmp_path / scan_id
    scan_root.mkdir()
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    vbm = np.ones((121, 145, 121), dtype=np.float32)
    labels = np.zeros(vbm.shape, dtype=np.int16)
    labels[10:12, 20:22, 30:32] = 10
    vbm_path = scan_root / f"{scan_id}_mwp1_cat12_vbm.nii.gz"
    labels_path = scan_root / f"{scan_id}_synthseg_in_cat12_vbm.nii.gz"
    nib.save(nib.Nifti1Image(vbm, affine), vbm_path)
    nib.save(nib.Nifti1Image(labels, affine), labels_path)
    vbm_path.chmod(0o444)
    labels_path.chmod(0o444)

    record = _signed(
        {
            "schema": Connect4Dataset.CAT12_INPUT_PROVENANCE_SCHEMA,
            "scan_id": scan_id,
            "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
            "source_sha256": {"t1w": "1" * 64, "segmentation": "2" * 64},
            "outputs": {
                "cat12_mwp1": {
                    "relative_path": f"{scan_id}/{vbm_path.name}",
                    "sha256": sha256_file(vbm_path),
                    "size": vbm_path.stat().st_size,
                    "shape": [121, 145, 121],
                    "voxel_size_mm": [1.5, 1.5, 1.5],
                },
                "segmentation_in_cat12_vbm": {
                    "relative_path": f"{scan_id}/{labels_path.name}",
                    "sha256": sha256_file(labels_path),
                    "size": labels_path.stat().st_size,
                    "shape": [121, 145, 121],
                    "voxel_size_mm": [1.5, 1.5, 1.5],
                },
            },
        }
    )
    record_path = scan_root / "provenance.json"
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    record_path.chmod(0o444)
    manifest = _signed(
        {
            "schema": CAT12_AUTHORITY_SCHEMA,
            "purpose": "target-free-train-anatcl-vbm-inputs",
            "pipeline": dict(Connect4Dataset.CAT12_PREPROCESSING_CONTRACT),
            "scans": {
                scan_id: {
                    "record_relative_path": f"{scan_id}/provenance.json",
                    "record_file_sha256": sha256_file(record_path),
                    "record_sha256": record["record_sha256"],
                }
            },
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    loaded_manifest, manifest_sha = load_cat12_authority_manifest(
        manifest_path,
        expected_sha256=sha256_file(manifest_path),
        expected_scan_ids=[scan_id],
    )
    loaded_vbm, loaded_labels, identity = load_cat12_scan_inputs(
        tmp_path,
        loaded_manifest,
        manifest_sha,
        scan_id=scan_id,
        source_t1_sha256="1" * 64,
        source_segmentation_sha256="2" * 64,
    )
    assert loaded_vbm.shape == (121, 145, 121)
    assert loaded_labels.dtype == np.int32
    assert identity["scan_record_sha256"] == record["record_sha256"]


def test_cat12_authority_rejects_old_or_tampered_pipeline(tmp_path):
    manifest = _signed(
        {
            "schema": CAT12_AUTHORITY_SCHEMA,
            "purpose": "target-free-train-anatcl-vbm-inputs",
            "pipeline": {
                **Connect4Dataset.CAT12_PREPROCESSING_CONTRACT,
                "output_kind": "raw T1w masquerading as VBM",
            },
            "scans": {},
        }
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="pipeline differs"):
        load_cat12_authority_manifest(
            path,
            expected_sha256=sha256_file(path),
            expected_scan_ids=[],
        )


def test_pyradiomics_lock_is_fail_closed_until_official_linux_runtime_exists():
    payload = _signed(
        {
            "schema": PYRADIOMICS_RUNTIME_LOCK_SCHEMA,
            "production_ready": False,
            "distribution": "pyradiomics",
            "package_version": "3.0.1",
            "upstream_revision": "08bea7067e350303eead533b471aa60697b3b8c3",
            "source_distribution": {
                "filename": "pyradiomics-3.0.1.tar.gz",
                "sha256": (
                    "47c57f441d6cb7973fa3b2ea48d3948df78e3348e1c69e1e2ff19001601fc2f5"
                ),
                "size": 34473030,
            },
            "runtime": {
                "python_abi": "cp310",
                "platform": "linux_x86_64",
                "runtime_root_relative_path": ("pyradiomics_official_3_0_1_runtime"),
                "site_packages_relative_path": "lib/python3.10/site-packages",
                "installed_files_manifest_relative_path": (
                    "pyradiomics_official_3_0_1.installed-files.json"
                ),
                "runtime_tree_sha256": None,
                "installed_files_manifest_sha256": None,
            },
            "rejected_distributions": ["pyradiomics-cuda"],
        }
    )
    with pytest.raises(RuntimeError, match="not frozen"):
        _parse_pyradiomics_runtime_lock(payload, lock_sha256="f" * 64)


def test_pyradiomics_lock_rejects_cuda_fork_identity():
    payload = _signed(
        {
            "schema": PYRADIOMICS_RUNTIME_LOCK_SCHEMA,
            "production_ready": True,
            "distribution": "pyradiomics-cuda",
            "package_version": "1.0.4",
            "upstream_revision": "08bea7067e350303eead533b471aa60697b3b8c3",
            "source_distribution": {
                "filename": "pyradiomics-3.0.1.tar.gz",
                "sha256": (
                    "47c57f441d6cb7973fa3b2ea48d3948df78e3348e1c69e1e2ff19001601fc2f5"
                ),
                "size": 34473030,
            },
            "runtime": {
                "python_abi": "cp310",
                "platform": "linux_x86_64",
                "runtime_root_relative_path": ("pyradiomics_official_3_0_1_runtime"),
                "site_packages_relative_path": "lib/python3.10/site-packages",
                "installed_files_manifest_relative_path": (
                    "pyradiomics_official_3_0_1.installed-files.json"
                ),
                "runtime_tree_sha256": "a" * 64,
                "installed_files_manifest_sha256": "b" * 64,
            },
            "rejected_distributions": ["pyradiomics-cuda"],
        }
    )
    with pytest.raises(RuntimeError, match="source identity differs"):
        _parse_pyradiomics_runtime_lock(payload, lock_sha256="f" * 64)


def test_pyradiomics_installed_tree_rejects_mutated_source_with_same_metadata(
    tmp_path,
    monkeypatch,
):
    runtime_root = tmp_path / "pyradiomics_official_3_0_1_runtime"
    site_packages = runtime_root / "lib" / "python3.10" / "site-packages"
    package = site_packages / "radiomics"
    metadata = site_packages / "pyradiomics-3.0.1.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    source = package / "__init__.py"
    source.write_text('__version__ = "3.0.1"\n')
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: pyradiomics\nVersion: 3.0.1\n"
    )
    for path in (source, metadata / "METADATA"):
        path.chmod(0o444)
    for directory in sorted(
        (path for path in runtime_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    runtime_root.chmod(0o555)
    inventory = _snapshot_frozen_runtime_tree(runtime_root)
    runtime_tree_sha256 = canonical_sha256(inventory)
    manifest = _signed(
        {
            "schema": PYRADIOMICS_INSTALLED_FILES_SCHEMA,
            "distribution": "pyradiomics",
            "package_version": "3.0.1",
            **inventory,
            "runtime_tree_sha256": runtime_tree_sha256,
        }
    )
    manifest_path = tmp_path / "pyradiomics.installed-files.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    runtime = {
        "python_abi": "cp310",
        "platform": "linux_x86_64",
        "runtime_root_relative_path": runtime_root.name,
        "site_packages_relative_path": "lib/python3.10/site-packages",
        "installed_files_manifest_relative_path": manifest_path.name,
        "runtime_tree_sha256": runtime_tree_sha256,
        "installed_files_manifest_sha256": sha256_file(manifest_path),
    }
    monkeypatch.setattr(
        "preprocessing.extract_roi_features.sys.prefix", str(runtime_root)
    )
    resolved_root, resolved_site, identity = _verify_pyradiomics_installed_files(
        tmp_path / "runtime.lock.json", runtime
    )
    assert resolved_root == runtime_root
    assert resolved_site == site_packages
    assert identity == {
        "manifest_sha256": sha256_file(manifest_path),
        "runtime_tree_sha256": runtime_tree_sha256,
    }

    source.chmod(0o644)
    source.write_text('__version__ = "3.0.1"\nBACKDOOR = True\n')
    source.chmod(0o444)
    with pytest.raises(RuntimeError, match="bytes or inventory differ"):
        _verify_pyradiomics_installed_files(tmp_path / "runtime.lock.json", runtime)
