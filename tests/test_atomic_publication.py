from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import numpy as np
import nibabel as nib
import pytest
import torch

import eval.publication as publication_module
from eval.publication import publish_paired_prediction_bundle


def _inputs():
    prediction = torch.randn(1, 1, 4, 4, 4, 4)
    real = prediction + 0.01 * torch.randn_like(prediction)
    brain_mask = torch.ones(1, 1, 4, 4, 4)
    roi_masks = torch.zeros(1, 2, 4, 4, 4)
    roi_masks[:, 0, :2] = 1.0
    roi_masks[:, 1, 2:] = 1.0
    return prediction, real, brain_mask, roi_masks


def _texture_evaluator(*_args, **_kwargs):
    output_dir = Path(_args[3])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = output_dir / "texture_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "quality_gate": {
                    "release_gate_passed": True,
                    "exit_code": 0,
                    "verdict": "pass",
                }
            }
        )
    )
    return {"manifest": manifest}


def _publish(tmp_path, quality_evaluator, texture_evaluator=_texture_evaluator):
    prediction, real, brain_mask, roi_masks = _inputs()
    return publish_paired_prediction_bundle(
        tmp_path / "published",
        scan_id="sub-01_run-1",
        prediction=prediction,
        real=real,
        brain_mask=brain_mask,
        roi_masks=roi_masks,
        architecture_affine=np.diag([3.0, 3.0, 3.0, 1.0]),
        grid_contract=None,
        quality_evaluator=quality_evaluator,
        texture_evaluator=texture_evaluator,
    )


def _assert_no_partial_publication(tmp_path):
    assert not (tmp_path / "published").exists()
    assert not list(tmp_path.glob(".sub-01_run-1-gated-*"))


def test_paired_quality_failure_never_publishes_a_prediction(tmp_path):
    def failed_quality(*_args, **_kwargs):
        return {"passed": False, "verdict": "fail_quality_gate"}

    with pytest.raises(RuntimeError, match="paired 4D quality gate rejected"):
        _publish(tmp_path, failed_quality)
    _assert_no_partial_publication(tmp_path)


def test_texture_or_anti_gaming_failure_never_publishes_a_prediction(tmp_path):
    def passed_quality(*_args, **_kwargs):
        return {"passed": True, "verdict": "pass"}

    def failed_texture(*args, **_kwargs):
        outputs = _texture_evaluator(*args, **_kwargs)
        manifest = Path(outputs["manifest"])
        payload = json.loads(manifest.read_text())
        payload["quality_gate"].update(
            {
                "release_gate_passed": False,
                "exit_code": 2,
                "verdict": "fail_anti_gaming",
            }
        )
        manifest.write_text(json.dumps(payload))
        return outputs

    with pytest.raises(RuntimeError, match="texture/anti-gaming gate rejected"):
        _publish(tmp_path, passed_quality, failed_texture)
    _assert_no_partial_publication(tmp_path)


def test_prediction_bundle_appears_only_after_both_gates_pass(tmp_path):
    def passed_quality(*_args, **_kwargs):
        return {"passed": True, "verdict": "pass", "checks": {}}

    result = _publish(tmp_path, passed_quality)
    assert result == (tmp_path / "published").resolve()
    publication = json.loads((result / "publication.json").read_text())
    assert publication["paired_4d_gate_passed"] is True
    assert publication["texture_retention_gate_passed"] is True
    assert publication["texture_anti_gaming_gate_passed"] is True
    assert (result / "synthetic.nii.gz").is_file()
    assert (result / "real.nii.gz").is_file()
    assert (result / "paired_quality.json").is_file()
    assert not list(tmp_path.glob(".sub-01_run-1-gated-*"))


def test_destination_created_after_initial_check_is_never_replaced(
    tmp_path, monkeypatch
):
    original = publication_module._atomic_rename_noreplace

    def insert_competing_destination(
        parent_fd, parent_path, source_name, destination_name
    ):
        (parent_path / destination_name).mkdir()
        original(parent_fd, parent_path, source_name, destination_name)

    monkeypatch.setattr(
        publication_module,
        "_atomic_rename_noreplace",
        insert_competing_destination,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish(
            tmp_path,
            lambda *_args, **_kwargs: {
                "passed": True,
                "verdict": "pass",
            },
        )
    assert (tmp_path / "published").is_dir()
    assert not list((tmp_path / "published").iterdir())
    assert not list(tmp_path.glob(".sub-01_run-1-gated-*"))


def test_two_concurrent_publishers_commit_exactly_one_complete_bundle(tmp_path):
    barrier = threading.Barrier(2)

    def synchronized_texture(*args, **kwargs):
        outputs = _texture_evaluator(*args, **kwargs)
        barrier.wait(timeout=10)
        return outputs

    def publish_once():
        return _publish(
            tmp_path,
            lambda *_args, **_kwargs: {
                "passed": True,
                "verdict": "pass",
            },
            synchronized_texture,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish_once) for _index in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError as error:
                outcomes.append(error)

    successes = [value for value in outcomes if isinstance(value, Path)]
    failures = [value for value in outcomes if isinstance(value, Exception)]
    assert successes == [(tmp_path / "published").resolve()]
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert "refusing to overwrite" in str(failures[0])
    final = successes[0]
    assert (final / "publication.json").is_file()
    assert (final / "synthetic.nii.gz").is_file()
    assert (final / "real.nii.gz").is_file()
    assert (final / "paired_quality.json").is_file()
    assert (final / "texture" / "texture_manifest.json").is_file()
    assert not list(tmp_path.glob(".sub-01_run-1-gated-*"))


def test_publication_crops_patch_padding_and_uses_anatomical_affine(tmp_path):
    prediction = torch.randn(1, 1, 4, 6, 6, 6)
    real = prediction + 0.01 * torch.randn_like(prediction)
    mask = torch.ones(1, 1, 6, 6, 6)
    rois = torch.zeros(1, 2, 6, 6, 6)
    rois[:, 0, :3] = 1.0
    rois[:, 1, 3:] = 1.0
    anatomical_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    anatomical_affine[:3, 3] = (12.0, 15.0, 18.0)
    contract = {
        "architecture_shape": [6, 6, 6],
        "anatomical_shape": [4, 4, 4],
        "anatomical_affine": anatomical_affine.tolist(),
        "architecture_padding": {"before": [1, 1, 1], "after": [1, 1, 1]},
        "contract_sha256": "a" * 64,
    }

    result = publish_paired_prediction_bundle(
        tmp_path / "cropped",
        scan_id="sub-02_run-1",
        prediction=prediction,
        real=real,
        brain_mask=mask,
        roi_masks=rois,
        architecture_affine=np.eye(4),
        grid_contract=contract,
        quality_evaluator=lambda *_args, **_kwargs: {
            "passed": True,
            "verdict": "pass",
        },
        texture_evaluator=_texture_evaluator,
    )
    synthetic = nib.load(result / "synthetic.nii.gz")
    assert synthetic.shape == (4, 4, 4, 4)
    np.testing.assert_allclose(synthetic.affine, anatomical_affine)
    publication = json.loads((result / "publication.json").read_text())
    assert publication["architecture_padding_cropped"] is True
