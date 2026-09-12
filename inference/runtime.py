"""Unreleased V10 inference implementation kept behind the signed-launch stop.

    python -m inference.infer --config configs/connect4.yaml \
        --checkpoint checkpoints/connect4_epoch199.pt \
        --split test --out_dir outputs

For each subject it builds the frozen structural conditioning, generates the
full 4D volume, and seals the prediction before any paired evaluation is
allowed.  This entry point never opens a held-out fMRI target.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset_precomputed import (
    Connect4PrecomputedDataset,
    discover_structural_cache_scan_ids,
)
from data.collate import connect4_collate_fn
from data.protocol import (
    RUN_ARTIFACT_IDENTITY_SCHEMA,
    build_external_cache_protocol_context,
    build_run_artifact_identity,
    build_split_identity,
    load_completed_synthesis_checkpoint,
    patient_level_split_from_manifest,
    validate_fixed_protocol_config,
    validate_training_cohort_manifest,
    validate_external_scaler_provenance,
)
from data.provenance import canonical_sha256, sha256_file
from models.connect4 import Connect4Model, roi_masks_to_tensor
from eval.development_quality import validate_selectable_checkpoint_development_qa
from eval.visualize import save_fmri_nifti
from preprocessing.conform import anatomical_affine, crop_architecture_array
from utils.config import load_config, resolve_configured_value


POSTSEAL_EVALUATOR_GUIDANCE = (
    "target-blind predictions were sealed successfully; paired metrics and "
    "real-versus-synthetic visualization must now run through the separately "
    "staged, authenticated post-seal evaluator described in "
    "POSTSEAL_HELDOUT_EVALUATION.md. This synthesis process will not open "
    "held-out targets"
)


def _require_prediction_domain(prediction: torch.Tensor) -> torch.Tensor:
    """Reject non-finite or out-of-contract synthesis without clamping."""

    if not torch.is_tensor(prediction) or prediction.ndim != 6:
        raise ValueError("prediction must be [B,C,T,D,H,W]")
    if not torch.isfinite(prediction).all():
        raise RuntimeError(
            "target-blind synthesis produced NaN or infinity; refusing to seal"
        )
    if bool((prediction < 0).any()) or bool((prediction > 1).any()):
        raise RuntimeError(
            "target-blind synthesis left the authenticated [0,1] domain; "
            "refusing to clamp or seal it"
        )
    return prediction


def _inverse_pad_native_prediction(
    prediction: torch.Tensor,
    structural_binding: Mapping,
    architecture_affine: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    """Exactly crop signed Stage-B padding without interpolation or resampling."""

    if prediction.ndim != 6 or prediction.shape[0:2] != (1, 1):
        raise ValueError("native prediction must be exactly [1,1,T,D,H,W]")

    def triplet(name: str, *, positive: bool) -> tuple[int, int, int]:
        value = structural_binding.get(name)
        minimum = 1 if positive else 0
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < minimum
                for item in value
            )
        ):
            raise ValueError(f"native structural binding has invalid {name}")
        return tuple(int(item) for item in value)

    native_shape = triplet("native_shape", positive=True)
    padded_shape = triplet("architecture_shape", positive=True)
    before = triplet("padding_before", positive=False)
    after = triplet("padding_after", positive=False)
    if tuple(before[index] + native_shape[index] + after[index] for index in range(3)) != padded_shape:
        raise ValueError("native structural padding does not reconstruct architecture_shape")
    if tuple(prediction.shape[-3:]) != padded_shape:
        raise ValueError("prediction does not match the signed padded architecture grid")
    if structural_binding.get("padding_mode") != "constant-zero-no-interpolation":
        raise ValueError("native structural padding is not the exact no-interpolation mode")
    if structural_binding.get("interpolation_after_native_preprocessing") is not False:
        raise ValueError("native structural binding permits forbidden interpolation")

    native_affine = np.asarray(structural_binding.get("native_affine"), dtype=np.float64)
    padded_affine = np.asarray(structural_binding.get("padded_affine"), dtype=np.float64)
    supplied_affine = np.asarray(architecture_affine, dtype=np.float64)
    if native_affine.shape != (4, 4) or not np.isfinite(native_affine).all():
        raise ValueError("native structural binding has an invalid native affine")
    if padded_affine.shape != (4, 4) or not np.isfinite(padded_affine).all():
        raise ValueError("native structural binding has an invalid padded affine")
    if supplied_affine.shape != (4, 4) or not np.allclose(
        supplied_affine, padded_affine, rtol=0.0, atol=1e-6
    ):
        raise ValueError("inference affine differs from the signed padded affine")

    spatial_slices = tuple(
        slice(start, start + size) for start, size in zip(before, native_shape)
    )
    cropped = prediction[(slice(None), slice(None), slice(None), *spatial_slices)]
    if tuple(cropped.shape[-3:]) != native_shape:
        raise RuntimeError("inverse padding did not restore the signed native shape")
    return cropped, native_affine.copy()


def _generation_config(config: dict) -> dict:
    """Settings that must match the checkpoint for faithful DDIM synthesis."""
    data = config["data"]
    return {
        "data": {
            key: data.get(key)
            for key in (
                "protocol_profile",
                "preprocessing_evidence_status",
                "require_paper_preprocessing",
                "common_grid_contract_path",
                "common_grid_contract_sha256",
                "common_grid_contract_sha256_env",
                "native_alignment_authority_sha256",
                "native_alignment_authority_sha256_env",
                "native_alignment_authority_path",
                "structural_stage_root",
                "native_selection_manifest_path",
                "native_selection_manifest_sha256",
                "native_selection_manifest_sha256_env",
                "native_selection_root_review_path",
                "native_selection_root_review_sha256",
                "native_selection_root_review_sha256_env",
                "native_completed_set_path",
                "native_completed_set_sha256",
                "native_completed_set_sha256_env",
                "native_completed_set_commit_marker_path",
                "native_completed_set_commit_marker_sha256",
                "native_completed_set_commit_marker_sha256_env",
                "native_reviewed_source_path",
                "native_reviewed_source_sha256",
                "native_reviewed_source_sha256_env",
                "native_runtime_attester_sha256",
                "native_runtime_attester_sha256_env",
                "native_verifier_sha256",
                "native_verifier_sha256_env",
                "architecture_shape",
                "num_frames",
                "voxel_size_mm",
                "tr_seconds",
                "out_channels",
                "normalize_intensity",
            )
        },
        "models": config["models"],
        # Scan-bound DDIM noise hashes this seed, so changing it changes every
        # generated sample even when all model/data settings are identical.
        "sampling_seed": config["training"].get("seed", 42),
    }


def require_completed_training_checkpoint(checkpoint: dict) -> None:
    """Prevent test-set inspection before the configured optimization completes."""
    try:
        configured_epochs = checkpoint["config"]["training"]["epochs"]
    except Exception as exc:
        raise RuntimeError("checkpoint has no configured training horizon") from exc
    if (
        checkpoint.get("partial") is not False
        or checkpoint.get("next_batch_index") != 0
        or checkpoint.get("next_epoch") != configured_epochs
    ):
        raise RuntimeError(
            "inference requires a completed-run checkpoint; "
            "partial or intermediate checkpoints cannot inspect the held-out test set"
        )
    validate_selectable_checkpoint_development_qa(checkpoint)


def require_structural_artifact_compatibility(
    checkpoint_identity: dict, current_identity: dict
) -> None:
    """Compare only target-blind run identities before sealed prediction."""
    fields = (
        "num_scans",
        "scaler_identity_sha256",
        "conditioning_identity",
        "conditioning_identity_sha256",
        "common_grid_contract_sha256",
        "native_alignment_authority_sha256",
        "spatial_authority",
        "spatial_authority_sha256",
        "structural_artifact_identities_sha256",
    )
    if not isinstance(checkpoint_identity, dict) or not isinstance(
        current_identity, dict
    ):
        raise RuntimeError("run-artifact identity is missing")
    if (
        checkpoint_identity.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
        or current_identity.get("format") != RUN_ARTIFACT_IDENTITY_SCHEMA
    ):
        raise RuntimeError(
            "stale run-artifact identity is forbidden; "
            f"{RUN_ARTIFACT_IDENTITY_SCHEMA} is required"
        )
    for identity in (checkpoint_identity, current_identity):
        structural_digest = identity.get(
            "structural_artifact_identities_sha256"
        )
        if (
            not isinstance(structural_digest, str)
            or len(structural_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in structural_digest
            )
        ):
            raise RuntimeError(
                "run-artifact identity has no valid target-independent "
                "structural root"
            )
    differences = [
        field
        for field in fields
        if checkpoint_identity.get(field) != current_identity.get(field)
    ]
    if differences:
        raise RuntimeError(
            "checkpoint structural/conditioning artifacts differ from the "
            f"target-blind inference dataset: {differences}"
        )


def _target_blind_prediction_record(
    *,
    scan_id: str,
    prediction_path: Path,
    prediction_shape: tuple[int, ...],
    checkpoint_sha256: str,
    checkpoint_artifact_identity: dict,
    protocol_profile: str,
) -> dict:
    """Describe one sealed prediction without consulting a target artifact."""
    if (
        not scan_id
        or scan_id in {".", ".."}
        or Path(scan_id).name != scan_id
        or "/" in scan_id
        or "\\" in scan_id
    ):
        raise ValueError(f"unsafe inference scan ID {scan_id!r}")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction was not written: {prediction_path}")
    if len(prediction_shape) != 4 or prediction_shape[-1] != 128:
        raise ValueError(
            "sealed prediction must be a 4D NIfTI with exactly 128 frames"
        )
    record = {
        "format": "connect4-target-blind-prediction-v1",
        "status": "SEALED_TARGET_BLIND_PREDICTION",
        "scan_id": scan_id,
        "protocol_profile": protocol_profile,
        "target_opened_before_prediction": False,
        "paired_quality_gate_run": False,
        "prediction": {
            "relative_path": f"subjects/{scan_id}/prediction.nii.gz",
            "sha256": sha256_file(prediction_path),
            "size_bytes": prediction_path.stat().st_size,
            "shape": list(prediction_shape),
            "repetition_time_seconds": 3.0,
        },
        "synthesis_checkpoint_sha256": checkpoint_sha256,
        "training_run_artifact_identity_sha256": checkpoint_artifact_identity.get(
            "sha256"
        ),
        "training_target_artifact_identities_sha256": (
            checkpoint_artifact_identity.get("target_artifact_identities_sha256")
        ),
        "spatial_authority_sha256": checkpoint_artifact_identity.get(
            "spatial_authority_sha256"
        ),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _target_blind_prediction_set_record(
    *,
    split_name: str,
    subject_records: list[dict],
    expected_scan_ids: list[str],
    checkpoint_sha256: str,
    checkpoint_artifact_identity: dict,
) -> dict:
    """Build the root seal for a complete target-blind prediction set."""
    scan_ids = [record.get("scan_id") for record in subject_records]
    if (
        not isinstance(expected_scan_ids, list)
        or not expected_scan_ids
        or any(
            not isinstance(scan_id, str) or not scan_id
            for scan_id in expected_scan_ids
        )
        or len(expected_scan_ids) != len(set(expected_scan_ids))
    ):
        raise ValueError("prediction-set seal requires unique expected scan IDs")
    expected_sorted = sorted(expected_scan_ids)
    if (
        not subject_records
        or any(not isinstance(scan_id, str) or not scan_id for scan_id in scan_ids)
        or scan_ids != sorted(set(scan_ids))
        or scan_ids != expected_sorted
        or any(
            record.get("target_opened_before_prediction") is not False
            or record.get("status") != "SEALED_TARGET_BLIND_PREDICTION"
            or record.get("record_sha256")
            != canonical_sha256(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
            for record in subject_records
        )
    ):
        raise ValueError("prediction-set seal requires unique authenticated records")
    subject_bindings = [
        {
            "scan_id": record["scan_id"],
            "prediction_sha256": record["prediction"]["sha256"],
            "subject_record_sha256": record["record_sha256"],
        }
        for record in subject_records
    ]
    record = {
        "format": "connect4-target-blind-prediction-set-v1",
        "status": "SEALED_TARGET_BLIND_PREDICTION_SET",
        "split": split_name,
        "num_subjects": len(subject_bindings),
        "subjects": subject_bindings,
        "subjects_sha256": canonical_sha256(subject_bindings),
        "sealed_targets_opened": False,
        "paired_quality_gate_run": False,
        "synthesis_checkpoint_sha256": checkpoint_sha256,
        "training_run_artifact_identity_sha256": checkpoint_artifact_identity.get(
            "sha256"
        ),
        "training_target_artifact_identities_sha256": (
            checkpoint_artifact_identity.get("target_artifact_identities_sha256")
        ),
        "spatial_authority_sha256": checkpoint_artifact_identity.get(
            "spatial_authority_sha256"
        ),
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def validate_inference_mode(
    requested_split: str | None,
    *,
    metrics: bool,
    visualize: bool,
) -> str:
    """Resolve the mode before loading targets and reject impossible requests."""
    split_name = requested_split or ("test" if metrics or visualize else "all")
    if split_name not in {"all", "test", "external"}:
        raise ValueError(f"unsupported inference split {split_name!r}")
    if split_name == "external" and (metrics or visualize):
        raise ValueError(
            "--split external is structural-only; --metrics and real-vs-synthetic "
            "--visualize require paired fMRI targets"
        )
    if metrics and split_name != "test":
        raise ValueError("--metrics is valid only on the fixed disjoint test split")
    if visualize and split_name != "test":
        raise ValueError(
            "real-vs-synthetic visualization is valid only after sealing the "
            "fixed disjoint test predictions"
        )
    return split_name


def validate_subject_limit(split_name: str, subject_limit: int | None) -> None:
    """Forbid incomplete sets from receiving a target-blind set seal.

    ``--num`` is a turnaround aid only for explicitly unpaired external
    candidates. A non-external run publishes a prediction-set seal, so it must
    traverse the complete selected split or fail before dataset/model setup.
    """

    if subject_limit is None:
        return
    if (
        isinstance(subject_limit, bool)
        or not isinstance(subject_limit, int)
        or subject_limit < 1
    ):
        raise ValueError("--num must be a positive integer")
    if split_name != "external":
        raise ValueError(
            "--num is forbidden for sealed synthesis; the complete selected "
            "split is required"
        )


def fixed_inference_indices(
    scan_ids,
    config: dict,
    cohort_evidence,
    *,
    requested_split: str | None,
    metrics: bool,
    checkpoint_split_identity: dict | None,
):
    """Resolve all-vs-test synthesis and prove test identity when requested."""
    split_name = requested_split or ("test" if metrics else "all")
    if metrics and split_name != "test":
        raise ValueError("--metrics is valid only on the fixed disjoint test split")
    if split_name == "all":
        return list(range(len(scan_ids))), "all"
    if split_name != "test":
        raise ValueError(f"unsupported inference split {split_name!r}")

    manifest_path = config["training"].get("split_manifest")
    if not manifest_path or not Path(manifest_path).expanduser().is_file():
        raise FileNotFoundError(
            "fixed test inference requires the existing training.split_manifest; "
            "inference will not create a new assignment"
        )
    train_indices, validation_indices, test_indices = patient_level_split_from_manifest(
        scan_ids,
        cohort_evidence.patient_by_scan,
        val_frac=config["training"].get("val_frac", 0.15),
        test_frac=config["training"].get("test_frac", 0.15),
        seed=config["training"].get("seed", 42),
        manifest_path=manifest_path,
        manifest_sha256=resolve_configured_value(
            config["training"],
            "split_manifest_sha256",
            "split_manifest_sha256_env",
        ),
        protocol_profile=str(config["data"].get("protocol_profile", "")),
    )
    identity = build_split_identity(
        scan_ids,
        train_indices,
        validation_indices,
        test_indices,
        cohort_evidence.cohort_by_scan,
        cohort_evidence.patient_by_scan,
    )
    if checkpoint_split_identity != identity:
        raise RuntimeError(
            "checkpoint cohort/split identity does not match the requested fixed test set"
        )
    return test_indices, "test"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/connect4.yaml")
    ap.add_argument(
        "--checkpoint", required=True,
        help="trained CONNECT-4 checkpoint (required; random-weight synthesis is invalid)",
    )
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument(
        "--num",
        type=int,
        default=None,
        help=(
            "optional subject limit for explicitly unpaired external candidates; "
            "sealed synthesis always requires its complete selected split"
        ),
    )
    ap.add_argument(
        "--split",
        choices=("all", "test", "external"),
        default=None,
        help=(
            "dataset partition; external synthesizes a manifest-proven unseen "
            "structural-only cohort"
        ),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument(
        "--unpaired-candidate-only",
        action="store_true",
        help=(
            "permit structural-only external generation as an explicitly "
            "UNPUBLISHED candidate; paired quality publication remains forbidden"
        ),
    )
    args = ap.parse_args()
    selected_split = validate_inference_mode(
        args.split, metrics=args.metrics, visualize=args.visualize
    )
    validate_subject_limit(selected_split, args.num)
    external_mode = selected_split == "external"
    if external_mode and not args.unpaired_candidate_only:
        raise ValueError(
            "external structural-only generation has no paired quality reference; "
            "use --unpaired-candidate-only to create a clearly non-published candidate"
        )
    if not external_mode and args.unpaired_candidate_only:
        raise ValueError("--unpaired-candidate-only is valid only with --split external")

    cfg = load_config(args.config)
    validate_fixed_protocol_config(cfg)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BrainLM is a training-only perceptual reference and is not needed to
    # synthesize a volume. Only its state keys may be absent from this model.
    model = Connect4Model(cfg, build_loss=False).to(device).eval()
    ckpt, _, synthesis_checkpoint_sha256 = load_completed_synthesis_checkpoint(
        args.checkpoint
    )
    if _generation_config(ckpt["config"]) != _generation_config(cfg):
        raise RuntimeError(
            "checkpoint generation config differs from the requested inference config"
        )
    # This loader's "all" selection also contains the held-out test patients.
    # Any standalone inference on the synthesis cohort must therefore wait until
    # optimization is complete, even when metrics are not requested.
    require_completed_training_checkpoint(ckpt)
    state = ckpt["model"]
    incompat = model.load_state_dict(state, strict=False)
    unexpected = [
        key for key in incompat.unexpected_keys
        if not key.startswith("loss_fn.")
    ]
    if incompat.missing_keys or unexpected:
        raise RuntimeError(
            "checkpoint does not match the paper-faithful CONNECT-4 architecture; "
            f"missing={incompat.missing_keys}, unexpected={unexpected}"
        )
    print(f"[infer] loaded checkpoint {args.checkpoint}")

    d = cfg["data"]
    cohort_manifest = d.get("cohort_manifest")
    if not cohort_manifest:
        raise ValueError("data.cohort_manifest is required to prove cohort identity")
    external_protocol_context = None
    if external_mode:
        external_scan_ids = discover_structural_cache_scan_ids(
            d["root_dir"],
            d["precomputed_dir"],
            d.get("structural_stage_root"),
        )
        _, external_protocol_context = build_external_cache_protocol_context(
            manifest_path=cohort_manifest,
            scan_ids=external_scan_ids,
            synthesis_split_identity=ckpt["split_identity"],
            synthesis_artifact_identity=ckpt["artifact_identity"],
            synthesis_checkpoint_sha256=synthesis_checkpoint_sha256,
        )
        validate_external_scaler_provenance(
            d.get("scaler_dir"),
            external_scan_ids,
            ckpt["artifact_identity"],
            ckpt["split_identity"],
        )
    ds = Connect4PrecomputedDataset(
        root_dir=d["root_dir"], precomputed_dir=d["precomputed_dir"],
        target_shape=tuple(d["architecture_shape"]), num_frames=int(d["num_frames"]),
        normalize_intensity=d.get("normalize_intensity", True),
        require_paper_preprocessing=d.get("require_paper_preprocessing", True),
        dwi_matrix_path=d.get("dwi_matrix_path"),
        scaler_dir=d.get("scaler_dir"),
        normative_csv_path=d.get("normative_csv_path"),
        cohort_manifest_path=cohort_manifest,
        brainiac_model_path=cfg["models"].get("brainiac_path"),
        brainiac_checkpoint_sha256=resolve_configured_value(
            cfg["models"],
            "brainiac_checkpoint_sha256",
            "brainiac_checkpoint_sha256_env",
        ),
        brainiac_source_sha256=resolve_configured_value(
            cfg["models"], "brainiac_source_sha256", "brainiac_source_sha256_env"
        ),
        modernbert_model_name=cfg["models"].get("modernbert_name"),
        modernbert_revision=cfg["models"].get("modernbert_revision"),
        patch_size=tuple(cfg["models"]["graphs"]["patch_size"]),
        image_node_dim=int(cfg["models"]["fusion"]["image_embed_dim"]),
        mask_node_dim=int(cfg["models"]["fusion"]["mask_embed_dim"]),
        roi_node_dim=int(cfg["models"]["fusion"]["roi_embed_dim"]),
        load_fmri_targets=False,
        target_scan_ids=[],
        external_protocol_context=external_protocol_context,
        sealed_prediction_mode=not external_mode,
        common_grid_contract_path=d.get("common_grid_contract_path"),
        common_grid_contract_sha256=resolve_configured_value(
            d,
            "common_grid_contract_sha256",
            "common_grid_contract_sha256_env",
        ),
        structural_stage_root=d.get("structural_stage_root"),
        native_alignment_authority_path=d.get(
            "native_alignment_authority_path"
        ),
        native_alignment_authority_sha256=resolve_configured_value(
            d,
            "native_alignment_authority_sha256",
            "native_alignment_authority_sha256_env",
        ),
        allow_synthetic_grid_override=d.get("allow_synthetic_grid_override", False),
    )
    if ds.load_fmri_targets or ds.target_scan_ids:
        raise RuntimeError("inference dataset unexpectedly received target access")
    if external_mode:
        selected_indices = list(range(len(ds.scan_ids)))
    else:
        cohort_evidence = validate_training_cohort_manifest(
            cohort_manifest,
            ds.scan_ids,
            expected_scan_counts=d.get("expected_cohort_scan_counts"),
            protocol_profile=d.get("protocol_profile"),
        )
        structural_identity = build_run_artifact_identity(
            ds,
            cohort_manifest_path=cohort_manifest,
            evaluation_extractor_spec=None,
            perceptual_extractor_spec=cfg.get("training", {})
            .get("loss", {})
            .get("perceptual_extractor"),
            target_scan_ids=[],
        )
        require_structural_artifact_compatibility(
            ckpt["artifact_identity"], structural_identity
        )
        selected_indices, selected_split = fixed_inference_indices(
            ds.scan_ids,
            cfg,
            cohort_evidence,
            requested_split=selected_split,
            metrics=args.metrics,
            checkpoint_split_identity=ckpt["split_identity"],
        )
    if args.num is not None:
        selected_indices = selected_indices[: args.num]
    if not selected_indices:
        raise ValueError(f"inference selection {selected_split!r} contains no subjects")
    loader = DataLoader(
        Subset(ds, selected_indices),
        batch_size=1,
        shuffle=False,
        collate_fn=connect4_collate_fn,
    )
    print(
        f"[infer] selected {len(selected_indices)} subject(s) from {selected_split} split"
    )

    final_root = out_dir / (
        "unpaired_candidates" if external_mode else "sealed_predictions"
    )
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite inference output: {final_root}")
    staged_root = Path(
        tempfile.mkdtemp(prefix=".target-blind-predictions-", dir=out_dir)
    )
    try:
        prediction_records = []
        for i, batch in enumerate(loader):
            batch = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            if batch.get("fmri") is not None:
                raise RuntimeError(
                    "target-blind inference received an fMRI target before prediction"
                )
            roi_masks = roi_masks_to_tensor(
                batch.get("roi_masks"),
                model.target_shape,
                device,
                expected_num_rois=model.num_rois,
            )
            if roi_masks is None:
                raise ValueError(
                    "ROI masks are required for structural brain masking"
                )
            if batch.get("brain_mask") is None:
                raise ValueError(
                    "target-blind inference requires the explicit authenticated "
                    "structural brain_mask; ROI-union fallback is forbidden"
                )
            amp_on = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
            use_bf16 = amp_on and torch.cuda.is_bf16_supported()
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                sampled = model.sampled_ddim_decode(
                    batch, sampling_context="evaluation"
                )
            pred = _require_prediction_domain(sampled["prediction"])

            sid_value = batch.get("scan_id", [f"subj{i:03d}"])
            if isinstance(sid_value, (list, tuple)) and len(sid_value) == 1:
                sid = str(sid_value[0])
            elif isinstance(sid_value, str):
                sid = sid_value
            else:
                raise ValueError("batch must identify exactly one inference scan")
            if (
                not sid
                or sid in {".", ".."}
                or Path(sid).name != sid
                or "/" in sid
                or "\\" in sid
            ):
                raise ValueError(f"unsafe inference scan ID {sid!r}")
            affine_value = batch.get("t1w_affine")
            if affine_value is None:
                raise ValueError(
                    "the certified T1w affine is required for NIfTI output"
                )
            affine = affine_value[0].detach().cpu().numpy()
            if external_mode:
                if ds.common_grid_contract is None:
                    raise RuntimeError(
                        "external candidates require a checkpoint-compatible "
                        "common-grid authority"
                    )
                candidate_dir = staged_root / "subjects" / sid
                candidate_dir.mkdir(parents=True, exist_ok=False)
                candidate = crop_architecture_array(pred, ds.common_grid_contract)
                candidate_affine = anatomical_affine(ds.common_grid_contract)
                candidate_path = candidate_dir / "UNPUBLISHED_candidate.nii.gz"
                save_fmri_nifti(
                    candidate,
                    str(candidate_path),
                    affine=candidate_affine,
                    repetition_time=3.0,
                )
                (candidate_dir / "candidate.json").write_text(
                    json.dumps(
                        {
                            "status": "UNPUBLISHED_UNPAIRED_CANDIDATE",
                            "paired_quality_gate_run": False,
                            "publication_permitted": False,
                            "reason": "no real fMRI target was available",
                            "common_grid_contract_sha256": ds.common_grid_contract[
                                "contract_sha256"
                            ],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[infer] {sid}: wrote explicitly unpublished candidate")
                continue

            subject_dir = staged_root / "subjects" / sid
            subject_dir.mkdir(parents=True, exist_ok=False)
            if ds.common_grid_contract is None:
                if not ds.recovery_profile:
                    raise RuntimeError(
                        "prediction publication has neither common-grid nor "
                        "authenticated native geometry"
                    )
                source_identity = ds.source_dataset.source_fingerprint(sid).get(
                    "structural_source_identity"
                )
                if not isinstance(source_identity, Mapping):
                    raise RuntimeError(
                        "native prediction has no authenticated structural binding"
                    )
                prediction, output_affine = _inverse_pad_native_prediction(
                    pred,
                    source_identity,
                    affine,
                )
            else:
                prediction = crop_architecture_array(
                    pred, ds.common_grid_contract
                )
                output_affine = anatomical_affine(ds.common_grid_contract)
            _require_prediction_domain(prediction)
            prediction_path = subject_dir / "prediction.nii.gz"
            save_fmri_nifti(
                prediction,
                str(prediction_path),
                affine=output_affine,
                repetition_time=3.0,
            )
            prediction_image = nib.load(str(prediction_path))
            record = _target_blind_prediction_record(
                scan_id=sid,
                prediction_path=prediction_path,
                prediction_shape=tuple(int(value) for value in prediction_image.shape),
                checkpoint_sha256=synthesis_checkpoint_sha256,
                checkpoint_artifact_identity=ckpt["artifact_identity"],
                protocol_profile=str(d.get("protocol_profile", "")),
            )
            (subject_dir / "prediction.json").write_text(
                json.dumps(record, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            prediction_records.append(record)
            print(f"[infer] {sid}: staged target-blind prediction")

        if not external_mode:
            prediction_records.sort(key=lambda record: record["scan_id"])
            set_record = _target_blind_prediction_set_record(
                split_name=selected_split,
                subject_records=prediction_records,
                expected_scan_ids=[
                    ds.scan_ids[index] for index in selected_indices
                ],
                checkpoint_sha256=synthesis_checkpoint_sha256,
                checkpoint_artifact_identity=ckpt["artifact_identity"],
            )
            (staged_root / "prediction_set_seal.json").write_text(
                json.dumps(set_record, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        os.replace(staged_root, final_root)
        print(f"[infer] atomic output -> {final_root}")
    except Exception:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise

    if args.metrics or args.visualize:
        raise RuntimeError(POSTSEAL_EVALUATOR_GUIDANCE)
