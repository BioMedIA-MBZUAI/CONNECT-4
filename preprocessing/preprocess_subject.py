"""
One-shot per-subject preprocessing driver (CONNECT-4).

Runs the full chain for a subject:
    SynthSeg   -> authenticate and conform label map to the grid (preprocess_seg)
    T1w        -> normalize inside that authenticated support    (preprocess_t1)
    rs-fMRI    -> verified fMRIPrep T1w-space derivative, then smoothing,
                  temporal filtering and CONNECT-4 harmonisation

Segmentation must be produced **externally** with SynthSeg
(https://github.com/BBillot/SynthSeg) and passed via `--seg`.

    python -m preprocessing.preprocess_subject \
        --t1 sub-01_T1w.nii.gz --seg sub-01_synthseg.nii.gz \
        --fmri sub-01_space-T1w_desc-preproc_bold.nii.gz \
        --fmriprep-t1 sub-01_desc-preproc_T1w.nii.gz \
        --fmriprep-dataset-description /derivatives/dataset_description.json \
        --fmriprep-execution-record /derivatives/logs/connect4-fmriprep_sub-01_task-rest_execution.json \
        --structural-source-authority /authority/sub-01_structural_source.json \
        --structural-source-authority-sha256 STRUCTURAL_ONLY_SHA256 \
        --source-acquisition-identity /authority/sub-01_source_acquisition.json \
        --source-acquisition-identity-sha256 EXTERNALLY_PINNED_SHA256 \
        --motion-confounds sub-01_desc-confounds_timeseries.tsv \
        --coregistration-transform sub-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt \
        --out_dir derivatives/sub-01

Outputs in `--out_dir`: T1w_common.nii.gz, seg_common.nii.gz, and
bold_common.nii.gz. The grid is supplied by a T1-only cohort contract; 128
refers only to real fMRI frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .preprocess_t1 import preprocess_t1
from .preprocess_seg import preprocess_seg
from .preprocess_fmri import preprocess_fmri
from .source_acquisition_identity import (
    SourceAcquisitionError,
    load_source_acquisition_identity,
    load_structural_source_authority,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument(
        "--seg", required=True, help="external SynthSeg label map for this subject"
    )
    ap.add_argument(
        "--fmri",
        required=True,
        help="fMRIPrep space-T1w desc-preproc_bold derivative",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--fmri-metadata",
        default=None,
        help="BIDS JSON sidecar (auto-discovered if omitted)",
    )
    ap.add_argument(
        "--fmriprep-t1",
        default=None,
        help=(
            "same-subject fMRIPrep desc-preproc_T1w target named by the BOLD-to-T1w "
            "transform (required for production; distinct from the source --t1)"
        ),
    )
    ap.add_argument(
        "--fmriprep-dataset-description",
        default=None,
        help="fMRIPrep derivative-root dataset_description.json (required for production)",
    )
    ap.add_argument(
        "--fmriprep-execution-record",
        default=None,
        help="successful preprocessing.run_fmriprep execution record (required for production)",
    )
    ap.add_argument(
        "--structural-source-authority",
        required=True,
        help="externally SHA-pinned target-blind raw T1w/SynthSeg authority JSON",
    )
    ap.add_argument(
        "--structural-source-authority-sha256",
        required=True,
        help="expected SHA-256 of --structural-source-authority",
    )
    ap.add_argument(
        "--source-acquisition-identity",
        default=None,
        help=(
            "externally SHA-pinned raw T1w/BOLD/SynthSeg identity JSON "
            "(required for production)"
        ),
    )
    ap.add_argument(
        "--source-acquisition-identity-sha256",
        default=None,
        help="expected SHA-256 of --source-acquisition-identity",
    )
    ap.add_argument(
        "--motion-confounds",
        default=None,
        help="matching fMRIPrep desc-confounds_timeseries.tsv (required for production)",
    )
    ap.add_argument(
        "--coregistration-transform",
        default=None,
        help=(
            "matching fMRIPrep from-boldref_to-T1w (or legacy from-scanner_to-T1w) "
            "image transform (required for production)"
        ),
    )
    ap.add_argument(
        "--preprocessing-backend",
        choices=("fmriprep", "custom-debug"),
        default="fmriprep",
        help="custom-debug outputs cannot be used for paper-faithful training",
    )
    ap.add_argument(
        "--coregistration-backend",
        choices=("auto", "flirt", "python", "header"),
        default="auto",
    )
    ap.add_argument(
        "--motion-backend", choices=("auto", "mcflirt", "python"), default="auto"
    )
    ap.add_argument(
        "--mcflirt", action="store_true", help="backward-compatible MCFLIRT shortcut"
    )
    ap.add_argument("--smoothing-fwhm-mm", type=float, default=3.0)
    ap.add_argument("--common-grid-contract", required=True)
    ap.add_argument("--common-grid-contract-sha256", required=True)
    ap.add_argument("--high-pass-hz", type=float, default=0.01)
    ap.add_argument("--low-pass-hz", type=float, default=0.10)
    a = ap.parse_args()

    if a.preprocessing_backend == "fmriprep" and not a.fmriprep_t1:
        ap.error("--fmriprep-t1 is required with --preprocessing-backend fmriprep")
    if a.preprocessing_backend == "fmriprep" and (
        not a.source_acquisition_identity or not a.source_acquisition_identity_sha256
    ):
        ap.error(
            "--source-acquisition-identity and "
            "--source-acquisition-identity-sha256 are required with "
            "--preprocessing-backend fmriprep"
        )
    if a.preprocessing_backend == "fmriprep":
        try:
            load_structural_source_authority(
                Path(a.structural_source_authority),
                expected_sha256=a.structural_source_authority_sha256,
                expected_raw_t1=Path(a.t1),
                expected_synthseg_mask=Path(a.seg),
            )
            load_source_acquisition_identity(
                Path(a.source_acquisition_identity),
                expected_sha256=a.source_acquisition_identity_sha256,
                expected_raw_t1=Path(a.t1),
                expected_synthseg_mask=Path(a.seg),
            )
        except SourceAcquisitionError as exc:
            ap.error(f"source-acquisition identity admission failed: {exc}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_common = preprocess_seg(
        a.seg,
        str(out / "seg_common.nii.gz"),
        common_grid_contract_path=a.common_grid_contract,
        common_grid_contract_sha256=a.common_grid_contract_sha256,
        structural_source_authority_path=a.structural_source_authority,
        structural_source_authority_sha256=(
            a.structural_source_authority_sha256
        ),
    )
    t1_common = preprocess_t1(
        a.t1,
        str(out / "T1w_common.nii.gz"),
        common_grid_contract_path=a.common_grid_contract,
        common_grid_contract_sha256=a.common_grid_contract_sha256,
        segmentation_path=seg_common,
        structural_source_authority_path=a.structural_source_authority,
        structural_source_authority_sha256=(
            a.structural_source_authority_sha256
        ),
    )
    preprocess_fmri(
        a.fmri,
        a.fmriprep_t1 if a.preprocessing_backend == "fmriprep" else t1_common,
        str(out / "bold_common.nii.gz"),
        mcflirt=a.mcflirt,
        metadata_path=a.fmri_metadata,
        fmriprep_dataset_description_path=a.fmriprep_dataset_description,
        fmriprep_execution_record_path=a.fmriprep_execution_record,
        source_acquisition_identity_path=a.source_acquisition_identity,
        source_acquisition_identity_sha256=a.source_acquisition_identity_sha256,
        motion_confounds_path=a.motion_confounds,
        coregistration_transform_path=a.coregistration_transform,
        brain_mask_path=seg_common,
        common_grid_contract_path=a.common_grid_contract,
        common_grid_contract_sha256=a.common_grid_contract_sha256,
        preprocessing_backend=a.preprocessing_backend,
        coregistration_backend=a.coregistration_backend,
        motion_backend=a.motion_backend,
        smoothing_fwhm_mm=a.smoothing_fwhm_mm,
        high_pass_hz=a.high_pass_hz,
        low_pass_hz=a.low_pass_hz,
    )
    print(f"[done] subject preprocessed -> {out}")


if __name__ == "__main__":
    main()
