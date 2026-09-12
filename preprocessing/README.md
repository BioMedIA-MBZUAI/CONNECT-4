# CONNECT-4 preprocessing

Brings every subject onto an evidence-derived CONNECT-4 production grid:

| property        | value                          | status |
|-----------------|--------------------------------|--------|
| matrix size     | **derived from T1-only cohort evidence** | not stated in the manuscript |
| voxel size      | **3 mm isotropic**             | manuscript |
| rs-fMRI frames  | **128**                        | manuscript |
| rs-fMRI TR      | **3 s**                        | manuscript |

Both the T1w and rs-fMRI are conformed to one externally SHA-pinned, T1-only
cohort-common 3-mm grid. The builder derives the anatomical field of view from
a structural reference. It pads only to the configured patch multiple, records
that padding as non-anatomical, and publication crops it back off. No code path
turns the paper's 128 temporal frames into a 128³ spatial matrix.
The production path first uses the paper-cited **fMRIPrep** pipeline for
BOLD/T1w co-registration,
slice-timing correction, and head-motion correction. CONNECT-4 then applies
the stated spatial smoothing and temporal filtering before harmonising to
exactly 128 frames sampled at TR = 3 s.

Before subject processing, create the structural manifest and grid contract.
The manifest schema is `connect4-structural-cohort-v2`, sets
`functional_data_used` to `false`, and contains a `structural_reference` object
with path, SHA-256, `modality: T1w`, `cohort_derived: true`,
`functional_data_used: false`, and a non-empty `derivation_method`. It also
contains one row per scan with
`scan_id`, `t1w_path`, and `t1w_sha256`. Functional/BOLD fields are rejected.

```bash
python -m preprocessing.build_common_grid_contract \
    --structural-reference /path/to/T1-only-cohort-reference.nii.gz \
    --structural-manifest /path/to/structural_manifest.json \
    --patch-multiple 16 16 16 \
    --out /path/to/connect4_structural_common_grid.json
sha256sum /path/to/connect4_structural_common_grid.json
```

## Segmentation — SynthSeg (run externally)

Anatomical labels are produced with **SynthSeg** (Billot et al.), a contrast- and
resolution-agnostic CNN segmenter: <https://github.com/BBillot/SynthSeg>.

> **SynthSeg is run by you, externally** (per its own repo/instructions). This
> repository does **not** call SynthSeg — it only **consumes** the label map you
> produce. `preprocess_seg.py` simply conforms your SynthSeg output onto the
> CONNECT-4 grid (nearest-neighbour).

Structural preprocessing requires an independently signed and externally
SHA-pinned `connect4-structural-source-authority-v2` record per scan. Its exact
payload contains only `scan_id`, the raw T1 artifact, and the SynthSeg
mask/provenance/producer; BOLD and the digest of any BOLD-bearing authority are
forbidden. The fMRI path separately uses
`connect4-source-acquisition-identity-v2`, which binds the raw T1w and BOLD
plus a complete `connect4-bids-input-inventory-v1` snapshot of the dataset-root
files and exact participant tree.
The nested signed
`connect4-synthseg-parcellation-provenance-v1` record names the SynthSeg
version, source revision, model and container digests and must contain
`source_t1_sha256` equal to the raw T1w digest. The identity file's own SHA-256
is an external input to the corresponding command; a self-reported hash is
never an authority.

The labels drive the mask patches / ROI statistics (Fig 1A), the ROI graph and
ROI-coverage hyperedges (Fig 1B), and the measured ROI volumes used by the
normative / atrophy module (`normative/`).

## Pipeline

```bash
# 0) FIRST run SynthSeg yourself on the T1w (see the SynthSeg repo), e.g.:
#    mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz --robust

# 1) run the cited pipeline through the evidence-producing wrapper.
#    The wrapper fixes --output-spaces T1w and rejects --ignore slicetiming.
python -m preprocessing.run_fmriprep \
    --bids-root /bids \
    --derivatives-root /derivatives/fmriprep/sub-01_task-rest_generation-001 \
    --participant-label 01 \
    --scan-id sub-01_task-rest \
    --source-acquisition-identity /authority/sub-01_source_acquisition.json \
    --source-acquisition-identity-sha256 "$CONNECT4_SOURCE_ACQUISITION_SHA256" \
    --fmriprep-runtime-identity /authority/fmriprep_runtime_identity.json \
    --fmriprep-runtime-identity-sha256 "$CONNECT4_FMRIPREP_RUNTIME_SHA256" \
    --fmriprep-executable /runtime/bin/fmriprep

# 2) whole CONNECT-4 subject (T1 + SynthSeg + fMRIPrep derivative)
python -m preprocessing.preprocess_subject \
    --t1 /bids/sub-01/anat/sub-01_T1w.nii.gz \
    --seg sub-01_synthseg.nii.gz \
    --fmri /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz \
    --fmriprep-t1 /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz \
    --fmriprep-dataset-description /derivatives/fmriprep/sub-01_task-rest_generation-001/dataset_description.json \
    --fmriprep-execution-record /derivatives/fmriprep/sub-01_task-rest_generation-001/logs/connect4-fmriprep_sub-01_task-rest_execution.json \
    --structural-source-authority /authority/sub-01_structural_source.json \
    --structural-source-authority-sha256 "$CONNECT4_STRUCTURAL_SOURCE_SHA256" \
    --source-acquisition-identity /authority/sub-01_source_acquisition.json \
    --source-acquisition-identity-sha256 "$CONNECT4_SOURCE_ACQUISITION_SHA256" \
    --motion-confounds /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_desc-confounds_timeseries.tsv \
    --coregistration-transform /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt \
    --common-grid-contract /path/to/connect4_structural_common_grid.json \
    --common-grid-contract-sha256 "$CONNECT4_COMMON_GRID_SHA256" \
    --out_dir derivatives/connect4/sub-01
```

Or step by step (the authenticated segmentation must be produced first because
it defines the T1 normalization support):

```bash
python -m preprocessing.preprocess_seg \
    --in sub-01_synthseg.nii.gz --out seg_common.nii.gz \
    --common-grid-contract /path/to/connect4_structural_common_grid.json \
    --common-grid-contract-sha256 "$CONNECT4_COMMON_GRID_SHA256" \
    --structural-source-authority /authority/sub-01_structural_source.json \
    --structural-source-authority-sha256 "$CONNECT4_STRUCTURAL_SOURCE_SHA256"
python -m preprocessing.preprocess_t1 \
    --in sub-01_T1w.nii.gz --out T1w_common.nii.gz \
    --segmentation seg_common.nii.gz \
    --common-grid-contract /path/to/connect4_structural_common_grid.json \
    --common-grid-contract-sha256 "$CONNECT4_COMMON_GRID_SHA256" \
    --structural-source-authority /authority/sub-01_structural_source.json \
    --structural-source-authority-sha256 "$CONNECT4_STRUCTURAL_SOURCE_SHA256"
python -m preprocessing.preprocess_fmri \
    --fmri /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz \
    --metadata /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_space-T1w_desc-preproc_bold.json \
    --t1 /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz \
    --fmriprep-dataset-description /derivatives/fmriprep/sub-01_task-rest_generation-001/dataset_description.json \
    --fmriprep-execution-record /derivatives/fmriprep/sub-01_task-rest_generation-001/logs/connect4-fmriprep_sub-01_task-rest_execution.json \
    --source-acquisition-identity /authority/sub-01_source_acquisition.json \
    --source-acquisition-identity-sha256 "$CONNECT4_SOURCE_ACQUISITION_SHA256" \
    --motion-confounds /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_desc-confounds_timeseries.tsv \
    --coregistration-transform /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt \
    --brain-mask seg_common.nii.gz \
    --common-grid-contract /path/to/connect4_structural_common_grid.json \
    --common-grid-contract-sha256 "$CONNECT4_COMMON_GRID_SHA256" \
    --out bold_common.nii.gz
```

Production validation does not trust a per-file `GeneratedBy` claim. It reads
`GeneratedBy` from the standard fMRIPrep derivative-root
`dataset_description.json`, then checks it against the wrapper's successful
execution record. The record proves that the participant-level command fixed
`--output-spaces T1w`, did not use `--ignore slicetiming`, and completed with
return code zero. It contains a SHA-256 inventory of every matching derivative
candidate found for that participant, plus the dataset description and
stdout/stderr logs. The postprocessing command explicitly selects one BOLD run,
its JSON, a same-subject fMRIPrep `desc-preproc_T1w`, that run's explicit
`from-boldref_to-T1w_mode-image_desc-coreg_xfm` transform, and motion-confounds
TSV. The validator requires every selected artifact to appear in the inventory,
requires their run/subject identities to agree, and requires six finite rigid
motion columns and one confounds row per BOLD frame. It fails closed on a
missing file, path/run mismatch, modified hash, or incomplete evidence chain.
It additionally requires the selected derivative BOLD and T1w JSON `Sources`
to name the exact raw BOLD and T1w from the signed acquisition record. The
wrapper authenticates the exact scan, complete raw-BIDS input inventory, and
complete pinned runtime before starting fMRIPrep and again after successful
execution. It embeds both verifications, the signed source/runtime identities,
and the before/after executable version probes in its execution record.

The A4 native-data recovery path is separate and is never relabelled as the
paper profile. `native_structural_identity.py` authenticates one externally
pinned, BOLD-free `connect4-native-structural-batch-v4` cohort authority whose
rows bind signed `connect4-native-structural-scan-v4` records. It normalizes
each exact 61x73x61 T1 inside its exact SynthSeg support and applies constant
zero padding `[before=(1,3,1), after=(2,4,2)]` to T1 and segmentation for the
64x80x64 architecture. It performs no post-native interpolation, shifts the
affine so the unpadded crop retains its native world coordinates, and records
`matrix_size_reported_by_paper=false` and `paper_certified=false`.
The independent allowlisted target receipt must reproduce the canonical
`connect4-native-spatial-target-join-v1` geometry before applying the same
padding to BOLD; no structural authority or cache embeds a BOLD field/digest.
For compatibility with fMRIPrep 23.x derivatives, the documented legacy
`from-scanner_to-T1w_mode-image_xfm` name is also accepted and explicitly
identified as `TransformFrom: "scanner"` in provenance.

The derivative BOLD JSON must set `SliceTimingCorrected: true`, fMRIPrep's
standard completion flag after it removes now-invalid raw slice acquisition
times from a realigned derivative. Source runs shorter than 128 acquired frames
are rejected even if TR interpolation could create 128 samples. Runs that become
shorter than 128 after TR harmonisation are also rejected; reflect padding and
all other synthetic-frame creation are forbidden. If extra
fMRIPrep arguments are needed, place them after `--`; the wrapper rejects
attempts to override its participant/output space or disable slice timing:

```bash
python -m preprocessing.run_fmriprep \
    --bids-root /bids \
    --derivatives-root /derivatives/fmriprep/sub-01_task-rest_generation-001 \
    --participant-label 01 \
    --scan-id sub-01_task-rest \
    --source-acquisition-identity /authority/sub-01_source_acquisition.json \
    --source-acquisition-identity-sha256 "$CONNECT4_SOURCE_ACQUISITION_SHA256" \
    --fmriprep-runtime-identity /authority/fmriprep_runtime_identity.json \
    --fmriprep-runtime-identity-sha256 "$CONNECT4_FMRIPREP_RUNTIME_SHA256" \
    --fmriprep-executable /runtime/bin/fmriprep \
    -- --fs-no-reconall
```

Production applies the versioned recovery choices of 3-mm FWHM mask-normalized
Gaussian smoothing and a 0.01-0.10 Hz resting-state band-pass filter. The paper
names the operations but does not report these numeric settings. Signal and
smoothed-mask division prevents zero-padding from darkening the cortical edge;
the measured in-mask spatial-gradient retention ratio must be strictly greater
than 0.30:

```bash
python -m preprocessing.preprocess_fmri \
    --fmri /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz \
    --metadata /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_space-T1w_desc-preproc_bold.json \
    --t1 /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz \
    --fmriprep-dataset-description /derivatives/fmriprep/sub-01_task-rest_generation-001/dataset_description.json \
    --fmriprep-execution-record /derivatives/fmriprep/sub-01_task-rest_generation-001/logs/connect4-fmriprep_sub-01_task-rest_execution.json \
    --source-acquisition-identity /authority/sub-01_source_acquisition.json \
    --source-acquisition-identity-sha256 "$CONNECT4_SOURCE_ACQUISITION_SHA256" \
    --motion-confounds /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_desc-confounds_timeseries.tsv \
    --coregistration-transform /derivatives/fmriprep/sub-01_task-rest_generation-001/sub-01/func/sub-01_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt \
    --brain-mask seg_common.nii.gz \
    --common-grid-contract /path/to/connect4_structural_common_grid.json \
    --common-grid-contract-sha256 "$CONNECT4_COMMON_GRID_SHA256" \
    --out bold_common.nii.gz \
    --smoothing-fwhm-mm 3 --high-pass-hz 0.01 --low-pass-hz 0.10
```

For `bold_common.nii.gz`, provenance is written to `bold_common.json`. The production
contract requires `PreprocessingSchemaVersion ==
"connect4-fmriprep-preprocessing-v9"`; `FMRIPrepEvidence` has
`EvidenceSchemaVersion == "connect4-fmriprep-evidence-v4"`. Only wrapper
execution schema `connect4-fmriprep-execution-v4` is accepted. It binds the
exact `ScanIdentity`, complete BIDS-input inventory, fresh per-scan derivative
generation, executable and complete `connect4-fmriprep-runtime-identity-v1`
inventory, fixed slice-time reference 0.5, closed command arguments, and
before/after version probes. Its derivative artifact fields
are `ExecutionRecord`, `DerivativeDatasetDescription`, `BOLDDerivative`,
`BOLDMetadata`, `BOLDBrainMask`, `MotionConfounds`, `BOLDToT1wTransform`, `T1wReference`,
`T1wMetadata`, `StdoutLog`, and `StderrLog`; each contains `Path`, `RelativePath`, `SHA256`, and
`SizeBytes`. `ExecutionWrapper`, `ExecutionSuccess`, `ExecutionReturnCode`,
`ExecutionStartedAtUTC`, and `ExecutionCompletedAtUTC` snapshot the verified
run. `DerivativeSliceTimingCorrected` snapshots the selected derivative's
standard `SliceTimingCorrected: true` flag. `CoregistrationBinding` records the
BOLD run prefix, transform directions,
T1 subject/session, and hashes for the BOLD, transform, and exact target T1w.
It also binds the raw T1w, raw BOLD and SynthSeg-mask digests.
`SourceAcquisitionIdentity` records the external identity path/hash/size/record
digest, while `SourceAcquisition` embeds the complete authenticated record.
The top level also records
`CoregistrationTransformSHA256`, `MotionParametersSHA256`, and `OutputSHA256`.
`OutputSHA256` binds the sidecar to the exact saved harmonised fMRI bytes. The
production loader recomputes that digest and each source artifact's
`SHA256`/`SizeBytes`; therefore the fMRIPrep derivative root and wrapper logs
must remain available and unchanged for training-time validation.

`ManuscriptClaims` records 128 frames, TR 3 s and 3-mm voxels while explicitly
leaving spatial matrix, smoothing FWHM and filter cutoffs null. The separate
`VersionedRecoveryChoices` object records the structural-grid schema, 3-mm
mask-normalized smoothing, >0.30 gradient-retention policy and filter cutoffs.
It also requires exactly one post-fMRIPrep interpolation of the target data:
the already coregistered T1w-space BOLD derivative is transformed directly
from its own sampled grid to the cohort-common grid. An intermediate resampling
onto the fMRIPrep T1 image is forbidden because chaining two linear
interpolations erases fine spatial texture before the smoothing-retention gate.
`CommonGridContract`, `AnatomicalMaskSHA256`, `FunctionalValidityMask`, and
`TemporalPaddingApplied=false` are mandatory training evidence. The anatomical
mask remains the preserved structural label support used for conditioning; it
is not the fMRI smoothing or loss support. Production automatically resolves
the exact same-run fMRIPrep `space-T1w_desc-brain_mask.nii.gz`, binds it through
the execution inventory and evidence record, resamples it with nearest-neighbour
interpolation, and intersects it with structural support. The emitted
`bold_common_functional_validity_mask.nii.gz` is used for smoothing,
normalization, final zeroing, and model-loss validity. Its bytes are hash-bound,
and the sidecar asserts that it equals the exact nonzero support of the saved
4D fMRI; a missing, changed, empty, or unequal mask fails closed.

The paper does not report an fMRI intensity normalizer. The explicitly
unpublished recovery contract
`connect4-unpublished-v54b-functional-validity-q995-nonnegative-v2` clamps
negative values, computes q99.5 from 4D values inside that functional-validity
mask only, clips to `max(q99.5, 1)`, and divides by the same value. Background
and padded voxels therefore cannot change a valid signal's scale. Provenance
records the in-support value count, quantile, negative fraction, and fraction
actually clipped at the ceiling; the production loader rejects the retired
whole-volume contract and malformed or inconsistent records.

The output JSON snapshots the fMRIPrep name/version and every CONNECT-4
postprocessing parameter. The local FLIRT/MCFLIRT/SciPy routines are available
only with `--preprocessing-backend custom-debug`; such outputs set
`PaperRequiredStepsComplete=false` and are rejected by the training loader.

Downstream loaders can reject legacy/incomplete targets by requiring the JSON
sidecar fields `PreprocessingSchemaVersion == "connect4-fmriprep-preprocessing-v9"`,
`PaperRequiredStepsComplete == true`, `FMRIPrepApplied == true`, a valid
`FMRIPrepGeneratedBy` name/version snapshot, `CoregistrationValidated == true`,
`CoregistrationEvidenceValidated == true`, a complete hash-bound
`FMRIPrepEvidence` including its exact `BOLDBrainMask`,
`SliceTimingApplied == true`, `MotionCorrectionApplied == true`,
`SpatialSmoothingApplied == true`, `TemporalFilteringApplied == true`, the
hash-bound common-grid/mask fields, exact configured architecture shape plus
128 frames, `RepetitionTime == 3.0`, and `VoxelSize == [3.0,3.0,3.0]`.

The separate native ec46 preprocessing lineage does not contain this
raw-BIDS/SynthSeg/fMRIPrep acquisition chain and remains explicitly
noncertified for this paper path. Rehashing or relabelling an ec46 artifact
cannot upgrade it to `connect4-fmriprep-preprocessing-v9`.

Outputs on the contract's padded architecture grid are `T1w_common.nii.gz`,
`seg_common.nii.gz`, `bold_common.nii.gz` (128 real frames, TR = 3 s), and
`bold_common_functional_validity_mask.nii.gz`.
Only architecture internals retain the recorded padding; released prediction
bundles are cropped to the contract's anatomical shape and affine.

After arranging the conformed files under the configured `T1/` and `Masks/`
directories, build an independently reviewed, target-free CAT12 authority for
the train T1w scans. AnatCL does **not** accept those native/common-grid T1w
volumes directly: version 0.0.2 is a 3D model trained on CAT12 modulated,
normalised gray-matter VBM (`mwp1`/`mwp1u`) at 1.5 mm. The authority must use
the pinned brainprep/CAT12/SPM/MCR pipeline, resample the SynthSeg labels onto
the exact VBM grid with nearest-neighbour interpolation, and publish an
externally pinned manifest. Then extract the two ROI feature families:

```bash
python -m preprocessing.extract_roi_features \
  --root /path/to/data \
  --anatcl-weights /immutable/official/weights.pth \
  --cat12-authority-root /immutable/train-cat12-authority \
  --cat12-authority-manifest /immutable/train-cat12-authority.json \
  --cat12-authority-manifest-sha256 EXTERNALLY_REVIEWED_SHA256 \
  --pyradiomics-runtime-lock /immutable/pyradiomics-runtime-lock.json \
  --pyradiomics-runtime-lock-sha256 EXTERNALLY_REVIEWED_SHA256
python -m scripts.preprocess_graphs --config configs/connect4.yaml
```

The first command accepts only the exact official AnatCL 0.0.2
resnet18/global/fold-0 source and weight hashes. The paper does not specify the
3D ROI-to-AnatCL adaptation, so the implementation records a single explicit
rule: mask the CAT12 VBM to one resampled ROI, center-crop 121x145x121 to
121x128x121 exactly as upstream, zero-pad to 128 cubed, and obtain one 512-D
global descriptor. Raw/native T1w and old three-plane caches are rejected.

Radiomics uses official PyRadiomics 3.0.1 defaults from a separately frozen
CPython-3.10 Linux runtime. `pyradiomics-cuda` is a different fork and is
categorically rejected. The source-controlled
`pyradiomics_official_3_0_1.lock.json` pins the official sdist and upstream
revision but deliberately has `production_ready=false` until a Linux runtime
and its complete installed-file manifest have been built, tested, frozen, and
externally reviewed. Thus feature extraction currently fails closed rather
than mislabelling the CUDA fork or silently using unfrozen dependencies.
When that lock is activated, the extractor requires the locked CPython ABI and
platform, rehashes every regular file in the non-writable runtime through
descriptor-held `O_NOFOLLOW` snapshots, rejects missing/extra files, links and
special files, and proves that both distribution metadata and the actual
`radiomics` import resolve inside the authenticated tree before importing it.
Graph-cache provenance extracts the scan-independent AnatCL model/extraction
state, PyRadiomics version and ordered feature schema, and canonical 32-ROI
order into a shared conditioning identity. It also hashes the BrainIAC
loader/architecture source, not only its checkpoint, and requires both expected
hashes up front. Clinical-ModernBERT is fixed to its upstream model ID and an
immutable revision. Training and external preprocessing must therefore use the
same feature-generation protocol.

## Modules

| file | does |
|------|------|
| `build_common_grid_contract.py` | derive, hash and record the T1-only cohort-common grid and patch-only padding |
| `conform.py`            | validate the pinned contract, resample to its 3-mm grid, crop padding, and reject short runs |
| `preprocess_t1.py`      | T1w → conform + intensity z-score |
| `preprocess_seg.py`     | conform your **external SynthSeg** label map onto the grid |
| `run_fmriprep.py`       | enforce production fMRIPrep arguments and write the hashed execution record |
| `preprocess_fmri.py`    | verified fMRIPrep T1w-space BOLD → smoothing → temporal filtering → 128 frames @ TR 3 s, 3 mm |
| `preprocess_subject.py` | consumes an already completed wrapper-recorded fMRIPrep derivative, then runs the T1/segmentation/fMRI postprocessing steps for one subject |
| `extract_roi_features.py` | subject-specific PyRadiomics + pretrained AnatCL features with source/model provenance |

## Notes

* Motion correction and BOLD→T1w co-registration come from fMRIPrep in the
  production path. The local backends exist only for explicit diagnostics and
  never produce training-eligible provenance.
* If the source TR differs from 3 s, the temporal series is genuinely resampled
  with anti-alias filtering before frame-count harmonisation; the code does not
  merely relabel the NIfTI header.
* After ROI extraction, run `scripts/preprocess_graphs.py` to extract the
  remaining frozen-FM node features and the coverage-weighted hypergraphs
  consumed by training.
