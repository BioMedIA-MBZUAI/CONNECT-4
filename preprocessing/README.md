# CONNECT-4 preprocessing

Brings every subject onto the paper's common acquisition grid:

| property        | value                          |
|-----------------|--------------------------------|
| matrix size     | **128 × 128 × 128**            |
| voxel size      | **3 mm isotropic**             |
| rs-fMRI frames  | **128**                        |
| rs-fMRI TR      | **3 s**                        |

Both the T1w and the rs-fMRI are conformed to 128³ @ 3 mm; the rs-fMRI is
registered into the subject's T1 space and forced to exactly 128 frames.

## Segmentation — SynthSeg (run externally)

Anatomical labels are produced with **SynthSeg** (Billot et al.), a contrast- and
resolution-agnostic CNN segmenter: <https://github.com/BBillot/SynthSeg>.

> **SynthSeg is run by you, externally** (per its own repo/instructions). This
> repository does **not** call SynthSeg — it only **consumes** the label map you
> produce. `preprocess_seg.py` simply conforms your SynthSeg output onto the
> CONNECT-4 grid (nearest-neighbour).

The labels drive the mask patches / ROI statistics (Fig 1A), the ROI graph and
ROI-coverage hyperedges (Fig 1B), and the measured ROI volumes used by the
normative / atrophy module (`normative/`).

## Pipeline

```bash
# 0) FIRST run SynthSeg yourself on the T1w (see the SynthSeg repo), e.g.:
#    mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz --robust

# 1) whole subject (T1 + provided SynthSeg labels + fMRI) in one call
python -m preprocessing.preprocess_subject \
    --t1 sub-01_T1w.nii.gz --seg sub-01_synthseg.nii.gz \
    --fmri sub-01_bold.nii.gz --out_dir derivatives/sub-01
```

Or step by step:

```bash
python -m preprocessing.preprocess_t1   --in sub-01_T1w.nii.gz --out T1w_128.nii.gz
python -m preprocessing.preprocess_seg  --in sub-01_synthseg.nii.gz --out seg_128.nii.gz   # your external SynthSeg output
python -m preprocessing.preprocess_fmri --fmri sub-01_bold.nii.gz --t1 T1w_128.nii.gz --out bold_128.nii.gz --mcflirt
```

Outputs (all 128³ @ 3 mm): `T1w_128.nii.gz`, `seg_128.nii.gz`,
`bold_128.nii.gz` (128 frames, TR = 3 s).

## Modules

| file | does |
|------|------|
| `conform.py`            | resample to 128³ @ 3 mm (trilinear / nearest), frame-count + z-score helpers |
| `preprocess_t1.py`      | T1w → conform + intensity z-score |
| `preprocess_seg.py`     | conform your **external SynthSeg** label map onto the grid |
| `preprocess_fmri.py`    | rs-fMRI → register to T1 → conform → 128 frames @ TR 3 s → temporal z-score |
| `preprocess_subject.py` | runs all of the above for one subject |

## Notes

* Motion correction uses FSL `mcflirt` if available (`--mcflirt`); otherwise it
  is skipped — run `mcflirt`/SynthMorph beforehand for best results.
* fMRI→T1 registration reuses `utils/registration.register_fmri_to_t1w_nib`
  (nibabel `resample_from_to`, RAS+).
* After preprocessing, run `scripts/preprocess_graphs.py` /
  `scripts/precompute_hypergraphs.py` to extract the frozen-FM node features and
  hypergraphs consumed by training.
```
