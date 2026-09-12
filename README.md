# CONNECT-4

Official implementation of **CONNECT-4: Brain Connectivity-Guided Hyperedge
Graph Fusion for Structural MRI to 4D Rest Functional MRI Synthesis** (MICCAI
2026).

![CONNECT-4 architecture](figures/Arch_Fig.jpg)

CONNECT-4 builds three aligned structural streams from T1w MRI, anatomical
segmentation and ROI features. The streams are enriched with DWI connectivity,
known functional-connectivity descriptions and subject-specific normative
volumes, then fused through coverage-weighted hyperedges into conditioning
tokens for token-space diffusion and a temporal UNet decoder.

## Paper-to-code map

- `preprocessing/`: fMRIPrep-compatible target preparation, structural
  alignment, fixed 128-frame/3-second-TR contract and provenance.
- `graphs/`: Chebyshev-distance image graph, DWI mask graph, DWI ROI graph and
  fractional-coverage hyperedges.
- `models/fusion.py`: shared graph-attention and group-aware Nodes2Token fusion.
- `models/dit4d_temporal.py`: spatiotemporal token diffusion with timestep and
  fused-graph conditioning.
- `models/tc_film_unet.py`: temporal-attention UNet reconstruction.
- `models/losses.py`: voxel, frame-wise SSIM, temporal coherence, BrainLM
  perceptual, ROI intensity/distribution and functional-connectivity losses.
- `eval/metrics.py`: MSE, SSIM, voxel, ROI and frame-to-frame correlations, and
  PSNR; the sealed evaluator additionally supports authenticated FID/IS.
- `training/train.py`: four-GPU DDP training with patient-disjoint fixed splits,
  resumable checkpoints and full-development anti-collapse QA.
- `inference/infer.py`: target-blind 4D synthesis and NIfTI publication.

The code rejects the previously mistaken generic image ViT-MAE artifact:
training-time perceptual features come only from the pinned official BrainLM
checkpoint. SLIM-Brain is evaluation-only and has no synthetic fallback.

## Installation

Create an isolated Python environment and install:

```bash
pip install -r requirements.txt
```

Foundation-model weights are intentionally not bundled. Set their locations
and immutable hashes in a run-specific YAML copied from
`configs/connect4.yaml`. The committed configuration is a template and does
not contain workstation or cluster paths.

## Data preparation

Prepare T1w images, SynthSeg label maps and fMRI targets first. For the paper
profile, fMRI preprocessing evidence must include co-registration, slice-timing
and motion correction, smoothing, temporal filtering, 3 mm isotropic spatial
harmonisation, and exactly 128 frames at TR=3 seconds.

```bash
python -m preprocessing.preprocess_subject --help
python scripts/preprocess_graphs.py --help
python scripts/build_split_manifest.py --help
```

Precomputed graph records include the three streams, ROI coverage, DWI matrix,
normative covariates and immutable source identities. Patient roles are read
from a supplied split manifest; training never creates or changes the split.

## Training

CONNECT-4 uses a global batch of four, AdamW, and up to 200 epochs. Production
training requires exactly four CUDA ranks:

```bash
torchrun --standalone --nproc_per_node=4 -m training.train \
  --config /absolute/path/to/run.yaml
```

For Slurm, provide the isolated interpreter and run configuration explicitly:

```bash
export CONNECT4_PYTHON=/absolute/path/to/python
export CONNECT4_CONFIG=/absolute/path/to/run.yaml
export CONNECT4_RUNTIME_IDENTITY=/absolute/path/to/runtime.json
sbatch scripts/train.slurm
```

The launcher requests four A100 GPUs and refuses execution on a login node.
Training consumes only train and development roles; sealed-test targets are
not opened by the training module.

## Inference and visualization

```bash
python -m inference.infer \
  --config /absolute/path/to/run.yaml \
  --checkpoint /absolute/path/to/connect4_epoch199.pt \
  --out_dir /absolute/path/to/predictions \
  --num 8 --visualize
```

Inference writes target-blind 4D NIfTI predictions. Paired metrics and
real-versus-predicted figures belong to the post-seal evaluation stage, after
the complete prediction set has been frozen.

## Tests

```bash
python -m pytest -q
```

The suite covers preprocessing contracts, three-stream graph and hypergraph
fusion, temporal diffusion, losses, fixed patient splits, sealed inference,
metrics, texture/temporal anti-collapse checks and artifact provenance.
