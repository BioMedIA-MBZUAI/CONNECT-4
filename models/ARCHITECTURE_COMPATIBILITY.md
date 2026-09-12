# Reconciled production architecture checkpoint compatibility

The paper-alignment changes intentionally alter learned parameter semantics and
are not strictly compatible with checkpoints from the earlier research tree:

- `MultiModalFusion` now has modality input projections and one
  `shared_encoder`. Older `image_encoder.*`, `mask_encoder.*`,
  `roi_encoder.*`, and `hypergraph_encoder.*` weights cannot be mapped exactly
  because those checkpoints learned different modality-specific operations.
- `GroupAwareNodes2Token` now learns the paper's scalar hyperedge score and
  performs patch-local softmax pooling before adding a deterministic 3D
  positional encoding. The former nested multi-head pooling weights have no
  equation-preserving mapping.
- ImageGraph, Nodes2Token, and DiT now share the geometric patch-centre
  convention `start + width / 2 - 0.5` on the voxel-centre grid. Checkpoints
  whose positional encoding used `start + width / 2` for even patches are
  shifted by half a voxel on every even-width axis and are incompatible.
- `DiT4DTemporal` is now a token-latent epsilon denoiser with T1/graph
  cross-attention. A clean 4D target is patch-encoded once, iid noise is added
  and predicted in all token coordinates, and recovered x0 tokens are decoded
  to scalar patches exactly once before TC-UNet. The v9 raw-space objective
  tried to predict a 65,536-value iid epsilon patch through a 512-wide token,
  leaving at least 99.21875% of raw noise directions unreachable. Its weights
  and optimizer state are incompatible.
- Figure 1 requires a full-grid 64-channel TC-UNet input. The token x0 is
  decoded once to scalar patches on the configured D,H,W grid and a per-frame
  1x1x1 projection maps it to 64 channels. The linear token codec, this
  projection operator, and voxelwise channel normalization are explicit
  implementation choices because the manuscript does not specify them. The
  codec is constrained by an implementation-only voxel plus temporal/spatial
  first-difference reconstruction objective at fixed weight 1.0, normalized
  only over an explicit authenticated structural brain mask and adjacent
  in-brain pairs. ROI-union fallback is forbidden and every ROI foreground
  voxel must lie inside that structural mask. Published
  Figure-1E lambdas are unchanged. TC-UNet pooling and transposed convolution
  preserve D and scale only H,W, with channel widths 64/128/256/512. The final
  head is a per-frame 1x1x1 convolution at unchanged D,H,W. The former
  low-resolution latent, all-axis pooling, progressive reconstruction head, and
  interpolation repair are all incompatible and forbidden.
- The fixed spatial matrix has been removed. Tensor shapes now come from an
  externally SHA-pinned, T1-only common-grid contract, and patch-only padding is
  cropped before publication. A checkpoint is valid only with the same bound
  grid identity.
- `FullResolutionT1DetailPath` is forbidden in synthesis. Even a bounded,
  mask-normalized version copies anatomical high frequencies into the output
  and can make a smooth fMRI prediction look textured without learning fMRI
  dynamics. The module remains only as legacy audit code; `Connect4Model`
  rejects any configuration that enables it.
- The non-certified recovery DiT grid is now the complete signed 64x80x64
  padded grid with 16x16x16 patches, one-to-one with the 80 structural patches.
  The rejected v8 16x20x16 scalar latent and 4x reconstruction path caused the
  architecture mismatch and removed spatial detail before TC-UNet could use it.
- Figure 1 visibly labels the paper-profile structural tensors 128-cubed. The
  64x80x64 output above is therefore defensible only for the explicitly
  non-certified A4 recovery profile; it is not exact paper spatial fidelity.
  The paper profile now declares a full 128-cubed scalar DiT grid with 16-cubed
  patches, matching the 512 structural patches. It remains explicitly blocked:
  the reviewed contextual BrainLM authority covers only the signed recovery
  grid, and an omitted, synthetic, or silently remapped perceptual term cannot
  qualify a 128-cubed paper run. A distinct reviewed 128-grid BrainLM authority
  is required before the paper A100 benchmark may run.
  TC-UNet uses exact D slabs: its 14-voxel halo is derived from the instantiated
  longest convolutional path, overlap is cropped (never averaged), and local
  channel normalization keeps that receptive field finite. Cores 1, 2, 4, and
  larger powers of two are measured through full depth or OOM; production must
  SHA-bind the sweep's largest core that completes a full AdamW step at least
  four GiB below physical memory. The checked-in paper core is therefore null,
  not an unmeasured permanent value of one.
- Composite biological losses train through a single randomly noised,
  differentiable token-x0 estimate and its one-time patch decoder into fusion,
  the DiT denoiser, token decoder, and TC-UNet. This is
  a memory-bounded diffusion-training surrogate; the implementation does **not**
  claim to backpropagate through all 50 DDIM inference steps. The biological
  objective is multiplied by mean alpha-cumprod at the sampled noise level to
  bound x0-to-epsilon gradient amplification. Component lambdas remain the
  published values. Full target-blind DDIM is used for development/inference.
- Temporal coherence is target-delta-relative L1: each subject's first-order
  error is divided by the real in-mask mean absolute first difference. A static
  temporal-mean prediction therefore has loss one even when BOLD fluctuations
  are small relative to baseline. The manuscript publishes lambda_temporal=0.2
  but not the term's equation; this scale-normalized anti-collapse definition is
  a versioned recovery choice and adds no hidden auxiliary weight.
- The DDIM sampling seed is part of inference compatibility because it changes
  the scan-bound initial noise and therefore the generated 4D volume.

These changes use synthesis contract v10 and checkpoint format
`connect4_iteration_exact_v10`; v9 and earlier checkpoints must fail closed.

Production uses one frame per temporal group, reducing the linear token codec
from v9's 128x compression (65,536 scalar values to width 512) to 8x (4,096 to
512). Recovery therefore uses 10,240 DiT tokens (128 x 80 spatial patches); the
paper profile uses 65,536 (128 x 512). This materially raises attention memory,
so neither static arithmetic nor these CPU tests authorize training. Fresh
one-A100 peak allocated/reserved and exact four-A100 DDP gates are mandatory.

Start a fresh reconciled production training run. For feature-extractor transfer only,
load a legacy state dict with `strict=False` and explicitly inspect both missing
and unexpected keys; do not resume its optimizer state or describe the result
as a resumed CONNECT-4 diffusion run.
