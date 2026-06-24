"""
Dataset that loads pre-computed graphs and embeddings.
Much faster than computing on-the-fly.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import json
import sys
import nibabel as nib
from nibabel.processing import resample_from_to
from utils.registration import register_fmri_to_t1w_nib
from utils.scalers import FeatureScalerManager

from .patchify import Patchify3D

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class Connect4PrecomputedDataset(Dataset):
    """
    Dataset that loads pre-computed graphs and embeddings.
    """
    
    ROI_SPECS = [
        ("left_cerebral_white_matter", 2),
        ("left_cerebral_cortex", 3),
        ("left_lateral_ventricle", 4),
        ("left_inferior_lateral_ventricle", 5),
        ("left_cerebellum_white_matter", 7),
        ("left_cerebellum_cortex", 8),
        ("left_thalamus", 10),
        ("left_caudate", 11),
        ("left_putamen", 12),
        ("left_pallidum", 13),
        ("third_ventricle", 14),
        ("fourth_ventricle", 15),
        ("brain_stem", 16),
        ("left_hippocampus", 17),
        ("left_amygdala", 18),
        ("csf", 24),
        ("left_accumbens_area", 26),
        ("left_ventral_dc", 28),
        ("right_cerebral_white_matter", 41),
        ("right_cerebral_cortex", 42),
        ("right_lateral_ventricle", 43),
        ("right_inferior_lateral_ventricle", 44),
        ("right_cerebellum_white_matter", 46),
        ("right_cerebellum_cortex", 47),
        ("right_thalamus", 49),
        ("right_caudate", 50),
        ("right_putamen", 51),
        ("right_pallidum", 52),
        ("right_hippocampus", 53),
        ("right_amygdala", 54),
        ("right_accumbens_area", 58),
        ("right_ventral_dc", 60),
    ]
    
    NUM_ROIS = len(ROI_SPECS)
    ID_TO_SLUG = {label_id: slug for slug, label_id in ROI_SPECS}
    
    def __init__(
        self,
        root_dir: str,
        precomputed_dir: str = "/path/to/data/reconstructed_graphs",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        num_frames: int = 128,               # rs-fMRI frames to keep (paper = 128)
        validate_files: bool = True,         # drop scans with missing/empty files at startup
        normalize_intensity: bool = True,
        register_fmri_to_t1w: bool = False,  # Skip registration - assume data already aligned
        fmri_scale_factor: float = 1.0,  # Scale factor to adjust fMRI size (1.0 = no scaling, >1.0 = larger, <1.0 = smaller)
        brain_mask_dir: str = "/path/to/data/fMRI_mask",
        scaler_dir: Optional[str] = None,  # Directory containing saved scalers
        hypergraphs_dir: Optional[str] = "/path/to/data/graphs_final",  # Directory with precomputed hypergraphs
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.precomputed_dir = Path(precomputed_dir)
        self.target_shape = target_shape
        self.num_frames = num_frames
        self.normalize_intensity = normalize_intensity
        self.register_fmri_to_t1w = register_fmri_to_t1w
        self.fmri_scale_factor = fmri_scale_factor
        self.brain_mask_dir = Path(brain_mask_dir)
        # Structure to ROI index mapping
        self.structure_to_roi_idx = {
            label_id: idx for idx, (_, label_id) in enumerate(self.ROI_SPECS)
        }
        
        # Set hypergraphs directory (for precomputed hypergraphs)
        if hypergraphs_dir:
            self.hypergraphs_dir = Path(hypergraphs_dir) / "hypergraphs"
        else:
            # Fallback to old location
            self.hypergraphs_dir = self.precomputed_dir / "hypergraphs"
        
        # Load scalers if available
        self.scaler_manager = None
        if scaler_dir:
            scaler_dir_path = Path(scaler_dir)
            if scaler_dir_path.exists():
                self.scaler_manager = FeatureScalerManager()
                self.scaler_manager.load_scalers(scaler_dir_path)
                print(f"[Dataset] Loaded scalers from {scaler_dir_path}", flush=True)
            else:
                print(f"[Dataset] Warning: Scaler directory not found: {scaler_dir_path}", flush=True)
        
        # Discover scan IDs from precomputed graphs
        graphs_dir = self.precomputed_dir / "graphs"
        if not graphs_dir.exists():
            raise RuntimeError(f"Precomputed graphs directory not found: {graphs_dir}")
        
        # Get scan IDs from graph files
        graph_files = list(graphs_dir.glob("*_image_nodes.npy"))
        self.scan_ids = sorted(set(f.name.replace("_image_nodes.npy", "") for f in graph_files))
        
        if not self.scan_ids:
            raise RuntimeError(f"No precomputed graphs found in {graphs_dir}")

        print(f"Found {len(self.scan_ids)} precomputed samples")

        # Drop scans with missing or empty (0-byte) required files up front, so
        # they are never attempted during training (no per-sample skip spam, and
        # the patient split sees only usable scans). Result is cached.
        if validate_files:
            self.scan_ids = self._filter_valid_scans(self.scan_ids)
        
        # Load text descriptions CSV (optional - not needed for precomputed embeddings)
        text_csv = self.precomputed_dir / "patch_text_descriptions.csv"
        if text_csv.exists():
            self.text_df = pd.read_csv(text_csv)
        else:
            self.text_df = None
            # No warning needed - embeddings are already precomputed
        
        # Load DWI matrix (required)
        dwi_path = Path("/path/to/data/dwi_matrix.csv")
        if not dwi_path.exists():
            raise FileNotFoundError(
                f"DWI matrix not found at {dwi_path}. "
                f"This file is required for graph construction."
            )
        self.dwi_matrix = torch.from_numpy(np.loadtxt(dwi_path, delimiter=",", dtype=np.float32))
        
        # Validate DWI matrix size
        if self.dwi_matrix.shape != (self.NUM_ROIS, self.NUM_ROIS):
            raise ValueError(
                f"DWI matrix must be {self.NUM_ROIS}x{self.NUM_ROIS}, "
                f"got {self.dwi_matrix.shape}"
            )
        
    def _get_brain_mask_path(self, scan_id: str) -> Optional[Path]:
        """
        Attempt to find the brain mask file for a given scan_id.
        Filenames in /path/to/data/fMRI_mask can vary slightly,
        so try several common patterns.
        """
        candidates = [
            f"{scan_id}_fMRI.nii.gz",  # Current format: B10081264_004_fMRI.nii.gz
            f"{scan_id}brainmask.nii.gz",
            f"{scan_id}_brainmask.nii.gz",
            f"{scan_id}_brainmask.nii",
            f"{scan_id}brainmask.nii",
            f"{scan_id}_mask.nii.gz",
            f"{scan_id}_mask.nii",
        ]
        for fname in candidates:
            path = self.brain_mask_dir / fname
            if path.exists():
                return path
        return None
    
    def _get_mean_fmri_mask_path(self, scan_id: str) -> Optional[Path]:
        """
        Get the path to the mean fMRI binary mask.
        File naming pattern: {scan_id}brainmask.nii.gz
        """
        mean_mask_dir = Path("/path/to/data/fMRI_mean_mask")
        mask_path = mean_mask_dir / f"{scan_id}brainmask.nii.gz"
        return mask_path if mask_path.exists() else None
    
    def _load_nifti(self, path: Path, is_fmri: bool = False, is_mask: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
        """Load NIfTI file and return tensor with affine."""
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        affine = img.affine.copy()
        
        if data.ndim == 4 and is_fmri:
            tensor = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0).float()
            if tensor.shape[2:5] != self.target_shape:
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
        elif data.ndim == 3:
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
            # Use nearest neighbor for masks to preserve exact label values
            if is_mask:
                mode = "nearest"
                align_corners = None  # align_corners not valid for nearest mode
            else:
                mode = "trilinear" if not is_fmri else "nearest"
                align_corners = False
            
            if align_corners is not None:
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.target_shape,
                    mode=mode,
                    align_corners=align_corners,
                )
            else:
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.target_shape,
                    mode=mode,
                )
        else:
            raise ValueError(f"Expected 3D or 4D volume, got shape {data.shape}")
        
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # NO NORMALIZATION - Use raw values as-is (normalization completely removed)
        # Keep raw tensor values without any normalization, scaling, or processing
        
        return tensor, affine
    
    def _required_files(self, sid: str):
        """
        Files __getitem__ loads for one scan, split into:
          required : must exist and be non-empty
          optional : tolerated if missing, but must be non-empty if present
                     (fMRI_mean is recomputed on-the-fly; metadata.json defaults)
        """
        g = self.precomputed_dir / "graphs"
        hg_old = self.precomputed_dir / "hypergraphs"
        required = [
            self.root / "T1" / f"{sid}_T1.nii.gz",
            Path("/path/to/data/fMRI_norm") / f"{sid}_fMRI.nii.gz",
            self.root / "Masks" / f"{sid}_mask.nii.gz",
            g / f"{sid}_image_nodes.npy", g / f"{sid}_mask_nodes.npy", g / f"{sid}_roi_nodes.npy",
            hg_old / f"{sid}_patch_distributions.json",
            self.hypergraphs_dir / f"{sid}_hyperedge_index.pt",
            self.hypergraphs_dir / f"{sid}_hyperedge_weights.pt",
        ]
        optional = [
            Path("/path/to/data/fMRI_mean_norm") / f"{sid}_fMRI.nii.gz",
            hg_old / f"{sid}_metadata.json",
        ]
        return required, optional

    def _filter_valid_scans(self, scan_ids):
        """Keep only scans whose required files all exist and are non-empty.

        All the observed corruptions (Empty file / EOFError / 'Expecting value
        line 1 column 1') are 0-byte files, so a size check catches them. The
        valid list is cached next to the graphs to make subsequent runs instant.
        """
        cache = self.precomputed_dir / "_valid_scans_cache.txt"
        if cache.exists():
            try:
                valid = set(cache.read_text().split())
                kept = [s for s in scan_ids if s in valid]
                if kept:
                    print(f"[Dataset] {len(kept)}/{len(scan_ids)} scans valid (cached)", flush=True)
                    return kept
            except Exception:
                pass

        def ok(sid):
            required, optional = self._required_files(sid)
            for f in required:                       # must exist and be non-empty
                try:
                    if f.stat().st_size == 0:
                        return False
                except OSError:
                    return False
            for f in optional:                       # bad only if present-but-empty
                try:
                    if f.stat().st_size == 0:
                        return False
                except OSError:
                    pass
            return True

        valid = [s for s in scan_ids if ok(s)]
        print(f"[Dataset] validated files: {len(valid)}/{len(scan_ids)} usable "
              f"({len(scan_ids) - len(valid)} dropped)", flush=True)
        try:  # atomic write (tmp + rename) so concurrent ranks don't corrupt it
            tmp = cache.with_suffix(".tmp")
            tmp.write_text("\n".join(valid))
            tmp.replace(cache)
        except Exception:
            pass
        return valid

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load tensor from numpy file."""
        return torch.from_numpy(np.load(str(path)))

    def _load_mni_xfm(self, scan_id: str):
        """Cached offline native->MNI registration (4x4) if available, else None."""
        p = Path("/path/to/data/mni_xfm") / f"{scan_id}.npy"
        if p.exists():
            try:
                return np.load(str(p))
            except Exception:
                return None
        return None

    @staticmethod
    def _affine_resample_4d(fmri, fmri_affine, ref_affine, ref_shape, target_shape,
                            world_xfm=None):
        """
        Resample native-space 4D fMRI [T, X, Y, Z] into the reference (T1/MNI)
        grid at `target_shape`.

        The target grid is the reference volume (`ref_shape`, `ref_affine`)
        resampled to `target_shape` (align_corners=False, matching how the ROI
        masks are interpolated). One batched grid_sample handles all T frames.

        `world_xfm` (4x4, MNI-world -> native-world) is the offline dipy
        registration (`preprocessing/register_fmri_mni.py`). When provided this
        is a PROPER registration (Dice ~0.91 with the ROI masks); when None it
        falls back to the header affine alone (rough, ~0.69).
        """
        T, X, Y, Z = fmri.shape
        Dt, Ht, Wt = target_shape
        # target-grid voxel -> reference-volume voxel (align_corners=False)
        A = np.eye(4, dtype=np.float64)
        for a in range(3):
            r = ref_shape[a] / target_shape[a]
            A[a, a] = r
            A[a, 3] = 0.5 * r - 0.5
        ref_aff_tgt = np.asarray(ref_affine, dtype=np.float64) @ A          # target voxel -> MNI world
        inv_fmri = np.linalg.inv(np.asarray(fmri_affine, dtype=np.float64))  # native world -> native voxel
        if world_xfm is not None:
            # target voxel -> MNI world -> (dipy) native world -> native voxel
            M = inv_fmri @ np.asarray(world_xfm, dtype=np.float64) @ ref_aff_tgt
        else:
            M = inv_fmri @ ref_aff_tgt
        M = torch.tensor(M, dtype=torch.float32)

        ii, jj, kk = torch.meshgrid(
            torch.arange(Dt), torch.arange(Ht), torch.arange(Wt), indexing="ij")
        g = torch.stack([ii, jj, kk, torch.ones_like(ii)], dim=-1).float()  # [Dt,Ht,Wt,4]
        fv = torch.einsum("ij,dhwj->dhwi", M, g)[..., :3]                    # fmri voxel coords (x,y,z)
        nx = 2 * fv[..., 0] / max(X - 1, 1) - 1
        ny = 2 * fv[..., 1] / max(Y - 1, 1) - 1
        nz = 2 * fv[..., 2] / max(Z - 1, 1) - 1
        grid = torch.stack([nz, ny, nx], dim=-1).unsqueeze(0)               # grid_sample order (W,H,D)=(Z,Y,X)
        out = torch.nn.functional.grid_sample(
            fmri.unsqueeze(1), grid.expand(T, -1, -1, -1, -1),
            mode="bilinear", align_corners=True, padding_mode="zeros")
        return out[:, 0]                                                     # [T, Dt, Ht, Wt]
    
    def __len__(self) -> int:
        return len(self.scan_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        import os
        rank = os.environ.get('LOCAL_RANK', '0')
        # Print for first few samples or every 100th sample to track progress
        if idx < 3 or idx % 100 == 0:
            print(f"[Dataset RANK={rank}] Loading sample {idx} (scan_id: {self.scan_ids[idx] if idx < len(self.scan_ids) else 'N/A'})...", flush=True)
        scan_id = self.scan_ids[idx]
        
        # Load volumes
        t1_path = self.root / "T1" / f"{scan_id}_T1.nii.gz"
        # Load regular fMRI for temporal dimension (from normalized directory)
        fmri_path = Path("/path/to/data/fMRI_norm") / f"{scan_id}_fMRI.nii.gz"
        # Load precomputed mean fMRI separately (from normalized directory)
        fmri_mean_path = Path("/path/to/data/fMRI_mean_norm") / f"{scan_id}_fMRI.nii.gz"
        mask_path = self.root / "Masks" / f"{scan_id}_mask.nii.gz"
        
        if not all(p.exists() for p in [t1_path, fmri_path, mask_path]):
            raise FileNotFoundError(f"Missing files for {scan_id}")

        # Load T1w and fMRI at original resolution for proper registration
        if idx == 0:
            print(f"[Dataset] Loading T1w from {t1_path}...", flush=True)
        t1w_img = nib.load(str(t1_path))
        t1w_data_orig = t1w_img.get_fdata().astype(np.float32)
        t1w_affine = t1w_img.affine.copy()
        
        # Load precomputed mean fMRI first (faster, smaller file)
        fmri_mean_data_orig = None
        fmri_mean_affine = None
        if fmri_mean_path.exists():
            if idx == 0:
                print(f"[Dataset] Loading precomputed mean fMRI from {fmri_mean_path}...", flush=True)
            fmri_mean_img = nib.load(str(fmri_mean_path))
            fmri_mean_data_orig = fmri_mean_img.get_fdata().astype(np.float32)
            fmri_mean_affine = fmri_mean_img.affine.copy()
            
            # RAISE ERROR if loaded data has NaN/Inf - check immediately after loading from file
            if not np.isfinite(fmri_mean_data_orig).all():
                finite_mask = np.isfinite(fmri_mean_data_orig)
                finite_min = np.amin(fmri_mean_data_orig[finite_mask]) if np.any(finite_mask) else np.nan
                finite_max = np.amax(fmri_mean_data_orig[finite_mask]) if np.any(finite_mask) else np.nan
                num_nonfinite = (~finite_mask).sum()
                raise RuntimeError(
                    f"NaN/Inf in fmri_mean_data_orig loaded from file {fmri_mean_path}. "
                    f"Shape: {fmri_mean_data_orig.shape}, "
                    f"Non-finite count: {num_nonfinite}/{fmri_mean_data_orig.size}, "
                    f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                    f"This indicates a root problem in the data file that needs fixing, not masking."
                )
        else:
            if idx == 0:
                print(f"[Dataset] Precomputed mean fMRI not found at {fmri_mean_path}, will compute temporal mean on-the-fly", flush=True)
        
        # Only load full 4D fMRI if registration is enabled OR if we need to compute mean on-the-fly
        # ALWAYS load full 4D fMRI - we need temporal dimension for training
        # Even if registration is disabled, we still need the full 4D data
        if idx == 0:
            print(f"[Dataset] Loading full 4D fMRI from {fmri_path}...", flush=True)
        if not fmri_path.exists():
            raise FileNotFoundError(
                f"Full 4D fMRI not found at {fmri_path}. "
                f"Required for temporal diffusion training. "
                f"Please ensure fMRI_norm directory contains full 4D fMRI files."
            )
        fmri_img = nib.load(str(fmri_path))
        fmri_data_orig = fmri_img.get_fdata().astype(np.float32)
        fmri_affine = fmri_img.affine.copy()
        
        # Verify we have temporal dimension
        if fmri_data_orig.ndim == 3:
            raise ValueError(
                f"fMRI at {fmri_path} has only 3D (no temporal dimension). "
                f"Shape: {fmri_data_orig.shape}. "
                f"Full 4D fMRI with temporal dimension is required for training."
            )
        if idx == 0:
            print(f"[Dataset] Loaded full 4D fMRI with shape: {fmri_data_orig.shape} (T={fmri_data_orig.shape[0] if fmri_data_orig.ndim == 4 else 1})", flush=True)
        
        # Check if registration is enabled
        if not self.register_fmri_to_t1w:
            # Skip registration - assume data is already aligned
            if idx == 0:
                print(f"[Dataset] Registration disabled - skipping registration step", flush=True)
            
            # Load T1w without resampling
            # Return without batch dimension - DataLoader will add it
            t1w = torch.from_numpy(t1w_data_orig).unsqueeze(0).float()  # [1, D, H, W] - channel dim only
            t1w_affine_resampled = t1w_affine
            
            # Load fMRI without registration (assume already aligned)
            # We always load full 4D fMRI now (see above)
            if fmri_data_orig.ndim == 4:
                # CRITICAL: NIfTI rs-fMRI is stored (X, Y, Z, T) — time is the LAST
                # axis (e.g. (64, 58, 48, 200)). Move it to the FRONT -> (T, X, Y, Z).
                # The previous code treated axis 0 (a spatial dim) as time, scrambling
                # time<->space and interpolating real time-frames into a spatial axis.
                fmri_data_orig = np.ascontiguousarray(np.moveaxis(fmri_data_orig, 3, 0))  # (T, X, Y, Z)
                fmri = torch.from_numpy(fmri_data_orig).float()                            # [T, D, H, W]
                # standardise temporal length to num_frames (center window / repeat-pad)
                Tcur = fmri.shape[0]
                nf = self.num_frames
                if Tcur > nf:
                    s = (Tcur - nf) // 2
                    fmri = fmri[s:s + nf]
                elif Tcur < nf:
                    reps = (nf + Tcur - 1) // Tcur
                    fmri = fmri.repeat(reps, 1, 1, 1)[:nf]
            elif fmri_data_orig.ndim == 3:
                fmri = torch.from_numpy(fmri_data_orig).unsqueeze(0).float()  # [1, D, H, W] - channel dim only
            else:
                raise ValueError(f"Unexpected fMRI shape: {fmri_data_orig.shape}")
            
            # Bring the native-space 4D fMRI into the T1/MNI target grid using the
            # stored NIfTI affines (rough header-affine registration), so it aligns
            # (approximately) with the ROI masks / T1 which live in MNI space.
            # The native fMRI affine is only an approximate MNI transform, so this is
            # a ROUGH registration — adequate for ROI-masked losses, not exact.
            if list(fmri.shape[1:4]) != list(self.target_shape):
                fmri = self._affine_resample_4d(
                    fmri, fmri_affine, t1w_affine,
                    tuple(t1w_data_orig.shape[:3]), tuple(self.target_shape),
                    world_xfm=self._load_mni_xfm(scan_id),   # proper reg if cached, else rough
                )  # [T, *target_shape] in the MNI/T1 grid
            
            # Load precomputed mean fMRI if available (without registration)
            fmri_mean = None
            if fmri_mean_data_orig is not None:
                if idx == 0:
                    print(f"[Dataset] Loading precomputed mean fMRI (no registration)...", flush=True)
                # Mean fMRI is 3D: [D, H, W] -> [1, 1, D, H, W] (match registration path)
                fmri_mean = torch.from_numpy(fmri_mean_data_orig).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
                
                # RAISE ERROR if tensor has NaN/Inf - check after conversion
                if not torch.isfinite(fmri_mean).all():
                    finite = torch.isfinite(fmri_mean)
                    finite_min = torch.amin(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                    finite_max = torch.amax(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                    num_nonfinite = (~finite).sum().item()
                    raise RuntimeError(
                        f"NaN/Inf in fmri_mean tensor AFTER conversion from numpy (no registration path). "
                        f"Shape: {fmri_mean.shape}, "
                        f"Non-finite count: {num_nonfinite}/{fmri_mean.numel()}, "
                        f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                        f"This indicates a root problem in the tensor conversion that needs fixing, not masking."
                    )
                
                # Resize to target shape if needed
                if list(fmri_mean.shape[2:5]) != list(self.target_shape):
                    fmri_mean = torch.nn.functional.interpolate(
                        fmri_mean,  # [1, 1, D, H, W]
                        size=self.target_shape,
                        mode="trilinear",
                        align_corners=False,
                    )  # [1, 1, D', H', W']
                    
                    # RAISE ERROR if interpolation produced NaN/Inf
                    if not torch.isfinite(fmri_mean).all():
                        finite = torch.isfinite(fmri_mean)
                        finite_min = torch.amin(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                        finite_max = torch.amax(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                        num_nonfinite = (~finite).sum().item()
                        raise RuntimeError(
                            f"NaN/Inf in fmri_mean AFTER interpolation (no registration path). "
                            f"Shape: {fmri_mean.shape}, "
                            f"Non-finite count: {num_nonfinite}/{fmri_mean.numel()}, "
                            f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                            f"This indicates a root problem in the interpolation that needs fixing, not masking."
                        )
                
                # NO NORMALIZATION - Use raw mean fMRI values as-is (PSC normalization completely removed)
                # Keep raw mean fMRI values without any normalization, scaling, or processing
                
                if idx == 0:
                    print(f"[Dataset] Mean fMRI loaded (no registration). Shape: {fmri_mean.shape}", flush=True)
            
            # Load mask (assume already in T1w space)
            if idx == 0:
                print(f"[Dataset] Loading mask from {mask_path}...", flush=True)
            mask_img = nib.load(str(mask_path))
            mask_data_orig = mask_img.get_fdata().astype(np.float32)
            
            # Resize mask to target shape if needed
            if mask_data_orig.shape != self.target_shape:
                mask_tensor = torch.from_numpy(mask_data_orig).unsqueeze(0).unsqueeze(0).float()
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=self.target_shape,
                    mode="nearest",
                )
                mask_data = mask_tensor.squeeze(0).squeeze(0).numpy()
            else:
                mask_data = mask_data_orig
            
            # Keep multilabel mask for ROI masks (before binarization)
            multilabel_mask = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
            
            # Create ROI masks from multilabel mask
            roi_masks = {}
            for slug, label_id in self.ROI_SPECS:
                roi_idx = self.structure_to_roi_idx[label_id]
                roi_mask = (multilabel_mask == label_id).float()  # [1, 1, D, H, W]
                roi_masks[roi_idx] = roi_mask.squeeze(0).squeeze(0)  # [D, H, W]
            
            if idx == 0:
                print(f"[Dataset] Created {len(roi_masks)} ROI masks (keys: {list(roi_masks.keys())})", flush=True)
            
            # Binarize mask
            mask_data = (mask_data > 0).astype(np.float32)
            mask = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
            
            # NO BRAIN MASK - completely removed
            brain_mask = None
        else:
            # Registration enabled - use original registration code
            if idx == 0:
                print(f"[Dataset] Starting registration...", flush=True)
            # Register fMRI to T1w space using on-the-fly registration
            # Register directly to T1w space (masks are already in T1 space)
            try:
                from nibabel.processing import resample_from_to
                
                # Keep T1w fixed (no resampling); use original space as reference
                t1w_orig_img = nib.Nifti1Image(t1w_data_orig, t1w_affine)
                t1w_data = t1w_data_orig
                t1w = torch.from_numpy(t1w_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
                t1w_affine_resampled = t1w_affine  # unchanged
                
                # Reference image is the original T1 (fixed)
                t1w_ref_img = t1w_orig_img
                
                # Register full 4D fMRI to T1 space (batched) for alignment used in loss
                fmri_img = nib.Nifti1Image(fmri_data_orig, fmri_affine)
                fmri = register_fmri_to_t1w_nib(
                    fmri_img,
                    t1w_ref_img,
                    batch_temporal=10,
                    order=1,
                    cval=0.0,
                )  # [1, T, D, H, W]
                
                if idx == 0:
                    print(f"[Dataset] fMRI registration complete. Shape: {fmri.shape}", flush=True)
                
                # Register precomputed mean fMRI if available
                fmri_mean = None
                if fmri_mean_data_orig is not None:
                    if idx == 0:
                        print(f"[Dataset] Registering precomputed mean fMRI to T1w space...", flush=True)
                    fmri_mean_img = nib.Nifti1Image(fmri_mean_data_orig, fmri_mean_affine)
                    # Mean fMRI is 3D, so register directly (no temporal dimension)
                    fmri_mean_resampled = resample_from_to(
                        fmri_mean_img,
                        t1w_ref_img,
                        order=1,  # Linear interpolation
                        mode='constant',
                        cval=0.0,
                    )
                    fmri_mean_data = fmri_mean_resampled.get_fdata().astype(np.float32)
                    
                    # RAISE ERROR if resampled data has NaN/Inf - check immediately after resampling
                    if not np.isfinite(fmri_mean_data).all():
                        finite_mask = np.isfinite(fmri_mean_data)
                        finite_min = np.amin(fmri_mean_data[finite_mask]) if np.any(finite_mask) else np.nan
                        finite_max = np.amax(fmri_mean_data[finite_mask]) if np.any(finite_mask) else np.nan
                        num_nonfinite = (~finite_mask).sum()
                        raise RuntimeError(
                            f"NaN/Inf in fmri_mean_data AFTER resampling/registration. "
                            f"Shape: {fmri_mean_data.shape}, "
                            f"Non-finite count: {num_nonfinite}/{fmri_mean_data.size}, "
                            f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                            f"This indicates a root problem in the resampling/registration that needs fixing, not masking."
                        )
                    
                    # Convert to tensor: [D, H, W] -> [1, 1, D, H, W]
                    fmri_mean = torch.from_numpy(fmri_mean_data).unsqueeze(0).unsqueeze(0).float()
                    
                    # RAISE ERROR if tensor has NaN/Inf - check after conversion
                    if not torch.isfinite(fmri_mean).all():
                        finite = torch.isfinite(fmri_mean)
                        finite_min = torch.amin(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                        finite_max = torch.amax(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                        num_nonfinite = (~finite).sum().item()
                        raise RuntimeError(
                            f"NaN/Inf in fmri_mean tensor AFTER conversion from numpy. "
                            f"Shape: {fmri_mean.shape}, "
                            f"Non-finite count: {num_nonfinite}/{fmri_mean.numel()}, "
                            f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                            f"This indicates a root problem in the tensor conversion that needs fixing, not masking."
                        )
                    
                    # Resize to target shape if needed
                    if fmri_mean.shape[2:5] != self.target_shape:
                        fmri_mean = torch.nn.functional.interpolate(
                            fmri_mean,
                            size=self.target_shape,
                            mode="trilinear",
                            align_corners=False,
                        )
                        
                        # RAISE ERROR if interpolation produced NaN/Inf
                        if not torch.isfinite(fmri_mean).all():
                            finite = torch.isfinite(fmri_mean)
                            finite_min = torch.amin(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                            finite_max = torch.amax(fmri_mean[finite]) if finite.any() else torch.tensor(float("nan"), device=fmri_mean.device)
                            num_nonfinite = (~finite).sum().item()
                            raise RuntimeError(
                                f"NaN/Inf in fmri_mean AFTER interpolation (registration path). "
                                f"Shape: {fmri_mean.shape}, "
                                f"Non-finite count: {num_nonfinite}/{fmri_mean.numel()}, "
                                f"Finite range: [{finite_min:.6f}, {finite_max:.6f}]. "
                                f"This indicates a root problem in the interpolation that needs fixing, not masking."
                            )
                    
                    # NO NORMALIZATION - Use raw mean fMRI values as-is (PSC normalization completely removed)
                    # Keep raw mean fMRI values without any normalization, scaling, or processing
                    
                    if idx == 0:
                        print(f"[Dataset] Mean fMRI registration complete. Shape: {fmri_mean.shape}", flush=True)
                
                # Skip loading mean fMRI mask; rely on regular brain mask / fMRI-derived mask downstream
                mean_fmri_mask = None
                
                # Load mask at original resolution (used only to derive ROI masks; not returned)
                if idx == 0:
                    print(f"[Dataset] Loading mask from {mask_path}...", flush=True)
                mask_img = nib.load(str(mask_path))
                mask_data_orig = mask_img.get_fdata().astype(np.float32)
                mask_affine = mask_img.affine.copy()
                
                # Resample mask to T1w space (nearest neighbor to preserve labels)
                mask_orig_img = nib.Nifti1Image(mask_data_orig, mask_affine)
                mask_resampled = resample_from_to(
                    mask_orig_img,
                    t1w_ref_img,  # Use T1w as reference (same as fMRI)
                    order=0,  # Nearest neighbor for masks to preserve label values
                    mode='constant',
                    cval=0.0,
                )
                mask_data = mask_resampled.get_fdata().astype(np.float32)
                # Keep multilabel mask for ROI masks (before binarization)
                multilabel_mask = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
                
                # Create ROI masks from multilabel mask
                roi_masks = {}
                for slug, label_id in self.ROI_SPECS:
                    roi_idx = self.structure_to_roi_idx[label_id]
                    # Create binary mask for this ROI: 1 where label_id is present, 0 otherwise
                    roi_mask = (multilabel_mask == label_id).float()  # [1, 1, D, H, W]
                    roi_masks[roi_idx] = roi_mask.squeeze(0).squeeze(0)  # [D, H, W] - remove batch and channel dims
                
                if idx == 0:
                    print(f"[Dataset] Created {len(roi_masks)} ROI masks (keys: {list(roi_masks.keys())}) in registration path", flush=True)
                
                # Binarize mask for general use: any value > 0 is brain, 0 is non-brain (not returned)
                mask_data = (mask_data > 0).astype(np.float32)
                mask = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
            
                # NO BRAIN MASK - completely removed
                brain_mask = None
            
            except ImportError:
                # Fallback to original method if nibabel.processing not available
                from utils.registration import register_fmri_to_t1w
                # Convert numpy arrays to torch tensors for registration
                # fmri_data_orig: [T, D, H, W] or [D, H, W] -> [1, T, D, H, W]
                fmri_tensor = torch.from_numpy(fmri_data_orig).float()
                if fmri_tensor.ndim == 3:
                    fmri_tensor = fmri_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                elif fmri_tensor.ndim == 4:
                    fmri_tensor = fmri_tensor.unsqueeze(0)  # [1, T, D, H, W]
                
                # t1w_data_orig: [D, H, W] -> [1, 1, D, H, W]
                t1w_tensor = torch.from_numpy(t1w_data_orig).float().unsqueeze(0).unsqueeze(0)
                
                fmri = register_fmri_to_t1w(
                    fmri_tensor,
                    t1w_tensor,
                    fmri_affine,
                    t1w_affine,
                    target_shape=self.target_shape,
                    method='affine',
                )  # [1, T, D, H, W] registered to T1w space
                # Load mask normally if not using mean fMRI registration
                mask, _ = self._load_nifti(mask_path, is_fmri=False, is_mask=True)  # [1, 1, D, H, W]
                # Keep multilabel mask for ROI masks (before binarization)
                multilabel_mask = mask.clone()
                
                # Create ROI masks from multilabel mask
                roi_masks = {}
                for slug, label_id in self.ROI_SPECS:
                    roi_idx = self.structure_to_roi_idx[label_id]
                    # Create binary mask for this ROI: 1 where label_id is present, 0 otherwise
                    roi_mask = (multilabel_mask == label_id).float()  # [1, 1, D, H, W]
                    roi_masks[roi_idx] = roi_mask.squeeze(0).squeeze(0)  # [D, H, W] - remove batch and channel dims
                
                if idx == 0:
                    print(f"[Dataset] Created {len(roi_masks)} ROI masks (keys: {list(roi_masks.keys())}) in fallback path", flush=True)
                
                # NO BRAIN MASK - completely removed
                brain_mask = None
            
            # Mean fMRI not available in fallback path
            fmri_mean = None
            mean_fmri_mask = None
        
        # NO BRAIN MASK APPLICATION - completely removed
        # fMRI and mean fMRI are used as-is without any masking

        # Ensure mask is binary (0 or 1) for ROI derivation only
        mask = (mask > 0).float()
        
        # NO NORMALIZATION - Use raw fMRI values as-is (PSC normalization completely removed)
        # Keep raw fMRI values without any normalization, scaling, or processing
        # Just ensure consistent format [1, T, D, H, W] for model compatibility
        
        # Ensure fmri is in consistent format [1, T, D, H, W] for model compatibility
        # When registration is disabled, fmri might be [T, D, H, W] (4D)
        if fmri.ndim == 4:
            # [T, D, H, W] -> [1, T, D, H, W]
            fmri = fmri.unsqueeze(0)
        
        # Load precomputed graphs
        if idx == 0:
            print(f"[Dataset] Loading precomputed graphs...", flush=True)
        graphs_dir = self.precomputed_dir / "graphs"
        hypergraphs_dir_old = self.precomputed_dir / "hypergraphs"  # For metadata/patch_distributions
        
        # Load embeddings only (adjacency graphs will be built from scratch)
        image_nodes = self._load_tensor(graphs_dir / f"{scan_id}_image_nodes.npy")
        mask_nodes = self._load_tensor(graphs_dir / f"{scan_id}_mask_nodes.npy")
        roi_nodes = self._load_tensor(graphs_dir / f"{scan_id}_roi_nodes.npy")
        
        # Apply scalers if available (embeddings should already be scaled from preprocessing, but apply again for safety)
        if self.scaler_manager is not None:
            image_nodes = self.scaler_manager.scale_brainiac(image_nodes)
            mask_nodes = self.scaler_manager.scale_modernbert(mask_nodes)
            # ROI nodes are already scaled (radiomics + AnatCL) from preprocessing
            # Note: ROI embeddings are [num_rois, embed_dim] where embed_dim = anatcl_dim + radiomics_dim
            # The scalers are applied per-feature-type during preprocessing, so ROI nodes should already be scaled
        
        if idx == 0:
            print(f"[Dataset] Loaded embeddings (will load precomputed hypergraphs)...", flush=True)
        
        # Load metadata
        metadata_path = hypergraphs_dir_old / f"{scan_id}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'num_patches': image_nodes.shape[0]}
        
        # Load patch distributions (needed for return dict, even if using precomputed hypergraphs)
        dist_file = hypergraphs_dir_old / f"{scan_id}_patch_distributions.json"
        if dist_file.exists():
            with open(dist_file, 'r') as f:
                patch_distributions = json.load(f)
                # Convert keys back to int
                patch_distributions = [
                    {int(k): v for k, v in dist.items()}
                    for dist in patch_distributions
                ]
        else:
            raise FileNotFoundError(
                f"Patch distributions file not found for {scan_id}. "
                f"Expected: {dist_file}. "
                f"Please run preprocessing script to generate this file."
            )
        
        # Load precomputed hypergraphs (instead of building on-the-fly)
        hyperedge_index_path = self.hypergraphs_dir / f"{scan_id}_hyperedge_index.pt"
        hyperedge_weights_path = self.hypergraphs_dir / f"{scan_id}_hyperedge_weights.pt"
        
        if hyperedge_index_path.exists() and hyperedge_weights_path.exists():
            # Load precomputed hypergraphs
            if idx == 0:
                print(f"[Dataset] Loading precomputed hypergraphs from {self.hypergraphs_dir}...", flush=True)
            hyperedge_index = torch.load(hyperedge_index_path, map_location='cpu')
            hyperedge_weights = torch.load(hyperedge_weights_path, map_location='cpu')
            
            if idx == 0:
                print(f"[Dataset] Loaded precomputed hypergraph. "
                      f"hyperedges: {hyperedge_index.shape[1]}, "
                      f"weights: {hyperedge_weights.shape[0]}", flush=True)
        else:
            # Fallback: build on-the-fly if precomputed not available
            if idx == 0:
                print(f"[Dataset] WARNING: Precomputed hypergraphs not found, building on-the-fly...", flush=True)
                print(f"[Dataset] Expected: {hyperedge_index_path}", flush=True)
            
            # Construct hypergraph on-the-fly (fallback)
            # Note: patch_distributions is already loaded above
            from graphs.hypergraph import HypergraphBuilder
            
            num_patches = image_nodes.shape[0]
            num_rois = len(self.structure_to_roi_idx)
            
            hypergraph_builder = HypergraphBuilder(
                num_patches=num_patches,
                num_rois=num_rois,
            )
            
            # Build hyperedges using patch distributions
            device = torch.device('cpu')  # Build on CPU, will move to GPU in collate if needed
            hyperedge_index, hyperedge_weights = hypergraph_builder.build_hyperedges(
                patch_distributions=patch_distributions,
                structure_to_roi_idx=self.structure_to_roi_idx,
                device=device,
            )
            
            if idx == 0:
                print(f"[Dataset] Hypergraph constructed on-the-fly. "
                      f"hyperedges: {hyperedge_index.shape[1]}, "
                      f"weights: {hyperedge_weights.shape[0]}", flush=True)
        
        # Validate hyperedges
        if hyperedge_index.shape[1] == 0 or len(hyperedge_weights) == 0:
            print(f"WARNING: Empty hyperedges for scan {scan_id}. "
                  f"hyperedge_index shape: {hyperedge_index.shape}, "
                  f"hyperedge_weights shape: {hyperedge_weights.shape}.")
        
        if idx == 0:
            print(f"[Dataset] Precomputed graphs loaded.", flush=True)
        
        # Validate roi_masks before returning
        if 'roi_masks' not in locals() or roi_masks is None or len(roi_masks) == 0:
            raise RuntimeError(
                f"ROI masks are REQUIRED but not created for scan_id={scan_id}. "
                f"roi_masks in locals: {'roi_masks' in locals()}, "
                f"roi_masks value: {roi_masks if 'roi_masks' in locals() else 'N/A'}, "
                f"ROI_SPECS: {self.ROI_SPECS}, "
                f"structure_to_roi_idx: {self.structure_to_roi_idx}"
            )
        
        result = {
            "t1w": t1w,
            "fmri": fmri,
            "fmri_mean": fmri_mean,  # Precomputed mean fMRI [1, 1, D, H, W] or None
            # NO BRAIN MASK - completely removed
            "t1w_affine": torch.from_numpy(t1w_affine_resampled).float(),  # affine used for alignment/visualization
            "roi_masks": roi_masks,  # Multilabel ROI masks for histogram matching
            "image_patch_embeddings": image_nodes,  # Already computed
            "mask_patch_embeddings": mask_nodes,  # Already computed
            "roi_embeddings": roi_nodes,  # Already computed
            "patch_distributions": patch_distributions,
            "structure_to_roi_idx": self.structure_to_roi_idx,
            "dwi_matrix": self.dwi_matrix,
            "scan_id": scan_id,
            # Precomputed embeddings (adjacency graphs will be built from scratch)
            "image_nodes": image_nodes,
            "mask_nodes": mask_nodes,
            "roi_nodes": roi_nodes,
            "hyperedge_index": hyperedge_index,
            "hyperedge_weights": hyperedge_weights,
        }
        
        # Print completion for first few samples or every 100th to track progress
        if idx < 3 or idx % 100 == 0:
            print(f"[Dataset RANK={rank}] Completed loading sample {idx} (scan_id: {scan_id})", flush=True)
            print(f"[Dataset RANK={rank}] Result keys: {list(result.keys())}, roi_masks in result: {'roi_masks' in result}, roi_masks type: {type(result.get('roi_masks', None))}, roi_masks keys: {list(result.get('roi_masks', {}).keys()) if isinstance(result.get('roi_masks', None), dict) else 'N/A'}", flush=True)
        
        return result

