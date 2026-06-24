"""
Dataset for CONNECT_4_V2
Loads T1w, mask, fMRI, and constructs all required features.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
try:
    # Optional: only needed for on-the-fly AnatCL ROI extraction.
    # The precomputed pipeline loads AnatCL embeddings from .pth files on disk.
    from anatcl import AnatCL
except ImportError:
    AnatCL = None

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.patchify import Patchify3D
from models.brainiac_wrapper import BrainIACWrapper
from models.modernbert_wrapper import ModernBERTWrapper
from utils.registration import register_fmri_to_t1w
from utils.scalers import FeatureScalerManager
try:
    from utils.registration_optimized import register_fmri_to_t1w_optimized
    USE_OPTIMIZED_REGISTRATION = True
except ImportError:
    USE_OPTIMIZED_REGISTRATION = False


class Connect4Dataset(Dataset):
    """
    Dataset for CONNECT_4_V2 pipeline.
    """
    
    # ROI specifications (structure ID to name mapping)
    ROI_SPECS: List[Tuple[str, int]] = [
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
    SLUG_TO_ID = {slug: label_id for slug, label_id in ROI_SPECS}
    ID_TO_SLUG = {label_id: slug for slug, label_id in ROI_SPECS}
    ROI_LABEL_TO_INDEX = {
        0: -1,  # Background
        **{label_id: idx for idx, (_, label_id) in enumerate(ROI_SPECS)},
    }
    
    def __init__(
        self,
        root_dir: str,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        dwi_matrix_path: str = "/path/to/data/dwi_matrix.csv",
        normative_csv_path: str = "/path/to/data/patient_roi_descriptions_with_norms.csv",
        normalize_intensity: bool = True,
        brainiac_model_path: Optional[str] = None,
        modernbert_model_path: Optional[str] = None,
        register_fmri_to_t1w: bool = True,  # Whether to register fMRI to T1w space
        apply_brain_mask: bool = True,  # Whether to apply brain mask (set to False if masking causes issues)
        fmri_scale_factor: float = 1.05,  # Scale factor to adjust fMRI size (1.0 = no scaling, >1.0 = larger, <1.0 = smaller)
        scaler_dir: Optional[str] = None,  # Directory containing saved scalers
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.patch_size = patch_size
        self.target_shape = target_shape
        self.normalize_intensity = normalize_intensity
        self.register_fmri_to_t1w = register_fmri_to_t1w
        self.apply_brain_mask = apply_brain_mask
        self.fmri_scale_factor = fmri_scale_factor
        
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
        
        # Initialize patchifier and feature extractors
        self.patchifier = Patchify3D(patch_size)
        self.brainiac = BrainIACWrapper(
            model_path=brainiac_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.modernbert = ModernBERTWrapper(model_path=modernbert_model_path)

        # Load DWI matrix
        self.dwi_matrix = self._load_dwi_matrix(dwi_matrix_path)

        # Load normative ROI descriptions (per patient, per structure)
        self.normative_descriptions = self._load_normative_descriptions(normative_csv_path)

        # Load radiomics features
        self.radiomics_features = self._load_radiomics_table()

        # Discover scan IDs
        self.scan_ids = self._discover_scan_ids()
        if not self.scan_ids:
            raise RuntimeError(f"No scan IDs found under {self.root}")

        # Load AnatCL embeddings cache
        self.anatcl_cache = {}

    def _load_normative_descriptions(self, csv_path: str) -> Dict[Tuple[str, str], str]:
        """
        Load normative ROI descriptions per patient and structure.
        
        Priority:
          1) Use `full_description` column if present and non-empty.
          2) Else fallback to `normative_description` / `radiomics_description`.
          3) Else build from NormPredicted/NormLower/NormUpper if finite.
        
        Notes:
          - Rows with 'nan' ranges (e.g., "range nan–nan") are dropped.
          - Index by (base_patient_id, Structure) where base_patient_id is
            the part of PatientID before any '_' suffix.
        """
        path = Path(csv_path)
        if not path.exists():
            return {}

        df = pd.read_csv(path)
        required_cols = {"PatientID", "Structure"}
        missing = required_cols - set(df.columns)
        if missing:
            return {}

        # Create base_id for robust matching (e.g., 'B32059094_999' -> 'B32059094')
        df["base_id"] = df["PatientID"].astype(str).str.split("_").str[0]

        full_col = "full_description" if "full_description" in df.columns else None
        normative_col = "normative_description" if "normative_description" in df.columns else None
        radiomics_col = "radiomics_description" if "radiomics_description" in df.columns else None

        use_norm_numbers = all(col in df.columns for col in ["NormPredicted", "NormLower", "NormUpper"])

        table: Dict[Tuple[str, str], str] = {}

        def clean_text(txt: str) -> str:
            """Remove nan ranges and trim whitespace."""
            if not isinstance(txt, str):
                return ""
            t = txt.strip()
            if not t:
                return ""
            # Drop phrases containing 'nan'
            if "nan" in t.lower():
                return ""
            return t

        for _, row in df.iterrows():
            base_id = str(row["base_id"])
            structure = str(row["Structure"])

            parts: List[str] = []

            # 1) Prefer full_description
            if full_col:
                txt = clean_text(row.get(full_col, ""))
                if txt:
                    parts.append(txt)

            # 2) Fallback to normative/radiomics
            if not parts and normative_col:
                txt = clean_text(row.get(normative_col, ""))
                if txt:
                    parts.append(txt)
            if not parts and radiomics_col:
                txt = clean_text(row.get(radiomics_col, ""))
                if txt:
                    parts.append(txt)

            # 3) Fallback to numeric range if available and finite
            if not parts and use_norm_numbers:
                try:
                    pred = float(row["NormPredicted"])
                    low = float(row["NormLower"])
                    high = float(row["NormUpper"])
                    if np.isfinite(pred) and np.isfinite(low) and np.isfinite(high):
                        parts.append(f"Normative volume {pred:.2f} (range {low:.2f}–{high:.2f}).")
                except Exception:
                    pass

            if not parts:
                continue

            key = (base_id, structure)
            table[key] = " ".join(parts)

        return table
    
    def _discover_scan_ids(self) -> List[str]:
        """Discover scan IDs from T1 directory."""
        t1_dir = self.root / "T1"
        scan_ids = []
        if not t1_dir.exists():
            return scan_ids
        
        patterns = ["*_T1.nii.gz", "*_T1.nii"]
        for pattern in patterns:
            for file in t1_dir.glob(pattern):
                name = file.name
                if name.endswith(".nii.gz"):
                    base = name[:-len(".nii.gz")]
                elif name.endswith(".nii"):
                    base = name[:-len(".nii")]
                else:
                    base = file.stem
                scan_ids.append(base.replace("_T1", ""))
        
        return sorted(set(scan_ids))
    
    def _load_dwi_matrix(self, path: str) -> torch.Tensor:
        """Load DWI connectivity matrix."""
        dwi_path = Path(path)
        if not dwi_path.exists():
            raise FileNotFoundError(f"DWI matrix not found at {dwi_path}")
        
        adj_np = np.loadtxt(dwi_path, delimiter=",", dtype=np.float32)
        adj_tensor = torch.from_numpy(adj_np)
        
        if adj_tensor.shape != (self.NUM_ROIS, self.NUM_ROIS):
            raise ValueError(
                f"DWI matrix must be {self.NUM_ROIS}x{self.NUM_ROIS}, got {adj_tensor.shape}"
            )
        
        return adj_tensor
    
    def _load_radiomics_table(self) -> Dict[int, torch.Tensor]:
        """Load radiomics features table."""
        csv_path = self.root / "all_radiomics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Radiomics CSV not found at {csv_path}. "
                f"This file is required for ROI graph construction."
            )
        
        df = pd.read_csv(csv_path)
        
        # Handle both formats: structure_id (numeric) or Structure (name)
        if "structure_id" in df.columns:
            # Format: scan_id, structure_id, features...
            df = df.sort_values("structure_id").drop_duplicates("structure_id", keep="first")
            keep_cols = [c for c in df.columns if c not in ("scan_id", "structure_id", "PatientID")]
            id_col = "structure_id"
        elif "Structure" in df.columns:
            # Format: PatientID, Structure, features...
            # Map structure names to IDs
            structure_name_to_id = {slug: label_id for slug, label_id in self.ROI_SPECS}
            df["structure_id"] = df["Structure"].map(structure_name_to_id)
            df = df.dropna(subset=["structure_id"])  # Remove unmapped structures
            df["structure_id"] = df["structure_id"].astype(int)
            df = df.sort_values("structure_id").drop_duplicates("structure_id", keep="first")
            keep_cols = [c for c in df.columns if c not in ("scan_id", "structure_id", "PatientID", "Structure")]
            id_col = "structure_id"
        else:
            raise ValueError(
                f"Radiomics CSV must have either 'structure_id' or 'Structure' column. "
                f"Found columns: {list(df.columns)}"
            )
        
        features = {}
        for _, row in df.iterrows():
            sid = int(row[id_col])
            feats = torch.tensor(row[keep_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
            feats = torch.nan_to_num(feats)
            features[sid] = feats
        
        return features
    
    def _load_nifti(self, path: Path, is_fmri: bool = False, is_mask: bool = False, 
                     skip_resample: bool = False) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load NIfTI file.
        
        Args:
            path: Path to NIfTI file
            is_fmri: Whether this is an fMRI file
            is_mask: Whether this is a mask file
            skip_resample: If True, skip resampling (useful when registration will handle it)
        
        Returns:
            tensor: Loaded tensor
            affine: Affine matrix from NIfTI header
        """
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        affine = img.affine.copy()
        
        if data.ndim == 4 and is_fmri:
            # 4D fMRI: [D, H, W, T] -> [1, T, D, H, W]
            tensor = torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0).float()
            
            # Resize spatial dimensions (skip if registration will handle it)
            if not skip_resample and tensor.shape[2:5] != self.target_shape:
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
        elif data.ndim == 3:
            # 3D: [D, H, W] -> [1, 1, D, H, W]
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
            # Use nearest neighbor for masks to preserve exact label values
            # Use trilinear for images, nearest for fMRI masks
            if is_mask:
                mode = "nearest"
                align_corners = None  # align_corners not valid for nearest mode
            elif is_fmri:
                mode = "nearest"
                align_corners = None  # align_corners not valid for nearest mode
            else:
                mode = "trilinear"
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
        
        # Round mask values to integers to preserve exact label values
        # Do NOT normalize masks - they contain discrete label values
        if is_mask:
            tensor = tensor.round().long().float()
            return tensor, affine
        
        if self.normalize_intensity:
            mean = tensor.mean()
            std = tensor.std()
            tensor = (tensor - mean) / (std + 1e-6)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        return tensor, affine
    
    def _load_anatcl_embeddings(self, scan_id: str) -> Dict[str, torch.Tensor]:
        """Load AnatCL embeddings for a scan."""
        if scan_id in self.anatcl_cache:
            return self.anatcl_cache[scan_id]
        
        folder = self.root / "AnatCL" / scan_id
        embeddings = {}
        
        if folder.exists():
            for file in folder.glob(f"{scan_id}_*.pth"):
                slug = file.stem.replace(f"{scan_id}_", "")
                emb = torch.load(file)
                emb = emb.view(-1).float()
                emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
                embeddings[slug] = emb
        
        self.anatcl_cache[scan_id] = embeddings
        return embeddings
    
    def _build_roi_embeddings(self, scan_id: str) -> torch.Tensor:
        """Build ROI embeddings from AnatCL + radiomics (with scaling applied)."""
        anatcl_embs = self._load_anatcl_embeddings(scan_id)
        
        roi_embs = []
        for slug, label_id in self.ROI_SPECS:
            if slug not in anatcl_embs:
                raise ValueError(
                    f"AnatCL embedding missing for structure '{slug}' (ID: {label_id}) "
                    f"in scan {scan_id}. Expected file: AnatCL/{scan_id}/{scan_id}_{slug}.pth"
                )
            anatcl_emb = anatcl_embs[slug]
            
            # Apply AnatCL scaler if available
            if self.scaler_manager is not None:
                anatcl_emb = self.scaler_manager.scale_anatcl(anatcl_emb)
            
            radio_feat = self.radiomics_features.get(label_id)
            if radio_feat is None:
                raise ValueError(
                    f"Radiomics features missing for structure ID {label_id} ({slug}). "
                    f"Check all_radiomics.csv"
                )
            
            # Apply radiomics scaler if available
            if self.scaler_manager is not None:
                radio_feat = self.scaler_manager.scale_radiomics(radio_feat)
            
            combined = torch.cat([anatcl_emb, radio_feat], dim=-1)
            roi_embs.append(combined)
        
        return torch.stack(roi_embs)  # [num_rois, embed_dim]
    
    def _compute_patch_distribution(
        self,
        mask: torch.Tensor,
        patch_idx: int,
    ) -> Dict[int, float]:
        """Compute structure distribution in a patch."""
        d_patches = self.target_shape[0] // self.patch_size[0]
        h_patches = self.target_shape[1] // self.patch_size[1]
        w_patches = self.target_shape[2] // self.patch_size[2]
        
        d_idx = patch_idx // (h_patches * w_patches)
        h_idx = (patch_idx // w_patches) % h_patches
        w_idx = patch_idx % w_patches
        
        d_start = d_idx * self.patch_size[0]
        d_end = min(d_start + self.patch_size[0], self.target_shape[0])
        h_start = h_idx * self.patch_size[1]
        h_end = min(h_start + self.patch_size[1], self.target_shape[1])
        w_start = w_idx * self.patch_size[2]
        w_end = min(w_start + self.patch_size[2], self.target_shape[2])
        
        patch_mask = mask[0, 0, d_start:d_end, h_start:h_end, w_start:w_end]
        total_voxels = patch_mask.numel()
        
        if total_voxels == 0:
            return {}

        unique_labels, counts = torch.unique(patch_mask, return_counts=True)
        distribution: Dict[int, float] = {}
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            label_int = int(label)
            # Include ALL labels (including background and non-ROI structures)
            distribution[label_int] = count / float(total_voxels)

        return distribution

    def _compute_patch_center_mm(
        self,
        patch_idx: int,
        orig_shape: Tuple[int, int, int],
        affine: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute the 3D center of mass of the patch in physical space (mm).

        Steps:
          1. Compute the patch center in the resampled target grid.
          2. Map that center back to the original voxel grid by linear scaling.
          3. Convert original voxel coordinates to mm via the NIfTI affine.
        """
        D_t, H_t, W_t = self.target_shape
        pd, ph, pw = self.patch_size

        d_patches = D_t // pd
        h_patches = H_t // ph
        w_patches = W_t // pw

        d_idx = patch_idx // (h_patches * w_patches)
        h_idx = (patch_idx // w_patches) % h_patches
        w_idx = patch_idx % w_patches

        # Center in target voxel coordinates
        center_d_t = d_idx * pd + pd / 2.0
        center_h_t = h_idx * ph + ph / 2.0
        center_w_t = w_idx * pw + pw / 2.0

        D_o, H_o, W_o = orig_shape
        # Map to original voxel coordinates via linear scaling
        center_d_o = center_d_t * (D_o / float(D_t))
        center_h_o = center_h_t * (H_o / float(H_t))
        center_w_o = center_w_t * (W_o / float(W_t))

        # Homogeneous coordinates for affine multiplication
        coord = np.array([center_d_o, center_h_o, center_w_o, 1.0], dtype=np.float32)
        phys = affine @ coord  # [x, y, z, 1]
        x_mm, y_mm, z_mm = phys[:3].tolist()
        return float(x_mm), float(y_mm), float(z_mm)
    
    def __len__(self) -> int:
        return len(self.scan_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        scan_id = self.scan_ids[idx]
        scan_base = scan_id.split("_")[0]
        
        # Load volumes
        t1_path = self.root / "T1" / f"{scan_id}_T1.nii.gz"
        fmri_path = self.root / "fMRI" / f"{scan_id}_fMRI.nii.gz"
        mask_path = self.root / "Masks" / f"{scan_id}_mask.nii.gz"
        
        if not all(p.exists() for p in [t1_path, fmri_path, mask_path]):
            raise FileNotFoundError(f"Missing files for {scan_id}")

        # Load original mask header (for physical-space coordinates)
        mask_img = nib.load(str(mask_path))
        orig_shape = mask_img.shape[:3]
        affine = mask_img.affine

        # Load T1w and fMRI at original resolution for proper registration
        t1w_img = nib.load(str(t1_path))
        t1w_data_orig = t1w_img.get_fdata().astype(np.float32)
        t1w_affine = t1w_img.affine.copy()
        
        fmri_img = nib.load(str(fmri_path))
        fmri_data_orig = fmri_img.get_fdata().astype(np.float32)
        fmri_affine = fmri_img.affine.copy()
        # Keep raw fMRI tensor for fallbacks: [1, T, D, H, W]
        if fmri_data_orig.ndim == 4:  # [D, H, W, T]
            fmri_raw = torch.from_numpy(fmri_data_orig).permute(3, 0, 1, 2).unsqueeze(0).float()
        elif fmri_data_orig.ndim == 3:  # [D, H, W]
            fmri_raw = torch.from_numpy(fmri_data_orig).unsqueeze(0).unsqueeze(0).float()
        else:
            raise ValueError(f"Unexpected fMRI shape: {fmri_data_orig.shape}")
        
        # Register fMRI to T1w space while preserving temporal dimension (if enabled)
        if self.register_fmri_to_t1w:
            # Register directly to T1w space (masks are already in T1 space)
            try:
                from nibabel.processing import resample_from_to
                
                # Resize T1w to target_shape using nibabel (preserves physical space correctly)
                t1w_orig_img = nib.Nifti1Image(t1w_data_orig, t1w_affine)
                
                # Use nibabel's zoom to calculate proper spacing
                # Get original spacing from header
                orig_zooms = t1w_img.header.get_zooms()[:3]  # Get spatial zooms (D, H, W)
                orig_shape = np.array(t1w_data_orig.shape)
                target_shape = np.array(self.target_shape)
                
                # Calculate zoom factors to preserve physical size
                # Apply scale adjustment if needed (reduce spacing to make fMRI appear larger)
                zoom_factors = orig_shape / target_shape
                # fmri_scale_factor > 1.0 makes fMRI appear larger (by reducing spacing)
                # fmri_scale_factor < 1.0 makes fMRI appear smaller (by increasing spacing)
                scale_adjustment = 1.0 / self.fmri_scale_factor if self.fmri_scale_factor != 1.0 else 1.0
                new_zooms = orig_zooms * zoom_factors * scale_adjustment
                
                # Create new affine that preserves physical space
                # Extract rotation and translation from original affine
                rotation = t1w_affine[:3, :3]
                translation = t1w_affine[:3, 3]
                
                # Normalize rotation matrix to get directions
                orig_spacing = np.sqrt(np.sum(rotation**2, axis=0))
                directions = rotation / (orig_spacing + 1e-10)
                
                # Create new affine with updated spacing but same directions and origin
                t1w_target_affine = t1w_affine.copy()
                for i in range(3):
                    t1w_target_affine[:3, i] = directions[:, i] * new_zooms[i]
                # Keep translation (origin) the same
                t1w_target_affine[:3, 3] = translation
                
                # Resample T1w to target shape
                t1w_resampled = resample_from_to(
                    t1w_orig_img,
                    (tuple(self.target_shape), t1w_target_affine),
                    order=1,
                    mode='constant',
                    cval=0.0,
                )
                t1w_data = t1w_resampled.get_fdata().astype(np.float32)
                t1w = torch.from_numpy(t1w_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
                t1w_affine_resampled = t1w_resampled.affine
                
                # Create reference image from resampled T1w (this is our target space)
                t1w_ref_img = nib.Nifti1Image(t1w_data, t1w_affine_resampled)
                
                # Now register each temporal frame of 4D fMRI to T1w space
                registered_frames = []
                T = fmri_data_orig.shape[3] if fmri_data_orig.ndim == 4 else 1
                batch_temporal = 10  # Process 10 frames at a time
                
                for t_start in range(0, T, batch_temporal):
                    t_end = min(t_start + batch_temporal, T)
                    batch_frames = []
                    
                    for t in range(t_start, t_end):
                        if fmri_data_orig.ndim == 4:
                            fmri_frame = fmri_data_orig[:, :, :, t]  # [D, H, W]
                        else:
                            fmri_frame = fmri_data_orig  # [D, H, W]
                        
                        # Create temporary NIfTI image for this frame
                        fmri_frame_img = nib.Nifti1Image(fmri_frame, fmri_affine)
                        
                        # Resample to T1w space
                        resampled_img = resample_from_to(
                            fmri_frame_img,
                            t1w_ref_img,  # Use T1w as reference
                            order=1,  # Linear interpolation
                            mode='constant',
                            cval=0.0,
                        )
                        
                        registered_frame = torch.from_numpy(resampled_img.get_fdata().astype(np.float32))
                        batch_frames.append(registered_frame)
                    
                    registered_frames.extend(batch_frames)
                
                # Stack: [T, D, H, W] and ensure float32 dtype
                fmri = torch.stack(registered_frames, dim=0).unsqueeze(0).float()  # [1, T, D, H, W]
                
                # Register mask to T1w space (masks are already in T1 space, just need resizing)
                # Load mask at original resolution
                mask_img = nib.load(str(mask_path))
                mask_data_orig = mask_img.get_fdata().astype(np.float32)
                mask_affine = mask_img.affine.copy()
                
                # Register mask to T1w space (same as fMRI)
                mask_orig_img = nib.Nifti1Image(mask_data_orig, mask_affine)
                mask_resampled = resample_from_to(
                    mask_orig_img,
                    t1w_ref_img,  # Use T1w as reference (same as fMRI)
                    order=0,  # Nearest neighbor for masks to preserve label values
                    mode='constant',
                    cval=0.0,
                )
                mask_data = mask_resampled.get_fdata().astype(np.float32)
                # Binarize mask: any value > 0 is brain, 0 is non-brain
                mask_data = (mask_data > 0).astype(np.float32)
                mask = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
                
            except ImportError:
                # Fallback to original method
                if USE_OPTIMIZED_REGISTRATION:
                    fmri = register_fmri_to_t1w_optimized(
                        fmri_raw,
                        t1w,
                        fmri_affine,
                        t1w_affine,
                        target_shape=self.target_shape,
                        batch_temporal=10,
                    )
                else:
                    fmri = register_fmri_to_t1w(
                        fmri_raw,
                        t1w,
                        fmri_affine,
                        t1w_affine,
                        target_shape=self.target_shape,
                        method='affine',
                    )  # [1, T, D, H, W] registered to T1w space
                    # Load mask normally if not using mean fMRI registration
                    mask, _ = self._load_nifti(mask_path, is_fmri=False, is_mask=True)  # [1, 1, D, H, W]
        else:
            # Use fMRI as-is (already resampled to target_shape in _load_nifti)
            fmri = fmri_raw
            # Load mask normally
        mask, _ = self._load_nifti(mask_path, is_fmri=False, is_mask=True)  # [1, 1, D, H, W]
        
        # Keep fMRI as is (do not apply mask to real fMRI)
        # Ensure mask is binary (0 or 1) for later use
        mask = (mask > 0).float()
        
        # Normalize fMRI (if enabled)
        # DISABLED: Normalization removed to fix visualization issues
        # if self.normalize_intensity:
        #     # Only compute mean/std from brain voxels (non-zero after masking)
        #     brain_voxels = fmri[fmri != 0]
        #     if len(brain_voxels) > 0:
        #         mean = brain_voxels.mean()
        #         std = brain_voxels.std() + 1e-6
        #         # Normalize only brain voxels, keep background at 0
        #         fmri = torch.where(fmri != 0, (fmri - mean) / std, fmri)
        #         fmri = torch.nan_to_num(fmri, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Patchify T1w
        t1w_patches = self.patchifier.patchify(t1w)  # List of [C, patch_d, patch_h, patch_w]
        num_patches = len(t1w_patches)
        
        # Get BrainIAC embeddings for patches (OPTIMIZED - batch processing)
        # NOTE: For preprocessing, this is done in batches. For training, use precomputed embeddings.
        image_patch_embeddings = []
        if num_patches > 100:
            print(f"  Encoding {num_patches} patches with BrainIAC (batch processing)...")
        
        # Process in batches for efficiency
        batch_size = 512
        with torch.no_grad():
            all_patches_tensor = torch.stack(t1w_patches)  # [num_patches, C, D, H, W]
            for batch_start in range(0, num_patches, batch_size):
                batch_end = min(batch_start + batch_size, num_patches)
                batch_tensor = all_patches_tensor[batch_start:batch_end]
                batch_embs = self.brainiac.encode(batch_tensor)  # [batch_size, embed_dim]
                image_patch_embeddings.append(batch_embs)
        
        image_patch_embeddings = torch.cat(image_patch_embeddings, dim=0)  # [num_patches, embed_dim]
        
        # Apply BrainIAC scaler if available
        if self.scaler_manager is not None:
            image_patch_embeddings = self.scaler_manager.scale_brainiac(image_patch_embeddings)
        
        # Patchify mask and compute distributions
        mask_patches = self.patchifier.patchify(mask)
        num_patches = len(mask_patches)
        
        patch_distributions: List[Dict[int, float]] = []
        patch_text_descriptions: List[str] = []

        for patch_idx in range(num_patches):
            dist = self._compute_patch_distribution(mask, patch_idx)
            patch_distributions.append(dist)

            # Compute 3D center of mass in physical space (mm)
            x_mm, y_mm, z_mm = self._compute_patch_center_mm(
                patch_idx,
                orig_shape=orig_shape,
                affine=affine,
            )

            # Create text description with center-of-mass and per-structure normative text
            desc_parts: List[str] = [
                f"Patch {patch_idx} at center ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm "
                f"contains {len(dist)} structures:"
            ]

            # Sort structures by coverage (descending)
            for struct_id, coverage in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                struct_slug = self.ID_TO_SLUG.get(struct_id, f"structure_{struct_id}")
                struct_name_readable = struct_slug.replace("_", " ")
                base_text = f"{struct_name_readable} ({coverage*100:.1f}%)."

                # Append normative description if available for this patient + structure
                norm_key = (scan_base, struct_slug)
                norm_text = self.normative_descriptions.get(norm_key, "")
                if norm_text:
                    desc_parts.append(f"{base_text} {norm_text}")
                else:
                    desc_parts.append(base_text)

            patch_text_descriptions.append(" ".join(desc_parts))
        
        # Get modernBERT embeddings for mask patches
        # NOTE: This is slow! Consider caching embeddings
        mask_patch_embeddings = []
        if len(patch_text_descriptions) > 100:
            print(f"  Encoding {len(patch_text_descriptions)} text descriptions with ModernBERT...")
        
        # Use batch encoding if available
        if hasattr(self.modernbert, 'forward'):
            # Batch process in chunks
            batch_size_text = 32
            for i in range(0, len(patch_text_descriptions), batch_size_text):
                batch_texts = patch_text_descriptions[i:i+batch_size_text]
                if len(patch_text_descriptions) > 100 and i > 0 and i % 500 == 0:
                    print(f"    ModernBERT: {i}/{len(patch_text_descriptions)} descriptions encoded...")
                batch_embs = self.modernbert.forward(batch_texts)  # [batch_size, embed_dim]
                mask_patch_embeddings.append(batch_embs)
            mask_patch_embeddings = torch.cat(mask_patch_embeddings, dim=0)  # [num_patches, embed_dim]
        else:
            # Individual encoding (slower)
            for i, desc in enumerate(patch_text_descriptions):
                if len(patch_text_descriptions) > 100 and (i + 1) % 500 == 0:
                    print(f"    ModernBERT: {i+1}/{len(patch_text_descriptions)} descriptions encoded...")
                emb = self.modernbert.encode(desc)  # [embed_dim]
                mask_patch_embeddings.append(emb)
            mask_patch_embeddings = torch.stack(mask_patch_embeddings)  # [num_patches, embed_dim]
        
        # Apply ModernBERT scaler if available
        if self.scaler_manager is not None:
            mask_patch_embeddings = self.scaler_manager.scale_modernbert(mask_patch_embeddings)
        
        # Build ROI embeddings
        roi_embeddings = self._build_roi_embeddings(scan_id)  # [num_rois, embed_dim]
        
        # Structure to ROI index mapping
        structure_to_roi_idx = {
            label_id: idx
            for idx, (_, label_id) in enumerate(self.ROI_SPECS)
        }
        
        return {
            "t1w": t1w,
            "fmri": fmri,
            "mask": mask,
            "image_patch_embeddings": image_patch_embeddings,
            "mask_patch_embeddings": mask_patch_embeddings,
            "roi_embeddings": roi_embeddings,
            "patch_distributions": patch_distributions,
            "structure_to_roi_idx": structure_to_roi_idx,
            "dwi_matrix": self.dwi_matrix,
            "scan_id": scan_id,
        }

