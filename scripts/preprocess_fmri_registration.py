#!/usr/bin/env python3
"""
Preprocessing pipeline to register all 4D fMRI files to T1w space.
This registers fMRI files once and saves them, avoiding on-the-fly registration during training.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def register_fmri_batch(
    fmri_path: Path,
    t1w_path: Path,
    output_path: Path,
    target_shape: tuple = (128, 128, 128),
    normalize: bool = True,
) -> bool:
    """
    Register a single fMRI file to T1w space and save it.
    
    Args:
        fmri_path: Path to original 4D fMRI file
        t1w_path: Path to T1w file
        output_path: Path to save registered fMRI
        target_shape: Target spatial shape (D, H, W)
        normalize: Whether to normalize intensities
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load files
        fmri_img = nib.load(str(fmri_path))
        t1w_img = nib.load(str(t1w_path))
        
        fmri_data = fmri_img.get_fdata().astype(np.float32)
        t1w_data = t1w_img.get_fdata().astype(np.float32)
        
        fmri_affine = fmri_img.affine
        t1w_affine = t1w_img.affine
        
        # Validate shapes
        if fmri_data.ndim not in [3, 4]:
            print(f"ERROR: Unexpected fMRI shape {fmri_data.shape}")
            return False
        
        # Register fMRI to T1w using nibabel's resampling (more reliable)
        from nibabel.processing import resample_from_to
        
        # Process each temporal frame
        registered_frames = []
        T = fmri_data.shape[3] if fmri_data.ndim == 4 else 1
        
        # Create reference T1w image for resampling
        t1w_ref_img = nib.Nifti1Image(t1w_data, t1w_affine)
        
        for t in range(T):
            if fmri_data.ndim == 4:
                fmri_frame = fmri_data[:, :, :, t]  # [D, H, W]
            else:
                fmri_frame = fmri_data  # [D, H, W]
            
            # Create temporary NIfTI image for this frame
            fmri_frame_img = nib.Nifti1Image(fmri_frame, fmri_affine)
            
            # Resample using nibabel (handles coordinate systems correctly)
            # resample_from_to expects: (source_image, (target_shape, target_affine))
            resampled_img = resample_from_to(
                fmri_frame_img,
                (target_shape, t1w_affine),
                order=1,  # Linear interpolation
                mode='constant',
                cval=0.0,
            )
            
            registered_frame = resampled_img.get_fdata()
            registered_frames.append(registered_frame)
        
        # Stack temporal frames: [T, D, H, W]
        registered_fmri = np.stack(registered_frames, axis=0)
        
        # Normalize if requested (after registration)
        if normalize:
            registered_fmri_mean = registered_fmri.mean()
            registered_fmri_std = registered_fmri.std()
            if registered_fmri_std > 0:
                registered_fmri = (registered_fmri - registered_fmri_mean) / (registered_fmri_std + 1e-6)
        
        # Convert to NIfTI format: [T, D, H, W] -> [D, H, W, T]
        registered_data = registered_fmri.transpose(1, 2, 3, 0)  # [D, H, W, T] for NIfTI
        
        # Create NIfTI image with T1w affine (so it's in T1w space)
        registered_img = nib.Nifti1Image(registered_data, t1w_affine)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        nib.save(registered_img, str(output_path))
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {fmri_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess fMRI files: register to T1w space")
    parser.add_argument("--root-dir", type=Path, required=True, help="Root directory with T1/, fMRI/, Masks/ subdirectories")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for registered fMRI (default: root_dir/fMRI_registered)")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128], help="Target spatial shape [D, H, W]")
    parser.add_argument("--normalize", action="store_true", help="Normalize intensities")
    parser.add_argument("--t1-suffix", type=str, default="_T1.nii.gz", help="T1 file suffix")
    parser.add_argument("--fmri-suffix", type=str, default="_fMRI.nii.gz", help="fMRI file suffix")
    args = parser.parse_args()
    
    root_dir = args.root_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root_dir / "fMRI_registered"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t1_dir = root_dir / "T1"
    fmri_dir = root_dir / "fMRI"
    
    if not t1_dir.exists():
        print(f"ERROR: T1 directory not found: {t1_dir}")
        sys.exit(1)
    if not fmri_dir.exists():
        print(f"ERROR: fMRI directory not found: {fmri_dir}")
        sys.exit(1)
    
    # Find all fMRI files
    fmri_files = sorted(fmri_dir.glob(f"*{args.fmri_suffix}"))
    
    if not fmri_files:
        print(f"ERROR: No fMRI files found in {fmri_dir}")
        sys.exit(1)
    
    print(f"Found {len(fmri_files)} fMRI files to process")
    print(f"Output directory: {output_dir}")
    print(f"Target shape: {args.target_shape}")
    
    # Process each file
    success_count = 0
    failed = []
    
    for fmri_path in tqdm(fmri_files, desc="Registering fMRI files"):
        # Extract scan ID
        scan_id = fmri_path.name.replace(args.fmri_suffix, "")
        
        # Find corresponding T1 file
        t1_path = t1_dir / f"{scan_id}{args.t1_suffix}"
        
        if not t1_path.exists():
            print(f"WARNING: T1 file not found for {scan_id}, skipping")
            failed.append(scan_id)
            continue
        
        # Output path
        output_path = output_dir / fmri_path.name
        
        # Skip if already exists
        if output_path.exists():
            print(f"Skipping {scan_id} (already exists)")
            success_count += 1
            continue
        
        # Register
        if register_fmri_batch(
            fmri_path,
            t1_path,
            output_path,
            target_shape=tuple(args.target_shape),
            normalize=args.normalize,
        ):
            success_count += 1
        else:
            failed.append(scan_id)
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Successfully processed: {success_count}/{len(fmri_files)}")
    if failed:
        print(f"Failed: {len(failed)} files")
        print(f"Failed scan IDs: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    print(f"Registered fMRI files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

