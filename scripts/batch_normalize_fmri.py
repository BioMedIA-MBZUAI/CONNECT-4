"""
Batch script for fMRI normalization using global z-score normalization.
Processes all .nii.gz files in the source directory and saves normalized versions.
Uses parallel processing with 128 CPU cores.
"""

from nilearn.image import smooth_img, load_img
from nilearn.masking import compute_epi_mask
import nibabel as nib
import numpy as np
import os
import glob
from joblib import Parallel, delayed

# Define input and output directories
input_dir = '/path/to/data/fMRI'
output_dir = '/path/to/data/fMRI_norm'
n_jobs = 128  # Number of parallel workers

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def process_single_file(fmri_file_path, output_dir, idx, total):
    """
    Process a single fMRI file: smooth, mask, normalize, and save.
    
    Args:
        fmri_file_path: Path to input fMRI file
        output_dir: Output directory for normalized files
        idx: File index (for progress reporting)
        total: Total number of files (for progress reporting)
    
    Returns:
        tuple: (success: bool, filename: str, error_message: str or None)
    """
    filename = os.path.basename(fmri_file_path)
    output_file = os.path.join(output_dir, filename)
    
    # Skip if output file already exists
    if os.path.exists(output_file):
        return (True, filename, None, "skipped")
    
    try:
        # Load fMRI data
        img = load_img(fmri_file_path)
        data = img.get_fdata()
        
        # Step 1: Smooth
        img_smooth = smooth_img(img, fwhm=6.0)
        data = img_smooth.get_fdata()
        
        # Step 2: Brain masking
        mask = compute_epi_mask(img_smooth)
        mask_data = mask.get_fdata()
        
        # Step 3: Normalize globally (z-score across all voxels and time)
        x, y, z, t = data.shape
        
        # Global normalization: (data - global_mean) / global_std
        global_mean = data.mean()
        global_std = data.std()
        normalized_data = (data - global_mean) / (global_std + 1e-8)
        
        # Step 4: Apply mask (set background to exactly 0)
        normalized_data[mask_data == 0] = 0  # Force background to 0
        
        # Save
        normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
        nib.save(normalized_img, output_file)
        
        return (True, filename, None, "completed")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return (False, filename, error_msg, "failed")


# Find all .nii.gz files in the input directory
fmri_files = glob.glob(os.path.join(input_dir, '*.nii.gz'))
fmri_files.sort()  # Sort for consistent processing order

if not fmri_files:
    print(f"No .nii.gz files found in {input_dir}")
    exit(1)

print(f"Found {len(fmri_files)} fMRI files to process")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Using {n_jobs} parallel workers")
print("=" * 80)

# Process files in parallel
print("\nProcessing files in parallel...")
results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_single_file)(fmri_file_path, output_dir, idx, len(fmri_files))
    for idx, fmri_file_path in enumerate(fmri_files, 1)
)

# Count successes and failures
successful = 0
failed = 0
skipped = 0

print("\n" + "=" * 80)
print("Processing Results:")
print("=" * 80)

for success, filename, error_msg, status in results:
    if status == "skipped":
        skipped += 1
        print(f"  [SKIP] {filename}")
    elif success:
        successful += 1
        print(f"  [OK]   {filename}")
    else:
        failed += 1
        print(f"  [FAIL] {filename}: {error_msg}")

# Summary
print("\n" + "=" * 80)
print(f"Processing complete!")
print(f"  Successful: {successful}/{len(fmri_files)}")
print(f"  Skipped:    {skipped}/{len(fmri_files)}")
print(f"  Failed:     {failed}/{len(fmri_files)}")
print(f"  Output directory: {output_dir}")
print("=" * 80)

