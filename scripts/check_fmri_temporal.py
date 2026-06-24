#!/usr/bin/env python3
"""
Script to check if fMRI files have temporal dimension and BOLD information.
Scans a directory and reports dimensions, temporal length, and basic statistics.
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def analyze_fmri_file(fmri_path: Path) -> Dict:
    """Analyze a single fMRI file and return statistics."""
    try:
        img = nib.load(str(fmri_path))
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        result = {
            'path': str(fmri_path),
            'name': fmri_path.name,
            'shape': data.shape,
            'ndim': data.ndim,
            'dtype': str(data.dtype),
            'has_temporal': data.ndim == 4,
            'temporal_length': data.shape[3] if data.ndim == 4 else None,
            'spatial_shape': data.shape[:3] if data.ndim >= 3 else None,
            'affine': affine,
            'header_info': {
                'xyzt_units': header.get_xyzt_units(),
                'pixdim': header.get('pixdim', None),
                'qform_code': header.get('qform_code', None),
                'sform_code': header.get('sform_code', None),
            },
            'stats': {
                'mean': float(np.nanmean(data)),
                'std': float(np.nanstd(data)),
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data)),
                'non_zero_voxels': int(np.count_nonzero(data)),
                'total_voxels': int(data.size),
            },
            'error': None,
        }
        
        # Check if it's BOLD-like (should have reasonable dynamic range)
        if data.ndim == 4:
            # Compute temporal mean and std
            temporal_mean = data.mean(axis=3)
            temporal_std = data.std(axis=3)
            result['bold_info'] = {
                'temporal_mean_range': (float(temporal_mean.min()), float(temporal_mean.max())),
                'temporal_std_range': (float(temporal_std.min()), float(temporal_std.max())),
                'mean_temporal_std': float(temporal_std.mean()),
            }
        else:
            result['bold_info'] = None
            
        return result
        
    except Exception as e:
        return {
            'path': str(fmri_path),
            'name': fmri_path.name,
            'error': str(e),
        }


def main():
    import sys
    if len(sys.argv) > 1:
        fmri_dir = Path(sys.argv[1])
    else:
        fmri_dir = Path("/path/to/data/fMRI")
    
    if not fmri_dir.exists():
        print(f"Error: Directory not found: {fmri_dir}")
        sys.exit(1)
    
    # Find all NIfTI files
    nifti_files = list(fmri_dir.glob("*.nii.gz")) + list(fmri_dir.glob("*.nii"))
    
    if not nifti_files:
        print(f"No NIfTI files found in {fmri_dir}")
        sys.exit(1)
    
    print(f"Found {len(nifti_files)} NIfTI files in {fmri_dir}\n")
    print("=" * 80)
    
    results = []
    for fmri_path in sorted(nifti_files):
        result = analyze_fmri_file(fmri_path)
        results.append(result)
    
    # Print summary
    print("\nSUMMARY:")
    print("=" * 80)
    
    temporal_files = [r for r in results if r.get('has_temporal', False)]
    non_temporal_files = [r for r in results if not r.get('has_temporal', False) and 'error' not in r]
    error_files = [r for r in results if 'error' in r and r['error'] is not None]
    
    print(f"\nTotal files: {len(results)}")
    print(f"  ✓ 4D (temporal) files: {len(temporal_files)}")
    print(f"  ✗ 3D (no temporal) files: {len(non_temporal_files)}")
    print(f"  ✗ Error loading: {len(error_files)}")
    
    if temporal_files:
        print(f"\n4D Files with Temporal Dimension:")
        print("-" * 80)
        for r in temporal_files:
            print(f"  {r['name']}")
            print(f"    Shape: {r['shape']} (spatial: {r['spatial_shape']}, temporal: {r['temporal_length']})")
            print(f"    Value range: [{r['stats']['min']:.2f}, {r['stats']['max']:.2f}]")
            print(f"    Mean: {r['stats']['mean']:.2f}, Std: {r['stats']['std']:.2f}")
            if r['bold_info']:
                print(f"    Temporal std range: [{r['bold_info']['temporal_std_range'][0]:.2f}, {r['bold_info']['temporal_std_range'][1]:.2f}]")
                print(f"    Mean temporal std: {r['bold_info']['mean_temporal_std']:.2f}")
            print()
    
    if non_temporal_files:
        print(f"\n3D Files (No Temporal Dimension):")
        print("-" * 80)
        for r in non_temporal_files:
            print(f"  {r['name']}")
            print(f"    Shape: {r['shape']} (spatial only)")
            print(f"    Value range: [{r['stats']['min']:.2f}, {r['stats']['max']:.2f}]")
            print()
    
    if error_files:
        print(f"\nFiles with Errors:")
        print("-" * 80)
        for r in error_files:
            print(f"  {r['name']}")
            print(f"    Error: {r['error']}")
            print()
    
    # Detailed report for each file
    print("\nDETAILED REPORT:")
    print("=" * 80)
    for r in results:
        print(f"\n{r['name']}")
        print("-" * 80)
        if 'error' in r and r['error']:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Shape: {r['shape']}")
            print(f"  Dimensions: {r['ndim']}D")
            print(f"  Has temporal dimension: {r['has_temporal']}")
            if r['has_temporal']:
                print(f"  Temporal length: {r['temporal_length']} timepoints")
            print(f"  Spatial shape: {r['spatial_shape']}")
            print(f"  Data type: {r['dtype']}")
            print(f"  Statistics:")
            print(f"    Mean: {r['stats']['mean']:.4f}")
            print(f"    Std: {r['stats']['std']:.4f}")
            print(f"    Min: {r['stats']['min']:.4f}")
            print(f"    Max: {r['stats']['max']:.4f}")
            print(f"    Non-zero voxels: {r['stats']['non_zero_voxels']:,} / {r['stats']['total_voxels']:,}")
            if r['bold_info']:
                print(f"  BOLD Information:")
                print(f"    Temporal mean range: [{r['bold_info']['temporal_mean_range'][0]:.2f}, {r['bold_info']['temporal_mean_range'][1]:.2f}]")
                print(f"    Temporal std range: [{r['bold_info']['temporal_std_range'][0]:.2f}, {r['bold_info']['temporal_std_range'][1]:.2f}]")
                print(f"    Mean temporal std: {r['bold_info']['mean_temporal_std']:.4f}")
            print(f"  Header info:")
            print(f"    XYZT units: {r['header_info']['xyzt_units']}")
            if r['header_info']['pixdim'] is not None:
                print(f"    Pixel dimensions: {r['header_info']['pixdim']}")
            print(f"    QForm code: {r['header_info']['qform_code']}")
            print(f"    SForm code: {r['header_info']['sform_code']}")


if __name__ == "__main__":
    main()

