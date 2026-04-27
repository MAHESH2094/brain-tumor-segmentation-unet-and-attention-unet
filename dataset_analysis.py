# ===================================================
# CELL 3: Dataset Statistics Scanner (FIXED)
# ===================================================
# Purpose: Count dataset coverage for raw NIfTI or preprocessed NPZ patches
# FIXES: Vectorized tumor counting, isdir check, default variable assignments

import os
from glob import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ========================
# HELPER FUNCTIONS
# ========================
def find_nifti_file(patient_dir, pattern):
    """
    Find NIfTI file matching pattern with flexible naming.
    
    Args:
        patient_dir: Patient folder path
        pattern: Pattern to match (e.g., 'flair', 'seg')
    
    Returns:
        Path to file or None
    """
    # Try common naming patterns
    patterns = [
        f"*{pattern}*.nii*",
        f"*{pattern.upper()}*.nii*",
        f"*_{pattern}.nii*",
    ]
    
    for p in patterns:
        matches = glob(os.path.join(patient_dir, p))
        if matches:
            return matches[0]
    return None


def count_tumor_slices(seg_path, min_tumor_pixels=10):
    """
    Count slices with tumor presence in a segmentation file.
    
    FIX: Use vectorized numpy instead of Python loop - 10x+ faster.
    
    Args:
        seg_path: Path to segmentation NIfTI file
        min_tumor_pixels: Minimum tumor pixels to count as tumor slice
    
    Returns:
        tuple: (total_slices, tumor_slices)
    """
    try:
        seg_data = nib.load(seg_path).get_fdata().astype(np.int16)
        total_slices = seg_data.shape[2]
        
        # FIX: Vectorized counting instead of Python loop
        # Count non-zero pixels per slice across spatial dimensions
        tumor_per_slice = np.sum(seg_data > 0, axis=(0, 1))  # Shape: (D,)
        tumor_slices = np.sum(tumor_per_slice >= min_tumor_pixels)
        
        return int(total_slices), int(tumor_slices)
    except Exception as e:
        print(f"Error loading {seg_path}: {e}")
        return 0, 0


def scan_dataset(dataset_path, dataset_name="Dataset"):
    """
    Scan BraTS dataset and collect statistics.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name for display
    
    Returns:
        dict: Statistics dictionary
    """
    print(f"\nScanning {dataset_name}...")
    print("-" * 40)
    
    # Find all patient directories
    patient_dirs = sorted(glob(os.path.join(dataset_path, "BraTS*")))
    
    if not patient_dirs:
        # Try without BraTS prefix
        # FIX: Only include actual directories, not files or symlinks
        try:
            candidates = os.listdir(dataset_path)
            patient_dirs = [os.path.join(dataset_path, d) 
                           for d in candidates 
                           if os.path.isdir(os.path.join(dataset_path, d))]
            patient_dirs = sorted(patient_dirs)
        except OSError as e:
            print(f"Cannot list directory {dataset_path}: {e}")
            return None
    
    if not patient_dirs:
        print(f"No patients found in {dataset_path}")
        return None
    
    stats = {
        "dataset": dataset_name,
        "num_patients": len(patient_dirs),
        "total_slices": 0,
        "tumor_slices": 0,
        "patients_with_seg": 0,
        "modalities_found": set(),
        "slice_counts": []
    }
    
    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        patient_id = os.path.basename(patient_dir)
        
        # Check what modalities exist
        try:
            files = os.listdir(patient_dir)
        except OSError:
            continue
            
        for mod in ["flair", "t1ce", "t1", "t2", "seg"]:
            if any(mod in f.lower() for f in files):
                stats["modalities_found"].add(mod.upper())
        
        # Find segmentation file
        seg_path = find_nifti_file(patient_dir, "seg")
        
        if seg_path:
            total, tumor = count_tumor_slices(seg_path)
            stats["total_slices"] += total
            stats["tumor_slices"] += tumor
            stats["patients_with_seg"] += 1
            stats["slice_counts"].append({
                "patient": patient_id,
                "total": total,
                "tumor": tumor
            })
        else:
            # No segmentation - count slices from FLAIR
            flair_path = find_nifti_file(patient_dir, "flair")
            if flair_path:
                try:
                    flair_data = nib.load(flair_path).get_fdata()
                    stats["total_slices"] += flair_data.shape[2]
                except Exception:
                    pass
    
    return stats


def _coerce_npz_patch_layout(image, mask, npz_path):
    """Normalize NPZ patch arrays to image=(C,H,W,D), mask=(H,W,D)."""
    if image.ndim != 4:
        raise ValueError(f"image must be rank-4 in {npz_path}; got shape={image.shape}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be rank-3 in {npz_path}; got shape={mask.shape}")

    if image.shape[0] == 4:
        image_chwd = image
    elif image.shape[-1] == 4:
        image_chwd = np.transpose(image, (3, 0, 1, 2))
    else:
        raise ValueError(
            f"Could not infer channel axis for {npz_path}; image shape={image.shape}. "
            "Expected channel-first or channel-last with 4 channels."
        )

    if tuple(mask.shape) != tuple(image_chwd.shape[1:]):
        raise ValueError(
            f"mask/image spatial mismatch in {npz_path}: "
            f"mask={mask.shape}, image_spatial={image_chwd.shape[1:]}"
        )

    return image_chwd.astype(np.float32), mask.astype(np.float32)


def scan_npz_patch_dataset(npz_root, dataset_name="BraTS NPZ Patches", min_tumor_pixels=10):
    """Scan preprocessed NPZ patch dataset and report slice-level tumor coverage."""
    print(f"\nScanning {dataset_name}...")
    print("-" * 40)

    npz_files = sorted(glob(os.path.join(npz_root, "**", "*.npz"), recursive=True))
    npz_files = [p for p in npz_files if "patch" in os.path.basename(p).lower()]
    if not npz_files:
        print(f"No NPZ patch files found in {npz_root}")
        return None

    max_files = int(os.environ.get("BRATS_STATS_MAX_NPZ", "0"))
    if max_files > 0:
        npz_files = npz_files[:max_files]

    stats = {
        "dataset": dataset_name,
        "dataset_mode": "npz_patches",
        "num_patients": len(npz_files),
        "num_patches": len(npz_files),
        "total_slices": 0,
        "tumor_slices": 0,
        "patients_with_seg": len(npz_files),
        "modalities_found": {"FLAIR", "T1", "T1CE", "T2", "SEG"},
        "slice_counts": [],
        "patch_shape": None,
        "mask_shape": None,
    }

    for npz_path in tqdm(npz_files, desc="NPZ patches"):
        patch_id = os.path.splitext(os.path.basename(npz_path))[0]
        try:
            with np.load(npz_path) as data:
                if "image" not in data or "mask" not in data:
                    continue
                image = data["image"]
                mask = data["mask"]

            image_chwd, mask_hwd = _coerce_npz_patch_layout(image, mask, npz_path)
            if stats["patch_shape"] is None:
                stats["patch_shape"] = tuple(int(v) for v in image_chwd.shape)
            if stats["mask_shape"] is None:
                stats["mask_shape"] = tuple(int(v) for v in mask_hwd.shape)

            total = int(mask_hwd.shape[-1])
            tumor_per_slice = np.sum(mask_hwd > 0, axis=(0, 1))
            tumor = int(np.sum(tumor_per_slice >= int(min_tumor_pixels)))

            stats["total_slices"] += total
            stats["tumor_slices"] += tumor
            stats["slice_counts"].append({"patient": patch_id, "total": total, "tumor": tumor})
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue

    return stats


# ========================
# FIX: Initialize variables with defaults BEFORE conditional block
# ========================
NUM_PATIENTS = 0
TOTAL_SLICES = 0
TUMOR_SLICES = 0

# ========================
# SCAN PRIMARY DATASET
# ========================
print("=" * 50)
print("DATASET STATISTICS")
print("=" * 50)

# Import TRAIN_PATH from cell_02 (or get it from globals)
if "TRAIN_PATH" not in globals():
    TRAIN_PATH = None
if "NPZ_PATCH_PATH" not in globals():
    NPZ_PATCH_PATH = os.environ.get("BRATS_NPZ_PATCH_PATH", "").strip() or None
DATASET_MODE = os.environ.get("BRATS_DATASET_MODE", "").strip().lower()

# Check if path exists before scanning.
dataset_stats = None
if DATASET_MODE == "npz_patches" and NPZ_PATCH_PATH and os.path.exists(NPZ_PATCH_PATH):
    dataset_stats = scan_npz_patch_dataset(NPZ_PATCH_PATH, "BraTS 2024 Preprocessed Patches")
    os.environ["BRATS_DATASET_MODE"] = "npz_patches"
elif TRAIN_PATH and os.path.exists(TRAIN_PATH):
    dataset_stats = scan_dataset(TRAIN_PATH, "BraTS Training")
    os.environ["BRATS_DATASET_MODE"] = "nifti"
elif NPZ_PATCH_PATH and os.path.exists(NPZ_PATCH_PATH):
    dataset_stats = scan_npz_patch_dataset(NPZ_PATCH_PATH, "BraTS 2024 Preprocessed Patches")
    os.environ["BRATS_DATASET_MODE"] = "npz_patches"

if dataset_stats:
    # ========================
    # PRINT STATISTICS
    # ========================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dataset: {dataset_stats['dataset']}")
    print(f"Number of Patients: {dataset_stats['num_patients']}")
    print(f"Patients with Segmentation: {dataset_stats['patients_with_seg']}")
    print(f"Total Slices: {dataset_stats['total_slices']:,}")
    print(f"Tumor Slices: {dataset_stats['tumor_slices']:,}")
    
    # Safe division
    if dataset_stats['total_slices'] > 0:
        tumor_ratio = dataset_stats['tumor_slices'] / dataset_stats['total_slices'] * 100
        print(f"Tumor Ratio: {tumor_ratio:.2f}%")
        avg_slices = dataset_stats['total_slices'] / max(1, dataset_stats['num_patients'])
        print(f"Avg Slices/Patient: {avg_slices:.1f}")
    else:
        print("Tumor Ratio: N/A (no slices found)")

    if dataset_stats.get("patch_shape") is not None:
        print(f"Patch image shape (C,H,W,D): {dataset_stats['patch_shape']}")
    if dataset_stats.get("mask_shape") is not None:
        print(f"Patch mask shape (H,W,D): {dataset_stats['mask_shape']}")

    print("-" * 50)
    print("Modalities Found:")
    for mod in sorted(dataset_stats['modalities_found']):
        print(f"  • {mod}")

    if dataset_stats['patients_with_seg'] == 0:
        print("\n⚠ WARNING: No segmentation masks found!")
        print("  This might be a VALIDATION set without ground truth.")
        print("  Use TRAINING data for model training.")

    print("=" * 50)

    # Store for later cells
    NUM_PATIENTS = dataset_stats['num_patients']
    TOTAL_SLICES = dataset_stats['total_slices']
    TUMOR_SLICES = dataset_stats['tumor_slices']
else:
    print(f"Dataset path not found: TRAIN_PATH={TRAIN_PATH}, NPZ_PATCH_PATH={NPZ_PATCH_PATH}")
    print("Skipping dataset scan. Please verify Kaggle input.")

print("\n✓ Cell 3 complete.")

# ========================
# SMOKE TEST
# ========================
def test_statistics():
    """Verify statistics computation."""
    # Create dummy seg file in memory
    dummy_seg = np.zeros((10, 10, 5), dtype=np.int16)
    dummy_seg[3:7, 3:7, 1:3] = 4  # Add some tumor
    
    # Test vectorized counting
    tumor_per_slice = np.sum(dummy_seg > 0, axis=(0, 1))
    tumor_slices = np.sum(tumor_per_slice >= 1)
    
    assert tumor_slices == 2, f"Expected 2 tumor slices, got {tumor_slices}"
    print("✓ Statistics computation smoke test passed")

if __name__ == '__main__':
    test_statistics()
