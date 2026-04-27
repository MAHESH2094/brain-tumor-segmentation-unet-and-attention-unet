# ===================================================
# CELL 4: Preprocessing Pipeline (MULTIMODAL + BINARY) (FIXED)
# ===================================================
# Purpose: Multimodal MRI loading, CLAHE enhancement, 
#          Z-score normalization, binary mask creation
# FIXES: CLAHE zero guard, affine/header assignment, channel assertion,
#        deprecation warning for CLAHE param, patient logging

import os
from glob import glob
import warnings
import cv2
import nibabel as nib
import numpy as np

if "NUM_CHANNELS" not in globals():
    NUM_CHANNELS = 4
if "NUM_CLASSES" not in globals():
    NUM_CLASSES = 1
if "CLASS_NAMES" not in globals():
    CLASS_NAMES = ['Tumor']
if "MODALITIES" not in globals():
    MODALITIES = ['flair', 't1', 't1ce', 't2']
if "TRAIN_PATH" not in globals():
    TRAIN_PATH = None
if "NPZ_PATCH_PATH" not in globals():
    NPZ_PATCH_PATH = os.environ.get("BRATS_NPZ_PATCH_PATH", "").strip() or None

# ========================
# CONFIGURATION
# ========================
IMG_SIZE = int(os.environ.get("BRATS_IMG_SIZE", "128"))
os.environ["BRATS_IMG_SIZE"] = str(IMG_SIZE)
MIN_TUMOR_PIXELS = 50             # Minimum tumor pixels to keep slice

# ========================
# 1. CLAHE ENHANCEMENT
# ========================
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast while limiting noise amplification.
    
    FIX: Guard against all-zero images which would cause division by zero.
    
    Args:
        image: 2D numpy array (H, W), normalized to [0, 1] or unnormalized
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        CLAHE enhanced image as float32
    """
    # Convert to uint8 for CLAHE (required by OpenCV)
    # FIX: Guard against zero max (all-black slices)
    if image.max() > 0:
        img_normalized = image / image.max()
    else:
        # All-zero slice - return as-is
        return image.astype(np.float32)
    
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    enhanced = clahe.apply(img_uint8)
    
    # Convert back to float32
    return enhanced.astype(np.float32) / 255.0


def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=0):
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
        image: 2D numpy array (H, W)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation (0 = auto-calculate)
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image.astype(np.float32), kernel_size, sigma)


# ========================
# 2. Z-SCORE NORMALIZATION
# ========================
def normalize_image(image, volume_stats=None, epsilon=1e-8):
    """
    Z-score normalization using volume statistics if provided.
    
    FIX: Supports volume-level normalization to preserve inter-slice
    intensity relationships. Falls back to slice-level if no volume_stats.
    
    Args:
        image: 2D numpy array (H, W)
        volume_stats: dict with 'mean' and 'std' for entire volume (optional)
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized image as float32
    """
    image = image.astype(np.float32)
    
    # Create brain mask (non-zero regions)
    brain_mask = image > 0
    
    if np.sum(brain_mask) == 0:
        # Empty slice - return zeros
        return np.zeros_like(image, dtype=np.float32)
    
    if volume_stats is not None:
        # Use volume-level statistics for consistent normalization
        mean_val = volume_stats['mean']
        std_val = volume_stats['std']
    else:
        # Fallback to slice-level (backward compatible)
        brain_pixels = image[brain_mask]
        mean_val = np.mean(brain_pixels)
        std_val = np.std(brain_pixels)
    
    # Normalize with epsilon to prevent division by zero
    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[brain_mask] = (image[brain_mask] - mean_val) / (std_val + epsilon)
    
    return normalized


def compute_volume_stats(volume):
    """
    Compute mean and std over the entire 3D volume (brain region only).
    
    Args:
        volume: 3D numpy array (H, W, D)
    
    Returns:
        dict with 'mean' and 'std' keys
    """
    brain_mask = volume > 0
    if np.sum(brain_mask) == 0:
        return {'mean': 0.0, 'std': 1.0}
    brain_pixels = volume[brain_mask]
    return {
        'mean': float(np.mean(brain_pixels)),
        'std': float(np.std(brain_pixels))
    }


# ========================
# 3. SPATIAL RESIZING
# ========================
def resize_image(image, target_size=IMG_SIZE, interpolation='bilinear'):
    """
    Resize image to target size.
    
    Args:
        image: 2D numpy array (H, W)
        target_size: Target dimension (square output)
        interpolation: 'bilinear' for images, 'nearest' for masks
    
    Returns:
        Resized image as float32
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D/3D image, got shape {image.shape}")
    if int(target_size) <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")
    if not np.isfinite(image).all():
        raise ValueError("Input image contains NaN/Inf values")

    h, w = image.shape[:2]
    if h == int(target_size) and w == int(target_size):
        return image.astype(np.float32)

    if interpolation == 'bilinear':
        interp_flag = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        interp_flag = cv2.INTER_NEAREST
    else:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    
    resized = cv2.resize(
        image.astype(np.float32),
        (target_size, target_size),
        interpolation=interp_flag
    )
    
    return resized.astype(np.float32)


def resize_mask(mask, target_size=IMG_SIZE):
    """
    Resize mask using NEAREST NEIGHBOR interpolation.
    Critical: Never use bilinear on masks - destroys label integrity.
    
    Args:
        mask: 2D or 3D numpy array (H, W) or (H, W, C)
        target_size: Target dimension
    
    Returns:
        Resized mask as float32
    """
    if len(mask.shape) == 3:
        # Multichannel mask - resize each channel
        resized_channels = []
        for c in range(mask.shape[-1]):
            resized_c = resize_image(mask[:, :, c], target_size, interpolation='nearest')
            resized_channels.append(resized_c)
        return np.stack(resized_channels, axis=-1).astype(np.float32)
    else:
        return resize_image(mask, target_size, interpolation='nearest')


# ========================
# 4. BINARY MASK CREATION
# ========================
def create_binary_mask(segmentation):
    """Convert BraTS labels to a binary tumor-vs-background mask."""
    return (segmentation > 0).astype(np.float32)


# ========================
# 5. SLICE FILTERING
# ========================
def has_sufficient_tumor(mask, min_pixels=MIN_TUMOR_PIXELS):
    """
    Check if mask has enough tumor pixels.
    Supports binary masks in either (H, W) or (H, W, 1) format.
    
    Args:
        mask: Binary mask (H, W) or (H, W, 1)
        min_pixels: Minimum tumor pixels required
    
    Returns:
        bool: True if slice has sufficient tumor
    """
    if len(mask.shape) == 3:
        tumor_pixels = np.sum(mask[:, :, 0] > 0)
    else:
        # Binary mask
        tumor_pixels = np.sum(mask > 0)
    return tumor_pixels >= min_pixels


# ========================
# 6. DATA TYPE STANDARDIZATION
# ========================
def ensure_float32(array):
    """
    Convert array to float32 for GPU efficiency.
    
    Args:
        array: Input numpy array
    
    Returns:
        float32 numpy array
    """
    return array.astype(np.float32)


# ========================
# 7. MULTIMODAL VOLUME LOADING
# ========================
def load_multimodal_volume(patient_dir, require_seg=True):
    """
    Load all 4 MRI modalities for a patient.
    
    Args:
        patient_dir: Path to patient folder
        require_seg: If True, require and return segmentation volume.

    Returns:
        tuple: (volumes_dict, seg_volume_or_none, affine, header) or (None, None, None, None) if missing
    """
    strict_mode = os.environ.get("BRATS_STRICT_IO", "0") == "1"

    if not os.path.isdir(patient_dir):
        message = f"Patient directory not found: {patient_dir}"
        if strict_mode:
            raise FileNotFoundError(message)
        warnings.warn(message, RuntimeWarning)
        return None, None, None, None

    volumes = {}
    modalities_to_load = ['flair', 't1', 't1ce', 't2']
    affine = None
    header = None
    
    for i, modality in enumerate(modalities_to_load):
        pattern = os.path.join(patient_dir, f'*{modality}*.nii*')
        files = glob(pattern)
        
        if not files:
            # Try alternative patterns
            if modality == 't1ce':
                pattern = os.path.join(patient_dir, '*t1Gd*.nii*')
                files = glob(pattern)
            if modality == 't1' and not files:
                # Avoid matching t1ce when looking for t1
                all_t1 = glob(os.path.join(patient_dir, '*t1*.nii*'))
                files = [f for f in all_t1 if 'ce' not in f.lower() and 'gd' not in f.lower()]
        
        if not files:
            message = f"Missing modality '{modality}' under {patient_dir}"
            if strict_mode:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning)
            return None, None, None, None

        try:
            nii = nib.load(files[0])
            volumes[modality] = nii.get_fdata(dtype=np.float32)
        except Exception as exc:
            message = f"Failed to load modality '{modality}' for {patient_dir}: {exc}"
            if strict_mode:
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning)
            return None, None, None, None
        
        # FIX: Always set affine and header on first iteration
        if i == 0:
            affine = nii.affine
            header = nii.header
    
    seg_volume = None
    if require_seg:
        # Load segmentation
        seg_pattern = os.path.join(patient_dir, '*seg*.nii*')
        seg_files = glob(seg_pattern)

        if not seg_files:
            message = f"Missing segmentation file under {patient_dir}"
            if strict_mode:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning)
            return None, None, None, None

        try:
            seg_volume = nib.load(seg_files[0]).get_fdata(dtype=np.float32)
        except Exception as exc:
            message = f"Failed to load segmentation for {patient_dir}: {exc}"
            if strict_mode:
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning)
            return None, None, None, None

    return volumes, seg_volume, affine, header


# ========================
# 8. SINGLE SLICE PREPROCESSING (MULTIMODAL)
# ========================
def preprocess_multimodal_slice(modality_slices, seg_slice, target_size=IMG_SIZE, volume_stats=None):
    """
    Complete preprocessing pipeline for a single multimodal slice.
    
    Cleaned Pipeline:
    1. Gaussian blur (noise reduction)
    2. Z-score normalization per channel (volume-level if stats provided)
    3. Stack into 4-channel image
    4. Create binary tumor mask
    5. Resize both
    6. Ensure float32
    
    Args:
        modality_slices: dict with keys ['flair', 't1', 't1ce', 't2'], each (H, W)
        seg_slice: Raw segmentation slice (H, W)
        target_size: Output spatial size
        volume_stats: dict mapping modality -> {'mean', 'std'} for volume-level norm
    
    Returns:
        tuple: (processed_image (H,W,4), processed_mask (H,W,1)) both float32
    """
    # FIX: Assert that input has correct number of modalities
    assert set(modality_slices.keys()) == {'flair', 't1', 't1ce', 't2'}, \
        f"Expected 4 modalities, got {set(modality_slices.keys())}"
    
    processed_channels = []
    
    for modality in ['flair', 't1', 't1ce', 't2']:
        slice_data = modality_slices[modality]
        
        # Get volume stats for this modality if available
        stats = volume_stats.get(modality) if volume_stats else None
        
        # Step 1: Gaussian blur for noise reduction
        blurred = apply_gaussian_blur(slice_data, kernel_size=(3, 3))
        
        # Step 2: Z-score normalization (volume-level if stats provided)
        normalized = normalize_image(blurred, volume_stats=stats)
        
        processed_channels.append(normalized)
    
    # Step 3: Stack into 4-channel image (H, W, 4)
    img_stacked = np.stack(processed_channels, axis=-1)
    # FIX: Verify channel dimension before resizing
    assert img_stacked.shape[-1] == NUM_CHANNELS, \
        f"Expected {NUM_CHANNELS} channels, got {img_stacked.shape[-1]}"
    
    # Step 4: Create binary mask (H, W, 1)
    mask_binary = create_binary_mask(seg_slice)[..., np.newaxis]
    
    # Step 5: Resize (no-op when already at target size, e.g., 128x128 patches)
    img_resized = resize_image(img_stacked, target_size, interpolation='bilinear')
    mask_resized = resize_mask(mask_binary, target_size)
    
    # Step 6: Ensure float32
    img_final = ensure_float32(img_resized)
    mask_final = ensure_float32(mask_resized)
    
    return img_final, mask_final


# ========================
# 9. PATIENT-LEVEL PREPROCESSING (MULTIMODAL)
# ========================
def preprocess_patient_multimodal(patient_dir, min_tumor_pixels=MIN_TUMOR_PIXELS):
    """
    Preprocess all valid slices from a patient directory using multimodal data.
    Memory-safe: yields slices one at a time.
    
    FIX: Clear patient ID logging for debugging.
    
    Args:
        patient_dir: Path to patient folder
        min_tumor_pixels: Minimum tumor pixels to include slice
    
    Yields:
        tuple: (patient_id, slice_idx, image (H,W,4), mask (H,W,1))
    """
    patient_id = os.path.basename(patient_dir)
    
    # Load all modalities
    volumes, seg_vol, affine, header = load_multimodal_volume(patient_dir)
    
    if volumes is None:
        # FIX: Log with patient ID for clarity
        print(f"⚠ Patient '{patient_id}': Missing modalities, skipping...")
        return
    
    # FIX: Compute volume-level statistics for consistent normalization
    volume_stats = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        volume_stats[mod] = compute_volume_stats(volumes[mod])
    
    num_slices = seg_vol.shape[2]
    
    # Process each slice
    for slice_idx in range(num_slices):
        # Extract slices for each modality
        modality_slices = {
            mod: volumes[mod][:, :, slice_idx] 
            for mod in ['flair', 't1', 't1ce', 't2']
        }
        seg_slice = seg_vol[:, :, slice_idx]
        
        # Create binary mask for filtering check
        binary_mask = create_binary_mask(seg_slice)
        
        # Filter: skip slices with insufficient tumor
        if not has_sufficient_tumor(binary_mask, min_tumor_pixels):
            continue
        
        # Full preprocessing with volume-level normalization
        img_processed, mask_processed = preprocess_multimodal_slice(
            modality_slices, seg_slice, volume_stats=volume_stats
        )
        
        yield patient_id, slice_idx, img_processed, mask_processed


# Legacy single-modality functions for compatibility
def preprocess_slice(image_slice, seg_slice, target_size=IMG_SIZE):
    """
    Legacy: Single modality preprocessing (for backward compatibility).
    Use preprocess_multimodal_slice for new code.
    """
    # Normalize image
    img_normalized = normalize_image(image_slice)
    
    # Create binary mask
    mask_binary = create_binary_mask(seg_slice)[..., np.newaxis]
    
    # Resize
    img_resized = resize_image(img_normalized, target_size, 'bilinear')
    mask_resized = resize_mask(mask_binary, target_size)
    
    # Add channel dimension for single modality
    img_final = ensure_float32(img_resized[..., np.newaxis])
    mask_final = ensure_float32(mask_resized)
    
    return img_final, mask_final


# ========================
# 10. SHAPE & VALUE VALIDATION
# ========================
def validate_preprocessing(image, mask, verbose=True):
    """
    Validate preprocessed multimodal image and binary mask.
    
    Checks:
    - Correct shapes (H, W, 4) for image, (H, W, 1) for mask
    - Correct dtypes
    - Valid value ranges
    - Binary mask values per class
    
    Args:
        image: Preprocessed image (H, W, 4)
        mask: Preprocessed mask (H, W, 1)
        verbose: Print validation results
    
    Returns:
        bool: True if all checks pass
    """
    checks_passed = True
    errors = []
    
    # Shape checks
    expected_img_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    expected_mask_shape = (IMG_SIZE, IMG_SIZE, NUM_CLASSES)
    
    if image.shape != expected_img_shape:
        errors.append(f"Image shape {image.shape} != expected {expected_img_shape}")
        checks_passed = False
    if mask.shape != expected_mask_shape:
        errors.append(f"Mask shape {mask.shape} != expected {expected_mask_shape}")
        checks_passed = False
    
    # Dtype checks
    if image.dtype != np.float32:
        errors.append(f"Image dtype {image.dtype} != float32")
        checks_passed = False
    if mask.dtype != np.float32:
        errors.append(f"Mask dtype {mask.dtype} != float32")
        checks_passed = False
    
    # Value range checks for each class
    for c, class_name in enumerate(CLASS_NAMES):
        unique_vals = np.unique(mask[:, :, c])
        valid_vals = set(unique_vals).issubset({0.0, 1.0})
        if not valid_vals:
            errors.append(f"Mask {class_name} contains invalid values: {unique_vals}")
            checks_passed = False
    
    if verbose:
        print("=" * 50)
        print("PREPROCESSING VALIDATION (MULTIMODAL + BINARY)")
        print("=" * 50)
        print(f"Image Shape: {image.shape}")
        print(f"Mask Shape: {mask.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
        print("-" * 50)
        print("Per-channel image stats:")
        for c, mod in enumerate(MODALITIES):
            ch = image[:, :, c]
            print(f"  {mod.upper()}: mean={ch.mean():.4f}, std={ch.std():.4f}")
        print("-" * 50)
        print("Binary mask stats:")
        for c, class_name in enumerate(CLASS_NAMES):
            unique_vals = np.unique(mask[:, :, c])
            pixel_count = np.sum(mask[:, :, c] > 0)
            print(f"  {class_name}: {pixel_count:,} pixels, unique={unique_vals}")
        print("-" * 50)
        
        if checks_passed:
            print("✓ All validation checks PASSED")
        else:
            print("✗ Validation FAILED:")
            for err in errors:
                print(f"  - {err}")
        print("=" * 50)
    
    return checks_passed


# ========================
# DEMONSTRATION / SANITY CHECK
# ========================
print("=" * 50)
print("CELL 4: PREPROCESSING FUNCTIONS LOADED")
print("=" * 50)
print(f"Target Image Size: {IMG_SIZE} × {IMG_SIZE}")
print(f"Minimum Tumor Pixels: {MIN_TUMOR_PIXELS}")
print(f"Input Modalities: {MODALITIES}")
print(f"Output Classes: {CLASS_NAMES}")
print(f"Image Shape: ({IMG_SIZE}, {IMG_SIZE}, {NUM_CHANNELS})")
print(f"Mask Shape: ({IMG_SIZE}, {IMG_SIZE}, {NUM_CLASSES})")
print("-" * 50)
print("Key functions available:")
print("  • load_multimodal_volume(patient_dir)")
print("  • apply_clahe(image)")
print("  • apply_gaussian_blur(image)")
print("  • normalize_image(image)")
print("  • create_binary_mask(segmentation)")
print("  • preprocess_multimodal_slice(modality_slices, seg)")
print("  • preprocess_patient_multimodal(patient_dir)")
print("  • validate_preprocessing(image, mask)")
print("=" * 50)

# ========================
# OPTIONAL: TEST ON SAMPLE PATIENT
# ========================
if TRAIN_PATH and os.path.exists(TRAIN_PATH):
    print("\nRunning sample preprocessing test...")
    patient_dirs = sorted(glob(os.path.join(TRAIN_PATH, "BraTS*")))
    
    if patient_dirs:
        test_patient = patient_dirs[0]
        sample_count = 0
        
        for patient_id, slice_idx, img, mask in preprocess_patient_multimodal(test_patient):
            if sample_count == 0:
                # Validate first slice
                validate_preprocessing(img, mask)
                
                # Visual confirmation
                print(f"\nSample from: {patient_id}, Slice: {slice_idx}")
                print(f"Image shape: {img.shape}")
                print(f"Mask shape: {mask.shape}")
            sample_count += 1
        
        print(f"\nTotal valid slices from {os.path.basename(test_patient)}: {sample_count}")
else:
    if NPZ_PATCH_PATH and os.path.exists(NPZ_PATCH_PATH):
        print("\nNPZ patch dataset detected - skipping raw NIfTI sample test in Cell 4.")
        print("Functions are ready; Cell 5 will use NPZ patches to build HDF5.")
    else:
        print("\nDataset not found - skipping sample test.")
        print("Functions are ready for use when dataset is available.")

print("\n✓ Cell 4 complete. Multimodal preprocessing pipeline ready.")


def run_smoke_tests():
    """Required synthetic smoke tests for Cell 4."""
    h, w = 240, 240
    modality_slices = {
        'flair': np.random.rand(h, w).astype(np.float32),
        't1': np.random.rand(h, w).astype(np.float32),
        't1ce': np.random.rand(h, w).astype(np.float32),
        't2': np.random.rand(h, w).astype(np.float32),
    }
    seg = np.zeros((h, w), dtype=np.float32)
    seg[80:160, 80:160] = 4

    img, mask = preprocess_multimodal_slice(modality_slices, seg)
    assert img.shape == (IMG_SIZE, IMG_SIZE, NUM_CHANNELS), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (IMG_SIZE, IMG_SIZE, NUM_CLASSES), f"Unexpected mask shape: {mask.shape}"
    assert img.dtype == np.float32 and mask.dtype == np.float32, "Dtype mismatch"
    assert set(np.unique(mask[..., 0])).issubset({0.0, 1.0}), "Mask contains non-binary values"
    print("✓ Cell 4 smoke tests passed")


if __name__ == '__main__':
    run_smoke_tests()
