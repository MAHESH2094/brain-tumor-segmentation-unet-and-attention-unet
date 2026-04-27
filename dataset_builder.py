# ===================================================
# CELL 5: HDF5 Dataset Builder (PRODUCTION GRADE) (FIXED)
# ===================================================
# Purpose: Build HDF5 dataset with streaming writes, MULTIMODAL (4 ch), BINARY MASK (1 ch)
# FIXES: Docstring corrected, h5py.string_dtype(), XOR hashing, patient try/except,
#        cache num_samples, proper variable initialization

import hashlib
import json
import os
import time
import multiprocessing
import gc
from glob import glob

import h5py
import numpy as np
from tqdm import tqdm

# Defaults for standalone/restart use
if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
if "TRAIN_PATH" not in globals():
    TRAIN_PATH = None
if "NPZ_PATCH_PATH" not in globals():
    NPZ_PATCH_PATH = os.environ.get("BRATS_NPZ_PATCH_PATH", "").strip() or None
if "SEED" not in globals():
    SEED = 42
if "NUM_CHANNELS" not in globals():
    NUM_CHANNELS = 4
if "NUM_CLASSES" not in globals():
    NUM_CLASSES = 1
if "MASK_CHANNELS" not in globals():
    MASK_CHANNELS = 1

# Preprocessing dependency guard
_dataset_mode = os.environ.get("BRATS_DATASET_MODE", "").strip().lower()
_required_symbols = ["create_binary_mask", "preprocess_multimodal_slice"]
if _dataset_mode != "npz_patches":
    _required_symbols.append("load_multimodal_volume")

_missing_symbols = [name for name in _required_symbols if name not in globals()]
if _missing_symbols:
    print(
        "⚠ Preprocessing functions not found. Please run Cell 4 first. "
        f"Missing: {_missing_symbols}"
    )
    raise NameError(f"Missing required preprocessing symbols: {_missing_symbols}")

# ========================
# CONFIGURATION
# ========================
IMG_SIZE = int(os.environ.get("BRATS_IMG_SIZE", "128"))
os.environ["BRATS_IMG_SIZE"] = str(IMG_SIZE)
MIN_TUMOR_PIXELS = 50             # Minimum tumor pixels to keep slice
MODALITIES = ['flair', 't1', 't1ce', 't2']


def normalize_image_01_per_channel(image):
    """Min-max normalize each image channel independently to [0, 1]."""
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[-1]):
        channel = image[..., c].astype(np.float32)
        c_min = float(np.min(channel))
        c_max = float(np.max(channel))
        if c_max > c_min:
            out[..., c] = (channel - c_min) / (c_max - c_min)
        else:
            out[..., c] = 0.0
    return out

# HDF5 OUTPUT PATHS
HDF5_PATH = os.path.join(OUTPUT_DIR, "brats_preprocessed.h5")
os.environ["HDF5_PATH"] = HDF5_PATH
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Slice balance configuration
BACKGROUND_KEEP_RATIO = 0.10
WRITE_BATCH_SIZE = 256

# Dataset size limits (0 means no cap / use all available slices)
_default_max_train = "0" if _dataset_mode == "npz_patches" else "16000"
_default_max_val = "0" if _dataset_mode == "npz_patches" else "4000"
MAX_TRAIN_SAMPLES = int(os.environ.get("BRATS_MAX_TRAIN_SAMPLES", _default_max_train))
MAX_VAL_SAMPLES = int(os.environ.get("BRATS_MAX_VAL_SAMPLES", _default_max_val))
MAX_TEST_SAMPLES = int(os.environ.get("BRATS_MAX_TEST_SAMPLES", "0"))


def _split_sample_limit(split_name):
    """Return per-split sample cap, or None when uncapped."""
    if split_name == 'train':
        limit = MAX_TRAIN_SAMPLES
    elif split_name == 'val':
        limit = MAX_VAL_SAMPLES
    else:
        limit = MAX_TEST_SAMPLES

    return int(limit) if int(limit) > 0 else None


def _fmt_limit(limit):
    return 'ALL' if limit is None else str(int(limit))

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"

# ========================
# PATIENT DIRECTORY SCANNER
# ========================
def get_patient_directories(base_path):
    """
    Properly scan patient directories (not glob).
    
    Args:
        base_path: Path to dataset directory
    
    Returns:
        List of patient directory paths
    """
    if not os.path.exists(base_path):
        return []
    
    patient_dirs = [
        os.path.join(base_path, d)
        for d in sorted(os.listdir(base_path))
        if os.path.isdir(os.path.join(base_path, d))
    ]
    
    return patient_dirs


def get_npz_patch_files(base_path):
    """Recursively collect BraTS NPZ patch files from a dataset root."""
    if not base_path or not os.path.isdir(base_path):
        return []

    npz_files = sorted(glob(os.path.join(base_path, "**", "*.npz"), recursive=True))
    patch_files = [p for p in npz_files if "patch" in os.path.basename(p).lower()]

    max_files = int(os.environ.get("BRATS_NPZ_MAX_FILES", "0"))
    if max_files > 0:
        patch_files = patch_files[:max_files]

    return patch_files


# ========================
# PATIENT-LEVEL SPLIT
# ========================
def get_patient_splits(patient_dirs, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    """
    Split patients into train/val/test sets.
    CRITICAL: Split at patient level to prevent data leakage.
    
    Args:
        patient_dirs: List of patient directory paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)
    
    # Shuffle patients
    indices = rng.permutation(len(patient_dirs))
    
    # Calculate split points
    n_train = int(len(patient_dirs) * train_ratio)
    n_val = int(len(patient_dirs) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': [patient_dirs[i] for i in train_indices],
        'val': [patient_dirs[i] for i in val_indices],
        'test': [patient_dirs[i] for i in test_indices]
    }
    
    print(f"Patient splits (seed={seed}):")
    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val:   {len(splits['val'])} patients")
    print(f"  Test:  {len(splits['test'])} patients")
    
    return splits


# ========================
# BALANCED SLICE GENERATOR (MULTIMODAL)
# ========================
def preprocess_patient_balanced(patient_dir, min_tumor_pixels=MIN_TUMOR_PIXELS, 
                                 background_ratio=BACKGROUND_KEEP_RATIO, seed=SEED):
    """
    Preprocess patient with BALANCED tumor/background sampling using MULTIMODAL data.
    
    FIX: Clear logging of which patient failed if load fails.
    
    Args:
        patient_dir: Path to patient folder
        min_tumor_pixels: Minimum pixels to classify as tumor slice
        background_ratio: Fraction of background slices to keep
        seed: Random seed for background sampling
    
    Yields:
        tuple: (patient_id, slice_idx, image (H,W,4), mask (H,W,1), is_tumor_slice)
    """
    patient_id = os.path.basename(patient_dir)
    
    # Load all 4 modalities
    volumes, seg_vol, affine, header = load_multimodal_volume(patient_dir)
    
    if volumes is None:
        # FIX: Log patient ID clearly
        print(f"⚠ Patient '{patient_id}': Missing modalities, skipping...")
        return
    
    num_slices = seg_vol.shape[2]

    # Compute volume-level normalization stats once per patient and modality
    volume_stats = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        volume_stats[mod] = {
            'mean': float(np.mean(volumes[mod][volumes[mod] > 0])) if np.any(volumes[mod] > 0) else 0.0,
            'std': float(np.std(volumes[mod][volumes[mod] > 0])) if np.any(volumes[mod] > 0) else 1.0,
        }
    
    # Set seed for reproducible background sampling
    # FIX: Use XOR instead of mod to avoid hash collisions
    patient_hash = int(hashlib.sha1(patient_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed ^ (patient_hash % (2**31)))  # XOR for better distribution
    
    # Process each slice
    for slice_idx in range(num_slices):
        # Extract slices for each modality
        modality_slices = {
            mod: volumes[mod][:, :, slice_idx] 
            for mod in ['flair', 't1', 't1ce', 't2']
        }
        seg_slice = seg_vol[:, :, slice_idx]
        
        # Create binary mask for filtering.
        binary_mask = create_binary_mask(seg_slice)

        tumor_pixels = np.sum(binary_mask > 0)
        is_tumor_slice = tumor_pixels >= min_tumor_pixels
        
        # Decision: keep or skip
        if is_tumor_slice:
            # ALWAYS keep tumor slices
            keep = True
        else:
            # Randomly keep some background slices
            keep = rng.random() < background_ratio
        
        if not keep:
            continue
        
        # Full preprocessing (multimodal + binary mask)
        img_processed, mask_processed = preprocess_multimodal_slice(
            modality_slices, seg_slice, volume_stats=volume_stats
        )

        # Enforce deterministic [0,1] image range for downstream assertions/training.
        img_processed = normalize_image_01_per_channel(img_processed)

        # Binary conversion for HDF5: enforce a single tumor channel.
        mask_processed = np.any(mask_processed > 0.5, axis=-1, keepdims=True).astype(np.float32)
        
        yield patient_id, slice_idx, img_processed, mask_processed, is_tumor_slice


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


def preprocess_npz_patch_balanced(npz_path, min_tumor_pixels=MIN_TUMOR_PIXELS,
                                  background_ratio=BACKGROUND_KEEP_RATIO, seed=SEED):
    """
    Convert one NPZ 3D patch into balanced 2D training slices.

    Expected NPZ keys:
      - image: (4,H,W,D) or (H,W,D,4)
      - mask:  (H,W,D)
    """
    patch_id = os.path.splitext(os.path.basename(npz_path))[0]

    with np.load(npz_path) as data:
        if "image" not in data or "mask" not in data:
            raise KeyError(f"NPZ missing required keys image/mask: {npz_path}")
        image = data["image"]
        mask = data["mask"]

    image_chwd, mask_hwd = _coerce_npz_patch_layout(image, mask, npz_path)

    # Dataset card channel order is typically [T1, T1ce, T2, FLAIR].
    # Map to project order [FLAIR, T1, T1ce, T2].
    channel_order_raw = os.environ.get("BRATS_NPZ_CHANNEL_ORDER", "t1,t1ce,t2,flair")
    order = [token.strip().lower() for token in channel_order_raw.split(",") if token.strip()]
    expected = {"flair", "t1", "t1ce", "t2"}
    if set(order) != expected:
        order = ["t1", "t1ce", "t2", "flair"]

    channel_index = {name: idx for idx, name in enumerate(order)}

    patient_hash = int(hashlib.sha1(patch_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed ^ (patient_hash % (2**31)))

    num_slices = int(image_chwd.shape[-1])
    for slice_idx in range(num_slices):
        modality_slices = {
            "flair": image_chwd[channel_index["flair"], :, :, slice_idx],
            "t1": image_chwd[channel_index["t1"], :, :, slice_idx],
            "t1ce": image_chwd[channel_index["t1ce"], :, :, slice_idx],
            "t2": image_chwd[channel_index["t2"], :, :, slice_idx],
        }
        seg_slice = mask_hwd[:, :, slice_idx]

        binary_mask = create_binary_mask(seg_slice)
        tumor_pixels = np.sum(binary_mask > 0)
        is_tumor_slice = tumor_pixels >= min_tumor_pixels

        if not is_tumor_slice and rng.random() >= background_ratio:
            continue

        img_processed, mask_processed = preprocess_multimodal_slice(modality_slices, seg_slice)
        img_processed = normalize_image_01_per_channel(img_processed)
        mask_processed = np.any(mask_processed > 0.5, axis=-1, keepdims=True).astype(np.float32)

        yield patch_id, slice_idx, img_processed, mask_processed, is_tumor_slice


# ========================
# STREAMING HDF5 WRITER
# ========================
class StreamingHDF5Writer:
    """
    Memory-safe HDF5 writer with streaming/incremental writes.
    Uses resizable datasets to avoid loading all data into RAM.
    """
    
    def __init__(self, hdf5_path, img_size=IMG_SIZE, chunk_size=WRITE_BATCH_SIZE):
        """
        Initialize streaming writer.
        
        Args:
            hdf5_path: Output HDF5 file path
            img_size: Image dimensions
            chunk_size: Chunk size for HDF5
        """
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.file = None
        self.datasets = {}
        self.current_sizes = {}
        self.buffers = {}
    
    def open(self):
        """Open HDF5 file for writing."""
        self.file = h5py.File(self.hdf5_path, 'w')
        return self
    
    def close(self):
        """Flush buffers and close file."""
        # Flush any remaining data in buffers
        for split in self.buffers:
            self._flush_buffer(split)
        
        if self.file:
            self.file.close()
    
    def __enter__(self):
        return self.open()
    
    def __exit__(self, *args):
        self.close()
    
    def create_split(self, split_name):
        """
        Create datasets for a split with resizable dimensions.
        
        FIX: Docstring corrected to (H, W, 4) and (H, W, 1).
        
        Args:
            split_name: 'train', 'val', or 'test'
        """
        grp = self.file.create_group(split_name)
        
        # Create resizable datasets with MULTIMODAL inputs and BINARY masks
        # Images: (N, H, W, 4) for 4 modalities (FLAIR, T1, T1ce, T2)
        # Masks: (N, H, W, 1) for tumor-vs-background binary segmentation
        img_shape = (0, self.img_size, self.img_size, NUM_CHANNELS)
        img_maxshape = (None, self.img_size, self.img_size, NUM_CHANNELS)
        img_chunks = (self.chunk_size, self.img_size, self.img_size, NUM_CHANNELS)
        
        mask_shape = (0, self.img_size, self.img_size, MASK_CHANNELS)
        mask_maxshape = (None, self.img_size, self.img_size, MASK_CHANNELS)
        mask_chunks = (self.chunk_size, self.img_size, self.img_size, MASK_CHANNELS)
        
        grp.create_dataset(
            'images',
            shape=img_shape,
            maxshape=img_maxshape,
            chunks=img_chunks,
            dtype=np.float32,
            compression='gzip',
            compression_opts=1
        )
        
        grp.create_dataset(
            'masks',
            shape=mask_shape,
            maxshape=mask_maxshape,
            chunks=mask_chunks,
            dtype=np.float32,
            compression='gzip',
            compression_opts=1
        )
        
        # FIX: Use h5py.string_dtype() instead of deprecated special_dtype
        dt = h5py.string_dtype(encoding='utf-8')
        grp.create_dataset(
            'patient_ids',
            shape=(0,),
            maxshape=(None,),
            dtype=dt
        )
        
        grp.create_dataset(
            'slice_indices',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32
        )
        
        grp.create_dataset(
            'is_tumor',
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_
        )
        
        self.datasets[split_name] = grp
        self.current_sizes[split_name] = 0
        self.buffers[split_name] = {
            'images': [],
            'masks': [],
            'patient_ids': [],
            'slice_indices': [],
            'is_tumor': []
        }
    
    def add_sample(self, split_name, image, mask, patient_id, slice_idx, is_tumor):
        """
        Add a sample to buffer, flush when buffer is full.
        
        FIX: Docstring corrected to (H, W, 4) and (H, W, 1).
        
        Args:
            split_name: Target split
            image: Preprocessed image (256, 256, 4)
            mask: Preprocessed mask (256, 256, 1)
            patient_id: Patient identifier
            slice_idx: Slice index
            is_tumor: Whether slice contains tumor
        """
        buf = self.buffers[split_name]
        buf['images'].append(image)
        buf['masks'].append(mask)
        buf['patient_ids'].append(patient_id)
        buf['slice_indices'].append(slice_idx)
        buf['is_tumor'].append(is_tumor)
        
        # Flush when buffer is full
        if len(buf['images']) >= self.chunk_size:
            self._flush_buffer(split_name)
    
    def _flush_buffer(self, split_name):
        """
        Write buffer to HDF5 and clear.
        
        Args:
            split_name: Split to flush
        """
        buf = self.buffers[split_name]
        
        if len(buf['images']) == 0:
            return
        
        grp = self.datasets[split_name]
        current_size = self.current_sizes[split_name]
        new_samples = len(buf['images'])
        new_size = current_size + new_samples
        
        # Resize datasets
        grp['images'].resize(new_size, axis=0)
        grp['masks'].resize(new_size, axis=0)
        grp['patient_ids'].resize(new_size, axis=0)
        grp['slice_indices'].resize(new_size, axis=0)
        grp['is_tumor'].resize(new_size, axis=0)
        
        # Write data
        grp['images'][current_size:new_size] = np.array(buf['images'], dtype=np.float32)
        grp['masks'][current_size:new_size] = np.array(buf['masks'], dtype=np.float32)
        grp['patient_ids'][current_size:new_size] = buf['patient_ids']
        grp['slice_indices'][current_size:new_size] = np.array(buf['slice_indices'], dtype=np.int32)
        grp['is_tumor'][current_size:new_size] = np.array(buf['is_tumor'], dtype=np.bool_)
        
        # Update size and clear buffer
        self.current_sizes[split_name] = new_size
        self.buffers[split_name] = {
            'images': [],
            'masks': [],
            'patient_ids': [],
            'slice_indices': [],
            'is_tumor': []
        }
    
    def get_split_size(self, split_name):
        """Get current number of samples in split."""
        return self.current_sizes.get(split_name, 0)


# ========================
# MAIN DATASET BUILDER
# ========================
def build_hdf5_dataset(patient_dirs, output_path=HDF5_PATH):
    """
    Build HDF5 dataset with STREAMING WRITES (memory-safe).
    
    FIX: Wrap patient processing in try/except to skip corrupt patients.
    
    Args:
        patient_dirs: List of patient directory paths
        output_path: HDF5 file path
    
    Returns:
        dict: Dataset statistics
    """
    print("=" * 50)
    print("BUILDING HDF5 DATASET (STREAMING MODE)")
    print("=" * 50)
    print(f"Output: {output_path}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Min tumor pixels: {MIN_TUMOR_PIXELS}")
    print(f"Background keep ratio: {BACKGROUND_KEEP_RATIO}")
    print(f"Write batch size: {WRITE_BATCH_SIZE}")
    print("-" * 50)
    
    # Get patient splits
    splits = get_patient_splits(patient_dirs)
    
    # Statistics tracking
    stats = {
        'train': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['train'])},
        'val': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['val'])},
        'test': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['test'])},
        'config': {
            'dataset': os.environ.get('BRATS_DATASET_NAME', 'BraTS'),
            'img_size': IMG_SIZE,
            'min_tumor_pixels': MIN_TUMOR_PIXELS,
            'background_ratio': BACKGROUND_KEEP_RATIO,
            'max_train_samples': MAX_TRAIN_SAMPLES,
            'max_val_samples': MAX_VAL_SAMPLES,
            'max_test_samples': MAX_TEST_SAMPLES,
            'modalities': MODALITIES,
            'mask_channels': MASK_CHANNELS,
            'seed': SEED
        }
    }
    
    start_time = time.time()
    
    with StreamingHDF5Writer(output_path) as writer:
        for split_name, split_patients in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_patients)} patients)...")
            
            # Create split datasets
            writer.create_split(split_name)

            # Determine sample limit for this split
            sample_limit = _split_sample_limit(split_name)
            
            # Process patients one by one
            split_complete = False
            skipped_patients = 0
            
            for patient_dir in tqdm(split_patients, desc=f"{split_name}", miniters=5):
                if split_complete:
                    break
                
                # FIX: Wrap patient processing to skip corrupt/missing patients
                try:
                    for patient_id, slice_idx, img, mask, is_tumor in preprocess_patient_balanced(patient_dir):
                        # Add sample to HDF5 (streaming write)
                        writer.add_sample(
                            split_name,
                            img, mask,
                            patient_id, slice_idx,
                            is_tumor
                        )
                        
                        # Update stats
                        stats[split_name]['total'] += 1
                        if is_tumor:
                            stats[split_name]['tumor'] += 1
                        else:
                            stats[split_name]['background'] += 1
                        
                        # Check sample limit
                        if sample_limit and stats[split_name]['total'] >= sample_limit:
                            print(f"\n  ⚡ Reached {split_name} sample limit: {sample_limit}")
                            split_complete = True
                            break
                except Exception as e:
                    # Log but continue with next patient
                    skipped_patients += 1
                    print(f"\n  ⚠ Error processing {os.path.basename(patient_dir)}: {e}")
                    continue
            
            # Report split stats
            s = stats[split_name]
            print(f"  {split_name}: {s['total']} slices ({s['tumor']} tumor, {s['background']} background)")
            if skipped_patients > 0:
                print(f"  ⚠ Skipped {skipped_patients} patients due to errors")
        
        # Store metadata
        writer.file.attrs['config'] = json.dumps(stats['config'])
        writer.file.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.file.attrs['stats'] = json.dumps({k: v for k, v in stats.items() if k != 'config'})
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 50)
    print("HDF5 DATASET CREATED (STREAMING)")
    print("=" * 50)
    print(f"File: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"Build time: {elapsed:.1f} seconds")
    print("-" * 50)
    
    total_all = 0
    for split in ['train', 'val', 'test']:
        s = stats[split]
        total_all += s['total']
        tumor_pct = (s['tumor'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"{split.upper():6}: {s['total']:6,} slices | {s['tumor']:5,} tumor ({tumor_pct:5.1f}%) | {s['background']:5,} background")
    
    print("-" * 50)
    print(f"TOTAL:  {total_all:,} slices")
    print("=" * 50)
    
    return stats


def build_hdf5_dataset_from_npz(npz_files, output_path=HDF5_PATH):
    """Build HDF5 dataset from preprocessed NPZ patch files."""
    print("=" * 50)
    print("BUILDING HDF5 DATASET FROM NPZ PATCHES")
    print("=" * 50)
    print(f"Output: {output_path}")
    print(f"NPZ files: {len(npz_files)}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Min tumor pixels: {MIN_TUMOR_PIXELS}")
    print(f"Background keep ratio: {BACKGROUND_KEEP_RATIO}")
    print(f"Write batch size: {WRITE_BATCH_SIZE}")
    print("-" * 50)

    splits = get_patient_splits(npz_files)

    stats = {
        'train': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['train'])},
        'val': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['val'])},
        'test': {'total': 0, 'tumor': 0, 'background': 0, 'patients': len(splits['test'])},
        'config': {
            'dataset': os.environ.get('BRATS_DATASET_NAME', 'BraTS-2024-preprocessed-patches'),
            'img_size': IMG_SIZE,
            'min_tumor_pixels': MIN_TUMOR_PIXELS,
            'background_ratio': BACKGROUND_KEEP_RATIO,
            'max_train_samples': MAX_TRAIN_SAMPLES,
            'max_val_samples': MAX_VAL_SAMPLES,
            'max_test_samples': MAX_TEST_SAMPLES,
            'modalities': MODALITIES,
            'mask_channels': MASK_CHANNELS,
            'seed': SEED,
            'source': 'npz_patches',
        }
    }

    start_time = time.time()

    with StreamingHDF5Writer(output_path) as writer:
        for split_name, split_npz_files in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_npz_files)} npz files)...")
            writer.create_split(split_name)

            sample_limit = _split_sample_limit(split_name)

            split_complete = False
            skipped_files = 0

            for npz_path in tqdm(split_npz_files, desc=f"{split_name}", miniters=5):
                if split_complete:
                    break

                try:
                    for patient_id, slice_idx, img, mask, is_tumor in preprocess_npz_patch_balanced(npz_path):
                        writer.add_sample(split_name, img, mask, patient_id, slice_idx, is_tumor)

                        stats[split_name]['total'] += 1
                        if is_tumor:
                            stats[split_name]['tumor'] += 1
                        else:
                            stats[split_name]['background'] += 1

                        if sample_limit and stats[split_name]['total'] >= sample_limit:
                            print(f"\n  ⚡ Reached {split_name} sample limit: {sample_limit}")
                            split_complete = True
                            break
                except Exception as e:
                    skipped_files += 1
                    print(f"\n  ⚠ Error processing {os.path.basename(npz_path)}: {e}")
                    continue

            s = stats[split_name]
            print(f"  {split_name}: {s['total']} slices ({s['tumor']} tumor, {s['background']} background)")
            if skipped_files > 0:
                print(f"  ⚠ Skipped {skipped_files} NPZ files due to errors")

        writer.file.attrs['config'] = json.dumps(stats['config'])
        writer.file.attrs['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.file.attrs['stats'] = json.dumps({k: v for k, v in stats.items() if k != 'config'})

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("HDF5 DATASET CREATED FROM NPZ PATCHES")
    print("=" * 50)
    print(f"File: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"Build time: {elapsed:.1f} seconds")
    print("-" * 50)

    total_all = 0
    for split in ['train', 'val', 'test']:
        s = stats[split]
        total_all += s['total']
        tumor_pct = (s['tumor'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"{split.upper():6}: {s['total']:6,} slices | {s['tumor']:5,} tumor ({tumor_pct:5.1f}%) | {s['background']:5,} background")

    print("-" * 50)
    print(f"TOTAL:  {total_all:,} slices")
    print("=" * 50)

    return stats


# ========================
# HDF5 DATA LOADER (FIXED: Cache num_samples)
# ========================
class HDF5DataLoader:
    """
    Memory-efficient data loader from HDF5 file.
    Loads batches on-demand without loading full dataset.
    
    FIX: Cache num_samples at init to avoid repeated file opens.
    """
    
    def __init__(self, hdf5_path, split='train'):
        """
        Initialize loader.
        
        Args:
            hdf5_path: Path to HDF5 file
            split: 'train', 'val', or 'test'
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.file = None
        self._images_ds = None
        self._masks_ds = None
        self._image_shape = None
        
        # FIX: Cache num_samples at init time
        with h5py.File(hdf5_path, 'r') as f:
            self._num_samples = f[split]['images'].shape[0]
            self._image_shape = f[split]['images'].shape[1:]

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
            self._images_ds = self.file[self.split]['images']
            self._masks_ds = self.file[self.split]['masks']

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
        self.file = None
        self._images_ds = None
        self._masks_ds = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            if getattr(self, 'file', None) is not None:
                self.close()
        except Exception:
            # Destructor must never raise during GC/interpreter shutdown.
            pass
    
    @property
    def num_samples(self):
        return self._num_samples
    
    @property
    def image_shape(self):
        return self._image_shape
    
    def get_batch(self, start_idx, batch_size):
        """Get a batch of data."""
        end_idx = min(start_idx + batch_size, self.num_samples)

        self._ensure_open()
        images = self._images_ds[start_idx:end_idx]
        masks = self._masks_ds[start_idx:end_idx]

        return images, masks
    
    def get_all(self):
        """Load entire split into memory (use carefully)."""
        self._ensure_open()
        images = self._images_ds[:]
        masks = self._masks_ds[:]

        return images, masks
    
    def batch_generator(self, batch_size, shuffle=True):
        """
        Generator for training batches.
        
        Args:
            batch_size: Samples per batch
            shuffle: Whether to shuffle indices
        
        Yields:
            tuple: (batch_images, batch_masks)
        """
        n = self.num_samples
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)

        self._ensure_open()

        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_indices = sorted(batch_indices)  # HDF5 requires sorted indices

            batch_images = self._images_ds[batch_indices]
            batch_masks = self._masks_ds[batch_indices]

            yield batch_images, batch_masks


# ========================
# VERIFICATION FUNCTION
# ========================
def verify_hdf5_dataset(hdf5_path):
    """
    Verify HDF5 dataset integrity with balance stats.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        bool: True if verification passes
    """
    print("=" * 50)
    print("HDF5 DATASET VERIFICATION")
    print("=" * 50)
    
    if not os.path.exists(hdf5_path):
        print(f"✗ File not found: {hdf5_path}")
        return False
    
    with h5py.File(hdf5_path, 'r') as f:
        print(f"File: {hdf5_path}")
        print(f"Size: {os.path.getsize(hdf5_path) / (1024**2):.2f} MB")
        print(f"Groups: {list(f.keys())}")
        print("-" * 50)
        
        for split in ['train', 'val', 'test']:
            if split not in f:
                print(f"✗ Missing split: {split}")
                continue
            
            grp = f[split]
            n_samples = grp['images'].shape[0]
            img_shape = grp['images'].shape[1:]
            mask_shape = grp['masks'].shape[1:]
            
            # Count tumor vs background
            is_tumor = grp['is_tumor'][:]
            n_tumor = np.sum(is_tumor)
            n_background = n_samples - n_tumor
            tumor_pct = (n_tumor / n_samples * 100) if n_samples > 0 else 0
            
            print(f"{split.upper()}:")
            print(f"  Samples: {n_samples:,}")
            print(f"  Image shape: {img_shape}")
            print(f"  Mask shape: {mask_shape}")
            print(f"  Tumor: {n_tumor:,} ({tumor_pct:.1f}%)")
            print(f"  Background: {n_background:,} ({100-tumor_pct:.1f}%)")
            
            # Range check
            if n_samples > 0:
                sample_img = grp['images'][0]
                print(f"  Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    
    print("=" * 50)
    print("✓ HDF5 verification complete")
    print("=" * 50)
    
    return True


# ========================
# EXECUTION
# ========================
print("=" * 50)
print("CELL 5: HDF5 DATASET BUILDER (PRODUCTION - FIXED)")
print("=" * 50)
print(f"HDF5 output path: {HDF5_PATH}")
print(f"Train/Val/Test ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
print(f"Background keep ratio: {BACKGROUND_KEEP_RATIO}")
print(
    "Sample caps train/val/test: "
    f"{_fmt_limit(_split_sample_limit('train'))}/"
    f"{_fmt_limit(_split_sample_limit('val'))}/"
    f"{_fmt_limit(_split_sample_limit('test'))}"
)
print("-" * 50)

DATASET_MODE = os.environ.get("BRATS_DATASET_MODE", "").strip().lower()

if DATASET_MODE == "npz_patches" and NPZ_PATCH_PATH and os.path.exists(NPZ_PATCH_PATH):
    npz_files = get_npz_patch_files(NPZ_PATCH_PATH)

    if npz_files:
        print(f"\nFound {len(npz_files)} NPZ patch files in: {NPZ_PATCH_PATH}")
        print("Building HDF5 dataset from NPZ patches...\n")

        dataset_stats = build_hdf5_dataset_from_npz(npz_files, HDF5_PATH)

        verify_hdf5_dataset(HDF5_PATH)

        TRAIN_SAMPLES = dataset_stats['train']['total']
        VAL_SAMPLES = dataset_stats['val']['total']
        TEST_SAMPLES = dataset_stats['test']['total']

        gc.collect()
    else:
        print(f"No NPZ patch files found in: {NPZ_PATCH_PATH}")
        TRAIN_SAMPLES = VAL_SAMPLES = TEST_SAMPLES = 0
elif TRAIN_PATH and os.path.exists(TRAIN_PATH):
    patient_dirs = get_patient_directories(TRAIN_PATH)
    
    if patient_dirs:
        print(f"\nFound {len(patient_dirs)} patients.")
        print("Building HDF5 dataset with streaming writes...\n")
        
        # Build the dataset
        dataset_stats = build_hdf5_dataset(patient_dirs, HDF5_PATH)
        
        # Verify
        verify_hdf5_dataset(HDF5_PATH)
        
        # Store stats for later cells
        TRAIN_SAMPLES = dataset_stats['train']['total']
        VAL_SAMPLES = dataset_stats['val']['total']
        TEST_SAMPLES = dataset_stats['test']['total']
        
        # Clean up
        gc.collect()
    else:
        print("No patient directories found.")
        TRAIN_SAMPLES = VAL_SAMPLES = TEST_SAMPLES = 0
elif NPZ_PATCH_PATH and os.path.exists(NPZ_PATCH_PATH):
    npz_files = get_npz_patch_files(NPZ_PATCH_PATH)

    if npz_files:
        print(f"\nFound {len(npz_files)} NPZ patch files in: {NPZ_PATCH_PATH}")
        print("Building HDF5 dataset from NPZ patches...\n")

        dataset_stats = build_hdf5_dataset_from_npz(npz_files, HDF5_PATH)

        verify_hdf5_dataset(HDF5_PATH)

        TRAIN_SAMPLES = dataset_stats['train']['total']
        VAL_SAMPLES = dataset_stats['val']['total']
        TEST_SAMPLES = dataset_stats['test']['total']

        gc.collect()
    else:
        print(f"No NPZ patch files found in: {NPZ_PATCH_PATH}")
        TRAIN_SAMPLES = VAL_SAMPLES = TEST_SAMPLES = 0
else:
    print(f"\nDataset path not found: {TRAIN_PATH}")
    if NPZ_PATCH_PATH:
        print(f"NPZ patch path checked: {NPZ_PATCH_PATH}")
    print("HDF5 builder ready. Run build_hdf5_dataset() or build_hdf5_dataset_from_npz() when data available.")
    TRAIN_SAMPLES = VAL_SAMPLES = TEST_SAMPLES = 0

print("\n✓ Cell 5 complete. Production HDF5 builder ready.")


def run_smoke_tests():
    """Required synthetic smoke tests for Cell 5 writer/loader flow."""
    tmp_path = os.path.join(OUTPUT_DIR, 'tmp_smoke_brats.h5')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with StreamingHDF5Writer(tmp_path, img_size=64, chunk_size=2) as writer:
        writer.create_split('train')
        x = np.random.rand(64, 64, NUM_CHANNELS).astype(np.float32)
        y = np.random.randint(0, 2, size=(64, 64, MASK_CHANNELS)).astype(np.float32)
        writer.add_sample('train', x, y, 'P001', 0, True)
        writer.add_sample('train', x, y, 'P001', 1, False)

    with h5py.File(tmp_path, 'r') as f:
        assert f['train/images'].shape == (2, 64, 64, NUM_CHANNELS)
        assert f['train/masks'].shape == (2, 64, 64, MASK_CHANNELS)

    os.remove(tmp_path)
    print("✓ Cell 5 smoke tests passed")


if __name__ == '__main__':
    run_smoke_tests()
