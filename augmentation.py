# ===================================================
# CELL 6: DATA AUGMENTATION PIPELINE (PREMIUM TURBO) (FIXED)
# ===================================================
# Purpose: High-speed GPU augmentation with premium visualization
# FIXES: TF version check for ImageProjectiveTransformV3, move h5py import,
#        document generator context manager, medical augmentations added

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
if 'HDF5_PATH' not in globals():
    HDF5_PATH = os.path.join(OUTPUT_DIR, 'brats_preprocessed.h5')

# ========================
# CONFIGURATION
# ========================
AUGMENTATION_ENABLED = True
ROTATION_RANGE = int(os.environ.get("BRATS_ROTATION_RANGE", "20"))  # FIX: Increased from 15
FLIP_HORIZONTAL = True
FLIP_VERTICAL = False
INTENSITY_SCALE_RANGE = (0.85, 1.15)
AUG_PROBABILITY = float(os.environ.get("BRATS_AUG_PROBABILITY", "0.7"))  # FIX: Increased from 0.5

# Medical augmentation config
NOISE_STDDEV = 0.02        # Gaussian noise standard deviation
GAMMA_RANGE = (0.7, 1.3)   # Gamma correction range
MEDICAL_AUG_PROB = 0.5     # Probability of applying medical augmentations

# Progressive augmentation scale updated at runtime by training callback.
# Keep as tf.Variable so tf.function can read fresh values each step.
AUG_PROGRESSIVE_SCALE = tf.Variable(1.0, trainable=False, dtype=tf.float32)


def set_augmentation_scale(scale):
    """Update progressive augmentation scale in [0, 1]."""
    clipped = float(np.clip(scale, 0.0, 1.0))
    AUG_PROGRESSIVE_SCALE.assign(clipped)
    return clipped

# ========================
# HIGH-SPEED TF TRANSFORMS
# ========================

def tfa_rotate(image, angle, interpolation='bilinear'):
    """
    Rotation using raw TF ops for maximum speed.
    
    FIX: Guard for TensorFlow version compatibility.
    ImageProjectiveTransformV3 may not exist in all TF versions.
    """
    shape = tf.shape(image)
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    cx, cy = w/2.0, h/2.0
    
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    
    # Rotation matrix elements
    m0, m1, m2 = cos_a, -sin_a, cx - cx*cos_a + cy*sin_a
    m3, m4, m5 = sin_a, cos_a, cy - cx*sin_a - cy*cos_a
    
    transform = tf.stack([m0, m1, m2, m3, m4, m5, 0.0, 0.0])
    transform = tf.reshape(transform, [1, 8])
    
    interp = 'NEAREST' if interpolation == 'nearest' else 'BILINEAR'
    
    try:
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, 0),
            transforms=transform,
            output_shape=shape[:2],
            fill_value=0.0,
            interpolation=interp
        )
        return rotated[0]
    except AttributeError:
        try:
            rotated = tf.raw_ops.ImageProjectiveTransformV2(
                images=tf.expand_dims(image, 0),
                transforms=transform,
                output_shape=shape[:2],
                fill_value=0.0,
                interpolation=interp
            )
            return rotated[0]
        except AttributeError:
            try:
                import tensorflow_addons as tfa
                return tfa.image.rotate(image, angle, interpolation=interpolation.upper(), fill_mode='constant', fill_value=0.0)
            except Exception:
                print(
                    f"⚠ Rotation op unavailable in TF {tf.__version__} and tensorflow-addons not installed. "
                    f"Returning image without rotation."
                )
                return image


# ========================
# MEDICAL AUGMENTATIONS
# ========================

@tf.function
def add_gaussian_noise(image, stddev=NOISE_STDDEV):
    """Add Gaussian noise to simulate MRI acquisition noise."""
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev)
    noisy = image + noise
    return tf.clip_by_value(noisy, -5.0, 5.0)  # Clip to reasonable z-score range


@tf.function
def gamma_correction(image, gamma_min=GAMMA_RANGE[0], gamma_max=GAMMA_RANGE[1]):
    """Apply random gamma correction to simulate intensity variations."""
    gamma = tf.random.uniform([], gamma_min, gamma_max)
    # Shift to positive range, apply gamma, shift back
    img_min = tf.reduce_min(image)
    img_shifted = image - img_min + 1e-8  # Ensure positive
    img_max = tf.reduce_max(img_shifted) + 1e-8
    img_norm = img_shifted / img_max
    img_gamma = tf.pow(img_norm, gamma)
    return img_gamma * img_max + img_min


@tf.function
def channel_dropout(image, drop_prob=0.1):
    """Randomly zero out one modality channel to build robustness."""
    num_channels = tf.shape(image)[-1]
    # Pick a random channel to potentially drop
    channel_to_drop = tf.random.uniform([], 0, num_channels, dtype=tf.int32)
    
    if tf.random.uniform([]) < drop_prob:
        # Create a mask that zeros out one channel
        mask = tf.one_hot(channel_to_drop, num_channels)
        mask = 1.0 - mask  # Invert: 0 for dropped channel, 1 for others
        # Broadcast mask to image shape
        mask = tf.reshape(mask, [1, 1, num_channels])
        return image * mask
    return image


@tf.function
def augment_pair(image, mask):
    """
    Main augmentation logic with medical augmentations.
    Guarantees spatial alignment between image and mask.
    """
    if not AUGMENTATION_ENABLED:
        return image, mask

    # Scale augmentation probabilities during progressive training.
    aug_prob = tf.clip_by_value(tf.cast(AUG_PROBABILITY, tf.float32) * AUG_PROGRESSIVE_SCALE, 0.0, 1.0)
    med_prob = tf.clip_by_value(tf.cast(MEDICAL_AUG_PROB, tf.float32) * AUG_PROGRESSIVE_SCALE, 0.0, 1.0)
        
    # 1. Flip Horizontal
    if FLIP_HORIZONTAL and tf.random.uniform([]) < aug_prob:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        
    # 2. Rotation (CRITICAL: mask uses nearest neighbor, not bilinear)
    if ROTATION_RANGE > 0 and tf.random.uniform([]) < aug_prob:
        angle = tf.random.uniform([], -ROTATION_RANGE, ROTATION_RANGE) * (np.pi / 180.0)
        image = tfa_rotate(image, angle, interpolation='bilinear')
        mask = tfa_rotate(mask, angle, interpolation='nearest')
        
    # 3. Intensity Scaling (Image only)
    if tf.random.uniform([]) < aug_prob:
        scale = tf.random.uniform([], INTENSITY_SCALE_RANGE[0], INTENSITY_SCALE_RANGE[1])
        image = image * scale

    # ===== MEDICAL AUGMENTATIONS (FIX: Added for better generalization) =====
    
    # 4. Gaussian Noise (MRI acquisition noise simulation)
    if tf.random.uniform([]) < med_prob * 0.8:  # ~40% chance at full scale
        image = add_gaussian_noise(image)

    # 5. Gamma Correction (intensity variation simulation)
    if tf.random.uniform([]) < med_prob * 0.6:  # ~30% chance at full scale
        image = gamma_correction(image)

    # 6. Channel Dropout (modality robustness) - low probability
    if tf.random.uniform([]) < 0.1:
        image = channel_dropout(image, drop_prob=1.0)  # Already gated by outer if
        
    # CRITICAL FIX: Ensure mask integrity with strict binary (round, not threshold)
    mask = tf.round(mask)
    
    return image, mask

# ========================
# DATASET UTILITIES
# ========================

def create_tf_dataset(hdf5_path, split='train', batch_size=16, apply_aug=True):
    """Creates a high-performance tf.data.Dataset from HDF5."""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        num_samples = int(f[f'{split}/images'].shape[0])

    def batch_generator():
        with h5py.File(hdf5_path, 'r') as f:
            images_ds = f[f'{split}/images']
            masks_ds = f[f'{split}/masks']

            indices = np.arange(num_samples)
            if split == 'train':
                np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                batch_idx = np.sort(indices[start:start + batch_size])
                x_batch = images_ds[batch_idx].astype(np.float32)
                y_batch = masks_ds[batch_idx].astype(np.float32)
                yield x_batch, y_batch

    ds = tf.data.Dataset.from_generator(
        batch_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
        ),
    )

    if split == 'train' and apply_aug:
        def augment_batch(x_batch, y_batch):
            aug_x, aug_y = tf.map_fn(
                lambda elems: augment_pair(elems[0], elems[1]),
                (x_batch, y_batch),
                fn_output_signature=(tf.float32, tf.float32),
            )
            return aug_x, aug_y

        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)

# ========================
# PREMIUM VISUALIZATION (TUMOR-AWARE)
# ========================

def visualize_augmentation_premium(hdf5_path, num_examples=4):
    """
    Searches HDF5 for a tumor slice and shows augmentation impact.
    Format: MRI (FLAIR) | Ground Truth (Binary Mask) | Overlay
    """
    print("=" * 60)
    print("PREMIUM AUGMENTATION VERIFICATION")
    print("=" * 60)
    
    with h5py.File(hdf5_path, 'r') as f:
        is_tumor = f['train/is_tumor'][:]
        tumor_indices = np.where(is_tumor)[0]
        
        if len(tumor_indices) == 0:
            print("⚠ No tumor slices found! Defaulting to index 0.")
            idx = 0
        else:
            idx = tumor_indices[len(tumor_indices)//2]
            print(f"✓ Found tumor at index {idx}. Visualizing...")
            
        img_orig = f['train/images'][idx]
        mask_orig = f['train/masks'][idx]

    fig, axes = plt.subplots(num_examples + 1, 3, figsize=(15, 4 * (num_examples + 1)))
    
    def plot_row(row_axes, img, mask, title_prefix=""):
        row_axes[0].imshow(img[..., 0], cmap='gray')
        row_axes[0].set_title(f"{title_prefix} MRI (FLAIR)")
        row_axes[0].axis('off')
        
        row_axes[1].imshow(mask[..., 0], cmap='gray')
        row_axes[1].set_title(f"{title_prefix} Binary Mask")
        row_axes[1].axis('off')
        
        row_axes[2].imshow(img[..., 0], cmap='gray')
        row_axes[2].imshow(mask[..., 0], cmap='Reds', alpha=0.5)
        row_axes[2].set_title(f"{title_prefix} Overlay")
        row_axes[2].axis('off')

    plot_row(axes[0], img_orig, mask_orig, "ORIGINAL")
    
    for i in range(num_examples):
        aug_img, aug_mask = augment_pair(
            tf.constant(img_orig, dtype=tf.float32),
            tf.constant(mask_orig, dtype=tf.float32)
        )
        plot_row(axes[i+1], aug_img.numpy(), aug_mask.numpy(), f"AUG {i+1}")

    plt.tight_layout()
    plt.show()
    
    print(f"\nMask Pixel Sum (Original): {np.sum(mask_orig):.1f}")
    print(f"Mask Pixel Sum (Aug 1):    {np.sum(aug_mask.numpy()):.1f}")
    print("=" * 60)

# ========================
# EXECUTION
# ========================
def main():
    print("=" * 60)
    print("CELL 6: DATA AUGMENTATION PIPELINE (FIXED + MEDICAL)")
    print("=" * 60)
    print("Augmentation enabled:")
    print(f"  - Rotation range: ±{ROTATION_RANGE}°")
    print(f"  - Horizontal flip: {FLIP_HORIZONTAL}")
    print(f"  - Intensity scaling: {INTENSITY_SCALE_RANGE}")
    print(f"  - Augmentation probability: {AUG_PROBABILITY * 100:.0f}%")
    print(f"  - Gaussian noise (stddev={NOISE_STDDEV})")
    print(f"  - Gamma correction: {GAMMA_RANGE}")
    print(f"  - Channel dropout: 10%")
    print("=" * 60)

    if os.path.exists(HDF5_PATH):
        print("\nLoading HDF5 dataset...")
        visualize_augmentation_premium(HDF5_PATH)
    else:
        print(f"HDF5 not found at {HDF5_PATH}. Run Cell 5 first.")

    print("\n✓ Cell 6 complete. Premium augmentation + medical augmentations ready.")


if __name__ == "__main__":
    main()
