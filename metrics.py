# ===================================================
# CELL 8: Loss Functions & Metrics (BINARY) (FIXED)
# ===================================================
# Purpose: Binary whole-tumor losses and metrics for 1-channel masks.
# FIXES:
# - Replaces legacy multi-channel objectives with binary Tversky + BCE
# - Keeps backward-compatible aliases expected by downstream cells (aliases map to binary Dice)
# - Simplifies threshold tuning to a single binary threshold
# - Added run_smoke_tests()

import os
import tensorflow as tf
import numpy as np

NUM_CLASSES = 1
THRESHOLD = 0.5
SMOOTH = 1e-6
EPSILON = SMOOTH
TVERSKY_ALPHA = float(os.environ.get("BRATS_TVERSKY_ALPHA", "0.3"))
TVERSKY_BETA = float(os.environ.get("BRATS_TVERSKY_BETA", "0.7"))
BCE_WEIGHT = float(
    os.environ.get("BRATS_BCE_WEIGHT", os.environ.get("BRATS_CE_WEIGHT", "0.2"))
)


def _ensure_binary_channel(tensor):
    """Collapse multi-channel masks/preds to one binary channel when needed."""
    tensor = tf.cast(tensor, tf.float32)
    if tensor.shape.rank == 4 and tensor.shape[-1] not in (None, 1):
        tensor = tf.reduce_max(tensor, axis=-1, keepdims=True)
    return tensor


# ========================
# CLASS WEIGHT HOOKS (BINARY)
# ========================
def calculate_class_weights(hdf5_path, split='train'):
    """
    Compute an optional positive-class weight from binary masks.
    Returns shape [1] for compatibility with older callers.
    """
    try:
        import h5py
    except ImportError:
        print("[WARN] h5py not available, using default class weight=1.0")
        return tf.constant([1.0], dtype=tf.float32)

    if not os.path.exists(hdf5_path):
        print(f"[WARN] HDF5 not found at {hdf5_path}, using default class weight=1.0")
        return tf.constant([1.0], dtype=tf.float32)

    with h5py.File(hdf5_path, 'r') as f:
        masks = f[f'{split}/masks']
        n_samples = int(masks.shape[0])
        sample_size = min(n_samples, 500)
        if sample_size == 0:
            return tf.constant([1.0], dtype=tf.float32)

        indices = np.random.choice(n_samples, sample_size, replace=False)
        indices = np.sort(indices)
        sampled_masks = masks[indices]

    if sampled_masks.ndim == 4 and sampled_masks.shape[-1] != 1:
        sampled_masks = np.any(sampled_masks > 0.5, axis=-1, keepdims=True).astype(np.float32)
    else:
        sampled_masks = (sampled_masks > 0.5).astype(np.float32)

    total_pixels = sampled_masks.shape[0] * sampled_masks.shape[1] * sampled_masks.shape[2]
    positive_pixels = float(np.sum(sampled_masks > 0.5))
    positive_frequency = positive_pixels / max(1.0, float(total_pixels))
    negative_frequency = 1.0 - positive_frequency

    positive_weight = negative_frequency / max(positive_frequency, 1e-6)
    positive_weight = float(np.clip(positive_weight, 1.0, 10.0))

    print(f"Binary class frequencies: pos={positive_frequency:.4f}, neg={negative_frequency:.4f}")
    print(f"Binary positive class weight: {positive_weight:.2f}")
    return tf.constant([positive_weight], dtype=tf.float32)


CLASS_WEIGHTS = tf.constant([1.0], dtype=tf.float32)


def set_dynamic_class_weights(hdf5_path, split='train'):
    """Optional updater for backward compatibility with older training code."""
    global CLASS_WEIGHTS
    CLASS_WEIGHTS = calculate_class_weights(hdf5_path, split)
    print(f"[OK] CLASS_WEIGHTS updated to: {CLASS_WEIGHTS.numpy()}")


# ========================
# LOSS FUNCTIONS (BINARY)
# ========================
def soft_dice_loss(y_true, y_pred, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def tversky_loss(y_true, y_pred, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=[1, 2, 3])

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(tversky)


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def combined_loss(y_true, y_pred):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tversky_loss(y_true, y_pred) + BCE_WEIGHT * bce(y_true, y_pred)


def binary_ce_loss(y_true, y_pred):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)
    pos_weight = tf.cast(CLASS_WEIGHTS[0], tf.float32)
    loss = -(pos_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def focal_dice_loss(y_true, y_pred, gamma=0.75, smooth=SMOOTH):
    """Compatibility shim for older imports."""
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    focal_weight = tf.pow(1.0 - dice, gamma)
    return tf.reduce_mean(focal_weight * (1.0 - dice))


# ========================
# EVALUATION METRICS (BINARY)
# ========================
def dice_coef(y_true, y_pred, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > THRESHOLD, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def dice_et(y_true, y_pred, smooth=SMOOTH):
    """Legacy alias kept for checkpoint compatibility in binary mode."""
    return dice_coef(y_true, y_pred, smooth=smooth)


def dice_tc(y_true, y_pred, smooth=SMOOTH):
    """Legacy alias kept for checkpoint compatibility in binary mode."""
    return dice_coef(y_true, y_pred, smooth=smooth)


def dice_wt(y_true, y_pred, smooth=SMOOTH):
    """Legacy alias kept for checkpoint compatibility in binary mode."""
    return dice_coef(y_true, y_pred, smooth=smooth)


def mean_dice(y_true, y_pred, smooth=SMOOTH):
    """Legacy alias kept for checkpoint compatibility in binary mode."""
    return dice_coef(y_true, y_pred, smooth=smooth)


def precision_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positive = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    predicted_positive = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((true_positive + smooth) / (predicted_positive + smooth))


def sensitivity_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positive = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    actual_positive = tf.reduce_sum(y_true, axis=[1, 2, 3])
    return tf.reduce_mean((true_positive + smooth) / (actual_positive + smooth))


def iou_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    union = tf.reduce_sum(y_pred, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def specificity_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_negative = tf.reduce_sum((1.0 - y_pred) * (1.0 - y_true), axis=[1, 2, 3])
    actual_negative = tf.reduce_sum(1.0 - y_true, axis=[1, 2, 3])
    return tf.reduce_mean((true_negative + smooth) / (actual_negative + smooth))


recall_metric = sensitivity_metric
jaccard_metric = iou_metric


def get_metrics():
    return [dice_coef, precision_metric, sensitivity_metric, specificity_metric, iou_metric]


def get_loss():
    return combined_loss


# ========================
# ADAPTIVE THRESHOLD TUNING (BINARY)
# ========================
def find_optimal_thresholds(model, val_generator, num_batches=20):
    """Find one binary threshold that maximizes validation Dice."""
    all_pred = []
    all_true = []
    for i in range(min(num_batches, len(val_generator))):
        x_batch, y_batch = val_generator[i]
        pred_batch = model.predict(x_batch, batch_size=len(x_batch), verbose=0)
        all_pred.append(pred_batch)
        all_true.append(y_batch)

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    if y_true.ndim == 4 and y_true.shape[-1] != 1:
        y_true = np.any(y_true > 0.5, axis=-1, keepdims=True).astype(np.float32)
    else:
        y_true = y_true.astype(np.float32)

    if y_pred.ndim == 4 and y_pred.shape[-1] != 1:
        y_pred = np.max(y_pred, axis=-1, keepdims=True)

    best_threshold = 0.5
    best_dice = 0.0

    for threshold in np.arange(0.10, 0.95, 0.05):
        y_pred_bin = (y_pred > threshold).astype(np.float32)
        inter = np.sum(y_pred_bin * y_true)
        union = np.sum(y_pred_bin) + np.sum(y_true)
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)

        if dice > best_dice:
            best_dice = float(dice)
            best_threshold = float(threshold)

    print(f"[OK] Optimal binary threshold: {best_threshold:.2f} (Dice={best_dice:.4f})")
    return [best_threshold]


def run_smoke_tests():
    """Minimal synthetic-data smoke tests for loss/metrics correctness."""
    y_true = tf.cast(tf.random.uniform((2, 32, 32, NUM_CLASSES), 0, 2, dtype=tf.int32), tf.float32)
    y_pred = tf.random.uniform((2, 32, 32, NUM_CLASSES), 0.0, 1.0, dtype=tf.float32)

    loss = combined_loss(y_true, y_pred)
    dc = dice_coef(y_true, y_pred)
    prec = precision_metric(y_true, y_pred)

    assert tf.math.is_finite(loss), 'Loss is not finite'
    assert 0.0 <= float(dc) <= 1.0, 'Dice out of range'
    assert 0.0 <= float(prec) <= 1.0, 'Precision out of range'
    print('[OK] Cell 8 smoke tests passed')


run_smoke_tests()
print('[OK] Cell 8 fixed and ready (binary mode).')
print(f'  TVERSKY_ALPHA={TVERSKY_ALPHA} | TVERSKY_BETA={TVERSKY_BETA}')
print(f'  BCE_WEIGHT={BCE_WEIGHT}')
print(f'  CLASS_WEIGHTS={CLASS_WEIGHTS.numpy()}')
