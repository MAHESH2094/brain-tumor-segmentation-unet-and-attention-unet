# ===================================================
# POSTPROCESSING: Binary Segmentation Utilities
# ===================================================
# Purpose: Standardize predictions to one tumor-probability channel and
#          apply thresholding for tumor-vs-background segmentation.

import numpy as np
import tensorflow as tf


def enforce_binary_channel_numpy(predictions):
    """Return numpy predictions with shape (..., 1) for binary segmentation."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got shape {predictions.shape}")

    if predictions.shape[-1] == 1:
        return predictions
    return np.max(predictions, axis=-1, keepdims=True).astype(np.float32)


@tf.function
def enforce_binary_channel_tf(predictions):
    """Return tensor predictions with shape (..., 1) for binary segmentation."""
    predictions = tf.cast(predictions, tf.float32)
    if predictions.shape.rank is not None and predictions.shape.rank < 3:
        raise ValueError(f"Expected rank >= 3, got {predictions.shape}")
    if predictions.shape[-1] == 1:
        return predictions
    return tf.reduce_max(predictions, axis=-1, keepdims=True)


def enforce_tumor_hierarchy_numpy(predictions):
    """Backward-compatible wrapper: now normalizes to a binary channel."""
    return enforce_binary_channel_numpy(predictions)


@tf.function
def enforce_tumor_hierarchy_tf(predictions):
    """Backward-compatible wrapper: now normalizes to a binary channel."""
    return enforce_binary_channel_tf(predictions)


def postprocess_segmentation(pred, enforce_hierarchy=True, threshold=0.5):
    """Threshold probabilities into binary tumor-vs-background masks."""
    _ = enforce_hierarchy
    probs = enforce_binary_channel_numpy(pred)
    return (probs > float(threshold)).astype(np.uint8)


def print_hierarchy_violations(pred, threshold=0.5):
    """Backward-compatible diagnostic: reports binary occupancy statistics."""
    binary = postprocess_segmentation(pred, threshold=threshold)
    total_pixels = int(binary.size)
    tumor_pixels = int(np.sum(binary > 0))
    background_pixels = total_pixels - tumor_pixels
    tumor_ratio = (tumor_pixels / total_pixels) if total_pixels else 0.0

    stats = {
        'total_pixels': total_pixels,
        'tumor_pixels': tumor_pixels,
        'background_pixels': background_pixels,
        'tumor_ratio': float(tumor_ratio),
    }

    print("\n" + "=" * 60)
    print("BINARY SEGMENTATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Tumor pixels: {stats['tumor_pixels']:,}")
    print(f"Background pixels: {stats['background_pixels']:,}")
    print(f"Tumor ratio: {stats['tumor_ratio'] * 100.0:.2f}%")
    print("=" * 60)

    return stats
