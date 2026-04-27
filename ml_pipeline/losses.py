import os

import numpy as np
import tensorflow as tf

SMOOTH = 1e-6
THRESHOLD = 0.5
POSITIVE_CLASS_WEIGHT = float(os.environ.get("BRATS_POS_WEIGHT", "1.0"))
TVERSKY_ALPHA = float(os.environ.get("BRATS_TVERSKY_ALPHA", "0.3"))
TVERSKY_BETA = float(os.environ.get("BRATS_TVERSKY_BETA", "0.7"))
BCE_WEIGHT = float(
    os.environ.get("BRATS_BCE_WEIGHT", os.environ.get("BRATS_CE_WEIGHT", "0.2"))
)


def _ensure_binary_channel(tensor):
    tensor = tf.cast(tensor, tf.float32)
    if tensor.shape.rank == 4 and tensor.shape[-1] not in (None, 1):
        tensor = tf.reduce_max(tensor, axis=-1, keepdims=True)
    return tensor


def soft_dice_loss(y_true, y_pred, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * inter + smooth) / (union + smooth))


def tversky_loss(y_true, y_pred, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=[1, 2, 3])
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(score)


def binary_ce_loss(y_true, y_pred, pos_weight=POSITIVE_CLASS_WEIGHT):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    weight = tf.cast(pos_weight, tf.float32)
    ce = -(weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(ce)


def combined_loss(y_true, y_pred, bce_weight=BCE_WEIGHT):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    return tversky_loss(y_true, y_pred) + bce_weight * binary_ce_loss(y_true, y_pred)


def focal_dice_loss(y_true, y_pred, gamma=0.75, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * inter + smooth) / (union + smooth)
    return tf.reduce_mean(tf.pow(1.0 - dice, gamma) * (1.0 - dice))


def dice_coef(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = _ensure_binary_channel(y_pred)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * inter + smooth) / (union + smooth))


def precision_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = tf.cast(_ensure_binary_channel(y_pred) > threshold, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    pp = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((tp + smooth) / (pp + smooth))


def sensitivity_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = tf.cast(_ensure_binary_channel(y_pred) > threshold, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    ap = tf.reduce_sum(y_true, axis=[1, 2, 3])
    return tf.reduce_mean((tp + smooth) / (ap + smooth))


def specificity_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = tf.cast(_ensure_binary_channel(y_pred) > threshold, tf.float32)
    tn = tf.reduce_sum((1.0 - y_pred) * (1.0 - y_true), axis=[1, 2, 3])
    an = tf.reduce_sum(1.0 - y_true, axis=[1, 2, 3])
    return tf.reduce_mean((tn + smooth) / (an + smooth))


def iou_metric(y_true, y_pred, threshold=THRESHOLD, smooth=SMOOTH):
    y_true = _ensure_binary_channel(y_true)
    y_pred = tf.cast(_ensure_binary_channel(y_pred) > threshold, tf.float32)
    inter = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    union = tf.reduce_sum(y_pred, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


def calculate_class_weights(hdf5_path, split="train"):
    try:
        import h5py
    except Exception:
        return tf.constant([1.0], dtype=tf.float32)

    if not os.path.exists(hdf5_path):
        return tf.constant([1.0], dtype=tf.float32)

    with h5py.File(hdf5_path, "r") as f:
        masks = f[f"{split}/masks"]
        n = int(masks.shape[0])
        if n == 0:
            return tf.constant([1.0], dtype=tf.float32)
        k = min(n, 500)
        idx = np.sort(np.random.choice(n, k, replace=False))
        y = masks[idx]

    if y.ndim == 4 and y.shape[-1] != 1:
        y = np.any(y > 0.5, axis=-1, keepdims=True).astype(np.float32)
    else:
        y = (y > 0.5).astype(np.float32)

    total = max(1.0, float(y.shape[0] * y.shape[1] * y.shape[2]))
    pos = float(np.sum(y > 0.5)) / total
    neg = 1.0 - pos
    w = float(np.clip(neg / max(pos, 1e-6), 1.0, 10.0))
    return tf.constant([w], dtype=tf.float32)


def set_dynamic_class_weights(hdf5_path, split="train"):
    global POSITIVE_CLASS_WEIGHT
    weights = calculate_class_weights(hdf5_path, split)
    POSITIVE_CLASS_WEIGHT = float(weights.numpy()[0])
    return weights


CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "soft_dice_loss": soft_dice_loss,
    "tversky_loss": tversky_loss,
    "binary_ce_loss": binary_ce_loss,
    "dice_coef": dice_coef,
    "precision_metric": precision_metric,
    "sensitivity_metric": sensitivity_metric,
    "specificity_metric": specificity_metric,
    "iou_metric": iou_metric,
    "focal_dice_loss": focal_dice_loss,
}
