import json
import os
import warnings

import cv2
import nibabel as nib
import numpy as np

from .config import get_output_dirs, get_thresholds_path
from .losses import CUSTOM_OBJECTS as PIPELINE_CUSTOM_OBJECTS
from .preprocessing import (
    compute_volume_stats,
    is_nonempty_brain_slice,
    load_multimodal_volume,
    preprocess_multimodal_slice,
)

try:
    from scipy import ndimage
except Exception:
    ndimage = None


DEFAULT_THRESHOLD = 0.5


def _get_custom_objects():
    try:
        from custom_objects_registry import get_custom_objects

        return get_custom_objects()
    except Exception:
        return PIPELINE_CUSTOM_OBJECTS


def _load_binary_threshold(results_dir=None):
    thresholds_path = get_thresholds_path(results_dir=results_dir)
    if not os.path.exists(thresholds_path):
        return float(DEFAULT_THRESHOLD)

    try:
        with open(thresholds_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        warnings.warn(
            f"Could not read thresholds file '{thresholds_path}': {exc}. Falling back to default.",
            RuntimeWarning,
        )
        return float(DEFAULT_THRESHOLD)

    if isinstance(payload, dict):
        if "binary" in payload:
            return float(payload["binary"])
    if isinstance(payload, (list, tuple)) and len(payload) >= 1:
        return float(payload[0])
    return float(DEFAULT_THRESHOLD)


def load_best_model(task="segmentation"):
    _, model_dir, _, _ = get_output_dirs()
    vit = os.path.join(model_dir, "attention_unet_vit_best.keras")
    attn = os.path.join(model_dir, "attention_unet_best.keras")
    unet = os.path.join(model_dir, "unet_best.keras")
    custom_objects = _get_custom_objects()

    if str(task).strip().lower() == "classification" and os.path.exists(vit):
        import tensorflow as tf

        return tf.keras.models.load_model(
            vit,
            custom_objects=custom_objects,
            compile=False,
        )

    if os.path.exists(attn):
        import tensorflow as tf

        return tf.keras.models.load_model(
            attn,
            custom_objects=custom_objects,
            compile=False,
        )
    if os.path.exists(unet):
        import tensorflow as tf

        return tf.keras.models.load_model(
            unet,
            custom_objects=custom_objects,
            compile=False,
        )
    return None


def preprocess_patient(patient_dir):
    volumes, _, affine, header = load_multimodal_volume(patient_dir, require_seg=False)
    if volumes is None:
        return None, None, None, None, None

    volume_stats = {
        mod: compute_volume_stats(volumes[mod])
        for mod in ["flair", "t1", "t1ce", "t2"]
    }

    n = volumes["flair"].shape[2]
    imgs, idxs = [], []
    for i in range(n):
        flair = volumes["flair"][:, :, i]
        if not is_nonempty_brain_slice(flair, min_nonzero_ratio=0.01):
            continue
        modality = {m: volumes[m][:, :, i] for m in ["flair", "t1", "t1ce", "t2"]}
        img, _ = preprocess_multimodal_slice(
            modality,
            np.zeros_like(flair),
            volume_stats=volume_stats,
        )
        imgs.append(img)
        idxs.append(i)
    if not imgs:
        return None, None, None, None, None
    return np.array(imgs, dtype=np.float32), idxs, affine, header, volumes["flair"].shape


def postprocess(pred, threshold=None, min_tumor_size=50):
    thr = float(_load_binary_threshold() if threshold is None else threshold)

    if pred.ndim != 4:
        raise ValueError(f"Expected prediction shape (N,H,W,C), got {pred.shape}")
    if pred.shape[-1] != 1:
        pred = np.max(pred, axis=-1, keepdims=True)

    out = (pred > thr).astype(np.float32)

    if ndimage is None:
        warnings.warn(
            "scipy.ndimage unavailable; skipping connected-component cleanup.",
            RuntimeWarning,
        )
        return out

    for i in range(out.shape[0]):
        labeled, count = ndimage.label(out[i, :, :, 0])
        for comp in range(1, count + 1):
            comp_mask = labeled == comp
            if np.sum(comp_mask) < min_tumor_size:
                out[i, :, :, 0][comp_mask] = 0
    return out


def reconstruct_3d(pred, slice_indices, original_shape):
    volume = np.zeros((original_shape[0], original_shape[1], original_shape[2], 1), dtype=np.float32)
    for i, s_idx in enumerate(slice_indices):
        x = pred[i]
        if x.shape[:2] != original_shape[:2]:
            resized = cv2.resize(
                x[:, :, 0],
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            x = resized[..., np.newaxis]
        volume[:, :, s_idx, :] = x
    return volume


def save_nifti(volume, affine, header, out_path, threshold=None):
    thr = float(_load_binary_threshold() if threshold is None else threshold)
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume (H,W,D,1), got {volume.shape}")

    if volume.shape[-1] != 1:
        volume = np.max(volume, axis=-1, keepdims=True)

    labels = (volume[..., 0] > thr).astype(np.uint8)
    nib.save(nib.Nifti1Image(labels, affine, header), out_path)
