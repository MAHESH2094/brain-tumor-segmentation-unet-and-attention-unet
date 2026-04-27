import os
from glob import glob

import cv2
import nibabel as nib
import numpy as np


def create_binary_mask(seg_slice):
    seg = np.asarray(seg_slice, dtype=np.float32)
    return (seg > 0).astype(np.float32)[..., np.newaxis]


def compute_volume_stats(volume):
    """Compute mean/std on non-zero voxels for volume-level normalization."""
    volume = volume.astype(np.float32)
    mask = volume > 0
    if np.sum(mask) == 0:
        return {"mean": 0.0, "std": 1.0}
    values = volume[mask]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def normalize_image(image, volume_stats=None, eps=1e-8):
    image = image.astype(np.float32)
    mask = image > 0
    if np.sum(mask) == 0:
        return np.zeros_like(image, dtype=np.float32)

    if volume_stats is not None:
        mean = float(volume_stats.get("mean", 0.0))
        std = float(volume_stats.get("std", 0.0))
    else:
        values = image[mask]
        mean, std = float(np.mean(values)), float(np.std(values))

    std = max(std, eps)
    out = np.zeros_like(image, dtype=np.float32)
    out[mask] = (image[mask] - mean) / std
    return out


def load_multimodal_volume(patient_dir, require_seg=True):
    volumes = {}
    affine = None
    header = None
    for mod in ["flair", "t1", "t1ce", "t2"]:
        files = glob(os.path.join(patient_dir, f"*{mod}*.nii*"))
        if mod == "t1ce" and not files:
            files = glob(os.path.join(patient_dir, "*t1Gd*.nii*"))
        if mod == "t1" and not files:
            all_t1 = glob(os.path.join(patient_dir, "*t1*.nii*"))
            files = [f for f in all_t1 if "ce" not in f.lower() and "gd" not in f.lower()]
        if not files:
            return None, None, None, None
        nii = nib.load(files[0])
        volumes[mod] = nii.get_fdata(dtype=np.float32)
        if affine is None:
            affine = nii.affine
            header = nii.header
    seg = None
    if require_seg:
        seg_files = glob(os.path.join(patient_dir, "*seg*.nii*"))
        if not seg_files:
            return None, None, None, None
        seg = nib.load(seg_files[0]).get_fdata(dtype=np.float32)
    return volumes, seg, affine, header


def preprocess_multimodal_slice(modality_slices, seg_slice, img_size=None, volume_stats=None):
    if img_size is None:
        img_size = int(os.environ.get("BRATS_IMG_SIZE", "128"))
    channels = []
    for mod in ["flair", "t1", "t1ce", "t2"]:
        x = cv2.GaussianBlur(modality_slices[mod].astype(np.float32), (3, 3), 0)
        stats = volume_stats.get(mod) if volume_stats is not None else None
        x = normalize_image(x, volume_stats=stats)
        channels.append(x)
    image = np.stack(channels, axis=-1)
    if image.shape[0] != int(img_size) or image.shape[1] != int(img_size):
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        image = image.astype(np.float32)

    mask = create_binary_mask(seg_slice)
    if mask.shape[0] != int(img_size) or mask.shape[1] != int(img_size):
        resized = cv2.resize(mask[..., 0], (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = resized[..., np.newaxis].astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    return image, mask


def is_nonempty_brain_slice(slice2d, min_nonzero_ratio=0.01):
    """Robust filter used during inference; avoids percentile/max bug."""
    return float(np.mean(slice2d > 0)) >= min_nonzero_ratio

