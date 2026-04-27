import json
import os
import warnings

import h5py
import numpy as np
import tensorflow as tf

from .config import get_output_dirs, get_thresholds_path
from .data import HDF5Generator
from .losses import dice_coef, iou_metric, precision_metric, sensitivity_metric
from .train import get_train_custom_objects, load_model


DEFAULT_THRESHOLD = 0.5


def _finite_float(value, default=0.0):
    """Convert values to finite floats, falling back when NaN/Inf appears."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _sanitize_array(x):
    """Replace NaN/Inf in arrays before metric computation."""
    if np.all(np.isfinite(x)):
        return x
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def _resolve_eval_split(hdf5_path):
    with h5py.File(hdf5_path, "r") as handle:
        if "test/images" in handle and "test/masks" in handle:
            return "test"
        if "val/images" in handle and "val/masks" in handle:
            return "val"
    raise RuntimeError("HDF5 missing both test and val splits for evaluation.")


def _binary_segmentation_batch_metrics(y_true, y_prob, threshold=0.5, eps=1e-6):
    y_true = _sanitize_array(np.asarray(y_true, dtype=np.float32))
    y_prob = _sanitize_array(np.asarray(y_prob, dtype=np.float32))

    if y_true.ndim != 4:
        raise RuntimeError(f"Expected y_true rank-4, got shape={y_true.shape}")
    if y_prob.ndim != 4:
        raise RuntimeError(f"Expected y_prob rank-4, got shape={y_prob.shape}")

    if y_prob.shape[-1] != 1:
        y_prob = np.max(y_prob, axis=-1, keepdims=True)
    if y_true.shape[-1] != 1:
        y_true = np.max(y_true, axis=-1, keepdims=True)

    y_true = (y_true > 0.5).astype(np.float32)
    y_prob = np.clip(y_prob, eps, 1.0 - eps).astype(np.float32)
    y_hard = (y_prob > float(threshold)).astype(np.float32)

    axes = (1, 2, 3)
    tp = np.sum(y_hard * y_true, axis=axes)
    fp = np.sum(y_hard * (1.0 - y_true), axis=axes)
    fn = np.sum((1.0 - y_hard) * y_true, axis=axes)
    tn = np.sum((1.0 - y_hard) * (1.0 - y_true), axis=axes)

    precision_per = (tp + eps) / (tp + fp + eps)
    recall_per = (tp + eps) / (tp + fn + eps)
    f1_per = (2.0 * precision_per * recall_per + eps) / (precision_per + recall_per + eps)
    iou_per = (tp + eps) / (tp + fp + fn + eps)
    pixel_acc_per = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    inter_soft = np.sum(y_true * y_prob, axis=axes)
    union_soft = np.sum(y_true + y_prob, axis=axes)
    dice_soft_per = (2.0 * inter_soft + eps) / (union_soft + eps)

    dice_hard_per = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)

    bce_map = -(y_true * np.log(y_prob + eps) + (1.0 - y_true) * np.log(1.0 - y_prob + eps))
    test_loss = float(np.mean(bce_map))

    return {
        "dice_coef_soft": _finite_float(np.mean(dice_soft_per), default=0.0),
        "dice_coef_hard": _finite_float(np.mean(dice_hard_per), default=0.0),
        "precision": _finite_float(np.mean(precision_per), default=0.0),
        "recall": _finite_float(np.mean(recall_per), default=0.0),
        "sensitivity": _finite_float(np.mean(recall_per), default=0.0),
        "f1_score": _finite_float(np.mean(f1_per), default=0.0),
        "pixel_accuracy": _finite_float(np.mean(pixel_acc_per), default=0.0),
        "iou": _finite_float(np.mean(iou_per), default=0.0),
        "test_loss": _finite_float(test_loss, default=0.0),
    }


def _mask_batch_to_class_labels(y_batch, num_classes=1):
    y_batch = _sanitize_array(np.asarray(y_batch, dtype=np.float32))
    if y_batch.ndim != 4:
        raise RuntimeError(f"Expected segmentation masks rank-4, got {y_batch.shape}")
    has_tumor = (np.max(y_batch, axis=(1, 2, 3)) > 0.5)
    if int(num_classes) == 1:
        return has_tumor.astype(np.float32).reshape((-1, 1))
    return has_tumor.astype(np.int32)


def _classification_batch_metrics(y_true, y_prob, threshold=0.5, eps=1e-6):
    y_prob = _sanitize_array(np.asarray(y_prob, dtype=np.float32))
    y_true = np.asarray(y_true)

    if y_prob.ndim != 2:
        raise RuntimeError(f"Expected classifier output rank-2, got {y_prob.shape}")

    if y_prob.shape[-1] == 1:
        probs = np.clip(y_prob[:, 0], eps, 1.0 - eps)
        true = (y_true.reshape(-1) > 0.5).astype(np.int32)
        pred = (probs > float(threshold)).astype(np.int32)

        tp = np.sum((pred == 1) & (true == 1))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        tn = np.sum((pred == 0) & (true == 0))

        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        bce = -(true * np.log(probs + eps) + (1 - true) * np.log(1.0 - probs + eps))
        test_loss = float(np.mean(bce))
    else:
        probs = np.clip(y_prob, eps, 1.0)
        probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), eps, None)
        true = y_true.reshape(-1).astype(np.int32)
        pred = np.argmax(probs, axis=-1).astype(np.int32)

        accuracy = float(np.mean(pred == true))

        # Binary-positive class metrics when 2-class softmax; macro fallback otherwise.
        if probs.shape[-1] == 2:
            tp = np.sum((pred == 1) & (true == 1))
            fp = np.sum((pred == 1) & (true == 0))
            fn = np.sum((pred == 0) & (true == 1))
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
        else:
            classes = np.unique(np.concatenate([true, pred], axis=0))
            p_vals = []
            r_vals = []
            f_vals = []
            for cls_id in classes:
                tp = np.sum((pred == cls_id) & (true == cls_id))
                fp = np.sum((pred == cls_id) & (true != cls_id))
                fn = np.sum((pred != cls_id) & (true == cls_id))
                p_cls = (tp + eps) / (tp + fp + eps)
                r_cls = (tp + eps) / (tp + fn + eps)
                f_cls = (2.0 * p_cls * r_cls + eps) / (p_cls + r_cls + eps)
                p_vals.append(p_cls)
                r_vals.append(r_cls)
                f_vals.append(f_cls)
            precision = float(np.mean(p_vals)) if p_vals else 0.0
            recall = float(np.mean(r_vals)) if r_vals else 0.0
            f1 = float(np.mean(f_vals)) if f_vals else 0.0

        row_idx = np.arange(true.shape[0])
        ce = -np.log(np.clip(probs[row_idx, true], eps, 1.0))
        test_loss = float(np.mean(ce))

    return {
        "accuracy": _finite_float(accuracy, default=0.0),
        "pixel_accuracy": _finite_float(accuracy, default=0.0),
        "precision": _finite_float(precision, default=0.0),
        "recall": _finite_float(recall, default=0.0),
        "sensitivity": _finite_float(recall, default=0.0),
        "f1_score": _finite_float(f1, default=0.0),
        "test_loss": _finite_float(test_loss, default=0.0),
    }


def _primary_metric_name(task_type):
    return "f1_score" if task_type == "classification" else "dice_coef_soft"


def _load_binary_threshold(results_dir):
    thresholds_path = get_thresholds_path(results_dir=results_dir)
    if not os.path.exists(thresholds_path):
        return float(DEFAULT_THRESHOLD)

    try:
        with open(thresholds_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return float(DEFAULT_THRESHOLD)

    if isinstance(payload, dict):
        if "binary" in payload:
            return _finite_float(payload["binary"], default=DEFAULT_THRESHOLD)
    if isinstance(payload, (list, tuple)) and len(payload) >= 1:
        return _finite_float(payload[0], default=DEFAULT_THRESHOLD)

    return float(DEFAULT_THRESHOLD)


def evaluate_models(hdf5_path):
    _, model_dir, _, results_dir = get_output_dirs()
    split = _resolve_eval_split(hdf5_path)
    if split != "test":
        warnings.warn("test split not found; evaluating on val split instead.", RuntimeWarning)

    seg_gen = HDF5Generator(hdf5_path, split, batch_size=8, shuffle=False, target_mode="segmentation")
    threshold = _load_binary_threshold(results_dir)
    custom_objects = get_train_custom_objects()
    out = {}

    model_specs = [
        ("U-Net", "unet_best.keras", "segmentation"),
        ("Attention U-Net", "attention_unet_best.keras", "segmentation"),
        ("Attention U-Net + ViT (Proposed)", "attention_unet_vit_best.keras", "classification"),
    ]

    for key, filename, task_type in model_specs:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            continue
        model = load_model(path, custom_objects=custom_objects)

        eval_gen = seg_gen
        if task_type == "classification":
            output_shape = model.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            num_classes = int(output_shape[-1]) if output_shape is not None else 1
            if num_classes not in {1, 2}:
                warnings.warn(
                    f"Skipping {key}: unsupported classifier output channels={num_classes}.",
                    RuntimeWarning,
                )
                continue
            eval_gen = HDF5Generator(
                hdf5_path,
                split,
                batch_size=8,
                shuffle=False,
                target_mode="classification",
                classification_num_classes=num_classes,
            )

        metric_sums = {}
        total = 0
        for i in range(len(eval_gen)):
            x, y = eval_gen[i]
            pred = model.predict(x, verbose=0).astype(np.float32)
            if task_type == "classification":
                batch_metrics = _classification_batch_metrics(y_true=y, y_prob=pred, threshold=threshold)
            else:
                batch_metrics = _binary_segmentation_batch_metrics(y_true=y, y_prob=pred, threshold=threshold)

            bs = int(y.shape[0])
            total += bs
            for metric_name, metric_value in batch_metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + metric_value * bs

        out[key] = {k: v / max(1, total) for k, v in metric_sums.items()}
        out[key]["threshold"] = float(threshold)
        out[key]["num_samples"] = int(total)
        out[key]["task_type"] = task_type
        out[key]["primary_metric"] = _primary_metric_name(task_type)

        if task_type == "classification":
            with np.errstate(all="ignore"):
                output_shape = model.output_shape
                if isinstance(output_shape, list):
                    output_shape = output_shape[0]
                out[key]["num_output_classes"] = int(output_shape[-1]) if output_shape is not None else 1

        if task_type == "classification" and eval_gen is not seg_gen:
            eval_gen.close()

    seg_gen.close()
    save_path = os.path.join(results_dir, "comparison_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
