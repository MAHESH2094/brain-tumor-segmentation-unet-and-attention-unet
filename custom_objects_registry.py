# ===================================================
# CUSTOM OBJECTS & REGISTRY (BINARY)
# ===================================================

import importlib
import importlib.util
import os
import sys
from contextlib import suppress

import tensorflow as tf


_REQUIRED_SYMBOLS = [
    "combined_loss",
    "dice_coef",
]

_OPTIONAL_SYMBOLS = [
    "soft_dice_loss",
    "tversky_loss",
    "binary_ce_loss",
    "precision_metric",
    "sensitivity_metric",
    "specificity_metric",
    "iou_metric",
    "focal_dice_loss",
    "calculate_class_weights",
    "set_dynamic_class_weights",
]


def _load_cell8_module():
    with suppress(Exception):
        return importlib.import_module("cell_08_loss_metrics_FIXED")

    candidates = []
    current_file = globals().get("__file__")
    if current_file:
        candidates.append(
            os.path.join(os.path.dirname(os.path.abspath(current_file)), "cell_08_loss_metrics_FIXED.py")
        )
    candidates.append(os.path.join(os.getcwd(), "cell_08_loss_metrics_FIXED.py"))

    for path in candidates:
        if not os.path.exists(path):
            continue
        with suppress(Exception):
            spec = importlib.util.spec_from_file_location("cell_08_loss_metrics_FIXED", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return None


def _resolve_symbols():
    resolved = {}

    cell8_module = _load_cell8_module()
    if cell8_module is not None:
        for name in _REQUIRED_SYMBOLS + _OPTIONAL_SYMBOLS:
            if hasattr(cell8_module, name):
                resolved[name] = getattr(cell8_module, name)
        if all(name in resolved for name in _REQUIRED_SYMBOLS):
            return resolved

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name in _REQUIRED_SYMBOLS + _OPTIONAL_SYMBOLS:
            if hasattr(main_module, name):
                resolved[name] = getattr(main_module, name)
        if all(name in resolved for name in _REQUIRED_SYMBOLS):
            return resolved

    with suppress(Exception):
        from ml_pipeline import losses as pipeline_losses

        for name in _REQUIRED_SYMBOLS + _OPTIONAL_SYMBOLS:
            if hasattr(pipeline_losses, name):
                resolved[name] = getattr(pipeline_losses, name)

    if not all(name in resolved for name in _REQUIRED_SYMBOLS):
        missing = [name for name in _REQUIRED_SYMBOLS if name not in resolved]
        raise RuntimeError(
            "Could not resolve required binary symbols for custom object registry. "
            "Run Cell 8 first, or ensure cell_08_loss_metrics_FIXED.py is available. "
            f"Missing: {missing}"
        )

    return resolved


def _fallback_precision_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    pp = tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((tp + smooth) / (pp + smooth))


def _fallback_sensitivity_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    ap = tf.reduce_sum(y_true, axis=[1, 2, 3])
    return tf.reduce_mean((tp + smooth) / (ap + smooth))


def _fallback_iou_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    inter = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
    union = tf.reduce_sum(y_pred, axis=[1, 2, 3]) + tf.reduce_sum(y_true, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


_resolved_symbols = _resolve_symbols()

combined_loss = _resolved_symbols["combined_loss"]
dice_coef = _resolved_symbols["dice_coef"]
soft_dice_loss = _resolved_symbols.get("soft_dice_loss", None)
tversky_loss = _resolved_symbols.get("tversky_loss", None)
binary_ce_loss = _resolved_symbols.get("binary_ce_loss", None)
precision_metric = _resolved_symbols.get("precision_metric", _fallback_precision_metric)
sensitivity_metric = _resolved_symbols.get("sensitivity_metric", _fallback_sensitivity_metric)
specificity_metric = _resolved_symbols.get("specificity_metric", None)
iou_metric = _resolved_symbols.get("iou_metric", _fallback_iou_metric)
focal_dice_loss = _resolved_symbols.get("focal_dice_loss", None)
calculate_class_weights = _resolved_symbols.get("calculate_class_weights", None)
set_dynamic_class_weights = _resolved_symbols.get("set_dynamic_class_weights", None)


CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "dice_coef": dice_coef,
    "precision_metric": precision_metric,
    "sensitivity_metric": sensitivity_metric,
    "iou_metric": iou_metric,
}

if soft_dice_loss is not None:
    CUSTOM_OBJECTS["soft_dice_loss"] = soft_dice_loss
if tversky_loss is not None:
    CUSTOM_OBJECTS["tversky_loss"] = tversky_loss
if binary_ce_loss is not None:
    CUSTOM_OBJECTS["binary_ce_loss"] = binary_ce_loss
if specificity_metric is not None:
    CUSTOM_OBJECTS["specificity_metric"] = specificity_metric
if focal_dice_loss is not None:
    CUSTOM_OBJECTS["focal_dice_loss"] = focal_dice_loss
if calculate_class_weights is not None:
    CUSTOM_OBJECTS["calculate_class_weights"] = calculate_class_weights
if set_dynamic_class_weights is not None:
    CUSTOM_OBJECTS["set_dynamic_class_weights"] = set_dynamic_class_weights

try:
    from cell_07a_building_blocks_FIXED import _hierarchy_sort, hierarchy_constraint_layer

    CUSTOM_OBJECTS["_hierarchy_sort"] = _hierarchy_sort
    CUSTOM_OBJECTS["hierarchy_constraint_layer"] = hierarchy_constraint_layer
except Exception:
    pass

try:
    from cell_07d_attention_unet_vit_FIXED import get_attention_unet_vit_custom_objects

    CUSTOM_OBJECTS.update(get_attention_unet_vit_custom_objects())
except Exception:
    pass

try:
    from cell_07d_attention_unet_vit_FIXED import (
        AddPositionEmbedding,
        ClassToken,
        TokenExtractor,
    )

    CUSTOM_OBJECTS["ClassToken"] = ClassToken
    CUSTOM_OBJECTS["AddPositionEmbedding"] = AddPositionEmbedding
    CUSTOM_OBJECTS["TokenExtractor"] = TokenExtractor
except Exception:
    pass


def get_custom_objects():
    return CUSTOM_OBJECTS


def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )


print("=" * 70)
print("CUSTOM OBJECTS REGISTRY LOADED (BINARY)")
print("=" * 70)
print(f"Total custom objects: {len(CUSTOM_OBJECTS)}")
for name in sorted(CUSTOM_OBJECTS.keys()):
    print(f"  - {name}")
print("=" * 70)
