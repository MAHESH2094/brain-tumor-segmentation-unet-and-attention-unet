import gc
import json
import math
import os
import time

import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.optimizers import Adam

from .config import PipelineConfig, get_output_dirs
from .data import HDF5Generator
from .losses import CUSTOM_OBJECTS as PIPELINE_CUSTOM_OBJECTS
from .losses import combined_loss, dice_coef, iou_metric, precision_metric, sensitivity_metric
from .models import build_unet, build_attention_unet_vit


def get_train_custom_objects():
    objects = dict(PIPELINE_CUSTOM_OBJECTS)
    objects.update(
        {
            "combined_loss": combined_loss,
            "dice_coef": dice_coef,
            "iou_metric": iou_metric,
            "precision_metric": precision_metric,
            "sensitivity_metric": sensitivity_metric,
        }
    )
    try:
        from cell_07d_attention_unet_vit_FIXED import get_attention_unet_vit_custom_objects

        objects.update(get_attention_unet_vit_custom_objects())
    except Exception:
        pass
    return objects


def _assert_val_has_positive_samples(hdf5_path):
    with h5py.File(hdf5_path, "r") as handle:
        if "val/masks" not in handle:
            raise RuntimeError("HDF5 missing required dataset: val/masks")
        masks = handle["val/masks"]
        if int(masks.shape[0]) == 0:
            raise RuntimeError("Validation split has zero samples.")

        scan_chunk = max(1, min(64, int(masks.shape[0])))
        has_positive = False
        for start in range(0, int(masks.shape[0]), scan_chunk):
            end = min(start + scan_chunk, int(masks.shape[0]))
            if (masks[start:end] > 0.5).any():
                has_positive = True
                break

        if not has_positive:
            raise RuntimeError(
                "Validation split contains zero tumor-positive pixels. "
                "Stop: metrics are not meaningful on an all-negative validation set."
            )


def _setup_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(), len(gpus)
    return tf.distribute.get_strategy(), len(gpus)


def _build_compile(model_type, cfg, strategy, mixed_precision=True, vit_num_classes=1):
    with strategy.scope():
        if model_type == "unet":
            model = build_unet(
                img_size=cfg.img_size,
                in_channels=cfg.num_channels,
                num_classes=cfg.num_classes,
                attention=False,
            )
            loss_fn = combined_loss
            metrics = [dice_coef, iou_metric, precision_metric, sensitivity_metric]
        elif model_type == "attention_unet":
            model = build_unet(
                img_size=cfg.img_size,
                in_channels=cfg.num_channels,
                num_classes=cfg.num_classes,
                attention=True,
            )
            loss_fn = combined_loss
            metrics = [dice_coef, iou_metric, precision_metric, sensitivity_metric]
        elif model_type == "attention_unet_vit":
            model = build_attention_unet_vit(
                img_size=cfg.img_size,
                in_channels=cfg.num_channels,
                num_classes=vit_num_classes,
            )
            if int(vit_num_classes) == 1:
                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                metrics = [
                    tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                    tf.keras.metrics.AUC(name="auc"),
                ]
            else:
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                metrics = [
                    tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
                ]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        opt = Adam(learning_rate=cfg.learning_rate, epsilon=1e-7, clipnorm=1.0)
        if mixed_precision:
            opt = LossScaleOptimizer(opt)

        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics,
        )
    return model


def run_dual_training(hdf5_path):
    cfg = PipelineConfig(
        epochs=int(os.environ.get("EPOCHS", "25")),
        steps_fraction=float(os.environ.get("STEPS_FRACTION", "1.0")),
        val_steps_fraction=float(os.environ.get("VAL_STEPS_FRACTION", "1.0")),
        batch_size_per_gpu=int(os.environ.get("BATCH_SIZE_PER_GPU", "16")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "4e-4")),
        patience_early_stopping=int(os.environ.get("PATIENCE_ES", "5")),
        patience_lr_reduce=int(os.environ.get("PATIENCE_LR", "2")),
        min_lr=float(os.environ.get("MIN_LR", "1e-6")),
    )

    _, model_dir, log_dir, results_dir = get_output_dirs()
    _assert_val_has_positive_samples(hdf5_path)
    strategy, num_gpus = _setup_strategy()
    mixed_precision = num_gpus > 0
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    enable_vit_classifier = os.environ.get("BRATS_ENABLE_VIT_CLASSIFIER", "0") == "1"
    vit_num_classes = int(os.environ.get("BRATS_VIT_NUM_CLASSES", "1"))
    if enable_vit_classifier and vit_num_classes not in {1, 2}:
        raise ValueError(
            f"Invalid BRATS_VIT_NUM_CLASSES='{vit_num_classes}'. Supported values: 1 or 2."
        )

    global_bs = cfg.batch_size_per_gpu * max(1, num_gpus)
    summary = {}

    model_specs = ["unet", "attention_unet"]
    if enable_vit_classifier:
        model_specs.append("attention_unet_vit")

    for model_name in model_specs:
        is_classifier = model_name == "attention_unet_vit"
        target_mode = "classification" if is_classifier else "segmentation"

        train_gen = HDF5Generator(
            hdf5_path,
            "train",
            global_bs,
            shuffle=True,
            target_mode=target_mode,
            classification_num_classes=vit_num_classes,
        )
        val_gen = HDF5Generator(
            hdf5_path,
            "val",
            global_bs,
            shuffle=False,
            target_mode=target_mode,
            classification_num_classes=vit_num_classes,
        )
        steps_per_epoch = min(len(train_gen), max(1, math.ceil(len(train_gen) * cfg.steps_fraction)))
        val_steps = min(len(val_gen), max(1, math.ceil(len(val_gen) * cfg.val_steps_fraction)))

        model = _build_compile(
            model_name,
            cfg,
            strategy,
            mixed_precision=mixed_precision,
            vit_num_classes=vit_num_classes,
        )

        monitor_metric = (
            "val_binary_accuracy"
            if (is_classifier and vit_num_classes == 1)
            else ("val_sparse_categorical_accuracy" if is_classifier else "val_dice_coef")
        )
        ckpt = os.path.join(model_dir, f"{model_name}_best.keras")
        callbacks = [
            TerminateOnNaN(),
            ModelCheckpoint(ckpt, monitor=monitor_metric, mode="max", save_best_only=True, verbose=1),
            EarlyStopping(
                monitor=monitor_metric,
                mode="max",
                patience=cfg.patience_early_stopping,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor=monitor_metric,
                mode="max",
                factor=cfg.lr_reduce_factor,
                patience=cfg.patience_lr_reduce,
                min_lr=cfg.min_lr,
                verbose=1,
            ),
            CSVLogger(os.path.join(log_dir, f"{model_name}_log.csv")),
        ]
        start = time.time()
        hist = model.fit(
            train_gen,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - start
        train_gen.close()
        val_gen.close()

        hist_payload = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        with open(os.path.join(results_dir, f"{model_name}_history.json"), "w", encoding="utf-8") as f:
            json.dump(hist_payload, f, indent=2)

        summary[model_name] = {
            "checkpoint": ckpt,
            "primary_metric": monitor_metric,
            "best_val_metric": float(max(hist.history.get(monitor_metric, [0.0]))),
            "runtime_minutes": float(elapsed / 60.0),
        }

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    with open(os.path.join(results_dir, "dual_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_model(model_path, custom_objects=None):
    objects = get_train_custom_objects()
    if custom_objects:
        objects.update(custom_objects)
    return tf.keras.models.load_model(
        model_path,
        custom_objects=objects,
        compile=False,
    )
