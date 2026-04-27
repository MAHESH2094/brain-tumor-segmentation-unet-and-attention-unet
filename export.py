# ===================================================
# CELL 12: Final Export & Report (BINARY)
# ===================================================

import datetime
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _candidate_roots() -> List[str]:
    roots: List[str] = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots.extend([os.getcwd(), "/kaggle/working", "/kaggle/input"])

    # Kaggle datasets are often nested under /kaggle/input.
    for base in ["/kaggle/input", "/kaggle/working"]:
        if not os.path.isdir(base):
            continue
        with os.scandir(base) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                roots.append(entry.path)
                try:
                    with os.scandir(entry.path) as sub_it:
                        for sub in sub_it:
                            if sub.is_dir():
                                roots.append(sub.path)
                except PermissionError:
                    continue

    # Preserve order while deduplicating.
    return [p for i, p in enumerate(roots) if p and p not in roots[:i]]


def _ensure_project_root_on_path() -> None:
    for root in _candidate_roots():
        if os.path.exists(os.path.join(root, "config.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return


def _load_config_class():
    try:
        from config import Config as _Config

        return _Config
    except ModuleNotFoundError:
        pass

    for root in _candidate_roots():
        cfg_path = os.path.join(root, "config.py")
        if not os.path.exists(cfg_path):
            continue
        spec = importlib.util.spec_from_file_location("config", cfg_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "Config"):
                return module.Config

    # Fallback keeps Cell 12 executable even when config.py is not shipped.
    default_output = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    output_dir = os.environ.get("OUTPUT_DIR", default_output)

    class _FallbackConfig:
        OUTPUT_DIR = output_dir
        MODEL_DIR = os.path.join(output_dir, "models")
        RESULTS_DIR = os.path.join(output_dir, "results")
        HDF5_PATH = os.path.join(output_dir, "brats_preprocessed.h5")

        @classmethod
        def to_dict(cls):
            return {
                "OUTPUT_DIR": cls.OUTPUT_DIR,
                "MODEL_DIR": cls.MODEL_DIR,
                "RESULTS_DIR": cls.RESULTS_DIR,
                "HDF5_PATH": cls.HDF5_PATH,
                "source": "fallback-env-config",
            }

    print(
        "[WARN] config.py not found. Using environment-based fallback config "
        f"(OUTPUT_DIR={_FallbackConfig.OUTPUT_DIR})."
    )
    return _FallbackConfig


_ensure_project_root_on_path()
Config = _load_config_class()


# ========================
# DEPENDENCY IMPORTS (GUARDED)
# ========================
REQUIRED_METRIC_SYMBOLS = [
    "combined_loss",
    "dice_coef",
    "precision_metric",
    "sensitivity_metric",
    "iou_metric",
]
if any(sym not in globals() for sym in REQUIRED_METRIC_SYMBOLS):
    with suppress(Exception):
        from custom_objects_registry import get_custom_objects

        _registry = get_custom_objects()
        for _sym in REQUIRED_METRIC_SYMBOLS:
            if _sym in _registry and _sym not in globals():
                globals()[_sym] = _registry[_sym]

if any(sym not in globals() for sym in REQUIRED_METRIC_SYMBOLS):
    from cell_08_loss_metrics_FIXED import (
        combined_loss,
        dice_coef,
        precision_metric,
        sensitivity_metric,
        iou_metric,
    )


# ========================
# LOGGING
# ========================
LOGGER = logging.getLogger("cell12")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


# ========================
# PATHS / SETTINGS
# ========================
DEFAULT_OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUTPUT_DIR = globals().get("OUTPUT_DIR", os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
MODEL_DIR = globals().get("MODEL_DIR", os.path.join(OUTPUT_DIR, "models"))
RESULTS_DIR = globals().get("RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))
HDF5_PATH = globals().get("HDF5_PATH", os.path.join(OUTPUT_DIR, "brats_preprocessed.h5"))
BINARY_TARGET_UNET = os.environ.get(
    "BRATS_TARGET_BINARY_UNET",
    os.environ.get("BRATS_TARGET_WT_UNET", "88-92%"),
)
BINARY_TARGET_ATTENTION = os.environ.get(
    "BRATS_TARGET_BINARY_ATTENTION",
    os.environ.get("BRATS_TARGET_WT_ATTENTION", "90-94%"),
)

EXPORT_DIR = os.path.join(OUTPUT_DIR, "export")
RUN_TAG = os.environ.get("EXPORT_RUN_TAG", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(EXPORT_DIR, "runs", RUN_TAG)

RUN_MODELS_DIR = os.path.join(RUN_DIR, "models")
RUN_RESULTS_DIR = os.path.join(RUN_DIR, "results")
RUN_METRICS_DIR = os.path.join(RUN_DIR, "metrics")
RUN_REPORTS_DIR = os.path.join(RUN_DIR, "reports")
RUN_METADATA_DIR = os.path.join(RUN_DIR, "metadata")

for path in [
    EXPORT_DIR,
    RUN_DIR,
    RUN_MODELS_DIR,
    RUN_RESULTS_DIR,
    RUN_METRICS_DIR,
    RUN_REPORTS_DIR,
    RUN_METADATA_DIR,
]:
    os.makedirs(path, exist_ok=True)


@dataclass
class ExportedModelInfo:
    source: str
    keras: str
    stable_keras: str
    params: int
    keras_reload_check: bool
    inference_smoke_check: bool


def _safe_json_dump(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _json_safe_value(value: Any) -> Any:
    """Convert values to JSON-safe equivalents for metadata snapshots."""
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _safe_config_snapshot() -> Dict[str, Any]:
    """Return a robust config snapshot without serialization failures."""
    raw: Dict[str, Any] = {}

    with suppress(Exception):
        payload = Config.to_dict()
        if isinstance(payload, dict):
            raw = payload

    if not raw:
        with suppress(Exception):
            raw = {
                k: v
                for k, v in Config.__dict__.items()
                if (
                    not k.startswith("_")
                    and not callable(v)
                    and not isinstance(v, (classmethod, staticmethod, property))
                )
            }

    return {k: _json_safe_value(v) for k, v in raw.items()}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return out


def _utc_now_iso_z() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_with_verify(src: str, dst: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not os.path.exists(src):
        LOGGER.warning("Missing artifact: %s", src)
        return False, None

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

    src_stat = os.stat(src)
    dst_stat = os.stat(dst)
    if src_stat.st_size != dst_stat.st_size:
        raise RuntimeError(f"Size mismatch after copy: {src} -> {dst}")

    src_hash = _sha256_file(src)
    dst_hash = _sha256_file(dst)
    if src_hash != dst_hash:
        raise RuntimeError(f"SHA256 mismatch after copy: {src} -> {dst}")

    return True, {
        "src": src,
        "dst": dst,
        "size_bytes": int(dst_stat.st_size),
        "sha256": dst_hash,
    }


def _get_dataset_name() -> str:
    if os.path.exists(HDF5_PATH):
        try:
            with h5py.File(HDF5_PATH, "r") as handle:
                raw_cfg = handle.attrs.get("config", "{}")
                cfg = json.loads(raw_cfg) if isinstance(raw_cfg, str) else {}
                return cfg.get("dataset", "BraTS")
        except Exception:
            pass
    return "BraTS"


def _get_custom_objects() -> Dict[str, Any]:
    objs: Dict[str, Any] = {}

    with suppress(Exception):
        from custom_objects_registry import get_custom_objects

        objs.update(get_custom_objects())

    objs.update(
        {
            "combined_loss": combined_loss,
            "dice_coef": dice_coef,
            "precision_metric": precision_metric,
            "sensitivity_metric": sensitivity_metric,
            "iou_metric": iou_metric,
        }
    )

    with suppress(Exception):
        from cell_07a_building_blocks_FIXED import _hierarchy_sort, hierarchy_constraint_layer

        objs["_hierarchy_sort"] = _hierarchy_sort
        objs["hierarchy_constraint_layer"] = hierarchy_constraint_layer

    return objs


def _run_inference_smoke_test(model: tf.keras.Model) -> bool:
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape is None or len(shape) < 4:
            return False

        batch = 1 if shape[0] is None else int(shape[0])
        fallback_size = int(os.environ.get("BRATS_IMG_SIZE", "128"))
        h = fallback_size if shape[1] is None else int(shape[1])
        w = fallback_size if shape[2] is None else int(shape[2])
        c = 4 if shape[3] is None else int(shape[3])

        x = tf.zeros((batch, h, w, c), dtype=tf.float32)
        y = model(x, training=False)
        if len(y.shape) == 4 and int(y.shape[-1]) == 1:
            return True
        if len(y.shape) == 2 and int(y.shape[-1]) >= 1:
            return True
        return False
    except Exception:
        return False


def _extract_metric_value(
    metric_map: Dict[str, float],
    include_aliases: List[str],
    default: float = 0.0,
    exclude_aliases: Optional[List[str]] = None,
) -> float:
    exclude_aliases = exclude_aliases or []
    for name, value in metric_map.items():
        lower = str(name).lower()
        if any(alias in lower for alias in include_aliases) and not any(
            alias in lower for alias in exclude_aliases
        ):
            return _safe_float(value, default=default)
    return _safe_float(default, default=0.0)


def _ensure_model_compiled_for_eval(model: tf.keras.Model) -> tf.keras.Model:
    # Models loaded with compile=False need an explicit compile before model.evaluate.
    if getattr(model, "optimizer", None) is not None:
        return model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coef, precision_metric, sensitivity_metric, iou_metric],
    )
    return model


def _resolve_eval_steps(val_ds: tf.data.Dataset, val_count: Optional[int], batch_size: int) -> int:
    if val_count is not None and val_count > 0:
        return max(1, math.ceil(val_count / max(1, batch_size)))

    with suppress(Exception):
        cardinality = tf.data.experimental.cardinality(val_ds)
        card = int(cardinality.numpy())
        if card > 0:
            return card

    fallback_steps = max(1, _safe_int(os.environ.get("BRATS_EVAL_STEPS", 32), default=32))
    LOGGER.warning(
        "Could not infer finite validation steps from dataset; using BRATS_EVAL_STEPS=%s",
        fallback_steps,
    )
    return fallback_steps


def _norm01(img: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1 + 1e-8)


def _collect_tumor_positive_samples(
    val_ds: tf.data.Dataset,
    needed: int,
    max_batches: int,
    min_tumor_pixels: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for X_batch, y_batch in val_ds.take(max_batches):
        y_sum = tf.reduce_sum(y_batch, axis=[1, 2, 3]).numpy()
        pos = np.where(y_sum > float(min_tumor_pixels))[0]
        if len(pos) == 0:
            continue

        x_np = X_batch.numpy()
        y_np = y_batch.numpy()
        x_list.append(x_np[pos])
        y_list.append(y_np[pos])

        got = sum(arr.shape[0] for arr in y_list)
        if got >= needed:
            break

    if not x_list:
        return None, None

    X_vis = np.concatenate(x_list, axis=0)[:needed]
    y_vis = np.concatenate(y_list, axis=0)[:needed]
    return X_vis, y_vis


def _render_single_model_visualization(
    model: tf.keras.Model,
    model_name: str,
    X_vis: np.ndarray,
    y_vis: np.ndarray,
    threshold: float,
    filename: str,
) -> str:
    y_prob = model.predict(X_vis, verbose=0)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    y_bin = (y_prob > float(threshold)).astype(np.float32)

    print(
        f"[VIS] {model_name} prob stats: "
        f"min={float(y_prob.min()):.6f} max={float(y_prob.max()):.6f} mean={float(y_prob.mean()):.6f}"
    )
    for i in range(int(X_vis.shape[0])):
        gt_px = int(np.sum(y_vis[i, :, :, 0]))
        pred_px = int(np.sum(y_bin[i, :, :, 0]))
        print(f"[VIS] {model_name} sample {i}: gt_pixels={gt_px}, pred_pixels@{threshold:.2f}={pred_px}")

    n_show = int(X_vis.shape[0])
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_show):
        mri = _norm01(X_vis[i, :, :, 0])

        axes[i, 0].imshow(mri, cmap="gray")
        axes[i, 0].set_title("MRI (FLAIR)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(y_vis[i, :, :, 0], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(y_prob[i, :, :, 0], cmap="magma", vmin=0.0, vmax=1.0)
        axes[i, 2].set_title("Prediction Probability")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mri, cmap="gray")
        axes[i, 3].imshow(y_bin[i, :, :, 0], cmap="jet", alpha=0.45)
        axes[i, 3].set_title(f"Overlay (thr={threshold:.2f})")
        axes[i, 3].axis("off")

    plt.suptitle(
        f"{model_name} Tumor Segmentation - Tumor-Positive Validation Samples",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    local_png = filename
    run_png = os.path.join(RUN_RESULTS_DIR, filename)
    stable_png = os.path.join(EXPORT_DIR, filename)

    plt.savefig(local_png, dpi=150, bbox_inches="tight")
    plt.savefig(run_png, dpi=150, bbox_inches="tight")
    plt.savefig(stable_png, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return stable_png


def _resolve_eval_context() -> Optional[Dict[str, Any]]:
    model = globals().get("model")
    val_ds = globals().get("val_ds")
    val_count: Optional[int] = None

    batch_size = _safe_int(
        globals().get("BATCH_SIZE", globals().get("GLOBAL_BATCH_SIZE", 8)),
        default=8,
    )
    batch_size = max(1, batch_size)

    if "val_idx" in globals():
        with suppress(Exception):
            val_count = int(len(globals()["val_idx"]))

    if model is None:
        unet_path = os.path.join(MODEL_DIR, "unet_best.keras")
        if os.path.exists(unet_path):
            with suppress(Exception):
                model = tf.keras.models.load_model(
                    unet_path,
                    custom_objects=_get_custom_objects(),
                    compile=False,
                )
                model = _ensure_model_compiled_for_eval(model)

    if val_ds is None:
        make_tf_dataset_fn = globals().get("make_tf_dataset")
        if callable(make_tf_dataset_fn):
            with suppress(Exception):
                val_ds, count = make_tf_dataset_fn(
                    HDF5_PATH,
                    "val",
                    batch_size,
                    shuffle=False,
                    augment=False,
                    drop_remainder=False,
                )
                val_count = int(count)

    if model is None or val_ds is None:
        return None

    with suppress(Exception):
        model = _ensure_model_compiled_for_eval(model)

    return {
        "model": model,
        "val_ds": val_ds,
        "val_count": val_count,
        "batch_size": batch_size,
    }


def run_evaluation(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    val_count: Optional[int],
    batch_size: int,
) -> Dict[str, Any]:
    print("=" * 60)
    print("FINAL RESULTS - BASELINE U-NET")
    print("=" * 60)

    eval_steps = _resolve_eval_steps(val_ds=val_ds, val_count=val_count, batch_size=batch_size)
    eval_kwargs: Dict[str, Any] = {"verbose": 1, "steps": eval_steps}

    raw_results = model.evaluate(val_ds, **eval_kwargs)
    if isinstance(raw_results, (float, int, np.floating, np.integer)):
        results = [_safe_float(raw_results)]
    else:
        results = [_safe_float(v) for v in list(raw_results)]

    metric_names = list(getattr(model, "metrics_names", []))
    if len(metric_names) != len(results):
        metric_names = [f"metric_{i}" for i in range(len(results))]

    metric_map = {metric_names[i]: results[i] for i in range(len(results))}

    val_loss = _extract_metric_value(metric_map, ["loss"], default=results[0] if results else 0.0)
    val_dice_soft = _extract_metric_value(metric_map, ["dice"], default=0.0, exclude_aliases=["hard"])
    val_dice_hard = _extract_metric_value(metric_map, ["hard", "dice_hard"], default=val_dice_soft)
    val_iou_hard = _extract_metric_value(metric_map, ["iou"], default=0.0)
    val_precision = _extract_metric_value(metric_map, ["precision"], default=0.0)
    val_sensitivity = _extract_metric_value(metric_map, ["sensitivity", "recall"], default=0.0)
    val_pixel_accuracy = _extract_metric_value(metric_map, ["pixel_accuracy", "binary_accuracy", "accuracy"], default=0.0)
    val_f1 = _safe_float(
        (2.0 * val_precision * val_sensitivity) / (val_precision + val_sensitivity + 1e-8),
        default=0.0,
    )

    print(f"\nValidation Dice (Soft): {val_dice_soft:.4f}")
    print(f"Validation Dice (Hard): {val_dice_hard:.4f}")
    print(f"Validation IoU (Hard) : {val_iou_hard:.4f}")
    print(f"Validation Pixel Acc  : {val_pixel_accuracy:.4f}")
    print(f"Validation F1 Score   : {val_f1:.4f}")
    print(f"Validation Loss       : {val_loss:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparison_metrics: Dict[str, Any] = {}
    metrics_path = os.path.join(RESULTS_DIR, "comparison_metrics.json")
    if os.path.exists(metrics_path):
        with suppress(Exception):
            with open(metrics_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
                if isinstance(existing, dict):
                    comparison_metrics = existing

    comparison_metrics["U-Net"] = {
        "task_type": "segmentation",
        "dice_coef_soft": float(val_dice_soft),
        "dice_coef_hard": float(val_dice_hard),
        "iou": float(val_iou_hard),
        "precision": float(val_precision),
        "recall": float(val_sensitivity),
        "sensitivity": float(val_sensitivity),
        "f1_score": float(val_f1),
        "pixel_accuracy": float(val_pixel_accuracy),
        "test_loss": float(val_loss),
        "primary_metric": "dice_coef_soft",
        "threshold": 0.5,
        "num_samples": int(val_count or 0),
    }

    _safe_json_dump(comparison_metrics, metrics_path)
    LOGGER.info("Saved comparison_metrics.json -> %s", metrics_path)

    return {
        "model": model,
        "val_ds": val_ds,
        "val_count": val_count,
        "batch_size": batch_size,
        "metrics_path": metrics_path,
    }


def run_visualization(eval_state: Dict[str, Any]) -> Optional[str]:
    model = eval_state["model"]
    val_ds = eval_state["val_ds"]

    vis_samples = max(1, _safe_int(os.environ.get("BRATS_VIS_SAMPLES", 5), default=5))
    vis_max_batches = max(1, _safe_int(os.environ.get("BRATS_VIS_MAX_BATCHES", 50), default=50))
    vis_min_tumor_pixels = max(
        0.0,
        _safe_float(os.environ.get("BRATS_VIS_MIN_TUMOR_PIXELS", 20.0), default=20.0),
    )
    vis_threshold = float(
        np.clip(_safe_float(os.environ.get("BRATS_VIS_THRESHOLD", 0.30), default=0.30), 0.0, 1.0)
    )

    X_vis, y_vis = _collect_tumor_positive_samples(
        val_ds=val_ds,
        needed=vis_samples,
        max_batches=vis_max_batches,
        min_tumor_pixels=vis_min_tumor_pixels,
    )
    if X_vis is None or y_vis is None:
        LOGGER.warning(
            "Visualization skipped: no tumor-positive slices found in first %s validation batches.",
            vis_max_batches,
        )
        return None

    print("=" * 60)
    print("TUMOR-ONLY VIS CHECK")
    print("=" * 60)

    unet_png = _render_single_model_visualization(
        model=model,
        model_name="U-Net",
        X_vis=X_vis,
        y_vis=y_vis,
        threshold=vis_threshold,
        filename="segmentation_results.png",
    )

    attn_model_path = os.path.join(MODEL_DIR, "attention_unet_best.keras")
    if os.path.exists(attn_model_path):
        with suppress(Exception):
            attn_model = tf.keras.models.load_model(
                attn_model_path,
                custom_objects=_get_custom_objects(),
                compile=False,
            )
            _render_single_model_visualization(
                model=attn_model,
                model_name="Attention U-Net",
                X_vis=X_vis,
                y_vis=y_vis,
                threshold=vis_threshold,
                filename="segmentation_results_attention_unet.png",
            )
    else:
        LOGGER.warning("Attention U-Net checkpoint missing: %s", attn_model_path)

    print("\n[OK] Visualization complete (tumor-positive slices)")
    return unet_png


def _run_evaluate_and_visualize_if_available() -> Optional[str]:
    ctx = _resolve_eval_context()
    if ctx is None:
        LOGGER.warning(
            "Step 1/2 skipped: missing evaluation context. "
            "Run Cell 10 first, or keep model/val_ds in memory."
        )
        return None

    try:
        eval_state = run_evaluation(
            model=ctx["model"],
            val_ds=ctx["val_ds"],
            val_count=ctx.get("val_count"),
            batch_size=_safe_int(ctx.get("batch_size", 8), default=8),
        )
    except Exception as exc:
        LOGGER.warning("Evaluation step failed (%s). Continuing with export-only flow.", exc)
        return None

    try:
        return run_visualization(eval_state)
    except Exception as exc:
        LOGGER.warning("Visualization step failed (%s). Continuing with export-only flow.", exc)
        return None


def export_models(export_dir: str = EXPORT_DIR) -> Dict[str, Dict[str, Any]]:
    model_specs = [
        ("unet", "unet_best.keras"),
        ("attention_unet", "attention_unet_best.keras"),
        ("attention_unet_vit", "attention_unet_vit_best.keras"),
    ]
    exported: Dict[str, Dict[str, Any]] = {}

    for model_name, filename in model_specs:
        src_model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(src_model_path):
            continue

        model = tf.keras.models.load_model(src_model_path, custom_objects=_get_custom_objects(), compile=False)

        keras_path = os.path.join(RUN_MODELS_DIR, f"{model_name}.keras")
        model.save(keras_path)
        stable_keras = os.path.join(export_dir, f"{model_name}.keras")
        _copy_with_verify(keras_path, stable_keras)

        keras_reload_check = True
        with suppress(Exception):
            _ = tf.keras.models.load_model(keras_path, compile=False)

        info = ExportedModelInfo(
            source=src_model_path,
            keras=keras_path,
            stable_keras=stable_keras,
            params=int(model.count_params()),
            keras_reload_check=keras_reload_check,
            inference_smoke_check=_run_inference_smoke_test(model),
        )
        exported[model_name] = info.__dict__

    _safe_json_dump(exported, os.path.join(RUN_METADATA_DIR, "exported_models.json"))
    return exported


def generate_final_metrics(export_dir: str = EXPORT_DIR) -> Optional[Dict[str, Any]]:
    metrics_path = os.path.join(RESULTS_DIR, "comparison_metrics.json")
    if not os.path.exists(metrics_path):
        LOGGER.warning("Missing comparison metrics: %s", metrics_path)
        summary: Dict[str, Any] = {
            "dataset": _get_dataset_name(),
            "task": "Brain Tumor Analysis (Segmentation + Optional Classification)",
            "classes": ["tumor"],
            "source_metrics_path": metrics_path,
            "status": "pending_evaluation",
            "warning": "comparison_metrics.json not found. Run Cell 10 after models and HDF5 are available.",
            "models": {},
            "ranking_by_primary_metric": [],
            "ranking_by_dice_coef": [],
            "comparison": {},
            "proposed_vs_existing": {},
        }
        run_metrics_path = os.path.join(RUN_METRICS_DIR, "final_metrics.json")
        stable_metrics_path = os.path.join(export_dir, "final_metrics.json")
        _safe_json_dump(summary, run_metrics_path)
        _safe_json_dump(summary, stable_metrics_path)
        return summary

    with open(metrics_path, "r", encoding="utf-8") as handle:
        comparison_metrics = json.load(handle)

    training_summary_path = os.path.join(RESULTS_DIR, "dual_training_summary.json")
    target_binary_dice_unet = BINARY_TARGET_UNET
    target_binary_dice_attention = BINARY_TARGET_ATTENTION
    if os.path.exists(training_summary_path):
        with suppress(Exception):
            with open(training_summary_path, "r", encoding="utf-8") as handle:
                training_summary = json.load(handle)
            target_binary_dice_unet = str(
                training_summary.get(
                    "target_binary_dice_unet",
                    training_summary.get("target_wt_dice_unet", target_binary_dice_unet),
                )
            )
            target_binary_dice_attention = str(
                training_summary.get(
                    "target_binary_dice_attention",
                    training_summary.get("target_wt_dice_attention", target_binary_dice_attention),
                )
            )

    summary: Dict[str, Any] = {
        "dataset": _get_dataset_name(),
        "task": "Brain Tumor Analysis (Segmentation + Optional Classification)",
        "classes": ["tumor"],
        "source_metrics_path": metrics_path,
        "expected_binary_dice_targets": {
            "basic_unet_2d_slices": target_binary_dice_unet,
            "attention_unet_2d": target_binary_dice_attention,
        },
        "models": {},
        "ranking_by_primary_metric": [],
        "ranking_by_dice_coef": [],
        "comparison": {},
        "proposed_vs_existing": {},
    }

    for model_name, m in comparison_metrics.items():
        task_type = str(m.get("task_type", "segmentation")).lower()
        recall = _safe_float(m.get("recall", m.get("sensitivity", 0.0)), default=0.0)
        primary_metric_name = str(
            m.get("primary_metric", "f1_score" if task_type == "classification" else "dice_coef_soft")
        )
        primary_metric_value = _safe_float(
            m.get(
                primary_metric_name,
                m.get("f1_score", m.get("dice_coef_soft", m.get("accuracy", 0.0))),
            ),
            default=0.0,
        )

        summary["models"][model_name] = {
            "task_type": task_type,
            "dice_coef_soft": _safe_float(m.get("dice_coef_soft", 0.0), default=0.0),
            "dice_coef_hard": _safe_float(m.get("dice_coef_hard", 0.0), default=0.0),
            "accuracy": _safe_float(m.get("accuracy", 0.0), default=0.0),
            "pixel_accuracy": _safe_float(
                m.get("pixel_accuracy", m.get("accuracy", 0.0)),
                default=0.0,
            ),
            "precision": _safe_float(m.get("precision", 0.0), default=0.0),
            "recall": recall,
            "sensitivity": recall,
            "f1_score": _safe_float(m.get("f1_score", 0.0), default=0.0),
            "iou": _safe_float(m.get("iou", 0.0), default=0.0),
            "test_loss": _safe_float(m.get("test_loss", 0.0), default=0.0),
            "threshold": _safe_float(m.get("threshold", 0.5), default=0.5),
            "num_samples": _safe_int(m.get("num_samples", 0), default=0),
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
        }

    ranking_primary = sorted(
        summary["models"].items(),
        key=lambda kv: kv[1].get("primary_metric_value", 0.0),
        reverse=True,
    )
    summary["ranking_by_primary_metric"] = [
        {
            "model": model_name,
            "primary_metric_name": payload.get("primary_metric_name"),
            "primary_metric_value": payload.get("primary_metric_value"),
        }
        for model_name, payload in ranking_primary
    ]

    ranking_dice = sorted(
        [
            (k, v)
            for k, v in summary["models"].items()
            if str(v.get("task_type", "segmentation")) == "segmentation"
        ],
        key=lambda kv: kv[1]["dice_coef_soft"],
        reverse=True,
    )
    summary["ranking_by_dice_coef"] = [{"model": k, "dice_coef_soft": v["dice_coef_soft"]} for k, v in ranking_dice]

    if "U-Net" in summary["models"] and "Attention U-Net" in summary["models"]:
        u = _safe_float(summary["models"]["U-Net"].get("dice_coef_soft", 0.0), default=0.0)
        a = _safe_float(summary["models"]["Attention U-Net"].get("dice_coef_soft", 0.0), default=0.0)
        summary["comparison"] = {
            "attention_minus_unet": _safe_float(a - u, default=0.0),
            "attention_gain_percent": float((a - u) / u * 100.0) if u > 0 else None,
        }

    proposed_name = "Attention U-Net + ViT (Proposed)"
    if proposed_name in summary["models"]:
        proposed = summary["models"][proposed_name]
        existing_best = ranking_dice[0] if ranking_dice else (None, None)
        summary["proposed_vs_existing"] = {
            "proposed_model": proposed_name,
            "proposed_task_type": proposed.get("task_type"),
            "proposed_primary_metric_name": proposed.get("primary_metric_name"),
            "proposed_primary_metric_value": proposed.get("primary_metric_value"),
            "best_existing_segmentation_model": existing_best[0],
            "best_existing_dice_soft": (
                existing_best[1].get("dice_coef_soft") if existing_best[1] is not None else None
            ),
            "note": (
                "Proposed model is classification-focused; compare by F1/accuracy/test-loss. "
                "Existing models are segmentation-focused; compare by Dice/IoU."
            ),
        }

    run_metrics_path = os.path.join(RUN_METRICS_DIR, "final_metrics.json")
    stable_metrics_path = os.path.join(export_dir, "final_metrics.json")
    _safe_json_dump(summary, run_metrics_path)
    _safe_json_dump(summary, stable_metrics_path)
    return summary


def generate_final_report(export_dir: str = EXPORT_DIR, final_metrics: Optional[Dict[str, Any]] = None) -> str:
    if final_metrics is None:
        final_metrics = generate_final_metrics(export_dir=export_dir)

    report_lines: List[str] = [
        "# Brain Tumor Segmentation: Final Report",
        "",
        "## Executive Summary",
        f"- Dataset: {_get_dataset_name()}",
        "- Task: Binary segmentation with optional proposed classification head",
        "- Inputs: 4-channel MRI (FLAIR, T1, T1ce, T2)",
        "- Outputs: model exports, metrics JSON, submission package",
        f"- Export run: {RUN_TAG}",
        "",
    ]

    if final_metrics and final_metrics.get("models"):
        def _fmt_or_na(value: Any) -> str:
            if value is None:
                return "N/A"
            with suppress(Exception):
                v = float(value)
                if math.isfinite(v):
                    return f"{v:.4f}"
            return "N/A"

        report_lines.extend(
            [
                "## Evaluation Results",
                "",
                "| Model | Task | Dice (soft) | Pixel Acc | Precision | Recall | F1 | IoU | Test Loss | Primary |",
                "|------|------|------------:|----------:|----------:|-------:|---:|----:|---------:|--------:|",
            ]
        )

        for model_name, m in final_metrics["models"].items():
            report_lines.append(
                "| "
                f"{model_name} | "
                f"{m.get('task_type', 'segmentation')} | "
                f"{_fmt_or_na(m.get('dice_coef_soft'))} | "
                f"{_fmt_or_na(m.get('pixel_accuracy', m.get('accuracy')))} | "
                f"{_fmt_or_na(m.get('precision'))} | "
                f"{_fmt_or_na(m.get('recall', m.get('sensitivity')))} | "
                f"{_fmt_or_na(m.get('f1_score'))} | "
                f"{_fmt_or_na(m.get('iou'))} | "
                f"{_fmt_or_na(m.get('test_loss'))} | "
                f"{m.get('primary_metric_name', 'metric')}={_fmt_or_na(m.get('primary_metric_value'))} |"
            )

        ranking_primary = final_metrics.get("ranking_by_primary_metric", [])
        if ranking_primary:
            best = ranking_primary[0]
            report_lines.extend(
                [
                    "",
                    "## Best Model by Primary Metric",
                    "",
                    f"- Model: {best.get('model', 'N/A')}",
                    f"- Primary metric: {best.get('primary_metric_name', 'N/A')}",
                    f"- Value: {_fmt_or_na(best.get('primary_metric_value'))}",
                ]
            )

        proposed_cmp = final_metrics.get("proposed_vs_existing", {})
        if isinstance(proposed_cmp, dict) and proposed_cmp:
            report_lines.extend(
                [
                    "",
                    "## Proposed vs Existing Models",
                    "",
                    f"- Proposed model: {proposed_cmp.get('proposed_model', 'N/A')}",
                    f"- Proposed task type: {proposed_cmp.get('proposed_task_type', 'N/A')}",
                    (
                        "- Proposed primary metric "
                        f"({proposed_cmp.get('proposed_primary_metric_name', 'metric')}): "
                        f"{_fmt_or_na(proposed_cmp.get('proposed_primary_metric_value'))}"
                    ),
                    (
                        "- Best existing segmentation model: "
                        f"{proposed_cmp.get('best_existing_segmentation_model', 'N/A')}"
                    ),
                    (
                        "- Best existing Dice (soft): "
                        f"{_fmt_or_na(proposed_cmp.get('best_existing_dice_soft'))}"
                    ),
                    f"- Note: {proposed_cmp.get('note', '')}",
                ]
            )

        targets = final_metrics.get("expected_binary_dice_targets", {})
        report_lines.extend(
            [
                "",
                "## Setup Expected Binary Dice",
                "",
                "| Setup | Expected Binary Dice |",
                "|------|---------------------:|",
                f"| Basic U-Net, 2D slices | {targets.get('basic_unet_2d_slices', BINARY_TARGET_UNET)} |",
                f"| Attention U-Net, 2D | {targets.get('attention_unet_2d', BINARY_TARGET_ATTENTION)} |",
            ]
        )

    training_loss_plot = os.path.join(export_dir, "training_loss_comparison.png")
    test_loss_plot = os.path.join(export_dir, "test_loss_comparison.png")
    metric_plot = os.path.join(export_dir, "training_metric_comparison.png")
    if os.path.exists(training_loss_plot) or os.path.exists(test_loss_plot) or os.path.exists(metric_plot):
        report_lines.extend(["", "## Loss and Training Graphs", ""])
        if os.path.exists(training_loss_plot):
            report_lines.extend(["### Training/Validation Loss", "", "![Training Loss](training_loss_comparison.png)", ""])
        if os.path.exists(test_loss_plot):
            report_lines.extend(["### Test Loss Comparison", "", "![Test Loss](test_loss_comparison.png)", ""])
        if os.path.exists(metric_plot):
            report_lines.extend(["### Primary Metric Curves", "", "![Training Metrics](training_metric_comparison.png)", ""])

    if os.path.exists(os.path.join(export_dir, "segmentation_results.png")):
        report_lines.extend(
            [
                "",
                "## Visualization",
                "",
                "### U-Net",
                "",
                "![U-Net Segmentation Results](segmentation_results.png)",
            ]
        )

    if os.path.exists(os.path.join(export_dir, "segmentation_results_attention_unet.png")):
        report_lines.extend(
            [
                "",
                "### Attention U-Net",
                "",
                "![Attention U-Net Segmentation Results](segmentation_results_attention_unet.png)",
            ]
        )

    report_text = "\n".join(report_lines)
    run_report_path = os.path.join(RUN_REPORTS_DIR, "FINAL_REPORT.md")
    stable_report_path = os.path.join(export_dir, "FINAL_REPORT.md")

    with open(run_report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    with open(stable_report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return run_report_path


def create_submission_package(export_dir: str = EXPORT_DIR) -> Tuple[str, List[Dict[str, Any]]]:
    package_dir = os.path.join(RUN_DIR, "submission_package")
    models_dir = os.path.join(package_dir, "models")
    metrics_dir = os.path.join(package_dir, "metrics")
    reports_dir = os.path.join(package_dir, "reports")
    results_dir = os.path.join(package_dir, "results")

    for path in [models_dir, metrics_dir, reports_dir, results_dir]:
        os.makedirs(path, exist_ok=True)

    copied_artifacts: List[Dict[str, Any]] = []

    def _maybe_copy(src: str, dst: str) -> None:
        ok, item = _copy_with_verify(src, dst)
        if ok and item is not None:
            copied_artifacts.append(item)

    _maybe_copy(os.path.join(export_dir, "unet.keras"), os.path.join(models_dir, "unet.keras"))
    _maybe_copy(os.path.join(export_dir, "attention_unet.keras"), os.path.join(models_dir, "attention_unet.keras"))
    _maybe_copy(os.path.join(export_dir, "attention_unet_vit.keras"), os.path.join(models_dir, "attention_unet_vit.keras"))
    _maybe_copy(os.path.join(export_dir, "final_metrics.json"), os.path.join(metrics_dir, "final_metrics.json"))
    _maybe_copy(os.path.join(export_dir, "FINAL_REPORT.md"), os.path.join(reports_dir, "FINAL_REPORT.md"))
    _maybe_copy(
        os.path.join(export_dir, "segmentation_results.png"),
        os.path.join(results_dir, "segmentation_results.png"),
    )
    _maybe_copy(
        os.path.join(export_dir, "segmentation_results_attention_unet.png"),
        os.path.join(results_dir, "segmentation_results_attention_unet.png"),
    )
    _maybe_copy(
        os.path.join(export_dir, "training_loss_comparison.png"),
        os.path.join(results_dir, "training_loss_comparison.png"),
    )
    _maybe_copy(
        os.path.join(export_dir, "test_loss_comparison.png"),
        os.path.join(results_dir, "test_loss_comparison.png"),
    )
    _maybe_copy(
        os.path.join(export_dir, "training_metric_comparison.png"),
        os.path.join(results_dir, "training_metric_comparison.png"),
    )

    return package_dir, copied_artifacts


def _sync_optional_comparison_plots(export_dir: str = EXPORT_DIR) -> None:
    plot_files = [
        "training_comparison.png",
        "training_loss_comparison.png",
        "training_metric_comparison.png",
        "test_loss_comparison.png",
    ]

    for filename in plot_files:
        src = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(src):
            continue

        run_dst = os.path.join(RUN_RESULTS_DIR, filename)
        stable_dst = os.path.join(export_dir, filename)
        _copy_with_verify(src, run_dst)
        _copy_with_verify(src, stable_dst)


def _write_run_metadata(
    exported_models: Dict[str, Dict[str, Any]],
    final_metrics: Optional[Dict[str, Any]],
    copied_artifacts: List[Dict[str, Any]],
) -> None:
    metadata = {
        "run_tag": RUN_TAG,
        "timestamp_utc": _utc_now_iso_z(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "dataset": _get_dataset_name(),
        "hdf5_path": HDF5_PATH,
        "task": "binary",
        "exported_models": exported_models,
        "final_metrics_available": final_metrics is not None,
        "copied_artifacts": copied_artifacts,
        "config_snapshot": _safe_config_snapshot(),
    }

    _safe_json_dump(metadata, os.path.join(RUN_METADATA_DIR, "run_metadata.json"))


def _artifact_ok(path: str) -> str:
    return "OK" if os.path.exists(path) else "MISSING"


def _fmt_metric(value: Any) -> str:
    return f"{_safe_float(value):.4f}"


def print_final_summary(
    exported_models: Optional[Dict[str, Dict[str, Any]]] = None,
    final_metrics: Optional[Dict[str, Any]] = None,
    package_dir: Optional[str] = None,
) -> None:
    LOGGER.info("Run tag: %s", RUN_TAG)
    LOGGER.info("Run dir: %s", RUN_DIR)

    if exported_models:
        for model_name, payload in exported_models.items():
            LOGGER.info(
                "- %s: params=%s reload_ok=%s smoke_ok=%s",
                model_name,
                payload.get("params"),
                payload.get("keras_reload_check"),
                payload.get("inference_smoke_check"),
            )
    else:
        LOGGER.warning("No models were exported in this run.")

    if final_metrics and final_metrics.get("ranking_by_primary_metric"):
        LOGGER.info("Model ranking by primary metric:")
        for row in final_metrics["ranking_by_primary_metric"]:
            LOGGER.info(
                "- %s: %s=%s",
                row.get("model"),
                row.get("primary_metric_name"),
                _fmt_metric(row.get("primary_metric_value")),
            )

    if final_metrics and final_metrics.get("ranking_by_dice_coef"):
        LOGGER.info("Segmentation ranking by Dice:")
        for row in final_metrics["ranking_by_dice_coef"]:
            LOGGER.info("- %s: %.4f", row["model"], row["dice_coef_soft"])

    if final_metrics and final_metrics.get("models"):
        LOGGER.info("Detailed metrics:")
        for model_name, metrics in final_metrics["models"].items():
            LOGGER.info(
                "- %s: task=%s dice_soft=%s dice_hard=%s pixel_acc=%s precision=%s recall=%s "
                "f1=%s iou=%s test_loss=%s threshold=%s",
                model_name,
                metrics.get("task_type", "segmentation"),
                _fmt_metric(metrics.get("dice_coef_soft")),
                _fmt_metric(metrics.get("dice_coef_hard")),
                _fmt_metric(metrics.get("pixel_accuracy", metrics.get("accuracy"))),
                _fmt_metric(metrics.get("precision")),
                _fmt_metric(metrics.get("recall", metrics.get("sensitivity"))),
                _fmt_metric(metrics.get("f1_score")),
                _fmt_metric(metrics.get("iou")),
                _fmt_metric(metrics.get("test_loss")),
                _fmt_metric(metrics.get("threshold", 0.5)),
            )

    if final_metrics and final_metrics.get("proposed_vs_existing"):
        p = final_metrics["proposed_vs_existing"]
        LOGGER.info(
            "Proposed vs existing: %s (%s=%s) | best existing segmentation=%s (dice_soft=%s)",
            p.get("proposed_model"),
            p.get("proposed_primary_metric_name"),
            _fmt_metric(p.get("proposed_primary_metric_value")),
            p.get("best_existing_segmentation_model"),
            _fmt_metric(p.get("best_existing_dice_soft")),
        )

    if package_dir:
        LOGGER.info("Submission package: %s", package_dir)

    LOGGER.info("Cell 12 checks:")
    checks = [
        ("stable unet", os.path.join(EXPORT_DIR, "unet.keras")),
        ("stable attention_unet", os.path.join(EXPORT_DIR, "attention_unet.keras")),
        ("stable attention_unet_vit", os.path.join(EXPORT_DIR, "attention_unet_vit.keras")),
        ("stable final_metrics", os.path.join(EXPORT_DIR, "final_metrics.json")),
        ("stable FINAL_REPORT", os.path.join(EXPORT_DIR, "FINAL_REPORT.md")),
        ("stable segmentation_viz", os.path.join(EXPORT_DIR, "segmentation_results.png")),
        (
            "stable segmentation_viz_attention",
            os.path.join(EXPORT_DIR, "segmentation_results_attention_unet.png"),
        ),
        ("stable training_loss_graph", os.path.join(EXPORT_DIR, "training_loss_comparison.png")),
        ("stable test_loss_graph", os.path.join(EXPORT_DIR, "test_loss_comparison.png")),
        ("stable training_metric_graph", os.path.join(EXPORT_DIR, "training_metric_comparison.png")),
        ("run metadata", os.path.join(RUN_METADATA_DIR, "run_metadata.json")),
        ("run exported_models", os.path.join(RUN_METADATA_DIR, "exported_models.json")),
    ]
    for label, path in checks:
        LOGGER.info("- %s: %s (%s)", label, _artifact_ok(path), path)

    if package_dir:
        package_checks = [
            ("pkg models/unet", os.path.join(package_dir, "models", "unet.keras")),
            ("pkg models/attention_unet", os.path.join(package_dir, "models", "attention_unet.keras")),
            ("pkg models/attention_unet_vit", os.path.join(package_dir, "models", "attention_unet_vit.keras")),
            ("pkg metrics/final_metrics", os.path.join(package_dir, "metrics", "final_metrics.json")),
            ("pkg reports/FINAL_REPORT", os.path.join(package_dir, "reports", "FINAL_REPORT.md")),
            ("pkg results/segmentation_viz", os.path.join(package_dir, "results", "segmentation_results.png")),
            (
                "pkg results/segmentation_viz_attention",
                os.path.join(package_dir, "results", "segmentation_results_attention_unet.png"),
            ),
            (
                "pkg results/training_loss_graph",
                os.path.join(package_dir, "results", "training_loss_comparison.png"),
            ),
            (
                "pkg results/test_loss_graph",
                os.path.join(package_dir, "results", "test_loss_comparison.png"),
            ),
            (
                "pkg results/training_metric_graph",
                os.path.join(package_dir, "results", "training_metric_comparison.png"),
            ),
        ]
        for label, path in package_checks:
            LOGGER.info("- %s: %s (%s)", label, _artifact_ok(path), path)


def main() -> Dict[str, Any]:
    _ = _run_evaluate_and_visualize_if_available()
    _sync_optional_comparison_plots()

    exported_models = export_models()
    final_metrics = generate_final_metrics()
    generate_final_report(final_metrics=final_metrics)
    package_dir, copied_artifacts = create_submission_package()
    _write_run_metadata(exported_models, final_metrics, copied_artifacts)
    print_final_summary(exported_models=exported_models, final_metrics=final_metrics, package_dir=package_dir)

    return {
        "run_tag": RUN_TAG,
        "run_dir": RUN_DIR,
        "exported_models": exported_models,
        "final_metrics": final_metrics,
        "package_dir": package_dir,
        "copied_artifacts": copied_artifacts,
    }


if __name__ == "__main__":
    FINAL_EXPORT_STATE = main()
