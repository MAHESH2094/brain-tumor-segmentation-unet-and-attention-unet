# ===================================================
# CELL 10: EVALUATION WRAPPER (BINARY)
# ===================================================

import json
import math
import os
import sys
import importlib.util
import matplotlib.pyplot as plt


def _candidate_roots():
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots.extend([os.getcwd(), "/kaggle/working", "/kaggle/input"])

    # Kaggle datasets are often nested one or two levels under /kaggle/input.
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


def _ensure_project_root_on_path():
    for root in _candidate_roots():
        if os.path.exists(os.path.join(root, "config.py")) or os.path.exists(
            os.path.join(root, "ml_pipeline", "evaluate.py")
        ):
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

    # Fallback: derive minimal paths from environment so evaluation can proceed.
    # This avoids crashing in Kaggle notebooks where config.py is not shipped.
    default_output = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    output_dir = os.environ.get("OUTPUT_DIR", default_output)

    class _FallbackConfig:
        OUTPUT_DIR = output_dir
        RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(output_dir, "results"))
        HDF5_PATH = os.environ.get("HDF5_PATH", os.path.join(output_dir, "brats_preprocessed.h5"))

    print(
        "[WARN] config.py not found. Using environment-based fallback config "
        f"(OUTPUT_DIR={_FallbackConfig.OUTPUT_DIR})."
    )
    return _FallbackConfig


def _load_evaluate_models_fn():
    try:
        from ml_pipeline.evaluate import evaluate_models as _evaluate_models

        return _evaluate_models
    except ModuleNotFoundError:
        pass

    for root in _candidate_roots():
        eval_path = os.path.join(root, "ml_pipeline", "evaluate.py")
        if not os.path.exists(eval_path):
            continue
        spec = importlib.util.spec_from_file_location("ml_pipeline.evaluate", eval_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "evaluate_models"):
                return module.evaluate_models

    print(
        "[WARN] ml_pipeline/evaluate.py not found. "
        "Falling back to in-cell evaluation implementation."
    )
    return _fallback_evaluate_models


def _fallback_load_threshold(results_dir):
    default_threshold = 0.5
    threshold_path = os.environ.get("BRATS_THRESHOLDS_PATH", "").strip()
    if not threshold_path:
        threshold_path = os.path.join(results_dir, "optimal_thresholds.json")

    if not os.path.exists(threshold_path):
        return float(default_threshold)

    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "binary" in payload:
                return _safe_float(payload.get("binary", default_threshold), default_threshold)
            if "threshold" in payload:
                return _safe_float(payload.get("threshold", default_threshold), default_threshold)
        if isinstance(payload, (list, tuple)) and len(payload) >= 1:
            return _safe_float(payload[0], default_threshold)
    except Exception:
        pass

    return float(default_threshold)


def _fallback_custom_objects():
    # Prefer the centralized registry if available.
    try:
        from custom_objects_registry import get_custom_objects

        return get_custom_objects()
    except Exception:
        pass

    # Legacy fallback for notebooks running isolated cells.
    try:
        from cell_08_loss_metrics_FIXED import (
            combined_loss,
            dice_coef,
            precision_metric,
            sensitivity_metric,
            iou_metric,
        )

        objs = {
            "combined_loss": combined_loss,
            "dice_coef": dice_coef,
            "precision_metric": precision_metric,
            "sensitivity_metric": sensitivity_metric,
            "iou_metric": iou_metric,
        }

        try:
            from cell_07a_building_blocks_FIXED import _hierarchy_sort, hierarchy_constraint_layer

            objs["_hierarchy_sort"] = _hierarchy_sort
            objs["hierarchy_constraint_layer"] = hierarchy_constraint_layer
        except Exception:
            pass

        return objs
    except Exception:
        return {}


def _fallback_binary_batch_metrics(y_true, y_prob, threshold=0.5, eps=1e-6):
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.ndim != 4:
        raise RuntimeError(f"Expected y_true rank-4, got shape={y_true.shape}")
    if y_prob.ndim != 4:
        raise RuntimeError(f"Expected y_prob rank-4, got shape={y_prob.shape}")

    y_true = (y_true > 0.5).astype(np.float32)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_hard = (y_prob > float(threshold)).astype(np.float32)

    axes = (1, 2, 3)

    inter_soft = np.sum(y_true * y_prob, axis=axes)
    union_soft = np.sum(y_true + y_prob, axis=axes)
    dice_soft = np.mean((2.0 * inter_soft + eps) / (union_soft + eps))

    true_positive = np.sum(y_hard * y_true, axis=axes)
    false_positive = np.sum(y_hard * (1.0 - y_true), axis=axes)
    false_negative = np.sum((1.0 - y_hard) * y_true, axis=axes)
    true_negative = np.sum((1.0 - y_hard) * (1.0 - y_true), axis=axes)

    precision_per = (true_positive + eps) / (true_positive + false_positive + eps)
    recall_per = (true_positive + eps) / (true_positive + false_negative + eps)
    f1_per = (2.0 * precision_per * recall_per + eps) / (precision_per + recall_per + eps)
    iou_per = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    pixel_acc_per = (true_positive + true_negative + eps) / (
        true_positive + true_negative + false_positive + false_negative + eps
    )
    dice_hard_per = (2.0 * true_positive + eps) / (
        2.0 * true_positive + false_positive + false_negative + eps
    )

    bce_map = -(y_true * np.log(y_prob + eps) + (1.0 - y_true) * np.log(1.0 - y_prob + eps))
    test_loss = float(np.mean(bce_map))

    return {
        "dice_coef_soft": _safe_float(dice_soft),
        "dice_coef_hard": _safe_float(np.mean(dice_hard_per)),
        "precision": _safe_float(np.mean(precision_per)),
        "recall": _safe_float(np.mean(recall_per)),
        "sensitivity": _safe_float(np.mean(recall_per)),
        "f1_score": _safe_float(np.mean(f1_per)),
        "pixel_accuracy": _safe_float(np.mean(pixel_acc_per)),
        "iou": _safe_float(np.mean(iou_per)),
        "test_loss": _safe_float(test_loss),
    }


def _fallback_classification_batch_metrics(y_true, y_prob, threshold=0.5, eps=1e-6):
    import numpy as np

    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_true = np.asarray(y_true)
    if y_prob.ndim != 2:
        raise RuntimeError(f"Expected classification output rank-2, got {y_prob.shape}")

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
        loss = -np.mean(true * np.log(probs + eps) + (1 - true) * np.log(1.0 - probs + eps))
    else:
        probs = np.clip(y_prob, eps, 1.0)
        probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), eps, None)
        true = y_true.reshape(-1).astype(np.int32)
        pred = np.argmax(probs, axis=-1).astype(np.int32)

        accuracy = np.mean(pred == true)
        if probs.shape[-1] == 2:
            tp = np.sum((pred == 1) & (true == 1))
            fp = np.sum((pred == 1) & (true == 0))
            fn = np.sum((pred == 0) & (true == 1))
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        loss = -np.mean(np.log(np.clip(probs[np.arange(true.shape[0]), true], eps, 1.0)))

    return {
        "accuracy": _safe_float(accuracy),
        "pixel_accuracy": _safe_float(accuracy),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "sensitivity": _safe_float(recall),
        "f1_score": _safe_float(f1),
        "test_loss": _safe_float(loss),
    }


def _fallback_evaluate_models(hdf5_path):
    import h5py
    import numpy as np
    import tensorflow as tf

    model_dir = globals().get("MODEL_DIR", os.path.join(OUTPUT_DIR, "models"))
    results_dir = globals().get("RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))
    os.makedirs(results_dir, exist_ok=True)

    threshold = _fallback_load_threshold(results_dir)
    custom_objects = _fallback_custom_objects()
    out = {}

    with h5py.File(hdf5_path, "r") as h5_file:
        split = "test" if "test/images" in h5_file and "test/masks" in h5_file else "val"
        if split == "val":
            print("[WARN] test split not found; evaluating on val split instead.")

        images_ds = h5_file[f"{split}/images"]
        masks_ds = h5_file[f"{split}/masks"]
        n = int(images_ds.shape[0])
        chunk = 8

        model_specs = [
            ("U-Net", "unet_best.keras", "segmentation"),
            ("Attention U-Net", "attention_unet_best.keras", "segmentation"),
            ("Attention U-Net + ViT (Proposed)", "attention_unet_vit_best.keras", "classification"),
        ]

        for key, filename, task_type in model_specs:
            model_path = os.path.join(model_dir, filename)
            if not os.path.exists(model_path):
                continue

            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            metric_sums = {}
            total = 0

            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                x_batch = images_ds[start:end].astype(np.float32)
                y_batch = masks_ds[start:end].astype(np.float32)
                pred = model.predict(x_batch, verbose=0).astype(np.float32)
                if task_type == "classification":
                    if pred.ndim != 2:
                        raise RuntimeError(
                            f"Expected classification output rank-2 for {key}, got {pred.shape}"
                        )
                    num_classes = int(pred.shape[-1]) if pred.shape[-1] is not None else 1
                    y_cls = (np.max(y_batch, axis=(1, 2, 3)) > 0.5).astype(np.int32)
                    if num_classes == 1:
                        y_cls = y_cls.astype(np.float32).reshape((-1, 1))
                    batch_metrics = _fallback_classification_batch_metrics(y_cls, pred, threshold=threshold)
                else:
                    batch_metrics = _fallback_binary_batch_metrics(y_batch, pred, threshold=threshold)

                bs = int(end - start)
                total += bs
                for name, value in batch_metrics.items():
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * bs

            if total > 0:
                out[key] = {name: float(value / total) for name, value in metric_sums.items()}
                out[key]["threshold"] = float(threshold)
                out[key]["num_samples"] = int(total)
                out[key]["task_type"] = task_type
                out[key]["primary_metric"] = "f1_score" if task_type == "classification" else "dice_coef_soft"

    save_path = os.path.join(results_dir, "comparison_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


_ensure_project_root_on_path()
Config = _load_config_class()
evaluate_models = _load_evaluate_models_fn()


def _pick_existing_path(candidates, want_file=False, want_dir=False):
    """Pick the first existing candidate path, otherwise first non-empty fallback."""
    normalized = []
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(str(candidate))
        if path not in normalized:
            normalized.append(path)

    for path in normalized:
        if want_file and os.path.isfile(path):
            return path
        if want_dir and os.path.isdir(path):
            return path
        if not want_file and not want_dir and os.path.exists(path):
            return path

    return normalized[0] if normalized else ""


DEFAULT_OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUTPUT_DIR = _pick_existing_path(
    [
        globals().get('OUTPUT_DIR'),
        os.environ.get('OUTPUT_DIR'),
        getattr(Config, 'OUTPUT_DIR', None),
        DEFAULT_OUTPUT_DIR,
        os.getcwd(),
    ],
    want_dir=True,
)
MODEL_DIR = _pick_existing_path(
    [
        globals().get('MODEL_DIR'),
        os.environ.get('MODEL_DIR'),
        os.path.join(OUTPUT_DIR, 'models'),
        os.path.join(os.getcwd(), 'models'),
        getattr(Config, 'MODEL_DIR', None),
    ],
    want_dir=True,
)
RESULTS_DIR = _pick_existing_path(
    [
        globals().get('RESULTS_DIR'),
        os.environ.get('RESULTS_DIR'),
        os.path.join(OUTPUT_DIR, 'results'),
        os.path.join(os.getcwd(), 'results'),
        getattr(Config, 'RESULTS_DIR', None),
    ],
    want_dir=True,
)
HDF5_PATH_CANDIDATES = [
    globals().get('HDF5_PATH'),
    os.environ.get('HDF5_PATH'),
    os.path.join(OUTPUT_DIR, 'brats_preprocessed.h5'),
    os.path.join(os.getcwd(), 'brats_preprocessed.h5'),
    getattr(Config, 'HDF5_PATH', None),
]
HDF5_PATH = _pick_existing_path(HDF5_PATH_CANDIDATES, want_file=True)

if RESULTS_DIR:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _history_best_val_dice_coef(results_dir=RESULTS_DIR):
    """Return best val_dice_coef and epoch from history JSON files when available."""
    hist_files = {
        'U-Net': os.path.join(results_dir, 'unet_history.json'),
        'Attention U-Net': os.path.join(results_dir, 'attention_unet_history.json'),
    }
    best = {}

    for model_name, hist_path in hist_files.items():
        if not os.path.exists(hist_path):
            continue
        with open(hist_path, 'r', encoding='utf-8') as f:
            hist = json.load(f)
        vals = hist.get('val_dice_coef', []) or []
        if vals:
            best_val = max(vals)
            best_epoch = vals.index(best_val) + 1
            best[model_name] = (float(best_val), int(best_epoch))

    return best


def print_evaluation_summary(results, results_dir=RESULTS_DIR):
    if not isinstance(results, dict) or not results:
        print('[WARN] No evaluation metrics found to summarize.')
        return

    print('\n' + '=' * 60)
    print('CELL 10: EVALUATION SUMMARY (BINARY)')
    print('=' * 60)

    for model_name, metrics in results.items():
        task_type = str(metrics.get('task_type', 'segmentation')).lower()
        precision = _safe_float(metrics.get('precision'))
        recall = _safe_float(metrics.get('recall', metrics.get('sensitivity', 0.0)))
        f1_score = _safe_float(metrics.get('f1_score'))
        pixel_acc = _safe_float(metrics.get('pixel_accuracy', metrics.get('accuracy', 0.0)))
        test_loss = _safe_float(metrics.get('test_loss'))

        print(f'\n{model_name} ({task_type}):')
        if task_type == 'classification':
            print(f'  Accuracy: {pixel_acc:.4f}')
        else:
            print(f'  Dice (soft): {_safe_float(metrics.get("dice_coef_soft")):.4f}')
            print(f'  Dice (hard): {_safe_float(metrics.get("dice_coef_hard")):.4f}')
            print(f'  IoU: {_safe_float(metrics.get("iou")):.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1 Score: {f1_score:.4f}')
        print(f'  Pixel Accuracy: {pixel_acc:.4f}')
        print(f'  Test Loss: {test_loss:.4f}')

    unet = results.get('U-Net')
    attn = results.get('Attention U-Net')
    proposed = results.get('Attention U-Net + ViT (Proposed)')

    if unet and attn:
        u_soft = _safe_float(unet.get('dice_coef_soft'))
        a_soft = _safe_float(attn.get('dice_coef_soft'))
        winner = 'U-Net' if u_soft >= a_soft else 'Attention U-Net'
        abs_gap = abs(u_soft - a_soft)
        rel_gap = (abs_gap / max(u_soft, a_soft, 1e-8)) * 100.0

        print('\n--- Existing Models (Segmentation) ---')
        print(f'Winner by Dice (soft): {winner}')
        print(f'Absolute gap (Dice soft): {abs_gap:.4f}')
        if winner == 'U-Net':
            print(f'Relative gap: Attention U-Net {rel_gap:.2f}% below U-Net')
        else:
            print(f'Relative gap: U-Net {rel_gap:.2f}% below Attention U-Net')

    if proposed:
        p_task = str(proposed.get('task_type', 'classification')).lower()
        p_metric = _safe_float(
            proposed.get('f1_score' if p_task == 'classification' else 'dice_coef_soft', 0.0)
        )
        p_metric_name = 'F1 Score' if p_task == 'classification' else 'Dice (soft)'
        print('\n--- Proposed Model Comparison ---')
        print(f"Proposed model: Attention U-Net + ViT ({p_task})")
        print(f"Primary metric ({p_metric_name}): {p_metric:.4f}")
        print('Note: Proposed model is classification-focused; compare by F1/accuracy and test loss.')

    history_best = _history_best_val_dice_coef(results_dir)
    if history_best:
        print('')
        print('Best val_dice_coef from training history:')
        for model_name in ['U-Net', 'Attention U-Net']:
            if model_name in history_best:
                best_val, best_epoch = history_best[model_name]
                print(f'  {model_name}: {best_val:.4f} (epoch {best_epoch})')

    print('=' * 60)


def plot_training_curves(results_dir=RESULTS_DIR):
    history_specs = [
        ('U-Net', os.path.join(results_dir, 'unet_history.json')),
        ('Attention U-Net', os.path.join(results_dir, 'attention_unet_history.json')),
        ('Attention U-Net + ViT (Proposed)', os.path.join(results_dir, 'attention_unet_vit_history.json')),
    ]

    histories = {}
    for name, path in history_specs:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                histories[name] = json.load(f)

    if not histories:
        print('[WARN] No training history files found. Skipping plots.')
        return

    # 1) Training/validation loss graph.
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 5))
    for name, h in histories.items():
        if h.get('loss'):
            ax_loss.plot(h.get('loss', []), label=f'{name} train')
        if h.get('val_loss'):
            ax_loss.plot(h.get('val_loss', []), '--', label=f'{name} val')
    ax_loss.set_title('Training vs Validation Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    fig_loss.tight_layout()
    loss_path = os.path.join(results_dir, 'training_loss_comparison.png')
    fig_loss.savefig(loss_path, dpi=150)
    # Backward-compatible artifact name.
    fig_loss.savefig(os.path.join(results_dir, 'training_comparison.png'), dpi=150)
    plt.show()
    plt.close(fig_loss)
    print(f'[OK] Saved plot: {loss_path}')

    # 2) Primary training metric graph per model.
    fig_metric, ax_metric = plt.subplots(1, 1, figsize=(10, 5))
    used_any_metric = False
    for name, h in histories.items():
        metric_pairs = [
            ('dice_coef', 'val_dice_coef', 'Dice'),
            ('binary_accuracy', 'val_binary_accuracy', 'Binary Accuracy'),
            ('sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', 'Sparse Categorical Accuracy'),
        ]
        for train_key, val_key, _ in metric_pairs:
            train_vals = h.get(train_key, [])
            val_vals = h.get(val_key, [])
            if train_vals or val_vals:
                if train_vals:
                    ax_metric.plot(train_vals, label=f'{name} train:{train_key}')
                if val_vals:
                    ax_metric.plot(val_vals, '--', label=f'{name} val:{val_key}')
                used_any_metric = True
                break

    if used_any_metric:
        ax_metric.set_title('Primary Training Metrics by Model')
        ax_metric.set_xlabel('Epoch')
        ax_metric.set_ylabel('Metric')
        ax_metric.legend(fontsize=8)
        fig_metric.tight_layout()
        metric_path = os.path.join(results_dir, 'training_metric_comparison.png')
        fig_metric.savefig(metric_path, dpi=150)
        plt.show()
        print(f'[OK] Saved plot: {metric_path}')
    else:
        print('[WARN] No metric curves found in history files.')
    plt.close(fig_metric)


def plot_test_loss_comparison(results, results_dir=RESULTS_DIR):
    if not isinstance(results, dict) or not results:
        print('[WARN] No evaluation results available for test-loss graph.')
        return

    labels = []
    values = []
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        if 'test_loss' not in metrics:
            continue
        labels.append(model_name)
        values.append(_safe_float(metrics.get('test_loss', 0.0), default=0.0))

    if not labels:
        print('[WARN] No test_loss values found. Skipping test-loss graph.')
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title('Test Loss Comparison (Existing vs Proposed)')
    ax.set_ylabel('Test Loss')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    fig.tight_layout()
    out_path = os.path.join(results_dir, 'test_loss_comparison.png')
    fig.savefig(out_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[OK] Saved plot: {out_path}')


def main():
    if not os.path.exists(HDF5_PATH):
        print(f'[ERR] HDF5 not found: {HDF5_PATH}')
        checked = [os.path.abspath(str(p)) for p in HDF5_PATH_CANDIDATES if p]
        if checked:
            print('Checked paths:')
            for path in checked:
                print(f'  - {path}')
        print('Run Cell 5 first to build dataset.')
        return None

    results = evaluate_models(HDF5_PATH)
    print('[OK] comparison_metrics.json written by ml_pipeline.evaluate')
    print_evaluation_summary(results, RESULTS_DIR)
    plot_training_curves(RESULTS_DIR)
    plot_test_loss_comparison(results, RESULTS_DIR)
    print('[OK] Cell 10 complete. Evaluation finished.')
    return results


if __name__ == '__main__':
    EVAL_RESULTS = main()
