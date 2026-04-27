# ===================================================
# CELL 11: Inference Pipeline (MULTIMODAL + BINARY)
# ===================================================

import json
import os
import importlib
import importlib.util
import io
from contextlib import suppress
from contextlib import redirect_stdout, redirect_stderr

import cv2
import nibabel as nib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    from scipy import ndimage
except Exception:
    ndimage = None


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

REQUIRED_PREPROC_SYMBOLS = ["load_multimodal_volume", "preprocess_multimodal_slice"]


def _safe_import_module(module_name, module_filename=None):
    with suppress(Exception):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return importlib.import_module(module_name)

    if module_filename is None:
        return None

    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend([os.getcwd(), "/kaggle/working"])

    for base in candidates:
        module_path = os.path.join(base, module_filename)
        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            with suppress(Exception):
                module = importlib.util.module_from_spec(spec)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    spec.loader.exec_module(module)
                return module

    return None


if any(sym not in globals() for sym in REQUIRED_PREPROC_SYMBOLS):
    preproc_module = None
    preproc_candidates = (
        ("cell_04_preprocessing_FIXED", "cell_04_preprocessing_FIXED.py"),
        ("cell_04_preprocessing", "cell_04_preprocessing.py"),
    )
    for module_name, module_filename in preproc_candidates:
        preproc_module = _safe_import_module(module_name, module_filename=module_filename)
        if preproc_module is not None:
            break
    if preproc_module is None:
        raise ImportError(
            "Could not import preprocessing helpers. Expected one of: "
            "cell_04_preprocessing_FIXED.py, cell_04_preprocessing.py"
        )
    load_multimodal_volume = preproc_module.load_multimodal_volume
    preprocess_multimodal_slice = preproc_module.preprocess_multimodal_slice


# ========================
# DEFAULT VARIABLES
# ========================
DEFAULT_OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUTPUT_DIR = globals().get("OUTPUT_DIR", os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
MODEL_DIR = globals().get("MODEL_DIR", os.path.join(OUTPUT_DIR, "models"))
RESULTS_DIR = globals().get("RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))

MIN_TUMOR_SIZE = int(globals().get("MIN_TUMOR_SIZE", os.environ.get("MIN_TUMOR_SIZE", "50")))
CONFIDENCE_THRESHOLD = float(
    globals().get("CONFIDENCE_THRESHOLD", os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
)
INFERENCE_BATCH_SIZE = int(globals().get("INFERENCE_BATCH_SIZE", os.environ.get("INFERENCE_BATCH_SIZE", "8")))


def _load_saved_threshold(results_dir=RESULTS_DIR):
    """Load tuned binary threshold saved by Cell 9, if available."""
    thresholds_path = os.path.join(results_dir, "optimal_thresholds.json")
    if not os.path.exists(thresholds_path):
        return None

    try:
        with open(thresholds_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            if "binary" in payload:
                return float(payload["binary"])
        if isinstance(payload, (list, tuple)) and len(payload) >= 1:
            return float(payload[0])
    except Exception as exc:
        print(f"[WARN] Could not load tuned threshold from {thresholds_path}: {exc}")

    return None


_saved_threshold = _load_saved_threshold()
if _saved_threshold is not None:
    CONFIDENCE_THRESHOLD = _saved_threshold
    print(f"[OK] Loaded tuned binary threshold: {CONFIDENCE_THRESHOLD:.2f}")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ========================
# MODEL LOADING
# ========================
def _get_custom_objects():
    objs = {}

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

    with suppress(Exception):
        from cell_07d_attention_unet_vit_FIXED import get_attention_unet_vit_custom_objects

        objs.update(get_attention_unet_vit_custom_objects())
    return objs


def load_inference_model(model_path=None, prefer_attention=True, use_ensemble=True, prefer_classifier=False):
    if model_path is None and use_ensemble:
        with suppress(Exception):
            from ensemble import load_ensemble_model

            ensemble_model = load_ensemble_model(model_dir=MODEL_DIR)
            if ensemble_model is not None:
                print("[OK] Using ensemble model for inference")
                return ensemble_model

    if model_path is None:
        classifier_path = os.path.join(MODEL_DIR, "attention_unet_vit_best.keras")
        attention_path = os.path.join(MODEL_DIR, "attention_unet_best.keras")
        unet_path = os.path.join(MODEL_DIR, "unet_best.keras")

        if prefer_classifier:
            candidates = [classifier_path, attention_path, unet_path]
        else:
            candidates = [attention_path, unet_path] if prefer_attention else [unet_path, attention_path]
        model_path = next((path for path in candidates if os.path.exists(path)), None)

        if model_path is None:
            print("[ERROR] No trained model found. Run Cell 9 first.")
            return None

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    print(f"Loading model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[OK] Model loaded with compile=False")
        return model
    except Exception as exc:
        print(f"[WARN] compile=False load failed: {exc}")

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=_get_custom_objects(), compile=False)
        print("[OK] Model loaded with custom_objects")
        return model
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")
        return None


# ========================
# INPUT DISCOVERY AND PREPROCESSING
# ========================
def _is_nonempty_brain_slice(flair_slice, min_nonzero_ratio=0.01):
    if flair_slice is None:
        return False
    nonzero_ratio = float(np.count_nonzero(flair_slice)) / float(flair_slice.size + 1e-8)
    return nonzero_ratio >= min_nonzero_ratio


def preprocess_patient_for_inference(patient_dir, min_nonzero_ratio=0.01):
    if not os.path.isdir(patient_dir):
        print(f"[ERROR] Invalid patient directory: {patient_dir}")
        return None, None, None, None, None

    print(f"Preprocessing patient: {os.path.basename(patient_dir)}")

    volumes, _, affine, header = load_multimodal_volume(patient_dir, require_seg=False)
    if volumes is None:
        print("[ERROR] Missing required modalities (flair/t1/t1ce/t2).")
        return None, None, None, None, None

    expected_keys = {"flair", "t1", "t1ce", "t2"}
    if set(volumes.keys()) != expected_keys:
        print(f"[ERROR] Modality mismatch. Found keys: {sorted(volumes.keys())}")
        return None, None, None, None, None

    original_shape = volumes["flair"].shape
    if len(original_shape) != 3:
        print(f"[ERROR] Expected 3D volume. Found shape: {original_shape}")
        return None, None, None, None, None

    num_slices = int(original_shape[2])
    processed_slices = []
    valid_indices = []

    for slice_idx in range(num_slices):
        modality_slices = {mod: volumes[mod][:, :, slice_idx] for mod in ["flair", "t1", "t1ce", "t2"]}

        if not _is_nonempty_brain_slice(modality_slices["flair"], min_nonzero_ratio=min_nonzero_ratio):
            continue

        image, _ = preprocess_multimodal_slice(
            modality_slices,
            np.zeros_like(modality_slices["flair"], dtype=np.float32),
        )

        if image is None or image.ndim != 3 or image.shape[-1] != 4:
            continue

        processed_slices.append(image.astype(np.float32, copy=False))
        valid_indices.append(slice_idx)

    if not processed_slices:
        print("[ERROR] No valid non-empty slices found for inference.")
        return None, None, None, None, None

    images = np.asarray(processed_slices, dtype=np.float32)
    print(f"[OK] Preprocessed {images.shape[0]} slices | input tensor shape: {images.shape}")

    return images, valid_indices, affine, header, original_shape


# ========================
# PREDICTION CORE
# ========================
def _predict_batch(model, x, training=False):
    x = tf.cast(x, tf.float32)
    if training:
        with suppress(Exception):
            y = model(x, training=True)
            return y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return model.predict(x, batch_size=INFERENCE_BATCH_SIZE, verbose=0)


def _flip_batch(x, mode):
    if mode == "h":
        return np.flip(x, axis=2)
    if mode == "v":
        return np.flip(x, axis=1)
    if mode == "hv":
        return np.flip(np.flip(x, axis=1), axis=2)
    return x


def _unflip_batch(y, mode):
    if mode == "h":
        return np.flip(y, axis=2)
    if mode == "v":
        return np.flip(y, axis=1)
    if mode == "hv":
        return np.flip(np.flip(y, axis=1), axis=2)
    if mode == "rot90":
        return np.rot90(y, k=-1, axes=(1, 2))
    if mode == "rot180":
        return np.rot90(y, k=-2, axes=(1, 2))
    if mode == "rot270":
        return np.rot90(y, k=-3, axes=(1, 2))
    return y


def predict_probabilities(model, images, use_tta=True, tta_modes=None):
    if tta_modes is None:
        tta_modes = ["h", "v", "hv", "rot90", "rot180", "rot270"]

    base_pred = _predict_batch(model, images, training=False).astype(np.float32)
    classification_mode = base_pred.ndim == 2
    if (not classification_mode) and base_pred.shape[-1] != 1:
        base_pred = np.max(base_pred, axis=-1, keepdims=True)

    if not use_tta:
        return base_pred

    preds = [base_pred]
    for mode in tta_modes:
        if mode in ("h", "v", "hv"):
            aug_x = _flip_batch(images, mode)
        elif mode == "rot90":
            aug_x = np.rot90(images, k=1, axes=(1, 2)).copy()
        elif mode == "rot180":
            aug_x = np.rot90(images, k=2, axes=(1, 2)).copy()
        elif mode == "rot270":
            aug_x = np.rot90(images, k=3, axes=(1, 2)).copy()
        else:
            continue
        aug_y = _predict_batch(model, aug_x, training=False).astype(np.float32)
        if classification_mode:
            preds.append(aug_y)
        else:
            if aug_y.shape[-1] != 1:
                aug_y = np.max(aug_y, axis=-1, keepdims=True)
            preds.append(_unflip_batch(aug_y, mode))

    stacked = np.stack(preds, axis=0)
    return np.mean(stacked, axis=0)


def predict_with_uncertainty(model, images, mc_passes=0, use_tta=False, tta_modes=None):
    mean_probs = predict_probabilities(model, images, use_tta=use_tta, tta_modes=tta_modes)

    uncertainty = None
    if mc_passes and mc_passes > 1:
        stochastic_preds = []
        for _ in range(int(mc_passes)):
            stochastic = _predict_batch(model, images, training=True).astype(np.float32)
            if stochastic.shape[-1] != 1:
                stochastic = np.max(stochastic, axis=-1, keepdims=True)
            stochastic_preds.append(stochastic)

        mc_stack = np.stack(stochastic_preds, axis=0)
        uncertainty = {
            "std_map": np.std(mc_stack, axis=0).astype(np.float32),
            "mean_std": float(np.mean(np.std(mc_stack, axis=0))),
            "max_std": float(np.max(np.std(mc_stack, axis=0))),
        }

    return mean_probs, uncertainty


# ========================
# POST-PROCESSING
# ========================
def _remove_small_components_per_slice(binary_masks, min_tumor_size):
    if ndimage is None:
        print("[WARN] scipy.ndimage unavailable; skipping small-component filtering.")
        return binary_masks

    out = binary_masks.copy()
    for i in range(out.shape[0]):
        labeled, num_components = ndimage.label(out[i, :, :, 0])
        for comp_id in range(1, num_components + 1):
            comp = labeled == comp_id
            if int(np.sum(comp)) < int(min_tumor_size):
                out[i, :, :, 0][comp] = 0
    return out


def postprocess_predictions(probabilities, threshold=None, min_tumor_size=MIN_TUMOR_SIZE):
    if probabilities is None or probabilities.ndim != 4:
        raise ValueError(f"Invalid probabilities shape: {None if probabilities is None else probabilities.shape}")

    probs = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    if probs.shape[-1] != 1:
        probs = np.max(probs, axis=-1, keepdims=True)

    thr = float(CONFIDENCE_THRESHOLD if threshold is None else threshold)
    binary_masks = (probs >= thr).astype(np.float32)

    if min_tumor_size > 0:
        binary_masks = _remove_small_components_per_slice(binary_masks, min_tumor_size)

    return binary_masks


# ========================
# RECONSTRUCTION AND EXPORT
# ========================
def reconstruct_3d_volume(predictions, slice_indices, original_shape):
    if len(slice_indices) != predictions.shape[0]:
        raise ValueError(
            f"Slice index count mismatch. indices={len(slice_indices)} preds={predictions.shape[0]}"
        )

    h, w, d = original_shape
    volume = np.zeros((h, w, d, 1), dtype=np.float32)

    for i, slice_idx in enumerate(slice_indices):
        pred_slice = predictions[i]

        if pred_slice.shape[0] != h or pred_slice.shape[1] != w:
            resized = cv2.resize(
                pred_slice[:, :, 0],
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            pred_slice = resized[..., np.newaxis]

        volume[:, :, int(slice_idx), :] = pred_slice

    return volume


def save_prediction_nifti(volume, affine, header, output_path):
    if volume is None or volume.ndim != 4:
        raise ValueError(f"Invalid volume shape for NIfTI export: {None if volume is None else volume.shape}")

    vol = volume.astype(np.float32)
    if vol.shape[-1] != 1:
        vol = np.max(vol, axis=-1, keepdims=True)

    label_volume = (vol[..., 0] > CONFIDENCE_THRESHOLD).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nii_image = nib.Nifti1Image(label_volume, affine, header)
    nib.save(nii_image, output_path)

    return output_path


# ========================
# HIGH-LEVEL APIS
# ========================
def predict_patient(
    patient_dir,
    model=None,
    output_dir=None,
    threshold=None,
    min_tumor_size=MIN_TUMOR_SIZE,
    use_tta=True,
    tta_modes=None,
    mc_passes=0,
):
    print("=" * 70)
    print("PATIENT INFERENCE")
    print("=" * 70)

    if model is None:
        model = load_inference_model()
    if model is None:
        return None, None

    images, slice_indices, affine, header, original_shape = preprocess_patient_for_inference(patient_dir)
    if images is None:
        return None, None

    probabilities, uncertainty = predict_with_uncertainty(
        model,
        images,
        mc_passes=mc_passes,
        use_tta=use_tta,
        tta_modes=tta_modes,
    )

    cleaned = postprocess_predictions(
        probabilities,
        threshold=threshold,
        min_tumor_size=min_tumor_size,
    )

    volume_3d = reconstruct_3d_volume(cleaned, slice_indices, original_shape)

    counts = {
        "tumor": int(np.sum(volume_3d[..., 0] > 0.5)),
    }

    if output_dir is None:
        output_dir = patient_dir
    os.makedirs(output_dir, exist_ok=True)

    output_nifti_path = os.path.join(output_dir, "predicted_mask.nii.gz")
    save_prediction_nifti(volume_3d, affine, header, output_nifti_path)

    print(f"[OK] Prediction saved: {output_nifti_path}")
    print(f"[OK] Tumor voxels: {counts['tumor']:,}")

    if uncertainty is not None:
        uncertainty_path = os.path.join(output_dir, "uncertainty_summary.json")
        with open(uncertainty_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mean_std": uncertainty["mean_std"],
                    "max_std": uncertainty["max_std"],
                    "mc_passes": int(mc_passes),
                },
                handle,
                indent=2,
            )
        print(f"[OK] Uncertainty summary saved: {uncertainty_path}")

    return volume_3d, counts


def predict_patient_classification(
    patient_dir,
    model=None,
    output_dir=None,
    threshold=None,
    use_tta=False,
    tta_modes=None,
    mc_passes=0,
):
    """Run slice-level classification and aggregate into one patient-level decision."""
    print("=" * 70)
    print("PATIENT CLASSIFICATION INFERENCE")
    print("=" * 70)

    if model is None:
        model = load_inference_model(use_ensemble=False, prefer_classifier=True)
    if model is None:
        return None

    images, slice_indices, _, _, _ = preprocess_patient_for_inference(patient_dir)
    if images is None:
        return None

    probabilities, uncertainty = predict_with_uncertainty(
        model,
        images,
        mc_passes=mc_passes,
        use_tta=use_tta,
        tta_modes=tta_modes,
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            "Classification model expected output rank-2 (N,C). "
            f"Got shape={probabilities.shape}."
        )

    thr = float(CONFIDENCE_THRESHOLD if threshold is None else threshold)
    num_classes = int(probabilities.shape[-1])

    if num_classes == 1:
        slice_scores = probabilities[:, 0].astype(np.float32)
        slice_labels = (slice_scores >= thr).astype(np.int32)
        patient_score = float(np.max(slice_scores))
        patient_label = int(patient_score >= thr)
        patient_probs = [float(1.0 - patient_score), float(patient_score)]
    else:
        slice_labels = np.argmax(probabilities, axis=-1).astype(np.int32)
        patient_prob_vec = np.mean(probabilities, axis=0).astype(np.float32)
        patient_prob_vec = patient_prob_vec / max(np.sum(patient_prob_vec), 1e-8)
        patient_label = int(np.argmax(patient_prob_vec))
        patient_score = float(patient_prob_vec[patient_label])
        patient_probs = [float(v) for v in patient_prob_vec]

    summary = {
        "num_slices": int(probabilities.shape[0]),
        "num_classes": num_classes,
        "slice_indices": [int(v) for v in slice_indices],
        "slice_labels": [int(v) for v in slice_labels.tolist()],
        "patient_label": int(patient_label),
        "patient_score": float(patient_score),
        "patient_probabilities": patient_probs,
        "threshold": thr,
    }

    if uncertainty is not None:
        summary["uncertainty"] = {
            "mean_std": float(uncertainty.get("mean_std", 0.0)),
            "max_std": float(uncertainty.get("max_std", 0.0)),
            "mc_passes": int(mc_passes),
        }

    if output_dir is None:
        output_dir = patient_dir
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "classification_summary.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[OK] Classification summary saved: {output_json}")
    print(f"[OK] Patient label: {summary['patient_label']} | score: {summary['patient_score']:.4f}")
    return summary


def predict_multiple_patients(
    patient_dirs,
    model=None,
    output_base_dir=RESULTS_DIR,
    use_tta=True,
    mc_passes=0,
):
    if model is None:
        model = load_inference_model()
    if model is None:
        return {}

    os.makedirs(output_base_dir, exist_ok=True)
    results = {}

    for patient_dir in tqdm(patient_dirs, desc="Inference"):
        patient_id = os.path.basename(patient_dir.rstrip("/\\"))
        out_dir = os.path.join(output_base_dir, "predictions", patient_id)

        try:
            _, counts = predict_patient(
                patient_dir,
                model=model,
                output_dir=out_dir,
                use_tta=use_tta,
                mc_passes=mc_passes,
            )
            results[patient_id] = counts if counts is not None else {"error": "prediction_failed"}
        except Exception as exc:
            results[patient_id] = {"error": str(exc)}

    print(f"[OK] Batch inference completed for {len(results)} patients")
    return results


def _print_cell11_ready_message():
    print("=" * 70)
    print("CELL 11 READY")
    print("=" * 70)
    print("[OK] Inference functions are loaded.")
    print("[INFO] No prediction runs automatically in this cell.")
    print("[INFO] Run one-patient inference like this:")
    print("  model = load_inference_model(use_ensemble=False)")
    print("  vol, counts = predict_patient('/path/to/patient', model=model,")
    print("      output_dir='/kaggle/working/results/predictions/debug_one',")
    print("      use_tta=False, mc_passes=0)")
    print("  print(counts)")
    print("=" * 70)


def run_cell11_smoke_once_from_env():
    """Optional smoke run controlled by environment variables.

    Required:
      CELL11_PATIENT_DIR=/path/to/patient

    Optional:
      CELL11_OUTPUT_DIR=/kaggle/working/results/predictions/cell11_smoke
      CELL11_USE_TTA=0|1
      CELL11_MC_PASSES=0|N
      CELL11_USE_ENSEMBLE=0|1
    """
    patient_dir = os.environ.get("CELL11_PATIENT_DIR", "").strip()
    if not patient_dir:
        print("[INFO] CELL11_PATIENT_DIR not set. Skipping smoke inference.")
        return None

    output_dir = os.environ.get(
        "CELL11_OUTPUT_DIR",
        os.path.join(RESULTS_DIR, "predictions", "cell11_smoke"),
    )
    use_tta = os.environ.get("CELL11_USE_TTA", "0") == "1"
    mc_passes = int(os.environ.get("CELL11_MC_PASSES", "0"))
    use_ensemble = os.environ.get("CELL11_USE_ENSEMBLE", "0") == "1"

    print(f"[INFO] CELL11 smoke inference -> patient: {patient_dir}")
    model = load_inference_model(use_ensemble=use_ensemble)
    if model is None:
        print("[ERROR] Smoke inference aborted: model could not be loaded.")
        return None

    _, counts = predict_patient(
        patient_dir,
        model=model,
        output_dir=output_dir,
        use_tta=use_tta,
        mc_passes=mc_passes,
    )

    print(f"[OK] CELL11 smoke inference counts: {counts}")
    return counts


if __name__ == "__main__":
    _print_cell11_ready_message()
    if os.environ.get("CELL11_AUTORUN", "0") == "1":
        run_cell11_smoke_once_from_env()
