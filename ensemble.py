# ===================================================
# MODEL ENSEMBLE: Multi-Model Prediction Averaging
# ===================================================
# Purpose: Combine predictions from multiple trained models to reduce
#          variance and improve robustness for binary tumor segmentation.
#
# Usage:
#   from ensemble import ModelEnsemble, load_ensemble_model
#   ensemble = load_ensemble_model()
#   probs = ensemble.predict(images)
# ===================================================

import os
from contextlib import suppress

import numpy as np
import tensorflow as tf


class ModelEnsemble:
    """Weighted ensemble of multiple Keras segmentation models.

    Combines predictions from N models via weighted averaging.
    Each model's prediction is a probability map (N, H, W, 1).

    Benefits:
      - Reduces variance → more stable Dice across patients
    - Different seeds capture different features → better tumor recall
      - Combining U-Net + Attention U-Net leverages architectural diversity
    """

    def __init__(self, models, weights=None):
        """
        Args:
            models: list of loaded Keras models
            weights: optional array of weights (auto-normalized to sum=1)
        """
        if not models:
            raise ValueError("ModelEnsemble requires at least one model.")

        self.models = models
        self.n_models = len(models)

        if weights is None:
            self.weights = np.ones(self.n_models, dtype=np.float32) / self.n_models
        else:
            w = np.array(weights, dtype=np.float32)
            self.weights = w / w.sum()

        print(f"[ENSEMBLE] {self.n_models} model(s), weights={self.weights.tolist()}")

    def predict(self, x, batch_size=8, verbose=0):
        """Weighted average of predictions from all models.

        Args:
            x: input array (N, H, W, 4)
            batch_size: prediction batch size per model
            verbose: Keras predict verbosity

        Returns:
            weighted average probabilities (N, H, W, 1)
        """
        weighted_sum = None

        for i, model in enumerate(self.models):
            pred = model.predict(x, batch_size=batch_size, verbose=verbose)
            pred = pred.astype(np.float32)

            if weighted_sum is None:
                weighted_sum = pred * self.weights[i]
            else:
                weighted_sum += pred * self.weights[i]

        return weighted_sum

    def predict_with_tta(self, x, batch_size=8, tta_modes=None):
        """Ensemble predictions + TTA for maximum accuracy.

        Each model runs with TTA independently, then results are ensembled.
        This is the highest-quality prediction mode (slowest).
        """
        # Lazy import to avoid circular dependency
        from cell_11_inference_FIXED import predict_probabilities

        weighted_sum = None

        for i, model in enumerate(self.models):
            pred = predict_probabilities(model, x, use_tta=True, tta_modes=tta_modes)
            pred = pred.astype(np.float32)

            if weighted_sum is None:
                weighted_sum = pred * self.weights[i]
            else:
                weighted_sum += pred * self.weights[i]

        return weighted_sum


def _try_load_model(path, custom_objects=None):
    """Try to load a model, returning None on failure."""
    if not os.path.exists(path):
        return None

    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception:
        pass

    if custom_objects:
        try:
            model = tf.keras.models.load_model(
                path, custom_objects=custom_objects, compile=False
            )
            return model
        except Exception:
            pass

    return None


def _get_custom_objects():
    """Import custom objects for model loading."""
    objs = {}
    with suppress(ImportError):
        from cell_08_loss_metrics_FIXED import (
            combined_loss, dice_coef,
            precision_metric, sensitivity_metric, iou_metric,
        )
        objs.update({
            "combined_loss": combined_loss,
            "dice_coef": dice_coef,
            "precision_metric": precision_metric,
            "sensitivity_metric": sensitivity_metric,
            "iou_metric": iou_metric,
        })

    with suppress(ImportError):
        from cell_07a_building_blocks_FIXED import _hierarchy_sort
        objs["_hierarchy_sort"] = _hierarchy_sort

    return objs


def load_ensemble_model(model_dir=None, model_paths=None, weights=None):
    """Load ensemble of best available models.

    Searches for standard model filenames if model_paths is not provided.
    Falls back to single model if only one is found.

    Args:
        model_dir: directory containing *.keras model files
        model_paths: explicit list of model file paths
        weights: optional per-model weights

    Returns:
        ModelEnsemble instance (or single model if only 1 found)
    """
    if model_dir is None:
        default_output = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        model_dir = os.path.join(default_output, "models")

    custom_objects = _get_custom_objects()

    if model_paths is None:
        # Standard model filenames to look for
        candidate_names = [
            "attention_unet_best.keras",
            "unet_best.keras",
            # Multi-seed models (if trained with different seeds)
            "unet_seed42_best.keras",
            "unet_seed123_best.keras",
            "unet_seed456_best.keras",
            "attention_unet_seed42_best.keras",
            "attention_unet_seed123_best.keras",
        ]
        model_paths = [os.path.join(model_dir, name) for name in candidate_names]

    # Load all available models
    models = []
    loaded_paths = []
    for path in model_paths:
        model = _try_load_model(path, custom_objects)
        if model is not None:
            models.append(model)
            loaded_paths.append(path)
            print(f"[ENSEMBLE] Loaded: {os.path.basename(path)}")

    if not models:
        print("[ERROR] No models found for ensemble. Run Cell 9 training first.")
        print(f"[ERROR] Searched in: {model_dir}")
        return None

    if len(models) == 1:
        print(f"[WARN] Only 1 model found — using single model (no ensemble benefit)")
        return models[0]

    return ModelEnsemble(models, weights=weights)


def create_multi_seed_training_configs(base_config=None, seeds=None):
    """Generate training configs for multi-seed ensemble training.

    Call this to get a list of seed values for training multiple models.
    Each should be trained independently with a different SEED env var.

    Args:
        base_config: dict of base config overrides
        seeds: list of int seeds (default: [42, 123, 456])

    Returns:
        list of dicts, each containing env var overrides for one training run
    """
    if seeds is None:
        seeds = [42, 123, 456]

    configs = []
    for seed in seeds:
        config = {
            "SEED": str(seed),
            "SKIP_TRAIN_IF_EXISTS": "0",  # Force retrain each seed
        }
        if base_config:
            config.update(base_config)
        configs.append(config)

    print(f"[ENSEMBLE] Generated {len(configs)} training configs for seeds: {seeds}")
    print("  To train: set os.environ with each config and run Cell 9")
    return configs


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL ENSEMBLE MODULE")
    print("=" * 60)
    print("Functions available:")
    print("  - ModelEnsemble(models, weights=None)")
    print("  - load_ensemble_model(model_dir=None)")
    print("  - create_multi_seed_training_configs(seeds=[42, 123, 456])")
    print("[OK] Ensemble module ready")
