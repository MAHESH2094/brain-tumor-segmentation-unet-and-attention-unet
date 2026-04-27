import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Centralized configuration for BraTS segmentation pipeline."""
    # Dataset & Preprocessing
    seed: int = 42
    img_size: int = int(os.environ.get("BRATS_IMG_SIZE", "128"))
    num_channels: int = 4
    num_classes: int = 1
    min_tumor_pixels: int = 50
    threshold: float = 0.5
    
    # Training
    batch_size_per_gpu: int = 16
    learning_rate: float = 4e-4
    epochs: int = 25
    steps_fraction: float = 1.0
    val_steps_fraction: float = 1.0
    
    # Loss & Metrics
    ce_weight: float = 0.4
    dice_weight: float = 0.6
    hierarchy_weight: float = 0.0
    positive_class_weight: float = float(os.environ.get("BRATS_POSITIVE_CLASS_WEIGHT", "1.0"))
    
    # Callbacks
    patience_early_stopping: int = 5
    patience_lr_reduce: int = 2
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Inference
    enforce_hierarchy: bool = False
    thresholds_path: str = os.environ.get("BRATS_THRESHOLDS_PATH", "")


def get_output_dirs():
    """Get or create standard output directories."""
    default_output = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    output_dir = os.environ.get("OUTPUT_DIR", default_output)
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    results_dir = os.path.join(output_dir, "results")
    for d in (model_dir, log_dir, results_dir):
        os.makedirs(d, exist_ok=True)
    return output_dir, model_dir, log_dir, results_dir


def get_thresholds_path(results_dir=None):
    """Return the tuned thresholds file path with env override support."""
    explicit_path = os.environ.get("BRATS_THRESHOLDS_PATH", "").strip()
    if explicit_path:
        return explicit_path

    filename = os.environ.get("BRATS_THRESHOLDS_FILENAME", "optimal_thresholds.json").strip()
    if not filename:
        filename = "optimal_thresholds.json"

    if results_dir is None:
        _, _, _, results_dir = get_output_dirs()
    return os.path.join(results_dir, filename)

