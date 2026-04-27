# ===================================================
# BRIDGE MODULE: Unified Interface
# ===================================================

from ml_pipeline.config import PipelineConfig, get_output_dirs
from ml_pipeline.losses import (
    combined_loss,
    dice_coef,
    precision_metric,
    sensitivity_metric,
    specificity_metric,
    iou_metric,
    CUSTOM_OBJECTS,
)
from ml_pipeline.models import build_unet
try:
    from cell_07d_attention_unet_vit_FIXED import build_attention_unet_vit as _build_attention_unet_vit
except Exception:
    _build_attention_unet_vit = None
from ml_pipeline.train import run_dual_training
from ml_pipeline.infer import (
    load_best_model,
    preprocess_patient,
    postprocess,
    reconstruct_3d,
    save_nifti,
)


# Keep name for backward compatibility with existing imports.
def build_attention_unet(*args, **kwargs):
    kwargs["attention"] = True
    return build_unet(*args, **kwargs)


def build_attention_unet_vit(*args, **kwargs):
    if _build_attention_unet_vit is None:
        raise RuntimeError(
            "AttentionUNetViT builder is unavailable. Ensure "
            "cell_07d_attention_unet_vit_FIXED.py exists and imports cleanly."
        )
    return _build_attention_unet_vit(*args, **kwargs)


__all__ = [
    "PipelineConfig",
    "get_output_dirs",
    "combined_loss",
    "dice_coef",
    "precision_metric",
    "sensitivity_metric",
    "specificity_metric",
    "iou_metric",
    "CUSTOM_OBJECTS",
    "build_unet",
    "build_attention_unet",
    "build_attention_unet_vit",
    "run_dual_training",
    "load_best_model",
    "preprocess_patient",
    "postprocess",
    "reconstruct_3d",
    "save_nifti",
]


def print_module_info():
    print("=" * 70)
    print("UNIFIED BRATS PIPELINE INTERFACE (BINARY)")
    print("=" * 70)
    print("Primary loss/metric exports: combined_loss, dice_coef")
    print("Models: build_unet(attention=False/True), build_attention_unet_vit(...)")
    print("Training: run_dual_training()")
    print("Inference: load_best_model(), preprocess_patient(), postprocess()")
    print("=" * 70)


if __name__ == '__main__':
    print_module_info()
