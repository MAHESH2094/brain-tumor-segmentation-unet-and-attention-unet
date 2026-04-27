# ===================================================
# CONFIGURATION: Centralized Settings
# ===================================================
# Purpose: Single source of truth for all configuration parameters
#          Eliminates hardcoded magic numbers across 13 cells
#
# USAGE:
#   from config import Config
#   width = Config.IMG_SIZE
#   loss_fn = Config.CE_WEIGHT
#
# OVERRIDE via environment variables:
#   export BRATS_IMG_SIZE=512
#   export BRATS_BATCH_SIZE=16
#   python cell_09_training.py

import os
from typing import List, Dict, Any


def _env_with_fallback(brats_key: str, default: str) -> str:
    """Read BRATS_* key with fallback to its unprefixed legacy equivalent."""
    legacy_key = brats_key.replace('BRATS_', '', 1) if brats_key.startswith('BRATS_') else brats_key
    return os.environ.get(brats_key, os.environ.get(legacy_key, default))


class Config:
    """Centralized configuration for BraTS pipeline."""
    
    # ========================
    # DATASET & PREPROCESSING
    # ========================
    IMG_SIZE: int = int(_env_with_fallback('BRATS_IMG_SIZE', '128'))
    NUM_CHANNELS: int = int(_env_with_fallback('BRATS_NUM_CHANNELS', '4'))
    NUM_CLASSES: int = int(_env_with_fallback('BRATS_NUM_CLASSES', '1'))
    
    MODALITIES: List[str] = ['flair', 't1', 't1ce', 't2']
    CLASS_NAMES: List[str] = ['Tumor']
    
    # Preprocessing parameters
    MIN_TUMOR_PIXELS: int = int(_env_with_fallback('BRATS_MIN_TUMOR_PIXELS', '50'))
    BACKGROUND_KEEP_RATIO: float = float(_env_with_fallback('BRATS_BACKGROUND_KEEP_RATIO', '0.10'))
    
    # CLAHE enhancement
    CLAHE_CLIP_LIMIT: float = float(_env_with_fallback('BRATS_CLAHE_CLIP_LIMIT', '2.0'))
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    
    # Normalization
    NORMALIZE_EPSILON: float = float(_env_with_fallback('BRATS_NORMALIZE_EPSILON', '1e-6'))
    
    # ========================
    # HDF5 DATASET
    # ========================
    HDF5_COMPRESSION: str = _env_with_fallback('BRATS_HDF5_COMPRESSION', 'gzip')
    HDF5_COMPRESSION_LEVEL: int = int(_env_with_fallback('BRATS_HDF5_COMPRESSION_LEVEL', '1'))
    HDF5_WRITE_BATCH_SIZE: int = int(_env_with_fallback('BRATS_HDF5_WRITE_BATCH_SIZE', '256'))
    
    # Dataset splits
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # Max samples (set to None to use all data)
    MAX_TRAIN_SAMPLES: int = int(_env_with_fallback('BRATS_MAX_TRAIN_SAMPLES', '16000'))
    MAX_VAL_SAMPLES: int = int(_env_with_fallback('BRATS_MAX_VAL_SAMPLES', '4000'))
    MAX_TEST_SAMPLES: int = int(_env_with_fallback('BRATS_MAX_TEST_SAMPLES', '2000'))
    
    # ========================
    # AUGMENTATION
    # ========================
    AUGMENTATION_ENABLED: bool = _env_with_fallback('BRATS_AUGMENTATION_ENABLED', '1') == '1'
    ROTATION_RANGE: float = float(_env_with_fallback('BRATS_ROTATION_RANGE', '15'))
    FLIP_HORIZONTAL: bool = _env_with_fallback('BRATS_FLIP_HORIZONTAL', '1') == '1'
    FLIP_VERTICAL: bool = _env_with_fallback('BRATS_FLIP_VERTICAL', '0') == '1'
    INTENSITY_SCALE_RANGE: tuple = (0.85, 1.15)
    AUG_PROBABILITY: float = float(_env_with_fallback('BRATS_AUG_PROBABILITY', '0.5'))
    
    # ========================
    # MODEL ARCHITECTURE
    # ========================
    INPUT_SHAPE: tuple = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    OUTPUT_SHAPE: tuple = (IMG_SIZE, IMG_SIZE, NUM_CLASSES)
    
    FILTERS: List[int] = [64, 128, 256, 512]
    BOTTLENECK_FILTERS: int = int(_env_with_fallback('BRATS_BOTTLENECK_FILTERS', '512'))
    KERNEL_SIZE: tuple = (3, 3)
    ACTIVATION: str = _env_with_fallback('BRATS_ACTIVATION', 'relu')
    USE_BATCH_NORM: bool = _env_with_fallback('BRATS_USE_BATCH_NORM', '1') == '1'
    DROPOUT_RATE: float = float(_env_with_fallback('BRATS_DROPOUT_RATE', '0.2'))
    
    # ========================
    # LOSS & METRICS
    # ========================
    # Binary class weighting hook (single output channel)
    CLASS_WEIGHTS: List[float] = [1.0]
    
    # Loss weighting
    CE_WEIGHT: float = float(_env_with_fallback('BRATS_BCE_WEIGHT', '0.2'))
    DICE_WEIGHT: float = float(_env_with_fallback('BRATS_DICE_WEIGHT', '0.7'))
    
    # Threshold for binarization
    THRESHOLD: float = float(_env_with_fallback('BRATS_THRESHOLD', '0.5'))
    SMOOTH_FACTOR: float = float(_env_with_fallback('BRATS_SMOOTH_FACTOR', '1e-6'))
    
    # ========================
    # TRAINING
    # ========================
    BATCH_SIZE_PER_GPU: int = int(_env_with_fallback('BRATS_BATCH_SIZE_PER_GPU', '8'))
    LEARNING_RATE: float = float(_env_with_fallback('BRATS_LEARNING_RATE', '4e-4'))
    EPOCHS: int = int(_env_with_fallback('BRATS_EPOCHS', '8'))
    
    # Training steps (fraction of total)
    STEPS_FRACTION: float = float(_env_with_fallback('BRATS_STEPS_FRACTION', '0.18'))
    VAL_STEPS_FRACTION: float = float(_env_with_fallback('BRATS_VAL_STEPS_FRACTION', '0.35'))
    
    # Callbacks
    PATIENCE_EARLY_STOPPING: int = int(_env_with_fallback('BRATS_PATIENCE_ES', '4'))
    PATIENCE_LR_REDUCE: int = int(_env_with_fallback('BRATS_PATIENCE_LR', '2'))
    LR_REDUCE_FACTOR: float = float(_env_with_fallback('BRATS_LR_FACTOR', '0.5'))
    MIN_LR: float = float(_env_with_fallback('BRATS_MIN_LR', '1e-6'))
    
    # Resume training
    SKIP_TRAIN_IF_EXISTS: bool = _env_with_fallback('BRATS_SKIP_TRAIN_IF_EXISTS', '1') == '1'
    
    # ========================
    # INFERENCE
    # ========================
    INFERENCE_THRESHOLD: float = float(_env_with_fallback('BRATS_INFERENCE_THRESHOLD', '0.5'))
    MIN_TUMOR_SIZE: int = int(_env_with_fallback('BRATS_MIN_TUMOR_SIZE', '50'))
    ENFORCE_HIERARCHY: bool = _env_with_fallback('BRATS_ENFORCE_HIERARCHY', '0') == '1'
    
    # ========================
    # I/O & PATHS
    # ========================
    SEED: int = int(_env_with_fallback('BRATS_SEED', '42'))
    OUTPUT_DIR: str = os.environ.get('OUTPUT_DIR', os.environ.get('BRATS_OUTPUT_DIR', './outputs'))
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR: str = os.path.join(OUTPUT_DIR, 'logs')
    RESULTS_DIR: str = os.path.join(OUTPUT_DIR, 'results')
    HDF5_PATH: str = os.path.join(OUTPUT_DIR, 'brats_preprocessed.h5')
    
    # ========================
    # MIXED PRECISION & OPTIMIZATION
    # ========================
    USE_MIXED_PRECISION: bool = _env_with_fallback('BRATS_MIXED_PRECISION', '1') == '1'
    USE_XLA_JIT: bool = _env_with_fallback('BRATS_USE_XLA', '0') == '1'
    ENABLE_TF32: bool = _env_with_fallback('BRATS_ENABLE_TF32', '1') == '1'
    
    # ========================
    # DEVELOPMENT MODE
    # ========================
    FAST_DEV_MODE: bool = os.environ.get('FAST_DEV_MODE', '0') == '1'
    VERBOSE: int = int(_env_with_fallback('BRATS_VERBOSE', '1'))
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in cls.__dict__.items()
            if (
                not k.startswith('_')
                and not callable(v)
                and not isinstance(v, (classmethod, staticmethod, property))
            )
        }
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BRATS PIPELINE CONFIGURATION")
        print("=" * 80)
        
        sections = {
            'DATASET': ['IMG_SIZE', 'NUM_CHANNELS', 'NUM_CLASSES', 'MIN_TUMOR_PIXELS'],
            'ARCHITECTURE': ['FILTERS', 'BOTTLENECK_FILTERS', 'DROPOUT_RATE'],
            'LOSS & METRICS': ['CLASS_WEIGHTS', 'CE_WEIGHT', 'DICE_WEIGHT', 'THRESHOLD'],
            'TRAINING': ['BATCH_SIZE_PER_GPU', 'LEARNING_RATE', 'EPOCHS'],
            'INFERENCE': ['INFERENCE_THRESHOLD', 'ENFORCE_HIERARCHY'],
            'DEVICE': ['USE_MIXED_PRECISION', 'ENABLE_TF32', 'FAST_DEV_MODE'],
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            print("-" * 80)
            for key in keys:
                if hasattr(cls, key):
                    value = getattr(cls, key)
                    print(f"  {key:<40} = {value}")
        
        print("\n" + "=" * 80)


# ========================
# BACKWARD COMPATIBILITY
# ========================
# Provide module-level constants for existing code
IMG_SIZE = Config.IMG_SIZE
NUM_CHANNELS = Config.NUM_CHANNELS
NUM_CLASSES = Config.NUM_CLASSES
CLASS_NAMES = Config.CLASS_NAMES
MODALITIES = Config.MODALITIES
THRESHOLD = Config.THRESHOLD
SMOOTH = Config.SMOOTH_FACTOR
CE_WEIGHT = Config.CE_WEIGHT
DICE_WEIGHT = Config.DICE_WEIGHT
CLASS_WEIGHTS = Config.CLASS_WEIGHTS
BATCH_SIZE_PER_GPU = Config.BATCH_SIZE_PER_GPU
LEARNING_RATE = Config.LEARNING_RATE
EPOCHS = Config.EPOCHS
OUTPUT_DIR = Config.OUTPUT_DIR
MODEL_DIR = Config.MODEL_DIR
LOG_DIR = Config.LOG_DIR
RESULTS_DIR = Config.RESULTS_DIR
HDF5_PATH = Config.HDF5_PATH


if __name__ == '__main__':
    Config.print_summary()
