# ===================================================
# CELL 1: Environment Setup & Reproducibility (FIXED)
# ===================================================
# Purpose: Import libraries, set random seeds, verify GPU
# FIXES: TF 2.9+ API guard, standardized to f-strings, explicit seed handling

import os
import random
import numpy as np
import tensorflow as tf
import nibabel as nib
import h5py
import cv2
import json
import time
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ========================
# BINARY CONFIGURATION
# ========================
NUM_CLASSES = 1    # Tumor vs background
NUM_CHANNELS = 4   # flair, t1, t1ce, t2
CLASS_NAMES = ['Tumor']
MODALITIES = ['flair', 't1', 't1ce', 't2']

# BraTS-2024 preprocessed patches are 128x128x128; keep 2D training at 128.
os.environ.setdefault('BRATS_IMG_SIZE', '128')
# Dataset card channel order is usually [T1, T1ce, T2, FLAIR].
os.environ.setdefault('BRATS_NPZ_CHANNEL_ORDER', 't1,t1ce,t2,flair')

# ========================
# PERFORMANCE FLAGS (SPEED OPTIMIZED)
# ========================
# XLA DISABLED: Causes MaxPool gradient fallback on T4, actually slower
tf.config.optimizer.set_jit(False)

# TF32 ENABLED: Use Tensor Float 32 for ~10% speed boost (T4 supports this)
# FIX: Guard for TF 2.4+ where this API became stable
tf_version = tuple(int(x) for x in tf.__version__.split('.')[:2])
if tf_version >= (2, 4):
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except (AttributeError, RuntimeError) as e:
        print(f"⚠ TF32 execution not available on this TensorFlow version: {e}")
else:
    # Safe for older TF versions
    tf.config.experimental.enable_tensor_float_32_execution(True)

# Reduce logging overhead
tf.get_logger().setLevel('ERROR')

# Clear any existing models
tf.keras.backend.clear_session()

# ========================
# REPRODUCIBILITY SETUP (FIX: Use explicit RNG where possible)
# ========================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# FIX: Prefer np.random.default_rng() for explicit seed control in new code
# Keep global seed set for backward compatibility with existing code
GLOBAL_RNG = np.random.default_rng(SEED)

# Deterministic ops DISABLED for maximum speed (uncomment for reproducibility)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ========================
# VERSION LOGGING
# ========================
print("=" * 50)
print("ENVIRONMENT CONFIGURATION")
print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Random Seed: {SEED}")
print(f"TensorFlow version tuple: {tf_version}")
print(f"Configured BRATS_IMG_SIZE: {os.environ.get('BRATS_IMG_SIZE')}")

# ========================
# GPU CONFIGURATION
# ========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {len(gpus)} GPU(s) detected")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    # Enable memory growth to avoid OOM.
    # In notebook reruns, runtime may already be initialized, so this must be best-effort.
    memory_growth_enabled = 0
    runtime_already_initialized = False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            memory_growth_enabled += 1
        except RuntimeError as exc:
            runtime_already_initialized = True
            print(f"  [INFO] Could not set memory growth for {gpu.name}: {exc}")
        except Exception as exc:
            print(f"  [WARN] Unexpected memory growth error for {gpu.name}: {exc}")

    if memory_growth_enabled:
        print(f"Memory growth enabled on {memory_growth_enabled}/{len(gpus)} GPU(s)")
    elif runtime_already_initialized:
        print(
            "Memory growth unchanged because TensorFlow runtime is already initialized. "
            "Restart the kernel and run Cell 1 first to apply memory growth."
        )
else:
    print("GPU Available: No GPU detected (using CPU)")

print("=" * 50)
print("Environment setup complete!")
print("=" * 50)

# ========================
# SMOKE TEST (FIX: Add minimal validation)
# ========================
def test_environment():
    """Minimal smoke test to verify environment is setup correctly."""
    # Test numpy
    arr = GLOBAL_RNG.random((10, 10))
    assert arr.shape == (10, 10), "NumPy RNG failed"
    
    # Test TensorFlow
    t = tf.random.normal((10, 10))
    assert t.shape == (10, 10), "TensorFlow random failed"
    
    # Test nibabel
    dummy_data = np.random.rand(50, 50, 50)
    dummy_img = nib.Nifti1Image(dummy_data, np.eye(4))
    assert dummy_img.get_fdata().shape == (50, 50, 50), "NiBabel failed"
    
    print("✓ All environment smoke tests passed")

if __name__ == '__main__':
    test_environment()
