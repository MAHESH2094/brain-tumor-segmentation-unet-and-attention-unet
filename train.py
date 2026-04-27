# ===================================================
# CELL 9: TRAINING - DUAL MODEL (PRODUCTION FIXED)
# ===================================================

import gc
import importlib.util
import json
import math
import os
import random
import threading
import time
import traceback
import io
from contextlib import suppress
from contextlib import redirect_stdout, redirect_stderr

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TerminateOnNaN,
)
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.optimizers import Adam


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        value = os.path.abspath(str(item))
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _candidate_project_roots():
    """Collect likely project roots for notebook/script/Kaggle execution."""
    roots = []

    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))

    env_root = os.environ.get("BRATS_PROJECT_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.extend([os.getcwd(), "/kaggle/working", "/kaggle/input", "/kaggle/input/datasets"])

    for base in ["/kaggle/working", "/kaggle/input", "/kaggle/input/datasets", os.getcwd()]:
        if not os.path.isdir(base):
            continue
        try:
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
        except PermissionError:
            continue

    return _dedupe_keep_order(roots)


def _bounded_find_filename(base_dir, filename, max_depth=5, max_hits=16):
    """Find file under base_dir using bounded recursive walk."""
    if not base_dir or not os.path.isdir(base_dir):
        return []

    hits = []
    base_norm = str(base_dir).replace("\\", "/").rstrip("/")
    base_depth = base_norm.count("/")

    for root, dirs, files in os.walk(base_dir):
        root_norm = root.replace("\\", "/").rstrip("/")
        depth = root_norm.count("/") - base_depth
        if depth > int(max_depth):
            dirs[:] = []
            continue

        if filename in files:
            hits.append(os.path.join(root, filename))
            if len(hits) >= int(max_hits):
                break

    return hits


def _resolve_hdf5_path(explicit_hdf5_path, output_dir):
    """Resolve brats_preprocessed.h5 from env/project/Kaggle candidate paths."""
    filename = "brats_preprocessed.h5"

    candidates = [
        explicit_hdf5_path,
        os.environ.get("HDF5_PATH", "").strip(),
        os.environ.get("BRATS_HDF5_PATH", "").strip(),
        os.path.join(output_dir, filename) if output_dir else None,
        os.path.join(os.getcwd(), filename),
        os.path.join("/kaggle/working", filename),
        os.path.join("/kaggle/working", "outputs", filename),
        os.path.join("/kaggle/working", "cap", filename),
    ]

    project_root = os.environ.get("BRATS_PROJECT_ROOT", "").strip()
    if project_root:
        candidates.append(os.path.join(project_root, filename))
        candidates.append(os.path.join(project_root, "outputs", filename))

    for root in _candidate_project_roots():
        candidates.append(os.path.join(root, filename))
        candidates.append(os.path.join(root, "outputs", filename))

    checked = _dedupe_keep_order(candidates)
    for path in checked:
        if os.path.isfile(path):
            return path, checked

    search_roots = _dedupe_keep_order(
        [
            project_root,
            os.getcwd(),
            "/kaggle/working",
        ]
    )
    discovered = []
    for base in search_roots:
        discovered.extend(_bounded_find_filename(base, filename, max_depth=5, max_hits=16))
        if discovered:
            break

    discovered = _dedupe_keep_order(discovered)
    if discovered:
        checked = _dedupe_keep_order(checked + discovered)
        return discovered[0], checked

    fallback = checked[0] if checked else os.path.join(output_dir or os.getcwd(), filename)
    return fallback, checked


# ========================
# REPRODUCIBILITY
# ========================
SEED = int(os.environ.get("SEED", "42"))
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
ENABLE_DETERMINISTIC_OPS = os.environ.get("BRATS_DETERMINISTIC_OPS", "0") == "1"
if ENABLE_DETERMINISTIC_OPS:
    with suppress(Exception):
        tf.config.experimental.enable_op_determinism()


# ========================
# PATHS
# ========================
IS_KAGGLE = os.path.isdir("/kaggle/working")
DEFAULT_OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else os.getcwd()
OUTPUT_DIR = globals().get(
    "OUTPUT_DIR",
    os.environ.get("OUTPUT_DIR", os.environ.get("BRATS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
)
MODEL_DIR = globals().get("MODEL_DIR", os.path.join(OUTPUT_DIR, "models"))
LOG_DIR = globals().get("LOG_DIR", os.path.join(OUTPUT_DIR, "logs"))
RESULTS_DIR = globals().get("RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))
HDF5_PATH, HDF5_PATH_CHECKED = _resolve_hdf5_path(
    globals().get("HDF5_PATH", os.environ.get("HDF5_PATH", "")),
    OUTPUT_DIR,
)
if os.path.isfile(HDF5_PATH):
    os.environ["HDF5_PATH"] = HDF5_PATH

for directory in (MODEL_DIR, LOG_DIR, RESULTS_DIR):
    os.makedirs(directory, exist_ok=True)

PROJECT_POSITIONING = (
    "This project focuses on building a complete deep learning pipeline for brain tumor "
    "segmentation, emphasizing efficiency, robustness, and practical deployment rather "
    "than proposing a new architecture."
)


# ========================
# BINARY LOSS & METRIC
# ========================
def soft_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1 - tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
TVERSKY_ALPHA = float(os.environ.get("BRATS_TVERSKY_ALPHA", "0.3"))
TVERSKY_BETA = float(os.environ.get("BRATS_TVERSKY_BETA", "0.7"))
COMBINED_BCE_WEIGHT = float(
    os.environ.get("BRATS_BCE_WEIGHT", os.environ.get("BRATS_CE_WEIGHT", "0.2"))
)


def tversky_loss(y_true, y_pred, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=[1, 2, 3])

    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(score)


def combined_loss(y_true, y_pred):
    return tversky_loss(y_true, y_pred) + COMBINED_BCE_WEIGHT * bce(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


# ========================
# REQUIRED SYMBOLS
# ========================
REQUIRED_SYMBOLS = (
    "build_unet",
    "build_attention_unet",
    "combined_loss",
    "dice_coef",
    "precision_metric",
    "sensitivity_metric",
    "iou_metric",
)

_TRAIN_METRIC_SYMBOLS = ()

# Optional builders (resolved dynamically for notebook/Kaggle compatibility).
build_unet = globals().get("build_unet", None)
build_attention_unet = globals().get("build_attention_unet", None)
build_attention_unet_vit = globals().get("build_attention_unet_vit", None)
get_attention_unet_vit_custom_objects = globals().get("get_attention_unet_vit_custom_objects", None)


def _import_cell_symbol(module_filename, symbol_name):
    module_name = module_filename.replace(".py", "")

    with suppress(Exception):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module = __import__(module_name, fromlist=[symbol_name])
            return getattr(module, symbol_name)

    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend([os.getcwd(), "/kaggle/working"])

    for base in candidates:
        module_path = os.path.join(base, module_filename)
        if not os.path.exists(module_path):
            continue
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            continue
        with suppress(Exception):
            module = importlib.util.module_from_spec(spec)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                spec.loader.exec_module(module)
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name)
    return None


def _resolve_module_file(module_filename):
    """Resolve a module filename across project/Kaggle roots."""
    if os.path.isabs(module_filename) and os.path.exists(module_filename):
        return module_filename

    for base in _candidate_project_roots():
        candidate = os.path.join(base, module_filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _directory_has_npz_patches(base_dir, max_depth=6):
    """Return True when a directory contains NPZ files (patch-preferred)."""
    if not base_dir or not os.path.isdir(base_dir):
        return False

    base_norm = str(base_dir).replace("\\", "/").rstrip("/")
    base_depth = base_norm.count("/")
    found_any_npz = False

    for root, dirs, files in os.walk(base_dir):
        root_norm = root.replace("\\", "/").rstrip("/")
        depth = root_norm.count("/") - base_depth
        if depth > int(max_depth):
            dirs[:] = []
            continue

        has_patch_npz = any(name.lower().endswith(".npz") and "patch" in name.lower() for name in files)
        if has_patch_npz:
            return True

        if not found_any_npz and any(name.lower().endswith(".npz") for name in files):
            found_any_npz = True

    return found_any_npz


def _discover_npz_patch_root():
    """Find likely NPZ patch dataset root for auto HDF5 build."""
    candidates = [
        os.environ.get("BRATS_NPZ_PATCH_PATH", "").strip(),
        os.environ.get("NPZ_PATCH_PATH", "").strip(),
        globals().get("NPZ_PATCH_PATH", None),
        "/kaggle/input/datasets/prathamhanda10/brats-2024-preprocessed-training-patches",
        "/kaggle/input/brats-2024-preprocessed-training-patches",
    ]

    for root in _candidate_project_roots():
        candidates.append(root)

    for candidate in _dedupe_keep_order(candidates):
        if _directory_has_npz_patches(candidate):
            return candidate
    return None


def _attempt_autobuild_hdf5_from_npz():
    """Attempt to build missing HDF5 automatically from NPZ patches via Cell 5."""
    global HDF5_PATH, HDF5_PATH_CHECKED

    if os.path.isfile(HDF5_PATH):
        return True

    npz_root = _discover_npz_patch_root()
    if not npz_root:
        print("[INFO] HDF5 auto-build skipped: NPZ patch dataset was not found.")
        return False

    print(f"[INFO] HDF5 missing. Attempting auto-build from NPZ patches: {npz_root}")

    target_hdf5 = os.path.join(OUTPUT_DIR, "brats_preprocessed.h5")
    os.environ["BRATS_DATASET_MODE"] = "npz_patches"
    os.environ["BRATS_NPZ_PATCH_PATH"] = npz_root
    os.environ["OUTPUT_DIR"] = OUTPUT_DIR
    os.environ["HDF5_PATH"] = target_hdf5

    cell5_path = _resolve_module_file("cell_05_hdf5_builder_FIXED.py")
    if not cell5_path:
        print("[WARN] Could not locate cell_05_hdf5_builder_FIXED.py for auto-build.")
        return False

    init_globals = {
        "OUTPUT_DIR": OUTPUT_DIR,
        "HDF5_PATH": target_hdf5,
        "TRAIN_PATH": globals().get("TRAIN_PATH", None),
        "NPZ_PATCH_PATH": npz_root,
        "SEED": SEED,
        "NUM_CHANNELS": int(os.environ.get("BRATS_NUM_CHANNELS", "4")),
        "NUM_CLASSES": int(os.environ.get("BRATS_NUM_CLASSES", "1")),
        "MASK_CHANNELS": 1,
    }

    for sym in ("create_binary_mask", "preprocess_multimodal_slice", "load_multimodal_volume"):
        resolved = globals().get(sym)
        if resolved is None:
            resolved = _import_cell_symbol("cell_04_preprocessing_FIXED.py", sym)
        if callable(resolved):
            init_globals[sym] = resolved

    try:
        import runpy

        runpy.run_path(cell5_path, init_globals=init_globals, run_name="cell_05_autobuild")
    except Exception as exc:
        print(f"[WARN] Auto-build via Cell 5 failed: {exc}")
        return False

    HDF5_PATH, HDF5_PATH_CHECKED = _resolve_hdf5_path(os.environ.get("HDF5_PATH", target_hdf5), OUTPUT_DIR)
    if os.path.isfile(HDF5_PATH):
        os.environ["HDF5_PATH"] = HDF5_PATH
        print(f"[OK] Auto-built HDF5: {HDF5_PATH}")
        return True

    print("[WARN] Auto-build completed but HDF5 was still not found.")
    return False

missing_symbols = [name for name in REQUIRED_SYMBOLS if name not in globals()]
if missing_symbols:
    if build_unet is None:
        _tmp = _import_cell_symbol("cell_07b_unet_FIXED.py", "build_unet")
        if _tmp is not None:
            globals()["build_unet"] = _tmp
            build_unet = _tmp

    if build_attention_unet is None:
        _tmp = _import_cell_symbol("cell_07c_attention_unet_FIXED.py", "build_attention_unet")
        if _tmp is not None:
            globals()["build_attention_unet"] = _tmp
            build_attention_unet = _tmp

    if build_attention_unet_vit is None:
        _tmp = _import_cell_symbol("cell_07d_attention_unet_vit_FIXED.py", "build_attention_unet_vit")
        if _tmp is not None:
            globals()["build_attention_unet_vit"] = _tmp
            build_attention_unet_vit = _tmp

    if get_attention_unet_vit_custom_objects is None:
        _tmp = _import_cell_symbol(
            "cell_07d_attention_unet_vit_FIXED.py", "get_attention_unet_vit_custom_objects"
        )
        if _tmp is not None:
            globals()["get_attention_unet_vit_custom_objects"] = _tmp
            get_attention_unet_vit_custom_objects = _tmp

    with suppress(Exception):
        from custom_objects_registry import get_custom_objects

        _registry = get_custom_objects()
        for _sym in (
            "combined_loss",
            "dice_coef",
            "precision_metric",
            "sensitivity_metric",
            "iou_metric",
            *_TRAIN_METRIC_SYMBOLS,
        ):
            if _sym in _registry and _sym not in globals():
                globals()[_sym] = _registry[_sym]
    with suppress(Exception):
        from cell_08_loss_metrics_FIXED import (
            combined_loss,
            dice_coef,
            precision_metric,
            sensitivity_metric,
            iou_metric,
        )

missing_symbols = [name for name in REQUIRED_SYMBOLS if name not in globals()]
if missing_symbols:
    raise RuntimeError(f"Run Cell 7 first. Missing symbols: {', '.join(missing_symbols)}")

CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "tversky_loss": tversky_loss,
    "dice_coef": dice_coef,
    "precision_metric": precision_metric,
    "sensitivity_metric": sensitivity_metric,
    "iou_metric": iou_metric,
}

# Add _hierarchy_sort so models with hierarchy constraint layer can be reloaded
try:
    from cell_07a_building_blocks_FIXED import _hierarchy_sort, hierarchy_constraint_layer
    CUSTOM_OBJECTS["_hierarchy_sort"] = _hierarchy_sort
    CUSTOM_OBJECTS["hierarchy_constraint_layer"] = hierarchy_constraint_layer
except ImportError:
    pass

try:
    if callable(get_attention_unet_vit_custom_objects):
        CUSTOM_OBJECTS.update(get_attention_unet_vit_custom_objects())
except Exception:
    pass


# ========================
# HARDWARE
# ========================
def setup_hardware():
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)
    active_gpus = 0
    strategy_pref = os.environ.get("BRATS_STRATEGY", "auto").strip().lower()

    if strategy_pref not in {"auto", "single", "mirrored", "cpu"}:
        raise ValueError(
            f"Invalid BRATS_STRATEGY='{strategy_pref}'. "
            "Use one of: auto, single, mirrored, cpu."
        )

    print("=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)

    if num_gpus == 0:
        strategy = tf.distribute.get_strategy()
        print("No GPU detected. Training on CPU.")
    elif strategy_pref == "cpu":
        cpu_forced = False
        with suppress(Exception):
            tf.config.set_visible_devices([], "GPU")
            cpu_forced = True

        strategy = tf.distribute.get_strategy()
        if cpu_forced:
            print("BRATS_STRATEGY=cpu -> GPU devices hidden. Training on CPU.")
        else:
            print("[WARN] BRATS_STRATEGY=cpu requested, but GPU visibility could not be changed.")
            print("[WARN] Continuing with default strategy; GPU may still be used.")
    elif num_gpus == 1 or strategy_pref == "single":
        with suppress(Exception):
            tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.get_strategy()
        active_gpus = 1
        if num_gpus == 1:
            print(f"1 GPU detected: {gpus[0].name}")
        else:
            print(f"{num_gpus} GPUs detected. BRATS_STRATEGY=single -> using /GPU:0.")
    else:
        for gpu in gpus:
            with suppress(Exception):
                tf.config.experimental.set_memory_growth(gpu, True)
        # Force mirrored multi-GPU for 2+ GPU setups.
        strategy = tf.distribute.MirroredStrategy()
        active_gpus = num_gpus
        if strategy_pref == "mirrored":
            print(f"{num_gpus} GPUs detected. BRATS_STRATEGY=mirrored -> using all GPUs.")
        else:
            print(f"{num_gpus} GPUs detected -> MirroredStrategy across all {num_gpus} GPUs.")

    print(f"Replicas in sync: {strategy.num_replicas_in_sync}")
    print(f"Detected GPUs: {num_gpus} | Active GPUs for training: {active_gpus}")
    print("=" * 60)
    return strategy, num_gpus, active_gpus


STRATEGY, NUM_GPUS, ACTIVE_GPUS = setup_hardware()


# ========================
# MIXED PRECISION
# ========================
MIXED_PRECISION_MODE = os.environ.get("BRATS_MIXED_PRECISION", "auto").strip().lower()
if MIXED_PRECISION_MODE not in {"auto", "on", "off"}:
    raise ValueError(
        f"Invalid BRATS_MIXED_PRECISION='{MIXED_PRECISION_MODE}'. "
        "Use one of: auto, on, off."
    )

if MIXED_PRECISION_MODE == "on":
    USE_MIXED_PRECISION = ACTIVE_GPUS > 0
    if not USE_MIXED_PRECISION:
        print("[WARN] BRATS_MIXED_PRECISION=on requested without active GPU. Keeping float32.")
elif MIXED_PRECISION_MODE == "off":
    USE_MIXED_PRECISION = False
else:
    USE_MIXED_PRECISION = ACTIVE_GPUS > 0

if USE_MIXED_PRECISION:
    try:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")
    except Exception as exc:
        USE_MIXED_PRECISION = False
        print(f"Mixed precision disabled due to error: {exc}")
else:
    mixed_precision.set_global_policy("float32")
    print("Mixed precision disabled: using float32")


# ========================
# HYPERPARAMETERS
# ========================
DEMO_MODE = os.environ.get("BRATS_DEMO_MODE", "0") == "1"
FAST_DEV_MODE = os.environ.get("FAST_DEV_MODE", "0") == "1"  # FIX: Default OFF for full training
_batch_size_env = os.environ.get("BATCH_SIZE_PER_GPU", "16")
if DEMO_MODE and "BATCH_SIZE_PER_GPU" not in os.environ:
    _batch_size_env = "4"
BATCH_SIZE_PER_GPU = int(_batch_size_env)  # Demo-friendly default keeps memory pressure lower
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, STRATEGY.num_replicas_in_sync)
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "4e-4"))
ASSERT_INPUT_VALUE_RANGE = os.environ.get("BRATS_ASSERT_INPUT_RANGE", "1") == "1"
DEFAULT_TRAINING_MODE = "unet_only" if DEMO_MODE else "both"
TRAINING_MODE = os.environ.get("BRATS_TRAINING_MODE", DEFAULT_TRAINING_MODE).strip().lower()
ENABLE_VIT_CLASSIFIER = os.environ.get("BRATS_ENABLE_VIT_CLASSIFIER", "0") == "1"
VIT_NUM_CLASSES = int(os.environ.get("BRATS_VIT_NUM_CLASSES", "1"))

if TRAINING_MODE not in {"both", "unet_only", "attention_only", "all", "vit_only"}:
    raise ValueError(
        f"Invalid BRATS_TRAINING_MODE='{TRAINING_MODE}'. "
        "Use one of: both, unet_only, attention_only, all, vit_only."
    )

if VIT_NUM_CLASSES not in {1, 2}:
    raise ValueError(
        f"Invalid BRATS_VIT_NUM_CLASSES='{VIT_NUM_CLASSES}'. "
        "Cell 9 classification targets are derived from binary masks, so only 1 or 2 are supported."
    )

AUGMENTATION_MODE = os.environ.get("BRATS_AUGMENTATION_MODE", "auto").strip().lower()
if AUGMENTATION_MODE not in {"auto", "fast", "cell6", "off"}:
    raise ValueError(
        f"Invalid BRATS_AUGMENTATION_MODE='{AUGMENTATION_MODE}'. "
        "Use one of: auto, fast, cell6, off."
    )
if AUGMENTATION_MODE == "auto":
    # Kaggle + Keras 3 often bottlenecks on Python-side per-sample augmentation.
    AUGMENTATION_MODE = "fast" if IS_KAGGLE else "cell6"

TRAIN_SHUFFLE = os.environ.get("BRATS_TRAIN_SHUFFLE", "1") == "1"
MAX_STEPS_PER_EPOCH = int(os.environ.get("BRATS_MAX_STEPS_PER_EPOCH", "0"))
MAX_VAL_STEPS = int(os.environ.get("BRATS_MAX_VAL_STEPS", "0"))
DATA_PIPELINE_MODE = os.environ.get("BRATS_DATA_PIPELINE_MODE", "auto").strip().lower()
RAM_CACHE_MAX_GIB = float(os.environ.get("BRATS_RAM_CACHE_MAX_GIB", "25"))
RAM_LOAD_CHUNK_SAMPLES = int(os.environ.get("BRATS_RAM_LOAD_CHUNK_SAMPLES", "64"))
RAM_CACHE_DTYPE_MODE = os.environ.get("BRATS_RAM_CACHE_DTYPE", "auto").strip().lower()
if RAM_CACHE_DTYPE_MODE == "auto":
    RAM_CACHE_DTYPE_MODE = "float16" if IS_KAGGLE else "float32"
if RAM_CACHE_DTYPE_MODE not in {"float16", "float32"}:
    raise ValueError(
        f"Invalid BRATS_RAM_CACHE_DTYPE='{RAM_CACHE_DTYPE_MODE}'. "
        "Use one of: auto, float16, float32."
    )
RAM_CACHE_NP_DTYPE = np.float16 if RAM_CACHE_DTYPE_MODE == "float16" else np.float32
DEFAULT_RAM_CACHE_SPLITS = "train" if IS_KAGGLE else "train,val"
RAM_CACHE_SPLITS = {
    s.strip().lower()
    for s in os.environ.get("BRATS_RAM_CACHE_SPLITS", DEFAULT_RAM_CACHE_SPLITS).split(",")
    if s.strip()
}
DEFAULT_STREAM_SHUFFLE_BUFFER = "512" if IS_KAGGLE else "2000"
STREAM_SHUFFLE_BUFFER = int(
    os.environ.get("BRATS_STREAM_SHUFFLE_BUFFER", DEFAULT_STREAM_SHUFFLE_BUFFER)
)
DEFAULT_VAL_EVERY_N_EPOCHS = "1" if DEMO_MODE else ("2" if IS_KAGGLE and not FAST_DEV_MODE else "1")
VAL_EVERY_N_EPOCHS = int(
    os.environ.get("BRATS_VAL_EVERY_N_EPOCHS", DEFAULT_VAL_EVERY_N_EPOCHS)
)
THRESHOLD_TUNE_BATCHES = int(
    os.environ.get("BRATS_THRESHOLD_TUNE_BATCHES", "4" if DEMO_MODE else "8")
)
ENABLE_THRESHOLD_TUNING = (
    os.environ.get("BRATS_ENABLE_THRESHOLD_TUNING", "1") == "1"
)
WT_TARGET_UNET = os.environ.get("BRATS_TARGET_WT_UNET", "88-92%")
WT_TARGET_ATTENTION = os.environ.get("BRATS_TARGET_WT_ATTENTION", "90-94%")
BINARY_TARGET_UNET = os.environ.get("BRATS_TARGET_BINARY_UNET", WT_TARGET_UNET)
BINARY_TARGET_ATTENTION = os.environ.get("BRATS_TARGET_BINARY_ATTENTION", WT_TARGET_ATTENTION)

if FAST_DEV_MODE:
    EPOCHS = int(os.environ.get("EPOCHS", "8"))
    STEPS_FRACTION = float(
        os.environ.get("STEPS_FRACTION", os.environ.get("TRAIN_FRACTION", "0.15"))
    )
    VAL_STEPS_FRACTION = float(
        os.environ.get("VAL_STEPS_FRACTION", os.environ.get("VAL_FRACTION", "0.30"))
    )
else:
    EPOCHS = int(os.environ.get("EPOCHS", "3" if DEMO_MODE else ("18" if IS_KAGGLE else "25")))
    STEPS_FRACTION = float(
        os.environ.get("STEPS_FRACTION", os.environ.get("TRAIN_FRACTION", "0.20" if DEMO_MODE else "1.00"))
    )
    VAL_STEPS_FRACTION = float(
        os.environ.get("VAL_STEPS_FRACTION", os.environ.get("VAL_FRACTION", "0.20" if DEMO_MODE else "1.00"))
    )

PATIENCE_ES = int(os.environ.get("PATIENCE_ES", "5"))
MIN_LR = float(os.environ.get("MIN_LR", "1e-6"))
SKIP_TRAIN_IF_EXISTS = os.environ.get("SKIP_TRAIN_IF_EXISTS", "1") == "1"
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "2"))

if not (0.0 < STEPS_FRACTION <= 1.0):
    raise ValueError("STEPS_FRACTION must be in the range (0, 1].")
if not (0.0 < VAL_STEPS_FRACTION <= 1.0):
    raise ValueError("VAL_STEPS_FRACTION must be in the range (0, 1].")
if DATA_PIPELINE_MODE not in {"auto", "ram", "stream"}:
    raise ValueError(
        f"Invalid BRATS_DATA_PIPELINE_MODE='{DATA_PIPELINE_MODE}'. "
        "Use one of: auto, ram, stream."
    )
if RAM_CACHE_MAX_GIB <= 0.0:
    raise ValueError("BRATS_RAM_CACHE_MAX_GIB must be > 0.")
if RAM_LOAD_CHUNK_SAMPLES <= 0:
    raise ValueError("BRATS_RAM_LOAD_CHUNK_SAMPLES must be > 0.")
if not RAM_CACHE_SPLITS:
    raise ValueError("BRATS_RAM_CACHE_SPLITS must include at least one split name.")
if not RAM_CACHE_SPLITS.issubset({"train", "val", "test"}):
    raise ValueError(
        "Invalid BRATS_RAM_CACHE_SPLITS value. "
        "Use comma-separated split names from: train,val,test."
    )
if STREAM_SHUFFLE_BUFFER <= 0:
    raise ValueError("BRATS_STREAM_SHUFFLE_BUFFER must be > 0.")
if VAL_EVERY_N_EPOCHS <= 0:
    raise ValueError("BRATS_VAL_EVERY_N_EPOCHS must be > 0.")
if ENABLE_THRESHOLD_TUNING and THRESHOLD_TUNE_BATCHES <= 0:
    raise ValueError("BRATS_THRESHOLD_TUNE_BATCHES must be > 0.")

print(f"\nGlobal batch size: {GLOBAL_BATCH_SIZE} | Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS} | FAST_DEV_MODE: {FAST_DEV_MODE}")
print(f"Train steps fraction: {STEPS_FRACTION:.2f} | Val steps fraction: {VAL_STEPS_FRACTION:.2f}")
print(f"Warmup epochs: {WARMUP_EPOCHS} | Patience ES: {PATIENCE_ES}")
print(f"Augmentation mode: {AUGMENTATION_MODE} | Train shuffle: {TRAIN_SHUFFLE}")
print(f"Training mode: {TRAINING_MODE}")
print(f"AttentionUNetViT enabled: {ENABLE_VIT_CLASSIFIER} | classes: {VIT_NUM_CLASSES}")
print(f"Demo mode: {DEMO_MODE}")
if MAX_STEPS_PER_EPOCH > 0:
    print(f"Max train steps/epoch cap: {MAX_STEPS_PER_EPOCH}")
if MAX_VAL_STEPS > 0:
    print(f"Max val steps cap: {MAX_VAL_STEPS}")
print(f"Validation every N epoch(s): {VAL_EVERY_N_EPOCHS}")
print(
    f"Target binary Dice ranges: U-Net {BINARY_TARGET_UNET} | "
    f"Attention U-Net {BINARY_TARGET_ATTENTION}"
)
print(f"Threshold tuning enabled: {ENABLE_THRESHOLD_TUNING}")
if ENABLE_THRESHOLD_TUNING:
    print(f"Threshold tuning batches: {THRESHOLD_TUNE_BATCHES}")
print(f"Data pipeline mode: {DATA_PIPELINE_MODE} | RAM cache cap: {RAM_CACHE_MAX_GIB:.1f} GiB")
print(f"RAM cache dtype: {RAM_CACHE_DTYPE_MODE}")
print(f"RAM cache splits: {sorted(RAM_CACHE_SPLITS)}")
print(f"RAM load chunk size: {RAM_LOAD_CHUNK_SAMPLES} samples")
print(f"Streaming shuffle buffer: {STREAM_SHUFFLE_BUFFER}")
print(f"Deterministic ops: {ENABLE_DETERMINISTIC_OPS}")
if IS_KAGGLE and not (1e-4 <= LEARNING_RATE <= 5e-4):
    print(
        "[WARN] For Kaggle T4 mixed-precision runs, a practical LR range is typically "
        "1e-4 to 5e-4. Current LEARNING_RATE may require extra tuning."
    )
if DEMO_MODE:
    print("[INFO] Demo profile active: workload reduced without changing image size or architecture.")


# ========================
# HDF5 VALIDATION
# ========================
def validate_hdf5(path):
    if not os.path.exists(path):
        checked = globals().get("HDF5_PATH_CHECKED", [])
        if checked:
            preview = "\n".join(f"  - {item}" for item in checked[:16])
        else:
            preview = "  - (no candidate paths recorded)"

        raise FileNotFoundError(
            "HDF5 file not found: "
            f"{path}\n"
            "Checked candidate paths:\n"
            f"{preview}\n"
            "Run Cell 5 first to build brats_preprocessed.h5, or set HDF5_PATH to the exact file path."
        )

    required = [
        "train/images",
        "train/masks",
        "val/images",
        "val/masks",
    ]

    stats = {}
    with h5py.File(path, "r") as h5_file:
        missing = [key for key in required if key not in h5_file]
        if missing:
            raise RuntimeError(f"HDF5 missing required datasets: {missing}")

        for split in ("train", "val"):
            images = h5_file[f"{split}/images"]
            masks = h5_file[f"{split}/masks"]

            if images.ndim != 4 or masks.ndim != 4:
                raise RuntimeError(
                    f"{split} tensors must be rank-4. Got images={images.shape}, masks={masks.shape}."
                )
            if images.shape[0] != masks.shape[0]:
                raise RuntimeError(
                    f"Sample count mismatch for {split}: images={images.shape[0]}, masks={masks.shape[0]}."
                )
            if images.shape[-1] != 4:
                raise RuntimeError(f"Expected 4 image channels for {split}, got {images.shape[-1]}.")
            if masks.shape[-1] != 1:
                raise RuntimeError(f"Expected 1 mask channel for {split}, got {masks.shape[-1]}.")
            if images.shape[0] == 0:
                raise RuntimeError(f"{split} split has zero samples.")

            if split == "val":
                has_positive = False
                scan_chunk = max(1, min(64, int(images.shape[0])))
                for start in range(0, int(images.shape[0]), scan_chunk):
                    end = min(start + scan_chunk, int(images.shape[0]))
                    if np.any(masks[start:end] > 0.5):
                        has_positive = True
                        break
                if not has_positive:
                    raise RuntimeError(
                        "Validation split contains zero tumor-positive pixels. "
                        "Stop: metrics are not meaningful on an all-negative validation set."
                    )

            # Sample-level value checks to fail fast on bad preprocessing.
            sample_n = min(16, int(images.shape[0]))
            image_sample = images[:sample_n].astype(np.float32)
            mask_sample = masks[:sample_n].astype(np.float32)

            if not np.all(np.isfinite(image_sample)):
                raise RuntimeError(f"{split} images contain NaN/Inf values.")
            if not np.all(np.isfinite(mask_sample)):
                raise RuntimeError(f"{split} masks contain NaN/Inf values.")

            if ASSERT_INPUT_VALUE_RANGE:
                img_min = float(np.min(image_sample))
                img_max = float(np.max(image_sample))
                if img_min < -1e-4 or img_max > 1.0001:
                    raise RuntimeError(
                        f"{split} images expected in [0,1], got [{img_min:.4f}, {img_max:.4f}]. "
                        "Regenerate HDF5 with range-normalized images or disable BRATS_ASSERT_INPUT_RANGE."
                    )

            unique_mask_values = np.unique(mask_sample)
            if not np.isin(unique_mask_values, [0.0, 1.0]).all():
                raise RuntimeError(
                    f"{split} masks must be binary 0/1. Found values: {unique_mask_values[:10]}"
                )

            pos_ratio = float(np.mean(mask_sample > 0.5))
            print(f"  [CHECK] {split} sampled tumor-positive ratio: {pos_ratio:.6f}")
            if pos_ratio < 0.005:
                print(
                    "  [WARN] Severe class imbalance detected (<0.5% positives). "
                    "If convergence stalls, consider focal BCE/Tversky or reweighting BCE term."
                )

            stats[split] = {
                "count": int(images.shape[0]),
                "image_shape": tuple(int(v) for v in images.shape[1:]),
                "mask_shape": tuple(int(v) for v in masks.shape[1:]),
            }

    print(f"[OK] HDF5 validated: {path}")
    print(f"Train samples: {stats['train']['count']} | Val samples: {stats['val']['count']}")
    return stats

def _safe_augment_pair(image, mask):
    """Fallback augmentation when Cell 6 callables are unavailable."""
    if tf.random.uniform([]) < 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask


def _resolve_cell6_callable(func_name):
    """Resolve a callable from cell_06_augmentation_FIXED across Kaggle contexts."""
    fn = globals().get(func_name)
    if callable(fn):
        return fn

    with suppress(Exception):
        from cell_06_augmentation_FIXED import augment_pair, set_augmentation_scale

        mapping = {
            "augment_pair": augment_pair,
            "set_augmentation_scale": set_augmentation_scale,
        }
        resolved = mapping.get(func_name)
        if callable(resolved):
            return resolved

    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.extend([os.getcwd(), "/kaggle/working"])

    for base in candidates:
        module_path = os.path.join(base, "cell_06_augmentation_FIXED.py")
        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location("cell_06_augmentation_FIXED", module_path)
        if not spec or not spec.loader:
            continue

        with suppress(Exception):
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            resolved = getattr(module, func_name, None)
            if callable(resolved):
                return resolved

    return None


# ========================
# TF.DATA PIPELINE
# ========================
_RAM_DATA_CACHE = {}
_RAM_CACHE_LOCK = threading.Lock()
_RAM_CACHE_BYTES_USED = 0
_RAM_CACHE_MAX_BYTES = int(RAM_CACHE_MAX_GIB * float(1024 ** 3))


def _bytes_to_gib(num_bytes):
    return float(num_bytes) / float(1024 ** 3)


def _get_split_meta(hdf5_path, split):
    """Return split sample count, tensor shapes, and configured RAM-cache footprint."""
    with h5py.File(hdf5_path, "r") as h5_file:
        images_ds = h5_file[f"{split}/images"]
        masks_ds = h5_file[f"{split}/masks"]
        num_samples = int(images_ds.shape[0])
        image_shape = tuple(int(v) for v in images_ds.shape[1:])
        mask_shape = tuple(int(v) for v in masks_ds.shape[1:])
        cache_bytes = (
            int(np.prod(images_ds.shape, dtype=np.int64)) * np.dtype(RAM_CACHE_NP_DTYPE).itemsize
            + int(np.prod(masks_ds.shape, dtype=np.int64)) * np.dtype(RAM_CACHE_NP_DTYPE).itemsize
        )
    return num_samples, image_shape, mask_shape, cache_bytes


def _iter_split_samples(hdf5_path, split):
    """Yield one sample at a time from chunked HDF5 reads to reduce I/O overhead."""
    with h5py.File(hdf5_path, "r") as h5_file:
        images_ds = h5_file[f"{split}/images"]
        masks_ds = h5_file[f"{split}/masks"]
        num_samples = int(images_ds.shape[0])
        chunk = int(max(1, RAM_LOAD_CHUNK_SAMPLES))
        for start in range(0, num_samples, chunk):
            end = min(start + chunk, num_samples)
            x_chunk = images_ds[start:end].astype(np.float32)
            y_chunk = masks_ds[start:end].astype(np.float32)
            for offset in range(end - start):
                yield x_chunk[offset], y_chunk[offset]


def _iter_ram_samples(images, masks):
    """Yield samples from in-memory arrays without creating duplicated Tensor constants."""
    for idx in range(int(images.shape[0])):
        yield images[idx], masks[idx]


def _load_split_to_ram(hdf5_path, split):
    """Load one split into RAM once and reuse across model runs."""
    global _RAM_CACHE_BYTES_USED

    cache_key = (
        os.path.abspath(hdf5_path),
        split,
        str(np.dtype(RAM_CACHE_NP_DTYPE)),
    )
    with _RAM_CACHE_LOCK:
        cached = _RAM_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if split not in RAM_CACHE_SPLITS:
        print(
            f"[DATA] Split '{split}' excluded from RAM cache via "
            f"BRATS_RAM_CACHE_SPLITS={','.join(sorted(RAM_CACHE_SPLITS))}. "
            "Using streaming tf.data."
        )
        return None

    num_samples, _, _, split_bytes = _get_split_meta(hdf5_path, split)

    if DATA_PIPELINE_MODE == "stream":
        return None

    with _RAM_CACHE_LOCK:
        projected_bytes = _RAM_CACHE_BYTES_USED + split_bytes
        if projected_bytes > _RAM_CACHE_MAX_BYTES:
            if DATA_PIPELINE_MODE == "ram":
                raise MemoryError(
                    f"BRATS_DATA_PIPELINE_MODE=ram but loading split '{split}' would use "
                    f"{_bytes_to_gib(projected_bytes):.2f} GiB, above cap "
                    f"{_bytes_to_gib(_RAM_CACHE_MAX_BYTES):.2f} GiB. "
                    "Increase BRATS_RAM_CACHE_MAX_GIB or use BRATS_DATA_PIPELINE_MODE=auto/stream."
                )
            print(
                f"[DATA] RAM cache cap reached for split '{split}'. "
                f"Projected {_bytes_to_gib(projected_bytes):.2f} GiB > "
                f"cap {_bytes_to_gib(_RAM_CACHE_MAX_BYTES):.2f} GiB. "
                "Falling back to streaming tf.data."
            )
            return None

    images = None
    masks = None
    try:
        with h5py.File(hdf5_path, "r") as h5_file:
            images_ds = h5_file[f"{split}/images"]
            masks_ds = h5_file[f"{split}/masks"]

            # Pre-allocate target arrays and stream-copy in chunks to avoid large transient spikes.
            images = np.empty(images_ds.shape, dtype=RAM_CACHE_NP_DTYPE)
            masks = np.empty(masks_ds.shape, dtype=RAM_CACHE_NP_DTYPE)

            chunk = int(max(1, RAM_LOAD_CHUNK_SAMPLES))
            for start in range(0, num_samples, chunk):
                end = min(start + chunk, num_samples)
                images[start:end] = images_ds[start:end].astype(RAM_CACHE_NP_DTYPE)
                masks[start:end] = masks_ds[start:end].astype(RAM_CACHE_NP_DTYPE)
    except Exception:
        with suppress(Exception):
            del images
        with suppress(Exception):
            del masks
        gc.collect()
        raise

    cache_bytes = int(images.nbytes + masks.nbytes)

    with _RAM_CACHE_LOCK:
        _RAM_DATA_CACHE[cache_key] = (images, masks)
        _RAM_CACHE_BYTES_USED += cache_bytes

    total_gib = _bytes_to_gib(cache_bytes)
    used_gib = _bytes_to_gib(_RAM_CACHE_BYTES_USED)
    print(
        f"[DATA] Loaded {split} split into RAM: {images.shape[0]} samples "
        f"({total_gib:.2f} GiB). Cache used: {used_gib:.2f}/{RAM_CACHE_MAX_GIB:.2f} GiB."
    )
    return images, masks


def clear_ram_data_cache():
    """Release cached RAM datasets when no longer needed."""
    global _RAM_CACHE_BYTES_USED

    with _RAM_CACHE_LOCK:
        _RAM_DATA_CACHE.clear()
        _RAM_CACHE_BYTES_USED = 0

    gc.collect()
    print("[DATA] Cleared in-memory dataset cache.")


def _fast_flip_augment(x, y):
    do_flip = tf.random.uniform([]) < 0.5
    x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(do_flip, lambda: tf.image.flip_left_right(y), lambda: y)
    return x, y


def _mask_batch_to_class_labels_tf(y_batch, num_classes=1):
    """Convert segmentation masks to per-slice binary class labels."""
    has_tumor = tf.reduce_any(y_batch > 0.5, axis=[1, 2, 3])
    if int(num_classes) == 1:
        return tf.cast(tf.expand_dims(has_tumor, axis=-1), tf.float32)
    if int(num_classes) == 2:
        return tf.cast(has_tumor, tf.int32)
    raise ValueError("Only num_classes in {1, 2} are supported for classification labels.")


def make_tf_dataset(
    hdf5_path,
    split,
    batch_size,
    shuffle=True,
    augment=False,
    drop_remainder=False,
    target_mode="segmentation",
    classification_num_classes=1,
):
    """Build a high-throughput tf.data pipeline backed by RAM arrays."""
    cached = _load_split_to_ram(hdf5_path, split)
    if cached is not None:
        images, masks = cached
        num_samples = int(images.shape[0])
        image_shape = tuple(int(v) for v in images.shape[1:])
        mask_shape = tuple(int(v) for v in masks.shape[1:])
        tf_dtype = tf.float16 if images.dtype == np.float16 else tf.float32
        ds = tf.data.Dataset.from_generator(
            lambda: _iter_ram_samples(images, masks),
            output_signature=(
                tf.TensorSpec(shape=image_shape, dtype=tf_dtype),
                tf.TensorSpec(shape=mask_shape, dtype=tf_dtype),
            ),
        )
    else:
        num_samples, image_shape, mask_shape, _ = _get_split_meta(hdf5_path, split)
        ds = tf.data.Dataset.from_generator(
            lambda: _iter_split_samples(hdf5_path, split),
            output_signature=(
                tf.TensorSpec(shape=image_shape, dtype=tf.float32),
                tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
            ),
        )
        print(
            f"[DATA] Using streaming tf.data for split '{split}' "
            f"(RAM cap {RAM_CACHE_MAX_GIB:.2f} GiB)."
        )

    if shuffle:
        buffer_size = max(1, min(num_samples, STREAM_SHUFFLE_BUFFER))
        ds = ds.shuffle(buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True)

    augmentation_mode = AUGMENTATION_MODE
    augmentation_enabled = os.environ.get("BRATS_AUGMENTATION_ENABLED", "1") == "1"
    use_aug = bool(
        augment
        and split == "train"
        and augmentation_enabled
        and augmentation_mode != "off"
    )

    if use_aug:
        if augmentation_mode == "fast":
            ds = ds.map(_fast_flip_augment, num_parallel_calls=tf.data.AUTOTUNE)
            print("[INFO] BRATS_AUGMENTATION_MODE=fast -> using tf.data fast flip augmentation.")
        else:
            augment_pair = _resolve_cell6_callable("augment_pair")
            if augment_pair is None:
                ds = ds.map(_fast_flip_augment, num_parallel_calls=tf.data.AUTOTUNE)
                print("[WARN] Cell 6 augment_pair unavailable. Using tf.data fast flip augmentation.")
            else:
                ds = ds.map(lambda x, y: augment_pair(x, y), num_parallel_calls=tf.data.AUTOTUNE)
                print("[INFO] BRATS_AUGMENTATION_MODE=cell6 -> using Cell 6 augment_pair in tf.data map.")
    elif augment and split == "train" and augmentation_mode == "off":
        print("[INFO] BRATS_AUGMENTATION_MODE=off -> training augmentation disabled by config.")
    elif augment and split == "train" and not augmentation_enabled:
        print("[INFO] BRATS_AUGMENTATION_ENABLED=0 -> training augmentation disabled by config.")

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if str(target_mode).strip().lower() == "classification":
        ds = ds.map(
            lambda x, y: (x, _mask_batch_to_class_labels_tf(y, num_classes=classification_num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, num_samples


# ========================
# LEGACY HDF5 GENERATOR
# ========================
class HDF5Generator(tf.keras.utils.Sequence):
    """Thread-safe HDF5 sequence with persistent file handle."""

    def __init__(
        self,
        hdf5_path,
        split,
        batch_size,
        shuffle=True,
        augment=False,
        target_mode="segmentation",
        classification_num_classes=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        with h5py.File(hdf5_path, "r") as h5_file:
            self.n = int(h5_file[f"{split}/images"].shape[0])

        self.hdf5_path = hdf5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation_mode = AUGMENTATION_MODE
        self.augmentation_enabled = os.environ.get("BRATS_AUGMENTATION_ENABLED", "1") == "1"
        self.augment = bool(
            augment
            and split == "train"
            and self.augmentation_enabled
            and self.augmentation_mode != "off"
        )
        self.target_mode = str(target_mode).strip().lower()
        self.classification_num_classes = int(classification_num_classes)
        if self.target_mode == "classification" and self.classification_num_classes not in {1, 2}:
            raise ValueError(
                "classification_num_classes must be 1 or 2 when target_mode='classification'."
            )
        self._augment_fn = None
        self._use_fast_numpy_aug = False
        self.indices = np.arange(self.n)
        self.file = None
        self._lock = threading.Lock()

        if self.augment:
            if self.augmentation_mode == "fast":
                self._augment_fn = _safe_augment_pair
                self._use_fast_numpy_aug = True
                print("[INFO] BRATS_AUGMENTATION_MODE=fast -> using fast numpy flip augmentation.")
            else:
                self._augment_fn = _resolve_cell6_callable("augment_pair")
                if self._augment_fn is None:
                    self._augment_fn = _safe_augment_pair
                    self._use_fast_numpy_aug = True
                    print("[WARN] Cell 6 augment_pair unavailable. Using fast fallback flip augmentation.")
        elif augment and split == "train" and self.augmentation_mode == "off":
            print("[INFO] BRATS_AUGMENTATION_MODE=off -> training augmentation disabled by config.")
        elif augment and split == "train" and not self.augmentation_enabled:
            print("[INFO] BRATS_AUGMENTATION_ENABLED=0 -> training augmentation disabled by config.")

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def _ensure_open(self):
        with self._lock:
            if self.file is None:
                self.file = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Batch index {idx} out of range [0, {len(self)}).")

        self._ensure_open()

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n)

        # h5py fancy indexing requires ascending indices.
        batch_indices = np.sort(self.indices[start:end])
        batch_len = int(batch_indices.shape[0])
        contiguous = False
        if batch_len > 0:
            contiguous = bool(
                (int(batch_indices[-1]) - int(batch_indices[0]) + 1 == batch_len)
                and (np.all(np.diff(batch_indices) == 1) if batch_len > 1 else True)
            )

        # Serialize HDF5 reads across Keras workers to avoid thread-contention edge cases.
        with self._lock:
            if contiguous:
                lo = int(batch_indices[0])
                hi = int(batch_indices[-1]) + 1
                batch_x = self.file[f"{self.split}/images"][lo:hi].astype(np.float32)
                batch_y = self.file[f"{self.split}/masks"][lo:hi].astype(np.float32)
            else:
                batch_x = self.file[f"{self.split}/images"][batch_indices].astype(np.float32)
                batch_y = self.file[f"{self.split}/masks"][batch_indices].astype(np.float32)

        if self.augment and self._augment_fn is not None:
            if self._use_fast_numpy_aug:
                flip_mask = np.random.rand(batch_x.shape[0]) < 0.5
                if np.any(flip_mask):
                    batch_x[flip_mask] = batch_x[flip_mask, :, ::-1, :]
                    batch_y[flip_mask] = batch_y[flip_mask, :, ::-1, :]
            else:
                x_tensor = tf.convert_to_tensor(batch_x, dtype=tf.float32)
                y_tensor = tf.convert_to_tensor(batch_y, dtype=tf.float32)
                aug_x, aug_y = tf.map_fn(
                    lambda elems: self._augment_fn(elems[0], elems[1]),
                    (x_tensor, y_tensor),
                    fn_output_signature=(tf.float32, tf.float32),
                )
                batch_x = aug_x.numpy()
                batch_y = aug_y.numpy()

        if self.target_mode == "classification":
            has_tumor = np.max(batch_y, axis=(1, 2, 3)) > 0.5
            if self.classification_num_classes == 1:
                batch_y = has_tumor.astype(np.float32).reshape((-1, 1))
            else:
                batch_y = has_tumor.astype(np.int32)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def close(self):
        if self.file is not None:
            with suppress(Exception):
                self.file.close()
            self.file = None

    def __del__(self):
        self.close()


# ========================
# HELPERS
# ========================
def build_and_compile(model_type, strategy, steps_per_epoch=None):
    with strategy.scope():
        if model_type == "unet":
            model = build_unet()
        elif model_type == "attention_unet":
            model = build_attention_unet()
        elif model_type == "attention_unet_vit":
            if "build_attention_unet_vit" not in globals():
                raise RuntimeError(
                    "build_attention_unet_vit is unavailable. "
                    "Run Cell 07D or ensure cell_07d_attention_unet_vit_FIXED.py is present."
                )
            model = build_attention_unet_vit(num_classes=VIT_NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Enable accumulation only for smaller per-GPU batches (<16) to preserve throughput.
        if BATCH_SIZE_PER_GPU < 16:
            accumulation_steps = max(1, int(round(16 / max(1, BATCH_SIZE_PER_GPU))))
            if accumulation_steps > 1:
                if steps_per_epoch is not None and steps_per_epoch % accumulation_steps != 0:
                    raise ValueError(
                        "Gradient accumulation requires steps_per_epoch to be divisible by "
                        f"accumulation_steps to avoid dropping tail gradients. "
                        f"Got steps_per_epoch={steps_per_epoch}, "
                        f"accumulation_steps={accumulation_steps}."
                    )
                try:
                    from gradient_accumulation import wrap_with_gradient_accumulation

                    model = wrap_with_gradient_accumulation(model, accumulation_steps=accumulation_steps)
                    print(
                        f"[GRAD ACCUM] Enabled {accumulation_steps} accumulation steps "
                        f"(effective global batch: {GLOBAL_BATCH_SIZE * accumulation_steps})"
                    )
                except Exception as exc:
                    print(f"[WARN] Could not enable gradient accumulation: {exc}")

        optimizer = Adam(learning_rate=LEARNING_RATE, epsilon=1e-7, clipnorm=1.0)
        if USE_MIXED_PRECISION:
            optimizer = LossScaleOptimizer(optimizer)

        if _is_classification_model(model_type):
            if int(VIT_NUM_CLASSES) == 1:
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
            loss_fn = combined_loss
            metrics = [dice_coef, iou_metric, precision_metric, sensitivity_metric]

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )

    return model


def _assert_input_batch_ranges(batch_x, batch_y):
    """Validate a single batch for expected binary-segmentation ranges/shapes."""
    if batch_x.ndim != 4 or batch_x.shape[-1] != 4:
        raise RuntimeError(f"Expected input batch shape (B,H,W,4), got {batch_x.shape}")
    if batch_y.ndim != 4 or batch_y.shape[-1] != 1:
        raise RuntimeError(f"Expected mask batch shape (B,H,W,1), got {batch_y.shape}")
    if not np.all(np.isfinite(batch_x)) or not np.all(np.isfinite(batch_y)):
        raise RuntimeError("Input batch contains NaN/Inf values.")
    if ASSERT_INPUT_VALUE_RANGE:
        x_min = float(np.min(batch_x))
        x_max = float(np.max(batch_x))
        if x_min < -1e-4 or x_max > 1.0001:
            raise RuntimeError(f"Images expected in [0,1], got [{x_min:.4f}, {x_max:.4f}]")
    mask_unique = np.unique(batch_y)
    if not np.isin(mask_unique, [0.0, 1.0]).all():
        raise RuntimeError(f"Masks expected to be binary 0/1, got values: {mask_unique[:10]}")


def run_pretraining_sanity_checks(hdf5_path, strategy):
    """Run one-batch forward checks on both models before expensive training."""
    print("[SANITY] Running one-batch shape/value checks...")
    sample_ds, _ = make_tf_dataset(
        hdf5_path,
        "train",
        batch_size=1,
        shuffle=False,
        augment=False,
        drop_remainder=False,
    )
    batch_x, batch_y = next(iter(sample_ds.take(1)))
    batch_x_np = batch_x.numpy().astype(np.float32)
    batch_y_np = batch_y.numpy().astype(np.float32)

    _assert_input_batch_ranges(batch_x_np, batch_y_np)

    with strategy.scope():
        check_models = {
            "unet": build_unet(),
            "attention_unet": build_attention_unet(),
        }
        if _should_train_vit_classifier():
            if "build_attention_unet_vit" not in globals():
                raise RuntimeError(
                    "BRATS_ENABLE_VIT_CLASSIFIER requested but build_attention_unet_vit is unavailable."
                )
            check_models["attention_unet_vit"] = build_attention_unet_vit(num_classes=VIT_NUM_CLASSES)

    for name, model in check_models.items():
        preds = model(batch_x, training=False)
        preds_np = preds.numpy().astype(np.float32)
        if name == "attention_unet_vit":
            expected_last = 1 if int(VIT_NUM_CLASSES) == 1 else int(VIT_NUM_CLASSES)
            if preds_np.ndim != 2 or preds_np.shape[-1] != expected_last:
                raise RuntimeError(f"[SANITY] {name} output shape mismatch: {preds_np.shape}")
        elif preds_np.ndim != 4 or preds_np.shape[-1] != 1:
            raise RuntimeError(f"[SANITY] {name} output shape mismatch: {preds_np.shape}")
        if not np.all(np.isfinite(preds_np)):
            raise RuntimeError(f"[SANITY] {name} produced NaN/Inf outputs.")

    del check_models
    tf.keras.backend.clear_session()
    gc.collect()
    checked_names = ["U-Net", "Attention U-Net"]
    if _should_train_vit_classifier():
        checked_names.append("Attention U-Net + ViT")
    print(f"[SANITY] One-batch checks passed for {', '.join(checked_names)}.")


def cosine_annealing_with_warmup(epoch, lr):
    """Cosine annealing with linear warmup for stable convergence."""
    if epoch < WARMUP_EPOCHS:
        # Linear warmup from 0 to LEARNING_RATE
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    else:
        # Cosine annealing after warmup
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return max(MIN_LR, LEARNING_RATE * cosine_decay)


class ProgressiveAugCallback(tf.keras.callbacks.Callback):
    """Progressive augmentation: ramp up augmentation strength during training.

    Early epochs use mild augmentation so the model learns basic features.
    Later epochs use stronger augmentation to improve generalization.
    This is more stable than progressive resolution (which requires data pipeline changes)
    and achieves similar benefits.

    Sets BRATS_AUG_PROGRESSIVE_SCALE env var that augmentation functions can read.
    """

    def __init__(self, warmup_epochs=4, min_scale=0.6, max_scale=1.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._set_aug_scale_fn = globals().get("set_augmentation_scale", None)
        if self._set_aug_scale_fn is None:
            self._set_aug_scale_fn = _resolve_cell6_callable("set_augmentation_scale")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            scale = self.min_scale + (self.max_scale - self.min_scale) * (epoch / self.warmup_epochs)
        else:
            scale = self.max_scale

        # Primary runtime hook used by Cell 6 augmentation tf.functions.
        with suppress(Exception):
            if self._set_aug_scale_fn is not None:
                self._set_aug_scale_fn(scale)

        # Backward-compatible env signal.
        os.environ["BRATS_AUG_PROGRESSIVE_SCALE"] = f"{scale:.2f}"

        if epoch % 5 == 0:
            print(f"\n[PROGRESSIVE] Epoch {epoch}: augmentation scale = {scale:.2f}")


def _is_classification_model(model_name):
    return str(model_name).strip().lower() == "attention_unet_vit"


def _should_train_vit_classifier():
    return bool(ENABLE_VIT_CLASSIFIER or TRAINING_MODE in {"all", "vit_only"})


def _vit_monitor_metric():
    return "val_binary_accuracy" if int(VIT_NUM_CLASSES) == 1 else "val_sparse_categorical_accuracy"


def _monitor_metric_for_model(model_name):
    if _is_classification_model(model_name):
        return _vit_monitor_metric()
    return "val_dice_coef"


def _monitor_mode_for_model(model_name):
    return "max"


def _train_metric_for_model(model_name):
    if _is_classification_model(model_name):
        return "binary_accuracy" if int(VIT_NUM_CLASSES) == 1 else "sparse_categorical_accuracy"
    return "dice_coef"


def make_callbacks(model_name):
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.keras")
    monitor_metric = _monitor_metric_for_model(model_name)
    monitor_mode = _monitor_mode_for_model(model_name)

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        LearningRateScheduler(cosine_annealing_with_warmup, verbose=1),
        ProgressiveAugCallback(warmup_epochs=4, min_scale=0.6, max_scale=1.0),
        CSVLogger(os.path.join(LOG_DIR, f"{model_name}_log.csv")),
    ]
    return callbacks, checkpoint_path


def save_history(history, model_name):
    payload = {key: [float(v) for v in values] for key, values in history.history.items()}
    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_model_comparison_table(summary):
    """Write a compact markdown table for viva-ready model comparison evidence."""

    def _fmt_metric(value):
        return "N/A" if value is None else f"{float(value):.4f}"

    def _fmt_runtime(value):
        return "N/A" if value is None else f"{float(value):.1f}"

    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.md")
    lines = [
        "# U-Net vs Attention U-Net",
        "",
        f"> {PROJECT_POSITIONING}",
        "",
        "| Model | Best primary val metric | Runtime (minutes) | Trained in this run |",
        "|---|---:|---:|:---:|",
        (
            f"| U-Net | {_fmt_metric(summary.get('u_net_best_val_dice_coef'))} "
            f"| {_fmt_runtime(summary.get('u_net_runtime_minutes'))} "
            f"| {'Yes' if summary.get('u_net_trained_this_run') else 'No'} |"
        ),
        (
            f"| Attention U-Net | {_fmt_metric(summary.get('attention_u_net_best_val_dice_coef'))} "
            f"| {_fmt_runtime(summary.get('attention_u_net_runtime_minutes'))} "
            f"| {'Yes' if summary.get('attention_u_net_trained_this_run') else 'No'} |"
        ),
    ]

    if summary.get("attention_u_net_vit_enabled"):
        lines.append(
            (
                f"| Attention U-Net + ViT ({summary.get('attention_u_net_vit_metric_name', 'metric')}) "
                f"| {_fmt_metric(summary.get('attention_u_net_vit_best_val_metric'))} "
                f"| {_fmt_runtime(summary.get('attention_u_net_vit_runtime_minutes'))} "
                f"| {'Yes' if summary.get('attention_u_net_vit_trained_this_run') else 'No'} |"
            )
        )

    lines.extend([
        "",
        f"Best segmentation model in available checkpoints: {summary.get('best_model_name', 'N/A')}",
        (
            "Relative gain (Attention over U-Net): "
            + (
                "N/A"
                if summary.get("gain_percent") is None
                else f"{float(summary['gain_percent']):+.2f}%"
            )
        ),
        "",
        "## Setup Expected Binary Dice",
        "",
        "| Setup | Expected Binary Dice |",
        "|---|---:|",
        f"| Basic U-Net, 2D slices | {summary.get('target_binary_dice_unet', BINARY_TARGET_UNET)} |",
        f"| Attention U-Net, 2D | {summary.get('target_binary_dice_attention', BINARY_TARGET_ATTENTION)} |",
    ])

    with open(comparison_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"[OK] Model comparison table saved to: {comparison_path}")
    return comparison_path


def get_best_metric_from_history(model_name):
    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    if not os.path.exists(history_path):
        return 0.0

    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metric_name = _monitor_metric_for_model(model_name)
        return _safe_best_metric(payload.get(metric_name, []), default=0.0)
    except json.JSONDecodeError as exc:
        print(f"Warning: malformed history JSON at {history_path}: {exc}")
        return 0.0
    except Exception as exc:
        print(f"Warning: could not parse history at {history_path}: {exc}")
        return 0.0


def get_best_dice_from_history(model_name):
    """Backward-compatible alias; now delegates to monitor-specific best metric."""
    return get_best_metric_from_history(model_name)


def _safe_best_metric(values, default=0.0):
    cleaned = []
    if values is None:
        values = []

    for value in values:
        with suppress(Exception):
            metric = float(value)
            if np.isfinite(metric):
                cleaned.append(metric)

    return float(max(cleaned)) if cleaned else float(default)


class TimeEstimateCallback(tf.keras.callbacks.Callback):
    """Prints live ETA so Kaggle runs show expected completion time."""

    def __init__(
        self,
        model_name,
        steps_per_epoch,
        total_epochs,
        report_every=25,
        train_metric_name="dice_coef",
        val_metric_name="val_dice_coef",
    ):
        super().__init__()
        self.model_name = model_name
        self.steps_per_epoch = int(max(1, steps_per_epoch))
        self.total_epochs = int(max(1, total_epochs))
        self.report_every = int(max(1, report_every))
        self.train_metric_name = str(train_metric_name)
        self.val_metric_name = str(val_metric_name)
        self.epoch_durations = []
        self._train_start = None
        self._epoch_start = None
        self._epoch_index = 0
        self._batch_start = None
        self._batch_times = []

    @staticmethod
    def _fmt_seconds(seconds):
        seconds = int(max(0, seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _fmt_metric(value):
        if value is None:
            return "N/A"
        with suppress(Exception):
            metric = float(value)
            if np.isfinite(metric):
                return f"{metric:.4f}"
        return "N/A"

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        print(f"[ETA] {self.model_name}: timing enabled for {self.total_epochs} epoch(s)")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        self._epoch_index = int(epoch)
        self._batch_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is None:
            return
        dt = time.time() - self._batch_start
        self._batch_times.append(dt)

        # During first epoch, print coarse estimate every N batches.
        completed_batches = batch + 1
        if completed_batches % self.report_every == 0 and self._batch_times:
            avg_step = float(np.mean(self._batch_times))
            epoch_est = avg_step * self.steps_per_epoch
            remaining_epochs = max(0, self.total_epochs - (self._epoch_index + 1))
            remaining_total = epoch_est * remaining_epochs
            print(
                f"[ETA] {self.model_name}: ~{self._fmt_seconds(epoch_est)} / epoch "
                f"| projected remaining after this epoch: ~{self._fmt_seconds(remaining_total)}"
            )

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start is None:
            return

        logs = logs or {}
        epoch_time = time.time() - self._epoch_start
        self.epoch_durations.append(epoch_time)

        avg_epoch = float(np.mean(self.epoch_durations))
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta = avg_epoch * remaining_epochs

        train_metric_value = logs.get(self.train_metric_name)
        loss_value = logs.get("loss")
        val_metric_value = logs.get(self.val_metric_name)
        val_loss_value = logs.get("val_loss")

        print(
            f"[METRICS] {self.model_name}: "
            f"{self.train_metric_name}: {self._fmt_metric(train_metric_value)} - "
            f"loss: {self._fmt_metric(loss_value)} - "
            f"{self.val_metric_name}: {self._fmt_metric(val_metric_value)} - "
            f"val_loss: {self._fmt_metric(val_loss_value)}"
        )
        print(
            f"[ETA] {self.model_name}: ~{self._fmt_seconds(avg_epoch)} / epoch "
            f"| projected remaining after this epoch: ~{self._fmt_seconds(eta)}"
        )

    def on_train_end(self, logs=None):
        if self._train_start is None:
            return
        total = time.time() - self._train_start
        print(f"[ETA] {self.model_name}: total training time {self._fmt_seconds(total)}")


def train_model(model_type, hdf5_path, strategy):
    print(f"\nTraining {model_type.upper()}...")
    is_classifier = _is_classification_model(model_type)
    target_mode = "classification" if is_classifier else "segmentation"
    val_monitor_metric = _monitor_metric_for_model(model_type)
    train_monitor_metric = _train_metric_for_model(model_type)

    _, cached_checkpoint = make_callbacks(model_type)
    if SKIP_TRAIN_IF_EXISTS and os.path.exists(cached_checkpoint):
        print(f"Found cached checkpoint for {model_type}: {cached_checkpoint}")
        try:
            model = tf.keras.models.load_model(cached_checkpoint, custom_objects=CUSTOM_OBJECTS)
            return model, None, cached_checkpoint, 0.0, False
        except Exception as exc:
            print(f"Cached model load failed ({exc}). Retraining {model_type}.")

    train_ds, train_count = make_tf_dataset(
        hdf5_path,
        "train",
        GLOBAL_BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        augment=True,
        drop_remainder=True,
        target_mode=target_mode,
        classification_num_classes=VIT_NUM_CLASSES,
    )
    train_drop_remainder = True
    if train_count < GLOBAL_BATCH_SIZE:
        train_drop_remainder = False
        print(
            f"[WARN] Train samples ({train_count}) < batch size ({GLOBAL_BATCH_SIZE}). "
            "Rebuilding train dataset with drop_remainder=False."
        )
        train_ds, train_count = make_tf_dataset(
            hdf5_path,
            "train",
            GLOBAL_BATCH_SIZE,
            shuffle=TRAIN_SHUFFLE,
            augment=True,
            drop_remainder=False,
            target_mode=target_mode,
            classification_num_classes=VIT_NUM_CLASSES,
        )

    val_ds, val_count = make_tf_dataset(
        hdf5_path,
        "val",
        GLOBAL_BATCH_SIZE,
        shuffle=False,
        augment=False,
        drop_remainder=False,
        target_mode=target_mode,
        classification_num_classes=VIT_NUM_CLASSES,
    )

    if train_drop_remainder:
        full_train_steps = max(1, train_count // GLOBAL_BATCH_SIZE)
    else:
        full_train_steps = max(1, math.ceil(train_count / GLOBAL_BATCH_SIZE))
    steps_per_epoch = max(1, min(math.ceil(full_train_steps * STEPS_FRACTION), full_train_steps))
    if MAX_STEPS_PER_EPOCH > 0:
        steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)
    full_val_steps = max(1, math.ceil(val_count / GLOBAL_BATCH_SIZE))
    val_steps = max(1, min(math.ceil(full_val_steps * VAL_STEPS_FRACTION), full_val_steps))
    if MAX_VAL_STEPS > 0:
        val_steps = min(val_steps, MAX_VAL_STEPS)

    print(f"Train steps per epoch: {steps_per_epoch} (full: {full_train_steps})")
    print(f"Validation steps: {val_steps} (full: {full_val_steps})")

    if steps_per_epoch < full_train_steps:
        train_ds = train_ds.take(steps_per_epoch)
    if val_steps < full_val_steps:
        val_ds = val_ds.take(val_steps)

    model = build_and_compile(model_type, strategy, steps_per_epoch=steps_per_epoch)
    callbacks, checkpoint_path = make_callbacks(model_type)
    callbacks.append(
        TimeEstimateCallback(
            model_type,
            steps_per_epoch,
            EPOCHS,
            train_metric_name=train_monitor_metric,
            val_metric_name=val_monitor_metric,
        )
    )

    start_time = time.time()
    history = None
    try:
        fit_kwargs = dict(
            x=train_ds,
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            epochs=EPOCHS,
            validation_freq=VAL_EVERY_N_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        history = model.fit(**fit_kwargs)
    except Exception as exc:
        print(f"Training failed for {model_type}: {exc}")
        traceback.print_exc()
        raise

    elapsed = time.time() - start_time
    best_val = _safe_best_metric(history.history.get(val_monitor_metric, []), default=0.0) if history else 0.0
    print(
        f"{model_type.upper()} finished in {elapsed / 3600.0:.2f}h | "
        f"Best {val_monitor_metric}: {best_val:.4f}"
    )

    if history is not None:
        save_history(history, model_type)

    return model, history, checkpoint_path, elapsed, True


def tune_thresholds_post_training(model_path, hdf5_path, num_batches=8):
    if not os.path.exists(model_path):
        print(f"[WARN] Model path missing for threshold tuning: {model_path}. Using 0.50")
        return [0.5]
    if not os.path.exists(hdf5_path):
        print(f"[WARN] HDF5 missing for threshold tuning: {hdf5_path}. Using 0.50")
        return [0.5]

    sample_batch_size = int(os.environ.get("BRATS_THRESHOLD_TUNE_BATCH_SIZE", "16"))
    max_samples = max(sample_batch_size, int(num_batches) * sample_batch_size)

    model = None
    try:
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        except Exception:
            model = tf.keras.models.load_model(model_path, compile=False)

        with h5py.File(hdf5_path, "r") as h5_file:
            if "val/images" not in h5_file or "val/masks" not in h5_file:
                print("[WARN] Validation split not found for threshold tuning. Using 0.50")
                return [0.5]

            n_samples = int(h5_file["val/images"].shape[0])
            if n_samples <= 0:
                print("[WARN] Validation split empty for threshold tuning. Using 0.50")
                return [0.5]

            sample_count = min(n_samples, max_samples)
            x_val = h5_file["val/images"][:sample_count].astype(np.float32)
            y_val = h5_file["val/masks"][:sample_count].astype(np.float32)

        if y_val.ndim == 4 and y_val.shape[-1] != 1:
            y_val = np.max(y_val, axis=-1, keepdims=True)
        y_val = (y_val > 0.5).astype(np.float32)

        preds = model.predict(x_val, batch_size=sample_batch_size, verbose=0).astype(np.float32)
        if preds.ndim == 4 and preds.shape[-1] != 1:
            preds = np.max(preds, axis=-1, keepdims=True)

        best_thresh, best_dice = 0.5, 0.0
        for threshold in np.arange(0.30, 0.71, 0.05):
            y_hard = (preds > threshold).astype(np.float32)
            inter = np.sum(y_val * y_hard, dtype=np.float64)
            union = np.sum(y_val, dtype=np.float64) + np.sum(y_hard, dtype=np.float64)
            dice = float((2.0 * inter + 1e-6) / (union + 1e-6))
            if dice > best_dice:
                best_dice = dice
                best_thresh = float(threshold)

        print(
            f"[OK] Optimal binary threshold: {best_thresh:.2f} "
            f"(val Dice={best_dice:.4f}, samples={sample_count})"
        )
        return [best_thresh]

    except Exception as exc:
        print(f"[WARN] Threshold tuning failed ({exc}). Falling back to 0.50")
        return [0.5]
    finally:
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()


# ========================
# MAIN
# ========================
def main():
    print("\n" + "=" * 60)
    print("CELL 9: DUAL MODEL TRAINING (PRODUCTION — ALL 12 FIXES)")
    print("=" * 60)
    print(PROJECT_POSITIONING)

    if not os.path.exists(HDF5_PATH):
        _attempt_autobuild_hdf5_from_npz()

    validate_hdf5(HDF5_PATH)
    run_pretraining_sanity_checks(HDF5_PATH, STRATEGY)
    total_start = time.time()

    u_ckpt = os.path.join(MODEL_DIR, "unet_best.keras")
    a_ckpt = os.path.join(MODEL_DIR, "attention_unet_best.keras")
    v_ckpt = os.path.join(MODEL_DIR, "attention_unet_vit_best.keras")
    u_best = get_best_dice_from_history("unet") if os.path.exists(u_ckpt) else None
    a_best = get_best_dice_from_history("attention_unet") if os.path.exists(a_ckpt) else None
    v_best = get_best_metric_from_history("attention_unet_vit") if os.path.exists(v_ckpt) else None
    u_elapsed = None
    a_elapsed = None
    v_elapsed = None
    u_trained_this_run = False
    a_trained_this_run = False
    v_trained_this_run = False

    # 1) Baseline U-Net (optional by mode)
    if TRAINING_MODE in {"both", "unet_only", "all"}:
        u_model, u_history, u_ckpt, u_elapsed, u_trained_this_run = train_model("unet", HDF5_PATH, STRATEGY)
        u_best = (
            _safe_best_metric(u_history.history.get("val_dice_coef", []), default=0.0)
            if u_history is not None
            else get_best_metric_from_history("unet")
        )

        del u_model
        del u_history
        tf.keras.backend.clear_session()
        gc.collect()
    else:
        print(f"[INFO] BRATS_TRAINING_MODE={TRAINING_MODE} -> skipping U-Net training.")

    # 2) Attention U-Net uses the same global strategy as U-Net (optional by mode)
    attention_strategy = STRATEGY
    print(f"Using shared strategy for attention model: {STRATEGY.__class__.__name__}")

    if TRAINING_MODE in {"both", "attention_only", "all"}:
        a_model, a_history, a_ckpt, a_elapsed, a_trained_this_run = train_model(
            "attention_unet", HDF5_PATH, attention_strategy
        )
        a_best = (
            _safe_best_metric(a_history.history.get("val_dice_coef", []), default=0.0)
            if a_history is not None
            else get_best_metric_from_history("attention_unet")
        )

        del a_model
        del a_history
        tf.keras.backend.clear_session()
        gc.collect()
    else:
        print(f"[INFO] BRATS_TRAINING_MODE={TRAINING_MODE} -> skipping Attention U-Net training.")

    # 3) Attention U-Net + ViT classifier (optional)
    if _should_train_vit_classifier():
        monitor_metric = _monitor_metric_for_model("attention_unet_vit")
        v_model, v_history, v_ckpt, v_elapsed, v_trained_this_run = train_model(
            "attention_unet_vit", HDF5_PATH, STRATEGY
        )
        v_best = (
            _safe_best_metric(v_history.history.get(monitor_metric, []), default=0.0)
            if v_history is not None
            else get_best_metric_from_history("attention_unet_vit")
        )

        del v_model
        del v_history
        tf.keras.backend.clear_session()
        gc.collect()
    else:
        print("[INFO] AttentionUNetViT classifier training disabled.")

    # Training datasets are no longer needed; free cache before post-training evaluation.
    clear_ram_data_cache()

    # 4) Post-training: binary threshold handling on best segmentation model
    candidates = []
    if u_best is not None and os.path.exists(u_ckpt):
        candidates.append(("U-Net", u_ckpt, float(u_best)))
    if a_best is not None and os.path.exists(a_ckpt):
        candidates.append(("Attention U-Net", a_ckpt, float(a_best)))

    best_name = None
    best_ckpt = None
    best_metric = None
    optimal_thresholds = None
    active_binary_threshold = 0.5

    if candidates:
        best_name, best_ckpt, best_metric = max(candidates, key=lambda item: item[2])
        print(f"\nBest segmentation model: {best_name} (val_dice_coef={best_metric:.4f})")

        if ENABLE_THRESHOLD_TUNING:
            optimal_thresholds = tune_thresholds_post_training(
                best_ckpt,
                HDF5_PATH,
                num_batches=THRESHOLD_TUNE_BATCHES,
            )
        else:
            print("[INFO] Threshold tuning disabled; using default binary threshold=0.5")

        active_binary_threshold = float(optimal_thresholds[0]) if optimal_thresholds else 0.5
    else:
        print("[INFO] No segmentation checkpoint found; skipping threshold tuning.")
    thresholds_path = os.environ.get("BRATS_THRESHOLDS_PATH", "").strip()
    if not thresholds_path:
        thresholds_path = os.path.join(RESULTS_DIR, "optimal_thresholds.json")

    thresholds_payload = {
        "binary": active_binary_threshold,
        "best_model": best_name,
        "best_model_checkpoint": best_ckpt,
        "source": "cell_09_training_FIXED",
    }
    thresholds_dir = os.path.dirname(thresholds_path)
    if thresholds_dir:
        os.makedirs(thresholds_dir, exist_ok=True)
    with open(thresholds_path, "w", encoding="utf-8") as handle:
        json.dump(thresholds_payload, handle, indent=2)
    print(f"[OK] Saved binary threshold to: {thresholds_path}")

    total_elapsed = time.time() - total_start
    summary = {
        "project_positioning": PROJECT_POSITIONING,
        "training_mode": TRAINING_MODE,
        "validation_freq_epochs": VAL_EVERY_N_EPOCHS,
        "target_binary_dice_unet": BINARY_TARGET_UNET,
        "target_binary_dice_attention": BINARY_TARGET_ATTENTION,
        "threshold_tuning_enabled": ENABLE_THRESHOLD_TUNING,
        "threshold_tune_batches": THRESHOLD_TUNE_BATCHES if ENABLE_THRESHOLD_TUNING else None,
        "u_net_best_val_dice_coef": float(u_best) if u_best is not None else None,
        "attention_u_net_best_val_dice_coef": float(a_best) if a_best is not None else None,
        "attention_u_net_vit_enabled": bool(_should_train_vit_classifier()),
        "attention_u_net_vit_metric_name": _monitor_metric_for_model("attention_unet_vit"),
        "attention_u_net_vit_best_val_metric": float(v_best) if v_best is not None else None,
        "gain_percent": (
            float((a_best - u_best) / u_best * 100.0)
            if (u_best is not None and a_best is not None and u_best > 0)
            else None
        ),
        "u_net_checkpoint": u_ckpt,
        "attention_u_net_checkpoint": a_ckpt,
        "attention_u_net_vit_checkpoint": v_ckpt,
        "u_net_trained_this_run": bool(u_trained_this_run),
        "attention_u_net_trained_this_run": bool(a_trained_this_run),
        "attention_u_net_vit_trained_this_run": bool(v_trained_this_run),
        "u_net_runtime_minutes": float(u_elapsed / 60.0) if u_elapsed is not None else None,
        "attention_u_net_runtime_minutes": float(a_elapsed / 60.0) if a_elapsed is not None else None,
        "attention_u_net_vit_runtime_minutes": float(v_elapsed / 60.0) if v_elapsed is not None else None,
        "best_model_name": best_name,
        "best_model_checkpoint": best_ckpt,
        "best_model_val_dice_coef": float(best_metric) if best_metric is not None else None,
        "total_runtime_minutes": float(total_elapsed / 60.0),
        "optimal_thresholds": {
            "binary": active_binary_threshold,
        },
    }

    summary_path = os.path.join(RESULTS_DIR, "dual_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_path = save_model_comparison_table(summary)

    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    print(f"\nDual training complete in {hours}h {minutes}m")
    if u_best is not None:
        print(f"U-Net best val_dice_coef: {u_best:.4f}")
    else:
        print("U-Net best val_dice_coef: N/A (not trained and no cached history).")
    if a_best is not None:
        print(f"Attention U-Net best val_dice_coef: {a_best:.4f}")
    else:
        print("Attention U-Net best val_dice_coef: N/A (not trained and no cached history).")
    if v_best is not None:
        print(
            "Attention U-Net + ViT best "
            f"{_monitor_metric_for_model('attention_unet_vit')}: {v_best:.4f}"
        )
    elif _should_train_vit_classifier():
        print("Attention U-Net + ViT best metric: N/A (not trained and no cached history).")
    if summary["gain_percent"] is not None:
        print(f"Relative gain: {summary['gain_percent']:+.2f}%")
    print(f"Threshold: {active_binary_threshold:.2f}")
    print(f"Summary saved to: {summary_path}")
    print(f"Comparison table saved to: {comparison_path}")


if __name__ == "__main__":
    main()
    print("\n[OK] Cell 9 complete.")
