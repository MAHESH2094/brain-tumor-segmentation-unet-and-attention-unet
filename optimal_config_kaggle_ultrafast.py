# ===================================================
# KAGGLE EMERGENCY SPEED CONFIG (ATTENTION-ONLY, ~1-2h)
# ===================================================
# Run this BEFORE any other notebook cells:
#   import optimal_config_kaggle_ultrafast
#
# This profile is designed for catastrophic slowdown scenarios
# (for example, 30s/step or multi-hour epochs).
# ===================================================

import os


# ========================
# CORE SPEED PROFILE
# ========================
os.environ["FAST_DEV_MODE"] = "1"
os.environ["EPOCHS"] = "8"
os.environ["BATCH_SIZE_PER_GPU"] = "8"
os.environ["STEPS_FRACTION"] = "0.25"
os.environ["VAL_STEPS_FRACTION"] = "0.35"
os.environ["BRATS_IMG_SIZE"] = "128"
os.environ["BRATS_NPZ_CHANNEL_ORDER"] = "t1,t1ce,t2,flair"
os.environ["BRATS_MAX_TRAIN_SAMPLES"] = "8000"
os.environ["BRATS_MAX_VAL_SAMPLES"] = "2000"
os.environ["BRATS_MAX_TEST_SAMPLES"] = "1000"
os.environ["BRATS_TRAINING_MODE"] = "attention_only"
os.environ["BRATS_AUGMENTATION_MODE"] = "fast"
os.environ["LEARNING_RATE"] = "6e-4"
os.environ["BRATS_BCE_WEIGHT"] = "0.2"
os.environ["BRATS_TVERSKY_ALPHA"] = "0.3"
os.environ["BRATS_TVERSKY_BETA"] = "0.7"


# ========================
# MEMORY / THROUGHPUT PROFILE
# ========================
os.environ["BRATS_DATA_PIPELINE_MODE"] = "auto"
os.environ["BRATS_RAM_CACHE_MAX_GIB"] = "22"
os.environ["BRATS_RAM_CACHE_DTYPE"] = "float16"
os.environ["BRATS_RAM_CACHE_SPLITS"] = "train"
os.environ["BRATS_RAM_LOAD_CHUNK_SAMPLES"] = "128"
os.environ["BRATS_STREAM_SHUFFLE_BUFFER"] = "512"
os.environ["BRATS_MAX_STEPS_PER_EPOCH"] = "120"
os.environ["BRATS_MAX_VAL_STEPS"] = "30"
os.environ["BRATS_ENABLE_THRESHOLD_TUNING"] = "0"


# ========================
# STABILITY DEFAULTS
# ========================
os.environ.setdefault("WARMUP_EPOCHS", "1")
os.environ.setdefault("PATIENCE_ES", "5")
os.environ.setdefault("SKIP_TRAIN_IF_EXISTS", "0")


print("=" * 64)
print("KAGGLE EMERGENCY SPEED CONFIG ACTIVE")
print("=" * 64)
print(f"FAST_DEV_MODE:           {os.environ['FAST_DEV_MODE']}")
print(f"EPOCHS:                  {os.environ['EPOCHS']}")
print(f"BATCH_SIZE_PER_GPU:      {os.environ['BATCH_SIZE_PER_GPU']}")
print(f"BRATS_IMG_SIZE:          {os.environ['BRATS_IMG_SIZE']}")
print(f"STEPS_FRACTION:          {os.environ['STEPS_FRACTION']}")
print(f"VAL_STEPS_FRACTION:      {os.environ['VAL_STEPS_FRACTION']}")
print(f"BRATS_TRAINING_MODE:     {os.environ['BRATS_TRAINING_MODE']}")
print(f"BRATS_AUGMENTATION_MODE: {os.environ['BRATS_AUGMENTATION_MODE']}")
print(f"BRATS_MAX_TRAIN_SAMPLES: {os.environ['BRATS_MAX_TRAIN_SAMPLES']}")
print(f"BRATS_MAX_VAL_SAMPLES:   {os.environ['BRATS_MAX_VAL_SAMPLES']}")
print(f"BRATS_MAX_TEST_SAMPLES:  {os.environ['BRATS_MAX_TEST_SAMPLES']}")
print(f"LEARNING_RATE:           {os.environ['LEARNING_RATE']}")
print(f"BRATS_DATA_PIPELINE_MODE:{os.environ['BRATS_DATA_PIPELINE_MODE']}")
print(f"BRATS_RAM_CACHE_MAX_GIB: {os.environ['BRATS_RAM_CACHE_MAX_GIB']}")
print(f"BRATS_RAM_CACHE_DTYPE:   {os.environ['BRATS_RAM_CACHE_DTYPE']}")
print(f"BRATS_RAM_CACHE_SPLITS:  {os.environ['BRATS_RAM_CACHE_SPLITS']}")
print(f"BRATS_STREAM_SHUFFLE_BUFFER:{os.environ['BRATS_STREAM_SHUFFLE_BUFFER']}")
print(f"BRATS_MAX_STEPS_PER_EPOCH:{os.environ['BRATS_MAX_STEPS_PER_EPOCH']}")
print(f"BRATS_MAX_VAL_STEPS:     {os.environ['BRATS_MAX_VAL_STEPS']}")
print("=" * 64)
print("Action: Stop current training run, then restart kernel before rerun.")
print("Run order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7a -> 7b -> 7c -> 8 -> 9")
print("Skip for speed: Cells 10-13 until training completes.")
print("Expected runtime: ~2-4h total with this emergency profile")
print("=" * 64)
