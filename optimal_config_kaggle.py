# ===================================================
# KAGGLE-OPTIMIZED CONFIG (25 EPOCHS — FITS IN 12h SESSION)
# ===================================================
# For an emergency speed profile (~1-2h), use:
#   import optimal_config_kaggle_ultrafast
#
# Run this BEFORE any other cells:
#   import optimal_config_kaggle
#
# Timeline:
#   Prep (Cells 1-8):   ~1.5h
#   U-Net (25 epochs):  ~4.2h
#   Attention U-Net:    ~5.2h
#   Eval + Export:      ~0.5h
#   TOTAL:              ~11h (1h buffer for Kaggle)
# ===================================================

import os

# ========================
# TRAINING
# ========================
os.environ['FAST_DEV_MODE'] = '0'
os.environ['EPOCHS'] = '25'                    # Full Kaggle profile
os.environ['BATCH_SIZE_PER_GPU'] = '16'
os.environ['LEARNING_RATE'] = '4e-4'
os.environ['WARMUP_EPOCHS'] = '2'
os.environ['SKIP_TRAIN_IF_EXISTS'] = '0'       # Force retrain
os.environ['BRATS_IMG_SIZE'] = '128'
os.environ['BRATS_NPZ_CHANNEL_ORDER'] = 't1,t1ce,t2,flair'
os.environ['BRATS_MAX_TRAIN_SAMPLES'] = '0'    # 0 = use all available slices
os.environ['BRATS_MAX_VAL_SAMPLES'] = '0'
os.environ['BRATS_MAX_TEST_SAMPLES'] = '0'
os.environ['BRATS_MAX_STEPS_PER_EPOCH'] = '1000'
os.environ['BRATS_MAX_VAL_STEPS'] = '220'
os.environ['BRATS_VAL_EVERY_N_EPOCHS'] = '2'

# ========================
# PATIENCE (tighter for 25 epochs)
# ========================
os.environ['PATIENCE_ES'] = '5'                # Stop early if plateau
os.environ['PATIENCE_LR'] = '2'               # Reduce LR faster

# ========================
# LOSS
# ========================
os.environ['BRATS_BCE_WEIGHT'] = '0.2'
os.environ['BRATS_TVERSKY_ALPHA'] = '0.3'
os.environ['BRATS_TVERSKY_BETA'] = '0.7'
os.environ['BRATS_POSITIVE_CLASS_WEIGHT'] = '4.0'

# ========================
# AUGMENTATION
# ========================
os.environ['BRATS_ROTATION_RANGE'] = '20'
os.environ['BRATS_AUG_PROBABILITY'] = '0.7'

# ========================
# BINARY INFERENCE SETTINGS
# ========================
os.environ['BRATS_ENFORCE_HIERARCHY'] = '0'
os.environ['BRATS_INFERENCE_THRESHOLD'] = '0.5'

# ========================
# PRINT SUMMARY
# ========================
print("=" * 60)
print("KAGGLE OPTIMIZED CONFIG (25 EPOCHS)")
print("=" * 60)
print(f"  EPOCHS:              {os.environ['EPOCHS']}")
print(f"  BATCH_SIZE:          {os.environ['BATCH_SIZE_PER_GPU']}")
print(f"  LEARNING_RATE:       {os.environ['LEARNING_RATE']}")
print(f"  WARMUP_EPOCHS:       {os.environ['WARMUP_EPOCHS']}")
print(f"  BRATS_IMG_SIZE:      {os.environ['BRATS_IMG_SIZE']}")
print(f"  MAX_TRAIN/VAL/TEST:  {os.environ['BRATS_MAX_TRAIN_SAMPLES']}/{os.environ['BRATS_MAX_VAL_SAMPLES']}/{os.environ['BRATS_MAX_TEST_SAMPLES']} (0=all)")
print(f"  MAX_STEPS_PER_EPOCH: {os.environ['BRATS_MAX_STEPS_PER_EPOCH']}")
print(f"  MAX_VAL_STEPS:       {os.environ['BRATS_MAX_VAL_STEPS']}")
print(f"  VAL_EVERY_N_EPOCHS:  {os.environ['BRATS_VAL_EVERY_N_EPOCHS']}")
print(f"  PATIENCE_ES:         {os.environ['PATIENCE_ES']}")
print(f"  PATIENCE_LR:         {os.environ['PATIENCE_LR']}")
print(f"  POSITIVE_CLASS_WEIGHT: {os.environ['BRATS_POSITIVE_CLASS_WEIGHT']}")
print(f"  HIERARCHY:           OFF (binary mode)")
print(f"  BINARY_THRESHOLD:    {os.environ['BRATS_INFERENCE_THRESHOLD']}")
print("=" * 60)
print(f"  Expected time:       ~8-10 hours")
print(f"  Kaggle 12h limit:    safer buffer OK")
print("=" * 60)
print()
print("Expected validation dice_coef progression:")
print("  Epoch  5 -> ~0.50-0.55")
print("  Epoch 10 -> ~0.60-0.65")
print("  Epoch 15 -> ~0.65-0.70")
print("  Epoch 20 -> ~0.70-0.75")
print("  Epoch 25 -> ~0.73-0.79")
