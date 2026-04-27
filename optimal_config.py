# ===================================================
# OPTIMAL CONFIGURATION FOR BRATS PIPELINE (BINARY)
# ===================================================
# Run this BEFORE any other cells to set binary-segmentation defaults.

import os

# ========================
# TRAINING CONFIGURATION
# ========================
os.environ['FAST_DEV_MODE'] = '0'
os.environ['EPOCHS'] = '50'
os.environ['BATCH_SIZE_PER_GPU'] = '16'
os.environ['LEARNING_RATE'] = '4e-4'
os.environ['WARMUP_EPOCHS'] = '2'
os.environ['BRATS_IMG_SIZE'] = '128'
os.environ['BRATS_NPZ_CHANNEL_ORDER'] = 't1,t1ce,t2,flair'
os.environ['BRATS_MAX_TRAIN_SAMPLES'] = '0'    # 0 = use all available slices
os.environ['BRATS_MAX_VAL_SAMPLES'] = '0'
os.environ['BRATS_MAX_TEST_SAMPLES'] = '0'

# ========================
# PATIENCE / SCHEDULING
# ========================
os.environ['PATIENCE_ES'] = '8'
os.environ['PATIENCE_LR'] = '3'
os.environ['SKIP_TRAIN_IF_EXISTS'] = '0'

# ========================
# LOSS FUNCTION BALANCE
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
# VERIFICATION
# ========================
print("=" * 60)
print("OPTIMAL BINARY CONFIGURATION LOADED")
print("=" * 60)
print(f"  FAST_DEV_MODE:             {os.environ['FAST_DEV_MODE']} (OFF)")
print(f"  EPOCHS:                    {os.environ['EPOCHS']}")
print(f"  BATCH_SIZE_PER_GPU:        {os.environ['BATCH_SIZE_PER_GPU']}")
print(f"  LEARNING_RATE:             {os.environ['LEARNING_RATE']}")
print(f"  WARMUP_EPOCHS:             {os.environ['WARMUP_EPOCHS']}")
print(f"  BRATS_IMG_SIZE:            {os.environ['BRATS_IMG_SIZE']}")
print(f"  MAX_TRAIN/VAL/TEST:        {os.environ['BRATS_MAX_TRAIN_SAMPLES']}/{os.environ['BRATS_MAX_VAL_SAMPLES']}/{os.environ['BRATS_MAX_TEST_SAMPLES']} (0=all)")
print(f"  PATIENCE_ES:               {os.environ['PATIENCE_ES']}")
print(f"  PATIENCE_LR:               {os.environ['PATIENCE_LR']}")
print(f"  BRATS_BCE_WEIGHT:          {os.environ['BRATS_BCE_WEIGHT']}")
print(f"  BRATS_TVERSKY_ALPHA:       {os.environ['BRATS_TVERSKY_ALPHA']}")
print(f"  BRATS_TVERSKY_BETA:        {os.environ['BRATS_TVERSKY_BETA']}")
print(f"  BRATS_POSITIVE_CLASS_WEIGHT: {os.environ['BRATS_POSITIVE_CLASS_WEIGHT']}")
print(f"  ROTATION_RANGE:            +/-{os.environ['BRATS_ROTATION_RANGE']} deg")
print(f"  AUG_PROBABILITY:           {os.environ['BRATS_AUG_PROBABILITY']}")
print(f"  ENFORCE_HIERARCHY:         {os.environ['BRATS_ENFORCE_HIERARCHY']} (OFF for binary)")
print(f"  BINARY_THRESHOLD:          {os.environ['BRATS_INFERENCE_THRESHOLD']}")
print("=" * 60)

print("\nBinary optimization notes:")
print("  1. Tumor-vs-background objective with combined_loss")
print("  2. dice_coef is the primary validation metric")
print("  3. Positive-class reweighting can help severe imbalance")
print("  4. Single-threshold inference keeps deployment simple")
