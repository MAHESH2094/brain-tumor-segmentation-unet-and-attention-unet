# ===================================================
# CELL 2: Dataset Paths & Verification (FIXED)
# ===================================================
# Purpose: Find BraTS training data (raw NIfTI+seg or preprocessed NPZ patches)
# FIXES: Error logging with path info, guard empty dataset, standardize f-strings

import os
from glob import glob


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _read_env_path_list(var_name):
    """Read a list of paths from env (os.pathsep-separated)."""
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(os.pathsep) if p.strip()]

# ========================
# OUTPUT PATHS
# ========================
DEFAULT_OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.environ.get("BRATS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
NPZ_PATCH_PATH = os.environ.get("BRATS_NPZ_PATCH_PATH", "").strip() or None
os.environ.setdefault("BRATS_IMG_SIZE", "128")

for dir_path in [MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ========================
# SCAN ALL KAGGLE INPUTS
# ========================
print("=" * 60)
print("SCANNING ALL KAGGLE INPUTS FOR TRAINING DATA")
print("=" * 60)

def find_brats_training_data(base_path, max_depth=5):
    """
    Recursively find folders with BraTS training data (with segmentation).
    
    FIX: Log the path when OSError occurs, instead of silently swallowing.
    """
    results = []
    
    def search(path, depth):
        if depth == 0 or not os.path.isdir(path):
            return
        
        try:
            items = os.listdir(path)
        except OSError as e:
            # FIX: Include the path in the warning for debugging
            print(f"⚠ Cannot read directory {path}: {e}")
            return
        
        # Check if this folder has patient subfolders with seg files
        patient_folders = [d for d in items if os.path.isdir(os.path.join(path, d))]
        
        # Check first patient for seg file
        for pf in patient_folders[:5]:
            patient_path = os.path.join(path, pf)
            seg_files = glob(os.path.join(patient_path, "*seg*.nii*"))
            if seg_files:
                results.append({
                    'path': path,
                    'patients': len(patient_folders),
                    'has_seg': True
                })
                return
        
        # Recurse
        for item in items:
            search(os.path.join(path, item), depth - 1)
    
    search(base_path, max_depth)
    return results


def find_preprocessed_patch_data(base_path, max_depth=5):
    """
    Recursively find folders containing BraTS-style NPZ patch files.

    A dataset is considered patch-ready when a directory contains at least
    one .npz file with "patch" in filename.
    """
    results = []

    if not os.path.isdir(base_path):
        return results

    base_depth = base_path.rstrip("/").count("/")

    for root, dirs, files in os.walk(base_path):
        depth = root.rstrip("/").count("/") - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        npz_files = [f for f in files if f.lower().endswith(".npz")]
        if not npz_files:
            continue

        patch_files = [f for f in npz_files if "patch" in f.lower()]
        if patch_files:
            results.append(
                {
                    "path": root,
                    "npz_files": len(npz_files),
                    "patch_files": len(patch_files),
                }
            )

            # Stop descending further once a patch root is identified.
            dirs[:] = []

    return results

# Search in /kaggle/input
training_datasets = []

if os.path.exists("/kaggle/input"):
    print("\nSearching for datasets with segmentation masks...\n")
    
    for item in os.listdir("/kaggle/input"):
        base = f"/kaggle/input/{item}"
        print(f"Scanning: {base}")
        
        found = find_brats_training_data(base)
        if found:
            for f in found:
                print(f"  ✓ FOUND TRAINING DATA: {f['path']} ({f['patients']} patients)")
                training_datasets.append(f)
        else:
            # Check subfolders one more time
            if os.path.isdir(base):
                for sub in os.listdir(base):
                    subpath = os.path.join(base, sub)
                    found = find_brats_training_data(subpath)
                    for f in found:
                        print(f"  ✓ FOUND TRAINING DATA: {f['path']} ({f['patients']} patients)")
                        training_datasets.append(f)

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

if training_datasets:
    print(f"\n✓ Found {len(training_datasets)} dataset(s) with segmentation:\n")
    for i, ds in enumerate(training_datasets):
        print(f"  [{i+1}] {ds['path']}")
        print(f"      Patients: {ds['patients']}")
    
    # FIX: Guard against calling max() on empty list
    best = max(training_datasets, key=lambda x: x['patients'])
    TRAIN_PATH = best['path']
    os.environ["BRATS_DATASET_MODE"] = "nifti"
    os.environ["BRATS_DATASET_NAME"] = os.path.basename(TRAIN_PATH.rstrip("/")) or "BraTS"
    print(f"\n→ SELECTED: {TRAIN_PATH} ({best['patients']} patients)")
else:
    print("\n✗ NO TRAINING DATASETS WITH SEGMENTATION FOUND!")
    TRAIN_PATH = None

    patch_datasets = []
    if os.path.exists("/kaggle/input"):
        print("\nSearching for preprocessed NPZ patch datasets...\n")
        for item in os.listdir("/kaggle/input"):
            base = f"/kaggle/input/{item}"
            found = find_preprocessed_patch_data(base)
            for ds in found:
                print(
                    f"  ✓ FOUND PATCH DATA: {ds['path']} "
                    f"({ds['patch_files']} patch npz files)"
                )
                patch_datasets.append(ds)

    if patch_datasets:
        best_patch = max(patch_datasets, key=lambda x: x['patch_files'])
        NPZ_PATCH_PATH = best_patch['path']
        os.environ["BRATS_NPZ_PATCH_PATH"] = NPZ_PATCH_PATH
        os.environ["BRATS_DATASET_MODE"] = "npz_patches"
        os.environ["BRATS_DATASET_NAME"] = os.path.basename(NPZ_PATCH_PATH.rstrip("/"))
        print(
            "\n→ SELECTED PATCH DATASET: "
            f"{NPZ_PATCH_PATH} ({best_patch['patch_files']} patch files)"
        )
        print(
            "  Note: No raw seg NIfTI dataset found. "
            "Cell 5 should build HDF5 from NPZ patches."
        )
    else:
        print("\nYou need to add a dataset with ground truth segmentation masks.")
        print("The segmentation files must have 'seg' in the filename.")
        print("\nRecommended: 'brats2020-training-data' or 'brats-2021-task1'")

print("=" * 60)

# ========================
# ALSO CHECK FOR SPECIFIC PATHS
# ========================
_default_training_paths = [
    # BraTS 2021
    os.environ.get("BRATS_2021_PATH", "/kaggle/input/brats-2021-task1/BraTS2021_Training_Data"),
    "/kaggle/input/datasets/dschettler8845/brats-2021-task1/BraTS2021_Training_Data",
    # BraTS 2020 Training
    os.environ.get("BRATS_2020_PATH", "/kaggle/input/brats2020-training-data/MICCAI_BraTS2020_TrainingData"),
    "/kaggle/input/datasets/awsaf49/brats2020-training-data/MICCAI_BraTS2020_TrainingData",
    # BraTS 2019
    os.environ.get(
        "BRATS_2019_PATH",
        "/kaggle/input/brain-tumor-segmentation-brats-2019/MICCAI_BraTS_2019_Data_Training",
    ),
    "/kaggle/input/datasets/aryashah2k/brain-tumor-segmentation-brats-2019/MICCAI_BraTS_2019_Data_Training",
]
_env_training_paths = _read_env_path_list("BRATS_TRAINING_PATHS")
POSSIBLE_TRAINING_PATHS = _dedupe_keep_order([p for p in (_env_training_paths + _default_training_paths) if p])
_default_patch_paths = [
    os.environ.get("BRATS_2024_PATCH_PATH", "/kaggle/input/brats-2024-preprocessed-training-patches"),
    "/kaggle/input/datasets/prathamhanda10/brats-2024-preprocessed-training-patches",
]
_env_patch_paths = _read_env_path_list("BRATS_NPZ_PATCH_PATHS")
POSSIBLE_PATCH_PATHS = _dedupe_keep_order([p for p in (_env_patch_paths + _default_patch_paths) if p])

print("\nChecking known paths...")
for path in POSSIBLE_TRAINING_PATHS:
    if os.path.exists(path):
        # Check for seg files
        patients = glob(os.path.join(path, "*"))
        if patients:
            sample = patients[0]
            has_seg = len(glob(os.path.join(sample, "*seg*.nii*"))) > 0
            if has_seg:
                print(f"✓ {path} - HAS SEGMENTATION")
                if TRAIN_PATH is None:
                    TRAIN_PATH = path
                    os.environ["BRATS_DATASET_MODE"] = "nifti"
                    os.environ["BRATS_DATASET_NAME"] = os.path.basename(TRAIN_PATH.rstrip("/")) or "BraTS"
            else:
                print(f"✗ {path} - no segmentation (validation only)")

for path in POSSIBLE_PATCH_PATHS:
    if not os.path.exists(path):
        continue
    npz_files = glob(os.path.join(path, "**", "*.npz"), recursive=True)
    patch_files = [f for f in npz_files if "patch" in os.path.basename(f).lower()]
    if patch_files:
        print(f"✓ {path} - HAS PREPROCESSED NPZ PATCHES ({len(patch_files)} files)")
        if NPZ_PATCH_PATH is None:
            NPZ_PATCH_PATH = path
            os.environ["BRATS_NPZ_PATCH_PATH"] = NPZ_PATCH_PATH
            os.environ["BRATS_DATASET_MODE"] = "npz_patches"
            os.environ["BRATS_DATASET_NAME"] = os.path.basename(NPZ_PATCH_PATH.rstrip("/"))

print("=" * 60)

if TRAIN_PATH:
    print(f"\n✓ PRIMARY DATASET: {TRAIN_PATH}")
    print("  Mode: raw NIfTI + seg")
    os.environ["BRATS_DATASET_MODE"] = "nifti"
elif NPZ_PATCH_PATH:
    print(f"\n✓ PRIMARY DATASET: {NPZ_PATCH_PATH}")
    print("  Mode: preprocessed NPZ patches")
    print("  Next: run Cell 5 to convert NPZ patches to brats_preprocessed.h5")
    os.environ["BRATS_DATASET_MODE"] = "npz_patches"
    os.environ.setdefault("BRATS_DATASET_NAME", os.path.basename(NPZ_PATCH_PATH.rstrip("/")))
else:
    print("\n⚠ No valid training dataset found!")
    print("  Please add 'brats2020-training-data' or 'brats-2021-task1' to your notebook")

# ========================
# SMOKE TEST
# ========================
def test_dataset_paths():
    """Verify dataset path logic."""
    # Test with a non-existent path
    result = find_brats_training_data("/nonexistent/path")
    assert result == [], "Should return empty list for non-existent path"
    patch_result = find_preprocessed_patch_data("/nonexistent/path")
    assert patch_result == [], "Should return empty list for non-existent patch path"
    print("✓ Dataset path detection smoke test passed")

if __name__ == '__main__':
    test_dataset_paths()
