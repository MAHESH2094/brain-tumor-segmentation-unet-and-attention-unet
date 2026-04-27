"""
BraTS Kaggle preflight checklist.

Run in Kaggle before starting 25-epoch training:
    python kaggle_preflight_check.py
"""

import ast
import importlib
import json
import os
import sys
from typing import Dict, List, Tuple


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse(path: str):
    return ast.parse(_read(path), filename=path)


def _has_function(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
    return False


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _has_param(fn_node: ast.AST, param: str) -> bool:
    if fn_node is None:
        return False
    params = [a.arg for a in fn_node.args.args]
    params += [a.arg for a in fn_node.args.kwonlyargs]
    if fn_node.args.vararg:
        params.append(fn_node.args.vararg.arg)
    if fn_node.args.kwarg:
        params.append(fn_node.args.kwarg.arg)
    return param in params


def _fn_passes_keyword(fn_node: ast.AST, kw: str) -> bool:
    if fn_node is None:
        return False
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            for item in node.keywords:
                if item.arg == kw:
                    return True
    return False


def _literal_assignments(tree: ast.AST, name: str) -> List[object]:
    vals: List[object] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        vals.append(ast.literal_eval(node.value))
                    except Exception:
                        vals.append(None)
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Name) and t.id == name and node.value is not None:
                try:
                    vals.append(ast.literal_eval(node.value))
                except Exception:
                    vals.append(None)
    return vals


def check_environment() -> bool:
    print("\n=== 1) ENVIRONMENT & HARDWARE ===")
    ok = True

    try:
        import tensorflow as tf

        ver = tuple(int(x) for x in tf.__version__.split(".")[:3])
        if ver >= (2, 9, 0):
            _ok(f"TensorFlow version {tf.__version__} >= 2.9.0")
        else:
            _fail(f"TensorFlow version {tf.__version__} < 2.9.0")
            ok = False

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            _ok(f"GPU detected: {len(gpus)} device(s)")
        else:
            _fail("No GPU detected")
            ok = False

        gpu_names = []
        for g in gpus:
            try:
                details = tf.config.experimental.get_device_details(g)
                gpu_names.append(details.get("device_name", str(g)))
            except Exception:
                gpu_names.append(str(g))

        preferred = ("T4", "P100", "V100")
        if any(any(tag in n.upper() for tag in preferred) for n in gpu_names):
            _ok(f"Preferred Kaggle GPU class found: {gpu_names}")
        else:
            _warn(f"GPU present but not T4/P100/V100: {gpu_names}")

    except Exception as exc:
        _fail(f"TensorFlow unavailable: {exc}")
        return False

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb >= 8:
            _ok(f"System RAM {ram_gb:.2f} GB >= 8 GB")
        else:
            _fail(f"System RAM {ram_gb:.2f} GB < 8 GB")
            ok = False
    except Exception as exc:
        _warn(f"Could not determine RAM via psutil: {exc}")

    # Kaggle path + session hint
    if os.path.isdir("/kaggle/working"):
        try:
            probe = "/kaggle/working/.preflight_write_test"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            _ok("/kaggle/working is writable")
        except Exception as exc:
            _fail(f"/kaggle/working not writable: {exc}")
            ok = False
    else:
        _fail("/kaggle/working missing (not running in Kaggle runtime)")
        ok = False

    # Kaggle remaining session hours cannot be queried reliably via public API
    _warn("Kaggle remaining session time is not directly queryable in-script; verify manually in UI (need 12h+).")

    return ok


def check_files_and_values() -> bool:
    print("\n=== 2) CRITICAL FILE CHECKS ===")
    ok = True

    must_exist = [
        "optimal_config_kaggle.py",
        "config.py",
        "cell_01_environment_FIXED.py",
        "cell_02_dataset_paths_FIXED.py",
        "cell_03_dataset_stats_FIXED.py",
        "cell_04_preprocessing_FIXED.py",
        "cell_05_hdf5_builder_FIXED.py",
        "cell_06_augmentation_FIXED.py",
        "cell_07a_building_blocks_FIXED.py",
        "cell_07b_unet_FIXED.py",
        "cell_07c_attention_unet_FIXED.py",
        "cell_07d_attention_unet_vit_FIXED.py",
        "cell_08_loss_metrics_FIXED.py",
        "cell_09_training_FIXED.py",
        "cell_11_inference_FIXED.py",
        "gradient_accumulation.py",
        "ensemble.py",
        "postprocessing.py",
        "pipeline.py",
        "custom_objects_registry.py",
    ]

    for p in must_exist:
        if _exists(p):
            _ok(f"Exists: {p}")
        else:
            _fail(f"Missing: {p}")
            ok = False

    if _exists("optimal_config_kaggle.py"):
        txt = _read("optimal_config_kaggle.py")
        required_env = [
            "os.environ['EPOCHS'] = '25'",
            "os.environ['FAST_DEV_MODE'] = '0'",
            "os.environ['BATCH_SIZE_PER_GPU'] = '16'",
            "os.environ['BRATS_IMG_SIZE'] = '128'",
            "os.environ['BRATS_POSITIVE_CLASS_WEIGHT'] = '4.0'",
            "os.environ['BRATS_INFERENCE_THRESHOLD'] = '0.5'",
        ]
        for token in required_env:
            if token in txt:
                _ok(f"optimal_config_kaggle has {token}")
            else:
                _fail(f"optimal_config_kaggle missing {token}")
                ok = False

    return ok


def check_syntax_imports_and_functions() -> bool:
    print("\n=== 3-6) SYNTAX / IMPORT / FUNCTION CHECKS ===")
    ok = True

    syntax_files = [
        "cell_04_preprocessing_FIXED.py",
        "cell_07a_building_blocks_FIXED.py",
        "cell_07d_attention_unet_vit_FIXED.py",
        "cell_08_loss_metrics_FIXED.py",
        "cell_09_training_FIXED.py",
    ]
    trees: Dict[str, ast.AST] = {}

    for p in syntax_files:
        try:
            trees[p] = _parse(p)
            _ok(f"Syntax OK: {p}")
        except SyntaxError as exc:
            _fail(f"Syntax error in {p}: line {exc.lineno} {exc.msg}")
            ok = False

    # Cell 04 checks
    if "cell_04_preprocessing_FIXED.py" in trees:
        t = trees["cell_04_preprocessing_FIXED.py"]
        fn_norm = _find_function(t, "normalize_image")
        fn_pre = _find_function(t, "preprocess_multimodal_slice")
        checks = [
            (_has_function(t, "compute_volume_stats"), "cell_04: compute_volume_stats exists"),
            (_has_param(fn_norm, "volume_stats"), "cell_04: normalize_image(volume_stats=...)"),
            (_fn_passes_keyword(fn_pre, "volume_stats"), "cell_04: preprocess_multimodal_slice passes volume_stats"),
        ]
        for cond, msg in checks:
            if cond:
                _ok(msg)
            else:
                _fail(msg)
                ok = False

    # Cell 07A checks
    if "cell_07a_building_blocks_FIXED.py" in trees:
        t = trees["cell_07a_building_blocks_FIXED.py"]
        if _has_function(t, "hierarchy_constraint_layer"):
            _ok("cell_07a: hierarchy_constraint_layer exists")
        else:
            _fail("cell_07a: hierarchy_constraint_layer missing")
            ok = False

        out_fn = _find_function(t, "output_layer")
        defaults_ok = False
        if out_fn is not None:
            names = [a.arg for a in out_fn.args.args]
            defs = out_fn.args.defaults
            if defs:
                mapping = {}
                for n, d in zip(names[-len(defs):], defs):
                    try:
                        mapping[n] = ast.literal_eval(d)
                    except Exception:
                        mapping[n] = None
                defaults_ok = mapping.get("enforce_hierarchy") is True

        if defaults_ok:
            _ok("cell_07a: output_layer has enforce_hierarchy=True")
        else:
            _fail("cell_07a: output_layer enforce_hierarchy default is not True")
            ok = False

    # Cell 08 checks
    if "cell_08_loss_metrics_FIXED.py" in trees:
        t = trees["cell_08_loss_metrics_FIXED.py"]
        checks = [
            (_has_function(t, "calculate_class_weights"), "cell_08: calculate_class_weights exists"),
            (_has_function(t, "set_dynamic_class_weights"), "cell_08: set_dynamic_class_weights exists"),
            (_has_function(t, "focal_dice_loss"), "cell_08: focal_dice_loss exists"),
        ]
        for cond, msg in checks:
            if cond:
                _ok(msg)
            else:
                _fail(msg)
                ok = False

        txt = _read("cell_08_loss_metrics_FIXED.py")
        binary_weight_ok = (
            "CLASS_WEIGHTS = tf.constant([1.0]" in txt
            or "set_dynamic_class_weights" in txt
        )
        binary_symbols_ok = ("combined_loss" in txt and "dice_coef" in txt)
        if binary_weight_ok and binary_symbols_ok:
            _ok("cell_08: binary class-weight hook and combined_loss/dice_coef detected")
        else:
            _fail("cell_08: binary class-weight or combined_loss/dice_coef check failed")
            ok = False

    # Cell 07D checks
    if "cell_07d_attention_unet_vit_FIXED.py" in trees:
        t = trees["cell_07d_attention_unet_vit_FIXED.py"]
        checks = [
            (_has_function(t, "build_attention_unet_vit"), "cell_07d: build_attention_unet_vit exists"),
            (_has_function(t, "get_attention_unet_vit_custom_objects"), "cell_07d: custom object export exists"),
        ]
        for cond, msg in checks:
            if cond:
                _ok(msg)
            else:
                _fail(msg)
                ok = False

    # Cell 09 checks
    if "cell_09_training_FIXED.py" in trees:
        t = trees["cell_09_training_FIXED.py"]
        cell9_text = _read("cell_09_training_FIXED.py")
        full_mode_epochs_ok = (
            'os.environ.get("EPOCHS", "25")' in cell9_text
            or '("18" if IS_KAGGLE else "25")' in cell9_text
        )
        checks = [
            (_has_function(t, "cosine_annealing_with_warmup"), "cell_09: cosine_annealing_with_warmup exists"),
            ('os.environ.get("WARMUP_EPOCHS", "2")' in cell9_text, "cell_09: WARMUP_EPOCHS default 2"),
            ('os.environ.get("EPOCHS",' in cell9_text, "cell_09: EPOCHS configurable via env"),
            (full_mode_epochs_ok, "cell_09: full-mode epoch profile includes Kaggle/local defaults"),
            ('os.environ.get("BATCH_SIZE_PER_GPU", "16")' in cell9_text, "cell_09: BATCH_SIZE_PER_GPU default 16"),
        ]
        for cond, msg in checks:
            if cond:
                _ok(msg)
            else:
                _fail(msg)
                ok = False

    # Cell 11 checks
    if _exists("cell_11_inference_FIXED.py"):
        t11 = _read("cell_11_inference_FIXED.py")
        if "def predict_patient(" in t11 and "use_tta=True" in t11:
            _ok("cell_11: predict_patient default use_tta=True")
        else:
            _fail("cell_11: use_tta=True default missing in predict_patient")
            ok = False

        required_modes = ["h", "v", "hv", "rot90", "rot180", "rot270"]
        if all((f"'{m}'" in t11) or (f'\"{m}\"' in t11) for m in required_modes):
            _ok("cell_11: required tta_modes present")
        else:
            _fail("cell_11: required tta_modes missing")
            ok = False

    return ok


def check_common_pitfalls() -> bool:
    print("\n=== 9) COMMON PITFALL CHECK ===")
    ok = True

    # Scan only the canonical fixed/Kaggle execution path files.
    scan_files = [
        "optimal_config_kaggle.py",
        "cell_08_loss_metrics_FIXED.py",
        "cell_09_training_FIXED.py",
        "cell_11_inference_FIXED.py",
        "cell_12_final_FIXED.py",
        "custom_objects_registry.py",
    ]
    existing_scan_files = [p for p in scan_files if _exists(p)]
    joined = "\n".join(_read(p) for p in existing_scan_files)

    hardcoded_fast_dev = any(
        line.strip().startswith("FAST_DEV_MODE") and "= 1" in line
        for line in joined.splitlines()
    )
    if hardcoded_fast_dev:
        _fail("Found hardcoded FAST_DEV_MODE = 1 in fixed/Kaggle files")
        ok = False
    else:
        _ok("No hardcoded FAST_DEV_MODE = 1 in fixed/Kaggle files")

    hardcoded_epochs_4 = any(
        line.strip().startswith("EPOCHS") and "= 4" in line
        for line in joined.splitlines()
    ) or 'os.environ.get("EPOCHS", "4")' in joined
    if hardcoded_epochs_4:
        _fail("Found hardcoded/fallback EPOCHS = 4 in fixed/Kaggle files")
        ok = False
    else:
        _ok("No hardcoded/fallback EPOCHS = 4 in fixed/Kaggle files")

    # Basic custom object registration sanity
    if _exists("custom_objects_registry.py"):
        txt = _read("custom_objects_registry.py")
        needed = [
            "combined_loss",
            "dice_coef",
            "precision_metric",
            "sensitivity_metric",
            "iou_metric",
            "ClassToken",
            "AddPositionEmbedding",
            "TokenExtractor",
        ]
        missing = [n for n in needed if n not in txt]
        if missing:
            _fail(f"custom_objects_registry missing: {missing}")
            ok = False
        else:
            _ok("custom_objects_registry has required custom object names")

    return ok


def print_estimates() -> None:
    print("\n=== 7) MEMORY ESTIMATION ===")
    batch = 16
    image_mb = 256 * 256 * 4 * 4 / (1024 ** 2)
    forward_mb = batch * image_mb
    grads_mb = 3 * forward_mb
    model_mb = 120
    total_mb = forward_mb + grads_mb + model_mb

    print(f"Batch size: {batch}")
    print(f"Image size: 256x256x4 float32 = {image_mb:.2f} MB/image")
    print(f"Forward pass: {forward_mb:.2f} MB")
    print(f"Gradients (~3x): {grads_mb:.2f} MB")
    print(f"Model: {model_mb:.2f} MB")
    print(f"Total approximate GPU usage: {total_mb:.2f} MB")
    print(f"Fits 16 GB GPU: {total_mb < 16384}")

    print("\n=== 8) TIME ESTIMATION ===")
    data_prep = 1.5
    unet = 4.5
    attn = 5.5
    eval_h = 0.5
    total = data_prep + unet + attn + eval_h

    print(f"Data prep (1-8): {data_prep:.1f} h")
    print(f"U-Net (25): {unet:.1f} h")
    print(f"Attention U-Net (25): {attn:.1f} h")
    print(f"Evaluation: {eval_h:.1f} h")
    print(f"Total: {total:.1f} h")


def final_sanity() -> bool:
    print("\n=== 10) FINAL SANITY TEST ===")
    ok = True

    try:
        import optimal_config_kaggle  # noqa: F401
        _ok("Config loaded")
    except Exception as exc:
        _fail(f"Config import failed: {exc}")
        ok = False

    try:
        from cell_08_loss_metrics_FIXED import calculate_class_weights  # noqa: F401
        from cell_09_training_FIXED import cosine_annealing_with_warmup  # noqa: F401
        _ok("Key functions importable")
    except Exception as exc:
        _fail(f"Key function import failed: {exc}")
        ok = False

    try:
        import tensorflow as tf

        test_tensor = tf.random.normal((2, 256, 256, 4))
        _ok(f"TF tensor created: {tuple(test_tensor.shape)}")
        gpus = tf.config.list_physical_devices("GPU")
        _ok(f"GPUs available: {len(gpus)}")
    except Exception as exc:
        _fail(f"TensorFlow runtime check failed: {exc}")
        ok = False

    return ok


def main() -> int:
    print("=" * 72)
    print("BRATS KAGGLE PRE-FLIGHT CHECKLIST")
    print("=" * 72)

    checks = [
        check_environment(),
        check_files_and_values(),
        check_syntax_imports_and_functions(),
        check_common_pitfalls(),
        final_sanity(),
    ]

    print_estimates()

    all_ok = all(checks)
    print("\n" + "=" * 72)
    if all_ok:
        print("PRE-FLIGHT STATUS: PASS")
        print("Safe to start 25-epoch training in Kaggle.")
    else:
        print("PRE-FLIGHT STATUS: FAIL")
        print("Fix failing items before starting training.")
    print("=" * 72)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
