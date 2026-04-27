# ===================================================
# CELL 7C: Attention U-Net (FIXED)
# ===================================================
# FIXES:
# - Verification uses already-built model (no duplicate model build)
# - compare_architectures uses gc.collect() after cleanup

import gc
import os
import sys
import importlib.util


def _candidate_module_roots():
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

    return [p for i, p in enumerate(roots) if p and p not in roots[:i]]


def _ensure_module_path(module_filename):
    """Ensure local cell modules are importable in notebook/Kaggle execution contexts."""
    for base in _candidate_module_roots():
        if not base:
            continue
        module_path = os.path.join(base, module_filename)
        if os.path.exists(module_path) and base not in sys.path:
            sys.path.insert(0, base)


_ensure_module_path("cell_07a_building_blocks_FIXED.py")
_ensure_module_path("cell_07b_unet_FIXED.py")

REQUIRED_07A_SYMBOLS = [
    "Input", "Model", "conv_block", "encoder_block", "decoder_block", "output_layer",
    "INPUT_SHAPE", "FILTERS", "BOTTLENECK_FILTERS", "NUM_OUTPUT_CLASSES", "IMG_SIZE",
]


def _load_07a_symbols_into_globals():
    missing = [name for name in REQUIRED_07A_SYMBOLS if name not in globals()]
    if not missing:
        return

    try:
        from cell_07a_building_blocks_FIXED import (  # noqa: F401
            Input, Model, conv_block, encoder_block, decoder_block, output_layer,
            INPUT_SHAPE, FILTERS, BOTTLENECK_FILTERS, NUM_OUTPUT_CLASSES, IMG_SIZE,
        )
        globals().update({
            "Input": Input,
            "Model": Model,
            "conv_block": conv_block,
            "encoder_block": encoder_block,
            "decoder_block": decoder_block,
            "output_layer": output_layer,
            "INPUT_SHAPE": INPUT_SHAPE,
            "FILTERS": FILTERS,
            "BOTTLENECK_FILTERS": BOTTLENECK_FILTERS,
            "NUM_OUTPUT_CLASSES": NUM_OUTPUT_CLASSES,
            "IMG_SIZE": IMG_SIZE,
        })
        return
    except ModuleNotFoundError:
        pass

    for base in _candidate_module_roots():
        module_path = os.path.join(base, "cell_07a_building_blocks_FIXED.py")
        if not os.path.exists(module_path):
            continue
        spec = importlib.util.spec_from_file_location("cell_07a_building_blocks_FIXED", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in REQUIRED_07A_SYMBOLS:
                if hasattr(module, name):
                    globals()[name] = getattr(module, name)
            break

    still_missing = [name for name in REQUIRED_07A_SYMBOLS if name not in globals()]
    if still_missing:
        raise ModuleNotFoundError(
            "Could not resolve Cell 7A symbols. Missing: "
            + ", ".join(still_missing)
            + ". Run Cell 7A first or place cell_07a_building_blocks_FIXED.py in the working directory."
        )


def _load_build_unet_into_globals():
    if "build_unet" in globals():
        return

    try:
        from cell_07b_unet_FIXED import build_unet  # noqa: F401
        globals()["build_unet"] = build_unet
        return
    except ModuleNotFoundError:
        pass

    for base in _candidate_module_roots():
        module_path = os.path.join(base, "cell_07b_unet_FIXED.py")
        if not os.path.exists(module_path):
            continue
        spec = importlib.util.spec_from_file_location("cell_07b_unet_FIXED", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "build_unet"):
                globals()["build_unet"] = getattr(module, "build_unet")
                return

    raise ModuleNotFoundError(
        "Could not resolve build_unet from Cell 7B. Run Cell 7B first or place "
        "cell_07b_unet_FIXED.py in the working directory."
    )


_load_07a_symbols_into_globals()
_load_build_unet_into_globals()

Input = globals()["Input"]
Model = globals()["Model"]
conv_block = globals()["conv_block"]
encoder_block = globals()["encoder_block"]
decoder_block = globals()["decoder_block"]
output_layer = globals()["output_layer"]
INPUT_SHAPE = globals()["INPUT_SHAPE"]
FILTERS = globals()["FILTERS"]
BOTTLENECK_FILTERS = globals()["BOTTLENECK_FILTERS"]
NUM_OUTPUT_CLASSES = globals()["NUM_OUTPUT_CLASSES"]
IMG_SIZE = globals()["IMG_SIZE"]
build_unet = globals()["build_unet"]


def build_attention_unet(input_shape=None, filters=None, bottleneck_filters=None):
    if input_shape is None:
        input_shape = INPUT_SHAPE
    if filters is None:
        filters = FILTERS
    if bottleneck_filters is None:
        bottleneck_filters = BOTTLENECK_FILTERS

    inputs = Input(shape=input_shape, name='input')
    skip_connections = []
    x = inputs

    for i, f in enumerate(filters):
        skip, x = encoder_block(x, f, name_prefix=f'enc{i+1}')
        skip_connections.append(skip)

    x = conv_block(x, bottleneck_filters, name_prefix='bottleneck')

    for i, f in enumerate(reversed(filters)):
        skip = skip_connections[-(i+1)]
        x = decoder_block(x, skip, f, use_attention=True, name_prefix=f'dec{len(filters)-i}')

    outputs = output_layer(x, num_classes=1, name='output')
    return Model(inputs=inputs, outputs=outputs, name='AttentionUNet')


def verify_attention_unet(model):
    """FIX: Verify passed model instead of building a second model."""
    print('=' * 60)
    print('VERIFYING ATTENTION U-NET')
    print('=' * 60)
    print(f'Model name: {model.name}')
    print(f'Input shape: {model.input_shape}')
    print(f'Output shape: {model.output_shape}')
    print(f'Parameters: {model.count_params():,}')

    expected_input = (None, IMG_SIZE, IMG_SIZE, 4)
    expected_output = (None, IMG_SIZE, IMG_SIZE, 1)
    assert model.input_shape == expected_input, f'Input mismatch: {model.input_shape}'
    assert model.output_shape == expected_output, f'Output mismatch: {model.output_shape}'

    attn_layers = [l for l in model.layers if 'attn_multiply' in l.name]
    print(f'Attention gates: {len(attn_layers)}')
    print('✓ Attention U-Net verification passed')


def compare_architectures():
    unet = build_unet()
    attn = build_attention_unet()

    up = unet.count_params()
    ap = attn.count_params()
    gain = (ap - up) / up * 100 if up > 0 else 0.0

    print(f'U-Net params: {up:,}')
    print(f'Attention U-Net params: {ap:,}')
    print(f'Overhead: +{gain:.2f}%')

    # FIX: collect garbage after deleting temp model
    del unet
    gc.collect()

    return attn


ATTENTION_UNET_MODEL = build_attention_unet()
verify_attention_unet(ATTENTION_UNET_MODEL)
compare_architectures()
print('✓ Cell 7C fixed and ready.')
