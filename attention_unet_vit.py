# ===================================================
# CELL 07D: ATTENTION U-NET + VIT CLASSIFIER (FIXED)
# ===================================================
# Purpose:
# Reuse the Attention U-Net encoder backbone and feed the bottleneck
# feature map into a lightweight Vision Transformer for slice-level
# classification.

import importlib.util
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)


def _candidate_module_roots():
    """Collect likely project roots for notebook/script/Kaggle execution."""
    roots = []

    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))

    env_root = os.environ.get("BRATS_PROJECT_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    roots.extend([os.getcwd(), "/kaggle/working", "/kaggle/input", "/kaggle/input/datasets"])

    # Add one and two nested levels under common Kaggle mount points.
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
    """Ensure local cell modules are importable in notebook/Kaggle contexts."""
    for base in _candidate_module_roots():
        if not base:
            continue
        module_path = os.path.join(base, module_filename)
        if os.path.exists(module_path) and base not in sys.path:
            sys.path.insert(0, base)


_ensure_module_path("cell_07a_building_blocks_FIXED.py")

REQUIRED_07A_SYMBOLS = [
    "conv_block",
    "encoder_block",
    "INPUT_SHAPE",
    "FILTERS",
    "BOTTLENECK_FILTERS",
    "IMG_SIZE",
]


def _load_07a_symbols():
    # Notebook fallback: when Cell 7A was executed inline (not as importable module).
    missing_in_globals = [name for name in REQUIRED_07A_SYMBOLS if name not in globals()]
    if not missing_in_globals:
        return {name: globals()[name] for name in REQUIRED_07A_SYMBOLS}

    # Also check interactive __main__ namespace used by notebooks.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        resolved_main = {
            name: getattr(main_module, name)
            for name in REQUIRED_07A_SYMBOLS
            if hasattr(main_module, name)
        }
        missing_main = [name for name in REQUIRED_07A_SYMBOLS if name not in resolved_main]
        if not missing_main:
            return resolved_main

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from cell_07a_building_blocks_FIXED import (
                BOTTLENECK_FILTERS,
                FILTERS,
                IMG_SIZE,
                INPUT_SHAPE,
                conv_block,
                encoder_block,
            )

        return {
            "conv_block": conv_block,
            "encoder_block": encoder_block,
            "INPUT_SHAPE": INPUT_SHAPE,
            "FILTERS": FILTERS,
            "BOTTLENECK_FILTERS": BOTTLENECK_FILTERS,
            "IMG_SIZE": IMG_SIZE,
        }
    except ModuleNotFoundError:
        pass

    for base in _candidate_module_roots():
        module_path = os.path.join(base, "cell_07a_building_blocks_FIXED.py")
        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location("cell_07a_building_blocks_FIXED", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                spec.loader.exec_module(module)
            resolved = {}
            for name in REQUIRED_07A_SYMBOLS:
                if hasattr(module, name):
                    resolved[name] = getattr(module, name)
            missing = [name for name in REQUIRED_07A_SYMBOLS if name not in resolved]
            if not missing:
                return resolved

    raise ModuleNotFoundError(
        "Could not resolve Cell 7A symbols. Run Cell 7A first in the same notebook "
        "or place cell_07a_building_blocks_FIXED.py in the working directory "
        "(or set BRATS_PROJECT_ROOT to your project folder)."
    )


_07A = _load_07a_symbols()
conv_block = _07A["conv_block"]
encoder_block = _07A["encoder_block"]
INPUT_SHAPE = _07A["INPUT_SHAPE"]
FILTERS = _07A["FILTERS"]
BOTTLENECK_FILTERS = _07A["BOTTLENECK_FILTERS"]
IMG_SIZE = _07A["IMG_SIZE"]
NUM_CHANNELS = int(INPUT_SHAPE[-1])
NUM_CLASSES = int(os.environ.get("BRATS_NUM_CLASSES", "1"))


@tf.keras.utils.register_keras_serializable(package="Cell07D")
class ClassToken(Layer):
    """Prepend a learnable class token to a token sequence."""

    def __init__(self, initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        if embed_dim is None:
            raise ValueError("ClassToken requires a known embedding dimension.")
        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, int(embed_dim)),
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, tf.shape(inputs)[-1]])
        return tf.concat([class_token, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Cell07D")
class AddPositionEmbedding(Layer):
    """Add a learnable positional embedding to a sequence."""

    def __init__(self, initializer=None, **kwargs):
        super().__init__(**kwargs)
        if initializer is None:
            initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        seq_len = input_shape[1]
        embed_dim = input_shape[2]
        if seq_len is None or embed_dim is None:
            raise ValueError(
                "AddPositionEmbedding requires a known sequence length and embedding dimension."
            )
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, int(seq_len), int(embed_dim)),
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + tf.cast(self.pos_embedding, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initializer": tf.keras.initializers.serialize(self.initializer),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="Cell07D")
class TokenExtractor(Layer):
    """Extract one token from a sequence by index."""

    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[:, self.index, :]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


def get_attention_unet_vit_custom_objects():
    return {
        "ClassToken": ClassToken,
        "AddPositionEmbedding": AddPositionEmbedding,
        "TokenExtractor": TokenExtractor,
    }


def _transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout_rate=0.1, name=None):
    """One Transformer encoder block using pre-norm residual connections."""
    if int(num_heads) < 1:
        raise ValueError("num_heads must be >= 1.")
    if int(embed_dim) < 1:
        raise ValueError("embed_dim must be >= 1.")

    prefix = name or "transformer"
    head_dim = max(1, int(embed_dim) // int(num_heads))

    attn_input = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln1")(x)
    attn_output = MultiHeadAttention(
        num_heads=int(num_heads),
        key_dim=head_dim,
        dropout=float(dropout_rate),
        name=f"{prefix}_mha",
    )(attn_input, attn_input)
    attn_output = Dropout(float(dropout_rate), name=f"{prefix}_attn_drop")(attn_output)
    x = Add(name=f"{prefix}_attn_add")([x, attn_output])

    ff_input = LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln2")(x)
    ff_output = Dense(int(ff_dim), activation=tf.nn.gelu, name=f"{prefix}_ffn_dense1")(ff_input)
    ff_output = Dropout(float(dropout_rate), name=f"{prefix}_ffn_drop1")(ff_output)
    ff_output = Dense(int(embed_dim), name=f"{prefix}_ffn_dense2")(ff_output)
    ff_output = Dropout(float(dropout_rate), name=f"{prefix}_ffn_drop2")(ff_output)
    return Add(name=f"{prefix}_ffn_add")([x, ff_output])


def _vit_head(
    token_sequence,
    embed_dim,
    depth=4,
    num_heads=8,
    mlp_dim=512,
    dropout_rate=0.1,
    name="vit",
):
    """Stack of Transformer encoder blocks."""
    x = token_sequence
    for i in range(int(depth)):
        x = _transformer_encoder(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=mlp_dim,
            dropout_rate=dropout_rate,
            name=f"{name}_enc{i + 1}",
        )
    return LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)


def _resolve_num_patches(tensor, input_shape, filters):
    spatial_h = tensor.shape[1]
    spatial_w = tensor.shape[2]
    if spatial_h is not None and spatial_w is not None:
        return int(spatial_h) * int(spatial_w)

    input_h, input_w = input_shape[:2]
    downsample_factor = 2 ** len(filters)
    if input_h is None or input_w is None:
        raise ValueError(
            "build_attention_unet_vit requires static input height/width so the token "
            "sequence length can be determined for positional embeddings."
        )
    if input_h % downsample_factor != 0 or input_w % downsample_factor != 0:
        raise ValueError(
            f"Input spatial dimensions {input_shape[:2]} must be divisible by {downsample_factor}."
        )
    return int((input_h // downsample_factor) * (input_w // downsample_factor))


def build_attention_unet_vit(
    input_shape=INPUT_SHAPE,
    filters=None,
    bottleneck_filters=None,
    num_classes=NUM_CLASSES,
    vit_depth=4,
    vit_heads=8,
    vit_mlp_dim=512,
    vit_dropout=0.1,
    name="AttentionUNetViT",
):
    """Build an Attention U-Net encoder followed by a ViT classifier head."""
    if filters is None:
        filters = FILTERS
    if bottleneck_filters is None:
        bottleneck_filters = BOTTLENECK_FILTERS

    if int(num_classes) < 1:
        raise ValueError("num_classes must be >= 1.")
    if int(vit_depth) < 1:
        raise ValueError("vit_depth must be >= 1.")
    if int(vit_heads) < 1:
        raise ValueError("vit_heads must be >= 1.")
    if int(vit_mlp_dim) < 1:
        raise ValueError("vit_mlp_dim must be >= 1.")

    inputs = Input(shape=input_shape, name="input")
    x = inputs

    for i, channels in enumerate(filters):
        _, x = encoder_block(x, int(channels), name_prefix=f"enc{i + 1}")

    x = conv_block(x, int(bottleneck_filters), name_prefix="bottleneck")

    num_patches = _resolve_num_patches(x, input_shape, filters)
    embed_dim = int(bottleneck_filters)

    token_seq = Reshape((num_patches, embed_dim), name="tokens")(x)
    token_seq = ClassToken(name="class_token")(token_seq)
    token_seq = AddPositionEmbedding(name="pos_embedding")(token_seq)
    token_seq = _vit_head(
        token_seq,
        embed_dim=embed_dim,
        depth=vit_depth,
        num_heads=vit_heads,
        mlp_dim=vit_mlp_dim,
        dropout_rate=vit_dropout,
        name="vit",
    )

    class_output = TokenExtractor(index=0, name="cls_token")(token_seq)

    if int(num_classes) == 1:
        outputs = Dense(1, activation="sigmoid", dtype="float32", name="pred")(class_output)
    else:
        outputs = Dense(int(num_classes), activation="softmax", dtype="float32", name="pred")(class_output)

    return Model(inputs=inputs, outputs=outputs, name=name)


def verify_attention_unet_vit(model, num_classes=NUM_CLASSES):
    """Basic structural verification for the classifier model."""
    expected_input = (None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    expected_output = (None, 1 if int(num_classes) == 1 else int(num_classes))

    print("=" * 60)
    print("VERIFYING ATTENTION U-NET + VIT CLASSIFIER")
    print("=" * 60)
    print(f"Model name: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")

    assert model.input_shape == expected_input, f"Input mismatch: {model.input_shape}"
    assert model.output_shape == expected_output, f"Output mismatch: {model.output_shape}"
    print("[OK] Attention U-Net + ViT verification passed")


def _run_smoke_test():
    """Validate build, forward pass, save, and reload."""
    import tempfile

    print("\n=== ATTENTION U-NET + VIT CLASSIFIER SMOKE TEST ===")
    model = build_attention_unet_vit(num_classes=2)
    verify_attention_unet_vit(model, num_classes=2)

    dummy_x = tf.random.normal((2, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    pred = model(dummy_x, training=False)
    print(f"Dummy forward-pass shape: {pred.shape}")
    assert tuple(pred.shape) == (2, 2), f"Unexpected prediction shape: {pred.shape}"

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "attention_unet_vit.keras")
        model.save(model_path)
        reloaded = tf.keras.models.load_model(
            model_path,
            custom_objects=get_attention_unet_vit_custom_objects(),
            compile=False,
        )
        reloaded_pred = reloaded(dummy_x, training=False)
        assert tuple(reloaded_pred.shape) == (2, 2), f"Reloaded shape mismatch: {reloaded_pred.shape}"

    print("[OK] Smoke test passed.\n")


if __name__ == "__main__":
    _run_smoke_test()
