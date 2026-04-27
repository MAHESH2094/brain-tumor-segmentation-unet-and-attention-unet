from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    Multiply,
)
from tensorflow.keras.models import Model


def conv_block(x, filters, name, dropout=0.2):
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{name}_conv1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_act1")(x)
    if dropout > 0:
        x = Dropout(dropout, name=f"{name}_drop1")(x)

    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", name=f"{name}_conv2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_act2")(x)
    if dropout > 0:
        x = Dropout(dropout, name=f"{name}_drop2")(x)
    return x


def attention_gate(x, g, filters, name):
    theta_x = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal", name=f"{name}_theta")(x)
    phi_g = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal", name=f"{name}_phi")(g)
    f = Add(name=f"{name}_add")([theta_x, phi_g])
    f = Activation("relu", name=f"{name}_relu")(f)
    psi = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", name=f"{name}_psi")(f)
    alpha = Activation("sigmoid", name=f"{name}_sigmoid")(psi)
    return Multiply(name=f"{name}_mul")([x, alpha])


def build_unet(img_size=128, in_channels=4, num_classes=1, attention=False):
    filters = [64, 128, 256, 512]
    inputs = Input(shape=(img_size, img_size, in_channels), name="input")

    skips = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, name=f"enc{i+1}")
        skips.append(x)
        x = MaxPooling2D((2, 2), name=f"enc{i+1}_pool")(x)

    x = conv_block(x, 512, name="bottleneck")

    for i, f in enumerate(reversed(filters)):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding="same", name=f"dec{i+1}_up")(x)
        skip = skips[-(i + 1)]
        if attention:
            skip = attention_gate(skip, x, f, name=f"dec{i+1}_attn")
        x = Concatenate(name=f"dec{i+1}_cat")([x, skip])
        x = conv_block(x, f, name=f"dec{i+1}")

    logits = Conv2D(num_classes, (1, 1), padding="same", kernel_initializer="he_normal", name="output_conv")(x)
    outputs = Activation("sigmoid", dtype="float32", name="output_sigmoid")(logits)
    model_name = "AttentionUNet" if attention else "UNet"
    return Model(inputs=inputs, outputs=outputs, name=model_name)


def build_attention_unet_vit(
    img_size=128,
    in_channels=4,
    num_classes=1,
    vit_depth=4,
    vit_heads=8,
    vit_mlp_dim=512,
    vit_dropout=0.1,
):
    """Build AttentionUNetViT via Cell 07D implementation."""
    try:
        from cell_07d_attention_unet_vit_FIXED import build_attention_unet_vit as _build_attention_unet_vit
    except Exception as exc:
        raise RuntimeError(
            "Could not import build_attention_unet_vit from cell_07d_attention_unet_vit_FIXED.py"
        ) from exc

    return _build_attention_unet_vit(
        input_shape=(int(img_size), int(img_size), int(in_channels)),
        num_classes=int(num_classes),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_dim=int(vit_mlp_dim),
        vit_dropout=float(vit_dropout),
    )

