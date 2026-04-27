# ===================================================
# CELL 7A: Common Architecture Building Blocks (FIXED)
# ===================================================
# FIXES:
# - Dropout only after second conv (less aggressive)
# - Attention gate spatial-alignment assertion/comment
# - Final sigmoid output forced to float32 for mixed-precision stability

import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Conv2DTranspose, Concatenate, Multiply, Add, Dropout, Lambda
)
import tensorflow as tf

try:
    from tensorflow.keras.layers import GroupNormalization
except Exception:
    GroupNormalization = None

IMG_SIZE = int(os.environ.get("BRATS_IMG_SIZE", str(globals().get("IMG_SIZE", "128"))))

NUM_INPUT_CHANNELS = 4
NUM_OUTPUT_CLASSES = 1
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, NUM_INPUT_CHANNELS)
FILTERS = [64, 128, 256, 512]
BOTTLENECK_FILTERS = 512
KERNEL_SIZE = (3, 3)
ACTIVATION = 'relu'
USE_BATCH_NORM = True
USE_GROUP_NORM = os.environ.get("BRATS_USE_GROUP_NORM", "0") == "1"
GROUP_NORM_MAX_GROUPS = int(os.environ.get("BRATS_GROUP_NORM_GROUPS", "8"))
DROPOUT_RATE = 0.2


def _normalization_layer(channels, name):
    """Return BatchNorm or safe GroupNorm depending on runtime configuration."""
    if USE_GROUP_NORM and GroupNormalization is not None:
        groups = int(max(1, min(GROUP_NORM_MAX_GROUPS, channels)))
        while groups > 1 and (channels % groups) != 0:
            groups -= 1
        return GroupNormalization(groups=groups, epsilon=1e-5, name=name)
    return BatchNormalization(name=name)


def conv_block(inputs, filters, kernel_size=KERNEL_SIZE, use_bn=USE_BATCH_NORM, dropout_rate=DROPOUT_RATE, name_prefix=''):
    """Double conv block with optional single dropout after second conv.

    FIX: apply dropout once (after conv2), not after both convolutions.
    """
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_conv1')(inputs)
    if use_bn:
        x = _normalization_layer(filters, name=f'{name_prefix}_bn1')(x)
    x = Activation(ACTIVATION, name=f'{name_prefix}_act1')(x)

    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_conv2')(x)
    if use_bn:
        x = _normalization_layer(filters, name=f'{name_prefix}_bn2')(x)
    x = Activation(ACTIVATION, name=f'{name_prefix}_act2')(x)

    if dropout_rate and dropout_rate > 0:
        x = Dropout(dropout_rate, name=f'{name_prefix}_drop')(x)
    return x


def encoder_block(inputs, filters, name_prefix=''):
    conv = conv_block(inputs, filters, name_prefix=f'{name_prefix}_enc')
    pool = MaxPooling2D(pool_size=(2, 2), name=f'{name_prefix}_pool')(conv)
    return conv, pool


def attention_gate(x, g, inter_channels, name_prefix=''):
    """Attention gate with channel projection on both inputs.

    Expected usage has matched spatial dimensions between x and g
    (as in decoder_block). This block aligns channels via 1x1 convs;
    it does not perform spatial resizing.
    """
    # Transform x and g to same channel dimension
    theta_x = Conv2D(inter_channels, (1, 1), padding='same',
                     kernel_initializer='he_normal',
                     name=f'{name_prefix}_theta')(x)
    phi_g = Conv2D(inter_channels, (1, 1), padding='same',
                   kernel_initializer='he_normal',
                   name=f'{name_prefix}_phi')(g)

    # Spatial dimensions must already match for the elementwise add.
    f = Add(name=f'{name_prefix}_add')([theta_x, phi_g])
    f = Activation('relu', name=f'{name_prefix}_relu')(f)

    # Attention coefficients
    psi = Conv2D(1, (1, 1), padding='same',
                 kernel_initializer='he_normal',
                 name=f'{name_prefix}_psi')(f)
    alpha = Activation('sigmoid', name=f'{name_prefix}_sigmoid')(psi)

    # Apply attention
    return Multiply(name=f'{name_prefix}_multiply')([x, alpha])


def decoder_block(inputs, skip_connection, filters, use_attention=False, name_prefix=''):
    up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', name=f'{name_prefix}_upconv')(inputs)
    if use_attention:
        skip_connection = attention_gate(skip_connection, up, filters, name_prefix=f'{name_prefix}_attn')
    concat = Concatenate(name=f'{name_prefix}_concat')([up, skip_connection])
    return conv_block(concat, filters, name_prefix=f'{name_prefix}_dec')


def _hierarchy_sort(logits):
    """Apply descending channel sort for optional compatibility ordering.

    Higher logit leads to higher sigmoid probability.
    In binary mode this path is a no-op because only one channel is produced.
    """
    # logits: (..., C) when compatibility ordering is enabled.
    sorted_logits = tf.sort(logits, axis=-1, direction='DESCENDING')
    # After sort: index 0 is highest confidence.
    return sorted_logits


def hierarchy_constraint_layer(logits):
    """Backward-compatible alias used by downstream loaders/registries."""
    return _hierarchy_sort(logits)


def output_layer(inputs, num_classes=NUM_OUTPUT_CLASSES, name='output', enforce_hierarchy=True):
    """Final output layer.

    Binary segmentation uses a single sigmoid channel by default.
    Optional ordering is only relevant if called with multiple channels.

    FIX: dtype='float32' is critical for mixed precision stability and must remain.
    """
    x = Conv2D(num_classes, (1, 1), padding='same', kernel_initializer='he_normal', name=f'{name}_conv')(inputs)

    if enforce_hierarchy and num_classes > 1:
        x = Lambda(hierarchy_constraint_layer, name=f'{name}_hierarchy')(x)

    return Activation('sigmoid', dtype='float32', name=f'{name}_sigmoid')(x)


print('✓ Cell 7A fixed and ready (binary output layer).')
