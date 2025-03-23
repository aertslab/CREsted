"""Borzoi model architecture."""

from __future__ import annotations

import collections.abc as cabc

import keras
import numpy as np

from crested.tl.zoo.utils import (
    activate,
    conv_block_bs,
    ffn_block_enf,
    mha_block_enf,
    pool,
)


def borzoi(
    seq_len: int,
    num_classes: int | cabc.Sequence[int],
    num_conv_blocks: int = 6,
    num_transformer_blocks: int = 8,
    num_transformer_heads: int = 8,
    target_length: int = 6144,
    start_filters: int = 512,
    tower_start_filters: int | None = None,
    filters: int = 1536,
    pointwise_filters: int | None = 1920,
    unet_connections: cabc.Sequence[int] = [5, 6],
    unet_filters: int = 1536,
    upsample_conv: bool = True,
    conv_activation: str = "gelu_approx",
    transformer_activation: str = "relu",
    output_activation: str = "softplus",
    pool_type: str = "max",
    first_kernel_size: int = 15,
    kernel_size: int = 5,
    transformer_dropout=0.2,
    pointwise_dropout: float = 0.1,
    bn_sync: bool = False,
    name: str = "Borzoi",
) -> keras.Model:
    """
    Construct an fully replicated Borzoi model.

    Note that unlike other CREsted model zoo architectures, this architecture is not suited for
    predicting individual regions out of the box.

    Parameters
    ----------
    seq_len
        Width of the input region.
        Borzoi default is 524288.
    num_classes
        Number of classes to predict.
        If an int, creates a single head with num_classes classes.
        If a list of integers, creates multiple heads in a list.
    num_conv_blocks
        Number of convolution blocks to include in the tower, after the stem block.
    num_transformer_blocks
        Number of transformer blocks to include in the transformer stack.
    target_length
        The target length in bins to crop to. Default is 6144, cropping away 5120 bins (164kb) on each side.
    start_filters
        Starting number of filters for the first DNA-facing block, exponentially increasing towards `filters` through the conv tower.
    tower_start_filters
        Optional: Different number of filters to start the conv tower with after the stem conv.
        By default, inferred starting from start_filters to filters.
    filters
        Number of filters at the end of the conv tower and in the upsampling.
    pointwise_filters
        Number of filters of the post-transformer/upsampling final pointwise convolution.
        If None, block is not included.
    unet_connections
        Levels in the convolution tower to add U-net skip connections past the transformer tower.
        1-indexed, so [5, 6] means after the 5th and 6th block.
    unet_filters
        Number of filters to use for the U-net connection skip blocks.
    upsample_conv
        Whether to include an extra convolution during the upsampling.
        Used in release Borzoi, removed later.
    conv_activation
        Activation function to use in the conv tower, in the upsampling, and in the final pointwise block.
    transformer_activation
        Activation function to use in the feedforward section of the transformer blocks.
    output_activation
        Final activation to use on the output heads, just before predicting the tracks.
    pool_type
        What kind of pooling type to use, one of 'max' or 'attention'.
    first_kernel_size
        Kernel size of the first conv layer, directly interfacing the sequence.
    kernel_size
        Kernel size of the convolutions in the conv tower.
    transformer_dropout
        Dropout rate used in the transformer blocks, both MHA and feed-forward.
    pointwise_dropout
        Dropout rate of the post-transformer final pointwise layer.
    bn_sync
        Whether to use synchronized cross-GPU BatchNormalisations.
        Default is False to preserve TensorFlow-PyTorch compatibility.

    Returns
    -------
    A Keras model.
    """
    if any(
        unet_connections_i > num_conv_blocks for unet_connections_i in unet_connections
    ):
        raise ValueError(
            f"U-net connections requested at levels ({unet_connections}) past the current conv tower size ({num_conv_blocks})"
        )
    upsampling_out_length = int(
        seq_len / (2 ** (num_conv_blocks - len(unet_connections) + 1))
    )  # Should be length after upsampling, so seq_len/32 for base Borzoi
    crop_length = int((upsampling_out_length - target_length) // 2)

    # Sequence input
    sequence = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Build stem (standard conv + pooling)
    current = keras.layers.Conv1D(
        filters=start_filters,
        kernel_size=first_kernel_size,
        padding="same",
        name="stem_conv",
    )(sequence)
    current = pool(
        current, pool_type=pool_type, pool_size=2, padding="same", name="stem_pool"
    )

    # Build convolution tower
    # Each block: (batchnorm + gelu + conv)
    # In enformer: stem has `start_filters`` filters, first layer of tower also has `start_filters` filters -> start exp_linspace_int at tower
    # In borzoi: stem has `start_filters` filters, first layer of tower already increases -> start exp_linspace_int at stem
    if tower_start_filters is not None:
        tower_filters = [start_filters] + exp_linspace_int(
            start=tower_start_filters,
            end=filters,
            num_modules=num_conv_blocks,
            divisible_by=32,
        )
    else:
        tower_filters = exp_linspace_int(
            start=start_filters,
            end=filters,
            num_modules=num_conv_blocks + 1,
            divisible_by=32,
        )
    unet_skips = []
    for cidx, layer_filters in enumerate(tower_filters[1:]):
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=conv_activation,
            residual=False,
            l2_scale=0,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_sync=bn_sync,
            bn_epsilon=1e-3,
            kernel_initializer="he_normal",
            name_prefix=f"tower_conv_{cidx+1}",
        )
        if cidx + 1 in unet_connections:
            unet_skips.append(
                conv_block_bs(
                    current,
                    filters=unet_filters,
                    kernel_size=1,
                    pool_size=1,
                    batch_norm=True,
                    activation=conv_activation,
                    residual=False,
                    l2_scale=0,
                    bn_momentum=0.9,
                    bn_gamma=None,
                    bn_sync=bn_sync,
                    bn_epsilon=1e-3,
                    kernel_initializer="he_normal",
                    name_prefix=f"unet_skip_{len(unet_skips)+1}",
                )
            )

        # Separate pool layer so that we can save unet skip after conv but before pool where needed
        current = pool(
            current,
            pool_type=pool_type,
            pool_size=2,
            padding="same",
            name=f"tower_conv_{cidx+1}_pool",
        )

    # Build transformer tower
    # Each block Residual(LayerNorm+MHA+Dropout) + Residual(LayerNorm+Conv+Dropout+ReLU+Conv+Dropout)
    for tidx in range(num_transformer_blocks):
        current = mha_block_enf(
            inputs=current,
            num_heads=num_transformer_heads,
            key_query_dim=64,
            value_dim=filters // num_transformer_heads,
            scaling=True,
            attn_dropout=0.05,
            pos_dropout=0.01,
            final_dropout=transformer_dropout,
            symmetric_pos_encoding=False,
            pos_encoding_funs="borzoi",
            num_pos_feats=32,
            zero_init=True,
            residual=True,
            ln_epsilon=1e-3,
            name_prefix=f"transformer_mha_{tidx+1}",
        )
        current = ffn_block_enf(
            inputs=current,
            filters=filters,
            dropout=transformer_dropout,
            activation=transformer_activation,
            residual=True,
            ln_epsilon=1e-3,
            name_prefix=f"transformer_ff_{tidx+1}",
        )

    # Build upsampling tower
    # Generate an upsampling region (((ConvBlock -> Upsampling) + skipblock) -> SeparableConv) for each unet connection
    for uidx, unet_skip_current in enumerate(unet_skips[::-1]):
        # First upsampling conv block
        # If upsample_conv=True, matches a conv_block_bs
        # If upsample_conv=False, keeps batchnorm/activation but drops conv, like baskerville unet_conv upsample_conv=False
        current = keras.layers.BatchNormalization(
            momentum=0.99,
            epsilon=1e-3,
            gamma_initializer="ones",
            synchronized=bn_sync,
            name=f"upsampling_conv_{uidx+1}_batchnorm"
        )(current)
        current = activate(
            current, conv_activation, name=f"upsampling_conv_{uidx+1}_activation"
        )
        if upsample_conv:
            current = keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=True,
                dilation_rate=1,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(0),
                name=f"upsampling_conv_{uidx+1}_conv"
            )(current)
        # Upsample
        current = keras.layers.UpSampling1D(size=2)(current)

        # Add skip
        current = keras.layers.Add()([current, unet_skip_current])

        # Run upsampling separable conv block
        current = keras.layers.SeparableConv1D(
            filters=filters,
            kernel_size=3,
            padding="same",
            name=f"upsampling_separable_{uidx+1}",
        )(current)

    # Crop outputs
    if crop_length > 0:
        current = keras.layers.Cropping1D(crop_length, name="crop")(current)

    if pointwise_filters is not None:
        # Run final pointwise convblock + dropout + gelu section
        current = conv_block_bs(
            current,
            filters=pointwise_filters,
            kernel_size=1,
            batch_norm=True,
            activation=conv_activation,
            activation_end=conv_activation,
            residual=False,
            l2_scale=0,
            dropout=pointwise_dropout,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_sync=bn_sync,
            bn_epsilon=1e-3,
            kernel_initializer="he_normal",
            name_prefix="final_conv",
        )
    else:
        # Add final activation only
        current = activate(current, conv_activation, name="final_conv_activation_end")

    # Build heads
    if isinstance(num_classes, int):
        outputs = keras.layers.Conv1D(
            num_classes, kernel_size=1, activation=output_activation, name="head"
        )(current)
    elif isinstance(num_classes, cabc.Sequence):
        outputs = []
        for i, n_tracks in enumerate(num_classes):
            outputs.append(
                keras.layers.Conv1D(
                    n_tracks,
                    kernel_size=1,
                    activation=output_activation,
                    name=f"head_{i}",
                )(current)
            )
    else:
        raise ValueError(
            f"Could not recognise num_classes argument ({num_classes}) as integer or list/tuple/sequence of integers."
        )

    # Construct model
    m = keras.Model(inputs=sequence, outputs=outputs, name=name)
    return m


def exp_linspace_int(start, end, num_modules, divisible_by=1):
    """Get an exponentially rising set of values, guaranteed to be integers."""

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num_modules - 1))

    return [_round(start * base**i) for i in range(num_modules)]
