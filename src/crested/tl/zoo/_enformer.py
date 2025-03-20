"""Enformer model architecture. Adapted from github.com/casblaauw/enformer_keras."""

from __future__ import annotations

import collections.abc as cabc

import keras
import numpy as np

from crested.tl.zoo.utils import activate, conv_block_bs, ffn_block_enf, mha_block_enf


def enformer(
    seq_len: int,
    num_classes: int | cabc.Sequence[int],
    num_conv_blocks: int = 6,
    num_transformer_blocks: int = 11,
    num_transformer_heads: int = 8,
    target_length: int = 896,
    start_filters: int = 768,
    filters: int = 1536,
    pointwise_filters: int = 3072,
    conv_activation: str = "gelu_enf",
    transformer_activation: str = "relu",
    output_activation: str = "softplus",
    pool_type: str = "attention",
    first_kernel_size: int = 15,
    kernel_size: int = 5,
    transformer_dropout=0.4,
    pointwise_dropout: float = 0.05,
    bn_sync: bool = False,
    name: str = "Enformer",
) -> keras.Model:
    """
    Construct an fully replicated Enformer model.

    Note that unlike other CREsted model zoo architectures, this architecture is not suited for
    predicting individual regions out of the box.

    Parameters
    ----------
    seq_len
        Width of the input region.
        Enformer default is 196608
    num_classes
        Number of classes to predict.
        If an int, creates a single head with num_classes classes.
        If a list of integers, creates multiple heads in a list.
    num_conv_blocks
        Number of convolution blocks to include in the tower, after the stem block.
    num_transformer_blocks
        Number of transformer blocks to include in the transformer stack.
    target_length
        The target length in bins to crop to. Default is 896, cropping away 320 bins (41kb) on each side.
    start_filters
        Starting number of filters for the first DNA-facing and first conv tower block,
        exponentially increasing towards `filters` through the conv tower.
    filters
        Number of filters at the end of the conv tower.
    pointwise_filters
        Number of filters of the post-transformer final pointwise convolution.
    conv_activation
        Activation function to use in the conv tower and in the final pointwise block.
    transformer_activation
        Activation function to use in the feedforward section of the transformer blocks.
    output_activation
        Final activation to use on the output heads, just before predicting the tracks.
    pool_type
        Pooling type to use, one of 'max' or 'attention'.
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
        Default is False.

    Returns
    -------
    A Keras model.
    """
    # Note: base Enformer sets first_filters as filters // 2, not sure how hard of a requirement that is.

    # Calculate derived parameters
    # Tower output length: n of bins after convolution pooling.
    #   every conv layer (and stem layer) halves length -> seq_len/binwidth = dimensionality
    # Crop length: (original dimensionality - target_length) // 2 = crop length from both sides
    tower_out_length = int(
        seq_len / (2 ** (num_conv_blocks + 1))
    )  # Should be same as filters
    crop_length = int((tower_out_length - target_length) // 2)

    # Sequence input
    sequence = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Build stem (standard conv + residual(batchnorm+gelu+conv)+pooling block)
    current = keras.layers.Conv1D(
        filters=start_filters,
        kernel_size=first_kernel_size,
        padding="same",
        name="stem_conv",
    )(sequence)

    current = conv_block_bs(
        current,
        filters=start_filters,
        kernel_size=1,
        pool_size=2,
        batch_norm=True,
        activation=conv_activation,
        residual=True,
        l2_scale=0,
        pool_type=pool_type,
        bn_momentum=0.9,
        bn_gamma=None,
        bn_sync=bn_sync,
        bn_epsilon=1e-5,
        kernel_initializer="he_normal",
        name_prefix="stem_pointwise",
    )

    # Build convolution tower
    # Each block: (batchnorm + gelu + conv) + residual(batchnorm+gelu+conv)+pooling
    tower_filters = exp_linspace_int(
        start=start_filters, end=filters, num_modules=num_conv_blocks, divisible_by=128
    )
    for cidx, layer_filters in enumerate(tower_filters):
        # Add first standard conv block
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=kernel_size,
            activation=conv_activation,
            batch_norm=True,
            residual=False,
            l2_scale=0,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_sync=bn_sync,
            bn_epsilon=1e-5,
            kernel_initializer="he_normal",
            name_prefix=f"tower_conv_{cidx+1}",
        )
        # Add residual pointwise conv block
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=1,
            pool_size=2,
            batch_norm=True,
            activation=conv_activation,
            residual=True,
            l2_scale=0,
            pool_type=pool_type,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_sync=bn_sync,
            bn_epsilon=1e-5,
            kernel_initializer="he_normal",
            name_prefix=f"tower_pointwise_{cidx+1}",
        )

    # Identity layer to use as stopping point for FastISM - after this operations are global
    # Covers an edge case according to devs

    # current = keras.layers.Layer()(current)

    # Build transformer tower
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
            pos_encoding_funs="enformer",
            num_pos_feats=filters // num_transformer_heads,
            zero_init=True,
            residual=True,
            ln_epsilon=1e-5,
            name_prefix=f"transformer_mha_{tidx+1}",
        )
        current = ffn_block_enf(
            inputs=current,
            filters=filters,
            dropout=transformer_dropout,
            activation=transformer_activation,
            residual=True,
            ln_epsilon=1e-5,
            name_prefix=f"transformer_ff_{tidx+1}",
        )

    # Build crop and pointwise final block
    if crop_length > 0:
        current = keras.layers.Cropping1D(crop_length, name="crop")(current)
    current = conv_block_bs(
        current,
        filters=pointwise_filters,
        kernel_size=1,
        batch_norm=True,
        activation=conv_activation,
        residual=False,
        l2_scale=0,
        bn_momentum=0.9,
        bn_gamma=None,
        bn_sync=bn_sync,
        bn_epsilon=1e-5,
        kernel_initializer="he_normal",
        name_prefix="final_pointwise",
    )
    current = keras.layers.Dropout(pointwise_dropout, name="final_pointwise_dropout")(
        current
    )
    current = activate(current, conv_activation, name="final_activation")

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
