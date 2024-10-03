"""Enformer model architecture. Adapted from github.com/casblaauw/enformer_keras."""

import keras
import numpy as np
import collections.abc as cabc

from crested.tl.zoo.utils import conv_block_bs, activate, mha_block_enf, ffn_block_enf

def enformer(
    seq_len: int,
    num_classes: int | cabc.Sequence[int] | cabc.Mapping[str, int],
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
    first_kernel_size: int = 15,
    kernel_size: int = 5,
    transformer_dropout = 0.4,
    pointwise_dropout: float = 0.05,
    name: str = 'Enformer'
) -> keras.Model:
    """"
    Construct an fully replicated Enformer model.
    Note that unlike other CREsted model zoo architectures, this architecture is not suited for 
    predicting individual regions out of the box.

    Parameters
    ----------
    seq_len
        Width of the input region.
    num_classes
        Number of classes to predict.
        If an int, creates a single head with num_classes classes.
        If a list of integers, creates multiple heads in a list. 
        If a dictionary of names and integers, creates multiple named heads. 
    num_conv_blocks
        Number of convolution blocks to include in the tower, after the stem block.
    num_transformer_blocks
        Number of transformer blocks to include in the transformer stack.
    target_length
        The target length in bins to crop to. Default is 896, cropping away 320 bins (41kb) on each side.
    start_filters
        Starting number of filters for the first block, exponentially increasing towards filters through the conv tower.
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
    first_kernel_size
        Kernel size of the first conv layer, directly interfacing the sequence.
    kernel_size
        Kernel size of the convolutions in the conv tower.
    transformer_dropout
        Dropout rate used in the transformer blocks, both MHA and feed-forward.
    pointwise_dropout
        Dropout rate of the post-transformer final pointwise layer.

    Returns
    -------
    A Keras model.
    """
    # Note: base Enformer sets first_filters as filters // 2, not sure how hard of a requirement that is.

    # Calculate derived parameters
    # Tower output length: n of bins after convolution pooling.
    #   every conv layer (and stem layer) halves length -> seq_len/binwidth = dimensionality
    # Crop length: (original dimensionality - target_length) // 2 = crop length from both sides 
    tower_out_length = int(seq_len/(2**(num_conv_blocks + 1)))
    crop_length = int((tower_out_length-target_length)//2)
    
    # Sequence input
    sequence = keras.layers.Input(shape=(seq_len, 4), name="input")
   
    # Build stem (standard conv + residual(batchnorm+gelu+conv)+pooling block)
    current = keras.layers.Conv1D(
        filters=start_filters, 
        kernel_size=first_kernel_size, 
        padding='same', 
        name='stem_conv'
        )(sequence)

    current = conv_block_bs(
        current,
        filters=start_filters,
        kernel_size=1,
        activation=conv_activation,
        activation_end=None,
        strides=1,
        dilation_rate=1,
        l2_scale=0,
        dropout=0,
        conv_type="standard",
        pool_type="attention",
        residual=True,
        pool_size=2,
        batch_norm=True,
        bn_momentum=0.9,
        bn_gamma=None,
        bn_type="standard",
        kernel_initializer="he_normal",
        padding="same",
        name_prefix="stem_pointwise"
    )

    # Build convolution tower
     # Each block: (batchnorm + gelu + conv) + residual(batchnorm+gelu+conv)+pooling
    tower_filters = exp_linspace_int(start=start_filters, end=filters, num_modules=num_conv_blocks, divisible_by=128)
    for cidx, layer_filters in enumerate(tower_filters):
        # Add first standard conv block
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=kernel_size,
            activation=conv_activation,
            activation_end=None,
            strides=1,
            dilation_rate=1,
            l2_scale=0,
            dropout=0,
            conv_type="standard",
            pool_type="attention",
            residual=False,
            pool_size=1,
            batch_norm=True,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_type="standard",
            kernel_initializer="he_normal",
            padding="same",
            name_prefix=f"tower_conv_{cidx+1}"
        )
        # Add residual pointwise conv block
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=1,
            activation=conv_activation,
            activation_end=None,
            strides=1,
            dilation_rate=1,
            l2_scale=0,
            dropout=0,
            conv_type="standard",
            pool_type="attention",
            residual=True,
            pool_size=2,
            batch_norm=True,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_type="standard",
            kernel_initializer="he_normal",
            padding="same",
            name_prefix=f"tower_pointwise_{cidx+1}"
        )
    
    # Identity layer to use as stopping point for FastISM - after this operations are global
    # Covers an edge case according to devs
    
    # current = keras.layers.Layer()(current)

    # Build transformer tower
    for tidx in range(num_transformer_blocks):
        current = mha_block_enf(
            inputs = current,
            num_heads = num_transformer_heads,
            key_query_dim = 64,
            value_dim = filters // num_transformer_heads,
            scaling = True,
            attn_dropout = 0.05,
            pos_dropout = 0.01,
            final_dropout = transformer_dropout,
            pos_encoding = True,
            symmetric_pos_encoding = False,
            pos_encoding_funs = ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma'],
            num_pos_feats = filters // num_transformer_heads,
            zero_init = True,
            residual = True,
            name_prefix = f"transformer_mha_{tidx+1}"
        )
        current = ffn_block_enf(
            inputs = current,
            filters = filters,
            dropout = transformer_dropout,
            activation = transformer_activation,
            residual = True,
            name_prefix = f"transformer_ff_{tidx+1}"

        )


    # Build crop and pointwise final block
    if crop_length > 0:
        current = keras.layers.Cropping1D(crop_length, name = 'crop')(current)
    current = conv_block_bs(
            current,
            filters=pointwise_filters,
            kernel_size=1,
            activation=conv_activation,
            activation_end=None,
            strides=1,
            dilation_rate=1,
            l2_scale=0,
            dropout=0,
            conv_type="standard",
            pool_type="attention",
            residual=True,
            pool_size=2,
            batch_norm=True,
            bn_momentum=0.9,
            bn_gamma=None,
            bn_type="standard",
            kernel_initializer="he_normal",
            padding="same",
            name_prefix=f"final_pointwise"
        )
    current = keras.layers.Dropout(pointwise_dropout, name = 'final_pointwise_dropout')(current)
    current = activate(current, conv_activation)
    
    # Build heads
    if isinstance(num_classes, int):
        outputs = keras.layers.Dense(num_classes, activation=output_activation, input_shape = (target_length, filters*2), name = head)(current)
    elif isinstance(num_classes, cabc.Mapping):
        outputs = {}
        for head, n_tracks in num_classes.items():
            outputs[head] = keras.layers.Dense(n_tracks, activation=output_activation, input_shape = (target_length, filters*2), name = head)(current)
    elif isinstance(num_classes, cabc.Sequence):
        outputs = []
        for head, n_tracks in num_classes:
            outputs.append(keras.layers.Dense(n_tracks, activation=output_activation, input_shape = (target_length, filters*2), name = head)(current))

    
    # Construct model
    m = keras.Model(inputs = sequence, outputs = outputs, name = name)
    return m

def exp_linspace_int(start, end, num_modules, divisible_by=1):
    def _round(x):
        return int(np.round(x/divisible_by)*divisible_by)
    base = np.exp(np.log(end/start)/(num_modules-1))
    
    return [_round(start*base**i) for i in range(num_modules)]