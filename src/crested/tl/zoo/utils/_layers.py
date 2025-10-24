"""Helper layers for zoo models."""

from __future__ import annotations

import keras
import numpy as np

from ._attention import AttentionPool1D, MultiheadAttention

__all__ = [
    "dense_block",
    "conv_block",
    "activate",
    "pool",
    "get_output",
    "conv_block_bs",
    "mha_block_enf",
    "ffn_block_enf",
    "dilated_residual",
]


def dense_block(
    inputs: keras.KerasTensor,
    units: int,
    activation: str,
    dropout: float = 0.5,
    l2: float = 1e-5,
    bn_gamma: str | None = None,
    bn_momentum: float = 0.90,
    normalization: str = "batch",
    name_prefix: str | None = None,
    use_bias: bool = True,
) -> keras.KerasTensor:
    """
    Dense building block.

    Parameters
    ----------
    inputs
        Input tensor.
    units
        Number of units in the dense layer.
    activation
        Activation function applied after dense layer.
    dropout
        Dropout rate (default is 0.5).
    l2
        L2 regularization weight (default is 1e-5).
    bn_gamma
        Gamma initializer for batch normalization.
    bn_momentum
        Batch normalization momentum (default is 0.90).
    normalization
        Type of normalization ('batch' or 'layer').
    name_prefix
        Prefix for layer names.
    use_bias
        Whether to use bias in the dense layer (default is True).

    Returns
    -------
    The output tensor of the dense block.
    """
    x = keras.layers.Dense(
        units,
        activation=None,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2),
        name=name_prefix + "_dense" if name_prefix else None,
    )(inputs)

    if normalization == "batch":
        x = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=bn_gamma,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization(
            name=name_prefix + "_layernorm" if name_prefix else None
        )(x)

    x = activate(
        x, activation, name=name_prefix + "_activation" if name_prefix else None
    )
    x = keras.layers.Dropout(
        dropout, name=name_prefix + "_dropout" if name_prefix else None
    )(x)
    return x


def conv_block(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int,
    pool_size: int,
    activation: str,
    conv_bias: bool = True,
    dropout: float = 0.1,
    normalization: str = "batch",
    res: bool = False,
    padding: str = "valid",
    l2: float = 1e-5,
    batchnorm_momentum: float = 0.99,
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Convolution building block.

    Parameters
    ----------
    inputs
        Input tensor representing the data.
    filters
        Number of filters in the convolutional layer.
    kernel_size
        Size of the convolutional kernel.
    pool_size
        Size of the max-pooling kernel.
    activation
        Activation function applied after convolution.
    conv_bias
        Whether to use bias in the convolutional layer (default is True).
    dropout
        Dropout rate (default is 0.1).
    normalization
        Type of normalization ('batch' or 'layer').
    res
        Whether to use residual connections (default is False).
    padding
        Padding type for the convolutional layer (default is "valid").
    l2
        L2 regularization weight (default is 1e-5).
    batchnorm_momentum
        Batch normalization momentum (default is 0.99).
    name_prefix
        Prefix for layer names.

    Returns
    -------
    The output tensor of the convolution block.
    """
    if res:
        residual = inputs

    x = keras.layers.Convolution1D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_regularizer=keras.regularizers.L2(l2),
        use_bias=conv_bias,
        name=name_prefix + "_conv" if name_prefix else None,
    )(inputs)
    if normalization == "batch":
        x = keras.layers.BatchNormalization(
            momentum=batchnorm_momentum,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization(
            name=name_prefix + "_layernorm" if name_prefix else None
        )(x)
    x = activate(
        x, activation, name=name_prefix + "_activation" if name_prefix else None
    )
    if res:
        if filters != residual.shape[2]:
            residual = keras.layers.Convolution1D(
                filters=filters,
                kernel_size=1,
                strides=1,
                name=name_prefix + "_resconv" if name_prefix else None,
            )(residual)
        x = keras.layers.Add()([x, residual])

    if pool_size > 1:
        x = keras.layers.MaxPooling1D(
            pool_size=pool_size,
            padding=padding,
            name=name_prefix + "_pool" if name_prefix else None,
        )(x)
    if dropout > 0:
        x = keras.layers.Dropout(
            dropout, name=name_prefix + "_dropout" if name_prefix else None
        )(x)

    return x


def activate(
    current: keras.KerasTensor, activation: str, verbose: bool = False, name=None
) -> keras.KerasTensor:
    """
    Apply activation function to a tensor.

    Parameters
    ----------
    current
        Input tensor.
    activation
        Activation function to apply.
    verbose
        Print verbose information (default is False).
    name
        Name to use in the activation layer. Default is None (no name).

    Returns
    -------
    Output tensor after applying activation.
    """
    if verbose:
        print("activate:", activation)

    if activation == "relu":
        current = keras.layers.Activation("relu", name=name)(current)
    elif activation == "swish":
        current = keras.layers.Activation("swish", name=name)(current)
    elif activation == "gelu":
        current = keras.layers.Activation("gelu", name=name)(current)
    elif activation == "gelu_approx":
        # layers.Activation('gelu') uses unapproximated (default in activations.gelu), we want approximated
        current = keras.layers.Activation(gelu_approx, name=name)(current)
    elif activation == "sigmoid":
        current = keras.layers.Activation("sigmoid", name=name)(current)
    elif activation == "tanh":
        current = keras.layers.Activation("tanh", name=name)(current)
    elif activation == "exponential":
        current = keras.layers.Activation("exponential", name=name)(current)
    elif activation == "softplus":
        current = keras.layers.Activation("softplus", name=name)(current)
    elif activation == "gelu_enf":
        current = keras.layers.Activation(gelu_enf, name=name)(current)
    else:
        raise ValueError(f'Unrecognized activation "{activation}"')

    return current


@keras.saving.register_keras_serializable(package="crested", name="gelu_approx")
def gelu_approx(x):
    """Wrap around keras.activations.gelu with approximate = True."""
    return keras.activations.gelu(x, approximate=True)


@keras.saving.register_keras_serializable(package="crested", name="gelu_enf")
def gelu_enf(x):
    """Very simple gelu approximation, used in Enformer, so needed to get equivalent results."""
    return keras.ops.sigmoid(1.702 * x) * x


def pool(
    current: keras.KerasTensor,
    pool_type: str,
    pool_size=2,
    padding="same",
    verbose: bool = False,
    name=None,
) -> keras.KerasTensor:
    """
    Apply activation function to a tensor.

    Parameters
    ----------
    current
        Input tensor.
    pool_type
        Pooling function to apply.
    verbose
        Print verbose information (default is False).
    name
        Name to use in the activation layer. Default is None (no name).

    Returns
    -------
    Output tensor after applying activation.
    """
    if verbose:
        print("pool:", pool_type)

    if pool_type == "max":
        current = keras.layers.MaxPooling1D(
            pool_size=pool_size, padding=padding, name=name
        )(current)
    elif pool_type == "attention":
        current = AttentionPool1D(pool_size=2, padding=padding, name=name)(current)
    elif pool_type == "average":
        current = keras.layers.AveragePooling1D(
            pool_size=2, padding=padding, name=name
        )(current)
    else:
        raise ValueError(f'Unrecognized pooling type "{pool_type}"')
    return current


def get_output(input_layer, hidden_layers):
    """
    Pass input layer through hidden layers.

    Parameters
    ----------
    input_layer
        Input layer.
    hidden_layers
        Hidden layers.

    Returns
    -------
    keras.KerasTensor
        Output tensor after passing through all hidden layers.
    """
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)

    return output


def conv_block_bs(
    inputs,
    filters: int | None = None,
    kernel_size: int = 1,
    pool_size: int = 1,
    batch_norm: bool = False,
    activation: str = "relu",
    activation_end: str | None = None,
    dropout: float = 0,
    residual: bool = False,
    strides: int = 1,
    dilation_rate: int = 1,
    l2_scale: float = 0,
    conv_type: str = "standard",
    conv_bias: bool = True,
    pool_type: str = "max",
    bn_momentum: float = 0.99,
    bn_gamma: keras.KerasTensor | None = None,
    bn_sync: str = True,
    bn_epsilon: float = 1e-5,
    kernel_initializer: str = "he_normal",
    padding: str = "same",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Construct a convolution block (for Basenji/Enformer).

    Important note: while conv_block() follows the conv-norm-activate-pool order,
    this follows Enformer/Borzoi's convention of norm-activate-conv-pool.

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Conv1D filters.
    kernel_size
        Conv1D kernel_size.
    pool_size
        Max pool width.
    batch_norm
        Apply batch normalization.
    activation
        Activation function.
    activation_end
        Activation to add at end of block, after residual and before pooling.
        Optional, can be None to disable.
    dropout
        Dropout rate probability.
    residual
        Residual connection boolean.
    strides
        Conv1D strides.
    dilation_rate
        Conv1D dilation rate.
    l2_scale
        L2 regularization weight.
    conv_type
        Conv1D layer type.
    conv_bias
        Whether to use a bias in the convolution layer.
        Should be True for Enformer/Borzoi and (maybe) False for Borzoi?
    pool_type
        Pooling type. Either 'max' or 'attention'.
    bn_momentum
        BatchNorm momentum.
    bn_gamma
        BatchNorm gamma (defaults according to residual).
    bn_sync
        Whether to synchronise the batchnorm when doing multi-GPU training.
    bn_epsilon
        Epsilon to use for the BatchNorm layer. Usual default is 1e-5, but Borzoi uses 1e-3.
    kernel_initializer
        Convolution kernel initializer.
    padding
        Padding type.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the convolution block.
    """
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            gamma_initializer=bn_gamma,
            synchronized=bn_sync,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )(current)

    # activation
    current = activate(
        current, activation, name=name_prefix + "_activation" if name_prefix else None
    )

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=conv_bias,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale),
        name=name_prefix + "_conv" if name_prefix else None,
    )(current)

    # dropout
    if dropout > 0:
        current = keras.layers.Dropout(
            rate=dropout, name=name_prefix + "_dropout" if name_prefix else None
        )(current)

    # residual add
    if residual:
        current = keras.layers.Add(name=name_prefix + "_add" if name_prefix else None)(
            [inputs, current]
        )

    # end activation
    if activation_end is not None:
        current = activate(
            current,
            activation_end,
            name=name_prefix + "_activation_end" if name_prefix else None,
        )

    # Pool
    if pool_size > 1:
        current = pool(
            current,
            pool_type=pool_type,
            pool_size=pool_size,
            padding=padding,
            name=name_prefix + "_pool" if name_prefix else None,
        )
    return current


def mha_block_enf(
    inputs,
    num_heads: int,
    key_query_dim: int,
    value_dim: int,
    scaling: bool = True,
    attn_dropout: float = 0.05,
    pos_dropout: float = 0.01,
    final_dropout: float = 0.4,
    symmetric_pos_encoding: bool = False,
    pos_encoding_funs: str = "borzoi",
    pos_encoding_abs: bool = True,
    num_pos_feats: int | None = None,
    zero_init: bool = True,
    residual: bool = True,
    ln_epsilon: float = 1e-5,
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Construct a MHA block (for Enformer/Borzoi), consisting of Residual(LayerNorm+MHSelfAttention+Dropout).

    Parameters
    ----------
    inputs
        Input tensor.
    num_heads
        Number of attention heads to use.
    key_query_dim
        Number of k (key) and q (query) dimensions in the attention mechanism.
    value_dim
        Number of v (value) dimensions in the attention mechanism.
    scaling
        Whether to use scaling.
    attn_dropout
        Attention dropout rate.
    pos_dropout
        Positional embedding dropout rate.
    final_dropout
        Block-included post-MHA dropout rate.
    symmetric_pos_encoding
        Whether to make positional encodings symmetric. Only relevant if pos_encoding = True.
    pos_encoding_funs
        Can be 'enformer' or 'borzoi'.
        Enformer default uses all, using exponential+central_mask_enf+gamma.,
        Borzoi default only uses its version of central_mask.
    pos_encoding_abs
        Whether to use the absolute of values before calculating the relative position encoding.
    num_pos_feats
        Number of positional features. If not supplied, calculated from value_dim and number of position encoding functions.
        Min 6 for default relative_position_functions, min 12 for positional_features_sin_cos.
    zero_init
        Whether to initialize MHA from zero.
    residual
        Whether to wrap the entire block in residual structure.
    ln_epsilon
        Epsilon to use in the layer normalisation layer.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the MHA block.
    """
    # MHA block
    current = keras.layers.LayerNormalization(
        epsilon=ln_epsilon,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        name=f"{name_prefix}_layernorm",
    )(inputs)
    current = MultiheadAttention(
        value_size=value_dim,
        key_size=key_query_dim,
        heads=num_heads,
        scaling=scaling,
        attention_dropout_rate=attn_dropout,
        relative_position_symmetric=symmetric_pos_encoding,
        relative_position_functions=pos_encoding_funs,
        relative_position_absolute=pos_encoding_abs,
        num_position_features=num_pos_feats,
        positional_dropout_rate=pos_dropout,
        zero_initialize=zero_init,
        initializer="he_normal",  # Unsure in Enf, think this is fine.
        l2_scale=1.0e-8,  # Doesn't seem to be set in Enf, is set in Borzoi.
        name=f"{name_prefix}_mhsa",
    )(current)
    current = keras.layers.Dropout(rate=final_dropout, name=f"{name_prefix}_dropout")(
        current
    )
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current


def ffn_block_enf(
    inputs,
    filters: int,
    expansion_rate: int = 2,
    dropout: int = 0.4,
    activation: str = "relu",
    residual: bool = True,
    ln_epsilon: float = 1e-5,
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Construct a feedforward block (for Enformer), consisting of Residual(LayerNorm+PointwiseConv+Dropout+ReLU+PointwiseConv+Dropout).

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Pointwise convolution filters.
    expansion_rate
        Scaling factor of base filters inside the FFN.
    dropout
        Dropout rate.
    activation
        Which activation function to use.
    residual
        Whether to wrap the entire block in residual structure.
    ln_epsilon
        Epsilon to use in the layer normalisation layer.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the feedforward block.
    """
    expansion_filters = int(expansion_rate * filters)

    # First half
    current = keras.layers.LayerNormalization(
        epsilon=ln_epsilon,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        name=f"{name_prefix}_layernorm",
    )(inputs)
    current = keras.layers.Conv1D(
        filters=expansion_filters, kernel_size=1, name=f"{name_prefix}_pointwise_1"
    )(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_1")(
        current
    )

    # Second half
    current = activate(
        current, activation, name=name_prefix + "_activation" if name_prefix else None
    )
    current = keras.layers.Conv1D(
        filters=filters, kernel_size=1, name=f"{name_prefix}_pointwise_2"
    )(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_2")(
        current
    )

    # Residual
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current


def dilated_residual(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: float = 2.0,
    dropout: float = 0,
    conv_type: str = "standard",
    repeat: int = 1,
    round: bool = False,
    **kwargs,
) -> keras.KerasTensor:
    """
    Construct a residual dilated convolution block.

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Number of filters in the convolutional layer.
    kernel_size
        Size of the convolutional kernel.
    rate_mult
        Rate multiplier for dilated convolution.
    dropout
        Dropout rate probability.
    conv_type
        Conv1D layer type.
    repeat
        Number of times to repeat the block.
    round
        Whether to round the dilation rate.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Output tensor after applying the dilated residual block.
    """
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for _ in range(repeat):
        rep_input = current

        # dilate
        current = conv_block_bs(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            conv_type=conv_type,
            bn_gamma="ones",
            **kwargs,
        )

        # return
        current = conv_block_bs(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            bn_gamma="zeros",
            **kwargs,
        )

        # residual add
        current = keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult
        if round:
            dilation_rate = np.round(dilation_rate)

    return current
