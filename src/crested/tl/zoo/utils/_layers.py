"""Helper layers for zoo models."""

from __future__ import annotations

import keras
import numpy as np

__all__ = [
    "dense_block",
    "conv_block",
    "activate",
    "get_output",
    "conv_block_bs",
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
):
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

    x = keras.layers.Activation(
        activation, name=name_prefix + "_activation" if name_prefix else None
    )(x)
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
    )(inputs)
    if normalization == "batch":
        x = keras.layers.BatchNormalization(momentum=batchnorm_momentum)(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    if res:
        if filters != residual.shape[2]:
            residual = keras.layers.Convolution1D(
                filters=filters, kernel_size=1, strides=1
            )(residual)
        x = keras.layers.Add()([x, residual])

    if pool_size > 1:
        x = keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding)(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)

    return x


def activate(
    current: keras.KerasTensor, activation: str, verbose: bool = False
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

    Returns
    -------
    Output tensor after applying activation.
    """
    if verbose:
        print("activate:", activation)

    if activation == "relu":
        current = keras.layers.Activation("relu")(current)
    elif activation == "swish":
        current = keras.layers.Activation("swish")(current)
    elif activation == "gelu":
        current = keras.layers.Activation("gelu")(current)
    elif activation == "sigmoid":
        current = keras.layers.Activation("sigmoid")(current)
    elif activation == "tanh":
        current = keras.layers.Activation("tanh")(current)
    elif activation == "exponential":
        current = keras.layers.Activation("exponential")(current)
    elif activation == "softplus":
        current = keras.layers.Activation("softplus")(current)
    else:
        print(f'Unrecognized activation "{activation}"')

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
    tf.Tensor
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
    w1: bool = False,
    bn_momentum: float = 0.99,
    bn_gamma: keras.KerasTensor | None = None,
    bn_type: str = "standard",
    kernel_initializer: str = "he_normal",
    padding: str = "same",
):
    """
    Construct a convolution block (for Basenji).

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Conv1D filters.
    kernel_size
        Conv1D kernel_size.
    activation
        Activation function.
    strides
        Conv1D strides.
    dilation_rate
        Conv1D dilation rate.
    l2_scale
        L2 regularization weight.
    dropout
        Dropout rate probability.
    conv_type
        Conv1D layer type.
    residual
        Residual connection boolean.
    pool_size
        Max pool width.
    batch_norm
        Apply batch normalization.
    bn_momentum
        BatchNorm momentum.
    bn_gamma
        BatchNorm gamma (defaults according to residual).
    kernel_initializer
        Convolution kernel initializer.
    padding
        Padding type.

    Returns
    -------
    Output tensor after applying the convolution block.
    """
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    elif w1:
        conv_layer = keras.layers.Conv2D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = activate(current, activation_end)

    # Pool
    if pool_size > 1:
        if w1:
            current = keras.layers.MaxPool2D(pool_size=pool_size, padding=padding)(
                current
            )
        else:
            current = keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def dilated_residual(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: int = 2,
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
