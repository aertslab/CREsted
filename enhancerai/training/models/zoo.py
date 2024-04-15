"""DeepPeak model implementations."""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import int_shape


######################################################
# Models
######################################################


def simple_convnet(
    input_shape: tuple,
    output_shape: tuple,
    num_conv_blocks: int = 3,
    num_dense_blocks: int = 2,
    residual: int = 0,
    first_activation: str = "exponential",
    activation: str = "swish",
    output_activation: str = "softplus",
    normalization: str = "batch",
    first_filters: int = 192,
    filters: int = 256,
    first_kernel_size: int = 13,
    kernel_size: int = 7,
    first_pool_size: int = 8,
    pool_size: int = 2,
    conv_dropout: float = 0.1,
    dense_dropout: float = 0.3,
    flatten: bool = True,
    dense_size: int = 256,
    bottleneck: int = 8,
):
    """
    Simple ConvNet that can serve as a baseline.

    Args:
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output data.
        num_conv_blocks (int): Number of conv blocks (default is 3).
        num_dense_blocks (int): Number of dense blocks (default is 2).
        residual (int): Whether to use residual connections (default is 0).
        first_activation (str): Activation function for the first conv block.
        activation (str): Activation function for subsequent conv and dense blocks.
        output_activation (str): Activation function for the output layer.
        normalization (str): Type of normalization ('batch' or 'layer').
        first_filters (int): Number of filters in the first conv block.
        filters (int): Number of filters in subsequent conv block.
        first_kernel_size (int): Size of the kernel in the first conv block.
        kernel_size (int): Size of the kernel in subsequent conv blocks.
        first_pool_size (int): Size of the pooling kernel in the first conv block.
        pool_size (int): Size of the pooling kernel in subsequent conv blocks.
        conv_dropout (float): Dropout rate for conv layers.
        dense_dropout (float): Dropout rate for dense layers.
        flatten (bool): Whether to flatten the output.
        dense_size (int): Number of neurons in dense layers.
        bottleneck (int): Size of the bottleneck layer

    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    output_len, num_tasks = output_shape
    inputs = layers.Input(shape=input_shape, name="sequence")

    x = conv_block(
        inputs,
        filters=first_filters,
        kernel_size=first_kernel_size,
        pool_size=first_pool_size,
        activation=first_activation,
        dropout=conv_dropout,
        normalization=normalization,
        res=residual,
    )

    if num_conv_blocks > 1:
        for i in range(1, num_conv_blocks):
            x = conv_block(
                x,
                filters=i * filters,
                kernel_size=kernel_size,
                pool_size=pool_size,
                activation=activation,
                dropout=i * conv_dropout,
                normalization=normalization,
                res=residual,
            )

    if flatten:
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    for i in range(1, num_dense_blocks):
        x = dense_block(
            x,
            dense_size,
            activation,
            dropout=dense_dropout,
            normalization=normalization,
        )

    x = dense_block(
        x,
        output_len * bottleneck,
        activation,
        dropout=dense_dropout,
        normalization=normalization,
    )

    outputs = layers.Dense(num_tasks, activation=output_activation)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def chrombpnet(
    input_shape: tuple,
    output_shape: tuple,
    first_conv_filters: int = 512,
    first_conv_filter_size: int = 5,
    first_conv_pool_size: int = 0,
    first_conv_activation: str = "gelu",
    first_conv_l2: float = 0.00001,
    first_conv_dropout: float = 0.1,
    n_dil_layers: int = 8,
    num_filters: int = 512,
    filter_size: int = 3,
    activation: str = "relu",
    l2: float = 0.00001,
    dropout: float = 0.1,
    batch_norm: bool = True,
    dense_bias: bool = True,
):
    """
    Construct a ChromBPNet like model.

    Args:
        input_shape (tuple): Shape of the input sequence.
        num_classes (int): Number of classes for output.
        first_conv_filters (int): Number of filters in the first convolutional layer.
        first_conv_filter_size (int): Size of the first convolutional layer.
        first_conv_pool_size (int): Size of the first convolutional pooling layer.
        first_conv_activation (str): Activation function in the first conv layer.
        first_conv_l2 (float): L2 regularization for the first convolutional layer.
        first_conv_dropout (float): Dropout rate for the first convolutional layer.
        n_dil_layers (int): Number of dilated convolutions.
        num_filters (int): Number of filters in the dilated convolutions.
        filter_size (int): Size of the dilated convolutions.
        activation (str): Activation function in the dilated convolutions.
        l2 (float): L2 regularization for the dilated convolutions.
        dropout (float): Dropout rate for the dilated convolutions.
        batch_norm (bool): Whether or not to use batch normalization.
        dense_bias (bool): Whether or not to add a bias to the dense layer.
    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    # Model
    inputs = layers.Input(shape=input_shape, name="sequence")

    # Convolutional block without dilation
    x = layers.Conv1D(
        filters=first_conv_filters,
        kernel_size=first_conv_filter_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(first_conv_l2),
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(x)
    x = layers.Activation(first_conv_activation)(x)
    if first_conv_pool_size > 1:
        x = layers.MaxPooling1D(pool_size=first_conv_pool_size, padding="same")(x)
    x = layers.Dropout(first_conv_dropout)(x)

    # Dilated convolutions
    layer_names = [str(i) for i in range(1, n_dil_layers + 1)]

    for i in range(1, n_dil_layers + 1):
        conv_layer_name = "bpnet_{}conv".format(layer_names[i - 1])
        conv_x = layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=1,
            activation=None,
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            use_bias=False,
            dilation_rate=2**i,
            name=conv_layer_name,
        )(x)
        if batch_norm:
            conv_x = layers.BatchNormalization(
                momentum=0.9,
                gamma_initializer="ones",
                name=f"bpnet_{layer_names[i - 1]}bn",
            )(conv_x)
        if activation != "none":
            conv_x = layers.Activation(
                activation, name=f"bpnet_{layer_names[i - 1]}activation"
            )(conv_x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0  # for symmetric cropping

        x = layers.Cropping1D(
            (x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i - 1]}crop"
        )(x)
        x = layers.add([conv_x, x])
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"bpnet_{layer_names[i-1]}dropout")(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(
        units=output_shape[-1], activation="linear", use_bias=dense_bias
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def basenji(
    input_shape: tuple,
    output_shape: tuple,
    first_activation: str = "gelu",
    activation: str = "gelu",
    output_activation: str = "softplus",
    first_filters: int = 256,
    filters: int = 2048,
    first_kernel_size: int = 15,
    kernel_size: int = 3,
):
    """
    Construct a Basenji model.

    Args:
        input_shape (tuple): Shape of the input data (sequence_length, 4).
        output_shape (tuple): Shape of the output data (output_length, num_tasks)
        first_activation (str): Activation function for the first conv block
        activation (str): Activation function for subsequent blocks (default is 'gelu').
        output_activation (str): Activation function for the output layer
        first_filters (int, optional): Number of filters in the first conv block
        filters (int, optional): Number of filters in subsequent conv blocks
        first_kernel_size (int, optional): Size of the kernel in the first conv block
        kernel_size (int, optional): Size of the kernel in subsequent conv blocks.

    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    window_size = int(input_shape[0] // output_shape[0] // 2)

    if window_size == 0:
        pool_1 = 1
        window_size = 1
    else:
        pool_1 = 2

    sequence = layers.Input(shape=input_shape, name="sequence")

    current = conv_block_bs(
        sequence,
        filters=first_filters,
        kernel_size=first_kernel_size,
        activation=first_activation,
        activation_end=None,
        strides=1,
        dilation_rate=1,
        l2_scale=0,
        dropout=0,
        conv_type="standard",
        residual=False,
        pool_size=pool_1,
        batch_norm=True,
        bn_momentum=0.9,
        bn_gamma=None,
        bn_type="standard",
        kernel_initializer="he_normal",
        padding="same",
    )

    current = dilated_residual(
        current,
        filters=int(first_filters // 2),
        kernel_size=kernel_size,
        rate_mult=1.5,
        conv_type="standard",
        dropout=0.3,
        repeat=11,
        round=False,
        activation=activation,
        batch_norm=True,
        bn_momentum=0.9,
    )

    current = conv_block_bs(
        current,
        filters=filters,
        kernel_size=1,
        activation=activation,
        dropout=0.05,
        batch_norm=True,
        bn_momentum=0.9,
    )

    current = layers.GlobalAveragePooling1D()(current)

    outputs = layers.Dense(
        units=output_shape[-1],
        use_bias=True,
        activation=output_activation,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0),
    )(current)

    model = tf.keras.Model(inputs=sequence, outputs=outputs)

    return model


######################################################
# Helper layers
######################################################


def dense_block(
    inputs,
    units,
    activation,
    dropout=0.5,
    l2=1e-5,
    bn_gamma=None,
    bn_momentum=0.90,
    normalization="batch",
):
    """
    Dense building block.

    Args:
        inputs (tf.Tensor): Input tensor.
        units (int): Number of units in the dense layer.
        activation (str): Activation function applied after dense layer.
        dropout (float, optional): Dropout rate (default is 0.5).
        l2 (float, optional): L2 regularization weight (default is 1e-5).
        bn_gamma (tf.Tensor, optional): Gamma initializer for batch normalization.
        bn_momentum (float, optional): Batch normalization momentum (default is 0.90).
        normalization (str, optional): Type of normalization ('batch' or 'layer').

    Returns:
        tf.Tensor: The output tensor of the dense block.
    """
    x = layers.Dense(
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
    )(inputs)

    if normalization == "batch":
        x = layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma)(
            x
        )
    elif normalization == "layer":
        x = layers.LayerNormalization()(x)

    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout)(x)

    return x


def conv_block(
    inputs,
    filters,
    kernel_size,
    pool_size,
    activation,
    dropout=0.1,
    normalization="batch",
    res=False,
):
    """
    Convolution building block.

    Args:
        inputs (tf.Tensor): Input tensor representing the data.
        filters (int): Number of filters in the convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
        pool_size (int): Size of the max-pooling kernel.
        activation (str): Activation function applied after convolution.
        dropout (float, optional): Dropout rate (default is 0.1).
        normalization (str, optional): Type of normalization ('batch' or 'layer').
        res (bool, optional): Whether to use residual connections (default is False).

    Returns:
        tf.Tensor: The output tensor of the convolution block.
    """
    if res:
        residual = inputs

    x = layers.Convolution1D(filters=filters, kernel_size=kernel_size, padding="valid")(
        inputs
    )
    if normalization == "batch":
        x = layers.BatchNormalization()(x)
    elif normalization == "layer":
        x = layers.LayerNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    if res:
        if filters != residual.shape[2]:
            residual = layers.Convolution1D(filters=filters, kernel_size=1, strides=1)(
                residual
            )
        x = layers.Add()([x, residual])
        x = layers.Activation(activation)(x)
        if normalization == "batch":
            x = layers.BatchNormalization()(x)
        elif normalization == "layer":
            x = layers.LayerNormalization()(x)

    return x


def activate(current, activation, verbose=False):
    """
    Apply activation function to a tensor.

    Args:
        current (tf.Tensor): Input tensor.
        activation (str): Activation function to apply.
        verbose (bool, optional): Print verbose information (default is False).

    Returns:
        tf.Tensor: Output tensor after applying activation.
    """
    if verbose:
        print("activate:", activation)

    if activation == "relu":
        current = tf.keras.layers.Activation("relu")(current)
    elif activation == "swish":
        current = tf.keras.layers.Activation("swish")(current)
    elif activation == "gelu":
        current = tf.keras.layers.Activation("gelu")(current)
    elif activation == "sigmoid":
        current = tf.keras.layers.Activation("sigmoid")(current)
    elif activation == "tanh":
        current = tf.keras.layers.Activation("tanh")(current)
    elif activation == "exponential":
        current = tf.keras.layers.Activation("exponential")(current)
    elif activation == "softplus":
        current = tf.keras.layers.Activation("softplus")(current)
    else:
        print('Unrecognized activation "%s"' % activation)

    return current


def conv_block_bs(
    inputs,
    filters=None,
    kernel_size=1,
    pool_size=1,
    batch_norm=False,
    activation="relu",
    activation_end=None,
    dropout=0,
    residual=False,
    strides=1,
    dilation_rate=1,
    l2_scale=0,
    conv_type="standard",
    w1=False,
    bn_momentum=0.99,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
    padding="same",
):
    """
    Construct a convolution block (for Basenji).

    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int, optional): Conv1D filters.
        kernel_size (int, optional): Conv1D kernel_size.
        activation (str, optional): Activation function.
        strides (int, optional): Conv1D strides.
        dilation_rate (int, optional): Conv1D dilation rate.
        l2_scale (float, optional): L2 regularization weight.
        dropout (float, optional): Dropout rate probability.
        conv_type (str, optional): Conv1D layer type.
        residual (bool, optional): Residual connection boolean.
        pool_size (int, optional): Max pool width.
        batch_norm (bool, optional): Apply batch normalization.
        bn_momentum (float, optional): BatchNorm momentum.
        bn_gamma (tf.Tensor, optional): BatchNorm gamma (defaults according to residual)
        kernel_initializer (str, optional): Convolution kernel initializer.
        padding (str, optional): Padding type.

    Returns:
        tf.Tensor: Output tensor after applying the convolution block.
    """
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = tf.keras.layers.SeparableConv1D
    elif w1:
        conv_layer = tf.keras.layers.Conv2D
    else:
        conv_layer = tf.keras.layers.Conv1D

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
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = activate(current, activation_end)

    # Pool
    if pool_size > 1:
        if w1:
            current = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding=padding)(
                current
            )
        else:
            current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )

    return current


def dilated_residual(
    inputs,
    filters,
    kernel_size=3,
    rate_mult=2,
    dropout=0,
    conv_type="standard",
    repeat=1,
    round=False,
    **kwargs,
):
    """
    Construct a residual dilated convolution block.

    Args:
        inputs (tf.Tensor): Input tensor.
        filters (int): Number of filters in the convolutional layer.
        kernel_size (int, optional): Size of the convolutional kernel.
        rate_mult (int, optional): Rate multiplier for dilated convolution.
        dropout (float, optional): Dropout rate probability.
        conv_type (str, optional): Conv1D layer type.
        repeat (int, optional): Number of times to repeat the block.
        round (bool, optional): Whether to round the dilation rate.
        **kwargs: Additional keyword arguments.

    Returns:
        tf.Tensor: Output tensor after applying the dilated residual block.
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
        current = tf.keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult
        if round:
            dilation_rate = np.round(dilation_rate)

    return current
