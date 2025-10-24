"""Chrombp net like model architecture for peak regression."""

import warnings

import keras


def chrombpnet(*args, **kwargs):
    """
    Use dilated_cnn instead.

    :meta private:
    """
    warnings.warn(
        "'chrombpnet' is deprecated and will be removed in a future release. "
        "Use its new name 'dilated_cnn' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dilated_cnn(*args, **kwargs)


def dilated_cnn(
    seq_len: int,
    num_classes: int,
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
    output_activation: str = "softplus",
    l2: float = 0.00001,
    dropout: float = 0.1,
    batch_norm: bool = True,
    dense_bias: bool = True,
) -> keras.Model:
    """
    Construct a CNN using dilated convolutions.

    This architecture is based on the ChromBPNet model described in :cite:`Pampari_Bias_factorized_base-resolution_2023`.
    This was renamed to DilatedCNN to avoid confusion with the original ChromBPNet framework.

    Parameters
    ----------
    seq_len
        Width of the input region.
    num_classes
        Number of classes to predict.
    first_conv_filters
        Number of filters in the first convolutional layer.
    first_conv_filter_size
        Size of the kernel in the first convolutional layer.
    first_conv_pool_size
        Size of the pooling kernel in the first convolutional layer.
    first_conv_activation
        Activation function in the first convolutional layer.
    first_conv_l2
        L2 regularization for the first convolutional layer.
    first_conv_dropout
        Dropout rate for the first convolutional layer.
    n_dil_layers
        Number of dilated convolutional layers.
    num_filters
        Number of filters in the dilated convolutional layers.
    filter_size
        Size of the kernel in the dilated convolutional layers.
    activation
        Activation function in the dilated convolutional layers.
    output_activation
        Activation function for the output layer.
    l2
        L2 regularization for the dilated convolutional layers.
    dropout
        Dropout rate for the dilated convolutional layers.
    batch_norm
        Whether or not to use batch normalization.
    dense_bias
        Whether or not to add a bias to the dense layer.

    Returns
    -------
    A Keras model.
    """
    # Model
    inputs = keras.layers.Input(shape=(seq_len, 4), name="sequence")

    # Convolutional block without dilation
    x = keras.layers.Conv1D(
        filters=first_conv_filters,
        kernel_size=first_conv_filter_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(first_conv_l2),
        use_bias=False,
    )(inputs)
    x = keras.layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(x)
    x = keras.layers.Activation(first_conv_activation)(x)
    if first_conv_pool_size > 1:
        x = keras.layers.MaxPooling1D(pool_size=first_conv_pool_size, padding="same")(x)
    x = keras.layers.Dropout(first_conv_dropout)(x)

    # Dilated convolutions
    layer_names = [str(i) for i in range(1, n_dil_layers + 1)]

    for i in range(1, n_dil_layers + 1):
        conv_layer_name = f"bpnet_{layer_names[i - 1]}conv"
        conv_x = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=filter_size,
            strides=1,
            activation=None,
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2),
            use_bias=False,
            dilation_rate=2**i,
            name=conv_layer_name,
        )(x)
        if batch_norm:
            conv_x = keras.layers.BatchNormalization(
                momentum=0.9,
                gamma_initializer="ones",
                name=f"bpnet_{layer_names[i - 1]}bn",
            )(conv_x)
        if activation != "none":
            conv_x = keras.layers.Activation(
                activation, name=f"bpnet_{layer_names[i - 1]}activation"
            )(conv_x)

        x_len = keras.ops.shape(x)[1]
        conv_x_len = keras.ops.shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0  # for symmetric cropping

        x = keras.layers.Cropping1D(
            (x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i - 1]}crop"
        )(x)
        x = keras.layers.add([conv_x, x])
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"bpnet_{layer_names[i-1]}dropout")(
                x
            )

    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(
        units=num_classes,
        activation=output_activation,
        use_bias=dense_bias,
        name="dense_out",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
