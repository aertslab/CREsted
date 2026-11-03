"""Chrombpnet like model architecture for peak regression."""

import keras

from crested.tl.zoo.utils import activate, normalize, pool


def dilated_cnn_decoupled(
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
    Construct a CNN using dilated convolutions with a separate dense head per output class.

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
    x = normalize(x, "batch", momentum=0.9, gamma_initializer="ones")
    x = activate(x, first_conv_activation)
    if first_conv_pool_size > 1:
        x = pool(x, "max", pool_size=first_conv_pool_size, padding="same")
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
            conv_x = normalize(
                conv_x,
                "batch",
                momentum=0.9,
                gamma_initializer="ones",
                name=f"bpnet_{layer_names[i - 1]}bn",
            )
        if activation != "none":
            conv_x = activate(
                conv_x, activation, name=f"bpnet_{layer_names[i - 1]}activation"
            )

        x_len = keras.ops.shape(x)[1]
        conv_x_len = keras.ops.shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0  # for symmetric cropping

        x = keras.layers.Cropping1D(
            (x_len - conv_x_len) // 2, name=f"bpnet_{layer_names[i - 1]}crop"
        )(x)
        x = keras.layers.add([conv_x, x])
        if dropout > 0:
            x = keras.layers.Dropout(
                dropout, name=f"bpnet_{layer_names[i - 1]}dropout"
            )(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    # Decouple the dense layers
    outputs = []
    for i in range(num_classes):
        outputs.append(
            keras.layers.Dense(
                units=1,
                activation=output_activation,  # Or your choice of activation
                name=f"dense_out_class_{i}",
                use_bias=dense_bias,
            )(x)  # `x` is the shared feature vector from previous layers
        )

    # Concatenate to restore original output shape
    final_output = keras.layers.Concatenate(axis=-1)(
        outputs
    )  # Shape: (batch_size, num_classes)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=final_output)

    return model
