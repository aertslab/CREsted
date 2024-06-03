"""Chrombp net like model architecture for peak regression."""

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import int_shape


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
) -> tf.keras.Model:
    """
    Construct a ChromBPNet like model.

    Parameters
    ----------
    input_shape
        Shape of the input sequence.
    output_shape
        Shape of the output data.
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
    tf.keras.Model
        A TensorFlow Keras model.
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
        conv_layer_name = f"bpnet_{layer_names[i - 1]}conv"
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
        units=output_shape[-1], activation="softplus", use_bias=dense_bias
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
