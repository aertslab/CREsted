"""DeepPeak model definition."""

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import int_shape


def bpnet(config: dict):
    """
    Construct a ChromBPNet model.

    Args:
        config (dict): A dictionary containing the model configuration.
            Keys that should be present: `seq_len`, `num_classes`, `first_conv_filters`,
            `first_conv_filter_size`, `first_conv_pool_size`, `first_conv_activation`,
            `first_conv_l2`, `first_conv_dropout`, `n_dil_layers`,
            `num_filters`, `filter_size`, `activation`, `l2`, `dropout`, `batch_norm`,
            `dense_bias`.

    Returns:
        tf.keras.Model: A TensorFlow Keras model.
    """
    # Configs
    seq_len = config["seq_len"]

    num_classes = config["num_classes"]
    first_conv_filters = config["first_conv_filters"]
    first_conv_filter_size = config["first_conv_filter_size"]
    first_conv_pool_size = config["first_conv_pool_size"]
    first_conv_activation = config["first_conv_activation"]
    first_conv_l2 = config["first_conv_l2"]
    first_conv_dropout = config["first_conv_dropout"]

    n_dil_layers = config["n_dil_layers"]
    num_filters = config["num_filters"]
    filter_size = config["filter_size"]
    activation = config["activation"]
    l2 = config["l2"]
    dropout = config["dropout"]
    batch_norm = config["batch_norm"]
    dense_bias = config["dense_bias"]

    # Model
    inputs = layers.Input(shape=(seq_len, 4), name="sequence")

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
            use_bias=True,
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
    outputs = layers.Dense(num_classes, activation="linear", use_bias=dense_bias)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
