"""Basenji model architecture."""

import tensorflow as tf
import tensorflow.keras.layers as layers

from crested.tl.zoo.utils import conv_block_bs, dilated_residual


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
) -> tf.keras.Model:
    """
    Construct a Basenji model.

    Parameters
    ----------
    input_shape
        Shape of the input data (sequence_length, 4).
    output_shape
        Shape of the output data (output_length, num_tasks).
    first_activation
        Activation function for the first convolutional block.
    activation
        Activation function for subsequent convolutional blocks.
    output_activation
        Activation function for the output layer.
    first_filters
        Number of filters in the first convolutional block.
    filters
        Number of filters in subsequent convolutional blocks.
    first_kernel_size
        Size of the kernel in the first convolutional block.
    kernel_size
        Size of the kernel in subsequent convolutional blocks.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras model.
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
