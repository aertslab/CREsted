"""Deeptopic CNN model architecture."""

import keras

from crested.tl.zoo.utils import conv_block, dense_block


def deeptopic_cnn(
    seq_len: int,
    num_classes: int,
    filters: int = 1024,
    first_kernel_size: int = 17,
    pool_size: int = 4,
    dense_out: int = 1024,
    first_activation: str = "gelu",
    activation: str = "relu",
    conv_do: float = 0.15,
    normalization: str = "batch",
    dense_do: float = 0.5,
    pre_dense_do: float = 0.5,
    first_kernel_l2: float = 1e-4,
    kernel_l2: float = 1e-5,
) -> keras.Model:
    """
    Construct a DeepTopicCNN model. Usually used for topic classification.

    Parameters
    ----------
    seq_len
        Width of the input region.
    num_classes
        Number of classes to predict.
    filters
        Number of filters in the first convolutional layer.
        Followed by halving in subsequent layers.
    first_kernel_size
        Size of the kernel in the first convolutional layer.
    pool_size
        Size of the pooling kernel.
    dense_out
        Number of neurons in the dense layer.
    first_activation
        Activation function for the first conv block.
    activation
        Activation function for subsequent blocks.
    conv_do
        Dropout rate for the convolutional layers.
    normalization
        Type of normalization ('batch' or 'layer').
    dense_do
        Dropout rate for the dense layers.
    pre_dense_do
        Dropout rate right before the dense layers.
    first_kernel_l2
        L2 regularization for the first convolutional layer.
    kernel_l2
        L2 regularization for the other convolutional layers.

    Returns
    -------
    A Keras model.
    """
    inputs = keras.layers.Input(shape=(seq_len, 4), name="sequence")

    x = conv_block(
        inputs,
        filters=filters,
        kernel_size=first_kernel_size,
        pool_size=pool_size,
        activation=first_activation,
        dropout=conv_do,
        conv_bias=False,
        normalization=normalization,
        res=False,
        padding="same",
        l2=first_kernel_l2,
        batchnorm_momentum=0.9,
    )
    # conv blocks without residual connections
    for _ in range(2):
        x = conv_block(
            x,
            filters=int(filters / 2),
            kernel_size=11,
            pool_size=pool_size,
            activation=activation,
            dropout=conv_do,
            conv_bias=False,
            normalization=normalization,
            res=False,
            padding="same",
            l2=kernel_l2,
            batchnorm_momentum=0.9,
        )

    # conv blocks with residual connections
    x = conv_block(
        x,
        filters=int(filters / 2),
        kernel_size=5,
        pool_size=pool_size,
        activation=activation,
        dropout=conv_do,
        conv_bias=False,
        normalization=normalization,
        res=True,
        padding="same",
        l2=kernel_l2,
        batchnorm_momentum=0.9,
    )

    x = conv_block(
        x,
        filters=int(filters / 2),
        kernel_size=2,
        pool_size=0,  # no pooling
        activation=activation,
        dropout=0,
        normalization=normalization,
        conv_bias=False,
        res=True,
        padding="same",
        l2=kernel_l2,
        batchnorm_momentum=0.9,
    )

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(pre_dense_do)(x)
    x = dense_block(
        x,
        dense_out,
        activation,
        dropout=dense_do,
        normalization=normalization,
        name_prefix="denseblock",
        use_bias=False,
    )
    logits = keras.layers.Dense(num_classes, activation="linear", use_bias=True)(x)
    outputs = keras.layers.Activation("sigmoid")(logits)
    return keras.Model(inputs=inputs, outputs=outputs)
