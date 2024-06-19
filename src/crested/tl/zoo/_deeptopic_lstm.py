"""Deeptopic LSTM model architecture."""

import pickle
import tensorflow as tf
import tensorflow.keras.layers as layers

from crested.tl.zoo.utils import get_output


def deeptopic_lstm(
    seq_len: int,
    num_classes: int,
    filters: int = 300,
    first_kernel_size: int = 30,
    max_pool_size: int = 15,
    max_pool_stride: int = 5,
    dense_out: int = 256,
    lstm_out: int = 128,
    first_activation: str = "relu",
    activation: str = "relu",
    lstm_do: float = 0.1,
    dense_do: float = 0.4,
    pre_dense_do: float = 0.2,
    motifs_path: str = None,
) -> tf.keras.Model:
    """
    Construct a DeepTopicLSTM model. Usually used for topic classification.

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
    max_pool_size
        Size of the max pooling kernel.
    max_pool_stride
        Stride of the max pooling kernel.
    dense_out
        Number of neurons in the dense layer.
    lstm_out
        Number of units in the lstm layer.
    first_activation
        Activation function for the first conv block.
    activation
        Activation function for subsequent blocks.
    lstm_do
        Dropout rate for the lstm layer.
    dense_do
        Dropout rate for the dense layers.
    pre_dense_do
        Dropout rate before the dense layers.
    motifs_path
        Path to the motif file to initialize the convolutional weights.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras model.
    """
    inputs = layers.Input(shape=(seq_len, 4), name="sequence")

    reverse_lambda_ax1 = layers.Lambda(lambda x: tf.reverse(x, [1]))
    reverse_lambda_ax2 = layers.Lambda(lambda x: tf.reverse(x, [2]))

    hidden_layers = [
        layers.Convolution1D(
            filters=filters,
            kernel_size=first_kernel_size,
            activation=first_activation,
            padding="valid",
            kernel_initializer='random_uniform'
        ),
        layers.MaxPooling1D(
            pool_size=max_pool_size,
            strides=max_pool_stride,
            padding='valid'
        ),
        layers.Dropout(pre_dense_do),
        layers.TimeDistributed(layers.Dense(lstm_out, activation=activation)),
        layers.Bidirectional(layers.LSTM(
            lstm_out,
            dropout=lstm_do,
            recurrent_dropout=lstm_do,
            return_sequences=True)
        ),
        layers.Dropout(pre_dense_do),
        layers.Flatten(),
        layers.Dense(dense_out, activation=activation),
        layers.Dropout(dense_do),
        layers.Dense(num_classes, activation='sigmoid')
    ]

    forward_output = get_output(inputs, hidden_layers)
    reverse_output = get_output(reverse_lambda_ax2(reverse_lambda_ax1(inputs)), hidden_layers)
    outputs = layers.average([forward_output, reverse_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if motifs_path is not None:
        f = open(motifs_path, "rb")
        motif_dict = pickle.load(f)
        f.close()
        conv_weights = model.layers[2].get_weights()

        for i, name in enumerate(motif_dict):
            conv_weights[0][:, :, i] = conv_weights[0][:, :, i] * 0.1
            conv_weights[0][int((30 - len(motif_dict[name])) / 2):int((30 - len(motif_dict[name])) / 2) + len(motif_dict[name]), :, i] = motif_dict[name]
        model.layers[2].set_weights(conv_weights)

    return model
