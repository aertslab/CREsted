"""Test whether deeppeak implementations output the same"""

from deeppeak.model import ChromBPNet
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import int_shape


import numpy as np
import tensorflow as tf
import yaml


def conv_block(
    filters,
    kernel_size,
    x,
    pool_size=2,
    activation="relu",
    l2=1e-5,
    dropout=0.25,
    res=False,
):
    if res:
        residual = x

    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        use_bias=False,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
    )(x)

    y = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(y)
    y = layers.Activation(activation)(y)
    if res:
        if filters != residual.shape[2]:
            residual = layers.Conv1D(filters=filters, kernel_size=1, strides=1)(
                residual
            )
        y = layers.add([y, residual])
    if pool_size > 1 and x.shape[1] > kernel_size:
        y = layers.MaxPool1D(pool_size=pool_size, padding="same")(y)
    if dropout > 0:
        y = layers.Dropout(dropout)(y)
    return y


def select_model(
    num_classes, n_dil_layers, num_filters, filter_size, activation, seq_shape
):
    input_ = layers.Input(shape=seq_shape)
    x = input_
    x = conv_block(int(num_filters), filter_size, x, 0, activation, 1e-5, 0.1, False)
    for i in range(1, n_dil_layers + 1):
        # dilated convolution
        conv_x = layers.Conv1D(
            num_filters,
            kernel_size=3,
            use_bias=False,
            padding="valid",
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            dilation_rate=2**i,
        )(x)
        conv_x = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")(
            conv_x
        )
        conv_x = layers.Activation("relu")(conv_x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0  # Necessary for symmetric cropping

        if num_filters != x.shape[2]:
            x = layers.Conv1D(filters=num_filters, kernel_size=1, strides=1)(x)
        x = layers.Cropping1D((x_len - conv_x_len) // 2)(x)
        x = layers.add([conv_x, x])

        x = layers.Dropout(0.1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.4)(x)
    # x = dense_block(256, 'relu', x, dropout=0.3, l2=1e-4)
    logits = layers.Dense(num_classes, activation="linear", use_bias=True)(x)

    output = logits

    return tf.keras.Model(inputs=input_, outputs=output)


# Define a random seed for reproducibility
with open("configs/user.yml", "r") as f:
    config = yaml.safe_load(f)

seed = 42
tf.random.set_seed(seed)

# Initialize both models
model_1 = select_model(19, 1, 512, 5, "gelu", (2114, 4))
model_2 = ChromBPNet(config)
assert config["first_conv_res"] is False

# Optional: Set both models to the same weights if possible
model_1.set_weights(model_2.get_weights())

# Generate some random data to simulate inputs to the models
input_data = np.random.random((1, 2114, 4)).astype(np.float32)

# Get outputs from both models
output_1 = model_1(input_data)
output_2 = model_2(input_data)

# Compare the outputs
np.testing.assert_allclose(output_1, output_2, rtol=1e-6, atol=1e-6)
print("The outputs are close enough. Models are behaving identically!")
