"""DeepPeak model definition."""

import tensorflow as tf
import tensorflow.keras.layers as layers
import keras


class ConvBlock(layers.Layer):
    """Convolution block with optional residual connection."""

    def __init__(
        self,
        filters,
        kernel_size,
        pool_size: int = 2,
        activation: str = "relu",
        l2: float = 1e-5,
        dropout: float = 0.25,
        res: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # configs
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.l2 = l2
        self.dropout = dropout
        self.res = res

        # layers
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            activation=None,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2),
        )
        self.conv_res = layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            activation=None,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2),
        )
        self.bn = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")
        self.activation_layer = layers.Activation(self.activation)
        self.maxpool = layers.MaxPool1D(pool_size=self.pool_size, padding="same")
        self.dropout_layer = layers.Dropout(self.dropout)

    def build(self, input_shape):
        # Explicitly build the primary convolution layer
        self.conv.build(input_shape)
        self.conv_res.build(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        if self.res:
            residual = inputs
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation_layer(x)

        if self.res:
            if self.filters != residual.shape[-1]:
                residual = self.conv_res(residual)
            x = tf.keras.layers.add([x, residual])

        if self.pool_size > 1 and inputs.shape[1] > self.kernel_size:
            x = self.maxpool(x)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        return x


class DeepPeak(tf.keras.Model):
    def __init__(self, num_classes: int, config: dict, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.num_classes = num_classes

        self.filter_size = self.config["filter_size"]
        self.num_filters = self.config["num_filters"]
        self.pool_size = self.config["pool_size"]
        self.activation = self.config["activation"]
        self.l2 = self.config["l2"]
        self.dropout = self.config["dropout"]
        self.res = self.config["res"]

        self.conv1 = ConvBlock(
            filters=self.num_filters,
            kernel_size=self.filter_size,
            pool_size=self.pool_size,
            activation=self.activation,
            l2=self.l2,
            dropout=0.3,
            res=False,
        )
        self.conv2a = ConvBlock(
            filters=int(self.num_filters / 2),
            kernel_size=3,
            pool_size=2,
            activation="relu",
            l2=self.l2,
            dropout=self.dropout,
            res=self.res,
        )
        self.conv2b = ConvBlock(
            filters=int(self.num_filters / 2),
            kernel_size=3,
            pool_size=2,
            activation="relu",
            l2=self.l2,
            dropout=self.dropout,
            res=self.res,
        )
        self.conv3a = ConvBlock(
            filters=int(self.num_filters / 4),
            kernel_size=3,
            pool_size=2,
            activation="relu",
            l2=self.l2,
            dropout=self.dropout,
            res=self.res,
        )
        self.conv3b = ConvBlock(
            filters=int(self.num_filters / 4),
            kernel_size=3,
            pool_size=2,
            activation="relu",
            l2=self.l2,
            dropout=self.dropout,
            res=self.res,
        )
        self.pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(
            units=self.num_classes,
            activation="linear",
            use_bias=True,
        )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pooling(x)
        x = self.dense(x)

        return x
