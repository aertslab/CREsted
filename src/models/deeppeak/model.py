"""DeepPeak model definition."""

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import int_shape


@tf.keras.utils.register_keras_serializable(package="ConvResBlock")
class ConvResBlock(layers.Layer):
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
        use_bias: bool = False,
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
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2),
            use_bias=use_bias,
        )
        if self.res:
            self.conv_res = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                strides=1,
                activation=None,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(self.l2),
            )
        self.bn = layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")
        self.activation_layer = layers.Activation(self.activation)
        self.maxpool = (
            layers.MaxPool1D(pool_size=self.pool_size, padding="same")
            if self.pool_size > 1
            else None
        )
        self.dropout_layer = layers.Dropout(self.dropout)

    # def build(self, input_shape):
    #     # Explicitly build the primary convolution layer

    #     self.conv.build(input_shape)
    #     if self.res:
    #         self.conv_res.build(input_shape)

    #     super().build(input_shape)

    def call(self, inputs):
        if self.res:
            residual = inputs
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation_layer(x)

        if self.res:
            if self.filters != residual.shape[-1]:
                residual = self.conv_res(residual)
            x = layers.add([x, residual])

        if self.pool_size > 1 and inputs.shape[1] > self.kernel_size:
            x = self.maxpool(x)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "activation": self.activation,
            "l2": self.l2,
            "dropout": self.dropout,
            "res": self.res,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="ChromBlock")
class ConvChromBlock(layers.Layer):
    """Convolution used in ChromBPNet."""

    def __init__(
        self,
        filters,
        kernel_size,
        activation: str = "relu",
        l2: float = 1e-5,
        dropout: float = 0.25,
        use_bias: bool = True,
        batch_norm: bool = True,
        dilation_rate: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # configs
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.l2 = l2
        self.dropout = dropout
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.dilation_rate = dilation_rate

        # layers
        self.conv = layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=1,
            activation=None,
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2),
            use_bias=self.use_bias,
            dilation_rate=self.dilation_rate,
        )

        self.bn = (
            layers.BatchNormalization(momentum=0.9, gamma_initializer="ones")
            if self.batch_norm
            else None
        )
        self.activation_layer = (
            layers.Activation(self.activation) if self.activation != "none" else None
        )
        self.dropout_layer = layers.Dropout(self.dropout) if self.dropout > 0 else None

    # def build(self, input_shape):
    #     # Explicitly build the primary convolution layer
    #     self.conv.build(input_shape)

    #     super().build(input_shape)

    def call(self, inputs):
        x = inputs
        conv_x = self.conv(inputs)
        if self.batch_norm:
            conv_x = self.bn(conv_x)
        if self.activation != "none":
            conv_x = self.activation_layer(conv_x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert (x_len - conv_x_len) % 2 == 0, "x_len - conv_x_len must be even"

        if self.num_filters != x.shape[2]:
            x = layers.Conv1D(filters=self.num_filters, kernel_size=1, strides=1)(x)
        x = layers.Cropping1D((x_len - conv_x_len) // 2)(x)
        x = layers.add([conv_x, x])
        if self.dropout > 0:
            x = self.dropout_layer(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "l2": self.l2,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "dilation_rate": self.dilation_rate,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ChromBPNet(config: dict):
    """Functioncal ChromBPNet model definition."""
    input_shape = (config["seq_len"], 4)
    num_classes = config["num_classes"]

    inputs = layers.Input(shape=input_shape)

    x = ConvResBlock(
        filters=config["first_conv_filters"],
        kernel_size=config["first_conv_filter_size"],
        pool_size=config["first_conv_pool_size"],
        activation=config["first_conv_activation"],
        l2=config["first_conv_l2"],
        dropout=config["first_conv_dropout"],
        res=config["first_conv_res"],
    )(inputs)

    for i in range(1, config["n_dil_layers"] + 1):
        x = ConvChromBlock(
            filters=config["num_filters"],
            kernel_size=config["filter_size"],
            activation=config["activation"],
            l2=config["l2"],
            dropout=config["dropout"],
            use_bias=False,
            batch_norm=config["batch_norm"],
            dilation_rate=2**i,
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(
        units=num_classes,
        activation="linear",
        use_bias=config["dense_bias"],
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
