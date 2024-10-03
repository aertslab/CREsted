"""Helper layers for zoo models."""

from __future__ import annotations

import keras
import numpy as np

__all__ = [
    "dense_block",
    "conv_block",
    "activate",
    "get_output",
    "conv_block_bs",
    "dilated_residual",
]


def dense_block(
    inputs: keras.KerasTensor,
    units: int,
    activation: str,
    dropout: float = 0.5,
    l2: float = 1e-5,
    bn_gamma: str | None = None,
    bn_momentum: float = 0.90,
    normalization: str = "batch",
    name_prefix: str | None = None,
    use_bias: bool = True,
):
    """
    Dense building block.

    Parameters
    ----------
    inputs
        Input tensor.
    units
        Number of units in the dense layer.
    activation
        Activation function applied after dense layer.
    dropout
        Dropout rate (default is 0.5).
    l2
        L2 regularization weight (default is 1e-5).
    bn_gamma
        Gamma initializer for batch normalization.
    bn_momentum
        Batch normalization momentum (default is 0.90).
    normalization
        Type of normalization ('batch' or 'layer').
    name_prefix
        Prefix for layer names.
    use_bias
        Whether to use bias in the dense layer (default is True).

    Returns
    -------
    The output tensor of the dense block.
    """
    x = keras.layers.Dense(
        units,
        activation=None,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2),
        name=name_prefix + "_dense" if name_prefix else None,
    )(inputs)

    if normalization == "batch":
        x = keras.layers.BatchNormalization(
            momentum=bn_momentum,
            gamma_initializer=bn_gamma,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization(
            name=name_prefix + "_layernorm" if name_prefix else None
        )(x)

    x = keras.layers.Activation(
        activation, name=name_prefix + "_activation" if name_prefix else None
    )(x)
    x = keras.layers.Dropout(
        dropout, name=name_prefix + "_dropout" if name_prefix else None
    )(x)
    return x


def conv_block(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int,
    pool_size: int,
    activation: str,
    conv_bias: bool = True,
    dropout: float = 0.1,
    normalization: str = "batch",
    res: bool = False,
    padding: str = "valid",
    l2: float = 1e-5,
    batchnorm_momentum: float = 0.99,
) -> keras.KerasTensor:
    """
    Convolution building block.

    Parameters
    ----------
    inputs
        Input tensor representing the data.
    filters
        Number of filters in the convolutional layer.
    kernel_size
        Size of the convolutional kernel.
    pool_size
        Size of the max-pooling kernel.
    activation
        Activation function applied after convolution.
    conv_bias
        Whether to use bias in the convolutional layer (default is True).
    dropout
        Dropout rate (default is 0.1).
    normalization
        Type of normalization ('batch' or 'layer').
    res
        Whether to use residual connections (default is False).
    padding
        Padding type for the convolutional layer (default is "valid").
    l2
        L2 regularization weight (default is 1e-5).
    batchnorm_momentum
        Batch normalization momentum (default is 0.99).

    Returns
    -------
    The output tensor of the convolution block.
    """
    if res:
        residual = inputs

    x = keras.layers.Convolution1D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_regularizer=keras.regularizers.L2(l2),
        use_bias=conv_bias,
    )(inputs)
    if normalization == "batch":
        x = keras.layers.BatchNormalization(momentum=batchnorm_momentum)(x)
    elif normalization == "layer":
        x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    if res:
        if filters != residual.shape[2]:
            residual = keras.layers.Convolution1D(
                filters=filters, kernel_size=1, strides=1
            )(residual)
        x = keras.layers.Add()([x, residual])

    if pool_size > 1:
        x = keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding)(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)

    return x


def activate(
    current: keras.KerasTensor, activation: str, verbose: bool = False
) -> keras.KerasTensor:
    """
    Apply activation function to a tensor.

    Parameters
    ----------
    current
        Input tensor.
    activation
        Activation function to apply.
    verbose
        Print verbose information (default is False).

    Returns
    -------
    Output tensor after applying activation.
    """
    if verbose:
        print("activate:", activation)

    if activation == "relu":
        current = keras.layers.Activation("relu")(current)
    elif activation == "swish":
        current = keras.layers.Activation("swish")(current)
    elif activation == "gelu":
        current = keras.layers.Activation("gelu")(current)
    elif activation == "sigmoid":
        current = keras.layers.Activation("sigmoid")(current)
    elif activation == "tanh":
        current = keras.layers.Activation("tanh")(current)
    elif activation == "exponential":
        current = keras.layers.Activation("exponential")(current)
    elif activation == "softplus":
        current = keras.layers.Activation("softplus")(current)
    elif activation == "gelu_enf":
        # Very simple gelu approximation, used in Enformer, so needed to get equivalent results.
        current = keras.layers.Activation(lambda x: keras.ops.sigmoid(1.702 * x) * x)(current)
    else:
        print(f'Unrecognized activation "{activation}"')

    return current

def get_output(input_layer, hidden_layers):
    """
    Pass input layer through hidden layers.

    Parameters
    ----------
    input_layer
        Input layer.
    hidden_layers
        Hidden layers.

    Returns
    -------
    keras.KerasTensor
        Output tensor after passing through all hidden layers.
    """
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)

    return output


def conv_block_bs(
    inputs,
    filters: int | None = None,
    kernel_size: int = 1,
    pool_size: int = 1,
    batch_norm: bool = False,
    activation: str = "relu",
    activation_end: str | None = None,
    dropout: float = 0,
    residual: bool = False,
    strides: int = 1,
    dilation_rate: int = 1,
    l2_scale: float = 0,
    conv_type: str = "standard",
    pool_type: str = "max",
    bn_momentum: float = 0.99,
    bn_gamma: keras.KerasTensor | None = None,
    bn_type: str = "standard",
    kernel_initializer: str = "he_normal",
    padding: str = "same",
    name_prefix: str | None = None
):
    """
    Construct a convolution block (for Basenji/Enformer).

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Conv1D filters.
    kernel_size
        Conv1D kernel_size.
    pool_size
        Max pool width.
    batch_norm
        Apply batch normalization.
    activation
        Activation function.
    activation_end
        Activation to add at end of block, after residual and before pooling.
        Optional, can be None to disable.
    dropout
        Dropout rate probability.
    residual
        Residual connection boolean.  
    strides
        Conv1D strides.
    dilation_rate
        Conv1D dilation rate.
    l2_scale
        L2 regularization weight.
    conv_type
        Conv1D layer type. 
    pool_type
        Pooling type. Either 'max' or 'attention'.
    bn_momentum
        BatchNorm momentum.
    bn_gamma
        BatchNorm gamma (defaults according to residual).
    kernel_initializer
        Convolution kernel initializer.
    padding
        Padding type.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the convolution block.
    """
    current = inputs

    # choose convolution type
    if conv_type == "separable":
        conv_layer = keras.layers.SeparableConv1D
    else:
        conv_layer = keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale),
        name = name_prefix + "_conv" if name_prefix else None
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_sync = True 
        current = keras.layers.BatchNormalization(
            momentum=bn_momentum, 
            gamma_initializer=bn_gamma, 
            synchronized=bn_sync,
            name = name_prefix + "_bnorm" if name_prefix else None
            )(current)

    # dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = activate(current, activation_end)

    # Pool
    if pool_size > 1:
        if pool_type == "max":
            current = keras.layers.MaxPool1D(pool_size=pool_size, padding=padding)(
                current
            )
        elif pool_type == "attention":
            current = AttentionPool1D(pool_size=pool_size, padding=padding)(
                current
            )
        else:
            raise ValueError(f"Unrecognised pooling function {pool_type}. Use 'max' or 'attention'.")

    return current

def mha_block_enf(
    inputs,
    num_heads: int,
    key_query_dim: int,
    value_dim: int,
    scaling: bool = True,
    attn_dropout: float = 0.05,
    pos_dropout: float = 0.01,
    final_dropout: float = 0.4,
    symmetric_pos_encoding: bool = False,
    pos_encoding_funs: str = 'all',
    pos_encoding_abs: bool = True,
    num_pos_feats: int | None = None,
    zero_init = True,
    residual = True,
    name_prefix: str | None = None):
    """
    Construct a MHA block (for Enformer), consisting of Residual(LayerNorm+MHSelfAttention+Dropout).
    
    Parameters
    ----------
    inputs
        Input tensor.
    num_heads
        Number of attention heads to use.
    key_query_dim
        Number of k (key) and q (query) dimensions in the attention mechanism.
    value_dim
        Number of v (value) dimensions in the attention mechanism.
    scaling
        Whether to use scaling.
    attn_dropout
        Attention dropout rate.
    pos_dropout
        Positional embedding dropout rate.
    final_dropout
        Block-included post-MHA dropout rate.
    symmetric_pos_encoding
        Whether to make positional encodings symmetric. Only relevant if pos_encoding = True.
    pos_encoding_funs
        Can be 'all' or 'central_mask'.
        Enformer default is 'all', using exponential+central_mask+gamma.
    pos_encoding_abs
        Whether to use the absolute of values before calculating the relative position encoding.
    num_pos_feats
        Number of positional features. If not supplied, calculated from value_dim and number of position encoding functions.
        Min 6 for default relative_position_functions, min 12 for positional_features_sin_cos.
    zero_init
        Whether to initialize MHA from zero.
    residual
        Whether to wrap the entire block in residual structure.
    name_prefix
        Prefix for layer names.
    
    Returns
    -------
    Output tensor after applying the MHA block.
    """

    # MHA block
    current = keras.layers.LayerNormalization(
        epsilon = 1e-05, 
        center = True, 
        scale = True, 
        beta_initializer = "zeros", 
        gamma_initializer = "ones", 
        name=f'{name_prefix}_lnorm'
    )(inputs)
    # current = MHSelfAttention(
    #     num_heads = num_heads,
    #     query_dim = query_dim,
    #     value_dim = value_dim,
    #     num_pos_feats = num_pos_feats,
    #     scaling = scaling,
    #     attn_dropout_rate = attn_dropout,
    #     pos_dropout_rate = pos_dropout,
    #     pos_encoding = pos_encoding,
    #     symmetric_pos_encoding = symmetric_pos_encoding,
    #     pos_encoding_funs = pos_encoding_funs,
    #     zero_init = zero_init,
    #     name = f"{name_prefix}_mhsa"
    # )(current)
    current = MultiheadAttention(
        value_size = value_dim,
        key_size = key_query_dim,
        heads = num_heads,
        scaling = scaling,
        attention_dropout_rate = attn_dropout,
        relative_position_symmetric = symmetric_pos_encoding,
        relative_position_functions = pos_encoding_funs,
        relative_position_absolute = pos_encoding_abs,
        num_position_features = num_pos_feats,
        positional_dropout_rate = pos_dropout,
        content_position_bias = True,
        zero_initialize = zero_init,
        initializer = "he_normal", # Unsure in Enf, think this is fine.
        l2_scale = 1.0e-8, # Doesn't seem to be set in Enf, is set in Borzoi.
        transpose_stride = 0,
        gated = False,
        qkv_width = 1,
        name = f"{name_prefix}_mhsa"
    )(current)
    current = keras.layers.Dropout(
        rate = final_dropout, 
        name = f"{name_prefix}_dropout"
    )(current)
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current

def ffn_block_enf(
        inputs,
        filters: int,
        expansion_rate: int = 2,
        dropout: int = 0.4,
        activation: str = 'relu',
        residual: bool = True,
        name_prefix: str | None = None
):
    """"
    Construct a feedforward block (for Enformer), consisting of Residual(LayerNorm+PointwiseConv+Dropout+ReLU+PointwiseConv+Dropout).

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Pointwise convolution filters.
    expansion_rate
        Scaling factor of base filters inside the FFN. 
    dropout
        Dropout rate.
    activation
        Which activation function to use.
    residual
        Whether to wrap the entire block in residual structure.  
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the feedforward block.
    """
    expansion_filters = int(expansion_rate * filters)

    # First half
    current = keras.layers.LayerNormalization(
        epsilon=1e-05, 
        center=True, 
        scale=True, 
        beta_initializer="zeros", 
        gamma_initializer="ones", 
        name=f'{name_prefix}_lnorm'
    )(inputs)
    current = keras.layers.Conv1D(filters=expansion_filters, kernel_size=1, name=f'{name_prefix}_pointwise_1')(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_1")(current)

    # Second half
    current = activate(current, activation)
    current = keras.layers.Conv1D(filters=filters, kernel_size=1, name=f'{name_prefix}_pointwise_2')(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_2")(current)

    # Residual
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current


# Attention pooling layer
class AttentionPool1D(keras.layers.Layer):
    """Pooling operation with optional weights."""
    def __init__(self,
                pool_size: int = 2,
                per_channel: bool = True,
                w_init_scale: float = 2.0,
                strides = None,
                padding = None,
                data_format = None,
                name: str = "AttentionPool1D",
                **kwargs):
        """AttentionPool from the FastISM repository.
        Softmax pooling.
        Args:
        pool_size: Pooling size, same as in Max/AvgPooling.
        per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
         w_init_scale: Initialisation of w. When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.  
        strides/padding/data_format: placeholder arguments to capture them from from_config. 
            Not used in setting up the layer.
        name: Module name.
        """
        super().__init__(name = name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale

        # Needed for compatibility with FastISM, not actually used to configure
        # self._strides = self._pool_size
        # self._padding = "valid" # ensure it behaves like MaxPooling1D with valid padding
        # self._data_format = "channels_last"

    def build(self, inputs_shape):
        # Construct learnable layer part
        # Put in build to have access to inputs_shape automatically
        num_features = inputs_shape[-1]
        output_size = num_features if self._per_channel else 1
        self.w = self.add_weight(
            shape=(num_features, output_size),
            initializer="random_normal",
            trainable=True,
            name = 'att_pool_weight'
        )  
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self._pool_size,
            "per_channel": self._per_channel,
            "w_init_scale": self._w_init_scale,
            "strides": self._strides,
            "padding": self._padding,
            "data_format": self._data_format
        })
        return config
    
    # @tf.function(jit_compile=True)
    def call(self, inputs, training = False):
        _, length, num_features = inputs.shape
        
        if length == None: # this can happen at when creating fast_ism_model
            return inputs # don't do anything for now
            
        inputs = keras.ops.reshape(
            inputs,
            (-1, length // self._pool_size, self._pool_size, num_features))
        return keras.ops.reduce_sum(
            inputs * keras.ops.softmax(keras.ops.matmul(inputs, self.w), axis=-2),
            axis=-2)

# Multi-head attention block from baskerville
class MultiheadAttention(keras.layers.Layer):
    """Multi-head attention."""

    def __init__(
        self,
        value_size,
        key_size,
        heads,
        scaling: bool = True,
        attention_dropout_rate=0,
        relative_position_symmetric=False,
        relative_position_functions: str = "central_mask",
        relative_position_absolute: bool = False,
        num_position_features=None,
        positional_dropout_rate=0,
        content_position_bias=True,
        zero_initialize=True,
        initializer="he_normal",
        l2_scale=0,
        transpose_stride=0,
        gated=False,
        qkv_width=1,
        name: str = 'mhsa',
        **kwargs
    ):
        """Creates a MultiheadAttention module.
           Original version written by Ziga Avsec.

        Parameters:
        ----------
        value_size
            The size of each value embedding per head.
        key_size
            The size of each key and query embedding per head.
        heads
            The number of independent queries per timestep.
        scaling
            Whether to scale the attention logits.
        attention_dropout_rate
            Dropout rate for attention logits.
        relative_position_symmetric
            If True, the symmetric version of basis functions will be used. 
            If False, a symmetric and asymmetric versions will be used.
        relative_position_functions
            Relative position functions to use. 
            Can be 'all' or 'central_mask'.
            Enformer default is 'all', using exponential+central_mask+gamma.
            Borzoi default is 'central_mask'.
        relative_position_absolute
            Whether to use the absolute of values before calculating the relative position encoding.
        num_position_features
            Number of relative positional features to compute. 
            If None, `value_size * num_heads` is used.
        positional_dropout_rate: Dropout rate for the positional encodings if
            relative positions are used.
        content_position_bias
            Whether to add shifted relative logits to content logits.
            Default in both Enformer and Borzoi.
        zero_initialize: 
            if True, the final linear layer will be 0 initialized.
        initializer: 
            Initializer for the projection layers. If unspecified,
            VarianceScaling is used with scale = 2.0.

        Unused arguments in Enformer/Borzoi:
        transpose_stride, gated, qkv_width.
        """
        super().__init__(name = name, **kwargs)
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._gated = gated
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        if num_position_features is None:
            # num_position_features needs to be divisible by the number of
            # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_position_features = (
                self._value_size // divisible_by
            ) * divisible_by
        else:
            self._num_position_features = num_position_features
        self._positional_dropout_rate = positional_dropout_rate
        self._content_position_bias = content_position_bias
        self._l2_scale = l2_scale
        self._initializer = initializer

        key_proj_size = self._key_size * self._num_heads
        embedding_size = self._value_size * self._num_heads

        self.relative_position_func = get_position_features_func(
            relative_position_functions=relative_position_functions, 
            absolute=relative_position_absolute)

        if qkv_width == 1:
            # standard dense layers
            self._q_layer = keras.layers.Dense(
                key_proj_size,
                name="q_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._k_layer = keras.layers.Dense(
                key_proj_size,
                name="k_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._v_layer = keras.layers.Dense(
                embedding_size,
                name="v_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
        else:
            # CvT separable convolutions
            self._q_layer = keras.layers.SeparableConv1D(
                key_proj_size,
                kernel_size=qkv_width,
                padding="same",
                name="q_layer",
                use_bias=False,
                depthwise_regularizer=keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._k_layer = keras.layers.SeparableConv1D(
                key_proj_size,
                kernel_size=qkv_width,
                padding="same",
                name="k_layer",
                use_bias=False,
                depthwise_regularizer=keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._v_layer = keras.layers.SeparableConv1D(
                embedding_size,
                kernel_size=qkv_width,
                padding="same",
                name="v_layer",
                use_bias=False,
                depthwise_regularizer=keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )

        if self._gated:
            self._gate_layer = keras.layers.Dense(
                embedding_size,
                activation="activation",
                name="gate",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )

        w_init = keras.initializers.Zeros() if zero_initialize else self._initializer
        if transpose_stride > 0:
            self._embedding_layer = keras.layers.Conv1DTranspose(
                filters=embedding_size,
                kernel_size=3,
                strides=transpose_stride,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )
        else:
            self._embedding_layer = keras.layers.Dense(
                embedding_size,
                name="embedding_layer",
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )

        # Create relative position layers
        self._r_k_layer = keras.layers.Dense(
            key_proj_size,
            name="r_k_layer",
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self._l2_scale),
            kernel_initializer=self._initializer,
        )
        self._r_w_bias = self.add_weight(
            "%s/r_w_bias" % self.name,
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype="float32",
        )
        self._r_r_bias = self.add_weight(
            "%s/r_r_bias" % self.name,
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype="float32",
        )

    def _multihead_output(self, linear_layer, inputs):
        """Applies a standard linear to inputs and returns multihead output."""
        output = linear_layer(inputs)  # [B, T, H * KV]
        _, seq_len, num_channels = output.shape

        # Split H * Channels into separate axes.
        num_kv_channels = num_channels // self._num_heads
        output = keras.ops.reshape(
            output, shape=[-1, seq_len, self._num_heads, num_kv_channels]
        )
        # [B, T, H, KV] -> [B, H, T, KV]
        return keras.ops.transpose(output, [0, 2, 1, 3])

    def call(self, inputs, training=False):
        # Initialise the projection layers.
        embedding_size = self._value_size * self._num_heads
        seq_len = inputs.shape[1]

        # Compute q, k and v as multi-headed projections of the inputs.
        q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K]
        k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K]
        v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V]

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size**-0.5

        # [B, H, T', T]
        content_logits = keras.ops.matmul(q + self._r_w_bias, k, transpose_b=True)

        if self._num_position_features == 0:
            logits = content_logits
        else:
            # Project positions to form relative keys.
            distances = keras.ops.expand_dims(keras.ops.arange(-seq_len + 1, seq_len, dtype="float32"), axis = -1)
            positional_encodings = self.relative_position_func(
                positions=distances,
                feature_size=self._num_position_features,
                seq_length=seq_len,
                symmetric=self._relative_position_symmetric,
            )
            # [1, 2T-1, Cr]

            if training:
                positional_encodings = keras.random.dropout(
                    positional_encodings, rate=self._positional_dropout_rate
                )

            # [1, H, 2T-1, K]
            r_k = self._multihead_output(self._r_k_layer, positional_encodings)

            # Add shifted relative logits to content logits.
            if self._content_position_bias:
                # [B, H, T', 2T-1]
                relative_logits = keras.ops.matmul(q + self._r_r_bias, r_k, transpose_b=True)
            else:
                # [1, H, 1, 2T-1]
                relative_logits = keras.ops.matmul(self._r_r_bias, r_k, transpose_b=True)
                # [1, H, T', 2T-1]
                relative_logits = keras.ops.broadcast_to(
                    relative_logits,
                    shape=(1, self._num_heads, seq_len, 2 * seq_len - 1),
                )

            #  [B, H, T', T]
            relative_logits = relative_shift(relative_logits)
            logits = content_logits + relative_logits

        # softmax across length
        weights = keras.ops.softmax(logits)

        # Dropout on the attention weights.
        if training:
            weights = keras.random.dropout(weights, rate=self._attention_dropout_rate)

        # Transpose and reshape the output.
        output = keras.ops.matmul(weights, v)  # [B, H, T', V]
        output_transpose = keras.ops.transpose(output, [0, 2, 1, 3])  # [B, T', H, V]
        attended_inputs = keras.ops.reshape(
            output_transpose, shape=[-1, seq_len, embedding_size]
        )

        # Gate
        if self._gated:
            attended_inputs = self._gate_layer(attended_inputs)

        # Final linear layer
        output = self._embedding_layer(attended_inputs)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"value_size": self._value_size, "key_size": self._key_size})
        return config



# Multi-head self-attention layer - from Enformer
class MHSelfAttention(keras.layers.Layer):
    def __init__(self, 
                 query_dim: int, 
                 value_dim: int, 
                 num_heads: int, 
                 scaling: bool = True, 
                 attn_dropout_rate: float = 0.1, 
                 pos_dropout_rate: float = 0.1, 
                 pos_encoding: bool = False, 
                 symmetric_pos_encoding: bool = False, 
                 pos_encoding_funs: list[str] = ['pos_feats_exponential', 'pos_feats_central_mask', 'pos_feats_gamma'], 
                 num_pos_feats: int | None = None, 
                 zero_init: bool = True, 
                 initializer: keras.initializers.Initializer | None = None, 
                 name: str = 'mhsa', 
                 **kwargs):
        """Creates a MultiheadAttention module.

        Args:
        name: Name of module.
        """
        super(MHSelfAttention, self).__init__(name = name, **kwargs)

        # Save parameters
        self._QK_dim = query_dim    # number of features of query and key matrices
        self._V_dim = value_dim     # number of features of the value matrix
        self._num_heads = num_heads # number of heads
        self._scaling = scaling
        self._attn_dropout = attn_dropout_rate
        self._pos_dropout = pos_dropout_rate
        self._pos_encoding = pos_encoding
        self._symmetric_pos_encoding = symmetric_pos_encoding
        self._pos_encoding_funs = pos_encoding_funs
        self._num_pos_feats = num_pos_feats
        self._zero_init = zero_init
        self._initializer = initializer

        # Use default functions if None is passed  
        if self._pos_encoding_funs is None:
            self._pos_encoding_funs = ['pos_feats_exponential',
                                       'pos_feats_central_mask',
                                       'pos_feats_gamma']
        if num_pos_feats is None:
            # pos_feats needs to be divisible by the number of
            # relative positional functions*2 (for symmetric & asymmetric version)
            divisible_by = 2*len(self._pos_encoding_funs)
            self._num_pos_feats = ((self._V_dim//divisible_by)*divisible_by)
        else:
            self._num_pos_feats = num_pos_feats
        
        if initializer is None:
            self._initializer = keras.initializers.VarianceScaling(scale=2.0)
        else:
            self._initializer = initializer

        # initializer for the embeddings
        self._w_init = keras.initializers.Zeros() if zero_init else self._initializer

        # number of features of the query/key matrix (_QK_size) multi-head projected (*_num_heads)
        # H*(Q|K)==512
        self._QK_proj_dim = self._QK_dim*self._num_heads
        # number of features of the value matrix (_V_size) multi-head projected (*_num_heads)
        # H*V==1536
        self._V_proj_dim = self._V_dim*self._num_heads
        
    def build(self, input_shape):
        # Input 
        # shape: [B, T, C] = [batch, 1536, 1536]
        # B batch, T num sequence bins, C input features/channels

        # query calculation layer
        # shape: [C, H*(Q|K)] = [1536, 512]
        # C num input features, H*(Q|K) key_proj_size = heads * key/query mat features
        self._Q_w = self.add_weight(name = 'Q_kernel', 
                                     shape = (input_shape[-1], self._QK_proj_dim), 
                                     dtype = "float32",
                                     initializer = self._initializer)
        # key calculation layer
        # shape: [C, H*(Q|K)] = [1536, 512]
        self._K_w = self.add_weight(name = 'K_kernel', 
                                     shape = (input_shape[-1], self._QK_proj_dim), 
                                     dtype = "float32",
                                     initializer = self._initializer)
        # value calculation layer
        # shape: [C, H*V] = [1536, 1536]
        # C num input features, H*V embedding_size
        self._V_w = self.add_weight(name = 'V_kernel', 
                                     shape = (input_shape[-1], self._V_proj_dim), 
                                     dtype = "float32",
                                     initializer = self._initializer)
        
        # embedding layer
        # shape: [C, H*V] = [1536, 1536]
        self._out_w = self.add_weight(name = 'out_kernel',
                                      shape = (input_shape[-1], self._V_proj_dim),
                                      dtype = "float32",
                                      initializer = self._w_init)
        # shape: [H*V]
        self._out_b = self.add_weight(name = 'out_bias',
                                      shape = self._V_proj_dim,
                                      dtype = "float32",
                                      initializer = keras.initializers.Zeros)
        
        # create additional layers if using relative positions
        if self._pos_encoding:
            # shape: [C//H, H*(Q|K)] = [192, 512]
            # C num input features, H heads, Q|K key/query mat features
            self._rel_K_w = self.add_weight(name = 'rel_K_kernel',
                                            shape = (self._num_pos_feats, self._QK_proj_dim),
                                            dtype = "float32",
                                            initializer = self._initializer)
            # shape: [1, H, 1, (Q|K)] = [1, 8, 1, 64]
            self._r_w_bias = self.add_weight(name = f'r_w_bias', 
                                             shape = (1, self._num_heads, 1, self._QK_dim), 
                                             dtype = "float32",
                                             initializer = self._initializer)
            # shape: [1, H, 1, (Q|K)] = [1, 8, 1, 64]
            self._r_r_bias = self.add_weight(name = f'r_r_bias',
                                             shape = (1, self._num_heads, 1, self._QK_dim),
                                             dtype = "float32",
                                             initializer = self._initializer)
    
    def get_config(self):
        config = super().get_config()
        config.update({"query_dim": self._QK_dim,
                       "value_dim": self._V_dim,
                       "num_heads": self._num_heads,
                       "scaling": self._scaling,
                       "attn_dropout_rate": self._attn_dropout,
                       "pos_dropout_rate": self._pos_dropout,
                       "pos_encoding": self._pos_encoding,
                       "symmetric_pos_encoding": self._symmetric_pos_encoding,
                       "pos_encoding_funs": self._pos_encoding_funs,
                       "num_pos_feats": self._num_pos_feats,
                       "zero_init": self._zero_init,
                       "initializer": self._initializer})
        return config
    
    def _multihead_output(self, weight, inputs):
        """Applies a standard matmul (linear layer w/o bias) to inputs and returns multihead output."""
        # apply layer on inputs in batches
        # output shape:[B, T, H*(Q|K) or H*V]
        # B batch size, T sequence length, H*(Q|K) QK_proj_dim or H*V V_proj_dim
        output = keras.ops.matmul(inputs, weight)
        # T sequence length
        seq_len = output.shape[-2]
        # number of features of the query/key matrix (_QK_dim) or value matrix (_V_dim) before projecting across heads
        # depending on whether Q, K or V input is passed
        # (Q|K) or V
        QKV_dim = output.shape[-1]//self._num_heads
        # split heads (H) * channels (H/Q or V) into separate axes
        # output shape:[B, T, H, (Q|K) or V]
        multihead_out = keras.ops.reshape(output, shape=(-1, seq_len, self._num_heads, QKV_dim))
        
        # shape:[B, T, H, (Q|K) or V] -> shape:[B, H, T, (Q|K) or V]
        # B batch size, T sequence length, H _num_heads, *(Q|K) _key_size or V _value_size
        return keras.ops.transpose(multihead_out, [0, 2, 1, 3])
    
    def call(self, inputs, training=False):
        # input sequence length
        seq_len = keras.ops.cast(inputs.shape[1], "float32")
        # compute a multi-headed projection of Q based on the inputs
        # output shape:[B, H, T, (Q|K)] confirmed shape:[1, 8, 1536, 64]
        Q = self._multihead_output(self._Q_w, inputs)
        # compute a multi-headed projection of K based on the inputs
        # output shape:[B, H, T, (Q|K)] confirmed shape:[1, 8, 1536, 64]
        K = self._multihead_output(self._K_w, inputs)
        # compute a multi-headed projection of V based on the inputs
        # output shape:[B, H, T, V] confirmed shape:[1, 8, 1536, 192]
        V = self._multihead_output(self._V_w, inputs)
        # scale the query by the square-root of query/key size
        # for some reason only scale the query and not both query and key
        if self._scaling:
            Q *= self._QK_dim**-0.5
        
        if self._pos_encoding:
            # project positions to form relative keys (seq_len*2)
            distances = keras.ops.expand_dims(keras.ops.arange(1-seq_len, seq_len, dtype="float32"), axis=-1)
            # Positional encodings output: [B, 2T-1, C//H] = [1, 3071, 192]
            # 2T-1 = Relative keys, C//H = num pos feats
            
            positional_encodings = pos_feats_all(positions = distances,
                                                 feature_size = self._num_pos_feats,
                                                 seq_length = seq_len,
                                                 feature_functions = self._pos_encoding_funs,
                                                 symmetric = self._symmetric_pos_encoding)
            # positional encoding DROPOUT
            if training:
                # positional_encodings.shape:[B, 2T-1, C//H] confirmed ([1, 3071, 192])
                positional_encodings = keras.layers.Dropout(rate=self._pos_dropout, name='pos_drop')(positional_encodings)
            
            # r_K output shape: [B, H, 2T-1, (Q|K)] = [1, 8, 3071, 64]
            r_K = self._multihead_output(self._rel_K_w, positional_encodings)
            # add shifted relative logits to content logits
            # content_logits.shape:[B, H, T', T] confirmed ([1, 8, 1536, 1536])
            # content_logits = keras.ops.matmul(Q + self._r_w_bias, K, transpose_b=True)
            content_logits = keras.ops.einsum('b h i d, b h j d -> b h i j', Q + self._r_w_bias, K)
            # relative_logits.shape:[B, H, T', 2T-1] confirmed shape:[1, 8, 1536, 3071]
            relative_logits = keras.ops.matmul(Q + self._r_r_bias, r_K, transpose_b=True)
            # relative_logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            relative_logits = relative_shift(relative_logits)
            # COMPUTE ATTENTION WEIGHTS
            # logits.shape:[B, H, T', T] confirmed shape:[1, 8, 1536, 1536]
            logits = content_logits + relative_logits
        else:
            # COMPUTE ATTENTION WEIGHTS
            # calculate q*kT
            # output shape:[B, H, T', T]
            logits = keras.ops.matmul(Q, K, transpose_b=True)
        # apply softmax(q*kT) to calculate the ATTENTION WEIGHTS matrix
        weights = keras.layers.Softmax()(logits)
        # attention DROPOUT
        if training:
            # apply dropout on the attention weights
            weights = keras.layers.Dropout(rate=self._attn_dropout, name='attn_drop')(weights)
        # COMPUTE ATTENTION
        # transpose and reshape the output
        # output shape:[B, H, T', V] confirmed shape:[1, 8, 1536, 192]
        attention = keras.ops.linalg.matmul(weights, V)
        
        # final linear layer
        # trans_out shape:[B, T', H, V] confirmed shape:[1, 1536, 8, 192]
        trans_out = keras.ops.transpose(attention, [0, 2, 1, 3])
        # attended_embeds shape:(B, T', H*V] confirmed shape:[1, 1536, 1536]
        attended_embeds = keras.ops.reshape(trans_out, shape=(-1, trans_out.shape[-3], self._V_proj_dim))
        # output = self._to_out(attended_embeds)
        output = keras.ops.matmul(attended_embeds, self._out_w) + self._out_b
        
        return output

def dilated_residual(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: int = 2,
    dropout: float = 0,
    conv_type: str = "standard",
    repeat: int = 1,
    round: bool = False,
    **kwargs,
) -> keras.KerasTensor:
    """
    Construct a residual dilated convolution block.

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Number of filters in the convolutional layer.
    kernel_size
        Size of the convolutional kernel.
    rate_mult
        Rate multiplier for dilated convolution.
    dropout
        Dropout rate probability.
    conv_type
        Conv1D layer type.
    repeat
        Number of times to repeat the block.
    round
        Whether to round the dilation rate.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Output tensor after applying the dilated residual block.
    """
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for _ in range(repeat):
        rep_input = current

        # dilate
        current = conv_block_bs(
            current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(np.round(dilation_rate)),
            conv_type=conv_type,
            bn_gamma="ones",
            **kwargs,
        )

        # return
        current = conv_block_bs(
            current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            bn_gamma="zeros",
            **kwargs,
        )

        # residual add
        current = keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult
        if round:
            dilation_rate = np.round(dilation_rate)

    return current

def relative_shift(x):
    # we prepend zeros on the final timescale dimension
    to_pad = keras.ops.zeros_like(x[..., :1])
    x = keras.ops.concat([to_pad, x], -1)
    # t1 and t2 are expected to be the same, as they are result of 
    # matmul(Q + self._r_w_bias, K, transpose_b=True) -> [B, H, T', T]
    # so should be the seq_lengths/num_bins of Q and K, which should be the same
    num_heads = x.shape[1]
    t1 = x.shape[2]
    t2 = x.shape[3]
    x = keras.ops.reshape(x, [-1, num_heads, t2, t1])
    x = keras.ops.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = keras.ops.reshape(x, [-1, num_heads, t1, t2-1])
    x = keras.ops.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2+1)//2])
    
    return x

def get_position_features_func(relative_position_functions: str, absolute: bool):
    """
    Generate a position features function. 
    
    Parameters:
    ----------
    relative_position_functions
        Relative position function(s) to use. 'all' (exponential+central_mask+gamma) or 'central_mask'. 
        Enformer uses 'all', Borzoi uses 'central_mask'
    absolute
        Whether to take the absolute of the values before calculating feature embeddings.
        Enformer uses this, Borzoi does not.
    
    """
    def _position_features(
        positions: keras.KerasTensor, 
        feature_size: int, # num_relative_position_features: total number of basis functions*n(int)
        seq_length: int | None = None, # length of the transformer input sequence (default 1536)
        symmetric=False
    ) -> keras.KerasTensor:
        num_components = 3
        # if symmetric == True, the resulting features will be symmetric across the relative position of 0 (i.e. only absolute value of positions will matter)
        # if symmetric == False, then both the symmetric and asymmetric versions (symmetric multiplied by sign(positions)) of the features will be used
        if not symmetric:
            num_components = 2*num_components

        # for now, we do not allow odd sized embeddings
        # num_relative_position_features must be divisible by the number of feature functions (*2 if symmetric False)
        if feature_size % num_components != 0:
            raise ValueError(f"feature_size has to be divisible by {num_components}")

        num_basis_per_class = feature_size // num_components
        if absolute:
            positions = keras.ops.abs(positions)
        if relative_position_functions == "all":
            embeddings = keras.ops.concat(
                [pos_feats_exponential(positions, num_basis_per_class, seq_length),
                pos_feats_central_mask(positions, num_basis_per_class, seq_length),
                pos_feats_gamma(positions, num_basis_per_class, seq_length)],
                axis=-1)
        elif relative_position_functions == "central_mask":
            embeddings = pos_feats_central_mask(positions, num_basis_per_class, seq_length)
        else:
            raise ValueError(f"Did not recognise relative_position_functions {relative_position_functions}")
        
        # if False, both symmetric and asymmetric versions of rel encodings will be contenated in rows
        if not symmetric:
            embeddings = keras.ops.concat(
                [embeddings, keras.ops.expand_dims(keras.ops.sign(positions), axis=-1)*embeddings],
                axis=-1)
            
        # TODO: port check to keras 3 -> not sure if possible
        # tf.TensorShape(embeddings.shape).assert_is_compatible_with(positions.shape + [feature_size])
        
        # tensor of shape: `positions.shape+(feature_size, )`
        return embeddings
    return _position_features

# def pos_feats_all(positions: keras.KerasTensor,
#                   # num_relative_position_features: total number of basis functions*n(int)
#                   feature_size: int,
#                   # length of the transformer input sequence (default 1536)
#                   seq_length: int = None,
#                   bin_size: int = None,
#                   # relative_position_functions
#                   feature_functions: list = ['pos_feats_exponential',
#                                              'pos_feats_central_mask',
#                                              'pos_feats_gamma'],
#                   symmetric=False):
#     if feature_functions is None:
#         # default relative_position_functions
#         feature_functions = ['pos_feats_exponential',
#                              'pos_feats_central_mask',
#                              'pos_feats_gamma']
#     # number of feature functions
#     num_components = len(feature_functions)  # 1 per each basis function
#     # if True, the resulting features will be symmetric across the relative position of 0 (i.e. only absolute value of positions will matter)
#     # if False, then both the symmetric and asymmetric versions (symmetric multiplied by sign(positions)) of the features will be used
#     if not symmetric:
#         # False, therefore both symmetric and asymmetric versions will be computed
#         num_components = 2*num_components
    
#     # for now, we do not allow odd sized embeddings
#     # num_relative_position_features must be divisible by the number of feature functions (*2 if symmetric False)
#     if feature_size%num_components!=0:
#         raise ValueError(f'feature_size has to be divisible by {num_components}')
    
#     # retrieve feature function names from the dictionary
#     # feature_functions = [get_pos_feat_fun(f) for f in feature_functions]
#     # num_relative_position_features // number of feature functions (*2 if symmetric False)
#     num_basis_per_class = feature_size // num_components
#     # calculate symmetric relative encodings with each function and concatenate them in rows
#     embeddings = keras.ops.concat([fun(keras.ops.abs(positions),
#                                 # feature_size pass to each function
#                                 num_basis_per_class,
#                                 seq_length,
#                                 bin_size) for fun in feature_functions],
#                            axis=-1)
    
#     # if False, both symmetric and asymmetric versions of rel encodings will be contenated in rows
#     if not symmetric:
#         embeddings = keras.ops.concat(
#             [embeddings, keras.ops.expand_dims(keras.ops.sign(positions), axis=-1)*embeddings],
#             axis=-1)

#     # TODO: port check to keras 3 -> not sure if possible
#     # tf.TensorShape(embeddings.shape).assert_is_compatible_with(positions.shape + [feature_size])
    
#     # tensor of shape: `positions.shape+(feature_size, )`
#     return embeddings

# def get_pos_feat_fun(name):
#     # available positional feature functions:
#     available = {'pos_feats_exponential': pos_feats_exponential,
#                  'pos_feats_central_mask': pos_feats_central_mask,
#                  'pos_feats_gamma': pos_feats_gamma}
#     if name not in available:
#         raise ValueError(f'function {name} not available in {available.keys()}')
#     # returns positional feature functions
#     return available[name]

def pos_feats_exponential(
    positions: keras.KerasTensor,
    num_basis: int, # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
    seq_length: int | None = None, # length of the transformer input sequence (default 1536)
    min_half_life: float = 3.0 # smallest exponential half life in the grid of half lives
):
    if seq_length is None:
        seq_length = keras.ops.max(keras.ops.abs(positions))+1
    # grid of half lifes from [3, seq_length/2] with feature_size distributed on the log scale
    seq_length = keras.ops.cast(seq_length, dtype="float32")
    max_range = keras.ops.log(seq_length)/keras.ops.log(2.0)
    # calculate half lifes
    half_life = keras.ops.pow(2.0, keras.ops.linspace(min_half_life, max_range, num_basis))
    # prepend 2 dimensions to the tensor half_life
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = keras.ops.abs(positions)
    # calculate symmetric positional encodings
    outputs = keras.ops.exp(-keras.ops.log(2.0)/half_life*keras.ops.expand_dims(positions, axis=-1))
    # TODO: convert to Keras 3
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape + [num_basis])
    
    # a tensor with shape [2*seq_length-1, num_basis]
    return outputs

def pos_feats_central_mask(
    positions: keras.KerasTensor, 
    num_basis: int, 
    seq_length: int
):
    """Positional features using a central mask (allow only central features)."""
    pow_rate = np.exp(np.log(seq_length + 1) / num_basis).astype("float32")
    center_widths = keras.ops.pow(pow_rate, keras.ops.arange(1, num_basis + 1, dtype="float32"))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = keras.ops.cast(
        center_widths > keras.ops.expand_dims(keras.ops.abs(positions), axis = -1),
        "float32")
    # TODO: convert to Keras 3
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(
    #     positions.shape + [num_basis]
    # )
    return outputs

# # positional features using a central mask (allow only central features)
# def pos_feats_central_mask(
#     positions: keras.KerasTensor,
#     num_basis: int,
#     # length of the transformer input sequence (default 1536
# ):
#     center_widths = keras.ops.pow(2.0, keras.ops.range(1, num_basis+1, dtype="float32"))
#     center_widths = center_widths-1
#     center_widths = _prepend_dims(center_widths, positions.shape.rank)
#     outputs = keras.ops.cast(
#         center_widths > keras.ops.expand_dims(keras.ops.abs(positions), axis = -1),
#         "float32")
#     # TODO: convert to Keras 3
#     # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
#     return outputs

# gamma probability distribution function: p(x|concentration, rate)
def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = keras.ops.xlogy(concentration-1., x)-rate*x
    log_normalization = (keras.ops.lgamma(concentration)-concentration*keras.ops.log(rate))
    
    return keras.ops.exp(log_unnormalized_prob-log_normalization)

# positional features computed using the gamma distributions
def pos_feats_gamma(
    positions: keras.KerasTensor,
    num_basis: int, # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
    seq_length: int | None = None, # length of the transformer input sequence (default 1536)
    stddev=None,
    start_mean=None
):
    if seq_length is None:\
        seq_length = keras.ops.max(keras.ops.abs(positions))+1
    if stddev is None:
        stddev = seq_length/(2*num_basis)
    if start_mean is None:
        start_mean = seq_length/num_basis
    mean = keras.ops.linspace(start_mean, seq_length, num=num_basis)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean/stddev)**2
    rate = mean/stddev**2
    probabilities = gamma_pdf(keras.ops.expand_dims(keras.ops.abs(keras.ops.cast(positions, dtype="float32")), axis=-1),
                              concentration,
                              rate)
    probabilities += 1e-8 # to ensure numerical stability
    outputs = probabilities/keras.ops.max(probabilities,
                                          axis=1,
                                          keepdims=True)
    # TODO: convert to Keras 3
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    
    return outputs

def _prepend_dims(x, num_dims):
    # Should be possible cleaner with keras.ops.expand_dims() but oh well
    return keras.ops.reshape(x, shape=[1]*num_dims+x.shape)