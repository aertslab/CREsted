"""Helper functions for the Enformer/Borzoi attention layers"""

import keras
import numpy as np


# Attention pooling layer
class AttentionPool1D(keras.layers.Layer):
    """
    AttentionPool from the FastISM repository. Does learnable Softmax pooling, for use in Enformer.

    Parameters
    ----------
    pool_size
        Pooling size, same as in Max/AvgPooling.
    per_channel
        If True, the logits/softmax weights will be computed for each channel separately.
        If False, same weights will be used across all channels.
    w_init_scale
        Initialisation of w. When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
    strides/padding/data_format
        placeholder arguments to capture them from from_config.
        Not used in setting up the layer.
    name
        Module name.
    **kwargs
        Extra arguments passed to keras.layers.Layer.
    """

    def __init__(
        self,
        pool_size: int = 2,
        per_channel: bool = True,
        w_init_scale: float = 2.0,
        strides = None,
        padding = None,
        data_format = None,
        name: str = "AttentionPool1D",
        **kwargs
        ):
        """Initialize the AttentionPool layer, which has learnable weights."""
        super().__init__(name = name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale

        # Needed for compatibility with FastISM, not actually used to configure
        self._strides = self._pool_size
        self._padding = "valid" # ensure it behaves like MaxPooling1D with valid padding
        self._data_format = "channels_last"

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

    def call(self, inputs, training = False):
        _, length, num_features = inputs.shape

        if length is None: # this can happen at when creating fast_ism_model
            return inputs # don't do anything for now

        inputs = keras.ops.reshape(
            inputs,
            (-1, length // self._pool_size, self._pool_size, num_features))
        return keras.ops.sum(
            inputs * keras.ops.softmax(keras.ops.matmul(inputs, self.w), axis=-2),
            axis=-2)

# Multi-head attention block from baskerville
class MultiheadAttention(keras.layers.Layer):
    """
    Creates a MultiheadAttention module.

    Adapted from Baskerville's MultiheadAttention, original version written by Ziga Avsec.

    Parameters
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
        Can be 'enformer' or 'borzoi'.
        Enformer default is 'enformer' (exponential & central_mask (scaling factor 2) & gamma).
        Borzoi default is 'borzoi' (central mask only (scaling factor depending on length)).
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

    def __init__(
        self,
        value_size,
        key_size,
        heads,
        scaling: bool = True,
        attention_dropout_rate: float = 0.,
        relative_position_symmetric: bool = False,
        relative_position_functions: str = "borzoi",
        relative_position_absolute: bool = False,
        num_position_features: int | None = None,
        positional_dropout_rate: float = 0.,
        content_position_bias: bool = True,
        zero_initialize: bool = True,
        initializer: str = "he_normal",
        l2_scale: float = 0.,
        transpose_stride: int = 0,
        gated: bool = False,
        qkv_width: int = 1,
        name: str = 'mhsa',
        **kwargs
        ):
        """
        Initialize the MultiheadAttention layer.

        Note: transpose_stride, gated, qkv_width are implemented,
        but not used in Enformer/Borzoi and therefore not tested.
        """
        super().__init__(name = name, **kwargs)
        # Save init parameters for use in the model and at get_config()
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        self._relative_position_absolute = relative_position_absolute
        if num_position_features is None:
            # num_position_features needs to be divisible by the number of
            # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_position_features = (self._value_size // divisible_by) * divisible_by
        else:
            self._num_position_features = num_position_features
        self._positional_dropout_rate = positional_dropout_rate
        self._content_position_bias = content_position_bias
        self._zero_initialize = zero_initialize
        self._initializer = initializer
        self._l2_scale = l2_scale
        self._transpose_stride = transpose_stride
        self._gated = gated
        self._qkv_width = qkv_width

        # Calculate/get derived parameters
        self._key_proj_size = self._key_size * self._num_heads
        self._embedding_size = self._value_size * self._num_heads

        self.relative_position_func = get_position_features_func(
            relative_position_functions=relative_position_functions,
            absolute=relative_position_absolute)

    def build(self, input_shape):
        if self._qkv_width == 1:
            # standard dense layers
            self._q_layer = keras.layers.Dense(
                self._key_proj_size,
                name="q_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._k_layer = keras.layers.Dense(
                self._key_proj_size,
                name="k_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
            self._v_layer = keras.layers.Dense(
                self._embedding_size,
                name="v_layer",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )
        else:
            # CvT separable convolutions
            self._q_layer = keras.layers.SeparableConv1D(
                self._key_proj_size,
                kernel_size=self._qkv_width,
                padding="same",
                name="q_layer",
                use_bias=False,
                depthwise_regularizer=keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._k_layer = keras.layers.SeparableConv1D(
                self._key_proj_size,
                kernel_size=self._qkv_width,
                padding="same",
                name="k_layer",
                use_bias=False,
                depthwise_regularizer=keras.regularizers.l2(self._l2_scale),
                pointwise_regularizer=keras.regularizers.l2(self._l2_scale),
                depthwise_initializer=self._initializer,
                pointwise_initializer=self._initializer,
            )
            self._v_layer = keras.layers.SeparableConv1D(
                self._embedding_size,
                kernel_size=self._qkv_width,
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
                self._embedding_size,
                activation="activation",
                name="gate",
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=self._initializer,
            )

        w_init = keras.initializers.Zeros() if self._zero_initialize else self._initializer
        if self._transpose_stride > 0:
            self._embedding_layer = keras.layers.Conv1DTranspose(
                filters=self._embedding_size,
                kernel_size=3,
                strides=self._transpose_stride,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )
        else:
            self._embedding_layer = keras.layers.Dense(
                self._embedding_size,
                name="embedding_layer",
                kernel_regularizer=keras.regularizers.l2(self._l2_scale),
                kernel_initializer=w_init,
            )

        # Create relative position layers
        self._r_k_layer = keras.layers.Dense(
            self._key_proj_size,
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self._l2_scale),
            kernel_initializer=self._initializer,
            name="r_k_layer"
        )
        self._r_w_bias = self.add_weight(
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype="float32",
            name = "r_w_bias",
        )
        self._r_r_bias = self.add_weight(
            shape=[1, self._num_heads, 1, self._key_size],
            initializer=self._initializer,
            dtype="float32",
            name = "r_r_bias"
        )

        # Create dropout layers (layers so that they handle randomness correctly in Keras 3)
        self.pos_dropout = keras.layers.Dropout(rate=self._positional_dropout_rate)
        self.attn_dropout = keras.layers.Dropout(rate=self._attention_dropout_rate)

    def _multihead_output(self, linear_layer, inputs):
        """Applies a standard linear to inputs and returns multihead output."""
        output = linear_layer(inputs)  # [B, T, H * KV]
        _, seq_len, num_channels = output.shape

        # Split H * Channels into separate axes.
        num_kv_channels = num_channels // self._num_heads
        output = keras.ops.reshape(
            output, [-1, seq_len, self._num_heads, num_kv_channels]
        )
        # [B, T, H, KV] -> [B, H, T, KV]
        return keras.ops.transpose(output, [0, 2, 1, 3])

    def call(self, inputs, training=False):
        # Initialise the projection layers.
        embedding_size = self._value_size * self._num_heads
        seq_len = inputs.shape[1]

        # Compute q, k and v as multi-headed projections of the inputs.
        q = self._multihead_output(self._q_layer, inputs)  # [B, H, T, K] -> (1, 8, 1536, 64)
        k = self._multihead_output(self._k_layer, inputs)  # [B, H, T, K] -> (1, 8, 1536, 64)
        v = self._multihead_output(self._v_layer, inputs)  # [B, H, T, V] -> (1, 8, 1536, 192)

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size**-0.5

        # content_logits: [B, H, T', T] -> (1, 8, 1536, 1536)
        content_logits = keras.ops.einsum('b h i d, b h j d -> b h i j', q + self._r_w_bias, k)

        if self._num_position_features == 0:
            logits = content_logits
        else:
            # Project positions to form relative keys.
            # distances: [1, 2T-1] -> (1, 3071)
            distances = keras.ops.expand_dims(keras.ops.arange(-seq_len + 1, seq_len, dtype="float32"), axis = 0)
            # positional_encodings: [1, 2T-1, Cr] -> (1, 3071, 192)
            positional_encodings = self.relative_position_func(
                positions=distances,
                feature_size=self._num_position_features,
                seq_length=seq_len,
                symmetric=self._relative_position_symmetric,
            )

            positional_encodings = self.pos_dropout(positional_encodings, training = training)

            # Relative position weights
            # r_k goal: [H, 2T-1, K] -> (8, 3071, 64)
            # r_k = self._multihead_output(self._r_k_layer, positional_encodings)
            r_k = self._r_k_layer(positional_encodings) # [1, 2T-1, H*K]
            r_k = keras.ops.reshape(
                r_k, [-1, self._num_heads, self._key_size] # [2T-1, H, K]
            )
            # [H, 2T-1, K] -> (8, 3071, 64)
            r_k =  keras.ops.transpose(r_k, [1, 0, 2])

            # Add shifted relative logits to content logits.
            if self._content_position_bias:
                # relative_logits: [B, H, T', 2T-1] -> (1, 8, 1536, 3071)
                relative_logits = keras.ops.einsum('b h i d, h j d -> b h i j', q + self._r_r_bias, r_k)

            else:
                # relative_logits: [1, H, 1, 2T-1] -> (1, 8, 1, 3071)
                relative_logits = keras.ops.einsum('b h i d, h j d -> b h i j', self._r_r_bias, r_k)
                # relative_logits post-broadcast: [1, H, T', 2T-1] -> (1, 8, 1536, 3071)
                relative_logits = keras.ops.broadcast_to(
                    relative_logits,
                    shape=(1, self._num_heads, seq_len, 2 * seq_len - 1),
                )

            # shifted relative_logits: [B, H, T', T] -> (1, 8, 1536, 1536)
            relative_logits = relative_shift(relative_logits)
            logits = content_logits + relative_logits

        # softmax across length
        output = keras.ops.softmax(logits)

        # Dropout on the attention weights.
        output = self.attn_dropout(output, training = training)

        # Transpose and reshape the output.
        output = keras.ops.matmul(output, v)  # [B, H, T', V] -> (1, 8, 1536, 192)
        output = keras.ops.transpose(output, [0, 2, 1, 3])  # [B, T', H, V] -> (1, 1536, 8, 192)
        output = keras.ops.reshape(
            output, [-1, seq_len, embedding_size]
        ) # [B, T, H*V] -> (1, 1536, 1536)

        # Gate
        if self._gated:
            output = self._gate_layer(output)

        # Final linear layer
        output = self._embedding_layer(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "value_size": self._value_size,
            "key_size": self._key_size,
            "heads": self._num_heads,
            "scaling": self._scaling,
            "attention_dropout_rate": self._attention_dropout_rate,
            "relative_position_symmetric": self._relative_position_symmetric,
            "relative_position_functions": self._relative_position_functions,
            "relative_position_absolute": self._relative_position_absolute,
            "num_position_features": self._num_position_features,
            "positional_dropout_rate": self._positional_dropout_rate,
            "content_position_bias": self._content_position_bias,
            "zero_initialize": self._zero_initialize,
            "initializer": self._initializer,
            "l2_scale": self._l2_scale,
            "transpose_stride": self._transpose_stride,
            "gated": self._gated,
            "qkv_width": self._qkv_width
        })
        return config


def relative_shift(x):
    # we prepend zeros on the final timescale dimension
    to_pad = keras.ops.zeros_like(x[..., :1])
    x = keras.ops.concatenate([to_pad, x], -1)
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

    Parameters
    ----------
    relative_position_functions
        Relative position function(s) to use. 'enformer' (exponential+enformer's central_mask+gamma) or 'borzoi' (borzoi's central_mask only).
    absolute
        Whether to take the absolute of the values before calculating feature embeddings.
        Enformer uses this, Borzoi does not.
    """
    if relative_position_functions == 'enformer':
        # Using gamma position function, which uses functions currently not implemented in keras 3 (lgamma, xlogy)
        # Solution: check backend on our own.
        backend = keras.src.backend.config.backend()
        if backend == 'tensorflow':
            gamma_pdf = gamma_pdf_tf
        elif backend == 'pytorch':
            gamma_pdf = gamma_pdf_pt
        else:
            raise NotImplementedError(f"Using gamma position functions (as part of relative_position_functions == 'enformer') currently only supports TensorFlow and PyTorch backends, not {backend}.")
    def _position_features(
        positions: keras.KerasTensor,
        feature_size: int, # num_relative_position_features: total number of basis functions*n(int)
        seq_length: int | None = None, # length of the transformer input sequence (default 1536)
        symmetric=False
    ) -> keras.KerasTensor:
        num_components = 3 if relative_position_functions == 'enformer' else 1
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
            positions_sign = keras.ops.sign(positions)
            positions = keras.ops.abs(positions)
        if relative_position_functions == "enformer":
            embeddings = keras.ops.concatenate(
                [pos_feats_exponential(positions, num_basis_per_class, seq_length),
                 pos_feats_central_mask_enformer(positions, num_basis_per_class),
                 pos_feats_gamma(positions, num_basis_per_class, seq_length, gamma_pdf_func = gamma_pdf)],
                axis=-1)
        elif relative_position_functions == "borzoi":
            embeddings = pos_feats_central_mask_borzoi(positions, num_basis_per_class, seq_length)
        else:
            raise ValueError(f"Did not recognise relative_position_functions {relative_position_functions}")

        # if False, both symmetric and asymmetric versions of rel encodings will be contenated in rows
        if not symmetric:
            embeddings = keras.ops.concatenate(
                [embeddings, keras.ops.expand_dims(positions_sign, axis=-1)*embeddings],
                axis=-1)
        # TODO: port check to keras 3 -> not sure if possible
        # tf.TensorShape(embeddings.shape).assert_is_compatible_with(positions.shape + [feature_size])

        # tensor of shape: `positions.shape+(feature_size, )`
        return embeddings
    return _position_features

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
    half_life = keras.ops.power(2.0, keras.ops.linspace(min_half_life, max_range, num_basis))
    # prepend 2 dimensions to the tensor half_life
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = keras.ops.abs(positions)
    # calculate symmetric positional encodings
    outputs = keras.ops.exp(-keras.ops.log(2.0)/half_life*keras.ops.expand_dims(positions, axis=-1))
    # TODO: convert to Keras 3
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape + [num_basis])

    # a tensor with shape [2*seq_length-1, num_basis]
    return outputs

def pos_feats_central_mask_borzoi(
    positions: keras.KerasTensor,
    num_basis: int,
    seq_length: int
):
    """
    Positional features using a central mask (allow only central features).

    Uses the Borzoi implementation, which calculates pow_rate based on seq_len (default).
    """
    pow_rate = np.exp(np.log(seq_length + 1) / num_basis).astype("float32")
    center_widths = keras.ops.power(pow_rate, keras.ops.arange(1, num_basis + 1, dtype="float32"))
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

# positional features using a central mask (allow only central features)
def pos_feats_central_mask_enformer(
    positions: keras.KerasTensor,
    num_basis: int,
):
    """
    Positional features using a central mask (allow only central features).

    Uses the Enformer implementation, which has a fixed power rate of 2.
    """
    center_widths = keras.ops.power(2.0, keras.ops.arange(1, num_basis+1, dtype="float32"))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = keras.ops.cast(
        center_widths > keras.ops.expand_dims(keras.ops.abs(positions), axis = -1),
        "float32")
    # TODO: convert to Keras 3
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(positions.shape+[num_basis])
    return outputs

def gamma_pdf_tf(x, concentration, rate):
    """Gamma probability distribution function: p(x|concentration, rate) in tensorflow."""
    import tensorflow as tf
    log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
    log_normalization = (tf.math.lgamma(concentration) -
                        concentration * tf.math.log(rate))
    return tf.exp(log_unnormalized_prob - log_normalization)

def gamma_pdf_pt(x, concentration, rate):
    """
    Gamma probability distribution function: p(x|concentration, rate) in pytorch.

    Taken from lucidrains/enformer-pytorch.
    """
    import torch
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

# positional features computed using the gamma distributions
def pos_feats_gamma(
    positions: keras.KerasTensor,
    num_basis: int, # num_basis_per_class=num_relative_position_features//num_components(*2 if symmetric False)
    seq_length: int | None = None, # length of the transformer input sequence (default 1536)
    stddev=None,
    start_mean=None,
    gamma_pdf_func = None
):
    if gamma_pdf_func is None:
        backend = keras.src.backend.config.backend()
        if backend == 'tensorflow':
            gamma_pdf_func = gamma_pdf_tf
        elif backend == 'pytorch':
            gamma_pdf_func = gamma_pdf_pt
        else:
            raise NotImplementedError(f"Using gamma position functions (as part of relative_position_functions == 'enformer') currently only supports TensorFlow and PyTorch backends, not {backend}.")

    if seq_length is None:
        seq_length = keras.ops.max(keras.ops.abs(positions))+1
    if stddev is None:
        stddev = seq_length/(2*num_basis)
    if start_mean is None:
        start_mean = seq_length/num_basis
    mean = keras.ops.linspace(start_mean, seq_length, num=num_basis)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean/stddev)**2
    rate = mean/stddev**2
    probabilities = gamma_pdf_func(keras.ops.expand_dims(keras.ops.abs(keras.ops.cast(positions, dtype="float32")), axis=-1),
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
    return keras.ops.reshape(x, [1]*num_dims+x.shape)