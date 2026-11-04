"""Helper layers for zoo models."""

from __future__ import annotations

import functools
import inspect

import keras
import numpy as np

from ._attention import AttentionPool1D, MultiheadAttention


def _batchnorm_tuple_axes():
    """
    Make keras BatchNormalization store its reduction axes as a tuple, on the torch backend.

    Keras (3.15 at least) stores them as a list, which layer tracking wraps in a TrackedList that torch.compile can't
    trace. That breaks the graph in every BatchNormalization call, so `jit_compile=True` models with batchnorm (e.g. the
    Cerberus conv tower) mostly run eagerly. Fixed upstream in keras-team/keras#24083 (open as of 2026-11): this is a
    no-op once that is released, and can be removed then. It doesn't change the layer's class or config, so saved
    models are unaffected.
    """
    build = keras.layers.BatchNormalization.build
    if getattr(build, "_crested_tuple_axes", False):
        return

    @functools.wraps(build)
    def tuple_axes_build(self, input_shape):
        build(self, input_shape)
        if isinstance(self._reduction_axes, list):
            object.__setattr__(self, "_reduction_axes", tuple(self._reduction_axes))  # bypass keras tracking

    tuple_axes_build._crested_tuple_axes = True
    keras.layers.BatchNormalization.build = tuple_axes_build


if keras.config.backend() == "torch":
    _batchnorm_tuple_axes()


@keras.saving.register_keras_serializable(package="crested", name="DropPath")
class DropPath(keras.layers.Layer):
    """
    Drop path (stochastic depth): zero the whole input of a sample with probability `rate`, and scale kept samples by 1/(1-rate).

    Meant for the residual branch of a block. Same as `keras.layers.Dropout(rate, noise_shape=(None, 1, ..., 1))`,
    but draws the per-sample mask with regular dropout on a (batch, 1, ..., 1) tensor, which (unseeded) compiles without
    graph breaks on the torch backend.

    Parameters
    ----------
    rate
        Probability of dropping a sample.
    seed
        Random seed.
    **kwargs
        Extra arguments passed to keras.layers.Layer.
    """

    def __init__(self, rate: float, seed: int | None = None, **kwargs):
        """Initialize the DropPath layer."""
        super().__init__(**kwargs)
        if not 0 <= rate <= 1:
            raise ValueError(f"rate should be between 0 and 1, got {rate}.")
        self.rate = rate
        self.seed = seed
        if rate > 0:
            self.seed_generator = keras.random.SeedGenerator(seed)
        self.supports_masking = True
        self._build_at_init()

    def call(self, inputs, training=False):
        """Drop samples during training."""
        if not training or self.rate == 0:
            return inputs
        shape = (keras.ops.shape(inputs)[0],) + (1,) * (len(inputs.shape) - 1)
        mask = keras.random.dropout(keras.ops.ones(shape, dtype=inputs.dtype), self.rate, seed=self.seed_generator)
        return inputs * mask

    def compute_output_shape(self, input_shape):
        """Output shape is the input shape."""
        return input_shape

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self.seed})
        return config


@keras.saving.register_keras_serializable(package="crested", name="RMSBatchNormalization")
class RMSBatchNormalization(keras.layers.Layer):
    """
    Batch normalization without centering, as used in AlphaGenome.

    Scales the inputs by the root mean square of each channel over all other axes (batch and positions):
    `x * gamma / sqrt(mean(x**2) + epsilon) + beta`. During training, the batch statistics are used and tracked
    in a moving average, which is used at inference (like :class:`keras.layers.BatchNormalization`).

    Parameters
    ----------
    momentum
        Momentum of the moving average of the mean square.
    epsilon
        Small constant added to the mean square.
    """

    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        """Create the scale (gamma), offset (beta) and moving mean square (moving_variance) per channel."""
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(shape=shape, initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=shape, initializer="zeros", name="beta")
        self.moving_variance = self.add_weight(
            shape=shape, initializer="ones", name="moving_variance", trainable=False, autocast=False
        )

    def call(self, inputs, training=None):
        """Normalize the inputs, by batch statistics in training and by the moving average otherwise."""
        if training:
            axes = tuple(range(len(inputs.shape) - 1))
            variance = keras.ops.stop_gradient(
                keras.ops.mean(keras.ops.square(keras.ops.cast(inputs, "float32")), axis=axes)
            )
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * variance)
        else:
            variance = self.moving_variance
        inv = self.gamma * keras.ops.cast(keras.ops.rsqrt(variance + self.epsilon), self.compute_dtype)
        return inputs * inv + self.beta

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config


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
) -> keras.KerasTensor:
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
        Type of normalization ('batch', 'layer', 'rms', 'group', 'unit' or 'none'). See :func:`normalize`.
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
        kernel_regularizer=keras.regularizers.l2(l2) if l2 > 0 else None,
        name=name_prefix + "_dense" if name_prefix else None,
    )(inputs)

    if normalization == "batch":
        x = normalize(
            x,
            normalization,
            momentum=bn_momentum,
            gamma_initializer=bn_gamma,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )
    else:
        x = normalize(x, normalization, name=f"{name_prefix}_{normalization}norm" if name_prefix else None)

    x = activate(x, activation, name=name_prefix + "_activation" if name_prefix else None)
    x = keras.layers.Dropout(dropout, name=name_prefix + "_dropout" if name_prefix else None)(x)
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
    pool_padding: str | None = None,
    l2: float = 1e-5,
    batchnorm_momentum: float = 0.99,
    pool_type: str = "max",
    name_prefix: str | None = None,
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
        Type of normalization ('batch', 'layer', 'rms', 'group', 'unit' or 'none'). See :func:`normalize`.
    res
        Whether to use residual connections (default is False).
    padding
        Padding type for the convolutional layer (default is "valid").
    pool_padding
        Padding type to be used for the pooling. If None (default), will use same value as `padding`.
    l2
        L2 regularization weight (default is 1e-5).
    batchnorm_momentum
        Batch normalization momentum (default is 0.99).
    pool_type
        Pooling type ('max', 'attention' or 'average'). See :func:`pool`.
    name_prefix
        Prefix for layer names.

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
        kernel_regularizer=keras.regularizers.L2(l2) if l2 > 0 else None,
        use_bias=conv_bias,
        name=name_prefix + "_conv" if name_prefix else None,
    )(inputs)
    if normalization == "batch":
        x = normalize(
            x,
            normalization,
            momentum=batchnorm_momentum,
            name=name_prefix + "_batchnorm" if name_prefix else None,
        )
    else:
        x = normalize(x, normalization, name=f"{name_prefix}_{normalization}norm" if name_prefix else None)
    x = activate(x, activation, name=name_prefix + "_activation" if name_prefix else None)
    if res:
        if filters != residual.shape[2]:
            residual = keras.layers.Convolution1D(
                filters=filters,
                kernel_size=1,
                strides=1,
                name=name_prefix + "_resconv" if name_prefix else None,
            )(residual)
        x = keras.layers.Add()([x, residual])

    if pool_size > 1:
        x = pool(
            x,
            pool_type,
            pool_size=pool_size,
            padding=pool_padding if pool_padding is not None else padding,
            name=name_prefix + "_pool" if name_prefix else None,
        )
    if dropout > 0:
        x = keras.layers.Dropout(dropout, name=name_prefix + "_dropout" if name_prefix else None)(x)

    return x


def activate(current: keras.KerasTensor, activation: str, verbose: bool = False, name=None) -> keras.KerasTensor:
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
    name
        Name to use in the activation layer. Default is None (no name).

    Returns
    -------
    Output tensor after applying activation.
    """
    if verbose:
        print("activate:", activation)

    if activation == "relu":
        current = keras.layers.Activation("relu", name=name)(current)
    elif activation == "swish":
        current = keras.layers.Activation("swish", name=name)(current)
    elif activation == "gelu":
        current = keras.layers.Activation("gelu", name=name)(current)
    elif activation == "gelu_approx":
        # layers.Activation('gelu') uses unapproximated (default in activations.gelu), we want approximated
        current = keras.layers.Activation(gelu_approx, name=name)(current)
    elif activation == "sigmoid":
        current = keras.layers.Activation("sigmoid", name=name)(current)
    elif activation == "tanh":
        current = keras.layers.Activation("tanh", name=name)(current)
    elif activation == "exponential":
        current = keras.layers.Activation("exponential", name=name)(current)
    elif activation == "softplus":
        current = keras.layers.Activation("softplus", name=name)(current)
    elif activation == "gelu_enf":
        current = keras.layers.Activation(gelu_enf, name=name)(current)
    else:
        # Any other activation name known to Keras (raises ValueError if unknown)
        current = keras.layers.Activation(activation, name=name)(current)

    return current


@keras.saving.register_keras_serializable(package="crested", name="gelu_approx")
def gelu_approx(x):
    """Wrap around keras.activations.gelu with approximate = True."""
    return keras.activations.gelu(x, approximate=True)


@keras.saving.register_keras_serializable(package="crested", name="gelu_enf")
def gelu_enf(x):
    """Very simple gelu approximation, used in Enformer, so needed to get equivalent results."""
    return keras.ops.sigmoid(1.702 * x) * x


def pool(
    current: keras.KerasTensor,
    pool_type: str,
    pool_size=2,
    padding="same",
    verbose: bool = False,
    name=None,
    **kwargs,
) -> keras.KerasTensor:
    """
    Apply a pooling layer to a tensor.

    Parameters
    ----------
    current
        Input tensor.
    pool_type
        Pooling function to apply: 'max', 'attention' or 'average'.
    pool_size
        Size of the pooling window.
    padding
        Padding type.
    verbose
        Print verbose information (default is False).
    name
        Name to use in the pooling layer. Default is None (no name).
    **kwargs
        Extra arguments passed to the pooling layer (e.g. `strides`).

    Returns
    -------
    Output tensor after applying pooling.
    """
    if verbose:
        print("pool:", pool_type)

    if pool_type == "max":
        current = keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding, name=name, **kwargs)(current)
    elif pool_type == "attention":
        current = AttentionPool1D(pool_size=pool_size, padding=padding, name=name, **kwargs)(current)
    elif pool_type == "average":
        current = keras.layers.AveragePooling1D(pool_size=pool_size, padding=padding, name=name, **kwargs)(current)
    else:
        raise ValueError(f'Unrecognized pooling type "{pool_type}"')
    return current


def normalize(
    current: keras.KerasTensor,
    normalization: str | None,
    verbose: bool = False,
    name: str | None = None,
    **kwargs,
) -> keras.KerasTensor:
    """
    Apply a normalization layer to a tensor.

    Parameters
    ----------
    current
        Input tensor.
    normalization
        Type of normalization: 'batch', 'layer', 'rms', 'group', 'unit', 'rms_batch' (:class:`RMSBatchNormalization`),
        or 'none'/None for no normalization.
    verbose
        Print verbose information (default is False).
    name
        Name to use in the normalization layer. Default is None (no name).
    **kwargs
        Extra arguments passed to the normalization layer.
        Arguments the chosen layer does not accept (e.g. `momentum` for layer normalization) are ignored,
        so that blocks can pass their batchnorm settings regardless of the normalization type.

    Returns
    -------
    Output tensor after applying normalization.
    """
    normalization = "none" if normalization is None else normalization.lower()
    if verbose:
        print("normalize:", normalization)

    norm_layers = {
        "batch": keras.layers.BatchNormalization,
        "layer": keras.layers.LayerNormalization,
        "rms": keras.layers.RMSNormalization,
        "group": keras.layers.GroupNormalization,
        "unit": keras.layers.UnitNormalization,
        "rms_batch": RMSBatchNormalization,
    }
    if normalization == "none":
        return current
    if normalization not in norm_layers:
        raise ValueError(f'Unrecognized normalization "{normalization}"')
    norm_layer = norm_layers[normalization]
    accepted = inspect.signature(norm_layer.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return norm_layer(name=name, **kwargs)(current)


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
    conv_bias: bool = True,
    pool_type: str = "max",
    bn_momentum: float = 0.99,
    bn_gamma: keras.KerasTensor | None = None,
    bn_sync: str = True,
    bn_epsilon: float = 1e-5,
    kernel_initializer: str = "he_normal",
    padding: str = "same",
    normalization: str = "batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Construct a convolution block (for Basenji/Enformer).

    Important note: while conv_block() follows the conv-norm-activate-pool order,
    this follows Enformer/Borzoi's convention of norm-activate-conv-pool.

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
        Apply normalization (of type `normalization`).
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
    conv_bias
        Whether to use a bias in the convolution layer.
        Should be True for Enformer/Borzoi and (maybe) False for Borzoi?
    pool_type
        Pooling type ('max', 'attention' or 'average'). See :func:`pool`.
    bn_momentum
        BatchNorm momentum.
    bn_gamma
        BatchNorm gamma (defaults according to residual).
    bn_sync
        Whether to synchronise the batchnorm when doing multi-GPU training.
    bn_epsilon
        Epsilon to use for the BatchNorm layer. Usual default is 1e-5, but Borzoi uses 1e-3.
    kernel_initializer
        Convolution kernel initializer.
    padding
        Padding type.
    normalization
        Type of normalization applied if `batch_norm` is True. See :func:`normalize`.
        The bn_* arguments are passed on where the normalization layer accepts them.
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

    # normalization
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        current = normalize(
            current,
            normalization,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            gamma_initializer=bn_gamma,
            synchronized=bn_sync,
            name=f"{name_prefix}_{normalization}norm" if name_prefix else None,
        )

    # activation
    current = activate(current, activation, name=name_prefix + "_activation" if name_prefix else None)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=conv_bias,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale) if l2_scale > 0 else None,
        name=name_prefix + "_conv" if name_prefix else None,
    )(current)

    # dropout
    if dropout > 0:
        current = keras.layers.Dropout(rate=dropout, name=name_prefix + "_dropout" if name_prefix else None)(current)

    # residual add
    if residual:
        current = keras.layers.Add(name=name_prefix + "_add" if name_prefix else None)([inputs, current])

    # end activation
    if activation_end is not None:
        current = activate(
            current,
            activation_end,
            name=name_prefix + "_activation_end" if name_prefix else None,
        )

    # Pool
    if pool_size > 1:
        current = pool(
            current,
            pool_type=pool_type,
            pool_size=pool_size,
            padding=padding,
            name=name_prefix + "_pool" if name_prefix else None,
        )
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
    pos_encoding_funs: str = "borzoi",
    pos_encoding_abs: bool = True,
    num_pos_feats: int | None = None,
    zero_init: bool = True,
    residual: bool = True,
    ln_epsilon: float = 1e-5,
    absolute_positions: bool = False,
    normalization: str = "layer",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Construct a MHA block (for Enformer/Borzoi), consisting of Residual(LayerNorm+MHSelfAttention+Dropout).

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
        Can be 'enformer' or 'borzoi'.
        Enformer default uses all, using exponential+central_mask_enf+gamma.,
        Borzoi default only uses its version of central_mask.
    pos_encoding_abs
        Whether to use the absolute of values before calculating the relative position encoding.
    num_pos_feats
        Number of positional features. If not supplied, calculated from value_dim and number of position encoding functions.
        Min 6 for default relative_position_functions, min 12 for positional_features_sin_cos.
    zero_init
        Whether to initialize MHA from zero.
    residual
        Whether to wrap the entire block in residual structure.
    ln_epsilon
        Epsilon to use in the layer normalisation layer.
    absolute_positions
        Whether to use absolute position encoding rather than relative. Default is False.
    normalization
        Type of normalization before the MHA (default is 'layer'). See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the MHA block.
    """
    # MHA block
    current = normalize(
        inputs,
        normalization,
        epsilon=ln_epsilon,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        name=f"{name_prefix}_{normalization}norm",
    )
    current = MultiheadAttention(
        value_size=value_dim,
        key_size=key_query_dim,
        heads=num_heads,
        scaling=scaling,
        attention_dropout_rate=attn_dropout,
        relative_position_symmetric=symmetric_pos_encoding,
        relative_position_functions=pos_encoding_funs,
        relative_position_absolute=pos_encoding_abs,
        num_position_features=num_pos_feats,
        positional_dropout_rate=pos_dropout,
        zero_initialize=zero_init,
        initializer="he_normal",  # Unsure in Enf, think this is fine.
        l2_scale=1.0e-8,  # Doesn't seem to be set in Enf, is set in Borzoi.
        absolute_positions=absolute_positions,
        name=f"{name_prefix}_mhsa",
    )(current)
    current = keras.layers.Dropout(rate=final_dropout, name=f"{name_prefix}_dropout")(current)
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current


def ffn_block_enf(
    inputs,
    filters: int,
    expansion_rate: int = 2,
    dropout: int = 0.4,
    activation: str = "relu",
    residual: bool = True,
    ln_epsilon: float = 1e-5,
    normalization: str = "layer",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
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
    ln_epsilon
        Epsilon to use in the layer normalisation layer.
    normalization
        Type of normalization at the start of the block (default is 'layer'). See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the feedforward block.
    """
    expansion_filters = int(expansion_rate * filters)

    # First half
    current = normalize(
        inputs,
        normalization,
        epsilon=ln_epsilon,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        name=f"{name_prefix}_{normalization}norm",
    )
    current = keras.layers.Conv1D(filters=expansion_filters, kernel_size=1, name=f"{name_prefix}_pointwise_1")(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_1")(current)

    # Second half
    current = activate(current, activation, name=name_prefix + "_activation" if name_prefix else None)
    current = keras.layers.Conv1D(filters=filters, kernel_size=1, name=f"{name_prefix}_pointwise_2")(current)
    current = keras.layers.Dropout(rate=dropout, name=f"{name_prefix}_dropout_2")(current)

    # Residual
    if residual:
        current = keras.layers.Add()([inputs, current])
    return current


def dilated_residual(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 3,
    rate_mult: float = 2.0,
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


def legnet_se_block(
    inputs: keras.KerasTensor,
    units: int,
    reduction: int = 4,
    activation: str = "swish",
    name: str | None = None,
):
    """
    Squeeze-and-Excite block.

    Parameters
    ----------
    inputs
        Input tensor with shape (batch, channels, length).
    units
        Number of units (filters, dimensions) for the dense blocks.
    reduction
        Reduction parameter. The default is 4.
    activation
        Activation function between the two dense layers. The default is "swish".
    name : str, optional
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor with same shape as input.
    """
    # Global average pooling
    y = keras.ops.mean(inputs, axis=1)
    # Linear
    y = keras.layers.Dense(units // reduction, activation=None, name=f"{name}_fc1" if name else None)(y)
    # SiLU
    y = activate(y, activation, name=f"{name}_swish1" if name else None)
    # Linear 2
    y = keras.layers.Dense(units, activation=None, name=f"{name}_fc2" if name else None)(y)
    # Sigmoid
    y = activate(y, "sigmoid", name=f"{name}_sigmoid" if name else None)
    # Multiplication
    output = keras.layers.Multiply(name=f"{name}_multiply" if name else None)([inputs, y])
    return output


def legnet_eff_block(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int,
    resize_factor: int,
    activation: str = "swish",
    se_reduction: int | None = None,
    se_activation: str = "swish",
    normalization: str = "batch",
    name: str | None = None,
):
    """
    EfficientNet-style block with inverted residuals and squeeze-excite.

    Parameters
    ----------
    inputs
        Input tensor.
    filters
        Output channels.
    kernel_size
        Kernel size.
    resize_factor
        Expansion factor for the inner dimension.
    activation
        Activation function.
    se_reduction
        SE layer reduction factor. If None, same as resize_factor.
    se_activation
        Activation function inside the squeeze-and-excite block.
    normalization
        Type of normalization. See :func:`normalize`.
    name
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor.
    """
    se_reduction = resize_factor if se_reduction is None else se_reduction
    inner_dim = filters * resize_factor

    # Expansion
    y = keras.layers.Conv1D(
        filters=inner_dim, kernel_size=1, padding="same", use_bias=False, name=f"{name}_expand_conv" if name else None
    )(inputs)
    y = normalize(y, normalization, name=f"{name}_expand_bn" if name else None)
    y = activate(y, activation, name=f"{name}_expand_act" if name else None)

    # Depthwise
    y = keras.layers.Conv1D(
        filters=inner_dim,
        kernel_size=kernel_size,
        groups=inner_dim,
        padding="same",
        use_bias=False,
        name=f"{name}_dw_conv" if name else None,
    )(y)
    y = normalize(y, normalization, name=f"{name}_dw_bn" if name else None)
    y = activate(y, activation, name=f"{name}_dw_act" if name else None)

    # Squeeze-and-Excite
    y = legnet_se_block(
        y, inner_dim, reduction=se_reduction, activation=se_activation, name=f"{name}_se" if name else None
    )

    # Projection
    y = keras.layers.Conv1D(
        filters=filters, kernel_size=1, padding="same", use_bias=False, name=f"{name}_project_conv" if name else None
    )(y)
    y = normalize(y, normalization, name=f"{name}_project_bn" if name else None)
    y = activate(y, activation, name=f"{name}_project_act" if name else None)

    return y
