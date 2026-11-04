"""Layers and blocks for the AlphaGenome architecture.

AlphaGenome architecture: copyright 2025 Google DeepMind (Apache 2.0, https://github.com/google-deepmind/alphagenome_research)
Ported via alphagenome-pytorch (Apache 2.0, https://github.com/genomicsxai/alphagenome-pytorch)
"""

from __future__ import annotations

import math

import keras

from ._layers import activate, normalize


@keras.saving.register_keras_serializable(package="crested", name="StandardizedConv1D")
class StandardizedConv1D(keras.layers.Conv1D):
    """
    1D convolution with weight standardization and a learned gain per filter, as used in AlphaGenome.

    In every call, the kernel of each filter is standardized to zero mean and unit variance (over the kernel width and
    the input channels, scaled by 1/sqrt(fan-in)) and multiplied by a learned per-filter `scale`.
    Takes the same arguments as :class:`keras.layers.Conv1D`.
    """

    def build(self, input_shape):
        """Create the convolution weights and the per-filter scale."""
        super().build(input_shape)
        self.scale = self.add_weight(shape=(self.filters,), initializer="ones", name="scale")

    def call(self, inputs):
        """Convolve the inputs with the standardized kernel."""
        kernel = self.kernel
        mean = keras.ops.mean(kernel, axis=(0, 1), keepdims=True)
        variance = keras.ops.var(kernel, axis=(0, 1), keepdims=True)
        fan_in = kernel.shape[0] * kernel.shape[1]
        kernel = (kernel - mean) * keras.ops.rsqrt(keras.ops.maximum(variance * fan_in, 1e-4)) * self.scale
        outputs = self.convolution_op(inputs, kernel)  # Conv1D.call doesn't go through convolution_op
        if self.use_bias:
            outputs = outputs + self.bias
        return self.activation(outputs) if self.activation is not None else outputs

    def save_own_variables(self, store):
        """Save the kernel and bias like Conv1D, followed by the scale (Conv1D only saves the former)."""
        super().save_own_variables(store)
        if self.built:
            store[str(len(store))] = self.scale

    def load_own_variables(self, store):
        """Load the kernel and bias like Conv1D, followed by the scale."""
        super().load_own_variables(store)
        if self.built:
            self.scale.assign(store[str(len(store) - 1)])


@keras.saving.register_keras_serializable(package="crested", name="Bias")
class Bias(keras.layers.Layer):
    """Add a learned bias per channel (last axis)."""

    def build(self, input_shape):
        """Create the bias."""
        self.bias = self.add_weight(shape=(input_shape[-1],), initializer="zeros", name="bias")

    def call(self, inputs):
        """Add the bias to the inputs."""
        return inputs + self.bias


@keras.saving.register_keras_serializable(package="crested", name="Scale")
class Scale(keras.layers.Layer):
    """
    Multiply by a learned scale, either one scalar or one per channel (last axis).

    Parameters
    ----------
    per_channel
        Whether to learn a scale per channel instead of a single scalar.
    activation
        Optional activation applied to the learned scale before multiplying (e.g. 'softplus' to keep it positive).
    """

    def __init__(self, per_channel: bool = False, activation: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.per_channel = per_channel
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        """Create the scale."""
        shape = (input_shape[-1],) if self.per_channel else (1,)
        self.scale = self.add_weight(shape=shape, initializer="ones", name="scale")

    def call(self, inputs):
        """Multiply the inputs by the (activated) scale."""
        return inputs * self.activation(self.scale)

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update({"per_channel": self.per_channel, "activation": keras.activations.serialize(self.activation)})
        return config


def apply_rope(x, max_position: int = 8192):
    """
    Apply rotary position embeddings to the last axis of x (batch, positions, heads, channels), as in AlphaGenome.

    Frequencies are spaced geometrically up to `max_position`, plus a linear term.
    """
    _, seq_len, num_heads, channels = x.shape
    num_freq = channels // 2
    positions = keras.ops.arange(seq_len, dtype="float32")
    base_freqs = keras.ops.exp(keras.ops.linspace(0.0, math.log(max_position - num_freq + 1), num_freq))
    inv_freq = 1.0 / (keras.ops.arange(num_freq, dtype="float32") + base_freqs)
    theta = keras.ops.repeat(positions[:, None] * inv_freq[None, :], 2, axis=-1)[None, :, None, :]
    cos_theta = keras.ops.cast(keras.ops.cos(theta), x.dtype)
    sin_theta = keras.ops.cast(keras.ops.sin(theta), x.dtype)
    x_rotated = keras.ops.reshape(
        keras.ops.stack([-x[..., 1::2], x[..., ::2]], axis=-1), (-1, seq_len, num_heads, channels)
    )
    return x * cos_theta + x_rotated * sin_theta


@keras.saving.register_keras_serializable(package="crested", name="MultiQueryAttention")
class MultiQueryAttention(keras.layers.Layer):
    """
    Multi-query self-attention with rotary position embeddings, as used in AlphaGenome.

    `num_heads` query heads share a single key and value head. Queries, keys and values are layer normalized,
    attention logits are soft-capped with `tanh` and can receive an additive bias per head, which is upsampled
    from a lower resolution (as from the AlphaGenome pair representation).

    Parameters
    ----------
    num_heads
        Number of query heads.
    key_dim
        Channels per query/key head.
    value_dim
        Channels of the value head.
    logits_soft_cap
        Attention logits are soft-capped to (-logits_soft_cap, logits_soft_cap).
    max_position
        Maximum position of the rotary embeddings.
    bias_upsampling
        Upsampling factor of the attention bias, from its resolution to the input resolution.
    """

    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 128,
        value_dim: int = 192,
        logits_soft_cap: float = 5.0,
        max_position: int = 8192,
        bias_upsampling: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.logits_soft_cap = logits_soft_cap
        self.max_position = max_position
        self.bias_upsampling = bias_upsampling

    def build(self, inputs_shape, bias_shape=None):
        """Create the projection and normalization layers."""
        channels = inputs_shape[-1]
        self.q_proj = keras.layers.Dense(self.num_heads * self.key_dim, use_bias=False, name="q_proj")
        self.k_proj = keras.layers.Dense(self.key_dim, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(self.value_dim, use_bias=False, name="v_proj")
        self.q_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="q_norm")
        self.k_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="k_norm")
        self.v_norm = keras.layers.LayerNormalization(epsilon=1e-5, name="v_norm")
        self.out_proj = keras.layers.Dense(channels, name="out_proj")
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            layer.build(inputs_shape)
        self.q_norm.build((*inputs_shape[:-1], self.num_heads, self.key_dim))
        self.k_norm.build((*inputs_shape[:-1], self.key_dim))
        self.v_norm.build((*inputs_shape[:-1], self.value_dim))
        self.out_proj.build((*inputs_shape[:-1], self.num_heads * self.value_dim))

    def call(self, inputs, bias=None):
        """
        Attend over the input positions.

        Parameters
        ----------
        inputs
            Input tensor (batch, positions, channels).
        bias
            Optional attention logit bias (batch, positions // bias_upsampling, positions // bias_upsampling, num_heads).
        """
        seq_len = inputs.shape[1]
        q = self.q_norm(keras.ops.reshape(self.q_proj(inputs), (-1, seq_len, self.num_heads, self.key_dim)))
        k = self.k_norm(self.k_proj(inputs))[:, :, None, :]
        v = self.v_norm(self.v_proj(inputs))
        q = keras.ops.transpose(apply_rope(q, self.max_position), (0, 2, 1, 3))  # (B, H, S, C)
        k = apply_rope(k, self.max_position)[:, :, 0, :]  # (B, S, C)

        logits = keras.ops.einsum("bhqc,bkc->bhqk", q, k) / math.sqrt(self.key_dim)
        if bias is not None:
            bias = keras.ops.repeat(keras.ops.repeat(bias, self.bias_upsampling, axis=1), self.bias_upsampling, axis=2)
            logits = logits + keras.ops.transpose(bias, (0, 3, 1, 2))
        logits = keras.ops.tanh(logits / self.logits_soft_cap) * self.logits_soft_cap
        weights = keras.ops.softmax(logits, axis=-1)

        outputs = keras.ops.einsum("bhqk,bkc->bqhc", weights, v)
        outputs = keras.ops.reshape(outputs, (-1, seq_len, self.num_heads * self.value_dim))
        return self.out_proj(outputs)

    def compute_output_shape(self, inputs_shape, bias_shape=None):
        """Output has the input shape."""
        return inputs_shape

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "logits_soft_cap": self.logits_soft_cap,
                "max_position": self.max_position,
                "bias_upsampling": self.bias_upsampling,
            }
        )
        return config


def _central_mask_features(distances, feature_size: int, max_distance: int):
    """Positional features of relative distances: whether each distance falls within exponentially spaced widths."""
    log_step = math.log(max_distance - feature_size + 1) / feature_size
    positions = keras.ops.arange(feature_size, dtype="float32")
    center_widths = positions + keras.ops.exp(positions * log_step)
    return keras.ops.cast(center_widths > distances[..., None], "float32")


def _relative_shift(x):
    """Turn logits per (position, relative distance) into logits per (query position, key position)."""
    *batch_shape, n_rows, n_diags = x.shape
    x = keras.ops.reshape(x, (-1, *batch_shape[1:], n_diags, n_rows))[..., 1:, :]
    return keras.ops.reshape(x, (-1, *batch_shape[1:], n_rows, n_diags - 1))[..., :n_rows]


@keras.saving.register_keras_serializable(package="crested", name="SequenceToPair")
class SequenceToPair(keras.layers.Layer):
    """
    Create a pair representation from a sequence representation, as in AlphaGenome.

    Each pair (i, j) gets the multi-head query-key products of positions i and j (with relative position terms),
    projected to `pair_dim` channels, plus projections of position i and position j.

    Parameters
    ----------
    pair_dim
        Number of channels of the pair representation.
    num_heads
        Number of query-key heads.
    head_dim
        Channels per query/key head.
    max_distance
        Maximum relative distance (in positions) of the positional features.
    """

    def __init__(
        self, pair_dim: int = 128, num_heads: int = 32, head_dim: int = 128, max_distance: int = 512, **kwargs
    ):
        super().__init__(**kwargs)
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_distance = max_distance

    def build(self, input_shape):
        """Create the projection layers and the relative position biases."""
        heads_dim = self.num_heads * self.head_dim
        self.q_proj = keras.layers.Dense(heads_dim, use_bias=False, name="q_proj")
        self.k_proj = keras.layers.Dense(heads_dim, use_bias=False, name="k_proj")
        self.pos_proj = keras.layers.Dense(heads_dim, name="pos_proj")
        self.y_q_proj = keras.layers.Dense(self.pair_dim, use_bias=False, name="y_q_proj")
        self.y_k_proj = keras.layers.Dense(self.pair_dim, use_bias=False, name="y_k_proj")
        self.pair_proj = keras.layers.Dense(self.pair_dim, name="pair_proj")
        for layer in (self.q_proj, self.k_proj, self.y_q_proj, self.y_k_proj):
            layer.build(input_shape)
        self.pos_proj.build((None, 2 * self.num_heads))
        self.pair_proj.build((None, self.num_heads))
        self.q_r_bias = self.add_weight(shape=(self.num_heads, self.head_dim), initializer="zeros", name="q_r_bias")
        self.k_r_bias = self.add_weight(shape=(self.num_heads, self.head_dim), initializer="zeros", name="k_r_bias")

    def call(self, inputs):
        """Create the pair representation (batch, positions, positions, pair_dim) from (batch, positions, channels)."""
        seq_len = inputs.shape[1]
        q = keras.ops.reshape(self.q_proj(inputs), (-1, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.reshape(self.k_proj(inputs), (-1, seq_len, self.num_heads, self.head_dim))

        # Relative position encodings for distances -seq_len..seq_len-1
        distances = keras.ops.arange(-seq_len, seq_len, dtype="float32")
        pos_features = _central_mask_features(keras.ops.abs(distances), self.num_heads, self.max_distance)
        pos_features = keras.ops.concatenate([pos_features, keras.ops.sign(distances)[:, None] * pos_features], axis=-1)
        pos_encoding = keras.ops.reshape(
            self.pos_proj(keras.ops.cast(pos_features, self.compute_dtype)),
            (2 * seq_len, self.num_heads, self.head_dim),
        )
        rel_q = _relative_shift(keras.ops.einsum("bqhc,phc->bhqp", q + self.q_r_bias, pos_encoding))
        rel_k = _relative_shift(keras.ops.einsum("bkhc,phc->bhkp", k + self.k_r_bias, pos_encoding))
        logits = keras.ops.einsum("bqhc,bkhc->bqkh", q, k) + 0.5 * (
            keras.ops.transpose(rel_q, (0, 2, 3, 1)) + keras.ops.transpose(rel_k, (0, 3, 2, 1))
        )

        x = keras.ops.gelu(inputs, approximate=False)
        y_q = self.y_q_proj(x)
        y_k = self.y_k_proj(x)
        return self.pair_proj(logits) + y_q[:, :, None, :] + y_k[:, None, :, :]

    def compute_output_shape(self, input_shape):
        """Output shape (batch, positions, positions, pair_dim)."""
        return (input_shape[0], input_shape[1], input_shape[1], self.pair_dim)

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update(
            {
                "pair_dim": self.pair_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "max_distance": self.max_distance,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="crested", name="RowAttention")
class RowAttention(keras.layers.Layer):
    """
    Single-head self-attention along the rows of a pair representation (batch, positions, positions, channels).

    Parameters
    ----------
    key_dim
        Channels of the queries and keys; also sets the logit scaling (1/sqrt(key_dim)).
    """

    def __init__(self, key_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.key_dim = key_dim

    def build(self, input_shape):
        """Create the projection layers."""
        self.q_proj = keras.layers.Dense(self.key_dim, use_bias=False, name="q_proj")
        self.k_proj = keras.layers.Dense(self.key_dim, use_bias=False, name="k_proj")
        self.v_proj = keras.layers.Dense(input_shape[-1], name="v_proj")
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            layer.build(input_shape)

    def call(self, inputs):
        """Attend over the positions in each row."""
        q, k, v = self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)
        weights = keras.ops.softmax(keras.ops.einsum("bpqc,bpkc->bpqk", q, k) / math.sqrt(self.key_dim), axis=-1)
        return keras.ops.einsum("bpqk,bpkc->bpqc", weights, v)

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update({"key_dim": self.key_dim})
        return config


def _name(name_prefix: str | None, suffix: str) -> str | None:
    return f"{name_prefix}_{suffix}" if name_prefix else None


def rms_norm_ag(inputs: keras.KerasTensor, name_prefix: str | None = None) -> keras.KerasTensor:
    """RMS normalization over the last axis with a learned scale and offset (AlphaGenome's `LayerNorm(rms_norm=True)`)."""
    current = keras.layers.RMSNormalization(epsilon=1e-5, name=_name(name_prefix, "rmsnorm"))(inputs)
    return Bias(name=_name(name_prefix, "rmsnorm_bias"))(current)


def conv_block_ag(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 5,
    activation: str = "gelu_enf",
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome convolution block: normalization + activation + standardized convolution.

    Parameters
    ----------
    inputs
        Input tensor (batch, positions, channels).
    filters
        Number of filters.
    kernel_size
        Kernel size. A kernel size of 1 uses a plain pointwise (dense) projection instead of a standardized convolution.
    activation
        Activation function. See :func:`activate`.
    normalization
        Type of normalization. See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the convolution block.
    """
    current = normalize(inputs, normalization, name=_name(name_prefix, f"{normalization}norm"))
    current = activate(current, activation, name=_name(name_prefix, "activation"))
    if kernel_size == 1:
        return keras.layers.Dense(filters, name=_name(name_prefix, "dense"))(current)
    return StandardizedConv1D(filters, kernel_size, padding="same", name=_name(name_prefix, "conv"))(current)


def down_res_block_ag(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int = 5,
    activation: str = "gelu_enf",
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome encoder block: two residual convolution blocks, the first increasing the channels to `filters`.

    The residual of the first block is the input, zero-padded to `filters` channels. Pooling happens outside the block.

    Parameters
    ----------
    inputs
        Input tensor (batch, positions, channels).
    filters
        Number of output filters, at least the number of input channels.
    kernel_size/activation/normalization
        Convolution block arguments, see :func:`conv_block_ag`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the block.
    """
    current = conv_block_ag(inputs, filters, kernel_size, activation, normalization, _name(name_prefix, "block1"))
    residual = keras.ops.pad(inputs, [[0, 0], [0, 0], [0, filters - inputs.shape[-1]]])
    current = keras.layers.Add(name=_name(name_prefix, "add1"))([current, residual])
    block2 = conv_block_ag(current, filters, kernel_size, activation, normalization, _name(name_prefix, "block2"))
    return keras.layers.Add(name=_name(name_prefix, "add2"))([current, block2])


def up_res_block_ag(
    inputs: keras.KerasTensor,
    skip: keras.KerasTensor,
    kernel_size: int = 5,
    activation: str = "gelu_enf",
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome decoder block: residual convolution, 2x upsampling and addition of the U-Net skip connection.

    The output has the channels and (doubled) resolution of the skip connection.

    Parameters
    ----------
    inputs
        Input tensor (batch, positions, channels).
    skip
        Skip connection from the encoder (batch, 2 * positions, skip channels), with at most as many channels as the input.
    kernel_size/activation/normalization
        Convolution block arguments, see :func:`conv_block_ag`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the block.
    """
    filters = skip.shape[-1]
    current = conv_block_ag(inputs, filters, kernel_size, activation, normalization, _name(name_prefix, "conv_in"))
    current = keras.layers.Add(name=_name(name_prefix, "add_in"))([current, inputs[:, :, :filters]])
    current = keras.layers.UpSampling1D(2, name=_name(name_prefix, "upsampling"))(current)
    current = Scale(name=_name(name_prefix, "residual_scale"))(current)
    pointwise = conv_block_ag(skip, filters, 1, activation, normalization, _name(name_prefix, "pointwise"))
    current = keras.layers.Add(name=_name(name_prefix, "add_skip"))([current, pointwise])
    conv_out = conv_block_ag(current, filters, kernel_size, activation, normalization, _name(name_prefix, "conv_out"))
    return keras.layers.Add(name=_name(name_prefix, "add_out"))([current, conv_out])


def mha_block_ag(
    inputs: keras.KerasTensor,
    attention_bias: keras.KerasTensor | None = None,
    num_heads: int = 8,
    key_dim: int = 128,
    value_dim: int = 192,
    max_position: int = 8192,
    bias_upsampling: int = 16,
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome residual attention block: normalization + multi-query attention + normalization.

    Parameters
    ----------
    inputs
        Input tensor (batch, positions, channels).
    attention_bias
        Optional attention logit bias per head at lower resolution (batch, positions', positions', num_heads),
        see :func:`attention_bias_block_ag`.
    num_heads/key_dim/value_dim/max_position/bias_upsampling
        Attention arguments, see :class:`MultiQueryAttention`.
    normalization
        Type of normalization. See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the attention block.
    """
    current = normalize(inputs, normalization, name=_name(name_prefix, f"{normalization}norm"))
    current = MultiQueryAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        max_position=max_position,
        bias_upsampling=bias_upsampling,
        name=_name(name_prefix, "attention"),
    )(current, attention_bias)
    current = normalize(current, normalization, name=_name(name_prefix, f"final_{normalization}norm"))
    return keras.layers.Add(name=_name(name_prefix, "add"))([inputs, current])


def mlp_block_ag(
    inputs: keras.KerasTensor,
    expansion: int = 2,
    activation: str = "relu",
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome residual feed-forward block: normalization + dense + activation + dense + normalization.

    Parameters
    ----------
    inputs
        Input tensor (batch, positions, channels).
    expansion
        Expansion factor of the hidden layer.
    activation
        Activation function of the hidden layer. See :func:`activate`.
    normalization
        Type of normalization. See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the feed-forward block.
    """
    channels = inputs.shape[-1]
    current = normalize(inputs, normalization, name=_name(name_prefix, f"{normalization}norm"))
    current = keras.layers.Dense(channels * expansion, name=_name(name_prefix, "dense1"))(current)
    current = activate(current, activation, name=_name(name_prefix, "activation"))
    current = keras.layers.Dense(channels, name=_name(name_prefix, "dense2"))(current)
    current = normalize(current, normalization, name=_name(name_prefix, f"final_{normalization}norm"))
    return keras.layers.Add(name=_name(name_prefix, "add"))([inputs, current])


def pair_update_block_ag(
    inputs: keras.KerasTensor,
    pair: keras.KerasTensor | None = None,
    pair_dim: int = 128,
    pair_pool_size: int = 16,
    num_heads: int = 32,
    head_dim: int = 128,
    max_distance: int = 512,
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome pair update block: update the pair representation from the sequence representation.

    The sequence is average pooled and turned into a pair representation (see :class:`SequenceToPair`), added to
    the existing pair representation (if any), followed by residual row attention and a residual feed-forward block.

    Parameters
    ----------
    inputs
        Sequence representation (batch, positions, channels).
    pair
        Existing pair representation (batch, positions', positions', pair_dim), or None to create one.
    pair_dim
        Channels of the pair representation.
    pair_pool_size
        Pooling factor from sequence to pair resolution.
    num_heads/head_dim/max_distance
        Sequence-to-pair arguments, see :class:`SequenceToPair`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Updated pair representation.
    """
    current = keras.layers.AveragePooling1D(pair_pool_size, name=_name(name_prefix, "pool"))(inputs)
    current = rms_norm_ag(current, _name(name_prefix, "seq2pair"))
    current = SequenceToPair(
        pair_dim=pair_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_distance=max_distance,
        name=_name(name_prefix, "seq2pair"),
    )(current)
    if pair is not None:
        current = keras.layers.Add(name=_name(name_prefix, "add_seq2pair"))([pair, current])

    row_attention = rms_norm_ag(current, _name(name_prefix, "row_attention"))
    row_attention = RowAttention(key_dim=pair_dim, name=_name(name_prefix, "row_attention"))(row_attention)
    current = keras.layers.Add(name=_name(name_prefix, "add_row_attention"))([current, row_attention])

    mlp = rms_norm_ag(current, _name(name_prefix, "pair_mlp"))
    mlp = keras.layers.Dense(2 * pair_dim, activation="relu", name=_name(name_prefix, "pair_mlp_dense1"))(mlp)
    mlp = keras.layers.Dense(pair_dim, name=_name(name_prefix, "pair_mlp_dense2"))(mlp)
    return keras.layers.Add(name=_name(name_prefix, "add_pair_mlp"))([current, mlp])


def attention_bias_block_ag(
    pair: keras.KerasTensor,
    num_heads: int = 8,
    activation: str = "gelu",
    normalization: str = "rms_batch",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    AlphaGenome attention bias block: project the pair representation to one attention logit bias per head.

    Parameters
    ----------
    pair
        Pair representation (batch, positions', positions', pair_dim).
    num_heads
        Number of attention heads to create a bias for.
    activation
        Activation function. See :func:`activate`. AlphaGenome uses the exact gelu here.
    normalization
        Type of normalization. See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Attention bias (batch, positions', positions', num_heads).
    """
    current = normalize(pair, normalization, name=_name(name_prefix, f"{normalization}norm"))
    current = activate(current, activation, name=_name(name_prefix, "activation"))
    return keras.layers.Dense(num_heads, use_bias=False, name=_name(name_prefix, "dense"))(current)
