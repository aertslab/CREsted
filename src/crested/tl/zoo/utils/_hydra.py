"""Hydra (bidirectional Mamba-2) layers for the Cerberus architecture.

Hydra: copyright 2024 Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu
Mamba-2/SSD: copyright 2024 Albert Gu, Tri Dao
Cerberus architecture: copyright Calico LLC
"""

from __future__ import annotations

import math

import keras
import numpy as np

from ._layers import DropPath, normalize

_UNSET = object()
_kernel = _UNSET


def _mamba_chunk_scan():
    """
    Return mamba_ssm's fused Triton SSD kernel, or None if unavailable (non-torch backend or no mamba_ssm).

    The kernel is wrapped as a torch custom op, which torch.compile (Keras `jit_compile=True`) treats as one opaque
    node; mamba_ssm's own autograd Function causes graph breaks on its Triton autotuner.
    Cached in a module global rather than with functools.cache, as torch.compile ignores the latter and would
    register the op again while tracing (graph breaks).
    """
    global _kernel
    if _kernel is _UNSET:
        _kernel = _load_mamba_chunk_scan()
    return _kernel


def _load_mamba_chunk_scan():
    if keras.config.backend() != "torch":
        return None
    try:
        from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd, _mamba_chunk_scan_combined_fwd
    except (ImportError, RuntimeError):
        return None
    return _register_ssd_op(_mamba_chunk_scan_combined_fwd, _mamba_chunk_scan_combined_bwd)


def _register_ssd_op(fwd, bwd):
    """Register mamba_ssm's SSD forward and backward kernels as torch custom ops with autograd."""
    import torch

    # Explicit schemas, as this module's annotations are strings (from __future__ import annotations)
    @torch.library.custom_op(
        "crested::ssd_scan",
        mutates_args=(),
        schema="(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, int chunk_size) -> Tensor",
    )
    def ssd_scan_op(x, dt, A, B, C, chunk_size):
        return fwd(x, dt, A, B, C, chunk_size)[0].contiguous()

    @ssd_scan_op.register_fake
    def _(x, dt, A, B, C, chunk_size):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    @torch.library.custom_op(
        "crested::ssd_scan_bwd",
        mutates_args=(),
        schema="(Tensor dout, Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, Tensor out, int chunk_size) -> Tensor[]",
    )
    def ssd_scan_bwd_op(dout, x, dt, A, B, C, out, chunk_size):
        grads = bwd(dout.contiguous(), x, dt, A, B, C, out, chunk_size)[:5]
        return [g.to(t.dtype).contiguous() for g, t in zip(grads, (x, dt, A, B, C), strict=True)]

    @ssd_scan_bwd_op.register_fake
    def _(dout, x, dt, A, B, C, out, chunk_size):
        return [torch.empty_like(t, memory_format=torch.contiguous_format) for t in (x, dt, A, B, C)]

    def setup_context(ctx, inputs, output):
        x, dt, A, B, C, chunk_size = inputs
        ctx.save_for_backward(x, dt, A, B, C, output)
        ctx.chunk_size = chunk_size

    def backward(ctx, dout):
        return (*ssd_scan_bwd_op(dout, *ctx.saved_tensors, ctx.chunk_size), None)

    ssd_scan_op.register_autograd(backward, setup_context=setup_context)
    return ssd_scan_op


def _segsum(x):
    """Stable segment sum: out[..., i, j] = sum(x[..., j+1:i+1]) for j <= i, -inf above the diagonal."""
    length = x.shape[-1]
    x = keras.ops.repeat(keras.ops.expand_dims(x, -1), length, axis=-1)
    x = keras.ops.where(np.tril(np.ones((length, length), dtype=bool), -1), x, 0.0)
    x = keras.ops.cumsum(x, axis=-2)
    return keras.ops.where(np.tril(np.ones((length, length), dtype=bool), 0), x, -np.inf)


def _ssd_scan_ref(x, dt, A, B, C, chunk_size):
    """Chunked SSD scan in keras.ops (Listing 1 of the Mamba-2 paper), matching `mamba_chunk_scan_combined`."""
    dtype = x.dtype
    x, dt, A, B, C = (keras.ops.cast(t, "float32") for t in (x, dt, A, B, C))
    seqlen, nheads, headdim = x.shape[1:]
    ngroups, d_state = B.shape[2:]
    B = keras.ops.repeat(B, nheads // ngroups, axis=2)
    C = keras.ops.repeat(C, nheads // ngroups, axis=2)
    x = x * keras.ops.expand_dims(dt, -1)
    a = A * dt

    # Zero-pad to whole chunks: the scan is causal, so trailing padding doesn't change the outputs we keep
    pad = (-seqlen) % chunk_size
    if pad:
        x = keras.ops.pad(x, [[0, 0], [0, pad], [0, 0], [0, 0]])
        a = keras.ops.pad(a, [[0, 0], [0, pad], [0, 0]])
        B = keras.ops.pad(B, [[0, 0], [0, pad], [0, 0], [0, 0]])
        C = keras.ops.pad(C, [[0, 0], [0, pad], [0, 0], [0, 0]])
    nchunks = (seqlen + pad) // chunk_size
    x = keras.ops.reshape(x, (-1, nchunks, chunk_size, nheads, headdim))
    B = keras.ops.reshape(B, (-1, nchunks, chunk_size, nheads, d_state))
    C = keras.ops.reshape(C, (-1, nchunks, chunk_size, nheads, d_state))
    a = keras.ops.transpose(keras.ops.reshape(a, (-1, nchunks, chunk_size, nheads)), (0, 3, 1, 2))
    a_cumsum = keras.ops.cumsum(a, axis=-1)

    # Intra-chunk outputs (diagonal blocks)
    decay = keras.ops.exp(_segsum(a))
    y_diag = keras.ops.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, decay, x)
    # State at the end of each chunk
    decay_states = keras.ops.exp(a_cumsum[..., -1:] - a_cumsum)
    states = keras.ops.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, x)
    # Inter-chunk recurrence on the chunk states
    states = keras.ops.concatenate([keras.ops.zeros_like(states[:, :1]), states], axis=1)
    decay_chunk = keras.ops.exp(_segsum(keras.ops.pad(a_cumsum[..., -1], [[0, 0], [0, 0], [1, 0]])))
    states = keras.ops.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # State -> output (off-diagonal blocks)
    y_off = keras.ops.einsum("bclhn,bchpn,bhcl->bclhp", C, states, keras.ops.exp(a_cumsum))

    y = keras.ops.reshape(y_diag + y_off, (-1, nchunks * chunk_size, nheads, headdim))[:, :seqlen]
    return keras.ops.cast(y, dtype)


def ssd_scan(x, dt, A, B, C, chunk_size: int = 256, use_kernel: bool = True):
    """
    Mamba-2 selective state space scan (SSD), without D skip or z gate.

    Uses the fused Triton kernel from `mamba_ssm` when running on the torch backend on a GPU with
    `mamba_ssm` installed, and an equivalent (slower, more memory hungry) keras.ops implementation otherwise.
    Note that on GPU the Triton kernel computes float32 products in TF32 by default (set `TRITON_F32_DEFAULT=ieee`
    for full float32), so its results differ from the keras.ops scan by ~1e-3 relative.

    Parameters
    ----------
    x
        Inputs, shape (batch, seqlen, nheads, headdim).
    dt
        Time steps (after softplus), shape (batch, seqlen, nheads).
    A
        State decay per head (negative), shape (nheads,).
    B
        Input projections, shape (batch, seqlen, ngroups, d_state).
    C
        Output projections, shape (batch, seqlen, ngroups, d_state).
    chunk_size
        Chunk length of the scan.
    use_kernel
        Whether to use the Triton kernel when available.

    Returns
    -------
    Outputs of shape (batch, seqlen, nheads, headdim).
    """
    kernel = _mamba_chunk_scan() if use_kernel else None
    if kernel is not None and x.is_cuda:
        return kernel(x, dt, A, B, C, chunk_size)
    return _ssd_scan_ref(x, dt, A, B, C, chunk_size)


@keras.saving.register_keras_serializable(package="crested", name="Hydra")
class Hydra(keras.layers.Layer):
    """
    Hydra sequence mixer: a bidirectional Mamba-2 (SSD) layer, as used in Cerberus.

    Follows baskerville's `Hydra` layer (Hwang et al. 2024), which runs the SSD scan
    forward and on the reversed sequence, shifts both by one position and adds a
    data-dependent diagonal skip term, followed by a gated RMSNorm and output projection.
    The scan uses mamba_ssm's Triton kernel where available (see :func:`ssd_scan`).
    It's highly recommended to use the `torch` backend with `jit_compile=True` when using this layer.

    Parameters
    ----------
    d_state
        SSM state dimension.
    d_conv
        Kernel width of the depthwise convolution.
    expand
        Expansion factor of the inner dimension relative to the input channels.
    headdim
        Channels per SSM head. The inner dimension must be divisible by this.
    ngroups
        Number of B/C groups shared between heads.
    chunk_size
        Chunk length of the scan.
    activation
        Activation after the depthwise convolution.
    dt_min/dt_max/dt_init_floor
        Range and floor of the initial time steps.
    **kwargs
        Extra arguments passed to keras.layers.Layer.
    """

    def __init__(
        self,
        d_state: int = 64,
        d_conv: int = 7,
        expand: float = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        activation: str = "silu",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.activation = activation
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self._activation = keras.activations.get(activation)

    def build(self, input_shape):
        """Create the projections and SSM parameters."""
        d_model = input_shape[-1]
        self.d_inner = int(self.expand * d_model)
        if self.d_inner % self.headdim != 0:
            raise ValueError(f"Inner dimension {self.d_inner} is not divisible by headdim {self.headdim}.")
        self.nheads = self.d_inner // self.headdim
        # Register the scan kernel's custom op now: registering it during torch.compile tracing breaks the graph
        _mamba_chunk_scan()
        bc_dim = 2 * self.ngroups * self.d_state  # B and C of one direction
        # Projection order: [z, x, B_fw, C_fw, B_bw, C_bw, dt_fw, dt_bw]
        self.in_proj = keras.layers.Dense(
            2 * self.d_inner + 2 * bc_dim + 2 * self.nheads, use_bias=False, dtype=self.dtype_policy, name="in_proj"
        )
        self.in_proj.build((None, None, d_model))
        self.conv1d = keras.layers.DepthwiseConv1D(self.d_conv, padding="same", dtype=self.dtype_policy, name="conv1d")
        self.conv1d.build((None, None, self.d_inner + 2 * bc_dim))

        def dt_bias_init(shape, dtype=None):
            # Inverse softplus of time steps log-uniform in [dt_min, dt_max]
            dt = keras.ops.exp(
                keras.random.uniform(shape) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
            )
            dt = keras.ops.maximum(dt, self.dt_init_floor)
            return dt + keras.ops.log(-keras.ops.expm1(-dt))

        # dt_bias and A_log feed softplus/exp, so keep them in float32 under mixed precision
        self.dt_bias = self.add_weight(shape=(self.nheads,), initializer=dt_bias_init, autocast=False, name="dt_bias")
        self.A_log = self.add_weight(shape=(self.nheads,), initializer="zeros", autocast=False, name="A_log")
        self.D = self.add_weight(shape=(self.nheads,), initializer="ones", name="D")
        self.fc_D = keras.layers.Dense(self.nheads, use_bias=False, dtype=self.dtype_policy, name="fc_D")
        self.fc_D.build((None, None, self.d_inner))
        self.norm_weight = self.add_weight(shape=(self.d_inner,), initializer="ones", name="norm_weight")
        self.out_proj = keras.layers.Dense(d_model, use_bias=False, dtype=self.dtype_policy, name="out_proj")
        self.out_proj.build((None, None, self.d_inner))

    def call(self, inputs):
        """Run the bidirectional scan."""
        seqlen = inputs.shape[1]
        d_inner, nheads, headdim = self.d_inner, self.nheads, self.headdim
        gs = self.ngroups * self.d_state

        zxbcdt = self.in_proj(inputs)
        z = zxbcdt[..., :d_inner]
        xbc = zxbcdt[..., d_inner : 2 * d_inner + 4 * gs]
        dt = zxbcdt[..., 2 * d_inner + 4 * gs :]
        xbc = self._activation(self.conv1d(xbc))
        x = xbc[..., :d_inner]

        # Both scan directions are stacked along the heads (and groups) axis, so one scan call covers both:
        # heads [0, nheads) see the sequence forward, heads [nheads, 2 * nheads) see it reversed.
        def _bidir(fw, bw, heads):
            fw = keras.ops.reshape(fw, (-1, seqlen, heads, fw.shape[-1] // heads))
            bw = keras.ops.reshape(keras.ops.flip(bw, axis=1), (-1, seqlen, heads, bw.shape[-1] // heads))
            return keras.ops.concatenate([fw, bw], axis=2)

        xs = _bidir(x, x, nheads)
        bc_fw, bc_bw = xbc[..., d_inner : d_inner + 2 * gs], xbc[..., d_inner + 2 * gs :]
        B = _bidir(bc_fw[..., :gs], bc_bw[..., :gs], self.ngroups)
        C = _bidir(bc_fw[..., gs:], bc_bw[..., gs:], self.ngroups)
        dt = keras.ops.concatenate([dt[..., :nheads], keras.ops.flip(dt[..., nheads:], axis=1)], axis=-1)
        dt = keras.ops.softplus(dt + keras.ops.tile(self.dt_bias, 2))
        A = keras.ops.tile(-keras.ops.exp(keras.ops.cast(self.A_log, "float32")), 2)

        y = ssd_scan(xs, dt, A, B, C, chunk_size=self.chunk_size)
        # Shift by one position, so each direction excludes the current position (the diagonal is added below)
        y = keras.ops.concatenate([keras.ops.zeros_like(y[:, :1]), y[:, :-1]], axis=1)
        y_fw = keras.ops.reshape(y[:, :, :nheads], (-1, seqlen, d_inner))
        y_bw = keras.ops.flip(keras.ops.reshape(y[:, :, nheads:], (-1, seqlen, d_inner)), axis=1)
        diag = keras.ops.repeat(self.fc_D(x) + self.D, headdim, axis=-1)
        y = y_fw + y_bw + x * keras.ops.cast(diag, x.dtype)

        # Gated RMSNorm (norm before gate), then output projection
        y = keras.ops.cast(y, "float32")
        y = y * keras.ops.rsqrt(keras.ops.mean(keras.ops.square(y), axis=-1, keepdims=True) + 1e-5)
        y = y * keras.ops.cast(self.norm_weight, "float32") * keras.ops.silu(keras.ops.cast(z, "float32"))
        return self.out_proj(keras.ops.cast(y, self.compute_dtype))

    def compute_output_shape(self, input_shape):
        """Output shape is the input shape."""
        return input_shape

    def get_config(self):
        """Get the config for this layer."""
        config = super().get_config()
        config.update(
            {
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
                "headdim": self.headdim,
                "ngroups": self.ngroups,
                "chunk_size": self.chunk_size,
                "activation": self.activation,
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "dt_init_floor": self.dt_init_floor,
            }
        )
        return config


def hydra_block(
    inputs: keras.KerasTensor,
    d_state: int = 64,
    d_conv: int = 7,
    expand: float = 2,
    headdim: int = 64,
    chunk_size: int = 256,
    dropout: float = 0,
    drop_path: float = 0,
    norm_epsilon: float = 1e-4,
    normalization: str = "rms",
    name_prefix: str | None = None,
) -> keras.KerasTensor:
    """
    Residual Hydra block: RMSNorm -> Hydra -> dropout -> drop path, added to the input.

    It's highly recommended to use the `torch` backend with `jit_compile=True` when using this block.

    Parameters
    ----------
    inputs
        Input tensor.
    d_state/d_conv/expand/headdim/chunk_size
        Hydra layer arguments, see :class:`Hydra`.
    dropout
        Dropout rate on the Hydra output.
    drop_path
        Stochastic depth rate: the probability of dropping the whole residual branch per sample.
    norm_epsilon
        Epsilon of the normalization layer.
    normalization
        Type of normalization (default is 'rms'). See :func:`normalize`.
    name_prefix
        Prefix for layer names.

    Returns
    -------
    Output tensor after applying the Hydra block.
    """
    current = normalize(
        inputs, normalization, epsilon=norm_epsilon, name=f"{name_prefix}_{normalization}norm" if name_prefix else None
    )
    current = Hydra(
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        chunk_size=chunk_size,
        name=name_prefix + "_hydra" if name_prefix else None,
    )(current)
    if dropout > 0:
        current = keras.layers.Dropout(dropout, name=name_prefix + "_dropout" if name_prefix else None)(current)
    if drop_path > 0:
        current = DropPath(drop_path, name=name_prefix + "_droppath" if name_prefix else None)(current)
    return keras.layers.Add(name=name_prefix + "_add" if name_prefix else None)([inputs, current])
