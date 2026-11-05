"""Test the Hydra (bidirectional Mamba-2) layers and the Cerberus architecture."""

import keras
import numpy as np
import pytest

import crested
from crested.tl.zoo.utils import DropPath, Hydra, ssd_scan
from crested.tl.zoo.utils._hydra import _mamba_chunk_scan


def _softplus(x):
    return np.log1p(np.exp(x))


def _silu(x):
    return x / (1 + np.exp(-x))


def _naive_scan(x, dt, A, B, C):
    """Sequential SSM recurrence: h_t = exp(dt_t * A) h_{t-1} + dt_t x_t B_t, y_t = C_t h_t."""
    batch, seqlen, nheads, headdim = x.shape
    rep = nheads // B.shape[2]
    B, C = np.repeat(B, rep, axis=2), np.repeat(C, rep, axis=2)
    state = np.zeros((batch, nheads, headdim, B.shape[-1]))
    y = np.zeros_like(x)
    for t in range(seqlen):
        decay = np.exp(dt[:, t] * A)[..., None, None]
        state = state * decay + (dt[:, t, :, None] * x[:, t])[..., None] * B[:, t, :, None, :]
        y[:, t] = np.einsum("bhpn,bhn->bhp", state, C[:, t])
    return y


def _random_scan_inputs(seqlen, nheads=4, headdim=3, ngroups=2, d_state=2, batch=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, seqlen, nheads, headdim)).astype("float32")
    dt = _softplus(rng.normal(size=(batch, seqlen, nheads))).astype("float32")
    A = -np.exp(rng.normal(size=(nheads,))).astype("float32")
    B = rng.normal(size=(batch, seqlen, ngroups, d_state)).astype("float32")
    C = rng.normal(size=(batch, seqlen, ngroups, d_state)).astype("float32")
    return x, dt, A, B, C


def _hydra_reference(layer, u):
    """Numpy port of baskerville's Hydra forward (directions stacked on the batch axis)."""
    w = {v.path.split("/", 1)[1]: keras.ops.convert_to_numpy(v).astype("float64") for v in layer.weights}
    batch, seqlen, _ = u.shape
    d_inner, nheads, headdim = layer.d_inner, layer.nheads, layer.headdim
    gs = layer.ngroups * layer.d_state

    zxbcdt = u @ w["in_proj/kernel"]
    z, xbc, dt = np.split(zxbcdt, [d_inner, 2 * d_inner + 4 * gs], axis=-1)
    dt = np.concatenate([dt[:, :, :nheads], dt[:, ::-1, nheads:]], axis=0)
    dt = _softplus(dt + w["dt_bias"])

    k = layer.d_conv
    xpad = np.pad(xbc, [(0, 0), (k // 2, k // 2), (0, 0)])
    kernel = w["conv1d/kernel"][:, :, 0]
    xbc = sum(xpad[:, j : j + seqlen] * kernel[j] for j in range(k)) + w["conv1d/bias"]
    xbc = _silu(xbc)

    x, bc = xbc[..., :d_inner], xbc[..., d_inner:]
    x_og = x
    x = np.concatenate([x, x[:, ::-1]], axis=0)
    bc = np.concatenate([bc[..., : 2 * gs], bc[:, ::-1, 2 * gs :]], axis=0)
    B, C = bc[..., :gs], bc[..., gs:]
    y = _naive_scan(
        x.reshape(2 * batch, seqlen, nheads, headdim),
        dt,
        -np.exp(w["A_log"]),
        B.reshape(2 * batch, seqlen, layer.ngroups, layer.d_state),
        C.reshape(2 * batch, seqlen, layer.ngroups, layer.d_state),
    ).reshape(2 * batch, seqlen, d_inner)
    y = np.roll(y, 1, axis=1)
    y[:, 0] = 0
    y = y[:batch] + y[batch:, ::-1] + x_og * np.repeat(x_og @ w["fc_D/kernel"] + w["D"], headdim, axis=-1)

    y = y / np.sqrt(np.mean(y**2, axis=-1, keepdims=True) + 1e-5) * w["norm_weight"] * _silu(z)
    return y @ w["out_proj/kernel"]


@pytest.mark.parametrize("seqlen, chunk_size", [(32, 8), (29, 8), (5, 16)])
def test_ssd_scan_matches_recurrence(seqlen, chunk_size):
    """The keras.ops chunked scan should equal the sequential recurrence, also when seqlen isn't a multiple of chunk_size."""
    x, dt, A, B, C = _random_scan_inputs(seqlen)
    y = ssd_scan(*(keras.ops.convert_to_tensor(t) for t in (x, dt, A, B, C)), chunk_size=chunk_size, use_kernel=False)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(y), _naive_scan(x, dt, A, B, C), rtol=1e-4, atol=1e-4)


def test_hydra_matches_reference():
    """Hydra should match baskerville's formulation, which stacks the two directions along the batch axis."""
    rng = np.random.default_rng(1)
    layer = Hydra(d_state=2, d_conv=7, expand=2, headdim=4, ngroups=1, chunk_size=8)
    u = rng.normal(size=(2, 21, 6)).astype("float32")
    layer.build(u.shape)
    # Randomise the weights that are initialised to constants, so they're tested too
    for variable in (layer.A_log, layer.D, layer.norm_weight, layer.conv1d.bias):
        variable.assign(rng.normal(size=variable.shape).astype("float32"))
    out = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(u)))
    assert out.shape == u.shape
    np.testing.assert_allclose(out, _hydra_reference(layer, u.astype("float64")), rtol=1e-4, atol=1e-4)


def _small_cerberus(**kwargs):
    return crested.tl.zoo.cerberus(
        seq_len=1024,
        start_filters=48,
        filters=96,
        filters_divisible_by=8,
        headdim=16,
        num_hydra_blocks=2,
        chunk_size=8,
        **kwargs,
    )


def test_cerberus_shapes_and_training():
    """A small Cerberus should give 32bp-resolution outputs per head and train for a step."""
    model = _small_cerberus(num_classes=[3, 2], drop_path=0.2)
    assert model.output_shape == [(None, 32, 3), (None, 32, 2)]
    assert model.get_layer("hydra_2_droppath").rate == pytest.approx(0.2)
    assert "hydra_1_droppath" not in [layer.name for layer in model.layers]  # drop path ramps up from 0

    x = np.eye(4, dtype="float32")[np.random.default_rng(3).integers(0, 4, (2, 1024))]
    y = [np.ones((2, 32, 3), dtype="float32"), np.ones((2, 32, 2), dtype="float32")]
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_cerberus_save_load(tmp_path):
    """Predictions of a saved Cerberus should be unchanged after loading."""
    model = _small_cerberus(num_classes=4)
    x = np.eye(4, dtype="float32")[np.random.default_rng(4).integers(0, 4, (1, 1024))]
    model.save(tmp_path / "model.keras")
    loaded = crested.utils.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6)


def _assert_close_tf32(actual, desired, tol=1e-2):
    """Relative error norm check: on GPU the Triton kernel's float32 dots run in TF32 (~1e-3 relative precision)."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert np.linalg.norm(actual - desired) / np.linalg.norm(desired) < tol


def _cuda_kernel_available():
    if _mamba_chunk_scan() is None:
        return False
    import torch

    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_kernel_available(), reason="Needs the torch backend, a GPU and mamba_ssm.")
@pytest.mark.parametrize("seqlen", [64, 50])
def test_ssd_scan_kernel_matches_reference(seqlen):
    """mamba_ssm's Triton kernel and the keras.ops scan should agree."""
    inputs = [keras.ops.convert_to_tensor(t) for t in _random_scan_inputs(seqlen, nheads=4, headdim=16)]
    y_kernel = ssd_scan(*inputs, chunk_size=16, use_kernel=True)
    y_ref = ssd_scan(*inputs, chunk_size=16, use_kernel=False)
    _assert_close_tf32(keras.ops.convert_to_numpy(y_kernel), keras.ops.convert_to_numpy(y_ref))


@pytest.mark.skipif(not _cuda_kernel_available(), reason="Needs the torch backend, a GPU and mamba_ssm.")
def test_hydra_kernel_gradients_match_reference():
    """Gradients through the Triton kernel should match those through the keras.ops scan."""
    import torch

    inputs = [torch.tensor(t, device="cuda", requires_grad=True) for t in _random_scan_inputs(48, nheads=4, headdim=16)]
    grads = []
    for use_kernel in (True, False):
        y = ssd_scan(*inputs, chunk_size=16, use_kernel=use_kernel)
        grads.append([g.cpu().numpy() for g in torch.autograd.grad(y.square().sum(), inputs)])
    for g_kernel, g_ref in zip(*grads, strict=True):
        _assert_close_tf32(g_kernel, g_ref)


def test_drop_path():
    """DropPath should drop whole samples at the given rate, scale kept ones by 1/(1-rate), and pass through at inference."""
    layer = DropPath(0.25)
    x = keras.ops.ones((4000, 8, 3))
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer(x)), keras.ops.convert_to_numpy(x))
    y = keras.ops.convert_to_numpy(layer(x, training=True)).reshape(4000, -1)
    np.testing.assert_array_equal(y.min(axis=1), y.max(axis=1))  # all or nothing per sample
    kept = y[:, 0] != 0
    np.testing.assert_allclose(y[kept], 1 / 0.75, rtol=1e-6)
    assert abs(1 - kept.mean() - 0.25) < 0.03
    assert DropPath.from_config(layer.get_config()).rate == 0.25


def test_hydra_sublayers_follow_dtype_policy():
    """Hydra's projections should use the layer's (mixed precision) dtype policy."""
    layer = Hydra(d_state=2, expand=2, headdim=4, chunk_size=8, dtype="mixed_bfloat16")
    layer.build((None, 16, 6))
    for sublayer in (layer.in_proj, layer.conv1d, layer.fc_D, layer.out_proj):
        assert sublayer.compute_dtype == "bfloat16"
        assert all(w.dtype == "float32" for w in sublayer.weights)


@pytest.mark.skipif(keras.config.backend() != "torch", reason="torch.compile only")
def test_batchnorm_and_drop_path_compile_without_graph_breaks():
    """BatchNormalization (tuple reduction axes) and DropPath should trace into a single torch.compile graph."""
    import torch

    bn = keras.layers.BatchNormalization(momentum=0.0)
    drop_path = DropPath(0.5)
    x = keras.ops.convert_to_tensor(np.random.default_rng(5).normal(size=(4, 8, 3)).astype("float32"))
    bn.build(x.shape)
    assert isinstance(bn._reduction_axes, tuple)
    torch._dynamo.reset()
    compiled = torch.compile(lambda t: drop_path(bn(t, training=True), training=True), fullgraph=True, backend="eager")
    assert compiled(x).shape == x.shape


@pytest.mark.skipif(not _cuda_kernel_available(), reason="Needs the torch backend, a GPU and mamba_ssm.")
def test_ssd_scan_custom_op():
    """The Triton scan custom op should pass torch's custom op checks (fake tensors, autograd registration)."""
    import torch

    inputs = [torch.tensor(t, device="cuda", requires_grad=True) for t in _random_scan_inputs(48, nheads=4, headdim=16)]
    torch.library.opcheck(torch.ops.crested.ssd_scan.default, (*inputs, 16))


@pytest.mark.skipif(not _cuda_kernel_available(), reason="Needs the torch backend, a GPU and mamba_ssm.")
def test_cerberus_compiles_as_one_graph():
    """A Cerberus training forward and backward pass should compile without graph breaks and match eager."""
    import torch

    model = _small_cerberus(num_classes=4, drop_path=0.2)
    x = torch.tensor(np.eye(4, dtype="float32")[np.random.default_rng(6).integers(0, 4, (2, 1024))], device="cuda")

    def loss_and_grads(fn):
        loss = fn(x).square().mean()
        return loss, torch.autograd.grad(loss, [v.value for v in model.trainable_variables])

    torch._dynamo.reset()
    compiled = torch.compile(lambda t: model(t, training=False), fullgraph=True, backend="eager")
    loss, grads = loss_and_grads(lambda t: model(t, training=False))
    loss_c, grads_c = loss_and_grads(compiled)
    _assert_close_tf32(loss_c.item(), loss.item(), tol=1e-5)
    for g_c, g in zip(grads_c, grads, strict=True):
        _assert_close_tf32(g_c.cpu().numpy(), g.cpu().numpy(), tol=1e-4)
    torch.compile(lambda t: model(t, training=True), fullgraph=True, backend="eager")(x)  # dropout + drop path
