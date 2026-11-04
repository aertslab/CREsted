"""Test the AlphaGenome layers and architecture."""

import keras
import numpy as np
import pytest

import crested
from crested.tl import zoo
from crested.tl.zoo.utils import RMSBatchNormalization, StandardizedConv1D, apply_rope

SMALL_ALPHAGENOME = {
    "start_filters": 16,
    "filters_step": 8,
    "num_transformer_blocks": 3,
    "num_heads": 2,
    "key_dim": 8,
    "value_dim": 12,
    "pair_dim": 8,
    "num_pair_heads": 4,
}


def _np(x):
    return keras.ops.convert_to_numpy(x)


def test_rms_batch_normalization():
    """RMSBatchNormalization should scale by the moving mean square at inference, and by the batch mean square in training."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 10, 3)).astype("float32")
    layer = RMSBatchNormalization(momentum=0.9, epsilon=1e-5)
    layer.build(x.shape)
    gamma, beta, moving = (rng.uniform(0.5, 2, size=3).astype("float32") for _ in range(3))
    layer.gamma.assign(gamma)
    layer.beta.assign(beta)
    layer.moving_variance.assign(moving)

    np.testing.assert_allclose(_np(layer(x)), x * gamma / np.sqrt(moving + 1e-5) + beta, rtol=1e-5)

    batch_ms = (x**2).mean(axis=(0, 1))
    np.testing.assert_allclose(_np(layer(x, training=True)), x * gamma / np.sqrt(batch_ms + 1e-5) + beta, rtol=1e-5)
    np.testing.assert_allclose(_np(layer.moving_variance), 0.9 * moving + 0.1 * batch_ms, rtol=1e-5)


def test_standardized_conv1d(tmp_path):
    """StandardizedConv1D should convolve with the standardized, scaled kernel, and save/load its scale."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 20, 3)).astype("float32")
    inputs = keras.Input((20, 3))
    model = keras.Model(inputs, StandardizedConv1D(6, 5, padding="same", name="conv")(inputs))
    layer = model.get_layer("conv")
    scale, bias = rng.uniform(0.5, 2, size=6).astype("float32"), rng.normal(size=6).astype("float32")
    layer.scale.assign(scale)
    layer.bias.assign(bias)

    kernel = _np(layer.kernel)
    kernel = (kernel - kernel.mean((0, 1))) / np.sqrt(kernel.var((0, 1)) * 15) * scale
    x_padded = np.pad(x, ((0, 0), (2, 2), (0, 0)))
    expected = np.stack([np.einsum("bki,kio->bo", x_padded[:, i : i + 5], kernel) for i in range(20)], axis=1) + bias
    np.testing.assert_allclose(_np(model(x)), expected, rtol=1e-4, atol=1e-5)

    model.save(tmp_path / "model.keras")
    loaded = crested.utils.load_model(tmp_path / "model.keras", compile=False)
    np.testing.assert_allclose(_np(loaded.get_layer("conv").scale), scale)
    np.testing.assert_allclose(_np(loaded(x)), _np(model(x)), rtol=1e-6)


def test_apply_rope_relative():
    """Rotary embeddings should keep norms, and make query-key products depend only on the relative position."""
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(1, 1, 1, 16)), rng.normal(size=(1, 1, 1, 16))
    seq_len = 12
    q_rot = _np(apply_rope(keras.ops.convert_to_tensor(np.repeat(q, seq_len, axis=1), "float32")))[0, :, 0]
    k_rot = _np(apply_rope(keras.ops.convert_to_tensor(np.repeat(k, seq_len, axis=1), "float32")))[0, :, 0]
    np.testing.assert_allclose(np.linalg.norm(q_rot, axis=-1), np.linalg.norm(q), rtol=1e-5)
    products = q_rot @ k_rot.T
    for offset in (-3, 0, 5):
        diagonal = np.diagonal(products, offset)
        np.testing.assert_allclose(diagonal, diagonal[0], rtol=1e-4, atol=1e-5)


def test_alphagenome_outputs():
    """Each head should predict at its own resolutions, followed by a symmetric contact map output."""
    model = zoo.alphagenome(4096, [3, 5], resolutions=[(1, 128), (128,)], num_contact_maps=2, **SMALL_ALPHAGENOME)
    assert model.output_names == ["head_0_1bp", "head_0_128bp", "head_1", "contact_maps"]
    assert model.output_shape == [(None, 4096, 3), (None, 32, 3), (None, 32, 5), (None, 2, 2, 2)]
    # 128 bp only: no decoder
    model_128 = zoo.alphagenome(4096, 3, resolutions=(128,), **SMALL_ALPHAGENOME)
    assert not any(layer.name.startswith("decoder") for layer in model_128.layers)
    # Named heads
    model_named = zoo.alphagenome(4096, {"atac": 3, "chip": 5}, resolutions=[(1, 128), (128,)], **SMALL_ALPHAGENOME)
    assert model_named.output_names == ["atac_1bp", "atac_128bp", "chip"]
    # Splice heads (and the decoder they need) also with only 128 bp track heads
    model_splice = zoo.alphagenome(
        4096, 3, resolutions=(128,), splice_sites=True, num_splice_site_usage=4, num_contact_maps=2, **SMALL_ALPHAGENOME
    )
    assert model_splice.output_names == ["head", "splice_sites", "splice_site_usage", "contact_maps"]
    assert model_splice.output_shape[1:3] == [(None, 4096, 5), (None, 4096, 4)]
    x = np.eye(4, dtype="float32")[np.random.default_rng(0).integers(0, 4, (1, 4096))]
    splice = _np(model_splice(x)[1])
    np.testing.assert_allclose(splice.sum(-1), 1, rtol=1e-5)

    x = np.eye(4, dtype="float32")[np.random.default_rng(0).integers(0, 4, (2, 4096))]
    outputs = [_np(o) for o in model(x)]
    assert all((o >= 0).all() for o in outputs[:3])
    np.testing.assert_allclose(outputs[3], outputs[3].transpose(0, 2, 1, 3), rtol=1e-5, atol=1e-6)


def test_alphagenome_resolution_args():
    """The trunk and pair resolutions should follow num_down_blocks and pair_pool_size."""
    model = zoo.alphagenome(
        2048, 3, resolutions=(1, 32), num_down_blocks=4, pair_pool_size=8, num_contact_maps=2, **SMALL_ALPHAGENOME
    )
    assert model.output_shape == [(None, 2048, 3), (None, 64, 3), (None, 8, 8, 2)]
    layer_names = [layer.name for layer in model.layers]
    assert "encoder_4_pool" in layer_names and "encoder_5_pool" not in layer_names
    assert "decoder_5_add_out" in layer_names and "decoder_6_add_out" not in layer_names
    x = np.eye(4, dtype="float32")[np.random.default_rng(0).integers(0, 4, (1, 2048))]
    assert all(np.isfinite(_np(o)).all() for o in model(x))


def test_alphagenome_fit():
    """A small AlphaGenome should train, updating the RMSBatchNormalization moving averages."""
    model = zoo.alphagenome(2048, 2, resolutions=(128,), **SMALL_ALPHAGENOME)
    model.compile(optimizer="adam", loss="mse")
    moving = model.get_layer("stem_block_rms_batchnorm").moving_variance
    before = _np(moving).copy()
    x = np.eye(4, dtype="float32")[np.random.default_rng(0).integers(0, 4, (2, 2048))]
    model.fit(x, np.ones((2, 16, 2), dtype="float32"), epochs=1, verbose=0)
    assert not np.allclose(_np(moving), before)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"seq_len": 3000, "num_classes": 2}, "multiple of 2048"),
        ({"seq_len": 2048, "num_classes": 2, "resolutions": (1, 32)}, "Resolutions must be"),
        ({"seq_len": 2048, "num_classes": 2, "pair_pool_size": 32}, "multiple of 4096"),
        ({"seq_len": 2048, "num_classes": [2, 3], "resolutions": [(1,)]}, "sequences of resolutions"),
    ],
)
def test_alphagenome_invalid(kwargs, match):
    """Invalid sequence lengths and resolutions should raise."""
    with pytest.raises(ValueError, match=match):
        zoo.alphagenome(**kwargs)
