"""Test that all crested.tl.zoo model architectures build and survive a save/load round trip."""

import keras
import pytest

import crested
from crested.tl import zoo
from crested.tl.zoo.utils import normalize

# Only seq_len/num_classes (which have no defaults) are overridden; all other
# hyperparameters use the architecture's own defaults.
ZOO_ARCHITECTURES = [
    ("simple_convnet", zoo.simple_convnet, {"seq_len": 500, "num_classes": 2}),
    ("basenji", zoo.basenji, {"seq_len": 131072, "num_classes": 2}),
    ("deeptopic_cnn", zoo.deeptopic_cnn, {"seq_len": 500, "num_classes": 2}),
    ("deeptopic_lstm", zoo.deeptopic_lstm, {"seq_len": 500, "num_classes": 2}),
    ("dilated_cnn", zoo.dilated_cnn, {"seq_len": 2114, "num_classes": 2}),
    (
        "dilated_cnn_decoupled",
        zoo.dilated_cnn_decoupled,
        {"seq_len": 2114, "num_classes": 2},
    ),
    ("enformer", zoo.enformer, {"seq_len": 196608, "num_classes": 2}),
    ("borzoi", zoo.borzoi, {"seq_len": 524288, "num_classes": 2}),
    ("borzoi_prime", zoo.borzoi_prime, {"seq_len": 524288, "num_classes": 2}),
    ("cerberus", zoo.cerberus, {"seq_len": 8192, "num_classes": 2}),
    ("alphagenome", zoo.alphagenome, {"seq_len": 2048, "num_classes": 2}),
]


@pytest.mark.parametrize(
    "factory, kwargs",
    [(factory, kwargs) for _, factory, kwargs in ZOO_ARCHITECTURES],
    ids=[name for name, _, _ in ZOO_ARCHITECTURES],
)
def test_zoo_model_save_load(factory, kwargs, tmp_path):
    """All zoo architectures should build and their saved .keras file should load back via crested.utils.load_model.

    Regression test for #141/#142: the custom attention layers (AttentionPool1D,
    MultiheadAttention) used by enformer/borzoi/borzoi_prime previously failed to
    deserialize because they lacked @keras.saving.register_keras_serializable().
    """
    model = factory(**kwargs)
    model_path = tmp_path / "model.keras"
    model.save(model_path)

    loaded_model = crested.utils.load_model(model_path)

    assert loaded_model.output_shape == model.output_shape


@pytest.mark.parametrize(
    "normalization, layer_type, kwargs",
    [
        ("batch", keras.layers.BatchNormalization, {}),
        ("layer", keras.layers.LayerNormalization, {}),
        ("rms", keras.layers.RMSNormalization, {"epsilon": 1e-4}),
        ("group", keras.layers.GroupNormalization, {"groups": 2}),
        ("unit", keras.layers.UnitNormalization, {}),
    ],
)
def test_normalize(normalization, layer_type, kwargs):
    """Test that normalize applies the normalization layer picked by name, passing on the keyword arguments."""
    inputs = keras.Input((10, 8))
    model = keras.Model(inputs, normalize(inputs, normalization, name="norm", **kwargs))
    layer = model.get_layer("norm")
    assert isinstance(layer, layer_type)
    for key, value in kwargs.items():
        assert getattr(layer, key) == value


@pytest.mark.parametrize("normalization", ["none", None])
def test_normalize_none(normalization):
    """Test that normalize with 'none' or None returns the input unchanged."""
    inputs = keras.Input((10, 8))
    assert normalize(inputs, normalization) is inputs


def test_normalize_unknown():
    """An unknown normalization name should raise."""
    with pytest.raises(ValueError, match="Unrecognized normalization"):
        normalize(keras.Input((10, 8)), "foo")


@pytest.mark.parametrize(
    "normalization, layer_type",
    [("layer", keras.layers.LayerNormalization), ("rms", keras.layers.RMSNormalization), ("none", None)],
)
def test_block_normalization(normalization, layer_type):
    """conv_block and dense_block should use the normalization picked by name."""
    model = zoo.simple_convnet(seq_len=500, num_classes=2, normalization=normalization)
    norm_layers = [layer for layer in model.layers if "normalization" in type(layer).__name__.lower()]
    if layer_type is None:
        assert not norm_layers
    else:
        assert norm_layers and all(isinstance(layer, layer_type) for layer in norm_layers)
