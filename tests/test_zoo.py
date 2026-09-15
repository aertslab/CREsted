"""Test that all crested.tl.zoo model architectures build and survive a save/load round trip."""

import pytest

import crested
from crested.tl import zoo

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
