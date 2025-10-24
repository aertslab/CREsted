"""Porting weights from the official Baskerville-based Borzoi model to the CREsted-based model."""
# Port weights from the official Baskerville-based Borzoi model to the CREsted-based model.
# Expects environment with CREsted

# Load modules and paths
import os

import h5py
import keras
import numpy as np

import crested

print(f"CREsted version {crested.__version__}")

# Dir to save your new models
output_dir = "xxx"
# Borzoi saved models dir: should have folders f0/f1/f2/f3
saved_models_dir = "xxx"  # From https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5, for f0-f3 and model0/model1

# Select folds to port, getting both human and mouse models
weights_paths = [
    (
        os.path.join(saved_models_dir, f"f{fold}/model0_best.h5"),
        os.path.join(saved_models_dir, f"f{fold}/model1_best.h5"),
    )
    for fold in range(4)
]


# Set up mappings of old layers (in hdf5) to new (in CREsted model)
conv_lookup = {
    "conv1d": "stem_conv",
    "conv1d_1": "tower_conv_1_conv",
    "conv1d_2": "tower_conv_2_conv",
    "conv1d_3": "tower_conv_3_conv",
    "conv1d_4": "tower_conv_4_conv",
    "conv1d_5": "tower_conv_5_conv",
    "conv1d_6": "tower_conv_6_conv",
    "separable_conv1d": "upsampling_separable_1",
    "separable_conv1d_1": "upsampling_separable_2",
    "conv1d_7": "final_conv_conv",
}

bnorm_lookup = {
    "sync_batch_normalization": "tower_conv_1_batchnorm",
    "sync_batch_normalization_1": "tower_conv_2_batchnorm",
    "sync_batch_normalization_2": "tower_conv_3_batchnorm",
    "sync_batch_normalization_3": "tower_conv_4_batchnorm",
    "sync_batch_normalization_4": "tower_conv_5_batchnorm",
    "sync_batch_normalization_5": "tower_conv_6_batchnorm",
    "sync_batch_normalization_6": "upsampling_conv_1_batchnorm",
    "sync_batch_normalization_7": "unet_skip_2_batchnorm",
    "sync_batch_normalization_8": "upsampling_conv_2_batchnorm",
    "sync_batch_normalization_9": "unet_skip_1_batchnorm",
    "sync_batch_normalization_10": "final_conv_batchnorm",
}

lnorm_lookup = {
    "layer_normalization": "transformer_mha_1_layernorm",
    "layer_normalization_1": "transformer_ff_1_layernorm",
    "layer_normalization_2": "transformer_mha_2_layernorm",
    "layer_normalization_3": "transformer_ff_2_layernorm",
    "layer_normalization_4": "transformer_mha_3_layernorm",
    "layer_normalization_5": "transformer_ff_3_layernorm",
    "layer_normalization_6": "transformer_mha_4_layernorm",
    "layer_normalization_7": "transformer_ff_4_layernorm",
    "layer_normalization_8": "transformer_mha_5_layernorm",
    "layer_normalization_9": "transformer_ff_5_layernorm",
    "layer_normalization_10": "transformer_mha_6_layernorm",
    "layer_normalization_11": "transformer_ff_6_layernorm",
    "layer_normalization_12": "transformer_mha_7_layernorm",
    "layer_normalization_13": "transformer_ff_7_layernorm",
    "layer_normalization_14": "transformer_mha_8_layernorm",
    "layer_normalization_15": "transformer_ff_8_layernorm",
}

mha_lookup = {
    "multihead_attention": "transformer_mha_1_mhsa",
    "multihead_attention_1": "transformer_mha_2_mhsa",
    "multihead_attention_2": "transformer_mha_3_mhsa",
    "multihead_attention_3": "transformer_mha_4_mhsa",
    "multihead_attention_4": "transformer_mha_5_mhsa",
    "multihead_attention_5": "transformer_mha_6_mhsa",
    "multihead_attention_6": "transformer_mha_7_mhsa",
    "multihead_attention_7": "transformer_mha_8_mhsa",
}

# Originally Dense layers (dim [input, filters]) that are basically pointwise convolutions (dim [1, input, filters])
# So we want to expand these with a new dimension before saving as an actual pointwise conv
pointwise_lookup = {
    "dense": "transformer_ff_1_pointwise_1",
    "dense_1": "transformer_ff_1_pointwise_2",
    "dense_2": "transformer_ff_2_pointwise_1",
    "dense_3": "transformer_ff_2_pointwise_2",
    "dense_4": "transformer_ff_3_pointwise_1",
    "dense_5": "transformer_ff_3_pointwise_2",
    "dense_6": "transformer_ff_4_pointwise_1",
    "dense_7": "transformer_ff_4_pointwise_2",
    "dense_8": "transformer_ff_5_pointwise_1",
    "dense_9": "transformer_ff_5_pointwise_2",
    "dense_10": "transformer_ff_6_pointwise_1",
    "dense_11": "transformer_ff_6_pointwise_2",
    "dense_12": "transformer_ff_7_pointwise_1",
    "dense_13": "transformer_ff_7_pointwise_2",
    "dense_14": "transformer_ff_8_pointwise_1",
    "dense_15": "transformer_ff_8_pointwise_2",
    "dense_16": "upsampling_conv_1_conv",
    "dense_17": "unet_skip_2_conv",
    "dense_18": "upsampling_conv_2_conv",
    "dense_19": "unet_skip_1_conv",
    "dense_20": "head_0",
}

mouse_lookup = {"dense_21": "head_1"}


# Build model porting functions
# copy convolutional/dense layers weights
def copy_convdense(mod, layers, name):
    """Copy convolution or dense layers."""
    if len(mod.weights) == 1:
        # Assuming a 1-weight conv/dense is biasless
        conv_w = layers[name][name]["kernel:0"][...]
        conv = [conv_w]
    elif len(mod.weights) == 2:
        # Standard behaviour for conv or dense layers
        conv_w = layers[name][name]["kernel:0"][...]
        conv_b = layers[name][name]["bias:0"][...]
        conv = [conv_w, conv_b]
    elif len(mod.weights) == 3:
        # Assuming that 3-weight conv is a separable conv
        conv_w_d = layers[name][name]["depthwise_kernel:0"][...]  # Depthwise kernel
        conv_w_p = layers[name][name]["pointwise_kernel:0"][...]  # Pointwise kernel
        conv_b = layers[name][name]["bias:0"][...]
        conv = [conv_w_d, conv_w_p, conv_b]

    assert (
        conv[0].shape == mod.weights[0].shape
    ), f"shape {conv[0].shape} != {mod.weights[0].shape}"
    if len(mod.weights) >= 2:
        assert (
            conv[1].shape == mod.weights[1].shape
        ), f"shape {conv[1].shape} != {mod.weights[1].shape}"
    if len(mod.weights) >= 3:
        assert (
            conv[2].shape == mod.weights[2].shape
        ), f"shape {conv[2].shape} != {mod.weights[2].shape}"
    mod.set_weights(conv)


def copy_dense_to_pointwise(mod, layers, name):
    """Copy a dense layer to a pointwise convolutional layer.

    Adds a dimension at the start. Lets you copy a dense layer used as pointwise conv (shape [input, filters])
    to a true pointwise conv layer (shape [width, input, filters] = [1, input, filters]).
    """
    dense_w = np.expand_dims(layers[name][name]["kernel:0"][...], 0)
    dense_b = layers[name][name]["bias:0"][...]
    dense = [dense_w, dense_b]
    assert (
        dense[0].shape == mod.weights[0].shape
    ), f"shape {dense[0].shape} != {mod.weights[0].shape}"
    assert (
        dense[1].shape == mod.weights[1].shape
    ), f"shape {dense[1].shape} != {mod.weights[1].shape}"
    mod.set_weights(dense)


# copy batch normalization layers weights
def copy_bn(mod, layers, name):
    """Copy batch normalisation layers."""
    bn_w = layers[name][name]["gamma:0"][...]  # Scale = gamma
    bn_b = layers[name][name]["beta:0"][...]  # Offset = Beta
    bn_mov_mean = layers[name][name]["moving_mean:0"][...]
    bn_mov_var = layers[name][name]["moving_variance:0"][...]
    bn = [bn_w, bn_b, bn_mov_mean.flatten(), bn_mov_var.flatten()]
    assert (
        bn[0].shape == mod.weights[0].shape
    ), f"shape {bn[0].shape} != {mod.weights[0].shape}"
    assert (
        bn[1].shape == mod.weights[1].shape
    ), f"shape {bn[1].shape} != {mod.weights[1].shape}"
    assert (
        bn[2].shape == mod.weights[2].shape
    ), f"shape {bn[2].shape} != {mod.weights[2].shape}"
    assert (
        bn[3].shape == mod.weights[3].shape
    ), f"shape {bn[3].shape} != {mod.weights[3].shape}"
    mod.set_weights(bn)


# copy layer normalization layers weights
def copy_ln(mod, layers, name):
    """Copy layer normalisation layers."""
    ln_w = layers[name][name]["gamma:0"][...]  # Scale = gamma
    ln_b = layers[name][name]["beta:0"][...]  # Offset = center = beta
    ln = [ln_w, ln_b]
    assert (
        ln[0].shape == mod.weights[0].shape
    ), f"shape {ln[0].shape} != {mod.weights[0].shape}"
    assert (
        ln[1].shape == mod.weights[1].shape
    ), f"shape {ln[0].shape} != {mod.weights[0].shape}"
    mod.set_weights(ln)


# copy multi-head attention layers weights
def copy_mhsa(mod, layers, name):
    """Copy multihead-self-attention layers."""
    Q_w = layers[name][name]["q_layer"]["kernel:0"][...]
    K_w = layers[name][name]["k_layer"]["kernel:0"][...]
    V_w = layers[name][name]["v_layer"]["kernel:0"][...]
    out_w = layers[name][name]["embedding_layer"]["kernel:0"][...]
    out_b = layers[name][name]["embedding_layer"]["bias:0"][...]
    rel_K_w = layers[name][name]["r_k_layer"]["kernel:0"][...]
    r_w_b = layers[name]["r_w_bias:0"][...]
    r_r_b = layers[name]["r_r_bias:0"][...]

    # mhsa = [Q_w, K_w, V_w, out_w, out_b, rel_K_w, r_w_b, r_r_b]
    # Positions from MultiHeadAttention.layers: [r_w_bias, r_r_bias, q_layer, k_layer, v_layer, embedding_layer/kernel, embedding_layer/bias, r_k_layer]
    mhsa = [r_w_b, r_r_b, Q_w, K_w, V_w, out_w, out_b, rel_K_w]
    for i in range(len(mhsa)):
        assert (
            mhsa[i].shape == mod.weights[i].shape
        ), f"shape {mhsa[i].shape} != {mod.weights[i].shape} (index {i})"

    # Specifically check two weird separate weights
    assert mod.weights[
        0
    ].name.endswith(
        "r_w_bias"
    ), f"You might be putting the MHSA weights in the wrong order. This weight should be put at r_w_bias, but it's called {mod.weights[6].name}"
    assert mod.weights[
        1
    ].name.endswith(
        "r_r_bias"
    ), f"You might be putting the MHSA weights in the wrong order. This weight should be put at r_r_bias, but it's called {mod.weights[7].name}"

    mod.set_weights(mhsa)


# Transfer weights to CREsted Borzoi models
for i, (human_tf_path, mouse_tf_path) in enumerate(weights_paths):
    model = crested.tl.zoo.borzoi(seq_len=524288, num_classes=[7611, 2608])
    # Convert weights
    with h5py.File(human_tf_path) as human_weights, h5py.File(
        mouse_tf_path
    ) as mouse_weights:
        for baskerville_name, crested_name in conv_lookup.items():
            copy_convdense(
                model.get_layer(crested_name),
                human_weights["model_weights"],
                baskerville_name,
            )
        for baskerville_name, crested_name in bnorm_lookup.items():
            copy_bn(
                model.get_layer(crested_name),
                human_weights["model_weights"],
                baskerville_name,
            )
        for baskerville_name, crested_name in lnorm_lookup.items():
            copy_ln(
                model.get_layer(crested_name),
                human_weights["model_weights"],
                baskerville_name,
            )
        for baskerville_name, crested_name in mha_lookup.items():
            copy_mhsa(
                model.get_layer(crested_name),
                human_weights["model_weights"],
                baskerville_name,
            )
        for baskerville_name, crested_name in pointwise_lookup.items():
            copy_dense_to_pointwise(
                model.get_layer(crested_name),
                human_weights["model_weights"],
                baskerville_name,
            )
        for baskerville_name, crested_name in mouse_lookup.items():
            copy_dense_to_pointwise(
                model.get_layer(crested_name),
                mouse_weights["model_weights"],
                baskerville_name,
            )

    # Check if all weights were used
    # layers_ported = list(conv_lookup.keys()) + list(bnorm_lookup.keys()) + list(lnorm_lookup.keys()) + list(mha_lookup.keys()) + list(pointwise_lookup.keys()) + list(mouse_lookup.keys())
    layers_ported = {
        l
        for lookup_dict in [
            conv_lookup,
            bnorm_lookup,
            lnorm_lookup,
            mha_lookup,
            pointwise_lookup,
            mouse_lookup,
        ]
        for l in lookup_dict.values()
    }
    model_layers = {l.name for l in model.layers}
    print(f"Model weights not set by porting: {model_layers - layers_ported}")
    if len(layers_ported - model_layers) > 0:
        print(f"Layers to port not found in model: {layers_ported - model_layers}")
    # Save model/weights to file
    model.save(os.path.join(output_dir, f"borzoi_crested_fold{i}.keras"))
    model.save_weights(os.path.join(output_dir, f"borzoi_crested_fold{i}.weights.h5"))
    print(
        f"Keras model saved to disk!\nModel: {os.path.join(output_dir, f'borzoi_crested_fold{i}.keras')}\nWeights: {os.path.join(output_dir, f'borzoi_crested_fold{i}.weights.h5')}"
    )

    model_human = keras.Model(model.inputs, model.outputs[0])
    model_human.save(os.path.join(output_dir, f"borzoi_crested_human_fold{i}.keras"))
    model_mouse = keras.Model(model.inputs, model.outputs[1])
    model_mouse.save(os.path.join(output_dir, f"borzoi_crested_mouse_fold{i}.keras"))
