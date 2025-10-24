"""Port weights from the official Sonnet Enformer model to the CREsted-based model."""
# Port weights from the official Sonnet Enformer model to the CREsted-based model.
# Requires the Enformer checkpoint - download from https://console.cloud.google.com/storage/browser/dm-enformer/models/enformer/sonnet_weights
# Problem： Don't have environment with both working Sonnet and working CREsted -> Sonnet seems to be broken with modern TF versions?
# Solution: read checkpoint directly without Sonnet.


import os

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.training import py_checkpoint_reader

import crested

print(f"TensorFlow version {tf.__version__}")
print(f"CREsted version {crested.__version__}")

# Set directories and paths
output_dir = os.getcwd()
checkpoint_path = "xxx"  # Path to folder with the three files from https://console.cloud.google.com/storage/browser/dm-enformer/models/enformer/sonnet_weights

# Load models
random_seq = np.eye(4)[np.newaxis, np.random.choice(4, 196608)]

# Load Sonnet enformer checkpoint
# Manually get the latest epoch in the checkpoint dir - can also directly pass checkpoint_path to checkpoint.restore()
latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
print(f"Found checkpoint {latest}")
reader = py_checkpoint_reader.NewCheckpointReader(latest)

# Create Keras enformer object
keras_model = crested.tl.zoo.enformer(seq_len=196608, num_classes=[5313, 1643])
# Initialize Keras model with random weights
# _ = keras_model(tf.constant(random_seq, dtype=tf.float32))


# define functions
# make a dictionary with model variable names and values
def get_vars(reader):
    """Get variables from the Sonnet model."""
    var_to_shape_map = reader.get_variable_to_shape_map()
    model_vars = {}
    for var_name in var_to_shape_map:
        if var_name.startswith("module"):
            model_vars[var_name] = reader.get_tensor(var_name)
    return model_vars


# copy convolutional or dense layers weights
def copy_convdense(mod, model_vars, name):
    """Copy convolutional or dense weights."""
    # Assuming a 1-weight conv/dense is biasless
    conv_w = model_vars.pop(f"{name}w/.ATTRIBUTES/VARIABLE_VALUE")
    conv = [conv_w]
    if len(mod.weights) == 2:
        conv_b = model_vars.pop(f"{name}b/.ATTRIBUTES/VARIABLE_VALUE")
        conv.append(conv_b)
    assert (
        conv[0].shape == mod.weights[0].shape
    ), f"shape {conv[0].shape} != {mod.weights[0].shape}"
    if len(mod.weights) == 2:
        assert (
            conv[1].shape == mod.weights[1].shape
        ), f"shape {conv[1].shape} != {mod.weights[1].shape}"
    mod.set_weights(conv)


# Copy dense layers used as pointwise convs to actual pointwise convs
def copy_dense_to_pointwise(mod, model_vars, name):
    """Copy a dense layer to a pointwise convolutional layer.

    Adds a dimension at the start, for when you want to copy a dense layer used as pointwise conv (shape [input, filters])
    to a true pointwise conv layer (shape [width, input, filters] = [1, input, filters])
    """
    dense_w = np.expand_dims(model_vars.pop(f"{name}w/.ATTRIBUTES/VARIABLE_VALUE"), 0)
    dense_b = model_vars.pop(f"{name}b/.ATTRIBUTES/VARIABLE_VALUE")
    dense = [dense_w, dense_b]
    assert (
        dense[0].shape == mod.weights[0].shape
    ), f"shape {dense[0].shape} != {mod.weights[0].shape}"
    assert (
        dense[1].shape == mod.weights[1].shape
    ), f"shape {dense[1].shape} != {mod.weights[1].shape}"
    mod.set_weights(dense)


# copy batch normalization layers weights
def copy_bn(mod, model_vars, name):
    """Copy batch normalisation weights."""
    bn_gamma = model_vars.pop(
        f"{name}scale/.ATTRIBUTES/VARIABLE_VALUE"
    )  # Scale = gamma
    bn_beta = model_vars.pop(
        f"{name}offset/.ATTRIBUTES/VARIABLE_VALUE"
    )  # Offset = beta
    bn_mov_mean = model_vars.pop(
        f"{name}moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE"
    )
    bn_mov_var = model_vars.pop(
        f"{name}moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE"
    )
    # Delete hidden states and counter, which aren't needed in Keras batchnorm implementation
    del model_vars[f"{name}moving_mean/_hidden/.ATTRIBUTES/VARIABLE_VALUE"]
    del model_vars[f"{name}moving_variance/_hidden/.ATTRIBUTES/VARIABLE_VALUE"]
    del model_vars[f"{name}moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE"]
    del model_vars[f"{name}moving_variance/_counter/.ATTRIBUTES/VARIABLE_VALUE"]
    bn = [bn_gamma, bn_beta, bn_mov_mean.flatten(), bn_mov_var.flatten()]
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


# copy attention pooling layers weights
def copy_attn_pool(mod, model_vars, name):
    """Copy attention pooling weights."""
    attn_pool = [model_vars.pop(f"{name}w/.ATTRIBUTES/VARIABLE_VALUE")]
    assert (
        attn_pool[0].shape == mod.weights[0].shape
    ), f"shape {attn_pool[0].shape} != {mod.weights[0].shape}"
    mod.set_weights(attn_pool)


# copy layer normalization layers weights
def copy_ln(mod, model_vars, name):
    """Copy layer normalisation weights."""
    ln_w = model_vars.pop(f"{name}scale/.ATTRIBUTES/VARIABLE_VALUE")  # Scale = gamma
    ln_b = model_vars.pop(
        f"{name}offset/.ATTRIBUTES/VARIABLE_VALUE"
    )  # Offset = center = beta
    ln = [ln_w, ln_b]
    assert (
        ln[0].shape == mod.weights[0].shape
    ), f"shape {ln[0].shape} != {mod.weights[0].shape}"
    assert (
        ln[1].shape == mod.weights[1].shape
    ), f"shape {ln[0].shape} != {mod.weights[0].shape}"
    mod.set_weights(ln)


# copy multi-head attention layers weights
def copy_mhsa(mod, model_vars, name):
    """Copy multi-head self-attention weights."""
    Q_w = model_vars.pop(f"{name}_q_layer/w/.ATTRIBUTES/VARIABLE_VALUE")
    K_w = model_vars.pop(f"{name}_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE")
    V_w = model_vars.pop(f"{name}_v_layer/w/.ATTRIBUTES/VARIABLE_VALUE")
    out_w = model_vars.pop(f"{name}_embedding_layer/w/.ATTRIBUTES/VARIABLE_VALUE")
    out_b = model_vars.pop(f"{name}_embedding_layer/b/.ATTRIBUTES/VARIABLE_VALUE")
    rel_K_w = model_vars.pop(f"{name}_r_k_layer/w/.ATTRIBUTES/VARIABLE_VALUE")
    r_w_b = model_vars.pop(f"{name}_r_w_bias/.ATTRIBUTES/VARIABLE_VALUE")
    r_r_b = model_vars.pop(f"{name}_r_r_bias/.ATTRIBUTES/VARIABLE_VALUE")

    # mhsa = [Q_w, K_w, V_w, out_w, out_b, rel_K_w, r_w_b, r_r_b]
    # Positions from MultiHeadAttention.layers: [r_w_bias, r_r_bias, q_layer, k_layer, v_layer, embedding_layer/kernel, embedding_layer/bias, r_k_layer]
    mhsa = [r_w_b, r_r_b, Q_w, K_w, V_w, out_w, out_b, rel_K_w]
    for i in range(len(mhsa)):
        assert (
            mhsa[i].shape == mod.weights[i].shape
        ), f"shape {mhsa[i].shape} != {mod.weights[i].shape} (index {i})"

    # Specifically check two weird separate weights
    # assert mod.weights[6].name.endswith('r_w_bias'), f"You might be indexing into the MHSA wrongly. this weight should be put at r_w_bias, but it's called {mod.weights[6].name}"
    # assert mod.weights[7].name.endswith('r_r_bias'), f"You might be indexing into the MHSA wrongly. this weight should be put at r_r_bias, but it's is called {mod.weights[7].name}"
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


# copy weights from sonnet to keras
def copy_snt_to_keras(ckpt_reader, keras_model):
    """Copy the entire sonnet model to the keras model."""
    n_tower_layers = 6
    n_transformer_layers = 11

    var_dict = get_vars(ckpt_reader)

    # Stem
    stem_conv = keras_model.get_layer("stem_conv")
    stem_res_bn = keras_model.get_layer("stem_pointwise_batchnorm")
    stem_res_conv = keras_model.get_layer("stem_pointwise_conv")
    stem_pool = keras_model.get_layer("stem_pointwise_pool")

    copy_convdense(stem_conv, var_dict, "module/_trunk/_layers/0/_layers/0/")
    copy_bn(
        stem_res_bn, var_dict, "module/_trunk/_layers/0/_layers/1/_module/_layers/0/"
    )
    copy_convdense(
        stem_res_conv, var_dict, "module/_trunk/_layers/0/_layers/1/_module/_layers/2/"
    )
    copy_attn_pool(
        stem_pool, var_dict, "module/_trunk/_layers/0/_layers/2/_logit_linear/"
    )

    # Convolution tower
    for i in range(n_tower_layers):
        tower_bn = keras_model.get_layer(f"tower_conv_{i+1}_batchnorm")
        tower_conv = keras_model.get_layer(f"tower_conv_{i+1}_conv")
        tower_res_bn = keras_model.get_layer(f"tower_pointwise_{i+1}_batchnorm")
        tower_res_conv = keras_model.get_layer(f"tower_pointwise_{i+1}_conv")
        tower_pool = keras_model.get_layer(f"tower_pointwise_{i+1}_pool")

        bn_name = f"module/_trunk/_layers/1/_layers/{i}/_layers/0/_layers/0/"
        conv_name = f"module/_trunk/_layers/1/_layers/{i}/_layers/0/_layers/2/"
        res_bn_name = (
            f"module/_trunk/_layers/1/_layers/{i}/_layers/1/_module/_layers/0/"
        )
        res_conv_name = (
            f"module/_trunk/_layers/1/_layers/{i}/_layers/1/_module/_layers/2/"
        )
        pool_name = f"module/_trunk/_layers/1/_layers/{i}/_layers/2/_logit_linear/"

        copy_bn(tower_bn, var_dict, bn_name)
        copy_convdense(tower_conv, var_dict, conv_name)
        copy_bn(tower_res_bn, var_dict, res_bn_name)
        copy_convdense(tower_res_conv, var_dict, res_conv_name)
        copy_attn_pool(tower_pool, var_dict, pool_name)

    # Transformer tower
    for i in range(n_transformer_layers):
        trans_mha_ln = keras_model.get_layer(f"transformer_mha_{i+1}_layernorm")
        trans_mha_mhsa = keras_model.get_layer(f"transformer_mha_{i+1}_mhsa")
        trans_ff_ln = keras_model.get_layer(f"transformer_ff_{i+1}_layernorm")
        trans_ff_conv1 = keras_model.get_layer(f"transformer_ff_{i+1}_pointwise_1")
        trans_ff_conv2 = keras_model.get_layer(f"transformer_ff_{i+1}_pointwise_2")

        ln1_name = f"module/_trunk/_layers/2/_layers/{i}/_layers/0/_module/_layers/0/"
        mhsa_name = f"module/_trunk/_layers/2/_layers/{i}/_layers/0/_module/_layers/1/"
        ln2_name = f"module/_trunk/_layers/2/_layers/{i}/_layers/1/_module/_layers/0/"
        ffn1_name = f"module/_trunk/_layers/2/_layers/{i}/_layers/1/_module/_layers/1/"
        ffn2_name = f"module/_trunk/_layers/2/_layers/{i}/_layers/1/_module/_layers/4/"

        # Copy MHA block
        copy_ln(trans_mha_ln, var_dict, ln1_name)
        copy_mhsa(trans_mha_mhsa, var_dict, mhsa_name)

        # Copy feedforward block
        copy_ln(trans_ff_ln, var_dict, ln2_name)
        copy_dense_to_pointwise(trans_ff_conv1, var_dict, ffn1_name)
        copy_dense_to_pointwise(trans_ff_conv2, var_dict, ffn2_name)

    # Pointwise final module
    final_bn = keras_model.get_layer("final_pointwise_batchnorm")
    final_conv = keras_model.get_layer("final_pointwise_conv")

    copy_bn(final_bn, var_dict, "module/_trunk/_layers/4/_layers/0/_layers/0/")
    copy_convdense(final_conv, var_dict, "module/_trunk/_layers/4/_layers/0/_layers/2/")

    # heads modules
    human_head = keras_model.get_layer("head_0")
    mouse_head = keras_model.get_layer("head_1")

    copy_dense_to_pointwise(human_head, var_dict, "module/_heads/human/_layers/0/")
    copy_dense_to_pointwise(mouse_head, var_dict, "module/_heads/mouse/_layers/0/")

    print("Weights successfully migrated!")
    if len(var_dict) > 0:
        print("Unmigrated sonnet weights:")
        for n, s in var_dict.items():
            print(f"{n} (shape {s})")


# apply the functions
copy_snt_to_keras(reader, keras_model)

# Manually save the entire model
keras_model.save(os.path.join(output_dir, "enformer_crested.keras"))

# Manually save weights
keras_model.save_weights(os.path.join(output_dir, "enformer_crested.weights.h5"))

# Save one-head versions of the model
model_human = keras.Model(keras_model.inputs, keras_model.outputs[0])
model_human.save(os.path.join(output_dir, "enformer_crested_human.keras"))
model_mouse = keras.Model(keras_model.inputs, keras_model.outputs[1])
model_mouse.save(os.path.join(output_dir, "enformer_crested_mouse.keras"))

print(
    f"Keras model saved to disk!\nModel: {os.path.join(output_dir, 'enformer_crested[_human/mouse].keras')}\nWeights: {os.path.join(output_dir, 'enformer_crested.weights.h5')}"
)
