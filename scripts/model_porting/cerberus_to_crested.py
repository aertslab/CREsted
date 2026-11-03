"""Porting weights from the official baskerville-based (PyTorch) Cerberus model to the CREsted-based model."""
# Port weights from the official baskerville-based Cerberus model to the CREsted-based model.
# Expects environment with CREsted and torch (only to read the checkpoints; any Keras backend works).

# Load modules and paths
import json
import os
import sys

import keras
import numpy as np
import torch

import crested

print(f"CREsted version {crested.__version__}")

# Dir to save your new models
output_dir = "xxx"
# Cerberus saved models dir: should have params.json and folders f0c0/.../f7c0 with model_best.pth
saved_models_dir = "xxx"  # From https://storage.googleapis.com/seqnn-share/cerberus/models: params.json and f{0-7}c0/model_best.pth
folds = [int(f) for f in sys.argv[1:]] or range(8)  # optionally pass folds as arguments

params = json.load(open(os.path.join(saved_models_dir, "params.json")))["model"]
conv_params, hydra_params = params["trunk"][1], params["trunk"][2]
num_conv_blocks, num_hydra_blocks = conv_params["repeat"], hydra_params["repeat"]
head_names = sorted(k for k in params if k.startswith("head"))  # head_human, head_mouse


# Set up mappings of old weights (in the state dict) to new (in the CREsted model, by variable path)
# Each value: (CREsted weight path, transform of the torch tensor)
def conv_kernel(w):
    """Conv1d weight (out, in, k) -> Conv1D kernel (k, in, out)."""
    return w.transpose(2, 1, 0)


def depthwise_kernel(w):
    """Depthwise Conv1d weight (channels, 1, k) -> DepthwiseConv1D kernel (k, channels, 1)."""
    return w.transpose(2, 0, 1)


def dense_kernel(w):
    """Linear weight (out, in) -> Dense kernel (in, out)."""
    return w.T


def identity(w):
    """Copy as is."""
    return w


def bn_lookup(torch_prefix, crested_name):
    """BatchNorm1d -> BatchNormalization."""
    return {
        f"{torch_prefix}.weight": (f"{crested_name}/gamma", identity),
        f"{torch_prefix}.bias": (f"{crested_name}/beta", identity),
        f"{torch_prefix}.running_mean": (f"{crested_name}/moving_mean", identity),
        f"{torch_prefix}.running_var": (f"{crested_name}/moving_variance", identity),
    }


stem_lookup = {
    "trunk.0.conv.weight": ("stem_conv/kernel", conv_kernel),
    "trunk.0.conv.bias": ("stem_conv/bias", identity),
}

# ConvTower layers alternate ConvBlock (even indices) and MaxPool (odd indices)
tower_lookup = {}
for i in range(num_conv_blocks):
    tower_lookup.update(bn_lookup(f"trunk.1.layers.{2 * i}.norm.layer", f"tower_conv_{i + 1}_batchnorm"))
    tower_lookup[f"trunk.1.layers.{2 * i}.conv.weight"] = (f"tower_conv_{i + 1}_conv/kernel", conv_kernel)

hydra_lookup = {}
for i in range(num_hydra_blocks):
    torch_prefix, crested_name = f"trunk.2.blocks.{i}", f"hydra_{i + 1}_hydra"
    hydra_lookup.update({
        f"{torch_prefix}.norm.layer.weight": (f"hydra_{i + 1}_rmsnorm/scale", identity),
        f"{torch_prefix}.hydra.in_proj.weight": (f"{crested_name}/in_proj/kernel", dense_kernel),
        f"{torch_prefix}.hydra.conv1d.weight": (f"{crested_name}/conv1d/kernel", depthwise_kernel),
        f"{torch_prefix}.hydra.conv1d.bias": (f"{crested_name}/conv1d/bias", identity),
        f"{torch_prefix}.hydra.dt_bias": (f"{crested_name}/dt_bias", identity),
        f"{torch_prefix}.hydra.A_log": (f"{crested_name}/A_log", identity),
        f"{torch_prefix}.hydra.D": (f"{crested_name}/D", identity),
        f"{torch_prefix}.hydra.fc_D.weight": (f"{crested_name}/fc_D/kernel", dense_kernel),
        f"{torch_prefix}.hydra.norm.weight": (f"{crested_name}/norm_weight", identity),
        f"{torch_prefix}.hydra.out_proj.weight": (f"{crested_name}/out_proj/kernel", dense_kernel),
    })

head_lookup = {}
for i in range(len(head_names)):
    head_lookup.update(bn_lookup(f"heads_cov.{i}.0.norm.layer", f"head_{i}_batchnorm"))
    head_lookup[f"heads_cov.{i}.0.conv.weight"] = (f"head_{i}/kernel", conv_kernel)
    head_lookup[f"heads_cov.{i}.0.conv.bias"] = (f"head_{i}/bias", identity)

lookup = {**stem_lookup, **tower_lookup, **hydra_lookup, **head_lookup}


# Build model porting function
def load_state_dict(path):
    """Load a baskerville checkpoint as numpy arrays, normalising torch.compile and older head prefixes."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    sd = {(k.replace("heads.", "heads_cov.", 1) if k.startswith("heads.") else k): v for k, v in sd.items()}
    return {k: v.float().numpy() for k, v in sd.items() if not k.endswith("num_batches_tracked")}


def copy_weights(model, state_dict):
    """Copy all state dict tensors into the model, checking that every tensor and every model weight is used once."""
    model_weights = {w.path: w for w in model.weights}
    unused = set(state_dict) - set(lookup)
    assert not unused, f"Checkpoint tensors without a CREsted counterpart: {sorted(unused)}"
    for torch_name, (crested_name, transform) in lookup.items():
        value = transform(state_dict[torch_name])
        weight = model_weights.pop(crested_name)
        assert value.shape == tuple(weight.shape), f"shape {value.shape} != {weight.shape} ({torch_name} -> {crested_name})"
        weight.assign(value)
    assert not model_weights, f"Model weights not set by porting: {sorted(model_weights)}"


# Transfer weights to CREsted Cerberus models
os.makedirs(output_dir, exist_ok=True)
for fold in folds:
    model = crested.tl.zoo.cerberus(
        seq_len=params["seq_length"],
        num_classes=[params[h]["num_targets"] for h in head_names],
        num_conv_blocks=num_conv_blocks,
        num_hydra_blocks=num_hydra_blocks,
        start_filters=params["trunk"][0]["out_channels"],
        filters=conv_params["out_channels"],
        filters_divisible_by=conv_params["divisible_by"],
        first_kernel_size=params["trunk"][0]["kernel_size"],
        kernel_size=conv_params["kernel_size"],
        conv_activation="swish",  # params["act_func"] == "silu"
        d_state=hydra_params["d_state"],
        expand=hydra_params["expand"],
        headdim=hydra_params["headdim"],
        hydra_dropout=hydra_params["dropout"],
    )
    # Convert weights
    copy_weights(model, load_state_dict(os.path.join(saved_models_dir, f"f{fold}c0/model_best.pth")))

    # Save model/weights to file
    model.save(os.path.join(output_dir, f"cerberus_crested_fold{fold}.keras"))
    model.save_weights(os.path.join(output_dir, f"cerberus_crested_fold{fold}.weights.h5"))
    print(f"Keras model saved to disk!\nModel: {os.path.join(output_dir, f'cerberus_crested_fold{fold}.keras')}\nWeights: {os.path.join(output_dir, f'cerberus_crested_fold{fold}.weights.h5')}")

    for i, head_name in enumerate(head_names):
        species = head_name.removeprefix("head_")
        model_species = keras.Model(model.inputs, model.outputs[i])
        model_species.save(os.path.join(output_dir, f"cerberus_crested_{species}_fold{fold}.keras"))
