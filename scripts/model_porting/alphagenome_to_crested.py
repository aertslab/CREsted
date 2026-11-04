"""Porting weights from the alphagenome-pytorch AlphaGenome models to single-organism CREsted-based models."""
# Port the AlphaGenome weights converted by alphagenome-pytorch (https://huggingface.co/gtca/alphagenome_pytorch,
# model_all_folds.safetensors and model_fold_{0-3}.safetensors) to CREsted, as one model per organism, in two variants:
# default (track heads only) and '_full' (track heads + splice site classification + splice site usage + contact maps).
# Organism embeddings are folded into the following biases, heads are sliced to the organism's real tracks (padding
# tracks and heads without tracks are left out), and the splice junction head is not ported.
# Track outputs are in model space (before AlphaGenome's track-mean unscaling).
# Expects environment with CREsted, torch and safetensors (only to read the weights; any Keras backend works),
# and pandas + pyarrow to read the track metadata (alphagenome-pytorch's track_metadata_{organism}.parquet).

# Load modules and paths
import os
import sys

import numpy as np
import pandas as pd
from safetensors.torch import load_file

import crested

print(f"CREsted version {crested.__version__}")

# Dir to save your new models
output_dir = "xxx"
# Dir with the safetensors files from https://huggingface.co/gtca/alphagenome_pytorch
weights_dir = "xxx"
# Dir with track_metadata_{human,mouse}.parquet, from alphagenome-pytorch's src/alphagenome_pytorch/data
metadata_dir = "xxx"
names = sys.argv[1:] or ["all_folds", "fold_0", "fold_1", "fold_2", "fold_3"]  # optionally pass weight names
seq_len = 1048576
organisms = {0: "human", 1: "mouse"}
splice_site_classes = ["donor +", "acceptor +", "donor -", "acceptor -", "none"]

# Track heads of the released model: name -> (num tracks, resolutions)
HEADS = {
    "atac": (256, (1, 128)),
    "dnase": (384, (1, 128)),
    "procap": (128, (1, 128)),
    "cage": (640, (1, 128)),
    "rna_seq": (768, (1, 128)),
    "chip_tf": (1664, (128,)),
    "chip_histone": (1152, (128,)),
}
# Other heads: name -> num tracks (splice site classes always all 5, for the softmax)
OTHER_HEADS = {"splice_sites": 5, "splice_site_usage": 734, "contact_maps": 28}
NUM_TRANSFORMER_BLOCKS = 9
SKIPPED_PREFIXES = ("splice_sites_junction_head.",)
TORCH_PREFIXES = {
    "splice_sites": "splice_sites_classification_head.",
    "splice_site_usage": "splice_sites_usage_head.",
    "contact_maps": "contact_maps_head.",
}


def full_tracks() -> dict:
    """Track indices per head including padding, for all ported heads: head name -> indices."""
    return {name: np.arange(n) for name, n in {**{k: n for k, (n, _) in HEADS.items()}, **OTHER_HEADS}.items()}


def build_model(seq_len: int, tracks: dict | None = None):
    """CREsted AlphaGenome with the released heads in tracks (head -> track indices; default all), leaving out empty heads.

    Output order: track heads in the order of HEADS, splice_sites, splice_site_usage, contact_maps.
    """
    num_tracks = {name: len(idx) for name, idx in (full_tracks() if tracks is None else tracks).items()}
    assert num_tracks.get("splice_sites", 5) in (0, 5), "splice_sites needs all 5 classes"
    track_heads = {name: num_tracks[name] for name in HEADS if num_tracks.get(name, 0) > 0}
    return crested.tl.zoo.alphagenome(
        seq_len=seq_len,
        num_classes=track_heads,
        resolutions=[HEADS[name][1] for name in track_heads],
        num_contact_maps=num_tracks.get("contact_maps", 0),
        splice_sites=num_tracks.get("splice_sites", 0) > 0,
        num_splice_site_usage=num_tracks.get("splice_site_usage", 0),
    )


def conv_kernel(w):
    """Conv1d weight (out, in, k) -> Conv1D kernel (k, in, out)."""
    return w.transpose(2, 1, 0)


def pointwise_kernel(w):
    """Conv1d weight (out, in, 1) -> Dense kernel (in, out)."""
    return w[:, :, 0].T


def dense_kernel(w):
    """Linear weight (out, in) -> Dense kernel (in, out)."""
    return w.T


def identity(w):
    """Copy as is."""
    return w


def build_lookup(state_dict: dict, organism: int, tracks: dict) -> dict:
    """Torch key -> list of (CREsted weight path, transform of the numpy tensor). Several torch keys may sum into one path."""
    org = organism
    lookup = {}

    def add(key, path, fn=identity):
        lookup.setdefault(key, []).append((path, fn))

    def rms_batch(prefix, name):
        add(f"{prefix}.weight", f"{name}/gamma")
        add(f"{prefix}.bias", f"{name}/beta")
        add(f"{prefix}.running_var", f"{name}/moving_variance")

    def rms_norm(prefix, name):  # LayerNorm(rms_norm=True) -> RMSNormalization + Bias
        add(f"{prefix}.weight", f"{name}_rmsnorm/scale")
        add(f"{prefix}.bias", f"{name}_rmsnorm_bias/bias")

    def conv_block(prefix, name, kernel_size=5):
        rms_batch(f"{prefix}.norm", f"{name}_rms_batchnorm")
        if kernel_size == 1:
            add(f"{prefix}.conv.weight", f"{name}_dense/kernel", pointwise_kernel)
            add(f"{prefix}.conv.bias", f"{name}_dense/bias")
        else:
            add(f"{prefix}.conv.weight", f"{name}_conv/kernel", conv_kernel)
            add(f"{prefix}.conv.bias", f"{name}_conv/bias")
            add(f"{prefix}.conv.scale", f"{name}_conv/scale", lambda w: w[:, 0, 0])

    def dense(prefix, name, bias=True):
        add(f"{prefix}.weight", f"{name}/kernel", dense_kernel)
        if bias:
            add(f"{prefix}.bias", f"{name}/bias")

    # Encoder
    add("encoder.dna_embedder.conv1.weight", "stem_conv/kernel", conv_kernel)
    add("encoder.dna_embedder.conv1.bias", "stem_conv/bias")
    conv_block("encoder.dna_embedder.block", "stem_block")
    for i in range(6):
        for b in (1, 2):
            conv_block(f"encoder.down_blocks.{i}.block{b}", f"encoder_{i + 1}_block{b}")
    add("organism_embed.weight", "organism_bias/bias", lambda w: w[org])

    # Transformer tower
    for i in range(NUM_TRANSFORMER_BLOCKS):
        tp, cp = f"tower.blocks.{i}", f"transformer_{i + 1}"
        if i % 2 == 0:
            sp, cs = f"{tp}.pair_update.seq2pair", f"{cp}_pair_seq2pair"
            rms_norm(f"{sp}.norm_seq2pair", cs)
            add(f"{sp}.q_r_bias", f"{cs}/q_r_bias", lambda w: w[0, 0])
            add(f"{sp}.k_r_bias", f"{cs}/k_r_bias", lambda w: w[0, 0])
            dense(f"{sp}.linear_q", f"{cs}/q_proj", bias=False)
            dense(f"{sp}.linear_k", f"{cs}/k_proj", bias=False)
            dense(f"{sp}.linear_pos_features", f"{cs}/pos_proj")
            dense(f"{sp}.linear_y_q", f"{cs}/y_q_proj", bias=False)
            dense(f"{sp}.linear_y_k", f"{cs}/y_k_proj", bias=False)
            dense(f"{sp}.linear_pair", f"{cs}/pair_proj")
            rp, cr = f"{tp}.pair_update.row_attn", f"{cp}_pair_row_attention"
            rms_norm(f"{rp}.norm", cr)
            dense(f"{rp}.linear_q", f"{cr}/q_proj", bias=False)
            dense(f"{rp}.linear_k", f"{cr}/k_proj", bias=False)
            dense(f"{rp}.linear_v", f"{cr}/v_proj")
            mp = f"{tp}.pair_update.pair_mlp"
            rms_norm(f"{mp}.norm", f"{cp}_pair_pair_mlp")
            dense(f"{mp}.linear1", f"{cp}_pair_pair_mlp_dense1")
            dense(f"{mp}.linear2", f"{cp}_pair_pair_mlp_dense2")
        rms_batch(f"{tp}.attn_bias.norm", f"{cp}_attention_bias_rms_batchnorm")
        dense(f"{tp}.attn_bias.proj", f"{cp}_attention_bias_dense", bias=False)
        hp, ch = f"{tp}.mha", f"{cp}_mha"
        rms_batch(f"{hp}.norm", f"{ch}_rms_batchnorm")
        for proj in "qkv":
            dense(f"{hp}.{proj}_proj", f"{ch}_attention/{proj}_proj", bias=False)
            add(f"{hp}.norm_{proj}.weight", f"{ch}_attention/{proj}_norm/gamma")
            add(f"{hp}.norm_{proj}.bias", f"{ch}_attention/{proj}_norm/beta")
        dense(f"{hp}.linear_embedding", f"{ch}_attention/out_proj")
        rms_batch(f"{hp}.final_norm", f"{ch}_final_rms_batchnorm")
        lp, cl = f"{tp}.mlp", f"{cp}_mlp"
        rms_batch(f"{lp}.norm", f"{cl}_rms_batchnorm")
        dense(f"{lp}.fc1", f"{cl}_dense1")
        dense(f"{lp}.fc2", f"{cl}_dense2")
        rms_batch(f"{lp}.final_norm", f"{cl}_final_rms_batchnorm")

    # Decoder
    for i in range(7):
        up, cu = f"decoder.up_blocks.{i}", f"decoder_{i + 1}"
        conv_block(f"{up}.conv_in", f"{cu}_conv_in")
        add(f"{up}.residual_scale", f"{cu}_residual_scale/scale")
        conv_block(f"{up}.pointwise", f"{cu}_pointwise", kernel_size=1)
        conv_block(f"{up}.conv_out", f"{cu}_conv_out")

    # Output embeddings: organism embeddings are added after the norm, so they fold into its offset
    for res in ("128bp", "1bp"):
        ep, ce = f"embedder_{res}", f"embedding_{res}"
        add(f"{ep}.project_in.weight", f"{ce}_dense/kernel", pointwise_kernel)
        add(f"{ep}.project_in.bias", f"{ce}_dense/bias")
        rms_batch(f"{ep}.norm", f"{ce}_rms_batchnorm")
        add(f"{ep}.organism_embed.weight", f"{ce}_rms_batchnorm/beta", lambda w: w[org])
    add("embedder_1bp.project_skip.weight", "embedding_1bp_skip_dense/kernel", pointwise_kernel)
    if has(tracks, "contact_maps"):
        rms_norm("embedder_pair.norm", "embedding_pair")
        add("embedder_pair.organism_embed.weight", "embedding_pair_rmsnorm_bias/bias", lambda w: w[org])

    # Heads (MultiOrganismConv1d weight: (organism, out, in)), sliced to the given tracks
    for name, (_, resolutions) in HEADS.items():
        if not has(tracks, name):
            continue
        idx = np.asarray(tracks[name])
        for res in resolutions:
            out = f"{name}_{res}bp" if len(resolutions) > 1 else name
            add(f"heads.{name}.convs.{res}.weight", f"{out}_dense/kernel", lambda w, i=idx: w[org, i].T)
            add(f"heads.{name}.convs.{res}.bias", f"{out}_dense/bias", lambda w, i=idx: w[org, i])
            add(f"heads.{name}.residual_scales.{res}", f"{out}/scale", lambda w, i=idx: w[org, i])
    for name, prefix in (
        ("splice_sites", "splice_sites_classification_head"),
        ("splice_site_usage", "splice_sites_usage_head"),
    ):
        if has(tracks, name):
            idx = np.asarray(tracks[name])
            add(f"{prefix}.conv.weight", f"{name}/kernel", lambda w, i=idx: w[org, i].T)
            add(f"{prefix}.conv.bias", f"{name}/bias", lambda w, i=idx: w[org, i])
    if has(tracks, "contact_maps"):
        idx = np.asarray(tracks["contact_maps"])  # (organism, in, out)
        add("contact_maps_head.linear.weight", "contact_maps/kernel", lambda w: w[org][:, idx])
        add("contact_maps_head.linear.bias", "contact_maps/bias", lambda w: w[org, idx])
    return lookup


def has(tracks: dict, name: str) -> bool:
    """Whether the head is in the model."""
    return len(tracks.get(name, ())) > 0


def port_weights(state_dict: dict, model, organism: int, tracks: dict | None = None) -> None:
    """Assign the state dict (torch tensors or numpy arrays) to the CREsted model for one organism, checking full coverage.

    `tracks` (head name -> track indices, default all incl. padding; as given to build_model) must match the model.
    """
    tracks = full_tracks() if tracks is None else tracks
    state_dict = {
        k: np.asarray(v.detach().float().cpu().numpy() if hasattr(v, "detach") else v) for k, v in state_dict.items()
    }
    lookup = build_lookup(state_dict, organism, tracks)
    skipped = SKIPPED_PREFIXES + tuple(f"heads.{name}." for name in HEADS if not has(tracks, name))
    skipped += tuple(prefix for name, prefix in TORCH_PREFIXES.items() if not has(tracks, name))
    if not has(tracks, "contact_maps"):
        skipped += ("embedder_pair.",)
    ported = [k for k in state_dict if not k.startswith(skipped) and not k.endswith(".track_means")]
    missing = sorted(set(ported) - set(lookup))
    unknown = sorted(set(lookup) - set(state_dict))
    assert not missing and not unknown, f"unmapped torch keys: {missing[:10]}, unknown keys: {unknown[:10]}"

    values = {}
    for key in ported:
        for path, fn in lookup[key]:
            value = fn(state_dict[key]).astype("float32")
            values[path] = values[path] + value if path in values else value
    weights = {w.path: w for w in model.weights}
    assert set(values) == set(weights), (
        f"CREsted weights without value: {sorted(set(weights) - set(values))[:10]}, "
        f"values without weight: {sorted(set(values) - set(weights))[:10]}"
    )
    for path, value in values.items():
        assert tuple(weights[path].shape) == value.shape, f"{path}: {weights[path].shape} vs {value.shape}"
        weights[path].assign(value)


# Real tracks per head (track indices and output class names) and the two model variants per organism
variants = {}
for org, organism in organisms.items():
    metadata = pd.read_parquet(os.path.join(metadata_dir, f"track_metadata_{organism}.parquet"))
    metadata = metadata.sort_values("track_index")
    metadata = metadata[metadata["track_name"].str.lower() != "padding"]
    indices, classes = {}, {}
    for head in [*HEADS, "splice_site_usage", "contact_maps"]:
        rows = metadata[metadata["output_type"] == head]
        indices[head] = rows["track_index"].to_numpy()
        classes[head] = [
            name.removeprefix("usage_") + (f" {strand}" if strand in ("+", "-") else "")
            for name, strand in zip(rows["track_name"], rows["strand"], strict=True)
        ]
    # Splice site classification: all 5 classes (the 5th, 'no splice site', is labelled padding in the metadata)
    indices["splice_sites"], classes["splice_sites"] = np.arange(5), splice_site_classes
    track_heads = {head: indices[head] for head in HEADS}
    other_heads = {head: indices[head] for head in ("splice_sites", "splice_site_usage", "contact_maps")}
    for suffix, tracks in (("_full", {**track_heads, **other_heads}), ("", track_heads)):
        variants[(org, suffix)] = tracks
        with open(os.path.join(output_dir, f"alphagenome_{organism}{suffix}_output_classes.tsv"), "w") as f:
            for head, idx in tracks.items():
                if len(idx) > 0:
                    f.writelines(f"{head}/{c}\n" for c in classes[head])

for name in names:
    state_dict = load_file(os.path.join(weights_dir, f"model_{name}.safetensors"))
    for (org, suffix), tracks in variants.items():
        model = build_model(seq_len, tracks)
        port_weights(state_dict, model, org, tracks)
        model.save(os.path.join(output_dir, f"alphagenome_{organisms[org]}{suffix}_{name}.keras"))
        print(f"Saved {organisms[org]}{suffix} {name}: {model.count_params():,} parameters")
