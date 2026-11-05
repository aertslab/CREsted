"""AlphaGenome model architecture.

AlphaGenome architecture: copyright 2025 Google DeepMind (Apache 2.0, https://github.com/google-deepmind/alphagenome_research)
"""

from __future__ import annotations

import collections.abc as cabc

import keras

from crested.tl.zoo.utils import (
    Bias,
    Scale,
    activate,
    attention_bias_block_ag,
    conv_block_ag,
    down_res_block_ag,
    mha_block_ag,
    mlp_block_ag,
    normalize,
    pair_update_block_ag,
    pool,
    rms_norm_ag,
    up_res_block_ag,
)


def alphagenome(
    seq_len: int,
    num_classes: int | cabc.Sequence[int] | cabc.Mapping[str, int],
    resolutions: cabc.Sequence[int] | cabc.Sequence[cabc.Sequence[int]] = (1, 128),
    num_contact_maps: int = 0,
    splice_sites: bool = False,
    num_splice_site_usage: int = 0,
    start_filters: int = 768,
    filters_step: int = 128,
    num_down_blocks: int = 6,
    first_kernel_size: int = 15,
    kernel_size: int = 5,
    num_transformer_blocks: int = 9,
    pair_update_every: int = 2,
    pair_pool_size: int = 16,
    num_heads: int = 8,
    key_dim: int = 128,
    value_dim: int = 192,
    max_relative_distance: int = 8192,
    mlp_expansion: int = 2,
    pair_dim: int = 128,
    num_pair_heads: int = 32,
    activation: str = "gelu_enf",
    normalization: str = "rms_batch",
    output_activation: str = "softplus",
    name: str = "AlphaGenome",
) -> keras.Model:
    """
    Construct an AlphaGenome model: a U-Net of convolutions around a transformer with a pair representation.

    The convolutional encoder downsamples the sequence to 128 bp resolution (the trunk resolution, by default), where a
    transformer tower runs that also builds a pairwise representation at 2048 bp resolution, which biases its attention.
    Track heads predict at the trunk resolution from the transformer output and/or at 1 bp resolution after a
    convolutional decoder with skip connections from the encoder. Contact map heads predict from the pair representation,
    and optional splice site heads at 1 bp resolution.
    Defaults match the released AlphaGenome models, for a single organism, without the contact map and splicing heads
    (and without the splice junction head, which is not implemented).

    Note that unlike other CREsted model zoo architectures, this architecture is not suited for
    predicting individual regions out of the box.

    Parameters
    ----------
    seq_len
        Width of the input region. Must be a multiple of the pair resolution,
        `2 ** (num_down_blocks + 1) * pair_pool_size` (2048 by default).
        AlphaGenome default is 1048576.
    num_classes
        Number of classes (tracks) to predict.
        If an int, creates a single track head with num_classes classes, named 'head'.
        If a list of integers, creates multiple track heads, named 'head_0', 'head_1', etc.
        If a dict of {name: num_classes}, creates multiple track heads with these names.
    resolutions
        Resolution(s) in bp at which each track head predicts: 1 and/or the trunk resolution
        `2 ** (num_down_blocks + 1)` (128 by default).
        Either one sequence of resolutions for all heads, or one sequence per head (in the order of `num_classes`).
        Each head predicts at each of its resolutions, with separate weights per resolution;
        if a head has several resolutions, its outputs are suffixed with the resolution (e.g. 'head_1bp').
    num_contact_maps
        Number of contact map tracks to predict at the pair resolution from the pair representation (0 for none).
        AlphaGenome default is 28.
    splice_sites
        Whether to add a splice site classification output 'splice_sites' at 1 bp resolution, with softmax probabilities
        of the 5 classes donor (+ strand), acceptor (+), donor (-), acceptor (-) and no splice site.
        AlphaGenome default is True.
    num_splice_site_usage
        Number of splice site usage tracks to predict at 1 bp resolution as output 'splice_site_usage'
        (fraction of transcripts using each splice site, with sigmoid activation; 0 for none).
        AlphaGenome default is 734.
    start_filters
        Number of filters of the first convolution, increasing by `filters_step` in each encoder block.
        The transformer works with `start_filters + num_down_blocks * filters_step` channels (1536 by default).
    filters_step
        Increase in filters per encoder block.
    num_down_blocks
        Number of encoder blocks, each halving the resolution. Together with the pooling before the transformer,
        this sets the trunk resolution to `2 ** (num_down_blocks + 1)` bp. The decoder mirrors the encoder.
    first_kernel_size
        Kernel size of the first convolution, directly interfacing the sequence.
    kernel_size
        Kernel size of the convolutions in the encoder and decoder.
    num_transformer_blocks
        Number of transformer blocks.
    pair_update_every
        Update the pair representation in every n-th transformer block, starting from the first.
    pair_pool_size
        Pooling factor from the trunk resolution to the pair representation resolution.
    num_heads
        Number of attention (query) heads in the transformer blocks.
    key_dim
        Channels per query/key head.
    value_dim
        Channels of the (single, shared) value head.
    max_relative_distance
        Maximum relative distance (in trunk positions) of the position encodings:
        the rotary embeddings in the attention and the relative position features of the pair representation.
    mlp_expansion
        Expansion factor of the hidden layer of the transformer feed-forward blocks.
    pair_dim
        Channels of the pair representation.
    num_pair_heads
        Number of query-key heads to create the pair representation from the sequence.
    activation
        Activation function in the encoder, decoder and output embeddings.
    normalization
        Type of normalization in the encoder, decoder, transformer and output embeddings.
        See :func:`crested.tl.zoo.utils.normalize`.
    output_activation
        Activation of the track heads, before the learned per-track scale.
    name
        Name of the model.

    Returns
    -------
    A Keras model.
    """
    trunk_resolution = 2 ** (num_down_blocks + 1)
    if seq_len % (trunk_resolution * pair_pool_size) != 0:
        raise ValueError(f"seq_len must be a multiple of {trunk_resolution * pair_pool_size}, got {seq_len}.")
    if isinstance(num_classes, int):
        head_names, head_classes = ["head"], [num_classes]
    elif isinstance(num_classes, cabc.Mapping):
        head_names, head_classes = list(num_classes.keys()), list(num_classes.values())
    elif isinstance(num_classes, cabc.Sequence):
        head_names, head_classes = [f"head_{i}" for i in range(len(num_classes))], list(num_classes)
    else:
        raise ValueError(
            f"Could not recognise num_classes argument ({num_classes}) as integer or list/tuple/sequence of integers."
        )
    if all(isinstance(res, int) for res in resolutions):
        resolutions = [resolutions] * len(head_classes)
    if len(resolutions) != len(head_classes):
        raise ValueError(f"Got {len(resolutions)} sequences of resolutions for {len(head_classes)} heads.")
    if not all(res in (1, trunk_resolution) for head_res in resolutions for res in head_res):
        raise ValueError(f"Resolutions must be 1 or {trunk_resolution}, got {resolutions}.")
    need_1bp = any(1 in head_res for head_res in resolutions) or splice_sites or num_splice_site_usage > 0
    block_args = {"activation": activation, "normalization": normalization}

    sequence = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Encoder: conv stem + residual blocks, each preceded by pooling; keep the outputs as skip connections
    current = keras.layers.Conv1D(start_filters, first_kernel_size, padding="same", name="stem_conv")(sequence)
    stem_block = conv_block_ag(current, start_filters, kernel_size, name_prefix="stem_block", **block_args)
    current = keras.layers.Add(name="stem_add")([current, stem_block])
    skips = [current]
    filters = start_filters
    for i in range(num_down_blocks):
        current = pool(current, pool_type="max", pool_size=2, padding="same", name=f"encoder_{i + 1}_pool")
        filters += filters_step
        current = down_res_block_ag(current, filters, kernel_size, name_prefix=f"encoder_{i + 1}", **block_args)
        skips.append(current)
    current = pool(current, pool_type="max", pool_size=2, padding="same", name="encoder_pool")
    current = Bias(name="organism_bias")(current)  # AlphaGenome's organism embedding, for one organism

    # Transformer tower, with the pair representation updated every `pair_update_every` blocks
    pair = None
    for i in range(num_transformer_blocks):
        prefix = f"transformer_{i + 1}"
        if i % pair_update_every == 0:
            pair = pair_update_block_ag(
                current,
                pair,
                pair_dim=pair_dim,
                pair_pool_size=pair_pool_size,
                num_heads=num_pair_heads,
                max_distance=max_relative_distance // pair_pool_size,
                name_prefix=f"{prefix}_pair",
            )
        attention_bias = attention_bias_block_ag(
            pair, num_heads=num_heads, normalization=normalization, name_prefix=f"{prefix}_attention_bias"
        )
        current = mha_block_ag(
            current,
            attention_bias,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            max_position=max_relative_distance,
            bias_upsampling=pair_pool_size,
            normalization=normalization,
            name_prefix=f"{prefix}_mha",
        )
        current = mlp_block_ag(
            current, expansion=mlp_expansion, normalization=normalization, name_prefix=f"{prefix}_mlp"
        )

    # Output embeddings at the trunk resolution, and at 1 bp after the decoder
    trunk_dim = current.shape[-1]
    embeddings = {
        trunk_resolution: _output_embedding(current, 2 * trunk_dim, block_args, f"embedding_{trunk_resolution}bp")
    }
    if need_1bp:
        for i, skip in enumerate(reversed(skips)):
            current = up_res_block_ag(current, skip, kernel_size, name_prefix=f"decoder_{i + 1}", **block_args)
        current = keras.layers.Dense(trunk_dim, name="embedding_1bp_dense")(current)
        skip = keras.layers.Dense(trunk_dim, use_bias=False, name="embedding_1bp_skip_dense")(
            embeddings[trunk_resolution]
        )
        skip = keras.layers.UpSampling1D(trunk_resolution, name="embedding_1bp_skip_upsampling")(skip)
        current = keras.layers.Add(name="embedding_1bp_add")([current, skip])
        embeddings[1] = _output_embedding(current, None, block_args, "embedding_1bp")

    # Track heads: dense + activation + learned positive scale per track
    outputs = []
    for head_name, n_tracks, head_res in zip(head_names, head_classes, resolutions, strict=True):
        for res in head_res:
            output_name = f"{head_name}_{res}bp" if len(head_res) > 1 else head_name
            current = keras.layers.Dense(n_tracks, activation=output_activation, name=f"{output_name}_dense")(
                embeddings[res]
            )
            outputs.append(Scale(per_channel=True, activation="softplus", name=output_name)(current))

    # Splice site heads at 1 bp
    if splice_sites:
        outputs.append(keras.layers.Dense(5, activation="softmax", name="splice_sites")(embeddings[1]))
    if num_splice_site_usage > 0:
        outputs.append(
            keras.layers.Dense(num_splice_site_usage, activation="sigmoid", name="splice_site_usage")(embeddings[1])
        )

    # Contact map head from the symmetrized pair representation
    if num_contact_maps > 0:
        current = keras.layers.Average(name="embedding_pair_symmetrize")(
            [pair, keras.layers.Permute((2, 1, 3), name="embedding_pair_transpose")(pair)]
        )
        current = rms_norm_ag(current, "embedding_pair")
        current = activate(current, activation, name="embedding_pair_activation")
        outputs.append(keras.layers.Dense(num_contact_maps, name="contact_maps")(current))

    return keras.Model(inputs=sequence, outputs=outputs[0] if len(outputs) == 1 else outputs, name=name)


def _output_embedding(inputs, filters, block_args, name_prefix):
    """Output embedding: (optional dense projection) + normalization + activation."""
    current = inputs
    if filters is not None:
        current = keras.layers.Dense(filters, name=f"{name_prefix}_dense")(current)
    normalization = block_args["normalization"]
    current = normalize(current, normalization, name=f"{name_prefix}_{normalization}norm")
    return activate(current, block_args["activation"], name=f"{name_prefix}_activation")
