"""Cerberus model architecture.

Cerberus architecture: copyright Calico LLC
"""

from __future__ import annotations

import collections.abc as cabc

import keras

from crested.tl.zoo._borzoi import exp_linspace_int
from crested.tl.zoo.utils import activate, conv_block_bs, hydra_block, normalize, pool


def cerberus(
    seq_len: int,
    num_classes: int | cabc.Sequence[int],
    num_conv_blocks: int = 4,
    num_hydra_blocks: int = 8,
    start_filters: int = 768,
    filters: int = 1536,
    filters_divisible_by: int = 64,
    first_kernel_size: int = 13,
    kernel_size: int = 5,
    conv_activation: str = "swish",
    d_state: int = 1,
    d_conv: int = 7,
    expand: float = 1.6667,
    headdim: int = 128,
    chunk_size: int = 256,
    hydra_dropout: float = 0.08,
    drop_path: float = 0,
    output_activation: str = "softplus",
    pool_type: str = "max",
    name: str = "Cerberus",
) -> keras.Model:
    """
    Construct a Cerberus model: a CNN + bidirectional Mamba-2 (Hydra) model.

    Defaults match the released human/mouse Cerberus models, predicting at 32bp resolution without cropping.
    The Hydra blocks use mamba_ssm's fused Triton scan when running on the torch backend on a GPU with `mamba_ssm`
    installed, and an equivalent but much slower and more memory-hungry keras.ops implementation otherwise.

    Note that unlike other CREsted model zoo architectures, this architecture is not suited for
    predicting individual regions out of the box.

    Parameters
    ----------
    seq_len
        Width of the input region.
        Cerberus default is 786432.
    num_classes
        Number of classes to predict.
        If an int, creates a single head with num_classes classes.
        If a list of integers, creates multiple heads in a list.
    num_conv_blocks
        Number of convolution blocks in the tower, after the stem block. Each halves the resolution.
    num_hydra_blocks
        Number of residual Hydra blocks.
    start_filters
        Number of filters of the stem conv, exponentially increasing towards `filters` through the conv tower.
    filters
        Number of filters at the end of the conv tower and in the Hydra blocks.
    filters_divisible_by
        Round conv tower filter numbers to a multiple of this.
    first_kernel_size
        Kernel size of the stem conv, directly interfacing the sequence.
    kernel_size
        Kernel size of the convolutions in the conv tower.
    conv_activation
        Activation function in the conv tower and the heads.
    d_state
        SSM state dimension of the Hydra layers.
    d_conv
        Kernel width of the Hydra layers' depthwise convolution.
    expand
        Expansion factor of the Hydra layers' inner dimension.
    headdim
        Channels per SSM head in the Hydra layers.
    chunk_size
        Chunk length of the SSM scan.
    hydra_dropout
        Dropout rate on the output of each Hydra layer.
    drop_path
        Stochastic depth rate of the last Hydra block, linearly ramped up from 0 at the first block.
    output_activation
        Final activation of the output heads.
    pool_type
        What kind of pooling type to use, one of 'max', 'average' or 'attention'.
    name
        Name of the model.

    Returns
    -------
    A Keras model.
    """
    sequence = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Stem (conv + pool)
    current = keras.layers.Conv1D(
        filters=start_filters,
        kernel_size=first_kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name="stem_conv",
    )(sequence)
    current = pool(current, pool_type=pool_type, pool_size=2, padding="same", name="stem_pool")

    # Conv tower: each block (batchnorm + activation + conv + pool)
    tower_filters = exp_linspace_int(
        start=start_filters,
        end=filters,
        num_modules=num_conv_blocks + 1,
        divisible_by=filters_divisible_by,
    )
    for cidx, layer_filters in enumerate(tower_filters[1:]):
        current = conv_block_bs(
            current,
            filters=layer_filters,
            kernel_size=kernel_size,
            pool_size=2,
            batch_norm=True,
            activation=conv_activation,
            conv_bias=False,
            pool_type=pool_type,
            bn_momentum=0.9,
            bn_sync=False,
            bn_epsilon=1e-4,
            kernel_initializer="he_normal",
            name_prefix=f"tower_conv_{cidx + 1}",
        )

    # Hydra tower: each block Residual(RMSNorm + Hydra + dropout + drop path)
    for hidx in range(num_hydra_blocks):
        current = hydra_block(
            current,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            dropout=hydra_dropout,
            drop_path=drop_path * hidx / (num_hydra_blocks - 1) if num_hydra_blocks > 1 else drop_path,
            name_prefix=f"hydra_{hidx + 1}",
        )

    # Heads: batchnorm + activation + pointwise conv
    def _head(inputs, n_tracks, head_name):
        current = normalize(inputs, "batch", momentum=0.9, epsilon=1e-4, name=f"{head_name}_batchnorm")
        current = activate(current, conv_activation, name=f"{head_name}_activation")
        return keras.layers.Conv1D(n_tracks, kernel_size=1, activation=output_activation, name=head_name)(current)

    if isinstance(num_classes, int):
        outputs = _head(current, num_classes, "head")
    elif isinstance(num_classes, cabc.Sequence):
        outputs = [_head(current, n_tracks, f"head_{i}") for i, n_tracks in enumerate(num_classes)]
    else:
        raise ValueError(
            f"Could not recognise num_classes argument ({num_classes}) as integer or list/tuple/sequence of integers."
        )

    return keras.Model(inputs=sequence, outputs=outputs, name=name)
