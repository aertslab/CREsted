from collections.abc import Sequence

import keras

from .utils import conv_block, legnet_eff_block


def legnet(
    seq_len: int,
    num_classes: int,
    stem_filters: int = 64,
    stem_kernel_size: int = 5,
    eff_kernel_size: int = 3,
    tower_filters: Sequence[int] = (80, 96, 112, 128),
    pool_sizes: Sequence[int] = (2, 2, 2, 2),
    resize_factor: int = 4,
    activation: str = "swish",
    name: str | None = "LegNet",
):
    """
    Build a model with efficientnetv2-like blocks, inspired by human_legnet/MPRALegNet (Agarwal et al. 2025, 10.1038/s41586-024-08430-9, github.com/autosome-ru/human_legnet).

    For the original implementation and the trained MPRALegNet models, please see github.com/autosome-ru/human_legnet.

    Parameters
    ----------
    seq_len
        Width of the input region.
    num_classes
        Number of classes to predict.
    stem_filters
        Number of convolution filters in the stem.
    stem_kernel_size
        Kernel size for the stem.
    eff_kernel_size
        Kernel size for EfficientNet blocks.
    tower_filters
        List of output filters for each stage in the convolution tower.
    pool_sizes
        List of pooling sizes for each stage.
    resize_factor
        Expansion factor for inverted residual blocks.
    activation
        Activation function. The default is "swish".
    name
        Model name.

    Returns
    -------
    keras.Model
        Compiled LegNet model.
    """
    assert len(pool_sizes) == len(tower_filters), "pool_sizes and tower_filters must have the same length"

    # Input layer
    inputs = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Stem layer
    x = conv_block(
        # Mandatory params
        inputs=inputs,
        filters=stem_filters,
        kernel_size=stem_kernel_size,
        pool_size=1,
        activation=activation,
        # adjusted params
        conv_bias=False,
        dropout=0.0,
        normalization="batch",
        padding="same",
        l2=0.0,
        # defaults to be kept
        res=False,
        batchnorm_momentum=0.99,
        name_prefix="stem",
    )

    # Main blocks
    tower_eff_filters = stem_filters  # in_ch
    tower_conv_filters = stem_filters  # out_ch
    for idx, (pool_size, tower_conv_filters) in enumerate(zip(pool_sizes, tower_filters, strict=True)):
        # Residual concat with EffBlock
        y = legnet_eff_block(
            x,
            out_ch=tower_eff_filters,
            ks=eff_kernel_size,
            resize_factor=resize_factor,
            activation=activation,
            name=f"stage{idx}_eff",
        )
        x = keras.layers.Concatenate(axis=-1, name=f"stage{idx}_rescat")([y, x])

        # Local block and pooling
        x = conv_block(
            # Mandatory params
            inputs=x,
            filters=tower_conv_filters,
            kernel_size=stem_kernel_size,
            pool_size=pool_size,
            activation=activation,
            # Adjusted defaults
            conv_bias=False,
            dropout=0.0,
            normalization="batch",
            padding="same",
            l2=0.0,
            # defaults to be kept
            res=False,
            batchnorm_momentum=0.99,
            name_prefix=f"stage{idx}_local",
        )

        tower_eff_filters = tower_conv_filters

    # Mapper block
    x = keras.layers.BatchNormalization(name="mapper_bn")(x)
    x = keras.layers.Conv1D(filters=tower_conv_filters * 2, kernel_size=1, name="mapper_conv")(y)

    # Global average pooling across sequence length
    x = keras.ops.mean(x, axis=1)

    # Final
    x = keras.layers.Dense(tower_conv_filters * 2, name="final_dense")(x)
    x = keras.layers.BatchNormalization(name="final_bn")(x)
    x = keras.layers.Activation(activation, name="final_act")(x)
    outputs = keras.layers.Dense(num_classes, name="head", activation="softplus")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model
