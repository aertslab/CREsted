from collections.abc import Callable, Sequence

import keras


def legnet_se_layer(x: keras.KerasTensor, inp: int, reduction: int = 4, name: str | None = None):
    """
    Squeeze-and-Excite layer.

    Parameters
    ----------
    x
        Input tensor with shape (batch, channels, length).
    inp
        Input channel size.
    reduction
        Reduction parameter. The default is 4.
    name : str, optional
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor with same shape as input.
    """
    # Global average pooling
    y = keras.ops.mean(x, axis=1)
    # Linear
    y = keras.layers.Dense(inp // reduction, activation=None, name=f"{name}_fc1" if name else None)(y)
    # SiLU
    y = keras.layers.Activation("swish", name=f"{name}_swish1" if name else None)(y)
    # Linear 2
    y = keras.layers.Dense(inp, activation=None, name=f"{name}_fc2" if name else None)(y)
    # Sigmoid
    y = keras.layers.Activation("sigmoid", name=f"{name}_sigmoid" if name else None)(y)
    # Multiplication
    output = keras.layers.Multiply(name=f"{name}_multiply" if name else None)([x, y])
    return output


def legnet_eff_block(x: keras.KerasTensor, out_ch: int, ks: int, resize_factor: int, activation: str = "swish", se_reduction: int | None = None, name: str | None = None):
    """
    EfficientNet-style block with inverted residuals and squeeze-excite.

    Parameters
    ----------
    x
        Input tensor.
    out_ch
        Output channels.
    ks
        Kernel size.
    resize_factor
        Expansion factor for the inner dimension.
    activation
        Activation function.
    se_reduction
        SE layer reduction factor. If None, same as resize_factor.
    name
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor.
    """
    se_reduction = resize_factor if se_reduction is None else se_reduction
    inner_dim = out_ch * resize_factor

    # Expansion
    y = keras.layers.Conv1D(
        filters=inner_dim, kernel_size=1, padding="same", use_bias=False, name=f"{name}_expand_conv" if name else None
    )(x)
    y = keras.layers.BatchNormalization(name=f"{name}_expand_bn" if name else None)(y)
    y = keras.layers.Activation(activation, name=f"{name}_expand_act" if name else None)(y)

    # Depthwise
    y = keras.layers.Conv1D(
        filters=inner_dim,
        kernel_size=ks,
        groups=inner_dim,
        padding="same",
        use_bias=False,
        name=f"{name}_dw_conv" if name else None,
    )(y)
    y = keras.layers.BatchNormalization(name=f"{name}_dw_bn" if name else None)(y)
    y = keras.layers.Activation(activation, name=f"{name}_dw_act" if name else None)(y)

    # Squeeze-and-Excite
    y = legnet_se_layer(y, inner_dim, reduction=se_reduction, name=f"{name}_se" if name else None)

    # Projection
    y = keras.layers.Conv1D(
        filters=out_ch, kernel_size=1, padding="same", use_bias=False, name=f"{name}_project_conv" if name else None
    )(y)
    y = keras.layers.BatchNormalization(name=f"{name}_project_bn" if name else None)(y)
    y = keras.layers.Activation(activation, name=f"{name}_project_act" if name else None)(y)

    return y


def legnet_local_block(x: keras.KerasTensor, out_ch: int, ks: int, activation: str = "swish", name: str | None = None):
    """
    Normal convolution block for the stem.

    Parameters
    ----------
    x
        Input tensor.
    out_ch
        Output channels.
    ks
        Kernel size.
    activation
        Activation function.
    name
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor.
    """
    y = keras.layers.Conv1D(
        filters=out_ch, kernel_size=ks, padding="same", use_bias=False, name=f"{name}_conv" if name else None
    )(x)
    y = keras.layers.BatchNormalization(name=f"{name}_bn" if name else None)(y)
    y = keras.layers.Activation(activation, name=f"{name}_act" if name else None)(y)

    return y


def residual_concat(x: keras.KerasTensor, fn: Callable, name: str | None = None):
    """
    Applies a function and concatenates the result with the input.

    Parameters
    ----------
    x
        Input tensor.
    fn
        Function to apply to x.
    name
        Name for the concatenation layer.

    Returns
    -------
    tensor
        Concatenated output.
    """
    y = fn(x)
    return keras.layers.Concatenate(axis=1, name=name)([y, x])


def legnet_mapper_block(x: keras.KerasTensor, out_features: int, name: str | None = None):
    """
    Mapper block for feature transformation.

    Parameters
    ----------
    x : tensor
        Input tensor.
    out_features : int
        Output feature dimension.
    name : str, optional
        Name prefix for keras.layers.

    Returns
    -------
    tensor
        Output tensor.
    """
    y = keras.layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    y = keras.layers.Conv1D(filters=out_features, kernel_size=1, name=f"{name}_conv" if name else None)(y)

    return y


def legnet(
    seq_len: int,
    num_classes: int,
    stem_ch: int = 64,
    stem_ks: int = 5,
    ef_ks: int = 3,
    ef_block_sizes: Sequence[int] = (80, 96, 112, 128),
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
    stem_ch
        Number of channels in the stem.
    stem_ks
        Kernel size for the stem.
    ef_ks
        Kernel size for EfficientNet blocks.
    ef_block_sizes
        List of output channel sizes for each stage.
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
    assert len(pool_sizes) == len(ef_block_sizes), "pool_sizes and ef_block_sizes must have the same length"

    # Input layer
    inputs = keras.layers.Input(shape=(seq_len, 4), name="input")

    # Stem layer
    x = legnet_local_block(inputs, out_ch=stem_ch, ks=stem_ks, activation=activation, name="stem")

    # Main blocks
    prev_filters = stem_ch
    for idx, (pool_size, out_filters) in enumerate(zip(pool_sizes, ef_block_sizes, strict=True)):
        # Residual concat with EffBlock
        y = legnet_eff_block(
            x, out_ch=prev_filters, ks=ef_ks, resize_factor=resize_factor, activation=activation, name=f"stage{idx}_eff"
        )
        x = keras.layers.Concatenate(axis=-1, name=f"stage{idx}_rescat")([y, x])

        # Local block
        x = legnet_local_block(x, out_ch=out_filters, ks=ef_ks, activation=activation, name=f"stage{idx}_local")

        # Pooling
        if pool_size != 1:
            x = keras.layers.MaxPooling1D(pool_size=pool_size, name=f"stage{idx}_pool")(x)

        prev_filters = out_filters

    # Mapper
    x = legnet_mapper_block(x, out_features=out_filters * 2, name="mapper")

    # Global average pooling across sequence length
    x = keras.ops.mean(x, axis=1)

    # Final
    x = keras.layers.Dense(out_filters * 2, name="final_dense")(x)
    x = keras.layers.BatchNormalization(name="final_bn")(x)
    x = keras.layers.Activation(activation, name="final_act")(x)
    outputs = keras.layers.Dense(num_classes, name="head", activation="softplus")(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model
