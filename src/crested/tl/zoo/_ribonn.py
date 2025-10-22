import keras


def RiboNN(
    num_targets,
    seq_len: int = 13318, # codebase says 12288, paper says 13318?
    num_convblocks: int = 10, # 10 blocks for human, 8 for mouse
    filters: int = 64,
    kernel_size: int = 5,
    pool_size: int = 2,
    activation: str = 'relu',
    dropout = 0.3,
    ln_eps = 0.007,
    padding = 'valid',
    name = 'ribonn'
):
    """
    Build a RiboNN model.

    Parameters
    ----------
    num_targets
        The number of output classes to predict. For pre-trained multi-task RiboNN, 78 for human and 68 for mouse.
    seq_len
        Length of the input sequences. Default is 13318.
    num_convblocks
        Number of convolution blocks to use. Default is 10 (as used in the human model), the mouse model used 8.
    filters
        Number of filters to use for the convolutional layers. Default is 64.
    kernel_size
        Kernel size to use int he convolutional layers. Default is 5.
    pool_size
        Pooling size after each convblock. Default is 2.
    activation
        Activation function to use, as recognised by keras.layers.Activation(). Default is 'relu'.
    dropout
        Dropout percentage to use across the model. Default is 0.3.
    ln_eps
        Epsilon (small value to prevent dividing by zero) to use in the layer normalisation. Default is 0.007 to match the PyTorch implementation.
    padding
        Padding mechanism to use for the convolutional layers. Default is 'valid'.
    name
        Name for the total model. Can be a string or None. Default is 'ribonn'

    Returns
    -------
    A keras.Model instance.
    """
    # Input channels: 4 nucleotides ORF label
    sequence = keras.layers.Input(shape=(seq_len, 5), name="input")

    # Initial conv
    x = keras.layers.Convolution1D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        use_bias = False,
        name="init_conv"
    )(sequence)

    # Conv blocks
    for conv_i in range(num_convblocks):
        x = conv_block_ribonn(
            inputs = x,
            filters = filters,
            kernel_size = kernel_size,
            pool_size = pool_size,
            activation = activation,
            dropout = dropout,
            ln_eps = ln_eps,
            padding = padding,
            name_prefix = f"convblock_{conv_i}",
        )
    # Head
    x = keras.layers.Activation(activation, name="head_activation1")(x)
    x = keras.ops.transpose(x, (0, 2, 1))
    x = keras.layers.Flatten(name = "head_flatten")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout1")(x)
    x = keras.layers.Dense(filters, use_bias = False, name="head_dense1")(x)
    x = keras.layers.Activation(activation, name="head_activation2")(x)
    x = keras.layers.BatchNormalization(
        momentum = 0.9, # Keras momentum is 1 - torch normalisation, they use torch default (0.1)
        epsilon = 1e-5,
        name="head_batchnorm"
    )(x)
    x = keras.layers.Dropout(dropout, name="head_dropout2")(x)
    x = keras.layers.Dense(num_targets, name="head_dense2")(x)

    return keras.Model(inputs=sequence, outputs=x, name=name)


def conv_block_ribonn(
    inputs: keras.KerasTensor,
    filters: int,
    kernel_size: int,
    pool_size: int,
    activation: str,
    dropout: float,
    ln_eps: float = 0.007,
    stride: int = 1,
    padding: str = 'same',
    name_prefix: str | None = None,
):
    """
    RiboNN convolution building block.

    Parameters
    ----------
    inputs
        Input tensor representing the data.
    filters
        Number of filters in the convolutional layer.
    kernel_size
        Size of the convolutional kernel.
    pool_size
        Size of the max-pooling kernel.
    activation
        Activation function applied after convolution.
    dropout
        Dropout rate (default is 0.1).
    ln_eps
        Epsilon value to use for the layer normalisation.
    stride
        Stride for the convolutional layer. Default is 1.
    padding
        Padding type for the convolutional layer (default is "same", which pads with zeros).
    name_prefix
        Prefix for layer names.

    Returns
    -------
    The output tensor of the convolution block.
    """
    # layernorm -> activation -> conv -> dropout -> pool
    x = keras.layers.LayerNormalization(
        axis = -1,
        epsilon = ln_eps,
        name=name_prefix + "_layernorm" if name_prefix else None
    )(inputs)
    x = keras.layers.Activation(
        activation, name=name_prefix + "_activation" if name_prefix else None
    )(x)
    x = keras.layers.Convolution1D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        name=name_prefix + "_conv" if name_prefix else None,
    )(x)
    x = keras.layers.Dropout(
        dropout,
        name=name_prefix + "_dropout" if name_prefix else None
    )(x)
    x = keras.layers.MaxPooling1D(
        pool_size=pool_size,
        name=name_prefix + "_pool" if name_prefix else None,
    )(x)
    return x
