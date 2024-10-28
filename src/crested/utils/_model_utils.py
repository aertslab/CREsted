"""Utility functions for models."""

import keras


def permute_model(
    model: keras.models.Model, new_input_shape: tuple[int, int]
) -> keras.models.Model:
    """
    Add a permutation layer to the input of a model to change the shape from (B, W, C) to (B, C, W) or vice versa.

    Useful to convert from tensorflow consenus format to torch (e.g. to use with tangermeme).

    Parameters
    ----------
    model
        The keras model to add the permutation layer to.
    new_input_shape
        The new input shape to the model (e.g. (4, 500))

    Returns
    -------
    The new model with the permutation layer added to the input.

    Example
    -------
    >>> inputs = keras.layers.Input(shape=(4, 500))
    >>> model = keras.models.Model(inputs=inputs, outputs=inputs)
    >>> new_model = crested.utils.permute_model(model, (500, 4))
    """
    new_input = keras.layers.Input(shape=new_input_shape)
    permuted_input = keras.layers.Permute((2, 1))(new_input)

    existing_model = model
    output = existing_model(permuted_input)

    new_model = keras.models.Model(inputs=new_input, outputs=output)

    return new_model
