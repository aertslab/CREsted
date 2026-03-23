"""Utility functions for models."""

import inspect
from os import PathLike


def load_model(model_path: str | PathLike, compile: bool = False, custom_objects: dict | None = None, **kwargs):
    """
    Load in a .keras model.

    Wrapper around :func:`keras.saving.load_model` with compile=False by default, as it is generally unnecessary in CREsted.
    When running into possible layer serialization issues, `load_model` also tries again with all custom CREsted model layers and functions from {mod}`crested.tl.zoo.utils`.

    Parameters
    ----------
    model_path
        Path to a .keras saved model file.
    compile
        Whether to compile the model after loading, including the loss, optimizer and metrics from training.
        Default is False, since this is not needed for prediction or evaluation, and can lead to problems with loading and/or unexpected behavior like ignoring newly supplied loss functions.
    custom_objects
        Optional dictionary mapping names (strings) to custom classes or functions to be considered during deserialization.
    kwargs
        Arguments passed to :func`keras.saving.load_model`.

    Returns
    -------
    A :func:`keras.Model` object.

    See Also
    --------
    crested.get_model
    """
    # Import crested.tl to make sure activations and layers are available if serialization went correctly
    import keras

    import crested.tl  # noqa: F401
    try:
        model = keras.saving.load_model(
            model_path,
            compile=compile,
            custom_objects=custom_objects,
            **kwargs
        )
    # If getting TypeError due to serialized layers it still can't find, try manually including all layers from `crested.tl.zoo.utils` and try again
    except (TypeError, ValueError):
        from crested.tl.zoo import utils as zoo_utils
        if custom_objects is None:
            custom_objects = {}
        custom_objects.update(
            {name: obj for name, obj in inspect.getmembers(zoo_utils) if inspect.isclass(obj) or inspect.isfunction(obj)}
        )
        model = keras.saving.load_model(
            model_path,
            compile=compile,
            custom_objects=custom_objects,
            **kwargs
        )
    return model


def permute_model(
    model, new_input_shape: tuple[int, int]
):
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
    import keras
    new_input = keras.layers.Input(shape=new_input_shape)
    permuted_input = keras.layers.Permute((2, 1))(new_input)

    existing_model = model
    output = existing_model(permuted_input)

    new_model = keras.models.Model(inputs=new_input, outputs=output)

    return new_model
