"""Tooling kit that handles predictions, contribution scores, enhancer design, ... ."""

from __future__ import annotations

import os
import re

import keras
import numpy as np
from anndata import AnnData

from crested.utils import (
    fetch_sequences,
    one_hot_encode_sequence,
)


def detect_input_type(input):
    """
    Detect the type of input provided.

    Parameters
    ----------
    input : str | list[str] | np.array | AnnData
        The input to detect the type of.

    Returns
    -------
    str
        One of ['sequence', 'region', 'anndata', 'array'], indicating the input type.
    """
    dna_pattern = re.compile("^[ACGTNacgtn]+$")
    if isinstance(input, AnnData):
        return "anndata"
    elif isinstance(input, list):
        if all(":" in str(item) for item in input):  # List of regions
            return "region"
        elif all(
            isinstance(item, str) and dna_pattern.match(item) for item in input
        ):  # List of sequences
            return "sequence"
        else:
            raise ValueError(
                "List input must contain only valid region strings (chrom:var-end) or DNA sequences."
            )
    elif isinstance(input, str):
        if ":" in input:  # Single region
            return "region"
        elif dna_pattern.match(input):  # Single DNA sequence
            return "sequence"
        else:
            raise ValueError(
                "String input must be a valid region string (chrom:var-end) or DNA sequence."
            )
    elif isinstance(input, np.ndarray):
        if input.ndim == 3:
            return "array"
        else:
            raise ValueError("Input one hot array must have shape (N, L, 4).")
    else:
        raise ValueError(
            "Unsupported input type. Must be AnnData, str, list, or np.ndarray."
        )


def _transform_input(input, genome: os.PathLike | None = None) -> np.ndarray:
    """
    Transform the input into a one-hot encoded matrix based on its type.

    Parameters
    ----------
    input : str | list[str] | np.array | AnnData
        Input data to preprocess. Can be a sequence, list of sequences, region, list of regions, or an AnnData object.
    genome : str | None
        Path to the genome file. Required if input is a region or AnnData.

    Returns
    -------
    One-hot encoded matrix of shape (N, L, 4), where N is the number of sequences/regions and L is the sequence length.
    """
    input_type = detect_input_type(input)

    if input_type == "anndata":
        if genome is None:
            raise ValueError(
                "Genome file is required to fetch sequences for regions in AnnData."
            )
        regions = list(input.var_names)
        sequences = fetch_sequences(regions, genome)
    elif input_type == "region":
        if genome is None:
            raise ValueError("Genome file is required to fetch sequences for regions.")
        sequences = fetch_sequences(input, genome)
    elif input_type == "sequence":
        sequences = input if isinstance(input, list) else [input]
    elif input_type == "array":
        assert input.ndim == 3, "Input one hot array must have shape (N, L, 4)."
        return input

    one_hot_data = np.array(
        [one_hot_encode_sequence(seq, expand_dim=False) for seq in sequences]
    )

    return one_hot_data


def get_embeddings(
    input: str | list[str] | np.array | AnnData,
    model: keras.Model,
    layer_name: str,
    genome: os.PathLike | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Extract embeddings from a specified layer for all inputs.

    Parameters
    ----------
    input
        Input data to get embeddings for. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    model
        A trained keras model from which to extract the embeddings.
    layer_name
        The name of the layer from which to extract the embeddings.
    genome
        Path to the genome file. Required if input is an anndata object or region names.
    **kwargs
        Additional keyword arguments to pass to the keras.Model.predict method.

    Returns
    -------
    Embeddings of shape (N, D), where N is the number of regions in the input and D is the size of the embedding layer.
    """
    layer_names = [layer.name for layer in model.layers]
    if layer_name not in layer_names:
        raise ValueError(
            f"Layer '{layer_name}' not found in model. Options (in reverse) are: {layer_names[::-1]}"
        )
    embedding_model = keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    input = _transform_input(input, genome)
    n_predict_steps = (
        input.shape[0] if os.environ["KERAS_BACKEND"] == "tensorflow" else None
    )
    embeddings = embedding_model.predict(input, steps=n_predict_steps, **kwargs)

    return embeddings


def predict(
    input: str | list[str] | np.array | AnnData,
    model: keras.Model,
    genome: os.PathLike | None = None,
) -> None | np.ndarray:
    """
    Make predictions using the model on the full dataset.

    If anndata and model_name are provided, will add the predictions to anndata as a .layers[model_name] attribute.
    Else, will return the predictions as a numpy array.

    Parameters
    ----------
    input
        Input data to make predictions on. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    model
        A trained keras model to make predictions with.
    genome
        Path to the genome file. Required if input is an anndata object or region names.

    Returns
    -------
    Predictions of shape (N, C)
    """
    input = _transform_input(input, genome)

    n_predict_steps = (
        input.shape[0] if os.environ["KERAS_BACKEND"] == "tensorflow" else None
    )

    predictions = model.predict(input, steps=n_predict_steps)

    return predictions
