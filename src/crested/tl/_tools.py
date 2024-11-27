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


def _detect_input_type(input):
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
    input_type = _detect_input_type(input)

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
    embeddings = predict(input, embedding_model, genome, **kwargs)

    return embeddings


def predict(
    input: str | list[str] | np.array | AnnData,
    model: keras.Model | list[keras.Model],
    genome: os.PathLike | None = None,
    **kwargs,
) -> None | np.ndarray:
    """
    Make predictions using the model(s) on the full dataset.

    If a list of models is provided, the predictions will be averaged across all models.

    Parameters
    ----------
    input
        Input data to make predictions on. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    model
        A (list of) trained keras models to make predictions with.
    genome
        Path to the genome file. Required if input is an anndata object or region names.
    **kwargs
        Additional keyword arguments to pass to the keras.Model.predict method.

    Returns
    -------
    Predictions of shape (N, C)
    """
    input = _transform_input(input, genome)

    n_predict_steps = (
        input.shape[0] if os.environ["KERAS_BACKEND"] == "tensorflow" else None
    )

    if isinstance(model, list):
        if not all(isinstance(m, keras.Model) for m in model):
            raise ValueError("All items in the model list must be Keras models.")

        all_predictions = []
        for m in model:
            predictions = m.predict(input, steps=n_predict_steps, **kwargs)
            all_predictions.append(predictions)

        averaged_predictions = np.mean(all_predictions, axis=0)
        return averaged_predictions
    else:
        if not isinstance(model, keras.Model):
            raise ValueError("Model must be a Keras model or a list of Keras models.")

        predictions = model.predict(input, steps=n_predict_steps, **kwargs)
        return predictions


def score_gene_locus(
    gene_locus: str,
    all_class_names: list[str],
    class_name: str,
    model: keras.Model | list[keras.Model],
    genome: os.PathLike,
    strand: str = "+",
    upstream: int = 50000,
    downstream: int = 10000,
    central_size: int = 1000,
    step_size: int = 50,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Score regions upstream and downstream of a gene locus using the model's prediction.

    The model predicts a value for the {central_size} of each window.

    Parameters
    ----------
    gene_locus
        The gene locus to score in the format 'chr:start-end'.
        Start is the TSS for + strand and TES for - strand.
    all_class_names
        List of all class names in the model. Usually obtained with `list(anndata.obs_names)`.
    class_name
        Output class name to be used for prediction. Required to index the predictions.
    model
        A (list of) trained keras model(s) to make predictions with.
    genome
        Path to the genome file.
    strand
        '+' for positive strand, '-' for negative strand. Default '+'.
    upstream
        Distance upstream of the gene to score.
    downstream
        Distance downstream of the gene to score.
    central_size
        Size of the central region that the model predicts for.
    step_size
        Distance between consecutive windows.
    **kwargs
        Additional keyword arguments to pass to the keras.Model.predict method.

    Returns
    -------
    scores
        An array of prediction scores across the entire genomic range.
    coordinates
        An array of tuples, each containing the chromosome name and the start and end positions of the sequence for each window.
    min_loc
        Start position of the entire scored region.
    max_loc
        End position of the entire scored region.
    tss_position
        The transcription start site (TSS) position.

    See Also
    --------
    crested.tl.predict
    """
    chr_name, gene_locus = gene_locus.split(":")
    gene_start, gene_end = map(int, gene_locus.split("-"))

    # Detect window size from the model input shape
    if isinstance(model, list):
        input_shape = model[0].input_shape
    else:
        input_shape = model.input_shape
    window_size = input_shape[1]

    # Adjust upstream and downstream based on the strand
    if strand == "+":
        start_position = gene_start - upstream
        end_position = gene_end + downstream
        tss_position = gene_start  # TSS is at the gene_start for positive strand
    elif strand == "-":
        end_position = gene_end + upstream
        start_position = gene_start - downstream
        tss_position = gene_end  # TSS is at the gene_end for negative strand
    else:
        raise ValueError("Strand must be '+' or '-'.")

    start_position = max(0, start_position)
    total_length = abs(end_position - start_position)

    # Ratio to normalize the score contributions
    ratio = central_size / step_size

    try:
        idx = all_class_names.index(class_name)
    except ValueError as e:
        raise ValueError(
            f"Class name '{class_name}' not found in all_class_names"
        ) from e
    positions = np.arange(start_position, end_position - window_size + 1, step_size)

    all_regions = [
        f"{chr_name}:{pos}-{pos + window_size}"
        for pos in range(start_position, end_position, step_size)
        if pos + window_size <= end_position
    ]
    predictions = predict(input=all_regions, model=model, genome=genome, **kwargs)
    predictions_class = predictions[:, idx]

    # Map predictions to the score array
    scores = np.zeros(total_length)
    for _, (pos, pred) in enumerate(zip(positions, predictions_class)):
        central_start = pos + (window_size - central_size) // 2
        central_end = central_start + central_size

        # Compute indices relative to the scores array
        relative_start = central_start - start_position
        relative_end = central_end - start_position

        # Add the prediction to the scores array
        scores[relative_start:relative_end] += pred

    window_starts = positions
    window_ends = positions + window_size
    coordinates = np.array(
        list(zip([chr_name] * len(positions), window_starts, window_ends))
    )
    # Normalize the scores based on the number of times each position is included in the central window
    return (
        scores / ratio,
        coordinates,
        start_position,
        end_position,
        tss_position,
    )
