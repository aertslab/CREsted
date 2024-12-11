"""Tooling kit that handles predictions, contribution scores, enhancer design, ... ."""

from __future__ import annotations

import os
import re

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from tqdm import tqdm

from crested.utils import fetch_sequences, one_hot_encode_sequence

if os.environ["KERAS_BACKEND"] == "tensorflow":
    from crested.tl._explainer_tf import Explainer
elif os.environ["KERAS_BACKEND"] == "torch":
    from crested.tl._explainer_torch import Explainer


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


def extract_layer_embeddings(
    input: str | list[str] | np.ndarray | AnnData,
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
        A (list of) trained keras model(s) to make predictions with.
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
    target_idx: int,
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
    target_idx
        Index of the target class to score.
        You can usually get this from running `list(anndata.obs_names).index(class_name)`.
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
    if not isinstance(target_idx, int):
        raise ValueError("Target index must be an integer.")
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

    positions = np.arange(start_position, end_position - window_size + 1, step_size)

    all_regions = [
        f"{chr_name}:{pos}-{pos + window_size}"
        for pos in range(start_position, end_position, step_size)
        if pos + window_size <= end_position
    ]
    predictions = predict(input=all_regions, model=model, genome=genome, **kwargs)
    predictions_class = predictions[:, target_idx]

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


def contribution_scores(
    input: str | list[str] | np.array | AnnData,
    target_idx: int | list[int] | None,
    model: keras.Model | list[keras.Model],
    method: str = "expected_integrated_grad",
    genome: os.PathLike | None = None,
    transpose: bool = False,
    all_class_names: list[str] | None = None,
    output_dir: os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method for the specified inputs.

    If mutliple models are provided, the contribution scores will be averaged across all models.

    These scores can then be plotted to visualize the importance of each base in the sequence
    using :func:`~crested.pl.patterns.contribution_scores`.

    Parameters
    ----------
    input
        Input data to calculate the contribution scores for. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    target_idx
        Index/indices of the target class(es) to calculate the contribution scores for.
        If this is an empty list, the contribution scores for the 'combined' class will be calculated.
        If this is None, the contribution scores for all classes will be calculated.
        You can get these for your classes of interest by running `list(anndata.obs_names).index(class_name)`.
    model
        A (list of) trained keras model(s) to calculate the contribution scores for.
    method
        Method to use for calculating the contribution scores.
        Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
    genome
        Path to the genome file. Required if input is an anndata object or region names.
    transpose
        Transpose the contribution scores to (N, C, 4, L) and one hots to (N, 4, L) (for compatibility with MoDISco).
    all_class_names
        Optional list of all class names in the dataset. If provided and output_dir is not None, will use these to name the output files.
    output_dir
        Path to the output directory to save the contribution scores and one hot seqs.
        Will create a separate npz file per class.
    verbose
        Boolean for disabling the logs and plotting progress of calculations using tqdm.

    Returns
    -------
    Contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4).

    See Also
    --------
    crested.pl.patterns.contribution_scores
    """
    if not isinstance(model, list):
        model = [model]
    if isinstance(target_idx, int):
        target_idx = [target_idx]
    elif target_idx is None:
        target_idx = list(range(0, model[0].output_shape[-1]))
    elif target_idx == []:
        if verbose:
            logger.info(
                "No class indices provided. Calculating contribution scores for the 'combined' class."
            )
        target_idx = [None]
    n_classes = len(target_idx)

    input_sequences = _transform_input(input, genome)

    if verbose:
        logger.info(
            f"Calculating contribution scores for {n_classes} class(es) and {input_sequences.shape[0]} region(s)."
        )
    N, L, D = input_sequences.shape

    # Initialize list to collect scores from each model
    scores_per_model = []

    # Iterate over models
    for m in tqdm(model, desc="Model", disable=not verbose):
        # Initialize scores for this model
        scores = np.zeros((N, n_classes, L, D))  # Shape: (N, C, L, 4)

        for i, class_index in enumerate(target_idx):
            # Initialize the explainer for the current model and class index
            explainer = Explainer(m, class_index=class_index)

            # Calculate contribution scores based on the selected method
            if method == "integrated_grad":
                scores[:, i, :, :] = explainer.integrated_grad(
                    input_sequences,
                    baseline_type="zeros",
                )
            elif method == "mutagenesis":
                scores[:, i, :, :] = explainer.mutagenesis(
                    input_sequences,
                    class_index=class_index,
                )
            elif method == "expected_integrated_grad":
                scores[:, i, :, :] = explainer.expected_integrated_grad(
                    input_sequences,
                    num_baseline=25,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

        # Collect scores from this model
        scores_per_model.append(scores)

    # Average the scores across models
    averaged_scores = np.mean(scores_per_model, axis=0)  # Shape: (N, C, L, 4)

    if transpose:
        averaged_scores = np.transpose(averaged_scores, (0, 1, 3, 2))
        input_sequences = np.transpose(input_sequences, (0, 2, 1))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for target_id in target_idx:
            class_name = all_class_names[target_id] if all_class_names else target_id
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_contrib.npz"),
                averaged_scores[:, target_id, :, :],
            )
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_oh.npz"),
                input_sequences,
            )

    return averaged_scores, input_sequences


def contribution_scores_specific(
    input: AnnData,
    target_idx: int | list[int] | None,
    model: keras.Model | list[keras.Model],
    genome: os.PathLike,
    method: str = "expected_integrated_grad",
    transpose: bool = True,
    output_dir: os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method only for the most specific regions per class.

    Contrary to :func:`~crested.tl.contribution_scores`, this function will only calculate one set of contribution scores per region per class.
    Expects the user to have ran `:func:~crested.pp.sort_and_filter_regions_on_specificity` beforehand.

    If multiple models are provided, the contribution scores will be averaged across all models.

    These scores can then be plotted to visualize the importance of each base in the sequence
    using :func:`~crested.pl.patterns.contribution_scores`.

    Parameters
    ----------
    input
        Input anndata to calculate the contribution scores for. Should have a 'Class name' column in .var.
    target_idx
        Index/indices of the target class(es) to calculate the contribution scores for.
        If this is an empty list, the contribution scores for the 'combined' class will be calculated.
        If this is None, the contribution scores for all classes will be calculated.
        You can get these for your classes of interest by running `list(anndata.obs_names).index(class_name)`.
    model
        A (list of) trained keras model(s) to calculate the contribution scores for.
    genome
        Path to the genome file.
    method
        Method to use for calculating the contribution scores.
        Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
    transpose
        Transpose the contribution scores to (N, C, 4, L) and one hots to (N, 4, L) (for compatibility with MoDISco).
        Defaults to True here since that is what modisco expects.
    output_dir
        Path to the output directory to save the contribution scores and one hot seqs.
        Will create a separate npz file per class.
    verbose
        Boolean for disabling the plotting progress of calculations using tqdm.

    Returns
    -------
    Contribution scores (N, 1, L, 4) and one-hot encoded sequences (N, L, 4).
    Since each region is specific to a class, the contribution scores are only calculated for that class.

    See Also
    --------
    crested.pl.patterns.contribution_scores
    crested.pp.sort_and_filter_regions_on_specificity
    """
    assert isinstance(input, AnnData), "Input should be an anndata object."
    if "Class name" not in input.var.columns:
        raise ValueError(
            "Run 'crested.pp.sort_and_filter_regions_on_specificity' first"
        )
    all_class_names = list(input.obs_names)
    if target_idx == []:
        raise ValueError("Can't calculate 'combined' scores for specific regions.")
    if target_idx is None:
        target_idx = list(range(0, len(all_class_names)))
    if not isinstance(target_idx, list):
        target_idx = [target_idx]
    all_scores = []
    all_one_hots = []

    for target_id in target_idx:
        class_name = all_class_names[target_id]
        class_regions = input.var[input.var["Class name"] == class_name].index.tolist()
        scores, one_hots = contribution_scores(
            input=class_regions,
            target_idx=target_id,
            model=model,
            method=method,
            genome=genome,
            verbose=verbose,
            output_dir=output_dir,
            all_class_names=all_class_names,
            transpose=transpose,
        )
        all_scores.append(scores)
        all_one_hots.append(one_hots)

    # Concatenate results across all classes
    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_one_hots, axis=0),
    )
