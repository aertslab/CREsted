"""Tooling kit that handles predictions, contribution scores, enhancer design, ... ."""

from __future__ import annotations

import math
import os
from typing import Any

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from tqdm import tqdm

from crested._genome import Genome
from crested.tl._explainer import (
    integrated_grad,
    mutagenesis,
    saliency_map,
    window_shuffle,
)
from crested.tl._utils import (
    create_random_sequences,
    generate_motif_insertions,
    parse_starting_sequences,
)
from crested.utils._seq_utils import (
    generate_mutagenesis,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
)
from crested.utils._utils import (
    EnhancerOptimizer,
    _transform_input,
    _weighted_difference,
)


def extract_layer_embeddings(
    input: str | list[str] | np.ndarray | AnnData,
    model: keras.Model,
    layer_name: str,
    genome: Genome | os.PathLike | None = None,
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
        Genome or path to the genome fasta. Required if no genome is registered and input is an anndata object or region names.
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

class PredictPyDataset(keras.utils.PyDataset):
    """
    A Keras-compatible dataset for batched prediction.

    Wraps an array-like input (e.g., one-hot encoded sequences) and yields batches for model prediction.
    """

    def __init__(self, input_array, batch_size=128):
        """
        Initialize the prediction dataset.

        Parameters
        ----------
        input_array : array-like
            Input data, typically a NumPy array of shape (N, L, 4).
        batch_size : int, optional
            Number of samples per batch. Default is 128.
        """
        super().__init__()
        self.input_array = input_array
        self.batch_size = batch_size

    def __len__(self):
        """
        Return the number of batches.

        Returns
        -------
        int
            Total number of batches per epoch.
        """
        return math.ceil(len(self.input_array) / self.batch_size)

    def __getitem__(self, idx):
        """
        Retrieve a single batch of data.

        Parameters
        ----------
        idx : int
            Index of the batch.

        Returns
        -------
        array-like
            A batch of input data.
        """
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.input_array))
        batch = self.input_array[low:high]
        return batch


def predict(
    input: str | list[str] | np.array | AnnData,
    model: keras.Model | list[keras.Model],
    genome: Genome | os.PathLike | None = None,
    batch_size: int = 128,
    **kwargs,
) -> None | np.ndarray:
    """
    Make predictions using the model(s) on some input that represents sequences.

    If a list of models is provided, the predictions will be averaged across all models.

    Parameters
    ----------
    input
        Input data to make predictions on. Can be a (list of) sequence(s), a (list of) region name(s),
        a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    model
        A (list of) trained keras model(s) to make predictions with.
    genome
        Genome or path to the genome file. Required if no genome is registered and input is an anndata object or region names.
    batch_size
        Batch size to use for predictions.
    **kwargs
        Additional keyword arguments to pass to the keras.Model.predict method.

    Returns
    -------
    Predictions of shape (N, C)

    Example
    -------
    >>> my_sequences = ["ACGT", "CGTA", "GTAC"]
    >>> predictions = predict(
    ...     input=my_sequences,
    ...     model=my_trained_model,
    ... )
    """
    input = _transform_input(input, genome)
    dataset = PredictPyDataset(input, batch_size)

    if isinstance(model, list):
        if not all(isinstance(m, keras.Model) for m in model):
            raise ValueError("All items in the model list must be Keras models.")
        all_preds = []
        for m in model:
            preds = m.predict(dataset, **kwargs)
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)
    else:
        if not isinstance(model, keras.Model):
            raise ValueError("Model must be a Keras model or a list of Keras models.")
        return model.predict(dataset, **kwargs)

def score_gene_locus(
    chr_name: str,
    gene_start: int,
    gene_end: int,
    target_idx: int,
    model: keras.Model | list[keras.Model],
    genome: Genome | os.PathLike | None = None,
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
    chrom
        The chromosome name.
    gene_start
        The start position of the gene locus (TSS for + strand).
    gene_end
        The end position of the gene locus (TSS for - strand).
    target_idx
        Index of the target class to score.
        You can usually get this from running `list(anndata.obs_names).index(class_name)`.
    model
        A (list of) trained keras model(s) to make predictions with.
    genome
        Genome or path to the genome file. Required if no genome is registered.
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
    window_size: int | None = 7,
    n_shuffles: int | None = 24,
    genome: Genome | os.PathLike | None = None,
    transpose: bool = False,
    all_class_names: list[str] | None = None,
    batch_size: int = 128,
    output_dir: os.PathLike | None = None,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method for the specified inputs.

    If multiple models are provided, the contribution scores will be averaged across all models.

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
        Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad', 'saliency_map', 'window_shuffle', 'window_shuffle_uniform'.
    window_size
        Window size to use if using the method 'window_shuffle' or 'window_shuffle_uniform'.
    n_shuffles
        Number of times to shuffle per window if using the method 'window_shuffle' or 'window_shuffle_uniform'.
    genome
        Genome or path to the genome fasta. Required if no genome is registered and input is an anndata object or region names.
    transpose
        Transpose the contribution scores to (N, C, 4, L) and one hots to (N, 4, L) (for compatibility with MoDISco).
    all_class_names
        Optional list of all class names in the dataset. If provided and output_dir is not None, will use these to name the output files.
    batch_size
        Maximum number of input sequences to predict at once when calculating scores.
        Useful for methods like 'integrated_grad' which also calculate 25 background sequence contributions together with the sequence's contributions in one batch.
        Default is 128.
    output_dir
        Path to the output directory to save the contribution scores and one hot seqs.
        Will create a separate npz file per class.
    seed
        Seed to use for shuffling regions. Only used in "expected_integrated_grad".
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

    scores_per_model = []
    for m in tqdm(model, desc="Model", disable=not verbose):
        scores = np.zeros((N, n_classes, L, D))  # Shape: (N, C, L, 4)

        for i, class_index in enumerate(target_idx):
            if method == "integrated_grad":
                scores[:, i, :, :] = integrated_grad(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    baseline_type="zeros",
                    num_baselines=1,
                    num_steps=25,
                    batch_size=batch_size,
                )
            elif method == "mutagenesis":
                scores[:, i, :, :] = mutagenesis(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    batch_size=batch_size,
                )
            elif method == "expected_integrated_grad":
                scores[:, i, :, :] = integrated_grad(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    baseline_type="random",
                    num_baselines=25,
                    num_steps=25,
                    batch_size=batch_size,
                    seed=seed,
                )
            elif method == "saliency_map":
                scores[:, i, :, :] = saliency_map(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    batch_size=batch_size,
                )
            elif method == "window_shuffle":
                scores[:, i, :, :] = window_shuffle(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    window_size=window_size,
                    n_shuffles=n_shuffles,
                    uniform=False,
                    batch_size=batch_size,
                )
            elif method == "window_shuffle_uniform":
                scores[:, i, :, :] = window_shuffle(
                    input_sequences,
                    model=m,
                    class_index=class_index,
                    window_size=window_size,
                    n_shuffles=n_shuffles,
                    uniform=True,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

        scores_per_model.append(scores)

    # Average the scores across models
    averaged_scores = np.mean(scores_per_model, axis=0)  # Shape: (N, C, L, 4)

    if transpose:
        averaged_scores = np.transpose(averaged_scores, (0, 1, 3, 2))
        input_sequences = np.transpose(input_sequences, (0, 2, 1))

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(n_classes):
            target_id = target_idx[i]
            class_name = (
                all_class_names[target_id]
                if all_class_names
                else f"class_id_{target_id}"
            )
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_contrib.npz"),
                averaged_scores[:, i, :, :],
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
    genome: Genome | os.PathLike | None = None,
    method: str = "expected_integrated_grad",
    transpose: bool = True,
    batch_size: int = 128,
    output_dir: os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method only for the most specific regions per class.

    Contrary to :func:`~crested.tl.contribution_scores`, this function will only calculate one set of contribution scores per region per class.
    Expects the user to have ran :func:`~crested.pp.sort_and_filter_regions_on_specificity` beforehand.

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
        Genome or Path to the genome file. Required if no genome is registered.
    method
        Method to use for calculating the contribution scores.
        Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad', 'saliency_map'.
    transpose
        Transpose the contribution scores to (N, C, 4, L) and one hots to (N, 4, L) (for compatibility with MoDISco).
        Defaults to True here since that is what modisco expects.
    batch_size
        Maximum number of input sequences to predict at once when calculating scores.
        Useful for methods like 'integrated_grad' which also calculate 25 background sequence contributions together with the sequence's contributions in one batch.
        Default is 128.
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
            batch_size=batch_size,
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


def enhancer_design_in_silico_evolution(
    n_mutations: int,
    target: int | np.ndarray,
    model: keras.Model | list[keras.Model],
    n_sequences: int = 1,
    return_intermediate: bool = False,
    no_mutation_flanks: tuple[int, int] | None = None,
    target_len: int | None = None,
    enhancer_optimizer: EnhancerOptimizer | None = None,
    starting_sequences: str | list | None = None,
    acgt_distribution: np.ndarray[float] | None = None,
    **kwargs: dict[str, Any],
) -> list | tuple[list[dict], list]:
    """
    Create synthetic enhancers for a specified class using in silico evolution (ISE).

    Parameters
    ----------
    n_mutations
        Number of mutations to make in each sequence.
        20 is a good starting point for most cases.
    target
        Using the default weighted_difference optimization function this should be the index of the target class to design enhancers for.
        This gets passed to the `get_best` function of the EnhancerOptimizer, so can represent other target values too.
    model
        A (list of) trained keras model(s) to design enhancers with.
        If a list of models is provided, the predictions will be averaged across all models.
    n_sequences
        Number of enhancers to design
    return_intermediate
        If True, returns a dictionary with predictions and changes made in intermediate steps for selected
        sequences
    no_mutation_flanks
        A tuple of integers which determine the regions in each flank to not do insertions.
    target_len
        Length of the area in the center of the sequence to make mutations in.
        Ignored if no_mutation_flanks is provided.
    acgt_distribution
        An array of floats representing the distribution of A, C, G, and T in the genome (in that order).
        If the array is of shape (L, 4), it will be assumed to be per position. If it is of shape (4,), it will be assumed to be overall.
        If None, a uniform distribution will be used.
        This will be used to generate random sequences if starting_sequences is not provided.
        You can calculate these using :func:`~crested.utils.calculate_nucleotide_distribution`.
    kwargs
        Keyword arguments that will be passed to the `get_best` function of the EnhancerOptimizer

    Returns
    -------
    A list of designed sequences. If return_intermediate is True, will also return a list of dictionaries of intermediate
    mutations and predictions.

    See Also
    --------
    crested.utils.EnhancerOptimizer
    crested.utils.calculate_nucleotide_distribution

    Examples
    --------
    >>> acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    ...     my_anndata, genome, per_position=True
    ... )  # shape (L, 4)
    >>> target_idx = my_anndata.obs_names.index("my_celltype")
    >>> (
    ...     intermediate_results,
    ...     designed_sequences,
    ... ) = crested.tl.enhancer_design_in_silico_evolution(
    ...     n_mutations=20,
    ...     target=target_idx,
    ...     model=my_trained_model,
    ...     n_sequences=1,
    ...     return_intermediate=True,
    ...     acgt_distribution=acgt_distribution,
    ... )
    """
    if enhancer_optimizer is None:
        enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)
    if not isinstance(model, list):
        model = [model]
    seq_len = model[0].input_shape[1]

    # determine the flanks without changes
    if no_mutation_flanks is not None and target_len is not None:
        logger.warning(
            "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
        )
    elif no_mutation_flanks is None and target_len is not None:
        if (seq_len - target_len) % 2 == 0:
            no_mutation_flanks = (
                int((seq_len - target_len) // 2),
                int((seq_len - target_len) // 2),
            )
        else:
            no_mutation_flanks = (
                int((seq_len - target_len) // 2),
                int((seq_len - target_len) // 2) + 1,
            )

    elif no_mutation_flanks is None and target_len is None:
        no_mutation_flanks = (0, 0)

    # create initial sequences
    if starting_sequences is None:
        if acgt_distribution is None:
            logger.warning(
                "No nucleotide distribution provided. Using uniform distribution."
            )
        initial_sequences = create_random_sequences(
            n_sequences=n_sequences,
            seq_len=seq_len,
            acgt_distribution=acgt_distribution,
        )
    else:
        initial_sequences = parse_starting_sequences(starting_sequences)
        n_sequences = initial_sequences.shape[0]

    # initialize
    designed_sequences: list[str] = []
    intermediate_info_list: list[dict] = []

    sequence_onehot_prev_iter = np.zeros((n_sequences, seq_len, 4), dtype=np.uint8)

    # calculate total number of mutations per sequence
    _, L, A = sequence_onehot_prev_iter.shape
    start, end = 0, L
    start = no_mutation_flanks[0]
    end = L - no_mutation_flanks[1]
    TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ = (end - start) * (A - 1)

    mutagenesis = np.zeros((n_sequences, TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4))

    for i, sequence in enumerate(initial_sequences):
        sequence_onehot_prev_iter[i] = one_hot_encode_sequence(sequence)

    for _iter in tqdm(range(n_mutations)):
        baseline_prediction = []
        for m in model:
            baseline_prediction.append(
                m.predict(sequence_onehot_prev_iter, verbose=False)
            )
        baseline_prediction = np.mean(baseline_prediction, axis=0)
        if _iter == 0:
            for i in range(n_sequences):
                # initialize info
                intermediate_info_list.append(
                    {
                        "initial_sequence": hot_encoding_to_sequence(
                            sequence_onehot_prev_iter[i]
                        ),
                        "changes": [(-1, "N")],
                        "predictions": [baseline_prediction[i]],
                        "designed_sequence": "",
                    }
                )

        # do all possible mutations
        for i in range(n_sequences):
            mutagenesis[i] = generate_mutagenesis(
                sequence_onehot_prev_iter[i : i + 1],
                include_original=False,
                flanks=no_mutation_flanks,
            )
        mutagenesis_predictions = []
        for m in model:
            mutagenesis_prediction = m.predict(
                mutagenesis.reshape(
                    (n_sequences * TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4)
                ),
                verbose=False,
            )

            mutagenesis_prediction = mutagenesis_prediction.reshape(
                (
                    n_sequences,
                    TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ,
                    mutagenesis_prediction.shape[1],
                )
            )
            mutagenesis_predictions.append(mutagenesis_prediction)
        mutagenesis_predictions = np.mean(mutagenesis_predictions, axis=0)
        for i in range(n_sequences):
            best_mutation = enhancer_optimizer.get_best(
                mutated_predictions=mutagenesis_predictions[i],
                original_prediction=baseline_prediction[i],
                target=target,
                **kwargs,
            )
            sequence_onehot_prev_iter[i] = mutagenesis[
                i, best_mutation : best_mutation + 1, :
            ]
            if return_intermediate:
                mutation_index = best_mutation // 3 + no_mutation_flanks[0]
                changed_to = hot_encoding_to_sequence(
                    sequence_onehot_prev_iter[i, mutation_index, :]
                )
                intermediate_info_list[i]["changes"].append(
                    (mutation_index, changed_to)
                )
                intermediate_info_list[i]["predictions"].append(
                    mutagenesis_predictions[i][best_mutation]
                )

    # get final sequence
    for i in range(n_sequences):
        best_mutation = enhancer_optimizer.get_best(
            mutated_predictions=mutagenesis_predictions[i],
            original_prediction=baseline_prediction[i],
            target=target,
            **kwargs,
        )

        designed_sequence = hot_encoding_to_sequence(
            mutagenesis[i, best_mutation : best_mutation + 1, :]
        )

        designed_sequences.append(designed_sequence)

        if return_intermediate:
            intermediate_info_list[i]["designed_sequence"] = designed_sequence

    if return_intermediate:
        return intermediate_info_list, designed_sequences
    else:
        return designed_sequences


def enhancer_design_motif_insertion(
    patterns: dict,
    model: keras.Model | list[keras.Model],
    target: int | np.ndarray,
    n_sequences: int = 1,
    insertions_per_pattern: dict | None = None,
    return_intermediate: bool = False,
    no_mutation_flanks: tuple[int, int] | None = None,
    target_len: int | None = None,
    preserve_inserted_motifs: bool = True,
    enhancer_optimizer: EnhancerOptimizer | None = None,
    starting_sequences: str | list | None = None,
    acgt_distribution: np.ndarray[float] | None = None,
    **kwargs: dict[str, Any],
) -> list | tuple[list[dict], list]:
    """
    Create synthetic enhancers using motif insertions.

    Parameters
    ----------
    patterns
        Dictionary of patterns to be implemented in the form {'pattern_name': 'pattern_sequence'}
    model
        A (list of) trained keras model(s) to design enhancers with.
        If a list of models is provided, the predictions will be averaged across all models.
    target
        Using the default weighted_difference optimization function this should be the index of the target class to design enhancers for.
        This gets passed to the `get_best` function of the EnhancerOptimizer, so can represent other target values too.
    n_sequences
        Number of enhancers to design.
    insertions_per_pattern
        Dictionary of number of patterns to be implemented in the form {'pattern_name': number_of_insertions}.
        If not provided, each pattern is inserted once.
    return_intermediate
        If True, returns a dictionary with predictions and changes made in intermediate steps.
    no_mutation_flanks
        A tuple specifying regions in each flank where no modifications should occur.
    target_len
        Length of the area in the center of the sequence to make insertions, ignored if `no_mutation_flanks` is set.
    preserve_inserted_motifs
        If True, prevents motifs from being inserted on top of previously inserted motifs.
    enhancer_optimizer
        An instance of EnhancerOptimizer, defining how sequences should be optimized.
        If None, a default EnhancerOptimizer will be initialized using `_weighted_difference`
        as optimization function.
    starting_sequences
        An optional DNA sequence or a list of DNA sequences that will be used instead of randomly generated
        sequences. If provided, n_sequences is ignored
    acgt_distribution
        An array of floats representing the distribution of A, C, G, and T in the genome (in that order).
        If the array is of shape (L, 4), it will be assumed to be per position. If it is of shape (4,), it will be assumed to be overall.
        If None, a uniform distribution will be used.
        This will be used to generate random sequences if starting_sequences is not provided.
        You can calculate these using :func:`~crested.utils.calculate_nucleotide_distribution`.
    kwargs
        Additional arguments passed to `get_best` function of EnhancerOptimizer.

    Returns
    -------
    A list of designed sequences, and if `return_intermediate=True`, a list of intermediate results.

    See Also
    --------
    crested.utils.EnhancerOptimizer
    crested.utils.calculate_nucleotide_distribution

    Examples
    --------
    >>> acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    ...     my_anndata, genome, per_position=True
    ... )  # shape (L, 4)
    >>> target_idx = my_anndata.obs_names.index("my_celltype")
    >>> my_motifs = {
    ...     "motif1": "ACGTTTGA",
    ...     "motif2": "TGCA",
    ... }
    >>> (
    ...     intermediate_results,
    ...     designed_sequences,
    ... ) = crested.tl.enhancer_design_motif_insertion(
    ...     patterns=my_motifs,
    ...     n_mutations=20,
    ...     target=target_idx,
    ...     model=my_trained_model,
    ...     n_sequences=1,
    ...     return_intermediate=True,
    ...     acgt_distribution=acgt_distribution,
    ... )
    """
    if enhancer_optimizer is None:
        enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)

    if not isinstance(model, list):
        model = [model]

    seq_len = model[0].input_shape[1]

    # Determine mutation flanks
    if no_mutation_flanks is not None and target_len is not None:
        logger.warning(
            "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
        )
    elif no_mutation_flanks is None and target_len is not None:
        if (seq_len - target_len) % 2 == 0:
            no_mutation_flanks = ((seq_len - target_len) // 2,) * 2
        else:
            no_mutation_flanks = (
                (seq_len - target_len) // 2,
                (seq_len - target_len) // 2 + 1,
            )
    elif no_mutation_flanks is None and target_len is None:
        no_mutation_flanks = (0, 0)

    if insertions_per_pattern is None:
        insertions_per_pattern = dict.fromkeys(patterns, 1)

    # Generate initial sequences
    if starting_sequences is None:
        if acgt_distribution is None:
            logger.warning(
                "No nucleotide distribution provided. Using uniform distribution."
            )
        initial_sequences = create_random_sequences(
            n_sequences=n_sequences,
            seq_len=seq_len,
            acgt_distribution=acgt_distribution,
        )
    else:
        initial_sequences = parse_starting_sequences(starting_sequences)
        n_sequences = initial_sequences.shape[0]

    designed_sequences = []
    intermediate_info_list = []

    for idx, sequence in enumerate(initial_sequences):
        sequence_onehot = one_hot_encode_sequence(sequence)
        inserted_motif_locations = np.array([]) if preserve_inserted_motifs else None

        if return_intermediate:
            baseline_prediction = np.mean(
                [m.predict(sequence_onehot, verbose=False) for m in model], axis=0
            )
            intermediate_info_list.append(
                {
                    "initial_sequence": sequence,
                    "changes": [(-1, "N")],
                    "predictions": [baseline_prediction[0]],
                    "designed_sequence": "",
                }
            )

        # Insert motifs sequentially
        for pattern_name, num_insertions in insertions_per_pattern.items():
            motif_onehot = one_hot_encode_sequence(patterns[pattern_name])
            motif_length = motif_onehot.shape[1]

            for _ in range(num_insertions):
                baseline_prediction = np.mean(
                    [m.predict(sequence_onehot, verbose=False) for m in model], axis=0
                )

                # Generate all motif insertion possibilities
                mutagenesis, insertion_locations = generate_motif_insertions(
                    sequence_onehot,
                    motif_onehot,
                    flanks=no_mutation_flanks,
                    masked_locations=inserted_motif_locations,
                )

                # Predict changes
                mutagenesis_predictions = np.mean(
                    [m.predict(mutagenesis, verbose=False) for m in model], axis=0
                )

                # Select best insertion site
                best_mutation = enhancer_optimizer.get_best(
                    mutated_predictions=mutagenesis_predictions,
                    original_prediction=baseline_prediction,
                    target=target,
                    **kwargs,
                )

                sequence_onehot = mutagenesis[best_mutation : best_mutation + 1]

                if preserve_inserted_motifs:
                    inserted_motif_locations = np.append(
                        inserted_motif_locations,
                        [
                            insertion_locations[best_mutation] + i
                            for i in range(motif_length)
                        ],
                    )

                if return_intermediate:
                    insertion_index = insertion_locations[best_mutation]
                    intermediate_info_list[idx]["changes"].append(
                        (insertion_index, patterns[pattern_name])
                    )
                    intermediate_info_list[idx]["predictions"].append(
                        mutagenesis_predictions[best_mutation]
                    )

        designed_sequence = hot_encoding_to_sequence(sequence_onehot)
        designed_sequences.append(designed_sequence)

        if return_intermediate:
            intermediate_info_list[idx]["designed_sequence"] = designed_sequence

    return (
        (intermediate_info_list, designed_sequences)
        if return_intermediate
        else designed_sequences
    )
