"""Tooling kit that handles predictions, contribution scores, ... ."""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Sequence
from typing import Literal

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from tqdm import tqdm

from crested._genome import Genome
from crested.tl import TaskConfig
from crested.tl._explainer import (
    integrated_grad,
    mutagenesis,
    saliency_map,
    window_shuffle,
)
from crested.utils._logging import log_and_raise
from crested.utils._utils import _transform_input


def extract_layer_embeddings(
    input: str | list[str] | np.ndarray | AnnData,
    model: keras.Model,
    layer_name: str,
    genome: Genome | str | os.PathLike | None = None,
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
    genome: Genome | str | os.PathLike | None = None,
    batch_size: int = 128,
    **kwargs,
) -> None | np.ndarray:
    """
    Make predictions using the model(s) on some input that represents sequences.

    If a list of models is provided, the predictions will be averaged across all models.
    Predictions can be visualized with the functions in :mod:`~crested.pl.region`.

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


def evaluate(
    adata: AnnData,
    model: keras.Model | list[keras.Model] | str,
    metrics: TaskConfig | list[keras.metrics.Metric | keras.losses.Loss]| None = None,
    split: str | None = "test",
    return_metrics: bool = False,
    **kwargs
):
    """
    Calculate metrics on the test set.

    If a list of models is provided, the predictions will be averaged across all models.

    Parameters
    ----------
    adata
        The AnnData to retrieve ground truth and region info from. Must have 'test' in `adata.var['split']`.
    model
        A (list of) trained keras model(s) to make predictions with, or a name of a saved prediction layer.
    metrics
        A {func}`~crested.tl.TaskConfig` object, a list of keras metrics and/or losses, or None (in which case it will try to use the metrics compiled with the model).
    split
        Which split to evaluate. Must be one of the values encoded in adata.var['split'], or None to evaluate the entire dataset.
    return_metrics
        Whether to return a dict of the results.
    kwargs
        Arguments passed on to {func}`~crested.tl.predict`, like `batch_size` or `genome`.

    Returns
    -------
    If `return_metrics=True`, a dict with of shape {metric_name: metric_value, ...}.

    Example
    -------
    >>> crested.tl.test(adata, model, config)
    """
    # Predict or extract values
    if split is not None and split not in adata.var['split'].values:
        raise ValueError(f"split {split} must be in adata.var['split'] or be None.")
    adata_split = adata[:, adata.var['split'] == split] if split is not None else adata
    truth = adata_split.X.T
    if isinstance(model, str):
        if model not in adata.layers:
            raise ValueError(f"model name {model} must be in adata.layers if providing a string.")
        preds = adata_split.layers[model].T
    else:
        preds = predict(adata_split, model, **kwargs)

    # If model is compiled and config is None, use metrics from there
    metrics_results = {}
    if metrics is None:
        if not isinstance(model, str) and not model.compiled:
            raise ValueError("Model must be compiled if not providing a config or list of metrics.")
        metrics_results['loss'] = model.metrics[0](truth, preds).numpy()
        for metric_name, metric_value in model.metrics[1](truth, preds).items():
            metrics_results[metric_name] = metric_value.numpy()
    # else, use provided config or list of metrics
    else:
        if not isinstance(model, str) and model.compiled:
            logger.warning("Ignoring metrics from the compiled model in favor of provided metrics.")
        if isinstance(metrics, TaskConfig):
            metrics = [metrics.loss, *metrics.metrics]
        # Calculate metrics
        for metric in metrics:
            try:
                metric.reset_states()
            except AttributeError:
                pass
            metrics_results[metric.name] = metric(truth, preds).numpy()

    # Print results
    for metric_name, metric_result in metrics_results.items():
        logger.info(f"Test {metric_name}: {metric_result:.4f}")
    gc.collect()
    if return_metrics:
        return metrics_results


def score_gene_locus(
    chr_name: str,
    gene_start: int,
    gene_end: int,
    target_idx: int,
    model: keras.Model | list[keras.Model],
    genome: Genome | str | os.PathLike | None = None,
    strand: str = "+",
    upstream: int = 50000,
    downstream: int = 10000,
    central_size: int = 1000,
    step_size: int = 50,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Score regions upstream and downstream of a gene locus using the model's prediction.

    The model predicts a value for the {central_size} of each window. These scores can be visualized with :func:`~crested.pl.locus.locus_scoring`.

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
    crested.pl.locus.locus_scoring
    """
    # Detect window size from the model input shape
    if not isinstance(target_idx, int):
        raise ValueError("Target index must be an integer.")
    if isinstance(model, Sequence):
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
    for _, (pos, pred) in enumerate(zip(positions, predictions_class, strict=False)):
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
        list(zip([chr_name] * len(positions), window_starts, window_ends, strict=False))
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
    method: Literal['integrated_grad', 'mutagenesis', 'expected_integrated_grad', 'saliency_map', 'window_shuffle', 'window_shuffle_uniform'] = "expected_integrated_grad",
    window_size: int | None = 7,
    n_shuffles: int | None = 24,
    genome: Genome | str | os.PathLike | None = None,
    transpose: bool = False,
    all_class_names: list[str] | None = None,
    batch_size: int = 128,
    output_dir: str | os.PathLike | None = None,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method for the specified inputs.

    If multiple models are provided, the contribution scores will be averaged across all models.

    These scores can then be plotted to visualize the importance of each base in the sequence
    using :func:`~crested.pl.explain.contribution_scores`.

    Parameters
    ----------
    input
        Input data to calculate the contribution scores for. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
        If the input regions are stranded, will return the sequence from the proper strand; i.e. chrI:0-100:- will return a (complement) sequence from 100 to 0, as it's reversed compared to the positive strand.
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
    crested.pl.explain.contribution_scores
    """

    @log_and_raise(ValueError)
    def _check_input_params(
        target_idx: Sequence[int]
    ):
        """Check contribution scores parameters."""
        if isinstance(target_idx, str):
            raise ValueError(
                "target_idx must be an integer, list of integers, None, or empty list, but not a string. "
                "You can get this index with `list(anndata.obs_names).index(class_name)` or `list(adata.obs_names.get_indexer(classes_of_interest))`"
            )
        if not isinstance(target_idx[0], (int, np.integer)) and target_idx[0] is not None:
            raise ValueError(f"target_idx must be an integer, list of integers, None, or empty list, not {type(target_idx[0])}")

    # Infer/repackage input values
    if not isinstance(model, Sequence):
        model = [model]

    if target_idx is None:
        target_idx = list(range(0, model[0].output_shape[-1]))
    if not isinstance(target_idx, Sequence):
        target_idx = [target_idx]
    if target_idx == []:
        if verbose:
            logger.info(
                "No class indices provided. Calculating contribution scores for the 'combined' class."
            )
        target_idx = [None]

    # Check inputs for correctness
    _check_input_params(target_idx=target_idx)

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
    genome: Genome | str | os.PathLike | None = None,
    method: str = "expected_integrated_grad",
    transpose: bool = True,
    batch_size: int = 128,
    output_dir: str | os.PathLike | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate contribution scores based on given method only for the most specific regions per class.

    Contrary to :func:`~crested.tl.contribution_scores`, this function will only calculate one set of contribution scores per region per class.
    Expects the user to have ran :func:`~crested.pp.sort_and_filter_regions_on_specificity` beforehand.

    If multiple models are provided, the contribution scores will be averaged across all models.

    These scores can then be plotted to visualize the importance of each base in the sequence
    using :func:`~crested.pl.explain.contribution_scores`.

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
    crested.pp.sort_and_filter_regions_on_specificity
    crested.pl.explain.contribution_scores
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
    if not isinstance(target_idx, Sequence):
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
