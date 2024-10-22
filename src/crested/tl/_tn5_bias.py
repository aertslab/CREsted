"""Tn5 bias prediction."""

import os

import keras
import numpy as np
from tqdm import tqdm

import crested.utils


def tn5_bias_prediction(
    sequences: list[str],
    model_path: os.PathLike,
    batch_size: int = 32,
    reduction: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Predict Tn5 bias for a list of sequences using :cite:p:`hu2023single`.

    Since the model was trained on 101 bp regions (1 + 2*50 bp context), the input sequences should be at least 101 bp long.
    The returned predictions will omit the first and last 50 bp of each region.

    Parameters
    ----------
    sequences
        List of sequence strings to predict Tn5 bias for.
    model
        Path to Tn5 bias prediction model.
    batch_size
        Number of sequences to process at once.
    reduction
        None, 'mean', 'sum', or 'max'. If None, return the predicted bias for each position in each sequence.
        Else, returns the mean, sum, or max of the predicted bias per sequence.
    verbose
        Whether to display a progress bar for batch processing.

    Returns
    -------
    Array of predicted Tn5 bias for each position in each sequence if reduction is None (n_regions, region_width-100).
    Else, returns the mean, sum, or max of the predicted bias per sequence (n_regions, ).
    """

    def _region_onehot_encode(region_seq: str, context_radius: int) -> np.ndarray:
        context_len = 2 * context_radius + 1
        region_width = len(region_seq) - 2 * context_radius
        region_onehot = crested.utils.one_hot_encode_sequence(
            region_seq, expand_dim=False
        )
        # Here we extract the region-wise one-hots
        region_onehots = np.array(
            [region_onehot[i : (i + context_len), :] for i in range(region_width)]
        )
        return region_onehots

    context_radius = 50

    assert len(sequences) > 0, "No sequences provided for Tn5 bias prediction."
    assert os.path.exists(model_path), f"Model file not found at {model_path}."
    assert isinstance(sequences, list), "Input sequences must be a list of strings."

    model = keras.models.load_model(model_path, compile=False)

    all_preds = []
    num_sequences = len(sequences)
    num_batches = (
        num_sequences + batch_size - 1
    ) // batch_size  # calculate total number of batches

    # Set up progress bar if verbose is enabled
    batch_iterator = range(num_batches)
    if verbose:
        batch_iterator = tqdm(batch_iterator, desc="Predicting batches", unit="batch")

    for batch_idx in batch_iterator:
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_sequences)
        batch_sequences = sequences[batch_start:batch_end]

        # Batch one-hot encoding for all sequences in the batch
        region_onehots = []
        for seq in batch_sequences:
            onehot_encoded = _region_onehot_encode(seq, context_radius)
            region_onehots.append(onehot_encoded)

        # Convert to 4D array (batch_size, num_positions, context_len, 4)
        region_onehots = np.concatenate(region_onehots, axis=0)

        # Model prediction on the batch of one-hots (predict on all one-hots at once)
        batch_preds = model.predict(region_onehots, verbose=0)
        batch_preds = batch_preds.reshape(len(batch_sequences), -1)
        batch_preds = np.power(10, (batch_preds - 0.5) * 2) - 0.01

        if reduction == "mean":
            batch_preds = np.mean(batch_preds, axis=1)
        elif reduction == "sum":
            batch_preds = np.sum(batch_preds, axis=1)
        elif reduction == "max":
            batch_preds = np.max(batch_preds, axis=1)
        elif reduction is not None:
            raise ValueError(
                f"Invalid reduction method: {reduction}. Must be 'mean' or 'sum'."
            )

        all_preds.append(batch_preds)

    # Concatenate all the batch predictions
    preds = np.concatenate(all_preds, axis=0)

    return preds
