"""Tn5 bias prediction."""

import os

import keras
import numpy as np

import crested.utils


def tn5_bias_prediction(
    sequences: list[str], model_path: os.PathLike, reduction: str | None = None
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
    reduction
        None, 'mean', 'sum', or 'max'. If None, return the predicted bias for each position in each sequence.
        Else, returns the mean, sum, or max of the predicted bias per sequence.

    Returns
    -------
    Array of predicted Tn5 bias for each position in each sequence if reduction is None (n_regions, region_width-100).
    Else, returns the mean, sum, or max of the predicted bias per sequence (n_regions, ).

    Examples
    --------
    >>> tn5_bias_model = crested.get_model("tn5_bias")
    >>> regions = ["chr1:1000000-1000200", "chr1:1000200-1000400"]
    >>> region_seqs = crested.utils.fetch_sequences(regions, genome_path)
    >>> predicted_biases = crested.tl.tn5_bias_prediction(region_seqs, model)
    >>> print(predicted_biases.shape)
    (2, 100)
    """

    def _region_onehot_encode(region_seq: str, context_radius: int) -> np.ndarray:
        context_len = 2 * context_radius + 1
        region_width = len(region_seq) - 2 * context_radius
        region_onehot = crested.utils.one_hot_encode_sequence(
            region_seq, expand_dim=False
        )
        region_onehot = np.array(
            [region_onehot[i : (i + context_len), :] for i in range(region_width)]
        )
        return region_onehot

    context_radius = 50
    context_len = 2 * context_radius + 1

    assert len(sequences) > 0, "No sequences provided for Tn5 bias prediction."
    assert os.path.exists(model_path), f"Model file not found at {model_path}."
    assert isinstance(sequences, list), "Input sequences must be a list of strings."

    model = keras.models.load_model(model_path, compile=False)

    region_onehots = []
    for seq_index, seq in enumerate(sequences):
        if "N" in seq:
            print(
                f"Region  {seq_index} contains N's. Bias prediction will be inaccurate."
            )
        assert (
            len(seq) >= context_len
        ), f"Region sequence {seq_index} is too short. Must be at least {context_len} bp."
        assert len(seq) == len(
            sequences[0]
        ), "All sequences must be of the same length."
        region_onehots.append(_region_onehot_encode(seq, context_radius))

    preds = np.concatenate([model.predict(onehot) for onehot in region_onehots])
    preds = preds.reshape(len(sequences), -1)
    preds = np.power(10, (preds - 0.5) * 2) - 0.01
    if reduction == "mean":
        preds = np.mean(preds, axis=1)
    elif reduction == "sum":
        preds = np.sum(preds, axis=1)
    elif reduction == "max":
        preds = np.max(preds, axis=1)
    elif reduction is not None:
        raise ValueError(
            f"Invalid reduction method: {reduction}. Must be 'mean' or 'sum'."
        )

    return preds
