from __future__ import annotations

import numpy as np
import torch
from tangermeme.tools import tomtom as ttomtom


def run_pairwise_tomtom(list_patterns: list[np.ndarray]) -> tuple(
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
):
    """
    Runs pair-wise tomtom for list of PWMs

    Parameters
    ----------
    list_patterns:
    list of PWMs each stored as ndarray (have to be of the same size, add 0-padding if needed)

    Returns
    -------
    p-values, scores, offsets, overlaps, strands, e-values
    """
    n_patterns = len(list_patterns)
    # convert ndarrays to pytorch tensor
    list_pat_tensors = [torch.from_numpy(pat) for pat in list_patterns]

    pvals, scores, offsets, overlaps, strands = ttomtom.tomtom(
        list_pat_tensors, list_pat_tensors
    )

    # compute e-values
    evals = pvals.numpy() * n_patterns

    return (
        pvals.numpy(),
        scores.numpy(),
        offsets.numpy(),
        overlaps.numpy(),
        strands.numpy(),
        evals,
    )


def add_padding_to_matrix(matrix: np.array, desired_size: int) -> np.array:
    diffSize = desired_size - matrix.shape[0]

    if diffSize < 0:
        raise ValueError("ERROR! Matrix is larger than desired size.")
    elif diffSize > 0:
        mat = np.concatenate(
            (matrix, np.zeros((diffSize, matrix.shape[1]), dtype=np.float32)), axis=0
        )
    else:
        mat = matrix

    return mat
