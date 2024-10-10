"""Utility functions for the PP module."""

from __future__ import annotations

import numpy as np


def _calc_gini(targets: np.ndarray) -> np.ndarray:
    """
    Return Gini scores for the given targets.

    This function calculates the Gini coefficient for each row in the targets array
    and assigns the score to the maximum value's position in the corresponding row
    of the Gini scores array.

    Parameters
    ----------
    targets
        A 2D numpy array where each row represents a set of target values.

    Returns
    -------
    gini scores
        A 2D numpy array with the same shape as `targets` containing Gini scores,
        where each score is assigned to the position of the maximum value in each row.
    """

    def _gini(array: np.ndarray) -> float:
        """Calculate the Gini coefficient of a numpy array."""
        array = (
            array.flatten().clip(0, None) + 0.0000001
        )  # Ensure non-negative values and avoid zero
        array = np.sort(array)
        index = np.arange(1, array.size + 1)
        return (np.sum((2 * index - array.size - 1) * array)) / (
            array.size * np.sum(array)
        )

    gini_scores = np.zeros_like(targets)

    for region_idx in range(targets.shape[0]):
        region_scores = targets[region_idx]
        max_idx = np.argmax(region_scores)
        gini_scores[region_idx, max_idx] = _gini(region_scores)

    return gini_scores


def _calc_proportion(arr: np.ndarray, scale=False):
    """
    Compute relative specificity scores for a given 1D or 2D array.

    This function calculates the proportion of each element relative to the sum of all elements
    in the array (for 1D arrays) or relative to the sum of elements in each row (for 2D arrays).
    Optionally, it scales the specificity scores by multiplying each element by itself.

    Parameters
    ----------
    arr
        Input array (1D or 2D).
    scale
        Whether to scale the specificity scores by multiplying with orginal array.

    Returns
    -------
    specificity_scores : np.ndarray
        The calculated specificity scores.

    Raises
    ------
    ValueError
        If the input array is not 1D or 2D.
    """
    if arr.ndim == 1:
        total = np.sum(arr)
        if total == 0:
            total = 1e-9
        specificity_scores = (arr / total) * arr if scale else arr / total
    elif arr.ndim == 2:
        total_per_row = np.sum(arr, axis=1, keepdims=True)
        total_per_row[total_per_row == 0] = 1e-9
        specificity_scores = (
            (arr / total_per_row) * arr if scale else arr / total_per_row
        )
    else:
        raise ValueError("Input array must be 1D or 2D.")

    return specificity_scores
