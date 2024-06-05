"""Preprocessing functionalities for peak data."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix


def _calc_gini(targets: np.ndarray) -> np.ndarray:
    """Returns gini scores for the given targets"""

    def _gini(array):
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


def normalize_peaks(
    adata: AnnData,
    peak_threshold: int = 0,
    gini_std_threshold: float = 1.0,
    top_k_percent: float = 0.01,
) -> np.ndarray:
    """
    Normalize the peaks in adata.X based on variability of the top values per cell type.

    This function applies a normalization factor to each cell type,
    focusing on regions with the most significant peaks above
    a defined threshold and considering the variability within those peaks.
    Only used for peak regression tasks.

    Example
    -------
    >>> normalized_adata = normalize_peaks(
    ...     adata,
    ...     peak_threshold=0,
    ...     gini_std_threshold=2.0,
    ...     top_k_percent=0.05,
    ... )

    Parameters
    ----------
    adata
        The AnnData object containing the matrix (celltypes, regions) to be normalized.
    peak_threshold
        The minimum value for a peak to be considered significant for
        the Gini score calculation.
    gini_std_threshold
        The number of standard deviations below the mean Gini score used to determine
        the threshold for low variability.
    top_k_percent
        The percentage (expressed as a fraction) of top values
        to consider for Gini score calculation.

    Returns
    -------
    anndata.AnnData
        The AnnData object with the normalized matrix and cell
        type weights used for normalization in the obsm attribute.
    """
    if isinstance(adata.X, csr_matrix):
        target_matrix = (
            adata.X.toarray().T
        )  # Convert to dense and transpose to (regions, cell types)
    else:
        target_matrix = adata.X.T

    top_k_percent_means = []
    gini_scores_all = []

    print("Filtering on top k Gini scores...")
    for i in range(target_matrix.shape[1]):
        filtered_col = target_matrix[:, i][target_matrix[:, i] > peak_threshold]
        sorted_col = np.sort(filtered_col)[::-1]
        top_k_index = int(len(sorted_col) * top_k_percent)

        gini_scores = _calc_gini(
            target_matrix[np.argsort(filtered_col)[::-1][:top_k_index]]
        )
        mean = np.mean(np.max(gini_scores, axis=1))
        std_dev = np.std(np.max(gini_scores, axis=1))
        gini_threshold = mean - gini_std_threshold * std_dev
        low_gini_indices = np.where(np.max(gini_scores, axis=1) < gini_threshold)[0]
        print(f"{len(low_gini_indices)} out of {top_k_index} remain for cell type {i}.")

        if len(low_gini_indices) > 0:
            top_k_mean = np.mean(sorted_col[low_gini_indices])
            gini_scores_all.append(np.max(gini_scores[low_gini_indices], axis=1))
        else:
            top_k_mean = 0
            gini_scores_all.append(0)

        top_k_percent_means.append(top_k_mean)

    max_mean = np.max(top_k_percent_means)
    weights = max_mean / np.array(top_k_percent_means)

    # Add the weights to the AnnData object
    adata.obsm["weights"] = weights

    normalized_matrix = target_matrix * weights
    return normalized_matrix, weights
