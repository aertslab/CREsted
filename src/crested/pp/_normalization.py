"""Preprocessing normalization functionality for continuous .X data based on gini scores."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import csr_matrix

from ._utils import _calc_gini


def normalize_peaks(
    adata: AnnData,
    peak_threshold: int = 0,
    gini_std_threshold: float = 1.0,
    top_k_percent: float = 0.01,
) -> None:
    """
    Normalize the adata.X based on variability of the top values per cell type.

    This function applies a normalization factor to each cell type,
    focusing on regions with the most significant peaks above
    a defined threshold and considering the variability within those peaks.
    Only used on continuous .X data. Modifies the input AnnData.X in place.

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
    The AnnData object with the normalized matrix and cell type weights used for normalization in the obsm attribute.

    Example
    -------
    >>> crested.pp.normalize_peaks(
    ...     adata,
    ...     peak_threshold=0,
    ...     gini_std_threshold=2.0,
    ...     top_k_percent=0.05,
    ... )
    """
    if isinstance(adata.X, csr_matrix):
        target_matrix = (
            adata.X.toarray().T
        )  # Convert to dense and transpose to (regions, cell types)
    else:
        target_matrix = adata.X.T

    regions_df = adata.var

    top_k_percent_means = []
    all_low_gini_indices = set()
    gini_scores_all = []

    overall_gini_scores = _calc_gini(target_matrix)
    mean = np.mean(np.max(overall_gini_scores, axis=1))
    std_dev = np.std(np.max(overall_gini_scores, axis=1))
    gini_threshold = mean - gini_std_threshold * std_dev

    logger.info("Filtering on top k Gini scores...")
    for i in range(target_matrix.shape[1]):
        filtered_col = target_matrix[:, i][target_matrix[:, i] > peak_threshold]
        sorted_col = np.sort(filtered_col)[::-1]
        top_k_index = int(len(sorted_col) * top_k_percent)

        top_indices = np.argsort(filtered_col)[::-1][:top_k_index]
        gini_scores = _calc_gini(target_matrix[top_indices])
        low_gini_indices = np.where(np.max(gini_scores, axis=1) < gini_threshold)[0]

        if len(low_gini_indices) > 0:
            top_k_mean = np.mean(sorted_col[low_gini_indices])
            gini_scores_all.append(np.max(gini_scores[low_gini_indices], axis=1))
            all_low_gini_indices.update(top_indices[low_gini_indices])
        else:
            top_k_mean = 0
            gini_scores_all.append(0)

        top_k_percent_means.append(top_k_mean)

    max_mean = np.max(top_k_percent_means)
    weights = max_mean / np.array(top_k_percent_means)

    # Add the weights to the AnnData object
    logger.info("Added normalization weights to adata.obsm['weights']...")
    adata.obsm["weights"] = weights

    normalized_matrix = target_matrix * weights

    if isinstance(adata.X, csr_matrix):
        normalized_matrix = csr_matrix(normalized_matrix.T)
    else:
        normalized_matrix = normalized_matrix.T

    filtered_regions_df = regions_df.iloc[list(all_low_gini_indices)]

    adata.X = normalized_matrix

    return filtered_regions_df
