"""Preprocessing normalization functionality for continuous .X data based on gini scores."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger
from pandas import DataFrame
from scipy.sparse import csr_matrix

from ._utils import _calc_gini


def normalize_peaks(
    adata: AnnData,
    peak_threshold: int = 0,
    gini_std_threshold: float = 1.0,
    top_k_percent: float = 0.01,
    inplace: bool = True
) -> DataFrame | (AnnData | DataFrame):
    """
    Normalize the adata.X based on variability of the top values per cell type.

    This function applies a normalization factor to each cell type,
    focusing on regions with the most significant peaks above
    a defined threshold and considering the variability within those peaks.
    Only used on continuous .X data. Modifies the input AnnData.X in place if `inplace=True`.

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
    inplace
        Perform computation and modify `adata` in-place or return a resulting copy of the `adata` instead.

    Returns
    -------
    If `inplace=True` (default), modifies the AnnData in-place with the normalized matrix and normalization weights saved to `adata.obsm['weights']`, and returns the filtered .var of the significant peaks, as a DataFrame.
    If `inplace=False`, returns (adata, filtered_df): a modified copy of the AnnData object instead, along with a the filtered .var of the significant peaks, as a DataFrame.

    See Also
    --------
    crested.pl.qc.normalization_weights

    Example
    -------
    >>> crested.pp.normalize_peaks(
    ...     adata,
    ...     peak_threshold=0,
    ...     gini_std_threshold=2.0,
    ...     top_k_percent=0.05,
    ... )
    """
    if "weights" in adata.obsm:
        raise ValueError("Your data is already peak-normalized ('weights' already in adata.obsm).")

    if isinstance(adata.X, csr_matrix):
        target_matrix = (
            adata.X.toarray().T
        )  # Convert to dense and transpose to (regions, cell types)
    else:
        target_matrix = adata.X.T

    regions_df = adata.var

    top_k_percent_means = []
    all_low_gini_indices = set()
    failed_classes = []

    overall_gini_scores = np.max(_calc_gini(target_matrix), axis=1)
    mean = np.mean(overall_gini_scores)
    std_dev = np.std(overall_gini_scores)
    gini_threshold = mean - gini_std_threshold * std_dev

    logger.info("Filtering on top k Gini scores...")
    for i in range(target_matrix.shape[1]):
        # Apply peak_threshold: minimum peak height filtering for this cell type
        filtered_indices = np.where(target_matrix[:, i] > peak_threshold)[0]
        filtered_col = target_matrix[filtered_indices, i]

        # Apply top_k_percent: Get top k of the values that pass threshold
        top_k_index = int(len(filtered_indices) * top_k_percent)
        sorted_filtered_indices = np.argsort(filtered_col)[::-1]
        top_indices = filtered_indices[sorted_filtered_indices[:top_k_index]]
        top_col = target_matrix[:, i][top_indices]

        # Apply gini_threshold: Retrieve gini scores for the regions passing previous filters and keep those below threshold
        gini_scores = overall_gini_scores[top_indices]
        low_gini_mask = gini_scores < gini_threshold

        # Save final scores
        if low_gini_mask.sum() > 0:
            top_k_mean = np.mean(top_col[low_gini_mask])
            all_low_gini_indices.update(top_indices[low_gini_mask])
            top_k_percent_means.append(top_k_mean)
        else:
            # No regions left after all filtering, saving cell type name to raise later
            failed_classes.append(adata.obs_names[i])

    # Raise error if no selected peaks found (since otherwise it'd create divide by zero and create NaNs in .X)
    if len(failed_classes) > 0:
        classes_to_show = ", ".join(failed_classes[:10]) + (f", ... ({len(failed_classes)} total)" if len(failed_classes) > 10 else "")
        raise ValueError(
            f"No peaks passed the top-k and Gini selection for {classes_to_show}, so their "
            "normalization weight is undefined. Lower gini_std_threshold to widen what "
            "counts as a broad peak, raise top_k_percent to select more peaks, or drop "
            "these cell types from the AnnData."
        )

    top_k_percent_means = np.array(top_k_percent_means)
    max_mean = np.max(top_k_percent_means)
    weights = max_mean / top_k_percent_means
    normalized_matrix = target_matrix * weights

    if isinstance(adata.X, csr_matrix):
        normalized_matrix = csr_matrix(normalized_matrix.T)
    else:
        normalized_matrix = normalized_matrix.T

    filtered_regions_df = regions_df.iloc[list(all_low_gini_indices)]

    # Modify the adata
    if not inplace:
        adata = adata.copy()
    logger.info("Added normalization weights to adata.obsm['weights']...")
    adata.obsm["weights"] = weights
    adata.X = normalized_matrix
    if inplace:
        return filtered_regions_df
    else:
        return adata, filtered_regions_df
