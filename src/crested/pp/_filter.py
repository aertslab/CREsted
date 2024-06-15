from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import csr_matrix

from ._utils import _calc_gini


def filter_regions_on_specificity(
    adata: AnnData,
    gini_std_threshold: float = 1.0,
) -> AnnData:
    """
    Filter bed regions & targets based on high Gini score.

    This function filters regions based on their specificity using Gini scores.
    The regions with high Gini scores are retained, and a new AnnData object
    is created with the filtered data.

    Example
    -------
    >>> filtered_adata = crested.pp.filter_regions_on_specificity(
    ...     adata,
    ...     gini_std_threshold=1.0,
    ... )

    Parameters
    ----------
    adata
        The AnnData object containing the matrix (celltypes, regions) to be filtered.
    gini_std_threshold
        The number of standard deviations above the mean Gini score used to determine
        the threshold for high variability.

    Returns
    -------
    ad.AnnData
        A new AnnData object with the filtered matrix and updated variable names.
    """
    if isinstance(adata.X, csr_matrix):
        target_matrix = (
            adata.X.toarray().T
        )  # Convert to dense and transpose to (regions, cell types)
    else:
        target_matrix = adata.X.T

    gini_scores = _calc_gini(target_matrix)
    mean = np.mean(np.max(gini_scores, axis=1))
    std_dev = np.std(np.max(gini_scores, axis=1))
    gini_threshold = mean + gini_std_threshold * std_dev
    selected_indices = np.argwhere(np.max(gini_scores, axis=1) > gini_threshold)[:, 0]

    target_matrix_filt = target_matrix[selected_indices]
    regions_filt = adata.var_names[selected_indices].tolist()

    logger.info(
        f"After specificity filtering, kept {len(target_matrix_filt)} out of {target_matrix.shape[0]} regions."
    )

    # Create a new AnnData object with the filtered data
    if isinstance(adata.X, csr_matrix):
        new_X = csr_matrix(target_matrix_filt.T)
    else:
        new_X = target_matrix_filt.T

    filtered_adata = AnnData(new_X)
    filtered_adata.obs = adata.obs.copy()
    filtered_adata.var = adata.var.iloc[selected_indices].copy()
    filtered_adata.var_names = regions_filt

    # Copy over any other attributes or layers if needed
    filtered_adata.obsm = adata.obsm.copy()

    return filtered_adata
