from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import csr_matrix

from ._utils import _calc_gini, _calc_proportion


def filter_regions_on_specificity(
    adata: AnnData,
    gini_std_threshold: float = 1.0,
    model_name: str | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """
    Filter bed regions & targets/predictions based on high Gini score.

    This function filters regions based on their specificity using Gini scores.
    The regions with high Gini scores are retained.
    If model_name is provided, will look for the corresponding predictions in the
    adata.layers[model_name] layer. Else, it will use the values in adata.X to decide
    which regions to keep.
    To get an idea for the impact of different possible `gini_std_threshold` values, see :func:`~crested.pl.qc.filter_cutoff`.

    Parameters
    ----------
    adata
        The AnnData object containing the matrix (celltypes, regions) to be filtered.
    gini_std_threshold
        The number of standard deviations above the mean Gini score used to determine the threshold for high variability.
    model_name
        The name of the model to look for in adata.layers[model_name] for predictions.
        If None or 'truth'/'groundtruth'/'X', will use the values in adata.X to select specific regions.
    inplace
        Perform computation and modify `adata` in-place or return a resulting copy of the `adata` instead.

    Returns
    -------
    If `inplace=True` (default), returns nothing and modifies the AnnData in-place with the filtered matrix and updated variable names.
    If `inplace=False`, returns a modified copy of the AnnData object instead.

    See Also
    --------
    crested.pl.qc.filter_cutoff

    Example
    -------
    >>> crested.pp.filter_regions_on_specificity(
    ...     adata,
    ...     gini_std_threshold=1.0,
    ... )
    """
    if model_name is None or model_name in ['X', 'truth', 'groundtruth']:
        if isinstance(adata.X, csr_matrix):
            target_matrix = (
                adata.X.toarray().T
            )  # Convert to dense and transpose to (regions, cell types)
        else:
            target_matrix = adata.X.T
    else:
        if model_name not in adata.layers:
            raise ValueError(
                f"Model name {model_name} not found in adata.layers. Please provide a valid model name."
            )
        target_matrix = adata.layers[model_name].T

    gini_scores = np.max(_calc_gini(target_matrix), axis=1)
    mean = np.mean(gini_scores)
    std_dev = np.std(gini_scores)
    gini_threshold = mean + gini_std_threshold * std_dev
    selected_indices = np.argwhere(gini_scores > gini_threshold)[:, 0]

    target_matrix_filt = target_matrix[selected_indices]
    regions_filt = adata.var_names[selected_indices]

    logger.info(
        f"After specificity filtering, kept {len(target_matrix_filt)} out of {target_matrix.shape[0]} regions."
    )

    # Filter the adata object inplace or return copy
    if inplace:
        adata._inplace_subset_var(regions_filt)
    else:
        return adata[:, selected_indices].copy()


def sort_and_filter_regions_on_specificity(
    adata: AnnData,
    top_k: int,
    model_name: str | None = None,
    method: str = "gini",
    inplace: bool = True,
) -> AnnData | None:
    """
    Sort bed regions & targets/predictions based on high Gini or proportion score per colum while keeping the top k rows per column.

    Combines them into a single AnnData object with extra columns indicating the original class name,
    the rank per column, and the score.
    To get an idea for the impact of different possible `top_k` values, see :func:`~crested.pl.qc.sort_and_filter_cutoff`.

    Parameters
    ----------
    adata
        The AnnData object containing the matrix (celltypes, regions) to be sorted.
    top_k
        The number of top regions to keep per column.
    model_name
        The name of the model to look for in adata.layers[model_name] for predictions.
        If None, will use the targets in adata.X to decide which regions to sort.
    method
        The method to use for calculating scores, either 'gini' or 'proportion'.
        Default is 'gini'.
    inplace
        Perform computation and modify `adata` in-place or return a resulting copy of the `adata` instead.

    Returns
    -------
    If `inplace=True` (default), returns nothing and modifies the AnnData in-place with the sorted and filtered matrix, and extra columns
    indicating the original class name, the rank per column, and the score.
    If `inplace=False`, returns a modified copy of the AnnData object instead.

    See Also
    --------
    crested.pl.qc.sort_and_filter_cutoff

    Example
    -------
    >>> crested.pp.sort_and_filter_regions_on_specificity(
    ...     adata,
    ...     top_k=500,
    ...     method="gini",
    ... )
    """
    class_names = list(adata.obs_names)
    if model_name is None or model_name in ['X', 'truth', 'groundtruth']:
        if isinstance(adata.X, csr_matrix):
            target_matrix = (
                adata.X.toarray().T
            )  # Convert to dense and transpose to (regions, cell types)
        else:
            target_matrix = adata.X.T
    else:
        if model_name not in adata.layers:
            raise ValueError(
                f"Model name {model_name} not found in adata.layers. Please provide a valid model name."
            )
        target_matrix = adata.layers[model_name].T

    if method == "gini":
        scores = _calc_gini(target_matrix)
    elif method == "proportion":
        scores = _calc_proportion(target_matrix)
    else:
        raise ValueError("Method must be either 'gini' or 'proportion'.")

    all_selected_indices = []
    column_indices = []
    ranks = []
    all_scores = []
    for col in range(scores.shape[1]):
        sorted_indices = np.argsort(scores[:, col])[::-1]
        top_indices = sorted_indices[:top_k]
        all_selected_indices.extend(top_indices)
        column_indices.extend([col] * len(top_indices))
        ranks.extend(range(1, len(top_indices) + 1))
        all_scores.extend(scores[top_indices, col])

    # Convert to numpy array for indexing
    all_selected_indices = np.array(all_selected_indices)
    column_indices = np.array(column_indices)
    ranks = np.array(ranks)
    all_scores = np.array(all_scores)

    target_matrix_filtered = target_matrix[all_selected_indices]
    regions_filtered = adata.var_names[all_selected_indices]
    class_names_filtered = [class_names[idx] for idx in column_indices]

    logger.info(
        f"After sorting and filtering, kept {target_matrix_filtered.shape[0]} regions."
    )

    # filter the adata object (inplace or via a copy)
    if inplace:
        adata._inplace_subset_var(regions_filtered)
    else:
        adata = adata[:, all_selected_indices].copy()
    adata.var["Class name"] = class_names_filtered
    adata.var["rank"] = ranks
    adata.var[f"{method}_score"] = all_scores
    if not inplace:
        return adata
