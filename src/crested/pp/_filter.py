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
) -> AnnData:
    """
    Filter bed regions & targets/predictions based on high Gini score.

    This function filters regions based on their specificity using Gini scores.
    The regions with high Gini scores are retained, and a new AnnData object
    is created with the filtered data.
    If model_name is provided, will look for the corresponding predictions in the
    adata.layers[model_name] layer. Else, it will use the targets in adata.X to decide
    which regions to keep.

    Parameters
    ----------
    adata
        The AnnData object containing the matrix (celltypes, regions) to be filtered.
    gini_std_threshold
        The number of standard deviations above the mean Gini score used to determine
        the threshold for high variability.
    model_name
        The name of the model to look for in adata.layers[model_name] for predictions.
        If None, will use the targets in adata.X to select specific regions.

    Returns
    -------
    A new AnnData object with the filtered matrix and updated variable names.


    Example
    -------
    >>> filtered_adata = crested.pp.filter_regions_on_specificity(
    ...     adata,
    ...     gini_std_threshold=1.0,
    ... )
    """
    if model_name is None:
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
    if model_name is None:
        if isinstance(adata.X, csr_matrix):
            new_X = csr_matrix(target_matrix_filt.T)
        else:
            new_X = target_matrix_filt.T
    else:
        if isinstance(adata.X, csr_matrix):
            new_X = csr_matrix(adata.X[:, selected_indices])
        else:
            new_X = adata.X[:, selected_indices]
        new_pred_matrix = target_matrix_filt.T

    filtered_adata = AnnData(new_X)
    filtered_adata.obs = adata.obs.copy()
    filtered_adata.var = adata.var.iloc[selected_indices].copy()
    filtered_adata.var_names = regions_filt

    # Copy over any other attributes or layers if needed
    filtered_adata.obsm = adata.obsm.copy()

    if model_name is not None:
        filtered_adata.layers[model_name] = new_pred_matrix

    return filtered_adata

def sort_and_filter_regions_on_specificity(
    adata: AnnData,
    top_k: int,
    class_names: list,
    model_name: str | None = None,
    method: str = 'gini'
) -> AnnData:
    """
    Sort bed regions & targets/predictions based on high Gini or proportion score per column, keep the top k rows per column,
    and combine them into a single AnnData object with extra columns indicating the original class name,
    the rank per column, and the score.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the matrix (celltypes, regions) to be sorted.
    top_k : int
        The number of top regions to keep per column.
    class_names : list
        List of class names corresponding to the original column indices.
    model_name : str | None, optional
        The name of the model to look for in adata.layers[model_name] for predictions.
        If None, will use the targets in adata.X to decide which regions to sort.
    method : str, optional
        The method to use for calculating scores, either 'gini' or 'proportion'.
        Default is 'gini'.

    Returns
    -------
    AnnData
        A new AnnData object with the sorted and filtered matrix, and extra columns indicating the original class name,
        the rank per column, and the score.

    Example
    -------
    >>> sorted_filtered_adata = sort_and_filter_regions_on_specificity(
    ...     adata,
    ...     top_k=500,
    ...     class_names=["Class1", "Class2", "Class3"],
    ...     method='proportion'
    ... )
    """
    if model_name is None:
        if isinstance(adata.X, csr_matrix):
            target_matrix = adata.X.toarray().T  # Convert to dense and transpose to (regions, cell types)
        else:
            target_matrix = adata.X.T
    else:
        if model_name not in adata.layers:
            raise ValueError(f"Model name {model_name} not found in adata.layers. Please provide a valid model name.")
        target_matrix = adata.layers[model_name].T

    if method == 'gini':
        scores = _calc_gini(target_matrix)
    elif method == 'proportion':
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
    regions_filtered = adata.var_names[all_selected_indices].tolist()

    class_names_filtered = [class_names[idx] for idx in column_indices]

    logger.info(f"After sorting and filtering, kept {target_matrix_filtered.shape[0]} regions.")

    # Create a new AnnData object with the filtered data
    if isinstance(adata.X, csr_matrix):
        new_X = csr_matrix(target_matrix_filtered.T)
    else:
        new_X = target_matrix_filtered.T

    filtered_adata = AnnData(new_X)
    filtered_adata.obs = adata.obs.copy()
    filtered_adata.var = adata.var.iloc[all_selected_indices].copy()
    filtered_adata.var_names = regions_filtered
    filtered_adata.var["Class name"] = class_names_filtered
    filtered_adata.var["rank"] = ranks
    filtered_adata.var[f"{method}_score"] = all_scores

    # Copy over any other attributes or layers if needed
    filtered_adata.obsm = adata.obsm.copy()

    if model_name is not None:
        filtered_adata.layers[model_name] = new_X

    return filtered_adata
