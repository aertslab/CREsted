"""Heatmap plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hc
import seaborn as sns
from anndata import AnnData
from loguru import logger
from scipy.spatial.distance import pdist

from crested.pl._utils import render_plot
from crested.utils._logging import log_and_raise


def _generate_heatmap(correlation_matrix, classes, vmin, vmax):
    fig, ax = plt.subplots()
    sns.heatmap(
        correlation_matrix,
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=classes,
        yticklabels=classes,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    return fig


def correlations_self(
    adata: AnnData,
    log_transform: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    reorder: bool = False,
    **kwargs,
):
    """
    Plot self correlation heatmaps of ground truth for different cell types.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    model_names
        List of model names to plot for predictions heatmap. Default is to plot all models in `adata.layers`.
    log_transform
        Whether to log-transform the data before plotting.
    vmin
        Minimum value for heatmap color scale.
    vmax
        Maximum value for heatmap color scale.
    reorder
        Whether or not to order the clases by similarity (boolean).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Examples
    --------
    >>> crested.pl.heatmap.correlations_self(
    ...     adata, log_transform=True, title="Self correlations heatmap"
    ... )

    .. image:: ../../../../docs/_static/img/examples/heatmap_self_correlations.png
    """
    x = adata.X
    classes = list(adata.obs_names)

    if log_transform:
        x = np.log1p(x)

    correlation_matrix = np.corrcoef(x)

    # Reorder the rows/columns to group related classes together
    if reorder:
        D = pdist(correlation_matrix, "correlation")
        Z = hc.linkage(D, "complete", optimal_ordering=True)
        ordering = hc.leaves_list(Z)
        correlation_matrix = correlation_matrix[ordering, :][:, ordering]
        classes = np.array(classes)[ordering]

    fig = _generate_heatmap(correlation_matrix, classes, vmin, vmax)

    return render_plot(fig, **kwargs)


def correlations_predictions(
    adata: AnnData,
    model_names: list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    reorder: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Plot correlation heatmaps of predictions vs ground truth or target values for different cell types.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    model_names
        List of model names (adata.layers) to plot for predictions heatmap. Default is to plot all models in `adata.layers`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before plotting.
    vmin
        Minimum value for heatmap color scale.
    vmax
        Maximum value for heatmap color scale.
    reorder
        Whether or not to order the clases by similarity (boolean).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Examples
    --------
    >>> crested.pl.heatmap.correlations_predictions(
    ...     adata,
    ...     model_names=None,
    ...     split="test",
    ...     log_transform=True,
    ...     vmin=0.4,
    ...     vmax=0.85,
    ...     title="Correlations: Predictions vs Ground Truth",
    ... )

    .. image:: ../../../../docs/_static/img/examples/heatmap_correlations_predictions.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if len(adata.layers) == 0:
            raise ValueError("No predictions found in adata.layers.")

        if model_names is not None:
            for model in model_names:
                if model not in adata.layers:
                    raise ValueError(f"Model {model} not found in adata.layers.")

        if split is not None:
            if "split" not in adata.var:
                raise ValueError(
                    "No split column found in anndata.var. Run `pp.train_val_test_split` first if 'split' is not None."
                )
            if split not in ["train", "val", "test", None]:
                raise ValueError("Split must be 'train', 'val', 'test', or None.")

    _check_input_params()

    classes = list(adata.obs_names)

    if model_names is None:
        model_names = list(adata.layers.keys())

    logger.info(
        f"Plotting heatmap correlations for split: {split}, models: {model_names}"
    )

    if split is not None:
        x = adata[:, adata.var["split"] == split].X
        predicted_values = {
            model: adata[:, adata.var["split"] == split].layers[model]
            for model in model_names
        }
    else:
        x = adata.X
        predicted_values = {model: adata.layers[model] for model in model_names}

    if log_transform:
        x = np.log1p(x)
        for key in predicted_values:
            predicted_values[key] = np.log1p(predicted_values[key])

    n_models = len(predicted_values)

    fig, axes = plt.subplots(1, n_models, sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y) in zip(axes, predicted_values.items()):
        # this is the same as
        # c = np.corrcoef(np.vstack([x, y]))
        # so c[0, 0] in the old funciton would correspond to
        # c[0, x.shape[0]] in this new function
        correlation_matrix = np.corrcoef(x, y)
        # reformat the array to only get correlations between x and y
        # and no self correlations
        correlation_matrix = np.hsplit(np.vsplit(correlation_matrix, 2)[1], 2)[0].T

        # Reorder the rows/columns to group related classes together
        if reorder:
            D = pdist(correlation_matrix, "correlation")
            Z = hc.linkage(D, "complete", optimal_ordering=True)
            ordering = hc.leaves_list(Z)
            correlation_matrix = correlation_matrix[ordering, :][:, ordering]
            classes = np.array(classes)[ordering]

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            xticklabels=classes,
            yticklabels=classes,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(f"{model_name}")

    default_width = 8 * n_models
    default_height = 8

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, **kwargs)
