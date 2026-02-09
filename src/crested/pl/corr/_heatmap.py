"""Heatmap plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import seaborn as sns
from anndata import AnnData
from scipy.spatial.distance import pdist

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def _generate_heatmap(
    ax: plt.Axes,
    correlation_matrix: np.ndarray,
    classes: list | np.ndarray | pd.Series,
    vmin: float | None = None,
    vmax: float | None = None,
    reorder: bool = False,
    cmap: str | plt.Colormap = 'coolwarm',
    cbar: bool = True,
    cbar_kws: None | dict = None,
    annot: bool = False,
    fmt: str = '.2f',
    square: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot the base correlation heatmap, wrapper around `:func:`~seaborn.heatmap`.

    Parameters
    ----------
    ax
        Axis to plot the heatmap on.
    correlation_matrix
        Rectangular matrix to plot.
    classes : list-like
        Classes to plot. Assumed to be labels for both x and y axis.
    vmin
        Minimum value to anchor the colormap. If None, inferred from data.
    vmax
        Maximum value to anchor the colormap. If None, inferred from data.
    reorder
        Whether to order classes by similarity.
    cmap
        Colormap to use.
    cbar
        Whether to draw a colorbar.
    cbar_kws
        Extra keyword arguments passed to the colorbar through `:func:`~seaborn.heatmap`.
        Default is `{'label': "Pearson correlations (of log1p-transformed values)"}`
    annot
        Whether to write the data value in each cell.
    fmt
        String formatting code to use when adding annotations with `annot`.
    square
        Whether to enforce Axes.aspect('equal') so the heatmap will be square.
    """
    # Reorder the rows/columns to group related classes together
    if reorder:
        D = pdist(correlation_matrix, "correlation")
        Z = hc.linkage(D, "complete", optimal_ordering=True)
        ordering = hc.leaves_list(Z)
        correlation_matrix = correlation_matrix[ordering, :][:, ordering]
        classes = np.array(classes)[ordering]

    ax = sns.heatmap(
        correlation_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        vmin=vmin,
        vmax=vmax,
        square=square,
        cbar=cbar,
        cbar_kws=cbar_kws,
        ax=ax,
        **kwargs
    )
    return ax


def heatmap_self(
    adata: AnnData,
    log_transform: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    reorder: bool = False,
    cmap: str | plt.Colormap = 'coolwarm',
    cbar: bool = True,
    cbar_kws: dict | None = None,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot self correlation heatmaps of ground truth for different cell types.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    log_transform
        Whether to log-transform the data before plotting.
    vmin
        Minimum value for heatmap color scale.
    vmax
        Maximum value for heatmap color scale.
    reorder
        Whether or not to order the clases by similarity.
    cmap
        Colormap to use.
    cbar
        Whether to draw a colorbar.
    cbar_kws
        Extra keyword arguments passed to the colorbar.
        Default is `{'label': "Pearson correlations (of log1p-transformed values)"}`
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.heatmap`.
        Adjusted defaults compared to the base function are `{'square': True, 'fmt': '.2f'}`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 10, or 8 if `cbar=False`.
    height
        Height of the newly created figure if `ax=None`. Default is 8.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `correlations_self`: `xtick_rotation=90`, `layout='compressed'`.

    See Also
    --------
    crested.pl.render_plot

    Examples
    --------
    >>> crested.pl.heatmap.correlations_self(
    ...     adata,
    ...     log_transform=True,
    ...     vmin=0,
    ...     vmax=1,
    ...     title="Self correlations heatmap",
    ... )

    .. image:: ../../../../docs/_static/img/examples/heatmap_self_correlations.png
    """
    # Set defaults
    if 'xtick_rotation' not in kwargs:
        kwargs['xtick_rotation'] = 90
    if 'layout' not in kwargs:
        kwargs['layout'] = 'compressed'
    plot_kws = {} if plot_kws is None else plot_kws.copy() # Most plot defaults handled in _generate_heatmap() defaults
    cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
    if 'label' not in cbar_kws:
        cbar_kws['label'] = "Pearson correlation"
        if log_transform:
            cbar_kws['label'] += " of log1p-transformed values"

    # Gather data
    x = adata.X
    classes = list(adata.obs_names)

    if log_transform:
        x = np.log1p(x)

    correlation_matrix = np.corrcoef(x)

    # Plot heatmap
    default_width = 10 if cbar else 8
    fig, ax = create_plot(ax=ax, kwargs_dict=kwargs, default_width=default_width, default_height=8)
    ax = _generate_heatmap(ax=ax, correlation_matrix=correlation_matrix, classes=classes, vmin=vmin, vmax=vmax, reorder=reorder, cmap=cmap, cbar=cbar, cbar_kws=cbar_kws, **plot_kws)

    return render_plot(fig, ax, **kwargs)


def heatmap_predictions(
    adata: AnnData,
    model_names: str | list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    reorder: bool = False,
    cmap: str | plt.Colormap = 'coolwarm',
    cbar: bool = True,
    cbar_kws: dict | None = None,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Plot correlation heatmaps of predictions vs ground truth for all cell types.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    model_names
        Model name or list of model names (adata.layers) to plot for predictions heatmap. Default is to plot all models in `adata.layers`.
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
    cmap
        Colormap to use.
    cbar
        whether to draw a colorbar.
    cbar_kws
        Extra keyword arguments passed to the colorbar.
        Default is `{'label': "Pearson correlations (of log1p-transformed values)"}`
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.heatmap`.
        Adjusted defaults compared to the base function are `square=True` and `fmt='.2f'`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch. Can only be supplied if plotting a single model.
    width
        Width of the newly created figure if `ax=None`. Default is 10 per model to plot, or 8 if `cbar=False`.
    height
        Height of the newly created figure if `ax=None`. Default is 8 per model to plot.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `correlations_predictions`: `xtick_rotation=90`, `layout='compressed'`, `title=list(adata.obs_names)`.

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
    ...     suptitle="Correlations: predictions vs ground truth",
    ... )

    .. image:: ../../../../docs/_static/img/examples/heatmap_correlations_predictions.png
    """
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if len(adata.layers) == 0:
            raise ValueError("No predictions found in adata.layers.")

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

        if ax is not None and len(model_names) > 1:
            raise ValueError("ax can only be set if plotting one model. Please pick one model in `model_names`.")

    classes = list(adata.obs_names)

    if isinstance(model_names, str):
        model_names = [model_names]
    elif model_names is None:
        model_names = list(adata.layers.keys())

    _check_input_params()

    n_models = len(model_names)

    # Set defaults
    if 'xtick_rotation' not in kwargs:
        kwargs['xtick_rotation'] = 90
    if 'layout' not in kwargs:
        kwargs['layout'] = 'compressed'
    if 'title' not in kwargs:
        kwargs['title'] = list(model_names)
    plot_kws = {} if plot_kws is None else plot_kws.copy() # Most plot defaults handled in _generate_heatmap() defaults
    cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
    if 'label' not in cbar_kws:
        cbar_kws['label'] = "Pearson correlation"
        if log_transform:
            cbar_kws['label'] += " of log1p-transformed values"

    # Prepare ground truth values
    x = adata[:, adata.var["split"] == split].X if split is not None else adata.X
    if log_transform:
        x = np.log1p(x)

    # Create plots
    default_width = 10*n_models if cbar else 8*n_models
    fig, axs = create_plot(ax=ax, kwargs_dict=kwargs, default_width=default_width, default_height=8, ncols=n_models)
    if n_models == 1:
        axs = [axs]

    # Calculate correlation coefficients with predictions and plot
    for i, model in enumerate(model_names):
        y = adata[:, adata.var["split"] == split].layers[model] if split is not None else adata.layers[model]
        if log_transform:
            y = np.log1p(y)
        # this is the same as
        # c = np.corrcoef(np.vstack([x, y]))
        # so c[0, 0] in the old function would correspond to
        # c[0, x.shape[0]] in this new function
        correlation_matrix = np.corrcoef(x, y)
        # reformat the array to only get correlations between x and y
        # and no self correlations
        correlation_matrix = np.hsplit(np.vsplit(correlation_matrix, 2)[1], 2)[0].T

        ax = _generate_heatmap(ax=axs[i], correlation_matrix=correlation_matrix, classes=classes, vmin=vmin, vmax=vmax, reorder=reorder, cmap=cmap, cbar=cbar, cbar_kws=cbar_kws, **plot_kws)

    return render_plot(fig, axs, **kwargs)
