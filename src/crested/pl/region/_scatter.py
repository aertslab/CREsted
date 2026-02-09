"""Scatterplot of region functions."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def scatter(
    adata: AnnData,
    region: str,
    model_names: str | list[str] | None = None,
    log_transform: bool = False,
    annotate_top: int | None = 5,
    top_method: Literal['sum', 'truth', 'pred'] = 'sum',
    square: bool = True,
    identity_line: bool = False,
    plot_kws: dict | None = None,
    annotate_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Scatterplot of ground truths and predictions for a specific region in your data, comparing different classes.

    Parameters
    ----------
    adata
        AnnData object containing the genomic data in `var`.
    region
        Region name from the AnnData, generally in format 'chr:start-end'.
    model_names
        The target to plot the distribution for, either None (to plot all) or one/multiple names of a prediction layer in adata.layers.
    log_transform
        Whether to apply a log1p transformation to the data.
    annotate_top
        Number of top classes to annotate with their class names. If None, does not annotate any.
    top_method
        How to determine the top n classes. "truth" or "pred" use highest on either, "sum" takes highest sum of both.
    square
        Whether to force the plots to be square, have equal aspect ratios, and equal shared axis ranges.
    identity_line
        Whether to plot a y=x line denoting perfect correlation.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.scatter`. Defaults: `'s': 10`, `'color': 'white'`, `'edgecolor': 'black'`.
    annotate_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.annotate`. Defaults: `'xytext': (5, 0)`, `'textcoords': 'offset points'`
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 7*n_models.
    height
        Height of the newly created figure if `ax=None`. Default is 7.
    sharex
        Whether to share the x axes of the created plots. Default is False. Setting `square=True` does equalize limits even if `sharex=False`
    sharey
        Whether to share the y axes of the created plots. Default is False. Setting `square=True` does equalize limits even if `sharey=False`.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `region`: `xlabel="Ground truth"`, `ylabel="Prediction", `title="region"`,
        `layout="compressed"`, `title="{region}-{model_name}"`(1 model)/`model_name`(>1 model), `suptitle=region` (>1 model).

    See Also
    --------
    crested.pl.render_plot
    crested.pl.region.bar

    Example
    -------
    >>> crested.pl.region.scatter(
    ...     adata,
    ...     "chr1:3111367-3113481",
    ...     model_names='Base model',
    ...     log_transform=True,
    ...     identity_line=True,
    ... )

    .. image:: ../../../../docs/_static/img/examples/scatter_region.png
    """
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        for model_name in model_names:
            if model_name not in adata.layers:
                raise ValueError(f"Model name {model_name} not found in adata.layers.")
        if region not in list(adata.var_names):
            raise ValueError(f"{region} not found in adata.var_names.")
        if n_models > 1 and ax is not None:
            raise ValueError("Cannot plot multiple models if providing 'ax'. Please specify one model_name.")

    if isinstance(model_names, str):
        model_names = [model_names]
    if model_names is None:
        model_names = list(adata.layers.keys())
    n_models = len(model_names)

    _check_input_params()

    # Gather inputs

    # Set defaults
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Ground truth"
        if log_transform:
            kwargs['xlabel'] = "Log1p-transformed " + kwargs['xlabel'].lower()
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Prediction'
        if log_transform:
            kwargs['ylabel'] = "Log1p-transformed " + kwargs['ylabel'].lower()
    if 'title' not in kwargs:
        if n_models == 1:
            kwargs['title'] = f"{region} - {model_names[0]}"
        else:
            kwargs['title'] = model_names
    if 'suptitle' not in kwargs and n_models > 1:
        kwargs['suptitle'] = region
    if 'layout' not in kwargs:
        kwargs['layout'] = 'compressed'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 's' not in plot_kws:
        plot_kws['s'] = 10
    if 'color' not in plot_kws:
        plot_kws['color']  = 'white'
    if 'edgecolor' not in plot_kws:
        plot_kws['edgecolor'] = 'black'
    annotate_kws = {} if annotate_kws is None else plot_kws.copy()
    if 'xytext' not in annotate_kws:
        annotate_kws['xytext'] = (5, 0)
    if 'textcoords' not in annotate_kws:
        annotate_kws['textcoords'] = 'offset points'

    # Create plot
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=n_models*7,
        default_height=7,
        ncols=n_models,
        default_sharex=False,
        default_sharey=True
    )
    if n_models == 1:
        axs = [axs]

    # Gather data
    region_idx = adata.var_names.get_loc(region)
    truth = adata.X[:, region_idx].squeeze()
    predicted_values = np.array([adata.layers[model_name][:, region_idx].squeeze() for model_name in model_names])

    # Log-transform data
    if log_transform:
        truth = np.log1p(truth)
        predicted_values = np.log1p(predicted_values)

    # Plot per model
    for i, ax in enumerate(axs):
        pred = predicted_values[i, :]
        ax.scatter(x = truth, y = pred, **plot_kws)
        if annotate_top:
            if top_method == 'sum':
                top_values = pred+truth
            elif top_method == 'truth':
                top_values = truth
            elif top_method == 'pred':
                top_values = pred
            else:
                raise ValueError(f"Did not recognise top method {top_method}. Must be one of ['sum', 'truth', 'pred'].")
            for top_idx in np.argsort(top_values)[-annotate_top:]:
                ax.annotate(adata.obs_names[top_idx], (truth[top_idx], pred[top_idx]), **annotate_kws)
        if square:
            ax.set_box_aspect(1)
            shared_range = np.min([np.min(truth), np.min(predicted_values)]), np.max([np.max(truth), np.max(predicted_values)]) # intentionally across all preds
            absolute_margin=0.05*(shared_range[1]-shared_range[0])
            ax.set_xlim(shared_range[0]-absolute_margin, shared_range[1]+absolute_margin)
            ax.set_ylim(shared_range[0]-absolute_margin, shared_range[1]+absolute_margin)
        if identity_line:
            ax.axline((0, 0), slope=1, color = 'black', alpha = 0.5, linestyle='--')

    return render_plot(fig, axs, **kwargs)
