"""Correlation violinplot plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from scipy.stats import pearsonr

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def violin(
    adata: AnnData,
    model_names: str | list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot correlation violinplots of predictions vs ground truth for different cell types.

    Calculates correlations over cell types, i.e. for each cell type, calculates the correlation between ground truth and predictions across all regions.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    model_names
        Model name or list of model names (adata.layers) to use to calculate correlations with ground truths. Default is to plot all models in `adata.layers`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before calculating correlations.
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.violinplot`.
        Defaults: `{'inner': 'point', 'orient': 'v'}`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default width is 6, +1 for every model > 5.
    height
        Height of the newly created figure if `ax=None`. Default is 8.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `correlations`: `ylabel='Pearson correlation'`, `grid='y'`, `ylim=(0.0, 1.0)`, `title="Class-wise prediction vs ground truth correlations"`,
        `xtick_rotation=55` (if any model_names > 15 characters).

    See Also
    --------
    crested.pl.render_plot
    crested.pl.corr.heatmap

    Examples
    --------
    >>> crested.pl.corr.violin(
    ...     adata,
    ...     model_names=["Base DilatedCNN", "Fine-tuned DilatedCNN"],
    ...     split="test",
    ...     log_transform=True,
    ...     title="Per-class model correlation",
    ... )

    .. image:: ../../../../docs/_static/img/examples/violin_correlations.png
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

    # Validate inputs
    if isinstance(model_names, str):
        model_names = [model_names]
    elif model_names is None:
        model_names = list(adata.layers.keys())
    n_models = len(model_names)
    _check_input_params()

    # Set defaults
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Pearson correlation"
        if log_transform:
            kwargs['ylabel'] += " of log1p-transformed values"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    if 'ylim' not in kwargs:
        kwargs['ylim'] = (0.0, 1.0)
    if 'title' not in kwargs:
        kwargs['title'] = 'Class-wise prediction vs ground truth correlations'
    if "xtick_rotation" not in kwargs:
        if any(len(model_name) > 15 for model_name in model_names):
            kwargs["xtick_rotation"] = 55
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'inner' not in plot_kws:
        plot_kws['inner'] = 'point'
    if 'orient' not in plot_kws:
        plot_kws['orient'] = 'v'

    # Gather ground truth and prediction data
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

    # Calculate correlations
    correlations = {}
    for model_name, y in predicted_values.items():
        correlations[model_name] = [
            pearsonr(x[class_idx, :], y[class_idx])[0]
            for class_idx in range(adata.n_obs)
        ]

    # Calculate values
    correlations = {}
    for model_name in model_names:
        if split is not None:
            x = adata[:, adata.var["split"] == split].X
            y = adata[:, adata.var["split"] == split].layers[model_name]
        else:
            x = adata.X
            y = adata.layers[model_name]

        if log_transform:
            x = np.log1p(x)
            y = np.log1p(y)

        correlations[model_name] = [
            pearsonr(x[class_idx, :], y[class_idx])[0]
            for class_idx in range(adata.n_obs)
        ]

     # Create plot
    default_width = 6+max(0, n_models-5)  # 1-5 models: 6 wide, 5+ models: add 1 extra width per model
    fig, ax = create_plot(ax=ax, kwargs_dict=kwargs, default_width=default_width, default_height=8)

    sns.violinplot(
        correlations,
        ax=ax,
        **plot_kws,
    )

    return render_plot(fig, ax, **kwargs)
