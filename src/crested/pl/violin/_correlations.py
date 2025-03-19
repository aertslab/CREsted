"""Correlation violinplot plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from scipy.stats import pearsonr

from crested.pl._utils import render_plot
from crested.utils._logging import log_and_raise


def correlations(
    adata: AnnData,
    model_names: str | list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    ylim: tuple(float, float) | None = (0., 1.),
    title: str = "Class-wise prediction vs ground truth correlations",
    **kwargs,
) -> plt.Figure:
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
    ylim
        Limits for the y axis.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Examples
    --------
    >>> crested.pl.violin.correlations(
    ...     adata,
    ...     model_names=['Base DilatedCNN', 'Fine-tuned DilatedCNN'],
    ...     split="test",
    ...     log_transform=True,
    ...     ylim=(0., 1.),
    ...     title="Class-wise prediction vs ground truth correlations",
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
    if model_names is not None and isinstance(model_names, str):
        model_names = [model_names]
    _check_input_params()
    if model_names is None:
        model_names = list(adata.layers.keys())

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
        correlations[model_name] = [pearsonr(x[class_idx, :], y[class_idx])[0] for class_idx in range(adata.n_obs)]

    # Plot
    fig, ax = plt.subplots(sharey=False)
    ax.grid(visible=True, which='major', axis='y', color = '.85')
    sns.violinplot(correlations, inner = 'point', ax = ax, zorder = 2.05, orient = 'v', inner_kws={'s': 20, 'marker': '.', 'edgecolor': '0', 'color': '0.01', 'alpha': 1})
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set layout options
    default_width = 6 + max(0, len(predicted_values)-5) # 1-5 models: 6 wide, 5+ models: add 1 extra width per model
    default_height = 8
    if any(len(model_name) > 8 for model_name in model_names) and "x_label_rotation" not in kwargs:
        kwargs["x_label_rotation"] = 55

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, title=title, **kwargs)
