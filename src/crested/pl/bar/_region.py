"""Bar plot prediction functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger

from crested._logging import log_and_raise
from crested.pl._utils import render_plot


def region_predictions(
    adata: AnnData,
    region: str,
    model_names: list[str] | None = None,
    share_y: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Barplots of all predictions in .layers vs the groundtruth for a specific region across comparing classes.

    Parameters
    ----------
    adata
        AnnData object containing the predictions in `layers`.
    region
        String in the format 'chr:start-end' representing the genomic location.
    model_names
        List of model names in `adata.layers`. If None, will create a plot per model in `adata.layers`.
    share_y
        Whether to rescale the y-axis to be the same across plots. Default is True.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.bar.region_predictions(
    ...     adata,
    ...     region='chr1:3094805-3095305'
    ...     model_names=["model_1", "model_2"],
    ...     share_y=False,
    ...     title="Region chr1:3094805-3095305"
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region_predictions.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if region not in list(adata.var_names):
            raise ValueError(f"{region} not found in adata.var_names.")

        if model_names is not None:
            for model_name in model_names:
                if model_name not in adata.layers:
                    raise ValueError(f"Model {model_name} not found in adata.layers.")

    _check_input_params()

    if model_names is None:
        model_names = list(adata.layers.keys())

    logger.info(f"Plotting bar plots for region: {region}, models: {model_names}")

    region_idx = adata.var_names.get_loc(region)

    x = adata.X[:, region_idx]
    predicted_values = {
        model: adata.layers[model][:, region_idx] for model in model_names
    }

    n_models = len(predicted_values)

    fig, axes = plt.subplots(
        n_models + 1,
        1,
        figsize=(kwargs.get("width", 20), kwargs.get("height", 3 * (n_models + 1))),
        sharex=True,
        sharey=share_y,
    )

    for _, (ax, (model_name, y)) in enumerate(zip(axes[:-1], predicted_values.items())):
        ax.bar(list(adata.obs_names), y, alpha=0.8, label=model_name)
        ax.set_ylabel(model_name)
        ax.grid(True)
        ax.set_xticks(np.arange(len(adata.obs_names)))
        ax.set_xticklabels(list(adata.obs_names))

    axes[-1].bar(
        list(adata.obs_names), x, color="green", alpha=0.7, label="Ground Truth"
    )
    axes[-1].set_ylabel("Ground truth")
    axes[-1].grid(True)

    default_height = 6 * n_models
    default_width = 18

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, **kwargs)


def region(
    adata: AnnData, region: str, target: str = "groundtruth", **kwargs
) -> plt.Figure:
    """
    Barplot of groundtruths or predictions for a specific region comparing classes.

    Parameters
    ----------
    adata
        AnnData object containing the genomic data in `var`.
    region
        String in the format 'chr:start-end' representing the genomic location.
    target
        The target to plot the distribution for, either "groundtruth" or the name of a prediction layer in adata.layers.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.bar.region(
    ...     adata,
    ...     region="chr1:3094805-3095305",
    ...     target="groundtruth",
    ...     xlabel="Cell Type",
    ...     ylabel="Peak height",
    ...     figsize=(20, 3),
    ...     figtitle="chr1:3094805-3095305",
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if target not in ["groundtruth"] + list(adata.layers.keys()):
            raise ValueError(f"Target {target} not found in adata.layers.")

        if region not in list(adata.var_names):
            raise ValueError(f"{region} not found in adata.var_names.")

    _check_input_params()

    if target == "groundtruth":
        data = adata.X[:, adata.var_names.get_loc(region)]
    else:
        data = adata.layers[target][:, adata.var_names.get_loc(region)]

    logger.info(f"Plotting bar plot for region: {region}, target: {target}")

    fig, ax = plt.subplots()
    ax.bar(list(adata.obs_names), data, alpha=0.8)

    default_height = 6
    default_width = 18

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, **kwargs)
