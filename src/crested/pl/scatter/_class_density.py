"""Scatter plotting functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from crested._logging import log_and_raise
from crested.pl._utils import render_plot


def class_density(
    adata: AnnData,
    class_name: str,
    model_names: list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    exclude_zeros: bool = True,
    density_indication: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Plot a density scatter plot of predictions vs ground truth for specified models and class.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    class_name
        Name of the class in `adata.obs_names`.
    model_names
        List of model names in `adata.layers`. If None, will create a plot per model in `adata.layers`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before plotting. Default is False.
    exclude_zeros
        Whether to exclude zero values from the plot. Default is True.
    density_indication
        Whether to indicate density in the scatter plot. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.scatter.class_density(
    ...     adata,
    ...     class_name="Astro",
    ...     model_names=["model1", "model2"],
    ...     split="test",
    ...     log_transform=True,
    ... )

    .. image:: ../../../../docs/_static/img/examples/scatter_class_density.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if model_names is not None:
            for model_name in model_names:
                if model_name not in adata.layers:
                    raise ValueError(f"Model {model_name} not found in adata.layers.")

        if split is not None and "split" not in adata.var:
            raise ValueError(
                "No split column found in anndata.var. Run `pp.train_val_test_split` first if 'split' is not None."
            )

        if class_name not in adata.obs_names:
            raise ValueError(f"Class {class_name} not found in adata.obs_names.")
        if split not in ["train", "val", "test", None]:
            raise ValueError("Split must be 'train', 'val', 'test', or None.")

    _check_input_params()

    classes = list(adata.obs_names)
    column_index = classes.index(class_name)
    if model_names is None:
        model_names = list(adata.layers.keys())

    if split is not None:
        x = adata[:, adata.var["split"] == split].X[column_index, :].flatten()
        predicted_values = {
            model: adata[:, adata.var["split"] == split]
            .layers[model][column_index, :]
            .flatten()
            for model in model_names
        }
    else:
        x = adata.X[column_index, :].flatten()
        predicted_values = {
            model: adata.layers[model][column_index, :].flatten()
            for model in model_names
        }

    if exclude_zeros:
        mask = x != 0
        x = x[mask]
        for model in predicted_values:
            predicted_values[model] = predicted_values[model][mask]

    if log_transform:
        x = np.log1p(x)
        for model in predicted_values:
            predicted_values[model] = np.log1p(predicted_values[model])

    n_models = len(predicted_values)

    logger.info(
        f"Plotting density scatter for class: {class_name}, models: {model_names}, split: {split}"
    )

    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y) in zip(axes, predicted_values.items()):
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)

        if density_indication:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            scatter = ax.scatter(x, y, c=z, s=50, edgecolor="k", alpha=0.25)
            plt.colorbar(scatter, ax=ax, label="Density")
        else:
            scatter = ax.scatter(x, y, edgecolor="k", alpha=0.25)

        ax.annotate(
            f"Pearson: {pearson_corr:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            verticalalignment="top",
        )
        ax.annotate(
            f"Spearman: {spearman_corr:.2f}",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
            verticalalignment="top",
        )

        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.set_title(f"{model_name}")

    default_width = 8 * n_models
    default_height = 8

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "Ground Truth"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Predictions"
    if "title" not in kwargs:
        kwargs["title"] = f"{class_name}"

    return render_plot(fig, **kwargs)
