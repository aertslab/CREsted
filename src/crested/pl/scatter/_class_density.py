"""Scatter plotting functions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def _fit_kde(x, y, downsample_density, max_threads):
    xy = np.vstack([x, y])
    # Fit KDE to data
    if downsample_density and downsample_density < xy.shape[1]:
        downsample_idxs = np.random.randint(
            xy.shape[1], size=downsample_density
        )
        kde = gaussian_kde(xy[:, downsample_idxs])
    else:
        kde = gaussian_kde(xy)
    # Evaluate data points to position on fitted KDE
    if max_threads > 1:
        xy_chunked = np.array_split(xy, max_threads, axis=1)
        z = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for chunk in executor.map(kde, xy_chunked):
                z.append(chunk)
            z = np.concatenate(z)
    else:
        z = kde(xy)
    return z

def class_density(
    adata: AnnData,
    class_name: str | None = None,
    model_names: str | list[str] | None = None,
    split: str | None = "test",
    log_transform: bool = False,
    exclude_zeros: bool = True,
    density_indication: bool = False,
    square: bool = False,
    identity_line: bool = False,
    colorbar: bool = False,
    downsample_density: int = 10000,
    max_threads: int = 8,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Plot a density scatter plot of predictions vs ground truth for specified models and class.

    Parameters
    ----------
    adata
        AnnData object containing the data in `X` and predictions in `layers`.
    class_name
        Name of the class in `adata.obs_names`. If None, plot is made for all the classes.
    model_names
        Model name or list of model names in `adata.layers`. If None, will create a plot per model in `adata.layers`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before plotting.
    exclude_zeros
        Whether to exclude zero ground truth values from the plot.
    density_indication
        Whether to indicate density in the scatter plot.
    square
        Whether to force the plots to be square and have equal aspect ratios.
    identity_line
        Whether to plot a y=x line denoting perfect correlation.
    colorbar
        Whether to plot the colorbar when using `density_indication`.
    downsample_density
        Number of points to downsample to when fitting the density if using the density indication.
        Note that one point denotes one region for one class, so the full set would be # of (test) regions * # classes.
        Default is 10000. If False, will not downsample.
    max_threads
        Maximum number of threads to use when evaluating the density if using the density indication. If 1, will not parallelize.
    plot_kws
        Extra keyword arguments passed to :func:`~matplotlib.Axes.scatter`. Defaults: `{'alpha': 0.25, 'edgecolor': 'k'}.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (7, 8) per model.
    sharex, sharey
        Whether to share x and y axes of the created plots. Default is `sharex=False, sharey=True`.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `class_density`: `xlabel="Ground truth"`, `ylabel='Predictions'`, `alpha='0.25'`,
        `title=({class_name} - ){model_name}`, `suptitle='Targets vs predictions (for {class_name})'` (if n_models>1).

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
    # Check params
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
        if class_name is not None and class_name not in adata.obs_names:
            raise ValueError(f"Class {class_name} not found in adata.obs_names.")
        if split not in ["train", "val", "test", None]:
            raise ValueError("Split must be 'train', 'val', 'test', or None.")
        if ax is not None and len(model_names) > 1:
            raise ValueError("ax can only be set if plotting one model. Please pick one model in `model_names`.")
        if colorbar is True and density_indication is False:
            raise ValueError("`colorbar` is only used if `density_indication` is True.")

    if isinstance(model_names, str):
        model_names = [model_names]
    elif model_names is None:
        model_names = list(adata.layers.keys())

    _check_input_params()

    classes = list(adata.obs_names)
    column_index = np.arange(0, len(classes)) if class_name is None else classes.index(class_name)

    n_models = len(model_names)

    # Gather data
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

    if class_name:
        logger.info(
            f"Plotting density scatter for class: {class_name}, models: {model_names}, split: {split}"
        )
    else:
        logger.info(
            f"Plotting density scatter for all targets and predictions, models: {model_names}, split: {split}"
        )

    # Set defaults
    default_width = 8*n_models if (colorbar and density_indication) else 7*n_models
    plot_width = kwargs.pop('width') if 'width' in kwargs else default_width
    plot_height = kwargs.pop('height') if 'height' in kwargs else 8
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else False
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else True
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "Ground truth"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Predictions"
    if 'title' not in kwargs:
        kwargs['title'] = model_names
        if class_name is not None:
            kwargs['title'] = [f"{class_name} - " + ax_title for ax_title in kwargs['title']]
    if "suptitle" not in kwargs and n_models > 1:
        kwargs["suptitle"] = "Targets vs predictions"
        if class_name is not None:
            kwargs["suptitle"] += f" for {class_name}"
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha' not in plot_kws:
        plot_kws['alpha'] = 0.25
    if 'edgecolor' not in plot_kws:
        plot_kws['edgecolor'] = "k"

    # subplot_kw = {'box_aspect': 1} if square else None
    # Create plot
    fig, axs = create_plot(
        ax=ax,
        width=plot_width,
        height=plot_height,
        ncols=n_models,
        sharex=sharex,
        sharey=sharey,
    )
    if n_models == 1:
        axs = [axs]

    # Plot values
    for ax, y in zip(axs, predicted_values.values(), strict=True):
        if identity_line:
            ax.axline((0, 0), slope=1, color = 'black', alpha = 0.5, linestyle='--')
        pearson_corr, _ = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)

        # Calculate density indication and plot
        if density_indication:
            z = _fit_kde(x, y, downsample_density, max_threads)
            scatter_pathcoll = ax.scatter(x, y, c=z, s=50, **plot_kws)
            scatter_pathcoll.set_rasterized(True)  # Rasterize only the scatter points
            if colorbar:
                fig.colorbar(ScalarMappable(cmap=scatter_pathcoll.cmap, norm=scatter_pathcoll.norm), ax = ax, label="Density")
        else:
            ax.scatter(x, y, **plot_kws)

        ax.annotate(
            text=f"Pearson: {pearson_corr:.2f}",
            xy=(0.05, 0.95),
            xycoords=("axes fraction", "axes fraction"),
            verticalalignment="top",
        )
        ax.annotate(
            text=f"Spearman: {spearman_corr:.2f}",
            xy=(0.05, 0.90),
            xycoords=("axes fraction", "axes fraction"),
            verticalalignment="top",
        )
        if square:
            ax.set_box_aspect(1)
            ax.set_aspect('equal', adjustable='datalim')

    return render_plot(fig, axs, **kwargs)
