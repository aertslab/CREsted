"""Bar plot prediction functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def scores(
    data: AnnData | np.ndarray,
    region: str | None = None,
    model_names: str | list[str] | None = None,
    classes: list[str] | None = None,
    log_transform: bool = False,
    pred_color: str = 'tab:blue',
    truth_color: str = 'green',
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    adata: str = 'deprecated',
    target: str = 'deprecated',
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Barplot of ground truths and/or predictions for a specific region in your data.

    Can plot either a region from an AnnData (`data=adata, region=region`) or manual prediction (`data=pred, classes=adata.obs_names`).
    When plotting from AnnData, can plot the ground truth and any available predictions.

    Parameters
    ----------
    data
        AnnData object containing the genomic data in `var` (requiring the `region` argument), or single prediction numpy array (requiring the `classes` argument).
    region
        Region name from the AnnData, generally in format 'chr:start-end'. Required if providing an AnnData object.
    model_names
        Source of the values to plot, as a name or list of names.
        Can be 'X'/'truth'/'groundtruth' for the ground truth from adata.X, or the name of prediction layers in adata.layers.
        If None, plots the ground truth and all layers in the AnnData.
        Disregarded if plotting a single prediction.
    classes
        The class names to use. Required if providing a single prediction. If using an AnnData object, read from there by default.
    log_transform
        Whether to apply a log1p transformation to the data.
    pred_color
        Plot color of any prediction barplot.
    truth_color
        Plot color of any ground truth barplot.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.bar`. Defaults: `'alpha': 0.8`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 18.
    height
        Height of the newly created figure if `ax=None`. Default is 4*n_model_names.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `region`: `title='Ground truth'`/`target`, `'color'=['tab:blue'`(for predictions)/`'green'`(for ground truth)`]`, `suptitle=region`, `ylabel="Ground truth"`/`"Prediction`, `grid='y'`.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.bar.region(
    ...     adata,
    ...     region='chr1:3093998-3096112',
    ...     target="Base model",
    ...     width=20,
    ...     height=3
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region_adata.png

    >>> crested.pl.bar.region(
    ...     pred,
    ...     classes=adata.obs_names,
    ...     title="Region chr18:61107770-61109884",
    ...     width=20,
    ...     height=3,
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region_pred.png
    """
    # Handle deprecated arguments
    if adata != 'deprecated':
        logger.warning("Argument 'adata' is deprecated, please use 'data' instead.")
        data = adata
    if target != 'deprecated':
        logger.warning("Argument 'target' is deprecated, please use 'model_names' instead.")
        model_names = target

    # Check input validity
    @log_and_raise(ValueError)
    def _check_adata_params():
        if region is None:
            raise ValueError("'region' must be provided if using an AnnData.")
        for target in model_names:
            if target is not None and (target not in data.layers and target != 'truth'):
                raise ValueError(f"Target {target} not found in data.layers or recognised as ground truth ('x', 'truth', 'groundtruth').")
        if region not in list(data.var_names):
            raise ValueError(f"Region {region} not found in data.var_names.")
        if ax is not None and len(model_names) > 1:
            raise ValueError("ax can only be set if plotting one target. Please pick one target in `model_names`.")

    @log_and_raise(ValueError)
    def _check_array_params():
        if classes is None:
            raise ValueError("Classes must be provided if using a single prediction.")
        if data.squeeze().ndim != 1:
            raise ValueError(f"If plotting a single prediction, 'data' must be a one-dimensional array, not shape {data.squeeze().shape}.")
        if len(classes) != data.squeeze().shape[-1]:
            raise ValueError(f"Number of classes provided ({len(classes)}) must be the same as the number of classes in the data {data.squeeze().shape[-1]}")
        if model_names is not None:
            logger.warning(f"'model_names' ({model_names}) provided when providing a single prediction rather than an adata for 'data'. 'model_names' will be ignored.")

    # Handle raw prediction input mode
    if not isinstance(data, AnnData):
        # Check whether params are valid
        _check_array_params()
        # Save values
        values = [data.squeeze()]
        model_names = ["Prediction"]
    # Handle adata input mode
    else:
        # Parse and clean up model_names values
        if model_names is None:
            model_names = ["truth", *data.layers.keys()]
        if isinstance(model_names, str):
            model_names = [model_names]
        model_names = ['truth' if target.lower() in ['x', 'truth', 'groundtruth'] else target for target in model_names]
        # Check whether params are valid
        _check_adata_params()
        # Gather values
        region_idx = data.var_names.get_loc(region)
        values = [data.X[:, region_idx] if target == 'truth' else data.layers[target][:, region_idx] for target in model_names]
        if classes is None:
            classes = list(data.obs_names)

    # Log-transform data
    if log_transform:
        values = [np.log1p(v) for v in values]

    n_model_names = len(values)

    # Set defaults
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = ['Ground truth' if target == 'truth' else "Prediction" for target in model_names]
        if log_transform:
            kwargs['ylabel'] = ["Log1p-transformed " + label.lower() for label in kwargs['ylabel']]
    if 'title' not in kwargs:
        if n_model_names > 1:
            kwargs['title'] = ["Ground truth" if target == "truth" else target for target in model_names]
        else:
            kwargs['title'] = f"{region} - {'Ground truth' if model_names[0] == 'truth' else model_names[0]}"
    if 'suptitle' not in kwargs and n_model_names > 1:
        kwargs['suptitle'] = region
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha' not in plot_kws:
        plot_kws['alpha'] = 0.8

    # Create plot
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=18,
        default_height=4*(n_model_names),
        default_sharex=True,
        default_sharey=True,
        nrows=n_model_names
    )
    if n_model_names == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.bar(
            classes,
            values[i],
            color=truth_color if model_names[i] == 'truth' else pred_color,
            **plot_kws
        )
    return render_plot(fig, axs, **kwargs)


def region(
    adata: AnnData | np.ndarray,
    region: str | None = None,
    target: str = "truth",
    classes: list[str] | None = None,
    log_transform: bool = False,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
):
    """
    Bar plot for a single region, comparing different classes.

    Deprecated in favor of :func:`~crested.pl.bar.scores`. Please use `scores(adata, region, model_names=model_name)` or `scores(adata, region, model_names='truth')` instead.

    :meta private:
    """
    # Deprecation warnings
    target_string = f"'{target}'"
    logger.warning(
        "region is deprecated since version 2.0.0 as its functionality is moved into `scores`. "
        f"Please use `scores(adata, region, model_names={target_string}, **kwargs)` instead."
    )

    return scores( # TODO: adjust when renamed
        data=adata,
        region=region,
        model_names=target,
        classes=classes,
        log_transform=log_transform,
        plot_kws=plot_kws,
        pred_color='tab:blue',
        truth_color='tab:blue',
        ax=ax,
        **kwargs
    )

def region_predictions(
    adata: AnnData,
    region: str,
    model_names: str | list[str] | None = None,
    share_y: bool = True,
    plot_kws: dict | None = None,
    **kwargs
):
    """
    Barplots of all predictions in .layers vs the ground truth for a specific region across comparing classes.

    Deprecated in favor of :func:`~crested.pl.bar.scores`. Please use `scores(adata, region, model_names=None)` or `scores(adata, region, model_names=['model_name', 'truth'])` instead.

    :meta private:
    """
    # Parse model_names into targets tuple
    if model_names is not None:
        if isinstance(model_names, str):
            model_names = [model_names]
        model_names = [*model_names, 'truth']
    else:
        model_names = [*adata.layers.keys(), 'truth']

    # Deprecation warnings
    targets_string = "None" if model_names is None else model_names
    logger.warning(
        "region_predictions is deprecated since version 2.0.0 as its functionality is moved into `scores`. "
        f"Please use `scores(adata, region, model_names={targets_string}, **kwargs)` instead."
    )
    if 'share_y' in kwargs:
        kwargs['sharey'] = kwargs.pop('share_y')
        logger.warning("Argument `share_y` is deprecated since version 2.0.0; please use sharey instead to align with matplotlib.")

    # Mimic old behavior of names at ylabel rather than at title
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = ['Ground truth' if target == 'truth' else target for target in model_names]
    if 'title' not in kwargs:
        kwargs['title'] = None

    return scores(
        data=adata,
        region=region,
        model_names=model_names,
        plot_kws=plot_kws,
        sharey=share_y,
        **kwargs
    )

def prediction(
    prediction: np.ndarray,
    classes: list[str],
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Bar plot for a single prediction, comparing different classes.

    Deprecated in favor of :func:`~crested.pl.bar.scores`. Please use `scores(data=prediction, classes=classes)` instead.

    :meta private:
    """
    logger.warning(
        "`prediction` is deprecated since version 2.0.0 as its functionality is moved into `scores`. "
        "Please use `scores(data=prediction, classes=classes, **kwargs)` instead."
    )
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'xlabel' in kwargs:
        logger.warning(f"'xlabel' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{xlabel={kwargs['xlabel']}}}")
        plot_kws['xlabel'] = kwargs.pop('xlabel')
    if 'ylabel' in kwargs:
        logger.warning(f"'ylabel' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{ylabel={kwargs['ylabel']}}}")
        plot_kws['ylabel'] = kwargs.pop('ylabel')
    if 'title' in kwargs:
        logger.warning(f"'title' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{title={kwargs['title']}}} or {{suptitle={kwargs['title']}}}")
        plot_kws['title'] = kwargs.pop('title')

    return scores(
        data=prediction,
        classes=classes,
        plot_kws=plot_kws,
        ax=ax,
        **kwargs
    )
