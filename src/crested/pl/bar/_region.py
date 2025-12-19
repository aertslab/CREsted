"""Bar plot prediction functions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger

import crested
from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def region_predictions(
    adata: AnnData,
    region: str,
    model_names: str | list[str] | None = None,
    pred_color: str = 'tab:blue',
    truth_color: str = 'green',
    plot_kws: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Barplots of all predictions in .layers vs the ground truth for a specific region across comparing classes.

    Parameters
    ----------
    adata
        AnnData object containing the predictions in `layers`.
    region
        String in the format 'chr:start-end' representing the genomic location.
    model_names
        Single model name or list of model names in `adata.layers`. If None, will create a plot per model in `adata.layers`.
    pred_color
        Plot color of the prediction barplot(s).
    truth_color
        Plot color of the ground truth barplot.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.bar`.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (20, 3*(1+n_models)).
    sharex, sharey
        Whether to share x and y axes of the created plots. Default is True for both.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `region_predictions`: `grid="y"`.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.bar.region_predictions(
    ...     adata,
    ...     region='chr1:3094805-3095305'
    ...     model_names=["model_1", "model_2"],
    ...     sharey=False,
    ...     title="Region chr1:3094805-3095305"
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region_predictions.png
    """
    # Handle deprecated arguments
    if 'share_y' in kwargs:
        kwargs['sharey'] = kwargs.pop('share_y')
        logger.warning("Argument `share_y` is deprecated; please use sharey instead to align with matplotlib.")

    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if region not in list(adata.var_names):
            raise ValueError(f"{region} not found in adata.var_names.")

        if model_names is not None:
            for model_name in model_names:
                if model_name not in adata.layers:
                    raise ValueError(f"Model {model_name} not found in adata.layers.")

    if isinstance(model_names, str):
        model_names = [model_names]
    elif model_names is None:
        model_names = list(adata.layers.keys())

    _check_input_params()

    n_models = len(model_names)

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 20
    plot_height = kwargs.pop('height') if 'height' in kwargs else 3*(n_models+1)
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else True
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else True
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()

    # Create figure scaffold
    fig, axs = plt.subplots(
        n_models+1,
        1,
        figsize=(plot_width, plot_height),
        sharex=sharex,
        sharey=sharey,
    )

    # Plot predictions
    plot_kws_pred = plot_kws.copy()
    if 'color' not in plot_kws_pred:
        plot_kws_pred['color'] = pred_color
    for i in range(n_models):
        _ = crested.pl.bar.region(
            adata=adata,
            region=region,
            target=model_names[i],
            grid=False, # Disable here so that final render_plot can set it
            show=False,
            plot_kws=plot_kws_pred,
            ax=axs[i],
        )

    # Plot ground truth
    plot_kws_truth = plot_kws.copy()
    if 'color' not in plot_kws_truth:
        plot_kws_truth['color'] = truth_color
    _ = crested.pl.bar.region(
        adata=adata,
        region=region,
        target=None,
        grid=False, # Disable here so that final render_plot can set it
        show=False,
        plot_kws=plot_kws_truth,
        ax=axs[-1],
    )
    return render_plot(fig, axs, **kwargs)

def region(
    adata: AnnData,
    region: str,
    target: str | None = None,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Barplot of ground truths or predictions for a specific region in your data, comparing different classes.

    Parameters
    ----------
    adata
        AnnData object containing the genomic data in `var`.
    region
        Region name from the AnnData, generally in format 'chr:start-end'.
    target
        The target to plot the distribution for, either None (for the ground truth from adata.X) or the name of a prediction layer in adata.layers.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.bar`. Defaults: `'alpha': 0.8`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (18, 6).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `region`: `xlabel="None"`, `ylabel="Ground truth"`/`target`, `grid='y'`.

    See Also
    --------
    crested.pl.render_plot
    crested.pl.bar.region_predictions
    crested.pl.bar.prediction

    Example
    -------
    >>> crested.pl.bar.region(
    ...     adata,
    ...     region="chr1:3094805-3095305",
    ...     target=None,
    ...     xlabel="Cell type",
    ...     ylabel="Peak height",
    ...     width=20,
    ...     height=3,
    ...     figtitle="chr1:3094805-3095305",
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_region.png
    """
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if target is not None and target not in adata.layers:
            raise ValueError(f"Target {target} not found in adata.layers.")
        if region not in list(adata.var_names):
            raise ValueError(f"{region} not found in adata.var_names.")

    if target == "groundtruth":
        target = None

    _check_input_params()

    # Gather inputs
    classes = list(adata.obs_names)
    if target is None:
        data = adata.X[:, adata.var_names.get_loc(region)]
    else:
        data = adata.layers[target][:, adata.var_names.get_loc(region)]

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 18
    plot_height = kwargs.pop('height') if 'height' in kwargs else 6
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = None
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Ground truth' if target is None else target
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha' not in plot_kws:
        plot_kws['alpha'] = 0.8

    # Create plot
    fig, ax = create_plot(ax=ax, width=plot_width, height=plot_height)
    ax.bar(classes, data, **plot_kws)
    return render_plot(fig, ax, **kwargs)


def prediction(
    prediction: np.array,
    classes: list[str],
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Bar plot for a single prediction, comparing different classes.

    Parameters
    ----------
    prediction
        An array containing the prediction values for each class or cell type. It is squeezed to remove 1-wide dimensions if necessary.
    classes
        A list of class or cell type labels corresponding to the predictions.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.bar`. Defaults: `'alpha': 0.8`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (18, 3).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `prediction`: `xlabel='Cell types'`, `ylabel='Prediction'`, `grid="y"`.

    See Also
    --------
    crested.pl.render_plot
    crested.pl.bar.region
    crested.pl.bar.region_predictions

    Example
    -------
    >>> crested.pl.bar.prediction(
    ...     pred,
    ...     classes=list(adata.obs_names),
    ...     title="Region chr1:3094805-3095305"
    ... )
    """
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if len(prediction) != len(classes):
            raise ValueError(
                f"The length of prediction array ({len(prediction)}) must match the number of classes ({len(classes)})."
            )

    # Ensure the prediction array is 1-dimensional
    prediction = prediction.squeeze()

    _check_input_params()

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 18
    plot_height = kwargs.pop('height') if 'height' in kwargs else 3
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Cell types"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Prediction"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha' not in plot_kws:
        plot_kws['alpha'] = 0.8

    # Create plot
    fig, ax = create_plot(ax=ax, width=plot_width, height=plot_height)
    ax.bar(classes, prediction, **plot_kws)

    # Use render_plot to finalize and return the figure
    return render_plot(fig, ax, **kwargs)
