"""Distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from loguru import logger

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def distribution(
    adata: AnnData,
    target: str | None = None,
    class_names: list[str] | None = None,
    split: str | None = None,
    log_transform: bool = True,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Histogram of region distribution for specified classes.

    Parameters
    ----------
    adata
        AnnData object containing the predictions in `layers`.
    target
        The target to plot the distribution for, either None (for the ground truth) or the name of a prediction layer in adata.layers.
    class_names
        Single class name or list of classes in `adata.obs`. If None, will create a plot per class in `adata.obs`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before plotting.
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.histplot`.
        Defaults: `{'kde': True, 'stat': 'frequency', 'color': 'skyblue', 'binwidth': np.ptp(data)}`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch. Can only be supplied if plotting a single model.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (8, 6) per class histogram.
    sharex, sharey
        Whether to share x and y axes of the created plots. Default is True for both.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `distribution`: `grid='both'`.

    See Also
    --------
    crested.pl.render_plot

    Example
    --------
    >>> crested.pl.hist.distribution(
    ...     adata,
    ...     split="test",
    ...     sharey=False,
    ...     class_names=["Astro", "Vip"]
    ... )

    .. image:: ../../../../docs/_static/img/examples/hist_distribution.png
    """
    # Handle deprecated arguments
    if 'share_y' in kwargs:
        kwargs['sharey'] = kwargs.pop('share_y')
        logger.warning("Argument `share_y` is deprecated; please use sharey instead to align with matplotlib.")

    # Check params
    @log_and_raise(ValueError)
    def _check_input_params():
        for class_name in class_names:
            if class_name not in list(adata.obs_names):
                raise ValueError(f"{class_name} not found in adata.obs_names.")

        if target is not None and target not in adata.layers:
            raise ValueError(f"{target} not found in adata.layers.")

        if split is not None:
            if "split" not in adata.var:
                raise ValueError(
                    "No split column found in adata.var. Run `pp.train_val_test_split` first if 'split' is not None."
                )
        if ax is not None and len(class_names) > 1:
            raise ValueError("ax can only be set if plotting one class. Please pick one class in `class_names`.")

    if target == "groundtruth":
        target = None

    if isinstance(class_names, str):
        class_names = [class_names]
    elif class_names is None:
        class_names = list(adata.obs_names)

    _check_input_params()

    n_classes = len(class_names)
    n_cols = int(np.ceil(np.sqrt(n_classes)))
    n_rows = int(np.ceil(n_classes / n_cols))

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 8*n_cols
    plot_height = kwargs.pop('height') if 'height' in kwargs else 6*n_rows
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else True
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else True
    if 'grid' not in kwargs:
        kwargs['grid'] = 'both'
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Ground truth' if target is None else target
        if log_transform:
            kwargs['xlabel'] = "Log1p-transformed " + kwargs['xlabel'].lower()

    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'kde' not in plot_kws:
        plot_kws['kde'] = True
    if 'color' not in plot_kws:
        plot_kws['color'] = 'skyblue'
    if 'stat' not in plot_kws:
        plot_kws['stat'] = 'frequency'

    # Create plots
    fig, axs = create_plot(ax=ax, width=plot_width, height=plot_height, nrows=n_rows, ncols=n_cols, sharex=sharex, sharey=sharey)
    if n_classes == 1:
        axs = [axs]
    else:
        axs = axs.ravel()

    for i, class_name in enumerate(class_names):
        # Gather data
        if target is None:
            data = adata.X[adata.obs_names.get_loc(class_name), :]
        else:
            data = adata.layers[target][adata.obs_names.get_loc(class_name), :]
        if split is not None:
            data = data[adata.var["split"] == split]
        if log_transform:
            data = np.log1p(data)

        # Plot data
        # See plot_kws defaults for most of the other defaults of this function
        binwidth = plot_kws['binwidth'] if 'binwidth' in plot_kws else np.ptp(data) / 50
        sns.histplot(
            data,
            binwidth=binwidth,
            ax=axs[i],
            **plot_kws,
        )
        axs[i].set_title(class_name)

    # Hide non-used plots
    for i in range(len(class_names), len(axs)):
        axs[i].set_axis_off()

    return render_plot(fig, axs, **kwargs)
