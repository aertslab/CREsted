"""Scatterplot of region functions."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData

from crested.pl._utils import create_plot, render_plot
from crested.pp._utils import _calc_gini


def gini_filtering(
    adata: AnnData,
    cutoffs: Sequence = (1.5, 1, 0.5, 0),
    color_points: bool = True,
    line_cmap: str | Sequence = 'tab10',
    plot_kws: dict | None = None,
    line_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
):
    """
    Plot the effect of different potential filtering cutoffs in :func:`~crested.pp.filter_regions_on_specificity` before doing the filtering.

    Parameters
    ----------
    adata
        AnnData object with region data. This should not be filtered yet!
    cutoffs
        List of considered gini standard deviation cutoffs to plot, as in :func:`~crested.pp.filter_regions_on_specificity`'s `gini_std_threshold` (where the default is 1).
    color_points
        Whether to color the region points according to their position inside/outside the different cutoffs.
        If True (default), adds `'c': cutoff_index`, `'cmap': 'Greys'` and `'vmin': 0` to `plot_kws` (unless manually specified).
    line_cmap
        Colors or colormap to draw colors for the different `cutoffs` lines from.
        Any valid input to :func:`~seaborn.color_palette` works. Examples are a matplotlib or seaborn colormap name or a list of colors.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.scatter`. Defaults: `'s': 4`.
    line_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.axvline`. Defaults: `'linestyle': '--'`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 10.
    height
        Height of the newly created figure if `ax=None`. Default is 8.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `gini_filtering`: `xlabel='Rank'`, `ylabel='Standard deviations from the mean Gini'`, `grid='x'`.

    See Also
    --------
    crested.pl.render_plot
    crested.pp.filter_regions_on_specificity

    Example
    -------
    >>> crested.pl.scatter.gini(adata)

    .. image:: ../../../../docs/_static/img/examples/scatter_gini_filtering.png
    """
    if not isinstance(cutoffs, Sequence):
        cutoffs = [cutoffs]

    # Calculate values
    gini_scores = np.max(_calc_gini(adata.X.T), axis=1)
    gini_mean = np.mean(gini_scores)
    gini_std_dev = np.std(gini_scores)

    x = np.arange(gini_scores.shape[0])
    y = np.sort(gini_scores)[::-1]
    y = (y - gini_mean) / gini_std_dev

    # Set defaults
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Rank"
    if 'ylabel' not in kwargs:
        # kwargs['ylabel'] = "Gini score"
        kwargs['ylabel'] = "Standard deviations from the mean Gini"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'x'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 's' not in plot_kws:
        plot_kws['s'] = 4
    line_kws = {} if line_kws is None else line_kws.copy()
    if 'linestyle' not in line_kws:
        line_kws['linestyle'] = '--'

    if color_points:
        cut_bins = np.digitize(y, bins=[-np.inf, *sorted(cutoffs)])
        if 'c' not in plot_kws:
            plot_kws['c'] = cut_bins
        if 'cmap' not in plot_kws:
            plot_kws['cmap'] = 'Greys'
        if 'vmin' not in plot_kws:
            plot_kws['vmin'] = 0
    else:
        if 'c' not in plot_kws:
            plot_kws['c'] = 'black'

    # Create plot
    fig, ax = create_plot(ax=ax, kwargs_dict=kwargs, default_width=10, default_height=8)
    line_palette = sns.color_palette(line_cmap, len(cutoffs))

    # Plot values
    ax.scatter(x=x, y=y, **plot_kws)
    for i, level in enumerate(cutoffs):
        gini_cutoff = gini_mean+level*gini_std_dev
        n_regions = np.sum(y > level)
        ax.axhline(level, color = line_palette[i], label = f"{level}: {n_regions} regions, ($\\mu$+{level}*$\\sigma$={gini_cutoff:.2f} Gini)", **line_kws)
    ax.legend()
    ax.margins(x=0.01)

    return render_plot(fig, ax, **kwargs)
