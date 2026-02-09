"""Scatterplot of region functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from scipy.sparse import csr_matrix

from crested.pl._utils import create_plot, render_plot
from crested.pp._utils import _calc_gini, _calc_proportion


def filter_cutoff(
    adata: AnnData,
    cutoffs: Sequence = (1.5, 1, 0.5, 0),
    model_name: str | None = None,
    color_points: bool = True,
    double_y_axis: bool = True,
    line_cmap: str | Sequence = 'tab10',
    plot_kws: dict | None = None,
    line_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
) -> (plt.Figure, plt.Axes) | None:
    """
    Plot the effect of different potential filtering cutoffs in :func:`~crested.pp.filter_regions_on_specificity` before doing the filtering.

    If the default cutoff (1 standard deviation above the mean) is too stringent or too lenient, try different cutoffs and pick the best one.

    Parameters
    ----------
    adata
        AnnData object with region data. This should not be filtered yet!
    cutoffs
        List of considered gini standard deviation cutoffs to plot, as in :func:`~crested.pp.filter_regions_on_specificity`'s `gini_std_threshold` (where the default is 1).
    model_name
        The name of the model to calculate scores from. If None or 'truth'/'groundtruth'/'X' (default), will use the values in adata.X.
    color_points
        Whether to color the region points according to their position inside/outside the different cutoffs.
        If True (default), adds `'c': cutoff_index`, `'cmap': 'Greys'` and `'vmin': 0` to `plot_kws` (unless manually specified).
    double_y_axis
        Whether to add a secondary y axis on the right side showing the Gini scores, augmenting the default axis showing standard deviations from the Gini mean.
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
    if model_name is None or model_name in ['X', 'truth', 'groundtruth']:
        if isinstance(adata.X, csr_matrix):
            target_matrix = adata.X.toarray().T
        else:
            target_matrix = adata.X.T
    else:
        if model_name not in adata.layers:
            raise ValueError(f"Model name {model_name} not found in adata.layers. Please provide a valid model name.")
        target_matrix = adata.layers[model_name].T
    gini_scores = np.max(_calc_gini(target_matrix), axis=1)
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
        ax.axhline(level, color = line_palette[i], label = f"{level}: {n_regions} regions ($\\mu$+{level}*$\\sigma$={gini_cutoff:.2f} Gini)", **line_kws)
    if double_y_axis:
        ylabel_fontsize = kwargs['ylabel_fontsize'] if 'ylabel_fontsize' in kwargs else 14
        secax = ax.secondary_yaxis('right', functions=(lambda sd: sd*gini_std_dev+gini_mean, lambda gini: (gini-gini_mean)/gini_std_dev))
        secax.set_ylabel("Gini score", fontsize=ylabel_fontsize)
    ax.legend()
    ax.margins(x=0.01)

    return render_plot(fig, ax, **kwargs)


def sort_and_filter_cutoff(
    adata: AnnData,
    cutoffs: list[int] | None = None,
    method: Literal['gini', 'proportion'] = 'gini',
    model_name: str | None = None,
    max_k: int = 2000,
    cmap: str | Sequence = 'tab20',
    legend: bool = True,
    class_labels: bool = True,
    ax: plt.Axes | None = None,
    plot_kws: dict | None = None,
    line_kws: dict | None = None,
    **kwargs
) -> (plt.Figure, plt.Axes) | None:
    """
    Plot the effect of different `top_k` cutoffs in :func:`~crested.pp.sort_and_filter_regions_on_specificity`, which takes the top k regions per class.

    This function plots the gini scores for those regions per class, indicating the specificity. If the specificity is too low for the lower-ranked genes, decrease `top_k`.

    Parameters
    ----------
    adata
        AnnData object with region data. This should not be filtered yet!
    cutoffs
        List of considered top amounts to plot an illustrative line at, as in :func:`~crested.pp.sort_and_filter_regions_on_specificity`'s `top_k`.
    model_name
        The name of the model to calculate scores from. If None or 'truth'/'groundtruth'/'X' (default), will use the values in adata.X.
    method
        The method to use for calculating scores, either 'gini' or 'proportion'.
        Default is 'gini'.
    max_k
        The maximum number of top regions per cell type to plot. Should be higher than the cutoffs you're considering.
    cmap
        Colors or colormap to draw colors for the different cell types from.
        Any valid input to :func:`~seaborn.color_palette` works. Examples are a matplotlib or seaborn colormap name or a list of colors.
    legend
        Whether to add a cell type legend.
    class_labels
        Whether to add labels on the per-class lines to the right edge of the plot.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.scatter`. Defaults: `'s': 4`.
    line_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.axvline`. Defaults: `'color': 'black'`, `'linestyle': '--'`.
    width
        Width of the newly created figure if `ax=None`. Default is 10.
    height
        Height of the newly created figure if `ax=None`. Default is 6.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `?`: `xlabel='Rank'`, `ylabel='Gini score'`, `grid='x'`.

    See Also
    --------
    crested.pl.render_plot
    crested.pp.sort_and_filter_regions_on_specificity

    Example
    -------
    >>> crested.pl.scatter.gini_filtering_class(adata, cutoffs = [350, 750])

    .. image:: ../../../../docs/_static/img/examples/scatter_gini_filtering_class.png
    """
    # Validate cutoffs
    if cutoffs is not None:
        if not isinstance(cutoffs, Sequence):
            cutoffs=[cutoffs]
        assert all(max_k >= cutoff for cutoff in cutoffs), f"All cutoffs must be smaller than max_k {max_k}"

    # Calculate scores
    if model_name is None or model_name in ['X', 'truth', 'groundtruth']:
        if isinstance(adata.X, csr_matrix):
            target_matrix = adata.X.toarray().T
        else:
            target_matrix = adata.X.T
    else:
        if model_name not in adata.layers:
            raise ValueError(f"Model name {model_name} not found in adata.layers. Please provide a valid model name.")
        target_matrix = adata.layers[model_name].T

    if method == "gini":
        scores = _calc_gini(target_matrix)
    elif method == "proportion":
        scores = _calc_proportion(target_matrix)
    else:
        raise ValueError("Method must be either 'gini' or 'proportion'.")

    # Set defaults
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Rank"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Gini score"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'x'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 's' not in plot_kws:
        plot_kws['s'] = 4
    line_kws = {} if line_kws is None else line_kws.copy()
    if 'color' not in line_kws:
        line_kws['color'] = 'black'
    if 'linestyle' not in line_kws:
        line_kws['linestyle'] = '--'

    # Create plot
    fig, ax = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=10,
        default_height=6,
    )

    # Get colormap
    colors = sns.color_palette(cmap, n_colors=adata.n_obs)

    # Plot values
    for i, class_name in enumerate(adata.obs_names):
        class_max_gini = np.sort(scores[:, i])[::-1][:max_k]
        ax.scatter(x=np.arange(max_k), y=class_max_gini, label=class_name, color = colors[i], **plot_kws)
        if class_labels:
            ax.annotate(class_name, (max_k, class_max_gini[-1]), ha='right', va='center')
    if cutoffs is not None:
        for cutoff in cutoffs:
            ax.axvline(cutoff, **line_kws)
    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=10)
    ax.margins(x=0.01)

    return render_plot(fig, ax, **kwargs)
