"""Locus scoring plotting function."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def locus_scoring(
    scores: np.ndarray,
    range: tuple[int, int],
    gene_start: int | None = None,
    gene_end: int | None = None,
    bigwig_values: np.ndarray | None = None,
    bigwig_midpoints: list[int] | None = None,
    highlight_positions: list[tuple[int, int]] | None = None,
    locus_plot_kws: dict | None = None,
    bigwig_plot_kws: dict | None = None,
    highlight_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Plot model predictions over a genomic locus and optionally indicate the gene body.

    Also plots values from a bigWig file if provided.

    Parameters
    ----------
    scores
        An array of prediction scores for each window.
    range
        The genomic range of the input. Can be (start, end) or (chr, start, end).
    gene_start
        The start position of the gene locus to highlight on the plot.
    gene_end
        The end position of the gene locus to highlight on the plot.
    ylim
        Manually set the y-range of the plot.
    bigwig_values
        A numpy array of values extracted from a bigWig file for the same coordinates.
    bigwig_midpoints
        A list of base pair positions corresponding to the bigwig_values.
    highlight_positions
        A list of tuples specifying ranges to highlight on the plot.
    locus_plot_kws
        Extra keyword arguments passed to :func:`~matplotlib.Axes.plot` for the prediction plot.
        Defaults: `{'markersize': 5.0, 'linewidth': 2.0, 'color': 'b', 'marker': 'o', 'label': 'Prediction score', 'rasterized': True}`.
    bigwig_plot_kws
        Extra keyword arguments passed to :func:`~matplotlib.Axes.plot` for the bigWig plot.
        Defaults: `{'color': 'b', 'linestyle'='-', 'label': 'bigWig values', 'rasterized': True}`.
    highlight_kws
        Keywords to use for plotting highlights with :func:`~matplotlib.Axes.axvspan`.
        Default is {'color': "green", 'alpha': 0.3}
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch. Can only be supplied if not plotting a bigWig.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (30, 3) or (30, 6) with a bigWig plot.
    sharex, sharey
        Whether to share x and y axes of the created plots. Default is True for both.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `locus_scoring`:  `grid='both'`, `xlabel='Genomic position'`,
        `title='Predictions across genomic regions'(, "bigWig coverage across genomic regions")`, `ylabel='Prediction_score'(, 'bigWig values')`.

    See Also
    --------
    crested.tl.Crested.score_gene_locus
    crested.utils.read_bigwig_region

    Example
    --------
    >>> crested.pl.locus.locus_scoring(
    ...     scores,
    ...     range=(0, 1000),
    ...     gene_start=100,
    ...     gene_end=200,
    ...     title=["Predictions across genomic regions", "bigWig coverage across genomic regions"],
    ...     bigwig_values=bigwig_values,
    ...     bigwig_midpoints=bigwig_midpoints,
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_locus_scoring.png
    """
    # Handle deprecated arguments
    if 'figsize' in kwargs:
        logger.warning("Argument `figsize` is deprecated; please use width and height instead.")
        figsize = kwargs.pop('figsize')
        kwargs['width'] = figsize[0]
        kwargs['height'] = figsize[1]
    if 'marker_size' in kwargs:
        logger.warning("Argument `marker_size` is deprecated; please use `locus_plot_kws={markersize=?}` instead. Note the lack of underscore in the new argument, to unify with matplotlib.")
        locus_plot_kws['markersize'] = kwargs.pop('marker_size')
    if 'line_width' in kwargs:
        logger.warning("Argument `line_width` is deprecated; please use `shared_plot_kws={linewidth=?}` instead. Note the lack of underscore in the new argument, to unify with matplotlib.")
        locus_plot_kws['linewidth'] = kwargs.pop('line_width')
    if 'line_colors' in kwargs:
        logger.warning("Argument `line_colors` is deprecated; please use `locus_plot_kws={color=?}`/`bigwig_plot_kws={color=?}` instead.")
        line_colors = kwargs.pop('line_colors')
        locus_plot_kws['color'] = line_colors[0]
        bigwig_plot_kws['color'] = line_colors[1]
    if 'title' in kwargs and isinstance(kwargs['title'], str) and bigwig_values is not None:
        logger.warning(f"Argument `title` only applying to the top plot is deprecated to make behavior consistent. To keep a primary title, please use `suptitle='{kwargs['title']}'` or `title=['{kwargs['title']}', '']`.")
        kwargs['title'] = [kwargs['title'], None]
    if 'ylim' in kwargs and not isinstance(kwargs['ylim'][0], Sequence) and bigwig_values is not None:
        logger.warning(f"Argument `ylim` only applying to the top plot is deprecated to make behavior consistent. To apply only to the top plot, please use `ylim=[{kwargs['ylim']}, None]`.")
        kwargs['ylim'] = [kwargs['ylim'], None]

    # Check params
    @log_and_raise(ValueError)
    def _check_input_params():
        if ax is not None and bigwig_values is not None:
            raise ValueError("ax can only be set if not adding a bigWig plot. Please don't supply `ax`, or only plot the locus scoring by disabling `bigwig_values`.")
        if not 2 <= len(range) <= 3:
            raise ValueError(f"Range must be (chr, start, end) or (start, end), so it cannot have length {len(range)}.")
        # Validate highlight_positions to ensure they fall within the specified range.
        if highlight_positions:
            for pos in highlight_positions:
                start, end = pos
                if start < range[-2] or end > range[-1]:
                    raise ValueError(
                        f"Highlighted position ({start}, {end}) falls outside the plotting range {range}."
                    )
        if bigwig_values is not None and bigwig_midpoints is None:
            raise ValueError("If providing bigwig_values, must also provide bigwig_midpoints.")
        if bigwig_midpoints is not None and bigwig_values is None:
            raise ValueError("If providing bigwig_midpoints, must also provide bigwig_values.")

    _check_input_params()
    bigwig_included = bigwig_values is not None and bigwig_midpoints is not None

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 30
    default_height = 6 if bigwig_included else 3
    plot_height = kwargs.pop('height') if 'height' in kwargs else default_height
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else False
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else False
    if 'title' not in kwargs:
        kwargs['title'] = "Predictions across genomic regions"
        if bigwig_included: # Add empty title for bottom plot
            kwargs['title'] = [kwargs['title'], "bigWig coverage across genomic regions"]
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Genomic position'
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Prediction score'
        if bigwig_included: # Set separate labels for both
            kwargs['ylabel'] = [kwargs['ylabel'], 'bigWig values']
    if 'grid' not in kwargs:
        kwargs['grid'] = 'both'

    locus_plot_kws = {} if locus_plot_kws is None else locus_plot_kws.copy()
    if 'markersize' not in locus_plot_kws:
        locus_plot_kws['markersize'] = 5.0
    if 'linewidth' not in locus_plot_kws:
        locus_plot_kws['linewidth'] = 2.0
    if 'color' not in locus_plot_kws:
        locus_plot_kws['color'] = 'b'
    if 'marker' not in locus_plot_kws:
        locus_plot_kws['marker'] = "o"
    if 'label' not in locus_plot_kws:
        locus_plot_kws['label'] = 'Prediction score'
    if 'rasterized' not in locus_plot_kws:
        locus_plot_kws['rasterized'] = True

    bigwig_plot_kws = {} if bigwig_plot_kws is None else bigwig_plot_kws.copy()
    if 'color' not in bigwig_plot_kws:
        bigwig_plot_kws['color'] = 'g'
    if 'linestyle' not in bigwig_plot_kws:
        bigwig_plot_kws['linestyle'] = '-'
    if 'label' not in bigwig_plot_kws:
        bigwig_plot_kws['label'] = 'bigWig values'
    if 'rasterized' not in bigwig_plot_kws:
        bigwig_plot_kws['rasterized'] = True

    highlight_kws = {} if highlight_kws is None else highlight_kws.copy()
    if 'color' not in highlight_kws:
        highlight_kws['color'] = 'green'
    if 'alpha' not in highlight_kws:
        highlight_kws['alpha'] = 0.3

    # Create plots
    nrows = 2 if bigwig_included else 1
    fig, axs = create_plot(ax=ax, width=plot_width, height=plot_height, nrows=nrows, sharex=sharex, sharey=sharey)
    if nrows == 1:
        axs = [axs]

    # Top plot: Model predictions
    axs[0].plot(
        np.arange(range[-2], range[-1]),
        scores,
        **locus_plot_kws
    )

    # Bottom plot: bigWig values
    if bigwig_included:
        axs[1].plot(
            bigwig_midpoints,
            bigwig_values,
            **bigwig_plot_kws
        )

    # Set shared properties
    for ax in axs:
        ax.margins(x=0)
        ax.xaxis.set_major_formatter("{x:,.0f}")
        ax.set_ylim(bottom=0)
        ax.legend()
        if gene_start is not None and gene_end is not None:
            ax.axvspan(
                gene_start, gene_end, color="red", alpha=0.2, label="Gene locus"
            )
        if highlight_positions:
            for start, end in highlight_positions:
                ax.axvspan(start, end, **highlight_kws)

    return render_plot(fig, axs, **kwargs)
