"""Track visualisation function."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import render_plot
from crested.utils._logging import log_and_raise


@log_and_raise(ValueError)
def track(
    scores: np.ndarray,
    range: tuple[chr, int, int] | tuple[int, int] | None = None,
    title: str | None = None,
    ylim: tuple(float, float) | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs
) -> plt.Figure:
    """Plot a predicted locus track, like a Borzoi prediction or BigWig track.

    Function is still in beta, and its API can be changed in future updates.

    Parameters
    ----------
    scores
        A 1d numpy array of heights along the track.
    range
        A tuple of coordinates that are being plotted between, as (chr, start, end) or (start, end)
    title
        The title of the plot.
    ylim
        Y limits for the plot.
    width
        Plot width. Default is 20.
    height
        Plot height. Default is 3 per subplot.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot
    crested.utils.read_bigwig_region

    Example
    --------
    >>> crested.pl.locus.track(
    ...     preds[0, :, class_idx],
    ...     range=(chrom, start, end),
    ...     title="Mouse Borzoi ATAC:MGL predictions around the FIRE enhancer"
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_track.png
    """
    # Temp shape handling - goal is to make this variable depending on scores shape to track multiple classes/sequences
    if scores.ndim != 1:
        raise ValueError("crested.pl.locus.track() currently only supports one-dimensional data.")
    n_subplots = 1
    n_bins = scores.shape[0]

    # Prep figure inputs
    if width is None:
        width = 20
    if height is None:
        height = 3*n_subplots
    fig, ax = plt.subplots(n_subplots)
    if range is not None:
        if len(range) == 2:
            start, end = range
            chrom = None
        elif len(range) == 3:
            chrom, start, end = range
        else:
            raise ValueError(f"range must be (start, end) or (chrom, start, end), not {range} (len {len(range)}).")
        x = np.linspace(start, end, num=n_bins)
    else:
        x = np.arange(n_bins)

    # Plot figure
    ax.fill_between(x, scores)
    ax.margins(x=0)
    ax.xaxis.set_major_formatter("{x:,.0f}")

    # Set layout options
    if title is not None:
        ax.set_title(title)

    if range is not None:
        xlabel = f"{start:,.0f}-{end:,.0f} ({end-start} bp)"
        if chrom is not None:
            xlabel = chrom + ":" + xlabel
        ax.set_xlabel(xlabel)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Set y bottom margin to 0
        ax.set_ylim(min(scores), None)

    return render_plot(fig, width, height, **kwargs)

