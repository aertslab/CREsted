"""Track visualisation function."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested.pl._utils import _parse_coordinates_input, create_plot, render_plot
from crested.utils._logging import log_and_raise


@log_and_raise(ValueError)
def track(
    scores: np.ndarray,
    class_idxs: int | list[int] | None = None,
    coordinates: str | tuple | None = None,
    class_names: Sequence[str] | str |  None = None,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    range: str = 'deprecated',
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """Plot a predicted locus track, like a Borzoi prediction or BigWig track.

    Parameters
    ----------
    scores
        A numpy array of heights along the track.
        Can be shapes (length) or (length, classes), and will automatically squeeze out one-wide dimensions.
    class_idxs
        Index or list of indices denoting classes to plot. If None, plots all classes.
    coordinates
        Optional, a string or tuple of coordinates that are being plotted between, to set the x coordinates.
        Can be a parsable chr:start-region(:strand) string, or a tuple with ((chr), start, end, (strand)), with chr and strand being optional.
        If strand is provided and '-', runs the x coordinates from end to start.
    class_names
        Optional, list of all possible class names to extract label names from.
        If class_idxs is supplied, picks from there. If not, will use these in order.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.fill_between`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 20.
    height
        Height of the newly created figure if `ax=None`. Default is 3 per class.
    sharex
        Whether to share the y axes of the created plots. Default is False
    sharey
        Whether to share the y axes of the created plots. Default is True.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `track`:
        `xlabel=f"{chr}:{start:,.0f}-{end:,.0f} ({end - start} bp)"`, (if `coordinates` is set),
        `title=class_names[class_idxs]` (if `class_names` is set)

    See Also
    --------
    crested.pl.render_plot
    crested.utils.read_bigwig_region

    Example
    --------
    >>> chrom, start, end = 'chr18', 61010523, 61207131
    >>> class_idxs = [output_names_borzoi.index(class_name) for class_name in ['ATAC:MGL', 'ATAC:SSTGA1', 'ATAC:VEC']]
    >>> crested.pl.locus.track(
    ...     borzoi_pred,
    ...     class_idxs=class_idxs,
    ...     class_names=output_names_borzoi,
    ...     coordinates=(chrom, start, end),
    ...     suptitle="Mouse Borzoi predictions around the FIRE enhancer",
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_track_pred.png

    >>> bw_values, midpoints = crested.utils.read_bigwig_region(bw_path, (chrom, start, end))
    >>> crested.pl.locus.track(
    ...     bw_values,
    ...     coordinates=(chrom, start, end),
    ...     title="Mouse Borzoi microglia values around the FIRE enhancer",
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_track_bw.png

    """
    if range != 'deprecated':
        coordinates = range
        logger.warning("Argument `range` is renamed; please use `coordinates` instead.")
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if scores.ndim != 2:
            raise ValueError("scores must be (length) or (length, classes)")
        if class_idxs is not None:
            for cidx in class_idxs:
                if cidx > scores.shape[0]:
                    raise ValueError(f"class_idxs {class_idxs} is beyond your input's number of classes ({n_classes}).")
                if class_names is not None and cidx >= len(class_names):
                    raise ValueError(f"class_idxs {cidx} is beyond the size of class_names ({len(class_names)}).")
        if ax is not None and n_classes > 1:
            raise ValueError("ax can only be set if plotting one class. Please pick one class in `class_idxs` or pass unidimensional data.")

    # Remove singleton dimensions like single-sequence batch dims
    scores = scores.squeeze()
    # Add class dim if only providing 1D track
    if scores.ndim == 1:
        scores = np.expand_dims(scores, -1)

    # Turn class_idxs into consistent list
    if class_idxs is None:
        class_idxs = list(np.arange(scores.shape[-1]))
    elif not isinstance(class_idxs, Sequence):
        class_idxs = [class_idxs]
    if isinstance(class_names, str):
        class_names = [class_names]

    n_bins = scores.shape[0]
    n_classes = len(class_idxs)

    _check_input_params()

    # Handle coordinates
    if coordinates is not None:
        chrom, start, end, strand = _parse_coordinates_input(coordinates)
        binsize = np.abs(end-start)//n_bins
        if strand == "-":
            x = np.arange(end-(binsize//2), start, step=-binsize)
        else:
            x = np.arange(start+(binsize//2), end, step=binsize)
    else:
        x = np.arange(n_bins)

    # Set defaults
    if 'title' not in kwargs and class_names is not None:
        kwargs['title'] = [class_names[cidx] for cidx in class_idxs]
    if 'xlabel' not in kwargs and coordinates is not None:
        default_xlabel = f"{start:,.0f}-{end:,.0f} ({np.abs(end - start)} bp)"
        if chrom is not None:
            default_xlabel = chrom + ":" + default_xlabel
        kwargs["xlabel"] = default_xlabel
    plot_kws = {} if plot_kws is None else plot_kws.copy()

    # Prep figure inputs
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=20,
        default_height=3*n_classes,
        nrows=n_classes,
        default_sharex=False,
        default_sharey=True
    )
    if n_classes == 1:
        axs = [axs]

    # Plot figure
    for i, ax in enumerate(axs):
        ax.fill_between(x, scores[:, class_idxs[i]], **plot_kws)
        ax.margins(x=0)
        ax.xaxis.set_major_formatter("{x:,.0f}")
        # Set layout options
        ax.set_ylim(bottom=min(scores[:, i]))
        if coordinates is not None and strand == "-":
            ax.xaxis.set_inverted(True)

    return render_plot(fig, axs, **kwargs)
