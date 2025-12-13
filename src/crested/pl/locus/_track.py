"""Track visualisation function."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


@log_and_raise(ValueError)
def track(
    scores: np.ndarray,
    class_idxs: int | list[int] | None = None,
    range: tuple[chr, int, int] | tuple[int, int] | None = None,
    class_names: Sequence[str] | str |  None = None,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot a predicted locus track, like a Borzoi prediction or BigWig track.

    Parameters
    ----------
    scores
        A numpy array of heights along the track.
        Can be shapes (length) or (length, classes), and will automatically squeeze out one-wide dimensions.
    class_idxs
        Index or list of indices denoting classes to plot. If None, plots all classes.
    range
        Optional, a tuple of coordinates that are being plotted between, as (chr, start, end) or (start, end), to set the x coordinates.
    class_names
        Optional, list of all possible class names to extract label names from.
        If class_idxs is supplied, picks from there. If not, will use these in order.
    plot_kws
        Extra keyword arguments passed to :func:`~matplotlib.Axes.fill_between`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (20, 3) per class.
    sharex, sharey
        Whether to share x and y axes of the created plots. Default is `sharex=False`, `sharey=True`.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `track`:
        `xlabel=f"{chr}:{start:,.0f}-{end:,.0f} ({end - start} bp)"`, (if `range` is set),
        `title=class_names[class_idxs]` (if `class_names` is set)

    See Also
    --------
    crested.pl.render_plot
    crested.utils.read_bigwig_region

    Example
    --------
    >>> crested.pl.locus.track(
    ...     preds,
    ...     class_idxs=class_idx,
    ...     range=(chrom, start, end),
    ...     title="Mouse Borzoi ATAC:MGL predictions around the FIRE enhancer",
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_track.png

    # TODO: add example for bigwig import?
    """
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
        if range is not None:
            if not 2 <= len(range) <= 3:
                raise ValueError(f"Range must be (chr, start, end) or (start, end), so it cannot have length {len(range)}.")
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

    # Handle range
    if range is not None:
        start, end = range[-2], range[-1]
        chrom = range[-3] if len(range) == 3 else None
        binsize = np.abs((end-start)//2)//n_bins
        x = np.linspace(start+(binsize//2), end, num=n_bins)
    else:
        x = np.arange(n_bins)

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 20
    plot_height = kwargs.pop('height') if 'height' in kwargs else 3*(n_classes)
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else False
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else True
    if 'title' not in kwargs and class_names is not None:
        kwargs['title'] = [class_names[cidx] for cidx in class_idxs]
    if 'xlabel' not in kwargs and range is not None:
        default_xlabel = f"{start:,.0f}-{end:,.0f} ({end - start} bp)"
        if chrom is not None:
            default_xlabel = chrom + ":" + default_xlabel
        kwargs["xlabel"] = default_xlabel
    if 'xlim' not in kwargs and range is not None:
        kwargs["xlim"] = (start, end)
    plot_kws = {} if plot_kws is None else plot_kws.copy()

    # Prep figure inputs
    fig, axs = create_plot(
        ax=ax,
        width=plot_width,
        height=plot_height,
        nrows=n_classes,
        sharex=sharex,
        sharey=sharey
    )
    if n_classes == 1:
        axs = [axs]

    # Plot figure
    for i, ax in enumerate(axs):
        ax.fill_between(x, scores[:, i], **plot_kws)
        ax.margins(x=0)
        ax.xaxis.set_major_formatter("{x:,.0f}")
        # Set layout options
        ax.set_ylim(bottom=min(scores[:, i]))

    return render_plot(fig, axs, **kwargs)
