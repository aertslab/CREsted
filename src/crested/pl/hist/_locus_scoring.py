"""Distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import render_plot


def locus_scoring(
    scores: np.ndarray,
    range: tuple[int, int],
    gene_start: int | None = None,
    gene_end: int | None = None,
    bigwig_values: np.ndarray | None = None,
    bigwig_midpoints: list[int] | None = None,
    **kwargs,
):
    """
    Plot the predictions as a line chart over the entire genomic input and optionally indicate the gene locus.

    Also plots values from a bigWig file if provided.

    Parameters
    ----------
    scores
        An array of prediction scores for each window.
    range
        The genomic range of the input.
    model_class
        The class index to plot from the prediction scores.
    gene_start
        The start position of the gene locus to highlight on the plot.
    gene_end
        The end position of the gene locus to highlight on the plot.
    bigwig_values
        A numpy array of values extracted from a bigWig file for the same coordinates.
    bigwig_midpoints
        A list of base pair positions corresponding to the bigwig_values.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.tl.Crested.score_gene_locus
    crested.utils.read_bigwig_region

    Example
    --------
    >>> crested.pl.hist.locus_scoring(
    ...     scores,
    ...     range=(0, 1000),
    ...     gene_start=100,
    ...     gene_end=200,
    ...     title="Predictions across Genomic Regions",
    ...     bigwig_values=bigwig_values,
    ...     bigwig_midpoints=bigwig_midpoints,
    ... )

    .. image:: ../../../../docs/_static/img/examples/hist_locus_scoring.png
    """
    # Top plot: Model predictions
    if bigwig_midpoints is not None and bigwig_values is not None:
        nrows = 2
    else:
        nrows = 1
    fig, axes = plt.subplots(
        nrows,
        1,
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    axes[0].plot(
        np.arange(range[0], range[1]),
        scores,
        marker="o",
        linestyle="-",
        color="b",
        label="Prediction Score",
    )
    if gene_start is not None and gene_end is not None:
        axes[0].axvspan(
            gene_start, gene_end, color="red", alpha=0.3, label="Gene Locus"
        )

    axes[0].set_xlabel("Genomic Position")
    for label in axes[0].get_xticklabels():
        label.set_rotation(90)
    axes[0].set_ylabel("Prediction Score")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True)
    axes[0].legend()

    # Bottom plot: bigWig values
    if bigwig_values is not None and bigwig_midpoints is not None:
        axes[1].plot(
            bigwig_midpoints,
            bigwig_values,
            linestyle="-",
            color="g",
            label="bigWig Values",
        )
        if gene_start is not None and gene_end is not None:
            axes[1].axvspan(
                gene_start, gene_end, color="red", alpha=0.3, label="Gene Locus"
            )
        axes[1].set_xlabel("Genomic Position")
        axes[1].set_ylabel("bigWig Values")
        axes[1].grid(True)
        axes[1].legend()

    plt.tight_layout()

    default_height = 5 * nrows
    default_width = 30

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height
    if "title" not in kwargs:
        kwargs["title"] = "Predictions across Genomic Regions"

    return render_plot(fig, **kwargs)
