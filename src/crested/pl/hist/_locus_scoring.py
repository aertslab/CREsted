"""Distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def locus_scoring(
    scores: np.ndarray,
    range: tuple[int, int],
    gene_start: int | None = None,
    gene_end: int | None = None,
    title: str = "Predictions across Genomic Regions",
    bigwig_values: np.ndarray | None = None,
    bigwig_midpoints: list[int] | None = None,
    filename: str | None = None,
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
    title
        The title of the plot.
    bigwig_values
        A numpy array of values extracted from a bigWig file for the same coordinates.
    bigwig_midpoints
        A list of base pair positions corresponding to the bigwig_values.
    filename
        The filename to save the plot to.

    See Also
    --------
    crested.tl.Crested.score_gene_locus
    crested.utils.extract_bigwig_values_per_bp

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
    # Plotting predictions
    plt.figure(figsize=(30, 10))

    # Top plot: Model predictions
    plt.subplot(2, 1, 1)
    plt.plot(
        np.arange(range[0], range[1]),
        scores,
        marker="o",
        linestyle="-",
        color="b",
        label="Prediction Score",
    )
    if gene_start is not None and gene_end is not None:
        plt.axvspan(gene_start, gene_end, color="red", alpha=0.3, label="Gene Locus")
    plt.title(title)
    plt.xlabel("Genomic Position")
    plt.ylabel("Prediction Score")
    plt.ylim(bottom=0)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()

    # Bottom plot: bigWig values
    if bigwig_values is not None and bigwig_midpoints is not None:
        plt.subplot(2, 1, 2)
        plt.plot(
            bigwig_midpoints,
            bigwig_values,
            linestyle="-",
            color="g",
            label="bigWig Values",
        )
        if gene_start is not None and gene_end is not None:
            plt.axvspan(
                gene_start, gene_end, color="red", alpha=0.3, label="Gene Locus"
            )
        plt.xlabel("Genomic Position")
        plt.ylabel("bigWig Values")
        plt.xticks(rotation=90)
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
