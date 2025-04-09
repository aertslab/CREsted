"""Locus scoring plotting function."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def locus_scoring(
    scores: np.ndarray,
    range: tuple[int, int],
    gene_start: int | None = None,
    gene_end: int | None = None,
    title: str = "Predictions across Genomic Regions",
    ylim: tuple(float, float) | None = None,
    bigwig_values: np.ndarray | None = None,
    bigwig_midpoints: list[int] | None = None,
    save_path: str | None = None,
    grid: bool = True,
    figsize: tuple[float, float] = (30, 5),
    highlight_positions: list[tuple[int, int]] | None = None,
    marker_size: float = 5.0,
    line_width: float = 2.0,
    line_colors: tuple(str, str) = ("b", "g"),
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
    ylim
        Manually set the y-range of the plot.
    bigwig_values
        A numpy array of values extracted from a bigWig file for the same coordinates.
    bigwig_midpoints
        A list of base pair positions corresponding to the bigwig_values.
    save_path
        The filename to save the plot to.
    grid
        Add grid to plot.
    figsize
        Size of figure.
    highlight_positions
        A list of tuples specifying ranges to highlight on the plot.
    marker_size
        Size of the markers in the plot. Default is 5.0.
    line_width
        Width of the lines in the plot. Default is 2.0.
    line_colors
        Tuple of colors for the prediction track (first) and BigWig track (second). Default blue and green.

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
    ...     title="Predictions across Genomic Regions",
    ...     bigwig_values=bigwig_values,
    ...     bigwig_midpoints=bigwig_midpoints,
    ... )

    .. image:: ../../../../docs/_static/img/examples/locus_locus_scoring.png
    """
    # Validate highlight_positions to ensure they fall within the specified range.
    if highlight_positions:
        for pos in highlight_positions:
            start, end = pos
            if start < range[0] or end > range[1]:
                raise ValueError(
                    f"Highlighted position ({start}, {end}) falls outside the plotting range {range}."
                )

    # Plotting predictions
    plt.figure(figsize=figsize)

    # Top plot: Model predictions
    plt.subplot(2, 1, 1)
    plt.plot(
        np.arange(range[0], range[1]),
        scores,
        marker="o",
        markersize=marker_size,
        linestyle="-",
        linewidth=line_width,
        color=line_colors[0],
        label="Prediction Score",
        rasterized=True,
    )
    if gene_start is not None and gene_end is not None:
        plt.axvspan(gene_start, gene_end, color="red", alpha=0.2, label="Gene Locus")
    if highlight_positions:
        for start, end in highlight_positions:
            plt.axvspan(start, end, color="green", alpha=0.3)
    plt.title(title)
    plt.xlabel("Genomic Position")
    plt.ylabel("Prediction Score")
    plt.ylim(bottom=0)
    plt.xticks(rotation=90)
    plt.grid(grid)
    plt.legend()
    if ylim:
        plt.ylim(ylim)

    # Bottom plot: bigWig values
    if bigwig_values is not None and bigwig_midpoints is not None:
        plt.subplot(2, 1, 2)
        plt.plot(
            bigwig_midpoints,
            bigwig_values,
            linestyle="-",
            color=line_colors[1],
            label="bigWig Values",
            rasterized="True",
        )
        if gene_start is not None and gene_end is not None:
            plt.axvspan(
                gene_start, gene_end, color="red", alpha=0.2, label="Gene Locus"
            )
        plt.xlabel("Genomic Position")
        plt.ylabel("bigWig Values")
        plt.xticks(rotation=90)
        plt.ylim(bottom=0)
        plt.grid(grid)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
