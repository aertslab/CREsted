"""Distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from loguru import logger

def locus_scoring(scores, coordinates, range, gene_start=None, gene_end=None, title='Predictions across Genomic Regions', bigwig_values=None, bigwig_midpoints=None, filename=None):
    """
    Plot the predictions as a line chart over the entire genomic input and optionally indicate the gene locus.
    Additionally, plot values from a bigWig file if provided.

    Parameters:
    scores (np.array): An array of prediction scores for each window.
    coordinates (np.array): An array of tuples, each containing the chromosome name and the start and end positions of the sequence for each window.
    model_class (int): The class index to plot from the prediction scores.
    gene_start (int, optional): The start position of the gene locus to highlight on the plot.
    gene_end (int, optional): The end position of the gene locus to highlight on the plot.
    title (str): The title of the plot.
    bigwig_values (np.array, optional): A numpy array of values extracted from a bigWig file for the same coordinates.
    bigwig_midpoints (list, optional): A list of base pair positions corresponding to the bigwig_values.
    """

    # Extract the midpoints of the coordinates for plotting
    midpoints = [(int(start) + int(end)) // 2 for _, start, end in coordinates]

    # Plotting predictions
    plt.figure(figsize=(30, 10))

    # Top plot: Model predictions
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(range[0], range[1]), scores, marker='o', linestyle='-', color='b', label='Prediction Score')
    if gene_start is not None and gene_end is not None:
        plt.axvspan(gene_start, gene_end, color='red', alpha=0.3, label='Gene Locus')
    plt.title(title)
    plt.xlabel('Genomic Position')
    plt.ylabel('Prediction Score')
    plt.ylim(bottom=0)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend()

    # Bottom plot: bigWig values
    if bigwig_values is not None and bigwig_midpoints is not None:
        plt.subplot(2, 1, 2)
        plt.plot(bigwig_midpoints, bigwig_values, linestyle='-', color='g', label='bigWig Values')
        if gene_start is not None and gene_end is not None:
            plt.axvspan(gene_start, gene_end, color='red', alpha=0.3, label='Gene Locus')
        plt.xlabel('Genomic Position')
        plt.ylabel('bigWig Values')
        plt.xticks(rotation=90)
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
