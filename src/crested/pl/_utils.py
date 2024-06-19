"""Utility functions for plotting in CREsted."""

from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np


def render_plot(
    fig,
    width: int = 8,
    height: int = 8,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Render a plot with customization options.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    fig
        The figure object to render.
    width
        Width of the plot (inches).
    height
        Height of the plot (inches).
    title
        Title of the plot.
    xlabel
        Label for the X-axis.
    ylabel
        Label for the Y-axis.
    fig_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    """
    fig.set_size_inches(width, height)
    if title:
        fig.suptitle(title)
    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def grad_times_input_to_df(x, grad, alphabet="ACGT"):
    """Generate pandas dataframe for saliency plot based on grad x inputs"""
    x_index = np.argmax(np.squeeze(x), axis=1)
    grad = np.squeeze(grad)
    L, A = grad.shape

    seq = ""
    saliency = np.zeros(L)
    for i in range(L):
        seq += alphabet[x_index[i]]
        saliency[i] = grad[i, x_index[i]]

    # create saliency matrix
    saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
    return saliency_df


def grad_times_input_to_df_mutagenesis(x, grad, alphabet="ACGT"):
    import pandas as pd

    """Generate pandas dataframe for mutagenesis plot based on grad x inputs"""
    x = np.squeeze(x)  # Ensure x is correctly squeezed
    grad = np.squeeze(grad)
    L, A = x.shape

    # Get original nucleotides' indices, ensure it's 1D
    x_index = np.argmax(x, axis=1)

    # Convert index array to nucleotide letters
    original_nucleotides = np.array([alphabet[idx] for idx in x_index])

    # Data preparation for DataFrame
    data = {
        "Position": np.repeat(np.arange(L), A),
        "Nucleotide": np.tile(list(alphabet), L),
        "Effect": grad.reshape(
            -1
        ),  # Flatten grad assuming it matches the reshaped size
        "Original": np.repeat(original_nucleotides, A),
    }
    df = pd.DataFrame(data)
    return df
