"""Sequence pattern utility functions for plotting."""

from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _plot_attribution_map(
    saliency_df,
    ax=None,
    return_ax: bool = True,
    spines: bool = True,
    figsize: tuple | None = (20, 1),
):
    """Plot an attribution map using logomaker"""
    if type(saliency_df) is not pd.DataFrame:
        saliency_df = pd.DataFrame(saliency_df, columns=["A", "C", "G", "T"])
    if figsize is not None:
        logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    else:
        logomaker.Logo(saliency_df, ax=ax)
    if ax is None:
        ax = plt.gca()
    if not spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    if return_ax:
        return ax


def _plot_mutagenesis_map(mutagenesis_df, ax=None):
    """Plot an attribution map for mutagenesis using different colored dots, with adjusted x-axis limits."""
    colors = {"A": "green", "C": "blue", "G": "orange", "T": "red"}
    if ax is None:
        ax = plt.gca()
    # Add horizontal line at y=0
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")

    # Scatter plot for each nucleotide type
    for nuc, color in colors.items():
        # Filter out dots where the variant is the same as the original nucleotide
        subset = mutagenesis_df[
            (mutagenesis_df["Nucleotide"] == nuc)
            & (mutagenesis_df["Nucleotide"] != mutagenesis_df["Original"])
        ]
        ax.scatter(
            subset["Position"], subset["Effect"], color=color, label=nuc, s=10
        )  # s is the size of the dot

    # Set the limits of the x-axis to match exactly the first and last position
    if not mutagenesis_df.empty:
        ax.set_xlim(
            mutagenesis_df["Position"].min() - 0.5,
            mutagenesis_df["Position"].max() + 0.5,
        )

    ax.legend(title="Nucleotide", loc="upper right")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("none")
    plt.xticks([])  # Optionally, hide x-axis ticks for a cleaner look
