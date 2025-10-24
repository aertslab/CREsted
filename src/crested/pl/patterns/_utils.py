"""Sequence pattern utility functions for plotting."""

from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def grad_times_input_to_df(x, grad, alphabet="ACGT"):
    """Generate pandas dataframe for saliency plot based on grad x inputs."""
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
    """Generate pandas dataframe for mutagenesis plot based on grad x inputs."""
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


def grad_times_input_to_df_mutagenesis_letters(x, grad, alphabet="ACGT"):
    """Generate pandas dataframe for mutagenesis plot based on grad x inputs."""
    x = np.squeeze(x)  # Ensure x is correctly squeezed
    grad = np.squeeze(grad)
    L, A = x.shape

    # Get original nucleotides' indices, ensure it's 1D
    x_index = np.argmax(x, axis=1)

    all_locs = np.array([0, 1, 2, 3])
    seq = ""
    saliency = np.zeros(L)
    for i in range(L):
        seq += alphabet[x_index[i]]
        saliency[i] = -np.mean(grad[i, np.delete(all_locs, x_index[i])])

    # create saliency matrix
    saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
    return saliency_df


def _plot_attribution_map(
    saliency_df,
    ax=None,
    return_ax: bool = True,
    spines: bool = True,
    figsize: tuple[int, int] = (20, 1),
    rotate: bool = False,
):
    """
    Plot an attribution map (PWM logo) and optionally rotate it by 90 degrees.

    Parameters
    ----------
        saliency_df (pd.DataFrame or np.ndarray): A DataFrame or array with attribution scores,
            where columns are nucleotide bases (A, C, G, T).
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Default is None,
            which creates a new Axes.
        return_ax (bool, optional): Whether to return the Axes object. Default is True.
        spines (bool, optional): Whether to display spines (axes borders). Default is True.
        figsize (tuple[int, int], optional): Figure size for temporary rendering. Default is (20, 1).
        rotate (bool, optional): Whether to rotate the resulting plot by 90 degrees. Default is False.

    Returns
    -------
        matplotlib.axes.Axes: The Axes object with the plotted attribution map, if `return_ax` is True.
    """
    # Convert input to DataFrame if needed
    if not isinstance(saliency_df, pd.DataFrame):
        saliency_df = pd.DataFrame(saliency_df, columns=["A", "C", "G", "T"])

    # Standard plotting (no rotation)
    if not rotate:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        logomaker.Logo(saliency_df, ax=ax)
        if not spines:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        if return_ax:
            return ax
        return

    # Rotation case: render plot to an image
    temp_fig, temp_ax = plt.subplots(figsize=figsize)
    logomaker.Logo(saliency_df, ax=temp_ax)
    temp_ax.axis("off")  # Remove axes for clean rendering

    # Render the plot as an image
    temp_fig.canvas.draw()
    renderer = temp_fig.canvas.get_renderer()
    width, height = map(int, temp_fig.get_size_inches() * temp_fig.get_dpi())
    image = np.frombuffer(renderer.buffer_rgba(), dtype="uint8").reshape(height, width, 4)[..., :3]
    #width, height = map(int, temp_fig.get_size_inches() * temp_fig.get_dpi())
    #image = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype="uint8").reshape(
    #    height, width, 3
    #)
    plt.close(temp_fig)  # Close the temporary figure to avoid memory leaks

    # Rotate the rendered image
    rotated_image = np.rot90(image)
    rotated_image_pil = Image.fromarray(rotated_image)

    # Display the rotated image on the given Axes
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.clear()
    ax.imshow(rotated_image_pil)
    ax.axis("off")  # Hide axes for a clean look

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
