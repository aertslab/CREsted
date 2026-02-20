"""Sequence pattern utility functions for plotting."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image


def _process_mutagenesis(seq: np.ndarray, scores: np.ndarray):
    """Process a mutagenesis scoring matrix for plotting by masking reference values.

    Parameters
    ----------
    seq
        A [n_bp, n_nuc] one-hot encoded array, of the specific sequence.
    scores
        A [n_classes, n_bp, n_nuc] or [n_bp, n_nuc] array, of scores per nucleotide for each location.

    Returns
    -------
    An array of the same shape as `scores`, but with `np.nan` at the non-alternative values.
    """
    # Where seq is True/1, set np.nan, otherwise grab value from scores
    return np.where(seq, np.nan, scores)

def _process_mutagenesis_letters(seq: np.ndarray, scores: np.ndarray):
    """Process a mutagenesis scoring matrix for plotting as letters by taking the average effect and inverting the sign.

    Parameters
    ----------
    seq
        A [n_bp, n_nuc] one-hot encoded array, of the specific sequence.
    scores
        A [n_classes, n_bp, n_nuc] or [n_bp, n_nuc] array, of scores per nucleotide for each location.

    Returns
    -------
    An array of the same shape as `scores`, but with the average drop in score over the three non-reference nucleotides at the reference and 0 elsewhere.
    """
    # Multiply reference values (seq == 1) with 0
    scores = scores * np.logical_not(seq)
    # Take the mean of the other nucleotides, negate
    scores = -scores.sum(axis=-1) / 3
    # Spread back out over nucleotide axis
    scores = scores[..., None] * seq[None, ...] # TODO: check whether this works
    return scores

def _process_gradients(seq: np.ndarray, scores: np.ndarray):
    """Process a gradient scoring matrix for plotting by selecting the values for the sequence in `seq`.

    Parameters
    ----------
    seq
        A [n_bp, n_nuc] one-hot encoded array, of the specific sequence.
    scores
        A [n_classes, n_bp, n_nuc] or [n_bp, n_nuc] array, of scores per nucleotide for each location.

    Returns
    -------
    An array the same shape as `scores`, with non-reference basepairs zero'd out.
    """
    return scores*seq


def _make_logomaker_df(
    scores: np.ndarray,
    start: int | None = None,
    end: int | None = None,
    alphabet: Sequence = ('A', 'C', 'G', 'T')
    ):
    """Turn an array into a LogoMaker-ready dataframe.

    Parameters
    ----------
    scores
        A [n_bp, n_nuc] array, of scores per nucleotide for each location, like from `_process_gradients()` or `_process_mutagenesis_letters()`.
    start
        The x-coordinate of the start of the sequence. If None, set to 0. Can be bigger than `end` to plot the values in reverse order.
    end
        The x-coordinate of the end of the sequence. If None, set to `start` + `n_bp`. Can be smaller than `start` to plot the values in reverse order.
        Must be `n_bp` bigger or smaller than `start`.
    alphabet
        The order of the nucleotides.

    Returns
    -------
    A DataFrame the same shape as `scores`, with scores as values, x-axis integers as index, and nucleotide letters as columns.
    """
    if start is None:
        start = 0
    if end is None:
        end = start + scores.shape[-2]
    step = -1 if start > end else 1
    x_values = np.arange(start, end, step)
    # Goal: a [n_bp, n_nuc] dataframe, with integer positions as row indices and DNA letters as columns.
    return pd.DataFrame(scores, index=x_values, columns=alphabet)


def _plot_attribution_map(
    saliency_df: pd.DataFrame | np.ndarray,
    ax: plt.Axis | None = None,
    start: int | None = None,
    end: int | None = None,
    return_ax: bool = True,
    spines: bool = True,
    figsize: tuple[int, int] = (20, 1),
    rotate: bool = False,
    **kwargs
):
    """
    Plot an attribution map (PWM logo) and optionally rotate it by 90 degrees.

    Parameters
    ----------
    saliency_df
        A DataFrame or array with attribution scores where columns are nucleotide bases (A, C, G, T).
    ax
        Axes object to plot on. Default is None which creates a new Axes.
    start
        The start of the sequence x-axis. If not supplied, set to 0. Ignored if `saliency_df` is already a dataframe.
    end
        The end of the sequence x-axis. If not supplied, set to start + the length of the sequence. Ignored if `saliency_df` is already a dataframe.
    return_ax
        Whether to return the Axes object. Default is True.
    spines
        Whether to display spines (axes borders). Default is True.
    figsize
        Figure size for temporary rendering. Default is (20, 1).
    rotate
        Whether to rotate the resulting plot by 90 degrees. Default is False.
    kwargs
        Arguments passed to `logomaker.Logo()`.

    Returns
    -------
    matplotlib.axes.Axes: The Axes object with the plotted attribution map, if `return_ax` is True.
    """
    # Convert input to DataFrame if needed
    if not isinstance(saliency_df, pd.DataFrame):
        saliency_df = _make_logomaker_df(saliency_df, start=start, end=end)
    else:
        if start is not None or end is not None:
            logger.warning("Setting `start` and/or `end` with a pre-made DataFrame. Using DataFrame info and ignoring `start`/`end`...")

    # Check matrix validity
    logomaker.validate_matrix(saliency_df)

    # Standard plotting (no rotation)
    if not rotate:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        logomaker.Logo(saliency_df, ax=ax, **kwargs)
        if saliency_df.index[0] > saliency_df.index[-1]:
            ax.xaxis.set_inverted(True)
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
    image = np.frombuffer(renderer.buffer_rgba(), dtype="uint8").reshape(
        height, width, 4
    )[..., :3]
    # width, height = map(int, temp_fig.get_size_inches() * temp_fig.get_dpi())
    # image = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype="uint8").reshape(
    #    height, width, 3
    # )
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


def _plot_mutagenesis_map(
        scores: np.ndarray,
        ax: plt.Axes,
        start: int | None = None,
        end: int | None = None,
        colors: dict | None = None,
        s: int = 10,
        spines: bool = False,
        **kwargs
    ):
    """
    Plot a mutagenesis map with one point for every nucleotide.

    Parameters
    ----------
    scores
        A [seq, nuc] matrix, with the reference nucleotide score masked as `np.nan`.
    ax
        Axes object to plot on.
    start
        The start of the sequence x-axis. If not supplied, set to 0.
    end
        The end of the sequence x-axis. If not supplied, set to start + the length of the sequence.
    colors
        A dictionary of nucleotide labels and colors, matching the order of the score `nuc` dimension.
        Default is None, which uses `{"A": "green", "C": "blue", "G": "orange", "T": "red"}`.
    s
        The size of the scatter points. Default is 10.
    spines
        Whether to display spines (axes borders). Default is True.
    figsize
        Figure size for temporary rendering. Default is (20, 1).
    kwargs
        Arguments passed to :meth:`~matplotlib.axes.Axes.scatter`.
    """
    # Set default colors if not supplied
    if colors is None:
        colors = {"A": "green", "C": "blue", "G": "orange", "T": "red"}

    # Create x axis values
    if start is None:
        start = 0
    if end is None:
        end = start + scores.shape[-2]
    step = -1 if start > end else 1
    x_positions = np.arange(start, end, step)

    # Plot all entries of each nucleotide - assumes reference/wt nucleotides are already set to None
    for i, (nuc, color) in enumerate(colors.items()):
        ax.scatter(x_positions, scores[:, i], color=color, label=nuc, s=s, **kwargs)
    ax.legend(title="Nucleotide", loc="upper right")
    if start > end:
        ax.xaxis.set_inverted(True)

    # Add horizontal line at y=0
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    # Prettify plot
    if not spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    ax.margins(x=0)

def grad_times_input_to_df(x, grad, alphabet="ACGT"):
    """Generate pandas dataframe for saliency plot based on grad x inputs.

    Deprecated, please use `_process_gradients` instead.
    """
    warnings.warn(
        "'grad_times_input_to_df' is deprecated since version 2.0.0 and will be removed in a future release. Please use `_process_gradients` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """Generate pandas dataframe for mutagenesis plot based on grad x inputs.

    Deprecated, please use `_process_mutagenesis` instead.
    """
    warnings.warn(
        "'grad_times_input_to_df_mutagenesis' is deprecated since version 2.0.0 and will be removed in a future release. Please use `_process_mutagenesis` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """Generate pandas dataframe for mutagenesis plot based on grad x inputs.

    Deprecated, please use `_process_mutagenesis_letters` instead.
    """
    warnings.warn(
        "'grad_times_input_to_df_mutagenesis_letters' is deprecated since version 2.0.0 and will be removed in a future release. Please use `_process_mutagenesis_letters` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
