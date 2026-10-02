"""Sequence pattern utility functions for plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from fast_logomaker import FastLogo
from matplotlib.transforms import Affine2D


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
    scores = scores[..., None] * seq[None, ...]
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
    return scores * seq


def _plot_attribution_map(
    data: FastLogo | np.ndarray,
    ax: plt.Axis,
    idx: int = 0,
    spines: bool = True,
    reversed_positions: bool = False,
    rotate: bool = False,
    **kwargs,
):
    """Draw an attribution map from a processed FastLogo or a matrix onto an ax, optionally rotated by 90 degrees.

    Parameters
    ----------
    data
        A `fast_logomaker.FastLogo` object (with `logo.process_all()` called) (preferred), or an array with attribution scores of shape (length, 4) or (1, length, 4).
    idx
        Index of the logo to draw within `logo`'s batch. Default is 0.
    ax
        Axes object to plot on.
    spines
        Whether to keep the top and right spines (axes borders). Default is True.
        Disregarded if `rotate=True`, since all spines are turned off with `ax.axis("off")` there.
    reversed_positions
        Whether `logo`'s positions run in descending order; if so, inverts the x-axis (non-rotated case only).
    rotate
        Whether to rotate the resulting plot by 90 degrees. Default is False.
    kwargs
        Extra arguments passed to logo.draw_single.

    Returns
    -------
    matplotlib.axes.Axes: The Axes object with the plotted logo, if `return_ax` is True.
    """
    # Check inputs
    if not isinstance(data, FastLogo):
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        logo = FastLogo(values=data, show_progress=False)
        logo.process_all()
    else:
        logo = data

    # Create plot
    logo.draw_single(idx, ax=ax, border=not rotate, baseline=not rotate, fixed_ylim=False, apply_layout=False, **kwargs)

    if not rotate:
        # Standard plotting (no rotation)
        if reversed_positions:
            ax.xaxis.set_inverted(True)
        if not spines:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
    else:
        # Rotation case: rotate the glyphs in place via a data-space transform
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Rotate the glyph patch collection(s) FastLogo just drew
        rotation = Affine2D().rotate_deg(90) + ax.transData
        for collection in ax.collections:
            collection.set_transform(rotation)

        # FastLogo's baseline does not rotate nicely, so we manually draw it instead
        baseline_width = logo.kwargs.get("baseline_width", 0.5)
        if baseline_width > 0:
            ax.axvline(x=0, color="black", linewidth=baseline_width, zorder=-1)

        ax.set_xlim(-ymax, -ymin)
        ax.set_ylim(xmin, xmax)
        ax.axis("off")  # Hide axes for a clean look


def _plot_mutagenesis_map(
    scores: np.ndarray,
    ax: plt.Axes,
    start: int | None = None,
    end: int | None = None,
    colors: dict | None = None,
    s: int = 10,
    spines: bool = False,
    **kwargs,
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
