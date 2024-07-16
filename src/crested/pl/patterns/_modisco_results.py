from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import modiscolite as modisco
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from crested._logging import log_and_raise
from crested.pl._utils import render_plot

from ._utils import _plot_attribution_map


@log_and_raise(ValueError)
def _trim_pattern_by_ic(
    pattern: dict,
    pos_pattern: bool,
    min_v: float,
    background: list[float] = None,
    pseudocount: float = 1e-6,
) -> dict:
    """
    Trims the pattern based on information content (IC).

    Parameters
    ----------
    pattern
        Dictionary containing the pattern data.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    min_v
        Minimum value for trimming.
    background
        Background probabilities for each nucleotide.
    pseudocount
        Pseudocount for IC calculation.

    Returns
    -------
        Trimmed pattern.
    """
    if background is None:
        background = [0.27, 0.23, 0.23, 0.27]
    contrib_scores = np.array(pattern["contrib_scores"])
    if not pos_pattern:
        contrib_scores = -contrib_scores
    contrib_scores[contrib_scores < 0] = 1e-9  # avoid division by zero

    ic = modisco.util.compute_per_position_ic(
        ppm=np.array(contrib_scores), background=background, pseudocount=pseudocount
    )
    np.nan_to_num(ic, copy=False, nan=0.0)
    v = (abs(np.array(contrib_scores)) * ic[:, None]).max(1)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)

    try:
        start_idx = min(np.where(np.diff((v > min_v) * 1))[0])
        end_idx = max(np.where(np.diff((v > min_v) * 1))[0]) + 1
    except ValueError:
        logger.error("No valid pattern found. Aborting...")

    return _trim(pattern, start_idx, end_idx)


def _trim(pattern: dict, start_idx: int, end_idx: int) -> dict:
    """
    Trims the pattern to the specified start and end indices.

    Parameters
    ----------
    pattern
        Dictionary containing the pattern data.
    start_idx
        Start index for trimming.
    end_idx (int)
        End index for trimming.

    Returns
    -------
        Trimmed pattern.
    """
    return {
        "sequence": np.array(pattern["sequence"])[start_idx:end_idx],
        "contrib_scores": np.array(pattern["contrib_scores"])[start_idx:end_idx],
        "hypothetical_contribs": np.array(pattern["hypothetical_contribs"])[
            start_idx:end_idx
        ],
    }


def _get_ic(
    contrib_scores: np.ndarray,
    pos_pattern: bool,
    background: list[float] = None,
) -> np.ndarray:
    """
    Computes the information content (IC) for the given contribution scores.

    Parameters
    ----------
    contrib_scores
        Array of contribution scores.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    background
        background probabilities for each nucleotide.

    Returns
    -------
        Information content for the contribution scores.
    """
    if background is None:
        background = [0.27, 0.23, 0.23, 0.27]
    background = np.array(background)
    if not pos_pattern:
        contrib_scores = -contrib_scores
    contrib_scores[contrib_scores < 0] = 1e-9
    ppm = contrib_scores / np.sum(contrib_scores, axis=1)[:, None]

    ic = (np.log((ppm + 0.001) / (1.004)) / np.log(2)) * ppm - (
        np.log(background) * background / np.log(2)
    )
    return ppm * (np.sum(ic, axis=1)[:, None])


def modisco_results(
    classes: list[str],
    contribution: str,
    contribution_dir: str,
    num_seq: int,
    viz: str = "contrib",
    min_seqlets: int = 0,
    verbose: bool = False,
    y_min: float = -0.05,
    y_max: float = 0.25,
    background: list[float] = None,
    **kwargs,
) -> None:
    """
    Plot genomic contributions for the given classes.

    Requires the modisco results to be present in the specified directory.
    The contribution scores are trimmed based on information content (IC).

    Parameters
    ----------
    classes
        List of classes to plot genomic contributions for.
    contribution
        Contribution type to plot. Choose either "positive" or "negative".
    contribution_dir
        Directory containing the modisco results.
        Each class should have a separate modisco .h5 results file in the format {class}_modisco_results.h5.
    num_seq
        Total number of sequences used for the modisco run.
        Necessary to calculate the percentage of sequences with the pattern.
    viz
        Visualization method. Choose either "contrib" or "pwm".
    min_seqlets
        Minimum number of seqlets required for a pattern to be considered.
    verbose
        Print verbose output.
    y_min
        Minimum y-axis limit for the plot if viz is "contrib".
    y_max
        Maximum y-axis limit for the plot if viz is "contrib".
    background
        Background probabilities for each nucleotide. Default is [0.27, 0.23, 0.23, 0.27].
    kwargs
        Additional keyword arguments for the plot.

    Examples
    --------
    >>> crested.pl.patterns.modisco_results(
    ...     classes=["Lamp5", "Pvalb", "Sst", ""Sst-Chodl", "Vip"],
    ...     contribution="positive",
    ...     contribution_dir="/path/to/modisco_results",
    ...     num_seq=1000,
    ...     viz="pwm",
    ...     save_path="/path/to/genomic_contributions.png",
    ... )

    .. image:: ../../../../docs/_static/img/examples/genomic_contributions.png

    See Also
    --------
    crested.tl.tfmodisco
    crested.pl.render_plot
    """
    if background is None:
        background = [0.27, 0.23, 0.23, 0.27]
    background = np.array(background)
    pos_pat = contribution == "positive"

    logger.info(f"Starting genomic contributions plot for classes: {classes}")

    max_num_patterns = 0
    all_patterns = []

    for cell_type in classes:
        if verbose:
            logger.info(cell_type)
        hdf5_results = h5py.File(
            f"{contribution_dir}/{cell_type}_modisco_results.h5", "r"
        )
        metacluster_names = list(hdf5_results.keys())

        if f"{contribution[:3]}_patterns" not in metacluster_names:
            raise ValueError(
                f"No {contribution[:3]}_patterns for {cell_type}. Aborting..."
            )

        for metacluster_name in [f"{contribution[:3]}_patterns"]:
            all_pattern_names = list(hdf5_results[metacluster_name])
            max_num_patterns = max(max_num_patterns, len(all_pattern_names))

    fig, axes = plt.subplots(
        nrows=max_num_patterns,
        ncols=len(classes),
        figsize=(8 * len(classes), 2 * max_num_patterns),
    )

    if verbose:
        logger.info(f"Max patterns for selected classes: {max_num_patterns}")

    motif_counter = 1

    for idx, cell_type in enumerate(classes):
        hdf5_results = h5py.File(
            f"{contribution_dir}/{cell_type}_modisco_results.h5", "r"
        )
        metacluster_names = list(hdf5_results.keys())

        if verbose:
            logger.info(metacluster_names)

        for metacluster_name in [f"{contribution[:3]}_patterns"]:
            all_pattern_names = list(hdf5_results[metacluster_name])

            for _pattern_idx, pattern_name in enumerate(all_pattern_names):
                if len(classes) > 1:
                    ax = axes[motif_counter - 1, idx]
                elif max_num_patterns > 1:
                    ax = axes[motif_counter - 1]
                else:
                    ax = axes
                motif_counter += 1
                all_patterns.append((metacluster_name, pattern_name))
                pattern = hdf5_results[metacluster_name][pattern_name]
                num_seqlets = list(
                    hdf5_results[metacluster_name][pattern_name]["seqlets"]["n_seqlets"]
                )[0]
                if verbose:
                    logger.info(metacluster_name, pattern_name)
                    logger.info("total seqlets:", num_seqlets)
                if num_seqlets < min_seqlets:
                    break
                pattern_trimmed = _trim_pattern_by_ic(pattern, pos_pat, 0.1)
                if viz == "contrib":
                    ax = _plot_attribution_map(
                        ax=ax,
                        saliency_df=np.array(pattern_trimmed["contrib_scores"]),
                        return_ax=True,
                        figsize=None,
                    )
                    ax.set_ylim([y_min, y_max])
                    ax.set_title(
                        f"{cell_type}: {np.around(num_seqlets / num_seq * 100, 2)}% seqlet frequency"
                    )
                elif viz == "pwm":
                    pwm = _get_ic(np.array(pattern_trimmed["contrib_scores"]), pos_pat)
                    ax = _plot_attribution_map(
                        ax=ax, saliency_df=pwm, return_ax=True, figsize=None
                    )
                    ax.set_title(
                        f"{cell_type}: {np.around(num_seqlets / num_seq * 100, 2)}% seqlet frequency - Average IC: {np.around(np.mean(pwm), 2)}"
                    )
                    ax.set_ylim([0, 2])
                else:
                    raise ValueError(
                        'Invalid visualization method. Choose either "contrib" or "pwm". Aborting...'
                    )
        motif_counter = 1

    plt.tight_layout()
    if "width" not in kwargs:
        kwargs["width"] = 6 * len(classes)
    if "height" not in kwargs:
        kwargs["height"] = 2 * max_num_patterns

    render_plot(fig, **kwargs)


def plot_custom_xticklabels(
    ax: plt.Axes,
    sequences: list[tuple[str, np.ndarray]],
    col_order: list[int],
    fontsize: int = 10,
    dy: float = 0.012,
) -> None:
    """
    Plot custom x-tick labels with varying letter heights.

    Parameters
    ----------
    ax
        The axes object to plot on.
    sequences
        List of tuples containing sequences and their corresponding heights.
    col_order
        List of column indices after clustering.
    fontsize
        Base font size for the letters.
    dy
        Vertical adjustment factor for letter heights.
    """
    ax.set_xticks(np.arange(len(sequences)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", length=0)

    for i, original_index in enumerate(col_order):
        sequence, heights = sequences[original_index]
        y_position = -0.02
        for _, (char, height) in enumerate(zip(sequence, heights)):
            char_fontsize = height * fontsize
            text = ax.text(
                i,
                y_position,
                char,
                ha="center",
                va="center",
                color="black",
                transform=ax.get_xaxis_transform(),
                fontsize=char_fontsize,
                rotation=270,
            )
            renderer = ax.figure.canvas.get_renderer()
            _ = text.get_window_extent(renderer=renderer).width
            y_position -= dy


def create_clustermap(
    pattern_matrix: np.ndarray,
    classes: list[str],
    figsize: tuple[int, int] = (15, 13),
    grid: bool = False,
    color_palette: str | list[str] = "hsv",
    cmap: str = "coolwarm",
    center: float = 0,
    method: str = "average",
    fig_path: str | None = None,
    pat_seqs: list[tuple[str, np.ndarray]] | None = None,
    dy: float = 0.012,
) -> sns.matrix.ClusterGrid:
    """
    Create a clustermap from the given pattern matrix and class labels with customizable options.

    Parameters
    ----------
    pattern_matrix
        2D NumPy array containing pattern data.
    classes
        List of class labels.
    figsize
        Size of the figure.
    grid
        Whether to add a grid to the heatmap.
    color_palette
        Color palette for the row colors.
    - cmap (str): Colormap for the clustermap.
    - center (float): Value at which to center the colormap.
    - method (str): Clustering method to use (e.g., 'average', 'single', 'complete').
    - fig_path (str, optional): Path to save the figure.
    - pat_seqs (list, optional): List of sequences to use as xticklabels.
    - dy (float): Vertical adjustment factor for letter heights.

    Returns
    -------
    The clustermap object.
    """
    data = pd.DataFrame(pattern_matrix)

    if isinstance(color_palette, str):
        palette = sns.color_palette(color_palette, len(set(classes)))
    else:
        palette = color_palette

    class_lut = dict(zip(set(classes), palette))
    row_colors = pd.Series(classes).map(class_lut)

    xtick_labels = False if pat_seqs is not None else True

    g = sns.clustermap(
        data,
        cmap=cmap,
        figsize=figsize,
        row_colors=row_colors,
        yticklabels=classes,
        center=center,
        xticklabels=xtick_labels,
        method=method,
    )
    col_order = g.dendrogram_col.reordered_ind

    for label in class_lut:
        g.ax_col_dendrogram.bar(0, 0, color=class_lut[label], label=label, linewidth=0)

    if grid:
        ax = g.ax_heatmap
        ax.grid(
            True,
            which="both",
            color="grey",
            linewidth=0.25,
        )
        g.fig.canvas.draw()

    if pat_seqs is not None:
        plot_custom_xticklabels(g.ax_heatmap, pat_seqs, col_order, dy=dy)

    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()
    return g


def plot_patterns(pattern_dict: dict, idcs: list[int]) -> None:
    """
    Plots the patterns specified by the indices in `idcs` from the `pattern_dict`.

    Parameters
    ----------
    pattern_dict
        A dictionary containing pattern data.
    idcs
        A list of indices specifying which patterns to plot.
    """
    figure, axes = plt.subplots(nrows=len(idcs), ncols=1, figsize=(8, 2 * len(idcs)))
    if len(idcs) == 1:
        axes = [axes]

    for i, idx in enumerate(idcs):
        ax = _plot_attribution_map(
            ax=axes[i],
            saliency_df=np.array(pattern_dict[str(idx)]["pattern"]["contrib_scores"]),
            return_ax=True,
            figsize=None,
        )
        ax.set_title(pattern_dict[str(idx)]["pattern"]["id"])

    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    indices: list,
    fig_size: tuple[int, int] = (30, 15),
    fig_path: str | None = None,
) -> None:
    """
    Plots a similarity heatmap of all pattern indices.

    Parameters
    ----------
    similarity_matrix
        A 2D numpy array containing the similarity values.
    indices
        List of pattern indices.
    fig_size
        Size of the figure for the heatmap.
    fig_path
        Path to save the figure. If None, the figure will be shown but not saved.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    heatmap = sns.heatmap(
        similarity_matrix,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        xticklabels=indices,
        yticklabels=indices,
        annot_kws={"size": 8},
    )

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")
        spine.set_linewidth(0.5)

    plt.title("Pattern Similarity Heatmap", fontsize=20)
    plt.xlabel("Pattern Index", fontsize=15)
    plt.ylabel("Pattern Index", fontsize=15)

    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()
