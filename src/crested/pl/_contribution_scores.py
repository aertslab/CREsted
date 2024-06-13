"""Plot contribution scores."""

from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested._logging import log_and_raise
from crested.pl._utils import grad_times_input_to_df


def _plot_attribution_map(saliency_df, ax=None, figsize=(20, 1)):
    """Plot an attribution map using logomaker"""
    logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    plt.xticks([])


@log_and_raise(ValueError)
def _check_contrib_params(
    zoom_n_bases: int | None,
    scores: np.ndarray,
):
    """Check contribution scores parameters."""
    if zoom_n_bases is not None and zoom_n_bases > scores.shape[2]:
        raise ValueError(
            f"zoom_n_bases ({zoom_n_bases}) must be less than or equal to the number of bases in the sequence ({scores.shape[2]})"
        )


def contribution_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    class_names: list,
    zoom_n_bases: int | None = None,
    highlight_positions: list[tuple[int, int]] | None = None,
    ylim: tuple | None = None,
    save_path: str | None = None,
):
    """Visualize interpretation scores with optional highlighted positions."""
    # Center and zoom
    _check_contrib_params(zoom_n_bases, scores)

    if zoom_n_bases is None:
        zoom_n_bases = scores.shape[2]
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)
    scores = scores[:, :, start_idx : start_idx + zoom_n_bases, :]

    global_min = scores.min()
    global_max = scores.max()

    # Plot
    logger.info(f"Plotting contribution scores for {seqs_one_hot.shape[0]} sequence(s)")
    for seq in range(seqs_one_hot.shape[0]):
        fig_height_per_class = 2
        fig = plt.figure(figsize=(50, fig_height_per_class * len(class_names)))
        for i, class_name in enumerate(class_names):
            seq_class_scores = scores[seq, i, :, :]
            seq_class_x = seqs_one_hot[seq, :, :]
            intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
            ax = plt.subplot(len(class_names), 1, i + 1)
            _plot_attribution_map(intgrad_df, ax=ax)
            text_to_add = class_name
            if ylim:
                ax.set_ylim(ylim[0], ylim[1])
                x_pos = 5
                y_pos = 0.75 * ylim[1]
            else:
                ax.set_ylim([global_min, global_max])
                x_pos = 5
                y_pos = 0.75 * global_max
            ax.text(x_pos, y_pos, text_to_add, fontsize=16, ha="left", va="center")

            # Draw rectangles to highlight positions
            if highlight_positions:
                for start, end in highlight_positions:
                    ax.add_patch(
                        plt.Rectangle(
                            (
                                start - start_idx - 0.5,
                                global_min,
                            ),
                            end - start,
                            global_max - global_min,
                            edgecolor="red",
                            facecolor="none",
                            linewidth=0.5,
                        )
                    )

        plt.xlabel("Position")
        plt.xticks(np.arange(0, zoom_n_bases, 50))
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
